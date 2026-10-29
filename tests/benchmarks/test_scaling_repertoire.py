"""Benchmarks of the import of immune receptor contigs and their analysis.

An import encodes the parsed contigs chunk by chunk, holding each distinct
gene call and junction once, and then ranks each cell's chains and labels its
receptor. Clonotypes group the cells by their chains, and clonal expansion
counts each clonotype's cells within samples. Each benchmark runs one step on
a simulated T cell repertoire in which most cells carry one alpha and one beta
chain, some a second alpha chain, and clones of many sizes share their
junctions, and checks the result, so it doubles as a test at the smoke size.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

LADDER = Ladder(sizes=(20_000, 40_000, 80_000, 160_000), smoke=500)
_CHUNK_ROWS = 50_000
_AMINO_ACIDS = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))


def _junctions(rng: np.random.Generator, count: int) -> np.ndarray:
    """Draw junction amino-acid sequences of 10 to 18 residues."""
    lengths = rng.integers(10, 19, size=count)
    letters = _AMINO_ACIDS[rng.integers(0, 20, size=int(lengths.sum()))]
    ends = np.cumsum(lengths)
    joined = "".join(letters.tolist())
    starts = ends - lengths
    return np.asarray(
        [
            "C" + joined[start:end] + "F"
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=object,
    )


@lru_cache(maxsize=1)
def _contigs(n_cells: int) -> pd.DataFrame:
    """Simulate the parsed contigs of ``n_cells`` T cells.

    Clone sizes follow a heavy-tailed draw, so a few clones hold many cells
    and most cells hold a clone of their own. Each cell has an alpha chain
    with probability 0.85, a beta chain with probability 0.9, and a second
    alpha chain with probability 0.1.
    """
    rng = np.random.default_rng([inputs.SEED, n_cells])
    clones = np.minimum(rng.zipf(2.2, size=n_cells), 500)
    clone_of_cell = np.repeat(np.arange(n_cells), clones)[:n_cells]
    n_clones = int(clone_of_cell.max()) + 1
    alpha = _junctions(rng, n_clones)
    beta = _junctions(rng, n_clones)
    second = _junctions(rng, n_clones)
    rows: dict[str, list] = {
        "cell": [],
        "locus": [],
        "junction_aa": [],
        "umi_count": [],
    }
    for locus, pool, probability in (
        ("TRA", alpha, 0.85),
        ("TRB", beta, 0.9),
        ("TRA", second, 0.1),
    ):
        cells = np.flatnonzero(rng.random(n_cells) < probability)
        rows["cell"].append(cells)
        rows["locus"].append(np.full(len(cells), locus, dtype=object))
        rows["junction_aa"].append(pool[clone_of_cell[cells]])
        rows["umi_count"].append(rng.integers(1, 40, size=len(cells)))
    frame = pd.DataFrame(
        {
            "cell": np.concatenate(rows["cell"]),
            "locus": pd.Series(np.concatenate(rows["locus"]), dtype=object),
            "junction_aa": pd.Series(np.concatenate(rows["junction_aa"]), dtype=object),
            "umi_count": np.concatenate(rows["umi_count"]).astype(np.int64),
        }
    )
    frame["productive"] = np.ones(len(frame), dtype=np.int8)
    frame["sequence_id"] = pd.Series(
        [f"c{index}" for index in range(len(frame))], dtype=object
    )
    return frame


def _encode(frame: pd.DataFrame):
    from cytearc.repertoire.contigs import ContigEncoder

    encoder = ContigEncoder()
    for start in range(0, len(frame), _CHUNK_ROWS):
        chunk = frame.iloc[start : start + _CHUNK_ROWS]
        encoder.add(chunk.drop(columns="cell"), chunk["cell"].to_numpy(), source=0)
    return encoder.finish()


@lru_cache(maxsize=1)
def _table(n_cells: int):
    return _encode(_contigs(n_cells))


def test_encode_contigs(bench) -> None:
    def make(n_cells: int):
        frame = _contigs(n_cells)
        return lambda: _encode(frame)

    def check(n_cells: int, table) -> None:
        frame = _contigs(n_cells)
        order = np.lexsort((np.arange(len(frame)), frame["cell"].to_numpy()))
        np.testing.assert_array_equal(table.cells, frame["cell"].to_numpy()[order])
        assert (
            table.column("junction_aa").tolist()
            == frame["junction_aa"].to_numpy()[order].tolist()
        )
        categories = table.text["junction_aa"].categories
        assert (categories[:-1] < categories[1:]).all()

    bench("repertoire.encode_contigs", make, LADDER, check=check)


def test_index_chains(bench) -> None:
    from cytearc.repertoire.chains import index_chains, receptor_quality

    def make(n_cells: int):
        table = _table(n_cells)
        locus = table.text["locus"]

        def run():
            index = index_chains(table, n_cells, locus=locus)
            return index, receptor_quality(index, locus)

        return run

    def check(n_cells: int, result) -> None:
        index, quality = result
        table = _table(n_cells)
        loci = table.column("locus")
        counts = pd.crosstab(table.cells, loci).reindex(range(n_cells), fill_value=0)
        np.testing.assert_array_equal(index.n_vj, counts.get("TRA", 0))
        np.testing.assert_array_equal(index.n_vdj, counts.get("TRB", 0))
        umis = table.values["umi_count"]
        both = (index.slots[:, 0] >= 0) & (index.slots[:, 1] >= 0)
        assert (umis[index.slots[both, 0]] >= umis[index.slots[both, 1]]).all()
        pairs = (index.slots[:, 0] >= 0) & (index.slots[:, 2] >= 0)
        assert (quality.receptor_type[pairs] == "TCR").all()
        assert set(quality.chain_pairing[~pairs]) <= {
            "orphan VJ",
            "orphan VDJ",
            "no IR",
        }

    bench("repertoire.index_chains", make, LADDER, check=check)


@lru_cache(maxsize=1)
def _slot_junctions(n_cells: int) -> np.ndarray:
    """Return the junction code in each cell's four chain slots."""
    from cytearc.repertoire.chains import index_chains

    table = _table(n_cells)
    index = index_chains(table, n_cells, locus=table.text["locus"])
    codes = np.asarray(table.text["junction_aa"].codes, dtype=np.int64)
    slots = np.full(index.slots.shape, -1, dtype=np.int64)
    filled = index.slots >= 0
    slots[filled] = codes[index.slots[filled]]
    return slots


def test_assign_clonotypes(bench) -> None:
    from cytearc.repertoire.clonotypes import assign_clonotypes

    def make(n_cells: int):
        sequences = _slot_junctions(n_cells)
        groups = np.where((sequences >= 0).any(axis=1), 0, -1)
        return lambda: assign_clonotypes(sequences, groups=groups)

    def check(n_cells: int, assignment) -> None:
        sequences = _slot_junctions(n_cells)
        has = (sequences >= 0).any(axis=1)
        np.testing.assert_array_equal(assignment.codes >= 0, has)
        assert assignment.sizes.sum() == has.sum()
        assert (np.diff(assignment.sizes) <= 0).all()
        # Cells with the same chains in every slot share a clonotype.
        frame = pd.DataFrame(sequences[has]).assign(code=assignment.codes[has])
        assert (frame.groupby([0, 1, 2, 3])["code"].nunique() == 1).all()

    bench("repertoire.assign_clonotypes", make, LADDER, check=check)


@lru_cache(maxsize=1)
def _clonotype_codes(n_cells: int) -> np.ndarray:
    from cytearc.repertoire.clonotypes import assign_clonotypes

    sequences = _slot_junctions(n_cells)
    groups = np.where((sequences >= 0).any(axis=1), 0, -1)
    return assign_clonotypes(sequences, groups=groups).codes


def _samples(n_cells: int) -> np.ndarray:
    """Spread the cells over eight samples."""
    return np.random.default_rng([inputs.SEED, n_cells, 8]).integers(0, 8, n_cells)


def test_clonal_expansion(bench) -> None:
    from cytearc.repertoire.expansion import (
        DEFAULT_BREAKPOINTS,
        DEFAULT_RANK_SPLITS,
        expansion_categories,
        group_clone_sizes,
        rank_windows,
    )

    def make(n_cells: int):
        codes = _clonotype_codes(n_cells)
        groups = np.where(codes >= 0, _samples(n_cells), -1)

        def run():
            sizes, frequencies = group_clone_sizes(codes, groups)
            categories, _labels = expansion_categories(
                sizes, frequencies, breakpoints=DEFAULT_BREAKPOINTS
            )
            windows = rank_windows(codes, groups, 8, DEFAULT_RANK_SPLITS)
            return sizes, categories, windows

        return run

    def check(n_cells: int, result) -> None:
        sizes, categories, windows = result
        codes = _clonotype_codes(n_cells)
        has = codes >= 0
        frame = pd.DataFrame({"sample": _samples(n_cells)[has], "code": codes[has]})
        expected = frame.groupby(["sample", "code"])["code"].transform("size")
        np.testing.assert_array_equal(sizes[has], expected.to_numpy())
        assert (sizes[~has] == 0).all() and (categories[~has] == -1).all()
        assert (categories[has] >= 0).all()
        np.testing.assert_array_equal(
            windows.sum(axis=1), np.bincount(frame["sample"], minlength=8)
        )

    bench("repertoire.clonal_expansion", make, LADDER, check=check)


def test_rarefaction_curve(bench) -> None:
    from cytearc.repertoire.diversity import (
        CloneSizes,
        default_sizes,
        hill_curve,
        indices,
    )

    def make(n_cells: int):
        codes = _clonotype_codes(n_cells)
        clones = CloneSizes.from_sizes(np.bincount(codes[codes >= 0]))
        sizes = default_sizes(clones.n).tolist()

        def run():
            curves = [hill_curve(clones, sizes, order) for order in (0, 1, 2)]
            return clones, indices(clones), curves

        return run

    def check(n_cells: int, result) -> None:
        clones, table, curves = result
        codes = _clonotype_codes(n_cells)
        assert clones.n == int((codes >= 0).sum())
        assert table["clonotypes"] == len(np.unique(codes[codes >= 0]))
        for values, methods in curves:
            observed = methods.index("observed")
            # Rarefied curves rise to the observed value.
            assert np.all(np.diff(values[: observed + 1]) >= -1e-9)
        richness, methods = curves[0]
        assert richness[methods.index("observed")] == clones.richness

    bench("repertoire.rarefaction_curve", make, LADDER, check=check)


def test_overlap_pairs(bench) -> None:
    from cytearc.repertoire.overlap import overlap_pairs

    n_groups = 16

    def counts(n_cells: int):
        codes = _clonotype_codes(n_cells)
        has = codes >= 0
        groups = np.random.default_rng([inputs.SEED, n_cells, n_groups]).integers(
            0, n_groups, n_cells
        )[has]
        pairs, cells = np.unique(
            groups * (int(codes.max()) + 1) + codes[has], return_counts=True
        )
        return pairs // (int(codes.max()) + 1), pairs % (int(codes.max()) + 1), cells

    def make(n_cells: int):
        pair_groups, pair_codes, pair_cells = counts(n_cells)
        return lambda: overlap_pairs(pair_groups, pair_codes, pair_cells, n_groups)

    def check(n_cells: int, table) -> None:
        pair_groups, pair_codes, _cells = counts(n_cells)
        assert len(table) == n_groups * (n_groups - 1) // 2
        first = set(pair_codes[pair_groups == 0].tolist())
        second = set(pair_codes[pair_groups == 1].tolist())
        assert table.iloc[0]["shared"] == len(first & second)
        assert ((table["jaccard"] >= 0) & (table["jaccard"] <= 1)).all()

    bench("repertoire.overlap_pairs", make, LADDER, check=check)


def test_neighbor_pairs(bench) -> None:
    from cytearc.repertoire.distances import neighbor_pairs

    def junctions(n_cells: int) -> list[str]:
        return sorted(set(_contigs(n_cells)["junction_aa"].tolist()))

    def make(n_cells: int):
        texts = junctions(n_cells)
        return lambda: (
            neighbor_pairs(texts, "hamming", 1),
            neighbor_pairs(texts, "levenshtein", 1),
        )

    def check(n_cells: int, result) -> None:
        texts = junctions(n_cells)
        (first, second), (edit_first, edit_second) = result
        for a, b in zip(first[:50].tolist(), second[:50].tolist(), strict=True):
            assert len(texts[a]) == len(texts[b])
            assert sum(x != y for x, y in zip(texts[a], texts[b], strict=True)) <= 1
        # Every pair within one substitution is within one edit.
        edits = set(zip(edit_first.tolist(), edit_second.tolist(), strict=True))
        assert set(zip(first.tolist(), second.tolist(), strict=True)) <= edits

    bench("repertoire.neighbor_pairs", make, LADDER, check=check)
