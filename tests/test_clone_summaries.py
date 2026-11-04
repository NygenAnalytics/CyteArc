"""Every clone summary on every clone source.

Each source saves the synthetic truth's clones its own way, in a store of the
truth's cells, and every summary must measure the truth's clones whatever
saved them. The expected values are counted cell by cell from the truth, so
they never come from the code under test. A pull request that adds a clone
source adds a row to ``SOURCES``, one to ``KEPT_APART`` when its clones keep
labels apart or to ``LABELS`` when they keep none, which gives the rule its
labels follow, or None, and one to ``TRUTHS`` when it cannot save every clone
of the truth; one that adds a summary adds a test here.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

import cytearc.metadata.selection as cell_values
import cytearc.storage.refs as artifact_refs
from cytearc.clones import (
    ClonalDiversity,
    ClonalExpansion,
    CloneCoupling,
    CloneFateBias,
    CloneOverlap,
    CloneTracking,
    Rarefaction,
)
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from cytearc.metadata.selection import CellValueSpec
from cytearc.repertoire import GeneUsage, Spectratype
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import collect_warnings
from cytearc.clones.allele_clones import ALLELE_DIAGNOSTICS
from cytearc.clones.barcode_clones import BARCODE_DIAGNOSTICS
from cytearc.metadata.arguments import JOIN_DIAGNOSTICS
from tests.clone_helpers import (
    coupling_reference,
    every_cell,
    import_allele_truth,
    import_calling_truth,
    import_truth_clones,
    import_truth_receptors,
    overlap_reference,
    write_truth_store,
)
from tests.clone_truth import (
    N_CELLS,
    PHYLOGENY_CUT_TIME,
    CloneTruth,
    allele_calling_truth,
    calling_truth,
    clone_truth,
    number_label,
    receptor_truth,
    truth_phylogeny,
    without_single_cell_clones,
    write_newick,
)
from tests.storage_helpers import insert_nullable_cell_column
from tests.variant_truth import (
    clone_variant_truth,
    genotype_clone_truth,
    write_genotype_tables,
    write_mgatk,
)

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()
RECEPTORS = receptor_truth(TRUTH)
MUTATED_RECEPTORS = receptor_truth(TRUTH, somatic_mutations=True)
SAMPLES = ("d2", "d4")


def _imported_clones(store: DataStore, folder: Path) -> ArtifactRef:
    return import_truth_clones(store, TRUTH, folder)


def _clonotypes(store: DataStore, folder: Path) -> ArtifactRef:
    # Under receptor_arms="any" a cell that lost one of its clone's chains
    # still joins the clone, so within each receptor type the clonotypes are
    # the truth's clones; its multichain and ambiguous cells are receptor
    # types of their own.
    receptors = import_truth_receptors(store, RECEPTORS, folder)
    return store.repertoire.clonotypes(receptors, receptor_arms="any")


def _clonotype_clusters(store: DataStore, folder: Path) -> ArtifactRef:
    # The later cells of each B cell clone carry chains one substitution away,
    # as somatic mutations give them, which identical junctions split;
    # clusters within one substitution keep each clone whole.
    receptors = import_truth_receptors(store, MUTATED_RECEPTORS, folder)
    return store.repertoire.clonotypes(
        receptors, receptor_arms="any", metric="hamming", max_distance=1
    )


def _barcode_clones(store: DataStore, folder: Path) -> ArtifactRef:
    # TREX's rules remove the errors, ambient molecules and partial cloneIDs
    # of the truth's barcodes and join the cells that lost one, keeping the
    # truth's clones, its singleton among them.
    calling = calling_truth(TRUTH, noise=("amplified", "ambient", "dropout", "partial"))
    with collect_warnings():
        barcodes = import_calling_truth(store, calling, folder, "trex")
    return store.clones.barcode_clones(barcodes, preset="trex")


def _variant_clones(store: DataStore, folder: Path) -> ArtifactRef:
    # Each clone carries two variants of its own, which every cell of it shows
    # and no other cell does, so the variants that two cells or more show join
    # exactly the cells of each clone of two cells or more.
    path = write_mgatk(clone_variant_truth(TRUTH), folder / "mgatk")
    with pytest.warns(UserWarning, match="match no cell of the store"):
        counts = store.variants.import_counts(path, cell_selection=every_cell(store))
    variants = store.variants.informative(counts, min_cells=2, min_vmr=0.01)
    return store.variants.frequency_clones(variants)


def _genotype_clones(store: DataStore, folder: Path) -> ArtifactRef:
    # Each clone carries a mutation of its own, called from one mutant UMI or
    # more; other targets read wild type or not at all, which leaves them
    # unknown, so cells of one clone agree wherever both were read.
    paths = write_genotype_tables(
        genotype_clone_truth(TRUTH), TRUTH.cell_ids.tolist(), folder / "got"
    )
    counts = store.variants.import_counts(paths, cell_selection=every_cell(store))
    genotypes = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    return store.variants.genotype_clones(genotypes)


def _allele_clones(store: DataStore, folder: Path) -> ArtifactRef:
    # DARLIN's rules: alleles of frequency below 2e-5 in at most one of the
    # bank's mice link cells, so neither the common alleles, the unedited
    # arrays and placeholders nor the homoplasic allele of three mice join
    # two clones; the truth's clones, its singleton among them, hold two
    # rare alleles each, below DARLIN's three.
    calling = allele_calling_truth(TRUTH, noise=("homoplasy",))
    alleles = import_allele_truth(store, calling, TRUTH, folder)
    return store.clones.allele_clones(
        alleles,
        method="graph",
        frequencies=calling.banks,
        max_frequency=2e-5,
        max_samples=1,
        max_alleles=3,
    )


def _phylogeny_clones(store: DataStore, folder: Path) -> ArtifactRef:
    # The truth's phylogeny splits into one clade per clone, whose root carries
    # the clone's label, so its cut at time 1 gives the truth's clones.
    path = write_newick([truth_phylogeny(TRUTH)], folder / "truth_tree.nwk")
    with pytest.warns(UserWarning, match="match no cell of the store"):
        phylogeny = store.clones.import_phylogeny(
            path, cell_selection=every_cell(store), time_unit="days"
        )
    return store.clones.phylogeny_clones(phylogeny, time=PHYLOGENY_CUT_TIME)


#: How each clone source saves the truth's clones, in the order the stack adds
#: them.
SOURCES: dict[str, Callable[[DataStore, Path], ArtifactRef]] = {
    "imported clones": _imported_clones,
    "clonotypes": _clonotypes,
    "clonotype clusters": _clonotype_clusters,
    "barcode clones": _barcode_clones,
    "variant clones": _variant_clones,
    "genotype clones": _genotype_clones,
    "allele clones": _allele_clones,
    "phylogeny clones": _phylogeny_clones,
}
#: The truth's clones as a source saves them, where it cannot save every one:
#: variant clones join each cell to the cells that share its variants, so the
#: truth's clone of one cell is no clone of theirs.
TRUTHS: dict[str, CloneTruth] = {
    "variant clones": without_single_cell_clones(TRUTH),
}
#: The labels that a source's clones keep apart, by key column, for each of the
#: truth's cells: every summary measures the cells of each label apart, so T
#: cells and B cells never share a group.
KEPT_APART: dict[str, dict[str, np.ndarray]] = {
    "clonotypes": {
        "receptor_type": RECEPTORS.expected["receptor_type"].to_numpy(dtype=object)
    },
    "clonotype clusters": {
        "receptor_type": MUTATED_RECEPTORS.expected["receptor_type"].to_numpy(
            dtype=object
        )
    },
}


def _ranked_label(clone: int) -> str:
    """A clone's label as barcode, genotype and allele clones number them, from the truth.

    ``clone_1`` is the largest clone, and clones of one size follow their
    first cell.
    """
    sizes = np.bincount(TRUTH.clones[TRUTH.clones >= 0], minlength=TRUTH.n_clones)
    first = [
        int(np.flatnonzero(TRUTH.clones == other)[0]) for other in range(TRUTH.n_clones)
    ]
    ranked = sorted(
        range(TRUTH.n_clones), key=lambda other: (-sizes[other], first[other])
    )
    return f"clone_{ranked.index(clone) + 1}"


#: How each source that keeps no labels apart labels the truth's clones: the
#: rule its labels follow from the truth, such as the labels of imported
#: clones' files and of the clade roots that phylogeny clones are named by,
#: and the numbers by size of barcode, genotype and allele clones, or None for
#: a source whose labels follow from something else, such as variant clones,
#: which number their clones by cluster.
LABELS: dict[str, Callable[[int], str] | None] = {
    "imported clones": number_label,
    "barcode clones": _ranked_label,
    "variant clones": None,
    "genotype clones": _ranked_label,
    "allele clones": _ranked_label,
    "phylogeny clones": number_label,
}


def test_every_source_says_how_it_labels_the_truths_clones() -> None:
    # The tracking row checks the labels of every source that keeps no labels
    # apart against its rule, so each source names its labels kept apart or
    # its rule, and never both.
    assert set(LABELS).isdisjoint(KEPT_APART)
    assert set(LABELS) | set(KEPT_APART) == set(SOURCES)


@pytest.fixture(scope="module", params=list(SOURCES))
def source(request, tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp(request.param.replace(" ", "_"))
    store = write_truth_store(folder / "store.zarr", TRUTH)
    store.cells.insert("is_d2", TRUTH.samples == "d2")
    store.cells.insert("is_d4", TRUTH.samples == "d4")
    insert_nullable_cell_column(
        store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    return SimpleNamespace(
        store=store,
        clones=SOURCES[request.param](store, folder),
        kept_apart=KEPT_APART.get(request.param, {}),
        labels=LABELS.get(request.param),
        truth=TRUTHS.get(request.param, TRUTH),
    )


def _groups(source, labels: np.ndarray | None = None) -> np.ndarray:
    """Each of the truth's cells' group, named as the summaries name it.

    A group is named by the cell's labels joined by commas: the labels given,
    then those that the source's clones keep apart; without any, every cell
    is in ``"all cells"``.
    """
    columns = [] if labels is None else [np.asarray(labels, dtype=object)]
    columns += [
        np.asarray(values, dtype=object) for values in source.kept_apart.values()
    ]
    if not columns:
        return np.full(N_CELLS, "all cells", dtype=object)
    return np.asarray(
        [", ".join(str(label) for label in parts) for parts in zip(*columns)],
        dtype=object,
    )


def _by_group(
    source, labels: np.ndarray | None = None, cells: np.ndarray | None = None
) -> dict[object, list[int]]:
    return source.truth.group_clone_sizes(_groups(source, labels), cells=cells)


def _group_cells(
    source, labels: np.ndarray, cells: np.ndarray | None = None
) -> dict[object, dict[int, int]]:
    """Each group's cells of each of the truth's clones, counted cell by cell.

    The cells of a group share the labels that the source's clones keep
    apart, so within a group each of the truth's clones is one of the
    source's clones.
    """
    return source.truth.group_clone_cells(_groups(source, labels), cells=cells)


def _kept_labels(source, labels: np.ndarray) -> dict[object, tuple[str, ...]]:
    """The labels that the source's clones keep apart, of each group by name."""
    kept = [np.asarray(values, dtype=object) for values in source.kept_apart.values()]
    found: dict[object, tuple[str, ...]] = {}
    for cell, group in enumerate(_groups(source, labels)):
        stratum = tuple(str(values[cell]) for values in kept)
        assert found.setdefault(group, stratum) == stratum
    return found


def _paired(groups, kept: dict[object, tuple[str, ...]]) -> set[frozenset]:
    """The pairs of groups that share the labels the clones keep apart."""
    return {
        frozenset((first, second))
        for first in groups
        for second in groups
        if str(first) < str(second) and kept[first] == kept[second]
    }


def _keys(source) -> list[tuple | None]:
    """Each of the truth's cells' clone as the source saves it.

    A clone of the truth within the labels that the source keeps apart, as
    its clone and those labels, or None for a cell without a clone.
    """
    kept = [np.asarray(values, dtype=object) for values in source.kept_apart.values()]
    return [
        None if clone < 0 else (int(clone), *(str(values[cell]) for values in kept))
        for cell, clone in enumerate(source.truth.clones.tolist())
    ]


def _source_clones(source) -> tuple[dict[tuple, str], dict[tuple, int]]:
    """The source's label and rank of each clone of ``_keys``.

    They come from the source's cells table, which must give each such clone
    one label of its own and no label to a cell without a clone; the rank is
    the clone's place in the source's table of clones, largest first.
    """
    loaded = source.store.clones.load_clones(source.clones)
    # Every source numbers its clones by size, largest first, which the ranks
    # and the tracking's ties follow, whatever rule its labels follow.
    assert loaded.clones["n_cells"].is_monotonic_decreasing
    ranks = {str(label): rank for rank, label in enumerate(loaded.clones["clone"])}
    of_cell = loaded.cells["clone"].reindex(TRUTH.cell_ids)
    labels: dict[tuple, str] = {}
    for key, label in zip(_keys(source), of_cell.tolist(), strict=True):
        if key is None:
            assert pd.isna(label)
            continue
        assert labels.setdefault(key, str(label)) == str(label)
    assert len(set(labels.values())) == len(labels)
    return labels, {key: ranks[label] for key, label in labels.items()}


def _grouping(source, labels: np.ndarray) -> dict[str, int]:
    """The grouping's diagnostics of a summary of every cell by labels that each holds, from the truth.

    The cells without a clone are left out, none lacks a label, and the
    groups are those that a cell with a clone holds.
    """
    return {
        "cellsWithoutClone": sum(key is None for key in _keys(source)),
        "cellsWithoutGroup": 0,
        "groups": len(_by_group(source, labels)),
    }


def test_every_source_records_the_clone_writers_diagnostics(source) -> None:
    # The clone writer counts the cells given a clone and the share of them
    # that the largest clone holds, counted here cell by cell from the
    # truth's clones as the source saves them, for clonotypes too.
    status = inspect_artifact(source.store.zw, source.clones)
    loaded = source.store.clones.load_clones(source.clones)
    assert dict(loaded.diagnostics) == status.diagnostics
    keys = [key for key in _keys(source) if key is not None]
    largest = max(Counter(keys).values())
    assert status.diagnostics["cellsWithClone"] == len(keys)
    assert status.diagnostics["largestCloneShare"] == largest / len(keys)
    # Each source's own keys follow the writer's, the join's before them.
    joined = ["import_clones"]
    own = {
        "import_clones": ["fileRows", "cellsInSeveralClones"],
        "call_barcode_clones": list(BARCODE_DIAGNOSTICS),
        "call_allele_clones": list(ALLELE_DIAGNOSTICS),
        "call_frequency_clones": [
            "variants",
            "cellsScored",
            "cellsWithoutCoverage",
            "cellsWithoutAlternative",
            "cellsWithoutNeighbors",
            "edges",
        ],
        "call_genotype_clones": ["sites", "mutantCells", "ambiguousCells"],
        "cut_phylogeny_clones": ["cellsBeforeCut"],
    }.get(str(status.operation), [])
    assert list(status.diagnostics) == [
        *(JOIN_DIAGNOSTICS if status.operation in joined else ()),
        "cellsWithClone",
        "largestCloneShare",
        *own,
    ]
    # Every cell of the truth's tree ends after the cut, and every cell of
    # the file is listed once, in one clone.
    if status.operation == "cut_phylogeny_clones":
        assert status.diagnostics["cellsBeforeCut"] == 0
    if status.operation == "import_clones":
        assert status.diagnostics["cellsInSeveralClones"] == 0


def test_clone_abundance_counts_the_clones_of_each_size(source) -> None:
    result = source.store.plots.clone_abundance(
        source.clones, group_by="sample", show=False
    )
    sizes = result.tables["sizes"]
    expected = _by_group(source, TRUTH.samples)
    assert set(sizes["group"]) == set(expected)
    for group, found in expected.items():
        rows = sizes[sizes["group"] == group]
        assert dict(zip(rows["clone_size"], rows["n_clones"], strict=True)) == {
            size: found.count(size) for size in set(found)
        }
    plt.close(result.figure)


def test_expansion_counts_each_clone_within_each_sample(source) -> None:
    store = source.store
    expansion = store.clones.expansion(source.clones, group_by="sample")
    loaded = store.clones.load_expansion(expansion)
    assert loaded.diagnostics == _grouping(source, TRUTH.samples)
    assert store.artifacts.inspect(expansion).diagnostics == loaded.diagnostics
    cells = loaded.cells.loc[TRUTH.cell_ids]
    np.testing.assert_array_equal(
        cells["group_clone_size"],
        source.truth.cell_clone_sizes(_groups(source, TRUTH.samples)),
    )
    expected = _by_group(source, TRUTH.samples)
    totals = loaded.categories.groupby("group")["n_cells"].sum().to_dict()
    assert totals == {group: sum(found) for group, found in expected.items()}
    clones = loaded.categories.groupby("group")["n_clones"].sum().to_dict()
    assert clones == {group: len(found) for group, found in expected.items()}


def test_diversity_measures_the_clone_sizes_of_each_sample(source) -> None:
    result = source.store.clones.diversity(source.clones, group_by="sample")
    assert result.diagnostics == _grouping(source, TRUTH.samples)
    indices = result.indices.set_index("group")
    expected = _by_group(source, TRUTH.samples)
    assert sorted(indices.index) == sorted(expected)
    for group, sizes in expected.items():
        found = np.asarray(sizes)
        shares = found / found.sum()
        row = indices.loc[group]
        assert row[
            ["n_cells", "n_clones", "n_singletons", "n_doubletons"]
        ].tolist() == [
            found.sum(),
            len(found),
            np.count_nonzero(found == 1),
            np.count_nonzero(found == 2),
        ]
        assert row["shannon"] == pytest.approx(-np.sum(shares * np.log(shares)))
        assert row["inverse_simpson"] == pytest.approx(1 / np.sum(shares**2))


def test_rarefaction_reaches_each_samples_observed_diversity(source) -> None:
    result = source.store.clones.rarefaction(
        source.clones, group_by="sample", orders=(0, 2)
    )
    assert result.diagnostics == _grouping(source, TRUTH.samples)
    observed = result.curves[result.curves["method"] == "observed"]
    expected = _by_group(source, TRUTH.samples)
    assert set(observed["group"]) == set(expected)
    for group, sizes in expected.items():
        found = np.asarray(sizes)
        shares = found / found.sum()
        rows = observed[observed["group"] == group].set_index("order")
        assert rows.loc[0, "size"] == found.sum()
        assert rows.loc[0, "diversity"] == len(found)
        assert rows.loc[2, "diversity"] == pytest.approx(1 / np.sum(shares**2))


def test_overlap_compares_the_clones_that_samples_share(source) -> None:
    result = source.store.clones.overlap(source.clones, group_by="sample")
    assert result.diagnostics == _grouping(source, TRUTH.samples)
    by_group = _group_cells(source, TRUTH.samples)
    kept = _kept_labels(source, TRUTH.samples)
    assert result.group_sizes.columns.tolist() == ["group", "n_cells", "n_clones"]
    assert result.group_keys.columns.tolist() == ["sample", *source.kept_apart]
    sizes = result.group_sizes.join(result.group_keys, on="group")
    columns = ["group", "sample", *source.kept_apart, "n_cells", "n_clones"]
    assert sorted(map(tuple, sizes[columns].values.tolist())) == sorted(
        (group, group.split(", ")[0], *kept[group], sum(found.values()), len(found))
        for group, found in by_group.items()
    )
    # Only groups of the same labels kept apart are compared.
    pairs = result.pairs.to_dict("records")
    assert {frozenset((row["group_1"], row["group_2"])) for row in pairs} == (
        _paired(by_group, kept)
    )
    for row in pairs:
        first, second = by_group[row["group_1"]], by_group[row["group_2"]]
        for column, value in overlap_reference(first, second).items():
            assert row[column] == pytest.approx(value, rel=1e-12)
    if not source.kept_apart:
        (row,) = pairs
        assert (row["group_1"], row["group_2"]) == SAMPLES


def test_track_follows_each_samples_largest_clones(source) -> None:
    tracked = source.store.clones.track(source.clones, group_by="sample", top_n=2)
    labels, ranks = _source_clones(source)
    if source.labels is not None:
        assert labels == {(clone,): source.labels(clone) for (clone,) in labels}
    by_group = _group_cells(source, TRUTH.samples)
    kept = _kept_labels(source, TRUTH.samples)
    # Each group's two largest clones, ties to the larger clone overall.
    chosen: set[tuple] = set()
    for group, found in by_group.items():
        keys = [(clone, *kept[group]) for clone in found]
        ranked = sorted(keys, key=lambda key: (-found[key[0]], ranks[key]))
        chosen |= set(ranked[:2])
    order = sorted(chosen, key=ranks.__getitem__)
    assert tracked.tracked == tuple(labels[key] for key in order)
    assert tracked.diagnostics == {
        **_grouping(source, TRUTH.samples),
        "trackedClones": len(order),
    }
    # Each clone is followed through the groups of its own labels kept apart,
    # zeros included.
    assert tracked.table.columns.tolist() == ["clone", "group", "n_cells", "fraction"]
    table = tracked.table.join(tracked.group_keys, on="group")
    columns = ["clone", "sample", *source.kept_apart, "n_cells"]
    assert sorted(map(tuple, table[columns].values.tolist())) == sorted(
        (labels[key], group.split(", ")[0], *kept[group], found.get(key[0], 0))
        for key in order
        for group, found in by_group.items()
        if kept[group] == key[1:]
    )
    assert tuple(tracked.table["clone"].drop_duplicates()) == tracked.tracked


def test_coupling_couples_the_cell_types_that_share_clones(source) -> None:
    result = source.store.clones.coupling(
        source.clones, group_by="cell_type", method="abundance_cosine"
    )
    by_group = _group_cells(source, TRUTH.cell_types)
    sizes = result.group_sizes
    assert dict(zip(sizes["group"], sizes["n_cells"], strict=True)) == {
        group: sum(found.values()) for group, found in by_group.items()
    }
    # Only groups of the same labels kept apart are coupled.
    assert {
        frozenset(pair) for pair in result.pairs[["group_1", "group_2"]].values
    } == _paired(by_group, _kept_labels(source, TRUTH.cell_types))
    for row in result.pairs.itertuples(index=False):
        assert row.coupling == pytest.approx(
            coupling_reference(by_group, row.group_1, row.group_2, "abundance_cosine"),
            rel=1e-12,
        )
    assert np.isfinite(result.pairs["z_score"]).all()
    assert sorted(result.order) == sorted(by_group)
    # Every pair's null spreads, so each z-score is finite.
    assert result.diagnostics == {
        **_grouping(source, TRUTH.cell_types),
        "pairsWithoutNullSpread": 0,
    }


def test_fate_bias_follows_early_cells_to_their_clones_late_cells(source) -> None:
    store = source.store
    early, late = TRUTH.samples == "d2", TRUTH.samples == "d4"
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(
            source.clones,
            group_by="cell_type",
            early=store.snapshot_cell_selection("is_d2"),
            late=store.snapshot_cell_selection("is_d4"),
        )
    )
    labels, _ranks = _source_clones(source)
    keys = _keys(source)
    fates = _groups(source, TRUTH.cell_types)
    # Each fate holds its late cells, those without a clone included unless
    # the clones keep labels apart, which a cell without a clone does not take.
    with_fate = [
        cell
        for cell in np.flatnonzero(late)
        if keys[cell] is not None or not source.kept_apart
    ]
    assert dict(
        zip(loaded.fates["fate"], loaded.fates["n_late_cells"], strict=True)
    ) == dict(Counter(fates[cell] for cell in with_fate))
    clone_fates: dict[tuple, Counter] = {}
    for cell in np.flatnonzero(late):
        if keys[cell] is not None:
            clone_fates.setdefault(keys[cell], Counter())[fates[cell]] += 1
    names = loaded.fates["fate"].tolist()
    assert loaded.cells.index.tolist() == TRUTH.cell_ids[early].tolist()
    assert loaded.cells.columns.tolist() == ["clone", "n_late_cells", "fate"]
    assert loaded.fractions.columns.tolist() == names
    assert loaded.fractions.index.tolist() == loaded.cells.index.tolist()
    for (cell_id, row), cell in zip(
        loaded.cells.iterrows(), np.flatnonzero(early), strict=True
    ):
        found = clone_fates.get(keys[cell], Counter()) if keys[cell] else Counter()
        total = sum(found.values())
        assert row["n_late_cells"] == total, cell_id
        if total:
            assert loaded.fractions.loc[cell_id, names].tolist() == pytest.approx(
                [found[name] / total for name in names]
            )
    assert set(loaded.clone_fates["clone"].astype(str)) == {
        labels[key] for key in clone_fates
    }
    # Each clone is tested in each fate of its labels kept apart, against the
    # late cells with a fate of those labels.
    kept = _kept_labels(source, TRUTH.cell_types)
    fate_cells = Counter(fates[cell] for cell in with_fate)
    expected: dict[tuple[str, str], tuple[int, float]] = {}
    for key, found in clone_fates.items():
        n_late = sum(found.values())
        own = [name for name in names if kept[name] == key[1:]]
        population = sum(fate_cells[name] for name in own)
        for name in own:
            x, in_fate = found[name], fate_cells[name]
            table = [[x, n_late - x], [in_fate - x, population - in_fate - n_late + x]]
            p_value = fisher_exact(table, alternative="greater").pvalue
            expected[(labels[key], name)] = (x, p_value)
    tests = {
        (str(row.clone), str(row.fate)): (row.n_cells, row.p_value)
        for row in loaded.clone_fates.itertuples(index=False)
    }
    assert tests.keys() == expected.keys()
    for name, (cells, p_value) in tests.items():
        assert cells == expected[name][0], name
        assert p_value == pytest.approx(expected[name][1], rel=1e-9), name
    # What the fate bias leaves out, counted from the truth: the late cells
    # without a clone, the early cells with a clone whose clone has no late
    # cell, and those whose clone's late cells tie between fates.
    early_found = [
        clone_fates.get(keys[cell], Counter())
        for cell in np.flatnonzero(early)
        if keys[cell] is not None
    ]
    tied = [
        found
        for found in early_found
        if found and list(found.values()).count(max(found.values())) > 1
    ]
    assert loaded.diagnostics == {
        "cellsWithoutClone": sum(keys[cell] is None for cell in np.flatnonzero(late)),
        "cellsWithoutGroup": 0,
        "groups": len(names),
        "earlyCellsWithoutLateCells": sum(not found for found in early_found),
        "earlyCellsWithTiedFates": len(tied),
        "testedClones": len(clone_fates),
    }


def _keyed_rows(table: pd.DataFrame, keys: pd.DataFrame, on: str = "group") -> set:
    """The labels of the groups that a table's rows name, from the record's keys.

    Every group that the table names is one of the keys' groups.
    """
    named = set(table[on].astype(str))
    assert named <= set(keys.index), named - set(keys.index)
    return set(map(tuple, keys.loc[sorted(named)].values.tolist()))


def test_summaries_report_each_label_their_clones_keep_apart(source) -> None:
    store = source.store
    columns = ["sample", *source.kept_apart]
    clonal = source.truth.clones >= 0
    labels = [TRUTH.samples, *source.kept_apart.values()]
    # Each group's labels in the key columns, from the truth's cells with a
    # clone.
    expected = sorted(
        {
            tuple(str(values[cell]) for values in labels)
            for cell in np.flatnonzero(clonal)
        }
    )
    diversity = store.clones.diversity(source.clones, group_by="sample")
    curves = store.clones.rarefaction(source.clones, group_by="sample")
    expansion = store.clones.load_expansion(
        store.clones.expansion(source.clones, group_by="sample")
    )
    overlap = store.clones.overlap(source.clones, group_by="sample")
    coupling = store.clones.coupling(source.clones, group_by="sample", n_permutations=0)
    for record, table in (
        (diversity, diversity.indices),
        (curves, curves.curves),
        (expansion, expansion.categories),
        (overlap, overlap.group_sizes),
        (coupling, coupling.group_sizes),
    ):
        keys = record.group_keys
        assert keys.index.name == "group"
        assert keys.columns.tolist() == columns
        assert sorted(map(tuple, keys.values.tolist())) == expected
        assert sorted(_keyed_rows(table, keys)) == expected
        # The tables name the groups and hold none of their labels.
        assert not set(columns) & set(table.columns)
    tracked = store.clones.track(source.clones, group_by="sample")
    assert tracked.group_keys.columns.tolist() == columns
    assert _keyed_rows(tracked.table, tracked.group_keys) <= set(expected)
    # The fates of the late cells by cell type, with the labels of their
    # clones when the clones keep labels apart.
    late = TRUTH.samples == "d4"
    with_fate = late & (clonal | (not source.kept_apart))
    by_type = [TRUTH.cell_types, *source.kept_apart.values()]
    fates = sorted(
        {
            tuple(str(values[cell]) for values in by_type)
            for cell in np.flatnonzero(with_fate)
        }
    )
    bias = store.clones.load_fate_bias(
        store.clones.fate_bias(
            source.clones,
            group_by="cell_type",
            early=store.snapshot_cell_selection("is_d2"),
            late=store.snapshot_cell_selection("is_d4"),
        )
    )
    assert bias.group_keys.index.name == "fate"
    assert bias.group_keys.columns.tolist() == ["cell_type", *source.kept_apart]
    assert sorted(map(tuple, bias.group_keys.values.tolist())) == fates
    assert sorted(_keyed_rows(bias.fates, bias.group_keys, on="fate")) == fates
    assert _keyed_rows(bias.clone_fates, bias.group_keys, on="fate") <= set(fates)
    if source.kept_apart:
        # T cells and B cells of one sample are measured apart.
        types = {key[:2] for key in expected}
        assert {("d2", "TCR"), ("d2", "BCR")} <= types


#: The tables of each summary's record, which name its groups and hold none of
#: their labels.
TABLES: dict[type, tuple[str, ...]] = {
    ClonalExpansion: ("cells", "categories", "rank_windows"),
    ClonalDiversity: ("indices", "hill"),
    Rarefaction: ("curves",),
    CloneOverlap: ("group_sizes", "pairs"),
    CloneTracking: ("table",),
    CloneCoupling: ("group_sizes", "pairs"),
    CloneFateBias: ("cells", "fractions", "clone_fates", "fates"),
    GeneUsage: ("table",),
    Spectratype: ("table",),
}
#: Names that the summaries' tables give columns of their own, which a
#: grouping may take as any other name.
OWN_COLUMNS = ("group", "clone", "n_cells", "fraction", "locus")


def _summaries(store: DataStore, clones: ArtifactRef, **grouping) -> list:
    """Every summary of the clones within the grouping, as its record."""
    records = [
        store.clones.load_expansion(store.clones.expansion(clones, **grouping)),
        store.clones.diversity(clones, **grouping),
        store.clones.rarefaction(clones, orders=(0, 2), **grouping),
        store.clones.overlap(clones, **grouping),
        store.clones.track(clones, top_n=2, **grouping),
        store.clones.coupling(clones, n_permutations=0, **grouping),
    ]
    if clones.kind == "clonotypes":
        # Gene usage and junction lengths read the clonotypes' chains.
        records.append(store.repertoire.gene_usage(clones, **grouping))
        records.append(store.repertoire.spectratype(clones, **grouping))
    return records


def _fate_bias(store: DataStore, clones: ArtifactRef, **grouping) -> CloneFateBias:
    return store.clones.load_fate_bias(
        store.clones.fate_bias(
            clones,
            early=store.snapshot_cell_selection("is_d2"),
            late=store.snapshot_cell_selection("is_d4"),
            **grouping,
        )
    )


def _assert_same_tables(found, expected) -> None:
    """The records hold equal tables, whatever their groupings were named."""
    assert type(found) is type(expected)
    for table in TABLES[type(found)]:
        pd.testing.assert_frame_equal(
            getattr(found, table), getattr(expected, table), obj=table
        )


def _assert_joins_recover(record, column: str, truth: dict[str, str]) -> None:
    """Each row of the record's tables of groups joins to its group's label.

    ``truth`` gives each group's label in ``column``, by the group's name, as
    the truth's cells give it. A key column named like a column of a table
    takes pandas' suffix in the join.
    """
    keys = record.group_keys
    for table in TABLES[type(record)]:
        rows = getattr(record, table)
        if keys.index.name not in rows.columns:
            continue
        joined = rows.join(keys, on=keys.index.name, rsuffix="_label")
        label = f"{column}_label" if column in rows.columns else column
        found = joined[[keys.index.name, label]].dropna()
        assert len(found) == rows[keys.index.name].notna().sum(), table
        for name, value in found.itertuples(index=False):
            assert value == truth[str(name)], (table, name)


def _labels_of_groups(
    source, grouping: np.ndarray, labels: np.ndarray, *, kept_apart: bool = True
) -> dict[str, str]:
    """Each group's label by its name, from the truth's cells of the group.

    The groups are those of the ``grouping`` labels, and of the labels that the
    source's clones keep apart unless ``kept_apart`` is False, as gene usage
    groups the cells.
    """
    names = (
        _groups(source, grouping) if kept_apart else np.asarray(grouping, dtype=object)
    )
    found: dict[str, str] = {}
    for name, label in zip(names, labels, strict=True):
        assert found.setdefault(str(name), str(label)) == str(label)
    return found


def _kept_apart_by(record) -> bool:
    """Whether a record's groups keep the clones' labels apart.

    Gene usage and junction lengths count chains within the groups alone.
    """
    return not isinstance(record, GeneUsage | Spectratype)


@pytest.mark.slow
def test_summaries_grouped_by_a_column_named_like_their_own_columns(source) -> None:
    store = source.store
    neutral = _summaries(store, source.clones, group_by="sample")
    neutral_bias = _fate_bias(store, source.clones, group_by="cell_type")
    for name in OWN_COLUMNS:
        # The samples under the summaries' own column name measure what the
        # samples do, and the joins give each row its sample, and the labels
        # that the clones keep apart.
        store.cells.insert(name, TRUTH.samples.astype(object), overwrite=True)
        for found, expected in zip(
            _summaries(store, source.clones, group_by=name), neutral, strict=True
        ):
            _assert_same_tables(found, expected)
            kept = _kept_apart_by(found)
            assert found.group_keys.columns.tolist() == [
                name,
                *(source.kept_apart if kept else ()),
            ]
            _assert_joins_recover(
                found,
                name,
                _labels_of_groups(
                    source, TRUTH.samples, TRUTH.samples, kept_apart=kept
                ),
            )
            for column, values in source.kept_apart.items() if kept else ():
                _assert_joins_recover(
                    found, column, _labels_of_groups(source, TRUTH.samples, values)
                )
        # Each cell of the expansion joins to its own sample.
        expansion = store.clones.load_expansion(
            store.clones.expansion(source.clones, group_by=name)
        )
        cells = expansion.cells.join(expansion.group_keys, on="group", rsuffix="_label")
        label = f"{name}_label" if name in expansion.cells.columns else name
        joined = cells.loc[TRUTH.cell_ids, label]
        grouped = cells.loc[TRUTH.cell_ids, "group"].notna().to_numpy()
        assert joined[grouped].tolist() == TRUTH.samples[grouped].tolist()
        assert joined[~grouped].isna().all()
        # The fates under the name measure what the cell types do.
        store.cells.insert(name, TRUTH.cell_types.astype(object), overwrite=True)
        bias = _fate_bias(store, source.clones, group_by=name)
        _assert_same_tables(bias, neutral_bias)
        assert bias.group_keys.columns.tolist() == [name, *source.kept_apart]
        _assert_joins_recover(
            bias, name, _labels_of_groups(source, TRUTH.cell_types, TRUTH.cell_types)
        )


@pytest.mark.parametrize("name", ["clone", "n_late_cells", "fate", "ids"])
def test_fates_named_like_the_tables_columns_keep_their_shares(source, name) -> None:
    store = source.store
    neutral = _fate_bias(store, source.clones, group_by="cell_type")
    # The monocytes' fate takes a name of the tables' own columns, which the
    # tables hold as a value; every share stays that of the cell types.
    renamed = np.where(TRUTH.cell_types == "Monocyte", name, TRUTH.cell_types)
    store.cells.insert("named_fate", renamed.astype(object), overwrite=True)
    bias = _fate_bias(store, source.clones, group_by="named_fate")
    names = {
        fate: fate.replace("Monocyte", name, 1) if fate.startswith("Monocyte") else fate
        for fate in neutral.fates["fate"].tolist()
    }
    order = bias.fates["fate"].tolist()
    assert sorted(order) == sorted(names.values())
    pd.testing.assert_frame_equal(
        bias.fractions, neutral.fractions.rename(columns=names)[order]
    )
    assert bias.cells.columns.tolist() == ["clone", "n_late_cells", "fate"]
    pd.testing.assert_series_equal(
        bias.cells["n_late_cells"], neutral.cells["n_late_cells"]
    )
    assert bias.cells["fate"].astype(object).tolist() == [
        names[fate] if isinstance(fate, str) else fate
        for fate in neutral.cells["fate"].astype(object).tolist()
    ]
    # The fates' labels hold the name under the grouping column's own name.
    renamed_fates = [fate for fate in order if fate.startswith(name)]
    assert renamed_fates
    assert set(bias.group_keys.loc[renamed_fates, "named_fate"]) == {name}


def test_a_label_artifact_of_a_kind_named_like_a_column_groups_cells(
    source, monkeypatch
) -> None:
    store = source.store
    # A kind of label artifact named like the summaries' count column, which
    # no release defines; its key column takes the kind's name.
    monkeypatch.setattr(
        artifact_refs, "ARTIFACT_KINDS", artifact_refs.ARTIFACT_KINDS | {"n_cells"}
    )
    monkeypatch.setattr(
        cell_values,
        "CELL_VALUE_NAMES",
        MappingProxyType(
            {**cell_values.CELL_VALUE_NAMES, "n_cells": CellValueSpec("values", True)}
        ),
    )
    planned = plan_cell_data_artifact(
        store.zw,
        scope="datastore",
        kind="n_cells",
        operation="label_cells_for_test",
        parameters={"labels": "samples"},
        inputs={},
        execution_options={},
        cell_selection=every_cell(store),
        arrays={"values": ((N_CELLS,), None)},
    )
    write_cell_data_artifact(
        store.zw, planned, {"values": TRUTH.samples.astype(object)}
    )
    labelled = store.snapshot_cluster_labels("sample", cell_selection=every_cell(store))
    for found, expected in zip(
        _summaries(store, source.clones, groups=planned.ref),
        _summaries(store, source.clones, groups=labelled),
        strict=True,
    ):
        _assert_same_tables(found, expected)
        kept = _kept_apart_by(found)
        assert found.group_keys.columns.tolist() == [
            "n_cells",
            *(source.kept_apart if kept else ()),
        ]
        _assert_joins_recover(
            found,
            "n_cells",
            _labels_of_groups(source, TRUTH.samples, TRUTH.samples, kept_apart=kept),
        )


def test_summaries_keep_to_a_cell_selection(source) -> None:
    store = source.store
    d2 = TRUTH.samples == "d2"
    selection = store.snapshot_cell_selection("is_d2")
    expected = _by_group(source, cells=d2)
    diversity = store.clones.diversity(source.clones, cell_selection=selection)
    assert {
        group: [cells, clones]
        for group, cells, clones in diversity.indices[
            ["group", "n_cells", "n_clones"]
        ].values.tolist()
    } == {group: [sum(found), len(found)] for group, found in expected.items()}
    expansion = store.clones.load_expansion(
        store.clones.expansion(source.clones, cell_selection=selection)
    )
    assert expansion.cells.index.tolist() == TRUTH.cell_ids[d2].tolist()
    np.testing.assert_array_equal(
        expansion.cells["group_clone_size"],
        source.truth.cell_clone_sizes(_groups(source), cells=d2)[d2],
    )
    by_type = _group_cells(source, TRUTH.cell_types, cells=d2)
    cells_of = {group: sum(found.values()) for group, found in by_type.items()}
    overlap = store.clones.overlap(
        source.clones, group_by="cell_type", cell_selection=selection
    )
    sizes = overlap.group_sizes
    assert dict(zip(sizes["group"], sizes["n_cells"], strict=True)) == cells_of
    # The source's largest clone, followed through the cells of d2.
    labels, ranks = _source_clones(source)
    largest = min(ranks, key=ranks.__getitem__)
    tracked = store.clones.track(
        source.clones, labels=[labels[largest]], cell_selection=selection
    )
    in_d2 = sum(key == largest for key, in_d2 in zip(_keys(source), d2) if in_d2)
    assert tracked.table[["clone", "n_cells"]].values.tolist() == [
        [labels[largest], in_d2]
    ]
    coupling = store.clones.coupling(
        source.clones, group_by="cell_type", cell_selection=selection, n_permutations=0
    )
    sizes = coupling.group_sizes
    assert dict(zip(sizes["group"], sizes["n_cells"], strict=True)) == cells_of


def test_summaries_refuse_a_grouping_that_leaves_out_every_clone(source) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")

    def fate_bias(clones, **options):
        return store.clones.fate_bias(clones, early=early, late=late, **options)

    for summary in (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
        store.clones.overlap,
        store.clones.track,
        store.clones.coupling,
        fate_bias,
    ):
        with pytest.raises(ValueError, match="No cell has both a clone and a group"):
            summary(source.clones, group_by="unlabelled")


def test_summaries_name_missing_and_incomplete_clones(source, tmp_path) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")

    def fate_bias(clones):
        return store.clones.fate_bias(
            clones, group_by="cell_type", early=early, late=late
        )

    summaries = (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
        lambda clones: store.clones.overlap(clones, group_by="sample"),
        store.clones.track,
        lambda clones: store.clones.coupling(
            clones, group_by="cell_type", n_permutations=0
        ),
        fate_bias,
    )

    def export_airr(clones):
        return store.repertoire.export_airr(clones, tmp_path / "contigs.tsv")

    if source.clones.kind == "clonotypes":
        # Gene usage, junction lengths and the AIRR export read clonotypes,
        # and their receptors, as the summaries read them; other clones hold
        # no chains.
        summaries += (store.repertoire.gene_usage, store.repertoire.spectratype)
        summaries += (export_airr,)
    else:
        for measure in (
            store.repertoire.gene_usage,
            store.repertoire.spectratype,
            export_airr,
        ):
            with pytest.raises(ValueError, match="immune_receptors or clonotypes"):
                measure(source.clones)
    # Each saved result, with the loader that reads it.
    saved = {
        store.clones.expansion(source.clones): store.clones.load_expansion,
        fate_bias(source.clones): store.clones.load_fate_bias,
    }
    # Artifacts that the store does not hold raise KeyError.
    for summary in summaries:
        with pytest.raises(KeyError, match="Artifact does not exist"):
            summary(_unsaved(source.clones))
    for result, load in saved.items():
        with pytest.raises(KeyError, match="Artifact does not exist"):
            load(_unsaved(result))
    # An import that stopped before it finished leaves incomplete clones,
    # which raise RuntimeError, also where a saved result reads them.
    loads_of_saved = tuple(
        lambda _ref, result=result, load=load: load(result)
        for result, load in saved.items()
    )
    # Clonotypes also read their receptors' types, through every summary.
    status = inspect_artifact(store.zw, source.clones)
    receptors = (
        [status.input_ref("receptors")] if "receptors" in (status.inputs or {}) else []
    )
    summaries_of_clones = tuple(
        lambda _ref, summary=summary: summary(source.clones) for summary in summaries
    )
    for ref, readers in (
        (source.clones, (*summaries, *loads_of_saved)),
        *((result, (load,)) for result, load in saved.items()),
        *((receptor_ref, summaries_of_clones) for receptor_ref in receptors),
    ):
        attributes = artifact_group(store.zw, ref).attrs
        attributes["complete"] = False
        try:
            for read in readers:
                with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                    read(ref)
        finally:
            attributes["complete"] = True
    for result, load in saved.items():
        assert len(load(result).cells) > 0
    # No refused export wrote its file.
    assert not (tmp_path / "contigs.tsv").exists()


def _unsaved(ref: ArtifactRef) -> ArtifactRef:
    """A reference of the same kind to an artifact that no store holds."""
    return ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)
