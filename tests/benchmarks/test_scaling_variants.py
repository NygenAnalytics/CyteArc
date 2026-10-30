"""Benchmarks of importing per-cell variant counts.

mgatk writes, for every cell, the reads of each base at each position of the
mitochondrial genome: about 1.2 rows per position at 100x. Importing them
parses the four base files and sorts their rows by cell, then turns each band
of cells into depth and counts. Each benchmark runs one step on synthetic
cells of 2,000 positions and checks its result against sums taken from the
drawn reads, so it doubles as a test at the smoke size.
"""

import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

N_POSITIONS = 2_000
LADDER = Ladder(sizes=(250, 500, 1_000, 2_000), smoke=12)


@lru_cache(maxsize=2)
def _reads(n_cells: int) -> np.ndarray:
    """Forward and reverse reads of each base, shaped (cells, positions, 4, 2):
    the reference base at 100x and a read error at about one position in six."""
    rng = np.random.default_rng([inputs.SEED, n_cells])
    reference = rng.integers(0, 4, size=N_POSITIONS)
    reads = np.zeros((n_cells, N_POSITIONS, 4, 2), dtype=np.int64)
    total = rng.poisson(100, size=(n_cells, N_POSITIONS))
    errors = rng.binomial(total, 0.002)
    forward = rng.binomial(total - errors, 0.5)
    cells, positions = np.indices((n_cells, N_POSITIONS))
    reads[cells, positions, reference, 0] = forward
    reads[cells, positions, reference, 1] = total - errors - forward
    wrong = (reference + 1) % 4
    reads[cells, positions, wrong, 0] = errors
    return reads


def _write_folder(n_cells: int, folder: Path) -> Path:
    """An mgatk folder of the drawn reads, cells in reverse order."""
    if folder.exists():
        return folder
    folder.mkdir(parents=True)
    reads = _reads(n_cells)
    for base_index, base in enumerate("ACGT"):
        lines = []
        for cell in range(n_cells)[::-1]:
            counts = reads[cell, :, base_index]
            for position in np.flatnonzero(counts.sum(axis=1)).tolist():
                lines.append(
                    f"{position + 1},c{cell},{counts[position, 0]},{counts[position, 1]}\n"
                )
        (folder / f"b.{base}.txt.gz").write_bytes(
            gzip.compress("".join(lines).encode(), 1)
        )
    (folder / "chrM_refAllele.txt").write_text(
        "".join(f"{position + 1}\tA\n" for position in range(N_POSITIONS))
    )
    return folder


def test_mgatk_base_files_parse(bench, tmp_path_factory) -> None:
    from cytearc.readers.mgatk import mgatk_source

    root = tmp_path_factory.mktemp("mgatk")

    def make(n_cells: int):
        folder = _write_folder(n_cells, root / str(n_cells))
        source = mgatk_source(folder, format=None, swap_strands=False)
        return lambda: sum(batch.values.sum() for batch in source.batches(200_000))

    def check(n_cells: int, total) -> None:
        assert total == _reads(n_cells).sum()

    bench("variants.mgatk_parse", make, LADDER, check=check)


def test_rows_sort_by_cell_and_finish_their_bands(bench) -> None:
    from cytearc.variants.counts import DEPTH_DTYPE, count_dtype, finish_band
    from cytearc.variants.sorting import CellRowSorter

    dtype = count_dtype(("forward", "reverse"))

    def rows(n_cells: int) -> np.ndarray:
        reads = _reads(n_cells)
        counts = reads.sum(axis=3)
        cells, positions, bases = np.nonzero(counts)
        drawn = np.zeros(len(cells), dtype=dtype)
        drawn["cell"] = cells
        drawn["key"] = positions * 4 + bases
        drawn["count"] = counts[cells, positions, bases]
        drawn["forward"] = reads[cells, positions, bases, 0]
        drawn["reverse"] = reads[cells, positions, bases, 1]
        # Rows arrive by base, as the four files give them.
        return drawn[np.argsort(bases, kind="stable")]

    def make(n_cells: int):
        from cytearc.variants.counts import SiteTable

        drawn = rows(n_cells)
        sites = SiteTable(
            contigs=(),
            site_contig=np.full(N_POSITIONS, -1),
            site_position=np.full(N_POSITIONS, -1),
            site_name=np.asarray(
                [f"p{index}" for index in range(N_POSITIONS)], dtype=object
            ),
            site_channel=np.full(N_POSITIONS, "", dtype=object),
            allele_site=np.repeat(np.arange(N_POSITIONS), 4),
            allele=np.tile(np.asarray(list("ACGT"), dtype=object), N_POSITIONS),
            allele_role=np.tile(
                np.asarray(
                    ["reference", "alternative", "alternative", "alternative"],
                    dtype=object,
                ),
                N_POSITIONS,
            ),
        )

        def run() -> np.ndarray:
            depth_total = np.zeros(n_cells, dtype=np.int64)
            with CellRowSorter(
                {"counts": (dtype, ("cell", "key"))},
                n_cells=n_cells,
                buffer_bytes=drawn.nbytes // 3,
            ) as sorter:
                for chunk in np.array_split(drawn, 8):
                    sorter.add("counts", chunk)
                for first, stop, band in sorter.bands(2_000_000):
                    depth, _counts = finish_band(
                        band["counts"],
                        np.empty(0, dtype=DEPTH_DTYPE),
                        sites=sites,
                        depth_roles=None,
                        reference_from_depth=False,
                        name_cell=str,
                    )
                    depth_total[first:stop] += np.bincount(
                        depth["cell"] - first,
                        weights=depth["depth"],
                        minlength=stop - first,
                    ).astype(np.int64)
            return depth_total

        return run

    def check(n_cells: int, depth_total) -> None:
        np.testing.assert_array_equal(depth_total, _reads(n_cells).sum(axis=(1, 2, 3)))

    bench("variants.sort_and_finish", make, LADDER, check=check)
