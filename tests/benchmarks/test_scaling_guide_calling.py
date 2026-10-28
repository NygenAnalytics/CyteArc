"""Benchmarks of the guide-calling kernels of pooled CRISPR screens.

Guide calling reads the CRISPR counts twice. The first pass counts how many
cells hold each UMI count of each guide, the fits set each guide's threshold
from its histogram, and the second pass calls the guides of every cell and
summarises its counts. Each benchmark runs the kernel of one step on a
simulated screen of 1,000 guides, in which every cell holds ambient guide
molecules and most cells carry one guide, and checks the result, so it doubles
as a test at the smoke size.
"""

from collections.abc import Iterator
from functools import lru_cache

import numpy as np
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

N_GUIDES = 1_000
LADDER = Ladder(sizes=(10_000, 20_000, 40_000, 80_000), smoke=600)
# Fits cost little per cell, so their ladder climbs higher.
FIT_LADDER = Ladder(sizes=(40_000, 80_000, 160_000, 320_000), smoke=600)
_ROWS = 5_000


def _screen_blocks(n_cells: int) -> Iterator[np.ndarray]:
    """Yield the guide counts of a simulated low-MOI screen in row blocks.

    Every cell holds Poisson ambient counts with mean 0.2 for every guide, and
    eight cells in ten also carry one guide at about 2**6 UMIs.
    """
    for index, start in enumerate(range(0, n_cells, _ROWS)):
        rng = np.random.default_rng([inputs.SEED, index])
        rows = min(_ROWS, n_cells - start)
        block = rng.poisson(0.2, size=(rows, N_GUIDES)).astype(np.uint16)
        carriers = np.flatnonzero(rng.random(rows) < 0.8)
        guides = rng.integers(0, N_GUIDES, size=len(carriers))
        signal = np.round(2.0 ** rng.normal(6.0, 1.0, size=len(carriers))) + 1
        block[carriers, guides] += np.minimum(signal, 60_000).astype(np.uint16)
        yield block


@lru_cache(maxsize=1)
def _screen(n_cells: int) -> np.ndarray:
    return np.concatenate(list(_screen_blocks(n_cells)))


def _blocks(counts: np.ndarray) -> list[np.ndarray]:
    return [counts[start : start + _ROWS] for start in range(0, len(counts), _ROWS)]


@lru_cache(maxsize=1)
def _histograms(n_cells: int):
    """Count the histograms block by block, so no full count matrix is held."""
    from cytearc.perturbation.histograms import guide_count_histograms

    return guide_count_histograms(
        _screen_blocks(n_cells),
        np.zeros(n_cells, dtype=np.int64),
        n_samples=1,
        n_guides=N_GUIDES,
    )


def test_guide_count_histograms(bench) -> None:
    from cytearc.perturbation.histograms import guide_count_histograms

    def make(n_cells: int):
        counts = _screen(n_cells)
        blocks = _blocks(counts)
        codes = np.zeros(n_cells, dtype=np.int64)
        return lambda: guide_count_histograms(
            blocks, codes, n_samples=1, n_guides=N_GUIDES
        )

    def check(n_cells: int, found) -> None:
        counts = _screen(n_cells).astype(np.int64)
        np.testing.assert_array_equal(found.nonzero_cells()[0], (counts > 0).sum(0))
        np.testing.assert_array_equal(found.max_counts()[0], counts.max(0))
        weighted = np.zeros(N_GUIDES, dtype=np.int64)
        entries = np.repeat(np.arange(N_GUIDES), np.diff(found.offsets))
        np.add.at(weighted, entries, found.counts * found.cells)
        np.testing.assert_array_equal(weighted, counts.sum(0))

    bench("perturbation.guide_count_histograms", make, LADDER, check=check)


@pytest.mark.parametrize("method", ["poisson_gaussian", "gaussian"])
def test_guide_threshold_fits(bench, method) -> None:
    from cytearc.perturbation.mixtures import (
        FITTED,
        fit_gaussian,
        fit_poisson_gaussian,
    )

    fit = fit_poisson_gaussian if method == "poisson_gaussian" else fit_gaussian

    def make(n_cells: int):
        histograms = _histograms(n_cells)
        return lambda: fit(histograms)

    def check(n_cells: int, fits) -> None:
        fitted = fits.status[0] == FITTED
        assert (fits.threshold[0, ~fitted] == 0).all()
        thresholds = fits.threshold[0, fitted]
        # Over every cell, the two Gaussians split cells without UMIs from
        # cells with ambient ones, the weakness that min_umi guards against.
        assert (
            (thresholds >= 1 if method == "gaussian" else thresholds >= 2)
            & (thresholds <= 40)
        ).all()
        # At the smoke size most guides have no carrier; on the ladder all fit.
        if n_cells >= FIT_LADDER.sizes[0]:
            assert fitted.all()

    bench(f"perturbation.fit_{method}", make, FIT_LADDER, check=check)


def test_threshold_calls(bench) -> None:
    from cytearc.perturbation.calling import threshold_calls

    thresholds = np.full((1, N_GUIDES), 8, dtype=np.uint32)

    def make(n_cells: int):
        counts = _screen(n_cells)
        blocks = _blocks(counts)
        codes = np.zeros(n_cells, dtype=np.int64)
        return lambda: threshold_calls(blocks, codes, thresholds)

    def check(n_cells: int, result) -> None:
        calls, summary = result
        counts = _screen(n_cells)
        rows, guides = np.nonzero(counts >= 8)
        np.testing.assert_array_equal(calls.cells, rows)
        np.testing.assert_array_equal(calls.codes, guides)
        np.testing.assert_array_equal(summary.total_umi, counts.sum(1, dtype=np.uint64))

    bench("perturbation.threshold_calls", make, LADDER, check=check)
