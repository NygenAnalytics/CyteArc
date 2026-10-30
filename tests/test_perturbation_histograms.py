import tracemalloc

import numpy as np
import pytest

from cytearc.perturbation import histograms
from cytearc.perturbation.histograms import (
    DENSE_COUNTS,
    MAX_COUNT,
    checked_sample_codes,
    guide_count_histograms,
    histogram_bytes,
)
from tests.perturbation_helpers import screen_counts, use_python_kernels


def _blocks(counts: np.ndarray, *sizes: int) -> list[np.ndarray]:
    """Split counts into consecutive row blocks of the given sizes."""
    edges = np.cumsum((0, *sizes))
    return [counts[start:stop] for start, stop in zip(edges[:-1], edges[1:])]


def _check_against_numpy(found, counts: np.ndarray, samples: np.ndarray) -> None:
    n_samples = found.n_samples
    assert found.n_cells.tolist() == np.bincount(samples, minlength=n_samples).tolist()
    for sample in range(n_samples):
        rows = counts[samples == sample].astype(np.int64)
        for guide in range(counts.shape[1]):
            column = rows[:, guide]
            values, cells = np.unique(column[column > 0], return_counts=True)
            found_values, found_cells = found.histogram(sample, guide)
            np.testing.assert_array_equal(found_values, values)
            np.testing.assert_array_equal(found_cells, cells)
            assert found.nonzero_cells()[sample, guide] == (column > 0).sum()
            assert found.zero_cells()[sample, guide] == (column == 0).sum()
            assert found.distinct_counts()[sample, guide] == len(values)
            assert found.max_counts()[sample, guide] == column.max(initial=0)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_histograms_count_every_cell_of_every_guide_exactly(dtype) -> None:
    counts = screen_counts(300, 6, seed=1, log2_mean=5.0, dtype=np.uint32)
    if dtype is np.uint8:
        counts = np.minimum(counts, 255)
    counts = counts.astype(dtype)
    samples = np.random.default_rng(2).integers(0, 3, size=300)
    found = guide_count_histograms(
        _blocks(counts, 120, 0, 100, 80), samples, n_samples=3, n_guides=6
    )
    _check_against_numpy(found, counts, samples)
    # Counts on both sides of the dense range are present.
    assert found.counts.max() > DENSE_COUNTS
    assert found.counts.min() == 1


def test_histograms_hold_samples_and_guides_without_counts() -> None:
    counts = np.zeros((4, 3), dtype=np.uint16)
    counts[0, 0] = 2
    counts[1, 0] = 2
    counts[2, 0] = 70
    samples = np.asarray([0, 0, 2, 2])
    found = guide_count_histograms([counts], samples, n_samples=3, n_guides=3)
    _check_against_numpy(found, counts, samples)
    assert found.histogram(0, 0)[0].tolist() == [2]
    assert found.histogram(0, 0)[1].tolist() == [2]
    assert found.histogram(1, 0)[0].tolist() == []
    assert found.max_counts().tolist() == [[2, 0, 0], [0, 0, 0], [70, 0, 0]]


def test_histograms_of_no_cells_are_empty() -> None:
    found = guide_count_histograms(
        [], np.zeros(0, dtype=np.int64), n_samples=1, n_guides=2
    )
    assert found.n_cells.tolist() == [0]
    assert found.offsets.tolist() == [0, 0, 0]
    assert found.zero_cells().tolist() == [[0, 0]]


def test_histograms_merge_large_counts_in_rounds(monkeypatch) -> None:
    counts = screen_counts(400, 5, seed=3, log2_mean=7.0)
    samples = np.random.default_rng(4).integers(0, 2, size=400)
    expected = guide_count_histograms(
        _blocks(counts, 150, 250), samples, n_samples=2, n_guides=5
    )
    # No reserved room per row, so a block with more than 64 large counts
    # takes the second pass, and every block merges what waits.
    monkeypatch.setattr(histograms, "_SPARSE_PER_ROW", 0)
    monkeypatch.setattr(histograms, "_MERGE_ENTRIES", 3)
    found = guide_count_histograms(
        _blocks(counts, 150, 250), samples, n_samples=2, n_guides=5
    )
    for name in ("offsets", "counts", "cells", "n_cells"):
        np.testing.assert_array_equal(getattr(found, name), getattr(expected, name))


def test_python_kernel_counts_like_the_compiled_one(monkeypatch) -> None:
    counts = screen_counts(200, 4, seed=5, log2_mean=6.0)
    samples = np.random.default_rng(6).integers(0, 2, size=200)
    expected = guide_count_histograms(
        _blocks(counts, 70, 130), samples, n_samples=2, n_guides=4
    )
    use_python_kernels(monkeypatch, histograms)
    monkeypatch.setattr(histograms, "_SPARSE_PER_ROW", 0)
    found = guide_count_histograms(
        _blocks(counts, 70, 130), samples, n_samples=2, n_guides=4
    )
    for name in ("offsets", "counts", "cells"):
        np.testing.assert_array_equal(getattr(found, name), getattr(expected, name))
    wide = np.full((1, 4), MAX_COUNT + 1, dtype=np.uint64)
    with pytest.raises(ValueError, match="at most"):
        guide_count_histograms([wide], [0], n_samples=1, n_guides=4)


@pytest.mark.parametrize("value", [MAX_COUNT + 1, 2**63 + 5])
def test_histograms_refuse_counts_beyond_uint32(value) -> None:
    counts = np.zeros((2, 3), dtype=np.uint64)
    counts[1, 2] = value
    with pytest.raises(ValueError, match=f"at most {MAX_COUNT}"):
        guide_count_histograms([counts], [0, 0], n_samples=1, n_guides=3)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n_samples": 0}, ValueError, "at least one sample"),
        ({"n_guides": 0}, ValueError, "at least one sample and one guide"),
        ({"n_samples": 2**16, "n_guides": 2**15}, ValueError, "exceed 2\\*\\*31"),
        ({"sample_codes": [[0, 0]]}, ValueError, "one-dimensional"),
        ({"sample_codes": [0.0, 0.0]}, ValueError, "must be integers"),
        ({"sample_codes": [0, 1]}, ValueError, "lie in \\[0, 1\\)"),
        ({"sample_codes": [0, -1]}, ValueError, "lie in \\[0, 1\\)"),
        ({"blocks": [np.zeros((2, 3), dtype=np.uint8)]}, ValueError, "2 columns"),
        ({"blocks": [np.zeros((2, 2), dtype=np.int32)]}, TypeError, "unsigned"),
        ({"blocks": [np.zeros((3, 2), dtype=np.uint8)]}, ValueError, "more than 2"),
        ({"blocks": [np.zeros((1, 2), dtype=np.uint8)]}, ValueError, "hold 1 rows"),
    ],
)
def test_histograms_refuse_inputs_they_cannot_count(arguments, error, message) -> None:
    call = {
        "blocks": [np.zeros((2, 2), dtype=np.uint8)],
        "sample_codes": [0, 0],
        "n_samples": 1,
        "n_guides": 2,
    }
    call.update(arguments)
    with pytest.raises(error, match=message):
        guide_count_histograms(
            call["blocks"],
            call["sample_codes"],
            n_samples=call["n_samples"],
            n_guides=call["n_guides"],
        )


def test_sample_codes_are_returned_as_int64() -> None:
    codes = checked_sample_codes(np.asarray([0, 1, 1], dtype=np.uint8), 2)
    assert codes.dtype == np.int64
    assert codes.tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    ("n_samples", "merge_entries", "rows"),
    [
        # The larger counts merge once, after the last block.
        (1, None, 2_000),
        (2, None, 2_000),
        # They merge after every block, each merge sorting every count so far.
        (1, 1_000, 2_000),
        # One block holds every count, which one merge sorts, and the last
        # merge sorts them again.
        (1, 1_000, 20_000),
    ],
)
def test_the_memory_estimate_bounds_the_histograms_peak(
    monkeypatch, n_samples, merge_entries, rows
) -> None:
    if merge_entries is not None:
        monkeypatch.setattr(histograms, "_MERGE_ENTRIES", merge_entries)
    # The first call loads the kernels.
    guide_count_histograms(
        [np.full((2, 3), 40, np.uint32)], np.zeros(2, np.int64), n_samples=1, n_guides=3
    )
    # Every cell carries two guides above the dense counts, as many as the
    # estimate expects, each at a count of its own, so that no two cells
    # share a count and the merges keep every one.
    n_cells, n_guides = 20_000, 50
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.3, size=(n_cells, n_guides)).astype(np.uint32)
    cells = np.arange(n_cells)
    counts[cells, cells % n_guides] = 33 + cells
    counts[cells, (cells + 1) % n_guides] = 33 + n_cells + cells
    blocks = [counts[start : start + rows] for start in range(0, n_cells, rows)]
    # The sample codes, like the blocks, belong to the caller.
    samples = cells % n_samples
    tracemalloc.start()
    try:
        found = guide_count_histograms(
            blocks, samples, n_samples=n_samples, n_guides=n_guides
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Each larger count is a histogram entry of its own.
    assert np.count_nonzero(found.counts > DENSE_COUNTS) == 2 * n_cells
    # Each block also reserves 16 bytes per row while it is counted.
    assert peak <= histogram_bytes(n_samples, n_guides, n_cells) + 16 * rows
