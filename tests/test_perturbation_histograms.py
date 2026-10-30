import re
import tracemalloc
from collections.abc import Callable

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
    # Two carried guides in a tenth of the cells, at counts that other cells
    # hold too, so that merges add cells to counts merged before.
    counts[::10, 4] = 40
    samples = np.random.default_rng(4).integers(0, 2, size=400)
    expected = guide_count_histograms(
        _blocks(counts, 150, 250), samples, n_samples=2, n_guides=5
    )
    _check_against_numpy(expected, counts, samples)
    # A buffer of three keys fills many times in each block, and its keys
    # merge into the earlier ones each time.
    monkeypatch.setattr(histograms, "_MOST_KEYS", 3)
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
    # The buffer fills within rows, so each pass ends and starts mid-row.
    monkeypatch.setattr(histograms, "_MOST_KEYS", 3)
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
        ({"memory_bytes": 1e9}, TypeError, "memory_bytes must be an integer"),
        ({"memory_bytes": True}, TypeError, "memory_bytes must be an integer"),
        ({"memory_bytes": -1}, ValueError, "memory_bytes must be at least 0"),
    ],
)
def test_histograms_refuse_inputs_they_cannot_count(arguments, error, message) -> None:
    call = {
        "blocks": [np.zeros((2, 2), dtype=np.uint8)],
        "sample_codes": [0, 0],
        "n_samples": 1,
        "n_guides": 2,
        "memory_bytes": None,
    }
    call.update(arguments)
    with pytest.raises(error, match=message):
        guide_count_histograms(
            call["blocks"],
            call["sample_codes"],
            n_samples=call["n_samples"],
            n_guides=call["n_guides"],
            memory_bytes=call["memory_bytes"],
        )


def test_sample_codes_are_returned_as_int64() -> None:
    codes = checked_sample_codes(np.asarray([0, 1, 1], dtype=np.uint8), 2)
    assert codes.dtype == np.int64
    assert codes.tolist() == [0, 1, 1]


def _carried_counts(
    n_cells: int, n_guides: int, per_cell: int, seed: int
) -> np.ndarray:
    """Ambient counts, and ``per_cell`` guides of each cell above the dense counts.

    Each carried count is drawn from 5,000 values, so that most of them are
    counts of their own and the histograms keep them apart.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.3, size=(n_cells, n_guides)).astype(np.uint32)
    cells = np.arange(n_cells)
    for carried in range(per_cell):
        guides = (cells * per_cell + carried) % n_guides
        counts[cells, guides] = DENSE_COUNTS + 1 + rng.integers(0, 5_000, n_cells)
    return counts


def _distinct_counts(counts: np.ndarray, samples: np.ndarray) -> int:
    """Count the distinct nonzero counts of every guide in every sample with NumPy."""
    rows, guides = np.nonzero(counts)
    entries = samples[rows] * counts.shape[1] + guides
    return len(np.unique(entries * 2**32 + counts[rows, guides]))


@pytest.mark.parametrize(
    ("per_cell", "n_samples", "rows", "most_keys"),
    [
        # A low-MOI screen, whose cells carry one guide.
        (1, 1, 2_000, None),
        # High-MOI screens, whose cells carry 3, 10 or 30 guides.
        (3, 1, 2_000, None),
        (10, 1, 2_000, None),
        (30, 1, 2_000, None),
        # Three samples, one block of every cell, and a buffer that fills
        # every hundred cells, so that the counts merge 2,000 times.
        (10, 3, 2_000, None),
        (10, 1, 20_000, None),
        (10, 1, 2_000, 1_000),
    ],
)
def test_histograms_hold_no_more_than_the_bytes_of_their_distinct_counts(
    monkeypatch, per_cell, n_samples, rows, most_keys
) -> None:
    if most_keys is not None:
        monkeypatch.setattr(histograms, "_MOST_KEYS", most_keys)
    # The first call loads the kernels.
    guide_count_histograms(
        [np.full((2, 3), 40, np.uint32)], np.zeros(2, np.int64), n_samples=1, n_guides=3
    )
    n_cells, n_guides = 20_000, 200
    counts = _carried_counts(n_cells, n_guides, per_cell, seed=3)
    blocks = [counts[start : start + rows] for start in range(0, n_cells, rows)]
    # The sample codes, like the blocks, belong to the caller.
    samples = np.arange(n_cells) % n_samples
    n_distinct = _distinct_counts(counts, samples)
    planned = histogram_bytes(n_samples, n_guides, n_cells, n_distinct=n_distinct)
    tracemalloc.start()
    try:
        found = guide_count_histograms(
            blocks,
            samples,
            n_samples=n_samples,
            n_guides=n_guides,
            memory_bytes=planned,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(found.counts) == n_distinct
    assert peak <= planned


def test_few_cells_hold_no_more_than_the_bytes_of_their_distinct_counts(
    monkeypatch,
) -> None:
    # 64 cells, whose larger counts merge after every three, so that the
    # call's Python objects and sorting take most of its memory.
    monkeypatch.setattr(histograms, "_MOST_KEYS", 3)
    guide_count_histograms(
        [np.full((2, 3), 40, np.uint32)], np.zeros(2, np.int64), n_samples=1, n_guides=3
    )
    counts = _carried_counts(64, 3, 1, seed=5)
    samples = np.zeros(64, dtype=np.int64)
    blocks = [counts[start : start + 7] for start in range(0, 64, 7)]
    n_distinct = _distinct_counts(counts, samples)
    tracemalloc.start()
    try:
        found = guide_count_histograms(blocks, samples, n_samples=1, n_guides=3)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(found.counts) == n_distinct
    assert peak <= histogram_bytes(1, 3, 64, n_distinct=n_distinct)


def _refusal(count: Callable[[], object]) -> tuple[str, int, int]:
    """Return a refused count's message, the bytes it names, and the peak until then.

    The peak is read where the MemoryError is caught, before pytest records
    it.
    """
    tracemalloc.start()
    try:
        count()
    except MemoryError as error:
        peak = tracemalloc.get_traced_memory()[1]
        found = re.search(r"need about (\d+) bytes", str(error))
        assert found is not None, str(error)
        return str(error), int(found[1]), peak
    finally:
        tracemalloc.stop()
    raise AssertionError("The count was not refused")


def test_histograms_raise_before_they_outgrow_their_memory() -> None:
    guide_count_histograms(
        [np.full((2, 3), 40, np.uint32)], np.zeros(2, np.int64), n_samples=1, n_guides=3
    )
    # Cells that carry ten guides each, and half the memory that their
    # histograms need.
    n_cells, n_guides = 20_000, 200
    counts = _carried_counts(n_cells, n_guides, 10, seed=4)
    blocks = [counts[start : start + 2_000] for start in range(0, n_cells, 2_000)]
    samples = np.zeros(n_cells, dtype=np.int64)
    planned = histogram_bytes(
        1, n_guides, n_cells, n_distinct=_distinct_counts(counts, samples)
    )
    message, needed, peak = _refusal(
        lambda: guide_count_histograms(
            blocks, samples, n_samples=1, n_guides=n_guides, memory_bytes=planned // 2
        )
    )
    assert "they may hold" in message
    # The message names what the next step needs, more than the limit and at
    # most what the whole count needs.
    assert planned // 2 < needed <= planned
    assert peak <= planned // 2


def test_histograms_raise_before_assembling_more_than_their_memory() -> None:
    # Ambient counts fill most dense bins of 200 guides and none is larger,
    # so that only assembling the histograms needs more than their least.
    counts = np.random.default_rng(6).poisson(6.0, size=(2_000, 200))
    counts = counts.astype(np.uint16)
    samples = np.zeros(2_000, dtype=np.int64)
    # The first call loads the kernels for uint16 counts.
    guide_count_histograms([counts[:2]], samples[:2], n_samples=1, n_guides=200)
    n_distinct = _distinct_counts(counts, samples)
    least = histogram_bytes(1, 200, 2_000, n_distinct=0)
    message, needed, peak = _refusal(
        lambda: guide_count_histograms(
            [counts], samples, n_samples=1, n_guides=200, memory_bytes=least
        )
    )
    # The assembly of every distinct count is refused.
    assert f"for {n_distinct} distinct counts" in message
    assert least < needed <= histogram_bytes(1, 200, 2_000, n_distinct=n_distinct)
    assert peak <= least


def test_histograms_refuse_memory_below_their_least_before_reading() -> None:
    def unread():
        raise AssertionError("A block was read")
        yield

    samples = np.asarray([0, 1, 0, 1], dtype=np.int64)
    least = histogram_bytes(2, 3, 4, n_distinct=0)
    with pytest.raises(MemoryError, match=f"need about {least} bytes for 0 "):
        guide_count_histograms(
            unread(), samples, n_samples=2, n_guides=3, memory_bytes=least - 1
        )
    # Codes that are not int64 are held as an int64 copy, 8 bytes per cell.
    with pytest.raises(MemoryError, match=f"need about {least + 32} bytes for 0 "):
        guide_count_histograms(
            unread(),
            samples.astype(np.uint8),
            n_samples=2,
            n_guides=3,
            memory_bytes=least,
        )
    counts = np.zeros((4, 3), dtype=np.uint8)
    found = guide_count_histograms(
        [counts], samples, n_samples=2, n_guides=3, memory_bytes=least
    )
    assert found.zero_cells().tolist() == [[2, 2, 2], [2, 2, 2]]
