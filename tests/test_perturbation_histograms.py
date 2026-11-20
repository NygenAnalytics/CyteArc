from collections.abc import Callable

import numpy as np
import pytest

from cytearc.perturbation import histograms
from cytearc.perturbation.histograms import (
    DENSE_COUNTS,
    MAX_COUNT,
    HistogramMemoryError,
    checked_sample_codes,
    guide_count_histograms,
    histogram_bytes,
)
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.memory_helpers import named_budget
from tests.perturbation_helpers import histogram_of, screen_counts, use_python_kernels


def _ledger(budget: int) -> MemoryLedger:
    """A ledger of ``budget`` bytes that holds nothing, which the histograms charge."""
    return MemoryLedger(int(budget), "Counting guide UMIs")


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
            found_values, found_cells = histogram_of(found, sample, guide)
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
    assert histogram_of(found, 0, 0)[0].tolist() == [2]
    assert histogram_of(found, 0, 0)[1].tolist() == [2]
    assert histogram_of(found, 1, 0)[0].tolist() == []
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
    ],
)
def test_histograms_refuse_inputs_they_cannot_count(arguments, error, message) -> None:
    call = {
        "blocks": [np.zeros((2, 2), dtype=np.uint8)],
        "sample_codes": [0, 0],
        "n_samples": 1,
        "n_guides": 2,
        "ledger": None,
    }
    call.update(arguments)
    with pytest.raises(error, match=message):
        guide_count_histograms(
            call["blocks"],
            call["sample_codes"],
            n_samples=call["n_samples"],
            n_guides=call["n_guides"],
            ledger=call["ledger"],
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
def test_histograms_within_their_planned_bytes_find_every_distinct_count(
    monkeypatch, per_cell, n_samples, rows, most_keys
) -> None:
    if most_keys is not None:
        monkeypatch.setattr(histograms, "_MOST_KEYS", most_keys)
    n_cells, n_guides = 20_000, 200
    counts = _carried_counts(n_cells, n_guides, per_cell, seed=3)
    blocks = [counts[start : start + rows] for start in range(0, n_cells, rows)]
    # The sample codes, like the blocks, belong to the caller.
    samples = np.arange(n_cells) % n_samples
    n_distinct = _distinct_counts(counts, samples)
    planned = histogram_bytes(n_samples, n_guides, n_cells, n_distinct=n_distinct)
    found = guide_count_histograms(
        blocks, samples, n_samples=n_samples, n_guides=n_guides, ledger=_ledger(planned)
    )
    assert len(found.counts) == n_distinct


def test_few_cells_merged_after_every_three_find_every_distinct_count(
    monkeypatch,
) -> None:
    # 64 cells, whose larger counts merge after every three.
    monkeypatch.setattr(histograms, "_MOST_KEYS", 3)
    counts = _carried_counts(64, 3, 1, seed=5)
    samples = np.zeros(64, dtype=np.int64)
    blocks = [counts[start : start + 7] for start in range(0, 64, 7)]
    n_distinct = _distinct_counts(counts, samples)
    found = guide_count_histograms(blocks, samples, n_samples=1, n_guides=3)
    assert len(found.counts) == n_distinct


def _refusal(count: Callable[[], object]) -> tuple[str, int]:
    """Return a refused count's message and the bytes it names."""
    with pytest.raises(MemoryError) as refused:
        count()
    return str(refused.value), named_budget(refused.value)


def test_histograms_raise_before_they_outgrow_their_memory() -> None:
    # Cells that carry ten guides each, and half the memory that their
    # histograms need.
    n_cells, n_guides = 20_000, 200
    counts = _carried_counts(n_cells, n_guides, 10, seed=4)
    blocks = [counts[start : start + 2_000] for start in range(0, n_cells, 2_000)]
    samples = np.zeros(n_cells, dtype=np.int64)
    planned = histogram_bytes(
        1, n_guides, n_cells, n_distinct=_distinct_counts(counts, samples)
    )
    message, needed = _refusal(
        lambda: guide_count_histograms(
            blocks,
            samples,
            n_samples=1,
            n_guides=n_guides,
            ledger=_ledger(planned // 2),
        )
    )
    assert "rows of counts remain to be read" in message
    # The message names what the next step needs, more than the limit and at
    # most what the whole count needs.
    assert planned // 2 < needed <= planned


def test_histograms_raise_before_assembling_more_than_their_memory() -> None:
    # Ambient counts fill most dense bins of 200 guides and none is larger,
    # so that only assembling the histograms needs more than their least.
    counts = np.random.default_rng(6).poisson(6.0, size=(2_000, 200))
    counts = counts.astype(np.uint16)
    samples = np.zeros(2_000, dtype=np.int64)
    n_distinct = _distinct_counts(counts, samples)
    least = histogram_bytes(1, 200, 2_000, n_distinct=0)
    message, needed = _refusal(
        lambda: guide_count_histograms(
            [counts], samples, n_samples=1, n_guides=200, ledger=_ledger(least)
        )
    )
    # The assembly of every distinct count is refused.
    assert f"histograms of {n_distinct} distinct counts," in message
    assert least < needed <= histogram_bytes(1, 200, 2_000, n_distinct=n_distinct)


def test_histograms_refuse_memory_below_their_least_before_reading() -> None:
    def unread():
        raise AssertionError("A block was read")
        yield

    samples = np.asarray([0, 1, 0, 1], dtype=np.int64)
    least = histogram_bytes(2, 3, 4, n_distinct=0)
    with pytest.raises(MemoryError) as refused:
        guide_count_histograms(
            unread(), samples, n_samples=2, n_guides=3, ledger=_ledger(least - 1)
        )
    assert named_budget(refused.value) == least
    assert "for the guide count histograms of 0 distinct counts, while 4 of 4" in str(
        refused.value
    )
    # Before any row is read, there is no rate to estimate every row from.
    assert refused.value.every_row_bytes is None
    # Codes that are not int64 are held as an int64 copy, 8 bytes per cell.
    with pytest.raises(MemoryError) as refused:
        guide_count_histograms(
            unread(),
            samples.astype(np.uint8),
            n_samples=2,
            n_guides=3,
            ledger=_ledger(least),
        )
    assert named_budget(refused.value) == least + 32
    counts = np.zeros((4, 3), dtype=np.uint8)
    found = guide_count_histograms(
        [counts], samples, n_samples=2, n_guides=3, ledger=_ledger(least)
    )
    assert found.zero_cells().tolist() == [[2, 2, 2], [2, 2, 2]]


def test_a_refusal_says_how_many_rows_remain_to_be_read() -> None:
    n_cells, n_guides = 6_000, 100
    counts = _carried_counts(n_cells, n_guides, 10, seed=5)
    samples = np.zeros(n_cells, dtype=np.int64)
    least = histogram_bytes(1, n_guides, n_cells, n_distinct=0)
    with pytest.raises(HistogramMemoryError) as refused:
        guide_count_histograms(
            [counts], samples, n_samples=1, n_guides=n_guides, ledger=_ledger(least + 1)
        )
    error = refused.value
    # The buffer of larger counts filled within the block, so rows remain.
    assert error.counts_remain
    assert 0 < error.rows_read < error.n_rows == n_cells
    assert f"{n_cells - error.rows_read} of {n_cells} rows of counts remain" in str(
        error
    )
    assert error.needed_bytes > least + 1
    # It refuses as every memory budget refusal does, naming the histograms
    # and the budget of the ledger that they charge.
    assert isinstance(error, MemoryBudgetError)
    assert error.item.startswith("the guide count histograms of ")
    assert error.budget_bytes == least + 1
    assert named_budget(error) == error.needed_bytes


def test_a_refusal_after_every_row_names_memory_that_finishes() -> None:
    n_cells, n_guides = 3_000, 100
    counts = _carried_counts(n_cells, n_guides, 5, seed=9)
    samples = np.zeros(n_cells, dtype=np.int64)
    guide_count_histograms([counts[:2]], samples[:2], n_samples=1, n_guides=n_guides)
    least = histogram_bytes(1, n_guides, n_cells, n_distinct=0)
    full = histogram_bytes(
        1, n_guides, n_cells, n_distinct=_distinct_counts(counts, samples)
    )
    finished = 0
    for limit in np.linspace(least, full, 40).astype(int):
        try:
            guide_count_histograms(
                [counts], samples, n_samples=1, n_guides=n_guides, ledger=_ledger(limit)
            )
        except HistogramMemoryError as error:
            if error.counts_remain:
                continue
            # With every row read, nothing remains to estimate, and the last
            # merge names what the assembly after it needs too, so the memory
            # it names finishes the count.
            assert error.every_row_bytes is None
            found = guide_count_histograms(
                [counts],
                samples,
                n_samples=1,
                n_guides=n_guides,
                ledger=_ledger(error.needed_bytes),
            )
            assert int(found.cells.sum()) == int(np.count_nonzero(counts))
            finished += 1
    assert finished


def _counts_of_their_own(n_cells: int, n_guides: int, per_cell: int) -> np.ndarray:
    """``per_cell`` counts above the dense range in each cell, each a count that no other cell holds.

    Each cell adds as many distinct counts, so the distinct counts grow
    exactly with the rows read.
    """
    counts = np.zeros((n_cells, n_guides), dtype=np.uint32)
    cells = np.arange(n_cells)
    for carried in range(per_cell):
        guides = (cells * per_cell + carried) % n_guides
        counts[cells, guides] = DENSE_COUNTS + 1 + cells * per_cell + carried
    return counts


def _dense_counts_first() -> tuple[np.ndarray, np.ndarray, int]:
    """Four samples whose first 32 cells each fill every dense count of 40 guides.

    Each of 2,000 cells also holds a count of its own of each of 10 other
    guides, so the larger counts grow exactly with the rows read while the
    dense ones stop after 128 rows.
    """
    n_cells, n_samples = 2_000, 4
    counts = np.hstack(
        [
            _counts_of_their_own(n_cells, 10, 10),
            np.zeros((n_cells, 40), dtype=np.uint32),
        ]
    )
    first = np.arange(DENSE_COUNTS * n_samples)
    counts[first, 10:] = (first // n_samples + 1)[:, None]
    return counts, np.arange(n_cells) % n_samples, n_samples


def _dense_counts_grow() -> tuple[np.ndarray, np.ndarray, int]:
    """Fifty samples of 100 guides whose dense counts outweigh the buffer.

    Each of 10,000 cells holds counts drawn from a million values above the
    dense range in 30 guides and dense counts in 20 others, so that 5,000
    histograms fill their dense counts as rows are read.
    """
    n_cells, n_guides, n_samples = 10_000, 100, 50
    rng = np.random.default_rng(12)
    guides = rng.random((n_cells, n_guides)).argsort(axis=1)
    rows = np.arange(n_cells)[:, None]
    counts = np.zeros((n_cells, n_guides), dtype=np.uint32)
    counts[rows, guides[:, :30]] = (
        DENSE_COUNTS + 1 + rng.integers(0, 10**6, (n_cells, 30))
    )
    counts[rows, guides[:, 30:50]] = rng.integers(1, DENSE_COUNTS + 1, (n_cells, 20))
    return counts, np.arange(n_cells) % n_samples, n_samples


def _one_sample(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    return counts, np.zeros(len(counts), dtype=np.int64), 1


# Screens whose histograms stop at several merges while rows remain.
_STOPPING_SCREENS = {
    # Every count is one of its own, so the distinct counts grow with the rows.
    "counts_grow_with_the_rows": lambda: _one_sample(
        _counts_of_their_own(2_000, 20, 20)
    ),
    # The buffer merges after every 33 cells, and the last merge holds the 20
    # counts of one cell, so that merge, not the assembly, needs the most,
    # and the estimate of every row has no cell to spare.
    "one_row_in_the_last_merge": lambda: _one_sample(_counts_of_their_own(298, 20, 20)),
    # Counts drawn from 100 values repeat, so the distinct counts stop growing.
    "counts_repeat": lambda: _one_sample(
        (
            DENSE_COUNTS
            + 1
            + np.random.default_rng(1).integers(0, 100, size=(3_000, 20))
        ).astype(np.uint32)
    ),
    "dense_counts_first": _dense_counts_first,
    "dense_counts_grow": _dense_counts_grow,
}


def _stops(
    counts: np.ndarray, samples: np.ndarray, n_samples: int
) -> list[HistogramMemoryError]:
    """Return the refusals of the histograms at each merge that needs more than those before it.

    From the histograms' least, each budget is what the refusal before it
    needed, which passes that merge and stops at the next that needs more.
    """
    n_guides = counts.shape[1]
    budget = histogram_bytes(n_samples, n_guides, len(counts), n_distinct=0)
    refusals = []
    while True:
        try:
            guide_count_histograms(
                [counts],
                samples,
                n_samples=n_samples,
                n_guides=n_guides,
                ledger=_ledger(budget),
            )
        except HistogramMemoryError as error:
            assert error.needed_bytes > budget
            refusals.append(error)
            budget = error.needed_bytes
        else:
            return [error for error in refusals if error.counts_remain]


@pytest.mark.parametrize("screen", list(_STOPPING_SCREENS))
def test_a_refusal_while_rows_remain_names_memory_that_counts_every_row(
    screen,
) -> None:
    counts, samples, n_samples = _STOPPING_SCREENS[screen]()
    n_guides = counts.shape[1]
    expected = guide_count_histograms(
        [counts], samples, n_samples=n_samples, n_guides=n_guides
    )
    stops = _stops(counts, samples, n_samples)
    assert len(stops) >= 3
    for error in stops:
        # A merge stopped the histograms after some rows were read.
        assert 0 < error.rows_read < error.n_rows
        assert error.every_row_bytes is not None
        assert error.every_row_bytes >= error.needed_bytes
        # The estimate gets past every later merge and the assembly.
        found = guide_count_histograms(
            [counts],
            samples,
            n_samples=n_samples,
            n_guides=n_guides,
            ledger=_ledger(error.every_row_bytes),
        )
        for name in ("offsets", "counts", "cells", "n_cells"):
            np.testing.assert_array_equal(getattr(found, name), getattr(expected, name))


@pytest.mark.parametrize(
    "screen",
    ["counts_grow_with_the_rows", "one_row_in_the_last_merge", "dense_counts_first"],
)
def test_counts_that_grow_with_the_rows_are_estimated_within_their_plan(
    screen,
) -> None:
    # Where the larger counts grow exactly with the rows read, the estimate
    # of every row holds no more than the histograms of the screen's distinct
    # counts, as histogram_bytes plans them; the dense counts of a histogram
    # count at most once each, however early they fill.
    counts, samples, n_samples = _STOPPING_SCREENS[screen]()
    planned = histogram_bytes(
        n_samples,
        counts.shape[1],
        len(counts),
        n_distinct=_distinct_counts(counts, samples),
    )
    stops = _stops(counts, samples, n_samples)
    assert stops
    for error in stops:
        assert error.every_row_bytes is not None
        assert error.every_row_bytes <= planned


@pytest.mark.parametrize(
    ("filled", "n_larger", "rows_read", "buffer_bytes", "expected"),
    [
        # With 3 of 4 rows read, 10 larger counts become 14 and 5 dense ones
        # 7, rounded up. A later merge holds 1,000 + 800 + 32 x 14 = 2,248
        # bytes, more than the assembly's 1,000 + 16 x 14 for the merged
        # counts + 16 for the offsets + 16 for the sample's cells + 16 x 21
        # for the histograms' counts and cells = 1,592.
        (5, 10, 3, 800, 2_248),
        # With 1 of 4 rows read, 30 dense counts would become 120, but one
        # histogram holds at most 32, and 2 larger counts become 8. The
        # assembly's 1,000 + 16 x 8 + 16 + 16 + 16 x 40 = 1,800 bytes exceed
        # the 1,000 + 32 x 8 = 1,256 of a later merge without a buffer.
        (30, 2, 1, 0, 1_800),
    ],
)
def test_the_estimate_of_every_row_scales_counts_up_to_what_a_histogram_holds(
    filled, n_larger, rows_read, buffer_bytes, expected
) -> None:
    dense = np.zeros((1, 1, DENSE_COUNTS), dtype=np.uint32)
    dense[0, 0, :filled] = 1
    found = histograms._every_row_bytes(
        1_000, buffer_bytes, dense, n_larger, rows_read=rows_read, n_rows=4
    )
    assert found == expected


@pytest.mark.parametrize("compiled", [True, False])
def test_the_calls_of_thresholds_count_each_cell_at_or_above_its_guides_threshold(
    monkeypatch, compiled
):
    if not compiled:
        use_python_kernels(monkeypatch, histograms)
    counts = screen_counts(300, 6, seed=8)
    samples = np.arange(300) % 3
    found = guide_count_histograms([counts], samples, n_samples=3, n_guides=6)
    thresholds = np.array(
        [[1, 40, 0, 7, 200, 65], [3, 0, 0, 2, 64, 65], [0, 0, 0, 0, 0, 0]]
    )

    # A guide whose threshold is 0 calls nothing in that sample.
    expected = sum(
        int(((counts[samples == s, g] >= t) & (t > 0)).sum())
        for s in range(3)
        for g, t in enumerate(thresholds[s])
    )
    assert found.n_calls(thresholds) == expected
    assert found.n_calls(np.zeros((3, 6), dtype=np.int64)) == 0
    assert found.n_calls(np.ones((3, 6), dtype=np.int64)) == int((counts > 0).sum())
