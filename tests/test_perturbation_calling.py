from fractions import Fraction

import numpy as np
import pytest

from cytearc.perturbation import calling
from cytearc.perturbation.calling import threshold_calls, top_guide_calls
from cytearc.perturbation.guide_thresholds import (
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_MIN_UMI,
    call_settings,
)
from cytearc.perturbation.guide_calls import GuideCountSummary, summarize_guide_counts
from cytearc.perturbation.histograms import MAX_COUNT
from tests.perturbation_helpers import screen_counts, use_python_kernels

THRESHOLDS = np.asarray([[5, 0, 8, 1, 30], [3, 3, 0, 12, 2]], dtype=np.uint32)


def _expected(counts: np.ndarray, samples: np.ndarray, thresholds: np.ndarray):
    """Calls by a direct comparison of every count with its threshold."""
    table = np.where(
        thresholds > 0, thresholds.astype(np.int64), np.iinfo(np.int64).max
    )
    rows, guides = np.nonzero(counts.astype(np.int64) >= table[samples])
    return rows, guides, counts[rows, guides]


def _check(calls, summary, counts: np.ndarray, samples: np.ndarray) -> None:
    rows, guides, umis = _expected(counts, samples, THRESHOLDS)
    np.testing.assert_array_equal(calls.cells, rows)
    np.testing.assert_array_equal(calls.codes, guides)
    np.testing.assert_array_equal(calls.umis, umis)
    assert calls.cells.dtype == np.int64
    assert calls.codes.dtype == np.int64
    assert calls.umis.dtype == np.uint32
    reference = summarize_guide_counts([counts], len(counts), counts.shape[1])
    for name in ("top_code", "top_umi", "total_umi"):
        np.testing.assert_array_equal(getattr(summary, name), getattr(reference, name))
        assert getattr(summary, name).dtype == getattr(reference, name).dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_threshold_calls_and_summary_equal_a_direct_comparison(dtype) -> None:
    counts = np.minimum(screen_counts(500, 5, seed=20, ambient=1.5), 255).astype(dtype)
    samples = np.random.default_rng(21).integers(0, 2, size=500)
    calls, summary = threshold_calls(
        [counts[:200], counts[200:200], counts[200:]], samples, THRESHOLDS
    )
    _check(calls, summary, counts, samples)
    assert len(calls.cells) > 500


def test_threshold_calls_hold_each_blocks_calls_exactly(monkeypatch) -> None:
    # Crowded rows call most guides; each block's calls are counted, then
    # written into arrays of exactly that length.
    counts = screen_counts(300, 5, seed=22, ambient=4.0)
    samples = np.random.default_rng(23).integers(0, 2, size=300)
    lengths: list[int] = []
    call_block = calling._call_block

    def recording(block, *arguments):
        found = call_block(block, *arguments)
        lengths.append(len(arguments[-1]))
        return found

    monkeypatch.setattr(calling, "_call_block", recording)
    calls, summary = threshold_calls([counts[:150], counts[150:]], samples, THRESHOLDS)
    _check(calls, summary, counts, samples)
    rows, _, _ = _expected(counts, samples, THRESHOLDS)
    in_blocks = [int(np.count_nonzero(rows < 150)), int(np.count_nonzero(rows >= 150))]
    assert lengths == [0, in_blocks[0], 0, in_blocks[1]]


def test_python_kernel_calls_like_the_compiled_one(monkeypatch) -> None:
    counts = screen_counts(200, 5, seed=24, ambient=2.0)
    samples = np.random.default_rng(25).integers(0, 2, size=200)
    use_python_kernels(monkeypatch, calling)
    calls, summary = threshold_calls([counts[:120], counts[120:]], samples, THRESHOLDS)
    _check(calls, summary, counts, samples)
    wide = np.full((1, 5), MAX_COUNT + 1, dtype=np.uint64)
    with pytest.raises(ValueError, match="at most"):
        threshold_calls([wide], [0], THRESHOLDS)


def test_cells_without_guide_umis_have_no_top_guide() -> None:
    counts = np.asarray([[0, 0, 0, 0, 0], [0, 7, 7, 0, 0]], dtype=np.uint16)
    calls, summary = threshold_calls([counts], [0, 1], THRESHOLDS)
    assert summary.top_code.tolist() == [-1, 1]
    assert summary.top_umi.tolist() == [0, 7]
    assert summary.total_umi.tolist() == [0, 14]
    assert calls.cells.tolist() == [1]
    assert calls.codes.tolist() == [1]


@pytest.mark.parametrize("value", [MAX_COUNT + 1, 2**63 + 5])
def test_threshold_calls_refuse_counts_beyond_uint32(value) -> None:
    counts = np.zeros((2, 5), dtype=np.uint64)
    counts[1, 3] = value
    with pytest.raises(ValueError, match=f"at most {MAX_COUNT}"):
        threshold_calls([counts], [0, 1], THRESHOLDS)


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        (np.asarray([1, 2]), "samples x guides"),
        (np.zeros((1, 0), dtype=np.int64), "samples x guides"),
        (np.asarray([[1.0, 2.0]]), "must be integers"),
        (np.asarray([[-1, 2]]), "lie in"),
        (np.asarray([[MAX_COUNT + 1, 2]], dtype=np.int64), "lie in"),
    ],
)
def test_threshold_calls_refuse_thresholds_they_cannot_use(thresholds, message) -> None:
    with pytest.raises(ValueError, match=message):
        threshold_calls([np.zeros((1, 2), dtype=np.uint8)], [0], thresholds)


def test_threshold_calls_check_samples_and_blocks() -> None:
    block = np.zeros((2, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="lie in \\[0, 2\\)"):
        threshold_calls([block], [0, 2], THRESHOLDS)
    with pytest.raises(ValueError, match="5 columns"):
        threshold_calls([block[:, :4]], [0, 1], THRESHOLDS)
    with pytest.raises(ValueError, match="hold 2 rows, but there are 3 cells"):
        threshold_calls([block], [0, 1, 1], THRESHOLDS)


def _summary() -> GuideCountSummary:
    # Shares: 4/5 (exactly 0.8), 5/6, 9/10, none, 2/2 with 2 UMIs, 3/6.
    return GuideCountSummary(
        top_code=np.asarray([0, 1, 2, -1, 3, 1], dtype=np.int32),
        top_umi=np.asarray([4, 5, 9, 0, 2, 3], dtype=np.uint32),
        total_umi=np.asarray([5, 6, 10, 0, 2, 6], dtype=np.uint64),
    )


@pytest.mark.parametrize(
    ("min_umi", "fraction", "cells"),
    [
        (5, 0.8, [1, 2]),  # 4 of 5 UMIs does not exceed 0.8
        (4, 0.79, [0, 1, 2]),
        (1, 0.0, [0, 1, 2, 4, 5]),  # every cell with guide UMIs
        (2, 0.9, [4]),  # 9 of 10 does not exceed 0.9
        (10, 0.0, [2]),  # 9 of 10: the cell's 10 guide UMIs reach the floor
        (11, 0.0, []),
    ],
)
def test_top_guide_calls_compare_the_share_strictly(min_umi, fraction, cells) -> None:
    summary = _summary()
    calls = top_guide_calls(summary, min_umi=min_umi, fraction_threshold=fraction)
    assert calls.cells.tolist() == cells
    assert calls.codes.tolist() == summary.top_code[cells].tolist()
    assert calls.umis.tolist() == summary.top_umi[cells].tolist()
    assert calls.cells.dtype == np.int64
    assert calls.codes.dtype == np.int64
    assert calls.umis.dtype == np.uint32


def test_top_guide_calls_give_ties_to_the_first_guide() -> None:
    counts = np.asarray([[0, 3, 3, 1]], dtype=np.uint16)
    summary = summarize_guide_counts([counts], 1, 4)
    calls = top_guide_calls(summary, min_umi=np.int64(3), fraction_threshold=0.3)
    assert calls.codes.tolist() == [1]


def _every_pair(largest: int) -> GuideCountSummary:
    """One cell for each top-guide and total UMI count up to ``largest``.

    The first cell holds no guide UMI; every other cell's top guide holds from
    1 of its UMIs to all of them.
    """
    pairs = [(0, 0)] + [
        (top, total) for total in range(1, largest + 1) for top in range(1, total + 1)
    ]
    top, total = np.asarray(pairs).T
    return GuideCountSummary(
        top_code=np.where(total > 0, np.arange(len(pairs)) % 7, -1).astype(np.int32),
        top_umi=top.astype(np.uint32),
        total_umi=total.astype(np.uint64),
    )


def _kept(summary: GuideCountSummary, fraction: Fraction, *, floor_of) -> np.ndarray:
    """Return the cells past a floor whose top guide holds more than ``fraction``.

    ``floor_of(top, total)`` says whether a cell reaches the floor, and the
    share is compared exactly, as a ratio of integers.
    """
    return np.asarray(
        [
            floor_of(top, total) and Fraction(top, total) > fraction
            for top, total in zip(
                summary.top_umi.tolist(), summary.total_umi.tolist(), strict=True
            )
        ]
    )


def _book_rule(summary: GuideCountSummary, min_umi: int, fraction: Fraction):
    """The cells that the SCEPTRE book's maximum rule finds with one guide.

    The book flags a cell with fewer guide UMIs than its floor as holding no
    guide, and a cell whose top guide holds at most its fraction of them as
    holding several.
    """
    return _kept(summary, fraction, floor_of=lambda top, total: total >= min_umi)


def _top_floor_rule(summary: GuideCountSummary, min_umi: int, fraction: Fraction):
    """The cells that a floor on the top guide's UMIs keeps at the same share."""
    return _kept(summary, fraction, floor_of=lambda top, total: top >= min_umi)


@pytest.mark.parametrize("min_umi", [1, 2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize("fraction", ["0", "0.5", "0.75", "0.8", "0.9"])
def test_top_guide_calls_bound_the_cells_guide_umis(min_umi, fraction) -> None:
    summary = _every_pair(40)
    exact = Fraction(fraction)

    calls = top_guide_calls(summary, min_umi=min_umi, fraction_threshold=float(exact))

    book = _book_rule(summary, min_umi, exact)
    np.testing.assert_array_equal(calls.cells, np.flatnonzero(book))
    np.testing.assert_array_equal(calls.codes, summary.top_code[book])
    np.testing.assert_array_equal(calls.umis, summary.top_umi[book])
    # A floor on the top guide's UMIs keeps some of these cells, and all of
    # them exactly when the fraction is at least 1 - 1 / min_umi.
    on_top = _top_floor_rule(summary, min_umi, exact)
    assert not (on_top & ~book).any()
    assert (on_top == book).all() == (exact >= 1 - Fraction(1, min_umi))


def test_at_the_defaults_both_floors_call_the_same_cells() -> None:
    # A top guide that holds more than 0.8 of at least 5 UMIs holds 5 itself.
    summary = _every_pair(60)
    min_umi = DEFAULT_MIN_UMI["top_guide"]
    fraction = Fraction(str(DEFAULT_FRACTION_THRESHOLD))
    assert (min_umi, fraction) == (5, Fraction(4, 5))

    calls = top_guide_calls(
        summary, min_umi=min_umi, fraction_threshold=DEFAULT_FRACTION_THRESHOLD
    )

    expected = _top_floor_rule(summary, min_umi, fraction)
    np.testing.assert_array_equal(calls.cells, np.flatnonzero(expected))
    np.testing.assert_array_equal(expected, _book_rule(summary, min_umi, fraction))


def test_a_cell_with_4_of_6_guide_umis_is_called_at_half() -> None:
    summary = GuideCountSummary(
        top_code=np.asarray([2], dtype=np.int32),
        top_umi=np.asarray([4], dtype=np.uint32),
        total_umi=np.asarray([6], dtype=np.uint64),
    )

    half = top_guide_calls(summary, min_umi=5, fraction_threshold=0.5)

    # The cell's 6 guide UMIs reach the floor of 5, though its top guide's 4
    # do not.
    assert (half.cells.tolist(), half.codes.tolist(), half.umis.tolist()) == (
        [0],
        [2],
        [4],
    )
    default = top_guide_calls(summary, min_umi=5, fraction_threshold=0.8)
    assert default.cells.tolist() == []


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"min_umi": True}, TypeError, "min_umi must be an integer"),
        ({"min_umi": 0}, ValueError, "min_umi must be at least 1"),
        ({"min_umi": 1.5}, TypeError, "min_umi must be an integer"),
        ({"fraction_threshold": 1}, ValueError, "at least 0 and below 1"),
        ({"fraction_threshold": -0.1}, ValueError, "at least 0 and below 1"),
        ({"fraction_threshold": float("nan")}, ValueError, "must be finite"),
        ({"fraction_threshold": True}, TypeError, "must be a real number"),
        ({"fraction_threshold": "0.8"}, TypeError, "must be a real number"),
    ],
)
def test_top_guide_settings_that_calls_cannot_use_are_refused(
    arguments, error, message
) -> None:
    # The settings are checked once, before any count is read; the calls
    # take them as checked.
    settings = {"min_umi": 5, "fraction_threshold": 0.8} | arguments
    with pytest.raises(error, match=message):
        call_settings(
            "top_guide", settings["min_umi"], settings["fraction_threshold"], None
        )
