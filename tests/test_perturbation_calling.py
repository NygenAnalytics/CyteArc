import numpy as np
import pytest

from cytearc.perturbation import calling
from cytearc.perturbation.calling import threshold_calls, top_guide_calls
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


def test_threshold_calls_take_a_second_pass_for_crowded_rows(monkeypatch) -> None:
    counts = screen_counts(300, 5, seed=22, ambient=4.0)
    samples = np.random.default_rng(23).integers(0, 2, size=300)
    monkeypatch.setattr(calling, "_CALLS_PER_ROW", 0)
    calls, summary = threshold_calls([counts[:150], counts[150:]], samples, THRESHOLDS)
    _check(calls, summary, counts, samples)


def test_python_kernel_calls_like_the_compiled_one(monkeypatch) -> None:
    counts = screen_counts(200, 5, seed=24, ambient=2.0)
    samples = np.random.default_rng(25).integers(0, 2, size=200)
    use_python_kernels(monkeypatch, calling)
    monkeypatch.setattr(calling, "_CALLS_PER_ROW", 0)
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
        (10, 0.0, []),
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


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_umi": True}, "min_umi must be a positive integer"),
        ({"min_umi": 0}, "min_umi must be a positive integer"),
        ({"min_umi": 1.5}, "min_umi must be a positive integer"),
        ({"fraction_threshold": 1}, "at least 0 and below 1"),
        ({"fraction_threshold": -0.1}, "at least 0 and below 1"),
        ({"fraction_threshold": float("nan")}, "at least 0 and below 1"),
        ({"fraction_threshold": True}, "at least 0 and below 1"),
        ({"fraction_threshold": "0.8"}, "at least 0 and below 1"),
    ],
)
def test_top_guide_calls_refuse_settings_they_cannot_use(arguments, message) -> None:
    settings = {"min_umi": 5, "fraction_threshold": 0.8} | arguments
    with pytest.raises(ValueError, match=message):
        top_guide_calls(_summary(), **settings)
