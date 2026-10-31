"""Kernels that call guides in cells from thresholds or from each cell's top guide.

A threshold method calls a guide in a cell when the cell's UMIs of the guide
reach the guide's threshold in the cell's sample, so a cell can have any
number of calls, and its calls never depend on its other guides. The
``top_guide`` method calls at most one guide per cell: the cell's top guide,
when the cell's guide UMIs reach a floor and the top guide's share of them
exceeds a set fraction.

The kernels read count blocks that a caller streams and never touch storage.
Their calls feed :func:`~cytearc.perturbation.guide_calls.guide_call_arrays`.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from cytearc.perturbation.guide_calls import GuideCountSummary, checked_count_blocks
from cytearc.perturbation.histograms import MAX_COUNT, checked_sample_codes
from cytearc.utils.numba import njit

__all__ = ["CalledGuides", "threshold_calls", "top_guide_calls"]

# Each block reserves room for this many calls per row before a second pass is
# needed; most screens call one or two guides per cell.
_CALLS_PER_ROW = 2


@dataclass(frozen=True, slots=True)
class CalledGuides:
    """Guides called in cells, one entry per guide called in a cell.

    Attributes:
        cells: The position of the cell of each call.
        codes: The position of the called guide in the assay.
        umis: The UMIs of the called guide in the cell.
    """

    cells: np.ndarray
    codes: np.ndarray
    umis: np.ndarray


@njit(cache=True, nogil=True)
def _call_block(
    block: np.ndarray,
    sample_codes: np.ndarray,
    thresholds: np.ndarray,
    top_code: np.ndarray,
    top_umi: np.ndarray,
    total_umi: np.ndarray,
    call_rows: np.ndarray,
    call_codes: np.ndarray,
    call_umis: np.ndarray,
) -> int:
    """Summarise and call one row block.

    Writes each row's top guide, its UMIs and the row's total, and writes calls
    while there is room. Returns the number of calls, which may exceed the
    room, or -1 when a count does not fit in uint32.
    """
    room = call_rows.shape[0]
    n_calls = 0
    for row in range(block.shape[0]):
        sample = sample_codes[row]
        total = 0
        best = 0
        best_code = -1
        for guide in range(block.shape[1]):
            count = int(block[row, guide])
            if count == 0:
                continue
            # A uint64 count above the int64 range wraps to a negative value.
            if count < 0 or count > MAX_COUNT:
                return -1
            total += count
            # A strict comparison keeps the first maximum, the lowest position.
            if count > best:
                best = count
                best_code = guide
            threshold = thresholds[sample, guide]
            if threshold > 0 and count >= threshold:
                if n_calls < room:
                    call_rows[n_calls] = row
                    call_codes[n_calls] = guide
                    call_umis[n_calls] = count
                n_calls += 1
        top_code[row] = best_code
        top_umi[row] = best
        total_umi[row] = total
    return n_calls


def threshold_calls(
    blocks: Iterable[np.ndarray],
    sample_codes: object,
    thresholds: object,
) -> tuple[CalledGuides, GuideCountSummary]:
    """Call every guide whose UMIs reach its threshold in the cell's sample.

    The same pass also finds each cell's top guide and total guide UMIs, as
    :func:`~cytearc.perturbation.guide_calls.summarize_guide_counts` does.

    Args:
        blocks: Consecutive row blocks of the cells x guides counts, which
            together hold one row per entry of ``sample_codes``.
        sample_codes: The sample position of each cell.
        thresholds: The smallest called UMI count of each guide in each
            sample, samples x guides, with 0 for a guide that calls nothing.

    Returns:
        The calls, ordered by cell and then by guide position, and the summary
        of every cell's counts.

    Raises:
        TypeError: If a block holds counts that are not unsigned integers.
        ValueError: If the thresholds are not a samples x guides table of
            integers from 0 to the uint32 maximum, a sample code is out of
            range, a block has the wrong width, the blocks hold more or fewer
            rows than there are cells, or a count exceeds the uint32 range.
    """
    table = np.asarray(thresholds)
    if table.ndim != 2 or 0 in table.shape:
        raise ValueError("Thresholds must be a samples x guides table")
    if table.dtype.kind not in "iu":
        raise ValueError(f"Thresholds must be integers, got {table.dtype}")
    if table.min() < 0 or table.max() > MAX_COUNT:
        raise ValueError(f"Thresholds must lie in [0, {MAX_COUNT}]")
    table = table.astype(np.int64, copy=False)
    n_samples, n_guides = table.shape
    codes = checked_sample_codes(sample_codes, n_samples)
    n_cells = len(codes)
    top_code = np.full(n_cells, -1, dtype=np.int32)
    top_umi = np.zeros(n_cells, dtype=np.uint32)
    total_umi = np.zeros(n_cells, dtype=np.uint64)
    cells: list[np.ndarray] = [np.zeros(0, dtype=np.int64)]
    guides: list[np.ndarray] = [np.zeros(0, dtype=np.int64)]
    umis: list[np.ndarray] = [np.zeros(0, dtype=np.uint32)]
    for start, block in checked_count_blocks(
        blocks, n_cells=n_cells, n_guides=n_guides
    ):
        stop = start + block.shape[0]
        room = _CALLS_PER_ROW * block.shape[0] + 64
        while True:
            call_rows = np.empty(room, dtype=np.int64)
            call_codes = np.empty(room, dtype=np.int64)
            call_umis = np.empty(room, dtype=np.uint32)
            n_calls = _call_block(
                block,
                codes[start:stop],
                table,
                top_code[start:stop],
                top_umi[start:stop],
                total_umi[start:stop],
                call_rows,
                call_codes,
                call_umis,
            )
            if n_calls <= room:
                break
            room = n_calls
        if n_calls < 0:
            raise ValueError(f"Guide UMI counts must be at most {MAX_COUNT}")
        cells.append(call_rows[:n_calls] + start)
        guides.append(call_codes[:n_calls])
        umis.append(call_umis[:n_calls])
    calls = CalledGuides(
        cells=np.concatenate(cells),
        codes=np.concatenate(guides),
        umis=np.concatenate(umis),
    )
    summary = GuideCountSummary(top_code=top_code, top_umi=top_umi, total_umi=total_umi)
    return calls, summary


def top_guide_calls(
    summary: GuideCountSummary,
    *,
    min_umi: int,
    fraction_threshold: float,
) -> CalledGuides:
    """Call each cell's top guide when it holds most of the cell's guide UMIs.

    A cell's top guide is called when the cell's guide UMIs reach ``min_umi``
    and the top guide's share of them exceeds ``fraction_threshold``. This is
    the maximum rule of the SCEPTRE book's "Assign gRNAs" chapter, which flags
    a cell with fewer guide UMIs than its floor, 5 by default, as holding no
    guide, and a cell whose top guide holds at most its fraction, 0.8 by
    default, as holding several. The share must exceed the fraction, as in
    crispat's ratio method too, so 4 of 5 UMIs is not called at 0.8. Among
    guides with equal UMIs, the top guide is the one first in the assay. A
    cell gets at most one call, and with ``fraction_threshold=0`` every cell
    whose guide UMIs reach ``min_umi`` is called.

    A floor on the top guide's UMIs instead would call the same cells whenever
    ``fraction_threshold`` is at least ``1 - 1 / min_umi``, as at the defaults
    of 0.8 and 5: a top guide that holds more than 0.8 of at least 5 UMIs
    holds 5 itself. Below that fraction the floors differ, and a cell with 4
    of its 6 guide UMIs in its top guide is called at 0.5 with a floor of 5.

    Args:
        summary: Each cell's top guide and guide UMIs.
        min_umi: The fewest guide UMIs of a called cell, at least 1.
        fraction_threshold: The share of the cell's guide UMIs that the top
            guide must exceed, at least 0 and below 1. Both settings are
            checked by
            :func:`~cytearc.perturbation.guide_thresholds.call_settings`.

    Returns:
        The calls, ordered by cell.
    """
    # The floor, at least 1, leaves out the cells without guide UMIs, which
    # have no share.
    called = (summary.total_umi >= min_umi) & (
        summary.top_fraction > fraction_threshold
    )
    cells = np.flatnonzero(called).astype(np.int64)
    return CalledGuides(
        cells=cells,
        codes=summary.top_code[cells].astype(np.int64),
        umis=summary.top_umi[cells].astype(np.uint32),
    )
