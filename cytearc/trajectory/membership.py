"""Which lineages each cell follows, and where two lineages part.

A fate map gives each scored cell one probability per sink, and those
probabilities sum to one, so a cell on no trajectory still splits its
probability among the sinks. Lineage curves add each cell's progress along
every lineage's curve and its distance from it. A cell follows a lineage when
its fate probability for that lineage, measured against what the cells at
the start of the lineage's curve receive, lies within a band of the largest
such ratio over the lineages, and when it lies on the trajectory: within the
spread that some lineage's own cells show around their curve at the same
progress. A cell that follows several lineages is shared by them, and one
that lies on no curve follows none, whatever its probabilities. Two lineages
part where their shared cells give way to cells that follow only one of them.

Progenitors split their probability unevenly among the fates, so a cell's
largest probability is no fair yardstick for a fate that the progenitors give
a small share. Dividing each lineage's probabilities by their mean over the
cells at the start of its curve puts every lineage on the scale of its own
trunk: a progenitor whose probabilities match the trunk's split scores one
for every lineage that starts there, however small the share of a lineage.
The cells at the start of a curve are those in the first bin of progress
along it that lie within its distance limit there, leaving out the cells of
the fate map's sinks. Each lineage takes them from its own curve, so the
scale does not depend on how many cells each branch holds, on what directs
the fate map, or on whether the map spans several graph components.

The spread is robust: the weighted median distance of a lineage's cells in
a bin of progress, plus a multiple of their weighted median absolute
deviation scaled to a normal standard deviation, so a separate state with
less weight than the lineage's own cells there cannot widen it. A state off
the trajectory, such as a dead end off a lineage, projects onto one stretch
of the curve, where it can outweigh the lineage's own cells, so each bin
takes the median limit of three neighbouring bins.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BRANCH_POINT_COLUMNS",
    "MEMBERSHIP_NONE",
    "MEMBERSHIP_SHARED",
    "branch_point",
    "branch_points",
    "branch_window",
    "distance_limits",
    "lineage_members",
    "membership_labels",
    "progress_bins",
    "weighted_median",
]

#: The label of a cell that follows several lineages.
MEMBERSHIP_SHARED = "shared"
#: The label of a cell that follows no lineage.
MEMBERSHIP_NONE = "none"
#: The columns of a table of branch points, one row per pair of lineages.
BRANCH_POINT_COLUMNS = (
    "first",
    "second",
    "pseudotime",
    "first_progress",
    "second_progress",
    "shared_cells",
    "parted_cells",
)
# The median absolute deviation of a normal distribution, in standard
# deviations, is 1 / 1.4826.
_MAD_SCALE = 1.4826


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the smallest value at which the weights up to it reach half their total.

    Values of zero weight count for nothing, and values without weight give
    NaN. With equal weights and an even count, this is the lower of the two
    middle values.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("values and weights must be one-dimensional and aligned")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("weights must be finite and non-negative")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if not len(cumulative) or cumulative[-1] <= 0:
        return math.nan
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order][index])


def progress_bins(progress: np.ndarray, n_bins: int) -> np.ndarray:
    """Return each progress value's bin among ``n_bins`` equal bins of ``[0, 1]``.

    Progress 1 falls in the last bin, and progress that is not finite in the
    first, for callers that leave such cells out.
    """
    values = np.asarray(progress, dtype=np.float64)
    finite = np.isfinite(values)
    codes = np.zeros(values.shape, dtype=np.int64)
    codes[finite] = np.clip(
        np.floor(values[finite] * n_bins).astype(np.int64), 0, n_bins - 1
    )
    return codes


def distance_limits(
    distance: np.ndarray,
    progress: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    *,
    n_bins: int,
    score: float,
) -> np.ndarray:
    """Return the farthest distance from each lineage's curve that lies on it, by bin.

    For lineage ``l`` and bin ``b`` of progress, the scored cells of the bin
    weigh by their fate probability for ``l``, and their spread is ``m +
    score * 1.4826 * mad``, with ``m`` the weighted median of their
    distances to the curve and ``mad`` the weighted median of their absolute
    deviations from ``m``. A bin's limit is the median spread of three
    neighbouring bins, the bin and one on each side, or the first or last
    three at either end, leaving out bins without weight. With fewer than
    three bins, every bin takes the median of all, and a lineage without
    weight has no limit, NaN, so no cell lies on it.

    Returns:
        Lineages by bins.
    """
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2:
        raise ValueError("distance must be cells by lineages")
    n_cells, n_lineages = distance.shape
    if np.shape(progress) != distance.shape or np.shape(weights) != distance.shape:
        raise ValueError("progress, distance and weights must be cells by lineages")
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (n_cells,):
        raise ValueError("valid must hold one entry per cell")
    limits = np.full((n_lineages, n_bins), np.nan)
    for lineage in range(n_lineages):
        lineage_weights = np.asarray(weights[:, lineage], dtype=np.float64)
        weighed = valid & (lineage_weights > 0)
        if not weighed.any():
            continue
        cell_bins = progress_bins(progress[weighed, lineage], n_bins)
        cell_distances = distance[weighed, lineage]
        cell_weights = lineage_weights[weighed]
        order = np.argsort(cell_bins, kind="stable")
        bounds = np.searchsorted(cell_bins[order], np.arange(n_bins + 1))
        for index in range(n_bins):
            rows = order[bounds[index] : bounds[index + 1]]
            if not len(rows):
                continue
            center = weighted_median(cell_distances[rows], cell_weights[rows])
            spread = weighted_median(
                np.abs(cell_distances[rows] - center), cell_weights[rows]
            )
            limits[lineage, index] = center + score * _MAD_SCALE * spread
    return _neighbouring_medians(limits)


def _neighbouring_medians(spreads: np.ndarray) -> np.ndarray:
    """Return each bin's median spread over three neighbouring bins, without empty ones."""
    n_bins = spreads.shape[1]
    medians = np.full(spreads.shape, np.nan)
    for index in range(n_bins):
        first = min(max(index - 1, 0), max(n_bins - 3, 0))
        window = spreads[:, first : first + 3]
        for lineage in range(spreads.shape[0]):
            present = window[lineage][np.isfinite(window[lineage])]
            if len(present):
                medians[lineage, index] = float(np.median(present))
    return medians


def lineage_members(
    weights: np.ndarray,
    distance: np.ndarray,
    progress: np.ndarray,
    valid: np.ndarray,
    limits: np.ndarray,
    sinks: np.ndarray,
    *,
    band: float,
    lineages: Sequence[Any],
) -> np.ndarray:
    """Return which lineages each scored cell follows, cells by lineages.

    A cell lies on lineage ``l``'s curve when it lies within ``l``'s limit of
    ``limits`` at the cell's bin of progress on it. Each lineage's
    probabilities are divided by their mean over the scored cells that lie
    on its curve in its first bin of progress, leaving out the cells that
    ``sinks`` marks. A cell follows lineage ``l`` when its probability for
    ``l`` is positive and its divided probability is at least ``band`` times
    its largest divided probability over the lineages, and when it lies on
    some lineage's curve. Unscored cells follow none and set no mean.

    Raises:
        ValueError: If no cell outside the sinks at the start of a lineage's
            curve has a probability for that lineage, so that nothing sets
            its scale.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be cells by lineages")
    n_cells, n_lineages = weights.shape
    if len(lineages) != n_lineages:
        raise ValueError("lineages must name one lineage per column of weights")
    valid = np.asarray(valid, dtype=bool)
    sinks = np.asarray(sinks, dtype=bool)
    if sinks.shape != (n_cells,):
        raise ValueError("sinks must hold one entry per cell")
    if limits.ndim != 2 or limits.shape[0] != n_lineages:
        raise ValueError("limits must be lineages by bins")
    bins = progress_bins(progress, limits.shape[1])
    lineage_limits = limits[np.arange(n_lineages)[None, :], bins]
    with np.errstate(invalid="ignore"):
        on_curve = valid[:, None] & (
            np.asarray(distance, dtype=np.float64) <= lineage_limits
        )
    scored = np.where(valid[:, None] & np.isfinite(weights), weights, 0.0)
    start = on_curve & (bins == 0) & ~sinks[:, None]
    totals = np.where(start, scored, 0.0).sum(axis=0)
    unscaled = [lineages[int(index)] for index in np.flatnonzero(totals <= 0)]
    if unscaled:
        named = ", ".join(repr(lineage) for lineage in unscaled)
        raise ValueError(
            "Lineage membership compares each lineage's probabilities with "
            "those of the cells outside the sinks at the start of its curve, "
            f"and no such cell has a probability for {named}. Such a curve "
            "starts among sink cells or in a graph component that holds only "
            "sink cells; fit the curves without that lineage, or check the "
            "score that directs the fate map"
        )
    rescaled = scored / (totals / start.sum(axis=0))
    largest = rescaled.max(axis=1, initial=0.0)
    compatible = (scored > 0) & (rescaled >= band * largest[:, None])
    members: np.ndarray = on_curve.any(axis=1)[:, None] & compatible
    return members


def membership_labels(members: np.ndarray, lineages: Sequence[Any]) -> np.ndarray:
    """Name what each cell follows: a lineage's sink label as text, shared, or none."""
    members = np.asarray(members, dtype=bool)
    if members.ndim != 2 or members.shape[1] != len(lineages):
        raise ValueError("members must be cells by lineages")
    names = [str(lineage) for lineage in lineages]
    meanings = {MEMBERSHIP_SHARED: "several lineages", MEMBERSHIP_NONE: "none"}
    for reserved, meaning in meanings.items():
        if reserved in names:
            raise ValueError(
                f"A lineage named {reserved!r} cannot be told apart from the "
                f"label of cells that follow {meaning}; rename that sink"
            )
    counts = members.sum(axis=1)
    labels = np.full(members.shape[0], MEMBERSHIP_NONE, dtype=object)
    labels[counts > 1] = MEMBERSHIP_SHARED
    single = counts == 1
    labels[single] = np.asarray(names, dtype=object)[members[single].argmax(axis=1)]
    return labels.astype(str)


def branch_point(
    pseudotime: np.ndarray, shared: np.ndarray, parted: np.ndarray
) -> float:
    """Return the pseudotime that best splits two lineages' shared cells from their parted cells.

    The shared cells follow both lineages, and the parted cells follow one
    of them. The split puts shared cells before it and parted cells after
    it with the fewest cells on the wrong side, the earliest such split, and
    lies halfway between the last cell before it and the first after it.
    The lineages have no branch point, NaN, when that split lies before the
    first cell, where it misplaces every shared cell, or after the last one,
    where it misplaces every parted cell: no stretch of shared cells comes
    before the parted cells. A split between two cells beats the one before
    the first cell only when the cells before it hold more shared than
    parted cells. Without shared or without parted cells the lineages have
    no branch point either.
    """
    pseudotime = np.asarray(pseudotime, dtype=np.float64)
    shared = np.asarray(shared, dtype=bool)
    parted = np.asarray(parted, dtype=bool)
    if np.any(shared & parted):
        raise ValueError("A cell cannot be both shared and parted")
    if not shared.any() or not parted.any():
        return math.nan
    pair = shared | parted
    order = np.argsort(pseudotime[pair], kind="stable")
    times = pseudotime[pair][order]
    is_shared = shared[pair][order]
    # Splitting after the first k cells puts the parted cells among them and
    # the shared cells among the rest on the wrong side.
    parted_before = np.concatenate(([0], np.cumsum(~is_shared)))
    shared_after = np.concatenate((np.cumsum(is_shared[::-1])[::-1], [0]))
    errors = parted_before + shared_after
    # A split falls between cells of different pseudotimes, or at either end.
    allowed = np.ones(len(times) + 1, dtype=bool)
    allowed[1:-1] = times[1:] != times[:-1]
    errors = np.where(allowed, errors, np.iinfo(np.int64).max)
    after = int(np.argmin(errors))
    if after == 0 or after == len(times):
        return math.nan
    return float(0.5 * (times[after - 1] + times[after]))


def _progress_at(pseudotime: float, curve_pseudotime: np.ndarray) -> float:
    """Return the progress at which a curve's fitted pseudotime reaches ``pseudotime``."""
    if not math.isfinite(pseudotime):
        return math.nan
    rising = np.maximum.accumulate(np.asarray(curve_pseudotime, dtype=np.float64))
    return float(np.interp(pseudotime, rising, np.linspace(0.0, 1.0, len(rising))))


def branch_points(
    members: np.ndarray,
    pseudotime: np.ndarray,
    curve_pseudotime: np.ndarray,
    lineages: Sequence[Any],
) -> pd.DataFrame:
    """Return where each pair of lineages parts, one row per pair in lineage order.

    For lineages ``first`` and ``second``, ``pseudotime`` is the
    :func:`branch_point` of the cells that follow both and those that follow
    one of them, NaN when they have none, and ``first_progress`` and
    ``second_progress`` are the progress on each lineage's curve at which its
    fitted pseudotime reaches it, NaN with it. ``shared_cells`` and
    ``parted_cells`` count those cells.
    """
    members = np.asarray(members, dtype=bool)
    if members.ndim != 2 or members.shape[1] != len(lineages):
        raise ValueError("members must be cells by lineages")
    if np.shape(curve_pseudotime)[0] != len(lineages):
        raise ValueError("curve_pseudotime must hold one curve per lineage")
    rows: list[dict[str, Any]] = []
    for first in range(len(lineages)):
        for second in range(first + 1, len(lineages)):
            shared = members[:, first] & members[:, second]
            parted = members[:, first] ^ members[:, second]
            point = branch_point(pseudotime, shared, parted)
            rows.append(
                {
                    "first": lineages[first],
                    "second": lineages[second],
                    "pseudotime": point,
                    "first_progress": _progress_at(point, curve_pseudotime[first]),
                    "second_progress": _progress_at(point, curve_pseudotime[second]),
                    "shared_cells": int(shared.sum()),
                    "parted_cells": int(parted.sum()),
                }
            )
    return pd.DataFrame(rows, columns=list(BRANCH_POINT_COLUMNS))


def branch_window(
    pseudotime: np.ndarray, pair: np.ndarray, point: float, share: float
) -> np.ndarray:
    """Return the cells of a pair of lineages nearest their branch point in pseudotime.

    ``pair`` marks the cells that follow either lineage. The window holds
    the ``share`` of them, rounded up, whose pseudotime lies closest to
    ``point``; of cells at the same distance, the earlier ones come first.
    """
    pseudotime = np.asarray(pseudotime, dtype=np.float64)
    pair = np.asarray(pair, dtype=bool)
    if not math.isfinite(point):
        raise ValueError("The lineages have no branch point")
    candidates = np.flatnonzero(pair)
    if not len(candidates):
        raise ValueError("No cell follows either lineage")
    n_window = max(1, math.ceil(share * len(candidates)))
    order = np.lexsort((pseudotime[candidates], np.abs(pseudotime[candidates] - point)))
    window = np.zeros(pair.shape, dtype=bool)
    window[candidates[order[:n_window]]] = True
    return window
