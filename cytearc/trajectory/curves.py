"""Fate-weighted curves through reduced coordinates, one per lineage.

A lineage curve is the fate-weighted mean of the cells' coordinates as a
smooth function of their weighted pseudotime rank along that lineage. The
rank of a cell is the share of the lineage's fate weight carried by cells
that precede it in pseudotime, so it runs from zero at the start to one at
the sink, and every stretch of rank holds the same weight. The mean is a
cubic penalized B-spline (Eilers and Marx 1996,
doi:10.1214/ss/1038425655) with a fixed number of effective degrees of
freedom, fitted to weighted means in fine rank bins, so the fit streams over
blocks of coordinates.

The curves are principal curves (Hastie and Stuetzle 1989,
doi:10.1080/01621459.1989.10478797) whose projection index is held at the
pseudotime rank. Hastie and Stuetzle's iterations, which reproject cells
orthogonally and refit, are not run: where a lineage starts or ends in a
compact cluster they fold the curve into loops that follow no ordering. Cells
are projected orthogonally onto a fitted curve only to report their position
along it and their distance from it.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numba
import numpy as np
from numba import prange, set_num_threads
from scipy.interpolate import BSpline

from ..utils.numba import njit, restore_numba_threads

__all__ = [
    "CURVE_FIT_BINS",
    "CURVE_SPLINE_BASIS",
    "LineageCurveFit",
    "curve_values_at",
    "fit_lineage_curves",
    "lineage_ranks",
    "project_onto_curve",
    "projection_scratch_bytes",
    "smooth_against_progress",
]

# Rank bins of the streamed fit and B-spline basis size. Both shape the
# fitted curves, so they are recorded with each result.
CURVE_FIT_BINS = 512
CURVE_SPLINE_BASIS = 32
# Bisection steps on the log penalty; 60 halvings of 24 decades reach the
# requested degrees of freedom far below their tolerance.
_PENALTY_STEPS = 60
_LOG_PENALTY_RANGE = (-12.0, 12.0)


@dataclass(frozen=True, slots=True)
class LineageCurveFit:
    """Curves fitted to every lineage.

    Attributes:
        points: Curve points, lineages by points by dimensions.
        ordering: The fitted pseudotime at each point, lineages by points.
        progress: The rank of each point, from zero to one.
    """

    points: np.ndarray
    ordering: np.ndarray
    progress: np.ndarray


def lineage_ranks(ordering: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return each cell's weighted pseudotime rank along every lineage.

    Along one lineage, cells are ordered by ``ordering``, ties by position,
    and a cell's rank is the weight of the cells before it plus half its own,
    divided by the lineage's total weight. Cells without weight have no rank.
    """
    order_values = np.asarray(ordering, dtype=np.float64)
    cell_weights = np.asarray(weights, dtype=np.float64)
    if cell_weights.ndim != 2 or order_values.shape != (cell_weights.shape[0],):
        raise ValueError("ordering must have one value per weight row")
    if not np.isfinite(order_values).all():
        raise ValueError("ordering must be finite")
    if not np.isfinite(cell_weights).all() or np.any(cell_weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    order = np.argsort(order_values, kind="stable")
    ranks = np.full(cell_weights.shape, np.nan, dtype=np.float64)
    for lineage in range(cell_weights.shape[1]):
        ordered = cell_weights[order, lineage]
        total = float(ordered.sum())
        if total <= 0.0:
            raise ValueError("Every lineage needs a positive total weight")
        midpoints = (np.cumsum(ordered) - 0.5 * ordered) / total
        positive = ordered > 0.0
        ranks[order[positive], lineage] = np.clip(midpoints[positive], 0.0, 1.0)
    return ranks


@njit(cache=True)
def _accumulate_rank_bins(
    block: np.ndarray,
    ranks: np.ndarray,
    weights: np.ndarray,
    ordering: np.ndarray,
    n_bins: int,
    mass: np.ndarray,
    rank_sums: np.ndarray,
    ordering_sums: np.ndarray,
    sums: np.ndarray,
) -> None:
    """Add a block's weighted coordinates, ranks, and pseudotime to rank bins."""
    for cell in range(block.shape[0]):
        for lineage in range(weights.shape[1]):
            weight = weights[cell, lineage]
            if weight <= 0.0:
                continue
            rank = ranks[cell, lineage]
            bin_index = int(rank * n_bins)
            if bin_index >= n_bins:
                bin_index = n_bins - 1
            mass[lineage, bin_index] += weight
            rank_sums[lineage, bin_index] += weight * rank
            ordering_sums[lineage, bin_index] += weight * ordering[cell]
            for dim in range(block.shape[1]):
                sums[lineage, bin_index, dim] += weight * block[cell, dim]


class _PenalizedSpline:
    """Cubic B-splines on [0, 1] with a second-order difference penalty."""

    def __init__(self, n_basis: int) -> None:
        inner = np.linspace(0.0, 1.0, n_basis - 2)
        step = inner[1] - inner[0]
        self.knots = np.concatenate(
            (
                [-3.0 * step, -2.0 * step, -step],
                inner,
                [1.0 + step, 1.0 + 2.0 * step, 1.0 + 3.0 * step],
            )
        )
        differences = np.diff(np.eye(n_basis), 2, axis=0)
        self.penalty = differences.T @ differences
        self.n_basis = n_basis

    def design(self, x: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
        return np.asarray(BSpline.design_matrix(values, self.knots, 3).toarray())

    def fit(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        degrees_of_freedom: float,
    ) -> np.ndarray:
        """Return coefficients whose fit spends ``degrees_of_freedom``.

        The effective degrees of freedom, the trace of the smoother, fall from
        the rank of the weighted design toward two, a straight line, as the
        penalty grows. The penalty is found by bisection on its logarithm.
        """
        basis = self.design(x)
        gram = basis.T @ (weights[:, np.newaxis] * basis)
        right = basis.T @ (weights[:, np.newaxis] * values)
        scale = float(np.trace(gram)) / float(np.trace(self.penalty))

        def effective(log_penalty: float) -> float:
            system = gram + scale * 10.0**log_penalty * self.penalty
            return float(np.trace(np.linalg.solve(system, gram)))

        low, high = _LOG_PENALTY_RANGE
        if effective(low) < degrees_of_freedom:
            raise ValueError(
                "A lineage has too few distinct pseudotime ranks for "
                f"{degrees_of_freedom:g} degrees of freedom; lower df"
            )
        for _ in range(_PENALTY_STEPS):
            middle = 0.5 * (low + high)
            if effective(middle) > degrees_of_freedom:
                low = middle
            else:
                high = middle
        system = gram + scale * 10.0 ** (0.5 * (low + high)) * self.penalty
        return np.asarray(np.linalg.solve(system, right))


def fit_lineage_curves(
    blocks: Iterable[tuple[int, np.ndarray]],
    ranks: np.ndarray,
    weights: np.ndarray,
    ordering: np.ndarray,
    *,
    n_dims: int,
    n_points: int,
    degrees_of_freedom: float,
    n_bins: int = CURVE_FIT_BINS,
    n_basis: int = CURVE_SPLINE_BASIS,
) -> LineageCurveFit:
    """Fit one curve per lineage from row blocks of coordinates.

    Args:
        blocks: ``(start, block)`` pairs that cover the cells in order, each
            block holding the coordinates of consecutive rows.
        ranks: Each cell's rank along every lineage, from
            :func:`lineage_ranks`.
        weights: Non-negative cell weights, cells by lineages.
        ordering: The pseudotime of each cell.
        n_dims: Coordinate dimensions.
        n_points: Points on each fitted curve.
        degrees_of_freedom: Effective degrees of freedom of each coordinate's
            spline.
        n_bins: Rank bins that summarize the cells.
        n_basis: Cubic B-splines in the basis.

    Returns:
        The curves, their pseudotime, and the rank of their points.

    Raises:
        ValueError: If the blocks do not cover the cells, hold non-finite
            coordinates, or a lineage cannot support the degrees of freedom.
    """
    cell_weights = np.ascontiguousarray(weights, dtype=np.float64)
    cell_ranks = np.ascontiguousarray(ranks, dtype=np.float64)
    order_values = np.ascontiguousarray(ordering, dtype=np.float64)
    n_cells, n_lineages = cell_weights.shape
    mass = np.zeros((n_lineages, n_bins), dtype=np.float64)
    rank_sums = np.zeros_like(mass)
    ordering_sums = np.zeros_like(mass)
    sums = np.zeros((n_lineages, n_bins, n_dims), dtype=np.float64)
    covered = 0
    for start, raw_block in blocks:
        block = np.ascontiguousarray(raw_block, dtype=np.float64)
        stop = start + block.shape[0]
        if start != covered or block.ndim != 2 or block.shape[1] != n_dims:
            raise ValueError("Coordinate blocks must cover the cells in order")
        if stop > n_cells:
            raise ValueError("Coordinate blocks extend beyond the cells")
        if not np.isfinite(block).all():
            raise ValueError("Coordinates must be finite")
        _accumulate_rank_bins(
            block,
            cell_ranks[start:stop],
            cell_weights[start:stop],
            order_values[start:stop],
            n_bins,
            mass,
            rank_sums,
            ordering_sums,
            sums,
        )
        covered = stop
    if covered != n_cells:
        raise ValueError("Coordinate blocks must cover every cell")
    spline = _PenalizedSpline(n_basis)
    progress = np.linspace(0.0, 1.0, n_points)
    design = spline.design(progress)
    points = np.empty((n_lineages, n_points, n_dims), dtype=np.float64)
    fitted_ordering = np.empty((n_lineages, n_points), dtype=np.float64)
    for lineage in range(n_lineages):
        filled = mass[lineage] > 0.0
        bin_mass = mass[lineage, filled]
        centers = rank_sums[lineage, filled] / bin_mass
        means = np.column_stack(
            (
                sums[lineage, filled] / bin_mass[:, np.newaxis],
                ordering_sums[lineage, filled] / bin_mass,
            )
        )
        coefficients = spline.fit(
            centers,
            bin_mass / bin_mass.mean(),
            means,
            degrees_of_freedom,
        )
        fitted = design @ coefficients
        points[lineage] = fitted[:, :n_dims]
        fitted_ordering[lineage] = np.maximum.accumulate(fitted[:, n_dims])
    return LineageCurveFit(
        points=points,
        ordering=fitted_ordering,
        progress=progress,
    )


def projection_scratch_bytes(n_points: int) -> int:
    """Return the bytes per projected row of :func:`project_onto_curve`.

    The fused kernel holds no per-segment arrays, so a row needs only its
    position and distance, whatever ``n_points``.
    """
    del n_points
    return 2 * np.dtype(np.float64).itemsize


@njit(parallel=True, cache=True)
def _project_rows(
    rows: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
    inverse_lengths: np.ndarray,
    progress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's position on a polyline and its distance from it.

    Every segment is tried for every row, in one fused loop that holds no
    per-segment arrays; ties go to the earliest segment. A zero-length
    segment has a zero inverse length, which measures the distance to its
    start point.
    """
    n_rows, n_dims = rows.shape
    n_segments = starts.shape[0]
    position = np.empty(n_rows, dtype=np.float64)
    distance = np.empty(n_rows, dtype=np.float64)
    for row in prange(n_rows):
        best = np.inf
        best_position = 0.0
        for segment in range(n_segments):
            along = 0.0
            for dim in range(n_dims):
                along += (rows[row, dim] - starts[segment, dim]) * vectors[segment, dim]
            fraction = along * inverse_lengths[segment]
            if fraction < 0.0:
                fraction = 0.0
            elif fraction > 1.0:
                fraction = 1.0
            squared = 0.0
            for dim in range(n_dims):
                gap = (
                    rows[row, dim]
                    - starts[segment, dim]
                    - fraction * vectors[segment, dim]
                )
                squared += gap * gap
            if squared < best:
                best = squared
                best_position = progress[segment] + fraction * (
                    progress[segment + 1] - progress[segment]
                )
        position[row] = best_position
        distance[row] = math.sqrt(best)
    return position, distance


@restore_numba_threads
def project_onto_curve(
    coordinates: np.ndarray,
    points: np.ndarray,
    progress: np.ndarray,
    *,
    nthreads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project rows orthogonally onto a polyline and return position and distance.

    Each row goes to the nearest point of the polyline through ``points``.
    Its position is the rank of that point, interpolated between the ranks
    of the segment's ends. Rows are projected in parallel over ``nthreads``
    Numba threads.

    Returns:
        The rank of each row's projection and its Euclidean distance from
        the curve.
    """
    rows = np.ascontiguousarray(coordinates, dtype=np.float64)
    curve = np.asarray(points, dtype=np.float64)
    point_ranks = np.ascontiguousarray(progress, dtype=np.float64)
    if rows.ndim != 2 or curve.ndim != 2 or rows.shape[1] != curve.shape[1]:
        raise ValueError("coordinates and points must share their dimensions")
    if curve.shape[0] < 2:
        raise ValueError("A curve needs at least two points")
    # The kernel indexes progress by segment without bounds checks.
    if point_ranks.shape != (curve.shape[0],):
        raise ValueError("progress must have one value per curve point")
    vectors = np.ascontiguousarray(np.diff(curve, axis=0))
    lengths = np.einsum("ij,ij->i", vectors, vectors)
    if not np.any(lengths > 0.0):
        raise ValueError("A curve must not collapse to one point")
    inverse = np.zeros_like(lengths)
    inverse[lengths > 0.0] = 1.0 / lengths[lengths > 0.0]
    set_num_threads(min(max(1, int(nthreads)), numba.config.NUMBA_NUM_THREADS))
    return _project_rows(
        rows,
        np.ascontiguousarray(curve[:-1]),
        vectors,
        inverse,
        point_ranks,
    )


def curve_values_at(
    position: np.ndarray,
    progress: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate per-point curve values, such as pseudotime, at positions."""
    return np.asarray(np.interp(position, progress, values), dtype=np.float64)


def smooth_against_progress(
    coordinates: np.ndarray,
    position: np.ndarray,
    weights: np.ndarray,
    *,
    n_points: int,
    degrees_of_freedom: float,
    n_bins: int = CURVE_FIT_BINS,
    n_basis: int = CURVE_SPLINE_BASIS,
) -> np.ndarray:
    """Return a curve through other coordinates, such as a 2D layout.

    The coordinates are averaged with ``weights`` in bins of position along
    a fitted curve and smoothed with the same penalized spline, which draws
    the curve in a layout that it was not fitted in.
    """
    rows = np.asarray(coordinates, dtype=np.float64)
    ranks = np.asarray(position, dtype=np.float64)
    cell_weights = np.asarray(weights, dtype=np.float64)
    keep = (cell_weights > 0.0) & np.isfinite(ranks) & np.isfinite(rows).all(axis=1)
    if not keep.any():
        raise ValueError("No weighted cells lie on the curve")
    bins = np.minimum((ranks[keep] * n_bins).astype(np.int64), n_bins - 1)
    bins = np.maximum(bins, 0)
    kept_weights = cell_weights[keep]
    mass = np.bincount(bins, weights=kept_weights, minlength=n_bins)
    filled = mass > 0.0
    centers = (
        np.bincount(bins, weights=kept_weights * ranks[keep], minlength=n_bins)[filled]
        / mass[filled]
    )
    means = np.column_stack(
        [
            np.bincount(bins, weights=kept_weights * rows[keep, dim], minlength=n_bins)[
                filled
            ]
            / mass[filled]
            for dim in range(rows.shape[1])
        ]
    )
    spline = _PenalizedSpline(n_basis)
    coefficients = spline.fit(
        centers,
        mass[filled] / mass[filled].mean(),
        means,
        degrees_of_freedom,
    )
    return np.asarray(spline.design(np.linspace(0.0, 1.0, n_points)) @ coefficients)
