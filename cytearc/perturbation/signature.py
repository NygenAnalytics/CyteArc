"""Perturbation signatures: each cell's expression against its nearest controls.

Mixscape (Papalexi et al. 2021) first removes the variation that a cell shares
with the control cells around it, such as its cell-cycle phase or its
replicate, so that what remains is the perturbation's effect. For every cell
and feature, the signature is the cell's log-normalized value minus the log of
the mean normalized value of the cell's nearest control cells:

    s = log1p(y_i) - log1p(mean of y_j over the k nearest control cells j),

where y = counts / total counts of the cell * scale_factor, so log1p(y) is
Seurat's LogNormalize. Dividing before multiplying, as LogNormalize does,
gives two cells with equal shares of their counts equal values, which then tie
in rank tests as they do in Seurat. Seurat's ``CalcPerturbSig`` stores the
negative, control minus cell; the classes of Mixscape do not depend on the
sign, and here a knockdown reads negative.

The nearest controls are found exactly, by Euclidean distance over the first
dimensions of coordinates such as a PCA; a control cell is its own nearest
control, and of two equally near controls the one that comes first is taken.

These functions read values that a caller passes and never touch storage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import prange

from cytearc.utils.arguments import float_argument, integer_argument
from cytearc.utils.numba import njit

__all__ = [
    "PerturbationSignature",
    "nearest_controls",
    "normalize_counts",
    "normalize_counts_by_feature",
    "signature_of_values",
    "signature_rows",
    "signature_settings",
]

# The bytes of distances that one block of query cells holds.
_DISTANCE_BYTES = 1 << 26
# Candidates beyond the k nearest by the fast distances, which the exact
# distances then rank.
_EXTRA_CANDIDATES = 8


def signature_settings(
    dims: object,
    n_neighbors: object,
    scale_factor: object,
    sample_column: object,
) -> tuple[int, int, float]:
    """Check the settings of a perturbation signature, as ``signature`` takes them.

    The pipeline recipe checks the same settings before its run starts.
    Whether ``dims`` fits the coordinates, and whether the sample column
    exists, are checked against the store.

    Returns:
        The coordinate dimensions, the neighbors and the scale factor.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If a number is out of range.
    """
    resolved_dims = integer_argument(dims, "dims", minimum=1)
    neighbors_k = integer_argument(n_neighbors, "n_neighbors", minimum=1)
    factor = float_argument(scale_factor, "scale_factor")
    if factor <= 0:
        raise ValueError("scale_factor must be above 0")
    if sample_column is not None and (
        not isinstance(sample_column, str) or not sample_column
    ):
        raise TypeError("sample_column must be a non-empty string or None")
    return resolved_dims, neighbors_k, factor


def nearest_controls(
    query: np.ndarray, controls: np.ndarray, n_neighbors: int
) -> np.ndarray:
    """Return the positions of each query's nearest controls, nearest first.

    Distances come from |q|^2 + |c|^2 - 2 q.c in blocks of queries. The
    nearest candidates by those distances are ranked again by their exact
    squared distances, ties going to the lower position, and a query whose
    k-th distance comes within rounding of a control outside the candidates is
    ranked exactly over every control, so the result is exact.

    Args:
        query: The query cells' coordinates, cells x dimensions.
        controls: The control cells' coordinates, cells x dimensions.
        n_neighbors: The controls to return for each query, from 1 to the
            number of controls.

    Returns:
        The positions in ``controls`` of each query's ``n_neighbors`` nearest
        controls, nearest first, as int64.

    Raises:
        ValueError: If the coordinates do not align, or ``n_neighbors`` is out
            of range.
    """
    queries = np.asarray(query, dtype=np.float64)
    pool = np.asarray(controls, dtype=np.float64)
    if queries.ndim != 2 or pool.ndim != 2 or queries.shape[1] != pool.shape[1]:
        raise ValueError(
            "query and controls must be cells x dimensions with the same dimensions"
        )
    n_controls, n_dims = pool.shape
    if not 1 <= n_neighbors <= n_controls:
        raise ValueError(
            f"n_neighbors must be from 1 to the {n_controls} controls, not {n_neighbors}"
        )
    n_candidates = min(n_controls, n_neighbors + max(n_neighbors, _EXTRA_CANDIDATES))
    pool_norms = np.einsum("ij,ij->i", pool, pool)
    largest_norm = float(pool_norms.max())
    # A bound on the rounding of the fast distances, with a wide margin.
    rounding = 4.0 * (n_dims + 2) * np.finfo(np.float64).eps
    block = max(1, _DISTANCE_BYTES // (8 * max(n_controls, n_candidates * n_dims)))
    positions = np.empty((len(queries), n_neighbors), dtype=np.int64)
    every = np.arange(n_controls)
    for start in range(0, len(queries), block):
        rows = queries[start : start + block]
        norms = np.einsum("ij,ij->i", rows, rows)
        fast = norms[:, None] + pool_norms[None, :] - 2.0 * (rows @ pool.T)
        if n_candidates < n_controls:
            # The n_candidates nearest come first, and the next one is the
            # nearest control outside them.
            parted = np.argpartition(fast, n_candidates, axis=1)
            candidates = parted[:, :n_candidates]
            outside = np.take_along_axis(
                fast, parted[:, n_candidates : n_candidates + 1], 1
            )[:, 0]
        else:
            candidates = np.broadcast_to(every, fast.shape)
            outside = np.full(len(rows), np.inf)
        difference = rows[:, None, :] - pool[candidates]
        exact = np.einsum("ijk,ijk->ij", difference, difference)
        order = np.lexsort((candidates, exact), axis=1)
        ranked = np.take_along_axis(candidates, order, 1)[:, :n_neighbors]
        kth = np.take_along_axis(exact, order[:, n_neighbors - 1 : n_neighbors], 1)[
            :, 0
        ]
        slack = rounding * (norms + largest_norm)
        for row in np.flatnonzero(outside - slack <= kth + slack):
            gap = pool - rows[row]
            distances = np.einsum("ij,ij->i", gap, gap)
            ranked[row] = np.lexsort((every, distances))[:n_neighbors]
        positions[start : start + len(rows)] = ranked
    return positions


def normalize_counts(
    counts: np.ndarray, totals: np.ndarray, scale_factor: float
) -> np.ndarray:
    """Return each cell's counts scaled to ``scale_factor`` total counts.

    Each value is count / total * scale_factor, divided first as Seurat's
    LogNormalize computes it, so that two cells with equal shares of their
    counts get equal values.

    Args:
        counts: The cells' counts, cells x features.
        totals: Each cell's total counts over every feature of the assay; a
            cell without counts gets values of 0.
        scale_factor: The counts each cell is scaled to.

    Returns:
        The normalized values, cells x features, as float64.
    """
    values = np.asarray(counts, dtype=np.float64)
    sums = np.asarray(totals, dtype=np.float64)[:, None]
    out = np.zeros(values.shape)
    np.divide(values, sums, out=out, where=sums > 0)
    out *= scale_factor
    return out


@njit(cache=True, nogil=True, parallel=True)
def _normalize_by_feature(
    counts: np.ndarray,
    totals: np.ndarray,
    scale_factor: float,
    out: np.ndarray,
    start: int,
) -> None:
    """Write the normalized counts of a block of cells into feature-major ``out``.

    Features go in tiles of 64, so that each cell's counts are read in runs.
    """
    n_cells, n_features = counts.shape
    for tile in prange((n_features + 63) // 64):
        first = tile * 64
        last = min(first + 64, n_features)
        for cell in range(n_cells):
            total = totals[cell]
            for feature in range(first, last):
                if total > 0:
                    out[feature, start + cell] = (
                        counts[cell, feature] / total * scale_factor
                    )
                else:
                    out[feature, start + cell] = 0.0


def normalize_counts_by_feature(
    counts: np.ndarray,
    totals: np.ndarray,
    scale_factor: float,
    out: np.ndarray,
    start: int,
) -> None:
    """Write a block of cells' normalized counts into features x cells ``out``.

    The values are those of :func:`normalize_counts`, to the last bit, written
    to the columns of ``out`` from ``start`` on.

    Args:
        counts: The block's counts, cells x features.
        totals: Each cell's total counts.
        scale_factor: The counts each cell is scaled to.
        out: The float64 array, features x cells, that receives the values.
        start: The column of ``out`` of the block's first cell.

    Raises:
        ValueError: If the block does not fit ``out`` at ``start``.
    """
    block = np.ascontiguousarray(counts)
    sums = np.asarray(totals, dtype=np.float64)
    if (
        block.ndim != 2
        or sums.shape != block.shape[:1]
        or out.dtype != np.float64
        or out.ndim != 2
        or out.shape[0] != block.shape[1]
        or not 0 <= start <= out.shape[1] - len(block)
    ):
        raise ValueError(
            "counts must be cells x features with one total per cell, and fit "
            "the float64 features x cells out from start on"
        )
    _normalize_by_feature(block, sums, float(scale_factor), out, int(start))


@njit(cache=True, nogil=True, parallel=True)
def _signature_rows(
    counts: np.ndarray,
    totals: np.ndarray,
    neighbors: np.ndarray,
    control_scaled: np.ndarray,
    scale_factor: float,
    out: np.ndarray,
) -> None:
    """Write the signature of each row of ``counts`` into ``out``."""
    n_rows, n_features = counts.shape
    n_neighbors = neighbors.shape[1]
    for row in prange(n_rows):
        mean = np.zeros(n_features)
        for neighbor in range(n_neighbors):
            source = control_scaled[neighbors[row, neighbor]]
            for feature in range(n_features):
                mean[feature] += source[feature]
        total = totals[row]
        for feature in range(n_features):
            value = counts[row, feature] / total * scale_factor if total > 0 else 0.0
            out[row, feature] = np.log1p(value) - np.log1p(mean[feature] / n_neighbors)


@njit(cache=True, nogil=True, parallel=True)
def _signature_of_values(
    values: np.ndarray,
    features: np.ndarray,
    cells: np.ndarray,
    neighbors: np.ndarray,
    out: np.ndarray,
) -> None:
    """Write the signature of ``cells`` in ``features`` of ``values`` into ``out``.

    ``values`` holds normalized values, features x cells, and each neighbor's
    values are added in the order of ``_signature_rows``.
    """
    n_neighbors = neighbors.shape[1]
    for column in prange(len(features)):
        source = values[features[column]]
        for index in range(len(cells)):
            total = 0.0
            for neighbor in range(n_neighbors):
                total += source[neighbors[index, neighbor]]
            out[index, column] = np.log1p(source[cells[index]]) - np.log1p(
                total / n_neighbors
            )


def signature_of_values(
    values: np.ndarray,
    features: np.ndarray,
    cells: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Return the signatures of some cells from normalized values, by feature.

    From values that :func:`normalize_counts` gives, the signatures equal
    those that :func:`signature_rows` computes from the counts, to the last
    bit, before :func:`signature_rows` rounds them to float32.

    Args:
        values: Normalized values, features x cells, of the cells and of
            their nearest controls.
        features: The rows of ``values`` to compute.
        cells: The columns of ``values`` of the cells to compute.
        neighbors: Each cell's nearest controls, as columns of ``values``,
            one row per cell.

    Returns:
        The signatures, cells x features, as float64.

    Raises:
        ValueError: If the arrays do not align.
    """
    held = np.ascontiguousarray(values, dtype=np.float64)
    rows = np.asarray(features, dtype=np.int64)
    columns = np.asarray(cells, dtype=np.int64)
    nearest = np.ascontiguousarray(neighbors, dtype=np.int64)
    width = held.shape[1] if held.ndim == 2 else 0
    if (
        held.ndim != 2
        or nearest.shape[:1] != columns.shape
        or nearest.ndim != 2
        or not np.all((rows >= 0) & (rows < len(held)))
        or not np.all((columns >= 0) & (columns < width))
        or not np.all((nearest >= 0) & (nearest < width))
    ):
        raise ValueError(
            "values must be features x cells, with features and cells inside "
            "it and one row of neighbors per cell"
        )
    out = np.empty((len(columns), len(rows)))
    _signature_of_values(held, rows, columns, nearest, out)
    return out


def signature_rows(
    counts: np.ndarray,
    totals: np.ndarray,
    neighbors: np.ndarray,
    control_scaled: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Return the signatures of a block of cells.

    Args:
        counts: The cells' counts of the signature's features, cells x
            features.
        totals: Each cell's total counts over every feature of the assay; a
            cell without counts has normalized values of 0.
        neighbors: Each cell's nearest controls, as rows of ``control_scaled``.
        control_scaled: The control cells' normalized values, as
            :func:`normalize_counts` gives them, controls x features.
        scale_factor: The counts each cell is scaled to.

    Returns:
        The signatures, cells x features, as float32.

    Raises:
        ValueError: If the arrays do not align.
    """
    values = np.ascontiguousarray(counts, dtype=np.float64)
    sums = np.ascontiguousarray(totals, dtype=np.float64)
    nearest = np.ascontiguousarray(neighbors, dtype=np.int64)
    scaled = np.ascontiguousarray(control_scaled, dtype=np.float64)
    if (
        values.ndim != 2
        or sums.shape != (len(values),)
        or nearest.ndim != 2
        or len(nearest) != len(values)
        or scaled.ndim != 2
        or scaled.shape[1] != values.shape[1]
    ):
        raise ValueError(
            "counts, totals and neighbors must hold one row per cell, and the "
            "control values one column per feature"
        )
    if nearest.size and (nearest.min() < 0 or nearest.max() >= len(scaled)):
        raise ValueError("neighbors must be rows of the control values")
    out = np.empty(values.shape, dtype=np.float32)
    _signature_rows(values, sums, nearest, scaled, float(scale_factor), out)
    return out


@dataclass(frozen=True, slots=True)
class PerturbationSignature:
    """The perturbation signature of each cell.

    Attributes:
        values: One row per cell, indexed by cell ``ids``, and one column per
            feature, named by feature ID: the cell's log-normalized value
            minus the log of its nearest controls' mean normalized value.
        cells: One row per cell, indexed by cell ``ids``, with ``is_control``
            and ``sample``.
        neighbors: Each cell's nearest control cells, nearest first, as cell
            ``ids``, one column per neighbor.
        from_assay: The assay of the features.
        dims: The coordinate dimensions the neighbors were found in.
        n_neighbors: The control cells averaged for each cell.
        scale_factor: The counts each cell was scaled to before log1p.
        sample_column: The cell metadata column whose samples kept their own
            controls, or None.
    """

    values: pd.DataFrame = field(repr=False)
    cells: pd.DataFrame = field(repr=False)
    neighbors: pd.DataFrame = field(repr=False)
    from_assay: str
    dims: int
    n_neighbors: int
    scale_factor: float
    sample_column: str | None
