"""Feature statistics resolved by lineage with fate-probability weights.

A fate map gives each scored cell one probability per sink. These helpers
treat those probabilities as cell weights, so a statistic for one lineage
counts each cell by how strongly it is connected to that lineage's sink,
the approach of CellRank's lineage drivers and gene trends (Lange et al.
2022, doi:10.1038/s41592-021-01346-6). They work on blocks of normalized
feature values, cells by features, so callers can stream features.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numba
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.sparse import csc_matrix, spmatrix
from scipy.stats import norm

from ..features.markers.regression import _regression_p_values
from ..features.statistical import adjust_pvalues
from ..utils.numba import restore_numba_threads

__all__ = [
    "LINEAGE_NONFINITE",
    "LINEAGE_OK",
    "LINEAGE_UNTESTED",
    "adjust_lineage_p_values",
    "bin_operator_bytes",
    "bin_smoothing_matrix",
    "equal_mass_bin_operator",
    "finalize_lineage_profiles",
    "fisher_z_statistics",
    "kish_effective_sizes",
    "kish_p_values",
    "lineage_profiles",
    "select_lineages",
    "stream_lineage_profiles",
    "stream_weighted_correlations",
    "weighted_correlations",
]

LINEAGE_OK = 0
LINEAGE_UNTESTED = 1
LINEAGE_NONFINITE = 2
# Bytes of scratch per streamed value: the float64 copy of a feature block.
LINEAGE_SCRATCH_ITEMSIZE = np.dtype(np.float64).itemsize


def select_lineages(
    sink_labels: Sequence[Any],
    lineages: Sequence[Any] | None,
    *,
    where: str = "a sink of the fate map",
) -> tuple[tuple[Any, ...], np.ndarray]:
    """Return the requested lineage labels and their columns.

    ``None`` selects every available label in its stored order. Otherwise
    each label must name one available label, once, and the columns follow
    the requested order. ``where`` completes the error for a label that is
    not available, such as ``"in the result"`` for a result's lineages.
    """
    available = tuple(sink_labels)
    if lineages is None:
        return available, np.arange(len(available), dtype=np.int64)
    if not isinstance(lineages, list | tuple):
        raise TypeError("lineages must be a list of lineage labels or None")
    if not lineages:
        raise ValueError("lineages must name at least one lineage")
    columns: list[int] = []
    for label in lineages:
        matches = [index for index, sink in enumerate(available) if bool(sink == label)]
        if not matches:
            raise ValueError(f"Lineage {label!r} is not {where}")
        if matches[0] in columns:
            raise ValueError("lineages must not repeat a lineage")
        columns.append(matches[0])
    return tuple(available[index] for index in columns), np.asarray(
        columns, dtype=np.int64
    )


def kish_effective_sizes(weights: np.ndarray) -> np.ndarray:
    """Return Kish's effective sample size of each weight column.

    For weights ``w`` it is ``(sum w)**2 / sum(w**2)``: the number of equally
    weighted cells that would give a weighted mean the same variance.
    """
    values = np.asarray(weights, dtype=np.float64)
    totals = values.sum(axis=0)
    squares = np.square(values).sum(axis=0)
    sizes = np.zeros(values.shape[1], dtype=np.float64)
    positive = squares > 0.0
    sizes[positive] = np.square(totals[positive]) / squares[positive]
    return sizes


def _centered_regressors(
    weights: np.ndarray,
    regressors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weight totals, weighted-centered regressors, and their sums of squares.

    Each regressor column is first scaled by a power of two below one in
    magnitude. Pearson r does not depend on the scale, and a power-of-two
    scale is exact, so the centered sums stay finite.
    """
    totals = weights.sum(axis=0)
    if np.any(totals <= 0.0):
        raise ValueError("Every lineage needs a positive total weight")
    # One cells-by-lineages copy is scaled and centred in place, and the
    # weighted sums contract without temporaries, so this holds no more than
    # the weights, the regressors, and the result.
    centered = np.array(regressors, dtype=np.float64, order="C", copy=True)
    magnitude = np.maximum(centered.max(axis=0), -centered.min(axis=0))
    np.ldexp(centered, -np.frexp(magnitude)[1], out=centered)
    centered -= np.einsum("ij,ij->j", weights, centered) / totals
    sums_of_squares = np.einsum("ij,ij,ij->j", weights, centered, centered)
    return totals, centered, sums_of_squares


def _prepare_regressors(
    weights: np.ndarray,
    regressors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate weights and regressors and centre the regressors once.

    Returns the C-ordered float64 weights, the lineage totals, the centred
    regressors, and their weighted sums of squares.
    """
    cell_weights = np.ascontiguousarray(weights, dtype=np.float64)
    lineage_regressors = np.asarray(regressors, dtype=np.float64)
    if cell_weights.ndim != 2:
        raise ValueError("values and weights must be two-dimensional")
    if lineage_regressors.shape != cell_weights.shape:
        raise ValueError("regressors must align with the weights")
    if not np.isfinite(cell_weights).all() or np.any(cell_weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isfinite(lineage_regressors).all():
        raise ValueError("regressors must be finite")
    totals, centered, regressor_ss = _centered_regressors(
        cell_weights,
        lineage_regressors,
    )
    return cell_weights, totals, centered, regressor_ss


@njit(parallel=True, cache=True)
def _weighted_correlation_kernel(
    data: np.ndarray,
    weights: np.ndarray,
    centered: np.ndarray,
    regressor_ss: np.ndarray,
    totals: np.ndarray,
    min_mass: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted Pearson r of every feature and lineage, with a status.

    ``data`` holds cells by features, ``weights`` and ``centered`` cells by
    lineages. A feature is untested in a lineage when its positive values
    carry less than ``min_mass`` of the lineage's weight, or when its values
    on the weighted cells span no more than ``eps`` times their magnitude.
    Each feature is scaled by a power of two below one in magnitude before
    its sums are formed, which is exact and leaves r unchanged.
    """
    n_cells = data.shape[0]
    n_features = data.shape[1]
    n_lineages = weights.shape[1]
    r_out = np.zeros((n_features, n_lineages), dtype=np.float64)
    status = np.full((n_features, n_lineages), LINEAGE_UNTESTED, dtype=np.int8)
    for g in prange(n_features):
        magnitude = 0.0
        finite = True
        for c in range(n_cells):
            value = data[c, g]
            if not np.isfinite(value):
                finite = False
                break
            if abs(value) > magnitude:
                magnitude = abs(value)
        if not finite:
            for lineage in range(n_lineages):
                status[g, lineage] = LINEAGE_NONFINITE
            continue
        if magnitude == 0.0:
            continue
        # Values below 2**-1024 need a scale above 2**1023, the largest power
        # of two in float64, so the scale is applied in two exact steps.
        exponent = -math.frexp(magnitude)[1]
        scale = math.ldexp(1.0, min(exponent, 1023))
        scale_rest = math.ldexp(1.0, max(exponent - 1023, 0))
        sums = np.zeros(n_lineages)
        expressing = np.zeros(n_lineages)
        lows = np.full(n_lineages, np.inf)
        highs = np.full(n_lineages, -np.inf)
        for c in range(n_cells):
            value = data[c, g] * scale * scale_rest
            for lineage in range(n_lineages):
                weight = weights[c, lineage]
                if weight > 0.0:
                    sums[lineage] += weight * value
                    if value > 0.0:
                        expressing[lineage] += weight
                    if value < lows[lineage]:
                        lows[lineage] = value
                    if value > highs[lineage]:
                        highs[lineage] = value
        means = sums / totals
        squares = np.zeros(n_lineages)
        cross = np.zeros(n_lineages)
        for c in range(n_cells):
            value = data[c, g] * scale * scale_rest
            for lineage in range(n_lineages):
                weight = weights[c, lineage]
                if weight > 0.0:
                    deviation = value - means[lineage]
                    squares[lineage] += weight * deviation * deviation
                    cross[lineage] += weight * deviation * centered[c, lineage]
        for lineage in range(n_lineages):
            span = highs[lineage] - lows[lineage]
            extent = max(-lows[lineage], highs[lineage])
            if (
                expressing[lineage] < min_mass
                or span <= eps * extent
                or squares[lineage] <= 0.0
                or regressor_ss[lineage] <= 0.0
            ):
                continue
            r = cross[lineage] / math.sqrt(squares[lineage] * regressor_ss[lineage])
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0
            r_out[g, lineage] = r
            status[g, lineage] = LINEAGE_OK
    return r_out, status


@restore_numba_threads
def weighted_correlations(
    values: np.ndarray,
    weights: np.ndarray,
    regressors: np.ndarray,
    *,
    min_mass: float,
    nthreads: int,
    feature_labels: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted Pearson r of each feature with each lineage's regressor.

    Args:
        values: Normalized values, cells by features.
        weights: Non-negative cell weights, cells by lineages.
        regressors: One regressor per lineage, cells by lineages.
        min_mass: Smallest weight that a feature's positive values must carry
            in a lineage for it to be tested there.
        nthreads: Numba threads for the kernel.
        feature_labels: Labels that name a feature in errors.

    Returns:
        Correlations and statuses, features by lineages. Untested entries
        hold zero and :data:`LINEAGE_UNTESTED`.

    Raises:
        ValueError: If a feature holds non-finite values, the weights are
            negative or misaligned, or a lineage has no weight.
    """
    return _correlate_block(
        values,
        _prepare_regressors(weights, regressors),
        min_mass=min_mass,
        nthreads=nthreads,
        feature_labels=feature_labels,
    )


def _correlate_block(
    values: np.ndarray,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    min_mass: float,
    nthreads: int,
    feature_labels: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate one block of features with regressors from :func:`_prepare_regressors`."""
    cell_weights, totals, centered, regressor_ss = prepared
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("values and weights must be two-dimensional")
    if cell_weights.shape[0] != data.shape[0]:
        raise ValueError("weights must have one row per cell")
    if len(feature_labels) != data.shape[1]:
        raise ValueError("feature_labels must name every feature")
    set_num_threads(min(max(1, int(nthreads)), numba.config.NUMBA_NUM_THREADS))
    r_values, status = _weighted_correlation_kernel(
        data,
        cell_weights,
        centered,
        regressor_ss,
        totals,
        float(min_mass),
        float(np.finfo(np.float64).eps),
    )
    bad = np.flatnonzero((status == LINEAGE_NONFINITE).any(axis=1))
    if bad.size:
        raise ValueError(
            f"Feature {feature_labels[bad[0]]!r} contains non-finite normalized values"
        )
    return r_values, status


def _check_block(
    labels: np.ndarray,
    feature_order: np.ndarray,
    start: int,
) -> int:
    stop = start + int(labels.shape[0])
    if stop > feature_order.shape[0] or not np.array_equal(
        labels, feature_order[start:stop]
    ):
        raise RuntimeError(
            "Normalized feature batches must follow the requested features"
        )
    return stop


@restore_numba_threads
def stream_weighted_correlations(
    blocks: Iterable[tuple[np.ndarray, np.ndarray]],
    feature_order: np.ndarray,
    weights: np.ndarray,
    regressors: np.ndarray,
    *,
    min_mass: float,
    nthreads: int,
    feature_names: Sequence[Any] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted correlations of streamed feature blocks.

    ``blocks`` yields cells-by-features values with the feature labels of
    their columns, which must follow ``feature_order`` without gaps. The
    weights and regressors are validated and centred once, before the first
    block, and the centred regressors are the only cells-by-lineages array
    this adds while blocks stream. ``feature_names``, aligned with
    ``feature_order``, name features in errors instead of their labels.

    Returns:
        Correlations and statuses, features by lineages, in ``feature_order``.
    """
    order = np.asarray(feature_order)
    names = order if feature_names is None else np.asarray(feature_names)
    if names.shape != order.shape:
        raise ValueError("feature_names must name every feature in feature_order")
    prepared = _prepare_regressors(weights, regressors)
    n_lineages = prepared[0].shape[1]
    r_values = np.empty((order.shape[0], n_lineages), dtype=np.float64)
    status = np.empty((order.shape[0], n_lineages), dtype=np.int8)
    start = 0
    for values, labels in blocks:
        stop = _check_block(np.asarray(labels), order, start)
        r_values[start:stop], status[start:stop] = _correlate_block(
            values,
            prepared,
            min_mass=min_mass,
            nthreads=nthreads,
            feature_labels=names[start:stop].tolist(),
        )
        start = stop
    if start != order.shape[0]:
        raise RuntimeError("Normalized feature batches must cover every feature")
    return r_values, status


@restore_numba_threads
def stream_lineage_profiles(
    blocks: Iterable[tuple[np.ndarray, np.ndarray]],
    feature_order: np.ndarray,
    operator: spmatrix,
    smoothing: np.ndarray | None,
    n_lineages: int,
    *,
    nthreads: int = 1,
    feature_names: Sequence[Any] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lineage profiles and mean values of streamed feature blocks.

    The operator is converted to columns once for every block; see
    :func:`lineage_profiles`. ``feature_names``, aligned with
    ``feature_order``, name features in errors instead of their labels.

    Returns:
        Profiles, features by lineages by bins, and each feature's mean value
        over all cells, in ``feature_order``.
    """
    order = np.asarray(feature_order)
    names = order if feature_names is None else np.asarray(feature_names)
    if names.shape != order.shape:
        raise ValueError("feature_names must name every feature in feature_order")
    columns = csc_matrix(operator)
    n_bins = columns.shape[0] // n_lineages
    profiles = np.empty((order.shape[0], n_lineages, n_bins), dtype=np.float64)
    means = np.empty(order.shape[0], dtype=np.float64)
    start = 0
    for values, labels in blocks:
        stop = _check_block(np.asarray(labels), order, start)
        block_profiles, totals, finite = _block_profiles(
            values, columns, smoothing, n_lineages, nthreads
        )
        if not finite.all():
            bad = int(np.flatnonzero(~finite)[0])
            label = names[start + bad : start + bad + 1].tolist()[0]
            raise ValueError(f"Feature {label!r} contains non-finite normalized values")
        profiles[start:stop] = block_profiles
        means[start:stop] = totals / columns.shape[1]
        start = stop
    if start != order.shape[0]:
        raise RuntimeError("Normalized feature batches must cover every feature")
    return profiles, means


def kish_p_values(
    r_values: np.ndarray,
    status: np.ndarray,
    effective_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two-sided Student-t p-values on Kish's effective sample size.

    A weighted correlation is tested as if ``effective_sizes`` independent
    cells had produced it, with ``n - 2`` degrees of freedom, by the same
    test as the pseudotime markers. Lineages whose effective size is at most
    two have no p-values, and their features are marked untested. Returns
    the p-values and the updated statuses.
    """
    p_values = np.full(r_values.shape, np.nan, dtype=np.float64)
    updated = np.array(status, dtype=np.int8, copy=True)
    for lineage, size in enumerate(np.asarray(effective_sizes, dtype=np.float64)):
        degrees = float(size) - 2.0
        tested = updated[:, lineage] == LINEAGE_OK
        if degrees <= 0.0:
            updated[tested, lineage] = LINEAGE_UNTESTED
            continue
        p_values[tested, lineage] = _regression_p_values(
            r_values[tested, lineage], float(size)
        )
    return p_values, updated


def fisher_z_statistics(
    r_values: np.ndarray,
    status: np.ndarray,
    n_cells: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Fisher z-test p-values and confidence intervals of correlations.

    The z-transform ``arctanh(r)`` of a Pearson correlation of ``n_cells``
    cells is approximately normal with standard error ``1 / sqrt(n - 3)``.
    With at most three cells nothing is tested. Returns the p-values, the
    lower and upper interval bounds, and the updated statuses.
    """
    shape = r_values.shape
    p_values = np.full(shape, np.nan, dtype=np.float64)
    low = np.full(shape, np.nan, dtype=np.float64)
    high = np.full(shape, np.nan, dtype=np.float64)
    updated = np.array(status, dtype=np.int8, copy=True)
    tested = updated == LINEAGE_OK
    if n_cells <= 3:
        updated[tested] = LINEAGE_UNTESTED
        return p_values, low, high, updated
    bound = 1.0 - np.finfo(np.float64).eps
    z = np.arctanh(np.clip(r_values[tested], -bound, bound))
    error = 1.0 / math.sqrt(n_cells - 3)
    quantile = float(norm.ppf(0.5 + confidence_level / 2.0))
    p_values[tested] = 2.0 * norm.sf(np.abs(z) / error)
    low[tested] = np.tanh(z - quantile * error)
    high[tested] = np.tanh(z + quantile * error)
    return p_values, low, high, updated


def adjust_lineage_p_values(p_values: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values within each lineage.

    The correction of a lineage covers its tested features only; untested
    entries stay missing.
    """
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    for lineage in range(p_values.shape[1]):
        tested = (status[:, lineage] == LINEAGE_OK) & np.isfinite(p_values[:, lineage])
        if np.any(tested):
            adjusted[tested, lineage] = adjust_pvalues(
                p_values[tested, lineage],
                "fdr_bh",
            )
    return adjusted


@njit(cache=True)
def _bin_overlaps(
    order: np.ndarray,
    weights: np.ndarray,
    totals: np.ndarray,
    n_bins: int,
    cursor: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    fill: bool,
) -> None:
    """Count, or write, each cell's shares of the equal-mass bins it covers.

    Every lineage walks the cells in ``order``. Without ``fill``, ``cursor``
    counts each cell's entries. With it, each entry is written at the cell's
    ``cursor`` position of ``indices`` and ``data``, the cell's column of a
    CSC operator, and the cursor advances. Both passes run the same
    arithmetic, so the counts match the entries written.
    """
    for lineage in range(weights.shape[1]):
        position = 0.0
        scale = n_bins / totals[lineage]
        row_offset = lineage * n_bins
        for index in range(order.shape[0]):
            cell = order[index]
            weight = weights[cell, lineage]
            if weight <= 0.0:
                continue
            start = position
            stop = min(position + weight * scale, float(n_bins))
            position = stop
            first = int(math.floor(start))
            last = min(int(math.ceil(stop)) - 1, n_bins - 1)
            for bin_index in range(first, last + 1):
                overlap = min(stop, bin_index + 1.0) - max(start, float(bin_index))
                if overlap > 0.0:
                    if fill:
                        slot = cursor[cell]
                        indices[slot] = row_offset + bin_index
                        data[slot] = overlap
                    cursor[cell] += 1


def bin_operator_bytes(n_cells: int, n_lineages: int, n_bins: int) -> int:
    """Return the most bytes :func:`equal_mass_bin_operator` holds while building.

    That is the pseudotime order, a cursor and the column pointers of every
    cell, and at most one entry per cell and lineage plus one per cut, each a
    64-bit row index and a value.
    """
    capacity = n_lineages * (n_cells + n_bins)
    return 3 * (n_cells + 1) * 8 + capacity * 16


def equal_mass_bin_operator(
    ordering: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> tuple[csc_matrix, np.ndarray, np.ndarray]:
    """Return the operator that averages cells into equal-mass lineage bins.

    Cells are ordered by ``ordering``, ties by position. Along that order,
    each lineage's weight is cut into ``n_bins`` bins of equal mass, and a
    cell whose weight straddles a cut is shared between the bins in
    proportion. Row ``lineage * n_bins + bin`` of the operator holds the
    cells' shares of one bin, which sum to one, so the operator times a
    cells-by-features block gives the weighted mean of every bin. The
    operator is stored by cell, the order in which
    :func:`lineage_profiles` reads it, and its arrays are written in place
    after a counting pass, so building it holds no more than
    :func:`bin_operator_bytes`.

    Returns:
        The operator, lineages times bins by cells; the weighted mean of
        ``ordering`` in every bin, lineages by bins; and each lineage's total
        weight.

    Raises:
        ValueError: If a lineage's total weight is below ``n_bins``, so that
            its bins would hold less than one cell's weight.
    """
    order_values = np.asarray(ordering, dtype=np.float64)
    cell_weights = np.ascontiguousarray(weights, dtype=np.float64)
    n_cells, n_lineages = cell_weights.shape
    if order_values.shape != (n_cells,):
        raise ValueError("ordering must have one value per cell")
    if not np.isfinite(order_values).all():
        raise ValueError("ordering must be finite")
    if not np.isfinite(cell_weights).all() or np.any(cell_weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    totals = cell_weights.sum(axis=0)
    small = np.flatnonzero(totals < n_bins)
    if small.size:
        raise ValueError(
            f"Lineage {int(small[0])} carries a total weight of "
            f"{float(totals[small[0]]):.3g}, below n_bins={n_bins}; "
            "request fewer bins"
        )
    order = np.asarray(np.argsort(order_values, kind="stable"), dtype=np.int64)
    n_rows = n_lineages * int(n_bins)
    cursor = np.zeros(n_cells, dtype=np.int64)
    _bin_overlaps(
        order,
        cell_weights,
        totals,
        int(n_bins),
        cursor,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        False,
    )
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(cursor, out=indptr[1:])
    n_entries = int(indptr[-1])
    # SciPy keeps 32-bit indices whenever they fit, so the arrays are built
    # in its index type and passed without a conversion copy.
    index_dtype = (
        np.int32
        if max(n_entries, n_rows, n_cells) <= np.iinfo(np.int32).max
        else np.int64
    )
    indices = np.empty(n_entries, dtype=index_dtype)
    data = np.empty(n_entries, dtype=np.float64)
    cursor[:] = indptr[:-1]
    _bin_overlaps(order, cell_weights, totals, int(n_bins), cursor, indices, data, True)
    del cursor, order
    operator = csc_matrix(
        (data, indices, indptr.astype(index_dtype, copy=False)),
        shape=(n_rows, n_cells),
    )
    bin_ordering = np.asarray(operator @ order_values).reshape(n_lineages, n_bins)
    return operator, bin_ordering, totals


def bin_smoothing_matrix(n_bins: int, smoothing: float) -> np.ndarray | None:
    """Return a row-normalized Gaussian smoother over bins, or None for none.

    ``smoothing`` is the kernel's standard deviation in bins; weights beyond
    four standard deviations are dropped, and each row is renormalized so
    the first and last bins average only the bins that exist.
    """
    if smoothing == 0.0:
        return None
    radius = int(math.ceil(4.0 * smoothing))
    positions = np.arange(n_bins, dtype=np.float64)
    distance = positions[:, np.newaxis] - positions[np.newaxis, :]
    kernel = np.exp(-0.5 * np.square(distance / smoothing))
    kernel[np.abs(distance) > radius] = 0.0
    kernel /= kernel.sum(axis=1, keepdims=True)
    return kernel


@njit(parallel=True, cache=True)
def _bin_sums(
    feature_major: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return operator products, totals, and finiteness of features by cells.

    One thread takes one feature and reads its cells in storage order. Each
    nonzero value is added, with the cell's operator column as weights, to
    the rows that share the cell. Zeros add nothing and are skipped.
    """
    n_features, n_cells = feature_major.shape
    products = np.zeros((n_features, n_rows), dtype=np.float64)
    totals = np.zeros(n_features, dtype=np.float64)
    finite = np.ones(n_features, dtype=np.bool_)
    for feature in prange(n_features):
        total = 0.0
        all_finite = True
        for cell in range(n_cells):
            value = feature_major[feature, cell]
            if value == 0.0:
                continue
            if not math.isfinite(value):
                all_finite = False
            total += value
            for entry in range(indptr[cell], indptr[cell + 1]):
                products[feature, indices[entry]] += data[entry] * value
        totals[feature] = total
        finite[feature] = all_finite
    return products, totals, finite


def _block_profiles(
    values: np.ndarray,
    columns: csc_matrix,
    smoothing: np.ndarray | None,
    n_lineages: int,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a block's profiles, each feature's total, and its finiteness."""
    # Feature streams yield the transpose of a C-order features-by-cells
    # block, which this reads in place.
    feature_major = np.ascontiguousarray(np.asarray(values, dtype=np.float64).T)
    if feature_major.ndim != 2 or feature_major.shape[1] != columns.shape[1]:
        raise ValueError("values must have one row per operator column")
    set_num_threads(min(max(1, int(nthreads)), numba.config.NUMBA_NUM_THREADS))
    products, totals, finite = _bin_sums(
        feature_major,
        columns.indptr,
        columns.indices,
        np.asarray(columns.data, dtype=np.float64),
        columns.shape[0],
    )
    profiles = products.reshape(
        feature_major.shape[0], n_lineages, columns.shape[0] // n_lineages
    )
    if smoothing is not None:
        profiles = np.einsum("ab,flb->fla", smoothing, profiles)
    return np.ascontiguousarray(profiles), totals, finite


@restore_numba_threads
def lineage_profiles(
    values: np.ndarray,
    operator: spmatrix,
    smoothing: np.ndarray | None,
    n_lineages: int,
    *,
    nthreads: int = 1,
) -> np.ndarray:
    """Return smoothed bin means of a feature block, features by lineages by bins.

    ``values`` holds cells by features. The product with the operator reads
    the operator by cell, so a CSC operator is used as it is and another
    format is converted first. Features are spread over ``nthreads`` Numba
    threads, and a block in Fortran order is read without a copy.
    """
    profiles, _, _ = _block_profiles(
        values, csc_matrix(operator), smoothing, n_lineages, nthreads
    )
    return profiles


def finalize_lineage_profiles(
    profiles: np.ndarray,
    mean_expression: np.ndarray,
    *,
    min_expression: float,
    z_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter features and optionally z-scale their profiles across all lineages.

    A feature is kept when its mean expression over the scored cells exceeds
    ``min_expression`` and its profiles vary. Scaling uses one mean and
    standard deviation per feature over every lineage and bin, so the
    lineages of a feature stay comparable. Rows of dropped features are set
    to zero.
    """
    data = np.array(profiles, dtype=np.float64, copy=True)
    if not np.isfinite(data).all():
        raise ValueError("Lineage profiles contain non-finite values")
    flat = data.reshape(data.shape[0], -1)
    deviation = flat.std(axis=1)
    valid = (
        (np.asarray(mean_expression, dtype=np.float64) > min_expression)
        & np.isfinite(deviation)
        & (deviation > np.finfo(np.float64).eps)
    )
    if z_scale and valid.any():
        centers = flat[valid].mean(axis=1, keepdims=True)
        flat[valid] = (flat[valid] - centers) / deviation[valid, np.newaxis]
    flat[~valid] = 0.0
    return flat.reshape(data.shape), valid
