from typing import Literal

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import ndtr

__all__ = [
    "MARKER_SORT_KEYS",
    "MarkerSortKey",
    "_batch_stats",
    "_exact_marker_p_values",
    "_exact_marker_p_values_gene_major",
    "_marker_stats_batch",
    "_marker_stats_gene_major",
    "_rank_sum_null_counts",
    "batch_rank_scratch_bytes",
    "exact_batch_p_values",
    "exact_rank_scratch_bytes",
    "exact_two_sided_p_value",
    "gene_major_rank_scratch_bytes",
    "mannwhitneyu_from_ranks",
    "marker_sort_key",
    "sort_marker_results",
    "tie_sum",
]

MarkerSortKey = Literal["score", "auc", "p_value", "fold_change"]
# Each statistic that orders a marker table, best first.
MARKER_SORT_KEYS: tuple[MarkerSortKey, ...] = ("score", "auc", "p_value", "fold_change")


def _descending(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.asarray(np.negative(frame[column].to_numpy(dtype=np.float64)))


def marker_sort_key(sort_by: object) -> MarkerSortKey:
    """Return ``sort_by`` when it names a marker order.

    Raises:
        ValueError: If ``sort_by`` is not one of ``MARKER_SORT_KEYS``.
    """
    if isinstance(sort_by, str) and sort_by in MARKER_SORT_KEYS:
        return sort_by
    raise ValueError(
        "sort_by must be one of "
        + ", ".join(repr(key) for key in MARKER_SORT_KEYS)
        + f"; got {sort_by!r}"
    )


def sort_marker_results(
    df: pd.DataFrame, sort_by: MarkerSortKey = "score"
) -> pd.DataFrame:
    """Order markers best first by one statistic, with deterministic ties.

    ``sort_by`` orders by descending ``score``, ``auc``, or ``fold_change``,
    or by ascending ``p_value``. Ties fall to the ascending p-value, except
    for ``"p_value"`` itself, whose ties fall to the descending AUC, so that
    positive markers precede negative ones of equal significance. A table
    with ``neg_log10_p_value`` orders p-values that are equal, such as those
    that underflow to 0, by it too. Remaining ties fall to the feature name,
    when the table has one, and then to the feature index: the
    ``feature_index`` column, or the row index when there is none. NaN fold
    changes come last and infinite ones first.

    A stable NumPy lexsort gives the same order as the equivalent pandas sort,
    without factorizing every key column.

    Raises:
        ValueError: If ``sort_by`` is not one of ``MARKER_SORT_KEYS``.
    """
    sort_by = marker_sort_key(sort_by)
    frame = df if "feature_index" in df.columns else df.assign(feature_index=df.index)
    # np.lexsort sorts by its last key first.
    keys: list[np.ndarray] = [frame["feature_index"].to_numpy()]
    if "feature_name" in frame.columns:
        names = frame["feature_name"].to_numpy()
        keys.append(names.astype(str) if names.dtype == object else names)
    significance = [frame["p_value"].to_numpy(dtype=np.float64)]
    if "neg_log10_p_value" in frame.columns:
        significance.insert(0, _descending(frame, "neg_log10_p_value"))
    if sort_by == "p_value":
        keys.append(_descending(frame, "auc"))
        keys.extend(significance)
    else:
        keys.extend(significance)
        keys.append(_descending(frame, sort_by))
    return frame.iloc[np.lexsort(tuple(keys))]


def tie_sum(values: np.ndarray) -> float:
    """Return the rank tie term, the sum of ``t**3 - t`` over tied value groups.

    Group sizes are cubed in float64 because an int64 cube wraps once a group
    holds more than about 2.1 million equal values.
    """
    _, counts = np.unique(np.asarray(values), return_counts=True)
    tied = counts[counts > 1].astype(np.float64)
    return float(np.sum(tied**3 - tied))


def mannwhitneyu_from_ranks(
    ranked_df: pd.DataFrame,
    groups: np.ndarray,
    group_set: np.ndarray,
) -> pd.DataFrame:
    """Calculate two-sided Mann-Whitney U p-values from precomputed ranks."""
    n_total = len(groups)
    rank_sums = ranked_df.groupby(groups).sum().reindex(group_set)
    group_counts = pd.Series(groups).value_counts().reindex(group_set).values
    tie_corrections = np.zeros(ranked_df.shape[1])

    for col_idx in range(ranked_df.shape[1]):
        ties = tie_sum(ranked_df.iloc[:, col_idx].to_numpy())
        if ties > 0:
            tie_corrections[col_idx] = ties / (n_total * (n_total - 1))

    pvals = {}
    for idx, cluster in enumerate(group_set):
        n1 = group_counts[idx]
        n2 = n_total - n1
        r1 = rank_sums.iloc[idx].values
        u1 = r1 - (n1 * (n1 + 1)) / 2
        mu_u = (n1 * n2) / 2
        sigma_u = np.sqrt((n1 * n2 / 12) * ((n_total + 1) - tie_corrections))
        delta = u1 - mu_u
        z = np.zeros_like(delta, dtype=float)
        np.divide(
            delta - 0.5 * np.sign(delta),
            sigma_u,
            out=z,
            where=sigma_u > 0,
        )
        pvals[cluster] = 2 * ndtr(-np.abs(z))

    return pd.DataFrame(pvals, index=ranked_df.columns).T


@njit(cache=True, nogil=True)
def _write_group_statistics(
    out: np.ndarray,
    sum_g: np.ndarray,
    nz_g: np.ndarray,
    rank_g: np.ndarray,
    drank_g: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    tie_total: float,
) -> None:
    """Write one feature's groups-by-statistics row from its group sums.

    ``rank_g`` and ``drank_g`` hold each group's average-rank and dense-rank
    sums, and ``tie_total`` the feature's rank tie term. Column 5 holds the
    fold change under ``MARKER_FOLD_CHANGE_POLICY`` of
    :mod:`cytearc.features.markers.table`: the group mean over the rest mean
    when both are non-negative and the rest mean is positive, +inf when only
    the group mean is positive, and NaN when both means are 0 or either is
    negative. Column 6 holds the continuity- and tie-corrected Mann-Whitney
    z statistic.
    """
    n_groups = group_counts.shape[0]
    total_sum = 0.0
    total_nz = 0.0
    for x in range(n_groups):
        total_sum += sum_g[x]
        total_nz += nz_g[x]
    rank_total = 0.0
    rank_values = np.empty(n_groups)
    for x in range(n_groups):
        count = group_counts[x]
        rank_values[x] = drank_g[x] / count if count > 0 else 0.0
        rank_total += rank_values[x]
    tie_correction = tie_total / (n_total * (n_total - 1.0)) if n_total > 1 else 0.0

    for x in range(n_groups):
        count = group_counts[x]
        rest = n_total - count
        mean = sum_g[x] / count if count > 0 else 0.0
        mean_rest = (total_sum - sum_g[x]) / rest if rest > 0 else 0.0
        fraction = nz_g[x] / count if count > 0 else 0.0
        fraction_rest = (total_nz - nz_g[x]) / rest if rest > 0 else 0.0
        # Every branch assigns its value, so no case divides by zero, which
        # raises in compiled code.
        if mean < 0.0 or mean_rest < 0.0:
            # Signed values give a ratio with no fold-change reading.
            fold_change = np.nan
        elif mean_rest > 0.0:
            fold_change = mean / mean_rest
        elif mean > 0.0:
            # The feature is absent from every other cell.
            fold_change = np.inf
        else:
            fold_change = np.nan
        score = rank_values[x] / rank_total if rank_total > 0 else 0.0
        n1 = count
        n2 = rest
        rank_sum = rank_g[x]
        u1 = rank_sum - (n1 * (n1 + 1.0)) / 2.0
        mu = (n1 * n2) / 2.0
        variance = (n1 * n2 / 12.0) * ((n_total + 1.0) - tie_correction)
        delta = u1 - mu
        z = (
            (delta - 0.5 * np.sign(delta)) / np.sqrt(variance)
            if variance > 0.0
            else 0.0
        )
        if n1 > 0.0 and n2 > 0.0:
            auc = u1 / (n1 * n2)
        else:
            auc = np.nan
        out[x, 0] = score
        out[x, 1] = mean
        out[x, 2] = mean_rest
        out[x, 3] = fraction
        out[x, 4] = fraction_rest
        out[x, 5] = fold_change
        out[x, 6] = z
        out[x, 7] = auc


@njit(parallel=True, cache=True)
def _marker_stats_batch(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
) -> np.ndarray:
    """Compute per-feature, per-group marker statistics for one batch."""
    n_cells = data.shape[0]
    n_genes = data.shape[1]
    n_groups = group_counts.shape[0]
    out = np.zeros((n_genes, n_groups, 8))
    for g in prange(n_genes):
        v = data[:, g]
        order = np.argsort(v)
        ar = np.empty(n_cells)
        dr = np.empty(n_cells)
        tie_total = 0.0
        i = 0
        rank = 0.0
        while i < n_cells:
            j = i
            vi = v[order[i]]
            while j + 1 < n_cells and v[order[j + 1]] == vi:
                j += 1
            avg = (i + j + 2) / 2.0
            rank += 1.0
            t = j - i + 1
            t_float = float(t)
            for k in range(i, j + 1):
                ar[order[k]] = avg
                dr[order[k]] = rank
            if t > 1:
                tie_total += t_float * t_float * t_float - t_float
            i = j + 1
        sum_g = np.zeros(n_groups)
        nz_g = np.zeros(n_groups)
        rank_g = np.zeros(n_groups)
        drank_g = np.zeros(n_groups)
        for c in range(n_cells):
            grp = int_indices[c]
            val = v[c]
            sum_g[grp] += val
            if val > 0:
                nz_g[grp] += 1.0
            rank_g[grp] += ar[c]
            drank_g[grp] += dr[c]
        _write_group_statistics(
            out[g],
            sum_g,
            nz_g,
            rank_g,
            drank_g,
            group_counts,
            n_total,
            tie_total,
        )
    return out


@njit(cache=True, nogil=True)
def _argsort_positive(
    values: np.ndarray,
    n: int,
    order: np.ndarray,
    scratch: np.ndarray,
    buckets: np.ndarray,
) -> None:
    """Stably argsort ``values[:n]`` into ``order[:n]``.

    Positive float32 values sort like their bit patterns read as unsigned
    integers, so three counting passes of 11, 11, and 10 bits replace a
    comparison sort.
    """
    keys = values.view(np.uint32)
    for index in range(n):
        order[index] = index
    source = order
    target = scratch
    shift = 0
    for bits in (11, 11, 10):
        mask = np.uint32((1 << bits) - 1)
        buckets[: mask + 1] = 0
        for index in range(n):
            buckets[(keys[source[index]] >> np.uint32(shift)) & mask] += 1
        total = 0
        for bucket in range(mask + 1):
            count = buckets[bucket]
            buckets[bucket] = total
            total += count
        for index in range(n):
            position = source[index]
            bucket = (keys[position] >> np.uint32(shift)) & mask
            target[buckets[bucket]] = position
            buckets[bucket] += 1
        source, target = target, source
        shift += bits
    if source is not order:
        order[:n] = source[:n]


@njit(cache=True, nogil=True, inline="always")
def _library_size_value(
    count: float,
    total: float,
    size_factor: float,
    log_transform: bool,
) -> np.float32:
    """Return one library-size value as the gene-major kernel ranks it.

    The value is computed in float64 and rounded once to float32.
    """
    exact = size_factor * np.float64(count) / total
    if log_transform:
        exact = np.log1p(exact)
    return np.float32(exact)


@njit(cache=True, nogil=True)
def _gene_major_feature(
    counts: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    out: np.ndarray,
    nz_values: np.ndarray,
    nz_cells: np.ndarray,
    order: np.ndarray,
    order_scratch: np.ndarray,
    buckets: np.ndarray,
    zero_g: np.ndarray,
) -> bool:
    """Write one feature's library-size statistics from its raw counts.

    Each value is computed in float64 and ranked and summed as its float32
    rounding. Zero counts share the lowest rank, so only nonzero values are
    sorted, which holds only while no value is negative. Returns False,
    leaving ``out`` unwritten, when a value is negative or not finite.
    """
    n_cells = counts.shape[0]
    n_groups = group_counts.shape[0]
    sum_g = np.zeros(n_groups)
    nz_g = np.zeros(n_groups)
    rank_g = np.zeros(n_groups)
    drank_g = np.zeros(n_groups)
    n_nz = 0
    for c in range(n_cells):
        count = counts[c]
        if count == 0:
            continue
        value = _library_size_value(count, totals[c], size_factor, log_transform)
        if value > 0.0:
            grp = int_indices[c]
            nz_values[n_nz] = value
            nz_cells[n_nz] = c
            n_nz += 1
            sum_g[grp] += value
            nz_g[grp] += 1.0
        elif value != 0.0:
            # Negative or NaN; a positive value that rounds to zero is a zero.
            return False
    for x in range(n_groups):
        # Only an infinite value makes a sum of positive float32 values infinite.
        if not sum_g[x] < np.inf:
            return False
        zero_g[x] = group_counts[x] - nz_g[x]

    n_zero = n_cells - n_nz
    tie_total = 0.0
    if n_zero > 0:
        zero_rank = (n_zero + 1.0) / 2.0
        zero_t = float(n_zero)
        if n_zero > 1:
            tie_total += zero_t * zero_t * zero_t - zero_t
        for x in range(n_groups):
            rank_g[x] = zero_g[x] * zero_rank
            drank_g[x] = zero_g[x]

    _argsort_positive(nz_values, n_nz, order, order_scratch, buckets)
    i = 0
    dense_rank = 1.0 if n_zero > 0 else 0.0
    while i < n_nz:
        j = i
        value = nz_values[order[i]]
        while j + 1 < n_nz and nz_values[order[j + 1]] == value:
            j += 1
        dense_rank += 1.0
        average_rank = n_zero + (i + j + 2.0) / 2.0
        tied = j - i + 1
        tied_float = float(tied)
        if tied > 1:
            tie_total += tied_float * tied_float * tied_float - tied_float
        for k in range(i, j + 1):
            cell = nz_cells[order[k]]
            grp = int_indices[cell]
            rank_g[grp] += average_rank
            drank_g[grp] += dense_rank
        i = j + 1

    _write_group_statistics(
        out,
        sum_g,
        nz_g,
        rank_g,
        drank_g,
        group_counts,
        n_total,
        tie_total,
    )
    return True


@njit(cache=True, nogil=True)
def _gene_major_slot(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    destination_rows: np.ndarray,
    rows: np.ndarray,
    slot: int,
    n_slots: int,
    out: np.ndarray,
) -> int:
    """Write the statistics of every ``n_slots``-th row of ``rows`` from ``slot``.

    The slot owns one set of per-cell scratch arrays. Returns the position in
    ``rows`` of the first row with a negative or non-finite normalized value,
    or the number of rows when every value is valid.
    """
    n_cells = raw.shape[1]
    nz_values = np.empty(n_cells, dtype=np.float32)
    nz_cells = np.empty(n_cells, dtype=np.int64)
    order = np.empty(n_cells, dtype=np.int64)
    order_scratch = np.empty(n_cells, dtype=np.int64)
    buckets = np.empty(2048, dtype=np.int64)
    zero_g = np.zeros(group_counts.shape[0])
    for position in range(slot, rows.shape[0], n_slots):
        row = rows[position]
        if not _gene_major_feature(
            raw[row],
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            out[destination_rows[row]],
            nz_values,
            nz_cells,
            order,
            order_scratch,
            buckets,
            zero_g,
        ):
            return position
    return int(rows.shape[0])


@njit(parallel=True, cache=True, nogil=True)
def _gene_major_slots(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    destination_rows: np.ndarray,
    rows: np.ndarray,
    n_slots: int,
    out: np.ndarray,
) -> int:
    """Run ``n_slots`` slots of ``rows`` on Numba threads.

    Rows run in increasing order within a slot, so the smallest position a
    slot returns is that of the first invalid row.
    """
    first_invalid = np.empty(n_slots, dtype=np.int64)
    for slot in prange(n_slots):
        first_invalid[slot] = _gene_major_slot(
            raw,
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            destination_rows,
            rows,
            slot,
            n_slots,
            out,
        )
    return int(first_invalid.min())


@njit(cache=True, nogil=True)
def _marker_stats_gene_major(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    destination_rows: np.ndarray,
    threads: int,
    out: np.ndarray,
) -> int:
    """Write library-size marker statistics from feature-major raw counts.

    Raw counts of any dtype are normalized by the float64 cell ``totals``.
    Up to ``threads`` slots each own one set of per-cell scratch arrays and
    take every slot-count-th selected row, and every row is computed alone,
    so results do not depend on ``threads``. One slot runs on the calling
    thread without Numba's parallel runtime, so concurrent callers can each
    run one. Rows with a negative ``destination_rows`` entry are skipped.
    Returns the local row of the first feature with a negative or non-finite
    normalized value, or -1 when every value is valid.
    """
    rows = np.flatnonzero(destination_rows >= 0)
    n_rows = rows.shape[0]
    n_slots = min(max(1, threads), n_rows)
    if n_slots > 1:
        first = _gene_major_slots(
            raw,
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            destination_rows,
            rows,
            n_slots,
            out,
        )
    else:
        first = _gene_major_slot(
            raw,
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            destination_rows,
            rows,
            0,
            1,
            out,
        )
    return int(rows[first]) if first < n_rows else -1


def gene_major_rank_scratch_bytes(
    *,
    n_cells: int,
    n_groups: int,
    n_features: int,
    nthreads: int,
) -> int:
    """Return the scratch of one gene-major call over ``n_features`` rows.

    Each of its ``nthreads`` threads holds the nonzero values, their cells,
    and the radix sort's order arrays for every cell.
    """
    cells = max(0, int(n_cells))
    groups = max(0, int(n_groups))
    threads = max(1, int(nthreads))
    int64 = np.dtype(np.int64).itemsize
    # Values, their cells, and the radix sort's order and scratch arrays.
    per_thread = (
        cells * (np.dtype(np.float32).itemsize + 3 * int64)
        + 2048 * int64
        + groups * 6 * np.dtype(np.float64).itemsize
    )
    # The selected rows and each thread's first invalid row.
    per_call = (max(0, int(n_features)) + threads) * int64
    return per_call + threads * per_thread


def batch_rank_scratch_bytes(
    *,
    n_cells: int,
    n_groups: int,
    n_features: int,
    nthreads: int,
) -> int:
    """Return the scratch of one dense rank call over ``n_features`` features.

    The call returns eight statistics per feature and group, and each of its
    ``nthreads`` threads ranks one feature at a time.
    """
    cells = max(0, int(n_cells))
    groups = max(0, int(n_groups))
    float64 = np.dtype(np.float64).itemsize
    # The argsort order and the average and dense ranks of every cell.
    per_thread = cells * (np.dtype(np.int64).itemsize + 2 * float64) + (
        groups * 5 * float64
    )
    statistics = max(0, int(n_features)) * groups * 8 * float64
    return statistics + max(1, int(nthreads)) * per_thread


def _batch_stats(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: int,
    feature_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Run the dense marker kernel over cells-by-features normalized values.

    Column 5 of the result holds the fold change, which is +inf or NaN where
    a ratio of the means is undefined, and column 6 the Mann-Whitney z
    statistic.
    """
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("Marker data must be a two-dimensional array")
    labels = (
        np.arange(values.shape[1])
        if feature_labels is None
        else np.asarray(feature_labels)
    )
    if labels.shape != (values.shape[1],):
        raise ValueError("Feature labels must match the marker data columns")
    for column in range(values.shape[1]):
        if not np.isfinite(values[:, column]).all():
            raise ValueError(
                f"Feature {labels[column]!r} contains non-finite normalized values"
            )
    out = _marker_stats_batch(
        _kernel_values(values),
        int_indices,
        group_counts.astype(np.float64),
        float(n_total),
    )
    return np.asarray(out)


def _kernel_values(values: np.ndarray) -> np.ndarray:
    """Return values as the dense kernels rank them: float32 kept, else float64."""
    kernel_dtype = np.float32 if values.dtype == np.float32 else np.float64
    return np.ascontiguousarray(values, dtype=kernel_dtype)


def exact_batch_p_values(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_sizes: np.ndarray,
    exact_groups: np.ndarray,
    max_size: int,
) -> np.ndarray:
    """Return exact p-values of cells-by-features values, ranked as ``_batch_stats`` ranks them.

    Returns:
        Features-by-groups p-values of ``exact_groups``, NaN for other groups.
    """
    values = np.asarray(data)
    out = np.full((values.shape[1], len(group_sizes)), np.nan)
    _exact_marker_p_values(
        _kernel_values(values),
        int_indices,
        np.asarray(group_sizes, dtype=np.int64),
        exact_groups,
        max_size,
        out,
    )
    return out


# Exact Mann-Whitney p-values. Midranks are multiples of one half, so doubled
# midranks are integers and the null distribution of a group's doubled rank
# sum can be counted exactly, ties included.


@njit(cache=True, nogil=True)
def _rank_sum_null_counts(
    values: np.ndarray,
    multiplicities: np.ndarray,
    max_size: int,
    width: int,
) -> np.ndarray:
    """Count the subsets of each size up to ``max_size`` by their rank sum.

    ``values`` holds the distinct doubled midranks of the pooled observations
    and ``multiplicities`` how many observations share each. Row ``k`` of the
    result counts the ``k``-subsets of the observations by the sum of their
    doubled midranks, from 0 to ``width - 1``; when ``width`` exceeds the sum
    of the ``max_size`` largest doubled midranks, row ``k`` holds all
    ``C(n, k)`` subsets. Choosing ``t`` of ``m`` tied observations can be
    done in ``C(m, t)`` ways, so a tie adds its observations at once.
    """
    counts = np.zeros((max_size + 1, width), dtype=np.int64)
    counts[0, 0] = 1
    for position in range(values.shape[0]):
        value = values[position]
        tied = multiplicities[position]
        # Larger subsets first, so that each reads counts without this value.
        for size in range(max_size, 0, -1):
            target = counts[size]
            ways = 1
            for taken in range(1, min(tied, size) + 1):
                ways = ways * (tied - taken + 1) // taken
                shift = taken * value
                source = counts[size - taken]
                for total in range(width - shift):
                    target[total + shift] += ways * source[total]
    return counts


@njit(cache=True, nogil=True)
def exact_two_sided_p_value(distribution: np.ndarray, observed: int) -> float:
    """Return the two-sided p-value of ``observed`` under counted sums.

    ``distribution`` counts the splits of the null by their doubled rank sum.
    The p-value doubles the smaller tail, as SciPy's permutation tests do,
    and is at most 1.
    """
    lower = 0
    upper = 0
    total = 0
    for value in range(distribution.shape[0]):
        count = distribution[value]
        total += count
        if value <= observed:
            lower += count
        if value >= observed:
            upper += count
    return min(1.0, 2.0 * min(lower, upper) / total)


@njit(cache=True, nogil=True)
def _exact_feature_p_values(
    values: np.ndarray,
    int_indices: np.ndarray,
    group_sizes: np.ndarray,
    exact_groups: np.ndarray,
    max_size: int,
    out: np.ndarray,
) -> None:
    """Write exact two-sided Mann-Whitney p-values of one feature's groups.

    Each group of ``exact_groups`` is compared with the rest of the cells
    through the smaller of the two, which holds at most ``max_size`` cells:
    its doubled rank sum is ranked against every split of the pooled doubled
    midranks into sides of the same sizes, ties included. ``out`` receives
    one p-value per group of ``exact_groups`` and keeps its other entries.
    """
    n_cells = values.shape[0]
    order = np.argsort(values)
    doubled = np.empty(n_cells, dtype=np.int64)
    distinct = np.empty(n_cells, dtype=np.int64)
    multiplicities = np.empty(n_cells, dtype=np.int64)
    n_distinct = 0
    start = 0
    while start < n_cells:
        stop = start
        value = values[order[start]]
        while stop + 1 < n_cells and values[order[stop + 1]] == value:
            stop += 1
        # Twice the midrank of sorted positions start to stop, counted from 1.
        rank = start + stop + 2
        for position in range(start, stop + 1):
            doubled[order[position]] = rank
        distinct[n_distinct] = rank
        multiplicities[n_distinct] = stop - start + 1
        n_distinct += 1
        start = stop + 1
    sums = np.zeros(group_sizes.shape[0], dtype=np.int64)
    for cell in range(n_cells):
        sums[int_indices[cell]] += doubled[cell]
    # No subset of max_size observations sums past the max_size largest.
    width = 1
    remaining = max_size
    position = n_distinct - 1
    while remaining > 0:
        taken = min(remaining, multiplicities[position])
        width += taken * distinct[position]
        remaining -= taken
        position -= 1
    counts = _rank_sum_null_counts(
        distinct[:n_distinct], multiplicities[:n_distinct], max_size, width
    )
    # The doubled ranks of all cells sum to n_cells * (n_cells + 1).
    total = n_cells * (n_cells + 1)
    for index in range(exact_groups.shape[0]):
        group = exact_groups[index]
        size = group_sizes[group]
        observed = sums[group]
        if n_cells - size < size:
            # The rest is the smaller side, and its sum mirrors the group's.
            size = n_cells - size
            observed = total - observed
        out[group] = exact_two_sided_p_value(counts[size], observed)


@njit(cache=True, nogil=True)
def _exact_marker_p_values(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_sizes: np.ndarray,
    exact_groups: np.ndarray,
    max_size: int,
    out: np.ndarray,
) -> None:
    """Write exact p-values of each feature of cells-by-features values.

    Values are ranked as the dense kernel ranks them; row ``f`` of ``out``
    receives the p-values of feature ``f``.
    """
    for feature in range(data.shape[1]):
        _exact_feature_p_values(
            data[:, feature],
            int_indices,
            group_sizes,
            exact_groups,
            max_size,
            out[feature],
        )


@njit(cache=True, nogil=True)
def _exact_marker_p_values_gene_major(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_sizes: np.ndarray,
    exact_groups: np.ndarray,
    max_size: int,
    destination_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    """Write exact p-values of the library-size values of feature-major counts.

    Each row with a non-negative ``destination_rows`` entry is ranked as
    the float32 values that the gene-major kernel ranks, with zero for a
    zero count, and its p-values go to that row of ``out``. The kernel has
    validated the values already.
    """
    n_cells = raw.shape[1]
    values = np.empty(n_cells, dtype=np.float32)
    for row in range(raw.shape[0]):
        destination = destination_rows[row]
        if destination < 0:
            continue
        counts = raw[row]
        for cell in range(n_cells):
            count = counts[cell]
            values[cell] = (
                _library_size_value(count, totals[cell], size_factor, log_transform)
                if count != 0
                else np.float32(0.0)
            )
        _exact_feature_p_values(
            values,
            int_indices,
            group_sizes,
            exact_groups,
            max_size,
            out[destination],
        )


def exact_rank_scratch_bytes(*, n_cells: int, n_groups: int, max_size: int) -> int:
    """Return what one exact p-value call holds while it ranks one feature.

    It holds the order, doubled midranks, distinct midranks, and their
    multiplicities for every cell, a float32 value of every cell for
    library-size counts, the doubled rank sum of every group, and the counts
    of subsets of up to ``max_size`` cells by doubled rank sums, which reach
    at most ``2 * max_size * n_cells``.
    """
    cells = max(0, int(n_cells))
    size = max(0, int(max_size))
    int64 = np.dtype(np.int64).itemsize
    counts = (size + 1) * (2 * size * cells + 1) * int64
    return (
        cells * (4 * int64 + np.dtype(np.float32).itemsize)
        + max(0, int(n_groups)) * int64
        + counts
    )
