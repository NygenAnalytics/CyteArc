"""Regulon specificity and co-activity modules.

The regulon specificity scores and the connection specificity index need a
regulon's activity only through sums over cells, so they read activity in
blocks of cells. ``DataStore.regulons.specificity`` and
``DataStore.regulons.connection_specificity`` stream a stored score artifact
through the same kernels that the functions here run on a table in memory.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from ..utils.arguments import float_argument, integer_argument
from ..utils.numba import njit

__all__ = [
    "connection_specificity",
    "regulon_modules",
    "regulon_specificity",
]

_FLOAT_BYTES = np.dtype(np.float64).itemsize
#: The float64 bytes of one block of rows of a table in memory, which the
#: functions here read one block at a time.
_TABLE_BLOCK_BYTES = 16 * 2**20


def _table_blocks(values: np.ndarray) -> Iterator[np.ndarray]:
    """Yield C-ordered blocks of rows of a cells-by-regulons table.

    Each block is float32 or float64 as the table is, and float64 otherwise.
    """
    dtype = values.dtype if values.dtype in (np.float32, np.float64) else np.float64
    rows = max(1, _TABLE_BLOCK_BYTES // (_FLOAT_BYTES * max(1, values.shape[1])))
    for start in range(0, values.shape[0], rows):
        yield np.ascontiguousarray(values[start : start + rows], dtype=dtype)


@njit(cache=True, nogil=True)
def _add_totals(values: np.ndarray, totals: np.ndarray) -> bool:
    """Add each column's sum to ``totals``; False at a negative or non-finite value."""
    n_rows, n_columns = values.shape
    for row in range(n_rows):
        for column in range(n_columns):
            value = np.float64(values[row, column])
            if not np.isfinite(value) or value < 0.0:
                return False
            totals[column] += value
    return True


@njit(cache=True, nogil=True)
def _add_group_terms(
    values: np.ndarray,
    codes: np.ndarray,
    totals: np.ndarray,
    weights: np.ndarray,
    in_group: np.ndarray,
    terms: np.ndarray,
) -> None:
    """Add each cell's share of activity, and its in-group divergence terms, to its group.

    A cell of group ``g`` adds ``p`` to ``in_group[g]`` and
    ``p ln(p / m) + q ln(q / m)`` to ``terms[g]``, with ``p`` its activity over
    the regulon's total, ``q = weights[g]`` and ``m = (p + q) / 2``. A cell in
    no group adds ``p`` to the last row of ``in_group``.
    """
    n_rows, n_columns = values.shape
    outside = in_group.shape[0] - 1
    for row in range(n_rows):
        group = codes[row]
        for column in range(n_columns):
            total = totals[column]
            if total <= 0.0:
                continue
            p = np.float64(values[row, column]) / total
            if group < 0:
                in_group[outside, column] += p
                continue
            q = weights[group]
            m = (p + q) / 2.0
            term = q * np.log(q / m)
            if p > 0.0:
                term += p * np.log(p / m)
            in_group[group, column] += p
            terms[group, column] += term


def specificity_bytes(n_groups: int, n_regulons: int) -> int:
    """Return the bytes that :func:`specificity_scores` holds beside the blocks it reads.

    Each regulon's total, the sums of each group and regulon and of the
    cells in no group, and the scores it returns, all float64.
    """
    sums = n_regulons * (1 + (n_groups + 1) + n_groups) + 2 * n_groups
    return _FLOAT_BYTES * (sums + n_groups * n_regulons)


def specificity_scores(
    blocks: Callable[[], Iterable[np.ndarray]],
    codes: np.ndarray,
    n_groups: int,
    n_regulons: int,
    *,
    name: str = "activity",
) -> np.ndarray:
    """Return the regulon specificity score of each group and regulon from blocks of cells.

    With ``p`` a regulon's activity in each cell divided by its total over
    all cells, ``q`` equal to ``1 / n_g`` on the ``n_g`` cells of group ``g``
    and 0 elsewhere, and ``m = (p + q) / 2``, the Jensen-Shannon divergence
    of ``p`` and ``q`` is half the sum over cells of
    ``p ln(p / m) + q ln(q / m)``. A cell outside the group adds ``p ln 2``
    to that sum, so the divergence is ``0.5 * (ln 2 * (1 - P) + S)``, where
    ``P`` sums ``p`` over the group's cells and ``S`` their terms of the sum,
    and the score is ``1 - sqrt(divergence)``. One pass over the
    blocks sums each regulon's activity, and a second sums ``P`` and ``S``
    of each group and regulon, each cell adding to its own group, so the
    sums hold two values for each group and regulon whatever the cells.
    The share of a regulon's activity outside a group is the sum of every
    group's ``P`` and of the cells in no group, less the group's own, which
    is exactly 0 for a regulon active only in the group, and a divergence
    that rounding leaves below 0 counts as 0.

    Args:
        blocks: Called once for each pass, it returns the cells' activity
            in blocks of rows, cells by regulons, in the order of ``codes``.
        codes: Each cell's group, from 0 to ``n_groups - 1``, or -1 for a
            cell in no group, whose activity counts in each regulon's total
            only.
        n_groups: The groups.
        n_regulons: The regulons, the columns of every block.
        name: What holds the activity, as errors name it.

    Returns:
        A groups-by-regulons float64 array, NaN for a regulon without
        activity and for a group without cells.

    Raises:
        ValueError: If a value of activity is negative or not finite, or the
            blocks do not hold one row for each code.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 1 or (
        len(codes) and (int(codes.min()) < -1 or int(codes.max()) >= n_groups)
    ):
        raise ValueError(f"codes must name groups from 0 to {n_groups - 1}, or -1")
    sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
    totals = np.zeros(n_regulons, dtype=np.float64)
    in_group = np.zeros((n_groups + 1, n_regulons), dtype=np.float64)
    terms = np.zeros((n_groups, n_regulons), dtype=np.float64)
    with np.errstate(divide="ignore"):
        weights = 1.0 / sizes.astype(np.float64)

    def checked(block: np.ndarray) -> np.ndarray:
        if block.ndim != 2 or block.shape[1] != n_regulons:
            raise ValueError(f"Each block of {name} must hold {n_regulons} columns")
        return block

    # Each block is let go before the next is read, so a stream holds one.
    rows = 0
    for block in blocks():
        if not _add_totals(checked(block), totals):
            raise ValueError(f"{name} must be finite and non-negative")
        rows += block.shape[0]
        del block
    if rows != len(codes):
        raise ValueError(f"{name} holds {rows} rows for {len(codes)} cells")
    start = 0
    for block in blocks():
        stop = start + checked(block).shape[0]
        if stop > len(codes):
            raise ValueError(f"{name} holds more rows than its {len(codes)} cells")
        _add_group_terms(block, codes[start:stop], totals, weights, in_group, terms)
        start = stop
        del block
    if start != len(codes):
        raise ValueError(f"{name} holds {start} rows for {len(codes)} cells")
    # The share of each regulon's activity outside each group, then the
    # divergence and the score, in the array that is returned.
    scores = in_group.sum(axis=0)[None, :] - in_group[:n_groups]
    scores *= np.log(2.0)
    scores += terms
    scores *= 0.5
    np.maximum(scores, 0.0, out=scores)
    np.sqrt(scores, out=scores)
    np.subtract(1.0, scores, out=scores)
    scores[:, totals <= 0.0] = np.nan
    scores[sizes == 0, :] = np.nan
    return scores


def regulon_specificity(
    activity: pd.DataFrame,
    labels: Sequence[object] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Regulon specificity scores (RSS) of each regulon for each group.

    ``RSS = 1 - JSD(p, q) ** 0.5`` where ``p`` is the regulon's activity over
    cells normalized to sum to one, ``q`` the group's indicator normalized the
    same way, and ``JSD`` the Jensen-Shannon divergence with natural
    logarithms, as pySCENIC's ``regulon_specificity_scores`` computes it. The
    score is 1 when the regulon is active only, and uniformly, in the group.

    The divergence follows from sums over cells, which this adds up block by
    block of rows; :meth:`ds.regulons.specificity
    <cytearc.datastore.namespaces.RegulonsAccessor.specificity>` computes the
    same scores from a stored score artifact without reading it whole.

    Args:
        activity: Cells-by-regulons non-negative activity, such as AUCell AUCs.
        labels: Group of each cell, aligned by position with ``activity``.

    Returns:
        A groups-by-regulons DataFrame, groups in order of first appearance.
        A regulon without activity scores NaN.
    """
    values = np.asarray(activity)
    groups = np.asarray(labels, dtype=object)
    if values.ndim != 2 or len(groups) != values.shape[0]:
        raise ValueError("labels must align with the activity rows")
    order = list(dict.fromkeys(groups.tolist()))
    codes = np.full(len(groups), -1, dtype=np.int64)
    for code, group in enumerate(order):
        codes[groups == group] = code
    scores = specificity_scores(
        lambda: _table_blocks(values), codes, len(order), values.shape[1]
    )
    return pd.DataFrame(
        scores,
        index=pd.Index(order, name="group"),
        columns=list(activity.columns),
        copy=False,
    )


def correlation_bytes(n_regulons: int) -> int:
    """Return the bytes that :func:`activity_correlation` holds beside the blocks it reads.

    The sums of products of deviations of each pair of regulons, which
    become the correlations it returns, and the products of one block, all
    float64, with each regulon's mean and its change.
    """
    return _FLOAT_BYTES * (2 * n_regulons * n_regulons + 3 * n_regulons)


def correlation_bytes_per_row(n_regulons: int) -> int:
    """Return the bytes that :func:`activity_correlation` makes of each row of a block: its float64 deviations."""
    return _FLOAT_BYTES * n_regulons


def activity_correlation(blocks: Iterable[np.ndarray], n_regulons: int) -> np.ndarray:
    """Return the Pearson correlation of each pair of regulons' activity from blocks of cells.

    Each block's sums of products of deviations from its own means merge into
    the running sums with the update of Chan, Golub, and LeVeque, which adds
    the product of the changes of the means weighted by the cells on each
    side, so no product of large sums is subtracted. The correlations then
    follow as ``numpy.corrcoef`` computes them from the covariances, clipped
    to [-1, 1]. A pair that holds a regulon of constant activity, whose
    correlation is undefined, correlates 0, as the connection specificity
    index reads it.

    Args:
        blocks: The cells' activity in blocks of rows, cells by regulons.
        n_regulons: The regulons, the columns of every block.

    Returns:
        A symmetric regulons-by-regulons float64 array.

    Raises:
        ValueError: If a block does not hold ``n_regulons`` columns.
    """
    count = 0
    mean = np.zeros(n_regulons, dtype=np.float64)
    comoments = np.zeros((n_regulons, n_regulons), dtype=np.float64)
    products = np.empty_like(comoments)
    for block in blocks:
        if block.ndim != 2 or block.shape[1] != n_regulons:
            raise ValueError(f"Each block of activity must hold {n_regulons} columns")
        deviations = np.array(block, dtype=np.float64)
        # The block is let go before the next is read, so a stream holds one.
        del block
        n_rows = deviations.shape[0]
        if n_rows == 0:
            continue
        block_mean = deviations.mean(axis=0)
        deviations -= block_mean
        np.matmul(deviations.T, deviations, out=products)
        del deviations
        comoments += products
        if count:
            merged = count + n_rows
            change = block_mean - mean
            np.multiply.outer(change, change, out=products)
            products *= count * n_rows / merged
            comoments += products
            mean += change * (n_rows / merged)
        else:
            mean = block_mean
        count += n_rows
    del products
    with np.errstate(divide="ignore", invalid="ignore"):
        comoments *= np.true_divide(1, count - 1)
        deviation = np.sqrt(np.diag(comoments))
        comoments /= deviation[:, None]
        comoments /= deviation[None, :]
    np.clip(comoments, -1.0, 1.0, out=comoments)
    return np.nan_to_num(comoments, copy=False, nan=0.0)


@njit(cache=True, nogil=True)
def _connection_specificity_kernel(
    correlation: np.ndarray, margin: float, result: np.ndarray
) -> None:
    """Fill ``result`` with the index of each pair, from the first's row and the pair's correlation."""
    n = correlation.shape[0]
    for first in range(n):
        for second in range(first, n):
            threshold = correlation[first, second] - margin
            below = 0
            for other in range(n):
                if (
                    correlation[first, other] < threshold
                    and correlation[second, other] < threshold
                ):
                    below += 1
            share = below / n
            result[first, second] = share
            result[second, first] = share


def connection_specificity_from_correlation(
    correlation: np.ndarray, margin: float
) -> np.ndarray:
    """Return the connection specificity index of each pair of regulons from their correlations.

    Args:
        correlation: A symmetric regulons-by-regulons array, such as
            :func:`activity_correlation` returns.
        margin: The correlation margin of :func:`connection_specificity`.

    Returns:
        A symmetric regulons-by-regulons float64 array.

    Raises:
        TypeError: If ``margin`` is not a real number.
        ValueError: If ``margin`` is not finite or ``correlation`` is not
            square.
    """
    margin = float_argument(margin, "margin")
    values = np.ascontiguousarray(correlation, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("correlation must be a square matrix")
    result = np.empty_like(values)
    _connection_specificity_kernel(values, margin, result)
    return result


def connection_specificity(
    activity: pd.DataFrame, *, margin: float = 0.05
) -> pd.DataFrame:
    """Connection specificity index (CSI) of every pair of regulons.

    For regulons ``a`` and ``b`` with activity correlation ``r``, the CSI is
    the fraction of all regulons whose correlation with ``a`` and with ``b``
    are both below ``r - margin`` (Fuxman Bass et al. 2013). Regulons that
    co-vary with each other but not with the rest score near 1, which SCENIC
    analyses use to group regulons into modules.

    The correlations follow from sums over cells, which this adds up block by
    block of rows; :meth:`ds.regulons.connection_specificity
    <cytearc.datastore.namespaces.RegulonsAccessor.connection_specificity>`
    computes the same index from a stored score artifact without reading it
    whole. A regulon of constant activity correlates 0 with every regulon.

    Args:
        activity: Cells-by-regulons activity.
        margin: Correlation margin.

    Returns:
        A symmetric regulons-by-regulons DataFrame.
    """
    margin = float_argument(margin, "margin")
    values = np.asarray(activity)
    if values.ndim != 2:
        raise ValueError("activity must be a cells-by-regulons table")
    correlation = activity_correlation(_table_blocks(values), values.shape[1])
    names = list(activity.columns)
    return pd.DataFrame(
        connection_specificity_from_correlation(correlation, margin),
        index=names,
        columns=names,
        copy=False,
    )


def regulon_modules(csi: pd.DataFrame, n_modules: int) -> pd.Series:
    """Group regulons into co-activity modules by clustering their CSI profiles.

    Rows of the CSI matrix are clustered with Ward linkage and the tree is cut
    into ``n_modules`` clusters, numbered by decreasing size.

    Args:
        csi: Output of ``connection_specificity``.
        n_modules: Number of modules.

    Returns:
        Module number of each regulon, indexed by regulon.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    integer_argument(n_modules, "n_modules", minimum=1)
    if len(csi) < 2:
        return pd.Series(
            np.ones(len(csi), dtype=np.int64), index=csi.index, name="module"
        )
    tree = linkage(csi.to_numpy(dtype=np.float64), method="ward")
    labels = fcluster(tree, t=min(n_modules, len(csi)), criterion="maxclust")
    sizes = pd.Series(labels).value_counts()
    renumber = {
        old: new
        for new, old in enumerate(
            sorted(sizes.index, key=lambda label: (-sizes[label], label)), start=1
        )
    }
    return pd.Series(
        [renumber[label] for label in labels], index=csi.index, name="module"
    )
