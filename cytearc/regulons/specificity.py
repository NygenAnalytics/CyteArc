"""Regulon specificity and co-activity modules."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..utils.arguments import integer_argument

__all__ = [
    "connection_specificity",
    "regulon_modules",
    "regulon_specificity",
]


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

    Args:
        activity: Cells-by-regulons non-negative activity, such as AUCell AUCs.
        labels: Group of each cell, aligned by position with ``activity``.

    Returns:
        A groups-by-regulons DataFrame, groups in order of first appearance.
    """
    from scipy.spatial.distance import jensenshannon

    values = np.asarray(activity, dtype=np.float64)
    groups = np.asarray(labels, dtype=object)
    if values.ndim != 2 or len(groups) != values.shape[0]:
        raise ValueError("labels must align with the activity rows")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("activity must be finite and non-negative")
    order = list(dict.fromkeys(groups.tolist()))
    totals = values.sum(axis=0)
    result = np.full((len(order), values.shape[1]), np.nan)
    for row, group in enumerate(order):
        indicator = (groups == group).astype(np.float64)
        indicator /= indicator.sum()
        for column in range(values.shape[1]):
            if totals[column] > 0:
                result[row, column] = 1.0 - jensenshannon(
                    values[:, column] / totals[column], indicator
                )
    return pd.DataFrame(
        result, index=pd.Index(order, name="group"), columns=list(activity.columns)
    )


def connection_specificity(
    activity: pd.DataFrame, *, margin: float = 0.05
) -> pd.DataFrame:
    """Connection specificity index (CSI) of every pair of regulons.

    For regulons ``a`` and ``b`` with activity correlation ``r``, the CSI is
    the fraction of all regulons whose correlation with ``a`` and with ``b``
    are both below ``r - margin`` (Fuxman Bass et al. 2013). Regulons that
    co-vary with each other but not with the rest score near 1, which SCENIC
    analyses use to group regulons into modules.

    Args:
        activity: Cells-by-regulons activity.
        margin: Correlation margin.

    Returns:
        A symmetric regulons-by-regulons DataFrame.
    """
    correlation = np.corrcoef(np.asarray(activity, dtype=np.float64).T)
    correlation = np.nan_to_num(correlation, nan=0.0)
    n = correlation.shape[0]
    result = np.zeros((n, n), dtype=np.float64)
    for first in range(n):
        thresholds = correlation[first] - margin
        below_first = correlation[first][None, :] < thresholds[:, None]
        for second in range(first, n):
            below = below_first[second] & (correlation[second] < thresholds[second])
            result[first, second] = result[second, first] = below.sum() / n
    names = list(activity.columns)
    return pd.DataFrame(result, index=names, columns=names)


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
