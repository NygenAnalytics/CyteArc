"""Regulon specificity scores and connection specificity indices computed from their definitions, one group or pair of regulons at a time."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def rss_by_definition(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Score each group and regulon with SciPy's Jensen-Shannon distance, as pySCENIC does.

    ``1 - jensenshannon(p, q)``, with ``p`` a regulon's activity over its
    total and ``q`` the group's indicator over its cells, on whole columns.
    Groups follow their first appearance in ``labels``, and a regulon
    without activity scores NaN.
    """
    order = list(dict.fromkeys(labels.tolist()))
    result = np.full((len(order), values.shape[1]), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for row, group in enumerate(order):
            indicator = (labels == group).astype(np.float64)
            for column in range(values.shape[1]):
                activity = values[:, column].astype(np.float64)
                if activity.sum() > 0:
                    result[row, column] = 1.0 - jensenshannon(
                        activity / activity.sum(), indicator / indicator.sum()
                    )
    return pd.DataFrame(result, index=order)


def csi_by_definition(values: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Count, for each pair of regulons, the share of regulons that correlate below the pair's correlation less ``margin`` with both.

    The correlations are ``numpy.corrcoef``'s, 0 for a regulon of constant
    activity, as Fuxman Bass et al. (2013) define the index.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(np.corrcoef(values.astype(np.float64).T), nan=0.0)
    n = len(correlation)
    result = np.empty((n, n))
    for first in range(n):
        for second in range(n):
            threshold = correlation[first, second] - margin
            below = (correlation[first] < threshold) & (correlation[second] < threshold)
            result[first, second] = below.sum() / n
    return result
