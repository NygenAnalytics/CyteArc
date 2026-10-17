"""Multiple-testing corrections that match R's ``p.adjust``."""

import numpy as np

__all__ = ["p_adjust_bh", "weighted_bh"]


def p_adjust_bh(p_values: np.ndarray) -> np.ndarray:
    """Return ``p.adjust(p, method="BH")``; missing values stay missing."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite = ~np.isnan(values)
    p = values[finite]
    n = p.size
    if n == 0:
        return adjusted
    # R: i <- lp:1; o <- order(p, decreasing=TRUE); pmin(1, cummin(n/i * p[o]))[order(o)]
    order = np.argsort(-p, kind="stable")
    ranks = np.arange(n, 0, -1, dtype=np.float64)
    scaled = np.minimum.accumulate(n / ranks * p[order])
    result = np.empty(n, dtype=np.float64)
    result[order] = np.minimum(1.0, scaled)
    adjusted[finite] = result
    return adjusted


def weighted_bh(p_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return cydar's weighted Benjamini-Hochberg adjustment, as miloR applies it.

    Missing p-values stay missing and their weights are ignored.
    """
    values = np.asarray(p_values, dtype=np.float64)
    w_all = np.asarray(weights, dtype=np.float64)
    if values.shape != w_all.shape:
        raise ValueError("p-values and weights must have the same shape")
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    has_p = ~np.isnan(values)
    p = values[has_p]
    w = w_all[has_p]
    if p.size == 0:
        return adjusted
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    w_sorted = w[order]
    raw = np.sum(w_sorted) * p_sorted / np.cumsum(w_sorted)
    sorted_adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = sorted_adjusted
    adjusted[has_p] = np.minimum(result, 1.0)
    return adjusted
