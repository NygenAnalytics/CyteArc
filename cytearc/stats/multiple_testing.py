"""False discovery rate control for families of p-values.

:func:`p_adjust_bh` is the step-up procedure of Benjamini and Hochberg
(1995). :func:`weighted_bh` is its weighted form (Benjamini and Hochberg
1997), which Lun, Richard and Marioni (2017) use to control the spatial FDR
of overlapping hyperspheres and Milo (Dann et al. 2022) applies to graph
neighbourhoods. Both were implemented from those papers; their agreement
with R's ``p.adjust`` and miloR's ``graphSpatialFDR`` is checked against
stored outputs of those functions.
"""

import numpy as np

__all__ = ["p_adjust_bh", "weighted_bh"]


def _step_up(p_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted step-up adjustment of complete p-values.

    The ordered p-value ``p_(i)`` is scaled by the total weight over the
    weight of the ``i`` smallest p-values, and each adjusted value is the
    minimum over itself and every larger p-value, capped at one.
    """
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    cumulative = np.cumsum(weights[order])
    scaled = ordered * (cumulative[-1] / cumulative)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def p_adjust_bh(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values.

    Missing p-values stay missing and do not count towards the number of
    tests, as in R's ``p.adjust(p, "BH")``.
    """
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(values.shape, np.nan)
    valid = ~np.isnan(values)
    if valid.any():
        result[valid] = _step_up(values[valid], np.ones(int(valid.sum())))
    return result


def weighted_bh(p_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return weighted Benjamini-Hochberg adjusted p-values.

    With equal weights this is :func:`p_adjust_bh`. Missing p-values stay
    missing and their weights are ignored.

    Raises:
        ValueError: If the shapes differ or a used weight is not positive
            and finite.
    """
    values = np.asarray(p_values, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if weight_values.shape != values.shape:
        raise ValueError("weights must have one value per p-value")
    result = np.full(values.shape, np.nan)
    valid = ~np.isnan(values)
    if not valid.any():
        return result
    used = weight_values[valid]
    if not np.all(np.isfinite(used) & (used > 0)):
        raise ValueError("weights of tested p-values must be positive and finite")
    result[valid] = _step_up(values[valid], used)
    return result
