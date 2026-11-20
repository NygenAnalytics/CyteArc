from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ..utils.arguments import require_labels_distinct_as_text


def label_concordance_score(
    label_sets: Sequence[np.ndarray],
    metric: Literal["ari", "nmi"] = "ari",
) -> float:
    """Compare two label partitions using ARI or NMI.

    Args:
        label_sets: Two arrays of labels to compare.
        metric: Either ``"ari"`` or ``"nmi"``.

    Returns:
        Label agreement. ARI ranges from -1 to 1 and NMI from 0 to 1.

    Raises:
        ValueError: If ``label_sets`` does not hold two one-dimensional
            arrays of one length without missing values, an array holds equal
            labels that print differently, such as ``1``, ``1.0`` and
            ``True``, or different labels that print alike, such as ``1`` and
            ``"1"``, or ``metric`` is neither ``"ari"`` nor ``"nmi"``.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(label_sets) != 2:
        raise ValueError("Exactly two label arrays are required")

    first = np.asarray(label_sets[0])
    second = np.asarray(label_sets[1])
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Label arrays must be one-dimensional")
    if len(first) != len(second):
        raise ValueError("Label arrays must have matching lengths")
    if pd.isna(first).any() or pd.isna(second).any():
        raise ValueError("Label arrays must not contain missing values")
    for index, labels in enumerate(label_sets):
        require_labels_distinct_as_text(
            labels,
            name=f"label_sets[{index}]",
            remedy="convert the labels to text, or to one type, first",
        )

    if metric == "ari":
        return float(adjusted_rand_score(first, second))
    if metric == "nmi":
        return float(normalized_mutual_info_score(first, second))
    raise ValueError(f"Metric {metric!r} is not one of 'ari' or 'nmi'")
