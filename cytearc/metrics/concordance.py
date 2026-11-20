from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ..utils.arguments import (
    categorical_label_codes,
    label_array,
    require_labels_match_their_text,
)


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
            arrays of one length without missing values, an array holds values
            that would form one label but print differently, such as ``1``,
            ``1.0`` and ``True``, or values that print alike but would form
            separate labels, such as ``1`` and ``"1"``, or ``metric`` is
            neither ``"ari"`` nor ``"nmi"``.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(label_sets) != 2:
        raise ValueError("Exactly two label arrays are required")

    first = label_array(label_sets[0])
    second = label_array(label_sets[1])
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Label arrays must be one-dimensional")
    if len(first) != len(second):
        raise ValueError("Label arrays must have matching lengths")
    if pd.isna(first).any() or pd.isna(second).any():
        raise ValueError("Label arrays must not contain missing values")
    require_labels_match_their_text(first, name="label_sets[0]")
    require_labels_match_their_text(second, name="label_sets[1]")
    # scikit-learn sorts the labels it is given, which fails for text beside
    # numbers. Codes in the labels' sorted order, where they sort, give the
    # scores that the labels themselves give.
    first_codes = categorical_label_codes(first)[0]
    second_codes = categorical_label_codes(second)[0]

    if metric == "ari":
        return float(adjusted_rand_score(first_codes, second_codes))
    if metric == "nmi":
        return float(normalized_mutual_info_score(first_codes, second_codes))
    raise ValueError(f"Metric {metric!r} is not one of 'ari' or 'nmi'")
