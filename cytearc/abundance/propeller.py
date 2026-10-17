"""Differences in category proportions between groups of samples (propeller).

A port of ``speckle::propeller`` 1.10.0 (Phipson et al., 2022). Cells are
counted per sample and category, proportions are transformed, and limma's
moderated statistics test each category: a t-test between two groups, or an
F-test (ANOVA) across three or more. Samples are the replicates; cells are
never treated as independent observations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..stats.limma import (
    ModeratedFit,
    contrasts_fit,
    e_bayes,
    lm_fit,
    subset_coefficients,
)
from ..stats.multiple_testing import p_adjust_bh

__all__ = [
    "PropellerFit",
    "PropellerTransform",
    "count_categories",
    "propeller",
    "transformed_proportions",
]

type PropellerTransform = Literal["logit", "asin"]


@dataclass(frozen=True, slots=True, eq=False)
class PropellerFit:
    """One propeller test.

    ``table`` has one row per category with ``baseline_prop``, one
    ``prop_mean_<group>`` column per group, ``prop_ratio`` (t-test only),
    the ``t_statistic`` or ``f_statistic``, ``p_value``, and ``fdr``.
    ``counts``, ``proportions`` and ``transformed`` are categories x samples.
    """

    method: Literal["t_test", "anova"]
    transform: PropellerTransform
    robust: bool
    trend: bool
    categories: tuple[str, ...]
    samples: tuple[str, ...]
    groups: tuple[str, ...]
    sample_groups: tuple[str, ...]
    counts: np.ndarray
    proportions: np.ndarray
    transformed: np.ndarray
    df_prior: np.ndarray
    s2_prior: np.ndarray
    table: pd.DataFrame


def count_categories(
    categories: np.ndarray,
    samples: np.ndarray,
    *,
    category_order: Sequence[str] | None = None,
    sample_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Count cells per category and sample, as ``table(sample, clusters)``.

    Returns:
        The categories x samples counts, the category order, and the sample
        order. Orders default to sorted labels, as R's factor levels.
    """
    category_labels = np.asarray([str(value) for value in categories], dtype=object)
    sample_labels = np.asarray([str(value) for value in samples], dtype=object)
    if category_labels.shape != sample_labels.shape:
        raise ValueError("categories and samples must have one value per cell")
    category_names = (
        tuple(sorted(set(category_labels.tolist())))
        if category_order is None
        else tuple(str(value) for value in category_order)
    )
    sample_names = (
        tuple(sorted(set(sample_labels.tolist())))
        if sample_order is None
        else tuple(str(value) for value in sample_order)
    )
    category_index = {name: idx for idx, name in enumerate(category_names)}
    sample_index = {name: idx for idx, name in enumerate(sample_names)}
    try:
        category_codes = np.fromiter(
            (category_index[value] for value in category_labels),
            dtype=np.int64,
            count=category_labels.size,
        )
        sample_codes = np.fromiter(
            (sample_index[value] for value in sample_labels),
            dtype=np.int64,
            count=sample_labels.size,
        )
    except KeyError as error:
        raise ValueError(f"Label {error.args[0]!r} is not in the given order") from None
    counts = np.zeros((len(category_names), len(sample_names)), dtype=np.int64)
    np.add.at(counts, (category_codes, sample_codes), 1)
    return counts, category_names, sample_names


def transformed_proportions(
    counts: np.ndarray, transform: PropellerTransform
) -> tuple[np.ndarray, np.ndarray]:
    """Return proportions and their transform, as ``getTransformedProps``.

    Args:
        counts: Categories x samples cell counts.
        transform: ``"logit"`` adds 0.5 to each count before the logit;
            ``"asin"`` takes the arcsine square root.
    """
    table = np.asarray(counts, dtype=np.float64)
    totals = table.sum(axis=0)
    if np.any(totals == 0):
        raise ValueError("Every sample must contain at least one cell")
    proportions = table / totals
    if transform == "asin":
        transformed = np.arcsin(np.sqrt(proportions))
    elif transform == "logit":
        pseudo = (table + 0.5) / (table + 0.5).sum(axis=0)
        transformed = np.log(pseudo / (1 - pseudo))
    else:
        raise ValueError("transform must be 'logit' or 'asin'")
    return proportions, transformed


def propeller(
    counts: np.ndarray,
    *,
    categories: Sequence[str],
    samples: Sequence[str],
    sample_groups: Sequence[Any],
    groups: Sequence[str] | None = None,
    transform: PropellerTransform = "logit",
    robust: bool = True,
    trend: bool = False,
) -> PropellerFit:
    """Test each category's proportion across groups of samples.

    Args:
        counts: Categories x samples cell counts.
        categories: Category label of each row.
        samples: Sample label of each column.
        sample_groups: Group of each sample.
        groups: Group order. With two groups the contrast is the first group
            minus the second. Defaults to sorted group labels.
        transform: Proportion transform, ``"logit"`` or ``"asin"``.
        robust: Robust empirical Bayes. It is turned off with two or fewer
            categories, as propeller does.
        trend: Let the prior variance depend on the mean transformed value.

    Raises:
        ValueError: If fewer than two groups remain, or the design leaves no
            residual degrees of freedom.
    """
    table = np.asarray(counts)
    category_names = tuple(str(value) for value in categories)
    sample_names = tuple(str(value) for value in samples)
    group_of_sample = tuple(str(value) for value in sample_groups)
    if table.shape != (len(category_names), len(sample_names)):
        raise ValueError("counts must be categories x samples")
    if len(group_of_sample) != len(sample_names):
        raise ValueError("sample_groups must have one group per sample")
    observed = sorted(set(group_of_sample))
    group_order = (
        tuple(observed) if groups is None else tuple(str(value) for value in groups)
    )
    missing = sorted(set(observed).difference(group_order))
    if missing:
        raise ValueError(f"groups do not include the sample groups {missing}")
    group_order = tuple(group for group in group_order if group in set(observed))
    if len(group_order) < 2:
        raise ValueError("propeller needs samples from at least two groups")
    design = np.array(
        [[float(group == level) for level in group_order] for group in group_of_sample]
    )
    if len(sample_names) <= len(group_order):
        raise ValueError(
            "propeller needs more samples than groups to estimate the variance"
        )
    proportions, transformed = transformed_proportions(table, transform)
    totals = table.sum(axis=1).astype(np.float64)
    baseline = totals / totals.sum()
    robust_used = robust and len(category_names) > 2
    mean_fit = lm_fit(proportions, design)
    prop_means = {
        f"prop_mean_{group}": mean_fit.coefficients[:, idx]
        for idx, group in enumerate(group_order)
    }
    if len(group_order) == 2:
        method: Literal["t_test", "anova"] = "t_test"
        contrast = np.array([1.0, -1.0])
        fit = contrasts_fit(lm_fit(transformed, design), contrast)
        moderated: ModeratedFit = e_bayes(fit, robust=robust_used, trend=trend)
        statistic = {"t_statistic": moderated.t[:, 0]}
        p_value = moderated.p_value[:, 0]
        ratio = {
            "prop_ratio": mean_fit.coefficients[:, 0] / mean_fit.coefficients[:, 1]
        }
    else:
        method = "anova"
        intercept_design = design.copy()
        intercept_design[:, 0] = 1.0
        fit = lm_fit(transformed, intercept_design)
        fit = subset_coefficients(fit, list(range(1, len(group_order))))
        moderated = e_bayes(fit, robust=robust_used, trend=trend)
        statistic = {"f_statistic": moderated.f_statistic}
        p_value = moderated.f_p_value
        ratio = {}
    frame = pd.DataFrame(
        {
            "baseline_prop": baseline,
            **prop_means,
            **ratio,
            **statistic,
            "p_value": p_value,
            "fdr": p_adjust_bh(p_value),
        },
        index=pd.Index(category_names, name="category"),
    )
    return PropellerFit(
        method=method,
        transform=transform,
        robust=robust_used,
        trend=trend,
        categories=category_names,
        samples=sample_names,
        groups=group_order,
        sample_groups=group_of_sample,
        counts=table.astype(np.int64),
        proportions=proportions,
        transformed=transformed,
        df_prior=moderated.df_prior,
        s2_prior=moderated.s2_prior,
        table=frame,
    )
