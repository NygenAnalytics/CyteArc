"""Differences in category proportions between groups of samples (propeller).

propeller (Phipson et al. 2022) counts the cells of each category in each
sample, turns the counts into proportions, and stabilizes their variance
with a logit or arcsine square-root transform. Each category is then fitted
by a linear model on the sample groups and tested with empirical Bayes
moderated statistics (Smyth 2004; Phipson et al. 2016): a t-test of the
first group against the second when there are two groups, and an F-test of
equal means otherwise. Samples, not cells, are the replicates.

The module was implemented from those papers and speckle's documentation.
Agreement with ``speckle::propeller`` is checked by comparing outputs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..stats.limma import contrasts_fit, e_bayes, lm_fit, subset_coefficients
from ..stats.multiple_testing import p_adjust_bh

__all__ = [
    "PropellerFit",
    "PropellerTransform",
    "count_categories",
    "propeller",
    "transformed_proportions",
]

PropellerTransform = Literal["logit", "asin"]

# Pseudo-count added to every count before the logit, so that zero counts
# have finite log-odds.
_LOGIT_PSEUDO_COUNT = 0.5


@dataclass(frozen=True, slots=True, eq=False)
class PropellerFit:
    """A propeller test of category proportions.

    ``counts``, ``proportions`` and ``transformed`` are categories x samples.
    ``df_prior`` and ``s2_prior`` are the empirical Bayes prior of each
    category. ``table`` holds one row per category, in category order; see
    :func:`propeller` for its columns.
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


def _order(
    labels: np.ndarray, order: Sequence[Any] | None, name: str
) -> tuple[str, ...]:
    observed = set(labels.tolist())
    if order is None:
        return tuple(sorted(observed))
    resolved = tuple(str(value) for value in order)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not repeat a label")
    missing = sorted(observed.difference(resolved))
    if missing:
        raise ValueError(f"{name} does not cover {missing}")
    return resolved


def count_categories(
    categories: Sequence[Any] | np.ndarray,
    samples: Sequence[Any] | np.ndarray,
    *,
    category_order: Sequence[Any] | None = None,
    sample_order: Sequence[Any] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Count cells per category (rows) and sample (columns).

    Labels are compared as strings. Orders default to the sorted labels.

    Returns:
        The counts, the category order and the sample order.

    Raises:
        ValueError: If the inputs differ in length or an order misses a
            label.
    """
    category_labels = np.asarray([str(value) for value in categories], dtype=object)
    sample_labels = np.asarray([str(value) for value in samples], dtype=object)
    if category_labels.shape != sample_labels.shape:
        raise ValueError("categories and samples must have one value per cell")
    category_names = _order(category_labels, category_order, "category_order")
    sample_names = _order(sample_labels, sample_order, "sample_order")
    rows = pd.Index(category_names).get_indexer(category_labels)
    columns = pd.Index(sample_names).get_indexer(sample_labels)
    counts = np.zeros((len(category_names), len(sample_names)), dtype=np.int64)
    np.add.at(counts, (rows, columns), 1)
    return counts, category_names, sample_names


def transformed_proportions(
    counts: np.ndarray, transform: PropellerTransform
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample category proportions and their transform.

    ``"logit"`` takes the log-odds of the proportions after adding 0.5 to
    every count; ``"asin"`` takes the arcsine of their square root.

    Raises:
        ValueError: If ``transform`` is unknown or a sample has no cells.
    """
    values = np.asarray(counts, dtype=np.float64)
    totals = values.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("Every sample needs at least one cell")
    proportions = values / totals
    if transform == "logit":
        pseudo = values + _LOGIT_PSEUDO_COUNT
        shares = pseudo / pseudo.sum(axis=0)
        transformed = np.log(shares / (1 - shares))
    elif transform == "asin":
        transformed = np.arcsin(np.sqrt(proportions))
    else:
        raise ValueError("transform must be 'logit' or 'asin'")
    return proportions, transformed


def propeller(
    counts: np.ndarray,
    *,
    categories: Sequence[str],
    samples: Sequence[str],
    sample_groups: Sequence[Any],
    groups: Sequence[Any] | None = None,
    transform: PropellerTransform = "logit",
    robust: bool = True,
    trend: bool = False,
) -> PropellerFit:
    """Test whether category proportions differ between sample groups.

    Each category's transformed proportions are fitted on one indicator per
    group. With two groups the moderated t-statistic tests the first group
    minus the second; with more, the moderated F-statistic tests that all
    group means are equal. Robust moderation is turned off with two or
    fewer categories.

    The ``table`` is indexed by category (in the order of ``categories``)
    with ``baseline_prop`` (the category's share of all cells), one
    ``prop_mean_<group>`` per group (the mean proportion over the group's
    samples), then ``prop_ratio`` (first over second group mean) and
    ``t_statistic`` for two groups, or ``f_statistic`` for more, followed by
    ``p_value`` and the Benjamini-Hochberg ``fdr``.

    Args:
        counts: Cells per category (rows) and sample (columns).
        categories: Category labels of the rows.
        samples: Sample labels of the columns.
        sample_groups: The group of each sample.
        groups: Groups to keep, in order; samples of other groups are left
            out. Defaults to the sorted group labels.
        transform: ``"logit"`` or ``"asin"``.
        robust: Robust empirical Bayes moderation.
        trend: Let the prior variance trend with each category's mean
            transformed proportion.

    Raises:
        ValueError: If shapes disagree, fewer than two groups remain, or the
            design leaves no residual degrees of freedom.
    """
    values = np.asarray(counts)
    category_names = tuple(str(value) for value in categories)
    sample_names = tuple(str(value) for value in samples)
    group_of = np.asarray([str(value) for value in sample_groups], dtype=object)
    if values.ndim != 2 or values.shape != (len(category_names), len(sample_names)):
        raise ValueError("counts must be categories x samples")
    if group_of.size != len(sample_names):
        raise ValueError("sample_groups must have one group per sample")
    if groups is None:
        group_names = tuple(sorted(set(group_of.tolist())))
    else:
        requested = tuple(str(value) for value in groups)
        if len(set(requested)) != len(requested):
            raise ValueError("groups must not repeat a group")
        present = set(group_of.tolist())
        group_names = tuple(group for group in requested if group in present)
    if len(group_names) < 2:
        raise ValueError("propeller needs at least two groups with samples")
    keep = np.isin(group_of, np.asarray(group_names, dtype=object))
    values = values[:, keep]
    sample_names = tuple(np.asarray(sample_names, dtype=object)[keep].tolist())
    group_of = group_of[keep]
    if values.shape[1] - len(group_names) < 1:
        raise ValueError(
            "propeller needs more samples than groups to estimate the variance"
        )
    proportions, transformed = transformed_proportions(values, transform)
    indicators = np.column_stack(
        [(group_of == group).astype(np.float64) for group in group_names]
    )
    robust_used = bool(robust) and len(category_names) > 2
    if len(group_names) == 2:
        contrast = np.array([1.0, -1.0])
        fit = contrasts_fit(lm_fit(transformed, indicators), contrast)
    else:
        # An intercept and the indicators of every group but the first span
        # the same model; testing the indicator coefficients tests equality.
        design = indicators.copy()
        design[:, 0] = 1.0
        fit = subset_coefficients(
            lm_fit(transformed, design), list(range(1, len(group_names)))
        )
    moderated = e_bayes(fit, trend=bool(trend), robust=robust_used)

    columns: dict[str, np.ndarray] = {
        "baseline_prop": values.sum(axis=1) / values.sum()
    }
    means = {
        group: proportions[:, group_of == group].mean(axis=1) for group in group_names
    }
    columns.update({f"prop_mean_{group}": means[group] for group in group_names})
    if len(group_names) == 2:
        method: Literal["t_test", "anova"] = "t_test"
        with np.errstate(divide="ignore", invalid="ignore"):
            columns["prop_ratio"] = means[group_names[0]] / means[group_names[1]]
        columns["t_statistic"] = moderated.t[:, 0]
        p_value = moderated.p_value[:, 0]
    else:
        method = "anova"
        columns["f_statistic"] = moderated.f_statistic
        p_value = moderated.f_p_value
    columns["p_value"] = p_value
    columns["fdr"] = p_adjust_bh(p_value)
    table = pd.DataFrame(columns, index=pd.Index(category_names, name="category"))
    return PropellerFit(
        method=method,
        transform=transform,
        robust=robust_used,
        trend=bool(trend),
        categories=category_names,
        samples=sample_names,
        groups=group_names,
        sample_groups=tuple(group_of.tolist()),
        counts=values,
        proportions=proportions,
        transformed=transformed,
        df_prior=np.asarray(moderated.df_prior, dtype=np.float64),
        s2_prior=np.asarray(moderated.s2_prior, dtype=np.float64),
        table=table,
    )
