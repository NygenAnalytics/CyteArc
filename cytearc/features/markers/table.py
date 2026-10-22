"""Rank marker results and the readers of their persisted statistics."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy.special import log_ndtr, ndtr

from ..statistical import (
    MANN_WHITNEY_P_VALUE_POLICY,
    _mann_whitney_p_value_method,
    adjust_pvalues,
)
from .rank import MarkerSortKey, sort_marker_results
from ...utils.arrays import has_duplicates

MARKER_STAT_COLUMNS = (
    "score",
    "mean",
    "mean_rest",
    "frac_exp",
    "frac_exp_rest",
    "fold_change",
    "p_value",
    "auc",
    "p_value_adjusted",
    "neg_log10_p_value",
)
MARKER_METHOD = "mannwhitneyu"
MARKER_ALTERNATIVE = "two-sided"
MARKER_TIE_CORRECTION = True
MARKER_CONTINUITY_CORRECTION = True
MARKER_ADJUSTMENT_METHOD = "fdr_bh"
MARKER_ADJUSTMENT_SCOPE = "within_group_all_tested_features"
# ``fold_change`` is the group mean over the rest mean when both are
# non-negative and the rest mean is positive, +inf when only the group mean is
# positive, and NaN when both means are 0 or either is negative. Marker slots
# record this policy as metadata, which the loader and the reuse validator
# require before a table can be read or reused.
MARKER_FOLD_CHANGE_POLICY = "ratio_of_means_inf_when_rest_zero_nan_when_undefined"
# A group's p-values are exact when it and the rest of the cells can be split
# in at most MANN_WHITNEY_EXACT_MAX_SPLITS ways, as for ``compare.test``, and
# otherwise come from the tie- and continuity-corrected normal approximation.
MARKER_P_VALUE_POLICY = MANN_WHITNEY_P_VALUE_POLICY
# Revision 1 of run_marker_search, before release 1.3.0, stored these columns,
# without neg_log10_p_value, rounded every statistic except the two p-value
# columns to five decimals, took every p-value from the tie- and
# continuity-corrected normal approximation, and recorded neither
# p_value_policy nor skipped_groups. The readers still load such a table: they
# report this p-value policy for it and derive its neg_log10_p_value from its
# p_value. Marker search never reuses it.
MARKER_REVISION_1_STAT_COLUMNS = tuple(
    name for name in MARKER_STAT_COLUMNS if name != "neg_log10_p_value"
)
MARKER_REVISION_1_P_VALUE_POLICY = "asymptotic_tie_and_continuity_corrected"
_MARKER_METADATA = {
    "method": MARKER_METHOD,
    "alternative": MARKER_ALTERNATIVE,
    "tie_correction": MARKER_TIE_CORRECTION,
    "continuity_correction": MARKER_CONTINUITY_CORRECTION,
    "adjustment_method": MARKER_ADJUSTMENT_METHOD,
    "adjustment_scope": MARKER_ADJUSTMENT_SCOPE,
    "fold_change_policy": MARKER_FOLD_CHANGE_POLICY,
    "p_value_policy": MARKER_P_VALUE_POLICY,
}

__all__ = [
    "MARKER_REVISION_1_P_VALUE_POLICY",
    "MARKER_REVISION_1_STAT_COLUMNS",
    "MARKER_STAT_COLUMNS",
    "MarkerFilter",
    "MarkerSlot",
    "RankMarkerResult",
    "exact_marker_groups",
    "load_marker_table",
    "rank_marker_tested_groups",
]

# The rank kernels compute the statistics before ``p_value_adjusted``, with
# the Mann-Whitney z statistic in place of ``p_value``.
_RANK_COLUMNS = MARKER_STAT_COLUMNS.index("p_value_adjusted")
_P_VALUE = MARKER_STAT_COLUMNS.index("p_value")
_ADJUSTED = MARKER_STAT_COLUMNS.index("p_value_adjusted")
_NEG_LOG10_P_VALUE = MARKER_STAT_COLUMNS.index("neg_log10_p_value")


def _statistics_violation(stats: np.ndarray, columns: Sequence[str]) -> str | None:
    """Return how one group's stored marker statistics break their contract.

    Every statistic but ``fold_change`` is finite, and ``fold_change``
    follows ``MARKER_FOLD_CHANGE_POLICY``: it is NaN or lies in [0, +inf], it
    is +inf only where ``mean_rest`` is 0, and it is NaN only where both
    means are 0 or either is negative.

    Returns:
        A description of the first broken rule, or None.
    """
    # One column at a time, so the checks hold no copy of the table.
    for position, name in enumerate(columns):
        if name != "fold_change" and not np.isfinite(stats[:, position]).all():
            return "statistics other than fold_change must all be finite"
    ratios = stats[:, columns.index("fold_change")]
    means = stats[:, columns.index("mean")]
    rest_means = stats[:, columns.index("mean_rest")]
    if np.any(ratios < 0.0):
        return "fold_change holds a negative value"
    if np.any(np.isposinf(ratios) & (rest_means != 0.0)):
        return "fold_change is +inf where mean_rest is not 0"
    # The sign bit marks a negative mean and a mean of -0.0.
    undefined = (
        ((means == 0.0) & (rest_means == 0.0))
        | np.signbit(means)
        | np.signbit(rest_means)
    )
    if np.any(np.isnan(ratios) & ~undefined):
        return (
            "fold_change is NaN where both means are non-negative and one is positive"
        )
    return None


def stored_table_bytes(n_features: int) -> int:
    """Return a bound on what finishing and writing one group's table holds.

    While a table is finished, the intermediates of its p-values and their
    adjustment take at most one more table's size; while it is written, the
    bytes Zarr encodes and compresses from it take at most two.
    """
    table = max(0, int(n_features)) * len(MARKER_STAT_COLUMNS)
    return 3 * table * np.dtype(np.float64).itemsize


def rank_marker_tested_groups(group_sizes: np.ndarray) -> np.ndarray:
    """Return which groups a marker table compares with the rest of the cells.

    A group needs at least two cells, and so does the rest of the cells.
    Marker search skips the other groups, whose cells stay in the rest of
    every tested group.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    return (sizes >= 2) & (int(sizes.sum()) - sizes >= 2)


def _describe_groups(group_ids: np.ndarray, group_sizes: np.ndarray) -> str:
    """Name up to ten groups with their cell counts."""
    described = [
        f"{str(group_id)!r} ({int(size)} {'cell' if size == 1 else 'cells'})"
        for group_id, size in zip(group_ids[:10], group_sizes[:10], strict=True)
    ]
    hidden = len(group_ids) - len(described)
    return ", ".join(described) + (f", and {hidden} more" if hidden else "")


def _validate_rank_marker_groups(
    group_ids: np.ndarray, group_sizes: np.ndarray
) -> np.ndarray:
    """Refuse groups of which no marker table can be stored.

    Returns:
        Which groups are tested, by :func:`rank_marker_tested_groups`.

    Raises:
        ValueError: If there are fewer than two groups, a group has no
            cells, or no group is tested.
    """
    if group_sizes.size < 2:
        raise ValueError("Rank markers require at least two populated groups")
    if np.any(group_sizes < 1):
        raise ValueError("Rank marker groups must hold at least one cell each")
    tested = rank_marker_tested_groups(group_sizes)
    if not tested.any():
        raise ValueError(
            "Rank markers require a group of at least two cells and at least two "
            "other cells; the groups are "
            + _describe_groups(group_ids, group_sizes)
            + ". Merge small groups in a cell metadata column, or select more "
            "cells, then freeze the labels with snapshot_cluster_labels."
        )
    return tested


def exact_marker_groups(group_sizes: np.ndarray) -> np.ndarray:
    """Return the tested groups whose marker p-values are exact.

    Their p-value method, under ``MARKER_P_VALUE_POLICY``, is that of
    ``compare.test`` for the group and the rest of the cells.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    n_cells = int(sizes.sum())
    tested = rank_marker_tested_groups(sizes)
    return np.asarray(
        [
            bool(tested[position])
            and _mann_whitney_p_value_method(int(size), n_cells - int(size)) == "exact"
            for position, size in enumerate(sizes)
        ],
        dtype=bool,
    )


def _ranked_table(
    stats: np.ndarray,
    columns: Sequence[str],
    feature_index: np.ndarray,
    feature_names: np.ndarray,
    group_id: Any,
    sort_by: MarkerSortKey = "score",
) -> pd.DataFrame:
    """Name and rank one group's stored statistics for display."""
    names = np.asarray(feature_names, dtype=object)
    if names.ndim != 1:
        raise ValueError("Feature names must be one-dimensional")
    frame = pd.DataFrame(stats, columns=list(columns))
    frame["group_id"] = group_id
    frame["feature_name"] = names[feature_index]
    frame["feature_index"] = feature_index
    return sort_marker_results(
        frame[["group_id", "feature_name", "feature_index", *MARKER_STAT_COLUMNS]],
        sort_by,
    )


@dataclass(frozen=True, slots=True, eq=False)
class RankMarkerResult:
    """Rank marker statistics of every group, as the rank kernels compute them.

    ``statistics`` holds, for each tested feature and group, the
    ``MARKER_STAT_COLUMNS`` before ``p_value_adjusted``, with the
    continuity- and tie-corrected Mann-Whitney z statistic in place of
    ``p_value``. A group is tested when it and the rest of the cells each
    hold at least two cells (:func:`rank_marker_tested_groups`); the other
    groups are skipped, and their cells stay in the rest of every tested
    group. ``stored_statistics`` finishes one tested group into the table
    that a marker artifact stores, and ``table`` into the ranked table that
    such an artifact reads back.

    Attributes:
        group_ids: Sorted unique group labels, at least two.
        group_sizes: Number of cells in each group, at least one, with at
            least one tested group.
        feature_index: Ascending assay indices of the tested features, at
            least one.
        statistics: Features-by-groups-by-8 rank statistics.
        exact_p_values: Features-by-groups exact two-sided p-values, in
            (0, 1], of the tested groups whose p-values are exact, and NaN in
            every other column; or None when no group's p-values are exact.
    """

    group_ids: np.ndarray = field(repr=False)
    group_sizes: np.ndarray = field(repr=False)
    feature_index: np.ndarray = field(repr=False)
    statistics: np.ndarray = field(repr=False)
    exact_p_values: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        group_ids = np.asarray(self.group_ids)
        group_sizes = np.asarray(self.group_sizes)
        feature_index = np.asarray(self.feature_index)
        statistics = np.asarray(self.statistics)
        if (
            group_ids.ndim != 1
            or has_duplicates(group_ids)
            or group_sizes.shape != group_ids.shape
            or group_sizes.dtype.kind not in {"i", "u"}
        ):
            raise ValueError(
                "Rank marker groups must be unique labels aligned with integer sizes"
            )
        tested = _validate_rank_marker_groups(group_ids, group_sizes)
        # Neighbor comparisons, unlike differences, do not wrap for unsigned
        # indices.
        if (
            feature_index.ndim != 1
            or feature_index.size == 0
            or feature_index.dtype.kind not in {"i", "u"}
            or np.any(feature_index[1:] <= feature_index[:-1])
            or feature_index[0] < 0
        ):
            raise ValueError(
                "Rank marker feature_index must hold one or more ascending "
                "unique non-negative indices"
            )
        if statistics.dtype != np.float64 or statistics.shape != (
            feature_index.size,
            group_ids.size,
            _RANK_COLUMNS,
        ):
            raise ValueError(
                "Rank marker statistics must be float64 with one row per feature "
                f"and {_RANK_COLUMNS} statistics per group"
            )
        arrays = [
            ("group_ids", group_ids),
            ("group_sizes", group_sizes),
            ("feature_index", feature_index),
            ("statistics", statistics),
        ]
        if self.exact_p_values is not None:
            exact = np.asarray(self.exact_p_values)
            if exact.dtype != np.float64 or exact.shape != statistics.shape[:2]:
                raise ValueError(
                    "Rank marker exact p-values must be float64 with one row per "
                    "feature and one column per group"
                )
            given = ~np.isnan(exact)
            exact_columns = given.all(axis=0)
            if (
                np.any(given.any(axis=0) & ~exact_columns)
                or np.any(exact_columns & ~tested)
                or not np.all((exact[given] > 0.0) & (exact[given] <= 1.0))
            ):
                raise ValueError(
                    "Rank marker exact p-values must fill whole columns of tested "
                    "groups with values in (0, 1]"
                )
            arrays.append(("exact_p_values", exact))
        for name, value in arrays:
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def tested_group_ids(self) -> np.ndarray:
        """Labels of the groups that a marker artifact stores, in order."""
        return np.asarray(self.group_ids[rank_marker_tested_groups(self.group_sizes)])

    @property
    def skipped_group_ids(self) -> np.ndarray:
        """Labels of the groups too small to compare with the rest, in order."""
        return np.asarray(self.group_ids[~rank_marker_tested_groups(self.group_sizes)])

    def _position(self, group_id: Any) -> int:
        matches = np.flatnonzero(self.group_ids == group_id)
        if matches.size != 1:
            raise ValueError(f"Rank marker result has no group {group_id!r}")
        return int(matches[0])

    def stored_statistics(self, group_id: Any) -> np.ndarray:
        """Return one group's marker statistics as a marker artifact stores them.

        Rows follow ``feature_index`` and columns ``MARKER_STAT_COLUMNS``. The
        rank statistics are stored at full float64 precision. ``p_value`` is
        the exact p-value of a group in ``exact_p_values``, and otherwise the
        two-sided p-value of the z statistic. ``neg_log10_p_value`` is
        ``-log10(p_value)``, taken in log space for z statistics, so it stays
        finite and keeps its order where ``p_value`` underflows to 0.
        ``p_value_adjusted`` holds the Benjamini-Hochberg values of the
        p-values over every tested feature.

        Args:
            group_id: One of ``tested_group_ids``.

        Raises:
            ValueError: If ``group_id`` is not a tested group of the result,
                a statistic other than ``fold_change`` is not finite, or
                ``fold_change`` breaks ``MARKER_FOLD_CHANGE_POLICY``.
        """
        position = self._position(group_id)
        n_group = int(self.group_sizes[position])
        n_rest = int(self.group_sizes.sum()) - n_group
        if not rank_marker_tested_groups(self.group_sizes)[position]:
            raise ValueError(
                f"Rank marker group {group_id!r} is skipped: it has {n_group} of "
                f"{n_group + n_rest} cells, and a stored marker table needs at "
                "least two cells in the group and two other cells"
            )
        rank = self.statistics[:, position]
        stored = np.empty((rank.shape[0], len(MARKER_STAT_COLUMNS)), dtype=np.float64)
        stored[:, :_RANK_COLUMNS] = rank
        exact = (
            None if self.exact_p_values is None else self.exact_p_values[:, position]
        )
        if exact is not None and not np.isnan(exact).any():
            p_values = exact
            neg_log10 = -np.log10(exact)
        else:
            z = np.abs(rank[:, _P_VALUE])
            p_values = 2.0 * ndtr(-z)
            neg_log10 = -(np.log(2.0) + log_ndtr(-z)) / np.log(10.0)
        stored[:, _P_VALUE] = p_values
        stored[:, _ADJUSTED] = adjust_pvalues(p_values, "fdr_bh")
        # A p-value of 1 gives -0.0, and its log-space form a rounding error.
        stored[:, _NEG_LOG10_P_VALUE] = np.where(neg_log10 > 0.0, neg_log10, 0.0)
        violation = _statistics_violation(stored, MARKER_STAT_COLUMNS)
        if violation is not None:
            raise ValueError(f"Marker {violation}")
        return stored

    def table(
        self,
        group_id: Any,
        feature_names: np.ndarray,
        *,
        sort_by: MarkerSortKey = "score",
    ) -> pd.DataFrame:
        """Return one tested group's ranked marker table.

        The table equals the one read back from a marker artifact written from
        this result, before ``markers.load`` filters it: ``group_id``,
        ``feature_name``, and ``feature_index`` columns followed by
        ``MARKER_STAT_COLUMNS``, ordered by ``sort_by`` as
        :func:`~cytearc.features.markers.sort_marker_results` orders them.

        Args:
            group_id: One of ``tested_group_ids``; the table's ``group_id``
                column holds the string form of that label, which names the
                group in a marker artifact.
            feature_names: Names of every feature of the assay, indexed by
                ``feature_index``.
            sort_by: The statistic that orders the rows.
        """
        return _ranked_table(
            self.stored_statistics(group_id),
            MARKER_STAT_COLUMNS,
            self.feature_index.astype(np.int64, copy=False),
            feature_names,
            str(self.group_ids[self._position(group_id)]),
            sort_by,
        )


@dataclass(frozen=True, slots=True)
class MarkerFilter:
    """Thresholds that the rows of a marker table must meet.

    A row passes when its ``score``, ``frac_exp``, and ``auc`` are at least
    ``min_score``, ``min_frac_exp``, and ``min_auc``, and its
    ``p_value_adjusted`` is at most ``max_p_adj``. ``None`` leaves a
    statistic unfiltered.

    Raises:
        TypeError: If a threshold is neither None nor a real number, or is
            a boolean.
        ValueError: If a threshold is NaN.
    """

    min_score: float | None = None
    min_frac_exp: float | None = None
    min_auc: float | None = None
    max_p_adj: float | None = None

    def __post_init__(self) -> None:
        for name in ("min_score", "min_frac_exp", "min_auc", "max_p_adj"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number or None")
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
            object.__setattr__(self, name, float(value))

    def keep(self, table: pd.DataFrame) -> np.ndarray:
        """Return which rows of a marker table pass every threshold."""
        keep = np.ones(len(table), dtype=bool)
        for column, threshold in (
            ("score", self.min_score),
            ("frac_exp", self.min_frac_exp),
            ("auc", self.min_auc),
        ):
            if threshold is not None:
                keep &= table[column].to_numpy(dtype=np.float64) >= threshold
        if self.max_p_adj is not None:
            adjusted = table["p_value_adjusted"].to_numpy(dtype=np.float64)
            keep &= adjusted <= self.max_p_adj
        return keep


def _array_values(group: zarr.Group, name: str) -> np.ndarray:
    value = group[name]
    if not isinstance(value, zarr.Array):
        raise TypeError(f"Marker field {name!r} must be an array")
    return np.asarray(value[:])


def _marker_group(slot_group: zarr.Group, group_name: str) -> zarr.Group:
    cluster_group = slot_group[group_name]
    if not isinstance(cluster_group, zarr.Group):
        raise TypeError(f"Marker group {group_name!r} must be a group")
    return cluster_group


def _stored_stat_columns(slot_group: zarr.Group) -> list[str]:
    stored = slot_group.attrs.get("stat_columns")
    if stored is None:
        raise ValueError("Canonical marker tables require stat_columns metadata")
    if isinstance(stored, str) or not isinstance(stored, Sequence):
        raise ValueError("Marker stat_columns metadata must be a sequence of names")
    columns: list[str] = []
    for name in stored:
        if not isinstance(name, str):
            raise ValueError("Marker stat_columns metadata must contain only strings")
        columns.append(name)
    if len(columns) != len(set(columns)):
        raise ValueError("Marker stat_columns metadata contains duplicate names")
    unknown = set(columns).difference(MARKER_STAT_COLUMNS)
    if unknown:
        raise ValueError(
            "Marker stat_columns metadata contains unknown columns: "
            + ", ".join(sorted(unknown))
        )
    return columns


def _validated_feature_indices(values: np.ndarray, *, n_features: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Marker feature_index must be one-dimensional")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("Canonical marker feature_index must use an integer dtype")
    indices = raw.astype(np.int64)
    invalid = (indices < 0) | (indices >= n_features)
    if invalid.any():
        unresolved = ", ".join(str(value) for value in indices[invalid][:5])
        raise ValueError(
            "Marker feature_index contains unresolved or out-of-range values: "
            + unresolved
        )
    if has_duplicates(indices):
        raise ValueError("Canonical marker feature_index must contain unique values")
    return indices


def _skipped_groups(slot_group: zarr.Group) -> dict[str, int]:
    """Return the cell count of each group that a slot records as skipped."""
    skipped = slot_group.attrs.get("skipped_groups")
    if not isinstance(skipped, dict) or not all(
        isinstance(name, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 1
        for name, count in skipped.items()
    ):
        raise ValueError(
            "Canonical marker tables require skipped_groups metadata that maps "
            "group labels to cell counts"
        )
    return dict(skipped)


def _canonical_slot(
    slot_group: zarr.Group, n_features: int
) -> tuple[list[str], np.ndarray, dict[str, int], str]:
    """Validate slot metadata.

    A slot without ``p_value_policy`` metadata is a table of revision 1: it
    must store exactly ``MARKER_REVISION_1_STAT_COLUMNS`` and record no
    ``skipped_groups``, and it reads under
    ``MARKER_REVISION_1_P_VALUE_POLICY``.

    Returns:
        The stat columns, the feature indices, the cell count of each skipped
        group, and the table's p-value policy.
    """
    if "feature_index" not in slot_group:
        raise ValueError("Canonical marker tables require feature_index and stats")
    if "fold_change_policy" not in slot_group.attrs:
        raise ValueError(
            "Marker table records no fold_change_policy metadata. Recompute it "
            "from the same clustering and feature selection with "
            "ds.markers.search; for a pipeline run, "
            'ds.markers.search(run["clusters"], '
            'features=run["feature_universe"]). A saved agent result keeps '
            "the table that it read, so run that analysis again"
        )
    revision_1 = "p_value_policy" not in slot_group.attrs
    for name, expected in _MARKER_METADATA.items():
        if revision_1 and name == "p_value_policy":
            continue
        if name not in slot_group.attrs:
            raise ValueError(f"Canonical marker tables require {name} metadata")
        value = slot_group.attrs[name]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Canonical marker metadata {name!r} must be {expected!r}")
    columns = _stored_stat_columns(slot_group)
    if revision_1:
        if set(columns) != set(MARKER_REVISION_1_STAT_COLUMNS):
            raise ValueError(
                "Marker tables without p_value_policy metadata must store the "
                "revision 1 stat_columns, which have no neg_log10_p_value"
            )
        if "skipped_groups" in slot_group.attrs:
            raise ValueError(
                "Marker tables without p_value_policy metadata must not record "
                "skipped_groups"
            )
        skipped: dict[str, int] = {}
        policy = MARKER_REVISION_1_P_VALUE_POLICY
    else:
        if set(columns) != set(MARKER_STAT_COLUMNS):
            raise ValueError(
                "Canonical marker tables must store the complete named stat_columns"
            )
        skipped = _skipped_groups(slot_group)
        policy = MARKER_P_VALUE_POLICY
    feature_index = _validated_feature_indices(
        _array_values(slot_group, "feature_index"),
        n_features=n_features,
    )
    if feature_index.shape[0] == 0:
        raise ValueError("Canonical marker groups must contain marker rows")
    return columns, feature_index, skipped, policy


def _canonical_cluster_stats(
    cluster_group: zarr.Group, columns: list[str], n_rows: int
) -> np.ndarray:
    """Validate one canonical marker group and return its stats."""
    for name in ("n_group", "n_reference"):
        value = cluster_group.attrs.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | np.integer)
            or int(value) < 2
        ):
            raise ValueError(
                f"Canonical marker groups require integer {name} metadata >= 2"
            )
    if "stats" not in cluster_group:
        raise ValueError("Canonical marker tables require feature_index and stats")
    stats = _array_values(cluster_group, "stats")
    if stats.ndim != 2 or stats.shape[1] != len(columns):
        raise ValueError("Canonical marker stats do not match stat_columns")
    if stats.shape[0] != n_rows:
        raise ValueError("Canonical marker stats do not align with feature_index")
    if stats.dtype.kind != "f":
        raise ValueError("Canonical marker stats must use a floating dtype")
    if not np.isfinite(stats[:, columns.index("p_value_adjusted")]).all():
        raise ValueError("Canonical marker p_value_adjusted values must all be finite")
    violation = _statistics_violation(stats, columns)
    if violation is not None:
        raise ValueError(f"Canonical marker {violation}")
    return stats


def _validate_marker_slot(
    slot_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    expected_group_cell_counts: dict[str, tuple[int, int]] | None = None,
    expected_skipped_groups: dict[str, int] | None = None,
    workers: int = 1,
) -> None:
    """Check every marker group without building display tables.

    ``expected_group_cell_counts`` maps each stored group to its group and
    reference cell counts, and ``expected_skipped_groups`` each skipped group
    to its cell count. ``workers`` reads groups concurrently, which hides
    object-store latency. A table of revision 1 fails the check, so marker
    search never reuses one, even under current provenance.
    """
    group_names = sorted(slot_group.group_keys())
    if not group_names:
        raise ValueError("Canonical marker tables must contain populated groups")
    if expected_group_cell_counts is not None and set(group_names) != set(
        expected_group_cell_counts
    ):
        raise ValueError("Canonical marker groups do not match the requested groups")
    columns, feature_index, skipped, policy = _canonical_slot(
        slot_group, len(feature_names)
    )
    if policy != MARKER_P_VALUE_POLICY:
        raise ValueError(
            "Canonical marker table has the revision 1 layout, which is read but "
            "never reused"
        )
    if expected_skipped_groups is not None and skipped != expected_skipped_groups:
        raise ValueError("Canonical marker skipped groups do not match the request")

    def check(group_name: str) -> None:
        cluster_group = _marker_group(slot_group, group_name)
        _canonical_cluster_stats(cluster_group, columns, feature_index.shape[0])
        if expected_group_cell_counts is None:
            return
        expected_group, expected_reference = expected_group_cell_counts[group_name]
        if (
            cluster_group.attrs.get("n_group") != expected_group
            or cluster_group.attrs.get("n_reference") != expected_reference
        ):
            raise ValueError(
                f"Canonical marker group {group_name!r} has stale cell counts"
            )

    if workers <= 1:
        for group_name in group_names:
            check(group_name)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(group_names))) as pool:
        list(pool.map(check, group_names))


def _with_neg_log10_p_values(
    stats: np.ndarray, columns: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Append ``neg_log10_p_value`` to statistics of a table of revision 1.

    It is ``-log10(p_value)`` of the stored p-values, which is +inf where a
    p-value underflowed to 0.
    """
    with np.errstate(divide="ignore"):
        neg_log10 = -np.log10(stats[:, columns.index("p_value")])
    # A p-value of 1 gives -0.0.
    neg_log10 = np.where(neg_log10 > 0.0, neg_log10, 0.0)
    return np.column_stack([stats, neg_log10]), [*columns, "neg_log10_p_value"]


@dataclass(frozen=True, slots=True, eq=False)
class MarkerSlot:
    """A marker slot whose metadata was validated once for all of its groups.

    A table of revision 1, written before release 1.3.0, keeps the statistics
    it stored, rounded to five decimals except for the p-values, which come
    from the tie- and continuity-corrected normal approximation. Its slot
    reports ``MARKER_REVISION_1_P_VALUE_POLICY``, and its tables gain
    ``neg_log10_p_value``, derived from the stored ``p_value``.

    Attributes:
        group: The slot's Zarr group.
        columns: The stored statistic columns, in stored order.
        feature_index: The tested feature of each stored row.
        skipped_groups: The cell count of each group that the slot skipped.
        p_value_policy: How the table's p-values were computed:
            ``MARKER_P_VALUE_POLICY``, or ``MARKER_REVISION_1_P_VALUE_POLICY``
            for a table of revision 1.
    """

    group: zarr.Group
    columns: list[str]
    feature_index: np.ndarray
    skipped_groups: dict[str, int]
    p_value_policy: str

    @classmethod
    def read(cls, slot_group: zarr.Group, n_features: int) -> "MarkerSlot":
        """Validate a slot's metadata and its feature index once.

        Args:
            slot_group: The marker slot.
            n_features: Number of features of the assay.
        """
        return cls(slot_group, *_canonical_slot(slot_group, n_features))

    def group_table(
        self,
        cluster_group: zarr.Group,
        feature_names: np.ndarray,
        *,
        group_id: Any,
        sort_by: MarkerSortKey = "score",
    ) -> pd.DataFrame:
        """Read one stored group of the slot into a named, ranked table.

        Args:
            cluster_group: The group's Zarr group in the slot.
            feature_names: Names of every feature of the assay.
            group_id: The label of the table's ``group_id`` column.
            sort_by: The statistic that orders the rows.
        """
        stats = _canonical_cluster_stats(
            cluster_group, self.columns, self.feature_index.shape[0]
        )
        columns = self.columns
        if self.p_value_policy == MARKER_REVISION_1_P_VALUE_POLICY:
            stats, columns = _with_neg_log10_p_values(stats, columns)
        return _ranked_table(
            stats, columns, self.feature_index, feature_names, group_id, sort_by
        )

    def table(
        self,
        group_name: str,
        feature_names: np.ndarray,
        *,
        sort_by: MarkerSortKey = "score",
    ) -> pd.DataFrame:
        """Read one stored group into a named, ranked table.

        A skipped group gives a table without rows.

        Args:
            group_name: A stored or skipped group label.
            feature_names: Names of every feature of the assay.
            sort_by: The statistic that orders the rows.
        """
        if group_name in self.skipped_groups:
            return _ranked_table(
                np.empty((0, len(self.columns))),
                self.columns,
                np.empty(0, dtype=np.int64),
                feature_names,
                group_name,
                sort_by,
            )
        return self.group_table(
            _marker_group(self.group, group_name),
            feature_names,
            group_id=group_name,
            sort_by=sort_by,
        )


def load_marker_table(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
    sort_by: MarkerSortKey = "score",
) -> pd.DataFrame:
    """Read one canonical marker group into a named, ranked table.

    A table of revision 1 reads as :class:`MarkerSlot` describes.
    """
    return MarkerSlot.read(slot_group, len(feature_names)).group_table(
        cluster_group, feature_names, group_id=group_id, sort_by=sort_by
    )
