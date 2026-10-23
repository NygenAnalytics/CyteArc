"""Marker tables: filters and orders of their readers, exact p-values of small
groups, skipped groups, stored precision, and the revision that changed them."""

import itertools
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr
from scipy.stats import PermutationMethod, mannwhitneyu, rankdata

import cytearc.features.markers.search as marker_search_module
import cytearc.features.markers.table as table_module
import cytearc.storage.operation_revisions as operation_revisions
from cytearc.datastore.datastore import DataStore
from cytearc.features.markers import (
    RankMarkerResult,
    find_markers_by_rank,
    sort_marker_results,
)
from cytearc.features.markers.rank import (
    MARKER_SORT_KEYS,
    _exact_feature_p_values,
    _exact_marker_p_values,
    _exact_marker_p_values_gene_major,
    _library_size_value,
    _rank_sum_null_counts,
    exact_batch_p_values,
    exact_rank_scratch_bytes,
    exact_two_sided_p_value,
    marker_sort_key,
)
from cytearc.features.markers.table import (
    MARKER_P_VALUE_POLICY,
    MARKER_REVISION_1_P_VALUE_POLICY,
    MARKER_REVISION_1_STAT_COLUMNS,
    MARKER_STAT_COLUMNS,
    MarkerFilter,
    MarkerSlot,
    exact_marker_groups,
    rank_marker_tested_groups,
)
from cytearc.features.statistical import (
    MANN_WHITNEY_P_VALUE_POLICY,
    adjust_pvalues,
)
from cytearc.storage.artifacts import artifact_path
from tests.storage_helpers import write_count_store

EVERY_ROW = {"min_frac_exp": None, "min_auc": None, "max_p_adj": None}
TABLE_COLUMNS = ["group_id", "feature_name", "feature_index", *MARKER_STAT_COLUMNS]


def _store(directory, counts: np.ndarray, labels: np.ndarray, *, assays=("RNA",)):
    """Write float32 counts and return the store, its clusters, and features."""
    path = str(directory / "markers.zarr")
    write_count_store(path, {assay: counts for assay in assays}, "float32")
    store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
    store.cells.insert("everyone", np.ones(len(labels), dtype=bool), overwrite=True)
    store.cells.insert("labels", np.asarray(labels, dtype=object), overwrite=True)
    clusters = store.snapshot_cluster_labels(
        "labels", cell_selection=store.snapshot_cell_selection("everyone")
    )
    return store, clusters


def _library_size_values(store: DataStore, counts: np.ndarray) -> np.ndarray:
    """Return the float32 library-size values that RNA marker search ranks."""
    totals = counts.sum(axis=1, dtype=np.float64)
    return (store.RNA.sf * counts / totals[:, None]).astype(np.float32)


def _float32_counts(values: np.ndarray) -> np.ndarray:
    """Return counts that float32 holds exactly, as float64."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


# Kernels. Each runs compiled and as Python, so that both agree.


def _subset_sum_counts(doubled: np.ndarray, size: int, width: int) -> np.ndarray:
    """Count every subset of ``size`` observations by its doubled rank sum."""
    counts = np.zeros(width, dtype=np.int64)
    for chosen in itertools.combinations(range(len(doubled)), size):
        counts[doubled[list(chosen)].sum()] += 1
    return counts


def test_rank_sum_null_counts_enumerate_every_subset_of_tied_values() -> None:
    values = np.array([0.0, 0.0, 0.0, 1.5, 2.0, 2.0, 3.0, 7.0])
    doubled = np.rint(2 * rankdata(values)).astype(np.int64)
    distinct, multiplicities = np.unique(doubled, return_counts=True)
    max_size = 4
    width = int(np.sort(doubled)[-max_size:].sum()) + 1

    for kernel in (_rank_sum_null_counts, _rank_sum_null_counts.py_func):
        counts = kernel(distinct, multiplicities.astype(np.int64), max_size, width)
        for size in range(max_size + 1):
            np.testing.assert_array_equal(
                counts[size], _subset_sum_counts(doubled, size, width)
            )
            assert counts[size].sum() == math.comb(len(values), size)


def test_exact_two_sided_p_value_doubles_the_smaller_tail_up_to_one() -> None:
    # Seven splits with doubled rank sums 0, 2, 2, 3, 3, 3, and 4.
    distribution = np.array([1, 0, 2, 3, 1], dtype=np.int64)
    for kernel in (exact_two_sided_p_value, exact_two_sided_p_value.py_func):
        assert kernel(distribution, 0) == 2 / 7
        assert kernel(distribution, 4) == 2 / 7
        assert kernel(distribution, 2) == 6 / 7
        assert kernel(distribution, 3) == 1.0


def _reference_p_value(inside: np.ndarray, outside: np.ndarray) -> float:
    """Return SciPy's exact p-value, enumerating splits when values tie.

    Values are widened to float64, which keeps their order and ties, because
    SciPy computes the p-value of float32 values in float32.
    """
    inside = np.asarray(inside, dtype=np.float64)
    outside = np.asarray(outside, dtype=np.float64)
    pooled = np.concatenate([inside, outside])
    method = (
        "exact"
        if len(np.unique(pooled)) == len(pooled)
        else PermutationMethod(n_resamples=np.inf)
    )
    return float(mannwhitneyu(inside, outside, method=method).pvalue)


def _asymptotic_p_value(inside: np.ndarray, outside: np.ndarray) -> float:
    """Return SciPy's tie- and continuity-corrected normal approximation."""
    return float(
        mannwhitneyu(
            np.asarray(inside, dtype=np.float64),
            np.asarray(outside, dtype=np.float64),
            method="asymptotic",
            use_continuity=True,
        ).pvalue
    )


def test_exact_feature_p_values_match_scipy_for_groups_and_mirrored_rests() -> None:
    rng = np.random.default_rng(5)
    codes = np.repeat(np.arange(3), [2, 3, 7]).astype(np.int64)
    sizes = np.bincount(codes).astype(np.int64)
    # Group 2 holds seven of twelve cells, so its five other cells are the
    # smaller side, and group 1 is left to the normal approximation.
    exact_groups = np.array([0, 2], dtype=np.int64)
    tie_free = rng.normal(size=12)
    tied = rng.integers(0, 3, 12).astype(np.float64)
    tied[codes == 0] = 5.0

    for values in (tie_free, tied):
        for kernel in (_exact_feature_p_values, _exact_feature_p_values.py_func):
            out = np.full(3, np.nan)
            kernel(values, codes, sizes, exact_groups, 5, out)
            for group in exact_groups:
                expected = _reference_p_value(
                    values[codes == group], values[codes != group]
                )
                assert out[group] == pytest.approx(expected, rel=1e-12)
            assert np.isnan(out[1])
    # Both of group 0's cells hold the largest value: one split in C(12, 2).
    out = np.full(3, np.nan)
    _exact_feature_p_values(tied, codes, sizes, exact_groups, 5, out)
    assert out[0] == 2 / math.comb(12, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_batch_p_values_rank_each_feature_as_the_dense_kernel(dtype) -> None:
    rng = np.random.default_rng(9)
    codes = np.repeat(np.arange(2), [3, 5]).astype(np.int64)
    sizes = np.array([3, 5], dtype=np.int64)
    exact_groups = np.array([0, 1], dtype=np.int64)
    data = rng.integers(0, 4, (8, 3)).astype(dtype)

    batch = exact_batch_p_values(data, codes, sizes, exact_groups, 3)
    for kernel in (_exact_marker_p_values, _exact_marker_p_values.py_func):
        out = np.full((3, 2), np.nan)
        kernel(np.ascontiguousarray(data), codes, sizes, exact_groups, 3, out)
        np.testing.assert_array_equal(out, batch)
    for feature in range(3):
        for group in (0, 1):
            expected = _reference_p_value(
                data[codes == group, feature], data[codes != group, feature]
            )
            assert batch[feature, group] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("log_transform", [False, True])
def test_library_size_value_rounds_its_float64_value_to_float32(log_transform) -> None:
    exact = 1000.0 * 3.0 / 7.0
    if log_transform:
        exact = float(np.log1p(exact))
    for kernel in (_library_size_value, _library_size_value.py_func):
        value = float(kernel(3, 7.0, 1000.0, log_transform))
        assert value == float(np.float32(exact))
        assert value != exact


@pytest.mark.parametrize("log_transform", [False, True])
def test_exact_gene_major_p_values_rank_library_size_values(log_transform) -> None:
    # Features by cells; feature 1 is skipped, and the others go to rows 1 and 0.
    raw = np.array(
        [[0, 3, 1, 0, 5, 2], [4, 0, 0, 1, 2, 2], [1, 1, 1, 1, 1, 9]], dtype=np.uint32
    )
    totals = np.array([10.0, 20.0, 5.0, 8.0, 16.0, 40.0])
    destinations = np.array([1, -1, 0], dtype=np.int64)
    codes = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    sizes = np.array([2, 4], dtype=np.int64)
    exact_groups = np.array([0, 1], dtype=np.int64)
    normalized = 1000.0 * raw / totals
    if log_transform:
        normalized = np.log1p(normalized)
    values = normalized.astype(np.float32)

    for kernel in (
        _exact_marker_p_values_gene_major,
        _exact_marker_p_values_gene_major.py_func,
    ):
        out = np.full((2, 2), np.nan)
        kernel(
            raw,
            totals,
            1000.0,
            log_transform,
            codes,
            sizes,
            exact_groups,
            2,
            destinations,
            out,
        )
        for row, destination in ((0, 1), (2, 0)):
            np.testing.assert_array_equal(
                out[destination],
                exact_batch_p_values(
                    values[row][:, None], codes, sizes, exact_groups, 2
                )[0],
            )


def test_exact_rank_scratch_counts_every_array_of_one_call() -> None:
    # Ten cells hold four int64 arrays and a float32 value each, three groups
    # one int64 sum each, and the counts of subsets of up to two cells reach
    # sums of 2 * 2 * 10.
    assert exact_rank_scratch_bytes(n_cells=10, n_groups=3, max_size=2) == (
        10 * (4 * 8 + 4) + 3 * 8 + 3 * 41 * 8
    )


def test_marker_groups_are_tested_and_exact_by_their_sizes() -> None:
    sizes = np.array([1, 2, 5, 12])
    assert rank_marker_tested_groups(sizes).tolist() == [False, True, True, True]
    # C(20, 2) and C(20, 5) are at most 100,000 splits; C(20, 8) is not.
    assert exact_marker_groups(sizes).tolist() == [False, True, True, False]
    # A group with a single other cell cannot be compared either.
    assert rank_marker_tested_groups(np.array([5, 1])).tolist() == [False, False]
    assert rank_marker_tested_groups(np.array([2, 1, 1])).tolist() == [
        True,
        False,
        False,
    ]
    assert MARKER_P_VALUE_POLICY == MANN_WHITNEY_P_VALUE_POLICY


# Rank marker results.


def _result(exact_p_values: np.ndarray | None = None) -> RankMarkerResult:
    statistics = np.random.default_rng(3).random((3, 3, 8))
    statistics[:, :, 6] = [[1.0, -2.0, 40.0], [0.0, 3.0, -50.0], [-0.5, 0.25, 60.0]]
    return RankMarkerResult(
        group_ids=np.array(["a", "b", "c"]),
        group_sizes=np.array([1, 2, 30]),
        feature_index=np.array([4, 7, 9]),
        statistics=statistics,
        exact_p_values=exact_p_values,
    )


def test_rank_marker_results_store_exact_p_values_and_skip_small_groups() -> None:
    exact = np.full((3, 3), np.nan)
    exact[:, 1] = [0.5, 1.0, 0.02]
    result = _result(exact)

    assert result.tested_group_ids.tolist() == ["b", "c"]
    assert result.skipped_group_ids.tolist() == ["a"]
    stored = result.stored_statistics("b")
    np.testing.assert_array_equal(stored[:, 6], [0.5, 1.0, 0.02])
    np.testing.assert_array_equal(
        stored[:, 8], adjust_pvalues(np.array([0.5, 1.0, 0.02]), "fdr_bh")
    )
    np.testing.assert_array_equal(stored[:, 9], -np.log10([0.5, 1.0, 0.02]))
    # A p-value of 1 stores 0, not -0.
    assert not np.signbit(stored[1, 9])
    with pytest.raises(ValueError, match="'a' is skipped: it has 1 of 33 cells"):
        result.stored_statistics("a")
    with pytest.raises(ValueError, match="'a' is skipped"):
        result.table("a", np.array([f"g{i}" for i in range(10)]))


def test_neg_log10_p_values_stay_finite_and_ordered_where_p_values_underflow() -> None:
    result = _result()
    stored = result.stored_statistics("c")
    z = np.array([40.0, 50.0, 60.0])

    # Two-sided p-values of these z statistics are below the float64 range.
    np.testing.assert_array_equal(stored[:, 6], [0.0, 0.0, 0.0])
    expected = -(np.log(2.0) + log_ndtr(-z)) / np.log(10.0)
    np.testing.assert_array_equal(stored[:, 9], expected)
    # -log10 of 2 * Phi(-z) is about (z**2 / 2 + log(z * sqrt(pi / 2))) / log(10).
    approximation = (z**2 / 2 + np.log(z * np.sqrt(np.pi / 2))) / np.log(10)
    np.testing.assert_allclose(stored[:, 9], approximation, rtol=1e-3)
    table = result.table("c", np.array([f"g{i}" for i in range(10)]), sort_by="p_value")
    assert table["feature_index"].tolist() == [9, 7, 4]


@pytest.mark.parametrize(
    ("exact", "message"),
    [
        (np.full((3, 3), np.nan, dtype=np.float32), "float64"),
        (np.full((3, 2), np.nan), "one column per group"),
        (np.array([[np.nan, 0.5, np.nan]] * 2 + [[np.nan, np.nan, np.nan]]), "whole"),
        (np.array([[0.5, np.nan, np.nan]] * 3), "tested groups"),
        (np.array([[np.nan, 0.0, np.nan]] * 3), r"\(0, 1\]"),
        (np.array([[np.nan, 1.5, np.nan]] * 3), r"\(0, 1\]"),
        (np.array([[np.nan, np.inf, np.nan]] * 3), r"\(0, 1\]"),
    ],
    ids=["dtype", "shape", "partial", "skipped", "zero", "above-one", "infinite"],
)
def test_rank_marker_results_reject_inconsistent_exact_p_values(exact, message):
    with pytest.raises(ValueError, match=message):
        _result(exact)


# Filters and orders.


def test_marker_filter_keeps_rows_that_meet_every_threshold() -> None:
    table = pd.DataFrame(
        {
            "score": [0.1, 0.3, 0.5, 0.7],
            "frac_exp": [0.2, 0.1, 0.9, 0.5],
            "auc": [0.6, 0.95, 0.59, 0.8],
            "p_value_adjusted": [0.05, 0.01, 0.001, 0.2],
        }
    )
    # Each threshold keeps its boundary.
    assert MarkerFilter().keep(table).tolist() == [True] * 4
    assert MarkerFilter(min_score=0.3).keep(table).tolist() == [
        False,
        True,
        True,
        True,
    ]
    assert MarkerFilter(min_frac_exp=0.2).keep(table).tolist() == [
        True,
        False,
        True,
        True,
    ]
    assert MarkerFilter(min_auc=0.6).keep(table).tolist() == [True, True, False, True]
    assert MarkerFilter(max_p_adj=0.05).keep(table).tolist() == [
        True,
        True,
        True,
        False,
    ]
    defaults = MarkerFilter(min_frac_exp=0.2, min_auc=0.6, max_p_adj=0.05)
    assert defaults.keep(table).tolist() == [True, False, False, False]
    resolved = MarkerFilter(min_score=np.int64(0), min_auc=-np.inf)
    assert type(resolved.min_score) is float and resolved.min_score == 0.0
    assert resolved.keep(table).all()


@pytest.mark.parametrize(
    ("threshold", "error", "message"),
    [
        (True, TypeError, "min_auc must be a real number or None"),
        (np.bool_(False), TypeError, "min_auc must be a real number or None"),
        ("0.6", TypeError, "min_auc must be a real number or None"),
        (np.nan, ValueError, "min_auc must not be NaN"),
    ],
)
def test_marker_filter_rejects_thresholds_that_are_not_real_numbers(
    threshold, error, message
) -> None:
    with pytest.raises(error, match=message):
        MarkerFilter(min_auc=threshold)


def _sort_table() -> pd.DataFrame:
    """Rows whose statistics tie in every order but one."""
    return pd.DataFrame(
        {
            "feature_name": ["g0", "g1", "g2", "g3", "g4", "g4"],
            "feature_index": [0, 1, 2, 3, 4, 5],
            "score": [0.5, 0.5, 0.7, 0.5, 0.5, 0.5],
            "auc": [0.9, 0.9, 0.6, 0.95, 0.1, 0.1],
            # Rows 0, 1, 4, and 5 underflowed to 0, which -log10 still orders.
            "p_value": [0.0, 0.0, 0.01, 0.01, 0.0, 0.0],
            "neg_log10_p_value": [300.0, 310.0, 2.0, 2.0, 310.0, 310.0],
            "fold_change": [2.0, np.inf, np.nan, 3.0, 0.5, 0.5],
        }
    )


@pytest.mark.parametrize(
    ("sort_by", "order"),
    [
        # Score ties fall to the p-value, then -log10 p, then name and index.
        ("score", [2, 1, 4, 5, 0, 3]),
        ("auc", [3, 1, 0, 2, 4, 5]),
        # Equal significance falls to the AUC, so positive markers lead.
        ("p_value", [1, 4, 5, 0, 3, 2]),
        # Infinite fold changes come first and NaN last.
        ("fold_change", [1, 3, 0, 4, 5, 2]),
    ],
)
def test_marker_orders_rank_best_first_with_deterministic_ties(sort_by, order) -> None:
    table = _sort_table()
    assert sort_marker_results(table, sort_by)["feature_index"].tolist() == order
    # The order does not depend on the input row order.
    shuffled = table.iloc[[5, 3, 1, 0, 4, 2]]
    assert sort_marker_results(shuffled, sort_by)["feature_index"].tolist() == order


def test_marker_orders_without_neg_log10_break_p_value_ties_by_name() -> None:
    table = _sort_table().drop(columns="neg_log10_p_value")
    assert sort_marker_results(table)["feature_index"].tolist() == [2, 0, 1, 4, 5, 3]
    assert MARKER_SORT_KEYS == ("score", "auc", "p_value", "fold_change")
    with pytest.raises(ValueError, match="sort_by must be one of 'score'"):
        sort_marker_results(table, "z")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="got None"):
        marker_sort_key(None)


# Marker search over small groups.


@pytest.fixture(scope="module")
def small_groups(tmp_path_factory):
    """A 20-cell store with groups of 1, 2, 5, and 12 cells and its tables."""
    rng = np.random.default_rng(7)
    labels = np.repeat(np.array(["a", "b", "c", "d"], dtype=object), [1, 2, 5, 12])
    counts = rng.uniform(1.0, 40.0, size=(len(labels), 5)).astype(np.float32)
    counts[labels != "c", 1] = 0.0  # a marker of c, tied at zero elsewhere
    counts[:, 2] = rng.integers(0, 3, len(labels))  # heavily tied
    counts[labels == "b", 3] += 200.0  # a marker of b without ties
    counts = _float32_counts(counts)
    store, clusters = _store(
        tmp_path_factory.mktemp("small_groups"), counts, labels, assays=("RNA", "ADT")
    )
    markers = {}
    for assay in ("RNA", "ADT"):
        with pytest.warns(UserWarning, match=r"skips 1 groups .*'a' \(1 cell\)"):
            markers[assay] = store.markers.search(
                clusters,
                from_assay=assay,
                features=store.features.universe(from_assay=assay),
            )
    return store, clusters, counts, labels, markers


def _ranked_values(store, assay: str, counts: np.ndarray) -> np.ndarray:
    if assay == "RNA":
        return _library_size_values(store, counts)
    cells = np.arange(counts.shape[0])
    features = np.arange(counts.shape[1])
    return np.asarray(store.ADT.normed(cell_idx=cells, feat_idx=features).compute())


@pytest.mark.parametrize("assay", ["RNA", "ADT"])
def test_small_groups_get_exact_p_values_and_single_cells_are_skipped(
    small_groups, assay
) -> None:
    store, _, counts, labels, markers = small_groups
    marker = markers[assay]
    slot = store.artifacts.load(marker)
    assert dict(slot.attrs["skipped_groups"]) == {"a": 1}
    assert slot.attrs["p_value_policy"] == MANN_WHITNEY_P_VALUE_POLICY
    assert sorted(slot.group_keys()) == ["b", "c", "d"]
    # The skipped cell stays in the rest of every tested group.
    assert slot["b"].attrs["n_reference"] == 18

    with pytest.warns(UserWarning, match=r"skipped groups .*'a' \(1 cell\)"):
        table = store.markers.load(marker, **EVERY_ROW)
    assert list(dict.fromkeys(table["group_id"])) == ["b", "c", "d"]
    values = _ranked_values(store, assay, counts)
    for (group, feature), row in table.set_index(
        ["group_id", "feature_index"]
    ).iterrows():
        inside = values[labels == group, feature]
        outside = values[labels != group, feature]
        if group == "d":
            # C(20, 8) splits are more than 100,000.
            assert row["p_value"] == pytest.approx(
                _asymptotic_p_value(inside, outside), rel=1e-12
            )
        else:
            assert row["p_value"] == pytest.approx(
                _reference_p_value(inside, outside), rel=1e-12
            )
        assert row["neg_log10_p_value"] == pytest.approx(
            -np.log10(row["p_value"]), rel=1e-12, abs=1e-15
        )
    # Both cells of b hold the two largest values of feature 3.
    assert set(np.argsort(values[:, 3])[-2:]) == {1, 2}
    b_marker = table[(table["group_id"] == "b") & (table["feature_index"] == 3)]
    assert b_marker["p_value"].item() == 2 / math.comb(20, 2)


def test_skipped_groups_have_no_rows_and_an_empty_export_column(
    small_groups, tmp_path
) -> None:
    store, _, _, _, markers = small_groups
    with pytest.warns(UserWarning, match=r"'a' \(1 cell\)"):
        skipped = store.markers.load(markers["RNA"], group_id="a")
    assert skipped.empty
    assert skipped.columns.tolist() == TABLE_COLUMNS

    with pytest.warns(UserWarning, match=r"'a' \(1 cell\)"):
        store.markers.export_csv(
            markers["RNA"], str(tmp_path / "markers.csv"), **EVERY_ROW
        )
    exported = pd.read_csv(tmp_path / "markers.csv")
    assert exported.columns.tolist() == ["a", "b", "c", "d"]
    assert exported["a"].isna().all()
    assert exported["b"].notna().all()


def test_marker_search_names_the_groups_when_none_can_be_tested(tmp_path) -> None:
    labels = np.array(["solo", "pair", "pair"], dtype=object)
    store, clusters = _store(tmp_path, _float32_counts(np.ones((3, 2))), labels)

    with pytest.raises(
        ValueError,
        match=r"the groups are 'pair' \(2 cells\), 'solo' \(1 cell\)\. Merge",
    ):
        store.markers.search(clusters, features=store.features.universe())
    assert store.artifacts.list(kind="marker_table") == []


@pytest.mark.parametrize("n_cells", [447, 448])
def test_two_cell_groups_are_exact_up_to_the_split_limit(tmp_path, n_cells) -> None:
    labels = np.array(["pair"] * 2 + ["rest"] * (n_cells - 2), dtype=object)
    counts = np.zeros((n_cells, 2), dtype=np.float32)
    counts[:2, 0] = [3.0, 5.0]  # expressed only by the pair
    counts[:, 1] = 1 + np.arange(n_cells) % 7
    counts = _float32_counts(counts)
    store, clusters = _store(tmp_path, counts, labels)
    marker = store.markers.search(clusters, features=store.features.universe())
    table = store.markers.load(marker, group_id="pair", **EVERY_ROW)
    p_value = table.set_index("feature_index").loc[0, "p_value"]

    if n_cells == 447:
        # C(447, 2) = 99,681 splits, of which one reaches the pair's ranks.
        assert p_value == 2 / math.comb(447, 2)
    else:
        values = _library_size_values(store, counts)[:, 0]
        expected = _asymptotic_p_value(values[:2], values[2:])
        assert p_value == pytest.approx(expected, rel=1e-9)
        # The normal approximation lies far below any split's probability.
        assert p_value < 1e-10 / math.comb(448, 2)


# A fine partition: 60 groups of five cells, each with its own marker.


@pytest.fixture(scope="module")
def sixty_groups(tmp_path_factory):
    rng = np.random.default_rng(60)
    n_groups, size = 60, 5
    labels = np.repeat(
        np.array([f"g{group:02d}" for group in range(n_groups)], dtype=object), size
    )
    counts = np.zeros((len(labels), n_groups + 2), dtype=np.float32)
    for group in range(n_groups):
        rows = slice(group * size, (group + 1) * size)
        counts[rows, group] = rng.choice(np.arange(1, 60), size, replace=False)
    counts[:, n_groups:] = rng.integers(1, 20, (len(labels), 2))
    counts = _float32_counts(counts)
    store, clusters = _store(tmp_path_factory.mktemp("sixty_groups"), counts, labels)
    marker = store.markers.search(clusters, features=store.features.universe())
    return store, marker, counts, labels


def test_default_filters_keep_every_group_marker_of_a_fine_partition(
    sixty_groups, tmp_path
) -> None:
    store, marker, counts, labels = sixty_groups
    values = _library_size_values(store, counts)
    groups = list(dict.fromkeys(labels))

    default = store.markers.load(marker)
    assert list(dict.fromkeys(default["group_id"])) == groups
    leaders = default.groupby("group_id", sort=False).head(1).set_index("group_id")
    assert leaders["feature_index"].tolist() == list(range(60))
    for group, row in leaders.iterrows():
        position = groups.index(group)
        # Distinct values in the group and zeros elsewhere: AUC 1, and the
        # dense-rank score (5 + 3) / (5 + 1 + 2 * 60) = 4 / 63.
        assert len(np.unique(values[labels == group, position])) == 5
        assert row["auc"] == 1.0 and row["frac_exp"] == 1.0
        assert row["frac_exp_rest"] == 0.0 and row["fold_change"] == np.inf
        assert row["score"] == 4 / 63
        assert row["p_value_adjusted"] <= 0.05
    # The former defaults, a score of at least 0.25 and no AUC or p-value
    # filter, kept no row at all.
    assert store.markers.load(
        marker, min_score=0.25, min_frac_exp=0.2, min_auc=None, max_p_adj=None
    ).empty

    full = store.markers.load(marker, **EVERY_ROW)
    expected = full[
        (full["frac_exp"] >= 0.2)
        & (full["auc"] >= 0.6)
        & (full["p_value_adjusted"] <= 0.05)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(default, expected)

    store.markers.export_csv(marker, str(tmp_path / "markers.csv"))
    exported = pd.read_csv(tmp_path / "markers.csv")
    assert exported.columns.tolist() == groups
    assert exported.iloc[0].tolist() == [f"RNA{group}" for group in range(60)]


@pytest.mark.parametrize("sort_by", ["score", "auc", "p_value", "fold_change"])
def test_marker_readers_order_each_group_by_sort_by(sixty_groups, sort_by) -> None:
    store, marker, _, _ = sixty_groups
    full = store.markers.load(marker, **EVERY_ROW)
    ordered = store.markers.load(marker, sort_by=sort_by, **EVERY_ROW)

    for group in ("g00", "g31", "g59"):
        expected = sort_marker_results(full[full["group_id"] == group], sort_by)
        pd.testing.assert_frame_equal(
            ordered[ordered["group_id"] == group].reset_index(drop=True),
            expected.reset_index(drop=True),
        )


def test_marker_readers_check_filters_and_order_before_reading(
    sixty_groups, monkeypatch
) -> None:
    store, marker, _, _ = sixty_groups

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the marker table was read")

    monkeypatch.setattr(store, "_resolve_marker_group", refuse)
    with pytest.raises(ValueError, match="sort_by must be one of"):
        store.markers.load(marker, sort_by="z")
    with pytest.raises(TypeError, match="max_p_adj must be a real number"):
        store.markers.export_csv(marker, "unused.csv", max_p_adj="0.05")
    with pytest.raises(ValueError, match="min_score must not be NaN"):
        store.markers.load(marker, min_score=np.nan)


def test_marker_load_reads_the_feature_index_once_for_every_group(
    sixty_groups, monkeypatch
) -> None:
    store, marker, _, _ = sixty_groups
    reads: list[str] = []
    original = table_module._array_values

    def counting(group, name):
        reads.append(name)
        return original(group, name)

    monkeypatch.setattr(table_module, "_array_values", counting)
    store.markers.load(marker)
    assert reads.count("feature_index") == 1
    assert reads.count("stats") == 60


# Stored precision, reuse, and the revision.


def test_marker_tables_store_full_precision_means_of_small_values(tmp_path) -> None:
    import cytearc.assay.normalization as normalization

    rng = np.random.default_rng(17)
    labels = np.repeat(np.array(["x", "y"], dtype=object), 6)
    # Means near 1e-5, as TF-IDF values of ATAC peaks are.
    counts = _float32_counts(rng.uniform(1e-6, 4e-5, size=(12, 3)))
    store, clusters = _store(tmp_path, counts, labels)
    store.RNA.normMethod = normalization.norm_dummy
    marker = store.markers.search(clusters, features=store.features.universe())

    table = store.markers.load(marker, **EVERY_ROW).set_index(
        ["group_id", "feature_index"]
    )
    for group in ("x", "y"):
        for feature in range(3):
            inside = counts[labels == group, feature]
            outside = counts[labels != group, feature]
            row = table.loc[(group, feature)]
            assert row["mean"] == pytest.approx(inside.mean(), rel=1e-12)
            assert row["mean_rest"] == pytest.approx(outside.mean(), rel=1e-12)
            assert row["fold_change"] == row["mean"] / row["mean_rest"]


def test_marker_search_records_revision_2_and_recomputes_revision_1_tables(
    sixty_groups, tmp_path, monkeypatch
) -> None:
    import shutil

    store, marker, _, _ = sixty_groups
    shutil.copytree(store.zarr_loc, tmp_path / "copy.zarr")
    copy = DataStore(
        str(tmp_path / "copy.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    inputs = copy.artifacts.inspect(marker).inputs
    from cytearc import ArtifactRef

    clusters = ArtifactRef.from_dict(inputs["clusters"])
    features = ArtifactRef.from_dict(inputs["feature_selection"])
    status = copy.artifacts.inspect(marker)
    assert status.revision == 2 and status.is_current
    assert copy.markers.search(clusters, features=features) == marker

    # A table that an earlier release wrote records no revision.
    monkeypatch.setattr(
        operation_revisions, "OPERATION_REVISIONS", MappingProxyType({})
    )
    earlier = copy.markers.search(clusters, features=features, invalidate_cache=True)
    assert earlier != marker
    assert copy.artifacts.inspect(earlier).revision == 1
    monkeypatch.undo()

    calls = 0
    original = marker_search_module.find_markers_by_rank

    def tracked(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    import cytearc.features.markers as marker_algorithms

    monkeypatch.setattr(marker_algorithms, "find_markers_by_rank", tracked)
    # Planning prefers the current table over the revision 1 one.
    assert copy.markers.search(clusters, features=features) == marker
    superseded = copy.artifacts.inspect(earlier)
    assert superseded.revision == 1 and not superseded.is_current
    assert superseded.current_revision == 2
    assert calls == 0


def test_revision_1_marker_tables_are_recomputed_not_reused(
    small_groups, tmp_path, monkeypatch
) -> None:
    import shutil

    store, clusters, _, _, _ = small_groups
    shutil.copytree(store.zarr_loc, tmp_path / "copy.zarr")
    copy = DataStore(
        str(tmp_path / "copy.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    features = copy.features.snapshot(feature_indexes=np.array([0, 1]))
    monkeypatch.setattr(
        operation_revisions, "OPERATION_REVISIONS", MappingProxyType({})
    )
    with pytest.warns(UserWarning, match="skips 1 groups"):
        earlier = copy.markers.search(clusters, features=features)
    monkeypatch.undo()

    with pytest.warns(UserWarning, match="skips 1 groups"):
        current = copy.markers.search(clusters, features=features)
    assert current != earlier
    assert copy.artifacts.inspect(earlier).revision == 1
    status = copy.artifacts.inspect(current)
    assert status.revision == 2 and status.is_current
    assert not copy.artifacts.inspect(earlier).is_current


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        # Without the policy, a slot must have the layout of revision 1.
        (
            lambda slot: slot.attrs.__delitem__("p_value_policy"),
            "without p_value_policy metadata must store the revision 1 stat_columns",
        ),
        (
            lambda slot: slot.attrs.update({"p_value_policy": "normal"}),
            "metadata 'p_value_policy' must be",
        ),
        (
            lambda slot: slot.attrs.update({"skipped_groups": {}}),
            None,
        ),
    ],
    ids=["no-policy", "other-policy", "skipped-groups"],
)
def test_marker_tables_with_other_p_values_or_skips_are_recomputed(
    small_groups, tmp_path, monkeypatch, edit, message
) -> None:
    import shutil

    store, clusters, _, _, markers = small_groups
    shutil.copytree(store.zarr_loc, tmp_path / "copy.zarr")
    copy = DataStore(
        str(tmp_path / "copy.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    saved = markers["RNA"]
    edit(copy.zw[artifact_path(saved)])
    if message is not None:
        with pytest.raises(ValueError, match=message):
            copy.markers.load(saved)

    with pytest.warns(UserWarning, match="skips 1 groups"):
        recomputed = copy.markers.search(
            clusters, features=copy.features.universe(from_assay="RNA")
        )
    assert recomputed != saved
    slot = copy.zw[artifact_path(recomputed)]
    assert dict(slot.attrs["skipped_groups"]) == {"a": 1}
    assert slot.attrs["p_value_policy"] == MANN_WHITNEY_P_VALUE_POLICY


# Tables of revision 1, which the readers load under an explicit policy.

REVISION_1_COLUMNS = list(MARKER_REVISION_1_STAT_COLUMNS)


def _replace_stats(cluster, stats: np.ndarray) -> None:
    del cluster["stats"]
    cluster.create_array("stats", data=stats)


def _as_revision_1(slot) -> None:
    """Rewrite a marker slot in the layout that revision 1 of the search wrote.

    Revision 1 stored no neg_log10_p_value, rounded every statistic except
    the two p-value columns to five decimals, and recorded neither
    p_value_policy nor skipped_groups.
    """
    stored = list(slot.attrs["stat_columns"])
    kept = [stored.index(name) for name in REVISION_1_COLUMNS]
    rounded = [
        position
        for position, name in enumerate(REVISION_1_COLUMNS)
        if name not in ("p_value", "p_value_adjusted")
    ]
    for name in slot.group_keys():
        stats = np.asarray(slot[name]["stats"][:])[:, kept]
        stats[:, rounded] = np.round(stats[:, rounded], 5)
        _replace_stats(slot[name], stats)
    slot.attrs["stat_columns"] = REVISION_1_COLUMNS
    del slot.attrs["p_value_policy"]
    del slot.attrs["skipped_groups"]


def _set_p_values(slot, group: str, feature: int, p_value: float) -> None:
    """Store one p-value, and its adjusted value, as revision 1 could."""
    row = int(np.flatnonzero(np.asarray(slot["feature_index"][:]) == feature)[0])
    stats = np.asarray(slot[group]["stats"][:])
    stats[row, REVISION_1_COLUMNS.index("p_value")] = p_value
    stats[row, REVISION_1_COLUMNS.index("p_value_adjusted")] = p_value
    _replace_stats(slot[group], stats)


def _copy_store(store: DataStore, directory) -> DataStore:
    import shutil

    shutil.copytree(store.zarr_loc, directory / "copy.zarr")
    return DataStore(
        str(directory / "copy.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


@pytest.fixture(scope="module")
def revision_1_store(tmp_path_factory):
    """A store whose marker table has the revision and layout of revision 1.

    Three groups of eight cells have one marker each, features 0, 2, and 4.
    Group b's marker has a p-value that underflowed to 0, as the normal
    approximation gave revision 1 beyond a z statistic of about 38, and a
    feature of group c has a p-value of 1.
    """
    rng = np.random.default_rng(12)
    labels = np.repeat(np.array(["a", "b", "c"], dtype=object), 8)
    counts = rng.integers(1, 6, size=(len(labels), 6)).astype(np.float64)
    for position, group in enumerate("abc"):
        counts[labels == group, 2 * position] += rng.choice(
            np.arange(30, 60), 8, replace=False
        )
    store, clusters = _store(
        tmp_path_factory.mktemp("revision_1"), _float32_counts(counts), labels
    )
    features = store.features.universe()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(operation_revisions, "OPERATION_REVISIONS", MappingProxyType({}))
        marker = store.markers.search(clusters, features=features)
    slot = store.zw[artifact_path(marker)]
    _as_revision_1(slot)
    _set_p_values(slot, "b", 2, 0.0)
    _set_p_values(slot, "c", 1, 1.0)
    return store, clusters, features, marker


def test_revision_1_marker_tables_load_under_their_policy(
    revision_1_store, tmp_path
) -> None:
    import cytearc.plotting as splt

    store, _, _, marker = revision_1_store
    status = store.artifacts.inspect(marker)
    assert status.revision == 1 and not status.is_current
    slot = store.zw[artifact_path(marker)]
    reader = MarkerSlot.read(slot, slot["feature_names"].shape[0])
    assert reader.p_value_policy == MARKER_REVISION_1_P_VALUE_POLICY
    assert MARKER_REVISION_1_P_VALUE_POLICY == "asymptotic_tie_and_continuity_corrected"
    assert reader.columns == REVISION_1_COLUMNS and reader.skipped_groups == {}
    feature_index = np.asarray(slot["feature_index"][:])
    stored = {name: np.asarray(slot[name]["stats"][:]) for name in "abc"}

    read_only = DataStore(store.zarr_loc, zarr_mode="r", nthreads=1)
    for reopened, name in ((store, "writable"), (read_only, "read_only")):
        full = reopened.markers.load(marker, **EVERY_ROW)
        assert full.columns.tolist() == TABLE_COLUMNS
        assert list(dict.fromkeys(full["group_id"])) == ["a", "b", "c"]
        for group, rows in full.groupby("group_id", sort=False):
            # The statistics come back as revision 1 stored them.
            positions = np.searchsorted(feature_index, rows["feature_index"])
            np.testing.assert_array_equal(
                rows[REVISION_1_COLUMNS].to_numpy(), stored[group][positions]
            )
            p_values = rows["p_value"].to_numpy()
            positive = p_values > 0.0
            np.testing.assert_array_equal(
                rows["neg_log10_p_value"].to_numpy()[positive],
                -np.log10(p_values[positive]),
            )
        derived = full.set_index(["group_id", "feature_index"])["neg_log10_p_value"]
        # A p-value that underflowed to 0 gives +inf, and a p-value of 1 gives 0.
        assert np.isposinf(derived[("b", 2)])
        assert derived[("c", 1)] == 0.0 and not np.signbit(derived[("c", 1)])

        # The filters and orders apply as to a table of revision 2.
        default = reopened.markers.load(marker)
        expected = full[
            (full["frac_exp"] >= 0.2)
            & (full["auc"] >= 0.6)
            & (full["p_value_adjusted"] <= 0.05)
        ].reset_index(drop=True)
        assert set(expected["feature_index"]) >= {0, 2, 4}
        pd.testing.assert_frame_equal(default, expected)
        by_p_value = reopened.markers.load(marker, sort_by="p_value", **EVERY_ROW)
        for group in "abc":
            pd.testing.assert_frame_equal(
                by_p_value[by_p_value["group_id"] == group].reset_index(drop=True),
                sort_marker_results(
                    full[full["group_id"] == group], "p_value"
                ).reset_index(drop=True),
            )
        group_b = by_p_value[by_p_value["group_id"] == "b"]
        assert group_b["feature_index"].iloc[0] == 2

        csv = tmp_path / f"{name}.csv"
        reopened.markers.export_csv(marker, str(csv))
        exported = pd.read_csv(csv)
        assert exported.columns.tolist() == ["a", "b", "c"]
        for group in "abc":
            assert exported[group].dropna().tolist() == (
                default[default["group_id"] == group]["feature_name"].tolist()
            )

        # The heatmap shows each group's two leading markers by score.
        result = splt.marker_heatmap(reopened, marker=marker, topn=2, show=False)
        shown = result.tables["markers"]
        for group in "abc":
            leading = (
                full[full["group_id"] == group]
                .sort_values(
                    ["score", "feature_name"], ascending=[False, True], kind="mergesort"
                )
                .head(2)
            )
            assert shown[shown["group"] == group]["feature_index"].tolist() == (
                leading["feature_index"].tolist()
            )
        result.close()


def test_revision_1_marker_tables_are_recomputed_only_where_writable(
    revision_1_store, tmp_path
) -> None:
    store, clusters, features, marker = revision_1_store
    read_only = DataStore(store.zarr_loc, zarr_mode="r", nthreads=1)
    # The search never reuses the table, and a read-only store cannot write
    # its replacement.
    with pytest.raises(PermissionError, match="run_marker_search requires"):
        read_only.markers.search(clusters, features=features)

    copy = _copy_store(store, tmp_path)
    current = copy.markers.search(clusters, features=features)
    assert current != marker
    status = copy.artifacts.inspect(current)
    assert status.revision == 2 and status.is_current
    slot = copy.zw[artifact_path(current)]
    assert slot.attrs["p_value_policy"] == MARKER_P_VALUE_POLICY
    assert list(slot.attrs["stat_columns"]) == list(MARKER_STAT_COLUMNS)
    # The table of revision 1 stays loadable beside its replacement.
    pd.testing.assert_frame_equal(
        copy.markers.load(marker, **EVERY_ROW),
        store.markers.load(marker, **EVERY_ROW),
    )


def test_revision_1_layouts_are_never_reused_under_current_provenance(
    revision_1_store, tmp_path, monkeypatch
) -> None:
    store, clusters, features, _ = revision_1_store
    copy = _copy_store(store, tmp_path)
    current = copy.markers.search(clusters, features=features)
    _as_revision_1(copy.zw[artifact_path(current)])
    # The edited table reads as one of revision 1.
    loaded = copy.markers.load(current, **EVERY_ROW)
    assert loaded.columns.tolist() == TABLE_COLUMNS

    calls = 0
    original = marker_search_module.find_markers_by_rank

    def tracked(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    import cytearc.features.markers as marker_algorithms

    monkeypatch.setattr(marker_algorithms, "find_markers_by_rank", tracked)
    recomputed = copy.markers.search(clusters, features=features)
    assert recomputed != current and calls == 1
    assert copy.zw[artifact_path(recomputed)].attrs["p_value_policy"] == (
        MARKER_P_VALUE_POLICY
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda slot: slot.__delitem__("feature_index"),
            "require feature_index and stats",
        ),
        (
            lambda slot: slot["a"].__delitem__("stats"),
            "require feature_index and stats",
        ),
        (
            lambda slot: slot.attrs.__delitem__("fold_change_policy"),
            "records no fold_change_policy metadata",
        ),
        (
            lambda slot: slot.attrs.update({"method": "ttest"}),
            "metadata 'method' must be",
        ),
        # A column of revision 2, or a missing one, breaks the layout.
        (
            lambda slot: slot.attrs.update(
                {"stat_columns": [*REVISION_1_COLUMNS, "neg_log10_p_value"]}
            ),
            "must store the revision 1 stat_columns",
        ),
        (
            lambda slot: slot.attrs.update(
                {"stat_columns": [c for c in REVISION_1_COLUMNS if c != "auc"]}
            ),
            "must store the revision 1 stat_columns",
        ),
        (
            lambda slot: slot.attrs.update({"skipped_groups": {}}),
            "must not record skipped_groups",
        ),
        # The policy of revision 2 needs its columns.
        (
            lambda slot: slot.attrs.update({"p_value_policy": MARKER_P_VALUE_POLICY}),
            "complete named stat_columns",
        ),
        (
            lambda slot: _replace_stats(slot["b"], np.zeros((6, 10))),
            "do not match stat_columns",
        ),
    ],
    ids=[
        "no-feature-index",
        "no-stats",
        "no-fold-change-policy",
        "other-method",
        "neg-log10-column",
        "no-auc-column",
        "skipped-groups",
        "current-policy",
        "ten-columns",
    ],
)
def test_corrupted_revision_1_marker_tables_still_raise(
    revision_1_store, tmp_path, edit, message
) -> None:
    import cytearc.plotting as splt

    store, _, _, marker = revision_1_store
    copy = _copy_store(store, tmp_path)
    edit(copy.zw[artifact_path(marker)])
    with pytest.raises(ValueError, match=message):
        copy.markers.load(marker)
    with pytest.raises(ValueError, match=message):
        copy.markers.export_csv(marker, str(tmp_path / "markers.csv"))
    with pytest.raises(ValueError, match=message):
        splt.marker_heatmap(copy, marker=marker, topn=2, show=False)


@pytest.mark.parametrize("sizes", [(3, 5), (20, 20)], ids=["exact", "asymptotic"])
def test_normalized_batches_get_exact_p_values_only_for_small_designs(sizes) -> None:
    rng = np.random.default_rng(41)
    n_cells = sum(sizes)
    values = rng.integers(0, 4, (n_cells, 3)).astype(np.float64)
    groups = np.repeat(np.array(["a", "b"]), sizes)

    class Assay:
        normMethod = None
        sf = None

        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame(values[:, :2], columns=[0, 1])
            yield pd.DataFrame(values[:, 2:], columns=[2])

    result = find_markers_by_rank(
        Assay(),
        groups=groups,
        cell_idx=np.arange(n_cells),
        feat_idx=np.arange(3),
        nthreads=1,
    )
    names = np.array(["g0", "g1", "g2"])
    table = result.table("a", names).set_index("feature_index")
    for feature in range(3):
        inside = values[groups == "a", feature]
        outside = values[groups == "b", feature]
        # C(8, 3) = 56 splits are exact, C(40, 20) are not.
        expected = (
            _reference_p_value(inside, outside)
            if sizes == (3, 5)
            else _asymptotic_p_value(inside, outside)
        )
        assert table.loc[feature, "p_value"] == pytest.approx(expected, rel=1e-12)
    assert (result.exact_p_values is None) == (sizes == (20, 20))


@pytest.mark.parametrize("method_name", ["norm_clr", "norm_dummy", "norm_tf_idf"])
def test_dense_adapters_compute_exact_p_values_of_small_designs(
    monkeypatch, method_name
) -> None:
    import cytearc.assay.normalization as normalization
    from cytearc.storage.budget import ResourceBudget
    from tests.test_feature_stream import _counts_t_with_plan

    values = (np.arange(8 * 6, dtype=np.uint32).reshape(8, 6) * 5) % 7
    counts_t = _counts_t_with_plan(values)
    feature_index = np.array([0, 2, 3, 5])
    term_totals = values.sum(axis=1).astype(np.float64)
    document_frequency = (values[:, feature_index] > 0).sum(axis=0)

    class FakeAssay:
        def __init__(self) -> None:
            self.normMethod = getattr(normalization, method_name)
            self.sf = 1000.0
            self.name = "RNA"
            self.resources = ResourceBudget(8 * 1024 * 1024, 2)
            self.rawDataT = counts_t

        def _fit_tf_idf(self, cell_idx, feat_idx, **_kwargs):
            return None, (term_totals, len(cell_idx), document_frequency)

        def _normalization_flags(self):
            return frozenset()

    monkeypatch.setattr(marker_search_module, "ATACassay", FakeAssay)
    groups = np.array(["a", "b", "b", "a", "b", "a", "b", "b"])
    result = find_markers_by_rank(
        FakeAssay(),
        groups=groups,
        cell_idx=np.arange(8),
        feat_idx=feature_index,
        nthreads=1,
    )

    raw = values[:, feature_index]
    if method_name == "norm_clr":
        reference = normalization.clr_values(raw)
    elif method_name == "norm_tf_idf":
        reference = normalization.tfidf_values(
            raw,
            term_totals,
            normalization.inverse_document_frequency(8, document_frequency),
        )
    else:
        reference = raw
    # Three and five cells split in C(8, 3) = 56 ways.
    assert result.exact_p_values is not None
    for column in range(len(feature_index)):
        for position, group in enumerate(("a", "b")):
            expected = _reference_p_value(
                reference[groups == group, column], reference[groups != group, column]
            )
            assert result.exact_p_values[column, position] == pytest.approx(
                expected, rel=1e-12
            )
            assert (
                result.stored_statistics(group)[column, 6]
                == (result.exact_p_values[column, position])
            )
