import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cytearc import DataStore
from cytearc.assay import norm_lib_size_log
from cytearc.features.aggregation import _accumulate_group_counts, aggregate_rna_groups
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.writers import SparseToZarr

from .doublet_fixtures import write_doublet_target_zarr
from .test_feature_stream import _counts_t_with_plan


def _bulk_store(tmp_path, counts, *, workers=1):
    path = str(tmp_path / "counts.zarr")
    ids = np.array([f"g{i}" for i in range(counts.shape[1])])
    write_doublet_target_zarr(
        path,
        "RNA",
        csr_matrix(counts),
        ids,
        ids,
        dtype=str(counts.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=128),
    )
    store = DataStore(
        path, default_assay="RNA", min_features_per_cell=0, nthreads=workers
    )
    store.cells.insert("all_cells", np.ones(len(counts), dtype=bool))
    return store, store.snapshot_cell_selection("all_cells")


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("return_fraction", [False, True])
def test_bulk_kernel_matches_grouped_counts(normalize, return_fraction):
    raw = np.array([[0, 90, 3, 7], [8, 80, 0, 2], [1, 70, 4, 0]], dtype=np.uint16)
    selected = np.array([0, 2, 3])
    codes = np.array([1, 0, 1])
    scalars = np.array([9.0, 7.0, 9.0]) if normalize else None
    expected = np.zeros((3, 3))
    expected_fractions = np.zeros((3, 3))
    for group in range(3):
        counts = raw[:, selected[codes == group]].astype(float)
        expected_fractions[:, group] = (counts > 0).sum(axis=1)
        if normalize:
            counts *= 100 / scalars[codes == group]
        expected[:, group] = counts.sum(axis=1)
    for kernel in (_accumulate_group_counts.py_func, _accumulate_group_counts):
        values = np.zeros((3, 3)).T
        fractions = np.zeros((3, 3)).T if return_fraction else None
        kernel(raw, selected, codes, scalars, 100.0, values, fractions)
        np.testing.assert_allclose(values, expected)
        if return_fraction:
            np.testing.assert_array_equal(fractions, expected_fractions)


@pytest.mark.parametrize(
    ("codes", "scalars", "message"),
    [
        ([0], None, "group codes"),
        ([-1, 0], None, "group codes"),
        ([0, 2], None, "group codes"),
        ([0, 1], [1.0], "normalization scalars must align"),
        # Divisors map a zero total to 1 before they reach the aggregation.
        ([0, 1], [1.0, 0.0], "scalars must be finite and positive"),
        ([0, 1], [np.nan, 1.0], "scalars must be finite and positive"),
    ],
)
def test_bulk_rejects_misaligned_groups_and_scalars(codes, scalars, message):
    counts_t = _counts_t_with_plan(np.ones((3, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match=message):
        aggregate_rna_groups(
            counts_t,
            np.arange(2),
            np.array(codes),
            2,
            scalars=None if scalars is None else np.array(scalars),
            size_factor=100,
            return_fraction=False,
            resources=ResourceBudget(1_000_000, 1),
        )


def _reopen(path, *, workers=1):
    """Open a written bulk store afresh, so no test sees another's assay state."""
    return DataStore(
        path, default_assay="RNA", min_features_per_cell=0, nthreads=workers
    )


@pytest.fixture(scope="module")
def grouped_counts_store(tmp_path_factory):
    """Write the grouped counts that every aggregation and worker count reads.

    Bulk profiles only read the store, so it is written once.
    """
    counts = np.random.default_rng(71).integers(0, 200, (7, 24), dtype=np.uint16)
    counts[:, -1] = 0
    store, _ = _bulk_store(tmp_path_factory.mktemp("grouped_counts"), counts)
    store.cells.insert("selected", np.arange(7) != 5)
    store.snapshot_cell_selection("selected")
    store.cells.insert("group", np.array(["a", "a", "b", "skip", "a", "b", "b"]))
    store.cells.insert("secondary", np.array(["x", "y", "x", "x", "x", "y", "y"]))
    return str(store.zarr_loc), counts


@pytest.mark.parametrize("keep_empty", [False, True])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize("replicates", [1, 3])
@pytest.mark.parametrize("workers", [1, 4])
def test_bulk_counts_t_matches_selected_groups_and_empty_replicates(
    grouped_counts_store,
    aggregation,
    replicates,
    workers,
    keep_empty,
):
    path, counts = grouped_counts_store
    store = _reopen(path, workers=workers)
    cells = store.snapshot_cell_selection("selected")
    actual, fractions = store.compare.compute_bulk(
        "group",
        cell_selection=cells,
        secondary_groups="secondary",
        null_vals=["skip"],
        aggr_type=aggregation,
        pseudo_reps=replicates,
        return_fraction=True,
        feature_label="id",
        random_seed=61,
        remove_empty_columns=not keep_empty,
    )
    expected_columns = []
    for name, rows in {"a_x": [0, 4], "a_y": [1], "b_x": [2], "b_y": [6]}.items():
        shuffled = np.random.RandomState(61).choice(rows, len(rows), replace=False)
        for i, part in enumerate(np.array_split(shuffled, replicates)):
            column = name if replicates == 1 else f"{name}_Rep{i + 1}"
            if len(part) == 0:
                # A pseudo-replicate without cells is left out, or kept as NaN.
                if keep_empty:
                    expected_columns.append(column)
                    assert actual[column].isna().all()
                    assert fractions[column].isna().all()
                continue
            expected_columns.append(column)
            raw = counts[sorted(part)]
            expected_fraction = (raw > 0).mean(axis=0)
            expected = (
                raw.sum(axis=0)
                if aggregation == "sum"
                else (raw.astype(float) * store.RNA.sf / raw.sum(axis=1)[:, None]).mean(
                    axis=0
                )
            )
            np.testing.assert_allclose(actual[column], expected[:-1])
            np.testing.assert_array_equal(fractions[column], expected_fraction[:-1])
    assert list(actual.columns) == list(fractions.columns) == expected_columns
    assert list(actual.index) == [f"g{i}" for i in range(23)]


@pytest.mark.parametrize("workers", [1, 4])
def test_bulk_sum_keeps_large_integers_and_ignores_custom_normalizer(tmp_path, workers):
    counts = np.array([[2**53 + 3, 1], [2**53 + 7, 2]], dtype=np.uint64)
    store, cells = _bulk_store(tmp_path, counts, workers=workers)
    store.cells.insert("group", np.array(["a", "a"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw bulk sums must not call the normalizer")

    store.RNA.normMethod = forbidden
    actual = store.compare.compute_bulk("group", cell_selection=cells, aggr_type="sum")
    np.testing.assert_array_equal(actual["a"], counts.sum(axis=0))
    assert actual["a"].dtype == np.uint64


@pytest.mark.parametrize("workers", [1, 4])
def test_bulk_mean_uses_stored_totals_and_averages_zero_total_cells_as_zero(
    tmp_path, workers
):
    counts = np.array([[100, 10], [200, 20], [0, 0]], dtype=np.uint16)
    store, cells = _bulk_store(tmp_path, counts, workers=workers)
    store.cells.insert("group", np.array(["a", "b", "b"]))
    store.zw["cellData"].create_array(
        "RNA_nCounts", data=np.array([220, 220, 0]), overwrite=True
    )
    normalized = store.RNA.normed(
        cell_idx=np.arange(3), feat_idx=np.arange(2)
    ).compute()
    actual, fractions = store.compare.compute_bulk(
        "group",
        cell_selection=cells,
        return_fraction=True,
        remove_empty_features=False,
    )
    np.testing.assert_allclose(
        actual["a"], counts[0].astype(float) * store.RNA.sf / 220
    )
    # The cell without counts normalizes to zero and still counts toward the
    # mean of its group, as in the average of ``normed``.
    np.testing.assert_allclose(actual["b"], normalized[1:].mean(axis=0))
    np.testing.assert_allclose(
        actual["b"], counts[1].astype(float) * store.RNA.sf / 220 / 2
    )
    np.testing.assert_array_equal(fractions["b"], 0.5)


@pytest.fixture(scope="module")
def two_cell_store(tmp_path_factory):
    """Write two cells of one group once; each test sets its normalizer."""
    counts = np.array([[100, 10], [200, 20]], dtype=np.uint32)
    store, _ = _bulk_store(tmp_path_factory.mktemp("two_cells"), counts)
    store.cells.insert("group", np.array(["a", "a"]))
    return str(store.zarr_loc), counts


@pytest.mark.parametrize("normalizer", [norm_lib_size_log, lambda assay, raw: raw + 7])
@pytest.mark.parametrize("replicates", [1, 3])
def test_bulk_mean_retains_other_normalizers(two_cell_store, normalizer, replicates):
    path, counts = two_cell_store
    store = _reopen(path)
    cells = store.snapshot_cell_selection("all_cells")
    store.RNA.normMethod = normalizer
    normalized = store.RNA.normed(
        cell_idx=np.arange(2), feat_idx=np.arange(2)
    ).compute()
    actual, fractions = store.compare.compute_bulk(
        "group",
        cell_selection=cells,
        pseudo_reps=replicates,
        return_fraction=True,
        random_seed=61,
    )
    shuffled = np.random.RandomState(61).choice(2, 2, replace=False)
    columns = []
    for i, rows in enumerate(np.array_split(shuffled, replicates)):
        column = "a" if replicates == 1 else f"a_Rep{i + 1}"
        if not len(rows):
            # Two cells fill two of three pseudo-replicates; the third is left out.
            continue
        columns.append(column)
        np.testing.assert_allclose(actual[column], normalized[rows].mean(axis=0))
        np.testing.assert_array_equal(
            fractions[column], (counts[rows] > 0).mean(axis=0)
        )
    assert list(actual.columns) == list(fractions.columns) == columns


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_excludes_masked_labels_like_null_values(tmp_path, aggregation):
    from tests.storage_helpers import insert_nullable_cell_column

    counts = np.random.default_rng(13).integers(1, 50, (6, 8), dtype=np.uint16)
    store, cells = _bulk_store(tmp_path, counts)
    # Placeholders equal real labels: 0 is a group and "" would be a sub-group.
    insert_nullable_cell_column(
        store,
        "group",
        np.array([0, 1, 0, 0, 1, 0]),
        np.array([False, False, True, False, False, True]),
    )
    insert_nullable_cell_column(
        store,
        "secondary",
        np.array(["x", "x", "y", "", "y", "x"]),
        np.array([False, False, False, True, False, False]),
    )
    store.cells.insert("group_nulls", np.array(["0", "1", "-", "0", "1", "-"]))
    store.cells.insert("secondary_nulls", np.array(["x", "x", "y", "-", "y", "x"]))
    options = {
        "cell_selection": cells,
        "aggr_type": aggregation,
        "feature_label": "id",
        "remove_empty_features": False,
    }

    masked = store.compare.compute_bulk(
        "group", secondary_groups="secondary", **options
    )
    nulled = store.compare.compute_bulk(
        "group_nulls",
        secondary_groups="secondary_nulls",
        null_vals=["-"],
        secondary_null_vals=["-"],
        **options,
    )

    # Group 0 has no cell of sub-group y once masked labels are left out, so
    # that combination has no column rather than a column of zeros.
    assert list(masked.columns) == ["0_x", "1_x", "1_y"]
    pd.testing.assert_frame_equal(masked, nulled)
    if aggregation == "sum":
        np.testing.assert_array_equal(masked["0_x"], counts[0])
    kept = store.compare.compute_bulk(
        "group", secondary_groups="secondary", remove_empty_columns=False, **options
    )
    assert list(kept.columns) == ["0_x", "0_y", "1_x", "1_y"]
    assert kept["0_y"].isna().all()
    pd.testing.assert_frame_equal(kept[masked.columns], masked)


def test_bulk_leaves_nan_none_and_blank_labels_out_of_every_group(tmp_path):
    counts = np.arange(1, 25, dtype=np.uint16).reshape(6, 4)
    store, cells = _bulk_store(tmp_path, counts)
    store.cells.insert("float_group", np.array([1.0, 1.0, np.nan, 2.0, np.nan, 2.0]))
    store.cells.insert(
        "text_group",
        np.array(["a", "", "a", None, "b", " "], dtype=object),
    )
    options = {
        "cell_selection": cells,
        "aggr_type": "sum",
        "remove_empty_features": False,
    }

    by_float = store.compare.compute_bulk("float_group", **options)
    by_text = store.compare.compute_bulk("text_group", **options)

    assert list(by_float.columns) == ["1.0", "2.0"]
    np.testing.assert_array_equal(by_float["1.0"], counts[[0, 1]].sum(axis=0))
    np.testing.assert_array_equal(by_float["2.0"], counts[[3, 5]].sum(axis=0))
    assert list(by_text.columns) == ["a", "b"]
    np.testing.assert_array_equal(by_text["a"], counts[[0, 2]].sum(axis=0))
    # Null values that name every label leave no group to read.
    assert store.compare.compute_bulk(
        "text_group", null_vals=["a", "b"], **options
    ).empty


def test_bulk_rejects_colliding_column_names(tmp_path):
    store, cells = _bulk_store(tmp_path, np.ones((4, 3), dtype=np.uint16))
    store.cells.insert("group", np.array(["a_b", "a_b", "a", "a"]))
    store.cells.insert("secondary", np.array(["c", "c", "b_c", "b_c"]))

    with pytest.raises(ValueError, match="'a_b_c' is produced by more than one"):
        store.compare.compute_bulk(
            "group", cell_selection=cells, secondary_groups="secondary"
        )


def test_bulk_by_a_column_reads_the_live_cells_without_a_snapshot(tmp_path):
    counts = np.arange(1, 25, dtype=np.uint16).reshape(6, 4)
    store, cells = _bulk_store(tmp_path, counts)
    store.cells.insert("group", np.array(["a", "a", "b", "b", "a", "b"]))
    store.cells.insert("sub", np.array(["x", "y", "x", "y", "x", "y"]))
    store.cells.insert("part", np.array([True, True, True, True, False, False]))
    store.cells.insert("gap", np.array([True, False, True, True, True, True]))
    labels = store.snapshot_cluster_labels("sub", cell_selection=cells)
    part = store.snapshot_cluster_labels(
        "sub", cell_selection=store.snapshot_cell_selection("part")
    )
    gap = store.snapshot_cluster_labels(
        "sub", cell_selection=store.snapshot_cell_selection("gap")
    )
    store.cells.insert("I", np.arange(6) < 5, overwrite=True, force=True)
    options = {"aggr_type": "sum", "remove_empty_features": False}

    def files():
        return {
            str(file.relative_to(tmp_path)): file.read_bytes()
            for file in sorted(tmp_path.rglob("*"))
            if file.is_file()
        }

    before = files()
    bulk = store.compare.compute_bulk("group", secondary_groups=labels, **options)

    assert files() == before
    assert list(bulk.columns) == ["a_x", "a_y", "b_x", "b_y"]
    np.testing.assert_array_equal(bulk["a_x"], counts[[0, 4]].sum(axis=0))
    np.testing.assert_array_equal(bulk["b_y"], counts[3])
    # A sub-grouping artifact must hold every live cell, past its last cell
    # or between two of its cells.
    for partial in (part, gap):
        with pytest.raises(ValueError, match="subset of the artifact cell selection"):
            store.compare.compute_bulk("group", secondary_groups=partial, **options)


def test_bulk_mean_fits_non_rna_normalization_once(tmp_path):
    counts = np.random.default_rng(5).integers(0, 30, (12, 5)).astype(np.uint32)
    counts[:, 0] += 1
    path = str(tmp_path / "adt.zarr")
    SparseToZarr(
        csr_matrix(counts),
        zarr_loc=path,
        cell_ids=[f"c{i}" for i in range(12)],
        feature_ids=[f"p{i}" for i in range(5)],
        assay_name="ADT",
        nthreads=1,
    ).dump(batch_size=4)
    store = DataStore(path, default_assay="ADT", min_features_per_cell=0)
    store.cells.insert("all_cells", np.ones(12, dtype=bool))
    cells = store.snapshot_cell_selection("all_cells")
    store.cells.insert("group", np.repeat(["a", "b", "c"], 4))
    options = {"cell_selection": cells, "remove_empty_features": False}

    every_group = store.compare.compute_bulk("group", **options)
    without_b = store.compare.compute_bulk("group", null_vals=["b"], **options)

    # CLR geometric means come from every selected cell, not from each group.
    normalized = np.log1p(counts / np.exp(np.log1p(counts).mean(axis=0)))
    for group, rows in (("a", slice(0, 4)), ("b", slice(4, 8)), ("c", slice(8, 12))):
        np.testing.assert_allclose(every_group[group], normalized[rows].mean(axis=0))
    assert list(without_b.columns) == ["a", "c"]
    np.testing.assert_allclose(without_b["a"], every_group["a"])


def test_bulk_rejects_unknown_aggregation(tmp_path):
    from cytearc.features.aggregation import aggregate_bulk_profiles, bulk_read_cells

    store, cells = _bulk_store(tmp_path, np.ones((2, 3), dtype=np.uint16))
    store.cells.insert("group", np.array(["a", "a"]))
    with pytest.raises(ValueError, match="aggr_type"):
        store.compare.compute_bulk("group", cell_selection=cells, aggr_type="median")

    # The aggregation functions refuse it too, without reading a cell.
    cell_idx = np.arange(2, dtype=np.int64)
    column_rows = {"a": cell_idx}
    read = bulk_read_cells(store.RNA, "median", cell_idx, column_rows)
    assert read.dtype == np.int64
    np.testing.assert_array_equal(read, [])
    with pytest.raises(
        ValueError, match=r"^ERROR: `aggr_type` can only be either 'sum' or 'mean'$"
    ):
        aggregate_bulk_profiles(
            store.RNA,
            column_rows,
            cell_idx,
            aggr_type="median",
            return_fraction=True,
            cells=store.cells,
            resources=store.resources,
            nthreads=1,
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"feature_label": "bogus"}, "feature_label must be 'index', 'id', or 'name'"),
        ({"aggr_type": "median"}, "aggr_type"),
        ({"pseudo_reps": 0}, "pseudo_reps must be at least 1"),
    ],
)
def test_bulk_checks_its_options_before_reading_any_cell(
    tmp_path, monkeypatch, options, message
):
    import cytearc.features.aggregation as aggregation

    store, cells = _bulk_store(tmp_path, np.ones((2, 3), dtype=np.uint16))
    store.cells.insert("group", np.array(["a", "a"]))

    def refuse(*_args, **_kwargs):
        raise AssertionError("bulk columns were read")

    monkeypatch.setattr(aggregation, "bulk_column_rows", refuse)
    # An unknown feature label fell back to positions.
    with pytest.raises(ValueError, match=message):
        store.compare.compute_bulk("group", cell_selection=cells, **options)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_leaves_out_combinations_without_cells_and_counts_cells(
    tmp_path, aggregation
):
    counts = np.random.default_rng(29).integers(0, 40, (8, 5), dtype=np.uint16)
    counts[:, 0] += 1
    store, cells = _bulk_store(tmp_path, counts)
    # Donor d2 has no cell of cluster B.
    store.cells.insert("cluster", np.array(["A", "A", "B", "B", "A", "B", "A", "A"]))
    store.cells.insert(
        "donor", np.array(["d1", "d2", "d1", "d1", "d2", "d1", "d2", "d1"])
    )
    options = {
        "cell_selection": cells,
        "secondary_groups": "donor",
        "aggr_type": aggregation,
        "remove_empty_features": False,
    }

    values = store.compare.compute_bulk("cluster", **options)
    with_counts, n_cells = store.compare.compute_bulk(
        "cluster", return_n_cells=True, **options
    )
    full, fractions, full_counts = store.compare.compute_bulk(
        "cluster", return_fraction=True, return_n_cells=True, **options
    )
    kept, kept_fractions, kept_counts = store.compare.compute_bulk(
        "cluster",
        return_fraction=True,
        return_n_cells=True,
        remove_empty_columns=False,
        **options,
    )

    rows = {"A_d1": [0, 7], "A_d2": [1, 4, 6], "B_d1": [2, 3, 5]}
    assert values.columns.tolist() == list(rows)
    for column, members in rows.items():
        raw = counts[members]
        expected = (
            raw.sum(axis=0)
            if aggregation == "sum"
            else (raw * store.RNA.sf / raw.sum(axis=1)[:, None]).mean(axis=0)
        )
        np.testing.assert_allclose(values[column], expected, rtol=1e-12)
        np.testing.assert_array_equal(fractions[column], (raw > 0).mean(axis=0))
    for frame in (with_counts, full):
        pd.testing.assert_frame_equal(frame, values)
    expected_counts = pd.Series(
        {column: len(members) for column, members in rows.items()},
        dtype=np.int64,
        name="n_cells",
    )
    for counted in (n_cells, full_counts):
        pd.testing.assert_series_equal(counted, expected_counts)

    # Kept, the combination holds NaN, never a profile of zeros.
    assert kept.columns.tolist() == [*rows, "B_d2"]
    assert kept["B_d2"].isna().all() and kept_fractions["B_d2"].isna().all()
    pd.testing.assert_frame_equal(kept[list(rows)], values)
    pd.testing.assert_frame_equal(kept_fractions[list(rows)], fractions)
    assert kept_counts.to_dict() == {**expected_counts.to_dict(), "B_d2": 0}


def test_bulk_sum_preserves_non_rna_assays(datastore_ephemeral):
    store = datastore_ephemeral
    selected = np.arange(store.cells.N) < 2
    store.cells.insert("bulk_cells", selected)
    store.cells.insert("bulk_group", np.repeat("a", store.cells.N))
    cells = store.snapshot_cell_selection("bulk_cells")
    counts = store.assay2.rawData[np.flatnonzero(selected)].compute()
    values, fractions = store.compare.compute_bulk(
        "bulk_group",
        cell_selection=cells,
        from_assay="assay2",
        aggr_type="sum",
        return_fraction=True,
        remove_empty_features=False,
        feature_label="id",
    )
    np.testing.assert_array_equal(values["a"], counts.sum(axis=0))
    np.testing.assert_array_equal(fractions["a"], (counts > 0).mean(axis=0))


def test_bulk_output_is_admitted_before_streaming(tmp_path):
    counts = np.ones((4, 24), dtype=np.uint16)
    store, _ = _bulk_store(tmp_path, counts)
    with pytest.raises(MemoryError, match="operation limit"):
        aggregate_rna_groups(
            store.RNA.rawDataT,
            np.arange(4),
            np.arange(4),
            1_000_000,
            scalars=None,
            size_factor=1_000,
            return_fraction=True,
            resources=ResourceBudget(1_000_000, 1),
        )


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_aggregates_floating_point_counts_in_float64(aggregation):
    counts = np.array(
        [[0.25, 1.5, 2.0], [200.0, 400.0, 1.0], [1.0, 0.0, 0.25], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    selected = np.array([0, 1, 3])
    codes = np.array([0, 0, 2])
    totals = counts[selected].sum(axis=1, dtype=np.float64)
    actual, fractions = aggregate_rna_groups(
        _counts_t_with_plan(counts),
        selected,
        codes,
        4,
        scalars=totals if aggregation == "mean" else None,
        size_factor=1000,
        return_fraction=True,
        resources=ResourceBudget(16 * 1024**2, 4),
    )
    assert actual.dtype == np.float64
    assert fractions is not None
    for code in range(4):
        members = codes == code
        raw = counts[selected[members]]
        if aggregation == "mean":
            values = raw.astype(np.float64) * 1000 / totals[members, None]
            expected = values.sum(axis=0) / max(1, members.sum())
        else:
            expected = raw.sum(axis=0)
        np.testing.assert_allclose(actual[:, code], expected)
        np.testing.assert_array_equal(
            fractions[:, code], (raw > 0).sum(axis=0) / max(1, members.sum())
        )


@pytest.mark.parametrize("codes", [[0, 1], [0, 1, 2], [-2, 0, 1]])
def test_normalized_aggregation_rejects_codes_off_its_rows(codes):
    from cytearc.features.aggregation import aggregate_normalized_groups
    from cytearc.matrix import ChunkedArray

    with pytest.raises(ValueError, match="align with normalized rows"):
        aggregate_normalized_groups(
            ChunkedArray.from_numpy(np.ones((3, 2))), np.array(codes), 2, nthreads=1
        )


def test_normalized_aggregation_reserves_its_group_sums(tmp_path):
    import tracemalloc

    import zarr
    from zarr.storage import LocalStore

    from cytearc.features.aggregation import aggregate_normalized_groups
    from cytearc.matrix import ChunkedArray
    from cytearc.storage.execution import execution_report_scope

    rng = np.random.default_rng(0)
    values = rng.poisson(0.3, size=(8_000, 64)).astype(np.uint16)
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    counts = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(1_000, 64),
        shards=(1_000, 64),
        dtype=np.uint16,
        fill_value=0,
    )
    counts[:] = values
    # The sums of 6,000 groups outweigh the two row blocks read at a time.
    codes = rng.integers(-1, 6_000, size=len(values))
    budget = 6_000_000

    def aggregate() -> np.ndarray:
        normalized = ChunkedArray(
            counts, nthreads=2, resources=ResourceBudget(budget, 2)
        )
        return aggregate_normalized_groups(normalized * 0.5, codes, 6_000, nthreads=2)

    aggregate()
    with execution_report_scope() as reports:
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            means = aggregate()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    (plan,) = [report.plan for report in reports]
    assert plan.residentBytes >= 6_000 * 64 * 8
    assert peak - base <= plan.reservedBytes <= budget
    grouped = codes >= 0
    expected = np.zeros((6_000, 64))
    np.add.at(expected, codes[grouped], values[grouped] * 0.5)
    expected /= np.maximum(np.bincount(codes[grouped], minlength=6_000), 1)[:, None]
    np.testing.assert_allclose(means, expected.T)


def test_bulk_sums_of_float_counts_are_exact_above_float32_precision():
    # Beyond 2**24 float32 holds only even integers, so a sum returned in the
    # storage dtype rounded an exact odd total.
    counts = np.array([[2.0**24], [1.0], [1.0], [1.0]], dtype=np.float32)
    selected = np.arange(4)

    actual, _ = aggregate_rna_groups(
        _counts_t_with_plan(counts),
        selected,
        np.zeros(4, dtype=np.int64),
        1,
        scalars=None,
        size_factor=1000,
        return_fraction=False,
        resources=ResourceBudget(16 * 1024**2, 1),
    )

    assert actual.dtype == np.float64
    assert actual[0, 0] == 2**24 + 3


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_aggregation_rejects_results_that_are_not_finite(aggregation):
    # Feature 1 of group 1 overflows float64: its sum, or its normalized values.
    counts = np.array([[1.0, 2.0, 3.0], [4.0, 1.5e308, 1.5e308]], dtype=np.float64)

    with pytest.raises(ValueError, match="feature 1 in group 'b' is not finite"):
        aggregate_rna_groups(
            _counts_t_with_plan(counts.T),
            np.arange(3),
            np.array([0, 1, 1]),
            2,
            scalars=np.ones(3) if aggregation == "mean" else None,
            size_factor=1000,
            return_fraction=False,
            resources=ResourceBudget(16 * 1024**2, 1),
            group_names=["a", "b"],
        )


def test_normalized_aggregation_names_the_group_of_a_non_finite_mean():
    from cytearc.features.aggregation import aggregate_normalized_groups
    from cytearc.matrix import ChunkedArray

    values = np.ones((4, 3))
    values[3, 2] = np.nan
    with pytest.raises(ValueError, match="feature 2 in group 1 is not finite"):
        aggregate_normalized_groups(
            ChunkedArray.from_numpy(values), np.array([0, 0, -1, 1]), 2, nthreads=1
        )
    # Rows outside every group are fitted but not averaged.
    means = aggregate_normalized_groups(
        ChunkedArray.from_numpy(values), np.array([0, 0, 1, -1]), 2, nthreads=1
    )
    np.testing.assert_array_equal(means, np.ones((3, 2)))


def _non_rna_store(tmp_path, counts, *, assay="ADT", workers=1):
    path = str(tmp_path / f"{assay.lower()}.zarr")
    ids = np.array([f"p{i}" for i in range(counts.shape[1])])
    write_doublet_target_zarr(
        path,
        assay,
        csr_matrix(counts),
        ids,
        ids,
        dtype=str(counts.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=128),
    )
    store = DataStore(
        path, default_assay=assay, min_features_per_cell=0, nthreads=workers
    )
    store.cells.insert("all_cells", np.ones(len(counts), dtype=bool))
    return store, store.snapshot_cell_selection("all_cells")


def _column_profiles(raw, column_rows, *, aggr_type, return_fraction):
    """Sum and count each column's raw rows one column at a time.

    This is how compare.compute_bulk aggregated assays without countsT
    before one pass replaced it.
    """
    vals, fracs = {}, {}
    sum_dtype = np.float64 if raw.dtype.kind == "f" else None
    for name, idx in column_rows.items():
        if aggr_type == "sum":
            vals[name] = np.asarray(raw[idx].sum(axis=0, dtype=sum_dtype).compute())
        if return_fraction:
            fracs[name] = (raw[idx] > 0).mean(axis=0).compute()
    return vals, fracs


@pytest.fixture(scope="module")
def grouped_adt_stores(tmp_path_factory):
    """Write the grouped ADT counts of a dtype once, for both aggregations.

    Bulk profiles only read the store, so each dtype's store is shared, with
    its pseudo-replicate columns and their per-column sums and fractions.
    """
    from cytearc.features.aggregation import bulk_column_rows

    stores = {}

    def open_store(dtype):
        if dtype not in stores:
            rng = np.random.default_rng(17)
            counts = rng.poisson(0.8, size=(60, 9)).astype(dtype)
            counts[:, 0] += 1
            store, cells = _non_rna_store(tmp_path_factory.mktemp("adt"), counts)
            groups = rng.choice(["a", "b", "c", "d"], size=60)
            donors = rng.choice(["x", "y"], size=60)
            store.cells.insert("group", groups)
            store.cells.insert("donor", donors)
            column_rows = bulk_column_rows(
                groups,
                sorted(set(groups)),
                np.ones(60, dtype=bool),
                np.arange(60, dtype=np.int64),
                secondary=(donors, ["x", "y"]),
                null_vals=None,
                secondary_null_vals=None,
                pseudo_reps=2,
                random_seed=4466,
            )
            filled = {name: rows for name, rows in column_rows.items() if len(rows)}
            expected = _column_profiles(
                store.ADT.rawData, filled, aggr_type="sum", return_fraction=True
            )
            stores[dtype] = (store, cells, filled, expected)
        return stores[dtype]

    return open_store


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint32, np.int32, np.float32, np.float64], ids=str
)
def test_non_rna_bulk_matches_the_per_column_aggregation(
    grouped_adt_stores, dtype, aggregation
):
    from cytearc.features.aggregation import aggregate_bulk_profiles

    store, cells, filled, (expected_vals, expected_fracs) = grouped_adt_stores(dtype)
    options = {
        "cell_selection": cells,
        "secondary_groups": "donor",
        "aggr_type": aggregation,
        "pseudo_reps": 2,
        "remove_empty_features": False,
        "remove_empty_columns": False,
    }

    values, fractions = store.compare.compute_bulk(
        "group", return_fraction=True, **options
    )

    actual_vals, actual_fracs = aggregate_bulk_profiles(
        store.ADT,
        filled,
        np.arange(60, dtype=np.int64),
        aggr_type=aggregation,
        return_fraction=True,
        cells=store.cells,
        resources=store.resources,
        nthreads=1,
    )
    assert list(actual_fracs) == list(expected_fracs) == list(filled)
    for name in filled:
        assert actual_fracs[name].dtype == expected_fracs[name].dtype
        assert actual_fracs[name].tobytes() == expected_fracs[name].tobytes()
        if aggregation == "sum":
            assert actual_vals[name].dtype == expected_vals[name].dtype
            # Integer-valued counts sum exactly in any order.
            assert actual_vals[name].tobytes() == expected_vals[name].tobytes()
            np.testing.assert_array_equal(values[name], expected_vals[name])
        np.testing.assert_array_equal(fractions[name], expected_fracs[name])
    if aggregation == "sum":
        assert list(actual_vals) == list(filled)


def test_non_rna_bulk_reads_each_raw_row_once(tmp_path, monkeypatch):
    import cytearc.matrix.chunked as chunked

    counts = np.random.default_rng(3).poisson(1.0, size=(40, 6)).astype(np.uint16)
    store, cells = _non_rna_store(tmp_path, counts)
    store.cells.insert("group", np.repeat(["a", "b", "c", "d", "e"], 8))
    read = chunked.read_selection
    rows_read: list[int] = []

    def counted(array, rows, columns):
        values = read(array, rows, columns)
        rows_read.append(len(values))
        return values

    monkeypatch.setattr(chunked, "read_selection", counted)

    values, fractions = store.compare.compute_bulk(
        "group",
        cell_selection=cells,
        aggr_type="sum",
        return_fraction=True,
        remove_empty_features=False,
    )

    # One pass sums every group and counts its positive values; each column
    # read its rows twice before.
    assert sum(rows_read) == 40
    for index, name in enumerate("abcde"):
        rows = counts[8 * index : 8 * (index + 1)]
        np.testing.assert_array_equal(values[name], rows.sum(axis=0))
        np.testing.assert_array_equal(fractions[name], (rows > 0).mean(axis=0))


def test_raw_group_sums_add_floating_rows_in_order():
    from cytearc.features.aggregation import aggregate_raw_groups
    from cytearc.matrix import ChunkedArray

    rng = np.random.default_rng(9)
    values = rng.normal(size=(50, 300)) * 10.0 ** rng.integers(-3, 4, size=(50, 300))
    codes = rng.integers(0, 3, size=50)

    sums, positives = aggregate_raw_groups(
        ChunkedArray.from_numpy(values, block_size=7),
        codes,
        3,
        sums=True,
        positives=True,
        nthreads=2,
    )

    for group in range(3):
        expected = np.zeros(300)
        for row in values[codes == group]:
            expected = expected + row
        assert sums[group].tobytes() == expected.tobytes()
        np.testing.assert_array_equal(
            positives[group], (values[codes == group] > 0).sum(axis=0)
        )


@pytest.mark.parametrize(
    "dtype", [np.float16, np.dtype(">u4"), np.dtype("<i8"), np.bool_], ids=str
)
def test_raw_group_sums_read_every_count_dtype(dtype):
    from cytearc.features.aggregation import aggregate_raw_groups
    from cytearc.matrix import ChunkedArray

    values = np.array([[1, 0, 3], [2, 5, 0], [0, 1, 1], [4, 0, 2]]).astype(dtype)
    codes = np.array([1, 0, 1, 0])

    sums, positives = aggregate_raw_groups(
        ChunkedArray.from_numpy(values), codes, 2, sums=True, positives=True, nthreads=1
    )

    for group in range(2):
        rows = values[codes == group]
        expected = rows.sum(
            axis=0, dtype=np.float64 if rows.dtype.kind == "f" else None
        )
        assert sums[group].dtype == expected.dtype
        np.testing.assert_array_equal(sums[group], expected)
        np.testing.assert_array_equal(positives[group], (rows > 0).sum(axis=0))
    only_positives = aggregate_raw_groups(
        ChunkedArray.from_numpy(values),
        codes,
        2,
        sums=False,
        positives=True,
        nthreads=1,
    )
    assert only_positives[0] is None
    np.testing.assert_array_equal(only_positives[1], positives)


def test_raw_group_sums_reject_codes_off_their_rows():
    from cytearc.features.aggregation import aggregate_raw_groups
    from cytearc.matrix import ChunkedArray

    raw = ChunkedArray.from_numpy(np.ones((3, 2), dtype=np.uint16))
    for codes in ([0, 1], [0, 1, 2], [-1, 0, 1]):
        with pytest.raises(ValueError, match="align with raw rows"):
            aggregate_raw_groups(
                raw, np.array(codes), 2, sums=True, positives=False, nthreads=1
            )


def test_accumulate_group_rows_kernel_adds_rows_in_order():
    from cytearc.features.aggregation import _accumulate_group_rows

    block = np.array([[1.5, -2.0, 0.0], [0.25, 3.0, np.nan], [2.0, 0.0, 1.0]])
    codes = np.array([1, 0, 1])
    sums = np.zeros((2, 3))
    positives = np.zeros((2, 3), dtype=np.int64)

    _accumulate_group_rows.py_func(block, codes, sums, positives)
    _accumulate_group_rows.py_func(block, codes, None, None)

    np.testing.assert_array_equal(sums, [[0.25, 3.0, np.nan], [3.5, -2.0, 1.0]])
    np.testing.assert_array_equal(positives, [[1, 1, 0], [2, 0, 1]])
