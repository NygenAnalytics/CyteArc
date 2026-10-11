from dataclasses import replace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.assay.normalization import (
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_lib_size_log,
)
from cytearc.datastore.datastore import DataStore
from cytearc.features.values import (
    normalization_is_cell_local,
    fetch_normalized_feature_matrix,
    iter_normalized_feature_blocks,
    measured_feature_means,
    resolve_feature,
    spread_measured_rows,
)
from cytearc.metadata.membership import membership_attributes
from cytearc.metadata.selection import NormalizationSpec
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.execution import execution_report_scope
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts


@pytest.fixture(scope="module", params=["columns", "rows", "membership"])
def feature_store(request):
    shape = (45, 25) if request.param == "rows" else (173, 401)
    counts = np.random.default_rng(19).integers(0, 40, shape, dtype=np.uint16)
    counts[0] = 0
    measured = np.ones(shape[0], dtype=bool)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cells = np.array([f"c{index}" for index in range(shape[0])])
    create_cell_data(root, None, ids=cells, names=cells)
    if request.param == "membership":
        measured[1::7] = False
        counts[~measured] = 0
        column = root["cellData"].create_array("RNA_I", data=measured)
        column.attrs.update(membership_attributes("RNA"))
    features = np.array([f"gene{index}" for index in range(shape[1])])
    array = create_zarr_count_assay(
        root,
        "RNA",
        None,
        shape[0],
        features,
        features,
        counts.dtype,
        policy=CountMatrixPolicy(unitBytes=2_000, chunkBytes=200),
    )
    array[:] = counts
    finalize_test_counts(array)
    finalize_writer_counts_t(root, "RNA", None)
    store = DataStore(root.store, min_features_per_cell=0, nthreads=1)
    store.cells.insert("everyone", np.ones(shape[0], dtype=bool))
    return store, measured, request.param


@pytest.mark.parametrize(
    ("method", "spec"),
    [
        (norm_lib_size, NormalizationSpec()),
        (norm_lib_size, NormalizationSpec(transform="log1p")),
        (norm_lib_size_log, NormalizationSpec()),
        (norm_dummy, NormalizationSpec()),
        (norm_dummy, NormalizationSpec(transform="log1p")),
        (norm_lib_size, NormalizationSpec(source="raw")),
        (norm_lib_size, NormalizationSpec(source="raw", transform="log1p")),
    ],
)
def test_feature_reads_choose_orientation_without_changing_blocks(
    feature_store, monkeypatch, method, spec
):
    store, measured, geometry = feature_store
    monkeypatch.setattr(store.RNA, "normMethod", method)
    cells = np.arange(store.cells.N)
    physical = np.array([1, 7, 21])
    read_cells = cells[measured]
    native = (
        store.RNA.rawData[:, physical][read_cells, :]
        if spec.source == "raw"
        else store.RNA.normed(
            read_cells, physical, log_transform=spec.transform == "log1p"
        )
    )
    row_size = native.chunksize[0]
    expected_blocks = list(native.stream_blocks(nthreads=1))
    if spec.source == "raw" and spec.transform == "log1p":
        expected_blocks = [np.log1p(b.astype(np.float64)) for b in expected_blocks]
    expected_blocks = [b.astype(np.float64) for b in expected_blocks]
    expected = np.concatenate(expected_blocks)
    refs = [resolve_feature(store, f"gene{index}") for index in physical]
    requested = [refs[2], replace(refs[0], indices=(1, 7), reduction="mean"), refs[2]]
    expected = expected[:, [2, 0, 2]]
    expected[:, 1] = np.concatenate(expected_blocks)[:, :2].mean(axis=1)
    if geometry == "membership":
        old_boundaries = [
            (start, len(block))
            for start, block in spread_measured_rows(
                expected_blocks, measured, len(physical), row_size
            )
        ]
        expanded = np.full((len(cells), 3), np.nan)
        expanded[measured] = expected
        expected = expanded
    else:
        old_boundaries = [
            (start, min(row_size, len(cells) - start))
            for start in range(0, len(cells), row_size)
        ]
    with execution_report_scope() as reports:
        blocks = list(iter_normalized_feature_blocks(store, requested, cells, spec))
    np.testing.assert_array_equal(np.concatenate([b for _, _, b in blocks]), expected)
    assert [(start, len(block)) for _, start, block in blocks] == old_boundaries
    assert {report.unitKind for report in reports} == {
        "countsRowBlock" if geometry == "rows" else "countsTCellBand"
    }


def test_cell_value_means_share_orientation_and_membership(feature_store):
    store, measured, geometry = feature_store
    cells = np.arange(store.cells.N)
    features = np.array([21, 1, 7])
    expected = np.full(len(cells), np.nan)
    expected[measured] = (
        store.RNA.normed(cells[measured], features).mean(axis=1).compute(nthreads=1)
    )
    with execution_report_scope() as reports:
        actual = measured_feature_means(store.RNA, features, cells, nthreads=1)
        single = store.get_cell_vals("RNA", "everyone", "gene21")
    np.testing.assert_array_equal(actual, expected)
    expected[measured] = store.RNA.normed(cells[measured], np.array([21])).compute(
        nthreads=1
    )[:, 0]
    np.testing.assert_array_equal(single, expected)
    assert {report.unitKind for report in reports} == {
        "countsRowBlock" if geometry == "rows" else "countsTCellBand"
    }


@pytest.mark.parametrize("cells", [np.array([12, 2, 12, 4]), np.array([0, 1])])
def test_feature_reads_keep_counts_for_reordered_or_small_requests(
    feature_store, cells
):
    store, measured, _ = feature_store
    expected = np.full((len(cells), 1), np.nan)
    read = measured[cells]
    expected[read] = store.RNA.normed(cells[read], np.array([7])).compute(nthreads=1)
    with execution_report_scope() as reports:
        values = fetch_normalized_feature_matrix(
            store, [resolve_feature(store, "gene7")], cells
        )
    np.testing.assert_array_equal(values, expected)
    assert {report.unitKind for report in reports} == {"countsRowBlock"}


def test_feature_reads_keep_custom_normalization_context(feature_store, monkeypatch):
    store, measured, geometry = feature_store
    monkeypatch.setattr(store.RNA, "normMethod", norm_clr)
    spec = NormalizationSpec()
    assert not normalization_is_cell_local(store.RNA, spec)
    cells = np.arange(store.cells.N)
    expected = np.full((len(cells), 1), np.nan)
    expected[measured] = store.RNA.normed(cells[measured], np.array([7])).compute()
    with execution_report_scope() as reports:
        values = fetch_normalized_feature_matrix(
            store, [resolve_feature(store, "gene7")], cells
        )
    np.testing.assert_array_equal(values, expected)
    # CLR reduces the row blocks of the counts over every read cell, and the
    # values of those cells may then be read from countsT.
    assert {report.unitKind for report in reports} == (
        {"countsRowBlock"}
        if geometry == "rows"
        else {"countsRowBlock", "countsTCellBand"}
    )


def test_feature_read_eligibility_preserves_validation(feature_store, monkeypatch):
    store, _, _ = feature_store
    monkeypatch.setattr(store.RNA, "sf", None)
    with pytest.raises(ValueError, match="requires a size factor"):
        normalization_is_cell_local(store.RNA, NormalizationSpec())
    monkeypatch.setattr(store.RNA, "sf", 1_000)
    monkeypatch.setattr(store.RNA, "normMethod", norm_lib_size_log)
    with pytest.raises(ValueError, match="does not support log_transform"):
        normalization_is_cell_local(store.RNA, NormalizationSpec(transform="log1p"))


def test_cell_local_normalization_is_limited_to_built_in_normed(
    feature_store, monkeypatch
):
    from cytearc.assay import ATACassay, Assay, RNAassay
    from cytearc.assay.normalization import _PER_CELL_NORMED, norm_clr

    store, _, _ = feature_store
    assay = store.RNA
    spec = NormalizationSpec()
    assert normalization_is_cell_local(assay, spec)
    assert _PER_CELL_NORMED[RNAassay.normed] == (
        norm_dummy,
        norm_lib_size,
        norm_lib_size_log,
    )
    assert _PER_CELL_NORMED[Assay.normed] == (norm_dummy,)
    assert ATACassay.normed not in _PER_CELL_NORMED

    class OverridesNormed(RNAassay):
        # An attribute cannot opt an override in: its values come from it.
        cellLocalNormalizers = (norm_dummy, norm_lib_size, norm_lib_size_log)

        def normed(self, *args, **kwargs):
            return super().normed(*args, **kwargs)

    class InheritsNormed(RNAassay):
        pass

    for cls, expected in ((OverridesNormed, False), (InheritsNormed, True)):
        with monkeypatch.context() as patch:
            patch.setattr(assay, "__class__", cls)
            assert normalization_is_cell_local(assay, spec) is expected
    with monkeypatch.context() as patch:
        # Patching the instance dict leaves no bound method behind on undo.
        patch.setitem(vars(assay), "normed", lambda *args, **kwargs: None)
        assert not normalization_is_cell_local(assay, spec)
    assert "normed" not in vars(assay)
    with monkeypatch.context() as patch:
        patch.setattr(assay, "normMethod", norm_clr)
        assert not normalization_is_cell_local(assay, spec)
        assert normalization_is_cell_local(assay, NormalizationSpec(source="raw"))


def test_feature_reads_keep_the_values_of_an_overriding_normed(
    feature_store, monkeypatch
):
    from cytearc.assay import RNAassay

    store, _, _ = feature_store
    cells = np.arange(store.cells.N)
    features = [resolve_feature(store, "gene7"), resolve_feature(store, "gene21")]
    with execution_report_scope() as reports:
        expected = fetch_normalized_feature_matrix(store, features, cells)
    orientation = {report.unitKind for report in reports}

    class Doubled(RNAassay):
        def normed(self, *args, **kwargs):
            return super().normed(*args, **kwargs) * 2

    monkeypatch.setattr(store.RNA, "__class__", Doubled)
    with execution_report_scope() as reports:
        actual = fetch_normalized_feature_matrix(store, features, cells)
    # The override transforms the counts of either orientation.
    np.testing.assert_array_equal(actual, 2 * expected)
    assert {report.unitKind for report in reports} == orientation


def test_feature_reads_accept_eager_normalized_values(feature_store, monkeypatch):
    from cytearc.assay import RNAassay

    store, measured, _ = feature_store
    cells = np.arange(store.cells.N)
    indices = np.array([7, 21])
    expected = np.full((len(cells), len(indices)), np.nan)
    expected[measured] = store.RNA.normed(cells[measured], indices).compute(nthreads=1)

    class Eager(RNAassay):
        def normed(self, *args, **kwargs):
            return super().normed(*args, **kwargs).compute(nthreads=1)

    monkeypatch.setattr(store.RNA, "__class__", Eager)
    features = [resolve_feature(store, f"gene{index}") for index in indices]
    np.testing.assert_array_equal(
        fetch_normalized_feature_matrix(store, features, cells), expected
    )
    np.testing.assert_array_equal(
        measured_feature_means(store.RNA, indices, cells, nthreads=1),
        expected.mean(axis=1),
    )


def test_feature_reads_plan_counts_when_transpose_buffers_do_not_fit(
    feature_store, monkeypatch
):
    store, measured, geometry = feature_store
    if geometry != "columns":
        pytest.skip(
            "The selected fixture already uses counts or holds a membership mask"
        )
    cells = np.arange(store.cells.N)
    expected = store.RNA.normed(cells, np.array([7])).compute(nthreads=1)
    budget = ResourceBudget(8_000, 1)
    monkeypatch.setattr(store.RNA, "resources", budget)
    monkeypatch.setattr(store.RNA.rawData, "_resources", budget)
    with execution_report_scope() as reports:
        actual = fetch_normalized_feature_matrix(
            store, [resolve_feature(store, "gene7")], cells
        )
    np.testing.assert_array_equal(actual, expected)
    assert {report.unitKind for report in reports} == {"countsRowBlock"}


def test_feature_transpose_read_errors_propagate(feature_store, monkeypatch):
    store, _, geometry = feature_store
    if geometry != "columns":
        pytest.skip(
            "The selected fixture already uses counts or holds a membership mask"
        )
    from zarr.core.array import AsyncArray

    read = AsyncArray.get_orthogonal_selection

    async def failing_read(array, *args, **kwargs):
        if array.path == "RNA/countsT":
            raise OSError("requested gene chunk is unavailable")
        return await read(array, *args, **kwargs)

    monkeypatch.setattr(AsyncArray, "get_orthogonal_selection", failing_read)
    with pytest.raises(BaseExceptionGroup) as raised:
        fetch_normalized_feature_matrix(
            store, [resolve_feature(store, "gene7")], np.arange(store.cells.N)
        )
    assert (
        raised.value.subgroup(
            lambda error: (
                isinstance(error, OSError)
                and str(error) == "requested gene chunk is unavailable"
            )
        )
        is not None
    )


def test_feature_reads_use_numpy_backing_without_storage_geometry(
    feature_store, monkeypatch
):
    from cytearc.matrix import ChunkedArray

    store, measured, _ = feature_store
    counts = store.RNA.rawData.compute(nthreads=1) + 1
    cells = np.arange(len(counts))
    monkeypatch.setattr(store.RNA, "rawData", ChunkedArray.from_numpy(counts))
    values = fetch_normalized_feature_matrix(
        store,
        [resolve_feature(store, "gene7")],
        cells,
        NormalizationSpec(source="raw"),
    )
    expected = counts[:, [7]].astype(np.float64)
    expected[~measured] = np.nan
    np.testing.assert_array_equal(values, expected)


def test_feature_values_without_measured_cells_skip_reads(feature_store, monkeypatch):
    from cytearc.matrix import ChunkedArray
    from zarr.core.array import AsyncArray

    store, measured, _ = feature_store
    cells = np.flatnonzero(~measured)

    def unexpected_read(*args, **kwargs):
        pytest.fail("An empty measured selection must not read count data")

    monkeypatch.setattr(ChunkedArray, "_read", unexpected_read)
    monkeypatch.setattr(AsyncArray, "get_orthogonal_selection", unexpected_read)
    actual = measured_feature_means(store.RNA, np.array([7]), cells, nthreads=1)
    np.testing.assert_array_equal(actual, np.full(len(cells), np.nan))
    matrix = fetch_normalized_feature_matrix(
        store, [resolve_feature(store, "gene7")], cells
    )
    np.testing.assert_array_equal(matrix, np.full((len(cells), 1), np.nan))


@pytest.mark.parametrize("fault", ["unrequested", "incomplete"])
def test_feature_transpose_assembly_rejects_invalid_streams(
    feature_store, monkeypatch, fault
):
    from cytearc.storage import feature_stream
    from cytearc.storage.feature_major_rows import iter_feature_major_rows

    store, _, _ = feature_store
    stream = feature_stream.map_feature_cell_bands
    delivered = []

    def faulty_stream(array, process, **options):
        if fault == "unrequested":
            options["feat_idx"] = np.array([store.RNA.feats.N - 1])

        def deliver(band):
            delivered.append(band)
            return process(band)

        for result in stream(array, deliver, **options):
            if fault == "incomplete" and result is not None:
                yield result[:-1]
            else:
                yield result

    monkeypatch.setattr(feature_stream, "map_feature_cell_bands", faulty_stream)
    message = (
        "no requested features" if fault == "unrequested" else "every requested cell"
    )
    with pytest.raises((RuntimeError, ExceptionGroup)) as raised:
        list(
            iter_feature_major_rows(
                store.RNA.rawDataT,
                np.arange(store.cells.N),
                np.array([0]),
                block_rows=7,
                resources=ResourceBudget(store.RNA.resources.memoryBytes, 1),
                io=None,
                scratch_bytes=0,
            )
        )
    assert delivered
    if isinstance(raised.value, ExceptionGroup):
        assert (
            raised.value.subgroup(
                lambda error: isinstance(error, RuntimeError) and message in str(error)
            )
            is not None
        )
    else:
        assert message in str(raised.value)
