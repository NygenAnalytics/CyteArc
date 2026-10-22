"""``DataStore.qc.call_cells`` over stored unfiltered droplet matrices."""

import inspect
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from loguru import logger

from cytearc import DataStore
from cytearc.datastore._operations import cell_calling as operations
from cytearc.datastore.namespaces import QcAccessor
from cytearc.metadata.membership import UnmeasuredCellsError, membership_attributes
from cytearc.quality_control import BarcodeRanks, barcode_ranks, empty_drops
from cytearc.quality_control import cell_calling as domain
from cytearc.quality_control.filtering import RAW_DROPLET_REMEDY
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.refs import ArtifactRef
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.storage.stores import load_zarr
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts
from tests.test_cell_calling import _droplets

N_ITER = 2000


def _write_store(
    path: Path,
    counts: np.ndarray,
    *,
    dtype: str = "uint16",
    measured: np.ndarray | None = None,
) -> None:
    """Write raw droplet counts in small chunks, so reads take many blocks."""
    root = load_zarr(zarr_loc=str(path), mode="w")
    barcodes = np.array([f"bc{index}" for index in range(len(counts))])
    cell_data = create_cell_data(root, None, ids=barcodes, names=barcodes)
    if measured is not None:
        cell_data.create_array("RNA_I", data=measured)
        cell_data["RNA_I"].attrs.update(membership_attributes("RNA"))
    features = np.array([f"gene{index}" for index in range(counts.shape[1])])
    stored = create_zarr_count_assay(
        root,
        "RNA",
        None,
        len(counts),
        feat_ids=features,
        feat_names=features,
        dtype=dtype,
        policy=CountMatrixPolicy(unitBytes=64 * 1024, chunkBytes=16 * 1024),
    )
    stored[:] = counts.astype(dtype)
    finalize_test_counts(stored)
    finalize_writer_counts_t(root, "RNA", None)


def _open(path: Path, **options) -> DataStore:
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=2,
        **options,
    )


def _mask(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(store.artifacts.load(ref)["values"][:], dtype=bool)


@pytest.fixture(scope="module")
def droplets() -> tuple[np.ndarray, np.ndarray]:
    counts, truth, _ = _droplets(seed=7, n_features=150, n_empty=2500)
    return counts, truth


@pytest.fixture(scope="module")
def store_template(tmp_path_factory, droplets) -> Path:
    path = tmp_path_factory.mktemp("cell_calling") / "raw.zarr"
    _write_store(path, droplets[0])
    _open(path)
    return path


@pytest.fixture
def store_path(store_template, tmp_path) -> Path:
    target = tmp_path / "raw.zarr"
    shutil.copytree(store_template, target)
    return target


@pytest.fixture(scope="module")
def expected(droplets) -> domain.EmptyDropsResult:
    return empty_drops(droplets[0], n_iter=N_ITER, random_seed=3, nthreads=2)


def test_call_cells_matches_empty_drops_and_records_its_lineage(
    store_path, droplets, expected
) -> None:
    counts, truth = droplets
    store = _open(store_path)

    cells = store.qc.call_cells(n_iter=N_ITER, random_seed=3)

    assert (cells.kind, cells.scope) == ("cell_selection", "datastore")
    np.testing.assert_array_equal(_mask(store, cells), expected.is_cell(0.001))
    assert _mask(store, cells)[truth].mean() > 0.9
    selection = inspect_artifact(store.zw, cells)
    assert selection.operation == "call_cells"
    assert selection.parameters == {"fdr": 0.001}
    statistics = selection.input_ref("cell_calling")
    barcodes = selection.input_ref("prior_cell_selection")
    assert (statistics.kind, statistics.scope, statistics.assay) == (
        "cell_calling",
        "assay",
        "RNA",
    )
    status = inspect_artifact(store.zw, statistics)
    assert status.operation == "run_cell_calling"
    assert status.parameters == {
        "lower": 100,
        "retain": None,
        "n_iter": N_ITER,
        "alpha": None,
        "random_seed": 3,
    }
    assert status.input_ref("cell_selection") == barcodes
    assert set(status.inputs) == {"cell_selection", "dataset_fingerprint"}
    assert status.execution_options == {"nthreads": 2, "invalidate_cache": False}
    every_barcode = inspect_artifact(store.zw, barcodes)
    assert every_barcode.operation == "select_all_cells"
    assert _mask(store, barcodes).all()
    group = artifact_group(store.zw, statistics)
    assert {name: group.attrs[name] for name in operations._RESULT_ATTRIBUTES} == {
        "retain": expected.retain,
        "knee": expected.knee,
        "inflection": expected.inflection,
        "alpha": expected.alpha,
        "n_ambient": expected.n_ambient,
    }
    assert group["ambient"].shape == (counts.shape[1],)

    loaded = store.qc.load_cell_calls(cells)
    for name in ("total", "log_probability", "p_value", "fdr", "limited", "ambient"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(expected, name))
    assert loaded.barcodes.tolist() == [f"bc{index}" for index in range(len(counts))]
    assert (loaded.lower, loaded.retain, loaded.n_iter, loaded.random_seed) == (
        100,
        expected.retain,
        N_ITER,
        3,
    )
    assert (loaded.knee, loaded.inflection, loaded.alpha, loaded.n_ambient) == (
        expected.knee,
        expected.inflection,
        expected.alpha,
        expected.n_ambient,
    )
    table = loaded.to_pandas()
    assert table.index.name == "barcode"
    assert len(table) == int((counts.sum(axis=1) > 100).sum())


def test_cell_calling_statistics_are_cell_aligned_values(store_path) -> None:
    store = _open(store_path)
    cells = store.qc.call_cells(n_iter=N_ITER, random_seed=3)
    statistics = inspect_artifact(store.zw, cells).input_ref("cell_calling")

    fdr = store.artifacts.load_values(statistics)
    limited = store.artifacts.load_values(statistics, value="limited")
    reselected = store.select_cells(statistics, high=0.001, keep_bounds=True)

    assert (fdr.value, fdr.categorical) == ("fdr", False)
    assert (limited.value, limited.categorical) == ("limited", True)
    np.testing.assert_array_equal(fdr.values, store.qc.load_cell_calls(statistics).fdr)
    np.testing.assert_array_equal(_mask(store, reselected), _mask(store, cells))


def test_call_cells_reuses_statistics_and_recomputes_on_request(
    store_path, monkeypatch
) -> None:
    store = _open(store_path)
    first = store.qc.call_cells(n_iter=N_ITER, random_seed=3)
    statistics = inspect_artifact(store.zw, first).input_ref("cell_calling")

    def refuse(*args, **kwargs):
        raise AssertionError("saved statistics must be reused")

    with monkeypatch.context() as patched:
        patched.setattr(operations, "run_empty_drops", refuse)
        assert store.qc.call_cells(n_iter=N_ITER, random_seed=3) == first
        looser = store.qc.call_cells(n_iter=N_ITER, random_seed=3, fdr=0.05)
    assert looser != first
    assert inspect_artifact(store.zw, looser).input_ref("cell_calling") == statistics
    assert _mask(store, looser).sum() >= _mask(store, first).sum()

    again = store.qc.call_cells(n_iter=N_ITER, random_seed=3, invalidate_cache=True)
    recomputed = inspect_artifact(store.zw, again).input_ref("cell_calling")
    assert recomputed != statistics
    np.testing.assert_array_equal(_mask(store, again), _mask(store, first))
    np.testing.assert_array_equal(
        store.qc.load_cell_calls(recomputed).p_value,
        store.qc.load_cell_calls(statistics).p_value,
    )
    other_seed = store.qc.call_cells(n_iter=N_ITER, random_seed=4)
    assert inspect_artifact(store.zw, other_seed).input_ref("cell_calling") not in {
        statistics,
        recomputed,
    }


def test_statistics_without_their_ambient_profile_are_not_reused(store_path) -> None:
    store = _open(store_path)
    first = store.qc.call_cells(n_iter=N_ITER, random_seed=3)
    statistics = inspect_artifact(store.zw, first).input_ref("cell_calling")
    del artifact_group(store.zw, statistics)["ambient"]

    again = store.qc.call_cells(n_iter=N_ITER, random_seed=3)

    assert inspect_artifact(store.zw, again).input_ref("cell_calling") != statistics
    np.testing.assert_array_equal(_mask(store, again), _mask(store, first))


def test_opening_a_raw_matrix_keeps_i_and_points_to_cell_calling(
    droplets, tmp_path
) -> None:
    # Only the first writable open of a store filters I, so open a new one.
    store_path = tmp_path / "unopened.zarr"
    _write_store(store_path, droplets[0])
    # The default floor of 10 features would remove most barcodes.
    with pytest.warns(UserWarning) as caught:
        store = DataStore(str(store_path), default_assay="RNA", nthreads=2)

    assert store.cells.fetch_all("I").all()
    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 1
    assert "Will not remove low quality cells automatically" in messages[0]
    assert messages[0].endswith(RAW_DROPLET_REMEDY)
    assert "call cells first with qc.call_cells" in RAW_DROPLET_REMEDY


def test_call_cells_reads_every_barcode_whatever_i_holds(store_path, droplets) -> None:
    counts, _ = droplets
    store = _open(store_path)
    baseline = store.qc.call_cells(n_iter=N_ITER, random_seed=3)
    # Remove the ambient pool from I, as an open-time feature floor can.
    store.cells.update_key(counts.sum(axis=1) > 100, key="I")

    assert store.qc.call_cells(n_iter=N_ITER, random_seed=3) == baseline
    ranks = store.qc.barcode_ranks()
    np.testing.assert_array_equal(ranks.total, barcode_ranks(counts.sum(axis=1)).total)


def test_call_cells_streams_bounded_blocks(store_path, expected, monkeypatch) -> None:
    calls: dict[str, list[int]] = {"dense": [], "sparse": [], "kept": []}
    dense = domain._accumulate_ambient
    sparse_terms = domain._barcode_terms_sparse
    dense_terms = domain._barcode_terms
    run = domain.run_empty_drops

    def ambient_blocks(block, *args):
        calls["dense"].append(len(block))
        return dense(block, *args)

    def kept_blocks(indptr, *args):
        calls["sparse"].append(len(indptr) - 1)
        return sparse_terms(indptr, *args)

    def reread_blocks(block, *args):
        calls["kept"].append(-len(block))
        return dense_terms(block, *args)

    def record(*args, keep_tested, **kwargs):
        calls["kept"].append(int(keep_tested))
        return run(*args, keep_tested=keep_tested, **kwargs)

    monkeypatch.setattr(domain, "_accumulate_ambient", ambient_blocks)
    monkeypatch.setattr(domain, "_barcode_terms_sparse", kept_blocks)
    monkeypatch.setattr(domain, "_barcode_terms", reread_blocks)
    monkeypatch.setattr(operations, "run_empty_drops", record)
    n_barcodes = len(expected.total)
    n_tested = int((expected.total > 100).sum())

    # Ample memory keeps the tested rows from the one read of the counts.
    roomy = _open(store_path).qc.call_cells(n_iter=N_ITER, random_seed=3)
    assert calls["kept"] == [1]
    assert len(calls["dense"]) > 1 and sum(calls["dense"]) == n_barcodes
    assert max(calls["dense"]) < n_barcodes
    assert sum(calls["sparse"]) == n_tested

    # A budget below twice what the computation holds with the kept rows,
    # which the prepared feature counts size, reads the tested rows again.
    for name in calls:
        calls[name].clear()
    store = _open(store_path)
    totals = store.cells.fetch_all("RNA_nCounts")
    detected = store.cells.fetch_all("RNA_nFeatures")[totals > 100]
    n_features = store.RNA.feats.N
    resident = (
        72 * n_barcodes + 8 * n_features * 103 + 8 * 2 * (n_features + n_barcodes)
    )
    kept = int(12 * detected.sum()) + 8 * n_tested
    tight = _open(store_path, mem_budget=2 * (resident + kept) - 2)
    cells = tight.qc.call_cells(n_iter=N_ITER, random_seed=3, invalidate_cache=True)
    assert calls["kept"][0] == 0
    assert calls["sparse"] == []
    assert -sum(calls["kept"][1:]) == n_tested
    assert len(calls["dense"]) > 1 and sum(calls["dense"]) == n_barcodes
    np.testing.assert_array_equal(_mask(tight, cells), expected.is_cell(0.001))
    np.testing.assert_array_equal(
        tight.qc.load_cell_calls(cells).p_value, expected.p_value
    )
    np.testing.assert_array_equal(_mask(tight, roomy), _mask(tight, cells))


def test_call_cells_refuses_a_budget_below_its_resident_arrays(store_path) -> None:
    store = _open(store_path, mem_budget="100K")
    with pytest.raises(MemoryError, match="more than the memory budget of 102400"):
        store.qc.call_cells(n_iter=N_ITER)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"lower": 0}, ValueError, "lower must be at least 1"),
        ({"retain": 50}, ValueError, "retain must be at least 101"),
        ({"n_iter": 1.5}, TypeError, "n_iter must be an integer"),
        ({"alpha": -2.0}, ValueError, "alpha must be positive"),
        ({"fdr": 0}, ValueError, "fdr must be greater than 0 and at most 1"),
        ({"fdr": "high"}, TypeError, "fdr must be a real number"),
        ({"random_seed": -3}, ValueError, "random_seed must be at least 0"),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache must be a boolean"),
        ({"cell_selection": "I"}, TypeError, "cell_selection must be an ArtifactRef"),
    ],
)
def test_call_cells_checks_arguments_before_writing(
    store_path, arguments, error, message
) -> None:
    store = _open(store_path)
    with pytest.raises(error, match=message):
        store.qc.call_cells(**arguments)
    assert store.artifacts.list(scope="datastore") == []
    assert store.artifacts.list(kind="cell_calling") == []


def test_call_cells_reports_selections_and_pools_it_cannot_use(store_path) -> None:
    store = _open(store_path)
    empty = store.cells.fetch_all("I") & False
    store.cells.insert("nothing", empty)
    with pytest.raises(ValueError, match="cell_selection selects no barcode"):
        store.qc.call_cells(cell_selection=store.snapshot_cell_selection("nothing"))
    store.cells.insert("cells_only", store.cells.fetch_all("RNA_nCounts") > 100)
    with pytest.raises(ValueError, match="no ambient pool"):
        store.qc.call_cells(
            cell_selection=store.snapshot_cell_selection("cells_only"), retain=5000
        )


def test_call_cells_raises_when_no_barcode_is_called(store_path) -> None:
    store = _open(store_path)
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        # Nine iterations give p-values of at least 0.1: no barcode is
        # significant, and the cells reach that smallest p-value.
        with pytest.raises(ValueError, match="no cell was called"):
            store.qc.call_cells(n_iter=9, retain=100_000)
    finally:
        logger.remove(sink)
    (saved,) = store.artifacts.list(kind="cell_calling")
    statistics = store.qc.load_cell_calls(saved)
    assert np.nanmin(statistics.p_value) == pytest.approx(0.1)
    undecided = int(statistics.limited.sum())
    assert undecided > 0
    # The 150 simulated features also fit a Good-Turing slope above -1.
    assert [message for message in messages if "barcodes have" in message] == [
        f"{undecided} barcodes have the smallest p-value that n_iter=9 iterations "
        "give, but an adjusted p-value above fdr=0.001; call cells again with more "
        "iterations to decide them.\n"
    ]


def test_call_cells_warns_about_limited_barcodes_when_it_reuses_them(
    store_path,
) -> None:
    store = _open(store_path)
    first = store.qc.call_cells(n_iter=200, fdr=0.05)
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        strict = store.qc.call_cells(n_iter=200, fdr=0.001)
    finally:
        logger.remove(sink)
    statistics = store.qc.load_cell_calls(strict)

    undecided = statistics.limited & (statistics.fdr > 0.001)
    assert undecided.any()
    assert [message for message in messages if "barcodes have" in message] == [
        f"{int(undecided.sum())} barcodes have the smallest p-value that n_iter=200 "
        "iterations give, but an adjusted p-value above fdr=0.001; call cells again "
        "with more iterations to decide them.\n"
    ]
    assert _mask(store, strict).sum() < _mask(store, first).sum()


def test_call_cells_needs_unsigned_integer_counts(tmp_path, droplets) -> None:
    counts = droplets[0][:600].astype(np.float32)
    counts[0, 0] = 0.5
    _write_store(tmp_path / "fractional.zarr", counts, dtype="float32")
    store = _open(tmp_path / "fractional.zarr")
    with pytest.raises(TypeError, match="non-negative integer counts.*float32"):
        store.qc.call_cells()


def test_call_cells_on_a_read_only_store_reuses_saved_results(store_path) -> None:
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        _open(store_path, zarr_mode="r").qc.call_cells(n_iter=N_ITER)
    cells = _open(store_path).qc.call_cells(n_iter=N_ITER, random_seed=3)

    reader = _open(store_path, zarr_mode="r")
    assert reader.qc.call_cells(n_iter=N_ITER, random_seed=3) == cells
    statistics = inspect_artifact(reader.zw, cells).input_ref("cell_calling")
    assert reader.qc.load_cell_calls(statistics).retain > 100
    # Saved statistics are reused, but a new selection cannot be written.
    with pytest.raises(PermissionError):
        reader.qc.call_cells(n_iter=N_ITER, random_seed=3, fdr=0.2)


def test_call_cells_calls_only_barcodes_the_assay_measured(tmp_path, droplets) -> None:
    counts, _ = droplets
    measured = np.ones(len(counts), dtype=bool)
    measured[::7] = False
    counts = counts.copy()
    counts[~measured] = 0
    _write_store(tmp_path / "measured.zarr", counts, measured=measured)
    store = _open(tmp_path / "measured.zarr")

    cells = store.qc.call_cells(n_iter=N_ITER, random_seed=3)

    selection = inspect_artifact(store.zw, cells)
    barcodes = selection.input_ref("prior_cell_selection")
    assert inspect_artifact(store.zw, barcodes).operation == "select_measured_cells"
    np.testing.assert_array_equal(_mask(store, barcodes), measured)
    expected = empty_drops(counts[measured], n_iter=N_ITER, random_seed=3, nthreads=2)
    called = np.zeros(len(counts), dtype=bool)
    called[np.flatnonzero(measured)[expected.is_cell()]] = True
    np.testing.assert_array_equal(_mask(store, cells), called)
    np.testing.assert_array_equal(
        store.qc.barcode_ranks().total,
        barcode_ranks(counts[measured].sum(axis=1)).total,
    )
    everything = store.snapshot_cell_selection("I")
    with pytest.raises(UnmeasuredCellsError, match="run_cell_calling"):
        store.qc.call_cells(cell_selection=everything)
    with pytest.raises(UnmeasuredCellsError, match="barcode_ranks"):
        store.qc.barcode_ranks(cell_selection=everything)


def test_load_cell_calls_accepts_a_calling_and_refuses_other_artifacts(
    store_path,
) -> None:
    store = _open(store_path)
    cells = store.qc.call_cells(n_iter=N_ITER, random_seed=3)
    statistics = inspect_artifact(store.zw, cells).input_ref("cell_calling")
    barcodes = inspect_artifact(store.zw, cells).input_ref("prior_cell_selection")

    from_selection = store.qc.load_cell_calls(cells)
    from_statistics = store.qc.load_cell_calls(statistics)
    np.testing.assert_array_equal(from_selection.fdr, from_statistics.fdr)
    np.testing.assert_array_equal(from_selection.barcodes, from_statistics.barcodes)

    with pytest.raises(TypeError, match="calls must be an ArtifactRef"):
        store.qc.load_cell_calls("cells")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="that qc.call_cells returned"):
        store.qc.load_cell_calls(barcodes)
    features = store.features.universe(from_assay="RNA")
    with pytest.raises(ValueError, match="not a feature_selection artifact"):
        store.qc.load_cell_calls(features)
    artifact_group(store.zw, statistics).attrs["complete"] = False
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        store.qc.load_cell_calls(statistics)


def test_barcode_ranks_reads_the_prepared_totals(store_path, droplets) -> None:
    counts, _ = droplets
    store = _open(store_path, zarr_mode="r")
    totals = counts.sum(axis=1)

    ranks = store.qc.barcode_ranks()
    expected = barcode_ranks(totals)
    assert isinstance(ranks, BarcodeRanks)
    for name in ("rank", "total", "fitted"):
        np.testing.assert_array_equal(getattr(ranks, name), getattr(expected, name))
    assert (ranks.knee, ranks.inflection) == (expected.knee, expected.inflection)
    assert store.artifacts.list(scope="datastore") == []

    writable = _open(store_path)
    writable.cells.insert("large", totals > 50)
    subset = writable.qc.barcode_ranks(
        cell_selection=writable.snapshot_cell_selection("large"), lower=60
    )
    np.testing.assert_array_equal(
        subset.total, barcode_ranks(totals[totals > 50]).total
    )
    assert subset.lower == 60
    with pytest.raises(ValueError, match="lower must be at least 0"):
        store.qc.barcode_ranks(lower=-1)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.qc.barcode_ranks(cell_selection="I")  # type: ignore[arg-type]


def test_cell_calling_methods_keep_their_signatures() -> None:
    assert str(inspect.signature(QcAccessor.call_cells)) == (
        "(self, from_assay: str | None = None, *, "
        "cell_selection: cytearc.storage.refs.ArtifactRef | None = None, "
        "lower: int = 100, retain: int | None = None, n_iter: int = 10000, "
        "alpha: float | None = None, fdr: float = 0.001, random_seed: int = 0, "
        "invalidate_cache: bool = False) -> cytearc.storage.refs.ArtifactRef"
    )
    assert str(inspect.signature(QcAccessor.load_cell_calls)) == (
        "(self, calls: cytearc.storage.refs.ArtifactRef) -> "
        "cytearc.quality_control.cell_calling.EmptyDropsResult"
    )
    assert str(inspect.signature(QcAccessor.barcode_ranks)) == (
        "(self, from_assay: str | None = None, *, "
        "cell_selection: cytearc.storage.refs.ArtifactRef | None = None, "
        "lower: int = 100) -> cytearc.quality_control.cell_calling.BarcodeRanks"
    )
    assert math.isclose(operations._DEFAULT_FDR, 0.001)
