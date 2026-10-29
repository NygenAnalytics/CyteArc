from pathlib import Path

import numpy as np
import pytest

import cytearc.perturbation.signature as signature_module
from cytearc.datastore._operations import perturbation as perturbation_operations
from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import PerturbationSignature
from cytearc.perturbation.signature import (
    nearest_controls,
    normalize_counts,
    normalize_counts_by_feature,
    signature_of_values,
    signature_rows,
)
from cytearc.storage.artifacts import inspect_artifact
from cytearc.utils.logging import logger
from tests.perturbation_helpers import use_python_kernels
from tests.storage_helpers import write_count_store

N_CELLS = 48
# Guide 0 targets GENE_A, guide 1 GENE_B, and guide 2 is non-targeting.
TARGETS = ["GENE_A", "GENE_B", "NT"]


def _screen() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(11)
    carried = np.arange(N_CELLS) % 3
    guides = np.zeros((N_CELLS, 3), dtype=np.uint16)
    guides[np.arange(N_CELLS), carried] = 20
    rna = rng.poisson(5.0, size=(N_CELLS, 6))
    coordinates = rng.normal(size=(N_CELLS, 16))
    samples = np.where(np.arange(N_CELLS) < N_CELLS // 2, "run1", "run2")
    return carried, guides, rna.astype(np.uint16), coordinates, samples


CARRIED, GUIDES, RNA, COORDINATES, SAMPLES = _screen()
CONTROLS = np.flatnonzero(CARRIED == 2)


def _store(tmp_path: Path) -> DataStore:
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": RNA, "CRISPR": GUIDES}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    store._get_assay("CRISPR").feats.insert(
        "target_gene_name", np.asarray(TARGETS, dtype=object)
    )
    store.cells.insert("run", np.asarray(SAMPLES, dtype=object))
    return store


def _labels(store: DataStore):
    cells = store.select_measured_cells("CRISPR")
    calls = store.perturbation.call_guides(cells, method="fixed", min_umi=10)
    return store.perturbation.labels(calls, controls=["NT"])


def _coordinates(store: DataStore, cells=None):
    """Import COORDINATES for ``cells``, every cell by default."""
    cells = cells or store.select_measured_cells("RNA")
    selected = np.asarray(store.artifacts.load(cells)["values"][:], dtype=bool)
    return store.reduction.import_coordinates(
        COORDINATES[selected], cell_selection=cells, name="X_test"
    )


def _features(store: DataStore):
    return store.features.detected(
        store.select_measured_cells("RNA"), from_assay="RNA", min_cells=0
    )


def _expected(
    coordinates: np.ndarray,
    cells: np.ndarray,
    controls: np.ndarray,
    *,
    dims: int,
    k: int,
    scale_factor: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's signature and nearest controls, computed directly.

    ``coordinates`` holds one row per store cell.
    """
    totals = RNA.sum(axis=1).astype(float)
    scaled = np.divide(
        scale_factor * RNA.astype(float),
        totals[:, None],
        out=np.zeros(RNA.shape),
        where=totals[:, None] > 0,
    )
    rows, neighbors = [], []
    for cell in cells:
        gap = coordinates[controls, :dims] - coordinates[cell, :dims]
        nearest = controls[np.lexsort((controls, (gap**2).sum(axis=1)))[:k]]
        rows.append(np.log1p(scaled[cell]) - np.log1p(scaled[nearest].mean(axis=0)))
        neighbors.append(nearest)
    return np.asarray(rows), np.asarray(neighbors)


def _stored_coordinates(store: DataStore, ref, cells: np.ndarray) -> np.ndarray:
    """The imported coordinates of ``cells``, as one row per store cell."""
    values = np.zeros((N_CELLS, COORDINATES.shape[1]))
    values[cells] = np.asarray(store.artifacts.load(ref)["data"][:], dtype=np.float64)
    return values


def test_signature_subtracts_each_cells_nearest_controls(tmp_path) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    coordinates = _coordinates(store)
    features = _features(store)

    ref = store.perturbation.signature(
        labels, coordinates, features=features, dims=3, n_neighbors=4
    )

    assert ref.kind == "perturbation_signature"
    assert ref.assay == "RNA"
    loaded = store.perturbation.load_signature(ref)
    assert isinstance(loaded, PerturbationSignature)
    assert (loaded.from_assay, loaded.dims, loaded.n_neighbors) == ("RNA", 3, 4)
    assert (loaded.scale_factor, loaded.sample_column) == (10_000.0, None)
    cells = np.arange(N_CELLS)
    assert loaded.values.index.tolist() == [f"cell{cell}" for cell in cells]
    assert loaded.values.columns.tolist() == [f"RNA{index}" for index in range(6)]
    assert loaded.cells["is_control"].to_numpy().tolist() == (CARRIED == 2).tolist()
    assert loaded.cells["sample"].isna().all()
    values = _stored_coordinates(store, coordinates, cells)
    expected, neighbors = _expected(values, cells, CONTROLS, dims=3, k=4)
    np.testing.assert_allclose(loaded.values.to_numpy(), expected, rtol=1e-6, atol=1e-6)
    assert loaded.neighbors.to_numpy().tolist() == [
        [f"cell{cell}" for cell in row] for row in neighbors
    ]
    # Each control cell is its own nearest control.
    assert (loaded.neighbors.loc[[f"cell{cell}" for cell in CONTROLS], 0]).tolist() == [
        f"cell{cell}" for cell in CONTROLS
    ]
    assert (
        store.perturbation.signature(
            labels, coordinates, features=features, dims=3, n_neighbors=4
        )
        == ref
    )
    status = inspect_artifact(store.zw, ref)
    assert status.parameters == {
        "dims": 3,
        "n_neighbors": 4,
        "sample_columns": [],
        "scale_factor": 10_000.0,
    }
    assert set(status.inputs) == {
        "labels",
        "coordinates",
        "features",
        "dataset_fingerprint",
        "sample_assignments_fingerprint",
        "cell_selection",
    }
    scaled = store.perturbation.signature(
        labels, coordinates, features=features, dims=3, n_neighbors=4, scale_factor=100
    )
    expected_scaled, _ = _expected(
        values, cells, CONTROLS, dims=3, k=4, scale_factor=100.0
    )
    np.testing.assert_allclose(
        store.perturbation.load_signature(scaled).values.to_numpy(),
        expected_scaled,
        rtol=1e-6,
        atol=1e-6,
    )


def test_loading_a_signature_holds_its_values_once(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    ref = store.perturbation.signature(
        _labels(store),
        _coordinates(store),
        features=_features(store),
        dims=3,
        n_neighbors=4,
    )
    read: list[np.ndarray] = []
    as_zarr_array = perturbation_operations.as_zarr_array

    class Recorded:
        """The stored values, keeping each array that an orthogonal read returns."""

        def __init__(self, stored) -> None:
            self.oindex = self
            self.stored = stored

        def __getitem__(self, selection) -> np.ndarray:
            read.append(np.asarray(self.stored.oindex[selection]))
            return read[-1]

    def recording(node, *, name=""):
        stored = as_zarr_array(node, name=name)
        return Recorded(stored) if name == "values" else stored

    monkeypatch.setattr(perturbation_operations, "as_zarr_array", recording)

    loaded = store.perturbation.load_signature(ref)

    # The frame wraps the float32 values as read, the 4 bytes per value that
    # the budget check counts, instead of a copy.
    assert len(read) == 1 and read[0].dtype == np.float32
    assert np.shares_memory(loaded.values.to_numpy(), read[0])


def test_signature_keeps_each_sample_to_its_own_controls(tmp_path) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    coordinates = _coordinates(store)
    ref = store.perturbation.signature(
        labels,
        coordinates,
        features=_features(store),
        dims=4,
        n_neighbors=3,
        sample_column="run",
    )
    loaded = store.perturbation.load_signature(ref)
    assert loaded.sample_column == "run"
    assert loaded.cells["sample"].tolist() == SAMPLES.tolist()
    values = _stored_coordinates(store, coordinates, np.arange(N_CELLS))
    for sample in ("run1", "run2"):
        cells = np.flatnonzero(SAMPLES == sample)
        controls = cells[CARRIED[cells] == 2]
        expected, neighbors = _expected(values, cells, controls, dims=4, k=3)
        rows = [f"cell{cell}" for cell in cells]
        np.testing.assert_allclose(
            loaded.values.loc[rows].to_numpy(), expected, rtol=1e-6, atol=1e-6
        )
        assert loaded.neighbors.loc[rows].to_numpy().tolist() == [
            [f"cell{cell}" for cell in row] for row in neighbors
        ]


def test_cells_without_coordinates_get_no_signature(tmp_path) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    subset = store.select_cells(labels, include=["GENE_A", "control"])
    coordinates = _coordinates(store, subset)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        ref = store.perturbation.signature(
            labels, coordinates, features=_features(store), dims=2, n_neighbors=2
        )
    finally:
        logger.remove(sink)
    assert "16 labelled cells have no coordinates and get no signature" in messages
    loaded = store.perturbation.load_signature(ref, features=["RNA2", "RNA0"])
    cells = np.flatnonzero(CARRIED != 1)
    assert loaded.values.index.tolist() == [f"cell{cell}" for cell in cells]
    assert loaded.values.columns.tolist() == ["RNA2", "RNA0"]
    values = _stored_coordinates(store, coordinates, cells)
    expected, _ = _expected(values, cells, CONTROLS, dims=2, k=2)
    np.testing.assert_allclose(
        loaded.values.to_numpy(), expected[:, [2, 0]], rtol=1e-6, atol=1e-6
    )


def test_signature_defaults_to_the_features_of_the_coordinates(tmp_path) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    cells = store.select_measured_cells("RNA")
    features = store.features.snapshot(feature_indexes=np.asarray([1, 3, 4]))
    normalized = store.features.normalize(cells, features)
    pca = store.reduction.pca(normalized, dims=2, local_cache=False)
    ref = store.perturbation.signature(labels, pca, dims=2, n_neighbors=3)
    assert inspect_artifact(store.zw, ref).input_ref("features") == features
    columns = store.perturbation.load_signature(ref).values.columns
    assert columns.tolist() == ["RNA1", "RNA3", "RNA4"]
    with pytest.raises(ValueError, match="record no feature selection; pass features"):
        store.perturbation.signature(labels, _coordinates(store))


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"dims": 0}, ValueError, "dims must be at least 1"),
        ({"dims": 17}, ValueError, "dims must be at most the 16 dimensions"),
        ({"n_neighbors": True}, TypeError, "n_neighbors must be an integer"),
        ({"n_neighbors": 0}, ValueError, "n_neighbors must be at least 1"),
        ({"scale_factor": "1e4"}, TypeError, "scale_factor must be a real number"),
        ({"scale_factor": 0}, ValueError, "scale_factor must be above 0"),
        ({"sample_column": ""}, TypeError, "sample_column must be a non-empty"),
        ({"sample_column": "lane"}, ValueError, "'lane' not found in cell metadata"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
    ],
)
def test_signature_refuses_arguments_it_cannot_use(
    tmp_path, arguments, error, message
) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    with pytest.raises(error, match=message):
        store.perturbation.signature(
            labels, _coordinates(store), features=_features(store), **arguments
        )


def test_signature_refuses_inputs_it_cannot_use(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    coordinates = _coordinates(store)
    features = _features(store)
    cells = store.select_measured_cells("RNA")
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        store.perturbation.signature("labels", coordinates)
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.signature(cells, coordinates)
    with pytest.raises(TypeError, match="coordinates must be an ArtifactRef"):
        store.perturbation.signature(labels, "X_pca")
    with pytest.raises(ValueError, match="normalized artifact, not 'cell_selection'"):
        store.perturbation.signature(labels, cells)
    with pytest.raises(TypeError, match="features must be an ArtifactRef or None"):
        store.perturbation.signature(labels, coordinates, features="hvgs")
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.signature(labels, coordinates, features=cells)
    # 16 cells are controls, 8 in each run.
    with pytest.raises(ValueError, match="The cells have 16 control cells"):
        store.perturbation.signature(
            labels, coordinates, features=features, n_neighbors=17
        )
    with pytest.raises(ValueError, match="sample 'run1', sample 'run2' have 8"):
        store.perturbation.signature(
            labels, coordinates, features=features, n_neighbors=9, sample_column="run"
        )
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="16 control cells in 6 features"):
        store.perturbation.signature(
            labels, coordinates, features=features, n_neighbors=5
        )
    assert not store.artifacts.list(kind="perturbation_signature")


def test_signature_is_reused_read_only_and_refused_when_new(tmp_path) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    coordinates = _coordinates(store)
    settings = {"features": _features(store), "n_neighbors": 5}
    ref = store.perturbation.signature(labels, coordinates, **settings)
    reader = DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )
    assert reader.perturbation.signature(labels, coordinates, **settings) == ref
    with pytest.raises(PermissionError, match="compute_perturbation_signature"):
        reader.perturbation.signature(labels, coordinates, dims=2, **settings)
    assert (
        store.perturbation.signature(
            labels, coordinates, invalidate_cache=True, **settings
        )
        != ref
    )


def test_loading_a_signature_refuses_what_it_cannot_read(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    labels = _labels(store)
    ref = store.perturbation.signature(
        labels, _coordinates(store), features=_features(store), n_neighbors=3
    )
    with pytest.raises(TypeError, match="signature must be an ArtifactRef"):
        store.perturbation.load_signature("signature")
    with pytest.raises(ValueError, match="not 'perturbation_labels'"):
        store.perturbation.load_signature(labels)
    with pytest.raises(TypeError, match="a sequence of feature IDs"):
        store.perturbation.load_signature(ref, features="RNA0")
    with pytest.raises(ValueError, match="has no feature 'GENE_A'"):
        store.perturbation.load_signature(ref, features=["RNA0", "GENE_A"])
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="48 cells in 6 features"):
        store.perturbation.load_signature(ref)


def _brute_nearest(query: np.ndarray, controls: np.ndarray, k: int) -> np.ndarray:
    order = np.arange(len(controls))
    return np.asarray(
        [np.lexsort((order, ((controls - row) ** 2).sum(axis=1)))[:k] for row in query]
    )


def test_nearest_controls_are_exact_with_ties_to_the_first(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    query = rng.normal(size=(50, 3))
    controls = rng.normal(size=(40, 3))
    np.testing.assert_array_equal(
        nearest_controls(query, controls, 5), _brute_nearest(query, controls, 5)
    )
    # Blocks of one query give the same neighbors.
    monkeypatch.setattr(signature_module, "_DISTANCE_BYTES", 8)
    np.testing.assert_array_equal(
        nearest_controls(query, controls, 5), _brute_nearest(query, controls, 5)
    )
    # Thirty controls at one point tie for every query, so the first ones win.
    tied = np.tile([[0.5, -1.0, 2.0]], (30, 1))
    assert nearest_controls(query[:3], tied, 4).tolist() == [[0, 1, 2, 3]] * 3
    # With few controls, every control is a candidate.
    few = controls[:6]
    np.testing.assert_array_equal(
        nearest_controls(query, few, 6), _brute_nearest(query, few, 6)
    )


def test_nearest_controls_check_their_inputs() -> None:
    with pytest.raises(ValueError, match="the same dimensions"):
        nearest_controls(np.zeros((2, 3)), np.zeros((4, 2)), 1)
    with pytest.raises(ValueError, match="from 1 to the 4 controls, not 5"):
        nearest_controls(np.zeros((2, 3)), np.zeros((4, 3)), 5)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_signature_rows_follow_the_formula(monkeypatch, python_kernels) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, signature_module)
    rng = np.random.default_rng(4)
    counts = rng.poisson(3.0, size=(5, 4)).astype(float)
    totals = np.asarray([10.0, 20.0, 0.0, 40.0, 50.0])
    control_scaled = rng.uniform(0, 100, size=(3, 4))
    neighbors = np.asarray([[0, 1], [2, 1], [0, 0], [1, 2], [2, 0]])
    out = signature_rows(counts, totals, neighbors, control_scaled, 1_000.0)
    scale = np.divide(1_000.0, totals, out=np.zeros(5), where=totals > 0)
    expected = np.log1p(counts * scale[:, None]) - np.log1p(
        control_scaled[neighbors].mean(axis=1)
    )
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_signature_rows_check_their_inputs() -> None:
    counts = np.ones((2, 3))
    scaled = np.ones((4, 3))
    with pytest.raises(ValueError, match="one row per cell"):
        signature_rows(counts, np.ones(3), np.zeros((2, 1)), scaled, 1.0)
    with pytest.raises(ValueError, match="one column per feature"):
        signature_rows(counts, np.ones(2), np.zeros((2, 1)), np.ones((4, 2)), 1.0)
    with pytest.raises(ValueError, match="rows of the control values"):
        signature_rows(counts, np.ones(2), np.full((2, 1), 4), scaled, 1.0)


def test_normalized_counts_divide_before_they_scale() -> None:
    counts = np.asarray([[1.0, 0.0], [3.0, 2.0], [5.0, 1.0]])
    values = normalize_counts(counts, np.asarray([3_000.0, 9_000.0, 0.0]), 10_000.0)
    # Equal shares of two cells' counts give equal values, as in LogNormalize.
    assert values[0, 0] == values[1, 0] == 1.0 / 3_000.0 * 10_000.0
    assert values[2].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("python_kernels", [False, True])
def test_signatures_of_values_equal_those_of_counts(
    monkeypatch, python_kernels
) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, signature_module)
    rng = np.random.default_rng(12)
    counts = rng.poisson(3.0, size=(6, 5)).astype(float)
    totals = np.asarray([10.0, 20.0, 0.0, 40.0, 30.0, 7.0])
    values = normalize_counts(counts, totals, 1_000.0)
    # Cells 3 to 5 are the controls.
    neighbors = np.asarray([[3, 4], [4, 5], [5, 3], [3, 3], [4, 3], [5, 4]])
    out = signature_of_values(values.T, np.r_[4, 0, 2], np.arange(6), neighbors)
    expected = np.empty((6, 5))
    signature_module._signature_rows(
        counts, totals, neighbors - 3, values[3:], 1_000.0, expected
    )
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, expected[:, [4, 0, 2]])
    with pytest.raises(ValueError, match="one row of neighbors per cell"):
        signature_of_values(values.T, np.r_[0], np.r_[0, 1], neighbors[:1])


@pytest.mark.parametrize("python_kernels", [False, True])
def test_counts_normalized_by_feature_equal_normalized_counts(
    monkeypatch, python_kernels
) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, signature_module)
    counts = np.random.default_rng(13).poisson(4.0, size=(7, 70)).astype(np.uint16)
    totals = np.asarray([100.0, 0.0, 37.0, 250.0, 9.0, 1.0, 64.0])
    out = np.full((70, 10), -1.0)
    # Seventy features fill two tiles of the kernel.
    normalize_counts_by_feature(counts, totals, 1_000.0, out, 2)
    np.testing.assert_array_equal(
        out[:, 2:9], normalize_counts(counts, totals, 1_000.0).T
    )
    assert (out[:, [0, 1, 9]] == -1.0).all()
    with pytest.raises(ValueError, match="float64 features x cells out from start"):
        normalize_counts_by_feature(counts, totals, 1_000.0, out, 4)
