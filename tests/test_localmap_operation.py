"""ds.embeddings.localmap: inputs, identity, reuse and its place among embeddings."""

import shutil

import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.embeddings import localmap
from cytearc.graph.feature_projection import resolve_native_graph_inputs
from cytearc.plotting._data import _validated_embedding_selection
from cytearc.storage.artifacts import ArtifactRef
from cytearc.storage.budget import ResourceBudget

pytestmark = pytest.mark.slow

SHORT = (15, 15, 21)


def _only(store: DataStore, kind: str) -> ArtifactRef:
    refs = store.artifacts.list(kind=kind, from_assay="RNA", complete_only=True)
    assert len(refs) == 1
    return refs[0]


@pytest.fixture(scope="module")
def analyzed(analyzed_datastore_zarr_root, tmp_path_factory):
    """The analyzed store with its graph and initialization, before tests add artifacts."""
    root = tmp_path_factory.mktemp("localmap_operation") / "data.zarr"
    shutil.copytree(analyzed_datastore_zarr_root, root)
    store = DataStore(str(root), default_assay="RNA", nthreads=2)
    return (
        store,
        _only(store, "connectivity_map"),
        _only(store, "embedding_initialization"),
    )


@pytest.fixture
def store(analyzed):
    store = analyzed[0]
    resources = store.resources
    yield store
    store.resources = resources


@pytest.fixture
def graph(analyzed) -> ArtifactRef:
    return analyzed[1]


@pytest.fixture
def initialization(analyzed) -> ArtifactRef:
    return analyzed[2]


def _values(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(store.artifacts.load(ref)["values"][:])


def _expected_layout(
    store: DataStore, graph: ArtifactRef, start: np.ndarray, *, cosine: bool = False
) -> np.ndarray:
    """The layout from the graph's own coordinates and edges, by the kernels."""
    lineage = resolve_native_graph_inputs(store.zw, graph)
    coordinates = np.asarray(
        store.artifacts.load(lineage.coordinates)["data"][:], dtype=np.float32
    )
    if cosine:
        coordinates /= np.linalg.norm(coordinates, axis=1, keepdims=True)
    matrix = store.graph.load(graph)
    return localmap.localmap_layout(
        localmap.initial_layout(start),
        matrix,
        localmap.mid_near_pairs(coordinates, 5, 4444),
        localmap.further_pairs(matrix, 20, 4444),
        iterations=SHORT,
        random_seed=4444,
    )


def _refuse_reads(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError(message)

    for name in ("_load_graph_artifact", "_get_ini_embed", "_coordinate_data"):
        monkeypatch.setattr(store, name, fail)


def test_localmap_lays_out_the_graph_from_its_coordinates(
    store, graph, initialization
) -> None:
    ref = store.embeddings.localmap(graph, initialization, iterations=SHORT)

    status = store.artifacts.inspect(ref)
    assert ref.kind == "embedding"
    assert status.operation == "run_localmap"
    assert status.parameters["iterations"] == list(SHORT)
    assert ArtifactRef.from_dict(status.inputs["graph"]) == graph
    values = _values(store, ref)
    assert values.dtype == np.float32
    start = store._get_ini_embed(initialization, graph, 2)
    np.testing.assert_array_equal(values, _expected_layout(store, graph, start))


def test_localmap_reuse_reads_nothing(
    store, graph, initialization, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = store.embeddings.localmap(graph, initialization, iterations=SHORT)

    _refuse_reads(store, monkeypatch, "a reused embedding must not read its inputs")
    # Equal settings spelled differently, and another thread count, name the same layout.
    again = store.embeddings.localmap(
        graph,
        initialization,
        iterations=list(SHORT),
        learning_rate=1,
        max_distance=10,
        nthreads=1,
    )
    assert again == ref


def test_localmap_does_not_depend_on_threads(store, graph, initialization) -> None:
    def run(nthreads: int) -> np.ndarray:
        ref = store.embeddings.localmap(
            graph,
            initialization,
            iterations=SHORT,
            nthreads=nthreads,
            invalidate_cache=True,
        )
        return _values(store, ref)

    np.testing.assert_array_equal(run(1), run(3))


def test_localmap_takes_an_array_initialization_without_changing_it(
    store, graph
) -> None:
    n_cells = store.graph.load(graph).shape[0]
    start = np.random.default_rng(0).normal(size=(n_cells, 2))
    copy = start.copy()

    first = store.embeddings.localmap(graph, start, iterations=SHORT)
    second = store.embeddings.localmap(graph, start * 2.0, iterations=SHORT)

    np.testing.assert_array_equal(start, copy)
    # Scaling an initialization changes its identity but not its layout.
    assert first != second
    np.testing.assert_array_equal(_values(store, first), _values(store, second))
    np.testing.assert_array_equal(
        _values(store, first), _expected_layout(store, graph, start)
    )


def test_localmap_draws_mid_near_pairs_by_cosine_on_a_cosine_graph(
    store, graph
) -> None:
    lineage = resolve_native_graph_inputs(store.zw, graph)
    ann = store.graph.ann_index(lineage.coordinates, ann_metric="cosine")
    cosine_graph = store.graph.connectivity(store.graph.neighbors(ann, k=11))
    n_cells = store.graph.load(cosine_graph).shape[0]
    start = np.random.default_rng(1).normal(size=(n_cells, 2))

    ref = store.embeddings.localmap(cosine_graph, start, iterations=SHORT)

    expected = _expected_layout(store, cosine_graph, start, cosine=True)
    np.testing.assert_array_equal(_values(store, ref), expected)


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"iterations": (100, 100)}, TypeError, "three integers"),
        ({"iterations": (100, 0, 250)}, ValueError, r"iterations\[1\]"),
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"learning_rate": float("nan")}, ValueError, "learning_rate"),
        ({"max_distance": -1.0}, ValueError, "max_distance"),
        ({"n_mid_near": 0}, ValueError, "n_mid_near"),
        ({"n_further": 0}, ValueError, "n_further"),
        ({"localmap_dims": 0}, ValueError, "localmap_dims"),
        ({"random_seed": -1}, ValueError, "random_seed"),
    ],
)
def test_localmap_rejects_invalid_settings_before_reading(
    store,
    graph,
    initialization,
    monkeypatch: pytest.MonkeyPatch,
    settings,
    error,
    message,
) -> None:
    _refuse_reads(
        store, monkeypatch, "invalid settings must be refused before any read"
    )

    with pytest.raises(error, match=message):
        store.embeddings.localmap(graph, initialization, **settings)


def test_localmap_needs_a_connectivity_map(store, graph, initialization) -> None:
    neighbors = resolve_native_graph_inputs(store.zw, graph).neighbors
    integrated = ArtifactRef(
        scope="datastore", kind="integrated_graph", artifact_id="0" * 64, assay=None
    )

    for other in (neighbors, integrated):
        with pytest.raises(ValueError, match="connectivity_map"):
            store.embeddings.localmap(other, initialization)
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.embeddings.localmap("graph", initialization)


def test_localmap_refuses_a_read_only_store_before_reading(
    store, graph, initialization, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_only = DataStore(store.zarr_loc, default_assay="RNA", zarr_mode="r")
    _refuse_reads(
        read_only, monkeypatch, "a read-only store must refuse before reading"
    )

    with pytest.raises(PermissionError, match="run_localmap"):
        read_only.embeddings.localmap(graph, initialization, random_seed=7)


def test_localmap_checks_its_memory_budget_before_reading(
    store, graph, initialization, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.resources = ResourceBudget(50_000, 1)
    for name in ("_load_graph_artifact", "_get_ini_embed"):
        monkeypatch.setattr(
            store, name, lambda *_a, **_k: pytest.fail("read over budget")
        )

    with pytest.raises(MemoryError, match="LocalMAP"):
        store.embeddings.localmap(graph, initialization, random_seed=8)

    incomplete = [
        ref
        for ref in store.artifacts.list(kind="embedding", from_assay="RNA")
        if not store.artifacts.inspect(ref).complete
    ]
    assert incomplete == []


def test_localmap_is_found_and_plotted_like_other_embeddings(
    store, graph, initialization
) -> None:
    # Other tests save layouts of the same settings; this seed is this test's own.
    ref = store.embeddings.localmap(
        graph, initialization, iterations=SHORT, random_seed=12
    )

    found = store.artifacts.find(
        "embedding", method="localmap", graph=graph, random_seed=12
    )
    assert found == ref
    selection = resolve_native_graph_inputs(store.zw, graph).cell_selection
    assert _validated_embedding_selection(store, ref) == selection
