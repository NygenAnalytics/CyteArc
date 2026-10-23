"""Checks that graph consumers run before they load a graph or compute.

Leiden, Paris, UMAP, and t-SNE refuse a read-only store before they read the
graph when they cannot reuse a saved result, and they, with the connectivity
conversion, admit their memory before the graph is read. The graph-loading
flags and initialization references are checked first too, and SNN
integration refuses graphs of different k before loading any of them.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix, triu
from zarr.storage import MemoryStore

from cytearc import DataStore
from cytearc.clustering.leiden import native_leiden_peak_bytes
from cytearc.datastore._operations.graph import _GraphOperationsMixin
from cytearc.datastore._pipeline_recipe import _resolve_params
from cytearc.embeddings.sgtsne import sgtsne_input_bytes
from cytearc.embeddings.umap import fuzzy_simplicial_set, umap_peak_bytes
from cytearc.graph.arguments import graph_load_flags, umap_graph_flags
from cytearc.neighbors.graph import (
    GraphLoadMemory,
    connectivity_peak_bytes,
    graph_load_memory,
)
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.budget import ResourceBudget
from tests.storage_helpers import write_count_store

N_CELLS = 150
K = 6


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=30), size=(N_CELLS, 30))
    rna[: N_CELLS // 2, :6] += 5
    rna[:, 0] += 1
    return {"RNA": rna, "ADT": rng.poisson(20.0, size=(N_CELLS, 5)) + 1}


def _open(zarr_loc: Path, **options) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


def _graph_chain(store: DataStore, assay: str, k: int) -> dict[str, ArtifactRef]:
    cells = store.snapshot_cell_selection()
    normalized = store.features.normalize(
        cells, store.features.universe(from_assay=assay)
    )
    pca = store.reduction.pca(normalized, dims=4)
    neighbors = store.graph.neighbors(store.graph.ann_index(pca), k=k)
    return {
        "pca": pca,
        "neighbors": neighbors,
        "graph": store.graph.connectivity(neighbors),
    }


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    zarr_loc = tmp_path_factory.mktemp("graph_consumer_guards") / "store.zarr"
    write_count_store(str(zarr_loc), _counts(), "uint16")
    store = _open(zarr_loc)
    rna = _graph_chain(store, "RNA", K)
    adt = _graph_chain(store, "ADT", K)
    adt_wider = _graph_chain(store, "ADT", K + 3)
    initialization = store.embeddings.initialization(rna["pca"], n_centroids=20)
    graph = rna["graph"]
    refs = {
        **{f"rna_{name}": ref for name, ref in rna.items()},
        "adt_graph": adt["graph"],
        "adt_wider_graph": adt_wider["graph"],
        "initialization": initialization,
        "leiden": store.clusters.leiden(graph),
        "umap": store.embeddings.umap(graph, initialization, n_epochs=5),
        "paris_fixed": store.clusters.paris(graph, n_clusters=3),
        "paris_auto": store.clusters.paris(graph),
    }
    return zarr_loc, refs


@pytest.fixture
def writable(template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open(target), refs


def _forbid(monkeypatch, store: DataStore, *targets: str) -> None:
    """Make the graph loader and the named kernels fail if they are reached."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a guard must run before the graph is read")

    monkeypatch.setattr(store, "_store_to_sparse", forbidden)
    for target in targets:
        monkeypatch.setattr(target, forbidden)


_KERNELS = (
    "cytearc.clustering.leiden.leiden_membership",
    "cytearc.embeddings.umap.fit_transform",
    "cytearc.embeddings.sgtsne.run_sgtsne",
    "cytearc.clustering.paris.fit_paris_hierarchy",
)


# Read-only stores


def test_read_only_consumers_reuse_saved_results_without_reading_the_graph(
    template, monkeypatch
) -> None:
    zarr_loc, refs = template
    read_only = _open(zarr_loc, zarr_mode="r")
    _forbid(monkeypatch, read_only, *_KERNELS)
    graph = refs["rna_graph"]

    assert read_only.clusters.leiden(graph) == refs["leiden"]
    assert (
        read_only.embeddings.umap(graph, refs["initialization"], n_epochs=5)
        == refs["umap"]
    )
    assert read_only.clusters.paris(graph, n_clusters=3) == refs["paris_fixed"]
    assert read_only.clusters.paris(graph) == refs["paris_auto"]


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            lambda store, refs: store.clusters.leiden(
                refs["rna_graph"], resolution=0.77
            ),
            "run_leiden_clustering",
        ),
        (
            lambda store, refs: store.embeddings.umap(
                refs["rna_graph"], refs["initialization"], n_epochs=7
            ),
            "run_umap",
        ),
        (
            lambda store, refs: store.embeddings.tsne(
                refs["rna_graph"], refs["initialization"], max_iter=33
            ),
            "run_tsne",
        ),
        # The hierarchy is saved, so only the new cut is missing.
        (
            lambda store, refs: store.clusters.paris(refs["rna_graph"], n_clusters=4),
            "cut_paris_hierarchy",
        ),
        (
            lambda store, refs: store.clusters.paris(
                refs["rna_graph"], min_cluster_size=3
            ),
            "cut_paris_hierarchy",
        ),
        # A new graph has no hierarchy either.
        (
            lambda store, refs: store.clusters.paris(refs["adt_graph"]),
            "fit_paris_hierarchy",
        ),
    ],
    ids=["leiden", "umap", "tsne", "paris-fixed", "paris-auto", "paris-new-graph"],
)
def test_read_only_consumers_refuse_before_reading_the_graph(
    template, monkeypatch, call, operation
) -> None:
    zarr_loc, refs = template
    read_only = _open(zarr_loc, zarr_mode="r")
    _forbid(
        monkeypatch,
        read_only,
        *_KERNELS,
        "cytearc.datastore._operations.paris_persistence.load_hierarchy_group",
    )

    with pytest.raises(
        PermissionError,
        match=f"^{operation} requires a DataStore opened with zarr_mode='r\\+'$",
    ):
        call(read_only, refs)


def test_read_only_paris_refuses_a_missing_dendrogram_before_loading(
    writable, monkeypatch
) -> None:
    store, refs = writable
    fixed = store.artifacts.inspect(refs["paris_fixed"])
    hierarchy = fixed.input_ref("cluster_hierarchy")
    (dendrogram,) = [
        ref
        for ref in store.artifacts.list(kind="dendrogram", from_assay="RNA")
        if store.artifacts.inspect(ref).input_ref("cluster_hierarchy") == hierarchy
    ]
    del store.zw[store.artifacts.inspect(dendrogram).path]
    read_only = _open(Path(store.zarr_loc), zarr_mode="r")
    _forbid(
        monkeypatch,
        read_only,
        "cytearc.datastore._operations.paris_persistence.load_hierarchy_group",
    )

    with pytest.raises(
        PermissionError, match="^materialize_paris_dendrogram requires a DataStore"
    ):
        read_only.clusters.paris(refs["rna_graph"], n_clusters=3)


# Memory admission


def _load(store: DataStore, graph: ArtifactRef, **flags) -> GraphLoadMemory:
    group = artifact_group(store.zw, graph)
    return graph_load_memory(
        N_CELLS,
        int(group["edges"].shape[0]),
        symmetric=flags.get("symmetric", False),
        upper_only=flags.get("upper_only", False),
    )


def _required(load: GraphLoadMemory, held: int, working: int) -> int:
    return held + max(load.peakBytes, load.heldBytes + working)


def _budget_message(work: str, required: int, limit: int) -> str:
    return (
        f"{work} of {N_CELLS} cells needs about {required} bytes in memory for a "
        f"graph of {N_CELLS * K} edges, but the operation limit is {limit} bytes. "
        "Raise mem_budget."
    )


def test_consumers_admit_their_graph_memory_before_reading_it(
    writable, monkeypatch
) -> None:
    store, refs = writable
    graph = refs["rna_graph"]
    initialization = refs["initialization"]
    edges = N_CELLS * K
    directed = _load(store, graph)
    symmetric = _load(store, graph, symmetric=True)
    upper = _load(store, graph, symmetric=True, upper_only=True)
    cases = [
        (
            lambda: store.clusters.leiden(graph, resolution=0.6),
            "Leiden clustering",
            _required(directed, 0, native_leiden_peak_bytes(N_CELLS, 2 * edges)),
        ),
        (
            lambda: store.clusters.leiden(graph, resolution=0.6, symmetric_graph=True),
            "Leiden clustering",
            _required(symmetric, 0, native_leiden_peak_bytes(N_CELLS, 2 * edges)),
        ),
        (
            lambda: store.clusters.leiden(
                graph, resolution=0.6, symmetric_graph=True, graph_upper_only=True
            ),
            "Leiden clustering",
            _required(upper, 0, native_leiden_peak_bytes(N_CELLS, 2 * edges)),
        ),
        # Only the graph load is counted for the GPL backends.
        (
            lambda: store.clusters.leiden(graph, resolution=0.6, backend="igraph"),
            "Leiden clustering",
            _required(directed, 0, 0),
        ),
        (
            lambda: store.embeddings.umap(graph, initialization, n_epochs=6),
            "UMAP",
            _required(
                directed,
                N_CELLS * 2 * 4,
                umap_peak_bytes(N_CELLS, edges, 2 * edges),
            ),
        ),
        (
            lambda: store.embeddings.umap(
                graph, initialization, n_epochs=6, use_density_map=True
            ),
            "UMAP",
            _required(
                directed,
                N_CELLS * 2 * 4,
                umap_peak_bytes(N_CELLS, edges, 2 * edges, densmap_edges=edges),
            ),
        ),
        (
            lambda: store.embeddings.tsne(
                graph,
                np.zeros((N_CELLS, 3)),
                tsne_dims=3,
                symmetric_graph=True,
                graph_upper_only=True,
            ),
            "t-SNE",
            _required(upper, N_CELLS * 3 * 8, sgtsne_input_bytes(N_CELLS, edges, 3)),
        ),
    ]
    monkeypatch.setattr(
        "cytearc.embeddings.sgtsne.require_sgtsnepi", lambda: lambda *_a, **_k: None
    )
    for call, work, required in cases:
        _forbid(monkeypatch, store, *_KERNELS)
        store.resources = ResourceBudget(required - 1, 1)
        with pytest.raises(MemoryError) as caught:
            call()
        assert str(caught.value) == _budget_message(work, required, required - 1)

    neighbors = refs["rna_neighbors"]
    required = connectivity_peak_bytes(N_CELLS, K)
    store.resources = ResourceBudget(required - 1, 1)
    monkeypatch.setattr(
        "cytearc.neighbors.graph.build_connectivity_arrays",
        lambda *_args, **_kwargs: pytest.fail("the neighbors were converted"),
    )
    with pytest.raises(MemoryError) as caught:
        store.graph.connectivity(neighbors, bandwidth=1.1)
    assert str(caught.value) == (
        f"Building the connectivity map of {N_CELLS} cells needs about {required} "
        f"bytes in memory for a graph of {N_CELLS * K} edges, but the operation "
        f"limit is {required - 1} bytes. Raise mem_budget."
    )
    assert store.artifacts.list(kind="cluster_labels", from_assay="RNA") == [
        refs["leiden"]
    ]


def test_consumers_run_within_the_admitted_budget(writable) -> None:
    store, refs = writable
    graph = refs["rna_graph"]
    edges = N_CELLS * K
    directed = _load(store, graph)
    leiden = _required(directed, 0, native_leiden_peak_bytes(N_CELLS, 2 * edges))
    umap = _required(
        directed, N_CELLS * 2 * 4, umap_peak_bytes(N_CELLS, edges, 2 * edges)
    )

    store.resources = ResourceBudget(leiden, 1)
    labels = store.clusters.leiden(graph, resolution=0.6)
    store.resources = ResourceBudget(umap, 1)
    layout = store.embeddings.umap(graph, refs["initialization"], n_epochs=6)
    store.resources = ResourceBudget(connectivity_peak_bytes(N_CELLS, K), 1)
    connectivity = store.graph.connectivity(refs["rna_neighbors"], bandwidth=1.1)

    for ref in (labels, layout, connectivity):
        assert store.artifacts.inspect(ref).complete


# Graph-loading flags and initialization references


def test_graph_load_flags_require_symmetric_for_the_upper_triangle() -> None:
    assert graph_load_flags(False, False) == (False, False)
    assert graph_load_flags(np.True_, False) == (True, False)
    assert graph_load_flags(True, np.True_) == (True, True)
    with pytest.raises(ValueError, match="graph_upper_only=True applies only with"):
        graph_load_flags(False, True)
    for value in (1, None):
        with pytest.raises(TypeError, match="graph_upper_only must be a boolean"):
            graph_load_flags(True, value)
        with pytest.raises(TypeError, match="symmetric_graph must be a boolean"):
            graph_load_flags(value, False)


def test_umap_graph_flags_accept_only_false() -> None:
    assert umap_graph_flags(np.False_, False) == (False, False)
    for flags in ((True, False), (False, True), (True, True)):
        with pytest.raises(ValueError, match="union twice") as caught:
            umap_graph_flags(*flags)
        assert str(caught.value) == (
            "embeddings.umap symmetrizes the graph itself with the fuzzy union "
            "w_ij + w_ji - w_ij * w_ji, so symmetric_graph=True would apply that "
            "union twice and graph_upper_only=True would change nothing; leave "
            "both False"
        )
    with pytest.raises(TypeError, match="symmetric_graph must be a boolean"):
        umap_graph_flags(1, False)


def test_umap_unites_a_graph_once_and_its_symmetrized_graph_twice() -> None:
    # Directed weights: 0 -> 1 is 0.5 and 1 -> 0 is 0.25; 1 -> 2 is 1.0.
    directed = csr_matrix(
        (np.array([0.5, 0.25, 1.0], dtype=np.float32), ([0, 1, 1], [1, 0, 2])),
        shape=(3, 3),
    )
    union = np.array(
        [[0.0, 0.625, 0.0], [0.625, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    symmetrized = (directed + directed.T) - directed.multiply(directed.T)

    np.testing.assert_array_equal(
        fuzzy_simplicial_set(directed.tocoo(), 1.0).toarray(), union
    )
    # The union of the upper triangle is the union of the graph, so
    # graph_upper_only adds nothing, while uniting the symmetrized graph again
    # turns each weight w into 2w - w**2.
    np.testing.assert_array_equal(
        fuzzy_simplicial_set(triu(symmetrized).tocoo(), 1.0).toarray(), union
    )
    np.testing.assert_allclose(
        fuzzy_simplicial_set(symmetrized.tocoo(), 1.0).toarray(),
        2 * union - union**2,
        rtol=1e-6,
    )


def test_graph_consumers_check_flags_before_reading(writable, monkeypatch) -> None:
    store, refs = writable
    graph = refs["rna_graph"]
    initialization = refs["initialization"]
    _forbid(monkeypatch, store, *_KERNELS)
    monkeypatch.setattr(
        store,
        "_graph_location",
        lambda _graph: pytest.fail("the graph was resolved"),
    )

    for flags in (
        {"symmetric_graph": True},
        {"graph_upper_only": True},
        {"symmetric_graph": True, "graph_upper_only": True},
    ):
        with pytest.raises(ValueError, match="union twice"):
            store.embeddings.umap(graph, initialization, **flags)
    for call in (
        lambda: store.clusters.leiden(graph, graph_upper_only=True),
        lambda: store.embeddings.tsne(graph, initialization, graph_upper_only=True),
    ):
        with pytest.raises(ValueError, match="applies only with symmetric_graph"):
            call()


@pytest.mark.parametrize(
    ("section", "flags", "message"),
    [
        ("umap", {"symmetric_graph": True}, "union twice"),
        ("umap", {"graph_upper_only": np.True_}, "union twice"),
        ("leiden", {"graph_upper_only": True}, "applies only with"),
        ("tsne", {"graph_upper_only": True}, "applies only with"),
        ("leiden", {"symmetric_graph": 1}, "symmetric_graph must be a boolean"),
    ],
)
def test_pipeline_params_check_graph_flags_before_a_run(section, flags, message):
    with pytest.raises((ValueError, TypeError), match=message):
        _resolve_params({section: flags})

    accepted = _resolve_params(
        {section: {"symmetric_graph": False, "graph_upper_only": False}}
    )[0]
    assert accepted == {section: {"symmetric_graph": False, "graph_upper_only": False}}


def test_embeddings_refuse_other_references_as_initialization(
    writable, monkeypatch
) -> None:
    store, refs = writable
    _forbid(monkeypatch, store, *_KERNELS)
    monkeypatch.setattr(
        "cytearc.datastore._operations.embeddings.graph_cell_selection",
        lambda *_args: pytest.fail("the graph was read"),
    )

    for method in (store.embeddings.umap, store.embeddings.tsne):
        with pytest.raises(ValueError) as caught:
            method(refs["rna_graph"], refs["rna_pca"])
        assert str(caught.value) == (
            "initialization must be an embedding_initialization ref, not a "
            "'reduction' ref; build one from the graph's coordinates with "
            "embeddings.initialization(coordinates), or pass an array of initial "
            "coordinates"
        )
    scoped = ArtifactRef(
        scope="datastore", kind="embedding_initialization", artifact_id="a" * 64
    )
    with pytest.raises(ValueError, match="must be an assay-scoped artifact"):
        store.embeddings.umap(refs["rna_graph"], scoped)


def test_load_symmetric_returns_the_fuzzy_union() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("graph")
    group.attrs.update({"n_cells": 3, "n_neighbors": 1})
    # The directed edges of the union test above, one row per stored edge.
    group.create_array(
        "edges", data=np.array([[0, 1], [1, 0], [1, 2]], dtype=np.uint32)
    )
    group.create_array("weights", data=np.array([0.5, 0.25, 1.0], dtype=np.float32))

    class _Reader:
        _get_graph_ncells_k = _GraphOperationsMixin._get_graph_ncells_k
        _store_to_sparse = _GraphOperationsMixin._store_to_sparse
        _load_graph_artifact = _GraphOperationsMixin._load_graph_artifact

        def __init__(self) -> None:
            self.zw = root

        def _graph_location(self, _graph):
            return "graph"

    reader = _Reader()
    loaded = reader._load_graph_artifact(
        None, symmetric=True, upper_only=False, use_k=None
    )
    upper = reader._load_graph_artifact(
        None, symmetric=True, upper_only=True, use_k=None
    )

    # a + b - a * b, not a + b: 0.5 + 0.25 - 0.125.
    np.testing.assert_array_equal(
        loaded.toarray(),
        [[0.0, 0.625, 0.0], [0.625, 0.0, 1.0], [0.0, 1.0, 0.0]],
    )
    np.testing.assert_array_equal(
        upper.toarray(), [[0.0, 0.625, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )


# SNN integration


def test_snn_integration_refuses_graphs_of_different_k_before_loading(
    writable, monkeypatch
) -> None:
    store, refs = writable
    _forbid(monkeypatch, store)
    before = store.artifacts.list(kind="integrated_graph", scope="datastore")

    with pytest.raises(ValueError) as caught:
        store.integration.modalities(
            [refs["rna_graph"], refs["adt_wider_graph"]], method="snn"
        )

    assert str(caught.value) == (
        f"SNN integration needs graphs with the same number of neighbors per cell, "
        f"but they have RNA: {K}, ADT: {K + 3}. Rebuild the connectivity maps from "
        f"neighbors of one k, such as graph.neighbors(ann_index, k={K}), or "
        "integrate the neighbors artifacts with method='wnn', which keeps the "
        "smallest k."
    )
    assert store.artifacts.list(kind="integrated_graph", scope="datastore") == before


def test_snn_integration_of_graphs_with_one_k_still_runs(writable) -> None:
    store, refs = writable

    integrated = store.integration.modalities(
        [refs["rna_graph"], refs["adt_graph"]], method="snn"
    )

    group = artifact_group(store.zw, integrated)
    assert group["edges"].shape == (N_CELLS * K, 2)
    assert int(group.attrs["n_neighbors"]) == K


# Estimates against measured peaks


def _knn_graph(n_cells: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and distances of a kNN graph of random points."""
    from sklearn.neighbors import NearestNeighbors

    values = np.random.default_rng(seed).normal(size=(n_cells, 8))
    distances, indices = (
        NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values)
    )
    return indices[:, 1:].astype(np.uint32), distances[:, 1:].astype(np.float32)


def _stored(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, ...]:
    from cytearc.neighbors.graph import build_connectivity_arrays

    return build_connectivity_arrays(
        indices, distances, local_connectivity=1.0, bandwidth=1.5
    )


def _random_stored(n_cells: int, k: int, seed: int) -> tuple[np.ndarray, ...]:
    """A directed graph whose edges are almost never mutual."""
    rng = np.random.default_rng(seed)
    edges = np.empty((n_cells * k, 2), dtype=np.uint32)
    edges[:, 0] = np.repeat(np.arange(n_cells, dtype=np.uint32), k)
    edges[:, 1] = np.concatenate(
        [rng.choice(n_cells, size=k, replace=False) for _ in range(n_cells)]
    )
    return edges, rng.random(n_cells * k).astype(np.float32)


def _graph_reader(edges: np.ndarray, weights: np.ndarray, n_cells: int):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("graph")
    group.attrs.update({"n_cells": n_cells, "n_neighbors": len(edges) // n_cells})
    # Zarr decodes uncompressed chunks in its event loop. It decodes compressed
    # ones in a thread pool, whose threads can still hold a decoded chunk while
    # the load builds its matrix, so a traced peak would count that chunk only
    # when those threads lag behind.
    group.create_array("edges", data=edges, chunks=edges.shape, compressors=None)
    group.create_array("weights", data=weights, chunks=weights.shape, compressors=None)

    class _Reader:
        _get_graph_ncells_k = _GraphOperationsMixin._get_graph_ncells_k
        _store_to_sparse = _GraphOperationsMixin._store_to_sparse
        _load_graph_artifact = _GraphOperationsMixin._load_graph_artifact
        zw = root

        def _graph_location(self, _graph):
            return "graph"

    return _Reader()


@pytest.mark.slow
@pytest.mark.parametrize("structure", ["knn", "random"])
@pytest.mark.parametrize(
    ("symmetric", "upper_only"), [(False, False), (True, False), (True, True)]
)
def test_graph_load_memory_bounds_the_traced_peak(structure, symmetric, upper_only):
    from tests.test_resident_memory_admission import _traced_peak

    n_cells, k = 20_000, 10
    if structure == "knn":
        edges, weights = _stored(*_knn_graph(n_cells, k, seed=4))
    else:
        edges, weights = _random_stored(n_cells, k, seed=4)
    reader = _graph_reader(edges, weights, n_cells)
    held: list[csr_matrix] = []

    def load() -> None:
        held.append(
            reader._load_graph_artifact(
                None, symmetric=symmetric, upper_only=upper_only, use_k=None
            )
        )

    # Load once outside the trace, so that imports and the coverage tracer's
    # first records of these lines are not measured as the load's memory.
    load()
    held.clear()
    peak = _traced_peak(load)
    (matrix,) = held
    memory = graph_load_memory(
        n_cells, n_cells * k, symmetric=symmetric, upper_only=upper_only
    )
    held_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert peak <= memory.peakBytes <= 2 * peak
    assert held_bytes <= memory.heldBytes
    assert matrix.nnz <= memory.maxEntries


@pytest.mark.slow
def test_connectivity_peak_bytes_bounds_the_allocated_peak() -> None:
    from cytearc.neighbors.graph import build_connectivity_arrays
    from tests.test_resident_memory_admission import (
        _allocated_peak,
        _glibc_allocated,
    )

    allocated = _glibc_allocated()
    if allocated is None:
        pytest.skip("needs glibc's count of allocated bytes")
    for n_cells, k in ((30_000, 5), (10_000, 30)):
        indices, distances = _knn_graph(n_cells, k, seed=6)
        _stored(indices[:50] % 50, distances[:50])

        def convert() -> None:
            build_connectivity_arrays(
                indices.copy(),
                distances.copy(),
                local_connectivity=1.0,
                bandwidth=1.5,
            )

        peak = max(_allocated_peak(allocated, convert) for _ in range(2))
        estimate = connectivity_peak_bytes(n_cells, k)
        assert peak <= estimate <= 2 * peak, (n_cells, k, peak, estimate)


@pytest.mark.slow
@pytest.mark.parametrize("structure", ["knn", "random"])
def test_umap_peak_bytes_bounds_the_traced_peak(structure) -> None:
    from cytearc.embeddings.umap import densmap_distance_graph, fit_transform
    from tests.test_resident_memory_admission import _traced_peak

    n_cells, k = 20_000, 10
    indices, distances = _knn_graph(n_cells, k, seed=8)
    if structure == "knn":
        edges, weights = _stored(indices, distances)
    else:
        edges, weights = _random_stored(n_cells, k, seed=8)
    graph = csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells))
    initial = np.random.default_rng(1).normal(size=(n_cells, 2)).astype(np.float32)

    def layout(densmap: bool) -> None:
        options = {}
        if densmap:
            options = {
                "lambda": 2.0,
                "frac": 0.3,
                "var_shift": 0.1,
                "n_neighbors": k,
                "knn_dists": densmap_distance_graph(indices.copy(), distances.copy()),
            }
        fit_transform(
            graph=graph.tocoo(),
            ini_embed=initial.copy(),
            spread=2.0,
            min_dist=1.0,
            n_epochs=3,
            random_seed=1,
            repulsion_strength=1.0,
            initial_alpha=1.0,
            negative_sample_rate=5.0,
            densmap_kwds=options,
            parallel=False,
            nthreads=1,
            verbose=False,
        )

    layout(False)
    edges_count = n_cells * k
    for densmap in (False, True):
        peak = _traced_peak(lambda densmap=densmap: layout(densmap))
        # The estimate bounds the union by twice the graph's entries.
        estimate = umap_peak_bytes(
            n_cells,
            edges_count,
            2 * edges_count,
            densmap_edges=edges_count if densmap else 0,
        )
        assert peak <= estimate <= 2 * peak, (densmap, peak, estimate)


@pytest.mark.slow
@pytest.mark.parametrize("structure", ["knn", "random"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_native_leiden_peak_bytes_bounds_the_allocated_peak(structure, symmetric):
    from cytearc.clustering.leiden import leiden_membership
    from tests.test_resident_memory_admission import (
        _allocated_peak,
        _glibc_allocated,
    )

    allocated = _glibc_allocated()
    if allocated is None:
        pytest.skip("needs glibc's count of allocated bytes")
    n_cells, k = 30_000, 10
    if structure == "knn":
        edges, weights = _stored(*_knn_graph(n_cells, k, seed=10))
    else:
        edges, weights = _random_stored(n_cells, k, seed=10)
    graph = csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells))
    if symmetric:
        graph = (graph + graph.T) - graph.multiply(graph.T)
    leiden_membership(graph[:300, :300], 1.0, 0, nthreads=2)

    peak = max(
        _allocated_peak(
            allocated, lambda: leiden_membership(graph, 1.0, 4444, nthreads=2)
        )
        for _ in range(2)
    )
    # As clusters.leiden bounds W + W^T: twice the stored edges, which is
    # also the most entries of a symmetrized graph.
    estimate = native_leiden_peak_bytes(n_cells, 2 * n_cells * k)
    # A random graph aggregates slowly, so it holds the three graphs the
    # estimate counts; a kNN graph aggregates into far smaller ones and has
    # fewer entries in W + W^T than the bound.
    slack = 2 if structure == "random" else 3
    assert peak <= estimate <= slack * peak, (peak, estimate)
    assert coo_matrix(graph).nnz <= 2 * n_cells * k
