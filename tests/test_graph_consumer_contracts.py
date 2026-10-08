import hashlib
import inspect
import shutil
from dataclasses import fields

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cytearc.datastore.namespaces import (
    ClustersAccessor,
    EmbeddingsAccessor,
    GraphAccessor,
    ImputationAccessor,
    IntegrationAccessor,
    QcAccessor,
    TrajectoryAccessor,
)
from cytearc.datastore.datastore import DataStore
from cytearc.embeddings.imported import write_imported_coordinates
from cytearc.metadata.arguments import (
    DoubletScoreArguments,
    FateMappingArguments,
    LeidenArguments,
    MembershipStrengthArguments,
    PseudotimeScoringArguments,
    TsneArguments,
    UmapArguments,
)
from cytearc.storage.artifacts import ArtifactRef, artifact_path, fingerprint_array
from cytearc.writers import SparseToZarr


@pytest.fixture(scope="module")
def zero_weight_template(tmp_path_factory):
    """A store whose RNA and ADT graphs both hold 54 zero-weight edges.

    The cells form two groups of nine far apart, so with eleven neighbors each
    cell reaches three cells of the other group at weight zero.
    """
    path = tmp_path_factory.mktemp("zero_weight") / "cells.zarr"
    cell_ids = np.array([f"cell_{i}" for i in range(18)])
    counts = np.random.default_rng(31).integers(10, 100, size=(18, 6), dtype=np.uint32)
    SparseToZarr(
        csr_matrix(counts),
        str(path),
        cell_ids=cell_ids,
        feature_ids=[f"gene_{i}" for i in range(6)],
        nthreads=1,
    ).dump()
    shutil.copytree(path / "RNA", path / "ADT")
    store = DataStore(
        str(path),
        default_assay="RNA",
        assay_types={"RNA": "RNA", "ADT": "ADT"},
        min_features_per_cell=0,
        nthreads=1,
    )
    cells = store.snapshot_cell_selection("I")
    positions = np.r_[np.arange(9) * 0.001, 100.0 + np.arange(9) * 0.001]
    coordinates = np.column_stack([positions, np.zeros(18)]).astype(np.float32)
    graphs = []
    for assay in ("RNA", "ADT"):
        imported = write_imported_coordinates(
            store.zw,
            assay=assay,
            dimreduc_key="pca",
            role="pca",
            coordinates=coordinates,
            source_digest=hashlib.sha256(coordinates.tobytes()).digest(),
            payload_fingerprints={"data": fingerprint_array(coordinates)},
            source_cell_ids=cell_ids,
            cell_selection=cells,
        )
        neighbors = store.graph.neighbors(store.graph.ann_index(imported), k=11)
        graphs.append(store.graph.connectivity(neighbors))
    return path, graphs


@pytest.fixture
def zero_weight_graphs(zero_weight_template, tmp_path):
    """A private writable copy of the zero-weight template."""
    template, graphs = zero_weight_template
    path = tmp_path / "cells.zarr"
    shutil.copytree(template, path)
    store = DataStore(
        str(path),
        default_assay="RNA",
        assay_types={"RNA": "RNA", "ADT": "ADT"},
        min_features_per_cell=0,
        nthreads=1,
    )
    return store, graphs


def test_zero_weight_neighbors_support_membership_and_snn(zero_weight_graphs):
    store, graphs = zero_weight_graphs
    graph = graphs[0]
    group = store.artifacts.load(graph)
    assert group["edges"].shape == (18 * 11, 2)
    assert np.count_nonzero(group["weights"][:] == 0) == 54
    np.testing.assert_array_equal(np.diff(store.graph.load(graph).indptr), 11)
    truncated = store.graph.load(graph, use_k=9)
    np.testing.assert_array_equal(np.diff(truncated.indptr), 9)
    assert np.count_nonzero(truncated.data == 0) == 18

    store.cells.insert("annotation", np.repeat("all", 18))
    columns = set(store.cells.columns)
    clusters = store.clusters.leiden(graph)
    labels = store.artifacts.load(clusters)["values"][:]
    assert np.unique(labels[:9]).size == np.unique(labels[9:]).size == 1
    assert labels[0] != labels[9]
    membership = store.clusters.membership_strength(clusters, graph)
    np.testing.assert_allclose(store.artifacts.load(membership)["values"][:], 0.727)
    assert store.clusters.membership_strength(clusters, graph) == membership
    assert store.integration.compute_graph_connectivity("annotation", graph) == 0.5
    assert set(store.cells.columns) == columns

    integrated = store.integration.modalities(graphs, method="snn")
    assert store.artifacts.inspect(integrated).complete
    np.testing.assert_array_equal(np.diff(store.graph.load(integrated).indptr), 11)
    assert store.integration.compute_graph_connectivity("annotation", integrated) == 0.5
    membership = store.clusters.membership_strength(clusters, integrated)
    np.testing.assert_allclose(store.artifacts.load(membership)["values"][:], 0.727)


def test_connectivity_rebuilds_compact_cached_graph(zero_weight_graphs):
    store, graphs = zero_weight_graphs
    original = graphs[0]
    neighbors = ArtifactRef.from_dict(
        store.artifacts.inspect(original).inputs["neighbors"]
    )
    group = store.zw[artifact_path(original)]
    edges, weights = group["edges"][:], group["weights"][:]
    positive = weights > 0
    group["edges"].resize((int(positive.sum()), 2))
    group["weights"].resize((int(positive.sum()),))
    group["edges"][:] = edges[positive]
    group["weights"][:] = weights[positive]

    rebuilt = store.graph.connectivity(neighbors)

    assert rebuilt != original
    assert store.artifacts.load(original)["edges"].shape == (144, 2)
    np.testing.assert_array_equal(store.artifacts.load(rebuilt)["edges"][:], edges)
    np.testing.assert_array_equal(store.artifacts.load(rebuilt)["weights"][:], weights)
    assert store.graph.connectivity(neighbors) == rebuilt


@pytest.mark.parametrize("failure", ["missing_edge", "source_order"])
def test_membership_rejects_invalid_stored_edge_layout(zero_weight_graphs, failure):
    store, graphs = zero_weight_graphs
    graph = graphs[0]
    clusters = store.clusters.leiden(graph)
    edges = store.zw[artifact_path(graph)]["edges"]
    if failure == "missing_edge":
        edges.resize((edges.shape[0] - 1, 2))
        message = "stored cell and k dimensions"
    else:
        edges[0, 0] = 1
        message = "cell-major order"

    with pytest.raises(ValueError, match=message):
        store.clusters.membership_strength(clusters, graph)


def _parameter_names(method: object) -> list[str]:
    return list(inspect.signature(method).parameters)


def test_frozen_imputation_and_membership_signatures() -> None:
    assert not hasattr(DataStore, "get_diffusion_operator")
    assert _parameter_names(ImputationAccessor.compute_imputed) == [
        "self",
        "feature_name",
        "diffusion",
        "from_assay",
    ]
    diffusion_runner = inspect.signature(ImputationAccessor.diffusion).parameters
    assert list(diffusion_runner) == ["self", "graph", "t", "invalidate_cache"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in list(diffusion_runner.values())[2:]
    )
    assert _parameter_names(ImputationAccessor.load_diffusion) == [
        "self",
        "diffusion",
    ]
    membership = inspect.signature(ClustersAccessor.membership_strength).parameters
    assert list(membership) == [
        "self",
        "clusters",
        "graph",
        "invalidate_cache",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in list(membership.values())[3:]
    )


def test_doublet_graph_is_explicit_without_feat_key() -> None:
    parameters = inspect.signature(QcAccessor.doublets).parameters
    assert list(parameters)[:5] == [
        "self",
        "clusters",
        "graph",
        "from_assay",
        "cluster_sample_fraction",
    ]
    assert "feat_key" not in parameters
    assert parameters["graph"].default is inspect.Parameter.empty


def test_graph_and_neighbor_consumers_have_no_path_selectors() -> None:
    graph_methods = (
        GraphAccessor.load,
        EmbeddingsAccessor.umap,
        EmbeddingsAccessor.tsne,
        ClustersAccessor.leiden,
        ClustersAccessor.paris,
        ImputationAccessor.diffusion,
        TrajectoryAccessor.pseudotime,
        IntegrationAccessor.compute_graph_connectivity,
    )
    for method in graph_methods:
        names = _parameter_names(method)
        assert "graph" in names
        assert "feat_key" not in names
        assert "integrated_graph" not in names
        assert "graph_loc" not in names

    fate_names = _parameter_names(TrajectoryAccessor.fate)
    assert "pseudotime" in fate_names
    assert "sink_labels" in fate_names
    assert "graph" not in fate_names

    neighbor_methods = (
        IntegrationAccessor.compute_ilisi,
        IntegrationAccessor.compute_clisi,
        IntegrationAccessor.compute_graph_silhouette,
        IntegrationAccessor.compute_batch_mixing,
    )
    for method in neighbor_methods:
        names = _parameter_names(method)
        assert "neighbors" in names
        assert "use_latest_knn" not in names
        assert "knn_loc" not in names


def test_graph_consumer_argument_records_have_no_feature_or_path_routes() -> None:
    models = (
        UmapArguments,
        TsneArguments,
        LeidenArguments,
        DoubletScoreArguments,
        PseudotimeScoringArguments,
        FateMappingArguments,
        MembershipStrengthArguments,
    )
    for model in models:
        names = {field.name for field in fields(model)}
        assert "feat_key" not in names
        assert "integrated_graph" not in names
        assert "graph_loc" not in names


def test_removed_public_path_locators_are_absent() -> None:
    assert not hasattr(DataStore, "get_latest_graph_loc")
    assert not hasattr(DataStore, "get_normalized_group_path")
