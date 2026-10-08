from dataclasses import fields
from importlib.util import find_spec
from inspect import signature

from cytearc.datastore.namespaces import (
    ClustersAccessor,
    EmbeddingsAccessor,
    IntegrationAccessor,
    TrajectoryAccessor,
)
from cytearc.datastore.datastore import DataStore
from cytearc.embeddings.sgtsne import run_sgtsne
from cytearc.trajectory.feature_dynamics import validate_pseudotime_regressor
from cytearc.trajectory.results import (
    FateMappingResult,
    PseudotimeAggregationResult,
    PseudotimeMarkerResult,
    PseudotimeScoreResult,
)
from tests.signature_contracts import signature_digest


def test_embedding_and_trajectory_entry_point_signatures_are_stable() -> None:
    methods = {
        "run_sgtsne": run_sgtsne,
        "validate_pseudotime_regressor": validate_pseudotime_regressor,
    }
    assert signature_digest(methods) == (
        "d1af98d226150b81c1f5d8be645627caf270964b97646e735a0a668283fe6f9b"
    )


def test_trajectory_results_describe_artifacts_not_metadata_columns() -> None:
    assert [field.name for field in fields(PseudotimeScoreResult)] == [
        "ref",
        "graph",
        "cell_selection",
        "values",
        "valid",
    ]
    assert [field.name for field in fields(FateMappingResult)] == [
        "ref",
        "graph",
        "pseudotime",
        "sink_labels_artifact",
        "cell_selection",
        "sink_labels",
        "values",
        "valid",
    ]
    assert [field.name for field in fields(PseudotimeMarkerResult)] == [
        "ref",
        "table",
        "assay",
        "cell_selection",
        "feature_selection",
        "pseudotime",
    ]
    assert [field.name for field in fields(PseudotimeAggregationResult)] == [
        "ref",
        "data",
        "feature_indices",
        "feature_clusters",
        "assay",
        "cell_selection",
        "feature_selection",
        "pseudotime",
    ]


def test_artifact_only_producers_drop_metadata_output_arguments() -> None:
    forbidden = {
        "cell_key",
        "output_assay",
        "label",
        "cluster_key",
        "cluster_label",
        "new_col_name",
        "pseudotime_key",
        "sink_key",
    }
    producers = (
        EmbeddingsAccessor.umap,
        EmbeddingsAccessor.tsne,
        ClustersAccessor.leiden,
        ClustersAccessor.paris,
        ClustersAccessor.membership_strength,
        ClustersAccessor.relabel_by_overlap,
        TrajectoryAccessor.pseudotime,
        TrajectoryAccessor.fate,
        TrajectoryAccessor.markers,
        TrajectoryAccessor.aggregation,
    )
    for producer in producers:
        assert forbidden.isdisjoint(signature(producer).parameters)


def test_trajectory_loaders_are_explicit_public_datastore_methods() -> None:
    for method in (
        ClustersAccessor.load_paris,
        TrajectoryAccessor.load_pseudotime,
        TrajectoryAccessor.load_fate,
        TrajectoryAccessor.load_markers,
        TrajectoryAccessor.load_aggregation,
    ):
        parameters = signature(method).parameters
        assert tuple(parameters) == ("self", "ref")


def test_run_consumers_take_explicit_artifacts() -> None:
    silhouette = signature(IntegrationAccessor.compute_graph_silhouette).parameters
    assert tuple(silhouette) == (
        "self",
        "neighbors",
        "clusters",
        "random_seed",
        "sample_size",
    )
    assert "run" in signature(DataStore.to_anndata).parameters


def test_moved_symbols_are_absent_from_old_hybrid_modules() -> None:
    from cytearc.datastore import datastore, graph_datastore
    from cytearc.features import markers

    assert find_spec("cytearc.knn_utils") is None
    retired = {
        markers: {"knn_clustering"},
        datastore: {
            "_scatter_feature_clusters",
            "_validated_pseudotime_regressor",
        },
        graph_datastore: {
            "_make_source_sink_vector",
            "_random_walk_laplacian_transpose",
            "_select_pseudotime_component",
            "_truncated_pba_potential",
            "_validate_source_sink_labels",
            "_validate_source_sink_vector",
        },
    }
    for module, names in retired.items():
        assert names.isdisjoint(vars(module))
