from collections.abc import Callable
from functools import wraps
from types import FunctionType
from typing import Any, Concatenate, cast

from ._operations.clustering import _ClusteringOperationsMixin
from ._operations.embeddings import _EmbeddingOperationsMixin
from ._operations.features import _FeatureOperationsMixin
from ._operations.graph import _GraphOperationsMixin
from ._operations.integration_metrics import _IntegrationMetricsOperationsMixin
from ._operations.mapping import _MappingOperationsMixin
from ._operations.mapping_reference import _MappingReferenceOperationsMixin
from ._operations.presentation import _PresentationOperationsMixin
from ._operations.quality_control import _QualityControlOperationsMixin
from ._operations.trajectory import _TrajectoryFeatureOperationsMixin
from ._operations.trajectory import _TrajectoryOperationsMixin
from .datastore import DataStore

__all__ = [
    "ClustersAccessor",
    "CompareAccessor",
    "EmbeddingsAccessor",
    "FeaturesAccessor",
    "GraphAccessor",
    "ImputationAccessor",
    "IntegrationAccessor",
    "MappingAccessor",
    "MarkersAccessor",
    "QcAccessor",
    "ReductionAccessor",
    "ScoresAccessor",
    "TrajectoryAccessor",
]


def _bind[**P, R](
    entry: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    @wraps(entry)
    def call(self: Any, /, *args: P.args, **kwargs: P.kwargs) -> R:
        method = cast(Callable[P, R], getattr(self._store, entry.__name__))
        return method(*args, **kwargs)

    return call


class _Accessor:
    __slots__ = ("_store",)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for name, method in vars(cls).items():
            if not name.startswith("_") and isinstance(method, FunctionType):
                method.__name__ = name
                method.__qualname__ = f"{cls.__qualname__}.{name}"
                method.__module__ = cls.__module__

    def __init__(self, store: DataStore) -> None:
        self._store = store


class ClustersAccessor(_Accessor):
    __slots__ = ()

    leiden = _bind(_ClusteringOperationsMixin._clusters_leiden)
    paris = _bind(_ClusteringOperationsMixin._clusters_paris)
    load_paris = _bind(_ClusteringOperationsMixin._clusters_load_paris)
    membership_strength = _bind(
        _PresentationOperationsMixin._clusters_membership_strength
    )
    relabel_by_overlap = _bind(
        _PresentationOperationsMixin._clusters_relabel_by_overlap
    )
    compute_concordance = _bind(
        _IntegrationMetricsOperationsMixin._clusters_compute_concordance
    )


class CompareAccessor(_Accessor):
    __slots__ = ()

    compute_bulk = _bind(_FeatureOperationsMixin._compare_compute_bulk)
    test = _bind(_FeatureOperationsMixin._compare_test)
    load_tests = _bind(_FeatureOperationsMixin._compare_load_tests)


class EmbeddingsAccessor(_Accessor):
    __slots__ = ()

    initialization = _bind(_GraphOperationsMixin._embeddings_initialization)
    umap = _bind(_EmbeddingOperationsMixin._embeddings_umap)
    tsne = _bind(_EmbeddingOperationsMixin._embeddings_tsne)


class FeaturesAccessor(_Accessor):
    __slots__ = ()

    universe = _bind(_FeatureOperationsMixin._features_universe)
    hvgs = _bind(_FeatureOperationsMixin._features_hvgs)
    detected = _bind(_FeatureOperationsMixin._features_detected)
    prevalent_peaks = _bind(_QualityControlOperationsMixin._features_prevalent_peaks)
    snapshot = _bind(_FeatureOperationsMixin._features_snapshot)
    resolve = _bind(DataStore._features_resolve)
    normalize = _bind(_GraphOperationsMixin._features_normalize)


class GraphAccessor(_Accessor):
    __slots__ = ()

    ann_index = _bind(_GraphOperationsMixin._graph_ann_index)
    neighbors = _bind(_GraphOperationsMixin._graph_neighbors)
    connectivity = _bind(_GraphOperationsMixin._graph_connectivity)
    load = _bind(_GraphOperationsMixin._graph_load)


class ImputationAccessor(_Accessor):
    __slots__ = ()

    diffusion = _bind(_TrajectoryOperationsMixin._imputation_diffusion)
    load_diffusion = _bind(_TrajectoryOperationsMixin._imputation_load_diffusion)
    compute_imputed = _bind(_TrajectoryOperationsMixin._imputation_compute_imputed)


class IntegrationAccessor(_Accessor):
    __slots__ = ()

    modalities = _bind(_GraphOperationsMixin._integration_modalities)
    compute_ilisi = _bind(_IntegrationMetricsOperationsMixin._integration_compute_ilisi)
    compute_clisi = _bind(_IntegrationMetricsOperationsMixin._integration_compute_clisi)
    compute_graph_connectivity = _bind(
        _IntegrationMetricsOperationsMixin._integration_compute_graph_connectivity
    )
    compute_graph_silhouette = _bind(
        _IntegrationMetricsOperationsMixin._integration_compute_graph_silhouette
    )
    compute_cluster_separability = _bind(
        _IntegrationMetricsOperationsMixin._integration_compute_cluster_separability
    )
    compute_batch_mixing = _bind(
        _IntegrationMetricsOperationsMixin._integration_compute_batch_mixing
    )


class MappingAccessor(_Accessor):
    __slots__ = ()

    build_reference = _bind(_MappingReferenceOperationsMixin._mapping_build_reference)
    load_reference = _bind(_MappingReferenceOperationsMixin._mapping_load_reference)
    run = _bind(_MappingOperationsMixin._mapping_run)
    load_result = _bind(_MappingOperationsMixin._mapping_load_result)
    compute_scores = _bind(_MappingOperationsMixin._mapping_compute_scores)
    transfer_labels = _bind(_MappingOperationsMixin._mapping_transfer_labels)
    load_label_transfer = _bind(_MappingOperationsMixin._mapping_load_label_transfer)


class MarkersAccessor(_Accessor):
    __slots__ = ()

    search = _bind(_FeatureOperationsMixin._markers_search)
    load = _bind(_FeatureOperationsMixin._markers_load)
    export_csv = _bind(_FeatureOperationsMixin._markers_export_csv)


class QcAccessor(_Accessor):
    __slots__ = ()

    filter = _bind(_QualityControlOperationsMixin._qc_filter)
    auto_filter = _bind(_QualityControlOperationsMixin._qc_auto_filter)
    feature_percentage = _bind(_QualityControlOperationsMixin._qc_feature_percentage)
    doublets = _bind(_QualityControlOperationsMixin._qc_doublets)
    hto_demultiplexing = _bind(_QualityControlOperationsMixin._qc_hto_demultiplexing)


class ReductionAccessor(_Accessor):
    __slots__ = ()

    pca = _bind(_GraphOperationsMixin._reduction_pca)
    lsi = _bind(_GraphOperationsMixin._reduction_lsi)
    custom = _bind(_GraphOperationsMixin._reduction_custom)
    harmony = _bind(_GraphOperationsMixin._reduction_harmony)


class ScoresAccessor(_Accessor):
    __slots__ = ()

    aucell = _bind(_FeatureOperationsMixin._scores_aucell)
    waggr = _bind(_FeatureOperationsMixin._scores_waggr)
    load = _bind(_FeatureOperationsMixin._scores_load)
    cell_cycle = _bind(_QualityControlOperationsMixin._scores_cell_cycle)


class TrajectoryAccessor(_Accessor):
    __slots__ = ()

    pseudotime = _bind(_TrajectoryOperationsMixin._trajectory_pseudotime)
    load_pseudotime = _bind(_TrajectoryOperationsMixin._trajectory_load_pseudotime)
    fate = _bind(_TrajectoryOperationsMixin._trajectory_fate)
    load_fate = _bind(_TrajectoryOperationsMixin._trajectory_load_fate)
    markers = _bind(_TrajectoryFeatureOperationsMixin._trajectory_markers)
    load_markers = _bind(_TrajectoryFeatureOperationsMixin._trajectory_load_markers)
    aggregation = _bind(_TrajectoryFeatureOperationsMixin._trajectory_aggregation)
    load_aggregation = _bind(
        _TrajectoryFeatureOperationsMixin._trajectory_load_aggregation
    )
