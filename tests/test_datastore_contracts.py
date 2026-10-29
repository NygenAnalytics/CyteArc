import inspect
import pydoc

import pytest

from cytearc.datastore.artifact_accessor import ArtifactAccessor
from cytearc.datastore.base_datastore import BaseDataStore
from cytearc.datastore.datastore import DataStore
from cytearc.datastore.graph_datastore import GraphDataStore
from cytearc.datastore.mapping_datastore import MappingDatastore
from cytearc.datastore.namespaces import (
    ClustersAccessor,
    CompareAccessor,
    EmbeddingsAccessor,
    FeaturesAccessor,
    GraphAccessor,
    ImputationAccessor,
    IntegrationAccessor,
    MappingAccessor,
    MarkersAccessor,
    PerturbationAccessor,
    QcAccessor,
    ReductionAccessor,
    RegulonsAccessor,
    RepertoireAccessor,
    ScoresAccessor,
    TrajectoryAccessor,
)
from tests.signature_contracts import signature_digest


_METHODS = {
    BaseDataStore: {
        "__init__": BaseDataStore.__init__,
        "get_cell_vals": BaseDataStore.get_cell_vals,
        "inspect_artifact": ArtifactAccessor.inspect,
        "lineage": ArtifactAccessor.lineage,
        "list_artifacts": ArtifactAccessor.list,
        "load_artifact": ArtifactAccessor.load,
        "load_cell_values": ArtifactAccessor.load_values,
        "snapshot_cell_selection": BaseDataStore.snapshot_cell_selection,
        "snapshot_cluster_labels": BaseDataStore.snapshot_cluster_labels,
        "summary": BaseDataStore.summary,
    },
    GraphDataStore: {
        "__init__": GraphDataStore.__init__,
        "build_ann_index": GraphAccessor.ann_index,
        "build_connectivity_map": GraphAccessor.connectivity,
        "build_embedding_initialization": EmbeddingsAccessor.initialization,
        "build_mapping_reference": MappingAccessor.build_reference,
        "get_imputed": ImputationAccessor.compute_imputed,
        "get_mapping_reference": MappingAccessor.load_reference,
        "integrate_assays": IntegrationAccessor.modalities,
        "load_diffusion_operator": ImputationAccessor.load_diffusion,
        "load_graph": GraphAccessor.load,
        "query_neighbors": GraphAccessor.neighbors,
        "run_fate_mapping": TrajectoryAccessor.fate,
        "run_leiden_clustering": ClustersAccessor.leiden,
        "run_lsi": ReductionAccessor.lsi,
        "run_custom_reduction": ReductionAccessor.custom,
        "run_harmony": ReductionAccessor.harmony,
        "run_normalization": FeaturesAccessor.normalize,
        "run_paris_clustering": ClustersAccessor.paris,
        "run_pca": ReductionAccessor.pca,
        "run_diffusion_operator": ImputationAccessor.diffusion,
        "run_pseudotime_scoring": TrajectoryAccessor.pseudotime,
        "run_tsne": EmbeddingsAccessor.tsne,
        "run_umap": EmbeddingsAccessor.umap,
    },
    MappingDatastore: {
        "get_label_transfer": MappingAccessor.load_label_transfer,
        "get_mapping_result": MappingAccessor.load_result,
        "get_mapping_score": MappingAccessor.compute_scores,
        "run_label_transfer": MappingAccessor.transfer_labels,
        "run_mapping": MappingAccessor.run,
    },
    DataStore: {
        "__init__": DataStore.__init__,
        "add_grouped_assay": DataStore.add_grouped_assay,
        "add_melded_assay": DataStore.add_melded_assay,
        "auto_filter_cells": QcAccessor.auto_filter,
        "calc_membership_strength": ClustersAccessor.membership_strength,
        "discard_interrupted_assay": DataStore.discard_interrupted_assay,
        "export_markers_to_csv": MarkersAccessor.export_csv,
        "filter_cells": QcAccessor.filter,
        "get_assay": DataStore.get_assay,
        "get_enrichment": ScoresAccessor.load,
        "get_markers": MarkersAccessor.load,
        "make_bulk": CompareAccessor.compute_bulk,
        "select_all_features": FeaturesAccessor.universe,
        "select_hvgs": FeaturesAccessor.hvgs,
        "select_prevalent_peaks": FeaturesAccessor.prevalent_peaks,
        "metric_clisi": IntegrationAccessor.compute_clisi,
        "metric_cluster_separability": IntegrationAccessor.compute_cluster_separability,
        "metric_graph_connectivity": IntegrationAccessor.compute_graph_connectivity,
        "metric_graph_silhouette": IntegrationAccessor.compute_graph_silhouette,
        "metric_ilisi": IntegrationAccessor.compute_ilisi,
        "metric_label_concordance": ClustersAccessor.compute_concordance,
        "metric_proportional_batch_mixing": IntegrationAccessor.compute_batch_mixing,
        "resolve_features": FeaturesAccessor.resolve,
        "run_aucell": ScoresAccessor.aucell,
        "run_cell_cycle_scoring": ScoresAccessor.cell_cycle,
        "run_doublet_detection": QcAccessor.doublets,
        "run_feature_percentage": QcAccessor.feature_percentage,
        "run_hto_demultiplexing": QcAccessor.hto_demultiplexing,
        "run_marker_search": MarkersAccessor.search,
        "run_pseudotime_aggregation": TrajectoryAccessor.aggregation,
        "run_pseudotime_marker_search": TrajectoryAccessor.markers,
        "run_waggr": ScoresAccessor.waggr,
        "select_cells": DataStore.select_cells,
        "select_detected_features": FeaturesAccessor.detected,
        "select_measured_cells": DataStore.select_measured_cells,
        "set_feature_selection": FeaturesAccessor.snapshot,
        "show_zarr_tree": DataStore.show_zarr_tree,
        "smart_label": ClustersAccessor.relabel_by_overlap,
        "to_anndata": DataStore.to_anndata,
        "import_coordinates": ReductionAccessor.import_coordinates,
    },
    ArtifactAccessor: {
        "find": ArtifactAccessor.find,
        "prune_incomplete": ArtifactAccessor.prune_incomplete,
    },
}

# Keep the existing digest keys so moving a method cannot hide a signature change.
_SIGNATURE_DIGESTS = {
    # Base and inherited graph constructors accept DataStore's optional setup values.
    # min_features_per_cell is int | None: None applies 10 on a store's first
    # writable open, and later opens never filter.
    BaseDataStore: "65b8c1277d428920d77b0766fe77bff11158da2c90ec36899b2c6257580da57c",
    # run_leiden_clustering gained the native backend: backend is now
    # Literal["cytearc", "igraph", "leidenalg"] with the default "cytearc".
    # The inherited constructor takes min_features_per_cell: int | None.
    # build_ann_index defaults to ann_efc=200, ann_ef=200, and ann_m=24, and
    # query_neighbors gained the keyword ann_ef: int | None = None.
    GraphDataStore: "a848c56b3f6b98c9a5a90a26af7a86b46294cb313a14734a2eee403188b5ab33",
    MappingDatastore: "a4839ac0372df95f021fd8c383c1ac2aff6796c6b093e05510c71ad7e8449a72",
    # run_hto_demultiplexing gained the keyword-only positive_quantile: float
    # with the default 0.99. auto_filter_cells gained the keyword-only
    # metric_policies: dict[str, dict[str, str]] | None with the default None.
    # to_anndata gained the keyword-only cell_selection, obsm, obs, graph, and
    # dtype, each None by default, reduction.import_coordinates is new, and
    # min_features_per_cell: int | None = None. select_hvgs gained flavor,
    # batch, span, theta, and clip, and its min_cells defaults to None, the
    # flavor's own default. get_markers and export_markers_to_csv take
    # min_score: float | None = None, min_frac_exp: float | None = 0.2,
    # min_auc: float | None = 0.6, max_p_adj: float | None = 0.05, and
    # sort_by: MarkerSortKey = "score"; make_bulk gained the keyword-only
    # remove_empty_columns: bool = True and return_n_cells: bool = False, and
    # returns their tuple shapes. run_doublet_detection gained sample_column,
    # expected_doublet_rate, and threshold, with new defaults for the parent
    # pool, simulation_ratio, heterotypic_fraction, save_k, smoothing_t, and
    # normalize_scores; select_cells gained value.
    DataStore: "10267b25dd0748137875d8d2df092819a063f57dc22abdc7222e3892df1b403e",
    # artifacts.prune_incomplete(*, dry_run: bool = True) -> list[str] was added.
    ArtifactAccessor: "6d4befabbbc91b00684038f44e0b3063905c84437587be0994c26370efa2570c",
}


_ACCESSORS = {
    "artifacts": ArtifactAccessor,
    "clusters": ClustersAccessor,
    "compare": CompareAccessor,
    "embeddings": EmbeddingsAccessor,
    "features": FeaturesAccessor,
    "graph": GraphAccessor,
    "imputation": ImputationAccessor,
    "integration": IntegrationAccessor,
    "mapping": MappingAccessor,
    "markers": MarkersAccessor,
    "perturbation": PerturbationAccessor,
    "qc": QcAccessor,
    "reduction": ReductionAccessor,
    "regulons": RegulonsAccessor,
    "repertoire": RepertoireAccessor,
    "scores": ScoresAccessor,
    "trajectory": TrajectoryAccessor,
}
_NAMESPACE_METHODS = [
    (namespace, accessor, name, method)
    for namespace, accessor in _ACCESSORS.items()
    for name, method in vars(accessor).items()
    if not name.startswith("_") and inspect.isfunction(method)
]


def test_datastore_namespace_bindings_cover_all_moved_methods_and_find():
    # reduction.import_coordinates, artifacts.prune_incomplete, the five
    # ambient RNA methods of qc, and qc.call_cells, qc.load_cell_calls, and
    # qc.barcode_ranks joined the moved methods and find, then the ten
    # lineage methods of trajectory: lineage_markers, drivers,
    # lineage_aggregation, curves, and lineage_tree with their loaders, then
    # perturbation.import_guide_calls and perturbation.load_guide_calls, then
    # perturbation.call_guides and perturbation.load_guide_thresholds, then
    # perturbation.labels and perturbation.load_labels, then
    # perturbation.efficiency and perturbation.load_efficiency, then
    # perturbation.edistance and perturbation.load_edistance, then
    # repertoire.import_receptors and repertoire.load_receptors.
    assert len(_NAMESPACE_METHODS) == 106
    assert {"import_receptors", "load_receptors"} == {
        name
        for namespace, _, name, _ in _NAMESPACE_METHODS
        if namespace == "repertoire"
    }
    assert {
        "import_guide_calls",
        "load_guide_calls",
        "call_guides",
        "load_guide_thresholds",
        "labels",
        "load_labels",
        "efficiency",
        "load_efficiency",
        "edistance",
        "load_edistance",
    } == {
        name
        for namespace, _, name, _ in _NAMESPACE_METHODS
        if namespace == "perturbation"
    }
    assert {
        "lineage_markers",
        "load_lineage_markers",
        "drivers",
        "load_drivers",
        "lineage_aggregation",
        "load_lineage_aggregation",
        "curves",
        "load_curves",
        "lineage_tree",
        "load_lineage_tree",
    } <= {
        name
        for namespace, _, name, _ in _NAMESPACE_METHODS
        if namespace == "trajectory"
    }
    assert {"call_cells", "load_cell_calls", "barcode_ranks"} <= {
        name for namespace, _, name, _ in _NAMESPACE_METHODS if namespace == "qc"
    }


@pytest.mark.parametrize(
    ("namespace", "accessor", "name", "method"),
    _NAMESPACE_METHODS,
    ids=[f"{namespace}.{name}" for namespace, _, name, _ in _NAMESPACE_METHODS],
)
def test_datastore_namespace_binding_preserves_entry_contract(
    namespace, accessor, name, method
):
    entry = getattr(DataStore, f"_{namespace}_{name}")
    assert method.__wrapped__ is entry
    assert method.__annotations__ == entry.__annotations__
    assert method.__doc__ == entry.__doc__
    assert inspect.signature(method) == inspect.signature(entry)

    store = object.__new__(DataStore)
    bound = getattr(getattr(store, namespace), name)
    assert isinstance(bound.__self__, accessor)
    assert bound.__self__._store is store
    assert inspect.signature(bound) == inspect.signature(entry).replace(
        parameters=list(inspect.signature(entry).parameters.values())[1:]
    )


@pytest.mark.parametrize(
    ("namespace", "accessor", "name", "method"),
    _NAMESPACE_METHODS,
    ids=[f"{namespace}.{name}" for namespace, _, name, _ in _NAMESPACE_METHODS],
)
def test_datastore_namespace_methods_show_public_names(
    namespace, accessor, name, method
):
    assert method.__name__ == name
    assert method.__qualname__ == f"{accessor.__qualname__}.{name}"
    assert method.__module__ == accessor.__module__
    bound = getattr(getattr(object.__new__(DataStore), namespace), name)
    assert repr(bound).startswith(f"<bound method {accessor.__qualname__}.{name} ")
    help_lines = pydoc.render_doc(bound, renderer=pydoc.plaintext).splitlines()
    assert help_lines[0] == (
        f"Python Library Documentation: method {name} in module {accessor.__module__}"
    )
    assert help_lines[2].startswith(f"{name}(")


def test_datastore_namespace_dispatch_honors_subclass_overrides():
    class RnaStore(DataStore):
        def _features_resolve(self, assay, features):
            if assay != "RNA":
                raise ValueError("This store requires the RNA assay")
            return super()._features_resolve(assay, features)

    store = object.__new__(RnaStore)
    with pytest.raises(ValueError, match="This store requires the RNA assay"):
        store.features.resolve("ATAC", None)
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        store.features.resolve(assay="RNA", features=None)


def test_datastore_public_method_signatures_are_stable():
    for cls, methods in _METHODS.items():
        assert signature_digest(methods) == _SIGNATURE_DIGESTS[cls]


def test_datastore_public_class_chain_is_stable():
    public_classes = {BaseDataStore, GraphDataStore, MappingDatastore, DataStore}
    assert [cls for cls in DataStore.mro() if cls in public_classes] == [
        DataStore,
        MappingDatastore,
        GraphDataStore,
        BaseDataStore,
    ]
    assert DataStore.__module__ == "cytearc.datastore.datastore"
    assert MappingDatastore.__module__ == "cytearc.datastore.mapping_datastore"
    assert GraphDataStore.__module__ == "cytearc.datastore.graph_datastore"
    assert BaseDataStore.__module__ == "cytearc.datastore.base_datastore"


def test_live_label_transfer_readers_are_removed():
    # Label transfer is a saved artifact: run_label_transfer produces it and
    # get_label_transfer loads it.
    for name in (
        "get_target_classes",
        "get_target_label_evidence",
        "_label_transfer_codes",
        "_reference_label_codes",
        "_iter_label_votes",
    ):
        assert not hasattr(MappingDatastore, name)


def test_stored_graph_path_lookup_is_removed():
    assert not hasattr(GraphDataStore, "lookup_stored_graph")
    assert not hasattr(GraphDataStore, "_lookup_stored_graph")
    assert not hasattr(GraphDataStore, "get_latest_graph_loc")
    assert not hasattr(GraphDataStore, "get_normalized_group_path")
    assert not hasattr(DataStore, "set_hvgs")
    assert not hasattr(DataStore, "mark_hto_identities")


def test_datastore_property_contracts_are_stable():
    for name in ("assay_names", "zw"):
        descriptor = inspect.getattr_static(BaseDataStore, name)
        assert isinstance(descriptor, property)
        assert descriptor.fget is not None
        assert list(inspect.signature(descriptor.fget).parameters) == ["self"]
        assert inspect.getattr_static(DataStore, name) is descriptor


def test_datastore_accessor_namespace_contract_is_stable():
    for name in (
        "pipeline",
        "plots",
        "artifacts",
        "qc",
        "features",
        "reduction",
        "graph",
        "embeddings",
        "integration",
        "clusters",
        "markers",
        "scores",
        "imputation",
        "trajectory",
        "mapping",
        "compare",
        "regulons",
        "perturbation",
        "repertoire",
    ):
        descriptor = inspect.getattr_static(DataStore, name)
        assert isinstance(descriptor, property)
        assert descriptor.fget is not None
        assert list(inspect.signature(descriptor.fget).parameters) == ["self"]
        assert name in DataStore.__dict__
        for cls in (BaseDataStore, GraphDataStore, MappingDatastore):
            assert not hasattr(cls, name)


def test_datastore_static_method_contracts_are_stable():
    static_methods = {
        GraphDataStore: ("_resolve_local_cache_plan",),
        MappingDatastore: (
            "_projection_block_size",
            "_query_batch_design",
        ),
        DataStore: ("_write_marker_slot",),
    }
    for cls, names in static_methods.items():
        assert all(
            isinstance(inspect.getattr_static(cls, name), staticmethod)
            for name in names
        )

    for name in (
        "_same_assay_store",
        "_validate_projection_arrays",
        "_projection_has_provenance",
        "_PROJECTION_SCHEMA_VERSION",
        "_LEGACY_PROJECTION_SCHEMA_VERSIONS",
        "_LEGACY_PROJECTION_ATTRS",
        "_LEGACY_PROJECTION_ARRAYS",
        "_projection_attr",
        "_projection_array_name",
    ):
        assert not hasattr(MappingDatastore, name)


def test_graph_datastore_private_mixin_order_is_stable():
    from cytearc.datastore._operations.clustering import _ClusteringOperationsMixin
    from cytearc.datastore._operations.embeddings import _EmbeddingOperationsMixin
    from cytearc.datastore._operations.graph import _GraphOperationsMixin
    from cytearc.datastore._operations.mapping_reference import (
        _MappingReferenceOperationsMixin,
    )
    from cytearc.datastore._operations.trajectory import _TrajectoryOperationsMixin

    assert GraphDataStore.__bases__ == (
        _EmbeddingOperationsMixin,
        _ClusteringOperationsMixin,
        _TrajectoryOperationsMixin,
        _MappingReferenceOperationsMixin,
        _GraphOperationsMixin,
        BaseDataStore,
    )
    assert not hasattr(GraphDataStore, "make_graph")


def test_mapping_datastore_private_mixin_order_is_stable():
    from cytearc.datastore._operations.mapping import _MappingOperationsMixin

    assert MappingDatastore.__bases__ == (
        _MappingOperationsMixin,
        GraphDataStore,
    )
    assert MappingDatastore.mro()[:3] == [
        MappingDatastore,
        _MappingOperationsMixin,
        GraphDataStore,
    ]


def test_datastore_private_mixin_order_is_stable():
    from cytearc.datastore._operations.ambient import _AmbientOperationsMixin
    from cytearc.datastore._operations.cell_calling import _CellCallingOperationsMixin
    from cytearc.datastore._operations.features import _FeatureOperationsMixin
    from cytearc.datastore._operations.integration_metrics import (
        _IntegrationMetricsOperationsMixin,
    )
    from cytearc.datastore._operations.perturbation import (
        _PerturbationOperationsMixin,
    )
    from cytearc.datastore._operations.presentation import _PresentationOperationsMixin
    from cytearc.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from cytearc.datastore._operations.regulons import _RegulonOperationsMixin
    from cytearc.datastore._operations.repertoire import _RepertoireOperationsMixin
    from cytearc.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
    )

    assert DataStore.__bases__ == (
        _CellCallingOperationsMixin,
        _QualityControlOperationsMixin,
        _AmbientOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
        _RegulonOperationsMixin,
        _PerturbationOperationsMixin,
        _RepertoireOperationsMixin,
        MappingDatastore,
    )
    assert DataStore.mro()[:12] == [
        DataStore,
        _CellCallingOperationsMixin,
        _QualityControlOperationsMixin,
        _AmbientOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
        _RegulonOperationsMixin,
        _PerturbationOperationsMixin,
        _RepertoireOperationsMixin,
        MappingDatastore,
    ]


def test_datastore_facades_only_own_composition_methods():
    def defined_methods(cls: type) -> set[str]:
        return {
            name
            for name, value in cls.__dict__.items()
            if inspect.isfunction(value)
            or isinstance(value, (classmethod, staticmethod))
        }

    assert defined_methods(GraphDataStore) == set()
    assert defined_methods(MappingDatastore) == set()
    assert defined_methods(DataStore) == {
        "__init__",
        "get_assay",
        "_features_resolve",
    }


def test_datastore_operation_mixins_have_unique_method_owners():
    from cytearc.datastore._operations.cell_calling import _CellCallingOperationsMixin
    from cytearc.datastore._operations.clustering import _ClusteringOperationsMixin
    from cytearc.datastore._operations.embeddings import _EmbeddingOperationsMixin
    from cytearc.datastore._operations.features import _FeatureOperationsMixin
    from cytearc.datastore._operations.graph import _GraphOperationsMixin
    from cytearc.datastore._operations.integration_metrics import (
        _IntegrationMetricsOperationsMixin,
    )
    from cytearc.datastore._operations.mapping import _MappingOperationsMixin
    from cytearc.datastore._operations.mapping_reference import (
        _MappingReferenceOperationsMixin,
    )
    from cytearc.datastore._operations.perturbation import (
        _PerturbationOperationsMixin,
    )
    from cytearc.datastore._operations.presentation import _PresentationOperationsMixin
    from cytearc.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from cytearc.datastore._operations.regulons import _RegulonOperationsMixin
    from cytearc.datastore._operations.repertoire import _RepertoireOperationsMixin
    from cytearc.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
        _TrajectoryOperationsMixin,
    )

    mixins = (
        _EmbeddingOperationsMixin,
        _ClusteringOperationsMixin,
        _TrajectoryOperationsMixin,
        _MappingReferenceOperationsMixin,
        _GraphOperationsMixin,
        _MappingOperationsMixin,
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
        _RegulonOperationsMixin,
        _CellCallingOperationsMixin,
        _PerturbationOperationsMixin,
        _RepertoireOperationsMixin,
    )
    owners: dict[str, list[str]] = {}
    for mixin in mixins:
        for name, value in mixin.__dict__.items():
            if inspect.isfunction(value) or isinstance(
                value, (classmethod, staticmethod)
            ):
                owners.setdefault(name, []).append(mixin.__name__)
    assert {name: owner for name, owner in owners.items() if len(owner) > 1} == {}


def test_feature_selection_and_pseudotime_methods_have_domain_owners():
    from cytearc.datastore._operations.features import _FeatureOperationsMixin
    from cytearc.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from cytearc.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
        _TrajectoryOperationsMixin,
    )

    assert "_features_hvgs" in _FeatureOperationsMixin.__dict__
    assert "_scores_load" in _FeatureOperationsMixin.__dict__
    assert "_scores_aucell" in _FeatureOperationsMixin.__dict__
    assert "_scores_waggr" in _FeatureOperationsMixin.__dict__
    assert "_features_hvgs" not in _QualityControlOperationsMixin.__dict__
    assert "_features_prevalent_peaks" in _QualityControlOperationsMixin.__dict__
    assert "_trajectory_fate" in _TrajectoryOperationsMixin.__dict__
    assert "_trajectory_fate" not in _TrajectoryFeatureOperationsMixin.__dict__
    assert "_trajectory_markers" in (_TrajectoryFeatureOperationsMixin.__dict__)
    assert "_trajectory_aggregation" in _TrajectoryFeatureOperationsMixin.__dict__
    assert "_trajectory_markers" not in _FeatureOperationsMixin.__dict__
    assert "_trajectory_aggregation" not in _FeatureOperationsMixin.__dict__
    assert not hasattr(GraphDataStore, "_trajectory_markers")
    assert not hasattr(GraphDataStore, "_trajectory_aggregation")
    assert not hasattr(MappingDatastore, "_trajectory_markers")
    assert not hasattr(MappingDatastore, "_trajectory_aggregation")
