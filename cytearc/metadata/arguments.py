from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from ..graph.arguments import (
    OperationArguments,
    artifact_input,
    execution,
    parameter,
)
from ..storage.artifacts import ArtifactRef


@dataclass(frozen=True, slots=True)
class UmapArguments(OperationArguments):
    operation: ClassVar[str] = "run_umap"
    artifact_kind: ClassVar[str] = "embedding"

    graph: ArtifactRef = artifact_input()
    initialization: Any = artifact_input()
    symmetric_graph: bool = parameter()
    graph_upper_only: bool = parameter()
    umap_dims: int = parameter()
    spread: float = parameter()
    min_dist: float = parameter()
    n_epochs: int = parameter()
    repulsion_strength: float = parameter()
    initial_alpha: float = parameter()
    negative_sample_rate: float = parameter()
    use_density_map: bool = parameter()
    dens_lambda: float = parameter()
    dens_frac: float = parameter()
    dens_var_shift: float = parameter()
    random_seed: int = parameter()
    # A parallel layout on several threads is not reproducible at any thread
    # count, so only the flag identifies an embedding. The requested thread
    # count and the Numba threads that the layout runs on depend on the
    # machine.
    parallel: bool = parameter()
    nthreads: int = execution()
    layout_threads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class TsneArguments(OperationArguments):
    operation: ClassVar[str] = "run_tsne"
    artifact_kind: ClassVar[str] = "embedding"

    graph: ArtifactRef = artifact_input()
    initialization: Any = artifact_input()
    symmetric_graph: bool = parameter()
    graph_upper_only: bool = parameter()
    tsne_dims: int = parameter()
    lambda_scale: float = parameter()
    max_iter: int = parameter()
    early_iter: int = parameter()
    alpha: int = parameter()
    box_h: float = parameter()
    verbose: bool = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LeidenArguments(OperationArguments):
    operation: ClassVar[str] = "run_leiden_clustering"
    artifact_kind: ClassVar[str] = "cluster_labels"

    graph: ArtifactRef = artifact_input()
    resolution: float = parameter()
    backend: Literal["cytearc", "igraph", "leidenalg"] = parameter()
    edge_weighting: Literal["graph"] = parameter()
    symmetric_graph: bool = parameter()
    graph_upper_only: bool = parameter()
    random_seed: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class DoubletScoreArguments(OperationArguments):
    operation: ClassVar[str] = "run_doublet_detection"
    artifact_kind: ClassVar[str] = "doublet_score"

    clusters: ArtifactRef = artifact_input()
    connectivity_map: ArtifactRef = artifact_input()
    neighbors: ArtifactRef = artifact_input()
    # The fingerprint of each selected cell's sample, or None for one sample.
    sample_assignments_fingerprint: str | None = artifact_input()
    cluster_sample_fraction: float = parameter()
    max_cells_per_cluster: int | None = parameter()
    sample_columns: tuple[str, ...] = parameter()
    simulation_ratio: float = parameter()
    heterotypic_fraction: float = parameter()
    # The neighbors used: the request, at most the largest sample's cells - 1.
    save_k: int = parameter()
    expected_doublet_rate: float = parameter()
    threshold: float | None = parameter()
    smoothing_t: int = parameter()
    normalize_scores: bool = parameter()
    random_seed: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class CellCycleArguments(OperationArguments):
    operation: ClassVar[str] = "run_cell_cycle_scoring"
    artifact_kind: ClassVar[str] = "cell_cycle"

    feature_summary: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    s_gene_indices: tuple[int, ...] = parameter()
    g2m_gene_indices: tuple[int, ...] = parameter()
    control_size: int = parameter()
    log_transform: bool = parameter()
    n_bins: int = parameter()
    rand_seed: int = parameter()
    invalidate_cache: bool = execution()
    # Genes matched by another HGNC symbol, mapped to the feature names they
    # matched. Recorded only when a gene needed one, so scores that matched
    # every gene by its listed name keep their identities.
    gene_aliases: dict[str, str] | None = parameter(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class MarkerTableArguments(OperationArguments):
    operation: ClassVar[str] = "run_marker_search"
    artifact_kind: ClassVar[str] = "marker_table"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    clusters: ArtifactRef = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    method: str = parameter()
    alternative: str = parameter()
    tie_correction: bool = parameter()
    continuity_correction: bool = parameter()
    adjustment_method: str = parameter()
    adjustment_scope: str = parameter()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PseudotimeScoringArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_scoring"
    artifact_kind: ClassVar[str] = "pseudotime"

    connectivity_map: ArtifactRef = artifact_input()
    source_sink: ArtifactRef | np.ndarray = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    n_singular_vals: int = parameter()
    sources: tuple[Any, ...] = parameter()
    sinks: tuple[Any, ...] = parameter()
    min_max_norm_ptime: bool = parameter()
    random_seed: int = parameter()
    component_policy: str = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class FateMappingArguments(OperationArguments):
    operation: ClassVar[str] = "run_fate_mapping"
    artifact_kind: ClassVar[str] = "fate_map"

    connectivity_map: ArtifactRef = artifact_input()
    pseudotime: ArtifactRef = artifact_input()
    sink_labels: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    sinks: tuple[Any, ...] = parameter()
    beta: float = parameter()
    solver_tol: float = parameter()
    max_iterations: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PseudotimeMarkerArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_marker_search"
    artifact_kind: ClassVar[str] = "pseudotime_markers"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    pseudotime: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    ordered_feature_ids_fingerprint: str = artifact_input()
    ordered_feature_names_fingerprint: str = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    association_method: str = parameter()
    p_value_method: str = parameter()
    adjustment_method: str = parameter()
    adjustment_scope: str = parameter()
    min_cells: int = parameter()
    gene_batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PseudotimeAggregationArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_aggregation"
    artifact_kind: ClassVar[str] = "pseudotime_aggregation"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    pseudotime: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    ordered_feature_ids_fingerprint: str = artifact_input()
    ordered_feature_names_fingerprint: str = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    min_exp: float = parameter()
    window_size: int = parameter()
    chunk_size: int = parameter()
    smoothen: bool = parameter()
    z_scale: bool = parameter()
    n_neighbours: int = parameter()
    n_clusters: int = parameter()
    ann_params: dict[str, Any] = parameter()
    nan_cluster_value: int = parameter()
    batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LineageMarkerArguments(OperationArguments):
    operation: ClassVar[str] = "run_lineage_marker_search"
    artifact_kind: ClassVar[str] = "lineage_markers"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    fate: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    ordered_feature_ids_fingerprint: str = artifact_input()
    ordered_feature_names_fingerprint: str = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    lineages: tuple[Any, ...] = parameter()
    association_method: str = parameter()
    p_value_method: str = parameter()
    adjustment_method: str = parameter()
    adjustment_scope: str = parameter()
    min_cells: float = parameter()
    gene_batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LineageDriverArguments(OperationArguments):
    operation: ClassVar[str] = "run_lineage_driver_search"
    artifact_kind: ClassVar[str] = "lineage_drivers"

    cell_selection: ArtifactRef = artifact_input()
    driver_cells: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    fate: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    ordered_feature_ids_fingerprint: str = artifact_input()
    ordered_feature_names_fingerprint: str = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    lineages: tuple[Any, ...] = parameter()
    include_sinks: bool = parameter()
    min_cells: int = parameter()
    confidence_level: float = parameter()
    association_method: str = parameter()
    p_value_method: str = parameter()
    adjustment_method: str = parameter()
    adjustment_scope: str = parameter()
    gene_batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LineageAggregationArguments(OperationArguments):
    operation: ClassVar[str] = "run_lineage_aggregation"
    artifact_kind: ClassVar[str] = "lineage_aggregation"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    fate: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    ordered_feature_ids_fingerprint: str = artifact_input()
    ordered_feature_names_fingerprint: str = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    lineages: tuple[Any, ...] = parameter()
    n_bins: int = parameter()
    smoothing: float = parameter()
    min_exp: float = parameter()
    z_scale: bool = parameter()
    n_neighbours: int = parameter()
    n_clusters: int = parameter()
    ann_params: dict[str, Any] = parameter()
    nan_cluster_value: int = parameter()
    batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LineageCurveArguments(OperationArguments):
    operation: ClassVar[str] = "fit_lineage_curves"
    artifact_kind: ClassVar[str] = "lineage_curves"

    fate: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    lineages: tuple[Any, ...] = parameter()
    n_points: int = parameter()
    df: float = parameter()
    fit_bins: int = parameter()
    spline_basis: int = parameter()
    batch_size: int | None = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LineageTreeArguments(OperationArguments):
    operation: ClassVar[str] = "build_lineage_tree"
    artifact_kind: ClassVar[str] = "lineage_tree"

    labels: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    root: Any = parameter()
    ends: tuple[Any, ...] = parameter()
    batch_size: int | None = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PrevalentPeakArguments(OperationArguments):
    operation: ClassVar[str] = "select_prevalent_peaks"
    artifact_kind: ClassVar[str] = "feature_selection"

    feature_summary: ArtifactRef = artifact_input()
    top_n: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class WaggrArguments(OperationArguments):
    operation: ClassVar[str] = "run_waggr"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    network_digest: str = artifact_input()
    mode: str = parameter()
    tmin: int = parameter()
    log_transform: bool = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class AucellArguments(OperationArguments):
    operation: ClassVar[str] = "run_aucell"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    network_digest: str = artifact_input()
    tmin: int = parameter()
    n_up: int = parameter()
    tie_seed: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class ChromvarArguments(OperationArguments):
    operation: ClassVar[str] = "run_chromvar"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    network_digest: str = artifact_input()
    gc_digest: str = artifact_input()
    tmin: int = parameter()
    n_backgrounds: int = parameter()
    background_seed: int = parameter()
    kernel_width: float = parameter()
    bins: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class GrnArguments(OperationArguments):
    operation: ClassVar[str] = "run_grn_inference"
    artifact_kind: ClassVar[str] = "grn_adjacencies"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    regulators_digest: str = artifact_input()
    method: str = parameter()
    values: str = parameter()
    learning_rate: float = parameter()
    max_stages: int = parameter()
    max_features: float = parameter()
    subsample: float = parameter()
    early_stop_window: int = parameter()
    early_stopping: str = parameter()
    random_seed: int = parameter()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class HtoIdentityArguments(OperationArguments):
    operation: ClassVar[str] = "run_hto_demultiplexing"
    artifact_kind: ClassVar[str] = "hto_identity"

    cell_selection: ArtifactRef = artifact_input()
    feature_ids_fingerprint: str = artifact_input()
    method: dict[str, object] = parameter()
    random_seed: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class CellCallingArguments(OperationArguments):
    operation: ClassVar[str] = "run_cell_calling"
    artifact_kind: ClassVar[str] = "cell_calling"

    cell_selection: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    lower: int = parameter()
    # None records the knee of the barcode-rank curve and an estimated alpha.
    retain: int | None = parameter()
    n_iter: int = parameter()
    alpha: float | None = parameter()
    random_seed: int = parameter()
    # The p-values do not depend on the thread count.
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class GuideCallImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_guide_calls"
    artifact_kind: ClassVar[str] = "guide_calls"

    cell_selection: ArtifactRef = artifact_input()
    # The guide counts give each cell's top guide and guide UMIs.
    dataset_fingerprint: str = artifact_input()
    guide_ids_fingerprint: str = artifact_input()
    # The SHA-256 of the call file's bytes; its path is not part of the identity.
    file_sha256: str = artifact_input()
    barcode_prefix: str | None = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class GuideThresholdArguments(OperationArguments):
    operation: ClassVar[str] = "fit_guide_thresholds"
    artifact_kind: ClassVar[str] = "guide_thresholds"

    cell_selection: ArtifactRef = artifact_input()
    # The guide counts give each guide's histogram in each sample.
    dataset_fingerprint: str = artifact_input()
    guide_ids_fingerprint: str = artifact_input()
    # The fingerprint of each selected cell's sample, or None for one sample.
    sample_assignments_fingerprint: str | None = artifact_input()
    method: str = parameter()
    sample_columns: tuple[str, ...] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class GuideCallArguments(OperationArguments):
    operation: ClassVar[str] = "call_guides"
    artifact_kind: ClassVar[str] = "guide_calls"

    cell_selection: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    guide_ids_fingerprint: str = artifact_input()
    # The fitted thresholds, or None for a method that fits none. The floor
    # and fraction belong to the calls, so changing them reuses the fits.
    guide_thresholds: ArtifactRef | None = artifact_input()
    method: str = parameter()
    min_umi: int = parameter()
    invalidate_cache: bool = execution()
    fraction_threshold: float | None = parameter(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class PerturbationLabelArguments(OperationArguments):
    operation: ClassVar[str] = "label_perturbations"
    artifact_kind: ClassVar[str] = "perturbation_labels"

    guide_calls: ArtifactRef = artifact_input()
    # The target of every guide, in assay order, from target_column and targets.
    targets_fingerprint: str = artifact_input()
    controls: tuple[str, ...] = parameter()
    policy: str = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PerturbationEfficiencyArguments(OperationArguments):
    operation: ClassVar[str] = "measure_perturbation_efficiency"
    artifact_kind: ClassVar[str] = "perturbation_efficiency"

    labels: ArtifactRef = artifact_input()
    # The expression counts that the means and tests read.
    dataset_fingerprint: str = artifact_input()
    # Each row with the feature of its target gene, from id_column or the names.
    target_features_fingerprint: str = artifact_input()
    by: str = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PerturbationSignatureArguments(OperationArguments):
    operation: ClassVar[str] = "compute_perturbation_signature"
    artifact_kind: ClassVar[str] = "perturbation_signature"

    # The labels name the control cells; the coordinates give the cells and
    # their neighbors; the features are the signature's columns.
    labels: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    features: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    # The fingerprint of each cell's sample, or None for one sample.
    sample_assignments_fingerprint: str | None = artifact_input()
    dims: int = parameter()
    n_neighbors: int = parameter()
    sample_columns: tuple[str, ...] = parameter()
    scale_factor: float = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class MixscapeArguments(OperationArguments):
    operation: ClassVar[str] = "run_mixscape"
    artifact_kind: ClassVar[str] = "mixscape_classes"

    # The signature records the labels, the coordinates and the features.
    signature: ArtifactRef = artifact_input()
    # The counts give the DE genes and the signatures of DE genes outside the
    # signature's features.
    dataset_fingerprint: str = artifact_input()
    # The fingerprint of each cell's split, or None for one split.
    split_assignments_fingerprint: str | None = artifact_input()
    split_columns: tuple[str, ...] = parameter()
    min_de_genes: int = parameter()
    min_cells: int = parameter()
    logfc_threshold: float = parameter()
    min_fraction: float = parameter()
    p_value_cutoff: float = parameter()
    scale: bool = parameter()
    max_iterations: int = parameter()
    perturbation_type: str = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PerturbationAssociationArguments(OperationArguments):
    operation: ClassVar[str] = "test_perturbation_associations"
    artifact_kind: ClassVar[str] = "perturbation_associations"

    labels: ArtifactRef = artifact_input()
    # The counts that the regressions and the QC read.
    dataset_fingerprint: str = artifact_input()
    # Trans pairs test every feature of a feature selection; explicit pairs
    # and positive controls are fingerprinted as target and feature IDs.
    features: ArtifactRef | None = artifact_input()
    pairs_fingerprint: str | None = artifact_input()
    positive_controls_fingerprint: str | None = artifact_input()
    # The fingerprint of each cell's sample and of the covariate columns over
    # the tested cells, the control cells and the cells of the tested targets
    # or guides, or None without them.
    sample_assignments_fingerprint: str | None = artifact_input()
    covariates_fingerprint: str | None = artifact_input()
    integration: str = parameter()
    side: str = parameter()
    n_resamples: tuple[int, int, int] = parameter()
    stage_threshold: float = parameter()
    approximation: str = parameter()
    min_nonzero: int = parameter()
    n_calibration_pairs: int | None = parameter()
    alpha: float = parameter()
    seed: int = parameter()
    sample_columns: tuple[str, ...] = parameter()
    covariate_columns: tuple[str, ...] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PerturbationEDistanceArguments(OperationArguments):
    operation: ClassVar[str] = "measure_edistance"
    artifact_kind: ClassVar[str] = "perturbation_edistance"

    labels: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    by: str = parameter()
    n_permutations: int = parameter()
    random_seed: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class MembershipStrengthArguments(OperationArguments):
    operation: ClassVar[str] = "calc_membership_strength"
    artifact_kind: ClassVar[str] = "membership_strength"

    connectivity_map: ArtifactRef = artifact_input()
    clusters: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    decimals: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class SmartLabelArguments(OperationArguments):
    operation: ClassVar[str] = "smart_label"
    artifact_kind: ClassVar[str] = "smart_label"

    values: ArtifactRef = artifact_input()
    base_labels: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    suffix_style: str = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class StatisticalTestingArguments(OperationArguments):
    operation: ClassVar[str] = "run_statistical_testing"
    artifact_kind: ClassVar[str] = "statistical_tests"

    grouping: ArtifactRef | None = artifact_input()
    cell_selection: ArtifactRef | None = artifact_input()
    tested_features: tuple[str, ...] = artifact_input()
    source_assays: tuple[str | None, ...] = artifact_input()
    source_dataset_fingerprint: str | None = artifact_input()
    cell_selection_fingerprint: str = artifact_input()
    group_fingerprint: str = artifact_input()
    subset_fingerprint: str | None = artifact_input()
    sample_fingerprint: str | None = artifact_input()
    pair_fingerprint: str | None = artifact_input()

    group_field: str | None = parameter()
    method: str = parameter()
    posthoc: str | None = parameter()
    adjustment_method: str = parameter()
    alternative: str = parameter()
    equal_var: bool | None = parameter()
    sample_stat: str = parameter()
    expression_cutoff: float = parameter()
    groups: tuple[Any, ...] | None = parameter()
    comparisons: tuple[tuple[Any, Any], ...] | None = parameter()
    sample_by: str | None = parameter()
    pair_by: str | None = parameter()
    subset_by: str | None = parameter()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] | None = parameter()
    size_factor: float | None = parameter()
    n_groups: int = parameter()
    n_cells: int = parameter()
    key_labels: tuple[str, ...] = parameter()
    from_assay: str | None = execution()
    invalidate_cache: bool = execution()
    # Only Mann-Whitney records carry its p-value policy, so other tests keep
    # their identities.
    p_value_policy: str | None = parameter(None, omit_if_none=True)
