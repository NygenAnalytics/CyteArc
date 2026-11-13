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

#: The diagnostics of joining a file's barcodes to the selected cells, which
#: every importer of barcodes saves: the distinct barcodes, counted by the
#: cell ID they name; those that name no cell of the store; those that name a
#: cell of the store outside the selection; and up to five examples of each.
JOIN_DIAGNOSTICS: tuple[str, ...] = (
    "fileBarcodes",
    "unmatchedBarcodes",
    "unselectedBarcodes",
    "unmatchedBarcodeExamples",
    "unselectedBarcodeExamples",
)
#: The diagnostics that the clone writer saves with the clones of every
#: clone source: the cells given a clone, and the share of them that the
#: largest clone holds.
CLONE_DIAGNOSTICS: tuple[str, ...] = ("cellsWithClone", "largestCloneShare")
#: The diagnostics of grouping cells for a summary: the measured cells
#: without a clone, the cells with a clone but no group, and the groups.
GROUPING_DIAGNOSTICS: tuple[str, ...] = (
    "cellsWithoutClone",
    "cellsWithoutGroup",
    "groups",
)


@dataclass(frozen=True, slots=True)
class UmapArguments(OperationArguments):
    operation: ClassVar[str] = "run_umap"
    artifact_kind: ClassVar[str] = "embedding"
    # The span of densMAP's log local radii, None without densMAP.
    diagnostics: ClassVar[tuple[str, ...]] = ("densityRadiusSpan",)
    call: ClassVar[str | None] = "ds.embeddings.umap"

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
class LocalmapArguments(OperationArguments):
    operation: ClassVar[str] = "run_localmap"
    artifact_kind: ClassVar[str] = "embedding"

    graph: ArtifactRef = artifact_input()
    initialization: Any = artifact_input()
    localmap_dims: int = parameter()
    n_mid_near: int = parameter()
    n_further: int = parameter()
    iterations: tuple[int, int, int] = parameter()
    learning_rate: float = parameter()
    max_distance: float = parameter()
    random_seed: int = parameter()
    # The layout does not depend on the thread count.
    nthreads: int = execution()
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
    # The samples whose cells share one cluster label while heterotypic_fraction
    # asks for doublets of two clusters, so that they form none, and up to five
    # of them.
    diagnostics: ClassVar[tuple[str, ...]] = (
        "oneClusterSamples",
        "oneClusterSampleExamples",
    )
    call: ClassVar[str | None] = "ds.qc.doublets"

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
class PotencyArguments(OperationArguments):
    operation: ClassVar[str] = "run_potency_scoring"
    artifact_kind: ClassVar[str] = "cell_potency"
    #: The network genes that the assay's features matched, those of the
    #: largest connected part that the scores used and its edges, and whether
    #: genes matched by Ensembl ID or by symbol.
    diagnostics: ClassVar[tuple[str, ...]] = (
        "networkGenesMatched",
        "networkGenesUsed",
        "networkEdgesUsed",
        "matchedBy",
    )
    call: ClassVar[str | None] = "ds.scores.potency"

    cell_selection: ArtifactRef = artifact_input()
    measure: str = parameter()
    species: str = parameter()
    #: The network file's source, version, name and SHA-256 digest.
    string_network: dict[str, str] = parameter()
    invalidate_cache: bool = execution()
    # Smoothing over a graph is recorded only when a graph smooths the scores.
    graph: ArtifactRef | None = artifact_input(None, omit_if_none=True)
    smoothing: float | None = parameter(None, omit_if_none=True)


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
    method: str = parameter()
    sources: tuple[Any, ...] = parameter()
    sinks: tuple[Any, ...] = parameter()
    min_max_norm_ptime: bool = parameter()
    component_policy: str = parameter()
    invalidate_cache: bool = execution()
    # The parameters of the truncated potential, None for the untruncated one.
    n_singular_vals: int | None = parameter(None, omit_if_none=True)
    random_seed: int | None = parameter(None, omit_if_none=True)
    # The solver's bounds, recorded only when the call sets them.
    solver_tol: float | None = parameter(None, omit_if_none=True)
    max_iterations: int | None = parameter(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class FateMappingArguments(OperationArguments):
    operation: ClassVar[str] = "run_fate_mapping"
    artifact_kind: ClassVar[str] = "fate_map"

    connectivity_map: ArtifactRef = artifact_input()
    sink_labels: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    sinks: tuple[Any, ...] = parameter()
    beta: float = parameter()
    solver_tol: float = parameter()
    max_iterations: int = parameter()
    invalidate_cache: bool = execution()
    # What directs the walk: a pseudotime or a per-cell direction score. A map
    # records only the one it took, so maps directed by a pseudotime keep
    # their identities.
    pseudotime: ArtifactRef | None = artifact_input(None, omit_if_none=True)
    direction: ArtifactRef | None = artifact_input(None, omit_if_none=True)


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
class LineageMembershipArguments(OperationArguments):
    operation: ClassVar[str] = "call_lineage_membership"
    artifact_kind: ClassVar[str] = "lineage_membership"

    curves: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    band: float = parameter()
    distance_score: float = parameter()
    n_bins: int = parameter()
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
    # The HTOs whose positive calls overlap their background, each counted in
    # every sample where it does, and up to five of them with their sample.
    diagnostics: ClassVar[tuple[str, ...]] = (
        "overlappingHtos",
        "overlappingHtoExamples",
    )
    call: ClassVar[str | None] = "ds.qc.hto_demultiplexing"

    cell_selection: ArtifactRef = artifact_input()
    feature_ids_fingerprint: str = artifact_input()
    method: dict[str, object] = parameter()
    random_seed: int = parameter()
    invalidate_cache: bool = execution()
    # The hashtags demultiplexed, in the assay's order: one list for every
    # sample, or each sample's list by its label; None for all of them.
    hashtags: list[str] | dict[str, list[str]] | None = parameter(
        None, omit_if_none=True
    )
    # The cell metadata column of each cell's sample, within which hashtags
    # are demultiplexed, and the fingerprint of the cells' samples; None for
    # one sample of every selected cell.
    sample_column: str | None = parameter(None, omit_if_none=True)
    sample_assignments_fingerprint: str | None = artifact_input(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class CellCallingArguments(OperationArguments):
    operation: ClassVar[str] = "run_cell_calling"
    artifact_kind: ClassVar[str] = "cell_calling"
    # The slope of the Good-Turing frequency fit of the ambient pool.
    diagnostics: ClassVar[tuple[str, ...]] = ("goodTuringSlope",)
    call: ClassVar[str | None] = "ds.qc.call_cells"

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
    diagnostics: ClassVar[tuple[str, ...]] = JOIN_DIAGNOSTICS
    call: ClassVar[str | None] = "ds.perturbation.import_guide_calls"

    cell_selection: ArtifactRef = artifact_input()
    # The guide counts give each cell's top guide and guide UMIs.
    dataset_fingerprint: str = artifact_input()
    guide_ids_fingerprint: str = artifact_input()
    # The SHA-256 of the call file's bytes; its path is not part of the identity.
    call_file_fingerprint: str = artifact_input()
    barcode_prefix: str | None = parameter()
    invalidate_cache: bool = execution()
    # Recorded only when given, so imports without them keep their identities.
    barcode_suffix: str | None = parameter(None, omit_if_none=True)
    barcode_pattern: str | None = parameter(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class CloneImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_clones"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *JOIN_DIAGNOSTICS,
        *CLONE_DIAGNOSTICS,
        "fileRows",
        "cellsInSeveralClones",
    )
    call: ClassVar[str | None] = "ds.clones.import_clones"

    cell_selection: ArtifactRef = artifact_input()
    # Each clone file's format, the SHA-256 of its bytes and of the files it
    # reads beside it, the columns it reads, how its barcodes name cells and
    # the namespace of its clone labels, in canonical order; paths are not
    # part of the identity.
    sources: tuple[dict[str, Any], ...] = artifact_input()
    multiple: Literal["error", "drop"] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class CloneColumnArguments(OperationArguments):
    operation: ClassVar[str] = "import_clones_from_column"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = CLONE_DIAGNOSTICS
    call: ClassVar[str | None] = "ds.clones.from_column"

    cell_selection: ArtifactRef = artifact_input()
    # The fingerprint of each selected cell's label in the column.
    labels_fingerprint: str = artifact_input()
    column: str = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class ClonalExpansionArguments(OperationArguments):
    operation: ClassVar[str] = "measure_clonal_expansion"
    artifact_kind: ClassVar[str] = "clonal_expansion"
    diagnostics: ClassVar[tuple[str, ...]] = GROUPING_DIAGNOSTICS
    call: ClassVar[str | None] = "ds.clones.expansion"

    clones: ArtifactRef = artifact_input()
    groups: ArtifactRef | None = artifact_input()
    # The cells measured: the selection given, or the clones' cells.
    cell_selection: ArtifactRef = artifact_input()
    # The fingerprint of each measured cell's group, or None without group_by
    # columns.
    group_assignments_fingerprint: str | None = artifact_input()
    group_by: tuple[str, ...] = parameter()
    # Exactly one of the two bounds is recorded; the other is None.
    breakpoints: tuple[int, ...] | None = parameter()
    frequency_breakpoints: tuple[float, ...] | None = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class CloneFateBiasArguments(OperationArguments):
    operation: ClassVar[str] = "measure_clone_fate_bias"
    artifact_kind: ClassVar[str] = "clone_fate_bias"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *GROUPING_DIAGNOSTICS,
        "earlyCellsWithoutLateCells",
        "earlyCellsWithTiedFates",
        "testedClones",
    )
    call: ClassVar[str | None] = "ds.clones.fate_bias"

    clones: ArtifactRef = artifact_input()
    groups: ArtifactRef | None = artifact_input()
    # The early cells, which the artifact labels.
    cell_selection: ArtifactRef = artifact_input()
    late: ArtifactRef = artifact_input()
    # The fingerprint of each late cell's fate, or None without group_by
    # columns.
    group_assignments_fingerprint: str | None = artifact_input()
    group_by: tuple[str, ...] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class ImmuneReceptorImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_immune_receptors"
    artifact_kind: ClassVar[str] = "immune_receptors"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *JOIN_DIAGNOSTICS,
        "fileContigs",
        "unmatchedContigs",
        "cellsWithChain",
    )
    call: ClassVar[str | None] = "ds.repertoire.import_receptors"

    cell_selection: ArtifactRef = artifact_input()
    # Each contig file's format, the SHA-256 of its bytes and how its barcodes
    # name cells, in canonical order; paths are not part of the identity.
    sources: tuple[dict[str, Any], ...] = artifact_input()
    productive_only: bool = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class ClonotypeArguments(OperationArguments):
    operation: ClassVar[str] = "define_clonotypes"
    artifact_kind: ClassVar[str] = "clonotypes"
    diagnostics: ClassVar[tuple[str, ...]] = CLONE_DIAGNOSTICS
    call: ClassVar[str | None] = "ds.repertoire.clonotypes"

    receptors: ArtifactRef = artifact_input()
    # The fingerprint of each receptor cell's labels in the within columns, or
    # None without columns.
    within_fingerprint: str | None = artifact_input()
    sequence: str = parameter()
    receptor_arms: str = parameter()
    dual_ir: str = parameter()
    same_v_gene: bool = parameter()
    same_j_gene: bool = parameter()
    within: tuple[str, ...] = parameter()
    within_receptor_type: bool = parameter()
    # "identity" links identical junctions, with a max_distance of 0.
    metric: str = parameter()
    max_distance: int = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LineageBarcodeImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_lineage_barcodes"
    artifact_kind: ClassVar[str] = "lineage_barcodes"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *JOIN_DIAGNOSTICS,
        "fileRows",
        "unmatchedRows",
        "emptyRows",
    )
    call: ClassVar[str | None] = "ds.clones.import_barcodes"

    cell_selection: ArtifactRef = artifact_input()
    # Each barcode file's format, columns and library, the SHA-256 of its
    # bytes and how its barcodes name cells, in canonical order; or the assay
    # whose features are the barcodes, by its name, dataset fingerprint and
    # feature IDs. Paths are not part of the identity.
    sources: tuple[dict[str, Any], ...] = artifact_input()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class BarcodeCloneArguments(OperationArguments):
    operation: ClassVar[str] = "call_barcode_clones"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *CLONE_DIAGNOSTICS,
        "distinctBarcodes",
        "expectedChanceMatches",
        "mergedBarcodes",
        "moleculesBelowMinReads",
        "moleculesOfLowComplexity",
        "recordsCorrected",
        "recordsOutsideAllowlist",
        "moleculesOfFewBases",
        "pairs",
        "pairsOfOneUmiShared",
        "pairsOfOneRead",
        "pairsBelowMinUmis",
        "pairsBelowMinFraction",
        "cellsWithBarcodes",
        "cellsOutsideBarcodesPerCell",
        "nodes",
        "edges",
        "edgesPruned",
        "doublets",
        "cellsInSmallClones",
    )
    call: ClassVar[str | None] = "ds.clones.barcode_clones"

    barcodes: ArtifactRef = artifact_input()
    # The fingerprint of each barcoded cell's labels in the within columns, or
    # None without columns.
    within_fingerprint: str | None = artifact_input()
    # The fingerprint of the allowlist's sorted distinct barcodes, or None
    # without one.
    allowlist_fingerprint: str | None = artifact_input()
    # The resolved rules: the preset's values, replaced by those given.
    preset: str | None = parameter()
    min_reads: int = parameter()
    max_distance: int = parameter()
    metric: str | None = parameter()
    min_umis: int = parameter()
    min_fraction: float = parameter()
    barcodes_per_cell: tuple[int, int | None] | None = parameter()
    method: str = parameter()
    # One similarity, or the similarity within a library and across libraries.
    similarity: float | tuple[float, float] | None = parameter()
    inclusive: bool | None = parameter()
    prune: bool = parameter()
    doublets: bool = parameter()
    min_size: int = parameter()
    within: tuple[str, ...] = parameter()
    # TREX's rules for its partial cloneIDs and its pair filters, which
    # preset="trex" applies unless trex_rules says otherwise.
    trex_rules: bool = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class VariantCountImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_variant_counts"
    artifact_kind: ClassVar[str] = "variant_counts"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *JOIN_DIAGNOSTICS,
        "listedCells",
        "sortedRuns",
        "repeatedCalls",
        "skippedEmptyAlternatives",
        "skippedOtherAlleles",
        "skippedVariantExamples",
    )
    call: ClassVar[str | None] = "ds.variants.import_counts"

    cell_selection: ArtifactRef = artifact_input()
    # Each source's format, the SHA-256 of each file it reads by role, and
    # how its barcodes name cells, in canonical order; paths are not part of
    # the identity. A Tapestri or DAb-seq source also records the variants
    # that it imports beside its passing ones.
    sources: tuple[dict[str, Any], ...] = artifact_input()
    # The unit of the counts: declared for formats whose files do not state
    # it, and the format's own unit otherwise.
    count_unit: Literal["reads", "umis", "molecules"] = parameter()
    swap_strands: bool = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class VariantSelectionArguments(OperationArguments):
    operation: ClassVar[str] = "select_informative_variants"
    artifact_kind: ClassVar[str] = "variant_selection"
    diagnostics: ClassVar[tuple[str, ...]] = (
        "cells",
        "variants",
        "informativeVariants",
        "maskedVariants",
    )
    call: ClassVar[str | None] = "ds.variants.informative"

    counts: ArtifactRef = artifact_input()
    # The cells whose evidence the statistics use, or None for every cell
    # that the counts' files name.
    cell_selection: ArtifactRef | None = artifact_input()
    preset: Literal["atac", "rna"] | None = parameter()
    # Each threshold as resolved from the preset and the arguments, None for
    # a test not made.
    min_cells: int | None = parameter()
    min_strand_correlation: float | None = parameter()
    min_vmr: float | None = parameter()
    min_depth: float | None = parameter()
    min_quality: float | None = parameter()
    # The names of the masks, sorted.
    mask: tuple[str, ...] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class FrequencyCloneArguments(OperationArguments):
    operation: ClassVar[str] = "call_frequency_clones"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *CLONE_DIAGNOSTICS,
        "variants",
        "cellsScored",
        "cellsWithoutCoverage",
        "cellsWithoutAlternative",
        "cellsWithoutNeighbors",
        "edges",
    )
    call: ClassVar[str | None] = "ds.variants.frequency_clones"

    # The cells of the selection's counts, which the clones label.
    cell_selection: ArtifactRef = artifact_input()
    informative: ArtifactRef = artifact_input()
    # The fingerprint of each scored cell's labels in the within columns, or
    # None without columns.
    within_fingerprint: str | None = artifact_input()
    n_neighbors: int = parameter()
    resolution: float = parameter()
    within: tuple[str, ...] = parameter()
    # Fixed choices of the method: the Leiden seed and the smallest Jaccard
    # index of the shared-neighbour graph that Seurat keeps.
    random_seed: int = parameter()
    shared_neighbor_prune: float = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class VariantGenotypeArguments(OperationArguments):
    operation: ClassVar[str] = "call_variant_genotypes"
    artifact_kind: ClassVar[str] = "variant_genotypes"
    diagnostics: ClassVar[tuple[str, ...]] = (
        "wildTypeCalls",
        "heterozygousCalls",
        "homozygousAlternativeCalls",
        "alternativePresentCalls",
        "missingCalls",
        "filteredCalls",
        "alternativesNotShown",
    )
    call: ClassVar[str | None] = "ds.variants.genotypes"

    # The counts' cells, which the genotypes cover.
    cell_selection: ArtifactRef = artifact_input()
    counts: ArtifactRef = artifact_input()
    method: Literal["source", "threshold", "gift"] = parameter()
    # The genotyped sites by label, in the counts' order.
    sites: tuple[str, ...] = parameter()
    # For each genotyped site, its wild-type and alternative alleles as the
    # method compares them; empty for adopted source calls.
    alleles: tuple[tuple[str, str], ...] = parameter()
    # The rules as resolved for the method, None where it applies none.
    min_umis: int | None = parameter()
    min_mutant_umis: int | None = parameter()
    min_fraction: float | None = parameter()
    min_posterior: float | None = parameter()
    # GIFT's swap probabilities by reads per UMI, as (reads, probability),
    # or empty for none.
    swap_probabilities: tuple[tuple[int, float], ...] = parameter()
    # The quality and depth that an adopted source call needs, or None.
    min_quality: float | None = parameter()
    min_depth: int | None = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class GenotypeCloneArguments(OperationArguments):
    operation: ClassVar[str] = "call_genotype_clones"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *CLONE_DIAGNOSTICS,
        "sites",
        "mutantCells",
        "ambiguousCells",
        "cellsInSmallClones",
        "dropoutProbability",
        "errorProbability",
        # The likelihood's searches, one per within group, that stopped
        # splitting at the most clones a fit holds, and the cells they fitted;
        # None for the exact rule.
        "cappedSearches",
        "cellsInCappedSearches",
    )
    call: ClassVar[str | None] = "ds.variants.genotype_clones"

    # The genotypes' cells, which the clones label.
    cell_selection: ArtifactRef = artifact_input()
    genotypes: ArtifactRef = artifact_input()
    # The fingerprint of each grouped cell's within labels, or None without
    # within columns: the cells with a call for "likelihood", those with a
    # mutant call for "exact".
    within_fingerprint: str | None = artifact_input()
    # The sites whose calls define the clones, by label.
    sites: tuple[str, ...] = parameter()
    within: tuple[str, ...] = parameter()
    method: Literal["likelihood", "exact"] = parameter()
    # The likelihood's rules: a dropout or error rate held at every site,
    # left out where each site's is estimated, the fewest cells of a clone and
    # the posterior a cell's clone must reach.
    dropout_probability: float | None = parameter(omit_if_none=True)
    error_probability: float | None = parameter(omit_if_none=True)
    min_size: int | None = parameter(omit_if_none=True)
    min_posterior: float | None = parameter(omit_if_none=True)
    # Fixed choices of the likelihood's model: each site's rates are shrunk
    # toward the pooled rates by priors worth these many calls, a site's
    # error rate stays at most this multiple of the pooled error, and an
    # estimated pooled error at most this rate.
    dropout_prior_calls: float | None = parameter(omit_if_none=True)
    error_prior_calls: float | None = parameter(omit_if_none=True)
    max_site_error_ratio: float | None = parameter(omit_if_none=True)
    max_pooled_error: float | None = parameter(omit_if_none=True)
    # Fixed choices of the exact rule: cells link through a Jaccard index of
    # 1 over the sites both know, and cells whose calls agree with two clones
    # that disagree are ambiguous and join neither.
    similarity: float | None = parameter(omit_if_none=True)
    inclusive: bool | None = parameter(omit_if_none=True)
    ambiguous: bool | None = parameter(omit_if_none=True)
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class RecorderAlleleImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_recorder_alleles"
    artifact_kind: ClassVar[str] = "lineage_alleles"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *JOIN_DIAGNOSTICS,
        "fileRows",
        "unmatchedRows",
        "slotsBeyondLength",
        "rowsWithoutIntegration",
    )
    call: ClassVar[str | None] = "ds.clones.import_alleles"

    cell_selection: ArtifactRef = artifact_input()
    # Each allele file's format, its one integration or the slots of its
    # shorter integrations, the columns it reads, the SHA-256 of its bytes
    # and of the files read beside it, and how its barcodes name cells, in
    # canonical order. Paths are not part of the identity.
    sources: tuple[dict[str, Any], ...] = artifact_input()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class AlleleCloneArguments(OperationArguments):
    operation: ClassVar[str] = "call_allele_clones"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *CLONE_DIAGNOSTICS,
        "alleles",
        "allelesTooFrequent",
        "allelesNotSignificant",
        "allelesInTooManySamples",
        "rareAlleles",
        "cellsWithRareAlleles",
        "clonesWithTooManyAlleles",
        "cellsInClonesWithTooManyAlleles",
        "allelesInBank",
        "observations",
        "observationsInBank",
    )
    call: ClassVar[str | None] = "ds.clones.allele_clones"

    alleles: ArtifactRef = artifact_input()
    # The fingerprint of each edited cell's labels in the within columns, or
    # None without columns.
    within_fingerprint: str | None = artifact_input()
    # The fingerprint of each character's bank, its sorted alleles with their
    # frequencies and samples, or None when the frequencies come from the
    # cells.
    frequencies_fingerprint: str | None = artifact_input()
    method: str = parameter()
    max_frequency: float | None = parameter()
    max_samples: int | None = parameter()
    max_alleles: int | None = parameter()
    # The false discovery rate of method="p_clonal", None for method="graph".
    fdr: float | None = parameter()
    within: tuple[str, ...] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PhylogenyImportArguments(OperationArguments):
    operation: ClassVar[str] = "import_phylogeny"
    artifact_kind: ClassVar[str] = "cell_phylogeny"
    diagnostics: ClassVar[tuple[str, ...]] = (
        *JOIN_DIAGNOSTICS,
        "cellsInNoTree",
        "unnamedLeaves",
        "leafTimeSpread",
    )
    call: ClassVar[str | None] = "ds.clones.import_phylogeny"

    cell_selection: ArtifactRef = artifact_input()
    # Each tree file's format, the SHA-256 of its bytes, how its leaves name
    # cells and the name its trees take, in canonical order; paths are not
    # part of the identity.
    sources: tuple[dict[str, Any], ...] = artifact_input()
    time_unit: str | None = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PhylogenyCloneArguments(OperationArguments):
    operation: ClassVar[str] = "cut_phylogeny_clones"
    artifact_kind: ClassVar[str] = "clones"
    diagnostics: ClassVar[tuple[str, ...]] = (*CLONE_DIAGNOSTICS, "cellsBeforeCut")
    call: ClassVar[str | None] = "ds.clones.phylogeny_clones"

    phylogeny: ArtifactRef = artifact_input()
    # The phylogeny's cells, which the clones label.
    cell_selection: ArtifactRef = artifact_input()
    # Exactly one of the two cuts is recorded; the other is None.
    time: float | None = parameter()
    depth: int | None = parameter()
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
    diagnostics: ClassVar[tuple[str, ...]] = (
        # The fits of each guide in each sample, those whose status is not
        # "fitted", which call no cell, those of each such status, and up to
        # five of them; None and no examples for a method that fits none.
        "guideFits",
        "fitsCallingNoCell",
        "fitsNoCells",
        "fitsTooSparse",
        "fitsNoFit",
        "fitsNoPositive",
        "fitCallingNoCellExamples",
        # The cells with a call, and those of them with several guides.
        "cellsWithCall",
        "cellsWithSeveralGuides",
    )
    call: ClassVar[str | None] = "ds.perturbation.call_guides"

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
    # The rows whose target matches no feature, and up to five of them.
    diagnostics: ClassVar[tuple[str, ...]] = (
        "rowsWithoutFeature",
        "rowWithoutFeatureExamples",
    )
    call: ClassVar[str | None] = "ds.perturbation.efficiency"

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
    # their neighbors; the feature selection gives the signature's columns.
    labels: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
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
    diagnostics: ClassVar[tuple[str, ...]] = (
        # Whether the calibration check ran and why not, the control guides
        # that its groups draw from, its grade, and the false discoveries
        # among its negative-control pairs.
        "calibrationStatus",
        "controlGuides",
        "calibrationGrade",
        "falseDiscoveries",
        "negativeControlPairs",
        # The tested rows and calibration groups whose pairs pass QC without
        # a statistic, and up to five of each.
        "rowsWithoutStatistic",
        "rowWithoutStatisticExamples",
        "calibrationGroupsWithoutStatistic",
        "calibrationGroupWithoutStatisticExamples",
    )
    call: ClassVar[str | None] = "ds.perturbation.associations"

    labels: ArtifactRef = artifact_input()
    # The counts that the regressions and the QC read.
    dataset_fingerprint: str = artifact_input()
    # Trans pairs test every feature of a feature selection; explicit pairs
    # and positive controls are fingerprinted as target and feature IDs.
    feature_selection: ArtifactRef | None = artifact_input()
    pairs_fingerprint: str | None = artifact_input()
    positive_controls_fingerprint: str | None = artifact_input()
    # The fingerprint of each cell's sample and of the covariate columns over
    # the tested cells, or None without them: at low MOI the control cells
    # and the cells of the tested targets or guides, at high MOI every
    # labelled cell.
    sample_assignments_fingerprint: str | None = artifact_input()
    covariates_fingerprint: str | None = artifact_input()
    integration: str = parameter()
    # "low" tests each target's cells against the control cells, "high"
    # against all other cells, with the resampling resolved from None.
    moi: str = parameter()
    resampling: str = parameter()
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
class PerturbationProfileArguments(OperationArguments):
    operation: ClassVar[str] = "compute_perturbation_profiles"
    artifact_kind: ClassVar[str] = "perturbation_profiles"
    # The perturbations with fewer than min_cells cells, which get no profile,
    # and up to five of them with their cells.
    diagnostics: ClassVar[tuple[str, ...]] = (
        "leftOutPerturbations",
        "leftOutPerturbationExamples",
    )
    call: ClassVar[str | None] = "ds.perturbation.profiles"

    # Perturbation labels, or Mixscape classes whose knocked-out cells alone
    # make each target's profile.
    labels: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    # The counts that the profiles average.
    dataset_fingerprint: str = artifact_input()
    # The fingerprint of each cell's sample over the control cells and the
    # cells of the profiled perturbations, or None for one sample.
    sample_assignments_fingerprint: str | None = artifact_input()
    # Each profiled perturbation with the feature ID of its target, from
    # id_column or the names, or None without masking.
    target_features_fingerprint: str | None = artifact_input()
    by: str = parameter()
    min_cells: int = parameter()
    mask_targets: bool = parameter()
    sample_columns: tuple[str, ...] = parameter()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PerturbationClusterArguments(OperationArguments):
    operation: ClassVar[str] = "cluster_perturbations"
    artifact_kind: ClassVar[str] = "perturbation_clusters"

    perturbation_profiles: ArtifactRef = artifact_input()
    # The names of the clustered perturbations, or None for every profile.
    perturbations_fingerprint: str | None = artifact_input()
    metric: str = parameter()
    min_cluster_size: int = parameter()
    min_samples: int = parameter()
    selection: str = parameter()
    seed: int = parameter()
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
