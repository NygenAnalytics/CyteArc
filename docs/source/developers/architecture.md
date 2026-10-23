---
description: Understand CyteArc package boundaries, dependency direction, and where storage, computation, orchestration, and plotting code belong.
---

(architecture)=
# Architecture

CyteArc uses concrete domain packages and a one-way module-load dependency structure.
The public API is exposed through package facades, while reusable computation and storage mechanics live in focused implementation modules.

## Dependency direction

Module-load dependencies point toward the earlier layers in this list:

1. Foundation: `storage`, `matrix`, and `utils`
2. Data model: `metadata`, `assay`, and `graph`
3. Domain algorithms: `neighbors`, `embeddings`, `clustering`, `trajectory`, `metrics`, `features`, `quality_control`, `mapping`, and `regulons`
4. Import and export: `readers`, `writers`, and `merge`
5. Orchestration: `datastore` and `datastore._operations`
6. Presentation: `plotting`

### Root facade

The root `cytearc` package is a public import facade.
It does not form another runtime layer and must not eagerly import the implementation graph.
Only modules listed by that facade are available as lazy root attributes.
Presentation and lower-level algorithm packages such as `plotting`, `clustering`, `neighbors`, and `trajectory` require explicit imports.
This keeps the root surface small and avoids loading optional presentation code.

### Plotting and local imports

`plotting` does not import `datastore`.
Unified plotting consumes a narrow datastore adapter instead of resolving projection paths.
Existing heatmap functions still read Zarr-backed values from their duck-typed `store` and `assay` inputs.
Removing that remaining storage coupling is deferred.

Function-local imports may cross toward presentation when a public method explicitly requests a plot.
Examples include RNA feature selection and datastore quality-control helpers.
These calls do not create module-load cycles.

## Package responsibilities

### Foundation

- `storage/` owns stores, layouts, schemas, arrays, sharding, copying, resource budgets, storage
  profiles, materialization, ANN persistence, selection snapshots, run/stage records, Zarr runtime
  guards, and artifact lineage reports.
  It also owns the operation revision registry (`storage/operation_revisions.py`, see [](operation_revisions.md)), the finite-value contract of artifact arrays (`storage/finite_values.py`), and the writer destination contract (`storage/destinations.py`).
  `storage/content_stamps.py` reuses a value computed from a stored array, such as the
  fingerprint of the ordered row IDs that every selection check compares, while a stamp of the
  array's metadata document and stored chunk bytes is unchanged; any change to those bytes
  computes the value again.
  `storage/virtual_arrays.py` is the one seam for arrays that a store computes, such as composite
  counts: a `VirtualArrayStore` hands out a `VirtualArrayReader` for such an array. Count selections
  are read through `read_selection`, and `storage.feature_stream` hands a virtual array's feature
  streams to its reader, so no other module checks for a virtual array.
- `matrix/` owns the lazy blockwise matrix abstraction used over NumPy and Zarr arrays.
  Its arithmetic, indexing, and reduction behavior keeps it separate from low-level storage mechanics.
  An assay's `ChunkedArray` over `counts` carries the validated `countsT`. Streams that allow it,
  such as feature value reads, take the selected rows from `countsT` through
  `storage/feature_major_rows.py` when that touches fewer chunk bytes, and the matrix's operations
  transform them exactly as rows of `counts`, so a normalization has one implementation.
- `utils/` owns generic array, argument validation, compute, logging, process, and progress helpers.
  The process-tree RSS sampler of pipeline stages finds a process's descendants through the
  `/proc/<pid>/task/<tid>/children` lists, and reads the parent of every process only where the
  kernel has no such lists.
  It also owns streamed column moments (`utils/moments.py`) and the one warning helper (`utils/warnings.py`), which points each warning at the caller's code.
  `utils/numba.py` gives third-party Numba kernels compiled without a cache, such as umap-learn's,
  an on-disk cache with its own files for each set of compile options, once per process and before
  their first use.
  Zarr-specific helpers belong in `storage`, not `utils`.

Facade aliases do not change implementation ownership.
`cytearc.load_zarr` is a facade alias for `storage.stores.load_zarr`.

### Data model

- `metadata/` owns Zarr-backed metadata tables, row streaming, and table queries.
  It is shared by datastore cell metadata and assay feature metadata, so neither `datastore`, `assay`, nor `storage` owns it.
  A table's columns are the members that Zarr lists with their metadata, so a stray object such
  as `.DS_Store` is never a column; `head` and the membership columns reuse the arrays that one
  listing opened, and `to_pandas_dataframe` opens only the requested columns, each once.
  Shared value-selection contracts also live here so domain, orchestration, and presentation code can use one typed contract without reversing dependencies.
  These include the nullable value contract of metadata columns (`metadata/encoding.py`), the assay-membership contract (`metadata/membership.py`), and `CELL_VALUE_NAMES`, the table of cell-aligned artifact kinds, with the reader that aligns their rows to cells (`metadata/selection.py`).
- `assay/` owns normalization, blockwise feature-summary computation, and the RNA, ATAC, and ADT assay types.
  `assay/hvg_statistics.py` streams the per-batch count statistics of the HVG flavors from
  `countsT`, and `RNAassay._batch_hvg_scores` turns them into each batch's scores with the trend
  and residual formulas of `features.variability`.
  `DataStore` owns planning and persistence of feature-summary artifacts; a bare `Assay.score_features` remains computation-only.
  `assay/normalization.py` owns the normalizer flag policy: each assay class declares the flags that its `normed` applies, and callers resolve flags through `applicable_normalization_flags` and `default_normalization_flags` instead of per-assay branches.
- `graph/` owns graph feature projection through named artifact inputs and rejects encoded-path inputs.
  Analysis execution follows explicit artifact references and must not resolve inputs by parsing
  encoded paths or choosing an implicit result.

Data-model modules may call domain algorithms from the method that needs them.
They must not import those packages at module load time.

### Domain algorithms

- `neighbors/` owns ANN construction, KNN queries, graph operations, diffusion, and weighted-neighbor integration.
  It does not own stored KNN graph arrays; persistence lives in `datastore` and `storage`.
- `embeddings/` owns PCA, LSI, Harmony2 batch correction, UMAP, SG-tSNE, embedding initialization,
  and the narrow storage adapter for imported coordinate artifacts.
- `clustering/` owns Leiden clustering and PARIS hierarchy operations.
  `leiden.py` holds the public `leiden_membership` and dispatches between the native `"cytearc"`
  backend and the `"igraph"` and `"leidenalg"` backends, whose GPL packages it imports only when
  one of them is requested (CyteArc does not install them). The native backend lives in the private
  `_leiden_*.py` modules, described in `cytearc/clustering/README.md`, and loads lazily, so
  importing `cytearc.clustering.leiden` loads no Numba kernel. After a release, a change that
  alters the native labels for unchanged parameters needs an `OperationRevision` of
  `run_leiden_clustering` scoped to `backend == "cytearc"` (placement rule 10).
- `trajectory/` owns pseudotime scoring, feature-profile aggregation, feature module clustering, and pseudotime result records.
- `metrics/` owns LISI, silhouette, graph, concordance, and integration scores, including the neighbor label agreement that membership strength stores.
- `features/` owns variability selection (the cytearc, seurat_v3, and pearson_residuals flavors,
  their LOWESS and LOESS trend fits, and batch-aware ranking), feature scoring, enrichment, rank and regression marker searches, group-wise statistical tests, bulk aggregation, GFF parsing, genomic intervals, coordinate-based feature construction, and the name-based gene-family registry.
  `features/genomic/` also owns genome sequence access (`sequence.py`: an indexed FASTA reader
  and GC content), gene annotations and region-to-gene windows (`annotation.py`), and motif
  matrices with MOODS-equivalent thresholds and a two-strand scanner (`motifs.py`).
  `features/enrichment/chromvar.py` owns chromVAR background peaks and deviation scoring beside
  AUCell and WAGGR.
  It also owns presentation-independent feature resolution and normalized value fetching used by datastore workflows and plots.
  `features/statistical.py` owns the statistical design of `compare.test`, and `features/aggregation.py` the bulk profiles of `compare.compute_bulk`; the datastore methods keep grouping resolution, value reads, artifact planning, and persistence.
- `quality_control/` owns filtering, HTO demultiplexing, doublet processing, cell-cycle assignment, and the default cell-cycle gene references.
- `mapping/` owns reference artifacts, feature alignment, confidence, Symphony-style correction, label transfer, and mapping results.
- `regulons/` owns transcription-factor regulon inference: GRNBoost2-equivalent importance
  networks (`grn.py`), co-expression modules (`modules.py`), cisTarget ranking databases and
  motif annotations (`databases.py`), recovery-curve enrichment (`cistarget.py`), pruning into
  regulons (`pruning.py`), cistromes from motifs or binding peaks (`cistromes.py`),
  region-to-gene links (`links.py`), co-regulated gene programs (`programs.py`), and regulon
  specificity (`specificity.py`). Its functions work on in-memory tables and reference files,
  and every set they return has `source` and `target` columns, the network shape that the
  enrichment scorers read. `polars` is an optional dependency for ranking databases, so the
  `regulons` extra does not install `pyarrow`. `results.py` holds `GrnResult`, which names
  `storage.refs` only. `DataStore.regulons.profiles` sums cell groups with
  `compare.compute_bulk` and writes nothing.

Domain algorithm packages must not import `datastore`, `plotting`, or general import/export packages at module load time.
A domain that persists an artifact may use a narrow, named `storage` adapter.

### Import and export

- `datasets.py` owns the public example-dataset downloads: `list_repositories`, `connect`, and
  `Repository` list and download the example repositories, such as `cytearc_docs`, in the public
  `Nygen/cytebase` Hugging Face bucket.
- `cytebase.py` keeps the earlier names of those downloads, `Repository`, `connect`, and
  `list_repositories`, through CyteArc 1.x. The CyteBase catalog SDK and its build pipeline are
  the separate `cytebase` package in the
  [CyteBase repository](https://github.com/NygenAnalytics/CyteBase), which depends on CyteArc.
  CyteArc never imports it, which `tests/test_import_architecture.py` enforces; reading a catalog
  name such as `cytearc.cytebase.Catalog` raises an `AttributeError` that names the package.
  CyteArc opens CyteBase datasets by their `hf://` URIs like any other remote store.
- `readers/` parses supported input formats.
- `writers/` materializes CyteArc stores and exports supported formats.
- `merge/` combines assays and datasets without importing `DataStore` during normal module loading.
- `composite/` presents selected source counts through an ordinary `DataStore` without copying
  them. It is an adapter: only the storage opener imports it, lazily, and core reads reach it
  only through the `storage/virtual_arrays.py` seam. `create.py` reuses merge alignment,
  metadata, and storage preparation. `store.py` is the virtual Zarr store that implements the
  seam, `reader.py` reads selections from the source matrices, `streams.py` holds the feature
  streams, which batch adjacent feature groups into one source read, and `source_cache.py` keeps
  remote source objects on local disk. These four modules import foundation modules only.
  Assays and pipeline stages use their existing array interfaces.

Readers parse, writers materialize, and merge combines.
Format-specific code belongs in a module named for that format.

### Orchestration

`DataStore` remains the primary workflow API.
Its analysis namespaces bind the existing scoped operations in `datastore._operations`.
`DataStore.zw` opens a workspace group once, as the root group is opened once: its members are
read from the store on each access, and CyteArc writes the workspace attributes through it or, after
writing them through another handle (a derived assay's type, a discarded pending assay), opens it
again.
`datastore.namespaces` owns these small accessors, and `datastore.artifact_accessor` owns saved
result access and discovery. Public names do not replace persisted operation names, parameters,
or input keys. The namespace boundary does not change computation or artifact reuse.
Its public class chain is kept for compatibility:

```text
BaseDataStore
  -> GraphDataStore
    -> MappingDatastore
      -> DataStore
```

Method implementations are grouped by responsibility under `datastore._operations`:

```text
graph
embeddings
clustering
trajectory
mapping
mapping_reference
quality_control
features
integration_metrics
presentation
regulons
```

Shared helpers under the same package include `enrichment_store`, `paris_persistence`, and `statistical_store`, which owns the layout, reuse check, writer, and reader of stored statistical-test results.
Operation mixins have no runtime inheritance from datastore facades, no `__init__`, and no runtime imports of sibling operation mixins.
`TYPE_CHECKING` imports of siblings are allowed.
Reusable algorithms must be placed in their domain package before being exposed through a datastore method.

`datastore.pipeline_accessor` orchestrates the fixed basic RNA recipe. Focused internal modules own
recipe validation, run/stage ledger bookkeeping, filtering, frozen field assembly, and cluster
decision persistence. The reusable bounded silhouette comparison lives in
`metrics.cluster_selection`; its datastore adapter validates graph-coordinate lineage and persists
the immutable decision. `metrics.cluster_selection` is not part of the public `cytearc.metrics`
facade. `datastore.pipeline_run` exposes the narrow durable `PipelineRun` handle and its frozen
cell and feature views. Pipeline execution creates immutable artifacts and a strict run/stage
ledger under `pipeline/runs`; it does not write live metadata. The ledger runs stages one at a
time in recipe order on the calling thread. DataStore-owned plotting, marker
loading, and export consume narrow frozen-run views. Completed runs can be reopened by their
immutable label or exact run ID.

`agent/` owns the optional bounded single-RNA workflow. Its lazy facade exposes `analyze_rna`,
`resume_rna`, their asynchronous counterparts, `open_analysis`, `AnalysisRun`, `Study`,
`AnalysisConfig`, and `RuntimeConfig`. `workflow.py` defines fixed stages and calls existing
`DataStore.pipeline` operations through `execution.py`. Agent-local evidence, diagnostics,
pure choice validators, prompts, and structured provider requests remain separate from core
numerical ownership. Prototype orchestrator and standalone-agent packages are retired.

`records.py` owns the external immutable request, events, measured evidence, visible decisions,
validation outcomes, and recovery references. The default directory is
`Path.cwd() / "agent_runs" / <runId>`; callers may choose another new external location.
`compact_result.py` publishes one immutable agent-owned summary under the local store's
`agent_results/<runId>`, linking the verified final core pipeline/configuration to the external
audit. It neither adds a core artifact type nor duplicates the stage history. `result.py`
exposes the saved analysis, and `rendering.py` derives reports for complete and incomplete
outcomes without provider calls or numerical recomputation.

### Presentation

`plotting/` owns the reusable plotting APIs.
It has no import dependency on `datastore`.
The removed `cytearc.plots`, `cytearc.plotting._legacy`, and `DataStore.plot_*` APIs must not be restored.
New plots should return the established plotting result types, accept documented data contracts, and use narrow adapters instead of adding storage-path knowledge.

The single-RNA agent uses the core embedding accessor for the exact final UMAP.
`agent/plots.py` adds a bounded marker panel from saved final marker statistics, with no count
matrix reads. These views return existing public `PlotResult` and provenance types. They
introduce no core plotting API or module-load dependency from plotting to datastore.

`DataStore.plots` is a thin, store-bound accessor over the canonical store-first functions in `cytearc.plotting`.
The accessor imports concrete plot implementations only when a method is called, so this convenience namespace does not reverse the dependency from plotting to datastore.

## Public facade policy

### Lazy and eager facades

The lazy facades in `cytearc`, `agent`, `features` and its subpackages, `readers`, `writers`, `merge`, `utils`, `neighbors`, `clustering`, `embeddings`, `trajectory`, `regulons`, and `plotting` are architectural boundaries, not temporary deprecation shims.
Their documented 1.x exports preserve stable import paths and defer optional or expensive implementations until an export is accessed.
All of them use one private helper, `cytearc._facade`: an export wins over a same-named submodule, resolving an export never replaces a bound value, and only `readers`, `writers`, `merge`, and `utils` present the objects they own under the facade's `__module__`.
`cytebase` is a plain module that keeps the earlier public names of the example downloads, re-exported from `datasets`, through CyteArc 1.x.

The `assay`, `mapping`, `matrix`, `metadata`, `metrics`, and `quality_control` package initializers are eager domain facades.
API reference pages and public contract tests define which of their exports carry compatibility guarantees.

Reloading a lazy facade clears cached exports before resolving them again.
This keeps reload behavior deterministic for tests and interactive work.

Private facade exports used by repository tests are patch seams, not additions to the documented user API.
New production code should import its canonical implementation directly unless it intentionally needs a public patch seam.

### Invalidation policy

Operation revisions are the one explicit mechanism that stops CyteArc from reusing stored results.
`storage/operation_revisions.py` gives each operation that has revisions an append-only tuple
of them; an operation that is not listed is at revision 1. Planning records a revision in
provenance only when it is 2 or more, so an operation without revisions keeps its identities. A
stored artifact that differs from a request only in its revision is superseded: it is never
reused, it stays listable, loadable, and traceable, and lineage reports mark it `stale`. Changing
recorded parameters or inputs is not an invalidation mechanism. [](operation_revisions.md)
holds the decision ladder.

### Stored data and API contracts

CyteArc validates persisted data and public arguments against the following contracts:

- `AssayState` and `IncompatibleAnalysisStateError` are removed. A store containing
  `{assay}/state` is rejected on open and must be rebuilt. CyteArc never reads, migrates, or uses that
  group to choose a current result.
- `DataStore.pipeline.run()` accepts only the documented recipe options and returns a durable
  `PipelineRun`. Removed options and prior return values have no aliases or adapters.
- Feature selection, graph construction, embeddings, clusterings, scores, markers, mapping, and
  trajectory operations exchange exact `ArtifactRef` values. Consumers do not parse encoded
  metadata names, resolve an implicit latest result, or accept a live result column in place of an
  artifact.
- Analysis producers do not rewrite live `I` columns and do not insert clustering, UMAP, score, or
  marker columns. Callers use artifact loaders, frozen run views, and plotting adapters instead.
- Public result records use their current artifact-based constructors. Older positional layouts
  and field sets are unsupported.
- Pipeline run, stage, and label-claim records have exact fields. Within 1.x a release may only
  add fields and must still read records without them, so a run of any 1.x release reopens in
  every later 1.x release.
- Pipeline stages run strictly in sequence. UMAP no longer runs on a worker thread beside the
  Leiden, Paris, cluster-selection, and membership-strength stages; outputs and artifact
  identities are unchanged. Durable stage records now attribute wall time and sampled memory to
  one stage at a time, and starting a stage while an earlier stage of its run is incomplete
  raises `ValueError`. Pipeline callbacks receive an enabled stage's `stage_started` and terminal
  event before the next stage starts; earlier, UMAP's `stage_completed` arrived after the events
  of those stages, with or without a worker thread. Earlier run records whose stage windows
  overlap still read. `cytearc.utils.background` is removed.
- `DataStore.pipeline.run` takes a `params` mapping of per-stage settings. Every run records two
  more stages, `membership_strength` and `tsne`, skipped unless requested, and its configuration
  records `params`, `species`, `tsne`, `membershipStrength`, and the Leiden `selected` resolution.
  `pca_dims=0` skips PCA and builds the graph on the normalized artifact (see the `pca_dims=0`
  entry).
- Mapping references and query projections use only their current exact-lineage contracts.
- Label transfer is a saved artifact. `get_target_classes` and `get_target_label_evidence` are
  removed. `mapping.transfer_labels` freezes the reference labels it reads, from a reference column or a
  reference cell-label artifact, into the query datastore as a `reference_labels` artifact and saves
  a `label_transfer` artifact; `mapping.load_label_transfer` loads it without the reference. An abstention is
  a missing label with an `abstentionReason` rather than an `na_val` string, `max_distance` applies
  to the saved labels, and `target_subset` is removed. Evidence renames `isUnknown` to `abstained`
  and adds `candidateLabel` and `nearestDistance`. It no longer repeats the projection's
  `featureCoverage` and `queryScaledDispersion` on every row; read them from
  `mapping.load_result(...).diagnostics`. Conformal sets come from
  `LabelTransferResult.prediction_sets` instead of a `predictionSet` column. `mapping_evidence`,
  `mapping_confusion`, and `mapping_calibration` take the transfer ref instead of a projection,
  reference, and threshold. `mapping_calibration` keeps the transfer's saved rules except the one
  on the swept metric, and without a `chosen_threshold` it marks that rule's own threshold.
  `mapping_confusion` names the abstention column with `abstention_label`. `mapping_score` takes `reference_labels` in place of
  `reference_class_group`. `ExternalArtifactRef` gains `anchor_assay` for an artifact of
  another datastore that is not scoped to the assay its dataset fingerprint describes.
- Integration label metrics are split by input contract. `integration.compute_clisi` and
  `integration.compute_graph_connectivity` use the keyword `annotation_column` for imported cell metadata;
  `clusters.compute_concordance(first, second, metric=...)` compares exact clustering artifacts.
  Their former `label_colname` keywords and column- or array-based concordance inputs are
  unsupported.
- `DataStore.regulons.grn` writes an assay-scoped `grn_adjacencies` artifact with operation
  `run_grn_inference`: parallel `tf_index`, `target_index`, `importance`, and `correlation`
  arrays (one row per link with positive importance), a `regulator_index` array of the factors
  offered to every target, and the method, layout, cell, feature, and regulator digests as
  attributes. It models raw counts, and `early_stopping` records which out-of-bag improvement
  ended each fit. `regulons.load_grn` returns a `GrnResult` and checks the cell and regulator
  digests.
- `enrichment_scores` artifacts gain the `chromvar` method, written by `scores.chromvar` with
  operation `run_chromvar` on an ATAC assay. Its `scores` are chromVAR z-scores and may hold NaN
  for a cell that expects fewer than one fragment in a set; a `deviations` array of the same
  shape holds the bias-corrected deviations, and `background_index` (selected peaks by
  iterations, physical indices) holds the drawn background peaks. `gc_digest` records the GC
  content of the selected peaks as an input. `scores.load` accepts ATAC assays for this method,
  and `EnrichmentResult.deviations` is set only for it. WAGGR and AUCell artifacts are
  unchanged.
- Derived assays are published atomically. `add_grouped_assay` and `add_melded_assay` stage the
  assay under a `cytearc:pending_assay` marker and remove what they created when an error stops them.
  Counts written by `create_zarr_count_assay` carry `complete=False` until they are finalized. A
  pending group left by a hard kill, an interruption, or a failed publication write blocks its name
  in every workspace, and repack refuses it, until `DataStore.discard_interrupted_assay` removes it.
- Mounted targets resolve their source's artifacts read only. A mount lists, loads, traces, and
  reuses the complete artifacts of its identity-checked source after its own, so a recipe that matches saved
  provenance reuses the source's results instead of recomputing them, and labels and embeddings
  imported into the source, such as the CyteBase `X_umap`, are visible on the mount. Nothing is
  migrated because no record changes: `ArtifactRef` stays a location-free name with a random ID,
  provenance and run records are unchanged, and `ExternalArtifactRef` stays the cross-dataset
  identity. Every write goes to the target, and a write inside a source artifact group raises
  `PermissionError`. Pipeline runs and their labels stay per store. The `matrixSource` top level is
  exact (`location`, `workspace`, `assays`), so a mount record with any other field fails to open
  and asks for a fresh target. `repack_zarr` reads a mount through the same namespace and writes a
  self-contained store. A read-only open of an unprepared store, such as the source that
  `mount_datastore` opens, asks for one writable open instead of a rebuild. `local_cache` follows
  the store that holds the normalized artifact rather than the datastore location, so a local mount
  stages a normalized artifact that it reuses from a remote source and never one that the target
  holds; staging on stores that are not mounts is unchanged.
- Public cell filters apply the pipeline filtering rules. `qc.filter` and `qc.auto_filter`
  exclude rows whose metric is missing and raise on missing sample labels among active cells,
  non-finite metrics used for automatic bounds, invalid bounds, duplicate attributes, and empty
  inputs or results. Selections over masked metadata columns record
  `missing_mask_fingerprints`; identities on unmasked stores are unchanged. Earlier selections
  over masked columns remain valid artifacts and must be recomputed.
- Cell-aligned artifact readers carry the linked missing mask. `select_cells`, groupings used by
  statistical testing and distribution plots, and integration metrics exclude or reject missing
  labels, and `qc.doublets` rejects clusterings with missing labels. These readers,
  trajectory cell-data inputs, and pipeline filtering accept only the canonical
  `__cytearc_missing__<name>` mask link.
- Readers show rows that a linked missing mask flags as missing. `MetaData.to_pandas_dataframe`
  and `head`, `get_cell_vals`, live `to_anndata` and `to_h5ad`, and plots show them as missing:
  numeric values become float64 NaN, and in H5AD files booleans become nullable booleans and text
  gets a missing category. `MetaData.fetch` and `fetch_all` stay raw. `compare.compute_bulk` leaves masked
  cells out of every group, dot and matrix plots report `dropped_group_cells`, and
  `cluster_connectivity` rejects masked inputs.
- `markers.search`, `clusters.membership_strength`, `clusters.relabel_by_overlap`, `imputation.compute_imputed` of a metadata
  column, and `silhouette_scoring` of a metadata column reject masked inputs before any reuse or
  write, so results from unmasked inputs keep their identities. `clusters.membership_strength` and
  `clusters.relabel_by_overlap` accept only categorical label kinds.
- `DataStore.snapshot_cluster_labels(labels, cell_selection=...)` freezes a cell metadata column,
  or a cell-label artifact read for a subset of its cell selection, into a datastore-scoped
  `cluster_labels` artifact with operation `snapshot_cluster_labels`. Floating-point labels that
  are whole numbers within the int64 range, such as the float64 ids that pandas writes for integer
  ids with missing values, are stored as int64; other floating-point labels raise `TypeError`, and
  missing, masked, and blank labels raise `ValueError`. Text is stored at the width of the selected
  labels, and integer and boolean labels keep their dtype. The identity records `source_column` as
  a parameter or `source_labels` as an input, the cell selection, and the `values_fingerprint` of
  the stored labels, and a snapshot is reused only while its payload holds exactly those labels.
  The missing-label errors of `markers.search`, `clusters.membership_strength`, and `clusters.relabel_by_overlap`
  name this method and the cell selection that each consumer needs. The group-name error of
  `markers.search` also names this method, and only `markers.search` applies the group-name
  rule.
- Query projections record the input `query_dataset_fingerprint`, which replaces
  `selected_expression_fingerprint`. A query cell with no counts in any shared reference feature
  is uninformative. The diagnostic `zeroNormCellCount` is renamed `uninformativeCellCount`, and
  `queryScaledDispersion` uses shared features only. Older projections fail to load with a
  request to rerun `mapping.run`.
- Mann-Whitney uses the exact permutation null when the two groups can be formed in at most
  100,000 ways. Its statistical-test artifacts record `p_value_policy`, and results carry
  `p_value_method`; identities of other tests are unchanged. Saved Mann-Whitney results without
  `p_value_method` fail to load with a request to recompute. A `StudyDesign`
  pairing column applies only to the paired Wilcoxon test.
- densMAP embeddings symmetrize neighbor distances.
- `embeddings.umap` records float parameters as floats and integer parameters as Python integers, so
  `min_dist=1` and `min_dist=1.0` identify the same embedding. The defaults `min_dist=1` and
  `negative_sample_rate=5` were recorded as integers, so every saved UMAP embedding recomputes on
  its next run. Numeric parameters reject booleans and non-finite values, `umap_dims`, `n_epochs`,
  and `random_seed` must be integers, and `parallel` and `use_density_map` must be booleans.
- Clustering inputs are canonical and strict. `clusters.leiden` records resolutions as finite
  positive floats and requires a non-negative integer `random_seed`. Paris is never refitted
  silently: a reused hierarchy that cannot be read raises
  `ArtifactResolutionError(code="corrupt_payload")`, hierarchies missing required arrays or
  attributes and cuts whose diagnostics do not match the schema are not reused, and
  `clusters.load_paris` rejects such cuts with the same error. A
  fixed Paris cut needs `n_clusters` of 1 or at least the number of connected components; merges
  tied at the cut height are applied in hierarchy order, so exactly `n_clusters` clusters are
  returned. Clustering and doublet entry points reject graph
  references that are not connectivity maps or integrated graphs.
- Leiden clustering defaults to CyteArc's own backend, `"cytearc"`, in `clusters.leiden`,
  `cytearc.clustering.leiden_membership`, the pipeline's Leiden stages, gene programs, and the
  profiling `leidenBackend`. igraph and leidenalg are no longer dependencies of CyteArc, in no
  extra either: users who want `backend="igraph"` or `"leidenalg"` install the GPL packages
  themselves, and asking for those backends without them raises `ImportError` that says so.
  `LeidenArguments.backend` gains `"cytearc"`, and the
  backend stays part of the artifact identity: a default call records `backend="cytearc"` and
  computes new labels instead of reusing an artifact recorded with `"igraph"`. Such artifacts stay
  valid and are reused by calls that pass `backend="igraph"`, without the packages. No operation
  revision is needed, because the GPL backends compute what they did before. `leiden_membership`
  gains a keyword-only `nthreads`, which `clusters.leiden` sets to the datastore's `nthreads`; the
  thread count is an execution option, not part of any identity, and the native labels do not
  depend on it. Native labels are numbered from 1 by decreasing cluster size.
- `markers.load` returns string `group_id` values in plot category order (numeric labels first in
  numeric order) and raises for an unknown group. `markers.export_csv` uses the same column
  order. Only plain decimal labels are numeric, so plots no longer read `"1_10"` as 110; other
  labels sort naturally, with digit runs before text and case-only ties broken by the exact text.
- `scores.waggr` and `scores.aucell` take `ambiguous_targets="drop"` and record
  `dropped_ambiguous_targets`. Enrichment artifacts without `layout` or
  `dropped_ambiguous_targets` fail to load.
- `cytearc.metrics.silhouette_scoring` and `process_cluster` take no positional `ann_obj` and no
  `data_is_reduced` keyword. `silhouette_scoring` requires the keyword-only `distance_metric` and
  compares rows as given.
- `trajectory.fate` treats `solver_tol` as an absolute bound on the largest Bellman residual of
  each solved sink column. Existing fate artifacts remain valid; this change alone does not stop
  their reuse.
- `cytearc.neighbors.diffusion_operator` is removed; it formed a powered operator with no memory
  bound. `bounded_diffusion_operator` is the only powered builder, and
  `neighbors.diffusion.transition_matrix` returns the graph-sized single step. The removal
  changes no doublet score and no diffusion or pipeline artifact identity.
- Assays are no longer `DataStore` instance attributes. `ds.<name>` resolves an assay only when no
  `DataStore` attribute has that name and the name does not start with an underscore;
  `get_assay(name)` returns any assay. Assays named like members, such as `cells` or
  `zarr_mode`, no longer break opening or replace the member.
- Producers that cannot reuse a saved result raise `PermissionError` on a read-only store before
  computing or writing. This covers every artifact producer, including marker search,
  `clusters.relabel_by_overlap`, membership strength, cell filters and selections, graph, embedding, trajectory,
  and mapping producers, and derived assays. Leiden, Paris, UMAP, and t-SNE refuse before they read
  the graph. `features.universe`, `features.snapshot`, `features.detected`, `features.hvgs`,
  `features.prevalent_peaks`, `scores.cell_cycle`, `mapping.build_reference`, and `mapping.run`
  plan first, so a read-only store returns a result that it holds; a feature summary that must be
  computed is refused before the counts are read. `mapping.build_reference` raises this
  `PermissionError` where it raised `ValueError` before planning, and `mapping.run` raises it where
  it raised `ValueError`. WAGGR, AUCell, chromVAR, and `regulons.grn` still refuse a read-only
  store before planning.
- Public arguments are validated before any artifact is written: `graph.load(use_k=...)`,
  `reduction.lsi` `skip_first` and `rand_state`, `reduction.custom` loadings, `reduction.harmony`
  parameters, `integration.modalities(chunk_size=...)`, `features.hvgs` keywords, `embeddings.umap` array
  initializations, and `compare.compute_bulk` column names.
- Normalization arithmetic is independent of the count storage dtype. Library-size, CLR, and
  TF-IDF calculations use float64 for integer and floating-point counts. Persisted normalized
  values and the library-size values ranked by marker search are rounded once to float32;
  marker search ranks values of other normalizations unrounded. Library sizes, subset totals,
  TF-IDF term totals, and feature percentages accumulate in float64.
  `Assay.score_features(log_transform=True)` takes logarithms in float64, including for assays
  without a normalization. Sums over cells combine partial sums of stored row blocks or `countsT`
  cell bands, so their last float64 bits can depend on count layout.
  `compare.compute_bulk` and synthetic doublet counts add integer counts in an integer dtype and
  floating-point counts in float64. RNA normalization maps a zero library total to 1, so a
  zero-count cell normalizes to zero.
- Grouped assays store counts, and rebuilding a store copies those values. Recompute a grouped
  assay with `add_grouped_assay` under a new `assay_label`. A grouped assay containing NaN cannot
  pass the count writers, including a data-only repack; re-import its source and build the assay
  there.
- Pseudotime markers and aggregations compute streamed library-size values in float64. Statistical
  tests compare the fingerprints of their input values for reuse. `invalidate_cache=True` forces
  recomputation.
- Library-size marker search ranks raw counts of every storage dtype with one zero-aware kernel,
  with or without `renormalize_subset`. The kernel validates its input while reading it: a
  negative or non-finite normalized value of a tested feature, or a negative or non-finite total
  of a selected cell, raises `ValueError` naming the feature or totals before anything is written.
  Cell totals come from `<assay>_nCounts`, or from the tested features with
  `renormalize_subset=True`. Rank statistics use float64 group sizes, and the two groups of a
  two-group search have equal p-values. Library-size statistics use float64 values rounded once
  to float32.
  `cytearc.features.find_markers_by_rank` returns a `RankMarkerResult` of the sorted group ids, their
  sizes, the ascending feature index, and the features-by-groups rank statistics instead of one
  DataFrame per group, and raises `IndexError` for a cell or feature index past the end of `countsT`
  instead of ranking an unwritten column as a cell. A `RankMarkerResult` requires at least two
  groups, at least one of them tested (at least two cells, and at least two other cells, as a
  saved marker table needs), and at least one feature; its `table` method gives the ranked table
  of one tested group, with the string `group_id`, that a saved marker table reads back. Feature-stream plans charge what Zarr holds while it reads a band
  (`ArrayGeometry.readBytes`) and the streams' band indices, and size read-group destinations by the
  selected cells, so tight budgets admit fewer reads in flight and budgets that fitted only
  uncharged read buffers raise `MemoryError`. The marker search also reserves its result, the stored
  tables its writers finish, and kernel scratch. Read-group streams process one group at a time in
  order while the next is read, or with `orderedCompute=False` up to the policy's compute workers
  each process one group while one more is read, in completion order. The band reads of the groups
  in flight share the requested read width, `StorageIoPolicy(readWorkers=...)` or eight reads per
  worker, which `WorkShape.maxUnitsInFlight` lets the planner split into inner reads, and an
  execution report names the cause in `reductionReason` when memory or the band count leaves fewer
  reads in flight. The library-size kernel uses as many threads as `nthreads`, which now caps them,
  and the budget admit: it splits the features of one group over the threads of one compute worker,
  or, when groups are too narrow to keep those threads busy, ranks whole groups at once with one
  serial kernel each. It reserves scratch for every kernel call that can run, so marker memory grows
  with the threads only as far as the budget allows, and a search that cannot rank one group with
  one thread raises `MemoryError` naming the bytes it needs. `map_feature_read_groups` loses its
  `extraItemsize` argument, and its `orderedCompute` argument defaults to `True`.
- `find_markers_by_regression`, which `trajectory.markers` calls, scales the regressor and
  each feature's values by a power of two below one in magnitude before it sums them. Pearson r does
  not depend on that exact scale, so ordinary inputs give bit-identical results, but a regressor or
  feature values whose squared deviations overflowed float64 (magnitudes above about 1e150 to 1e154,
  depending on the number of cells) now give their correlation instead of r = 0 with p = 1, or NaN,
  and a regressor whose squared deviations underflowed (below about 1e-160) is tested instead of
  reported as untested. Two-cell searches report r of exactly 1 or -1 instead of a least-squares
  value that could round below one. Pseudotime marker identities are unchanged. It treats a feature
  as constant, and untested, when its values span at most float64 epsilon times their largest
  magnitude; the rule was an absolute span of epsilon.
- Row-block streams over a sharded array reserve what Zarr holds while it reads a block: a
  shard-level copy of the selection, the compressed bytes of every chunk the block touches in a
  shard, charged at their decoded size, and the one chunk decoded at a time, next to the block and
  the output of its first operation. Each later operation of a chain holds the previous output
  beside its own, and a block reserves the larger of these steps, so row blocks of log-normalized
  RNA, CLR, and TF-IDF values, which hold two float64 outputs at once, reserve 16 bytes per value of
  uint8 or uint16 counts instead of 10 or 12. Normalization writes, row-block streams, and
  reductions over such values therefore run fewer blocks at once and raise `MemoryError` where only
  the smaller reservation fitted; single operations, and chains over counts of 32 bits or more,
  reserve what they did. `ArrayGeometry.readBytes` takes the number of chunks decoded at once and
  charges an unsharded read only its result and decoded chunks, and `plan_feature_stream` sizes the
  feature blocks of a sharded array by the same read. Normalization, PCA, LSI, graph construction,
  quality control, export, subset, materialization, melding, and merge plan or size their row blocks
  with these bytes, so a budget admits fewer blocks at once and raises `MemoryError` where only the
  uncharged read buffers fitted; outputs and identities are unchanged. A row-block stream keeps no
  block after it yields it, feature streams drop their own references to a unit before they release
  its reservation, and the storage runner's pool workers keep neither a call nor its result once it
  returns. HVG feature statistics reserve the selected cells' inverse totals, the outputs, every
  band's partial statistics, and each compute worker's band scratch, and no longer keep the totals
  beside their inverses. `compare.compute_bulk` means that stream normalized row blocks reserve their group
  sums and divide them in place.
- Sparse count imports that write several assays from one source, such as Cell Ranger, MTX, and H5AD
  files with antibody or peak features, no longer raise `MemoryError` when a band of a wider assay
  arrives while bands of a narrower assay are pending and every band fits on its own. The writer
  writes the longest run of pending bands that fits, so the earlier bands are written first.
- Storage operations run outside an event loop no longer chain their errors to a `RuntimeError: no
  running event loop`, and a storage read that its own I/O loop cancels raises one `CancelledError`
  instead of a group of two. `cytearc.storage.async_execution.ensure_zarr_host_ceiling` loses its
  `maxWorkers` argument, which nothing passed, and the `write_counts_t` metrics drop
  `sourceRepeatedDecodeCount` and `sourceRepeatedDecodeBytes`, which paired count layouts always
  left at 0 because every `countsT` shard holds whole `counts` chunks.
- ATAC `normed` no longer leaves its fitted TF-IDF state (`n_term_per_doc`, `n_docs`,
  `n_docs_per_term`) on the assay after the call, and RNA `normed` never changes `normMethod`.
  A `DataStore` is not designed for concurrent use from several threads.
- Count storage dtype: each assay's counts are stored unsigned, in the narrowest of uint8, uint16,
  uint32, and uint64 that holds the assay's largest value, exactly when every canonical
  (duplicate-summed) value of the assay is a non-negative integer. Other counts keep their source
  dtype: float32 or float64, or the signed integer dtype of a source with negative values. float16
  is not a count storage dtype; count writers reject it, and H5AD imports read float16 sources as
  float32. `cytearc.storage.count_dtype` holds the one policy, and `cytearc.utils.count_values` the
  value scans that readers and writers share; a scan takes the group of each feature and returns one
  range per group. Every import applies the policy to every encoding: H5AD, 10x HDF5 and Matrix
  Market (`CrToZarr` and `MtxToZarr`), CSV, `SparseToZarr`, and `SeuratToZarr`. Integral float,
  signed, and wider unsigned sources, unsorted or duplicate coordinates, CSC, and dense matrices
  therefore store the same dtype and content fingerprint at any memory budget, and 10x HDF5 and
  Matrix Market imports, which stored uint32, and Seurat imports, which kept float64, store the
  narrowest unsigned dtype of each assay. Imports that split one source matrix into assays (10x HDF5
  and Matrix Market feature types, H5AD `assay_split_key`) resolve each assay's dtype from its own
  features, so an assay's dtype and identity do not depend on the other assays of the source: an
  antibody, guide, or ATAC assay stores its own narrowest dtype beside RNA counts that need a wider
  one, and integral assays stay unsigned beside an assay with fractional or negative values. The
  count dtype arguments are removed: `CrToZarr` and `MtxToZarr` `dtype`, `MtxReader` `dtype` and its
  `consume` `dtype`, `CSVtoZarr` `dtype`, `SparseToZarr` `matrix_dtype`, and `DataStoreMerge`
  `dtype` (merge manifests no longer record it); `create_zarr_count_assay`,
  `create_empty_zarr_count_assay`, and `DerivedAssayTransaction.create_counts` require the dtype.
  `CrReader` subclasses implement `matrix_dtype` and `count_value_ranges(maxBytes,
  featureGroups=None)`, which returns the range of each group of features over the selected cells,
  and `H5adReader` has the same method. `MtxReader` scans the matrix once at construction for every
  orientation and filter mode, which replaces its coordinate-order probe, and keeps the largest
  count of each feature; it reads the file once more only when it dropped cells and an import splits
  its features into several assays. Its batches hold the counts of the kept cells in the narrowest
  unsigned dtype that holds them. `H5adReader` loses its `dtype` argument and its `matrixDtype`,
  `storageDtype`, and `infer_storage_dtype` members, and `H5adToZarr.storageDtype` becomes
  `storageDtypes`, the dtype of each imported assay; `consume` yields `sourceMatrixDtype` values for
  every encoding, except that the converted rows of a CSC integer source hold their duplicate-summed
  values in int64 or uint64 (`consumeDtype`), so duplicate sums past a narrow source dtype import as
  they do from CSR. Seurat count sources compressed over features, transposed, or stitched from
  Assay5 layers keep the duplicate coordinates of a cell in the source dtype, as sources compressed
  over cells do, and the writer sums them in 64 bits: such duplicates no longer raise OverflowError,
  and Assay5 duplicates whose sum exceeds a narrow source dtype, which were stored wrapped, store
  their sums. Subset and repack keep the source dtype, because they rebuild an existing dataset
  whose identity and copied artifacts must stay valid. Merges store the common type of the source
  count dtypes, widened when features summed by name could overflow it, instead of float64 for
  differing dtypes, and reject integer sources without a common integer dtype; an assay without
  features in any source stores uint8 instead of uint32. Derived assays (grouped and melded) keep
  their float64 values. Dense writers (CSV and dense Seurat counts) no longer cast batches before
  the checked cast, so a count that the stored dtype cannot hold raises OverflowError instead of
  wrapping. The CyteBase build no longer forces the source dtype, and its records drop
  `storageDtypePolicy`. Count matrices hold finite values: every count writer, including subset,
  merge, repack, and derived assays, rejects NaN and infinity, and the H5AD, 10x HDF5, Matrix
  Market, CSV, sparse, and Seurat imports read every value and reject them before they create the
  destination. Stores whose counts contain NaN or infinity cannot be subset, merged, or repacked,
  even with `--data-only`; re-import their sources. Negative values stay storable.
- Operations trust that prepared counts and artifacts do not change during a call. Writing to
  prepared data in place is outside the contract and is not detected.
- Minimum versions rise to scipy 1.15, statsmodels 0.14.5 (earlier releases fail to import
  with scipy 1.16), threadpoolctl 3.5 (earlier releases cannot see the OpenBLAS in NumPy and
  SciPy wheels, so BLAS thread limits had no effect), huggingface-hub 2.0, and, for the `agent`
  and `test` extras, pydantic-ai-slim 2.51.
- Count layout: plans whose countsT chunks fell below half the chunk target (awkward cell counts
  such as primes) now use whole-target chunks, so stores written with those plans fail layout replay
  and must be re-imported. Count assays require Zarr format 3. Every count writer admits its layout
  against `mem_budget` before it creates its destination: `H5adToZarr`, `CrToZarr` and
  `MtxToZarr`, `CSVtoZarr`, `SparseToZarr`, `SeuratToZarr`, `SubsetZarr`, `subset_assay_zarr`,
  `repack_zarr`, `DataStoreMerge`, and `add_grouped_assay`. The imports write the default layout
  unless they are given a `policy`, as before, and a layout that does not fit raises MemoryError
  naming the largest halving of it, with `unitBytes` and `chunkBytes` halved together, that fits.
  The other writers, which wrote the default layout at every budget, now halve the default
  `unitBytes` and `chunkBytes` together until the counts write and the countsT transpose fit, and
  a write that does not fit with one-row count shards, or with its explicit `policy`, raises
  MemoryError before the destination exists. Sparse writers admit their sparse band writes and
  dense writers their dense row bands. Writers that choose their source batches (the sparse
  imports, the Seurat imports, and merge) admit batches of one destination row band, the batch
  their write starts from, so a layout never starves the write to narrower batches; only one-row
  shards get one-row batches. `add_melded_assay` keeps sizing its shards to
  the melding band that fits `mem_budget`. The writers raise from their constructors, and
  `DataStoreMerge` from `plan`, instead of from `dump`; `SeuratToZarr` construction also prepares
  every selected source and reads its counts once, so source preparation errors surface there.
  `CrToZarr` and `MtxToZarr` take `lines_in_mem` in the constructor, and `dump` no longer accepts
  it, so the fit reserves the Matrix Market parse buffer that the write uses and a smaller buffer
  fits a smaller budget. An explicit `dump(batch_size=...)` reads at most one destination row band
  per batch, so it can no longer exceed what the fit admitted. Writers that write their assays one
  at a time (Seurat, subset, repack, and merge) admit each assay on its own. The fitted or named
  layout depends only on the budget, the data, the dtypes, and for Matrix Market imports
  `lines_in_mem`, never on the worker count. A resumed merge keeps the layout persisted with its completed counts, so a
  budget change between attempts cannot block it, and fits the layout of the counts it rewrites.
  Sparse imports admit the producer's buffering and the band writes as separate phases, so budgets
  that the summed plan refused now import. `write_counts_t`, `finalize_writer_counts_t`,
  `finalize_writer_counts_t_many`, and the merge writer's `write_assay_counts_t` lose their `policy`
  argument and replay the persisted layout. A `repack_zarr` that fails after it creates its
  destination removes it, so a failed copy, such as an unreadable label claim copied last, no longer
  leaves a store that opens without all its data. Identity does not depend on the layout: content,
  counts, and dataset fingerprints are unchanged, and only the layout fingerprint in
  `cytearc:countMatrixLayout` differs. Existing stores are never rewritten and keep their dtype,
  layout, and identity. A re-import can store a different dtype and therefore get different counts
  and dataset fingerprints and new artifact ids; mounts and mapping references bound to a replaced
  store fail closed.
- `SubsetZarr` loses `overwrite_cell_data`. The parameter had no effect: the constructor always
  opens the destination empty, so a subset never found cell data to keep or replace. Passing it now
  raises TypeError. Explicit `cell_idx` values must be distinct, non-empty, and non-negative;
  invalid values raise ValueError before the destination is created or overwritten. A `cell_key`
  that selects no cells still writes a subset without cells. `subset_assay_zarr` applies the same
  rules to `cells_idx` and `feat_idx` (one-dimensional distinct in-range integers, raising
  IndexError, ValueError, or TypeError) and requires at least one feature, before it creates
  `out_grp`; it wrapped negative indices and repeated duplicates before. The docstrings of
  `H5adToZarr.dump`, `CrToZarr.dump` (and `MtxToZarr.dump`), `SparseToZarr`, and `CSVtoZarr.dump` no
  longer list an AssertionError for a row-count mismatch, because that check could not fail. The
  shard writers raise ValueError for a count stream with more or fewer rows than the destination, or
  for a batch of another width. A CSV file that changes after the reader's pass reports this
  ValueError, which `CSVtoZarr.dump` now documents.
- Stored contracts are strict. ANN indexes carry their complete metadata record including
  `byte_length`; `graph.neighbors` requires recorded `ann_ef` and `ann_parallel`; mapping references
  require `ann_ef`; building and loading a Symphony mapping reference require recorded Harmony
  `batch_levels`, `batch_columns`, and `harmony_parameters`. HVG selections record
  `blacklist_fingerprint` and, for adaptive binning, `variance_estimator` and `variance_quantile`.
  Statistical-test artifacts missing any recorded attribute fail to load. Earlier artifacts without
  these records fail to load or are rebuilt.
- Metadata copies, column clearing, and run snapshots accept only the canonical
  `__cytearc_missing__<name>` link and reject multi-dimensional columns. Copies and snapshots store
  text as unicode and text fingerprints decode bytes as UTF-8, so identities over byte-string
  columns change. `MetaData.columns` lists `I`, `ids`, `names`, then the other columns sorted.
  `MetaData.sift`, `multi_sift`, and covariate partitions treat masked rows as missing, and `insert`
  stores an explicit `fill_value` as a real value without a mask (see the partial-insert entry
  below). `MetaData.get_index_by` matches values that are not text by their text and always returns
  int64 indices. `MetaData.insert` rejects names that are empty, `.` or `..`, contain `/` or `\`,
  because Zarr would nest them into groups, or start with the `__cytearc_missing__` mask prefix;
  `reset_key` and `update_key` apply the same rule. A lookup of a name with a separator suggests the
  `_` spelling that imports store, and a lookup of an empty, `.` or `..` name raises `KeyError`. A
  cell or feature table that holds such a nested group from an earlier import raises an error that
  asks for the source to be re-imported on every read, write, and drop of that name; stores are not
  migrated.
- Storage operations raise a single task failure as itself and a cooperative shutdown as
  `ShutdownRequested`, not as an exception group. A pipeline stage whose exception group holds
  `KeyboardInterrupt` or `ShutdownRequested` is recorded as interrupted and raises that
  interruption. CyteArc keeps no process-wide execution report history; collect reports with
  `execution_report_scope`.
- Recorded artifact inputs have one strict reader, `ArtifactStatus.input_ref(name)`. A missing or
  malformed input raises `ArtifactResolutionError(code="corrupt_payload")` naming the owning kind
  and input, in mapping references and projections, graph lineage, trajectory and
  cluster-selection inputs, plots, and pipeline run records. Query projections validate
  `cell_selection` with the stored-selection validator, and connectivity-map payloads follow the
  neighbor dimension rules, so `n_cells` above 2**32 - 1 is rejected.
- A failed or interrupted artifact write deletes its incomplete slot when the failure comes before
  the write that marks the artifact complete, so `artifacts.list` no longer shows orphaned
  incomplete artifacts after an error; once that publication write is issued nothing is deleted.
  `ProjectionWriter.abort` deletes an unfinished projection before its publication write. Starting
  any artifact on a read-only store raises `PermissionError` before writing, so `reduction.harmony` and
  `trajectory.pseudotime(ss_vec=...)` no longer surface the Zarr read-only `ValueError`.
- Gene families: `cytearc.features.gene_families` is the one registry of name-based families, and
  `ribosomal` always means RPS, RPL, MRPS, and MRPL. `DEFAULT_PERCENT_PATTERNS` moves there from
  `cytearc.assay.classification`, and newly prepared stores record the ribosomal percentage pattern
  as `^RPS|^RPL|^MRPS|^MRPL`. Genes, percentage values, and the HVG blacklist are unchanged.
  Feature identity requires a digit after a registry ID prefix, as `prefix_species` does, so IDs
  of other Ensembl species such as chicken `ENSGALG` are no longer release-drift misses.
- `ChunkedArray` follows NumPy broadcasting: a one-dimensional operand aligns with columns and
  rows are scaled with an `(n_rows, 1)` operand. Scalar keys, other axes, ufunc keywords other
  than `dtype`, and arithmetic between two ChunkedArrays raise; zero-row reductions return NumPy
  shapes.
- `reduction.pca` fits exactly with the Gram solver up to 4096 features, for every block layout,
  and with a randomized block Krylov solver above, which records `solver="randomized"`.
  IncrementalPCA and the `incremental_block_rows` parameter are removed, and revision 2 of
  `run_pca` marks the IncrementalPCA fits that recorded it as stale. PCA artifacts store the
  `variance` and `variance_ratio` of their `dims` components, which `show_elbow_plot` reads
  for a reused fit. Reuse does not require them, so an artifact of an earlier release is
  still reused, and its elbow plot is replaced by a warning.
  Embedding initialization seeds its PCA and keeps rarely used streamed k-means centroids.
  Default cell-cycle genes use CENPU, PIMREG, and JPT1 (mouse Cenpu, Pimreg, Jpt1); parameter
  tuning's native doublet graphs use the candidate ANN seed.
- Integer arguments share one validator: NumPy integers are accepted, integer-like objects such as
  0-d arrays are rejected, and messages read `<name> must be an integer` or `<name> must be at least
  N`. It also validates `DataStore(min_features_per_cell=...)` and `FeatureRef(value, by="index")`.
- Paris fits raise each merge to at least its child heights, so hierarchies of tied graphs
  validate, and straight and fixed cuts number equal-size clusters by hierarchy node order.
  `clusters.load_paris` rejects cuts that do not name their hierarchy.
- Inputs are validated before any lookup or write. Graph flags accept only booleans (NumPy
  booleans included). `markers.search` rejects labels that are blank, `.`, `..`, or contain
  `/` or `\`. `clusters.relabel_by_overlap` suffixes continue past `z` (`aa`, `ab`) and colliding names raise.
  `select_cells` raises when no cell is retained. `compare.compute_bulk` and integration metrics leave out or
  reject NaN, None, blank, and masked labels. Doublet, cell-cycle, prevalent-peak, membership,
  and statistical-testing arguments are checked first, and two-group tests reject `comparisons`
  that reverse the resolved group order. `mapping.run` rejects `query_batches` values that share
  text, such as `1` and `"1"`, and treats nested store locations as one store. `get_cell_vals`
  clipping covers every real numeric column. A stored `defaultAssay` must name an assay.
- Validation corrections that change only which error is raised, never a result or an artifact
  identity: `trajectory.pseudotime` compares the graph's recorded cell count with its stored cell
  selection, as `imputation.diffusion` does, and `cytearc.trajectory.select_pseudotime_component`
  checks that the graph is square with one row per selected cell, so mismatches raise
  `ValueError` instead of `IndexError`. `trajectory.markers` raises `ValueError` when a
  correlation is not finite (a raw pseudotime near the float64 limit) instead of writing an
  unloadable table. `select_cells` raises `ValueError` for an `include` integer beyond the
  float64 range on a floating artifact, instead of `OverflowError`. `integration.modalities` raises
  `ValueError` for sources over different cell selections, whatever their sizes, instead of
  reporting a healthy source as `corrupt_payload`. `graph.neighbors` refuses exactly 2**32 cells,
  which graph payloads cannot hold. `CYTEARC_ZARR_PROFILE` accepts only `fast_local` and `cloud`;
  any other non-empty value, such as `Cloud`, raises `ValueError` instead of being ignored.
- Feature selections must select at least one feature, and every reader applies that one rule. For a
  stored selection that selects none, including the feature universe of an assay without features,
  `DataStore.features.resolve` and every operation that takes a feature selection raise
  `ArtifactResolutionError` (a `ValueError`) with code `corrupt_payload`: `Feature selection must
  select at least one feature`. Before, such a selection resolved and its index read back empty, and
  `scores.waggr`, `scores.aucell`, `markers.search`, `trajectory.markers`,
  `trajectory.aggregation`, `features.normalize`, and `qc.feature_percentage` each raised their
  own error. No CyteArc producer writes such a selection, so only records written or edited outside
  CyteArc are affected. `features.hvgs` and `features.detected` no longer read a reused selection
  back to check it: a reused record that was emptied outside CyteArc is returned, and its consumers
  refuse it. Results and artifact identities do not change.
- `scores.waggr` raises the same `ValueError`s with clearer messages. A selected cell total that is
  negative or not finite now reads `<assay>_nCounts holds negative or non-finite totals of selected
  cells; library-size normalization requires finite non-negative counts`; before, it said only that
  the normalization scalars must be finite. An RNA assay whose `sf` is `None` now reads `WAGGR
  requires a finite positive size factor` instead of claiming a non-default normalization. Results
  and artifact identities do not change.
- `reduction.lsi` with the streaming solver reduces its block so that both the fit and the coordinate
  write fit the memory budget; budgets that logged a reduction and then raised `MemoryError` finish
  with smaller blocks. The block size is an execution option, so identities do not change. A
  parallel UMAP layout runs on `min(nthreads, NUMBA_NUM_THREADS)` Numba threads even when the
  calling thread had fewer, a serial one on one thread, and `Assay.score_features` no longer warns
  for an empty CLR cell selection.
- `DataStoreMerge` refuses destinations that alias, contain, or lie inside a source or that
  already hold content, and creates destinations with mode "w-". Source names cannot contain
  `__`. `overwrite=True` refuses destinations with a prepared assay and clears `defaultAssay`.
  Manifests require `sourceCountFingerprints`; without them, restart with `overwrite=True`.
  Differing unordered cell-column `levels` are unioned; other differing
  attributes are dropped with a warning.
- `DataStoreMerge` records, for each merged assay, the type that every source declares, so `HTO`
  stays `HTO`, `GeneActivity` stays `GeneActivity`, and `CRISPR` stays `CRISPR`; sources that
  declare different types for one assay raise `ValueError` before anything is written. Merged ADT
  and ATAC assays therefore open as `ADTassay` and `ATACassay` and keep CLR and TF-IDF
  normalization. The manifest requires `assayTypes`; without it, restart with `overwrite=True`.
  `plan()` reports a destination whose prepared assay has incomplete counts as
  blocked (`canDump=False`, "A damaged prepared assay requires a fresh destination"), the refusal
  that `dump()` already raised, instead of planning a resume. A merged matrix group counts as
  complete only when its `complete` attribute is `true`, as every other merge component does.
- Imports: H5AD import stores missing categorical, nullable, and string values under linked
  masks and keeps nullable booleans as booleans. `inspect_h5ad` reads group-encoded AnnData
  indexes and prefers an ID column such as `gene_ids`, and reads a one-element `uns` text dataset
  as its element; `H5adReader` reads the index named by `_index`. H5AD sparse matrices need an encoding and a stored shape. H5AD, CSV, and Cell
  Ranger readers reject missing or repeated identifiers. CSV import types values over every row
  and rejects missing or negative counts. Matrix Market BED sidecars give `chrom:start-end`
  Peaks, and feature references are left-joined. Cell Ranger HDF5 reads `matrix` and rejects
  several genome groups. Import writers accept `assay_type`. RDS parsing rejects malformed
  character vectors and xz payloads that need more than 256 MiB of decoder memory.
- Reader edge corrections: `CSVReader` raises `ValueError` for a file without count columns, where
  every column is the ID column or is listed in `cell_data_cols` or `skip_cols`, instead of
  importing an assay without features that `DataStore` cannot open, and it copies `pandas_kwargs`
  instead of adding its `read_csv` settings, such as `chunksize`, to the caller's dictionary.
  `CrH5Reader.matrix_dtype` is the stored dtype in native byte order, with float16 read as float32
  as `H5adReader` reads it, and `consume` yields that dtype, so float16 and big-endian 10x HDF5
  counts import (integral counts unsigned, other counts float32 or the native float dtype) instead
  of failing in SciPy after the destination was created. `H5adReader.sourceMatrixDtype` is likewise
  the stored dtype in native byte order, with float16 read as float32, so an H5AD file whose `X` is
  big-endian (sparse `data` in CSR or CSC, or a dense matrix) imports with the count storage dtype
  instead of failing in SciPy after the destination was created. `CrReader._read_dataset(key)` takes
  a required key and returns a list: the base reader no longer accepts None from it, which no
  shipped reader returned, so the unreachable fallback from missing feature names to feature IDs and
  its warning are removed. A Matrix Market cell sidecar emptied after `inspect_mtx` raises the
  cell-count mismatch error (`Cell sidecar has 0 rows, expected N`) instead of `Cell sidecar must
  contain at least one column`.
- CSV rows have the header's field count: `CSVReader` counts the fields of every row at construction
  and raises ValueError naming the line (`CSV line 3 has 4 fields, but line 1 has 3`) for a row with
  more or fewer fields than the header, or than the first row of a file without one. pandas reads
  the file in chunks of `batch_size` rows and compares a row only with the row before it in the same
  chunk, so a row that started a chunk lost its extra fields, a short row was padded with missing
  values, and a header without a field for the row names, as R `write.table` writes it, made pandas
  take the first column as an implicit index and drop the cell names; each of these files used to
  import without an error. The check reads the text that `read_csv` opens, with the same
  decompression, encoding, and byte order mark handling, after `skip_rows` rows and without blank
  lines, and splits rows with `sep` and the `quotechar`, `quoting`, `doublequote`, `escapechar`, and
  `skipinitialspace` settings of `pandas_kwargs`. `sep` must be one character, so a
  regular-expression separator such as `\s+` raises ValueError, and `pandas_kwargs` cannot set
  `comment`, `dialect`, or `lineterminator`. The check reads the file once more at construction,
  about 16 percent of the reader's pandas pass. `CSVReader` raises KeyError for `skip_cols` names
  that are not CSV columns, as it does for `cell_data_cols`. `CSVtoZarr` no longer compares the
  reader's cell IDs with its row count, because a `CSVReader` returns one ID per row. `MtxReader`
  loses `cell_metadata_path`, which had no effect: the reader extracted the named file from a ZIP
  archive but never read it; passing it raises TypeError, and `MtxCandidate.cellMetadataPath` stays
  as the inspection report of a Parse candidate's cell metadata file. `CrH5Reader` raises ValueError
  (`filtering_cutoff cannot be negative`) for a negative `filtering_cutoff` before it opens the
  file, as `MtxReader` does, whether or not `is_filtered` is set; an unfiltered read used to keep
  every barcode with such a cutoff.
- H5AD geometry: `H5adReader` checks the matrix geometry when it is constructed, before a writer
  opens its destination. The matrix must be two-dimensional, a sparse group needs a stored shape
  even when `obs` and `var` are present, its `indptr` must have one entry more than the rows (CSR)
  or columns (CSC) of that shape, and the shape must equal the `obs` and `var` lengths; an absent
  table takes the matrix's length. Such files used to import with the rows past `obs` dropped or an
  empty feature for each extra `var` row, or failed after the destination existed. Dense `consume`
  rejects a `batch_size` below one, as sparse `consume` does. An `obs` or `var` stored as a dataset
  without fields reads as a table without columns, as `inspect_h5ad` reads it, and cell IDs that are
  not text, such as an integer `cell_ids_key` column, import with embeddings.
- Matrix Market coordinates may come in any order. `MtxReader` and `CrDirReader` read cell-major
  and feature-major files as before. Any other order, such as the descending features within each
  barcode that Cell Ranger 1 and 2 write, used to raise `ValueError` (`coordinates are neither
  cell-major nor feature-major`); it now sets `coordinateOrder` to `"unordered"`, and construction
  reads the file once more to write the entries of each kept cell to temporary rows under
  `temp_dir`, sorted by feature with duplicate coordinates summed wherever they appear, which an
  import then reads. The totals, and so the filtering cutoff, are exact in any order, and the
  entries of each cell and the largest counts, and so `max_window_nnz` and the count dtype, come
  from the summed rows, so such a file stores the counts, dtype, and fingerprints of the same
  matrix in sorted order. The rows hold each count in a dtype that holds the largest kept total,
  which bounds every sum. They take `temporaryDiskBytes` of disk, which construction checks, and
  the end of an import or `close` removes them, as does the garbage collection of a reader that is
  never closed. A later import sorts the file again and raises `ValueError`, `RuntimeError`, or
  `OverflowError` when the file holds a different number of entries than at construction, when a
  kept cell holds a different number of entries or of distinct features, or when a count exceeds
  the largest one that construction found. Other changes, such as a lower count, import as the
  file now holds them, as the sorted paths import them. `CrDirReader` gains `temp_dir`.
- `MtxReader`, `CrDirReader`, `CrH5Reader`, and `H5adReader` take a keyword-only
  `duplicate_feature_ids`. The default `"error"` rejects repeated feature IDs, as before, with a
  message that names the option. `"make_unique"` keeps the first occurrence of each ID and
  renames each later one, in file order and across assays, by appending `-1`, `-2`, and so on,
  skipping a name that is already an ID, and logs a warning with the number of renamed IDs; an
  antibody that repeats the ID of an earlier gene becomes, for example, `CD4-1`. Feature names
  are not changed, so an H5AD file without a feature name column takes the names from the IDs it
  holds; a Matrix Market feature reference matches the IDs of the feature sidecar; and repeated
  cell IDs still raise. `H5adReader.from_inspect` passes the option to the reader, and
  `H5adReader` caches the resolved IDs, so it warns once per reader.
- Metadata column names: H5AD readers and inspection list dataframe columns from
  `column-order` and resolve each listed name as an HDF5 path, so columns that old AnnData
  versions nested under `/` are imported instead of skipped. Tables without `column-order`
  still list their direct children, and a nested index named by `_index` resolves either way.
  `H5adReader` rejects a `column-order` attribute that does not hold names when it is
  constructed, before a writer opens its destination. Every import writer stores a source cell or feature column
  whose name contains `/` or `\` under the name with `_` in their place, and logs the rename.
  Source names that are already valid keep their name; a renamed column whose name is taken
  gets the first free `_2`, `_3`, and so on, in source order. Reserved names are checked after
  renaming. Matrix Market feature-reference columns with separators are renamed instead of
  rejected. `inspect_h5ad` and the agent manifest see such columns, the manifest reports stored
  names, and `uns/batch_condition` columns map to stored names. CyteBase `obs_summary` uses
  stored names while `h5ad_keys` keeps source names. The original name is not recorded in the
  store, and `to_h5ad` exports the stored names. `DataStoreMerge` rejects a `source_column` or
  `prepend_text` with a separator or the mask prefix, at construction and again when planning,
  and run snapshots and stored selections treat `\` like `/` and name the `_` spelling. H5AD
  listing keeps a nested group visible when resolved names do not cover all of it, so the reader
  reports it, and leaves datasets with more than one dimension out of the planned names. A 10x
  feature-reference column that is empty for every feature is not planned.
- Seurat: `SeuratReader` and `inspect_seurat` resolve sidecars only inside `sidecar_root` (default:
  the `.rds` directory), and stream sources need it for sidecar-backed layers. Counts containing R
  `NA` raise `missing_count_value`. Dimnames and LogMap identifiers override names stored in
  sidecars. Factor metadata keeps empty levels. Transposed BPCells nodes, MergeFragments peak
  counts, and RegionSelect boundaries follow BPCells, so re-imported counts can differ. TileMatrix
  `fragments` mode counts each tile that holds one of a fragment's insertions, at `start` and `end -
  1`, so a fragment whose insertions fall in two tiles of one range counts in both instead of only
  the first; stores imported from such matrices must be re-imported. HDF5 sidecars in H5AD layout
  need `encoding-type` or `h5sparse_format` and take axis names only from the index named by
  `_index`; other HDF5 sparse groups ignore `sparse_layout` and `layout` attributes.
- Seurat partial imports: metadata columns decode one at a time, so a column that cannot be
  decoded (a list or matrix column, an empty or repeated name) is left out with a
  `skipped_metadata_column` notice, and `SeuratReader(cell_metadata=...)` includes or excludes
  columns by name; a column named in an include list must decode. `integer64` columns import as
  64-bit integers and `Date` and `POSIXct` columns as ISO 8601 text; `as.character()` of doubles
  (a `deferred_string`) is formatted with 15 significant digits as R does. Row names, Dimnames,
  and list names may be ALTREP. A legacy assay with an empty `counts` slot, such as
  `IntegrateData`'s `integrated` assay, reports `counts_layer_missing`. With the default
  selection, `SeuratToZarr` skips assays and reductions that cannot be imported, with
  `skipped_assay` and `skipped_reduction` notices, and takes `default_assay`, which an active
  assay that is not imported requires; an active `SCTAssay` gives way to the assay of raw counts
  that its models were fitted to. A reduction needs its `assay.used` only for loadings: without
  it, the coordinates import under the default assay without loadings. Reductions of some cells,
  or of the cells in another order, align by name over a cell selection of their cells, and
  Assay5 cells may follow any order. A reduction is a display embedding when it has two or three
  columns and its key names UMAP or t-SNE (or it is named `umap`, `tsne`, or `t-sne`), or as
  `reduction_roles` maps it. The RDS parser rejects attributes that are not a pairlist with
  `RdsFormatError`, reads small values through its read-ahead window, scans character vectors in
  bulk, and hashes compressed input while it decompresses it.
- Exports: `to_h5ad` writes AnnData 0.2.0 encodings and omits `_index` from `column-order`;
  `to_mtx(compress=True)` adds a feature-type column.
- Plotting: `dotplot` and `matrixplot` pool only features of one assay that share an explicit
  label and raise for other shared labels; `matrixplot` orders groups like `dotplot`.
  `marker_heatmap` clusters unclipped z-scores. Continuous color limits follow one policy across
  plots, category scales show only observed categories, `compose_results` keeps panel and raster
  colorbars and honors log scales, and plot functions close the figures they created when they
  raise. `distribution` reports a missing pair value with the `compare.test` message. Default
  category colors range over every value of the colored column or label artifact, so subsets keep
  their colors. Embeddings draw categories and unsorted continuous colors in a seeded shuffle
  (`seed=0`, `None` keeps stored order) and sort feature colors high on top; they settle layout
  without rendering points, and on-data labels move at most about one label height and stay
  nearest their own cluster.
- The agent uses typed inputs, explicit resume, immutable saved decisions, and external audit
  records.
- Removed public API: `cytearc.system_call`, `cytearc.get_log_level`, `cytearc.GffReader`,
  `cytearc.coordinate_melding`, `cytearc.utils.iter_column_blocks`, `cytearc.utils.rss_peak_tracker`,
  `cytearc.matrix.Block`, `ChunkedArray.blocks`, `map_blocks`, `dot`, `std`, `chunks`, and
  `nthreads`, `Assay.to_raw_sparse`, `Assay.mean_features`, `cytearc.assay.rna_assay_type_names`,
  `MetaData.mount_location`, `unmount_location`, `remove_trend`, and `insert(location=)`,
  `cytearc.cytebase.Repository.list_files` and `open_zarr`, `DataStore.set_default_assay`,
  `last_execution_report`, `calibrate_label_transfer_threshold`, `metric_lisi`, and
  `load_metric_lisi` (use `cytearc.metrics.compute_lisi`), `cytearc.clustering.balanced_cut`,
  `BalancedCut`, and `paris_dendrogram`, `cytearc.neighbors.wnn_integration`,
  `cytearc.embeddings.run_harmony` (use `fit_harmony` or `DataStore.reduction.harmony`),
  `cytearc.metrics.compute_simpson`, `knn_to_csr_matrix`, and `report_technical_nesting`,
  `cytearc.mapping.array_hash`, `array_store_hash`, and `conformal_prediction_sets`,
  `MappingReference.fetch_layout`, `cytearc.quality_control.write_doublet_target_zarr`,
  `cytearc.readers.get_file_handle` and `read_file`, `H5adReader.open_clone`,
  `LoomReader` and `LoomToZarr` (Loom import, including agent ingest; convert Loom files to
  H5AD first), import-result `artifactRefs`, `cytearc.writers.bed_to_sparse_array`,
  `create_cell_data`, `load_count_store`, and `load_zarr`, `cytearc.utils.load_zarr` (use
  `cytearc.load_zarr`), `storage.parallel.map_shards`, `cytearc.plotting.collect_legends`,
  `FeatureSummary`, and `register_theme`, `recipes.run_plot_recipe`, `PlotOutput.step_name` and
  `written_path`, `PlotRecipeResult.results`, several `SeuratReader` members, the parameter
  tuning handoff, Pareto, WNN, and refinement exports, `AgentRunConfig.thinkingOffProfile`, and
  `AgentOrchestrator.initialize_request`.
- Pipeline records keep at most 512 characters of an error or interruption message and end a longer
  one with `...`. An interruption with a longer message now ends the run as interrupted instead of
  raising `ValueError`.
- Pipeline signal handling never waits for a lock, so a second signal cannot deadlock a run.
- A pipeline stage setting means exactly what the same keyword means on the stage's method, and an
  omitted setting takes that method's default. Non-finite settings raise before a run record is
  created.
- Pipeline t-SNE settings get the checks of `embeddings.tsne` when the recipe is resolved, before a run
  record exists. A missing `sgtsnepi` only logs a warning there, because a stored embedding can be
  reused without it.
- One writer per store is the documented contract: write a store from one process at a time.
- Importers no longer delete their destination (`cytearc.storage.destinations`). They and
  `SubsetZarr` raise `FileExistsError` for a destination that holds any key, and `ValueError` for a
  local destination inside a Zarr store or a destination whose root group is an assay group. Each
  importer gains `overwrite: bool = False`, which replaces only a CyteArc store that no `DataStore`
  has opened, as `SubsetZarr(overwrite_existing_file=True)` now does; a prepared store and content
  that CyteArc did not write are never replaced. A subset source is prepared, so
  `overwrite_existing_file=True` no longer deletes it through another path or store object.
- Local paths reach Zarr as `pathlib.Path`, so names that hold `#`, `?`, or `;` open as written.
- `SubsetZarr` raises `ValueError` before it writes anything unless all its assays come from one
  `DataStore`. It compared only their numbers of cells, so assays of two datastores of one size
  were accepted and the subset paired one dataset's cell IDs with the other's counts.
- `SubsetZarr`, `DataStoreMerge`, `mount_datastore`, and `repack_store` refuse a source store that
  holds a pending derived assay before they write anything.
- `subset_assay_zarr` never replaces anything in its store: an `out_grp` that overlaps `in_grp`
  raises `ValueError`, and an existing `out_grp` raises `FileExistsError`.
- `to_h5ad` writes a temporary file in the target's directory and moves it over the target once
  complete, so a failed export leaves an earlier file of the same name intact.
- Replacing a metadata column (`MetaData.insert(..., overwrite=True)`, `update_key`, `reset_key`)
  encodes the new values first and restores the previous column on any failure or interruption. Text
  that is not valid UTF-8 raises `ValueError`.
- `MetaData.insert` records missing values in the column's missing mask instead of inventing them:
  its `fill_value` default changes from `np.nan` to `None`. An explicit `fill_value` must fit the
  column's dtype exactly, or `ValueError` is raised before the store changes. `I`, `ids`, and
  `names` never hold missing values.
- `MetaData.multi_sift` raises `TypeError` for a string `columns`, `lows`, or `highs` (a string
  `columns` was split into characters), and `ValueError` when the columns and bounds differ in
  number.
- `CSVtoZarr` requires one `cellDataDtypes` entry for each `cellDataCols` column and raises
  `ValueError` otherwise; unpaired columns used to be dropped silently.
- Assay types are strict presets: `RNA`, `ATAC`, `ADT`, `HTO`, `CRISPR`, `ANTIGEN`, `CUSTOM`,
  `GeneActivity`, `GeneScores`, `URNA`, or `Assay`. `DataStore(assay_types=...)` raises `ValueError`
  for an unknown key or value, and a read-only open whose explicit type differs from the recorded
  one raises `ValueError` naming the remedy (open once with `zarr_mode='r+'`). Each assay exposes
  its resolved type as `Assay.assayType`.
- `DataStore.summary()` reports each assay's resolved type (`Assay.assayType`) instead of the raw
  `assayTypes` record.
- `DataStore(min_features_per_cell=...)` must be an integer of at least -1. A writable open keeps
  `I`, with a warning, when the filter would remove at least half of the active cells; the earlier
  guard compared the threshold with the median, so an ADT panel of ten features lost every cell.
  Derived assays no longer filter `I`. A writable open prepares assays and records `assayTypes`
  and `defaultAssay`; only the first writable open of a store filters `I`, and it records the
  threshold in the cell table. A read-only open writes nothing.
- `mount_datastore` checks its forwarded `DataStore` options before it creates the target, and
  deletes a target whose first open fails, so the mount can be retried.
- `DataStoreMerge` keeps per-cell assay membership: the merged `<assay>_I` is False for the cells
  whose source lacks the assay.
- Membership survives export and import. `to_h5ad` and `to_anndata` without `run` declare the
  exported assay's `<assay>_I` in `uns["cytearc"]["assayMembership"]`, and `H5adToZarr` restores it.
  Import writers skip, with a warning, other source columns named `<assay>_I`.
- Membership columns are protected: `MetaData.insert`, `update_key`, and `reset_key` refuse
  `<assay>_I` of any assay, and `drop` refuses a column that records membership. `add_grouped_assay`
  and `add_melded_assay` copy the source assay's membership to the new assay.
- Seurat imports with `assay_layers` record a cell that only an unselected layer holds as unmeasured
  (False in `<assay>_I`).
- Analysis fails closed on cells that an assay did not measure (`<assay>_I` False): every operation
  that reads an assay's values raises `UnmeasuredCellsError`, a `ValueError` that names the counts
  and the remedy, before it plans or writes. Build analyses over a selection from
  `select_measured_cells`.
- `DataStore.select_measured_cells(assay, *, cell_selection=None)` keeps the cells of a selection
  that an assay measured, and returns the selection itself when the assay measured every cell.
- Display reads show unmeasured cells as missing: their feature values are NaN, normalization is
  fitted over measured cells only, embedding plots draw them in `missing_color`, and summary plots
  leave them out of every statistic and of `n_cells`.
- Plots of feature values record `provenance.extras["unmeasured_cells"]` when an assay did not
  measure some plotted cells, and an embedding color without any value records None in
  `extras["color_limits"]` and draws no colorbar unless its scale sets limits.
- Quality-control filters treat `<assay>_nCounts`, `<assay>_nFeatures`, and the assay's percentage
  columns as reads of that assay, so `qc.auto_filter`, `qc.filter`, and pipeline filtering
  refuse unmeasured cells with `UnmeasuredCellsError`. Earlier, their zero metrics set the MAD
  bounds.
- `compare.compute_bulk` checks only the cells that it reads. A metadata grouping without `cell_selection`
  reads the live `I` cells without a snapshot, so `compare.compute_bulk` writes nothing.
- An operation checks its arguments and assay type before membership, and both before the
  `PermissionError` of a read-only store.
- Exports that cannot declare membership refuse unmeasured cells: `to_mtx`, and `to_anndata` layers
  of an assay other than the exported one. `to_h5ad` and `to_anndata` without layers declare
  membership.
- The agent asks for a `cellKey` of measured cells (`NeedsInput`) during source inspection, before
  any model decision.
- `cytearc.metadata.selection.CELL_VALUE_NAMES` names the canonical per-cell array of every
  cell-aligned artifact kind and replaces `GROUPING_VALUE_NAMES`. Readers no longer fall back to a
  `values` array, and other kinds raise `ValueError`. `select_cells` accepts `pseudotime` artifacts.
- `DataStore.artifacts.load_values(ref, *, value=None, cell_selection=None)` returns a
  `cytearc.metadata.CellValues` (values, cell ids, and missing mask) for any cell-aligned artifact.
  The read is checked against the memory budget before anything is read.
- `select_cells` reads its values through the reader of `artifacts.load_values`, so its refusals take
  that reader's messages, and the read is checked against the memory budget.
- `FeatureRef` checks `value` against `by`: an index must be an integer and a name or id must be
  text, else `TypeError`. `1.9`, `True`, and `"2"` no longer select features.
- `clip_fraction` of `get_cell_vals` and of embedding plots must be at least 0 and less than 0.5,
  checked before any value is read. `cytearc.rescale_array` requires `frac` greater than 0.5 and at
  most 1.
- `DataStore.to_anndata` raises `ImportError` when `anndata` is not installed, instead of logging an
  error and returning None. `to_h5ad` does not need `anndata`.
- `to_anndata(run=run, matrix="normed")` and `to_h5ad(..., run=run, matrix="normed")` export the
  run's stored `normalized` artifact over its highly variable features: the values that the run's
  PCA read. Before, a run export normalized again over the whole feature universe and never matched
  the run.
- One streaming H5AD writer (`cytearc.writers.export`) writes every export with h5py block by block
  and converts dense blocks to CSR in bounded steps, so no complete matrix is held. Live export
  files are unchanged.
- Run exports decide their categoricals themselves: repeated or missing text becomes a categorical
  with naturally ordered categories, so `to_anndata(run=...)` and the H5AD file agree. Run-export
  files differ from those that AnnData wrote before: `X` is compressed with int64 indexes, text is
  `string-array`, and empty groups are not written.
- Artifact provenance gains an optional `revision`, an integer of at least 2; revision 1 is recorded
  by omission, so no existing identity changes until an operation gains a revision
  ([](operation_revisions.md)). A superseded artifact is not reused, and planning logs `Recomputing
  <operation>: ...`. `ArtifactStatus` gains `revision`, `current_revision`, `is_current`, and
  `superseded_by`, and lineage reports mark superseded artifacts `stale`.
- Producers check the arrays that must hold only finite values as they write them
  (`cytearc.storage.finite_values`): normalized data, reductions, batch corrections, embeddings,
  neighbor distances, and graph weights. A non-finite value raises `NonFiniteArtifactError`, a
  `ValueError` that names the operation, the array, and the row, and nothing is published.
- Reuse never checks for non-finite values: an artifact is reused when its provenance matches.
  To recompute a normalized artifact that holds NaN, pass `invalidate_cache=True`.
- Normalized data is held to the finite-value contract: a NaN from a custom normalizer, or the
  `log1p` of a count below -1, raises `NonFiniteArtifactError` from `features.normalize` instead of
  being stored. The public `cytearc.writers.chunked_to_zarr` and `write_renorm_subset_to_zarr` do not
  check.
- Saved normalization applies the configured normalizer: with `norm_dummy` or a custom RNA
  normalizer, `renormalize_subset=True` no longer stores library-size values.
- `log_transform` means `log1p` of the configured normalizer's output, in float64, on every path. A
  normalizer applies only the flags that it supports (`applicable_normalization_flags`), and
  defaults turn flags on only for the RNA library-size normalizers (`default_normalization_flags`).
- `log_transform=True` with CLR, `norm_lib_size_log`, or an ATAC normalizer, and
  `renormalize_subset=True` with CLR, `norm_dummy`, or an ADT or generic assay, raise `ValueError`.
  To plot logged counts of any assay, use `NormalizationSpec(source="raw", transform="log1p")`.
- Every library-size path divides by one definition,
  `cytearc.assay.normalization.library_size_divisors`: a zero total becomes 1, and a negative or
  non-finite total raises `ValueError` naming its source before any result is saved. Results for
  valid totals are unchanged.
- `scores.aucell` and query projections accept bool count stores, and query projections reject complex
  counts with `TypeError`.
- `ChunkedArray.mean` accumulates in float64 on every axis, and variances merge per-block moments,
  so a constant feature's variance is exactly zero.
- Connectivity weights follow umap-learn: each cell's kernel is fitted with the cell itself at
  distance zero, so rows sum to `bandwidth * log2(k + 1)`.
- UMAP identities record only machine-independent settings. `nthreads` controls execution, and
  `symmetric_graph` and `graph_upper_only` accept only booleans.
- ANN thread counts control execution; `ann_parallel` identifies a parallel index. A serial index is
  still built on one thread, while `graph.neighbors` queries any index on the datastore's
  `nthreads`, which never changes the neighbors.
- `graph.ann_index` defaults to `ann_m=24`, `ann_efc=200`, and `ann_ef=200` (before, 48, 50, and
  50). An `ef_construction` below `2 * M` built indices whose searches missed every true neighbour
  of some cells between well-separated clusters, 4.5% of 200,000 clustered cells (recall 0.916).
  On about 1,000,000 cells (Tabula Sapiens and a lupus PBMC atlas at 21 and 50 PCs, and a
  clustered mixture), the earlier defaults gave recall@11 of 0.896 to 0.991, with up to 0.92% of
  cells without a true neighbour. The new defaults keep recall@11 and recall@30 at or above 0.995
  and such cells below 0.05% on four of the five sets, and reach recall@30 of 0.988 on 50-PC
  PBMCs, whose remaining misses lie within 1% of the exact k-th distance. The search depth of 200
  is the larger gain on homogeneous cells; `ann_m` 24 matched 32 there and builds faster in less
  memory. The serial build takes about three times as long as with the earlier defaults.
  Defaults record their own values, so default graphs, and the pipeline's, get new identities,
  and explicit settings keep theirs. The index's `ann_ef` (default 200) is the smallest default
  search depth of its queries; it never changes the index. `graph.neighbors` takes `ann_ef`, by
  default `max(index ann_ef, 2 * k)`, never fewer than `k + 1`. Its identity records the depth it
  used as the parameter `ann_ef` only when that differs from `max(index ann_ef, k + 1)`, the depth
  of the index's own `ann_ef` as hnswlib applies it, which every earlier query searched. This is
  the one canonical rule: a neighbors artifact without `ann_ef` searched its index's depth. An
  earlier neighbors artifact whose depth is unchanged, such as a default query of `k` up to 25 on
  an index of `ann_ef` 50, or up to 100 on one of 200, therefore keeps its identity and is reused,
  on a read-only store too.
  An explicit `ann_ef` equal to that depth names the same artifact. A query whose depth grew, such
  as `k` of 30 at depth 60, records the depth and is computed again from the same index. Mapping
  projections still query with the index's `ann_ef`. A query whose `self_hit_rate` is below
  99.5% warns.
- `embeddings.umap` forms the fuzzy union of the graph itself, so `symmetric_graph=True`, which
  united an already symmetrized graph a second time (each weight `w` became `2w - w**2`), and
  `graph_upper_only=True`, which left the union unchanged, raise `ValueError`. UMAP embeddings
  saved with `symmetric_graph=True` stay loadable but can no longer be requested; recompute them
  without the flag. `clusters.leiden`, `embeddings.tsne`, and the pipeline's `leiden` and `tsne`
  settings refuse `graph_upper_only=True` without `symmetric_graph=True`, which loaded the graph
  unchanged under a second identity. No accepted request computes anything new, so neither change
  needs an operation revision.
- `graph.ann_index` and `embeddings.initialization` accept `rand_state=0`.
- `embeddings.umap` and `embeddings.tsne` check that an initialization reference is an
  assay-scoped `embedding_initialization` before they read the graph, and name
  `embeddings.initialization`; they read the initialization before the graph.
- SNN integration refuses connectivity maps of different `k` before it plans or loads any of them,
  naming the remedy; before, it failed after loading them.
- densMAP over equal local radii adds no density term and returns the UMAP layout of the same seed
  instead of NaN coordinates; a non-finite radius raises `ValueError`.
- t-SNE has one backend, the optional `sgtsnepi` package, and `embeddings.tsne` never runs an `sgtsne`
  executable. `temp_file_loc`, `parallel`, and `nthreads` are removed, so t-SNE identities change
  and earlier embeddings are recomputed. `export_knn_to_mtx` is removed.
- `embeddings.tsne` validates its numeric settings before it reads the graph and records `lambda_scale` and
  `box_h` as floats.
- `embeddings.tsne` raises `ImportError` naming the `tsne` extra when `sgtsnepi` is missing (wheels exist
  for Linux x86_64 and for macOS 26 or newer on arm64). Reusing an existing embedding needs no
  backend.
- `pca_dims=0` builds the pipeline's graph on the normalized artifact instead of an identity
  reduction, so the run has no `pca` or `reduction` output. It needs `doublets=False` and no Harmony
  batch columns; either raises `ValueError` before the run starts.
- Harmony: a cluster whose cells cancel keeps a zero centroid instead of NaN, `fit_harmony` raises
  `ValueError` for non-finite results, and `batch_levels` follows design order.
- `clusters.membership_strength` stores the fraction of a cell's graph neighbors that carry the cell's
  own label.
- Marker `fold_change` is `mean / mean_rest`: `+inf` for a feature that no other cell expresses,
  and NaN when both means are 0 or either is negative. `load_marker_table` refuses tables without
  `fold_change_policy`.
- `features.hvgs` and `params["hvg"]` check every option before anything is written: `min_cells` an
  integer of at least 0, `top_n` and `n_bins` integers of at least 1, `lowess_frac` from 0 to 1,
  `keep_bounds` a boolean, and `bin_strategy` `"adaptive"` or `"fixed"`. Floats such as
  `min_cells=20.0` now raise.
- `features.hvgs` takes `flavor` (`"cytearc"`, the earlier method and default; `"seurat_v3"`;
  `"pearson_residuals"`) and `batch`, a cell column that it snapshots or a label artifact, and
  combines batches by the number of batches in which a gene is highly variable, then its median
  rank. Selections of the cytearc flavor over all selected cells keep their provenance; the other
  flavors and batch-aware selections record `cell_selection`, `dataset_fingerprint`,
  `feature_snapshot`, and `batch` inputs, record `flavor` and its options only for the new flavors,
  and store `mean_nonzero`, `detected`, and with batches `batch_count` and `median_batch_rank`.
  `min_mean`, `max_mean`, `min_var`, and `max_var` filter candidates before `top_n`; before, a
  finite `min_var` ignored `top_n` and `max_var` applied after the cut. `min_cells` defaults to
  None: 1% of the selected cells rounded up, from 3 to 20, and `max_cells` None excludes genes
  detected in more than 99.5% of them, recorded as the resolved integers; before they were 20 and
  `n_selected - 20`. The gene-family blacklist spares CCN1 to CCN6 (`^CCN[A-Z]`) and matches the
  current histone symbols. `select_hvgs` revision 2 covers selections with a finite variance bound,
  with `min_cells` 20 (every selection with the earlier default bounds), or with the earlier default
  blacklist. `norm_pearson_residuals` is an RNA normalizer that takes only `renormalize_subset`.
- `compare.compute_bulk(aggr_type="mean")` on RNA divides each cell by its library-size divisor, as `normed`
  does, which is 1 for a cell without counts. Before, such a cell made its group NaN, which was
  returned as zeros. `compare.compute_bulk` no longer replaces NaN with 0 and raises `ValueError` for a
  non-finite mean or sum. Recompute saved bulk tables of groups that hold cells without counts.
- A stored statistical-test result is reused only when it records every attribute that the request
  determines, `sample_by`, `pair_by`, and `summary_scope` included.
- Doublet simulation meets its heterotypic fraction (`simulate_doublet_pairs` loses `max_tries`),
  and `qc.doublets` raises `ValueError` when every selected cell has one cluster label.
- `lisi_batch_mixing_score` ignores declared categories that no cell has, so labels from one
  observed batch raise `ValueError` instead of returning NaN or 1.0.
- `mapping_calibration` counts every cell with a known label in its coverage, cells without evidence
  included. Provenance records `n_without_evidence`.
- Harmony, WNN, and SNN check their memory need against the datastore budget before they read
  coordinates or graphs, and raise `MemoryError` naming the bytes needed.
- `graph.ann_index` and `graph.neighbors` check the hnswlib index against the memory budget before
  they build or load it. An index without a recorded `ann_m` raises `ValueError` asking to re-run
  `graph.ann_index`.
- Leiden clustering, UMAP, t-SNE, and `graph.connectivity` check the graph or neighbor matrices
  they load, and what they compute beside them, against the memory budget before they read them,
  and raise `MemoryError` naming the bytes, the cells, the edges, and the limit. The estimates
  live in the domain packages (`neighbors.graph.graph_load_memory` and
  `connectivity_peak_bytes`, `embeddings.umap.umap_peak_bytes`,
  `clustering.leiden.native_leiden_peak_bytes`, `embeddings.sgtsne.sgtsne_input_bytes`); the
  native Leiden estimate leaves out the per-worker scratch that a hub raises, and the t-SNE one
  the working memory of `sgtsnepi`.
- `run_cluster_selection` reads its sampled coordinates under the memory budget and raises
  `MemoryError` before reading when they do not fit.
- Dotplot size legends draw `SizeScale.areas` exactly, without the 180 pt² cap, and composite size
  legends show the child's 25% to 100% entries.
- `cluster_tree` keeps constant fill values (a fill of 5 has limits 5 to 6) and draws a single text,
  categorical, or boolean value as one category instead of a colorbar.
- Dot plot and matrix plot tables name their grouping columns by role (`group`, `subgroup`,
  `sample`), so `group_by` keys named like summary columns work. With `standardize="feature"`,
  `mean` keeps the raw means and a new `zscore` column holds the plotted values. The
  `tables["matrix"]` of `matrixplot` and `marker_heatmap` is indexed by `feature`, with one column
  per `group`.
- With `run=`, `DataStore.plots.embedding` colors by frozen cell fields, run outputs as
  `ArtifactRef`, or genes with an explicit `normalization=`, and `facet_by`, `subset_by`,
  `Highlight`, and `DensityOverlay` take frozen fields. Run plots record the run in
  `provenance.extras["run"]`.
- `matrixplot` and `marker_heatmap` validate `cluster_method` and `cluster_metric` before they read
  any data; `centroid`, `median`, and `ward` require `cluster_metric='euclidean'` and raise
  `ValueError` otherwise.
- CyteArc's warnings point at the caller's own line (`cytearc.utils.warnings.warn`), so a warning filter
  that matches a CyteArc `module` no longer matches them; filter by message or category instead.
- CyteArc imports on Windows: no module imports a POSIX-only module when it loads.
  `cytearc.utils.process_rss_mb` returns None where `/proc` is absent, and pipeline stages there
  record no RSS samples.
- The wheel is pure Python: the `sgtsne` executable, `bin/`, and `setup.py` are gone, and `sgtsnepi`
  is the optional `tsne` extra. The Docker image installs only the `extra` extra; build it with
  `--extra tsne` for t-SNE.
- A GitHub release uploads to PyPI only after the test and documentation workflows, the build, and a
  wheel smoke check pass on the tagged commit.
- Ensembl GFF3 file names whose assembly contains dots are recognized, so fly and rat references
  download again.
- A Seurat matrix specification that holds a mapping where a vector belongs raises
  `MatrixSourceError` naming the slot, instead of `KeyError`.
- Profiling fixtures name ribosomal features, so their files and checksums change; prepare fixtures
  under a new `datasetPrefixUri`.
- A forced profiling `createStore` job deletes its store prefix before it downloads; an unforced one
  refuses any non-empty destination.
- Harmony is CyteArc's implementation of Harmony2: k-means++ seeding, a stabilized diversity
  penalty, a dynamic ridge penalty of `ridge_alpha` times each batch level's expected mass in a
  cluster, and batch pruning. `harmony_params` and the `fit_harmony` keywords rename `nclust` to
  `n_clusters`, `max_iter_kmeans` to `max_iter_cluster`, and `lamb` to `ridge_penalty`, remove
  `cluster_fn`, and add `ridge_alpha` and `prune_threshold`. The datastore calls
  `validate_harmony_parameters` before it reads data, and the pipeline when it resolves its recipe,
  before it creates a run record; it raises `ValueError` for an earlier key, naming its
  replacement or why it was removed, and `fit_harmony` raises `TypeError` for it.
  `max_iter_harmony` must be at least 1 and `ridge_penalty` positive. `HarmonyResult.ridge` is
  replaced by `ridge_penalty` and `active_levels`, both clusters by batch levels.
  `harmony_peak_bytes` takes `n_cells`, `dims`, `n_clusters`, `column_levels`, and the keywords
  `n_combinations` and `block_size`, without `nthreads`; the second parameter of
  `harmony_cluster_count` is `n_clusters`. These two and `validate_harmony_parameters` are public
  in `cytearc.embeddings.harmony`. The `Harmony` class (in `cytearc.embeddings` and
  `cytearc.embeddings.harmony`), `moe_correct_ridge`, `safe_entropy`, and `ClusterFn` are removed.
- `batch_correction` artifacts replace the `ridge` array with `ridge_penalty` (float64) and
  `active_levels` (bool), both clusters by batch levels, and add the `fit_parameters` attribute.
  Reuse requires both arrays, so a correction saved by the earlier implementation is never reused
  but stays loadable, and the next request recomputes the correction and everything built on it.
  The arrays and attributes that Symphony mapping references copy are unchanged, so references
  built from either implementation load and map without a compatibility branch.
- `qc.hto_demultiplexing` and `cytearc.quality_control.hto_demux` take a keyword-only
  `positive_quantile`, Seurat's `positive.quantile`, with the default 0.99: a real number, not a
  boolean, strictly between 0 and 1. Each HTO's negative-binomial cutoff is that quantile of its
  background fit. `hto_identity` artifacts already recorded the fixed 0.99 as
  `method.cutoff.quantile`, and the quantile is recorded there, so artifacts of the default keep
  their identities and are reused without a compatibility branch, while each other quantile has
  its own artifact. `qc.hto_demultiplexing` checks the quantile before it reads the selection.
- Agent run records: each event records the SHA-256 of every evidence, request, and response
  file it references, and reads verify it. Histories saved before 1.3.0 recorded none; their
  references are read as saved for presentation only, because resume and replay already require
  the procedure identity of the release that wrote them. Agent source fingerprints are version 2,
  hashing whole metadata columns, and inspection evidence records `fingerprintVersion`; an
  analysis with an earlier fingerprint raises when it opens numerical results and asks for a new
  analysis, while its reports and records stay readable. Inspection evidence no longer saves QC
  outlier row lists, and invocations save only the names and a SHA-256 of `modelSettings`.
- HTO demultiplexing assigns a singlet the one tag that its negative-binomial cutoff calls
  positive, not the tag with the highest CLR value, which in unbalanced pools could be a tag
  called negative. `hto_identity` artifacts gain the diagnostic arrays `max_id`, `second_id`
  (labels), and `margin` (float64), which `artifacts.load_values` reads with `value=`. This is
  revision 2 of `run_hto_demultiplexing` for every artifact; the method record keeps its fields,
  `singlet_assignment` included, so revision 1 artifacts are superseded matches that are logged
  and recomputed. A background cluster of only zero counts no longer aborts demultiplexing: its
  HTO gets a cutoff of 0, with a warning. A warning also names each HTO whose positive calls
  overlap its background, where fewer than half of its positive cells exceed the count that the
  background fit exceeds with probability `(1 - positive_quantile) ** 3`.
- `qc.auto_filter` takes `metric_policies`, the MAD policy of named metrics: `transform`
  (`log1p` or `identity`) and `bound_direction` (`two_sided`, `upper`, or the new `lower`).
  Default policies follow what CyteArc wrote: `<assay>_nCounts` and `<assay>_nFeatures` are
  counts, an assay's `percentFeatures` columns and `quality_metric` artifacts of
  `run_feature_percentage` are percentages. Other metrics keep the name rule of earlier releases,
  with a warning. Default calls record the policies and bounds they recorded before, so their
  results and identities are unchanged; a percentage artifact whose name the rule did not
  recognize, such as `percentHb`, now records an upper bound only, a different identity. Only a
  percentage keeps its upper bound within 0 to 100, so an explicit raw upper bound of another
  metric is not cut to 100. Automatic filtering warns when its bounds remove more than half of the
  selected cells, and the MAD bounds of a log1p metric warn when they miss a second population,
  as empty droplets of a raw droplet matrix cause; provenance records neither warning.
- A writable open that skips its `min_features_per_cell` filter because at least half of the
  active cells would go issues a `UserWarning` that names a small antibody panel and a raw droplet
  matrix as causes, instead of only logging it.
- `qc.filter` checks bound types against the column before it reads metadata: a numeric or
  boolean column takes numbers, a text column text, and other columns no bounds. A mismatch raises
  `TypeError` naming the column instead of a NumPy comparison error. Bounds stay exclusive unless
  `keep_bounds=True`, unlike scanpy's `filter_cells`.
- Cell-cycle scoring and `Assay.score_features` count a repeated target feature once, at its first
  position. The recorded `s_gene_indices` and `g2m_gene_indices` keep the repeats, and revision 2
  of `run_cell_cycle_scoring` applies only to scores that record a repeated index, so other scores
  keep their identities and results. A default gene that its symbol misses matches its previous
  HGNC symbol, from `cytearc.quality_control.cell_cycle_previous_symbols`, and the artifact
  records such matches as the `gene_aliases` parameter, which is omitted when empty. Matched
  aliases change the recorded indices, so they change identities without a revision. S and G2M
  scores read their genes and controls in one pass, from `countsT` when its bands touch fewer chunk
  bytes, with bit-identical results.
- `qc.feature_percentage` reads only the selected features and divides by the prepared
  `<assay>_nCounts` instead of summing every count of each cell (see
  {ref}`stable_identity_result_changes`).
- H5AD categoricals keep their categories: `H5adToZarr` stores `levels`, `ordered`, and, for cell
  columns, a categorical `display` with the colors of `uns["<column>_colors"]` on each imported
  categorical column of at most 10,000 categories and on imported cluster labels. Exports write
  such a column as that categorical, with its colors in `uns`. Earlier imports stored plain values
  and are not migrated; they export as before.
- A dataframe index that is neither the ID nor the name column no longer imports as a column named
  `_index`: it is left out when it repeats the IDs or names, and imports as `source_index`
  otherwise. Exports rename a store's `_index` column to `source_index` and a column whose name the
  export already uses to `<name>_2`, with a warning, instead of raising; stores are not changed.
- `inspect_h5ad` ranks count matrices by conventional name (`raw/X`, `layers/counts`,
  `layers/raw_counts`, then `X`, `layers/spliced`, others, and `unspliced` and `ambiguous` last),
  then by estimated total, samples up to 32 windows of 1,024 values across a matrix to decide
  whether it holds counts, which must be finite, non-negative, and whole, and warns when several
  matrices hold counts. It never takes a floating-point column as cell IDs, and when the obs
  index repeats IDs it says so and names `duplicate_cell_ids="make_unique"`, a new keyword of
  `H5adReader`. A missing feature name takes the feature ID.
- `embedding_roles` accepts `"coordinates"`, which imports an `obsm` array as
  `imported_coordinates`, and `DataStore.reduction.import_coordinates` imports an array or a
  DataFrame aligned by cell ID, recording an optional `source` record in its parameters.
  `write_imported_coordinates` records `source` only when given, so Seurat and earlier
  imported coordinates keep their identities.
- `to_anndata` and `to_h5ad` take `cell_selection`, `obsm`, `obs`, `graph`, and `dtype`. Run
  exports write label fields with a categorical display (`clusters`, `leiden_*`, `paris`,
  `cell_cycle_phase`) as categoricals of string labels in natural order, with the display colors,
  where they wrote numbers before, and add the run's PCA as `obsm["X_pca"]` with `varm["PCs"]`.
  Raw `uint8` and `uint16` counts export as `float32` unless `dtype` names another dtype.
- `to_mtx` writes a feature-type column without compression too, one type per assay, formats
  integer counts with NumPy, and compresses with gzip level 6; the matrix text is unchanged.
- `CSVReader` reads its ID column as text and rejects a missing ID. `SparseToZarr` rejects missing
  or repeated cell and feature IDs and converts COO, CSC, and dense input to CSR.
- `run_marker_search` is at operation revision 2, so marker tables of earlier releases are
  recomputed rather than reused. Tables store their statistics at full float64 precision instead
  of rounding them to five decimals, which kept one or two significant digits of ATAC TF-IDF
  means. `MARKER_STAT_COLUMNS` ends with `neg_log10_p_value`, `-log10` of the p-value, taken in log
  space for the normal approximation so that it neither underflows nor ties where `p_value` reaches
  0. A group that can be split from the other cells in at most 100,000 ways gets exact p-values
  from the permutation null of its rank sum, ties included, by the counting kernel that the exact
  Mann-Whitney test of `compare.test` shares; slots record that policy as `p_value_policy`. A group
  of fewer than two cells, or with fewer than two other cells, is skipped with a warning naming it
  instead of failing the search; the slot records each skipped group's cell count in
  `skipped_groups`, its cells stay in the rest of every tested group, and `markers.load` gives it
  no rows. `RankMarkerResult` gains `exact_p_values`, `tested_group_ids`, and
  `skipped_group_ids`.
- Marker tables of revision 1 stay loadable under an explicit read policy. A slot without
  `p_value_policy` must store exactly `MARKER_REVISION_1_STAT_COLUMNS`, the columns before
  `neg_log10_p_value`, and record no `skipped_groups`. `load_marker_table`, `markers.load`,
  `markers.export_csv`, and `marker_heatmap` read it with the five-decimal statistics and the
  asymptotic p-values it stored; `MarkerSlot.p_value_policy` reports
  `MARKER_REVISION_1_P_VALUE_POLICY`, `"asymptotic_tie_and_continuity_corrected"`, which
  `markers.load` logs, and `neg_log10_p_value` is derived as `-log10(p_value)`, `+inf` where the
  p-value underflowed to 0, before the filters and `sort_by` apply. The reuse validator rejects
  that layout, so `markers.search` recomputes such a table at revision 2 on a writable store and
  never reuses it, even under current provenance. Any other slot without `p_value_policy`, and any
  table without `fold_change_policy`, raises `ValueError`.
- `markers.load` and `markers.export_csv` filter by default on `frac_exp >= 0.2`, `auc >= 0.6`,
  and `p_value_adjusted <= 0.05`; `min_score` defaults to `None` because the score falls with the
  number of groups and is capped by a group's size, and `None` disables any filter. They read the
  slot's metadata and feature index once for all groups, and `sort_by` orders rows by `"score"`,
  `"auc"`, `"fold_change"`, or `"p_value"` with deterministic ties, as
  `cytearc.features.sort_marker_results(table, sort_by)` does.
- `compare.compute_bulk` leaves out bulk columns without cells, which it returned as zeros, keeps
  them as NaN with `remove_empty_columns=False`, and with `return_n_cells=True` appends the cells
  of each column. It raises for a `pseudo_reps` below 1, which it treated as 1, and for an unknown
  `feature_label`, which fell back to positions.
- `compare.test` validates and encodes its design once per call, through
  `prepare_group_comparison`, instead of once per key; `compare_group_distributions` gives each
  array the same result, bit for bit.
- `read_gmt` drops a UTF-8 byte order mark. `create_bed_from_coord_ids` parses ids with
  `parse_coordinate_ids`, so chromosome names may contain `:` and other id forms raise
  `ValueError` naming the id. `binary_search` raises `ValueError` for ranges that are not sorted by
  start, for which it returned wrong overlaps.

Compatibility exists only where a current public facade or an explicit file-schema test says it
does. There are no silent migrations, implicit compatibility branches, or forwarding shims for
retired internal modules. Incompatible stores and artifacts fail with an actionable error.

## Placement rules

Use these rules when adding code:

1. Put Zarr mechanics in `storage`, blockwise matrix behavior in `matrix`, and generic operational helpers in `utils`.
2. Keep metadata and assay focused on table access, normalization, and persistence.
3. Put reusable computation in a concrete domain package.
4. Keep domain packages independent of datastore and plotting.
5. Use a named storage adapter when a domain persists an artifact.
6. Put parsing in readers, materialization in writers, and combination in merge.
7. Keep plotting free of datastore imports and use narrow adapters for new storage-backed inputs.
8. Add compatibility only at an existing public facade.
9. Use a concrete biological or computational package name.
   Do not introduce catch-all packages such as `core` or `analysis`.
10. Record a change to what an operation computes as an `OperationRevision` in
    `storage/operation_revisions.py` ([](operation_revisions.md)), with a test of the revision;
    never change recorded parameters to invalidate results.

Architecture boundaries are enforced in `tests/test_import_architecture.py`.
Public imports, result records, facade behavior, and wheel contents have separate contract tests.

## Accepted and deferred decisions

### Accepted

- The datastore class chain remains for public compatibility.
- Same-path package facades remain part of the public architecture.
- A small set of domain modules has narrow storage dependencies for persisted artifacts.
- Marker statistics, statistical-test design, bulk aggregation, and genomic feature construction
  live under `features`.
- Pseudotime-specific feature aggregation and module clustering live under `trajectory.feature_dynamics`.
- `metadata` remains a root data-model package because assays and datastore orchestration both depend on it.
- Unified plotting uses a datastore adapter instead of reading Zarr paths.
- Arrays that a store computes are read through `storage/virtual_arrays.py` only. A virtual array
  may supply its own feature streams, which keep the contract of `storage.feature_stream` and are
  built from its public stream building blocks; composite streams batch reads there instead of
  adding composite cases to the core streams.
- Store-backed plotting is available through the lazy `DataStore.plots` accessor without moving implementation ownership out of `plotting`.
- Old flat compatibility modules remain deleted.
- Operation revisions are the only explicit invalidation mechanism; reuse stays exact on canonical
  provenance, and superseded artifacts remain inspectable.
- D8: `assayTypes` values are semantic labels, such as `RNA`, `ADT`, `HTO`, or `GeneActivity`,
  each naming the preset class that the store declares for an assay. CyteArc records no second type
  attribute, such as a modality or a measurement technology, beside them.

### Deferred

Deferred to a later structural phase:

- The graph and mapping operation modules remain large.
  Splitting them requires a separate behavioral and performance gate.
- `metadata` and `assay` retain established convenience methods, with their domain imports deferred to call time.
- Function-local cycles inside `assay` and `readers` remain accepted because there are no module-load cycles.
- Heatmap plotting still reads Zarr-backed values from duck-typed store and assay inputs.
  Replacing those reads requires a separate plotting adapter design.
- D5 pure open: deferred to 2.0. A writable open still prepares assays, records `assayTypes`
  and `defaultAssay`, and, the first time only, filters `I` by `min_features_per_cell` and records
  the threshold, because the destination contract decides what `overwrite=True` may replace by
  whether a store is prepared. The 2.0 plan is an explicit `prepare_store` that writers call, with
  `min_features_per_cell` replaced by recorded cell selections.
- D7 per-cell panels wait until an operation consumes them.
- D9 persisted ATAC TF-IDF state waits for an ATAC mapping reference, which can recompute it from
  what the normalized artifact records.
- D10 WNN sources keyed by artifact reference wait for a 1.x design; `select_measured_cells`
  already gives one selection that every assay measured.
- D11 a recipe protocol waits until a second pipeline recipe exists.

### Rejected

- A vague `core` or `analysis` package.
- Restoring forwarding modules for retired private import paths.
- Restoring legacy plotting modules or datastore plotting methods.
- Moving storage-aware algorithms into datastore orchestration.

## Implementation references

Read [](zarr_internals.md) for the on-disk implementation boundary and the public API reference for current contracts.
Use [](contributing.md) for the test, documentation, and review workflow before making a change.
