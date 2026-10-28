---
description: Use CyteArc safely in an autonomous or AI-assisted single-cell analysis.
---

(analysis_with_agents)=
# Analysis with AI agents

This page is a routing and reasoning guide for an AI agent that uses CyteArc to analyse data.
It does not replace the workflow tutorials or define one correct analysis.
The study question, experimental design, and user instructions remain authoritative.
For step-by-step recipes, read the `cytearc-single-cell` skill in the repository's `skills/`
directory.

## Scope and authority

An agent may inspect data, run documented public methods, create reversible analysis branches, and compare alternatives.
Report a supported provisional choice and label each statement as:

- a computational fact, such as which artifact or cells produced a result;
- statistical evidence, such as a mixing score or marker AUC;
- a biological interpretation, which may require study context and independent validation.

Do not convert a default, visual pattern, or single metric into biological ground truth.
State assumptions when study metadata are incomplete.
Stop rather than claim an identified treatment, disease, or batch effect when the relevant variables are perfectly confounded.

## How CyteArc represents an analysis

### DataStore, selections, and artifacts

A `DataStore` contains count matrices, cell metadata, feature metadata, and persisted results.
A Boolean {term}`cell key` can define an initial cohort. Snapshot it before analysis so downstream
operations and agents consume an immutable cell-selection artifact. An immutable
{term}`feature selection` artifact selects assay features.
Analytical producers return exact [`ArtifactRef`](/api/artifacts.html#cytearc.ArtifactRef) values and leave metadata
unchanged. Filtering returns a cell-selection artifact without changing live `I` or
deleting counts.

Persisted results are immutable {term}`artifacts <artifact>`.
Their provenance records the operation, scientific parameters, and input artifacts.
A durable [`PipelineRun`](/api/pipeline.html#cytearc.PipelineRun) records one complete recipe invocation and exposes frozen
views over its selections and result fields. Granular workflows retain returned artifact refs and
pass them explicitly.

### Branches and mounts

Alternative results can coexist.
Retain each granular method's returned [`ArtifactRef`](/api/artifacts.html#cytearc.ArtifactRef) and pass that reference to the
next operation. No branch becomes a global implicit result.
See [](tutorials/custom_analyses.ipynb) for a complete example and [](concepts/provenance.md) for the storage model.

Opening a mounted store with another workspace name does not create a separate analysis.
A mount records one source workspace, and reopening that target with a different workspace is rejected.
Create a separate mount target for each analysis that needs independent metadata and results.
Use artifact branches within one target when only parameters or downstream methods differ.
Mount behavior is documented in [](tutorials/remote_stores.ipynb); general layout is documented in [](tutorials/data_organization.ipynb).

## Start or continue safely

### Inspect the current store

Start with `snapshot = ds.summary()`.
It reports the workspace, default assay, resource budget, metadata column names, complete or
incomplete artifacts, pipeline-run counts, and completed run labels without exposing a store
location.
`active_cells` and each assay's `active_features` count the literal `I` columns.
They do not follow a historical artifact or pipeline-run selection.
Resolve and inspect feature-selection artifacts separately when evaluating a branch.
Use `snapshot.to_dict()` for a deterministic JSON-safe record.

Before choosing the next operation:

1. Identify the assays and the current default assay.
2. Inspect cell and feature metadata columns, including active selections.
3. Use `ds.pipeline.list_runs()` and run reports to identify durable recipe invocations;
   `run.recipe` names the recipe of each run.
4. Use `ds.artifacts.list(complete_only=True)` to find persisted alternatives.
5. Use `ds.artifacts.inspect(ref)` before consuming an explicit result.
6. Use `ds.artifacts.lineage(ref)` to verify upstream inputs.

`lineage.to_markdown()` and `lineage.to_mermaid()` produce compact provenance records for an analysis log.
Do not infer state by reading private Zarr paths.
Use public methods instead of mutating `ds.z`, `ds.zw`, or stored run records.

Inspect unfamiliar H5AD files with [`cytearc.inspect_h5ad`](/api/import_export.html#cytearc.inspect_h5ad) before creating a reader.
This reports matrix candidates, encodings, dimensions, and suggested assays instead of relying on guessed keys.
Use the format-specific import guides for other inputs.

### Prospective execution record

Before the first mutating operation, make a short execution record containing:

- the scientific question and unit of inference;
- the cell-selection and feature-selection references, plus other input artifact references;
- the operations that will persist artifacts or export data;
- the alternatives and independent evidence that will be compared;
- the criteria for selecting a branch, preserving uncertainty, or stopping.

Update this record before changing the cohort, inputs, or decision criteria.
This prospective boundary makes unintended writes and retrospective justifications visible.

### When to use the pipeline

The pipeline recipes are persistent baseline workflows, one for each modality and one that
integrates modalities:

- `ds.pipeline.rna()` filters cells on their QC metrics, scores the cell cycle, selects highly
  variable genes, normalizes them, reduces them with PCA, builds a neighbour graph, embeds it with
  UMAP, clusters it with Leiden at several resolutions, scores doublets, and ranks markers.
- `ds.pipeline.atac()` filters cells, selects the most prevalent peaks, normalizes them with
  TF-IDF, reduces them with LSI, and builds, embeds, and clusters a graph.
- `ds.pipeline.adt()` leaves isotype controls out of the antibody panel, normalizes the panel with
  CLR, reduces it with PCA, and builds, embeds, and clusters a graph. It filters cells only when
  asked, because it usually analyzes the QC-passing cells of an RNA run. Its default pattern
  leaves out antibodies whose names contain "isotype", "control", or "ctrl"; check the excluded
  names that the run records against the panel's controls, and set `panel={"exclude": ...}` when
  they differ.
- `ds.pipeline.multimodal(runs)` integrates completed runs of the same cells into one graph, then
  embeds and clusters it.

A recipe checks its call against the store before it writes anything: its keywords and settings,
the columns and cells that it names, the combinations of its stages, and the setting values that
can be checked without reading data. A `TypeError`, `ValueError`, or `KeyError` from the call
therefore means that nothing was written: correct the call and run it again. A few values, such as
`ann_index` `ann_metric`, a Leiden `backend`, and the numeric `doublets` settings, are checked only
when their stage runs, as are limits that depend on the data. A `PipelineExecutionError` names the
stage that failed, and the run is recorded as failed; the stages before it keep their artifacts,
which a corrected run reuses. A run writes immutable artifacts and a durable run ledger, but does
not change live `I` or metadata columns.

Configure a stage with its own keyword. Its settings are the keywords of the stage's method and
mean what they mean there: `hvg={"top_n": 2000}` for `features.hvgs`, `pca={"dims": 30}` for
`reduction.pca`, `neighbors={"k": 21}` for `graph.neighbors`, or
`harmony={"batch_columns": ["sample"]}` for Harmony. An optional stage, such as `cell_cycle`,
`doublets`, `paris`, or `tsne`, takes `False` to skip it, `True` for its defaults, or a mapping of
settings. An omitted setting takes the method's default, except for a few defaults that a recipe
fits to its data, such as an ADT run's PCA dimensions, at most one fewer than its antibodies.
`cells` names the cells to analyze: a boolean cell column (`"I"` by default), a cell-selection
artifact, or a completed run, whose analysis cells are reused as they are. An RNA or ATAC run over
a column filters its cells on their QC metrics by default, and so does an RNA run that calls cells
or demultiplexes hashtags, whatever its `cells`. Otherwise a run over a selection or over another
run's cells filters only when `filtering` asks, and an ADT run filters only when asked.

The single-modality recipes score each Leiden candidate in the coordinates that built the graph
(PCA, LSI, or Harmony coordinates or, with `pca={"dims": 0}`, the normalized values), persist the
decision as `run["cluster_selection"]`, and expose the selected candidate's ref as
`run["clusters"]`. `leiden={"partitions": [...], "selected": r}` saves the candidate at resolution
`r` instead. Silhouette supplies a reproducible baseline, not validation or ground truth. Paris is
off by default; `paris=True` adds a Paris clustering, `run["paris"]`, for diagnosis.

Use `run.cells` and `run.features` for frozen inspection. Keep presentation and storage mutation on
the datastore: `ds.plots.embedding(run=run, ...)`, `ds.markers.load(marker=run["markers"], ...)`,
and `ds.to_anndata(run=run)`. Use an immutable run label when a completed run needs a
human-readable name. Use granular methods and explicit artifact refs for alternatives outside the
recipe stages. For example,
`ds.plots.embedding(layout=embedding_ref, color_by=cluster_ref)` consumes granular outputs without
materializing either as metadata.

### Raw droplets and hashed captures

Two optional RNA stages run before QC filtering, in this order:

- `cell_calling=True` calls cells in an unfiltered droplet matrix with EmptyDrops, so that QC
  filtering and its automatic bounds see cells rather than empty droplets. A matrix of called cells
  needs no cell calling. The stage tests the barcodes of `cells`, which must therefore include the
  empty droplets: open a store of raw droplets with `min_features_per_cell=-1`, which keeps every
  barcode in `I`. Each capture has its own ambient profile and knee, so pass
  `cell_calling={"sample_column": "<capture column>"}` to call the captures of a merged store
  apart.
- `hashing=True` demultiplexes the store's HTO assay (`hashing={"assay": ...}` names one when the
  store has several) and keeps the cells with exactly one hashtag, so negative droplets and
  multiplets of different hashtags leave the run. MAD filtering bounds are then computed within
  each hashtag's cells unless `filtering` names a `sample_column`. Hashing cannot see a doublet of
  two cells with the same hashtag, so keep doublet scoring on.

Report each stage's summary: the knee and the number of called cells, which a run that calls its
captures apart reports for each capture as `kneePerCapture` and `calledPerCapture`, and the cells
of each hashtag with the negatives and multiplets.

### Ambient RNA between two runs

Ambient RNA correction is not a recipe stage. It is an explicit step between two runs, because the
contamination estimate needs clusters of the uncorrected cells and the correction writes a new
assay:

1. Run a first RNA analysis over the called cells, `first = ds.pipeline.rna(...)`, for its
   clusters.
2. Estimate the soup from empty droplets with `qc.ambient_profile`, which needs a store of raw
   droplets.
3. Estimate the contamination of the first run's cells with `qc.ambient_contamination`, passing
   `first["analysis_cell_selection"]`, the profile, `gene_sets`, and
   `clusters=first["clusters"]`. Choose gene sets that some cell types of the sample cannot
   express, such as haemoglobin genes outside erythroid cells or immunoglobulin genes outside B and
   plasma cells.
4. Write the corrected assay with `qc.correct_ambient(estimate, assay_label="RNA_soupx")`.
5. Analyze the corrected counts over the same cells:
   `ds.pipeline.rna(assay="RNA_soupx", cells=first)`.

An estimate of exactly 0 comes with a warning: the gene sets held no counts in the cells that
should not express them, which bounds the contamination but does not measure it. Report such an
estimate rather than correct with it. Compare the markers of both runs: the correction should
lower soup genes in clusters that do not express them, and a cluster that only one run finds needs
its own marker evidence.

### Doublets in a second pass

The RNA recipe scores doublets after it clusters the cells (`doublets=True` by default), but it
removes nothing: the run's graph, clusters, and markers include the predicted doublets. To analyze
singlets, select them and run again over them:

```python
# Select the cells that the first run's doublet scoring calls singlets.
singlets = ds.select_cells(first["doublets"], value="predicted_doublets", include=[False])

# Analyze the singlets again, from feature selection to markers.
second = ds.pipeline.rna(cells=singlets, doublets=False, label="singlets")
```

The second run rebuilds the highly variable genes, PCA, graph, and clusters without the doublets.
With ambient RNA correction, score doublets in the run over the corrected counts and remove them
in a run after it. Doublets form among the cells of one capture whatever their hashtags, so score
them within each capture with `doublets={"sample_column": "<capture column>"}`, never within each
hashtag; a run with Harmony scores within its batches unless `doublets` names another
`sample_column`. Report the predicted doublet rate of each sample from the stage summary and
compare it with the rate that the capture's loading leads one to expect. A doublet of two cells of
one type looks like a singlet and is rarely called. Cells between two populations, such as an
intermediate state, can also score high, so check the markers of a cluster that is rich in
predicted doublets before removing it as an artifact.

### Modalities of the same cells

A multimodal run integrates completed runs that analyze exactly the same cells, and it refuses
runs over different cells. Run the modality whose QC defines the cohort first, usually RNA, and
pass that run as `cells` to the others, which reuse its analysis cells as they are:

```python
# Filter and analyze the RNA cells; this run defines the cohort.
rna = ds.pipeline.rna(label="rna")

# Analyze the antibodies of the same cells.
adt = ds.pipeline.adt(cells=rna, label="adt")

# Integrate the two graphs with WNN, then embed and cluster the joint graph.
joint = ds.pipeline.multimodal([rna, adt], label="rna_adt_wnn")
```

When the cohort should pass the QC of every modality, name the QC metrics of each modality in the
first run's `filtering={"attrs": [...]}`, or pass one QC selection as `cells` to every run. WNN
(`method="wnn"`, the default) weighs each modality for each cell and needs runs whose graphs were
built on PCA, LSI, or Harmony coordinates. Inspect the modality weights, `weight_<assay>` in
`joint.cells` and `meanWeights` in the integration stage's summary: a joint cluster whose cells
rely on one modality is supported by that modality alone. The integrated graph has no shared
coordinates in which to compare Leiden candidates by silhouette, so a multimodal run clusters at
one resolution, 1.0 by default, and several `partitions` need a `selected` resolution.
`joint["markers_<assay>"]` ranks each modality's features for the joint clusters; name a joint
cluster from the markers of every modality.

## Scientific decision loop

Use the following loop for each consequential choice:

1. **Frame the question.** Decide whether the target is population discovery, technical integration, reference mapping, abundance, within-population state, differential expression, or a continuous trajectory.
2. **Establish the unit of inference.** Identify subjects, biological replicates, conditions, batches, paired measurements, and repeated measures.
   Inspect their contingency before correction or statistical comparison.
3. **Keep a baseline.** Evaluate uncorrected data before batch correction and preserve broad, pre-subclustering labels before splitting populations.
4. **Create alternatives.** Change one consequential choice at a time where practical, use explicit artifact references, and keep seeds fixed.
5. **Compare independent evidence.** Combine graph, marker, metadata, replicate, mapping, and method-specific diagnostics.
   Do not optimize one score in isolation.
6. **Decide or stop.** Select a fit-for-purpose branch when evidence is coherent.
   Preserve alternatives and report uncertainty when evidence conflicts.
   Stop when the design cannot identify the requested effect.

## Route by task

Use [API reference](/api/index.html) for exact public signatures and result contracts.
The pages below explain when and how to evaluate those operations.

### Import, quality control, and feature selection

- Use [](tutorials/import_and_export.ipynb) for supported inputs and exports.
- Use [](tutorials/quality_control.ipynb) to inspect modality-specific distributions and derive dataset-specific, often sample-aware selections.
- Use [](tutorials/feature_selection.ipynb) to compare feature sets without treating partition agreement as proof of biological usefulness.

Report retained cells by sample or condition when those columns are available.
Do not copy thresholds from another tissue or protocol without inspecting the current distributions.

### Reduction, graph construction, and clustering

- Use [](tutorials/graph_construction.ipynb) for granular normalization, reduction, neighbour, and graph methods.
- Use [](tutorials/dimensionality_reduction.ipynb) to compare dimension counts and layouts.
- Use [](tutorials/clustering.ipynb) to compare Leiden resolutions, Paris cuts, graph connectivity, membership strength, and marker specificity.

There is no universally optimal partition.
A defensible partition should fit the question, avoid being driven solely by technical covariates, retain replicate support, and have interpretable positive and negative evidence.
A split population should rebuild feature selection and its graph inside the subset.
Preserve the parent labels so the split remains auditable.

### Integration, mapping, and paired modalities

Start with [](tutorials/dataset_merging.ipynb) when compatible datasets need one joint datastore, then use [](tutorials/batch_correction.ipynb) only when a defensible technical covariate should be reduced.

- Merge without correction when compatible datasets need joint inspection but no technical effect has been identified.
- Compare an uncorrected graph with partial PCA or Harmony when a defensible technical covariate should be reduced.
- Use fixed-reference mapping when queries must remain comparable to one reference over time.
- Use WNN by default for modalities measured in the same cells. Use SNN only when equal graph
  support is the intended comparison. Neither method integrates independent batches.
  `ds.pipeline.multimodal(runs, method="wnn")` or `method="snn"` builds either one from modality
  runs over the same cells (see [Modalities of the same cells](#modalities-of-the-same-cells)).

The batch-correction workflow compares source mixing and structural preservation.
CyteArc's `metric_*` methods provide evidence, not an automatic winner.
Never correct a variable that is indistinguishable from the condition of interest and then claim the condition was preserved.

### Annotation, contrasts, and differential expression

Use [](tutorials/annotation.ipynb) to combine marker specificity, AUC, expression fraction, and known positive and negative markers.
Identify mixed populations explicitly and retain uncertain labels.

Marker search treats cells as observations.
For condition-level inference, use [](tutorials/pseudobulk_and_differential_expression.ipynb) to aggregate by biological replicate and export counts to a replicate-aware statistical method.
Descriptive pseudo-replicates made by splitting the same cells are not independent biological replicates.

Separate three questions that require different evidence:

- whether population abundance changes across samples;
- whether expression changes within a defined population;
- whether the population definition itself changes.

### Pseudotime and fate mapping

Use pseudotime only when the graph and biological question support a plausible continuous process.
Source and sink labels supervise the orientation; CyteArc does not discover terminal states.

- Use [](tutorials/pseudotime.ipynb) for source and sink scoring.
- Use [](tutorials/expression_dynamics.ipynb) for expression dynamics.
- Use [](tutorials/fate_mapping.ipynb) for multiple terminal outcomes.
- Use [](tutorials/lineage_dynamics.ipynb) for genes along each lineage, drivers of commitment, and lineage modules.
- Use [](tutorials/lineage_curves.ipynb) for lineage curves, placing points on them, and candidate sinks from a cluster tree.
- Use [](tutorials/trajectory_validation.md) to compare boundaries, graph components, marker tests, modules, and fate-probability validity.

Compare plausible source and sink definitions when the endpoints are uncertain.
Check validity keys, graph components, terminal probabilities, and expected marker trends.
When a process branches, prefer lineage markers or drivers to a pooled pseudotime for fate-specific genes, and leave the endpoint cells out of driver searches.
Treat cluster-tree leaves as candidate sinks to check with markers; a tree can chain similar endpoints.
A pseudotime or fate artifact is a model-based summary, not evidence of causal lineage.

### Regulons, chromatin activity, and gene programs

Use [](tutorials/regulons.ipynb) for RNA regulons, motif and binding-peak activity in ATAC
cells, and co-regulated gene programs.
Treat every regulon as a hypothesis about regulation, not as evidence of binding or causality.

- A SCENIC regulon combines co-expression with motif enrichment near target genes.
  GRNBoost2 is stochastic: rerun `regulons.grn` with a second `random_seed` and report regulons
  and targets that persist across seeds.
- Use a cisTarget database for the same species and gene naming as the expression data, and a
  JASPAR or cisTarget motif collection with the factor annotations it was built for. Module
  genes missing from the database are skipped; check how many modules survive.
- The Aerts lab's cisTarget databases and motif tables state no license terms, and the v10
  motif collection includes motifs derived from TRANSFAC Professional, a commercial database.
  Confirm their terms before commercial work.
- chromVAR measures accessibility of every peak that matches a motif. Paralogous factors share
  motifs, so a motif deviation names a motif family; pair it with the factor's expression or a
  binding experiment before naming one factor. Cells that expect fewer than one fragment in a
  set have no value.
- CUT&RUN, CUT&Tag, and ChIP-seq peaks measure binding in the profiled cells. Peaks from a cell
  line or another cell type describe that context; remove blacklisted regions before mapping
  them onto your peaks.
- Region-to-gene links need RNA and ATAC from the same cells. Links from metacells are
  associations, so report them as candidate enhancers.
- Gene programs group genes that co-vary across metacells. Build them from independent samples
  (metacells or pseudobulk), never from smoothed cells, whose shared counts inflate correlations;
  the membership threshold screens chance groupings but is not a calibrated test of a program.
  Factors nominated by motif enrichment of a program are candidates to test, not its
  demonstrated regulators.

### Pooled CRISPR screens

A pooled screen labels each cell by the guides it carries, so every downstream comparison depends on guide calls. Call guides from the guide counts with `perturbation.call_guides`, or import Cell Ranger's calls with `perturbation.import_guide_calls`, and inspect `plots.guide_calls` before labelling cells by perturbation with `perturbation.labels`, which names the genes that each cell's guides target.

- Call with the default `method="poisson_gaussian"` unless the study needs another method. The Poisson-Gaussian model fits each guide to the cells that hold any of its UMIs, as a Poisson background of ambient molecules and a Gaussian of the cells that carry the guide (Replogle et al. 2022), so cells without a UMI of the guide do not shape its threshold, and a cell can carry any number of guides.
- `method="gaussian"` reproduces Cell Ranger's calls. The two-Gaussian fit includes the cells without a UMI of the guide, so when few cells carry each guide the fit can split those cells from the cells with ambient UMIs and call the latter. A warning names this case when more than half of the cells with a call have several guides; inspect the fits with `plots.guide_thresholds` before using such calls, or call with `poisson_gaussian`.
- `method="fixed"` calls each guide from one UMI floor; choose the floor before seeing the calls, such as one that a published analysis used.
- `method="top_guide"` suits a low-MOI screen where each cell should carry one guide. A cell whose top guide holds too small a share of its guide UMIs gets the label `"Negative"` from `top_guide`, while SCEPTRE (Barry et al. 2021) flags such a cell as holding several guides; tell these cells from cells with too few guide UMIs by `top_fraction` and `total_umi`, and do not count them as untransduced. Perturbation labels keep them `"Negative"`.
- Check the fits before using fitted calls. The warning of `call_guides` counts the fits that call no cell, `load_guide_thresholds(calls).table` lists the status and threshold of each guide in each sample, and `plots.guide_thresholds(calls, guides=[...])` shows a guide's counts, the posterior of the cells that carry it, and its threshold. A guide whose fits call no cell has no calls, which does not show that no cell carries the guide.
- Check the multiplicity of infection (MOI) that the design intended. In a low-MOI screen most called cells carry one guide, and cells with several calls are droplet doublets or cells with several integrations; analyse the cells with one called guide, which `select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)` selects. Dual-guide constructs and high-MOI screens put several guides in each cell by design.
- Every cell holds a few ambient guide molecules. A cell called `"Multiple"` whose top guide holds nearly all of its guide UMIs (`top_fraction`) may carry one guide plus ambient molecules of others; report how many cells this affects instead of reassigning them.
- A cell called `"Negative"` has no confident guide: its guide was not captured, or the cell was not transduced. Negative cells are not controls. Controls are the cells that carry control guides only, which `perturbation.labels` labels `"control"` from the control targets it is given.
- Background guide levels differ between droplet runs. In a merged store, import each run's call file with its own `barcode_prefix` and a cell selection of that run. `call_guides` fits each run separately with `sample_column`; use `sample_column` only when each run holds enough cells that carry each guide, since a fit sets the guide's threshold from those cells. Otherwise fit all cells together, as Replogle et al. (2022) did after evening the sequencing depth across GEM groups.
- Name the controls from the guide assay's target column, which a 10x import fills from Cell Ranger's feature reference: read the distinct targets of `ds.CRISPR.feats.fetch_all("target_gene_name")` and pass the exact names, such as `controls=["Non-Targeting"]`. CyteArc never guesses controls. A library can hold several kinds of control guide; in a knockout screen, safe-targeting guides cut the genome at sites without a known function while non-targeting guides do not cut, so name the kinds that the study designed as its control group. Pass `targets` for guides whose target the column misses or names wrongly.
- Choose the policy from the design. The default `policy="single"` labels a cell whose guides target several genes `"Multiple"`, which suits a low-MOI screen, where such cells are mostly doublets or cells with several integrations. `policy="combinations"` labels such a cell with its genes, such as `"GENE1+GENE2"`, for a screen designed to perturb gene pairs. In a high-MOI screen each cell carries several guides at random, so most combinations hold few cells; analyse each gene across the cells whose guides target it, from each cell's genes in `load_labels(labels).targets`, which both policies keep.
- Compare each gene's cells with the `"control"` cells, and keep `"Multiple"` and `"Negative"` cells out of per-gene comparisons: a `"Multiple"` cell carries guides of several genes, and a `"Negative"` cell has no confident guide, so neither belongs to one gene or to the controls. Under `"combinations"`, keep the combination labels out of single-gene comparisons too.
- Labels merge the guides of each gene. A cell with two guides of one gene, or with a gene's guides and control guides, gets the gene's label, while its guide calls say `"Multiple"`. Merging changes the counts most where cells carry several guides: dual-guide constructs whose two guides target one gene, and screens at a higher MOI, where a gene's guide can sit beside a control guide or a second guide of the same gene. Report how many cells the merge moved by comparing `guide_label` with `label` in `load_labels(labels).cells`.
- A guide can fail to edit its target, and the cells that carry it then look like controls. A gene's label pools the cells of all its guides, so compare the guides of a gene before trusting a gene-level effect; `select_cells(calls, include=["GUIDE-1"])` selects the cells of one guide.
- Cells from one droplet run are not independent replicates. Test differential expression against controls across replicates, such as runs or donors, by pooling cells per perturbation label and replicate with `compare.compute_bulk`. The labels belong to the guide assay, so name the expression assay with `from_assay`, and leave out `"Negative"` and `"Multiple"` with `null_vals`.

### Methods outside CyteArc

Use [](tutorials/custom_analyses.ipynb) for block streams, graph access, custom selections, external reductions, and supported export paths.
Export when the question needs RNA velocity, scVI, peak calling, FRiP/TSS, or another method outside CyteArc.
Do not write arbitrary artifact groups directly.

## Troubleshooting

Classify the problem before retrying:

- **Input or schema:** inspect the source format and verify assay, feature, and cell identifiers.
- **Run or provenance:** inspect the pipeline report (each stage's status, summary, and warnings),
  stored cell selection, feature-selection refs, artifact status, and lineage.
  `ArtifactResolutionError.code` distinguishes missing, incomplete, or changed selection inputs.
  Do not consume incomplete artifacts.
- **Resource or I/O:** inspect the configured memory budget, worker count, storage profile, and remote latency.
  If an RNA store fails to open, treat it as a missing or outdated count layout and rebuild or `repack_zarr` rather than retrying the analysis stage.
  See [](concepts/memory_and_execution.md), [measurement guidance](concepts/memory_and_execution.md#measuring-resource-use), and [](tutorials/remote_stores.ipynb).
- **Numerical or graph:** check dimensions, graph components, neighbour count, convergence, validity masks, and method-specific diagnostics.
- **Scientific ambiguity:** preserve branches, seek another independent form of evidence, narrow the claim, or report that the available design does not resolve the alternatives.

In a granular workflow, retry the lowest failed stage. A failed, interrupted, or cancelled pipeline
run is not resumable; start a new run, which can reuse matching complete artifacts from the earlier
attempt.

## Progress, cancellation, and deterministic comparisons

Every recipe takes `callback=`, which receives each
[`PipelineEvent`](/api/pipeline.html#cytearc.datastore.pipeline_accessor.PipelineEvent) of the run,
and `with cytearc.pipeline_listener(listener):` sends the events of every run started in its block
to `listener`. Use either to update an external progress record. A run delivers its events in
order, on the thread that runs it: `run_started`; for each stage, `stage_started`, any
`stage_progress` events, and then `stage_completed`, `stage_failed`, or `stage_interrupted`, or
only `stage_skipped` for a skipped stage; and last `run_completed`, `run_failed`, or
`run_interrupted`. While a stage runs, CyteArc's progress bars send throttled `stage_progress`
events, even when progress display is off. `event.to_dict()` returns the event as JSON values for
a log. A callback or listener that raises is logged and ignored, so it cannot stop or fail a run;
a slow one delays the run, so hand events to a queue instead.

To stop a run, create a token, `token = cytearc.CancellationToken()`, pass it as `cancel=token`,
and call `token.cancel("reason")` from any thread, such as on a user's stop request. The run stops
at its next safe checkpoint and is recorded as interrupted with the interruption kind `cancelled`:
a stage whose work the cancellation cut short is recorded as interrupted, and one whose work had
finished as completed. The call raises `cytearc.PipelineCancelled`, which names the run, the stage
whose work was cut short, or None when no stage's work was, and the reason. A token that is
already cancelled stops the next run before it writes anything. Completed stages keep their
artifacts, so a new run with the same settings reuses them. Do not interrupt a run by mutating the
store or its records.

The durable report, `run.report()` or `run.report(format="markdown")`, records each stage's status,
timing, sampled process-tree RSS, exact created or reused artifact plans, a summary of its outputs,
and the warnings that CyteArc logged while it ran. Read the summaries before interpreting a run:
the cells that filtering removed, the variance that PCA explains, each Leiden candidate's number of
clusters and their largest and smallest sizes, the silhouette scores of cluster selection, and the
doublet rate of each sample. Report every stage warning with the result it qualifies. Granular
analytical methods return artifact references.

When comparing branches, retain the same cells, features, neighbour settings, and seeds unless one is the variable being tested.
Two runs that differ in one stage keyword, or only in their `snapshot_columns`, share every artifact
that does not depend on that difference, so their difference isolates that choice: runs with and
without Harmony share their filtered cells, highly variable genes, and PCA.
Record every deliberate difference.
A layout can vary in orientation or spacing without representing different biology.

## Analysis handoff

A useful handoff reports:

- the scientific question and unit of inference;
- the store workspace, exact cell selection, resolved feature selections, and relevant artifact refs;
- metadata roles and any confounding;
- alternatives considered and the evidence used to compare them;
- the selected result and why it is fit for the question;
- unresolved uncertainty, unsupported claims, and required external validation;
- exported files or external methods that continue the analysis.

Artifact provenance records how CyteArc produced a result.
It does not replace this study-level reasoning record.
See [](reference/methods.md) for the implemented methods and current boundaries.
