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

The pipeline recipes are persistent baseline workflows, one for each modality, one that
integrates modalities, and one for pooled CRISPR screens:

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
- `ds.pipeline.perturbation(labels={"controls": [...]})` calls the guides of a pooled CRISPR screen, labels each cell by the genes its guides target, and measures each perturbation over the RNA recipe's highly variable genes and PCA ([Pooled CRISPR screens](#pooled-crispr-screens)).

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

A pooled screen labels each cell by the guides it carries, so every downstream comparison depends on guide calls. Call guides from the guide counts with `perturbation.call_guides`, or import Cell Ranger's calls with `perturbation.import_guide_calls`, and inspect `plots.guide_calls` before labelling cells by perturbation with `perturbation.labels`, which names the genes that each cell's guides target. Then measure with `perturbation.efficiency` whether each perturbation changed its own target gene, and with `perturbation.edistance` how far it moved its cells from the control cells. Use [](tutorials/crispr_screens.ipynb) for a screen analyzed from download to results, the [Perturbation API reference](/api/perturbation.html) for each method's settings, checks and records, and the [perturbation recipe](/api/pipeline.html#perturbation) for the stages of one run.

- Run the analysis as one durable run with `ds.pipeline.perturbation(labels={"controls": [...]})` once the guide assay, the control targets and the expression assay are known. The run calls the guides, labels the cells, measures each target's efficiency and E-distance and, on request, Mixscape's classes, the association tests over the highly variable genes, and the clusters of the profiles; it records every setting and each stage's numbers, reuses what earlier calls computed, and refuses settings that a stage would refuse before it writes anything. Give it the cells and the RNA settings of the RNA run that the analysis already uses, such as `cells=rna_run` with that run's `hvg`, `normalization`, `pca` and `harmony`, so that its E-distances and signatures read that run's PCA. Call the methods one by one when a step needs a choice that the recipe does not offer, such as a table of pairs to test, and read the stage summaries of `run.report()` in the order of the checks below before any result. In a high-MOI screen, pass `associations={"moi": "high"}` with `efficiency=False` and `edistance=False`, and leave `mixscape` and `clusters` off: the run refuses the stages that compare each perturbation with control-only cells.
- Call with the default `method="poisson_gaussian"` unless the study needs another method. The Poisson-Gaussian model fits each guide to the cells that hold any of its UMIs, as a Poisson background of ambient molecules and a Gaussian of the cells that carry the guide (Replogle et al. 2022), so cells without a UMI of the guide do not shape its threshold, and a cell can carry any number of guides.
- `method="gaussian"` reproduces Cell Ranger's calls. The two-Gaussian fit includes the cells without a UMI of the guide, so when few cells carry each guide the fit can split those cells from the cells with ambient UMIs and call the latter. A warning names this case when more than half of the cells with a call have several guides; inspect the fits with `plots.guide_thresholds` before using such calls, or call with `poisson_gaussian`.
- `method="fixed"` calls each guide from one UMI floor; choose the floor before seeing the calls, such as one that a published analysis used.
- `method="top_guide"` suits a low-MOI screen where each cell should carry one guide. A cell whose top guide holds too small a share of its guide UMIs gets the label `"Negative"` from `top_guide`, while the maximum rule of the [SCEPTRE book](https://timothy-barry.github.io/sceptre-book/assign-grnas.html) flags such a cell as holding several guides; tell these cells from cells with too few guide UMIs by `top_fraction` and `total_umi`, and do not count them as untransduced. Perturbation labels keep them `"Negative"`.
- Check the fits before using fitted calls. The warning of `call_guides` counts the fits that call no cell, `load_guide_thresholds(calls).table` lists the status and threshold of each guide in each sample, and `plots.guide_thresholds(calls, guides=[...])` shows a guide's counts, the posterior of the cells that carry it, and its threshold. A guide whose fits call no cell has no calls, which does not show that no cell carries the guide.
- Check the multiplicity of infection (MOI) that the design intended. In a low-MOI screen most called cells carry one guide, and cells with several calls are droplet doublets or cells with several integrations; analyse the cells with one called guide, which `select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)` selects. Dual-guide constructs and high-MOI screens put several guides in each cell by design.
- Every cell holds a few ambient guide molecules. A cell called `"Multiple"` whose top guide holds nearly all of its guide UMIs (`top_fraction`) may carry one guide plus ambient molecules of others; report how many cells this affects instead of reassigning them.
- A cell called `"Negative"` has no confident guide: its guide was not captured, or the cell was not transduced. Negative cells are not controls. Controls are the cells that carry control guides only, which `perturbation.labels` labels `"control"` from the control targets it is given.
- Background guide levels differ between droplet runs. In a merged store, import each run's call file with its own `barcode_prefix` and a cell selection of that run. `call_guides` fits each run separately with `sample_column`; use `sample_column` only when each run holds enough cells that carry each guide, since a fit sets the guide's threshold from those cells. Otherwise fit all cells together, as Replogle et al. (2022) did after evening the sequencing depth across GEM groups.
- Name the controls from the guide assay's target column, which a 10x import fills from Cell Ranger's feature reference: read the distinct targets of `ds.CRISPR.feats.fetch_all("target_gene_name")` and pass the exact names, such as `controls=["Non-Targeting"]`. CyteArc never guesses controls. A library can hold several kinds of control guide; in a knockout screen, safe-targeting guides cut the genome at sites without a known function while non-targeting guides do not cut, so name the kinds that the study designed as its control group. Pass `targets` for guides whose target the column misses or names wrongly.
- Choose the policy from the design. The default `policy="single"` labels a cell whose guides target several genes `"Multiple"`, which suits a low-MOI screen, where such cells are mostly doublets or cells with several integrations. `policy="combinations"` labels such a cell with its genes, such as `"GENE1+GENE2"`, for a screen designed to perturb gene pairs. In a high-MOI screen each cell carries several guides at random, so most combinations hold few cells; analyse each gene across the cells whose guides target it, from each cell's genes in `load_labels(labels).targets`, which both policies keep.
- Compare each gene's cells with the `"control"` cells, and keep `"Multiple"` and `"Negative"` cells out of per-gene comparisons: a `"Multiple"` cell carries guides of several genes, and a `"Negative"` cell has no confident guide, so neither belongs to one gene or to the controls. Under `"combinations"`, keep the combination labels out of single-gene comparisons too.
- Labels merge the guides of each gene. A cell with two guides of one gene, or with a gene's guides and control guides, gets the gene's label, while its guide calls say `"Multiple"`. Merging changes the counts most where cells carry several guides: dual-guide constructs whose two guides target one gene, and screens at a higher MOI, where a gene's guide can sit beside a control guide or a second guide of the same gene. Report how many cells the merge moved by comparing `guide_label` with `label` in `load_labels(labels).cells`.
- Check that each perturbation changed its own target gene before reading its effects on other genes: `perturbation.efficiency(labels)` compares each target gene's expression in the cells of its perturbation with the `"control"` cells. Read `remaining_fraction`, the share of the control expression left in the perturbed cells after scaling each cell to a common library size, rather than the ratio of `mean_umi` to `control_mean_umi`, which also follows each group's sequencing depth. CRISPR interference should leave less than 1, such as 0.2 for an 80% knockdown, and CRISPR activation more than 1. A Cas9 knockout lowers its target's transcripts mainly when nonsense-mediated decay degrades the edited transcripts, so in a knockout screen a fraction near 1 does not show that the edit failed. Report perturbations whose target did not move as unconfirmed, not as perturbations without an effect.
- A guide can fail to edit its target, and the cells that carry it then look like controls. A gene's label pools the cells of all its guides, so compare the guides of a gene before trusting a gene-level effect: `perturbation.efficiency(labels, by="guide")` measures each guide on its target gene, and `select_cells(calls, include=["GUIDE-1"])` selects the cells of one guide.
- Weigh each row of the efficiency table by its cells and by the expression of its gene. Few cells (`n_cells`), or a gene with few UMIs in the control cells (`control_mean_umi`), give a noisy fraction and a test with little power. A `log2_fold_change` of minus infinity means that the perturbed cells hold no UMI of the gene, which can happen by chance when the gene is lowly expressed and the perturbation has few cells; read it with `n_cells` and `adjusted_p_value`. The p-values treat cells as independent, which suits this check of a perturbation's own target; test effects on other genes across replicates, as below.
- In a row with the status `"no_control_expression"`, the control cells hold no UMI of the target gene, so the row has no fraction. A knockdown of such a gene cannot be measured, since the cell type may not express the gene at all; in an activation screen, a `mean_umi` above 0 with a small `adjusted_p_value` shows that the gene was switched on.
- Check each `"no_feature"` row, whose target matched no feature of the expression assay, neither by the feature ID that `id_column` gives its guides (`target_gene_id` in a 10x import) nor by a unique feature name equal to the target; the warning of `efficiency` counts these rows and names a few. An ID from another annotation release, a gene symbol that the expression assay spells differently or holds twice, and a target that is not a gene, such as an enhancer, leave a row unmatched. Fix the match with `id_column`, or with the `targets` of `perturbation.labels`, rather than reporting the perturbation as ineffective. In a screen of enhancers or other non-coding regions, measure each target's effect on nearby genes with a differential expression test instead.
- Efficiency compares single-target cells only: the cells whose guides target one gene, with or without control guides, and with `by="guide"` the cells whose one called guide is that guide. `"Multiple"` cells, combination labels and `"Negative"` cells stay out. In a dual-guide screen, where each cell carries two guides of one gene, few cells have one called guide, so measure by target. Report `n_cells` beside each fraction, and in a high-MOI screen, where few cells carry one target alone, measure a gene across all the cells whose guides target it, from `load_labels(labels).targets`.
- Rank perturbations by the strength of their effect on the cells with `perturbation.edistance(labels, coordinates)`, which compares the cells of each perturbation with the `"control"` cells over the whole space of the coordinates. Use a space fitted on all the cells of the screen, perturbed and control alike, such as the PCA of a standard run, so that it holds the directions in which perturbations move cells. E-distances depend on the dimensions and scale of the space, so compare them within one table, or between tables measured on the same coordinates, never across datasets or spaces.
- When the cells come from several runs or lanes and the perturbations are spread unevenly across them, differences between runs also move cells. Measure on coordinates corrected for the runs, such as a `batch_correction` artifact, so that a perturbation sequenced mostly in one run does not look distant because of its run.
- Read each E-distance with its `n_cells` and `adjusted_p_value`. Few cells give a noisy distance, which can fall far below 0 for a few widely spread cells, and a test with little power, and a row's smallest p-value is 1 / (`n_permutations` + 1), so raise `n_permutations` when many rows are tested and the smallest adjusted p-values sit at that floor. The E-distance measures how far a perturbation moves the centre of its cells, so a perturbation that only widens or splits its cells without moving their centre, or that changes a few genes strongly without moving far in a PCA, scores low: a small E-distance does not show that a perturbation had no effect. A perturbation whose target gene did not move in `efficiency` but whose cells moved far can point to a guide that acts away from its target; compare its guides with `by="guide"` before reporting the effect.
- The permutation test treats cells as exchangeable, which suits ranking the perturbations of one screen. Report effects that must hold across runs or donors from tests across replicates, as below.
- In a knockout screen, compute Mixscape's perturbation signature with `perturbation.signature(labels, coordinates)` before asking which cells a perturbation changed. Use the PCA that the analysis already uses, fitted on perturbed and control cells alike, with `dims` up to its components (Seurat's vignette uses 40), and pass the replicate or lane column as `sample_column` when the runs differ, so that each cell is compared with control cells of its own run. Keep `n_neighbors` (20) well below the control cells of the smallest sample; a sample with fewer control cells than `n_neighbors` is refused, and a sample with barely more gives cells almost the same neighbors.
- The signature subtracts what a cell shares with nearby control cells, such as its cell-cycle phase, so a perturbation that moves cells along a direction where control cells also vary, for instance into another cell-cycle phase, can lose part of its effect in the signature. Read the expression of the perturbed cells as well before concluding that a perturbation changed nothing.
- Call knocked-out cells and escapers with `perturbation.mixscape(signature)` in a knockout screen, where some cells keep their target's function, and measure a target's effects on its KO cells, `select_cells(classes, include=["GENE1 KO"])`, so that escapers do not dilute them. Report each target's KO share from `load_mixscape(classes).targets` beside its efficiency: a target with a low KO share and a remaining fraction near 1 likely failed, while a high KO share confirms the edit.
- A `"few_de_genes"` target has fewer DE genes than `min_de_genes` against the control cells, so Mixscape calls all its cells escapers; this means no detectable effect, not proof that the edit failed. A `"few_cells"` target had fewer KO cells than `min_cells`. In a CRISPR interference or activation screen, where every cell's target moves by degree, prefer the efficiency and E-distance of each target, and use Mixscape's classes to describe responders rather than to discard cells.
- Mixscape's DE genes treat cells as independent and its posterior assumes equal priors, so borderline cells can change class between settings. Read `posterior` near 0.5 as uncertain, and compare KO shares rather than single cells.
- Ask which genes each perturbation changes with `perturbation.associations(labels, features=hvg)`, which tests every target against every gene of a feature selection, or pass a table of `target` and `feature` pairs for a focused question, such as a pathway's genes. Pass the replicate or lane column as `sample_column`, and add as `covariates` any numeric cell column, such as the mitochondrial share, that differs between guides for reasons other than the perturbation.
- In a high-MOI screen, where each cell carries several guides by design, test with `moi="high"`: a target's cells are all the cells whose guides target it, whatever else they carry, compared with every other labelled cell, and the covariates add each cell's log detected guides and log guide UMIs, so every labelled cell needs guide UMIs; call guides on `select_cells(calls, value="total_umi", low=1, keep_bounds=True)` when some cells lack them. Keep the default `resampling="conditional"`, which stays valid when sequencing depth drives both guide detection and expression. When the calibration grade is moderate or severe, consider covariates the formula lacks, such as cell type, and a higher `min_nonzero`, such as 15; with few pairs, test again with `approximation="none"`, which takes each p-value from the random sets alone, and `resampling="permutation"` for speed, as the [SCEPTRE book](https://timothy-barry.github.io/sceptre-book/) suggests.
- For a CRISPRi screen of enhancers, pass a table of each candidate enhancer with the genes near it, taken from all expressed genes, since an enhancer's genes need not be highly variable, and `side="left"`, since CRISPRi lowers the genes that an enhancer drives. An enhancer is not a gene and has no positive control of its own; guides that target transcription start sites give the positive controls. A test of every target against every gene costs about 499 random sets for each pair, each summing the target's cells, so in a screen of thousands of targets keep trans tests to the targets and genes that a question needs.
- Read the calibration check before the discoveries. `load_associations(tests).calibration_grade` grades the false discoveries among negative-control pairs, groups of control guides tested against the other control cells, or at high MOI against all other cells; it should be `"excellent"` or `"mild"`. A `"moderate"` or `"severe"` grade means the test is miscalibrated for this screen, for instance because a control guide behaves unlike the others or a covariate the formula lacks separates the cells, so fix the covariates or drop the guide before trusting any discovery. Count the false discoveries by their `control_guides` in `load_associations(tests).calibration`: when they concentrate in a few control guides, as large shifts of single genes, those guides act on genes of their own, through an off-target effect for instance, rather than the test being miscalibrated; leave them out of the controls by leaving out the cells that called them, and test again. Do not give them a target of their own instead, which would make every later stage test them as perturbations. [Methods and limitations](reference/methods.md#interpreting-the-comparisons) reports such guides in Replogle et al.'s K562 screen.
- Read the positive controls, each target with its own gene, as the power check: in a knockout or CRISPRi screen most targets should lower their own gene. A target whose own gene does not move had a weak perturbation, or its gene is too sparse to test; compare it with its efficiency before reading its trans effects as absent.
- The fold changes come from the pair's own fit, as SCEPTRE's do, so they shrink toward 0 when a target has many cells next to the controls. Use them for the direction and the ranking of effects, and the p-values for significance; for effect sizes to report, compare the target's cells with the control cells directly, as `efficiency` does for the target gene.
- Testing every target against thousands of genes makes many tests, and the Benjamini-Hochberg adjustment spans them all: restricting the features to the highly variable genes is a choice that changes which effects pass. Say which features were tested, and treat a target's discoveries as a set of affected genes rather than as proof about any single gene. With `integration="singleton"`, each guide is tested alone, which finds guides that disagree with the other guides of their target.
- Ask which perturbations act alike with `perturbation.profiles(labels, features)` and `perturbation.cluster(profiles)`. Pass the GEM group or lane column as `sample_column`, since each sample's control cells set its z-scores. Profile Mixscape classes in a knockout screen so that escapers do not dilute the profiles, and profile by guide (`by="guide"`) to check that a target's guides agree before reading its cluster. Mixscape classes keep only the targets it could fit, so in a small screen, or one where few targets have knocked-out cells, profile every labelled cell (`profiles={"cells": "labelled"}` in `pipeline.perturbation`) and restrict the clustering to the targets with a clear effect instead.
- Cluster only perturbations with a clear effect: weak perturbations have flat profiles whose correlations are noise and land in clusters by chance. Pass `perturbations`, such as the targets with a significant E-distance and a remaining fraction below about 0.7, and say which filter chose them. Replogle et al. required 50 differentially expressed genes, 25 cells and 30% knockdown, so a different filter gives a different set.
- Read a cluster as a hypothesis that its perturbations share a function, such as a complex or a pathway, and check it against known biology before reporting it. HDBSCAN leaves many perturbations unassigned (-1), which means no dense group, not no effect; their `probability` of 0 says so, and a low probability marks a member at a cluster's edge. Correlation ignores the strength of an effect, so pair a cluster with the efficiency and E-distance of its members.
- Cells from one droplet run are not independent replicates. Test differential expression against controls across replicates, such as runs or donors, by pooling cells per perturbation label and replicate with `compare.compute_bulk`. The labels belong to the guide assay, so name the expression assay with `from_assay`, and leave out `"Negative"` and `"Multiple"` with `null_vals`.

### Clones and lineage tracing

A clone is the set of cells that descend from one ancestor cell. Import the clones that a lineage tracing tool assigned with `clones.import_clones`, or take them from a cell metadata column with `clones.from_column`, and check the artifact's count of unmatched barcodes before measuring how the clones expand, how diverse each sample is, which clones groups share, which fates are coupled and which fates the clones of early cells took.

- Measure expansion within each sample with `clones.expansion(clones, group_by=sample_column)`: a clone size is then a count of one sample's cells. Without `group_by` the cells of a clone in several samples add up, which suits a single sample or a question about the whole population.
- Keep the default bounds in cells to compare samples of different sizes. Frequency bounds, such as scRepertoire's `(0.0001, 0.001, 0.01, 0.1, 1)`, depend on the sample's size: one cell of a sample of 1,000 cells already has the frequency 0.001, so compare frequency categories only between samples of similar size.
- Expansion categories are per-cell labels. Colour the embedding by them and draw `plots.clonal_expansion(expansion, group_by=cell_type_column)` to see which cell types hold the expanded cells. Select the cells of large clones for differential expression with `select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])`.
- Read the clonal space of each sample from `load_expansion(expansion).rank_windows` or the plot's second panel: the share of its cells that its 10 largest clones hold measures how much a few clones dominate it. A shallow sample has fewer clones to rank, so compare samples of similar depth.
- Measure diversity with `clones.diversity(clones, group_by=sample_column)` and compare samples by the Hill numbers at a common sample coverage (`hill.at_coverage`) or a common size (`hill.at_size`), not by observed indices: a sample with more cells finds more clones, so its observed richness and Shannon entropy are higher whatever its clonal structure.
- Draw `plots.clonal_rarefaction` before trusting any comparison. Curves that still rise steeply at their observed point are far from complete, as when most clones are singletons: their richness and Chao1 are poorly determined, while the Hill numbers of order 2, dominated by expanded clones, are stable. Report orders 1 and 2 when coverage is low.
- Chao1 has two common forms: `chao1`, iNEXT's and immunarch's, and `chao1_bias_corrected`, vegan's and scRepertoire's. Name the one you report. ACE is undefined when every rare clone is a singleton.
- A group far smaller than the others sets the default common size and coverage low: leave such groups out with `cell_selection`, or pass `size` or `coverage`. Diversity within clusters measures the clonal structure of each cell state, but a cluster of few cells is mostly singletons whatever its biology.
- Compare samples, tissues or cell types with `clones.overlap(clones, group_by=column)`. Report the Morisita-Horn index for how much the same expanded clones dominate two groups, and `shared_fraction_1` and `shared_fraction_2` for how much of each group's cells belong to clones the other holds. Presence indices such as Jaccard grow with both groups' cells, so compare them only between pairs of similar depth.
- Follow clones across time points with `plots.clone_tracking(clones, group_by=time_point_column, order=[...])`, and across cell types to find clones whose cells took several fates. A clone found in one group only may be below detection in the others, not absent.
- Measure which cell types share clones with `clones.coupling(clones, group_by=cell_type_column, cell_selection=late_cells)`, on the cells of one time point. Read `z_score` as a ranking: shuffled labels spread each clone over every cell type, so when clones commit to few fates most pairs fall below their null, and the coupled pairs are those with the highest z-scores. Keep the default `share_cosine`, which counts every clone once, so that a few large clones do not set a pair's coupling; `abundance_cosine` weighs clones by their cells.
- Coupling and its null treat the cells as exchangeable: the permutations shuffle labels across all cells with a clone, so a z-score says how unusual a pair's sharing is within this sample, not across animals or replicates. Treat coupling as exploratory, and compare it between conditions by computing it per replicate and comparing the values with a method that treats replicates as observations, such as those of [CytePool](https://docs.nygen.io/CytePool/).
- Ask which fates the clones of early cells took with `clones.fate_bias(clones, group_by=cell_type_column, early=early_cells, late=late_cells)`, the early cells at the first time point and the late cells at later ones, sharing no cell. Every early cell of a clone gets its clone's fractions, so count clones rather than early cells when you summarise, and report how many early cells have late clone-mates at all, since the others have no bias.
- Read the tests of `load_fate_bias(bias).clone_fates` by `q_value` within each fate. A clone of a few late cells cannot reach a small p-value, so the significant clones are the large ones. Each clone is compared with all late cells, those without a clone included; when the barcode is detected less in some cell types, pass only late cells with a clone as `late`, so that clones are compared with each other. Like coupling, the tests treat the late cells of one sample as exchangeable, so they rank the clones within it; compare fate bias between conditions per replicate. The dominant fate of early cells is also the outcome against which to score fate predictions from expression alone, such as `trajectory.fate`.
- Compare expansion or diversity between conditions with one value per sample, such as each sample's fraction of cells in clones of more than two cells or its Hill number at a common coverage, since the cells of one sample are not independent observations.

### Immune receptor repertoires

Single-cell immune profiling reads each T or B cell's receptor beside its transcriptome. Import the contigs with `repertoire.import_receptors` and inspect `plots.receptor_qc` before defining clonotypes or reading any repertoire statistic, since every later result counts cells by their chains.

- Import the TCR and the BCR files of one library in one call. `cellranger multi` writes them to `per_sample_outs/<sample>/vdj_t/` and `vdj_b/`; a cell with chains of both kinds is `"ambiguous"` only when both files are imported. Use `filtered_contig_annotations.csv`; `all_contig_annotations.csv` gives the same chains, since contigs outside a called cell or of low confidence do not count, and Cell Ranger's `airr_rearrangement.tsv` gives the same chains with each contig's full sequence and alignment. For other pipelines, import their AIRR Rearrangement files when they name each contig's cell in `cell_id`.
- Import for the cells of the expression analysis, such as the analysis cells of a completed run or `ds.snapshot_cell_selection()`. The V(D)J pipeline calls cells on its own, so some of its barcodes are not cells of the expression library: the warning and the artifact's `n_unmatched_contigs` and `n_unmatched_barcodes` count them. A few percent is usual; most barcodes unmatched points to a wrong `barcode_prefix` in a merged store, or to barcodes written with another suffix than the cell IDs, which `barcode_suffix` or `barcode_pattern` fixes.
- A cell without a receptor chain is `"no IR"`. In PBMCs most `"no IR"` cells are monocytes, NK cells and other cells without a T or B cell receptor; among cells whose expression marks them as T or B cells, many `"no IR"` cells point to a shallow V(D)J library.
- Read the chain pairings by sample with `plots.receptor_qc(receptors, group_by=sample_column)`. `"orphan VJ"` and `"orphan VDJ"` cells lost a chain in capture or sequencing, which is common; they still carry clonal information. `"extra VJ"` cells, with two alpha or two light chains, are often real: the alpha locus is not allelically excluded, so many T cells express two alpha chains, and some B cells express two light chains. `"extra VDJ"`, `"two full chains"`, `"multichain"` and `"ambiguous"` cells are more often doublets, or carry ambient chains of other cells; count them, and compare with the expression-based doublet calls of `qc.doublets` before removing them.
- A T cell with a heavy chain, or a B cell with a beta chain, is `"ambiguous"`. Look at where such cells sit in the expression clusters: scattered among T or B cells they point to ambient contigs, and in a cluster of their own to doublets.
- Count only productive contigs as chains, the default. `productive_only=False` keeps non-productive rearrangements, such as the second allele of an allelically excluded cell, which belong to studies of rearrangement itself, not to clonotypes.
- The receptor type, subtype and chain pairing are per-cell labels: select T cells with paired chains by `select_cells(receptors, value="chain_pairing", include=["single pair", "extra VJ"])` and colour embeddings by them, as with cluster labels.
- Define clonotypes with `repertoire.clonotypes(receptors)` and its defaults, identical nucleotide junctions of both arms with either of two chains per arm, unless the study needs another definition, and report the options. `sequence="aa"` merges T cell clones whose junctions differ only in silent nucleotides, which arise independently, so keep nucleotides to count clones and amino acids to look for shared specificity. `dual_ir="all"` reproduces scRepertoire's clone keys and Cell Ranger's clonotypes before version 5; the default `"any"` lets a cell with a second alpha chain join the cells of either chain.
- Pass the donor as `within` when cells come from several donors: an identical receptor in two donors is a public receptor, recruited twice, not one clone. Do not pass the sample of one donor, such as two tissues or time points, since the clones that the study follows cross those samples.
- Under the default `receptor_arms="all"`, orphan cells with one chain form clonotypes of their own, apart from paired cells with the same chain. Read clone sizes with that in mind, or use `receptor_arms="VDJ"` to group T cells by their beta chain alone when many cells lost their alpha chain.
- B cells of one clone acquire somatic mutations, so identical junctions split a B cell clone into several clonotypes. Treat B cell clonotypes from identical junctions as exact sequence variants, not as clonal families.
- Read the largest clonotypes from `load_clonotypes(clonotypes).clonotypes` and `plots.clone_abundance`, with `group_by` set to the sample: a clonotype that dominates one sample of many is expanded there. Select the cells of expanded clonotypes with `select_cells(clonotypes, value="clone_size", low=2, keep_bounds=True)`.
- Measure clonotypes with the clone summaries above: `clones.expansion`, `clones.diversity` and `clones.rarefaction` read a `clonotypes` artifact as clones and keep each receptor type apart, so a sample's T cells and B cells are two groups. Ambiguous and multichain cells form small groups of their own, which set the default common size and coverage of `clones.diversity` low; measure one receptor type at a time with `cell_selection=select_cells(receptors, include=["TCR"])`.

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
