---
myst:
  html_meta:
    description: Run CyteArc's RNA, ATAC, ADT, multimodal, and CRISPR screen pipeline recipes, follow and cancel runs, and inspect durable PipelineRun records and frozen outputs.
---

# Analysis pipeline API reference

`DataStore.pipeline` runs CyteArc's modality recipes and records each run in the datastore. A call
that fails its checks writes nothing
([Validation before a run starts](#validation-before-a-run-starts)). `rna`, `atac`, and `adt` each
analyze one assay, `multimodal` integrates completed runs of the same cells, and `perturbation`
analyzes a pooled CRISPR screen from its RNA and guide assays. Each recipe
returns a durable, read-only {py:class}`~cytearc.PipelineRun` whose outputs are exact immutable
{py:class}`~cytearc.ArtifactRef` values.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cytearc.PipelineRun
   cytearc.datastore.pipeline_run.PipelineAxisView
   cytearc.PipelineExecutionError
   cytearc.PipelineCancelled
   cytearc.CancellationToken
   cytearc.pipeline_listener
   cytearc.datastore.pipeline_accessor.PipelineAccessor
   cytearc.datastore.pipeline_accessor.PipelineAccessor.rna
   cytearc.datastore.pipeline_accessor.PipelineAccessor.atac
   cytearc.datastore.pipeline_accessor.PipelineAccessor.adt
   cytearc.datastore.pipeline_accessor.PipelineAccessor.multimodal
   cytearc.datastore.pipeline_accessor.PipelineAccessor.perturbation
   cytearc.datastore.pipeline_accessor.PipelineAccessor.open
   cytearc.datastore.pipeline_accessor.PipelineAccessor.list_runs
   cytearc.datastore.pipeline_accessor.PipelineAccessor.abandon_label_claim
   cytearc.datastore.pipeline_accessor.PipelineEvent
   cytearc.datastore.pipeline_accessor.PipelineProgress
```

## Recipes

| Recipe | Analyzes | Graph coordinates | Feature field | Off by default |
| --- | --- | --- | --- | --- |
| `rna` | an RNA assay | PCA of the highly variable genes | `highly_variable_features` | cell calling, hashing, Harmony, Paris, membership strength, t-SNE |
| `atac` | an ATAC assay of peak counts | LSI of the most prevalent peaks | `prevalent_peaks` | Harmony, Paris, membership strength, t-SNE |
| `adt` | an ADT assay of antibody counts | PCA of the antibody panel | `panel` | filtering, Harmony, Paris, membership strength, t-SNE |
| `multimodal` | two or more completed runs of the same cells | none; it integrates the runs' graphs | none | Paris, membership strength, t-SNE |
| `perturbation` | the RNA assay and the guide assay of a pooled CRISPR screen | none; E-distances and signatures read the PCA of the highly variable genes, or its Harmony correction | `highly_variable_features` | Harmony, Mixscape, association tests, clusters |

`rna`, `atac`, `adt`, and `perturbation` take `assay=None` by default: the default assay when it has
the recipe's modality, else the store's only assay of that modality. A store with several such assays needs
`assay=`. An assay of another modality raises `TypeError`, such as
`The rna recipe analyzes RNA assays; assay 'ADT' is of type 'ADT'`.

Recipes write, so they need a store opened with `zarr_mode="r+"`; a read-only store raises
`PermissionError` before the recipe checks its settings. Opening, listing, and reporting runs work
on a read-only store.

### RNA

```python
rna = ds.pipeline.rna(label="rna")

rna["clusters"]
rna.cells.to_pandas_dataframe(["umap_1", "umap_2", "clusters"])
ds.plots.embedding(run=rna, layout="umap", color_by="clusters")
ds.markers.load(marker=rna["markers"], group_id=1)
```

The `rna` recipe runs these stages in order:

`input_snapshot`, `cell_calling`, `hashing`, `filtering`, `cell_cycle`, `highly_variable_features`,
`normalization`, `pca`, `harmony`, `ann_index`, `neighbors`, `connectivity`,
`embedding_initialization`, `umap`, one `leiden_<resolution>` stage per candidate, `paris`,
`cluster_selection`, `membership_strength`, `tsne`, `doublets`, `markers`.

By default it filters cells on their QC metrics with MAD bounds, scores the cell cycle, selects
1,000 highly variable genes, normalizes them, keeps 21 PCA dimensions, builds a graph of 11
neighbours, embeds it with UMAP, clusters it with Leiden at resolutions 0.5, 0.75, 1.0, and 1.25,
saves the candidate with the best silhouette as `run["clusters"]`, scores doublets, and ranks
markers. These numbers are the defaults of the stages' methods: `features.hvgs` `top_n`,
`reduction.pca` `dims`, and `graph.neighbors` `k`.

### ATAC

```python
atac = ds.pipeline.atac(label="atac")

atac["lsi"]
atac.features.fetch("prevalent_peaks")
```

The `atac` recipe runs `input_snapshot`, `filtering`, `prevalent_peaks`, `normalization`, `lsi`,
`harmony`, `ann_index`, `neighbors`, `connectivity`, `embedding_initialization`, `umap`, one
`leiden_<resolution>` stage per candidate, `paris`, `cluster_selection`, `membership_strength`,
`tsne`, and `markers`.

By default it filters cells on `<assay>_nCounts` and `<assay>_nFeatures` with MAD bounds, keeps the
10,000 most prevalent peaks (`features.prevalent_peaks` `top_n`), normalizes them with the assay's
normalizer, TF-IDF for an ATAC assay, and reduces them to 11 LSI dimensions without the first
component (`reduction.lsi` `dims` and `skip_first`). It then builds, embeds, and clusters a graph
and ranks markers as the RNA recipe does. With Harmony, the run corrects its LSI coordinates.

`prevalent_peaks` `top_n` defaults to 10,000 or one fewer than the assay's number of peaks,
whichever is smaller, and a given `top_n` must be below the assay's number of peaks; the run records
the resolved `top_n`. The LSI needs `dims` prevalent peaks, one more when `skip_first` leaves out
the first component, so `lsi` `dims` with that component must not exceed `top_n`, and `skip_first`
must be a bool.

### ADT

```python
# Analyze the antibodies of the RNA run's cells.
adt = ds.pipeline.adt(cells=rna, label="adt")

adt.features.fetch("panel")
```

The `adt` recipe runs `input_snapshot`, `filtering`, `panel`, `normalization`, `pca`, `harmony`,
`ann_index`, `neighbors`, `connectivity`, `embedding_initialization`, `umap`, one
`leiden_<resolution>` stage per candidate, `paris`, `cluster_selection`, `membership_strength`,
`tsne`, and `markers`.

The assay must be of type `ADT`; an `HTO` assay is read by the RNA recipe's `hashing` stage
instead. The panel leaves out the antibodies whose names match
`cytearc.datastore.pipeline_accessor.DEFAULT_PANEL_EXCLUDE`, the regular expression
`(?i)isotype|control|ctrl`, so isotype controls stay out of the reduction. `panel={"exclude": ...}`
sets another pattern, and `panel={"exclude": None}` keeps every antibody. The panel must keep at
least two antibodies, and the run records the excluded names. The antibodies are normalized with
the assay's normalizer, CLR for an ADT assay, and PCA keeps one dimension fewer than the panel
holds, at most 21, unless `pca` sets `dims`, which must be at least 1 and below the panel's number
of antibodies. The recipe filters cells only when `filtering` asks for it, because an ADT run
usually takes the QC-passing cells of an RNA run.

### Multimodal

```python
wnn = ds.pipeline.multimodal([rna, adt], label="wnn")

wnn["integrated_graph"]
wnn.cells.fetch("weight_ADT")
ds.markers.load(marker=wnn["markers_ADT"], group_id=1)
```

The `multimodal` recipe runs `input_snapshot`, `integration`, `embedding_initialization`, `umap`,
one `leiden_<resolution>` stage per candidate, `paris`, `cluster_selection`, which is always
skipped, `membership_strength`, `tsne`, and `markers`.

It integrates the runs' graphs with weighted nearest neighbours (`method="wnn"`, the default), which
weighs each run's neighbours, or with shared nearest neighbours (`method="snn"`), which shares each
run's connectivity-map edges ({doc}`integration`). It then embeds and clusters the integrated graph
and ranks each run's assay's features for the joint clusters as one `markers_<assay>` output per
run. `integration` takes the keywords of `integration.modalities`: `chunk_size` and, for WNN only,
`l2_normalize`.

`runs` is a sequence of two or more completed runs of different assays that analyze the same
cells: their `analysis_cell_selection` outputs must be the same ref, which a run gets by taking an
earlier run as its `cells`. WNN weighs each modality by the coordinates that built its neighbours,
so each run's graph must be built on PCA, LSI, or Harmony coordinates; an RNA run with
`pca={"dims": 0}` can be integrated with SNN only. SNN shares edges between graphs of one
neighbour count, so with `method="snn"` the runs must have been built with the same `neighbors`
`k`. The joint embedding starts from the first run's coordinates, so the first run needs a
`neighbors` output when UMAP or t-SNE runs. The joint run's `assay` and `feature_universe` are the
first run's, and its feature view covers that assay's features with `I`, `ids`, and `names` only,
without the first run's feature field.

The integrated graph has no shared coordinates in which to compare Leiden candidates by
silhouette. `leiden` partitions therefore default to `[1.0]`, the only candidate is the saved
clustering, and several partitions need a `selected` resolution.

### Compose runs of the same cells

A CITE-seq analysis runs the RNA recipe, analyzes the antibodies of the same cells, and integrates
the two runs:

```python
rna = ds.pipeline.rna(label="rna")
adt = ds.pipeline.adt(cells=rna, label="adt")
wnn = ds.pipeline.multimodal([rna, adt], label="wnn")
```

`adt` takes the RNA run's analysis cells as they are, so both runs share one cell selection. The
ADT assay must have measured each of those cells. Each run is a separate durable record with its
own label, and the multimodal run records its sources as `config["sources"]`. A TEA-seq analysis
adds `atac = ds.pipeline.atac(cells=rna, label="atac")` and integrates `[rna, atac, adt]`.

### Perturbation

```python
screen = ds.pipeline.perturbation(labels={"controls": ["Non-Targeting"]}, label="screen")

ds.perturbation.load_efficiency(screen["efficiency"]).table
ds.perturbation.load_edistance(screen["edistance"]).table
print(screen.report(format="markdown"))
```

The `perturbation` recipe analyzes a pooled CRISPR screen ({doc}`perturbation`): an RNA assay and a guide assay, declared `CRISPR`, that measured the same cells. It runs these stages in order:

`input_snapshot`, `filtering`, `guide_calling`, `perturbation_labels`, `efficiency`, `highly_variable_features`, `normalization`, `pca`, `harmony`, `edistance`, `signature`, `mixscape`, `associations`, `profiles`, `perturbation_clusters`.

By default it filters cells on their RNA QC metrics as the `rna` recipe does, calls the guides of the analysis cells with `perturbation.call_guides` and its default `method="poisson_gaussian"`, labels each cell by the genes that its guides target, and measures each target's efficiency on its own gene. It then selects 1,000 highly variable genes, normalizes them, keeps 21 PCA dimensions, and measures each target's E-distance from the control cells in the PCA, with 1,000 permutations. Harmony, Mixscape, the association tests, and the clusters are off. The guides are called and the cells labelled before the PCA, so a labelling that fails stops the run early.

`labels` is required and must name `controls`, the targets of the control guides, which `perturbation.labels` needs: CyteArc never guesses which guides are controls. `guide_assay` defaults to the store's only assay declared `CRISPR`. `guide_calling` takes the keywords of `perturbation.call_guides`, or `path` and `barcode_prefix`, with which the stage imports Cell Ranger's `protospacer_calls_per_cell.csv` through `perturbation.import_guide_calls` instead. Fitted calls output their fits as `guide_thresholds` beside the `guide_calls`. Efficiency matches each target to a gene by the guide assay's `target_gene_id` column, which 10x imports fill, so a guide assay without it needs `efficiency={"id_column": None}`, which matches targets by name; the run refuses the default otherwise.

Cell Ranger writes a call file for each sample, and the import matches each of its barcodes to the cell whose ID is `barcode_prefix` followed by the barcode. A cell that no barcode can reach would be labelled `"Negative"` and left out of every later stage, so a run that imports a call file refuses such cells before it writes anything: with `barcode_prefix`, cells of `cells` whose IDs do not start with it, and without one, cells of `cells` from several samples of a merged store, which names each cell `<sample>__<barcode>`. In a merged store, restrict `cells` to the cells of one sample and pass its prefix, such as `barcode_prefix="lane1__"`, or call the guides from their counts. The stage summary's `unmatchedBarcodes` counts the file's barcodes that match no analysis cell, which with the default filtering include those of the cells that QC filtering removed; the import's warning counts them too, so compare the count with the `filtering` stage's `removedCells` before suspecting `barcode_prefix`.

`signature` configures the signature that `mixscape=True` computes first, and `profiles` the profiles that `clusters=True` clusters; their settings without that stage raise `ValueError`. The signature holds the highly variable genes, and its `dims` must not exceed the PCA's, so its default of 15 needs `pca` `dims` of at least 15. E-distances and the signature read the Harmony coordinates when the run has Harmony, and the PCA otherwise. The association tests test every target against every highly variable gene (`pairs="trans"`), so `associations` takes neither `pairs` nor `features`, and its `positive_controls` is `"targets"` or None; call `ds.perturbation.associations` for a table of pairs. In a run with Mixscape, the profiles average each target's knocked-out cells by default, from the `mixscape` output, so that escapers do not dilute them. A target with fewer than `min_cells` knocked-out cells gets no profile, so a screen whose targets Mixscape leaves with few knocked-out cells can keep them with `profiles={"cells": "labelled"}`, which averages every cell of each label, as a run without Mixscape does. `"knocked_out"` without Mixscape raises `ValueError` before the run writes anything. The run records which output the profiles read as `config["profileLabels"]`.

Each perturbation stage calls a `ds.perturbation` method with the run's refs and settings, so the method called with the same refs and settings returns the run's artifact, and a run reuses the artifacts of such calls. A reused artifact repeats its method's warnings, so the stage records them as the stage that computed it did. The stages from `filtering` to `harmony` are those of the `rna` recipe. A run with the cells and the `filtering`, `hvg`, `normalization`, `pca`, and `harmony` settings of an `rna` run reuses that run's analysis cells, highly variable genes, normalized values, PCA, and Harmony coordinates, and so does a run given the RNA run as `cells`, which keeps its analysis cells, with the same `hvg`, `normalization`, `pca`, and `harmony` settings:

```python
rna = ds.pipeline.rna(label="rna", hvg={"top_n": 2000}, pca={"dims": 30})

# Reuse the RNA run's cells, genes, and PCA, and run every stage.
screen = ds.pipeline.perturbation(
    cells=rna,
    labels={"controls": ["Non-Targeting"]},
    hvg={"top_n": 2000},
    pca={"dims": 30},
    signature={"dims": 30, "sample_column": "lane"},
    mixscape=True,
    associations={"sample_column": "lane"},
    clusters=True,
    label="screen",
)
```

The run's `assay` is the RNA assay. Its cell view holds each cell's `guide_call`, the call of the run's `guide_calls`: the ID of its one called guide, `"Negative"` without a guide, or `"Multiple"` with several. It holds `perturbation_label`, the label of the run's `perturbation_labels`, and `is_control`, which is True for a cell with control guides only, labelled `"control"`. A cell with several guides of one gene, or with a gene's guides beside control guides, is called `"Multiple"` but labelled by the gene; [Label cells by perturbation](perturbation.md#label-cells-by-perturbation) lists every case. With Mixscape, the view also holds `mixscape_class` and `mixscape_posterior`. The guide calls and labels belong to the guide assay, and the view reads them by the run's cells ([Frozen fields](#frozen-fields)). `ds.perturbation.load_guide_calls(screen["guide_calls"])` and `ds.perturbation.load_labels(screen["perturbation_labels"])` read their other columns, such as each cell's guides and genes:

```python
screen.cells.to_pandas_dataframe(["ids", "guide_call", "perturbation_label"])
adata = ds.to_anndata(run=screen)  # obs holds the three fields as categories
```

## Stage keywords

Each stage keyword configures one stage:

- An optional stage, `cell_calling`, `hashing`, `filtering`, `cell_cycle`, `umap`, `tsne`, `leiden`,
  `paris`, `doublets`, `efficiency`, `edistance`, `mixscape`, `associations`, or `clusters`, takes
  `False` to skip it, `True` to run it with its defaults, or a mapping of settings, which also runs
  it. `None` is not a switch: `umap=None` raises `TypeError`.
  `filtering=None` is the default of the `filtering` keyword, described in
  [Filtering](#filtering).
- A stage that always runs, `hvg`, `normalization`, `pca`, `prevalent_peaks`, `lsi`, `panel`,
  `ann_index`, `neighbors`, `connectivity`, `embedding_initialization`, `integration`, or
  `guide_calling`, takes None for its defaults or a mapping of settings. A bool raises `TypeError`, such as
  `pca must be a mapping because the stage always runs`.
- `harmony` is None, the default, or `False`, which leave Harmony off, or a mapping that names
  `batch_columns`. `harmony=True` raises `ValueError`
  (`harmony needs a mapping with batch_columns`), and any other value, such as a list of columns,
  raises `TypeError`.
- `membership_strength` and `markers` are plain bools.
- `labels`, the settings of a `perturbation` run's labels, is a required mapping that names `controls`.
- `signature` and `profiles` configure the stages that a `perturbation` run's `mixscape` and `clusters` run: None, the default, or a mapping of settings, which raises `ValueError` while its stage is off. A bool raises `TypeError`.

A setting means exactly what the same keyword means on the stage's method, and an omitted setting takes that method's default, with three exceptions that fit the recipe to its data: an `adt` run's `pca` `dims` defaults to one fewer than the panel's antibodies, at most 21 ([ADT](#adt)); an `atac` run's `prevalent_peaks` `top_n` to 10,000 or one fewer than the assay's peaks ([ATAC](#atac)); and with Harmony, an `rna` run's `doublets` `sample_column` to the Harmony batch columns ([Doublets and Harmony](#doublets-and-harmony)). The pipeline supplies the artifacts that each stage consumes, so a mapping holds settings only. A few keys configure the recipe instead: Leiden `partitions` and `selected`, Harmony `batch_columns`, cell-calling `sample_column`, hashing `assay`, panel `exclude`, guide-calling `path` and `barcode_prefix`, which import Cell Ranger's calls in place of calling guides, and profiles `cells`, which chooses the cells that the profiles average ([Perturbation](#perturbation)). An unknown key raises `ValueError` that lists the allowed keys, such as `Unknown pca settings ['dim']; allowed: ['batch_size', 'dims', 'feat_scaling']`.

```python
run = ds.pipeline.rna(
    label="tuned",
    filtering={"method": "manual", "attrs": ["RNA_nCounts"], "lows": [500], "highs": [20000]},
    hvg={"flavor": "seurat_v3", "top_n": 2000},
    pca={"dims": 30},
    neighbors={"k": 15},
    umap={"n_epochs": 400, "min_dist": 1.0, "spread": 2.0, "parallel": True},
    leiden={"partitions": [0.6, 0.8, 1.0, 1.2, 1.4], "selected": 1.0},
    membership_strength=True,
    tsne={"max_iter": 800, "early_iter": 200, "alpha": 10, "box_h": 0.7},
    species="homo_sapiens",
)
```

| Keyword | Recipes | Method | Settings |
| --- | --- | --- | --- |
| `cell_calling` | `rna` | {py:meth}`~cytearc.datastore.namespaces.QcAccessor.call_cells` | `lower`, `retain`, `n_iter`, `alpha`, `fdr`, `random_seed`, and `sample_column`, a cell column of captures |
| `hashing` | `rna` | {py:meth}`~cytearc.datastore.namespaces.QcAccessor.hto_demultiplexing` | `assay`, the HTO assay; `random_seed`, `positive_quantile` |
| `filtering` | `rna`, `atac`, `adt`, `perturbation` | the rules of {py:meth}`~cytearc.datastore.namespaces.QcAccessor.auto_filter` | `method`, `attrs`, `n_mads`, `sample_column`, `min_cells_per_sample`, `min_p`, `max_p`; with `method="manual"`, `attrs`, `lows`, `highs`, `keep_bounds` |
| `cell_cycle` | `rna` | {py:meth}`~cytearc.datastore.namespaces.ScoresAccessor.cell_cycle` | `s_genes`, `g2m_genes`, `ctrl_size`, `log_transform`, `n_bins`, `rand_seed` |
| `hvg` | `rna`, `perturbation` | {py:meth}`~cytearc.datastore.namespaces.FeaturesAccessor.hvgs` | `top_n`, `flavor`, `batch` (a cell metadata column), `min_cells`, `max_cells`, `min_mean`, `max_mean`, `min_var`, `max_var` (the last four are log2 values), `n_bins`, `lowess_frac`, `blacklist`, `keep_bounds`, `bin_strategy`, `span`, `theta`, `clip` |
| `prevalent_peaks` | `atac` | {py:meth}`~cytearc.datastore.namespaces.FeaturesAccessor.prevalent_peaks` | `top_n` |
| `panel` | `adt` | the antibody panel | `exclude`, a regular expression or None |
| `normalization` | `rna`, `atac`, `adt`, `perturbation` | {py:meth}`~cytearc.datastore.namespaces.FeaturesAccessor.normalize` | `log_transform`, `renormalize_subset` |
| `pca` | `rna`, `adt`, `perturbation` | {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.pca` | `dims`, `feat_scaling`, `batch_size` |
| `lsi` | `atac` | {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.lsi` | `dims`, `skip_first`, `rand_state`, `solver`, `n_iter`, `n_oversamples`, `batch_size` |
| `harmony` | `rna`, `atac`, `adt`, `perturbation` | {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.harmony` | `batch_columns` (required), `harmony_params` (the keyword options of {py:func}`~cytearc.embeddings.fit_harmony`), `batch_size` |
| `ann_index` | `rna`, `atac`, `adt` | {py:meth}`~cytearc.datastore.namespaces.GraphAccessor.ann_index` | `ann_metric`, `ann_efc`, `ann_ef`, `ann_m`, `ann_parallel`, `rand_state`, `batch_size` |
| `neighbors` | `rna`, `atac`, `adt` | {py:meth}`~cytearc.datastore.namespaces.GraphAccessor.neighbors` | `k`, `ann_ef`, `batch_size` |
| `connectivity` | `rna`, `atac`, `adt` | {py:meth}`~cytearc.datastore.namespaces.GraphAccessor.connectivity` | `local_connectivity`, `bandwidth` |
| `integration` | `multimodal` | {py:meth}`~cytearc.datastore.namespaces.IntegrationAccessor.modalities` | `chunk_size`, and for WNN `l2_normalize` |
| `embedding_initialization` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.EmbeddingsAccessor.initialization` | `n_centroids`, `rand_state`, `batch_size`, `kmeans_sampling`, `kmeans_batch_size` |
| `umap` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.EmbeddingsAccessor.umap` | `umap_dims`, `spread`, `min_dist`, `n_epochs`, `repulsion_strength`, `initial_alpha`, `negative_sample_rate`, `use_density_map`, `dens_lambda`, `dens_frac`, `dens_var_shift`, `random_seed`, `parallel`, `symmetric_graph`, `graph_upper_only` |
| `tsne` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.EmbeddingsAccessor.tsne` | `tsne_dims`, `lambda_scale`, `max_iter`, `early_iter`, `alpha`, `box_h`, `symmetric_graph`, `graph_upper_only` |
| `leiden` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.ClustersAccessor.leiden` | `partitions`, `selected`, `backend`, `random_seed`, `symmetric_graph`, `graph_upper_only` |
| `paris` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.ClustersAccessor.paris` | `n_clusters`, `min_cluster_size` |
| `membership_strength` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.ClustersAccessor.membership_strength` | none; a bool |
| `doublets` | `rna` | {py:meth}`~cytearc.datastore.namespaces.QcAccessor.doublets` | `cluster_sample_fraction`, `max_cells_per_cluster`, `sample_column`, `simulation_ratio`, `heterotypic_fraction`, `save_k`, `expected_doublet_rate`, `threshold`, `smoothing_t`, `normalize_scores`, `random_seed` |
| `markers` | all but `perturbation` | {py:meth}`~cytearc.datastore.namespaces.MarkersAccessor.search` | none; a bool |
| `guide_calling` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.call_guides`, or {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls` with `path` | `method`, `min_umi`, `fraction_threshold`, `sample_column`; or `path` and `barcode_prefix` |
| `labels` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.labels` | `controls` (required), `target_column`, `targets`, `policy` |
| `efficiency` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.efficiency` | `by`, `id_column` |
| `edistance` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.edistance` | `by`, `n_permutations`, `random_seed` |
| `signature` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.signature` | `dims`, `n_neighbors`, `sample_column`, `scale_factor` |
| `mixscape` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.mixscape` | `split_column`, `min_de_genes`, `min_cells`, `logfc_threshold`, `min_fraction`, `p_value_cutoff`, `scale`, `max_iterations`, `perturbation_type` |
| `associations` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.associations` | `positive_controls` (`"targets"` or None), `id_column`, `integration`, `sample_column`, `covariates`, `side`, `n_resamples`, `stage_threshold`, `approximation`, `min_nonzero`, `n_calibration_pairs`, `alpha`, `seed` |
| `profiles` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.profiles` | `by`, `sample_column`, `min_cells`, `mask_targets`, `id_column`; and `cells`, `"knocked_out"` or `"labelled"` |
| `clusters` | `perturbation` | {py:meth}`~cytearc.datastore.namespaces.PerturbationAccessor.cluster` | `perturbations`, `metric`, `min_cluster_size`, `min_samples`, `selection`, `seed` |

The other keywords configure the run itself:

- `label`: an immutable name for the completed run, unique in the store
  ([Open, list, and report runs](#open-list-and-report-runs)).
- `cells`: the cells the run analyzes ([Cells](#cells)); `multimodal` takes the cells of its runs.
- `snapshot_columns`: cell columns frozen into the run's cell view. They must exist and must not
  use a reserved field name ([Frozen fields](#frozen-fields)).
- `species`: a species key recorded with the run, one of `homo_sapiens`, `mus_musculus`,
  `rattus_norvegicus`, `danio_rerio`, `drosophila_melanogaster`, or `caenorhabditis_elegans`.
  `multimodal` and `perturbation` take none.
- `guide_assay`: the guide assay of a `perturbation` run ([Perturbation](#perturbation)).
- `callback`: a callable that receives each
  {py:class}`~cytearc.datastore.pipeline_accessor.PipelineEvent` of the run
  ([Events and listeners](#events-and-listeners)).
- `cancel`: a {py:class}`~cytearc.CancellationToken` that stops the run
  ([Cancellation](#cancellation)).

Settings are JSON values: numbers, strings, bools, None, lists, and mappings with string keys.
Infinity and NaN are never settings. Some settings spell a derived default as `None`: `hvg`
`min_cells`, 1% of the selected cells rounded up, from 3 to 20, for the default flavor; `hvg`
`max_cells`, which for the default flavor excludes genes detected in more than 99.5% of the selected
cells; `batch_size` of every stage that takes it; `cell_cycle` `ctrl_size` and `log_transform`;
`paris` `min_cluster_size`; and `normalization` `log_transform` and `renormalize_subset`, whose
defaults follow the assay's normalizer ({doc}`assays`). The run records the settings it was given
under `config["stages"]`, and each artifact records the values its method resolved.

### Validation before a run starts

A recipe checks its call against the store before it creates a run record, and a call that fails a
check raises an error such as `TypeError`, `ValueError`, or `KeyError` without writing anything.
Unknown keywords or settings, settings without a JSON value, a bool for a stage that always runs,
an assay of another modality, missing columns, cells that an assay did not measure, reserved
snapshot fields, invalid stage combinations, invalid values of the settings listed below, and an
already completed label fail this way. Among the errors:

- `doublets and markers require at least one Leiden candidate` (`rna` with `leiden=False`;
  `atac` and `adt` raise `markers require at least one Leiden candidate`), and
  `membership_strength requires at least one Leiden candidate`;
- `pca={'dims': 0} builds the graph on normalized values, which Harmony cannot correct; ...` and
  `pca={'dims': 0} builds the graph on normalized values, but doublet scoring needs a PCA graph; ...`;
- `PCA settings need dims above 0; dims=0 skips PCA`, for other `pca` settings beside `dims=0`;
- `leiden selected must be one of the partitions`;
- `snapshot_columns collide with reserved run fields: [...]`;
- `TypeError("callback must be callable")` and a `cancel` that is not a `CancellationToken`.

Before the run starts, the recipe also checks these settings, with the checks of the stage's
method where it has them:

- `hvg`: the checks of `features.hvgs`. `min_cells` must be None or an integer of at least 0,
  `top_n` and `n_bins` integers of at least 1, `lowess_frac` a finite number from 0 to 1 for either
  `bin_strategy`, `keep_bounds` a bool, `bin_strategy` `"adaptive"` or `"fixed"`, `flavor`
  `"cytearc"`, `"seurat_v3"`, or `"pearson_residuals"`, `span` greater than 0 and at most 1,
  `theta` positive, and `clip` None or at least 0; the options of one flavor keep their defaults
  under another, and `batch` must name a cell metadata column. An omitted setting is checked as its
  `features.hvgs` default.
- `pca` `dims`: an integer of at least 0 for `rna` and at least 1 for `adt`, below the number of
  features that PCA reads, because PCA needs `dims + 1` of them: the `hvg` `top_n` of an `rna` run,
  or the panel's antibodies of an `adt` run.
- `prevalent_peaks` `top_n` and the `lsi` `dims` and `skip_first`, as [ATAC](#atac) describes.
- `neighbors` `k`, `umap` `umap_dims`, and `tsne` `tsne_dims`: integers of at least 1.
- `tsne`: the checks of `embeddings.tsne`, because the stage runs after most others. `tsne_dims`,
  `max_iter`, and `alpha` must be integers of at least 1, `early_iter` an integer of at least 0,
  `lambda_scale` and `box_h` finite positive numbers, and `symmetric_graph` and
  `graph_upper_only` bools.
- The graph flags: the `umap` `symmetric_graph` and `graph_upper_only` settings must be `False`,
  and a `leiden` or `tsne` `graph_upper_only` set to `True` needs `symmetric_graph=True`.
- `harmony` `harmony_params`: the checks of `reduction.harmony`. Its keys are the keyword options
  of {py:func}`~cytearc.embeddings.fit_harmony` other than `nthreads`, and an unknown key, a key of
  an earlier release such as `nclust` (now `n_clusters`), or an invalid value raises `ValueError`
  or `TypeError`; {doc}`integration` lists the keys.
- `cell_calling`: the checks of `qc.call_cells` for `lower`, `retain`, `n_iter`, `alpha`, `fdr`,
  and `random_seed`. With `sample_column`, every cell of `cells` needs a capture label: a missing
  value, or NaN in a float column, raises `ValueError`. The stage checks its frozen labels again.
- `hashing`: `positive_quantile`, greater than 0 and less than 1, and the HTO assay.
- `filtering`: its method, columns, and options ([Filtering](#filtering)).
- `panel` `exclude`: a valid regular expression that keeps at least two antibodies.
- The runs of `multimodal` ([Multimodal](#multimodal)). With `method="snn"`, runs built with
  different `neighbors` `k` raise `ValueError` that names `neighbors={'k': ...}` and
  `method='wnn'`.
- The guide assay of `perturbation`: the store's only assay declared `CRISPR`, or the one that `guide_assay` names, which must be declared `CRISPR` and store unsigned counts. A store with no CRISPR assay, or several, raises `ValueError` that lists them.
- `labels`: `controls`, which a mapping without them, or `labels=None`, lacks (`ValueError`, `labels needs controls, the targets of the control guides, ...`), and the checks of `perturbation.labels` against the guide assay's targets: a control that no guide has, a target that is a reserved label or, with `policy="combinations"`, holds `"+"`, `targets` that name a guide that the assay lacks, and a `target_column` that the guide assay lacks. A called guide without a target shows only in the calls, so the `perturbation_labels` stage refuses it, and the run fails.
- `guide_calling`: the checks of `perturbation.call_guides` for `method`, `min_umi`, `fraction_threshold`, and `sample_column`, or a `path` that is a file (`FileNotFoundError` otherwise) with a `barcode_prefix` of text or None. A `path` beside a calling setting, or `barcode_prefix` without `path`, raises `ValueError`, and so do cells of `cells` that the call file cannot reach: with `barcode_prefix`, cells whose IDs do not start with it, and without one, cells from several samples of a merged store.
- The settings of `efficiency`, `edistance`, `signature`, `mixscape`, `associations`, `profiles`, and `clusters`: the checks of their methods, an `id_column` that the guide assay lacks where the stage reads it (`efficiency`, `associations` with `positive_controls="targets"`, and `profiles` with `mask_targets`), sample, split, and covariate columns that the store lacks (`KeyError`), and covariates that do not hold numbers.
- The `sample_column` of `guide_calling`, `signature`, `associations`, or `profiles`, and a Mixscape `split_column`, must label every cell of `cells`: a missing value, NaN in a float column, or empty or whitespace text raises `ValueError`. Each covariate of `associations` must hold a finite value for every cell of `cells`: a missing, NaN, or infinite value raises `ValueError`. The methods need labels and covariate values only on the cells that they read, such as the control cells and the cells of the tested targets, but which cells those are depends on the data, so the run asks for them on every cell of `cells`.
- `pca` `dims` of a `perturbation` run: at least 1 and below `hvg` `top_n`, and at least `signature` `dims` when the run calls Mixscape.
- `signature` settings without `mixscape`, `profiles` settings without `clusters`, a `profiles` `cells` other than `"knocked_out"` or `"labelled"`, `"knocked_out"` without `mixscape`, and `pairs`, `features`, or a table of `positive_controls` in `associations` raise `ValueError`.

Every other value is checked by the stage's method when the stage runs, such as `ann_index`
`ann_metric`, a Leiden `backend`, the numeric `doublets` settings, and the `umap` settings other
than `umap_dims` and the graph flags, and so are limits that depend on the data, such as the memory
that the ANN index or cluster selection needs, a sample with fewer control cells than the signature's
`n_neighbors`, or clusters of fewer than 3 profiles. Such a value or limit fails its stage after the
earlier stages have run: the run is recorded as failed and raises `PipelineExecutionError`.

A handled stage failure raises {py:class}`~cytearc.PipelineExecutionError` after the stage and
the run are recorded as failed. Use its `run_id` to inspect the persisted failure, and its `stage`
to identify the failing stage, or `"finalize"` when completing the run record failed. The original
exception is retained as `__cause__`.

```python
from cytearc import PipelineExecutionError

try:
    run = ds.pipeline.rna(label="baseline")
except PipelineExecutionError as error:
    failed_run = ds.pipeline.open(run_id=error.run_id)
    report = failed_run.report(format="dict")
```

## Migrate from `pipeline.run`

`DataStore.pipeline.run()` and its `params` mapping are removed. This is a breaking change within
the 1.x releases, and no alias remains: `ds.pipeline.run` raises `AttributeError`. The `rna` recipe
replaces it, with one keyword per stage:

| Removed | Replacement |
| --- | --- |
| `ds.pipeline.run(...)` | `ds.pipeline.rna(...)` |
| `cell_key="x"` | `cells="x"` |
| `hvg_count=N` | `hvg={"top_n": N}` |
| `pca_dims=N` | `pca={"dims": N}` |
| `neighbors_k=N` | `neighbors={"k": N}` |
| `harmony_batch_columns=[c]` | `harmony={"batch_columns": [c]}` |
| `params={"<stage>": {...}}` | `<stage>={...}`, such as `umap={"n_epochs": 400}` |
| `params={"membership_strength": True}` | `membership_strength=True` |
| `params={"species": s}` | `species=s` |
| `leiden={"partitions": [...]}` | unchanged |
| Paris on by default | `paris=True`; Paris is off by default |
| `filtering=True` by default | `filtering=None` by default, which filters a `cells` column as before |
| the `pipeline_interrupted` event | `run_interrupted` |

The stages of the `rna` recipe record the same provenance as those of `pipeline.run()`, so a call
with the same settings reuses the artifacts of an earlier `pipeline.run()` call. Pass `paris=True`
to reuse its Paris clustering too. A run with Harmony or `snapshot_columns` is the exception: the
recipe freezes those columns apart from the QC columns that filtering reads
([Run snapshots](#run-snapshots)), so its filtered cells and its Harmony fit, and the artifacts
computed from them, get new identities. The first such call computes them again and later calls
reuse them; nothing is migrated, and the earlier artifacts stay in the store. Callbacks receive
more events than before: run events, `stage_progress`, and `stage_skipped`, and `PipelineEvent` has
more fields ([Events and listeners](#events-and-listeners)). Runs that `pipeline.run()` saved still
open and report ([Runs saved by other releases](#runs-saved-by-other-releases)).

## Cells

`cells` names the cells that a single-assay run analyzes:

- a boolean cell column, `"I"` by default, which the run snapshots into an immutable selection;
- a datastore cell-selection artifact, such as the result of `ds.select_measured_cells("ADT")` or
  `ds.select_cells(...)`; or
- a completed `PipelineRun`, whose `analysis_cell_selection` the run reuses as the same ref.

The run's `input_cell_selection` output is the snapshot of the column, or the given selection
itself. A run that is not completed, or a run of another datastore, raises `ValueError`.
The recipe checks every assay that it reads over these cells, the analyzed assay and, with
hashing, the HTO assay, and the assay of each QC metric that it filters on. A cell that one of them
did not measure raises {py:exc}`~cytearc.metadata.membership.UnmeasuredCellsError` with the
operation `pipeline.rna`, `pipeline.atac`, `pipeline.adt`, or `pipeline.perturbation`, whose guide
assay is checked too, and a message that says how to pass
measured cells as `cells`; when the cells are an earlier run's, it says to run that earlier run on
cells that the assay measured too, so that the runs analyze the same cells. The run never narrows
`cells` on its own.

## Quality control stages

The `rna` recipe calls cells, demultiplexes hashtags, and filters, in that order. Each stage reads
the cells that the one before it kept, and the cells that remain are the run's
`analysis_cell_selection`.

### Cell calling

`cell_calling=True` calls cells in a raw droplet matrix with EmptyDrops
({py:meth}`~cytearc.datastore.namespaces.QcAccessor.call_cells`). It tests the barcodes of `cells`,
so open a raw droplet store with `min_features_per_cell=-1`, which keeps every barcode, the ambient
pool included, in `I`. The run gains the `cell_calling` statistics and the `called_cell_selection`.
`cell_calling={"sample_column": "capture"}` calls each capture apart, with its own ambient profile
and knee, and analyzes the union of the called cells; that run's `called_cell_selection` records
each capture's calls as its inputs, and the run has no `cell_calling` output. Every cell of `cells`
needs a capture label. The input snapshot freezes the capture column apart, as its
`capture_snapshot`, so each capture's calls stay the same when the filtering or snapshot settings
change. When the calling of one capture fails, the stage's error names that capture, such as
`Cell calling failed for capture 'a' of 'capture': ...`.

### Hashing

`hashing=True` demultiplexes hashtags with
{py:meth}`~cytearc.datastore.namespaces.QcAccessor.hto_demultiplexing`. It reads the store's only
assay of type `HTO`, or the one that `hashing={"assay": ...}` names. The run keeps the cells of
exactly one hashtag, those not called `Negative` or `Doublet`, as its `hashed_cell_selection`, and
gains the `hashing` labels. With MAD filtering and no `sample_column`, the filter computes its bounds
within each hashtag, and the run records `"sampleSource": "hashing"` in its filtering settings. A
hashing stage that calls no cell a singlet fails.

### Filtering

`filtering=None`, the default, filters the cells of an `rna` or `atac` run whose `cells` names a
column, and keeps a given selection's or run's cells as they are, which are taken as already
filtered. An `rna` run that calls cells or demultiplexes hashtags filters the cells that those
stages keep whatever its `cells`, because called cells and hashtag singlets are not filtered yet.
The `adt` recipe filters only when `filtering` is `True` or a mapping. `True` filters with the
defaults, `False` keeps the input cells, and a mapping configures the filter.

Filtering reads the QC values frozen in the run's cell snapshot. Without `attrs`, it reads the QC
columns that the assay's preparation wrote: `<assay>_nCounts`, `<assay>_nFeatures`,
`<assay>_percentMito`, and `<assay>_percentRibo` for RNA, `<assay>_nCounts` and
`<assay>_nFeatures` for ATAC, and `<assay>_nCounts` alone for ADT, because nearly every cell detects
nearly every antibody of a small panel. If filtering is requested and none of those columns exists,
validation raises instead of silently analyzing the unfiltered cells. Pass `filtering=False` only
when that choice is deliberate.

The filter applies the rules of `qc.auto_filter`. The default and `method="auto"` both resolve to
`method="mad"`. MAD filtering accepts `attrs`, `n_mads`, `min_cells_per_sample`, and optionally
`sample_column`; without a sample column it uses pooled bounds. Groups with fewer than 20 active
cells are retained with a warning by default, including small pooled selections. Pass
`method="gaussian"` for the Gaussian policy, with `attrs`, `min_p`, and `max_p`; Gaussian filtering
does not accept a sample column. `method="manual"` requires `lows` and `highs`, one bound for each
of the `attrs`, which default to the assay's QC columns as for the other methods, with an optional
bool `keep_bounds`. Probability, MAD, and manual-bound values must be finite numbers when present;
bools and numeric strings are rejected rather than coerced.

```python
run = ds.pipeline.rna(
    filtering={
        "method": "manual",
        "attrs": ["RNA_nCounts", "RNA_nFeatures"],
        "lows": [1000, 500],
        "highs": [15000, 4000],
    },
)
```

The run records the filter's resolved parameters as `config["stages"]["filtering"]`, such as
`{"enabled": True, "method": "mad", "attrs": [...], "minP": 0.01, "maxP": 0.99,
"sampleColumn": None, "nMads": 3.0, "minCellsPerSample": 20}`, or `{"enabled": False}` when the
run does not filter.

## Stage contracts

### Embeddings

`umap=False` skips UMAP, and the embedding initialization unless t-SNE, which uses it, runs; a
skipped stage's artifact does not appear in the completed run. UMAP forms the fuzzy union of the
graph itself, so `umap` `symmetric_graph` and `graph_upper_only` must stay `False`. `neighbors`
`ann_ef` sets the search depth of the neighbour query, by default `max(200, 2 * k)` with the
default index; the neighbors artifact records a depth other than its index's own as its `ann_ef`.

t-SNE embeds the run's graph from the same initialization, after the clustering stages; its
coordinates appear as `tsne_1`, `tsne_2`, and so on. A new t-SNE embedding needs the optional
`sgtsnepi` package of the `tsne` extra; without it the `tsne` stage fails, after the earlier stages
have completed, and the run raises a `PipelineExecutionError` whose `__cause__` is the
`ImportError` that names the extra. A run whose t-SNE embedding the store already holds reuses it
without the package, so a run that enables t-SNE without `sgtsnepi` is not refused: the run logs a
warning that names the extra before it creates a run record, and then runs.
`tsne={"parallel": ...}` is rejected as an unknown setting.

### Leiden and Paris

`leiden` runs one stage per resolution of `partitions`, by default 0.5, 0.75, 1.0, and 1.25, or
1.0 for a multimodal run. `leiden=True` and `leiden={}` use the defaults. Partitions must be a
non-empty sequence without duplicate resolutions. `selected` names one of the partitions as the
saved clustering, `run["clusters"]`, in place of the silhouette choice, and the cluster-selection
stage is skipped; the other partitions still run beside it.

`backend` defaults to `"cytearc"`, CyteArc's own multi-threaded Leiden implementation, which runs on
the datastore's worker budget and gives the same labels for any thread count. `"igraph"` and
`"leidenalg"` need those GPL-licensed packages, which CyteArc does not install. Without them, a
Leiden stage that must compute a new partition with either backend fails, after the earlier stages
have completed, and the run raises a `PipelineExecutionError` whose `__cause__` is the
`ImportError` naming the missing package; a partition that the store already holds is reused
without the packages.

Doublets, markers, and membership strength require at least one Leiden candidate. Paris can still
run as a diagnostic output when those stages are off: `ds.pipeline.rna(leiden=False, paris=True,
doublets=False, markers=False)` has `run["paris"]` and no `run["clusters"]`. Membership strength
is computed on the saved clustering and appears as `membership_strength`: the fraction of each
cell's graph neighbors that carry the cell's own cluster label.

### Graphs without PCA

`pca={"dims": 0}` skips the `pca` stage of an `rna` run. The graph is then built on the normalized
values of the selected features, one coordinate per feature: the ANN index, the neighbors, the
embedding initialization, and cluster selection use `run["normalized"]` as their coordinates, and
the run records no `pca` output. Harmony corrects reduced coordinates and doublet scoring needs a
PCA graph, so `dims=0` with `harmony`, or with doublet scoring on, which it is by default, raises
`ValueError` before a run record is created; pass `doublets=False`.

With `dims=0` the graph has one coordinate per highly variable gene, so the memory and time of the
ANN index, the neighbour queries, the embedding initialization, and cluster selection scale with
`hvg` `top_n` instead of the PCA dimensions. The hnswlib index holds about
`4 * top_n + 8 * ann_m + 130` bytes per cell on x86-64 Linux, about 4.3 GB for 1,000,000 cells
with 1,000 genes and 8.3 GB with 2,000 at the default `ann_m=24`, and the `ann_index` and
`neighbors` stages fail before they create or load it when it does not fit the memory budget: the
run raises a `PipelineExecutionError` whose `__cause__` is the `MemoryError`. By default the
embedding initialization reads the coordinates in their stored row bands, and fits k-means in memory
over every cell's coordinates, 4 bytes per gene per cell, when one band holds every cell; set
`embedding_initialization={"batch_size": ...}` below the cell count to select the streamed fit,
which holds a sample of the cells and one block of coordinates at a time.

### Doublets and Harmony

`harmony={"batch_columns": [...]}` corrects the run's reduced coordinates, PCA or LSI, and the main
graph then uses the Harmony coordinates. Doublets form only within one capture, so when an `rna` run
with Harmony scores doublets, it simulates and scores them within each batch, the cells that share
every Harmony batch label, on the uncorrected PCA coordinates that Harmony started from; no other
graph is built. `doublets={"sample_column": ...}` names the column of each cell's capture instead,
and cannot be None with Harmony. Without Harmony, doublets are scored within each sample of that
column, or across every cell when it is omitted. The input snapshot stage freezes the sample labels
as its `doublet_sample_snapshot` output and the Harmony batch labels as its `harmony_snapshot`
output, apart from the cell snapshot that filtering reads ([Run snapshots](#run-snapshots)), and the
run configuration records the doublet sample columns as `doubletSampleColumns`.

The `doublets` stage simulates doublets from any two cells of a sample by default. With
`doublets={"heterotypic_fraction": ...}` above 0, it forces that share of the simulated doublets to
pair parents from two different clusters, so it needs a saved clustering with two or more
clusters. The silhouette choice never saves a partition with one cluster, but `leiden` `selected`
can. The stage then fails with a `PipelineExecutionError` that names the settings to change: select
a partition with two or more clusters, set `doublets={"heterotypic_fraction": 0}` to simulate
doublets from any two sampled cells, or pass `doublets=False`. A run with doublets has the frozen
cell fields `doublet_score` and `predicted_doublet`, the call of each cell
([Doublet detection](../reference/methods#doublet-method)).

## Automatic cluster selection

The `cluster_selection` stage of the `rna`, `atac`, and `adt` recipes scores the Leiden candidates
in the coordinates that built the graph: PCA, LSI, or Harmony coordinates, or with
`pca={"dims": 0}` the normalized values. A multimodal run has no automatic selection: its integrated
graph has no shared coordinates, so its only or `selected` Leiden candidate is `run["clusters"]`,
and its `cluster_selection` stage is always skipped. Paris remains `run["paris"]` for diagnosis and
comparison; it is never the automatic `run["clusters"]` winner.

Selection uses one deterministic shared sample of at most 10,000 selected cells with seed `4466`.
The sample reserves up to two seeded cells per cluster across every Leiden candidate, then fills
remaining capacity without replacement. The sampled cells' coordinates are read in blocks under the
datastore memory budget, and pairwise distances are computed in chunks of a third of the budget
that remains after the sampled coordinates in float64 and a small allowance for each sampled cell,
at most 1 GiB. Before it reads a coordinate, the stage fails when the sampled coordinates, in their
stored dtype and as float64, with one block of the read or with the distance chunks, exceed the
budget: the run raises a `PipelineExecutionError` whose `__cause__` is the `MemoryError`. Reading
a sample spread over a large normalized matrix, as with `pca={"dims": 0}`, decodes most of its
stored chunks.

This silhouette comparison is a reproducible provisional baseline. It is not biological validation
or ground truth. Keep alternative Leiden refs and Paris when the study question needs other
evidence.

The resulting `cluster_selection` artifact records candidate keys and refs, silhouette scores,
the sample definition (`sampleStrategy="sharedClusterQuota"` and `minClusterQuota=2`),
invalid-candidate reasons, deterministic tie order, and the selected key.
`run["clusters"]` is the selected Leiden candidate's exact ref. It is not a copied artifact or
metadata column. If no Leiden candidate can be scored, the stage fails. A single valid Leiden
candidate still receives a decision artifact.

Persisted cluster selection is intentionally pipeline-only. The recipes own the candidate set,
deterministic sampling policy, and run-ledger decision. Granular callers can evaluate their own
partitions with public APIs such as `evaluate_cluster_separability` in `cytearc.metrics`. There is
no public "best clustering" function and no separate `DataStore` persistence method that could
imply the pipeline policy outside a run.

## Durable outputs and frozen views

A completed run exposes its outputs in a fixed order; an output that the run did not produce is
absent. A default `rna` run has:

```text
input_cell_selection, analysis_cell_selection, feature_universe, cell_cycle,
highly_variable_features, normalized, pca, ann_index, neighbors, connectivity_map,
embedding_initialization, umap, leiden_0.5, leiden_0.75, leiden_1.0, leiden_1.25,
cluster_selection, clusters, doublets, markers
```

Optional stages add their outputs in place: `cell_calling`, `called_cell_selection`, `hashing`, and
`hashed_cell_selection` after `input_cell_selection`; `harmony` after `pca`; `paris` after the Leiden
candidates; and `membership_strength` and `tsne` after `clusters`. A run with `leiden` `selected`
has no `cluster_selection` output.

A default `atac` run has:

```text
input_cell_selection, analysis_cell_selection, feature_universe, prevalent_peaks, normalized, lsi,
ann_index, neighbors, connectivity_map, embedding_initialization, umap,
leiden_0.5, leiden_0.75, leiden_1.0, leiden_1.25, cluster_selection, clusters, markers
```

A default `adt` run has the same outputs with `panel` in place of `prevalent_peaks` and `pca` in
place of `lsi`. A default WNN `multimodal` run of an `RNA` run and an `ADT` run has:

```text
input_cell_selection, analysis_cell_selection, feature_universe, integrated_graph,
embedding_initialization, umap, leiden_1.0, clusters, markers_RNA, markers_ADT
```

Its `input_cell_selection` and `analysis_cell_selection` are both the selection that its runs
share, and its `feature_universe` is the first run's. A default `perturbation` run has:

```text
input_cell_selection, analysis_cell_selection, feature_universe, guide_thresholds, guide_calls,
perturbation_labels, efficiency, highly_variable_features, normalized, pca, edistance
```

Its optional stages add `harmony` after `pca`, and `signature`, `mixscape`, `associations`, `profiles`, and `perturbation_clusters`, in that order, after `edistance`. Calls that fit no thresholds, with `method="fixed"` or `"top_guide"` or imported from Cell Ranger, have no `guide_thresholds`.

`run.recipe` names the recipe, `"rna"`, `"atac"`, `"adt"`, `"multimodal"`, or `"perturbation"`,
and `run.assay` the analyzed assay, the RNA assay of a perturbation run, or the first run's assay
for a multimodal run. Normal mapping operations such as `run["pca"]`, `list(run)`, and
`run.items()` are available only on a successfully completed run.

### Run snapshots

The `input_snapshot` stage freezes the metadata that later stages and the run's fields read, so a
later change to a live column never changes the run's results or fields. Its snapshots are
outputs of the stage, which its stage record lists, not run outputs: `run["cell_snapshot"]` raises
`KeyError`. Besides the cell and feature snapshots, each other group of cell columns has a snapshot
of its own, which the stage writes only when the run reads that group:

| Stage output | Holds | Read by |
| --- | --- | --- |
| `cell_snapshot` | the cell IDs and `names`, and the QC columns that filtering reads: its `attrs` and a MAD `sample_column` | the `filtering` stage, and the run's `ids` and `names` cell fields |
| `feature_snapshot` | the feature IDs and `names` of the run's assay | the stages that name features, such as `highly_variable_features`, `cell_cycle`, and `markers`, and the run's feature fields |
| `capture_snapshot` | the capture column of per-capture cell calling, `cell_calling` `sample_column` | the `cell_calling` stage |
| `harmony_snapshot` | the Harmony `batch_columns` | the `harmony` stage |
| `doublet_sample_snapshot` | the columns whose labels split doublet scoring, `doubletSampleColumns` | the `doublets` stage |
| `column_snapshot` | the run's `snapshot_columns` | the run's cell fields of those columns |

A column that one stage reads therefore never changes the identity of what another stage computes:
runs that differ only in their Harmony batch columns or their `snapshot_columns` filter to the same
`analysis_cell_selection` artifact and share every artifact that does not depend on that
difference, and per-capture cell calls stay the same when the filtering or snapshot settings
change. A multimodal run's `cell_snapshot` holds the cell IDs and `names` only, beside a
`column_snapshot` when the run names `snapshot_columns`, and its input stage stores a
`feature_snapshot_<assay>` and a `feature_universe_<assay>` for each source run's assay.

### Frozen fields

`run.cells` and `run.features` are narrow read-only views over captured selections, metadata
snapshots, and result artifacts:

```python
view.columns
view.fetch(column)                 # rows selected by the run's stored I
view.fetch_all(column)             # values aligned to the complete stored axis
view.to_pandas_dataframe(columns)  # selected rows
view.head(n=5)
```

| Field | View | Present when |
| --- | --- | --- |
| `I`, `ids`, `names` | cells and features | always |
| the `snapshot_columns` | cells | named |
| `s_score`, `g2m_score`, `cell_cycle_phase` | cells | an `rna` run scores the cell cycle |
| `umap_1`, `umap_2`, ... | cells | UMAP runs |
| `tsne_1`, `tsne_2`, ... | cells | t-SNE runs |
| `leiden_<resolution>` | cells | one per Leiden candidate |
| `paris` | cells | Paris runs |
| `clusters` | cells | a Leiden candidate is saved |
| `membership_strength` | cells | `membership_strength=True` |
| `doublet_score`, `predicted_doublet` | cells | an `rna` run scores doublets |
| `weight_<assay>` | cells | a WNN `multimodal` run, one per assay |
| `guide_call`, `perturbation_label`, `is_control` | cells | a `perturbation` run |
| `mixscape_class`, `mixscape_posterior` | cells | a `perturbation` run calls Mixscape |
| `highly_variable_features`, `prevalent_peaks`, or `panel` | features | an `rna` or `perturbation` run, an `atac` run, or an `adt` run |

Cell `I` is the analysis selection. Feature `I` is backed by the immutable
`run["feature_universe"]` selection created during the input stage, not the live feature `I`
column; a multimodal run's feature view covers its first run's assay with `I`, `ids`, and `names`
only. The views retain captured names and requested metadata even if live values or live `I` later
change. They fail closed if ordered cell or feature identities change. There is no live overlay or
mutation surface.

Every assay of a store shares its cells, so a cell field can read an artifact of any assay, such as the guide calls of a `perturbation` run, which belong to the guide assay. The cell view checks that each artifact was computed on the run's cells, as its recorded cell selection shows, and refuses one computed on other cells. Each assay has its own features, so a feature field reads only artifacts of the run's assay. A run opens both views before it completes, so a field that a view refuses fails the run at `"finalize"`, and the run does not take its label.

`snapshot_columns` cannot use the name of a field that the run produces or could produce: `I`, `ids`, `names`, `s_score`, `g2m_score`, `cell_cycle_phase`, the run's `umap_<n>`, `tsne_<n>`, and `leiden_<resolution>` fields, `paris`, `clusters`, `membership_strength`, `doublet_score`, `predicted_doublet`, and the recipe's feature field, or for a multimodal run its `weight_<assay>` fields. A `perturbation` run also reserves `guide_call`, `perturbation_label`, and `is_control`, and `mixscape_class` and `mixscape_posterior`, which a run without Mixscape does not produce.

Pipeline execution writes immutable artifacts and run records only. It never inserts result
columns or rewrites live `I`. Plotting, marker loading, and export stay on `DataStore`:

```python
ds.plots.embedding(run=run, layout="umap", color_by="clusters")
markers = ds.markers.load(marker=run["markers"], group_id=1)
adata = ds.to_anndata(run=run)
# adata.obsm["X_umap"] holds frozen UMAP; cluster labels stay in adata.obs
```

A run export writes an RNA or ADT run's PCA as `obsm["X_pca"]` and an ATAC run's LSI as
`obsm["X_lsi"]`, with their Harmony corrections as `obsm["X_pca_harmony"]` and
`obsm["X_lsi_harmony"]`. `matrix="normed"` exports the run's normalized values over its feature
field's features. A multimodal run exports the raw counts of its first run's assay and has no
normalized output ({doc}`import_export`).

The marker table of a run saved before CyteArc 1.0.0 cannot be read, because its `fold_change`
column held sentinels: `markers.load` raises and names
`ds.markers.search(run["clusters"], features=run["feature_universe"])`, which recomputes the
table for the run's clusters.

### Run configuration

`run.report()["run"]["config"]` records how the run was called. For `rna`, `atac`, and `adt`, it
holds `cells` (`{"column": ...}`, `{"selection": ref}`, or `{"run": {"runId", "label"}, "selection":
ref}`), `snapshotColumns`, `species`, `doubletSampleColumns` (`rna` only), `stages`, and
`shutdown`, the signal protection of the run. `stages` maps each stage keyword to the settings
that the call gave it, `False` for an optional stage that the call turned off, and a bool for
`membership_strength` and `markers`. The recipe resolves a few of them: `pca` records its `dims`,
`prevalent_peaks` its `top_n`, `leiden` its `partitions` and `selected` resolution, `panel` its
`exclude` pattern and the `excluded` names, `hashing` its `assay`, and `filtering` the filter's
parameters. A multimodal run records `method`, `sources` (each run's `runId`, `label`, `recipe`,
and `assay`), `cells` (`{"selection": ref}`), `snapshotColumns`, `stages`, and `shutdown`.

A `perturbation` run records `cells`, `snapshotColumns`, `guideAssay`, the guide assay, `profileLabels`, the output whose cells the profiles average (`"mixscape"` for knocked-out cells, the default in a run with Mixscape, `"perturbation_labels"` for every labelled cell, and None without clusters), `stages`, and `shutdown`. Its `stages` record `False` for `signature` without Mixscape and for `profiles` without clusters, and a `guide_calling` `path` as the text of the path.

## Open, list, and report runs

A label is optional. A run that completes takes its label as an immutable name, unique in the
store:

```python
run = ds.pipeline.open(label="baseline")
same_run = ds.pipeline.open(run_id=run.run_id)

recent = ds.pipeline.list_runs(limit=20)
failed = ds.pipeline.list_runs(status="failed")
unfinished = ds.pipeline.list_runs(status=["running", "interrupted"])
```

`open` requires exactly one of `run_id` or `label`. Only a successfully completed run acquires its
requested label. Failed and interrupted attempts do not reserve it. `list_runs` returns newest
first and includes every status, `running`, `completed`, `failed`, and `interrupted`, unless
`status` names one or a sequence of them. Concurrent finalizers use an atomic label claim, so at
most one completed run can acquire a name. A storage backend without atomic conditional creation
rejects `label=` before a run record or computation starts; unlabeled runs are unaffected. An
unclean incomplete finalizer blocks reuse of its requested label and fails closed.

After confirming that the owner process has stopped, explicitly abandon that exact claim before
retrying the label:

```python
interrupted = ds.pipeline.abandon_label_claim(
    label="baseline",
    run_id=stopped_run_id,
    reason="worker terminated after finalization began",
)
replacement = ds.pipeline.rna(label="baseline")
```

The recovery call succeeds only for the exact current owner while `complete=False`, including a
torn terminal payload, and records an `abandoned_label_claim` interruption. It refuses every
`complete=True` owner and has no timeout or automatic stale-process heuristic.

Every status exposes identity, status, recipe, and `run.report(format="dict" | "markdown")`. Only a
completed run exposes mapping outputs and frozen views. Reports include stage timing, sampled
process-tree RSS, artifact plans with `created` or `reused` dispositions, stage summaries and
warnings ([Diagnostics](#diagnostics)), failures, interruption details, and signal-guard
availability. RSS peaks are sampled lower bounds. An unavailable measurement is reported as null
with a reason. RSS comes from the Linux `/proc` filesystem; on platforms without it, such as macOS
and Windows, nothing is sampled, so `sampleCount` is 0.

A hard process death can leave `complete=False`; this is reported as an unclean incomplete run.
There is no resume, repair, or same-ID retry. A new invocation may reuse only complete artifacts.

### Runs saved by other releases

A run completed under any 1.x release reopens under every later 1.x release. `open` by label or
run ID, `list_runs`, `run[key]`, `run.report()`, and the frozen `run.cells` and `run.features`
views work as long as the artifacts the run references exist and satisfy their payload contracts.

Runs that `pipeline.run()` saved open unchanged. Their `run.recipe` is `"basic_rna_analysis"`, and
their configuration keeps the shape that their release recorded, such as `cellKey`, `hvgCount`,
`pcaDims`, and `params`. Reopening never checks a configuration against the current recipes, so
such a run reopens and reports although no current recipe accepts its configuration. To run it
again, translate its configuration into `ds.pipeline.rna` keywords with the
[migration table](#migrate-from-pipeline-run); a new run never modifies the earlier one. Their
stage records have no summaries or warnings, which their reports show as None and an empty list.

Running a configuration again recomputes a stage whose operation gained a revision that applies to
it, or whose recorded provenance a later release changed. Lineage reports of the earlier run's
outputs mark those that a revision supersedes `stale`. A changed default changes the recorded
provenance too. The default Leiden backend changed from `"igraph"` to `"cytearc"`, so a run that
does not set `leiden={"backend": ...}` computes new Leiden partitions where an earlier release
stored igraph partitions, and the stages that use them, such as cluster selection, doublet scoring,
and marker search, compute new results too. The earlier partitions are not superseded or marked
`stale`: set `leiden={"backend": "igraph"}` to reuse them, which needs no igraph installation while
the store holds them.

A run whose records a newer CyteArc saved with fields this release does not know fails closed:
opening it by run ID, or reading its stage records with `report()`, raises a `ValueError` that
names those fields. Upgrade CyteArc to read it. Within 1.x, releases only add record fields, so runs
from earlier releases never meet this error. The pipeline recipes added the stage fields `summary`
and `warnings`: a 1.x release from before the recipes opens the runs that they save, but its
`report()` raises on those fields. A 2.0 release may change run records and lists each change in
its compatibility inventory.

A cell field can read an artifact of another assay than the run's, as the guide fields of a `perturbation` run do ([Frozen fields](#frozen-fields)). Such a field is an ordinary entry of the record's `fields`, so run records keep their shape, with no new record field, schema version, or migration, and every saved run reads as before. CyteArc 1.3.0 and earlier releases read cell fields of the run's assay only: they refuse the cell view of a run with such a field, with an `ArtifactResolutionError` whose code is `pipeline_field_axis_mismatch`, while its outputs and feature view open. They do not know the artifact kinds of a `perturbation` run's guide calls and labels, such as `guide_calls`, so they open no `perturbation` run. A run keeps the fields that it recorded and gains none later: a `perturbation` run saved without the guide fields gets them only from a new run of the recipe with the same settings, under another label or none, which reuses every artifact of the earlier run.

Run and stage records are otherwise strict: a missing or unknown field, or a malformed value,
fails closed. `list_runs` and open-by-label skip records they cannot read so healthy runs remain
accessible, and log a warning that names each skipped run ID and why it could not be read. Opening
that run by its exact ID raises its error.

## Diagnostics

A completed stage records a small summary when its outputs have diagnostics. The summary is
computed from the artifacts that the stage produced, so a reused artifact reports the same numbers
as a computed one, and it is computed within the stage, so its time, memory, and warnings belong to
that stage. A summary reads attributes and bounded blocks of small arrays, never a matrix. A stage
with one summarized output records that output's summary; a stage with several records one summary
for each output key. Metadata snapshots and ANN indexes have no diagnostics, so `input_snapshot`
reports only its input cells and feature universe, and `ann_index` records None. A summary that
cannot be computed is logged as a warning and left out. For example:

| Stage | Summary |
| --- | --- |
| `input_snapshot` | `input_cell_selection` (`cells`), when the stage made the selection, and `feature_universe` (`features`); for a multimodal run, `features` of each `feature_universe_<assay>` |
| `cell_calling` | `cell_calling` (`retain`, `knee`, `inflection`, `n_ambient`) and `called_cell_selection` (`cells`, `inputCells`, `removedCells`); per capture, `cells`, `inputCells`, `removedCells`, and each capture's called cells and knee as `calledPerCapture` and `kneePerCapture` |
| `filtering` | `cells`, `inputCells`, `removedCells`, and `samples`, the number of MAD groups |
| `hashing` | `hashing` (`hashtags`, `negative`, `multiplets`) and `hashed_cell_selection` (`cells`, `inputCells`, `removedCells`) |
| `pca` | `cells`, `dims`, `varianceRatio` (at most 50 components), `varianceExplained` |
| `leiden_<resolution>` | `clusters`, `largestCluster`, `smallestCluster` |
| `cluster_selection` | `selected` and the silhouette `scores` of each candidate |
| `membership_strength` | `mean`, and `belowHalf`, the number of cells whose membership strength is below 0.5 |
| `doublets` | `predictedDoublets`, `rate`, and `samples`, with each sample's threshold and calls |
| `integration` | `assays` and `meanWeights`, the mean modality weight of each assay |
| `guide_calling` | `guide_calls` (`cells`; `negative`, `single`, and `multiple`, the cells with no, one, and several calls; `calledGuides`, the guides called in some cell; and for imported calls `fileBarcodes` and `unmatchedBarcodes`, the file's barcodes that match no analysis cell, those of the cells that filtering removed included), and for fitted calls `guide_thresholds` (`samples`, 1 without a `sample_column`, `guides`, and `statuses`, the fits of each status) |
| `perturbation_labels` | `cells`, `controls`, `negative`, `multipleTargets` (cells whose guides target several genes), `targets` (the targets that are not controls), `targetCells` (the `min`, `median`, and `max` of each target's cells that carry it alone), `fewCellTargets` (the targets with fewer than 25 such cells and their cells, fewest first, at most 50), and `fewCellTargetCount` |
| `efficiency` | `by`, `rows`, `statuses`, `knockedDown` (tested rows with an adjusted p-value below 0.05 and a remaining fraction below 1), `medianRemainingFraction` (of the tested rows), and `noFeature` (the rows without a feature, at most 50) |
| `edistance` | `by`, `rows`, `statuses`, `significant` (tested rows with an adjusted p-value below 0.05), `atFloor` (tested rows at the smallest p-value of their permutations, 1 / (`n_permutations` + 1)), and `nPermutations` |
| `signature` | `cells`, `features`, `controls`, `samples` (the samples of `sample_column`, which the loader's `cells["sample"]` names, 0 without one), `neighbors` |
| `mixscape` | `rows` (targets by split), `statuses`, `cells`, `knockedOut`, `fewDeGenes` (the targets with too few DE genes in some split, at most 50), `perturbationType` |
| `associations` | `discoveryPairs`, `testedDiscoveryPairs` (those that pass QC), `discoveries`, `calibrationPairs`, `falseDiscoveries`, `calibrationGrade`, `calibrationStatus`, `falseDiscoveriesByControlGuides` (the false discoveries of each group of control guides, most first, at most 50), `positivePairs`, `positiveDetected`, `alpha` |
| `profiles` | `perturbations`, `features`, `leftOut`, `zeroSdFeatures` (features without spread over some sample's control cells), `samples` (the rows of the loader's `samples`, 1 without a `sample_column`) |
| `perturbation_clusters` | `perturbations`, `clusters`, `unassigned`, `largestCluster` |

The perturbation summaries count adjusted p-values below 0.05 and targets with fewer than 25 cells, fixed thresholds; the loaders of {doc}`perturbation` read every row for others.

Each stage record also keeps the warnings that CyteArc logged or issued while the stage ran,
oldest first: at most 50, each cut to 512 characters. The stage's own worker threads report into
it; warnings that unrelated work logs on other threads or asyncio tasks at the same time are not
collected. Only a completed stage holds a summary; a failed or interrupted stage keeps its
warnings.

`run.report()` returns three entries:

- `run`: the run record, with `runId`, `recipe`, `requestedLabel`, `label`, `assay`,
  `startedAtNs`, `finishedAtNs`, `status`, `complete`, `cytearcVersion`, `config`, `stageOrder`,
  `outputs`, `fields`, `error`, and `interruption`;
- `stages`: each stage record written so far, skipped stages included, with `stage`, `ordinal`,
  `status`, its times, `outputs`, `plans`, `metrics`, `error`, `interruption`, `summary`, and
  `warnings`;
- `summary`: `createdArtifacts`, `reusedArtifacts`, `uncleanIncomplete`, `signalProtection`, and
  `warnings`, a list of `{"stage", "message"}` entries for every stage's warnings in stage order.

`run.report(format="markdown")` adds a `Warnings` column with each stage's count to its stage
table and a `## Diagnostics` section with each stage's summary as JSON and its warnings.

## Events and listeners

A recipe's `callback`, and every listener registered with {py:func}`cytearc.pipeline_listener`,
receives a {py:class}`~cytearc.datastore.pipeline_accessor.PipelineEvent` for each step of a run.
Events report a run; they cannot change it. To stop a run, use a
[cancellation token](#cancellation).

```python
import cytearc

events = []

with cytearc.pipeline_listener(events.append):
    rna = ds.pipeline.rna(label="rna")
    adt = ds.pipeline.adt(cells=rna, label="adt")

[(event.recipe, event.kind, event.stage) for event in events]
```

Every event carries `kind`, `run_id`, `recipe`, `label` (the requested label, or None), `stage`,
`ordinal` (the stage's position in `run.report()["run"]["stageOrder"]`), `stages` (how many stages
the run has, skipped stages included), and `time_ns`. The other fields depend on the kind:

| Kind | Emitted | Other fields |
| --- | --- | --- |
| `run_started` | after the run record is created | |
| `stage_started` | after a stage's record is written, before the stage runs | |
| `stage_progress` | while a stage runs, as a CyteArc progress bar advances | `progress` |
| `stage_completed` | after the stage's outputs are recorded | `metrics`, `summary`, `outputs`, `reused`, `warnings` |
| `stage_skipped` | after a stage that the run does not compute is recorded as skipped: one that the call turned off, or one that the recipe skips, such as `cluster_selection` with a `selected` resolution, `embedding_initialization` without UMAP or t-SNE, `pca` with `dims=0`, or a perturbation run's `signature` without Mixscape and `profiles` without clusters | `metrics` |
| `stage_failed` | after the stage and the run are recorded as failed | `metrics`, `warnings`, `error` |
| `stage_interrupted` | after the stage and the run are recorded as interrupted | `metrics`, `warnings`, `error` |
| `run_completed` | after the run record is complete | |
| `run_failed` | after the run is recorded as failed | `error` |
| `run_interrupted` | after the run is recorded as interrupted | `error` |

`outputs` holds the `(key, ref)` pairs that a completed stage produced, and `reused` the keys whose
artifacts were reused rather than computed. `metrics` and `summary` are the values that the stage
record holds. `error` is the exception: the stage's own for `stage_failed`, the
`PipelineExecutionError` for `run_failed`, and the interruption for `stage_interrupted` and
`run_interrupted`.

A run emits `run_started` first. Each stage then emits `stage_skipped` alone, or `stage_started`,
any number of `stage_progress` events, and one of `stage_completed`, `stage_failed`, or
`stage_interrupted`, before the next stage emits anything. The run then ends with one run event:
`run_completed` after the last stage, `run_failed`, or `run_interrupted`. `run_started` and
`run_completed` name no stage.

- `run_failed` always names a stage. It follows `stage_failed` and names that stage. A failure to
  write a stage's first record emits it without a stage event, and a failure to complete the run
  record names the step `finalize`, which has no ordinal. An error while a skipped stage is
  recorded emits `stage_failed` without `stage_started`.
- `run_interrupted` names a stage only when the interruption cut that stage's work short: it then
  follows `stage_interrupted`. An interruption between two stages or while a skipped stage is
  recorded emits `run_interrupted` alone, with no stage. So does one that a stage notices just after
  its work finished: that stage is recorded as completed and emits `stage_completed` first.

Events are delivered synchronously, on the thread that runs the recipe, after the durable state
that they describe is written. The run's own `callback` is called first, then the listeners in the
order they were registered. A slow callback delays the run, so hand events to a queue for slow work
such as updating a remote UI. Callbacks run in a copy of the context in which the recipe was called,
taken before the run installs its shutdown and cancellation handling, and outside every stage. What
a callback does, such as logging a warning, drawing a progress bar, or computing an artifact, never
enters a stage's record or events, and a pending cancellation or signal never interrupts the
CyteArc work that a callback does. An exception from a callback or listener, `BaseException`
included, is logged and never changes the run.

`pipeline_listener(listener)` is a context manager. Its registration is context-local: it covers
the runs started in the `with` block on that thread or asyncio task, and registrations nest. A run
reads its listeners when it starts, so a listener registered later does not receive its events. A
run started on another thread receives a listener only when that thread runs in a copy of the
registering context, as `asyncio.to_thread` and `contextvars.copy_context().run` provide.

`event.to_dict()` returns the event as JSON-compatible values for a UI or a log, with camelCase
keys: `kind`, `runId`, `recipe`, `label`, `stage`, `ordinal`, `stages`, `timeNs`, `progress`,
`metrics`, `summary`, `outputs` as `{"key", "artifact", "reused"}` entries, `warnings`, and `error`
as `{"type", "message"}` with the message cut to 512 characters. `PipelineEvent` and
`PipelineProgress` are frozen dataclasses, and an event pickles when its `error` does.

### Progress events

While a stage runs, the CyteArc progress bars that advance on the run's thread report to the run's
callbacks and listeners as `stage_progress` events. Their `progress` is a
{py:class}`~cytearc.datastore.pipeline_accessor.PipelineProgress`: the bar's `description`, or
None, the units `completed`, and the `total` units, or None when the bar has no total. A stage
emits at most one progress event every 0.25 seconds for each progress bar, so bars that share a
description are throttled apart, and it always emits a bar's first update and the update that
reaches the bar's total; later updates past the total are throttled like the others. Progress
events are emitted while bars are hidden with `configure_output(progress=False)` too, and only when
the run has a callback or listener. A stage that streams the matrix in blocks, such as HVG
statistics or marker ranking, advances its bar on the run's thread as each block's result is used.
Work that draws no CyteArc bar emits none: UMAP's training bar belongs to umap-learn and is not
forwarded.

## Cancellation

A {py:class}`~cytearc.CancellationToken` stops a run from any thread, such as a UI's stop button:

```python
import threading

import cytearc

token = cytearc.CancellationToken()
outcome = {}


def analyze() -> None:
    try:
        outcome["run"] = ds.pipeline.rna(label="rna", cancel=token)
    except cytearc.PipelineCancelled as error:
        outcome["cancelled"] = error


worker = threading.Thread(target=analyze)
worker.start()
token.cancel("stopped from the UI")
worker.join()
```

The run stops at its next safe checkpoint. CyteArc checks between stages and between the bounded
blocks of streamed work, so a step that holds no checkpoint finishes before the run stops. The run
records the stage that was running, if any, as completed when its work had finished and otherwise
as interrupted, and records the run as interrupted, with interruption kind `cancelled` and the
reason as its message. It emits `run_interrupted` after that stage's event, and raises
{py:class}`~cytearc.PipelineCancelled`, a `RuntimeError` whose `run_id`, `stage`, and `reason`
name the run, the stage whose work the cancellation cut short, and the reason. Like
`run_interrupted`, it names no stage, `stage` None, when the run stopped between stages or just
after a stage's work finished. Completed stages keep their artifacts, so a new run with the same
settings reuses them.

A cancellation that arrives once the run record is complete comes too late to stop the run, and
the call returns the completed run. One that races with a stage failure leaves the run failed and
raises `PipelineExecutionError`.

A token stays cancelled. A run given a cancelled token raises `PipelineCancelled` with `run_id` and
`stage` None before it writes anything; its call is still validated first. `cancel()` returns True
only for the token's first cancellation, and `reason` keeps the first reason. One token may stop
several runs at once. A signal that stops a run never cancels the token, and a cancellation never
runs a prior signal handler. `PipelineCancelled` pickles, so it can cross a process boundary.

A signal that arrives after a cancellation does not escalate, because it is the run's first signal.
The run still stops at its next checkpoint and is recorded as interrupted with the kind of the
first request, `cancelled`. Once that is written, the signal takes its usual effect: its prior
handler runs, or the signal is raised again. With the default handlers, the call therefore raises
`KeyboardInterrupt` after `SIGINT`, and `SIGTERM` or `SIGHUP` terminates the process, instead of
the call raising `PipelineCancelled`.

## Graceful interruption and signals

On the main interpreter thread, a run temporarily installs cooperative handlers for available
`SIGTERM`, `SIGINT`, and `SIGHUP`, while respecting signals already set to ignore and handlers
installed outside Python. The first signal requests shutdown at a safe checkpoint. The run records
the stage that was running, if any, as completed when its work had finished and otherwise as
interrupted, and records the run as interrupted, with interruption kind `signal`. It emits
`run_interrupted` after that stage's event, naming the stage only when the signal cut its work
short, restores prior handlers, and propagates the original signal behavior. A second catchable
termination signal escalates immediately. A signal that arrives after a cancellation was requested
is the first signal, so it does not escalate ([Cancellation](#cancellation)). A run on another
thread records that signal protection was unavailable, in
`run.report()["summary"]["signalProtection"]`; stop it with a cancellation token.

`KeyboardInterrupt` and an escaped `asyncio.CancelledError` use the same durable interruption
boundary, with interruption kinds `keyboard_interrupt` and `asyncio_cancelled`. Ordinary stage
exceptions produce a failed run instead. Run and stage records hold at most 512 characters of an
error or interruption message. A longer message keeps its first 509 characters followed by `...`;
the raised exception keeps its full message. If a termination signal races with an ordinary failure
or the final successful handoff, the run preserves that durable outcome and still propagates the
pending signal after cleanup.

`SIGKILL`, `SIGSTOP`, out-of-memory termination, power loss, and expired shutdown grace periods
cannot perform cleanup. Their durable contract is an incomplete artifact or run, followed by a new
invocation that reuses only complete artifacts.

## Follow a run from another process

A run writes its run and stage records as it goes, so another process can follow it without a
callback. Open the store read-only and report its running runs:

```python
import cytearc

monitor = cytearc.DataStore("analysis.zarr", zarr_mode="r")

for running in monitor.pipeline.list_runs(status="running"):
    report = running.report()
    current = report["stages"][-1]["stage"] if report["stages"] else None
    print(report["run"]["requestedLabel"], current, len(report["run"]["stageOrder"]))
```

`report()` reads the stage records again on every call, but a `PipelineRun` keeps the run record
that it opened, so open the run again with `monitor.pipeline.open(run_id=...)` to see its status
change. A running run has no `label` until it completes; its `requestedLabel` names the label that
it will claim. A stage record shows `running` while its stage runs and its terminal status,
metrics, summary, and warnings once it ends; stages that have not started have no record yet. A
process that died without cleanup leaves its run `running`, and the record has no heartbeat, so
confirm that the process has stopped before treating such a run as abandoned.

## Public types

```{eval-rst}
.. autoclass:: cytearc.datastore.pipeline_accessor.PipelineAccessor
    :members: rna, atac, adt, multimodal, perturbation, open, list_runs, abandon_label_claim

.. autoclass:: cytearc.PipelineRun
    :members:

.. autoclass:: cytearc.datastore.pipeline_run.PipelineAxisView
    :members: columns, fetch, fetch_all, to_pandas_dataframe, head

.. autoclass:: cytearc.PipelineExecutionError
    :members:

.. autoclass:: cytearc.PipelineCancelled
    :members:

.. autoclass:: cytearc.CancellationToken
    :members:

.. autofunction:: cytearc.pipeline_listener

.. autoclass:: cytearc.datastore.pipeline_accessor.PipelineEvent
    :members:

.. autoclass:: cytearc.datastore.pipeline_accessor.PipelineProgress
    :members:
```

See [Quick start](../quickstart) for an executable first run and {doc}`graph_construction` for
explicit stage-by-stage APIs.
