---
description: Translate a Scanpy workflow to CyteArc and move data through H5AD.
---

(scanpy-users)=
# CyteArc for Scanpy users

This guide maps familiar Scanpy stages to CyteArc's store-backed API. Use the workflow map
below to find the equivalent task, or try the {ref}`Quick start <quickstart>` to inspect a prepared
result. For optional background on how CyteArc stores and reuses results, see
[](scanpy_and_seurat.md).

## Workflow map

Scanpy commonly composes stages as separate `sc.pp`, `sc.tl`, and `sc.pl` calls. CyteArc provides
the same level of control through individual methods, while `ds.pipeline.run()` is the shortest
path through the standard RNA workflow. It returns a durable `PipelineRun` with frozen views and
leaves live metadata unchanged.

The rows below map intent, not identical statistical implementations. CyteArc selects highly
variable genes before normalizing on that feature set, which differs from the common Scanpy order
of normalization, log transformation, and then HVG selection.

| Goal | Scanpy | CyteArc |
|---|---|---|
| Load counts | `sc.read_*` returns an `AnnData` | A reader and `*ToZarr` writer create the store; `DataStore` opens it |
| Calculate QC metrics | `sc.pp.calculate_qc_metrics` | Opening a new `DataStore` calculates count and feature metrics, plus detected RNA mitochondrial and ribosomal percentages |
| Filter cells | `sc.pp.filter_cells` or an `obs` mask | `ds.qc.filter` or `ds.qc.auto_filter` returns a cell-selection artifact without deleting cells or changing `I`. Bounds are exclusive: `lows=[200]` removes a cell with exactly 200 genes, which `min_genes=200` keeps, so pass `keep_bounds=True` to match. For imported `total_counts` or `pct_counts_mt` columns, pass `metric_policies` to `qc.auto_filter` |
| Select and normalize features | `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.pp.highly_variable_genes` | `ds.features.hvgs`, then `ds.features.normalize`, using the same cell-selection ref |
| Select genes by raw-count variance or Pearson residuals | `sc.pp.highly_variable_genes(flavor="seurat_v3")`, `sc.experimental.pp.highly_variable_genes(flavor="pearson_residuals")`, with `batch_key` | `ds.features.hvgs(cells, flavor="seurat_v3")` or `flavor="pearson_residuals"`, with `batch=` a cell column or label artifact; batches combine in Seurat's order, that of scanpy's `flavor="seurat_v3_paper"`. Pass `blacklist=""` to keep every gene name, as scanpy does |
| Normalize to Pearson residuals | `sc.experimental.pp.normalize_pearson_residuals` | `ds.RNA.normMethod = cytearc.assay.norm_pearson_residuals`, then `ds.features.normalize` and `ds.reduction.pca(..., feat_scaling=False)` |
| Run PCA and find neighbours | `sc.pp.pca`, then `sc.pp.neighbors` | `ds.reduction.pca`, then CyteArc's neighbour-graph methods |
| Embed the graph | `sc.tl.umap` | `ds.embeddings.umap` |
| Cluster cells | `sc.tl.leiden` | `ds.clusters.leiden`; Paris hierarchy diagnostics are covered in the advanced clustering guide |
| Find marker genes | `sc.tl.rank_genes_groups` | `ds.markers.search`, then `ds.markers.load` |
| Plot results | `sc.pl.*` | `ds.plots.embedding`, `ds.plots.dotplot`, and other `ds.plots` methods |
| Export an assay | `adata.write_h5ad` | `ds.to_anndata()` or `cytearc.to_h5ad` |

CyteArc marker search reports AUC, two-sided Mann-Whitney p-values, and within-group
Benjamini-Hochberg adjustment over tested features. It is not replicate-aware differential
expression. By default `markers.load` keeps genes detected in at least 20% of a group's cells,
with an AUC of at least 0.6 and an adjusted p-value of at most 0.05, and orders them by
CyteArc's specificity score; `sort_by="auc"` or `sort_by="p_value"` orders them more like
`rank_genes_groups`, which sorts by its z score. See [](tutorials/graph_construction.ipynb) for the complete manual graph chain and
[](tutorials/scrna_seq.ipynb) for stage-by-stage biological interpretation.

## Move data through H5AD

CyteArc reads and writes H5AD, so CyteArc and Scanpy can be used at different stages of one project.
Inspect an H5AD file first, and review which matrix will become the counts:

```python
import cytearc

inspection = cytearc.inspect_h5ad("data.h5ad")
print(inspection.matrixKey, inspection.integerLike)
print(inspection.matrixCandidates)
```

`inspect_h5ad` ranks matrices that hold counts first, preferring `raw/X`, `layers/counts`, or
`layers/raw_counts`, then `X`, then `spliced` counts, and ranks matrices of one kind by their
estimated total. It warns when several matrices hold integer counts, such as the `spliced`,
`unspliced`, `ambiguous`, and `matrix` layers of velocity data. If `matrixKey` is not the raw RNA
counts, inspect again with the right one, for example
`cytearc.inspect_h5ad("data.h5ad", matrix_key="layers/counts")`.

Then import the file into a CyteArc store:

```python
reader = cytearc.H5adReader.from_inspect(
    inspection,
    embedding_roles={"X_umap": "umap", "X_scVI": "coordinates"},
    cluster_keys=("clusters",),
)
imported = cytearc.H5adToZarr(
    reader,
    zarr_loc="data.zarr",
    analysis_assay="RNA",
).dump()
ds = cytearc.DataStore("data.zarr")
ds.plots.embedding(
    layout=imported.embeddingArtifacts["X_umap"],
    color_by=imported.clusterArtifacts["clusters"],
)
```

Selected embeddings and clusterings become immutable artifacts rather than live metadata
columns. Use their returned refs directly or inspect a payload with `ds.artifacts.load(ref)`. The
`"coordinates"` role imports a latent space, such as `X_pca`, `X_scVI`, or `X_pca_harmony`, as
coordinates that `ds.graph.ann_index` builds a graph on. Categorical `obs` columns keep their
category order, unused categories, `ordered` flag, and the colors of `uns["<column>_colors"]`,
which CyteArc's plots then use. A file whose cell IDs repeat, as `anndata.concat` writes without
`index_unique`, imports with `duplicate_cell_ids="make_unique"` passed to `from_inspect`.

Export to an in-memory `AnnData` object or directly to H5AD. Pass the artifacts that Scanpy should
see: coordinates and embeddings go to `obsm`, labels and scores to `obs`, and a connectivity graph
to `obsp` with `uns["neighbors"]` in Scanpy's layout:

```python
adata = ds.to_anndata(
    cell_selection=cells,
    obsm={"X_pca": pca, "X_umap": umap},
    obs={"leiden": leiden},
    graph=graph,
)
cytearc.to_h5ad(ds.RNA, "analysis.h5ad", cell_selection=cells, obsm={"X_pca": pca})
```

Labels such as `leiden` arrive as categoricals of text labels, so `sc.tl.rank_genes_groups(adata,
"leiden")` and `sc.pl.umap(adata, color="leiden")` work directly, and the graph lets
`sc.tl.paga` and `sc.tl.leiden` run without `sc.pp.neighbors`. `obsm["X_pca"]` also writes the PCA
loadings to `varm["PCs"]`. Each artifact must hold every exported cell, so export the cells of the
selection that the analysis used with `cell_selection=`. Raw counts that CyteArc stores as `uint8`
or `uint16` are exported as `float32`, on which arithmetic cannot wrap; pass `dtype=` to choose
another dtype.

To export a completed pipeline's frozen cells, feature universe, and result fields, pass its run:

```python
adata = ds.to_anndata(run=run)
cytearc.to_h5ad(ds.RNA, "pipeline-analysis.h5ad", run=run)
```

Run export writes frozen UMAP coordinates to `obsm["X_umap"]`, the run's PCA to `obsm["X_pca"]`,
and its cluster labels (`clusters`, `leiden_*`, and `paris`) to `obs` as categoricals with string
categories in natural order and the run's colors. A run with `umap=False` does not invent an
embedding. Add the run's graph with `graph=run["connectivity_map"]`. Both calls export the same
object: `ds.to_anndata` holds it in memory, while `cytearc.to_h5ad` streams it to disk and needs
neither the `anndata` package nor memory for the complete matrix. `X` holds raw counts over the
run's feature universe.

`matrix="normed"` means different values with and without a run. With `run`, `X` holds the values
that the run's PCA read: its stored normalized values, the log1p of counts scaled over its highly
variable genes, float32, over those genes only, which then become `var`. Without `run`,
`ds.to_anndata(matrix="normed")` holds the assay's library-size normalized values over all
exported genes, counts scaled to 1,000 per cell without a log transform, and `cytearc.to_h5ad`
rejects it.

`ds.to_anndata()` defaults to active cells and all features. Pass `feature_indexes` or
`feature_names` to subset features. Without `run` or `cell_selection`, `cytearc.to_h5ad` writes
the full assay to disk, including cells with `I=False`, without first constructing an in-memory
`AnnData`.

## Bring results back from Scanpy

A latent space computed with another tool re-enters the store as graph coordinates. Pass an array
in the order of a cell selection, or a pandas DataFrame indexed by cell ID, with a record of the
tool that made it:

```python
import pandas as pd

latent = ds.reduction.import_coordinates(
    pd.DataFrame(adata.obsm["X_scVI"], index=adata.obs_names),
    cell_selection=cells,
    name="X_scVI",
    source={"tool": "scvi-tools", "version": "1.2.0"},
)
graph = ds.graph.connectivity(ds.graph.neighbors(ds.graph.ann_index(latent), k=11))
leiden = ds.clusters.leiden(graph)
```

The record and fingerprints of the values and cell IDs become the artifact's identity and lineage.
See [](tutorials/import_and_export.ipynb) for a complete round trip, format details, and export
options.
