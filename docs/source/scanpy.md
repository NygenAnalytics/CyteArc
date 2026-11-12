---
description: Find the CyteArc call for each step of a Scanpy analysis, the settings that give different results, and how counts and results move between the two through H5AD files.
---

(scanpy-users)=
# CyteArc for Scanpy users

Find the CyteArc call for each step of a Scanpy analysis, the settings where the two give different results, and how counts and results move between them through H5AD files. Where Scanpy writes each result into an `AnnData` object, CyteArc saves it in the store as an {term}`artifact` and returns an {term}`ArtifactRef`, a reference that the next step takes as its input. The [data model comparison](scanpy_and_seurat.md) explains how the two models differ.

## Translate each step

In the calls below, `ds` is a `cytearc.DataStore`, the object that opens a store. `ds.pipeline.rna()` runs the standard RNA workflow in one call, from quality-control filtering to marker genes, and `ds.pipeline.atac()` and `ds.pipeline.adt()` do the same for peak and antibody counts. Each returns a {term}`PipelineRun`, a saved record of the run's settings and results, and leaves `ds.cells`, the cell table that plays the role of `obs`, unchanged. Each stage of a recipe takes the keywords of the method it runs, such as `ds.pipeline.rna(hvg={"top_n": 2000}, pca={"dims": 30}, neighbors={"k": 14})`. The methods in the table run the same steps one at a time:

| Step | Scanpy | CyteArc |
|---|---|---|
| Load counts | `sc.read_h5ad`, `sc.read_10x_h5` | A reader and a writer, such as `H5adReader` and `H5adToZarr`, write a store that `cytearc.DataStore` opens |
| Call cells in raw droplets | No Scanpy function | `ds.qc.call_cells`, which implements EmptyDrops ([](tutorials/cell_calling.ipynb)) |
| Compute QC metrics | `sc.pp.calculate_qc_metrics` | Opening a new store adds each cell's total counts, detected genes, and mitochondrial and ribosomal percentages, such as `RNA_nCounts` and `RNA_percentMito` |
| Filter cells | `sc.pp.filter_cells`, or a mask on `obs` | `ds.qc.filter` with fixed bounds, or `ds.qc.auto_filter` with bounds from the median absolute deviation; both return a cell selection and delete no cells |
| Select and normalize genes | `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.pp.highly_variable_genes` | `ds.features.hvgs`, then `ds.features.normalize` over the same cell selection |
| Select genes on raw counts or Pearson residuals | `sc.pp.highly_variable_genes(flavor="seurat_v3")`, `sc.experimental.pp.highly_variable_genes(flavor="pearson_residuals")` | `ds.features.hvgs(cells, flavor="seurat_v3")` or `flavor="pearson_residuals"`; `batch=` takes the place of `batch_key`, with a cell column or a label artifact |
| Normalize to Pearson residuals | `sc.experimental.pp.normalize_pearson_residuals` | `ds.RNA.normMethod = cytearc.assay.norm_pearson_residuals`, then `ds.features.normalize` and `ds.reduction.pca(..., feat_scaling=False)` |
| Scale and reduce | `sc.pp.scale`, `sc.pp.pca` | `ds.reduction.pca`, which standardizes each gene unless `feat_scaling=False` |
| Correct batches with Harmony | `sc.external.pp.harmony_integrate` | `ds.reduction.harmony` after PCA, then the graph steps |
| Build the neighbor graph | `sc.pp.neighbors` | `ds.graph.ann_index`, `ds.graph.neighbors`, then `ds.graph.connectivity` |
| Embed the graph | `sc.tl.umap` | `ds.embeddings.umap` |
| Cluster cells | `sc.tl.leiden` | `ds.clusters.leiden` ([](tutorials/clustering.ipynb)) |
| Find marker genes | `sc.tl.rank_genes_groups` | `ds.markers.search`, then `ds.markers.load` |
| Plot | `sc.pl.umap`, `sc.pl.dotplot` | `ds.plots.embedding`, `ds.plots.dotplot` |
| Write an H5AD file | `adata.write_h5ad` | `cytearc.to_h5ad`, or `ds.to_anndata()` for an `AnnData` object in memory |

The [neighbor graph tutorial](tutorials/graph_construction.ipynb) runs the chain from PCA to clusters one call at a time, and the [PBMC analysis](tutorials/scrna_seq.ipynb) names the populations of a saved run from their marker genes.

## Which settings give different results

- **Order of gene selection and normalization.** Scanpy usually normalizes every gene, takes the log and then selects highly variable genes. CyteArc selects them first, by default from the variance of library-size normalized values without a log, and `ds.features.normalize` then divides each cell's counts by its total over the selected genes before taking the log.
- **Filter bounds.** `ds.qc.filter` bounds are exclusive: `ds.qc.filter(["RNA_nFeatures"], lows=[200], highs=[None])` removes a cell with exactly 200 detected genes, which `sc.pp.filter_cells(min_genes=200)` keeps; `keep_bounds=True` keeps values equal to a bound. `ds.qc.auto_filter` sets two-sided bounds on the raw values of a metric that CyteArc did not write, such as Scanpy's `total_counts` or `pct_counts_mt`, and warns; `metric_policies` sets each such metric's policy, such as `{"total_counts": {"transform": "log1p", "bound_direction": "lower"}}`.
- **Neighbor count and default sizes.** CyteArc's `k` leaves out the cell itself and Scanpy's `n_neighbors` counts it, so `k=14` gives each cell as many neighbors as Scanpy's default `n_neighbors=15`; CyteArc's default is `k=11`. By default, `ds.features.hvgs` selects 1,000 genes, and `ds.reduction.pca` keeps 21 components where `sc.pp.pca` keeps 50. `ds.pipeline.rna()` clusters at Leiden resolutions 0.5, 0.75, 1.0 and 1.25 and saves the one with the best silhouette as the run's `clusters`; `leiden={"selected": 1.0}` saves resolution 1.0, the default of `sc.tl.leiden`, instead.
- **Excluded gene families.** By default, `ds.features.hvgs` never selects mitochondrial, ribosomal protein or replication-dependent histone genes, human HLA or mouse H2 genes, cyclins such as `CCNA2` and `CCND1`, or ten sex-linked genes such as `XIST` and `DDX3Y`; it recognizes each family by gene name. `blacklist=""` keeps every gene, as Scanpy does.
- **Batch-aware selection.** With `batch=`, genes rank by the number of batches that select them and then by their median rank, the order of Seurat's `SelectIntegrationFeatures` and of Scanpy's `flavor="seurat_v3_paper"`; Scanpy's `flavor="seurat_v3"` ranks by median rank first.
- **Marker statistics and order.** `ds.markers.search` reports each gene's AUC and two-sided Mann-Whitney p-value, with Benjamini-Hochberg adjustment within each group. `ds.markers.load` keeps genes detected in at least 20% of a group's cells, with an AUC of at least 0.6 and an adjusted p-value of at most 0.05, and orders them by `score`, the gene's mean rank in the group divided by the sum of its mean ranks over all groups, which favors genes specific to one group. `sort_by="auc"` or `sort_by="p_value"` gives an order closer to that of `rank_genes_groups`, which sorts by its test statistic. Both tools test cells, not donors; the [donor-level comparisons tutorial](tutorials/condition_comparisons.ipynb) compares conditions across donors.

## Move data through H5AD

`cytearc.inspect_h5ad` reads the layout of an H5AD file, and `H5adReader.from_inspect` with `H5adToZarr` writes the file to a store. Each part of an `AnnData` object has a counterpart in the store:

| AnnData | Import | Export |
|---|---|---|
| `X`, `raw/X`, `layers` | The raw counts: inspection ranks the matrices that hold counts first, preferring `raw/X` and `layers/counts` to `X`, and `matrix_key` picks another. A file that holds only normalized values imports too; `H5adToZarr(..., values=...)` names their kind when CyteArc cannot tell it from the values | `X` holds raw counts; a run export with `matrix="normed"` holds the normalized values that the run's PCA read |
| `obs`, `var` | The cell and feature tables; categorical columns keep their categories, order and `uns["<column>_colors"]` | The cell and feature tables |
| Label columns of `obs` | `cluster_keys` imports them as clusterings instead of table columns | `obs={"leiden": leiden}` writes a clustering as a categorical column |
| `obsm` | `embedding_roles` imports `"umap"` and `"tsne"` layouts, and latent spaces such as `X_scVI` as `"coordinates"` that a neighbor graph can be built on | `obsm={"X_pca": pca, "X_umap": umap}`; `X_pca` also writes the PCA loadings to `varm["PCs"]` |
| `obsp`, `uns["neighbors"]` | Not imported; build the graph in CyteArc | `graph=` writes `obsp["connectivities"]`, `obsp["distances"]` and `uns["neighbors"]` in Scanpy's layout |

A completed run exports in one call: `cytearc.to_h5ad(ds.RNA, "analysis.h5ad", run=run)` writes the raw counts of the run's cells with its PCA, UMAP and cluster labels, and the colors of its labels in `uns`. After `adata = sc.read_h5ad("analysis.h5ad")`, `sc.pl.umap(adata, color="clusters")` runs without changes. Because `X` holds raw counts, run `sc.pp.normalize_total` and `sc.pp.log1p` before `sc.tl.rank_genes_groups(adata, "clusters")`, which expects log-normalized values. Without a run, pass the artifacts to export with the `cell_selection` that they cover, because each artifact must hold every exported cell. `cytearc.to_h5ad` streams the file block by block and needs neither the `anndata` package nor memory for the whole matrix, and `ds.to_anndata` returns the same content as an `AnnData` object. [](tutorials/import_and_export.ipynb) converts a pancreas H5AD file with its UMAP and cell types and exports it again, and the [export reference](/api/import_export.html#artifacts-in-exports) lists every exported field.

## Import a latent space from another tool

Import a latent space computed outside CyteArc, such as an scVI embedding, as coordinates that a neighbor graph, a UMAP and clusters can build on. `cells` is the cell selection whose cells the latent space describes, such as `run["analysis_cell_selection"]` for the cells of an exported run:

```python
import pandas as pd

# Import the latent space for the cells it describes, with a record of the tool that made it.
latent = ds.reduction.import_coordinates(
    pd.DataFrame(adata.obsm["X_scVI"], index=adata.obs_names),
    cell_selection=cells,
    name="X_scVI",
    source={"tool": "scvi-tools", "version": "1.2.0"},
)

# Build a neighbor graph and Leiden clusters on the imported coordinates.
graph = ds.graph.connectivity(ds.graph.neighbors(ds.graph.ann_index(latent), k=14))
leiden = ds.clusters.leiden(graph)
```

The rows of the DataFrame are matched to `cells` by cell ID. The `source` record and a fingerprint of the values become part of the artifact's {term}`provenance`, so lineage reports show where the coordinates came from. [](tutorials/import_and_export.ipynb) imports a latent space computed with scikit-learn and compares its clusters with those of CyteArc's PCA.
