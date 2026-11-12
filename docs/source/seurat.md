---
description: Find the CyteArc call for each step of a Seurat analysis, the settings that give different results, and how to import a saved Seurat object without R.
---

(seurat-users)=
# CyteArc for Seurat users

Find the CyteArc call for each step of a Seurat analysis, the settings where the two give different results, and how to bring a saved Seurat object into CyteArc without an R session. Where Seurat keeps assays, metadata and results in a `SeuratObject`, CyteArc keeps them in a Zarr store that a `DataStore` opens, and saves each result as an {term}`artifact` that later steps take by reference. [](scanpy_and_seurat.md) compares the two models.

## Translate each step

In the calls below, `ds` is a `cytearc.DataStore`, the object that opens a store. `ds.pipeline.rna()`, `ds.pipeline.adt()` and `ds.pipeline.atac()` run each modality's standard workflow in one call and return a {term}`PipelineRun`, a saved record of the run's settings and results. The methods in the table run the same steps one at a time:

| Step | Seurat | CyteArc |
|---|---|---|
| Hold the analysis | `SeuratObject` | `DataStore` |
| Load a saved project | `readRDS()` | `inspect_seurat`, `SeuratReader` and `SeuratToZarr`, then `DataStore` |
| Hold several modalities | Assays such as `RNA` and `ADT` | Assays of the same store |
| Compute QC metrics | The `nCount_RNA` and `nFeature_RNA` columns, and `PercentageFeatureSet(obj, pattern = "^MT-")` | Opening a new store adds `RNA_nCounts`, `RNA_nFeatures`, `RNA_percentMito` and `RNA_percentRibo` to the cell table |
| Filter cells | `subset(obj, subset = nFeature_RNA > 200 & percent.mt < 5)` | `ds.qc.filter(["RNA_nFeatures", "RNA_percentMito"], lows=[200, None], highs=[None, 5])`, whose bounds exclude equal values as `>` and `<` do, or `ds.qc.auto_filter` with bounds from the median absolute deviation; both return a cell selection and delete no cells |
| Select and normalize genes | `NormalizeData`, `FindVariableFeatures` | `ds.features.hvgs`, then `ds.features.normalize` over the same cell selection |
| Scale and reduce | `ScaleData`, `RunPCA` | `ds.reduction.pca`, which standardizes each gene unless `feat_scaling=False` |
| Build the neighbor graph | `FindNeighbors` | `ds.graph.ann_index`, `ds.graph.neighbors`, then `ds.graph.connectivity` |
| Embed and cluster | `RunUMAP`, `FindClusters` | `ds.embeddings.umap`, `ds.clusters.leiden` |
| Find marker genes | `FindAllMarkers` | `ds.markers.search`, then `ds.markers.load` |
| Correct batches with Harmony | Harmony integration after PCA | `ds.reduction.harmony` after PCA, then the graph steps |
| Integrate modalities with weighted nearest neighbors (WNN) | `FindMultiModalNeighbors` | `ds.pipeline.multimodal([rna_run, adt_run])` on runs of the same cells, or `ds.integration.modalities` on a neighbor artifact of each assay |

[](tutorials/scrna_seq.ipynb) interprets each stage of an RNA analysis, and [](tutorials/cite_seq.ipynb) analyzes RNA and antibody counts of the same cells and integrates them with WNN.

## Which settings give different results

- **WNN candidates and bandwidth.** CyteArc's WNN follows the published weighting equations [@Hao_2021] but scores only the neighbors that each modality's graph already holds, and sets each modality's bandwidth from the distances to a cell's nearest and k-th neighbors. Seurat's `FindMultiModalNeighbors` searches a wider pool of candidates by default and computes its bandwidth differently. With the same candidates and bandwidth, CyteArc reproduces Seurat 5.5.1 to float32 precision; with Seurat's defaults on a CITE-seq subset, 89% of the neighbors that CyteArc selects match Seurat's, and the per-cell RNA weights of the two correlate at 0.76 ([FAQ](reference/faq.md#cytearc-wnn-versus-seurat)). `method="snn"` combines the modalities by shared neighbors instead, and [](tutorials/multimodal_diagnostics.ipynb) compares the two methods on a CITE-seq dataset.
- **Gene selection and normalization.** Seurat's default `FindVariableFeatures` method, `vst`, corresponds to `ds.features.hvgs` with `flavor="seurat_v3"`, which scores genes on raw counts by variance stabilization; CyteArc's default flavor scores the variance of library-size normalized values instead. CyteArc selects highly variable genes before it normalizes, and `ds.features.normalize` divides each cell's counts by its total over the selected genes, not over all genes, before taking the log.
- **Marker order.** `ds.markers.search` reports each gene's AUC and two-sided Mann-Whitney p-value, and `ds.markers.load` orders the genes of each group by `score`, which favors genes specific to that group; pass `sort_by="p_value"` to order them by p-value.

## Import a saved Seurat object

CyteArc reads a Seurat `.rds` file directly, without R. `inspect_seurat` reports what the file holds and which parts can be imported, `SeuratReader` selects the parts to read, and `SeuratToZarr` writes them to a store:

```python
import cytearc

# Read the RNA counts and both reductions, and write them to a new store.
with cytearc.SeuratReader("pbmc.rds", assays=["RNA"], reductions=["pca", "umap"]) as reader:
    imported = cytearc.SeuratToZarr(reader, zarr_loc="pbmc.zarr").dump()

# Keep every imported cell: by default, the first open of a new store
# leaves out cells with 10 or fewer detected genes.
ds = cytearc.DataStore("pbmc.zarr", min_features_per_cell=-1)
```

The store holds the `counts` layer of each imported assay and the cell metadata. `imported.activeIdentity` is a clustering artifact, so plots and `ds.markers.search` accept Seurat's identities as they accept a Leiden result; `imported.reductionArtifacts["umap"]` is a layout that plots draw, and a PCA imports as coordinates that a neighbor graph can be built on. `imported.notices` lists what stayed behind, such as the normalized `data` and `scale.data` layers and the command history; CyteArc computes normalized values from the raw counts when a step needs them.

The `counts` layer of an SCT assay holds SCTransform-corrected counts, not raw counts. When SCT is the active assay, the import therefore makes the assay whose raw counts the SCT models were fitted to, usually `RNA`, the store's default assay, and reports an `sct_default_assay` notice; `SeuratToZarr(..., default_assay="SCT")` keeps SCT as the default.

[](tutorials/seurat_import.ipynb) imports a PBMC object of monocytes and dendritic cells, reads the notices of a partial import, such as `skipped_metadata_column` and `sct_default_assay`, and shows which parts an import skips or must import and how to set a reduction's role; the [Seurat import reference](/api/import_export.html#seurat-import) lists every option. CyteArc does not read `.h5seurat` files or write Seurat files; to return to R, export H5AD or Matrix Market files as [](tutorials/import_and_export.ipynb) shows.
