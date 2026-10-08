---
description: Translate a Seurat workflow to CyteArc and import a saved Seurat object.
---

(seurat-users)=
# CyteArc for Seurat users

This guide maps familiar Seurat stages to CyteArc's store-backed API. Use the workflow map
below to find the equivalent task, or try the {ref}`Quick start <quickstart>` to inspect a prepared
result. For optional background on how CyteArc stores and reuses results, see
[](scanpy_and_seurat.md).

## Workflow map

The methods below are approximate counterparts rather than identical implementations. After
importing an `.rds` file, the CyteArc column describes analysis on the resulting Zarr store.

| Goal | Seurat | CyteArc |
|---|---|---|
| Hold the analysis | `SeuratObject` | `DataStore` |
| Load a saved project | `readRDS()` | `inspect_seurat`, `SeuratReader`, `SeuratToZarr`, then `DataStore` |
| Work with modalities | Assays such as `RNA` and `ADT` | Assays in the same Zarr store |
| Select and normalize features | `NormalizeData`, `FindVariableFeatures` | `ds.features.hvgs`, then `ds.features.normalize`, using exact refs |
| Scale, reduce, and find neighbours | `ScaleData`, `RunPCA`, `FindNeighbors` | `ds.reduction.pca` standardizes features by default, followed by CyteArc's neighbour-graph methods |
| Embed and cluster | `RunUMAP`, `FindClusters` | `ds.embeddings.umap`, `ds.clusters.leiden` |
| Correct batches with Harmony | Harmony integration after PCA | `ds.reduction.harmony` after PCA, followed by graph construction |
| Integrate modalities with WNN | `FindMultiModalNeighbors` | Build neighbours per assay, then call `ds.integration.modalities(...)`; WNN is the default |

CyteArc WNN follows the published weighting equations but uses the supplied neighbour candidates
rather than Seurat's wider default search. See [](tutorials/multimodal_diagnostics.ipynb) before
interpreting differences between integration methods. SNN remains available with `method="snn"`.

## Import a saved Seurat object

CyteArc imports a saved Seurat object from an `.rds` file. It reads the on-disk RDS document and
does not attach to a live R session. It does not read `.h5seurat`.

Inspect the RDS file, select importable assays and reductions, then write a Zarr store:

```python
import cytearc

inspection = cytearc.inspect_seurat("pbmc.rds")
with cytearc.SeuratReader(
    "pbmc.rds",
    assays=["RNA"],
    reductions=["pca"],
) as reader:
    imported = cytearc.SeuratToZarr(reader, zarr_loc="pbmc.zarr").dump()
ds = cytearc.DataStore("pbmc.zarr")
imported.activeIdentity, imported.reductionArtifacts["pca"]
```

The importer brings across supported count layers and literal cell metadata. It returns exact
artifact refs for `active.ident` and selected reductions. Neighbour graphs, images, commands, and
most tool slots stay behind. Graphs, clusterings, marker searches, and integrated analyses are
rebuilt in CyteArc rather than imported from the RDS object. CyteArc does not write `.rds` or
`.h5seurat`.

Typical next steps are `ds.pipeline.run()`, [](tutorials/scrna_seq.ipynb), or
[](tutorials/graph_construction.ipynb). For multimodal data, build a neighbour artifact per assay
before calling `integration.modalities`. When you only need raw matrices, original 10x HDF5 or Matrix
Market counts are still preferable to an RDS export.

To return to Seurat, write H5AD or Matrix Market from CyteArc and convert or import it with the tools
used by your R workflow. See [](tutorials/import_and_export.ipynb) for the full Seurat import contract
and the other format paths.
