---
description: Translate a Seurat workflow to CyteArc and import a saved Seurat object.
---

(seurat-users)=
# CyteArc for Seurat users

This guide maps familiar Seurat stages to CyteArc's store-backed API. Use the workflow map
below to find the equivalent task, or try the {ref}`Quick start <quickstart>` to inspect a prepared
result. For optional background on how CyteArc stores and reuses results, see
[](scanpy_and_seurat.md). Follow [](tutorials/seurat_import.ipynb) for an executed import
with metadata selection, coordinate roles and partial-import notices.

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

An object imports in part when some of it cannot be imported:

- Without `assays` or `reductions`, the reader selects everything, and `SeuratToZarr` skips an
  assay or reduction that cannot be imported, with a `skipped_assay` or `skipped_reduction`
  notice. An assay or reduction that you name must import.
- A cell metadata column that cannot be decoded, such as a list column, is left out with a
  `skipped_metadata_column` notice. `cell_metadata=["a", "b"]` reads only those columns and
  raises for one that cannot be decoded; `cell_metadata={"exclude": ["a"]}` leaves columns out.
  `Date` and `POSIXct` columns import as ISO 8601 text and bit64 `integer64` columns as 64-bit
  integers.
- The `integrated` assay of `IntegrateData` holds normalized data but no raw counts, so it
  reports `counts_layer_missing`. When the active assay cannot be imported, pass
  `default_assay`, such as `SeuratToZarr(reader, zarr_loc="pbmc.zarr", default_assay="RNA")`.
  An active `SCTAssay` holds SCTransform-corrected counts, so the default assay becomes the
  assay of raw counts that its models were fitted to, usually `RNA`, with an
  `sct_default_assay` notice; `default_assay="SCT"` keeps it.
- A reduction keeps its coordinates without the assay it was computed on (`assay.used`); only
  its feature loadings need that assay. Such a reduction is stored under the default assay. A
  reduction of some of the cells, as in the Seurat v5 sketch workflow, is aligned to the cells
  by name and stored over a cell selection of its cells.
- A reduction becomes a UMAP or t-SNE embedding when it has two or three columns and its key
  names the method (`UMAP_`, `wnnUMAP_`, `tSNE_`) or it is named `umap`, `tsne`, or `t-sne`;
  other reductions import as coordinates for graph construction.
  `reduction_roles={"phate": "umap", "umap.layout": "coordinates"}` sets the role by name.

Inspection reports each reduction's `assayUsed` and `role`, along with input notices, before
anything is written. Review the writer's returned notices too: choices made during import,
including the SCT-to-RNA default-assay change, are reported there.

Typical next steps are `ds.pipeline.run()`, [](tutorials/scrna_seq.ipynb), or
[](tutorials/graph_construction.ipynb). For multimodal data, build a neighbour artifact per assay
before calling `integration.modalities`. When you only need raw matrices, original 10x HDF5 or Matrix
Market counts are still preferable to an RDS export.

To return to Seurat, write H5AD or Matrix Market from CyteArc and convert or import it with the tools
used by your R workflow. See [](tutorials/seurat_import.ipynb) for the Seurat import examples
and [](tutorials/import_and_export.ipynb) for H5AD and Matrix Market export.
