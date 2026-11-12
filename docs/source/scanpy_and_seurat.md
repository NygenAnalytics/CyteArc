---
description: Compare how CyteArc, Scanpy and Seurat hold counts and results, and decide which steps of an analysis to run in CyteArc.
---

(scanpy_and_seurat)=
# How CyteArc compares with Scanpy and Seurat

Compare how CyteArc keeps counts and results with how Scanpy and Seurat keep them, and decide which steps of an analysis to run in CyteArc. All three tools run the same steps: quality control, feature selection, normalization, reduction, neighbor graphs, embeddings, clustering and marker discovery. The three differ in where counts and results are kept and in how each step finds the results it builds on.

- [](scanpy.md) maps each Scanpy step to CyteArc and shows how to move data through H5AD files.
- [](seurat.md) maps each Seurat step to CyteArc and shows how to import a saved Seurat object.

## How the data model differs

| In Scanpy or Seurat | In CyteArc |
|---|---|
| An `AnnData` or `SeuratObject` in memory holds counts, metadata and results | A `DataStore` opens a Zarr store of counts, metadata and results on local disk or in object storage, and reads the counts in blocks of bounded size instead of loading them whole |
| A step changes the object in the current session | A step saves its result in the store as an immutable {term}`artifact`, which later sessions can inspect or reuse |
| Filtering subsets the object or creates a view | Filtering returns a cell selection, an artifact that marks the kept cells; no cell is deleted |
| Graphs and embeddings fill named slots, such as `obsp` and `obsm`, or `graphs` and `reductions` | Each graph and embedding is an artifact, and the step that writes it returns an {term}`ArtifactRef` that the next step takes as its input |
| One object holds one current set of results | Each {term}`PipelineRun` keeps its own results, and a new run leaves earlier runs unchanged |

Because each step names its inputs, analyses with different settings share one store. Given a cell selection `cells`, such as the one `ds.qc.filter` returns, `features = ds.features.hvgs(cells, top_n=2000)` returns a reference to a gene selection that `ds.features.normalize(cells, features)` takes, and a second selection with `top_n=1000` is saved beside the first without replacing it. [](concepts/provenance.md) explains what each result records and when a saved result is reused.

## Choose where each step runs

A common split runs quality control through clustering and marker search in CyteArc, where the counts stay on disk or in object storage, and exports the cells and results that a method in another tool needs. CyteArc does not include scVI, Scanorama or RNA velocity; an scVI or Scanorama embedding comes back as coordinates through `ds.reduction.import_coordinates`. For differential expression across donors, CyteArc compares donor-level summaries, as [](tutorials/condition_comparisons.ipynb) shows, and builds {ref}`pseudobulk counts <pseudobulk_and_differential_expression>` for replicate-aware models such as DESeq2 and edgeR.
