---
description: Understand how CyteArc records the operation, parameters and inputs of each saved result, and how that record drives reuse, branching and lineage.
---

(provenance)=

# Why CyteArc records provenance

Every analysis result that CyteArc saves records its {term}`provenance`: the operation that produced it, the operation's parameters and the exact inputs it read. CyteArc uses that record to return a saved result instead of computing it again, to keep analyses with different settings apart in one store, and to trace any result back through each step to the counts.

## How analyses branch in one store

A single-cell result is a chain of choices. A marker table depends on a clustering, the clustering on a neighbor graph, and the graph on the exact cells, genes, normalization and reduction it was built from. When several analyses share one store, a file name or a cluster column cannot say which chain produced a result.

CyteArc therefore saves each analysis result, such as a cell selection, a PCA, a neighbor graph or a clustering, as an immutable {term}`artifact`. A method returns an {term}`ArtifactRef`, a reference to the saved result, and the next method takes that reference as its input, so every result names its inputs exactly. Two analyses that differ in one setting share every artifact upstream of that setting and branch from there:

```{mermaid}
flowchart LR
    counts["Counts"]
    selection["Cell and feature selections"]
    norm["Normalization"]
    pca30["PCA, 30 dimensions"]
    pca40["PCA, 40 dimensions"]
    graph30["Neighbor graph"]
    graph40["Neighbor graph"]
    clusters["Clusters"]
    markers["Marker table"]
    counts --> selection --> norm
    norm --> pca30 --> graph30 --> clusters --> markers
    norm --> pca40 --> graph40
```

Both branches read the same counts, selections and normalization. The two PCAs are different artifacts, so each branch builds its own graph, and neither branch becomes the store's current result.

## When a saved result is reused

A call whose operation, parameters and inputs match a saved artifact returns that artifact instead of computing it again. A changed parameter gives a new artifact, and each later step that reads it computes a new result too. The [analysis reuse tutorial](../tutorials/reuse_and_tracing.ipynb) changes the neighbor count and the number of principal components and traces which results each change computes again.

When a CyteArc release changes what an operation computes for the same parameters and inputs, the operation gets a new revision. Results saved under an earlier revision are then computed again instead of reused; they stay loadable, and lineage reports mark them stale. [](../developers/operation_revisions.md) explains when a change needs a revision.

## How inputs that can change are tracked

Cell columns and another store's labels can change after a result reads them. A filter fingerprints the cell columns it reads, so once a column's values change, the same filter call computes a new selection instead of returning the old one. Label transfer instead copies the reference labels it reads into the query store, so the transferred labels keep pointing at the annotations, threshold and projection that produced them after the reference atlas is re-annotated.

Provenance covers what CyteArc computes. A column that your own code writes with `ds.cells.insert`, as the [custom analyses tutorial](../tutorials/custom_analyses.ipynb) does, carries no record of how its values were computed. Coordinates imported with `ds.reduction.import_coordinates` keep the tool, version and parameters passed as `source`, as [](../tutorials/import_and_export.ipynb) shows.

## What a pipeline run adds

A [`PipelineRun`](/api/pipeline.html#cytearc.PipelineRun) records one complete recipe call, such as `ds.pipeline.rna()`: its settings, its named outputs, and `run.cells` and `run.features`, tables of the cells and genes it analyzed that read the run's own selections and results, never the live columns. A run reuses every saved result whose operation, parameters and inputs match, as a single call does, and a run given an earlier run as `cells`, such as an ADT run after an RNA run, reuses that run's cell selection. The run also records each stage, one step of the recipe such as `pca` or `umap`, with its wall and CPU time, sampled memory use, warnings and summary; these records never decide whether a result is reused.

The [artifacts, lineage, and summaries API reference](/api/artifacts.html) documents the calls that list, inspect and trace saved results.
