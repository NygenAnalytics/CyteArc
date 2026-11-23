---
description: Understand how CyteArc records the operation, parameters and inputs of each saved result, and how that record drives reuse, branching, lineage and analyses in several processes at once.
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

## Several processes on one store

Several processes, such as two notebooks or batch jobs, can analyze one store at once: one can score doublets while another clusters, or each can run a trajectory call. An analysis call, such as a clustering, an embedding or a trajectory call, writes only a new artifact, at a path named by an ID of its own, and marks it complete last; it changes no saved result and no cell or feature column. A pipeline run writes only its own run record and its artifacts. Calls and runs in different processes therefore do not overwrite each other, and each saves what it would save alone: CyteArc's tests compare calls made at once in separate processes with the same calls made one after another.

A call reuses a complete result whose operation, parameters and inputs match its own, whichever process saved it. Two processes that start the same call with the same inputs at the same moment both compute it, and the store keeps both results, as after a recompute with `invalidate_cache=True`. The two hold the same values unless the call cannot reproduce its result, as a UMAP layout with `parallel=True` or an ANN index with `ann_parallel=True` cannot on several threads; later calls reuse the newer one, and `ds.artifacts.find` reports the two as ambiguous. Compute a step that several branches share, such as the neighbor graph, before starting the branches, so that each branch reuses it. Two runs that request one label at once both run; the first to complete takes the label, and the other ends failed.

A call reads only the saved results that its arguments name, so calls split across processes save what the same calls save in one process, with the same provenance. A trajectory can therefore run each of its calls in a job of its own, started from the refs of the calls before it, which `ref.to_dict()` turns into JSON and `cytearc.ArtifactRef.from_dict` reads back. `ds.trajectory.plan()` tells beforehand which of those calls the store already holds. A session setting that a result records, such as an RNA assay's size factor `ds.RNA.sf` or its normalization `ds.RNA.normMethod`, must be the same in each process.

Some calls change what every process reads: run them only while no other process uses the store. Beside another process, they can lose that process's change, leave a column holding part of one write and part of another, or remove a result that the other process is still writing. This list comes from reading the code, not from a measurement of these calls beside other processes:

- editing the cell or feature tables, such as adding annotations with `ds.cells.insert`;
- adding or discarding an assay, with `add_grouped_assay`, `add_melded_assay`, `qc.correct_ambient` or `discard_interrupted_assay`;
- opening the store with a `default_assay` or `assay_types` other than those it records, which rewrites them;
- `ds.artifacts.prune_incomplete(dry_run=False)`, which removes the artifacts that other processes are still writing;
- `ds.pipeline.abandon_label_claim`, which ends a run that may still be running.

The first writable open of a fresh import prepares the store, and other processes that open it meanwhile are refused until it has finished.

Labeled runs, pauses, resumes and `ds.pipeline.end_paused` need a store that creates a file only where none exists, in one step: a local folder, or an `s3://`, `gs://` or `az://` store. On other stores, such as `hf://` and fsspec stores, they are refused before anything is written, and a process may miss a result that another process saved moments before, which it then computes again. Each process plans its memory and threads alone, within its own `mem_budget` and `nthreads`, so give each process its share of the machine ([Memory budget](memory_and_execution.md#memory-budget)). Processes can share one `local_cache` folder for the normalized values of a remote store ([Local scratch for reductions](../reference/remote_storage.md#local-scratch-for-reductions)).

## How inputs that can change are tracked

Cell columns and another store's labels can change after a result reads them. A filter fingerprints the cell columns it reads, so once a column's values change, the same filter call computes a new selection instead of returning the old one. Label transfer instead copies the reference labels it reads into the query store, so the transferred labels keep pointing at the annotations, threshold and projection that produced them after the reference atlas is re-annotated.

Provenance covers what CyteArc computes. A column that your own code writes with `ds.cells.insert`, as the [custom analyses tutorial](../tutorials/custom_analyses.ipynb) does, carries no record of how its values were computed. Coordinates imported with `ds.reduction.import_coordinates` keep the tool, version and parameters passed as `source`, as [](../tutorials/import_and_export.ipynb) shows.

## What a pipeline run adds

A [`PipelineRun`](/api/pipeline.html#cytearc.PipelineRun) records one complete recipe call, such as `ds.pipeline.rna()`: its settings, its named outputs, and `run.cells` and `run.features`, tables of the cells and genes it analyzed that read the run's own selections and results, never the live columns. A run reuses every saved result whose operation, parameters and inputs match, as a single call does, and a run given an earlier run as `cells`, such as an ADT run after an RNA run, reuses that run's cell selection. The run also records each stage, one step of the recipe such as `pca` or `umap`, with its wall and CPU time, sampled memory use, warnings and summary; these records never decide whether a result is reused.

The [artifacts, lineage, and summaries API reference](/api/artifacts.html) documents the calls that list, inspect and trace saved results.
