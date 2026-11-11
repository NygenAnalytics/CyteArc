---
title: Out-of-core single-cell analysis
description: Mount remote single-cell datasets, analyze large cell atlases, and reuse shared references while keeping your own results separate.
---

# CyteArc

**Analyze single-cell datasets on local disk or in object storage, without loading the count matrix into memory.**

CyteArc analyzed Tahoe-100M, a single-cell perturbation dataset of 100 million cells, in under 20 hours on commodity hardware, from cell filtering through UMAP, clustering and marker genes, without downsampling the data or using GPUs.

Open a published atlas, plot marker expression, and save your own annotations. Counts stay in object storage, and your results are saved to a store you control. CyteArc supports RNA, ATAC, and multimodal analysis on your computer, an HPC cluster, or cloud compute.

Start with [remote dataset mounting](tutorials/remote_stores.ipynb) or the [Quick start](quickstart.ipynb), and look up exact signatures in the [API reference](/api/index.html).

## Reanalyze, map, and explore a shared atlas

A published atlas can hold counts, prepared reference models, and saved analysis results. CyteArc lets you use each of these directly. Mounting creates a writable analysis store with its own metadata and results, while the count data stay in the shared source.

```{mermaid}
flowchart TB
    atlas["Shared atlas in object storage<br/>Counts, reference models, and saved results"]
    reanalyze["Reanalyze cells"]
    map["Map and label new cells"]
    explore["Explore and compare"]
    results["Your writable analysis store<br/>New results and annotations"]
    atlas -->|Stream count blocks| reanalyze
    atlas -->|Reuse a prepared reference| map
    atlas -->|Read saved results| explore
    reanalyze --> results
    map --> results
    explore -.->|Save new results| results
    style atlas fill:#f3f4f6,stroke:#6b7280,color:#172b3a
    style reanalyze fill:#eef9f0,stroke:#32834a,color:#172b3a
    style map fill:#eaf7fd,stroke:#379ac5,color:#172b3a
    style explore fill:#f3f4f6,stroke:#6b7280,color:#172b3a
    style results fill:#ffffff,stroke:#172b3a,color:#172b3a
```

- **Reanalyze shared data.** Choose cells and run a new analysis, from feature selection and PCA to graphs, clusters, and markers. Counts stream from the source; your results are written separately. See [remote dataset mounting](tutorials/remote_stores.ipynb).
- **Map and label new cells.** Reuse a prepared RNA reference's genes, PCA model, and neighbor index to place query cells and transfer labels. Keep the query analysis separate from the reference. See [](tutorials/mapping_and_label_transfer.ipynb).
- **Explore before recomputing.** Inspect saved embeddings and annotations, select populations, and read gene expression for the question at hand. Try the [1.1-million-cell Tabula Sapiens atlas](tutorials/remote_stores.ipynb), then [](tutorials/plotting.ipynb).

The available starting points depend on what the atlas contains: mapping needs a prepared reference, while exploration can start from published annotations and embeddings.

## Built for large cell atlases

CyteArc streams count matrices in bounded blocks and stores RNA counts in both cell-wise and gene-wise layouts so each operation can read in the direction it needs. Memory and worker settings let you fit the work to your machine. The count matrix does not have to fit in RAM; graphs, neighbor indexes, and other intermediate results still need memory and storage.

Completed steps are saved and reused when their inputs and settings match. Change a clustering choice or examine another analysis branch without rebuilding every upstream result. See [](tutorials/reuse_and_tracing.ipynb) for a worked example and [](concepts/provenance.md) for how results record their inputs.

[](concepts/memory_and_execution.md) explains resource settings and how to measure runtime and memory on your workload.

## Get your first result

Set up your environment with the {ref}`Installation <installation>` guide, then start from the section that matches your question.

- **Getting started:** [mount Tabula Sapiens](tutorials/remote_stores.ipynb), plot its saved UMAP and a marker gene, then save a cell selection locally; no credentials are needed. The {ref}`Quick start <quickstart>` opens a prepared analysis of 100,000 cells sampled from the same atlas, and its [RNA-seq workflow](quickstart.ipynb#analyze-your-own-rna-seq-data) runs the standard pipeline on your own counts.
- **scRNA-seq preprocessing and structure:** the [PBMC walkthrough](tutorials/scrna_seq.ipynb) names the blood cell populations of a prepared analysis from their marker genes. For your own data, [call cells](tutorials/cell_calling.ipynb) in a raw droplet matrix, [remove ambient RNA](tutorials/ambient_rna.ipynb), and [assign hashed cells to their samples](tutorials/hto_demultiplexing.ipynb) where needed, then review [quality control](tutorials/quality_control.ipynb) before interpreting the results.
- **Annotation, programs, and visualization:** [annotate cell types](tutorials/annotation.ipynb) from supporting and competing markers, score [gene sets](tutorials/gene_set_scoring.ipynb) and [cell-cycle phases](tutorials/cell_cycle.ipynb), and [build figures](tutorials/plotting.ipynb) with CyteArc's plotting API.
- **Chromatin and multimodal integration:** the [scATAC-seq tutorial](tutorials/scatac_seq.ipynb) reads chromatin accessibility through peaks and gene scores, and the [CITE-seq](tutorials/cite_seq.ipynb) and [TEA-seq](tutorials/tea_seq.ipynb) tutorials join RNA, protein, and ATAC measured in the same cells with WNN.
- **Integration, mapping, and donor comparisons:** [analyze several CyteBase datasets together](tutorials/composite_datasets.ipynb) without copying their counts, [map query cells onto a fixed reference](tutorials/mapping_and_label_transfer.ipynb) and transfer its labels, and [compare conditions across matched donors](tutorials/condition_comparisons.ipynb).
- **Trajectories and fate mapping:** [order pancreas cell states with pseudotime](tutorials/pseudotime.ipynb), then [map the candidate fates](tutorials/fate_mapping.ipynb) of each cell, directed by the pseudotime or by potency scores without source cells.
- **Lineage tracing and immune repertoires:** [score a fate map](tutorials/lineage_tracing.ipynb) against the fates that LARRY's barcoded clones took, [cut a mouse embryo's PEtracer trees](tutorials/lineage_recorders.ipynb) into clones, [call clones in a colorectal tumor](tutorials/mitochondrial_lineage_tracing.ipynb) from mitochondrial DNA variants, and [group T and B cells into clonotypes](tutorials/immune_repertoires.ipynb) to find the expanded T cells.
- **Perturbation screens:** the [pooled CRISPR screen tutorial](tutorials/crispr_screens.ipynb) calls guides, separates knocked-out cells from escapers, and tests what each perturbation changes.
- **Data management and interoperability:** [import counts](tutorials/import_and_export.ipynb) from 10x, Matrix Market, H5AD, CSV, or a [saved Seurat object](tutorials/seurat_import.ipynb), and export counts and results to AnnData. The [Scanpy guide](scanpy.md) and [Seurat guide](seurat.md) translate individual steps.

## Methods and help

[](reference/methods.md) describes the supported workflows, validation evidence, and scientific limitations. For help, check the [](reference/faq.md) and [](reference/glossary.md), [ask a question](https://github.com/NygenAnalytics/CyteArc/discussions), or [report a bug](https://github.com/NygenAnalytics/CyteArc/issues). See [](developers/contributing.md) to contribute code or documentation.

The [documentation index for LLMs](https://docs.nygen.io/CyteArc/llms.txt) lists guides and API references for assistants that read this site.
