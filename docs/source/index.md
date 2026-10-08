---
description: Mount remote single-cell datasets, analyze large cell atlases, and reuse shared references while keeping your own results separate.
---

# CyteArc

**High-performance single-cell analysis where your data live.**

Open a published atlas, plot marker expression, and save your own annotations. Counts stay in
object storage; your results live in a store you control. CyteArc supports RNA, ATAC, and
multimodal analysis on your computer, an HPC cluster, or cloud compute.

Benchmarked on datasets with **up to 100 million cells**.

[Try remote dataset mounting](tutorials/remote_stores.ipynb) · [Quick start](quickstart.ipynb) ·
[Explore the API](/api/index.html)

## One shared atlas, many analyses

A published atlas can hold counts, prepared reference models, and saved analysis results.
CyteArc lets you use each of these directly. Mounting creates a writable analysis store with
its own metadata and results, while the count data stay in the shared source.

```{mermaid}
flowchart TB
    atlas["Shared atlas in object storage<br/>Counts · Reference models · Saved results"]
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

- **Reanalyze shared data.** Choose cells and run a new analysis, from feature selection and PCA
  to graphs, clusters, and markers. Counts stream from the source; your results are written
  separately. See [remote dataset mounting](tutorials/remote_stores.ipynb).
- **Map and label new cells.** Reuse a prepared RNA reference's genes, PCA model, and neighbour
  index to place query cells and transfer labels. Keep the query analysis separate from the
  reference. See [](tutorials/mapping_and_label_transfer.ipynb).
- **Explore before recomputing.** Inspect saved embeddings and annotations, select populations,
  and read gene expression for the question at hand. Try the [1.1-million-cell Tabula Sapiens atlas](tutorials/remote_stores.ipynb), then
  [](tutorials/plotting.ipynb).

The available starting points depend on what the atlas contains: mapping needs a prepared
reference, while exploration can start from published annotations and embeddings.

## Built for large cell atlases

CyteArc streams count matrices in bounded blocks and stores RNA counts in both cell-wise and
gene-wise layouts so each operation can read in the direction it needs. Memory and worker
settings let you fit the work to your machine. Graphs, neighbour indexes, and other intermediate
results still need memory and storage; the count matrix does not have to fit in RAM.

Completed steps are saved and reused when their inputs and settings match. Change a clustering
choice or examine another analysis branch without rebuilding every upstream result.
See [](tutorials/reuse_and_tracing.ipynb) for a worked example and [](concepts/provenance.md) for how
results record their inputs.

[](concepts/memory_and_execution.md) explains resource settings and how to measure runtime and
memory on your workload.

## Get your first result

- **Start from remote data:** [mount Tabula Sapiens](tutorials/remote_stores.ipynb), plot its
  saved UMAP and a marker gene, then save a cell selection locally. No credentials are needed.
- **Try CyteArc:** the {ref}`Quick start <quickstart>` opens a prepared analysis of 100,000
  cells sampled from the same atlas. For a smaller example that works through the analysis
  step by step, follow the [PBMC walkthrough](tutorials/scrna_seq.ipynb).
- **Bring your own counts:** [](tutorials/import_and_export.ipynb) covers supported formats, and
  the [RNA-seq workflow](quickstart.ipynb#analyze-your-own-rna-seq-data) shows the standard
  pipeline. Review [](tutorials/quality_control.ipynb) before interpreting the results.
- **Use another assay:** follow [](tutorials/scatac_seq.ipynb), [](tutorials/cite_seq.ipynb), or
  [](tutorials/tea_seq.ipynb).

Start with {ref}`Installation <installation>` to set up your environment, then follow the
{ref}`Quick start <quickstart>`. For importing an existing analysis or translating individual
steps, see the [Scanpy guide](scanpy.md) or [Seurat guide](seurat.md).

## Methods and help

[](reference/methods.md) describes the supported workflows, validation evidence, and scientific
limitations. For help, check the [](reference/faq.md) and [](reference/glossary.md),
[ask a question](https://github.com/NygenAnalytics/CyteArc/discussions), or
[report a bug](https://github.com/NygenAnalytics/CyteArc/issues).
See [](developers/contributing.md) to contribute code or documentation.
