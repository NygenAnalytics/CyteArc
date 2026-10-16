# CyteArc

<p align="left">
  <a href="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml"><img src="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/NygenAnalytics/CyteArc"><img src="https://codecov.io/gh/NygenAnalytics/CyteArc/graph/badge.svg" alt="Coverage"></a>
  <a href="https://docs.nygen.io/CyteArc/"><img src="https://img.shields.io/github/actions/workflow/status/NygenAnalytics/CyteArc/pages.yml?label=Documentation" alt="Documentation"></a>
  <a href="https://pypi.org/project/cytearc"><img src="https://img.shields.io/pypi/v/cytearc.svg?color=4c72b0" alt="PyPI"></a>
  <a href="https://pypi.org/project/cytearc"><img src="https://img.shields.io/badge/python-3.13%20%7C%203.14%20%7C%203.15-4c72b0.svg" alt="Python 3.13, 3.14, and 3.15"></a>
</p>


CyteArc is a **fast Python framework for single-cell RNA, ATAC, protein, and multi-omic analysis that scales to 100M+ cells**. In benchmarks, it processed **one million cells within 15 minutes**, from cell filtering through UMAP and clustering to marker gene identification. On **Tahoe-100M**, the same workflow ran in **under 20 hours without any subsampling**.

[CyteBase](https://docs.nygen.io/CyteBase/) is a **ready-to-connect repository of CyteArc datastores**, containing approximately **200 million cells and 1.1 TB of data** on [Hugging Face](https://huggingface.co/buckets/Nygen/cytebase). CyteArc opens them by URI. Mount remote datasets for reanalysis and build or reuse references for mapping new cells, without downloading the complete count matrix. Counts stay remote while you save your own results in a store you control.

CyteArc's **built-in provenance** connects saved results to the exact cells, features, parameters, and upstream computations that produced them, so you can reuse completed work and trace each analysis.

| Problem | How CyteArc solves it | What you get |
| :-- | :-- | :-- |
| Your **dataset is larger than RAM** | Out-of-core algorithms, and neighbour search streams from cell-major and gene-major layouts, inside a memory budget you set | Analyse the selected cells with [explicit memory controls](https://docs.nygen.io/CyteArc/concepts/memory-and-execution) |
| The **data is stored remotely** and requires downloading | Fetches only the chunks an operation touches, and writes results to a store you own | Start analysing immediately, with one authoritative copy |
| A **single parameter change costs hours** of computation | Each step is fingerprinted by its settings and inputs, so reuse is by content, not by layer name | Only what changed recomputes, and the old version stays for comparison |
| Sub-population analysis leaves **scattered copies that nobody can trace back** | Subsets are masks in one file, and every result carries the cells and parameters behind it | A year later, a result still explains itself |

## Install

Python 3.13+.

```bash
uv venv --python 3.13
uv pip install --python .venv "cytearc[extra]"
```

Detailed installation instructions [here](https://docs.nygen.io/CyteArc/installation)

## Quick start

```python
import cytearc

ds = cytearc.DataStore(
    "s3://bucket/cells.zarr",  # also gs://, hf://, or a local path
)
run = ds.pipeline.run()  # durable QC → graph → UMAP → clustering → marker run

ds.plots.embedding(
    run=run,
    layout="umap",
    color_by="clusters",
)
```

Read the [scRNA-seq tutorial](https://docs.nygen.io/CyteArc/tutorials/scrna-seq) for a granular workflow, or [remote stores](https://docs.nygen.io/CyteArc/tutorials/remote-stores) for cloud setups.

## CyteBase

CyteBase provides a searchable catalog of public datasets stored as cloud-hosted CyteArc DataStores. Search by study, tissue, disease, or organism, then explore published cell annotations and available embeddings. Public access needs no credentials. CyteArc opens each dataset by its Hugging Face URI:

```python
import cytearc

# A CyteBase dataset in the October 2026 release.
uri = (
    "hf://buckets/Nygen/cytebase-2610/datasets/"
    "mcglade_2021_scrna_data_human_fallopian_tube_30f5e171/data.zarr"
)

# Open it for read-only exploration; counts stay remote.
ds = cytearc.DataStore(uri, zarr_mode="r", storage_options={"token": False}, min_features_per_cell=-1)

# Create a writable local store for your own analysis, keeping counts remote.
analysis = cytearc.mount_datastore(
    uri, at="fallopian_tube.zarr", storage_options={"token": False}, min_features_per_cell=-1
)
```

Use `analysis` with the pipeline shown above to save new results locally. Computation runs in your Python environment, reading count blocks over the network as needed.

To search the catalog for dataset URIs, install the separate [`cytebase`](https://github.com/NygenAnalytics/CyteBase) package (`pip install cytebase`) and see the [CyteBase documentation](https://docs.nygen.io/CyteBase/). [CyteBase datasets](https://docs.nygen.io/CyteArc/tutorials/cytebase-datasets) explains dataset URIs and releases.

## Documentation

Read workflow vignettes and API references in the **[documentation](https://docs.nygen.io/CyteArc/)**.

## CyteArc's capabilities

| Area | Methods |
| :-- | :-- |
| Modalities | scRNA-seq, scATAC-seq, CITE-seq, matched multi-omics |
| Core workflow | Quality control, feature selection, normalization, PCA and LSI, KNN graph, UMAP, densMAP, t-SNE (optional `tsne` extra), Leiden, Paris, marker search |
| Integration | Harmony, partial PCA, shared and weighted nearest neighbours, integration metrics |
| Mapping | Symphony-style reference mapping, label transfer, projection diagnostics |
| Trajectory | Population Balance Analysis pseudotime, expression dynamics and modules, multi-sink fate probabilities |
| Also included | Cell-cycle scoring, gene-set activity, graph-diffusion imputation, doublet scores, HTO demultiplexing, pseudobulk export |

## Support

[GitHub issues](https://github.com/NygenAnalytics/CyteArc/issues)

CyteArc is open source software released under the [BSD 3-Clause License](LICENSE) and maintained by [Nygen](https://nygen.io).
