# CyteArc

<p align="left">
  <a href="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml"><img src="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/NygenAnalytics/CyteArc"><img src="https://codecov.io/gh/NygenAnalytics/CyteArc/graph/badge.svg" alt="Coverage"></a>
  <a href="https://docs.nygen.io/CyteArc/"><img src="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pages.yml/badge.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/cytearc"><img src="https://img.shields.io/pypi/v/cytearc.svg?color=4c72b0" alt="PyPI"></a>
  <a href="https://pypi.org/project/cytearc"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-4c72b0.svg" alt="Python 3.12, 3.13, and 3.14"></a>
  <a href="https://pepy.tech/projects/cytearc"><img src="https://static.pepy.tech/personalized-badge/cytearc?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="Downloads"></a>
</p>


CyteArc is a **fast Python framework for single-cell RNA, ATAC, protein, and multi-omic analysis that scales to 100M+ cells**. In benchmarks, it processed **one million cells within 15 minutes**, from cell filtering through UMAP and clustering to marker gene identification. On **Tahoe-100M**, the same workflow ran in **under 20 hours without any subsampling**.

[CyteBase](https://docs.nygen.io/CyteArc/tutorials/cytebase) is a **ready-to-connect repository of CyteArc datastores**, containing approximately **200 million cells and 1.1 TB of data** on [Hugging Face](https://huggingface.co/buckets/Nygen/cytebase). Mount remote datasets for reanalysis and build or reuse references for mapping new cells, without downloading the complete count matrix. Counts stay remote while you save your own results in a store you control.

CyteArc's **built-in provenance** connects saved results to the exact cells, features, parameters, and upstream computations that produced them, so you can reuse completed work and trace each analysis.

| Problem | How CyteArc solves it | What you get |
| :-- | :-- | :-- |
| Your **dataset is larger than RAM** | Out-of-core algorithms, and neighbour search streams from cell-major and gene-major layouts, inside a memory budget you set | Analyse the selected cells with [explicit memory controls](https://docs.nygen.io/CyteArc/concepts/memory-and-execution) |
| The **data is stored remotely** and requires downloading | Fetches only the chunks an operation touches, and writes results to a store you own | Start analysing immediately, with one authoritative copy |
| A **single parameter change costs hours** of computation | Each step is fingerprinted by its settings and inputs, so reuse is by content, not by layer name | Only what changed recomputes, and the old version stays for comparison |
| Sub-population analysis leaves **scattered copies that nobody can trace back** | Subsets are masks in one file, and every result carries the cells and parameters behind it | A year later, a result still explains itself |

## Install

Python 3.12+.

```bash
uv venv --python 3.12
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

CyteBase provides a searchable catalog of public datasets stored as cloud-hosted CyteArc DataStores. Search by study, tissue, disease, or organism, then explore published cell annotations and available embeddings. Public access needs no credentials.

Install the CyteBase extra alongside the analysis extras:

```bash
uv pip install --python .venv --prerelease allow "cytearc[cytebase,extra]"
```

```python
from cytearc import cytebase

catalog = cytebase.Catalog()
matches = catalog.search("kidney")
print(matches)

# Open one of the matching datasets for read-only exploration.
dataset_id = matches[0]["cytebase_id"]
ds = catalog.open_datastore(dataset_id)

# Create a writable local store for your own analysis, keeping counts remote.
analysis = catalog.mount_datastore(dataset_id, at="kidney_analysis.zarr")
```

Use `analysis` with the pipeline shown above to save new results locally. Computation runs in your Python environment, reading count blocks over the network as needed.

Follow the [CyteBase tutorial](https://docs.nygen.io/CyteArc/tutorials/cytebase) for catalog filters, plotting, and example notebooks, or see the [CyteBase API reference](https://docs.nygen.io/CyteArc/api/cytebase.html).

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
