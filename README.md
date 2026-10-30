<h1><img src="https://docs.nygen.io/assets/brand/logo/cytearc-dark.png" alt="" width="96" align="right"> CyteArc</h1>

<p align="left">
  <a href="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml"><img src="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/NygenAnalytics/CyteArc"><img src="https://codecov.io/gh/NygenAnalytics/CyteArc/graph/badge.svg" alt="Coverage"></a>
  <a href="https://docs.nygen.io/CyteArc/"><img src="https://img.shields.io/github/actions/workflow/status/NygenAnalytics/CyteArc/pages.yml?label=Documentation" alt="Documentation"></a>
  <a href="https://pypi.org/project/cytearc"><img src="https://img.shields.io/pypi/v/cytearc.svg?color=4c72b0" alt="PyPI"></a>
</p>

CyteArc is a Python framework for **single-cell RNA, ATAC, protein, and matched multi-omic analysis**. Its Zarr layout and read engine are designed together for efficient access by cell or gene, locally or directly from remote storage.

CyteArc analysed **Tahoe-100M in under 20 hours on commodity hardware**, from cell filtering through UMAP, clustering, and marker gene identification, **without downsampling the dataset or using GPUs**. The same workflow processed one million cells within 15 minutes.

Mount remote datasets, combine studies into **virtual datasets**, and run pipelines with provenance stored alongside the results.

[Documentation](https://docs.nygen.io/CyteArc/) · [Quick start](https://docs.nygen.io/CyteArc/quickstart) · [Methods and validation](https://docs.nygen.io/CyteArc/reference/methods)

## Capabilities

| Analysis | What you can do |
| :-- | :-- |
| **[Data access](https://docs.nygen.io/CyteArc/tutorials/remote-stores)** | Mount local or remote stores; combine cells across studies into virtual datasets without copying their counts |
| **[Cell calling and QC](https://docs.nygen.io/CyteArc/tutorials/quality-control)** | EmptyDrops cell calling, ambient RNA correction, doublet detection within samples, cell-cycle scoring, and HTO demultiplexing |
| **[Cell populations](https://docs.nygen.io/CyteArc/tutorials/scrna-seq)** | RNA, ATAC, and protein pipelines; batch-aware feature selection, PCA/LSI, UMAP, densMAP, t-SNE, Leiden/Paris clustering, and marker tests |
| **[Integration](https://docs.nygen.io/CyteArc/tutorials/multimodal-diagnostics)** | Harmony2 batch correction, WNN/SNN integration of modalities measured in the same cells, and integration diagnostics |
| **[Reference mapping](https://docs.nygen.io/CyteArc/tutorials/mapping-and-label-transfer)** | Symphony-style mapping, label transfer, and diagnostics for projected cells |
| **[Condition comparisons](https://docs.nygen.io/CyteArc/tutorials/condition-comparisons)** | Sample-level and paired comparisons across donors, with pseudobulk export for downstream differential expression |
| **[Trajectories](https://docs.nygen.io/CyteArc/tutorials/lineage-dynamics)** | Population Balance Analysis pseudotime, fate probabilities, lineage-specific gene dynamics, and fate-weighted curves |
| **[Gene activity and regulation](https://docs.nygen.io/CyteArc/api/regulons.html)** | AUCell/WAGGR gene-set scores, SCENIC-style regulons, chromVAR motif activity, region-to-gene links, and gene programs |
| **[CRISPR screens](https://docs.nygen.io/CyteArc/tutorials/crispr-screens)** | Guide calling, perturbation labels, target knockdown and E-distances, Mixscape's knocked-out cells and escapers, calibrated association tests at low and high MOI, and perturbation clusters |
| **[Import and export](https://docs.nygen.io/CyteArc/tutorials/import-and-export)** | Cell Ranger, Matrix Market, AnnData, and Seurat import; export counts and selected analysis results to AnnData |

## Install

```bash
pip install "cytearc[extra]"
```

See [installation](https://docs.nygen.io/CyteArc/installation) for environment setup and platform requirements.

## Run a pipeline

```python
import cytearc

ds = cytearc.DataStore("cells.zarr")
run = ds.pipeline.rna(label="rna")
ds.plots.embedding(run=run, color_by="clusters")
```

Pipelines cover QC through marker discovery, with over 100 stage settings across RNA, ATAC, protein, and multimodal workflows. See the [RNA tutorial](https://docs.nygen.io/CyteArc/tutorials/scrna-seq) for details.

## Mount a remote dataset

Read datasets from Amazon S3, Google Cloud Storage, Hugging Face, or Cloudflare R2, and save your analysis locally.

```python
ds = cytearc.mount_datastore(
    "hf://buckets/Nygen/cytebase-2610/datasets/mcglade_2021_scrna_data_human_fallopian_tube_30f5e171/data.zarr",
    at="fallopian_tube.zarr",
    storage_options={"token": False},
)
run = ds.pipeline.rna(label="rna")
```

Find datasets in the companion [CyteBase catalog](https://docs.nygen.io/CyteBase/). See [remote access](https://docs.nygen.io/CyteArc/tutorials/remote-stores) for connection options.

## Analyse several studies as one virtual dataset

A virtual dataset combines cells from existing stores without copying their count matrices, saving the time and space of a physical merge. It works like a regular datastore.

```python
from cytearc.composite import create_composite

sources = {
    "study_a": cytearc.DataStore("study_a.zarr", zarr_mode="r"),
    "study_b": cytearc.DataStore("study_b.zarr", zarr_mode="r"),
}

joint = create_composite(sources, at="joint.zarr", features="intersection")
joint_run = joint.pipeline.rna(label="joint_rna")
```

See the [joint-analysis tutorial](https://docs.nygen.io/CyteArc/tutorials/composite-datasets) for details.

## Trace results and reuse work

CyteArc records pipeline provenance in the datastore, linking result artifacts to their exact inputs and parameters. You can trace results and check their status without rerunning the analysis or setting up a separate workflow system. Matching results are reused automatically.

```python
ds.artifacts.lineage(run["markers"])
```

See [provenance](https://docs.nygen.io/CyteArc/concepts/provenance) and [result reuse](https://docs.nygen.io/CyteArc/tutorials/reuse-and-tracing) for details.

---

CyteArc is open source under the [BSD 3-Clause License](LICENSE), maintained by [Nygen](https://nygen.io). [Third-party notices](THIRD_PARTY_NOTICES.md).
