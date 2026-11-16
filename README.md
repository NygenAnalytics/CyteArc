<h1><img src="https://docs.nygen.io/assets/brand/logo/cytearc-dark.png" alt="" width="96" align="right"> CyteArc</h1>

<p align="left">
  <a href="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml"><img src="https://github.com/NygenAnalytics/CyteArc/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/NygenAnalytics/CyteArc"><img src="https://codecov.io/gh/NygenAnalytics/CyteArc/graph/badge.svg" alt="Coverage"></a>
  <a href="https://docs.nygen.io/CyteArc/"><img src="https://img.shields.io/github/actions/workflow/status/NygenAnalytics/CyteArc/pages.yml?label=Documentation" alt="Documentation"></a>
  <a href="https://pypi.org/project/cytearc"><img src="https://img.shields.io/pypi/v/cytearc.svg?color=4c72b0" alt="PyPI"></a>
</p>

Analyze single-cell data where it is stored, on local disk or in cloud storage, without loading the count matrix into memory. CyteArc covers RNA, ATAC, protein, and matched multi-omic data, along with the CRISPR screens, lineage tracing, immune repertoires, and DNA variants measured in the same cells.

- **Atlas scale on CPUs.** Counts are stored by cell and by gene and streamed within a memory budget you set, so whole atlases are analyzed without GPUs or subsampling. On a single CPU node reading from cloud storage, CyteArc analyzed all 95.6 million cells of Tahoe-100M, from cell filtering through UMAP, clustering, and marker genes.
- **Remote data and cohorts in place.** Mount a published store and analyze it where it is, with your results saved to a store you control, or combine studies into a composite dataset without copying their counts.
- **Results that are reused, branched, and traced.** Each result is saved with the inputs and settings that produced it, so matching results are reused, a changed setting reruns only the stages after it, and a finished analysis serves as a reference for mapping new cells.

Start with the [quick start](https://docs.nygen.io/CyteArc/quickstart), browse the [documentation](https://docs.nygen.io/CyteArc/), or read the [methods and validation](https://docs.nygen.io/CyteArc/reference/methods).

## Capabilities

| Analysis | What you can do |
| :-- | :-- |
| **[Data access](https://docs.nygen.io/CyteArc/tutorials/remote-stores)** | Mount local or remote stores; combine cells across studies into composite datasets without copying their counts |
| **[Cell calling and QC](https://docs.nygen.io/CyteArc/tutorials/quality-control)** | EmptyDrops cell calling, ambient RNA correction, doublet detection within samples, cell-cycle scoring, and HTO demultiplexing |
| **[Cell populations](https://docs.nygen.io/CyteArc/tutorials/scrna-seq)** | RNA, ATAC, and protein pipelines; batch-aware feature selection, PCA/LSI, UMAP, densMAP, LocalMAP, t-SNE, Leiden/Paris clustering, and marker tests |
| **[Integration](https://docs.nygen.io/CyteArc/tutorials/multimodal-diagnostics)** | Harmony2 batch correction, WNN/SNN integration of modalities measured in the same cells, and integration diagnostics |
| **[Reference mapping](https://docs.nygen.io/CyteArc/tutorials/mapping-and-label-transfer)** | Symphony-style mapping onto a finished analysis, which becomes a reference without reading its counts; label transfer and diagnostics for projected cells |
| **[Donor comparisons and pseudobulk](https://docs.nygen.io/CyteArc/tutorials/condition-comparisons)** | Sample-level and paired comparisons across donors, with pseudobulk export for downstream differential expression |
| **[Trajectories and fate mapping](https://docs.nygen.io/CyteArc/tutorials/pseudotime)** | Population Balance Analysis pseudotime, fate probabilities directed by pseudotime or potency scores, lineage-specific gene dynamics, and fate-weighted curves |
| **[Lineage tracing and immune repertoires](https://docs.nygen.io/CyteArc/tutorials/lineage-tracing)** | Clones from lineage barcodes, lineage recorder alleles and phylogenies, and mitochondrial or targeted DNA variants; clonal expansion, diversity, fate coupling, and fate bias; TCR and BCR clonotypes, gene usage, and AIRR export |
| **[Gene activity and regulation](https://docs.nygen.io/CyteArc/tutorials/regulons)** | AUCell/WAGGR gene-set scores, SCENIC-style regulons, chromVAR motif activity, region-to-gene links, and gene programs |
| **[CRISPR screens](https://docs.nygen.io/CyteArc/tutorials/crispr-screens)** | Guide calling, perturbation labels, target knockdown and E-distances, Mixscape's knocked-out cells and escapers, calibrated association tests at low and high MOI, and perturbation clusters |
| **[Import and export](https://docs.nygen.io/CyteArc/tutorials/import-and-export)** | Cell Ranger, Matrix Market, CSV, AnnData, Seurat, and Mission Bio Tapestri import, of counts or normalized values; export counts and selected analysis results to AnnData |

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

Recipes for RNA, ATAC, protein, multimodal, and CRISPR screen data cover quality control through marker discovery. See the [analysis pipeline API reference](https://docs.nygen.io/CyteArc/api/pipeline.html) for every stage and its settings.

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

Find datasets in the companion [CyteBase catalog](https://docs.nygen.io/CyteBase/), which publishes primary-count datasets from CZ CELLxGENE Discover as CyteArc stores. See [remote access](https://docs.nygen.io/CyteArc/tutorials/remote-stores) for connection options.

## Analyze several studies as one composite dataset

A composite dataset combines cells from existing stores without copying their count matrices, saving the time and space of a physical merge. It works like a regular datastore.

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

## Reuse, branch, and trace results

A rerun that changes one setting reuses every saved stage before it, and `ds.artifacts.lineage` traces any result back to the counts it came from, with no separate workflow system.

```python
# Saved stages through UMAP are reused; only clustering and the stages that use it run again.
run_r2 = ds.pipeline.rerun(run, label="rna_r2", leiden={"partitions": [2.0]})
ds.artifacts.lineage(run_r2["markers"])
```

See [provenance](https://docs.nygen.io/CyteArc/concepts/provenance) and [result reuse](https://docs.nygen.io/CyteArc/tutorials/reuse-and-tracing) for details.

---

CyteArc is open source under the [BSD 3-Clause License](LICENSE), maintained by [Nygen](https://nygen.io). [Third-party notices](THIRD_PARTY_NOTICES.md).
