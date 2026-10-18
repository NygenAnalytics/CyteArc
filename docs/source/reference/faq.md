---
description: Find answers to common CyteArc questions about installation, integration, clustering, data formats, and existing stores.
---

(faq)=
# FAQ

## Who maintains CyteArc?

CyteArc is open source and maintained by [Nygen](https://nygen.io).
Bug reports and feature requests for the Python package belong on [GitHub issues](https://github.com/NygenAnalytics/CyteArc/issues).

## How does CyteArc compare to Scanpy?

See [](../scanpy.md) for a stage-by-stage workflow map and H5AD exchange notes.

## How does CyteArc compare to Seurat?

See [](../seurat.md) for a stage-by-stage workflow map and RDS import. The measured differences
between CyteArc and Seurat WNN are summarized below.

## How do I run Harmony batch correction in CyteArc?

After PCA, call `corrected = ds.reduction.harmony(pca, ["batch_column"])`, then pass `corrected`
explicitly to `embeddings.initialization` and `graph.ann_index`. Continue by passing the
returned refs to `graph.neighbors` and `graph.connectivity`.
See {ref}`Harmony batch correction <harmony_batch_correction>` and the {ref}`dataset integration guide <integration_guide>`.

## How does CyteArc's Harmony2 differ from its earlier Harmony?

CyteArc implements [Harmony2](https://doi.org/10.64898/2026.03.16.711825), which builds on the
original [Harmony](https://doi.org/10.1038/s41592-019-0619-0) algorithm. CyteArc wrote this
implementation from the two papers. Clusters start from k-means++ seeds, the diversity penalty
takes a stabilized form that stays bounded when a batch is missing from a cluster, the ridge
penalty of each batch level in each cluster scales with the level's expected mass in the cluster,
and levels with almost no cells in a cluster are left out of its regression.

Three `harmony_params` keys are renamed and one is removed. Each raises a `ValueError` that names
the change, such as `Unsupported Harmony parameter 'nclust': it was renamed to 'n_clusters'`:

- `nclust` is now `n_clusters`;
- `max_iter_kmeans` is now `max_iter_cluster`;
- `lamb` is now `ridge_penalty`, a fixed penalty that must be positive;
- `cluster_fn` is removed, because clusters are seeded with k-means++.

`max_iter_harmony` must now be at least 1, and it defaults to 40: a fit stops as soon as it
converges, so the limit only bounds fits that settle slowly. Rename the keys in saved pipeline configurations too: a
pipeline checks them before it creates a run record.

By default the ridge penalty is dynamic: `ridge_alpha` (0.2) times each level's expected mass in
the cluster, its share of all cells times the cluster's total assignment. Where a level has about
its expected mass in a cluster, the correction removes about `1 / (1 + ridge_alpha)`, roughly 83%,
of that level's shift, while a fixed penalty of 1, the 2019 default, removes nearly all of it in
large clusters. A level with far less than its expected mass gets almost no correction. Pass
`harmony_params={"ridge_penalty": 1.0}` for the fixed penalty; the other Harmony2 changes stay.

Corrections saved by the earlier implementation stay listable and loadable, but they are not
reused: they lack the per-cluster penalty arrays that reuse requires, so the next request computes a
new correction and everything built on it again. See the
[Integration API reference](/api/integration.html) for every key and the stored arrays.

## Which integration method should I choose?

- Separate scRNA-seq batches: start with [](../tutorials/dataset_merging.ipynb), then compare
  uncorrected and Harmony embeddings in [](../tutorials/batch_correction.ipynb).
- Multiple assays in the same cells (CITE-seq): use the default WNN integration in the
  {ref}`recommended workflow <multimodal_integration>`; compare it with explicit SNN in
  [](../tutorials/multimodal_diagnostics.ipynb) when the choice matters.
- Map onto an existing reference: [](../tutorials/mapping_and_label_transfer.ipynb).

CyteArc does not include Scanorama, BBKNN, scVI, ComBat, or other external integration packages.
Export with `to_anndata` or `to_h5ad` when you need those tools.

## How do I reduce log noise or disable progress?

Configure the two settings independently:

```python
cytearc.configure_output(level="WARNING", progress=False)
```

For a batch log file, use `set_verbosity(..., filepath=...)`.
Enable timestamps with `configure_output(timestamps=True)`.
See [Utilities API reference](/api/utilities.html) for all output settings and their defaults.
Tutorial pages show completed snapshots from their cached execution; live notebooks animate the same operations.

## What is the difference between SNN and WNN integration?

Both use `integration.modalities` with two or more explicit source refs.
WNN is the default. It consumes modality-specific neighbour artifacts, learns one weight per
assay and cell, and ranks candidates by the resulting blended affinity.
SNN requires `method="snn"`; it consumes modality-specific connectivity maps and combines shared
edge support.

### CyteArc WNN versus Seurat

CyteArc WNN follows the weighting equations from Hao et al. but does not reproduce Seurat's default search exactly.
It considers the union of the existing KNN rows, uses the distance span from each assay's nearest to its `k`-th nonself neighbour as that assay's bandwidth, and L2-normalizes rows only during scoring.
For each cell and ordered pair of modalities, it compares within-modality and cross-modality prediction affinity.
It then sums the exponentiated directed scores for each target modality and normalizes those grouped strengths across all modalities.

Seurat normally searches a wider `knn.range=200` pool and uses SNN-far bandwidth.
CyteArc instead reuses the candidates in the supplied neighbour artifacts.
With two modalities, avoiding the wider search at ten million cells saves two additional index builds, 20 million queries, and 4 billion materialized candidate records.

Both differences are measured rather than assumed.
Given the same candidate pool and bandwidth, CyteArc reproduces Seurat 5.5.1 to the float32 resolution of the stored graph in both two-modality and synthetic three-modality fixtures.
Against Seurat's shipped defaults on a two-modality CITE-seq subset, CyteArc selects 89 percent of the same neighbours and its per-cell RNA weight correlates at 0.76.
The reference values ship with the test suite.

### Scaling notes

The per-cell prediction work grows quadratically with the number of modalities, while scoring blended graph edges is linear in the union candidate pool.
Stored neighbour rows, output edges, and modality weights remain linear in cell count.
See {ref}`WNN integration <multimodal_integration>` for deviations and trade-offs.

## How do I compute LISI in CyteArc?

Use `integration.compute_ilisi` for a single scIB-scaled batch-mixing score and `integration.compute_clisi` for a scIB-scaled biological-label conservation score.
Each requires the exact neighbour artifact and reads matching metadata rows through its stored cell
selection.
Use `cytearc.metrics.compute_lisi` for raw per-cell LISI values from KNN arrays.
See {ref}`LISI metrics <lisi_metrics>`.

## Should I use tSNE or UMAP?

tSNE and UMAP are complementary visualization tools.
tSNE emphasizes local structure and can reveal fine-grained diversity.
UMAP preserves more global structure, which helps when cluster relationships matter.
CyteArc computes tSNE with the optional `sgtsnepi` package (see {ref}`Optional t-SNE <installation_tsne>`), which runs on one thread, so time a subset before running it on an atlas-scale dataset.
UMAP is part of every installation and can run in parallel.
In CyteArc, UMAP and tSNE both require an explicit graph and initialization. Pass the same refs when
you want the layouts to share those inputs.

## What is densMAP?

Enable density-preserving UMAP with
`embeddings.umap(graph_ref, initialization_ref, use_density_map=True)`.
Useful when preserving local density structure matters alongside cluster separation.

## Which clustering should we use, Paris or Leiden?

Leiden is faster than Paris, especially for large datasets.
On small datasets we have tested, Leiden results are often more concordant with UMAP clusters.
Paris provides a hierarchy that can show relationships between clusters.
Both methods have low computational requirements, so you can run both and inspect the Paris
hierarchy from its exact clustering artifact:

```python
leiden = ds.clusters.leiden(graph_ref)
paris = ds.clusters.paris(graph_ref)
ds.plots.cluster_tree(
    graph=graph_ref,
    clusters=paris,
)
```

Use silhouette scores or domain knowledge to compare the candidates. The RNA pipeline scores
Leiden resolutions automatically in graph space and exposes the selected ref as `run["clusters"]`.
That result is a reproducible baseline, not proof that the selected resolution is biologically
correct. Granular evaluation APIs such as `evaluate_cluster_separability` can inspect partitions
outside the pipeline.

## Why do Leiden clusters differ from an earlier CyteArc release?

`clusters.leiden` and the pipeline now use CyteArc's own Leiden implementation,
`backend="cytearc"`, by default; earlier releases used the igraph package. Both optimize the same
modularity objective, but their random choices differ, so their labels differ even for the same
seed, and the new backend numbers clusters by size, with cluster 1 the largest. Compare partitions
with `clusters.compute_concordance`, which ignores cluster numbers, rather than by number. The
native backend gives the same labels for any number of threads, so `nthreads` does not change
results. See {ref}`Leiden clustering <leiden_method>` for the method.

The backend is part of each clustering's identity, so a default call does not reuse a clustering
that an earlier release stored. Pass `backend="igraph"` to reuse it, which needs no extra package,
or to compute a new igraph clustering, which needs the igraph package installed separately
({ref}`Optional GPL Leiden backends <installation_leiden_gpl>`).

## Why does the first Leiden clustering take longer than later ones?

The first call of the default Leiden backend in a fresh environment compiles its Numba kernels
once, which took about 26 seconds on a recent laptop. Later processes load the compiled kernels
from Numba's on-disk cache, about 0.4 seconds for their first call, and later calls in the same
process add only a few milliseconds of overhead. The cache needs a writable directory
(`NUMBA_CACHE_DIR` sets one).

## Why will my existing RNA Zarr store not open?

Current CyteArc versions write RNA counts twice: cell-major `counts` and a gene-major `countsT` copy.
Opening an RNA assay fails if that copy is missing, incomplete, still Zarr v2, or does not match `counts`.
Every assay must also carry the current preparation record: finalized raw counts, summaries
recomputed from those counts, and a dataset identity. Stores missing those records or
containing the unsupported `{assay}/state` group fail on open. There is no silent rewrite,
migration, or state-based result selection.

Re-import the source, or rebuild it into a fresh location:

```bash
python -m cytearc.tools.repack_zarr input.zarr output.zarr --data-only
```

The rebuild keeps raw counts, annotations, and selection columns, omits saved analyses and retired
state, and prepares the new store from its own counts. Afterward, recompute HVG, normalization,
PCA, graph, and marker results with the current release.
A grouped assay is stored as counts, so the rebuild copies its values without recomputing them.
Build grouped assays again with `add_grouped_assay`, under a new `assay_label` in a rebuilt store.
Count matrices must hold finite values, so a store whose grouped assay holds NaN cannot be rebuilt;
re-import its source and build the grouped assay there.
Non-RNA assays do not use `countsT`.
See [](../concepts/memory_and_execution.md).

## How do I create a count matrix for my single-cell data?

Generating count matrices is the primary step of single-cell data analysis.
For scRNA-Seq you can use tools like [STARsolo], [alevin-fry] or [kallisto|bustools].
If your data was generated using 10x's commercial solution then you can use [Cell Ranger].
For single-cell ATAC-Seq data, [Cell Ranger ATAC] can be used if your data was generated using 10x's kit.
[Yan et al] reviews ATAC-seq analysis approaches; [Cusanovich et al] describes an scATAC-seq experimental approach.

[STARsolo]: https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md
[alevin-fry]: https://alevin-fry.readthedocs.io/en/stable/
[kallisto|bustools]: https://www.kallistobus.tools/
[Cell Ranger]: https://www.10xgenomics.com/support/software/cell-ranger/latest
[Yan et al]: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1929-3
[Cusanovich et al]: https://www.cell.com/cell/fulltext/S0092-8674(18)30855-9
[Cell Ranger ATAC]: https://www.10xgenomics.com/support/software/cell-ranger-atac/latest

## Can I use CyteArc from R?

Not yet.
Please open a discussion on GitHub if an R API would be useful.

## What Python version does CyteArc require?

Python 3.13 or newer (`requires-python >=3.13`).
See {ref}`installation <installation>`.
