---
description: Methods implemented by CyteArc, how they are validated, and their scientific limitations.
---

(methods_and_limitations)=

# Methods and limitations

## Supported workflows

CyteArc provides complete workflows for:

- cell calling from unfiltered droplet matrices, ambient RNA correction, and doublet detection within samples
- scRNA-seq quality control, feature selection, normalization, graph construction, embedding, clustering, and marker discovery
- scATAC-seq peak filtering, LSI-based graph construction, clustering, and gene-score analysis
- CITE-seq RNA and ADT processing with weighted-nearest-neighbour integration by default and
  explicit shared-nearest-neighbour integration as an alternative
- pseudotime ordering, expression dynamics, modules, and multi-sink fate probabilities
- dataset merging, Harmony or partial-PCA correction with quantitative diagnostics, reference mapping, and label transfer
- joint analysis of local or remote datasets through mounts and virtual datasets
- cell-cycle scoring, gene-set activity, imputation, and pseudobulk export
- comparisons between conditions using biological samples, including paired donor designs
- transcription-factor regulons from RNA (SCENIC), motif or binding-peak activity from ATAC
  (chromVAR), and co-regulated gene programs

## Implemented methods

CyteArc implements a computational stage itself when an external implementation would require materializing the full matrix or would detach the result from its inputs.
It uses established libraries where they fit the streaming and provenance model.
The aim is one coherent execution path, not reimplementation for its own sake.

- **Cell calling:** [EmptyDrops](https://doi.org/10.1186/s13059-019-1662-y) separates cells from
  empty droplets using a Good-Turing ambient profile, a Dirichlet-multinomial model, Monte Carlo
  p-values, and Benjamini-Hochberg FDR. The RNA pipeline can call cells separately for each
  capture before applying QC. See [](../tutorials/cell_calling.ipynb).
- **Reduction and graph construction:** streamed normalization, exact covariance (Gram-matrix) PCA with a randomized block Krylov solver for more than 4,096 features ({ref}`details <pca_method>`), randomized streaming {term}`LSI`, approximate nearest-neighbour search, UMAP, {term}`densMAP`, and graph-based t-SNE through the optional `tsne` extra (see {ref}`Optional t-SNE <installation_tsne>`).
- **Feature selection:** highly variable genes by CyteArc's mean-variance trend of normalized
  values, by the variance-stabilizing transformation of [Seurat v3](https://doi.org/10.1016/j.cell.2019.05.031)
  on raw counts, or by the variance of [analytic Pearson residuals](https://doi.org/10.1186/s13059-021-02451-7),
  with batch-aware selection in the order of Seurat's `SelectIntegrationFeatures`, and Pearson
  residual normalization.
- **Batch correction and mapping:** CyteArc implementations of [Harmony2](https://doi.org/10.64898/2026.03.16.711825), which builds on the original [Harmony](https://doi.org/10.1038/s41592-019-0619-0) algorithm, and of [Symphony-style](https://doi.org/10.1038/s41467-021-25957-x) fixed-reference mapping, with label transfer and mapping diagnostics.
- **Matched multi-omics:** {term}`SNN integration` and [Hao-inspired WNN](https://doi.org/10.1016/j.cell.2021.04.048) integration for two or more assays, with WNN reporting one per-cell weight per modality.
- **Clustering:** a native, multi-threaded implementation of [Leiden](https://doi.org/10.1038/s41598-019-41695-z) community detection ({ref}`details <leiden_method>`) and a native implementation of [Paris hierarchical clustering](https://doi.org/10.48550/arXiv.1806.01664) with fixed and branch-adaptive cuts.
- **Trajectory analysis:** memory-aware [Population Balance Analysis](https://doi.org/10.1073/pnas.1714723115), supervised terminal-state fate probabilities, pseudotime aggregation and modules, and [MAGIC-style graph diffusion](https://doi.org/10.1016/j.cell.2018.05.061) for imputation.
  Fate probabilities also weigh each cell in each lineage, as in [CellRank](https://doi.org/10.1038/s41592-021-01346-6), for lineage-resolved markers, profiles, and modules, and for lineage drivers: correlations with fate probability tested by Fisher's z-transform.
  Lineage curves are fate-weighted [principal curves](https://doi.org/10.1080/01621459.1989.10478797) with the pseudotime rank as their projection index, smoothed by [penalized B-splines](https://doi.org/10.1214/ss/1038425655), and a [Slingshot](https://doi.org/10.1186/s12864-018-4772-0) cluster tree proposes candidate sinks ({ref}`lineage_curves`).
- **Statistical comparisons:** [Mann-Whitney U](https://doi.org/10.1214/aoms/1177730491),
  Kruskal-Wallis with optional Dunn pairwise tests, and paired Wilcoxon signed-rank tests on
  sample summaries. Cell-level Welch t-tests and one-way ANOVA are also available.
  Multiple-testing options include [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
  Holm, and Bonferroni correction. Marker search reports AUC, expression fractions, fold changes,
  and Mann-Whitney statistics. See [](../tutorials/condition_comparisons.ipynb) for a paired
  donor analysis and [](../tutorials/pseudobulk_and_differential_expression.ipynb) for export.
- **Doublets and diagnostics:** [Scrublet](https://doi.org/10.1016/j.cels.2018.11.005)-style
  doublet scores and calls simulated within each sample, cluster separability, and integration
  metrics informed by the [scIB benchmark](https://doi.org/10.1038/s41592-021-01336-8).
- **Gene sets, protein, and sample tags:** [AUCell](https://doi.org/10.1038/nmeth.4463) and [WAGGR](https://doi.org/10.1093/bioadv/vbac016) activity scores, CITE-seq processing, and [cell-hashing](https://doi.org/10.1186/s13059-018-1603-1) demultiplexing.
- **Ambient RNA:** a CyteArc implementation of [SoupX](https://doi.org/10.1093/gigascience/giaa151)
  from its paper: the soup profile of empty droplets, contamination fractions from marker gene
  sets with cluster-level exclusion, and the removal of soup counts from each cell, rounded at
  random as in the paper with seeded [systematic sampling](https://doi.org/10.1214/aoms/1177729988)
  so that every gene loses its share of the soup on average ({ref}`ambient_rna`). The paper's
  automatic estimate, its hierarchical cell-level model, and its correction of cluster-aggregated
  counts are not implemented, nor is DecontX; corrected counts from CellBender import as a Cell
  Ranger HDF5 file.
- **Regulons and chromatin activity:** [GRNBoost2](https://doi.org/10.1093/bioinformatics/bty916)
  co-expression networks and [SCENIC](https://doi.org/10.1038/nmeth.4463) cisTarget pruning
  ([pySCENIC](https://doi.org/10.1038/s41596-020-0336-2)), [chromVAR](https://doi.org/10.1038/nmeth.4401)
  motif deviations with MOODS-equivalent motif matching, cistromes from motifs or CUT&RUN and
  ChIP-seq peaks, region-to-gene links from paired RNA and ATAC, regulon specificity and
  connection specificity, and co-expression gene programs.
- **Pooled CRISPR screens:** guide calling from the guide counts by a [Poisson-Gaussian mixture](https://doi.org/10.1016/j.cell.2022.05.013) fitted to each guide as in [crispat](https://doi.org/10.1093/bioinformatics/btae535) and pertpy, by the two-Gaussian mixture that Cell Ranger describes, by a fixed UMI floor, or by each cell's top guide as in [SCEPTRE](https://doi.org/10.1186/s13059-024-03254-2), with the mixtures fitted to each sample separately when asked; and guide calls imported from Cell Ranger's CRISPR Guide Capture analysis. Either way, an artifact labels each cell with its one called guide, `"Negative"` or `"Multiple"`, with each cell's top guide and guide UMIs read from the streamed guide counts. The mixtures are fitted to exact histograms of each guide's UMI counts, which one streamed pass counts, so the cost of a fit follows the number of distinct counts rather than the number of cells. Perturbation labels merge the guides of each gene: a cell is labelled by the gene that its called guides target, with or without control guides, `"control"` with control guides only, `"Negative"` without a call, and `"Multiple"`, or its genes joined by `"+"`, when its guides target several genes; each cell's full set of genes is kept for high-MOI analyses, and the control targets are named, never guessed. Target efficiency and E-distance effect sizes are not implemented yet.

(pca_method)=

## Principal component analysis

`reduction.pca` and the PCA stage of the `rna` and `adt` pipeline recipes stream the normalized
values of the selected cells in row blocks and never hold the whole matrix. With
`feat_scaling=True`, each feature is first standardized with its mean and population standard
deviation over the normalized cells. The solver depends only on the number of features, so the
result does not depend on the memory budget, and the block size changes it only by rounding.

- **Up to 4,096 features**, PCA is exact. One pass over the cells accumulates the float64
  covariance matrix with symmetric rank updates, and LAPACK's eigensolver returns its leading
  `dims` eigenvectors as the loadings. The covariance takes at most 128 MiB at this width, and the
  eigensolver's cost grows with the cube of the feature count: on two threads it took 0.35 s for
  2,000 features, 5.6 s for 4,096, and 38 s for 8,192.
- **Above 4,096 features**, a randomized block Krylov solver fits the components. It applies
  the covariance to a block of `dims + max(10, dims)` Gaussian vectors, seeded by a fixed seed,
  and keeps every product in its basis, following the randomized range finder of
  [Halko, Martinsson and Tropp](https://doi.org/10.1137/090771806) with the block Krylov space of
  [Musco and Musco](https://doi.org/10.48550/arXiv.1504.05477), which converges in far fewer
  passes than subspace iteration when the trailing variances are close. The Rayleigh-Ritz
  projection of the covariance onto the basis gives the components and their variances. Each
  pass reads the selected cells once; the solver stops when the leading `dims` components move
  less than 0.1 degrees between two passes, and warns if they still move after 24 passes. The
  artifact records `solver="randomized"`.
- **Data passes.** The Gram solver reads the normalized values once and the randomized solver
  once per pass. Writing the coordinates reads them once more. The scaling statistics come from
  the moments that `features.normalize` stores beside the values. On a remote store, the default
  `local_cache="auto"` stages the normalized values locally, so the passes read a local copy.

A PCA artifact stores the variance of each of its `dims` components and that variance's share
of the total variance of the fitted values, as `variance` and `variance_ratio`, and
`show_elbow_plot=True` plots the shares for a new or a reused fit. Artifacts stored before
release 1.3.0 have neither array; they are still reused, and their elbow plot is replaced by a
warning. Signs follow scikit-learn's `svd_flip` convention: the largest loading of each component
is positive.

On the pancreas example (3,696 cells, standardized log-normalized values), the randomized solver
took 11 to 14 passes for 5,000 or 17,750 features and 21 or 50 components. It stayed within 0.035
degrees of the exact subspace, and its variance shares within a relative 2e-7 of the exact ones.
The IncrementalPCA fit that CyteArc used before release 1.3.0 for wide or narrowly blocked
inputs put its trailing components up to 89 degrees from the exact subspace, with variance
shares off by 10 to 21%. Those fits recorded `incremental_block_rows`, and lineage reports mark
them as stale.

(neighbour_search)=

## Neighbour search

CyteArc uses HNSW approximate nearest-neighbour search. The defaults are `ann_m=24`
for index connectivity, `ann_efc=200` for the search depth during construction, and
`ann_ef=200` for the minimum query depth. Queries use at least twice the requested neighbour
count. These settings balance index size and search work against neighbour recall at atlas scale.

Across five approximately million-cell settings, including Tabula Sapiens, a lupus PBMC atlas,
and a clustered synthetic dataset, these defaults reached recall@11 and recall@30 of at least
0.995 in four settings. The 50-PC PBMC setting reached recall@30 of 0.988; its missed neighbours
were within 1% of the exact k-th neighbour distance. The earlier defaults reached recall@11 of
0.896 to 0.991 and left up to 0.92% of cells without a true neighbour, which motivated the
deeper searches. Recall is the fraction of exact nearest neighbours recovered.

See [](../tutorials/graph_construction.ipynb) for graph settings and diagnostics.

(leiden_method)=

## Leiden clustering

`clusters.leiden`, the Leiden stages of the pipeline recipes, gene programs, and
`cytearc.clustering.leiden_membership` partition a graph with the Leiden algorithm of
[Traag, Waltman and van Eck](https://doi.org/10.1038/s41598-019-41695-z). Every stored entry
`(i, j)` of the input matrix `W` is an undirected edge of weight `W[i, j]`, so the clustered graph
has the adjacency `A = W + W^T`. Explicitly stored zeros are ignored, and weights must be finite
and non-negative. Leiden maximizes modularity with the resolution `γ`, the `resolution` argument,
under the configuration null model: a larger resolution gives more, smaller clusters. Every
cluster is connected.

The default backend, `"cytearc"`, is CyteArc's own implementation of the published method. It
runs on up to `nthreads` threads, which `clusters.leiden` and the pipeline recipes set to the
datastore's worker budget, and it returns identical labels for every thread count. Labels run
from 1: cluster 1 has the most cells, and clusters of equal size are ordered by their smallest
cell index. The
clustering package's README (`cytearc/clustering/README.md` in the
[CyteArc repository](https://github.com/NygenAnalytics/CyteArc)) describes the implementation, how
it was written and verified, and how it compares with igraph and leidenalg.

`backend="igraph"` and `backend="leidenalg"` call the GPL-licensed igraph and leidenalg packages
instead, which CyteArc does not install
({ref}`Optional GPL Leiden backends <installation_leiden_gpl>`). They optimize the same objective
on one thread and ignore `nthreads`, and their labels differ from the native backend's even for
the same seed. They are kept to reproduce results computed with them; `"igraph"` was CyteArc's
default before `"cytearc"`. The backend is part of a clustering's artifact identity, so results of
different backends are never reused for one another.

(doublet_method)=

## Doublet detection

`qc.doublets` and the doublet stage of the `rna` pipeline recipe follow
[Scrublet](https://doi.org/10.1016/j.cels.2018.11.005) (Wolock, Lopez and Klein, 2019). Doublets
form only among the cells of one capture, so each sample, the cells that share a label of
`sample_column`, is simulated and scored on its own; without a sample column every cell is one
sample. A sample of `n` cells simulates `round(simulation_ratio * n)` doublets, at least one, by
adding the counts of two of its cells drawn at random, so cell types parent doublets as often as
they occur. With `heterotypic_fraction` above 0, that share of the doublets draws its second parent
from the cells outside its first parent's cluster. The doublets are normalized like the graph's
cells, their library size the sum of their parents', and projected onto the graph's PCA without
refitting it.

Each cell and simulated doublet is then scored on its `k = round(save_k * (1 + r))` nearest
neighbors among the sample's cells and simulated doublets, with `r` simulated doublets per cell, so
that about `save_k` of them are cells where doublets are as dense as cells. These neighbors come
from an hnswlib index of the sample, built with the settings of the graph's ANN index, on one thread
unless that index was parallel. With `d` simulated doublets among the neighbors, `q = (d + 1) /
(k + 2)` estimates their fraction, and `q / ((1 - q) r)` how much denser doublets are than singlets
around the cell. Bayes' rule with the prior `expected_doublet_rate` gives the score, the probability
that the cell is a doublet. `raw_scores` keeps `d / k`.

The scores of a sample's simulated doublets are bimodal: doublets of two cells of one type, which
look like a singlet, score low and the others high. The threshold is the minimum between the two
modes, the method of [Prewitt and Mendelsohn](https://doi.org/10.1111/j.1749-6632.1965.tb11715.x)
as described by [Glasbey](https://doi.org/10.1006/cgip.1993.1040): their histogram of 256
equal-width bins is smoothed with a running mean of three bins until it has two maxima, and the
threshold is the center of the lowest bin between them. When it never has two,
[Otsu's threshold](https://doi.org/10.1109/TSMC.1979.4310076) of the same histogram is used. A cell
whose score is above its sample's threshold, or above `threshold` when one is given, is a predicted
doublet. The fraction of simulated doublets above the threshold is the detectable doublet fraction,
and the fraction of cells called doublets divided by it is Scrublet's estimate of the sample's
doublet rate. Doublets of two cells of one type cannot be told from singlets this way.

## Validation

The comparisons below cover reference implementations, direct numerical checks, and biological
outcomes. Inputs and settings are matched unless the table says otherwise.

### Comparisons with other tools

| Method | Reference | Data | Main result |
| :-- | :-- | :-- | :-- |
| **Cell calling** | DropletUtils 1.30 `emptyDrops` | 10x PBMC 1k and 10k | Identical 1,045 calls on PBMC 1k; Jaccard 0.9975 on PBMC 10k with matching settings |
| **Ambient RNA correction** | SoupX 1.6.2 | 10x PBMC 1k and 10k | Identical ambient profiles; contamination fractions agree within $2 \times 10^{-8}$ and cell-level subtraction within $10^{-12}$ counts before rounding |
| **HTO demultiplexing** | Seurat 5.5.1 `HTODemux` | 1,000 cells, 10 hashtags | Identical calls with matched settings; singlet naming can differ in unbalanced pools, as described below |
| **Guide call import** | Cell Ranger 6.0.0 `protospacer_calls_per_cell.csv` | 10x 5k A549 cells transduced with a CRISPR pool, 5,867 cells, 93 guides | All 7,040 calls of 4,867 cells reproduced, and the 1,000 cells without a confident call labelled `"Negative"`; every called UMI count equals the guide count matrix |
| **Guide calling, Poisson-Gaussian** | pertpy `main` (commit 9b5c53b) `assign_mixture_model`; crispat 0.9.11 `ga_poisson_gauss` | The same A549 screen | All 7,500 of pertpy's calls, the same call set in all 5,867 cells, and equal thresholds on all 93 guides; every one of crispat's 7,392 calls with 10,000 optimiser steps, as described below |
| **Guide calling, two Gaussians** | Cell Ranger 6.0.0 | The same A549 screen | 7,035 of Cell Ranger's 7,040 calls, and the same call set in 5,862 of 5,867 cells; the smallest called count equals Cell Ranger's threshold on 90 of 93 guides |
| **Guide calling, fixed floor and top guide** | pertpy `main` (commit 9b5c53b) `assign_by_threshold` and `assign_to_max_guide`; crispat 0.9.11 `ga_ratio` | The same A549 screen | All 12,535 calls at a floor of 5 UMIs and all 5,529 top-guide calls at 5 UMIs; all 4,539 and 3,825 ratio calls at fractions 0.5 and 0.7 |
| **Highly variable genes** | Scanpy 1.12.4, scikit-misc 0.5.3 | 892 CITE-seq cells, 36,601 genes | Seurat v3 selection Jaccard 0.998; Pearson residuals select the same genes, including the matched two-batch comparison |
| **Doublet detection** | Scanpy 1.12.4 `pp.scrublet` | 3,696 pancreas cells with 300 mixed-type and 60 same-type simulated doublets | AUROC 0.939–0.945 across four seeds, versus 0.913–0.919 for Scrublet |
| **Marker statistics** | SciPy; presto, Scanpy, BPCells | Synthetic tests and three datasets of about one million cells | Statistics checked against SciPy; AUC differences from presto and Scanpy are at most $5.2 \times 10^{-6}$ on shared cells, genes, groups, and expression |
| **Group comparisons** | SciPy and direct rank calculations | Independent and paired sample designs | Mann-Whitney, Kruskal-Wallis, Wilcoxon, Welch, ANOVA, and Dunn comparisons checked against reference calculations |
| **WNN integration** | Seurat 5.5.1 | 300 CITE-seq cells; 48 synthetic cells with three modalities | Same neighbours and weights/affinities within $3 \times 10^{-8}$ with matched candidate neighbours; 89% neighbour overlap with Seurat defaults |
| **Symphony-style query correction** | R Symphony 0.1.3 `mapQuery` | Four-cell, two-gene fixture | Coordinates agree on a small fixture with effectively zero correction |
| **Cell-cycle scores** | Scanpy 1.12.4 | 3,696 pancreas cells; 3,948 PBMCs | Scores agree within $3.3 \times 10^{-16}$ with the same control genes |
| **AUCell and WAGGR** | decoupler 2.2.0 | 3,948 PBMCs, 16 gene sets | Maximum score differences of $3 \times 10^{-8}$ and $9.2 \times 10^{-7}$, respectively; identical after rounding to single precision |
| **Graph-diffusion imputation** | magic-impute 3.0.0 | 3,948 PBMCs, 200 genes | Values agree within $6 \times 10^{-6}$ on the same graph and diffusion time; median gene-wise correlation 0.95 with MAGIC defaults |
| **iLISI and cLISI** | scib-metrics 0.6.1 | Two PBMC datasets | Per-cell scores agree within $1.6 \times 10^{-14}$ on the same neighbours and perplexity |
| **Fate probabilities** | CellRank 2 | 3,696 pancreas cells; 1,110 LARRY progenitors | Correlations exceed 0.99999999 with matched graphs, pseudotime, terminal cells, and kernel settings; 0.985 on the pancreas with CellRank defaults |
| **Lineage drivers** | CellRank 2.3.3 | 1,234 pancreas branch cells, 2,000 genes, four lineages | Correlations, confidence intervals, and significance results agree within $10^{-12}$ when both tools test every variable gene |
| **GRNBoost2 networks** | pySCENIC 0.12.1 | 2,612 PBMC multiome cells, 13,308 genes, 1,201 factors | Importance correlation 0.998; Jaccard 0.96 for the top 1,000 to 100,000 links |
| **chromVAR deviations** | chromVAR 1.34.1 | 879 motifs over 98,316 peaks | Deviations and z-scores agree within $4 \times 10^{-12}$ with the same motif matches and background peaks |
| **Gene programs** | Hotspot 1.1.1 | PBMC multiome cells and highly variable genes | Six programs contain 89% of the 503 genes in Hotspot's ten modules; adjusted Rand index 0.45 on shared genes |

In the LARRY lineage-tracing data, predicted neutrophil fate distinguished the later majority
fate of 725 clones with AUROC 0.76 (95% confidence interval 0.73–0.80). This comparison used
1,110 day-2 progenitors and measured outcomes from their later descendants. CellRank gave the
same AUROC. This evaluates a biological prediction separately from agreement between tools.

### Numerical and graph checks

- **PCA and LSI:** the PCA comparisons are described in {ref}`pca_method`. Streaming LSI is
  checked against the leading subspace of a dense singular-value decomposition and across
  different block sizes.
- **Neighbour search:** recall is measured against exact nearest neighbours, as described in
  {ref}`neighbour_search`. Regression tests also compare the returned neighbours with exact
  searches on clustered synthetic data.
- **Harmony2:** ridge regression, batch pruning, cluster-assignment updates, and the clustering
  objective are checked against direct dense calculations of the published equations. A
  synthetic benchmark with two groups of batches that share no cell types checks that batch
  mixing improves while cell-type purity stays at its uncorrected level.
- **Leiden and Paris:** native Leiden is checked against brute-force calculations on small
  random graphs, connectedness of every cluster, and identical labels on one to four threads.
  Tests also check that merging two clusters cannot improve modularity. Against igraph,
  modularity agrees within about 0.001 on graphs from 808 to ten million cells. Paris hierarchies
  reproduce the scikit-network clades on tie-free graphs.
- **Population Balance Analysis:** on 3,696 pancreas cells, the 30-mode solution has Spearman
  correlation 0.99999 with the full dense potential and a maximum difference of $9.1 \times 10^{-4}$
  on a unit scale. This checks the numerical approximation to the same equations.
- **Fate and lineage analysis:** sparse fate solutions are checked against dense Dirichlet
  solutions. Lineage markers, drivers, and profiles are checked against dense recomputations
  from expression and fate probabilities. Lineage curves recover a known noisy curve, and
  cluster trees recover a branched cluster layout.
- **Cell calling and doublets:** tests check EmptyDrops likelihoods and simulations against
  direct calculations, and reproduce calls across thread counts. Doublet tests check that
  simulated parents come from the same capture and compare neighbour counts with exact searches.

### Interpreting the comparisons

**Cell calling and sample tags.** EmptyDrops comparisons use matched model precision and
retention thresholds. CyteArc estimates the precision and finds the barcode-rank knee by spline
curvature; DropletUtils 1.30 defaults to a multinomial model and traces the knee differently.
The HTO comparison uses k-means and a positive quantile of 0.99. CyteArc names a singlet by its
one positive tag, while Seurat names it by its highest CLR value; these agree in the comparison
pool but can differ in strongly unbalanced pools.

**Guide calls.** The guide-calling comparisons ran on the A549 screen with the settings of each reference: `fixed` with `min_umi=5` against pertpy's `assign_by_threshold(5)`, `top_guide` with `min_umi=5` and `fraction_threshold=0` against `assign_to_max_guide(5)`, and `top_guide` with `min_umi=1` and `fraction_threshold` 0.5 or 0.7 against crispat's `ga_ratio` at the same fraction. With its default of 500 optimiser steps, crispat stops short of convergence on 88 of the 91 guides it fits; with 10,000 steps, CyteArc's Poisson-Gaussian fit reproduces every one of crispat's 7,392 calls. For 2 guides, SRPK1-1 and TPR-2, crispat fits no model, while CyteArc and pertpy call them. With `method="gaussian"`, the smallest UMI count that CyteArc calls for a guide equals the threshold that Cell Ranger reports on 90 of 93 guides and is one UMI above it on the other 3. The comparison uses the smallest called count because Cell Ranger reports the smallest count it called, while CyteArc's `threshold` is the model's cutoff, which sits lower when no cell holds the counts in between. In its ratio method, crispat keeps every guide tied at a cell's largest count, while CyteArc calls the first of them in assay order; no tie reached the fractions compared here.

**Feature selection and scores.** The Seurat v3 trend agrees with scikit-misc's interpolated
LOESS within $10^{-12}$ on the log10 scale. Four genes tied at the selection boundary account
for the single-batch selection difference. In the two-batch comparison, eight selected genes
differ (Jaccard 0.996) because floating-point ties in batch means affect the fitted trend.
CyteArc clips each batch's Pearson residuals using that batch's cell count; the comparison sets
the clip explicitly to match Scanpy. With independently sampled cell-cycle control genes,
pancreas score correlations are 0.995 for S and 0.997 for G2/M.

**Ambient RNA.** CyteArc uses seeded systematic sampling to round removed counts. Summed across
cells, removal in bands of ambient-gene abundance stays within 2.5% of the unrounded removal on
PBMC 1k and within 0.4% on PBMC 10k. The implemented SoupX steps are listed above.

**Mapping.** The small Symphony fixture checks projection and correction mechanics; it does
not establish agreement on substantial batch correction. Label-transfer vote fractions measure
neighbour support, not a calibrated probability that a cell label is correct. The
[mapping tutorial](../tutorials/mapping_and_label_transfer.ipynb) covers confidence and
projection diagnostics.

**Regulons.** GRNBoost2 links are not bit-identical to pySCENIC's because its Docker image uses
an older scikit-learn tree implementation. For context, two pySCENIC seeds give importance
correlation 0.946 and link Jaccard 0.64 in the same comparison. cisTarget AUC, NES, recovery
envelopes, and leading edges reproduce all 297 pySCENIC regulons and 2,184 enriched motif rows
within $10^{-15}$ when supplied the recovery curves that pySCENIC 0.12.1 assigns. CyteArc uses
each motif's own curve, rather than another enriched motif's curve as pySCENIC sometimes does.
On pySCENIC's network, the regulon names agree but median target Jaccard is 0.25; CyteArc's
targets are more consistent across GRNBoost2 seeds (median Jaccard 0.33 versus 0.20–0.24).
AUCell divides recovery area by the maximum attainable for the regulon's size, as R AUCell and
decoupler do, rather than by the area of the whole ranking window as pySCENIC does.

**Chromatin activity.** With an even sequence background, motif matching recovers every
motifmatchr match (Jaccard 0.9998); the additional 2,510 matches are four-base motifs. CyteArc
uses proper log-odds background adjustment and separate forward- and reverse-strand thresholds.
Before the separate thresholds were introduced, end-to-end motif z-scores correlated at a
median of 0.96 with chromVAR and population means at 0.995; these end-to-end values have not
been measured again since. On the multiome sample, CUT&RUN and ChIP-seq peak activity is highest
in the expected populations for SPI1, LEF1, PAX5, EBF1, POU2F2, and POU2AF1, but not for BCL11B
binding measured in Jurkat cells. Against ChIP-Atlas binding near gene starts, factor-to-gene
links exceed random binding expectations by 8% for CyteArc regulons, 7% for pySCENIC regulons,
and 3% for the top GRNBoost2 targets.

**Gene programs.** The Hotspot comparison used the igraph Leiden backend. The native backend
also forms six programs in the saved regulon analysis, each within 11 genes of its earlier size.
Hotspot separates the T-cell, monocyte, and B-cell programs into finer modules. Independent
noise profiles form no programs, but smoothed cells can produce spurious ones: the membership
threshold screens chance groupings and is not a significance test of the discovered programs.

## Current boundaries

### Inputs and provenance

CyteArc starts from count matrices; it does not process FASTQ files or perform alignment.
Use tools such as Cell Ranger, STARsolo, or alevin-fry first.

Provenance covers supported store-backed operations.
It does not capture arbitrary Python calculations, the full software environment, hardware, or study-level experimental records, and it describes computational relationships rather than scientific validity.

### Methods outside CyteArc

CyteArc does not ship every method in the single-cell ecosystem.
It does not provide scVI, Scanorama, RNA velocity, peak calling, FRiP/TSS APIs, or a complete replicate-aware differential-expression framework.
Marker searches report cell-level Mann-Whitney statistics, AUC, and within-group multiple-testing correction, which are not a substitute for inference across biological replicates.
{term}`WNN integration` requires two or more cell-aligned modalities, and CyteArc does not align unpaired modalities.
Fate mapping needs user-supplied terminal states and does not infer them or use RNA velocity.
A lineage tree proposes candidate sinks from clusters, but it can chain similar terminal clusters, so it does not discover them either.
Imputation is intended for exploratory visualization rather than differential expression.
Export selected data to AnnData or another supported format when a downstream method is outside CyteArc.
