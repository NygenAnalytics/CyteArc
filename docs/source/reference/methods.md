---
description: Methods implemented by CyteArc, how they are validated, and their scientific limitations.
---

(methods_and_limitations)=

# Methods and limitations

## Supported workflows

CyteArc provides complete workflows for:

- scRNA-seq quality control, feature selection, normalization, graph construction, embedding, clustering, and marker discovery
- scATAC-seq peak filtering, LSI-based graph construction, clustering, and gene-score analysis
- CITE-seq RNA and ADT processing with weighted-nearest-neighbour integration by default and
  explicit shared-nearest-neighbour integration as an alternative
- pseudotime ordering, expression dynamics, modules, and multi-sink fate probabilities
- dataset merging, Harmony or partial-PCA correction with quantitative diagnostics, reference mapping, and label transfer
- cell-cycle scoring, gene-set activity, imputation, and pseudobulk export
- transcription-factor regulons from RNA (SCENIC), motif or binding-peak activity from ATAC
  (chromVAR), and co-regulated gene programs

## Implemented methods

CyteArc implements a computational stage itself when an external implementation would require materializing the full matrix or would detach the result from its inputs.
It uses established libraries where they fit the streaming and provenance model.
The aim is one coherent execution path, not reimplementation for its own sake.

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
- **Statistics and diagnostics:** marker AUC, the [Mann-Whitney U test](https://doi.org/10.1214/aoms/1177730491) with [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) correction, [Scrublet](https://doi.org/10.1016/j.cels.2018.11.005) doublet scores and calls simulated within each sample, cluster separability, and integration metrics informed by the [scIB benchmark](https://doi.org/10.1038/s41592-021-01336-8).
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

(pca_method)=

## Principal component analysis

`reduction.pca` and the pipeline's PCA stage stream the normalized values of the selected cells
in row blocks and never hold the whole matrix. With `feat_scaling=True`, each feature is first
standardized with its mean and population standard deviation over the normalized cells. The
solver depends only on the number of features, so the result does not depend on the memory
budget, and the block size changes it only by rounding.

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

(leiden_method)=

## Leiden clustering

`clusters.leiden`, the pipeline's Leiden stages, gene programs, and
`cytearc.clustering.leiden_membership` partition a graph with the Leiden algorithm of
[Traag, Waltman and van Eck](https://doi.org/10.1038/s41598-019-41695-z). Every stored entry
`(i, j)` of the input matrix `W` is an undirected edge of weight `W[i, j]`, so the clustered graph
has the adjacency `A = W + W^T`. Explicitly stored zeros are ignored, and weights must be finite
and non-negative. Leiden maximizes modularity with the resolution `γ`, the `resolution` argument,
under the configuration null model: a larger resolution gives more, smaller clusters. Every
cluster is connected.

The default backend, `"cytearc"`, is CyteArc's own implementation of the published method. It
runs on up to `nthreads` threads, which `clusters.leiden` and the pipeline set to the datastore's
worker budget, and it returns identical labels for every thread count. Labels run from 1: cluster
1 has the most cells, and clusters of equal size are ordered by their smallest cell index. The
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

`qc.doublets` and the pipeline's doublet stage follow
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

Validation is method-specific.
Agreement measured for one method is not evidence of equivalence for the rest of the package, and none of these comparisons establish biological ground truth.

- HTO demultiplexing is checked against Seurat 5.5.1 `HTODemux` calls with k-means clustering and
  the default `positive.quantile` of 0.99, on a pool where every singlet's highest CLR value is its
  positive tag. CyteArc names a singlet by its one positive tag, while `HTODemux` names it by its
  highest CLR value, so the two can differ in strongly unbalanced pools.
- Doublet detection is compared with scanpy 1.12.4 `pp.scrublet` on the mouse pancreas sample
  of 3,696 cells with 300 doublets of two cell types and 60 of one type added, each the sum of two
  cells' counts. Over seeds 0 to 3, CyteArc ranks the doublets at AUROC 0.939 to 0.945 (0.971 to
  0.974 for those of two types) and Scrublet at 0.913 to 0.919 (0.953 to 0.958). CyteArc's
  automatic thresholds call 221 to 223 cells, 211 to 213 of the 300 doublets of two types and 9 or
  10 singlets, and estimate a doublet rate of 0.09, the rate of the added doublets; Scrublet's
  call 241 to 252 cells, 222 to 229 of them and 17 to 22 singlets. Scores correlate across seeds at Spearman 0.92
  to 0.95 for CyteArc and 0.95 for Scrublet; the two methods' scores at 0.64. Tests check scores
  and calls of doublets added to synthetic counts of one and of two captures, that no simulated
  doublet pairs cells of two samples, and the neighbor counts against exact nearest neighbors.
- Symphony-style query correction is checked against a golden fixture generated with the R Symphony 0.1.3 `mapQuery` implementation, including a case with nonzero correction.
- Ambient RNA removal is compared with SoupX 1.6.2 on the 10x Genomics PBMC 1k (v2) and 10k (v3)
  samples. With the same empty-droplet range the soup profiles are identical, the global,
  per-gene-set, and per-cluster contamination fractions agree to within 2e-8, the tolerance of
  SoupX's fit, and the same cells measure each gene set. Before rounding, the removal of each
  gene from each cell matches SoupX's cell-level subtraction to within 1e-12 counts. CyteArc then
  rounds each removal at random, as the paper does, with systematic sampling seeded by
  `random_seed`, so a seed reproduces the counts. Summed over the cells, the removal of the genes
  in each band of soup ranks, from the 10 most abundant soup genes to those beyond rank 5,000,
  stays within 2.5% of the removal before rounding on PBMC 1k and within 0.4% on PBMC 10k, and
  each gene's deviation from it has the binomial spread of its rounding.
- Harmony2's ridge regression solves, batch pruning, stabilized cluster assignment update, and
  clustering objective are checked against direct dense calculations from the published equations.
  A synthetic benchmark in which two groups of batches share no cell types checks that batch mixing
  rises while cell-type purity stays at its uncorrected level.
  CyteArc's Harmony2 was written from the two papers, independently of other Harmony
  implementations, and is not expected to match them number for number.
- Highly variable gene flavors are compared with scanpy 1.12.4 and scikit-misc 0.5.3 on the
  892 cells and 36,601 genes of the 1k PBMC CITE-seq test fixture, selecting 2,000 genes.
  The seurat_v3 trend matches scikit-misc's interpolated LOESS to within 1e-12 on the log10 scale,
  so the scores agree to the 10 digits the fixture keeps; the selections differ only in 4 genes tied
  at the last selected score (Jaccard 0.998). Pearson residual variances agree to the same digits and select
  the same genes, with and without two batches. With two batches, seurat_v3 scores differ from
  scanpy's `seurat_v3_paper` by up to 7e-4 and 8 selected genes differ (Jaccard 0.996), because
  scanpy's Welford batch means break ties between genes with equal means, which moves the
  interpolated trend. scanpy clips every batch's residuals at the square root of the first
  batch's cell count; CyteArc clips each batch at its own, so the batch comparison passes an
  explicit clip. The LOESS, the standardized variance, and the residuals are also checked against
  direct NumPy computations of their published formulas.
- Marker statistics are checked against SciPy, including continuity and large-tie corrections.
- Paris hierarchies are compared with the scikit-network reference implementation on tie-free graphs.
- The native Leiden backend is checked against brute-force calculations on small random graphs,
  and tests check that every cluster is connected, that no two clusters can be merged to increase
  modularity, and that labels are identical on 1 to 4 threads. A separate implementation written
  from the paper reached the same modularity within seed noise, and against the igraph backend
  the modularity agrees within about 0.001 on graphs from 808 to 10 million cells. The clustering
  package's README gives the details.
- WNN is checked against scalar reference calculations, a two-modality CITE-seq fixture, and a synthetic three-modality Seurat 5.5.1 fixture, plus modality-order symmetry, row-order invariance, and degenerate bandwidths.
  It follows the published weighting equations but is not bit-identical to Seurat defaults, as described in {ref}`the FAQ <faq>`.
- AUCell and WAGGR scores are checked against frozen decoupler 2.2 fixtures.
- Fate probabilities are checked against dense Dirichlet solutions on controlled graphs.
- SCENIC regulons are compared with pySCENIC 0.12.1 (its Docker image, scikit-learn 1.1.3) on a
  10x Genomics PBMC multiome sample of 2,612 cells, 13,308 genes, and 1,201 factors.
  With the same seed, GRNBoost2 importances correlate at Pearson 0.998 and the top 1,000 to
  100,000 links overlap at Jaccard 0.96; two pySCENIC seeds give 0.946 and 0.64. Links are not
  bit-identical because scikit-learn's tree code has changed since 1.1.3.
- cisTarget AUC, NES, recovery envelopes, and leading edges reproduce all 297 pySCENIC regulons
  and 2,184 enriched motif rows (within 1e-15) when each motif is given the recovery curve that
  pySCENIC 0.12.1 assigns it.
  pySCENIC 0.12.1 gives most annotated motifs another enriched motif's curve; `prune_modules`
  uses each motif's own, so on pySCENIC's network the regulon names agree but the median target
  Jaccard is 0.25. These targets are more reproducible across GRNBoost2 seeds (median Jaccard
  0.33, against 0.20 to 0.24 with pySCENIC's pruning).
- AUCell regulon scores differ from pySCENIC's by design. CyteArc divides each cell's
  recovery area by the largest area a regulon of that size can reach, as R AUCell and
  decoupler do, while pySCENIC divides by the area of the whole ranking window. For an
  unweighted 20-gene regulon among 1,000 genes, pySCENIC's scores are 0.79 times CyteArc's.
- chromVAR deviations are compared with chromVAR 1.34.1 and motifmatchr 1.34.0 for 879 JASPAR
  2024 motifs over 98,316 peaks. Given R's motif matches and background peaks, deviations and
  z-scores agree within 4e-12, with the same missing values. With an even background, motif
  matching finds every motifmatchr match (Jaccard 0.9998); all 2,510 additional matches belong
  to four-base motifs. With the default subject background CyteArc departs from motifmatchr
  twice: motifmatchr applies the background adjustment with the opposite sign to a log-odds
  score, and it reuses each motif's forward-strand threshold for the reverse strand, which misses
  the requested p-value when complementary bases differ in frequency. CyteArc scores proper
  log-odds and gives each strand its own threshold, as MOODS does. Before the per-strand
  thresholds, matches overlapped motifmatchr's at Jaccard 0.94, and end to end, with
  independently drawn background peaks, per-motif z-scores correlated at a median Pearson of
  0.96 and population means at 0.995; these have not been measured again since.
- On the multiome sample, chromVAR scores CUT&RUN and ChIP-seq peak sets highest in the
  expected population for SPI1 (dendritic cells), LEF1 (CD4 T cells), and the B-cell factors
  PAX5, EBF1, POU2F2, and POU2AF1, but not for BCL11B binding measured in Jurkat cells.
  Against ChIP-Atlas binding within 10 kb of a gene's start (the GRETA gold standard), the
  fraction of factor-to-gene links with binding exceeds its random expectation by 8% for CyteArc
  regulons, 7% for pySCENIC regulons, and 3% for each factor's top 50 GRNBoost2 targets.
- Gene programs are compared with Hotspot 1.1.1 modules on the same cells and highly variable
  genes. The six programs contain 89% of the 503 genes in Hotspot's ten modules, and 76% of those
  genes fall in their module's majority program (adjusted Rand index 0.45 on shared genes,
  against at most 0.008 for 100 permuted labelings). These figures were measured when gene
  programs used the igraph Leiden backend, the default of earlier releases; with the native
  default, the regulons tutorial forms the same six programs, each within 11 genes of its earlier
  size. Hotspot splits the T-cell, monocyte, and B-cell programs into finer modules. Independent
  noise profiles form no programs, but the membership threshold assumes independent samples and
  ignores that programs are chosen from the same data: it screens chance groupings rather than
  testing programs, and profiles of smoothed cells form spurious ones.

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
Imputation is intended for exploratory visualization rather than differential expression.
Export selected data to AnnData or another supported format when a downstream method is outside CyteArc.

