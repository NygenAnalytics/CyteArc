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
- differential abundance of cell categories and kNN neighbourhoods between conditions, with samples as replicates

## Implemented methods

CyteArc implements a computational stage itself when an external implementation would require materializing the full matrix or would detach the result from its inputs.
It uses established libraries where they fit the streaming and provenance model.
The aim is one coherent execution path, not reimplementation for its own sake.

- **Reduction and graph construction:** streamed normalization, covariance (Gram-matrix) PCA with an incremental fallback, randomized streaming {term}`LSI`, approximate nearest-neighbour search, UMAP, {term}`densMAP`, and graph-based t-SNE through the optional `tsne` extra (see {ref}`Optional t-SNE <installation_tsne>`).
- **Batch correction and mapping:** CyteArc implementations of [Harmony2](https://doi.org/10.64898/2026.03.16.711825), which builds on the original [Harmony](https://doi.org/10.1038/s41592-019-0619-0) algorithm, and of [Symphony-style](https://doi.org/10.1038/s41467-021-25957-x) fixed-reference mapping, with label transfer and mapping diagnostics.
- **Matched multi-omics:** {term}`SNN integration` and [Hao-inspired WNN](https://doi.org/10.1016/j.cell.2021.04.048) integration for two or more assays, with WNN reporting one per-cell weight per modality.
- **Clustering:** a native, multi-threaded implementation of [Leiden](https://doi.org/10.1038/s41598-019-41695-z) community detection ({ref}`details <leiden_method>`) and a native implementation of [Paris hierarchical clustering](https://doi.org/10.48550/arXiv.1806.01664) with fixed and branch-adaptive cuts.
- **Trajectory analysis:** memory-aware [Population Balance Analysis](https://doi.org/10.1073/pnas.1714723115), supervised terminal-state fate probabilities, pseudotime aggregation and modules, and [MAGIC-style graph diffusion](https://doi.org/10.1016/j.cell.2018.05.061) for imputation.
- **Statistics and diagnostics:** marker AUC, the [Mann-Whitney U test](https://doi.org/10.1214/aoms/1177730491) with [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) correction, doublet scores, cluster separability, and integration metrics informed by the [scIB benchmark](https://doi.org/10.1038/s41592-021-01336-8).
- **Differential abundance:** independent implementations of [propeller](https://doi.org/10.1093/bioinformatics/btac582) for category proportions, with the empirical Bayes moderated t- and F-tests of [Smyth (2004)](https://doi.org/10.2202/1544-6115.1027) and the robust and trended priors of [Phipson et al. (2016)](https://doi.org/10.1214/16-AOAS920), and of the [Milo](https://doi.org/10.1038/s41587-021-01033-z) neighbourhood test: refined neighbourhood sampling on a stored kNN graph, blockwise counting, the [edgeR](https://doi.org/10.1093/bioinformatics/btp616) quasi-likelihood test with TMM normalization (the legacy QL method that miloR uses is the default), and the weighted spatial FDR of [Lun et al. (2017)](https://doi.org/10.1038/nmeth.4295). Both were written from the published methods, and their agreement with speckle and miloR is checked by comparing outputs. The edgeR steps run in [edgePython](https://github.com/pachterlab/edgePython), a Python port of edgeR installed with the optional `milo` extra ({ref}`installation_milo`). Neither needs R at run time.
- **Gene sets, protein, and sample tags:** [AUCell](https://doi.org/10.1038/nmeth.4463) and [WAGGR](https://doi.org/10.1093/bioadv/vbac016) activity scores, CITE-seq processing, and [cell-hashing](https://doi.org/10.1186/s13059-018-1603-1) demultiplexing.
- **Regulons and chromatin activity:** [GRNBoost2](https://doi.org/10.1093/bioinformatics/bty916)
  co-expression networks and [SCENIC](https://doi.org/10.1038/nmeth.4463) cisTarget pruning
  ([pySCENIC](https://doi.org/10.1038/s41596-020-0336-2)), [chromVAR](https://doi.org/10.1038/nmeth.4401)
  motif deviations with MOODS-equivalent motif matching, cistromes from motifs or CUT&RUN and
  ChIP-seq peaks, region-to-gene links from paired RNA and ATAC, regulon specificity and
  connection specificity, and co-expression gene programs.

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

## Validation

Validation is method-specific.
Agreement measured for one method is not evidence of equivalence for the rest of the package, and none of these comparisons establish biological ground truth.

- HTO demultiplexing is checked against Seurat reference outputs.
- Symphony-style query correction is checked against a golden fixture generated with the R Symphony 0.1.3 `mapQuery` implementation, including a case with nonzero correction.
- Harmony2's ridge regression solves, batch pruning, stabilized cluster assignment update, and
  clustering objective are checked against direct dense calculations from the published equations.
  A synthetic benchmark in which two groups of batches share no cell types checks that batch mixing
  rises while cell-type purity stays at its uncorrected level.
  CyteArc's Harmony2 was written from the two papers, independently of other Harmony
  implementations, and is not expected to match them number for number.
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
- Propeller is checked against the outputs of speckle 1.10.0 (limma 3.66.0) on logit and arcsine transforms, robust and non-robust priors, the variance trend, a three-group ANOVA, and zero counts. Statistics, p-values, FDR, and the empirical Bayes prior agree to within 1.2e-13 relative. Separately, the empirical Bayes steps are compared with limma's `squeezeVar` and `eBayes` on constructed inputs: the moment estimator agrees to 5e-13, and the robust prior degrees of freedom to 2e-7, the precision of limma's own root finder. When a robust variance trend passes exactly through most of ten to twelve rows, limma's result depends on rounding error and CyteArc's can differ.
- Milo is checked stage by stage against miloR 2.6.0 on four simulated cases, and end to end through `DataStore.abundance` on two. On a live opt-in test with the Wilk et al. COVID-19 atlas, miloR tests CyteArc's own kNN graph and index cells ({ref}`how to run it <live_abundance_test>`). Sampling, membership, counts, k-distances, and annotation match miloR exactly. The test statistics, computed with edgePython at the commit that CyteArc's tests pin ({ref}`installation_milo`), match edgeR 4.8.2 to near its convergence tolerance: p-values and spatial FDR within 2e-6 relative, and logFC within 6e-4. CyteArc calls the same neighbourhoods as miloR at spatial FDR below 0.1 in every case, including all 931 on Wilk and no others. One deliberate difference: miloR weights spatial FDR over every neighbourhood even when `min.mean` leaves some untested, which misaligns its weights, and CyteArc uses the tested neighbourhoods only.

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

### Differential abundance

Milo neighbourhoods are built on a stored graph from an `ann_metric="l2"` index, matching miloR's Euclidean kNN; an approximate index can miss a few exact neighbours that miloR's own search would find.
edgePython 0.2.6, the latest release, departs from edgeR on Milo's path, so its test statistics only approximate miloR's; install edgePython from GitHub until its next release ({ref}`installation_milo`).
CyteArc does not implement Milo's graph-overlap spatial FDR, `refinement_scheme="graph"`, the GLMM model with random effects or kinship, neighbourhood expression testing, or pseudobulk differential expression.

