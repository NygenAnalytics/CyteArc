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
- pooled CRISPR screens: guide calling, perturbation labels, each perturbation's effect on its target gene and its E-distance from the control cells, Mixscape's knocked-out cells and escapers, calibrated association tests at low and high MOI, and clusters of perturbation profiles

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
- **Pooled CRISPR screens:** guide calling from the guide counts by a [Poisson-Gaussian mixture](https://doi.org/10.1016/j.cell.2022.05.013) fitted to each guide as in [crispat](https://doi.org/10.1093/bioinformatics/btae535) and pertpy, by the two-Gaussian mixture that Cell Ranger describes, by a fixed UMI floor, or by each cell's top guide as in the maximum rule of the [SCEPTRE book](https://timothy-barry.github.io/sceptre-book/assign-grnas.html), whose UMI floor bounds the cell's total guide UMIs where the floor of the other methods bounds each guide's UMIs, with the mixtures fitted to each sample separately when asked; and guide calls imported from Cell Ranger's CRISPR Guide Capture analysis. Either way, an artifact labels each cell with its one called guide, `"Negative"` or `"Multiple"`, with each cell's top guide and guide UMIs read from the streamed guide counts. The mixtures are fitted to exact histograms of each guide's UMI counts, which one streamed pass counts, so the cost of a fit follows the number of distinct counts rather than the number of cells. Perturbation labels merge the guides of each gene: a cell is labelled by the gene that its called guides target, with or without control guides, `"control"` with control guides only, `"Negative"` without a call, and `"Multiple"`, or its genes joined by `"+"`, when its guides target several genes; each cell's full set of genes is kept for high-MOI analyses, and the control targets are named, never guessed. Perturbation efficiency compares each target gene's expression in the cells that carry its perturbation alone, by target or by guide, with the control cells: the mean UMIs of each group as Cell Ranger reports them, the fraction of the control expression that remains after scaling each cell to the median library size of the compared cells that hold a UMI, and a two-sided [Mann-Whitney U](https://doi.org/10.1214/aoms/1177730491) test, by its normal approximation with the corrections for ties and continuity, with [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) adjustment, reading only the target genes of the compared cells. The [E-distance](https://doi.org/10.1038/s41592-023-02144-y) of scPerturb measures how far the cells of each perturbation sit from the control cells in a coordinate space such as a PCA, computed exactly from each group's sums and squared norms rather than from pairwise distances, with a permutation test whose p-values count the observed groups as one draw ([Phipson and Smyth 2010](https://doi.org/10.2202/1544-6115.1585)) and are adjusted by Benjamini-Hochberg. The perturbation signature of [Mixscape](https://doi.org/10.1038/s41588-021-00778-2) gives each cell its log-normalized values minus the log of the mean normalized values of its nearest control cells, found exactly in a coordinate space within each sample, as Seurat's `CalcPerturbSig` computes them; Mixscape's classes follow Seurat's `RunMixscape`: DE genes by Seurat's Wilcoxon rules, tested for every target in one pass over the counts with each split's control values sorted once, scaled signatures projected onto each target's perturbation direction, and a two-component normal mixture with a fixed control component, iterated until the calls of knocked-out cells and escapers stop changing. Association tests follow [SCEPTRE](https://doi.org/10.1186/s13059-024-03254-2)'s low-MOI test: a Poisson regression of each pair's counts on its treatment and control cells with the negative binomial size of the likelihood at the Poisson means, a score statistic compared with random sets of the pair's cells in up to three rounds, a skew-normal fit to the second round, and a calibration check on groups of control guides; and its [high-MOI test](https://doi.org/10.1186/s13059-021-02545-2), which compares the cells that carry any guide of a target with all other cells, fits each gene once on all cells with log guide counts among the covariates, and draws random sets by the conditional randomization test, from a logistic regression of each target's cells on the covariates, or by permutation. In both tests, a pair whose treatment cells the covariates determine, such as a target alone in a sample without control cells, gets no statistic, p-value or fold change, and the p-values count only the random sets that have a statistic. Perturbation profiles follow [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013): each cell's counts are scaled to the median total of the control cells, each feature is z-normalized against the mean and the standard deviation, with the n - 1 divisor, of the control cells of the cell's sample, and the profile of a target, or of a guide with each control guide its own, is its cells' mean z-score, from two streamed passes over the counts in bands of features. Perturbations are clustered by HDBSCAN, scikit-learn's implementation, on 1 minus the Pearson correlation of their profiles, as Replogle et al. clustered them, with an average-linkage order for heatmaps and a UMAP layout of the same distances. The [Perturbation API reference](/api/perturbation.html) describes each step, and [](../tutorials/crispr_screens.ipynb) runs them on a published screen.
- **Clones and lineage tracing:** clones imported from the files that lineage tracing tools write, such as CoSpar's, TREX's, CellTag-multi's, Numbat's and LARRY's, or from a cell metadata column, and joined to the store's cells by barcode. Clonal expansion places each cell by its clone's cells within its sample, with scirpy's bounds in cells or scRepertoire's bounds on the clone frequency, and describes each sample's clonal space by the cells of its largest clones, as scRepertoire's `clonalProportion` does. Diversity indices, Chao1, ACE and Hill numbers are reported per group, with the rarefaction and extrapolation of [Chao et al. (2014)](https://doi.org/10.1890/13-0133.1) to a common size or sample coverage, the Shannon extrapolation of [Chao et al. (2023)](https://doi.org/10.1002/ecm.1588), and rarefied probabilities computed as products of ratios rather than from log-gamma functions, which keeps them exact for large groups. Groups are compared by presence indices of the clones they share and by the abundance-weighted Morisita-Horn index, and chosen clones are followed across groups. Fate coupling follows the definitions of CoSpar ([Wang et al. 2022](https://doi.org/10.1038/s41587-022-01209-1)) and compares each pair of groups with a null of shuffled labels, which keeps each clone's and each group's cells and draws each permutation from a stream of its own, so the null does not depend on threads; the fate bias of early cells is the share of their clone's later cells in each fate, with a one-sided Fisher test of each clone in each fate.

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
| **Guide calling, fixed floor and top guide** | pertpy `main` (commit 9b5c53b) `assign_by_threshold` and `assign_to_max_guide`; crispat 0.9.11 `ga_ratio` | The same A549 screen | All 12,535 calls at a floor of 5 UMIs and all 5,529 of pertpy's top-guide calls at 5 UMIs; all 4,539 and 3,825 ratio calls at fractions 0.5 and 0.7 |
| **Perturbation efficiency** | Cell Ranger 6.0.0 `perturbation_efficiencies_by_target.csv` and `perturbation_efficiencies_by_feature.csv` | The same A549 screen, with Cell Ranger's calls imported and `controls=["Non-Targeting"]` | The same perturbed cells, the same 396 control cells and the same mean UMIs, within $2 \times 10^{-16}$, for all 45 targets and 77 guides; log2 fold changes with Spearman correlation 0.97 and the same sign for 40 of 41 targets and 66 of 67 guides, as described below |
| **E-distance** | scPerturb 0.1.0 `edist` and `etest`; pertpy `main` (commit 9b5c53b) `Distance` and `DistanceTest` | The same A549 screen, with Cell Ranger's calls imported, in a 50-component PCA of all 5,867 cells | E-distances within a relative $2.4 \times 10^{-12}$ of scPerturb's for all 45 targets and 88 guides; permutation p-values with Spearman correlation 0.988 by target and 0.997 by guide with scPerturb's, as described below |
| **Mixscape** | Seurat 5.5.1 `CalcPerturbSig` and `RunMixscape`; pertpy `main` (commit 9b5c53b) `Mixscape` | The THP-1 ECCITE-seq screen of [Papalexi et al. 2021](https://doi.org/10.1038/s41588-021-00778-2), 20,729 cells and 25 targets, with Seurat's PCA and variable genes | The same 20 nearest control cells for every cell; signatures within float32 rounding; the same DE genes for all 25 targets, with p-values within a relative $2 \times 10^{-13}$; and Seurat's class for all 20,729 cells, as described below |
| **Association tests** | sceptre 0.99.1 (GitHub commit 20c33349), run only | The same THP-1 screen: 25 targets against the 2,000 variable genes, 50,000 pairs, with the positive controls and the calibration check | The same QC for every pair; statistics within $10^{-6}$ for 39,062 of 40,517 tested pairs; p-values within the Monte Carlo noise between two sceptre runs; no false discovery among 40,517 negative-control pairs, as described below |
| **Association tests at high MOI** | sceptre 0.99.1 (GitHub commit 20c33349), run only | sceptredata's high-MOI example, 45,919 cells of the CRISPRi enhancer screen of [Gasperini et al. 2019](https://doi.org/10.1016/j.cell.2018.11.029) with 526 genes and 35 targets: 610 cis pairs and all 18,410 pairs of a target and a gene, by conditional resampling and by permutation, with the positive controls and the calibration check | The same QC for every pair; statistics within $10^{-6}$ for 17,809 of 18,142 tested pairs; p-values within the Monte Carlo noise between two runs of either tool; null distributions of 20 pairs alike, as described below |
| **Perturbation profiles** | The published pseudo-bulk profiles of [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013), `K562_essential_normalized_bulk_01.h5ad` | Their K562 essential screen, 310,385 cells, 2,057 targets and 48 GEM groups: profiles by guide pair over all 8,563 genes, with the GEM groups as samples | All 2,273 profiles within $1.9 \times 10^{-6}$ of the published ones, one float32 step at z-scores of 16 to 32, and 56.8% of the values bit for bit, as described below |
| **Perturbation clusters** | scikit-learn 1.9.1 `HDBSCAN`; Table S3 of [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013), made with the `hdbscan` package | Their published pseudo-bulk profiles of the 1,973 perturbations of the genome-wide K562 screen that they clustered, in 2,236 features that approximate their 2,319 | The labels of a direct scikit-learn call, exactly; against Table S3, 64 clusters with an adjusted Rand index of 0.977, and 1.000 over the 700 perturbations that both assign, as described below |
| **Clonal expansion** | scirpy 0.25.1 `clonal_expansion` and `summarize_clonal_expansion`; scRepertoire 2.8.0 `clonalHomeostasis` and `clonalProportion` | The clonotypes of two 10x immune profiling runs, PBMC 10k (6,189 cells with a receptor) and CD8+ T cells of healthy donor 1 (46,526 cells), with four pseudo samples and six pseudo clusters from a hash of each barcode | The category of every cell equals scirpy's, within samples and across all cells, and the summaries by cluster equal its counts of cells and clones; on the PBMC T cells, the frequency fractions of each sample equal `clonalHomeostasis` within $4 \times 10^{-16}$ and the cells of each rank window equal `clonalProportion`, for `CTstrict` and `CTaa` clonotypes |
| **Clonal diversity** | iNEXT 3.0.2 without bootstraps; scRepertoire 2.8.0 `clonalDiversity` without bootstraps; immunarch 0.10.3 `repDiversity`; vegan 2.7.6 `estimateR` | iNEXT's two examples, the pseudo samples of both 10x runs, all 46,526 CD8+ cells, and six edge cases | Rarefied, observed and extrapolated Hill numbers of orders 0, 1 and 2 within $1.4 \times 10^{-12}$ of iNEXT at every size, coverage within $3 \times 10^{-9}$, asymptotic estimates within $7 \times 10^{-13}$, standardized estimates within $10^{-7}$; every index within $6 \times 10^{-14}$ of scRepertoire, immunarch and vegan |
| **Clone overlap** | immunarch 0.10.3 `repOverlap`; scRepertoire 2.8.0 `clonalOverlap` | The clonotypes of the pseudo samples of CD8+ donor 1, about 500 shared clonotypes per pair, and of PBMC 10k with `CTstrict` and `CTaa` clonotypes | Shared clones, Jaccard, Sørensen, overlap, cosine, Morisita-Horn and abundance cosine within $5 \times 10^{-14}$ for every pair |
| **Fate coupling and fate bias** | CoSpar 0.5.0 `fate_coupling`, `pvalue_for_fate_coupling`, `fate_map` on `infer_Tmap_from_clonal_info_alone`, and `clonal_fate_bias` | A synthetic experiment of 990 cells at two time points with 120 clones of known fate profiles in six fates | Couplings within $6 \times 10^{-17}$ for the cosine of clone sizes and Jaccard, a relative $1.4 \times 10^{-15}$ for the normalized covariance once CoSpar's pseudocount is taken out, and a relative $5.1 \times 10^{-9}$ for the cosine of clone shares; null means within 3 standard errors and standard deviations within 7%; fate fractions within a relative $10^{-10}$; equal p-values for all 108 clones in all six fates |
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

**Guide calls.** The guide-calling comparisons ran on the A549 screen with the settings of each reference: `fixed` with `min_umi=5` against pertpy's `assign_by_threshold(5)`, `top_guide` with `min_umi=5` and `fraction_threshold=0` against `assign_to_max_guide(5)`, and `top_guide` with `min_umi=1` and `fraction_threshold` 0.5 or 0.7 against crispat's `ga_ratio` at the same fraction. Where pertpy assigns a cell its most expressed guide when that guide has at least 5 counts, `top_guide` follows the SCEPTRE book and bounds the cell's total guide UMIs, so it calls every cell that pertpy assigns, with the same guide, and also the cells whose guide UMIs reach 5 while their top guide's do not. The two floors call the same cells whenever `fraction_threshold` is at least `1 - 1 / min_umi`, such as 0.8 with the default floor of 5, and with any fraction at a floor of 1, as in the comparison with crispat. With its default of 500 optimiser steps, crispat stops short of convergence on 88 of the 91 guides it fits; with 10,000 steps, CyteArc's Poisson-Gaussian fit reproduces every one of crispat's 7,392 calls. For 2 guides, SRPK1-1 and TPR-2, crispat fits no model, while CyteArc and pertpy call them. With `method="gaussian"`, the smallest UMI count that CyteArc calls for a guide equals the threshold that Cell Ranger reports on 90 of 93 guides and is one UMI above it on the other 3. The comparison uses the smallest called count because Cell Ranger reports the smallest count it called, while CyteArc's `threshold` is the model's cutoff, which sits lower when no cell holds the counts in between. In its ratio method, crispat keeps every guide tied at a cell's largest count, while CyteArc calls the first of them in assay order; no tie reached the fractions compared here.

**Perturbation efficiency.** Cell Ranger estimates its fold changes with sSeq, which shrinks them, so they correlate with CyteArc's rather than equal them. The sign comparison covers the rows whose fold change is finite in CyteArc. The control cells hold no UMI of ACE2, DPP4 or TMPRSS2, so their 3 targets and 6 guides have the status `"no_control_expression"` and no fold change, while Cell Ranger reports positive fold changes for them with a p-value of 1. The perturbed cells of CTNNB1 and of 4 guides hold no UMI of their target gene, which gives a fold change of minus infinity in CyteArc and a negative one in Cell Ranger. Cell Ranger also lists 22 rows for pairs of targets, one for each gene of a pair, from the cells whose guides target both genes; CyteArc's rows cover cells with one target.

**E-distance.** scPerturb's `edist` computes the same statistic as CyteArc, with squared Euclidean distances and within-group means over distinct pairs. pertpy's `edistance` uses Euclidean distances and averages within each group over all pairs, each cell with itself included, so its values differ; with squared distances set on its metric and its within-group means converted to distinct pairs, they match CyteArc's within a relative $2.9 \times 10^{-12}$. Both tools adjust their p-values by Holm-Sidak by default, with the control row among the rows, which no guide can pass at 1,000 permutations. Adjusted by Benjamini-Hochberg instead, scPerturb's p-values give the same rows below 0.05 as CyteArc's for all 45 targets and 87 of 88 guides, and so do pertpy's. With 1,000 permutations, the tests by target took 0.15 s in CyteArc, 62 s in `etest` and 5.2 s in `DistanceTest`.

**Mixscape.** On the THP-1 screen of Seurat's Mixscape vignette, `signature` finds the same 20 nearest control cells as `CalcPerturbSig` for every cell, and its values equal minus Seurat's within float32 rounding, at most $2.4 \times 10^{-7}$. `mixscape` tests the same genes as `RunMixscape` for all 25 targets and finds the same DE genes, with p-values within a relative $2 \times 10^{-13}$ of those of `FindMarkers`; on the same values, its rank-sum kernel gives the p-values of SciPy's `mannwhitneyu` to the last bit. Every one of the 20,729 cells gets Seurat's class, 5,112 KO and 13,231 NP, with Seurat's random seed at 0, 1 or 2, which leaves Seurat's calls unchanged, and the posteriors differ by at most $7.5 \times 10^{-6}$, because the two EM implementations stop at slightly different points of the same fit. pertpy's `Mixscape` gives 96.8% of the cells the same class: it tests only the highly variable genes, adjusts its p-values by Benjamini-Hochberg, stops its EM at a looser tolerance, and weighs its posterior by the mixture weights, so it calls fewer KO cells for weak targets. On one thread, `signature` took 3.8 to 5.1 s and `mixscape` 10.4 to 11.5 s, 7.4 to 7.5 s on four, against 13.2 s for `CalcPerturbSig` and 587 to 830 s for `RunMixscape` with base R's Wilcoxon test, at peaks of 0.9 and 1.6 GiB against 4.6 and 7.7 GiB.

**Association tests.** On the THP-1 screen, with the same cells, guide assignments and covariates as sceptre, `associations` passes the same 40,517 of 50,000 pairs through QC, with the same nonzero cell counts. Its statistics equal sceptre's within $10^{-6}$ for 39,062 pairs, with a median difference of $2.2 \times 10^{-9}$; for 765 more, mostly cell-cycle genes such as RAD54L, SPC25 and HJURP, a negative binomial size of 0.01 reproduces sceptre's statistic within $10^{-5}$, although the likelihood peaks at larger sizes, where CyteArc's size lies. The random sets differ between the tools, so the p-values differ by Monte Carlo noise: the median difference of log10 p is 0.041 and the Spearman correlation of -log10 p 0.986, against 0.040 and 0.992 between two sceptre runs on shuffled cells, and over the pairs whose statistics agree the discoveries at FDR 0.1 overlap with a Jaccard index of 0.957, against 0.959 between two sceptre runs. The calibration check finds no false discovery among 40,517 negative-control pairs from 100 groups of 4 control guides, as sceptre does. sceptre floors p-values at $10^{-250}$ and computes skew-normal tails below about $10^{-10}$ inaccurately, so such p-values compare on a capped scale only. On a four-core laptop, the call that runs the discovery pairs, the positive controls and the calibration check took 126 s on four threads and 198 s on one, at peaks of 2.4 and 2.2 GiB, against 829 s for sceptre's three calls run one after another, or 585 s with its discovery on four processes, at peaks of 1.8 and 4.2 GiB.

**Association tests at high MOI.** On sceptredata's high-MOI example, with sceptre's cells, its guide calls at 5 UMIs, which `call_guides(method="fixed", min_umi=5)` reproduces in all 53,292 cases, and its default high-MOI formula with the batch as `sample_column`, `associations(moi="high")` gives every discovery, positive-control and calibration pair the nonzero cell counts and QC of sceptre. Its statistics equal sceptre's within $10^{-6}$ for 17,809 of the 18,142 tested pairs of every target with every gene, with a median difference of $3 \times 10^{-10}$. 85 of the others are all the pairs of three sparse genes, MT1G, NKG7 and SYT3, expressed in 327 to 430 cells: a negative binomial size of 0.01 reproduces sceptre's statistics within $3 \times 10^{-8}$, although the likelihood there is lower by 36 to 186 than at CyteArc's sizes, Poisson for SYT3. The other 248, all pairs of 17 genes, differ by at most $5 \times 10^{-6}$, as much as a change of the gene's size by at most 0.02% moves them, which leaves the likelihood within $10^{-4}$ of its maximum. Log2 fold changes agree within $2 \times 10^{-7}$. sceptre's standard error of the fold change is $\sqrt{\sum (y - r\mu)^2} / \sum \mu$ over the treatment cells, from the residuals at the fold change $r$, for all 18,142 pairs; CyteArc's comes from the negative binomial variance and the information of the fit, and is a median 1.02 times sceptre's, 0.94 to 1.08 for 90% of the pairs. The random sets differ between the tools, so the p-values differ by Monte Carlo noise: with conditional resampling over every target and gene, the median difference of log10 p is 0.034, the Spearman correlation of -log10 p 0.986 and the Jaccard index of the discoveries at FDR 0.1 0.94, against 0.034, 0.987 and 0.94 between two CyteArc runs with different seeds; over the 600 tested cis pairs they are 0.017, 0.997 and 0.92, against 0.015, 0.997 and 0.92 between two sceptre runs on shuffled cells. The null statistics of 20 pairs, the 499 of round 1 or the 4,999 of round 2 by each resampling, have the mean, standard deviation and skewness of sceptre's within 0.12, 0.1 and 0.27, and a two-sample Kolmogorov-Smirnov test gives a p-value below 0.05 for 1 of the 40 comparisons. The pairs split between the rounds as sceptre's do, 95.9%, 4.1% and 0.02% with conditional resampling, all 10 positive controls fall below $10^{-50}$ in both tools, and the calibration checks are excellent or mild in both, with 0 to 2 false discoveries among 600 or 18,142 negative-control pairs. With permutations, sceptre shares one set of permutations among all targets, so the null statistics of one gene under two targets correlate at 0.75 to 0.94, while CyteArc draws each target's sets, as with conditional resampling. On Modal, in containers of 1 CPU and 6 GiB like sceptre's, the call that tests every target with every gene by conditional resampling, with the positive controls and the calibration check, took 151.9 s on one thread, at a peak of 1,375 MiB, against 1,170 s for sceptre's three calls, at a peak of 889 MiB; in a container of 4 CPUs it took 37.9 s on four threads. CyteArc's times are those of a second call, after the first had compiled its kernels.

**Calibration at genome scale.** On the K562 essential CRISPRi screen of Replogle et al., 310,385 cells with 2,057 targets and 97 non-targeting guide pairs, the low-MOI tests of every target against 2,000 highly variable genes, with the default covariates, passed 3,928,249 of 4,114,000 pairs through QC, and the positive controls found 1,337 of the 1,361 targets whose own gene passes QC. The calibration check graded the run severe, with 37 false discoveries among 806,245 negative-control pairs. The 37 fell in 18 of the 97 non-targeting guide pairs, 12 of them in one pair, as large shifts of single genes, such as falls of 2.2 and 1.7 log2 for the adjacent paralogs MAGEA6 and MAGEA3 in one pair. The cells of these guide pairs spread over the 48 GEM groups like the other control cells, and their GEM composition predicts shifts of at most 0.03 log2, so these guide pairs act on genes of their own, as off-target repression would, rather than the test being miscalibrated.

**Perturbation profiles.** Replogle et al.'s methods leave four choices open, which rebuilding their K562 essential profiles from the cells settles: the row sums of the stored counts scale each cell's depth, while the totals stored with the file, which also count genes left out of the matrix, miss the published profiles by up to 1.04; the standard deviation divides by n - 1, while n misses them by up to 0.09; every non-targeting cell of a GEM group sets its z-scores; and every cell of a guide pair enters the pair's profile. With these choices, every profile correlates with its published profile at 1 - $1.6 \times 10^{-14}$ or more.

**Perturbation clusters.** Replogle et al. clustered the 1,973 perturbations of their genome-wide K562 screen with the strongest effects by HDBSCAN, from the `hdbscan` package, on 1 minus the Pearson correlation of their pseudo-bulk profiles, and Table S3 lists 64 clusters. Their published profiles give these distances once the features are chosen; the Anderson-Darling statistics that chose their 2,319 features are unpublished, so 2,236 features approximate them: the ten largest absolute mean z-scores of each profile, with its own target masked, and the expressed genes whose profiles vary most. On these distances, the functions that `cluster` calls give the labels of a direct call of scikit-learn's `HDBSCAN(metric="precomputed", min_cluster_size=4, min_samples=2, cluster_selection_method="eom")` exactly, and so do the distances that CyteArc computes from the profiles, which differ from the reference distances by at most $6 \times 10^{-8}$. The labels match Table S3 with an adjusted Rand index of 0.977 over all 1,973 perturbations and 1.000 over the 700 that both assign, 52 of the 64 clusters exactly; Mediator, the 40S and 60S ribosome, the RNA exosome, the proteasome, the chaperonin and the m6A writer keep the members of the table and leave out the same ones. scikit-learn's `min_samples` counts the point itself, so the package's `min_samples=1` is 2 here, and on these distances 1 and 2 give the same labels. Clustering the 1,973 profiles, with the order and the layout, takes 2.5 s on two threads.

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
