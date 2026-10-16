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
  (chromVAR), enhancer-driven regulons from paired RNA and ATAC (SCENIC+), and co-regulated gene
  programs

## Implemented methods

CyteArc implements a computational stage itself when an external implementation would require materializing the full matrix or would detach the result from its inputs.
It uses established libraries where they fit the streaming and provenance model.
The aim is one coherent execution path, not reimplementation for its own sake.

- **Reduction and graph construction:** streamed normalization, covariance (Gram-matrix) PCA with an incremental fallback, randomized streaming {term}`LSI`, approximate nearest-neighbour search, UMAP, {term}`densMAP`, and graph-based t-SNE through the optional `tsne` extra (see {ref}`Optional t-SNE <installation_tsne>`).
- **Batch correction and mapping:** CyteArc implementations of [Harmony](https://doi.org/10.1038/s41592-019-0619-0) and [Symphony-style](https://doi.org/10.1038/s41467-021-25957-x) fixed-reference mapping, with label transfer and mapping diagnostics.
- **Matched multi-omics:** {term}`SNN integration` and [Hao-inspired WNN](https://doi.org/10.1016/j.cell.2021.04.048) integration for two or more assays, with WNN reporting one per-cell weight per modality.
- **Clustering:** [Leiden](https://doi.org/10.1038/s41598-019-41695-z) and a native implementation of [Paris hierarchical clustering](https://doi.org/10.48550/arXiv.1806.01664) with fixed and branch-adaptive cuts.
- **Trajectory analysis:** memory-aware [Population Balance Analysis](https://doi.org/10.1073/pnas.1714723115), supervised terminal-state fate probabilities, pseudotime aggregation and modules, and [MAGIC-style graph diffusion](https://doi.org/10.1016/j.cell.2018.05.061) for imputation.
- **Statistics and diagnostics:** marker AUC, the [Mann-Whitney U test](https://doi.org/10.1214/aoms/1177730491) with [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) correction, doublet scores, cluster separability, and integration metrics informed by the [scIB benchmark](https://doi.org/10.1038/s41592-021-01336-8).
- **Gene sets, protein, and sample tags:** [AUCell](https://doi.org/10.1038/nmeth.4463) and [WAGGR](https://doi.org/10.1093/bioadv/vbac016) activity scores, CITE-seq processing, and [cell-hashing](https://doi.org/10.1186/s13059-018-1603-1) demultiplexing.
- **Regulons and chromatin activity:** [GRNBoost2](https://doi.org/10.1093/bioinformatics/bty916)
  co-expression networks and [SCENIC](https://doi.org/10.1038/nmeth.4463) cisTarget pruning
  ([pySCENIC](https://doi.org/10.1038/s41596-020-0336-2)), [chromVAR](https://doi.org/10.1038/nmeth.4401)
  motif deviations with MOODS-equivalent motif matching, cistromes from motifs or CUT&RUN and
  ChIP-seq peaks, [SCENIC+](https://doi.org/10.1038/s41592-023-01938-4) enhancer-driven
  regulons, regulon specificity and connection specificity, and co-expression gene programs.

## Validation

Validation is method-specific.
Agreement measured for one method is not evidence of equivalence for the rest of the package, and none of these comparisons establish biological ground truth.

- HTO demultiplexing is checked against Seurat reference outputs.
- Symphony-style query correction is checked against a golden fixture generated with the R Symphony 0.1.3 `mapQuery` implementation, including a case with nonzero correction.
- Marker statistics are checked against SciPy, including continuity and large-tie corrections.
- Paris hierarchies are compared with the scikit-network reference implementation on tie-free graphs.
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
  pySCENIC 0.12.1 assigns it, and AUCell regulon scores agree within 2e-16.
  pySCENIC 0.12.1 gives most annotated motifs another enriched motif's curve; `prune_modules`
  uses each motif's own, so on pySCENIC's network the regulon names agree but the median target
  Jaccard is 0.25. These targets are more reproducible across GRNBoost2 seeds (median Jaccard
  0.33, against 0.20 to 0.24 with pySCENIC's pruning).
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
- Enhancer-driven regulons were not compared with a SCENIC+ run. Preranked GSEA matches a
  line-by-line transcription of gseapy 0.10.8's statistics and permutations in unit tests, and
  region-to-gene links use SCENIC+'s regressor settings. On the multiome sample, chromVAR scores
  CUT&RUN and ChIP-seq peak sets highest in the expected population for SPI1 (dendritic cells),
  LEF1 (CD4 T cells), and the B-cell factors PAX5, EBF1, POU2F2, and POU2AF1, but not for
  BCL11B binding measured in Jurkat cells.
  Against ChIP-Atlas binding within 10 kb of a gene's start (the GRETA gold standard), the
  fraction of factor-to-gene links with binding exceeds its random expectation by 8% for CyteArc
  regulons, 7% for pySCENIC regulons, 9% for enhancer-driven regulons, and 3% for each factor's
  top 50 GRNBoost2 targets.
- Gene programs are compared with Hotspot 1.1.1 modules on the same cells and highly variable
  genes. The six programs contain 89% of the 503 genes in Hotspot's ten modules, and 76% of those
  genes fall in their module's majority program (adjusted Rand index 0.45 on shared genes,
  against at most 0.008 for 100 permuted labelings). Hotspot splits the T-cell, monocyte, and
  B-cell programs into finer modules. Independent noise profiles form no programs, but the
  membership threshold assumes independent samples and ignores that programs are chosen from
  the same data: it screens chance groupings rather than testing programs, and profiles of
  smoothed cells form spurious ones.

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

