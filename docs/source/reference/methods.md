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

