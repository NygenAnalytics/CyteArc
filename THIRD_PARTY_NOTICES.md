# Third-party notices

CyteArc is copyright Nygen Analytics AB and is licensed under the BSD 3-Clause License in the `LICENSE` file. Parts of CyteArc derive from the third-party software listed below, and the repository includes small excerpts of public data. Each keeps its own license, and the full license texts are at the end of this file. The dataset sections credit the data that the documentation and tests download, the networks that CyteArc downloads for its operations, and the data that validated the screen methods, and a later section lists optional packages that CyteArc can use but does not install.

## Software

### MyST Book Theme

- Source: <https://github.com/jupyter-book/myst-theme>
- Copyright (c) 2023 Executable Books.
- License: MIT
- Used in: `docs/build_docs.py` adjusts the documentation theme so navigation sections
  are expanded in the initial page and remain expanded when browser code starts.

### Scanpy

- Source: <https://github.com/scverse/scanpy>
- Copyright (c) 2025 scverse®; Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
- License: BSD-3-Clause
- Used in: `cytearc/features/scoring.py` (gene-set scoring adapted from `score_genes`),
  `cytearc/datastore/_operations/quality_control.py` (the control-set size of cell-cycle
  scoring, and the `simulation_ratio` and `expected_doublet_rate` defaults of doublet detection,
  those of `pp.scrublet`), and the categorical palettes in `cytearc/plotting/_style.py`.

### Scrublet

- Source: <https://github.com/swolock/scrublet>
- Copyright (c) 2018 Samuel Wolock
- License: MIT
- Used in: `cytearc/quality_control/doublets.py`, written from the method of Wolock, Lopez and
  Klein (2019) without its code: the doublet score from the fraction of simulated doublets among a
  cell's neighbors with its `(d + 1) / (k + 2)` estimate, the neighbors scaled by
  `1 + simulation_ratio`, the threshold between the two modes of the simulated doublets' scores,
  and the estimated doublet rate.

### umap-learn

- Source: <https://github.com/lmcinnes/umap>
- Copyright (c) 2017, Leland McInnes
- License: BSD-3-Clause
- Used in: `cytearc/embeddings/umap.py` (the densMAP density terms, the fuzzy set union, and the
  epoch loop of `_optimize_layout`, adapted from `optimize_layout_euclidean` in `umap/layouts.py`
  at 0.5.12) and the UMAP parameter descriptions in `cytearc/datastore/_operations/embeddings.py`.

### Seurat

- Source: <https://github.com/satijalab/seurat>
- Copyright (c) 2021 Seurat authors
- License: MIT
- Used in: `cytearc/neighbors/integration.py` (the modality weights and constants of
  `FindModalityWeights` and `MultiModalNN`), `cytearc/quality_control/hto.py` (the HTODemux
  procedure), `scripts/export_seurat_wnn_golden.R` (expressions copied from
  `FindModalityWeights`), the reference equations in `tests/test_knn_utils.py`, and
  `cytearc/perturbation/signature.py` (the perturbation signature of `CalcPerturbSig` and
  `PerturbDiff` in `R/mixscape.R`, read at the main branch in October 2026: each cell's nearest
  non-targeting cells within its split, and its log-normalized values against the log of their
  mean normalized values, normalized in the order of the operations of `LogNorm` in
  `src/data_manipulation.cpp`), `cytearc/perturbation/mixscape.py` and the `mixscape` operation in
  `cytearc/datastore/_operations/perturbation_mixscape.py` (the iterations of `RunMixscape` and
  `TopDEGenesMixscape` in the same file, the filters, fold change and Bonferroni adjustment of
  `FindMarkers` and `FoldChange` in `R/differential_expression.R`, and the scaling of
  `FastDenseRowScale` in `src/data_manipulation.cpp`), and `cytearc/neighbors/graph.py`
  (`shared_neighbor_graph`: the Jaccard index of two neighbourhoods and the pruning below 1/15 of
  `ComputeSNN` in `src/snn.cpp`, with each cell counted among its own neighbours as
  `FindNeighbors` in `R/clustering.R` counts it, read at the main branch in October 2026).

### mgatk

- Source: <https://github.com/caleblareau/mgatk>
- Copyright (c) 2020 Caleb Lareau
- License: MIT
- Used in: `cytearc/variants/statistics.py`, the per-variant statistics of `call_variants` in
  `mgatk/processing/variant_calling.py` at 0.9.0 (commit
  7ca7d74bf6104101e94f53f1a6a0f4f7d263e8ab): the pooled mean, the variance of the covered cells'
  allele frequencies and its ratio to the mean with the `1e-11` offset, the cells with 2 reads of
  the allele on each strand, the strand correlation over the cells that show the allele, the cells
  over 5, 10, 20 and 95%, the largest frequency and the mean coverage.

### Signac

- Source: <https://github.com/stuart-lab/signac>
- Copyright (c) 2021 Tim Stuart
- License: MIT
- Used in: `cytearc/variants/statistics.py` (the thresholds of the `"atac"` preset, as
  `vignettes/mito.Rmd` applies them) and `cytearc/variants/frequencies.py` (the method of
  `FindClonotypes` in `R/mito.R`: square-root allele frequencies, cosine neighbours with `k = 10`
  and clustering of the shared-neighbour graph), at commit 12d1ff82774890b08e55b3101f36fab83e674b88.

### MAESTER-2021

- Source: <https://github.com/petervangalen/MAESTER-2021>
- Copyright (c) 2021 vangalenlab
- License: MIT
- Used in: `cytearc/variants/statistics.py`, the mean base quality of a variant, the mean of the
  cells' and strands' mean qualities where they read the allele, as
  `3_Cell_line_mixes_variants/3.1_SW_CellLineMix_variants.R` computes it at commit
  6853e8c1f2342f7ec4ef28d8466b285a2b5ec0f3.

### chromVAR

- Source: <https://github.com/GreenleafLab/chromVAR>
- Copyright (c) 2017 Stanford University
- License: MIT
- Used in: `cytearc/features/enrichment/chromvar.py` (background-peak sampling translated from
  `getBackgroundPeaks`, and the deviation computation) and `tests/test_enrichment_chromvar.py`
  (chromVAR's test inputs and expected values).

### MOODS

- Source: <https://github.com/jhkorhonen/MOODS>
- Copyright (C) 2007-2015  Pasi Rastas, Janne H. Korhonen, Petri Martinmäki
- License: MOODS is licensed under the GPL-3.0-or-later or the Biopython License Agreement, at the
  recipient's choice. CyteArc uses it under the Biopython License Agreement.
- Used in: `cytearc/features/genomic/motifs.py` (log-odds scoring, `min_delta`, and the p-value
  threshold computation, translated from `core/moods_tools.cpp`).

### BPCells

- Source: <https://github.com/bnprks/BPCells>
- Copyright (c) 2021-2023 BPCells contributors
- License: BPCells is licensed under the Apache-2.0 or the MIT License, at the recipient's choice.
  CyteArc uses it under the MIT License.
- Used in: `cytearc/readers/_seurat/operations.py` (`_rank_values` follows `ColwiseRank`).

### decoupler

- Source: <https://github.com/scverse/decoupler>
- Copyright (c) 2025, scverse®; Copyright (c) 2025, Pau Badia i Mompel, Saez lab
- License: BSD-3-Clause
- Used in: `cytearc/features/enrichment/aucell.py` (the AUC kernel and the `n_up` default).

### arboreto

- Source: <https://github.com/aertslab/arboreto>
- Copyright (c) 2017, Thomas Moerman
- License: BSD-3-Clause
- Used in: `cytearc/regulons/grn.py` (the early-stopping monitor, the GRNBoost2 defaults, and the
  importance scaling).

### SciPy

- Source: <https://github.com/scipy/scipy>
- Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
- License: BSD-3-Clause
- Used in: `cytearc/features/markers/regression.py` (the correlation and its p-value, adapted from
  `scipy.stats.linregress`), and `cytearc/perturbation/mixscape.py` (the normal approximation of
  the rank-sum test with its tie and continuity corrections, in the order of the operations of
  `scipy.stats.mannwhitneyu`, so that the p-values agree).

### scib-metrics

- Source: <https://github.com/YosefLab/scib-metrics>
- Copyright (c) 2022, Adam Gayoso
- License: BSD-3-Clause
- Used in: `cytearc/metrics/lisi.py` (the names, defaults, and scaling of `ilisi_knn` and
  `clisi_knn`).

### EoN (Epidemics on Networks)

- Source: <https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks>
- Copyright (c) 2016 Joel C Miller
- License: MIT
- Used in: `cytearc/plotting/cluster_tree.py` (`_hierarchy_positions`, the tree layout adapted
  from `hierarchy_pos`).

### scikit-learn

- Source: <https://github.com/scikit-learn/scikit-learn>
- Copyright (c) 2007-2026 The scikit-learn developers.
- License: BSD-3-Clause
- Used in: `tests/paris_quality_gate.py` (the adjusted Rand index in scikit-learn's
  pair-confusion form). This is test code and is not part of the installed package.

### AnnData

- Source: <https://github.com/scverse/anndata>
- Copyright (c) 2025, scverse®; Copyright (c) 2017-2018, P. Angerer, F. Alexander Wolf, Theis Lab
- License: BSD-3-Clause
- Used in: `cytearc/readers/_text.py` (the renaming of repeated feature IDs under
  `duplicate_feature_ids="make_unique"`, which follows `make_index_unique`, the function behind
  `var_names_make_unique`).

### netlib loess (dloess)

- Source: <https://www.netlib.org/a/dloess>, the local regression code that R's `loess` and
  scikit-misc's `loess` wrap
- Copyright (c) 1989, 1992 by AT&T; William S. Cleveland, Eric Grosse, and Ming-Jen Shyu
- License: the netlib dloess permission notice, reproduced below
- Used in: `cytearc/features/variability.py` (`fit_loess(surface="interpolate")`, which places the
  vertices of its k-d tree as dloess does: the bounding interval widened by 0.5% of the data range,
  cells of at most `floor(0.2 * span * n)` points, and median splits moved off tied values). The
  rules were written in Python from that description, not translated from the Fortran, and are
  checked against the fitted values of scikit-misc 0.5.3.

### pertpy

- Source: <https://github.com/scverse/pertpy>
- Copyright (c) 2021, Lukas Heumos; Copyright (c) 2025, scverse®
- License: MIT
- Used in: `cytearc/perturbation/mixtures.py` (the Poisson-Gaussian guide fit: its expectation maximisation steps, its starting points, the rules that reject a fit, and the threshold rule, adapted from `pertpy/preprocessing/_guide_rna_mixture.py` at commit `9b5c53b`).

### crispat

- Source: <https://github.com/velten-group/crispat>
- Copyright (c) 2024 Jana M. Braunger
- License: MIT
- Used in: `cytearc/perturbation/mixtures.py` (the Poisson-Gaussian model of `ga_poisson_gauss` at v0.9.11: its priors, the cells and guides it fits, and its threshold rule).

### CoSpar

- Source: <https://github.com/ShouWenWang-Lab/cospar>
- Copyright (c) 2020 Shou-Wen Wang
- License: MIT
- Used in: `cytearc/clones/fate.py`, whose fate coupling methods follow the definitions of CoSpar 0.5.0's `get_normalized_covariance` and `coarse_grain_clone_over_cell_clusters` (the `"SW"`, `"Weinreb"` and `"Jaccard"` methods and the normalisation of each group and then each clone), whose null shuffles labels as `pvalue_for_fate_coupling` does, and whose fate fractions and Fisher tests follow `compute_state_potential` on the clonal transition map of `infer_Tmap_from_clonal_info_alone` and `clonal_fate_bias`, written from that source without its pseudocounts. `tests/cospar_0_5_0_golden.json` holds CoSpar 0.5.0's outputs on a synthetic experiment that the test suite compares with.

### scirpy

- Source: <https://github.com/scverse/scirpy>
- Copyright (c) 2020 Gregor Sturm, Tamas Szabo, Francesca Finotello, and the Institute of Bioinformatics, Medical University of Innsbruck; Copyright (c) 2025 scverse®
- License: BSD-3-Clause
- Used in: `cytearc/repertoire/chains.py` (the receptor model of version 0.25.1: the loci of each receptor arm, the rules that make a contig a chain and rank a cell's chains into two slots per arm, the inference of a contig's locus from its gene calls, and the names and rules of the receptor types, subtypes and chain pairings of `index_chains` and `chain_qc`), reimplemented on encoded contigs, `cytearc/repertoire/clonotypes.py` (the options and rules of `define_clonotypes` and `define_clonotype_clusters` that decide which chains of two cells must match), reimplemented on encoded chains, and `cytearc/clones/sequences.py` (the Hamming, normalized Hamming and Levenshtein distances of `ir_dist`, with the normalized distance's percentage rounded half up as scirpy rounds it), reimplemented as an exact search for the pairs within a cutoff.

### TREX

- Source: <https://github.com/frisen-lab/TREX>
- Copyright 2018 Leonie von Berlin
- License: MIT
- Used in: `cytearc/clones/calling.py` (the clone graph of `trex/clone.py` and `trex/graph.py` at commit `c1ade47`: cells of identical cloneID sets as one node, as `CellGraph._precluster_cells` makes them, edges above a Jaccard threshold, as `_make_graph` and `_is_similar` draw them, the bridges of `CellGraph.bridges`, the local cut vertices of `Graph.local_cut_vertices` that `CellGraph.doublets` removes when they hold one cell, and the order of `run10x`'s rounds of bridges and doublets), reimplemented on arrays; `cytearc/clones/barcode_clones.py` (the rules of `trex/cli/run10x.py`, `trex/filters.py` and `trex/cell.py`: the low-complexity filter of `is_low_complexity`, the correction of `correct_clone_ids` into each cluster's most frequent cloneID, ties to the larger, the minimum length of `compute_cells`, and the two filters of `filter_cells`), reimplemented on records; and `cytearc/clones/sequences.py` (the comparison of `is_similar` at the positions where both cloneIDs hold a base, with a minimum of compared positions).

### GIFT reproducibility code

- Source: <https://github.com/clareaulab/gift_figures_reproducibility>
- Copyright (c) 2026 Lareau Lab
- License: MIT
- Used in: `cytearc/variants/genotypes.py` (the genotype model of `get_p`, `get_cell_genotypes` and `assign_genotypes` in `1_figure_CL_proof_of_concept/code/utils_00.py` at commit `35564bc`: each gapfill's probability of a probe's two alleles, normalized over them; each UMI's score of a genotype, the larger of its swap probability and its probability under the genotype, of which the heterozygote takes the larger half; a cell's likelihood as the product of its UMIs' scores; the posterior under a uniform prior; and a call when the largest posterior exceeds `min_p` and enough UMIs have a swap probability below 0.1), reimplemented on arrays as sums of logarithms.

### Cassiopeia

- Source: <https://github.com/YosefLab/Cassiopeia>
- Copyright (c) 2019, Matthew G Jones
- License: MIT
- Used in: `cytearc/readers/recorder_alleles.py` (the eight lineage marks of each PEtracer site, as `presets_petracer.yaml` of branch 3.0.0 at commit 2867d2d74181937b75d79d76afe8839662762044 lists them, with `CTTGGG` read as an unedited `EMX1` site; the tokens of Cassiopeia's allele tables, where text holding `None` is an uncut site; the `pass` column and the `sites` attribute of its 3.0 parquet tables; and the default states of a character matrix, `-1`, `-` and `NA` missing and `0` and `*` unedited), `cytearc/clones/recorders.py` (characters named `{integration}-{site}`, and an integration's most abundant allele kept, as `alleles_to_characters` and `filter_alleles` keep it) and `cytearc/readers/_treedata.py` (the `obsm['characters']`, `uns['missing_state']`, `uns['unmodified_state']` and `uns['character_states']` that Cassiopeia 3.0's solvers read). Written from Cassiopeia's documentation and code; no code was copied.

### treedata

- Source: <https://github.com/YosefLab/treedata>
- Copyright (c) 2024, William Colgan
- License: BSD-3-Clause
- Used in: `cytearc/readers/_treedata.py` (the on-disk layout of `docs/fileformat.md` and of what treedata writes: a TreeData file is an AnnData file whose root's `encoding-type` is `treedata` and which holds `label`, `allow_overlap`, `alignment`, `obst` and `vart` beside AnnData's elements, with one group per tree in a `.zarr` store). Written from that layout; no code was copied.
- Used in: `cytearc/readers/phylogeny.py` (where the Newick grammar leaves a choice, the reader decides as `read_nwk` of treedata 0.4.0 does: an unquoted number on an internal node is a support value rather than a name, a missing `;` after the last tree is accepted, and a length written NaN counts as no length; TreeData files are read from the layout of `obst` that treedata writes). The scan was written in Python and Numba from those rules, not translated from treedata's code, and is checked against `read_nwk`'s output on 18 adversarial trees.

### CARLIN

- Source: <https://gitlab.com/hormozlab/carlin>
- Copyright (c) 2021 Duluxan Sritharan
- License: MIT
- Used in: `cytearc/clones/allele_clones.py` (the clonal p-value of `@Bank/Bank.m` at commit 7b10f94611473de8c70dd78c2f911507f52c626b: `clonal_pvalue_eqn`, the chance that an allele of Poisson rate λ seen among M observations arose more than once, and its use in `compute_clonal_pvalue`, which tests each allele of a sample against its bank rate with M the sample's observations of edited alleles and gives the unedited template a p-value of 1), reimplemented on arrays from the Poisson tails. Alleles that the bank lacks take a rate of 0, where `compute_clonal_pvalue` takes its model's mean rate of unobserved alleles.

## Data in this repository

- **JASPAR** (<https://jaspar.elixir.no>), CC BY 4.0: the motif matrices MA0004.1 (Arnt) and
  MA0006.1 (Ahr::Arnt), and an excerpt of the MEME export of MA0004.1, in
  `tests/test_genomic_motifs.py`. Rauluseviciute et al. (2024), Nucleic Acids Research,
  doi:10.1093/nar/gkad1059.
- **10x Genomics**, CC BY 4.0: `tests/seurat_wnn_5_5_1_golden.json` holds Seurat 5.5.1 outputs on principal components of 300 cells of the `tenx_8K_pbmc_citeseq` dataset, `tests/scanpy_hvg_1_12_4_golden.json` holds scanpy 1.12.4 highly variable gene selections and scores computed from the RNA counts of the 1k PBMC CITE-seq test fixture by `scripts/export_scanpy_hvg_golden.py`, and the reference images in `tests/visual/showcase/` are rendered from the same fixture. Both datasets are credited below. `tests/test_readers_lineage_barcodes.py` checks three UMIs against the 2-bit values that `pbmc_1k_protein_v3_molecule_info.h5` (Cell Ranger 3.0.0) and the `SC3_v3_NextGem_DI_CRISPR_A549_5K` sample's `molecule_info.h5` (Cell Ranger 6.0.0) store for them.
- **GEO GSE245108**: hashtag counts in `tests/seurat_hto_5_5_1_golden.json`, from Zhang et al.
  (2024), Nature Immunology, doi:10.1038/s41590-024-01782-4.
- **HGNC** (<https://www.genenames.org>), CC0 1.0: the previous symbols of the default cell-cycle
  genes in `cytearc/quality_control/cell_cycle_genes.py` (`cell_cycle_previous_symbols`), from the
  HGNC REST service in October 2026. Seal et al. (2023), Nucleic Acids Research,
  doi:10.1093/nar/gkac888.
- **TREX** (<https://github.com/frisen-lab/TREX>), MIT, Copyright 2018 Leonie von Berlin: `tests/data/clones/trex_clones.txt` holds 41 rows of `tests/expected/clones.txt` at commit c1ade47971bfcb5242ebf213f7160215567647b4, every cell of three clones and 30 cells of the fourth. `tests/data/barcodes/trex_molecules.txt` and `tests/data/barcodes/trex_molecules_corrected.txt` hold every row of 31 cells of `tests/expected/molecules.txt` and `tests/expected/molecules_corrected.txt` at the same commit, and `tests/data/barcodes/trex_molecules_2022.txt` the same cells' rows of `tests/expected/molecules.txt` at commit dff02eea348ad870c148b8d590e6b973b25dc17a, each with its header. `tests/data/barcodes/trex_10x_molecules.txt.gz` and `tests/data/barcodes/trex_10x_clones.txt.gz` are `tests/expected/molecules.txt` and `tests/expected/clones.txt` at commit c1ade47971bfcb5242ebf213f7160215567647b4, and `tests/data/barcodes/trex_smartseq3_molecules.txt.gz` and `tests/data/barcodes/trex_smartseq3_clones.txt.gz` are `tests/expected_smartseq3/molecules.txt` and `tests/expected_smartseq3/clones.txt` at the same commit, gzip-compressed and otherwise unchanged. `tests/data/barcodes/trex_doublets_molecules.txt.gz` is a synthetic molecule table that TREX's `write_molecules` wrote, and `tests/trex_0_4_dev412_golden.json` holds the clones and doublets that TREX 0.4.dev412 (commit c1ade47) called from it with its own functions. The license text is at the end of this file.
- **pycashier** (<https://github.com/brocklab/pycashier>), MIT, Copyright (c) 2024 brocklab, and Copyright (c) 2021 brocklab at tag v23.1.2: `tests/data/barcodes/pycashier_test.umi_cell_labeled.barcode.tsv` is `tests/data/reference/outs-scrna/test.umi_cell_labeled.barcode.tsv` at commit 1ced13ff46deb88d731386c578ed5ead628b9228, and `tests/data/barcodes/pycashier_test.cell_record_labeled.barcode.tsv` is `tests/data/reference/outs-scrna/test.cell_record_labeled.barcode.tsv` at tag v23.1.2, unchanged. The license text is at the end of this file.
- **CellTag-multi**, CC BY 4.0: `tests/data/clones/celltag_multi_iep_clones.csv` holds the 24
  rows of six clones of `b4.rna&atac.r1&2_master_rev1_annotated.csv`, the iEP clone table of
  "Processed single-cell RNA and ATAC data for the CellTag-multi iEP reprogramming time course"
  by Samantha Annette Morris and Kunal Jindal (Zenodo, version 1.1.0,
  doi:10.5281/zenodo.23442375), with its columns unchanged.
- **Signac** (<https://github.com/stuart-lab/signac>), MIT, Copyright (c) 2021 Tim Stuart:
  `tests/data/variants/signac_test_mgatk/` holds `inst/extdata/test_mgatk` at commit
  12d1ff82774890b08e55b3101f36fab83e674b88, unchanged, and `tests/test_readers_variants.py`
  checks the allele frequencies that Signac's `tests/testthat/test-mito.R` pins.
- **MiTo** (<https://github.com/andrecossa5/MiTo>), MIT, Copyright (c) 2023 acox1, with data
  under CC BY 4.0: `tests/data/variants/mito_mtscatac/` holds 5 cells of
  `docs/source/data/mtscATAC_pL1000/` at commit 7f5a519f85e1680d58335328f0594651c8bfafe5, with the
  reference gzipped; the pL1000 data come from "Lentiviral barcoding ground-truth datasets for
  benchmarking mtDNA-based lineage tracing" (Figshare, doi:10.6084/m9.figshare.31323481).
  `tests/data/variants/mito_maester/` holds 3 cells of `docs/source/data/maester_MDA_clones/` at
  the same commit, from "nf-MiTo data (AFM MDA_clones and metadata)" by Andrea Cossa (Zenodo,
  doi:10.5281/zenodo.17225334), with the hg38 mitochondrial reference of the pL1000 folder as
  `chrM_refAllele.txt.gz`, which maegatk writes beside its tables.
- **Variant goldens**: `tests/mgatk_0_9_0_golden.json`, `tests/signac_1_17_1_golden.json` and
  `tests/mito_2_0_0_golden.json` hold what mgatk 0.9.0, Signac 1.17.1 with Seurat 5.6.0, and MiTo
  2.0.0 (all MIT) computed on Signac's `test_mgatk` fixture, the synthetic truths of
  `tests/variant_truth.py`, and MiTo's 50-cell `mtscATAC_pL1000` and `maester_MDA_clones` folders
  at commit 05725c3e69ad2294bff03bb9dcbfcb8499600e10 (CC BY 4.0, from the sources named above).
  `tests/test_variant_goldens.py` downloads MiTo's folders for its tests marked `integration`.
- **Variant masks** in `cytearc/variants/masks.py`: the 25 NUMT-derived false positives of
  Supplemental Table S1 and the artifact-prone positions of Laricchia et al. (2022), Genome Research
  32:569-582, doi:10.1101/gr.276013.121, and the RNA-DNA differences of Bar-Yaacov et al. (2013),
  Genome Research 23:1789-1796, doi:10.1101/gr.161265.113, as lists of positions and alleles.
- **cellsnp-lite** (<https://github.com/single-cell-genetics/cellsnp-lite>), Apache-2.0:
  `tests/data/variants/cellsnp_lite/` holds the first 60 records and the first 40 cells of
  `data/csp_mode1` at commit 8e8ba00c2adfe6fc758385f99c61ab5545387838.
- **MQuad** (<https://github.com/single-cell-genetics/MQuad>), Apache-2.0:
  `tests/data/variants/mquad/` holds `example_test/passed_ad.mtx`, `passed_dp.mtx` and
  `passed_variant_names.txt` at commit c2d750c5f279dca4f200274fc126d89e619fdd58, the rows of
  `example_test/BIC_params.csv` of the 12 passed variants, and in `cellSNP.samples.tsv` the 81
  sample names of `example/example.vcf.gz`, the cells that MQuad's example read.
- **VarTrix** (<https://github.com/10XGenomics/vartrix>), MIT, Copyright (c) 2018 10x Genomics,
  Inc.: `tests/data/variants/vartrix/` holds `test/test.vcf`, `test/barcodes.tsv`,
  `test/test_coverage.mtx` and `test/test_coverage_ref.mtx` at v1.1.23 (commit
  e8164f98cc1eaa2c958c9e133cddf023f88df2e6), unchanged.
- **giftwrap** (<https://github.com/clareaulab/giftwrap>), MIT:
  `tests/data/variants/giftwrap_counts.1.filtered.h5` holds the first 60 cells of
  `sample_data/counts.1.filtered.h5` at v0.6.2 and the 494 probe and gapfill columns they hold,
  in the file's layout.
- **Single-cell_BPDCN** (<https://github.com/petervangalen/Single-cell_BPDCN>), MIT, Copyright (c) 2022 Peter van Galen: `tests/data/variants/got/` holds the XV-seq tables `04_XV-seq/Pt1Dx/TET2.2847/Patient1_TET2.2847_summTable.txt` and `04_XV-seq/Pt5Dx/ZRSR2.500/Patient5_ZRSR2.500_summTable.txt` at commit 5c5f361aafcc6fa6a7d6f8b33556bb0bfd76a03b, unchanged, and in `Pt1Dx_CellBarcode_Allowlist_in_TET2.2847.txt` and `Pt5Dx_CellBarcode_Allowlist_in_ZRSR2.500.txt` the barcodes of `05_Stats/5.1_CB-AllowLists/Pt1Dx_CellBarcode_Allowlist.txt` and `Pt5Dx_CellBarcode_Allowlist.txt` at the same commit that each table names, in the allowlist's order. `tests/test_variant_genotype_studies.py` compares the calls with `04_XV-seq/4.2_Genotyping_stats.txt`.
- **GIFT goldens**: `tests/giftwrap_0_6_2_golden.json` holds what giftwrap 0.6.2 (MIT) read from its 60-cell sample and from the BC8 plex of GEO GSM9714721 (GSE319999; Blattman et al. (2026), bioRxiv, doi:10.64898/2026.04.11.717967), each probe's UMI and read totals and a digest of the sample's entries, and the calls and posteriors that the GIFT reproducibility code (MIT, credited above) made from them. GEO states no license, and the NCBI places no restrictions on the use of its data.
- **CARLIN**, CC BY 4.0: `tests/data/recorders/carlin_fo864_lf/` holds `AlleleAnnotations.txt`, `AlleleColonies.txt` and `Results.txt` of `Analyzed/EB/FO864/SC/LF/Amplicon/`, unchanged, and `tests/data/recorders/carlin_bulk_24h/` the first five lines of `AlleleAnnotations.txt` and of `AlleleColonies.txt` of `Analyzed/ChronicInduction/NegDox/Trial1/24h/`, the latter cut to the first 20 UMIs of each line, from `Analyzed.zip` of the Zenodo record <https://zenodo.org/records/4817818>. Bowling et al. (2020), Cell 181:1410-1422, doi:10.1016/j.cell.2020.04.048.
- **DARLIN**, CC BY 4.0: `tests/data/recorders/darlin_ll837_skull_CA/`, `_TA/` and `_RA/` hold the alleles of 60 cells read on all three arrays and of 10 more read on each array alone, the lines of `AlleleAnnotations.txt` and `AlleleColonies.txt` that name them, and each array's `Results.txt` unchanged, from `DARLIN_public/DATA/20220815_LL837_10X_3A/<array>/DARLIN/results_cutoff_override_3/LL837-skull-<array>_S<n>/` of `DARLIN_public.zip` (Zenodo, doi:10.5281/zenodo.11929508). Li et al. (2023), Cell 186:5183-5199, doi:10.1016/j.cell.2023.09.019.
- **darlin-core** (<https://github.com/JarningGau/darlin-core>), MIT, Copyright (c) 2024 LiliLab: `tests/data/recorders/darlin_core_CA_benchmark.tsv` is `tests/data/CA_benchmark.tsv` at commit a07eaf16b1a916b6d6be936237438b32a51e0a20, unchanged. The license text is at the end of this file.
- **Cassiopeia** (<https://github.com/YosefLab/Cassiopeia>), MIT, Copyright (c) 2019, Matthew G Jones: `tests/data/recorders/cassiopeia_3432_NT_T1_alleletable.txt` holds every row of 30 cells of `notebooks/data/3432_NT_T1_alleletable.txt` at commit 1ee5959eb9d3f8d4d26e2af5678234493bf54d6d, with its header. The license text is at the end of this file.
- **PEtracer**, MIT: `tests/data/recorders/petracer_E7.5-R1_characters.csv.gz` is `GSM9925880_E7.5-R1_characters.csv.gz` of GEO GSE342226, renamed and otherwise unchanged: the character matrix of embryo E7.5-R1. Colgan et al. (2026), Cell 189:7154-7176, doi:10.1016/j.cell.2026.09.050. The same data are released under the MIT License as the Mouse Embryonic Lineage Atlas by William Colgan and Luke Koblan (Zenodo, doi:10.5281/zenodo.19892784); the license text is at the end of this file.
- **PEtracer**, CC BY 4.0: `tests/data/recorders/petracer_barcoded_tracing_alleles.csv`, `petracer_B16F10_kinetics_alleles.csv` and `petracer_preedited_10x_invitro_alleles.csv` hold whole rows of `barcoded_tracing_alleles.csv`, `B16F10_kinetics_alleles.csv` and `preedited_10x_invitro_alleles.csv` of "PEtracer Data" by William Colgan, Luke Koblan, Kathryn Yost, Pu Zheng and others (figshare, doi:10.6084/m9.figshare.28473866): rows with insertions outside the eight lineage marks, the rows of cells with two alleles of one integration, and rows chosen at random, with each table's header. Koblan et al. (2025), Science 390, doi:10.1126/science.adx3800.
- **DNA Typewriter**, CC BY 4.0: `tests/data/recorders/typewriter_2022_hek293t.csv` holds every row of 50 cells of Supplementary Data `41586_2022_4922_MOESM5_ESM.csv`, every row of six slot fields, and rows that edit a contracted tape beyond its length or leave a slot blank before an edit, with its header. Choi et al. (2022), Nature 608:98-107, doi:10.1038/s41586-022-04922-8.
- **ConvexML** (<https://github.com/songlab-cal/ConvexML>), BSD-3-Clause, Copyright (c) 2025, Song Lab @ Cal: `tests/data/recorders/convexml_tree_0_character_matrix.csv` is `tests/public_api_tests/data/tree_0_character_matrix.csv` at commit e93f6aee113044c29c1f123023f11fe9e6753bde, unchanged. The license text is at the end of this file.
- **PEtracer**, MIT: `tests/data/phylogenies/E7.5-R1-C1.nwk.gz`, `E7.5-R1-C2.nwk.gz` and `E7.5-R1_barcodes.tsv.gz` are the files `GSM9925880_E7.5-R1_E7.5-R1-C1.nwk.gz`, `GSM9925880_E7.5-R1_E7.5-R1-C2.nwk.gz` and `GSM9925879_E7.5-R1_barcodes.tsv.gz` of GEO GSE342226, renamed and otherwise unchanged: the two clone trees of embryo E7.5-R1 and its cells. Colgan et al. (2026), Cell 189:7154-7176, doi:10.1016/j.cell.2026.09.050. The same data are released under the MIT License as the Mouse Embryonic Lineage Atlas by William Colgan and Luke Koblan (Zenodo, doi:10.5281/zenodo.19892784); the license text is at the end of this file.
- **ConvexML** (<https://github.com/songlab-cal/ConvexML>), BSD-3-Clause, Copyright (c) 2025, Song Lab @ Cal: `tests/data/phylogenies/convexml_tree_0.nwk` is `tests/public_api_tests/data/tree_0_newick.txt` at commit e93f6aee113044c29c1f123023f11fe9e6753bde, unchanged. The license text is at the end of this file.
- **treedata** (<https://github.com/YosefLab/treedata>), BSD-3-Clause: treedata 0.4.0 wrote `tests/data/phylogenies/truth_tree_treedata_0_4_0.h5td`, `truth_tree_treedata_0_4_0.zarr` and `truth_forest_treedata_0_4_0.h5td` from the synthetic experiment of `tests/clone_truth.py`, so they hold no third-party data.
- **pycea** (<https://github.com/YosefLab/pycea>), BSD-3-Clause: `tests/pycea_0_4_0_golden.json` holds the clades that pycea 0.4.0's `tl.clades` cut from the PEtracer and ConvexML trees above, after treedata 0.4.0's `read_nwk` read them. No code of pycea was copied.
- **Potency reference**: `tests/data/potency/scent_reference.npz` holds the counts of 64 cells of the `bastidas-ponce_4K_pancreas-d15_rnaseq` example, credited under "Example datasets", on 5,778 genes, the edges among those genes of the mouse STRING v12.5 network credited under "Networks that CyteArc downloads" (CC BY 4.0), and the signaling entropy rate and CCAT that SCENT 1.1 (<https://github.com/aet21/SCENT>, GPL-3, commit 02b4bc64a45431860d7d756bca58ca3306b5eae9) computed from them; `scent_reference.json` records how they were made. SCENT was run to produce the values, and no code of SCENT was copied.
- **Color palettes** in `cytearc/plotting/_style.py`: the 10- and 20-color sets are Tableau 10 and
  20 as distributed with matplotlib (the 10-color set with Scanpy's colorblind substitutions), the
  28-color set is from Zeileis, Hornik, and Murrell (2009), Computational Statistics & Data
  Analysis, doi:10.1016/j.csda.2008.11.033, and the 102-color set is from Tatarize's "Color
  Distribution Methodology". All four are as compiled in Scanpy.

## Example datasets

`cytearc.datasets.connect("cytearc_docs")` downloads these from Nygen's public `Nygen/cytebase`
bucket on Hugging Face. They are not part of the CyteArc package. Nygen converted each source to a
CyteArc store and analyzed it, and the `manifest.json` file next to each store records the steps.
Cite the original studies when you use the data.

| Dataset | Source | License or terms |
| --- | --- | --- |
| `tenx_5K_pbmc_rnaseq` | "5k Peripheral blood mononuclear cells (PBMCs) from a healthy donor (v3 chemistry)", 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| `tenx_8K_pbmc_citeseq` | "10k PBMCs from a Healthy Donor - Gene Expression and Cell Surface Protein", 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| `tenx_10K_pbmc-v1_atacseq` | "10k Peripheral blood mononuclear cells (PBMCs) from a healthy donor", Single Cell ATAC v1.0, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| `tenx_1.3M_brain_rnaseq` | "1.3 Million Brain Cells from E18 Mice", 10x Genomics, <https://www.10xgenomics.com/datasets>. Unmodified Cell Ranger matrix | CC BY 4.0 |
| `kang_15K_pbmc_rnaseq`, `kang_14K_ifnb-pbmc_rnaseq`, `kang_29K_ctrl-ifnb_pbmc_rnaseq` | Kang et al. (2018), Nature Biotechnology 36:89-94, doi:10.1038/nbt.4042; GEO GSE96583 | GEO; no license stated |
| `bastidas-ponce_4K_pancreas-d15_rnaseq` | Bastidas-Ponce et al. (2019), Development 146:dev173849, doi:10.1242/dev.173849; GEO GSE132188. The processed file `data/Pancreas/endocrinogenesis_day15.h5ad` comes from the Theis lab repository <https://github.com/theislab/scvelo_notebooks>, published with scVelo (Bergen et al. (2020), Nature Biotechnology, doi:10.1038/s41587-020-0591-3) | GEO; the processed file states no license |
| `xin_1K_pancreas_rnaseq` | Xin et al. (2016), Cell Metabolism 24:608-615, doi:10.1016/j.cmet.2016.08.018; GEO GSE81608 | GEO; no license stated |
| `swanson_7K_pbmc_teaseq` | Swanson et al. (2021), eLife 10:e63632, doi:10.7554/eLife.63632. Built from `GSM5123951_PBMC_permcells_TEA-seq_SeuratObject.rds` by Lucas Graybuck, Elliott Swanson, and Peter Skene (Zenodo, doi:10.5281/zenodo.6360802) and the paper's Figure 4 source data 2 | CC BY 4.0 |
| `tabula_sapiens`, `tabula_sapiens_100k` | The Tabula Sapiens Consortium (2024), bioRxiv, doi:10.1101/2024.12.03.626516; "Tabula Sapiens - All Cells" from CZ CELLxGENE Discover, <https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5>. `tabula_sapiens_100k` is a 100,000-cell sample | CC BY 4.0, and the Tabula Sapiens Data Release Policy, <https://tabula-sapiens.sf.czbiohub.org/whereisthedata>, whose full text is in each store's manifest |
| `annotations` | Gene-body coordinates derived from GENCODE, Frankish et al. (2023), Nucleic Acids Research, doi:10.1093/nar/gkac1071 | No restrictions stated |

CC BY 4.0 is the Creative Commons Attribution 4.0 International license,
<https://creativecommons.org/licenses/by/4.0/>.

## Test datasets

`tests/download_fixtures.py` downloads these from the same bucket. They are not part of the CyteArc
package.

| File | Source | License or terms |
| --- | --- | --- |
| `1K_pbmc_citeseq.h5` | "1k Human PBMCs with TotalSeq-B Human TBNK Antibody Cocktail, 3' LT v3.1", 10x Genomics, <https://www.10xgenomics.com/datasets>. Unmodified filtered matrix | CC BY 4.0 |
| `500_pbmc_atac.zarr.tar.gz` | "500 Peripheral Blood Mononuclear Cells (PBMCs) from a Healthy Donor (Next GEM v1.1)", 10x Genomics, <https://www.10xgenomics.com/datasets>. Converted to a CyteArc store | CC BY 4.0 |
| `cell_attributes.csv`, `knn_*.npy`, `atac_knn_*.npy`, `markers_cluster1.csv`, `pseudotime_*`, `aggregated_*`, `ptime_modules_group_1.npy` | Analysis outputs computed from the two 10x Genomics datasets above | CC BY 4.0 |
| `toy_cr_dir.tar.gz`, `toy_cr_dir_empty.tar.gz` | Synthetic Cell Ranger-style directories | BSD-3-Clause, as in `LICENSE` |
| `seurat_assay5_synthetic.rds` | `tests/data/synthetic.rds` from <https://github.com/harryhaller001/readseurat> at commit 8a688b47df27f90e98a4c57ddd9e47c0e5ded01e | MIT, Copyright (c) 2026 harryhaller001 |
| `seurat_v4_1_3_pbmc_mye.rds` | `pbmc10k_mye_small_velocyto.rds` by Hua Yichao, SeuratExtend example data (Zenodo, doi:10.5281/zenodo.10944066), derived from 10x Genomics PBMC data | CC0 1.0; the underlying 10x Genomics data are CC BY 4.0 |

`tests/test_clone_files.py` downloads `GSM4185644_stateFate_cytokinePerturbation_clone_matrix.mtx.gz`
and `GSM4185644_stateFate_cytokinePerturbation_metadata.txt.gz` from GEO GSE140802 (Weinreb et al.
(2020), Science 367:eaaw3381, doi:10.1126/science.aaw3381) for a test marked `integration`; GEO
states no license, and the NCBI places no restrictions on the use of its data.

`tests/test_barcode_files.py` downloads `GSM9076301_SG05-Capture-1-Rep1_S1.counts.tsv.gz` and `GSM9076301_SG05-Capture-1-Rep1_S1_cell_barcode_annotation.tsv.gz`, BARtab's outputs for one sample of GEO GSE301145 (Gordon et al. (2025), Cancer Discovery 15:2117-2138, doi:10.1158/2159-8290.CD-24-1517), for a test marked `integration`; GEO states no license, and the NCBI places no restrictions on the use of its data. BARtab itself is licensed under the GPL-3.0, so CyteArc reads its outputs from its documentation and these files, and none of its code was read.

`tests/test_readers_variants.py`, `tests/test_variant_genotype_goldens.py` and `tests/test_variant_genotype_studies.py` download these for tests marked `integration`, which skip when the host is unreachable: `chrM_refAllele.txt` and `pL1000_mgatk.variant_stats.tsv.gz` of Figshare doi:10.6084/m9.figshare.31323481 (CC BY 4.0); GIFT's per-UMI tables of the BC8 plex of GEO GSM9714721 (GSE319999); the GoT tables `GSM3309839_ET02.CALR.txt.gz` of GEO GSE117825 (Nam et al. (2019), Nature 571:355-360, doi:10.1038/s41586-019-1367-0) and `GSM6198713_MDS01.SF3B1.tsv.gz` and `GSM6198716_MDS04.SF3B1.tsv.gz` of GEO GSE204845 (Cell Stem Cell (2023) 30:1262-1281.e8, doi:10.1016/j.stem.2023.07.012); and GoT-Multi-ML's example outputs `example_run/got_multi_ml_run/ironthrone_out_pp.csv` and `ml_denoise/got_multi_out_refined.csv` at commit c4a6721d64c0710e6fc2fa3356f9ad4f55e796e8 of <https://github.com/AnnaNamLab/GoT-Multi-ML>, which carries no license, so they are read and never checked in. GEO states no license, and the NCBI places no restrictions on the use of its data.

`tests/test_tapestri_files.py` downloads these for tests marked `integration`, which skip when the host is unreachable: Mission Bio's Tapestri example `4-cell-lines-AML-multiomics.dna+protein.h5` of Zenodo doi:10.5281/zenodo.20818440 (CC BY-SA 4.0), and the DAb-seq genotypes `GSM9567643.hdf5` of GEO GSE324062. Both are read and never checked in; GEO states no license, and the NCBI places no restrictions on the use of its data. Mission Bio's mosaic is licensed under the GPL-3.0 and its Tapestri Pipeline is proprietary, so the Tapestri reader follows Mission Bio's documentation and these files, and none of their code was read. The DAb-seq reader follows the layout that the DAb-seq pipeline (<https://github.com/AbateLab/DAb-seq>, Apache-2.0) writes, checked in its source; none of its code is copied.

`tests/test_allele_downloads.py` downloads `sc_tape/example_data/e3v8_subsample.cell_consensus.tsv.gz` of <https://github.com/shendurelab/dtt-mouse> at commit a491be02509d5ae862181a076f9e9006600dc096, a 1-in-40 subsample of the single-cell tape table of Yu et al. (2026), Science, doi:10.1126/science.ael0508, for a test marked `integration`. The repository is licensed under the GPL-3.0, so the file is never checked in, and CyteArc reads DNA Typewriter's tables from their documentation and this file; none of its code was read.

The Seurat import tutorial also uses `seurat_v4_1_3_pbmc_mye.rds` and embeds a 581-byte
compressed synthetic RDS with three cells and two genes. The synthetic object was authored
for CyteArc using the repository's R serialization helpers. It demonstrates skipped list
metadata, an active SCT assay and an explicitly assigned coordinate role; it contains no
third-party observations and is licensed under BSD-3-Clause, as in `LICENSE`.

## Data that tutorials download from their publishers

| Tutorial | Source | License or terms |
| --- | --- | --- |
| Regulons and gene programs | "PBMC from a Healthy Donor - Granulocytes Removed Through Cell Sorting (3k)", Single Cell Multiome ATAC + Gene Expression, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| Regulons and gene programs | cisTarget ranking database, motif-to-factor table, and factor list from the Stein Aerts lab, <https://resources.aertslab.org/cistarget/> | No terms stated. The v10 motif collection includes motifs derived from TRANSFAC Professional, a commercial database, so confirm the terms before commercial use |
| Regulons and gene programs | JASPAR 2024 CORE vertebrate motifs, Rauluseviciute et al. (2024), Nucleic Acids Research, doi:10.1093/nar/gkad1059 | CC BY 4.0 |
| Regulons and gene programs | GENCODE release 32 genome and annotation, Frankish et al. (2023), Nucleic Acids Research, doi:10.1093/nar/gkac1071 | No restrictions stated |
| Regulons and gene programs | SPI1 CUT&RUN peaks of experiment SRX26995709 from ChIP-Atlas, Zou et al. (2024), Nucleic Acids Research, doi:10.1093/nar/gkae358. ChIP-Atlas (c) Shinya Oki (Kyoto University) licensed under CC Attribution-Share Alike 4.0 International | CC BY-SA 4.0, <https://creativecommons.org/licenses/by-sa/4.0/> |
| Harmony batch correction, Donor-level comparisons, Pseudobulk export | Binvignat et al. (2024), JCI Insight, doi:10.1172/jci.insight.178499, from CZ CELLxGENE Discover, <https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b> | CC BY 4.0 |
| HTO demultiplexing | Stoeckius et al. (2018), Genome Biology 19:224, doi:10.1186/s13059-018-1603-1; GEO GSE108313 (GSM2895282, GSM2895283) | GEO; no license stated |
| Pooled CRISPR screens | Papalexi et al. (2021), Nature Genetics 53:322-331, doi:10.1038/s41588-021-00778-2; GEO GSE153056 (GSM4633614, GSM4633615, GSM4633616, GSM4633618, and the series' ECCITE-seq metadata) | GEO; no license stated |
| Lineage tracing | Weinreb et al. (2020), Science 367:eaaw3381, doi:10.1126/science.aaw3381; GEO GSE140802 (GSM4185642: the in vitro experiment's cell metadata, clone matrix, gene names and normalized expression) | GEO; no license stated |
| Mitochondrial lineage tracing | Lareau et al. (2020), Nature Biotechnology 39:451-461, doi:10.1038/s41587-020-0645-6; "Selected data for Massively parallel single-cell mitochondrial DNA genotyping and chromatin profiling" by Caleb Lareau and Leif Ludwig (Zenodo, doi:10.5281/zenodo.3977808): the colorectal cancer sample's peak-barcode matrix, per-barcode metrics and mgatk allele counts | CC BY 4.0 |
| Cell calling | "1k PBMCs from a Healthy Donor (v2 chemistry)", raw feature-barcode matrix, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| Ambient RNA removal | "1k PBMCs from a Healthy Donor (v2 chemistry)", 10x Genomics, <https://www.10xgenomics.com/datasets>. Raw and filtered Cell Ranger matrices | CC BY 4.0 |
| Immune repertoires | "Human PBMC from a Healthy Donor, 10k cells (v2)", 5' gene expression with antibody, TCR and BCR libraries, Cell Ranger 5.0.0, 10x Genomics, <https://www.10xgenomics.com/datasets>. Filtered feature-barcode matrix and the AIRR Rearrangement files of the TCR and BCR libraries | CC BY 4.0 |
| Lineage recorders | Colgan et al. (2026), Cell 189:7154-7176.e20, doi:10.1016/j.cell.2026.09.050; GEO GSE342226 (GSM9925879: the expression matrix of embryo E7.5-R1; GSM9925880: its character matrix and its two clone trees). The same data are released as the Mouse Embryonic Lineage Atlas by William Colgan and Luke Koblan (Zenodo, doi:10.5281/zenodo.19892784), whose cell table `E7.5-E10_cells.csv` annotates the embryos' cells; the tutorial reads its first 185,933 bytes, the header and the rows of E7.5-R1's donor cells | MIT; the license text is at the end of this file |
| CyteBase datasets and CyteBase composite datasets | Datasets from CZ CELLxGENE Discover, each credited in its CyteBase catalog entry | The license of each dataset on CELLxGENE Discover, usually CC BY 4.0 |

## Data that validated the screen methods

The comparisons of pooled CRISPR screen methods in [Methods and limitations](https://docs.nygen.io/CyteArc/reference/methods) ran on these datasets. The repository holds none of them, and neither the tests nor the documentation download them.

| Validation | Source | License or terms |
| --- | --- | --- |
| Guide calls, perturbation efficiency and E-distances | "5k A549, Lung Carcinoma Cells, No Treatment Transduced with a CRISPR Pool", Cell Ranger 6.0.0, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| Mixscape and association tests | Papalexi et al. (2021), Nature Genetics 53:322-331, doi:10.1038/s41588-021-00778-2, as the SeuratData package `thp1.eccite.SeuratData` 3.1.5 | CC BY 4.0 |
| Association tests at high MOI | The high-MOI example of the `sceptredata` R package, <https://github.com/Katsevich-Lab/sceptredata>, downsampled from Gasperini et al. (2019), Cell 176:377-390.e19, doi:10.1016/j.cell.2018.11.029 | MIT |
| Perturbation profiles, and the calibration check at genome scale | The K562 essential screen of Replogle et al. (2022), Cell 185:2559-2575.e28, doi:10.1016/j.cell.2022.05.013, from "Replogle et al. 2022 processed Perturb-seq datasets", figshare+, doi:10.25452/figshare.plus.20029387.v1 | CC BY 4.0 |
| Perturbation clusters | The published genome-wide K562 profiles of Replogle et al. (2022) from the same figshare+ record, and Table S3 of the paper; the names of the 1,973 perturbations that they clustered, from the authors' data browser, <https://gwps.wi.mit.edu> | CC BY 4.0; the data browser states no license |

## Data that validated the potency score

The comparison of potency scores in [Methods and limitations](https://docs.nygen.io/CyteArc/reference/methods#potency-method) ran on these datasets, read from CyteBase's public stores of release 2610, which convert them from CZ CELLxGENE Discover. The repository holds none of them, and neither the tests nor the documentation download them.

| Validation | Source | License or terms |
| --- | --- | --- |
| Pancreas | The `bastidas-ponce_4K_pancreas-d15_rnaseq` example, credited under "Example datasets" | GEO; the processed file states no license |
| Bone marrow | Zeng et al. (2025), Blood Cancer Discovery 6:307-324, doi:10.1158/2643-3230.BCD-24-0342, as `zeng_2025_balanced_bone_marrow_reference_map_hematopoietic_development_cd2f23c1` | CC BY 4.0 |
| Thymus | Steier et al. (2023), Nature Immunology 24:1579-1590, doi:10.1038/s41590-023-01584-0, as `steier_2023_mouse_single_cell_multi_omic_cd4_cd8_lineage_commitment_c14c54f8` | CC BY 4.0 |
| Hippocampus | De Donno et al. (2026), Molecular Psychiatry 31:1823-1836, doi:10.1038/s41380-025-03417-y, as `de_donno_2026_mouse_single_cell_adult_ventral_hippocampus_b2370cf4` | CC BY 4.0 |
| Gut | Elmentaite et al. (2020), Developmental Cell 55:771-783.e5, doi:10.1016/j.devcel.2020.11.010, as `elmentaite_2020_paediatric_human_gut_4_14y_8e47ed12` | CC BY 4.0 |
| Epidermis | Cheng et al. (2018), Cell Reports 25:871-883, doi:10.1016/j.celrep.2018.09.006, as `cheng_2018_skin_f512b8b6` | CC BY 4.0 |

## Data that validated the clone import

The clone sizes that `ds.clones.import_clones` saves were checked against counts taken with SciPy and pandas from these files. The repository holds none of them, and the tests download neither; the lineage tracing tutorial downloads LARRY's, as listed above.

| Validation | Source | License or terms |
| --- | --- | --- |
| Clones of LARRY's in vitro clone matrix | `GSM4185642_stateFate_inVitro_clone_matrix.mtx.gz` and `GSM4185642_stateFate_inVitro_metadata.txt.gz` from GEO GSE140802, Weinreb et al. (2020), Science 367:eaaw3381, doi:10.1126/science.aaw3381 | GEO; no license stated |
| Clones of CellTag-multi's clone table | `b4.rna&atac.r1&2_master_rev1_annotated.csv` of "Processed single-cell RNA and ATAC data for the CellTag-multi iEP reprogramming time course" by Samantha Annette Morris and Kunal Jindal (Zenodo, version 1.1.0, doi:10.5281/zenodo.23442375) | CC BY 4.0 |

## Data that validated the immune repertoire methods

The receptor import, the clonotypes, the gene usage and junction lengths, and the AIRR export in the [repertoire API page](https://docs.nygen.io/CyteArc/api/repertoire.html) were compared with scirpy, scRepertoire and Cell Ranger, and the export checked with the AIRR Community's airr library, on these datasets. The repository holds none of them, and the tests download neither; the immune repertoires tutorial downloads the first dataset's matrix and AIRR files, as listed above, rather than the contig annotations compared here.

| Validation | Source | License or terms |
| --- | --- | --- |
| Receptor import, clonotypes, gene usage and junction lengths, AIRR export | "Human PBMC from a Healthy Donor, 10k cells (v2)", Single Cell Immune Profiling, the TCR and BCR contig annotations of Cell Ranger 5.0.0, and for the import also those of Cell Ranger 4.0.0, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| Receptor import and clonotypes | "CD8+ T cells of Healthy Donor 1", Single Cell Immune Profiling, the contig annotations of Cell Ranger 3.0.2, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |

## Data that validated allele clones

The clones called from recorder alleles in the [clones API page](https://docs.nygen.io/CyteArc/api/clones.html) were compared with DARLIN's published clones on these data, read at run time in a Modal container. The repository holds none of them, and neither the tests nor the documentation download them.

| Validation | Source | License or terms |
| --- | --- | --- |
| Allele import and clones | DARLIN's experiment LL837, `DARLIN_public/DATA/20220815_LL837_10X_3A/` of `DARLIN_public.zip`: the CARLIN pairs of its five tissues on three arrays, and MosaicLineage's `df_sc_data_10.csv`, `df_sc_data_10_HQ.csv`, `df_sc_data_10_assigned_clones.csv` and `df_sc_data_10_cells_with_assigned_clones.csv` (Zenodo, doi:10.5281/zenodo.11929508). Li et al. (2023), Cell 186:5183-5199, doi:10.1016/j.cell.2023.09.019 | CC BY 4.0 |
| Allele banks | `reference/reference_merged_alleles_{CA,TA,RA}.csv` and `reference_merged_alleles_{CA,TA,RA}_Gr.csv` of MosaicLineage (<https://github.com/ShouWenWang-Lab/MosaicLineage>) at commit 7660afbd10c4cb31cbe973c918f3b7dac35a9b75 | No license file; read, not redistributed |
| Allele banks | `reference/allele_bank/allele_bank_Gr_{CA,TA,RA}.csv.gz` of the `darlin` CLI (<https://github.com/JarningGau/darlin>) at commit 82ae9db957ce752c26c7536c46a2f3daca97cfa8 | "Proprietary", no license file; read, not redistributed |

## Networks that CyteArc downloads

CyteArc downloads these files from Nygen's public `Nygen/cytebase` bucket on Hugging Face into its cache, `$XDG_CACHE_HOME/cytearc/networks`, the first time it needs them, and checks each copy against the size and SHA-256 digest that `cytearc.datasets.STRING_NETWORKS` records. They are not part of the CyteArc package.

- **STRING v12.5** (<https://string-db.org>), CC BY 4.0: `human_9606.npz` and `mouse_10090.npz` in the bucket's `cytearc_networks` repository, built by `scripts/build_string_networks.py` from STRING v12.5's `protein.links.detailed`, `protein.info` and `protein.aliases` files for human (taxon 9606) and mouse (taxon 10090). The changes: STRING's `database` channel is left out, since it draws on curated resources such as KEGG whose own terms apply; the other channel scores are recombined with STRING's documented formula; links scoring at least 0.700 are kept; proteins become genes through STRING's preferred names and `Ensembl_gene` aliases; and the largest connected component is kept. Each file's `metadata` names STRING, its version, the license, the citation, these changes and the URL, size and SHA-256 digest of each source file. Cite Szklarczyk D, Nastou K, Koutrouli M, et al. (2025), "The STRING database in 2025: protein networks with directionality of regulation", Nucleic Acids Research 53:D730-D737, doi:10.1093/nar/gkae1113. The license text is the last one below.

## Optional packages that CyteArc does not install

CyteArc contains no code from these GPL-licensed packages, and its default installation does not
include them:

- [sgtsnepi](https://github.com/alex-unofficial/sgtsnepi-python) (GPL-3.0) computes t-SNE. The
  `tsne` extra installs it, and the `test` and `docs` extras install it on Linux x86_64.
- [igraph](https://pypi.org/project/igraph/) (GPL-2.0-or-later) and
  [leidenalg](https://pypi.org/project/leidenalg/) (GPL-3.0-or-later) serve the optional
  `"igraph"` and `"leidenalg"` Leiden backends. No CyteArc extra installs them.

## License texts

### Scanpy

```text
BSD 3-Clause License

Copyright (c) 2025 scverse®
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### umap-learn

```text
BSD 3-Clause License

Copyright (c) 2017, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Seurat

```text
# MIT License

Copyright (c) 2021 Seurat authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Scrublet

```text
Copyright (c) 2018 Samuel Wolock

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### chromVAR

```text
Copyright (c) 2017 Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MOODS (Biopython License Agreement)

```text
Biopython License Agreement

Permission to use, copy, modify, and distribute this software and its
documentation with or without modifications and for any purpose and
without fee is hereby granted, provided that any copyright notices
appear in all copies and that both those copyright notices and this
permission notice appear in supporting documentation, and that the
names of the contributors or copyright holders not be used in
advertising or publicity pertaining to distribution of the software
without specific prior permission.

THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL
WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT
OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
OR PERFORMANCE OF THIS SOFTWARE.
```

### BPCells (MIT License)

```text
Copyright (c) 2021-2023 BPCells contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### decoupler

```text
BSD 3-Clause License

Copyright (c) 2025, scverse®
Copyright (c) 2025, Pau Badia i Mompel, Saez lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### arboreto

```text
BSD 3-Clause License

Copyright (c) 2017, Thomas Moerman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### SciPy

```text
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### scib-metrics

```text
BSD 3-Clause License

Copyright (c) 2022, Adam Gayoso
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### EoN

```text
Copyright (c) 2016   Joel C Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS OR SPRINGER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### scikit-learn

```text
BSD 3-Clause License

Copyright (c) 2007-2026 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### AnnData

```text
BSD 3-Clause License

Copyright (c) 2025, scverse®
Copyright (c) 2017-2018, P. Angerer, F. Alexander Wolf, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### netlib loess (dloess)

```text
The authors of this software are Cleveland, Grosse, and Shyu.
Copyright (c) 1989, 1992 by AT&T.
Permission to use, copy, modify, and distribute this software for any
purpose without fee is hereby granted, provided that this entire notice
is included in all copies of any software which is or includes a copy
or modification of this software and in all copies of the supporting
documentation for such software.
THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR AT&T MAKE ANY
REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
```

### readseurat

```text
MIT License

Copyright (c) 2026 harryhaller001

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### pertpy (MIT License)

```text
Copyright (c) 2021, Lukas Heumos
Copyright (c) 2025, scverse®

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### crispat (MIT License)

```text
Copyright (c) 2024 Jana M. Braunger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### scirpy (BSD 3-Clause License)

```text
BSD 3-Clause License

Copyright (c) 2020 Gregor Sturm
Copyright (c) 2020 Tamas Szabo
Copyright (c) 2020 Francesca Finotello
Copyright (c) 2020 Institute of Bioinformatics, Medical University of Innsbruck
Copyright (c) 2025 scverse®
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### TREX (MIT License)

```text
Copyright 2018 Leonie von Berlin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### CoSpar (MIT License)

```text
The MIT License (MIT)
Copyright (c) 2020 Shou-Wen Wang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### CARLIN (MIT License)

```text
MIT License

Copyright (c) 2021 Duluxan Sritharan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### pycashier (MIT License)

```text
MIT License

Copyright (c) 2024 brocklab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Signac (MIT License)

```text
Copyright (c) 2021 Tim Stuart

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### mgatk (MIT License)

```text
MIT License

Copyright (c) 2020 Caleb Lareau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MAESTER-2021 (MIT License)

```text
MIT License

Copyright (c) 2021 vangalenlab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MiTo (MIT License)

```text
Copyright (c) 2023 acox1

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### VarTrix (MIT License)

```text
Copyright (c) 2018 10x Genomics, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
```

### giftwrap (MIT License)

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### GIFT reproducibility code (MIT License)

```text
MIT License

Copyright (c) 2026 Lareau Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Single-cell_BPDCN (MIT License)

```text
Copyright (c) 2022 Peter van Galen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### cellsnp-lite and MQuad (Apache License 2.0)

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### Cassiopeia (MIT License)

```text
MIT License

Copyright (c) 2019, Matthew G Jones

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### darlin-core (MIT License)

```text
MIT License

Copyright (c) 2024 LiliLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### PEtracer data (MIT License)

```text
MIT License

Copyright (c) 2026 William Colgan and Luke Koblan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ConvexML (BSD 3-Clause License)

```text
BSD 3-Clause License

Copyright (c) 2025, Song Lab @ Cal

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### treedata (BSD 3-Clause License)

```text
BSD 3-Clause License

Copyright (c) 2024, William Colgan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### STRING (Creative Commons Attribution 4.0 International)

```text
Attribution 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More considerations
     for the public:
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution 4.0 International Public License ("Public License"). To the
extent this Public License may be interpreted as a contract, You are
granted the Licensed Rights in consideration of Your acceptance of
these terms and conditions, and the Licensor grants You such rights in
consideration of benefits the Licensor receives from making the
Licensed Material available under these terms and conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.

  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  j. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  k. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part; and

            b. produce, reproduce, and Share Adapted Material.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.


=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.
```
