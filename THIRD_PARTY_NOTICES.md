# Third-party notices

CyteArc is copyright Nygen Analytics AB and is licensed under the BSD 3-Clause License in the `LICENSE` file. Parts of CyteArc derive from the third-party software listed below, and the repository includes small excerpts of public data. Each keeps its own license, and the full license texts are at the end of this file. The dataset sections credit the data that the documentation and tests download and the data that validated the screen methods, and a later section lists optional packages that CyteArc can use but does not install.

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
- Used in: `cytearc/embeddings/umap.py` (the densMAP density terms and the fuzzy set union)
  and the UMAP parameter descriptions in `cytearc/datastore/_operations/embeddings.py`.

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
  `FastDenseRowScale` in `src/data_manipulation.cpp`).

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

## Data in this repository

- **JASPAR** (<https://jaspar.elixir.no>), CC BY 4.0: the motif matrices MA0004.1 (Arnt) and
  MA0006.1 (Ahr::Arnt), and an excerpt of the MEME export of MA0004.1, in
  `tests/test_genomic_motifs.py`. Rauluseviciute et al. (2024), Nucleic Acids Research,
  doi:10.1093/nar/gkad1059.
- **10x Genomics**, CC BY 4.0: `tests/seurat_wnn_5_5_1_golden.json` holds Seurat 5.5.1 outputs on
  principal components of 300 cells of the `tenx_8K_pbmc_citeseq` dataset,
  `tests/scanpy_hvg_1_12_4_golden.json` holds scanpy 1.12.4 highly variable gene selections and
  scores computed from the RNA counts of the 1k PBMC CITE-seq test fixture by
  `scripts/export_scanpy_hvg_golden.py`, and the reference images in `tests/visual/showcase/` are
  rendered from the same fixture. Both datasets are credited below.
- **GEO GSE245108**: hashtag counts in `tests/seurat_hto_5_5_1_golden.json`, from Zhang et al.
  (2024), Nature Immunology, doi:10.1038/s41590-024-01782-4.
- **HGNC** (<https://www.genenames.org>), CC0 1.0: the previous symbols of the default cell-cycle
  genes in `cytearc/quality_control/cell_cycle_genes.py` (`cell_cycle_previous_symbols`), from the
  HGNC REST service in October 2026. Seal et al. (2023), Nucleic Acids Research,
  doi:10.1093/nar/gkac888.
- **TREX** (<https://github.com/frisen-lab/TREX>), MIT, Copyright 2018 Leonie von Berlin:
  `tests/data/clones/trex_clones.txt` holds 41 rows of `tests/expected/clones.txt` at commit
  c1ade47971bfcb5242ebf213f7160215567647b4, every cell of three clones and 30 cells of the fourth.
  The license text is at the end of this file.
- **CellTag-multi**, CC BY 4.0: `tests/data/clones/celltag_multi_iep_clones.csv` holds the 24
  rows of six clones of `b4.rna&atac.r1&2_master_rev1_annotated.csv`, the iEP clone table of
  "Processed single-cell RNA and ATAC data for the CellTag-multi iEP reprogramming time course"
  by Samantha Annette Morris and Kunal Jindal (Zenodo, version 1.1.0,
  doi:10.5281/zenodo.23442375), with its columns unchanged.
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
| Cell calling | "1k PBMCs from a Healthy Donor (v2 chemistry)", raw feature-barcode matrix, 10x Genomics, <https://www.10xgenomics.com/datasets> | CC BY 4.0 |
| Ambient RNA removal | "1k PBMCs from a Healthy Donor (v2 chemistry)", 10x Genomics, <https://www.10xgenomics.com/datasets>. Raw and filtered Cell Ranger matrices | CC BY 4.0 |
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
