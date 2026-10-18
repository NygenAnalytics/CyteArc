# Third-party notices

CyteArc is copyright Nygen Analytics AB and is licensed under the BSD 3-Clause License in the
`LICENSE` file. Parts of CyteArc derive from the third-party software listed below, and the
repository includes a font and small excerpts of public data. Each keeps its own license, and the
full license texts are at the end of this file. The dataset sections credit the data that the
documentation and tests download, and a later section lists optional packages that CyteArc can
use but does not install.

## Software

### Scanpy

- Source: <https://github.com/scverse/scanpy>
- Copyright (c) 2025 scverse®; Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
- License: BSD-3-Clause
- Used in: `cytearc/features/scoring.py` (gene-set scoring adapted from `score_genes`),
  `cytearc/datastore/_operations/quality_control.py` (the control-set size of cell-cycle
  scoring), and the categorical palettes in `cytearc/plotting/_style.py`.

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
  `FindModalityWeights`), and the reference equations in `tests/test_knn_utils.py`.

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
  `scipy.stats.linregress`).

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

## Fonts

### Inter

- Source: <https://github.com/rsms/inter> (version 4.1, unmodified)
- Copyright (c) 2016 The Inter Project Authors (https://github.com/rsms/inter)
- License: SIL Open Font License 1.1
- Used in: `cytearc/agent/report_assets.py`. Every HTML report that CyteArc generates embeds the
  font and keeps its license in a source comment.

## Data in this repository

- **JASPAR** (<https://jaspar.elixir.no>), CC BY 4.0: the motif matrices MA0004.1 (Arnt) and
  MA0006.1 (Ahr::Arnt), and an excerpt of the MEME export of MA0004.1, in
  `tests/test_genomic_motifs.py`. Rauluseviciute et al. (2024), Nucleic Acids Research,
  doi:10.1093/nar/gkad1059.
- **10x Genomics**, CC BY 4.0: `tests/seurat_wnn_5_5_1_golden.json` holds Seurat 5.5.1 outputs on
  principal components of 300 cells of the `tenx_8K_pbmc_citeseq` dataset, and the reference
  images in `tests/visual/showcase/` are rendered from the 1k PBMC CITE-seq test fixture. Both
  datasets are credited below.
- **GEO GSE245108**: hashtag counts in `tests/seurat_hto_5_5_1_golden.json`, from Zhang et al.
  (2024), Nature Immunology, doi:10.1038/s41590-024-01782-4.
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
| CyteBase datasets and CyteBase composite datasets | Datasets from CZ CELLxGENE Discover, each credited in its CyteBase catalog entry | The license of each dataset on CELLxGENE Discover, usually CC BY 4.0 |

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

### Inter

```text
Copyright (c) 2016 The Inter Project Authors (https://github.com/rsms/inter)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
http://scripts.sil.org/OFL

-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION AND CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
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
