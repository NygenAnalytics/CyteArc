# Differential abundance

`cytearc.abundance` tests whether conditions change how cells are distributed, with samples as the
replicates. It has two methods:
- **propeller** (Phipson et al. 2022) tests the proportions of annotated cell categories.
- **Milo** (Dann et al. 2022) tests overlapping kNN neighbourhoods and needs no annotation.

`DataStore.abundance` runs both on a store, keeps the results as artifacts, and `ds.plots` draws
them. Running Milo's test needs the optional `milo` extra (edgePython, GPL-3.0). CyteArc does not
distribute edgePython.

## Modules

| Module | Responsibility |
| --- | --- |
| `abundance/propeller.py` | Category counts per sample, transformed proportions, and the propeller test |
| `abundance/milo.py` | The storage-free Milo steps: index-cell sampling and refinement, membership blocks, counting, k-distances, the edgePython test, spatial FDR, and annotation |
| `abundance/results.py` | The result records the `DataStore` returns, and their tables of counts and proportions |
| `stats/limma.py` | Linear models, contrasts, and empirical Bayes moderated t- and F-statistics |
| `stats/multiple_testing.py` | Benjamini-Hochberg and weighted Benjamini-Hochberg FDR |
| `stats/design.py` | Sample-level treatment-coded design matrices, R-style factor levels, and contrast weights |
| `datastore/_operations/abundance.py` | `DataStore.abundance`: design validation, streaming over the stored graph in row bands, and artifacts |
| `datastore/_operations/abundance_store.py` | The Zarr layouts of the four result kinds |
| `plotting/abundance.py` | `category_proportions`, `neighborhood_beeswarm` (with density-scaled van der Corput jitter), and `neighborhood_graph` |

## Method

**propeller.**
1. Counts cells per category and sample, and turns the counts into proportions.
2. Transforms the proportions with a logit or an arcsine square root.
3. Fits a linear model per category on the sample-level design.
4. Moderates the variances with an empirical Bayes prior (Smyth 2004). The prior is robust to
   outlier categories (Phipson et al. 2016) unless there are two categories or fewer.
5. Tests the result:
   - two groups by a moderated t-test;
   - more groups by a moderated F-test.
6. Corrects the p-values with Benjamini-Hochberg.

**Milo.**
1. Draws a share of cells and refines each to the cell nearest the median of its neighbours.
2. Takes each refined cell's neighbours on the undirected kNN graph as its neighbourhood.
3. Counts each neighbourhood's cells per sample, streaming over blocks of the stored graph.
4. Tests the counts with edgeR's quasi-likelihood F-test, run by edgePython, with TMM, RLE, or
   logMS normalization.
5. Corrects the p-values with a spatial FDR (Lun et al. 2017): weighted Benjamini-Hochberg, where
   each neighbourhood is weighted by the inverse of its distance to its k-th neighbour.
6. Labels each neighbourhood with the most common label among its cells.

Where the papers leave a choice open, the implementation makes these choices:
- **Random draw.** Candidate index cells are drawn with NumPy's generator. The same seed therefore
  draws different cells than miloR.
- **Spatial FDR.** It is weighted over the tested neighbourhoods only. miloR also weights the
  neighbourhoods that `min.mean` leaves untested.
- **Designs.** Designs are additive, with treatment coding, and contrasts are weights over design
  columns. Text levels are ordered by code point, R's order in the C locale; pass `levels` where
  another locale would order mixed-case labels differently.

Docstrings give the details.

## How it was written

limma, speckle, and miloR are licensed under the GPL, so CyteArc may not carry code derived from
them. Every module in the table above except `abundance_store.py` is a clean-room implementation,
written in two rounds.

The statistics: `stats/limma.py`, `stats/multiple_testing.py`, `abundance/propeller.py`, and
`abundance/milo.py`.
- An earlier version of the four modules, and a module of ported R numerics, were deleted before
  the rewrite.
- A separate agent wrote the new modules from the papers listed below and the R help pages.
- It never saw the deleted files, the repository history, the source of any GPL package, or any
  other port.
- Where the papers are silent, it settled details by running R as a black box on constructed
  inputs and comparing outputs.
- edgePython was used only through its public API.
- The modules depend only on NumPy, SciPy, and pandas.

The orchestration and presentation: `stats/design.py`, `abundance/results.py`,
`datastore/_operations/abundance.py`, and `plotting/abundance.py`.
- Their earlier versions had been written alongside a GPL-derived port, so they were deleted, with
  every compiled cache, before the rewrite, and a separate agent wrote new modules with the same
  public interface.
- It never saw the deleted files, the repository history, the source of any GPL package or port,
  or earlier work on this branch outside the repository.
- It worked from the papers, the R help pages of `model.matrix`, `factor`, `testNhoods`,
  `annotateNhoods`, `plotDAbeeswarm`, and `plotNhoodGraphDA`, the rest of this repository (the
  clean-room statistics, `abundance_store.py`, the tests, and the documentation), and R run as a
  black box.
- Level order and column names of designs were settled by calling `factor` and `model.matrix` on
  constructed factors, including numeric, logical, and non-syntactic ones, and comparing outputs.
  Text levels follow R's order in the C locale.
- Tie-breaking in neighbourhood annotation was settled by calling `annotateNhoods` on constructed
  neighbourhoods: the label seen first along the cells wins.
- The plots follow what the help pages and the Milo paper say they show, and an image of miloR's
  beeswarm rendered on synthetic results; their drawing code is CyteArc's own, on Matplotlib.

## How it was verified

The tests compare against outputs generated by running R 4.5.2, not against a transcription of R
code.
- **propeller.** `tests/test_propeller_speckle_golden.py` compares the full output with
  speckle 1.10.0 (limma 3.66.0) on eight cases: logit and arcsine transforms, robust and non-robust
  priors, a variance trend, a three-group ANOVA, and zero counts. Statistics, p-values, FDR, and the
  prior agree to within 1.2e-13 relative.
- **Milo.** `tests/test_milo_golden.py` compares each stage with miloR 2.6.0 on four simulated
  cases, and also runs two of them end to end through `DataStore.abundance`.
  - Sampling, membership, counts, k-distances, and annotation match exactly.
  - p-values and spatial FDR agree within 1.5e-6 relative, and logFC within 5.3e-4.
  - The neighbourhoods called at spatial FDR below 0.1 are identical.
- **Real data.** The opt-in `tests/test_abundance_live_wilk.py` runs miloR on CyteArc's own graph
  and index cells from the Wilk et al. (2020) COVID-19 PBMC atlas (44,721 cells, 7 against 6
  donors).
  - CyteArc calls the same 931 of 1,877 neighbourhoods.
  - p-values agree within 1.1e-6 relative.
  - propeller agrees with speckle within 1e-8.
- **Contracts.** `tests/test_abundance_datastore.py` and `tests/test_plotting_abundance.py` cover the
  `DataStore` API, the stored results, and the plots.

## References

- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach
  to multiple testing. J R Stat Soc B 57, 289-300 (1995). doi:10.1111/j.2517-6161.1995.tb02031.x
- Cleveland WS. Robust locally weighted regression and smoothing scatterplots. J Am Stat Assoc 74,
  829-836 (1979). doi:10.1080/01621459.1979.10481038
- Dann E, Henderson NC, Teichmann SA, Morgan MD, Marioni JC. Differential abundance testing on
  single-cell data using k-nearest neighbor graphs. Nat Biotechnol 40, 245-253 (2022).
  doi:10.1038/s41587-021-01033-z
- Lun ATL, Richard AC, Marioni JC. Testing for differential abundance in mass cytometry data. Nat
  Methods 14, 707-709 (2017). doi:10.1038/nmeth.4295
- Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK. Robust hyperparameter estimation protects
  against hypervariable genes and improves power to detect differential expression. Ann Appl Stat
  10, 946-963 (2016). doi:10.1214/16-AOAS920
- Phipson B, Sim CB, Porrello ER, Hewitt AW, Powell J, Oshlack A. propeller: testing for
  differences in cell type proportions in single cell data. Bioinformatics 38, 4720-4726 (2022).
  doi:10.1093/bioinformatics/btac582
- Smyth GK. Linear models and empirical Bayes methods for assessing differential expression in
  microarray experiments. Stat Appl Genet Mol Biol 3, Article 3 (2004). doi:10.2202/1544-6115.1027
- Wilk AJ, et al. A single-cell atlas of the peripheral immune response in patients with severe
  COVID-19. Nat Med 26, 1070-1076 (2020). doi:10.1038/s41591-020-0944-y
