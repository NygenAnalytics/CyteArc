---
myst:
  html_meta:
    description: CyteArc perturbation API for pooled single-cell CRISPR screens, with guide calling from guide counts, guide calls imported from Cell Ranger, and the records of the calls and of the fitted thresholds.
---

# Perturbation API reference

Pooled single-cell CRISPR screens, such as Perturb-seq, CROP-seq, ECCITE-seq and 10x CRISPR Guide Capture, give each cell guide RNAs that switch one target gene off or on, and read each cell's guides beside its transcriptome, from a transcript that carries the guide's sequence or from the guide RNA itself ([Replogle et al. 2020](https://doi.org/10.1038/s41587-020-0470-y)). CyteArc imports the guide counts as an assay declared `CRISPR`. Guide calling decides which guides each cell carries, since every cell also holds a few ambient guide molecules; `cytearc.perturbation` holds the records of the calls and of the thresholds fitted to make them.

The steps run on a `DataStore`. {meth}`ds.perturbation.call_guides <cytearc.datastore.namespaces.PerturbationAccessor.call_guides>` calls the guides of each cell from the guide counts, and {meth}`ds.perturbation.import_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls>` saves Cell Ranger's calls from `protospacer_calls_per_cell.csv` instead. Both save a `guide_calls` artifact. {meth}`ds.perturbation.load_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls>` reads the artifact as a {class}`~cytearc.perturbation.GuideCalls` record, and {meth}`ds.plots.guide_calls <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_calls>` plots how many guides each cell carries and how many cells carry each guide. {meth}`ds.perturbation.load_guide_thresholds <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_thresholds>` reads the fits of the fitted methods as a {class}`~cytearc.perturbation.GuideThresholds` record, and {meth}`ds.plots.guide_thresholds <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_thresholds>` draws them.

A `guide_calls` artifact labels each cell with its one called guide, with `"Negative"` when no guide is called, and with `"Multiple"` when several are. The labels group cells like cluster labels do, so `ds.select_cells(calls, include=["GUIDE-1"])` selects the cells of one guide and plots colour cells by their call; `ds.select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)` keeps the cells with exactly one called guide. [Read per-cell values](cell_aligned_kinds) lists the artifact's arrays.

## Call guides from counts

`call_guides` calls a guide in a cell only when the cell's UMIs of the guide stand out from the ambient molecules. `method` chooses how:

| Method | Model | Cells in the fit | Default `min_umi` | Suits |
|---|---|---|---|---|
| `"poisson_gaussian"`, the default | A Poisson background and a Gaussian on log2 counts ([Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013)), fitted as in crispat ([Braunger and Velten 2024](https://doi.org/10.1093/bioinformatics/btae535)) and pertpy | The cells with at least one UMI of the guide | 1 | Most screens, at low or high multiplicity of infection (MOI) |
| `"gaussian"` | Two Gaussians with one shared variance on log10(count + 1), as Cell Ranger describes | Every selected cell | 3, Cell Ranger's floor | Calls that follow Cell Ranger's |
| `"fixed"` | No model: every guide is called from `min_umi` UMIs | No fit | Required | A floor chosen in advance |
| `"top_guide"` | No model: each cell's top guide is called when it has `min_umi` UMIs and holds more than `fraction_threshold` (default 0.8) of the cell's guide UMIs, as SCEPTRE's maximum rule does ([Barry et al. 2024](https://doi.org/10.1186/s13059-024-03254-2)) | No fit | 5 | Low-MOI screens, where each cell should carry one guide |

The fitted methods fit each guide on its own and set the guide's `threshold` to the smallest UMI count, from 1 to the guide's largest, whose posterior for the cells that carry the guide exceeds one half. A guide is called in every cell whose UMIs of the guide reach the larger of `threshold` and `min_umi`, the `call_threshold` of the `GuideThresholds` table, so a cell can have any number of calls. `top_guide` makes at most one call per cell and labels a cell whose top guide falls short `"Negative"`.

`sample_column` names a cell metadata column, such as the GEM well of each cell, whose samples the fitted methods fit separately; every selected cell needs a label. Without it, each guide gets one fit over every selected cell.

Each fit ends with a status, and a fit whose status is not `"fitted"` calls no cell:

- `"no_cells"`: no cell of the sample holds a UMI of the guide.
- `"too_sparse"`: too few cells or counts to fit; for `poisson_gaussian`, fewer than 2 cells with a UMI of the guide or a largest count below 2, and for `gaussian`, a single distinct count.
- `"no_fit"`: no starting point led to an admissible fit.
- `"no_positive"`: the fit is admissible but calls no count up to the guide's largest.

`call_guides` warns with the number of fits that call no cell, by status, and names a few; the warning comes again when the calls are reused. With `gaussian`, the fit of a guide that few cells carry can split the cells without a UMI of the guide from the cells with ambient UMIs, and call the latter. `call_guides` warns when more than half of the cells with a call have several guides, which points to thresholds set among the ambient counts.

The fitted methods save their fits as a `guide_thresholds` artifact, which the calls record as an input: each fit's status, threshold and parameters, and the histogram of each guide's UMI counts in each sample. `min_umi` and `fraction_threshold` belong to the calls rather than to the fits, so calling again with another value reuses the saved fits and reads the counts once, to call. Calling again with the same arguments returns the saved calls without reading the counts.

```python
cells = ds.select_measured_cells("CRISPR")
calls = ds.perturbation.call_guides(cells)
fits = ds.perturbation.load_guide_thresholds(calls)
fits.table.query("status != 'fitted'")  # the fits that call no cell
ds.plots.guide_thresholds(calls, guides=["GUIDE-1"])
stricter = ds.perturbation.call_guides(cells, min_umi=3)  # reuses the fits
```

`load_guide_thresholds` reads the fits through the calls or from their `guide_thresholds` artifact, for every guide or for the guides that `guides` names. Its `table` holds one row per sample and guide, and its `histograms` the number of cells at each UMI count of each guide in each sample, zero included. Read through the calls, the table's `call_threshold` applies their `min_umi`. `load_guide_thresholds` refuses calls of `fixed` and `top_guide`, which fit no thresholds, and calls imported from Cell Ranger, which writes its thresholds to `crispr_analysis/protospacer_umi_thresholds.csv`.

`ds.plots.guide_thresholds` draws the fits. Without `guides`, it draws an overview: each fit's threshold against the fit's cells with a UMI of the guide, and the number of fits of each status. With `guides`, it draws one panel per guide and sample: the cells at each UMI count, the posterior of the population of cells that carry the guide, and lines at the threshold and at the call threshold. `samples` keeps the panels or points of some samples, and a selection of more than `max_panels` panels raises `ValueError`. Its tables are `thresholds`, `histograms` and, with `guides`, `posterior`.

## Guide calls

```{eval-rst}
.. autoclass:: cytearc.perturbation.GuideCalls
```

## Guide thresholds

```{eval-rst}
.. autoclass:: cytearc.perturbation.GuideThresholds
```
