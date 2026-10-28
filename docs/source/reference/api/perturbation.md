---
myst:
  html_meta:
    description: CyteArc perturbation API for pooled single-cell CRISPR screens, starting with guide calls imported from Cell Ranger and the record that reads them.
---

# Perturbation API reference

Pooled single-cell CRISPR screens, such as Perturb-seq, CROP-seq, ECCITE-seq and 10x CRISPR Guide Capture, give each cell guide RNAs that switch one target gene off or on, and capture the guides beside the transcriptome. CyteArc imports the guide counts as an assay declared `CRISPR`. Guide calling decides which guides each cell carries, since every cell also holds a few ambient guide molecules; `cytearc.perturbation` holds the record of those calls.

The steps run on a `DataStore`. {meth}`ds.perturbation.import_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls>` saves Cell Ranger's calls from `protospacer_calls_per_cell.csv` as a `guide_calls` artifact, {meth}`ds.perturbation.load_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls>` reads that artifact as a {class}`~cytearc.perturbation.GuideCalls` record, and {meth}`ds.plots.guide_calls <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_calls>` plots how many guides each cell carries and how many cells carry each guide.

A `guide_calls` artifact labels each cell with its one called guide, with `"Negative"` when no guide is called, and with `"Multiple"` when several are. The labels group cells like cluster labels do, so `ds.select_cells(calls, include=["GUIDE-1"])` selects the cells of one guide and plots colour cells by their call; `ds.select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)` keeps the cells with exactly one called guide. [Read per-cell values](cell_aligned_kinds) lists the artifact's arrays.

## Guide calls

```{eval-rst}
.. autoclass:: cytearc.perturbation.GuideCalls
```
