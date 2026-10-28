# Pooled CRISPR screens

Label cells of a pooled CRISPR screen (Perturb-seq, CROP-seq, ECCITE-seq, 10x CRISPR Guide Capture) by the guides they carry, and check those calls before any perturbation analysis. Docs: <https://docs.nygen.io/CyteArc/api/perturbation.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

## When to use

- A store whose guide counts imported as an assay declared `CRISPR`: 10x HDF5, Matrix Market and H5AD imports send "CRISPR Guide Capture" and "CRISPR Direct Capture" features there (`data-access.md`). `call_guides` calls guides from these counts.
- To import Cell Ranger's calls instead, its call file of the same run: `outs/crispr_analysis/protospacer_calls_per_cell.csv` from `cellranger count`, or `outs/per_sample_outs/<sample>/count/crispr_analysis/protospacer_calls_per_cell.csv` from `cellranger multi`.

## Key concepts

- A guide call decides that a cell carries a guide. Every cell also holds a few ambient guide molecules, so a nonzero count is not a call.
- `ds.perturbation.call_guides(cells)` calls guides from the counts, and `ds.perturbation.import_guide_calls(path, cell_selection=cells)` saves Cell Ranger's calls. Both save an assay-scoped `guide_calls` artifact. Its canonical `values` label each cell with its one called guide, `"Negative"` (no call), or `"Multiple"` (several calls); its other arrays are `n_guides`, `top_guide`, `top_umi`, `top_fraction` and `total_umi`, plus the two-dimensional `guide_codes` and `guide_umis`.
- `method=` of `call_guides`, with its default `min_umi`:
  - `"poisson_gaussian"` (default, `min_umi=1`): a Poisson background plus a Gaussian on log2 counts, fitted to each guide over the cells with at least one UMI of the guide (Replogle et al. 2022, as crispat and pertpy fit it).
  - `"gaussian"` (`min_umi=3`): Cell Ranger's two Gaussians with one shared variance on log10(count + 1), fitted over every cell. Reproduces Cell Ranger, but when few cells carry each guide the fit can call cells with ambient UMIs; a warning says so when more than half of the called cells have several guides.
  - `"fixed"`: every guide from `min_umi` UMIs, which it requires.
  - `"top_guide"` (`min_umi=5`, `fraction_threshold=0.8`): each cell's top guide, when it has `min_umi` UMIs and holds more than the fraction of the cell's guide UMIs, as SCEPTRE's maximum rule does. At most one call per cell; a cell whose top guide falls short is `"Negative"`.
- The fitted methods save a `guide_thresholds` artifact that the calls record as an input, and call a guide from the larger of its fitted `threshold` and `min_umi`. A fit whose status is not `"fitted"` (`"no_cells"`, `"too_sparse"`, `"no_fit"`, `"no_positive"`) calls no cell; a warning counts these fits and names a few, again on every reuse.
- `min_umi` and `fraction_threshold` belong to the calls: another value reuses the saved fits and reads the counts once, to call. The same arguments return the saved calls without reading the counts; `invalidate_cache=True` fits and calls again.
- `sample_column=` (fitted methods only) fits each sample, such as each GEM group, separately; every selected cell needs a label. Use it only when each sample holds enough cells that carry each guide.
- `ds.perturbation.load_guide_thresholds(calls)` returns a `GuideThresholds` record: `.table` (one row per sample and guide: `sample`, `guide`, `status`, `threshold`, `call_threshold`, `n_cells`, `n_umi_cells`, `max_umi`, `iterations` and the fitted parameters), `.histograms` (`sample`, `guide`, `umis`, `cells`, zero counts included), `.method`, `.sample_column` and `.min_umi`. `guides=[...]` reads fewer guides. Imported, `"fixed"` and `"top_guide"` calls have no fits and raise `ValueError`.
- Barcodes of an import join cells exactly: a cell's ID is `barcode_prefix` plus the barcode. A merged store names cells `<sample>__<barcode>`, so import each sample's file with `barcode_prefix="<sample>__"` and a selection of that sample's cells. The artifact attributes `n_unmatched_barcodes` and `unmatched_barcode_examples` count the barcodes that matched no selected cell.
- Reuse of an import follows the file's bytes, not its path: the same file under another name returns the saved artifact, and an edited file makes a new one.
- `ds.perturbation.load_guide_calls(calls)` returns a `GuideCalls` record: `.cells` (one row per cell), `.calls` (one row per called guide, most UMIs first), `.guides` (every guide ID) and `.source` (`"call_guides"` or `"import_guide_calls"`).
- `ds.plots.guide_calls(calls)` draws guides per cell, cells per guide, and the top guide's share of each cell's guide UMIs by call.
- `ds.plots.guide_thresholds(calls)` draws each fit's threshold against its cells with a UMI of the guide, and the number of fits of each status. With `guides=[...]` it draws one panel per guide and sample: cells per UMI count, the posterior of the cells that carry the guide, and lines at the threshold and the call threshold. `samples=[...]` keeps some samples; more panels than `max_panels=12` raise `ValueError`. Tables: `thresholds`, `histograms`, and with `guides` also `posterior`.

## Recipe

```python
import warnings

import cytearc

ds = cytearc.DataStore("screen.zarr")
cells = ds.select_measured_cells("CRISPR")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    calls = ds.perturbation.call_guides(cells)  # poisson_gaussian, min_umi=1
for warning in caught:
    print(warning.message)  # fits that call no cell, by status, with examples

fits = ds.perturbation.load_guide_thresholds(calls)
print(fits.table["status"].value_counts())
columns = ["sample", "guide", "status", "threshold", "call_threshold", "n_umi_cells"]
print(fits.table.sort_values("n_umi_cells")[columns].head(20))  # the sparsest fits

plot = ds.plots.guide_thresholds(calls, show=False)  # overview of every fit
plot.save("guide_thresholds.png")
plot.close()
plot = ds.plots.guide_thresholds(calls, guides=["ACE2-1"], show=False)
plot.save("guide_thresholds_ACE2-1.png")
plot.close()

floor = ds.perturbation.call_guides(cells, min_umi=3)  # reuses the fits, calls again
for name, ref in {"min_umi=1": calls, "min_umi=3": floor}.items():
    n_guides = ds.perturbation.load_guide_calls(ref).cells["n_guides"]
    print(name, n_guides.clip(upper=2).value_counts().sort_index().to_dict())  # 0, 1, 2+

plot = ds.plots.guide_calls(calls, show=False)
plot.save("guide_calls.png")
plot.close()

table = ds.perturbation.load_guide_calls(calls)
print(table.cells["call"].value_counts().head(20))
one_guide = ds.select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)
guide_cells = ds.select_cells(calls, include=["ACE2-1"])

# Cell Ranger's calls of the same run, saved as the same artifact kind.
imported = ds.perturbation.import_guide_calls(
    "outs/crispr_analysis/protospacer_calls_per_cell.csv",
    cell_selection=cells,
)
```

## Checks

- Read every warning of `call_guides`. A `"no_cells"` or `"too_sparse"` fit has too few UMIs of the guide to fit; inspect `"no_fit"` and `"no_positive"` fits with `plots.guide_thresholds(calls, guides=[...])` before reporting their guides as absent.
- With `method="gaussian"`, the warning that most called cells have several guides means thresholds among the ambient counts: check `plots.guide_thresholds`, or call with `"poisson_gaussian"`.
- Compare UMI floors on the calls they make: another `min_umi` reuses the fits, so trying one costs a single pass over the counts.
- A `"Negative"` cell of `"top_guide"` holds too few guide UMIs, or several guides with no dominant one, which SCEPTRE flags as holding several guides. Report the two apart, using `top_fraction` and `total_umi`.
- Read the share of cells in each call before going on: a low-MOI screen should have most called cells with one guide. A large `"Multiple"` share points to doublets or to a high-MOI design.
- `"Multiple"` cells whose `top_fraction` is near 1 may carry one guide plus ambient molecules of others. Report how many there are; do not reassign them.
- `"Negative"` cells are not controls. Controls are cells that carry non-targeting guides, named explicitly.
- Check `n_unmatched_barcodes` after each import. Many unmatched barcodes mean a wrong `barcode_prefix`, a call file from another run, or a cell selection that leaves out filtered cells.
- `select_cells` reads one-dimensional arrays only; read `guide_codes` and `guide_umis` with `ds.artifacts.load_values(calls, value="guide_codes")` or from `load_guide_calls(calls).calls`.
