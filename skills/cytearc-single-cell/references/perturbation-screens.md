# Pooled CRISPR screens

Label cells of a pooled CRISPR screen (Perturb-seq, CROP-seq, ECCITE-seq, 10x CRISPR Guide Capture) by the guides they carry, and check those calls before any perturbation analysis. Docs: <https://docs.nygen.io/CyteArc/api/perturbation.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

## When to use

- A store whose guide counts imported as an assay declared `CRISPR`: 10x HDF5, Matrix Market and H5AD imports send "CRISPR Guide Capture" and "CRISPR Direct Capture" features there (`data-access.md`).
- Cell Ranger's call file of the same run: `outs/crispr_analysis/protospacer_calls_per_cell.csv` from `cellranger count`, or `outs/per_sample_outs/<sample>/count/crispr_analysis/protospacer_calls_per_cell.csv` from `cellranger multi`.

## Key concepts

- A guide call decides that a cell carries a guide. Every cell also holds a few ambient guide molecules, so a nonzero count is not a call.
- `ds.perturbation.import_guide_calls(path, cell_selection=cells)` saves Cell Ranger's calls as an assay-scoped `guide_calls` artifact. Its canonical `values` label each cell with its one called guide, `"Negative"` (no call), or `"Multiple"` (several calls); its other arrays are `n_guides`, `top_guide`, `top_umi`, `top_fraction` and `total_umi`, plus the two-dimensional `guide_codes` and `guide_umis`.
- Barcodes join cells exactly: a cell's ID is `barcode_prefix` plus the barcode. A merged store names cells `<sample>__<barcode>`, so import each sample's file with `barcode_prefix="<sample>__"` and a selection of that sample's cells. The artifact attributes `n_unmatched_barcodes` and `unmatched_barcode_examples` count the barcodes that matched no selected cell.
- Reuse follows the file's bytes, not its path: the same file under another name returns the saved artifact, and an edited file makes a new one.
- `ds.perturbation.load_guide_calls(calls)` returns a `GuideCalls` record: `.cells` (one row per cell), `.calls` (one row per called guide, most UMIs first), and `.guides` (every guide ID).
- `ds.plots.guide_calls(calls)` draws guides per cell, cells per guide, and the top guide's share of each cell's guide UMIs by call.

## Recipe

```python
import cytearc

ds = cytearc.DataStore("screen.zarr")
cells = ds.select_measured_cells("CRISPR")
calls = ds.perturbation.import_guide_calls(
    "outs/crispr_analysis/protospacer_calls_per_cell.csv",
    cell_selection=cells,
)
plot = ds.plots.guide_calls(calls, show=False)
plot.save("guide_calls.png")
plot.close()

table = ds.perturbation.load_guide_calls(calls)
print(table.cells["call"].value_counts().head(20))
one_guide = ds.select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)
guide_cells = ds.select_cells(calls, include=["ACE2-1"])
```

## Checks

- Read the share of cells in each call before going on: a low-MOI screen should have most called cells with one guide. A large `"Multiple"` share points to doublets or to a high-MOI design.
- `"Multiple"` cells whose `top_fraction` is near 1 may carry one guide plus ambient molecules of others. Report how many there are; do not reassign them.
- `"Negative"` cells are not controls. Controls are cells that carry non-targeting guides, named explicitly.
- Check `n_unmatched_barcodes` after each import. Many unmatched barcodes mean a wrong `barcode_prefix`, a call file from another run, or a cell selection that leaves out filtered cells.
- `select_cells` reads one-dimensional arrays only; read `guide_codes` and `guide_umis` with `ds.artifacts.load_values(calls, value="guide_codes")` or from `load_guide_calls(calls).calls`.
