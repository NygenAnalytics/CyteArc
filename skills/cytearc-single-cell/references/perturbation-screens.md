# Pooled CRISPR screens

Label cells of a pooled CRISPR screen (Perturb-seq, CROP-seq, ECCITE-seq, 10x CRISPR Guide Capture) by the guides they carry, check those calls, label each cell by the genes its guides target, check that each perturbation changed its own target gene, and rank perturbations by how far they move their cells from the control cells. Docs: <https://docs.nygen.io/CyteArc/api/perturbation.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

## When to use

- A store whose guide counts imported as an assay declared `CRISPR`: 10x HDF5, Matrix Market and H5AD imports send "CRISPR Guide Capture" and "CRISPR Direct Capture" features there (`data-access.md`). `call_guides` calls guides from these counts.
- To import Cell Ranger's calls instead, its call file of the same run: `outs/crispr_analysis/protospacer_calls_per_cell.csv` from `cellranger count`, or `outs/per_sample_outs/<sample>/count/crispr_analysis/protospacer_calls_per_cell.csv` from `cellranger multi`.
- To label cells by perturbation, each guide's target: the guide assay's `target_gene_name` column, which 10x imports fill from Cell Ranger's feature reference, or a mapping of guide IDs to targets.
- To measure perturbation efficiency, an expression assay of the same cells and a way to find each target gene in it: the guide assay's `target_gene_id` column, which 10x imports fill with the gene's feature ID, or targets named like the expression assay's features.
- To measure E-distances, coordinates of the labelled cells: a PCA, a batch correction, imported coordinates or normalized values, fitted on perturbed and control cells alike.

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
- `ds.perturbation.labels(calls, controls=["Non-Targeting"])` labels each cell of called or imported calls by the genes its called guides target, as an assay-scoped `perturbation_labels` artifact:
  - `"Negative"`: no called guide.
  - `"control"`: control guides only.
  - the gene: guides of one gene, with or without control guides; two guides of one gene give the gene.
  - `"Multiple"`: guides of two or more genes; with `policy="combinations"`, the genes sorted and joined by `"+"`, such as `"GENE1+GENE2"`.
- `controls=` is required: the exact targets of the control guides, never guessed. A control that no guide has raises `ValueError`; the order of `controls` does not change the saved labels.
- Each guide's target comes from `target_column=` of the guide assay's features (default `"target_gene_name"`). `targets={guide_id: target}` replaces the column's target of the guides it names, and `target_column=None` takes every target from `targets`. A guide may lack a target unless a cell calls it.
- Refused with `ValueError`: a target equal to `"control"`, `"Negative"` or `"Multiple"`; under `"combinations"`, a target holding `"+"`; a control that no guide has; a called guide without a target (`"N called guides have no target"`).
- Arrays: canonical `values`, `guide_label` (the cell's label in the calls), `is_control`, `n_targets`, and the two-dimensional `target_codes`, which keeps each cell's full set of genes under either policy. `targets`, `target_is_control` and `guide_target_codes` are not cell-aligned. `select_cells` reads the one-dimensional arrays.
- Reuse follows the calls, each guide's resolved target, the controls and the policy: another target for any guide makes new labels, and `invalidate_cache=True` labels again. Labelling reads no counts.
- `"Negative"` cells of `"top_guide"` calls stay `"Negative"`; the calls' `total_umi` and `top_fraction` tell cells with too few guide UMIs from cells whose guide UMIs are split among several guides.
- `ds.perturbation.load_labels(labels)` returns a `PerturbationLabels` record: `.cells` (indexed by `ids`: `label`, `guide_label`, `is_control`, `n_targets`), `.targets` (`ids`, `target`; one row per cell and gene, none for control guides), `.guides` (`guide`, `target`, `is_control`), `.controls` and `.policy`.
- The labels belong to the guide assay: `ds.compare.compute_bulk(labels, from_assay="RNA", ...)` pools expression, and without `from_assay` it pools the guide counts.
- `ds.perturbation.efficiency(labels)` checks that each perturbation changed its own target gene: the target gene's expression in the cells that carry the perturbation alone against the `"control"` cells of the labels. `by="target"` (default) gives one row per target gene; `by="guide"` one row per guide. A cell carries a target alone when its guides target that gene and no other, with or without control guides, and a guide alone when the guide is its one called guide. `"Multiple"`, combination and `"Negative"` cells stay out. Saved as a `perturbation_efficiency` table scoped to `from_assay` (default `"RNA"`), not cell-aligned.
- `ds.perturbation.load_efficiency(efficiency)` returns a `PerturbationEfficiency` record: `.table`, `.by`, `.from_assay`. Table columns, one row per target (sorted) or guide (assay order), controls left out:
  - `guide` (by guide only), `target`, `feature` (the matched feature ID, None without one), `status`.
  - `n_cells` (cells with the perturbation alone), `n_control_cells`.
  - `mean_umi`, `control_mean_umi`: raw mean UMIs of the gene in each group, as Cell Ranger reports them.
  - `mean_expression`, `control_mean_expression`: means after scaling each cell to the median library size of the compared cells.
  - `remaining_fraction`: `mean_expression / control_mean_expression`; below 1 for a knockdown, above 1 for an activation. `log2_fold_change`: its log2, minus infinity when the perturbed cells hold no UMI of the gene.
  - `p_value`: two-sided Mann-Whitney on the scaled expression; `adjusted_p_value`: Benjamini-Hochberg across the rows with a p-value.
- Each target matches the expression feature whose ID its guides give in `id_column=` (default `"target_gene_id"`), otherwise the one feature whose name equals the target; a name shared by several features matches none. `id_column=None` matches by name only. Only the matched genes of the compared cells are read.
- `status`: `"tested"`; `"no_feature"` (no feature matched; the row keeps its cells, and a warning `"N of M rows have no feature of assay 'RNA' ..."` names a few); `"no_cells"` (no cell carries the perturbation alone); `"no_control_expression"` (the controls hold no UMI of the gene: no fraction or fold change, but the means and p-value are kept).
- Labels without a `"control"` cell raise `ValueError`. Reuse follows the labels, the expression data, each row's matched feature and `by`; `invalidate_cache=True` measures again.
- `ds.plots.efficiency(efficiency)` draws each `"tested"` row's log2 fold change against the control cells, strongest knockdown at the top, coloured by `adjusted_p_value < alpha` (0.05) and sized by its cells; with more than `max_rows=40` tested rows it keeps those with the smallest adjusted p-values. Minus infinity is a triangle at the left edge, labelled "no UMI remains". Rows of other statuses are not drawn; the title counts them. Table: `efficiency` (the drawn rows, every column, in drawing order).
- `ds.perturbation.edistance(labels, coordinates)` measures how far the cells of each perturbation sit from the `"control"` cells in a coordinate space (scPerturb's E-distance: twice the mean squared distance between groups minus the mean squared distance between distinct cells within each group). Its expected value is twice the squared distance between the group means: it detects a shift of the cells' centre, not a change of spread alone. Near 0, and possibly below, without an effect; strongly negative for a few widely spread cells. Rows and cells follow `efficiency`: `by="target"` (default) or `by="guide"`, single-target or single-guide cells only. `coordinates`: a `reduction`, `batch_correction`, `imported_coordinates` or `normalized` artifact of any assay; labelled cells without coordinates are left out and counted in a log message. Saved as a `perturbation_edistance` table scoped to the coordinates' assay, not cell-aligned.
- Permutation test: `n_permutations=1000` draws of a group of the perturbed group's size from the pooled perturbed and control cells; `p_value = (draws reaching the observed E-distance + 1) / (n_permutations + 1)`, so its floor is about 0.001 at the default; `adjusted_p_value` is Benjamini-Hochberg across rows. `random_seed=0` seeds each row with the row, so results do not depend on `nthreads`. `n_permutations=0` skips the tests (NaN p-values).
- `ds.perturbation.load_edistance(edistance)` returns a `PerturbationEDistance` record: `.table`, `.by`, `.n_permutations`, `.random_seed`. Table columns, one row per target (sorted) or guide (assay order), controls left out: `guide` (by guide only), `target`, `status` (`"tested"` or `"too_few_cells"`, fewer than two perturbed cells with coordinates, which has no distance or p-value), `n_cells`, `n_control_cells`, `edistance`, `p_value`, `adjusted_p_value`.
- Labels without a `"control"` cell, or fewer than two control cells with coordinates, raise `ValueError`. Reuse follows the labels, the coordinates, `by`, `n_permutations` and `random_seed`; `invalidate_cache=True` measures again.
- `ds.plots.edistance(edistance)` draws each `"tested"` row's E-distance, largest at the top, coloured by `adjusted_p_value < alpha` (0.05) and sized by its cells; with more than `max_rows=40` tested rows it keeps those with the largest E-distances. Rows of other statuses are not drawn; the title counts them. Table: `edistance` (the drawn rows, every column, in drawing order).

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

# Name the controls from the guide assay's target column, then label cells by gene.
guides = ds.CRISPR.feats.to_pandas_dataframe(["ids", "target_gene_name"])
print(guides["target_gene_name"].value_counts())  # the control target, such as "Non-Targeting"
labels = ds.perturbation.labels(calls, controls=["Non-Targeting"])
labelled = ds.perturbation.load_labels(labels)
counts = labelled.cells["label"].value_counts()
reserved = ["control", "Negative", "Multiple"]
genes = counts.drop(reserved, errors="ignore")
print(counts.reindex(reserved, fill_value=0).to_dict(), "one gene:", int(genes.sum()))
print(genes.describe())  # cells per gene
merged = labelled.cells.query("guide_label == 'Multiple' and label != 'Multiple'")
print(len(merged), "cells with several called guides but one gene, or controls only")
gene_cells = ds.select_cells(labels, include=["ACE2"])
control_cells = ds.select_cells(labels, include=["control"])

# Check that each perturbation changed its own target gene in the expression assay.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    efficiency = ds.perturbation.efficiency(labels)  # from_assay="RNA", by="target"
for warning in caught:
    print(warning.message)  # targets that match no feature of the expression assay
measured = ds.perturbation.load_efficiency(efficiency).table
print(measured["status"].value_counts())
columns = ["target", "n_cells", "control_mean_umi", "remaining_fraction", "adjusted_p_value"]
print(measured.query("status == 'tested'").sort_values("remaining_fraction")[columns])
print(measured.query("status != 'tested'")[["target", "feature", "status", "n_cells"]])
by_guide = ds.perturbation.efficiency(labels, by="guide")
per_guide = ds.perturbation.load_efficiency(by_guide).table
guide_columns = ["target", "guide", "n_cells", "remaining_fraction", "adjusted_p_value"]
print(per_guide.query("status == 'tested'").sort_values(["target", "guide"])[guide_columns])
plot = ds.plots.efficiency(efficiency, show=False)
plot.save("efficiency.png")
plot.close()

# Rank perturbations by how far they move their cells, in a PCA of all the cells.
# `pca` is the PCA of the expression assay, such as the one of a ds.pipeline.rna run.
edistance = ds.perturbation.edistance(labels, pca)  # by="target", n_permutations=1000
distances = ds.perturbation.load_edistance(edistance).table
print(distances["status"].value_counts())
distance_columns = ["target", "n_cells", "edistance", "adjusted_p_value"]
print(distances.query("status == 'tested'").sort_values("edistance", ascending=False)[distance_columns])
plot = ds.plots.edistance(edistance, show=False)
plot.save("edistance.png")
plot.close()

# Cell Ranger's calls of the same run, saved as the same artifact kind and labelled alike.
imported = ds.perturbation.import_guide_calls(
    "outs/crispr_analysis/protospacer_calls_per_cell.csv",
    cell_selection=cells,
)
imported_labels = ds.perturbation.labels(imported, controls=["Non-Targeting"])
```

## Checks

- Read every warning of `call_guides`. A `"no_cells"` or `"too_sparse"` fit has too few UMIs of the guide to fit; inspect `"no_fit"` and `"no_positive"` fits with `plots.guide_thresholds(calls, guides=[...])` before reporting their guides as absent.
- With `method="gaussian"`, the warning that most called cells have several guides means thresholds among the ambient counts: check `plots.guide_thresholds`, or call with `"poisson_gaussian"`.
- Compare UMI floors on the calls they make: another `min_umi` reuses the fits, so trying one costs a single pass over the counts.
- A `"Negative"` cell of `"top_guide"` holds too few guide UMIs, or several guides with no dominant one, which SCEPTRE flags as holding several guides. Report the two apart, using `top_fraction` and `total_umi`.
- Read the share of cells in each call before going on: a low-MOI screen should have most called cells with one guide. A large `"Multiple"` share points to doublets or to a high-MOI design.
- `"Multiple"` cells whose `top_fraction` is near 1 may carry one guide plus ambient molecules of others. Report how many there are; do not reassign them.
- `"Negative"` cells are not controls. Controls are cells that carry control guides only, which `labels` names `"control"` from the targets given as `controls=`.
- Check `n_unmatched_barcodes` after each import. Many unmatched barcodes mean a wrong `barcode_prefix`, a call file from another run, or a cell selection that leaves out filtered cells.
- `select_cells` reads one-dimensional arrays only; read `guide_codes` and `guide_umis` with `ds.artifacts.load_values(calls, value="guide_codes")` or from `load_guide_calls(calls).calls`.
- Read the label counts before comparing genes: `"control"`, `"Negative"`, `"Multiple"` and the cells of each gene. Report genes with few cells instead of pooling them with others.
- Count the cells that the merge moved: a `guide_label` of `"Multiple"` beside a gene's label means several guides of that gene, or a gene's guides and control guides. Dual-guide constructs and higher MOI move the most cells.
- A `ValueError` that names a control means no guide has that target: copy the spelling from the target column (`"Non-Targeting"`, not `"Non-targeting"`). `"N called guides have no target"` means the column misses called guides; pass `targets={guide: target}`.
- Compare each gene with the `"control"` cells. Keep `"Multiple"` and `"Negative"` out of per-gene comparisons (`null_vals=["Negative", "Multiple"]` in `compare.compute_bulk`), and under `"combinations"` the combination labels too.
- In a high-MOI screen, analyse each gene across the cells whose guides target it, from `load_labels(labels).targets` or `target_codes`, rather than by combination labels, which hold few cells each.
- Check each perturbation's own target before reading its effects on other genes. CRISPR interference expects `remaining_fraction` below 1 and CRISPR activation above 1. A Cas9 knockout lowers its target's transcripts mainly through nonsense-mediated decay, so a fraction near 1 in a knockout screen does not show a failed edit. Report targets that did not move as unconfirmed; do not drop them.
- Compare by `remaining_fraction`, not by `mean_umi` against `control_mean_umi`, which also follow each group's sequencing depth.
- Weigh each row by `n_cells` and `control_mean_umi`: few cells or a lowly expressed gene give a noisy fraction and a weak test. Minus infinity means no UMI of the gene in the perturbed cells, which can happen by chance in few cells of a lowly expressed gene.
- Fix every `"no_feature"` row before reporting: compare the target with the guide assay's `target_gene_id` and the expression assay's feature IDs and names, then write the right feature IDs as a guide-assay column (`ds.CRISPR.feats.insert("gene_id", ids)`) and pass `id_column="gene_id"`, or relabel with `targets=` so the target equals a feature name. An enhancer or other non-coding target stays `"no_feature"`; test its nearby genes instead.
- A `"no_control_expression"` row has no fraction: the controls hold no UMI of the gene, so a knockdown cannot be measured. In an activation screen, `mean_umi` above 0 with a small `adjusted_p_value` shows the gene switched on.
- Compare the guides of a gene with `by="guide"`: a guide whose fraction stays near 1 while another's falls has likely failed, and its cells dilute the gene's label. A guide row counts only cells whose one called guide it is, so a dual-guide screen (two guides of one gene per cell) has few such cells; measure it by target.
- Efficiency covers cells with one target only. Cell Ranger's `perturbation_efficiencies_by_target.csv` also lists rows for pairs of targets, which CyteArc's table does not.
- Measure E-distances in a space fitted on perturbed and control cells alike, such as the PCA of a standard run. When perturbations spread unevenly across runs or lanes, use coordinates corrected for the runs (`batch_correction`). Compare E-distances within one table or between tables on the same coordinates only; they depend on the dimensions and scale of the space.
- Read each E-distance with `n_cells` and `adjusted_p_value`. When many rows sit at the p-value floor of 1 / (`n_permutations` + 1), raise `n_permutations`. A small E-distance does not show that a perturbation had no effect; a large one whose target did not move in `efficiency` calls for comparing its guides with `by="guide"`.
- Check the log message of `edistance` for labelled cells without coordinates: coordinates fitted on a subset of cells leave the rest out.
