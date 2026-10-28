---
myst:
  html_meta:
    description: CyteArc perturbation API for pooled single-cell CRISPR screens, with guide calling from guide counts, guide calls imported from Cell Ranger, perturbation labels that name the genes each cell's guides target, the efficiency of each perturbation on its target gene, the E-distance of each perturbation's cells from the control cells, and the records of the calls, the fitted thresholds, the labels, the efficiency and the E-distances.
---

# Perturbation API reference

Pooled single-cell CRISPR screens, such as Perturb-seq, CROP-seq, ECCITE-seq and 10x CRISPR Guide Capture, give each cell guide RNAs that switch one target gene off or on, and read each cell's guides beside its transcriptome, from a transcript that carries the guide's sequence or from the guide RNA itself ([Replogle et al. 2020](https://doi.org/10.1038/s41587-020-0470-y)). CyteArc imports the guide counts as an assay declared `CRISPR`. Guide calling decides which guides each cell carries, since every cell also holds a few ambient guide molecules, perturbation labels name the genes that those guides target, perturbation efficiency measures how far each perturbation moved the expression of its target gene, and the E-distance how far it moved its cells from the control cells. `cytearc.perturbation` holds the records of the calls, of the thresholds fitted to make them, of the labels, of the efficiency, and of the E-distances.

The steps run on a `DataStore`. {meth}`ds.perturbation.call_guides <cytearc.datastore.namespaces.PerturbationAccessor.call_guides>` calls the guides of each cell from the guide counts, and {meth}`ds.perturbation.import_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls>` saves Cell Ranger's calls from `protospacer_calls_per_cell.csv` instead. Both save a `guide_calls` artifact. {meth}`ds.perturbation.load_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls>` reads the artifact as a {class}`~cytearc.perturbation.GuideCalls` record, and {meth}`ds.plots.guide_calls <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_calls>` plots how many guides each cell carries and how many cells carry each guide. {meth}`ds.perturbation.load_guide_thresholds <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_thresholds>` reads the fits of the fitted methods as a {class}`~cytearc.perturbation.GuideThresholds` record, and {meth}`ds.plots.guide_thresholds <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_thresholds>` draws them. {meth}`ds.perturbation.labels <cytearc.datastore.namespaces.PerturbationAccessor.labels>` labels each cell by the genes that its called guides target and saves a `perturbation_labels` artifact, which {meth}`ds.perturbation.load_labels <cytearc.datastore.namespaces.PerturbationAccessor.load_labels>` reads as a {class}`~cytearc.perturbation.PerturbationLabels` record. {meth}`ds.perturbation.efficiency <cytearc.datastore.namespaces.PerturbationAccessor.efficiency>` compares each target gene's expression in the cells of its perturbation with the control cells and saves a `perturbation_efficiency` table, which {meth}`ds.perturbation.load_efficiency <cytearc.datastore.namespaces.PerturbationAccessor.load_efficiency>` reads as a {class}`~cytearc.perturbation.PerturbationEfficiency` record and {meth}`ds.plots.efficiency <cytearc.datastore.plot_accessor.DataStorePlotAccessor.efficiency>` draws. {meth}`ds.perturbation.edistance <cytearc.datastore.namespaces.PerturbationAccessor.edistance>` measures how far the cells of each perturbation sit from the control cells in a space such as a PCA and saves a `perturbation_edistance` table, which {meth}`ds.perturbation.load_edistance <cytearc.datastore.namespaces.PerturbationAccessor.load_edistance>` reads as a {class}`~cytearc.perturbation.PerturbationEDistance` record and {meth}`ds.plots.edistance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.edistance>` draws.

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

## Label cells by perturbation

Guide calls name guides, but a screen asks what perturbing each gene does: a library holds several guides for each gene, and control guides, such as non-targeting guides, that perturb no gene. `labels` merges the guides of each gene, so the cells of a gene's guides form one group and the cells of control guides another. A cell's label follows from the targets of its called guides:

| Called guides of the cell | Label | `is_control` | `n_targets` |
|---|---|---|---|
| None | `"Negative"` | False | 0 |
| Control guides only | `"control"` | True | 0 |
| Guides of one gene, with or without control guides | The gene, such as `"GENE1"` | False | 1 |
| Guides of two or more genes | `"Multiple"`, or with `policy="combinations"` the genes sorted and joined by `"+"`, such as `"GENE1+GENE2"` | False | 2 or more |

Two guides of one gene give that gene's label, and so does a gene's guide beside a control guide, while the guide calls label both cells `"Multiple"`. The labels keep each cell's label in the calls as `guide_label`, so the two can be compared. Cells that `call_guides(method="top_guide")` labelled `"Negative"` because their top guide fell short stay `"Negative"`; the calls' `total_umi` and `top_fraction` tell cells with too few guide UMIs from cells whose guide UMIs are split among several guides. `cytearc.perturbation` exports the three labels of cells without one gene as `CONTROL`, `NEGATIVE` and `MULTIPLE`.

`controls` names the targets of the control guides, such as `["Non-Targeting"]` for 10x data. CyteArc never guesses the controls: each name must be the target of at least one guide, so a misspelt control raises `ValueError`.

Each guide's target comes from `target_column` of the guide assay's features, `"target_gene_name"` by default, which 10x imports fill, beside `target_gene_id`, from Cell Ranger's feature reference. `targets` maps guide IDs to targets and replaces the column's target of the guides it names; with `target_column=None`, `targets` gives every target. A guide may lack a target as long as no cell calls it. `labels` raises `ValueError` for a target equal to `"control"`, `"Negative"` or `"Multiple"`, which label cells without one gene; with `policy="combinations"`, for a target that holds `"+"`; for a control that no guide has; and for a called guide without a target.

`policy` decides the label of a cell whose guides target several genes. `"single"`, the default, labels such a cell `"Multiple"`, which suits a screen that perturbs one gene per cell. `"combinations"` names the genes, which suits a screen designed to perturb pairs or sets of genes. Under either policy the artifact keeps each cell's full set of genes in `target_codes`, so a high-MOI analysis can follow each gene across all the cells whose guides target it, whatever their labels.

Labelling reads the calls and the guide assay's target column, not the counts. The labels record the calls, the target of each guide, the controls and the policy, so labelling again with the same values returns the saved labels, with `controls` in any order, while another target for any guide makes new labels and `invalidate_cache=True` labels again.

```python
calls = ds.perturbation.call_guides(cells)
labels = ds.perturbation.labels(calls, controls=["Non-Targeting"])
table = ds.perturbation.load_labels(labels)
table.cells["label"].value_counts()
table.cells.query("guide_label == 'Multiple' and label != 'Multiple'")  # merged cells
gene_cells = ds.select_cells(labels, include=["GENE1"])
control_cells = ds.select_cells(labels, include=["control"])
pairs = ds.perturbation.labels(calls, controls=["Non-Targeting"], policy="combinations")
```

`load_labels` returns a {class}`~cytearc.perturbation.PerturbationLabels` record. Its `cells` table holds one row per cell with `label`, `guide_label`, `is_control` and `n_targets`; its `targets` table one row per cell and gene, with no row for a control guide; and its `guides` table one row per guide with the guide's `target` and `is_control`. `controls` holds the controls, sorted, and `policy` the policy. The labels group cells like cluster labels do, so `select_cells` selects the cells of a gene and plots colour cells by label; [Read per-cell values](cell_aligned_kinds) lists the artifact's arrays. The artifact belongs to the guide assay, so a pseudobulk of the labels names the expression assay, as in `ds.compare.compute_bulk(labels, from_assay="RNA")`.

On the 10x A549 screen, 5,867 cells transduced with guides of 45 genes and non-targeting guides, `controls=["Non-Targeting"]` gives these counts under the default policy:

| Calls | One gene | Several genes, `"Multiple"` | `"control"` | `"Negative"` |
|---|---|---|---|---|
| Cell Ranger's, imported | 3,244 | 1,227 | 396 | 1,000 |
| `call_guides` with its defaults | 3,349 | 1,355 | 387 | 776 |

## Measure perturbation efficiency

A perturbation explains a cell's response only if it changed its own target gene, and the cells of a guide that failed look like controls. `efficiency` compares each target gene's expression in the cells that carry its perturbation alone with the gene's expression in the `"control"` cells of the labels, the check that Cell Ranger calls perturbation efficiency. A knockdown lowers the target's expression, and an activation raises it.

`by="target"`, the default, gives one row per target gene, and `by="guide"` one row per guide, so that the guides of a gene can be compared. A cell carries a target alone when its called guides target that gene and no other, with or without control guides, so the cells of a target are those that the labels give the target's label. A cell carries a guide alone when the guide is its one called guide. A cell with two guides of one gene, or with a gene's guide beside a control guide, therefore counts for the gene but for none of its guides. Cells whose guides target several genes, and `"Negative"` cells, stay out of the comparison. Each row holds:

| Column | Meaning |
|---|---|
| `n_cells`, `n_control_cells` | The cells that carry the perturbation alone, and the control cells |
| `mean_umi`, `control_mean_umi` | The mean UMIs of the target gene in each group, as Cell Ranger reports them |
| `mean_expression`, `control_mean_expression` | The mean in each group after scaling each cell's UMIs to the median library size of the compared cells, so that sequencing depth does not change the comparison |
| `remaining_fraction` | `mean_expression` over `control_mean_expression`, the fraction of the control expression that remains: below 1 for a knockdown and above 1 for an activation |
| `log2_fold_change` | The base-2 logarithm of `remaining_fraction`, minus infinity when the perturbed cells hold no UMI of the gene |
| `p_value` | A two-sided Mann-Whitney test of the scaled expression, which asks whether the values of one group tend to rank above those of the other |
| `adjusted_p_value` | The Benjamini-Hochberg adjustment of `p_value` across the rows that have one, which controls the false discovery rate |

A cell's library size is its total UMIs in the expression assay, and the compared cells are the perturbed cells of every row together with the control cells.

Each target is matched to a feature of `from_assay`, `"RNA"` by default. The guide assay's `id_column`, `"target_gene_id"` by default, which 10x imports fill from Cell Ranger's feature reference, gives the feature ID of each guide's target, and a target whose guides give one ID that the expression assay holds is matched to that feature. Any other target is matched to the one feature whose name equals the target; a name that several features share matches none. `id_column=None` matches by name only. `efficiency` reads only the matched genes of the compared cells, feature-major when that reads less.

Each row ends with a status:

- `"tested"`: the target gene was compared.
- `"no_feature"`: the target matches no feature of `from_assay`. The row keeps its cells, and a warning names a few such rows.
- `"no_cells"`: no cell carries the perturbation alone.
- `"no_control_expression"`: the control cells hold no UMI of the gene, so no fraction remains. The row keeps its means and its p-value, which can still show the activation of a gene that the control cells do not express.

The table belongs to the expression assay rather than to the guide assay of the labels. It records the labels, the expression data, the feature matched to each row's target and `by`, so measuring again with the same values returns the saved table, an `id_column` that matches a target to another feature makes a new table, and `invalidate_cache=True` measures again. Labels without a `"control"` cell raise `ValueError`.

```python
labels = ds.perturbation.labels(calls, controls=["Non-Targeting"])
efficiency = ds.perturbation.efficiency(labels)
table = ds.perturbation.load_efficiency(efficiency).table
table["status"].value_counts()
table.query("status == 'tested'").sort_values("log2_fold_change")  # strongest knockdowns first
by_guide = ds.perturbation.efficiency(labels, by="guide")  # one row per guide
ds.plots.efficiency(efficiency)
```

`load_efficiency` returns a {class}`~cytearc.perturbation.PerturbationEfficiency` record. Its `table` holds one row per target gene, sorted by target, or one row per guide of a target gene, in the order of the guide assay; control targets and their guides have no row. Its columns are `guide` for rows by guide, `target`, `feature` (the matched feature ID, None without one), `status` and the columns above. `by` and `from_assay` name the rows and the expression assay.

`ds.plots.efficiency` draws one row per tested target or guide, from the strongest knockdown at the top to the strongest activation at the bottom: a point at the row's log2 fold change against the control cells, coloured by whether its `adjusted_p_value` is below `alpha` (0.05), with an area that grows with its cells. A fold change of minus infinity is a triangle at the left edge, and the row's label says that no UMI remains. When more rows were tested than `max_rows` (40), the plot keeps the `max_rows` rows with the smallest adjusted p-values. Rows of the other statuses are not drawn, and the title counts them. The plot's table, `efficiency`, holds the drawn rows in drawing order.

On the 10x A549 screen, with Cell Ranger's calls imported and `controls=["Non-Targeting"]`, `efficiency` takes 0.8 s. The cells and mean UMIs of all 45 targets and 77 guides equal those in Cell Ranger's `perturbation_efficiencies_by_target.csv` and `perturbation_efficiencies_by_feature.csv`, and the log2 fold changes have a Spearman correlation of 0.97 with Cell Ranger's, which shrinks its estimates. Cell Ranger also lists rows for pairs of targets, from the cells whose guides target both genes, while CyteArc's rows cover cells with one target.

## Measure how far each perturbation moves its cells

Efficiency checks one gene. `edistance` asks whether a perturbation changed its cells at all, over every gene that a coordinate space such as a PCA summarises, with the E-distance of scPerturb ([Peidli et al. 2024](https://doi.org/10.1038/s41592-023-02144-y)). For the cells that carry a perturbation alone and the `"control"` cells of the labels, it is twice the mean squared Euclidean distance between a perturbed and a control cell, minus the mean squared distance between two distinct perturbed cells and between two distinct control cells. Its expected value is twice the squared distance between the means of the two groups, so it grows as a perturbation moves the centre of its cells away from the control cells, while a change in their spread alone leaves it near 0. The estimate subtracts the spread within each group, so a perturbation without effect scores near 0 and can fall below it, far below for a few widely spread cells.

`by` and the cells of each row follow `efficiency`: one row per target gene by default, or one row per guide with `by="guide"`, from the cells that carry the target or the guide alone. `coordinates` can be a reduction, such as a PCA, a batch correction, imported coordinates or normalized values, of any assay. Distances depend on the space, so compare the rows of one table, or of tables measured on the same coordinates. Cells of the labels without coordinates, such as cells left out of the PCA, are left out, and a log message counts them.

A permutation test asks whether a perturbation's cells sit farther from the control cells than a group drawn at random from both. Each of `n_permutations` (1,000) draws takes, from the pooled perturbed and control cells, a group of the perturbed group's size, and compares it with the rest of the pool. The p-value is the share of draws, counting the observed groups as one more draw, whose E-distance reaches the observed one, so the smallest p-value is 1 / (`n_permutations` + 1). A row with few cells has few distinct groups in its pool and cannot reach small p-values: three perturbed cells beside six control cells allow only 84 groups. `adjusted_p_value` adjusts the p-values across rows by Benjamini-Hochberg. Each row draws from a seed derived from `random_seed` (0) and its position, so the same arguments give the same table whatever `nthreads`, and `n_permutations=0` skips the tests.

| Column | Meaning |
|---|---|
| `n_cells`, `n_control_cells` | The cells with coordinates that carry the perturbation alone, and the control cells with coordinates |
| `edistance` | The E-distance of the perturbed cells to the control cells |
| `p_value` | The permutation p-value |
| `adjusted_p_value` | The Benjamini-Hochberg adjustment of `p_value` across the rows that have one |

A row is `"tested"`, or `"too_few_cells"` when fewer than two of its cells have coordinates, so that no distance between two distinct perturbed cells exists; such a row has no E-distance and no p-value.

The table belongs to the assay of the coordinates. It records the labels, the coordinates, `by`, `n_permutations` and `random_seed`, so measuring again with the same values returns the saved table, and `invalidate_cache=True` measures again. Labels without a `"control"` cell, and coordinates of fewer than two control cells, raise `ValueError`. Each E-distance follows from the sums and squared norms of each group's coordinates, so no matrix of pairwise distances is formed, but the coordinates of the compared cells must fit the memory budget.

```python
pca = ds.reduction.pca(normalized, dims=30)  # a PCA of the cells' expression
edistance = ds.perturbation.edistance(labels, pca)
table = ds.perturbation.load_edistance(edistance).table
table.query("status == 'tested'").sort_values("edistance", ascending=False)  # strongest first
by_guide = ds.perturbation.edistance(labels, pca, by="guide")  # one row per guide
ds.plots.edistance(edistance)
```

`load_edistance` returns a {class}`~cytearc.perturbation.PerturbationEDistance` record. Its `table` holds one row per target gene, sorted by target, or one row per guide of a target gene, in the order of the guide assay; control targets and their guides have no row. Its columns are `guide` for rows by guide, `target`, `status` and the columns above. `by`, `n_permutations` and `random_seed` record how the table was measured.

`ds.plots.edistance` draws one row per tested target or guide, from the largest E-distance at the top: a point at the row's E-distance, coloured by whether its `adjusted_p_value` is below `alpha` (0.05), with an area that grows with its cells. Without permutation tests, every point takes the colour of a small p-value and the title says that no test ran. When more rows were tested than `max_rows` (40), the plot keeps the `max_rows` rows with the largest E-distances. Rows of the other status are not drawn, and the title counts them. The plot's table, `edistance`, holds the drawn rows in drawing order.

On the 10x A549 screen, in a 50-component PCA of all 5,867 cells with Cell Ranger's calls labelled by `controls=["Non-Targeting"]`, the E-distances of all 45 targets and 88 guides equal those of scPerturb's `edist` within a relative $2.4 \times 10^{-12}$. The p-values have a Spearman correlation of 0.988 by target and 0.997 by guide with those of scPerturb's `etest`, and adjusted by Benjamini-Hochberg both give the same rows below 0.05 for every target and for 87 of 88 guides. `edistance` takes 0.15 s with 1,000 permutations, where `etest` takes 62 s by target. pertpy's `edistance` is another statistic: it uses Euclidean rather than squared distances and counts each cell with itself, so its values, all positive, differ from CyteArc's and scPerturb's.

## Guide calls

```{eval-rst}
.. autoclass:: cytearc.perturbation.GuideCalls
```

## Guide thresholds

```{eval-rst}
.. autoclass:: cytearc.perturbation.GuideThresholds
```

## Perturbation labels

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationLabels
```

## Perturbation efficiency

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationEfficiency
```

## Perturbation E-distance

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationEDistance
```
