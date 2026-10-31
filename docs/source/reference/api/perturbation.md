---
myst:
  html_meta:
    description: CyteArc perturbation API for pooled single-cell CRISPR screens, with guide calling from guide counts, guide calls imported from Cell Ranger, perturbation labels that name the genes each cell's guides target, the efficiency of each perturbation on its target gene, the E-distance of each perturbation's cells from the control cells, Mixscape's signatures and calls of knocked-out cells and escapers, SCEPTRE-style tests of whether each perturbation changes each gene with calibration and positive-control checks, each perturbation's mean expression against the control cells and HDBSCAN clusters of the perturbations whose profiles agree, and the records of the calls, the fitted thresholds, the labels, the efficiency, the E-distances, the signatures, the Mixscape classes, the association tests, the profiles and the clusters.
---

# Perturbation API reference

Pooled single-cell CRISPR screens, such as Perturb-seq, CROP-seq, ECCITE-seq and 10x CRISPR Guide Capture, give each cell guide RNAs that switch one target gene off or on, and read each cell's guides beside its transcriptome, from a transcript that carries the guide's sequence or from the guide RNA itself ([Replogle et al. 2020](https://doi.org/10.1038/s41587-020-0470-y)). CyteArc imports the guide counts as an assay declared `CRISPR`. Guide calling decides which guides each cell carries, since every cell also holds a few ambient guide molecules, perturbation labels name the genes that those guides target, perturbation efficiency measures how far each perturbation moved the expression of its target gene, and the E-distance how far it moved its cells from the control cells. Profiles average each perturbation's expression against the control cells, and clusters group the perturbations whose profiles agree. `cytearc.perturbation` holds the records of the calls, of the thresholds fitted to make them, of the labels, of the efficiency, of the E-distances, and of the later steps. [Pooled CRISPR screens](../tutorials/crispr-screens) runs every step on a published screen.

The steps run on a `DataStore`. {meth}`ds.perturbation.call_guides <cytearc.datastore.namespaces.PerturbationAccessor.call_guides>` calls the guides of each cell from the guide counts, and {meth}`ds.perturbation.import_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls>` saves Cell Ranger's calls from `protospacer_calls_per_cell.csv` instead. Both read the guide counts of the store's only assay declared `CRISPR`, whatever its name, or of the assay that `from_assay` names. A store with several such assays raises `ValueError`, and so does a store with none: an assay takes its type when the store is opened, so declare the guide assay as `DataStore(path, assay_types={"Guides": "CRISPR"})`. A `from_assay` that names no assay of the store raises `ValueError` listing its assays. Both save a `guide_calls` artifact. {meth}`ds.perturbation.load_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls>` reads the artifact as a {class}`~cytearc.perturbation.GuideCalls` record, and {meth}`ds.plots.guide_calls <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_calls>` plots how many guides each cell carries and how many cells carry each guide. {meth}`ds.perturbation.load_guide_thresholds <cytearc.datastore.namespaces.PerturbationAccessor.load_guide_thresholds>` reads the fits of the fitted methods as a {class}`~cytearc.perturbation.GuideThresholds` record, and {meth}`ds.plots.guide_thresholds <cytearc.datastore.plot_accessor.DataStorePlotAccessor.guide_thresholds>` draws them. {meth}`ds.perturbation.labels <cytearc.datastore.namespaces.PerturbationAccessor.labels>` labels each cell by the genes that its called guides target and saves a `perturbation_labels` artifact, which {meth}`ds.perturbation.load_labels <cytearc.datastore.namespaces.PerturbationAccessor.load_labels>` reads as a {class}`~cytearc.perturbation.PerturbationLabels` record. {meth}`ds.perturbation.efficiency <cytearc.datastore.namespaces.PerturbationAccessor.efficiency>` compares each target gene's expression in the cells of its perturbation with the control cells and saves a `perturbation_efficiency` table, which {meth}`ds.perturbation.load_efficiency <cytearc.datastore.namespaces.PerturbationAccessor.load_efficiency>` reads as a {class}`~cytearc.perturbation.PerturbationEfficiency` record and {meth}`ds.plots.efficiency <cytearc.datastore.plot_accessor.DataStorePlotAccessor.efficiency>` draws. {meth}`ds.perturbation.edistance <cytearc.datastore.namespaces.PerturbationAccessor.edistance>` measures how far the cells of each perturbation sit from the control cells in a space such as a PCA and saves a `perturbation_edistance` table, which {meth}`ds.perturbation.load_edistance <cytearc.datastore.namespaces.PerturbationAccessor.load_edistance>` reads as a {class}`~cytearc.perturbation.PerturbationEDistance` record and {meth}`ds.plots.edistance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.edistance>` draws. {meth}`ds.perturbation.signature <cytearc.datastore.namespaces.PerturbationAccessor.signature>` computes each cell's perturbation signature against its nearest control cells, the first step of Mixscape, and saves a `perturbation_signature` artifact, which {meth}`ds.perturbation.load_signature <cytearc.datastore.namespaces.PerturbationAccessor.load_signature>` reads as a {class}`~cytearc.perturbation.PerturbationSignature` record. {meth}`ds.perturbation.mixscape <cytearc.datastore.namespaces.PerturbationAccessor.mixscape>` calls the cells of each target knocked out or escapers and saves a `mixscape_classes` artifact, which {meth}`ds.perturbation.load_mixscape <cytearc.datastore.namespaces.PerturbationAccessor.load_mixscape>` reads as a {class}`~cytearc.perturbation.MixscapeResult` record and {meth}`ds.plots.mixscape <cytearc.datastore.plot_accessor.DataStorePlotAccessor.mixscape>` draws. {meth}`ds.perturbation.associations <cytearc.datastore.namespaces.PerturbationAccessor.associations>` tests whether each perturbation changes each gene, with negative-control pairs that check the calibration of its p-values, and saves a `perturbation_associations` table, which {meth}`ds.perturbation.load_associations <cytearc.datastore.namespaces.PerturbationAccessor.load_associations>` reads as a {class}`~cytearc.perturbation.PerturbationAssociations` record and {meth}`ds.plots.association_volcano <cytearc.datastore.plot_accessor.DataStorePlotAccessor.association_volcano>` and {meth}`ds.plots.association_calibration <cytearc.datastore.plot_accessor.DataStorePlotAccessor.association_calibration>` draw. {meth}`ds.perturbation.profiles <cytearc.datastore.namespaces.PerturbationAccessor.profiles>` averages each perturbation's z-normalized expression into a profile and saves a `perturbation_profiles` artifact, which {meth}`ds.perturbation.load_profiles <cytearc.datastore.namespaces.PerturbationAccessor.load_profiles>` reads as a {class}`~cytearc.perturbation.PerturbationProfiles` record, and {meth}`ds.perturbation.cluster <cytearc.datastore.namespaces.PerturbationAccessor.cluster>` clusters the perturbations whose profiles agree and saves a `perturbation_clusters` artifact, which {meth}`ds.perturbation.load_clusters <cytearc.datastore.namespaces.PerturbationAccessor.load_clusters>` reads as a {class}`~cytearc.perturbation.PerturbationClusters` record and {meth}`ds.plots.perturbation_clusters <cytearc.datastore.plot_accessor.DataStorePlotAccessor.perturbation_clusters>` draws. Each of these figures raises `ValueError` that names what is missing when it has nothing to draw, such as an efficiency table without a tested row, rather than draw an empty panel.

A `guide_calls` artifact labels each cell with its one called guide, with `"Negative"` when no guide is called, and with `"Multiple"` when several are. The labels group cells like cluster labels do, so `ds.select_cells(calls, include=["GUIDE-1"])` selects the cells of one guide and plots colour cells by their call; `ds.select_cells(calls, value="n_guides", low=1, high=1, keep_bounds=True)` keeps the cells with exactly one called guide. [Read per-cell values](cell_aligned_kinds) lists the artifact's arrays.

## Call guides from counts

`call_guides` calls a guide in a cell only when the cell's UMIs of the guide stand out from the ambient molecules. `method` chooses how:

| Method | Model | Cells in the fit | Default `min_umi` | Suits |
|---|---|---|---|---|
| `"poisson_gaussian"`, the default | A Poisson background and a Gaussian on log2 counts ([Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013)), fitted as in crispat ([Braunger and Velten 2024](https://doi.org/10.1093/bioinformatics/btae535)) and pertpy | The cells with at least one UMI of the guide | 1 | Most screens, at low or high multiplicity of infection (MOI) |
| `"gaussian"` | Two Gaussians with one shared variance on log10(count + 1), as Cell Ranger describes | Every selected cell | 3, Cell Ranger's floor | Calls that follow Cell Ranger's |
| `"fixed"` | No model: every guide is called from `min_umi` UMIs | No fit | Required | A floor chosen in advance |
| `"top_guide"` | No model: each cell's top guide is called when the cell holds `min_umi` guide UMIs and the top guide holds more than `fraction_threshold` (default 0.8) of them, as the maximum rule of the [SCEPTRE book](https://timothy-barry.github.io/sceptre-book/assign-grnas.html) does | No fit | 5 | Low-MOI screens, where each cell should carry one guide |

`min_umi` is a floor on a guide's UMIs in a cell for the fitted methods and `fixed`, and on a cell's total guide UMIs for `top_guide`. The fitted methods fit each guide on its own and set the guide's `threshold` to the smallest UMI count, from 1 to the guide's largest, whose posterior for the cells that carry the guide exceeds one half. A guide is called in every cell whose UMIs of the guide reach the larger of `threshold` and `min_umi`, the `call_threshold` of the `GuideThresholds` table, so a cell can have any number of calls. `top_guide` makes at most one call per cell and labels `"Negative"` a cell with fewer guide UMIs than `min_umi` or whose top guide holds no more than `fraction_threshold` of them.

`sample_column` names a cell metadata column, such as the GEM well of each cell, whose samples the fitted methods fit separately; every selected cell needs a label. Without it, each guide gets one fit over every selected cell.

Each fit ends with a status, and a fit whose status is not `"fitted"` calls no cell:

- `"no_cells"`: no cell of the sample holds a UMI of the guide.
- `"too_sparse"`: too few cells or counts to fit; for `poisson_gaussian`, fewer than 2 cells with a UMI of the guide or a largest count below 2, and for `gaussian`, a single distinct count.
- `"no_fit"`: no starting point led to an admissible fit.
- `"no_positive"`: the fit is admissible but calls no count up to the guide's largest.

`call_guides` warns with the number of fits that call no cell, by status, and names a few; the warning comes again when the calls are reused. With `gaussian`, the fit of a guide that few cells carry can split the cells without a UMI of the guide from the cells with ambient UMIs, and call the latter. `call_guides` warns when more than half of the cells with a call have several guides, which points to thresholds set among the ambient counts.

The fitted methods save their fits as a `guide_thresholds` artifact, which the calls record as an input: each fit's status, threshold and parameters, and the histogram of each guide's UMI counts in each sample. `min_umi` and `fraction_threshold` belong to the calls rather than to the fits, so calling again with another value reuses the saved fits and reads the counts once, to call. Calling again with the same arguments returns the saved calls without reading the counts.

`call_guides` and `import_guide_calls` check the memory budget before they read the counts or parse the call file. The budget must hold the calls padded to the most guides called in one cell, which each cell's detected guides bound under `fixed` and the fitted methods and the most calls of one row of the file bound for an import, the text of the cells' labels, a row of streamed counts, and for an import the parsed call file; a budget too small is refused, and the refusal names a budget that runs. The histograms of the fitted methods grow with the distinct UMI counts, which only reading the counts tells, so a fit can stop while it reads them; that refusal names the budget that gets past the point where it stopped and says when more counts remained to be read.

```python
cells = ds.select_measured_cells("CRISPR")
calls = ds.perturbation.call_guides(cells)
fits = ds.perturbation.load_guide_thresholds(calls)
fits.table.query("status != 'fitted'")  # the fits that call no cell
ds.plots.guide_thresholds(calls, guides=["GUIDE-1"])
stricter = ds.perturbation.call_guides(cells, min_umi=3)  # reuses the fits
```

`load_guide_thresholds` reads the fits through the calls or from their `guide_thresholds` artifact, for every guide or for the guides that `guides` names, as a list, an array or the `guides` index of a loaded record. Its `table` holds one row per sample and guide, and its `histograms` the number of cells at each UMI count of each guide in each sample, zero included. Read through the calls, the table's `call_threshold` applies their `min_umi`. `load_guide_thresholds` refuses calls of `fixed` and `top_guide`, which fit no thresholds, and calls imported from Cell Ranger, which writes its thresholds to `crispr_analysis/protospacer_umi_thresholds.csv`.

`ds.plots.guide_thresholds` draws the fits. Without `guides`, it draws an overview: each fit's threshold against the fit's cells with a UMI of the guide, and the number of fits of each status. With `guides`, it draws one panel per guide and sample: the cells at each UMI count, the posterior of the population of cells that carry the guide, and lines at the threshold and at the call threshold. `samples` keeps the panels or points of some samples, and a selection of more than `max_panels` panels raises `ValueError`. Its tables are `thresholds`, `histograms` and, with `guides`, `posterior`.

## Label cells by perturbation

Guide calls name guides, but a screen asks what perturbing each gene does: a library holds several guides for each gene, and control guides, such as non-targeting guides, that perturb no gene. `labels` merges the guides of each gene, so the cells of a gene's guides form one group and the cells of control guides another. A cell's label follows from the targets of its called guides:

| Called guides of the cell | Label | `is_control` | `n_targets` |
|---|---|---|---|
| None | `"Negative"` | False | 0 |
| Control guides only | `"control"` | True | 0 |
| Guides of one gene, with or without control guides | The gene, such as `"GENE1"` | False | 1 |
| Guides of two or more genes | `"Multiple"`, or with `policy="combinations"` the genes sorted and joined by `"+"`, such as `"GENE1+GENE2"` | False | 2 or more |

Two guides of one gene give that gene's label, and so does a gene's guide beside a control guide, while the guide calls label both cells `"Multiple"`. The labels keep each cell's label in the calls as `guide_label`, so the two can be compared. Cells that `call_guides(method="top_guide")` labelled `"Negative"` stay `"Negative"`; the calls' `total_umi` and `top_fraction` tell cells with too few guide UMIs from cells whose guide UMIs are split among several guides. `cytearc.perturbation` exports the three labels of cells without one gene as `CONTROL`, `NEGATIVE` and `MULTIPLE`.

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

The table belongs to the expression assay rather than to the guide assay of the labels. It records the labels, the expression data, the feature matched to each row's target and `by`, so measuring again with the same values returns the saved table and repeats its warning, an `id_column` that matches a target to another feature makes a new table, and `invalidate_cache=True` measures again. Labels without a `"control"` cell raise `ValueError`.

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

`ds.plots.efficiency` draws one row per tested target or guide, from the strongest knockdown at the top to the strongest activation at the bottom: a point at the row's log2 fold change against the control cells, coloured by whether it is significant at the false discovery rate `alpha` (0.05), its `adjusted_p_value` at most `alpha`, with an area that grows with its cells. A fold change of minus infinity, when the perturbed cells hold no counts of the gene, is a triangle at the left edge, and the row's label ends in "(no counts)". When more rows were tested than `max_rows` (40), the plot keeps the `max_rows` rows with the smallest adjusted p-values. Rows of the other statuses are not drawn, and the title counts them. The plot's table, `efficiency`, holds the drawn rows in drawing order.

On the 10x A549 screen, with Cell Ranger's calls imported and `controls=["Non-Targeting"]`, `efficiency` takes 0.8 s and reproduces the cells and mean UMIs of Cell Ranger's perturbation efficiency tables; [Methods and limitations](../reference/methods#validation) compares the two.

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

The table belongs to the assay of the coordinates. It records the labels, the coordinates, `by`, `n_permutations` and `random_seed`, so measuring again with the same values returns the saved table, and `invalidate_cache=True` measures again. Labels without a `"control"` cell, and coordinates of fewer than two control cells, raise `ValueError`. Each E-distance follows from the sums and squared norms of each group's coordinates, so no matrix of pairwise distances is formed. The memory budget must hold about 16 bytes for each compared cell and dimension, and for each thread a block of permutation draws of up to 8 MiB, or one permutation of the control cells when that is larger.

```python
pca = ds.reduction.pca(normalized, dims=30)  # a PCA of the cells' expression
edistance = ds.perturbation.edistance(labels, pca)
table = ds.perturbation.load_edistance(edistance).table
table.query("status == 'tested'").sort_values("edistance", ascending=False)  # strongest first
by_guide = ds.perturbation.edistance(labels, pca, by="guide")  # one row per guide
ds.plots.edistance(edistance)
```

`load_edistance` returns a {class}`~cytearc.perturbation.PerturbationEDistance` record. Its `table` holds one row per target gene, sorted by target, or one row per guide of a target gene, in the order of the guide assay; control targets and their guides have no row. Its columns are `guide` for rows by guide, `target`, `status` and the columns above. `by`, `n_permutations` and `random_seed` record how the table was measured, and `from_assay` names the assay of the coordinates.

`ds.plots.edistance` draws one row per tested target or guide, from the largest E-distance at the top: a point at the row's E-distance, coloured by whether it is significant at the false discovery rate `alpha` (0.05), its `adjusted_p_value` at most `alpha`, with an area that grows with its cells. Without permutation tests, every point takes the colour of a small p-value and the title says that no test ran. When more rows were tested than `max_rows` (40), the plot keeps the `max_rows` rows with the largest E-distances. Rows of the other status are not drawn, and the title counts them. The plot's table, `edistance`, holds the drawn rows in drawing order.

On the 10x A549 screen, the E-distances equal those of scPerturb's `edist`, while pertpy's `edistance` uses Euclidean rather than squared distances and counts each cell with itself, so its values differ; [Methods and limitations](../reference/methods#validation) compares the distances, the p-values and the time of each tool.

## Compute perturbation signatures

In a knockout screen, some cells that carry a targeting guide keep their target's function, so the cells of a perturbation mix changed and unchanged cells. Mixscape ([Papalexi et al. 2021](https://doi.org/10.1038/s41588-021-00778-2)) tells them apart, and its first step, the perturbation signature, removes the variation that a cell shares with the control cells around it, such as its cell-cycle phase or its replicate, so that what remains is the perturbation's effect. `signature` matches every cell of `coordinates`, such as a PCA, with its `n_neighbors` (20) nearest `"control"` cells of the labels over the first `dims` (15) dimensions, found exactly, and gives each feature the cell's log-normalized value minus the log of its neighbors' mean normalized value. A control cell is its own nearest control, and of two equally near controls the first in store order is taken. Values are the raw counts scaled to `scale_factor` (10,000) counts per cell and passed through log1p, Seurat's LogNormalize, whatever the assay's own normalization.

`features` defaults to the feature selection of the coordinates, such as the highly variable genes that a PCA reduces, as Seurat's `CalcPerturbSig` defaults to the variable features; imported coordinates record no feature selection and need it. `dims` cannot exceed the dimensions of the coordinates, and Seurat's vignette uses 40. With `sample_column`, such as a replicate column, each cell takes its neighbors from the control cells of its own sample, as Seurat's `split.by` does.

The signature belongs to the assay of `features` and holds one row per cell of the coordinates. Labelled cells without coordinates get none, and a log message counts them; a sample with fewer control cells than `n_neighbors` raises `ValueError` that gives each such sample's count. The artifact records the labels, the coordinates, the features, the counts and each cell's sample, with `dims`, `n_neighbors`, `sample_column` and `scale_factor`, so computing again with the same values returns the saved signature. Seurat stores control minus cell; CyteArc stores cell minus control, so a knockdown reads negative, which leaves Mixscape's classes unchanged.

The neighbor search compares blocks of cells with every control cell of their sample, as many cells as the memory budget leaves room for. The control cells' scaled counts stay in memory while every cell's counts stream, a stored chunk of rows at a time, beside the writer of the signatures, which are stored in row bands of at most 8 MiB. A memory budget that cannot hold one cell's distances, or one block of counts beside the writer's band, is refused, and the refusal names the smallest budget that runs the signature.

```python
signature = ds.perturbation.signature(
    labels, pca, dims=40, n_neighbors=20, sample_column="replicate"
)
record = ds.perturbation.load_signature(signature, features=["ENSG00000027697"])
record.values  # one row per cell, one column per feature
record.neighbors  # each cell's nearest control cells, nearest first
```

`load_signature` returns a {class}`~cytearc.perturbation.PerturbationSignature` record. Its `values` hold one row per cell, indexed by cell ID, and one column per feature, named by feature ID; `features` reads some of them. The rows are read a stored band at a time, every feature of it, so the memory budget must hold the values read, one band and the IDs of the cells and their neighbors. Its `cells` table holds each cell's `is_control` flag and `sample`, its `neighbors` table each cell's nearest control cells as cell IDs, nearest first, and the record keeps `from_assay`, `dims`, `n_neighbors`, `scale_factor` and `sample_column`.

## Call knocked-out cells and escapers with Mixscape

`mixscape` reads a signature and, among the cells that carry each target alone, calls the cells whose target was knocked out (`"KO"`) and the escapers that kept its function (`"NP"`, not perturbed), as Seurat's `RunMixscape` does. For each target, within each split of `split_column`:

1. The DE genes compare the target's cells with the control cells on log-normalized counts, at the signature's `scale_factor`, over every feature of the signature's assay, by the rules of Seurat's `FindMarkers`. A gene is tested when at least `min_fraction` (0.1) of the cells of either group express it, rounded to 3 digits, and when its log2 fold change, the difference of each group's log2((sum + 1) / cells) of normalized values, is at least `logfc_threshold` (0.25) in size. A two-sided Wilcoxon rank-sum test with continuity correction gives each tested gene a p-value, and the gene is DE when that p-value times the number of features is below `p_value_cutoff` (0.05). A group of fewer than 3 cells has no DE gene.
2. A target with fewer than `min_de_genes` (5) DE genes keeps every cell as an escaper, with the status `"few_de_genes"`.
3. The signatures of the DE genes, read from the signature or computed from its neighbors for DE genes outside its features, are centred and scaled over the target's and the control cells (`scale`). Starting with every cell of the target called KO, each cell is projected onto the difference between the mean of the KO cells and the mean of the control cells. A mixture of two normal distributions fits the projections of the target's and the control cells, with the control component fixed at the control cells' mean and standard deviation, and a cell is KO when its likelihood under the fitted component exceeds its likelihood under the control component. The calls repeat with the new KO cells until they stop changing, at most `max_iterations` (10) times. Fewer than `min_cells` (5) KO cells after an iteration make every cell of the target an escaper, with the status `"few_cells"`, and a fitted component that collapses onto one value gives `"no_fit"`.

A cell's class is `"<target> KO"`, with `perturbation_type` in place of `"KO"`, or `"<target> NP"`; a control cell's class is `"control"`, and any other cell keeps its label, such as `"Negative"` or `"Multiple"`, or is `"unlabelled"` when the signature covers it but the labels do not. The classes form a cell-aligned `mixscape_classes` artifact, so `select_cells` selects the KO cells of a target, plots colour cells by class, and a pseudobulk of KO cells leaves the escapers out.

```python
signature = ds.perturbation.signature(labels, pca, dims=40, sample_column="replicate")
classes = ds.perturbation.mixscape(signature)
result = ds.perturbation.load_mixscape(classes)
result.targets[["target", "status", "n_cells", "n_ko", "n_de_genes"]]
knocked_out = ds.select_cells(classes, include=["IFNGR1 KO"])
ds.plots.mixscape(classes)  # the KO and NP shares of each target
```

`load_mixscape` returns a {class}`~cytearc.perturbation.MixscapeResult` record. Its `cells` table holds each cell's `mixscape_class`, `global_class` (`"KO"`, `"NP"`, `"control"` or the cell's label), `posterior`, the probability of being knocked out under equal priors, `score`, the projection of the last iteration, and `split`; `posterior` and `score` are NaN outside a fitted target. Its `targets` table holds one row per target and split with the `status`, `n_cells`, `n_ko`, `n_de_genes`, `iterations`, `converged` and the fitted mixture, `control_mean`, `control_sd`, `perturbed_mean`, `perturbed_sd` and `perturbed_weight`, and its `de_genes` table one row per target, split and DE gene. The record keeps `perturbation_type`, `split_column` and `from_assay`, the expression assay of the signature. The classes record the signature, the counts and each cell's split, with every setting above, so classifying again with the same values returns the saved classes.

`mixscape` reads the counts once, in bands of features that leave room for their read, and tests the genes of every target in the same pass, with the control cells' values sorted once for all the targets of their split. It then classifies the targets in batches that fit the memory budget; each batch reads the stored signatures once, a stored band of rows at a time, and computes those of DE genes outside the signature's features from the stored neighbors. A memory budget that cannot hold one feature's values and read is refused before the DE tests, and one that cannot classify a target's cells with the control cells is refused before any class is saved; each refusal names the smallest budget that its step needs. `perturbation_type` cannot be `"NP"`, `"control"`, `"Negative"`, `"Multiple"` or `"unlabelled"`, the classes of other cells, since those cells would merge with the knocked-out ones. The counts are normalized as `count / total * scale_factor`, dividing first as Seurat's `LogNormalize` does, so that two cells with equal shares of their counts tie in the Wilcoxon test as they do in Seurat.

CyteArc follows `RunMixscape` with two differences. The mixture's weights start at one half, while Seurat starts them at 0.416 for the control component and 0.584 for the other: mixtools draws its starting weights at random, but Seurat's `WhichCells` resets R's random seed to 1 just before each fit, so Seurat's calls do not change between runs. Both reach the same fit, by different steps, and stop where the log-likelihood stops rising by 1e-8, points about 1e-5 apart in the fitted mean. When an iteration leaves fewer than `min_cells` KO cells, CyteArc calls every cell of the target an escaper, while Seurat relabels `"NP"` the cells it called KO before that iteration and keeps the cells it newly calls KO. Seurat stores its signature with the opposite sign and clips its scaled values above 10; CyteArc clips below -10, which gives the same classes.

On the THP-1 screen of Seurat's Mixscape vignette ([Papalexi et al. 2021](https://doi.org/10.1038/s41588-021-00778-2)), `signature` finds the nearest control cells of `CalcPerturbSig` and `mixscape` gives every cell the class of `RunMixscape`; [Methods and limitations](../reference/methods#validation) compares the values, p-values, posteriors and times.

`ds.plots.mixscape` draws, with `panel="classes"`, one bar per target split into the shares of KO and NP cells, and with `panel="scores"`, one panel per fitted target with the scores of its KO and NP cells and the two fitted normal distributions, as Seurat's `PlotPerturbScore` does.

## Test which genes each perturbation changes

A screen asks which genes each perturbation changes, and the answer is only as good as the calibration of its p-values: single-cell counts are sparse and overdispersed, cells differ in depth, and a screen makes many thousands of tests. SCEPTRE ([Barry et al. 2024](https://doi.org/10.1186/s13059-024-03254-2)) keeps the p-values calibrated with permutations of a score statistic, and `associations` runs its low-MOI test, or with `moi="high"` its test for screens with several guides per cell ([below](#screens-with-several-guides-per-cell)). A pair is a target and a feature, and its treatment cells are the cells that carry the target alone, compared with the `"control"` cells of the labels; with `integration="singleton"`, each guide is tested alone.

For each pair, a negative binomial regression of the feature's counts on the cells' covariates is fitted under the null hypothesis on the pair's treatment and control cells together: Poisson coefficients, then the size that maximizes the likelihood with the Poisson means held fixed, as the [SCEPTRE book](https://timothy-barry.github.io/sceptre-book/) describes it and as sceptre's results show. The covariates are an intercept, log detected features and log UMIs of each cell in `from_assay`, indicators of the samples of `sample_column` and the numeric cell columns of `covariates`, sceptre's default formula. Only the tested cells, the control cells and the cells of the tested targets or guides, need a sample label and covariate values, and a missing value among them is refused. The statistic is the treatment cells' summed score divided by its standard deviation under the fit, positive when they express the feature more than the fit predicts. When the counts show no overdispersion that the likelihood can see, the size solves the Pearson moment equation instead, or the counts are taken as Poisson.

Random sets of as many cells as the treatment cells, drawn without replacement from the pair's cells, score the same statistic. On each side, the p-value is one more than the sets at least as extreme, over one more than the sets with a statistic, and `side="both"` takes twice the smaller side. `n_resamples`, a tuple or a list of three integers, sets the rounds, and `n_resamples[0]` (499) sets come first. A pair whose p-value is then at most `stage_threshold` (0.02) takes `n_resamples[1]` (4,999) fresh sets, whose statistics a skew-normal distribution fits by its moments, with a skewness beyond the skew-normal's reach replaced by 0.9 times its limit, as in sceptre's fits; the fitted tail, computed in log space, gives p-values far below the reciprocal of the sets. When that tail beyond the 99th percentile of the statistics, on the observed statistic's side, holds less than half of the statistics' own 1%, its extrapolation would be too thin, and `n_resamples[2]` (24,999) fresh sets give the p-value instead; `approximation="none"` keeps the second round's p-value. The first round's p-values are at least 2 / (`n_resamples[0]` + 1), or 1 / (`n_resamples[0]` + 1) for one side, so a positive `stage_threshold` below that, which no pair could reach, is refused: the default of 0.02 needs 99 first-round sets, or 49 for one side, and `stage_threshold=0` keeps the first round's p-values. Every feature of a target sees the same sets, drawn by a counter-based generator keyed by `seed`, the target and the round, so the tests do not depend on threads or on the memory budget. Since the fit treats the pair's cells alike, the permutation test is exact when those cells are exchangeable under the null hypothesis.

A pair is tested when at least `min_nonzero` (7) treatment and control cells express the feature, and the p-values are adjusted by Benjamini-Hochberg within each set of pairs; `significant` marks those at most `alpha` (0.1). A pair that passes QC has no statistic when the covariates determine which of its cells are treatment cells, as they do when a target's cells are exactly the cells of samples, or of a binary covariate, that hold no control cell: the treatment cells' score then has no variance. Such a pair gets NaN `z`, `p_value`, `log2_fold_change` and `se_log2_fold_change`, since the fit matches its treatment cells' counts by construction, and `stage` 0; it is not significant, not counted as tested, and not a false discovery among the calibration pairs, and a warning names each such target, or guide, and calibration group, again when the tests are reused. The log2 fold change is the treatment cells' counts over their means under the pair's fit, the estimator of the SCEPTRE book, which shrinks toward 0 when the treatment cells are many next to the control cells; its standard error comes from the negative binomial variance of the counts and the information of the fit.

Three sets of pairs are tested:

| Set | Pairs | What it shows |
|---|---|---|
| Discovery | With `pairs="trans"`, every target with every feature of `features`, such as the highly variable genes; or the rows of a table of `target` and `feature`, feature IDs or names that one feature has | The perturbations' effects |
| Calibration | Up to 100 distinct groups of control guides, drawn at random with as many guides as the median target, possibly sharing guides, each with the control cells whose one guide is in the group as its treatment cells and every other control cell as its comparison; paired at random with features of the whole assay that pass QC, as many pairs as the discovery pairs that pass QC unless `n_calibration_pairs` says otherwise | Whether the p-values are calibrated: these pairs have no effect, so their discoveries are false discoveries |
| Positive controls | With `positive_controls="targets"`, each target with its own gene, matched as `efficiency` matches it; or a table of pairs | Whether the test finds known effects |

The calibration check grades its false discoveries as the SCEPTRE book does: excellent (0 or 1), mild (2 to 6), moderate (7 to 14) or severe (15 or more), and a moderate or severe grade raises a warning. Its control guides are those that are the one guide of a control cell. A group's comparison cells are all the other control cells: the cells of the other control guides, and the control cells that carry several control guides, even guides of the group. Under the null hypothesis no control guide has an effect, so these cells keep the check valid. The grade is `"none"` when no check runs: with `n_calibration_pairs=0`, with fewer than 2 control guides, when no pair of a group and a feature passes QC, when no calibration pair has a statistic, or by default when no discovery pair passes QC, since the default check tests as many pairs as pass. A warning says why in all but the first case. A positive `n_calibration_pairs` runs the check without discovery pairs too.

```python
hvg = ds.features.hvgs(cells, top_n=2000)  # the features that trans pairs test
tests = ds.perturbation.associations(labels, features=hvg, sample_column="replicate")
result = ds.perturbation.load_associations(tests)
result.calibration_grade  # "excellent" before reading any discovery
result.discovery[result.discovery["significant"]]
ds.plots.association_calibration(tests)
ds.plots.association_volcano(tests, targets=["IFNGR1", "JAK2"])
```

`load_associations` returns a {class}`~cytearc.perturbation.PerturbationAssociations` record with one table per set, which must fit the memory budget. Each row holds the `target`, its `guide` with `integration="singleton"` or the `control_guides` of a calibration group, the `feature` ID and `feature_name`, `n_cells`, `n_control_cells`, `n_nonzero_cells`, `n_nonzero_control_cells`, `pass_qc`, `z`, `p_value`, `adjusted_p_value`, `significant`, `log2_fold_change`, `se_log2_fold_change`, `stage`, the round that gave the p-value (0 for a pair without one), and the skew-normal fit of the second round, `skew_location`, `skew_scale` and `skew_shape`, NaN unless that fit gave the p-value. The record keeps `alpha`, `side`, `integration`, `moi`, `resampling`, `from_assay`, `n_false_discoveries` and `calibration_grade`. The tests record the labels, the counts, the features or tables of pairs, each tested cell's sample and covariates, and every setting above, so testing again with the same values returns the saved tests and repeats their warnings.

`associations` reads the counts once, in bands of features, to count each feature's expressing cells in each target, each control guide and the other control cells, which gives every pair's QC; it holds the counts as float32 for the tests when they fit half the memory left, and reads them again in bands otherwise. Each thread fits a batch of a target's pairs and scores the batch's random sets together, one contiguous record per cell. A memory budget too small to hold one feature's counts and their read beside the tests and their random sets is refused, and the refusal names the smallest budget that runs the tests.

`ds.plots.association_volcano` draws each tested discovery pair's log2 fold change against its -log10 p-value, coloured by significance, with the Benjamini-Hochberg cut and the `label_top` (10) pairs of smallest p-values labelled. `ds.plots.association_calibration` draws the book's calibration figure: the negative-control p-values against uniform quantiles with a 95% band and the discovery p-values beside them, the negative-control fold changes, and the positive-control and negative-control p-values with the count of false discoveries.

### Screens with several guides per cell

In a high-MOI screen each cell carries several guides at random, so few cells carry a target alone and almost none carry control guides only. `moi="high"` runs SCEPTRE's high-MOI test ([Barry et al. 2021](https://doi.org/10.1186/s13059-021-02545-2)). A target's treatment cells are the labelled cells whose guides target it, whatever else they carry, or with `integration="singleton"` the cells that carry the guide. Every other labelled cell is its comparison: cells of other targets and of sibling guides, cells with control guides only, and `"Negative"` cells. `n_control_cells` counts them, and QC counts the expressing cells of a target against all of them. A target whose cells are exactly the cells of samples, or of a binary covariate, has no statistic, as at low MOI.

Every labelled cell is tested, so each one needs a sample label, covariate values and guide UMIs. The covariates add each cell's log detected guides and log guide UMIs in the guide assay, the terms of sceptre's default formula at high MOI, since sequencing depth drives both the detection of guides and the expression of genes; a cell without guide UMIs is refused, and `select_cells(calls, value="total_umi", low=1, keep_bounds=True)` selects the cells to call guides on instead. Every target is compared within the same cells, so each feature's regression is fitted once, on all labelled cells, and the feature's pairs share that fit. The fold change barely shrinks, since a target's cells are a small share of all cells.

`resampling` chooses the random sets. `"conditional"`, the default at high MOI, is the conditional randomization test: a logistic regression of the target's cells on the covariates gives each cell its probability of carrying the target, and each set takes every cell independently with that probability, so the sets keep the link between the covariates and guide detection that would otherwise confound the test. `"permutation"` draws as many cells as the target's, uniformly from all cells; the SCEPTRE book describes permutations as faster and slightly better with sparse genes, and conditional sets as slightly better with confounding covariates, and suggests trying both. Each target draws its own sets in both. At low MOI the sets are permutations, and `"conditional"` is refused.

The calibration groups draw from the control guides that some labelled cell carries, as many as the median target's guides or all of them when there are fewer. A group's treatment cells carry any of its guides, and all other labelled cells are its comparison, so one control guide is enough; the grade is `"none"` when no labelled cell carries a control guide. The comparison cells hold the cells of every target, so a target that strongly lowers a gene lowers it in the comparison cells of every other group too, and a calibration group can then show that gene as raised.

```python
import pandas as pd

# Each candidate enhancer with the genes near it
cis_pairs = pd.DataFrame(
    {
        "target": ["candidate_enh_1", "candidate_enh_1"],
        "feature": ["ENSG00000174529", "ENSG00000117222"],
    }
)
tests = ds.perturbation.associations(
    labels,
    moi="high",
    pairs=cis_pairs,
    side="left",  # CRISPRi of an enhancer lowers its genes
    sample_column="gem_group",
    # An enhancer has no gene of its own; a table like cis_pairs of targets at
    # transcription start sites and their genes checks the tests' power instead
    positive_controls=None,
)
result = ds.perturbation.load_associations(tests)
result.calibration_grade  # read before the discoveries
result.discovery[result.discovery["significant"]]
```

The tests read the counts once for the QC, every feature of the assay when a calibration check runs, and hold them as float32 when they fit half the memory left. They then test bands of genes: each gene is fitted on all n labelled cells, and its scores and weights, 16 n bytes, are held cell by cell so that each cell of a random set adds one row of the band. Each thread holds 32 n bytes while it fits a gene, and while it draws a target's conditional sets, 24 n bytes plus a copy of some of the target's genes, which takes at most the larger of 16 n bytes and 64 MiB. Before the QC, each target's cells come from the stored guide calls and labels, read a stored chunk of up to 100,000 cells at a time; a chunk holds as many guides, or targets, for every cell as the cell with the most, so a few cells with many guides widen every chunk, and the budget covers one chunk as stored, decoded and read. The memory budget sets the band, so a run that cannot hold one gene is refused, and the refusal names the smallest budget that runs it. A random set costs about its cells times the target's genes in the band times the covariates plus 2 multiply-adds, so round 1 of P pairs of targets of m cells costs about 499 P m (p + 2) for p covariates, and the second round about 10 times that for each pair that takes it. [Methods and limitations](../reference/methods#validation) times the tests of every target with every gene on sceptredata's example.

## Average each perturbation's expression into a profile

A screen with many perturbations asks which of them act alike, such as the subunits of one complex or the steps of one pathway. [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013) compared perturbations by their mean expression profiles, each feature measured in standard deviations of the control cells, and `profiles` computes these profiles:

1. Each cell's counts in the features of `features` are scaled so that its total, the sum of its stored counts over every feature of the assay, equals the median total of the `"control"` cells of the labels.
2. Within each sample of `sample_column`, such as a GEM group, each feature is centred on the mean of the sample's control cells and divided by their standard deviation with the n - 1 divisor, `z = (x - mean) / sd`, for every cell, control cells included. A sample needs 2 control cells or more.
3. A perturbation's profile is the mean z-score of its cells over all samples, computed in float64 and stored as float32.

These steps rebuild the published pseudo-bulk profiles of Replogle et al.'s K562 essential screen from its cells to float32 rounding ([Methods and limitations](../reference/methods#validation)). Where the control cells of a sample hold one value of a feature, the feature has no spread there: CyteArc gives it the z-score 0 in that sample, the control mean, and counts such samples in `n_zero_sd`, while Replogle et al.'s genome-wide profiles hold infinite values there, which leave a correlation undefined.

By default each target that is not a control gets a profile from the cells that carry it alone; the control cells get none, since their mean z-score is 0. With `by="guide"`, each guide gets a profile from the cells whose one called guide it is, each control guide included, as Replogle et al.'s profiles have a row for each guide pair, the control pairs included, and the guide IDs name the rows. With Mixscape classes as `labels`, only the cells that Mixscape called knocked out count for a target or a targeting guide, so that escapers do not dilute the profiles, while a control guide keeps its cells.

A perturbation with fewer than `min_cells` (25) cells, Replogle et al.'s floor, gets no profile, and a warning names it, again when the profiles are reused. With `mask_targets` (on), each profile's own target feature is set to 0, the control mean, so that the knockdown of the target does not by itself make two perturbations alike, as Replogle et al. set it; a target is matched to a feature as `efficiency` matches it, by `id_column` and then by name, and control guides mask nothing. Any feature selection of the expression assay serves as `features`: Replogle et al. took the ten strongest differentially expressed genes of each perturbation with the expressed genes of most variable expression, and the highly variable genes are a common choice.

```python
hvg = ds.features.hvgs(cells, top_n=2000)  # the features of the profiles
profiles = ds.perturbation.profiles(labels, hvg, sample_column="gem_group")
record = ds.perturbation.load_profiles(profiles)
record.values  # one row per target, one column per feature
record.left_out  # the targets with fewer than min_cells cells
guides = ds.perturbation.profiles(labels, hvg, by="guide", sample_column="gem_group")
```

`load_profiles` returns a {class}`~cytearc.perturbation.PerturbationProfiles` record. Its `values` hold one row per profiled perturbation, named by its target or guide ID, and one column per feature, named by feature ID. Its `perturbations` table gives each row's `target`, `is_control`, `n_cells` and `masked_feature`, its `left_out` table the perturbations without a profile with their `target` and `n_cells`, its `features` table each feature's `feature_name` and `n_zero_sd`, and its `samples` table each sample's `n_control_cells`, with the control means and standard deviations in `control_mean` and `control_sd`. The profiles record the labels, the features, the counts, the sample of each cell they read, each profile's masked feature, and `by`, `min_cells`, `mask_targets` and `sample_column`, so computing again with the same values returns the saved profiles.

`profiles` reads the counts twice, the control cells for their means and standard deviations and then the cells of the profiled perturbations for their z-scores, and holds one float64 sum for each perturbation and feature; when those sums and the reads do not fit the memory budget, it reads the features in bands. A memory budget that cannot hold one feature's sums beside its read is refused, and the refusal names the smallest budget that runs the profiles. Each sum adds the cells one after another in store order, so the profiles depend neither on the bands nor on the threads.

## Cluster perturbations by their profiles

`cluster` groups the perturbations whose profiles change expression alike, as Replogle et al. grouped theirs. The distance between two profiles is 1 minus their Pearson correlation over the features, `metric="correlation"`, which ignores how strongly each perturbation acts, since related perturbations often differ in strength; `metric="euclidean"` keeps the strength. HDBSCAN finds the clusters on these distances, through scikit-learn's `HDBSCAN(metric="precomputed")` with `min_cluster_size` (4), `min_samples` (2) and `selection`, `"eom"` to select clusters by excess of mass or `"leaf"` to take the leaves of its cluster tree, and leaves the perturbations that it cannot place unassigned, with the cluster -1. scikit-learn's `min_samples` counts the point itself, so the `min_samples=1` of the `hdbscan` package, which Replogle et al. ran, is 2 here, the default.

Average linkage of the same distances orders the perturbations, so that a heatmap of their correlations in that order shows each cluster as a block. A UMAP layout places them in two dimensions: the 15 nearest perturbations of each by the same distances form its graph, and the layout starts from the first two principal components of the profiles as the metric sees them and runs 500 epochs on one thread with umap-learn's default spread and smallest distance, so that it depends only on `seed`. Replogle et al. drew their map with pymde instead, so the layouts differ where the clusters agree.

`perturbations` restricts the clustering to some of the profiles, named in any order. Replogle et al. clustered only the perturbations with strong effects: at least 50 genes that an Anderson-Darling test found changed, 25 cells, and a knockdown of at least 30% where it was measured. CyteArc measures effects in other ways, so a comparable filter takes the targets with a significant E-distance and a low remaining fraction, such as below 0.7. Profiles by guide include the control guides, whose profiles sit near 0 without resembling each other: within each sample the z-scores of the control cells sum to 0, so the control guides' profiles, weighted by their cells, sum to about 0 and correlate negatively with each other, by about -1/(K - 1) on average for K control guides. Under the correlation distance they therefore scatter, unassigned or in other clusters by chance, and only the Euclidean distance groups them; leave them out with `perturbations` to cluster the targeting guides alone.

```python
distance = ds.perturbation.load_edistance(edistance).table
remaining = ds.perturbation.load_efficiency(efficiency).table
strong = set(distance.loc[distance["adjusted_p_value"] <= 0.05, "target"])
strong &= set(remaining.loc[remaining["remaining_fraction"] < 0.7, "target"])
strong &= set(ds.perturbation.load_profiles(profiles).values.index)
clusters = ds.perturbation.cluster(profiles, perturbations=sorted(strong))
result = ds.perturbation.load_clusters(clusters)
result.table.sort_values("cluster")  # each perturbation's cluster, -1 unassigned
correlations = 1 - result.distances.loc[result.order, result.order]
ds.plots.perturbation_clusters(clusters)  # the correlations, with a bar of clusters
ds.plots.perturbation_clusters(clusters, panel="embedding", label=["MED12", "RPS3"])
```

`load_clusters` returns a {class}`~cytearc.perturbation.PerturbationClusters` record. Its `table` holds one row per clustered perturbation, in the order of the profiles, with `cluster`, `probability`, HDBSCAN's strength of membership (0 when unassigned), `n_cells`, and `x` and `y`, the place in the layout; its `order` names the perturbations in leaf order; and its `distances` are those between their profiles, computed again from the saved profiles, so that `1 - distances` holds the correlations of a heatmap. The clusters record the profiles, the clustered perturbations and every setting above, so clustering again with the same values returns the saved clusters.

For P perturbations in F features, the distances hold 8 P² bytes and the profiles, as float64, 8 P F bytes. With the float32 profiles that it reads, the clustering holds at its peak the larger of 4.5 times the distances plus 1.5 times the profiles, while HDBSCAN runs, and the distances plus 7.5 times the profiles, while the layout's PCA runs, beside four float64 matrices of F x 12 from scikit-learn's randomized solver, which outweigh the profiles below 12 perturbations; the second is larger once F exceeds about 7P/12. That is about 3.8 GB for 10,000 perturbations in 2,000 features and 270 MB for 2,000 perturbations in 2,000 features; `cluster` refuses a clustering that does not fit the memory budget. On the 1,973 perturbations that Replogle et al. clustered from their genome-wide screen, the clusters agree with Table S3 of the paper ([Methods and limitations](../reference/methods#validation)).

`ds.plots.perturbation_clusters` draws the clusters in one of two panels. The heatmap holds the correlations of the profiles in leaf order on a fixed scale from -1 to 1, or their Euclidean distances, beside a bar that colours each perturbation by its cluster, grey when it is unassigned, and numbers the clusters; it names the perturbations when it draws at most 50, and `perturbations` or `cluster_ids` keep fewer. The embedding places each perturbation in the layout, coloured by its cluster with the unassigned perturbations in grey beneath, numbers each cluster at its median, and names the perturbations of `label`. A cluster keeps its colour in both panels and whatever the plot draws.

## Run the steps as one recipe

{meth}`ds.pipeline.perturbation <cytearc.datastore.pipeline_accessor.PipelineAccessor.perturbation>` runs these steps as one durable run: the QC filtering, highly variable genes, PCA, and Harmony of the RNA recipe, the guide calls, the labels, the efficiency, and the E-distances, and on request the signatures and Mixscape, the association tests of every target against every highly variable gene, and the profiles and their clusters. It checks every setting before it writes anything, reuses the artifacts of earlier calls with the same refs and settings, and records each stage's numbers as the stage's summary, such as the cells of each label, the knocked-down targets, and the calibration grade of the tests ([Perturbation recipe](pipeline.md#perturbation)).

```python
screen = ds.pipeline.perturbation(
    labels={"controls": ["Non-Targeting"]},
    signature={"sample_column": "lane"},
    mixscape=True,
    associations={"sample_column": "lane"},
    label="screen",
)
ds.perturbation.load_mixscape(screen["mixscape"]).targets
```

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

## Perturbation signature

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationSignature
```

## Mixscape classes

```{eval-rst}
.. autoclass:: cytearc.perturbation.MixscapeResult
```

## Perturbation associations

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationAssociations
```

## Perturbation profiles

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationProfiles
```

## Perturbation clusters

```{eval-rst}
.. autoclass:: cytearc.perturbation.PerturbationClusters
```
