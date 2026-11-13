---
myst:
  html_meta:
    description: Plot CyteArc embeddings, gene expression, clusters, trajectories, and diagnostics using standalone functions or datastore accessors.
---

# Plotting API reference

`cytearc.plotting` is CyteArc's plotting API.
Import it as `splt` and call functions such as `splt.embedding(...)`, `splt.dotplot(...)`, and `splt.cluster_tree(...)`.
For functions whose first argument is a datastore, `ds.plots.embedding(...)` and related accessor methods provide the same behavior with that argument already bound.
Store-backed graph plotters mirror their `DataStore` counterparts and require an exact `graph=`
reference.
`dotplot` and `matrixplot` take `features=` as gene names or `FeatureRef` values, or as a mapping of group names to such lists. Their color bar names the means it draws: "Mean expression" for the RNA assay and otherwise the assay, such as "Mean of GeneScores", and so does the color bar of a stacked violin that `distribution(kind="stacked_violin", color_by="mean")` colors by its group means.
Diagnostics remain standalone: `qc` takes a DataFrame, `elbow` and `highly_variable_features` take arrays, and `graph_qc` takes a sparse graph.

A completed {py:class}`~cytearc.PipelineRun` can be passed to
`ds.plots.embedding(run=run, layout="umap", color_by="clusters")`. The run supplies the layout
(`layout` names a run output, `"umap"` by default) and the cells, and every cell input comes from
the run. There is no live fallback in run mode:

- `color_by` takes one item or a list, drawn as one panel each. A string or `CellField` names a frozen run cell field from `run.cells.columns`, such as `clusters`, `doublet_score`, a multimodal run's `weight_<assay>` modality weights, or a column frozen with `snapshot_columns`. An `ArtifactRef` must be an output of the run, such as one of its Leiden candidates in `run["leiden"]`, which `ds.artifacts.find("clusters", graph=run["connectivity_map"], method="leiden", resolution=1.0)` also finds. Any other ref raises `ValueError`; color by it with the granular `layout=run["umap"]` route instead.
- `facet_by`, `subset_by`, `Highlight(by=...)` and `DensityOverlay(group_by=...)` name frozen run
  cell fields. A name that is neither a frozen field nor, for `color_by`, a gene raises `KeyError`
  listing the run's cell fields.
- A gene name or `FeatureRef` reads the live counts of the run's assay, `run.assay`, over the run's
  cells and normalizes them with the assay's live normalizer. A multimodal run's assay is its first
  source run's assay. A run freezes no gene values, so a gene color requires an explicit
  `normalization=NormalizationSpec(...)`, which provenance records; without one it raises
  `ValueError`.
- `cell_key` other than `"I"`, a `from_assay` other than the run's assay, and `point_sizes` raise
  `ValueError`. For per-cell sizes, pass `layout=run["umap"]` with sizes in that layout's cell
  order.

Run plots record the run as `provenance.extras["run"]`, a mapping with `runId` and `label`.
`layout_key=` remains the live-metadata source. Mixed source modes are rejected. For large
continuous fields, `ds.plots.embedding_raster(run=run, layout="umap",
color_by="doublet_score")` uses the same frozen run selection blockwise; its `color_by` (a string
or `CellField`) and `subset_by` name frozen run cell fields.

Granular workflows pass exact refs to the same datastore-owned plotting surface:

| Plot | Explicit artifact inputs |
|---|---|
| embedding | `layout=embedding_ref`, optionally `color_by=cluster_ref` |
| dot or matrix plot | `groups=cluster_ref` |
| composition | `categories=cluster_ref` |
| distribution | `grouping=cluster_ref` |
| cluster connectivity | `graph=graph_ref`, `groups=cluster_ref`, `layout=embedding_ref` |
| modality weights | `graph=wnn_graph_ref`, `layout=embedding_ref` |
| Paris hierarchy | `graph=graph_ref`, `clusters=paris_ref` |
| marker heatmap | `marker=marker_ref` |
| pseudotime heatmap | `aggregation=aggregation_ref` |
| lineage heatmap and trends | `aggregation=lineage_aggregation_ref` |
| lineage curves | `curves=curves_ref`, `layout=embedding_ref` |
| lineage tree | `tree=tree_ref`, `layout=embedding_ref` |
| mapping score | `mapping_score(result_ref, reference=reference, layout=embedding_ref)` |
| label-transfer evidence, confusion, and calibration | `mapping_evidence(transfer_ref)`, `mapping_confusion(transfer_ref, known_labels=...)`, `mapping_calibration(transfer_ref, known_labels=...)` |
| guide calls of a CRISPR screen | `guide_calls(calls_ref)` |
| fitted guide thresholds of a CRISPR screen | `guide_thresholds(calls_ref)`, or `guide_thresholds(calls_ref, guides=[...])` for one panel per guide and sample |
| perturbation efficiency of a CRISPR screen | `efficiency(efficiency_ref)` |
| E-distance of each perturbation from the control cells | `edistance(edistance_ref)` |
| Mixscape's knocked-out and escaping cells of each target | `mixscape(classes_ref)`, or `mixscape(classes_ref, panel="scores")` for each target's scores and fitted distributions |
| fold change and p-value of each tested pair of a target and a gene | `association_volcano(tests_ref)`, or `association_volcano(tests_ref, targets=[...])` for the pairs of some targets |
| calibration of the association tests on negative controls, and their power on positive controls | `association_calibration(tests_ref)` |
| clusters of perturbations by their profiles | `perturbation_clusters(clusters_ref)` for the heatmap of their correlations in leaf order, or `perturbation_clusters(clusters_ref, panel="embedding", label=[...])` for their layout with chosen perturbations named |
| clones per clone size and the largest clones, of clones or clonotypes | `clone_abundance(clones_ref)`, or `clone_abundance(clones_ref, group_by="sample")` for one line per group, and per receptor type for clonotypes |
| expansion categories and the clonal space of each group of cells | `clonal_expansion(expansion_ref)`, or `clonal_expansion(expansion_ref, group_by="cell_type")` to regroup the cells |
| rarefaction and extrapolation curves of the Hill numbers of each group | `clonal_rarefaction(ds.clones.rarefaction(clones_ref, group_by="sample"))`, or `base="coverage"` against the sample coverage |
| diversity indices of each group, one panel of bars per index | `clonal_diversity(ds.clones.diversity(clones_ref, group_by="sample"))`, or `indices=["chao1", "gini"]` for other indices |
| clones shared between samples or cell types | `clone_overlap(ds.clones.overlap(clones_ref, group_by="cell_type"))`, or `index="jaccard"` for another index |
| chosen clones across time points or cell types | `clone_tracking(ds.clones.track(clones_ref, group_by="time_point"), order=[...])` |
| fate coupling of every pair of groups with the tree of their order | `clone_coupling(ds.clones.coupling(clones_ref, group_by="cell_type"))`, or `value="z_score"` for the z-scores against shuffled labels |
| each clone's late cells by fate, by dominant fate | `clone_fate_bias(bias_ref)`, or `normalize=True` for shares |
| barcodes per cell, UMIs per barcode and reads per molecule of imported lineage barcodes | `barcode_calls(barcodes_ref)`, or `barcode_calls(barcodes_ref, max_barcodes_per_cell=5)` to gather cells of five barcodes or more in the last bar |
| the fraction of cells that each site of a lineage recorder reads as marked, unknown, unedited or missing | `recorder_states(alleles_ref)`, or `recorder_states(alleles_ref, max_sites=20)` to draw the 20 sites read in the most cells |
| a cell phylogeny with the clones cut from it, its clades collapsed to at most 100 wedges | `phylogeny(clones_ref)` of clones from `ds.clones.phylogeny_clones`, or `phylogeny(clones_ref, max_leaves=1_000, tree="E7.5-R1-C1")` for more wedges of one tree |
| receptor types, chain pairings and chains per cell of imported immune receptors | `receptor_qc(receptors_ref)`, or `receptor_qc(receptors_ref, group_by="sample")` for one bar per group |
| V, D, J or constant gene usage of each locus | `gene_usage(ds.repertoire.gene_usage(receptors_ref, group_by="sample"))`, or the usage of `genes="j_call"` or `level="family"` |
| V-J gene pairing of each locus | `gene_pairing(ds.repertoire.gene_usage(receptors_ref, genes=("v_call", "j_call"), cell_selection=t_cells))` |
| junction lengths of each locus | `spectratype(ds.repertoire.spectratype(receptors_ref, group_by="sample"))`, or the lengths of `sequence="nt"` |
| depth of variant counts at each site and in each cell | `variant_depth(counts_ref)`, or `variant_depth(counts_ref, group_by="sample")` for one histogram per group |
| each variant's variance-mean ratio against its strand correlation, informative variants marked | `variant_statistics(selection_ref)` |
| each cell's allele frequency of each informative variant, uncovered entries apart | `allele_frequencies(selection_ref)`, or `allele_frequencies(selection_ref, clones=clones_ref)` for cells ordered by clone |
| the fraction of cells with each genotype call at each site, by group, and the calls of one site on an embedding | `genotypes(genotypes_ref, group_by="cell_type")`, or `genotypes(genotypes_ref, sites=["CALR"], layout=embedding_ref)` to color the cells of an embedding of the genotypes' cells by their call at CALR |

An embedding colored by an artifact, `embedding(layout=embedding_ref, color_by=labels_ref)`, reads the artifact's values for the layout's cells, so the artifact can hold more cells than the layout: labels computed on every cell color a run's layout of fewer cells, and keep the colors that a layout of all their cells gives them. An artifact that holds no value for some cells of the layout raises `ValueError`; compute it on cells that include the layout's, such as the run's `analysis_cell_selection`.

Labels beside another artifact are read the same way: `composition(categories=cluster_ref, grouping=sample_labels_ref)` reads the grouping for the cells of `categories`, and `cluster_connectivity` reads `groups` for the graph's cells, so labels of more cells serve, and labels that miss some of those cells raise `ValueError` that names the cells to compute them on. A layout must hold the cells that it draws: those of the graph in `cluster_connectivity` and `modality_weights`, of the trajectory in `lineage_curves` and `lineage_tree`, and of the genotypes in `genotypes`. Any cell selection of those cells serves, whichever operation made it, such as one that `select_cells` makes of a run's cells; a layout of other cells raises `ValueError`.

`ds.clusters.connectivity(graph, groups)` returns the weights that `cluster_connectivity` draws as `nodes` and `edges` tables, without a layout, a figure or the plotting dependencies ({doc}`datastore`). `ds.trajectory.curves_in_layout(curves, layout)` returns the points that `lineage_curves` draws, the same way.

A figure of a clone or repertoire summary takes the record that the summary returns, such as the `CloneOverlap` of `ds.clones.overlap`, so the summary is computed once, can be read before it is drawn, and sets the figure's groups and cells; from the store it reads only what it calls the clones and the labels that its groups' colors range over. A figure of a saved result takes its artifact. Figures that split cells by group take `group_by`, one cell metadata column or several, and `groups`, a label artifact, alone or together, and group the cells as the clone summaries do: by the combinations of their labels, named by the labels joined by commas. A limit on what a figure draws is named for the items it limits, such as `max_clones`, `max_genes` and `max_barcodes_per_cell`.

The figures of clones, `clone_abundance`, `clonal_expansion`, `clonal_rarefaction`, `clonal_diversity`, `clone_overlap`, `clone_tracking`, `clone_coupling` and `clone_fate_bias`, those of immune receptors and lineage records, `receptor_qc`, `gene_usage`, `gene_pairing`, `spectratype`, `barcode_calls`, `recorder_states` and `phylogeny`, and those of variants, `variant_depth`, `variant_statistics`, `allele_frequencies` and `genotypes`, keep one set of conventions:

- Names. Titles, axes and legends name the clones as their artifact does, clones, clonotypes or clonotype clusters, such as "Clonotype sizes" and "Clonotypes across groups". Clonotypes are labeled by their representative's junction, with their number where two drawn share one, and tables that list clones hold `clone`, the ID, beside `label`, the text drawn.
- Group colors. A grouping by one cell metadata column takes the column's stored display palette, or colors assigned over every label of the column, in their stored order for a column stored as ordered, so a figure of some cells colors each group as a figure of all of them does. A label artifact's groups take the categories and colors that the artifact records, as imported clusters keep them, or else colors over every label of the artifact, receptor types one palette of their own, and groups of several columns colors over the groups drawn. Only a figure that colors its groups reads those labels: a column's a stored chunk at a time, and a label artifact's within the store's memory budget.
- Vocabularies. Receptor types, chain pairings, the VJ and VDJ arms of a receptor, loci, the states of a recorder site and genotype calls each take one palette, in every figure that draws them. No value, such as `"no IR"`, a site without a read, a genotype call of no evidence or an allele frequency of a cell not covered, takes the scales' missing gray, and what carries nothing, such as an unedited site, a filtered call or a variant that is not informative, a muted gray. Legends name values in words, such as "Unknown mark" or "Wild type", and tables keep their tokens, such as `unknown_mark` or `wild_type`. The libraries of lineage barcodes take colors over every library of their artifact.
- Heatmaps. `clone_overlap` and `clone_coupling` take `color_scale`, a `ColorScale` that sets the color map and limits, so two figures can share them. By default overlap indices and couplings of at least 0 take viridis over the values drawn, and z-scores and couplings below 0 take RdBu_r centered on 0. The diagonal is blank, and a pair that was not compared, or whose z-score is not finite, takes the scale's missing color. The locus panels of `gene_usage` and `gene_pairing` share one color bar, from 0 to the largest fraction of any locus, so a color means one fraction in every panel, and `allele_frequencies` colors frequencies from 0 to 1.
- Legends. Each legend is titled and stands beside its panel, a grouping's legend titled by its columns joined by commas; panels that share colors, such as a rarefaction's orders or a spectratype's loci, share one legend beside their first row, the calls of `genotypes` one legend beside its last panel, and a figure without a grouping, whose cells form the one group "all cells", draws none of its groups. A legend lists at most 80 entries, in up to four columns that fit the row of its panel, and says how many of how many it lists when it leaves some out, such as "Clone (80 of 120)". Every legend is reported in `PlotResult.legends` with its scale in `PlotResult.scales`, with `show_legend=False` too, so `compose_results` draws a legend that several figures share once.
- Bars and axes. A stacked bar draws its first part at the left, or at the top of a vertical bar, in its legend's order, and fractions span the axis from 0 to 1; parts that make no whole, such as the chains of a receptor's two arms, stand side by side. A part of a whole is a fraction, such as "Fraction of the site's cells". An axis of bars or of a heatmap names as many of its items as its length holds once the figure is laid out, at most 40: every one, or else every second, third or nth from the first, so that names never run into each other. A heatmap prints its values when it has at most 12 rows and columns, and the tables hold every item. A log axis reads plain numbers: whole numbers on an axis of counts, and numbers such as 0.5 on one of mean depths or variance-mean ratios. A panel that leaves items out says so in its title or on its axis, such as "The 40 of 312 clones with the most late cells", "V gene (30 of 52)" or "Cells (2,000 of 5,000)", or "Largest 20 clonotypes" for the largest clones.
- Sizes. A figure is at most a page wide, 7.5 inches. A panel that grows with its items grows for each item it names, small multiples such as the orders of a rarefaction or the loci of gene usage fill rows of three, heatmaps of loci rows of two, and a row too wide for the page shrinks its panels to share it.
- Nothing to draw. A figure without anything to draw raises `ValueError`, and a panel beside others that draw, such as one of `clonal_diversity` whose index is defined for no group, one of `barcode_calls` that no library fills or the cells panel of `variant_depth` whose grouped cells have no depth, says so.
- Provenance. `notes` holds the plot's name, and `extras` holds the reference of each artifact drawn: `clones` in the figures of clones, any other artifact that the figure takes under its argument's name, such as `expansion`, `bias`, `receptors`, `barcodes`, `alleles`, `counts`, `variants` or `genotypes`, and `source`, the receptors or clonotypes that gene usage and junction lengths count. It holds the settings under their names, and the `grouping` of a figure that groups cells, its columns joined by commas.

`layout_key` and string forms on plotters that still accept them refer to deliberate live metadata
inputs. Distribution grouping instead requires either an exact categorical artifact or an explicit
`CellField`, with `cell_selection=` when a frozen metadata subset is intended.

Dot plots and matrix plots name the grouping columns of their tables by role, whatever the grouping
columns are called: `group` holds the first `group_by` key's values or the `groups=` labels,
`subgroup` the second key's values, and `sample` the `sample_by` value. They are followed by
`feature`, `feature_group`, `mean`, `fraction`, `n_cells` and `variance`, and an aggregate over
samples adds `n_samples`. A grouping column can therefore be named `feature`, `mean` or `sample`,
and `provenance.extras["group_by"]` keeps the source column names. With `standardize="feature"`,
`mean` stays raw and a `zscore` column of `tables["aggregate"]` holds the plotted values, while
`tables["per_sample"]` keeps the per-sample means without it; `provenance.extras["color_values"]`
names the plotted column. A matrix plot's `tables["matrix"]` is the drawn matrix, like the matrix
of `marker_heatmap`: one row per feature (index `feature`) and one column per group label (columns
`group`). `marker_heatmap`'s matrix holds the standardized values of its marker features, one
column per group of the marker search. The annotation tables of both are indexed by the same
labels.

Default category colors are assigned over every value of the colored column or label artifact, not only over the plotted cells, so `groups`, `subset_by`, or a `cell_key` subset never recolors a category: Alpha keeps its color in an embedding, a violin, and a composition of any of its subsets. Categories take their natural order, so `"2"` precedes `"10"`.

A column's stored display sets the order, colors and labels of its categories instead, such as the categories and `uns["<column>_colors"]` that an H5AD import keeps. The clusters that `cluster_keys` imports record a display on their label artifact, which plots read in the same way, and a run records the display of each column that `snapshot_columns` freezes, so run plots draw the column as plots of the live column do. A column stored as ordered and without a display, a categorical column whose categories are ordered or a column whose `levels` attribute comes with `ordered=True`, keeps its categories in their stored order, and default colors follow that order. A `CategoricalScale` sets only the fields it names, and the others keep the stored values: `labels` alone renames categories in their stored order and colors, `order` alone keeps the stored colors, `palette` alone keeps the stored order, and `palette_name` replaces the stored colors in the stored order. Labels replace the stored labels category by category. A `palette` names every category that the plot shows. Requested `groups` order an embedding's legend unless the scale sets `order`. Tick and legend labels in every plot, including violins, histograms, dot plots, and matrix plots, show the resulting labels.

Embeddings draw categorical colors, and continuous colors they do not sort, in a seeded shuffle, so
cells stored last, such as the last sample of a merged store, do not cover the others. Each cell
draws at its rank in `numpy.random.default_rng(seed).permutation(n)` over the `n` cells of the
layout, so the order is reproducible and the same in every panel; `seed=None` keeps the stored
order. Gene and other feature colors draw high values on top by default, as in scanpy;
`sort_values=False` shuffles them too, and `sort_values=True` also sorts metadata colors.
`provenance.extras["draw_order"]` records each color's order. On-data labels start at their
cluster's median and, once the layout is final, move apart by at most about one label height
without coming closer to another cluster's median than to their own. Side legends list the
categories that fit beside the figure, which grows to hold them when CyteArc sizes it. Embeddings
settle their layout and point sizes without rendering the points, so the scatter is drawn only when
the figure is shown or saved.

`missing_color` replaces the missing color of whichever scale an embedding uses, and a log
`ColorScale` draws values that are not positive in that color instead of raising. `clip_fraction`
takes its quantiles from the drawn cells. Cells without a `facet_by` value get their own facet,
titled with the facet column's missing label, and facets wrap at four columns, each color or layout
starting a new row. `embedding_raster` paints a pixel whose cells all lack a value in the missing
color, while empty pixels show the axes background.

Violins scale every body, including each half of a split violin, to the full width, as in scanpy.
A violin estimates its density from at most `max_density_points` values, a seeded sample that keeps
its smallest and largest value. Stacked compositions draw the first category on top, in the
legend's order. With `swap_axes=True`, dot plots list the first group at the top. A clustered matrix
plot draws the dendrogram of each clustered axis beside the matrix, and a mapping of feature groups
draws one bracketed row per feature and group, indexing `tables["matrix"]` by `feature_group` and
`feature`.

Plots treat rows that a nullable metadata column or label artifact flags in its linked missing mask
as missing, never as their stored placeholder. Dot plots and matrix plots leave those cells out of
every group and report their number as `dropped_group_cells`. Composition plots count a missing
category in the NA category and leave cells with a missing sample label out of every sample.
Embeddings and cluster trees show missing colors or fill values as missing, and
`cluster_connectivity` rejects missing group labels and coordinates. An embedding colored by a label artifact also draws a blank label, such as that of a cell without a clone or clonotype, in the missing color rather than as a category. The legends of embeddings, compositions, `perturbation_clusters` and the annotation strips of `matrixplot` and `marker_heatmap` list the values drawn in the missing color last, under the scale's missing label, such as "NA". The legends that `compose_results` draws for them list that entry too, and a legend that several figures share lists it when any of them draws such values.

A feature has no value in a cell that its assay did not measure, which the assay's membership
column `<assay>_I` marks False ([Assay membership](../developers/zarr-internals#assay-membership)); such a cell holds zero counts, which are no
measurement. Plots read only the measured cells and normalize only them, so a normalizer fitted over
the cells that it reads, such as ATAC TF-IDF, gives a measured cell the value of a read of the
measured cells alone. Embeddings draw an unmeasured cell in the scale's `missing_color` and leave it
out of the color limits, and `provenance.extras["color_limits"]` holds None for a color or panel
without a value among the drawn cells. Distribution plots leave it out of the feature's panel and
table, and a feature panel whose assay measured none of the selected cells raises `ValueError` that
names the membership column. Dot plots and matrix plots compute `mean`, `fraction`, and `variance`
over the cells of a group with a value, which `n_cells` counts, so `fraction` is the share of
measured cells above `expression_cutoff`; a group without a measured cell keeps its row with
`n_cells` 0 and NaN statistics and draws no dot or a missing matrix cell, and an aggregate over
samples skips such a sample in every statistic and in `n_samples`. Cluster-tree fill values and the
group means of `marker_heatmap` leave out unmeasured cells. Dot plots, matrix plots, embeddings, distribution plots, and
`marker_heatmap` record `provenance.extras["unmeasured_cells"]`, a mapping from each feature assay
to the number of plotted cells that it did not measure, when that number is not zero.

WNN is the default for `DataStore.integration.modalities`. Its integrated graph stores one weight per input assay and cell. Use `ds.plots.modality_weights(graph=wnn_graph, layout=embedding)` or {py:func}`cytearc.plotting.modality_weights` to show those weights over an explicit embedding of the graph's cells. Explicit SNN graphs do not contain modality weights.

Store-backed plotters and diagnostics generally return a `PlotResult` and render by default with `show=True`.
Pass `show=False` before accessing, saving, or reusing an owned figure.
In Jupyter, `PlotResult.show()` displays the figure and then closes it in pyplot, as `plt.show()` does, so the cell draws it once; this includes a figure passed with `target=` or composed with `compose_results`, which can still be saved or drawn on.
`run_recipe` returns a `PlotRecipeResult` and defaults to `show=False`.
Helpers differ: `label_panels` returns `None`, `theme_context` is an iterator, and `compose_results` returns a `PlotResult` without a `show` parameter.

```{eval-rst}
.. automodule:: cytearc.plotting
    :members: embedding, embedding_raster, dotplot, matrixplot, modality_weights, composition, distribution, cluster_connectivity, mapping_score, mapping_evidence, mapping_confusion, mapping_calibration, guide_calls, guide_thresholds, efficiency, edistance, mixscape, association_volcano, association_calibration, perturbation_clusters, clone_abundance, clonal_expansion, clonal_rarefaction, clonal_diversity, clone_overlap, clone_tracking, clone_coupling, clone_fate_bias, barcode_calls, recorder_states, phylogeny, receptor_qc, gene_usage, gene_pairing, spectratype, variant_depth, variant_statistics, allele_frequencies, genotypes, qc, graph_qc, elbow, highly_variable_features, label_panels, compose_results, theme_context, marker_heatmap, cluster_tree, pseudotime_heatmap, lineage_heatmap, lineage_trends, lineage_curves, lineage_tree, run_recipe
    :imported-members:
    :undoc-members:
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.PlotResult
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.FeatureRef
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.CellField
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.StudyDesign
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.NormalizationSpec
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.ColorScale
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.CategoricalScale
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.SizeScale
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.DensityOverlay
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.Highlight
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.PlotRecipe
    :members:

.. autoclass:: cytearc.plotting.PlotStep
    :members:

.. autoclass:: cytearc.plotting.PlotPanelTarget
    :members:

.. autoclass:: cytearc.plotting.PlotOutputSettings
    :members:

.. autoclass:: cytearc.plotting.PlotOutput
    :members:

.. autoclass:: cytearc.plotting.PlotRecipeResult
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.PlotProvenance
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.plotting.LegendSpec
    :members:
```

```{eval-rst}
.. py:data:: THEMES

    Registry of built-in plotting theme definitions.
```
