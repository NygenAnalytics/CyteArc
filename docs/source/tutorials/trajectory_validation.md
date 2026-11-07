---
description: Check trajectory assumptions, changing genes, and candidate fate probabilities.
---
(trajectory_validation)=

# Trajectory validation checklist

A plausible UMAP is only a starting point. Use these checks after [](pseudotime.ipynb),
[](expression_dynamics.ipynb), or [](fate_mapping.ipynb) to decide which conclusions the result
supports. The snippets below continue those examples; this page does not run a separate
analysis.

## Check the biological orientation

Ask whether known early markers occur toward the start and terminal markers toward the
end. Use genes or other evidence beyond the markers that defined the endpoints: checking
only the populations used to orient the graph is circular. Intermediate populations
should be allowed intermediate scores. Also check whether sequencing depth, mitochondrial
read percentage, batch, or another process could explain the ordering.

Try plausible alternative endpoints and modest graph changes. If they give different
orderings, describe that uncertainty instead of treating one smooth map as a unique
answer. The custom source/sink weighting in the pancreas examples is part of their
assumptions; changing it can change the scores.

Compare these alternatives visually as well as numerically. A smooth-looking ordering
cannot by itself distinguish biological progression from a technical gradient.

A direction score that orients `fate` in place of a pseudotime needs the same checks. Its higher values must mark later cells, so reverse a score that is highest at the start, such as a potency score. The map leaves out the graph components that the score does not cover and those without a sink, so compare the `valid` mask that `load_fate` returns with the cells you meant to map.

## Count the cells that were scored

```python
# Load pseudotime scores and their validity mask.
pseudotime = ds.trajectory.load_pseudotime(pseudotime_ref)

# Report the fraction of cells with valid pseudotime scores.
float(pseudotime.valid.mean())

# List the cell counts of the graph's connected components, largest first.
pseudotime.component_sizes
```

By default, pseudotime uses the largest connected graph component. Other cells receive undefined values and `valid=False`. `component` gives each cell's component, numbered as `component_sizes` lists them, so you can find the excluded cells without recomputing the graph's components. A separate component may reflect biology, filtering, or graph settings. Report how many cells were excluded and investigate where they lie. When the sources lie in a smaller component, `component_policy="sources"` scores the component that holds them instead. Use the validity mask for summaries you write yourself; CyteArc applies it internally when searching for pseudotime markers or aggregating expression.

The mask tells us which cells the result actually describes. A high fraction alone is
not sufficient: a small excluded component could contain an entire rare population.
Including undefined values in custom summaries or exports can also distort the result.

## Check a trajectory of some groups

To follow a process through some groups of a large graph, score their cells on the graph you have instead of building a new one:

```python
# Select the cells of the groups the process runs through.
groups_ref = ds.select_cells(clusters_ref, include=[2, 3, 5])

# Score them on the subgraph that they induce in the existing graph.
pseudotime_ref = ds.trajectory.pseudotime(
    graph_ref, cells=groups_ref, source_sink=clusters_ref, sources=[2], sinks=[5]
)
```

The subgraph keeps the edges that the whole graph chose among these cells, so the order follows the variation of the whole dataset within the groups. Cells at the edge of the groups lose their edges to cells outside them, and the subgraph can split where the whole graph does not; read `component_sizes` before reading the scores. A graph built on the groups' own variable genes and principal components follows the variation within the groups more closely, at the cost of normalizing, reducing and finding neighbours again. When both are feasible, compare their orderings: an order that only the rebuilt graph shows rests on variation that the whole-data reduction does not resolve.

## Inspect changing genes and modules

```python
# Load the marker table and adjusted p-values.
markers = ds.trajectory.load_markers(marker_ref)

# Inspect the first few rows of the result.
markers.table.head()
```

Inspect the expression profiles with the module plots in [](expression_dynamics.ipynb).
Load the aggregation result when you need its values for further checks:

```python
# Load the saved aggregation of changing genes.
modules = ds.trajectory.load_aggregation(modules_ref)

# Inspect the loaded aggregation result.
modules
```

Correlations can miss transient or branch-specific changes. Inspect the profiles and
several genes in a module, not just its name or size. Compare smoothing widths and
feature choices. Module numbers are identifiers, not developmental stages.

Untested marker features have missing p-values, and correction covers tested features
only. The aggregation result contains features that passed its expression and variance
checks. Neither result establishes that a gene causes the process.

## Check lineage-resolved results

[](lineage_dynamics.ipynb) weighs each cell in each lineage by its fate probability. Check how
many equally weighted cells each lineage amounts to before comparing their statistics:

```python
# Load the per-lineage correlations with pseudotime.
lineage_markers = ds.trajectory.load_lineage_markers(lineage_ref)

# Show each lineage's effective number of cells.
lineage_markers.effective_cells
```

A lineage with few effective cells has noisy profiles and wide intervals. Shared trunk cells
count in every lineage, so a lineage's strongest correlations often belong to all lineages;
compare lineages before calling a gene fate-specific. Lineage drivers leave the endpoint cells
out by default, because their probabilities are fixed by the model; including them mostly
recovers the markers that defined the endpoints.

Candidate endpoints from a cluster tree, as in [](lineage_curves.ipynb), depend on the
clustering and can chain similar populations. Check them with markers, and compare the fate
maps of plausible endpoint sets, before treating a lineage as established.

Fate probabilities sum to one over the endpoints, so a cell that lies on no trajectory still splits its probability among them. Call which lineages each cell follows from the lineage curves before reading a cell's lineage from its largest probability:

```python
# Call each cell's lineage, "shared" or "none", with the rule saved.
membership_ref = ds.trajectory.lineage_membership(curves_ref)

# Load the labels and where each pair of lineages parts.
membership = ds.trajectory.load_lineage_membership(membership_ref)
membership.branch_points

# Test the genes that separate two endpoints where they part.
branch_ref = ds.trajectory.branch_cells(membership_ref, [2, 3])
drivers_ref = ds.trajectory.drivers(fate_ref, features=features_ref, cells=branch_ref)
```

Cells called `"none"` lie farther from every curve than its own cells do at the same progress; count them and look at where they lie before excluding them, since a population off the trajectory may be the biology. A pair whose branch point is NaN never shares a cell or never parts. The branch point is where the two lineages' fate probabilities leave the `band`, so it moves with `band`; compare a few values when the exact place matters.

## Separate probability checks from biological evidence

```python
# Load the saved fate probabilities and validity mask.
fate = ds.trajectory.load_fate(fate_ref)

# Restrict probability checks to scored cells.
valid_probabilities = fate.values[fate.valid]

# Measure the largest deviation from a row sum of one.
row_sum_error = abs(valid_probabilities.sum(axis=1) - 1.0).max()

# Show the largest probability row-sum error.
row_sum_error
```

Valid rows should be finite, non-negative, and sum to one. Probabilities at selected
sinks are fixed by the model, so high values there check the calculation rather than
validate the biology. Ignore invalid rows when interpreting probabilities.

For biological support, use independent endpoint evidence and compare plausible sink
sets. Include the possibility that a real outcome is missing. A smooth gradient can
reflect graph geometry even when the underlying biological decision is abrupt. Numerical
convergence and a gradual change in probability do not establish that a biological fate
decision is gradual or reversible.

Lineage tracing gives independent endpoint evidence: [](lineage_tracing.ipynb) scores fate probabilities against the fates that the clones of hematopoietic progenitors took.

## Keep the inputs with the results

Record which graph, cells, features, and endpoints produced the result. Saved references
let you check the inputs and how they connect:

```python
# Inspect the saved result's parameters and inputs.
ds.artifacts.inspect(fate_ref)

# Display the saved result and its upstream inputs.
ds.artifacts.lineage(fate_ref)
```

The trajectory results should refer to the intended graph and cell selection. Marker
and module results should use the intended feature selection and pseudotime result.
Keep alternatives separate when comparing graph or endpoint changes; see
[](reuse_and_tracing.ipynb) for following saved analyses.

When reporting a trajectory, include the endpoint evidence, graph and smoothing
choices, valid-cell counts, sensitivity checks, and remaining uncertainty alongside
the saved result references.
