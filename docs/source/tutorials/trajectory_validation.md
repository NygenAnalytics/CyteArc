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

## Count the cells that were scored

```python
# Load pseudotime scores and their validity mask.
pseudotime = ds.trajectory.load_pseudotime(pseudotime_ref)

# Report the fraction of cells with valid pseudotime scores.
float(pseudotime.valid.mean())
```

By default, pseudotime uses the largest connected graph component. Other cells receive
undefined values and `valid=False`. A separate component may reflect biology, filtering,
or graph settings. Report how many cells were excluded and investigate where they lie.
Use the validity mask for summaries you write yourself; CyteArc applies it internally
when searching for pseudotime markers or aggregating expression.

The mask tells us which cells the result actually describes. A high fraction alone is
not sufficient: a small excluded component could contain an entire rare population.
Including undefined values in custom summaries or exports can also distort the result.

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
