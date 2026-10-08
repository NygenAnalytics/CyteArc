# Integration and metrics API reference

Prefer the `DataStore` methods below in analysis code.

`integration.modalities` accepts two or more exact source refs. WNN is the default; it consumes neighbour
artifacts and follows each one's named native reduction or batch-correction coordinates. SNN is
explicit and consumes connectivity-map artifacts. The returned integrated-graph reference is the
exact downstream `graph` argument.
WNN does not accept imported-coordinate ancestry. `coordinates` is a provenance input on neighbour
and ANN artifacts, not a separate artifact kind.

```python
wnn = ds.integration.modalities([rna_neighbors, adt_neighbors])
snn = ds.integration.modalities([rna_graph, adt_graph], method="snn")
```

WNN also rejects neighbours built on normalized values, as a pipeline run with `pca_dims=0`
builds them, because it weighs each assay by its reduced coordinates; SNN integrates such graphs.

Integration holds its inputs in memory: SNN every source graph with a float64 matrix of
shared-neighbour fractions per graph, about `16 * n_graphs + 20` bytes per edge, and WNN every
assay's neighbours and coordinates with the integrated graph and its modality weights. Both
compare that estimate with the datastore's memory budget before they load a graph or coordinate
and raise `MemoryError` over it, naming the bytes, the cells, the neighbours, the dimensions, and
the limit. `reduction.harmony` does the same before it reads coordinates: it holds the float64
coordinates several times over and float64 matrices of soft cluster assignments, about 5 KB per
cell with 30 dimensions and 100 clusters, so `nclust` in `harmony_params` and the dimensions set
its memory. Its admission also counts the batch labels, which it reads before it admits the fit and
holds until the fit ends, at the size of their text objects: about 60 bytes per cell for each batch
column of short labels. These estimates are near-exact counts of what each step allocates, as
traced while it runs, plus stated allowances; the budget compares them with its limit, and they
are not a cap on the memory of the process (see [Memory and execution](../concepts/memory-and-execution)).

The default WNN artifact stores one per-cell weight for each input assay. Plot those values with
{py:func}`cytearc.plotting.modality_weights` or the bound
`ds.plots.modality_weights(graph=wnn, layout=wnn_layout)` accessor. An explicit SNN artifact does
not contain modality weights.

Neighbour-based metrics require `neighbors`; graph metrics require `graph`; reduction metrics
require their exact coordinate artifact. Batch-mixing methods and biological-annotation cLISI or
graph connectivity intentionally read imported metadata columns over the rows selected by artifact
lineage. They are not an indirect path for scoring CyteArc-produced clusterings. Concordance instead
accepts two exact clustering refs with the same frozen cell selection. None accepts a storage path
or an omitted artifact input.

```python
mixing = ds.integration.compute_ilisi("batch", rna_neighbors)
connectivity = ds.integration.compute_graph_connectivity("cell_type", snn)
agreement = ds.clusters.compute_concordance(rna_clusters, adt_clusters, metric="ari")
```

The metric methods return dataset-level scalar summaries such as iLISI, cLISI, and graph
connectivity. {py:func}`cytearc.metrics.compute_lisi` returns per-cell LISI values from KNN arrays.
Metric label columns must label every scored cell: masked, `NaN`, `None`, and blank labels raise
`ValueError`.

All LISI entry points resolve `perplexity=None` to `floor(k / 3)`, where `k` is the
number of neighbours. Explicit values are capped at `k / 3`
with a warning when the graph is too small. With three to five neighbours, the default
is 1, so scores can approach 1 when the nearest distance is unique. Tied distances can
produce larger values.

Cluster separability summaries expose `macro_f1_fold_sd`, the sample standard deviation
of macro F1 across validation folds. Interpret it as fold variability.

## DataStore methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   cytearc.datastore.namespaces.IntegrationAccessor.modalities
   cytearc.datastore.namespaces.IntegrationAccessor.compute_ilisi
   cytearc.datastore.namespaces.IntegrationAccessor.compute_clisi
   cytearc.datastore.namespaces.IntegrationAccessor.compute_batch_mixing
   cytearc.datastore.namespaces.IntegrationAccessor.compute_graph_connectivity
   cytearc.datastore.namespaces.IntegrationAccessor.compute_graph_silhouette
   cytearc.datastore.namespaces.ClustersAccessor.compute_concordance
   cytearc.datastore.namespaces.IntegrationAccessor.compute_cluster_separability
```

```{eval-rst}
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.modalities
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.compute_ilisi
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.compute_clisi
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.compute_batch_mixing
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.compute_graph_connectivity
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.compute_graph_silhouette
.. automethod:: cytearc.datastore.namespaces.ClustersAccessor.compute_concordance
.. automethod:: cytearc.datastore.namespaces.IntegrationAccessor.compute_cluster_separability
```

## Harmony

```{eval-rst}
.. autofunction:: cytearc.embeddings.fit_harmony
```

```{eval-rst}
.. autoclass:: cytearc.embeddings.HarmonyResult
    :members:
```

## Metrics

```{eval-rst}
.. automodule:: cytearc.metrics
    :members: compute_lisi, ilisi_knn, clisi_knn, graph_connectivity, silhouette_scoring, label_concordance_score, lisi_batch_mixing_score, ClusterSeparabilityResult
    :imported-members:
```

CyteArc's iLISI and cLISI use the scIB median and scaling definitions over CyteArc's self-free persisted KNN arrays.
Graph connectivity follows the original scIB symmetrized-graph definition.
YosefLab `scib-metrics` currently uses directed strong components for graph connectivity, so those values need not match.
See Luecken et al. 2022, doi: 10.1038/s41592-021-01336-8.
