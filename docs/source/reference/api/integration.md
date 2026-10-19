---
myst:
  html_meta:
    description: Integrate CyteArc modalities with WNN or SNN, correct batches with Harmony, and evaluate integration metrics.
---

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
the limit. `reduction.harmony` does the same before it reads coordinates, with the estimate of
{py:func}`~cytearc.embeddings.harmony.harmony_peak_bytes`: it holds the float64 coordinates three
times, as its input, its original copy, and its corrected copy, and float64 soft cluster
assignments with `n_clusters` values per cell, about 1.5 KB per cell with 30 dimensions and 100
clusters, plus per-level regression sums of `levels x n_clusters x dims` floats, so `n_clusters`
in `harmony_params` and the dimensions set its memory. Its admission also counts the batch
labels, which it reads before it admits the fit and holds until the fit ends, at the size of their
text objects: about 60 bytes per cell for each batch column of short labels. These estimates are
near-exact counts of what each step allocates, as traced while it runs, plus stated allowances;
the budget compares them with its limit, and they are not a cap on the memory of the process (see
[Memory and execution](../concepts/memory-and-execution)).

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

`ds.reduction.harmony(reduction, batch_columns, harmony_params=...)` corrects reduced coordinates
for one or more batch columns with CyteArc's implementation of
[Harmony2](https://doi.org/10.64898/2026.03.16.711825) (Patikas et al. 2026), which builds on
[Harmony](https://doi.org/10.1038/s41592-019-0619-0) (Korsunsky et al. 2019). It was written from
the two papers and is not expected to reproduce other Harmony implementations number for number.

```python
corrected = ds.reduction.harmony(pca, ["batch"], harmony_params={"theta": 1.0, "n_clusters": 50})
```

A fit repeats rounds of two steps:

- Soft clustering. Each cell, scaled to unit length, is assigned softly to `n_clusters` clusters
  whose centroids start from k-means++ seeds drawn from the cells. `sigma` sets how soft the
  assignments are, and `theta` penalizes clusters whose mix of batch levels departs from that of
  all cells. The penalty takes Harmony2's stabilized form: when a batch level has no cells in a
  cluster, the pull of the cluster on that level's cells stays bounded, where in the 2019 form it
  grew with the cluster's size. Cells are updated in blocks of `block_size` of the cells, each
  block against batch counts that leave its own cells out.
- Correction. For every cluster, a ridge regression weighted by the cells' assignments to the
  cluster fits the original coordinates on an unpenalized intercept and one term per batch level.
  A cell's corrected coordinates are its original coordinates minus the terms of its own batch
  levels, summed over the clusters with its assignments as weights; the intercepts, which hold the
  clusters' positions, are not subtracted. By default the penalty of a level in a cluster is
  `ridge_alpha` times the level's expected mass there, its share of all cells times the cluster's
  total assignment, so a level with far less than its expected mass in a cluster gets almost no
  correction from that cluster. A level whose assignment mass in a cluster is below
  `prune_threshold` of its cells is left out of that cluster's regression.

Each round runs clustering passes until the objective changes by at most `epsilon_cluster` of its
value or `max_iter_cluster` passes have run, then corrects the coordinates. The fit has converged
when the objective changes between rounds by at most `epsilon_harmony` of its value twice in a
row, or, from the fifth round on, when it drifts by at most `epsilon_harmony` of its value per
round once its alternation from round to round is averaged out. The second test ends fits whose
clustering and correction settle into two states that take turns, so that the objective
alternates between two close values instead of repeating one; asking for two small changes in a
row keeps such an alternation from stopping a fit by chance. A fit stops as soon as it
converges, so `max_iter_harmony` only bounds fits that settle slowly; its default of 40 rounds,
where the 2019 paper used 10, leaves room for large, heterogeneous datasets. A fit that reaches
`max_iter_harmony` rounds, two or more, while its objective is still drifting stops there and
warns; raising `max_iter_harmony` gives it more rounds. With several batch columns, their
penalties add and each regression has one term per level of every column.

`harmony_params` takes the keyword options of {py:func}`~cytearc.embeddings.fit_harmony`, all but
`nthreads`, which the datastore sets:

| Key | Default | Meaning |
| --- | --- | --- |
| `n_clusters` | the cell count divided by 30, rounded half to even, between 1 and 100 | number of soft clusters; an explicit value lies between 1 and the cell count |
| `theta` | `2.0` | diversity penalty: one value, one per batch column, or one per batch level; finite and at least 0 |
| `sigma` | `0.1` | softness of the assignments: one value or one per cluster; finite and positive |
| `tau` | `0.0` | multiplies each level's `theta` by `1 - exp(-(level_cells / (n_clusters * tau))**2)`, which weakens the penalty of levels with few cells; 0 leaves `theta` unchanged |
| `ridge_penalty` | `None` | a fixed ridge penalty for every cluster: one value, one per batch column, or one per batch level; finite and positive. With several batch columns it must be at least 1e-6 times each cluster's assignment mass, or the fit raises `ValueError`, because smaller penalties cannot be solved accurately in float64; with one batch column any positive value works. `None` uses the dynamic penalty |
| `ridge_alpha` | `None`, which is `0.2` with the dynamic penalty | scale of the dynamic penalty; positive. Passing it with `ridge_penalty` raises `ValueError` |
| `prune_threshold` | `1e-5` | smallest assignment mass in a cluster, as a fraction of a level's cells, for the level to enter that cluster's regression; at least 0 and below 1. With 0, only levels without mass are left out |
| `block_size` | `0.05` | fraction of the cells updated together; above 0 and at most 1 |
| `max_iter_cluster` | `200` | most clustering passes per round; at least 1 |
| `max_iter_harmony` | `40` | most rounds of clustering and correction; at least 1. The fit stops as soon as it converges, so the limit only bounds slow fits; the 2019 paper used 10 |
| `epsilon_cluster` | `1e-5` | relative change of the objective between consecutive passes that ends a round's clustering, after at least two passes; at least 0 |
| `epsilon_harmony` | `1e-4` | relative change of the objective per round that ends the fit: the objective changed by at most this fraction of its value in each of the last two rounds (in the one round when only two have run), or, from five rounds on, the average of two consecutive round objectives changes by at most this fraction per round. The second test ends fits whose clustering and correction settle into a cycle that returns to nearly the same state every other round; at least 0 |
| `random_state` | `4466` | seed of the cell update order and the k-means++ seeds; an integer of at least 0 |

Values given per batch level follow design order: the batch columns in their order and, within a
column, a categorical's used categories in their declared order or the sorted values of any other
column. `HarmonyResult.batch_levels` lists that order, and `HarmonyResult.parameters["phiColumns"]`
names each level `"{column}_{level}"`.

The datastore checks `harmony_params` with
{py:func}`~cytearc.embeddings.harmony.validate_harmony_parameters` before it reads data, and a
pipeline checks `params["harmony"]["harmony_params"]` when it resolves its recipe, before it
creates a run record. Keys of the earlier Harmony implementation raise `ValueError`:

| Earlier key | Use instead | Message |
| --- | --- | --- |
| `nclust` | `n_clusters` | `Unsupported Harmony parameter 'nclust': it was renamed to 'n_clusters'` |
| `max_iter_kmeans` | `max_iter_cluster` | `Unsupported Harmony parameter 'max_iter_kmeans': it was renamed to 'max_iter_cluster'` |
| `lamb` | `ridge_penalty` for a fixed penalty, or nothing for the dynamic penalty | `Unsupported Harmony parameter 'lamb': it was renamed to 'ridge_penalty'; ridge_penalty is a fixed penalty, so leave it out to use Harmony2's dynamic penalty` |
| `cluster_fn` | nothing; clusters are seeded with k-means++ | `Unsupported Harmony parameter 'cluster_fn': Harmony2 seeds its clusters with k-means++ and has no initialisation hook` |

`max_iter_harmony=0`, which clustered without correcting, now raises `max_iter_harmony must be at
least 1`, and `ridge_penalty` must be positive where `lamb` accepted 0. Other unknown keys raise
`ValueError` listing the supported keys. Passing an earlier key to `fit_harmony` itself raises
Python's `TypeError` for an unexpected keyword argument.

A `batch_correction` artifact stores the corrected coordinates and the fitted state that
Symphony-style mapping references copy. The soft assignments are not stored.

| Array or attribute | Shape, dtype | Contents |
| --- | --- | --- |
| `data` | cells by dims, float32 | corrected coordinates |
| `cluster_mass` | clusters, float64 | total assignment of each cluster |
| `raw_centroids` | clusters by dims, float64 | assignment-weighted means of the uncorrected coordinates |
| `corrected_centroids` | clusters by dims, float64 | assignment-weighted means of the corrected coordinates |
| `centroids` | dims by clusters, float64 | unit-length centroids of the final clustering; a cluster whose cells cancel keeps a zero centroid |
| `sigma` | clusters, float64 | `sigma` of each cluster |
| `ridge_penalty` | clusters by batch levels, float64 | ridge penalties of the last correction |
| `active_levels` | clusters by batch levels, bool | levels that entered each cluster's last regression |
| attribute `batch_levels` | one list of strings per batch column | batch levels in design order |
| attribute `fit_parameters` | mapping | `HarmonyResult.parameters` |

`HarmonyResult.parameters` records the settings the fit used, with camelCase keys: `algorithm`
(`"harmony2"`), `nClusters`, `theta` per batch level after `tau`, `tau`, `sigma` per cluster,
`ridge` (`"dynamic"` or `"fixed"`) with `ridgeAlpha` or `ridgePenalty`, `pruneThreshold`,
`blockSize`, `maxIterCluster`, `maxIterHarmony`, `epsilonCluster`, `epsilonHarmony`,
`randomState`, `initialization` (`"kmeans++"`), and `phiColumns`. It also records how the fit ran:
`rounds`, `converged`, `objective`, the objective at the end of each round's clustering, and
`clusterPasses`, the clustering passes of each round.

Reuse requires every array in this table. A correction saved by the earlier implementation stored a
`ridge` array instead of `ridge_penalty` and `active_levels`, so it stays listable, loadable, and
traceable, but no request reuses it: the next request computes a new correction, and the ANN index,
neighbours, graphs, embeddings, clusters, and mapping references built on it are computed again.
Mapping references built from an earlier correction still load and map.

```{eval-rst}
.. autofunction:: cytearc.embeddings.fit_harmony
```

```{eval-rst}
.. autofunction:: cytearc.embeddings.harmony.validate_harmony_parameters
.. autofunction:: cytearc.embeddings.harmony.harmony_cluster_count
.. autofunction:: cytearc.embeddings.harmony.harmony_peak_bytes
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
