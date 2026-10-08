# Graph construction API reference

These `DataStore` methods build a neighbourhood graph as explicit, provenance-backed stages. Each
builder returns an {py:class}`~cytearc.ArtifactRef`; pass that exact ref to the next stage.

Prefer `ds.pipeline.run()` for the fixed rich RNA recipe. Use the individual methods for a partial
workflow, an alternative branch, or parameters intentionally kept out of the pipeline surface.
See [Building neighbourhood graphs step by step](../tutorials/graph-construction) for an executable walkthrough.

## Explicit chain

Capture the live Boolean cell column once at the workflow boundary. The returned selection is
immutable; later live changes do not alter it.

```python
cells = ds.snapshot_cell_selection("I")
features = ds.features.hvgs(cells, top_n=1000, show_plot=False)

normalized = ds.features.normalize(cells, features)
pca = ds.reduction.pca(normalized, dims=21)
ann = ds.graph.ann_index(pca)
neighbors = ds.graph.neighbors(ann, k=11)
graph = ds.graph.connectivity(neighbors)
initialization = ds.embeddings.initialization(pca)
```

The full stage order is:

1. {py:meth}`~cytearc.DataStore.snapshot_cell_selection` and an exact feature-selection ref
2. {py:meth}`~cytearc.datastore.namespaces.FeaturesAccessor.normalize`
3. {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.pca`, {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.lsi`, or
   {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.custom`, or none
4. Optional {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.harmony`
5. {py:meth}`~cytearc.datastore.namespaces.GraphAccessor.ann_index`
6. {py:meth}`~cytearc.datastore.namespaces.GraphAccessor.neighbors`
7. {py:meth}`~cytearc.datastore.namespaces.GraphAccessor.connectivity`

Without a reduction, `graph.ann_index(normalized)` builds the graph on the normalized artifact
itself: its float32 values of the selected features are the coordinates, one per feature, as in a
pipeline run with `pca_dims=0`. Every later stage, including the embedding initialization and
an SNN integration, accepts such a graph. Harmony corrects reduced coordinates only, and WNN
integration, mapping references, and doublet scoring need them too, so each rejects normalized
coordinates with an error that names a reduction such as `reduction.pca`. Because normalized values can
be coordinates, `features.normalize` writes them with a checked writer: a value that is NaN or
infinite once rounded to float32, such as one from a custom normalizer or the `log1p` of a
negative count, raises `NonFiniteArtifactError`, a `ValueError` that names `features.normalize`,
and no complete artifact is saved.

The hnswlib index holds every cell in memory, about `4 * dims + 8 * ann_m + 130` bytes per cell on
x86-64 Linux (`cytearc.neighbors.index.ann_index_peak_bytes`), where `dims` is the number of
coordinate dimensions: the PCA dimensions, or the selected features of normalized coordinates.
`graph.ann_index` holds it while it adds the coordinate blocks and while it saves it, and
`graph.neighbors` while it loads and queries it, beside the indices and distances of every cell's
neighbours, 8 bytes per neighbour. Each compares what it holds with the datastore's memory budget
before it creates or loads the index and raises `MemoryError` over it, naming the bytes, the cells,
the dimensions, and the limit. The index moves between the store and memory through a temporary
file of about `4 * dims + 8 * ann_m + 20` bytes per cell
(`cytearc.neighbors.index.ann_index_file_bytes`) in the system's temporary directory, which needs
that much free space.

{py:meth}`~cytearc.datastore.namespaces.EmbeddingsAccessor.initialization` depends on explicit reduction, Harmony,
imported, or normalized coordinates. It is needed for UMAP unless an initialization array is
supplied, but is not part of connectivity construction.

`graph.neighbors` reads the coordinate ref named by the ANN artifact. Its optional `coordinates=`
argument is only an equality check; it cannot redirect an index to different coordinates.

## Downstream consumers

UMAP, t-SNE, clustering, diffusion, trajectories, graph metrics, and graph-backed plots
require an exact graph or neighbour artifact. Analytical producers return artifacts and do not add
live metadata fields:

```python
umap = ds.embeddings.umap(
    graph,
    initialization,
)
leiden = ds.clusters.leiden(
    graph,
    resolution=0.5,
)
ds.plots.embedding(layout=umap, color_by=leiden)
```

The scientific cell selection comes from graph lineage. Load a payload from its exact ref, or pass
the ref to another consumer.

Graph consumers do not accept an independent feature selection. Named lineage edges identify the
normalized feature selections used to construct a native graph. Imported-coordinate graphs have
no such selection; integrated graphs preserve the ordered projections of their explicit sources.

## Methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   cytearc.DataStore.snapshot_cell_selection
   cytearc.datastore.namespaces.FeaturesAccessor.normalize
   cytearc.datastore.namespaces.ReductionAccessor.pca
   cytearc.datastore.namespaces.ReductionAccessor.lsi
   cytearc.datastore.namespaces.ReductionAccessor.custom
   cytearc.datastore.namespaces.ReductionAccessor.harmony
   cytearc.datastore.namespaces.GraphAccessor.ann_index
   cytearc.datastore.namespaces.GraphAccessor.neighbors
   cytearc.datastore.namespaces.GraphAccessor.connectivity
   cytearc.datastore.namespaces.EmbeddingsAccessor.initialization
   cytearc.datastore.namespaces.GraphAccessor.load
```

```{eval-rst}
.. automethod:: cytearc.DataStore.snapshot_cell_selection
.. automethod:: cytearc.datastore.namespaces.FeaturesAccessor.normalize
.. automethod:: cytearc.datastore.namespaces.ReductionAccessor.pca
.. automethod:: cytearc.datastore.namespaces.ReductionAccessor.lsi
.. automethod:: cytearc.datastore.namespaces.ReductionAccessor.custom
.. automethod:: cytearc.datastore.namespaces.ReductionAccessor.harmony
.. automethod:: cytearc.datastore.namespaces.GraphAccessor.ann_index
.. automethod:: cytearc.datastore.namespaces.GraphAccessor.neighbors
.. automethod:: cytearc.datastore.namespaces.GraphAccessor.connectivity
.. automethod:: cytearc.datastore.namespaces.EmbeddingsAccessor.initialization
.. automethod:: cytearc.datastore.namespaces.GraphAccessor.load
```
