---
description: Open public CyteBase datasets in CyteArc by their Hugging Face URIs, read-only for exploration or as a writable mount for your own analysis.
---
(cytebase_datasets)=

# CyteBase datasets

[CyteBase](https://docs.nygen.io/CyteBase/) publishes public single-cell datasets, converted
from CZ CELLxGENE Discover, as CyteArc DataStores in public Hugging Face buckets. CyteArc opens
them like any other remote store: by URI, reading metadata and count blocks over the network as
an operation needs them, without downloading the dataset.

## Dataset URIs

Each public CyteBase release is a read-only bucket named by year and month, such as
`Nygen/cytebase-2610` for the October 2026 release. A dataset's store lives at:

```text
hf://buckets/Nygen/cytebase-<release>/datasets/<cytebase_id>/data.zarr
```

`cytebase_id` is the dataset's CyteBase ID, for example
`mcglade_2021_scrna_data_human_fallopian_tube_30f5e171`. Keep the release in the URIs you
record, so that your analysis names the exact build it used.

To find datasets, search the catalog with the separate `cytebase` package. Its results include
each dataset's URI in the `zarr_uri` column:

```python
# Install the catalog client first: pip install cytebase
import cytebase

# Search the October 2026 release anonymously and take the first match's URI.
catalog = cytebase.Catalog(version="2610", token=False)
matches = catalog.search("fallopian tube", limit=5)
uri = matches[0]["zarr_uri"]
```

The [CyteBase documentation](https://docs.nygen.io/CyteBase/) covers catalog search, facet
filters, SQL queries, and releases.

## Explore a dataset read-only

```python
import cytearc

# Choose a dataset in the October 2026 release.
uri = (
    "hf://buckets/Nygen/cytebase-2610/datasets/"
    "mcglade_2021_scrna_data_human_fallopian_tube_30f5e171/data.zarr"
)

# Open the published store read-only; its counts stay in the bucket.
ds = cytearc.DataStore(
    uri, zarr_mode="r", storage_options={"token": False}, min_features_per_cell=-1
)

# Read the published cell annotations for the active cells.
meta = ds.cells.to_pandas_dataframe(["ids", "cell_type"], key="I")

# Plot the authors' UMAP, which the build imports when the source provides one.
umap = ds.artifacts.find("embedding", imported="X_umap", from_assay="RNA")
ds.plots.embedding(layout=umap, color_by="cell_type")
```

A read-only open writes nothing: an operation that would compute a new result raises
`PermissionError`. `min_features_per_cell=-1` keeps the published cell selection, and
`storage_options={"token": False}` reads anonymously; omit it to use your Hugging Face login.
Plotting needs CyteArc's plotting dependencies, `cytearc[extra]`.

## Analyze a dataset

Mount the dataset to save your own results. A mount is a local, writable store whose counts
stay in the bucket:

```python
# Create a local analysis store backed by the remote counts.
analysis = cytearc.mount_datastore(
    uri,
    at="fallopian_tube.zarr",
    storage_options={"token": False},
    min_features_per_cell=-1,
)

# Run the RNA pipeline; its results are saved in fallopian_tube.zarr.
run = analysis.pipeline.run(label="baseline")

# Color the computed UMAP by the selected clustering.
analysis.plots.embedding(run=run, color_by="clusters")
```

Reopen the mount later with `cytearc.DataStore("fallopian_tube.zarr", min_features_per_cell=-1)`.
Its counts still resolve from the bucket, so later reads need network access.
[](remote_stores.ipynb) explains mounted analysis in detail, and [](composite_datasets.ipynb)
analyzes several CyteBase datasets together without copying their counts.

## Verified opens with the cytebase package

The `cytebase` package opens the same stores with `Catalog.open_datastore` and
`Catalog.mount_datastore`. These also check the published dataset record and build receipt
against the opened arrays, and a catalog mount writes a `.cytebase.json` receipt that pins the
build it used. See the [CyteBase documentation](https://docs.nygen.io/CyteBase/) when you need
those checks.
