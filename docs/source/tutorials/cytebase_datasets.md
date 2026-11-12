---
description: Find a public CyteBase dataset, build its Hugging Face URI, and open it in CyteArc without downloading it.
---
(cytebase_datasets)=

# CyteBase datasets

Find a public CyteBase dataset and open it in CyteArc by its URI, without downloading it. [CyteBase](https://docs.nygen.io/CyteBase/) converts single-cell datasets from CZ CELLxGENE Discover into CyteArc stores and publishes them in public Hugging Face buckets, and CyteArc reads a dataset's metadata and count blocks over the network as each step needs them.

## Build a dataset URI

Each CyteBase release is a read-only bucket named by its year and month, such as `Nygen/cytebase-2610` for the October 2026 release; a later release in the same month adds a number, such as `Nygen/cytebase-2610.1`. A dataset's store is at:

```text
hf://buckets/Nygen/cytebase-<release>/datasets/<cytebase_id>/data.zarr
```

`<cytebase_id>` is the dataset's CyteBase ID, such as `mcglade_2021_scrna_data_human_fallopian_tube_30f5e171`. Keep the release in every URI you record, so that your analysis names the release it read.

## Search the catalog

The separate `cytebase` package searches the catalog, and each row of its results holds the dataset's URI in the `zarr_uri` column:

```python
# Install the catalog client first: uv pip install cytebase
import cytebase

# Search the October 2026 release anonymously and take the first match's URI.
catalog = cytebase.Catalog(version="2610", token=False)
matches = catalog.search("fallopian tube", limit=5)
uri = matches[0]["zarr_uri"]
```

`search` keeps the datasets whose ID, title, citation, first author or labels match every word of the query. The [CyteBase documentation](https://docs.nygen.io/CyteBase/) covers facet filters, SQL queries and releases.

## Open a dataset

Open the store read-only to explore it:

```python
import cytearc

# Open the published store read-only and anonymously; its counts stay in the bucket.
ds = cytearc.DataStore(uri, zarr_mode="r", storage_options={"token": False})
```

The cell table holds the published annotations, such as `cell_type`, and when the source dataset has a UMAP, `ds.artifacts.find("embedding", imported="X_umap", from_assay="RNA")` returns it. `storage_options={"token": False}` reads anonymously; leave it out to use your Hugging Face login. A read that the bucket leaves unanswered is retried and then raises `TimeoutError`; on a slow connection, raise the [time limits](../reference/remote_storage.md#time-limits-of-hugging-face-stores) in `storage_options`.

A read-only store refuses any step that would save a result, with `PermissionError`. To save your own columns and results, mount the store with `cytearc.mount_datastore`: the mount is a local directory that holds a copy of the cell and feature metadata and everything you write, while the counts and published results stay in the bucket, so the mount needs network access in every session. The [remote dataset mounting tutorial](remote_stores.ipynb) mounts the 1.1 million cells of the Tabula Sapiens atlas this way, saves the author-annotated B cells as a column of the mount and reopens it, and the [CyteBase composite datasets tutorial](composite_datasets.ipynb) analyzes several CyteBase datasets together without copying their counts.

## Open with catalog checks

The `cytebase` package opens the same stores with `Catalog.open_datastore` and `Catalog.mount_datastore`, which also check the opened arrays against the catalog's record of the dataset and of the build that produced it. A catalog mount writes a receipt named `<mount>.cytebase.json` beside the mount that pins the build it read, and refuses to reopen once that build is replaced.
