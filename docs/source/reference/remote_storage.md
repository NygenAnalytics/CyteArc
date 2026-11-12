---
description: Connect CyteArc to object storage, set the time limits of Hugging Face stores, mount shared count matrices, and configure local scratch space for remote analysis.
---

(remote_storage)=

# Remote storage configuration

For a working public example, start with the [remote dataset mounting tutorial](../tutorials/remote_stores.ipynb). This reference covers your own object storage, the time limits of Hugging Face stores, what a mount preserves, and scratch space for repeated reads. The examples require CyteArc's `extra` dependencies.

## Connect your own bucket

These templates are not executed. Replace the example locations and storage options with those for your own bucket, including its endpoint and authentication settings.

A mounted datastore separates its count source from its writable analysis target. The source can be a local path or an object-store URI. The target stores copied metadata plus new artifacts, while count blocks continue to resolve from the source. Mounting does not by itself mean that either location is remote.

`mount_datastore` copies cell and feature metadata into the target and validates the source's primary `counts` matrix. RNA sources must also contain a matching `countsT` copy on Zarr v3: `counts` stores cells by genes, while `countsT` stores genes by cells for efficient gene-wise reads. Mounting uses both copies from the source without transposing or copying them into the target. Non-RNA assays do not require `countsT`. New metadata and analysis artifacts are written only to the target.

### Mount an object-store count source

A mounted analysis can keep shared counts at an object-store URI while writing metadata and artifacts to a local target. Counts remain in the source:

```python
# Open count stores and run CyteArc analyses.
import cytearc

# Mount the count source into a separate writable target.
mounted = cytearc.mount_datastore(
    's3://shared-bucket/atlas.zarr',
    at='my-analysis.zarr',
    storage_options={'skip_signature': True},
    zarrProfile='fast_local',
)
```

The source must remain available at the recorded URI whenever the target is opened. Use a new or empty target outside the source and any other store.

### Give the source its own options

`storage_options` opens the target, and the source too unless the source has options of its own. When the two need different options, such as a target in your own bucket over a public Hugging Face source, pass the source's options as `source_storage_options`, and pass both again whenever the target is reopened. Neither is saved in the target.

```python
# Read the bucket credentials from the environment.
import os

# Open count stores and run CyteArc analyses.
import cytearc

# Mount a public Hugging Face source into a target in your own bucket.
bucket_options = {
    "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
}
mounted = cytearc.mount_datastore(
    "hf://buckets/org/name/atlas.zarr",
    at="s3://my-bucket/project/analysis.zarr",
    storage_options=bucket_options,
    source_storage_options={"token": False},
)

# Reopen the target later with the options of both stores.
analysis = cytearc.DataStore(
    "s3://my-bucket/project/analysis.zarr",
    storage_options=bucket_options,
    source_storage_options={"token": False},
)
```

Without `source_storage_options`, the source opens with the target's options, which a store on another backend refuses. `python -m cytearc.tools.repack_zarr` takes the source's options as `--source-storage-options`.

### Write a composite to your bucket

A composite combines selected cells of several sources into one store and reads their counts from the sources, without copying them. `create_composite` writes it to a local path, a URL with the `storage_options` that open it, or a Zarr store, and writes its metadata with the storage profile of that destination. Each reopen needs the options of the composite's own store and of every source, by source name:

```python
# Create a composite of two public sources in your own bucket.
from cytearc.composite import create_composite

public = {"token": False}
sources = {
    "study_a": cytearc.DataStore(
        "hf://buckets/org/name/a.zarr", zarr_mode="r", storage_options=public
    ),
    "study_b": cytearc.DataStore(
        "hf://buckets/org/name/b.zarr", zarr_mode="r", storage_options=public
    ),
}
joint = create_composite(
    sources,
    at="s3://my-bucket/project/joint.zarr",
    storage_options=bucket_options,
    features="intersection",
    source_column="study",
)

# Reopen the composite with the options of its store and of each source.
joint = cytearc.DataStore(
    "s3://my-bucket/project/joint.zarr",
    storage_options=bucket_options,
    source_storage_options={name: public for name in sources},
)
```

`source_column="study"` names the cell column that holds each cell's source name, `dataset` by default.

### Open a datastore directly

Pass an object-store URI as `zarr_loc` and provider options as `storage_options`. This anonymous S3 shape is a template, not a tested public dataset:

```python
# Open the count store for this analysis.
ds = cytearc.DataStore(
    "s3://bucket/path/to/data.zarr",
    zarr_mode="r",
    storage_options={"skip_signature": True},
)
```

For a writable remote store, use the `cloud` profile for newly written arrays. Existing arrays retain the layout chosen when they were created. Read credentials from the environment rather than embedding secrets in notebooks:

```python
# Read object-store credentials from the environment.
import os

# Open count stores and run CyteArc analyses.
import cytearc

# Open a writable remote store with credentials from the environment.
remote_writable = cytearc.DataStore(
    "s3://my-bucket/project/data.zarr",
    zarr_mode="r+",
    zarrProfile="cloud",
    storage_options={
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],

        # "endpoint": "https://...",  # S3-compatible endpoints
    },
)
```

Google Cloud Storage uses a `gs://` URI. Pass the provider options your environment already uses for obstore or fsspec, such as application-default credentials on the VM or an explicit token in `storage_options`.

After opening a writable store, use the same analysis calls as for local data.

## Time limits of Hugging Face stores

CyteArc reads an `hf://` URI, such as a [CyteBase dataset](../tutorials/cytebase_datasets.md), from Hugging Face. `storage_options` takes the options of Hugging Face's file system, such as `"token": False` for anonymous reads, and two time limits in seconds that bound each request of the store, whether it reads a key, looks a key up or lists keys:

| Option | Default | Limit |
|---|---|---|
| `request_timeout` | `60` | Each attempt at a request |
| `retry_timeout` | `180` | The request with all of its attempts |

An attempt that runs out of time, loses its connection, or receives a 408 or 5xx response is retried after 2, 4, 8 and more seconds, or after the delay the server asks for if that is longer. When the next attempt could not start within `retry_timeout`, the request raises `TimeoutError`, which names it, the store and the time limit. A 429 response, which asks the reader to slow down, is retried at most three times, after the delay it asks for or after 2, 4 and 8 seconds if that is longer, and only for a delay of up to 300 seconds; that wait does not count toward `retry_timeout`. Other failures, such as a 401 or 403 response, raise at once.

A store opened read-only lists its objects once, the first time it needs them, with one request to Hugging Face's API for each page of the listing; a CyteBase dataset fits on one page. Each read is then one request to Hugging Face's file resolver for exactly the bytes it needs, and a key that the listing does not hold reads as missing without a request. The listing is a snapshot: the store does not see objects written after it listed them, and reading a listed object that has since been deleted raises `OSError`. Reopen the store to read what it holds now. A writable store reads and looks up keys through Hugging Face's file system, and reads a key as missing only when Hugging Face reports it missing. When Zarr creates an array or a group, a failed lookup of a parent group raises, instead of letting Zarr replace the parent's metadata and attributes with an empty group.

Hugging Face limits the API and resolver requests of each account, and of each network address for anonymous requests, over 5-minute windows ([Hub rate limits](https://huggingface.co/docs/hub/rate-limits)). Anonymous readers on one address share its limits, so pass a token in `storage_options`, or log in to Hugging Face, to read under your own account's limits.

On a slow connection, give the reads more time:

```python
# Open a CyteBase dataset with longer time limits.
ds = cytearc.DataStore(
    "hf://buckets/Nygen/cytebase-2610/datasets/"
    "mcglade_2021_scrna_data_human_fallopian_tube_30f5e171/data.zarr",
    zarr_mode="r",
    storage_options={"token": False, "request_timeout": 120, "retry_timeout": 600},
)
```

## What a mount preserves

The mount records matrix shape, dtype, and source identity. Reopening fails if the source no longer matches that identity. Metadata is copied at mount time, so later source metadata changes are not synchronized into the target.

The mount records its source's absolute path or URI, so the target alone can move. With `relative_sources=True`, a source in the directory that holds the target, or below it, is recorded by its path relative to that directory instead, so the target and its source can be moved, copied, or uploaded to a bucket together and still open. Releases before this option cannot open a target that records a relative path, except from the directory that holds it.

The target also resolves the source's artifacts read only, after its own. Results already saved in the source, including labels and embeddings imported with it, can be listed, loaded, traced, and used as inputs on the mount, and a step whose provenance matches a saved result reuses it instead of writing a copy. New artifacts are still written only to the target, and pipeline runs and their labels stay with the store that holds them. Results that reuse source artifacts need the source, as counts do; `python -m cytearc.tools.repack_zarr` copies a mount into a self-contained store.

## Local scratch for reductions

PCA fitting and score projection make multiple passes over normalized expression. `local_cache` can copy that normalized artifact to local scratch so repeated passes do not fetch it remotely. The decision depends on the store holding the normalized artifact, not on where the original counts live.

For a mounted analysis, this gives two useful cases. A normalized result newly written to a local target needs no staging, even when its counts came from a remote source. A normalized result reused from the remote source does need staging under the default policy. Harmony, ANN, and neighbor queries read saved reduced coordinates and do not use this normalized-expression scratch.

| Value | Behavior |
|---|---|
| `"auto"` (default) | Stage for remote stores; skip for local stores |
| `True` | Stage a remote artifact in temporary scratch, deleted when the stage ends |
| `False` | No staging; every pass reads the store URI |
| `"/path/to/scratch"` | Stage a remote artifact in persistent scratch keyed by artifact ID |

A repeated PCA call can reuse an existing result before reading normalized blocks at all.

For a writable remote store opened from the non-executed template above, the same path-string policy stages normalized blocks and keeps the cache for inspection or reuse. The following is also a non-executed template:

```python
# Freeze the current active cells before selecting genes.
cell_selection = remote_writable.snapshot_cell_selection(cell_key="I")

# Select variable genes for the remote reduction.
features = remote_writable.features.hvgs(
    cell_selection,
    min_cells=20,
    top_n=2000,
    show_plot=False,
)

# Normalize counts over the selected features.
normalized = remote_writable.features.normalize(cell_selection, features)

# Fit PCA with persistent local scratch for remote normalized data.
reduction = remote_writable.reduction.pca(
    normalized,
    dims=15,
    local_cache="/tmp/cytearc_pca_scratch",
)
```

`local_cache` controls where a stage reads its working data. It does not change what the stage computes or the result's identity, so a completed result can be reused under a different scratch policy later. Temporary scratch (`True` or `"auto"` on a remote store) is deleted when the stage ends, on both success and failure. A path-string cache is kept for reuse or inspection.

Plan local disk for float32 dense blocks roughly as `n_cells × n_features × 4` bytes (about 8 GiB for 1M cells × 2000 HVGs).

## Choosing remote or local storage

Object-store latency, request costs, credentials, and provider behavior depend on the service. Repeated passes over counts may be faster with a local copy. The prepared example datasets can be downloaded as local stores, as shown in the [](../quickstart.ipynb).

For timing and memory measurements, see [Measuring resource use](../concepts/memory_and_execution.md#measuring-resource-use). For custom statistics over mounted graphs or count blocks and selective export, see [](../tutorials/custom_analyses.ipynb).
