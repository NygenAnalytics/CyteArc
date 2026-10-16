---
myst:
  html_meta:
    description: Download CyteArc's public example datasets with cytearc.datasets, and find the CyteBase catalog in the separate cytebase package.
---

# Example datasets API reference

`cytearc.datasets` lists and downloads the public example datasets that the CyteArc tutorials
use. They live in top-level repositories of the public `Nygen/cytebase` Hugging Face bucket, such
as `cytearc_docs` for the documentation datasets. Downloads need network access but no
credentials. A repository names each dataset by its directory.

```python
import cytearc

repository = cytearc.datasets.connect("cytearc_docs")
repository.list_datasets()

# Download and extract a prepared CyteArc store, then open it read-only.
path = repository.download_dataset(
    "tenx_5K_pbmc_rnaseq", destination="cytearc_datasets", zarr=True
)
ds = cytearc.DataStore(f"{path}/data.zarr", zarr_mode="r")
```

`download_dataset(name, zarr=True)` downloads the dataset's `data.zarr.tar.gz` archive and
extracts it to `data.zarr` in the returned directory, keeping the archive beside it. Without
`zarr=True`, it downloads the dataset's other files, such as its source count files.
`download(path)` downloads one file, or every file under a directory, and keeps its path within
the repository.

```{eval-rst}
.. autofunction:: cytearc.datasets.list_repositories

.. autofunction:: cytearc.datasets.connect

.. autoclass:: cytearc.datasets.Repository
   :members:
```

## CyteBase catalog

The CyteBase catalog of public datasets stored as cloud-hosted CyteArc DataStores is the
separate `cytebase` package, which CyteArc does not install. CyteArc opens CyteBase datasets by
their `hf://` URIs, as <https://docs.nygen.io/CyteArc/tutorials/cytebase-datasets> shows. To
search the catalog, install and import the package:

```python
# Install the catalog client first: pip install cytebase
import cytebase

catalog = cytebase.Catalog()
```

See the CyteBase documentation at <https://docs.nygen.io/CyteBase/> for its API and tutorials.

`cytearc.cytebase` keeps the earlier names of the example downloads through CyteArc 1.x:
`Repository`, `connect`, and `list_repositories` are the `cytearc.datasets` objects. The catalog
names it once exported, such as `Catalog`, `releases`, and `embedding`, moved to the `cytebase`
package, and reading one from `cytearc.cytebase` raises an `AttributeError` that says so. The
catalog build pipeline moved to the CyteBase repository at
<https://github.com/NygenAnalytics/CyteBase>.
