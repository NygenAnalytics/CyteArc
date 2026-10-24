---
myst:
  html_meta:
    description: Read, write, merge, and subset CyteArc datasets using Cell Ranger, H5AD, Matrix Market, Seurat, and Zarr formats.
---

# Import and export API reference

## Readers

```{eval-rst}
.. autoclass:: cytearc.readers.CrReader
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.readers.CrH5Reader
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.readers.CrDirReader
    :members:
    :inherited-members:
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: cytearc.readers.MtxReader
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.readers.H5adReader
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.readers.CSVReader
    :members:
```

## H5AD inspection

Use this before `H5adReader` when you do not know which matrix and metadata keys an H5AD file uses.
Review the result before importing: `matrixKey` names the matrix that will become the counts, and
`matrixCandidates` lists every candidate in rank order.

`inspect_h5ad` ranks the candidate matrices so that counts come first. A matrix holds counts when a
sample of its stored values, up to 32 windows of 1,024 values spread over the whole matrix, is
finite, non-negative, and whole. Among counts, a conventional name wins: `raw/X`, `layers/counts`,
and `layers/raw_counts`, then `X`, then `layers/spliced`, then other matrices, and the `unspliced`
and `ambiguous` layers of velocity data last. Matrices of one tier rank by their total, estimated
from the same sample, so the one that holds the most reads wins. When other matrices of the same
cells also hold counts, a warning lists them with their estimated totals; pass `matrix_key` to
choose another. A velocity file whose `X`, `spliced`, and `unspliced` are normalized imports its
`matrix` layer, the total counts, rather than its `ambiguous` reads.

The cell IDs are the `obs` index when it is unique, else a preferred ID column such as `cell_id` or
`barcode` that holds unique text or integers, else the first unique text column. A column of
distinct floating-point values, such as a score, is never chosen. When no column holds unique IDs,
`cellIdsKey` names the index anyway and a warning says that its IDs repeat: such a file, as
`anndata.concat` writes without `index_unique`, imports with
`H5adReader.from_inspect(inspection, duplicate_cell_ids="make_unique")`, which renames each later
occurrence of a repeated ID by appending `-1`, `-2`, and so on, as AnnData's
`obs_names_make_unique` does. Only a file without an index gets generated IDs.

`H5adReader` checks the matrix shape against the lengths of `obs` and `var` when it is
constructed, and a mismatch raises `ValueError`. Overriding `matrix_key`, for example with
`raw/X`, in the constructor or through `H5adReader.from_inspect`, therefore also needs the
matching `feature_attrs_key`, here `raw/var`. `inspect_h5ad(path, matrix_key="raw/X")` returns
reader arguments that pair them.

`H5adReader` decodes AnnData categorical columns and pandas nullable columns, including the
nullable string arrays that AnnData writes for dataframe indexes. By default it reads the index
that the dataframe's `_index` attribute names.
Missing categorical or object values become `None`; missing numeric values become `NaN`.
Unsupported group encodings and multi-dimensional columns are skipped with a warning.
`H5adToZarr` keeps missing rows under a linked missing mask and stores nullable booleans as
booleans.

Categorical columns keep their categories. `H5adToZarr` stores, with each imported categorical
column of `obs` or `var` that has at most 10,000 categories, a `levels` attribute that lists every
category in its order, unused ones included, and an `ordered` attribute, as `SeuratToZarr` stores
factor levels. A cell column also gets a categorical `display` attribute in that order, with the
colors of `uns["<column>_colors"]` when the file holds a color for each category, or else
CyteArc's palette; plots of the column show its categories in that order and colors. Exports write
such a column as the same categorical. Imported cluster labels keep the same attributes on their
`values` array.

A dataframe index that is neither the ID nor the name column imports as metadata only when it holds
other values: an index that repeats the IDs or the names is left out, and AnnData's unnamed index,
which no AnnData column may be named, imports as `source_index`, or `source_index_2` and so on when
that name is taken. A missing or blank feature name, such as a missing category, takes the
feature's ID as the file holds it, with a warning.

Pass `embedding_roles` and `cluster_keys` to select analytical H5AD values for artifact import.
Selected `obsm` arrays and cluster labels are excluded from live metadata and returned as exact
refs by `H5adToZarr.dump()`. Other supported `obs` columns remain literal metadata. The role
`"umap"` or `"tsne"` makes an `embedding` artifact, and `"coordinates"` an `imported_coordinates`
artifact, such as of `X_pca`, `X_scVI`, or `X_pca_harmony`, which `graph.ann_index`, embedding
initialization, and mapping accept as they accept a PCA. Coordinates computed after an import enter
a store through {py:meth}`~cytearc.datastore.namespaces.ReductionAccessor.import_coordinates`
({doc}`graph_construction`).

CSC input is converted to temporary row storage on local disk. Pass `temp_dir` to
`H5adReader` or `H5adReader.from_inspect` to choose its parent directory; the default uses
the system temporary directory, including `TMPDIR`. The temporary files are removed when the
reader closes.

```{eval-rst}
.. autofunction:: cytearc.inspect_h5ad
```

```{eval-rst}
.. autoclass:: cytearc.H5adInspectResult
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.H5adImportResult
    :members:
```

## Matrix Market inspection

`inspect_mtx` reports every complete matrix, feature, and cell triplet in a supported source.
Pass one returned candidate to `MtxReader`.

Matrix Market coordinates may come in any order, and duplicate coordinates are summed wherever
they appear. `MtxReader` and `CrDirReader` stream a matrix sorted by cell, as Cell Ranger 3 and
later write it, straight from the file, and sort a matrix sorted by feature into temporary rows
on local disk when an import starts. Any other order, such as the descending features within each
barcode that Cell Ranger 1 and 2 write, or shuffled entries, is read once more at construction to
sort the entries of each kept cell into the same temporary rows. The totals, the filtering
cutoff, and the count dtype follow the summed counts, so every order imports the counts and
identity of the same matrix in sorted order. `coordinateOrder` reports `cellMajor`,
`featureMajor`, or `unordered`, and `temporaryDiskBytes` the size of the rows, a count and a
feature index for each entry. Pass `temp_dir` to choose their parent directory; the default uses
the system temporary directory, including `TMPDIR`. The rows are removed at the end of an import
and when the reader closes, and a later import sorts the file again.

```{eval-rst}
.. autofunction:: cytearc.inspect_mtx
```

```{eval-rst}
.. autoclass:: cytearc.readers.MtxCandidate
```

## Seurat import

Import a serialized Seurat object from an `.rds` file.
Inspect with `inspect_seurat`, open `SeuratReader`, then write with `SeuratToZarr`.
This path does not attach to a live R session and does not read or write `.h5seurat`.
Sidecar matrices such as BPCells directories and HDF5 files resolve inside `sidecar_root`, which
defaults to the directory of the `.rds` file. A reader opened from a stream needs an explicit
`sidecar_root` before it reads sidecar-backed layers.
Without `assays` or `reductions`, `SeuratToZarr` imports every assay and reduction that can be
imported and reports the others as `skipped_assay` and `skipped_reduction` notices; named ones
must import. `cell_metadata` includes or excludes cell metadata columns, and a column that cannot
be decoded is otherwise left out with a `skipped_metadata_column` notice. `default_assay` chooses
the store's default assay, which an active assay that is not imported, such as an `integrated`
assay without raw counts (`counts_layer_missing`), requires. `reduction_roles` sets whether a
reduction imports as a UMAP or t-SNE embedding or as graph coordinates.
See [Import and export](../tutorials/import-and-export) for the worked contract and [CyteArc for Seurat users](../seurat) for
workflow mapping.

```{eval-rst}
.. autoclass:: cytearc.SeuratReader
    :members:

.. autofunction:: cytearc.inspect_seurat

.. autoclass:: cytearc.SeuratInspectResult
    :members:

.. autoclass:: cytearc.SeuratImportResult
    :members:

.. autoclass:: cytearc.SeuratToZarr
    :members:
```

## Storage controls

Writers, merge, subset, and ``DataStore`` accept the same optional storage
controls. ``profile`` chooses the physical encoding. ``policy`` chooses paired
count-matrix geometry, and a writer uses it exactly. Without it, the imports
write the default geometry, and an import whose counts write or ``countsT``
transpose does not fit ``mem_budget`` raises ``MemoryError`` naming the largest
policy, with the default ``unitBytes`` and ``chunkBytes`` halved together, that
fits. A smaller geometry makes every later ``countsT`` read slower. Subset,
``repack_zarr``, merge, and ``add_grouped_assay`` instead fit the geometry to
``mem_budget``: they halve the default ``unitBytes`` and ``chunkBytes`` together
until their writes fit. Writers that choose their source batches (the sparse
and Seurat imports, and merge) admit batches of one destination row band, so a
geometry never leaves the write narrower batches than its bands. A write that
does not fit, with one-row count shards or with its explicit ``policy``, fails
before it creates the destination: writers raise when they are constructed,
and merge when it plans. ``MtxToZarr`` and ``CrToZarr`` take ``lines_in_mem`` in the
constructor, so the fit reserves the Matrix Market parse buffer that the write
uses. A resumed merge keeps the geometry of its completed counts. The geometry
never changes the store's identity. ``io`` overrides automatic read, compute,
and write widths. Unset values stay under automatic planning from
``mem_budget`` and ``nthreads``.

```{eval-rst}
.. autoclass:: cytearc.storage.io_policy.StorageIoPolicy
    :members:

.. autoclass:: cytearc.storage.count_matrix.CountMatrixPolicy
    :members:
```

### Count storage dtype

Every import stores each assay's counts in a dtype resolved from the values of
that assay, whatever the reader and source encoding, and no import takes a count
dtype argument. When every canonical (duplicate-summed) value is a non-negative
integer, the counts are stored in the narrowest of ``uint8``, ``uint16``,
``uint32``, and ``uint64`` that holds them. Other counts keep their source
dtype in native byte order, with ``float16`` read as ``float32``. Assays split from one source
matrix (10x HDF5 and Matrix Market feature types, H5AD ``assay_split_key``)
each resolve their own dtype. Each import reads every count once before it
creates the destination: ``CrReader`` subclasses and ``H5adReader`` report the
range of each group of features over the selected cells through
``count_value_ranges``, ``CSVReader`` records it as ``countRange`` in its first
pass, and ``SparseToZarr`` and ``SeuratToZarr`` scan their sources. Count
matrices hold finite values, so imports reject NaN and infinity before they
create the destination. Writers cast through a checked cast, so a count that
the stored dtype cannot hold raises instead of wrapping.

Subset and repack keep the source dtype. Merge stores the common type of the
source count dtypes, widened so that features summed by name cannot overflow
it, and rejects integer sources without a common integer dtype. Grouped and
melded assays keep their ``float64`` values.

```{eval-rst}
.. autofunction:: cytearc.storage.count_dtype.count_storage_dtype

.. autoclass:: cytearc.utils.count_values.CountValueRange
    :members:
```

## Writers

Every writer owns the reserved metadata columns `ids`, `names` and `I` and the
`__cytearc_missing__` prefix of missing-value masks. Source cell or feature metadata columns
with these names are skipped with a warning, so imported identifiers are never replaced.
Zarr reads `/` and `\` in a column name as path separators, so writers store a source column
whose name contains either one under the name with `_` in their place and log the rename. A
source name that is already valid keeps its name; a renamed column whose name is taken gets
the first free `_2`, `_3`, and so on. `SeuratToZarr` raises for reserved names instead of
skipping them.
The H5AD, CSV, Matrix Market, and Cell Ranger readers reject missing or repeated cell IDs, and so
does `SparseToZarr`, which also rejects missing or repeated feature IDs and converts a COO, CSC, or
dense matrix to CSR before it scans it. `CSVReader` reads its ID column as text, as the file holds
it, so `001` stays `"001"`. `H5adReader` renames repeated cell IDs on request
(`duplicate_cell_ids="make_unique"`). All
but the CSV reader also reject missing or repeated feature IDs: `CSVReader` takes its feature IDs
from the header as pandas reads it, which names a column with a blank header
`Unnamed: <position>` and renames each later occurrence of a repeated name by appending `.1`,
`.2`, and so on, without a warning. Some exporters use gene symbols as feature IDs, and these
repeat. Pass
`duplicate_feature_ids="make_unique"` to `MtxReader`, `CrDirReader`, `CrH5Reader`, or
`H5adReader`, also through `H5adReader.from_inspect`, to keep the first occurrence of each feature
ID and rename each later one, in file order, by appending `-1`, `-2`, and so on, as AnnData's
`var_names_make_unique` does. A suffixed name that is already an ID is
skipped, so the result is unique, and the reader logs how many IDs it renamed. IDs are made
unique across all features, whatever their assay, so an antibody that repeats the ID of an
earlier gene is renamed in its own assay, for example to `CD4-1`. Feature names are
not changed: an H5AD file without a feature name column takes the names from the IDs it holds,
and a Matrix Market feature reference matches the IDs of the feature sidecar. Cell IDs must be
unique in either mode.

The cell column `<assay>_I` of every assay that an import writes is reserved for the membership
of that assay ([Assay membership](../developers/zarr-internals#assay-membership)), because a merge reads a column of that name as which
cells the assay measured. `H5adToZarr` restores such a column as the membership of the assay
only when the file declares it in `uns["cytearc"]["assayMembership"]`, which maps each assay to
its column, as CyteArc's exports write it; the declared column must be boolean without missing
values, and a declaration that names another column or a column that `obs` lacks raises
`ValueError` before the destination is created. Every other source column named for an
imported assay, in an H5AD file without the declaration or from `CSVtoZarr`, `CrToZarr`, or
`MtxToZarr`, is skipped with a warning, as `I` is. `SeuratToZarr` writes the membership itself
and raises for a metadata column named for any imported assay. A column named for an assay that
the import does not write, such as `ADT_I` in an RNA-only import, stays an ordinary column.
`SparseToZarr` imports no cell metadata.

Import writers take their assay type from the assay name. Pass `assay_type` (or
`assay_types` on `CrToZarr`) to declare a custom-named assay as a preset such as `RNA`. A value
that is not a preset raises `ValueError`; use `Assay` for a generic assay. `DataStore(assay_types=...)`
and `add_melded_assay(assay_type=...)` apply the same rule. The presets and the persisted
`assayTypes` attribute are described in [Assay types](../developers/zarr-internals#assay-types-attribute).

Writers create a new store and replace only a store that no `DataStore` has opened. `CrToZarr`,
`MtxToZarr`, `H5adToZarr`, `SparseToZarr`, `CSVtoZarr`, `SeuratToZarr`, and `SubsetZarr` accept an
absent path, an empty directory, or an empty store, and raise `FileExistsError` for a destination
that holds any key, such as an earlier store or a `.DS_Store` file. With `overwrite=True`
(`overwrite_existing_file=True` for `SubsetZarr`) they replace a store whose root group holds no
prepared assay, no `matrixSource`, and only members that CyteArc writes (`cellData`, `matrices`,
`artifacts`, `pipeline`, assays, and workspaces of them).
A store that a `DataStore` has opened and content that CyteArc did not write, also inside its groups
and arrays, are never replaced, and `SubsetZarr` refuses a destination that overlaps a source. A
local destination inside a Zarr store, such as `s.zarr/RNA` or `s.zarr/new.zarr`, given as a path
or a local store, and a destination whose root group is an assay group raise `ValueError`.
`CrToZarr`, `MtxToZarr`, `H5adToZarr`, and `SeuratToZarr` check their destination before they
read the counts, and `SubsetZarr` when it is constructed. Each checks again as it creates the store
with Zarr mode `"w-"`, which refuses keys that appeared since the check, or with mode `"w"` for the
store that `overwrite=True` replaces. A local path is opened as it is written, also when it holds
`#`, `?`, or `;`. `subset_assay_zarr` adds one count array to an existing store and replaces
nothing: `out_grp` must not exist and must not be, contain, or lie inside `in_grp`, and its parent
group must not already record the layout of another count matrix.

`SubsetZarr`, `DataStoreMerge`, `mount_datastore`, and `repack_store` refuse a source store that
holds a pending derived assay, one that `add_grouped_assay` or `add_melded_assay` has not
published, and raise `ValueError` naming it before they write anything. Once no process is
writing it, remove it with `discard_interrupted_assay` on a `DataStore` of its workspace.

```{eval-rst}
.. autofunction:: cytearc.storage.destinations.check_destination

.. autofunction:: cytearc.storage.destinations.create_destination

.. autofunction:: cytearc.storage.destinations.refuse_pending_assays
```

```{eval-rst}
.. autoclass:: cytearc.writers.CrToZarr
    :members:
```

`MtxToZarr` is an alias of `CrToZarr` for use with `MtxReader`.

```{eval-rst}
.. autoclass:: cytearc.writers.MtxToZarr
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.writers.H5adToZarr
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.writers.SparseToZarr
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.writers.CSVtoZarr
    :members:
```

```{eval-rst}
.. autoclass:: cytearc.writers.SubsetZarr
    :members:
```

```{eval-rst}
.. autofunction:: cytearc.writers.to_h5ad
```

```{eval-rst}
.. autofunction:: cytearc.writers.to_mtx
```

```{eval-rst}
.. autofunction:: cytearc.writers.chunked_to_zarr
```

```{eval-rst}
.. autofunction:: cytearc.writers.create_zarr_dataset
```

```{eval-rst}
.. autofunction:: cytearc.writers.create_zarr_obj_array
```

```{eval-rst}
.. autofunction:: cytearc.writers.create_zarr_count_assay
```

```{eval-rst}
.. autofunction:: cytearc.writers.subset_assay_zarr
```

```{eval-rst}
.. autofunction:: cytearc.writers.write_renorm_subset_to_zarr
```

## Selection and layout behavior

{py:meth}`cytearc.datastore.datastore.DataStore.to_anndata` supports cell selection plus either `feature_names` or `feature_indexes`.
The two feature selectors are mutually exclusive.
`to_anndata` needs the optional `anndata` package and raises `ImportError` without it; install it
with `pip install 'cytearc[extra]'`. `to_h5ad` writes with h5py, which CyteArc always installs, and
needs no `anndata`, with or without a `run`.
`SubsetZarr` selects cells of one `DataStore` but retains every feature in each assay it writes.
It keeps the membership column `<assay>_I` of each assay it writes, for the selected cells, and
does not copy the membership column of an assay that `assays=` leaves out
([Assay membership](../developers/zarr-internals#assay-membership)).

Without a `run`, `to_h5ad` and `to_mtx` export a complete assay; `to_h5ad(..., cell_selection=ref)`
exports the cells of a cell-selection artifact instead.
For feature-selective disk export outside a pipeline run, call `to_anndata` and use AnnData's
writer.
`to_mtx` writes a feature ID, name, and type column in the features file whether or not it
compresses, `features.tsv.gz` with `compress=True` and `genes.tsv` without, so `CrDirReader`
reads an ADT export back as ADT, and it compresses with gzip level 6.
A Matrix Market directory holds no cell metadata, so `to_mtx` cannot say which cells an assay
measured, and a cell that it did not measure would be written as a zero row that `MtxToZarr`
reads back as a measured cell. `to_mtx` therefore raises
{py:exc}`~cytearc.metadata.membership.UnmeasuredCellsError` before it writes anything when the
assay's membership column `<assay>_I` marks a cell unmeasured ([Assay membership](../developers/zarr-internals#assay-membership)). Export
the measured cells from a subset that `SubsetZarr(..., cell_key="<assay>_I")` writes, or use
`to_h5ad` or `to_anndata` without layers, which declare the membership.

`to_h5ad` streams the matrix into the file one block of rows at a time, so it never holds the
complete matrix, and it converts a dense block to CSR a bounded step of rows at a time, which the
block's stream charges against the memory budget. It writes a temporary file in the same directory
and moves it onto the target once complete, so a failed export leaves no partial file and keeps an
earlier file of the same name; a replaced file keeps its permissions. Without a `run`, text without missing values is a string array and
both indexes are stored as `_index`. A column named `_index`, which AnnData reserves, as stores
imported before this release may hold, exports as `source_index`, and a column whose name the
export already uses, such as a `gene_short_name` beside the feature names that `to_h5ad` writes
under that name, takes the first free `<name>_2`, with a warning; the store is not changed.

A column that keeps a source categorical's `levels` and `ordered` attributes, as H5AD and Seurat
imports store them, exports as that categorical, with its category order, unused categories, and
`ordered` flag, and the colors of its `display` attribute as `uns["<column>_colors"]` in category
order. A value that the levels do not list is appended to the categories with a warning.

Exported raw counts take the dtype that `dtype` names. Without it they keep the stored dtype,
except that `uint8` and `uint16` counts become `float32`: arithmetic on narrow unsigned integers
wraps silently, as `X * 300` does in `uint16`, and `numpy.log1p` of `uint8` returns `float16`,
while `float32` holds every such count exactly and is the dtype that Scanpy's readers return. A
count that an explicit dtype cannot hold raises `OverflowError`.

Nullable metadata columns keep a stored placeholder in each row that their linked missing mask
flags. `to_anndata` and `to_h5ad` export those rows as missing values with or without `run`:
numeric columns become float64 with `NaN`, boolean columns become nullable booleans, and other
columns hold a missing value, which H5AD files store as a missing category.

### Artifacts in exports

`to_anndata` and `to_h5ad`, with or without a `run`, export artifacts beside the cells:

```python
adata = ds.to_anndata(
    cell_selection=cells,
    obsm={"X_pca": pca, "X_umap": umap},
    obs={"leiden": leiden, "s_score": (cell_cycle, "s_score")},
    graph=graph,
)
cytearc.to_h5ad(ds.RNA, "analysis.h5ad", cell_selection=cells, obsm={"X_pca": pca})
```

`obsm` takes two-dimensional per-cell arrays, such as a `reduction`, a `batch_correction`,
`imported_coordinates`, or an `embedding`. The entry named `X_pca` also writes the loadings of a
PCA as `varm["PCs"]`, with a row per exported feature and zeros for features that the PCA did not
use, as Scanpy stores them; imported coordinates with a `stdev` add `uns["pca"]["variance"]`.
`obs` takes one-dimensional per-cell arrays: labels, such as a clustering, become categoricals of
text labels in natural order, or of the categories and colors that imported labels keep, and
scores stay numbers. A pair `(ref, name)` exports another per-cell array of an artifact. Every
artifact must hold every exported cell; `ValueError` names the remedy otherwise, which is to export
the cells of its cell selection with `cell_selection=`.

`graph`, a `connectivity_map` or `integrated_graph` built over exactly the exported cells, becomes
`obsp["connectivities"]`, the fuzzy union `W + W.T - W * W.T` of its directed weights, which
CyteArc's UMAP reads and Scanpy expects to be symmetric, and, for a connectivity map,
`obsp["distances"]`, its KNN distances with `k` entries per cell. `uns["neighbors"]` follows
Scanpy's layout: `connectivities_key`, `distances_key`, and `params` with `n_neighbors` (`k + 1`,
because Scanpy counts each cell among its own neighbors), `method`, `metric`, and `use_rep` when
the graph's coordinates are also exported in `obsm`. Exporting a graph holds about 96 bytes per
stored edge, which is checked against the memory budget. `uns["cytearc"]["artifacts"]` records the
reference of every exported artifact by element, such as `uns["cytearc"]["artifacts"]["obsm"]["X_pca"]`.

The two meanings of `matrix="normed"` differ. Without a `run`, `to_anndata(matrix="normed")`
exports the assay's `normed` values of the selected cells and features: for RNA, counts scaled by
each cell's library size over all of its features to 1,000, without a log transform. A run export
with `matrix="normed"` exports the run's stored `normalized` artifact over the run's selected
features only, which are then the only features: for an RNA run, log1p of counts scaled over its
highly variable genes, as described below.

Without `run`, `to_anndata` and `to_h5ad` export the membership column `<assay>_I` of the
exported assay as an ordinary `obs` column and declare it in `uns["cytearc"]["assayMembership"]`,
for example `{"RNA": "RNA_I"}`, so `H5adToZarr` restores it as the membership of an imported
assay of that name and a merged store survives export, import, and a second merge. The membership
columns of other assays, including the assays of `to_anndata` `layers`, describe assays that the
file does not hold and are not exported, as a merge or subset that leaves an assay out does not
copy its membership column; an import would otherwise keep them as ordinary metadata, which a
later merge with `prepend_text=None` that includes those assays would refuse. A run's frozen fields
declare no membership. Because a layer of another assay cannot declare its membership, every
exported cell must be one that the layer's assay measured: `to_anndata` with such a layer, with or
without `run`, raises `UnmeasuredCellsError` before it reads any value, and a `cell_key` of the
cells that the layer's assay measured exports them. A layer of the exported assay is declared with
it.

Pass a completed run to write its frozen cells, feature universe, and result fields directly:

```python
cytearc.to_h5ad(ds.RNA, "analysis.h5ad", run=run)
cytearc.to_h5ad(ds.RNA, "analysis_normalized.h5ad", run=run, matrix="normed")
```

The assay must be the run's assay, `run.assay`, as the exact assay object owned by the datastore
that opened the run. A run export
to a file holds exactly what {py:meth}`~cytearc.datastore.datastore.DataStore.to_anndata` returns
for the same `run` and `matrix`, because one resolution of the run's rows, columns, and values
serves both: the run's cells, its frozen cell fields in `obs` with the consecutive UMAP fields in
`obsm["X_umap"]`, the PCA of an RNA or ADT run in `obsm["X_pca"]` with the loadings in
`varm["PCs"]` and its Harmony correction, if any, in `obsm["X_pca_harmony"]`, the LSI of an ATAC
run in `obsm["X_lsi"]` and its Harmony correction, if any, in `obsm["X_lsi_harmony"]`, and its
frozen feature fields in `var`. `to_anndata` holds that object in memory; `to_h5ad` streams it
into the file. The run's plan decides which fields are categoricals. Label fields, those with a
categorical display (`clusters`, `leiden_*`,
`paris`, and `cell_cycle_phase`), are categoricals whose categories are their labels as text in
natural order, with the run's display colors as `uns["<field>_colors"]`, so that Scanpy's
`rank_genes_groups(adata, "clusters")` and categorical plots work directly. Other text that repeats
or has missing values is a categorical whose categories are in natural order (`d2` before `d10`,
with a superscript or circled digit kept as text), and text missing in every row is a categorical
without categories. `to_anndata` returns those fields as pandas categoricals with exactly those
categories, and the file stores the same categories and codes, so the object, the file, and a file
that `AnnData.write_h5ad` writes from the object agree whatever the AnnData version. Other text is a
string array, and the indexes are named `ids` and `gene_ids`. The export stores `X` compressed,
stores distinct text as a string array, and leaves out empty `layers`, `obsp`, `uns`, `varm`, and
`varp` groups. `embeddings_cols`, feature-count recalculation, a cell selection, and a
writer-specific thread override apply only to ordinary assay export and are rejected when `run` is
supplied; `obsm`, `obs`, and `graph` add artifacts over the run's cells, such as
`graph=run["connectivity_map"]`.

With `matrix="raw"`, the default, `X` holds the raw counts of the run's cells over its feature
universe. With `matrix="normed"`, `X` is the run's `normalized` artifact: the float32 values that
{py:meth}`~cytearc.datastore.namespaces.FeaturesAccessor.normalize` stored for the run, over the
run's selected features in stored feature order: the highly variable genes of an RNA run, the
prevalent peaks of an ATAC run, or the panel of an ADT run. `var` then holds the frozen fields of
those features only, and `to_anndata` `layers` align to them. The values are read in the
artifact's stored row bands and never converted to float64. The export checks that the store's
dataset fingerprint still matches the artifact and raises `ValueError` when the run has no
normalized output, or when the artifact's cell selection is not the run's cell selection or its
feature selection is not the run's selected features. A multimodal run has no normalized output
and exports with `matrix="raw"`: `X` holds the raw counts of `run.assay`, its first source run's
assay, beside the run's joint fields, and `to_h5ad` takes that assay as its `assay`. Without a
`run`, `to_anndata(matrix="normed")` normalizes the selected cells and features with the assay's
`normed`, and `to_h5ad` rejects `matrix="normed"`.

`H5adToZarr.dump()` returns an `H5adImportResult` containing the written assays, analysis assay,
all-cell selection, and imported embedding and clustering refs. Set `analysis_assay` when a
multi-assay import selects analytical values. Use `DataStore.artifacts.load(ref)` for payload access
or pass the exact ref to a consumer. Import does not flatten these results into metadata columns.

Ordinary `to_h5ad` export writes a complete assay and live metadata. Run-aware export reads only
the completed run's frozen selections and fields, and for normalized values its `normalized`
artifact, so export does not require physical result columns.

`CrToZarr`, `MtxToZarr`, `H5adToZarr`, `SparseToZarr`, and `SeuratToZarr` select source batch
rows automatically when `batch_size` is omitted, starting from the smallest destination
row-shard height, which the fitted count layout admits. An explicit positive `batch_size` is
capped at that height.

## Merge

Use `DataStoreMerge` to merge DataStores.
Pass `assays=["RNA"]` when only one assay type is needed.
Features are matched by exact feature ID by default. Gene symbols are display labels and do not
merge distinct IDs; suffixes such as `_1` are preserved. An assay present in multiple inputs
raises an error if none of its IDs overlap.
When inputs use different identifier conventions for the same genes, for example Ensembl IDs in
one input and gene symbols in another, pass `feature_key="names"` to match features by name.
The merged feature IDs are then the names, and features that share a name within one input are
summed.
Feature annotation columns are merged when the inputs agree. A column whose values differ for a
shared feature, such as per-dataset highly variable gene flags, is left out with a warning. The same
applies when features that share a name within one input disagree.
Merged cell IDs are `{name}__{cell_id}`, so source names cannot contain `__`. Cell metadata columns
keep their attributes when the inputs agree. Differing `levels` of an unordered categorical column
become their union in first-seen order; any other differing attribute, including ordered levels, is
dropped with a warning.
Each merged assay keeps the type that its sources declare in `assayTypes`, such as `HTO` or
`GeneActivity`, rather than the preset of its assay class, so a merged HTO assay can be
demultiplexed. Sources that declare different types for one assay are rejected before anything is
written; reopen the source whose declaration is wrong with `zarr_mode="r+"` and a matching
`assay_types`.
RNA assays write both `counts` and a gene-major `countsT` copy, which roughly doubles stored counts for those assays.
Non-RNA assays never write `countsT`.
Every merged assay gets the membership column `<assay>_I` described in [Assay membership](../developers/zarr-internals#assay-membership). A
merged cell keeps its source's membership: False when the source lacks the assay, True when the
source measured every cell with it, and the source's own value when the source records partial
membership. A source's membership column is not merged again as an ordinary column such as
`orig_<assay>_I`, and the membership column of a source assay that `assays=` leaves out is not
merged at all, so a merge with `assays=["RNA"]` writes no `ADT_I` or `orig_ADT_I` that a later
merge including ADT would trip over. Planning rejects a malformed source membership column, an
ordinary column whose merged name equals a membership column, and a source whose cells outside an
assay have counts of it. A source column named
`<assay>_I` for a merged assay that lacks the membership attributes raises `Cell column
'<assay>_I' is reserved for the membership of assay ...`. Import the file again, which
restores a declared membership and otherwise skips the column with a warning, or drop the plain
column from the source with `source.cells.drop("<assay>_I")`, after which the source counts every
cell of the assay as measured. `MetaData` writes no membership column itself: `insert`,
`update_key`, and `reset_key` refuse `<assay>_I` for any assay of the store, and `drop` refuses one
that carries the membership role ([Assay membership](../developers/zarr-internals#assay-membership)).
The destination must be empty or hold a merge written by `DataStoreMerge`. A destination that
aliases, contains, or lies inside a source store, or that already holds other content, is refused.
`overwrite=True` replaces only a merge-owned destination whose assays have not been prepared by
opening it as a `DataStore`, and it clears the recorded default assay.
Interrupted merges resume at whole-component boundaries (`cellData`, each assay `counts`, and each RNA `countsT`) rather than mid-matrix.
A resume requires the same configuration, the same source counts, assay types, and membership; a
completed `countsT` whose layout differs from the plan is not rewritten in place. The manifest
must record `assayTypes` and `sourceMembership`; without those fields, restart with
`overwrite=True`. Completed counts keep the layout persisted
with them, so a resume under another `mem_budget` reuses them, while counts that a resume rewrites
get a layout fitted to its budget.

```{eval-rst}
.. autoclass:: cytearc.merge.DataStoreMerge
    :members:

.. autoclass:: cytearc.merge.MergePlan
    :members:

.. autoclass:: cytearc.merge.AssayMergePlan
    :members:

.. autoclass:: cytearc.merge.MergeResult
    :members:

.. autoclass:: cytearc.merge.ComponentResult
    :members:
```
