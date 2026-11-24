---
description: Inspect the CyteArc Zarr layout for count matrices, assay metadata, immutable artifacts, and saved pipeline runs.
---

(zarr_internals)=
# Zarr internals

This page is for contributors and advanced users who need the on-disk layout.
Analysts should start with the [cell and feature metadata tutorial](../tutorials/data_organization.ipynb).
Why RNA stores two orientations, and what to do when an older store will not open, is in [](../concepts/memory_and_execution.md).

## Layout overview

A CyteArc Zarr store is a directory hierarchy.
Typical top-level groups include:

- cell metadata
- one or more assay groups (for example `RNA`, `ADT`, `ATAC`)
- per-assay feature metadata
- immutable feature selections, summaries, normalized matrices, reductions, and graph artifacts under assay-specific artifact paths
- durable pipeline records under `pipeline/runs/{runId}` with ordered stage records below each run

Artifact group names use kind and identity, not encoded graph-stage parameters.
Prefer inspecting a store with `zarr.open` or CyteArc's `DataStore` summary rather than hard-coding internal paths in analysis scripts.

Analytical outputs are artifacts only. Feature selection, embedding, clustering, score, and marker
operations leave metadata tables unchanged. Consumers validate exact refs,
artifact completion, lineage, and ordered-axis identity.

(cell_column_chunks)=

## Cell and feature column chunks

A metadata column stores text as fixed-width unicode at the width of its widest value, four bytes a character, and a read of any of its rows decodes the whole stored chunk that holds them. Every writer of a cell table, `cellData` or a workspace's, and of an assay's `featureData`, therefore chunks a new column by `storage.layout.column_chunk_rows`: a chunk holds at most 100,000 rows, no more rows than the table has, and no more than 16 MiB of values, and always one row at least. A store of 48 cells keeps each column in one chunk of 48 rows, a text column wider than 41 characters takes fewer than 100,000 rows to a chunk, 20,971 at 200 characters, and a value wider than 16 MiB is a chunk of its own. A missing-value mask takes the chunks of its column. The imports, `cells.insert` and `feats.insert`, `update_key` and `reset_key`, the membership columns of imports and derived assays, subset, merge, repack, mount, and composites write cell and feature columns this way, so the 64 guide IDs of a screen are one chunk of 64 rows; the Seurat import caps its metadata and membership columns at 65,536 rows, the rows of one of its blocks. A pseudotime or lineage feature analysis freezes the names and IDs of its assay's features by the same rule, where analyses saved before keep chunks of 100,000 rows, which their checks decode whole. A column replaced through `MetaData` is chunked by the rule in either table. Stores written before keep their chunks. They are read as stored, and every read is costed by the stored chunks of the array it reads, so a read of 48 rows of a column stored in 100,000-row chunks is charged that chunk. A column written into such a store, or written again, takes the new chunks.

(categorical_columns)=

## Categorical columns

A label column can store integer codes and its categories instead of a text value per row, as AnnData stores a categorical. A categorical column `<name>` stores one code per row, the position of the row's category or -1 for a missing row, in the smallest signed integer type that holds the position of every category: int8 up to 128 categories, int16 up to 32,768, then int32 and int64. Its categories are the one-dimensional array `__cytearc_categories__<name>` beside it, every category in its order, unused ones included: text as fixed-width unicode at the width of the widest category, or int64, float64, complex128 or bool, as `MetaData.insert` types values. The codes name the categories array in their `categories` attribute, as a nullable column names its mask, and the categories array records in its `ordered` attribute whether the categories are ordered. Codes that hold -1 also link the missing mask `__cytearc_missing__<name>`, which flags exactly those rows, so every reader of missing values works on either layout. The codes and their mask are chunked by `column_chunk_rows` from the codes' itemsize, so a column of 600,000 cells with a few hundred categories is six chunks of 100,000 int16 codes, and the categories by the same rule over their own rows. A release before categorical columns opens a table only when each of its arrays holds one value per row, so the categories array never holds as many entries as its table has rows: when the categories number exactly the rows, it holds one reserved entry after the last category, the placeholder of their dtype, which no code names, and records it with `padded: true`. Such a release therefore refuses every table that holds a categorical column, with `Not all columns are of same length`, rather than read its codes as values, and this release leaves the entry out of every read, so labels, `fetch_categorical`, fingerprints and identities do not depend on it. `cells.insert` and `feats.insert` write this layout for a pandas `Categorical` or a Series of categorical dtype, and the text layout for other values; `I`, `ids` and `names` always store their values.

`H5adToZarr` writes this layout by default for each categorical of `obs` and `var`, in AnnData's current encoding and in the legacy ones that keep its categories in a `__categories` group beside the columns or in `uns/<column>_categories`. A code below 0 or past the last category becomes -1. The categories keep their source order, unused ones included, and take the type that the labels of an import with `categorical_columns=False` take, which the categories that rows hold decide: text, or int64, float64, complex128 or bool. Every category takes it, so a column whose rows hold no category stores text categories, as such an import stores the column as text. Each assay of a split import types the categories from the codes of its own features. The codes carry the `levels`, `ordered` and `display` attributes that the column of labels carries, so plots, merges and exports read the same attributes. A categorical of more than 10,000 categories, or whose categories this layout cannot store, such as categories that read alike once stored, NaN, or numbers beside text, is stored as a column of its labels instead, with a warning that names the column and the reason. CyteArc 1.4.0 is the first release that reads this layout, so `categorical_columns=False`, which stores every column as its labels, writes a store that earlier releases open.

Every read returns a categorical column's labels: `MetaData` hands it out as a `CategoricalColumn` (`cytearc.storage.categorical`), which has the shape, chunks and attributes of the codes and the dtype of the categories, selects codes as Zarr selects them and returns the categories at them, with the placeholder of their dtype in missing rows, empty text, 0, NaN or False, as a text column holds there. `fetch`, `fetch_all`, `to_pandas_dataframe`, `head`, `get_index_by`, row blocks, groupings, samples, `artifacts.load_values` of a `CellField` and plots therefore see the same labels from either layout, and `MetaData.fetch_categorical` returns the codes and categories as a pandas `Categorical`. The reads that a memory ledger charges read a categorical column's codes and charge them, its categories and the labels they return, and the checks that plan a read of selected cells count the same: `read_text_rows` returns one Python string per category, which the rows share, rather than a string per row. Groupings work on the codes and show the categories: `compare.compute_bulk` compares each group's code rather than its label, clone summaries that group cells by a column (`cell_groups`) code them through their categories, and so do the samples of guide calling, perturbation signatures, Mixscape, perturbation profiles and associations, and hashtag demultiplexing, which read a pandas `Categorical` of the cells' codes and decode no label; their groups, names and fingerprints are those of the labels. Plots decode the codes of the plotted cells alone, and when a plotted cell's label is missing, into one Python string per category. Exports read the codes and categories through `fetch_categorical`: `to_anndata` and `to_h5ad` write a categorical column as an AnnData categorical of every category in its stored order, the reserved entry left out, so an `H5adToZarr` import of the file stores its column as the same codes and categories, while a text column of the same labels exports its labels.

Fingerprints built from labels do not depend on the layout. `metadata_column_fingerprint`, the column fingerprints of run snapshots and the snapshots themselves decode a categorical column at the dtype that a column storing its labels has, text as wide as the widest label a row holds, so the same labels give the same fingerprint and the same artifact identity in either layout; sample assignments, groupings and `snapshot_cluster_labels` fingerprint the labels they read. `metadata_column_stamp`, which `clones.from_column` records without decoding the column, stamps the stored codes and categories, so it differs between the layouts, as it differs between two text columns stored in other chunks.

Subset, repack and mount copy a categorical column as codes and categories: the codes of the rows they copy and every category, with a reserved entry when the categories number exactly the copied rows. A merge writes the labels of a categorical column as a column of values. A link that names another array, codes that are not signed integers or too narrow for their categories, a categories array of another kind or without `ordered`, a `padded` attribute that is not true or whose array does not hold one entry more than the column has rows, and a code outside -1 and the last category raise `ValueError` when the column is read, and a table whose `I`, `ids` or `names` is categorical raises `ValueError` when it opens. Stores written before hold no categorical column and read, fingerprint and copy as before.

(cell_order)=

## Cell order

An import that shuffles cells, as every import does unless `shuffle_cells=False`, writes row `r` of every per-cell array from row `order[r]` of the cells it imports: `counts` and `countsT`, the cell table's columns, assay membership, and imported embeddings, cluster labels, reductions and active identities. The cells it imports are the file's rows, or for a reader that drops cells, such as `CrH5Reader(is_filtered=False)`, the rows it keeps, in the file's order. `storage.cell_order.shuffled_order(n_rows, seed)` gives `order`: each row index, plus SplitMix64's increment, goes through SplitMix64's finalizer, its result is XORed with the finalized seed, salted with the ASCII text `CellOrdr`, and finalized again, and the rows are ranked by that hash, ties by index. The arithmetic is exact in 64 bits, so a seed gives the same order everywhere.

The int64 cell column `sourceRow` holds each cell's row in the imported file, from 0: for a reader that drops cells, the file row of each cell it keeps, its barcode's index, so the column has gaps where the dropped cells were. Readers report these rows through `CrReader._file_rows`. The column is chunked as every new cell column is, with the attribute `role="source_row"`. The cell table's `cellOrder` attribute records the shuffle as `{"shuffled": true, "seed": <seed>}`, which `recorded_shuffle_seed` reads, refusing any other value. An import in the file's order, `shuffle_cells=False`, writes neither, so its store is the one that imports wrote before shuffles existed. [Why CyteArc shuffles cells](../concepts/cell_order.md) explains the order for users. `source_order_rows(cells, selection)` returns the selected rows sorted by `sourceRow`, or in store order for a table without a `sourceRow` of that role. `protect_metadata_column` refuses every insert, update, reset or drop of a cell column named `sourceRow`, except the drop of a column without the role, such as a source column that an import in the file's order kept. The name is not among the reserved columns that every table lists, so stores without it open as before.

The rows travel through `readers/_sparse.SparseRowStore` in the reader's `temp_dir`, which `readers/_row_spill.py` plans alike for every format. As the value scan of an H5AD, 10x HDF5 or Tapestri import reads each block of rows, it appends them, tagged with their destination rows, to its own writer part of the store; dense rows go in as sparse rows, their zeros left out. A CSC H5AD matrix's row conversion appends each entry at its destination row. Matrix Market, CSV and Seurat imports append batches of rows that they read in the file's order: Matrix Market and CSV imports as they read the rows for the counts write, before they create the store, and a Seurat import during the value scan of each assay, one store for each. `SparseToZarr` gathers each batch's rows from its matrices in memory and writes no temporary files. Buckets hold consecutive destination rows, sized to read in 256 MiB, or a thirty-second of `mem_budget` when that is less, independently of the count shards, whose dtype the scan has yet to decide. Each producer of the counts write reads its window of destination rows a bucket at a time with the next bucket loading ahead, and deletes each bucket that lies wholly inside its window once read; the writer removes the buckets that straddle two windows when the counts are written. Before it writes, an import checks the free disk against a quarter more than the stored values take at the rate of the source's first million values, encoded as the store encodes them, and 16 bytes for each row's place and length, at most what their two codes take; dense rows store their nonzero values, whose number is estimated from the share of them among those first million values. The measured rate holds the places and lengths of the measured rows spread over their values, so it would charge rows of fewer values too little for theirs, as when a file holds its densest cells first. A CSC conversion checks against the values before compression. The check of every store, an export's too, also counts what the store writes beside the values: a block for each part in each bucket, its settings, and its records. The record of each part, the owner's among them, keeps each row's int64 count and, for whole rows, its uint8 mark, and the finalized record the int64 row pointers, so an export's records take up to 17 bytes per row, and a shuffle's 9 more for each part beside the owner's: one in a single process, and one for each value-scan process of an H5AD import. Every writer checks the free disk again before each block it writes, which catches a disk that fills, or dense rows that hold more nonzero values than the first rows suggested.

Exports write cells in the file's order by default. `exported_cell_order(cells, rows, cell_order)` gives the positions of the exported store rows in that order, from `source_order_positions`, or None for store order: with `cell_order="stored"`, for a table without a `sourceRow` of that role, and where the rows' source rows never decrease. In-memory exports gather their values at those positions. `writers/export.reordered_matrix` writes the counts of `to_h5ad` and `to_mtx`, and a run's normalized values and layers, through a `SparseRowStore` of its own: the stored blocks stream in store order, converted to CSR a step of rows at a time, gathered into batches of up to 2,097,152 values that a thread appends as whole rows keyed by their output row while the stream converts the next, and the export reads the buckets back in output order, deleting each once read. The number of values sizes the buckets and the disk check, at the bytes per value that the first converted rows encode to, beside 16 bytes for each row's place and length. `to_h5ad` and `to_mtx` take it from the count summaries of the exported cells (`storage.identity.nonzero_value_count`). Where the summaries cannot give it, for counts that can be negative in a store whose summaries predate `rowNonzero`, `to_mtx` passes the count of its header and `to_h5ad` the count of its preliminary pass when it makes one; otherwise, as for a run's values, whose feature universe the summaries of whole rows cannot count, the reorder counts the values in a read of the matrix before it writes a row. The reorder holds the order and the destination of each row and the bucket writer's count and mark of each, 25 bytes per row, and then the row pointers, 8 bytes per row; the batches and the pending rows take what the budget leaves, and reading a bucket back may take a third of what the row pointers leave of the budget, which bounds the widest row that an export reads back. `writers/export.reordered_text_rows` sends the rows of `repertoire.export_airr`, rendered in store order, through a `SparseRowStore` of ragged rows: whole rows of bytes, whose blocks store no column indices and are not compressed, appended by a thread while the next block renders and read back a bucket at a time with the next one loading ahead. `plan_text_reorder` sizes the buckets to read in about one of the export's blocks, from the widest row that the fields' longest values allow, and keeps at most a block of rows waiting, so the reorder holds a few blocks beside the export's own; the first block estimates the rows' bytes, which size the buckets and the disk check. Compressing those blocks at zstd level 1, as sparse rows are stored, cut the files to a third of the rows' bytes and took half to three quarters of the time that the reorder added to an export; Blosc decodes either kind of frame from its header.

A merge writes `sourceRow` for its own cells: the offset of the cell's source, the cells that the merge takes from the sources before it, plus the cell's rank among its source's cells in their export order (`merge.metadata.merged_source_rows`). The column carries the `source_row` role. A merge or composite interleaves its sources' cells in the order that `shuffled_order` draws over their concatenation, and each source gives its cells in its order from a first cell that the seed draws, wrapping round (`merge/row_plan.py`). When every source's cell table records a shuffle and their seeds differ, the merged order is uniformly random too, and the merged cell table records `{"shuffled": true, "seed": <seed>}` with the seed of the interleaving (`merge.metadata.merged_shuffle_seed`); otherwise it records none, as sources that share an import seed make it only close to uniform. A subset, a mount and a repack, a data-only one included, keep the store's order of rows, so they copy `sourceRow` as stored and the `cellOrder` record with it (`copy_cell_order`). Since readers that drop cells and subsets leave gaps in `sourceRow`, exports and merges only sort and rank its values and never take one as a position.

## Composite sources and virtual counts

A composite has ordinary cell and feature metadata, preparation summaries, and analysis artifacts. Its root `composite` attribute records `assay`, `sources`, `arrays`, `maps`, and `complete`. Each source binding holds its `name`, `location`, `workspace`, `datasetFingerprint`, and `countsFingerprint`. Source credentials are supplied at reopen through `DataStore(source_storage_options={source_name: options})` and are never persisted.

`create_composite` writes the composite to any destination that the store layer opens: a local path, a URL with the `storage_options` that open it, or a Zarr store. New arrays take the storage profile of the destination, as other writers' do, and a destination that holds anything raises `FileExistsError`. The cell column of source names is `dataset` unless `source_column` names another, checked as `DataStoreMerge` checks its own.

(source_locations)=

A composite records a source in the directory that holds the composite, or below it, by a path relative to that directory, with `/` separators, such as `a.zarr` or `sources/a.zarr`, when the source and the composite are both local or both at URLs of one scheme and host. Any other source's `location` is its absolute path or URL. Opening resolves a relative location against the directory of the path or URL that the composite is opened from, so a directory that holds a composite and its sources can be moved, copied, or served from a bucket, and the identity checks still refuse a source with other content at that place. A composite opened from a store object that has no path or URL, such as an object store, cannot resolve a relative location and raises `ValueError`. A mount's `location` follows the same rule. CyteArc 1.4.0 is the first release that resolves a relative location; an earlier release resolves it against its working directory, so it finds the source only when run from the directory that holds the store. `relative_sources=False`, which `create_composite` and `mount_datastore` take, records every source by its absolute path or URL, so the store opens in every release and can move without its sources.

The `_composite` group stores `rowSources` and `sourceRows`, which map output cells to source
names and row indices, and `featureColumns`, which maps output features to columns in each
source. A feature column of `-1` contributes zeros. The manifest records mapping fingerprints;
reopening verifies these and the saved source identities. Reads load mapping windows as needed.

`arrays` holds the virtual Zarr metadata for `{assay}/counts` and, when required by the assay,
`{assay}/countsT`. Their `zarr.json` documents and chunks are not stored as ordinary arrays:
the adapter supplies them during a CyteArc open. Bare Zarr therefore cannot silently substitute
zeros for the missing counts. Virtual reads gather source chunks in the requested orientation.
They do not create a stored composite transpose. Published mappings and count metadata are
immutable; analysis artifacts remain writable. A failed creation leaves `complete=False`, and
CyteArc refuses to reopen it.

CyteArc reads the virtual counts through the composite store's `VirtualArrayReader` rather than
through computed chunk bytes. A read gathers only the selected values and, for each source,
reads whichever of its `counts` and `countsT` touches fewer chunk bytes. Feature streams batch
adjacent feature groups into one read when the budget allows, so a source chunk is decoded once
per batch. Zarr reads of the computed chunks remain correct for code that reads the arrays
directly.

Remote sources read through a temporary cache in the system temporary directory (`TMPDIR` on
Linux and macOS). It keeps the encoded shards that a read covers densely and the shard-index
suffixes of sharded reads, up to `CYTEARC_SOURCE_CACHE_BYTES` per open composite: bytes or a size
such as `512M`, 2 GiB when unset, and `0` to read the sources directly. The directory is created
by the first fetch and removed when the store closes or the process exits. A fault of the local
directory turns the cache off, and reads continue from the sources.

The manifest uses a strict, unversioned shape. Missing or unknown fields fail closed; there is
no automatic migration. Changing this persisted shape requires an explicit compatibility
decision before changing the reader or writer.

Sources must remain available and unchanged. Nested composites, mounts, and repacking are
unsupported. Existing `SubsetZarr` and `DataStoreMerge` writers can read a composite and create
ordinary stores with copied counts; there is no separate composite materialization API.

## Artifact publication

A producer creates an artifact group with `complete=False` at a path that ends in its random 256-bit `artifact_id`, so the path belongs to that producer alone. After it validates the payload, one metadata write sets `complete=True` and publishes the artifact. The producer deletes the group that it created after an exception only before that write; once the write is issued, the group stays, complete or not, because the write can persist even when it reports an error. An interruption (`KeyboardInterrupt`, `asyncio.CancelledError`, the `concurrent.futures.CancelledError` that Zarr's synchronous API raises for a cancelled call, `SystemExit`, or a shutdown request) keeps the group with `complete=False`, as derived assays keep a pending assay (Rule C below), because Zarr may still be running the interrupted write. `DataStore.artifacts.prune_incomplete` removes the incomplete groups and the artifact-ID directories that hold chunks but no `zarr.json` once no process writes to the store.

Producers in separate processes write the same key only for a group above their artifacts that does not exist yet, such as `{assay}/artifacts/{kind}` for the first artifact of a kind. Zarr writes such a group, which holds no attributes, with `set_if_not_exists` as it creates the group below it, so of processes that create it at once none fails, and on a store whose conditional writes are not atomic each writes the same bytes. A pipeline run's group is created by its full path for the same reason, so that runs which start at once on a store without runs make `pipeline` and `pipeline/runs` together. Processes that plan one provenance before any of them publishes create one artifact each; later plans reuse the one with the largest `(created_at_ns, artifact_id)`, and [Several processes on one store](../concepts/provenance.md#several-processes-on-one-store) states what users may run at once.

Listings and planning read each child of an artifact kind on its own when one child's `zarr.json`
cannot be read, and skip that child and any kind that the release does not know with a warning.

Producers check the arrays that must hold only finite values as they write them
(`cytearc.storage.finite_values`): a NaN or infinite value raises `NonFiniteArtifactError`, a
`ValueError` that names the operation, the array, and the row, and nothing is published.

(normalized_feature_statistics)=

### Normalized feature statistics

A `normalized` artifact holds `data`, the float32 values with one row per selected cell, and two
float64 arrays with one value per feature: `feature_sum`, the sum of the stored values, and
`feature_m2`, the sum of their squared deviations from the feature mean. Feature scaling for PCA
reads the mean `feature_sum / n_cells` and the standard deviation `sqrt(feature_m2 / n_cells)`
from them without reading `data`. Each stored row band adds its sums and merges its `feature_m2`
with the deviation of its mean from the running mean (`cytearc.utils.moments.ColumnMoments`), so
the variance never subtracts a squared mean from a mean of squares: it is never negative, and a
feature whose stored values are all equal has a `feature_m2` of exactly zero and a scale of 1.

Reuse requires `data` and `feature_sum`, but does not require `feature_m2`. If `feature_m2` or
`feature_sum` is unavailable, feature scaling streams the means and standard deviations from
`data`. Running `features.normalize` with `invalidate_cache=True` writes `feature_m2` and lets
scaling read it directly.

An artifact group's attributes hold `artifact_id`, `kind`, `provenance`, `execution_options`,
`created_at_ns`, `cytearc_version`, and `complete`. `provenance` is a mapping with `operation`,
`parameters`, and `inputs`, serialized to canonical JSON values, and with an integer `revision` only
when the operation revision is 2 or more. An artifact that records `revision` 1, or a revision
that is not an integer, is not current and is never reused. An artifact without `revision` is
revision 1, so stores written before operation revisions existed keep their identities. Reuse
compares the whole `provenance` mapping exactly; `execution_options`, `created_at_ns`, and
`cytearc_version` never affect it. [](operation_revisions.md) explains when a revision changes.

(marker_tables)=

### Marker tables

A `marker_table` artifact from `markers.search` holds three arrays with one value per feature, `feature_index`, the assay index of each tested feature, and `feature_names` and `feature_ids`, the names and IDs that the search froze, and one child group for each tested group. A child group records the group's cell count in its `n_group` attribute and the count of the other cells in `n_reference`, and holds the group's statistics in `stats`, one row per tested feature and one column per name in the slot's `stat_columns` attribute. The slot's other attributes record the test, its adjustment, and its fold-change and p-value policies.

Each child group is named by its group's label, and the slot's `skipped_groups` attribute maps the label of each group too small to test to its cell count. Some labels cannot name a child group, because Zarr would refuse the name or read it as a path or as another node in either Zarr format, or because the slot already uses it: an empty label, one that contains `/`, `\` or a NUL character, the paths `.` and `..`, the metadata documents `zarr.json`, `.zattrs`, `.zgroup`, `.zarray` and `.zmetadata`, and the names of the slot's arrays. Any other label, `.x` among them, names its child group as in earlier releases. A table with a label that cannot stores its groups by code instead, as AnnData stores a categorical. The slot's `group_labels` attribute lists every label, in the order that the search sorts them, which is the categories; each tested group's child group is named by its code, the position of its label in `group_labels`; and `skipped_group_codes` maps the code of each skipped group to its cell count. `markers.load`, `markers.export_csv` and `marker_heatmap` read either layout and show labels in the same natural order, and `markers.search` reuses a table of either layout for the same labels.

A table whose labels can all name child groups keeps the layout of earlier releases byte for byte, so they read it as before. A table stored by code needs a release that reads that layout: an earlier release finds no `skipped_groups` in it and raises `ValueError` rather than showing codes as labels. No earlier table is stored by code: earlier releases refused labels with `/` or `\` and the labels `.` and `..` before they searched, and failed on `zarr.json`, a NUL character and the slot's array names when they wrote the table. They stored a group labeled `.zattrs`, `.zgroup`, `.zarray` or `.zmetadata` under that name in a Zarr v3 store, where it is no metadata document; this release stores such a label by code and still reads and reuses those tables.

## Pipeline records and snapshots

Pipeline run, stage, and label-claim documents use one strict unversioned shape. A run has exactly `runId`, `recipe`, `requestedLabel`, `label`, `assay`, `startedAtNs`, `finishedAtNs`, `status`, `complete`, `cytearcVersion`, `config`, `stageOrder`, `outputs`, `fields`, `error`, and `interruption`, and may also hold the optional `outputLists` and `segments`. A stage has exactly `stage`, `ordinal`, `startedAtNs`, `finishedAtNs`, `status`, `complete`, `outputs`, `plans`, `metrics`, `error`, and `interruption`, and may also hold the optional `summary`, `warnings`, and `outputLists`. A label claim has exactly `label` and `runId`. Unknown or missing fields fail closed.

Each entry of a run's `outputs` names one artifact, `{"key", "artifact"}`, and each entry of a stage's `outputs` one artifact with whether the stage reused it, `{"outputKey", "artifact", "reused"}`. An output that holds several artifacts in a fixed order is an entry of `outputLists` instead: `{"key", "artifacts"}` on a run, with a non-empty list of distinct refs, and `{"outputKey", "artifacts"}` on a stage, with one `{"artifact", "reused"}` for each. A key names one output of a record, in `outputs` or in `outputLists`, and only a completed run or stage holds either. The recipes' `leiden` output, their Leiden candidates ordered by resolution, is the one output list that CyteArc writes.

Within 1.x a release may only add fields, and a release that adds one must still read records written without it. No release removes, renames, retypes, or reinterprets a field, so a run of any 1.x release reads in every later 1.x release. The stage fields `summary` and `warnings` were added this way: CyteArc writes both on every stage record it creates, and reads a stage record of an earlier release, which lacks them, as one without a summary or warnings. The stage metrics `cpuSeconds`, `threadsPeak`, `rssHighWaterStartBytes`, `rssHighWaterEndBytes`, `networkRxBytes`, `networkTxBytes`, `resources`, `trace`, and `unavailable` were added the same way: CyteArc writes all nine on every stage record, and reads metrics that hold none of them, as earlier releases wrote them, with each as null, while metrics that hold only some of them fail closed. The run and stage field `outputLists` was added the same way: CyteArc writes it on every run and stage record, empty when the record has no output list, and reads a record without it as one without output lists. The run field `segments` and the run `status` `paused` were added the same way, but CyteArc writes `segments` only on the record of a run that paused, so the record of every other run keeps the shape that earlier releases read; they skip a record with `segments` when they list runs and refuse it when they open it, as they refuse a status that they do not know. Runs of earlier releases keep the outputs, stages, and fields that they recorded: one `leiden_<resolution>` output, stage, and cell field for each Leiden candidate, and no `leiden` output. A run's `config` and a field descriptor's `display` are JSON mappings whose readers ignore keys that they do not know: a later release may add keys to them, and code that reads such a key treats its absence as the behavior of the release that wrote the run. Run plots check the keys that they know. Each entry of a categorical display's `categories` holds exactly `value`, `label`, and `color`, and run plots refuse an entry with any other key. CyteArc 1.4.0 and earlier releases refuse to plot a field whose display holds a key that they do not know.

A run's `recipe` names the recipe that wrote it: `rna`, `atac`, `adt`, `multimodal`, or `perturbation`. Runs of earlier 1.x releases record `basic_rna_analysis`, the RNA pipeline that `pipeline.run` executed. The `config` of a single-modality run records `cells` (the boolean cell column, or the cell selection and, when it came from a run, that run's ID and label), `snapshotColumns`, `species`, and under `stages` the resolved settings of each stage keyword, `false` for a skipped optional stage and `{"enabled": false}` for skipped filtering; an RNA run, and an ATAC run recorded since the ATAC recipe scores doublets, also records `doubletSampleColumns`. A multimodal run's `config` records its `method`, its source runs under `sources`, the shared cell selection under `cells`, `snapshotColumns`, and its `stages`. A perturbation run's `config` records `cells`, `snapshotColumns`, `guideAssay`, its guide assay, `profileLabels`, the output whose cells its profiles average (null without clusters), and under `stages` the settings of each stage keyword, `false` for an optional stage that is off and for `signature` and `profiles` whose stage does not run. Every run's `config` also records `shutdown`, whether the run could install its signal guard.

Reopening a run reads its `config` as recorded and never checks it against the current recipe, so
a run whose settings a later release rejects, such as the t-SNE setting `parallel`, still reopens
and reports. A `basic_rna_analysis` run therefore reopens, reports, plots, and exports with the
`cellKey`, per-option, and `params` configuration and the stage order that it recorded.
A 2.0 release may change records with an entry in the compatibility inventory.

The run record is created before `input_snapshot`; final outputs and field descriptors are written only when the complete recipe succeeds. A handled failure or interruption first commits terminal stage and run details. An interrupted run records an `interruption`, whose `kind` names the cause: `signal`, `shutdown_request`, `keyboard_interrupt`, `asyncio_cancelled` for an escaped `asyncio.CancelledError` or the `concurrent.futures.CancelledError` of a cancelled Zarr call, or `cancelled` for a run stopped through its `CancellationToken`, whose reason becomes the `message`. A run that receives a signal after a cancellation records the kind of the first request, `cancelled`. The stage whose work the interruption cut short records the same `interruption`. A stage whose work had finished is recorded as completed, so a run that stops between stages, or just after a stage's work finished, holds the interruption on its run record only. `abandoned_label_claim` is a run-only kind as well: it marks an unfinished label owner that `pipeline.abandon_label_claim` interrupted, with its `reason` as the `message`. So is `ended_pause`, which marks a paused run that `pipeline.end_paused` ended, with its `reason` as the `message`. A hard process death can leave a run or stage incomplete. Only a paused run resumes; a run in any other state has no on-disk resume, repair, or same-ID retry protocol.

A run that a recipe called with `run_until` commits its pause as a terminal write does, unless another run has committed its requested label, which fails the run instead: the record takes `status` `paused` with `complete` set last, keeps `finishedAtNs`, `label`, `outputs`, `fields`, `error`, and `interruption` empty, and gains `segments`. Each segment is the part of the run that one process ran, with exactly `startedAtNs`, `stoppedAtNs`, `stages`, the stage names whose records the process wrote, in stage order, `reason`, and `shutdown`, which holds the process's `signalGuardAvailable` and `unavailableReason` as `config["shutdown"]` holds them for the process that started the run. The first segment starts at the run's `startedAtNs`. A resume appends a segment whose `stoppedAtNs` and `reason` are null while it runs, and writes the running record, `complete` false, in one metadata write. A segment stops with `reason` `paused`, or with the status that the run ended with: `completed`, `failed`, or `interrupted`. Every segment but the last stopped to pause, and the segments' stages, concatenated, are the first stages of `stageOrder`, all of them for a completed run. A resume claims its pause with an append-only atomic key, `pipeline/runs/.resume-claims/{runId}/{segment}.json`, where `segment` is the number of segments that the paused record holds; the key holds exactly `runId`, `segment`, and `token`, a random value that tells the process whose write created the key. Of the processes that resume one pause, only the one whose token the key holds writes the record. A process that stops between the claim and the record leaves the run paused with its pause claimed, so the run cannot resume, and a new run reuses its finished stages. `pipeline.end_paused` claims a pause with the same key, so that a resume and an end of one pause cannot both go ahead, and records the run as interrupted: `finishedAtNs` and the interruption's `requestedAtNs` hold the time of the end, and the last segment, which keeps its `stoppedAtNs` and `stages`, takes the reason `interrupted`. With `release_claim`, an end also proceeds when a resume or an end holds the claim without having recorded itself, and when the pause's own write is incomplete; such ends write the same interrupted record, so two of them that race leave one valid record. A lost claim is followed by a second read of the run, so that no end writes over a resume that recorded itself after the end read the run. A zero-length `uint8` array named `.resume-claims` marks the namespace, as for label claims, and a storage backend without atomic conditional creation refuses `run_until` and resumes before anything is written.

Requested run labels use append-only atomic claims below
`pipeline/runs/.label-claims/{labelDigest}` while a run is finalized. The completed run record is
the public label owner. Failed, interrupted, and removed predecessors can be bypassed by a later
claim. A live or unclean incomplete predecessor blocks the same label and fails closed. After the
operator has confirmed that its process stopped, the exact owner can be marked interrupted with
`pipeline.abandon_label_claim(label=..., run_id=..., reason=...)`; CyteArc never infers abandonment
from elapsed time. A storage backend without atomic conditional creation rejects a labeled run
before its run record or any computation is started.

Catalog scans skip run children they cannot read, so one crash cannot hide healthy runs, and log a
warning that names each skipped run ID and its error. Opening such a `run_id` exactly raises that
error.

The first stage, `input_snapshot`, stores a cell-selection artifact of the run's input cells, a
`feature_universe` all-feature selection, and full-axis cell and feature metadata snapshots. Its
`cell_snapshot` holds the cell IDs and names and the QC columns that filtering reads, the `attrs`
and a MAD `sample_column`; the `filtering` stage and the run's `ids` and `names` cell fields read
it. Its `feature_snapshot` holds the feature IDs and names of the run's assay, which the stages that
name features and the run's feature fields read. Every other group of cell columns that a run reads
is frozen apart, in a snapshot that the stage writes only when the run uses that group:

- `capture_snapshot`, the capture column of per-capture cell calling, which the `cell_calling`
  stage reads;
- `harmony_snapshot`, the Harmony batch columns, which the `harmony` stage reads;
- `doublet_sample_snapshot`, the columns whose labels split doublet scoring, which the `doublets`
  stage reads;
- `column_snapshot`, the run's `snapshot_columns`, which the run's cell fields of those columns
  read.

A column that one stage reads therefore never enters the provenance of what another stage
computes: a run with Harmony or snapshot columns filters to the same cell-selection artifact as a
run without them. The snapshots are outputs of the stage record, not run outputs. A run given a
cell selection or a completed run as `cells` keeps that selection as its input instead of storing
another one, so runs that share cells share one ref. A multimodal run keeps the selection that its
source runs share, stores a `cell_snapshot` of the cell IDs and names only, beside a
`column_snapshot` when it names `snapshot_columns`, and stores a feature snapshot and a feature
universe for each modality. Frozen `run.features["I"]` is backed by that immutable feature
selection rather than the live feature `I` column. Stored selection integrity compares its Boolean
payload and current ordered row-ID fingerprint, not the later value of the source metadata column.
Frozen run views therefore survive live `I` changes but fail if row identities are replaced or
reordered. Every check computes the current fingerprint from the IDs'
stored bytes: it reads their metadata document and encoded chunk objects, and decodes and hashes
the IDs again only when those bytes differ from the ones that a previous check in the process
fingerprinted (`cytearc.storage.content_stamps`). An in-place chunk write, a rewritten or deleted
chunk, and a rewritten array are therefore detected wherever they were made.

Each completed or failed stage stores exact nested artifact-plan dispositions and its `metrics`: the wall time, the process tree's RSS, sampled every 0.1 s, and what the stage used beside what its process had. Artifact reuse comes from planning receipts, never timestamp inference. Timing, memory, run identity, and reuse state remain in the run ledger and never mutate an artifact. New artifacts store immutable creation time and creator CyteArc version for diagnostics only.

A stage's `cpuSeconds` is the user and system CPU time of the process and its child processes during the stage, and `threadsPeak` is the most OS threads that the process tree held at a sample. `rssHighWaterStartBytes` and `rssHighWaterEndBytes` are the kernel's mark of the process's peak RSS, which `getrusage` reports, read as the stage starts and ends. On Linux a process starts with the mark of the process that started it, so the start can exceed what the process itself has held. When the end exceeds the start, the stage set a new peak and the end is that peak, including a peak inside a call that holds the GIL, where no sample runs. `networkRxBytes` and `networkTxBytes` count the bytes received and sent on every interface but loopback by every process of the network namespace, from `/proc/self/net/dev`: on a container worker that is the container, and elsewhere often the whole machine. `resources` holds the store's `workers` and `memoryBudgetBytes` beside `cpusAvailable`, from `os.process_cpu_count()`, `cpuQuotaCores`, the cgroup CPU quota, and `memoryLimitBytes`, physical memory or a lower cgroup limit, all read as the stage starts, so stages that different processes or machines ran stay comparable.

A stage's `trace` holds at most 64 samples as arrays of one length: `seconds` since the stage started, `cpuSeconds`, `networkRxBytes`, and `networkTxBytes` counted from the start, and the largest `rssBytes` and `threads` sampled since the previous point. When the trace is full, every other point merges into the next, which keeps its own counted values and the larger RSS and thread counts, so each point remains a sample, and the first and last points are the readings that open and close the stage, whose counted values equal the stage's totals. A series that the platform cannot sample is null, and a point whose reading failed holds null. `unavailable` maps each measurement that is null to the reason, such as `threadsPeak` off Linux: a measurement is null exactly where `unavailable` names it, except `cpuQuotaCores`, whose null means no quota.

A completed stage's `summary` is a JSON mapping of diagnostics that CyteArc computes from the
stage's artifacts, such as the cells that filtering kept and removed or the scores of cluster
selection, so a reused stage reports what a computed one would. A stage with several summarized
outputs keys its summary by output, and a summary lists at most 50 samples, hashtags, captures,
or components. Metadata snapshots and ANN indexes have no diagnostics, so the summary of
`input_snapshot` holds only its input cell selection, when the stage made one, and its feature
universe. A stage that did not complete, or whose outputs have no diagnostics, such as `ann_index`,
holds a `summary` of `null`. A completed, failed, or interrupted stage's `warnings` lists, oldest
first, the warnings that CyteArc logged or issued while the stage ran and computed its summary. It
keeps at most the first 50, each at most 512 characters long: a longer warning is cut and ends with
`...`, as a longer error message does. Running and skipped stages hold no warnings.

## Removed assay state

The former `{assay}/state` analysis document is not part of this layout. `DataStore` rejects a
store containing that group before it initializes datastore metadata. CyteArc does not inspect its
contents, migrate it, or use it to recover a current graph. Re-import or rebuild the dataset with
the current release.

`repack_zarr` copies run records and their append-only label claims because it preserves the axes. This applies to the root datastore and nested workspaces. It does not copy resume claims, which a process needs only while it resumes a run, so a run paused in the source resumes in the copy. Subset and merge outputs do not copy source runs. The explicit rewrite omits retired `{assay}/state` groups. Recompute analysis artifacts after the rewrite; the removed document is never translated into current lineage. An overwriting merge clears pipeline records and datastore-scoped artifacts in its destination workspace while preserving unrelated root siblings.

(mounted_targets)=

## Mounted targets

`mount_datastore` creates a target whose root attribute `matrixSource` binds it to the source that owns the counts. The record holds exactly `location` (a path relative to the directory that holds the target for a source in it or below it, and otherwise an absolute path or URI, as for {ref}`composite sources <source_locations>`), `workspace`, and `assays`, and each assay entry holds exactly `datasetFingerprint`, `countsFingerprint`, and `requiresTranspose`. Every open checks the source against these identities. Any other shape, including a field that a later release adds, is rejected rather than ignored, and a source that is itself a mount is refused.

The record holds no backend options. Each open passes the source's options, as `source_storage_options` of `mount_datastore`, `DataStore`, and `repack_store`, or opens the source with the target's `storage_options` when none are given.

The target holds copied cell and feature tables, its own artifacts, and its own pipeline runs.
Opening it reopens the target root through a namespace that resolves keys in this order:

1. Keys outside the artifact roots, `[workspace/]artifacts` and `[workspace/]{assay}/artifacts`
   of the mounted assays, belong to the target alone. Cell and feature tables, counts, and
   `pipeline/` never fall back to the source.
2. A key inside an artifact group `{root}/{kind}/{id}` is read from the target when the target
   holds the group's `zarr.json`, and otherwise from the source when the source holds it.
3. Listings of an artifact root and of its kind directories are the union of both stores, and a
   directory document that the target lacks is read from the source.

Every write goes to the target. Writing or deleting inside a source group raises
`PermissionError`, so the source is never modified. Artifact IDs are random 256-bit tokens, so a
group that the source holds is exactly the artifact a copy in the target would be. The target
lists, loads, traces, and reuses source artifacts with their identities unchanged, and a recipe
whose provenance matches the published results creates nothing. Pipeline runs and their atomic
label claims stay per store: open a source run on the source datastore and pass its refs to the
mount.

Refs, provenance, and run records never record a location, and nothing in the target records
which store holds an artifact. Results that use source artifacts therefore need the source, as
counts do, and a source artifact removed by an outside edit fails closed as a missing input. A
target group opened directly with `zarr.open_group` shows only the target's own artifacts.
`repack_zarr` reads a mount through the same namespace, so its output holds the artifacts of both
stores, the target's runs, and copied counts, and no longer needs the source.

(writer_destinations)=

## Writer destinations

Import writers and `SubsetZarr` create their store with
`cytearc.storage.destinations.create_destination`, and most of them run `check_destination` first,
before they read their source. A destination without keys is created with mode `"w-"`, which
refuses keys that appear after the check. With `overwrite=True` a destination that holds keys is replaced with
mode `"w"` only when its root group holds no prepared assay and no `matrixSource`, and every
top-level member is a group that CyteArc writes: `cellData`, `matrices`, `artifacts`, `pipeline`,
an assay group (`is_assay`), a pending derived assay, or a workspace group that
holds only such members; inside them, every key is Zarr metadata or an array's chunks. Any other
key, such as a `.DS_Store` file or a root array, and keys without a root group are refused with
`FileExistsError`. Every source that a writer reads is prepared, and `SubsetZarr` also refuses a
destination that overlaps a source, so no writer replaces its source. A local destination below a
directory that holds `zarr.json`, `.zgroup`, or `.zarray`, as a path or a local store, and a
destination whose root group carries `is_assay`, raise `ValueError`. Local paths reach Zarr as `pathlib.Path`, because Zarr reads a string as a URL and
would end the path at `#`, `?`, or `;`.

`DataStoreMerge`, `mount_datastore`, and `repack_store` create their destination with mode `"w-"`
and refuse a destination that overlaps a source. They and `SubsetZarr` refuse a source store that
holds a pending derived assay (`cytearc:pending_assay`) in any workspace, through
`refuse_pending_assays`.

(assay_types_attribute)=

## Assay types

The root or workspace attribute `assayTypes` maps each assay name to a preset: `RNA`, `ATAC`,
`ADT`, `HTO`, `CRISPR`, `ANTIGEN`, `CUSTOM`, `GeneActivity`, `GeneScores`, `URNA`, or `Assay`, the
generic type without modality-specific normalization. The preset selects the assay class, and so
the normalization; `HTO` shares the ADT class but stays `HTO`, which `qc.hto_demultiplexing`
requires. Import writers seed the entry from an explicit `assay_type` or from the assay name, and
a writable `DataStore` open records the types it resolved. An explicit type that is not a preset,
an `assay_types` key that names no assay of the store, and a recorded type that is not a preset
raise `ValueError` before anything is written. An assay without an explicit or recorded type takes
the preset of its name, or `Assay` with a warning. A read-only open cannot record a type, so it
raises `ValueError` for an explicit type that differs from the type the store declares; open the
store once with `zarr_mode="r+"` to record it. An `assayTypes` attribute that is present but not a
mapping raises the same remedy instead of being read as empty: a writable open whose
`assay_types` names a preset for every assay replaces it.

Each open assay carries the type that its `DataStore` resolved, `Assay.assayType`, which `cytearc.assay.classification.declared_assay_type` returns. Merge, subset, and `qc.hto_demultiplexing` read that type rather than the attribute, so a workspace store opened with `assay_types={"tags": "HTO"}` declares `HTO` everywhere in the session that recorded it, and a type written to the attribute behind an open `DataStore` takes effect at its next open, or when it adds an assay: `add_grouped_assay`, `add_melded_assay` and `qc.correct_ambient` read `assayTypes` from the store again and write their assay's type beside the attributes that the group then holds, such as another session's `defaultAssay`. Derived assays carry the type they were registered with. Subset, repack, and mount copy the declarations, and `DataStoreMerge` records the one type that every source holding an assay declares, rejecting sources that declare different types. A mount opens each target assay as its `assay_types` entry, or else as the source declares it, and requires the source's `countsT` for exactly the assays that open as RNA.

(normalized_values_attribute)=

## Normalized values

An assay group with the attribute `normalizedValues` holds values that were normalized before import, and one without it holds counts. The record is `{"importedAs": ..., "cellTotal": ...}`: `importedAs` is `normalized`, `log_normalized`, or `log2_normalized` for an import, and null for an assay derived or combined from other values; `cellTotal` is the total that every sampled cell's values share, or null. The stored values are always linear, since imports undo a log before they write. Import writers pass the record to `create_zarr_count_assay`, which validates it, and `Assay.__init__` reads it; a record of another shape raises `ValueError`. Mount copies it with the assay attributes, a full repack copies every attribute and a data-only repack keeps it, `SubsetZarr` passes it on, `DataStoreMerge` and `create_composite` record the import kind that every source shares and no cell total, and grouped and melded assays record a null import. The dataset fingerprint does not cover it, since no operation changes it after import. ADT and ATAC classes refuse an assay that has it.

(library_size_normalization)=

## Library-size normalization

An RNA class normalizes by library size to the size factor in the assay group's `size_factor` attribute, a finite positive number that it reads as stored, so that 2500.5 stays 2500.5; any other value raises `ValueError` when the assay opens. An assay without one takes 1000, which a writable store records. A melded assay records `tfDenominator`, what the term frequencies of its TF-IDF values divide each cell's counts by, and an RNA class that opens such an assay passes its values through with `norm_dummy` instead of dividing them by their own totals again. Mount, both repacks and `SubsetZarr` copy both attributes. Ambient correction records the size factor that its source normalizes with, and `DataStoreMerge` and `create_composite` keep both attributes when their sources share them. The size factor of a source is the session's `sf`, or, when the session sets `sf` to None, the factor that a new session opens the source with: the one it stores, or 1000; a source of a class without a size factor, such as ADT, gives none. Before anything is written, all three refuse a session `sf` that is not a finite positive number, and merges and composites refuse sources of different size factors, or of melded and other values.

## Pending derived assays

`add_grouped_assay` and `add_melded_assay` create the logical group `{assay}` or
`{workspace}/{assay}` with `cytearc:pending_assay`, the name of the operation, and no `is_assay`, so
assay scans skip it. Its counts (below `matrices/{assay}` in a workspace layout), feature
metadata, RNA `countsT`, and membership column `cellData/{assay}_I` ({ref}`assay_membership`)
follow, and one attribute write then replaces `cytearc:pending_assay` with `is_assay=True`.

- Rule A: nothing is deleted once that publication write is issued, even when it fails.
- Rule B: cleanup deletes only the groups that the failed call created.
- Rule C: an ordinary exception deletes them. An interruption (`KeyboardInterrupt`, `asyncio.CancelledError`, the `concurrent.futures.CancelledError` of a cancelled Zarr call, `SystemExit`) keeps the pending assay, because Zarr may still be running the interrupted write, and logs how to remove it.

Every workspace keeps the counts of `{assay}` in `matrices/{assay}`, so a name that is pending in
any workspace is refused in all of them, and `DataStore.discard_interrupted_assay` never removes
counts that another workspace published under that name. It removes the pending assay with its
matrix group, `assayTypes` entry, and membership column; call it only after confirming that no
process is writing it. Repacking refuses a pending assay.

(assay_membership)=

## Assay membership

Every assay of a store spans the store's whole cell axis, so an assay that measured only some cells
still has a `counts` row for every cell. The boolean cell column `<assay>_I` marks the cells that
the assay measured. It has no missing-value mask, and its attributes include
`role="assay_membership"` and `assay="<assay>"`; other attributes are allowed, and a merge does not
carry them into the merged column. A store without the column measured every cell with the assay.
A cell outside an assay holds none of its counts, so its `counts` row is zero and its
`<assay>_nCounts` is 0. Those zeros are no measurement, so analysis fails closed on them.

An operation that reads an assay's values over cells checks the cells against the assay's membership column after its own argument and assay-type checks and before it plans, reuses, or writes a result, on a read-only store too: `features.normalize`, `integration.modalities` (the shared cell selection against every source assay), `features.hvgs`, `features.detected`, `scores.waggr`, `scores.aucell`, `scores.chromvar`, `regulons.grn`, `regulons.profiles`, `markers.search`, `compare.compute_bulk` (the cells whose counts it reads: those of its bulk columns, or every selected cell for a mean that fits the assay's normalization over them, found from the live `I` column before a metadata grouping without `cell_selection` snapshots it), `compare.test` (feature keys, over the cells that the design keeps), `features.prevalent_peaks`, `scores.cell_cycle`, `scores.potency`, `qc.feature_percentage`, `qc.hto_demultiplexing`, `perturbation.import_guide_calls` and `perturbation.call_guides` (the guide assay), `perturbation.efficiency` and `perturbation.associations` (the expression assay), `perturbation.signature` and `perturbation.profiles` (the assay of their features), `qc.doublets`, `trajectory.markers` and `trajectory.aggregation` (the valid cells of the pseudotime), `trajectory.lineage_markers` and `trajectory.lineage_aggregation` (the valid cells of the fate map), `trajectory.drivers` (the cells it correlates), `imputation.compute_imputed` (features), `mapping.run` (the query assay), `to_anndata` with `matrix="normed"` and without a run, the recipes `pipeline.rna`, `pipeline.atac`, `pipeline.adt`, and `pipeline.perturbation`, which check the cells of `cells` before any stage runs and never narrow them, also for the HTO assay that an RNA run's hashing stage reads, for the guide assay of a perturbation run, and for each other assay whose metric its filtering stage names, and the quality-control filters `qc.auto_filter` and `qc.filter` for each metric of an assay that they filter on. A cell column is a metric of the assay whose preparation wrote it: `<assay>_nCounts`, `<assay>_nFeatures`, and the percentage columns that the assay records in its `percentFeatures` attribute (`cytearc.storage.identity.generated_cell_columns`); the column is found among each assay's own columns, never by parsing its name. A `quality_metric` artifact scoped to an assay is a metric of that assay. A cell that the column marks False raises `cytearc.metadata.membership.UnmeasuredCellsError`, a `ValueError` whose `operation`, `assay`, `column`, `unmeasured`, `selected`, `remedy`, and `cell_key` attributes name the refusal, and whose message names the remedy for the operation's input: a cell selection from `select_measured_cells`, passed as `cell_selection=` to `compare.compute_bulk`, `compare.test`, and the QC filters and as `cells=` to a pipeline recipe, labels narrowed with `snapshot_cluster_labels`, a graph built over measured cells, or a boolean column that is True only for the cells of the caller's column that the assay measured, passed as `cell_key` or as a recipe's `cells`, such as one that `ds.cells.insert("RNA_measured", ds.cells.fetch_all("I") & ds.cells.fetch_all("RNA_I"))` writes; the membership column alone also holds cells outside `I`. Operations that read only other results or metadata, such as PCA, neighbors, clustering, layouts, `snapshot_cluster_labels`, `perturbation.labels` (the guide calls and the guide assay's target column), `repertoire.import_receptors`, whose contigs come from files and belong to no assay, and `clones.import_barcodes` of files and `clones.import_alleles`, do not check; the operations at the roots of their lineage do, and `clones.import_barcodes` with `from_assay` checks the assay it reads. Derived assays copy the membership. Raw exports that can declare it keep unmeasured cells: `to_h5ad` and `to_anndata` without layers. An export that cannot declare it refuses them before it writes or reads anything: `to_mtx`, whose zero rows `MtxToZarr` would read back as measured cells, and a `to_anndata` layer of an assay other than the exported one.

The checks read the membership column in bounded blocks. A boolean column or mask is read beside
the column, block by block; a cell-selection artifact is read as its stored mask. Integer rows of
at least one cell in 16 stream the column once into a copy of one byte per cell; fewer rows read
only the chunks that hold them, in their requested order, repeats included.

`select_measured_cells(assay, cell_selection=...)` keeps the cells of a selection, or of the live
`I` column when it is omitted, that the assay measured, as a datastore `cell_selection` artifact
whose operation is `select_measured_cells`, whose parameters are `{"assay": <assay>}`, and whose
inputs record the prior selection and the fingerprint of the membership column. It returns the
prior selection itself when the assay has no membership column or measured every selected cell,
so a fully measured store keeps its identities, and raises `ValueError` when the assay measured
none of the selected cells. Display reads show unmeasured cells as missing: feature values that
plots fetch, those of `get_cell_vals`, and cluster-tree fill values are NaN for them, which plots
draw in the missing-value color or leave out. Only the measured cells are read and normalized, so a
normalizer fitted over the cells that it reads, such as ATAC TF-IDF, whose document frequency would
otherwise count the unmeasured cells, gives a measured cell the value that a read of the measured
cells alone gives; the stream yields the rows of unmeasured cells as NaN pieces of at most one block
of rows. Summary plots compute their statistics over the cells with a value, which `n_cells`
counts, and the plots record the number of plotted cells that each assay did not measure as
`provenance.extras["unmeasured_cells"]`. On a store whose sources measured different features, a
display read also shows a feature as missing in a cell whose source did not measure it
([features each source measured](#measured_features)).

Only imports, `DataStoreMerge`, and derived assays write the column, directly in storage and with
its attributes in its first metadata write, so a reader never finds the column without them; the
copies that `SubsetZarr`, `mount_datastore`, and `repack_zarr` make carry them the same way. A
cell table reserves the name `<assay>_I` of every assay of its store or
workspace, including a pending derived assay, whether or not the column exists:
`MetaData.insert`, with or without `overwrite`, `update_key`, and `reset_key` raise `ValueError`
before they read any value, and `drop` raises for a column that carries the membership `role`,
because a store without the column counts every cell as measured
(`cytearc.storage.identity.protect_metadata_column`). A column of that name without the role is not a membership
column, and `drop` removes it to repair the store. Feature tables and other groups reserve
nothing.

`SeuratToZarr` writes the column for an `Assay5` whose selected count layers hold only some cells.
Membership is the union of the cells of the selected layers, so with `assay_layers` a cell that only
an unselected layer holds is not a member.

`add_grouped_assay` and `add_melded_assay` write a row for every cell, and a cell that the source
assay did not measure holds none of its counts, so the new assay measures the cells of its source.
When the source has a membership column, the new assay's `<assay_label>_I` is a copy of it, written
while the assay is pending, so a failed write and `discard_interrupted_assay` remove it with the
pending assay. A source without the column gives an assay without one. Both methods refuse, before
they compute anything, a new assay whose membership name is already a cell column; drop that column
or choose another name.

`SubsetZarr` keeps the membership column of each assay that it writes, with the rows of the selected
cells, and leaves out the column of every other assay. `mount_datastore` mounts every assay of its
source, so it copies every membership column with the cell table, as `repack_zarr` does.

`DataStoreMerge` writes the column for every merged assay. A merged cell keeps the value of its
source: False when the source lacks the assay (its counts are zero-filled), True when the source has
no membership column, and the source's own value otherwise. A source's membership column is merged
only into that column, never as an ordinary column such as `orig_<assay>_I`, and ordinary merged
columns never carry the `role` or `assay` attributes. The membership column of a source assay that
`assays=` leaves out is not merged at all, so it cannot take the membership name of an assay that a
later merge includes. Planning raises, before anything is written, for a source column with the
reserved name that breaks the contract (another dtype, a missing-value mask, or `role` and `assay`
attributes that do not name the assay), for an ordinary source column whose merged name is a
membership column, and for a source whose cells outside the assay have a nonzero
`<assay>_nCounts`. The merge manifest records each source's membership state per assay: `missing`,
`all`, or the fingerprint of the source column, so a resume requires the same membership.

Export and import keep membership explicit. Without a pipeline run, `to_h5ad` and `to_anndata`
write the membership column of the exported assay as an ordinary `obs` column and declare it in
`uns["cytearc"]["assayMembership"]`, a mapping from the assay to its column. They do not write the
membership columns of other assays, which describe assays that the file does not hold, as a merge or
subset that leaves an assay out does not copy its column. An import reserves the
column name `<assay>_I` of every assay it writes: `H5adToZarr` restores a declared column, which
must be boolean without missing values, with the membership attributes, and skips any other column
of that name with a warning, as `CSVtoZarr`, `CrToZarr`, and `MtxToZarr` always do;
`SeuratToZarr` raises for such a metadata column. A column named for an assay that the import does
not write stays ordinary metadata. A merged store therefore keeps its partial membership through
`to_h5ad`, `H5adToZarr`, and a second merge.

Opening saved results does not retroactively validate assay membership. A pipeline configuration
that selects unmeasured cells raises `UnmeasuredCellsError` before any stage runs. Re-import or
rebuild a store from its source when its membership metadata is incorrect. A derived assay without
a membership column counts every cell as measured; rebuild it under a new label if its source
was partially measured. A column for an assay that is absent from a store is ordinary metadata,
and `drop` can remove it.

(measured_features)=

## Features each source measured

A merge and a composite whose sources measured different features record in their feature table which features each source that holds the assay measured (`merge.features.feature_record`): a Boolean column for each such source, true for the features that the source measured, without a missing-value mask. The column is named `measured_in_<source>` with `_` in place of each `/` and `\` (`storage.metadata_keys.measured_features_column`), and its attributes are `role="measured_features"` and `source`, the source's own name (`measured_features_attributes`). Readers list the table's keys from its store and open only those named `measured_in_`, which they take by their role (`metadata.membership.measured_feature_columns`); a column of that role whose `source` does not name it, that is not Boolean, or that has missing values raises `ValueError` when read, naming the `feats.drop` call that removes it. `measured_features_columns` refuses sources whose columns would be named alike once case is ignored. A source whose record names its cell column of sources, which holds that column with `role="cell_sources"`, is recorded by its own sources (`merge.metadata.nested_sources`): the categories of the column, or the labels of a column of text without its blank and missing ones, each recorded as `<source>__<own source>` (`merge.features.nested_source_name`), which starts the IDs of its cells. In each merged assay, an own source that the source's record of the assay names gets its own column, mapped to the merged features; for an assay without a record, each own source whose cells the assay measured, as one pass over the column and the assay's membership column finds them, gets the source's features, and an own source that never held the assay gets none, as a source without the assay gets none. Each of the source's cells takes the code of its own source in the merged column of sources, and a cell without one, or with a code beyond the column's categories, is refused. Such a merge records each merged source's column and own sources in its manifest under `nestedSources`, so that a destination planned without them is not resumed. A source without a record, or whose record names no such column, counts a feature as measured when every one of its own columns is true for it. The merge does not copy a source's own record columns; a source column of the same name as one that the merge writes is left out with a warning. Subset, repack, data-only repack and mount copy the columns with their attributes, and an H5AD export declares them in `uns["cytearc"]["measuredFeatures"]`, which maps the exported assay to each column and its source, so that an import of an assay of that name restores the attributes. `features_measured_by_every_source` combines the columns, and counts every feature of a table without them, as imports and earlier releases write it, as measured by every source. Beside the record, the merge or composite writes the cell column of each cell's source, `dataset` unless `source_column` names another: categorical codes of the source names, every source a category, with `role="cell_sources"` (`storage.metadata_keys.cell_sources_attributes`), which each record column names in its `sourceColumn` attribute. While a record names it, the column cannot be inserted, overwritten, reset or dropped (`storage.identity.protect_metadata_column`), and a merge of a merged store also copies it as an ordinary column of text. `metadata.membership.cell_sources` reads the codes of the selected cells a block at a time, or the labels of a column of text, as an import with `categorical_columns=False` writes it, and refuses a record whose cell column is missing or lacks that role, or a cell whose source the record lacks, with the `feats.drop` calls that remove the record. A read that needs only which sources some cells hold counts them a block at a time, and display reads charge the cell masks that their groups hold. An H5AD export declares the column in `uns["cytearc"]["cellSources"]`, and an import restores its role and the record's link, refusing a declared record without it. `cells_measured_features` gives the features that the source of every cell of a selection measured, with the record and the sources of the cells, which messages name; it reads the cells, which its caller may pass as a function, only when the record marks a feature as unmeasured by some source. The ADT recipe's panel and markers leave out the others over the run's input cells. `leave_out_unmeasured_features` leaves them out, with one warning that names how many each source lacks, of the feature selections of `markers.search`, `scores.waggr`, `scores.aucell`, `scores.chromvar`, `qc.feature_percentage` and `regulons.grn`, which the results then record as their input; of `features.hvgs`, which scores the others as a store of them alone, each cell's total taken over them, of the candidates of `features.prevalent_peaks` and of `features.detected`, which record the fingerprint of the features that every source of their cells measured as the parameter `measured_features_fingerprint` only when they leave some out; of the phase genes, bins and controls of `scores.cell_cycle`; of each group of `add_grouped_assay`, which records the fingerprint in the assay's attribute `grouped_measured_features_fingerprint`; and of `compare.compute_bulk`'s profiles. `measured_feature_mask` gives the features that every source of some cells measured when some source did not measure some feature, over which cell-cycle scores, WAGGR and the term frequencies of ATAC normalization take each cell's total, as `features.hvgs` does, recording the mask's fingerprint as `measured_features_fingerprint`: `assay.hvg_statistics.cell_totals` takes the prepared total less the cell's counts of the other features, or sums the mask's own when they are fewer. A feature summary of the cell-cycle scores or prevalent peaks covers those features alone and records it too, and LSI doublet scoring reads that summary and divides each simulated doublet by its parents' totals over them, refusing a normalization whose recorded features the record no longer gives. `mapping.run` reads each query cell's source and fills the reference features that the source did not measure as missing features of the cell, refusing a source that measured none of them, and records what each source measured as the projection's `measured_features_fingerprint`. Display reads (`features.values`) group the requested features by the cells that measured each of them and read and normalize each group over those cells, NaN elsewhere, as for cells that the assay did not measure. `require_measured_features` refuses such a feature in `compare.test` and `imputation.compute_imputed`, and `require_features_measured_by_cells` in `features.normalize` and `to_anndata(matrix="normed")`. These readers read the cells' sources only when the record marks a requested feature as unmeasured by some source.

## Count arrays

`counts` is the cell-major assay matrix (`n_cells` × `n_features`).
RNA assays (`RNAassay` and aliases such as `GeneActivity` / `URNA`) also store `countsT`, the same values in gene-major order (`n_features` × `n_cells`).
Import, subset, merge, and `repack_zarr` write both arrays together on Zarr v3.
Non-RNA assays (ATAC, ADT, and similar) write `counts` only.

The two RNA arrays are orientations of one matrix, not independent datastores.
New count arrays are sharded Zarr v3 arrays.
Normalized arrays stay unsharded (`shards=None`).

The matrix group, `counts`, and `countsT` each carry a `cytearc:countMatrixLayout` attribute.
CyteArc checks that those records agree with each other and with the live arrays.
Missing or mismatched layout metadata is an error.
There is no silent rewrite on open and no automatic upgrade of older count layouts.

Writers write every chunk of `counts` and `countsT`, a chunk of zeros too: every array handle that
CyteArc creates for them, such as the one that `create_zarr_count_assay` returns, sets Zarr's
`write_empty_chunks=True`. CyteArc therefore reads them with Zarr's
`read_missing_chunks=False`: a chunk or shard object that a partial copy or an interrupted upload
lost raises Zarr's `ChunkNotFoundError`, a `ValueError` that names the array's store path and the
missing chunk key, instead of reading as zero counts. Restore the object from a complete copy or
rebuild the store; do not set `read_missing_chunks` back to True, which reads the lost counts as
zeros. Zarr checks only the objects that it fetches, the shards of these sharded arrays, so a chunk
missing inside a stored shard still reads as zeros. Artifact arrays keep Zarr's default: their
writers skip chunks that hold only the fill value, so a missing chunk is not lost data there.

Every group is read from its own `zarr.json`. CyteArc never reads consolidated metadata, which a
tool such as `zarr.consolidate_metadata` writes and which writes to the store do not update, so a
consolidated store opens, lists, and plans with its current contents.

Writers compute the raw-count identity while they write `counts`, so no second pass reads the matrix back. `counts` records it as `content_fingerprint`. The sibling `countSummaries` group holds per-cell totals (`rowSums`), per-cell detected-feature counts (`rowPositive`), per-cell nonzero-value counts (`rowNonzero`), and per-feature detected-cell counts (`columnPositive`), and its `source_fingerprint` attribute names the counts it describes. Each writer computes `rowNonzero` in the same pass as the digests, which already count each row's nonzero values, so the fingerprint is unchanged. Preparation reads these summaries instead of streaming the matrix. Summaries that are missing, malformed, or bound to other counts are an error; rebuild the store with `--data-only`. Exports and merges take each cell's number of values from `rowNonzero`, which counts the negative values of signed and float counts too (`storage.identity.row_nonzero_counts`). Stores written by earlier CyteArc versions lack `rowNonzero` and keep working without a rebuild: `rowPositive` counts the positive values of each cell, so it counts every nonzero value whenever the counts hold no negative value, which unsigned counts guarantee, since the count dtype policy below stores a matrix unsigned exactly when its values are whole non-negative counts. Exports and merges of such stores take unsigned counts' values from `rowPositive` and count the values of other counts in a read. Every `repack_store` rewrites the counts through `finalize_counts`, so a plain repack of such a store writes `rowNonzero` and keeps its saved results and dataset fingerprint; `--data-only` writes it too but drops the saved analyses. A merge plans that read within its own `mem_budget` and reads a stored band of the widest run of whole chunk columns that fits at a time, so that it decodes each chunk once.

### Count dtype

`counts` and `countsT` share one dtype, which `cytearc.storage.count_dtype` resolves from the
canonical (duplicate-summed) values of the assay. Counts are stored unsigned, in the narrowest of
`uint8`, `uint16`, `uint32`, and `uint64` that holds the largest value, exactly when every value
is a non-negative integer. Other counts keep their source dtype: `float32` or `float64`, or a
signed integer dtype when values are negative. `float16` is not a count storage dtype. The dtype
depends only on the assay's own values, not on the reader, source encoding, index order,
orientation, memory budget, or the other assays of a source matrix, so the same counts imported
from H5AD, 10x HDF5, Matrix Market, CSV, an in-memory sparse matrix, or a Seurat object give the
same store identity. Imports that split one source matrix into assays (10x HDF5 and Matrix Market
feature types, H5AD `assay_split_key`) resolve the dtype of each assay from its own features.
Every import reads all its counts once before it creates the destination to find their ranges:
readers of 10x HDF5, Matrix Market, and H5AD files report the range of each group of features
over the selected cells (`count_value_ranges`), the CSV reader finds it in its first pass, and
`SparseToZarr` and `SeuratToZarr` scan their sources. The scans sum duplicate coordinates as the
writers do, integers exactly in 64 bits and floats in source order. No import takes a count dtype
argument. Count matrices hold finite values: writers reject NaN and infinity, and every import
rejects them before it creates the destination. Writers cast through a checked cast, so a count
that the stored dtype cannot hold raises instead of wrapping.

Writers that rebuild or combine existing counts follow their sources instead. Subset and repack
keep the source dtype, because they rebuild an existing dataset whose identity and copied
artifacts must stay valid. A merge stores the common type of its source count dtypes, widened so
that features summed by name cannot overflow it; integer sources without a common integer dtype
are rejected rather than stored as floats. Derived assays (grouped and melded) keep their
`float64` values.

### Count layout

The layout policy (`unitBytes` and `chunkBytes`) is recorded in `cytearc:countMatrixLayout`, and readers and `countsT` writers replay it. The imports (H5AD, 10x HDF5, Matrix Market, CSV, sparse, and Seurat) write the default policy unless they are given one, so the layout of an imported store does not depend on the budget of its import. When the policy's counts write or `countsT` transpose does not fit `mem_budget`, an import refuses before it creates its destination with a `CountLayoutMemoryError`, which names the largest halving of the policy, with `unitBytes` and `chunkBytes` halved together, that fits and holds it as `fittingPolicy`; a smaller layout makes every later `countsT` read slower. An import given `policy="fit"` instead fits the default policy to `mem_budget`, as subset does, and writes that halving. Every import writer reports the layout it wrote as its `policy` attribute, and the store records it in `cytearc:countMatrixLayout`. Subset, `repack_zarr`, merge, and `add_grouped_assay` fit the policy to `mem_budget` before they create their destination: when the default policy's counts write or `countsT` transpose does not fit, they halve `unitBytes` and `chunkBytes` together, keeping the chunks per shard, down to count shards of one row. Sparse writers admit the band writes of their sparse sources, and dense writers, such as those of CSV counts and dense H5AD matrices, the dense row bands they write. Writers that choose their source batches (the sparse imports, the Seurat imports, and merge) admit batches of one destination row band, the batch their write starts from, so a layout never leaves the write narrower batches than its bands. A dense H5AD import instead copies batches of about a million values, or whole rows of the matrix's HDF5 chunks, into a band of each assay, so its write holds the bands and no sparse copy of the values. An explicit policy is used exactly or refused before the destination exists, with the largest halving that fits named. `add_melded_assay` sizes its shards to the melding band that fits `mem_budget`. Writers that write their assays one at a time (Seurat, subset, repack, and merge) admit each assay on its own. A resumed merge keeps the layout persisted with its completed counts, so a budget change between attempts cannot block it, and fits the layout of the counts it rewrites. A store written with a smaller policy, chosen or fitted, therefore has smaller shards and more objects. The layout never changes identity: `content_fingerprint`, the counts fingerprint, and the dataset fingerprint are computed from the stored values, so only the layout fingerprint differs between layouts.

Existing stores are never rewritten under a new dtype or layout rule and keep their dtype, layout,
and identity. A re-import of the same source can store a different dtype, and so get different
fingerprints and artifact ids; mounts and mapping references bound to the replaced store fail
closed.

## Preparation

The first writable `DataStore` open of a fresh import prepares its assays: it derives their cell and feature summaries and publishes their identity. One process prepares a store: the preparing open holds a claim, the group `.cytearc-preparation` below the root or workspace group, which it creates with the store's conditional write (`set_if_not_exists`) and removes when the preparation ends, however it ends. Another process that opens the store meanwhile raises `RuntimeError`, which names the process and host that hold the claim, instead of writing the same columns beside it; open it again once that open has finished. A killed preparing process leaves its claim, and the error names the claim to remove once that process has stopped. The claim also excludes a process that opens the store at the same moment only where the conditional write is atomic, on the stores that `storage.stores.has_atomic_conditional_writes` names and that label claims require: Zarr's local, memory and object stores. Any other store, such as an `hf://` store or another fsspec store, looks the claim up and then writes it, so its claim refuses a process that opens the store after the claim is written but not one that opens it at the same moment, and the open that takes the claim warns of that. A store that refuses the conditional write prepares without a claim, with a warning.

The first writable open of a store also removes the low-feature cells of its default assay from
`I`, with `min_features_per_cell`, and records what it applied as the cell table's
`minFeaturesPerCell` attribute: a mapping of exactly the default `assay` and the integer
`threshold`. A later open never filters `I` again, whatever its `min_features_per_cell`, so the
analysis input does not depend on how a store was last opened. A store whose cell table records no
filter, such as one prepared by a release before 1.3.0 or a new mount, subset, or merge, is filtered
by its next writable open, which records the filter. A malformed record raises `ValueError`.
Repacking copies the record with the cell table; a data-only repack does not.

## Opening an RNA assay

`RNAassay` construction requires a complete `countsT` on Zarr v3 plus matching layout metadata.
The open fails if `countsT` is missing, incomplete, unsharded, a Zarr v2 array, or out of agreement with `counts`.

`repack_zarr --data-only` reads the raw store without constructing `RNAassay`.
The user-facing repair is to re-import the source or run:

```bash
uv run python -m cytearc.tools.repack_zarr input.zarr output.zarr --profile fast_local --data-only
```

Without `--data-only`, `repack_zarr` accepts only a prepared source and preserves its results after
verifying that the destination has the same dataset identity.

After a rewrite, recompute HVG, normalization, PCA, graph, and marker artifacts.
Do not resume them from pre-rewrite lineage.

Non-RNA assays in a multi-assay store remain usable when tooling opens without forcing every assay class to construct.

## Zarr versions

New stores default to Zarr v3.
RNA assays require v3 because sharded `countsT` is a v3 feature.
A Zarr v2 RNA store will not open as `RNAassay`.

(zarr_memory_controls)=
## Memory controls

`DataStore(..., mem_budget='8G')` bounds streaming and concurrency (blocks, concurrent work, feature batches).
A Zarr read holds more than its result. A read of a sharded array, such as `counts` or `countsT`, fills a shard-level copy of its selection, holds the compressed bytes of every inner chunk it touches in a shard, and decodes those chunks next to them. Plans charge this through `ArrayGeometry.readBytes`, with each chunk's compressed bytes at the bound that `ArrayGeometry.storedChunkBytes` adds to the decoded chunk, the decoded size plus 1/128 of it and 1 KiB, because a plan cannot know them before it reads and data that does not compress is stored larger than it decodes to: random bits in chunks of 64 KiB and 1 MiB were stored 10 to 33 bytes larger. A `counts` row block of a whole shard reserves twice its bytes, the shard's chunks at that bound, and the one chunk that row-block streams decode at a time, and a `countsT` cell band reserves twice its bytes and each chunk it touches at that bound and decoded. A read of part of a shard, such as one of selected genes, fetches the chunks it takes as byte ranges, which Zarr merges into one fetch when they lie within `sharding_coalesce_max_gap_bytes` of each other, 1 MiB by default in the array's configuration, up to `sharding_coalesce_max_bytes`, 16 MiB, so the read also holds the stored bytes between them. `storage.partition.coalesced_gap_bytes` charges those bytes with the setting of the opened array, `ArrayGeometry.coalesceGapBytes`: in a shard of one row of chunks, such as one of `counts`, each gap between two selected chunks at the bound of the chunks it holds and no more than the setting, and in a shard of several rows of chunks, whose chunks Zarr may store in any order, one such gap between each pair of the chunks taken, up to every chunk not taken. In the default layouts, chunks of 83 to 100 MB lie further apart than the setting, so it matters for small chunks, such as those of a layout fitted to a tight budget. A dense band write runs its producer in each band's task, under the write's I/O concurrency, up to the destination's chunks per band, so the band reads of `copy_zarr_array`, `repack_zarr` and `subset_assay_zarr` run at that concurrency: Zarr reads up to that many of the shards that a band touches at once, as `storage.partition.band_shards` counts them, and decodes up to that many chunks of each at once. Their plans price the read at each concurrency that the write can run at, powers of two and the destination's chunks per band, through `row_read_parts` with its `concurrency` and `shards`, the read taking the place of what the write holds once the read has ended, and keep the plan that runs the most reads and writes at once, and among those the one with the most bands at once. On 8,000 cells by 2,000 genes of random bits in shards of 1,000 cells and ten chunks, the bands of a subset of every fourth cell, which read four shards whole at ten reads at once, held 1.60 times the reserve of a plan that priced one shard and one decode at a time, and hold 0.89 times the reserve of these plans, still at ten reads at once. An unsharded read, such as one of normalized data, holds its result and each chunk it decodes at once beside that chunk's compressed bytes. Plans charge each of those chunks as stored and decoded, through `ArrayGeometry.storedChunkBytes`, as `wholeReadBytes` and `bandReadBytes` charge the same read: the decoded chunk beside the bound of its compressed bytes, the decoded size plus 1/128 of it and 1 KiB. In the `fast_local` layout, a 134 MB chunk of log-normalized values over 20,000 genes is stored in 59 MB, and reading its rows held 327 MB, within the 404 MB that the plan reserves. A thread of a plain thread pool keeps the last call it ran until it next runs: the chunk that the call decoded, its future, and for a sharded array the buffer that the read fetched from the shard, which each inner chunk is decoded from. On a busy machine that can be long after the read has copied the chunk: on one core shared with three busy processes, 10 of 25 reads of a shard of four 4 MiB chunks, decoded one at a time, held the previous chunk while the next one decoded, and 1 of 50 held two. CyteArc's codec pools, the pools that `storage.async_execution` installs on Zarr's event loop and the codec pool of each `AsyncStorageRunner`, therefore hand each result back to the event loop that awaits it: a call leaves its result in a parcel and returns nothing, and the loop moves the result into the future that the read awaits, so no pool thread holds a chunk once the read has it, however late the thread runs again. A read on CyteArc's pools therefore holds no previous decoded chunk, only the objects of Zarr's read beside its buffers: 31 to 45 KB for a read of one chunk, 2.2 to 2.4 KB for each further chunk it takes, as Zarr makes a task for every chunk before it reads any, and 12.5 KB for each further chunk it decodes at once, traced on Zarr 3.4 after a full collection has emptied CPython's free lists. `ArrayGeometry.heldDecodeBytes` charges every read those objects, as `ArrayGeometry.objectBytes` counts them, at 32 KiB a read, 4 KiB a chunk and 16 KiB a decode at once, and `tests/test_stream_reservations.py` checks that reads of tiny chunks trace between half of that charge and the whole of it. A read that takes its rows a band at a time, as whole reads of a sharded array and reads of selected rows do, makes a Zarr read of each band. CPython keeps the objects of a few kinds that such reads free on its free lists for reuse, which only a collection of the oldest generation empties ([gc.c](https://github.com/python/cpython/blob/v3.13.18/Python/gc.c#L1464-L1468)), and on Python 3.13 the reads fill its lists of tuples: its `tuple()` of an iterator without a length, as Zarr builds the coordinates and selections of the chunks it takes, allocates ten items and shrinks the tuple to those it got ([abstract.c](https://github.com/python/cpython/blob/v3.13.18/Objects/abstract.c#L2039-L2120)), so each such tuple, once freed, joins the free list of its final length without having come from it, until that list holds 2,000 tuples; each length from 1 to 20 items has a list ([pycore_freelist.h](https://github.com/python/cpython/blob/v3.13.18/Include/internal/pycore_freelist.h#L16-L18)). No reference cycle holds these objects. `ArrayGeometry.bandsLeftoverBytes` charges the earlier bands' reads beside the last, 32 KiB and 4 KiB for each of their chunks, up to what the lists and the objects that the reads keep come to. Reads of a one-dimensional array fill the list of tuples of 1 item, and what 100 to 4,000 band reads of flags left levelled off at 140 to 154 KB, within their limit of 192 KiB. Reads of an array of more axes also fill the lists of tuples of one item an axis and of one item a chunk that a band takes along an axis of a shard, so what 1,000 reads of a row of a two-dimensional array left grows with its chunks up to 20, the longest tuple a list keeps, and drops beyond: 261 KiB at 1 chunk a band, 603 KiB at 16, 665 KiB at 20, and 276 to 284 KiB from 21 chunks on. Their limit, 743 KiB, holds full lists of tuples of 1, 3 and 20 items, 624,000 bytes, which covers arrays of up to three axes, beside the other free lists that the reads use, 38,560 bytes when full, and 96 KiB for what a whole read keeps beside them once it returns, objects that a full collection does not free and those of its last band's read: up to 82 KB after a whole read of 1,000 bands of 20 chunks. Such a read left 715 KB beside its result for rows of 20 chunks, and 734 KB for 1 by 20 chunks of an array of three axes. Python 3.14 builds a tuple from an iterator at its final length ([python/cpython#127758](https://github.com/python/cpython/pull/127758)), so there the same reads leave 30 to 57 KB, under the same limits. A whole read of 100,000 rows of float32 in 1,000 bands of two to five chunks held 1.007 to 1.056 times its charge at 192 KiB on Python 3.13, and holds 0.71 to 0.99 of it now. A read of selected rows keeps a block of each band's rows, 400 to 420 bytes, which `ROW_BLOCK_BYTES` counts. `ArrayGeometry.readScratchBytes` of an array of Zarr's local store counts the buffer of the file that each fetch reads through: Zarr reads a whole file, a chunk or a whole shard, with `Path.read_bytes`, and part of a shard with `open`, which before Python 3.14 both buffer the file system's block size, or `io.DEFAULT_BUFFER_SIZE`, 8 KiB, where the platform reports none, as Windows does not; from Python 3.14 a whole file is read unbuffered and `open` buffers the block size, at least `io.DEFAULT_BUFFER_SIZE`, 128 KiB, and at most 8 MiB, so an array whose shards hold one chunk, which every read takes whole, holds none. A whole-shard read in the default count layout now reserves about 100 MB less, 3% of its charge, and a whole read of normalized values at a concurrency of 4 about 537 MB less, while a read of chunks smaller than its objects reserves up to those objects more. Only Zarr's opt-in `FusedCodecPipeline`, with `codec_pipeline.max_workers` above 1, decodes on a pool of its own, whose threads keep the chunk they decoded: `array_geometry` records that an array was opened with it as `ArrayGeometry.poolKeepsDecodedChunk`, and the plans of its reads also charge the previous chunk for each decode under way. A user who configures that pipeline gets it in place of CyteArc's `StoreThreadsCodecPipeline`, which `ensure_zarr_host_ceiling` installs only in place of Zarr's default, so neither a store's registered threads, `zarr_decoding_threads` nor Zarr's async concurrency bound its reads: on a store that serves Zarr's synchronous I/O, such as a local or memory store, it reads as many chunks, or shards, at once as its pool has threads, `codec_pipeline.max_workers` or the more that an earlier read grew the pool to, each shard decoding its chunks one at a time, and on any other store it decodes every chunk of an unsharded read before it places any, and reads the shards of a sharded one at Zarr's async concurrency, each decoding as many of its chunks at once as the pool has threads. `array_geometry` records the pool's threads as `ArrayGeometry.fusedPoolWorkers` and the second kind of store as `fusedReadsAsync`, and `ArrayGeometry.readsAtOnce` charges each read at those reads at once in place of the plan's concurrency, as traced reads of sharded and unsharded arrays check in `tests/test_stream_reservations.py`, on local stores at one and four workers and on asynchronous stores at four workers and at eight with two reads at once.
The reads that the memory ledger of `utils.memory_budget` charges, `read_rows`, `read_text_rows` and `read_whole` of `metadata.rows` with the repertoire operations' other reads, decode on a pool of exactly the store's `nthreads` threads, whatever the machine. `storage.async_execution` installs that pool as the default executor of Zarr's event loop, where Zarr runs each fetch and each compressor's decode of a read made through its synchronous API, in the context of the call that made the read: a read within `zarr_decoding_threads(n)` runs them on a pool of `n` threads, which the reads of every call of that count share, from any thread, and any other read runs them on the host's threads, as many as Zarr's pool had before. They read a sharded array a band of inner chunks at a time, so that no fetch takes more than one band of a shard, and a whole unsharded array in one call at the highest concurrency, up to the operation's, whose chunks fit the ledger's room. `ArrayGeometry.wholeReadBytes` and `bandReadBytes` state what these reads hold; a read of selected rows of an unsharded array also holds the copy that Zarr takes of the rows each decoded chunk holds before it places them, which `metadata.rows.row_read_bytes` adds. The memory page lists every charge ({ref}`what a refusal names <memory_refusals>`). Every read and write of a store that a `DataStore` opened decodes and encodes on at most the store's `nthreads`, whoever makes it: the store's open, methods and operations, its plots, exports and recipes, its cell and feature tables, a completed run's views, and the lazy arrays and Zarr groups that it hands out. The store registers itself, each store it wraps and a mount's source with `register_store_threads` of `storage.async_execution` as it opens, and CyteArc's codec pipeline, `StoreThreadsCodecPipeline`, which `ensure_zarr_host_ceiling` makes Zarr's while Zarr's default `BatchedCodecPipeline` is configured, runs each read and write within the threads registered for its chunks' store and the stores that store wraps. A read takes the fewest threads of the registrations that match: those of its store object and those of every registered store of the same location. `_location_key` names a location by what the read-only copy that `zarr.open_group(store, mode="r")` makes through `with_read_only` shares with its store, never by the stored contents: a local store's root, a memory store's dict and its path, an fsspec store's protocol and path, `HfReadStore`'s among them, and an obstore store's type with its URL, or with its prefix and the endpoint, bucket, account and container of its configuration, or else the obstore store it wraps. So a read-only copy, such as the group that `artifacts.load` returns, decodes on the store's threads, and so does a caller's own open of the location through the same kind of store, such as `zarr.open_group` of a folder or of an `hf://` URL. CyteArc opens `s3://`, `gs://`, `az://` and `https://` locations with obstore, while `zarr.open_group` of such a URL makes an fsspec store, which is another kind. Two DataStores of one location, or of one store object, each keep within their own bound, and a registration ends once its store is collected. Other stores match only themselves. Threads that the caller sets win over the store's: a pipeline stage runs within `zarr_decoding_threads` of the store's workers, a memory ledger's reads within its `codec_threads`, `SubsetZarr`, `subset_assay_zarr`, `DataStoreMerge.dump`, `create_composite`, `repack_store` and a mount's copy within their own `nthreads`, and a composite's reads of its sources within the composite's threads, through `run_array_read`. An import registers the store it creates with its own `nthreads`. Block streams and concurrent storage tasks run each call in a copy of their caller's context, so a call's workers decode where the call does. A codec pipeline that the user configured in place of Zarr's default stays, as does the pipeline of an array opened before CyteArc first decoded, and their reads outside such contexts decode on the host's threads. `zarr.config.reset()` restores Zarr's default pipeline too, so the arrays opened after it take Zarr's until CyteArc next opens a store, reads on an event loop of its own or registers a store, which installs CyteArc's again. A Hugging Face store's requests run within `zarr_host_threads`, on the host's threads, since they wait on the network rather than compute.
The text of a cell-data artifact is converted and written one stored chunk at a time, so `cell_data_write_bytes` in `metadata/artifacts.py` counts a text array's chunk and its encoded copy, and a numeric array's chunks that Zarr encodes at once, up to its async concurrency.
A read whose rows or columns are an index array, such as a read of selected cells, holds more than a read of slices. Zarr's orthogonal indexer orders the positions of an index array by chunk in arrays of int64, 40 bytes a position while it builds them, and a sharded read builds such an indexer again inside each shard, 64 bytes a position in all; Zarr 3.4.2 traced 40 and 64 bytes for shuffled positions and less for sorted ones. Zarr also copies the values that the read takes from each chunk it decodes before it places them, where a read of slices places a view of the chunk, and NumPy indexes a chunk by two index arrays at once through buffers of 8,192 elements of each array. `ArrayGeometry.readBytes` charges the indexers of its `indexes` positions and the copy, `takenBytes`, as `ArrayGeometry.takenChunkBytes` counts it, once a read: CyteArc's codec pipelines hand each decoded chunk back to the read's event loop, which copies and places its values before it takes the next, while Zarr's opt-in `FusedCodecPipeline` copies on its pool's threads and is charged a copy for each chunk it decodes at once. Row-block streams, `row_read_parts` and `indexed_read_parts` pass both for the reads that take index arrays, and charge the copy of every row of a block at the most columns that it takes from one chunk, which the rows of a block of one chunk's rows reach. A merge's held cells are charged through the same row-block charge. A block of cells spread over several shards is one Zarr read, which makes a task for every shard it touches before it reads any, as it does for every chunk of an unsharded read, so `readBytes` charges the shards that a read does not read at once the objects of a chunk each, 4 KiB. Row-block streams pass the rows of shards that a block touches, which `storage.partition.band_shards` counts over the stream's blocks, each shard once in whatever order the cells come, and an unsharded read the chunks of the block's rows of chunks and of its columns, each a read of its own; Zarr's fused pipeline reads as many of those chunks, or shards, at once as its pool has threads, as `readsAtOnce` counts them. A stream that shrinks its blocks to fit its budget charges each size of block the shards that blocks of that size touch. The reads of selected cells that RNA feature means, doublet parents, reference mapping and subsets plan through `row_read_parts` pass the rows of shards of their own blocks the same way. `band_shards` and `is_contiguous`, which finds whether an index array is a run that a read takes as a slice, take a selection's indices at most 65,536 at a time, or one band where `band_shards` takes whole bands that hold more, so they hold 9 to 11 bytes for each index of such a piece, 590 to 720 KB for 65,536 indices, not for each index of the selection; `band_shards` sorts a piece only where its indices are out of order.
Index arrays on both axes of a sharded array never reach Zarr together: Zarr would turn them into coordinates of every selected value inside each shard, which hold about thirty times the result. `read_selection` instead reads one stored chunk of one axis at a time, as a band that slices the axis over the indices the chunk holds while Zarr indexes the other axis. It slices the axis whose bands hold less, their values beside Zarr's indexers and copies of the other axis's indices and its own ordering of the sliced ones, as `storage.partition.index_band_bytes` counts them at Zarr's async concurrency. A band of columns lies in one column shard and touches the rows of shards that the rows touch, and a band of rows lies in one row of shards and touches the column shards of the columns, so neither waits on the tasks of shards that it does not read. The reads of the earlier bands leave their objects while the last one runs, which `ArrayGeometry.chunksLeftoverBytes` counts for the chunks they took, up to the limit of whole reads and reads of selected rows of such arrays, 743 KiB, as the reads in bands above. The reader also keeps where each band starts and stops among the ordered indices. Plans charge the result beside one Zarr read of such a band and what the earlier bands leave through `indexed_read_parts`, with the same counts at a chunk's rows, so the bands that a read picks hold no more than its plan. A block whose rows form one run is read in one Zarr request instead, over no more rows of shards than a run of a block's rows touches, as `storage.partition.run_shards` counts them.
Environment variables used by the documentation executor (`CYTEARC_MEM_BUDGET`, `CYTEARC_WORKERS`, …) are also useful for local large runs.

## Related guides

- [](../tutorials/data_organization.ipynb)
- [](../concepts/memory_and_execution.md)
- [](contributing.md)
- [](operation_revisions.md)
