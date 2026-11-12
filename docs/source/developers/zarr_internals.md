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

A metadata column stores text as fixed-width unicode at the width of its widest value, four bytes a character, and a read of any of its rows decodes the whole stored chunk that holds them. Every writer of a cell table, `cellData` or a workspace's, and of an assay's `featureData`, therefore chunks a new column by `storage.layout.column_chunk_rows`: a chunk holds at most 100,000 rows, no more rows than the table has, and no more than 16 MiB of values, and always one row at least. A store of 48 cells keeps each column in one chunk of 48 rows, a text column wider than 41 characters takes fewer than 100,000 rows to a chunk, 20,971 at 200 characters, and a value wider than 16 MiB is a chunk of its own. A missing-value mask takes the chunks of its column. The imports, `cells.insert` and `feats.insert`, `update_key` and `reset_key`, the membership columns of imports and derived assays, subset, merge, repack, mount, and composites write cell and feature columns this way, so the 64 guide IDs of a screen are one chunk of 64 rows; the Seurat import caps its metadata and membership columns at 65,536 rows, the rows of one of its blocks. A column replaced through `MetaData` is chunked by the rule in either table. Stores written before keep their chunks. They are read as stored, and every read is costed by the stored chunks of the array it reads, so a read of 48 rows of a column stored in 100,000-row chunks is charged that chunk. A column written into such a store, or written again, takes the new chunks.

(categorical_columns)=

## Categorical columns

A label column can store integer codes and its categories instead of a text value per row, as AnnData stores a categorical. A categorical column `<name>` stores one code per row, the position of the row's category or -1 for a missing row, in the smallest signed integer type that holds the position of every category: int8 up to 128 categories, int16 up to 32,768, then int32 and int64. Its categories are the one-dimensional array `__cytearc_categories__<name>` beside it, every category in its order, unused ones included: text as fixed-width unicode at the width of the widest category, or int64, float64, complex128 or bool, as `MetaData.insert` types values. The codes name the categories array in their `categories` attribute, as a nullable column names its mask, and the categories array records in its `ordered` attribute whether the categories are ordered. Codes that hold -1 also link the missing mask `__cytearc_missing__<name>`, which flags exactly those rows, so every reader of missing values works on either layout. The codes and their mask are chunked by `column_chunk_rows` from the codes' itemsize, so a column of 600,000 cells with a few hundred categories is six chunks of 100,000 int16 codes, and the categories by the same rule over their own rows. A release before categorical columns opens a table only when each of its arrays holds one value per row, so the categories array never holds as many entries as its table has rows: when the categories number exactly the rows, it holds one reserved entry after the last category, the placeholder of their dtype, which no code names, and records it with `padded: true`. Such a release therefore refuses every table that holds a categorical column, with `Not all columns are of same length`, rather than read its codes as values, and this release leaves the entry out of every read, so labels, `fetch_categorical`, fingerprints and identities do not depend on it. `cells.insert` and `feats.insert` write this layout for a pandas `Categorical` or a Series of categorical dtype, and the text layout for other values; `I`, `ids` and `names` always store their values.

`H5adToZarr(categorical_columns=True)` writes this layout for each categorical of `obs` and `var`, in AnnData's current encoding and in the legacy ones that keep its categories in a `__categories` group beside the columns or in `uns/<column>_categories`. A code below 0 or past the last category becomes -1. The categories keep their source order, unused ones included, and take the type that the labels of a default import take, which the categories that rows hold decide: text, or int64, float64, complex128 or bool. Every category takes it, so a column whose rows hold no category stores text categories, as a default import stores such a column as text. Each assay of a split import types the categories from the codes of its own features. The codes carry the `levels`, `ordered` and `display` attributes that the column of labels carries, so plots, merges and exports read the same attributes. An import with the option refuses a categorical whose categories this layout cannot store, such as categories that read alike once stored, NaN, or numbers beside text. Imports write label columns by default in this release, because a release before categorical columns refuses to open a table that holds a categorical column.

Every read returns a categorical column's labels: `MetaData` hands it out as a `CategoricalColumn` (`cytearc.storage.categorical`), which has the shape, chunks and attributes of the codes and the dtype of the categories, selects codes as Zarr selects them and returns the categories at them, with the placeholder of their dtype in missing rows, empty text, 0, NaN or False, as a text column holds there. `fetch`, `fetch_all`, `to_pandas_dataframe`, `head`, `get_index_by`, row blocks, groupings, samples, `artifacts.load_values` of a `CellField` and plots therefore see the same labels from either layout, and `MetaData.fetch_categorical` returns the codes and categories as a pandas `Categorical`. The reads that a memory ledger charges read a categorical column's codes and charge them, its categories and the labels they return, and the checks that plan a read of selected cells count the same: `read_text_rows` returns one Python string per category, which the rows share, rather than a string per row. Groupings work on the codes and show the categories: `compare.compute_bulk` compares each group's code rather than its label, clone summaries that group cells by a column (`cell_groups`) code them through their categories, and so do the samples of guide calling, perturbation signatures, Mixscape, perturbation profiles and associations, and hashtag demultiplexing, which read a pandas `Categorical` of the cells' codes and decode no label; their groups, names and fingerprints are those of the labels. Plots decode the codes of the plotted cells alone, and when a plotted cell's label is missing, into one Python string per category. Exports read the codes and categories through `fetch_categorical`: `to_anndata` and `to_h5ad` write a categorical column as an AnnData categorical of every category in its stored order, the reserved entry left out, so `H5adToZarr(categorical_columns=True)` imports the file's column as the same codes and categories, while a text column of the same labels exports its labels.

Fingerprints built from labels do not depend on the layout. `metadata_column_fingerprint`, the column fingerprints of run snapshots and the snapshots themselves decode a categorical column at the dtype that a column storing its labels has, text as wide as the widest label a row holds, so the same labels give the same fingerprint and the same artifact identity in either layout; sample assignments, groupings and `snapshot_cluster_labels` fingerprint the labels they read. `metadata_column_stamp`, which `clones.from_column` records without decoding the column, stamps the stored codes and categories, so it differs between the layouts, as it differs between two text columns stored in other chunks.

Subset, repack and mount copy a categorical column as codes and categories: the codes of the rows they copy and every category, with a reserved entry when the categories number exactly the copied rows. A merge writes the labels of a categorical column as a column of values. A link that names another array, codes that are not signed integers or too narrow for their categories, a categories array of another kind or without `ordered`, a `padded` attribute that is not true or whose array does not hold one entry more than the column has rows, and a code outside -1 and the last category raise `ValueError` when the column is read, and a table whose `I`, `ids` or `names` is categorical raises `ValueError` when it opens. Stores written before hold no categorical column and read, fingerprint and copy as before.

## Composite sources and virtual counts

A composite has ordinary cell and feature metadata, preparation summaries, and analysis artifacts. Its root `composite` attribute records `assay`, `sources`, `arrays`, `maps`, and `complete`. Each source binding holds its `name`, `location`, `workspace`, `datasetFingerprint`, and `countsFingerprint`. Source credentials are supplied at reopen through `DataStore(source_storage_options={source_name: options})` and are never persisted.

`create_composite` writes the composite to any destination that the store layer opens: a local path, a URL with the `storage_options` that open it, or a Zarr store. New arrays take the storage profile of the destination, as other writers' do, and a destination that holds anything raises `FileExistsError`. The cell column of source names is `dataset` unless `source_column` names another, checked as `DataStoreMerge` checks its own.

(source_locations)=

A source's `location` is its absolute path or URL. A composite created with `relative_sources=True` records a source in the directory that holds the composite, or below it, by a path relative to that directory instead, with `/` separators, such as `a.zarr` or `sources/a.zarr`, when the source and the composite are both local or both at URLs of one scheme and host. Opening resolves a relative location against the directory of the path or URL that the composite is opened from, so a directory that holds a composite and its sources can be moved, copied, or served from a bucket, and the identity checks still refuse a source with other content at that place. A composite opened from a store object that has no path or URL, such as an object store, cannot resolve a relative location and raises `ValueError`. A mount's `location` follows the same rule, with `mount_datastore(relative_sources=True)`. Relative locations are opt-in because a release before them resolves a relative location against its working directory, so it finds the source only when run from the directory that holds the store; a store written without the option opens in every release as before.

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

A producer creates an artifact group with `complete=False` at a path that ends in its random
256-bit `artifact_id`, so the path belongs to that producer alone. After it validates the payload,
one metadata write sets `complete=True` and publishes the artifact. The producer deletes the group
that it created after an exception only before that write; once the write is issued, the group
stays, complete or not, because the write can persist even when it reports an error. An
interruption (`KeyboardInterrupt`, `asyncio.CancelledError`, `SystemExit`, or a shutdown request)
keeps the group with `complete=False`, as derived assays keep a pending assay (Rule C below),
because Zarr may still be running the interrupted write. `DataStore.artifacts.prune_incomplete`
removes the incomplete groups and the artifact-ID directories that hold chunks but no `zarr.json`
once no process writes to the store.

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

A table whose labels can all name child groups keeps the layout of earlier releases byte for byte, so they read it as before. A table stored by code needs a release that reads that layout: an earlier release finds no `skipped_groups` in it and raises `ValueError` rather than showing codes as labels. No earlier table is stored by code: earlier releases refused labels with `/` or `\` and the labels `.` and `..` before they searched, and failed on `zarr.json`, a NUL character and the slot's array names when they wrote the table. They stored a group labelled `.zattrs`, `.zgroup`, `.zarray` or `.zmetadata` under that name in a Zarr v3 store, where it is no metadata document; this release stores such a label by code and still reads and reuses those tables.

## Pipeline records and snapshots

Pipeline run, stage, and label-claim documents use one strict unversioned shape. A run has exactly
`runId`, `recipe`, `requestedLabel`, `label`, `assay`, `startedAtNs`, `finishedAtNs`, `status`,
`complete`, `cytearcVersion`, `config`, `stageOrder`, `outputs`, `fields`, `error`, and
`interruption`. A stage has exactly `stage`, `ordinal`, `startedAtNs`, `finishedAtNs`, `status`,
`complete`, `outputs`, `plans`, `metrics`, `error`, and `interruption`, and may also hold the
optional `summary` and `warnings`. A label claim has exactly `label` and `runId`. Unknown or
missing fields fail closed.

Within 1.x a release may only add fields, and a release that adds one must still read records written without it. No release removes, renames, retypes, or reinterprets a field, so a run of any 1.x release reads in every later 1.x release. The stage fields `summary` and `warnings` were added this way: CyteArc writes both on every stage record it creates, and reads a stage record of an earlier release, which lacks them, as one without a summary or warnings. The stage metrics `cpuSeconds`, `threadsPeak`, `rssHighWaterStartBytes`, `rssHighWaterEndBytes`, `networkRxBytes`, `networkTxBytes`, `resources`, `trace`, and `unavailable` were added the same way: CyteArc writes all nine on every stage record, and reads metrics that hold none of them, as earlier releases wrote them, with each as null, while metrics that hold only some of them fail closed. A run's `config` and a field descriptor's `display` are JSON mappings that readers do not check key by key: a later release may add keys to them, and code that reads such a key treats its absence as the behavior of the release that wrote the run.

A run's `recipe` names the recipe that wrote it: `rna`, `atac`, `adt`, `multimodal`, or `perturbation`. Runs of earlier 1.x releases record `basic_rna_analysis`, the RNA pipeline that `pipeline.run` executed. The `config` of a single-modality run records `cells` (the boolean cell column, or the cell selection and, when it came from a run, that run's ID and label), `snapshotColumns`, `species`, and under `stages` the resolved settings of each stage keyword, `false` for a skipped optional stage and `{"enabled": false}` for skipped filtering; an RNA run also records `doubletSampleColumns`. A multimodal run's `config` records its `method`, its source runs under `sources`, the shared cell selection under `cells`, `snapshotColumns`, and its `stages`. A perturbation run's `config` records `cells`, `snapshotColumns`, `guideAssay`, its guide assay, `profileLabels`, the output whose cells its profiles average (null without clusters), and under `stages` the settings of each stage keyword, `false` for an optional stage that is off and for `signature` and `profiles` whose stage does not run. Every run's `config` also records `shutdown`, whether the run could install its signal guard.

Reopening a run reads its `config` as recorded and never checks it against the current recipe, so
a run whose settings a later release rejects, such as the t-SNE setting `parallel`, still reopens
and reports. A `basic_rna_analysis` run therefore reopens, reports, plots, and exports with the
`cellKey`, per-option, and `params` configuration and the stage order that it recorded.
A 2.0 release may change records with an entry in the compatibility inventory.

The run record is created before `input_snapshot`; final outputs and field descriptors are written
only when the complete recipe succeeds. A handled failure or interruption first commits terminal
stage and run details. An interrupted run records an `interruption`, whose `kind` names the cause:
`signal`, `shutdown_request`, `keyboard_interrupt`, `asyncio_cancelled`, or `cancelled` for a run
stopped through its `CancellationToken`, whose reason becomes the `message`. A run that receives a
signal after a cancellation records the kind of the first request, `cancelled`. The stage whose
work the interruption cut short records the same `interruption`. A stage whose work had finished
is recorded as completed, so a run that stops between stages, or just after a stage's work
finished, holds the interruption on its run record only. `abandoned_label_claim` is a run-only
kind as well: it marks an unfinished label owner that `pipeline.abandon_label_claim` interrupted,
with its `reason` as the `message`. A hard process death can leave a run or stage incomplete.
There is no on-disk resume, repair, or same-ID retry protocol.

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

`repack_zarr` copies run records and their append-only label claims because it preserves the axes.
This applies to the root datastore and nested workspaces. Subset and merge outputs do not copy
source runs.
The explicit rewrite omits retired `{assay}/state` groups. Recompute analysis artifacts after the
rewrite; the removed document is never translated into current lineage.
An overwriting merge clears pipeline records and datastore-scoped artifacts in its destination
workspace while preserving unrelated root siblings.

(mounted_targets)=

## Mounted targets

`mount_datastore` creates a target whose root attribute `matrixSource` binds it to the source that owns the counts. The record holds exactly `location` (an absolute path or URI, or with `relative_sources=True` a path relative to the directory that holds the target for a source in it or below it, as for {ref}`composite sources <source_locations>`), `workspace`, and `assays`, and each assay entry holds exactly `datasetFingerprint`, `countsFingerprint`, and `requiresTranspose`. Every open checks the source against these identities. Any other shape, including a field that a later release adds, is rejected rather than ignored, and a source that is itself a mount is refused.

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

Each open assay carries the type that its `DataStore` resolved, `Assay.assayType`, which
`cytearc.assay.classification.declared_assay_type` returns. Merge, subset, and
`qc.hto_demultiplexing` read that type rather than the attribute, so a workspace store opened
with `assay_types={"tags": "HTO"}` declares `HTO` everywhere in the session that recorded it, and
a type written to the attribute behind an open `DataStore` takes effect at its next open. Derived
assays carry the type they were registered with. Subset, repack, and mount copy the declarations,
and `DataStoreMerge` records the one type that every source holding an assay declares, rejecting
sources that declare different types. A mount opens each target assay as its `assay_types` entry,
or else as the source declares it, and requires the source's `countsT` for exactly the assays that
open as RNA.

(normalized_values_attribute)=

## Normalized values

An assay group with the attribute `normalizedValues` holds values that were normalized before import, and one without it holds counts. The record is `{"importedAs": ..., "cellTotal": ...}`: `importedAs` is `normalized`, `log_normalized`, or `log2_normalized` for an import, and null for an assay derived or combined from other values; `cellTotal` is the total that every sampled cell's values share, or null. The stored values are always linear, since imports undo a log before they write. Import writers pass the record to `create_zarr_count_assay`, which validates it, and `Assay.__init__` reads it; a record of another shape raises `ValueError`. Mount copies it with the assay attributes, a full repack copies every attribute and a data-only repack keeps it, `SubsetZarr` passes it on, `DataStoreMerge` and `create_composite` record the import kind that every source shares and no cell total, and grouped and melded assays record a null import. The dataset fingerprint does not cover it, since no operation changes it after import. ADT and ATAC classes refuse an assay that has it.

## Library-size normalization

An RNA class normalizes by library size to the size factor in the assay group's `size_factor` attribute, a finite positive number that it reads as stored, so that 2500.5 stays 2500.5; any other value raises `ValueError` when the assay opens. An assay without one takes 1000, which a writable store records. A melded assay records `tfDenominator`, what the term frequencies of its TF-IDF values divide each cell's counts by, and an RNA class that opens such an assay passes its values through with `norm_dummy` instead of dividing them by their own totals again. Mount, both repacks and `SubsetZarr` copy both attributes, and ambient correction keeps its source's size factor. `DataStoreMerge` and `create_composite` keep both when their sources share them, and refuse sources of different size factors, or of melded and other values, before anything is written.

## Pending derived assays

`add_grouped_assay` and `add_melded_assay` create the logical group `{assay}` or
`{workspace}/{assay}` with `cytearc:pending_assay`, the name of the operation, and no `is_assay`, so
assay scans skip it. Its counts (below `matrices/{assay}` in a workspace layout), feature
metadata, RNA `countsT`, and membership column `cellData/{assay}_I` ({ref}`assay_membership`)
follow, and one attribute write then replaces `cytearc:pending_assay` with `is_assay=True`.

- Rule A: nothing is deleted once that publication write is issued, even when it fails.
- Rule B: cleanup deletes only the groups that the failed call created.
- Rule C: an ordinary exception deletes them. An interruption (`KeyboardInterrupt`,
  `asyncio.CancelledError`, `SystemExit`) keeps the pending assay, because Zarr may still be
  running the interrupted write, and logs how to remove it.

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
`provenance.extras["unmeasured_cells"]`.

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

Writers compute the raw-count identity while they write `counts`, so no second pass reads the
matrix back. `counts` records it as `content_fingerprint`. The sibling `countSummaries` group holds
per-cell totals (`rowSums`), per-cell detected-feature counts (`rowPositive`), and per-feature
detected-cell counts (`columnPositive`), and its `source_fingerprint` attribute names the counts
it describes. Preparation reads these summaries instead of streaming the matrix. Summaries that are
missing, malformed, or bound to other counts are an error; rebuild the store with `--data-only`.

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

The first writable `DataStore` open of a fresh import prepares its assays: it derives their cell
and feature summaries and publishes their identity. One process prepares a store: the preparing
open holds a claim, the group `.cytearc-preparation` below the root or workspace group, which it
creates with the store's conditional write (`set_if_not_exists`) and removes when the preparation
ends, however it ends. Another process that opens the store meanwhile raises `RuntimeError`, which
names the process and host that hold the claim, instead of writing the same columns beside it; open
it again once that open has finished. A killed preparing process leaves its claim, and the error
names the claim to remove once that process has stopped. A backend whose conditional write is not
atomic gives a best-effort claim, and one that refuses the conditional write prepares without a
claim, with a warning.

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

After a rewrite, recompute HVG, normalization, PCA, graph, and marker artefacts.
Do not resume them from pre-rewrite lineage.

Non-RNA assays in a multi-assay store remain usable when tooling opens without forcing every assay class to construct.

## Zarr versions

New stores default to Zarr v3.
RNA assays require v3 because sharded `countsT` is a v3 feature.
A Zarr v2 RNA store will not open as `RNAassay`.

## Memory controls

`DataStore(..., mem_budget='8G')` bounds streaming and concurrency (blocks, concurrent work, feature batches).
A Zarr read holds more than its result. A read of a sharded array, such as `counts` or `countsT`, fills a shard-level copy of its selection, holds the compressed bytes of every inner chunk it touches in a shard, and decodes those chunks next to them. Plans charge this through `ArrayGeometry.readBytes`, at the decoded size of the compressed bytes because a plan cannot know them before it reads: a `counts` row block of a whole shard reserves twice its bytes, the shard's chunks, and the one chunk that row-block streams decode at a time, and a `countsT` cell band reserves twice its bytes and two of each chunk it touches. An unsharded read, such as one of normalized data, is charged its result and the chunks it decodes; the compressed bytes of a chunk it decodes stay outside the plan. A thread of Zarr's pool that decoded a chunk can keep it until the thread next runs, and for a sharded array also the buffer that the read fetched from the shard, since each inner chunk is decoded from a view of it. Plans count what a read holds while it runs, not what a thread keeps after it, which is part of the headroom that the memory page asks for. That leftover was the chunk of the read just before in every measurement, at most about 270 MB of normalized values in the default layouts, as the memory page states.
The reads that the memory ledger of `utils.memory_budget` charges, `read_rows`, `read_text_rows` and `read_whole` of `metadata.rows` with the repertoire operations' other reads, decode on a pool of exactly the store's `nthreads` threads, whatever the machine. `storage.async_execution` installs that pool as the default executor of Zarr's event loop, where Zarr runs each fetch and each compressor's decode of a read made through its synchronous API, in the context of the call that made the read: a read within `zarr_decoding_threads(n)` runs them on a pool of `n` threads, which the reads of every call of that count share, from any thread, and any other read runs them on the host's threads, as many as Zarr's pool had before. They read a sharded array a band of inner chunks at a time, so that no fetch takes more than one band of a shard, and a whole unsharded array in one call at the highest concurrency, up to the operation's, whose chunks fit the ledger's room. `ArrayGeometry.wholeReadBytes` and `bandReadBytes` state what these reads hold, and the memory page lists every charge ({ref}`what a refusal names <memory_refusals>`).
The text of a cell-data artifact is converted and written one stored chunk at a time, so `cell_data_write_bytes` in `metadata/artifacts.py` counts a text array's chunk and its encoded copy, and a numeric array's chunks that Zarr encodes at once, up to its async concurrency.
Index arrays on both axes of a sharded array never reach Zarr together: Zarr would turn them into coordinates of every selected value inside each shard, which hold about thirty times the result. `read_selection` instead reads one stored chunk of one axis at a time, as a band that slices the axis over the indices the chunk holds while Zarr indexes the other axis, and slices the axis whose band is narrower. Plans charge the result beside one Zarr read of that band through `indexed_read_parts`.
Environment variables used by the documentation executor (`CYTEARC_MEM_BUDGET`, `CYTEARC_WORKERS`, …) are also useful for local large runs.

## Related guides

- [](../tutorials/data_organization.ipynb)
- [](../concepts/memory_and_execution.md)
- [](contributing.md)
- [](operation_revisions.md)
