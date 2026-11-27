---
description: Configure CyteArc memory budgets, worker concurrency, and storage profiles, and inspect resource use during analysis.
---

(memory_and_execution)=
# Memory and execution

CyteArc keeps large matrices in Zarr and streams planned blocks through memory. It does not load the full count matrix simply because a `DataStore` is opened. Memory use still depends on the operation: graph construction and clustering can hold structures in addition to the streamed count blocks.

## Set resources and inspect a run

Set a budget and worker count when sharing a machine or submitting a batch job. For an existing RNA store, this example allows up to four workers and a 16 GiB operation budget:

```python
# Import CyteArc for the analysis.
import cytearc

# Open the existing store with the chosen resource limits.
ds = cytearc.DataStore("data.zarr", mem_budget="16G", nthreads=4)

# Run the RNA recipe under a new analysis label.
run = ds.pipeline.rna(label="rna_analysis")

# Inspect recorded stage timings, memory use, and reused results.
run.report()
```

Use a new label for each run. To inspect a completed run without recomputing it, reopen its label with `ds.pipeline.open(label="rna_analysis")` and call `report()` on that result. The report includes stage timings, sampled resident memory (RSS), and the warnings that each stage logged. Sampled peaks can miss brief allocations.

`mem_budget` is an operation budget, not a hard cap on total process resident memory. It covers the data that CyteArc holds: the arrays and blocks that grow with the cells, features, values and chunks an operation works on, and an allowance for each read of stored arrays in flight. Python, Zarr and NumPy hold a little more of their own, such as objects and temporaries, so leave some headroom beside the budget. Memory the process already holds when an operation starts, such as the interpreter, native libraries, graph structures, and earlier results, is not subtracted either, so a process can peak at its resident memory plus the budget.

The sections below explain what the budget covers, how workers are chosen, and how to compare measurements.

## Resource controls

(memory-budget)=

### Memory budget

`mem_budget` accepts bytes, a size such as `"8G"`, or a fraction of detected system memory such as `"0.6"`. The size suffixes K, M, G, and T are powers of 1024, alone or followed by `B` or `iB`, so `"8G"`, `"8GB"`, and `"8GiB"` are the same budget. A budget that is not a positive finite size, such as `"inf"`, and a worker count below one, from `nthreads` or `CYTEARC_WORKERS`, raise `ValueError`. Without an explicit value or `CYTEARC_MEM_BUDGET`, it defaults to 75% of detected memory, using the smaller of physical memory and any detected cgroup limit: the total, not the memory free when the budget is resolved, so leave room for other processes on a shared host. Physical memory comes from `/proc/meminfo` on Linux, `sysconf` on macOS and other Unix systems, and `GlobalMemoryStatusEx` on Windows. Where neither physical memory nor a cgroup limit can be read, a default or fractional budget raises `RuntimeError`, and a size such as `mem_budget="16G"` still works. Explicit budgets, including fractions of detected memory, are not reduced.

The budget sizes the work. CyteArc estimates the arrays that grow with the data, such as count blocks, coordinates, graphs, and per-cell IDs and labels, plans each step so that they fit, and refuses a step that cannot fit before it reads or writes. Streamed steps size their blocks, their concurrent reads and writes, and their feature batches from the budget, so a smaller budget takes more passes over the data and gives the same result. Each read of stored arrays in flight is also charged an allowance for Zarr's objects, NumPy's index buffers and the buffer of the file it reads: 128 KiB, and 12 KiB for each chunk that it decodes at once beyond the 10 that Zarr decodes at once by default. At that default, a read held at most 97 KB of these on a local file system; a file system with large blocks can give each read a larger buffer than the allowance covers. The estimates leave out what does not grow with the data, such as the interpreter, the other objects and buffers of Python, Zarr and NumPy, and their temporaries, so a step can hold a little more than its estimate, and a process peaks at about the budget plus that baseline and the memory it already held.

Imports write the default count layout unless they are given a `policy`, and an import whose default count shards do not fit stops before it writes, naming the smaller `policy` that fits; given `policy="fit"`, it writes that policy instead. Subset, repack, merge, and grouped assays fit the count layout to the budget by default, writing smaller count shards. The layout changes how the counts are stored, never the store's identity.

Some steps hold whole structures beside their streamed blocks, and compare an estimate of them with the budget before they read a coordinate or graph. The largest are:

- a streamed read: its block, and the chunks that Zarr holds compressed and decoded while it reads them; a sparse import also holds its source batch, with a row and a column index beside each value, and the band of rows it writes;
- an approximate nearest-neighbor index: every cell's coordinates and links, about `4 * dims + 8 * ann_m + 130` bytes per cell, while `graph.ann_index` builds and saves it and while `graph.neighbors` loads and queries it, and a query's indices and distances, 8 bytes per neighbor;
- an in-memory k-means embedding initialization: the coordinates of every cell; pipeline cluster selection: the coordinates of its sample of at most 10,000 cells and the silhouette's distance chunks, each at most 1 GiB;
- Harmony: the float64 coordinates three times and the soft cluster assignments, about 1.5 KB per cell with 30 dimensions and 100 clusters;
- an import that shuffles cells: the order of the cells and its inverse, 16 bytes per cell; H5AD and Seurat imports also hold each per-cell array other than counts, such as embeddings, cluster labels and cell IDs, whole while they write it;
- WNN and SNN integration: every source graph and the integrated graph;
- the connectivity conversion: every cell's neighbors and the edges built from them, about 29 bytes per neighbor;
- Leiden clustering, UMAP and t-SNE: the whole graph, about 28 bytes per stored edge and twice that when they symmetrize it, beside what they compute, such as UMAP's fuzzy union and sampling schedules, about 100 bytes per stored edge; for t-SNE only the arrays that CyteArc passes to `sgtsnepi` are counted;
- the trajectory calls that score or load a pseudotime or a fate map, `imputation.diffusion`, the smoothing of `scores.potency` and `graph.load`: the graph they load, and for the trajectory calls up to three more copies of it while they number its connected components, beside the solve of the scored component;
- the loaders of saved feature analyses along a pseudotime or a fate map: the saved table of every feature, then the selected features' rows.

Over the budget, such a step raises `MemoryBudgetError` before it reads anything or creates or loads an index; it never changes its algorithm on its own.

`graph.ann_index` saves the index through a temporary file of about `4 * dims + 8 * ann_m + 20` bytes per cell, and `graph.neighbors` loads it through another one of the same size, in the system's temporary directory (`TMPDIR` on Linux and macOS); leave that much free space there. `variants.import_counts` sorts the rows that outgrow its budget in runs in the same directory, about 4 bytes per count row of mgatk's files, and removes them when the import ends. An import that shuffles cells keeps the cells' rows in the reader's `temp_dir`, about one byte per stored count, and deletes them as it writes the counts. A composite with remote sources keeps downloaded source objects in the same temporary directory, up to `CYTEARC_SOURCE_CACHE_BYTES` per open composite (2 GiB unless set; `0` reads the sources directly). That disk space is outside `mem_budget`, and where the temporary directory is memory-backed, such as a `tmpfs` `/tmp`, it uses memory.

An RNA run with `pca={"dims": 0}` builds its graph on the normalized values of the selected features, one coordinate per highly variable gene, so the memory and time of the ANN index, the neighbor queries, the embedding initialization, and cluster selection scale with the HVG count instead of the PCA dimensions. At 1,000,000 cells the index alone holds about 4.3 GB with 1,000 HVGs and 8.3 GB with 2,000 at the default `ann_m=24`, and the embedding initialization holds every cell's coordinates, 4 bytes per HVG per cell, when one stored band holds every cell; pass `embedding_initialization={"batch_size": ...}` with a batch size below the cell count to select its streamed fit, which holds a sample of the cells and one block of coordinates at a time.

A stream that closes early, such as one whose consumer stops or fails, can leave a chunk fetch or decode of one of its reads still running on a codec thread. That call ends just after, holding at most one chunk per busy codec thread, which is part of the headroom to leave: in the default layouts a chunk holds at most 128 MiB of normalized values, 100 MB of counts and 16 MiB of a new store's cell column.

(memory_refusals)=
#### What a refusal names

A step that would pass the budget raises `MemoryBudgetError`, a `MemoryError`, before it reads or writes. Its message names the step, the bytes that the call needs at that step and the budget, and closes with a budget that runs the step:

```text
Importing contigs for 8 selected cells holds about 1154586 bytes for a chunk of 1000 contig rows
as parsed, more than the memory budget of 1000000 bytes. Raise mem_budget to at least 1154586
bytes, or pass a smaller cell_selection.
```

The error holds the same numbers as `needed_bytes` and `budget_bytes`, and the step as `item`. Planners and other checks word their refusals in their own terms, with the same three fields. To act on a refusal:

- Rerun with `mem_budget` of at least `needed_bytes`, which runs the refused step. A later step can need more and refuses the same way, so a call can be refused more than once as its budget rises.
- Or take the remedy that the message names after "or", such as a smaller `cell_selection`, which makes the step fit the same budget.
- `needed_bytes` is always a whole `mem_budget`, also where a step plans with part of the budget, as normalization's producer does beside its writer.
- A step whose blocks take a share of the room, such as a file's chunks, reads larger blocks at a larger budget, and can also run a little below the named budget, with smaller blocks.
- A converter whose default count layout does not fit raises `CountLayoutMemoryError`, a `MemoryBudgetError`, whose `fittingPolicy` is the largest smaller layout that fits; `policy="fit"` writes it.

A count write handed a batch larger than its producer reserved names what the write would reserve with that batch, which can be below the budget, since the producer broke its own reservation. The checks of the variant operations raise a `MemoryError` of their own, and the screen's `efficiency`, `edistance`, `signature`, `mixscape`, `associations`, `profiles` and `cluster` raise a `MemoryBudgetError` in their own words.

The identity checks of stored cell and feature selections, which most operations run once on their inputs, read the row identifiers one stored chunk at a time and hold that chunk outside the budget, at most 16 MiB of values in a new store's cell table. glibc can also keep freed buffers of 32 MiB or less resident after an operation; setting the `MALLOC_MMAP_THRESHOLD_` environment variable, for example to `131072`, before Python starts returns them to the system.

### Worker concurrency

Worker concurrency is auto-detected from the process environment. On shared hosts, multiprocess jobs, or remote object stores, set `CYTEARC_WORKERS` or pass the advanced `nthreads` constructor argument to bound the maximum worker budget. Zarr decompresses and compresses a store's chunks on at most `nthreads` threads for every read and write through its `DataStore`: opening the store, its operations, plots, exports and recipes, and the arrays that you compute from it yourself, such as `ds.RNA.rawData`, so `nthreads=1` keeps that work on one core. DataStores open on one store, or on one location, such as a local folder or a remote URL, decompress its chunks on the fewest of their `nthreads`, and an import, a subset, a merge, a mount, a composite or a repack does this work on its own `nthreads`. A Zarr codec pipeline that you configure in place of Zarr's default decompresses on all of the host's threads. More workers can increase concurrent buffers or remote requests, so pair a large worker budget with an explicit `mem_budget`. Opt-in parallel UMAP layouts and ANN index builds are faster but not reproducible: two runs on several threads can differ even with the same thread count. Their artifact identities therefore record only the `parallel` or `ann_parallel` flag, and the thread counts are execution options, so the same request reuses its result on any machine. A serial UMAP layout runs on one thread and releases the GIL during each epoch, so other Python threads of the process, such as an application's heartbeat, keep running. Doublet detection scores as many samples at once as the workers and the memory budget hold. A sample of 50,000 cells and simulated doublets or more is indexed in 4 parts, built at once, and every index is built on one thread, whatever `ann_parallel` says, so a sample's scores are the same for any thread count. Leiden clustering with the default `"cytearc"` backend runs on the worker budget and returns identical labels for any thread count, so its artifacts record no worker count either; the optional `"igraph"` and `"leidenalg"` backends run on one thread. t-SNE always runs on one thread through the optional `sgtsnepi` package, so its artifacts record no worker count. GRN inference with `nthreads` above 1 fits the targets left after about two seconds of fitting in worker processes, as many as `nthreads` allows; `regulons.grn` runs fewer when the memory budget holds fewer of them, each charged its own copy of the regulators and one fit. Every target's regressor has the same seed, so the links are the same for any thread count. The workers start as new interpreters that import the main module, so a script keeps its own work under `if __name__ == "__main__":`.

Each process plans within its own `mem_budget` and `nthreads` and knows nothing of what other processes hold, so processes that analyze one store at once ([Several processes on one store](provenance.md#several-processes-on-one-store)), or share a machine, each need their share: two processes might each take `mem_budget="0.35"` and half the cores.

## Why RNA stores have two count orientations

Most analysis steps walk the matrix by cell. Quality control, library-size normalization, and graph construction read rows of `counts`. Gene-wise steps such as highly variable gene selection and marker search walk the matrix by feature.

Those two access patterns compete on a single physical layout. CyteArc therefore stores RNA counts twice: `counts` is cell-major, and `countsT` contains the same values in gene-major order. They are two orientations of one assay matrix, not two datastores. The second orientation roughly doubles stored RNA counts. ATAC, ADT, and other non-RNA assays keep only `counts`.

Current RNA stores require a matching current-layout `countsT`. Store inspection, import, and offline rewrite procedures belong to [](../tutorials/import_and_export.ipynb) and the [](../reference/faq.md), rather than the memory-planning model on this page.

## Storage profiles

New writers use Zarr v3 and choose one of two profiles:

- `fast_local` uses LZ4 with bit-shuffle and is selected automatically for local filesystem, memory, and `file://` targets.
- `cloud` uses Zstandard level 3 and is selected automatically for remote URI targets and other non-local stores.

Override automatic selection with a writer's `profile=`, a datastore's `zarrProfile=`, or `CYTEARC_ZARR_PROFILE=fast_local|cloud`. The profile determines the physical encoding when arrays are written. Changing it while reopening an existing store does not rewrite the arrays.

See [](../tutorials/remote_stores.ipynb) to mount a remote dataset and save your work separately. Storage options, source identity, and `local_cache` scratch behavior are covered in [](../reference/remote_storage.md).

## Batch and HPC output

Disable animated progress in non-interactive logs and add timestamps:

```python
# Configure output for a non-interactive job.
import cytearc

# Disable progress bars and add log timestamps.
cytearc.configure_output(progress=False, timestamps=True)
```

Progress rendering and log severity are independent. To also select a log level or file:

```python
# Write informational logs to a local file.
cytearc.set_verbosity(level="INFO", filepath="cytearc-run.log")
```

A job that configures Python's `logging` can receive CyteArc's records there instead: `cytearc.configure_output(stdlib=True)` sends them to the standard-library logger named `cytearc`, and a record logged during a pipeline stage carries the run ID and stage as its `cytearc_run_id` and `cytearc_stage` attributes. `set_verbosity` returns the records to CyteArc's own output. With progress bars off, a pipeline run still sends `stage_progress` events to its `callback` and listeners, so a job can record progress without rendering bars.

See the [utilities API reference](/api/utilities.html) for the exact output contract.

(measuring-resource-use)=

## Measuring resource use

Measure the complete workflow on the hardware and storage service you intend to use. Runtime and memory depend on dataset sparsity, selected features, graph parameters, worker count, cache state, and software versions.

Record the dataset and cell selection, analysis settings, code revision, dependency versions, CPU and memory limits, storage location, and whether the run reused existing artifacts. Keep conversion and download time separate from analysis time, and report both stage timings and the complete elapsed time. Repeat measurements under the same conditions before comparing changes.

A pipeline run records, for each stage, its wall and CPU time, sampled resident memory and the process's peak, OS threads, network bytes, the worker count and memory budget it was given, the CPUs and memory its process could use, and a trace of samples, beside its warnings and summary, in `run.report()` ([Stage resources](/api/pipeline.html#stage-resources)). `ds.pipeline.plan` reports these values from the latest earlier run that computed each stage, so the stages that stay on one core, read the most over the network, or need the most memory show before a run starts. Sampled peaks can miss brief allocations; the process's peak RSS, which the kernel keeps, holds them when a stage raised it. `mem_budget` controls planned work and is not a hard limit on the process's resident memory; see [Memory budget](#memory-budget).

For focused performance checks, the repository's benchmark suite measures input-size ladders and projects their scaling to larger workloads:

```bash
# Measure the benchmark input sizes and project their scaling.
CYTEARC_RUN_BENCHMARKS=1 uv run pytest -n 0 tests/benchmarks
```

Without a baseline, this records measurements without a regression comparison. See [](../developers/contributing.md) for creating a baseline, comparing revisions on the same machine, and interpreting projected delays.

Compare remote and local storage only with matched datasets, analysis settings, resources, and cache conditions. Measurements from one workload do not establish biological correctness or hardware requirements for another. See [](../tutorials/remote_stores.ipynb) for storage and mounting mechanics.
