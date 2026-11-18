---
description: Understand why CyteArc writes the cells of each import in a random order, what that order guarantees, how exports restore the file's order, what the shuffle costs, and how to keep the file's order.
---

(cell_order_concepts)=

# Why CyteArc shuffles cells

Count files usually list their cells grouped by sample, donor, plate or lane. A store in the file's order holds the cells of one or two samples in any contiguous block of rows, such as one count shard of about 11,000 cells or a mini-batch that a training loader reads, so the block is a biased sample of the dataset. CyteArc therefore writes the cells of every import in a uniformly random order. Any contiguous block of rows is then a random sample of the store's cells, and a loader can read large blocks, the fastest way to read a store, and still mix every sample in each batch.

## What the order guarantees

The order is a uniform random permutation of the imported cells: every order is equally likely, so every contiguous block of rows, of any size and at any position, is a simple random sample of the cells. `shuffle_seed`, 0 by default, chooses the permutation, and one seed gives the same order on every platform and NumPy version, so an import of the same file with the same seed writes the same store.

Every per-cell array follows the one order: `counts` and `countsT`, the cell IDs and names, the cell metadata columns, assay membership, and the embeddings, cluster labels, reductions and identities that an import brings. Values that you attach to cells later, such as labels from another tool or an scVI latent space, are matched to the cells by cell ID, so the order never misplaces them: `ds.cells.insert` and the other calls that take one value per cell take a pandas Series or DataFrame indexed by cell ID and refuse a bare array.

Analyses that draw random numbers, such as UMAP and Leiden clustering, draw them over the cells in store order. A store imported in another order therefore gives slightly different layouts and clusters, as another seed would, and its cluster numbers can differ.

## Where each cell came from

Each cell keeps its row in the imported file, counted from 0, in the int64 cell column `sourceRow`, and the cell table records the seed in its `cellOrder` attribute. A reader that drops cells, such as `CrH5Reader(is_filtered=False)`, keeps the row in the file of each cell it keeps, so the column has gaps where the dropped cells were. `insert`, `update_key`, `reset_key` and `drop` refuse `sourceRow`.

Exports write the cells back in the file's order. `to_h5ad`, `to_mtx`, `DataStore.to_anndata` and the clone exports sort the exported cells by `sourceRow`, so an import followed by an export gives back the file's cells in the file's order, and `cell_order="stored"` keeps the store's order instead. Subsets, repacks and mounts keep each cell's `sourceRow`, and a merge gives each cell its row in the concatenation of its sources, so their exports follow the files too. The [import and export reference](/api/import_export.html#cell-order-in-exports) describes each export's order.

## What the shuffle costs

These figures come from imports of gzip-compressed samples of a CELLxGENE embryo dataset on 8 CPUs with a 24 GiB memory budget and a local destination.

- **Import time.** An H5AD import of 10 million cells took 1,531 seconds shuffled against 1,467 seconds in the file's order, 4% longer; at 1 million cells the two took the same time. A shuffled 10x HDF5 import of 1 million cells was faster, 411 and 427 seconds against 455 and 474 seconds, because its counts write reads local temporary files instead of the gzip file.
- **Temporary disk.** The shuffled rows wait in temporary files in the reader's `temp_dir`, about 1.1 bytes per stored count: 18.5 GB for the 17 billion counts of the 10 million cell import. The counts write deletes each file once it has read it, so a local import needs no disk beyond its final store.
- **Store size.** `counts` keeps its size and `countsT` takes 4% more disk, because a gene's values over cells in random order compress a little less.
- **Exports.** An export in the file's order reorders the rows through the same kind of temporary files. A `to_h5ad` export of 1 million cells took 148 seconds against 129 seconds in store order, 14% longer, and its 1.85 GB of temporary files were deleted as the file was written.
- **Reading one sample.** A sample's cells are spread over the whole store, so reading the counts of one sample decodes every chunk of `counts`, where a store in the file's order decodes only the chunks that hold that sample. When an analysis reads one sample at a time from a large store, write each sample to a store of its own once with `SubsetZarr(..., cell_key=...)`, or import in the file's order.

## Keep the file's order

Pass `shuffle_cells=False` to the writer to write the cells in the file's order:

```python
import cytearc

# Import the file with its cells in the file's order.
reader = cytearc.H5adReader.from_inspect(cytearc.inspect_h5ad("counts.h5ad"))
cytearc.H5adToZarr(reader, zarr_loc="counts.zarr", shuffle_cells=False).dump()
```

Every import writer takes the option: `H5adToZarr`, `CrToZarr` and `MtxToZarr`, `TapestriToZarr`, `CSVtoZarr`, `SparseToZarr` and `SeuratToZarr`. An import in the file's order writes neither `sourceRow` nor `cellOrder`, and its store is the store that CyteArc 1.4.0 and earlier releases write. Stores imported by those releases keep their order, open unchanged and export in their stored order, which is their file's order.

A file imported again with the shuffle gets a store in another order than the store that an earlier release imported from it, so the two stores have different dataset fingerprints, and a composite or a mount that recorded the earlier store refuses the new one in its place. Import with `shuffle_cells=False` to rebuild the earlier store.
