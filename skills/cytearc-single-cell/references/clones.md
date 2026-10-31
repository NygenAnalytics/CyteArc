# Clones and lineage tracing

Import the clones that a lineage tracing tool or a study assigned to cells, join them to the store's cells, check how many barcodes joined, and see how large the clones are. Docs: <https://docs.nygen.io/CyteArc/api/clones.html>.

## When to use

- A file that lists each cell's clone: CoSpar's `Cell_ID,Clone_ID` table, TREX's `clones.txt`, CellTag-multi's or CellTagR's clone table, STICR's Cell Browser `meta.tsv`, Cassiopeia's allele table (`lineageGrp`), Numbat's `clone_post_{i}.tsv`, LARRY's GEO clone matrix with its metadata, an AnnData file with CoSpar's `obsm['X_clone']` or MiTo's `obs['MiTo_clone']`, or any table with a cell and a clone column.
- A cell metadata column that already holds each cell's clone: `ds.clones.from_column`.
- Not for files that list barcodes per cell or per molecule without clone IDs (TREX `molecules.txt`, pycashier, BARtab counts): those need clone calling, which this module does not do.

## Key concepts

- `ds.clones.import_clones(paths, *, cell_selection, barcode_prefix=None, barcode_suffix=None, barcode_pattern=None, format=None, columns=None, multiple="error", invalidate_cache=False)` saves a datastore-scoped `clones` artifact. The format comes from the file: HDF5 is AnnData, `%%MatrixMarket` a clone matrix, and a text header names the tool. `format` is one of `"cospar"`, `"trex"`, `"celltag_multi"`, `"celltagr"`, `"sticr"`, `"cassiopeia"`, `"numbat"`, `"table"`, `"matrix"`, `"cospar_h5ad"`, `"mito"`, `"h5ad"`.
- `columns={"cell": ..., "clone": ..., "probability": ...}` reads any other table (format `"table"`) or an AnnData `obs` column (`"h5ad"`, clone and probability only). `"cell"` may list columns joined by `"__"`, as LARRY's `["Library", "Cell barcode"]`. A table with R row names (header one field shorter than its rows, such as the Biddy workflow's `celltag_data.txt`, or a first column with an empty name, as `write.csv` writes) takes cells from the row names, so `columns={"clone": "CellTagD13"}` reads one labelling round. Blank lines hold no rows, and a delimiter after every row's last field ends no column.
- A clone matrix reads its cell list beside it: `<name>metadata.txt.gz` beside `<name>clone_matrix.mtx.gz` (LARRY on GEO), or `cells.tsv` with `clones.tsv` (the export). The axis whose length equals the cell list holds the cells, so LARRY's in vitro (cells by clones) and in vivo and cytokine (clones by cells) files read alike. Unnamed clones are labelled by position from 0.
- The join: a cell's ID is `barcode_prefix + barcode + barcode_suffix`; `barcode_pattern` keeps the named group `barcode` of each barcode (`re.search`), and a barcode it does not match joins no cell. Each option takes one value for every file or one per file. A merged store names cells `<sample>__<barcode>`, so a sample's 10x file needs `barcode_prefix="<sample>__"`, and a TREX file of bare barcodes also `barcode_suffix="-1"`.
- Check the join: the artifact attributes `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` count the files' barcodes and those that matched no selected cell, and a warning names up to five; a reused import warns again. Many unmatched barcodes mean a wrong prefix, suffix or pattern, a file of another run, or a selection that left out cells. Non-cell droplets in a file are expected to be unmatched.
- Several files: each file's clone labels start with its `barcode_prefix`, or with its name before the first `.` and `_`, because clone IDs are unique only within one run (TREX numbers clones from 1 in every run). Two files with one namespace, such as two `clones.txt` without prefixes, raise `ValueError`.
- A cell in two different clones raises `ValueError` naming examples; `multiple="drop"` leaves such cells without a clone and counts them in `n_cells_in_several_clones`. Repeated rows of one cell and clone (Cassiopeia lists a cell per target site) count once. Cassiopeia's lineage group 0, MiTo's `"unassigned"`, and empty, `NA`, `NaN` or `nan` labels mean no clone.
- The file's bytes identify it: a moved file reuses the saved clones, an edited one gives new ones, and the order of files does not matter. `invalidate_cache=True` imports again.
- `ds.clones.from_column(column, *, cell_selection, invalidate_cache=False)` saves the clones of a cell metadata column; missing or blank labels mean no clone, and float whole numbers read as integers.
- Memory: the store keeps each cell's label at the width of the longest, so very long labels cost memory for every cell; an import that would not fit `mem_budget` raises `MemoryError` before saving.
- The artifact: canonical `values` (clone label, empty for none), `codes` (clones by size, largest first, then label; -1 for none), `clone_size` (0 for none), and `clone_probability` where the source gives one (Numbat's `p_opt`, or `columns["probability"]`). `ds.select_cells(clones, include=["3"])` selects one clone's cells, and `ds.artifacts.load_values(clones, value="clone_size")` reads the sizes.
- `ds.clones.load_clones(clones)` returns a `Clones` record: `.cells` (indexed by cell ID: `clone` categorical in rank order, `clone_size`, `clone_probability` when stored), `.clones` (`clone`, `n_cells`, largest first) and `.source` (`"import_clones"` or `"clones_from_column"`).
- `ds.clones.export_clones(clones, path, format="table")` writes CoSpar's `Cell_ID,Clone_ID` CSV; `format="matrix"` writes a cells by clones Matrix Market file with `cells.tsv` and `clones.tsv` beside it, replacing files of those names. Both import back to the same clones; probabilities are not written. The table refuses labels that read as no clone, such as `NA` from `from_column`, and the matrix refuses labels with line breaks: use the other format. Returns the cells written.
- `ds.plots.clone_abundance(clones, group_by=None, groups=None, top=20)` draws clones per clone size on log scales and the largest clones; with a grouping, one line per group and each clone's cells stacked by group. Tables: `sizes` (`group`, `clone_size`, `n_clones`) and `top`.

## Recipe

```python
cells = ds.snapshot_cell_selection("I")
# One TREX run per sample, with bare 10x barcodes, in a merged store.
clones = ds.clones.import_clones(
    ["d2/clones.txt", "d4/clones.txt"],
    cell_selection=cells,
    barcode_prefix=["d2__", "d4__"],
    barcode_suffix="-1",
)
attributes = ds.artifacts.load(clones).attrs
print(attributes["n_unmatched_barcodes"], "of", attributes["n_file_barcodes"], "barcodes unmatched")
table = ds.clones.load_clones(clones)
table.clones.head(10)
ds.plots.clone_abundance(clones, group_by="sample")
```

```python
# LARRY's GEO clone matrix and metadata; the store names cells "<Library>__<Cell barcode>".
clones = ds.clones.import_clones(
    "GSM4185642_stateFate_inVitro_clone_matrix.mtx.gz",
    cell_selection=cells,
    columns={"cell": ["Library", "Cell barcode"]},
)
```

## Checks

- Compare `n_unmatched_barcodes` with what the file should hold outside the selection before suspecting the prefix, suffix or pattern.
- Check `table.clones["n_cells"]` against the study's reported clone count and sizes; LARRY's in vitro and cytokine sets hold only clones of two or more cells, while TREX keeps singletons.
- With several files, check that `table.clones["clone"]` carries each file's namespace, and that clones expected to span samples were called jointly in one run; clones of separate runs never merge.
