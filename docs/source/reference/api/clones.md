---
myst:
  html_meta:
    description: CyteArc clones API for lineage tracing, with clones imported from CoSpar, TREX, CellTag-multi, CellTagR, STICR, Cassiopeia, Numbat, MiTo and LARRY's clone matrices or from a cell metadata column, joined to cells by barcode prefix, suffix or pattern, saved as a clones artifact, read as tables of cells and clones, exported as CoSpar's clone table or a Matrix Market file, and drawn as clone sizes.
---

# Clones API reference

A clone is the set of cells that descend from one ancestor cell. Lineage tracing methods mark the ancestor with a heritable barcode, such as LARRY's, CellTag's or TREX's expressed barcodes, and other methods infer clones from the cells' own mutations, such as copy-number changes or mitochondrial variants. Each tool writes which clone each cell belongs to in its own format. `ds.clones` reads those files, joins their cells to a store's cells, and saves the clones as a `clones` artifact that CyteArc's clone analyses read whatever defined the clones.

{meth}`ds.clones.import_clones <cytearc.datastore.namespaces.ClonesAccessor.import_clones>` imports the clones of one file or several, and {meth}`ds.clones.from_column <cytearc.datastore.namespaces.ClonesAccessor.from_column>` takes them from a cell metadata column. {meth}`ds.clones.load_clones <cytearc.datastore.namespaces.ClonesAccessor.load_clones>` reads the artifact as a {class}`~cytearc.clones.Clones` record, {meth}`ds.clones.export_clones <cytearc.datastore.namespaces.ClonesAccessor.export_clones>` writes it as a file that other tools read, and {meth}`ds.plots.clone_abundance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_abundance>` draws how many clones have each size.

```python
cells = ds.snapshot_cell_selection("I")
clones = ds.clones.import_clones("trex_run/clones.txt", cell_selection=cells, barcode_suffix="-1")
table = ds.clones.load_clones(clones)
table.clones.head()  # the largest clones and their cells
ds.plots.clone_abundance(clones, group_by="sample")
```

## Import clones from a file

`import_clones` recognises each file's format from its contents: an HDF5 file is AnnData, a file that starts with `%%MatrixMarket` is a clone matrix, and a text table is recognised by the names of its columns. `format=` names the format when the contents leave it open, and `columns=` reads any other table by the columns it names.

| `format` | File | Columns read |
|---|---|---|
| `"cospar"` | [CoSpar](https://github.com/ShouWenWang-Lab/cospar)'s clone table | `Cell_ID`, `Clone_ID` |
| `"trex"` | [TREX](https://github.com/frisen-lab/TREX)'s `clones.txt` | `cell_id`, and `clone_nr`, `clone#` or, in the 2022 releases' CSV, `#clone_id` |
| `"celltag_multi"` | [CellTag-multi](https://github.com/morris-lab/CellTag-multi-2023)'s clone tables, such as the iEP table on [Zenodo](https://zenodo.org/records/23442375) | `cell.bc`, and `clone_id` or `clone.id`; an unnamed index column and annotation columns are skipped |
| `"celltagr"` | [CellTagR](https://github.com/morris-lab/CellTagR)'s clone composition | `cell.barcode`, `clone.id` |
| `"sticr"` | STICR's cell metadata on the [UCSC Cell Browser](https://cells.ucsc.edu/?ds=human-cortical-lineage) | `cell_Id`, `VBC` |
| `"cassiopeia"` | [Cassiopeia](https://github.com/YosefLab/Cassiopeia)'s allele table, one row per cell and target site | `cellBC`, `lineageGrp`; group 0 means no clone |
| `"numbat"` | [Numbat](https://github.com/kharchenkolab/numbat)'s `clone_post_{i}.tsv` | `cell`, `clone_opt`, and `p_opt` as each cell's `clone_probability` |
| `"table"` | any delimited table | the columns that `columns` names |
| `"matrix"` | a Matrix Market clone matrix with its cell list beside it, such as LARRY's on [GEO GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802) | the cell list's columns that `columns["cell"]` names |
| `"cospar_h5ad"` | an AnnData file with CoSpar's `obsm['X_clone']` | the cells by clones matrix |
| `"mito"` | an AnnData file with [MiTo](https://github.com/andrecossa5/MiTo)'s `obs['MiTo_clone']` | `"unassigned"` means no clone |
| `"h5ad"` | an AnnData file | the `obs` column that `columns["clone"]` names |

`columns` maps `"cell"`, `"clone"` and `"probability"` to column names. `"cell"` may list several columns, whose values joined by `"__"` form the barcode, as a merged store joins a sample and a barcode. A table whose header holds one field fewer than its rows, as R's `write.table` writes it, takes each cell from its row names, so a wide table with one clone column per labelling round, such as the `celltag_data.txt` of the Biddy et al. workflow, needs only `columns={"clone": "CellTagD13"}` to read one round. Clone labels are text as the file writes them; an empty field, `NA` or `NaN` means no clone.

Tables may be plain or gzip-compressed, whatever their names, separated by tabs, commas or spaces, quoted as R quotes them, and without a final newline. A header that starts with `#`, as TREX writes, is a header. A row with more fields than the table's first row raises `ValueError`.

A clone matrix holds a nonzero entry where a cell belongs to a clone. Its cell list is a file beside it: `<name>metadata.txt.gz` beside `<name>clone_matrix.mtx.gz`, as GEO ships LARRY's, or `cells.tsv`, with `clones.tsv` naming the clones, as `export_clones` writes. The axis whose length equals the cell list's rows holds the cells, so LARRY's in vitro matrix, cells by clones, and its in vivo and cytokine matrices, clones by cells, read alike; a matrix that fits both ways holds its cells in rows. Clones that no list names, and the columns of CoSpar's `X_clone`, are labelled by their position from 0, as SciPy and AnnData number them.

```python
# LARRY's cytokine perturbation set from GEO: the metadata names each cell
# by its library and inDrops barcode.
clones = ds.clones.import_clones(
    "GSM4185644_stateFate_cytokinePerturbation_clone_matrix.mtx.gz",
    cell_selection=cells,
    columns={"cell": ["Library", "Cell barcode"]},
)
```

### Join barcodes to cells

A cell's ID is `barcode_prefix`, the barcode and `barcode_suffix`, matched exactly. `barcode_pattern` is a regular expression whose named group `barcode` keeps the part of each barcode that names the cell, found as `re.search` finds it; a barcode that the pattern does not match names no cell. Each option takes one value for every file or one per file.

```python
# STICR names cells "<sample>_<barcode>-1"; a store of one sample names them by barcode.
clones = ds.clones.import_clones(
    "meta.tsv",
    cell_selection=sample_cells,
    barcode_pattern=r"^GW15_GZ_6wk_Rep1_1_(?P<barcode>[ACGT]{16}-1)$",
)
```

Barcodes that match no selected cell, such as droplets that cell calling rejected or cells of another sample, are counted by the cell ID they name and saved in the artifact's `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` attributes. A warning names up to five of them, and an import that reuses the saved clones gives the warning again. {meth}`ds.perturbation.import_guide_calls <cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls>` joins barcodes by the same rules.

A file is identified by the SHA-256 of its bytes, and a clone matrix also by its lists, so a moved file reuses the saved clones, an edited one gives new clones, and files listed in another order give the same clones. A file listed twice with the same prefix and suffix raises `ValueError`.

### Several files

Clone IDs are unique only within one run of a tool: TREX numbers its clones from 1 in every run. With several files, each file's labels therefore start with its `barcode_prefix`, or, without one, with its name before the first `.` and `_`, so equal IDs of two files stay two clones. Two files with the same namespace raise `ValueError`, as two TREX runs that both write `clones.txt` do without prefixes.

```python
# One TREX run per sample, named by bare 10x barcodes, in a merged store.
clones = ds.clones.import_clones(
    ["d2/clones.txt", "d4/clones.txt"],
    cell_selection=cells,
    barcode_prefix=["d2__", "d4__"],
    barcode_suffix="-1",
)
```

### Cells in several clones

A cell listed in two different clones, in one file or in two, raises `ValueError` that names a few such cells and their clones. `multiple="drop"` leaves such cells without a clone and counts them in the `n_cells_in_several_clones` attribute. Rows that repeat a cell's clone, as Cassiopeia lists a cell once per target site, count the cell once. A selected cell that no file lists has no clone.

## Clones from a cell metadata column

`from_column` saves the clones that a cell metadata column assigns, such as a column imported with an AnnData file. A missing or blank label means no clone. Integer labels read as their text, and floating-point whole numbers, which pandas writes for integer labels beside missing ones, read as integers; other floating-point labels raise `TypeError`. The artifact records the fingerprint of the labels, so editing the column gives new clones.

```python
clones = ds.clones.from_column("clone_id", cell_selection=cells)
```

## Read, export and plot clones

A `clones` artifact labels each selected cell with its clone, an empty label for a cell without one, which groupings treat as missing. `codes` numbers the clones by size, largest first, then by label in natural order, -1 for none, and `clone_size` holds the cells of each cell's clone, 0 for none. A source that gives probabilities adds `clone_probability`, NaN where none is given. `ds.select_cells(clones, include=["3"])` selects the cells of one clone, and plots colour cells by clone. [Read per-cell values](cell_aligned_kinds) lists the arrays.

`load_clones` returns a {class}`~cytearc.clones.Clones` record. Its `cells` table holds one row per cell, indexed by cell ID: `clone`, a categorical of the clone labels in rank order, `clone_size` and, where saved, `clone_probability`. Its `clones` table holds one row per clone, largest first: `clone` and `n_cells`. Its `source` names the operation that saved the clones, `"import_clones"` or `"clones_from_column"`.

`export_clones(clones, path)` writes CoSpar's clone table, `Cell_ID,Clone_ID`, one row per cell with a clone, and `format="matrix"` writes a cells by clones Matrix Market file with `cells.tsv` and `clones.tsv` beside it, replacing files of those names. Both hold this store's cell IDs and the clone labels, and `import_clones` reads them back to the same clones; clone probabilities are not written. A path ending in `.gz` is gzip-compressed. Each returns the number of cells written.

```python
ds.clones.export_clones(clones, "clones.csv")  # for CoSpar
ds.clones.export_clones(clones, "clones/matrix.mtx", format="matrix")
```

`ds.plots.clone_abundance(clones)` draws two panels: the number of clones of each size, on log scales, and the `top` largest clones (default 20). `group_by` names a cell metadata column, such as the sample or time point, and `groups` a cell-label artifact, such as a clustering; with either, the first panel draws one line per group, counting each clone's cells within the group, and the second stacks each clone's cells by group. Cells without a group are left out, and the plot raises `ValueError` when no cell with a group has a clone. Its tables are `sizes` (`group`, `clone_size`, `n_clones`) and `top` (`clone`, then the cells of each group, or `cells`).

## Clones

```{eval-rst}
.. autoclass:: cytearc.clones.Clones
```
