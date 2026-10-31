---
myst:
  html_meta:
    description: CyteArc clones API for lineage tracing, with clones imported from CoSpar, TREX, CellTag-multi, CellTagR, STICR, Cassiopeia, Numbat, MiTo and LARRY's clone matrices or from a cell metadata column, joined to cells by barcode prefix, suffix or pattern, saved as a clones artifact, read as tables of cells and clones, exported as CoSpar's clone table or a Matrix Market file, drawn as clone sizes, measured within groups of cells, immune receptor clonotypes among them, as clonal expansion, diversity indices and Hill numbers with rarefaction and extrapolation, compared between groups by the clones they share, with chosen clones followed across groups, and measured as fate coupling between groups against shuffled labels and as the fate bias of early cells, with CoSpar's definitions.
---

# Clones API reference

A clone is the set of cells that descend from one ancestor cell. Lineage tracing methods mark the ancestor with a heritable barcode, such as LARRY's, CellTag's or TREX's expressed barcodes, and other methods infer clones from the cells' own mutations, such as copy-number changes or mitochondrial variants. Each tool writes which clone each cell belongs to in its own format. `ds.clones` reads those files, joins their cells to a store's cells, and saves the clones as a `clones` artifact that CyteArc's clone analyses read whatever defined the clones. The T cell and B cell clonotypes of {meth}`ds.repertoire.clonotypes <cytearc.datastore.namespaces.RepertoireAccessor.clonotypes>` are clones too: every function on this page reads a `clonotypes` artifact as it reads `clones`, and measures each receptor type apart, as the {doc}`repertoire` describes.

{meth}`ds.clones.import_clones <cytearc.datastore.namespaces.ClonesAccessor.import_clones>` imports the clones of one file or several, and {meth}`ds.clones.from_column <cytearc.datastore.namespaces.ClonesAccessor.from_column>` takes them from a cell metadata column. {meth}`ds.clones.load_clones <cytearc.datastore.namespaces.ClonesAccessor.load_clones>` reads the artifact as a {class}`~cytearc.clones.Clones` record, {meth}`ds.clones.export_clones <cytearc.datastore.namespaces.ClonesAccessor.export_clones>` writes it as a file that other tools read, and {meth}`ds.plots.clone_abundance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_abundance>` draws how many clones have each size. {meth}`ds.clones.expansion <cytearc.datastore.namespaces.ClonesAccessor.expansion>` places each cell in an expansion category by its clone's size within its sample or other group, {meth}`ds.clones.diversity <cytearc.datastore.namespaces.ClonesAccessor.diversity>` and {meth}`ds.clones.rarefaction <cytearc.datastore.namespaces.ClonesAccessor.rarefaction>` measure how many clones each group holds and how evenly its cells spread over them, {meth}`ds.clones.overlap <cytearc.datastore.namespaces.ClonesAccessor.overlap>` and {meth}`ds.clones.track <cytearc.datastore.namespaces.ClonesAccessor.track>` compare groups by the clones they share and follow chosen clones across them, {meth}`ds.clones.coupling <cytearc.datastore.namespaces.ClonesAccessor.coupling>` measures how strongly groups share clones against shuffled labels, and {meth}`ds.clones.fate_bias <cytearc.datastore.namespaces.ClonesAccessor.fate_bias>` follows each early cell's clone to its later cells and tests which fates each clone took.

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
| `"cassiopeia"` | [Cassiopeia](https://github.com/YosefLab/Cassiopeia)'s allele table, one row per cell and target site | `cellBC`, `lineageGrp`; group 0 means no clone, and a group written as `4.0` is group `4` |
| `"numbat"` | [Numbat](https://github.com/kharchenkolab/numbat)'s `clone_post_{i}.tsv` | `cell`, `clone_opt`, and `p_opt` as each cell's `clone_probability` |
| `"table"` | any delimited table | the columns that `columns` names |
| `"matrix"` | a Matrix Market clone matrix with its cell list beside it, such as LARRY's on [GEO GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802) | the cell list's columns that `columns["cell"]` names |
| `"cospar_h5ad"` | an AnnData file with CoSpar's `obsm['X_clone']` | the cells by clones matrix |
| `"mito"` | an AnnData file with [MiTo](https://github.com/andrecossa5/MiTo)'s `obs['MiTo_clone']` | `"unassigned"` means no clone |
| `"h5ad"` | an AnnData file | the `obs` column that `columns["clone"]` names |

`columns` maps `"cell"`, `"clone"` and `"probability"` to column names. `"cell"` may list several columns, whose values joined by `"__"` form the barcode, as a merged store joins a sample and a barcode. A table whose header holds one field fewer than its rows, as R's `write.table` writes it, takes each cell from its row names, as does one whose first column has an empty name, as R's `write.csv` writes row names, so a wide table with one clone column per labelling round, such as the `celltag_data.txt` of the Biddy et al. workflow, needs only `columns={"clone": "CellTagD13"}` to read one round. Clone labels are text as the file writes them; an empty field, `NA`, `NaN` or `nan` means no clone.

Tables may be plain or gzip-compressed, whatever their names, separated by tabs, commas or spaces, quoted as R quotes them, and without a final newline. A header that starts with `#`, as TREX writes, is a header, and blank lines hold no rows. A delimiter after the last field of every row also gives a row one field more than its header, an empty one, so a table whose every row ends in its delimiter with nothing after it is refused rather than read either way: name the row names in the header, as R's `write.table(..., col.names=NA)` writes them, or end the header with the delimiter too. A row with more fields than the table's first row raises `ValueError`, and errors name the lines of the file.

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

`from_column` saves the clones that a cell metadata column assigns, such as a column imported with an AnnData file. A missing or blank label means no clone. Integer labels read as their text, and floating-point whole numbers, which pandas writes for integer labels beside missing ones, read as integers; other floating-point labels raise `TypeError`. The artifact records a fingerprint of the column, so editing it gives new clones.

```python
clones = ds.clones.from_column("clone_id", cell_selection=cells)
```

## Read, export and plot clones

A `clones` artifact labels each selected cell with its clone, an empty label for a cell without one, which groupings treat as missing. `codes` numbers the clones by size, largest first, then by label in natural order, -1 for none, and `clone_size` holds the cells of each cell's clone, 0 for none. A source that gives probabilities adds `clone_probability`, NaN where none is given; a cell's rows that give it no probability leave the one its other rows give. The store keeps each cell's label at the width of the longest label, so a very long label costs memory for every cell, and an import whose labels would not fit the memory budget raises `MemoryError` before it saves anything. `ds.select_cells(clones, include=["3"])` selects the cells of one clone, and plots colour cells by clone. [Read per-cell values](cell_aligned_kinds) lists the arrays.

`load_clones` returns a {class}`~cytearc.clones.Clones` record. Its `cells` table holds one row per cell, indexed by cell ID: `clone`, a categorical of the clone labels in rank order, `clone_size` and, where saved, `clone_probability`. Its `clones` table holds one row per clone, largest first: `clone` and `n_cells`. Its `source` names the operation that saved the clones, `"import_clones"`, `"clones_from_column"` or, for clonotypes, `"define_clonotypes"`. Its `kept_apart` names the columns of `cells` whose labels the clones keep apart, which every summary measures apart: none for clones, and `("receptor_type",)` for clonotypes, whose `cells` table then holds each cell's receptor type as a categorical.

`export_clones(clones, path)` writes CoSpar's clone table, `Cell_ID,Clone_ID`, one row per cell with a clone, and `format="matrix"` writes a cells by clones Matrix Market file with `cells.tsv` and `clones.tsv` beside it, replacing files of those names. Both hold this store's cell IDs and the clone labels, and `import_clones` reads them back to the same clones; clone probabilities are not written. The table refuses labels that a clone table reads as no clone, such as `NA`, which the matrix keeps, and the matrix refuses labels and cell IDs that hold a line break, which the table keeps. A path ending in `.gz` is gzip-compressed. Each returns the number of cells written.

```python
ds.clones.export_clones(clones, "clones.csv")  # for CoSpar
ds.clones.export_clones(clones, "clones/matrix.mtx", format="matrix")
```

`ds.plots.clone_abundance(clones)` draws two panels: the number of clones of each size, on log scales, and the `top` largest clones (default 20). `group_by` names a cell metadata column, such as the sample or time point, and `groups` a cell-label artifact, such as a clustering; with either, the first panel draws one line per group, counting each clone's cells within the group, and the second stacks each clone's cells by group. Cells without a group are left out, and the plot raises `ValueError` when no cell with a group has a clone. Clonotypes are counted within each receptor type, so their groups are the receptor types, or each group's receptor types, named like `"s1, TCR"`. Its tables are `sizes` (`group`, `clone_size`, `n_clones`) and `top` (`clone`, then the cells of each group, or `cells`).

## Measure clonal expansion

A clone that holds more than one cell has expanded: its founding cell divided. {meth}`ds.clones.expansion <cytearc.datastore.namespaces.ClonesAccessor.expansion>` counts each clone's cells within groups of cells and places each cell in an expansion category by its clone's size in its group. The groups are the combinations of the labels of the `group_by` cell metadata columns, such as the sample, and of a `groups` label artifact, such as a clustering. Without a grouping, a clone's cells count in every sample together, as scirpy's `clonal_expansion` counts them without `expanded_in`. A cell without a clone or a label is counted in no group, and when no measured cell has both, every summary raises `ValueError` and saves nothing.

```python
expansion = ds.clones.expansion(clones, group_by="sample")
table = ds.clones.load_expansion(expansion)
table.categories  # cells and clones of each sample in each category
large = ds.select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])
ds.plots.clonal_expansion(expansion, group_by="cell_type")
```

One of two kinds of bounds sets the categories:

| Option | Categories | As in |
|---|---|---|
| `breakpoints`, the default `(1, 2, 5, 20, 100)` | By the clone's cells in the group: `"1 cell"`, `"2 cells"`, `"3 to 5 cells"`, `"6 to 20 cells"`, `"21 to 100 cells"` and `"more than 100 cells"` | scirpy's `breakpoints`, whose categories are `"<= 1"`, `"<= 2"`, `"<= 5"` and so on |
| `frequency_breakpoints`, such as `(0.0001, 0.001, 0.01, 0.1, 1)` | By the clone's frequency, its share of the group's cells with a clone, in bins `(lower, upper]` that end at 1: `"0 to 0.0001"`, `"0.0001 to 0.001"`, `"0.001 to 0.01"`, `"0.01 to 0.1"` and `"0.1 to 1"` | scRepertoire's `clonalHomeostasis` and `cloneSize`, which name these bins Rare, Small, Medium, Large and Hyperexpanded |

A frequency depends on the size of its group: in a sample of 1,300 cells a clone of one cell already has the frequency 0.00077, so the first bin of these bounds stays empty, and a clone of two cells is Medium. Compare frequencies between groups of similar sizes, or use bounds in cells. The names of bounds in cells sort in their order, so groupings and legends that sort labels naturally keep them from the smallest clones to the largest. Frequency bounds are named with six significant digits, or with as many more as tell every bound apart, such as `"0.1234561 to 0.1234562"`.

The artifact labels the cells of `cell_selection`, or the clones' cells without one; a selected cell that the clones do not hold has no clone. Its canonical `values` hold each cell's category, empty for a cell without a clone or a group, so `select_cells` keeps the cells of large clones and `ds.plots.embedding(layout=umap, color_by=expansion)` colours cells by category. `group_clone_size`, the cells of the cell's clone in its group, `group_clone_frequency`, their share of the group's cells with a clone, and `group_codes`, the cell's group, are its other per-cell arrays. The result records the clones, the cells, the label artifact, the bounds and, with `group_by`, each cell's group, so measuring again with the same values returns the saved result. The measurement holds about 300 bytes per cell and 2,000 bytes per group at its peak, beside the clones and labels that it reads.

`load_expansion` returns a {class}`~cytearc.clones.ClonalExpansion` record with three tables, whose groups are keyed by the `group_by` columns, by the kind of the `groups` artifact, such as `cluster_labels`, and for clonotypes by `receptor_type`, and named in `group` by their labels joined by commas, or `"all cells"`:

| Table | Rows | Columns |
|---|---|---|
| `cells` | One per cell | `clone`, `category`, `group`, the group's key, `group_clone_size`, `group_clone_frequency` |
| `categories` | One per group and category, empty categories included | `group`, the group's key, `category`, `cells`, `clones`, and `fraction`, the share of the group's cells with a clone that the category holds |
| `rank_windows` | One per group and window of ranks | `group`, the group's key, `window`, `cells`, `fraction` |

The labels of the `cells` table are pandas categoricals, coded by the saved codes rather than held as text per cell; `category` is ordered, smallest clones first, and the labels of each key column follow their natural order, as the groups do. Groups need one name each, so labels that hold commas and give two groups one name are refused when the expansion is measured, and a column of labels may not share the name of a column of the tables.

The rank windows describe a group's clonal space as scRepertoire's `clonalProportion` does: the group's clones are ranked by their cells in the group, and each window counts the cells of the clones ranked in it. `splits` sets the windows; the default `(10, 100, 1000, 10000, 30000, 100000)` gives scRepertoire's windows `"1 to 10"` to `"30001 to 100000"`, and `"after 100000"` holds the cells that scRepertoire leaves out. Tied clones hold the same cells, so ties do not change the counts. A sample whose ten largest clones hold most of its cells is dominated by a few clones.

{meth}`ds.plots.clonal_expansion <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonal_expansion>` draws one stacked bar per group of the share of its cells in each category, or of its clones with `count="clones"`, as scirpy's `summarize_clonal_expansion` counts them, and one stacked bar per group of the share of its cells in each rank window, leaving out windows without cells. The groups are the expansion's own unless `group_by` names a cell metadata column or `groups` a cell-label artifact: each cell keeps the category measured within its sample, so grouping by cell type shows which cell types hold the expanded cells of their samples, and the ranks are counted within the plotted groups. Labels that the clones keep apart stay apart, so the plotted groups of clonotypes are each group's receptor types, such as `"blood, TCR"`. `normalize=False` draws counts. Its tables are `categories` and `rank_windows`.

The categories were validated on the clonotypes of two 10x Genomics immune profiling runs, "Human PBMC from a Healthy Donor, 10k cells (v2)" (6,189 cells with a receptor) and "CD8+ T cells of Healthy Donor 1" (46,526 cells), whose expanded CD8+ T cell clones fill every category. Each run holds one donor, so four pseudo samples and six pseudo clusters were assigned from a hash of each barcode. With clonotypes equal to scirpy's, the category of every cell equals that of scirpy 0.25.1's `clonal_expansion`, within the pseudo samples and across all cells, with the default bounds and with `(1, 2)`, and `summarize_clonal_expansion` by pseudo cluster, counting cells and clones, equals the plot's tables. On the T cell clonotypes of the PBMC run, which equal scRepertoire's `CTstrict` and `CTaa` keys, the fractions of `frequency_breakpoints=(0.0001, 0.001, 0.01, 0.1, 1)` in the four pseudo samples equal scRepertoire 2.8.0's `clonalHomeostasis` within 4e-16, and the cells of the rank windows equal its `clonalProportion`. On a million simulated cells in eight samples, 899,879 of them in 314,595 clones of up to 465 cells, measuring takes about 0.7 s and holds 220 MiB of arrays, about 230 bytes per cell, and loading the tables takes about 0.7 s and 290 MiB.

## Measure diversity

The diversity of a population of cells is how many clones it holds and how evenly its cells spread over them. {meth}`ds.clones.diversity <cytearc.datastore.namespaces.ClonesAccessor.diversity>` measures it for groups of cells: the combinations of the labels of the `group_by` cell metadata columns, such as the sample, and of a `groups` label artifact, such as a clustering. `cell_selection` keeps only some cells, such as one cell type. Cells without a clone or a label are left out, and a clone's size in a group is its cells there. Nothing is saved: the result follows from the clones and the labels in about 0.3 s for a million cells.

```python
result = ds.clones.diversity(clones, group_by="sample")
result.indices  # one row per sample: Shannon entropy, Simpson indices, Chao1, ...
result.hill  # Hill numbers: observed, asymptotic, at a common size and coverage
curves = ds.clones.rarefaction(clones, group_by="sample")
ds.plots.clonal_rarefaction(clones, group_by="sample")
```

The `indices` table holds one row per group: `group`, its labels joined by commas (`"all cells"` without labels), each label in its key column, and, for its `n` cells and `S` clones of frequencies `p`, `f1` of one cell and `f2` of two:

| Column | Definition | As in |
|---|---|---|
| `cells`, `clones`, `singletons`, `doubletons` | `n`, `S`, `f1`, `f2` | |
| `coverage` | The sample coverage, the estimated share of the population's cells whose clone the sample holds: `1 - f1 / n * (n - 1) f1 / ((n - 1) f1 + 2 f2)`, or with `f1 - 1` and `f2 = 1` without doubletons ([Chao and Jost 2012](https://doi.org/10.1890/11-1952.1)) | iNEXT's `SC` |
| `shannon` | `-sum(p log p)`, in nats | scRepertoire's `shannon` |
| `pielou` | Shannon entropy over its maximum, `log S`; NaN for one clone | scRepertoire's `pielou` and `norm.entropy` |
| `simpson`, `inverse_simpson`, `gini_simpson` | `sum(p**2)`, its inverse, and one minus it | scRepertoire's and immunarch's `inv.simpson` and `gini.simpson` |
| `gini` | The Gini coefficient of the clone sizes, 0 when all are equal | scRepertoire's and immunarch's `gini` |
| `d50` | The fewest clones, largest first, that hold at least half the cells | scRepertoire's and immunarch's `d50` |
| `chao1` | `S + (n - 1) / n * f1**2 / (2 f2)`, or `S + (n - 1) / n * f1 (f1 - 1) / 2` without doubletons | iNEXT's and immunarch's Chao1 |
| `chao1_bias_corrected` | `S + f1 (f1 - 1) / (2 (f2 + 1))` | vegan's `estimateR` and scRepertoire's `chao1` |
| `ace` | The abundance-based coverage estimate with clones of at most 10 cells rare ([Chao and Lee 1992](https://doi.org/10.1080/01621459.1992.10475194)); NaN when every rare clone is a singleton | vegan's `estimateR` and scRepertoire's `ace` |

Every observed index depends on how many cells a group holds: a larger sample finds more clones. Hill numbers ([Hill 1973](https://doi.org/10.2307/1934352)) put the indices on one scale, the effective number of clones: order 0 counts clones, order 1, the exponential of Shannon entropy, weighs them by frequency, and order 2, the inverse Simpson index, counts the dominant ones. The `hill` table holds, for each group, named in `group` and the key columns, and order 0, 1 and 2, the `observed` Hill number; the `estimated` asymptotic one of the whole population, Chao1 for order 0, the exponential of the entropy estimate of [Chao, Wang and Jost (2013)](https://doi.org/10.1111/2041-210X.12108) for order 1, and the inverse of Good's unbiased Simpson estimate for order 2, infinite when no clone holds two cells; the Hill number `at_size`, the common size, with its `coverage_at_size` and `method_at_size`; and the Hill number `at_coverage`, the common coverage, with its `size_at_coverage` and `method_at_coverage`. Below a group's size its Hill numbers are rarefied, the expected values of a random subsample, and above it extrapolated, as [Chao et al. (2014)](https://doi.org/10.1890/13-0133.1) derive, with the Shannon extrapolation of [Chao et al. (2023)](https://doi.org/10.1002/ecm.1588) that iNEXT 3.0.2 uses: the curve approaches the asymptote geometrically from the observed value and passes through the rarefied value one cell earlier. A size between whole numbers interpolates linearly, as iNEXT does. By default the common size is the smallest of the groups' doubled sizes and the common coverage the lowest of their coverages at doubled sizes, iNEXT's `estimateD` defaults, so no group is extrapolated beyond twice its size; `size` and `coverage` set them. Compare groups by the Hill numbers at a common coverage, or at a common size; a group far smaller than the others sets both defaults low, so pass `size` or `coverage`, or leave it out. When every group's coverage is 1, as when no clone of a group holds a single cell, the common coverage is 1, which each group reaches at its observed size, so the Hill numbers at it are the observed ones.

{meth}`ds.clones.rarefaction <cytearc.datastore.namespaces.ClonesAccessor.rarefaction>` computes each group's curves: one row per group, order and size, with `group` and the key columns, the `size`, `method` (`"rarefaction"`, `"observed"` or `"extrapolation"`), `diversity` and `coverage`. By default each group gets iNEXT's 40 sizes, about half evenly from 1 cell to its size and half from there to twice its size; `endpoint` and `knots` change them, and `sizes` gives every group the same sizes. Compiled loops rarefy each group to all its sizes at once, for every order and the coverage. {meth}`ds.plots.clonal_rarefaction <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonal_rarefaction>` draws one panel per order, the Hill number against the cells (`base="size"`) or against the sample coverage (`base="coverage"`), solid where rarefied, dashed where extrapolated, with a point at each group's observed size, and with `base="size"` a panel of each group's coverage against its size. A curve that still rises steeply at its observed point has not reached its plateau: more cells would find more clones. iNEXT's bootstrap confidence intervals are not computed; compare conditions with each sample's estimate as one observation, in a test that treats samples as replicates, such as those of [CytePool](https://docs.nygen.io/CytePool/).

The implementation follows the papers rather than the code of iNEXT, which is licensed under the GPL, and validates against its outputs. On two small abundance vectors, the clonotypes of the four pseudo samples of both 10x runs above (11,523 to 11,745 cells each in the CD8+ run), all 46,526 CD8+ cells together, and six edge cases (no singletons, no doubletons, every clone a singleton, a lone singleton, one clone, two cells), the Hill numbers of iNEXT 3.0.2's curves equal CyteArc's within 1.4e-12 at every size, rarefied, observed and extrapolated, the coverages within 3e-9, the asymptotic estimates within 7e-13, and `estimateD` by size within 5e-13 and by coverage within 1e-7, the tolerance of iNEXT's root finding; the default sizes are iNEXT's. Where the two differ, CyteArc's rarefied coverage equals exact rational arithmetic within 1e-16, where iNEXT's carries errors up to 3e-12, because the chance of missing a clone is computed as a product of ratios rather than from log-gamma functions. The indices equal scRepertoire 2.8.0's `clonalDiversity` without bootstraps for all twelve of its metrics, immunarch 0.10.3's `repDiversity` (`chao1`, `d50`, `div`, `gini`, `gini.simp`, `inv.simp` and Hill numbers of orders 1 to 6) and vegan 2.7.6's `estimateR` (`S.chao1`, `S.ACE`), within 6e-14, on the same pseudo samples and edge cases. On the same million simulated cells in eight samples, `diversity` takes about 0.3 s and `rarefaction` 0.7 s, holding at most 115 MiB of arrays. Split into 2,000 groups of about 450 cells, such as samples by clusters, the cells take 1.9 s for `diversity` and 1.3 s for `rarefaction`.

### Memory

Every summary checks the memory budget before each step that reads or allocates, from the sizes and widths that the stored arrays declare, and raises `MemoryError` before it reads what would not fit or writes anything. The steps hold about:

| Step | Bytes |
|---|---|
| Reading the clones | 12 per cell of the clones, and for `track` and the tables of `load_expansion` 100 and three times the stored width of each clone's label; for clonotypes also the stored width of each cell's receptor type and 108 more |
| Matching the clones to other cells, when `cell_selection` of `expansion` differs from the clones' cells | 56 per cell of the clones and 32 per measured cell, 8 per measured cell for each column of labels that the clones keep apart, and the read before, below |
| Grouping | 100 per cell, 70 per cell for each key column, and three times the stored width of each `group_by` column's labels; 32 per cell of `cell_selection`, with the read before, and 130 and three times the label width per cell of the `groups` artifact, whatever cells the clones hold; one chunk of the widest column of labels, and that chunk again with the read before when it reads several columns and labels or a column of several chunks; 500 per distinct label and per group |
| Measuring expansion | 300 per measured cell, and 8 for each column of labels that the clones keep apart |
| Loading expansion | 500 per cell and the cells' IDs as text |
| `load_clones` and `export_clones` | 250 and 200 per cell, the clones read without their labels, and the clone labels and the cells' IDs as text |
| Rarefaction | 40 for each size below a group's cells and each of its distinct clone sizes, and 100 per row of the curves |
| Comparing groups | 250 per pair of groups and 24 per entry of the groups-by-groups matrices, 40 per clone of each group, and the clones read |
| Following clones | 100 per row of the table and 16 more per row for each key column, a byte per chosen clone and group, 40 per clone of each group, and the clones and labels read |
| Measuring coupling | 250 per pair of groups, 40 per product of two groups, 80 per clone of each group and 12 per cell with a clone; for each thread of the null 4 per cell with a clone, 8 per product of two groups and 32 per group; 8 per pair for each permutation measured at once; 40 per clone of each group and the clones read |
| Checking that no cell is early and late | 40 per cell of the two selections, and the clones read |
| Measuring fate bias | 64 per early cell and 24 more for each fate, 40 and four times the longest fate name's characters to write each early cell's fate, 80 per fate of each clone with late cells, 1,000 per fate, 24 per clone of each fate, and the clones read |
| Loading fate bias | 200 per early cell and 24 per fraction, 160 per row of clone and fate and 16 more per key column, the early cells' IDs as text, and the clones and labels read |
| Each summary | 2,000 per group |

A grouping with a label for each cell therefore needs far more memory than its cells. A step that reads two cell selections one after the other, the measured cells and the clones' cells, also counts the read before: the thread of Zarr's pool that decoded a stored chunk lets go of it only when that thread next runs, which on a busy machine can come after the next read has begun, so the largest stored chunk of the selections and the clones' codes, as stored and decoded, and 64 KiB for the objects of both reads are held beside the next read. Grouping reads its columns a stored chunk at a time, so when it reads more than one chunk, of several columns and labels or of a column stored in several chunks, it counts the largest chunk again for the read before. A selection stores a flag for each cell of the store, so its chunk can outweigh the cells it selects. Reading cell IDs as text holds one chunk of the stored IDs, by default 100,000 of them, both decoded and as stored, and for each cell three times the stored width of an ID, 128 bytes of positions, and a string of 64 bytes and four per character. Coupling measures its null on as many threads as `nthreads` allows, with up to 16 permutations at once on each thread within 64 MiB, and on fewer threads and permutations at once when the budget is tight; it refuses only when one thread measuring one permutation does not fit, and its permutations depend on neither. Peaks that tracemalloc recorded stayed within these budgets on long and emoji labels, long clone labels, a few measured cells of many clone cells, label artifacts and selections of more cells than the clones hold, a label for each cell, many pairs of groups, clones of one cell each, 13 key columns, and long clone labels held beside many rows.

## Compare groups

Two samples of one animal share the clones that both sampled, such as two time points or two tissues, and two cell types share the clones whose cells took both fates. {meth}`ds.clones.overlap <cytearc.datastore.namespaces.ClonesAccessor.overlap>` compares every pair of groups by their clones. The groups are the combinations of the labels of `group_by` cell metadata columns and of a `groups` label artifact, among the cells of `cell_selection`, as for diversity. Cells without a clone or a label are left out, and at least two groups must hold a cell with a clone. Nothing is saved.

```python
result = ds.clones.overlap(clones, group_by="sample")
result.pairs  # one row per pair of samples
ds.plots.clonal_overlap(clones, group_by="cell_type")
tracked = ds.clones.track(clones, group_by="time_point", top=10)
ds.plots.clone_tracking(clones, group_by="time_point", order=["day 2", "day 4", "day 6"])
```

The `pairs` table holds one row per pair, `group_1` and `group_2` named by their labels joined by commas, with these columns, for groups of `S1` and `S2` clones and clone sizes `x` and `y` of `X` and `Y` cells:

| Column | Definition | As in |
|---|---|---|
| `shared` | The clones of both groups | scRepertoire's `raw`, immunarch's `public` |
| `jaccard` | `shared / (S1 + S2 - shared)` | scRepertoire's and immunarch's `jaccard` |
| `sorensen` | `2 shared / (S1 + S2)` | immunarch's `tversky` with its default weights |
| `overlap` | `shared / min(S1, S2)` | scRepertoire's and immunarch's `overlap` |
| `cosine` | `shared / sqrt(S1 S2)`, the cosine of presence | scRepertoire's `cosine` |
| `morisita_horn` | `2 sum(x y) / ((sum(x^2) / X^2 + sum(y^2) / Y^2) X Y)` ([Horn 1966](https://doi.org/10.1086/282436)) | scRepertoire's and immunarch's `morisita` |
| `abundance_cosine` | `sum(x y) / sqrt(sum(x^2) sum(y^2))`, the cosine of the clone sizes | immunarch's `cosine` |
| `shared_fraction_1`, `shared_fraction_2` | The share of each group's cells whose clone the other group holds | |

The first five count clones and grow with the cells of both groups, since larger samples find more of the rare clones that two samples share by chance; the Morisita-Horn index and the cosine of clone sizes weigh clones by their cells, depend little on sample size, and come near 1 when the same expanded clones dominate both groups. `group_sizes` holds each group's cells with a clone and its clones; `cytearc.clones.OVERLAP_INDICES` names the indices. A comparison that the memory budget cannot hold, as the memory table above counts it, raises a `MemoryError`; fewer groups, from fewer `group_by` columns or a `cell_selection`, avoid it. {meth}`ds.plots.clonal_overlap <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonal_overlap>` draws one `index` as a heatmap of the pairs, `"morisita_horn"` by default, on a figure at most a page wide, with values printed for at most 12 groups and group names for at most 40; its tables are `matrix` and `pairs`, which name every group.

{meth}`ds.clones.track <cytearc.datastore.namespaces.ClonesAccessor.track>` follows clones across groups, such as time points: by default the `top` (10) with the most cells in each group, together, as scRepertoire's `clonalCompare` chooses them, with ties going to the larger clone overall; `labels` names them instead. Its `cells` table holds each chosen clone's cells and share of the cells with a clone of every group, zeros included, so a clone that expands between time points or spreads between cell types shows it. A table that the memory budget cannot hold raises a `MemoryError`, which a smaller `top` or fewer groups avoid. {meth}`ds.plots.clone_tracking <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_tracking>` draws one line per clone, labelled with its ID, across the groups, in their natural order or in `order`, which may name time points as numbers; on a figure at most a page wide, its legend lists at most 80 clones, the largest first, and its axis names at most 40 groups; its table is `cells`.

The indices were validated on the clonotypes of the two 10x runs above. On the four pseudo samples of CD8+ donor 1, which share about 500 clonotypes per pair and whose expanded clones give a Morisita-Horn index of 0.994 to 0.997, and on PBMC 10k with clonotypes equal to scRepertoire's `CTstrict` and `CTaa`, every index equals immunarch 0.10.3's `repOverlap` and scRepertoire 2.8.0's `clonalOverlap` within 5e-14, for every pair. scRepertoire's help page gives abundance-weighted formulas for its `overlap`, `raw` and `cosine`, but its outputs follow the presence formulas above, which CyteArc reproduces. On the same million simulated cells in eight samples, `overlap` takes about 0.5 s and `track` 0.6 s, each holding at most 130 MiB of arrays. Split into 2,000 groups of samples by clusters, the cells take 1.9 s and 480 MiB for `overlap`, which compares 2 million pairs, and 1.1 s and 145 MiB for following the largest clone of each group, 546,000 rows. Each ran within the smallest memory budget it accepts: 187 MiB by sample, and 617 MiB and 302 MiB for the 2,000 groups.

## Measure fate coupling

Cell types whose cells often share a clone descend from common progenitors, so the clones that two groups share measure how closely their fates are coupled. How much two groups share also grows with their sizes and with how far their clones expanded, so a coupling is read against the coupling that the same cells give when clone and fate are independent. {meth}`ds.clones.coupling <cytearc.datastore.namespaces.ClonesAccessor.coupling>` measures every pair of groups, the combinations of the labels of `group_by` columns and of a `groups` label artifact among the cells of `cell_selection`, such as the cell types of one time point, and returns a {class}`~cytearc.clones.CloneCoupling` record. Cells without a clone or a label are left out, at least two groups must hold a cell with a clone, and nothing is saved.

```python
late = ds.snapshot_cell_selection("is_day_6")
coupling = ds.clones.coupling(clones, group_by="cell_type", cell_selection=late)
coupling.pairs.sort_values("z_score", ascending=False).head()  # the most coupled pairs
ds.plots.clone_coupling(coupling)
```

`method` names how a pair's coupling is measured from the cells `x[g, c]` of each clone `c` in each group `g`, of `n[g]` cells with a clone, as CoSpar's `fate_coupling` measures it from a clone matrix:

| `method` | Coupling of two groups | CoSpar 0.5.0's `fate_coupling(source="X_clone")` |
|---|---|---|
| `"share_cosine"`, the default | The cosine over the clones of the two groups' shares `q[g, c] = (x[g, c] / n[g]) / sum_h(x[h, c] / n[h])`: each group's frequency of the clone over the clone's frequencies in all groups, so every clone counts once whatever its size | `method="SW"` with `normalize=True`, its defaults |
| `"abundance_cosine"` | The cosine of the two groups' clone sizes, `sum(x y) / sqrt(sum(x^2) sum(y^2))`, which `overlap` also reports | `method="SW"` with `normalize=False` |
| `"normalized_covariance"` | The covariance over the clones of the two groups' cells, with `C - 1` degrees of freedom for `C` clones, over the product of their mean cells per clone; below 0 when the groups share fewer clones than independent fates would | `method="Weinreb"` with `normalize=False` |
| `"jaccard"` | The clones of both groups over the clones of either, which `overlap` also reports | `method="Jaccard"` |

Each coupling is compared with a null: the group labels of the cells with a clone are shuffled `permutations` times, by default 1,000, which keeps each clone's cells and each group's cells, as CoSpar's `pvalue_for_fate_coupling` shuffles them. The `pairs` table holds, for each pair of groups `group_1` and `group_2`, its `coupling`, the `null_mean` and `null_sd` of its shuffled couplings, with `permutations - 1` degrees of freedom, and the `z_score`, `(coupling - null_mean) / null_sd`, which puts pairs of large and small groups on one scale. A null without spread gives a z-score of NaN, or an infinite one when the coupling differs from it, and `permutations=0` measures no null. Permutation `k` draws from a random stream keyed by `seed`, a non-negative integer below 2**64, and `k`, so the result is the same whatever `nthreads` and memory budget measure it. Shuffled labels spread each clone over the groups at random, so when clones commit to few fates, as lineage tracing finds, most pairs fall below their null, and the order of the z-scores, the most coupled pairs highest, says more than their sign. `group_sizes` holds each group's cells with a clone and its clones, and `cytearc.clones.COUPLING_METHODS` names the methods.

`order` lists the groups in the leaf order of average linkage on the coupling, which joins first the two clusters of groups whose average coupling is highest, and `linkage` holds that tree as SciPy's linkage matrix. {meth}`ds.plots.clone_coupling <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_coupling>` draws the z-scores, or the couplings with `value="coupling"`, as a heatmap in that order with the tree above it, on a scale centred on 0 for z-scores and for couplings below 0; a coupling measured without permutations has no z-scores, so it draws only with `value="coupling"`. It prints the values of at most 12 groups and names at most 40, on a figure at most a page wide; its tables are `matrix` and `pairs`.

CyteArc follows CoSpar's definitions and differs from it in five places. CoSpar divides each clone's shares by their sum within each time point when every fate holds cells of one time point, where CyteArc divides them over the groups compared; pass the cells of one time point as `cell_selection` to compare the fates within it. CoSpar's `pvalue_for_fate_coupling` also shuffles the labels within each time point, which keeps each clone's cells at each time point, where CyteArc shuffles them over all the cells compared, so the two nulls differ when the groups span time points. CoSpar adds 0.0001 to each group's mean cells per clone for `"Weinreb"` and 1e-10 to the sums that it divides by, where CyteArc divides exactly. CoSpar counts every clone of its matrix in the means and the covariance, those without a cell in the compared groups included, where CyteArc counts the clones that the groups hold. CoSpar's `"Weinreb"` with `normalize=True`, its `fate_normalize_source="state_info"`, which divides by every cell of a fate, and its `ignore_cell_number` are not offered; `overlap`'s `cosine` is the cosine of presence.

The definitions were checked against CoSpar 0.5.0 run on a synthetic experiment of 990 cells at day 2 and day 4, with 120 clones of known fate profiles in six fates. On the day-4 cells, every coupling equals that of CoSpar's `fate_coupling` within 6e-17 for `"abundance_cosine"` and `"jaccard"`, within a relative 1.4e-15 for `"normalized_covariance"` once CoSpar's 0.0001 is taken out, and within a relative 5.1e-9 for `"share_cosine"`, the effect of CoSpar's 1e-10. The null means of 2,000 permutations are within 3 standard errors of those of CoSpar's `pvalue_for_fate_coupling`, and the standard deviations within 7%. On a million simulated cells, 899,879 of them in 314,595 clones, the coupling of 250 clusters, 31,125 pairs, against 1,000 permutations takes 2.4 s on the 20 threads of a Modal container that reserved 4 CPUs, and its arrays peak at 190 MiB; within the smallest budget it accepts, 211 MiB, it measures fewer permutations at once and peaks at 132 MiB.

## Measure the fate bias of early cells

The cells of an early cell's clone sampled later, such as the clone-mates at day 4 or day 6 of a progenitor at day 2, show which fates its progenitor took. {meth}`ds.clones.fate_bias <cytearc.datastore.namespaces.ClonesAccessor.fate_bias>` takes the `early` and `late` cells as two cell selections that share no cell, and the fates as the groups of the late cells: the combinations of the labels of `group_by` columns and of a `groups` label artifact. It saves a `clone_fate_bias` artifact that labels the early cells, which {meth}`ds.clones.load_fate_bias <cytearc.datastore.namespaces.ClonesAccessor.load_fate_bias>` reads as a {class}`~cytearc.clones.CloneFateBias` record.

```python
early = ds.snapshot_cell_selection("is_day_2")
late = ds.snapshot_cell_selection("is_day_4_or_6")
bias = ds.clones.fate_bias(clones, group_by="cell_type", early=early, late=late)
table = ds.clones.load_fate_bias(bias)
table.cells.head()  # each early cell's share of its clone's late cells in each fate
table.clone_fates.query("fate == 'Neutrophil' and q_value < 0.05")
towards_neutrophils = ds.select_cells(bias, include=["Neutrophil"])
ds.plots.clone_fates(bias)
```

For each early cell, the artifact holds the share of its clone's late cells in each fate, the clonal fate map that CoSpar's `fate_map(method="sum")` computes on the transition map of `infer_Tmap_from_clonal_info_alone`; its dominant fate, the fate that holds the most of them; and its clone's late cells. An early cell without a clone, or whose clone has no late cell, has no fractions and no dominant fate, and neither has one whose clone's late cells tie between fates. The artifact's canonical `values` hold the dominant fate, empty for none, so `select_cells` keeps the early cells whose clones went mostly to chosen fates, and `codes`, each cell's clone, `fate_codes`, its dominant fate's position, `late_cells` and `fractions`, one column per fate, are its other per-cell arrays.

For each clone with late cells, it holds their count in each fate and a one-sided Fisher test per fate: for a clone of `N` late cells, `x` of them in a fate of `K` of the `M` late cells with a fate, those without a clone included, the p-value is the chance of at least `x` in a draw of `N` of the `M` cells. Clonotypes keep receptor types apart, so a late cell without a clonotype has no fate, and each clonotype is tested among the late cells of its receptor type that have one, as the {doc}`repertoire` describes. This is the test that CoSpar's `clonal_fate_bias` makes with `alternative="less"` on the late cells, which computes this upper tail where its `"greater"` computes the lower one; given every cell, CoSpar counts the early cells in `M` too. The p-values of each fate are adjusted by the method of Benjamini and Hochberg over the clones with late cells, where CoSpar adjusts within the clones of each size. A clone of a few late cells cannot reach a small p-value: two late cells, both of a fate that holds 40% of the late cells, give about 0.16.

`load_fate_bias` returns three tables: `cells`, one row per early cell indexed by cell ID, with `clone`, `late_cells`, `fate`, the dominant fate, missing for none, and one column per fate of its share, NaN without late cells; `clone_fates`, one row per clone with late cells and fate, by clone and then fate, with `clone`, `fate`, the fate's key columns, `early_cells`, `late_cells`, `cells`, the clone's late cells in the fate, `fraction`, `p_value` and `q_value`; and `fates`, one row per fate with its late cells, those without a clone included unless the clones keep labels apart. A fate may not be named `clone`, `late_cells` or `fate`, the other columns of the `cells` table. The result records the clones, the early and late cells, the label artifact and, with `group_by`, each late cell's fate, so measuring again with the same values returns the saved result.

{meth}`ds.plots.clone_fates <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_fates>` draws the `top` (40) clones with the most late cells, grouped by their dominant fate in the order of the fates and within a fate from the most late cells to the fewest, with clones whose late cells tie last: each clone's late cells stacked by fate, or their shares with `normalize=True`, under its early cells. Its table is `clones`.

On the synthetic experiment above, every early cell's fractions equal CoSpar 0.5.0's clonal fate map within a relative 1e-10, the effect of the 1e-10 that CoSpar adds to each clone's late cells, and CoSpar gives the 64 early cells without late clone-mates 0 in every fate, where CyteArc gives them none. The p-values of all 108 clones with late cells, in all six fates, equal those of `clonal_fate_bias`, and the adjusted ones those of statsmodels' Benjamini-Hochberg within 2.1e-16. On the same million cells, with the 124,841 cells of one sample early, the cells of the other seven samples late and ten cell types as fates, measuring the fate bias of 287,201 clones takes 1.7 s and peaks at 135 MiB of arrays, and loading its tables, 2.9 million rows of clone and fate, takes 0.6 s and peaks at 397 MiB, within the smallest budgets they accept, 284 MiB and 601 MiB.

## Clones

```{eval-rst}
.. autoclass:: cytearc.clones.Clones
```

## Clonal expansion

```{eval-rst}
.. autoclass:: cytearc.clones.ClonalExpansion
```

## Clonal diversity

```{eval-rst}
.. autoclass:: cytearc.clones.ClonalDiversity
.. autoclass:: cytearc.clones.Rarefaction
```

## Clone overlap

```{eval-rst}
.. autoclass:: cytearc.clones.CloneOverlap
.. autoclass:: cytearc.clones.CloneTracking
```

## Fate coupling and fate bias

```{eval-rst}
.. autoclass:: cytearc.clones.CloneCoupling
.. autoclass:: cytearc.clones.CloneFateBias
```
