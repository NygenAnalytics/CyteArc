# Clones and lineage tracing

Import the clones that a lineage tracing tool or a study assigned to cells, join them to the store's cells, check how many barcodes joined, see how large the clones are, measure clonal expansion and diversity within samples or other groups of cells, and compare groups by the clones they share. Docs: <https://docs.nygen.io/CyteArc/api/clones.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

## When to use

- A file that lists each cell's clone: CoSpar's `Cell_ID,Clone_ID` table, TREX's `clones.txt`, CellTag-multi's or CellTagR's clone table, STICR's Cell Browser `meta.tsv`, Cassiopeia's allele table (`lineageGrp`), Numbat's `clone_post_{i}.tsv`, LARRY's GEO clone matrix with its metadata, an AnnData file with CoSpar's `obsm['X_clone']` or MiTo's `obs['MiTo_clone']`, or any table with a cell and a clone column.
- A cell metadata column that already holds each cell's clone: `ds.clones.from_column`.
- Not for files that list barcodes per cell or per molecule without clone IDs (TREX `molecules.txt`, pycashier, BARtab counts): those need clone calling, which this module does not do.

## Key concepts

- `ds.clones.import_clones(paths, *, cell_selection, barcode_prefix=None, barcode_suffix=None, barcode_pattern=None, format=None, columns=None, multiple="error", invalidate_cache=False)` saves a datastore-scoped `clones` artifact. The format comes from the file: HDF5 is AnnData, `%%MatrixMarket` a clone matrix, and a text header names the tool. `format` is one of `"cospar"`, `"trex"`, `"celltag_multi"`, `"celltagr"`, `"sticr"`, `"cassiopeia"`, `"numbat"`, `"table"`, `"matrix"`, `"cospar_h5ad"`, `"mito"`, `"h5ad"`.
- `columns={"cell": ..., "clone": ..., "probability": ...}` reads any other table (format `"table"`) or an AnnData `obs` column (`"h5ad"`, clone and probability only). `"cell"` may list columns joined by `"__"`, as LARRY's `["Library", "Cell barcode"]`. A table with R row names (header one field shorter than its rows, such as the Biddy workflow's `celltag_data.txt`, or a first column with an empty name, as `write.csv` writes) takes cells from the row names, so `columns={"clone": "CellTagD13"}` reads one labelling round. Blank lines hold no rows. A table whose every row ends in its delimiter, one field more than its header with nothing in the last, is refused, since R's row names with an empty last column read alike; end the header with the delimiter too, or write the row names under a header name (`write.table(..., col.names=NA)`).
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
- Summaries take `clones, *, group_by=None, groups=None, cell_selection=None` and count each clone's cells within groups: the combinations of the `group_by` columns' labels and the `groups` label artifact's (key column named by its kind, such as `cluster_labels`). Cells without a clone or a label count in no group, and when no cell has both, every summary raises `ValueError`. A group is named by its labels joined by `", "`, or `"all cells"`; labels with commas that give two groups one name, and a key column named like a column of the result, raise `ValueError`.
- `ds.clones.expansion(clones, group_by=None, groups=None, cell_selection=None, breakpoints=None, frequency_breakpoints=None, invalidate_cache=False)` saves a datastore-scoped `clonal_expansion` artifact (operation `measure_clonal_expansion`) of the cells of `cell_selection`, or of the clones' cells; a selected cell the clones do not hold has no clone. Pass at most one bound: `breakpoints`, default `(1, 2, 5, 20, 100)`, gives `"1 cell"`, `"2 cells"`, `"3 to 5 cells"`, `"6 to 20 cells"`, `"21 to 100 cells"`, `"more than 100 cells"` (scirpy's `clonal_expansion`); `frequency_breakpoints` (bins `(lower, upper]` ending at 1, such as scRepertoire's `(0.0001, 0.001, 0.01, 0.1, 1)`) bins each clone's share of its group's cells with a clone.
- Expansion arrays: canonical `values` (category, empty without a clone or group), `group_clone_size`, `group_clone_frequency`, `group_codes`. `select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])` keeps the cells of large clones. Reuse follows the clones, cells, label artifact, bounds and each cell's `group_by` labels. Every summary checks the memory budget before it reads: about 300 bytes per cell to measure expansion and 500 to load it beside the cells' IDs as text, 100 per cell to group plus three times the stored width of each `group_by` column's labels, and 2,000 per group, so a label for each cell costs far more than the cells; the clones API page lists every step.
- `ds.clones.load_expansion(expansion, splits=(10, 100, 1000, 10000, 30000, 100000))` returns a `ClonalExpansion` record: `.cells` (`clone`, `category` (ordered), `group`, the key columns, `group_clone_size`, `group_clone_frequency`; labels are categoricals), `.categories` (`group`, key columns, `category`, `cells`, `clones`, `fraction`; every category of every group), `.rank_windows` (`group`, key columns, `window` such as `"1 to 10"`, ..., `"after 100000"`, `cells`, `fraction`: scRepertoire's `clonalProportion`), `.labels`, `.breakpoints`, `.frequency_breakpoints`, `.splits`, `.group_by`, `.groups`, `.cell_selection`, `.clones`.
- `ds.plots.clonal_expansion(expansion, group_by=None, groups=None, count="cells", normalize=True, splits=...)` draws stacked bars per group of the categories (cells, or `count="clones"`) and of the rank windows. Groups default to the expansion's own; `group_by`/`groups` regroup the cells (categories stay those measured within the expansion's groups; ranks are recounted within the plotted groups). Tables: `categories` (`group`, `category`, `cells` or `clones`, `fraction`), `rank_windows` (`group`, `window`, `cells`, `fraction`).
- `ds.clones.diversity(clones, group_by=None, groups=None, cell_selection=None, size=None, coverage=None)` returns a `ClonalDiversity` record and saves nothing: `.indices`, one row per group (`group`, key columns, `cells`, `clones`, `singletons`, `doubletons`, `coverage`, `shannon` in nats, `pielou`, `simpson`, `inverse_simpson`, `gini_simpson`, `gini`, `d50`, `chao1`, `chao1_bias_corrected`, `ace`), and `.hill`, one row per group and order 0, 1, 2 (`observed`, `estimated` asymptotic, `at_size`, `coverage_at_size`, `method_at_size`, `at_coverage`, `size_at_coverage`, `method_at_coverage`; methods `"rarefaction"`, `"observed"`, `"extrapolation"`), with `.size` and `.coverage`, by default iNEXT's smallest doubled size and lowest coverage at doubled sizes. `cytearc.clones.ORDERS` is `(0, 1, 2)`.
- `ds.clones.rarefaction(clones, group_by, groups, cell_selection, orders=(0, 1, 2), sizes=None, endpoint=None, knots=40)` returns a `Rarefaction` record: `.curves` (`group`, key columns, `order`, `size`, `method`, `diversity`, `coverage`); default sizes are iNEXT's 40 knots per group (half up to its size, half to twice it); `sizes` gives every group the same sizes.
- `ds.plots.clonal_rarefaction(clones, group_by=..., cell_selection=..., orders=(0, 1, 2), base="size" or "coverage")`: one panel per order (`order_0`, `order_1`, `order_2`) plus `coverage` (sample completeness) with `base="size"`; solid rarefied, dashed extrapolated, a point at the observed size. Table: `curves`.
- `ds.clones.overlap(clones, group_by=None, groups=None, cell_selection=None)` returns a `CloneOverlap` record and saves nothing; at least two groups must hold a cell with a clone. `.group_sizes`: `group`, key columns, `cells`, `clones`. `.pairs`, one row per pair of groups: `group_1`, `group_2`, `shared`, `jaccard`, `sorensen`, `overlap` (shared over the smaller group's clones), `cosine` (of presence), `morisita_horn`, `abundance_cosine` (of clone sizes), `shared_fraction_1`, `shared_fraction_2` (the share of each group's cells whose clone the other holds). `cytearc.clones.OVERLAP_INDICES` names the indices. Memory: about 250 bytes per pair of groups; `track` about 100 bytes per row and 16 per row for each key column.
- `ds.clones.track(clones, group_by, groups, cell_selection, top=10, labels=None)` returns a `CloneTracking` record: `.tracked` (labels, largest first) and `.cells`, one row per chosen clone and group, zeros included: `clone`, `group`, key columns, `cells`, `fraction` (of the group's cells with a clone). By default the union of each group's `top` largest clones (scRepertoire's `clonalCompare`; ties to the larger clone overall); `labels` names clones instead.
- `ds.plots.clonal_overlap(clones, group_by=..., index="morisita_horn")` draws one index as a heatmap of the pairs (values printed for at most 12 groups, group names for at most 40; tables `matrix`, `pairs`); `ds.plots.clone_tracking(clones, group_by=..., top=10, labels=None, order=None)` draws one line per clone of its share of each group's cells (`order` may hold numbers, such as time points; the legend lists at most 80 clones, largest first; table `cells`).
- Validated on immune receptor clonotypes: every cell's category equals scirpy 0.25.1's `clonal_expansion`, frequency fractions equal scRepertoire 2.8.0's `clonalHomeostasis` and rank windows its `clonalProportion`; Hill-number curves within 1.4e-12 of iNEXT 3.0.2, `estimateD` within 1e-7, every index within 6e-14 of scRepertoire 2.8.0 `clonalDiversity`, immunarch 0.10.3 `repDiversity` and vegan 2.7.6 `estimateR`; every overlap index within 5e-14 of immunarch 0.10.3 `repOverlap` and scRepertoire 2.8.0 `clonalOverlap`.

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
expansion = ds.clones.expansion(clones, group_by="sample")
ds.clones.load_expansion(expansion).categories  # cells and clones per sample and category
ds.plots.clonal_expansion(expansion, group_by="cell_type", show=False).save("clonal_expansion.png")
diversity = ds.clones.diversity(clones, group_by="sample")
print(diversity.indices[["sample", "cells", "clones", "coverage", "shannon", "inverse_simpson", "chao1"]])
print(diversity.hill.pivot(index="sample", columns="order", values="at_coverage"))
ds.plots.clonal_rarefaction(clones, group_by="sample", show=False).save("clonal_rarefaction.png")
overlap = ds.clones.overlap(clones, group_by="cell_type")
print(overlap.pairs[["group_1", "group_2", "shared", "morisita_horn", "shared_fraction_1", "shared_fraction_2"]])
ds.plots.clone_tracking(clones, group_by="time_point", top=5, show=False).save("clone_tracking.png")
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
- Measure expansion and diversity `group_by` the sample: without it, a clone's cells in all samples add up. Frequency categories depend on sample size (one cell of 1,000 is already 0.001), so compare them only between samples of similar size; bounds in cells compare any samples.
- Compare diversity by Hill numbers at a common coverage or size (`hill.at_coverage`, `hill.at_size`), not observed indices, which grow with the cells sampled. Check `plots.clonal_rarefaction`: steep curves at the observed point mean richness and Chao1 are poorly determined; order 2 is stable. Name the Chao1 form reported. Leave out tiny groups with `cell_selection`, or set `size`/`coverage`, since the smallest group sets the defaults.
- Read overlap by the Morisita-Horn index and the shared fractions; presence indices (Jaccard, Sørensen, overlap, cosine) grow with both groups' cells, so compare them only between pairs of similar depth. A clone seen in one group only may be below detection in the others.
- Compare conditions with one value per sample (such as the fraction of cells in clones of more than two cells), never with cells as replicates; test such values with a method that treats samples as replicates, such as CytePool's (https://docs.nygen.io/CytePool/).
