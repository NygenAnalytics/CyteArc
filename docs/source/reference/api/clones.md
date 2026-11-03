---
myst:
  html_meta:
    description: CyteArc clones API for lineage tracing, with clones imported from CoSpar, TREX, CellTag-multi, CellTagR, STICR, Cassiopeia, Numbat, MiTo and LARRY's clone matrices or from a cell metadata column, lineage barcode reads, molecules and counts imported from TREX, pycashier, CellTag-multi, FateMap, QuiCAT, BARtab and Cell Ranger, lineage recorder alleles imported from CARLIN, DARLIN, Cassiopeia, PEtracer, DNA Typewriter and TreeData files and exported as character matrices for Cassiopeia and LAML, joined to cells by barcode prefix, suffix or pattern, saved as a clones artifact, read as tables of cells and clones, exported as CoSpar's clone table or a Matrix Market file, drawn as clone sizes, measured within groups of cells, immune receptor clonotypes among them, as clonal expansion, diversity indices and Hill numbers with rarefaction and extrapolation, compared between groups by the clones they share, with chosen clones followed across groups, measured as fate coupling between groups against shuffled labels and as the fate bias of early cells, with CoSpar's definitions, and cut from cell phylogenies read from Newick and TreeData files at a time or a depth.
---

# Clones API reference

A clone is the set of cells that descend from one ancestor cell. Lineage tracing methods mark the ancestor with a heritable barcode, such as LARRY's, CellTag's or TREX's expressed barcodes, and other methods infer clones from the cells' own mutations, such as copy-number changes or mitochondrial variants. Each tool writes which clone each cell belongs to in its own format. `ds.clones` reads those files, joins their cells to a store's cells, and saves the clones as a `clones` artifact that CyteArc's clone analyses read whatever defined the clones. The T cell and B cell clonotypes of {meth}`ds.repertoire.clonotypes <cytearc.datastore.namespaces.RepertoireAccessor.clonotypes>` are clones too: every function on this page reads a `clonotypes` artifact as it reads `clones`, and measures each receptor type apart, as the {doc}`repertoire` describes.

{meth}`ds.clones.import_clones <cytearc.datastore.namespaces.ClonesAccessor.import_clones>` imports the clones of one file or several, and {meth}`ds.clones.from_column <cytearc.datastore.namespaces.ClonesAccessor.from_column>` takes them from a cell metadata column. {meth}`ds.clones.load_clones <cytearc.datastore.namespaces.ClonesAccessor.load_clones>` reads the artifact as a {class}`~cytearc.clones.Clones` record, {meth}`ds.clones.export_clones <cytearc.datastore.namespaces.ClonesAccessor.export_clones>` writes it as a file that other tools read, and {meth}`ds.plots.clone_abundance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_abundance>` draws how many clones have each size. {meth}`ds.clones.expansion <cytearc.datastore.namespaces.ClonesAccessor.expansion>` places each cell in an expansion category by its clone's size within its sample or other group, {meth}`ds.clones.diversity <cytearc.datastore.namespaces.ClonesAccessor.diversity>` and {meth}`ds.clones.rarefaction <cytearc.datastore.namespaces.ClonesAccessor.rarefaction>` measure how many clones each group holds and how evenly its cells spread over them, {meth}`ds.clones.overlap <cytearc.datastore.namespaces.ClonesAccessor.overlap>` and {meth}`ds.clones.track <cytearc.datastore.namespaces.ClonesAccessor.track>` compare groups by the clones they share and follow chosen clones across them, {meth}`ds.clones.coupling <cytearc.datastore.namespaces.ClonesAccessor.coupling>` measures how strongly groups share clones against shuffled labels, and {meth}`ds.clones.fate_bias <cytearc.datastore.namespaces.ClonesAccessor.fate_bias>` follows each early cell's clone to its later cells and tests which fates each clone took. {meth}`ds.clones.import_barcodes <cytearc.datastore.namespaces.ClonesAccessor.import_barcodes>` reads the lineage barcodes of each cell, from which clones are called, and {meth}`ds.clones.import_alleles <cytearc.datastore.namespaces.ClonesAccessor.import_alleles>` reads the alleles of lineage recorders, from whose rare alleles {meth}`ds.clones.allele_clones <cytearc.datastore.namespaces.ClonesAccessor.allele_clones>` calls clones and whose edits tree builders read from the matrices that {meth}`ds.clones.export_characters <cytearc.datastore.namespaces.ClonesAccessor.export_characters>` writes. {meth}`ds.clones.import_phylogeny <cytearc.datastore.namespaces.ClonesAccessor.import_phylogeny>` imports the trees that lineage recorders reconstruct and {meth}`ds.clones.phylogeny_clones <cytearc.datastore.namespaces.ClonesAccessor.phylogeny_clones>` cuts them into clones.

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

Saved clones are reused when the files hold the same bytes, a clone matrix's lists among them, with the same options and, with several files, the same namespaces ({ref}`reuse of imports <import_reuse>`).

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

## Import lineage barcodes

Expressed lineage barcodes, such as LARRY's, CellTag's and TREX's, are random sequences written into each founding cell's genome; the cell and its descendants transcribe them, so the barcodes that cells share mark their common ancestor. Tools that extract the barcodes from sequencing reads write them at one of three levels. {meth}`ds.clones.import_barcodes <cytearc.datastore.namespaces.ClonesAccessor.import_barcodes>` reads any of them, joins each row to a selected cell by its cell barcode and saves a `lineage_barcodes` artifact. {meth}`ds.clones.load_barcodes <cytearc.datastore.namespaces.ClonesAccessor.load_barcodes>` reads it as a {class}`~cytearc.clones.LineageBarcodes` record, and {meth}`ds.plots.barcode_calls <cytearc.datastore.plot_accessor.DataStorePlotAccessor.barcode_calls>` draws the counts that clone calling filters.

```python
barcodes = ds.clones.import_barcodes("trex_run/molecules.txt", cell_selection=cells, barcode_suffix="-1")
table = ds.clones.load_barcodes(barcodes)
table.records.head()  # one row per molecule: cell, library, barcode, UMI, UMIs and reads
ds.plots.barcode_calls(barcodes)
```

| Level | One row per | Counts | Saved as |
|---|---|---|---|
| reads | read, naming its cell, UMI and barcode | each row is one read | molecules whose reads are their rows, or, without UMIs, pairs of a cell and a barcode |
| molecules | cell, UMI and barcode | the molecule's reads, or none | molecules |
| pairs | cell and barcode | its UMIs or reads, or none | pairs |

Nothing is filtered at import: one-read molecules, barcodes with sequencing errors and ambient barcodes are kept, so that clone calling applies the rules of the method that made the data.

### Formats

`import_barcodes` recognises each file's format from its header, and pycashier's and FateMap's tables without a header from the fields of their first row. `format=` names the format, and `"reads"`, `"molecules"` or `"pairs"` reads any other table by the columns that `columns` names. A file of clone IDs, such as TREX's `clones.txt`, is refused with a pointer to `import_clones`.

| `format` | File | Level | Columns read |
|---|---|---|---|
| `"trex"` | [TREX](https://github.com/frisen-lab/TREX)'s `molecules.txt` and `molecules_corrected.txt` of every release | molecules; reads for Smart-seq2 runs, which write no `umi` | `cell_id`, `umi`, `clone_id`, and `read_count` since September 2024; `-` and `0` in a cloneID are kept |
| `"pycashier"` | [pycashier](https://github.com/brocklab/pycashier)'s `<sample>.umi_cell_labeled.barcode.tsv`, or, before release 2024.1001, its `cell_record_labeled` table without a header | reads | `cell`, `umi`, `barcode` |
| `"celltag"` | [CellTag-multi](https://github.com/morris-lab/CellTag-multi-2023)'s per-read tables, and the parsed reads of the Biddy et al. workflow | reads; scATAC-seq tables have no `UMI` | `Cell.BC`, `UMI`, `Cell.Tag` |
| `"fatemap"` | [FateMap](https://github.com/arjunrajlaboratory/FateMap_Goyal2023)'s `stepThreeStarcodeShavedReads.txt`, with a header or without | molecules | `cellID`, `UMI`, `BC50StarcodeD8` |
| `"quicat"` | [QuiCAT](https://github.com/theislab/quicat)'s per-sample `<sample>_output.csv` | pairs | `cell`, `barcode`, `counts` as UMIs |
| `"quicat_wide"` | QuiCAT's merged `barcodes_output.csv` | pairs | `cell`, and one column of UMIs per barcode |
| `"h5ad"` | an AnnData file of cells by barcodes, such as QuiCAT's `barcodes_output.h5ad` | pairs | the `obs` and `var` names, and `X` as UMIs |
| `"bartab"` | [BARtab](https://github.com/DaneVass/BARtab)'s `<sample>.counts.tsv` | pairs | `gene`, `cell`, `count` as UMIs |
| `"bartab_annotation"` | BARtab's `<sample>_cell_barcode_annotation.tsv` | pairs | each row's name, and its lists `barcode` and `bc.umi.count`, joined by `;`, or by `,` in early files |
| `"molecule_info"` | Cell Ranger's `molecule_info.h5`, the molecules of one feature barcode library | molecules | each molecule's cell barcode, UMI, feature ID and reads |

A file in TREX's format must list each cell and UMI once, as its molecule tables do. TREX's `reads.txt` has the header of its early `molecules.txt` but one row per read, so a TREX table that lists a cell and UMI twice is refused. FateMap's table of several samples is refused, since their cells share bare barcodes; read it with `format="molecules"` and `columns={"cell": ["SampleNum", "cellID"], "umi": "UMI", "barcode": "BC50StarcodeD8"}`, which joins each cell's sample to its barcode.

A tool's table must hold every value that the tool writes in each row, so a row cut short, such as a TREX molecule without its `read_count` or a BARtab count without its UMIs, is refused, as is a gzip file cut short. Every field of QuiCAT's merged table must hold a whole number, and a merged table saved again with an index column is refused.

QuiCAT counts distinct UMIs for 10x Genomics data and reads for a technology without UMIs. Its formats read the counts as UMIs, so read a table of reads with `format="pairs"` and `columns={"cell": "cell", "barcode": "barcode", "reads": "counts"}`, and an AnnData file with `format="h5ad"` and `columns={"reads": "X"}`.

Cell Ranger counts a lineage library only when its barcodes are listed in the feature reference, as `Custom` or `CRISPR Guide Capture` features. The import reads the molecules of the file's only library that is not gene expression, or of the type that `library` names, decodes each 2-bit UMI with the length that the file's `metrics_json` records, as Cell Ranger does from release 6, names each barcode by its feature ID, and appends each molecule's GEM well to its cell barcode, as Cell Ranger's matrices name cells. Barcodes that Cell Ranger did not call cells are counted as unmatched.

BARtab is licensed under the GPL, so its formats follow its documentation and published files only.

### Tables of other tools

`format="reads"`, `"molecules"` or `"pairs"` names what each row of a table is, and `columns` maps roles to its columns: `"cell"` and `"barcode"`, each a column or a list of columns whose values are joined by `"__"`; `"umi"`, which molecules need and reads may have; `"umis"` and `"reads"`, a pair's counts or a molecule's reads; `"library"`, each row's library; and `"separator"`, one of `";"`, `","`, `":"` and `"|"`, which splits the barcode and count fields of pairs into lists whose items stay aligned. Without `"cell"`, a table in R's row-name layout takes each cell from its row names. Counts are whole numbers. In such a table, a barcode, UMI, count or library that is empty or holds text that pandas reads as missing, such as `NA`, `N/A`, `#N/A`, `NaN` or `null`, has no value: a row without a barcode holds none, a count is unknown, and a molecule without its UMI or a row without its library is refused. A tool's own format must hold the fields that the tool fills in every row, so there such a value marks a row cut short, which is refused. Cell barcodes, the barcodes that head the columns of QuiCAT's merged table and the names in an AnnData file are read as written.

```python
# CloneTracker XP's barcode has two parts, 14 and 30 bases; each pair counts UMIs.
barcodes = ds.clones.import_barcodes(
    "clonetracker.tsv",
    cell_selection=cells,
    format="pairs",
    columns={"cell": "cell", "barcode": ["bc14", "bc30"], "umis": "n"},
)
# Watermelon's metadata lists each cell's barcodes in its row, without counts.
barcodes = ds.clones.import_barcodes(
    "GSE150949_metaData_with_lineage.txt.gz",
    cell_selection=cells,
    format="pairs",
    columns={"barcode": "lineage_barcode", "separator": ","},
)
```

### Libraries

Each file's rows belong to a library, which `library` names, one for every file or one per file; without it, a file's library is its name before the first `.`. A library holds molecules or pairs, never both, and its records keep apart from those of other libraries, even in one cell. Rows of one library that name one molecule or pair merge, summing their reads, so a library sequenced in two runs gives each molecule the reads of both.

```python
# CellTag-multi's RNA reads have UMIs and its ATAC reads none: two libraries.
barcodes = ds.clones.import_barcodes(
    ["parse_rna.txt.gz", "parse_atac.txt.gz"],
    cell_selection=cells,
    library=["RNA", "ATAC"],
)
```

### Barcodes counted by an assay

A lineage library counted as features of an assay, such as a Cell Ranger feature barcode matrix, is imported with `from_assay` instead of files: each nonzero count of a selected cell is a pair of the cell and the feature's ID, and the count its UMIs. The library is named after the assay unless `library` names it, and file options such as `barcode_prefix` or `format` are refused.

```python
barcodes = ds.clones.import_barcodes(cell_selection=cells, from_assay="LINEAGE")
```

### Join, reuse and memory

Cell barcodes join cells as clone files do: a cell's ID is its file's `barcode_prefix`, the cell barcode and `barcode_suffix`, and `barcode_pattern` keeps the part that its `barcode` group matches. Cell barcodes that match no selected cell, such as the droplets that pycashier, BARtab and Cell Ranger list, are counted in the artifact's `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` attributes, and a warning names them, which a reused import gives again. `n_file_rows` and `n_unmatched_rows` count the rows read, one for each item of a list field and for each nonzero count of a merged table or a matrix. A row without a barcode and a pair counted 0 hold none and are counted in `n_empty_rows`. Saved barcodes are reused when each file holds the same bytes under the same library, its name unless `library` names it ({ref}`reuse of imports <import_reuse>`), so a file moved to another folder reuses them, as does a renamed file whose library is named.

A file is read in chunks of about a twentieth of the memory budget: at most 50,000 rows of a text table, rows of a merged table, whose fields are parsed in one pass, and stored entries of an AnnData matrix, read in tiles of whole HDF5 chunks when it is dense. Each chunk's reads collapse into molecules at once, and the chunks merge into the records once they outnumber them, so an import holds at most about twice as many rows as it saves records, however many reads a file lists. Before it reads the selected cells' IDs, the import checks their lookup and the read of their IDs as text, which decodes a stored chunk of them. It then checks its peak against the memory budget before it reads a file and as it reads: the lookup of the selected cells with their IDs as text; an AnnData file's names of cells and barcodes and its column of libraries, at the size they take decoded, which it reads once to count; one chunk, with HDF5's buffer for the chunk it inflates; the cell barcodes that match no selected cell; the keys that find a TREX table's repeated molecules; about 1,300 bytes for each chunk that waits to merge, of which at most 256 wait; and, at the peak of a merge, about 150 bytes per row held and 140 per distinct barcode and UMI. It refuses with `MemoryError` before it writes, and suggests a smaller `cell_selection` only when the selected cells take most of the peak.

On one thread, CellTag-multi's per-read table of 100,000 cells, 9.6 million reads in 198 MiB of gzip, imports into 1.2 million molecules in 35 s, holding at most 2.3 million rows and 583 MiB above its start; TREX's 2 million molecules of 100,000 cells import in 13 s and 544 MiB; and BARtab's counts of 1,000,000 cells with five barcodes each, 5 million pairs, import in 21 s and 1,203 MiB, which the check estimates as 1,337 MiB. QuiCAT's merged table of 50,000 cells by 5,000 barcodes, 478 MiB of text, imports in 14 s within a budget of 512 MiB, holding 39 MiB above its start, and a dense AnnData matrix of 20,000 cells by 50,000 barcodes, 3.7 GiB once read, imports in 15 s within 1 GiB, holding 74 MiB. `load_barcodes` reads the first three in 0.6, 1.0 and 1.7 s.

### Read and plot lineage barcodes

A `lineage_barcodes` artifact holds one record per molecule or pair, ordered by cell, library, barcode and UMI, with each distinct barcode and UMI stored once. A molecule's `umis` is 1, and a pair's UMIs and either's reads are -1 where its library counts none. The canonical `n_barcodes` holds each cell's distinct barcodes over its libraries, and `n_umis`, `n_reads` and `n_records` hold its UMIs, reads and records, -1 where a library of the cell counts none. `select_cells` keeps the values above `low`, so `ds.select_cells(barcodes, low=0)` keeps the cells with a barcode and `low=1` those with two or more.

`load_barcodes` returns a {class}`~cytearc.clones.LineageBarcodes` record. Its `cells` table holds one row per selected cell, indexed by cell ID, with the per-cell counts. Its `records` table holds one row per record: `cell`, `library`, `barcode` and `umi` as categoricals, `umi` missing for a pair, and `umis` and `reads` as nullable integers. Its `libraries` table names each library's `level`, `"molecules"` or `"pairs"`, its `records`, and whether they count UMIs and reads. `pairs()` collapses the records into one row per cell, library and barcode, with the pair's distinct UMIs, reads and molecules.

`ds.plots.barcode_calls(barcodes)` draws three panels: the cells by their distinct barcodes, cells without a barcode included and `max_barcodes` or more (default 10) in the last bar; each library's pairs by their UMIs; and each library's molecules by their reads, on log scales, so molecules of 0 reads appear in the table only. Barcodes of one or two UMIs and molecules of one read are mostly sequencing errors and ambient molecules, which clone calling filters out. Its tables are `barcodes_per_cell` (`barcodes`, `cells`), `umis_per_barcode` (`library`, `umis`, `pairs`) and `reads_per_molecule` (`library`, `reads`, `molecules`).

## Call clones from lineage barcodes

Cells that carry the same lineage barcodes descend from one labelled cell, but the barcodes that extraction tools report also hold sequencing and PCR errors, ambient molecules from other cells and barcodes outside the library. {meth}`ds.clones.barcode_clones <cytearc.datastore.namespaces.ClonesAccessor.barcode_clones>` removes them by the published rules of the method that made the data, groups the cells by the barcodes that remain, and saves a `clones` artifact that every clone summary reads.

```python
barcodes = ds.clones.import_barcodes("trex_run/molecules.txt", cell_selection=cells, barcode_suffix="-1")
clones = ds.clones.barcode_clones(barcodes, preset="trex")
ds.clones.load_clones(clones).cells.head()  # clone, clone_size, n_barcodes, doublet
ds.clones.diversity(clones, group_by="sample")
```

The rules run in this order:

1. Molecule filter: a molecule with fewer than `min_reads` reads is dropped.
2. Correction: within each group of `within` and each library, barcodes within `distance` of each other by `metric` are linked, `"hamming"` counting substitutions between barcodes of one length and `"levenshtein"` substitutions, insertions and deletions. Every pair within the distance is found exactly by the neighbour search that clonotype clusters use, whose keys mask runs of positions of long barcodes, so that 100,000 barcodes of 28 bases within 3 substitutions take about 0.1 s. Each connected component of linked barcodes merges into its most abundant barcode, by UMIs, or reads in a library that counts no UMIs, and a tie goes to the barcode that sorts last, as TREX picks its representatives. Molecules of one cell and UMI that correction gives one barcode become one molecule. Then the barcodes outside `allowlist` are dropped.
3. Pair filters: within each library, a cell's barcode of fewer than `min_umis` UMIs is dropped, and so is one whose UMIs divided by `min_fraction` stay below the UMIs of the cell's top barcode, the rule of BARtab's `--umi_fraction_filter`.
4. Cell filter: a cell with fewer or more barcodes than `barcodes_per_cell`, a pair of the fewest and the most, gets no clone.
5. Grouping: `method="sets"` makes the cells of each identical barcode set a clone. `method="jaccard"` makes each identical set a node and links two nodes that share a barcode when the Jaccard index of their sets, the barcodes they share over the barcodes either holds, exceeds `similarity`, or reaches it with `inclusive=True`. `prune=True` removes TREX's bridges, the edges whose two ends share no neighbour while either end has another, and `doublets=True` TREX's doublets, the nodes of one cell whose neighbours fall apart into two or more groups without them, in TREX's order: pruning, doublets, pruning again, doublets again. The connected components are the clones, and a clone of fewer than `min_size` cells leaves its cells without one.

Every rule runs within the groups of `within`, cell metadata columns whose labels keep clones apart, such as the mouse, so each group's clones are those it would get alone; every cell with a barcode needs a label. `similarity` may also be two numbers: the index that links cells whose barcodes come from the same libraries, and the index that links any others, as CellTag-multi links scRNA-seq and scATAC-seq cells. Each library counts as one modality, and each file is its own library unless `import_barcodes` names it, so import the files of one modality under one name, such as `library=["RNA", "RNA", "ATAC"]` for two scRNA-seq lanes and one scATAC-seq file.

### Presets

`preset` sets every rule to a method's published values, and an argument given replaces its preset's value; a count of 0 turns its filter off, so `preset="larry", min_reads=0` keeps every molecule. Without a preset, every filter is off and correction does not run, and `method` is required, with `similarity` and `inclusive` for `method="jaccard"` and a `metric` for a `distance`, since the tools differ in each. Whether the Jaccard test is strict is part of each method's rules: TREX and CellTagR link above the threshold, bartools and STICR at it or above. The artifact records the resolved rules as its parameters.

| `preset` | Published rules of | Molecule, correction and pair rules | Grouping |
|---|---|---|---|
| `"larry"` | LARRY, Weinreb et al. ([2018](https://doi.org/10.1101/467886)) | `min_reads=10`, `distance=3` with `metric="hamming"`, `min_umis=3` | `method="sets"`, `min_size=1` |
| `"trex"` | TREX's `run10x`, Ratz et al. ([2022](https://doi.org/10.1038/s41593-022-01011-x)) | TREX's rules below, with `distance=5` and `metric="hamming"` | `method="jaccard"`, `similarity=0.7`, `inclusive=False`, `prune=True`, `doublets=True`, `min_size=1` |
| `"celltag"` | CellTagR and Biddy et al. ([2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6635140/)) | the library's `allowlist`, `min_umis=2`, `barcodes_per_cell=(2, 20)` | `method="jaccard"`, `similarity=0.7`, `inclusive=False`, `min_size=3` |
| `"celltag_multi"` | CellTag-multi, Jindal et al. ([2024](https://doi.org/10.1038/s41587-023-01931-4)) | `min_reads=2`, `distance=2` with `metric="levenshtein"`, the library's `allowlist`, `min_umis=2`, `barcodes_per_cell=(2, 25)` | `method="jaccard"`, `similarity=(0.6, 0.5)`, `inclusive=False`, `min_size=2` |

The CellTag presets keep the CellTags of the library's allowlist, which only the experiment holds, so they require `allowlist`, a sequence of barcodes such as `pd.read_csv("v1_whitelist.csv")["CellTag"]`; `allowlist=False` calls them without one. CellTag-multi's scATAC-seq tags count reads, so `min_umis=2` keeps the tags of two reads or more there, as the paper's "more than 1 read" does, while `min_reads=2` drops the one-read molecules of scRNA-seq. Its preset follows the paper's hematopoiesis data, whose scRNA-seq cells kept 2 to 25 CellTags; its scATAC-seq cells kept 1 to 25, and the reprogramming data kept 1 to 25 in both and linked cells above 0.6 within and across modalities, which `barcodes_per_cell=(1, 25)` and `similarity=0.6` give. Called on scATAC-seq alone, the paper linked cells above 0.5, which `similarity=0.5` gives, where the preset links cells of one modality above 0.6. CellTag-multi corrected its tags with Starcode and grouped cells by graph clustering in unlicensed code; CyteArc merges the connected components of near tags, as LARRY and TREX do, and takes the connected components of the graph.

The `celltag` preset applies CellTagR's filters and Jaccard threshold without three steps of the original workflows: the Starcode collapse of similar CellTags within each cell, which CellTagR's README strongly recommends and Biddy et al. ran within one edit; Biddy et al.'s removal of the CellTags in more than 5% of the first time point's cells; and CellTagR's selection of a clone when the similarities within it pass the cutoff, where Kong et al. ([2020](https://doi.org/10.1038/s41596-019-0247-2)) cluster cells hierarchically. `distance=1` with `metric="levenshtein"` merges near CellTags across the cells of each library rather than within each cell, and CyteArc takes the connected components of the graph.

LARRY's preset follows the order of the paper's methods: triples of a cell, a UMI and a barcode with at least 10 reads, then barcodes within Hamming distance 3 collapsed, then pairs of at least 3 UMIs. LARRY also dropped the pairs of cells in two sequencing libraries that share a cell barcode and a lineage barcode, and clones over-abundant in one library, which its methods do not state exactly enough to reproduce; its GEO in vitro set holds clones of two or more cells.

### TREX's rules

`preset="trex"` adds TREX's rules for its cloneIDs, in which `-` marks a base that no read covered and `0` a deletion, adapted from TREX's MIT-licensed `run10x`. They have no argument of their own, and the artifact records them as `trex_rules`:

- A molecule whose cloneID has a Shannon entropy below 1 bit, such as `AAAAAAAAAAAAAAAAAAAAAAAAAAAAGA`, is dropped before correction.
- Two cloneIDs are compared at the positions where both hold a base, and need 20 such positions, so a partial cloneID merges into the full one that it matches.
- A cloneID needs 20 bases after correction.
- A cell's cloneID of one UMI is dropped when another cell of its group and library holds the cloneID, and so is a cloneID whose only molecule has one read, which needs each molecule's reads. TREX writes `read_count` to `molecules.txt` since September 2024; an earlier run's `reads.txt` gives the reads instead, imported with `format="reads"` and `columns={"cell": "cell_id", "umi": "umi", "barcode": "clone_id"}`, and its molecules keep the cloneID of each read, where `molecules.txt` holds their consensus.

TREX compares two cloneIDs only when they share a substring of 7 characters, so it misses some pairs within 5 mismatches: full cloneIDs 4 mismatches apart, spread along them, or a partial cloneID of 20 bases 2 mismatches from a full one. CyteArc finds every pair exactly, which on TREX's expected outputs merges the same cloneIDs; on noisier data the pairs that TREX misses can change its clones. Two random cloneIDs that share 20 bases are within 5 mismatches about 4 times in a million, so in a library of 250,000 distinct cloneIDs a partial cloneID of 20 bases matches one unrelated cloneID by chance, and correction merges the two. On the million cells timed below, whose clones carry a million distinct cloneIDs, correction chained four in five cells into one clone. TREX's `smartseq3` command applies neither the entropy filter nor doublet removal: on its expected output the entropy filter changes no clone, and `doublets=False` turns doublet removal off.

### Counts each rule reads

Each library counts its records in one unit. A molecule is one UMI, and a pair counts its UMIs where its library counts UMIs for every pair, or its reads in a library that counts no UMIs for any pair but reads for every one, such as CellTag-multi's scATAC-seq tags. A library that counts UMIs for some pairs and not for others, such as a table read through `columns` whose UMI field is `NA` on one row, has no unit, and neither has one that counts neither. A rule refuses with `ValueError` a count it cannot read rather than guess one: `min_reads` needs the reads of every molecule, correction and the pair filters a unit for each library whose pairs they rank or test, and TREX's rules molecules with their reads. `min_reads` tests molecules alone, so a library of pairs passes it whole, and it refuses barcodes whose libraries all hold pairs. Rules that read no count, such as `method="sets"` alone, call clones from every pair. The artifact's `library_units` attribute names each library's unit, `"umis"`, `"reads"` or None.

### The clones artifact

The clones are labelled by size, `clone_1` the largest, and clones of one size by their first cell. Beside the `values`, `codes` and `clone_size` of every `clones` artifact, the artifact holds `n_barcodes`, each cell's barcodes after the pair filters, and `doublet`, whether TREX's rule removed the cell. Its `rule_counts` attribute counts what each rule removed or changed: molecules below `min_reads` or of low complexity, barcodes and records corrected, records outside the allowlist, molecules of few bases, pairs, pairs of TREX's two rules and below `min_umis` or `min_fraction`, cells with barcodes and outside `barcodes_per_cell`, the nodes and edges of the graph, the edges pruned, the doublets, and the cells in clones below `min_size`. `load_clones` adds `n_barcodes` and `doublet` to its `cells` table, and its `source` is `"call_barcode_clones"`. The artifact is identified by the barcodes, the preset and the resolved rules, the fingerprint of the allowlist's distinct barcodes, whatever their order, and of each cell's `within` labels.

### Diagnostics

Correction links every pair of barcodes within `distance`, and the more barcodes it compares, the more unrelated pairs fall within it by chance: TREX's rule, 5 mismatches over 20 shared bases, chained four in five of a million cells into one clone this way. The artifact's `diagnostics` attribute records four numbers that show it, and `load_clones(clones).diagnostics` returns them as a dictionary, so that a pipeline can turn them into warnings of its own:

| Name | What it counts |
|---|---|
| `distinctBarcodes` | The distinct barcodes that correction compares: those of the molecules and pairs that pass the molecule filters, counted in each group of `within` and each library apart, since correction compares no others. Under TREX's rules, cloneIDs of fewer than 20 bases, which no other is compared with, do not count. |
| `expectedChanceMatches` | The pairs of those barcodes expected within `distance` by chance, if barcodes of different components that correction links agreed at each position independently, as often as the components' barcodes do. |
| `mergedBarcodes` | The distinct barcodes that correction merged into another, the count that `rule_counts` names `barcodes_corrected`. |
| `largestCloneShare` | The cells of the largest clone divided by the cells given a clone: the cells of the clones of at least `min_size` cells, without the cells that no clone holds, such as doublets and cells outside `barcodes_per_cell`. It is 0 when no cell is given a clone. |

Without correction, at `distance=0`, the expectation and the merged barcodes are 0, and the distinct barcodes count what correction would compare.

Correction can go wrong in two ways, and two numbers show them. While correction leaves many components, its chance merges show in the expected chance matches beside the merged barcodes. Once its chance links join most barcodes into a few components, the chaining shows in the largest clone's share. The expectation describes pairs of barcodes of different components, so a group and library that holds one barcode and its errors expects nothing, and so does one that correction chains into a single component: there the largest clone's share reads close to 1.

The expectation counts the pairs that the search compares. Under `"hamming"`, the search compares the barcodes of one length within each group and library, and under TREX's rules each two groups of cloneIDs on the positions where both hold a base, when those are 20 or more. Each component of the barcodes of one length in a group and library gives one barcode, picked as correction picks the barcode that the component merges into: the most abundant, and of equally abundant ones the one that sorts last. Two barcodes agree at a position with the share of pairs of these barcodes of different components that hold one base there: the sum over the bases of each base's count times one less, over the components times one less. Under TREX's rules the pick is made at each position among the component's cloneIDs that hold a base there, so a component whose top cloneID was cut still counts where its others hold bases, and cloneIDs of fewer than 20 bases, which no other is compared with, count in no share. A set of compared barcodes that fewer than two components hold, or two groups of cut cloneIDs that compare a position that only one component holds, pairs only barcodes of one component and adds nothing. Otherwise, the chance of each number of mismatches over the positions that a pair compares follows from the chances of agreeing exactly, and each set adds its pairs times their chance of falling within the distance. Two distinct barcodes of one length never agree everywhere, so that case is left out of their chance. Under `"levenshtein"`, whose alignments have no fixed positions, each pair of lengths of a group and library follows the edit distance's dynamic programming over the chance that each pair of positions agrees, the share of pairs of the two lengths' barcodes of different components that hold one base there, treating those pairs as independent. This comes within a tenth of exact counts on random barcodes of 6 to 20 bases where the bases are about equally frequent, also with bases fixed at some positions, and overstates them where one base dominates, more so for longer barcodes and larger distances: on barcodes of 28 bases with one base at seven tenths of the positions, by 1.9 times within 3 edits and 3.4 to 3.5 times within 6 to 8. Its cost grows about eightfold with every two edits, so `metric="levenshtein"` takes a `distance` of at most 8.

Read the numbers together. The merged barcodes are what correction removed as errors, and the expected chance matches are the merges that unrelated barcodes alone would give: when correction merged barcodes, an expectation near or above their count means that it merges unrelated ones. Each chance match also links two barcodes, so once the expectation nears half the distinct barcodes, each barcode meets about one chance match, and correction's links start to chain barcodes, and the cells that carry them, into a few components. On synthetic clones of at most 9 cells, TREX's preset expected 0.30 chance matches per compared cloneID at 200,000 cells, where its largest clone already held 1,393 cells, and 1.5 at a million cells, 4,153,049 in all and more than the 2,666,822 cloneIDs it merged, where its largest clone held 0.80 of the cells given a clone. The expectation grows with the square of the barcodes compared, so a library ten times larger expects about a hundred times more. A largest clone's share far above that of the next clones, or above what the labelling allows, shows chaining however far it went, up to a library chained into one component, whose expectation is 0. Either one calls for a smaller `distance`, or `within` groups that compare fewer barcodes at once.

Error variants copy their barcode but for a few bases, so pairs of every compared barcode would agree as a few abundant barcodes and their errors do, and overstate chance by far. TREX's own 10x test output, most of whose 96 compared cloneIDs lie within 2 mismatches of one, expects 0.0002 chance matches beside 89 merged, where pairs of every compared cloneID would expect 2,211, about half of their 4,560 pairs; its largest clone holds 0.98 of the cells. TREX's Smart-seq3 output expects 0.30 beside 543 merged. Barcodes that correction links to no other are components of their own, so where it links none, the expectation is that of pairs of every compared barcode, which matches brute-force counts of random barcodes. Pairs of a barcode with itself would raise each position's chance of agreeing by about its complement over the barcodes, which counts where components are few: one to three barcode families of many errors would expect tens to hundreds of chance matches. Error variants farther than `distance` from their barcode stay components of their own, and lean the chances of agreeing towards their barcode.

```python
import warnings

diagnostics = ds.clones.load_clones(clones).diagnostics
expected = diagnostics["expectedChanceMatches"]
merged = diagnostics["mergedBarcodes"]
distinct = diagnostics["distinctBarcodes"]
# Chance alone would give at least one merge and half of those made, or link a
# tenth of the barcodes compared.
if (merged and expected >= max(1.0, 0.5 * merged)) or expected >= 0.1 * distinct:
    warnings.warn("Barcode correction merges unrelated barcodes; call again with a smaller distance")
# No clone of this experiment should hold more than a twentieth of its cells.
if diagnostics["largestCloneShare"] > 0.05:
    warnings.warn("One clone holds more cells than the experiment allows; correction may chain barcodes")
```

### Memory while calling

Calling checks the memory budget before each step that reads or allocates, and raises `MemoryError` before it reads what would not fit or writes anything. The steps hold about:

| Step | Bytes |
|---|---|
| Reading the records | 48 per record, 8 more while one is converted, the stored barcodes with 8 per distinct barcode, 16 per cell, and 300 per allowlisted barcode, with 4 per character of a fixed-width array |
| Saving the clones | 24 per cell for its group and the calls' flags and barcodes, each clone's label as a Python string, and what [saving clones](#read-export-and-plot-clones) holds, checked before reading as if every cell were a clone |
| Every rule | 40 per cell, and 18 per record until the records collapse into pairs |
| Looking up barcodes, for TREX's rules and the allowlist | 32 per record, 120 and 4 per character for each stored barcode decoded, and 1 per stored barcode |
| Correction | 96 per molecule or pair kept, 64 per distinct barcode of a block with its text, and what the search holds, within what is left; the expected chance matches then encode the barcodes as the search does and hold, while one barcode per component is chosen, 57 bytes and its length for each barcode of a length, or under TREX's rules what the search holds for the positions of no base, 64 bytes for each cloneID and 4 KiB for each position; under `"levenshtein"`, 48 bytes for each barcode of a group and library, 2 KiB for each position of each length, 8 bytes for each pair of positions of two lengths, 3 bytes for each position for each barcode chosen of a length, and 160 bytes for each state of the edit distances they follow |
| Counting the distinct barcodes, without correction | 48 per molecule or pair kept |
| Collapsing records into pairs | 136 per molecule or pair kept |
| Pair filters and sets | 48 per pair held, 80 per pair while TREX's rules or the pair filters run, and 16 per pair, beside the sets themselves, while the cells' sets are built |
| Grouping | 180 per cell and 16 per item of a set to find identical sets; 64 per item and 40 per node for the Jaccard graph, beside 16 per edge; 80 per edge and 48 per node to prune or find doublets; 80 per node for the components; 40 per cell to number the clones |

On one thread, for 1,000,000 cells that fall in 200,000 clones of five barcodes of 28 bases, LARRY's preset calls clones from 14.7 million molecules, with a PCR error in every fourth cell, in 14 s, and accepts a budget of 2,976 MiB, at which tracemalloc records a peak of 2,325 MiB. CellTag's preset calls them from 4.8 million pairs in 4.9 s within 1,296 MiB. TREX's preset takes 18 minutes on the same molecules with three in ten cloneIDs cut at one end, nearly all of it correcting the cut cloneIDs, whose 20 to 27 bases within 5 mismatches meet many candidates by chance; its graph, pruning and doublets take 30 s.

## Import lineage recorder alleles

Lineage recorders such as DARLIN, PEtracer and DNA Typewriter write heritable edits into target arrays in the genome, so cells that share an edit descend from the cell in which it arose. Their pipelines write each cell's alleles, the edits that each copy of an array carries, in tables of their own. {meth}`ds.clones.import_alleles <cytearc.datastore.namespaces.ClonesAccessor.import_alleles>` reads them, joins each row to a selected cell by its cell barcode and saves a `lineage_alleles` artifact. {meth}`ds.clones.load_alleles <cytearc.datastore.namespaces.ClonesAccessor.load_alleles>` reads it as a {class}`~cytearc.clones.LineageAlleles` record, {meth}`ds.plots.recorder_states <cytearc.datastore.plot_accessor.DataStorePlotAccessor.recorder_states>` draws each site's share of cells in each state, and {meth}`ds.clones.export_characters <cytearc.datastore.namespaces.ClonesAccessor.export_characters>` writes the character matrix from which [Cassiopeia](https://doi.org/10.1186/s13059-020-02000-8) and [LAML](https://doi.org/10.1101/2024.03.05.583638) build trees. CyteArc builds no trees itself.

```python
# The character matrix of one PEtracer embryo, whose cell IDs the store shares.
alleles = ds.clones.import_alleles("E7.5-R1_characters.csv.gz", cell_selection=cells)
ds.plots.recorder_states(alleles)
ds.clones.export_characters(alleles, "characters.zarr")  # for Cassiopeia
ds.clones.export_characters(alleles, "characters.csv", format="laml")
```

### Sites and states

An integration is one copy of a target array, named as its tool names it: a DARLIN array (`CA`, `TA` or `RA`), a Cassiopeia `intBC`, a PEtracer `intID` or a DNA Typewriter `TargetBC`. An integration's sites are of one of three kinds.

| Kind | Sites | Recorders |
|---|---|---|
| independent | Sites edited independently, each with a fixed lookup of marks | [PEtracer](https://doi.org/10.1126/science.adx3800)'s `RNF2`, `HEK3` and `EMX1`, each written with one of eight 5-base insertions; the columns of a character matrix |
| ordered | Slots edited one after the other, so slot k is edited only after slots 1 to k-1 | [DNA Typewriter](https://doi.org/10.1038/s41586-022-04922-8)'s tapes |
| indel | The edits of a CRISPR target array, as text | [DARLIN](https://doi.org/10.1016/j.cell.2023.09.019) and [CARLIN](https://doi.org/10.1016/j.cell.2020.04.048) alleles over a whole array, whose one site has no name; Cassiopeia's cut sites `r1` to `rN`; GESTALT-family targets |

Each site of a cell is in one of five states: unedited; a mark, an edit that the recorder names; an unknown mark, an edit that a lookup does not name or that the tool could not resolve, such as an insertion outside PEtracer's lookup, written as its sequence, or a placeholder such as PEtracer's `!` or CARLIN's `X`, which names no allele, so cells that hold it share no edit; missing, a site that was not read or a read whose bases are ambiguous; and not applicable, a slot beyond a contracted tape or an integration that the cell's embryo lacks. Each format writes the states with tokens of its own, which the table of formats lists. Ordered slots follow the prefix rule: a slot before an edited slot was edited, so one that reads as unedited there is missing, and a slot at or beyond its integration's length is not applicable, whatever the file holds there. A warning counts such slots that hold an insertion, and the `n_beyond_length` attribute keeps the count.

### Formats

`import_alleles` recognises each file's format from its contents unless `format` names it. A character matrix of another tool needs `format="characters"`, and any other table `format="table"` with `columns`. Files of the formats `darlin`, `cassiopeia`, `petracer`, `typewriter_chains` and `table` read R's row names as [clone tables](#import-clones-from-a-file) do, so one whose rows all end in a delimiter is refused. A character matrix whose header leaves out the cells' column, as R's `write.table` writes it, takes its cells from its rows even where every row ends in an empty state, as PEtracer leaves a site of an integration that the cell's embryo lacks; a matrix whose rows do end in a delimiter after a named cells' column is refused, by its empty last states or, in PEtracer's matrices, by the cells' column read among the characters. DNA Typewriter's table of 2022 is read by position, and its rows may hold empty slots beyond those its header names, as 16 rows of the published table do. A space-separated matrix keeps an empty state at the end of a row only where its fields are quoted, as R's `write.table` quotes them by default, so a PEtracer matrix whose first column is named as a character, as such a file reads without quotes, is refused.

| `format` | File | Integration | Tokens |
|---|---|---|---|
| `"carlin"` | [CARLIN](https://gitlab.com/hormozlab/carlin)'s `AlleleAnnotations.txt` and `AlleleColonies.txt`, as CARLIN and Custom_CARLIN write them for every published DARLIN dataset; either file, or the folder that holds them, names the pair | named by `integrations` | `[]` unedited; `X` and `?` an unknown allele; other text an allele in CARLIN's notation, such as `23_268del,55_248del` |
| `"darlin"` | The `darlin` CLI's `step4_final.tsv`, with the reads of each row summed over its molecules in `step4_annotated.tsv` beside it | named by `integrations` | `[]` unedited, and empty text, as darlin-core wrote it before 1.1.0 |
| `"cassiopeia"` | Cassiopeia's allele tables before 3.0: `cellBC`, `intBC`, `r1` to `rN`, `UMI` and `readCount` | `intBC` | text that holds `None`, such as `GCCGA[None]AAAAT`, uncut; empty, `NaN`, `nan`, `NA`, `NC` and `Missing` missing |
| `"cassiopeia_parquet"` | Cassiopeia 3.0's `<name>.alleles.parquet`, the rows whose `pass` is true, with its sites, edit models and lookups as its `attrs` record them, or its categorical columns as sites; a table of one row per UMI is refused | `intBC` | indel sites: text that holds `None` uncut; sites of known insertions, as PEtracer's: a label of the lookup a mark, `None` unedited, another insertion or a deletion, written `D` and its length, an unknown mark; tape slots in order, as DNA Typewriter's; a null or empty site missing |
| `"petracer"` | PEtracer's long tables of 2025: `intID`, `cellBC`, `RNF2`, `HEK3`, `EMX1`, `UMI` and `readCount` | `intID` | a site's 5-base insertion, one of its eight marks or, otherwise, an unknown mark; `None`, and `CTTGGG` at `EMX1`, unedited; empty, and an insertion that holds `N`, missing |
| `"petracer_characters"` | PEtracer's character matrices of [2026](https://doi.org/10.1016/j.cell.2026.09.050): a cell column, then `intID<N>-<site>` columns of its three sites | the column name before its last hyphen | `*` unedited, `-` missing, `1` to `8` a mark, `!` a placeholder of another insertion, empty not applicable |
| `"typewriter"` | DNA Typewriter's table of 2022: `Cell`, `TargetBC`, `Site1` to `Site5` and `nUMI` | `TargetBC` | a slot's 6-base insertion; blank unedited; an insertion that holds `N` missing |
| `"typewriter_chains"` | DNA Typewriter's per-cell tables of [2026](https://doi.org/10.1126/science.ael0508): `cell` or `cell_id`, one column per 12-base integration barcode, and quality columns, which are not read | the column | the symbols written, joined by `-`, with the slots after them unedited, `-` no edit and empty no call; or six slots joined by `\|`, `U` unedited and `NA` no call |
| `"treedata"` | A TreeData `.h5td` file or `.zarr` store's `obsm['characters']`: a dataframe, or an array whose columns `uns['embryo_characters']` names, as in PEtracer's published files, which are read as its character matrices | the column name before its last hyphen | the states that `uns['missing_state']` and `uns['unmodified_state']` name, or Cassiopeia's: `-1`, `-` and `NA` missing, `0` and `*` unedited; `!` a placeholder of an unknown mark |
| `"characters"` | A delimited character matrix, as Cassiopeia's `to_csv`, LAML and ConvexML write it | the column name before its last hyphen, or the whole name | `0` and `*` unedited, `-1`, `-`, `?` and `NA` missing, other text a state, a whole number written as a float, such as `3.0`, as the number |
| `"table"` | Any table, read by the columns that `columns` names | a column, or named by `integrations` | the tokens that `columns` names; text that pandas reads as missing is missing |

Text is read as written, so Cassiopeia's `None` stays an uncut site where pandas would read it as missing. A token that a format does not write is refused with the lines that hold it: text other than bases, `None` or nothing at a site of PEtracer's 2025 tables, and in a character matrix the text that pandas reads as missing, such as `None`, `nan`, `null` and `#N/A`, other than its own `NA`, and negative numbers other than `-1`, where other tools write other states. An insertion that holds `N`, an ambiguous base, names no allele, so it is missing, as PEtracer's GEO tables of 2025 hold some at `EMX1` and `HEK3`. A format named by `format` that does not fit the file is refused, as is a TreeData array whose columns have no names, whose integrations would be unknown. A row of a tool's table without its counts is refused as a row cut short, as is a gzip file cut short, and a file that holds no row, such as a header alone, adds no records. CARLIN's files must have equal lines and list each barcode once, and a bulk run, which lists UMIs instead of cells, is refused: by the run type that `Results.txt` beside the pair records, or, without it, by tags shorter than 14 bases. Several alleles of one cell and integration are kept, as the `darlin` CLI, Cassiopeia and PEtracer write them; only the CARLIN pair lists one per cell. No output of the `darlin` CLI is public, so its reader follows the CLI's documented columns and checks that each row's UMIs are the molecules that `step4_annotated.tsv` lists for its cell and lineage barcode. DNA Typewriter's 2026 code is licensed under the GPL, so its tables are read from their documentation and published files only. Reading Cassiopeia's parquet tables needs polars 1.30 or later, which reads the attributes that Cassiopeia keeps in a parquet file's metadata.

### Integrations

A CARLIN pair and a `darlin` table hold one array without naming it, so `integrations` names it: one for every file, or one per file. It is never inferred, from folder names or otherwise. Files of different arrays with the same `barcode_prefix`, `barcode_suffix` and `barcode_pattern` hold the arrays of one library, such as DARLIN's three, and each must share a cell with another, so a file of another library's barcodes is refused. For DNA Typewriter's tapes, `integrations` maps each integration shorter than the recorder's arrays to its number of slots, as the HEK293T line of 2022 has three tapes of four slots and one of two; a tape that no row or column of the import names is refused, as a misspelt name would leave its tape at full length.

```python
# DARLIN's three arrays of one skull library, each a CARLIN pair.
alleles = ds.clones.import_alleles(
    ["LL837-skull-CA_S2", "LL837-skull-TA_S7", "LL837-skull-RA_S12"],
    cell_selection=cells,
    integrations=["CA", "TA", "RA"],
    barcode_suffix="-1",
)
tapes = ds.clones.import_alleles(
    "41586_2022_4922_MOESM5_ESM.csv",
    cell_selection=cells,
    integrations={"TGGACGAC": 4, "TTTCGTGA": 4, "TGGTTTTG": 4, "TTCACGTA": 2},
)
```

Rows that name no integration, as PEtracer leaves `intID` empty where an intBC is not on its whitelist, are left out, counted in the `n_rows_without_integration` attribute and named in a warning, which a reused import gives again.

### Tables of other tools

`format="table"` reads any table by the columns that `columns` maps to roles: `"cell"`; `"integration"`, unless `integrations` names the file's one; the sites, as `"sites"`, a column per site, or as `"allele"`, with `"site"` when each row holds one site, with `"separator"` when one field joins the sites, which are then numbered from 1, or alone for one site of no name; `"umis"` and `"reads"`; `"kind"`, `"indel"` by default, `"independent"` or `"ordered"`, with `"slots"` for the slots of ordered sites; and the tokens `"unedited"`, `"missing"`, `"not_applicable"` and `"unknown"`, each a text or a list of texts, the unknown tokens placeholders that name no allele. A separator splits every row's allele into as many sites as the file's first allele, and a row of another number is refused; ordered slots need `"slots"`, and a row may fill fewer. A row without an allele reads its sites as missing. Text that pandas reads as missing, such as `NA`, `NaN`, `None` or an empty field, is missing unless a role names it, and a count that holds it is unknown. When each row holds one site, the rows of one cell and integration form one allele, and a site listed twice for one cell and integration is refused. MosaicLineage's `UMI_count` counts the cells of an allele, not its UMIs, so it is no `"umis"` column.

```python
# scGESTALT joins its ten targets with "-": NONE is unedited and UNKNOWN missing.
alleles = ds.clones.import_alleles(
    "GestMaster.txt.gz",
    cell_selection=cells,
    format="table",
    columns={
        "cell": "CellBarcode",
        "allele": "HMID",
        "separator": "-",
        "unedited": "NONE",
        "missing": "UNKNOWN",
    },
    integrations="HMID",
)
```

### Join, reuse and memory

Cell barcodes join cells as the other importers join them: a cell's ID is its file's `barcode_prefix`, the cell barcode and `barcode_suffix`, and `barcode_pattern` keeps the part that its `barcode` group matches. Barcodes that match no selected cell, such as the droplets that CARLIN calls against the whole 10x list, are counted in the artifact's `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` attributes and named in a warning, which a reused import gives again. Saved alleles are reused when each path and the files read beside it, such as a CARLIN pair's annotations and a `darlin` table's molecules, hold the same bytes, every file of a `.zarr` store included ({ref}`reuse of imports <import_reuse>`).

A file is read in chunks of about a twentieth of the memory budget, sized from the longest of the file's first lines. The import checks its peak against the budget before it reads a file and after each chunk: the selected cells' lookup and IDs; one chunk; a CARLIN pair's annotation lines and the keys that find a barcode listed twice; a `darlin` table's sums of reads per cell and lineage barcode; the records held; at the merge of the records into one table, twice their encoded bytes and, for each distinct integration, site and allele of a chunk, 150 bytes and, for each byte of its text, 1 byte and three times the bytes that Python holds a character of the chunk's widest text in: 1 up to U+00FF, 2 up to U+FFFF and 4 beyond; at the write, twice their encoded bytes, since the memory that the merged chunks held stays resident for reuse, 44 bytes per record and one of io's default buffers for each write that Zarr runs at once, up to its async concurrency: 8 KiB each before Python 3.14 and 128 KiB since; and 8 MiB of buffers whatever the size. It refuses with `MemoryError` before it writes. Loading counts 120 bytes per record, 160 per distinct text and cell, and for each stored byte of the distinct integrations, sites and alleles 2 bytes and twice the bytes of a character of the widest, as decoding them holds; the export counts 260 per record, 150 per distinct allele, the decoded text as loading does, 1,000 per character, 2 per byte of the alleles that a Cassiopeia store writes, with Zarr's write buffers as the import counts them, and, once the matrix's size is known, 4 bytes per cell and character; LAML's text is written as many cells at a time as a twentieth of the budget holds, at 40 bytes per state.

On one thread, PEtracer's character matrix of 100,000 cells by 300 sites, 30 million records, imports in 22 s and holds 4,214 MiB above its start, within its check of 4,264 MiB, while tracemalloc traces its peak at 2,787 MiB. DNA Typewriter's chains of 400,000 cells on 11 tapes, 26.4 million slots, import in 20 s and 3,547 MiB, against a check of 3,817 MiB; Cassiopeia's allele table of 1 million rows, 3 million records, in 6.7 s and 453.0 MiB, against 453.2 MiB; and a CARLIN pair of 200,000 cells in 1.5 s. Under a budget of 1 GiB, the PEtracer import refuses after 6 s, before it writes anything. `load_alleles` reads its 30 million records in 1.6 s and `site_states()` counts them in 1.4 s; exporting them takes 14 s for Cassiopeia and 12 s for LAML, and Cassiopeia's table of 1,200 characters exports in 19 to 22 s to Cassiopeia and 6 to 11 s to LAML. The peaks that tracemalloc records stay within the smallest budget that each step accepts, on a wide character matrix of 1,000 cells by 300 characters, 30,000 CARLIN alleles of about 150 characters each, a long table of 20,000 cells among 300 integrations, whose matrix is mostly missing, and alleles of 20,000 characters, ASCII, ending in a character up to U+00FF, up to U+FFFF or beyond it, or ASCII but for one that starts with an emoji.

### Read and plot recorder alleles

A `lineage_alleles` artifact holds one record per cell, integration, site and allele row, ordered by cell, file and the row's line, with each distinct integration, site and allele stored once. A record's `allele` is the edit of a mark or an unknown mark as the file writes it, and missing for the other states, whose tokens are notation, and for placeholders of unknown marks, such as `!` and `X`, which name no allele; its `umis` and `reads` are its allele row's, -1 where the file counts none, and its `row` is the row's line in a text file, counting from 1, or its position in another file. The canonical `n_edits` holds each cell's sites with a mark or an unknown mark, over its allele rows, so `ds.select_cells(alleles, low=0)` keeps the cells with an edit; `n_integrations` and `n_records` hold its integrations with a site read and its records. The `files` attribute records each file's format, tool and the pipeline whose notation its alleles follow, which an allele bank must share: CARLIN's MATLAB notation and darlin-core's place some events one base apart.

`load_alleles` returns a {class}`~cytearc.clones.LineageAlleles` record. Its `cells` table holds one row per selected cell, indexed by cell ID, with the per-cell counts. Its `records` table holds one row per record: `cell`, `file`, `integration`, `site`, `state` and `allele` as categoricals, `umis` and `reads` as nullable integers, and `row`, the allele row's line in its file: a table's header is line 1 and each row takes the next line, so blank lines and line breaks within quoted fields do not count; a CARLIN pair gives the line of the allele in both its files, and a parquet table or TreeData file the row's position from 1. Its `files` table names each file's `format`, `tool`, `pipeline`, `kind`, its one `integration` where it holds one, and its `rows` and `records`. `site_states()` counts the cells in each state at each site.

`ds.plots.recorder_states(alleles)` draws one bar per site, named `{integration}-{site}`, stacked by the share of the site's cells that hold a mark, an unknown mark, an unedited site or no read. A site's cells are the cells with a record, of which those whose integration lacks the site are left out; a cell with several alleles at a site counts once, in the first of these states, and a cell without a record there is missing. With more than `max_sites` (default 60) sites, the sites read in the most cells are drawn and the table keeps every site. Its table is `site_states` (`integration`, `site`, `state`, `cells`, `share`).

### Export character matrices

`export_characters(alleles, path)` writes a character matrix: one row per cell and one column per site of an integration, named `{integration}-{site}`, or after its integration when the site has no name, as a DARLIN array's allele does. A state is 0 for an unedited site and missing where a site is missing or not applicable. A mark written as a whole number from 1 without leading zeros, such as PEtracer's marks `1` to `8` or a character matrix's state, is its own state, and a character's other marks, by their text, then its unknown marks that name their allele, such as an insertion outside PEtracer's lookup, take the states after the largest such number, so each state names one allele of its character in every cell, and a character matrix keeps its states through an import and an export. A placeholder of an unknown mark names no allele, so it is missing: cells that hold PEtracer's `!` or CARLIN's `X` share no edit. `unknown="missing"` writes every unknown mark as missing. Two integrations whose characters would share a name, and states beyond the int32 range, are refused. Where a cell holds several allele rows of one integration, the row of most UMIs, then most reads, then most sites read is exported, as Cassiopeia keeps each integration's most abundant allele, and a warning counts such cells and integrations. Ordered slots export one character per slot, so the prefix rule is left to the tree builder's model. Cells that read no character and characters that no cell reads are left out, so 1,032 of the 1,342 cells of PEtracer's E7.5-R1 matrix are written. It returns the number of cells written.

`format="cassiopeia"`, the default, creates a TreeData `.zarr` store, which must not exist: `obsm['characters']` with one int64 column per character, `uns['missing_state']` -1, `uns['unmodified_state']` 0, and `uns['character_states']`, the `character`, `state` and `allele` of each edited state, beside an empty `var` and no trees. `format="laml"` writes LAML's CSV, a header `cell,<characters>` and one row per cell with `?` for missing, and beside it `<stem>_priors.csv`, headerless rows of a character's position from 0, a state and its probability, each edited state's share of the cells with an edit at its character, and `<stem>_states.csv`, the `character`, `state`, `allele`, `kind` and `cells` of each edited state; it replaces files of those names, and refuses cell and character names that hold a comma, semicolon, colon, bracket, quote or space, which LAML's CSV and the Newick trees it writes cannot hold. Cell IDs are this store's.

```python
ds.clones.export_characters(alleles, "characters.zarr")
ds.clones.export_characters(alleles, "characters.csv", format="laml", unknown="missing")
```

Cassiopeia 3.0.0, built from its repository at commit 2867d2d, and treedata 0.4.0 read the stores exported from PEtracer's E7.5-R1 characters (1,032 cells by 99 characters), ConvexML's matrix (400 by 39), DARLIN's skull library (90 cells on three arrays), the DNA Typewriter subset of 2022 (90 cells by 79 characters) and the Cassiopeia subset of 3432_NT_T1 (30 by 30), with every character an int64 column, `missing_state` -1 and `unmodified_state` 0, and their neighbor-joining and UPGMA solvers return trees whose leaves are the exported cells. LAML 1.0.5, with its Scipy solver, reads the CSV and priors of the three smaller matrices and of 100 cells of the two larger ones, and returns trees whose leaves are the exported cells, from its own starting tree and from Cassiopeia's neighbor-joining tree. A character that no cell edits has no rows of priors, and LAML reports it as uninformative. LAML asks for a MOSEK licence before it starts, as its documentation says, whichever solver it is given.

## Call clones from recorder alleles

A recorder's edits are inherited, so cells that share an allele descend from the cell in which it arose, unless it arose more than once: CRISPR arrays repeat their most likely edits in unrelated cells. {meth}`ds.clones.allele_clones <cytearc.datastore.namespaces.ClonesAccessor.allele_clones>` links the cells that share a rare allele, at any character, and saves the connected components as a `clones` artifact, by the rules of [DARLIN](https://doi.org/10.1016/j.cell.2023.09.019) or of [CARLIN](https://doi.org/10.1016/j.cell.2020.04.048), which differ in how they decide that an allele is rare.

```python
import pandas as pd

# DARLIN's three arrays of one skull library, each a CARLIN pair, and the bank of each array.
alleles = ds.clones.import_alleles(
    ["LL837-skull-CA_S2", "LL837-skull-TA_S7", "LL837-skull-RA_S12"],
    cell_selection=cells,
    integrations=["CA", "TA", "RA"],
    barcode_suffix="-1",
)
banks = {array: pd.read_csv(f"reference_merged_alleles_{array}_Gr.csv") for array in ("CA", "TA", "RA")}
clones = ds.clones.allele_clones(
    alleles, method="graph", frequencies=banks, max_frequency=2e-5, max_samples=1, max_alleles=3
)
ds.clones.load_clones(clones).cells.head()  # clone, clone_size, n_alleles
```

An allele is one character's edit: the text of a mark, or of an unknown mark that names its allele, at one site of one integration, such as a DARLIN array's `23_268del,55_248del` or a PEtracer site's insertion. The same text at two characters is two alleles. The unedited state, placeholders of unknown marks such as CARLIN's `X` and PEtracer's `!`, and missing or inapplicable sites name no allele, so they never link cells. A cell that holds an allele twice at a character, in two allele rows, holds it once.

- `method="graph"` follows DARLIN: an allele is rare when its frequency is below `max_frequency` and it was found in at most `max_samples` of its bank's samples. DARLIN's paper linked cells through alleles below 2×10⁻⁵ found in at most one of its bank's three mice, and below 2×10⁻⁴ where it expected about 100 clones, in embryos labelled at E10; its tutorial uses 2×10⁻⁵ and one mouse. The comparison is strict, so an allele at exactly `max_frequency` does not link. DARLIN's paper scales each array's frequencies by a factor, 1 for CA and fitted for TA and RA so that one generation probability gives the arrays similar homoplasy, while its banks' `normalized_count` and its tutorial carry no factor, and neither does `allele_clones`: scale a bank's `normalized_count` to apply one. The graph needs `max_frequency`, `max_samples` or both, since a graph through every allele joins clones through their most frequent edits.
- `method="p_clonal"` follows CARLIN: an allele of bank rate λ among the M cells of its group with an edit at its character arises more than once with probability p_clonal = (1 − e^(−λM) − λM·e^(−λM)) / (1 − e^(−λM)), the chance that a Poisson count of mean λM exceeds 1 given that it exceeds 0, adapted from CARLIN's MIT-licensed `Bank.m`. An allele is rare when its p-value survives the Benjamini-Hochberg procedure at `fdr`, by default CARLIN's 0.05, over every allele of its group, those of every array in one family, and when it passes `max_samples`. M counts the cells with an edit, placeholders among them, as CARLIN counts the cells with an edited allele. p_clonal approaches λM/2 as λM approaches 0, and CyteArc computes it from the Poisson tails, which keep that end exact. It grows with M: among a million edited cells an allele of rate 10⁻⁶ has λM = 1 and p_clonal 0.42, so in large experiments only the alleles missing from the bank, of rate 0 here, survive. CARLIN gives those alleles the mean rate of the alleles its bank never observed, and its paper notes that beyond about 35,000 cells even that rate is expected to mark several progenitors at 0.05, so there CARLIN's test keeps no allele.

`max_alleles` leaves the cells of a clone whose cells hold more distinct rare alleles than it allows without a clone, as DARLIN keeps clones of at most three: a cell holds at most one allele per array, so a clone of more is likely two clones that a homoplasic allele joined. A clone's alleles are the rare ones, those that link, as the joint clones of DARLIN's published tables count them. Clones of one cell are kept: a cell with a rare allele that no other cell shares is a clone of its own.

`within` names cell metadata columns whose labels keep clones apart, such as the mouse. Each group is called as it would be alone: its frequencies without a bank, its M and its family of Benjamini-Hochberg tests are its own, and two groups never share a clone. Every cell with an edit needs a label.

### Allele banks

DARLIN measured how often each allele arises in bulk libraries of granulocytes of three mice, where a clone's expansion barely shows, and its tables of each allele's `normalized_count`, its share of the bank's edited UMIs, and `sample_count`, the mice that hold it, are what `frequencies` reads: a mapping from each character's name, its integration, or `{integration}-{site}` for a named site as `export_characters` names it, to a DataFrame with the columns `allele` and `normalized_count`, and `sample_count` for `max_samples`. Other columns are not read. CARLIN's bank, a MATLAB object, reads the same way once each allele's rate, its share of the bank's edited transcripts, is exported as `normalized_count`. Every character with an allele needs a bank, and a character without one may have one; a bank that names no character of the alleles, an allele listed twice, a frequency outside 0 to 1 and a sample count that is not a whole number are refused. Neither DARLIN bank may be redistributed, so CyteArc ships none.

An allele missing from its bank counts as rare, with frequency and rate 0, as DARLIN's paper counts the de novo alleles that made up about 80% of its experiments' alleles. CARLIN gives such alleles the mean rate of the alleles that its bank's model never observed, which a table of rates does not hold, so p_clonal here gives them 0 and they always survive.

A bank fits only the alleles annotated by the pipeline that annotated it. DARLIN's two banks of the same granulocytes, MosaicLineage's in CARLIN's MATLAB notation and the `darlin` CLI's in darlin-core's, share 43,836 of their edited CA alleles, about half of each bank's edited UMIs, since the two notations place some events one base apart. A bank in the other notation lists few of the alleles, which then count as rare and join clones through frequent edits. The artifact's `bank_coverage` attribute reports, for each character, its `alleles` and the `alleles_in_bank`, and its `observations`, the pairs of a cell and an allele, and the `observations_in_bank`; `load_alleles(alleles).files` names each file's pipeline.

Without `frequencies`, an allele's frequency is the share of the cells of its group with an edit at its character that hold it. Every allele can be measured this way, but clonal expansion inflates it, so the alleles of large clones look as common as alleles that arise often, and an allele that every edited cell of a character holds is never below `max_frequency`. p_clonal needs a bank: a rate measured in the cells gives every allele seen in n cells a mean λM of n, and so a p_clonal of at least (e − 2)/(e − 1), about 0.42, so no allele would survive.

### The allele clones artifact

The clones are labelled by size, `clone_1` the largest, and clones of one size by their first cell. Beside the `values`, `codes` and `clone_size` of every `clones` artifact, the artifact holds `n_alleles`, each cell's distinct rare alleles, which cells of a clone that `max_alleles` left out keep. Its parameters record the method and its rules, with `fdr` for `"p_clonal"` only, the `within` columns, and the fingerprints of the banks' sorted alleles, frequencies and samples and of the cells' `within` labels, so the same banks in another row order reuse saved clones. Its `rule_counts` attribute counts the `alleles` tested, one per group, character and allele, those `alleles_too_frequent` or `alleles_not_significant` and then, among the others, those `alleles_in_too_many_samples`, each allele counted by the first rule that removes it, the `rare_alleles` and the `cells_with_rare_alleles`, and the `clones_with_too_many_alleles` and their `cells_in_clones_with_too_many_alleles`; `n_clones` and `bank_coverage` sit beside it.

### Memory while calling recorder alleles

Calling checks the memory budget before each step that reads or allocates, and raises `MemoryError` before it reads what would not fit or writes anything. The steps hold about:

| Step | Bytes |
|---|---|
| Checking the banks | 128 per row of a bank and 4 per character of its alleles, and 80 per row and 4 per character held after |
| Grouping the cells | 9 per cell, beside what the grouping of `within` holds |
| Saving the clones | 8 per cell for the calls' rare alleles, each clone's label as a Python string, and what [saving clones](#read-export-and-plot-clones) holds, checked before reading as if every cell were a clone, and again before writing with the clones' number |
| Decoding the alleles | 168 per distinct allele, integration and site, and for each byte of their text 2 and twice the bytes that Python holds a character of the widest in: 1 up to U+00FF, 2 up to U+FFFF and 4 beyond; 68 per text and that width per byte held after |
| Reading the records | 4 per record while the states are read, then 9 per record, while both columns of integrations and sites are read, and 80 per edited site; 24 per edited site held after |
| Finding each cell's distinct edits | 16 per cell while each cell's group is copied, and 88 per edited site |
| Testing the alleles | 96 per edit that names an allele, and 160 per allele of a group and character, beside 120 per allele and per row of its bank while a bank looks it up |
| Linking the cells | 40 per edit that names an allele, then the steps of `clones/calling.py` that link cells through any shared item |
| Counting each clone's alleles | 40 per rare allele of a cell |

The tests trace each step's peak against the bytes its check states at millions of records and two million edits, on PEtracer-like characters of 12,000 cells by 300 sites, dense or sparse in edits, on DARLIN-like arrays with banks and groups, on 100,000 cells that each hold an allele of their own on three arrays, on 400,000 cells in clones of four, on alleles of 5,000 characters ending in a character up to U+00FF, up to U+FFFF or beyond it, and on 100 alleles of 50,000 characters among 200 cells, and each call holds no more than the smallest budget it accepts. The calls get the budget less what the operation holds, its fixed bytes included. On one thread, calling a million DARLIN-like cells against banks of about 99,000 alleles per array takes about 3 s, and the smallest budget the call accepts is 372 MiB, at which tracemalloc traces a peak of 231 MiB. Calling PEtracer-like characters of 100,000 cells by 300 sites takes 19 to 37 s, and the smallest budget that call accepts is 3,072 MiB, at which the traced peak is 2,273 MiB. A fresh process also loads the compiled kernels, about 95 MiB resident at 300,000 cells, which the budget does not count and a second call does not repeat.

### DARLIN's published clones

DARLIN's experiment LL837 on [Zenodo](https://zenodo.org/records/11929508) (CC BY 4.0) holds five tissues of one mouse read on the CA, TA and RA arrays as CARLIN pairs, and MosaicLineage's tables of them: `df_sc_data_10.csv` gives each cell's allele per array at a read cutoff of 10, with the bank's `normalized_count` and `sample_count` joined, and the cell's joint clone. The pairs at that cutoff, imported into a store of the table's 24,712 cells, give every cell the table's allele on each array, all 32,793 of them. The table's valid alleles are the edited ones found in at most one mouse of the bank, so with the bank it joined, `method="graph"` and `max_samples=1` give 3,488 clones of 12,619 cells, exactly the connected components of the cells and their valid alleles that networkx finds. 3,466 of them, holding 8,359 cells, are MosaicLineage's joint clones exactly; MosaicLineage splits the other 22 into 221 joint clones, among them a component of 3,813 cells and 341 alleles. With `max_alleles=3`, 3,458 clones of 8,089 cells remain, and 3,453 of them are MosaicLineage's joint clones of at most three alleles. The other 5 hold cells that share a rare allele on one array and differ on another, which MosaicLineage puts in two joint clones, and MosaicLineage also keeps 138 joint clones of at most three alleles, 1,019 cells, out of the 30 components of more alleles that DARLIN's rule leaves out. These differences follow MosaicLineage's tables, not the connected components that DARLIN's paper describes.

The banks show how coverage reads. MosaicLineage's bank of all its mice, read at run time from its repository at commit 7660afb, lists 99.2 to 99.6% of each array's alleles and gives the same 3,458 clones under DARLIN's tutorial rules. Its granulocyte bank of three mice lists 14 to 23% of each array's alleles, held by 23 to 54% of their cells, and gives 3,447 clones, 3,224 of them the same. The `darlin` CLI's granulocyte bank, in darlin-core's notation, lists 0.8 to 13% of the alleles, held by 0.6 to 17% of the cells; the alleles it misses count as rare and join clones of 12,109 cells, which `max_alleles=3` then leaves out. CARLIN's p_clonal on MosaicLineage's granulocyte bank, with M the mouse's edited cells of each array, keeps 5,039 of the 5,464 alleles and gives 3,453 clones, 3,367 of them those of the graph on the same bank. In a container of 2 CPUs, the import took 0.4 s and each call 0.15 to 1.8 s, the first with its compilation.

## Read, export and plot clones

A `clones` artifact labels each selected cell with its clone, an empty label for a cell without one, which groupings treat as missing. `codes` numbers the clones by size, largest first, then by label in natural order, -1 for none, and `clone_size` holds the cells of each cell's clone, 0 for none. A source that gives probabilities adds `clone_probability`, NaN where none is given; a cell's rows that give it no probability leave the one its other rows give. Clones called from barcodes add `n_barcodes` and `doublet`, and clones called from recorder alleles `n_alleles`. The store keeps each cell's label at the width of the longest label, four bytes per character, so a very long label costs memory for every cell. Saving clones holds each cell's code as assigned and as stored, the size of its clone and a reference to its label, up to 45 bytes per cell while it builds them, and writes the labels a stored chunk of 100,000 cells at a time: the chunk at that width and Zarr's encoded copy of it, and from the second chunk on the one before, which Zarr's writing thread can still hold on a busy machine. Every source counts this before it saves and raises `MemoryError` before it writes anything when the clones would not fit the memory budget. `ds.select_cells(clones, include=["3"])` selects the cells of one clone, and plots colour cells by clone. [Read per-cell values](cell_aligned_kinds) lists the arrays.

`load_clones` returns a {class}`~cytearc.clones.Clones` record. Its `cells` table holds one row per cell, indexed by cell ID: `clone`, a categorical of the clone labels in rank order, `clone_size` and, where saved, `clone_probability`, the `n_barcodes` and `doublet` of barcode clones, the `ambiguous` flags of genotype clones, or the `n_alleles` of allele clones. Its `clones` table holds one row per clone, largest first: `clone` and `n_cells`. Its `source` names the operation that saved the clones, `"import_clones"`, `"clones_from_column"`, `"call_barcode_clones"`, `"variant_clones"`, `"genotype_clones"`, `"call_allele_clones"`, `"phylogeny_clones"` or, for clonotypes, `"define_clonotypes"`. Its `kept_apart` names the columns of `cells` whose labels the clones keep apart, which every summary measures apart: none for clones, and `("receptor_type",)` for clonotypes, whose `cells` table then holds each cell's receptor type as a categorical. Its `diagnostics` maps names to the numbers that the operation recorded about how it defined the clones: the [four numbers](#diagnostics) of clones called from barcodes, and none for other clones.

`export_clones(clones, path)` writes CoSpar's clone table, `Cell_ID,Clone_ID`, one row per cell with a clone, and `format="matrix"` writes a cells by clones Matrix Market file with `cells.tsv` and `clones.tsv` beside it, replacing files of those names. Both hold this store's cell IDs and the clone labels, and `import_clones` reads them back to the same clones; clone probabilities are not written. The table refuses labels that a clone table reads as no clone, such as `NA`, which the matrix keeps, and the matrix refuses labels and cell IDs that hold a line break, which the table keeps. A path ending in `.gz` is gzip-compressed. Each returns the number of cells written.

```python
ds.clones.export_clones(clones, "clones.csv")  # for CoSpar
ds.clones.export_clones(clones, "clones/matrix.mtx", format="matrix")
```

`ds.plots.clone_abundance(clones)` draws two panels: the number of clones of each size, on log scales, and the `top` largest clones (default 20). `group_by` names a cell metadata column, such as the sample or time point, and `groups` a cell-label artifact, such as a clustering; with either, the first panel draws one line per group, counting each clone's cells within the group, and the second stacks each clone's cells by group. Cells without a group are left out, and the plot raises `ValueError` when no cell with a group has a clone. Clonotypes are counted within each receptor type, so their groups are the receptor types, or each group's receptor types, named like `"s1, TCR"`. The largest clones are labelled by their IDs, and clonotypes by the junction amino acids of their representative cell's `VDJ_1` chain, or of its `VJ_1` chain without a VDJ chain, such as `CASSLGQAYEQYF`; clonotypes defined by `receptor_arms="VJ"` are named by the `VJ_1` chain first, which their cells share. Where two of the clonotypes drawn share a junction, each label also shows the clonotype's number, the rank in its ID, such as `CAVA 1` and `CAVA 3`, and for clonotypes the panel's title reads `Largest 20 clonotypes`, or `Largest 20 clonotype clusters` for clonotypes that a distance metric linked. Its tables are `sizes` (`group`, `clone_size`, `n_clones`) and `top` (`clone`, `label`, then the cells of each group, or `cells`).

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

The diversity of a population of cells is how many clones it holds and how evenly its cells spread over them. {meth}`ds.clones.diversity <cytearc.datastore.namespaces.ClonesAccessor.diversity>` measures it for groups of cells: the combinations of the labels of the `group_by` cell metadata columns, such as the sample, and of a `groups` label artifact, such as a clustering. `cell_selection` keeps only some cells, such as one cell type. Cells without a clone or a label are left out, and a clone's size in a group is its cells there. Nothing is saved: the result follows from the clones and the labels in about 0.5 s for a million cells.

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

The implementation follows the papers rather than the code of iNEXT, which is licensed under the GPL, and validates against its outputs. On two small abundance vectors, the clonotypes of the four pseudo samples of both 10x runs above (11,523 to 11,745 cells each in the CD8+ run), all 46,526 CD8+ cells together, and six edge cases (no singletons, no doubletons, every clone a singleton, a lone singleton, one clone, two cells), the Hill numbers of iNEXT 3.0.2's curves equal CyteArc's within a relative 5.6e-11 at every size, rarefied, observed and extrapolated, the asymptotic estimates within 1.3e-12, and `estimateD` by size within 5.6e-11 and by coverage within 2.1e-7, where iNEXT's root finding stops within 1.7e-8 of the coverage level; the coverages along the curves differ by at most 1.8e-11, and the default sizes are iNEXT's. Against values computed in 60-digit decimal arithmetic, CyteArc is the closer where the two differ most. CyteArc's rarefied coverages are within 1.3e-16 of those values and iNEXT's up to 1.8e-11 away, because CyteArc computes the chance of missing a clone as a product of ratios rather than from log-gamma functions. At the largest differences in Shannon diversity, CyteArc's relative errors are 5.6e-15 rarefied, 3.6e-15 asymptotic and 1.7e-11 extrapolated, against iNEXT's 3.1e-11, 1.2e-12 and 7.2e-11. Extrapolation errs most for both, because its rate is computed from the difference between the nearly equal diversities at n and n - 1 cells. The indices equal scRepertoire 2.8.0's `clonalDiversity` without bootstraps for all twelve of its metrics, immunarch 0.10.3's `repDiversity` (`chao1`, `d50`, `div`, `gini`, `gini.simp`, `inv.simp` and Hill numbers of orders 1 to 6) and vegan 2.7.6's `estimateR` (`S.chao1`, `S.ACE`), within 6e-14, on the same pseudo samples and edge cases. On the same million simulated cells in eight samples, `diversity` and `rarefaction` each take about 0.5 s, holding at most 115 MiB of arrays. Split into 2,000 groups of about 450 cells, such as samples by clusters, the cells take about 0.9 s for `diversity` and 1.1 s for `rarefaction`. Groups that hold equal clone sizes are measured once, so with a label for each of 200,000 cells, which makes 179,775 groups of one cell, `diversity` takes about 1.4 s.

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
| The labels of the largest clonotypes in `clone_abundance` | Three times the stored width of each clonotype's ID, 2 per clonotype and 400 per clonotype drawn; then, a chunk of each receptor array at a time, the representatives' slots, junction codes and bounds, and 200 per junction and 7 per byte of its text |
| Summarizing each group's clone sizes, for `diversity` and `rarefaction` | 56 per count of a clone's cells in a group, and 128 per distinct clone size of a group beyond its first |
| Rarefaction | 192 per sample size, and 128 per row of the curves and 8 more per row for each key column |
| Comparing groups | 250 per pair of groups and 24 per entry of the groups-by-groups matrices, 40 per clone of each group, and the clones read |
| Following clones | 100 per row of the table and 16 more per row for each key column, a byte per chosen clone and group, 40 per clone of each group, and the clones and labels read |
| Measuring coupling | 250 per pair of groups, 40 per product of two groups, 80 per clone of each group and 12 per cell with a clone; for each thread of the null 4 per cell with a clone, 8 per product of two groups and 32 per group; 8 per pair for each permutation measured at once; 40 per clone of each group and the clones read |
| Checking that no cell is early and late | 40 per cell of the two selections, and the clones read |
| Measuring fate bias | 64 per early cell and 24 more for each fate, 40 more per early cell to write, and what writing holds as for [saving clones](#read-export-and-plot-clones), each early cell's fate stored at four bytes per character of the longest fate name; 80 per fate of each clone with late cells, 1,000 per fate, 24 per clone of each fate, and the clones read |
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

{meth}`ds.clones.track <cytearc.datastore.namespaces.ClonesAccessor.track>` follows clones across groups, such as time points: by default the `top` (10) with the most cells in each group, together, as scRepertoire's `clonalCompare` chooses them, with ties going to the larger clone overall; `labels` names them instead. Its `cells` table holds each chosen clone's cells and share of the cells with a clone of every group, zeros included, so a clone that expands between time points or spreads between cell types shows it. A table that the memory budget cannot hold raises a `MemoryError`, which a smaller `top` or fewer groups avoid. {meth}`ds.plots.clone_tracking <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_tracking>` draws one line per clone across the groups, in their natural order or in `order`, which may name time points as numbers, and labels each line as `clone_abundance` labels the largest clones, so a clonotype by its representative's junction; on a figure at most a page wide, its legend lists at most 80 clones, the largest first, and its axis names at most 40 groups; its table is `cells`.

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

For each clone with late cells, it holds their count in each fate and a one-sided Fisher test per fate: for a clone of `N` late cells, `x` of them in a fate of `K` of the `M` late cells with a fate, those without a clone included, the p-value is the chance of at least `x` in a draw of `N` of the `M` cells. Clonotypes keep receptor types apart, so a late cell without a clonotype has no fate, and each clonotype is tested among the late cells of its receptor type that have one; a clonotype whose early and late cells span receptor types, which `within_receptor_type=False` allows, is refused, as the {doc}`repertoire` describes. This is the test that CoSpar's `clonal_fate_bias` makes with `alternative="less"` on the late cells, which computes this upper tail where its `"greater"` computes the lower one; given every cell, CoSpar counts the early cells in `M` too. The p-values of each fate are adjusted by the method of Benjamini and Hochberg over the clones with late cells, where CoSpar adjusts within the clones of each size. A clone of a few late cells cannot reach a small p-value: two late cells, both of a fate that holds 40% of the late cells, give about 0.16.

`load_fate_bias` returns three tables: `cells`, one row per early cell indexed by cell ID, with `clone`, `late_cells`, `fate`, the dominant fate, missing for none, and one column per fate of its share, NaN without late cells; `clone_fates`, one row per clone with late cells and fate, by clone and then fate, with `clone`, `fate`, the fate's key columns, `early_cells`, `late_cells`, `cells`, the clone's late cells in the fate, `fraction`, `p_value` and `q_value`; and `fates`, one row per fate with its late cells, those without a clone included unless the clones keep labels apart. A fate may not be named `clone`, `late_cells` or `fate`, the other columns of the `cells` table. The result records the clones, the early and late cells, the label artifact and, with `group_by`, each late cell's fate, so measuring again with the same values returns the saved result.

{meth}`ds.plots.clone_fates <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_fates>` draws the `top` (40) clones with the most late cells, grouped by their dominant fate in the order of the fates and within a fate from the most late cells to the fewest, with clones whose late cells tie last: each clone's late cells stacked by fate, or their shares with `normalize=True`, under its early cells. Its table is `clones`.

On the synthetic experiment above, every early cell's fractions equal CoSpar 0.5.0's clonal fate map within a relative 1e-10, the effect of the 1e-10 that CoSpar adds to each clone's late cells, and CoSpar gives the 64 early cells without late clone-mates 0 in every fate, where CyteArc gives them none. The p-values of all 108 clones with late cells, in all six fates, equal those of `clonal_fate_bias`, and the adjusted ones those of statsmodels' Benjamini-Hochberg within 2.1e-16. On the same million cells, with the 124,841 cells of one sample early, the cells of the other seven samples late and ten cell types as fates, measuring the fate bias of 287,201 clones takes 1.7 s and peaks at 135 MiB of arrays, and loading its tables, 2.9 million rows of clone and fate, takes 0.6 s and peaks at 397 MiB, within the smallest budgets they accept, 284 MiB and 601 MiB.

## Import cell phylogenies

Lineage recorders such as PEtracer ([Colgan et al. 2026](https://doi.org/10.1016/j.cell.2026.09.050)) and the DNA Typewriter ([Yu et al. 2026](https://doi.org/10.1126/science.ael0508)) record edits as cells divide, and their authors reconstruct trees whose leaves are cells and whose internal nodes are their ancestors. {meth}`ds.clones.import_phylogeny <cytearc.datastore.namespaces.ClonesAccessor.import_phylogeny>` reads those trees, joins each leaf to a selected cell by barcode as `import_clones` joins its barcodes, and saves the forest as a `cell_phylogeny` artifact. {meth}`ds.clones.phylogeny_clones <cytearc.datastore.namespaces.ClonesAccessor.phylogeny_clones>` cuts it into clones, saved as a `clones` artifact that every clone summary reads, and {meth}`ds.clones.load_phylogeny <cytearc.datastore.namespaces.ClonesAccessor.load_phylogeny>` reads the trees as a {class}`~cytearc.clones.CellPhylogeny` record. CyteArc reads trees that tools or authors reconstructed; it does not reconstruct them.

```python
# PEtracer's trees on GEO, one per clone of embryo E7.5-R1, in days from the root.
phylogeny = ds.clones.import_phylogeny(
    ["E7.5-R1-C1.nwk.gz", "E7.5-R1-C2.nwk.gz"], cell_selection=cells, time_unit="days"
)
ds.artifacts.load(phylogeny).attrs["n_cells_in_no_tree"]  # cells that no leaf names
clones = ds.clones.phylogeny_clones(phylogeny, time=6.5)
ds.clones.overlap(clones, group_by="cell_type")
```

Each file's format is recognised from its contents: a Zarr store or an HDF5 file holds TreeData, and other files hold Newick text.

| `format` | Files | Read |
|---|---|---|
| `"newick"` | Newick text, plain or gzip-compressed, with one tree or several, each ending in `;`: PEtracer's trees on GEO, DNA Typewriter's `support_data/*.nwk`, and the trees of LAML, Cassiopeia and ConvexML | Each node's label and branch length |
| `"treedata"` | [TreeData](https://treedata.readthedocs.io) `.h5td` files, whose `obst` is one JSON string, and `.zarr` stores, with one group of columns per tree as treedata writes them since 0.3.0 | Each node's `time` and each edge's `length`, nothing else |

Newick is read in one pass without recursion, so a tree of a million leaves on one 59 MB line, as DNA Typewriter's is, or a caterpillar a million nodes deep reads like a small one. Bracket comments are skipped wherever they fall, nested or not, such as `[&R]` before a tree or NHX and BEAST annotations; labels in single quotes may hold any mark, with `''` for a quote; polytomies, single-child nodes, zero and negative lengths, and a length above the root are kept as written, as are underscores. Whitespace around a label is dropped, and as in treedata's reader, a missing `;` after the last tree is accepted and an unquoted number on an internal node, such as a bootstrap value, is a support value rather than a name; quote it, as `'95'`, to keep it as a name. NEXUS files are refused, and so are Zarr stores of treedata releases before 0.3.0, which hold `obst` as one JSON string; write them again with a later release. An error gives the byte where the text goes wrong, counted in the decompressed text of a gzip file, and quotes the text around it.

A node's time runs from its tree's root, plus any length above the root, in `time_unit`, which no file states and the import never guesses, such as `"days"` for PEtracer's and DNA Typewriter's trees. A TreeData node's own `time` is preferred over the time that its edge's `length` implies, and a root without a time starts at 0; a node below a branch without a length has no time, and a time or length of NaN, as treedata writes for branches without one, or a Newick length written `nan`, is unknown too. `time_unit=None` declares lengths that are not times, such as edit counts, and such trees are cut at depths only.

A leaf names the cell whose ID is its file's `barcode_prefix`, the leaf's name and `barcode_suffix`, after `barcode_pattern` as for [clone files](#join-barcodes-to-cells). Leaves that name no selected cell are counted in `n_unmatched_barcodes` and named in a warning, which a reused import gives again, and the selected cells in no tree are counted in `n_cells_in_no_tree`, such as 310 of the 1,342 cells of PEtracer's E7.5-R1 embryo, which neither of its clone trees holds. Two leaves that name one cell raise `ValueError`, so a file of bootstrap replicates, several trees over the same cells, is refused, as is a tree that names no selected cell.

Each tree is named by its file's `barcode_prefix` and the file's name without `.gz` and its extension, such as `E7.5-R1-C1` for `E7.5-R1-C1.nwk.gz`; a file of several trees adds `_` and each tree's position from 1, or its TreeData key. The names must differ, and the forest holds its trees in their natural order. Saved trees are reused when each file holds the same bytes, every file of a Zarr store included, and its trees take the same name ({ref}`reuse of imports <import_reuse>`), so a moved file reuses them and a renamed one does not. A directory that is not a Zarr store is refused before it is read.

## Cut phylogenies into clones

`phylogeny_clones(phylogeny, time=...)` makes a clone of each clade whose root is the first node at or past `time` on the way down from its tree's root: the cells that descend from one lineage alive at that time. `depth=` counts branchings below each root instead, after merging each node that has one child with that child, so PEtracer's roots, which have one child each, start at depth 0 like that child. Each child of a polytomy that the cut passes is a clone of its own, and `depth=0`, or a `time` at or before every root's time, such as `time=0` when no negative length lies above a root, makes each tree one clone, as for one tree per clone. These are the clades of pycea's `tl.clades`, which cuts at the first node whose depth key reaches the cut.

A cell whose leaf ends before the cut, earlier than `time` or fewer than `depth` branchings below its root, is in no clone; the artifact counts such cells in `n_cells_before_cut`, and a warning names how many. In an ultrametric tree every leaf ends at one time, and `leaf_time_spread`, saved on both artifacts, is the largest gap between a tree's earliest and latest leaf: PEtracer's leaves end within 16 millionths of a day of E7.5, from lengths written to six digits, while a tree of another unit or of uneven sampling can leave cells out of every cut past its earliest leaf. A cut that leaves every cell out raises `ValueError`.

A clone is named by its root's name when every node of its tree has a name of its own, as PEtracer's and Cassiopeia's trees do, and otherwise by `node` and its root's position in the tree, the `node` column of the `nodes` table; with several trees, each label starts with its tree's name and `/`, as `E7.5-R1-C1/0D4vcs3c`. The clones are like any others: `load_clones` gives `source` `"phylogeny_clones"`, and the summaries above measure them.

`load_phylogeny` returns a {class}`~cytearc.clones.CellPhylogeny` record. Its `nodes` table holds one row per node, each tree in preorder: `tree`, `node`, `name`, `parent`, `time`, `depth`, `is_leaf` and `cell`, the ID of the cell a leaf names; its `trees` table one row per tree with its nodes, leaves, cells, root time and earliest and latest leaf time; and `time_unit`.

The cuts equal those of pycea 0.4.0's `tl.clades` on PEtracer's two E7.5-R1 trees, at six times and five depths, and on ConvexML's 400-leaf tree, at five times and five depths, after `read_nwk` of treedata 0.4.0 parsed the trees, and the reader agrees node by node with `read_nwk` on 18 adversarial trees. On DNA Typewriter's tree of a mouse embryo, 59 MB on one line with 1,281,141 leaves and 1,109,894 internal nodes, in one container of 4 CPUs on Modal, reading took 1.6 s, importing it into a store of its cells and 128,114 others 3.9 s, loading the node table 1.6 s, and cutting it 0.5 s for the 4 clones at day 2 and 5.3 s for the 733,791 clones at day 13. The peaks that tracemalloc traced were 317 MiB for reading, 477 MiB for importing, 610 MiB for loading and at most 427 MiB for cutting, each within the smallest memory budget the step accepts.

### Memory of phylogenies

The import checks the memory budget before it reads any file, for every file at once: all the trees that the files may hold and the largest peak of reading one. It checks again as it reads each file, once a Newick scan has counted the names and once a TreeData file's JSON is read; then before it joins the leaves and before it writes the names. Loading and cutting check it before they read. A refusal names the bytes the step needs, so a budget raised to that figure runs it.

| Step | Bytes |
|---|---|
| Reading Newick | The text's bytes, 41 for each node that its `(`, `,` and `;` allow, and 8 for each `(` and `;`; then 65 per node, and for each name 49 and its bytes in ASCII text, or 80 and four times its bytes in other text |
| Reading TreeData | In a `.h5td` file, the bytes of its JSON, bounded by the file's bytes outside fixed-size datasets, and its characters twice more at one, two or four bytes each, by its widest character; then 600 per object of the JSON, 64 per member of an object, and 64 per value that the objects being parsed hold at once outside the objects within them. In a `.zarr` store, 400 per node and twice the text of its arrays, counted as they are read |
| Joining the leaves | 88 per node and 2,000 per tree; while the cells' IDs are read, for each selected cell 188 and eight per character of the IDs' stored width, besides one stored chunk of the IDs, decoded twice, and once they are read, the join's lookup of 100 per cell and the IDs as text, which the join counts; and for each block of 65,536 leaves, 300 per leaf and twice its name |
| Writing names | Three copies of a chunk of names at the width of the longest, four bytes a character, of at most 100,000 rows and 16 MiB, and one of io's default buffers for each write that Zarr runs at once, up to its async concurrency: 8 KiB each before Python 3.14 and 128 KiB since. Names are stored at the width of the longest, so one long name widens the storage of every name |
| Loading | 160 per node, 8 per selected cell, for each cell in a tree 88 and eight per character of the IDs' stored width, besides one stored chunk of the IDs, decoded twice, and a stored chunk of names twice, besides the names read |
| Cutting | 64 per node; for each clone 608 and eight per character of the widest label; 56 per cell; a stored chunk of names twice; and what [saving clones](#read-export-and-plot-clones) holds |
| Each import, load or cut | 4 MiB |

Peaks that tracemalloc recorded stayed within these budgets on 52,000 leaves on one line, 240-character Greek names, 200 files, TreeData with character states, in Zarr, as a star of 50,000 leaves, with attributes that hold objects, and with a million lists and 300,000 keys beside its tree, loading, and cuts into a few clones, into one clone per cell and at a depth, and a 59 MB tree of 1.28 million leaves on one line parsed within its smallest budget. Master stores cell IDs in chunks of 100,000 rows, so reading a few IDs costs a whole chunk, which dominates the budget of small imports.

## Clones

```{eval-rst}
.. autoclass:: cytearc.clones.Clones
```

## Lineage barcodes

```{eval-rst}
.. autoclass:: cytearc.clones.LineageBarcodes
    :members: pairs
```

## Recorder alleles

```{eval-rst}
.. autoclass:: cytearc.clones.LineageAlleles
    :members: site_states
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

## Cell phylogenies

```{eval-rst}
.. autoclass:: cytearc.clones.CellPhylogeny
```
