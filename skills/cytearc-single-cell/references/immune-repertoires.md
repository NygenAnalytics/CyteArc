# Immune receptor repertoires

Import the T cell and B cell receptor contigs of single-cell immune profiling (10x Genomics Chromium Single Cell Immune Profiling, BD Rhapsody, TRUST4 and other AIRR producers), check each cell's chains, label cells by receptor type, subtype and chain pairing, group them into clonotypes, measure clonal expansion within samples, measure clonal diversity with rarefaction and extrapolation, compare samples or cell types by the clonotypes they share, count gene usage and junction lengths, and export AIRR Rearrangement files. Docs: <https://docs.nygen.io/CyteArc/tutorials/immune-repertoires>, <https://docs.nygen.io/CyteArc/api/repertoire.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

## When to use

- A store of the expression library's cells, and the V(D)J output of the same cells: `outs/filtered_contig_annotations.csv` from `cellranger vdj`, or `outs/per_sample_outs/<sample>/vdj_t/filtered_contig_annotations.csv` and `vdj_b/filtered_contig_annotations.csv` from `cellranger multi`.
- An AIRR Rearrangement TSV with a `cell_id` column: Cell Ranger's `airr_rearrangement.tsv`, TRUST4's barcode AIRR file, BD Rhapsody's AIRR output, IgBLAST with Change-O, MiXCR, dandelion.

## Key concepts

- `ds.repertoire.import_receptors(paths, cell_selection=cells)` reads one file or a list, each told apart by its header (Cell Ranger CSV or AIRR TSV, plain or gzip), joins each contig to a selected cell by barcode, and saves a datastore-scoped `immune_receptors` artifact. Import the TCR and BCR files of one library in one call.
- Contig fields keep their AIRR names; Cell Ranger's columns are renamed: `barcode` to `cell_id`, `contig_id` to `sequence_id`, `chain` to `locus`, `v_gene`/`d_gene`/`j_gene`/`c_gene` to `v_call`/`d_call`/`j_call`/`c_call`, `cdr3`/`cdr3_nt` to `junction_aa`/`junction` (Cell Ranger's CDR3 includes the conserved C and F/W), `umis` to `umi_count`, `reads` to `consensus_count`, and region columns such as `fwr1`/`fwr1_nt` to `fwr1_aa`/`fwr1`. `cytearc.repertoire.FIELDS` lists the kept fields; other columns are dropped.
- Required per file: a contig ID, `locus` or gene calls (an AIRR file without `locus` has it inferred from the calls that all name one locus), `junction_aa` and `productive`. AIRR files also need `cell_id`.
- A cell ID is `barcode_prefix` + barcode, matched exactly: one prefix for every file, or a list with one prefix per file. Merged stores name cells `<sample>__<barcode>`.
- Contigs of barcodes outside the selection are counted (`n_unmatched_contigs`, `n_unmatched_barcodes`, `unmatched_barcode_examples` attributes) and named in a warning. Selected cells without contigs are kept as `"no IR"`.
- Reuse follows each file's bytes and prefix, in any order; a moved file reuses, an edited file re-imports, `invalidate_cache=True` imports again.
- Chains (scirpy's receptor model): a contig counts when it has a locus of a receptor chain, a `junction_aa`, is productive (`productive_only=True`, default) and is not flagged `is_cell` or `high_confidence` False. Ranking within each arm: `umi_count`, `duplicate_count`, `consensus_count` (most first, missing as 0), then `junction` and `junction_aa` (later in sort order first), then contig ID. Arms: VJ (TRA, TRG, IGK, IGL) and VDJ (TRB, TRD, IGH). Up to two chains per arm fill `VJ_1`, `VJ_2`, `VDJ_1`, `VDJ_2`; more than two in an arm makes the cell `"multichain"`.
- Labels, from the slots only:
  - `receptor_type` (canonical `values`): `"TCR"`, `"BCR"`, `"ambiguous"` (both kinds), `"multichain"`, `"no IR"`.
  - `receptor_subtype`: `"TRA+TRB"`, `"TRG+TRD"`, `"IGH"`, `"IGH+IGK"`, `"IGH+IGL"`, `"IGH+IGK/L"`, `"ambiguous"`, `"multichain"`, `"no IR"`. A light chain without a heavy chain still names its subtype (`"IGH+IGK"`), as in scirpy.
  - `chain_pairing`: `"single pair"`, `"orphan VJ"`, `"orphan VDJ"`, `"extra VJ"`, `"extra VDJ"`, `"two full chains"`, `"ambiguous"`, `"multichain"`, `"no IR"`.
- Other per-cell arrays: `multichain`, `n_vj` and `n_vdj` (chains of each arm before the slots keep two), `n_contigs` (imported contigs, chains or not). `select_cells` and `artifacts.load_values` read them; `chain_slots` and the `contig_*` arrays are not cell-aligned.
- `ds.repertoire.load_receptors(receptors, fields=None)` returns an `ImmuneReceptors` record: `.cells` (indexed by `ids`: the three labels, `multichain`, `n_vj`, `n_vdj`, `n_contigs`, then per slot `<slot>_locus`, `_v_call`, `_d_call` (VDJ slots only), `_j_call`, `_c_call`, `_junction`, `_junction_aa`, `_umi_count`, `_consensus_count`), `.chains` (one row per contig: `ids`, `slot` (`""` for none), `chain` (counts as a chain) and each field; missing text is NA, counts `Int64`, flags `boolean`), `.fields` and `.productive_only`. `fields=[...]` loads only those fields, into `.chains` and into the slots of `.cells`; large AIRR files with sequences and alignments need it.
- `ds.plots.receptor_qc(receptors)` draws cells per receptor type, per chain pairing, and cells by chains of each arm (0, 1, 2, 3+). `group_by="sample"` (a cell column) or `groups=clusters_ref` (a label artifact, which may hold fewer cells) stacks one bar per group, as fractions unless `normalize=False`. Tables: `receptor_type`, `chain_pairing`, `chains_per_cell`.
- Validated: every cell's labels and every slot's locus, gene calls and junctions equal scirpy 0.25.1's on 10x PBMC 10k (TCR and BCR, Cell Ranger 4.0.0 and 5.0.0) and CD8+ donor 1 (Cell Ranger 3.0.2), from the filtered and unfiltered contig annotations and from the AIRR files.
- `ds.repertoire.clonotypes(receptors)` groups cells into clonotypes by identical chains (scirpy's `define_clonotypes` rules) and saves a datastore-scoped `clonotypes` artifact. Options:
  - `sequence="nt"` (default) or `"aa"`: junction nucleotides or residues.
  - `receptor_arms="all"` (default: both arms for cells with both, one arm for cells with one; a paired cell never joins an orphan), `"any"` (either arm), `"VJ"` or `"VDJ"` (that arm; cells without it get no clonotype).
  - `dual_ir="any"` (default: one chain of an arm matches one of the other's), `"all"` (same number of chains, all matched, either order), `"primary_only"`.
  - `same_v_gene=False`, `same_j_gene=False`: also require matched chains' gene calls to match.
  - `within=None`: cell metadata columns (such as the donor) whose labels keep clonotypes apart; every cell with a receptor needs a label. `within_receptor_type=True` keeps TCR, BCR, ambiguous and multichain cells apart.
  - `metric="identity"` (default; `cutoff=None`), or clonotype clusters by similar junctions: `"hamming"` (≤ `cutoff` substitutions, equal length), `"normalized_hamming"` (≤ `cutoff` percent of the length, rounded half up, equal length), `"levenshtein"` (≤ `cutoff` edits). `cutoff` is required for these (scirpy's default 2 means 2% for normalized_hamming). Same rules for arms, dual chains, genes, groups. Record: `.metric`, `.cutoff`. Pairs within cutoffs ≤ 3 are found through keys (100k junctions: 0.6 s for 1 substitution, 4.6 s for 2 edits).
- Clonotypes are the connected components of the links, so a cell matching two others joins them. Labels `clonotype_1` (largest), `clonotype_2`, ...; ties follow the sorted chains, so labels do not depend on cell order. Arrays: canonical `values` (label, empty and treated as missing for a cell without one), `codes` (rank minus one, -1 without), `clone_size` (0 without). `select_cells(clonotypes, include=["clonotype_1"])` selects a clonotype's cells; `select_cells(clonotypes, value="clone_size", low=2, keep_bounds=True)` the expanded cells.
- Reuse follows the receptors, the options and the `within` labels; `invalidate_cache=True` defines again. About 1,000 bytes per cell at the peak, checked against `mem_budget`.
- `ds.repertoire.load_clonotypes(clonotypes)` returns a `Clonotypes` record: `.cells` (indexed by `ids`: `clonotype`, a categorical of the labels in rank order, NA without, and `clone_size`), `.clonotypes` (one row per clonotype, largest first: `clonotype`, `n_cells`, `fraction` of cells with a clonotype, `n_configurations`, `receptor_type`, each `within` column, and the representative cell's chains per slot: `VJ_1_junction_aa`, `VJ_1_junction`, `VJ_1_v_call`, `VJ_1_j_call`, ..., `VDJ_2_j_call`), and the options (`.sequence`, `.receptor_arms`, `.dual_ir`, `.same_v_gene`, `.same_j_gene`, `.within`, `.within_receptor_type`, `.receptors`).
- `ds.plots.clonotype_abundance(clonotypes, top=20)` draws clonotypes per clone size (log-log) and the largest clonotypes labelled with their `VDJ_1` junction (`VJ_1` without a VDJ chain); `group_by` or `groups` gives one line per group and stacks the bars. Tables: `sizes` (`group`, `clone_size`, `n_clonotypes`), `top` (`clonotype`, `label`, cells per group or `cells`).
- Validated: equal to scirpy 0.25.1's identity clonotypes for every cell under ten option combinations on PBMC 10k and CD8+ donor 1; with `dual_ir="all"`, equal to scRepertoire 2.8.0's `CTnt`, `CTaa` (`sequence="aa"`) and `CTstrict` (both gene matches) and to Cell Ranger 3.0.2's clonotypes.
- `ds.repertoire.clonal_expansion(clonotypes, within=None, within_receptor_type=True, breakpoints=None, frequency_breakpoints=None)` counts each clonotype's cells within groups (the `within` columns' labels, such as the sample, times the receptor type) and saves a datastore-scoped `clonal_expansion` artifact. Every cell with a clonotype needs a `within` label. Pass at most one bound:
  - `breakpoints` (default `(1, 2, 5, 20, 100)`): categories `"1 cell"`, `"2 cells"`, `"3 to 5 cells"`, `"6 to 20 cells"`, `"21 to 100 cells"`, `"more than 100 cells"` (scirpy's `<= 1`, `<= 2`, `<= 5`, ..., `> 100`).
  - `frequency_breakpoints`, strictly increasing and ending at 1, such as scRepertoire's `(0.0001, 0.001, 0.01, 0.1, 1)` (Rare to Hyperexpanded): categories `"0 to 0.0001"`, ..., `"0.1 to 1"`, bins `(lower, upper]` of the clone's share of the group's cells with a clonotype.
- Arrays: canonical `values` (category, empty without a clonotype), `group_clone_size`, `group_clone_frequency`. `select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])` keeps the cells of large clones. Reuse follows the clonotypes, the bounds and the `within` labels. About 300 bytes per cell at the peak.
- `ds.repertoire.load_clonal_expansion(expansion, splits=(10, 100, 1000, 10000, 30000, 100000))` returns a `ClonalExpansion` record: `.cells` (`clonotype`, `category` (ordered), `group` (labels joined by ", "), the group key columns (`within` columns, `receptor_type`), `group_clone_size`, `group_clone_frequency`; labels are categoricals), `.categories` (`group`, group key, `category`, `cells`, `clonotypes`, `fraction` of the group's cells; every category of every group), `.rank_windows` (`group`, group key, `window` such as `"1 to 10"`, `"11 to 100"`, ..., `"after 100000"`, `cells`, `fraction`: the cells of each group's clonotypes ranked by size into windows, scRepertoire's `clonalProportion`), `.labels`, `.breakpoints`, `.frequency_breakpoints`, `.splits`, `.within`, `.within_receptor_type`, `.clonotypes`.
- `ds.plots.clonal_expansion(expansion, group_by=None, groups=None, count="cells", normalize=True, splits=...)` draws stacked bars per group of the categories (cells, or `count="clonotypes"`) and of the rank windows. Groups default to the expansion's own; `group_by`/`groups` regroup the cells (categories stay those measured within the expansion's groups; ranks are recounted within the plotted groups). Tables: `categories` (`group`, `category`, `cells` or `clonotypes`, `fraction`), `rank_windows` (`group`, `window`, `cells`, `fraction`).
- Validated: every cell's category equals scirpy 0.25.1's `clonal_expansion` within pseudo samples and across all cells on PBMC 10k and CD8+ donor 1, and the plot's tables equal `summarize_clonal_expansion`; frequency fractions equal scRepertoire 2.8.0's `clonalHomeostasis` and rank windows its `clonalProportion` on PBMC 10k T cells.
- `ds.repertoire.diversity(clonotypes, group_by=None, groups=None, within_receptor_type=True, cell_selection=None, size=None, coverage=None)` returns a `ClonalDiversity` record and saves nothing. Groups: `group_by` columns × `groups` label artifact (its key column is the artifact kind, such as `cluster_labels`) × `receptor_type`; `cell_selection` keeps some cells (such as `select_cells(receptors, include=["TCR"])`); cells without a clonotype or label are left out. A key column named twice (a `receptor_type` metadata column with `within_receptor_type=True`) raises.
  - `.indices`, one row per group: `group` (labels joined by `", "`, `"all cells"` without groups), the key columns, `cells`, `clonotypes`, `singletons`, `doubletons`, `coverage` (Chao and Jost sample coverage, iNEXT's `SC`), `shannon` (nats), `pielou`, `simpson`, `inverse_simpson`, `gini_simpson`, `gini`, `d50`, `chao1` (`S + (n-1)/n f1²/(2 f2)`, iNEXT and immunarch), `chao1_bias_corrected` (`S + f1(f1-1)/(2(f2+1))`, vegan and scRepertoire), `ace` (rare ≤ 10 cells; NaN when all rare are singletons).
  - `.hill`, one row per group and order 0, 1, 2: `group`, the key columns, `order`, `observed`, `estimated` (asymptotic: Chao1, exp of Chao-Wang-Jost entropy, inverse of Good's Simpson estimate, inf without doubletons or larger), `at_size`, `coverage_at_size`, `method_at_size`, `at_coverage`, `size_at_coverage`, `method_at_coverage` (methods `"rarefaction"`, `"observed"`, `"extrapolation"`).
  - `.size` and `.coverage`: the common size (default: smallest doubled group size) and coverage (default: lowest coverage at doubled sizes), iNEXT's `estimateD` defaults.
- `ds.repertoire.rarefaction(clonotypes, group_by, groups, within_receptor_type, cell_selection, orders=(0, 1, 2), sizes=None, endpoint=None, knots=40)` returns a `Rarefaction` record: `.curves` (`group`, key columns, `order`, `size`, `method`, `diversity`, `coverage`); default sizes are iNEXT's 40 knots per group (half up to its size, half to twice it); `sizes` gives every group the same sizes.
- `ds.repertoire.overlap(clonotypes, group_by=None, groups=None, within_receptor_type=True, cell_selection=None)` returns a `ClonotypeOverlap` record and saves nothing; with `within_receptor_type` only groups of one receptor type are compared. `.group_sizes` (`group`, key columns, `cells`, `clonotypes`); `.pairs` (`group_1`, `group_2`, `shared`, `jaccard`, `sorensen`, `overlap`, `cosine` (presence, Ochiai), `morisita_horn`, `abundance_cosine`, `shared_fraction_1`, `shared_fraction_2`). Needs at least two groups. Checked against `mem_budget` at about 250 bytes per pair of groups of one receptor type; use fewer groups (fewer `group_by` columns, a `cell_selection`) when it refuses.
- `ds.repertoire.track(clonotypes, group_by=..., top=10, labels=None, ...)` returns a `ClonotypeTracking` record: `.cells` (`clonotype`, `group`, key columns, `cells`, `fraction` of the group's cells; zeros included, within the clonotype's receptor type), `.tracked` (labels), `.junctions` (each tracked clonotype's representative `VDJ_1` junction_aa, `VJ_1` without a VDJ chain). Default: each group's `top` largest clonotypes, together (ties to the larger clonotype overall); `labels=["clonotype_1"]` instead. The table is checked against `mem_budget` at about 150 bytes per row; lower `top` or use fewer groups when it refuses.
- `ds.plots.clonal_overlap(clonotypes, group_by=..., index="morisita_horn")` heatmap (diagonal blank, values printed up to 12 groups; tables `matrix`, `pairs`); `ds.plots.clonotype_tracking(clonotypes, group_by=..., top=..., order=[...])` one line per clonotype across groups (table `cells`).
- Validated: every overlap index within 5e-14 of immunarch 0.10.3 `repOverlap` (`public`, `overlap`, `jaccard`, `tversky`, `cosine`, `morisita`) and scRepertoire 2.8.0 `clonalOverlap` (`raw`, `overlap`, `jaccard`, `cosine`, `morisita`) on CD8+ donor 1 and PBMC 10k pseudo samples. scRepertoire's `overlap`, `raw` and `cosine` are presence-based in its outputs, whatever its help page says.
- `ds.repertoire.gene_usage(source, genes="v_call", level="gene", slots=("VJ_1", "VDJ_1"), count="cells", group_by=None, groups=None, cell_selection=None)` returns a `GeneUsage` record (saves nothing): `.table` (`group`, key columns, `locus`, one column per gene field, `cells` or `clonotypes`, `fraction` within group and locus; rows by group in natural order, locus, count most first). `source`: `immune_receptors` (cells with a chain) or `clonotypes` (cells with a clonotype; `count="clonotypes"` counts each clonotype once per group and value). `genes`: one of `v_call`, `d_call`, `j_call`, `c_call`, or several for combinations such as V-J pairs. `level`: `"allele"` (`TRBV12-3*01`), `"gene"` (`TRBV12-3`), `"family"` (`TRBV12`; `TRAV29/DV5` → `TRAV29`). First call of a multi-gene call is kept; chains without the value are left out.
- `ds.repertoire.spectratype(source, sequence="aa" or "nt", slots=..., count=..., group_by=...)` returns a `Spectratype` record: `.table` (`group`, key columns, `locus`, `length`, count, `fraction`; rows by group in natural order, locus, length).
- Plots: `ds.plots.gene_usage(source, gene="v_call", group_by=..., top=30)` (one panel per locus; heatmap of fractions by group, bars without groups; table `usage`), `ds.plots.gene_pairing(source, cell_selection=..., top=20)` (V × J heatmap per locus, of its `top` V and `top` J genes; table `pairs`), `ds.plots.spectratype(source, group_by=...)` (lines per group per locus; table `spectratype`).
- Validated: every V, J and V-J pair count and junction length count of `VJ_1` and `VDJ_1` equals scirpy 0.25.1 `group_abundance` and `spectratype` on PBMC 10k pseudo samples. scRepertoire's `percentGenes` counts via immApex string parsing and is not comparable.
- `ds.repertoire.export_airr(source, path, fields=None, chains_only=False)` writes an AIRR Rearrangement TSV (`.gz`: gzip level 6) and returns the contigs written: required AIRR fields first (empty when not stored, e.g. sequences from Cell Ranger CSVs), then `cell_id` (store cell IDs), `clone_id` (each cell's clonotype when `source` is a `clonotypes` artifact), `locus`, `c_call`, `umi_count`, other fields; logical fields `T`/`F`; repeated `sequence_id`s (shared by TCR and BCR files) get `_2`, `_3` with a warning. Streams blocks of about 32 MiB of text: holds one block plus each cell's ID and clonotype (at most about 190 MiB for 200,000 cells); refuses with `MemoryError` beyond `mem_budget`. Gzip dominates `.gz` exports (88 s for 414,060 TCR contigs, against 3 s uncompressed). Validated: the `airr` library accepts the file; scirpy `read_airr` gives the same chains as from the 10x CSVs; CyteArc re-import gives the same receptors and clonotypes. `cytearc.writers.write_airr_rearrangements(table, path)` writes any contig table; `cytearc.writers.airr.AirrRearrangementWriter(path, columns, whole=...)` writes one block at a time (`with ... as writer: writer.write(block)`, a DataFrame or a mapping of columns).
- `ds.plots.clonal_rarefaction(clonotypes, group_by=..., cell_selection=..., orders=(0, 1, 2), base="size" or "coverage")`: one panel per order (`order_0`, `order_1`, `order_2`) plus `coverage` (sample completeness) with `base="size"`; solid rarefied, dashed extrapolated, a point at the observed size. Table: `curves` (with `group`).
- Validated: Hill-number curves within 1.4e-12 of iNEXT 3.0.2 (its Shannon extrapolation is Chao et al. 2023's), asymptotic estimates within 7e-13, `estimateD` within 1e-7; every index within 6e-14 of scRepertoire 2.8.0 `clonalDiversity` (no bootstraps), immunarch 0.10.3 `repDiversity` and vegan 2.7.6 `estimateR`.

## Recipe

```python
import cytearc

ds = cytearc.DataStore("pbmc.zarr")
cells = ds.snapshot_cell_selection()  # or the analysis cells of a completed run

receptors = ds.repertoire.import_receptors(
    [
        "outs/per_sample_outs/pbmc/vdj_t/filtered_contig_annotations.csv",
        "outs/per_sample_outs/pbmc/vdj_b/filtered_contig_annotations.csv",
    ],
    cell_selection=cells,
)
attributes = ds.artifacts.load(receptors).attrs
print(attributes["n_file_contigs"], attributes["n_unmatched_contigs"], attributes["unmatched_barcode_examples"])

table = ds.repertoire.load_receptors(receptors, fields=["junction_aa", "umi_count"])
print(table.cells["receptor_type"].value_counts())
print(table.cells["chain_pairing"].value_counts())
print(table.cells.query("chain_pairing == 'single pair'")[["VJ_1_junction_aa", "VDJ_1_junction_aa"]].head())

plot = ds.plots.receptor_qc(receptors, group_by="sample", show=False)
plot.save("receptor_qc.png")
plot.close()

clonotypes = ds.repertoire.clonotypes(receptors)  # nt, both arms, either chain
table = ds.repertoire.load_clonotypes(clonotypes)
print(table.clonotypes.head(10)[["clonotype", "n_cells", "receptor_type", "VJ_1_junction_aa", "VDJ_1_junction_aa"]])
print((table.cells["clone_size"] >= 2).sum(), "cells in expanded clonotypes")
plot = ds.plots.clonotype_abundance(clonotypes, group_by="sample", show=False)
plot.save("clonotype_abundance.png")
plot.close()
by_donor = ds.repertoire.clonotypes(receptors, within="donor")  # several donors
residues = ds.repertoire.clonotypes(receptors, sequence="aa")  # shared specificity
similar = ds.repertoire.clonotypes(receptors, sequence="aa", metric="hamming", cutoff=1)  # likely one epitope
families = ds.repertoire.clonotypes(  # B cell clonal families
    receptors, sequence="nt", metric="normalized_hamming", cutoff=15,
    same_v_gene=True, same_j_gene=True, within="donor",
)
expanded = ds.select_cells(clonotypes, value="clone_size", low=2, keep_bounds=True)

expansion = ds.repertoire.clonal_expansion(clonotypes, within="sample")
table = ds.repertoire.load_clonal_expansion(expansion)
print(table.categories.query("receptor_type == 'TCR'").pivot(index="sample", columns="category", values="fraction"))
print(table.rank_windows.query("window == '1 to 10'")[["sample", "receptor_type", "fraction"]])
plot = ds.plots.clonal_expansion(expansion, group_by="cell_type", show=False)
plot.save("clonal_expansion.png")
plot.close()
large = ds.select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])
homeostasis = ds.repertoire.clonal_expansion(
    clonotypes, within="sample", frequency_breakpoints=(0.0001, 0.001, 0.01, 0.1, 1)
)  # scRepertoire's Rare to Hyperexpanded

t_cells = ds.select_cells(receptors, include=["TCR"])
diversity = ds.repertoire.diversity(clonotypes, group_by="sample", cell_selection=t_cells)
print(diversity.indices[["sample", "cells", "clonotypes", "coverage", "shannon", "inverse_simpson", "chao1"]])
print(diversity.size, diversity.coverage)  # the common size and coverage
print(diversity.hill.pivot(index="sample", columns="order", values="at_coverage"))
plot = ds.plots.clonal_rarefaction(clonotypes, group_by="sample", cell_selection=t_cells, show=False)
plot.save("clonal_rarefaction.png")
plot.close()
curves = ds.repertoire.rarefaction(clonotypes, group_by="sample", cell_selection=t_cells).curves

overlap = ds.repertoire.overlap(clonotypes, group_by="tissue", cell_selection=t_cells)
print(overlap.pairs[["group_1", "group_2", "shared", "morisita_horn", "shared_fraction_1", "shared_fraction_2"]])
plot = ds.plots.clonal_overlap(clonotypes, group_by="tissue", cell_selection=t_cells, show=False)
plot.save("clonal_overlap.png")
plot.close()
ds.repertoire.export_airr(clonotypes, "repertoire.tsv.gz")  # for Immcantation, dandelion, scirpy
usage = ds.repertoire.gene_usage(clonotypes, genes="v_call", count="clonotypes", group_by="sample")
print(usage.table.query("locus == 'TRB'").head())
pairs = ds.repertoire.gene_usage(receptors, genes=("v_call", "j_call"))
print(pairs.table.query("v_call == 'TRAV1-2' and j_call == 'TRAJ33'"))  # MAIT cells
lengths = ds.repertoire.spectratype(receptors, group_by="sample").table
for name in ("gene_usage", "gene_pairing", "spectratype"):
    plot = getattr(ds.plots, name)(receptors, show=False)
    plot.save(f"{name}.png")
    plot.close()
tracked = ds.repertoire.track(clonotypes, group_by="timepoint", top=5, cell_selection=t_cells)
plot = ds.plots.clonotype_tracking(clonotypes, group_by="timepoint", top=5, order=["d0", "d7", "d28"], show=False)
plot.save("clonotype_tracking.png")
plot.close()

t_cells = ds.select_cells(receptors, include=["TCR"])
paired = ds.select_cells(receptors, value="chain_pairing", include=["single pair", "extra VJ"])
# `umap_ref` is a UMAP of the receptors' cells: to colour run["umap"] of a ds.pipeline.rna run,
# import the receptors with cell_selection=run["analysis_cell_selection"].
plot = ds.plots.embedding(layout=umap_ref, color_by=receptors, show=False)  # receptor type
plot.save("receptor_type_umap.png")
plot.close()

# A merged store: each sample's files with that sample's prefix.
merged = ds.repertoire.import_receptors(
    ["s1/vdj_t.csv", "s1/vdj_b.csv", "s2/vdj_t.csv", "s2/vdj_b.csv"],
    cell_selection=cells,
    barcode_prefix=["s1__", "s1__", "s2__", "s2__"],
)
```

## Checks

- Read `n_unmatched_contigs` and the warning. A few percent of contigs outside the expression cells is usual (V(D)J and expression call cells separately); most contigs unmatched means a wrong `barcode_prefix` or barcodes with another suffix than the cell IDs.
- `"no IR"` among cells whose expression marks them as T or B cells points to a shallow V(D)J library; `"no IR"` monocytes and NK cells are expected.
- `"orphan VJ"`/`"orphan VDJ"` cells lost a chain in capture and still carry clonal information. `"extra VJ"` (two alpha or two light chains) is often biological. `"extra VDJ"`, `"two full chains"`, `"multichain"` and `"ambiguous"` are more often doublets or ambient chains: count them per sample and compare with `qc.doublets` before removing anything.
- An `"ambiguous"` T cell with a heavy chain, or B cell with a beta chain: scattered in expression clusters points to ambient contigs, a cluster of its own to doublets.
- Keep `productive_only=True` for clonotypes; `False` is for studying non-productive rearrangements.
- Import TCR and BCR files together: imported apart, a cell with both kinds of chain is not `"ambiguous"` in either artifact.
- Gamma-delta chains need a gamma-delta enrichment kit; a standard TCR library holds few TRG or TRD contigs.
- Report the clonotype options. Keep `sequence="nt"` to count clones (identical amino acids from different nucleotides are independent recombinations); use `"aa"` for shared specificity.
- With several donors, pass the donor as `within`; never the tissue or time point of one donor, whose clones the study follows across samples.
- Orphan cells form their own clonotypes under `receptor_arms="all"`; with many orphan VDJ cells, also try `receptor_arms="VDJ"` and report both.
- B cell clonotypes from identical junctions split clones that carry somatic mutations; they are exact sequence variants, not clonal families. Use `metric="normalized_hamming"` on nucleotides with V and J matches and the donor as `within` for families, and report the cutoff.
- Measure expansion `within` the sample: without it, a clonotype's cells in all samples add up. Frequency categories depend on sample size (one cell of 1,000 is already 0.001), so compare them only between samples of similar size; bounds in cells compare any samples.
- Compare expansion between conditions with one value per sample (such as the fraction of cells in clonotypes of more than two cells), never with cells as replicates; test such values with a method that treats samples as replicates, such as CytePool's (https://docs.nygen.io/CytePool/).
- Use `count="clonotypes"` for gene usage compared between samples, so expanded clones do not decide shares; compare genes within their locus.
- Report Morisita-Horn for overlap between groups of different depth; presence indices (Jaccard, overlap) grow with cells. A clonotype absent from one group may be below detection there.
- Compare diversity by Hill numbers at a common coverage or size (`hill.at_coverage`, `hill.at_size`), not observed indices, which grow with the cells sampled. Check `plots.clonal_rarefaction`: steep curves at the observed point mean richness and Chao1 are poorly determined; order 2 is stable. Name the Chao1 form reported. Leave out tiny groups (such as multichain cells, with `cell_selection`) or set `size`/`coverage`, since the smallest group sets the defaults.
