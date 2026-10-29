# Immune receptor repertoires

Import the T cell and B cell receptor contigs of single-cell immune profiling (10x Genomics Chromium Single Cell Immune Profiling, BD Rhapsody, TRUST4 and other AIRR producers), check each cell's chains, label cells by receptor type, subtype and chain pairing, group them into clonotypes, and measure clonal expansion within samples. Docs: <https://docs.nygen.io/CyteArc/api/repertoire.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

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
- `ds.repertoire.load_receptors(receptors, fields=None)` returns an `ImmuneReceptors` record: `.cells` (indexed by `ids`: the three labels, `multichain`, `n_vj`, `n_vdj`, `n_contigs`, then per slot `<slot>_locus`, `_v_call`, `_d_call` (VDJ slots only), `_j_call`, `_c_call`, `_junction`, `_junction_aa`, `_umi_count`, `_consensus_count`), `.chains` (one row per contig: `ids`, `slot` (`""` for none), `chain` (counts as a chain) and each field; missing text is NA, counts `Int64`, flags `boolean`), `.fields` and `.productive_only`. `fields=[...]` loads fewer contig fields into `.chains`; large AIRR files with sequences and alignments need it.
- `ds.plots.receptor_qc(receptors)` draws cells per receptor type, per chain pairing, and cells by chains of each arm (0, 1, 2, 3+). `group_by="sample"` (a cell column) or `groups=clusters_ref` (a label artifact, which may hold fewer cells) stacks one bar per group, as fractions unless `normalize=False`. Tables: `receptor_type`, `chain_pairing`, `chains_per_cell`.
- Validated: every cell's labels and every slot's locus, gene calls and junctions equal scirpy 0.25.1's on 10x PBMC 10k (TCR and BCR, Cell Ranger 4.0.0 and 5.0.0) and CD8+ donor 1 (Cell Ranger 3.0.2), from the filtered and unfiltered contig annotations and from the AIRR files.
- `ds.repertoire.clonotypes(receptors)` groups cells into clonotypes by identical chains (scirpy's `define_clonotypes` rules) and saves a datastore-scoped `clonotypes` artifact. Options:
  - `sequence="nt"` (default) or `"aa"`: junction nucleotides or residues.
  - `receptor_arms="all"` (default: both arms for cells with both, one arm for cells with one; a paired cell never joins an orphan), `"any"` (either arm), `"VJ"` or `"VDJ"` (that arm; cells without it get no clonotype).
  - `dual_ir="any"` (default: one chain of an arm matches one of the other's), `"all"` (same number of chains, all matched, either order), `"primary_only"`.
  - `same_v_gene=False`, `same_j_gene=False`: also require matched chains' gene calls to match.
  - `within=None`: cell metadata columns (such as the donor) whose labels keep clonotypes apart; every cell with a receptor needs a label. `within_receptor_type=True` keeps TCR, BCR, ambiguous and multichain cells apart.
- Clonotypes are the connected components of the links, so a cell matching two others joins them. Labels `clonotype_1` (largest), `clonotype_2`, ...; ties follow the sorted chains, so labels do not depend on cell order. Arrays: canonical `values` (label, empty and treated as missing for a cell without one), `codes` (rank minus one, -1 without), `clone_size` (0 without). `select_cells(clonotypes, include=["clonotype_1"])` selects a clonotype's cells; `select_cells(clonotypes, value="clone_size", low=2, keep_bounds=True)` the expanded cells.
- Reuse follows the receptors, the options and the `within` labels; `invalidate_cache=True` defines again. About 1,000 bytes per cell at the peak, checked against `mem_budget`.
- `ds.repertoire.load_clonotypes(clonotypes)` returns a `Clonotypes` record: `.cells` (indexed by `ids`: `clonotype`, NA without, and `clone_size`), `.clonotypes` (one row per clonotype, largest first: `clonotype`, `n_cells`, `fraction` of cells with a clonotype, `n_configurations`, `receptor_type`, each `within` column, and the representative cell's chains per slot: `VJ_1_junction_aa`, `VJ_1_junction`, `VJ_1_v_call`, `VJ_1_j_call`, ..., `VDJ_2_j_call`), and the options (`.sequence`, `.receptor_arms`, `.dual_ir`, `.same_v_gene`, `.same_j_gene`, `.within`, `.within_receptor_type`, `.receptors`).
- `ds.plots.clonotype_abundance(clonotypes, top=20)` draws clonotypes per clone size (log-log) and the largest clonotypes labelled with their `VDJ_1` junction (`VJ_1` without a VDJ chain); `group_by` or `groups` gives one line per group and stacks the bars. Tables: `sizes` (`group`, `clone_size`, `n_clonotypes`), `top` (`clonotype`, `label`, cells per group or `cells`).
- Validated: equal to scirpy 0.25.1's identity clonotypes for every cell under ten option combinations on PBMC 10k and CD8+ donor 1; with `dual_ir="all"`, equal to scRepertoire 2.8.0's `CTnt`, `CTaa` (`sequence="aa"`) and `CTstrict` (both gene matches) and to Cell Ranger 3.0.2's clonotypes.
- `ds.repertoire.clonal_expansion(clonotypes, within=None, within_receptor_type=True, breakpoints=None, frequency_breakpoints=None)` counts each clonotype's cells within groups (the `within` columns' labels, such as the sample, times the receptor type) and saves a datastore-scoped `clonal_expansion` artifact. Every cell with a clonotype needs a `within` label. Pass at most one bound:
  - `breakpoints` (default `(1, 2, 5, 20, 100)`): categories `"1 cell"`, `"2 cells"`, `"3 to 5 cells"`, `"6 to 20 cells"`, `"21 to 100 cells"`, `"more than 100 cells"` (scirpy's `<= 1`, `<= 2`, `<= 5`, ..., `> 100`).
  - `frequency_breakpoints`, strictly increasing and ending at 1, such as scRepertoire's `(0.0001, 0.001, 0.01, 0.1, 1)` (Rare to Hyperexpanded): categories `"0 to 0.0001"`, ..., `"0.1 to 1"`, bins `(lower, upper]` of the clone's share of the group's cells with a clonotype.
- Arrays: canonical `values` (category, empty without a clonotype), `group_clone_size`, `group_clone_frequency`. `select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])` keeps the cells of large clones. Reuse follows the clonotypes, the bounds and the `within` labels. About 300 bytes per cell at the peak.
- `ds.repertoire.load_clonal_expansion(expansion, splits=(10, 100, 1000, 10000, 30000, 100000))` returns a `ClonalExpansion` record: `.cells` (`clonotype`, `category`, the group key columns (`within` columns, `receptor_type`), `group_clone_size`, `group_clone_frequency`), `.categories` (group key, `category`, `cells`, `clonotypes`, `fraction` of the group's cells; every category of every group), `.rank_windows` (group key, `window` such as `"1 to 10"`, `"11 to 100"`, ..., `"after 100000"`, `cells`, `fraction`: the cells of each group's clonotypes ranked by size into windows, scRepertoire's `clonalProportion`), `.labels`, `.breakpoints`, `.frequency_breakpoints`, `.splits`, `.within`, `.within_receptor_type`, `.clonotypes`.
- `ds.plots.clonal_expansion(expansion, group_by=None, groups=None, count="cells", normalize=True, splits=...)` draws stacked bars per group of the categories (cells, or `count="clonotypes"`) and of the rank windows. Groups default to the expansion's own; `group_by`/`groups` regroup the cells (categories stay those measured within the expansion's groups; ranks are recounted within the plotted groups). Tables: `categories` (`group`, `category`, `cells` or `clonotypes`, `fraction`), `rank_windows` (`group`, `window`, `cells`, `fraction`).
- Validated: every cell's category equals scirpy 0.25.1's `clonal_expansion` within pseudo samples and across all cells on PBMC 10k and CD8+ donor 1, and the plot's tables equal `summarize_clonal_expansion`; frequency fractions equal scRepertoire 2.8.0's `clonalHomeostasis` and rank windows its `clonalProportion` on PBMC 10k T cells.

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
paired = ds.select_cells(receptors, value="chain_pairing", include=["single pair", "extra VJ"])
# `umap_ref` is a UMAP of the same cells, such as the one of a ds.pipeline.rna run.
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
- B cell clonotypes from identical junctions split clones that carry somatic mutations; they are exact sequence variants, not clonal families.
- Measure expansion `within` the sample: without it, a clonotype's cells in all samples add up. Frequency categories depend on sample size (one cell of 1,000 is already 0.001), so compare them only between samples of similar size; bounds in cells compare any samples.
- Compare expansion between conditions with one value per sample (such as the fraction of cells in clonotypes of more than two cells), never with cells as replicates.
