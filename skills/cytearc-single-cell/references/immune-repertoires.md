# Immune receptor repertoires

Import the T cell and B cell receptor contigs of single-cell immune profiling (10x Genomics Chromium Single Cell Immune Profiling, BD Rhapsody, TRUST4 and other AIRR producers), check each cell's chains, and label cells by receptor type, subtype and chain pairing. Docs: <https://docs.nygen.io/CyteArc/api/repertoire.html>, <https://docs.nygen.io/CyteArc/analysis_with_agents.html>.

## When to use

- A store of the expression library's cells, and the V(D)J output of the same cells: `outs/filtered_contig_annotations.csv` from `cellranger vdj`, or `outs/per_sample_outs/<sample>/vdj_t/filtered_contig_annotations.csv` and `vdj_b/filtered_contig_annotations.csv` from `cellranger multi`.
- An AIRR Rearrangement TSV with a `cell_id` column: Cell Ranger's `airr_rearrangement.tsv`, TRUST4's barcode AIRR file, BD Rhapsody's AIRR output, IgBLAST with Change-O, MiXCR, dandelion.

## Key concepts

- `ds.repertoire.import_receptors(paths, cell_selection=cells)` reads one file or a list, each told apart by its header (Cell Ranger CSV or AIRR TSV, plain or gzip, whatever the name), joins each contig to a selected cell by barcode, and saves a datastore-scoped `immune_receptors` artifact. `format="cellranger"` or `format="airr"` names the format of every file instead; a file of the other format then fails on the columns it lacks. Import the TCR and BCR files of one library in one call.
- Contig fields keep their AIRR names; Cell Ranger's columns are renamed: `barcode` to `cell_id`, `contig_id` to `sequence_id`, `chain` to `locus`, `v_gene`/`d_gene`/`j_gene`/`c_gene` to `v_call`/`d_call`/`j_call`/`c_call`, `cdr3`/`cdr3_nt` to `junction_aa`/`junction` (Cell Ranger's CDR3 includes the conserved C and F/W), `umis` to `umi_count`, `reads` to `consensus_count`, and region columns such as `fwr1`/`fwr1_nt` to `fwr1_aa`/`fwr1`. `cytearc.repertoire.FIELDS` lists the kept fields; other columns are dropped.
- Required per file: a contig ID, `locus` or gene calls (an AIRR file without `locus` has it inferred from the calls that all name one locus), `junction_aa` and `productive`. AIRR files also need `cell_id`.
- A cell ID is `barcode_prefix` + barcode + `barcode_suffix`, matched exactly; `barcode_pattern` (a regex with a named group `barcode`) keeps part of each barcode first. Each takes one value for every file or a list with one per file. Merged stores name cells `<sample>__<barcode>`. The rules are those of `clones.import_clones` and `perturbation.import_guide_calls`.
- Contigs and barcodes outside the selection are counted (`n_file_contigs`, `n_unmatched_contigs`, `n_file_barcodes`, `n_unmatched_barcodes`, `unmatched_barcode_examples` attributes) and named in a warning, which a reused import gives again. Selected cells without contigs are kept as `"no IR"`.
- Reuse follows each file's format, bytes, prefix, suffix and pattern, in any order; a moved file reuses, an edited file re-imports, `invalidate_cache=True` imports again. About 4x the encoded contigs at the peak, checked against `mem_budget` (about 1 GB per million cells with two contigs each).
- Chains (scirpy's receptor model): a contig counts when it has a locus of a receptor chain, a `junction_aa`, is productive (`productive_only=True`, default) and is not flagged `is_cell` or `high_confidence` False. Ranking within each arm: `umi_count`, `duplicate_count`, `consensus_count` (most first, missing as 0), then `junction` and `junction_aa` (later in sort order first), then contig ID. Arms: VJ (TRA, TRG, IGK, IGL) and VDJ (TRB, TRD, IGH). Up to two chains per arm fill `VJ_1`, `VJ_2`, `VDJ_1`, `VDJ_2`; more than two in an arm makes the cell `"multichain"`.
- Labels, from the slots only:
  - `receptor_type` (canonical `values`): `"TCR"`, `"BCR"`, `"ambiguous"` (both kinds), `"multichain"`, `"no IR"`.
  - `receptor_subtype`: `"TRA+TRB"`, `"TRG+TRD"`, `"IGH"`, `"IGH+IGK"`, `"IGH+IGL"`, `"IGH+IGK/L"`, `"ambiguous"`, `"multichain"`, `"no IR"`. A light chain without a heavy chain still names its subtype (`"IGH+IGK"`), as in scirpy.
  - `chain_pairing`: `"single pair"`, `"orphan VJ"`, `"orphan VDJ"`, `"extra VJ"`, `"extra VDJ"`, `"two full chains"`, `"ambiguous"`, `"multichain"`, `"no IR"`.
- Other per-cell arrays: `multichain`, `n_vj` and `n_vdj` (chains of each arm before the slots keep two), `n_contigs` (imported contigs, chains or not). `select_cells` and `artifacts.load_values` read them; `chain_slots` and the `contig_*` arrays are not cell-aligned.
- `ds.repertoire.load_receptors(receptors, fields=None)` returns an `ImmuneReceptors` record: `.cells` (indexed by `ids`: the three labels, `multichain`, `n_vj`, `n_vdj`, `n_contigs`, then per slot `<slot>_locus`, `_v_call`, `_d_call` (VDJ slots only), `_j_call`, `_c_call`, `_junction`, `_junction_aa`, `_umi_count`, `_consensus_count`), `.chains` (one row per contig: `ids`, `slot` (`""` for none), `chain` (counts as a chain) and each field; missing text is NA, counts `Int64`, flags `boolean`), `.fields` and `.productive_only`. `fields=[...]` loads only those fields, into `.chains` and into the slots of `.cells`; large AIRR files with sequences and alignments need it.
- `ds.plots.receptor_qc(receptors)` draws cells per receptor type, per chain pairing, and cells by chains of each arm (0, 1, 2, 3+). `group_by="sample"` (a cell column) or `groups=clusters_ref` (a label artifact, which may hold fewer cells) stacks one bar per group, as fractions unless `normalize=False`. Tables: `receptor_type`, `chain_pairing`, `chains_per_cell`.
- Validated: every cell's labels and every slot's locus, gene calls and junctions equal scirpy 0.25.1's on 10x PBMC 10k (TCR and BCR, Cell Ranger 4.0.0 and 5.0.0) and CD8+ donor 1 (Cell Ranger 3.0.2), from the filtered and unfiltered contig annotations and from the AIRR files.

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
# Cell IDs that end in "-1" while an AIRR file names bare barcodes.
airr = ds.repertoire.import_receptors("airr_rearrangement.tsv", cell_selection=cells, barcode_suffix="-1")
```

## Checks

- Read `n_unmatched_contigs`, `n_unmatched_barcodes` and the warning. A few percent of barcodes outside the expression cells is usual (V(D)J and expression call cells separately); most barcodes unmatched means a wrong `barcode_prefix`, or barcodes with another suffix than the cell IDs (`barcode_suffix`, `barcode_pattern`).
- `"no IR"` among cells whose expression marks them as T or B cells points to a shallow V(D)J library; `"no IR"` monocytes and NK cells are expected.
- `"orphan VJ"`/`"orphan VDJ"` cells lost a chain in capture and still carry clonal information. `"extra VJ"` (two alpha or two light chains) is often biological. `"extra VDJ"`, `"two full chains"`, `"multichain"` and `"ambiguous"` are more often doublets or ambient chains: count them per sample and compare with `qc.doublets` before removing anything.
- An `"ambiguous"` T cell with a heavy chain, or B cell with a beta chain: scattered in expression clusters points to ambient contigs, a cluster of its own to doublets.
- Keep `productive_only=True` for clonotypes; `False` is for studying non-productive rearrangements.
- Import TCR and BCR files together: imported apart, a cell with both kinds of chain is not `"ambiguous"` in either artifact.
- Gamma-delta chains need a gamma-delta enrichment kit; a standard TCR library holds few TRG or TRD contigs.
