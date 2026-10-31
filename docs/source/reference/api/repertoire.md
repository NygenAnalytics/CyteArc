---
myst:
  html_meta:
    description: CyteArc repertoire API for single-cell T cell and B cell receptor data, with contigs imported from Cell Ranger and AIRR Rearrangement files and joined to cells by barcode prefix, suffix or pattern, the ranked chains of each cell, its receptor type, subtype and chain pairing, clonotypes of cells that share their receptor, and the records of the receptors and clonotypes.
---

# Repertoire API reference

Single-cell immune profiling reads the T cell receptor (TCR) or B cell receptor (BCR) of each cell beside its transcriptome. A V(D)J library, such as 10x Genomics Chromium Single Cell Immune Profiling, assembles each cell's receptor transcripts into contigs: one chain each, with its locus, its V, D, J and constant genes, and its junction, the CDR3 with the conserved cysteine and phenylalanine or tryptophan on either side. A T cell receptor pairs an alpha chain (TRA) with a beta chain (TRB), or a gamma chain (TRG) with a delta chain (TRD), and a B cell receptor pairs a heavy chain (IGH) with a kappa (IGK) or lambda (IGL) light chain. `cytearc.repertoire` holds the record of the imported receptors and the constants of the receptor model; contig fields keep their names in the AIRR Community's Rearrangement schema ([Vander Heiden et al. 2018](https://doi.org/10.3389/fimmu.2018.02206)).

The steps run on a `DataStore` whose cells are the cells of the expression library. {meth}`ds.repertoire.import_receptors <cytearc.datastore.namespaces.RepertoireAccessor.import_receptors>` reads the contigs of those cells and saves an `immune_receptors` artifact, {meth}`ds.repertoire.load_receptors <cytearc.datastore.namespaces.RepertoireAccessor.load_receptors>` reads it as an {class}`~cytearc.repertoire.ImmuneReceptors` record, and {meth}`ds.plots.receptor_qc <cytearc.datastore.plot_accessor.DataStorePlotAccessor.receptor_qc>` plots the receptor types, chain pairings and chains per cell. {meth}`ds.repertoire.clonotypes <cytearc.datastore.namespaces.RepertoireAccessor.clonotypes>` groups the cells into clonotypes and saves a `clonotypes` artifact, which {meth}`ds.repertoire.load_clonotypes <cytearc.datastore.namespaces.RepertoireAccessor.load_clonotypes>` reads as a {class}`~cytearc.repertoire.Clonotypes` record; the clone summaries that the {doc}`clones` describes measure clonotypes, each receptor type apart.

## Import contigs

`import_receptors` reads two kinds of file and tells them apart by their header; `format="cellranger"` or `format="airr"` names the format of every file instead:

| File | Written by | Holds |
|---|---|---|
| `filtered_contig_annotations.csv` | Cell Ranger, in `outs/` of `cellranger vdj` or `outs/per_sample_outs/<sample>/vdj_t/` and `vdj_b/` of `cellranger multi` | The contigs that Cell Ranger assembled with high confidence in the barcodes it calls cells |
| `all_contig_annotations.csv` | Cell Ranger | Every contig; those outside a called cell or of low confidence are kept but do not count as chains |
| AIRR Rearrangement TSV | Cell Ranger (`airr_rearrangement.tsv`) and other pipelines that write the AIRR schema | One row per contig with the AIRR fields; single-cell files name each contig's cell in `cell_id` |

Cell Ranger's columns take their AIRR names: `barcode` becomes `cell_id`, `contig_id` `sequence_id`, `chain` `locus`, `v_gene`, `d_gene`, `j_gene` and `c_gene` become `v_call`, `d_call`, `j_call` and `c_call`, `cdr3` and `cdr3_nt` become `junction_aa` and `junction`, since Cell Ranger's CDR3 includes the conserved residues of the junction, `umis` `umi_count` and `reads` `consensus_count`. An AIRR file without `locus` has each contig's locus inferred from its gene calls, as the one locus that all of them name; a V call such as `TRAV29/DV5`, of a gene that alpha and delta chains share, takes no part. Every file needs a contig ID, a locus or gene calls, a junction amino-acid sequence and a productive flag, and an AIRR file also a `cell_id`; another pipeline's AIRR file reads as Cell Ranger's does when it holds them. `cytearc.repertoire.FIELDS` lists the fields that an import keeps, and other columns are not kept. Files may be plain or gzip-compressed, whatever their names, and blank lines hold no contig.

Pass the TCR and BCR files of one library together, so a cell with both kinds of chain is found:

```python
cells = ds.snapshot_cell_selection()
receptors = ds.repertoire.import_receptors(
    ["vdj_t/filtered_contig_annotations.csv", "vdj_b/filtered_contig_annotations.csv"],
    cell_selection=cells,
)
table = ds.repertoire.load_receptors(receptors)
table.cells["chain_pairing"].value_counts()
ds.plots.receptor_qc(receptors, group_by="sample")
```

### Join barcodes to cells

A contig's cell ID is its file's `barcode_prefix`, its barcode and `barcode_suffix`, matched exactly, and `barcode_pattern` keeps the part of each barcode that its named group `barcode` matches first, as {meth}`ds.clones.import_clones <cytearc.datastore.namespaces.ClonesAccessor.import_clones>` joins barcodes. Each option takes one value for every file or one per file. A merged store names cells `<sample>__<barcode>`, so pass each sample's files with that sample's prefix:

```python
receptors = ds.repertoire.import_receptors(
    ["s1/vdj_t.csv", "s1/vdj_b.csv", "s2/vdj_t.csv", "s2/vdj_b.csv"],
    cell_selection=cells,
    barcode_prefix=["s1__", "s1__", "s2__", "s2__"],
)
```

Contigs whose barcode matches no selected cell, such as the contigs of barcodes that the expression library did not call cells, are counted: the artifact's `n_file_contigs` and `n_unmatched_contigs` attributes count the files' contigs and those that match no cell, and `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` count their barcodes by the cell ID they name. A warning names up to five such barcodes, and an import that reuses the saved receptors gives the warning again. A selected cell without contigs is kept, with the receptor type `"no IR"`.

A file is identified by its format and the SHA-256 of its bytes, with its prefix, suffix and pattern, so a moved file reuses the saved receptors, an edited file gives new ones, and the order of the files does not matter. A file listed twice with the same prefix and suffix raises `ValueError`, as do a contig of a selected cell without an ID and a contig ID that repeats among the contigs of selected cells within one file; the contigs of other barcodes are only counted; the same ID in two files, such as `<barcode>_contig_1` in the TCR and the BCR file of one cell, is allowed. An import holds each distinct gene call and junction once, as bytes, with a code per contig, and each contig's ID, sequence and alignments as bytes of their own. At its peak, while it orders the contigs, it holds up to three times those bytes, about 400 bytes per selected cell and each cell's ID as text, which must fit the memory budget, as must reading the IDs, which decodes a stored chunk of 100,000 of them; an import that would not fit is refused before it writes anything. Encoded contigs take about 130 bytes each in Cell Ranger's annotations without region sequences, about 800 with them, and about 1,500 in AIRR files with sequences and alignments, so a million cells with two contigs each of the first kind need about 1.2 GB.

## Chains of each cell

CyteArc follows the receptor model of scirpy ([Sturm et al. 2020](https://doi.org/10.1093/bioinformatics/btaa611)). A cell has up to two chains of each arm of its receptor: the VJ arm, of the chains that join V and J segments (TRA, TRG, IGK, IGL), and the VDJ arm, of those that also join a D segment (TRB, TRD, IGH). Allelically included cells, with two alpha or two light chains, keep both.

A contig counts as a chain when it has the locus of a receptor chain and a junction amino-acid sequence, is productive, and is not flagged by Cell Ranger as outside a called cell (`is_cell`) or of low confidence (`high_confidence`). `productive_only=False` counts non-productive contigs too, such as the second, unproductive rearrangement of an allelically excluded cell. Within each arm, chains are ranked by UMI count (`umi_count`), then by AIRR's deprecated `duplicate_count`, which some files hold instead, then by read count (`consensus_count`), each most first and a missing count as 0, then by the junction nucleotide and amino-acid sequences, the one later in sort order first, as scirpy ranks them, and last by contig ID, so the ranking never depends on file order. The first two chains of each arm fill the cell's `VJ_1`, `VJ_2`, `VDJ_1` and `VDJ_2` slots, and a cell with more than two chains of an arm is a multichain cell, often a doublet.

The slots give each cell three labels, with scirpy's categories and rules:

| Label | Values |
|---|---|
| `receptor_type` | `"TCR"` (only T cell receptor chains), `"BCR"` (only B cell receptor chains), `"ambiguous"` (both), `"multichain"`, `"no IR"` (no chain) |
| `receptor_subtype` | `"TRA+TRB"` and `"TRG+TRD"` (chains of that receptor only), `"IGH"` (heavy chains only), `"IGH+IGK"` and `"IGH+IGL"` (light chains of one kind, with or without heavy chains), `"IGH+IGK/L"` (heavy chains with kappa and lambda chains), `"ambiguous"` (any other combination, such as TRA with TRD, or kappa and lambda chains without a heavy chain), `"multichain"`, `"no IR"` |
| `chain_pairing` | `"single pair"` (one chain of each arm), `"orphan VJ"` and `"orphan VDJ"` (chains of one arm only), `"extra VJ"` and `"extra VDJ"` (a pair and a second chain of that arm), `"two full chains"` (two chains of each arm), `"ambiguous"` (an ambiguous subtype), `"multichain"`, `"no IR"` |

Only the slots count towards the type and subtype, so a multichain cell's further chains decide nothing but its `"multichain"` label. `cytearc.repertoire` exports the categories in display order as `RECEPTOR_TYPES`, `RECEPTOR_SUBTYPES` and `CHAIN_PAIRINGS`.

The artifact's canonical `values` hold each cell's receptor type, and its other per-cell arrays are `receptor_subtype`, `chain_pairing`, `multichain`, `n_vj` and `n_vdj` (the cell's chains of each arm, before the slots keep two) and `n_contigs` (its imported contigs, chains or not). The labels group cells like cluster labels do: `ds.select_cells(receptors, include=["TCR"])` keeps the T cells with a receptor, `ds.select_cells(receptors, value="chain_pairing", include=["single pair"])` the cells with one chain of each arm, and `ds.plots.embedding(layout=run["umap"], color_by=receptors)` colours cells by receptor type. A colouring artifact must share the layout's cells, so import the receptors of a run's cells with `cell_selection=run["analysis_cell_selection"]`. [Read per-cell values](cell_aligned_kinds) lists the arrays.

On the public 10x Genomics run "Human PBMC from a Healthy Donor, 10k cells (v2)" (CC BY 4.0), with the TCR and BCR files imported together for the cells of its expression matrix, the receptor type, subtype and chain pairing of every cell, and the locus, gene calls, junction and junction amino-acid sequence in every slot, equal those of scirpy 0.25.1 from `read_10x_vdj`, `merge_airr`, `index_chains` and `chain_qc`: for the filtered and the unfiltered contig annotations of the Cell Ranger 4.0.0 release, 10,553 cells, whose files write `True` and `None` and whose filtered contigs include non-productive ones, and of the 5.0.0 release, 10,548 cells, and for the AIRR files of 5.0.0 read with `read_airr`, whose loci both tools infer. The same holds for the 55,206 cells of "CD8+ T cells of Healthy Donor 1" (Cell Ranger 3.0.2), whose unfiltered contig annotations name genes as reference 2.0.0 does, such as `TRAV29DV5`.

## Read and plot receptors

`load_receptors` returns an {class}`~cytearc.repertoire.ImmuneReceptors` record. Its `cells` table holds one row per cell: the three labels, `multichain`, `n_vj`, `n_vdj` and `n_contigs`, and the locus, gene calls, junction, junction amino-acid sequence, UMI count and read count of each slot side by side, such as `VDJ_1_junction_aa`, missing for an empty slot. Its `chains` table holds one row per contig: the cell's `ids`, the contig's `slot` (`""` for a contig in no slot), `chain` (whether it counts as a chain) and its fields. `fields=[...]` loads only those fields, into `chains` and into the slots of `cells`, which matters for files with full sequences and alignments; `fields=[]` loads each cell's labels and chain counts alone.

`ds.plots.receptor_qc` draws cells by receptor type, cells by chain pairing, and cells by their number of chains of each arm, three or more sharing the last bar. Many `"orphan VDJ"` or `"orphan VJ"` cells point to chains lost in capture, and many `"multichain"` or `"ambiguous"` cells to doublets. `group_by` names a cell metadata column, such as the sample, and `groups` a cell-label artifact, such as a clustering, which may hold fewer cells than the receptors; either draws one stacked bar per group, of the fraction of its cells unless `normalize=False`. Its tables are `receptor_type`, `chain_pairing` and `chains_per_cell`.

## Define clonotypes

A clonotype groups the cells of one T or B cell clone, which share their receptor because they descend from one cell. `clonotypes` links two cells when the junctions of their chains are identical, as scirpy's `define_clonotypes` does, and takes the connected groups of these links as clonotypes, so a cell whose chains match two cells that do not match each other joins them. The options set which chains must agree:

| Option | Values | Two cells are linked when |
|---|---|---|
| `sequence` | `"nt"`, the default, or `"aa"` | Their junctions are identical in nucleotides, or in amino acids |
| `receptor_arms` | `"all"`, the default | Both arms agree, for cells with a primary chain in both; one arm agrees, for cells with one; a cell with both arms is never linked to a cell with one |
| | `"any"` | Either arm agrees |
| | `"VJ"`, `"VDJ"` | That arm agrees; cells without a chain of that arm have no clonotype |
| `dual_ir` | `"any"`, the default | One chain of an arm matches one chain of the other cell's arm |
| | `"all"` | The cells have the same number of chains in the arm and every chain matches, in either order |
| | `"primary_only"` | The primary chains match; second chains are ignored |
| `same_v_gene`, `same_j_gene` | `False`, the defaults, or `True` | Matched chains also share their V or J gene call |

Cells of different receptor types never share a clonotype unless `within_receptor_type=False`, so the beta chain of an ambiguous cell does not join a T cell clonotype to its heavy chain's B cells, and `within` names cell metadata columns, such as the donor, whose labels keep clonotypes apart: identical receptors of two donors are public receptors, not one clone. Every cell with a receptor needs a label in the `within` columns. A cell without a receptor, or without a compared sequence among the chains that take part, has no clonotype.

```python
clonotypes = ds.repertoire.clonotypes(receptors)  # nucleotide junctions, both arms
same_residues = ds.repertoire.clonotypes(receptors, sequence="aa", within="donor")
table = ds.repertoire.load_clonotypes(clonotypes)
table.clonotypes.head(10)  # the largest clonotypes with their chains
ds.select_cells(clonotypes, value="clone_size", low=2, keep_bounds=True)  # expanded cells
ds.plots.clone_abundance(clonotypes, group_by="sample")
ds.clones.diversity(clonotypes, group_by="sample")  # each sample's T and B cells apart
```

Clonotypes are numbered by size: `clonotype_1` holds the most cells, and clonotypes of equal size follow their receptor type in sorted order, then their `within` labels in natural order, then their sorted chains, so the numbers do not depend on the order of the cells. The artifact's canonical `values` hold each cell's clonotype label, empty for a cell without one, which groupings and `select_cells` treat as missing and embeddings draw in the missing color; `clone_size` holds the cells of each cell's clonotype, 0 without one, and `codes` its number minus one, -1 without one. Defining clonotypes reads the chains of the receptors, not the contig files: a million cells of about two chains each take about 2 s and hold about 340 bytes per cell at the peak. The memory budget must allow the stored width of each cell's receptor type and 100 bytes more to read the types, the grouping of the cells by their `within` labels, which counts as the clone summaries' grouping does, and 1,000 bytes per cell and 12 per contig to define the clonotypes, which leaves room for cells with more chains. `load_clonotypes` counts 300 bytes per cell and 700 per clonotype beside the cells' IDs and the `within` labels as text, and for each representative field 25 bytes per value that it stores, 64 per value that a representative holds and 7 per byte of its packed text; it decodes the values 65,536 at a time. The clonotypes record the receptors, the options and, with `within`, the cells' labels, so defining them again with the same values returns the saved clonotypes. In the record that `load_clonotypes` returns, each cell's `clonotype` is a pandas categorical whose categories are the clonotype labels in rank order.

Each distinct combination of a cell's compared chains, its configuration, gets keys that the rules derive from its chains: under `"any"` one key per chain, under `"all"` the arm's sorted chains as one key, and with both arms required every combination of a VJ and a VDJ key. Two configurations are linked exactly when they share a key, and clonotypes are the connected components of the graph of configurations and keys, so the cost grows with the number of chains rather than the number of pairs of cells.

On "Human PBMC from a Healthy Donor, 10k cells (v2)", 6,189 cells with a receptor, and "CD8+ T cells of Healthy Donor 1", 46,526 cells with a receptor, the clonotypes equal scirpy 0.25.1's `define_clonotype_clusters` with the identity metric, cell for cell, for ten combinations of the options: `"nt"` and `"aa"`, every value of `receptor_arms` and `dual_ir`, the V and J gene matches, and without receptor types. With `dual_ir="all"` they also equal the clone keys of scRepertoire 2.8.0's `combineTCR` for all 5,213 PBMC T cells (`CTnt` with `"nt"`, `CTaa` with `"aa"`, and `CTstrict` with `"nt"` and both gene matches), and Cell Ranger 3.0.2's clonotypes, sets of identical productive junctions, for all 46,342 T cells of donor 1. Cell Ranger 5.0.0 groups clonotypes with enclone, which also merges cells that lack a chain, and agrees for 5,155 of the 5,213 PBMC T cells. On one core, defining the clonotypes of donor 1 takes 0.2 s, where scirpy's `ir_dist` and `define_clonotype_clusters` take 3.9 s on the same cells.

`load_clonotypes` returns a {class}`~cytearc.repertoire.Clonotypes` record. Its `cells` table holds each cell's `clonotype` and `clone_size`, and its `clonotypes` table one row per clonotype, largest first: `n_cells`, `fraction` (its share of the cells with a clonotype), `n_configurations`, its `receptor_type` and `within` labels, and the chains of its representative cell, the first cell of its most frequent configuration, slot by slot, such as `VDJ_1_junction_aa`, `VDJ_1_v_call` and `VJ_1_junction`, as pandas text, missing where the slot is empty.

### Measure clonotypes as clones

Clonotypes are clones to every clone summary that the {doc}`clones` describes. {meth}`ds.clones.load_clones <cytearc.datastore.namespaces.ClonesAccessor.load_clones>` reads them as a {class}`~cytearc.clones.Clones` record whose `cells` table also holds each cell's `receptor_type`, {meth}`ds.plots.clone_abundance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clone_abundance>` draws their sizes, labelled by rank, and {meth}`ds.clones.expansion <cytearc.datastore.namespaces.ClonesAccessor.expansion>`, {meth}`ds.clones.diversity <cytearc.datastore.namespaces.ClonesAccessor.diversity>` and {meth}`ds.clones.rarefaction <cytearc.datastore.namespaces.ClonesAccessor.rarefaction>` measure them within groups of cells. Each summary measures each receptor type apart, since T cell and B cell repertoires are populations of their own: the receptor type joins the groups' key columns as `receptor_type`, so the T cells and B cells of a sample are two groups, named `"s1, TCR"` and `"s1, BCR"`, and without a grouping the groups are the receptor types. A clonotype of several receptor types, which only `within_receptor_type=False` can give, counts its cells of each type in that type's group. `group_by="receptor_type"` is refused, since the labels are kept apart already.

```python
expansion = ds.clones.expansion(clonotypes, group_by="sample")
ds.clones.load_expansion(expansion).categories  # each sample's T and B cells
ds.clones.rarefaction(clonotypes, group_by="sample")
ds.clones.export_clones(clonotypes, "clonotypes.csv")  # CoSpar's clone table
```

## Immune receptors

```{eval-rst}
.. autoclass:: cytearc.repertoire.ImmuneReceptors
```

## Clonotypes

```{eval-rst}
.. autoclass:: cytearc.repertoire.Clonotypes
```
