---
myst:
  html_meta:
    description: CyteArc repertoire API for single-cell T cell and B cell receptor data, with contigs imported from Cell Ranger and AIRR Rearrangement files, the ranked chains of each cell, its receptor type, subtype and chain pairing, clonotypes of cells that share their receptor, clonal expansion within samples, clonal diversity with rarefaction and extrapolation, the clonotypes that samples or cell types share, gene usage and junction lengths, and the records of the receptors, clonotypes, expansion, diversity, overlap and usage.
---

# Repertoire API reference

Single-cell immune profiling reads the T cell receptor (TCR) or B cell receptor (BCR) of each cell beside its transcriptome. A V(D)J library, such as 10x Genomics Chromium Single Cell Immune Profiling, assembles each cell's receptor transcripts into contigs: one chain each, with its locus, its V, D, J and constant genes, and its junction, the CDR3 with the conserved cysteine and phenylalanine or tryptophan on either side. A T cell receptor pairs an alpha chain (TRA) with a beta chain (TRB), or a gamma chain (TRG) with a delta chain (TRD), and a B cell receptor pairs a heavy chain (IGH) with a kappa (IGK) or lambda (IGL) light chain. `cytearc.repertoire` holds the record of the imported receptors and the constants of the receptor model; contig fields keep their names in the AIRR Community's Rearrangement schema ([Vander Heiden et al. 2018](https://doi.org/10.3389/fimmu.2018.02206)).

The steps run on a `DataStore` whose cells are the cells of the expression library. {meth}`ds.repertoire.import_receptors <cytearc.datastore.namespaces.RepertoireAccessor.import_receptors>` reads the contigs of those cells and saves an `immune_receptors` artifact, {meth}`ds.repertoire.load_receptors <cytearc.datastore.namespaces.RepertoireAccessor.load_receptors>` reads it as an {class}`~cytearc.repertoire.ImmuneReceptors` record, and {meth}`ds.plots.receptor_qc <cytearc.datastore.plot_accessor.DataStorePlotAccessor.receptor_qc>` plots the receptor types, chain pairings and chains per cell. {meth}`ds.repertoire.clonotypes <cytearc.datastore.namespaces.RepertoireAccessor.clonotypes>` groups the cells into clonotypes and saves a `clonotypes` artifact, {meth}`ds.repertoire.load_clonotypes <cytearc.datastore.namespaces.RepertoireAccessor.load_clonotypes>` reads it as a {class}`~cytearc.repertoire.Clonotypes` record, and {meth}`ds.plots.clonotype_abundance <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonotype_abundance>` plots the clone sizes and the largest clonotypes. {meth}`ds.repertoire.clonal_expansion <cytearc.datastore.namespaces.RepertoireAccessor.clonal_expansion>` places each cell in an expansion category by its clone size within its sample and saves a `clonal_expansion` artifact, {meth}`ds.repertoire.load_clonal_expansion <cytearc.datastore.namespaces.RepertoireAccessor.load_clonal_expansion>` reads it as a {class}`~cytearc.repertoire.ClonalExpansion` record, and {meth}`ds.plots.clonal_expansion <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonal_expansion>` plots the expansion and clonal space of each sample or cell type. {meth}`ds.repertoire.diversity <cytearc.datastore.namespaces.RepertoireAccessor.diversity>` measures the diversity of each group's clonotypes as a {class}`~cytearc.repertoire.ClonalDiversity` record, {meth}`ds.repertoire.rarefaction <cytearc.datastore.namespaces.RepertoireAccessor.rarefaction>` rarefies and extrapolates it as a {class}`~cytearc.repertoire.Rarefaction` record, and {meth}`ds.plots.clonal_rarefaction <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonal_rarefaction>` plots the curves. {meth}`ds.repertoire.overlap <cytearc.datastore.namespaces.RepertoireAccessor.overlap>` compares groups by the clonotypes they share as a {class}`~cytearc.repertoire.ClonotypeOverlap` record, {meth}`ds.repertoire.track <cytearc.datastore.namespaces.RepertoireAccessor.track>` follows chosen clonotypes across groups as a {class}`~cytearc.repertoire.ClonotypeTracking` record, and {meth}`ds.plots.clonal_overlap <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonal_overlap>` and {meth}`ds.plots.clonotype_tracking <cytearc.datastore.plot_accessor.DataStorePlotAccessor.clonotype_tracking>` plot them. {meth}`ds.repertoire.gene_usage <cytearc.datastore.namespaces.RepertoireAccessor.gene_usage>` counts the genes of groups' chains as a {class}`~cytearc.repertoire.GeneUsage` record, {meth}`ds.repertoire.spectratype <cytearc.datastore.namespaces.RepertoireAccessor.spectratype>` their junction lengths as a {class}`~cytearc.repertoire.Spectratype` record, and {meth}`ds.plots.gene_usage <cytearc.datastore.plot_accessor.DataStorePlotAccessor.gene_usage>`, {meth}`ds.plots.gene_pairing <cytearc.datastore.plot_accessor.DataStorePlotAccessor.gene_pairing>` and {meth}`ds.plots.spectratype <cytearc.datastore.plot_accessor.DataStorePlotAccessor.spectratype>` plot them.

## Import contigs

`import_receptors` reads two kinds of file, and tells them apart by their header:

| File | Written by | Holds |
|---|---|---|
| `filtered_contig_annotations.csv` | Cell Ranger, in `outs/` of `cellranger vdj` or `outs/per_sample_outs/<sample>/vdj_t/` and `vdj_b/` of `cellranger multi` | The contigs that Cell Ranger assembled with high confidence in the barcodes it calls cells |
| `all_contig_annotations.csv` | Cell Ranger | Every contig; those outside a called cell or of low confidence are kept but do not count as chains |
| AIRR Rearrangement TSV | Cell Ranger (`airr_rearrangement.tsv`), TRUST4, BD Rhapsody, IgBLAST with Change-O, MiXCR, dandelion | One row per contig with the AIRR fields; single-cell files name each contig's cell in `cell_id` |

Cell Ranger's columns take their AIRR names: `barcode` becomes `cell_id`, `contig_id` `sequence_id`, `chain` `locus`, `v_gene`, `d_gene`, `j_gene` and `c_gene` become `v_call`, `d_call`, `j_call` and `c_call`, `cdr3` and `cdr3_nt` become `junction_aa` and `junction`, since Cell Ranger's CDR3 includes the conserved residues of the junction, `umis` `umi_count` and `reads` `consensus_count`. An AIRR file without `locus` has each contig's locus inferred from its gene calls, as the one locus that all of them name; a V call such as `TRAV29/DV5`, of a gene that alpha and delta chains share, takes no part. Every file needs a contig ID, a locus or gene calls, a junction amino-acid sequence and a productive flag; `cytearc.repertoire.FIELDS` lists the fields that an import keeps, and other columns are not kept.

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

Each contig joins the selected cell whose ID is its file's `barcode_prefix` followed by its barcode, matched exactly. A merged store names cells `<sample>__<barcode>`, so pass each sample's files with that sample's prefix: one prefix for every file, or one per file. Contigs whose barcode matches no selected cell, such as the contigs of barcodes that the expression library did not call cells, are counted in the artifact's `n_unmatched_contigs` and `n_unmatched_barcodes` attributes, a few are named in `unmatched_barcode_examples`, and a warning reports them. A selected cell without contigs is kept, with the receptor type `"no IR"`.

A file is identified by the SHA-256 of its bytes and its prefix, so a moved file reuses the saved receptors, an edited file gives new ones, and the order of the files does not matter. A file listed twice with the same prefix, a contig without an ID, and a contig ID repeated within one file raise `ValueError`; the same ID in two files, such as `<barcode>_contig_1` in the TCR and the BCR file of one cell, is allowed. An import holds each distinct gene call and junction once, as bytes, and a code per contig, and at its peak, while it orders the contigs, about four times the bytes of those codes and values, which must fit the memory budget: about 1 GB for a million cells with two contigs each.

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

The artifact's canonical `values` hold each cell's receptor type, and its other per-cell arrays are `receptor_subtype`, `chain_pairing`, `multichain`, `n_vj` and `n_vdj` (the cell's chains of each arm, before the slots keep two) and `n_contigs` (its imported contigs, chains or not). The labels group cells like cluster labels do: `ds.select_cells(receptors, include=["TCR"])` keeps the T cells with a receptor, `ds.select_cells(receptors, value="chain_pairing", include=["single pair"])` the cells with one chain of each arm, and `ds.plots.embedding(color_by=receptors)` colours cells by receptor type. [Read per-cell values](cell_aligned_kinds) lists the arrays.

On the public 10x Genomics run "Human PBMC from a Healthy Donor, 10k cells (v2)" (CC BY 4.0), with the TCR and BCR files imported together for the cells of its expression matrix, the receptor type, subtype and chain pairing of every cell, and the locus, gene calls, junction and junction amino-acid sequence in every slot, equal those of scirpy 0.25.1 from `read_10x_vdj`, `merge_airr`, `index_chains` and `chain_qc`: for the filtered and the unfiltered contig annotations of the Cell Ranger 4.0.0 release, 10,553 cells, whose files write `True` and `None` and whose filtered contigs include non-productive ones, and of the 5.0.0 release, 10,548 cells, and for the AIRR files of 5.0.0 read with `read_airr`, whose loci both tools infer. The same holds for the 55,206 cells of "CD8+ T cells of Healthy Donor 1" (Cell Ranger 3.0.2), whose unfiltered contig annotations name genes as reference 2.0.0 does, such as `TRAV29DV5`.

`load_receptors` returns an {class}`~cytearc.repertoire.ImmuneReceptors` record. Its `cells` table holds one row per cell: the three labels, `multichain`, `n_vj`, `n_vdj` and `n_contigs`, and the locus, gene calls, junction, junction amino-acid sequence, UMI count and read count of each slot side by side, such as `VDJ_1_junction_aa`, missing for an empty slot. Its `chains` table holds one row per contig: the cell's `ids`, the contig's `slot` (`""` for a contig in no slot), `chain` (whether it counts as a chain) and its fields. `fields=[...]` loads fewer fields into `chains`, which matters for files with full sequences and alignments; the table of cells keeps its slot fields.

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
ds.plots.clonotype_abundance(clonotypes, group_by="sample")
```

Clonotypes are numbered by size: `clonotype_1` holds the most cells, and clonotypes of equal size follow their sorted chains, so the numbers do not depend on the order of the cells. The artifact's canonical `values` hold each cell's clonotype label, empty for a cell without one, which groupings and `select_cells` treat as missing; `clone_size` holds the cells of each cell's clonotype, 0 without one, and `codes` its number minus one, -1 without one. Defining clonotypes reads the chains of the receptors, not the contig files, and holds about 1,000 bytes per cell at its peak, which must fit the memory budget. The clonotypes record the receptors, the options and, with `within`, the cells' labels, so defining them again with the same values returns the saved clonotypes.

Each distinct combination of a cell's compared chains, its configuration, gets keys that the rules derive from its chains: under `"any"` one key per chain, under `"all"` the arm's sorted chains as one key, and with both arms required every combination of a VJ and a VDJ key. Two configurations are linked exactly when they share a key, and clonotypes are the connected components of the graph of configurations and keys, so the cost grows with the number of chains rather than the number of pairs of cells.

On "Human PBMC from a Healthy Donor, 10k cells (v2)", 6,189 cells with a receptor, and "CD8+ T cells of Healthy Donor 1", 46,526 cells with a receptor, the clonotypes equal scirpy 0.25.1's `define_clonotype_clusters` with the identity metric, cell for cell, for ten combinations of the options: `"nt"` and `"aa"`, every value of `receptor_arms` and `dual_ir`, the V and J gene matches, and without receptor types. With `dual_ir="all"` they also equal the clone keys of scRepertoire 2.8.0's `combineTCR` for all 5,213 PBMC T cells (`CTnt` with `"nt"`, `CTaa` with `"aa"`, and `CTstrict` with `"nt"` and both gene matches), and Cell Ranger 3.0.2's clonotypes, sets of identical productive junctions, for all 46,342 T cells of donor 1. Cell Ranger 5.0.0 groups clonotypes with enclone, which also merges cells that lack a chain, and agrees for 98.9% of the PBMC T cells. Defining the clonotypes of donor 1 takes 0.4 s, where scirpy's `ir_dist` and `define_clonotypes` take 5.8 s on its 47,584 V(D)J barcodes.

### Clonotype clusters

T cells whose receptors differ by a residue or two often recognize the same epitope, and the B cells of one clone differ by somatic mutations, so identical junctions split groups that belong together. `metric` links chains whose junctions are similar instead, as scirpy's `define_clonotype_clusters` does, under the same rules for arms, dual chains, gene calls, groups and receptor types:

| `metric` | Two junctions are linked when |
|---|---|
| `"identity"`, the default | They are identical; `cutoff` stays None |
| `"hamming"` | They have one length and at most `cutoff` substitutions |
| `"normalized_hamming"` | They have one length and their substitutions are at most `cutoff` percent of it, rounded half up, as scirpy rounds them |
| `"levenshtein"` | At most `cutoff` substitutions, insertions and deletions turn one into the other |

```python
similar = ds.repertoire.clonotypes(receptors, sequence="aa", metric="hamming", cutoff=1)
families = ds.repertoire.clonotypes(
    b_receptors, sequence="nt", metric="normalized_hamming", cutoff=15,
    same_v_gene=True, same_j_gene=True, within="donor",
)
```

`cutoff` has no default for the distance metrics, since it decides the clusters; scirpy defaults to 2, which for `normalized_hamming` means 2 percent, and its B cell tutorial passes 15. The clusters label cells like clonotypes do and feed expansion, diversity and overlap alike; the saved record holds the metric and cutoff. Every pair of distinct junctions within the cutoff is found exactly: for cutoffs up to 3, two junctions of one length within that many substitutions share a key in which those positions are masked, and two junctions within that many edits share a variant with at most that many positions deleted from each, so only the junctions that share a key are compared; longer junctions with larger cutoffs, such as nucleotide junctions at 15 percent, compare every pair of one length, or of lengths within the cutoff. On 100,000 distinct junctions of 11 to 18 residues, the pairs take 0.6 s within one substitution, 2.9 s within two, and 0.7 s and 4.6 s within one and two edits.

The clusters equal scirpy 0.25.1's `ir_dist` and `define_clonotype_clusters` for every cell of both 10x runs, 6,189 PBMC cells and 46,526 CD8+ cells with a receptor, in nine settings: residues within one and two substitutions with both arms and either chain, within one with every chain matched, with either arm and primary chains, and with the VDJ arm and V gene matches; within one edit, both arms; within two edits with the VJ arm and every chain matched, and with V and J gene matches; and nucleotides within 15 percent with V gene matches. A setting takes 0.6 to 3.5 s on the CD8+ cells, where scirpy's distances take 8 to 209 s on two cores.

`load_clonotypes` returns a {class}`~cytearc.repertoire.Clonotypes` record. Its `cells` table holds each cell's `clonotype` and `clone_size`, and its `clonotypes` table one row per clonotype, largest first: `n_cells`, `fraction` (its share of the cells with a clonotype), `n_configurations`, its `receptor_type` and `within` labels, and the chains of its representative cell, the first cell of its most frequent configuration, slot by slot, such as `VDJ_1_junction_aa`, `VDJ_1_v_call` and `VJ_1_junction`.

`ds.plots.clonotype_abundance` draws how many clonotypes have each size on log scales, and the `top` (20) clonotypes with the most cells, each labelled with its representative's `VDJ_1` junction, or its `VJ_1` junction without a VDJ chain. With `group_by` or `groups`, sizes count the cells of each clonotype within each group, one line per group, and the bars stack by group. Its tables are `sizes` and `top`.

## Measure clonal expansion

A clonotype that holds more than one cell has expanded: its founding cell divided, as T and B cells do when they meet their antigen. `clonal_expansion` counts each clonotype's cells within groups of cells and places each cell in an expansion category by its clonotype's size in its group. The groups are the combinations of the labels of the `within` cell metadata columns, such as the sample, and of the receptor type, so T cells and B cells are counted apart unless `within_receptor_type=False`. Without `within`, a clonotype's cells count in every sample together, as scirpy's `clonal_expansion` counts them without `expanded_in`; every cell with a clonotype needs a label in the `within` columns.

```python
expansion = ds.repertoire.clonal_expansion(clonotypes, within="sample")
table = ds.repertoire.load_clonal_expansion(expansion)
table.categories  # cells and clonotypes of each sample in each category
large = ds.select_cells(expansion, include=["21 to 100 cells", "more than 100 cells"])
ds.plots.clonal_expansion(expansion, group_by="cell_type")
```

One of two kinds of bounds sets the categories:

| Option | Categories | As in |
|---|---|---|
| `breakpoints`, the default `(1, 2, 5, 20, 100)` | By the clonotype's cells in the group: `"1 cell"`, `"2 cells"`, `"3 to 5 cells"`, `"6 to 20 cells"`, `"21 to 100 cells"` and `"more than 100 cells"` | scirpy's `breakpoints`, whose categories are `"<= 1"`, `"<= 2"`, `"<= 5"` and so on |
| `frequency_breakpoints`, such as `(0.0001, 0.001, 0.01, 0.1, 1)` | By the clonotype's frequency, its share of the group's cells with a clonotype, in bins `(lower, upper]` that end at 1: `"0 to 0.0001"`, `"0.0001 to 0.001"`, `"0.001 to 0.01"`, `"0.01 to 0.1"` and `"0.1 to 1"` | scRepertoire's `clonalHomeostasis` and `cloneSize`, which name these bins Rare, Small, Medium, Large and Hyperexpanded |

A frequency depends on the size of its group: in a sample of 1,300 cells a clonotype of one cell already has the frequency 0.00077, so the first bin of these bounds stays empty, and a clonotype of two cells is Medium. Compare frequencies between groups of similar sizes, or use bounds in cells. The names of bounds in cells sort in their order, so groupings and legends that sort labels naturally keep them from the smallest clonotypes to the largest.

The artifact's canonical `values` hold each cell's category, empty for a cell without a clonotype, so `select_cells` keeps the cells of large clonotypes and `ds.plots.embedding(color_by=expansion)` colours cells by category. `group_clone_size`, the cells of the cell's clonotype in its group, and `group_clone_frequency`, their share of the group's cells with a clonotype, are its other per-cell arrays. The result records the clonotypes, the bounds and, with `within`, the cells' labels, so measuring again with the same values returns the saved result. The measurement holds about 300 bytes per cell at its peak.

`load_clonal_expansion` returns a {class}`~cytearc.repertoire.ClonalExpansion` record with three tables, whose groups are keyed by the `within` columns and `receptor_type`:

| Table | Rows | Columns |
|---|---|---|
| `cells` | One per cell | `clonotype`, `category`, the group's key, `group_clone_size`, `group_clone_frequency` |
| `categories` | One per group and category, empty categories included | The group's key, `category`, `cells`, `clonotypes`, and `fraction`, the share of the group's cells with a clonotype that the category holds |
| `rank_windows` | One per group and window of ranks | The group's key, `window`, `cells`, `fraction` |

The rank windows describe a group's clonal space as scRepertoire's `clonalProportion` does: the group's clonotypes are ranked by their cells in the group, and each window counts the cells of the clonotypes ranked in it. `splits` sets the windows; the default `(10, 100, 1000, 10000, 30000, 100000)` gives scRepertoire's windows `"1 to 10"` to `"30001 to 100000"`, and `"after 100000"` holds the cells that scRepertoire leaves out. Tied clonotypes hold the same cells, so ties do not change the counts. A sample whose ten largest clonotypes hold most of its cells is dominated by a few clones.

`ds.plots.clonal_expansion` draws one stacked bar per group of the share of its cells in each category, or of its clonotypes with `count="clonotypes"`, as scirpy's `summarize_clonal_expansion` counts them, and one stacked bar per group of the share of its cells in each rank window, leaving out windows without cells. The groups are the expansion's own unless `group_by` names a cell metadata column or `groups` a cell-label artifact: each cell keeps the category measured within its sample, so grouping by cell type shows which cell types hold the expanded cells of their samples, and the ranks are counted within the plotted groups. `normalize=False` draws counts. Its tables are `categories` and `rank_windows`.

Each 10x Genomics run below holds one donor, so four pseudo samples and six pseudo clusters were assigned from a hash of each barcode. The categories equal those of scirpy 0.25.1's `clonal_expansion` for every cell, within the pseudo samples and across all cells, with the default bounds and with `(1, 2)`: 6,189 cells of "Human PBMC from a Healthy Donor, 10k cells (v2)" and 46,526 cells of "CD8+ T cells of Healthy Donor 1", whose expanded CD8+ T cell clones fill every category. `summarize_clonal_expansion` by pseudo cluster, counting cells and clonotypes, equals the plot's tables. On the T cell contigs of the PBMC run, with clonotypes that equal scRepertoire's `CTstrict` and `CTaa` keys and the four pseudo samples, the fractions of `frequency_breakpoints=(0.0001, 0.001, 0.01, 0.1, 1)` equal scRepertoire 2.8.0's `clonalHomeostasis` within 4e-16, and the rank windows' cells equal its `clonalProportion`. On a million simulated T cells in eight samples, measuring takes 1.7 to 2.5 s and 260 MiB, and loading the tables 1.5 to 2.3 s and up to 460 MiB.

## Measure diversity

A repertoire's diversity is how many clonotypes it holds and how evenly its cells spread over them. `diversity` measures it for groups of cells: the combinations of the labels of the `group_by` cell metadata columns, such as the sample, of a `groups` label artifact, such as a clustering, and of the receptor type, so T cells and B cells are measured apart unless `within_receptor_type=False`. `cell_selection` keeps only some cells, such as the T cells. Cells without a clonotype or a label are left out, and a clonotype's size in a group is its cells there. Nothing is saved: the result follows from the clonotypes and the labels in about a second for a million cells.

```python
t_cells = ds.select_cells(receptors, include=["TCR"])
result = ds.repertoire.diversity(clonotypes, group_by="sample", cell_selection=t_cells)
result.indices  # one row per sample: Shannon entropy, Simpson indices, Chao1, ...
result.hill  # Hill numbers: observed, asymptotic, at a common size and coverage
curves = ds.repertoire.rarefaction(clonotypes, group_by="sample", cell_selection=t_cells)
ds.plots.clonal_rarefaction(clonotypes, group_by="sample", cell_selection=t_cells)
```

The `indices` table holds one row per group, with the group's key and, for its `n` cells and `S` clonotypes of frequencies `p`, `f1` of one cell and `f2` of two:

| Column | Definition | As in |
|---|---|---|
| `cells`, `clonotypes`, `singletons`, `doubletons` | `n`, `S`, `f1`, `f2` | |
| `coverage` | The sample coverage, the estimated share of the repertoire's cells whose clonotype the sample holds: `1 - f1 / n * (n - 1) f1 / ((n - 1) f1 + 2 f2)`, or with `f1 - 1` and `f2 = 1` without doubletons ([Chao and Jost 2012](https://doi.org/10.1890/11-1952.1)) | iNEXT's `SC` |
| `shannon` | `-sum(p log p)`, in nats | scRepertoire's `shannon` |
| `pielou` | Shannon entropy over its maximum, `log S`; NaN for one clonotype | scRepertoire's `pielou` and `norm.entropy` |
| `simpson`, `inverse_simpson`, `gini_simpson` | `sum(p**2)`, its inverse, and one minus it | scRepertoire's and immunarch's `inv.simpson` and `gini.simpson` |
| `gini` | The Gini coefficient of the clone sizes, 0 when all are equal | scRepertoire's and immunarch's `gini` |
| `d50` | The fewest clonotypes, largest first, that hold at least half the cells | scRepertoire's and immunarch's `d50` |
| `chao1` | `S + (n - 1) / n * f1**2 / (2 f2)`, or `S + (n - 1) / n * f1 (f1 - 1) / 2` without doubletons | iNEXT's and immunarch's Chao1 |
| `chao1_bias_corrected` | `S + f1 (f1 - 1) / (2 (f2 + 1))` | vegan's `estimateR` and scRepertoire's `chao1` |
| `ace` | The abundance-based coverage estimate with clonotypes of at most 10 cells rare ([Chao and Lee 1992](https://doi.org/10.1080/01621459.1992.10475194)); NaN when every rare clonotype is a singleton | vegan's `estimateR` and scRepertoire's `ace` |

Every observed index depends on how many cells a group holds: a larger sample finds more clonotypes. Hill numbers ([Hill 1973](https://doi.org/10.2307/1934352)) put the indices on one scale, the effective number of clonotypes: order 0 counts clonotypes, order 1, the exponential of Shannon entropy, weighs them by frequency, and order 2, the inverse Simpson index, counts the dominant ones. The `hill` table holds, for each group and order 0, 1 and 2, the `observed` Hill number; the `estimated` asymptotic one of the whole repertoire, Chao1 for order 0, the exponential of the entropy estimate of [Chao, Wang and Jost (2013)](https://doi.org/10.1111/2041-210X.12108) for order 1, and the inverse of Good's unbiased Simpson estimate for order 2, infinite when no clonotype holds two cells; the Hill number `at_size`, the common size, with its `coverage_at_size` and `method_at_size`; and the Hill number `at_coverage`, the common coverage, with its `size_at_coverage` and `method_at_coverage`. Below a group's size its Hill numbers are rarefied, the expected values of a random subsample, and above it extrapolated, as [Chao et al. (2014)](https://doi.org/10.1890/13-0133.1) derive, with the Shannon extrapolation of [Chao et al. (2023)](https://doi.org/10.1002/ecm.1588) that iNEXT 3.0.2 uses: the curve approaches the asymptote geometrically from the observed value and passes through the rarefied value one cell earlier. A size between whole numbers interpolates linearly, as iNEXT does. By default the common size is the smallest of the groups' doubled sizes and the common coverage the lowest of their coverages at doubled sizes, iNEXT's `estimateD` defaults, so no group is extrapolated beyond twice its size; `size` and `coverage` set them. Compare groups by the Hill numbers at a common coverage, or at a common size; a group far smaller than the others sets both defaults low, so pass `size` or `coverage`, or leave it out.

`rarefaction` computes each group's curves: one row per group, order and size, with the `size`, `method` (`"rarefaction"`, `"observed"` or `"extrapolation"`), `diversity` and `coverage`. By default each group gets iNEXT's 40 sizes, about half evenly from 1 cell to its size and half from there to twice its size; `endpoint` and `knots` change them, and `sizes` gives every group the same sizes. `ds.plots.clonal_rarefaction` draws one panel per order, the Hill number against the cells (`base="size"`) or against the sample coverage (`base="coverage"`), solid where rarefied, dashed where extrapolated, with a point at each group's observed size, and with `base="size"` a panel of each group's coverage against its size. A curve that still rises steeply at its observed point has not reached its plateau: more cells would find more clonotypes. iNEXT's bootstrap confidence intervals are not computed; compare conditions with each sample's estimate as one observation.

The implementation follows the papers rather than the code of iNEXT, which is licensed under the GPL, and validates against its outputs. On iNEXT's two example vectors, the four pseudo samples of the PBMC run and of the CD8+ run (11,523 to 11,745 cells each), all 46,526 CD8+ cells together, and six edge cases (no singletons, no doubletons, every clonotype a singleton, a lone singleton, one clonotype, two cells), the Hill numbers of iNEXT 3.0.2's curves equal CyteArc's within 1.4e-12 at every size, rarefied, observed and extrapolated, the coverages within 3e-9, the asymptotic estimates within 7e-13, and `estimateD` by size within 5e-13 and by coverage within 1e-7, the tolerance of iNEXT's root finding; the default sizes are iNEXT's. Where the two differ, CyteArc's rarefied coverage equals exact rational arithmetic within 1e-16, where iNEXT's carries errors up to 3e-12, because the chance of missing a clonotype is computed as a product of ratios rather than from log-gamma functions. The indices equal scRepertoire 2.8.0's `clonalDiversity` without bootstraps for all twelve of its metrics, immunarch 0.10.3's `repDiversity` (`chao1`, `d50`, `div`, `gini`, `gini.simp`, `inv.simp` and Hill numbers of orders 1 to 6) and vegan 2.7.6's `estimateR` (`S.chao1`, `S.ACE`), within 6e-14, on the same pseudo samples and edge cases. On a million simulated T cells in eight samples, `diversity` takes 1.2 s and `rarefaction` 1.5 s, holding at most 260 MiB; grouping the cells is checked against the memory budget at about 300 bytes per cell.

## Compare groups

Two samples of one donor share the clonotypes of clones that both sampled, such as a tumour and the blood, two time points, or a sorted and an unsorted population, and two cell types share the clonotypes of clones whose cells took both fates. `overlap` compares every pair of groups by their clonotypes. The groups are the combinations of the labels of `group_by` cell metadata columns and of a `groups` label artifact, among the cells of `cell_selection`; with `within_receptor_type`, the default, the receptor type is part of each group and only groups of one receptor type are compared, since a T cell and a B cell never share a clonotype. Cells without a clonotype or a label are left out. Nothing is saved.

```python
result = ds.repertoire.overlap(clonotypes, group_by="tissue", cell_selection=t_cells)
result.pairs  # one row per pair of tissues
ds.plots.clonal_overlap(clonotypes, group_by="tissue", cell_selection=t_cells)
tracked = ds.repertoire.track(clonotypes, group_by="timepoint", top=10)
ds.plots.clonotype_tracking(clonotypes, group_by="timepoint", order=["day 0", "day 7", "day 28"])
```

The `pairs` table holds one row per pair, `group_1` and `group_2` named by their labels joined by commas, with these columns, for groups of `S1` and `S2` clonotypes and clone sizes `x` and `y` of `X` and `Y` cells:

| Column | Definition | As in |
|---|---|---|
| `shared` | The clonotypes of both groups | scRepertoire's `raw`, immunarch's `public` |
| `jaccard` | `shared / (S1 + S2 - shared)` | scRepertoire's and immunarch's `jaccard` |
| `sorensen` | `2 shared / (S1 + S2)` | immunarch's `tversky` with its default weights |
| `overlap` | `shared / min(S1, S2)` | scRepertoire's and immunarch's `overlap` |
| `cosine` | `shared / sqrt(S1 S2)`, the cosine of presence | scRepertoire's `cosine` |
| `morisita_horn` | `2 sum(x y) / ((sum(x^2) / X^2 + sum(y^2) / Y^2) X Y)` ([Horn 1966](https://doi.org/10.1086/282436)) | scRepertoire's and immunarch's `morisita` |
| `abundance_cosine` | `sum(x y) / sqrt(sum(x^2) sum(y^2))`, the cosine of the clone sizes | immunarch's `cosine` |
| `shared_fraction_1`, `shared_fraction_2` | The share of each group's cells whose clonotype the other group holds | |

The first five count clonotypes and grow with the cells of both groups, since larger samples find more of the rare clonotypes that two samples share by chance; the Morisita-Horn index and the cosine of clone sizes weigh clonotypes by their cells, depend little on sample size, and come near 1 when the same expanded clones dominate both groups. `group_sizes` holds each group's cells and clonotypes. `ds.plots.clonal_overlap` draws one `index` as a heatmap of the pairs, `"morisita_horn"` by default, with values printed for at most 12 groups; its tables are `matrix` and `pairs`.

`track` follows clonotypes across groups, such as time points: by default the `top` (10) with the most cells in each group, together, as scRepertoire's `clonalCompare` chooses them, with ties going to the larger clonotype overall; `labels` names them instead. Its `cells` table holds each clonotype's cells and share of the cells of each group of its receptor type, zeros included. `ds.plots.clonotype_tracking` draws one line per clonotype across the groups, in their natural order or in `order`, each labelled with its representative's junction.

On the four pseudo samples of CD8+ donor 1, which share about 500 clonotypes per pair and whose expanded clones give a Morisita-Horn index of 0.994 to 0.997, and on PBMC 10k with clonotypes equal to scRepertoire's `CTstrict` and `CTaa`, every index equals immunarch 0.10.3's `repOverlap` and scRepertoire 2.8.0's `clonalOverlap` within 5e-14, for every pair. scRepertoire's help page gives abundance-weighted formulas for its `overlap`, `raw` and `cosine`, but its outputs follow the presence formulas above, which CyteArc reproduces. On a million simulated T cells in eight samples, `overlap` and `track` each take about 1.3 s and hold at most 270 MiB.

## Gene usage and junction lengths

A repertoire's genes and junction lengths summarize how its receptors were assembled: an expanded clone shows its genes and length as peaks, invariant cells such as MAIT and iNKT cells show their gene pairs (TRAV1-2 with TRAJ33, and TRAV10 with TRAJ18), and naive repertoires show smooth length distributions. `gene_usage` and `spectratype` count each cell by its chains in `slots`, by default its primary chain of each arm, `VJ_1` and `VDJ_1`, as scirpy reads a slot, and each chain within its locus, so TRA and TRB genes get their own shares. Nothing is saved.

```python
usage = ds.repertoire.gene_usage(receptors, genes="v_call", group_by="sample")
usage.table  # sample, locus, v_call, cells, fraction
pairs = ds.repertoire.gene_usage(receptors, genes=("v_call", "j_call"))
families = ds.repertoire.gene_usage(clonotypes, level="family", count="clonotypes")
lengths = ds.repertoire.spectratype(receptors, group_by="sample")
ds.plots.gene_usage(receptors, group_by="sample")
ds.plots.gene_pairing(receptors, cell_selection=t_cells)
ds.plots.spectratype(receptors, group_by="sample")
```

`genes` names one gene field, `v_call`, `d_call`, `j_call` or `c_call`, or several, such as `("v_call", "j_call")` for the V-J pairs of each chain. `level` names calls as called, with their allele (`"allele"`, such as `TRBV12-3*01`), as genes (`"gene"`, the default, `TRBV12-3`) or as families (`"family"`, `TRBV12`, the locus, segment and first number, so `TRAV29/DV5` becomes `TRAV29` and `IGKV1D-39` `IGKV1`); a call naming several genes, such as `TRBV6-2,TRBV6-3`, keeps its first. `spectratype` counts junction lengths in residues (`sequence="aa"`) or nucleotides (`"nt"`); a junction includes its conserved cysteine and phenylalanine or tryptophan, as Cell Ranger's CDR3 does. A chain without the counted value is left out.

The `source` is an `immune_receptors` artifact, whose cells with a chain count, or a `clonotypes` artifact, whose cells with a clonotype count; with a clonotypes artifact, `count="clonotypes"` counts each clonotype once per group and value instead of each cell, so expanded clones do not dominate the shares. Groups follow `group_by` columns and a `groups` label artifact among the cells of `cell_selection`; without either, every cell is one group. Each table holds the groups' key columns, `locus`, the counted values, `cells` or `clonotypes`, and `fraction`, the share of the group's counted chains of that locus. `ds.plots.gene_usage` draws, one panel per locus, the `top` (30) genes with the most chains as a heatmap of fractions by group, or bars without groups; `ds.plots.gene_pairing` draws each locus's V genes against its J genes; `ds.plots.spectratype` draws each group's length distribution, one panel per locus.

On "Human PBMC from a Healthy Donor, 10k cells (v2)" with the TCR and BCR files imported together, the counts of every V gene, J gene and V-J pair of the `VJ_1` and `VDJ_1` slots, and of every junction length in residues and nucleotides, equal scirpy 0.25.1's `group_abundance` of the slot's calls and its `spectratype` for each of the four pseudo samples: 95 VJ and 85 VDJ V genes, 1,651 and 658 V-J pairs, and 16 and 22 lengths. scRepertoire's `percentGenes` and `percentAA` parse its concatenated clone strings through immApex rather than counting chains; its gene shares of a pseudo sample divide by 1,179 cells, which neither the cells with a beta chain (1,322) nor those with both chains (1,185) reproduce, so they serve as no reference here. On a million simulated T cells in eight samples, `gene_usage` takes 1.5 s, V-J pairs 1.6 s and `spectratype` 1.9 s, holding at most 260 MiB.

## Immune receptors

```{eval-rst}
.. autoclass:: cytearc.repertoire.ImmuneReceptors
```

## Clonotypes

```{eval-rst}
.. autoclass:: cytearc.repertoire.Clonotypes
```

## Clonal expansion

```{eval-rst}
.. autoclass:: cytearc.repertoire.ClonalExpansion
```

## Clonal diversity

```{eval-rst}
.. autoclass:: cytearc.repertoire.ClonalDiversity
.. autoclass:: cytearc.repertoire.Rarefaction
```

## Clonotype overlap

```{eval-rst}
.. autoclass:: cytearc.repertoire.ClonotypeOverlap
.. autoclass:: cytearc.repertoire.ClonotypeTracking
```

## Gene usage

```{eval-rst}
.. autoclass:: cytearc.repertoire.GeneUsage
.. autoclass:: cytearc.repertoire.Spectratype
```
