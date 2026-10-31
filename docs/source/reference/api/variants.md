---
myst:
  html_meta:
    description: CyteArc variants API for mitochondrial lineage tracing and targeted genotyping, with per-cell allele counts imported from mgatk, maegatk, cellsnp-lite, MQuad, VarTrix, MiTo, giftwrap and the GoT tools, joined to cells by barcode, saved as a variant_counts artifact, read as tables of sites, alleles, depth, counts and calls, and drawn as depth per site and per cell.
---

# Variants API reference

Mitochondrial lineage tracing (mtscATAC-seq, MAESTER) and targeted genotyping (GoT, GIFT) count, for each cell, the reads or UMIs of each allele at a set of sites: every position of the mitochondrial genome, or targets and probes without coordinates. Each tool writes those counts in its own files. `ds.variants` reads them into one `variant_counts` artifact that holds the sites and their alleles, and each cell's depth at each site, its count of each allele and the calls the tool made itself.

{meth}`ds.variants.import_counts <cytearc.datastore.namespaces.VariantsAccessor.import_counts>` imports the counts of one file or folder, or several of one format, {meth}`ds.variants.load_counts <cytearc.datastore.namespaces.VariantsAccessor.load_counts>` reads them as a {class}`~cytearc.variants.VariantCounts` record, and {meth}`ds.plots.variant_depth <cytearc.datastore.plot_accessor.DataStorePlotAccessor.variant_depth>` draws the depth at each site and in each cell.

```python
cells = ds.snapshot_cell_selection("I")
counts = ds.variants.import_counts("mgatk/final", cell_selection=cells)
loaded = ds.variants.load_counts(counts, sites=[310, 3244])
loaded.counts.head()  # each cell's reads of each base at the two positions
ds.plots.variant_depth(counts, group_by="sample")
```

## Import counts

`import_counts` recognises each path's format from its contents: a folder by the names of its files, an HDF5 file by its groups and a table by its columns. `format=` names the format when the contents leave it open.

| `format` | Files | Sites and alleles | Unit | Depth counts |
|---|---|---|---|---|
| `"mgatk"` | an [mgatk](https://github.com/caleblareau/mgatk) folder from 0.4.7 on: `<name>.A.txt.gz` to `<name>.T.txt.gz`, `<contig>_refAllele.txt` and `<name>.depthTable.txt` | every position of the reference but `N`, with the four bases | reads | every base |
| `"maegatk"` | a [maegatk](https://github.com/caleblareau/maegatk) folder of the same files, always with six columns | as mgatk | molecules (UMI consensus reads) | every base |
| `"cellsnp_lite"` | a [cellsnp-lite](https://github.com/single-cell-genetics/cellsnp-lite) folder: `cellSNP.base.vcf`, `cellSNP.samples.tsv`, `cellSNP.tag.AD.mtx` and `.DP.mtx`, and `.OTH.mtx` where written | each record, with its reference, its alternative and, with OTH, `"other"`, the other bases | declared | DP plus OTH, or DP, the reference and alternative, without OTH |
| `"mquad"` | an [MQuad](https://github.com/single-cell-genetics/MQuad) folder: `passed_ad.mtx`, `passed_dp.mtx`, `passed_variant_names.txt` and `BIC_params.csv`, with the `cellSNP.samples.tsv` of the cellsnp-lite run it read | each passed variant, with its reference and alternative | declared | reference and alternative |
| `"vartrix"` | a folder of [VarTrix](https://github.com/10XGenomics/vartrix)'s coverage mode: the input VCF, the barcode list and the two matrices, the reference one with `ref` in its name | each record with one alternative, with both alleles; one record per position | declared | reference and alternative |
| `"mito"` | [MiTo](https://github.com/andrecossa5/MiTo)'s allele frequency matrix, an `.h5ad` file with `AD` and `DP` layers | each position, with its reference and alternatives | molecules where `uns` says so, otherwise declared | every base, the largest DP of the position's variants |
| `"giftwrap"` | [giftwrap](https://github.com/clareaulab/giftwrap)'s `counts.N.h5` or `counts.N.filtered.h5`, or a folder of GEO's `probe_reads.tsv.gz`, `barcodes.tsv.gz` and `manifest.tsv` of one sample | each probe, with the gapfills that cells show or the manifest names | UMIs | every gapfill |
| `"ironthrone"` | IronThrone-GoT's `<prefix>.summTable.txt` | the target, with `WT`, `MUT` and `ambiguous` | UMIs | every call |
| `"got"` | a per-cell GoT table of one target, as deposited on GEO: `BC` with `num.WT.call` and `num.MUT.call`, `WT.calls` and `MUT.calls`, or `wtUMIs` and `mutUMIs`, and a `Genotype` call | the target, with `WT` and `MUT` | UMIs | `WT` and `MUT` |
| `"got_multi"` | GoT-Multi-ML's `ironthrone_out_pp.csv` and `got_multi_out_refined.csv` | each target, and each target's gene-sequence and probe-barcode channels | UMIs | every call |
| `"gotcha"` | Gotcha's tables written with R's `write.csv`: `WTcount`, `MUTcount` and `genotype_pred`, or `<gene>_WTcount` and `<gene>_MUTcount` | each target, with `WT` and `MUT` | reads | `WT` and `MUT` |

mgatk with `--emit-base-qualities` and maegatk both write six columns, `position,barcode,forward,forward quality,reverse,reverse quality`, and count strands the other way round, so a six-column folder needs `format`. maegatk counts reverse-aligned reads as forward, and so did mgatk's `call` and `bcall` from 0.4.0 to 0.5.8: maegatk's strands are swapped back on import, and old mgatk files need `swap_strands=True`, since their files do not reveal the version. In RNA libraries, such as MAESTER's, strand follows the transcript, so strand tests apply to DNA libraries such as mtscATAC-seq.

Deposits rename and compress files, so files are matched by suffix: `TF1_filtered.chrM_refAllele.txt.gz` is a reference of the contig `chrM`, and nf-MiTo's `A.txt.gz` and `refAllele.txt`, which name no contig, read as mgatk files of an unnamed contig. maegatk writes its reference beside its tables, while nf-MiTo's MAESTER folders drop it; copy the `<contig>_refAllele.txt` of the run's reference into such a folder.

A one-target GoT table names its target after the file, without its extensions and `summTable`, so `Patient1_TET2.2847_summTable.txt` counts the site `Patient1_TET2.2847`. IronThrone's per-UMI lists are not read; the file's SHA-256 keeps them traceable. GoT-Multi counts each target in two evidence channels, which become the sites `BCOR (geneseq)` and `BCOR (probebc)`, while its genotypes, `genotype_geneseq`, `genotype_probebc` and `genotype_merged`, and the refined table's `genotype_final`, concern the target `BCOR`; both tables import together.

A GIFT probe's gapfills are read in the probe's orientation, the reverse complement of the transcript. The manifest's wild-type gapfill (`original_sequence`, or `original_gap_probe_sequence` on GEO) is the reference, its expected mutant (`gap_probe_sequence`) an alternative, and other gapfills, among them a control probe's empty gapfill, have the role `"other"`. Manifests often name neither, and some name one gapfill as both; that gapfill is the reference, and the source record lists such probes as `same_gapfills`. GEO's per-UMI tables give each cell's UMIs, reads (the sum of `pcr_duplicate_count`) and support (the mean of `percent_supporting`) per probe and gapfill, as giftwrap's HDF5 file does.

### Join barcodes to cells

A cell's ID is the path's `barcode_prefix`, the barcode and `barcode_suffix`, matched exactly, and `barcode_pattern` keeps the part of each barcode that its `barcode` group matches, by the rules that {doc}`clones` describes. Barcodes that match no selected cell are counted, saved in the artifact's `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` attributes, and named in a warning, which a reused import gives again.

Cell Ranger ARC writes the gene expression barcode into the ATAC BAM's `CB` tag, so mgatk run with `-bt CB` names cells as the RNA library does. A multiplexed Flex barcode holds 24 nucleotides, the GEM barcode and the probe barcode, so a giftwrap file of a multiplexed run joins a singleplex expression library with `barcode_pattern=r"^(?P<barcode>[ACGT]{16})"` and `barcode_suffix="-1"`.

### Several libraries

Several paths of one format import together, such as one mgatk folder per Cell Ranger ATAC library, each with its own prefix. Their shared sites are one site, so they must lie on the same reference: a contig with another length or sequence raises `ValueError`, as do one sequence under two contig names, such as mgatk references renamed `lib1_chrM_refAllele.txt` and `lib2_chrM_refAllele.txt`, a named and an unnamed contig, files that give one allele two roles, and, outside GIFT's open set of gapfills, files that give a site different alleles. The paths must also count in one unit, define depth alike and give their counts the same fields, so four-column and six-column mgatk folders, or cellsnp-lite folders with and without OTH, import apart; a table of calls alone, such as GoT-Multi's refined table, gives no counts. A cell that two files count at one site raises `ValueError`. A call that a source repeats for one cell and allele, as GoT-Multi-ML's refined table does for a few cells, is kept once and counted in the `n_repeated_calls` attribute, while two different calls raise `ValueError`. A source is identified by the SHA-256 of its file, or of each file of a folder, so moved files reuse the saved counts and the order of the paths does not matter.

```python
counts = ds.variants.import_counts(
    ["lib1/mgatk/final", "lib2/mgatk/final"],
    cell_selection=cells,
    barcode_prefix=["lib1__", "lib2__"],
)
```

### Units and depth

The unit of the counts is the format's own, except for cellsnp-lite, MQuad and VarTrix, which count UMIs or reads depending on how they ran: their import needs `count_unit="umis"` or `count_unit="reads"`. A MiTo matrix counts molecules when its `uns` says so: `uns['molecules']`, which MiTo 2 records when it turns mtscATAC reads into molecules, or a `pp_method` of maegatk or ReDeeM. Other MiTo matrices, such as mgatk reads that MiTo 1 assembled, need `count_unit`. Tools define depth two ways, which the artifact's `depth_definition` records: every read or UMI at the site (`"all"`), or those of the reference and alternative only (`"reference_and_alternative"`). cellsnp-lite's reference count is DP less AD, and its depth DP plus OTH.

Depth 0 means the cell was not covered at the site, so an allele frequency there is undefined rather than 0. Signac's `AlleleFreq` turns such cells into frequency 0; CyteArc stores no depth for them, so a reader can tell "not covered" from "covered, without the allele", which VarTrix's explicit zeros mark.

The import sorts each file's rows by cell, whatever their order, and rows that outgrow the memory budget go to sorted runs in the system's temporary folder (`TMPDIR` on Linux and macOS), about 4 bytes per count row of mgatk's files, removed when the import ends. mgatk's four base files parse on up to four threads when the budget is 1 GiB or more.

## Read counts

`load_counts` returns a {class}`~cytearc.variants.VariantCounts` record. Its `sites` table holds one row per site, indexed by label, such as `chrM:310`, `310` on an unnamed contig, or `IDH2_Mut1`: `contig`, `position`, `name`, `channel`, `reference`, and `depth` and `cells`, the depth summed over all of the artifact's cells and the cells covered. `alleles` holds each site's alleles and their roles, `cells` each cell's `listed`, `mean_depth` and `covered_sites`, and `contigs` each contig's name, length and the SHA-256 of its sequence.

The long tables hold one row per stored entry: `depth` (`cell`, `site`, `depth`) where a cell was covered, `counts` (`cell`, `site`, `allele`, `count`, then the fields the source has: `forward` and `reverse`, `quality_forward` and `quality_reverse`, NaN where the strand count is 0, `reads` and `support`) where a cell showed an allele, and `calls` (`cell`, `site`, `allele`, `source`, `call`) where the source made a call, on the mutant allele. A frequency is a count over the depth at its cell and site:

```python
loaded = ds.variants.load_counts(counts, sites=["chrM:310"])
table = loaded.counts.merge(loaded.depth, on=["cell", "site"])
table["frequency"] = table["count"] / table["depth"]
```

`sites` takes site labels, or positions on an artifact of one contig, and `cells` a cell selection within the counts' cells or cell IDs, as a list, a NumPy array or a pandas index; `cells=[]` reads only the sites with their sums. The tables are checked against the memory budget, at about 120 bytes per row, with the two stored chunks that a read's first and last rows decompress whole.

## The variant_counts artifact

The artifact holds a row per selected cell in its canonical `mean_depth`, the cell's depth summed over the sites and divided by their number, NaN for a cell that no file names, `covered_sites` and `listed`, so `ds.select_cells(counts, low=10)` keeps cells with a mean depth above 10. mgatk scores the cells whose mean depth exceeds 10 too, but averages over every position of its reference, the `N` at 3107 of rCRS included, so a cell within a few thousandths of 10 can fall on the other side. {ref}`Read per-cell values <cell_aligned_kinds>` lists them, and {doc}`artifacts` describes the sites, alleles and long tables, which are sorted by cell in row bands.

## Plot depth

`ds.plots.variant_depth(counts)` draws two panels: each site's mean depth over the cells that the files name, on a log scale, along the contig for positions and as bars for named sites; and how many cells have each mean depth over the sites. `group_by` names a cell metadata column, such as the sample, and `groups` a cell-label artifact; with either, the second panel draws one histogram per group. Its tables are `sites` (`site`, `contig`, `position`, `mean_depth`, `covered_fraction`) and `cells` (`cell`, `group`, `mean_depth`, `covered_sites`).

## VariantCounts

```{eval-rst}
.. autoclass:: cytearc.variants.VariantCounts
```
