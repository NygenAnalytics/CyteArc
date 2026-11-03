---
myst:
  html_meta:
    description: CyteArc variants API for mitochondrial lineage tracing and targeted genotyping, with per-cell allele counts imported from mgatk, maegatk, cellsnp-lite, MQuad, VarTrix, MiTo, giftwrap, the GoT tools, Tapestri and DAb-seq, joined to cells by barcode, scored with mgatk's per-variant statistics, filtered to informative variants by Signac's or MAESTER's rules, clustered into clones as Signac's FindClonotypes does, and genotyped at targeted sites by GoT's rules, GIFT's model or the tools' own calls.
---

# Variants API reference

Mitochondrial lineage tracing (mtscATAC-seq, MAESTER) and targeted genotyping (GoT, GIFT, Tapestri) count, for each cell, the reads or UMIs of each allele at a set of sites: every position of the mitochondrial genome or of a targeted panel, or targets and probes without coordinates. Each tool writes those counts in its own files. `ds.variants` reads them into one `variant_counts` artifact that holds the sites and their alleles, and each cell's depth at each site, its count of each allele and the calls the tool made itself.

{meth}`ds.variants.import_counts <cytearc.datastore.namespaces.VariantsAccessor.import_counts>` imports the counts of one file or folder, or several of one format, {meth}`ds.variants.load_counts <cytearc.datastore.namespaces.VariantsAccessor.load_counts>` reads them as a {class}`~cytearc.variants.VariantCounts` record, and {meth}`ds.plots.variant_depth <cytearc.datastore.plot_accessor.DataStorePlotAccessor.variant_depth>` draws the depth at each site and in each cell. {meth}`ds.variants.informative <cytearc.datastore.namespaces.VariantsAccessor.informative>` scores every variant across cells and selects the informative ones, and {meth}`ds.variants.clones <cytearc.datastore.namespaces.VariantsAccessor.clones>` calls the clones that those variants mark. For targeted sites, {meth}`ds.variants.genotypes <cytearc.datastore.namespaces.VariantsAccessor.genotypes>` calls each cell's genotype and {meth}`ds.variants.genotype_clones <cytearc.datastore.namespaces.VariantsAccessor.genotype_clones>` the clones that the mutant calls define.

```python
cells = ds.snapshot_cell_selection("I")
counts = ds.variants.import_counts("mgatk/final", cell_selection=cells)
loaded = ds.variants.load_counts(counts, sites=[310, 3244])
loaded.counts.head()  # each cell's reads of each base at the two positions
ds.plots.variant_depth(counts, group_by="sample")

deep = ds.select_cells(counts, low=10)
informative = ds.variants.informative(counts, cells=deep, preset="atac")
ds.plots.variant_statistics(informative)
clones = ds.variants.clones(counts, variants=informative)
ds.plots.allele_frequencies(informative, clones=clones)
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
| `"tapestri"` | a Tapestri Pipeline 2.0 `<sample>.dna+protein.h5` or `.dna.h5`, whose `assays/dna_variants` holds the layers `AF`, `DP`, `NGT` and `GQ` | each position of the variants that pass and those that `variants` names, with its reference and the variants' alternatives | reads | every read at the position, `DP` |
| `"dabseq"` | a [DAb-seq](https://github.com/AbateLab/DAb-seq) `.hdf5` file: `AD`, `RD`, `DP`, `GQ`, `GT`, `VARIANTS` and `CELL_BARCODES` | each position of its variants, with its reference and alternatives | reads | every read at the position, `DP` |

mgatk with `--emit-base-qualities` and maegatk both write six columns, `position,barcode,forward,forward quality,reverse,reverse quality`, and count strands the other way round, so a six-column folder needs `format`. maegatk counts reverse-aligned reads as forward, and so did mgatk's `call` and `bcall` from 0.4.0 to 0.5.8: maegatk's strands are swapped back on import, and old mgatk files need `swap_strands=True`, since their files do not reveal the version. In RNA libraries, such as MAESTER's, strand follows the transcript, so strand tests apply to DNA libraries such as mtscATAC-seq.

Deposits rename and compress files, so files are matched by suffix. mgatk names its reference after the contig, and a deposit that prefixes the base files prefixes the reference alike, so the contig is the reference's name without the base files' prefix: `TF1_filtered.chrM_refAllele.txt.gz` beside `TF1_filtered.A.txt.gz` names `chrM`, and `NC_012920.1_refAllele.txt` names `NC_012920.1`. A name that holds another dot, such as a prefix that the base files do not carry, leaves the contig unclear and raises `ValueError`. nf-MiTo's `A.txt.gz` and `refAllele.txt`, which name no contig, read as mgatk files of an unnamed contig. maegatk writes its reference beside its tables, while nf-MiTo's MAESTER folders drop it; copy the `<contig>_refAllele.txt` of the run's reference into such a folder.

A one-target GoT table names its target after the file, without its extensions and `summTable`, so `Patient1_TET2.2847_summTable.txt` counts the site `Patient1_TET2.2847`. IronThrone's per-UMI lists are not read; the file's SHA-256 keeps them traceable. GoT-Multi counts each target in two evidence channels, which become the sites `BCOR (geneseq)` and `BCOR (probebc)`, while its genotypes, `genotype_geneseq`, `genotype_probebc` and `genotype_merged`, and the refined table's `genotype_final`, concern the target `BCOR`; both tables import together. A header with one field fewer than the rows, as R's `write.table` writes it, leaves out the row names, which name the cells when no column does; a table whose every row ends in its delimiter with nothing after it is refused, as [clone tables](clones.md#import-clones-from-a-file) are.

A GIFT probe's gapfills are read in the probe's orientation, the reverse complement of the transcript. The manifest's wild-type gapfill (`original_sequence`, or `original_gap_probe_sequence` on GEO) is the reference, its expected mutant (`gap_probe_sequence`) an alternative, and other gapfills, among them a control probe's empty gapfill, have the role `"other"`. Manifests often name neither, and some name one gapfill as both; that gapfill is the reference, and the source record lists such probes as `same_gapfills`. GEO's per-UMI tables give each cell's UMIs, reads (the sum of `pcr_duplicate_count`) and support (the mean of `percent_supporting`) per probe and gapfill, as giftwrap's HDF5 file does.

### Tapestri and DAb-seq

A Tapestri Pipeline 2.0 file lists the candidate variants of its panel, 27,719 in Mission Bio's AML example, of which its default filters pass a few dozen (`filtered` 0), 40 there. The import reads the variants that pass and those that `variants` names: a variant as the file names it, such as `"chr1:115256513:G/T"`, or a position, such as `"chr1:115256513"`, which takes every variant there. Each position is a site, whose alleles are its reference and the alternatives of its chosen variants, so the variants of one position share its depth, `DP`. Where their references differ, as a SNV and a deletion of one position do once Tapestri splits them from one record, the longest reference is the site's and each shorter reference is extended, with its alternative, by the bases it lacks, as VCF writes several alternatives of one record: `T/G` beside `TC/T` reads as the allele `GC` of the reference `TC`. A variant's alternative reads are `AF × DP / 100`, a whole number of reads; a frequency and depth that give another number raise `ValueError`. Tapestri counts no reference reads, and the frequencies of the alternatives of one position may add up to more than 100%, so the depth bounds each alternative's reads rather than their sum. `NGT` becomes the call column `NGT`, with labels `0` to `3`, and `GQ` its `quality`, where the cell has reads at the position: a call without reads says nothing of the cell, and Tapestri writes some such cells as wild type at `GQ` 0.

Rows with an empty `ALT`, or alleles that are not sequences of bases, such as a `REF` of `*` or an `ALT` of `C+T`, which no documentation explains, are skipped with a warning and counted in the source record's `n_skipped_empty_alternative` and `n_skipped_other_alleles`, with up to five in `skipped_examples`; a variant that `variants` names exactly but cannot be read raises `ValueError`. An `ALT` of `*`, VCF's allele of a deletion that spans the position, is read as that allele. The source record also counts the file's variants, those that pass, those imported and those that `variants` adds, which it names by the SHA-256 of their names, so names that add the same variants reuse the import. It names the samples, the panel, the genome and the pipeline version, and keeps the thresholds of the pipeline's genotype filters, `gq_cutoff`, `dp_cutoff` and `af_cutoff`, 30, 10 and 20 in Mission Bio's example. A file's cells are named `<sample>__<barcode>`, as {class}`~cytearc.TapestriToZarr` names the cells of the store it writes from the same file, so the cells of several samples stay apart.

A DAb-seq file holds each cell's alternative and reference reads (`AD`, `RD`), its depth (`DP`) and its genotype (`GT`, with `GQ`) at every variant, named `<gene>:<chrom>:<pos>:<ref>/<alt>`, or `<chrom>:<pos>:<ref>/<alt>` by the pipeline's mode for other species, with missing values written as -1. `GT` is read as a diploid genotype, coded as `NGT`; the pipeline's haploid mode writes 1 for a mutant, which then reads as `heterozygous`. It flags no variant, so every variant whose alleles are bases is read; `variants` may name variants by their full name or as `chrom:pos:ref/alt`. The variants of one position count the same reference reads, which are stored once and must agree. The FLT3-ITD variants that the pipeline adds from ITDseek hold ITDseek's depth in `DP` and `RD`, so each is a named site of its own, such as `FLT3-ITD:chr13:28608251:T/TCTC`, with alternative reads alone; the pipeline calls FLT3-ITD wild type in every cell where ITDseek found none, and those calls, without reads, are left out like Tapestri's. Cells are named as `CELL_BARCODES` names them, such as `AACAACCTAGTCTTCTCG-1-all`, the barcode, the tube and the sample.

Before it reads a file's candidate variants and cells, which it holds as text while it chooses the variants, the import checks them against the memory budget: about 700 bytes for each Tapestri variant (1,100 for DAb-seq, whose names it splits into more parts) and 300 for each cell, with two bytes for each byte of their longest texts, four for DAb-seq's names, so Mission Bio's example of 27,719 variants and 1,313 cells needs about 26 MB. Each block of cells then reads its cells' IDs with its reads, so the import holds no cell's ID beyond its block.

```python
counts = ds.variants.import_counts(
    "sample.dna+protein.h5",
    cell_selection=cells,
    variants=["chr1:115256513:G/T"],
)
genotypes = ds.variants.genotypes(counts, method="source")
```

### Join barcodes to cells

A cell's ID is the path's `barcode_prefix`, the barcode and `barcode_suffix`, matched exactly, and `barcode_pattern` keeps the part of each barcode that its `barcode` group matches, by the rules that {doc}`clones` describes. Barcodes that match no selected cell are counted, saved in the artifact's `n_file_barcodes`, `n_unmatched_barcodes` and `unmatched_barcode_examples` attributes, and named in a warning, which a reused import gives again.

Cell Ranger ARC writes the gene expression barcode into the ATAC BAM's `CB` tag, so mgatk run with `-bt CB` names cells as the RNA library does. A multiplexed Flex barcode holds 24 nucleotides, the GEM barcode and the probe barcode, so a giftwrap file of a multiplexed run joins a singleplex expression library with `barcode_pattern=r"^(?P<barcode>[ACGT]{16})"` and `barcode_suffix="-1"`.

### Several libraries

Several paths of one format import together, such as one mgatk folder per Cell Ranger ATAC library, each with its own prefix. Their shared sites are one site, so they must lie on the same reference: a contig with another length or sequence raises `ValueError`, as do one sequence under two contig names, such as mgatk references renamed `lib1_chrM_refAllele.txt` and `lib2_chrM_refAllele.txt`, a named and an unnamed contig, files that give one allele two roles, and, outside GIFT's open set of gapfills, files that give a site different alleles. The paths must also count in one unit, define depth alike and give their counts the same fields, so four-column and six-column mgatk folders, or cellsnp-lite folders with and without OTH, import apart; a table of calls alone, such as GoT-Multi's refined table, gives no counts. A cell that two files count at one site raises `ValueError`. A call that a source repeats for one cell and allele, as GoT-Multi-ML's refined table does for a few cells, is kept once and counted in the `n_repeated_calls` attribute, while two different calls raise `ValueError`. Saved counts are reused when each file that a source's reader takes holds the same bytes ({ref}`reuse of imports <import_reuse>`), so moved files reuse them.

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

The import sorts each file's rows by cell, whatever their order, and rows that outgrow the memory budget go to sorted runs in the system's temporary folder (`TMPDIR` on Linux and macOS), about 4 bytes per count row of mgatk's files, removed when the import ends. mgatk's four base files parse on up to four threads when the budget is 1 GiB or more. Before it reads a row, the import checks the selected cells' IDs against the budget, as text and as their read, which decodes a stored chunk of IDs at the width of the longest; the parsed chunks, the sort buffer and the bands share what the barcode join leaves.

## Read counts

`load_counts` returns a {class}`~cytearc.variants.VariantCounts` record. Its `sites` table holds one row per site, indexed by label, such as `chrM:310`, `310` on an unnamed contig, or `IDH2_Mut1`: `contig`, `position`, `name`, `channel`, `reference`, and `depth` and `cells`, the depth summed over all of the artifact's cells and the cells covered. `alleles` holds each site's alleles and their roles, `cells` each cell's `listed`, `mean_depth` and `covered_sites`, and `contigs` each contig's name, length and the SHA-256 of its sequence.

The long tables hold one row per stored entry: `depth` (`cell`, `site`, `depth`) where a cell was covered, `counts` (`cell`, `site`, `allele`, `count`, then the fields the source has: `forward` and `reverse`, `quality_forward` and `quality_reverse`, NaN where the strand count is 0, `reads` and `support`) where a cell showed an allele, and `calls` (`cell`, `site`, `allele`, `source`, `call`, `quality`) where the source made a call, on the mutant allele, with the call's quality where the source gives one, such as Tapestri's `GQ`, and NaN elsewhere. A frequency is a count over the depth at its cell and site:

```python
loaded = ds.variants.load_counts(counts, sites=["chrM:310"])
table = loaded.counts.merge(loaded.depth, on=["cell", "site"])
table["frequency"] = table["count"] / table["depth"]
```

`sites` takes site labels, or positions on an artifact of one contig, and `cells` a cell selection within the counts' cells or cell IDs, as a list, a NumPy array or a pandas index; `cells=[]` reads only the sites with their sums. The tables are checked against the memory budget, at about 120 bytes per row, with the two stored chunks that a read's first and last rows decompress whole, the cells' IDs as text, whose read decodes a stored chunk of them, and 40 bytes for each cell of the counts, whose offsets and per-cell arrays are read whole, so loading a few cells of many counts needs room for every cell. Cells named by ID are found among the IDs of every cell of the counts, which a cell selection does not need.

## The variant_counts artifact

The artifact holds a row per selected cell in its canonical `mean_depth`, the cell's depth summed over the sites and divided by their number, NaN for a cell that no file names, `covered_sites` and `listed`, so `ds.select_cells(counts, low=10)` keeps cells with a mean depth above 10. mgatk scores the cells whose mean depth exceeds 10 too, but averages over every position of its reference, the `N` at 3107 of rCRS included, so a cell within a few thousandths of 10 can fall on the other side. {ref}`Read per-cell values <cell_aligned_kinds>` lists them, and {doc}`artifacts` describes the sites, alleles and long tables, which are sorted by cell in row bands.

## Plot depth

`ds.plots.variant_depth(counts)` draws two panels: each site's mean depth over the cells that the files name, on a log scale, along the contig for positions and as bars for named sites; and how many cells have each mean depth over the sites. `group_by` names a cell metadata column, such as the sample, and `groups` a cell-label artifact; with either, the second panel draws one histogram per group. Its tables are `sites` (`site`, `contig`, `position`, `mean_depth`, `covered_fraction`) and `cells` (`cell`, `group`, `mean_depth`, `covered_sites`).

## Find informative variants

Most bases that differ from the reference in a few reads are sequencing errors, and a variant marks clones only when some cells carry it and others do not. `informative` scores every variant that a chosen cell shows, an alternative allele at a site such as `310T>C`, with the statistics of mgatk's `variant_stats` table, computed as [mgatk](https://github.com/caleblareau/mgatk) 0.9.0 computes them. A cell's allele frequency is its count of the allele over its depth at the site, and is undefined in a cell without depth there.

| Column | What it holds |
|---|---|
| `mean` | the allele's reads over the depth at its site, both summed over the cells |
| `variance` | the sample variance of the covered cells' allele frequencies |
| `vmr` | `variance / (mean + 1e-11)`, the variance-mean ratio |
| `n_cells_conf_detected` | cells with at least 2 reads of the allele on each strand |
| `strand_correlation` | the Pearson correlation of the forward and reverse reads of the cells that show the allele |
| `n_cells_over_5`, `n_cells_over_10`, `n_cells_over_20`, `n_cells_over_95` | cells whose allele frequency reaches 5, 10, 20 and 95% |
| `max_heteroplasmy` | the highest allele frequency of a cell |
| `mean_coverage` | the depth at the site summed over the cells, over their number |
| `n_cells_covered` | cells with depth at the site, which the variance covers |
| `mean_quality` | the mean base quality of the allele, as MAESTER computes it: the mean over the cells and strands with reads of the allele of their mean quality |

A variant is informative when it passes every threshold that is set and no mask lists it. `preset` names a published rule, and a threshold given by name replaces the preset's:

| `preset` | Thresholds | Source |
|---|---|---|
| `"atac"` | `min_cells=5`, `min_strand_correlation=0.65`, `min_vmr=0.01` | the thresholds that [Lareau et al. (2020)](https://doi.org/10.1038/s41587-020-0645-6) set for mtscATAC-seq, as [Signac](https://doi.org/10.1038/s41592-021-01282-5)'s vignette applies them |
| `"rna"` | `min_depth=20`, `min_quality=30`, `min_vmr=0.01` | the depth and quality that [Miller et al. (2022)](https://doi.org/10.1038/s41587-022-01210-8) required of MAESTER variants, with Signac's VMR |

VMR, depth and quality must exceed their thresholds, and the confidently detected cells and the strand correlation must reach theirs, as the published rules compare them; a statistic that is undefined, such as the strand correlation of a variant that one cell shows, fails. Errors appear on one strand, so the strand tests separate them from variants in DNA libraries such as mtscATAC-seq. In RNA libraries, strand follows the transcript, so `"rna"` tests no strand. MAESTER then chose variants by percentiles of their allele frequencies, which mgatk's columns do not hold; the VMR stands in for that test of variability and keeps many more variants: on 50 cells of MiTo's MAESTER data, `"rna"` keeps 470 of the 5,664 variants they show, where MiTo's own filter keeps 3, and a larger `min_vmr` keeps fewer. A threshold that the counts cannot test, such as a strand correlation of cellsnp-lite counts or a base quality of four-column mgatk files, raises `ValueError` before any row is read.

`cells` restricts the statistics to a cell selection within the counts' cells. mgatk scores cells whose mean depth exceeds 10, and `ds.select_cells(counts, low=10)` selects nearly the same cells, for the reason that the variant_counts artifact above gives. Signac scores the cells of its object, often filtered the same way. Cells that no file names are left out either way. The statistics read the stored rows in bands of cells within the memory budget, which counts each band's rows, the two stored chunks that a band's reads decompress beyond its rows, and 48 bytes for each cell of the counts, chosen or not, whose offsets are read whole.

### Masks

`mask` names lists of variants that artifacts mimic, in the coordinates of the revised Cambridge Reference Sequence, which hg38's `chrM` and GRCh38's `MT` share. A masked variant is scored but never informative. A listed variant matches by position and alternative allele, and the masks refuse counts whose contig is not 16,569 bases long, such as hg19's `chrM`.

| `mask` | Variants | Source |
|---|---|---|
| `"numt"` | the 25 false positives that two common polymorphic NUMTs, nuclear copies of mitochondrial DNA, cause when their reads align to the mitochondrial genome, such as `16293A>C` | Supplemental Table S1 of [Laricchia et al. (2022)](https://doi.org/10.1101/gr.276013.121) |
| `"rna_editing"` | the RNA-DNA differences of human mitochondrial transcripts, `295C>T`, `2617A>G`, `2617A>T`, `13710A>G` and `13710A>T`, which RNA libraries read as variants | [Bar-Yaacov et al. (2013)](https://doi.org/10.1101/gr.161265.113) |
| `"artifact_prone"` | every variant at 301, 302, 310, 316, 3107 and 16182, homopolymers that sequencers misread and the reference's `N`, which gnomAD filters | [Laricchia et al. (2022)](https://doi.org/10.1101/gr.276013.121) |

### How the statistics differ from Signac's

Signac's `IdentifyVariants` computes the same columns but gives a cell without depth an allele frequency of 0, and by default replaces the frequency of a cell with depth below 10 by the pooled mean before taking the variance. CyteArc keeps the frequency of an uncovered cell undefined, as mgatk does: the variance and the cell counts cover the covered cells. Where every cell is covered, the statistics equal Signac's without its variance stabilisation; where coverage is sparse, a variant that covered cells rarely carry has a larger variance and VMR in CyteArc than in Signac, whose zeros for uncovered cells lower both. Signac also scores every alternative base at every position, while `informative` lists those that some chosen cell shows, and mgatk those seen on both strands.

`load_informative` returns a {class}`~cytearc.variants.VariantSelection` record: its `variants` table holds one row per scored variant, indexed by a label such as `chrM:310T>C`, with `site`, `position`, `reference`, `alternative`, the statistics, `masked` and `informative`, and the record names the counts, the cells, the preset, the thresholds as applied, the masks and the number of cells, and says whether the counts give strands and base qualities. `ds.plots.variant_statistics(informative)` draws each variant's VMR against its strand correlation, as mgatk draws it, with the informative and masked variants marked and the thresholds drawn as lines; for counts without strands, the x axis is the mean depth instead.

## Call clones

`clones(counts, variants=informative)` calls clones from the allele frequencies of the informative variants, as Signac's `FindClonotypes` does. Each cell's neighbourhood holds the cell itself and the `n_neighbors - 1` cells most similar to it by the cosine similarity of the square roots of their allele frequencies; two cells that share a neighbour are joined with the Jaccard index of their neighbourhoods, their shared cells over the cells in either, and joins below 1/15 are dropped, as Seurat's `FindNeighbors` builds its shared-neighbour graph; CyteArc's Leiden algorithm then partitions the graph at `resolution`, where Signac uses Seurat's `FindClusters`. `n_neighbors=10` and `resolution=1` are Signac's defaults. Every cell is compared with every other cell, so the time grows with the square of the cells, and `within` compares the cells of each group apart. Cells with collinear profiles, such as cells that each show one informative variant, tie at similarity 1 and take the same few cells as neighbours, so every pair of them shares those and the graph grows with the square of their number too; a graph that does not fit the memory budget raises `MemoryError` before anything is saved.

Signac turns an uncovered cell's allele frequency into 0, which makes a cell that was not read at a variant look like one that lacks it. CyteArc compares two cells over the variants that both cover, and a pair that shares no covered variant, or no alternative allele at the variants it shares, has similarity 0 and is never joined. Cells covered at no informative variant, cells that show none of them and cells similar to no other cell carry no evidence of a clone and get none, where Signac assigns every cell to a cluster; the artifact counts them in `n_cells_without_coverage`, `n_cells_without_alternative` and `n_cells_without_neighbors`. The cosine similarity ignores how frequent an allele is, so a cell that shows a clone's variant in one read is joined to that clone, as in Signac.

`within` names a cell metadata column, such as the donor: clones never join cells of two groups, each clone's label starts with its group, as in `donor1:3`, and cells without a group get no clone. The groups are those that the clone summaries group cells into by a column, read and coded within the memory budget, and the artifact records them by each cell's group and the groups' labels. The clones cover the cells that `variants` scored, over the counts' cells, and are saved as a `clones` artifact, which `ds.clones.load_clones`, every clone summary and `ds.plots.clone_abundance` read. The bands of frequencies count what the statistics' bands count and 24 bytes more for each cell of the counts, its rows and group, and writing the clones is checked at about 50 bytes for each of the counts' cells beside its label, which the store saves at the width of the longest, so counts over many cells of which the files name few need room for every cell.

`ds.plots.allele_frequencies(informative, clones=clones)` draws each cell's allele frequency of each informative variant, cells ordered by clone under a bar of clone colours, with uncovered entries grey and apart from a frequency of 0. Without `clones`, it draws the scored cells in their order; `max_cells` caps the cells drawn, chosen evenly along the order.

## Call genotypes

Targeted genotyping asks what each cell carries at a few known mutation sites. `genotypes` calls each cell's genotype at the sites of a `variant_counts` artifact in one vocabulary and saves a `variant_genotypes` artifact, {meth}`ds.variants.load_genotypes <cytearc.datastore.namespaces.VariantsAccessor.load_genotypes>` reads it as a {class}`~cytearc.variants.VariantGenotypes` record, `genotype_clones` calls clones from the combinations of mutant calls that cells carry, and {meth}`ds.plots.genotypes <cytearc.datastore.plot_accessor.DataStorePlotAccessor.genotypes>` draws the calls.

```python
counts = ds.variants.import_counts(
    ["Patient1_CALR_summTable.txt", "Patient1_SF3B1_summTable.txt"],
    cell_selection=cells,
)
calls = ds.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
ds.variants.load_genotypes(calls).calls.head()  # each cell's call at each site
clones = ds.variants.genotype_clones(calls, within="donor")
ds.plots.genotypes(calls, group_by="cell_type", layout_key="UMAP", site="Patient1_CALR")
```

### Calls

A call says what a cell carries at a site, and keeps the source's own label, such as GoT's `MUT`:

| Call | What the cell carries |
|---|---|
| `wild_type` | the wild type only |
| `heterozygous` | the alternative allele on one copy |
| `homozygous_alternative` | the alternative allele on every copy |
| `alternative_present` | the alternative allele, without a claim of how many copies carry it, as GoT's UMIs of one transcript give |
| `missing` | no evidence at the site |
| `filtered` | evidence that the rule did not call, such as ambiguous UMIs alone |

A cell that the calls do not list at a site is `missing` there. `cytearc.variants.CALLS` lists the calls in the order of their stored codes, and `cytearc.variants.SOURCE_CALLS` maps the labels of each tool's call column to them.

### Methods

`method` chooses how the calls are made:

- `"source"` adopts the calls that the tool wrote into the imported files: GoT's `Genotype` on GEO, GoT-Multi's `genotype_merged` and its refined `genotype_final`, whose `FalsePositive`, a mutant call that its model rejected, is `filtered`, Gotcha's `genotype_pred`, whose `HET` is `heterozygous`, and Tapestri's `NGT` and DAb-seq's `GT`, whose 0 to 3 are `wild_type`, `heterozygous`, `homozygous_alternative` and no call. GoT's `MUT` is `alternative_present`. Tapestri and DAb-seq call each variant of a position apart, so a site with coordinates is genotyped once per alternative that holds calls, labelled by the site and its alleles, such as `chr1:115256513:G/A`, with the site's reference and the alternative as its `reference` and `alternative`; the label is Tapestri's name of the variant unless its reference was extended, as `chr1:2000:T/G` beside `TC/T` reads as `chr1:2000:TC/GC`. A named site, such as a GoT target or DAb-seq's FLT3-ITD, is genotyped once. Where the files wrote several call columns for a site, each cell takes its call from the first that holds one for it: the refined call, then the merged call, then a single call column, then the channels' calls, so the cells that GoT-Multi's model did not refine keep their merged call. A label of no call, such as GoT-Multi's `Unprofiled`, is `filtered` where the cell holds UMIs at the site or at its channels and `missing` where it holds none, and a cell with UMIs but no call is `filtered`. `min_quality` and `min_depth` filter the adopted calls, as Tapestri's pipeline filters genotypes: a call whose quality, such as Tapestri's `GQ`, is below `min_quality` or missing, or whose cell's depth at the site, or at the deepest of its channels, is below `min_depth`, is `filtered`, and keeps its label. Only Tapestri's and DAb-seq's calls carry a quality, so counts of other formats refuse `min_quality` with `ValueError` rather than filter every call, and so does `min_depth` at a genotyped site where no cell holds depth, as in a GoT table of calls alone. `min_quality=30, min_depth=10` applies the thresholds that Mission Bio's example records as `gq_cutoff` and `dp_cutoff`. Tapestri's `FILTER_MASK` layer, which marks the calls that fail those thresholds or the allele frequency's, `af_cutoff`, is not read, since the frequency is the counts' alternative reads over the depth.
- `"threshold"` calls each cell from its UMIs of the site's wild type and alternative, by the rule given, since GoT studies apply different ones, and refuses Tapestri's counts, which hold no reference reads. `min_mutant_umis=1` calls a cell `alternative_present` with one mutant UMI or more and `wild_type` with none, the rule that GEO GSE204845 states for its patient samples and that Single-cell_BPDCN's XV-seq analysis applies. `min_fraction=0.9` calls a cell mutant or wild type when that allele holds more than 90% of its UMIs, the rule that [Nam et al. (2019)](https://doi.org/10.1038/s41586-019-1367-0) applied to cell lines, whose many transcripts make PCR recombination more likely; the two may be combined. `min_umis`, 1 by default, is the fewest UMIs of the two alleles that a call needs, and a cell that no rule calls is `filtered`. On GoT-Multi's counts each channel of a target is a site of its own, so a cell's `n_mutant_sites` counts a target once per channel.
- `"gift"` applies the model of GIFT ([Blattman et al. 2026](https://doi.org/10.64898/2026.04.11.717967)) to giftwrap's counts, as the next section describes.

`"threshold"` and `"gift"` genotype the sites that have one wild type and one alternative, and `"source"` those with calls, unless `sites` names them. `alleles` names the wild type and alternative of a site whose files do not, such as a GIFT probe whose manifest lacks its gapfills: `alleles={"JAK2 c.1849G>T": ("CAC", "AAC")}`, in the probe's orientation, as the files give the gapfills. A wild type that no cell shows at the site raises `ValueError`, and an alternative that none shows gives a warning. `"gift"` reads the gapfills of giftwrap's counts and refuses counts of other tools. Saved genotypes are reused when the counts, the method and its rules are the same.

### GIFT's model

GIFT reads a mutation from the gapfill that a probe pair captures, and a UMI may come from template switching during PCR, a swap, which carries no genotype information. Each UMI scores each genotype with the larger of its swap probability and its gapfill's probability under the genotype. A gapfill's probability of the wild type is its likelihood under the wild type over the sum of its likelihoods under the wild type and the mutant, where each edit between the gapfill and an allele multiplies the likelihood by 0.01. GIFT gives any gapfill that does not match an allele a likelihood of 0.01 in cell lines, which equals this for a gapfill one edit away; the heterozygote scores the larger of half of each allele's probability. A cell's likelihood of a genotype is the product of its UMIs' scores, summed as logarithms so that many UMIs do not underflow, and the posterior divides it by the sum over the three genotypes, a uniform prior. The most probable genotype is called when its posterior exceeds `min_posterior`, 0.6 by default, and the cell holds at least `min_umis` UMIs, 1 by default, whose swap probability lies below 0.1; otherwise the cell is `filtered`. Each call keeps the three posteriors.

`swap_probabilities` gives the probability of a swap by a UMI's reads, starting at 1 read, such as `{1: 0.5, 2: 0.2, 4: 0.05}`, and a UMI takes the value of the most reads given that it reaches. GIFT's reproducibility code stores each library's likelihood that a UMI of so many reads is no swap, so give one minus it, and a probability of 1 for UMIs of few reads drops them, as GIFT drops them in cell lines. giftwrap stores each cell's UMIs and reads per probe and gapfill rather than per UMI, so each UMI takes its entry's reads per UMI, rounded down. Without `swap_probabilities` every UMI counts, as GIFT treats cell lines once their UMIs of few reads are dropped. GIFT learns the swap probabilities and the gapfill error rates of patient samples from controls; here the swap probabilities are given, and a gapfill that matches neither allele is weighed by its edits.

### The variant_genotypes artifact

The artifact holds a row per cell of the counts in its canonical `n_mutant_sites`, the sites called `heterozygous`, `homozygous_alternative` or `alternative_present`, and `n_called_sites`, those and the sites called `wild_type`, so `ds.select_cells(calls, low=0)` keeps the cells with a mutant call. {doc}`artifacts` describes its arrays.

`load_genotypes` returns a {class}`~cytearc.variants.VariantGenotypes` record. Its `sites` table holds each genotyped site's `reference` and `alternative`, `source`, the call columns that `"source"` adopted in the order a cell's call is taken from them, and the cells of each call, those that the calls do not list counted `missing`; `cells` holds each cell's `n_mutant_sites` and `n_called_sites` at the sites shown; and `calls` holds one row per cell and site with evidence: `cell`, `site`, `call`, `label`, `reference_umis` and `alternative_umis`, -1 for adopted calls, and GIFT's `posterior_wild_type`, `posterior_heterozygous` and `posterior_homozygous_alternative`, NaN for the other methods. `sites` reads the calls of some sites, which the `cells` table then counts.

Calling reads the counts in bands of cells within the memory budget, which counts each band's rows, about 144 bytes per row for `"threshold"`, 168 for `"source"` and 360 for `"gift"`, whose scores and posteriors hold the most; 72 bytes for each cell of the counts; the chunks that a band's reads decompress and that appending its calls rewrites; and 1 MiB that the store's objects hold whatever the counts. A band holds at least 1,000 rows, or every row, and every row of the cell that holds the most. Loading checks about 220 bytes per call row, 200 per cell and the read of the cells' IDs.

### Clones from genotypes

Cells that descend from one mutated ancestor share its mutations. `genotype_clones(calls)` takes each cell's mutant sites as a set and the sites at which it is `missing` or `filtered` as unknown rather than wild type. Two cells are linked when they agree at every site that both were called at, a Jaccard index of 1 over the sites that both know. Cells whose calls agree with two cells that disagree at a site they were not called at cannot be placed: cells that carry a founder mutation and were not read where a subclone differs from the rest of the founder's cells could belong to either. Such cells are left out as ambiguous, however many there are, so they never join the founder and its subclone into one clone. A cell read where its clone's partly read cells were not stays with them, since cells that share no read site do not disagree. The linked cells are the clones, numbered by size as `clone_1` and on, and a cell without a mutant call gets no clone. A doublet of two clones read at the sites where they differ carries a combination of its own and forms a clone of one cell. `sites` chooses the sites whose calls define the clones, and `within` names cell metadata columns, such as the donor, whose groups clones never cross.

Exact agreement makes the choice of sites matter. Single-cell DNA amplification loses one allele of a cell at some sites, allelic dropout, which turns a heterozygous call into a wild-type or homozygous one, and a cell whose call turns disagrees with the rest of its clone. Every site adds such chances, while only the sites that some but not most cells carry tell clones apart. In Mission Bio's example of four cell lines, the 40 variants that pass give 587 clones of at most 51 cells, and the 16 variants that 15% to 40% of the called cells carry give four clones of 194 to 351 cells, beside clones of 43 cells or fewer:

```python
genotypes = ds.variants.genotypes(counts, method="source")
sites = ds.variants.load_genotypes(genotypes).sites
mutant = sites["heterozygous"] + sites["homozygous_alternative"]
share = mutant / (mutant + sites["wild_type"])
clones = ds.variants.genotype_clones(
    genotypes, sites=share[share.between(0.15, 0.4)].index.tolist()
)
```

Filtering calls by `min_quality` and `min_depth` leaves more sites unknown, and so more cells ambiguous: with `min_quality=30, min_depth=10`, the 17 variants that 15% to 40% of the called cells then carry leave 809 of the 1,313 cells ambiguous.

The clones are saved as a `clones` artifact, which `ds.clones.load_clones`, every clone summary and `ds.plots.clone_abundance` read; its `ambiguous` array marks the cells left out as ambiguous, and its attributes count the clones, the sites, the mutant cells and the ambiguous cells. The calls are read whole, at about 48 bytes per call row and 80 per cell, beside a mask of each mutant cell's known sites and the 1 MiB that the store's objects hold.

### Plot genotypes

`ds.plots.genotypes(calls)` draws, for each site, one bar per group whose segments are the shares of the group's cells with each call; cells that the calls do not list at a site count as `missing`. `group_by` names a cell metadata column, such as the cell type, and `groups` a cell-label artifact; without either, each site has one bar of all cells, and cells without a group are left out. With `layout_key`, such as `"UMAP"`, a second panel draws the cells at the cell metadata columns `UMAP1` and `UMAP2`, coloured by their call at `site`, which may be left out when one site is drawn. Its tables are `calls` (`site`, `group`, `call`, `cells`, `share`) and, with a layout, `cells` (`cell`, `x`, `y`, `call`).

## VariantCounts

```{eval-rst}
.. autoclass:: cytearc.variants.VariantCounts
```

## VariantSelection

```{eval-rst}
.. autoclass:: cytearc.variants.VariantSelection
```

## VariantGenotypes

```{eval-rst}
.. autoclass:: cytearc.variants.VariantGenotypes
```
