# Mitochondrial and targeted variants

Import each cell's counts of each allele at variant sites, from mitochondrial lineage tracing (mtscATAC-seq, MAESTER) or targeted genotyping (GoT, GIFT), join them to the store's cells, check depth, read the counts as tables, select informative mitochondrial variants and call the clones they mark. Docs: <https://docs.nygen.io/CyteArc/api/variants.html>.

## When to use

- Per-cell allele counts from mgatk (`<name>.{A,C,G,T}.txt.gz` with `<contig>_refAllele.txt`), maegatk (the same files, six columns), cellsnp-lite (`cellSNP.tag.{AD,DP,OTH}.mtx`), MQuad (`passed_ad.mtx`, `passed_dp.mtx`), VarTrix's coverage mode (alt and ref matrices, VCF, barcodes), MiTo's allele frequency matrix (`.h5ad` with `AD` and `DP` layers), giftwrap (`counts.N.h5`, or GEO's `probe_reads.tsv.gz` with `barcodes.tsv.gz` and `manifest.tsv`), and GoT tables (IronThrone's `summTable.txt`, GEO's per-cell tables, GoT-Multi's `ironthrone_out_pp.csv` and `got_multi_out_refined.csv`, Gotcha's CSVs).
- Not for BAM or FASTQ files: run the tool first. Not for clones that a tool already called (MiTo's `obs['MiTo_clone']`, Numbat): use `ds.clones.import_clones`.

## Key concepts

- `ds.variants.import_counts(paths, *, cell_selection, barcode_prefix=None, barcode_suffix=None, barcode_pattern=None, format=None, count_unit=None, swap_strands=False, invalidate_cache=False)` saves a datastore-scoped `variant_counts` artifact. Each path is a folder (mgatk, maegatk, cellsnp-lite, MQuad, VarTrix, giftwrap's GEO tables) or a file (giftwrap's HDF5, MiTo's h5ad, a GoT table), and several paths must share a format. `format` is one of `"mgatk"`, `"maegatk"`, `"cellsnp_lite"`, `"mquad"`, `"vartrix"`, `"mito"`, `"giftwrap"`, `"ironthrone"`, `"got"`, `"got_multi"`, `"gotcha"`. Rows beyond the memory budget spill to sorted runs in `TMPDIR`, about 4 bytes per mgatk count row.
- Six-column mgatk files (`--emit-base-qualities`) look like maegatk's, which count strands the other way round, so such a folder raises `ValueError` until `format` names it. maegatk's strands are swapped back on import; mgatk `call` or `bcall` output from 0.4.0 to 0.5.8 needs `swap_strands=True`. Strand only means something in DNA libraries (mtscATAC-seq), not RNA (MAESTER).
- cellsnp-lite, MQuad and VarTrix need `count_unit="umis"` or `"reads"`, since their files do not say; other formats count in their own unit (mgatk reads, maegatk molecules, giftwrap and GoT UMIs, Gotcha reads), and a different `count_unit` raises `ValueError`. A MiTo matrix counts molecules when its `uns` holds `molecules` or a `pp_method` of maegatk or ReDeeM; otherwise, such as for mgatk reads that MiTo 1 assembled, it needs `count_unit`.
- Sites: every reference position but `N` (mgatk, maegatk; 16,568 for rCRS), each VCF record (cellsnp-lite, VarTrix), each passed variant (MQuad), each position of the matrix (MiTo), each probe (giftwrap) or target (GoT). A one-target GoT table names its site after the file (`Patient1_TET2.2847_summTable.txt` gives `Patient1_TET2.2847`). GoT-Multi's channels are sites `BCOR (geneseq)` and `BCOR (probebc)`, and its genotypes sit on `BCOR`.
- Alleles have roles: `"reference"` (reference base, wild type), `"alternative"` (other base, mutant, a GIFT manifest's expected gapfill) and `"other"` (cellsnp-lite's other bases, GoT's ambiguous UMIs, gapfills no manifest names).
- Depth 0 means not covered: an allele frequency there is undefined, never 0. The `depth_definition` attribute says whether depth counts every read (`"all"`) or only the reference and alternative (`"reference_and_alternative"`: MQuad, VarTrix, GoT tables without ambiguous calls, Gotcha). cellsnp-lite's depth is DP plus OTH and its reference count DP less AD.
- The join: a cell's ID is `barcode_prefix + barcode + barcode_suffix`, and `barcode_pattern`'s named group `barcode` keeps part of a barcode. mgatk with `-bt CB` on a Cell Ranger ARC ATAC BAM gives the RNA barcodes with `-1`. Multiplexed Flex barcodes (24 nt) join a singleplex library with `barcode_pattern=r"^(?P<barcode>[ACGT]{16})", barcode_suffix="-1"`. Several libraries take one prefix each.
- Several paths share sites, so they must lie on one reference: a contig of another length or sequence, one sequence under two contig names (mgatk's contig is the name of `<contig>_refAllele.txt` without the base files' prefix, so name copies alike), files that give an allele two roles, or a cell counted twice at a site raise `ValueError`. They must also share the unit, the depth definition and the count fields: import four- and six-column mgatk folders, or cellsnp-lite folders with and without OTH, apart. GoT-Multi's refined table holds calls alone and imports with the stage-1 table.
- The artifact: canonical `mean_depth` (depth summed over sites over the number of sites; NaN for a cell no file names), `covered_sites`, `listed`. `ds.select_cells(counts, low=10)` keeps cells with a mean depth above 10. mgatk scores cells by the same bound but averages over every reference position, the `N` at 3107 included, so cells within a few thousandths of 10 can differ.
- `ds.variants.load_counts(counts, *, sites=None, cells=None)` returns a `VariantCounts` record: `.sites` (indexed by label such as `chrM:310`: `contig`, `position`, `name`, `channel`, `reference`, `depth`, `cells`), `.alleles`, `.cells` (`listed`, `mean_depth`, `covered_sites`), `.depth` (`cell`, `site`, `depth`), `.counts` (`cell`, `site`, `allele`, `count`, plus `forward`, `reverse`, `quality_forward`, `quality_reverse`, `reads`, `support` where the source has them), `.calls` (`cell`, `site`, `allele`, `source`, `call`), `.contigs`, `.format`, `.count_unit`, `.depth_definition`. `sites` takes labels or positions (one contig), `cells` a cell selection or IDs, as a list, NumPy array or pandas index; `cells=[]` gives only the sites with their sums.
- `ds.plots.variant_depth(counts, group_by=None, groups=None)` draws each site's mean depth over named cells and the distribution of the cells' mean depth, one histogram per group. Tables: `sites` and `cells`.
- `ds.variants.informative(counts, *, cells=None, preset=None, min_cells=None, min_strand_correlation=None, min_vmr=None, min_depth=None, min_quality=None, mask=None, invalidate_cache=False)` scores every variant (alternative allele) that a chosen cell shows with mgatk's statistics and saves a `variant_selection` artifact. `preset="atac"`: `n_cells_conf_detected >= 5`, `strand_correlation >= 0.65`, `vmr > 0.01` (mtscATAC-seq). `preset="rna"`: `mean_coverage > 20`, `mean_quality > 30`, `vmr > 0.01`, no strand test (MAESTER). A threshold given by name replaces the preset's; a test the counts cannot make (strands of cellsnp-lite counts, quality of four-column mgatk) raises `ValueError`. `cells=ds.select_cells(counts, low=10)` scores the cells mgatk scores.
- `mask`: `"numt"` (25 NUMT false positives, Laricchia et al. 2022), `"rna_editing"` (295, 2617, 13710, Bar-Yaacov et al. 2013), `"artifact_prone"` (301, 302, 310, 316, 3107, 16182). Masked variants are scored but never informative; masks need a 16,569-base rCRS contig (hg38 `chrM`, GRCh38 `MT`) and refuse hg19.
- Allele frequencies are undefined where a cell was not covered, never 0: the variance covers covered cells, as in mgatk. Signac sets them to 0 and stabilises the variance of low-depth cells, so its VMRs differ where coverage is sparse.
- `ds.variants.load_informative(informative)` returns a `VariantSelection`: `.variants` indexed by label (`chrM:310T>C`) with `site`, `position`, `reference`, `alternative`, `mean`, `variance`, `vmr`, `n_cells_conf_detected`, `strand_correlation`, `n_cells_over_5/10/20/95`, `max_heteroplasmy`, `mean_coverage`, `n_cells_covered`, `mean_quality`, `masked`, `informative`; plus `.counts`, `.cells`, `.preset`, `.thresholds`, `.masks`, `.n_cells`.
- `ds.variants.clones(counts, *, variants, n_neighbors=10, resolution=1.0, within=None, invalidate_cache=False)` calls clones as Signac's `FindClonotypes`: cosine similarity of square-root allele frequencies over the variants two cells both cover, neighbour edges weighed by the Jaccard index of the two neighbourhoods as Seurat weighs them (k includes the cell, joins below 1/15 dropped), Leiden. Cells without coverage, without any alternative read, or similar to no cell get no clone (attributes `n_cells_without_coverage`, `n_cells_without_alternative`, `n_cells_without_neighbors`). `within="donor"` keeps groups apart and labels clones `donor1:3`. Saves a `clones` artifact that every clone summary reads.
- `ds.plots.variant_statistics(informative)` draws VMR against strand correlation (mean depth without strands); `ds.plots.allele_frequencies(informative, clones=clones)` draws cells by informative variants, ordered by clone, uncovered entries grey.

## Recipe

```python
cells = ds.snapshot_cell_selection("I")
counts = ds.variants.import_counts("mgatk/final", cell_selection=cells)
attributes = ds.artifacts.load(counts).attrs
print(attributes["n_unmatched_barcodes"], "of", attributes["n_file_barcodes"], "barcodes unmatched")
ds.plots.variant_depth(counts, group_by="sample")
deep = ds.select_cells(counts, low=10)

# One variant's frequency per covered cell: count over depth.
loaded = ds.variants.load_counts(counts, sites=[3244], cells=deep)
table = loaded.counts.merge(loaded.depth, on=["cell", "site"])
table["frequency"] = table["count"] / table["depth"]
```

```python
# Informative mitochondrial variants and the clones they mark (mtscATAC-seq).
deep = ds.select_cells(counts, low=10)
informative = ds.variants.informative(counts, cells=deep, preset="atac", mask=["numt", "artifact_prone"])
selection = ds.variants.load_informative(informative)
selection.variants[selection.variants["informative"]].sort_values("vmr", ascending=False).head()
ds.plots.variant_statistics(informative)
clones = ds.variants.clones(counts, variants=informative)
ds.plots.allele_frequencies(informative, clones=clones)
ds.clones.load_clones(clones).clones.head()
```

```python
# GoT-Multi's stage 1 counts and refined genotypes together.
evidence = ds.variants.import_counts(
    ["got_multi/ironthrone_out_pp.csv", "got_multi/ml_denoise/got_multi_out_refined.csv"],
    cell_selection=cells,
)
calls = ds.variants.load_counts(evidence).calls
calls[calls["source"] == "genotype_final"]["call"].value_counts()
```

## Checks

- Check `n_unmatched_barcodes` first: a wrong prefix, suffix or pattern leaves every cell unlisted.
- Check `.contigs`: mgatk picks hg19's 16,571-bp sequence when the BAM's contig has that length, and nf-MiTo's `refAllele.txt` names no contig, which then never combines with a named one.
- Look at the depth plot before calling variants: most cells should cover most positions in mtscATAC-seq, while MAESTER and targeted panels cover few sites deeply.
- Frequencies need the depth: never treat a missing depth row as frequency 0.
- Check `ds.plots.variant_statistics` before trusting a preset: informative variants should sit apart, top right. Too many informative variants usually means noise passed the thresholds; raise `min_vmr` or add `mask`.
- Check the clone attributes: many `n_cells_without_coverage` or `n_cells_without_alternative` cells mean shallow coverage or variants that mark few clones; such cells are not clones.
