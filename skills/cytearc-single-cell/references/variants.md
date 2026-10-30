# Mitochondrial and targeted variants

Import each cell's counts of each allele at variant sites, from mitochondrial lineage tracing (mtscATAC-seq, MAESTER) or targeted genotyping (GoT, GIFT), join them to the store's cells, check depth, and read the counts as tables. Docs: <https://docs.nygen.io/CyteArc/api/variants.html>.

## When to use

- Per-cell allele counts from mgatk (`<name>.{A,C,G,T}.txt.gz` with `<contig>_refAllele.txt`), maegatk (the same files, six columns), cellsnp-lite (`cellSNP.tag.{AD,DP,OTH}.mtx`), MQuad (`passed_ad.mtx`, `passed_dp.mtx`), VarTrix's coverage mode (alt and ref matrices, VCF, barcodes), MiTo's allele frequency matrix (`.h5ad` with `AD` and `DP` layers), giftwrap (`counts.N.h5`, or GEO's `probe_reads.tsv.gz` with `barcodes.tsv.gz` and `manifest.tsv`), and GoT tables (IronThrone's `summTable.txt`, GEO's per-cell tables, GoT-Multi's `ironthrone_out_pp.csv` and `got_multi_out_refined.csv`, Gotcha's CSVs).
- Not for BAM or FASTQ files: run the tool first. Not for clones that a tool already called (MiTo's `obs['MiTo_clone']`, Numbat): use `ds.clones.import_clones`.

## Key concepts

- `ds.variants.import_counts(paths, *, cell_selection, barcode_prefix=None, barcode_suffix=None, barcode_pattern=None, format=None, count_unit=None, swap_strands=False, invalidate_cache=False)` saves a datastore-scoped `variant_counts` artifact. Each path is a folder (mgatk, maegatk, cellsnp-lite, MQuad, VarTrix, giftwrap's GEO tables) or a file (giftwrap's HDF5, MiTo's h5ad, a GoT table), and several paths must share a format. `format` is one of `"mgatk"`, `"maegatk"`, `"cellsnp_lite"`, `"mquad"`, `"vartrix"`, `"mito"`, `"giftwrap"`, `"ironthrone"`, `"got"`, `"got_multi"`, `"gotcha"`.
- Six-column mgatk files (`--emit-base-qualities`) look like maegatk's, which count strands the other way round, so such a folder raises `ValueError` until `format` names it. maegatk's strands are swapped back on import; mgatk `call` or `bcall` output from 0.4.0 to 0.5.8 needs `swap_strands=True`. Strand only means something in DNA libraries (mtscATAC-seq), not RNA (MAESTER).
- cellsnp-lite, MQuad and VarTrix need `count_unit="umis"` or `"reads"`, since their files do not say; other formats count in their own unit (mgatk reads, maegatk and MiTo molecules, giftwrap and GoT UMIs, Gotcha reads), and a different `count_unit` raises `ValueError`.
- Sites: every reference position but `N` (mgatk, maegatk; 16,568 for rCRS), each VCF record (cellsnp-lite, VarTrix), each passed variant (MQuad), each position of the matrix (MiTo), each probe (giftwrap) or target (GoT). A one-target GoT table names its site after the file (`Patient1_TET2.2847_summTable.txt` gives `Patient1_TET2.2847`). GoT-Multi's channels are sites `BCOR (geneseq)` and `BCOR (probebc)`, and its genotypes sit on `BCOR`.
- Alleles have roles: `"reference"` (reference base, wild type), `"alternative"` (other base, mutant, a GIFT manifest's expected gapfill) and `"other"` (cellsnp-lite's other bases, GoT's ambiguous UMIs, gapfills no manifest names).
- Depth 0 means not covered: an allele frequency there is undefined, never 0. The `depth_definition` attribute says whether depth counts every read (`"all"`) or only the reference and alternative (`"reference_and_alternative"`: MQuad, VarTrix, GoT tables without ambiguous calls, Gotcha). cellsnp-lite's depth is DP plus OTH and its reference count DP less AD.
- The join: a cell's ID is `barcode_prefix + barcode + barcode_suffix`, and `barcode_pattern`'s named group `barcode` keeps part of a barcode. mgatk with `-bt CB` on a Cell Ranger ARC ATAC BAM gives the RNA barcodes with `-1`. Multiplexed Flex barcodes (24 nt) join a singleplex library with `barcode_pattern=r"^(?P<barcode>[ACGT]{16})", barcode_suffix="-1"`. Several libraries take one prefix each.
- Several paths share sites, so they must lie on one reference: a contig of another length or sequence, files that give an allele two roles, or a cell counted twice at a site raise `ValueError`.
- The artifact: canonical `mean_depth` (depth summed over sites over the number of sites; NaN for a cell no file names), `covered_sites`, `listed`. `ds.select_cells(counts, low=10)` keeps cells with mean depth of at least 10, as Signac and mgatk filter.
- `ds.variants.load_counts(counts, *, sites=None, cells=None)` returns a `VariantCounts` record: `.sites` (indexed by label such as `chrM:310`: `contig`, `position`, `name`, `channel`, `reference`, `depth`, `cells`), `.alleles`, `.cells` (`listed`, `mean_depth`, `covered_sites`), `.depth` (`cell`, `site`, `depth`), `.counts` (`cell`, `site`, `allele`, `count`, plus `forward`, `reverse`, `quality_forward`, `quality_reverse`, `reads`, `support` where the source has them), `.calls` (`cell`, `site`, `allele`, `source`, `call`), `.contigs`, `.format`, `.count_unit`, `.depth_definition`. `sites` takes labels or positions (one contig), `cells` a cell selection or IDs; `cells=[]` gives only the sites with their sums.
- `ds.plots.variant_depth(counts, group_by=None, groups=None)` draws each site's mean depth over named cells and the distribution of the cells' mean depth, one histogram per group. Tables: `sites` and `cells`.

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
