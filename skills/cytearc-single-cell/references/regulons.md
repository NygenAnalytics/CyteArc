# Regulons, chromatin activity and gene programs

Infer transcription-factor regulons from RNA the SCENIC way, score motif or binding-peak activity
in ATAC cells with chromVAR, link peaks to genes in paired RNA and ATAC, and find co-regulated
gene programs. Docs: <https://docs.nygen.io/CyteArc/tutorials/regulons>,
<https://docs.nygen.io/CyteArc/api/regulons.html>.

## When to use

- A clustered RNA store (`pipeline-runs-and-artifacts.md`) and a question about which factors
  drive which populations.
- An ATAC (or multiome, or scCUT&Tag) assay whose features are `chrom:start-end` peaks.
- Binding peaks of a factor (CUT&RUN, CUT&Tag, ChIP-seq) to score in cells or to define targets.

## Key concepts

- Every set is a table with `source` (the set) and `target` (a member): modules, regulons,
  programs, and cistromes. `ds.scores.aucell` scores gene sets on RNA and
  `ds.scores.chromvar` peak sets on ATAC; both return `kind="enrichment_scores"`.
  `ds.scores.load(ref).to_pandas()` gives cells x sources; chromVAR also has `.deviations`.
  chromVAR z-scores are NaN where a cell expects under one fragment.
- Regions are always peak ids (`chrom:start-end`), the ATAC feature `ids`: `match_motifs`,
  `link_regions_to_genes`, `gc_content`, `binding_cistromes`, and chromVAR targets.
- `ds.regulons.grn(cells, features=..., tfs=..., random_seed=...)` runs GRNBoost2 on **raw counts**
  of the selected cells and genes (held in memory as float32; subsample cells for large data) and
  returns `kind="grn_adjacencies"`; `ds.regulons.load_grn(ref).adjacencies` has `TF`, `target`,
  `importance`, `correlation`, as does the in-memory `infer_grn`. The default
  `early_stopping="stage"` matches the pySCENIC 0.12.1 container. Fits are deterministic for a
  seed; different seeds give different networks.
- SCENIC steps are plain functions in `cytearc.regulons`: `modules_from_adjacencies` (pySCENIC
  rules), `prune_modules` (cisTarget with each motif's own recovery curve), `regulons_from_motifs`
  (`TF(+)`). pySCENIC 0.12.1 gives most motifs another motif's curve, so expect the same regulon
  names as pySCENIC but different targets (median target Jaccard about 0.25 on one network).
  Ranking databases need `pip install "cytearc[regulons]"` (polars).
- Motifs: `read_jaspar_motifs`/`read_meme_motifs`, `match_motifs(peak_ids, IndexedFasta(genome.fa),
  motifs)` (motifmatchr settings: p 5e-5, log2, pseudocount 0.8, `background="subject"`; scores are
  proper log-odds and each strand has its own threshold, as in MOODS, so matches differ from
  motifmatchr's, which reuses the forward threshold). Use the genome
  build of the peaks (10x GRCh38-2020-A: GENCODE v32 primary assembly). An uncompressed FASTA is
  indexed in memory or read from `.fai`.
- Cistromes: `cistromes_from_motifs(matches, annotations)` joins on motif ID.
  `motif_annotations(motifs)` annotates JASPAR or MEME motifs by name (upper-cased, dimers split);
  a cisTarget `.tbl` uses its own IDs (`jaspar__MA0080.6`) and gives an empty table for JASPAR
  scans. `binding_cistromes({"SPI1": "peaks.narrowPeak"}, peak_ids)` for CUT&RUN/ChIP peaks.
- Metacells: `ds.regulons.profiles(groups, from_assay=...)` gives groups x features log counts per
  million, genes named by name and peaks by id. Group with a fine Leiden clustering
  (`resolution=15` gave 117 metacells for about 2,600 PBMCs).
- Gene programs: `gene_programs(rna[genes])`; `program_regulators(programs, database, annotations,
  profiles=rna)` runs cisTarget on each program and correlates each factor's expression with it;
  `set_overlaps(programs, regulons, genes)` relates programs to regulons. `gene_programs` splits
  its gene graph with Leiden's default backend (`"cytearc"`), so programs from earlier releases,
  whose default was `"igraph"`, can differ from new ones.
- Region-to-gene links: `link_regions_to_genes(peak_ids, read_gene_annotation(gtf), upstream=...,
  downstream=..., anchor="tss" or "gene_body")` gives candidate pairs, which
  `region_gene_links(atac, rna, candidates)` scores on metacell profiles: Spearman `rho` and a
  gradient-boosting `importance`, one regressor per gene (`importance="none"` skips it). Links
  are associations across metacells; report them as candidate enhancers.

## Recipes

### RNA regulons (SCENIC)

```python
import numpy as np
import cytearc
from cytearc import regulons as rg

ds = cytearc.DataStore("analysis.zarr", nthreads=8, min_features_per_cell=-1)
run = ds.pipeline.open(label="baseline")
cells = run["analysis_cell_selection"]
tfs = [line.strip() for line in open("allTFs_hg38.txt") if line.strip()]
grn = ds.regulons.grn(cells, features=run["feature_universe"], tfs=tfs, random_seed=777)
adjacencies = ds.regulons.load_grn(grn).adjacencies
modules = rg.modules_from_adjacencies(adjacencies)
database = rg.RankingDatabase("hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")
annotations = rg.read_motif_annotations("motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
regulons = rg.regulons_from_motifs(rg.prune_modules(modules, [database], annotations, nthreads=8))
auc = ds.scores.aucell(regulons, cells, features=run["feature_universe"], tmin=5)
activity = ds.scores.load(auc).to_pandas()
rss = rg.regulon_specificity(activity, run.cells.fetch("clusters"))
```

### Motif and binding activity in ATAC cells (chromVAR)

```python
# chromVAR needs a fragment in every selected peak.
peaks = ds.features.detected(cells, from_assay="ATAC", min_cells=1)
peak_ids = np.asarray(ds.ATAC.feats.fetch_all("ids"))[np.flatnonzero(ds.artifacts.load(peaks)["values"][:])]
motifs = rg.read_jaspar_motifs("JASPAR2024.txt")
with rg.IndexedFasta("GRCh38.primary_assembly.genome.fa") as genome:
    matches = rg.match_motifs(peak_ids, genome, motifs)
z = ds.scores.chromvar(matches, cells, from_assay="ATAC", features=peaks,
                       genome="GRCh38.primary_assembly.genome.fa")
motif_z = ds.scores.load(z).to_pandas()
binding = rg.binding_cistromes({"SPI1": "SPI1_cutrun.narrowPeak"}, peak_ids)
spi1 = ds.scores.chromvar(binding, cells, from_assay="ATAC", features=peaks,
                          genome="GRCh38.primary_assembly.genome.fa", tmin=5)
```

### Gene programs and region-to-gene links (multiome)

```python
metacells = ds.clusters.leiden(run["connectivity_map"], resolution=15)
rna = ds.regulons.profiles(metacells, from_assay="RNA")
atac = ds.regulons.profiles(metacells, from_assay="ATAC")
names = np.asarray(ds.RNA.feats.fetch_all("names")).astype(str)
variable = np.asarray(run.features.fetch_all("highly_variable_features"), dtype=bool)
genes = [gene for gene in names[variable] if gene in rna.columns]
programs = rg.gene_programs(rna[genes])
program_tfs = rg.program_regulators(programs, database, annotations, profiles=rna)

annotation = rg.read_gene_annotation("gencode.v32.primary_assembly.annotation.gtf.gz",
                                     gene_types={"protein_coding"})
candidates = rg.link_regions_to_genes(peak_ids, annotation, upstream=150_000,
                                      downstream=150_000, anchor="gene_body")
links = rg.region_gene_links(atac, rna, candidates[candidates["gene"].isin(genes)], nthreads=8)
```

## Pitfalls

- A motif deviation is a motif family's accessibility, not one factor's activity; pair it with
  the factor's expression or binding data before naming the factor.
- Do not mix genome builds: peaks, FASTA, GTF, and cisTarget databases must agree.
- The Aerts lab's cisTarget databases and motif tables state no license terms, and the v10
  motif collection includes motifs derived from TRANSFAC Professional, a commercial database.
  Confirm their terms before commercial work.
- Regulon targets from one GRNBoost2 seed include seed-specific links; report seed-stable ones.
- The chromVAR background model holds about four bytes per motif-peak match per background
  iteration; score fewer motifs per call if the budget refuses it.
- Binding peaks from cell lines describe that cell line's context.
- `motif_annotations` upper-cases factor names like human gene symbols; recase its `TF` column
  for other species.
- `gene_programs` assumes independent samples (metacells, pseudobulk); smoothed cells share
  counts and produce spurious programs. Its threshold screens chance groupings, not a calibrated
  test of each program.
