# Regulons, chromatin activity and gene programs

Infer transcription-factor regulons from RNA the SCENIC way, score motif or binding-peak activity
in ATAC cells with chromVAR, build enhancer-driven regulons from paired RNA and ATAC the SCENIC+
way, and find co-regulated gene programs. Docs: <https://docs.nygen.io/CyteArc/tutorials/regulons>,
<https://docs.nygen.io/CyteArc/api/regulons.html>.

## When to use

- A clustered RNA store (`pipeline-runs-and-artifacts.md`) and a question about which factors
  drive which populations.
- An ATAC (or multiome, or scCUT&Tag) assay whose features are `chrom:start-end` peaks.
- Binding peaks of a factor (CUT&RUN, CUT&Tag, ChIP-seq) to score in cells or to define targets.

## Key concepts

- The scorers take a table with `source` and `target` columns: `ds.scores.aucell` scores gene
  targets on RNA; `ds.scores.chromvar` scores peak targets, matched to ATAC feature ids
  (`chrom:start-end`). Motif matches and cistromes have these columns; rename `regulon` of
  regulons, `program`/`gene` of programs, and `eRegulon`/`gene` of eRegulons. Both return
  `kind="enrichment_scores"`; `ds.scores.load(ref)` gives `.data` (cells x sources) and, for
  chromVAR, `.deviations`. chromVAR z-scores are NaN where a cell expects under one fragment.
- `ds.regulons.grn(cells, features=..., tfs=..., random_seed=...)` runs GRNBoost2 on **raw counts**
  of the selected cells and genes (held in memory as float32; subsample cells for large data) and
  returns `kind="grn_adjacencies"`; `ds.regulons.load_grn(ref).adjacencies` has `TF`, `target`,
  `importance`, `correlation`. The default `early_stopping="stage"` matches the pySCENIC 0.12.1
  container. Fits are deterministic for a seed; different seeds give different networks.
- SCENIC steps are plain functions in `cytearc.regulons`: `modules_from_adjacencies` (pySCENIC
  rules), `prune_modules` (cisTarget with each motif's own recovery curve), `regulons_from_motifs`
  (`TF(+)`). pySCENIC 0.12.1 gives most motifs another motif's curve, so expect the same regulon
  names as pySCENIC but different targets (median target Jaccard about 0.25 on one network).
  Ranking databases need `pip install "cytearc[regulons]"` (polars).
- Motifs: `read_jaspar_motifs`/`read_meme_motifs`, `match_motifs(chroms, starts, ends,
  IndexedFasta(genome.fa), motifs, names=peak_ids)` (motifmatchr settings: p 5e-5, log2,
  pseudocount 0.8, `background="subject"`; scores are proper log-odds, so matches overlap
  motifmatchr's at Jaccard 0.94, not exactly). Use the genome build of the peaks (10x GRCh38-2020-A:
  GENCODE v32 primary assembly). An uncompressed FASTA is indexed in memory or read from `.fai`.
- Cistromes: `cistromes_from_motifs(matches, annotations)` joins on motif ID, so the annotation's
  `motif` column must equal the scanned IDs. A cisTarget `.tbl` uses its own IDs
  (`jaspar__MA0080.6`), and a mismatch returns an empty table. For JASPAR scans build
  `pd.DataFrame([(m.motifId, m.name.upper(), "direct") for m in motifs], columns=["motif", "TF",
  "category"])`. `binding_cistromes({"SPI1": "peaks.narrowPeak"}, peak_ids)` for CUT&RUN/ChIP peaks.
- eRegulons: `search = link_regions_to_genes(..., upstream=150_000, downstream=150_000,
  anchor="gene_body")` gives region positions; name them with `pd.DataFrame({"region":
  peak_ids[search["region"]], "gene": search["gene_name"]})`, score those candidates with
  `region_gene_links` on metacell profiles, then `build_eregulons(cistromes, r2g, tf2g)`.
- Gene programs: `gene_programs(metacell_profiles, genes)`; `program_regulators` runs cisTarget on
  each program; `set_overlaps` relates programs to regulons.

## Recipes

### RNA regulons (SCENIC)

```python
import numpy as np
import pandas as pd
import cytearc
from cytearc import regulons as rg

ds = cytearc.DataStore("analysis.zarr", nthreads=8, min_features_per_cell=-1)
run = ds.pipeline.open(label="baseline")
cells = run["analysis_cell_selection"]
tfs = [line.strip() for line in open("allTFs_hg38.txt") if line.strip()]
grn = ds.regulons.grn(cells, features=run["feature_universe"], tfs=tfs, random_seed=777)
adjacencies = ds.regulons.load_grn(grn).adjacencies
modules = rg.modules_from_adjacencies(adjacencies)
databases = [rg.RankingDatabase("hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather")]
annotations = rg.read_motif_annotations("motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
motifs = rg.prune_modules(modules, databases, annotations, nthreads=8)
regulons = rg.regulons_from_motifs(motifs)
auc = ds.scores.aucell(regulons.rename(columns={"regulon": "source"}), cells,
                       features=run["feature_universe"], tmin=5)
scores = ds.scores.load(auc)
activity = pd.DataFrame(np.asarray(scores.data.compute()), columns=scores.source_names)
rss = rg.regulon_specificity(activity, run.cells.fetch("clusters"))
```

### Motif and binding activity in ATAC cells (chromVAR)

```python
# chromVAR needs a fragment in every selected peak.
peaks = ds.features.detected(cells, from_assay="ATAC", min_cells=1)
ids = ds.ATAC.feats.fetch_all("ids")[np.flatnonzero(ds.artifacts.load(peaks)["values"][:])]
chroms, starts, ends = rg.parse_coordinate_ids(ids)
with rg.IndexedFasta("GRCh38.primary_assembly.genome.fa") as genome:
    matches = rg.match_motifs(chroms, starts, ends, genome, rg.read_jaspar_motifs("JASPAR2024.txt"), names=ids)
z = ds.scores.chromvar(matches, cells, from_assay="ATAC", features=peaks,
                       genome="GRCh38.primary_assembly.genome.fa")
binding = rg.binding_cistromes({"SPI1": "SPI1_cutrun.narrowPeak"}, ids)
spi1 = ds.scores.chromvar(binding, cells, from_assay="ATAC", features=peaks,
                          genome="GRCh38.primary_assembly.genome.fa", tmin=5)
```

## Pitfalls

- A motif deviation is a motif family's accessibility, not one factor's activity; pair it with
  the factor's expression or binding data before naming the factor.
- Do not mix genome builds: peaks, FASTA, GTF, and cisTarget databases must agree.
- Regulon targets from one GRNBoost2 seed include seed-specific links; report seed-stable ones.
- The chromVAR background model holds about four bytes per motif-peak match per background
  iteration; score fewer motifs per call if the budget refuses it.
- Binding peaks from cell lines describe that cell line's context.
