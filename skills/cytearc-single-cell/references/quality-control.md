# Quality control

Check what the count matrix holds, inspect per-cell QC metrics, build immutable filtered cell
selections, audit removals without author labels, and score doublets. Docs:
<https://docs.nygen.io/CyteArc/tutorials/quality-control>, <https://docs.nygen.io/CyteArc/api/datastore.html>,
<https://docs.nygen.io/CyteArc/api/pipeline.html>.

## When to use

- Before the first graph on a new store, or whenever the analysed cell population changes.
- When the store comes from a public atlas, CELLxGENE or a Seurat submission: counts may be
  corrected, genes removed, or cells already filtered.
- When a run kept fewer cells than expected, and after a first clustering (doublets, QC clusters).

## Key concepts

- QC columns are written once, at the first writable open: `RNA_nCounts`, `RNA_nFeatures`,
  `RNA_percentMito` (`^MT-`, any case) and `RNA_percentRibo` (`^RPS|^RPL|^MRPS|^MRPL`) on feature
  `names`. For any other gene set use `qc.feature_percentage` (an artifact, not a column).
- Filters never edit `I` (the live cell key): each returns a new immutable `ArtifactRef`
  (`kind="cell_selection"`). Omit `cell_selection=` to snapshot `I`; pass a selection to compose.
- `qc.auto_filter` defaults to pooled MAD (`n_mads=3`, scaled MAD = 1.4826 * MAD). Each metric's
  policy follows what CyteArc wrote: `RNA_nCounts`/`RNA_nFeatures` are filtered on log1p values
  with two-sided bounds; the prepared `percentMito`/`percentRibo` columns and every
  `qc.feature_percentage` artifact on raw values with an upper bound only. Other columns, such as
  imported scanpy `total_counts`, `n_genes_by_counts` or `pct_counts_mt`, fall back to the
  name suffix (raw two-sided bounds for those three) with a warning: pass
  `metric_policies={"total_counts": {"transform": "log1p", "bound_direction": "two_sided"}}`
  (`bound_direction` also takes `"lower"` and `"upper"`). A metric with zero MAD is skipped with a
  warning. The resolved policies are `parameters["metric_policies"]`.
- Bounds are exclusive: `qc.filter(lows=[200])` removes a cell with exactly 200 genes, which
  scanpy's `filter_cells(min_genes=200)` keeps; pass `keep_bounds=True` for inclusive bounds. MAD
  bounds are always exclusive, and `min_features_per_cell=k` removes cells with at most `k`.
- Raw droplet matrices break MAD bounds, which then center on empty droplets. A `UserWarning`
  says so when the open skips its `min_features_per_cell` floor (at least half of the cells would
  go), when automatic bounds remove most cells, or when the MAD bounds of counts miss a second
  population. Call cells or import the filtered matrix before QC.
- `parameters["resolved_bounds"]` are reported back on the raw scale: `low` and `high` are counts,
  genes or percent; never apply `expm1`. Example (10x 5K PBMC docs dataset): `RNA_nCounts` low
  3377.1 and high 18389.9 mean 3,377 to 18,390 counts.
- Every bound assumes raw UMI counts from unfiltered cell calls. Corrected (for example
  SCTransform) or pre-filtered matrices change what each bound means: check the matrix first.
- Pooled MAD erases real populations: platelets, erythrocytes and neutrophils fall under lower
  bounds (floors near 200 to 650 genes); plasma, cycling and progenitor cells cross upper bounds;
  myeloid cells carry more mito, so a pooled mito bound trims them first. Example (10x 5K PBMC
  docs dataset): even on cells filtered once before, pooled 3-MAD kept 6 of 15 platelet-like cells.

## Recipes

### Check the matrix first

`scripts/inspect_store.py STORE.zarr` (relative to the skill root) runs these checks with hints.
By hand, read-only; the count read is the first 300 cells only (one small read on a mount):

```python
import numpy as np
import pandas as pd
import cytearc

ds = cytearc.DataStore("analysis.zarr", zarr_mode="r")
sample = ds.RNA.rawData[:300].compute()
nonzero = sample[sample != 0]
print("integer-like", bool(np.all(nonzero == np.round(nonzero))), "negative", bool((sample < 0).any()))
author = [c for c in ds.cells.columns if c.startswith(("nCount_", "nFeature_"))]  # adapt names
qc = ds.cells.to_pandas_dataframe(["RNA_nCounts", "RNA_nFeatures"] + author, key="I")
print(qc.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(1))
print("share within 5% of min:", {c: round(float((qc[c] <= qc[c].min() * 1.05).mean()), 4) for c in qc})
spread = {c: np.subtract(*np.percentile(np.log1p(qc[c]), [95, 5])) for c in ("RNA_nCounts", "RNA_nFeatures")}
print("spread ratio nFeatures/nCounts:", round(spread["RNA_nFeatures"] / spread["RNA_nCounts"], 2))
if author:
    print(qc.corr(method="spearman").loc[["RNA_nCounts", "RNA_nFeatures"], author].round(3))
```

| Signal | Reading |
| --- | --- |
| Nonzero values not integers, or negatives | Normalized, scaled or log values, not counts. Re-import a raw layer (`data-access.md`). |
| Spread ratio above 1: genes vary more than totals | Depth-equalized counts, for example SCTransform. Raw UMIs are usually below 1. |
| Store `nFeatures` matches author `nFeature_SCT` (Spearman about 1) | SCT-corrected counts. Integer values do not prove raw UMIs. |
| Store totals below author raw `nCount_RNA`, Spearman well below 1 | Corrected counts, removed genes (for example rRNA) or another layer. |
| 0.5% or more of cells within 5% of the minimum; a minimum or maximum just inside a round number (1,000 counts, 200 genes, 20% mito) | A hard cutoff before import: cells were already filtered. |
| Fewer active than total cells | An earlier filter edited `I`; active ranges are truncated. |
| Most barcodes hold fewer than 100 counts, tens of thousands of barcodes | An unfiltered droplet matrix: call cells before QC (below). |

Example (10x 5K PBMC docs dataset): integer `uint32` counts and a spread ratio of 0.78 (raw UMIs),
but `I` keeps 3,948 of 5,025 cells, and active cells sit just inside 1,000 to 15,000 counts, 500 to
4,000 genes and 15% mito: an earlier filter edited `I`.

With corrected or pre-filtered counts, QC bounds lose their meaning: depth bounds on SCT totals
measure the correction, and refiltering filtered cells mostly trims real populations. Skip
`nCounts` bounds; use flag-only QC (a status column, nothing removed) or a gentle, audited filter
(per-sample 5-MAD on `RNA_nFeatures` and `RNA_percentMito`). The report states what the matrix
holds, any visible prior bounds, and why QC was gentle. Count models need raw UMIs.

### Unfiltered droplet matrices: call cells first

A raw droplet matrix (Cell Ranger `raw_feature_bc_matrix`) is mostly empty droplets: MAD bounds
centre on them and remove the cells. Call cells with EmptyDrops first, then run QC and the pipeline
on the called cells. The `rna` recipe's `cell_calling` stage does both in one run; call by hand to
inspect the calls first.

```python
reader = cytearc.CrH5Reader("raw_feature_bc_matrix.h5", is_filtered=False, filtering_cutoff=0)
cytearc.CrToZarr(reader, zarr_loc="raw.zarr").dump()  # keeps the low-count ambient barcodes
ds = cytearc.DataStore("raw.zarr", min_features_per_cell=-1)  # keep every barcode in I
ranks = ds.qc.barcode_ranks()  # knee and inflection totals from stored totals, no count read
run = ds.pipeline.rna(label="called", cell_calling=True)  # EmptyDrops, then MAD on called cells
summary = {s["stage"]: s["summary"] for s in run.report()["stages"]}
calling = summary["cell_calling"]  # keyed by output
print(calling["cell_calling"])  # retain, knee, inflection, n_ambient
print(calling["called_cell_selection"], summary["filtering"])  # cells, inputCells, removedCells

cells = ds.qc.call_cells()  # by hand: EmptyDrops, FDR 0.001 -> ArtifactRef(kind="cell_selection")
calls = ds.qc.load_cell_calls(cells)  # per barcode: total, log_probability, p_value, fdr, limited
called = np.asarray(ds.artifacts.load(cells)["values"][:], dtype=bool)
print(int(called.sum()), "cells; retain", calls.retain, "alpha", calls.alpha)
auto = ds.qc.auto_filter(cell_selection=cells)  # QC on the called cells only
by_hand = ds.pipeline.rna(label="called_by_hand", cells=cells, filtering=True)  # QC the given cells
```

- `filtering_cutoff` defaults to 500 for unfiltered reads, which removes the ambient pool
  (barcodes with at most `lower=100` counts); `call_cells` then raises. Import with 0.
- Barcodes at or above `retain` (the knee total) are cells without a test. A warning names
  `limited` barcodes that were not called: rerun with a larger `n_iter`. In a run, such warnings
  are in the stage record and `run.report()["summary"]["warnings"]`.
- A merged store holds one ambient pool per capture: call each capture with
  `cell_selection=` and combine the selections, or run the recipe with
  `cell_calling={"sample_column": "capture"}`, which calls each capture apart and analyzes the
  union. Every input cell needs a capture label; a missing value, or NaN in a float column, is
  refused before the run starts. A capture that EmptyDrops cannot call fails the stage with
  `Cell calling failed for capture 'a' of 'capture': <cause>`. A per-capture run has no
  `cell_calling` output; its stage summary is the `called_cell_selection` one plus
  `calledPerCapture` and `kneePerCapture`. The capture labels are frozen apart
  (`capture_snapshot`), so changing `filtering`, `snapshot_columns` or Harmony reuses every
  capture's calls.
- The recipe's outputs are `cell_calling` (the per-barcode statistics) and `called_cell_selection`
  before `analysis_cell_selection`; `cell_calling={"fdr": 0.01, "n_iter": 20000}` takes the
  keywords of `qc.call_cells`, whose values are checked before the run starts. With `cell_calling`
  the run filters the called cells by default, also when `cells` is a selection or a run.

### Distributions, default filter and candidate policies

Distributions are metadata-only; group them by sample or capture, never by held-out labels. Then
build several policies, including gentle ones that keep high-RNA and low-complexity populations.

```python
from cytearc.plotting import CellField

ds = cytearc.DataStore("analysis.zarr", min_features_per_cell=-1)  # writable reopen
qc_cols = [c for c in ("RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo")
           if c in ds.cells.columns]
cells = ds.snapshot_cell_selection("I")  # ArtifactRef(kind="cell_selection")
ds.plots.distribution(keys=qc_cols, cell_selection=cells, kind="violin", max_points=2000)
ds.plots.distribution(keys=["RNA_nCounts", "RNA_nFeatures"], grouping=CellField("sample_id"),
                      cell_selection=cells, max_points=0)  # "sample_id" is your column name

def mask(ref):
    return np.asarray(ds.artifacts.load(ref)["values"][:], dtype=bool)

auto = ds.qc.auto_filter(cell_selection=cells)  # pooled MAD over the default columns
params = ds.artifacts.inspect(auto).parameters
print(pd.DataFrame(params["resolved_bounds"]["all"]).T[["low", "high", "skip_reason"]])  # raw scale
mito = ds.qc.auto_filter(["RNA_percentMito"], cell_selection=cells)  # upper MAD bound only
floors = ds.qc.filter(["RNA_nFeatures", "RNA_nCounts"], lows=[200, 500], highs=[None, None],
                         cell_selection=mito)  # composes with the mito selection
loose = ds.qc.auto_filter(cell_selection=cells, n_mads=5)
manual = ds.qc.filter(["RNA_nFeatures", "RNA_percentMito"], lows=[200, None], highs=[None, 20],
                         cell_selection=cells)
candidates = {"mad3": auto, "mad5": loose, "mito+floors": floors, "manual": manual}
print({name: int(mask(ref).sum()) for name, ref in candidates.items()})
```

### Audit what was removed, without labels

Do not read author labels. Run an unfiltered scaffold clustering, name its clusters by markers,
and report each candidate's retention per scaffold cluster and per marker-positive group. Costs
one pipeline run plus one small count pass (the panel export).

```python
scaffold = ds.pipeline.rna(label="qc_scaffold", filtering=False, cell_cycle=False)
in_run = mask(scaffold["analysis_cell_selection"])
clusters = np.asarray(scaffold.cells.fetch("clusters"))
kept = pd.DataFrame({name: mask(ref)[in_run] for name, ref in candidates.items()})
top = ds.markers.load(marker=scaffold["markers"]).groupby("group_id", sort=False).head(5)
markers = top.groupby("group_id")["feature_name"].agg(", ".join)
by_cluster = kept.groupby(clusters).mean().round(2).assign(n=kept.groupby(clusters).size())
by_cluster["markers"] = markers.reindex(by_cluster.index.astype(str)).to_numpy()
print(by_cluster.to_string())

panel = ["PPBP", "PF4", "JCHAIN", "MZB1", "HBB", "CSF3R", "FCGR3B", "MKI67", "TOP2A", "CD34",
         "CD3E", "MS4A1", "LYZ", "CD14", "MALAT1"]  # human blood; adapt genes and rules to tissue
names = set(ds.RNA.feats.fetch_all("names"))  # mouse symbols differ (Ppbp, Hbb-bs); map Ensembl IDs
print("panel genes not in this store:", [g for g in panel if g not in names])
panel = [g for g in panel if g in names]  # an absent name raises KeyError in to_anndata
ad = ds.to_anndata(feature_names=panel)  # raw counts of active cells, rows named by cell ids
counts = pd.DataFrame(ad.X.toarray(), index=ad.obs_names, columns=panel)
counts = counts.reindex(scaffold.cells.fetch("ids"))
rules = {"platelet PPBP+PF4+": {"PPBP": 1, "PF4": 1},  # minimum raw count of every gene
         "plasma MZB1>=3 JCHAIN>=3": {"MZB1": 3, "JCHAIN": 3},
         "erythroid HBB>=10": {"HBB": 10},
         "neutrophil CSF3R+FCGR3B+": {"CSF3R": 1, "FCGR3B": 1},
         "cycling MKI67+TOP2A+": {"MKI67": 1, "TOP2A": 1},
         "progenitor CD34+": {"CD34": 1}}
groups = {g: (counts[list(r)] >= pd.Series(r)).all(axis=1)
          for g, r in rules.items() if set(r) <= set(panel)}  # a group missing a gene is skipped
by_group = pd.DataFrame({g: kept[m.to_numpy()].mean() for g, m in groups.items()}).T
print(by_group.assign(n=[int(m.sum()) for m in groups.values()]).round(2))

debris = pd.DataFrame({"nFeatures": ds.cells.fetch_all("RNA_nFeatures")[in_run],
                       "percentMito": ds.cells.fetch_all("RNA_percentMito")[in_run],
                       "marker_positive": pd.DataFrame(groups, index=counts.index)
                       .any(axis=1).to_numpy()})
if "MALAT1" in panel:
    debris["MALAT1"] = counts["MALAT1"].to_numpy() > 0
for name in candidates:  # rows: False = removed, True = kept
    print(name, debris.groupby(kept[name].to_numpy()).agg(["median", "mean"]).round(2), sep="\n")
```

- A scaffold cluster or marker group losing more than about a third of its cells is biology being
  removed: pick a gentler policy or exempt that population explicitly.
- Debris shows low `nFeatures`, no `MALAT1` (a nuclear lncRNA), high mito and no lineage marker.
  Platelets and erythrocytes also lack `MALAT1` (no nucleus) but carry `PPBP`/`PF4` or `HBB`.
- Marker gates are starting points on raw counts. Ambient RNA (`JCHAIN`, `HBB`, `PPBP`) is
  detected at low levels in many cells, hence two markers or a higher count per gate.

Optional post-hoc audit, only after your own labels are frozen or when labels are not held out:

```python
labels = ds.cells.fetch_all("author_cell_type")[in_run]  # post hoc only; your column name
print(kept.groupby(labels).mean().round(2))
```

### Per-sample (capture-aware) MAD

Bounds are estimated within each sample of a real sample or capture column. Groups below
`min_cells_per_sample` are kept unfiltered with a warning. Missing labels among active cells raise.
For hashtag data pass `sample_artifact=NamedCellArtifact("hto", hto_ref)` (`kind="hto_identity"`),
or let the recipe's `hashing` stage do it (below).

```python
per_sample = ds.qc.auto_filter(cell_selection=cells, sample_column="sample_id", n_mads=3.0,
                                  min_cells_per_sample=20)  # "sample_id" is your column name
bounds = ds.artifacts.inspect(per_sample).parameters["resolved_bounds"]  # {sample: {metric: {...}}}
for side in ("low", "high"):  # raw scale; a low near zero means no floor in that sample
    print(side, pd.DataFrame({s: {m: b[side] for m, b in d.items()} for s, d in bounds.items()}).T)
```

### Demultiplex hashtags in the recipe

With an assay of type `HTO`, `hashing=True` calls each cell's hashtag (`qc.hto_demultiplexing`),
keeps the cells of exactly one hashtag (`hashed_cell_selection`; `Negative` and `Doublet` cells
leave), and computes MAD bounds within each hashtag unless `filtering` names a `sample_column`
(the filtering config records `"sampleSource": "hashing"`).

```python
hashed = ds.pipeline.rna(label="hashed", hashing=True)  # or {"positive_quantile": 0.999}
summary = {s["stage"]: s["summary"] for s in hashed.report()["stages"]}
print(summary["hashing"])  # {"hashing": {"hashtags", "negative", "multiplets"}, "hashed_cell_selection": ...}
margin = ds.artifacts.load_values(hashed["hashing"], value="margin")  # CLR diagnostics per cell
```

Settings are those of `qc.hto_demultiplexing` plus `assay`, needed when the store has several HTO
assays; `positive_quantile` is checked before the run starts. The run filters the singlets by
default, also when `cells` is a selection or a run. Hashed samples share a capture, so doublets
stay scored together unless `doublets` names a capture column. Known limitation: one
demultiplexing fit covers every capture, even with per-capture cell calling, although each
capture's HTO background differs, and each hashtag's MAD bounds pool its captures; on several
captures, check the calls and `margin` per capture.

### Pipeline filtering and feeding a custom selection

With `cells` naming a column (the default `"I"`), and in a run with `cell_calling` or `hashing`
whatever its `cells`, `filtering=None` is pooled MAD on the assay's QC columns (the four `RNA_*`
columns here), stored as its own artifact
(`run["analysis_cell_selection"]`; compare masks, not refs). Other accepted values: `False`;
`True`; `{"method": "mad", "n_mads": 5, "sample_column": "sample_id", "min_cells_per_sample": 20}`;
`{"method": "gaussian", "min_p": 0.01, "max_p": 0.99}`; and
`{"method": "manual", "attrs": [...], "lows": [...], "highs": [...], "keep_bounds": False}`.
Unknown keys raise early. A granular selection goes in as `cells=`, and the run keeps its cells
as they are unless `filtering` asks for more:

```python
run_manual = ds.pipeline.rna(label="qc_manual", filtering={
    "method": "manual", "attrs": ["RNA_nFeatures", "RNA_percentMito"], "lows": [200, None],
    "highs": [None, 20]}, snapshot_columns=("RNA_nCounts", "RNA_percentMito"))
run_custom = ds.pipeline.rna(label="qc_custom", cells=floors)  # not filtered again
assert run_custom["analysis_cell_selection"] == floors  # the given ref itself
```

### Custom feature-percentage metric

```python
from cytearc.metadata.selection import NamedCellArtifact

names = ds.RNA.feats.fetch_all("names").astype(str)
hb = ds.features.snapshot(from_assay="RNA", mask=np.isin(names, ["HBA1", "HBA2", "HBB"]))
pct_hb = ds.qc.feature_percentage(cells, hb)  # kind="quality_metric"; reads only the 3 genes; no column
no_rbc = ds.select_cells(pct_hb, high=5.0, cell_selection=floors)  # fixed bound, composed
mad_hb = ds.qc.auto_filter(["RNA_percentMito"], cell_selection=cells,  # percentHb: upper bound only
                              artifact_metrics=[NamedCellArtifact("percentHb", pct_hb)])
# A mostly-zero metric (percentHb in PBMC) has zero MAD and is skipped: use a fixed bound.
```

### Ambient RNA (soup) removal

SoupX-style correction (Young and Behjati 2020), tutorial
<https://docs.nygen.io/CyteArc/tutorials/ambient-rna>. It needs the channel's unfiltered droplets:
import the raw Cell Ranger matrix with `cytearc.CrH5Reader(path, is_filtered=False,
filtering_cutoff=1)` and open with `min_features_per_cell=-1`. Correction is an explicit step
between two runs: the first clusters the called cells, the estimate uses those clusters, and the
second analyzes the corrected assay over the same cells.

```python
first = ds.pipeline.rna(label="called_cells", cell_calling=True, filtering=False,
                        cell_cycle=False, doublets=False)  # or cells=<your called-cell column>
profile = ds.qc.ambient_profile(total_counts=(2, 10))  # or ds.qc.ambient_profile(empty_selection)
soup = ds.qc.load_ambient_profile(profile)  # name, counts, fraction per feature
gene_sets = {"haemoglobin": ["HBA1", "HBA2", "HBB"],  # keep genes the store has
             "immunoglobulin": ["IGKC", "IGHM", "IGLC2", "JCHAIN"]}
rho = ds.qc.ambient_contamination(first["analysis_cell_selection"], profile,
                                  gene_sets=gene_sets, clusters=first["clusters"])
est = ds.qc.load_ambient_contamination(rho)  # .estimate, .set_estimates, .evidence
ds.qc.correct_ambient(rho, assay_label="RNA_soupx", random_seed=0)  # new RNA assay of integer counts
corrected = ds.pipeline.rna(label="soupx", assay="RNA_soupx", cells=first)  # the same cells
```

- `total_counts` must sit below the cells on the barcode-rank curve; the paper uses 2 to 10.
- Gene sets must be absent from some clusters and abundant in the soup. A cluster holding one
  expressing cell is left out of that set; `level="cluster"` raises for a cluster that expresses
  every set. Without `clusters=`, weakly expressing cells inflate the estimate.
- A known rate goes in `fraction=` instead of `gene_sets=`.
- An estimate of exactly 0 (global, gene set, or cluster) warns: its cells hold no counts of their
  gene sets although pure soup would give some, so the fraction is only bounded, below about
  3 / expected. Correcting with it leaves those cells uncorrected; check the gene sets and profile.
- Each gene's removal is rounded at random as in the paper (systematic sampling, one draw per
  cell from `random_seed`): every gene, even a rare soup gene, loses its share of the soup on
  average, and each cell loses its share rounded down or up. A seed reproduces the counts; the
  assay records it as `ambient_random_seed`.
- The corrected assay measures only the estimated cells (`RNA_soupx_I`): run it with
  `assay="RNA_soupx"` and `cells=` the first run or its selection, and pass `from_assay=` with a
  selection of those cells elsewhere. Other cells raise `UnmeasuredCellsError`.
- One profile and estimate per channel. Not implemented: SoupX's automatic estimate,
  cluster-aggregated correction, DecontX. CellBender output imports with `CrH5Reader`.

### Doublet scores

The `rna` recipe scores doublets by default (it needs a graph and a clustering), as Scrublet does:
`doublet_score` is the probability that a cell is a doublet and `predicted_doublet` the call
above its sample's automatic threshold. Doublets form within one capture, so with several
captures pass the capture column: `doublets={"sample_column": "sample"}`; a Harmony run scores
within its batch columns by default. Do not pass HTO identities: hashed samples share a capture.
High-RNA types such as plasma cells score high as a whole, so check calls and scores within their
cluster and require co-expressed markers of two lineages before removing cells.

```python
scores = pd.Series(np.asarray(scaffold.cells.fetch("doublet_score")))
called = np.asarray(scaffold.cells.fetch("predicted_doublet"))
print(pd.Series(called).groupby(clusters).mean().round(3))  # call rate per cluster
print(scores.groupby(clusters).median().round(2))  # compare clusters within a lineage
spread = scores.groupby(clusters).transform(lambda s: 1.4826 * (s - s.median()).abs().median())
relative = (scores - scores.groupby(clusters).transform("median")) / (spread + 1e-6)
two_lineage = (((counts.CD3E > 0) & (counts.LYZ >= 5)) | ((counts.MS4A1 > 0) & (counts.LYZ >= 5))
               | ((counts.MZB1 >= 3) & (counts.CD14 > 0))).to_numpy()
suspect = (called | (relative.to_numpy() > 3)) & two_lineage
singlet = in_run.copy()
singlet[np.flatnonzero(in_run)[suspect]] = False
ds.cells.insert("qc_singlet", singlet, overwrite=True)
# Then run cells=ds.snapshot_cell_selection("qc_singlet"); cells="qc_singlet" MAD-filters again.
# Explicit, per capture; thresholds and estimated doublet rates per sample.
doublets = ds.qc.doublets(scaffold["clusters"], scaffold["connectivity_map"],
                          sample_column="sample")
print(ds.artifacts.load(doublets).attrs["samples"])
called_singlets = ds.select_cells(doublets, value="predicted_doublets", include=[False])
```

Remove doublets with a second run over the first run's singlets. The given selection is not
filtered again; HVGs, PCA, the graph, clusters and markers are recomputed without the doublets:

```python
run = ds.pipeline.open(label="baseline")  # a completed rna run with doublet scores
singlets = ds.select_cells(run["doublets"], value="predicted_doublets", include=[False])
clean = ds.pipeline.rna(label="baseline_singlets", cells=singlets)  # or the qc_singlet selection
```

## Parameters that matter

| Parameter | Default | Change when |
| --- | --- | --- |
| `qc.auto_filter(n_mads=)` | `3.0` | Raise (for example 5) when real populations sit in the tails. |
| `qc.auto_filter(attrs=)` | the four `RNA_*` QC columns | Drop count bounds on corrected counts or when they remove high-RNA types. |
| `qc.auto_filter(metric_policies=)` | from what CyteArc wrote, else the name suffix | Imported metrics (`total_counts`, `pct_counts_mt`), or lower-only count bounds (`"bound_direction": "lower"`). |
| `qc.filter(keep_bounds=)` / `select_cells(keep_bounds=)` | `False` (exclusive) | `True` to keep values equal to a bound, as scanpy's `min_genes` does. |
| `sample_column=` / `sample_artifact=` | `None` (pooled) | Several samples, captures or depths. MAD only. |
| `min_cells_per_sample=` | `20` | Small samples; groups below it are kept unfiltered. |
| `method=` (`min_p`, `max_p`) | `"mad"` | `"gaussian"` only to reproduce the old policy; `min_p`/`max_p` with MAD raise. |
| `pipeline.rna(filtering=False, cell_cycle=False)` | `filtering=None`: MAD on a column's cells, or on the cells that `cell_calling` or `hashing` keep; none for a given selection or run otherwise | Scaffold run for the label-free audit. |
| `pipeline.rna(cell_calling=, hashing=)` | `False`, `False` | Raw droplets (EmptyDrops in the run) or an HTO assay (keep one-hashtag cells, MAD per hashtag; one fit over all captures). |
| `qc.doublets(random_seed=)` | `4444` | Keep fixed; change only to test score stability. |
| `qc.doublets(sample_column=)` | `None` (one sample) | Cells from several captures (10x channels or lanes): doublets are then simulated and scored within each. Never HTO identities. A Harmony pipeline run uses its batch columns. |
| `qc.doublets(threshold=)` | `None` (per sample, automatic) | The `samples` attribute shows a `thresholdMethod` of `"otsu"` (no clear valley) or an estimated rate far from the loading's; then set one score, from 0 to 1, for every sample. |
| `qc.doublets(expected_doublet_rate=)` | `0.05` | The loading predicts another multiplet rate (about 0.8% per 1,000 cells recovered on 10x). |
| `qc.doublets(heterotypic_fraction=)` | `0.0` | Above 0 forces that share of simulated doublets to pair two clusters; it needs two clusters and biases the score. |
| `qc.ambient_contamination(max_contamination=, fdr=)` | `1.0`, `0.05` | Keep; they decide which cells express a gene set (Poisson test against pure soup). |
| `qc.correct_ambient(random_seed=)` | `0` | Keep fixed; change only to check that a result does not depend on the rounding draws. |
| `qc.hto_demultiplexing(positive_quantile=)` | `0.99` (Seurat's `positive.quantile`) | Raise (for example `0.999`) when doublet calls exceed the multiplet rate the loading predicts and their second tag sits near background; each quantile is its own `hto_identity` artifact. A singlet takes its one positive tag; `max_id`, `second_id` and `margin` hold the CLR diagnostics (`ds.artifacts.load_values(ref, value="margin")`). |

## Check before moving on

- The report states what the matrix holds: raw UMIs, corrected counts, removed genes, pre-filtered.
- Retention per sample, per scaffold cluster and per marker-positive group. A population losing
  more than about a third of its cells needs a reason, not a default.
- `resolved_bounds` (raw scale) sit in distribution tails, not inside a mode. Per sample, a low near
  zero means no floor there. Negative lows (Gaussian on skewed counts) mean no lower filtering.
- Removed cells look like debris, not platelets, plasma, erythroid, neutrophil, cycling or CD34+.
- After clustering, QC metrics by cluster (`ds.plots.distribution(keys=qc_cols,
  grouping=run["clusters"])`): a cluster defined only by low counts or high mito is a QC suspect.
- Doublet suspects are high within their own cluster and co-express markers of two lineages.

## Pitfalls

- Accepting pipeline defaults without the audit above.
- Porting scanpy thresholds unchanged: `lows=[200]` is `> 200`, not `>= 200`; use
  `keep_bounds=True`. Imported scanpy QC columns get name-based MAD policies with a warning; pass
  `metric_policies` rather than ignoring it.
- Ignoring a QC `UserWarning` about empty droplets or a second population: the bounds were set
  by the wrong population. On a raw droplet matrix, call cells or import the filtered matrix.
- Keeping HTO calls of a tag that a warning says overlaps its background (an unused tag or a
  failed stain): its positive calls follow the background tail.
- Applying `expm1` to `resolved_bounds` (already raw scale), or taking integer values as raw UMIs
  (SCT-corrected counts are integers too). On corrected or pre-filtered counts, filter gently.
- Pooling samples with different depths under one bound; use `sample_column`.
- Per-sample MAD on a bimodal sample, for example one dominated by erythrocytes: the wide MAD can
  put the lower bound near zero (no floor), or the dominant mode pushes the other population out.
  Read per-sample lows and marker-group retention in that sample.
- Upper `nCounts`/`nFeatures` bounds or a global doublet-score threshold as a doublet filter: both
  remove plasma and other large cells. Compare scores within a lineage and check markers.
- Calling CD34+ progenitors multiplets. Progenitors show a coherent program (`CD34`, `PRSS57`,
  `SPINK2`, `SOX4`, often `GATA2`, `CYTL1`) and no second lineage, even with high counts and
  doublet scores. Multiplets co-express full marker sets of two lineages (`CD3E` with `LYZ` or
  `CD14`; `MZB1` with `CD14`), exceed both parents in counts and sit between their clusters.
- Expecting `I` to shrink after filtering. Pass the returned ref on (`cell_selection=`, or a
  recipe's `cells=`).
- Expecting a recipe to QC-filter a selection or run passed as `cells=`: it keeps those cells as
  they are, unless the run has `cell_calling` or `hashing`. Pass `filtering=True` or a mapping to
  filter them.
- `run.cells` exposes `names`, run results and `snapshot_columns`, not filter columns. QC columns
  need `snapshot_columns=(...)` or `ds.cells.fetch_all(col)[mask]`.

## See also

- `data-access.md`: raw layers on import; `scripts/inspect_store.py` for a first look.
- `features-and-graphs.md`: feature selection and graph construction on the filtered selection.
- `pipeline-runs-and-artifacts.md`: run records, lineage and artifact inspection.
- `plotting.md`: distribution, dotplot and embedding options.
- `integration-and-comparisons.md`: per-sample reporting for condition comparisons.
