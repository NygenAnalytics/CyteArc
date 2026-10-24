# Pipeline runs and artifacts

Run the modality recipes (RNA, ATAC, ADT, and their multimodal integration), follow and stop a
run, reopen and audit runs, inspect and trace immutable artifacts, branch from a run with exact
refs, and record the analysis for handoff. Docs:
<https://docs.nygen.io/CyteArc/api/pipeline.html>, <https://docs.nygen.io/CyteArc/api/artifacts.html>,
<https://docs.nygen.io/CyteArc/concepts/provenance>, <https://docs.nygen.io/CyteArc/tutorials/reuse-and-tracing>,
<https://docs.nygen.io/CyteArc/analysis-with-agents>.

## When to use

- First pass on any RNA store: one `ds.pipeline.rna(label=...)` gives QC filtering, HVGs, PCA,
  graph, UMAP, Leiden candidates, doublet scores and markers as one durable record.
  `ds.pipeline.atac()` and `ds.pipeline.adt()` do the same for peaks and antibodies, and
  `ds.pipeline.multimodal([...])` integrates runs of the same cells (CITE-seq, multiome, TEA-seq).
- Resuming work: open a run by label or id; inspect any result before consuming it.
- Alternatives outside the fixed recipe: take refs from a completed run and pass them on.

## Key concepts

- Every result is an immutable artifact addressed by an `ArtifactRef` (frozen dataclass):
  `scope` (`"assay"` or `"datastore"`), `kind`, `artifact_id` (64 hex chars; there is no `.id`),
  `assay` (`None` for datastore scope). Cell selections and metadata snapshots are datastore-scoped.
- Reuse is content-addressed: same operation, parameters and exact input refs returns the existing
  complete artifact. Changing anything creates a new artifact; upstream ones are reused.
  `invalidate_cache=True` on a granular method forces a fresh one.
- Reuse also needs the operation revision of the running release. When a release fixes an
  operation, its older artifacts are recomputed (one INFO log line names the replaced artifact and
  the change; this needs `zarr_mode="r+"`), stay loadable, and show as `stale` in lineage;
  `status.is_current` is `False`. Old refs stay valid and listed.
- Results built on a superseded artifact keep `status.is_current` `True` (it judges only their own
  revision), and are recomputed once their input is, because the recomputed input has a new ref.
- A recipe takes one keyword per stage. An optional stage (`cell_calling`, `hashing`, `filtering`,
  `cell_cycle`, `umap`, `tsne`, `leiden`, `paris`, `doublets`) takes `False` to skip it, `True`
  for its defaults, or a mapping that configures and enables it. A stage that always runs (`hvg`,
  `normalization`, `pca`, `ann_index`, `neighbors`, `connectivity`, `embedding_initialization`;
  ATAC `prevalent_peaks` and `lsi`; ADT `panel`; multimodal `integration`) takes `None` for its
  defaults or a mapping; a bool raises `TypeError`. `harmony` is `None` (off) or a mapping with
  `batch_columns`. `membership_strength` and `markers` are plain bools.
- A setting means what the same keyword means on the stage's method (`hvg` takes the keywords of
  `features.hvgs`, `neighbors` those of `graph.neighbors`), and an omitted setting takes that
  method's default. Before anything is written, a recipe refuses an unknown key (`ValueError`
  listing the allowed ones) and values that a later stage would refuse, including: `cell_calling`
  and `hashing` values (such as `fdr` or `positive_quantile`); a per-capture column that leaves an
  input cell unlabelled; `pca` `dims` not below the `hvg` `top_n` (RNA) or the panel size (ADT);
  an ATAC `prevalent_peaks` `top_n` not below the assay's peak count, or below LSI `dims` plus
  `skip_first`; and SNN source runs of different `k`. A value that only the method itself
  rejects, such as `ann_metric`, Leiden `backend`, a doublet setting or
  `paris={"min_cluster_size": -5}`, fails its stage.
- `cells` is a boolean cell column (default `"I"`), a datastore cell-selection ref, or a completed
  run, whose `analysis_cell_selection` is reused as the same ref. `filtering=None` (the default)
  filters the cells of a column (RNA, ATAC), and an `rna` run with `cell_calling` or `hashing`
  filters the cells those stages keep, whatever its `cells`. Otherwise a given selection's or
  run's cells stay as they are, and an ADT run filters only when asked.
- A recipe writes artifacts plus a run record only. It never edits live `I` or adds metadata
  columns, and needs `zarr_mode="r+"` (the default); a read-only store raises `PermissionError`,
  though `open`, reports and plots still work.
- `PipelineRun` is a read-only `Mapping[str, ArtifactRef]` (`keys`, `items`, `values`, `get`,
  `run["pca"]`) plus `run_id`, `label`, `status`, `recipe` (`"rna"`, `"atac"`, `"adt"` or
  `"multimodal"`), `assay`, `started_at_ns`, `finished_at_ns`, `report()`, `cells`, `features`.
  Outputs and views work only on `status == "completed"`; otherwise they raise `RuntimeError`.
  Runs saved by earlier releases record recipe `"basic_rna_analysis"` and still open.
- Labels are immutable. Only a completed run acquires its label; until then `run.label` is `None`
  and `report()["run"]["requestedLabel"]` holds it. A completed label cannot be reused
  (`ValueError` before any work). Failed and interrupted runs reserve nothing.
- There is no resume. A new run reuses a complete artifact from an earlier attempt only when its
  operation, parameters, inputs and operation revision match.
- `run.cells` and `run.features` are frozen views of captured fields only. Live columns such as
  author labels are absent unless listed in `snapshot_columns` on the run.

## Recipes

### Run the baseline

Default `rna` outputs in record order, with their kinds: `input_cell_selection`,
`analysis_cell_selection` (`cell_selection`), `feature_universe`, `cell_cycle`,
`highly_variable_features` (`feature_selection`), `normalized`, `pca` (`reduction`), `ann_index`,
`neighbors`, `connectivity_map`, `embedding_initialization`, `umap` (`embedding`), `leiden_0.5`
... `leiden_1.25` (`cluster_labels`), `cluster_selection`, `clusters` (the selected Leiden ref
itself, not a copy), `doublets` (`doublet_score`), `markers` (`marker_table`). Optional stages add
keys in place: `cell_calling` and `called_cell_selection`, then `hashing` (`hto_identity`) and
`hashed_cell_selection` before `analysis_cell_selection`; `harmony` (`batch_correction`) after
`pca`; `paris` (`cluster_cut`) after the Leiden keys; `membership_strength` and `tsne` before
`doublets`. Disabled stages omit their keys. Timings measured on the 10x 5K PBMC docs dataset
when the default also ran Paris: a fresh baseline with an empty Numba cache took 109 s on local
counts, about 90 s of it one-time compilation (graph stage 24 s, Leiden kernels 39 s, Paris and
markers 21 s); a second run in the same process took 19 s, and a later process, which loads most
kernels from Numba's cache, took 34 s. On remote counts expect about a minute for a few thousand
cells.

```python
import cytearc

cytearc.configure_output(level="WARNING", progress=False)
ds = cytearc.DataStore("analysis.zarr", nthreads=4, min_features_per_cell=-1)
run = ds.pipeline.rna(label="baseline")
list(run.keys())
run["clusters"] == run["leiden_0.5"]  # True on the 10x 5K PBMC docs dataset
```

### Other modalities and their outputs

`rna`, `atac` and `adt` end in the same tail (`ann_index`, `neighbors`, `connectivity_map`,
`embedding_initialization`, `umap`, Leiden candidates, `cluster_selection`, `clusters`,
`markers`). The rest differs:

| Recipe | Stages before the graph (defaults) | Feature output, also a `run.features` field | Other outputs |
| --- | --- | --- | --- |
| `rna` | `cell_calling` and `hashing` off; `filtering`; `cell_cycle`; `hvg` (`top_n` 1000); `normalization`; `pca` (`dims` 21); `harmony` off | `highly_variable_features` | `cell_cycle`, `pca`, `doublets` |
| `atac` | `filtering` on `<assay>_nCounts` and `<assay>_nFeatures`; `prevalent_peaks` (`top_n` `min(10000, peaks - 1)`); TF-IDF `normalization`; `lsi` (`dims` 11, first component skipped); `harmony` off | `prevalent_peaks` | `lsi`; no doublets or cell cycle |
| `adt` | `filtering` off unless asked (then on `<assay>_nCounts`); `panel` (leaves out names matching `DEFAULT_PANEL_EXCLUDE`, `(?i)isotype\|control\|ctrl`); CLR `normalization`; `pca` (`dims` one fewer than the panel, at most 21); `harmony` off | `panel` | `pca` |
| `multimodal` | `integration` of the source runs' graphs (`method="wnn"` or `"snn"`) | none; `run.features` and `assay` are the first run's | `integrated_graph` in place of the three graph outputs; no `cluster_selection`; `markers_<assay>` per source run instead of `markers` |

`run.cells` holds the frozen fields of the stages that ran: `I, ids, names`, any
`snapshot_columns`, `s_score, g2m_score, cell_cycle_phase` (RNA), `umap_1, umap_2`, `tsne_1,
tsne_2`, `leiden_*`, `paris`, `clusters`, `membership_strength`, `doublet_score,
predicted_doublet` (RNA), and `weight_<assay>` for each modality of a WNN run.

### Analyze several modalities of the same cells

CITE-seq: analyze RNA first, then the antibodies over the RNA run's QC-passing cells, then
integrate. Multiome and TEA-seq follow the same pattern with `ds.pipeline.atac(cells=rna)`.

```python
rna = ds.pipeline.rna(label="rna")
adt = ds.pipeline.adt(label="adt", cells=rna)  # the RNA run's cells as they are; not filtered
wnn = ds.pipeline.multimodal([rna, adt], label="wnn")  # weighted nearest neighbours
snn = ds.pipeline.multimodal([rna, adt], method="snn", label="snn")  # shared nearest neighbours
list(wnn)  # ..., integrated_graph, embedding_initialization, umap, leiden_1.0, clusters, markers_RNA, markers_ADT
wnn.cells.fetch("weight_ADT")  # each cell's ADT weight; weights sum to 1 per cell
ds.plots.modality_weights(graph=wnn["integrated_graph"], layout=wnn["umap"], show=False).close()

# TEA-seq: three modalities of one cell selection.
atac = ds.pipeline.atac(label="atac", cells=rna)
teaseq = ds.pipeline.multimodal([rna, atac, adt], label="wnn3")
```

- Integrated runs must share one cell-selection artifact, not only the same cells: two runs over
  `cells="I"` make two selections even of identical cells, and `multimodal` raises `ValueError`
  ("Run ... analyzes another cell selection than run ..."). Pass `cells=<first run>` to every
  later modality, without `filtering`, which would make a new selection. One run per assay.
- WNN needs graphs built on PCA, LSI or Harmony coordinates, so not a `pca={"dims": 0}` run, and
  takes runs of different `k` (it keeps the smallest). SNN integrates any graph, but its source
  runs must share one `neighbors` `k`; other counts are refused before the run starts.
- The integrated graph has no shared coordinates for silhouettes, so `leiden` defaults to one
  resolution, 1.0; one other, `leiden={"partitions": [1.75]}`, needs nothing more, and several
  need `leiden={"partitions": [0.8, 1.2], "selected": 1.2}`. `cluster_selection` is always
  skipped.
- The joint embedding starts from the first run's coordinates. Export writes the first run's raw
  counts with the joint UMAP and clusters and has no normalized output.
- The `integration` stage summary records `meanWeights` per assay; per-cell weights are run fields.

### Reopen, list and report runs

`open` needs exactly one of `label` or `run_id`; an unknown label raises `KeyError`. `list_runs`
is newest first, all statuses unless filtered, `limit=20` by default. Runs saved by any earlier 1.x
release reopen. `list_runs` skips a run that cannot be read, such as one a newer CyteArc saved, with a
warning naming it.

```python
run = ds.pipeline.open(label="baseline")
same = ds.pipeline.open(run_id=run.run_id)
ds.pipeline.list_runs(status="completed", limit=5)  # repr shows recipe, status, assay, label
report = run.report(format="dict")  # keys: run, stages, summary
[(s["stage"], s["status"], s["metrics"]["wallSeconds"]) for s in report["stages"]]
report["run"]["config"]  # cells, snapshotColumns, species, stages: one entry per stage keyword
{s["stage"]: s["summary"] for s in report["stages"] if s["summary"]}  # stage diagnostics
report["summary"]["warnings"]  # [{"stage", "message"}] logged while each stage ran
len(report["summary"]["createdArtifacts"]), len(report["summary"]["reusedArtifacts"])
print(run.report(format="markdown"))  # config, stage table, diagnostics, outputs, failure
```

Stage summaries are small and read no matrix, so check them before loading artifacts. Each is
computed inside its stage, so its time, memory and warnings count as the stage's. filtering
`{"cells", "inputCells", "removedCells", "samples"}`; pca `{"cells", "dims", "varianceRatio",
"varianceExplained"}`; each `leiden_<res>` `{"clusters", "largestCluster", "smallestCluster"}`;
cluster_selection `{"selected", "scores"}`; membership_strength `{"mean", "belowHalf"}`;
doublets `{"predictedDoublets", "rate", "samples"}`; integration `{"assays", "meanWeights"}`.
`ann_index` records `None`, and metadata snapshots add nothing to a summary. A stage with several
summarized outputs keys its summary by output: input_snapshot `{"input_cell_selection": {"cells"}, "feature_universe":
{"features"}}` (given a selection or a run: `{"features"}` alone); cell_calling `{"cell_calling":
{"retain", "knee", "inflection", "n_ambient"}, "called_cell_selection": {"cells", "inputCells",
"removedCells"}}`, but per capture only the `called_cell_selection` summary, which adds
`calledPerCapture` and `kneePerCapture`; hashing `{"hashing": {"hashtags", "negative",
"multiplets"}, "hashed_cell_selection": {"cells", ...}}`. The markdown report adds a Warnings
column to the stage table and a Diagnostics section with each stage's summary and warnings.

### Follow a run: events, progress and logs

A recipe calls `callback=` with each `PipelineEvent`, and so does every listener registered with
`cytearc.pipeline_listener(fn)` around the call (context-local; nests; read when the run starts).
Callbacks run synchronously on the run's thread, so keep them fast or hand events to a queue; an
exception in one is logged and ignored. They run in the caller's context, outside the stages:
what they log stays out of the stage records, and a pending stop never interrupts CyteArc reads
they make.

```python
import json


def log_event(event):
    if event.kind == "stage_progress":  # per bar: the first, then one per 0.25 s, and the last
        return
    record = event.to_dict()  # JSON-compatible, camelCase keys
    keep = ("kind", "stage", "ordinal", "stages", "summary", "warnings", "error")
    print(json.dumps({key: record[key] for key in keep}, default=str), flush=True)


run = ds.pipeline.rna(label="baseline", callback=log_event)
with cytearc.pipeline_listener(log_event):  # every run started in this block
    adt = ds.pipeline.adt(label="adt", cells=run)
```

- Order: `run_started`; per stage `stage_started`, any `stage_progress`, then `stage_completed`;
  or `stage_skipped` (no start event); or `stage_failed` then `run_failed`; or
  `stage_interrupted` then `run_interrupted`. `run_completed` comes once the record is complete.
  A stop seen just after a stage's work finished records that stage completed: `stage_completed`,
  then `run_interrupted` naming no stage. A stop between stages, also while a skipped stage is
  recorded, emits `run_interrupted` alone, naming no stage. So `run_interrupted` names a stage
  only when that stage's work was cut short.
- Fields: `kind`, `run_id`, `recipe`, `label`, `stage`, `ordinal` and `stages` (the position in
  `report()["run"]["stageOrder"]`, skipped stages included), `time_ns`, `progress`
  (`description`, `completed`, `total`), `metrics` (wall seconds and sampled RSS), `summary`,
  `outputs` (`(key, ref)` pairs), `reused` (keys), `warnings`, `error`.
- Progress events arrive even with `configure_output(progress=False)`; UMAP's own bar is not
  forwarded.
- `cytearc.configure_output(stdlib=True)` sends CyteArc's log records to the standard-library
  logger `"cytearc"`; records logged inside a stage carry `cytearc_run_id` and `cytearc_stage`
  attributes. `stdlib=False`, or `set_verbosity()`, restores CyteArc's own sink.

### Cancel a run

`cytearc.CancellationToken` stops a run from any thread, such as a timer, a UI or a callback that
watches a budget. The run stops at its next checkpoint, records the run as `interrupted`
(interruption kind `"cancelled"`), keeps completed artifacts reusable, and raises
`cytearc.PipelineCancelled` (`run_id`, `stage`, `reason`). A stage whose work was cut short is
recorded `interrupted` and named by `stage`; a stage whose work had finished is recorded
`completed`, and `stage` is `None`, as it is for a stop between stages.

```python
import threading

token = cytearc.CancellationToken()
timer = threading.Timer(1800, token.cancel, args=("time budget spent",))
timer.start()
try:
    run = ds.pipeline.rna(label="baseline", cancel=token)
except cytearc.PipelineCancelled as error:
    stopped = ds.pipeline.open(run_id=error.run_id)
    print(error.stage, error.reason, stopped.status, stopped.report()["run"]["interruption"]["kind"])
finally:
    timer.cancel()
```

A token stays cancelled: one cancelled before the call raises `PipelineCancelled` with `run_id`
`None` and writes nothing, so make a new token per attempt. `token.cancel()` returns `True` only
the first time. Signals never cancel a token. A signal that arrives after a cancellation keeps
the interruption kind `"cancelled"`, but once the record is written the signal takes its usual
effect instead of `PipelineCancelled`: with default handlers, SIGINT raises `KeyboardInterrupt`
and SIGTERM ends the process. Only a second signal escalates at once.

### Configure a run

Validation errors (an unknown setting, a bool for a stage that always runs, a snapshot column that
collides with a run field, a reused label, and the value checks in Key concepts) raise before any
run record exists. `leiden={"selected": ...}` replaces the silhouette choice and skips
`cluster_selection`. A run that adds only `snapshot_columns` creates one `column_snapshot` and
reuses every other artifact.

```python
custom = ds.pipeline.rna(label="leiden_custom", leiden={"partitions": [0.4, 0.8]})
pinned = ds.pipeline.rna(
    label="leiden_pinned",
    leiden={"partitions": [0.5, 1.0], "selected": 1.0},
    umap=False, doublets=False, markers=False, cell_cycle=False,
)
pinned["clusters"] == pinned["leiden_1.0"]  # True; no cluster_selection key
branch = ds.pipeline.rna(
    label="hvg2k_pc30_k21",
    hvg={"top_n": 2000}, pca={"dims": 30}, neighbors={"k": 21},
    paris=True, membership_strength=True,  # off by default
)
labelled = ds.pipeline.rna(label="with_labels", snapshot_columns=["cell_type"])  # reuses baseline
"cell_type" in labelled.cells.columns  # True
labelled["clusters"] == run["clusters"]  # True: same cells, same artifacts
```

`cell_type` stands for a label column you own; never snapshot held-out author labels.

### Migrate from `pipeline.run`

Releases up to 1.3.0 had one `ds.pipeline.run(...)` with shortcuts and a `params` mapping; both
are gone, with no alias (`ds.pipeline.run` raises `AttributeError`). An `rna` run with the same
settings reuses the artifacts of an earlier `pipeline.run`, except that the filtered cells of a
run with Harmony or `snapshot_columns`, and its Harmony fit, get new identities: those columns are
now frozen apart from the ones filtering reads. Such a run recomputes once from there on; nothing
is migrated.

| Earlier call | Recipe call |
| --- | --- |
| `ds.pipeline.run(...)` | `ds.pipeline.rna(...)` |
| `cell_key="qc_pass"` | `cells="qc_pass"` |
| `hvg_count=2000`, `pca_dims=30`, `neighbors_k=21` | `hvg={"top_n": 2000}`, `pca={"dims": 30}`, `neighbors={"k": 21}` |
| `harmony_batch_columns=["batch"]` | `harmony={"batch_columns": ["batch"]}` |
| `params={"hvg": {"blacklist": s}}`, `params={"<stage>": {...}}` | `hvg={"blacklist": s}`, `<stage>={...}` |
| `params={"species": s}`, `params={"membership_strength": True}` | `species=s`, `membership_strength=True` |
| Paris on by default | `paris=True` |
| recipe `"basic_rna_analysis"`, event `pipeline_interrupted` | recipe `"rna"`, events `stage_interrupted` and `run_interrupted` |

### Handle failures and interruptions

A stage error raises `PipelineExecutionError` (`run_id`, `stage`, cause in `__cause__`). SIGTERM,
SIGINT, SIGHUP or `KeyboardInterrupt` stop at the next safe checkpoint (in testing, the running
stage finished and kept its artifact) and record `status="interrupted"`. Rerun with the same
label; completed artifacts are reused (interrupted during UMAP: rerun 10 s, 16 reused; 10x 5K
PBMC docs dataset). Run it as `timeout 1800 python job.py`; behind a wrapper such as `uv run`, put
`timeout` inside it (`uv run timeout 1800 python job.py`), because `timeout 1800 uv run ...` killed the run without the handler (performance-and-export.md).

```python
from cytearc import PipelineExecutionError

try:
    ds.pipeline.rna(label="bad", paris={"min_cluster_size": -5})
except PipelineExecutionError as error:
    failed = ds.pipeline.open(run_id=error.run_id)
    error.stage, failed.status, failed.report(format="dict")["run"]["error"]
interrupted = ds.pipeline.list_runs(status="interrupted")
```

A process killed mid-stage (SIGKILL, OOM, or a `timeout` outside a `uv run` wrapper) leaves a record
stuck at `status="running"` with `summary["uncleanIncomplete"]` true; it claimed no label, so a
rerun with the label works. Check the process, not the status. Not executed (needs a hard-killed
finalizer): if a process died after claiming its label, the label stays blocked. After confirming
the process is gone, call `ds.pipeline.abandon_label_claim(label=..., run_id=..., reason=...)`,
then rerun. It refuses completed owners and has no timeout heuristic.

### Inspect, load and find artifacts

```python
status = ds.artifacts.inspect(run["pca"])
status.exists, status.complete, status.path, status.operation  # ..., "run_pca"
status.parameters  # {'dims': 21, 'feat_scaling': True}
status.inputs  # serialized refs: normalized, feature_scaling, pca_cell_selection
status.input_ref("normalized") == run["normalized"]  # True
status.execution_options, status.created_at_ns, status.cytearc_version
status.revision, status.current_revision, status.is_current  # 1, 1, True until a fix
group = ds.artifacts.load(run["pca"])  # zarr group: data, loadings, center
group["data"].shape  # (analysed cells, 21)

ds.artifacts.list(complete_only=True)  # assay scope, default assay
ds.artifacts.list(kind="cell_selection", scope="datastore")
on_graph = ds.artifacts.list(
    kind="cluster_labels",
    operation="run_leiden_clustering",
    inputs={"graph": run["connectivity_map"]},
)
[ds.artifacts.inspect(ref).parameters["resolution"] for ref in on_graph]

decision = ds.artifacts.load(run["cluster_selection"])
decision.attrs["selectedKey"], decision.attrs["candidateKeys"], decision["scores"][:]
```

### Branch from a completed run and trace lineage

Pass exact run refs to granular methods. Same inputs return the run's own artifacts; new
parameters create a branch. A run saved by an earlier release, whose default Leiden backend was
`"igraph"`, holds Leiden refs that only `ds.clusters.leiden(..., backend="igraph")` reuses.
Explicit normalization/PCA/graph chains: features-and-graphs.md. Clustering choices:
clustering-and-embedding.md.

```python
graph = run["connectivity_map"]
ds.clusters.leiden(graph, resolution=0.5) == run["leiden_0.5"]  # reused, no recompute
fine_clusters = ds.clusters.leiden(graph, resolution=2.0)  # new cluster_labels ref
fine_markers = ds.markers.search(fine_clusters, features=run["feature_universe"])

cells = run["analysis_cell_selection"]
hvgs = run["highly_variable_features"]
ds.features.normalize(cells, hvgs) == run["normalized"]  # True: reused
ds.plots.embedding(layout=run["umap"], color_by=fine_clusters, show=False).close()

lineage = ds.artifacts.lineage({"selected": run["clusters"], "fine": fine_clusters})
markdown = lineage.to_markdown()  # Mermaid graph (shared upstream once) plus details
mermaid = lineage.to_mermaid()
snapshot = ds.summary().to_dict()  # columns, artifact inventory, run counts, labels
```

### Record the analysis for handoff

Write the prospective record (question, unit of inference, input refs, planned writes,
alternatives, decision criteria) before the first mutating call, then extend it into the handoff:
add metadata roles and confounding, evidence per alternative, unsupported claims, required
validation and exported files. Refs round-trip with `to_dict()` and `ArtifactRef.from_dict()`.

```python
import json
from pathlib import Path

out = Path("handoff")
out.mkdir(exist_ok=True)
refs = {"cellSelection": "analysis_cell_selection", "featureSelection": "highly_variable_features",
        "graph": "connectivity_map", "selectedClusters": "clusters"}
record = {
    "question": "Which broad populations are present?",
    "unitOfInference": "cells from one sample; no replicate-level claims",
    "pipelineRun": {"runId": run.run_id, "label": run.label, "recipe": run.recipe,
                    "status": run.status},
    "refs": {name: run[key].to_dict() for name, key in refs.items()}
    | {"fineClusters": fine_clusters.to_dict()},
    "alternatives": "Leiden 0.5 (pipeline choice) vs 2.0",
    "decision": "provisional", "openQuestions": ["Does MAD filtering remove a real population?"],
}
(out / "handoff.json").write_text(json.dumps(record, indent=2))
(out / "run_report.md").write_text(run.report(format="markdown"))
(out / "lineage.md").write_text(lineage.to_markdown())
saved = json.loads((out / "handoff.json").read_text())
cytearc.ArtifactRef.from_dict(saved["refs"]["fineClusters"]) == fine_clusters  # True
```

## Recipe arguments

| Argument | Default | Change when |
| --- | --- | --- |
| `label` | `None` | Always name runs you will reopen; pick a new name per variant. |
| `assay` | the default assay when it has the recipe's modality, else the store's only such assay | Several assays of one modality, such as `RNA` and an ambient-corrected `RNA_soupx`. |
| `cells` | `"I"` | A boolean column, a selection ref (a QC choice, singlets, measured cells) or a completed run (its analysis cells; how modalities share cells). On a merged store whose assay measured only some cells, the run raises `UnmeasuredCellsError` before any stage and never narrows the cells: pass `ds.select_measured_cells("RNA", cell_selection=ds.snapshot_cell_selection("I"))` with `filtering=True`, or a column of the cells of `I` that RNA measured, `ds.cells.insert("RNA_measured", ds.cells.fetch_all("I") & ds.cells.fetch_all("RNA_I"))`; `RNA_I` alone also holds cells that a filter removed from `I`. |
| `filtering` | `None`: pooled MAD, 3 MADs, over `RNA_nCounts/nFeatures/percentMito/percentRibo` (ATAC: `<assay>_nCounts/nFeatures`) when `cells` is a column, or in an `rna` run with `cell_calling` or `hashing`; otherwise none for a selection or a run, and none for an ADT run (`True`: `<assay>_nCounts`) | Data already filtered (`False`), a given selection to filter (`True`), or per-sample or manual bounds (mapping; see quality-control.md). |
| `cell_calling`, `hashing` (rna) | `False` | Raw droplets (EmptyDrops; `{"sample_column": ...}` calls each capture apart and needs a capture label for every input cell) or HTO-hashed cells (one demultiplexing fit over all captures); see quality-control.md. |
| `cell_cycle` (rna) | `True` | `False` to save time (the slowest stage on remote counts) or when feature names are not symbols; `{"s_genes": [...], "g2m_genes": [...]}` for other lists. |
| `hvg` (rna) | `None`: `top_n` 1000, `DEFAULT_HVG_BLACKLIST` | A representation choice is under test (`{"top_n": 2000}`); species blacklist (`{"blacklist": ...}`); `flavor`, `batch` (features-and-graphs.md). |
| `pca` (rna, adt) | `None`: `dims` 21; ADT one fewer than the panel, at most 21 | A representation choice is under test. `dims` must be below the `hvg` `top_n` (RNA) or the panel size (ADT). RNA `{"dims": 0}` skips PCA: the graph uses `run["normalized"]` (no `pca` output) and needs `doublets=False` and no Harmony. Its memory and time then scale with the HVG count: the ANN index holds about `4 * top_n + 8 * ann_m + 130` bytes per cell (4.3 GB at 1M cells and 1,000 HVGs with the default `ann_m=24`) and raises `MemoryError` over `mem_budget`; on large data pass `embedding_initialization={"batch_size": ...}` below the cell count for the streamed k-means fit. |
| `prevalent_peaks`, `lsi` (atac) | `None`: `top_n` `min(10000, peaks - 1)`; `dims` 11, `skip_first=True` | More peaks or LSI components for large or deep data. `top_n` must stay below the assay's peak count and at least LSI `dims` plus `skip_first` (a bool); the run records the resolved `top_n`. |
| `panel` (adt) | `None`: `exclude=DEFAULT_PANEL_EXCLUDE` | Controls named otherwise (`{"exclude": "(?i)control"}`); `{"exclude": None}` keeps every antibody. The run records the excluded names. |
| `harmony` | `None` (off) | Real technical batch columns: `{"batch_columns": ["batch"]}`; see integration-and-comparisons.md. |
| `ann_index`, `neighbors`, `connectivity`, `embedding_initialization` | `None`: `k` 11 | `neighbors={"k": 21}` for smoother graphs; other keywords of the `graph` and `embeddings` methods. Modality runs that an SNN run integrates need one `k`. |
| `umap` | `True` | `False` skips `embedding_initialization` (unless t-SNE runs) and `umap`; a mapping sets `embeddings.umap` keywords. |
| `tsne` | `False` | `True` or a mapping of `embeddings.tsne` keywords; a new t-SNE needs the `tsne` extra. |
| `membership_strength` | `False` | `True` to score each cell's graph support for its cluster (a `membership_strength` field). |
| `leiden` | `True`: 0.5, 0.75, 1.0, 1.25 with a silhouette pick (multimodal: 1.0) | Other candidates: `{"partitions": [...]}`; pin with `{"selected": ...}`; `backend`, `random_seed`. |
| `paris` | `False` | `True` for a hierarchy to diagnose (`ds.plots.cluster_tree`); `{"n_clusters": ...}`. |
| `doublets` (rna) | `True` | `False` to skip; `{"sample_column": "capture"}` with several captures. Needs Leiden and PCA. |
| `markers` | `True` | `False` to skip; needs Leiden. |
| `snapshot_columns` | `()` | Run-mode plots or `run.cells` need live columns. A new run that adds them reuses every artifact (see Pitfalls). |
| `species` | `None` | Recorded with the run only. |
| `callback`, `cancel` | `None` | Progress events, logging, a stop button or a time budget (above). |
| `method`, `integration` (multimodal) | `"wnn"`, `None` | `"snn"` for graphs without reduced coordinates (its runs must share one `k`); `{"chunk_size": ...}`, WNN `{"l2_normalize": ...}`. |

## Check before moving on

- `run.status == "completed"`, `run.recipe` is the one you meant, and the expected keys are present.
- `run.report(format="markdown")`: stage statuses, wall seconds, created versus reused counts,
  resolved `filtering` config, the Diagnostics section and any warnings, and any failure or
  interruption section.
- A run you reopen from an earlier release: `ds.artifacts.lineage(run).to_markdown()` marks `stale` each
  output that a later operation revision supersedes; running the configuration again recomputes it
  and the results built on it.
- Kept cells: the filtering summary, or `run.cells.fetch_all("I").sum()` versus the store total.
  Audit removed cells (by your own clusters or a marker panel, not held-out labels) before trusting
  clusters.
- `decision["scores"]` from `cluster_selection`: silhouette is a provisional baseline, not ground
  truth.
- For any ref you consume: `ds.artifacts.inspect(ref).complete` is `True`, and `operation`,
  `parameters` and `inputs` match what you intend.

## Pitfalls

- Changing `filtering` changes `analysis_cell_selection` and so everything downstream, but
  `snapshot_columns` and Harmony's batch columns are frozen apart from the QC columns that
  filtering reads (as `column_snapshot` and `harmony_snapshot` outputs of `input_snapshot`). A run
  that adds only `snapshot_columns` creates its `column_snapshot` and reuses every other artifact,
  and a run that adds Harmony reuses everything up to PCA.
- A given selection or run is not filtered by default, unless the `rna` run has `cell_calling` or
  `hashing`: pass `filtering=True` to filter it, or `filtering=False` to keep what those stages
  keep.
- `ds.artifacts.list(kind="cell_selection")` silently returns `[]`; pass `scope="datastore"`.
- Compare cell selections from different routes by mask, not by ref: equal values can have
  different ids.
- A `PipelineRun` is bound to the `DataStore` object that opened it. After reopening the store,
  reopen the run, or plotting raises `ValueError`.
- `run.get(key)` returns `None` for a disabled stage, but raises `RuntimeError` on a run that did
  not complete. `run["paris"]` raises `KeyError` unless the run had `paris=True`.
- `run.cells.fetch("clusters")` is integer, while `markers.load(...)["group_id"]` is a string.
- In a backgrounded non-interactive shell, SIGINT is ignored; use SIGTERM to stop a run cleanly.
  `SIGKILL` or OOM leaves an incomplete run with no cleanup.
- Never read private Zarr paths or mutate `ds.z`, `ds.zw` or run records; use the methods above.

## See also

- quality-control.md: filtering policies behind `analysis_cell_selection`, cell calling, hashtags,
  ambient RNA and doublet removal with recipes.
- features-and-graphs.md: explicit normalization, PCA, ANN, neighbours and graph chains.
- clustering-and-embedding.md: re-clustering, UMAP/t-SNE branches, partition comparison.
- markers-and-annotation.md: `markers.load`, marker tables and labels.
- plotting.md: `ds.plots.*(run=run, ...)` versus explicit refs.
- performance-and-export.md: `ds.to_anndata(run=run)`, background runs and resource settings.
