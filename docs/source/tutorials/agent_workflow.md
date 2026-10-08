---
description: Run a bounded RNA analysis with structured agent decisions and inspect its evidence.
---

(agent_workflow)=

# Automate an RNA analysis

CyteArc agents run an RNA analysis, compare a small set of analysis settings, and propose cell
identities from measured markers. You supply a prepared store and a model provider. The result
includes clusters, UMAP, marker tables, provisional annotations, and a report explaining the
choices. Review the proposed identities against marker expression and the biology of your study.

The first sections show how to start an analysis and review its results. For a worked example
on intestinal inflammatory bowel disease and healthy-control samples, see
[](garrido_trigo_agents.md). The optional [scripted example](../developers/agent_decisions.md) uses synthetic data and a scripted
model to show how decisions enter the workflow without calling a provider.

## Prepare your input and model

Install the optional agent dependencies:

```bash
# Install the optional dependencies for agent workflows.
uv pip install "cytearc[agent]"
```

Import or mount the data before calling the agent. Follow [](import_and_export.ipynb) for count
files and [](remote_stores.ipynb) for Cytebase mounts. The agent does not download, convert, mount,
or publish a dataset. A local mount may still read remote count bytes during numerical work.

During development, a new analysis rejects complete numerical artifacts from earlier analyses,
including artifacts inherited from a mount's source. Imported labels and embeddings are allowed.
Prepare a clean input separately if needed; `repack_store(..., data_only=True)` copies count data
and rebuilds preparation metadata without the old analysis artifacts. The agent does not delete
old results. Its own completed artifacts can be reused during the same run and explicit resume.

Pass a Pydantic AI model object or a supported `provider:model-name` identifier. Configure
credentials on the provider or in the environment, not in saved study text or runtime settings.
Provider calls may incur charges. The model receives no shell, web retrieval, scientific tools,
or permission to execute generated code.

## Start a real analysis

In a notebook, await the asynchronous entry point. Here `model` is your configured Pydantic AI
model, and `study.zarr` must already be initialized.

```python
# Describe the study and start an asynchronous notebook analysis.
from cytearc.agent import Study, analyze_rna_async

# Start a new RNA analysis with the supplied study and model.
run = await analyze_rna_async(
    "study.zarr",
    model=model,
    study=Study(
        context="Human blood from one healthy donor; no treatment comparison.",
        objective="Describe the major immune populations and uncertain identities.",
        organism="Homo sapiens",
        tissue="blood",
        excludedColumns=["author_annotation"],
    ),
)

# Show whether the analysis completed or needs attention.
print(run.status)

# Show where the analysis history was saved.
print(run.run_dir)

# Write the report and show its local path.
print(run.report())
```

Use `analyze_rna` in a normal Python script. Both interfaces execute numerical stages sequentially.
By default, the complete audit and report live in `agent_runs/<runId>` under the current working
directory. Set `run_dir="analyses/my-study"` to choose another new directory outside the numerical
store. An existing directory is rejected; resume is a separate operation.

`Study` separates supplied facts from scientific configuration. Declare `sampleColumn`,
`captureColumn`, `technicalBatchColumns`, and `protectedColumns` when supported by the experimental
design. These roles distinguish technical variation from biological differences the analysis
should preserve. Repeated samples are not independent biological replicates. Hold evaluation labels out
with `excludedColumns`. Local UTF-8 excerpts in `referenceFiles` have a combined 16 KiB limit.
Missing replication does not block descriptive population discovery.

## Read and continue a run

The returned status is `running`, `needsInput`, `completed`, `failed`, or `interrupted`.
Inspect it before using finalized numerical results. Every persisted outcome supports a report,
including questions and failures. A report error does not downgrade scientific status.

```python
# Open saved evidence and explicitly resume unfinished work.
from cytearc.agent import open_analysis, resume_rna_async

# Reopen the saved agent analysis for inspection.
run = open_analysis(run.run_dir)

# Inspect the saved status and any unresolved questions.
print(run.status, run.pending_questions)

# Inspect which parameter alternatives were measured.
print(run.exploration_coverage)

# Inspect how the recorded decisions were resolved.
print(run.decision_resolutions)

# Write the report from saved evidence without provider calls or numerical computation.
run.report()

# Explicitly resume unfinished work after reviewing its recorded outcome.
run = await resume_rna_async(run.run_dir, model=model)
```

A pending question identifies a missing fact, such as which RNA assay to analyze. The analysis
saves its state and stops; it does not keep a process waiting for your reply. Read
`run.pending_questions`, then call `resume_rna_async` with `answers={questionId: answer}` for
every exact pending ID. Your answer fills a missing detail and cannot replace a scientific setting
already fixed for that analysis.

Resume requires matching scientific inputs and CyteArc
procedure/prompt identity. Operational settings or the provider may change for unfinished
choices. A relocated source must match the fingerprint and saved pipeline/artifact history.
Prototype histories cannot be resumed.

Completed results expose `run.pipeline`, `run.artifacts`, `run.get_markers()`,
`run.plot_embedding()`, `run.plot_markers()`, and `run.annotations`. Numerical access verifies
the source and exact final artifacts. Annotations remain provisional and do not overwrite cell
metadata. A named identity requires observed supporting markers, but this validation cannot
establish that the biological identity is correct. Explicit `unassigned` clusters are permitted.

The external directory retains `run.json`, immutable events, evidence, visible model exchanges,
annotations, reports, and previews. A compact summary in `agent_results/<runId>` inside the
local Zarr store links to the exact final core pipeline and its workspace, selected configuration,
rationale, and external audit location. Read it through `run.compact_result`. It does not duplicate the
full history or make the external audit disposable. See [Agent analysis API reference](/api/agent.html) for details.

## What the agent compares

The baseline uses 1,000 variable genes, 21 PCs, and 11 neighbors. CyteArc then measures alternatives
that change one of these settings at a time. It compares clusterings on the same retained cells,
checks markers for up to two finalists, and makes a final UMAP.

The model interprets those measurements and proposes labels. CyteArc executes the numerical
operations and checks the returned decisions. A clean UMAP or many marker genes does not, by
itself, establish that the chosen identities are correct.

The main analysis uses every retained cell. Some diagnostics use bounded samples: up to 10,000
cells for covariate checks and 2,000 for silhouette assessment. See the
[Agent analysis API reference](/api/agent.html) reference for the full comparison rules and execution limits.

## QC, correction, and uncertainty

The default `qcPolicy="retain"` records QC outlier flags while keeping every supplied cell.
The report also shows how many cells other supported policies would retain; those comparisons
do not apply extra filtering. Global manual thresholds or `qcPolicy="gentleMad5"` require
explicit configuration.

The gentle profile uses five scaled median absolute deviations to identify distant outliers.
A scaled MAD is 1.4826 times the median distance from the median. Counts and detected-feature
metrics are evaluated after `log1p`; mitochondrial percentage is evaluated on its original scale.
The gentle profile removes cells with low counts or few detected features and cells with high
mitochondrial percentage. High counts and feature numbers remain advisory flags. Missing optional
metrics stay unknown, so inspect the QC evidence before deciding whether this policy suits the tissue.

The workflow explicitly supplies its HVG blacklist, normally excluding mitochondrial names
matching `^mt-` case-insensitively. HLA/H2, sex-linked, cell-cycle, and reporter features are
preserved unless explicitly excluded. An organism name alone does not resolve gene identifiers.

Harmony requires declared technical batches, complete labels, protected biological variables,
and supplied evidence separating technical variation from biology. The current design check
requires protected groups across technical batches. Unknown or confounded roles retain native
analysis. A corrected finalist needs its exact native counterpart at the same resolution, plus
measured mixing, preservation, marker, and doublet evidence. Checks use a `0.05` tolerance and
require improvement in at least one mixing measure.

**Doublet scoring is always opt-in.** With `scoreDoublets=False`, correction requiring doublet
evidence is unavailable. The agent never enables scoring implicitly, and scoring never removes
cells automatically.

With `interactionMode="lenient"`, supported optional ambiguities use recorded conservative
rules instead of stopping for an answer. Missing optional metadata stays unknown. If the model
finds several partitions acceptable and they tie, CyteArc prefers the native analysis, then uses
the fixed trial and resolution order. This keeps the choice consistent; it does not establish
that the native analysis is biologically better.

Strict mode leaves those questions pending for you to answer. In either mode, a missing essential
fact stops the run with `needsInput`. Provider failures, invalid output after bounded repair, and
unknown numerical failures also stop work.
