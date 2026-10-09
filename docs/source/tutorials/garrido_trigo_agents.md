---
description: Run an unattended CyteArc agent analysis on a local copy of the Garrido-Trigo CyteBase RNA dataset.
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(garrido_trigo_agents)=

# Analyze Garrido-Trigo RNA with CyteArc agents

This worked example shows an automated RNA analysis from input preparation to review of the
agent's proposed identities. The data come from an intestinal inflammatory bowel disease (IBD)
and healthy-control study. We will prepare a local copy, describe the study, run the analysis
with default scientific settings, and inspect its evidence before accepting any cell labels.

The input preparation is longer than the analysis call because this published store needs
local preparation before the current agent can use it. Do this once; later visits can resume
the saved analysis. For the shorter general API example, start with [](agent_workflow.md).

Use the page download menu to save the source notebook.
The general workflow and an example without provider credentials are in
[](agent_workflow.md).

## Set up the notebook folder

Use an environment with CyteArc's `agent`, `cytebase`, `docs`, and `extra` extras.
Configure `AGENT_API_KEY` in your environment or a local `.env` file. This example
uses the OpenAI-compatible Baseten endpoint and `deepseek-ai/DeepSeek-V4.1-Flash`;
model calls send supplied metadata and measured marker evidence to that provider.
CyteArc requests disabled reasoning on every call. Provider calls incur usage, and
the configured request limits are not a universal spending guarantee.

Run the notebook from its own folder. `CYTEARC_AGENT_NOTEBOOK_DIR` can choose another
working folder, and `CYTEARC_AGENT_ENV_FILE` can point to a local environment file.
Neither credentials nor private source locations are displayed or saved in the
notebook outputs.

```{code-cell} ipython3
# Load local configuration and tools for reviewing the analysis.
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import Image, display

import cytearc
from cytearc.agent import (
    AnalysisConfig,
    RuntimeConfig,
    Study,
    analyze_rna_async,
    resume_rna_async,
)

# Load provider credentials from the environment without displaying them.
env_file = os.environ.get("CYTEARC_AGENT_ENV_FILE")
_ = load_dotenv(env_file) if env_file else load_dotenv()

# Keep the dataset and saved agent run in one working directory.
work_dir = Path(os.environ.get("CYTEARC_AGENT_NOTEBOOK_DIR", Path.cwd())).resolve()
work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(work_dir)

# Hide routine progress while retaining analysis warnings.
cytearc.configure_output(level="WARNING", progress=False)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

# Choose the public cohort and local paths for its analysis.
dataset_id = "garridotrigo_2023_ibd_hc_10x_single_cell_transcriptomics_data_9bfecd44"
data_path = work_dir / "data.zarr"
run_dir = work_dir / "agent_runs" / "garrido-trigo"
```

## Prepare the input once

As in [](agent_workflow.md), prepare the dataset before asking the agent to analyze it.
`Catalog.mount_datastore()` creates a writable local mount whose counts remain
remote. It has no full-copy option. To obtain a complete
local copy, resolve the source through CyteBase, download its files read-only, and
use the public `repack_store(data_only=True)` operation to prepare `data.zarr` with the required metadata.
This preparation leaves published data unchanged and excludes prior numerical
analysis artifacts from the working copy.

The downloaded snapshot stays in `cytearc_datasets/`. On subsequent notebook runs,
the existing prepared `data.zarr` is retained and its saved analysis can be resumed.
A different source or scientific policy should use a new notebook folder.

```{code-cell} ipython3
# Download the catalog source and prepare a separate local count store.
from huggingface_hub import BucketFile, download_bucket_files, get_token, list_bucket_tree
from huggingface_hub.utils import disable_progress_bars

from cytearc import cytebase
from cytearc.tools.repack_zarr import repack_store

# Prepare the local dataset once; later visits reuse this directory.
if not data_path.exists():

    # Resolve this dataset through the public CyteBase catalog.
    catalog = cytebase.Catalog()
    entry = catalog.dataset(dataset_id)
    source_uri = entry.row["zarr_uri"]

    # Check that the source uses the bucket layout expected by this recipe.
    if not source_uri.startswith("hf://buckets/"):
        raise ValueError("This copy recipe expects a CyteBase HF bucket source")
    pieces = source_uri.removeprefix("hf://buckets/").split("/")
    bucket, prefix = "/".join(pieces[:2]), "/".join(pieces[2:]).rstrip("/") + "/"

    # Keep the downloaded source separate from the prepared working copy.
    raw_path = work_dir / "cytearc_datasets" / f"{dataset_id}.zarr"
    raw_path.mkdir(parents=True, exist_ok=True)
    disable_progress_bars()
    token = get_token() or False

    # List only files belonging to the selected dataset.
    objects = [
        item for item in list_bucket_tree(bucket, prefix=prefix, recursive=True, token=token)
        if isinstance(item, BucketFile)
    ]

    # Validate each relative destination before constructing the download list.
    transfers = []
    for item in objects:
        if not item.path.startswith(prefix):
            raise ValueError("A source object is outside the dataset prefix")
        relative = Path(item.path.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid source object path")
        transfers.append((item, raw_path / relative))
    if not transfers:
        raise ValueError("The published source contains no downloadable files")

    # Download the source, then copy its counts without prior numerical results.
    download_bucket_files(bucket, transfers, token=token, raise_on_missing_files=True)
    repack_store(
        str(raw_path), str(data_path), data_only=True,
        profile="fast_local", nthreads=4, mem_budget="4G",
    )

# Inspect the prepared input without modifying its selected cells.
store = cytearc.DataStore(
    str(data_path), zarr_mode="r", min_features_per_cell=-1,
    nthreads=4, mem_budget="4G",
)

# Record cells and genes so the final check can verify that they stayed unchanged.
input_cells = store.cells.fetch_all("I").copy()
input_ids = store.cells.fetch_all("ids").copy()
input_features = store.RNA.feats.fetch_all("ids").copy()

# Report the cohort dimensions before starting the analysis.
print({"dataset": dataset_id, "cells": int(input_cells.sum()), "genes": len(input_features)})
```

## Supply the scientific context

This is descriptive population discovery in the supplied IBD and healthy-control cohort.
The dataset name contains `hc`, but its disease metadata includes Crohn disease,
ulcerative colitis, and healthy controls; the full supplied cohort is retained.
Author cell-type annotations are held out so the agent must use measured marker evidence.
Donor or sample names do not authorize technical correction. The donors, rather than the cells
sampled from them, are the biological replicates for comparisons between individuals.
The default QC policy retains the supplied cells with advisory outlier flags.
Doublet scoring is opt-in and is left disabled here.

```{code-cell} ipython3
# Summarize the available study metadata before supplying scientific context.
columns = set(store.cells.columns)
metadata_facts = {}

# Count observed levels and missing values without inferring missing study details.
for name in ("organism", "tissue", "disease", "donor_id", "sample_id"):
    if name in columns:
        values = store.cells.to_pandas_dataframe([name])[name]
        counts = values.dropna().astype(str).value_counts()
        metadata_facts[name] = {
            "distinct": len(counts),
            "missing": int(values.isna().sum()),
            "levels": {str(key): int(value) for key, value in counts.head(20).items()},
        }

# Encode measured metadata as text for the study description.
import json

# Describe the biological objective and hold author annotations out of the analysis.
study = Study(
    context=(
        "Garrido-Trigo 2023 IBD and healthy-control intestinal RNA cohort from CyteBase. "
        "Describe the supplied populations with measured markers. Author labels are held out. "
        "No technical batch correction or doublet scoring is authorized. "
        "Donors are biological identities; missing capture or replication details do not "
        "block descriptive analysis. Do not make disease-comparison or causal claims.\n"
        "Observed source metadata: " + json.dumps(metadata_facts, ensure_ascii=False)
    ),
    objective="Identify defensible major cell populations and report provisional identities and uncertainty.",
    organism="Homo sapiens",
    sampleColumn="donor_id" if "donor_id" in columns else None,
    protectedColumns=[name for name in ("disease", "tissue", "sex") if name in columns],
    excludedColumns=[name for name in columns if name.lower().startswith(("skill_", "agent_"))],
)

# Release the inspection handle before the agent opens the working store.
del store
```

## Run the automated analysis

The first execution starts a fresh analysis. Re-executing this cell explicitly
resumes the same saved run; completed numerical work and accepted decisions are
reused. Required missing facts or exhausted provider repairs produce a saved
non-completed outcome rather than an invented answer.

This example supplies a memorable `run_dir`. Omitting it creates
`./agent_runs/<run-id>/` in the directory where the analysis call starts.

```{code-cell} ipython3
# Configure the provider used for structured agent decisions.
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Create the provider client only when credentials are available.
api_key = os.environ.get("AGENT_API_KEY")
model = None
if api_key:
    model = OpenAIChatModel(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        provider=OpenAIProvider(base_url="https://inference.baseten.co/v1", api_key=api_key),
    )

# Bound the numerical work to four workers and a 4 GB memory budget.
runtime = RuntimeConfig(nthreads=4, memBudget="4G")

# Resume saved work when present, or explicitly start a new analysis.
if run_dir.exists():
    run = await resume_rna_async(run_dir, model=model, runtime=runtime)
else:
    if model is None:
        raise RuntimeError("Configure AGENT_API_KEY before starting a fresh analysis")
    run = await analyze_rna_async(
        data_path, run_dir=run_dir, model=model, study=study,
        config=AnalysisConfig(assay="RNA"), runtime=runtime,
    )

# Stop if the run needs attention before accessing final numerical results.
if run.status != "completed":
    raise RuntimeError(f"Analysis returned {run.status}; inspect its saved report before continuing")

# Confirm completion and report the number of numerical pipeline invocations.
print({"status": run.status, "pipelineInvocations": len(run.pipeline_runs)})
```

## Inspect coverage and provisional identities

The native baseline and feasible HVG, PC, and neighbor probes use the full retained
cohort. Every probe is compared with its actual parent at matching resolutions.
Marker evidence is collected for at most two finalists; finalization reuses the
selected marker artifacts and adds UMAP. These are descriptive markers rather than
replicated differential-expression tests.

```{code-cell} ipython3
# Inspect which parameter alternatives were measured.
display(pd.DataFrame(run.exploration_coverage["slots"]))

# Compare provisional identities with their reported confidence and rationale.
display(pd.DataFrame(run.annotations)[["clusterId", "identity", "confidence", "rationale"]])
```

```{code-cell} ipython3
# Locate the final clusters in the saved UMAP.
display(Image(filename=str(run.run_dir / "umap_clusters.png")))

# Review marker expression supporting the proposed identities.
display(Image(filename=str(run.run_dir / "marker_dotplot.png")))
```

An `unassigned` cluster is an explicit uncertainty outcome. Review its measured
markers and QC evidence before assigning a biological name. A separated UMAP
island alone does not establish a distinct lineage. Check whether its marker combination,
quality metrics, and representation across donors support the proposed identity.

The saved execution retained all 46,700 cells and completed all four native
representations, two marker assessments, and finalization in one uninterrupted
invocation. It used seven model requests without rejected responses or pending
questions. The selected baseline used 1,000 HVGs, 21 PCs, 11 neighbors, and Leiden
resolution 0.5, producing 18 clusters with two explicitly unassigned. Selecting
baseline settings here followed the measured alternatives; it did not skip
exploration. These provisional identities are not an annotation accuracy benchmark.

## Keep the results and report

Keep both the local dataset and the agent run directory. The dataset contains the numerical
results; the run directory contains the report, figures, annotations, and decision history.
The checks below confirm that the saved result points to the selected analysis and that the
input cells and genes stayed unchanged.

```{code-cell} ipython3
# Compare the final store with its recorded input cells and genes.
import numpy as np

# Confirm that the compact result points to this exact final pipeline.
compact = run.compact_result
assert compact is not None
assert compact["finalPipelineRunId"] == run.pipeline.run_id

# Display the selected scientific settings.
selected = compact["selectedParameters"]
display(pd.Series({
    key: selected[key]
    for key in ("requestedHvgCount", "actualHvgCount", "pcaDims", "neighborsK", "resolution", "useHarmony")
}, name="Final settings").to_frame())

# Reopen the store and verify that the input cohort and gene axis are unchanged.
verification = cytearc.DataStore(str(data_path), zarr_mode="r", min_features_per_cell=-1)
assert np.array_equal(verification.cells.fetch_all("I"), input_cells)
assert np.array_equal(verification.cells.fetch_all("ids"), input_ids)
assert np.array_equal(verification.RNA.feats.fetch_all("ids"), input_features)

# Validate the saved decisions against their recorded evidence.
replayed = run.replay_decisions()
assert all(item["valid"] for item in replayed)

# Summarize the completed consistency checks.
print({
    "compactResultStored": True,
    "pipelineReferenceVerified": True,
    "inputCohortUnchanged": True,
    "decisionsReplayed": len(replayed),
})

# Write the reviewable report and show its path within the working folder.
print("Report:", run.report().relative_to(work_dir))
```

Open that report in a browser for the complete six-step account, recorded
limitations, cluster sizes, and marker evidence. Its HTML embeds the saved figures.
Copy both the dataset and external run directory to retain numerical results and
the full decision history. A store-only copy keeps the compact result, but its
external-history locator may need rebinding. Data-only repacking omits analysis
records, and a fresh CyteBase mount does not inherit this local agent result.
