"""The multimodal recipe: a WNN or SNN graph over completed modality runs.

The modality runs must analyze the same cells, which a run gets by passing an
earlier run as its ``cells``. The recipe integrates their graphs, embeds and
clusters the integrated graph, and ranks each modality's markers for the
joint clusters.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from ....storage.artifacts import ArtifactRef, inspect_artifact
from ....storage.pipeline_runs import load_pipeline_run_record
from ....storage.selections import snapshot_run_metadata
from ..cluster_selection import cluster_label_values
from ..executor import PlannedStage, RecipePlan, RunState
from ..fields import cell_fields, feature_fields
from ..settings import leiden_config, resolve_leiden, stage_settings
from ..stages import (
    Outputs,
    cluster_selection_stage,
    frozen_feature_names,
    initialization_stage,
    leiden_stages,
    membership_strength_stage,
    paris_stage,
    tsne_stage,
    umap_stage,
)
from ..summaries import summarize_stage
from ...pipeline_run import PipelineRun
from .common import (
    check_tsne,
    resolve_embeddings,
    require_bool,
    resolve_label,
    resolve_snapshot_columns,
    run_reserved_fields,
)

RECIPE = "multimodal"
_GRAPH = "integrated_graph"
# WNN weighs each modality's neighbors, and SNN shares each one's graph edges.
_SOURCE_OUTPUT = {"wnn": "neighbors", "snn": "connectivity_map"}


def _coordinates(store: Any, neighbors: ArtifactRef) -> ArtifactRef:
    """Return the coordinates that built a run's neighbour graph."""
    raw = (inspect_artifact(store.zw, neighbors).inputs or {}).get("coordinates")
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Neighbors artifact {neighbors.artifact_id} has no coordinates"
        )
    return ArtifactRef.from_dict(dict(raw))


def _resolve_runs(
    store: Any,
    runs: object,
    method: str,
) -> tuple[tuple[PipelineRun, ...], ArtifactRef]:
    """Validate the source runs and return them with their shared cells."""
    if isinstance(runs, PipelineRun) or not isinstance(runs, Sequence):
        raise TypeError("runs must be a sequence of completed pipeline runs")
    resolved = tuple(runs)
    if len(resolved) < 2:
        raise ValueError("A multimodal run integrates at least two modality runs")
    if any(not isinstance(run, PipelineRun) for run in resolved):
        raise TypeError("runs must contain PipelineRun values")
    for run in resolved:
        if run.status != "completed":
            raise ValueError(
                f"Run {run.run_id} is {run.status!r}; integrate completed runs only"
            )
        try:
            load_pipeline_run_record(store.zw, run.run_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Integrate runs of this datastore; run {run.run_id} is not "
                "recorded in it"
            ) from error
        missing = [
            key
            for key in ("analysis_cell_selection", _SOURCE_OUTPUT[method])
            if key not in run
        ]
        if missing:
            raise ValueError(
                f"Run {run.run_id} ({run.recipe}) has no {missing!r} output to "
                "integrate"
            )
    assays = [run.assay for run in resolved]
    if len(assays) != len(set(assays)):
        raise ValueError(f"Integrate one run per assay, not {assays!r}")
    selection = resolved[0]["analysis_cell_selection"]
    for run in resolved[1:]:
        if run["analysis_cell_selection"] != selection:
            raise ValueError(
                f"Run {run.run_id} ({run.assay}) analyzes another cell selection "
                f"than run {resolved[0].run_id} ({resolved[0].assay}). Integrated "
                "modalities share one cell-selection artifact, and two runs make "
                "two selections even of the same cells, as two runs over "
                "cells='I' do: run each modality after the first with "
                "cells=<first run>, such as ds.pipeline.adt(cells=rna_run)."
            )
    if method == "snn":
        # SNN shares edges between graphs of one neighbor count, which the
        # source runs fixed; the integration stage would fail on them.
        counts = {
            run.assay: store._get_graph_ncells_k(
                store._graph_location(run["connectivity_map"])
            )[1]
            for run in resolved
        }
        if len(set(counts.values())) > 1:
            raise ValueError(
                "SNN integration needs graphs with the same number of neighbors "
                f"per cell, but the runs have {counts!r}: run each modality with "
                "the same neighbors={'k': ...}, or integrate with method='wnn', "
                "which keeps the smallest k"
            )
    if method == "wnn":
        for run in resolved:
            coordinates = _coordinates(store, run["neighbors"])
            if coordinates.kind not in {"reduction", "batch_correction"}:
                raise ValueError(
                    f"WNN weighs each modality's reduced coordinates, but run "
                    f"{run.run_id} ({run.assay}) built its graph on "
                    f"{coordinates.kind} values; run it with a PCA or LSI reduction"
                )
    return resolved, selection


def _input_stage(
    runs: Sequence[PipelineRun],
    selection: ArtifactRef,
    snapshot_columns: Sequence[str],
) -> PlannedStage:
    """Freeze the cell metadata and each modality's features for the joint run."""

    def action(state: RunState) -> Outputs:
        store = state.store
        # The modality runs share this selection, which the joint run keeps.
        state.artifacts["input_cell_selection"] = selection
        state.artifacts["analysis_cell_selection"] = selection
        snapshots: dict[str, tuple[str, ...]] = {"cell_snapshot": ("names",)}
        if snapshot_columns:
            snapshots["column_snapshot"] = tuple(snapshot_columns)
        outputs: list[tuple[str, ArtifactRef]] = [
            (
                key,
                snapshot_run_metadata(
                    store.zw,
                    table_path="cellData",
                    id_column="ids",
                    columns=columns,
                    axis="cell",
                ),
            )
            for key, columns in snapshots.items()
        ]
        modalities: dict[str, dict[str, Any]] = {}
        for run in runs:
            assay = run.assay
            feature_snapshot = snapshot_run_metadata(
                store.zw,
                table_path=f"{assay}/featureData",
                id_column="ids",
                columns=("names",),
                axis="feature",
                assay=assay,
            )
            universe = store._features_universe(from_assay=assay)
            modalities[assay] = {
                "feature_snapshot": feature_snapshot,
                "feature_universe": universe,
                "feature_names": frozen_feature_names(store, feature_snapshot),
            }
            outputs.extend(
                (
                    (f"feature_snapshot_{assay}", feature_snapshot),
                    (f"feature_universe_{assay}", universe),
                )
            )
        first = runs[0].assay
        state.artifacts["feature_snapshot"] = modalities[first]["feature_snapshot"]
        state.artifacts["feature_universe"] = modalities[first]["feature_universe"]
        state.extras["modalities"] = modalities
        return tuple(outputs)

    return PlannedStage("input_snapshot", action)


def _integration_stage(
    runs: Sequence[PipelineRun],
    method: str,
    settings: Mapping[str, Any],
) -> PlannedStage:
    sources = [run[_SOURCE_OUTPUT[method]] for run in runs]

    def action(state: RunState) -> Outputs:
        ref = state.store._integration_modalities(sources, method=method, **settings)
        return ((_GRAPH, ref),)

    return PlannedStage("integration", action)


def _markers_stage(enabled: bool, assays: Sequence[str]) -> PlannedStage:
    """Rank each modality's features for the joint clusters."""
    if not enabled:
        return PlannedStage("markers", None)

    def action(state: RunState) -> Outputs:
        store = state.store
        clusters = state.artifacts["clusters"]
        values = cluster_label_values(store.zw, clusters)
        outputs = []
        for assay in assays:
            modality = state.extras["modalities"][assay]
            ref = store._run_marker_search_artifact(
                assay=store._get_assay(assay),
                cell_selection=state.artifacts["analysis_cell_selection"],
                clusters=clusters,
                cluster_values=values,
                feature_selection=modality["feature_universe"],
                feature_names=modality["feature_names"],
                feature_snapshot=modality["feature_snapshot"],
            )
            outputs.append((f"markers_{assay}", ref))
        return tuple(outputs)

    return PlannedStage("markers", action)


def resolve_multimodal(
    store: Any,
    runs: object,
    *,
    label: object,
    method: object,
    integration: object,
    embedding_initialization: object,
    umap: object,
    tsne: object,
    leiden: object,
    paris: object,
    membership_strength: object,
    markers: object,
    snapshot_columns: object,
) -> RecipePlan:
    """Validate a multimodal recipe call against ``store``."""
    if not isinstance(method, str) or method not in _SOURCE_OUTPUT:
        raise ValueError("method must be 'wnn' or 'snn'")
    resolved_label = resolve_label(label)
    sources, selection = _resolve_runs(store, runs, method)
    membership_strength = require_bool(membership_strength, "membership_strength")
    markers = require_bool(markers, "markers")
    _integration, integration_settings = stage_settings(
        "integration", integration, optional=False
    )
    if method == "snn" and "l2_normalize" in integration_settings:
        raise ValueError("l2_normalize applies only to method='wnn'")
    _init, initialization_settings = stage_settings(
        "embedding_initialization", embedding_initialization, optional=False
    )
    embeddings = resolve_embeddings(umap=umap, tsne=tsne)
    # WNN and SNN graphs have no shared coordinates in which silhouettes could
    # compare candidates, so one resolution is the default and several need a
    # selected one.
    partitions, selected, leiden_settings = resolve_leiden(
        leiden, default_partitions=(1.0,)
    )
    if len(partitions) > 1 and selected is None:
        raise ValueError(
            "A multimodal run cannot choose among several Leiden partitions by "
            "silhouette; pass leiden={'partitions': [...], 'selected': ...}"
        )
    if len(partitions) == 1:
        selected = partitions[0][0]
    paris_enabled, paris_settings = stage_settings("paris", paris, optional=True)
    if not partitions and (markers or membership_strength):
        raise ValueError(
            "markers and membership_strength require at least one Leiden candidate"
        )
    assays = [run.assay for run in sources]
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=run_reserved_fields(
            embeddings, partitions, extra=tuple(f"weight_{assay}" for assay in assays)
        ),
    )
    first_neighbors = sources[0]["neighbors"] if "neighbors" in sources[0] else None
    if (embeddings.umap or embeddings.tsne) and first_neighbors is None:
        raise ValueError(
            f"Run {sources[0].run_id} has no neighbors output whose coordinates "
            "could initialize the joint embedding"
        )
    check_tsne(embeddings)

    def start(store: Any) -> RunState:
        state = RunState(store=store, assay=store._get_assay(assays[0]))
        if first_neighbors is not None:
            # The joint embedding starts from the first modality's layout.
            state.coordinates = _coordinates(store, first_neighbors)
        return state

    stages = (
        _input_stage(sources, selection, snapshots),
        _integration_stage(sources, method, integration_settings),
        initialization_stage(
            embeddings.umap or embeddings.tsne, initialization_settings
        ),
        umap_stage(embeddings.umap, embeddings.umap_settings, graph_key=_GRAPH),
        *leiden_stages(partitions, leiden_settings, graph_key=_GRAPH),
        paris_stage(paris_enabled, paris_settings, graph_key=_GRAPH),
        cluster_selection_stage(partitions, selected),
        membership_strength_stage(membership_strength, graph_key=_GRAPH),
        tsne_stage(embeddings.tsne, embeddings.tsne_settings, graph_key=_GRAPH),
        _markers_stage(markers, assays),
    )
    config = {
        "method": method,
        "sources": [
            {
                "runId": run.run_id,
                "label": run.label,
                "recipe": run.recipe,
                "assay": run.assay,
            }
            for run in sources
        ],
        "cells": {"selection": selection.to_dict()},
        "snapshotColumns": list(snapshots),
        "stages": {
            "integration": integration_settings,
            "embedding_initialization": initialization_settings,
            **embeddings.to_config(),
            "leiden": leiden_config(partitions, selected, leiden_settings),
            "paris": paris_settings if paris_enabled else False,
            "membership_strength": membership_strength,
            "markers": markers,
        },
    }
    output_keys = (
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        _GRAPH,
        "embedding_initialization",
        "umap",
        *(f"leiden_{key}" for key, _value in partitions),
        "paris",
        "clusters",
        "membership_strength",
        "tsne",
        *(f"markers_{assay}" for assay in assays),
    )
    leiden_keys = [key for key, _value in partitions]

    def fields(state: RunState) -> tuple[Any, ...]:
        return (
            *cell_fields(
                store,
                state.artifacts,
                snapshot_columns=snapshots,
                leiden_keys=leiden_keys,
            ),
            *feature_fields(store, assays[0], state.artifacts, selection_key=None),
        )

    return RecipePlan(
        recipe=RECIPE,
        assay=assays[0],
        label=resolved_label,
        config=config,
        stages=stages,
        output_keys=output_keys,
        fields=fields,
        start=start,
        summarize=lambda stage, outputs: summarize_stage(store, stage, outputs),
    )
