"""Stages that several recipes share, from the input snapshot to markers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...metadata.membership import require_measured_cells
from ...storage.artifacts import ArtifactRef, artifact_group
from ...storage.pipeline_runs import load_pipeline_run_record
from ...storage.selections import (
    resolve_stored_selection_artifact,
    snapshot_run_metadata,
    validate_stored_selection_integrity,
)
from ...storage.types import as_zarr_array
from ...utils.logging import logger
from ..pipeline_run import PipelineRun
from .cluster_selection import cluster_label_values, run_cluster_selection
from .executor import PlannedStage, RunState
from .filtering import filter_pipeline_selection

type Outputs = tuple[tuple[str, ArtifactRef], ...]


@dataclass(frozen=True, slots=True)
class CellSource:
    """The cells a run analyzes: a boolean column, a selection, or a run's cells.

    Attributes:
        column: The boolean cell column, or None.
        selection: The datastore cell selection, or None for a column.
        run: The run whose analysis cells these are, as ``(run_id, label)``.
    """

    column: str | None
    selection: ArtifactRef | None
    run: tuple[str, str | None] | None = None

    @property
    def source_column(self) -> str:
        """The execution option that names where a selection's values came from."""
        return "artifact" if self.column is None else self.column

    def to_config(self) -> dict[str, Any]:
        if self.column is not None:
            return {"column": self.column}
        assert self.selection is not None
        if self.run is not None:
            run_id, label = self.run
            return {
                "run": {"runId": run_id, "label": label},
                "selection": self.selection.to_dict(),
            }
        return {"selection": self.selection.to_dict()}


def resolve_cells(
    store: Any,
    cells: object,
    *,
    assays: Sequence[str],
    operation: str,
) -> CellSource:
    """Validate the ``cells`` keyword and refuse cells an assay did not measure.

    A run reads each of ``assays`` over these cells, so it refuses cells that
    one did not measure before any stage runs; it never narrows them.
    """
    if isinstance(cells, str):
        if not cells:
            raise TypeError("cells must name a non-empty boolean cell column")
        if cells not in store.cells.columns:
            raise KeyError(f"Cell selection column {cells!r} was not found")
        if np.dtype(store.cells.get_dtype(cells)) != np.dtype(bool):
            raise TypeError("cells must name a boolean metadata column")
        for assay in assays:
            require_measured_cells(
                store.cells,
                assay,
                cells,
                operation=operation,
                remedy="cells",
            )
        return CellSource(column=cells, selection=None)
    run: tuple[str, str | None] | None = None
    if isinstance(cells, PipelineRun):
        if cells.status != "completed":
            raise ValueError(
                f"cells must be a completed pipeline run; run {cells.run_id} is "
                f"{cells.status!r}"
            )
        try:
            load_pipeline_run_record(store.zw, cells.run_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"cells must be a pipeline run of this datastore; run "
                f"{cells.run_id} is not recorded in it"
            ) from error
        selection = cells["analysis_cell_selection"]
        run = (cells.run_id, cells.label)
    elif isinstance(cells, ArtifactRef):
        selection = cells
    else:
        raise TypeError(
            "cells must be a boolean cell column name, a cell-selection "
            "ArtifactRef, or a completed PipelineRun"
        )
    if selection.kind != "cell_selection" or selection.scope != "datastore":
        raise ValueError("cells must be a datastore cell-selection artifact")
    validate_stored_selection_integrity(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    for assay in assays:
        store._require_measured_cells(
            assay,
            selection,
            operation=operation,
            remedy="cells",
        )
    return CellSource(column=None, selection=selection, run=run)


def frozen_feature_names(store: Any, feature_snapshot: ArtifactRef) -> np.ndarray:
    return np.asarray(
        as_zarr_array(
            artifact_group(store.zw, feature_snapshot)["names"],
            name="names",
        )[:]
    )


def input_snapshot_stage(
    cells: CellSource,
    *,
    assay: str,
    cell_snapshot_columns: Sequence[str],
    extra_snapshots: Mapping[str, Sequence[str]],
) -> PlannedStage:
    """Freeze the run's input cells, cell and feature metadata, and features.

    ``extra_snapshots`` maps an output key to cell columns frozen apart from
    the cell snapshot, so they leave the identities of the stages that read
    the cell snapshot alone.
    """
    extras = dict(extra_snapshots)

    def action(state: RunState) -> Outputs:
        store = state.store
        outputs: list[tuple[str, ArtifactRef]] = []
        if cells.column is not None:
            input_selection = resolve_stored_selection_artifact(
                store.zw,
                table_path="cellData",
                id_column="ids",
                source_column=cells.column,
                scope="datastore",
                kind="cell_selection",
                operation="snapshot_pipeline_input_selection",
                parameters={"assay": assay},
                inputs={},
            )
            outputs.append(("input_cell_selection", input_selection))
        else:
            # A given selection is the run's input as it is, so runs that
            # share it, such as the modalities of a WNN run, share its ref.
            assert cells.selection is not None
            state.artifacts["input_cell_selection"] = cells.selection
        cell_snapshot = snapshot_run_metadata(
            store.zw,
            table_path="cellData",
            id_column="ids",
            columns=tuple(cell_snapshot_columns),
            axis="cell",
        )
        feature_snapshot = snapshot_run_metadata(
            store.zw,
            table_path=f"{assay}/featureData",
            id_column="ids",
            columns=("names",),
            axis="feature",
            assay=assay,
        )
        all_features = store._features_universe(from_assay=assay)
        state.feature_names = frozen_feature_names(store, feature_snapshot)
        outputs.extend(
            (
                ("cell_snapshot", cell_snapshot),
                ("feature_snapshot", feature_snapshot),
                ("feature_universe", all_features),
            )
        )
        for key, columns in extras.items():
            outputs.append(
                (
                    key,
                    snapshot_run_metadata(
                        store.zw,
                        table_path="cellData",
                        id_column="ids",
                        columns=tuple(columns),
                        axis="cell",
                    ),
                )
            )
        return tuple(outputs)

    return PlannedStage("input_snapshot", action)


def filtering_stage(
    config: Mapping[str, Any],
    cells: CellSource,
    *,
    input_key: str = "input_cell_selection",
    hashtags: bool = False,
    called: bool = False,
) -> PlannedStage:
    """Filter the cells of ``input_key`` into the run's analysis cells.

    ``hashtags`` computes MAD bounds within each hashtag of the run's
    ``hashing`` output, and ``called`` says the run called these cells.
    """

    def skip(state: RunState) -> None:
        state.artifacts["analysis_cell_selection"] = state.artifacts[input_key]

    if not config["enabled"]:
        return PlannedStage("filtering", None, on_skip=skip)

    def action(state: RunState) -> Outputs:
        source = state.artifacts[input_key]
        ref = filter_pipeline_selection(
            state.store,
            config=config,
            input_selection=source,
            cell_snapshot=state.artifacts["cell_snapshot"],
            source_column=cells.source_column,
            hashing=state.artifacts["hashing"] if hashtags else None,
            called=called,
        )
        if ref == source:
            state.artifacts["analysis_cell_selection"] = ref
            return ()
        return (("analysis_cell_selection", ref),)

    return PlannedStage("filtering", action)


def hvg_stage(settings: Mapping[str, Any]) -> PlannedStage:
    """Select the highly variable genes of the run's analysis cells."""

    def action(state: RunState) -> Outputs:
        ref = state.store._select_hvgs_artifact(
            assay=state.assay,
            cell_selection=state.artifacts["analysis_cell_selection"],
            feature_names=state.feature_names,
            feature_snapshot=state.artifacts["feature_snapshot"],
            show_plot=False,
            **settings,
        )
        return (("highly_variable_features", ref),)

    return PlannedStage("highly_variable_features", action)


def normalization_stage(features_key: str, settings: Mapping[str, Any]) -> PlannedStage:
    def action(state: RunState) -> Outputs:
        ref = state.store._features_normalize(
            state.artifacts["analysis_cell_selection"],
            state.artifacts[features_key],
            **settings,
        )
        state.coordinates = ref
        return (("normalized", ref),)

    return PlannedStage("normalization", action)


def pca_stage(dims: int, settings: Mapping[str, Any]) -> PlannedStage:
    """Reduce the normalized values; ``dims=0`` builds the graph on them instead."""
    if not dims:
        return PlannedStage("pca", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._reduction_pca(
            state.artifacts["normalized"],
            dims=dims,
            **settings,
        )
        state.coordinates = ref
        return (("pca", ref),)

    return PlannedStage("pca", action)


def harmony_stage(
    batch_columns: Sequence[str],
    settings: Mapping[str, Any],
) -> PlannedStage:
    if not batch_columns:
        return PlannedStage("harmony", None)

    def action(state: RunState) -> Outputs:
        assert state.coordinates is not None
        ref = state.store._run_harmony_artifact(
            state.coordinates,
            state.artifacts["harmony_snapshot"],
            list(batch_columns),
            **settings,
        )
        state.coordinates = ref
        return (("harmony", ref),)

    return PlannedStage("harmony", action)


def graph_stages(
    *,
    ann_index: Mapping[str, Any],
    neighbors: Mapping[str, Any],
    connectivity: Mapping[str, Any],
) -> tuple[PlannedStage, ...]:
    def ann_action(state: RunState) -> Outputs:
        assert state.coordinates is not None
        ref = state.store._graph_ann_index(state.coordinates, **ann_index)
        return (("ann_index", ref),)

    def neighbors_action(state: RunState) -> Outputs:
        ref = state.store._graph_neighbors(
            state.artifacts["ann_index"],
            coordinates=state.coordinates,
            **neighbors,
        )
        return (("neighbors", ref),)

    def connectivity_action(state: RunState) -> Outputs:
        ref = state.store._graph_connectivity(
            state.artifacts["neighbors"], **connectivity
        )
        return (("connectivity_map", ref),)

    return (
        PlannedStage("ann_index", ann_action),
        PlannedStage("neighbors", neighbors_action),
        PlannedStage("connectivity", connectivity_action),
    )


def initialization_stage(
    enabled: bool,
    settings: Mapping[str, Any],
) -> PlannedStage:
    if not enabled:
        return PlannedStage("embedding_initialization", None)

    def action(state: RunState) -> Outputs:
        assert state.coordinates is not None
        ref = state.store._embeddings_initialization(state.coordinates, **settings)
        return (("embedding_initialization", ref),)

    return PlannedStage("embedding_initialization", action)


def umap_stage(
    enabled: bool,
    settings: Mapping[str, Any],
    *,
    graph_key: str = "connectivity_map",
) -> PlannedStage:
    if not enabled:
        return PlannedStage("umap", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._run_umap_artifact(
            state.artifacts[graph_key],
            state.artifacts["embedding_initialization"],
            **settings,
        )
        return (("umap", ref),)

    return PlannedStage("umap", action)


def leiden_stages(
    partitions: Sequence[tuple[str, float]],
    settings: Mapping[str, Any],
    *,
    graph_key: str = "connectivity_map",
) -> tuple[PlannedStage, ...]:
    stages = []
    for key, resolution in partitions:
        output_key = f"leiden_{key}"

        def action(
            state: RunState,
            output_key: str = output_key,
            resolution: float = resolution,
        ) -> Outputs:
            ref = state.store._run_leiden_artifact(
                state.artifacts[graph_key],
                resolution=resolution,
                **settings,
            )
            return ((output_key, ref),)

        stages.append(PlannedStage(output_key, action))
    return tuple(stages)


def paris_stage(
    enabled: bool,
    settings: Mapping[str, Any],
    *,
    graph_key: str = "connectivity_map",
) -> PlannedStage:
    if not enabled:
        return PlannedStage("paris", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._run_paris_artifact(state.artifacts[graph_key], **settings)
        return (("paris", ref),)

    return PlannedStage("paris", action)


def cluster_selection_stage(
    partitions: Sequence[tuple[str, float]],
    selected: str | None,
) -> PlannedStage:
    """Choose the saved clustering among the Leiden candidates.

    A requested resolution is the saved clustering and the stage is skipped;
    the other partitions stay available as candidates. Otherwise silhouette
    scores rank the candidates in the coordinates that built the graph.
    """
    if selected is not None:

        def alias(state: RunState) -> None:
            state.artifacts["clusters"] = state.artifacts[f"leiden_{selected}"]

        return PlannedStage("cluster_selection", None, on_skip=alias)
    if not partitions:
        return PlannedStage("cluster_selection", None)

    def action(state: RunState) -> Outputs:
        assert state.coordinates is not None
        decision, selected_key, selected_ref = run_cluster_selection(
            state.store,
            coordinates=state.coordinates,
            connectivity_map=state.artifacts["connectivity_map"],
            cell_selection=state.artifacts["analysis_cell_selection"],
            candidates=[
                (f"leiden_{key}", state.artifacts[f"leiden_{key}"])
                for key, _resolution in partitions
            ],
        )
        # The saved clustering is the chosen candidate's own artifact.
        state.artifacts["clusters"] = selected_ref
        logger.info(f"Selected clustering candidate: {selected_key}")
        return (("cluster_selection", decision),)

    return PlannedStage("cluster_selection", action)


def membership_strength_stage(
    enabled: bool,
    *,
    graph_key: str = "connectivity_map",
) -> PlannedStage:
    if not enabled:
        return PlannedStage("membership_strength", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._clusters_membership_strength(
            state.artifacts["clusters"], state.artifacts[graph_key]
        )
        return (("membership_strength", ref),)

    return PlannedStage("membership_strength", action)


def tsne_stage(
    enabled: bool,
    settings: Mapping[str, Any],
    *,
    graph_key: str = "connectivity_map",
) -> PlannedStage:
    if not enabled:
        return PlannedStage("tsne", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._embeddings_tsne(
            state.artifacts[graph_key],
            state.artifacts["embedding_initialization"],
            verbose=False,
            **settings,
        )
        return (("tsne", ref),)

    return PlannedStage("tsne", action)


def markers_stage(enabled: bool) -> PlannedStage:
    """Rank every feature of the run's assay for each saved cluster."""
    if not enabled:
        return PlannedStage("markers", None)

    def action(state: RunState) -> Outputs:
        store = state.store
        clusters = state.artifacts["clusters"]
        ref = store._run_marker_search_artifact(
            assay=state.assay,
            cell_selection=state.artifacts["analysis_cell_selection"],
            clusters=clusters,
            cluster_values=cluster_label_values(store.zw, clusters),
            feature_selection=state.artifacts["feature_universe"],
            feature_names=state.feature_names,
            feature_snapshot=state.artifacts["feature_snapshot"],
        )
        return (("markers", ref),)

    return PlannedStage("markers", action)


def clustering_tail(
    *,
    partitions: Sequence[tuple[str, float]],
    selected: str | None,
    leiden: Mapping[str, Any],
    paris: tuple[bool, Mapping[str, Any]],
    umap: tuple[bool, Mapping[str, Any]],
    tsne: tuple[bool, Mapping[str, Any]],
    initialization: Mapping[str, Any],
    membership_strength: bool,
) -> tuple[tuple[PlannedStage, ...], tuple[PlannedStage, ...]]:
    """Return the embedding and clustering stages after the graph, in two parts.

    The first part runs from embedding initialization to membership strength,
    and the second is t-SNE; a recipe places its own stages, such as doublet
    scoring and markers, after them.
    """
    umap_enabled, umap_settings = umap
    tsne_enabled, tsne_settings = tsne
    paris_enabled, paris_settings = paris
    first = (
        initialization_stage(umap_enabled or tsne_enabled, initialization),
        umap_stage(umap_enabled, umap_settings),
        *leiden_stages(partitions, leiden),
        paris_stage(paris_enabled, paris_settings),
        cluster_selection_stage(partitions, selected),
        membership_strength_stage(membership_strength),
    )
    return first, (tsne_stage(tsne_enabled, tsne_settings),)


# The order of a single-modality run's outputs in its record; a key that the
# run did not produce is left out.
def graph_output_keys(
    *,
    head: Sequence[str],
    reduction: Sequence[str],
    partitions: Sequence[tuple[str, float]],
    tail: Sequence[str],
    qc: Sequence[str] = (),
) -> tuple[str, ...]:
    return (
        "input_cell_selection",
        *qc,
        "analysis_cell_selection",
        "feature_universe",
        *head,
        "normalized",
        *reduction,
        "harmony",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "embedding_initialization",
        "umap",
        *(f"leiden_{key}" for key, _value in partitions),
        "paris",
        "cluster_selection",
        "clusters",
        "membership_strength",
        "tsne",
        *tail,
    )
