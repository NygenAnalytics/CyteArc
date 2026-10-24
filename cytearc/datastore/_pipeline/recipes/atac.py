"""The ATAC recipe: QC, prevalent peaks, TF-IDF, LSI, a graph, embeddings, clusters."""

from collections.abc import Mapping
from typing import Any

from ....assay import ATACassay
from ..executor import PlannedStage, RecipePlan, RunState
from ..fields import cell_fields, feature_fields
from ..filtering import default_filter_columns
from ..settings import (
    LSI_DIMS_DEFAULT,
    LSI_SKIP_FIRST_DEFAULT,
    PEAKS_TOP_N_DEFAULT,
    leiden_config,
    resolve_leiden,
    resolve_species,
    stage_settings,
)
from ..stages import (
    Outputs,
    clustering_tail,
    filtering_stage,
    graph_output_keys,
    graph_stages,
    harmony_stage,
    input_snapshot_stage,
    markers_stage,
    normalization_stage,
    resolve_cells,
)
from ..summaries import summarize_stage
from .common import (
    cell_snapshots,
    check_tsne,
    harmony_config,
    modality_assay,
    resolve_embeddings,
    resolve_graph,
    resolve_harmony,
    require_bool,
    resolve_label,
    resolve_run_filtering,
    resolve_snapshot_columns,
    run_reserved_fields,
)

RECIPE = "atac"
OPERATION = "pipeline.atac"
_FILTER_SUFFIXES = ("nCounts", "nFeatures")


def _prevalent_peaks_stage(assay: str, settings: Mapping[str, Any]) -> PlannedStage:
    def action(state: RunState) -> Outputs:
        ref = state.store._features_prevalent_peaks(
            state.artifacts["analysis_cell_selection"],
            from_assay=assay,
            **settings,
        )
        return (("prevalent_peaks", ref),)

    return PlannedStage("prevalent_peaks", action)


def _lsi_stage(settings: Mapping[str, Any]) -> PlannedStage:
    def action(state: RunState) -> Outputs:
        ref = state.store._reduction_lsi(state.artifacts["normalized"], **settings)
        state.coordinates = ref
        return (("lsi", ref),)

    return PlannedStage("lsi", action)


def resolve_atac(
    store: Any,
    *,
    assay: object,
    label: object,
    cells: object,
    filtering: object,
    prevalent_peaks: object,
    normalization: object,
    lsi: object,
    harmony: object,
    ann_index: object,
    neighbors: object,
    connectivity: object,
    embedding_initialization: object,
    umap: object,
    tsne: object,
    leiden: object,
    paris: object,
    membership_strength: object,
    markers: object,
    snapshot_columns: object,
    species: object,
) -> RecipePlan:
    """Validate an ATAC recipe call against ``store`` before anything is written."""
    assay_name = modality_assay(
        store, assay, recipe=RECIPE, modality="ATAC", classes=(ATACassay,)
    )
    resolved_label = resolve_label(label)
    resolved_species = resolve_species(species)
    membership_strength = require_bool(membership_strength, "membership_strength")
    markers = require_bool(markers, "markers")
    _peaks, peak_settings = stage_settings(
        "prevalent_peaks", prevalent_peaks, optional=False
    )
    # A small assay keeps every peak but one unless the call sets top_n, as
    # prevalent peaks are fewer than the assay's.
    n_peaks = int(store._get_assay(assay_name).feats.N)
    top_n = peak_settings.setdefault("top_n", min(PEAKS_TOP_N_DEFAULT, n_peaks - 1))
    if not 0 < top_n < n_peaks:
        raise ValueError(
            f"prevalent_peaks top_n ({top_n}) must be at least 1 and below the "
            f"{n_peaks} peaks of assay {assay_name!r}"
        )
    _norm, normalization_settings = stage_settings(
        "normalization", normalization, optional=False
    )
    _lsi, lsi_settings = stage_settings("lsi", lsi, optional=False)
    skip_first = lsi_settings.get("skip_first", LSI_SKIP_FIRST_DEFAULT)
    if not isinstance(skip_first, bool):
        raise TypeError("lsi skip_first must be a boolean")
    lsi_rank = lsi_settings.get("dims", LSI_DIMS_DEFAULT) + int(skip_first)
    if lsi_rank > top_n:
        raise ValueError(
            f"lsi dims, with the first component that skip_first leaves out, "
            f"need {lsi_rank} of the prevalent peaks, but prevalent_peaks "
            f"top_n is {top_n}"
        )
    harmony_columns, harmony_settings = resolve_harmony(store, harmony)
    graph = resolve_graph(
        ann_index=ann_index,
        neighbors=neighbors,
        connectivity=connectivity,
        embedding_initialization=embedding_initialization,
    )
    embeddings = resolve_embeddings(umap=umap, tsne=tsne)
    partitions, selected, leiden_settings = resolve_leiden(leiden)
    paris_enabled, paris_settings = stage_settings("paris", paris, optional=True)
    if not partitions and markers:
        raise ValueError("markers require at least one Leiden candidate")
    if not partitions and membership_strength:
        raise ValueError("membership_strength requires at least one Leiden candidate")
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=run_reserved_fields(
            embeddings, partitions, extra=("prevalent_peaks",)
        ),
    )
    # The settings are checked, so the run's cells are checked last.
    source = resolve_cells(store, cells, assays=(assay_name,), operation=OPERATION)
    filtering_config = resolve_run_filtering(
        store,
        filtering,
        cells=source,
        assay=assay_name,
        default_attrs=default_filter_columns(store, assay_name, _FILTER_SUFFIXES),
        operation=OPERATION,
    )
    check_tsne(embeddings)

    first_tail, tsne_tail = clustering_tail(
        partitions=partitions,
        selected=selected,
        leiden=leiden_settings,
        paris=(paris_enabled, paris_settings),
        umap=(embeddings.umap, embeddings.umap_settings),
        tsne=(embeddings.tsne, embeddings.tsne_settings),
        initialization=graph.initialization,
        membership_strength=membership_strength,
    )
    cell_columns, extra_snapshots = cell_snapshots(
        filtering_config, harmony_columns=harmony_columns, snapshot_columns=snapshots
    )
    stages = (
        input_snapshot_stage(
            source,
            assay=assay_name,
            cell_snapshot_columns=cell_columns,
            extra_snapshots=extra_snapshots,
        ),
        filtering_stage(filtering_config, source),
        _prevalent_peaks_stage(assay_name, peak_settings),
        normalization_stage("prevalent_peaks", normalization_settings),
        _lsi_stage(lsi_settings),
        harmony_stage(harmony_columns, harmony_settings),
        *graph_stages(
            ann_index=graph.ann_index,
            neighbors=graph.neighbors,
            connectivity=graph.connectivity,
        ),
        *first_tail,
        *tsne_tail,
        markers_stage(markers),
    )
    config = {
        "cells": source.to_config(),
        "snapshotColumns": list(snapshots),
        "species": resolved_species,
        "stages": {
            "filtering": filtering_config,
            "prevalent_peaks": peak_settings,
            "normalization": normalization_settings,
            "lsi": lsi_settings,
            "harmony": harmony_config(harmony_columns, harmony_settings),
            **graph.to_config(),
            **embeddings.to_config(),
            "leiden": leiden_config(partitions, selected, leiden_settings),
            "paris": paris_settings if paris_enabled else False,
            "membership_strength": membership_strength,
            "markers": markers,
        },
    }
    output_keys = graph_output_keys(
        head=("prevalent_peaks",),
        reduction=("lsi",),
        partitions=partitions,
        tail=("markers",),
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
            *feature_fields(
                store,
                assay_name,
                state.artifacts,
                selection_key="prevalent_peaks",
            ),
        )

    return RecipePlan(
        recipe=RECIPE,
        assay=assay_name,
        label=resolved_label,
        config=config,
        stages=stages,
        output_keys=output_keys,
        fields=fields,
        start=lambda store: RunState(store=store, assay=store._get_assay(assay_name)),
        summarize=lambda stage, outputs: summarize_stage(store, stage, outputs),
    )
