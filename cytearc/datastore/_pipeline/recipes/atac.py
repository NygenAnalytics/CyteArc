"""The ATAC recipe: QC, prevalent peaks, TF-IDF, LSI, a graph, embeddings, clusters."""

from collections.abc import Mapping
from typing import Any

from ....assay import ATACassay, NormMethod, norm_tf_idf
from ..executor import COORDINATES, PlannedStage, RecipePlan, RunState
from ..fields import cell_fields, feature_fields
from ..filtering import default_filter_columns
from ..normalizers import normalizer_label
from ..settings import (
    ANN_METRIC_DEFAULT,
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
    doublet_stage,
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
from ....utils.arguments import bool_argument
from .common import (
    check_harmony_labels,
    cell_snapshots,
    check_tsne,
    harmony_config,
    modality_assay,
    recorded_modality_call,
    resolve_doublet_samples,
    resolve_embeddings,
    resolve_graph,
    resolve_harmony,
    resolve_label,
    resolve_normalization,
    resolve_run_filtering,
    resolve_snapshot_columns,
    run_reserved_fields,
)

RECIPE = "atac"
OPERATION = "pipeline.atac"
FILTER_SUFFIXES = ("nCounts", "nFeatures")


def _prevalent_peaks_stage(assay: str, settings: Mapping[str, Any]) -> PlannedStage:
    def action(state: RunState) -> Outputs:
        ref = state.store._features_prevalent_peaks(
            state.artifacts["analysis_cell_selection"],
            from_assay=assay,
            **settings,
        )
        return (("prevalent_peaks", ref),)

    return PlannedStage(
        "prevalent_peaks",
        action,
        inputs=("analysis_cell_selection",),
        outputs=("prevalent_peaks",),
    )


def _lsi_stage(settings: Mapping[str, Any]) -> PlannedStage:
    def action(state: RunState) -> Outputs:
        ref = state.store._reduction_lsi(state.artifacts["normalized"], **settings)
        state.coordinates = ref
        return (("lsi", ref),)

    return PlannedStage(
        "lsi", action, inputs=("normalized",), outputs=("lsi", COORDINATES)
    )


def _check_doublet_settings(
    assay: str, normalizer: NormMethod, ann_metric: object
) -> None:
    """Refuse doublet scoring of a run whose LSI graph it cannot score.

    Doublet scoring weights the counts of each simulated doublet by TF-IDF
    and projects them onto the LSI of the graph's cells, so the run must
    normalize those cells by TF-IDF too, and it scores cells by Euclidean
    distances, so the graph's index must measure them.
    """
    if normalizer is not norm_tf_idf:
        raise ValueError(
            f"The run normalizes assay {assay!r} with "
            f"{normalizer_label(normalizer)}, but doublet scoring weights its "
            "simulated doublets by TF-IDF and needs tf_idf; pass "
            "normalization={'method': 'tf_idf'} or doublets=False"
        )
    if ann_metric != "l2":
        raise ValueError(
            f"ann_index={{'ann_metric': {ann_metric!r}}} builds the graph on "
            f"{ann_metric} distances, but doublet scoring of an ATAC assay "
            "scores cells by Euclidean distances in the LSI; pass "
            "ann_index={'ann_metric': 'l2'} or doublets=False"
        )


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
    localmap: object,
    tsne: object,
    leiden: object,
    paris: object,
    membership_strength: object,
    doublets: object,
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
    membership_strength = bool_argument(membership_strength, "membership_strength")
    markers = bool_argument(markers, "markers")
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
    normalization_settings = resolve_normalization(store, assay_name, normalization)
    # stage_settings checked the settings, so skip_first is a bool.
    _lsi, lsi_settings = stage_settings("lsi", lsi, optional=False)
    skip_first = lsi_settings.get("skip_first", LSI_SKIP_FIRST_DEFAULT)
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
    embeddings = resolve_embeddings(umap=umap, localmap=localmap, tsne=tsne)
    partitions, selected, leiden_settings = resolve_leiden(leiden)
    paris_enabled, paris_settings = stage_settings("paris", paris, optional=True)
    doublets_enabled, doublet_settings = stage_settings(
        "doublets", doublets, optional=True
    )
    # Doublets and markers both read the saved clustering.
    clustered = [
        name
        for name, enabled in (("doublets", doublets_enabled), ("markers", markers))
        if enabled
    ]
    if not partitions and clustered:
        raise ValueError(
            f"{' and '.join(clustered)} require at least one Leiden candidate"
        )
    if not partitions and membership_strength:
        raise ValueError("membership_strength requires at least one Leiden candidate")
    if doublets_enabled:
        _check_doublet_settings(
            assay_name,
            normalization_settings.normalizer,
            graph.ann_index.get("ann_metric", ANN_METRIC_DEFAULT),
        )
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=run_reserved_fields(
            embeddings, partitions, extra=("prevalent_peaks",)
        ),
    )
    doublet_samples = (
        resolve_doublet_samples(
            store, doublet_settings, harmony_columns, reduction="LSI"
        )
        if doublets_enabled
        else ()
    )
    # The settings are checked, so the run's cells are checked last.
    source = resolve_cells(store, cells, assays=(assay_name,), operation=OPERATION)
    check_harmony_labels(store, source, harmony_columns)
    filtering_config = resolve_run_filtering(
        store,
        filtering,
        cells=source,
        assay=assay_name,
        default_attrs=default_filter_columns(store, assay_name, FILTER_SUFFIXES),
        operation=OPERATION,
    )
    check_tsne(embeddings)

    first_tail, tsne_tail = clustering_tail(
        partitions=partitions,
        selected=selected,
        leiden=leiden_settings,
        paris=(paris_enabled, paris_settings),
        umap=(embeddings.umap, embeddings.umap_settings),
        localmap=(embeddings.localmap, embeddings.localmap_settings),
        tsne=(embeddings.tsne, embeddings.tsne_settings),
        initialization=graph.initialization,
        membership_strength=membership_strength,
    )
    cell_columns, extra_snapshots = cell_snapshots(
        filtering_config,
        harmony_columns=harmony_columns,
        snapshot_columns=snapshots,
        doublet_samples=doublet_samples,
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
        normalization_stage("prevalent_peaks", normalization_settings.flags),
        _lsi_stage(lsi_settings),
        harmony_stage(harmony_columns, harmony_settings),
        *graph_stages(
            ann_index=graph.ann_index,
            neighbors=graph.neighbors,
            connectivity=graph.connectivity,
        ),
        *first_tail,
        *tsne_tail,
        doublet_stage(doublets_enabled, doublet_settings, doublet_samples),
        markers_stage(markers),
    )
    config = {
        "cells": source.to_config(),
        "snapshotColumns": list(snapshots),
        "species": resolved_species,
        "doubletSampleColumns": list(doublet_samples),
        "stages": {
            "filtering": filtering_config,
            "prevalent_peaks": peak_settings,
            "normalization": normalization_settings.to_config(),
            "lsi": lsi_settings,
            "harmony": harmony_config(harmony_columns, harmony_settings),
            **graph.to_config(),
            **embeddings.to_config(),
            "leiden": leiden_config(partitions, selected, leiden_settings),
            "paris": paris_settings if paris_enabled else False,
            "membership_strength": membership_strength,
            "doublets": doublet_settings if doublets_enabled else False,
            "markers": markers,
        },
    }
    output_keys = graph_output_keys(
        head=("prevalent_peaks",),
        reduction=("lsi",),
        tail=("doublets", "markers"),
    )

    def fields(state: RunState) -> tuple[Any, ...]:
        return (
            *cell_fields(
                store,
                state.artifacts,
                snapshot_columns=snapshots,
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
        normalizers={assay_name: normalization_settings.normalizer},
    )


def recorded_call(
    store: Any, assay: str, config: Mapping[str, Any]
) -> dict[str, object]:
    """Return the keywords of the ``atac`` call that recorded ``config``."""
    return {
        **recorded_modality_call(store, assay, config),
        "species": config["species"],
    }
