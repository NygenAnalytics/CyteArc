"""The ADT recipe: an antibody panel, CLR, PCA, a graph, embeddings, clusters."""

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ....assay import ADTassay
from ....metadata.membership import features_measured_by_every_source
from ....utils.arguments import integer_argument, quoted_value
from ....utils.logging import logger
from ..executor import FEATURE_NAMES, PlannedStage, RecipePlan, RunState
from ..fields import cell_fields, feature_fields
from ..filtering import default_filter_columns
from ..settings import (
    PCA_DIMS_DEFAULT,
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
    pca_stage,
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
    resolve_embeddings,
    resolve_graph,
    resolve_harmony,
    resolve_label,
    resolve_normalization,
    resolve_run_filtering,
    resolve_snapshot_columns,
    run_reserved_fields,
)

RECIPE = "adt"
OPERATION = "pipeline.adt"
# Nearly every cell detects nearly every antibody of a small panel, so a
# requested filter reads only each cell's total antibody counts by default.
FILTER_SUFFIXES = ("nCounts",)
DEFAULT_PANEL_EXCLUDE = r"(?i)isotype|control|ctrl"
"""Antibody names that the panel leaves out by default: isotype controls."""
# The most antibodies that the warning about unmeasured antibodies names.
_UNMEASURED_EXAMPLES = 5


def _excluded(names: np.ndarray, pattern: re.Pattern[str] | None) -> np.ndarray:
    if pattern is None:
        return np.zeros(len(names), dtype=bool)
    return np.fromiter(
        (pattern.search(str(name)) is not None for name in names),
        dtype=bool,
        count=len(names),
    )


def _panel_stage(
    assay: str, pattern: re.Pattern[str] | None, measured: np.ndarray
) -> PlannedStage:
    """Return the stage that keeps the antibodies of ``measured`` that ``pattern`` does not match."""

    def action(state: RunState) -> Outputs:
        assert state.feature_names is not None
        ref = state.store._features_snapshot(
            from_assay=assay,
            mask=~_excluded(state.feature_names, pattern) & measured,
        )
        return (("panel", ref),)

    return PlannedStage("panel", action, inputs=(FEATURE_NAMES,), outputs=("panel",))


def _resolve_panel(
    store: Any,
    assay: str,
    value: object,
) -> tuple[re.Pattern[str] | None, np.ndarray, dict[str, Any], int]:
    """Return the exclusion pattern, the antibodies that every source measured, the recorded settings, and the panel size.

    A merged store records which antibodies each of its sources measured,
    and the panel leaves out those that not every source measured, whose
    zero counts in the other sources' cells are no measurement, with a
    warning that names them. The run records them as ``unmeasured``.
    """
    _enabled, settings = stage_settings("panel", value, optional=False)
    raw = settings.get("exclude", DEFAULT_PANEL_EXCLUDE)
    if raw is not None and (not isinstance(raw, str) or not raw):
        raise TypeError("panel['exclude'] must be a non-empty pattern or None")
    try:
        pattern = None if raw is None else re.compile(raw)
    except re.error as error:
        raise ValueError(f"panel['exclude'] is not a valid pattern: {error}") from error
    features = store._get_assay(assay).feats
    names = np.asarray(features.fetch_all("names")).astype(str)
    excluded = _excluded(names, pattern)
    measured = features_measured_by_every_source(features)
    unmeasured = ~measured & ~excluded
    kept = int(np.count_nonzero(~excluded & measured))
    if kept < 2:
        unmeasured_words = (
            f" and the {int(unmeasured.sum())} that not every source of the "
            "merged store measured"
            if unmeasured.any()
            else ""
        )
        raise ValueError(
            f"The ADT panel keeps {kept} of {len(names)} antibodies after "
            f"panel['exclude']={raw!r}{unmeasured_words}; it needs at least two"
        )
    recorded: dict[str, Any] = {"exclude": raw, "excluded": names[excluded].tolist()}
    if unmeasured.any():
        left_out = names[unmeasured].tolist()
        listed = ", ".join(map(quoted_value, left_out[:_UNMEASURED_EXAMPLES]))
        more = len(left_out) - _UNMEASURED_EXAMPLES
        logger.warning(
            "The ADT panel leaves out the antibodies that not every source of the "
            "merged store measured, whose zero counts in the cells of the other "
            f"sources are no measurement: {listed}"
            + (f" and {more} more" if more > 0 else "")
            + ". The run records them as the panel's 'unmeasured'."
        )
        recorded["unmeasured"] = left_out
    return pattern, measured, recorded, kept


def resolve_adt(
    store: Any,
    *,
    assay: object,
    label: object,
    cells: object,
    filtering: object,
    panel: object,
    normalization: object,
    pca: object,
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
    markers: object,
    snapshot_columns: object,
    species: object,
) -> RecipePlan:
    """Validate an ADT recipe call against ``store`` before anything is written."""
    assay_name = modality_assay(
        store,
        assay,
        recipe=RECIPE,
        modality="ADT",
        classes=(ADTassay,),
        types=frozenset({"ADT"}),
    )
    resolved_label = resolve_label(label)
    resolved_species = resolve_species(species)
    membership_strength = bool_argument(membership_strength, "membership_strength")
    markers = bool_argument(markers, "markers")
    pattern, measured, panel_config, panel_size = _resolve_panel(
        store, assay_name, panel
    )
    normalization_settings = resolve_normalization(store, assay_name, normalization)
    _pca, pca_settings = stage_settings("pca", pca, optional=False)
    # A panel holds few antibodies, so PCA keeps one dimension fewer than the
    # panel unless the call sets dims.
    pca_dims = integer_argument(
        pca_settings.pop("dims", min(PCA_DIMS_DEFAULT, panel_size - 1)),
        "pca dims",
        minimum=1,
    )
    if pca_dims >= panel_size:
        raise ValueError(
            f"pca dims ({pca_dims}) must be below the {panel_size} antibodies of "
            "the panel: PCA needs at least dims + 1 features"
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
    if not partitions and markers:
        raise ValueError("markers require at least one Leiden candidate")
    if not partitions and membership_strength:
        raise ValueError("membership_strength requires at least one Leiden candidate")
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=run_reserved_fields(embeddings, partitions, extra=("panel",)),
    )
    # The settings are checked, so the run's cells are checked last.
    source = resolve_cells(store, cells, assays=(assay_name,), operation=OPERATION)
    check_harmony_labels(store, source, harmony_columns)
    # ADT runs usually share the QC-passing cells of an RNA run, so they
    # filter only when asked.
    filtering_config = resolve_run_filtering(
        store,
        filtering,
        cells=source,
        assay=assay_name,
        default_attrs=default_filter_columns(store, assay_name, FILTER_SUFFIXES),
        operation=OPERATION,
        default_on=False,
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
        _panel_stage(assay_name, pattern, measured),
        normalization_stage("panel", normalization_settings.flags),
        pca_stage(pca_dims, pca_settings),
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
            "panel": panel_config,
            "normalization": normalization_settings.to_config(),
            "pca": {"dims": pca_dims, **pca_settings},
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
        head=("panel",),
        reduction=("pca",),
        tail=("markers",),
    )

    def fields(state: RunState) -> tuple[Any, ...]:
        return (
            *cell_fields(
                store,
                state.artifacts,
                snapshot_columns=snapshots,
            ),
            *feature_fields(store, assay_name, state.artifacts, selection_key="panel"),
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
    """Return the keywords of the ``adt`` call that recorded ``config``.

    The run records the antibodies that its panel excluded beside the
    ``exclude`` pattern, which the keyword takes alone.
    """
    call = recorded_modality_call(store, assay, config)
    return {
        **call,
        "panel": {"exclude": config["stages"]["panel"]["exclude"]},
        "species": config["species"],
    }
