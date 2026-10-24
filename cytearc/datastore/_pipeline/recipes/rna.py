"""The RNA recipe: QC, HVGs, PCA, a graph, embeddings, clusters, doublets, markers."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ....assay import ADTassay, RNAassay
from ....metadata.rows import apply_missing_mask
from ....storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ....storage.selections import (
    read_stored_selection_indices,
    read_stored_selection_mask,
    resolve_generated_selection_artifact,
)
from ....storage.types import as_zarr_group
from ....utils.arguments import integer_argument
from ..cluster_selection import cluster_label_values
from ..executor import PlannedStage, RecipePlan, RunState
from ..fields import cell_fields, feature_fields
from ..filtering import default_filter_columns, hashtag_labels, snapshot_column_values
from ..settings import (
    HVG_TOP_N_DEFAULT,
    PCA_DIMS_DEFAULT,
    leiden_config,
    resolve_leiden,
    resolve_species,
    stage_settings,
)
from ..stages import (
    CellSource,
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

RECIPE = "rna"
OPERATION = "pipeline.rna"
# The QC columns that preparation writes for an RNA assay, which filtering
# reads when its settings name no columns.
_FILTER_SUFFIXES = ("nCounts", "nFeatures", "percentMito", "percentRibo")

# The settings that make doublet scoring possible when the saved clustering
# has one cluster. Silhouette selection never picks such a partition.
ONE_CLUSTER_DOUBLET_REMEDY = (
    "Select a Leiden partition with two or more clusters with "
    "leiden={'selected': ...}, or omit 'selected' so that cluster selection "
    "chooses one; set doublets={'heterotypic_fraction': 0} to simulate "
    "doublets from any two sampled cells; or pass doublets=False."
)


def _selection_mask(store: Any, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        store.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _derived_selection(
    store: Any,
    values: np.ndarray,
    *,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
) -> ArtifactRef:
    return resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=store.cells._get_array("ids"),
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        source_column="artifact",
    )[0]


def _require_capture_labels(
    labels: np.ndarray,
    missing: np.ndarray | None,
    active: np.ndarray,
    column: str,
) -> None:
    """Refuse capture labels that leave one of the active cells unlabeled."""
    unlabeled = np.zeros(len(active), dtype=bool) if missing is None else missing
    if labels.dtype.kind == "f":
        unlabeled = unlabeled | np.isnan(labels)
    count = int(np.count_nonzero(unlabeled & active))
    if count:
        verb = "has" if count == 1 else "have"
        raise ValueError(
            f"Cell calling per capture needs a {column!r} label for every cell, "
            f"but {count} of the run's cells {verb} none"
        )


def _check_capture_labels(store: Any, source: CellSource, column: str) -> None:
    """Refuse a capture column with unlabeled input cells before the run starts.

    The stage checks its frozen labels again, since the live column can change
    before the snapshot.
    """
    labels, missing = snapshot_column_values(
        as_zarr_group(store.zw["cellData"], name="cellData"), column
    )
    if source.column is not None:
        active = np.asarray(store.cells.fetch_all(source.column), dtype=bool)
    else:
        assert source.selection is not None
        active = _selection_mask(store, source.selection)
    _require_capture_labels(labels, missing, active, column)


def _cell_calling_stage(
    enabled: bool,
    assay: str,
    settings: Mapping[str, Any],
    capture_column: str | None,
) -> PlannedStage:
    """Call cells in raw droplets with EmptyDrops, per capture when asked.

    Each capture's ambient profile and knee are its own, so a run of several
    captures calls each one apart and analyzes the union of the called cells.
    """
    if not enabled:
        return PlannedStage("cell_calling", None)

    def action(state: RunState) -> Outputs:
        store = state.store
        source = state.artifacts["input_cell_selection"]
        if capture_column is None:
            called = store._qc_call_cells(assay, cell_selection=source, **settings)
            calls = (inspect_artifact(store.zw, called).inputs or {})["cell_calling"]
            return (
                ("cell_calling", ArtifactRef.from_dict(dict(calls))),
                ("called_cell_selection", called),
            )
        active = _selection_mask(store, source)
        # The capture labels are frozen apart from the cell snapshot, so the
        # filtering and snapshot settings leave each capture's calls alone.
        snapshot = state.artifacts["capture_snapshot"]
        labels, missing = snapshot_column_values(
            artifact_group(store.zw, snapshot), capture_column
        )
        _require_capture_labels(labels, missing, active, capture_column)
        labels = np.asarray(labels).astype(str)
        union = np.zeros(len(active), dtype=bool)
        inputs: dict[str, Any] = {"input_cell_selection": source}
        captures = sorted(set(labels[active].tolist()))
        for index, capture in enumerate(captures):
            capture_cells = _derived_selection(
                store,
                active & (labels == capture),
                operation="select_capture_cells",
                parameters={"sampleColumn": capture_column, "capture": capture},
                inputs={"input_cell_selection": source, "capture_snapshot": snapshot},
            )
            try:
                called = store._qc_call_cells(
                    assay, cell_selection=capture_cells, **settings
                )
            except ValueError as error:
                raise ValueError(
                    f"Cell calling failed for capture {capture!r} of "
                    f"{capture_column!r}: {error}"
                ) from error
            union |= _selection_mask(store, called)
            inputs[f"capture_{index}"] = called
        merged = _derived_selection(
            store,
            union,
            operation="combine_capture_cell_calls",
            parameters={"sampleColumn": capture_column, "captures": captures},
            inputs=inputs,
        )
        return (("called_cell_selection", merged),)

    return PlannedStage("cell_calling", action)


def _hashing_stage(
    enabled: bool,
    hto_assay: str | None,
    settings: Mapping[str, Any],
    input_key: str,
) -> PlannedStage:
    """Demultiplex hashtags and keep the cells with exactly one."""
    if not enabled:
        return PlannedStage("hashing", None)
    assert hto_assay is not None

    def action(state: RunState) -> Outputs:
        store = state.store
        source = state.artifacts[input_key]
        hashing = store._qc_hto_demultiplexing(source, from_assay=hto_assay, **settings)
        active = _selection_mask(store, source)
        labels = hashtag_labels(store, hashing, source, active)
        singlets = active & ~np.isin(labels, ("Negative", "Doublet"))
        if not singlets.any():
            raise ValueError("Hashing called no cell a singlet of one hashtag")
        kept = _derived_selection(
            store,
            singlets,
            operation="select_hashtag_singlets",
            parameters={"excluded": ["Negative", "Doublet"]},
            inputs={"cell_selection": source, "hashing": hashing},
        )
        return (("hashing", hashing), ("hashed_cell_selection", kept))

    return PlannedStage("hashing", action)


def _resolve_hto_assay(store: Any, settings: dict[str, Any]) -> str:
    """Return the HTO assay that hashing reads: given, or the store's only one."""
    name = settings.pop("assay", None)
    hto = [
        str(assay)
        for assay in store.assay_names
        if isinstance(store._get_assay(assay), ADTassay)
        and getattr(store._get_assay(assay), "assayType", None) == "HTO"
    ]
    if name is None:
        if len(hto) != 1:
            raise ValueError(
                "hashing needs an assay of type 'HTO'; name it with "
                f"hashing={{'assay': ...}}; HTO assays: {hto!r}"
            )
        return hto[0]
    if not isinstance(name, str) or name not in hto:
        raise ValueError(f"hashing['assay'] must name an HTO assay, one of {hto!r}")
    return name


def _resolve_doublet_samples(
    store: Any,
    settings: Mapping[str, Any],
    harmony_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the cell metadata columns whose labels split doublet scoring.

    The doublets ``sample_column`` setting names one column. Without it, a run
    with Harmony scores doublets within each batch, the cells that share every
    Harmony batch label, and a run without Harmony scores every cell together.
    """
    if "sample_column" not in settings:
        return harmony_columns
    column = settings["sample_column"]
    if column is None:
        if harmony_columns:
            raise ValueError(
                "A run with Harmony scores doublets within each sample on the "
                "uncorrected PCA, so doublets['sample_column'] cannot be None; "
                "set it to the column of each cell's capture, or omit it to use "
                "the Harmony batch columns"
            )
        return ()
    if not isinstance(column, str) or not column:
        raise TypeError("doublets['sample_column'] must be a non-empty string or None")
    if column not in store.cells.columns:
        raise KeyError(f"Doublet sample column {column!r} was not found")
    return (column,)


def _cell_cycle_stage(enabled: bool, settings: Mapping[str, Any]) -> PlannedStage:
    if not enabled:
        return PlannedStage("cell_cycle", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._run_cell_cycle_scoring_artifact(
            **settings,
            assay=state.assay,
            cell_selection=state.artifacts["analysis_cell_selection"],
            feature_names=state.feature_names,
            feature_snapshot=state.artifacts["feature_snapshot"],
        )
        return (("cell_cycle", ref),)

    return PlannedStage("cell_cycle", action)


def _hvg_stage(settings: Mapping[str, Any]) -> PlannedStage:
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


def _doublet_stage(
    enabled: bool,
    settings: Mapping[str, Any],
    sample_columns: tuple[str, ...],
) -> PlannedStage:
    if not enabled:
        return PlannedStage("doublets", None)
    # The recipe resolved the sample columns, Harmony's included.
    scoring = {key: value for key, value in settings.items() if key != "sample_column"}

    def action(state: RunState) -> Outputs:
        store = state.store
        analysis_selection = state.artifacts["analysis_cell_selection"]
        clusters = state.artifacts["clusters"]
        sample_labels = None
        if sample_columns:
            rows = read_stored_selection_indices(
                store.zw,
                analysis_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            snapshot = artifact_group(
                store.zw, state.artifacts["doublet_sample_snapshot"]
            )
            sample_labels = []
            for column in sample_columns:
                values, missing = snapshot_column_values(snapshot, column)
                sample_labels.append(
                    apply_missing_mask(
                        values[rows],
                        None if missing is None else missing[rows],
                        labels=True,
                    )
                )
        # With Harmony the graph is the corrected one, and each sample is
        # scored on the uncorrected PCA that Harmony took as input, so no
        # other graph is built.
        ref = store._run_doublet_detection_artifact(
            source_assay=state.assay,
            cell_selection=analysis_selection,
            clusters=clusters,
            cluster_values=cluster_label_values(store.zw, clusters),
            connectivity=state.artifacts["connectivity_map"],
            feature_snapshot=state.artifacts["feature_snapshot"],
            sample_column=sample_columns or None,
            sample_labels=sample_labels,
            one_cluster_remedy=ONE_CLUSTER_DOUBLET_REMEDY,
            **scoring,
        )
        return (("doublets", ref),)

    return PlannedStage("doublets", action)


def resolve_rna(
    store: Any,
    *,
    assay: object,
    label: object,
    cells: object,
    cell_calling: object,
    hashing: object,
    filtering: object,
    cell_cycle: object,
    hvg: object,
    normalization: object,
    pca: object,
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
    doublets: object,
    markers: object,
    snapshot_columns: object,
    species: object,
) -> RecipePlan:
    """Validate an RNA recipe call against ``store`` before anything is written."""
    assay_name = modality_assay(
        store, assay, recipe=RECIPE, modality="RNA", classes=(RNAassay,)
    )
    resolved_label = resolve_label(label)
    hashing_enabled, hashing_settings = stage_settings(
        "hashing", hashing, optional=True
    )
    hto_assay = _resolve_hto_assay(store, hashing_settings) if hashing_enabled else None
    resolved_species = resolve_species(species)
    membership_strength = require_bool(membership_strength, "membership_strength")
    markers = require_bool(markers, "markers")
    calling_enabled, calling_settings = stage_settings(
        "cell_calling", cell_calling, optional=True
    )
    capture_column = calling_settings.pop("sample_column", None)
    if capture_column is not None:
        if not isinstance(capture_column, str) or not capture_column:
            raise TypeError("cell_calling['sample_column'] must be a cell column")
        if capture_column not in store.cells.columns:
            raise KeyError(f"Capture column {capture_column!r} was not found")

    cell_cycle_enabled, cell_cycle_settings = stage_settings(
        "cell_cycle", cell_cycle, optional=True
    )
    _hvg_enabled, hvg_settings = stage_settings("hvg", hvg, optional=False)
    hvg_batch = hvg_settings.get("batch")
    if hvg_batch is not None and hvg_batch not in store.cells.columns:
        raise KeyError(f"HVG batch column {hvg_batch!r} was not found")
    _norm_enabled, normalization_settings = stage_settings(
        "normalization", normalization, optional=False
    )
    _pca_enabled, pca_settings = stage_settings("pca", pca, optional=False)
    pca_dims = integer_argument(
        pca_settings.pop("dims", PCA_DIMS_DEFAULT), "pca dims", minimum=0
    )
    if pca_dims == 0 and pca_settings:
        raise ValueError("PCA settings need dims above 0; dims=0 skips PCA")
    hvg_top_n = hvg_settings.get("top_n", HVG_TOP_N_DEFAULT)
    if pca_dims >= hvg_top_n:
        raise ValueError(
            f"pca dims ({pca_dims}) must be below hvg top_n ({hvg_top_n}): PCA "
            "needs at least dims + 1 features"
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
    doublets_enabled, doublet_settings = stage_settings(
        "doublets", doublets, optional=True
    )
    if not partitions and (doublets_enabled or markers):
        raise ValueError("doublets and markers require at least one Leiden candidate")
    if not partitions and membership_strength:
        raise ValueError("membership_strength requires at least one Leiden candidate")
    # Without PCA the graph is built on the normalized values themselves. The
    # stages that need reduced coordinates are refused here, before the run
    # record exists, never skipped on the run's behalf.
    if pca_dims == 0 and harmony_columns:
        raise ValueError(
            "pca={'dims': 0} builds the graph on normalized values, which Harmony "
            "cannot correct; set pca dims above 0 or omit harmony"
        )
    if pca_dims == 0 and doublets_enabled:
        raise ValueError(
            "pca={'dims': 0} builds the graph on normalized values, but doublet "
            "scoring needs a PCA graph; pass doublets=False or set pca dims "
            "above 0"
        )
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=run_reserved_fields(
            embeddings, partitions, extra=("highly_variable_features",)
        ),
    )
    doublet_samples = (
        _resolve_doublet_samples(store, doublet_settings, harmony_columns)
        if doublets_enabled
        else ()
    )
    # The settings are checked, so the run's cells are checked last: a run
    # refuses cells that one of its assays did not measure.
    source = resolve_cells(
        store,
        cells,
        assays=(assay_name,) if hto_assay is None else (assay_name, hto_assay),
        operation=OPERATION,
    )
    filtering_config = resolve_run_filtering(
        store,
        filtering,
        cells=source,
        assay=assay_name,
        default_attrs=default_filter_columns(store, assay_name, _FILTER_SUFFIXES),
        operation=OPERATION,
        qc_stages=calling_enabled or hashing_enabled,
    )
    if calling_enabled and capture_column is not None:
        _check_capture_labels(store, source, capture_column)
    cell_columns, extra_snapshots = cell_snapshots(
        filtering_config,
        harmony_columns=harmony_columns,
        snapshot_columns=snapshots,
        capture_column=capture_column,
        doublet_samples=doublet_samples,
    )
    # The cells that each QC stage reads: the input, the called cells, and
    # the hashtag singlets, in that order.
    calling_input = "input_cell_selection"
    hashing_input = "called_cell_selection" if calling_enabled else calling_input
    filtering_input = "hashed_cell_selection" if hashing_enabled else hashing_input
    # A hashed run's MAD bounds come from each hashtag's cells unless the
    # filtering settings name a sample column.
    hashtag_samples = (
        hashing_enabled
        and filtering_config.get("method") == "mad"
        and filtering_config.get("sampleColumn") is None
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
    stages = (
        input_snapshot_stage(
            source,
            assay=assay_name,
            cell_snapshot_columns=cell_columns,
            extra_snapshots=extra_snapshots,
        ),
        _cell_calling_stage(
            calling_enabled, assay_name, calling_settings, capture_column
        ),
        _hashing_stage(hashing_enabled, hto_assay, hashing_settings, hashing_input),
        filtering_stage(
            filtering_config,
            source,
            input_key=filtering_input,
            hashtags=hashtag_samples,
            called=calling_enabled,
        ),
        _cell_cycle_stage(cell_cycle_enabled, cell_cycle_settings),
        _hvg_stage(hvg_settings),
        normalization_stage("highly_variable_features", normalization_settings),
        pca_stage(pca_dims, pca_settings),
        harmony_stage(harmony_columns, harmony_settings),
        *graph_stages(
            ann_index=graph.ann_index,
            neighbors=graph.neighbors,
            connectivity=graph.connectivity,
        ),
        *first_tail,
        *tsne_tail,
        _doublet_stage(doublets_enabled, doublet_settings, doublet_samples),
        markers_stage(markers),
    )
    config = {
        "cells": source.to_config(),
        "snapshotColumns": list(snapshots),
        "species": resolved_species,
        "doubletSampleColumns": list(doublet_samples),
        "stages": {
            "cell_calling": (
                {
                    **calling_settings,
                    **(
                        {}
                        if capture_column is None
                        else {"sample_column": capture_column}
                    ),
                }
                if calling_enabled
                else False
            ),
            "hashing": (
                {"assay": hto_assay, **hashing_settings} if hashing_enabled else False
            ),
            "filtering": (
                {**filtering_config, "sampleSource": "hashing"}
                if hashtag_samples
                else filtering_config
            ),
            "cell_cycle": cell_cycle_settings if cell_cycle_enabled else False,
            "hvg": hvg_settings,
            "normalization": normalization_settings,
            "pca": {"dims": pca_dims, **pca_settings},
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
        qc=(
            "cell_calling",
            "called_cell_selection",
            "hashing",
            "hashed_cell_selection",
        ),
        head=("cell_cycle", "highly_variable_features"),
        reduction=("pca",),
        partitions=partitions,
        tail=("doublets", "markers"),
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
                selection_key="highly_variable_features",
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
