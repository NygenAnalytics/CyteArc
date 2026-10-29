"""The perturbation recipe: the analysis of a pooled CRISPR screen, end to end.

The recipe calls the guides of its cells, or imports Cell Ranger's calls,
labels each cell by the genes that its guides target, and measures each
perturbation: its efficiency on its own gene, how far it moves its cells from
the control cells, which of its cells Mixscape calls knocked out, which genes
it changes, and the clusters of the perturbations whose profiles agree. The
stages of the RNA recipe from QC filtering to Harmony give the cells, the
highly variable genes and the coordinates, so a run with an RNA run's settings
reuses that run's artifacts.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ....assay import RNAassay
from ....perturbation.associations import covariate_columns
from ....perturbation.guide_calls import validated_guide_ids
from ....perturbation.labels import guide_targets, label_settings
from ....storage.artifacts import ArtifactRef, inspect_artifact
from ....storage.types import as_zarr_group
from ....utils.arguments import integer_argument
from ..executor import PlannedStage, RecipePlan, RunState
from ..fields import cell_fields, feature_fields, reserved_fields
from ..filtering import default_filter_columns, snapshot_column_values
from ..settings import (
    LABELS_NEED_CONTROLS,
    PCA_DIMS_DEFAULT,
    SIGNATURE_DIMS_DEFAULT,
    method_setting,
    stage_settings,
)
from ..stages import (
    CellSource,
    Outputs,
    filtering_stage,
    harmony_stage,
    hvg_stage,
    input_snapshot_stage,
    normalization_stage,
    pca_stage,
    resolve_cells,
)
from ..summaries import summarize_stage
from .common import (
    cell_snapshots,
    check_labeled_cells,
    check_pca_features,
    harmony_config,
    input_cell_mask,
    modality_assay,
    resolve_harmony,
    resolve_hvg,
    resolve_label,
    resolve_run_filtering,
    resolve_snapshot_columns,
)
from .rna import FILTER_SUFFIXES

RECIPE = "perturbation"
OPERATION = "pipeline.perturbation"
# The declared type of the guide assay whose counts the guides are called from.
GUIDE_ASSAY_TYPE = "CRISPR"
# The run's cell fields of Mixscape's classes, which no snapshot column may
# take, whether or not the run runs Mixscape.
MIXSCAPE_FIELDS = ("mixscape_class", "mixscape_posterior")
# The run's outputs, in record order; an output that the run did not produce
# is left out.
OUTPUT_KEYS = (
    "input_cell_selection",
    "analysis_cell_selection",
    "feature_universe",
    "guide_thresholds",
    "guide_calls",
    "perturbation_labels",
    "efficiency",
    "highly_variable_features",
    "normalized",
    "pca",
    "harmony",
    "edistance",
    "signature",
    "mixscape",
    "associations",
    "profiles",
    "perturbation_clusters",
)
# The recipe tests trans pairs over its highly variable genes, so a stage
# setting cannot name other pairs or features.
TRANS_PAIRS_ONLY = (
    "The recipe tests trans pairs over the run's highly variable genes; for "
    "tables of pairs or positive controls, call ds.perturbation.associations"
)
# The text between the sample's name and the cell's own ID in the cell IDs of
# a merged store, as DataStoreMerge writes them.
MERGED_SEPARATOR = "__"


def _resolve_guide_assay(store: Any, name: object) -> tuple[str, Any]:
    """Return the guide assay, given or the store's only CRISPR assay, by name.

    The assay must be declared ``CRISPR`` and store unsigned counts, as guide
    calling needs.
    """
    if name is None:
        crispr = [
            str(assay)
            for assay in store.assay_names
            if getattr(store._get_assay(assay), "assayType", None) == GUIDE_ASSAY_TYPE
        ]
        if len(crispr) != 1:
            raise ValueError(
                f"The perturbation recipe needs an assay declared {GUIDE_ASSAY_TYPE!r}; "
                f"name it with guide_assay=...; CRISPR assays: {crispr!r}"
            )
        name = crispr[0]
    elif not isinstance(name, str) or not name:
        raise TypeError("guide_assay must be a non-empty string or None")
    return name, store._guide_assay(name)


def _cell_column(store: Any, settings: Mapping[str, Any], stage: str, key: str) -> None:
    """Refuse a setting that names a cell column the store lacks."""
    column = settings.get(key)
    if column is None:
        return
    if not isinstance(column, str) or not column:
        raise TypeError(f"{stage}['{key}'] must be a non-empty string or None")
    if column not in store.cells.columns:
        raise KeyError(f"{stage}['{key}'] names {column!r}, which is not a cell column")


def _resolve_guide_calling(store: Any, value: object) -> dict[str, Any]:
    """Return the settings that call guides or import Cell Ranger's calls."""
    if isinstance(value, Mapping) and isinstance(value.get("path"), os.PathLike):
        # A path object is recorded as the text of its path.
        value = {**value, "path": os.fspath(value["path"])}
    _enabled, settings = stage_settings("guide_calling", value, optional=False)
    if "path" in settings:
        if not os.path.isfile(settings["path"]):
            raise FileNotFoundError(
                f"guide_calling['path'] {settings['path']!r} is not a file"
            )
    else:
        _cell_column(store, settings, "guide_calling", "sample_column")
    return settings


def _resolve_labels(store: Any, guide: Any, value: object) -> dict[str, Any]:
    """Return the label settings after checking every guide's target."""
    if value is None or (isinstance(value, Mapping) and "controls" not in value):
        raise ValueError(LABELS_NEED_CONTROLS)
    _enabled, settings = stage_settings("labels", value, optional=False)
    target_column, targets, policy = (
        method_setting("labels", settings, name)
        for name in ("target_column", "targets", "policy")
    )
    controls = label_settings(settings["controls"], target_column, targets, policy)
    # The targets that the guide assay gives, as labelling resolves them; a
    # called guide without a target needs the calls, so its stage refuses it.
    guide_targets(
        validated_guide_ids(guide.feats.fetch_all("ids")),
        store._guide_feature_column(guide, target_column, "target_column"),
        targets,
        controls,
        policy=policy,
    )
    return settings


def _require_id_column(
    store: Any, guide: Any, settings: Mapping[str, Any], stage: str
) -> None:
    """Refuse an ``id_column`` that the guide assay lacks."""
    store._guide_feature_column(
        guide, method_setting(stage, settings, "id_column"), f"{stage}['id_column']"
    )


def _resolve_associations(
    store: Any, guide: Any, value: object
) -> tuple[bool, dict[str, Any]]:
    """Return whether association tests run and their settings."""
    if isinstance(value, Mapping) and (
        "pairs" in value
        or "features" in value
        or not isinstance(value.get("positive_controls", "targets"), str | None)
    ):
        raise ValueError(TRANS_PAIRS_ONLY)
    enabled, settings = stage_settings("associations", value, optional=True)
    if not enabled:
        return False, {}
    if method_setting("associations", settings, "positive_controls") == "targets":
        _require_id_column(store, guide, settings, "associations")
    _cell_column(store, settings, "associations", "sample_column")
    for column in covariate_columns(settings.get("covariates")):
        if column not in store.cells.columns:
            raise KeyError(
                f"associations['covariates'] names {column!r}, which is not a "
                "cell column"
            )
        # The tests adjust for numbers only, as associations checks them.
        if np.dtype(store.cells.get_dtype(column)).kind not in "biuf":
            raise ValueError(f"covariate '{column}' must be numeric")
    return True, settings


def _settings_only(
    stage: str, value: object, runs_with: str, enabled: bool
) -> dict[str, Any]:
    """Return the settings of a stage that runs with another one.

    A signature runs with Mixscape and profiles with clusters; their settings
    without that stage are refused, never recorded unused.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"{stage} must be a mapping or None; {runs_with} runs the {stage} stage"
        )
    _enabled, settings = stage_settings(stage, value, optional=False)
    if settings and not enabled:
        raise ValueError(
            f"{stage} settings apply only with {runs_with}, which runs the "
            f"{stage} stage"
        )
    return settings


def _guide_calling_stage(guide_assay: str, settings: Mapping[str, Any]) -> PlannedStage:
    """Call the guides of the analysis cells, or import Cell Ranger's calls."""

    def action(state: RunState) -> Outputs:
        store = state.store
        cells = state.artifacts["analysis_cell_selection"]
        if "path" in settings:
            calls = store._perturbation_import_guide_calls(
                settings["path"],
                cell_selection=cells,
                from_assay=guide_assay,
                barcode_prefix=settings.get("barcode_prefix"),
            )
            return (("guide_calls", calls),)
        calls = store._perturbation_call_guides(
            cells, from_assay=guide_assay, **settings
        )
        # The fitted methods save the fits that the calls take as an input.
        fits = (inspect_artifact(store.zw, calls).inputs or {}).get("guide_thresholds")
        if fits is None:
            return (("guide_calls", calls),)
        return (
            ("guide_thresholds", ArtifactRef.from_dict(dict(fits))),
            ("guide_calls", calls),
        )

    return PlannedStage("guide_calling", action)


def _labels_stage(settings: Mapping[str, Any]) -> PlannedStage:
    def action(state: RunState) -> Outputs:
        ref = state.store._perturbation_labels(
            state.artifacts["guide_calls"], **settings
        )
        return (("perturbation_labels", ref),)

    return PlannedStage("perturbation_labels", action)


def _efficiency_stage(
    enabled: bool, assay: str, settings: Mapping[str, Any]
) -> PlannedStage:
    if not enabled:
        return PlannedStage("efficiency", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._perturbation_efficiency(
            state.artifacts["perturbation_labels"], from_assay=assay, **settings
        )
        return (("efficiency", ref),)

    return PlannedStage("efficiency", action)


def _edistance_stage(enabled: bool, settings: Mapping[str, Any]) -> PlannedStage:
    """Measure E-distances in the Harmony coordinates, else the PCA."""
    if not enabled:
        return PlannedStage("edistance", None)

    def action(state: RunState) -> Outputs:
        assert state.coordinates is not None
        ref = state.store._perturbation_edistance(
            state.artifacts["perturbation_labels"], state.coordinates, **settings
        )
        return (("edistance", ref),)

    return PlannedStage("edistance", action)


def _signature_stage(enabled: bool, settings: Mapping[str, Any]) -> PlannedStage:
    """Compute signatures in the highly variable genes against near controls.

    The nearest controls are found in the Harmony coordinates, else the PCA.
    """
    if not enabled:
        return PlannedStage("signature", None)

    def action(state: RunState) -> Outputs:
        assert state.coordinates is not None
        ref = state.store._perturbation_signature(
            state.artifacts["perturbation_labels"],
            state.coordinates,
            features=state.artifacts["highly_variable_features"],
            **settings,
        )
        return (("signature", ref),)

    return PlannedStage("signature", action)


def _mixscape_stage(enabled: bool, settings: Mapping[str, Any]) -> PlannedStage:
    if not enabled:
        return PlannedStage("mixscape", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._perturbation_mixscape(
            state.artifacts["signature"], **settings
        )
        return (("mixscape", ref),)

    return PlannedStage("mixscape", action)


def _associations_stage(
    enabled: bool, assay: str, settings: Mapping[str, Any]
) -> PlannedStage:
    """Test every target against every highly variable gene."""
    if not enabled:
        return PlannedStage("associations", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._perturbation_associations(
            state.artifacts["perturbation_labels"],
            from_assay=assay,
            pairs="trans",
            features=state.artifacts["highly_variable_features"],
            **settings,
        )
        return (("associations", ref),)

    return PlannedStage("associations", action)


def _profiles_stage(
    enabled: bool, labels_key: str, settings: Mapping[str, Any]
) -> PlannedStage:
    """Profile the highly variable genes of each perturbation's cells.

    ``labels_key`` names the output whose cells count: the labels, or the
    Mixscape classes, whose knocked-out cells alone count for a target.
    """
    if not enabled:
        return PlannedStage("profiles", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._perturbation_profiles(
            state.artifacts[labels_key],
            state.artifacts["highly_variable_features"],
            **settings,
        )
        return (("profiles", ref),)

    return PlannedStage("profiles", action)


def _clusters_stage(enabled: bool, settings: Mapping[str, Any]) -> PlannedStage:
    if not enabled:
        return PlannedStage("perturbation_clusters", None)

    def action(state: RunState) -> Outputs:
        ref = state.store._perturbation_cluster(state.artifacts["profiles"], **settings)
        return (("perturbation_clusters", ref),)

    return PlannedStage("perturbation_clusters", action)


def _check_labeled_inputs(
    store: Any,
    source: CellSource,
    columns: Mapping[str, str | None],
) -> None:
    """Refuse sample and split columns that leave an input cell unlabeled.

    ``columns`` maps what needs each column's labels to the column, or None.
    Empty or whitespace text is no label, since the methods refuse it as a
    sample.
    """
    for purpose, column in columns.items():
        if column is not None:
            check_labeled_cells(
                store, source, column, purpose=purpose, blank_is_label=False
            )


def _check_covariates(store: Any, source: CellSource, columns: Sequence[str]) -> None:
    """Refuse covariates without a finite value for every input cell.

    The association tests refuse a missing or non-finite covariate value of a
    cell that they test, and every input cell can be one.
    """
    if not columns:
        return
    active = input_cell_mask(store, source)
    cell_data = as_zarr_group(store.zw["cellData"], name="cellData")
    for column in columns:
        values, missing = snapshot_column_values(cell_data, column)
        unset = ~np.isfinite(values.astype(np.float64, copy=False))
        if missing is not None:
            unset |= missing
        count = int(np.count_nonzero(unset & active))
        if count:
            verb = "has" if count == 1 else "have"
            raise ValueError(
                f"Covariate {column!r} needs a finite value for every cell, but "
                f"{count} of the run's cells {verb} a missing or non-finite value"
            )


def _check_call_file_cells(
    store: Any, source: CellSource, barcode_prefix: str | None
) -> None:
    """Refuse input cells that the barcodes of one call file cannot reach.

    Cell Ranger writes a call file for each sample, and the import matches
    each of its barcodes to the cell whose ID is ``barcode_prefix`` followed
    by the barcode. A cell that no barcode can reach would be labelled
    ``"Negative"`` and left out of every later stage.
    """
    ids = np.asarray(store.cells.fetch_all("ids"))[input_cell_mask(store, source)]
    ids = ids.astype(str)
    remedy = (
        "or call the guides from their counts with guide_calling settings without path"
    )
    if barcode_prefix:
        outside = ~np.strings.startswith(ids, barcode_prefix)
        count = int(np.count_nonzero(outside))
        if count:
            verb = "does" if count == 1 else "do"
            raise ValueError(
                "The call file's barcodes reach only the cells whose IDs start "
                f"with barcode_prefix {barcode_prefix!r}, but {count} of the "
                f"run's cells, such as {str(ids[np.argmax(outside)])!r}, {verb} "
                f"not; restrict cells to the cells of that sample, {remedy}"
            )
        return
    # A merged store names each cell <sample>__<barcode>.
    position = np.strings.find(ids, MERGED_SEPARATOR)
    merged = np.flatnonzero(position >= 0)
    if not len(merged):
        return
    first = str(ids[merged[0]])[: int(position[merged[0]])]
    other = merged[~np.strings.startswith(ids[merged], first + MERGED_SEPARATOR)]
    if len(other):
        second = str(ids[other[0]])[: int(position[other[0]])]
        raise ValueError(
            "The call file holds the barcodes of one sample, but the run's cells "
            f"come from several samples of a merged store, such as {first!r} and "
            f"{second!r}; restrict cells to the cells of one sample and pass its "
            f"barcode_prefix, such as {first + MERGED_SEPARATOR!r}, {remedy}"
        )


def resolve_perturbation(
    store: Any,
    *,
    assay: object,
    guide_assay: object,
    label: object,
    cells: object,
    filtering: object,
    guide_calling: object,
    labels: object,
    efficiency: object,
    hvg: object,
    normalization: object,
    pca: object,
    harmony: object,
    edistance: object,
    signature: object,
    mixscape: object,
    associations: object,
    profiles: object,
    clusters: object,
    snapshot_columns: object,
) -> RecipePlan:
    """Validate a perturbation recipe call against ``store`` before anything is written."""
    assay_name = modality_assay(
        store, assay, recipe=RECIPE, modality="RNA", classes=(RNAassay,)
    )
    guide_name, guide = _resolve_guide_assay(store, guide_assay)
    resolved_label = resolve_label(label)
    calling_settings = _resolve_guide_calling(store, guide_calling)
    label_settings_ = _resolve_labels(store, guide, labels)
    efficiency_enabled, efficiency_settings = stage_settings(
        "efficiency", efficiency, optional=True
    )
    if efficiency_enabled:
        _require_id_column(store, guide, efficiency_settings, "efficiency")
    hvg_settings = resolve_hvg(store, hvg)
    _norm_enabled, normalization_settings = stage_settings(
        "normalization", normalization, optional=False
    )
    _pca_enabled, pca_settings = stage_settings("pca", pca, optional=False)
    # E-distances and signatures read the PCA, so the recipe always runs it.
    pca_dims = integer_argument(
        pca_settings.pop("dims", PCA_DIMS_DEFAULT), "pca dims", minimum=1
    )
    check_pca_features(pca_dims, hvg_settings)
    harmony_columns, harmony_settings = resolve_harmony(store, harmony)
    edistance_enabled, edistance_settings = stage_settings(
        "edistance", edistance, optional=True
    )
    mixscape_enabled, mixscape_settings = stage_settings(
        "mixscape", mixscape, optional=True
    )
    signature_settings = _settings_only(
        "signature", signature, "mixscape", mixscape_enabled
    )
    if mixscape_enabled:
        _cell_column(store, mixscape_settings, "mixscape", "split_column")
        _cell_column(store, signature_settings, "signature", "sample_column")
        signature_dims = signature_settings.get("dims", SIGNATURE_DIMS_DEFAULT)
        if signature_dims > pca_dims:
            raise ValueError(
                f"signature dims ({signature_dims}) must be at most pca dims "
                f"({pca_dims}): the signature finds each cell's nearest controls "
                "in the run's coordinates"
            )
    associations_enabled, association_settings = _resolve_associations(
        store, guide, associations
    )
    clusters_enabled, cluster_settings = stage_settings(
        "clusters", clusters, optional=True
    )
    profile_settings = _settings_only(
        "profiles", profiles, "clusters", clusters_enabled
    )
    if clusters_enabled:
        _cell_column(store, profile_settings, "profiles", "sample_column")
        if method_setting("profiles", profile_settings, "mask_targets"):
            _require_id_column(store, guide, profile_settings, "profiles")
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=reserved_fields(
            umap_dims=0,
            tsne_dims=0,
            leiden_keys=(),
            extra=("highly_variable_features", *MIXSCAPE_FIELDS),
        ),
    )
    # The settings are checked, so the run's cells are checked last: a run
    # refuses cells that the RNA or the guide assay did not measure.
    source = resolve_cells(
        store, cells, assays=(assay_name, guide_name), operation=OPERATION
    )
    filtering_config = resolve_run_filtering(
        store,
        filtering,
        cells=source,
        assay=assay_name,
        default_attrs=default_filter_columns(store, assay_name, FILTER_SUFFIXES),
        operation=OPERATION,
    )
    # One call file reaches the cells of one sample.
    if "path" in calling_settings:
        _check_call_file_cells(store, source, calling_settings.get("barcode_prefix"))
    # The stages that read these columns read some of the input cells, which
    # of them depends on the data, so every input cell needs a label and a
    # finite covariate value.
    _check_labeled_inputs(
        store,
        source,
        {
            "Guide calling per sample": calling_settings.get("sample_column"),
            "The signature per sample": (
                signature_settings.get("sample_column") if mixscape_enabled else None
            ),
            "Mixscape per split": (
                mixscape_settings.get("split_column") if mixscape_enabled else None
            ),
            "Association testing per sample": association_settings.get("sample_column"),
            "Profiling per sample": (
                profile_settings.get("sample_column") if clusters_enabled else None
            ),
        },
    )
    _check_covariates(
        store, source, covariate_columns(association_settings.get("covariates"))
    )
    cell_columns, extra_snapshots = cell_snapshots(
        filtering_config, harmony_columns=harmony_columns, snapshot_columns=snapshots
    )
    # Profiles of a run with Mixscape average the knocked-out cells of each
    # target, so that escapers do not dilute them.
    profile_labels = "mixscape" if mixscape_enabled else "perturbation_labels"
    stages = (
        input_snapshot_stage(
            source,
            assay=assay_name,
            cell_snapshot_columns=cell_columns,
            extra_snapshots=extra_snapshots,
        ),
        filtering_stage(filtering_config, source),
        _guide_calling_stage(guide_name, calling_settings),
        _labels_stage(label_settings_),
        _efficiency_stage(efficiency_enabled, assay_name, efficiency_settings),
        hvg_stage(hvg_settings),
        normalization_stage("highly_variable_features", normalization_settings),
        pca_stage(pca_dims, pca_settings),
        harmony_stage(harmony_columns, harmony_settings),
        _edistance_stage(edistance_enabled, edistance_settings),
        _signature_stage(mixscape_enabled, signature_settings),
        _mixscape_stage(mixscape_enabled, mixscape_settings),
        _associations_stage(associations_enabled, assay_name, association_settings),
        _profiles_stage(clusters_enabled, profile_labels, profile_settings),
        _clusters_stage(clusters_enabled, cluster_settings),
    )
    config = {
        "cells": source.to_config(),
        "snapshotColumns": list(snapshots),
        "guideAssay": guide_name,
        "profileLabels": profile_labels if clusters_enabled else None,
        "stages": {
            "filtering": filtering_config,
            "guide_calling": calling_settings,
            "labels": label_settings_,
            "efficiency": efficiency_settings if efficiency_enabled else False,
            "hvg": hvg_settings,
            "normalization": normalization_settings,
            "pca": {"dims": pca_dims, **pca_settings},
            "harmony": harmony_config(harmony_columns, harmony_settings),
            "edistance": edistance_settings if edistance_enabled else False,
            "signature": signature_settings if mixscape_enabled else False,
            "mixscape": mixscape_settings if mixscape_enabled else False,
            "associations": (association_settings if associations_enabled else False),
            "profiles": profile_settings if clusters_enabled else False,
            "clusters": cluster_settings if clusters_enabled else False,
        },
    }

    def fields(state: RunState) -> tuple[Any, ...]:
        return (
            *cell_fields(
                store, state.artifacts, snapshot_columns=snapshots, leiden_keys=()
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
        output_keys=OUTPUT_KEYS,
        fields=fields,
        start=lambda store: RunState(store=store, assay=store._get_assay(assay_name)),
        summarize=lambda stage, outputs: summarize_stage(store, stage, outputs),
    )
