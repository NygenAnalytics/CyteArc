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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ....assay import RNAassay
from ....metadata.rows import read_metadata_rows_chunkwise
from ....perturbation.associations import covariate_columns
from ....perturbation.guide_calls import validated_guide_ids
from ....perturbation.labels import GuideTargets, guide_targets, label_settings
from ....readers.cellranger import protospacer_calls_size, read_protospacer_calls
from ....storage.artifacts import ArtifactRef, inspect_artifact
from ....storage.types import as_zarr_group
from ..._operations.perturbation_guides import (
    _barcode_ids,
    _call_file_check_bytes,
    _called_guide_positions,
    _require_call_file_cells,
)
from ..._operations.perturbation_store import _require_budget, _require_cell_column
from ..executor import PlannedStage, RecipePlan, RunState
from ..fields import (
    PERTURBATION_CELL_FIELDS,
    cell_fields,
    feature_fields,
    reserved_fields,
)
from ..filtering import default_filter_columns, snapshot_column_values
from ..settings import (
    LABELS_NEED_CONTROLS,
    SIGNATURE_DIMS_DEFAULT,
    method_setting,
    stage_settings,
)
from ..stages import (
    CellSource,
    Outputs,
    filtering_stage,
    input_snapshot_stage,
    resolve_cells,
)
from ..summaries import summarize_stage
from .common import (
    cell_snapshots,
    check_labeled_cells,
    input_cell_mask,
    modality_assay,
    resolve_label,
    resolve_rna_head,
    resolve_run_filtering,
    resolve_snapshot_columns,
)
from .rna import FILTER_SUFFIXES

RECIPE = "perturbation"
OPERATION = "pipeline.perturbation"
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
# The cells that the profiles average, by the output that holds them: every
# cell of each label, or the cells that Mixscape calls knocked out.
PROFILE_CELLS = {"labelled": "perturbation_labels", "knocked_out": "mixscape"}
# The stage keywords whose stages compare each perturbation with the cells that
# carry control guides only, which a high-MOI screen hardly holds: the
# efficiency, the E-distances, Mixscape with its signature, and the clusters
# with the profiles, which z-score each cell against those cells.
CONTROL_CELL_STAGES = ("efficiency", "edistance", "mixscape", "clusters")
# The other way out of a call file that cannot label the run's cells.
CALL_FROM_COUNTS = (
    "or call the guides from their counts with guide_calling settings without path"
)


def _resolve_guide_assay(store: Any, name: object) -> tuple[str, Any]:
    """Return the guide assay, given or the store's only CRISPR assay, by name.

    The perturbation methods choose the assay by the same rule. The assay must
    be declared ``CRISPR`` and store unsigned counts, as guide calling needs.
    """
    if name is not None and (not isinstance(name, str) or not name):
        raise TypeError("guide_assay must be a non-empty string or None")
    guide = store._guide_assay(name, argument="guide_assay")
    return str(guide.name), guide


def _cell_column(store: Any, settings: Mapping[str, Any], stage: str, key: str) -> None:
    """Refuse a setting that names a cell column the store lacks, as its method does.

    The stage's settings check has refused a value that is not a name.
    """
    _require_cell_column(store, settings.get(key), f"{stage}['{key}']")


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


def _resolve_labels(
    store: Any, guide: Any, value: object
) -> tuple[dict[str, Any], np.ndarray, GuideTargets]:
    """Return the label settings, the guide IDs, and each guide's target.

    Every guide's target is checked as labelling resolves it.
    """
    # The settings check refuses a mapping without controls; None takes the
    # method's defaults unchecked, and controls have no default.
    if value is None:
        raise ValueError(LABELS_NEED_CONTROLS)
    _enabled, settings = stage_settings("labels", value, optional=False)
    target_column, targets, policy = (
        method_setting("labels", settings, name)
        for name in ("target_column", "targets", "policy")
    )
    controls = label_settings(settings["controls"], target_column, targets, policy)
    # The targets that the guide assay gives, as labelling resolves them; a
    # called guide without a target needs the calls, so its stage refuses it.
    guide_ids = validated_guide_ids(guide.feats.fetch_all("ids"))
    resolved = guide_targets(
        guide_ids,
        store._guide_feature_column(guide, target_column, "target_column"),
        targets,
        controls,
        policy=policy,
    )
    return settings, guide_ids, resolved


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
        _require_cell_column(store, column, "covariate")
        # The tests adjust for numbers only, as associations checks them.
        if np.dtype(store.cells.get_dtype(column)).kind not in "biuf":
            raise ValueError(f"covariate '{column}' must be numeric")
    return True, settings


def _check_cluster_perturbations(
    settings: Mapping[str, Any], by: str, guide_ids: np.ndarray, targets: GuideTargets
) -> None:
    """Refuse perturbations to cluster that can have no profile.

    By target the profiles hold the targets that are not controls, and with
    ``by="guide"`` every guide with a target, the control guides included.
    Whether a perturbation has the profiles' ``min_cells`` cells depends on
    the data, so the clustering stage checks it.
    """
    requested = settings.get("perturbations")
    if requested is None:
        return
    if by == "target":
        rows = targets.names[~targets.is_control].astype(str)
        held = "the targets that are not controls"
    else:
        rows = guide_ids[targets.guide_codes >= 0].astype(str)
        held = "the guides with a target"
    unknown = sorted(set(requested) - set(rows.tolist()))
    if unknown:
        examples = (
            "such as " + ", ".join(repr(name) for name in rows[:3].tolist())
            if len(rows)
            else "of which the guide assay has none"
        )
        raise ValueError(
            "perturbations names perturbations without a profile: "
            + ", ".join(repr(name) for name in unknown[:5])
            + f"; by {by}, the profiles hold {held}, {examples}"
        )


def _profile_labels(settings: Mapping[str, Any], mixscape: bool) -> str:
    """Return the output whose cells the profiles average.

    The profiles' ``cells`` setting chooses: ``"knocked_out"``, the default
    of a run with Mixscape, averages the cells that Mixscape calls knocked
    out, so that escapers do not dilute the profiles, and ``"labelled"``,
    the default otherwise, every cell of each label.

    Raises:
        ValueError: If ``cells`` is neither, or is ``"knocked_out"`` in a run
            without Mixscape.
    """
    cells = settings.get("cells", "knocked_out" if mixscape else "labelled")
    if not isinstance(cells, str) or cells not in PROFILE_CELLS:
        raise ValueError(
            f"profiles['cells'] must be 'knocked_out' or 'labelled', not {cells!r}"
        )
    if cells == "knocked_out" and not mixscape:
        raise ValueError(
            "profiles['cells'] 'knocked_out' applies only with mixscape, which "
            "calls the knocked-out cells"
        )
    return PROFILE_CELLS[cells]


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


def _stage(
    name: str, enabled: bool, compute: Callable[[RunState], ArtifactRef]
) -> PlannedStage:
    """Return the stage ``name``, which saves what ``compute`` returns under its name.

    A stage that is not enabled is recorded as skipped.
    """
    if not enabled:
        return PlannedStage(name, None)
    return PlannedStage(name, lambda state: ((name, compute(state)),))


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
    """Refuse covariates without a finite value for every input cell, or of one value.

    The association tests refuse a missing or non-finite covariate value of a
    cell that they test, and every input cell can be one. They refuse a
    covariate that is constant over the tested cells too, which a covariate of
    one value over the input cells is.
    """
    if not columns:
        return
    active = input_cell_mask(store, source)
    cell_data = as_zarr_group(store.zw["cellData"], name="cellData")
    for column in columns:
        values, missing = snapshot_column_values(cell_data, column)
        numbers = values.astype(np.float64, copy=False)
        unset = ~np.isfinite(numbers)
        if missing is not None:
            unset |= missing
        count = int(np.count_nonzero(unset & active))
        if count:
            verb = "has" if count == 1 else "have"
            raise ValueError(
                f"Covariate {column!r} needs a finite value for every cell, but "
                f"{count} of the run's cells {verb} a missing or non-finite value"
            )
        # The values themselves, since the standard deviation of one value
        # repeated can round above 0.
        held = numbers[active]
        if len(held) and held.min() == held.max():
            raise ValueError(
                f"The covariate {column!r} is constant over the run's cells"
            )


def _check_high_moi_stages(
    associations: Mapping[str, Any], enabled: Mapping[str, bool]
) -> None:
    """Refuse the stages that need control-only cells in a run that tests at high MOI.

    ``enabled`` holds whether each stage of :data:`CONTROL_CELL_STAGES` runs.
    In a high-MOI screen each cell carries several guides, so hardly any cell
    carries control guides only, and these stages would fail on such labels
    after the stages before them had written their artifacts.
    """
    if associations.get("moi") != "high":
        return
    running = [stage for stage in CONTROL_CELL_STAGES if enabled[stage]]
    if running:
        names = running[-1]
        verb, pronoun = "compares", "it"
        if len(running) > 1:
            names = ", ".join(running[:-1]) + " and " + names
            verb, pronoun = "compare", "them"
        raise ValueError(
            "associations={'moi': 'high'} tests a screen whose cells carry several "
            f"guides, where hardly any cell carries control guides only, but {names} "
            f"{verb} each perturbation with such cells; switch {pronoun} off for a "
            "high-MOI screen with " + ", ".join(f"{stage}=False" for stage in running)
        )


def _check_association_counts(
    store: Any,
    source: CellSource,
    expression: Any,
    guide: Any,
    associations: Mapping[str, Any] | None,
) -> None:
    """Refuse input cells without the counts that the association tests adjust for.

    The tests adjust for the log of each tested cell's UMIs in the expression
    assay, and at high MOI also for the log of its guide UMIs, and every input
    cell can be a tested cell. ``associations`` holds the tests' settings, or
    None when they do not run. Only the input cells' totals are read.
    """
    if associations is None:
        return
    rows = np.flatnonzero(input_cell_mask(store, source))
    checks = [
        (
            expression,
            "Association tests adjust for the log of each cell's UMIs in assay "
            f"{expression.name!r}",
            "cells with counts",
        )
    ]
    if associations.get("moi") == "high":
        checks.append(
            (
                guide,
                "High-MOI association tests adjust for the log of each cell's "
                "guide UMIs",
                "cells with guide UMIs",
            )
        )
    for assay, adjusted, remedy in checks:
        totals = np.asarray(
            read_metadata_rows_chunkwise(store.cells, assay._totals_name, rows),
            dtype=np.float64,
        )
        count = int(np.count_nonzero(~(totals > 0)))
        if count:
            verb = "has" if count == 1 else "have"
            raise ValueError(
                f"{adjusted}, but {count} of the run's cells {verb} none; restrict "
                f"cells to {remedy}"
            )


def _check_call_file_cells(
    store: Any, source: CellSource, barcode_prefix: str | None
) -> np.ndarray:
    """Refuse input cells that the barcodes of one call file cannot reach.

    Cell Ranger writes a call file for each sample, and the import matches
    each of its barcodes to the cell whose ID is ``barcode_prefix`` followed
    by the barcode. A cell that no barcode can reach would be labelled
    ``"Negative"`` and left out of every later stage.

    Returns:
        The input cells' IDs, as text.
    """
    # Only the input cells' IDs are read; the import checks its cells alike.
    rows = np.flatnonzero(input_cell_mask(store, source))
    ids = np.asarray(read_metadata_rows_chunkwise(store.cells, "ids", rows)).astype(str)
    _require_call_file_cells(
        ids,
        barcode_prefix,
        cells="the run's cells",
        restrict="restrict cells to",
        remedy=CALL_FROM_COUNTS,
    )
    return ids


def _check_call_file(
    store: Any,
    guide: Any,
    path: str,
    cell_ids: np.ndarray,
    barcode_prefix: str | None,
) -> None:
    """Refuse a call file that the import would refuse, or that labels no input cell.

    The file is read as the import reads it, once its read fits the memory
    budget, so a malformed file, a missing column, or a called guide that the
    guide assay lacks fails before anything is written. A file none of whose
    barcodes names one of the input cells, whose IDs are ``cell_ids``, would
    label every cell ``"Negative"``, and the stages after the labels would find
    no control cell.

    Raises:
        MemoryError: If reading the file beside the cells' IDs does not fit
            the memory budget.
        ValueError: If the import would refuse the file, or it labels no
            input cell.
    """
    guide_ids = validated_guide_ids(guide.feats.fetch_all("ids"))
    needed = _call_file_check_bytes(
        protospacer_calls_size(path),
        n_cells=len(cell_ids),
        guide_ids=guide_ids,
        id_array=store.cells._get_array("ids"),
        prefix_length=len(barcode_prefix or ""),
    )
    _require_budget(
        needed,
        store.memoryBytes,
        f"Checking the call file against the run's {len(cell_ids)} cells holds about",
        remedy=f"Raise mem_budget to at least {needed} bytes.",
    )
    calls = read_protospacer_calls(path)
    _called_guide_positions(calls.features, guide_ids, guide)
    if not len(calls.barcodes):
        raise ValueError(
            "The call file holds no barcode, so the import would label all of the "
            "run's cells 'Negative'; pass the call file of these cells, "
            + CALL_FROM_COUNTS
        )
    file_ids = _barcode_ids(calls.barcodes, barcode_prefix)
    if not np.any(pd.Index(cell_ids).get_indexer(file_ids) >= 0):
        raise ValueError(
            "No barcode of the call file names one of the run's cells, so the "
            "import would label them all 'Negative': a cell matches a barcode "
            "when its ID is barcode_prefix followed by the barcode, which gives "
            f"IDs such as {str(file_ids[0])!r}, but the run's cells have IDs such "
            f"as {str(cell_ids[0])!r}; check barcode_prefix and that the file "
            "holds the calls of these cells, " + CALL_FROM_COUNTS
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
    label_settings_, guide_ids, targets = _resolve_labels(store, guide, labels)
    efficiency_enabled, efficiency_settings = stage_settings(
        "efficiency", efficiency, optional=True
    )
    if efficiency_enabled:
        _require_id_column(store, guide, efficiency_settings, "efficiency")
    # E-distances and signatures read the PCA, so the recipe always runs it.
    head = resolve_rna_head(
        store,
        assay=assay_name,
        hvg=hvg,
        normalization=normalization,
        pca=pca,
        harmony=harmony,
        min_dims=1,
    )
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
        if signature_dims > head.pca_dims:
            raise ValueError(
                f"signature dims ({signature_dims}) must be at most pca dims "
                f"({head.pca_dims}): the signature finds each cell's nearest "
                "controls in the run's coordinates"
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
        _check_cluster_perturbations(
            cluster_settings,
            method_setting("profiles", profile_settings, "by"),
            guide_ids,
            targets,
        )
    profile_labels = _profile_labels(profile_settings, mixscape_enabled)
    _check_high_moi_stages(
        association_settings,
        {
            "efficiency": efficiency_enabled,
            "edistance": edistance_enabled,
            "mixscape": mixscape_enabled,
            "clusters": clusters_enabled,
        },
    )
    # No snapshot column may take the cell field of a perturbation result,
    # whether or not the run runs Mixscape.
    snapshots = resolve_snapshot_columns(
        store,
        snapshot_columns,
        reserved=reserved_fields(
            umap_dims=0,
            tsne_dims=0,
            leiden_keys=(),
            extra=("highly_variable_features", *PERTURBATION_CELL_FIELDS),
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
    # One call file reaches the cells of one sample, and the file is read
    # before the run, as the import reads it.
    if "path" in calling_settings:
        prefix = calling_settings.get("barcode_prefix")
        cell_ids = _check_call_file_cells(store, source, prefix)
        _check_call_file(store, guide, calling_settings["path"], cell_ids, prefix)
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
    head.check_batch_labels(store, source)
    _check_covariates(
        store, source, covariate_columns(association_settings.get("covariates"))
    )
    _check_association_counts(
        store,
        source,
        store._get_assay(assay_name),
        guide,
        association_settings if associations_enabled else None,
    )
    cell_columns, extra_snapshots = cell_snapshots(
        filtering_config,
        harmony_columns=head.harmony_columns,
        snapshot_columns=snapshots,
    )
    # The profiles' cells setting configures the recipe, not the method.
    profile_method_settings = {
        key: value for key, value in profile_settings.items() if key != "cells"
    }
    stages = (
        input_snapshot_stage(
            source,
            assay=assay_name,
            cell_snapshot_columns=cell_columns,
            extra_snapshots=extra_snapshots,
        ),
        filtering_stage(filtering_config, source),
        _guide_calling_stage(guide_name, calling_settings),
        _stage(
            "perturbation_labels",
            True,
            lambda state: state.store._perturbation_labels(
                state.artifacts["guide_calls"], **label_settings_
            ),
        ),
        _stage(
            "efficiency",
            efficiency_enabled,
            lambda state: state.store._perturbation_efficiency(
                state.artifacts["perturbation_labels"],
                from_assay=state.assay.name,
                **efficiency_settings,
            ),
        ),
        *head.stages(),
        # The E-distances and the signature's nearest controls are found in
        # the Harmony coordinates, else the PCA; the signature covers the
        # highly variable genes.
        _stage(
            "edistance",
            edistance_enabled,
            lambda state: state.store._perturbation_edistance(
                state.artifacts["perturbation_labels"],
                state.coordinates,
                **edistance_settings,
            ),
        ),
        _stage(
            "signature",
            mixscape_enabled,
            lambda state: state.store._perturbation_signature(
                state.artifacts["perturbation_labels"],
                state.coordinates,
                features=state.artifacts["highly_variable_features"],
                **signature_settings,
            ),
        ),
        _stage(
            "mixscape",
            mixscape_enabled,
            lambda state: state.store._perturbation_mixscape(
                state.artifacts["signature"], **mixscape_settings
            ),
        ),
        # Every target against every highly variable gene.
        _stage(
            "associations",
            associations_enabled,
            lambda state: state.store._perturbation_associations(
                state.artifacts["perturbation_labels"],
                from_assay=state.assay.name,
                pairs="trans",
                features=state.artifacts["highly_variable_features"],
                **association_settings,
            ),
        ),
        # The profiles of the highly variable genes average each label's
        # cells, or a target's knocked-out cells from Mixscape's classes.
        _stage(
            "profiles",
            clusters_enabled,
            lambda state: state.store._perturbation_profiles(
                state.artifacts[profile_labels],
                state.artifacts["highly_variable_features"],
                **profile_method_settings,
            ),
        ),
        _stage(
            "perturbation_clusters",
            clusters_enabled,
            lambda state: state.store._perturbation_cluster(
                state.artifacts["profiles"], **cluster_settings
            ),
        ),
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
            **head.to_config(),
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
