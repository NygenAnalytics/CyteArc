from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from ...metadata.selection import resolve_cell_aligned_artifact
from ...quality_control.filtering import (
    CALLED_CELLS_REMEDY,
    RAW_DROPLET_REMEDY,
    filter_cell_metrics,
    validate_filter_bounds,
)
from ...storage.arrays import linked_missing_mask
from ...storage.artifacts import ArtifactRef, artifact_group
from ...storage.selections import (
    read_stored_selection_mask,
    resolve_generated_selection_artifact,
)
from ...storage.types import as_zarr_array
from ...utils.arguments import integer_argument
from ...utils.logging import logger
from .._operations.quality_control import _QualityControlOperationsMixin
from .settings import column_sequence, finite_real, keyword_default

# Automatic filtering applies the rules of ``qc.auto_filter``, so its options
# default to that method's defaults. MAD filtering keeps the default
# quantiles, and Gaussian filtering keeps the default MAD options.
_FILTER_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        name: keyword_default(_QualityControlOperationsMixin._qc_auto_filter, name)
        for name in ("min_p", "max_p", "n_mads", "min_cells_per_sample")
    }
)
FILTERING_SETTINGS = frozenset(
    {
        "method",
        "attrs",
        "min_p",
        "max_p",
        "sample_column",
        "n_mads",
        "min_cells_per_sample",
        "lows",
        "highs",
        "keep_bounds",
    }
)


def snapshot_column_values(
    snapshot: Any,
    column: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one snapshot column and its linked nullable mask."""
    source = as_zarr_array(snapshot[column], name=column)
    values = np.asarray(source[:])
    if values.ndim != 1:
        raise ValueError(f"Snapshot column {column!r} must be one-dimensional")
    missing = linked_missing_mask(
        snapshot,
        column,
        label=f"Snapshot column {column!r}",
        values=source,
    )
    return values, None if missing is None else np.asarray(missing[:])


def default_filter_columns(
    store: Any,
    assay: str,
    suffixes: Sequence[str],
) -> tuple[str, ...]:
    """Return the QC columns of ``assay`` that preparation wrote, in order."""
    return tuple(
        column
        for suffix in suffixes
        if (column := f"{assay}_{suffix}") in store.cells.columns
    )


def _manual_bound(value: Any, name: str) -> float | int | None:
    """Return one validated pipeline bound in canonical form.

    The pipeline filters numeric QC columns, so text bounds are rejected.
    """
    if isinstance(value, str):
        raise TypeError(f"{name} values must be finite numbers or None")
    return value if value is None or isinstance(value, int) else float(value)


def resolve_filtering(
    store: Any,
    value: bool | Mapping[str, object],
    *,
    default_attrs: Sequence[str],
) -> dict[str, Any]:
    """Validate the ``filtering`` keyword into the stage's recorded settings.

    The settings are the filtering artifact's parameters, so equal settings
    reuse one filtered selection.
    """
    if value is False:
        return {"enabled": False}
    if value is True:
        options: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        options = dict(value)
        unknown_settings = sorted(set(options) - FILTERING_SETTINGS)
        if unknown_settings:
            raise ValueError(
                f"Unknown filtering settings {unknown_settings!r}; allowed: "
                f"{sorted(FILTERING_SETTINGS)!r}"
            )
    else:
        raise TypeError("filtering must be a mapping or bool")
    method = options.pop("method", "auto")
    if method == "auto":
        method = "mad"
    if method not in {"mad", "gaussian", "manual"}:
        raise ValueError(
            "filtering method must be 'auto', 'mad', 'gaussian', or 'manual'"
        )
    attrs = column_sequence(
        options.pop("attrs", tuple(default_attrs)),
        "filtering attrs",
    )
    missing = [column for column in attrs if column not in store.cells.columns]
    if missing:
        raise KeyError(f"Filtering columns were not found: {missing!r}")
    if not attrs:
        raise ValueError(
            "Filtering was requested, but no QC columns were found; "
            "pass filtering=False to analyze the unfiltered cell selection"
        )
    if method == "manual":
        allowed = {"lows", "highs", "keep_bounds"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown manual filtering options: {sorted(unknown)!r}")
        if "lows" not in options or "highs" not in options:
            raise ValueError("Manual filtering requires lows and highs")
        lows = list(options["lows"])
        highs = list(options["highs"])
        if len(lows) != len(attrs) or len(highs) != len(attrs):
            raise ValueError("Manual filtering bounds must align with attrs")
        keep_bounds = options.get("keep_bounds", False)
        validate_filter_bounds(lows, highs, keep_bounds=keep_bounds)
        return {
            "enabled": True,
            "method": "manual",
            "attrs": list(attrs),
            "lows": [_manual_bound(bound, "lows") for bound in lows],
            "highs": [_manual_bound(bound, "highs") for bound in highs],
            "keepBounds": keep_bounds,
        }
    allowed = {
        "min_p",
        "max_p",
        "sample_column",
        "n_mads",
        "min_cells_per_sample",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown automatic filtering options: {sorted(unknown)!r}")
    min_p = finite_real(options.get("min_p", _FILTER_DEFAULTS["min_p"]), "min_p")
    max_p = finite_real(options.get("max_p", _FILTER_DEFAULTS["max_p"]), "max_p")
    if not 0 < min_p < max_p < 1:
        raise ValueError("Automatic filtering requires 0 < min_p < max_p < 1")
    sample_column = options.get("sample_column")
    if sample_column is not None and (
        not isinstance(sample_column, str) or not sample_column
    ):
        raise TypeError("sample_column must be a non-empty string or None")
    if sample_column is not None and sample_column not in store.cells.columns:
        raise KeyError(f"Sample column {sample_column!r} was not found")
    n_mads = finite_real(options.get("n_mads", _FILTER_DEFAULTS["n_mads"]), "n_mads")
    if n_mads <= 0:
        raise ValueError("n_mads must be finite and positive")
    min_cells = integer_argument(
        options.get("min_cells_per_sample", _FILTER_DEFAULTS["min_cells_per_sample"]),
        "min_cells_per_sample",
        minimum=2,
    )
    if method == "mad" and (
        min_p != _FILTER_DEFAULTS["min_p"] or max_p != _FILTER_DEFAULTS["max_p"]
    ):
        raise ValueError(
            "min_p and max_p cannot be changed with method='mad'; "
            "use method='gaussian' for quantile bounds"
        )
    if method == "gaussian":
        if sample_column is not None:
            raise ValueError("Gaussian filtering does not support a sample source")
        if (
            n_mads != _FILTER_DEFAULTS["n_mads"]
            or min_cells != _FILTER_DEFAULTS["min_cells_per_sample"]
        ):
            raise ValueError(
                "n_mads and min_cells_per_sample apply only to method='mad'"
            )
    return {
        "enabled": True,
        "method": method,
        "attrs": list(attrs),
        "minP": min_p,
        "maxP": max_p,
        "sampleColumn": sample_column,
        "nMads": n_mads,
        "minCellsPerSample": min_cells,
    }


def filter_columns(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the cell columns that a filtering config reads."""
    columns = tuple(config.get("attrs", ()))
    sample_column = config.get("sampleColumn")
    if isinstance(sample_column, str):
        columns = (*columns, sample_column)
    return columns


def hashtag_labels(
    store: Any,
    hashing: ArtifactRef,
    cells: ArtifactRef,
    active: np.ndarray,
) -> np.ndarray:
    """Return each active cell's hashtag over the full cell axis.

    Inactive cells hold an empty label that no filter reads.
    """
    resolved = resolve_cell_aligned_artifact(
        store.zw,
        hashing,
        cell_selection=cells,
        expected_kind="hto_identity",
    )
    labels = np.full(len(active), "", dtype=object)
    labels[active] = np.asarray(resolved.values).astype(str)
    return labels


def filter_pipeline_selection(
    store: Any,
    *,
    config: Mapping[str, Any],
    input_selection: ArtifactRef,
    cell_snapshot: ArtifactRef,
    source_column: str,
    hashing: ArtifactRef | None = None,
    called: bool = False,
) -> ArtifactRef:
    """Filter the run's cells on their frozen QC metrics.

    With ``hashing``, MAD bounds are computed within each hashtag, as
    ``qc.auto_filter(sample_artifact=...)`` computes them. ``called`` says the
    run called these cells from droplets, so a warning about a second
    population does not suggest calling cells first.
    """
    active = read_stored_selection_mask(
        store.zw,
        input_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    attrs = list(config.get("attrs", ()))
    if not attrs:
        return input_selection
    snapshot = artifact_group(store.zw, cell_snapshot)
    values_by_attr: dict[str, np.ndarray] = {}
    missing_by_attr: dict[str, np.ndarray] = {}
    for attr in attrs:
        values, missing = snapshot_column_values(snapshot, attr)
        values_by_attr[attr] = values
        if missing is not None:
            missing_by_attr[attr] = missing
    method = config["method"]
    options: dict[str, Any]
    if method == "manual":
        options = {
            "lows": config["lows"],
            "highs": config["highs"],
            "keep_bounds": config["keepBounds"],
        }
    else:
        options = {
            "min_p": config["minP"],
            "max_p": config["maxP"],
            "n_mads": config["nMads"],
            "min_cells_per_sample": config["minCellsPerSample"],
        }
        sample_column = config["sampleColumn"]
        if method == "mad" and sample_column is not None:
            labels, sample_missing = snapshot_column_values(snapshot, sample_column)
            options.update(
                sample_labels=labels,
                sample_missing=sample_missing,
                sample_label_name=f"sample column {sample_column!r}",
            )
        elif method == "mad" and hashing is not None:
            options.update(
                sample_labels=hashtag_labels(store, hashing, input_selection, active),
                sample_missing=None,
                sample_label_name="hashtag",
            )
    result = filter_cell_metrics(
        values_by_attr,
        missing_by_attr,
        active,
        method=method,
        remedy=CALLED_CELLS_REMEDY if called else RAW_DROPLET_REMEDY,
        **options,
    )
    parameters = dict(config)
    inputs: dict[str, Any] = {
        "input_cell_selection": input_selection,
        "cell_snapshot": cell_snapshot,
    }
    if "sample_labels" in options and hashing is not None:
        # Only a run that filters within hashtags records them, so other
        # runs keep the identity of their filtered cells.
        parameters["sampleSource"] = "hashing"
        inputs["hashing"] = hashing
    if result.gaussian_bounds is not None:
        parameters["resolvedBounds"] = result.gaussian_bounds
    if result.mad_provenance is not None:
        for message in result.mad_provenance["warnings"]:
            logger.warning(message)
        parameters["mad"] = result.mad_provenance
    return resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=result.retained,
        row_ids=store.cells._get_array("ids"),
        operation="filter_pipeline_cells",
        parameters=parameters,
        inputs=inputs,
        source_column=source_column,
    )[0]
