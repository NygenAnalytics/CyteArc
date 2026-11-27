from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from ..metadata.rows import STRING_BYTES, operation_concurrency, row_read_bytes
from ..metadata.selection import resolve_cell_aligned_artifact
from ..storage.arrays import create_zarr_dataset
from ..storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
)
from ..storage.content_stamps import stored_content_stamp_bytes
from ..storage.geometry import array_geometry
from ..storage.partition import row_band, scan_band
from ..storage.selections import validate_cell_selection
from ..storage.types import as_zarr_array
from .parameters import (
    AGGREGATION_ANN_PARAMETER_NAMES,
    AGGREGATION_ANN_STATIC_PARAMETER_NAMES,
)
from ..utils.arguments import RANDOM_STATE_MAX, bool_argument, integer_argument
from ..utils.memory_budget import READ_ALLOWANCE_BYTES, MemoryBudgetError

PSEUDOTIME_INPUTS = frozenset({"connectivity_map", "source_sink", "cell_selection"})
# The potentials that a pseudotime records as its method. A record without a
# method comes from a release before 1.4.0, which computed the truncated one.
PSEUDOTIME_METHODS = ("untruncated", "truncated")
# The parameters that every pseudotime records.
PSEUDOTIME_PARAMETERS = frozenset(
    {"sources", "sinks", "min_max_norm_ptime", "component_policy"}
)
# The parameters of the truncated potential, which a record holds with the
# method "truncated" or with no method.
PSEUDOTIME_TRUNCATED_PARAMETERS = frozenset({"n_singular_vals", "random_seed"})
# The solver bounds that a pseudotime records only when its call set them.
PSEUDOTIME_SOLVER_PARAMETERS = frozenset({"solver_tol", "max_iterations"})
FATE_INPUTS = frozenset(
    {"connectivity_map", "pseudotime", "sink_labels", "cell_selection"}
)
# A fate map directed by a per-cell score records the score in place of a
# pseudotime, beside the graph that its walk takes.
FATE_DIRECTION_INPUTS = frozenset(
    {"connectivity_map", "direction", "sink_labels", "cell_selection"}
)
FATE_PARAMETERS = frozenset({"sinks", "beta", "solver_tol", "max_iterations"})
# A state search records its labels and its eigen-solve's bound only when set.
STATES_INPUTS = frozenset({"connectivity_map", "direction", "cell_selection"})
STATES_OPTIONAL_INPUTS = frozenset({"labels"})
STATES_PARAMETERS = frozenset({"n_states", "cells_per_state", "beta"})
STATES_OPTIONAL_PARAMETERS = frozenset({"max_iterations"})
#: The missing-value mask of a state search's labels.
STATES_LABELS_MISSING = "__cytearc_missing__values"
#: The arrays of a state search: the cell-aligned labels with their mask,
#: memberships and validity, then one row per state and the eigenvalues.
STATES_PAYLOAD = (
    "values",
    STATES_LABELS_MISSING,
    "memberships",
    "valid",
    "state_names",
    "state_labels",
    "roles",
    "coarse_transitions",
    "cell_share",
    "walk_share",
    "direction",
    "cells",
    "eigenvalues",
)
MARKER_INPUTS = frozenset(
    {
        "cell_selection",
        "feature_selection",
        "pseudotime",
        "dataset_fingerprint",
        "ordered_feature_ids_fingerprint",
        "ordered_feature_names_fingerprint",
    }
)
MARKER_PARAMETERS = frozenset(
    {
        "normalization",
        "normalization_method",
        "size_factor",
        "association_method",
        "p_value_method",
        "adjustment_method",
        "adjustment_scope",
        "min_cells",
    }
)
AGGREGATION_INPUTS = MARKER_INPUTS
AGGREGATION_PARAMETERS = frozenset(
    {
        "normalization",
        "normalization_method",
        "size_factor",
        "min_exp",
        "window_size",
        "chunk_size",
        "smoothen",
        "z_scale",
        "n_neighbours",
        "n_clusters",
        "ann_params",
        "nan_cluster_value",
    }
)
PSEUDOTIME_PAYLOAD = ("pseudotime", "valid")
FATE_PAYLOAD = ("probabilities", "valid")
MARKER_PAYLOAD = (
    "r_value",
    "p_value",
    "p_value_adjusted",
    "feature_names",
    "feature_ids",
)
AGGREGATION_PAYLOAD = (
    "data",
    "feature_indices",
    "valid_features",
    "feature_clusters",
    "cluster_values",
    "feature_names",
    "feature_ids",
)
DIFFUSION_PAYLOAD = ("row", "col", "data")
_BASE_ARTIFACT_ATTRIBUTES = frozenset(
    {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "cytearc_version",
        "complete",
    }
)
_RESULT_ATTRIBUTES = _BASE_ARTIFACT_ATTRIBUTES | {"payload_fingerprint"}
DIFFUSION_ATTRIBUTES = _RESULT_ATTRIBUTES | {"n_cells"}
_AGGREGATION_ATTRIBUTES = _RESULT_ATTRIBUTES | {
    "input_fingerprints",
    "nan_cluster_value",
    "effective_window",
    "effective_bins",
}


_MARKER_METHODS = {
    "association_method": "pearson",
    "p_value_method": "student_t",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features",
}


def _real_parameter(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_open: bool = False,
    maximum_open: bool = False,
) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        raise TypeError(f"{name} must be a real number")
    try:
        resolved = float(value)
    except OverflowError:
        # A JSON integer beyond the float range is not a finite real.
        raise ValueError(f"{name} must be finite") from None
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (
        resolved < minimum or (minimum_open and resolved == minimum)
    ):
        qualifier = "greater than" if minimum_open else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}")
    if maximum is not None and (
        resolved > maximum or (maximum_open and resolved == maximum)
    ):
        qualifier = "less than" if maximum_open else "at most"
        raise ValueError(f"{name} must be {qualifier} {maximum}")
    return resolved


def _label_parameters(
    values: Any,
    name: str,
    *,
    required: bool,
) -> tuple[Any, ...]:
    if not isinstance(values, list | tuple):
        raise TypeError(f"{name} must be a list")
    labels: list[Any] = []
    for value in values:
        label = value.item() if isinstance(value, np.generic) else value
        if isinstance(label, float) and not np.isfinite(label):
            raise ValueError(f"{name} must contain finite scalar labels")
        if not isinstance(label, str | bool | int | float):
            raise TypeError(f"{name} must contain scalar labels")
        if any(bool(label == existing) for existing in labels):
            raise ValueError(f"{name} must not contain duplicate labels")
        labels.append(label)
    if required and not labels:
        raise ValueError(f"{name} must contain at least one label")
    return tuple(labels)


def _normalization_parameters(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != {
        "log_transform",
        "renormalize_subset",
    }:
        raise ValueError("normalization parameters are malformed")
    return {
        name: bool_argument(value[name], f"normalization.{name}")
        for name in ("log_transform", "renormalize_subset")
    }


def _normalization_method(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("normalization_method must be a mapping")
    keys = set(value)
    if keys not in (
        {"identity"},
        {"module", "qualname"},
        {"code", "module", "qualname"},
    ) or any(not isinstance(value[key], str) or not value[key] for key in keys):
        raise ValueError("normalization_method is malformed")
    return {key: value[key] for key in sorted(keys)}


def _size_factor(value: Any) -> float | None:
    if value is None:
        return None
    return _real_parameter(value, "size_factor", minimum=0.0, minimum_open=True)


def pseudotime_method(value: Any) -> str:
    """Return a pseudotime's method, ``"untruncated"`` or ``"truncated"``."""
    if not isinstance(value, str) or value not in PSEUDOTIME_METHODS:
        raise ValueError("method must be 'untruncated' or 'truncated'")
    return value


def pseudotime_parameter_names(parameters: Mapping[str, Any]) -> frozenset[str]:
    """Return the parameter names that a pseudotime with ``parameters`` records.

    A record names its method, and holds the parameters of the truncated
    potential with the method ``"truncated"`` or, from a release before
    1.4.0, without a method.
    """
    names = PSEUDOTIME_PARAMETERS | (PSEUDOTIME_SOLVER_PARAMETERS & parameters.keys())
    if "method" in parameters:
        names |= {"method"}
        if parameters["method"] != "truncated":
            return names
    return names | PSEUDOTIME_TRUNCATED_PARAMETERS


def validate_pseudotime_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pseudotime's parameters, recorded or requested.

    ``method`` is ``"truncated"`` for a record without one. ``n_singular_vals``
    and ``random_seed`` are None for the untruncated potential, and
    ``solver_tol`` and ``max_iterations`` are None when absent, as a record
    leaves them out unless its call set them.
    """
    require_exact_record_keys(
        parameters, pseudotime_parameter_names(parameters), "Pseudotime parameters"
    )
    method = pseudotime_method(parameters.get("method", "truncated"))
    truncated = method == "truncated"
    sources = _label_parameters(parameters["sources"], "sources", required=False)
    sinks = _label_parameters(parameters["sinks"], "sinks", required=False)
    if any(bool(source == sink) for source in sources for sink in sinks):
        raise ValueError("sources and sinks must be disjoint")
    component_policy = parameters["component_policy"]
    if component_policy not in {"largest", "sources", "error"}:
        raise ValueError("component_policy must be 'largest', 'sources' or 'error'")
    solver_tol = None
    if "solver_tol" in parameters:
        solver_tol = _real_parameter(
            parameters["solver_tol"],
            "solver_tol",
            minimum=0.0,
            maximum=1.0,
            minimum_open=True,
            maximum_open=True,
        )
    max_iterations = None
    if "max_iterations" in parameters:
        max_iterations = integer_argument(
            parameters["max_iterations"],
            "max_iterations",
            minimum=1,
        )
    return {
        "method": method,
        "n_singular_vals": integer_argument(
            parameters["n_singular_vals"],
            "n_singular_vals",
            minimum=2,
        )
        if truncated
        else None,
        "sources": sources,
        "sinks": sinks,
        "min_max_norm_ptime": bool_argument(
            parameters["min_max_norm_ptime"],
            "min_max_norm_ptime",
        ),
        "random_seed": integer_argument(
            parameters["random_seed"],
            "random_seed",
            minimum=0,
            maximum=RANDOM_STATE_MAX,
        )
        if truncated
        else None,
        "component_policy": component_policy,
        "solver_tol": solver_tol,
        "max_iterations": max_iterations,
    }


def validate_fate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_record_keys(parameters, FATE_PARAMETERS, "Fate-map parameters")
    return {
        "sinks": _label_parameters(parameters["sinks"], "sinks", required=True),
        "beta": _real_parameter(parameters["beta"], "beta", minimum=0.0),
        "solver_tol": _real_parameter(
            parameters["solver_tol"],
            "solver_tol",
            minimum=0.0,
            maximum=1.0,
            minimum_open=True,
            maximum_open=True,
        ),
        "max_iterations": integer_argument(
            parameters["max_iterations"],
            "max_iterations",
            minimum=1,
        ),
    }


def validate_states_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Check a state search's parameters, as the call takes them or its record holds them."""
    expected = STATES_PARAMETERS | (STATES_OPTIONAL_PARAMETERS & set(parameters))
    require_exact_record_keys(
        parameters, frozenset(expected), "State-search parameters"
    )
    validated: dict[str, Any] = {
        "n_states": integer_argument(parameters["n_states"], "n_states", minimum=2),
        "cells_per_state": integer_argument(
            parameters["cells_per_state"], "cells_per_state", minimum=1
        ),
        "beta": _real_parameter(parameters["beta"], "beta", minimum=0.0),
    }
    if "max_iterations" in parameters:
        validated["max_iterations"] = integer_argument(
            parameters["max_iterations"], "max_iterations", minimum=1
        )
    return validated


def validate_marker_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        MARKER_PARAMETERS,
        "Pseudotime-marker parameters",
    )
    validated: dict[str, Any] = {
        "normalization": _normalization_parameters(parameters["normalization"]),
        "normalization_method": _normalization_method(
            parameters["normalization_method"]
        ),
        "size_factor": _size_factor(parameters["size_factor"]),
        "min_cells": integer_argument(
            parameters["min_cells"],
            "min_cells",
            minimum=1,
        ),
    }
    for name, expected in _MARKER_METHODS.items():
        if parameters[name] != expected:
            raise ValueError(f"{name} must be {expected!r}")
        validated[name] = expected
    return validated


def _ann_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("ann_params must be a mapping")
    names = set(value)
    if not names <= AGGREGATION_ANN_PARAMETER_NAMES:
        raise ValueError("ann_params contains unsupported parameters")
    validated: dict[str, Any] = {}
    for name, raw in value.items():
        if name == "space":
            if raw not in {"l2", "ip", "cosine"}:
                raise ValueError("ann_params.space is unsupported")
            validated[name] = raw
        else:
            minimum = 0 if name == "random_seed" else 1
            validated[name] = integer_argument(
                raw,
                f"ann_params.{name}",
                minimum=minimum,
                maximum=(2**32 - 1 if name == "random_seed" else None),
            )
    return validated


def validate_resolved_ann_parameters(
    value: Any,
    *,
    dim: int,
) -> dict[str, Any]:
    validated = _ann_parameters(value)
    required = AGGREGATION_ANN_STATIC_PARAMETER_NAMES | {"dim"}
    if not required <= set(validated) or validated.get("dim") != dim:
        raise ValueError("ann_params does not match the resolved ANN contract")
    return validated


def validate_aggregation_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        AGGREGATION_PARAMETERS,
        "Pseudotime-aggregation parameters",
    )
    n_clusters = integer_argument(
        parameters["n_clusters"],
        "n_clusters",
        minimum=1,
    )
    nan_cluster_value = integer_argument(
        parameters["nan_cluster_value"],
        "nan_cluster_value",
    )
    if 1 <= nan_cluster_value <= n_clusters:
        raise ValueError("nan_cluster_value conflicts with assigned cluster labels")
    return {
        "normalization": _normalization_parameters(parameters["normalization"]),
        "normalization_method": _normalization_method(
            parameters["normalization_method"]
        ),
        "size_factor": _size_factor(parameters["size_factor"]),
        "min_exp": _real_parameter(parameters["min_exp"], "min_exp", minimum=0.0),
        "window_size": integer_argument(
            parameters["window_size"],
            "window_size",
            minimum=1,
        ),
        "chunk_size": integer_argument(
            parameters["chunk_size"],
            "chunk_size",
            minimum=1,
        ),
        "smoothen": bool_argument(parameters["smoothen"], "smoothen"),
        "z_scale": bool_argument(parameters["z_scale"], "z_scale"),
        "n_neighbours": integer_argument(
            parameters["n_neighbours"],
            "n_neighbours",
            minimum=1,
        ),
        "n_clusters": n_clusters,
        "ann_params": _ann_parameters(parameters["ann_params"]),
        "nan_cluster_value": nan_cluster_value,
    }


def require_exact_record_keys(
    values: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(values) != expected:
        raise ValueError(f"{label} does not match its persisted contract")


def selection_size(root: Any, selection: ArtifactRef) -> int:
    return int(validate_cell_selection(root, selection).selected_count)


def true_array_indices(array: Any) -> np.ndarray:
    """Return true-row indices from a non-empty stored boolean vector in blocks."""
    values = as_zarr_array(array, name="boolean values")
    block_rows = row_band(array_geometry(values), unit="chunk", fallback=1)
    parts: list[np.ndarray] = []
    for start in range(0, int(values.shape[0]), block_rows):
        stop = min(start + block_rows, int(values.shape[0]))
        block_indices = np.flatnonzero(
            np.asarray(values[start:stop], dtype=bool)
        ).astype(np.int64, copy=False)
        parts.append(block_indices + start)
    return np.concatenate(parts)


def load_cell_artifact_values(
    root: Any,
    ref: ArtifactRef,
    *,
    value_name: str | None = None,
) -> tuple[np.ndarray, ArtifactRef, np.ndarray | None]:
    """Return a cell artifact's values, its cell selection, and its missing-row mask."""
    resolved = resolve_cell_aligned_artifact(
        root,
        ref,
        value_name=value_name,
        ndim=None,
    )
    return resolved.values, resolved.source_cell_selection, resolved.missing_mask


def listed_labels(
    values: np.ndarray,
    missing: np.ndarray | None,
    listed: Sequence[Any],
    label: str,
) -> pd.Categorical:
    """Keep each cell's label where it is one of ``listed``, as a categorical of those labels.

    A label matches a listed one as pandas matches values, by value, so the
    categorical compares with a label as the labels do; labels outside the
    list and missing labels become missing, and a missing value is never a
    listed label. The categorical holds a code for each cell rather than its
    label. Text is compared as stored, a listed
    label at a time, which matches as pandas does without a Python string
    for each label; other labels are matched by pandas. ``label`` names the
    labels in the refusal of values that are not one-dimensional.
    """
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    categories = pd.Index(
        [value for value in dict.fromkeys(listed) if not pd.isna(value)],
        dtype=object,
    )
    small = np.int8 if len(categories) <= np.iinfo(np.int8).max else np.int32
    if labels.dtype.kind == "U":
        codes = np.full(len(labels), -1, dtype=small)
        for code, category in enumerate(categories):
            codes[labels == category] = code
    else:
        codes = categories.get_indexer(labels).astype(small)
    if missing is not None:
        codes[np.asarray(missing, dtype=bool)] = -1
    return pd.Categorical.from_codes(codes, categories=categories)


def listed_labels_bytes(n_values: int, dtype: Any) -> int:
    """Return the most bytes that :func:`listed_labels` holds beside the labels.

    Text holds the codes and the cells that match one listed label at a
    time, up to five bytes a label. pandas matches other labels as Python
    objects, a number as a new object, with an int64 code and up to 48 bytes
    more for each label while it finds them.
    """
    n_values = int(n_values)
    if np.dtype(dtype).kind == "U":
        return 5 * n_values
    return 56 * n_values


def payload_fingerprint_matches(group: Any, names: Sequence[str]) -> bool:
    if set(group.array_keys()) != set(names) or set(group.group_keys()):
        return False
    expected = group.attrs.get("payload_fingerprint")
    if not isinstance(expected, str) or not expected:
        return False
    try:
        return fingerprint_stored_arrays(group, tuple(names)) == expected
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return False


_MALFORMED_DIFFUSION = "Diffusion-operator sparse payload is malformed"


def _diffusion_payload_arrays(group: zarr.Group, *, n_cells: int) -> list[zarr.Array]:
    raw_n_cells = group.attrs.get("n_cells")
    if (
        set(group.attrs) != DIFFUSION_ATTRIBUTES
        or set(group.array_keys()) != set(DIFFUSION_PAYLOAD)
        or set(group.group_keys())
        or isinstance(raw_n_cells, bool | np.bool_)
        or not isinstance(raw_n_cells, int | np.integer)
        or int(raw_n_cells) != n_cells
    ):
        raise ValueError(_MALFORMED_DIFFUSION)
    arrays = [as_zarr_array(group[name], name=name) for name in DIFFUSION_PAYLOAD]
    rows, cols, data = arrays
    if (
        any(array.ndim != 1 or set(array.attrs) for array in arrays)
        or rows.shape != cols.shape
        or rows.shape != data.shape
        or rows.dtype != np.dtype(np.uint64)
        or cols.dtype != np.dtype(np.uint64)
        or data.dtype != np.dtype(np.float64)
    ):
        raise ValueError(_MALFORMED_DIFFUSION)
    return arrays


# Chunk length of the stored diffusion arrays, and entries written per block.
_DIFFUSION_BLOCK_ENTRIES = 1_000_000


def diffusion_load_bytes(nnz: int, n_cells: int, imputed_features: int = 0) -> int:
    """Return the peak bytes needed to load a persisted diffusion operator.

    Loading reads the three stored arrays and builds a COO matrix. With
    ``imputed_features``, the estimate also covers conversion to CSC and the
    dense imputed output.
    """
    index_bytes = 4 if max(n_cells, nnz) <= np.iinfo(np.int32).max else 8
    coo_bytes = nnz * (8 + 2 * index_bytes)
    load_bytes = nnz * (24 + 2 * index_bytes)
    if imputed_features:
        csc_bytes = nnz * (8 + index_bytes) + (n_cells + 1) * index_bytes
        output_bytes = n_cells * imputed_features * 8
        load_bytes = max(
            load_bytes, coo_bytes + 2 * csc_bytes, 3 * output_bytes + 2 * csc_bytes
        )
    return load_bytes


def diffusion_write_bytes(nnz: int, n_cells: int) -> int:
    """Return the most bytes that writing a diffusion operator and checking it hold beside it.

    :func:`write_diffusion_payload` fingerprints the stored arrays, and the
    check before the operator is published fingerprints them again; each
    reads the arrays back as :func:`diffusion_load_bytes` states, beside a
    stored chunk of 8-byte values, at most one of ``nnz`` entries, and the
    allowance of a read in flight, ``READ_ALLOWANCE_BYTES``.
    """
    return (
        diffusion_load_bytes(nnz, n_cells)
        + 8 * min(int(nnz), _DIFFUSION_BLOCK_ENTRIES)
        + READ_ALLOWANCE_BYTES
    )


def write_diffusion_payload(group: zarr.Group, diffusion: csr_matrix) -> None:
    """Write a square CSR diffusion operator as row, column, and value arrays.

    Entries are written in CSR order, one bounded block at a time, so only
    one block of widened indices is held in memory.
    """
    n_cells = int(diffusion.shape[0])
    nnz = int(diffusion.indptr[-1])
    shape = (nnz,)
    chunks = (_DIFFUSION_BLOCK_ENTRIES,)
    rows = create_zarr_dataset(group, "row", chunks, np.uint64, shape)
    cols = create_zarr_dataset(group, "col", chunks, np.uint64, shape)
    data = create_zarr_dataset(group, "data", chunks, np.float64, shape)
    indptr = np.asarray(diffusion.indptr)
    for start in range(0, nnz, _DIFFUSION_BLOCK_ENTRIES):
        stop = min(start + _DIFFUSION_BLOCK_ENTRIES, nnz)
        positions = np.arange(start, stop, dtype=np.int64)
        rows[start:stop] = (
            np.searchsorted(indptr, positions, side="right") - 1
        ).astype(np.uint64)
        del positions
        cols[start:stop] = np.asarray(diffusion.indices[start:stop], dtype=np.uint64)
        data[start:stop] = np.asarray(diffusion.data[start:stop], dtype=np.float64)
    group.attrs["n_cells"] = n_cells
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group,
        DIFFUSION_PAYLOAD,
    )


def load_diffusion_payload(
    group: zarr.Group,
    *,
    n_cells: int,
    memory_bytes: int,
    imputed_features: int = 0,
) -> coo_matrix:
    arrays = _diffusion_payload_arrays(group, n_cells=n_cells)
    nnz = int(arrays[0].size)
    load_bytes = diffusion_load_bytes(nnz, n_cells, imputed_features)
    if load_bytes >= memory_bytes:
        raise MemoryBudgetError(
            "Diffusion operator and imputed output exceed the memory budget "
            "during loading or sparse conversion; increase the memory budget "
            "or request fewer features per call. They need at least "
            f"{load_bytes + 1} bytes.",
            needed_bytes=load_bytes + 1,
            budget_bytes=int(memory_bytes),
            item="loading the diffusion operator beside the imputed output",
        )
    loaded = [np.asarray(array[:]) for array in arrays]
    builder = ValueFingerprintBuilder()
    for name, values in zip(DIFFUSION_PAYLOAD, loaded, strict=True):
        builder.update_array(name, values)
    row_values, col_values, data_values = loaded
    if (
        builder.hexdigest() != group.attrs.get("payload_fingerprint")
        or not np.isfinite(data_values).all()
        or np.any(data_values < 0)
        or np.any(row_values >= n_cells)
        or np.any(col_values >= n_cells)
    ):
        raise ValueError(_MALFORMED_DIFFUSION)
    return coo_matrix((data_values, (row_values, col_values)), shape=(n_cells, n_cells))


def diffusion_payload_is_valid(group: zarr.Group, *, n_cells: int) -> bool:
    """Validate a reusable payload blockwise without loading the operator."""
    try:
        _diffusion_payload_arrays(group, n_cells=n_cells)
    except (KeyError, TypeError, ValueError):
        return False
    return payload_fingerprint_matches(group, DIFFUSION_PAYLOAD)


def _array_contract(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype_kinds: frozenset[str],
) -> Any | None:
    try:
        array = as_zarr_array(group[name], name=name)
    except (KeyError, TypeError):
        return None
    if tuple(int(value) for value in array.shape) != shape:
        return None
    if np.dtype(array.dtype).kind not in dtype_kinds:
        return None
    if set(array.attrs):
        return None
    return array


def pseudotime_payload_is_valid(
    group: Any,
    *,
    n_cells: int,
    min_max_normalized: bool,
    expected_valid: np.ndarray,
) -> bool:
    if set(group.attrs) != _RESULT_ATTRIBUTES:
        return False
    values_array = _array_contract(
        group,
        "pseudotime",
        shape=(n_cells,),
        dtype_kinds=frozenset({"f"}),
    )
    valid_array = _array_contract(
        group,
        "valid",
        shape=(n_cells,),
        dtype_kinds=frozenset({"b"}),
    )
    if values_array is None or valid_array is None:
        return False
    if not payload_fingerprint_matches(group, PSEUDOTIME_PAYLOAD):
        return False
    expected = np.asarray(expected_valid, dtype=bool)
    block_rows = min(
        row_band(array_geometry(values_array), unit="chunk", fallback=1),
        row_band(array_geometry(valid_array), unit="chunk", fallback=1),
    )
    minimum = np.inf
    maximum = -np.inf
    maximum_absolute = 0.0
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        values = np.asarray(values_array[start:stop], dtype=np.float64)
        valid = np.asarray(valid_array[start:stop], dtype=bool)
        if not np.array_equal(valid, expected[start:stop]):
            return False
        valid_values = values[valid]
        if valid_values.size:
            if not np.isfinite(valid_values).all():
                return False
            minimum = min(minimum, float(valid_values.min()))
            maximum = max(maximum, float(valid_values.max()))
            maximum_absolute = max(
                maximum_absolute,
                float(np.abs(valid_values).max()),
            )
        if np.any(~valid) and not np.isnan(values[~valid]).all():
            return False
    # Without valid cells the range stays at -inf and is rejected below.
    value_range = maximum - minimum
    scale = max(1.0, maximum_absolute)
    if value_range <= np.finfo(np.float64).eps * scale:
        return False
    if min_max_normalized and (
        not np.isclose(minimum, 0.0, rtol=0.0, atol=1e-12)
        or not np.isclose(maximum, 1.0, rtol=0.0, atol=1e-12)
    ):
        return False
    return True


def fate_payload_is_valid(
    group: Any,
    *,
    n_cells: int,
    n_sinks: int,
    expected_valid: np.ndarray,
    sink_values: np.ndarray,
    sink_labels: Sequence[Any],
) -> bool:
    """Check a stored fate map against the cells it must cover and its sinks.

    ``expected_valid`` marks the cells that the map covers: the scored cells
    of the graph components that hold a sink.
    """
    if set(group.attrs) != _RESULT_ATTRIBUTES:
        return False
    if not payload_fingerprint_matches(group, FATE_PAYLOAD):
        return False
    probabilities_array = _array_contract(
        group,
        "probabilities",
        shape=(n_cells, n_sinks),
        dtype_kinds=frozenset({"f"}),
    )
    valid_array = _array_contract(
        group,
        "valid",
        shape=(n_cells,),
        dtype_kinds=frozenset({"b"}),
    )
    if probabilities_array is None or valid_array is None:
        return False
    valid = np.asarray(valid_array[:], dtype=bool)
    expected_valid = np.asarray(expected_valid, dtype=bool)
    if expected_valid.shape != (n_cells,) or not np.array_equal(valid, expected_valid):
        return False
    tolerance = 1e-3
    for sink in sink_labels:
        matches = np.asarray(sink_values == sink, dtype=bool) & expected_valid
        if not matches.any() or not valid[matches].all():
            return False
    block_rows = row_band(
        array_geometry(probabilities_array),
        unit="chunk",
        fallback=1,
    )
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        block = np.asarray(probabilities_array[start:stop], dtype=np.float64)
        block_valid = valid[start:stop]
        valid_probabilities = block[block_valid]
        if (
            valid_probabilities.size
            and (
                not np.isfinite(valid_probabilities).all()
                or float(valid_probabilities.min()) < -tolerance
                or float(valid_probabilities.max()) > 1.0 + tolerance
                or not np.allclose(
                    valid_probabilities.sum(axis=1, dtype=np.float64),
                    1.0,
                    rtol=0.0,
                    atol=tolerance,
                )
            )
        ) or (np.any(~block_valid) and not np.isnan(block[~block_valid]).all()):
            return False
        for column, sink in enumerate(sink_labels):
            block_matches = (
                np.asarray(sink_values[start:stop] == sink, dtype=bool) & block_valid
            )
            if not block_matches.any():
                continue
            expected = np.zeros(n_sinks, dtype=np.float64)
            expected[column] = 1.0
            if not np.allclose(
                block[block_matches],
                expected[np.newaxis, :],
                rtol=0.0,
                atol=tolerance,
            ):
                return False
    return True


def stored_states_bytes(group: Any, *, n_cells: int, n_states: int) -> int:
    """Return the most bytes that checking and loading the stored state search in ``group`` holds.

    The check of :func:`states_payload_is_valid` holds which cells are
    valid and which labeled, and reads the stored arrays a chunk of rows at
    a time: the float32 memberships and their float64 copy, the searched
    cells' memberships, their sums and comparisons, and the labels as text
    and as Python objects, up to ``33 * n_states + 12 * width + 152`` bytes
    a row, the labels' text ``width`` characters wide. Loading then holds
    the labels' text beside the codes that :func:`listed_labels` makes of
    them, or the float32 memberships and their read-only copy, beside 6
    bytes a cell for the codes, the valid cells and their copy. Either holds
    the allowance of a read in flight, ``READ_ALLOWANCE_BYTES``. The width
    and the rows of a chunk are read from the stored arrays' metadata.
    """
    n, m = int(n_cells), int(n_states)
    stored_memberships = group.get("memberships")
    rows = (
        min(n, row_band(array_geometry(stored_memberships), unit="chunk", fallback=1))
        if isinstance(stored_memberships, zarr.Array)
        else n
    )
    stored_labels = group.get("values")
    width = (
        max(1, np.dtype(stored_labels.dtype).itemsize // 4)
        if isinstance(stored_labels, zarr.Array)
        and np.dtype(stored_labels.dtype).kind == "U"
        else 1
    )
    checking = 2 * n + rows * (33 * m + 12 * width + 152)
    loading = max(4 * width * n, 8 * n * m) + 6 * n
    return max(checking, loading) + READ_ALLOWANCE_BYTES


def states_payload_is_valid(
    group: Any,
    *,
    n_cells: int,
    n_states: int,
    expected_valid: np.ndarray | None = None,
) -> bool:
    """Check a stored state search against its cells, its states and their roles.

    ``expected_valid`` marks the cells that the search must cover, the
    largest connected component of scored cells, or is None to accept the
    cells that the artifact marks. Every searched cell's memberships are
    finite and sum to one, every other cell's are NaN, and a label names a
    state and marks only searched cells.
    """
    if set(group.attrs) != _RESULT_ATTRIBUTES | {"diagnostics"}:
        return False
    if not payload_fingerprint_matches(group, STATES_PAYLOAD):
        return False
    contracts = {
        "memberships": ((n_cells, n_states), "f"),
        "valid": ((n_cells,), "b"),
        STATES_LABELS_MISSING: ((n_cells,), "b"),
        "coarse_transitions": ((n_states, n_states), "f"),
        "cell_share": ((n_states,), "f"),
        "walk_share": ((n_states,), "f"),
        "direction": ((n_states,), "f"),
        "cells": ((n_states,), "i"),
        "eigenvalues": ((n_states + 1, 2), "f"),
    }
    for name, (shape, kind) in contracts.items():
        if (
            _array_contract(group, name, shape=shape, dtype_kinds=frozenset({kind}))
            is None
        ):
            return False
    # The fingerprint's check found exactly the payload's arrays.
    labels = as_zarr_array(group["values"], name="values")
    names_array = as_zarr_array(group["state_names"], name="state_names")
    labels_array = as_zarr_array(group["state_labels"], name="state_labels")
    roles_array = as_zarr_array(group["roles"], name="roles")
    if (
        tuple(labels.shape) != (n_cells,)
        or dict(labels.attrs) != {"missing_mask": STATES_LABELS_MISSING}
        or any(
            tuple(array.shape) != (n_states,)
            for array in (names_array, labels_array, roles_array)
        )
        or any(
            np.dtype(array.dtype).kind != "U"
            for array in (labels, names_array, labels_array, roles_array)
        )
    ):
        return False
    from .states import state_names

    names = [str(name) for name in np.asarray(names_array[:])]
    roles = [str(role) for role in np.asarray(roles_array[:])]
    if (
        # Each state's name is its label, or its number, with a suffix that
        # tells apart the states that share one.
        tuple(names)
        != state_names([str(label) for label in np.asarray(labels_array[:])])
        or roles[0] != "initial"
        or any(role not in ("intermediate", "terminal") for role in roles[1:])
    ):
        return False
    valid = np.asarray(group["valid"][:], dtype=bool)
    if expected_valid is not None and not np.array_equal(
        valid, np.asarray(expected_valid, dtype=bool)
    ):
        return False
    if not valid.any():
        return False
    missing = np.asarray(group[STATES_LABELS_MISSING][:], dtype=bool)
    memberships = as_zarr_array(group["memberships"], name="memberships")
    known = np.asarray(names, dtype=object)
    block_rows = row_band(array_geometry(memberships), unit="chunk", fallback=1)
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        block = np.asarray(memberships[start:stop], dtype=np.float64)
        block_valid = valid[start:stop]
        inside = block[block_valid]
        if inside.size and (
            not np.isfinite(inside).all()
            or float(inside.min()) < -1e-6
            or not np.allclose(inside.sum(axis=1), 1.0, rtol=0.0, atol=1e-4)
        ):
            return False
        if np.any(~block_valid) and not np.isnan(block[~block_valid]).all():
            return False
        block_missing = missing[start:stop]
        if np.any(~block_missing & ~block_valid):
            return False
        named = np.asarray(labels[start:stop])[~block_missing].astype(object)
        if named.size and not np.isin(named, known).all():
            return False
    return True


def marker_payload_is_valid(
    group: Any,
    *,
    n_features: int,
    selected_features: np.ndarray,
    expected_feature_ids_fingerprint: str,
    expected_feature_names_fingerprint: str,
) -> bool:
    """Check a stored marker payload against the selected features.

    ``selected_features`` are the indices of a feature selection, so they are
    unique and smaller than ``n_features``.
    """
    if set(group.attrs) != _RESULT_ATTRIBUTES:
        return False
    if not payload_fingerprint_matches(group, MARKER_PAYLOAD):
        return False
    stat_arrays = [
        _array_contract(
            group,
            name,
            shape=(n_features,),
            dtype_kinds=frozenset({"f"}),
        )
        for name in MARKER_PAYLOAD[:3]
    ]
    names_array = _array_contract(
        group,
        "feature_names",
        shape=(n_features,),
        dtype_kinds=frozenset({"O", "S", "T", "U"}),
    )
    ids_array = _array_contract(
        group,
        "feature_ids",
        shape=(n_features,),
        dtype_kinds=frozenset({"O", "S", "T", "U"}),
    )
    if (
        any(array is None for array in stat_arrays)
        or names_array is None
        or ids_array is None
    ):
        return False
    r_array, p_array, adjusted_array = stat_arrays
    assert r_array is not None
    assert p_array is not None
    assert adjusted_array is not None
    try:
        identities_match = (
            fingerprint_stored_strings(ids_array) == expected_feature_ids_fingerprint
            and fingerprint_stored_strings(names_array)
            == expected_feature_names_fingerprint
        )
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    if not identities_match:
        return False
    selected_indices = np.asarray(selected_features, dtype=np.int64)
    block_rows = min(
        row_band(array_geometry(array), unit="chunk", fallback=1)
        for array in (
            r_array,
            p_array,
            adjusted_array,
        )
    )
    for start in range(0, n_features, block_rows):
        stop = min(start + block_rows, n_features)
        selected = np.isin(
            np.arange(start, stop, dtype=np.int64),
            selected_indices,
            assume_unique=True,
        )
        r_values = np.asarray(r_array[start:stop], dtype=np.float64)
        p_values = np.asarray(p_array[start:stop], dtype=np.float64)
        adjusted = np.asarray(adjusted_array[start:stop], dtype=np.float64)
        if (
            not np.isfinite(r_values[selected]).all()
            or np.any(np.abs(r_values[selected]) > 1.0 + 1e-12)
            or not np.isnan(r_values[~selected]).all()
            or not np.isnan(p_values[~selected]).all()
            or not np.isnan(adjusted[~selected]).all()
        ):
            return False
        for values in (p_values[selected], adjusted[selected]):
            if np.isinf(values).any():
                return False
            finite = values[np.isfinite(values)]
            if len(finite) and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
                return False
    return True


def aggregation_payload_is_valid(
    group: Any,
    *,
    n_features: int,
    selected_features: np.ndarray,
    n_bins: int,
    n_clusters: int,
    n_neighbors: int,
    nan_cluster_value: int,
    ann_params: Mapping[str, Any],
    expected_input_fingerprints: Sequence[str],
    expected_feature_ids_fingerprint: str,
    expected_feature_names_fingerprint: str,
    effective_window: int,
) -> bool:
    """Check a stored aggregation payload against the selected features.

    ``selected_features`` are the indices of a feature selection, so they
    increase strictly and are smaller than ``n_features``.
    """
    try:
        ann_params = validate_resolved_ann_parameters(ann_params, dim=n_bins)
    except (TypeError, ValueError):
        return False
    if set(group.attrs) != _AGGREGATION_ATTRIBUTES:
        return False
    if not payload_fingerprint_matches(group, AGGREGATION_PAYLOAD):
        return False
    n_selected = len(selected_features)
    data_array = _array_contract(
        group,
        "data",
        shape=(n_selected, n_bins),
        dtype_kinds=frozenset({"f"}),
    )
    indices_array = _array_contract(
        group,
        "feature_indices",
        shape=(n_selected,),
        dtype_kinds=frozenset({"u"}),
    )
    valid_array = _array_contract(
        group,
        "valid_features",
        shape=(n_selected,),
        dtype_kinds=frozenset({"b"}),
    )
    clusters_array = _array_contract(
        group,
        "feature_clusters",
        shape=(n_selected,),
        dtype_kinds=frozenset({"i", "u"}),
    )
    cluster_values_array = _array_contract(
        group,
        "cluster_values",
        shape=(n_features,),
        dtype_kinds=frozenset({"i", "u"}),
    )
    names_array = _array_contract(
        group,
        "feature_names",
        shape=(n_features,),
        dtype_kinds=frozenset({"O", "S", "T", "U"}),
    )
    ids_array = _array_contract(
        group,
        "feature_ids",
        shape=(n_features,),
        dtype_kinds=frozenset({"O", "S", "T", "U"}),
    )
    if any(
        array is None
        for array in (
            data_array,
            indices_array,
            valid_array,
            clusters_array,
            cluster_values_array,
            names_array,
            ids_array,
        )
    ):
        return False
    assert data_array is not None
    assert indices_array is not None
    assert valid_array is not None
    assert clusters_array is not None
    assert cluster_values_array is not None
    assert names_array is not None
    assert ids_array is not None
    if group.attrs.get("input_fingerprints") != list(expected_input_fingerprints):
        return False
    if group.attrs.get("nan_cluster_value") != nan_cluster_value:
        return False
    if group.attrs.get("effective_window") != effective_window:
        return False
    if group.attrs.get("effective_bins") != n_bins:
        return False
    expected_indices = np.asarray(selected_features, dtype=np.int64)
    try:
        identities_match = (
            fingerprint_stored_strings(ids_array) == expected_feature_ids_fingerprint
            and fingerprint_stored_strings(names_array)
            == expected_feature_names_fingerprint
        )
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    if not identities_match:
        return False

    selected_block_rows = min(
        row_band(array_geometry(array), unit="chunk", fallback=1)
        for array in (data_array, indices_array, valid_array, clusters_array)
    )
    valid_count = 0
    for start in range(0, n_selected, selected_block_rows):
        stop = min(start + selected_block_rows, n_selected)
        feature_indices = np.asarray(indices_array[start:stop], dtype=np.int64)
        if not np.array_equal(feature_indices, expected_indices[start:stop]):
            return False
        valid = np.asarray(valid_array[start:stop], dtype=bool)
        clusters = np.asarray(clusters_array[start:stop], dtype=np.int64)
        data = np.asarray(data_array[start:stop], dtype=np.float64)
        valid_count += int(np.count_nonzero(valid))
        if (
            not np.isfinite(data[valid]).all()
            or np.any(data[~valid] != 0.0)
            or np.any(clusters[~valid] != nan_cluster_value)
            or np.any(clusters[valid] < 1)
            or np.any(clusters[valid] > n_clusters)
        ):
            return False
    if (
        valid_count < 2
        or not 1 <= n_neighbors < valid_count
        or not 1 <= n_clusters <= valid_count
    ):
        return False
    if "max_elements" in ann_params and ann_params["max_elements"] < valid_count:
        return False

    feature_block_rows = row_band(
        array_geometry(cluster_values_array),
        unit="chunk",
        fallback=1,
    )
    for start in range(0, n_features, feature_block_rows):
        stop = min(start + feature_block_rows, n_features)
        expected_clusters = np.full(
            stop - start,
            nan_cluster_value,
            dtype=np.int64,
        )
        left = int(np.searchsorted(expected_indices, start, side="left"))
        right = int(np.searchsorted(expected_indices, stop, side="left"))
        if right > left:
            selected_valid = np.asarray(valid_array[left:right], dtype=bool)
            selected_clusters = np.asarray(
                clusters_array[left:right],
                dtype=np.int64,
            )
            positions = expected_indices[left:right][selected_valid] - start
            expected_clusters[positions] = selected_clusters[selected_valid]
        if not np.array_equal(
            np.asarray(cluster_values_array[start:stop], dtype=np.int64),
            expected_clusters,
        ):
            return False
    return True


LINEAGE_FEATURE_INPUTS = frozenset(
    {
        "cell_selection",
        "feature_selection",
        "fate",
        "dataset_fingerprint",
        "ordered_feature_ids_fingerprint",
        "ordered_feature_names_fingerprint",
    }
)
LINEAGE_MARKER_INPUTS = LINEAGE_FEATURE_INPUTS
LINEAGE_MARKER_PARAMETERS = frozenset(
    {
        "normalization",
        "normalization_method",
        "size_factor",
        "lineages",
        "association_method",
        "p_value_method",
        "adjustment_method",
        "adjustment_scope",
        "min_cells",
    }
)
LINEAGE_DRIVER_INPUTS = LINEAGE_FEATURE_INPUTS | {"driver_cells"}
LINEAGE_DRIVER_PARAMETERS = frozenset(
    {
        "normalization",
        "normalization_method",
        "size_factor",
        "lineages",
        "include_sinks",
        "min_cells",
        "confidence_level",
        "association_method",
        "p_value_method",
        "adjustment_method",
        "adjustment_scope",
    }
)
LINEAGE_AGGREGATION_INPUTS = LINEAGE_FEATURE_INPUTS
LINEAGE_AGGREGATION_PARAMETERS = frozenset(
    {
        "normalization",
        "normalization_method",
        "size_factor",
        "lineages",
        "n_bins",
        "smoothing",
        "min_exp",
        "z_scale",
        "n_neighbours",
        "n_clusters",
        "ann_params",
        "nan_cluster_value",
    }
)
LINEAGE_CURVE_INPUTS = frozenset({"fate", "coordinates", "cell_selection"})
LINEAGE_CURVE_PARAMETERS = frozenset(
    {"lineages", "n_points", "df", "fit_bins", "spline_basis"}
)
LINEAGE_TREE_INPUTS = frozenset({"labels", "coordinates", "cell_selection"})
LINEAGE_TREE_PARAMETERS = frozenset({"root", "ends"})
LINEAGE_MEMBERSHIP_INPUTS = frozenset({"curves", "cell_selection"})
LINEAGE_MEMBERSHIP_PARAMETERS = frozenset({"band", "distance_score", "n_bins"})
LINEAGE_MARKER_PAYLOAD = (
    "r_value",
    "p_value",
    "p_value_adjusted",
    "effective_cells",
    "feature_names",
    "feature_ids",
)
LINEAGE_DRIVER_PAYLOAD = (
    "r_value",
    "p_value",
    "p_value_adjusted",
    "ci_low",
    "ci_high",
    "feature_names",
    "feature_ids",
)
LINEAGE_AGGREGATION_PAYLOAD = (
    "data",
    "feature_indices",
    "valid_features",
    "feature_clusters",
    "cluster_values",
    "bin_pseudotime",
    "lineage_weight",
    "feature_names",
    "feature_ids",
)
LINEAGE_CURVE_PAYLOAD = ("points", "curve_pseudotime", "progress", "distance", "valid")
LINEAGE_TREE_PAYLOAD = ("sizes", "means", "edges", "lengths")
LINEAGE_MEMBERSHIP_PAYLOAD = (
    "values",
    "members",
    "valid",
    "distance_limits",
    "branch_pairs",
    "branch_pseudotime",
    "branch_progress",
    "branch_cells",
    "trunk_components",
    "trunk_probabilities",
)
LINEAGE_RESULT_ATTRIBUTES = _RESULT_ATTRIBUTES
LINEAGE_DRIVER_ATTRIBUTES = _RESULT_ATTRIBUTES | {"n_cells"}
LINEAGE_AGGREGATION_ATTRIBUTES = _RESULT_ATTRIBUTES | {"input_fingerprints"}
LINEAGE_TREE_ATTRIBUTES = _RESULT_ATTRIBUTES | {"cluster_labels"}

LINEAGE_MARKER_METHODS = {
    "association_method": "fate_weighted_pearson",
    "p_value_method": "student_t_kish",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features_per_lineage",
}
LINEAGE_DRIVER_METHODS = {
    "association_method": "pearson",
    "p_value_method": "fisher_z",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features_per_lineage",
}


def _fixed_methods(
    parameters: Mapping[str, Any],
    methods: Mapping[str, str],
) -> dict[str, str]:
    for name, expected in methods.items():
        if parameters[name] != expected:
            raise ValueError(f"{name} must be {expected!r}")
    return dict(methods)


def _feature_analysis_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "normalization": _normalization_parameters(parameters["normalization"]),
        "normalization_method": _normalization_method(
            parameters["normalization_method"]
        ),
        "size_factor": _size_factor(parameters["size_factor"]),
        "lineages": _label_parameters(
            parameters["lineages"], "lineages", required=True
        ),
    }


def validate_lineage_marker_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        LINEAGE_MARKER_PARAMETERS,
        "Lineage-marker parameters",
    )
    return {
        **_feature_analysis_parameters(parameters),
        "min_cells": _real_parameter(
            parameters["min_cells"], "min_cells", minimum=0.0, minimum_open=True
        ),
        **_fixed_methods(parameters, LINEAGE_MARKER_METHODS),
    }


def validate_lineage_driver_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        LINEAGE_DRIVER_PARAMETERS,
        "Lineage-driver parameters",
    )
    return {
        **_feature_analysis_parameters(parameters),
        "include_sinks": bool_argument(parameters["include_sinks"], "include_sinks"),
        "min_cells": integer_argument(parameters["min_cells"], "min_cells", minimum=1),
        "confidence_level": _real_parameter(
            parameters["confidence_level"],
            "confidence_level",
            minimum=0.0,
            maximum=1.0,
            minimum_open=True,
            maximum_open=True,
        ),
        **_fixed_methods(parameters, LINEAGE_DRIVER_METHODS),
    }


def validate_lineage_aggregation_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        LINEAGE_AGGREGATION_PARAMETERS,
        "Lineage-aggregation parameters",
    )
    n_clusters = integer_argument(parameters["n_clusters"], "n_clusters", minimum=1)
    nan_cluster_value = integer_argument(
        parameters["nan_cluster_value"], "nan_cluster_value"
    )
    if 1 <= nan_cluster_value <= n_clusters:
        raise ValueError("nan_cluster_value conflicts with assigned cluster labels")
    return {
        **_feature_analysis_parameters(parameters),
        "n_bins": integer_argument(parameters["n_bins"], "n_bins", minimum=2),
        "smoothing": _real_parameter(parameters["smoothing"], "smoothing", minimum=0.0),
        "min_exp": _real_parameter(parameters["min_exp"], "min_exp", minimum=0.0),
        "z_scale": bool_argument(parameters["z_scale"], "z_scale"),
        "n_neighbours": integer_argument(
            parameters["n_neighbours"], "n_neighbours", minimum=1
        ),
        "n_clusters": n_clusters,
        "ann_params": _ann_parameters(parameters["ann_params"]),
        "nan_cluster_value": nan_cluster_value,
    }


def validate_lineage_curve_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        LINEAGE_CURVE_PARAMETERS,
        "Lineage-curve parameters",
    )
    spline_basis = integer_argument(
        parameters["spline_basis"], "spline_basis", minimum=8, maximum=4096
    )
    return {
        "lineages": _label_parameters(
            parameters["lineages"], "lineages", required=True
        ),
        "n_points": integer_argument(parameters["n_points"], "n_points", minimum=2),
        "df": _real_parameter(
            parameters["df"],
            "df",
            minimum=2.0,
            maximum=float(spline_basis),
            minimum_open=True,
            maximum_open=True,
        ),
        "fit_bins": integer_argument(parameters["fit_bins"], "fit_bins", minimum=16),
        "spline_basis": spline_basis,
    }


def validate_lineage_membership_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the rule of a lineage membership: its band, distance score and bins."""
    require_exact_record_keys(
        parameters,
        LINEAGE_MEMBERSHIP_PARAMETERS,
        "Lineage-membership parameters",
    )
    return {
        "band": _real_parameter(
            parameters["band"], "band", minimum=0.0, maximum=1.0, minimum_open=True
        ),
        "distance_score": _real_parameter(
            parameters["distance_score"],
            "distance_score",
            minimum=0.0,
            minimum_open=True,
        ),
        "n_bins": integer_argument(
            parameters["n_bins"], "n_bins", minimum=1, maximum=1000
        ),
    }


def validate_lineage_tree_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_record_keys(
        parameters,
        LINEAGE_TREE_PARAMETERS,
        "Lineage-tree parameters",
    )
    raw_root = parameters["root"]
    root = (
        None
        if raw_root is None
        else _label_parameters([raw_root], "root", required=True)[0]
    )
    ends = _label_parameters(parameters["ends"], "ends", required=False)
    if root is not None and any(bool(root == end) for end in ends):
        raise ValueError("The root cannot also be an end")
    return {"root": root, "ends": ends}


def _identity_arrays_match(
    group: Any,
    *,
    n_features: int,
    expected_feature_ids_fingerprint: str,
    expected_feature_names_fingerprint: str,
) -> bool:
    names_array = _array_contract(
        group,
        "feature_names",
        shape=(n_features,),
        dtype_kinds=frozenset({"O", "S", "T", "U"}),
    )
    ids_array = _array_contract(
        group,
        "feature_ids",
        shape=(n_features,),
        dtype_kinds=frozenset({"O", "S", "T", "U"}),
    )
    if names_array is None or ids_array is None:
        return False
    try:
        return (
            fingerprint_stored_strings(ids_array) == expected_feature_ids_fingerprint
            and fingerprint_stored_strings(names_array)
            == expected_feature_names_fingerprint
        )
    except (TypeError, ValueError, UnicodeDecodeError):
        return False


def _feature_statistics_valid(
    arrays: Sequence[Any],
    *,
    n_features: int,
    selected_features: np.ndarray,
    correlations: frozenset[int],
    probabilities: frozenset[int],
) -> bool:
    """Check per-feature statistics, features by lineages, blockwise.

    Rows of features outside the selection must be missing. The arrays at
    the positions in ``correlations`` hold values in [-1, 1] and those in
    ``probabilities`` values in [0, 1]; untested entries may be missing.
    """
    selected_indices = np.asarray(selected_features, dtype=np.int64)
    block_rows = min(
        row_band(array_geometry(array), unit="chunk", fallback=1) for array in arrays
    )
    for start in range(0, n_features, block_rows):
        stop = min(start + block_rows, n_features)
        selected = np.isin(
            np.arange(start, stop, dtype=np.int64),
            selected_indices,
            assume_unique=True,
        )
        for position, array in enumerate(arrays):
            values = np.asarray(array[start:stop], dtype=np.float64)
            if np.isinf(values).any() or not np.isnan(values[~selected]).all():
                return False
            finite = values[selected][np.isfinite(values[selected])]
            if position in correlations and np.any(np.abs(finite) > 1.0 + 1e-12):
                return False
            if (
                position in probabilities
                and len(finite)
                and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0)
            ):
                return False
    return True


def lineage_marker_payload_is_valid(
    group: Any,
    *,
    n_features: int,
    n_lineages: int,
    selected_features: np.ndarray,
    expected_feature_ids_fingerprint: str,
    expected_feature_names_fingerprint: str,
) -> bool:
    """Check a stored lineage-marker payload against its features and lineages."""
    if set(group.attrs) != LINEAGE_RESULT_ATTRIBUTES:
        return False
    if not payload_fingerprint_matches(group, LINEAGE_MARKER_PAYLOAD):
        return False
    stat_arrays = [
        _array_contract(
            group,
            name,
            shape=(n_features, n_lineages),
            dtype_kinds=frozenset({"f"}),
        )
        for name in LINEAGE_MARKER_PAYLOAD[:3]
    ]
    sizes = _array_contract(
        group,
        "effective_cells",
        shape=(n_lineages,),
        dtype_kinds=frozenset({"f"}),
    )
    if any(array is None for array in stat_arrays) or sizes is None:
        return False
    effective = np.asarray(sizes[:], dtype=np.float64)
    if not np.isfinite(effective).all() or np.any(effective <= 0.0):
        return False
    if not _identity_arrays_match(
        group,
        n_features=n_features,
        expected_feature_ids_fingerprint=expected_feature_ids_fingerprint,
        expected_feature_names_fingerprint=expected_feature_names_fingerprint,
    ):
        return False
    return _feature_statistics_valid(
        stat_arrays,
        n_features=n_features,
        selected_features=selected_features,
        correlations=frozenset({0}),
        probabilities=frozenset({1, 2}),
    )


def lineage_driver_payload_is_valid(
    group: Any,
    *,
    n_features: int,
    n_lineages: int,
    n_cells: int,
    selected_features: np.ndarray,
    expected_feature_ids_fingerprint: str,
    expected_feature_names_fingerprint: str,
) -> bool:
    """Check a stored lineage-driver payload against its features and cells."""
    if set(group.attrs) != LINEAGE_DRIVER_ATTRIBUTES:
        return False
    raw_cells = group.attrs.get("n_cells")
    if (
        isinstance(raw_cells, bool | np.bool_)
        or not isinstance(raw_cells, int | np.integer)
        or int(raw_cells) != n_cells
    ):
        return False
    if not payload_fingerprint_matches(group, LINEAGE_DRIVER_PAYLOAD):
        return False
    stat_arrays = [
        _array_contract(
            group,
            name,
            shape=(n_features, n_lineages),
            dtype_kinds=frozenset({"f"}),
        )
        for name in LINEAGE_DRIVER_PAYLOAD[:5]
    ]
    if any(array is None for array in stat_arrays):
        return False
    if not _identity_arrays_match(
        group,
        n_features=n_features,
        expected_feature_ids_fingerprint=expected_feature_ids_fingerprint,
        expected_feature_names_fingerprint=expected_feature_names_fingerprint,
    ):
        return False
    return _feature_statistics_valid(
        stat_arrays,
        n_features=n_features,
        selected_features=selected_features,
        correlations=frozenset({0, 3, 4}),
        probabilities=frozenset({1, 2}),
    )


def lineage_aggregation_payload_is_valid(
    group: Any,
    *,
    n_features: int,
    selected_features: np.ndarray,
    n_lineages: int,
    n_bins: int,
    n_clusters: int,
    n_neighbors: int,
    nan_cluster_value: int,
    expected_input_fingerprints: Sequence[str],
    expected_feature_ids_fingerprint: str,
    expected_feature_names_fingerprint: str,
) -> bool:
    """Check a stored lineage aggregation against its features and bins.

    ``selected_features`` are the increasing indices of a feature selection.
    """
    if set(group.attrs) != LINEAGE_AGGREGATION_ATTRIBUTES:
        return False
    if group.attrs.get("input_fingerprints") != list(expected_input_fingerprints):
        return False
    if not payload_fingerprint_matches(group, LINEAGE_AGGREGATION_PAYLOAD):
        return False
    n_selected = len(selected_features)
    data_array = _array_contract(
        group,
        "data",
        shape=(n_selected, n_lineages, n_bins),
        dtype_kinds=frozenset({"f"}),
    )
    indices_array = _array_contract(
        group,
        "feature_indices",
        shape=(n_selected,),
        dtype_kinds=frozenset({"u"}),
    )
    valid_array = _array_contract(
        group,
        "valid_features",
        shape=(n_selected,),
        dtype_kinds=frozenset({"b"}),
    )
    clusters_array = _array_contract(
        group,
        "feature_clusters",
        shape=(n_selected,),
        dtype_kinds=frozenset({"i"}),
    )
    cluster_values_array = _array_contract(
        group,
        "cluster_values",
        shape=(n_features,),
        dtype_kinds=frozenset({"i"}),
    )
    ordering_array = _array_contract(
        group,
        "bin_pseudotime",
        shape=(n_lineages, n_bins),
        dtype_kinds=frozenset({"f"}),
    )
    weight_array = _array_contract(
        group,
        "lineage_weight",
        shape=(n_lineages,),
        dtype_kinds=frozenset({"f"}),
    )
    if (
        data_array is None
        or indices_array is None
        or valid_array is None
        or clusters_array is None
        or cluster_values_array is None
        or ordering_array is None
        or weight_array is None
    ):
        return False
    if not _identity_arrays_match(
        group,
        n_features=n_features,
        expected_feature_ids_fingerprint=expected_feature_ids_fingerprint,
        expected_feature_names_fingerprint=expected_feature_names_fingerprint,
    ):
        return False
    bin_ordering = np.asarray(ordering_array[:], dtype=np.float64)
    weights = np.asarray(weight_array[:], dtype=np.float64)
    if (
        not np.isfinite(bin_ordering).all()
        or np.any(np.diff(bin_ordering, axis=1) < -1e-12)
        or not np.isfinite(weights).all()
        or np.any(weights < n_bins)
    ):
        return False
    expected_indices = np.asarray(selected_features, dtype=np.int64)
    if not np.array_equal(
        np.asarray(indices_array[:], dtype=np.int64), expected_indices
    ):
        return False
    valid = np.asarray(valid_array[:], dtype=bool)
    clusters = np.asarray(clusters_array[:], dtype=np.int64)
    valid_count = int(np.count_nonzero(valid))
    if (
        valid_count < 2
        or not 1 <= n_neighbors < valid_count
        or not 1 <= n_clusters <= valid_count
        or np.any(clusters[~valid] != nan_cluster_value)
        or np.any(clusters[valid] < 1)
        or np.any(clusters[valid] > n_clusters)
    ):
        return False
    block_rows = row_band(array_geometry(data_array), unit="chunk", fallback=1)
    for start in range(0, n_selected, block_rows):
        stop = min(start + block_rows, n_selected)
        data = np.asarray(data_array[start:stop], dtype=np.float64)
        block_valid = valid[start:stop]
        if not np.isfinite(data[block_valid]).all() or np.any(
            data[~block_valid] != 0.0
        ):
            return False
    expected_clusters = np.full(n_features, nan_cluster_value, dtype=np.int64)
    expected_clusters[expected_indices[valid]] = clusters[valid]
    return np.array_equal(
        np.asarray(cluster_values_array[:], dtype=np.int64),
        expected_clusters,
    )


# What checking and loading a saved feature analysis holds


@dataclass
class _Steps:
    """The most bytes that steps run in turn hold.

    Each step builds what it lets go beside what the steps before it keep,
    and keeps what it adds to them.
    """

    held: int = 0
    most: int = 0

    def step(self, building: int, keeping: int = 0) -> None:
        self.most = max(self.most, self.held + int(building) + int(keeping))
        self.held += int(keeping)


def _payload_arrays(group: Any, names: Sequence[str]) -> dict[str, zarr.Array]:
    """The stored arrays among ``names`` of a payload; its check refuses one that is missing."""
    arrays = {}
    for name in names:
        try:
            arrays[name] = as_zarr_array(group[name], name=name)
        except (KeyError, TypeError):
            continue
    return arrays


def _read_bytes(array: zarr.Array, rows: int) -> int:
    """What reading ``rows`` rows of a stored array from a chunk's first row holds, with them.

    The read holds the rows and the chunks that Zarr decodes at once, each
    as stored, at the bound of its compressed bytes, and decoded, as the
    memory budget charges a read.
    """
    geometry = array_geometry(array)
    assert geometry is not None
    rows = min(max(0, int(rows)), geometry.shape[0])
    touched = max(1, -(-rows // geometry.axisChunk(0))) * geometry.bandChunks()
    return geometry.readBytes(
        rows * geometry.rowBytes(), chunks=touched, decodes=operation_concurrency()
    )


def _chunk_rows(array: zarr.Array) -> int:
    return row_band(array_geometry(array), unit="chunk", fallback=1)


def _row_bytes(array: zarr.Array) -> int:
    """The bytes of one row of a stored array, every value of its later axes."""
    return int(np.prod(array.shape[1:], dtype=np.int64)) * array.dtype.itemsize


def _check_bytes(arrays: Mapping[str, zarr.Array], *steps: int) -> int:
    """What a check of a saved feature analysis holds: the most of its steps, which it takes in turn.

    Every check first fingerprints the payload's arrays, one array's scan
    band at a time as ``fingerprint_stored_arrays`` reads them, and stamps
    the stored bytes of the frozen feature names and IDs, whose fingerprints
    are reused while those bytes are unchanged. ``steps`` are what the check
    then holds, which a payload that lacks one of its arrays never reaches.
    """
    fingerprint = max(
        (
            _read_bytes(
                array,
                scan_band(array_geometry(array), fallback=max(1, array.shape[0])),
            )
            for array in arrays.values()
        ),
        default=0,
    )
    stamps = max(
        (
            stored_content_stamp_bytes(arrays[name])
            for name in ("feature_names", "feature_ids")
            if name in arrays
        ),
        default=0,
    )
    return max(fingerprint, stamps, *steps)


def _selection_bytes(rows: int) -> int:
    """What ``np.isin`` holds to mark which of ``rows`` consecutive features are selected, with their positions.

    NumPy marks the selected features in a table as long as their range,
    about 28 bytes a feature of the block. It sorts them instead only in an
    assay many blocks long, whose fingerprint then holds more.
    """
    return 32 * int(rows)


def _statistics_bytes(
    arrays: Sequence[zarr.Array], n_selected: int, *, blocks: int, copies: int
) -> int:
    """What checking per-feature statistics holds, a block of their shortest stored chunks at a time.

    Each block marks which of its features are selected, then reads each
    array's block beside ``blocks`` blocks and ``copies`` copies of a
    block's selected features that the check keeps while it reads one,
    those of the block before it among them.
    """
    rows = min(_chunk_rows(array) for array in arrays)
    row_bytes = max(_row_bytes(array) for array in arrays)
    kept = blocks * rows * row_bytes + copies * min(rows, int(n_selected)) * row_bytes
    reading = max(_read_bytes(array, rows) for array in arrays)
    return kept + max(reading, _selection_bytes(rows))


def marker_check_bytes(group: Any, *, n_selected: int) -> int:
    """Return the most bytes that :func:`marker_payload_is_valid` holds.

    Beside fingerprinting the payload and stamping the frozen feature names
    and IDs, the check reads the correlations, p-values and adjusted
    p-values of the features a block of stored chunk rows at a time, and
    keeps the three blocks of one, which it checks together, while it reads
    those of the next.
    """
    arrays = _payload_arrays(group, MARKER_PAYLOAD)
    if len(arrays) < len(MARKER_PAYLOAD):
        return _check_bytes(arrays)
    statistics = [arrays[name] for name in MARKER_PAYLOAD[:3]]
    return _check_bytes(
        arrays, _statistics_bytes(statistics, n_selected, blocks=3, copies=2)
    )


def _read_rows_bytes(steps: _Steps, array: zarr.Array, n_rows: int) -> None:
    """Add the read of ``n_rows`` scattered rows of ``array``, which stay, a band of chunks at a time, as ``metadata.rows.row_read_bytes`` charges it."""
    geometry = array_geometry(array)
    assert geometry is not None
    kept = int(n_rows) * _row_bytes(array)
    steps.step(max(0, row_read_bytes(geometry, int(n_rows)) - kept), keeping=kept)


def marker_load_bytes(group: Any, *, n_selected: int) -> int:
    """Return the most bytes that loading a saved pseudotime-marker table holds, beside its pseudotime.

    The payload is checked as :func:`marker_check_bytes` states, then the
    names, IDs and statistics of the ``n_selected`` selected features are
    read and kept, and the table copies the names and IDs as text, makes a
    Python string of each, and copies the statistics and the features'
    positions.
    """
    arrays = _payload_arrays(group, MARKER_PAYLOAD)
    check = marker_check_bytes(group, n_selected=n_selected)
    if len(arrays) < len(MARKER_PAYLOAD):
        return check
    steps = _Steps()
    steps.step(check)
    for name in ("feature_names", "feature_ids", *MARKER_PAYLOAD[:3]):
        _read_rows_bytes(steps, arrays[name], n_selected)
    text = _row_bytes(arrays["feature_names"]) + _row_bytes(arrays["feature_ids"])
    steps.step(int(n_selected) * (2 * text + 2 * STRING_BYTES + 32))
    return steps.most


def aggregation_check_bytes(group: Any, *, n_selected: int) -> int:
    """Return the most bytes that :func:`aggregation_payload_is_valid` holds.

    Beside fingerprinting the payload and stamping the frozen feature names
    and IDs, the check reads the ``n_selected`` selected features a block
    of stored chunk rows at a time: their positions, which it copies as
    int64, whether each is valid, their clusters as int64 and their
    profiles, of which it takes a copy, keeping a block while it reads the
    next. It then reads every feature's cluster a stored chunk at a time,
    beside the clusters that the chunk's selected features give it, as int64,
    and their validity and positions.
    """
    arrays = _payload_arrays(group, AGGREGATION_PAYLOAD)
    if len(arrays) < len(AGGREGATION_PAYLOAD):
        return _check_bytes(arrays)
    blockwise = [
        arrays[name]
        for name in ("data", "feature_indices", "valid_features", "feature_clusters")
    ]
    rows = min(_chunk_rows(array) for array in blockwise)
    selected = 17 * rows + max(_read_bytes(array, rows) for array in blockwise)
    cluster_values = arrays["cluster_values"]
    feature_rows = _chunk_rows(cluster_values)
    clusters = 8 * feature_rows + _read_bytes(cluster_values, feature_rows)
    return _check_bytes(arrays, selected, clusters)


def aggregation_load_bytes(group: Any, *, n_selected: int) -> int:
    """Return the most bytes that loading a saved pseudotime aggregation holds, beside its pseudotime.

    The payload is checked as :func:`aggregation_check_bytes` states, then
    the positions of the valid features among the ``n_selected`` selected
    ones are found, and their positions, clusters, names and IDs read and
    kept, the clusters as int64 and the names and IDs copied as text. The
    profiles stay on disk until they are read.
    """
    arrays = _payload_arrays(group, AGGREGATION_PAYLOAD)
    n_selected = int(n_selected)
    check = aggregation_check_bytes(group, n_selected=n_selected)
    if len(arrays) < len(AGGREGATION_PAYLOAD):
        return check
    steps = _Steps()
    steps.step(check)
    valid = arrays["valid_features"]
    steps.step(
        _read_bytes(valid, _chunk_rows(valid)) + 8 * n_selected,
        keeping=8 * n_selected,
    )
    for name in ("feature_indices", "feature_clusters", "feature_names", "feature_ids"):
        _read_rows_bytes(steps, arrays[name], n_selected)
        if name != "feature_indices":
            # The rows are copied, and the copy kept in their place.
            steps.step(n_selected * max(8, _row_bytes(arrays[name])))
    return steps.most


def stored_profile_sizes(group: Any) -> tuple[int, int]:
    """Return the values of each feature's profile in a saved aggregation and the bytes of its stored name and ID.

    A pseudotime aggregation's profile holds a value for each bin, and a
    lineage aggregation's for each bin of each lineage. Each count is 0
    where the payload lacks its arrays, which its check refuses.
    """
    arrays = _payload_arrays(group, ("data", "feature_names", "feature_ids"))
    profiles = arrays.get("data")
    values = (
        int(np.prod(profiles.shape[1:], dtype=np.int64))
        if profiles is not None and profiles.ndim
        else 0
    )
    return values, sum(
        _row_bytes(arrays[name])
        for name in ("feature_names", "feature_ids")
        if name in arrays
    )


def lineage_statistics_check_bytes(
    group: Any, names: Sequence[str], *, n_selected: int
) -> int:
    """Return the most bytes that checking saved lineage markers or drivers holds.

    ``names`` are the payload's arrays. Beside fingerprinting them and
    stamping the frozen feature names and IDs, the check reads each array of
    statistics, features by lineages, a block of stored chunk rows at a
    time, keeping the block before it while it reads one.
    """
    arrays = _payload_arrays(group, names)
    if len(arrays) < len(names):
        return _check_bytes(arrays)
    statistics = [
        array
        for name, array in arrays.items()
        if name not in ("feature_names", "feature_ids", "effective_cells")
    ]
    return _check_bytes(
        arrays, _statistics_bytes(statistics, n_selected, blocks=1, copies=1)
    )


def lineage_statistics_load_bytes(
    group: Any, names: Sequence[str], *, n_selected: int, n_lineages: int
) -> int:
    """Return the most bytes that loading saved lineage markers or drivers holds, beside its fate map's cells.

    The payload is checked as :func:`lineage_statistics_check_bytes` states.
    Then the names and IDs of the ``n_selected`` selected features are read
    and copied as text, and each array of statistics read at them. The long
    table takes a frame for each of the ``n_lineages`` lineages, which makes
    a Python string of each name and ID and copies the positions, the
    lineage's label and statistics, and then joins the frames into a copy of
    every column, whose text it shares.
    """
    arrays = _payload_arrays(group, names)
    n_selected = int(n_selected)
    n_lineages = int(n_lineages)
    check = lineage_statistics_check_bytes(group, names, n_selected=n_selected)
    if len(arrays) < len(names):
        return check
    steps = _Steps()
    steps.step(check)
    text = 0
    for name in ("feature_names", "feature_ids"):
        _read_rows_bytes(steps, arrays[name], n_selected)
        # The rows are copied as text, and the copy kept in their place.
        text += _row_bytes(arrays[name])
        steps.step(n_selected * _row_bytes(arrays[name]))
    statistics = [
        name
        for name in names
        if name not in ("feature_names", "feature_ids", "effective_cells")
    ]
    for name in statistics:
        _read_rows_bytes(steps, arrays[name], n_selected)
    columns = 8 * (len(statistics) + 2)
    frames = n_lineages * n_selected * (2 * STRING_BYTES + text + columns)
    joined = n_lineages * n_selected * (columns + 16)
    steps.step(frames + joined)
    return steps.most


def lineage_aggregation_check_bytes(group: Any, *, n_selected: int) -> int:
    """Return the most bytes that :func:`lineage_aggregation_payload_is_valid` holds.

    Beside fingerprinting the payload and stamping the frozen feature names
    and IDs, the check reads the positions of the ``n_selected`` selected
    features whole, which it copies as int64, then keeps their validity and
    their clusters as int64 while it reads their profiles a block of stored
    chunk rows at a time, of which it takes a copy, keeping a block while it
    reads the next. Beside the last block, it then holds every feature's
    expected cluster, as int64, and reads every feature's stored cluster
    whole, as int64, which it compares.
    """
    arrays = _payload_arrays(group, LINEAGE_AGGREGATION_PAYLOAD)
    if len(arrays) < len(LINEAGE_AGGREGATION_PAYLOAD):
        return _check_bytes(arrays)
    profiles = arrays["data"]
    cluster_values = arrays["cluster_values"]
    n_features = int(cluster_values.shape[0])
    return _check_bytes(
        arrays,
        9 * int(n_selected) + _read_bytes(profiles, _chunk_rows(profiles)),
        8 * n_features + _read_bytes(cluster_values, n_features),
    )


def lineage_aggregation_load_bytes(group: Any, *, n_selected: int) -> int:
    """Return the most bytes that loading a saved lineage aggregation holds, beside its fate map's cells.

    The payload is checked as :func:`lineage_aggregation_check_bytes`
    states, then the positions of the valid features among the
    ``n_selected`` selected ones are found, and their positions, profiles
    and clusters read and copied, as int64 and float64, and their names and
    IDs read and copied as text.
    """
    arrays = _payload_arrays(group, LINEAGE_AGGREGATION_PAYLOAD)
    n_selected = int(n_selected)
    check = lineage_aggregation_check_bytes(group, n_selected=n_selected)
    if len(arrays) < len(LINEAGE_AGGREGATION_PAYLOAD):
        return check
    steps = _Steps()
    steps.step(check)
    valid = arrays["valid_features"]
    steps.step(
        _read_bytes(valid, _chunk_rows(valid)) + 8 * n_selected,
        keeping=8 * n_selected,
    )
    for name in (
        "feature_indices",
        "data",
        "feature_clusters",
        "feature_names",
        "feature_ids",
    ):
        _read_rows_bytes(steps, arrays[name], n_selected)
        # The rows are copied, and the copy kept in their place.
        steps.step(n_selected * max(8, _row_bytes(arrays[name])))
    for name in ("bin_pseudotime", "lineage_weight"):
        steps.step(
            _read_bytes(arrays[name], arrays[name].shape[0]),
            keeping=2 * arrays[name].nbytes,
        )
    return steps.most


def lineage_curve_payload_is_valid(
    group: Any,
    *,
    n_cells: int,
    n_lineages: int,
    n_points: int,
    n_dims: int,
    expected_valid: np.ndarray,
) -> bool:
    """Check stored lineage curves and cell projections against their shapes."""
    if set(group.attrs) != LINEAGE_RESULT_ATTRIBUTES:
        return False
    if not payload_fingerprint_matches(group, LINEAGE_CURVE_PAYLOAD):
        return False
    points_array = _array_contract(
        group,
        "points",
        shape=(n_lineages, n_points, n_dims),
        dtype_kinds=frozenset({"f"}),
    )
    ordering_array = _array_contract(
        group,
        "curve_pseudotime",
        shape=(n_lineages, n_points),
        dtype_kinds=frozenset({"f"}),
    )
    progress_array = _array_contract(
        group,
        "progress",
        shape=(n_cells, n_lineages),
        dtype_kinds=frozenset({"f"}),
    )
    distance_array = _array_contract(
        group,
        "distance",
        shape=(n_cells, n_lineages),
        dtype_kinds=frozenset({"f"}),
    )
    valid_array = _array_contract(
        group,
        "valid",
        shape=(n_cells,),
        dtype_kinds=frozenset({"b"}),
    )
    if (
        points_array is None
        or ordering_array is None
        or progress_array is None
        or distance_array is None
        or valid_array is None
    ):
        return False
    points = np.asarray(points_array[:], dtype=np.float64)
    curve_ordering = np.asarray(ordering_array[:], dtype=np.float64)
    if (
        not np.isfinite(points).all()
        or not np.isfinite(curve_ordering).all()
        or np.any(np.diff(curve_ordering, axis=1) < 0.0)
    ):
        return False
    valid = np.asarray(valid_array[:], dtype=bool)
    if not np.array_equal(valid, np.asarray(expected_valid, dtype=bool)):
        return False
    block_rows = min(
        row_band(array_geometry(array), unit="chunk", fallback=1)
        for array in (progress_array, distance_array)
    )
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        block_valid = valid[start:stop]
        progress = np.asarray(progress_array[start:stop], dtype=np.float64)
        distance = np.asarray(distance_array[start:stop], dtype=np.float64)
        scored_progress = progress[block_valid]
        scored_distance = distance[block_valid]
        if (
            not np.isfinite(scored_progress).all()
            or np.any(scored_progress < 0.0)
            or np.any(scored_progress > 1.0)
            or not np.isfinite(scored_distance).all()
            or np.any(scored_distance < 0.0)
            or not np.isnan(progress[~block_valid]).all()
            or not np.isnan(distance[~block_valid]).all()
        ):
            return False
    return True


def lineage_tree_payload_is_valid(
    group: Any,
    *,
    n_clusters: int,
    n_dims: int,
) -> bool:
    """Check a stored cluster tree: sizes, means, and edges that form a tree."""
    if set(group.attrs) != LINEAGE_TREE_ATTRIBUTES:
        return False
    labels = group.attrs.get("cluster_labels")
    if not isinstance(labels, list) or len(labels) != n_clusters:
        return False
    if not payload_fingerprint_matches(group, LINEAGE_TREE_PAYLOAD):
        return False
    sizes_array = _array_contract(
        group, "sizes", shape=(n_clusters,), dtype_kinds=frozenset({"i"})
    )
    means_array = _array_contract(
        group, "means", shape=(n_clusters, n_dims), dtype_kinds=frozenset({"f"})
    )
    edges_array = _array_contract(
        group, "edges", shape=(n_clusters - 1, 2), dtype_kinds=frozenset({"i"})
    )
    lengths_array = _array_contract(
        group, "lengths", shape=(n_clusters - 1,), dtype_kinds=frozenset({"f"})
    )
    if (
        sizes_array is None
        or means_array is None
        or edges_array is None
        or lengths_array is None
    ):
        return False
    sizes = np.asarray(sizes_array[:], dtype=np.int64)
    means = np.asarray(means_array[:], dtype=np.float64)
    edges = np.asarray(edges_array[:], dtype=np.int64)
    lengths = np.asarray(lengths_array[:], dtype=np.float64)
    if (
        np.any(sizes < 1)
        or not np.isfinite(means).all()
        or not np.isfinite(lengths).all()
        or np.any(lengths < 0.0)
        or np.any(edges < 0)
        or np.any(edges >= n_clusters)
        or np.any(edges[:, 0] >= edges[:, 1])
    ):
        return False
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_clusters, n_clusters),
    )
    n_components, _ = connected_components(adjacency, directed=False)
    return int(n_components) == 1


def lineage_membership_payload_is_valid(
    group: Any,
    *,
    n_cells: int,
    lineage_names: Sequence[str],
    n_bins: int,
    expected_valid: np.ndarray,
) -> bool:
    """Check a stored lineage membership against its cells, lineages and rule.

    Each label names what its row of ``members`` holds: the one lineage's
    name, ``"shared"`` for several, or ``"none"``; unscored cells follow
    none. Branch points cover every pair of lineages in order. The trunk
    probabilities hold one row per graph component, numbered without
    repeats in ascending order, of positive values or NaN, with a positive
    value for every lineage.
    """
    if set(group.attrs) != LINEAGE_RESULT_ATTRIBUTES:
        return False
    if not payload_fingerprint_matches(group, LINEAGE_MEMBERSHIP_PAYLOAD):
        return False
    n_lineages = len(lineage_names)
    n_pairs = n_lineages * (n_lineages - 1) // 2
    text = frozenset({"O", "S", "T", "U"})
    contracts = {
        name: _array_contract(group, name, shape=shape, dtype_kinds=kinds)
        for name, shape, kinds in (
            ("values", (n_cells,), text),
            ("members", (n_cells, n_lineages), frozenset({"b"})),
            ("valid", (n_cells,), frozenset({"b"})),
            ("distance_limits", (n_lineages, n_bins), frozenset({"f"})),
            ("branch_pairs", (n_pairs, 2), frozenset({"i"})),
            ("branch_pseudotime", (n_pairs,), frozenset({"f"})),
            ("branch_progress", (n_pairs, 2), frozenset({"f"})),
            ("branch_cells", (n_pairs, 2), frozenset({"i"})),
        )
    }
    arrays: dict[str, Any] = {
        name: array for name, array in contracts.items() if array is not None
    }
    if len(arrays) != len(contracts):
        return False
    # The payload's fingerprint check found trunk_components among the arrays.
    stored_components = as_zarr_array(
        group["trunk_components"], name="trunk_components"
    )
    if stored_components.ndim != 1:
        return False
    n_components = int(stored_components.shape[0])
    components = _array_contract(
        group,
        "trunk_components",
        shape=(n_components,),
        dtype_kinds=frozenset({"i", "u"}),
    )
    trunk_array = _array_contract(
        group,
        "trunk_probabilities",
        shape=(n_components, n_lineages),
        dtype_kinds=frozenset({"f"}),
    )
    if components is None or trunk_array is None or not n_components:
        return False
    numbers = np.asarray(components[:], dtype=np.int64)
    trunk = np.asarray(trunk_array[:], dtype=np.float64)
    finite = np.isfinite(trunk)
    if (
        np.any(numbers < 0)
        or np.any(np.diff(numbers) <= 0)
        or np.any(np.isinf(trunk))
        or np.any(trunk[finite] <= 0)
        or not finite.any(axis=0).all()
    ):
        return False
    valid = np.asarray(arrays["valid"][:], dtype=np.bool_)
    if not np.array_equal(valid, np.asarray(expected_valid, dtype=np.bool_)):
        return False
    members = np.asarray(arrays["members"][:], dtype=np.bool_)
    labels = np.asarray(arrays["values"][:]).astype(str)
    counts = members.sum(axis=1)
    single = counts == 1
    names = np.asarray(list(lineage_names), dtype=str)
    if (
        np.any(members[~valid])
        or not np.all(labels[counts == 0] == "none")
        or not np.all(labels[counts > 1] == "shared")
        or not np.array_equal(labels[single], names[members[single].argmax(axis=1)])
    ):
        return False
    expected_pairs = [
        (first, second)
        for first in range(n_lineages)
        for second in range(first + 1, n_lineages)
    ]
    pairs = np.asarray(arrays["branch_pairs"][:], dtype=np.int64)
    progress = np.asarray(arrays["branch_progress"][:], dtype=np.float64)
    point = np.asarray(arrays["branch_pseudotime"][:], dtype=np.float64)
    counted = np.asarray(arrays["branch_cells"][:], dtype=np.int64)
    limits = np.asarray(arrays["distance_limits"][:], dtype=np.float64)
    finite_progress = progress[np.isfinite(progress)]
    return bool(
        [tuple(pair) for pair in pairs.tolist()] == expected_pairs
        and np.array_equal(np.isfinite(point), np.isfinite(progress).all(axis=1))
        and not np.any(finite_progress < 0.0)
        and not np.any(finite_progress > 1.0)
        and not np.any(counted < 0)
        and not np.any(limits[np.isfinite(limits)] < 0.0)
        and not np.any(np.isinf(limits))
    )
