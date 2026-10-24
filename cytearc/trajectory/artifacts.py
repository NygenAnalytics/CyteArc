from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from ..metadata.selection import resolve_cell_aligned_artifact
from ..storage.arrays import create_zarr_dataset
from ..storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
)
from ..storage.geometry import array_geometry
from ..storage.partition import row_band
from ..storage.selections import validate_cell_selection
from ..storage.types import as_zarr_array
from .parameters import (
    AGGREGATION_ANN_PARAMETER_NAMES,
    AGGREGATION_ANN_STATIC_PARAMETER_NAMES,
)
from ..utils.arguments import integer_argument

PSEUDOTIME_INPUTS = frozenset({"connectivity_map", "source_sink", "cell_selection"})
PSEUDOTIME_PARAMETERS = frozenset(
    {
        "n_singular_vals",
        "sources",
        "sinks",
        "min_max_norm_ptime",
        "random_seed",
        "component_policy",
    }
)
FATE_INPUTS = frozenset(
    {"connectivity_map", "pseudotime", "sink_labels", "cell_selection"}
)
FATE_PARAMETERS = frozenset({"sinks", "beta", "solver_tol", "max_iterations"})
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

_MISSING_LABEL = object()

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


def _boolean_parameter(value: Any, name: str) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


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
        name: _boolean_parameter(value[name], f"normalization.{name}")
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


def validate_pseudotime_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_record_keys(
        parameters, PSEUDOTIME_PARAMETERS, "Pseudotime parameters"
    )
    sources = _label_parameters(parameters["sources"], "sources", required=False)
    sinks = _label_parameters(parameters["sinks"], "sinks", required=False)
    if any(bool(source == sink) for source in sources for sink in sinks):
        raise ValueError("sources and sinks must be disjoint")
    component_policy = parameters["component_policy"]
    if component_policy not in {"largest", "error"}:
        raise ValueError("component_policy must be 'largest' or 'error'")
    return {
        "n_singular_vals": integer_argument(
            parameters["n_singular_vals"],
            "n_singular_vals",
            minimum=2,
        ),
        "sources": sources,
        "sinks": sinks,
        "min_max_norm_ptime": _boolean_parameter(
            parameters["min_max_norm_ptime"],
            "min_max_norm_ptime",
        ),
        "random_seed": integer_argument(
            parameters["random_seed"],
            "random_seed",
            minimum=0,
            maximum=2**32 - 1,
        ),
        "component_policy": component_policy,
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
        "smoothen": _boolean_parameter(parameters["smoothen"], "smoothen"),
        "z_scale": _boolean_parameter(parameters["z_scale"], "z_scale"),
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
    if not isinstance(ref, ArtifactRef):
        raise TypeError("cell data input must be an ArtifactRef")
    resolved = resolve_cell_aligned_artifact(
        root,
        ref,
        value_name=value_name,
        ndim=None,
    )
    return resolved.values, resolved.source_cell_selection, resolved.missing_mask


def labels_with_missing_mask(
    values: np.ndarray,
    missing: np.ndarray | None,
    label: str,
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if missing is None:
        return labels
    masked = labels.astype(object)
    masked[missing] = _MISSING_LABEL
    return masked


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
    if diffusion_load_bytes(nnz, n_cells, imputed_features) >= memory_bytes:
        raise MemoryError(
            "Diffusion operator and imputed output exceed the memory budget "
            "during loading or sparse conversion; increase the memory budget "
            "or request fewer features per call."
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
    pseudotime_valid: np.ndarray,
    sink_values: np.ndarray,
    sink_labels: Sequence[Any],
) -> bool:
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
    expected_valid = np.asarray(pseudotime_valid, dtype=bool)
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
    n_neighbours: int,
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
        or not 1 <= n_neighbours < valid_count
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
        "include_sinks": _boolean_parameter(
            parameters["include_sinks"], "include_sinks"
        ),
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
        "z_scale": _boolean_parameter(parameters["z_scale"], "z_scale"),
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
    n_neighbours: int,
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
        or not 1 <= n_neighbours < valid_count
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
