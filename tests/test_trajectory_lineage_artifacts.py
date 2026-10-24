"""Persisted lineage records and payloads are validated before reuse or loading.

Each parameter validator normalizes a valid persisted record and refuses each
malformed field with a matching error. Each payload validator accepts a small
payload stored the way its producer writes it, then refuses it after one
persisted fact changes. Tampers refresh the payload fingerprint unless the
fingerprint is what they change, so the check under test is the one that
refuses the payload.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.storage.arrays import create_metadata_column
from cytearc.storage.artifacts import (
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
)
from cytearc.storage.copy import copy_metadata_array
from cytearc.trajectory.artifacts import (
    LINEAGE_AGGREGATION_ATTRIBUTES,
    LINEAGE_AGGREGATION_INPUTS,
    LINEAGE_AGGREGATION_PARAMETERS,
    LINEAGE_AGGREGATION_PAYLOAD,
    LINEAGE_CURVE_INPUTS,
    LINEAGE_CURVE_PARAMETERS,
    LINEAGE_CURVE_PAYLOAD,
    LINEAGE_DRIVER_ATTRIBUTES,
    LINEAGE_DRIVER_INPUTS,
    LINEAGE_DRIVER_METHODS,
    LINEAGE_DRIVER_PARAMETERS,
    LINEAGE_DRIVER_PAYLOAD,
    LINEAGE_FEATURE_INPUTS,
    LINEAGE_MARKER_INPUTS,
    LINEAGE_MARKER_METHODS,
    LINEAGE_MARKER_PARAMETERS,
    LINEAGE_MARKER_PAYLOAD,
    LINEAGE_RESULT_ATTRIBUTES,
    LINEAGE_TREE_ATTRIBUTES,
    LINEAGE_TREE_INPUTS,
    LINEAGE_TREE_PARAMETERS,
    LINEAGE_TREE_PAYLOAD,
    lineage_aggregation_payload_is_valid,
    lineage_curve_payload_is_valid,
    lineage_driver_payload_is_valid,
    lineage_marker_payload_is_valid,
    lineage_tree_payload_is_valid,
    validate_lineage_aggregation_parameters,
    validate_lineage_curve_parameters,
    validate_lineage_driver_parameters,
    validate_lineage_marker_parameters,
    validate_lineage_tree_parameters,
)
from cytearc.trajectory.curves import CURVE_FIT_BINS, CURVE_SPLINE_BASIS
from cytearc.trajectory.parameters import resolve_aggregation_ann_params

# Persisted parameter records

_NORMALIZATION = {"log_transform": False, "renormalize_subset": False}
# The keys are out of order; validation returns them sorted.
_NORMALIZATION_METHOD = {
    "qualname": "norm_lib_size",
    "module": "cytearc.assay.normalization",
}
_MARKER_RECORD: dict[str, Any] = {
    "normalization": _NORMALIZATION,
    "normalization_method": _NORMALIZATION_METHOD,
    "size_factor": 1000.0,
    "lineages": ["A", "B"],
    "association_method": "fate_weighted_pearson",
    "p_value_method": "student_t_kish",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features_per_lineage",
    "min_cells": 10.0,
}
_DRIVER_RECORD: dict[str, Any] = {
    "normalization": _NORMALIZATION,
    "normalization_method": _NORMALIZATION_METHOD,
    "size_factor": 1000.0,
    "lineages": ["A", "B"],
    "include_sinks": False,
    "min_cells": 10,
    "confidence_level": 0.95,
    "association_method": "pearson",
    "p_value_method": "fisher_z",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features_per_lineage",
}
# A stored record holds the ANN parameters resolved for two lineages of 50 bins.
_AGGREGATION_RECORD: dict[str, Any] = {
    "normalization": _NORMALIZATION,
    "normalization_method": _NORMALIZATION_METHOD,
    "size_factor": 1000.0,
    "lineages": ["A", "B"],
    "n_bins": 50,
    "smoothing": 1.0,
    "min_exp": 1e-3,
    "z_scale": True,
    "n_neighbours": 11,
    "n_clusters": 10,
    "ann_params": resolve_aggregation_ann_params(None, dim=100),
    "nan_cluster_value": -1,
}
_CURVE_RECORD: dict[str, Any] = {
    "lineages": ["A", "B"],
    "n_points": 100,
    "df": 10.0,
    "fit_bins": CURVE_FIT_BINS,
    "spline_basis": CURVE_SPLINE_BASIS,
}
_TREE_RECORD: dict[str, Any] = {"root": "stem", "ends": ["A", "B"]}

_Validator = Callable[[Mapping[str, Any]], dict[str, Any]]
_RECORDS: dict[str, tuple[_Validator, dict[str, Any], str]] = {
    "markers": (
        validate_lineage_marker_parameters,
        _MARKER_RECORD,
        "Lineage-marker parameters",
    ),
    "drivers": (
        validate_lineage_driver_parameters,
        _DRIVER_RECORD,
        "Lineage-driver parameters",
    ),
    "aggregation": (
        validate_lineage_aggregation_parameters,
        _AGGREGATION_RECORD,
        "Lineage-aggregation parameters",
    ),
    "curves": (
        validate_lineage_curve_parameters,
        _CURVE_RECORD,
        "Lineage-curve parameters",
    ),
    "tree": (
        validate_lineage_tree_parameters,
        _TREE_RECORD,
        "Lineage-tree parameters",
    ),
}
_FEATURE_RECORD_KINDS = ("aggregation", "drivers", "markers")
_LINEAGE_RECORD_KINDS = ("aggregation", "curves", "drivers", "markers")

# The normalized fields that markers, drivers and aggregations share.
_FEATURE_FIELDS = {
    "normalization": {"log_transform": False, "renormalize_subset": False},
    "normalization_method": {
        "module": "cytearc.assay.normalization",
        "qualname": "norm_lib_size",
    },
    "size_factor": 1000.0,
    "lineages": ("A", "B"),
}

_BASE_ATTRIBUTES = frozenset(
    {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "cytearc_version",
        "complete",
        "payload_fingerprint",
    }
)


def test_lineage_contracts_name_their_persisted_fields() -> None:
    feature_inputs = {
        "cell_selection",
        "feature_selection",
        "fate",
        "dataset_fingerprint",
        "ordered_feature_ids_fingerprint",
        "ordered_feature_names_fingerprint",
    }
    assert LINEAGE_FEATURE_INPUTS == feature_inputs
    assert LINEAGE_MARKER_INPUTS == feature_inputs
    assert LINEAGE_AGGREGATION_INPUTS == feature_inputs
    assert LINEAGE_DRIVER_INPUTS == feature_inputs | {"driver_cells"}
    assert LINEAGE_CURVE_INPUTS == {"fate", "coordinates", "cell_selection"}
    assert LINEAGE_TREE_INPUTS == {"labels", "coordinates", "cell_selection"}

    assert LINEAGE_MARKER_PARAMETERS == set(_MARKER_RECORD)
    assert LINEAGE_DRIVER_PARAMETERS == set(_DRIVER_RECORD)
    assert LINEAGE_AGGREGATION_PARAMETERS == set(_AGGREGATION_RECORD)
    assert LINEAGE_CURVE_PARAMETERS == set(_CURVE_RECORD)
    assert LINEAGE_TREE_PARAMETERS == set(_TREE_RECORD)
    assert LINEAGE_MARKER_METHODS == {
        "association_method": "fate_weighted_pearson",
        "p_value_method": "student_t_kish",
        "adjustment_method": "fdr_bh",
        "adjustment_scope": "tested_features_per_lineage",
    }
    assert LINEAGE_DRIVER_METHODS == {
        "association_method": "pearson",
        "p_value_method": "fisher_z",
        "adjustment_method": "fdr_bh",
        "adjustment_scope": "tested_features_per_lineage",
    }

    # Payload fingerprints cover the arrays in this order.
    assert LINEAGE_MARKER_PAYLOAD == (
        "r_value",
        "p_value",
        "p_value_adjusted",
        "effective_cells",
        "feature_names",
        "feature_ids",
    )
    assert LINEAGE_DRIVER_PAYLOAD == (
        "r_value",
        "p_value",
        "p_value_adjusted",
        "ci_low",
        "ci_high",
        "feature_names",
        "feature_ids",
    )
    assert LINEAGE_AGGREGATION_PAYLOAD == (
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
    assert LINEAGE_CURVE_PAYLOAD == (
        "points",
        "curve_pseudotime",
        "progress",
        "distance",
        "valid",
    )
    assert LINEAGE_TREE_PAYLOAD == ("sizes", "means", "edges", "lengths")

    assert LINEAGE_RESULT_ATTRIBUTES == _BASE_ATTRIBUTES
    assert LINEAGE_DRIVER_ATTRIBUTES == _BASE_ATTRIBUTES | {"n_cells"}
    assert LINEAGE_AGGREGATION_ATTRIBUTES == _BASE_ATTRIBUTES | {"input_fingerprints"}
    assert LINEAGE_TREE_ATTRIBUTES == _BASE_ATTRIBUTES | {"cluster_labels"}


@pytest.mark.parametrize("change", ["missing", "extra"])
@pytest.mark.parametrize("kind", sorted(_RECORDS))
def test_lineage_records_require_their_exact_parameter_names(
    kind: str,
    change: str,
) -> None:
    validate, record, label = _RECORDS[kind]
    changed = dict(record)
    if change == "missing":
        del changed[sorted(record)[0]]
    else:
        changed["schema_version"] = 1

    with pytest.raises(
        ValueError,
        match=re.escape(f"{label} does not match its persisted contract"),
    ):
        validate(changed)


def test_marker_record_round_trips_as_normalized_values() -> None:
    validated = validate_lineage_marker_parameters(_MARKER_RECORD)

    assert validated == {**_FEATURE_FIELDS, "min_cells": 10.0, **LINEAGE_MARKER_METHODS}
    assert list(validated["normalization_method"]) == ["module", "qualname"]


def test_marker_record_normalizes_numpy_scalars_and_integer_reals() -> None:
    validated = validate_lineage_marker_parameters(
        {
            **_MARKER_RECORD,
            "normalization": {
                "log_transform": np.bool_(True),
                "renormalize_subset": False,
            },
            "size_factor": np.float32(1e4),
            "lineages": (np.str_("A"), np.int64(2)),
            "min_cells": 3,
            "adjustment_method": np.str_("fdr_bh"),
        }
    )

    assert validated["normalization"]["log_transform"] is True
    assert validated["size_factor"] == 1e4
    assert type(validated["size_factor"]) is float
    assert validated["lineages"] == ("A", 2)
    assert [type(label) for label in validated["lineages"]] == [str, int]
    assert validated["min_cells"] == 3.0
    assert type(validated["min_cells"]) is float
    assert type(validated["adjustment_method"]) is str


@pytest.mark.parametrize(
    "method",
    [
        # A callable with an ``artifact_identity`` attribute is identified by it.
        pytest.param({"identity": "lab.normalize:v2"}, id="declared-identity"),
        # A normalizer defined outside CyteArc is identified by its code as well.
        pytest.param(
            {"code": "0f" * 16, "module": "__main__", "qualname": "normalize"},
            id="custom-code",
        ),
    ],
)
@pytest.mark.parametrize("kind", _FEATURE_RECORD_KINDS)
def test_feature_records_keep_every_normalizer_identity(
    kind: str,
    method: dict[str, str],
) -> None:
    validate, record, _label = _RECORDS[kind]

    validated = validate({**record, "normalization_method": method})

    assert validated["normalization_method"] == method


@pytest.mark.parametrize("kind", _FEATURE_RECORD_KINDS)
def test_feature_records_of_an_assay_without_size_factor_are_valid(kind: str) -> None:
    # Only RNA assays set a size factor; other assays record None.
    validate, record, _label = _RECORDS[kind]

    assert validate({**record, "size_factor": None})["size_factor"] is None


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"normalization": None},
            ValueError,
            "normalization parameters are malformed",
            id="no-normalization",
        ),
        pytest.param(
            {"normalization": {"log_transform": False}},
            ValueError,
            "normalization parameters are malformed",
            id="partial-normalization",
        ),
        pytest.param(
            {"normalization": {**_NORMALIZATION, "scale": True}},
            ValueError,
            "normalization parameters are malformed",
            id="extra-normalization-flag",
        ),
        pytest.param(
            {"normalization": {**_NORMALIZATION, "renormalize_subset": 0}},
            TypeError,
            "normalization.renormalize_subset must be a boolean",
            id="integer-normalization-flag",
        ),
        pytest.param(
            {"normalization_method": "norm_lib_size"},
            TypeError,
            "normalization_method must be a mapping",
            id="method-name",
        ),
        pytest.param(
            {"normalization_method": {"module": "cytearc.assay.normalization"}},
            ValueError,
            "normalization_method is malformed",
            id="method-without-qualname",
        ),
        pytest.param(
            {"normalization_method": {"identity": ""}},
            ValueError,
            "normalization_method is malformed",
            id="blank-identity",
        ),
        pytest.param(
            {"normalization_method": {"module": "m", "qualname": 7}},
            ValueError,
            "normalization_method is malformed",
            id="numeric-qualname",
        ),
        pytest.param(
            {"size_factor": "1e4"},
            TypeError,
            "size_factor must be a real number",
            id="text-size-factor",
        ),
        pytest.param(
            {"size_factor": True},
            TypeError,
            "size_factor must be a real number",
            id="boolean-size-factor",
        ),
        pytest.param(
            {"size_factor": 0.0},
            ValueError,
            "size_factor must be greater than 0.0",
            id="zero-size-factor",
        ),
        pytest.param(
            {"size_factor": -1},
            ValueError,
            "size_factor must be greater than 0.0",
            id="negative-size-factor",
        ),
        pytest.param(
            {"size_factor": np.inf},
            ValueError,
            "size_factor must be finite",
            id="infinite-size-factor",
        ),
        pytest.param(
            {"size_factor": np.nan},
            ValueError,
            "size_factor must be finite",
            id="nan-size-factor",
        ),
    ],
)
@pytest.mark.parametrize("kind", _FEATURE_RECORD_KINDS)
def test_feature_records_reject_malformed_normalization(
    kind: str,
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    validate, record, _label = _RECORDS[kind]

    with pytest.raises(error, match=re.escape(message)):
        validate({**record, **changes})


@pytest.mark.parametrize(
    ("lineages", "error", "message"),
    [
        pytest.param("A", TypeError, "lineages must be a list", id="text"),
        pytest.param({"A": 0}, TypeError, "lineages must be a list", id="mapping"),
        pytest.param(
            [], ValueError, "lineages must contain at least one label", id="empty"
        ),
        pytest.param(
            ["A", "A"],
            ValueError,
            "lineages must not contain duplicate labels",
            id="repeated-label",
        ),
        pytest.param(
            [1, 1.0],
            ValueError,
            "lineages must not contain duplicate labels",
            id="equal-numbers",
        ),
        pytest.param(
            [np.int64(2), 2],
            ValueError,
            "lineages must not contain duplicate labels",
            id="numpy-duplicate",
        ),
        pytest.param(
            [None], TypeError, "lineages must contain scalar labels", id="none"
        ),
        pytest.param(
            [["A"]],
            TypeError,
            "lineages must contain scalar labels",
            id="nested-list",
        ),
        pytest.param(
            [np.nan],
            ValueError,
            "lineages must contain finite scalar labels",
            id="nan",
        ),
        pytest.param(
            ["A", np.float32(np.inf)],
            ValueError,
            "lineages must contain finite scalar labels",
            id="numpy-infinity",
        ),
    ],
)
@pytest.mark.parametrize("kind", _LINEAGE_RECORD_KINDS)
def test_lineage_records_reject_malformed_lineages(
    kind: str,
    lineages: Any,
    error: type[Exception],
    message: str,
) -> None:
    validate, record, _label = _RECORDS[kind]

    with pytest.raises(error, match=re.escape(message)):
        validate({**record, "lineages": lineages})


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"min_cells": 0.0},
            ValueError,
            "min_cells must be greater than 0.0",
            id="zero-min-cells",
        ),
        pytest.param(
            {"min_cells": -2},
            ValueError,
            "min_cells must be greater than 0.0",
            id="negative-min-cells",
        ),
        pytest.param(
            {"min_cells": np.inf},
            ValueError,
            "min_cells must be finite",
            id="infinite-min-cells",
        ),
        pytest.param(
            {"min_cells": "10"},
            TypeError,
            "min_cells must be a real number",
            id="text-min-cells",
        ),
        pytest.param(
            {"min_cells": True},
            TypeError,
            "min_cells must be a real number",
            id="boolean-min-cells",
        ),
        pytest.param(
            {"association_method": "pearson"},
            ValueError,
            "association_method must be 'fate_weighted_pearson'",
            id="unweighted-association",
        ),
        pytest.param(
            {"p_value_method": "student_t"},
            ValueError,
            "p_value_method must be 'student_t_kish'",
            id="unweighted-p-values",
        ),
        pytest.param(
            {"adjustment_method": "bonferroni"},
            ValueError,
            "adjustment_method must be 'fdr_bh'",
            id="other-adjustment",
        ),
        pytest.param(
            {"adjustment_scope": "tested_features"},
            ValueError,
            "adjustment_scope must be 'tested_features_per_lineage'",
            id="pooled-adjustment",
        ),
    ],
)
def test_marker_record_rejects_each_malformed_parameter(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        validate_lineage_marker_parameters({**_MARKER_RECORD, **changes})


def test_driver_record_round_trips_as_normalized_values() -> None:
    assert validate_lineage_driver_parameters(_DRIVER_RECORD) == {
        **_FEATURE_FIELDS,
        "include_sinks": False,
        "min_cells": 10,
        "confidence_level": 0.95,
        **LINEAGE_DRIVER_METHODS,
    }

    validated = validate_lineage_driver_parameters(
        {
            **_DRIVER_RECORD,
            "include_sinks": np.bool_(True),
            "min_cells": np.int64(4),
            "confidence_level": np.float32(0.5),
        }
    )
    assert validated["include_sinks"] is True
    assert validated["min_cells"] == 4
    assert type(validated["min_cells"]) is int
    assert validated["confidence_level"] == 0.5
    assert type(validated["confidence_level"]) is float


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"include_sinks": 0},
            TypeError,
            "include_sinks must be a boolean",
            id="integer-include-sinks",
        ),
        pytest.param(
            {"include_sinks": "no"},
            TypeError,
            "include_sinks must be a boolean",
            id="text-include-sinks",
        ),
        pytest.param(
            {"min_cells": 10.0},
            TypeError,
            "min_cells must be an integer",
            id="real-min-cells",
        ),
        pytest.param(
            {"min_cells": True},
            TypeError,
            "min_cells must be an integer",
            id="boolean-min-cells",
        ),
        pytest.param(
            {"min_cells": 0},
            ValueError,
            "min_cells must be at least 1",
            id="zero-min-cells",
        ),
        pytest.param(
            {"confidence_level": 0.0},
            ValueError,
            "confidence_level must be greater than 0.0",
            id="no-confidence",
        ),
        pytest.param(
            {"confidence_level": 1.0},
            ValueError,
            "confidence_level must be less than 1.0",
            id="certain-confidence",
        ),
        pytest.param(
            {"confidence_level": 95},
            ValueError,
            "confidence_level must be less than 1.0",
            id="percent-confidence",
        ),
        pytest.param(
            {"confidence_level": np.nan},
            ValueError,
            "confidence_level must be finite",
            id="nan-confidence",
        ),
        pytest.param(
            {"confidence_level": "0.95"},
            TypeError,
            "confidence_level must be a real number",
            id="text-confidence",
        ),
        pytest.param(
            {"association_method": "fate_weighted_pearson"},
            ValueError,
            "association_method must be 'pearson'",
            id="weighted-association",
        ),
        pytest.param(
            {"p_value_method": "student_t_kish"},
            ValueError,
            "p_value_method must be 'fisher_z'",
            id="kish-p-values",
        ),
        pytest.param(
            {"adjustment_method": "fdr_by"},
            ValueError,
            "adjustment_method must be 'fdr_bh'",
            id="other-adjustment",
        ),
        pytest.param(
            {"adjustment_scope": "tested_features"},
            ValueError,
            "adjustment_scope must be 'tested_features_per_lineage'",
            id="pooled-adjustment",
        ),
    ],
)
def test_driver_record_rejects_each_malformed_parameter(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        validate_lineage_driver_parameters({**_DRIVER_RECORD, **changes})


def test_aggregation_record_round_trips_as_normalized_values() -> None:
    assert validate_lineage_aggregation_parameters(_AGGREGATION_RECORD) == {
        **_FEATURE_FIELDS,
        "n_bins": 50,
        "smoothing": 1.0,
        "min_exp": 1e-3,
        "z_scale": True,
        "n_neighbours": 11,
        "n_clusters": 10,
        "ann_params": resolve_aggregation_ann_params(None, dim=100),
        "nan_cluster_value": -1,
    }


def test_aggregation_record_accepts_the_unresolved_settings_of_a_request() -> None:
    # The producer validates the requested ANN parameters before resolving them.
    validated = validate_lineage_aggregation_parameters(
        {
            **_AGGREGATION_RECORD,
            "ann_params": {},
            "n_bins": 2,
            "smoothing": 0,
            "min_exp": 0.0,
        }
    )

    assert validated["ann_params"] == {}
    assert validated["n_bins"] == 2
    assert validated["smoothing"] == 0.0
    assert type(validated["smoothing"]) is float
    assert validated["min_exp"] == 0.0


@pytest.mark.parametrize("nan_cluster_value", [-1, 0, 11])
def test_aggregation_record_accepts_unassigned_labels_outside_the_clusters(
    nan_cluster_value: int,
) -> None:
    validated = validate_lineage_aggregation_parameters(
        {**_AGGREGATION_RECORD, "nan_cluster_value": nan_cluster_value}
    )

    assert validated["nan_cluster_value"] == nan_cluster_value


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"n_clusters": 0},
            ValueError,
            "n_clusters must be at least 1",
            id="no-clusters",
        ),
        pytest.param(
            {"n_clusters": 10.0},
            TypeError,
            "n_clusters must be an integer",
            id="real-clusters",
        ),
        pytest.param(
            {"nan_cluster_value": "-1"},
            TypeError,
            "nan_cluster_value must be an integer",
            id="text-unassigned-label",
        ),
        pytest.param(
            {"nan_cluster_value": 1},
            ValueError,
            "nan_cluster_value conflicts with assigned cluster labels",
            id="first-cluster-label",
        ),
        pytest.param(
            {"nan_cluster_value": 10},
            ValueError,
            "nan_cluster_value conflicts with assigned cluster labels",
            id="last-cluster-label",
        ),
        pytest.param(
            {"n_bins": 1},
            ValueError,
            "n_bins must be at least 2",
            id="one-bin",
        ),
        pytest.param(
            {"n_bins": 50.0},
            TypeError,
            "n_bins must be an integer",
            id="real-bins",
        ),
        pytest.param(
            {"smoothing": -0.5},
            ValueError,
            "smoothing must be at least 0.0",
            id="negative-smoothing",
        ),
        pytest.param(
            {"smoothing": np.inf},
            ValueError,
            "smoothing must be finite",
            id="infinite-smoothing",
        ),
        pytest.param(
            {"smoothing": True},
            TypeError,
            "smoothing must be a real number",
            id="boolean-smoothing",
        ),
        pytest.param(
            {"min_exp": -1e-3},
            ValueError,
            "min_exp must be at least 0.0",
            id="negative-min-exp",
        ),
        pytest.param(
            {"min_exp": np.nan},
            ValueError,
            "min_exp must be finite",
            id="nan-min-exp",
        ),
        pytest.param(
            {"z_scale": 1},
            TypeError,
            "z_scale must be a boolean",
            id="integer-z-scale",
        ),
        pytest.param(
            {"n_neighbours": 0},
            ValueError,
            "n_neighbours must be at least 1",
            id="no-neighbours",
        ),
        pytest.param(
            {"n_neighbours": np.float64(11)},
            TypeError,
            "n_neighbours must be an integer",
            id="real-neighbours",
        ),
        pytest.param(
            {"ann_params": [("M", 16)]},
            TypeError,
            "ann_params must be a mapping",
            id="ann-pairs",
        ),
        pytest.param(
            {"ann_params": {"n_trees": 10}},
            ValueError,
            "ann_params contains unsupported parameters",
            id="unsupported-ann-parameter",
        ),
        pytest.param(
            {"ann_params": {"space": "hamming"}},
            ValueError,
            "ann_params.space is unsupported",
            id="unsupported-ann-space",
        ),
        pytest.param(
            {"ann_params": {"M": 0}},
            ValueError,
            "ann_params.M must be at least 1",
            id="no-ann-links",
        ),
        pytest.param(
            {"ann_params": {"ef": 1.5}},
            TypeError,
            "ann_params.ef must be an integer",
            id="real-ann-ef",
        ),
        pytest.param(
            {"ann_params": {"random_seed": 2**32}},
            ValueError,
            "ann_params.random_seed must be at most 4294967295",
            id="wide-ann-seed",
        ),
    ],
)
def test_aggregation_record_rejects_each_malformed_parameter(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        validate_lineage_aggregation_parameters({**_AGGREGATION_RECORD, **changes})


def test_curve_record_round_trips_as_normalized_values() -> None:
    assert validate_lineage_curve_parameters(_CURVE_RECORD) == {
        "lineages": ("A", "B"),
        "n_points": 100,
        "df": 10.0,
        "fit_bins": 512,
        "spline_basis": 32,
    }


@pytest.mark.parametrize(
    ("df", "spline_basis"),
    [
        pytest.param(2.000001, 8, id="nearly-straight"),
        pytest.param(7.999, 8, id="nearly-the-small-basis"),
        pytest.param(10, 32, id="integer-df"),
        pytest.param(31.5, 32, id="nearly-the-default-basis"),
    ],
)
def test_curve_record_accepts_df_strictly_inside_its_basis(
    df: float,
    spline_basis: int,
) -> None:
    validated = validate_lineage_curve_parameters(
        {
            **_CURVE_RECORD,
            "n_points": 2,
            "df": df,
            "fit_bins": 16,
            "spline_basis": spline_basis,
        }
    )

    assert validated["df"] == float(df)
    assert type(validated["df"]) is float
    assert validated["spline_basis"] == spline_basis


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"spline_basis": 7},
            ValueError,
            "spline_basis must be at least 8",
            id="small-basis",
        ),
        pytest.param(
            {"spline_basis": 32.0},
            TypeError,
            "spline_basis must be an integer",
            id="real-basis",
        ),
        pytest.param(
            {"n_points": 1},
            ValueError,
            "n_points must be at least 2",
            id="one-point",
        ),
        pytest.param(
            {"n_points": "100"},
            TypeError,
            "n_points must be an integer",
            id="text-points",
        ),
        pytest.param(
            {"df": 2.0},
            ValueError,
            "df must be greater than 2.0",
            id="straight-line-df",
        ),
        pytest.param(
            {"df": 1},
            ValueError,
            "df must be greater than 2.0",
            id="small-df",
        ),
        pytest.param(
            {"df": 32.0},
            ValueError,
            "df must be less than 32.0",
            id="df-at-the-basis",
        ),
        pytest.param(
            {"df": 8.0, "spline_basis": 8},
            ValueError,
            "df must be less than 8.0",
            id="df-at-a-smaller-basis",
        ),
        pytest.param(
            {"df": np.inf},
            ValueError,
            "df must be finite",
            id="infinite-df",
        ),
        pytest.param(
            {"df": "10"},
            TypeError,
            "df must be a real number",
            id="text-df",
        ),
        pytest.param(
            {"fit_bins": 15},
            ValueError,
            "fit_bins must be at least 16",
            id="few-fit-bins",
        ),
        pytest.param(
            {"fit_bins": True},
            TypeError,
            "fit_bins must be an integer",
            id="boolean-fit-bins",
        ),
    ],
)
def test_curve_record_rejects_each_malformed_parameter(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        validate_lineage_curve_parameters({**_CURVE_RECORD, **changes})


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(_TREE_RECORD, {"root": "stem", "ends": ("A", "B")}, id="rooted"),
        pytest.param({"root": None, "ends": []}, {"root": None, "ends": ()}, id="free"),
        pytest.param(
            {"root": None, "ends": ["A"]},
            {"root": None, "ends": ("A",)},
            id="unrooted-with-ends",
        ),
        # Zero is a cluster label, not a missing root.
        pytest.param(
            {"root": np.int64(0), "ends": (np.int64(3), np.str_("B"))},
            {"root": 0, "ends": (3, "B")},
            id="numpy-labels",
        ),
    ],
)
def test_tree_record_round_trips_as_native_labels(
    record: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    validated = validate_lineage_tree_parameters(record)

    assert validated == expected
    assert [type(label) for label in validated["ends"]] == [
        type(label) for label in expected["ends"]
    ]
    assert type(validated["root"]) is type(expected["root"])


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"root": ["stem"]},
            TypeError,
            "root must contain scalar labels",
            id="listed-root",
        ),
        pytest.param(
            {"root": {"label": "stem"}},
            TypeError,
            "root must contain scalar labels",
            id="mapping-root",
        ),
        pytest.param(
            {"root": np.nan},
            ValueError,
            "root must contain finite scalar labels",
            id="nan-root",
        ),
        pytest.param({"ends": "A"}, TypeError, "ends must be a list", id="text-ends"),
        pytest.param({"ends": None}, TypeError, "ends must be a list", id="no-ends"),
        pytest.param(
            {"ends": ["A", "A"]},
            ValueError,
            "ends must not contain duplicate labels",
            id="repeated-end",
        ),
        pytest.param(
            {"ends": [3, 3.0]},
            ValueError,
            "ends must not contain duplicate labels",
            id="equal-numeric-ends",
        ),
        pytest.param(
            {"ends": [None]},
            TypeError,
            "ends must contain scalar labels",
            id="none-end",
        ),
        pytest.param(
            {"ends": [-np.inf]},
            ValueError,
            "ends must contain finite scalar labels",
            id="infinite-end",
        ),
        pytest.param(
            {"root": "A"},
            ValueError,
            "The root cannot also be an end",
            id="root-is-an-end",
        ),
        pytest.param(
            {"root": 1, "ends": [2.0, 1.0]},
            ValueError,
            "The root cannot also be an end",
            id="root-equals-a-numeric-end",
        ),
        pytest.param(
            {"root": np.str_("B")},
            ValueError,
            "The root cannot also be an end",
            id="numpy-root-is-an-end",
        ),
    ],
)
def test_tree_record_rejects_each_malformed_parameter(
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        validate_lineage_tree_parameters({**_TREE_RECORD, **changes})


# JSON integers are unbounded, but a float stops near 1.8e308.
_HUGE_INTEGER = 10**400


@pytest.mark.parametrize(
    ("kind", "changes"),
    [
        pytest.param("markers", {"min_cells": -_HUGE_INTEGER}, id="marker-min-cells"),
        pytest.param("markers", {"size_factor": -_HUGE_INTEGER}, id="size-factor"),
        pytest.param(
            "drivers", {"confidence_level": _HUGE_INTEGER}, id="driver-confidence"
        ),
        pytest.param(
            "aggregation", {"smoothing": -_HUGE_INTEGER}, id="aggregation-smoothing"
        ),
        pytest.param("curves", {"df": _HUGE_INTEGER}, id="curve-df"),
    ],
)
def test_out_of_range_huge_integers_are_refused_as_malformed(
    kind: str,
    changes: dict[str, Any],
) -> None:
    validate, record, _label = _RECORDS[kind]

    with pytest.raises(ValueError, match="must be"):
        validate({**record, **changes})


def test_a_huge_spline_basis_is_validated_without_overflowing() -> None:
    try:
        validated = validate_lineage_curve_parameters(
            {**_CURVE_RECORD, "spline_basis": _HUGE_INTEGER}
        )
    except (TypeError, ValueError):
        return
    assert validated["spline_basis"] == _HUGE_INTEGER


# Stored payloads

_RECORD_ATTRIBUTES = {
    "artifact_id": "a" * 64,
    "kind": "test_payload",
    "provenance": {},
    "execution_options": {},
    "created_at_ns": 1,
    "cytearc_version": "test",
    "complete": True,
}


@dataclass(frozen=True)
class _Candidate:
    """A stored payload and the validator call that should accept it."""

    group: zarr.Group
    names: tuple[str, ...]
    validator: Callable[..., bool]
    arguments: dict[str, Any]

    def is_valid(self, **changes: Any) -> bool:
        return self.validator(self.group, **{**self.arguments, **changes})

    def refresh(self) -> None:
        """Fingerprint the stored arrays again, as a consistent writer would."""
        self.group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
            self.group, self.names
        )


_Tamper = Callable[[_Candidate], None]


@dataclass(frozen=True)
class _Stored:
    """The store contents of a complete payload and how to validate it."""

    contents: dict[str, Any]
    names: tuple[str, ...]
    validator: Callable[..., bool]
    arguments: dict[str, Any]


def _store(
    group: zarr.Group,
    name: str,
    values: Any,
    *,
    chunk_rows: int | None = None,
) -> None:
    data = np.asarray(values)
    if data.ndim == 0:
        group.create_array(name, data=data, overwrite=True)
        return
    rows = data.shape[0] if chunk_rows is None else chunk_rows
    chunks = tuple(max(1, size) for size in (rows, *data.shape[1:]))
    group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _stored_payload(
    payload: dict[str, Any],
    names: tuple[str, ...],
    validator: Callable[..., bool],
    arguments: dict[str, Any],
    *,
    chunk_rows: int | None,
    n_features: int | None = None,
    **attributes: Any,
) -> _Stored:
    """Store a complete payload, with copied feature identities when it has them."""
    contents: dict[str, Any] = {}
    root = zarr.open_group(store=MemoryStore(contents), mode="w")
    group = root.create_group("candidate")
    for name, values in payload.items():
        _store(group, name, values, chunk_rows=chunk_rows)
    if n_features is not None:
        # As a feature analysis does, copy the assay's feature identities and
        # expect the fingerprints taken from the assay.
        table = root.create_group("featureData")
        feature_names = create_metadata_column(
            table,
            "names",
            data=np.array([f"g{index}" for index in range(n_features)]),
            dtype=str,
        )
        feature_ids = create_metadata_column(
            table,
            "ids",
            data=np.array([f"id{index}" for index in range(n_features)]),
            dtype=str,
        )
        copy_metadata_array(feature_names, group, "feature_names")
        copy_metadata_array(feature_ids, group, "feature_ids")
        arguments = {
            **arguments,
            "expected_feature_ids_fingerprint": fingerprint_stored_strings(feature_ids),
            "expected_feature_names_fingerprint": fingerprint_stored_strings(
                feature_names
            ),
        }
    group.attrs.update({**_RECORD_ATTRIBUTES, **attributes})
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, names)
    return _Stored(contents, names, validator, arguments)


_NAN = np.nan
# Features 0, 1 and 4 of five are selected, and feature 1 is untested in the
# first lineage. Statistics reach their bounds: feature 4 correlates perfectly,
# and feature 0 has p-values of one in the second lineage.
_STAT_SELECTED = np.array([0, 1, 4])
_R_VALUES = np.array(
    [[0.5, 0.0], [_NAN, 0.75], [_NAN, _NAN], [_NAN, _NAN], [-1.0, 1.0]]
)
_P_VALUES = np.array(
    [[0.01, 1.0], [_NAN, 0.001], [_NAN, _NAN], [_NAN, _NAN], [0.0, 0.0]]
)
_ADJUSTED_P_VALUES = np.array(
    [[0.02, 1.0], [_NAN, 0.002], [_NAN, _NAN], [_NAN, _NAN], [0.0, 0.0]]
)
_CI_LOW = np.array([[0.2, -0.4], [_NAN, 0.5], [_NAN, _NAN], [_NAN, _NAN], [-1.0, 1.0]])
_CI_HIGH = np.array([[0.7, 0.4], [_NAN, 0.9], [_NAN, _NAN], [_NAN, _NAN], [-1.0, 1.0]])


def _marker_payload(chunk_rows: int | None) -> _Stored:
    return _stored_payload(
        {
            "r_value": _R_VALUES,
            "p_value": _P_VALUES,
            "p_value_adjusted": _ADJUSTED_P_VALUES,
            "effective_cells": np.array([12.5, 8.0]),
        },
        LINEAGE_MARKER_PAYLOAD,
        lineage_marker_payload_is_valid,
        {"n_features": 5, "n_lineages": 2, "selected_features": _STAT_SELECTED},
        chunk_rows=chunk_rows,
        n_features=5,
    )


def _driver_payload(chunk_rows: int | None) -> _Stored:
    return _stored_payload(
        {
            "r_value": _R_VALUES,
            "p_value": _P_VALUES,
            "p_value_adjusted": _ADJUSTED_P_VALUES,
            "ci_low": _CI_LOW,
            "ci_high": _CI_HIGH,
        },
        LINEAGE_DRIVER_PAYLOAD,
        lineage_driver_payload_is_valid,
        {
            "n_features": 5,
            "n_lineages": 2,
            "n_cells": 40,
            "selected_features": _STAT_SELECTED,
        },
        chunk_rows=chunk_rows,
        n_features=5,
        n_cells=40,
    )


# Features 0, 2, 3 and 4 of six are profiled over two lineages of three bins.
# Feature 2 is excluded from the modules, so its profile rows are zero.
_PROFILED = np.array([0, 2, 3, 4])
_PROFILES = np.array(
    [
        [[-1.0, 0.0, 1.0], [0.5, 0.5, -1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.2, 0.1, -1.3], [0.0, 0.0, 0.0]],
        [[0.3, -0.3, 0.0], [1.0, -1.0, 0.0]],
    ]
)
_INPUT_FINGERPRINTS = ["cells", "features", "ordering", "weights"]


def _aggregation_payload(chunk_rows: int | None) -> _Stored:
    return _stored_payload(
        {
            "data": _PROFILES,
            "feature_indices": _PROFILED.astype(np.uint64),
            "valid_features": np.array([True, False, True, True]),
            "feature_clusters": np.array([1, -1, 2, 1], dtype=np.int64),
            "cluster_values": np.array([1, -1, -1, 2, 1, -1], dtype=np.int64),
            # Bins may share a pseudotime, and a lineage may weigh exactly
            # one cell per bin.
            "bin_pseudotime": np.array([[0.1, 0.1, 0.6], [0.0, 0.5, 1.0]]),
            "lineage_weight": np.array([3.0, 7.5]),
        },
        LINEAGE_AGGREGATION_PAYLOAD,
        lineage_aggregation_payload_is_valid,
        {
            "n_features": 6,
            "selected_features": _PROFILED,
            "n_lineages": 2,
            "n_bins": 3,
            "n_clusters": 2,
            "n_neighbours": 2,
            "nan_cluster_value": -1,
            "expected_input_fingerprints": list(_INPUT_FINGERPRINTS),
        },
        chunk_rows=chunk_rows,
        n_features=6,
        input_fingerprints=list(_INPUT_FINGERPRINTS),
    )


# Two curves of four points in two dimensions; cells 2 and 3 are unscored.
_CURVE_VALID = np.array([True, True, False, False, True, True])


def _curve_payload(chunk_rows: int | None) -> _Stored:
    return _stored_payload(
        {
            "points": np.array(
                [
                    [[0.0, 0.0], [1.0, 0.5], [2.0, 1.5], [3.0, 3.0]],
                    [[0.0, 0.0], [1.0, -0.5], [2.0, -1.5], [3.0, -3.0]],
                ]
            ),
            "curve_pseudotime": np.array([[0.0, 0.2, 0.2, 0.9], [0.0, 0.4, 0.7, 1.0]]),
            "progress": np.array(
                [
                    [0.0, 0.0],
                    [0.3, 0.25],
                    [_NAN, _NAN],
                    [_NAN, _NAN],
                    [0.9, 0.6],
                    [1.0, 1.0],
                ]
            ),
            "distance": np.array(
                [
                    [0.0, 0.0],
                    [0.4, 1.2],
                    [_NAN, _NAN],
                    [_NAN, _NAN],
                    [2.5, 0.1],
                    [0.05, 3.0],
                ]
            ),
            "valid": _CURVE_VALID,
        },
        LINEAGE_CURVE_PAYLOAD,
        lineage_curve_payload_is_valid,
        {
            "n_cells": 6,
            "n_lineages": 2,
            "n_points": 4,
            "n_dims": 2,
            "expected_valid": _CURVE_VALID,
        },
        chunk_rows=chunk_rows,
    )


def _tree_payload(chunk_rows: int | None) -> _Stored:
    """Store a tree that joins clusters 0, 2 and 3 to cluster 1.

    Clusters 1 and 2 share a mean, so the edge between them has length zero.
    """
    return _stored_payload(
        {
            "sizes": np.array([5, 3, 4, 1], dtype=np.int64),
            "means": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, -1.0]]),
            "edges": np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64),
            "lengths": np.array([1.5, 0.0, 2.25]),
        },
        LINEAGE_TREE_PAYLOAD,
        lineage_tree_payload_is_valid,
        {"n_clusters": 4, "n_dims": 2},
        chunk_rows=chunk_rows,
        cluster_labels=["stem", "mid", "A", "B"],
    )


_PAYLOADS: dict[str, Callable[[int | None], _Stored]] = {
    "lineage_aggregation": _aggregation_payload,
    "lineage_curves": _curve_payload,
    "lineage_drivers": _driver_payload,
    "lineage_markers": _marker_payload,
    "lineage_tree": _tree_payload,
}


@cache
def _pristine(kind: str, chunk_rows: int | None) -> _Stored:
    stored = _PAYLOADS[kind](chunk_rows)
    group = zarr.open_group(
        store=MemoryStore(stored.contents, read_only=True),
        mode="r",
        path="candidate",
    )
    # A tamper test shows a check only when its change refuses a valid payload.
    assert stored.validator(group, **stored.arguments), f"{kind} payload is invalid"
    return stored


def _candidate(kind: str, *, chunk_rows: int | None = None) -> _Candidate:
    """Return a private copy of one kind's valid payload, which a test may change.

    Each payload is stored and validated once; every test changes a copy of its
    stored bytes.
    """
    stored = _pristine(kind, chunk_rows)
    contents = {
        key: type(value).from_bytes(value.to_bytes())
        for key, value in stored.contents.items()
    }
    group = zarr.open_group(store=MemoryStore(contents), mode="r+", path="candidate")
    return _Candidate(group, stored.names, stored.validator, dict(stored.arguments))


_FEATURE_KINDS = ("lineage_aggregation", "lineage_drivers", "lineage_markers")


def _set(name: str, index: Any, value: Any) -> _Tamper:
    """Write ``value`` at ``index`` of one array and refresh the fingerprint."""

    def tamper(candidate: _Candidate) -> None:
        candidate.group[name][index] = value
        candidate.refresh()

    return tamper


def _rewrite(name: str, change: Callable[[np.ndarray], Any]) -> _Tamper:
    """Store ``change`` of one array's values in its place and refresh."""

    def tamper(candidate: _Candidate) -> None:
        _store(candidate.group, name, change(np.asarray(candidate.group[name][...])))
        candidate.refresh()

    return tamper


def _update(**attributes: Any) -> _Tamper:
    def tamper(candidate: _Candidate) -> None:
        candidate.group.attrs.update(attributes)

    return tamper


def _drop_attribute(name: str) -> _Tamper:
    def tamper(candidate: _Candidate) -> None:
        del candidate.group.attrs[name]

    return tamper


def _annotate(name: str) -> _Tamper:
    """Give one array an attribute; payload fingerprints do not cover them."""

    def tamper(candidate: _Candidate) -> None:
        candidate.group[name].attrs["missing_mask"] = "valid"

    return tamper


def _drop_first_array(candidate: _Candidate) -> None:
    del candidate.group[candidate.names[0]]


def _annotate_first_array(candidate: _Candidate) -> None:
    _annotate(candidate.names[0])(candidate)


_COMMON_TAMPERS: dict[str, _Tamper] = {
    "unknown-attribute": _update(schema_version=1),
    "missing-attribute": _drop_attribute("created_at_ns"),
    "missing-array": _drop_first_array,
    "extra-array": lambda candidate: candidate.group.create_array(
        "notes", data=np.zeros(2)
    ),
    "nested-group": lambda candidate: candidate.group.create_group("notes"),
    "other-fingerprint": _update(payload_fingerprint="0" * 64),
    "annotated-array": _annotate_first_array,
}


# One-row chunks put unselected features, excluded features and unscored cells
# in blocks of their own.
@pytest.mark.parametrize("chunk_rows", [None, 1], ids=["whole-chunks", "row-chunks"])
@pytest.mark.parametrize("kind", sorted(_PAYLOADS))
def test_lineage_payloads_as_their_producers_store_them_are_valid(
    kind: str,
    chunk_rows: int | None,
) -> None:
    assert _candidate(kind, chunk_rows=chunk_rows).is_valid()


@pytest.mark.parametrize("tamper", sorted(_COMMON_TAMPERS))
@pytest.mark.parametrize("kind", sorted(_PAYLOADS))
def test_lineage_payloads_reject_changed_records_and_array_sets(
    kind: str,
    tamper: str,
) -> None:
    candidate = _candidate(kind)

    _COMMON_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


@pytest.mark.parametrize("kind", sorted(_PAYLOADS))
def test_lineage_payload_fingerprints_cover_their_stored_values(kind: str) -> None:
    candidate = _candidate(kind)
    array = candidate.group[candidate.names[0]]
    values = np.asarray(array[...])
    # The change keeps the array within its contract.
    array[...] = values + 1 if values.dtype.kind == "i" else values * 0.5

    assert not candidate.is_valid()
    candidate.refresh()
    assert candidate.is_valid()


_LAST_BLOCK_TAMPERS: dict[str, _Tamper] = {
    "lineage_aggregation": _set("data", (3, 1, 2), _NAN),
    "lineage_curves": _set("distance", (5, 1), -1.0),
    "lineage_drivers": _set("ci_high", (4, 0), 2.0),
    "lineage_markers": _set("r_value", (4, 1), 1.5),
}


@pytest.mark.parametrize("kind", sorted(_LAST_BLOCK_TAMPERS))
def test_lineage_payloads_are_checked_in_every_row_block(kind: str) -> None:
    candidate = _candidate(kind, chunk_rows=1)

    _LAST_BLOCK_TAMPERS[kind](candidate)
    assert not candidate.is_valid()


def _undecodable(values: np.ndarray) -> np.ndarray:
    encoded = [value.encode("utf-8") for value in values.astype(str)]
    return np.array([b"\xff", *encoded[1:]], dtype="S4")


_IDENTITY_TAMPERS: dict[str, _Tamper] = {
    "renamed-feature": _set("feature_names", 1, "gX"),
    "reordered-ids": _rewrite("feature_ids", lambda ids: ids[::-1]),
    "short-names": _rewrite("feature_names", lambda names: names[:-1]),
    "numeric-ids": _rewrite("feature_ids", lambda ids: np.arange(len(ids))),
    "undecodable-ids": _rewrite("feature_ids", _undecodable),
    "annotated-names": _annotate("feature_names"),
}


# Byte-string identities exercise decoding, which Zarr warns has no V3 spec.
@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize("tamper", sorted(_IDENTITY_TAMPERS))
@pytest.mark.parametrize("kind", _FEATURE_KINDS)
def test_feature_payloads_reject_changed_feature_identities(
    kind: str,
    tamper: str,
) -> None:
    candidate = _candidate(kind)

    _IDENTITY_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


_STATISTIC_TAMPERS: dict[str, _Tamper] = {
    "correlation-above-one": _set("r_value", (0, 0), 1.5),
    "correlation-below-minus-one": _set("r_value", (4, 0), -1.01),
    "infinite-correlation": _set("r_value", (1, 1), np.inf),
    "negative-p-value": _set("p_value", (0, 0), -0.01),
    "p-value-above-one": _set("p_value", (1, 1), 1.5),
    "infinite-adjusted-p-value": _set("p_value_adjusted", (0, 1), -np.inf),
    "adjusted-p-value-above-one": _set("p_value_adjusted", (4, 1), 1.01),
    "correlation-of-unselected-feature": _set("r_value", (2, 0), 0.1),
    "p-value-of-unselected-feature": _set("p_value", (3, 1), 0.5),
    "adjusted-p-value-of-unselected-feature": _set("p_value_adjusted", (2, 1), 0.5),
    "integer-correlations": _rewrite(
        "r_value", lambda values: np.zeros(values.shape, dtype=np.int64)
    ),
    "short-correlations": _rewrite("r_value", lambda values: values[:-1]),
    "p-values-of-one-lineage": _rewrite("p_value", lambda values: values[:, 0]),
    "transposed-adjusted-p-values": _rewrite(
        "p_value_adjusted", lambda values: values.T
    ),
    "correlations-of-an-extra-lineage": _rewrite(
        "r_value",
        lambda values: np.hstack([values, np.full((len(values), 1), _NAN)]),
    ),
}


@pytest.mark.parametrize("tamper", sorted(_STATISTIC_TAMPERS))
@pytest.mark.parametrize("kind", ["lineage_drivers", "lineage_markers"])
def test_marker_and_driver_payloads_reject_each_tampered_statistic(
    kind: str,
    tamper: str,
) -> None:
    candidate = _candidate(kind)

    _STATISTIC_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


@pytest.mark.parametrize("kind", ["lineage_drivers", "lineage_markers"])
def test_correlations_may_round_just_beyond_one(kind: str) -> None:
    candidate = _candidate(kind)
    candidate.group["r_value"][4, 1] = 1.0 + 1e-13
    candidate.refresh()
    assert candidate.is_valid()

    candidate.group["r_value"][4, 1] = 1.0 + 1e-9
    candidate.refresh()
    assert not candidate.is_valid()


_MARKER_TAMPERS: dict[str, _Tamper] = {
    "zero-effective-cells": _set("effective_cells", 0, 0.0),
    "negative-effective-cells": _set("effective_cells", 1, -8.0),
    "nan-effective-cells": _set("effective_cells", 0, _NAN),
    "infinite-effective-cells": _set("effective_cells", 1, np.inf),
    "integer-effective-cells": _rewrite(
        "effective_cells", lambda values: values.astype(np.int64)
    ),
    "effective-cells-per-feature": _rewrite(
        "effective_cells", lambda values: np.ones(5)
    ),
    "annotated-effective-cells": _annotate("effective_cells"),
}


@pytest.mark.parametrize("tamper", sorted(_MARKER_TAMPERS))
def test_marker_payload_rejects_each_tampered_effective_size(tamper: str) -> None:
    candidate = _candidate("lineage_markers")

    _MARKER_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


_DRIVER_TAMPERS: dict[str, _Tamper] = {
    "missing-cell-count": _drop_attribute("n_cells"),
    "other-cell-count": _update(n_cells=41),
    "real-cell-count": _update(n_cells=40.0),
    "text-cell-count": _update(n_cells="40"),
    "lower-bound-above-one": _set("ci_low", (0, 0), 1.5),
    "upper-bound-below-minus-one": _set("ci_high", (0, 1), -1.5),
    "infinite-lower-bound": _set("ci_low", (1, 1), -np.inf),
    "upper-bound-of-unselected-feature": _set("ci_high", (3, 0), 0.3),
    "integer-upper-bounds": _rewrite(
        "ci_high", lambda values: np.zeros(values.shape, dtype=np.int64)
    ),
    "short-lower-bounds": _rewrite("ci_low", lambda values: values[:-1]),
    "annotated-lower-bounds": _annotate("ci_low"),
}


@pytest.mark.parametrize("tamper", sorted(_DRIVER_TAMPERS))
def test_driver_payload_rejects_each_tampered_record(tamper: str) -> None:
    candidate = _candidate("lineage_drivers")

    _DRIVER_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


@pytest.mark.parametrize("n_cells", [0, 1])
def test_driver_payload_refuses_a_boolean_cell_count(n_cells: int) -> None:
    candidate = _candidate("lineage_drivers")
    candidate.group.attrs["n_cells"] = n_cells
    assert candidate.is_valid(n_cells=n_cells)

    candidate.group.attrs["n_cells"] = bool(n_cells)
    assert not candidate.is_valid(n_cells=n_cells)


def _exclude(*positions: int) -> _Tamper:
    """Exclude profiled features from the modules, consistently everywhere."""

    def tamper(candidate: _Candidate) -> None:
        group = candidate.group
        for position in positions:
            group["valid_features"][position] = False
            group["data"][position] = 0.0
            group["feature_clusters"][position] = -1
            group["cluster_values"][int(_PROFILED[position])] = -1
        candidate.refresh()

    return tamper


def _relabel(position: int, label: int) -> _Tamper:
    """Give a profiled feature another module in both module arrays."""

    def tamper(candidate: _Candidate) -> None:
        candidate.group["feature_clusters"][position] = label
        candidate.group["cluster_values"][int(_PROFILED[position])] = label
        candidate.refresh()

    return tamper


_AGGREGATION_TAMPERS: dict[str, _Tamper] = {
    "missing-input-fingerprints": _drop_attribute("input_fingerprints"),
    "other-input-fingerprints": _update(
        input_fingerprints=["cells", "features", "ordering", "other"]
    ),
    "reordered-input-fingerprints": _update(
        input_fingerprints=["features", "cells", "ordering", "weights"]
    ),
    # The pseudotime aggregation records this attribute; lineages do not.
    "pseudotime-aggregation-attribute": _update(nan_cluster_value=-1),
    "flattened-profiles": _rewrite(
        "data", lambda values: values.reshape(len(values), -1)
    ),
    "integer-profiles": _rewrite("data", lambda values: values.astype(np.int64)),
    "signed-feature-indices": _rewrite(
        "feature_indices", lambda values: values.astype(np.int64)
    ),
    "integer-validity": _rewrite(
        "valid_features", lambda values: values.astype(np.int8)
    ),
    "unsigned-feature-modules": _rewrite(
        "feature_clusters", lambda values: np.abs(values).astype(np.uint64)
    ),
    "real-module-values": _rewrite(
        "cluster_values", lambda values: values.astype(np.float64)
    ),
    "module-values-of-profiled-features": _rewrite(
        "cluster_values", lambda values: values[_PROFILED]
    ),
    "transposed-bin-pseudotime": _rewrite("bin_pseudotime", lambda values: values.T),
    "integer-lineage-weight": _rewrite(
        "lineage_weight", lambda values: np.ceil(values).astype(np.int64)
    ),
    "decreasing-bin-pseudotime": _set("bin_pseudotime", (1, 2), 0.4),
    "nan-bin-pseudotime": _set("bin_pseudotime", (0, 1), _NAN),
    "infinite-bin-pseudotime": _set("bin_pseudotime", (1, 2), np.inf),
    "lineage-lighter-than-its-bins": _set("lineage_weight", 0, 2.5),
    "nan-lineage-weight": _set("lineage_weight", 1, _NAN),
    "infinite-lineage-weight": _set("lineage_weight", 1, np.inf),
    "shifted-feature-indices": _set("feature_indices", 1, 1),
    "reordered-feature-indices": _rewrite(
        "feature_indices", lambda values: values[[1, 0, 2, 3]]
    ),
    "one-valid-feature": _exclude(2, 3),
    "as-many-valid-features-as-neighbours": _exclude(3),
    "unassigned-module-for-valid-feature": _relabel(0, 0),
    "module-beyond-the-modules": _relabel(0, 3),
    "other-unassigned-module": _set("feature_clusters", 1, -2),
    "profile-of-excluded-feature": _set("data", (1, 0, 1), 0.5),
    "nan-profile": _set("data", (0, 1, 2), _NAN),
    "infinite-profile": _set("data", (3, 0, 0), -np.inf),
    "relabelled-module-value": _set("cluster_values", 0, 2),
    "module-value-of-unselected-feature": _set("cluster_values", 5, 1),
    "module-value-of-excluded-feature": _set("cluster_values", 2, 1),
}


@pytest.mark.parametrize("tamper", sorted(_AGGREGATION_TAMPERS))
def test_aggregation_payload_rejects_each_tampered_record(tamper: str) -> None:
    candidate = _candidate("lineage_aggregation")

    _AGGREGATION_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


def test_aggregation_payload_tolerates_rounding_in_bin_pseudotime() -> None:
    candidate = _candidate("lineage_aggregation")
    candidate.group["bin_pseudotime"][0, 1] = 0.1 - 1e-13
    candidate.refresh()
    assert candidate.is_valid()

    candidate.group["bin_pseudotime"][0, 1] = 0.1 - 1e-9
    candidate.refresh()
    assert not candidate.is_valid()


def test_aggregation_payload_allows_a_module_for_each_valid_feature() -> None:
    # Three features are valid: one or two neighbours and up to three modules.
    candidate = _candidate("lineage_aggregation")

    assert candidate.is_valid(n_neighbours=1, n_clusters=3)


def _score_unscored_cell(candidate: _Candidate) -> None:
    group = candidate.group
    group["valid"][2] = True
    group["progress"][2] = 0.5
    group["distance"][2] = 0.25
    candidate.refresh()


_CURVE_TAMPERS: dict[str, _Tamper] = {
    "flattened-points": _rewrite("points", lambda values: values.reshape(2, -1)),
    "integer-points": _rewrite("points", lambda values: values.astype(np.int64)),
    "pseudotime-of-one-curve": _rewrite("curve_pseudotime", lambda values: values[0]),
    "short-progress": _rewrite("progress", lambda values: values[:-1]),
    "integer-distance": _rewrite(
        "distance", lambda values: np.zeros(values.shape, dtype=np.int64)
    ),
    "integer-validity": _rewrite("valid", lambda values: values.astype(np.int8)),
    "annotated-progress": _annotate("progress"),
    "nan-point": _set("points", (0, 1, 0), _NAN),
    "infinite-point": _set("points", (1, 3, 1), np.inf),
    "nan-curve-pseudotime": _set("curve_pseudotime", (1, 2), _NAN),
    "decreasing-curve-pseudotime": _set("curve_pseudotime", (0, 3), 0.1),
    "scored-cell-outside-the-fate-map": _score_unscored_cell,
    "progress-above-one": _set("progress", (5, 0), 1.5),
    "negative-progress": _set("progress", (1, 1), -0.1),
    "nan-progress-of-scored-cell": _set("progress", (4, 0), _NAN),
    "infinite-progress": _set("progress", (0, 1), np.inf),
    "negative-distance": _set("distance", (5, 1), -0.5),
    "nan-distance-of-scored-cell": _set("distance", (0, 1), _NAN),
    "infinite-distance": _set("distance", (1, 0), np.inf),
    "progress-of-unscored-cell": _set("progress", (2, 0), 0.5),
    "distance-of-unscored-cell": _set("distance", (3, 1), 0.0),
}


@pytest.mark.parametrize("tamper", sorted(_CURVE_TAMPERS))
def test_curve_payload_rejects_each_tampered_array(tamper: str) -> None:
    candidate = _candidate("lineage_curves")

    _CURVE_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


def _tree_edges(edges: list[list[int]], lengths: list[float]) -> _Tamper:
    def tamper(candidate: _Candidate) -> None:
        _store(candidate.group, "edges", np.asarray(edges, dtype=np.int64))
        _store(candidate.group, "lengths", np.asarray(lengths, dtype=np.float64))
        candidate.refresh()

    return tamper


_TREE_TAMPERS: dict[str, _Tamper] = {
    "missing-cluster-labels": _drop_attribute("cluster_labels"),
    "cluster-labels-as-text": _update(cluster_labels="SMAB"),
    "cluster-labels-as-mapping": _update(
        cluster_labels={"0": "stem", "1": "mid", "2": "A", "3": "B"}
    ),
    "fewer-cluster-labels": _update(cluster_labels=["stem", "mid", "A"]),
    "more-cluster-labels": _update(cluster_labels=["stem", "mid", "A", "B", "C"]),
    "real-sizes": _rewrite("sizes", lambda values: values.astype(np.float64)),
    "one-coordinate-of-each-mean": _rewrite("means", lambda values: values[:, 0]),
    "too-few-edges": _tree_edges([[0, 1], [1, 2]], [1.5, 0.0]),
    "too-many-edges": _tree_edges(
        [[0, 1], [0, 2], [1, 2], [1, 3]], [1.5, 1.5, 0.0, 2.25]
    ),
    "unsigned-edges": _rewrite("edges", lambda values: values.astype(np.uint64)),
    "integer-lengths": _rewrite("lengths", lambda values: values.astype(np.int64)),
    "length-per-cluster": _rewrite("lengths", lambda values: np.ones(4)),
    "empty-cluster": _set("sizes", 3, 0),
    "nan-mean": _set("means", (2, 1), _NAN),
    "infinite-length": _set("lengths", 0, np.inf),
    "nan-length": _set("lengths", 2, _NAN),
    "negative-length": _set("lengths", 1, -0.5),
    "negative-cluster-index": _set("edges", (0, 0), -1),
    "cluster-index-out-of-range": _set("edges", (2, 1), 4),
    "reversed-edge": _set("edges", 0, [1, 0]),
    "self-loop": _set("edges", 1, [2, 2]),
    "cycle-and-isolated-cluster": _tree_edges(
        [[0, 1], [0, 2], [1, 2]], [1.5, 1.5, 0.0]
    ),
    "repeated-edge": _tree_edges([[0, 1], [0, 1], [1, 3]], [1.5, 1.5, 2.25]),
}


@pytest.mark.parametrize("tamper", sorted(_TREE_TAMPERS))
def test_tree_payload_rejects_each_tampered_record(tamper: str) -> None:
    candidate = _candidate("lineage_tree")

    _TREE_TAMPERS[tamper](candidate)
    assert not candidate.is_valid()


@pytest.mark.parametrize(
    "edges",
    [
        pytest.param([[0, 1], [1, 2], [2, 3]], id="path"),
        pytest.param([[0, 3], [1, 3], [2, 3]], id="star"),
    ],
)
def test_tree_payload_accepts_any_spanning_tree(edges: list[list[int]]) -> None:
    candidate = _candidate("lineage_tree")

    _tree_edges(edges, [1.0, 1.0, 1.0])(candidate)
    assert candidate.is_valid()


@pytest.mark.parametrize(
    ("kind", "changes"),
    [
        pytest.param("lineage_markers", {"n_features": 6}, id="markers-features"),
        pytest.param("lineage_markers", {"n_lineages": 3}, id="markers-lineages"),
        pytest.param(
            "lineage_markers",
            {"selected_features": np.array([0, 1])},
            id="markers-tested-feature-outside-the-selection",
        ),
        pytest.param("lineage_drivers", {"n_cells": 41}, id="drivers-cells"),
        pytest.param("lineage_drivers", {"n_lineages": 1}, id="drivers-lineages"),
        pytest.param(
            "lineage_drivers",
            {"selected_features": np.array([1, 4])},
            id="drivers-tested-feature-outside-the-selection",
        ),
        pytest.param(
            "lineage_aggregation", {"n_features": 7}, id="aggregation-features"
        ),
        pytest.param(
            "lineage_aggregation",
            {"selected_features": np.array([0, 2, 3, 5])},
            id="aggregation-other-selection",
        ),
        pytest.param(
            "lineage_aggregation",
            {"selected_features": np.array([0, 2, 3])},
            id="aggregation-smaller-selection",
        ),
        pytest.param(
            "lineage_aggregation", {"n_lineages": 3}, id="aggregation-lineages"
        ),
        pytest.param("lineage_aggregation", {"n_bins": 4}, id="aggregation-bins"),
        pytest.param(
            "lineage_aggregation",
            {"n_neighbours": 3},
            id="aggregation-neighbours-for-every-valid-feature",
        ),
        pytest.param(
            "lineage_aggregation", {"n_neighbours": 0}, id="aggregation-no-neighbours"
        ),
        pytest.param(
            "lineage_aggregation",
            {"n_clusters": 4},
            id="aggregation-more-modules-than-valid-features",
        ),
        pytest.param(
            "lineage_aggregation", {"n_clusters": 0}, id="aggregation-no-modules"
        ),
        pytest.param(
            "lineage_aggregation",
            {"nan_cluster_value": -2},
            id="aggregation-other-unassigned-module",
        ),
        pytest.param(
            "lineage_aggregation",
            {
                "expected_input_fingerprints": [
                    "weights",
                    "ordering",
                    "features",
                    "cells",
                ]
            },
            id="aggregation-other-inputs",
        ),
        pytest.param("lineage_curves", {"n_cells": 7}, id="curves-cells"),
        pytest.param("lineage_curves", {"n_lineages": 1}, id="curves-lineages"),
        pytest.param("lineage_curves", {"n_points": 5}, id="curves-points"),
        pytest.param("lineage_curves", {"n_dims": 3}, id="curves-dimensions"),
        pytest.param(
            "lineage_curves",
            {"expected_valid": np.array([True, True, True, False, True, True])},
            id="curves-other-validity",
        ),
        pytest.param(
            "lineage_curves",
            {"expected_valid": np.ones(5, dtype=bool)},
            id="curves-short-validity",
        ),
        pytest.param("lineage_tree", {"n_clusters": 5}, id="tree-more-clusters"),
        pytest.param("lineage_tree", {"n_clusters": 3}, id="tree-fewer-clusters"),
        pytest.param("lineage_tree", {"n_dims": 3}, id="tree-dimensions"),
        *(
            pytest.param(kind, {argument: "0" * 64}, id=f"{kind}-{label}")
            for kind in _FEATURE_KINDS
            for argument, label in (
                ("expected_feature_ids_fingerprint", "feature-ids"),
                ("expected_feature_names_fingerprint", "feature-names"),
            )
        ),
    ],
)
def test_lineage_payloads_must_match_their_expected_axes(
    kind: str,
    changes: dict[str, Any],
) -> None:
    candidate = _candidate(kind)

    assert not candidate.is_valid(**changes)
