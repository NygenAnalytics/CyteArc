from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.storage.arrays import create_metadata_column
from cytearc.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from cytearc.storage.artifacts import (
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
)
from cytearc.storage.selections import (
    resolve_metadata_snapshot,
    resolve_stored_selection_artifact,
)
from cytearc.trajectory.artifacts import (
    AGGREGATION_PAYLOAD,
    FATE_PAYLOAD,
    LINEAGE_AGGREGATION_PAYLOAD,
    LINEAGE_DRIVER_PAYLOAD,
    LINEAGE_MARKER_PAYLOAD,
    MARKER_PAYLOAD,
    PSEUDOTIME_PAYLOAD,
    aggregation_check_bytes,
    aggregation_load_bytes,
    aggregation_payload_is_valid,
    fate_payload_is_valid,
    lineage_aggregation_check_bytes,
    lineage_aggregation_load_bytes,
    lineage_aggregation_payload_is_valid,
    lineage_driver_payload_is_valid,
    lineage_marker_payload_is_valid,
    lineage_statistics_check_bytes,
    lineage_statistics_load_bytes,
    listed_labels,
    listed_labels_bytes,
    load_cell_artifact_values,
    marker_check_bytes,
    marker_load_bytes,
    marker_payload_is_valid,
    pseudotime_payload_is_valid,
    validate_aggregation_parameters,
    validate_fate_parameters,
    validate_marker_parameters,
    validate_pseudotime_parameters,
    validate_resolved_ann_parameters,
)
from cytearc.trajectory.parameters import resolve_aggregation_ann_params
from tests.memory_helpers import traced_charge, traced_reruns


def test_diffusion_loading_validates_and_builds_from_one_read_per_array(monkeypatch):
    from cytearc.trajectory.artifacts import load_diffusion_payload

    root = zarr.group()
    payload = {
        "row": np.array([0, 1, 2], dtype=np.uint64),
        "col": np.array([0, 1, 2], dtype=np.uint64),
        "data": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }
    for name, values in payload.items():
        root.create_array(name, data=values)
    root.attrs.update(
        {
            "artifact_id": "a" * 64,
            "kind": "diffusion_operator",
            "provenance": {},
            "execution_options": {},
            "complete": True,
            "created_at_ns": 1,
            "cytearc_version": "test",
            "n_cells": 3,
            "payload_fingerprint": fingerprint_stored_arrays(root, tuple(payload)),
        }
    )
    reads = []
    original = zarr.Array.__getitem__

    def read(array, key):
        reads.append(array.path)
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", read)
    with pytest.raises(MemoryError, match="exceed the memory budget"):
        load_diffusion_payload(root, n_cells=3, memory_bytes=1)
    assert reads == []
    operator = load_diffusion_payload(root, n_cells=3, memory_bytes=4096)
    assert reads == ["row", "col", "data"]
    np.testing.assert_array_equal(operator.toarray(), np.diag([1.0, 2.0, 3.0]))


def _payload_group(payload: dict[str, np.ndarray]) -> zarr.Group:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("candidate")
    for name, values in payload.items():
        group.create_array(name, data=np.asarray(values))
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group,
        tuple(payload),
    )
    group.attrs.update(
        {
            "artifact_id": "a" * 64,
            "kind": "test_payload",
            "provenance": {},
            "execution_options": {},
            "created_at_ns": 1,
            "cytearc_version": "test",
            "complete": True,
        }
    )
    return group


def _refresh_payload_fingerprint(group: zarr.Group, names: tuple[str, ...]) -> None:
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, names)


def _selection_store() -> tuple[zarr.Group, object]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    create_metadata_column(
        table,
        "ids",
        data=np.array(["c0", "c1", "c2"]),
        dtype=str,
    )
    create_metadata_column(
        table,
        "I",
        data=np.array([True, False, True]),
        dtype=bool,
    )
    selection = resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="manual_selection",
        parameters={},
        inputs={},
    )
    return root, selection


def test_pseudotime_payload_rejects_same_shape_semantic_corruption() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.5, 1.0, np.nan]),
            "valid": np.array([True, True, True, False]),
        }
    )
    assert pseudotime_payload_is_valid(
        group,
        n_cells=4,
        min_max_normalized=True,
        expected_valid=np.array([True, True, True, False]),
    )

    group["pseudotime"][1] = np.nan
    _refresh_payload_fingerprint(group, PSEUDOTIME_PAYLOAD)
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=4,
        min_max_normalized=True,
        expected_valid=np.array([True, True, True, False]),
    )


def test_pseudotime_payload_requires_a_nonconstant_exact_normalized_range() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.25, 0.5, 0.75]),
            "valid": np.array([True, True, True]),
        }
    )
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.ones(3, dtype=bool),
    )

    group["pseudotime"][:] = np.array([0.5, 0.5, 0.5])
    _refresh_payload_fingerprint(group, PSEUDOTIME_PAYLOAD)
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=False,
        expected_valid=np.ones(3, dtype=bool),
    )


def test_pseudotime_payload_requires_the_exact_component_validity_mask() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 1.0, np.nan]),
            "valid": np.array([True, True, False]),
        }
    )
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.array([True, False, True]),
    )


def test_pseudotime_payload_rejects_array_attribute_tampering() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.5, 1.0]),
            "valid": np.array([True, True, True]),
        }
    )
    group["pseudotime"].attrs["missing_mask"] = "valid"

    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.ones(3, dtype=bool),
    )


def test_pseudotime_payload_rejects_unknown_artifact_attributes() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.5, 1.0]),
            "valid": np.array([True, True, True]),
        }
    )
    group.attrs["schema_version"] = 1

    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.ones(3, dtype=bool),
    )


def test_fate_payload_rejects_invalid_rows_and_boundary_columns() -> None:
    labels = np.array(["A", "middle", "B", "missing"], dtype=object)
    group = _payload_group(
        {
            "probabilities": np.array(
                [[1.0, 0.0], [0.4, 0.6], [0.0, 1.0], [np.nan, np.nan]],
                dtype=np.float32,
            ),
            "valid": np.array([True, True, True, False]),
        }
    )
    assert fate_payload_is_valid(
        group,
        n_cells=4,
        n_sinks=2,
        expected_valid=np.array([True, True, True, False]),
        sink_values=labels,
        sink_labels=["A", "B"],
    )

    group["probabilities"][0] = np.array([0.0, 1.0], dtype=np.float32)
    _refresh_payload_fingerprint(group, FATE_PAYLOAD)
    assert not fate_payload_is_valid(
        group,
        n_cells=4,
        n_sinks=2,
        expected_valid=np.array([True, True, True, False]),
        sink_values=labels,
        sink_labels=["A", "B"],
    )


def test_marker_payload_freezes_feature_identity_for_reuse() -> None:
    names = np.array(["g0", "g1", "g2"])
    ids = np.array(["id0", "id1", "id2"])
    group = _payload_group(
        {
            "r_value": np.array([0.5, np.nan, -0.4]),
            "p_value": np.array([0.01, np.nan, 0.02]),
            "p_value_adjusted": np.array([0.02, np.nan, 0.03]),
            "feature_names": names,
            "feature_ids": ids,
        }
    )
    expected_ids = fingerprint_stored_strings(group["feature_ids"])
    expected_names = fingerprint_stored_strings(group["feature_names"])
    assert marker_payload_is_valid(
        group,
        n_features=3,
        selected_features=np.array([0, 2]),
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
    )

    group["feature_names"][0] = "renamed"
    _refresh_payload_fingerprint(group, MARKER_PAYLOAD)
    assert not marker_payload_is_valid(
        group,
        n_features=3,
        selected_features=np.array([0, 2]),
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
    )

    group["feature_names"][:] = names
    group["p_value"][0] = np.inf
    _refresh_payload_fingerprint(group, MARKER_PAYLOAD)
    assert not marker_payload_is_valid(
        group,
        n_features=3,
        selected_features=np.array([0, 2]),
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
    )


def test_aggregation_payload_rejects_reordered_feature_rows() -> None:
    names = np.array(["g0", "g1", "g2", "g3"])
    ids = np.array(["id0", "id1", "id2", "id3"])
    group = _payload_group(
        {
            "data": np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
            "feature_indices": np.array([0, 2, 3], dtype=np.uint64),
            "valid_features": np.array([True, False, True]),
            "feature_clusters": np.array([1, -1, 2]),
            "cluster_values": np.array([1, -1, -1, 2]),
            "feature_names": names,
            "feature_ids": ids,
        }
    )
    group.attrs["input_fingerprints"] = ["cells", "features", "ordering"]
    group.attrs["nan_cluster_value"] = -1
    group.attrs["effective_window"] = 2
    group.attrs["effective_bins"] = 2
    expected_ids = fingerprint_stored_strings(group["feature_ids"])
    expected_names = fingerprint_stored_strings(group["feature_names"])
    resolved_ann = resolve_aggregation_ann_params(None, dim=2)
    assert aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbors=1,
        nan_cluster_value=-1,
        ann_params=resolved_ann,
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )
    assert not aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbors=1,
        nan_cluster_value=-1,
        ann_params={},
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )
    assert not aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbors=1,
        nan_cluster_value=-1,
        ann_params={**resolved_ann, "max_elements": 1},
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )

    group["feature_indices"][:] = np.array([2, 0, 3], dtype=np.uint64)
    _refresh_payload_fingerprint(group, AGGREGATION_PAYLOAD)
    assert not aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbors=1,
        nan_cluster_value=-1,
        ann_params=resolved_ann,
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )


def test_raw_source_sink_snapshot_is_replayable_and_selection_aligned() -> None:
    root, selection = _selection_store()
    values = np.array([-1.0, 1.0])
    snapshot = resolve_metadata_snapshot(
        root,
        values=values,
        row_ids=np.array(["c0", "c2"]),
        operation="snapshot_pseudotime_source_sink",
        parameters={},
        inputs={"cell_selection": selection},
        source_columns=["ss_vec"],
    )

    loaded, loaded_selection, missing = load_cell_artifact_values(root, snapshot)
    np.testing.assert_array_equal(loaded, values)
    assert loaded_selection == selection
    assert missing is None
    assert (inspect_artifact(root, snapshot).inputs or {})["cell_selection"] == (
        selection.to_dict()
    )
    # Values that are not an artifact are refused before anything is read.
    with pytest.raises(TypeError, match="^artifact must be an ArtifactRef$"):
        load_cell_artifact_values(root, values)  # type: ignore[arg-type]


def test_nullable_zero_label_is_not_treated_as_an_endpoint() -> None:
    root, selection = _selection_store()
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        operation="import_cluster_labels",
        parameters={},
        inputs={"cell_selection": selection},
        execution_options={},
    )
    group = start_artifact(root, planned)
    values = group.create_array("values", data=np.array([0, 1]))
    group.create_array("__cytearc_missing__values", data=np.array([True, False]))
    values.attrs["missing_mask"] = "__cytearc_missing__values"
    finish_artifact(group, planned)

    loaded, loaded_selection, missing = load_cell_artifact_values(root, planned.ref)
    assert loaded_selection == selection
    assert missing is not None
    masked = listed_labels(loaded, missing, [0, 1], "labels")
    np.testing.assert_array_equal(masked == 0, np.array([False, False]))
    np.testing.assert_array_equal(masked == 1, np.array([False, True]))


def test_trajectory_parameter_contracts_reject_ambiguous_or_coerced_values() -> None:
    pseudotime = {
        "n_singular_vals": np.int64(3),
        "sources": [0],
        "sinks": [1],
        "min_max_norm_ptime": np.bool_(True),
        "random_seed": np.int64(7),
        "component_policy": "largest",
    }
    # A record without a method, as releases before 1.4.0 wrote, holds the
    # truncated potential.
    validated = validate_pseudotime_parameters(pseudotime)
    assert validated["method"] == "truncated"
    assert validated["n_singular_vals"] == 3
    assert type(validated["n_singular_vals"]) is int
    assert validated["min_max_norm_ptime"] is True
    assert validate_pseudotime_parameters({**pseudotime, "method": "truncated"}) == (
        validated
    )
    untruncated = {
        key: value
        for key, value in pseudotime.items()
        if key not in {"n_singular_vals", "random_seed"}
    } | {"method": "untruncated"}
    assert validate_pseudotime_parameters(untruncated) | {
        "n_singular_vals": 3,
        "random_seed": 7,
    } == validated | {"method": "untruncated"}
    assert validate_pseudotime_parameters(untruncated)["n_singular_vals"] is None
    with pytest.raises(ValueError, match="disjoint"):
        validate_pseudotime_parameters({**pseudotime, "sources": [1], "sinks": [True]})
    with pytest.raises(TypeError, match="n_singular_vals"):
        validate_pseudotime_parameters({**pseudotime, "n_singular_vals": True})
    # The seed starts the truncated solve from NumPy's RandomState, which
    # takes 0 to 2**32 - 1.
    largest = validate_pseudotime_parameters({**pseudotime, "random_seed": 2**32 - 1})
    assert largest["random_seed"] == 2**32 - 1
    with pytest.raises(ValueError, match="^random_seed must be at most 4294967295$"):
        validate_pseudotime_parameters({**pseudotime, "random_seed": 2**32})
    with pytest.raises(ValueError, match="method must be 'untruncated' or 'truncated'"):
        validate_pseudotime_parameters({**untruncated, "method": "exact"})
    # Each method records the parameters of its own potential, and no others.
    for parameters in (
        {**untruncated, "n_singular_vals": 3},
        {**pseudotime, "method": "untruncated"},
        {key: value for key, value in pseudotime.items() if key != "random_seed"},
    ):
        with pytest.raises(ValueError, match="does not match its persisted contract"):
            validate_pseudotime_parameters(parameters)

    fate = {
        "sinks": ["terminal"],
        "beta": 10.0,
        "solver_tol": 1e-6,
        "max_iterations": 1000,
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_fate_parameters({**fate, "sinks": [1, 1.0]})
    with pytest.raises(ValueError, match="solver_tol"):
        validate_fate_parameters({**fate, "solver_tol": 0.0})

    marker = {
        "normalization": {
            "log_transform": False,
            "renormalize_subset": False,
        },
        "normalization_method": {"module": "m", "qualname": "normalize"},
        "size_factor": 1000.0,
        "association_method": "pearson",
        "p_value_method": "student_t",
        "adjustment_method": "fdr_bh",
        "adjustment_scope": "tested_features",
        "min_cells": 10,
    }
    with pytest.raises(TypeError, match="log_transform"):
        validate_marker_parameters(
            {
                **marker,
                "normalization": {
                    **marker["normalization"],
                    "log_transform": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="association_method"):
        validate_marker_parameters({**marker, "association_method": "spearman"})

    aggregation = {
        "normalization": marker["normalization"],
        "normalization_method": marker["normalization_method"],
        "size_factor": marker["size_factor"],
        "min_exp": 1e-3,
        "window_size": 20,
        "chunk_size": 10,
        "smoothen": True,
        "z_scale": True,
        "n_neighbours": 2,
        "n_clusters": 3,
        "ann_params": {},
        "nan_cluster_value": -1,
    }
    with pytest.raises(TypeError, match="smoothen"):
        validate_aggregation_parameters({**aggregation, "smoothen": 1})
    with pytest.raises(ValueError, match="conflicts"):
        validate_aggregation_parameters({**aggregation, "nan_cluster_value": 1})
    with pytest.raises(ValueError, match="unsupported"):
        validate_aggregation_parameters({**aggregation, "ann_params": {"ignored": 1}})
    with pytest.raises(ValueError, match="resolved ANN contract"):
        validate_resolved_ann_parameters({}, dim=10)
    with pytest.raises(ValueError, match="effective bin count"):
        resolve_aggregation_ann_params({"dim": 9}, dim=10)
    with pytest.raises(TypeError, match="mapping or None"):
        resolve_aggregation_ann_params([("M", 20)], dim=10)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["a2", "1", "True", "", "b2", "a2"]),
        np.array(["a2", "1", None, "b2", 3, "a2"], dtype=object),
        np.array([2, 1, 0, 3, 1, 2]),
        np.array([2.0, 1.0, np.nan, 3.5, 1.0, 2.0]),
    ],
    ids=["text", "objects", "integers", "floats"],
)
def test_listed_labels_match_as_pandas_matches_them(labels: np.ndarray) -> None:
    # Each listed label, of any type, holds the cells that pandas finds for
    # it, and every other cell holds none of them.
    listed = ["a2", 1, True, 2.0, "", "b2"]
    codes = listed_labels(labels, None, listed, "labels")
    series = pd.Series(labels, dtype=object)
    for label in dict.fromkeys(listed):
        np.testing.assert_array_equal(
            np.asarray(codes == label), series.isin([label]).to_numpy()
        )
    assert codes.codes.dtype == np.int8
    # A missing label holds none of them.
    missing = np.zeros(len(labels), dtype=bool)
    missing[0] = True
    assert listed_labels(labels, missing, listed, "labels").codes[0] == -1


def test_listed_labels_keep_wide_codes_and_refuse_other_shapes() -> None:
    labels = np.array([f"c{index}" for index in range(300)])
    listed = list(labels[::-1])
    codes = listed_labels(labels, None, listed, "labels")
    # More than 127 listed labels need wider codes.
    assert codes.codes.dtype.itemsize > 1
    assert np.asarray(codes == "c0").nonzero()[0].tolist() == [0]
    with pytest.raises(ValueError, match="^Sink labels must be one-dimensional$"):
        listed_labels(np.ones((2, 2)), None, [1.0], "Sink labels")


@pytest.mark.parametrize(
    "labels",
    [
        np.array([("a2" if index % 3 else "t1") + "x" * 24 for index in range(5_000)]),
        np.array(["a2", "b2", None] * 2_000, dtype=object),
        np.arange(5_000) % 4,
        np.arange(5_000) * 1_000_003 + 0.5,
    ],
    ids=["text", "objects", "integers", "floats"],
)
def test_listed_labels_hold_what_their_estimate_states(labels: np.ndarray) -> None:
    traced = traced_charge(
        lambda _ledger: listed_labels(
            labels, None, ["a2" + "x" * 24, "t1" + "x" * 24, 1, 2], "labels"
        )
    )
    assert traced.held <= listed_labels_bytes(len(labels), labels.dtype)


# What checking a saved feature analysis holds


def _feature_payload(
    root: zarr.Group,
    kind: str,
    *,
    n_features: int,
    identity_rows: int,
    n_lineages: int = 3,
    n_bins: int = 20,
    width: int = 15,
    every: int = 10,
) -> tuple[zarr.Group, dict[str, object]]:
    """Store a valid payload of a feature analysis of ``n_features`` features and the arguments that check it.

    Every ``every``-th feature is selected, and every seventh selected one
    has no profile. The statistics are chunked as the analyses write them, at most
    100,000 features to a chunk, and the frozen names and IDs of ``width``
    characters in chunks of ``identity_rows``, as analyses saved before
    their chunks followed a feature table's did with 100,000.
    """
    rng = np.random.default_rng(3)
    selected = np.arange(0, n_features, every)
    n_selected = len(selected)
    rows = min(n_features, 100_000)
    group = root.create_group(kind)
    # A width shorter than the features' numbers keeps their last digits.
    names = np.asarray([f"{index:0{width}d}"[-width:] for index in range(n_features)])
    for name, values in (
        ("feature_names", names),
        ("feature_ids", np.char.upper(names)),
    ):
        group.create_array(name, data=values, chunks=(identity_rows,))
    arguments: dict[str, object] = {
        "n_features": n_features,
        "selected_features": selected,
        "expected_feature_ids_fingerprint": fingerprint_stored_strings(
            group["feature_ids"]
        ),
        "expected_feature_names_fingerprint": fingerprint_stored_strings(
            group["feature_names"]
        ),
    }

    def statistics(columns: int | None, *names: str) -> None:
        shape = (n_features,) if columns is None else (n_features, columns)
        for name in names:
            values = np.full(shape, np.nan)
            values[selected] = rng.uniform(0.0, 1.0, size=values[selected].shape)
            group.create_array(name, data=values, chunks=(rows, *shape[1:]))

    valid = np.arange(n_selected) % 7 != 0
    clusters = np.where(valid, 1 + np.arange(n_selected) % 4, -1)
    cluster_values = np.full(n_features, -1, dtype=np.int64)
    cluster_values[selected[valid]] = clusters[valid]

    def modules(data_shape: tuple[int, ...]) -> None:
        data = rng.normal(size=data_shape)
        data[~valid] = 0.0
        if len(data_shape) == 2:
            # A pseudotime aggregation keeps its profiles in one shard of one
            # chunk, and a lineage aggregation in chunks of about a million
            # values.
            group.create_array("data", data=data, chunks=data_shape, shards=data_shape)
        else:
            band = max(1, min(n_selected, 1_000_000 // (n_lineages * n_bins)))
            group.create_array("data", data=data, chunks=(band, *data_shape[1:]))
        selected_rows = (min(n_selected, 100_000),)
        for name, values in (
            ("feature_indices", selected.astype(np.uint64)),
            ("valid_features", valid),
            ("feature_clusters", clusters.astype(np.int64)),
        ):
            group.create_array(name, data=values, chunks=selected_rows)
        group.create_array("cluster_values", data=cluster_values, chunks=(rows,))
        arguments.update(
            n_clusters=4, n_neighbors=3, nan_cluster_value=-1, n_bins=data_shape[-1]
        )

    if kind == "markers":
        statistics(None, "r_value", "p_value", "p_value_adjusted")
        names_stored = MARKER_PAYLOAD
    elif kind == "lineage_markers":
        statistics(n_lineages, "r_value", "p_value", "p_value_adjusted")
        group.create_array("effective_cells", data=np.full(n_lineages, 12.5))
        arguments["n_lineages"] = n_lineages
        names_stored = LINEAGE_MARKER_PAYLOAD
    elif kind == "drivers":
        statistics(n_lineages, *LINEAGE_DRIVER_PAYLOAD[:5])
        group.attrs["n_cells"] = 40
        arguments.update(n_lineages=n_lineages, n_cells=40)
        names_stored = LINEAGE_DRIVER_PAYLOAD
    elif kind == "aggregation":
        modules((n_selected, n_bins))
        fingerprints = ["cells", "features", "ordering"]
        group.attrs.update(
            input_fingerprints=fingerprints,
            nan_cluster_value=-1,
            effective_window=20,
            effective_bins=n_bins,
        )
        arguments.update(
            ann_params=resolve_aggregation_ann_params(None, dim=n_bins),
            expected_input_fingerprints=fingerprints,
            effective_window=20,
        )
        names_stored = AGGREGATION_PAYLOAD
    else:
        modules((n_selected, n_lineages, n_bins))
        group.create_array(
            "bin_pseudotime",
            data=np.tile(np.linspace(0.0, 1.0, n_bins), (n_lineages, 1)),
        )
        group.create_array("lineage_weight", data=np.full(n_lineages, 2.0 * n_bins))
        fingerprints = ["cells", "features", "ordering", "weights"]
        group.attrs["input_fingerprints"] = fingerprints
        arguments.update(
            n_lineages=n_lineages, expected_input_fingerprints=fingerprints
        )
        names_stored = LINEAGE_AGGREGATION_PAYLOAD
    group.attrs.update(
        {
            "artifact_id": "a" * 64,
            "kind": kind,
            "provenance": {},
            "execution_options": {},
            "created_at_ns": 1,
            "cytearc_version": "test",
            "complete": True,
            "payload_fingerprint": fingerprint_stored_arrays(group, names_stored),
        }
    )
    return group, arguments


_FEATURE_CHECKS = {
    "markers": (marker_payload_is_valid, marker_check_bytes),
    "aggregation": (aggregation_payload_is_valid, aggregation_check_bytes),
    "lineage_markers": (
        lineage_marker_payload_is_valid,
        lambda group, n_selected: lineage_statistics_check_bytes(
            group, LINEAGE_MARKER_PAYLOAD, n_selected=n_selected
        ),
    ),
    "drivers": (
        lineage_driver_payload_is_valid,
        lambda group, n_selected: lineage_statistics_check_bytes(
            group, LINEAGE_DRIVER_PAYLOAD, n_selected=n_selected
        ),
    ),
    "lineage_aggregation": (
        lineage_aggregation_payload_is_valid,
        lineage_aggregation_check_bytes,
    ),
}


_FEATURE_LOADS = {
    "markers": lambda group, n_selected: marker_load_bytes(
        group, n_selected=n_selected
    ),
    "aggregation": lambda group, n_selected: aggregation_load_bytes(
        group, n_selected=n_selected
    ),
    "lineage_markers": lambda group, n_selected: lineage_statistics_load_bytes(
        group, LINEAGE_MARKER_PAYLOAD, n_selected=n_selected, n_lineages=3
    ),
    "drivers": lambda group, n_selected: lineage_statistics_load_bytes(
        group, LINEAGE_DRIVER_PAYLOAD, n_selected=n_selected, n_lineages=3
    ),
    "lineage_aggregation": lambda group, n_selected: lineage_aggregation_load_bytes(
        group, n_selected=n_selected
    ),
}


def _traced_check(
    check: Any, group: zarr.Group, arguments: dict[str, object]
) -> tuple[bool, int]:
    """Run a payload's check, traced at its most after an untraced check, and return its verdict and the bytes it held."""
    import gc
    import tracemalloc

    from cytearc.storage.content_stamps import clear_content_cache

    def checked() -> tuple[bool, int]:
        # The first check of a process fingerprints the frozen names and IDs,
        # and a full collection empties CPython's free lists, so the check
        # allocates, and is traced at, its most.
        clear_content_cache()
        gc.collect()
        before, _ = tracemalloc.get_traced_memory()
        verdict = check(group, **arguments)
        _, peak = tracemalloc.get_traced_memory()
        return verdict, peak - before

    (traced,) = traced_reruns(checked)
    return traced


# Each payload makes one step of its check hold the most: kind, features,
# rows to a chunk of the frozen names and IDs, lineages, bins, characters of
# a name, and how far apart the selected features are.
_TRACED_PAYLOADS = [
    # Frozen names and IDs in chunks of 100,000 rows, as analyses saved
    # before their chunks followed a feature table's: fingerprinting them
    # holds the most.
    *(
        pytest.param(kind, 30_000, 100_000, 3, 20, 15, 10, id=f"{kind}-old-names")
        for kind in _FEATURE_CHECKS
    ),
    # 400 features, whose checks hold little beside their own objects.
    pytest.param("aggregation", 400, 400, 3, 20, 5, 10, id="aggregation-small"),
    pytest.param(
        "lineage_aggregation", 400, 400, 3, 20, 5, 10, id="lineage_aggregation-small"
    ),
    # Markers of each of 200,000 features, in two chunks: the check keeps a
    # chunk's three statistics and two copies of their selected features
    # while it marks which features of the next chunk are selected.
    pytest.param("markers", 200_000, 200_000, 3, 20, 1, 1, id="markers-every-feature"),
    # Lineage statistics: the check reads each array's block beside the one
    # before it and, where every feature is selected, a copy of its values.
    pytest.param(
        "lineage_markers", 100_000, 100_000, 3, 20, 1, 10, id="lineage_markers-tenth"
    ),
    pytest.param(
        "lineage_markers", 30_000, 30_000, 3, 20, 1, 1, id="lineage_markers-every"
    ),
    pytest.param("drivers", 30_000, 30_000, 3, 20, 1, 1, id="drivers-every-feature"),
    # Profiles of two bins of every feature, read beside the positions,
    # validity and clusters of their block.
    pytest.param("aggregation", 30_000, 30_000, 3, 2, 1, 1, id="aggregation-profiles"),
    pytest.param(
        "lineage_aggregation",
        30_000,
        30_000,
        2,
        2,
        1,
        1,
        id="lineage_aggregation-profiles",
    ),
    # Every feature's cluster, read beside the clusters expected of them.
    pytest.param(
        "aggregation", 100_000, 100_000, 3, 2, 1, 10, id="aggregation-clusters"
    ),
    pytest.param(
        "lineage_aggregation",
        100_000,
        100_000,
        3,
        2,
        1,
        10,
        id="lineage_aggregation-clusters",
    ),
]


@pytest.mark.parametrize(
    ("kind", "n_features", "identity_rows", "n_lineages", "n_bins", "width", "every"),
    _TRACED_PAYLOADS,
)
def test_checking_a_saved_feature_analysis_holds_what_its_estimate_states(
    tmp_path,
    kind: str,
    n_features: int,
    identity_rows: int,
    n_lineages: int,
    n_bins: int,
    width: int,
    every: int,
) -> None:
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    group, arguments = _feature_payload(
        root,
        kind,
        n_features=n_features,
        identity_rows=identity_rows,
        n_lineages=n_lineages,
        n_bins=n_bins,
        width=width,
        every=every,
    )
    check, estimate = _FEATURE_CHECKS[kind]
    verdict, held = _traced_check(check, group, arguments)
    assert verdict
    expected = estimate(group, n_selected=len(arguments["selected_features"]))
    assert held <= expected
    # Charged at the bound of their compressed bytes, the reads err high, by
    # no more than the decoded chunks they count once more.
    assert expected < 3 * held


@pytest.mark.parametrize("kind", list(_FEATURE_CHECKS))
def test_a_payload_that_lacks_an_array_is_refused_within_its_estimates(
    tmp_path, kind: str
) -> None:
    # A saved analysis whose frozen feature IDs are gone: its check refuses
    # it within what its estimates name, which read only what is stored.
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    group, arguments = _feature_payload(
        root, kind, n_features=3_000, identity_rows=3_000
    )
    del group["feature_ids"]
    check, estimate = _FEATURE_CHECKS[kind]
    n_selected = len(arguments["selected_features"])
    verdict, held = _traced_check(check, group, arguments)
    assert not verdict
    assert held <= estimate(group, n_selected=n_selected)
    # Loading the payload checks it first.
    assert _FEATURE_LOADS[kind](group, n_selected) >= estimate(
        group, n_selected=n_selected
    )
