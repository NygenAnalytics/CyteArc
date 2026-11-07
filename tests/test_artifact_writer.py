from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.storage import operation_revisions
from cytearc.storage.artifact_writer import (
    ArrayRequirement,
    ArtifactComputation,
    ArtifactComputationRequired,
    AttributeRequirement,
    artifact_forecast_scope,
    artifact_plan_scope,
    artifact_transaction,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from cytearc.storage.artifacts import ArtifactRef, artifact_path, inspect_artifact
from cytearc.storage.diagnostics import DIAGNOSTICS_ATTRIBUTE, declare_diagnostics
from cytearc.utils.logging import logger

# An operation that only the forecast tests plan; it saves one diagnostic.
_FORECAST_OPERATION = "forecast_test_counts"
declare_diagnostics(
    _FORECAST_OPERATION, ("forecastCount",), call="ds.tests.forecast_counts"
)


def test_artifact_writer_streams_to_random_path_then_reuses_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import cytearc
    import cytearc.storage.artifact_writer as artifact_writer

    # Each artifact the writer starts is stamped one tick later.
    ticks = iter(range(1_000, 10_000, 1_000))
    monkeypatch.setattr(
        artifact_writer, "time", SimpleNamespace(time_ns=lambda: next(ticks))
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": "run_normalization",
        "parameters": {"log_transform": False},
        "inputs": {"selection": {"artifact_id": "a" * 64}},
        "execution_options": {"batch_size": 100},
    }
    planned = plan_artifact(root, **arguments)
    assert not planned.reused
    group = start_artifact(root, planned)
    assert not inspect_artifact(root, planned.ref).complete
    group.create_array("data", data=np.array([1.0, 2.0, 3.0]))
    finish_artifact(group, planned)
    status = inspect_artifact(root, planned.ref)
    assert status.complete
    assert status.created_at_ns == 1_000
    assert status.cytearc_version == cytearc.__version__
    assert status.execution_options == {"batch_size": 100}
    assert status.provenance == {
        "operation": "run_normalization",
        "parameters": {"log_transform": False},
        "inputs": {"selection": {"artifact_id": "a" * 64}},
    }
    assert group.path == artifact_path(planned.ref)
    completed_attrs = dict(group.attrs)

    reused = plan_artifact(root, **arguments)
    assert reused.reused
    assert reused.ref == planned.ref
    assert reused_artifact_group(root, reused).path == group.path
    assert dict(group.attrs) == completed_attrs

    invalidated = plan_artifact(
        root,
        **arguments,
        invalidate_cache=True,
    )
    assert not invalidated.reused
    assert invalidated.ref != planned.ref
    assert artifact_path(invalidated.ref) not in root
    refreshed_group = start_artifact(root, invalidated)
    refreshed_group.create_array("data", data=np.array([4.0, 5.0, 6.0]))
    finish_artifact(refreshed_group, invalidated)
    assert inspect_artifact(root, invalidated.ref).created_at_ns == 2_000
    # Both records match; the newer one is preferred.
    preferred = plan_artifact(root, **arguments)
    assert preferred.reused
    assert preferred.ref == invalidated.ref


def test_artifact_plan_scope_records_nested_created_and_reused_decisions() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": "run_normalization",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }
    with artifact_plan_scope() as outer:
        with artifact_plan_scope() as inner:
            created = plan_artifact(root, **arguments)
        group = start_artifact(root, created)
        finish_artifact(group, created)
        reused = plan_artifact(root, **arguments)

    assert [(item.operation, item.ref, item.disposition) for item in inner] == [
        ("run_normalization", created.ref, "created")
    ]
    assert [(item.operation, item.ref, item.disposition) for item in outer] == [
        ("run_normalization", created.ref, "created"),
        ("run_normalization", reused.ref, "reused"),
    ]


def test_incomplete_artifact_is_not_reused() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "ann_index",
        "operation": "build_ann_index",
        "parameters": {"ann_parallel": True},
        "inputs": {"coordinates": {"artifact_id": "b" * 64}},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    start_artifact(root, first)

    second = plan_artifact(root, **arguments)
    assert not second.reused
    assert second.ref != first.ref


def test_execution_options_use_artifact_value_serialization() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
        execution_options={
            "nan": float("nan"),
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
            "bytes": b"\x00\xff",
            "set": {2, 1},
            "numpy_scalar": np.int64(3),
        },
    )

    start_artifact(root, planned)
    status = inspect_artifact(root, planned.ref)

    assert status.execution_options == {
        "nan": {"special_float": "nan"},
        "positive_infinity": {"special_float": "inf"},
        "negative_infinity": {"special_float": "-inf"},
        "bytes": {"bytes_hex": "00ff"},
        "set": [1, 2],
        "numpy_scalar": 3,
    }


def test_missing_required_attribute_prevents_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "mapping_reference",
        "operation": "build_mapping_reference",
        "parameters": {"method": "symphony"},
        "inputs": {"reduction": {"artifact_id": "c" * 64}},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    group = start_artifact(root, first)
    group.create_array("data", data=np.array([1.0]))
    finish_artifact(group, first)

    requirements = {
        "required_arrays": ("data",),
        "required_attributes": ("reference_metadata",),
    }
    second = plan_artifact(root, **arguments, **requirements)

    assert not second.reused
    assert second.ref != first.ref
    # The same record with the attribute present is reused.
    group.attrs["reference_metadata"] = {"method": "symphony"}
    third = plan_artifact(root, **arguments, **requirements)
    assert third.reused
    assert third.ref == first.ref


def test_invalid_required_attribute_type_prevents_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "mapping_reference",
        "operation": "build_mapping_reference",
        "parameters": {"method": "symphony"},
        "inputs": {"reduction": {"artifact_id": "d" * 64}},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    group = start_artifact(root, first)
    group.attrs["reference_metadata"] = "invalid"
    finish_artifact(group, first)

    requirement = AttributeRequirement("reference_metadata", expected_types=(dict,))
    second = plan_artifact(root, **arguments, required_attributes=(requirement,))

    assert not second.reused
    assert second.ref != first.ref
    group.attrs["reference_metadata"] = {"method": "symphony"}
    third = plan_artifact(root, **arguments, required_attributes=(requirement,))
    assert third.reused
    assert third.ref == first.ref


def test_finish_rejects_payload_that_violates_declared_shape() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
        execution_options={},
        required_arrays=(ArrayRequirement("data", shape=(3,), dtype_kind="f"),),
    )
    group = start_artifact(root, planned)
    group.create_array("data", data=np.array([1.0, 2.0]))

    with pytest.raises(
        ValueError, match="Artifact array 'data' does not satisfy its contract"
    ):
        finish_artifact(group, planned)

    assert not inspect_artifact(root, planned.ref).complete


def test_exact_dtype_requirement_prevents_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    # An operation that saves no diagnostics, so reuse turns on the array alone.
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "reduction",
        "operation": "test_reduction",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    group = start_artifact(root, first)
    group.create_array("data", data=np.array([1.0], dtype=np.float64))
    finish_artifact(group, first)

    same_kind = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", dtype_kind="f"),),
    )
    exact = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", dtype=np.float32),),
    )

    assert same_kind.reused
    assert not exact.reused


def _planned_normalized(root: zarr.Group, **extra: object):
    return plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={},
        execution_options={},
        required_arrays=(ArrayRequirement("data", shape=(3,)),),
        **extra,
    )


def test_artifact_transaction_finishes_the_written_slot() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    with artifact_transaction(root, planned) as group:
        assert not inspect_artifact(root, planned.ref).complete
        group.create_array("data", data=np.arange(3.0))
    assert inspect_artifact(root, planned.ref).complete
    assert _planned_normalized(root).ref == planned.ref


@pytest.mark.parametrize("error", [RuntimeError("write failed"), OSError("lost")])
def test_artifact_transaction_removes_a_failed_slot(error: BaseException) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    with pytest.raises(type(error)):
        with artifact_transaction(root, planned) as group:
            group.create_array("data", data=np.arange(3.0))
            raise error
    assert artifact_path(planned.ref) not in root
    assert not inspect_artifact(root, planned.ref).exists


def test_artifact_transaction_removes_a_slot_that_fails_its_contract() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    with pytest.raises(ValueError, match="does not satisfy its contract"):
        with artifact_transaction(root, planned) as group:
            group.create_array("data", data=np.arange(4.0))
    assert not inspect_artifact(root, planned.ref).exists


def test_start_artifact_refuses_a_read_only_root_before_writing() -> None:
    store = MemoryStore()
    zarr.open_group(store=store, mode="w")
    read_only = zarr.open_group(store=store.with_read_only(True), mode="r")
    planned = _planned_normalized(read_only)
    with pytest.raises(PermissionError, match=r"run_normalization.*zarr_mode='r\+'"):
        start_artifact(read_only, planned)
    with pytest.raises(PermissionError):
        with artifact_transaction(read_only, planned):
            raise AssertionError("the body must not run")
    assert not inspect_artifact(read_only, planned.ref).exists


def test_artifact_transaction_never_deletes_once_publication_is_issued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.storage.artifacts import artifact_group

    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    original = zarr.Group.update_attributes
    in_flight = []

    def interrupted(group: zarr.Group, attributes: dict) -> zarr.Group:
        if attributes.get("complete") is True:
            # Ctrl-C stops the caller while Zarr's I/O thread still runs the write.
            in_flight.append(lambda: original(group, attributes))
            raise KeyboardInterrupt
        return original(group, attributes)

    with monkeypatch.context() as patch:
        patch.setattr(zarr.Group, "update_attributes", interrupted)
        with pytest.raises(KeyboardInterrupt):
            with artifact_transaction(root, planned) as group:
                group.create_array("data", data=np.arange(3.0))

    status = inspect_artifact(root, planned.ref)
    assert status.exists and not status.complete
    # The late write lands on the intact slot, never on a deleted one.
    (land,) = in_flight
    land()
    assert inspect_artifact(root, planned.ref).complete
    np.testing.assert_array_equal(
        artifact_group(root, planned.ref)["data"][:], np.arange(3.0)
    )


def _forecast_arguments(**changes: Any) -> dict[str, Any]:
    """The planning arguments of the forecast tests' artifact, with ``changes``."""
    return {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": _FORECAST_OPERATION,
        "parameters": {"log_transform": True, "scale": 2},
        "inputs": {"cells": {"artifact_id": "a" * 64}},
        "execution_options": {},
        "required_arrays": (ArrayRequirement("data", shape=(3,)),),
        **changes,
    }


def _stored(
    root: zarr.Group,
    *,
    diagnostics: dict[str, Any] | None = None,
    complete: bool = True,
    **changes: Any,
) -> ArtifactRef:
    """Store the forecast tests' artifact as a run would, and return its ref."""
    planned = plan_artifact(root, **_forecast_arguments(**changes))
    group = start_artifact(root, planned)
    group.create_array("data", data=np.arange(3.0))
    if diagnostics is not None:
        group.attrs[DIAGNOSTICS_ATTRIBUTE] = diagnostics
    if complete:
        finish_artifact(group, planned)
    return planned.ref


def _forecast(root: zarr.Group, **changes: Any) -> ArtifactComputation:
    """Plan the artifact in a forecast and return why it would be computed."""
    with (
        artifact_forecast_scope(),
        pytest.raises(ArtifactComputationRequired) as raised,
    ):
        plan_artifact(root, **_forecast_arguments(**changes))
    return raised.value.computation


def _keys(store: MemoryStore) -> dict[str, bytes]:
    return {key: bytes(value.to_bytes()) for key, value in store._store_dict.items()}


def test_a_forecast_reuses_as_planning_does_and_writes_nothing() -> None:
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    ref = _stored(root, diagnostics={"forecastCount": 3})
    before = _keys(store)

    with artifact_plan_scope() as receipts, artifact_forecast_scope():
        reused = plan_artifact(root, **_forecast_arguments())

    assert reused.reused and reused.ref == ref
    assert [(item.ref, item.disposition) for item in receipts] == [(ref, "reused")]
    # A request that would compute raises before it plans a new artifact.
    assert _forecast(root, parameters={"log_transform": False, "scale": 2}).reason == (
        "changed"
    )
    assert _keys(store) == before
    # Outside the scope, planning plans a new artifact as before.
    planned = plan_artifact(
        root, **_forecast_arguments(parameters={"log_transform": False, "scale": 2})
    )
    assert not planned.reused and planned.ref != ref


def test_a_forecast_of_an_empty_store_finds_no_artifact_of_the_operation() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")

    computation = _forecast(root)

    assert computation == ArtifactComputation(
        operation=_FORECAST_OPERATION, kind="normalized", reason="absent"
    )
    assert computation.describe() == (
        "The store holds no forecast_test_counts artifact"
    )


def test_a_forecast_names_the_parameters_and_inputs_that_differ() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    far = _stored(
        root,
        diagnostics={"forecastCount": 1},
        parameters={"log_transform": False, "scale": 5},
        inputs={"cells": {"artifact_id": "b" * 64}},
    )
    near = _stored(
        root,
        diagnostics={"forecastCount": 1},
        parameters={"log_transform": True, "scale": 3},
    )

    computation = _forecast(root)

    # The artifact of the same inputs differs in one parameter, the other in
    # two parameters and an input.
    assert computation.reason == "changed"
    assert computation.existing == near
    assert computation.changed_parameters == ("scale",)
    assert computation.changed_inputs == ()
    assert computation.describe() == (
        "The closest forecast_test_counts artifact differs in parameters scale"
    )
    moved = _forecast(root, parameters={"log_transform": False, "scale": 5})
    assert (moved.existing, moved.changed_parameters, moved.changed_inputs) == (
        far,
        (),
        ("cells",),
    )
    assert moved.describe().endswith("differs in inputs cells")
    # A parameter that one side lacks differs too.
    added = _forecast(root, parameters={"log_transform": True, "scale": 3, "new": 1})
    assert (added.existing, added.changed_parameters) == (near, ("new",))
    # Among artifacts that differ as little, the newest is the closest.
    newest = _stored(
        root,
        diagnostics={"forecastCount": 1},
        parameters={"log_transform": True, "scale": 4},
    )
    assert _forecast(root).existing == newest
    both = _forecast(
        root,
        parameters={"log_transform": True, "scale": 9},
        inputs={"cells": {"artifact_id": "c" * 64}},
    )
    assert (both.existing, both.changed_parameters, both.changed_inputs) == (
        newest,
        ("scale",),
        ("cells",),
    )
    assert both.describe().endswith("differs in parameters scale and inputs cells")


def test_a_forecast_names_the_revision_that_supersedes_the_stored_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = _stored(root, diagnostics={"forecastCount": 1})
    revision = operation_revisions.OperationRevision(
        revision=2,
        release="9.9.9",
        change="Forecast counts are scaled twice",
        applies=None,
    )
    monkeypatch.setattr(
        operation_revisions,
        "OPERATION_REVISIONS",
        MappingProxyType({_FORECAST_OPERATION: (revision,)}),
    )

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        computation = _forecast(root)
        # The forecast recomputes nothing, so only planning logs a recomputation.
        assert not [message for message in messages if "Recomputing" in message]
        plan_artifact(root, **_forecast_arguments())
    finally:
        logger.remove(sink)

    (line,) = [message for message in messages if "Recomputing" in message]
    assert line.startswith(
        f"Recomputing forecast_test_counts: artifact {stored.artifact_id[:12]}"
    )
    assert computation.reason == "superseded"
    assert computation.existing == stored
    assert computation.revision_changes == ("Forecast counts are scaled twice",)
    assert computation.describe() == (
        "A revision supersedes the stored forecast_test_counts artifact: "
        "Forecast counts are scaled twice"
    )


def test_a_forecast_tells_an_interrupted_artifact_apart() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    interrupted = _stored(root, diagnostics={"forecastCount": 1}, complete=False)

    computation = _forecast(root)

    assert (computation.reason, computation.existing) == ("incomplete", interrupted)
    assert computation.describe() == (
        "The stored forecast_test_counts artifact is incomplete"
    )


def test_a_forecast_names_an_artifact_without_the_diagnostics_of_its_operation() -> (
    None
):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = _stored(root)

    computation = _forecast(root)

    assert (computation.reason, computation.existing) == (
        "missing_diagnostics",
        stored,
    )
    assert computation.describe() == (
        "The stored forecast_test_counts artifact lacks the diagnostics that "
        "forecast_test_counts saves"
    )


def test_a_forecast_names_an_artifact_that_fails_its_payload_checks() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = _stored(root, diagnostics={"forecastCount": 1})

    shape = _forecast(root, required_arrays=(ArrayRequirement("data", shape=(4,)),))
    rejected = _forecast(root, reuse_validator=lambda _ref, _group: False)
    attribute = _forecast(root, required_attributes=("missing",))

    for computation in (shape, rejected, attribute):
        assert (computation.reason, computation.existing) == ("invalid", stored)
    assert shape.describe() == (
        "The stored forecast_test_counts artifact fails its payload checks"
    )
    recompute = _forecast(root, invalidate_cache=True)
    assert (recompute.reason, recompute.existing) == ("invalidate_cache", None)
    assert recompute.describe() == (
        "The call asks to compute the forecast_test_counts artifact again"
    )
