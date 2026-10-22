"""Interrupted artifact writes keep their slot, and prune_incomplete removes residue."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from cytearc import DataStore, mount_datastore
from cytearc.storage.artifact_writer import (
    artifact_transaction,
    plan_artifact,
    prune_incomplete_artifacts,
)
from cytearc.storage.artifacts import ArtifactRef, artifact_path, inspect_artifact
from cytearc.utils.logging import logger
from cytearc.utils.shutdown import ShutdownRequest, ShutdownRequested
from tests.storage_helpers import write_count_store


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _plan(root: zarr.Group, label: str, *, assay: str | None = None):
    return plan_artifact(
        root,
        scope="datastore" if assay is None else "assay",
        assay=assay,
        kind="cell_selection" if assay is None else "feature_selection",
        operation="manual_selection",
        parameters={"label": label},
        inputs={},
        execution_options={},
        invalidate_cache=True,
    )


def _write(root: zarr.Group, label: str, *, assay: str | None = None) -> ArtifactRef:
    planned = _plan(root, label, assay=assay)
    with artifact_transaction(root, planned) as group:
        group.create_array("values", data=np.array([True, False]))
    return planned.ref


def _interrupted(root: zarr.Group, label: str, *, assay: str | None = None) -> str:
    planned = _plan(root, label, assay=assay)
    with pytest.raises(KeyboardInterrupt):
        with artifact_transaction(root, planned) as group:
            group.create_array("values", data=np.array([True, False]))
            raise KeyboardInterrupt
    return artifact_path(planned.ref)


@pytest.mark.parametrize(
    "interruption",
    [
        KeyboardInterrupt(),
        asyncio.CancelledError(),
        SystemExit(1),
        ShutdownRequested(ShutdownRequest(requested_at_ns=1, reason="SIGTERM")),
    ],
    ids=["keyboard", "cancelled", "exit", "shutdown"],
)
def test_an_interrupted_write_keeps_its_slot_marked_incomplete(interruption):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _plan(root, "a")

    with _warnings() as warnings, pytest.raises(type(interruption)):
        with artifact_transaction(root, planned) as group:
            group.create_array("values", data=np.array([True, False]))
            raise interruption

    status = inspect_artifact(root, planned.ref)
    assert status.exists and not status.complete
    path = artifact_path(planned.ref)
    assert warnings == [
        f"An interruption stopped the write of artifact {path}, which Zarr may "
        "still be running, so the artifact was kept, marked incomplete; nothing "
        "reuses it. Once no process is writing to this store, remove it with "
        "ds.artifacts.prune_incomplete(dry_run=False)."
    ]
    # A late chunk of the interrupted write lands in an intact array.
    root[f"{path}/values"][:] = np.array([False, True])
    # The kept slot is never reused.
    assert not _plan(root, "a").reused


def test_an_interruption_while_the_slot_is_created_is_reported(monkeypatch):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _plan(root, "a")
    original = zarr.Group.create_group

    def interrupted(group, *args, **kwargs):
        original(group, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(zarr.Group, "create_group", interrupted)
    with _warnings() as warnings, pytest.raises(KeyboardInterrupt):
        with artifact_transaction(root, planned):
            raise AssertionError("the body must not run")

    assert not inspect_artifact(root, planned.ref).complete
    assert len(warnings) == 1 and "kept, marked incomplete" in warnings[0]


def test_a_slot_that_failed_to_start_is_never_deleted():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _plan(root, "a")
    existing = root.create_group(artifact_path(planned.ref))
    existing.attrs["note"] = "not this call's group"

    with pytest.raises(FileExistsError, match="Artifact path already exists"):
        with artifact_transaction(root, planned):
            raise AssertionError("the body must not run")

    assert root[artifact_path(planned.ref)].attrs["note"] == "not this call's group"


def _put(root: zarr.Group, key: str, payload: bytes) -> None:
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(root.store.set(key, buffer))


def test_prune_incomplete_finds_and_removes_only_the_residue_of_writes():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    complete = _write(root, "complete")
    incomplete = _interrupted(root, "incomplete")
    assay_incomplete = _interrupted(root, "assay", assay="RNA")
    torn = _write(root, "torn")
    _put(root, f"{artifact_path(torn)}/zarr.json", b'{"zarr_format": 3, ')
    malformed = _write(root, "malformed")
    root[artifact_path(malformed)].attrs["complete"] = "yes"
    # Chunks of a write that landed after its group was removed.
    orphan = f"artifacts/cell_selection/{'e' * 64}"
    _put(root, f"{orphan}/values/c/0", b"chunk")
    # Things CyteArc did not write.
    _put(root, "artifacts/cell_selection/backup/values/c/0", b"chunk")
    _put(root, f"artifacts/cell_selection/{'d' * 64}", b"a plain object")
    root.create_array(f"artifacts/cell_selection/{'c' * 64}", data=np.zeros(2))
    root.create_group(f"artifacts/regulon_set/{'b' * 64}").attrs["complete"] = False

    expected = sorted([incomplete, orphan, assay_incomplete])
    assert prune_incomplete_artifacts(root, ["RNA"], dry_run=True) == expected
    assert inspect_artifact(
        root,
        ArtifactRef.from_dict(
            {
                "type": "artifact",
                "scope": "datastore",
                "kind": "cell_selection",
                "artifact_id": incomplete.rsplit("/", 1)[-1],
            }
        ),
    ).exists

    assert prune_incomplete_artifacts(root, ["RNA"], dry_run=False) == expected
    assert prune_incomplete_artifacts(root, ["RNA"], dry_run=True) == []
    for path in expected:
        assert not sync(root.store.exists(f"{path}/zarr.json"))
    assert not sync(root.store.exists(f"{orphan}/values/c/0"))
    assert inspect_artifact(root, complete).complete
    assert sync(root.store.exists("artifacts/cell_selection/backup/values/c/0"))
    assert sync(root.store.exists(f"artifacts/cell_selection/{'d' * 64}"))
    assert f"artifacts/cell_selection/{'c' * 64}" in root
    assert f"artifacts/regulon_set/{'b' * 64}" in root
    assert sync(root.store.exists(f"{artifact_path(torn)}/zarr.json"))
    assert root[artifact_path(malformed)].attrs["complete"] == "yes"


def test_prune_incomplete_of_a_store_without_artifacts_finds_nothing():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assert prune_incomplete_artifacts(root, ["RNA"], dry_run=False) == []


def test_prune_incomplete_refuses_to_remove_from_a_read_only_store():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    incomplete = _interrupted(root, "incomplete")
    reader = zarr.open_group(store=store.with_read_only(True), mode="r")

    assert prune_incomplete_artifacts(reader, [], dry_run=True) == [incomplete]
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        prune_incomplete_artifacts(reader, [], dry_run=False)


_COUNTS = np.array([[1, 0, 2], [0, 3, 1], [4, 0, 0], [1, 1, 1]], dtype=np.uint32)


def _store(path: Path, **options) -> DataStore:
    return DataStore(str(path), default_assay="RNA", nthreads=1, **options)


def test_datastore_prune_incomplete_lists_by_default_and_removes_on_request(
    tmp_path,
):
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": _COUNTS}, "uint32")
    ds = _store(path)
    kept = ds.snapshot_cell_selection("I")
    interrupted = _interrupted(ds.zw, "interrupted", assay="RNA")

    assert ds.artifacts.prune_incomplete() == [interrupted]
    reader = _store(path, zarr_mode="r")
    assert reader.artifacts.prune_incomplete(dry_run=True) == [interrupted]
    with pytest.raises(PermissionError, match="artifacts.prune_incomplete requires"):
        reader.artifacts.prune_incomplete(dry_run=False)
    with pytest.raises(TypeError, match="dry_run must be a bool"):
        ds.artifacts.prune_incomplete(dry_run=0)  # type: ignore[arg-type]

    assert ds.artifacts.prune_incomplete(dry_run=False) == [interrupted]
    assert ds.artifacts.prune_incomplete() == []
    assert ds.artifacts.inspect(kept).complete


def test_a_mount_prunes_its_own_artifacts_and_never_its_source(tmp_path):
    source_path = tmp_path / "source.zarr"
    write_count_store(str(source_path), {"RNA": _COUNTS}, "uint32")
    source = _store(source_path)
    in_source = _interrupted(source.zw, "source")
    mounted = mount_datastore(
        str(source_path), at=str(tmp_path / "target.zarr"), default_assay="RNA"
    )
    in_target = _interrupted(mounted.zw, "target")

    # The source's incomplete artifact is read through the mount but belongs to
    # the source, which the mount never writes.
    assert in_source in [
        artifact_path(ref)
        for ref in mounted.artifacts.list(scope="datastore", kind="cell_selection")
    ]
    assert mounted.artifacts.prune_incomplete(dry_run=False) == [in_target]
    assert sync(source.zw.store.exists(f"{in_source}/zarr.json"))
    assert source.artifacts.prune_incomplete() == [in_source]
