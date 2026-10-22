"""One damaged or unknown artifact never hides the rest of a listing."""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from cytearc.storage.artifact_writer import artifact_transaction, plan_artifact
from cytearc.storage.artifacts import (
    ArtifactRef,
    _child_groups,
    artifact_path,
    list_artifacts,
)
from cytearc.utils.logging import logger


def _write(root: zarr.Group, kind: str, label: str) -> ArtifactRef:
    planned = plan_artifact(
        root,
        scope="datastore",
        kind=kind,
        operation="manual_selection",
        parameters={"label": label},
        inputs={},
        execution_options={},
    )
    with artifact_transaction(root, planned) as group:
        group.create_array("values", data=np.array([True, False]))
    return planned.ref


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


def _tear(root: zarr.Group, ref: ArtifactRef) -> None:
    document = default_buffer_prototype().buffer.from_bytes(b'{"zarr_format": 3, ')
    sync(root.store.set(f"{artifact_path(ref)}/zarr.json", document))


def test_a_torn_artifact_record_is_skipped_with_a_warning():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    healthy = _write(root, "cell_selection", "a")
    torn = _write(root, "cell_selection", "b")
    _tear(root, torn)

    with _warnings() as warnings:
        listed = list_artifacts(root, scope="datastore", kind="cell_selection")
        everything = list_artifacts(root, scope="datastore", complete_only=True)

    assert listed == [healthy]
    assert everything == [healthy]
    assert len(warnings) == 2
    assert warnings[0].startswith(
        "Skipped 1 unreadable cell_selection artifacts below artifacts/cell_selection: "
        f"{torn.artifact_id} (JSONDecodeError: "
    )


def test_planning_reuses_a_healthy_artifact_beside_a_torn_one():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    healthy = _write(root, "cell_selection", "a")
    _tear(root, _write(root, "cell_selection", "b"))

    planned = plan_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        operation="manual_selection",
        parameters={"label": "a"},
        inputs={},
        execution_options={},
    )

    assert planned.reused and planned.ref == healthy


def test_an_unknown_kind_is_skipped_and_the_known_kinds_listed():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    known = _write(root, "cell_selection", "a")
    # A kind written by a newer release, or a stray group below artifacts/.
    root.create_group("artifacts/regulon_set/" + "c" * 64)

    with _warnings() as warnings:
        listed = list_artifacts(root, scope="datastore")

    assert listed == [known]
    assert warnings == [
        "Skipped artifact kinds that this CyteArc release does not know below "
        "artifacts: regulon_set"
    ]
    with pytest.raises(ValueError, match="Unknown artifact kind: 'regulon_set'"):
        list_artifacts(root, scope="datastore", kind="regulon_set")


def test_child_groups_fall_back_to_one_read_per_child(monkeypatch):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    first = _write(root, "cell_selection", "a")
    second = _write(root, "cell_selection", "b")
    kind = root["artifacts/cell_selection"]
    kind.create_array("stray", data=np.zeros(2))
    calls = []
    original = zarr.Group.groups

    def flaky(self):
        calls.append(self.path)
        if len(calls) == 1:
            raise ValueError("listing failed once")
        return original(self)

    monkeypatch.setattr(zarr.Group, "groups", flaky)
    with _warnings() as warnings:
        children = _child_groups(kind, "cell_selection artifacts")

    # Every child opened on its own; an array is not an artifact group.
    assert [name for name, _ in children] == sorted(
        [first.artifact_id, second.artifact_id]
    )
    assert warnings == []
