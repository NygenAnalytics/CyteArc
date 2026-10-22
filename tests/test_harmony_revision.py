"""Operation revision 2 of run_harmony marks corrections of the earlier Harmony."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.embeddings.harmony import validate_harmony_parameters
from cytearc.storage.artifact_writer import artifact_transaction, plan_artifact
from cytearc.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    inspect_artifact,
    make_provenance,
)
from cytearc.storage.lineage import ArtifactLineage
from cytearc.storage.operation_revisions import (
    OPERATION_REVISIONS,
    _HARMONY2,
    effective_revision,
)

_INPUTS = {
    "reduction": ArtifactRef(
        scope="assay", assay="RNA", kind="reduction", artifact_id="a" * 64
    ),
    "batch_snapshot": ArtifactRef(
        scope="datastore", kind="metadata_snapshot", artifact_id="b" * 64
    ),
}


def _parameters(options: dict) -> dict:
    return {"batch_columns": ["batch"], "harmony_parameters": options}


def _plan(root: zarr.Group, options: dict):
    return plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="batch_correction",
        operation="run_harmony",
        parameters=_parameters(options),
        inputs=_INPUTS,
        execution_options={},
    )


def _write_record(root: zarr.Group, options: dict, token: str) -> ArtifactRef:
    """Write a correction as a release before this entry recorded it."""
    ref = ArtifactRef(
        scope="assay", assay="RNA", kind="batch_correction", artifact_id=token * 64
    )
    root.create_group(
        artifact_path(ref),
        attributes={
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": make_provenance(
                operation="run_harmony", parameters=_parameters(options), inputs=_INPUTS
            ),
            "execution_options": {},
            "complete": True,
        },
    )
    return ref


def test_the_revision_applies_to_the_options_only_the_earlier_harmony_took():
    assert OPERATION_REVISIONS["run_harmony"] == (_HARMONY2,)
    assert (_HARMONY2.revision, _HARMONY2.release) == (2, "1.3.0")
    for options, revision in (
        ({"nclust": 30}, 2),
        ({"max_iter_kmeans": 20}, 2),
        ({"lamb": [1.0, 0.5]}, 2),
        ({"cluster_fn": {"external_hook": True, "module": "m", "qualname": "f"}}, 2),
        ({"nclust": 30, "theta": 2.0}, 2),
        ({}, 1),
        ({"n_clusters": 5, "ridge_penalty": 1.0, "max_iter_cluster": 20}, 1),
        ({"random_state": 3, "theta": [2.0], "sigma": 0.1}, 1),
    ):
        assert (
            effective_revision(
                "run_harmony", "batch_correction", _parameters(options), {}
            )
            == revision
        )
    assert effective_revision("run_harmony", "batch_correction", {}, {}) == 1
    assert (
        effective_revision(
            "run_harmony", "batch_correction", {"harmony_parameters": ["nclust"]}, {}
        )
        == 1
    )
    # No request records those options: Harmony2 refuses them.
    for option in ("nclust", "max_iter_kmeans", "lamb", "cluster_fn"):
        with pytest.raises(
            ValueError, match=f"Unsupported Harmony parameter '{option}'"
        ):
            validate_harmony_parameters({option: 1})


def test_a_correction_of_the_earlier_harmony_is_reported_stale():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    earlier = _write_record(root, {"nclust": 30}, "c")

    status = inspect_artifact(root, earlier)
    assert status.revision == 1 and status.current_revision == 2
    assert not status.is_current
    assert status.superseded_by == (_HARMONY2,)
    markdown = ArtifactLineage.from_store(root, earlier).to_markdown()
    assert "Status: `stale`" in markdown
    assert "Revision: `1, current 2: Harmony2 replaced the earlier Harmony" in markdown


def test_corrections_recorded_by_harmony2_keep_their_identity_and_are_reused():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    # A 1.2 correction with default options records no revision.
    released = _write_record(root, {}, "d")
    assert inspect_artifact(root, released).is_current

    planned = _plan(root, {})
    assert "revision" not in planned.provenance
    assert planned.reused and planned.ref == released

    fresh = _plan(root, {"n_clusters": 4})
    assert "revision" not in fresh.provenance and not fresh.reused
    with artifact_transaction(root, fresh) as group:
        group.create_array("data", data=np.zeros((2, 2), dtype=np.float32))
    status = inspect_artifact(root, fresh.ref)
    assert status.revision == 1 and status.is_current
    assert _plan(root, {"n_clusters": 4}).ref == fresh.ref
