"""Lineage reports trace chains of any depth and mark corrupt records."""

import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_path,
    make_provenance,
)
from cytearc.storage.lineage import ArtifactLineage


def _ref(kind: str, index: int) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore", kind=kind, artifact_id=f"{index:064x}", assay=None
    )


def _record(root: zarr.Group, ref: ArtifactRef, inputs: dict, **attributes) -> None:
    root.create_group(
        artifact_path(ref),
        attributes={
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": make_provenance(
                operation="manual_selection", parameters={}, inputs=inputs
            ),
            "execution_options": {},
            "complete": True,
            **attributes,
        },
    )


def test_a_chain_deeper_than_the_recursion_limit_is_traced():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    depth = 1_200
    refs = [_ref("cell_selection", index + 1) for index in range(depth)]
    _record(root, refs[0], {})
    for previous, ref in zip(refs, refs[1:]):
        _record(root, ref, {"cell_selection": previous})

    lineage = ArtifactLineage.from_store(root, refs[-1])

    assert lineage.graph.number_of_nodes() == depth
    assert lineage.graph.number_of_edges() == depth - 1
    assert all(
        lineage.graph.edges[previous, ref]["inputs"] == ("cell_selection",)
        for previous, ref in zip(refs, refs[1:])
    )
    assert lineage._ordered_refs() == refs
    assert lineage.to_markdown().count("- Status: `complete`") == depth


def test_a_corrupt_record_is_marked_and_the_rest_of_the_report_kept():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    source = _ref("cell_selection", 1)
    middle = _ref("cell_selection", 2)
    other = _ref("cell_selection", 3)
    output = _ref("cluster_labels", 4)
    _record(root, source, {})
    # The record names another artifact, as a copy into the wrong group does.
    _record(root, middle, {"cell_selection": source}, artifact_id="f" * 64)
    _record(root, other, {})
    _record(root, output, {"first": middle, "second": other})

    lineage = ArtifactLineage.from_store(root, output)

    assert set(lineage.graph) == {middle, other, output}
    node = lineage.graph.nodes[middle]
    assert node["status"] is None
    assert node["error"] == (
        f"ValueError: Artifact at {artifact_path(middle)} has mismatched artifact_id"
    )
    assert lineage.graph.nodes[other]["error"] is None
    assert lineage.graph.nodes[output]["status"].complete
    markdown = lineage.to_markdown()
    assert markdown.count("- Status: `corrupt`") == 1
    assert markdown.count("- Status: `complete`") == 2
    assert (
        f"- Error: `ValueError: Artifact at {artifact_path(middle)} has mismatched "
        "artifact_id`" in markdown
    )
    mermaid = lineage.to_mermaid()
    assert "record unreadable" in mermaid and "status: corrupt" in mermaid


def test_a_corrupt_external_record_is_marked_while_a_wrong_root_raises():
    query_root = zarr.open_group(store=MemoryStore(), mode="w")
    external_root = zarr.open_group(store=MemoryStore(), mode="w")
    external_root.create_group("RNA").attrs["dataset_fingerprint"] = "reference"
    referenced = ArtifactRef(
        scope="assay", assay="RNA", kind="mapping_reference", artifact_id="a" * 64
    )
    _record(external_root, referenced, {}, complete="yes")
    external = ExternalArtifactRef("reference", referenced)
    projection = _ref("projection", 9)
    _record(query_root, projection, {"mapping_reference": external})

    lineage = ArtifactLineage.from_store(
        query_root, projection, external_roots={"reference": external_root}
    )

    assert lineage.graph.nodes[external]["error"] == (
        f"TypeError: Artifact complete attr at {artifact_path(referenced)} must be "
        "boolean"
    )
    assert "- Status: `corrupt`" in lineage.to_markdown()
    external_root["RNA"].attrs["dataset_fingerprint"] = "another"
    with pytest.raises(ValueError, match="Expected 'reference', received 'another'"):
        ArtifactLineage.from_store(
            query_root, projection, external_roots={"reference": external_root}
        )
