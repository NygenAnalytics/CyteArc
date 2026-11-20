import sys
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec

import cytearc.storage.pipeline_runs as pipeline_run_storage
from cytearc.storage.ann_index import (
    ANN_INDEX_CHUNK_BYTES,
    ANN_INDEX_FORMAT_VERSION,
    _ANN_INDEX_METADATA,
)
from tests.storage_helpers import finalize_test_counts
from cytearc.storage.layout import normalize_chunks
from cytearc.storage.pipeline_runs import (
    PipelineOutputRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_run_record,
    start_pipeline_stage_record,
)
from cytearc.storage.refs import ArtifactRef, artifact_path
from cytearc.storage.types import array_metadata_shards
from cytearc.tools import repack_zarr as repack_module
from cytearc.tools.repack_zarr import repack_store


def _add_raw_metadata(root):
    from cytearc.storage.schema import create_cell_data
    from cytearc.storage.arrays import create_metadata_column

    for name, workspace in repack_module._count_assays(root):
        namespace = root if workspace is None else root[workspace]
        assay = namespace[name]
        matrix = assay if workspace is None else root[f"matrices/{name}"]
        rows, features = matrix["counts"].shape
        if "cellData" not in namespace:
            ids = np.array([f"c{i + 1}" for i in range(rows)])
            create_cell_data(root, workspace, ids, ids)
        else:
            cells = namespace["cellData"]
            if "names" not in cells:
                create_metadata_column(cells, "names", data=cells["ids"][:], dtype=str)
            if "I" not in cells:
                create_metadata_column(
                    cells, "I", data=np.ones(rows, dtype=bool), dtype=bool
                )
        if "featureData" not in assay:
            table = assay.create_group("featureData")
            ids = np.array([f"f{i}" for i in range(features)])
            for column in ("ids", "names"):
                create_metadata_column(table, column, data=ids, dtype=str)
            create_metadata_column(
                table, "I", data=np.ones(features, dtype=bool), dtype=bool
            )


def _prepare_source(root):
    from cytearc.assay.classification import preset_assay_types
    from cytearc.metadata import MetaData
    from cytearc.storage.count_matrix import create_product_counts_array
    from cytearc.writers.counts_t import finalize_writer_counts_t

    _add_raw_metadata(root)
    for name, workspace in repack_module._count_assays(root):
        namespace = root if workspace is None else root[workspace]
        matrix = namespace[name] if workspace is None else root[f"matrices/{name}"]
        values = matrix["counts"][:]
        del matrix["counts"]
        counts = create_product_counts_array(
            matrix, *values.shape, values.dtype, profile="fast_local"
        )
        counts[:] = values
        namespace[name].attrs["prepared"] = False
        finalize_test_counts(counts)
        finalize_writer_counts_t(root, name, workspace)
        assay = preset_assay_types()[name](
            z=root,
            name=name,
            workspace=workspace,
            cell_data=MetaData(namespace["cellData"]),
            nthreads=1,
        )
        assay.prepare({})


def test_repack_store_round_trip(toy_crdir_writer, tmp_path):
    output = tmp_path / "repacked.zarr"
    from cytearc import DataStore

    DataStore(toy_crdir_writer, default_assay="RNA", nthreads=1)
    repack_store(toy_crdir_writer, str(output), profile="fast_local")

    src = zarr.open_group(toy_crdir_writer, mode="r")
    dst = zarr.open_group(str(output), mode="r")

    assert set(src.keys()) == set(dst.keys())
    assay_names = [name for name in src.keys() if src[name].attrs.get("is_assay")]
    from cytearc.assay.classification import is_rna_assay_type

    for assay_name in assay_names:
        src_assay = src[assay_name]
        dst_assay = dst[assay_name]
        assert src_assay.attrs.get("is_assay") is True
        assert "counts" in dst_assay
        np.testing.assert_array_equal(
            dst_assay["counts"][...], src_assay["counts"][...]
        )
        # The copy keeps the identities that saved artifacts are bound to.
        assert dst_assay["counts"].dtype == src_assay["counts"].dtype
        assert (
            dst_assay.attrs["dataset_fingerprint"]
            == src_assay.attrs["dataset_fingerprint"]
        )
        assert (
            dst_assay["counts"].attrs["content_fingerprint"]
            == src_assay["counts"].attrs["content_fingerprint"]
        )
        if is_rna_assay_type(assay_name):
            assert dst_assay["countsT"].attrs["complete"] is True
            np.testing.assert_array_equal(
                dst_assay["countsT"][:],
                np.asarray(dst_assay["counts"][:]).T,
            )
        else:
            assert "countsT" not in dst_assay


def test_repack_v2_without_counts_t_builds_complete_transpose(tmp_path):
    source = tmp_path / "source_v2.zarr"
    output = tmp_path / "output_v3.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=2)
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assay.create_array("counts", data=values, chunks=(2, 2))

    _add_raw_metadata(root)
    repack_store(str(source), str(output), data_only=True)

    result = zarr.open_group(str(output), mode="r")
    assert result.metadata.zarr_format == 3
    assert result["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(result["RNA/counts"][:], values)
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)


def test_repack_rebuilds_incorrect_source_counts_t(tmp_path):
    source = tmp_path / "source_wrong.zarr"
    output = tmp_path / "output_fixed.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(15, dtype=np.uint32).reshape(5, 3)
    assay.create_array("counts", data=values, chunks=(2, 3))
    stale = assay.create_array(
        "countsT",
        data=np.zeros((3, 5), dtype=np.uint32),
    )
    stale.attrs["complete"] = True

    _add_raw_metadata(root)
    repack_store(str(source), str(output), data_only=True)

    result = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)
    assert result["RNA/countsT"].attrs["complete"] is True


def test_repack_discards_retired_assay_state(tmp_path):
    source = tmp_path / "source_with_state.zarr"
    output = tmp_path / "output_without_state.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts",
        data=np.arange(6, dtype=np.uint32).reshape(2, 3),
    )
    state = assay.create_group("state")
    state.create_array("legacy", data=np.array([1], dtype=np.uint8))

    _add_raw_metadata(root)
    repack_store(str(source), str(output), data_only=True)

    result = zarr.open_group(str(output), mode="r")
    assert "state" not in result["RNA"]
    np.testing.assert_array_equal(result["RNA/counts"][:], assay["counts"][:])


def test_repack_discards_state_from_every_workspace_sharing_an_assay(tmp_path):
    source = tmp_path / "source_with_workspace_state.zarr"
    output = tmp_path / "output_without_workspace_state.zarr"
    root = zarr.open_group(str(source), mode="w")
    for workspace_name in ("first", "second"):
        assay = root.create_group(f"{workspace_name}/RNA")
        assay.attrs["is_assay"] = True
        assay.create_array("metadata", data=np.array([1], dtype=np.uint8))
        state = assay.create_group("state")
        state.create_array("legacy", data=np.array([1], dtype=np.uint8))
    counts = root.create_group("matrices/RNA")
    values = np.arange(6, dtype=np.uint32).reshape(2, 3)
    counts.create_array("counts", data=values)

    _add_raw_metadata(root)
    repack_store(str(source), str(output), data_only=True)

    result = zarr.open_group(str(output), mode="r")
    assert "state" not in result["first/RNA"]
    assert "state" not in result["second/RNA"]
    assert "metadata" not in result["first/RNA"]
    assert "metadata" not in result["second/RNA"]
    np.testing.assert_array_equal(result["matrices/RNA/counts"][:], values)


def test_repack_workspace_counts_uses_requested_profile(tmp_path):
    source = tmp_path / "source_workspace.zarr"
    output = tmp_path / "output_workspace.zarr"
    root = zarr.open_group(str(source), mode="w")
    metadata = root.create_group("workspace/RNA")
    metadata.attrs["is_assay"] = True
    counts_group = root.create_group("matrices/RNA")
    values = np.arange(8, dtype=np.uint32).reshape(4, 2)
    counts_group.create_array("counts", data=values, chunks=(2, 2))

    _add_raw_metadata(root)
    repack_store(str(source), str(output), profile="cloud", data_only=True)

    result = zarr.open_group(str(output), mode="r")
    assay = result["matrices/RNA"]
    counts = assay["counts"]
    counts_t = assay["countsT"]
    np.testing.assert_array_equal(counts_t[:], values.T)
    assert counts_t.attrs["complete"] is True
    from cytearc.storage.count_matrix import load_count_matrix_plan

    spec = load_count_matrix_plan(counts)["counts"]
    assert list(spec["chunks"]) == list(counts.chunks)
    stored_shards = array_metadata_shards(counts)
    assert spec["shards"] == (None if stored_shards is None else list(stored_shards))
    assert isinstance(counts.compressors[0], ZstdCodec)
    assert isinstance(counts_t.compressors[0], ZstdCodec)


def test_repack_rejects_source_destination_alias_before_overwrite(tmp_path):
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.create_array("sentinel", data=np.array([1, 2, 3]))
    equivalent_path = source / ".." / source.name

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(str(source), str(equivalent_path))

    reopened = zarr.open_group(str(source), mode="r")
    np.testing.assert_array_equal(reopened["sentinel"][:], [1, 2, 3])


@pytest.mark.parametrize("destination_relation", ["child", "parent"])
def test_repack_rejects_overlapping_local_paths_before_overwrite(
    tmp_path,
    destination_relation,
):
    if destination_relation == "child":
        source = tmp_path / "source.zarr"
        output = source / "nested.zarr"
    else:
        output = tmp_path / "container.zarr"
        source = output / "source.zarr"

    root = zarr.open_group(str(source), mode="w")
    root.create_array("sentinel", data=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(str(source), str(output))

    reopened = zarr.open_group(str(source), mode="r")
    np.testing.assert_array_equal(reopened["sentinel"][:], [1, 2, 3])


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("s3://bucket/source.zarr", "s3://bucket/source.zarr/nested.zarr"),
        ("s3://bucket/container.zarr/source.zarr", "s3://bucket/container.zarr"),
    ],
)
def test_repack_rejects_overlapping_uri_paths_before_open(
    source,
    output,
    monkeypatch,
):
    monkeypatch.setattr(
        repack_module,
        "open_store",
        lambda *_args, **_kwargs: pytest.fail(
            "overlap must be rejected before opening either store"
        ),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(source, output)


def test_repack_preserves_root_attrs(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=2)
    root.attrs["defaultAssay"] = "RNA"
    root.attrs["assayTypes"] = {"RNA": "RNA"}
    root.attrs["complete"] = True
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(6, dtype=np.uint32).reshape(2, 3)
    assay.create_array("counts", data=values, chunks=(2, 3))

    _add_raw_metadata(root)
    repack_store(str(source), str(output), data_only=True)

    result = zarr.open_group(str(output), mode="r")
    assert result.attrs["defaultAssay"] == "RNA"
    assert result.attrs["assayTypes"] == {"RNA": "RNA"}
    assert "complete" not in result.attrs


@pytest.fixture(scope="module")
def repacked_extras(tmp_path_factory):
    """One prepared source with every kind of non-count array, repacked once."""
    directory = tmp_path_factory.mktemp("repack_extras")
    source = directory / "source.zarr"
    output = directory / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts", data=np.arange(12, dtype=np.uint32).reshape(3, 4), chunks=(2, 2)
    )
    arrays = SimpleNamespace(
        embedding=np.arange(15, dtype=np.float32).reshape(3, 5),
        sharded=np.arange(40, dtype=np.float32).reshape(8, 5),
        tall=np.arange(150 * 10, dtype=np.float32).reshape(150, 10),
        ann=np.arange(ANN_INDEX_CHUNK_BYTES + 1000, dtype=np.uint8),
    )
    assay.create_array("embedding", data=arrays.embedding, chunks=(2, 5))
    assay.create_array(
        "sharded_embedding",
        data=arrays.sharded,
        chunks=(2, 5),
        shards=(4, 5),
        fill_value=np.nan,
    )
    # Chunks and shards taller than the array clamp against its shape.
    assay.create_array(
        "tall_embedding", data=arrays.tall, chunks=(200, 10), shards=(200, 10)
    )
    cell = root.create_group("cellData")
    cell.attrs["complete"] = True
    cell.create_array("ids", data=np.array(["c1", "c2", "c3"]))
    slot = root.create_group("artifacts/marker_table/slot")
    slot.attrs["complete"] = True
    slot.create_array("values", data=np.array([1.0, 2.0]))
    ann = root.create_group("ann").create_array(
        "ann_idx_bytes", data=arrays.ann, chunks=(ANN_INDEX_CHUNK_BYTES,)
    )
    ann.attrs.update(
        {
            "ann_index_format_version": ANN_INDEX_FORMAT_VERSION,
            "metric": "l2",
            "dimensions": 15,
            "element_count": 100,
            "payload_sha256": "abc",
            "byte_length": int(arrays.ann.size),
        }
    )

    _prepare_source(root)
    repack_store(str(source), str(output), profile="cloud")
    return SimpleNamespace(
        arrays=arrays,
        ann_attrs=dict(ann.attrs),
        result=zarr.open_group(str(output), mode="r"),
    )


def test_repack_preserves_non_count_completion_attrs(repacked_extras):
    result = repacked_extras.result
    assert result["cellData"].attrs["complete"] is True
    assert result["artifacts/marker_table/slot"].attrs["complete"] is True
    np.testing.assert_array_equal(
        result["artifacts/marker_table/slot/values"][:], [1.0, 2.0]
    )


def _seed_labeled_pipeline_records(
    root,
    *,
    artifact_id,
    completed_run_id,
    running_run_id,
    completed_label,
    running_label,
):
    artifact = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=artifact_id,
    )
    artifact_node = root.create_group(artifact_path(artifact))
    artifact_node.attrs.update(
        {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "provenance": {
                "operation": "test_selection",
                "parameters": {},
                "inputs": {},
            },
            "execution_options": {},
            "complete": True,
        }
    )
    artifact_node.create_array("values", data=np.asarray([True, False]))

    completed = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=completed_label,
        assay="RNA",
        config={"filtering": False},
        stage_order=("snapshot",),
        cytearc_version="1.0.0",
        run_id=completed_run_id,
        started_at_ns=100,
    )
    start_pipeline_stage_record(
        root,
        run_id=completed.run_id,
        ordinal=0,
        stage="snapshot",
        started_at_ns=110,
    )
    finish_pipeline_stage_record(
        root,
        run_id=completed.run_id,
        ordinal=0,
        status="completed",
        outputs=(PipelineStageOutputRecord("selection", artifact, False),),
        metrics=PipelineStageMetrics(
            wall_seconds=0.01,
            rss_baseline_bytes=None,
            rss_peak_bytes=None,
            rss_incremental_peak_bytes=None,
            sample_interval_seconds=0.1,
            sample_count=0,
            sampling_error_count=0,
            rss_unavailable_reason="test",
        ),
        finished_at_ns=120,
    )
    complete_pipeline_run_record(
        root,
        run_id=completed.run_id,
        outputs=(PipelineOutputRecord("selection", artifact),),
        fields=(),
        finished_at_ns=130,
    )
    running = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=running_label,
        assay="RNA",
        config={"filtering": True},
        stage_order=("snapshot",),
        cytearc_version="1.0.0",
        run_id=running_run_id,
        started_at_ns=200,
    )
    start_pipeline_stage_record(
        root,
        run_id=running.run_id,
        ordinal=0,
        stage="snapshot",
        started_at_ns=210,
    )
    pipeline_run_storage._claim_pipeline_label(
        root,
        running_label,
        running.run_id,
    )
    return completed.run_id, running.run_id


def test_repack_copies_pipeline_records_and_label_claims_in_all_workspaces(tmp_path):
    source = tmp_path / "source_runs.zarr"
    output = tmp_path / "output_runs.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts",
        data=np.arange(6, dtype=np.uint32).reshape(2, 3),
        chunks=(2, 3),
    )

    root_run_ids = _seed_labeled_pipeline_records(
        root,
        artifact_id="c" * 64,
        completed_run_id="a" * 64,
        running_run_id="b" * 64,
        completed_label="baseline",
        running_label="interrupted",
    )
    workspace = root.create_group("analysis_workspace")
    workspace_run_ids = _seed_labeled_pipeline_records(
        workspace,
        artifact_id="f" * 64,
        completed_run_id="d" * 64,
        running_run_id="e" * 64,
        completed_label="workspace-baseline",
        running_label="workspace-interrupted",
    )
    namespaces = (
        ("", root, root_run_ids, "baseline", "interrupted"),
        (
            "analysis_workspace",
            workspace,
            workspace_run_ids,
            "workspace-baseline",
            "workspace-interrupted",
        ),
    )
    record_paths = tuple(
        f"{prefix + '/' if prefix else ''}pipeline/runs/{run_id}{suffix}"
        for prefix, _namespace, run_ids, _completed_label, _running_label in namespaces
        for run_id in run_ids
        for suffix in ("", "/stages/0")
    )
    source_attrs = {path: dict(root[path].attrs) for path in record_paths}

    _prepare_source(root)
    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r+")
    assert {path: dict(result[path].attrs) for path in record_paths} == source_attrs
    for prefix, source_root, run_ids, completed_label, running_label in namespaces:
        result_root = result if prefix == "" else result[prefix]
        completed_run_id, running_run_id = run_ids
        assert load_pipeline_run_record(
            result_root,
            completed_run_id,
        ) == load_pipeline_run_record(source_root, completed_run_id)
        assert load_pipeline_run_record(
            result_root,
            running_run_id,
        ) == load_pipeline_run_record(source_root, running_run_id)

        with pytest.raises(ValueError, match="already committed"):
            pipeline_run_storage._claim_pipeline_label(
                result_root,
                completed_label,
                "1" * 64,
            )
        with pytest.raises(ValueError, match="currently being finalized"):
            pipeline_run_storage._claim_pipeline_label(
                result_root,
                running_label,
                "2" * 64,
            )


def test_repack_skips_copying_counts_t_when_sharding(tmp_path, monkeypatch):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assay.create_array("counts", data=values, chunks=(2, 2))
    stale = assay.create_array("countsT", data=np.zeros((4, 3), dtype=np.uint32))
    stale.attrs["complete"] = True

    created: list[str] = []
    real_create = zarr.Group.create_array

    def tracking_create(self, name, *args, **kwargs):
        created.append(name)
        return real_create(self, name, *args, **kwargs)

    monkeypatch.setattr(zarr.Group, "create_array", tracking_create)
    _add_raw_metadata(root)
    repack_store(str(source), str(output), data_only=True)

    assert created.count("countsT") == 1
    result = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)


def test_repack_streams_non_count_2d_arrays(repacked_extras):
    result = repacked_extras.result
    np.testing.assert_array_equal(
        result["RNA/embedding"][:], repacked_extras.arrays.embedding
    )
    assert result["RNA/embedding"].chunks == (2, 5)
    np.testing.assert_array_equal(result["cellData/ids"][:], ["c1", "c2", "c3"])


def test_repack_preserves_ann_like_1d_chunks_and_attrs(repacked_extras):
    result = repacked_extras.result["ann/ann_idx_bytes"]
    payload = repacked_extras.arrays.ann
    # The payload spans two 8 MiB chunks, the second one short.
    assert result.chunks == (ANN_INDEX_CHUNK_BYTES,)
    assert result.chunks == normalize_chunks((ANN_INDEX_CHUNK_BYTES,), payload.shape)
    np.testing.assert_array_equal(result[:], payload)
    for key in _ANN_INDEX_METADATA:
        assert result.attrs[key] == repacked_extras.ann_attrs[key]
    assert result.attrs["byte_length"] == payload.size


def test_repack_preserves_non_count_2d_shards(repacked_extras):
    result = repacked_extras.result["RNA/sharded_embedding"]
    np.testing.assert_array_equal(result[:], repacked_extras.arrays.sharded)
    assert result.chunks == (2, 5)
    assert array_metadata_shards(result) == (4, 5)
    assert isinstance(result.compressors[0], ZstdCodec)


def test_repack_realigns_shards_when_chunks_clamp(repacked_extras):
    result = repacked_extras.result["RNA/tall_embedding"]
    np.testing.assert_array_equal(result[:], repacked_extras.arrays.tall)
    assert result.chunks == (150, 10)
    assert result.chunks == normalize_chunks((200, 10), (150, 10))
    assert array_metadata_shards(result) == result.chunks


def test_main_keeps_remote_uris_and_storage_options(monkeypatch):
    captured: dict = {}

    def fake_repack(input_path, output_path, **kwargs):
        captured["input"] = input_path
        captured["output"] = output_path
        captured["storage_options"] = kwargs.get("storage_options")
        captured["profile"] = kwargs.get("profile")
        captured["mem_budget"] = kwargs.get("mem_budget")
        captured["nthreads"] = kwargs.get("nthreads")

    monkeypatch.setattr(repack_module, "repack_store", fake_repack)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repack_zarr",
            "s3://bucket/in.zarr",
            "s3://bucket/out.zarr",
            "--profile",
            "cloud",
            "--mem-budget",
            "8G",
            "--nthreads",
            "4",
            "--storage-options",
            '{"skip_signature": true}',
        ],
    )
    repack_module.main()
    assert captured["input"] == "s3://bucket/in.zarr"
    assert captured["output"] == "s3://bucket/out.zarr"
    assert captured["profile"] == "cloud"
    assert captured["mem_budget"] == "8G"
    assert captured["nthreads"] == 4
    assert captured["storage_options"] == {"skip_signature": True}


def test_repack_that_fails_after_creating_the_destination_removes_it(tmp_path):
    import hashlib

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    source = tmp_path / "source.zarr"
    counts = np.random.default_rng(0).poisson(2.0, size=(12, 6)).astype(np.uint32)
    write_count_store(str(source), {"RNA": counts}, np.uint32)
    DataStore(str(source), default_assay="RNA", min_features_per_cell=1)
    root = zarr.open_group(str(source), mode="r+")
    run = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label="alpha",
        assay="RNA",
        config={},
        stage_order=("one",),
        cytearc_version="1.0.0",
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=0, stage="one")
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=0,
        status="completed",
        metrics=PipelineStageMetrics(
            wall_seconds=0.1,
            rss_baseline_bytes=None,
            rss_peak_bytes=None,
            rss_incremental_peak_bytes=None,
            sample_interval_seconds=0.1,
            sample_count=0,
            sampling_error_count=0,
            rss_unavailable_reason="test",
        ),
    )
    complete_pipeline_run_record(root, run_id=run.run_id, outputs=(), fields=())
    # Label claims are copied last, after the destination holds everything
    # else; an unreadable one fails the copy.
    assert "0" * 64 < hashlib.sha256(b"alpha").hexdigest()
    sync(
        root.store.set(
            f"pipeline/runs/.label-claims/{'0' * 64}/head.json",
            default_buffer_prototype().buffer.from_bytes(b"not a claim"),
        )
    )
    output = tmp_path / "repacked.zarr"
    with pytest.raises(ValueError, match="Pipeline label claim is invalid"):
        repack_store(str(source), str(output))
    # No destination is left that could open as a complete store.
    assert not output.exists()


def _sharded_counts_source(path, values: np.ndarray) -> zarr.Array:
    """Write ``values`` as counts of 1,000-row shards that hold ten 40-gene chunks."""
    from scipy.sparse import csr_matrix

    from cytearc.storage.count_matrix import CountMatrixPolicy
    from cytearc.writers import SparseToZarr

    n_cells, n_genes = values.shape
    SparseToZarr(
        csr_matrix(values),
        str(path),
        cell_ids=[f"c{index}" for index in range(n_cells)],
        feature_ids=[f"g{index}" for index in range(n_genes)],
        nthreads=1,
        shuffle_cells=False,
        policy=CountMatrixPolicy(unitBytes=1_600_000, chunkBytes=160_000),
    ).dump()
    counts = zarr.open_group(str(path), mode="r")["RNA/counts"]
    assert (counts.chunks, counts.shards) == ((1_000, 40), (1_000, 400))
    return counts


def _poisson_counts() -> np.ndarray:
    values = np.random.default_rng(5).poisson(2.0, size=(2_000, 400))
    # One count past uint16 keeps the counts uint32, 1,600 bytes a cell.
    values[0, 0] = 70_000
    return values.astype(np.uint32)


@pytest.fixture(scope="module")
def sharded_counts_store(tmp_path_factory):
    """Counts of 2,000 cells by 400 genes in shards of ten chunks, ready to repack."""
    path = tmp_path_factory.mktemp("sharded_counts") / "source.zarr"
    _sharded_counts_source(path, _poisson_counts())
    return path


def test_a_repack_reserves_what_reading_a_band_of_sharded_counts_holds(
    tmp_path, monkeypatch
):
    import tracemalloc

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.layout import array_shard_rows

    # Random bits do not compress, so each chunk is stored at the size that
    # the charge takes for it.
    values = np.random.default_rng(4).integers(
        0, 1 << 32, size=(2_000, 400), dtype=np.uint32
    )
    source = _sharded_counts_source(tmp_path / "source.zarr", values)
    recorded: list[tuple[int, int]] = []
    write = repack_module.write_dense_in_shard_rows

    def recording(destination, produce, **kwargs):
        recorded.append((array_shard_rows(destination), kwargs["producerBytes"]))
        return write(destination, produce, **kwargs)

    monkeypatch.setattr(repack_module, "write_dense_in_shard_rows", recording)
    output = tmp_path / "output.zarr"
    repack_store(
        str(tmp_path / "source.zarr"),
        str(output),
        data_only=True,
        mem_budget=4 * 1024**2,
        nthreads=1,
    )
    ((band_rows, producer),) = recorded
    # The budget fits count shards shorter than a source shard, so each band
    # of the copy reads part of one source shard.
    assert band_rows < 1_000

    # A band read decodes one chunk at a time, as the charge prices it,
    # after a read that warms Zarr's own buffers.
    produce = repack_module._row_block_producer(source)
    with zarr_io_concurrency(1):
        produce(0, band_rows)
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            band = produce(0, band_rows)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    held = peak - base - band.nbytes
    # Beside the band, the read holds the ten stored 160,000-byte chunks of
    # the shard and a shard-level copy of the band, which one decoded chunk,
    # the charge before, left out.
    assert held > 10 * 160_000 + band.nbytes
    assert held <= producer
    np.testing.assert_array_equal(
        zarr.open_group(str(output), mode="r")["RNA/counts"][:], values
    )


def test_a_repack_at_a_tight_budget_reads_each_band_within_its_reserve(
    tmp_path, monkeypatch, sharded_counts_store
):
    import tracemalloc

    from cytearc.storage import sharding

    # At 4 MiB the copy writes count shards shorter than the source's, and a
    # band of them still reads whole source chunks, compressed and decoded.
    reads: list[tuple[int, int]] = []
    write = repack_module.write_dense_in_shard_rows

    def traced(destination, produce, **kwargs):
        plans = []
        plan_write = sharding.plan_dense_write

        def recording_plan(*args, **plan_kwargs):
            plans.append(plan_write(*args, **plan_kwargs))
            return plans[-1]

        def traced_produce(start: int, end: int) -> np.ndarray:
            tracemalloc.start()
            try:
                base, _ = tracemalloc.get_traced_memory()
                block = produce(start, end)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            (plan,) = plans
            per_worker = (plan.reservedBytes - plan.residentBytes) // plan.readWorkers
            reads.append((peak - base, per_worker))
            return block

        with monkeypatch.context() as patched:
            patched.setattr(sharding, "plan_dense_write", recording_plan)
            return write(destination, traced_produce, **kwargs)

    monkeypatch.setattr(repack_module, "write_dense_in_shard_rows", traced)
    output = tmp_path / "output.zarr"
    repack_store(
        str(sharded_counts_store),
        str(output),
        data_only=True,
        mem_budget=4 * 1024**2,
        nthreads=1,
    )
    # The first read also fills Zarr's own buffers.
    assert len(reads) > 2
    for held, reserved in reads[1:]:
        assert held <= reserved
    np.testing.assert_array_equal(
        zarr.open_group(str(output), mode="r")["RNA/counts"][:], _poisson_counts()
    )


def test_a_repack_that_cannot_hold_a_band_read_fails_before_the_destination_exists(
    tmp_path, sharded_counts_store
):
    from cytearc.utils.memory_budget import MemoryBudgetError

    # Whatever count shards the copy writes, a band read holds the ten
    # stored chunks of a source shard, over 1.6 MB, which 1 MiB cannot.
    output = tmp_path / "output.zarr"
    with pytest.raises(MemoryBudgetError, match="count shards of one row") as refused:
        repack_store(
            str(sharded_counts_store),
            str(output),
            data_only=True,
            mem_budget=1024**2,
            nthreads=1,
        )
    assert refused.value.needed_bytes > 10 * 160_000
    assert not output.exists()


def test_a_repack_reserves_the_shard_read_of_another_two_dimensional_array(
    tmp_path, monkeypatch
):
    import cytearc.storage.copy as copy_module

    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts", data=np.arange(32, dtype=np.uint32).reshape(8, 4), chunks=(2, 2)
    )
    embedding = np.arange(40, dtype=np.float32).reshape(8, 5)
    assay.create_array("embedding", data=embedding, chunks=(2, 5), shards=(4, 5))
    _prepare_source(root)
    recorded: dict[str, int] = {}
    write = copy_module.write_dense_in_shard_rows

    def recording(destination, produce, **kwargs):
        recorded[destination.basename] = kwargs["producerBytes"]
        return write(destination, produce, **kwargs)

    monkeypatch.setattr(copy_module, "write_dense_in_shard_rows", recording)
    output = tmp_path / "output.zarr"
    repack_store(str(source), str(output))

    # A band of all four rows of a shard fills an 80-byte shard-level copy of
    # them, holds the compressed bytes of the shard's two 40-byte chunks, each
    # at most 40 + 40 // 128 + 1,024 bytes, and decodes one chunk at a time
    # beside the chunk decoded before it, where the copy reserved one chunk.
    assert recorded["embedding"] == 80 + 2 * (40 + 1_024) + 40 + 40
    np.testing.assert_array_equal(
        zarr.open_group(str(output), mode="r")["RNA/embedding"][:], embedding
    )
