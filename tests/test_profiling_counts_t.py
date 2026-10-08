import re

import numpy as np
import pytest
import zarr

from profiling import stages as profiling_stages
from profiling.config import (
    CountMatrixConfig,
    StageResources,
    StorageIoConfig,
    WorkflowParameters,
)
from profiling.stages import run_stage
from cytearc.storage.count_matrix import (
    CountMatrixPolicy,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)


def _resources() -> StageResources:
    # Leave headroom above memory retained by earlier tests in this worker.
    return StageResources(
        modalMemoryRequestMb=4096,
        modalMemoryLimitMb=4096,
        modalCpuRequest=1.0,
        modalCpuLimit=1.0,
        cytearcMemoryBudget=int(profiling_stages.process_rss_mb() * 1024**2)
        + 2 * 1024**3,
        workers=1,
        timeoutSeconds=600,
        ephemeralDiskMb=524288,
    )


def _seed_counts(root_path, values: np.ndarray) -> None:
    policy = CountMatrixPolicy(unitBytes=2_000, chunkBytes=200)
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=policy,
    )
    root = zarr.open_group(str(root_path), mode="w")
    group = root.create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=plan.counts.shape,
        chunks=plan.counts.chunks,
        shards=plan.counts.shards,
        dtype=values.dtype,
        overwrite=True,
    )
    counts[:] = values
    from tests.storage_helpers import finalize_test_counts

    finalize_test_counts(counts)
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)


def test_write_counts_t_rewrites_incomplete_array(tmp_path):
    from cytearc.storage.sharding import write_counts_t

    root_path = tmp_path / "store.zarr"
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    _seed_counts(root_path, values)
    root = zarr.open_group(str(root_path), mode="r+")
    group = root["RNA"]
    counts = group["counts"]
    counts_t = write_counts_t(counts, group)
    counts_t.attrs["complete"] = False
    counts_t[0, 0] = 999

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "ok"
    assert result.details is not None
    assert result.details["complete"] is True
    assert result.details["beforeComplete"] is False
    reopened = zarr.open_group(str(root_path), mode="r")
    fixed = reopened["RNA/countsT"]
    assert fixed.attrs["complete"] is True
    np.testing.assert_array_equal(fixed[:], values.T)


def test_write_counts_t_runs_as_standard_profile_stage(tmp_path):
    root_path = tmp_path / "store.zarr"
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    _seed_counts(root_path, values)

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "ok"
    assert result.inputSetupSeconds is not None
    assert result.seconds is not None
    assert result.validationPersistenceSeconds is not None
    assert result.details is not None
    assert result.details["complete"] is True
    assert result.details["shape"] == [4, 6]
    assert result.details["beforeComplete"] is None
    reopened = zarr.open_group(str(root_path), mode="r")
    np.testing.assert_array_equal(reopened["RNA/countsT"][:], values.T)


def test_write_counts_t_accounts_for_process_resident_memory(tmp_path, monkeypatch):
    root_path = tmp_path / "store.zarr"
    _seed_counts(root_path, np.arange(24, dtype=np.uint32).reshape(6, 4))
    resources = _resources().model_copy(update={"cytearcMemoryBudget": 1024**2})

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=resources,
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "error"
    # The resident process memory alone exceeds the 1 MiB budget.
    assert result.error is not None
    assert re.fullmatch(
        r"MemoryError: countsT write needs at least \d+ bytes, "
        r"but the operation limit is 1048576 bytes",
        result.error,
    )
    assert "countsT" not in zarr.open_group(str(root_path), mode="r")["RNA"]

    # Without /proc the resident memory is unknown, so the write is refused.
    monkeypatch.setattr(profiling_stages, "process_rss_mb", lambda: None)
    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=resources,
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )
    assert result.error is not None
    assert result.error.startswith(
        "RuntimeError: Profiling stages measure resident memory through Linux /proc"
    )
    assert "countsT" not in zarr.open_group(str(root_path), mode="r")["RNA"]


def test_write_counts_t_forwards_storage_io(tmp_path):
    from tests.storage_helpers import reset_zarr_runtime

    reset_zarr_runtime()
    root_path = tmp_path / "store.zarr"
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    _seed_counts(root_path, values)

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        countMatrix=CountMatrixConfig(unitBytes=2_000, chunkBytes=200),
        storageIo=StorageIoConfig(
            readWorkers=2,
            writeWorkers=2,
            computeWorkers=1,
        ),
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "ok", result.error
    assert result.details is not None
    assert result.details["writer"] == "product"
    metrics = result.details["metrics"]
    assert metrics["requestedWriteWorkers"] == 2
    assert metrics["requestedDestCommitsInFlight"] == 2
    reopened = zarr.open_group(str(root_path), mode="r")
    np.testing.assert_array_equal(reopened["RNA/countsT"][:], values.T)
    reset_zarr_runtime()


def test_create_store_defers_counts_t_to_write_stage(tmp_path):
    import h5py
    from scipy.sparse import csr_matrix

    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    h5ad_path = tmp_path / "cells.h5ad"
    with h5py.File(h5ad_path, "w") as h5:
        group = h5.create_group("X")
        matrix = csr_matrix(values)
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int32))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int32))
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = values.shape
        obs = h5.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array([f"c{i}".encode() for i in range(values.shape[0])]),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array([f"f{i}".encode() for i in range(values.shape[1])]),
        )
        var.create_dataset(
            "feature_name",
            data=np.array([f"g{i}".encode() for i in range(values.shape[1])]),
        )

    store_path = tmp_path / "store.zarr"
    create = run_stage(
        "createStore",
        nRows=values.shape[0],
        storeUri=str(store_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        localH5adPath=h5ad_path,
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )
    assert create.status == "ok"
    assert create.details is not None
    assert create.details["storeOperations"]["sets"] > 0
    # The published tables time the counts write only; the result says so.
    assert create.details["timingDefinition"].startswith(
        "seconds covers writer._write_counts only"
    )
    after_create = zarr.open_group(str(store_path), mode="r")
    assert "countsT" not in after_create["RNA"]
    np.testing.assert_array_equal(after_create["RNA/counts"][:], values)

    write = run_stage(
        "writeCountsT",
        nRows=values.shape[0],
        storeUri=str(store_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )
    assert write.status == "ok"
    assert write.details is not None
    assert write.details["beforeComplete"] is None
    assert write.details["complete"] is True
    after_write = zarr.open_group(str(store_path), mode="r")
    np.testing.assert_array_equal(after_write["RNA/countsT"][:], values.T)


def _write_h5ad_input(path, matrix_key):
    import anndata
    import pandas as pd
    from scipy.sparse import csr_matrix

    counts = np.array([[1, 0, 3], [4, 2, 0], [0, 3, 1], [2, 1, 5]], dtype=np.float32)
    obs = pd.DataFrame(index=pd.Index(["c1", "c2", "c3", "c4"], name="cell_id"))
    var = pd.DataFrame(
        {"gene_symbols": ["RPL1", "MT-X", "GENE"]},
        index=pd.Index(["g1", "g2", "g3"], name="gene_id"),
    )
    data = anndata.AnnData(X=csr_matrix(counts / 10), obs=obs, var=var)
    if matrix_key == "raw/X":
        data.raw = anndata.AnnData(X=csr_matrix(counts), obs=obs, var=var)
        data = data[:, :2].copy()
    elif matrix_key == "layers/counts":
        data.layers["counts"] = csr_matrix(counts)
    elif matrix_key == "X":
        data.X = csr_matrix(counts)
    data.write_h5ad(path)
    return counts


@pytest.mark.parametrize("matrix_key", ["X", "raw/X", "layers/counts"])
def test_create_store_inspects_h5ad_count_matrix_and_feature_names(
    tmp_path, matrix_key
):
    path = tmp_path / "arbitrary-name.h5ad"
    counts = _write_h5ad_input(path, matrix_key)
    store_path = tmp_path / "store.zarr"

    result = run_stage(
        "createStore",
        nRows=len(counts),
        storeUri=str(store_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        localH5adPath=path,
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "ok", result.error
    root = zarr.open_group(str(store_path), mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], counts)
    np.testing.assert_array_equal(
        root["RNA/featureData/names"][:], ["RPL1", "MT-X", "GENE"]
    )
    assert result.details["h5adInput"] == {
        "matrixKey": matrix_key,
        "featureAttrsKey": "raw/var" if matrix_key == "raw/X" else "var",
        "featureNameKey": "gene_symbols",
        "nRows": 4,
        "nColumns": 3,
    }
    assert "countsT" not in root["RNA"]


@pytest.mark.parametrize(
    "matrix_key, n_rows, message",
    [
        ("X", 999, "H5AD contains 4 cells, but the requested size is 999"),
        ("normalized", 4, "H5AD has no integer-like count matrix"),
    ],
)
def test_create_store_rejects_wrong_size_or_transformed_input_before_writing(
    tmp_path, matrix_key, n_rows, message
):
    path = tmp_path / "arbitrary-name.h5ad"
    _write_h5ad_input(path, matrix_key)
    store_path = tmp_path / "store.zarr"

    result = run_stage(
        "createStore",
        nRows=n_rows,
        storeUri=str(store_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        localH5adPath=path,
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "error"
    assert message in result.error
    assert not store_path.exists()


def test_write_counts_t_clears_complete_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    root_path = tmp_path / "store.zarr"
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    _seed_counts(root_path, values)
    original_write = profiling_stages._write_counts_t

    def corrupt_counts_t(*args, **kwargs):
        counts_t, details = original_write(*args, **kwargs)
        counts_t[0, 0] = 999
        return counts_t, details

    monkeypatch.setattr(profiling_stages, "_write_counts_t", corrupt_counts_t)

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
        submissionId="testsubmission",
    )

    assert result.status == "error"
    assert result.error is not None
    assert "tile mismatch" in result.error
    reopened = zarr.open_group(str(root_path), mode="r")
    assert reopened["RNA/countsT"].attrs["complete"] is False
