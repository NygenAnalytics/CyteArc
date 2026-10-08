from pathlib import Path
import hashlib
import io
import json
import shutil
from dataclasses import replace

import h5py
import numpy as np
import pytest
from obstore.store import MemoryStore

from profiling.config import ProfilingConfig, load_profiling_config
from profiling.datasets import (
    SourceSpec,
    download_source,
    load_csr_source_into_memory,
    ordered_source_row_digest,
    prepare_local_datasets,
    select_nested_rows,
    write_fixture_h5ad,
    write_h5ad_sample_from_memory,
)


def _fixture_spec(path: Path, *, nRows: int, nColumns: int, nnz: int) -> SourceSpec:
    return SourceSpec(
        datasetId="fixture",
        versionId="fixture-v1",
        url="file://fixture",
        nRows=nRows,
        nColumns=nColumns,
        nnz=nnz,
        sourceBytes=path.stat().st_size,
    )


def test_load_csr_downcasts_indices_to_int32(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=40, nColumns=25, seed=3)
    spec = _fixture_spec(
        source,
        nRows=40,
        nColumns=25,
        nnz=artifact.nnz,
    )
    memory = load_csr_source_into_memory(source, spec=spec)
    assert memory.indices.dtype == np.dtype(np.int32)
    assert memory.indicesDtype == np.dtype(np.int64)
    assert memory.data.dtype == np.dtype(np.float32)
    assert int(memory.indptr[-1]) == artifact.nnz


def test_write_from_memory_selects_exact_source_rows(tmp_path: Path) -> None:
    from scipy.sparse import csr_matrix

    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=60, nColumns=20, seed=4)
    spec = _fixture_spec(
        source,
        nRows=60,
        nColumns=20,
        nnz=artifact.nnz,
    )
    rows = select_nested_rows(60, (15,), seed=1, sourceVersion=spec.versionId)[15]
    memory = load_csr_source_into_memory(source, spec=spec)

    sample_path = tmp_path / "memory.h5ad"
    written = write_h5ad_sample_from_memory(memory, sample_path, rows)

    with h5py.File(source, "r") as h5:
        expected = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=(60, 20),
        )[rows]
        expected_ids = h5["obs/_index"][:][rows]
    with h5py.File(sample_path, "r") as h5:
        assert np.array_equal(h5["X/data"][:], expected.data)
        assert np.array_equal(h5["X/indices"][:], expected.indices)
        assert np.array_equal(h5["X/indptr"][:], expected.indptr)
        assert np.array_equal(h5["obs/_index"][:], expected_ids)
    assert written.nnz == expected.nnz
    assert written.sourceRowsSha256 == ordered_source_row_digest(rows)
    assert written.finalSourceRow == int(rows[-1])


def test_prepare_local_datasets_uses_in_memory_path(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=80, nColumns=30, seed=5)
    spec = _fixture_spec(
        source,
        nRows=80,
        nColumns=30,
        nnz=artifact.nnz,
    )
    prepared = prepare_local_datasets(
        source,
        tmp_path / "subsets",
        targetRows=(10, 25),
        seed=0,
        spec=spec,
    )
    assert [item.targetRows for item in prepared.artifacts] == [10, 25]

    selections = select_nested_rows(
        80,
        (10, 25),
        seed=0,
        sourceVersion=spec.versionId,
    )
    assert set(selections[10].tolist()).issubset(set(selections[25].tolist()))
    with h5py.File(source, "r") as h5:
        source_ids = h5["obs/_index"][:]
    for artifact in prepared.artifacts:
        rows = selections[artifact.targetRows]
        assert (
            artifact.localPath == tmp_path / "subsets" / f"{artifact.targetRows}.h5ad"
        )
        assert artifact.sourceRowsSha256 == ordered_source_row_digest(rows)
        assert artifact.finalSourceRow == int(rows[-1])
        assert artifact.sourceSha256 == prepared.sourceSha256
        # Each sample holds exactly its selected source rows, in order.
        with h5py.File(artifact.localPath, "r") as h5:
            np.testing.assert_array_equal(h5["obs/_index"][:], source_ids[rows])
            assert h5["X"].attrs["shape"].tolist() == [artifact.targetRows, 30]


@pytest.fixture
def sample_preparation(tmp_path, monkeypatch):
    from profiling import app

    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=80, nColumns=30, seed=5)
    spec = _fixture_spec(source, nRows=80, nColumns=30, nnz=artifact.nnz)
    config = load_profiling_config(
        Path(__file__).parents[1] / "profiling" / "config.example.toml"
    ).model_copy(
        update={"datasetPrefixUri": "s3://bucket/samples", "targetSizes": (10, 25)}
    )
    store = MemoryStore()
    store.put("samples/source.h5ad", source)
    monkeypatch.setattr(
        "profiling.r2.open_r2_object",
        lambda uri: (store, uri.removeprefix("s3://bucket/")),
    )
    monkeypatch.setattr(app, "SOURCE_SPEC", spec)
    monkeypatch.setattr(app, "_WORK", tmp_path / "work")
    return app, config, store, source


@pytest.mark.parametrize("command", ["prepare", "prepare-fixture"])
@pytest.mark.parametrize("entrypoint", ["client", "worker"])
def test_prepare_rejects_direct_input_before_io(
    sample_preparation, monkeypatch, command, entrypoint
):
    app, config, store, _ = sample_preparation
    payload = config.model_dump()
    payload.update(inputUri="s3://bucket/samples/source.h5ad", targetSizes=(80,))
    config = ProfilingConfig.model_validate(payload)
    before = bytes(store.get("samples/source.h5ad").bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct input preparation must fail before I/O or spawn")

    monkeypatch.setattr("profiling.r2.open_r2_object", forbidden)
    monkeypatch.setattr(app, "_launch", forbidden)
    monkeypatch.setattr(app, "attach_client_provenance", forbidden)
    monkeypatch.setattr(app, "_load_config", lambda _: config)

    with pytest.raises(
        SystemExit if entrypoint == "client" else ValueError,
        match=f"{command} cannot be used with inputUri",
    ):
        if entrypoint == "client":
            app.main(command, "--config", "unused.toml")
        elif command == "prepare":
            app.prepare_datasets.local(config.model_dump())
        else:
            app.prepare_fixture_datasets_job.local(config.model_dump(), sizes=[80])

    assert not app._WORK.exists()
    assert bytes(store.get("samples/source.h5ad").bytes()) == before
    assert [item["path"] for batch in store.list() for item in batch] == [
        "samples/source.h5ad"
    ]


def test_prepare_records_identity_and_skips_without_source_io(sample_preparation):
    app, config, store, source = sample_preparation
    result = app.prepare_datasets.local(config.model_dump())
    assert [item["nRows"] for item in result["uploaded"]] == [10, 25]
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    for n_rows in config.targetSizes:
        payload = bytes(store.get(f"samples/{n_rows}.h5ad").bytes())
        metadata = json.loads(bytes(store.get(f"samples/{n_rows}.h5ad.json").bytes()))
        assert metadata["fileSha256"] == hashlib.sha256(payload).hexdigest()
        assert metadata["sourceSha256"] == source_sha
        assert metadata["fileBytes"] == len(payload)
        assert metadata["eTag"] == store.head(f"samples/{n_rows}.h5ad")["e_tag"]
        assert metadata["status"] == "complete"
        with h5py.File(io.BytesIO(payload), "r") as h5:
            rows = np.array(
                [
                    int(name.decode().removeprefix("cell-"))
                    for name in h5["obs/_index"][:]
                ]
            )
            assert metadata["sourceRowsSha256"] == ordered_source_row_digest(rows)
            assert metadata["nnz"] == int(h5["X/indptr"][-1])
            assert metadata["dataDtype"] == str(h5["X/data"].dtype)
    store.delete("samples/source.h5ad")
    shutil.rmtree(app._WORK)
    skipped = app.prepare_datasets.local(config.model_dump())
    assert skipped["uploaded"] == []
    assert len(skipped["skipped"]) == 2
    assert skipped["sourceOrigin"] == "not-needed"
    assert skipped["sourceSha256"] == source_sha
    assert not app._WORK.exists()


def test_prepare_rejects_unverified_large_fixture_before_source_io(sample_preparation):
    app, config, store, source = sample_preparation
    fixture = write_fixture_h5ad(
        source.parent / "large.h5ad", nRows=50_000, nColumns=500
    )
    assert fixture.fileBytes > 5_000_000
    store.put("samples/10.h5ad", fixture.localPath)
    before = {
        item["path"]: bytes(store.get(item["path"]).bytes())
        for batch in store.list()
        for item in batch
    }
    with pytest.raises(ValueError, match="no identity metadata"):
        app.prepare_datasets.local(config.model_dump())
    assert not app._WORK.exists()
    assert {
        item["path"]: bytes(store.get(item["path"]).bytes())
        for batch in store.list()
        for item in batch
    } == before


@pytest.mark.parametrize(
    "mismatch", ["seed", "source", "kind", "object", "reservation"]
)
def test_prepare_rejects_changed_identity_without_writes(
    sample_preparation, monkeypatch, mismatch
):
    app, config, store, _ = sample_preparation
    app.prepare_datasets.local(config.model_dump())
    shutil.rmtree(app._WORK)
    if mismatch == "seed":
        config = config.model_copy(update={"samplingSeed": config.samplingSeed + 1})
    elif mismatch == "source":
        monkeypatch.setattr(
            app, "SOURCE_SPEC", replace(app.SOURCE_SPEC, versionId="another-source")
        )
    elif mismatch == "object":
        payload = bytes(store.get("samples/10.h5ad").bytes())
        store.put("samples/10.h5ad", b"!" + payload[1:])
    else:
        metadata = json.loads(bytes(store.get("samples/10.h5ad.json").bytes()))
        metadata["kind" if mismatch == "kind" else "status"] = (
            "fixture" if mismatch == "kind" else "preparing"
        )
        store.put("samples/10.h5ad.json", json.dumps(metadata).encode())
    before = {
        item["path"]: bytes(store.get(item["path"]).bytes())
        for batch in store.list()
        for item in batch
    }
    with pytest.raises(ValueError, match="does not match|unfinished"):
        app.prepare_datasets.local(config.model_dump())
    assert not app._WORK.exists()
    assert {
        item["path"]: bytes(store.get(item["path"]).bytes())
        for batch in store.list()
        for item in batch
    } == before


def test_prepare_fills_only_missing_sizes(sample_preparation):
    app, config, store, _ = sample_preparation
    small = config.model_copy(update={"targetSizes": (10,)})
    app.prepare_datasets.local(small.model_dump())
    sample_before = bytes(store.get("samples/10.h5ad").bytes())
    metadata_before = bytes(store.get("samples/10.h5ad.json").bytes())
    result = app.prepare_datasets.local(config.model_dump())
    assert [item["nRows"] for item in result["uploaded"]] == [25]
    assert [item["nRows"] for item in result["skipped"]] == [10]
    assert bytes(store.get("samples/10.h5ad").bytes()) == sample_before
    assert bytes(store.get("samples/10.h5ad.json").bytes()) == metadata_before


def test_upload_sample_keeps_interrupted_reservation(sample_preparation, monkeypatch):
    app, config, store, source = sample_preparation
    artifact = write_fixture_h5ad(source.parent / "new.h5ad", nRows=10, nColumns=30)

    def interrupted(*args, **kwargs):
        raise OSError("upload interrupted")

    monkeypatch.setattr(app, "upload_file", interrupted)
    with pytest.raises(OSError, match="upload interrupted") as exc:
        app._upload_sample(config, artifact, kind="fixture")
    assert "samples/10.h5ad.json" in exc.value.__notes__[0]
    metadata = json.loads(bytes(store.get("samples/10.h5ad.json").bytes()))
    assert metadata["status"] == "preparing"
    with pytest.raises(FileExistsError, match="reservation already exists"):
        app._upload_sample(config, artifact, kind="fixture")
    with pytest.raises(FileNotFoundError):
        store.head("samples/10.h5ad")


def test_fixture_upload_records_identity_and_preserves_existing_sample(
    sample_preparation,
):
    app, config, store, _ = sample_preparation
    app.prepare_fixture_datasets_job.local(config.model_dump(), sizes=[10], nColumns=30)
    payload = bytes(store.get("samples/10.h5ad").bytes())
    metadata_payload = bytes(store.get("samples/10.h5ad.json").bytes())
    metadata = json.loads(metadata_payload)
    assert metadata["kind"] == "fixture"
    assert metadata["source"] is None
    assert metadata["fileSha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        app.prepare_fixture_datasets_job.local(
            config.model_dump(), sizes=[10], nColumns=30
        )
    assert bytes(store.get("samples/10.h5ad").bytes()) == payload
    assert bytes(store.get("samples/10.h5ad.json").bytes()) == metadata_payload
    with pytest.raises(ValueError, match="kind"):
        app.prepare_datasets.local(config.model_dump())


def test_example_config_loads_prepare_resources() -> None:
    config = load_profiling_config(
        Path(__file__).parents[1] / "profiling" / "config.example.toml"
    )
    assert config.prepareResources.modalMemoryRequestMb == 196_608
    assert config.prepareResources.modalMemoryLimitMb == 212_992


class _Response:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        self.payload = payload
        self.status = status
        self.headers = headers
        self.stall = False

    def read(self, size: int) -> bytes:
        if self.stall and not self.payload:
            raise TimeoutError("read timed out")
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk

    def close(self) -> None:
        pass


def test_download_source_resumes_a_stalled_transfer(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 40
    requests: list[int] = []

    def opener(_url: str, offset: int) -> _Response:
        requests.append(offset)
        if offset == 0:
            # The first connection stalls after 3000 bytes.
            response = _Response(
                payload[:3000],
                status=200,
                headers={"Content-Length": str(len(payload))},
            )
            response.stall = True
            return response
        return _Response(
            payload[offset:],
            status=206,
            headers={
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
            },
        )

    result = download_source(
        tmp_path / "source.h5ad",
        url="https://example.invalid/source.h5ad",
        expectedBytes=len(payload),
        chunkBytes=1000,
        opener=opener,
        retryDelaySeconds=0.0,
    )

    assert requests == [0, 3000]
    assert (tmp_path / "source.h5ad").read_bytes() == payload
    assert result.fileBytes == len(payload)
    assert [path.name for path in tmp_path.iterdir()] == ["source.h5ad"]


def test_download_source_rejects_a_server_that_ignores_the_range(
    tmp_path: Path,
) -> None:
    import pytest

    payload = b"x" * 5000

    def opener(_url: str, offset: int) -> _Response:
        response = _Response(
            payload,
            status=200,
            headers={"Content-Length": str(len(payload))},
        )
        if offset == 0:
            response.payload = payload[:2000]
            response.stall = True
        return response

    with pytest.raises(ValueError, match="did not resume at byte 2000"):
        download_source(
            tmp_path / "source.h5ad",
            url="https://example.invalid/source.h5ad",
            expectedBytes=len(payload),
            chunkBytes=1000,
            opener=opener,
            retryDelaySeconds=0.0,
        )
    assert list(tmp_path.iterdir()) == []
