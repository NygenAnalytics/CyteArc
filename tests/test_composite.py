import hashlib
import shutil
import tracemalloc

import numpy as np
import pytest
import zarr
from zarr.storage import FsspecStore, LocalStore

from cytearc import DataStore
from cytearc.composite import create_composite
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts


def _write_source(
    path, values, features, *, policy, ids=None, assay="RNA", workspace=None
):
    root = zarr.open_group(str(path), mode="w")
    ids = np.asarray([f"c{row}" for row in range(len(values))] if ids is None else ids)
    features = np.asarray(features)
    create_cell_data(root, workspace, ids=ids, names=ids)
    counts = create_zarr_count_assay(
        root,
        assay,
        workspace,
        len(ids),
        features,
        features,
        values.dtype,
        policy=policy,
    )
    counts[:] = values
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, assay, workspace, nthreads=1, mem_budget="64M")
    return DataStore(
        str(path),
        workspace=workspace,
        min_features_per_cell=-1,
        nthreads=1,
        mem_budget="64M",
    )


def _tree_digest(path):
    digest = hashlib.sha256()
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            digest.update(str(entry.relative_to(path)).encode())
            digest.update(entry.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def composite_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("composite_sources")
    sources = {}
    for name, n_cells, features, dtype, policy in (
        (
            "a",
            13,
            ["g4", "g0", "g8", "g2", "g6", "g1", "g7", "g3", "g5"],
            np.uint8,
            CountMatrixPolicy(128, 32),
        ),
        (
            "b",
            11,
            ["g9", "g7", "g2", "g0", "g5", "g6", "g3", "g4"],
            np.uint16,
            CountMatrixPolicy(192, 48),
        ),
    ):
        values = (
            np.arange(n_cells * len(features)).reshape(n_cells, len(features)) % 11
        ).astype(dtype)
        path = root / f"{name}.zarr"
        source = _write_source(path, values, features, policy=policy)
        source.cells.insert("quality", np.arange(n_cells), key=None)
        source.cells.update_key(np.arange(n_cells) % 2 == 0, "I")
        sources[name] = path, values, features
    return sources


@pytest.fixture
def sources(composite_sources, tmp_path):
    opened = {}
    for name, (source, _, _) in composite_sources.items():
        path = tmp_path / f"{name}.zarr"
        shutil.copytree(source, path)
        opened[name] = DataStore(str(path), nthreads=1, mem_budget="64M")
    return opened


def _expected_counts(joint, sources):
    features = joint.RNA.feats.fetch_all("ids").tolist()
    expected = np.zeros((joint.cells.N, len(features)), dtype=np.uint16)
    for dest, identity in enumerate(joint.cells.fetch_all("ids")):
        name, cell = str(identity).split("__")
        _, values, source_features = sources[name]
        row = int(cell.removeprefix("c"))
        for column, feature in enumerate(features):
            if feature in source_features:
                expected[dest, column] = values[row, source_features.index(feature)]
    return expected


@pytest.mark.parametrize("features", ["union", "intersection"])
def test_selected_composite_matches_values_and_reopens(
    sources, composite_sources, tmp_path, features
):
    before = {name: _tree_digest(tmp_path / f"{name}.zarr") for name in sources}
    selected = np.array([12, 0, 7, 4])
    joint = create_composite(
        sources,
        at=tmp_path / "joint.zarr",
        features=features,
        rows={"a": selected},
        mem_budget="64M",
        nthreads=1,
    )
    expected = _expected_counts(joint, composite_sources)
    assert set(joint.cells.fetch_all("ids")) == {
        *(f"a__c{row}" for row in selected),
        *(f"b__c{row}" for row in range(11)),
    }
    feature_sets = [set(value[2]) for value in composite_sources.values()]
    expected_features = (
        set.union(*feature_sets)
        if features == "union"
        else set.intersection(*feature_sets)
    )
    assert set(joint.RNA.feats.fetch_all("ids")) == expected_features
    assert joint.RNA._artifact_root is joint.z
    assert joint.RNA._artifact_root.attrs["composite"]["complete"] is True
    assert not (tmp_path / "joint.zarr/RNA/counts/zarr.json").exists()
    assert not (tmp_path / "joint.zarr/RNA/countsT/zarr.json").exists()
    with pytest.raises(PermissionError, match="counts are immutable"):
        joint.RNA.matrixGroup["counts"][0, 0] = 42
    with pytest.raises(PermissionError, match="mappings are immutable"):
        joint.z["_composite/sourceRows"][0] = 0
    np.testing.assert_array_equal(joint.RNA.rawData.compute(), expected)
    np.testing.assert_array_equal(
        joint.cells.fetch_all("RNA_nCounts"), expected.sum(axis=1)
    )
    np.testing.assert_array_equal(joint.cells.fetch_all("I"), True)
    for identity, dataset, quality in zip(
        joint.cells.fetch_all("ids"),
        joint.cells.fetch_all("dataset"),
        joint.cells.fetch_all("orig_quality"),
        strict=True,
    ):
        name, row = identity.split("__c")
        assert dataset == name
        assert quality == int(row)
    assert {name: _tree_digest(tmp_path / f"{name}.zarr") for name in sources} == before

    selected[:] = 1
    for source in sources.values():
        source.cells.reset_key("I")
    reopened = DataStore(str(tmp_path / "joint.zarr"), nthreads=1, mem_budget="64M")
    np.testing.assert_array_equal(
        reopened.cells.fetch_all("ids"), joint.cells.fetch_all("ids")
    )
    np.testing.assert_array_equal(reopened.RNA.rawData.compute(), expected)
    for array, reference in (
        (reopened.RNA.matrixGroup["counts"], expected),
        (reopened.RNA.matrixGroup["countsT"], expected.T),
    ):
        np.testing.assert_array_equal(array[:], reference)
        rows = np.array([reference.shape[0] - 1, 0, 2, 0])
        columns = np.array([reference.shape[1] - 1, 1, 0, 1])
        np.testing.assert_array_equal(
            array.oindex[rows, columns], reference[np.ix_(rows, columns)]
        )
        np.testing.assert_array_equal(array[-1:, -1:], reference[-1:, -1:])
        np.testing.assert_array_equal(array.oindex[[0], [0]], reference[:1, :1])


@pytest.mark.parametrize("features", ["union", "intersection"])
def test_disjoint_feature_composite(sources, tmp_path, features):
    values = np.arange(12, dtype=np.uint16).reshape(4, 3)
    other_features = ["other0", "other1", "other2"]
    other = _write_source(
        tmp_path / "other.zarr",
        values,
        other_features,
        policy=CountMatrixPolicy(128, 32),
    )
    selected_sources = {"a": sources["a"], "other": other}
    destination = tmp_path / "joint.zarr"
    if features == "intersection":
        with pytest.raises(ValueError, match="composite feature space is empty"):
            create_composite(
                selected_sources,
                at=destination,
                features=features,
                nthreads=1,
            )
        assert not destination.exists()
        return
    joint = create_composite(
        selected_sources,
        at=destination,
        features=features,
        rows={"a": [12, 0], "other": [3, 1]},
        nthreads=1,
    )
    assert set(joint.RNA.feats.fetch_all("ids")) == {
        *sources["a"].RNA.feats.fetch_all("ids"),
        *other_features,
    }
    expected = np.zeros(joint.RNA.rawData.shape, dtype=np.uint16)
    feature_ids = joint.RNA.feats.fetch_all("ids").tolist()
    for row, identity in enumerate(joint.cells.fetch_all("ids")):
        name, cell = str(identity).split("__c")
        source = selected_sources[name].RNA
        columns = [
            feature_ids.index(feature) for feature in source.feats.fetch_all("ids")
        ]
        expected[row, columns] = source.matrixGroup["counts"][int(cell), :]
    for current in (joint, DataStore(str(destination), nthreads=1, mem_budget="64M")):
        np.testing.assert_array_equal(current.RNA.matrixGroup["counts"][:], expected)
        np.testing.assert_array_equal(current.RNA.matrixGroup["countsT"][:], expected.T)


def test_composite_rejects_internal_assay_name_before_creating_destination(tmp_path):
    source = _write_source(
        tmp_path / "source.zarr",
        np.arange(8, dtype=np.uint16).reshape(4, 2),
        ["g0", "g1"],
        policy=CountMatrixPolicy(128, 32),
        assay="_composite",
    )
    destination = tmp_path / "joint.zarr"
    with pytest.raises(ValueError, match="'_composite' is reserved"):
        create_composite(
            {"source": source},
            at=destination,
            assay="_composite",
            features="union",
        )
    assert not destination.exists()


@pytest.mark.parametrize("features", ["union", "intersection"])
def test_empty_source_selection_still_participates_in_features(
    sources, composite_sources, tmp_path, features
):
    joint = create_composite(
        sources,
        at=tmp_path / "joint.zarr",
        features=features,
        rows={"a": [12], "b": []},
        nthreads=1,
    )
    np.testing.assert_array_equal(joint.cells.fetch_all("ids"), ["a__c12"])
    assert ("g9" in joint.RNA.feats.fetch_all("ids")) is (features == "union")
    assert ("g8" in joint.RNA.feats.fetch_all("ids")) is (features == "union")
    expected = _expected_counts(joint, composite_sources)
    np.testing.assert_array_equal(joint.RNA.matrixGroup["counts"][:], expected)
    np.testing.assert_array_equal(joint.RNA.matrixGroup["countsT"][:], expected.T)


@pytest.mark.parametrize(
    ("assay", "workspace"),
    [("RNA", "experiment"), ("ADT", None), ("ADT", "experiment")],
)
def test_workspace_and_non_rna_sources_reopen(tmp_path, assay, workspace):
    values = (np.arange(45) % 11).astype(np.uint16).reshape(9, 5)
    features = ["g4", "g0", "g3", "g1", "g2"]
    source = _write_source(
        tmp_path / "source.zarr",
        values,
        features,
        policy=CountMatrixPolicy(128, 32),
        assay=assay,
        workspace=workspace,
    )
    path = tmp_path / "joint.zarr"
    create_composite(
        {"a": source},
        at=path,
        assay=assay,
        features="union",
        rows={"a": [8, 0, 3]},
        nthreads=1,
        mem_budget="64M",
    )
    reopened = DataStore(str(path), min_features_per_cell=-1, nthreads=1)
    output = reopened.get_assay(assay)
    rows = [
        int(identity.split("__c")[1]) for identity in reopened.cells.fetch_all("ids")
    ]
    columns = [features.index(feature) for feature in output.feats.fetch_all("ids")]
    expected = values[np.ix_(rows, columns)]
    assert sorted(rows) == [0, 3, 8]
    assert reopened.z.attrs["composite"]["sources"][0]["workspace"] == workspace
    np.testing.assert_array_equal(output.matrixGroup["counts"][:], expected)
    assert ("countsT" in output.matrixGroup) is (assay == "RNA")
    if assay == "RNA":
        np.testing.assert_array_equal(output.matrixGroup["countsT"][:], expected.T)


def test_remote_source_credentials_are_runtime_only(
    sources, composite_sources, tmp_path, monkeypatch
):
    uri = "hf://buckets/composite-test/source/data.zarr"
    credentials = {"token": "composite-test-token"}
    from_url = FsspecStore.from_url

    def local_source(cls, url, *, storage_options=None, read_only=False):
        assert url == uri
        if storage_options != credentials:
            raise PermissionError("source credentials required")
        return from_url(str(tmp_path / "a.zarr"), read_only=read_only)

    monkeypatch.setattr(FsspecStore, "from_url", classmethod(local_source))
    remote = DataStore(
        uri, zarr_mode="r", storage_options=credentials, nthreads=1, mem_budget="64M"
    )
    path = tmp_path / "joint.zarr"
    joint = create_composite(
        {"a": remote, "b": sources["b"]},
        at=path,
        features="union",
        rows={"a": [0, 7, 12], "b": [3, 5]},
        nthreads=1,
        mem_budget="64M",
    )
    assert joint.z.attrs["composite"]["sources"][0]["location"] == uri
    for metadata in path.rglob("zarr.json"):
        assert credentials["token"].encode() not in metadata.read_bytes()
    for options in (None, {"a": {"token": "wrong-token"}}):
        with pytest.raises(PermissionError, match="source credentials required"):
            DataStore(str(path), source_storage_options=options, nthreads=1)
    reopened = DataStore(
        str(path), source_storage_options={"a": credentials}, nthreads=1
    )
    expected = _expected_counts(reopened, composite_sources)
    np.testing.assert_array_equal(reopened.RNA.matrixGroup["counts"][:], expected)
    np.testing.assert_array_equal(reopened.RNA.matrixGroup["countsT"][:], expected.T)
    with pytest.raises(ValueError, match="Unknown composite source options"):
        DataStore(
            str(path),
            source_storage_options={"a": credentials, "unknown": {}},
            nthreads=1,
        )


@pytest.mark.parametrize("writer", ["subset", "merge"])
def test_existing_writers_materialize_composite_counts(
    sources, composite_sources, tmp_path, writer
):
    from cytearc.merge import DataStoreMerge
    from cytearc.writers.subset import SubsetZarr

    joint = create_composite(
        sources,
        at=tmp_path / "joint.zarr",
        features="union",
        rows={"a": [0, 4, 8], "b": [1, 4]},
        nthreads=1,
        mem_budget="64M",
    )
    expected = _expected_counts(joint, composite_sources)
    identities = joint.cells.fetch_all("ids")
    path = tmp_path / "materialized.zarr"
    options = {"nthreads": 1, "mem_budget": "64M", "policy": CountMatrixPolicy(128, 32)}
    if writer == "subset":
        selected = np.array([0, 2, 4])
        SubsetZarr(str(path), [joint.RNA], cell_idx=selected, **options).dump()
        reference = dict(zip(identities[selected], expected[selected], strict=True))
    else:
        DataStoreMerge([joint, joint], str(path), ["left", "right"], **options).dump()
        reference = {
            f"{name}__{identity}": row
            for name in ("left", "right")
            for identity, row in zip(identities, expected, strict=True)
        }
    result = DataStore(str(path), min_features_per_cell=-1, nthreads=1)
    assert "composite" not in result.z.attrs
    rows = np.array([reference[identity] for identity in result.cells.fetch_all("ids")])
    np.testing.assert_array_equal(
        result.RNA.feats.fetch_all("ids"), joint.RNA.feats.fetch_all("ids")
    )
    np.testing.assert_array_equal(result.RNA.matrixGroup["counts"][:], rows)
    np.testing.assert_array_equal(result.RNA.matrixGroup["countsT"][:], rows.T)
    shutil.rmtree(tmp_path / "a.zarr")
    shutil.rmtree(tmp_path / "b.zarr")
    reopened = DataStore(str(path), min_features_per_cell=-1, nthreads=1)
    np.testing.assert_array_equal(reopened.RNA.rawData.compute(), rows)


@pytest.mark.parametrize("mapping", ["rowSources", "sourceRows", "featureColumns"])
def test_reopen_rejects_changed_mapping(sources, tmp_path, mapping):
    path = tmp_path / "joint.zarr"
    create_composite(sources, at=path, features="union", rows={"a": [0], "b": [0]})
    root = zarr.open_group(str(path), mode="r+")
    array = root[f"_composite/{mapping}"]
    position = (0,) * array.ndim
    array[position] = array[position] + 1
    with pytest.raises(ValueError, match=f"Composite mapping was modified: {mapping}"):
        DataStore(str(path))


@pytest.mark.parametrize("failure", ["missing", "identity"])
def test_reopen_rejects_unavailable_or_changed_source(sources, tmp_path, failure):
    path = tmp_path / "joint.zarr"
    create_composite(sources, at=path, features="intersection")
    if failure == "missing":
        (tmp_path / "a.zarr").rename(tmp_path / "missing.zarr")
        with pytest.raises(FileNotFoundError):
            DataStore(str(path))
    else:
        root = zarr.open_group(str(tmp_path / "a.zarr"), mode="r+")
        root["RNA"].attrs["dataset_fingerprint"] = "changed-source"
        with pytest.raises(ValueError, match="Composite source identity changed: a"):
            DataStore(str(path))


def test_source_chunk_failures_propagate(sources, tmp_path, monkeypatch):
    joint = create_composite(sources, at=tmp_path / "joint.zarr", features="union")
    get = LocalStore.get

    async def read(store, key, *args, **kwargs):
        if store.root == tmp_path / "a.zarr" and key.startswith("RNA/countsT/c/"):
            raise OSError("source read failed")
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", read)
    with pytest.raises(OSError, match="source read failed"):
        joint.RNA.matrixGroup["countsT"][:]


def test_complete_virtual_shards_read_each_source_chunk_once(tmp_path, monkeypatch):
    from collections import Counter
    from itertools import product

    from zarr.core.array import AsyncArray
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    features = [f"g{index}" for index in range(401)]
    values = {
        "a": np.arange(403 * 401, dtype=np.uint32).reshape(403, 401),
        "b": (np.arange(397 * 401, dtype=np.uint32) + 1_000_000).reshape(397, 401),
    }
    feature_positions = {"a": np.arange(401), "b": np.arange(400, -1, -1)}
    sources = {
        name: _write_source(
            tmp_path / name,
            matrix,
            np.asarray(features)[feature_positions[name]],
            policy=CountMatrixPolicy(131_072, 16_384),
        )
        for name, matrix in values.items()
    }
    joint = create_composite(
        sources,
        at=tmp_path / "joint.zarr",
        features="union",
        mem_budget="4M",
        nthreads=1,
    )
    np.testing.assert_array_equal(joint.RNA.feats.fetch_all("ids"), features)
    identities = [str(value).split("__c") for value in joint.cells.fetch_all("ids")]
    expected = np.asarray(
        [values[name][int(row), feature_positions[name]] for name, row in identities]
    )
    source_roots = {source.z.store.root: name for name, source in sources.items()}
    reads = []
    original_getitem = AsyncArray.getitem

    async def record_read(array, selection, *args, **kwargs):
        owner = source_roots.get(getattr(array.store, "root", None))
        if owner is not None and array.path in {"RNA/counts", "RNA/countsT"}:
            assert isinstance(selection, tuple) and len(selection) == 2
            bins = []
            for part, chunk, extent in zip(
                selection, array.chunks, array.shape, strict=True
            ):
                assert isinstance(part, slice) and part.step in {None, 1}
                assert 0 <= part.start < part.stop <= extent
                assert part.start // chunk == (part.stop - 1) // chunk
                bins.append(part.start // chunk)
            reads.append((owner, *bins))
        return await original_getitem(array, selection, *args, **kwargs)

    monkeypatch.setattr(AsyncArray, "getitem", record_read)
    for name, reference in (("counts", expected), ("countsT", expected.T)):
        array = joint.RNA.matrixGroup[name]
        chunks, shards = array.chunks, array.shards
        grid = tuple(s // c for s, c in zip(shards, chunks, strict=True))
        assert np.prod(grid) > 1
        assert any(n % s for n, s in zip(array.shape, shards, strict=True))
        cell_axis, feature_axis = (0, 1) if name == "counts" else (1, 0)
        for coordinates in product(
            *(range((n + s - 1) // s) for n, s in zip(array.shape, shards, strict=True))
        ):
            lower = tuple(i * s for i, s in zip(coordinates, shards, strict=True))
            upper = tuple(
                min(lo + s, n)
                for lo, s, n in zip(lower, shards, array.shape, strict=True)
            )
            cell_window = identities[lower[cell_axis] : upper[cell_axis]]
            feature_window = slice(lower[feature_axis], upper[feature_axis])
            expected_reads = set()
            for owner, source in sources.items():
                rows = [int(row) for member, row in cell_window if member == owner]
                columns = feature_positions[owner][feature_window]
                first, second = (rows, columns) if name == "counts" else (columns, rows)
                source_chunks = source.RNA.matrixGroup[name].chunks
                expected_reads.update(
                    (owner, first_bin, second_bin)
                    for first_bin, second_bin in product(
                        {int(index) // source_chunks[0] for index in first},
                        {int(index) // source_chunks[1] for index in second},
                    )
                )
            reads.clear()
            key = f"RNA/{name}/c/{coordinates[0]}/{coordinates[1]}"
            encoded = sync(joint.z.store.get(key, default_buffer_prototype()))
            assert encoded is not None
            data_bytes = int(np.prod(shards)) * array.dtype.itemsize
            decoded = (
                np.frombuffer(encoded.to_bytes()[:data_bytes], dtype=array.dtype)
                .reshape(*grid, *chunks)
                .transpose(0, 2, 1, 3)
                .reshape(shards)
            )
            padded = np.zeros(shards, dtype=array.dtype)
            padded[
                tuple(slice(0, hi - lo) for lo, hi in zip(lower, upper, strict=True))
            ] = reference[
                tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
            ]
            np.testing.assert_array_equal(decoded, padded)
            assert Counter(reads) == Counter({chunk: 1 for chunk in expected_reads})


def test_failed_preparation_leaves_an_incomplete_destination(sources, tmp_path):
    path = tmp_path / "joint.zarr"

    def failed(*args, **kwargs):
        raise RuntimeError("preparation failed")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DataStore, "_ini_cell_props", failed)
        with pytest.raises(RuntimeError, match="preparation failed"):
            create_composite(sources, at=path, features="intersection")
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["composite"]["complete"] is False
    with pytest.raises(ValueError, match="Composite creation is incomplete"):
        DataStore(str(path))


def test_one_cell_selection_scans_source_metadata_in_bands(
    sources, tmp_path, monkeypatch
):
    import cytearc.merge.metadata as metadata

    blocks = []
    read = metadata.iter_metadata_column_blocks

    def recorded(table, column, **kwargs):
        for values in read(table, column, **kwargs):
            if table is sources["a"].cells and column == "ids":
                blocks.append(values.copy())
            yield values

    monkeypatch.setattr(metadata, "iter_metadata_column_blocks", recorded)
    joint = create_composite(
        {"a": sources["a"]},
        at=tmp_path / "joint.zarr",
        features="intersection",
        rows={"a": [12]},
    )
    np.testing.assert_array_equal(joint.cells.fetch_all("ids"), ["a__c12"])
    assert blocks
    assert len(blocks) < sources["a"].cells.N
    np.testing.assert_array_equal(
        np.concatenate(blocks), sources["a"].cells.fetch_all("ids")
    )


def test_small_selection_is_admitted_without_full_source_summaries(tmp_path):
    values = np.arange(300_000 * 4, dtype=np.uint8).reshape(300_000, 4) % 11
    source = _write_source(
        tmp_path / "source.zarr",
        values,
        ["g0", "g1", "g2", "g3"],
        policy=CountMatrixPolicy(65_536, 16_384),
    )
    selected = np.arange(1024)
    path = tmp_path / "joint.zarr"
    joint = create_composite(
        {"a": source},
        at=path,
        features="intersection",
        rows={"a": selected},
        mem_budget="8M",
        nthreads=1,
    )
    reopened = DataStore(str(path), mem_budget="8M", nthreads=1)
    order = np.array(
        [int(value.split("__c")[1]) for value in joint.cells.fetch_all("ids")]
    )
    np.testing.assert_array_equal(np.sort(order), selected)
    np.testing.assert_array_equal(reopened.RNA.rawData.compute(), values[order])
    np.testing.assert_array_equal(
        reopened.RNA.matrixGroup["countsT"][:], values[order].T
    )


def test_creation_and_virtual_reads_fit_measured_buffer_budget(tmp_path):
    from cytearc.storage.geometry import array_geometry

    values = np.random.default_rng(29).integers(
        0, 30, size=(1024, 1024), dtype=np.uint16
    )
    source = _write_source(
        tmp_path / "source.zarr",
        values,
        [f"g{column:04d}" for column in range(values.shape[1])],
        policy=CountMatrixPolicy(4 * 1024**2, 2 * 1024**2),
    )
    tracemalloc.start()
    try:
        joint = create_composite(
            {"a": source},
            at=tmp_path / "joint.zarr",
            features="intersection",
            mem_budget="16M",
            nthreads=1,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= joint.memoryBytes
    row_order = np.array(
        [int(value.split("__c")[1]) for value in joint.cells.fetch_all("ids")]
    )
    for name in ("counts", "countsT"):
        array = joint.RNA.matrixGroup[name]
        geometry = array_geometry(array)
        chunks = geometry.shards[1] // geometry.chunks[1]
        planned = geometry.readBytes(array.shape[1] * array.dtype.itemsize, chunks)
        array[:1, :]
        tracemalloc.start()
        try:
            actual = array[:1, :]
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak <= planned
        expected = (
            values[row_order[:1]] if name == "counts" else values[row_order, :1].T
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation", ["scoring", "doublets", "mapping"])
def test_composite_consumers_admit_source_scratch_before_count_reads(
    tmp_path, monkeypatch, operation
):
    from cytearc.assay import norm_lib_size
    from cytearc.composite.store import CompositeStore
    from cytearc.mapping.features import AlignedFeatureStream
    from cytearc.quality_control.doublets import _load_parent_counts
    from cytearc.storage.artifacts import callable_identity

    values = np.random.default_rng(123).integers(
        1, 30, size=(1024, 1024), dtype=np.uint16
    )
    source = _write_source(
        tmp_path / "source.zarr",
        values,
        [f"g{column:04d}" for column in range(values.shape[1])],
        policy=CountMatrixPolicy(4 * 1024**2, 2 * 1024**2),
    )
    destination = tmp_path / "joint.zarr"
    joint = create_composite(
        {"a": source},
        at=destination,
        features="intersection",
        mem_budget="32M",
        nthreads=1,
    )
    order = np.array(
        [int(value.split("__c")[1]) for value in joint.cells.fetch_all("ids")]
    )
    rows = np.arange(len(values))
    columns = np.array([0])
    scalar = np.ones(len(rows))

    def read(current):
        if operation == "scoring":
            return current.RNA._mean_normed_union(
                rows,
                scalar,
                columns,
                {"score": columns},
                sf=1,
                log_transform=False,
                resident_bytes=rows.nbytes + scalar.nbytes + columns.nbytes,
            )["score"]
        if operation == "doublets":
            return _load_parent_counts(
                current.RNA.rawData, rows[:1], current.resources, 0
            ).toarray()
        stream = AlignedFeatureStream(
            current.RNA,
            rows,
            np.array(["g0000"]),
            np.zeros(1),
            {
                "normalization_method": callable_identity(norm_lib_size),
                "size_factor": 1,
                "log_transform": False,
                "renormalize_subset": False,
            },
            "error",
            current.resources,
        )
        return np.concatenate([block.values[:, 0] for block in stream.iter_blocks()])

    requests = []
    original_get = CompositeStore.get

    async def recorded_get(self, key, *args, **kwargs):
        if key.startswith("RNA/counts/c/"):
            requests.append(key)
        return await original_get(self, key, *args, **kwargs)

    monkeypatch.setattr(CompositeStore, "get", recorded_get)
    limited = DataStore(str(destination), mem_budget="4M", nthreads=1)
    with pytest.raises(MemoryError):
        read(limited)
    assert requests == []

    read(joint)
    tracemalloc.start()
    try:
        actual = read(joint)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert requests
    assert peak <= joint.memoryBytes
    expected = values[order, 0]
    if operation == "doublets":
        expected = values[order[:1]]
    elif operation == "mapping":
        expected = expected / values[order].sum(axis=1)
    np.testing.assert_allclose(actual, expected)


def test_concurrent_count_reads_fit_measured_buffer_budget(tmp_path):
    from cytearc.storage.execution import execution_report_scope
    from cytearc.storage.feature_stream import map_feature_read_groups

    rng = np.random.default_rng(8)
    values = rng.integers(1, 31, (4096, 2048), dtype=np.uint16)
    features = np.array([f"g{column:05d}" for column in range(values.shape[1])])
    sources = {}
    for name, order in (("a", np.arange(2048)), ("b", rng.permutation(2048))):
        sources[name] = _write_source(
            tmp_path / f"{name}.zarr",
            values[:, order],
            features[order],
            policy=CountMatrixPolicy(8 * 1024**2, 2 * 1024**2),
        )
    joint = create_composite(
        sources,
        at=tmp_path / "joint.zarr",
        features="intersection",
        mem_budget="48M",
        nthreads=2,
    )
    row_order = np.array(
        [int(value.split("__c")[1]) for value in joint.cells.fetch_all("ids")]
    )
    row_sums = values.sum(axis=1)
    feature_sums = 2 * values.sum(axis=0)
    selected = np.sort(rng.choice(len(row_order), size=2000, replace=False))

    def check_rows(array, order):
        start = 0
        for block in array.stream_blocks():
            stop = start + len(block)
            np.testing.assert_array_equal(
                block.sum(axis=1), row_sums[order[start:stop]]
            )
            start = stop
        assert start == len(order)

    def check_features(group):
        np.testing.assert_array_equal(
            group.values.sum(axis=1), feature_sums[group.featStart : group.featEnd]
        )

    def read_features():
        list(
            map_feature_read_groups(
                joint.RNA.matrixGroup["countsT"],
                check_features,
                resources=joint.resources,
            )
        )

    for read in (
        lambda: check_rows(joint.RNA.rawData, row_order),
        read_features,
        lambda: check_rows(joint.RNA.rawData[selected], row_order[selected]),
    ):
        read()
        with execution_report_scope() as reports:
            tracemalloc.start()
            try:
                read()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        assert peak <= joint.memoryBytes
        if read is read_features:
            assert any(report.as_metrics()["innerReads"] > 1 for report in reports)


def test_budget_fitted_virtual_chunks_stream_counts_and_transpose(tmp_path):
    from cytearc.storage.feature_stream import (
        map_feature_read_groups,
        read_group_stream_floor,
    )

    values = (np.arange(2048 * 512) % 19).astype(np.uint16).reshape(2048, 512)
    source = _write_source(
        tmp_path / "source.zarr",
        values,
        [f"g{column}" for column in range(512)],
        policy=CountMatrixPolicy(32_768, 4096),
    )
    path = tmp_path / "joint.zarr"
    create_composite(
        {"a": source},
        at=path,
        features="intersection",
        mem_budget="8M",
        nthreads=1,
    )
    joint = DataStore(str(path), mem_budget="8M", nthreads=1)
    row_order = np.array(
        [int(value.split("__c")[1]) for value in joint.cells.fetch_all("ids")]
    )
    feature_order = np.array(
        [int(value.removeprefix("g")) for value in joint.RNA.feats.fetch_all("ids")]
    )
    counts = joint.RNA.rawData
    stored_counts = joint.RNA.matrixGroup["counts"]
    counts_t = joint.RNA.matrixGroup["countsT"]
    assert any(
        chunk < extent
        for chunk, extent in zip(stored_counts.chunks, values.shape, strict=True)
    )
    assert counts._block_task_bytes() <= joint.memoryBytes
    assert read_group_stream_floor(counts_t) <= joint.memoryBytes
    start = 0
    for block in counts.stream_blocks():
        stop = start + len(block)
        np.testing.assert_array_equal(
            block, values[np.ix_(row_order[start:stop], feature_order)]
        )
        start = stop
    assert start == len(values)

    def check_group(group):
        np.testing.assert_array_equal(
            group.values,
            values[np.ix_(row_order, feature_order[group.featStart : group.featEnd])].T,
        )
        return group.featEnd - group.featStart

    assert (
        sum(map_feature_read_groups(counts_t, check_group, resources=joint.resources))
        == values.shape[1]
    )
    rows = np.array([2047, 0, min(2047, stored_counts.chunks[0])])
    features = np.array([511, 0, min(511, stored_counts.chunks[1])])
    np.testing.assert_array_equal(
        counts_t.oindex[features, rows],
        values[np.ix_(row_order[rows], feature_order[features])].T,
    )


def test_pipeline_matches_materialized_counts_and_reopens(tmp_path):
    rng = np.random.default_rng(72)
    features = np.array([f"g{column}" for column in range(64)])
    sources = {}
    reference_sources = {}
    for index, name in enumerate(("a", "b")):
        order = rng.permutation(len(features))
        values = rng.poisson(rng.uniform(0.5, 4, len(features)), size=(64, 64)).astype(
            np.uint16
        )
        path = tmp_path / f"{name}.zarr"
        sources[name] = _write_source(
            path,
            values,
            features[order],
            policy=CountMatrixPolicy(4096, 1024 + index * 1024),
        )
        reference_sources[name] = path, values, features[order].tolist()
    path = tmp_path / "joint.zarr"
    joint = create_composite(sources, at=path, features="intersection", nthreads=1)
    reference = _write_source(
        tmp_path / "reference.zarr",
        _expected_counts(joint, reference_sources),
        joint.RNA.feats.fetch_all("ids"),
        ids=joint.cells.fetch_all("ids"),
        policy=CountMatrixPolicy(4096, 1024),
    )
    params = {
        "filtering": False,
        "cell_cycle": False,
        "hvg": {"top_n": 40},
        "pca": {"dims": 3},
        "neighbors": {"k": 5},
        "umap": False,
        "leiden": False,
        "paris": False,
        "doublets": False,
        "markers": False,
    }
    actual_run = joint.pipeline.run(params=params)
    expected_run = reference.pipeline.run(params=params)
    assert actual_run.status == expected_run.status == "completed"
    assert list(actual_run) == list(expected_run)
    for name in actual_run:
        actual = joint.artifacts.inspect(actual_run[name])
        expected = reference.artifacts.inspect(expected_run[name])
        assert actual.complete and expected.complete
        actual_group = joint.z[actual.path]
        expected_group = reference.z[expected.path]
        arrays = sorted(actual_group.array_keys())
        assert arrays
        assert arrays == sorted(expected_group.array_keys())
        for key in arrays:
            actual_values = actual_group[key][:]
            expected_values = expected_group[key][:]
            if actual_values.dtype.kind == "f":
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=1e-7,
                    atol=1e-8,
                    err_msg=f"{name}/{key}",
                )
            else:
                np.testing.assert_array_equal(
                    actual_values, expected_values, err_msg=f"{name}/{key}"
                )
    reopened = DataStore(str(path), nthreads=1)
    saved_run = reopened.pipeline.open(run_id=actual_run.run_id)
    assert saved_run.status == "completed"
    assert dict(saved_run) == dict(actual_run)
