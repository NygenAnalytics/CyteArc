import copy
import hashlib
import json
import shutil
import tracemalloc
from contextlib import contextmanager

import numpy as np
import pytest
import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore

from cytearc import DataStore
from cytearc.composite import create_composite
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts


@pytest.fixture
def composite_adapter(sources, tmp_path, monkeypatch):
    import cytearc.composite.create as creation

    monkeypatch.setattr(
        creation, "DEFAULT_COUNT_MATRIX_POLICY", CountMatrixPolicy(48, 8)
    )
    joint = create_composite(
        sources,
        at=tmp_path / "joint.zarr",
        features="union",
        rows={"a": [0, 1, 2, 3], "b": [0, 1, 2]},
    )
    yield joint
    joint.z.store.close()


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


@contextmanager
def count_fallbacks(monkeypatch):
    """Record every Zarr read of composite counts.

    CyteArc reads composite counts through the store's reader; a Zarr read of
    their computed shards is many times slower, so workflows must make none.
    """
    from cytearc.composite.store import CompositeStore

    get = CompositeStore.get
    reads = []

    async def record(self, key, prototype, byte_range=None):
        if self._path(key) is not None and not key.endswith("zarr.json"):
            reads.append(key)
        return await get(self, key, prototype, byte_range)

    with monkeypatch.context() as patch:
        patch.setattr(CompositeStore, "get", record)
        yield reads


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


def test_composite_metadata_handles_preserve_boolean_selection_contract(sources):
    from cytearc.composite.create import _CachedMetadataTable

    table = sources["a"].cells
    handles = _CachedMetadataTable(table)
    np.testing.assert_array_equal(handles._bool_array("I")[:], table.fetch_all("I"))
    assert handles.default_block_rows() == table.default_block_rows()
    with pytest.raises(TypeError, match="must have boolean dtype"):
        handles._bool_array("quality")


def test_composite_schema_prefetch_skips_local_and_empty_tables(
    sources, tmp_path, monkeypatch
):
    from cytearc.composite.create import _CachedMetadataTable, _prefetch_metadata_schema
    from cytearc.composite.source_cache import SourceCache
    from cytearc.metadata.table import MetaData
    from cytearc.storage.budget import ResourceBudget

    root = zarr.open_group(str(tmp_path / "empty.zarr"), mode="w")
    create_cell_data(
        root, None, ids=np.array([], dtype="U1"), names=np.array([], dtype="U1")
    )
    empty = _CachedMetadataTable(MetaData(root["cellData"]))
    local = _CachedMetadataTable(sources["a"].cells)
    cache = SourceCache()
    empty.use_source_cache(cache)
    requests = []
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if "/c/" in key:
            requests.append(key)
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    try:
        _prefetch_metadata_schema(
            [local, empty],
            [frozenset(), frozenset()],
            ResourceBudget(2, 1024**2),
            cache=cache,
            resident_bytes=0,
            read_workers=2,
        )
        assert not requests
        assert not cache._entries
    finally:
        cache.close()


@pytest.mark.parametrize(
    "reason", ("memory", "empty_features", "sparse_shard", "entries")
)
def test_composite_count_prefetch_only_admits_selected_objects(
    tmp_path, monkeypatch, reason
):
    from cytearc.composite.create import _prefetch_counts
    from cytearc.composite.source_cache import SourceCache
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.geometry import array_geometry

    root = zarr.open_group(str(tmp_path / "source.zarr"), mode="w")
    sparse = reason == "sparse_shard"
    shape = (4, 8) if sparse else (64, 64)
    chunks = (2, 2) if sparse else (32, 32)
    shards = (4, 4) if sparse else shape
    for name in ("a", "b"):
        root.create_array(
            name,
            data=np.ones(shape, dtype=np.uint16),
            chunks=chunks,
            shards=shards,
        )
    cache = SourceCache(max_entries=1 if reason == "entries" else 16)
    cached = cache.wrap(root.store)
    arrays = [zarr.open_array(cached, path=name, mode="r") for name in ("a", "b")]
    selections = [np.array([0]), np.array([], dtype=np.int64)] if sparse else None
    plan = build_row_plan(
        [shape[0]] * 2, [shape[0]] * 2, ["a", "b"], selected_rows=selections
    )
    mappings = [np.arange(shape[1]) for _ in arrays]
    if reason == "empty_features":
        mappings = [np.full(shape[1], -1) for _ in arrays]
    elif sparse:
        mappings[0][5:] = -1
    requests = []
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if store.root == root.store.root and "/c/" in key:
            requests.append(key)
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    try:
        _prefetch_counts(
            arrays,
            plan,
            mappings,
            [array_geometry(array) for array in arrays],
            segment_rows=shape[0],
            read_workers=2,
            memory_bytes=4096 if reason == "memory" else 1024**2,
            cache=cache,
        )
        if reason in {"memory", "empty_features"}:
            assert not requests
        elif sparse:
            assert requests == ["a/c/0/0"]
        else:
            first = ("a", "b")[plan.coordinatesPermutations[0, 0]]
            assert requests == [f"{first}/c/0/0"]
            assert len(cache._entries) == 1
            np.testing.assert_array_equal(arrays[("a", "b").index(first)][:], 1)
            assert requests == [f"{first}/c/0/0"]
    finally:
        cache.close()


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
    assert joint.z.store.virtual_array_reader("cellData/I") is None
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


def test_remote_composite_reuses_metadata_table_listings(
    composite_sources, tmp_path, monkeypatch
):
    from huggingface_hub import BucketFile, HfFileSystem

    from cytearc.storage.huggingface import HfReadStore
    from tests.fixtures_hub import FakeHub

    hub = FakeHub(tmp_path / "hub")
    bucket_id = "tests/composite"
    fs = HfFileSystem(token=False, skip_instance_cache=True)
    fs._bucket_exists_cache[bucket_id] = (True, None)

    def list_tree(bucket_id, prefix=None, *, recursive=False):
        entries = hub.list_bucket_tree(bucket_id, prefix, recursive=recursive)
        return [
            entry for entry in entries if not recursive or isinstance(entry, BucketFile)
        ]

    def open_file(path, mode="rb", **options):
        info = fs.info(path)
        resolved = fs.resolve_path(path)
        opened = hub.path(resolved.path, resolved.bucket_id).open(mode)
        opened.size = info["size"]
        return opened

    monkeypatch.setattr(fs._api, "list_bucket_tree", list_tree)
    monkeypatch.setattr(fs, "_open", open_file)
    sources = {}
    joint = None
    try:
        for name, (path, _, _) in composite_sources.items():
            shutil.copytree(path, hub.path(f"{name}.zarr", bucket_id))
            store = HfReadStore.from_mapper(
                fs.get_mapper(f"hf://buckets/{bucket_id}/{name}.zarr"), read_only=True
            )
            sources[name] = DataStore(
                store,
                zarr_mode="r",
                min_features_per_cell=-1,
                nthreads=1,
                mem_budget="64M",
            )
        fs.dircache.clear()
        hub.calls.clear()

        joint = create_composite(
            sources,
            at=tmp_path / "joint.zarr",
            features="union",
            rows={"a": [0, 7, 12], "b": [3, 5]},
            nthreads=1,
            mem_budget="64M",
        )

        expected = _expected_counts(joint, composite_sources)
        np.testing.assert_array_equal(joint.RNA.rawData.compute(), expected)
        np.testing.assert_array_equal(joint.RNA.matrixGroup["countsT"][:], expected.T)
        np.testing.assert_array_equal(
            joint.RNA.feats.fetch_all("names"), joint.RNA.feats.fetch_all("ids")
        )
        for identity, dataset, quality in zip(
            joint.cells.fetch_all("ids"),
            joint.cells.fetch_all("dataset"),
            joint.cells.fetch_all("orig_quality"),
            strict=True,
        ):
            name, row = identity.split("__c")
            assert dataset == name
            assert quality == int(row)

        tables = {
            f"{name}.zarr/{table}"
            for name in sources
            for table in ("cellData", "RNA/featureData")
        }
        for table in tables:
            calls = [
                call
                for call in hub.calls
                if call[2] == table or call[2].startswith(table + "/")
            ]
            recursive = ("list_bucket_tree", bucket_id, table, True, None)
            assert calls.count(recursive) == 1
            assert calls[calls.index(recursive) :] == [recursive]
        assert {call[2] for call in hub.calls if call[3]} == tables
    finally:
        if joint is not None:
            joint.z.store.close()
        for source in sources.values():
            source.z.store.close()


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


def test_composite_rejects_duplicate_source_cell_ids_without_changing_source(tmp_path):
    source_path = tmp_path / "source.zarr"
    source = _write_source(
        source_path,
        np.ones((2, 2), dtype=np.uint8),
        ["g0", "g1"],
        ids=["duplicate", "duplicate"],
        policy=CountMatrixPolicy(128, 32),
    )
    before = _tree_digest(source_path)
    destination = tmp_path / "joint.zarr"
    with pytest.raises(ValueError, match="Composite cell IDs are not unique"):
        create_composite(
            {"source": source},
            at=destination,
            features="union",
            mem_budget="64M",
            nthreads=1,
        )
    assert (
        zarr.open_group(str(destination), mode="r").attrs["composite"]["complete"]
        is False
    )
    assert _tree_digest(source_path) == before
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], 1)


@pytest.mark.parametrize("phase", ("metadata", "identity", "counts"))
def test_composite_creation_rejects_unaffordable_work_before_count_reads(
    tmp_path, monkeypatch, phase
):
    metadata = phase == "metadata"
    identity = phase == "identity"
    shape = (100, 2) if metadata else (150_000, 1) if identity else (4, 256)
    source = _write_source(
        tmp_path / "source.zarr",
        np.ones(shape, dtype=np.uint8),
        [f"g{index}" for index in range(shape[1])],
        ids=[f"{index}_{'x' * 1000}" for index in range(shape[0])]
        if metadata
        else None,
        policy=(
            CountMatrixPolicy(1_048_576, 262_144)
            if identity
            else CountMatrixPolicy(1024, 256)
        ),
    )
    if not identity:
        for table in (source.cells, source.RNA.feats):
            for name, array in list(table._group.arrays()):
                values, attributes = array[:], dict(array.attrs)
                del table._group[name]
                table._group.create_array(
                    name,
                    data=values,
                    chunks=(min(table.N, 2 if metadata else 256),),
                    attributes=attributes,
                )
    requests = []
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if store.root == source.z.store.root and key.startswith("RNA/counts/c/"):
            requests.append(key)
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    destination = tmp_path / "joint.zarr"
    message = {
        "metadata": "Merged cell metadata cannot fit one row",
        "identity": "Composite cell identity validation exceeds mem_budget",
        "counts": "Composite count preparation cannot fit one row",
    }[phase]
    with pytest.raises(MemoryError, match=message):
        create_composite(
            {"source": source},
            at=destination,
            features="union",
            # Large enough to fit the composite layout, too small for one
            # prepared row.
            mem_budget="24M" if identity else 71_000 if phase == "counts" else 65_536,
            nthreads=1,
        )
    assert not requests
    if metadata:
        assert not destination.exists()
    else:
        root = zarr.open_group(str(destination), mode="r")
        assert root.attrs["composite"]["complete"] is False
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], 1)


def test_one_cell_selection_scans_source_metadata_in_bands(
    sources, tmp_path, monkeypatch
):
    import cytearc.merge.metadata as metadata

    blocks = []
    read = metadata.iter_metadata_column_blocks

    def recorded(table, column, **kwargs):
        for values in read(table, column, **kwargs):
            if (
                column == "ids"
                and table._get_array(column).store == sources["a"].z.store
            ):
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
        # Zarr reads the row's shards at once, and each shard request holds its
        # own allowance.
        planned = 0
        for start in range(0, array.shape[1], geometry.shards[1]):
            width = min(geometry.shards[1], array.shape[1] - start)
            planned += geometry.readBytes(
                width * array.dtype.itemsize, -(-width // geometry.chunks[1])
            )
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
    from zarr.core.array import AsyncArray

    from cytearc.assay import norm_lib_size
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
    original_getitem = AsyncArray.getitem
    source_root = str(source.z.store.root)

    async def recorded_getitem(self, selection, *args, **kwargs):
        if str(getattr(self.store, "root", "")) == source_root and self.path in {
            "RNA/counts",
            "RNA/countsT",
        }:
            requests.append((self.path, selection))
        return await original_getitem(self, selection, *args, **kwargs)

    monkeypatch.setattr(AsyncArray, "getitem", recorded_getitem)
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


def test_pipeline_matches_materialized_counts_and_reopens(tmp_path, monkeypatch):
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
    with count_fallbacks(monkeypatch) as fallbacks:
        actual_run = joint.pipeline.run(params=params)
    assert fallbacks == []
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


def test_composite_store_inventory_matches_readable_objects(composite_adapter):
    from itertools import product
    from math import ceil

    joint = composite_adapter
    store = joint.z.store
    prototype = default_buffer_prototype()
    physical = {
        str(path.relative_to(store.root))
        for path in store.root.rglob("*")
        if path.is_file()
    }
    virtual = set()
    for name in ("counts", "countsT"):
        array = joint.RNA.matrixGroup[name]
        virtual.add(f"RNA/{name}/zarr.json")
        grid = [
            range(ceil(n / s)) for n, s in zip(array.shape, array.shards, strict=True)
        ]
        virtual.update(f"RNA/{name}/c/{r}/{c}" for r, c in product(*grid))
    expected = physical | virtual

    async def check():
        keys = [key async for key in store.list()]
        assert len(keys) == len(set(keys))
        assert set(keys) == expected
        for key in keys:
            assert await store.exists(key)
            value = await store.get(key, prototype)
            assert value is not None
            assert await store.getsize(key) == len(value.to_bytes())
        for prefix in ("", "RNA", "RNA/counts/", "RNA/counts/c/", "absent"):
            assert set([key async for key in store.list_prefix(prefix)]) == {
                key for key in expected if key.startswith(prefix)
            }
            assert await store.is_empty(prefix) == (
                not any(key.startswith(prefix) for key in expected)
            )
        for prefix in (
            "",
            "RNA",
            "RNA/counts",
            "RNA/counts/c",
            "RNA/counts/c/0",
            "RNA/counts/c/0/0",
            "RNA/counts/c/999",
            "RNA/counts/c/nope",
        ):
            normalized = prefix.rstrip("/") + "/" if prefix else ""
            expected_children = {
                key[len(normalized) :].split("/", 1)[0]
                for key in expected
                if key.startswith(normalized)
            }
            children = [key async for key in store.list_dir(prefix)]
            assert len(children) == len(set(children))
            assert set(children) == expected_children
        for key in ("RNA/counts/c/999/0", "RNA/counts/c/nope/0", "absent"):
            assert not await store.exists(key)
            assert await store.get(key, prototype) is None
            with pytest.raises(FileNotFoundError):
                await store.getsize(key)

    sync(check())


def test_composite_store_batches_preserve_bytes_and_missing_keys(composite_adapter):
    store = composite_adapter.z.store
    prototype = default_buffer_prototype()
    ranges = [None, RangeByteRequest(2, 9), OffsetByteRequest(5), SuffixByteRequest(7)]

    async def check():
        for key in ("RNA/counts/zarr.json", "RNA/counts/c/0/0", "zarr.json"):
            body = (await store.get(key, prototype)).to_bytes()
            expected = [body, body[2:9], body[5:], body[-7:]]
            partial = await store.get_partial_values(
                prototype, [(key, r) for r in ranges]
            )
            assert [value.to_bytes() for value in partial] == expected
            batches = [
                batch
                async for batch in store.get_ranges(key, ranges, prototype=prototype)
            ]
            actual = {
                index: value.to_bytes() for batch in batches for index, value in batch
            }
            assert actual == dict(enumerate(expected))
        requests = [
            (key, prototype, None)
            for key in ("zarr.json", "RNA/counts/zarr.json", "missing")
        ]
        results = {key: value async for key, value in store._get_many(requests)}
        assert results["missing"] is None
        for key in ("zarr.json", "RNA/counts/zarr.json"):
            assert (
                results[key].to_bytes() == (await store.get(key, prototype)).to_bytes()
            )
        assert await store.get_partial_values(prototype, [("missing", None)]) == [None]
        with pytest.raises(TypeError, match="Unsupported byte request"):
            await store.get("RNA/counts/zarr.json", prototype, object())

    sync(check())


def test_composite_store_mutations_preserve_bindings_and_sources(
    composite_adapter, sources
):
    joint = composite_adapter
    store = joint.z.store
    prototype = default_buffer_prototype()
    before = _tree_digest(store.root)
    source_digests = {
        name: _tree_digest(source.z.store.root) for name, source in sources.items()
    }

    async def check():
        original = (await store.get("zarr.json", prototype)).to_bytes()
        changed = json.loads(original)
        del changed["attributes"]["composite"]
        consolidated = json.loads(original)
        consolidated["consolidated_metadata"] = {}
        for document, message in (
            (changed, "bindings are immutable"),
            (consolidated, "cannot be consolidated"),
        ):
            with pytest.raises(PermissionError, match=message):
                await store.set(
                    "zarr.json",
                    prototype.buffer.from_bytes(json.dumps(document).encode()),
                )
        await store.set("zarr.json", prototype.buffer.from_bytes(original))
        for key in (
            "zarr.json",
            "RNA/counts/zarr.json",
            "_composite/sourceRows/zarr.json",
        ):
            with pytest.raises(PermissionError):
                await store.delete(key)
        for prefix in ("", "RNA", "RNA/counts", "_composite"):
            with pytest.raises(PermissionError):
                await store.delete_dir(prefix)
        with pytest.raises(PermissionError, match="cannot be cleared"):
            await store.clear()
        readonly = store.with_read_only(True)
        try:
            with pytest.raises(PermissionError, match="read-only"):
                await readonly.set("result", prototype.buffer.from_bytes(b"new"))
            with pytest.raises(PermissionError):
                await readonly.delete("result")
            with pytest.raises(PermissionError):
                await readonly.delete_dir("result")
        finally:
            readonly.close()
        assert _tree_digest(store.root) == before
        await store._set_many(
            (key, prototype.buffer.from_bytes(value))
            for key, value in (("result/value", b"one"), ("result/other", b"two"))
        )
        await store.set_if_not_exists(
            "result/value", prototype.buffer.from_bytes(b"replacement")
        )
        assert (await store.get("result/value", prototype)).to_bytes() == b"one"
        await store.delete("result/value")
        assert not await store.exists("result/value")
        await store.delete_dir("result")
        assert await store.is_empty("result")

    sync(check())
    assert _tree_digest(store.root) == before
    assert {
        name: _tree_digest(source.z.store.root) for name, source in sources.items()
    } == source_digests


def test_composite_store_lifecycle_and_routing_contracts(composite_adapter, sources):
    from zarr.abc.store import _store_supports_sync_io

    from cytearc.storage.stores import is_remote_datastore

    joint = composite_adapter
    store = joint.z.store
    assert not _store_supports_sync_io(store)
    for method in (store.get_sync, store.set_sync, store.delete_sync):
        with pytest.raises(TypeError, match="asynchronous routed IO"):
            method()
    with pytest.raises(RuntimeError, match="already finalized"):
        store.seal(joint.z)
    assert not is_remote_datastore(None, joint.RNA.matrixGroup["counts"])
    assert not is_remote_datastore(None, joint.z["cellData"])
    with store as opened:
        np.testing.assert_array_equal(
            zarr.open_array(store=opened, path="RNA/counts", mode="r")[:],
            joint.RNA.matrixGroup["counts"][:],
        )
    assert not store._store._is_open
    for source in sources.values():
        assert source.z.store._is_open
        assert source.RNA.matrixGroup["counts"][0, 0] >= 0


def test_composite_shard_padding_and_missing_array_metadata(composite_adapter, sources):
    from cytearc.composite.store import CompositeStore

    joint = composite_adapter
    prototype = default_buffer_prototype()
    building = CompositeStore(
        LocalStore(joint.z.store.root),
        copy.deepcopy(joint.z.store.record),
        {name: source.z for name, source in sources.items()},
        constructing=True,
    )
    try:
        document = copy.deepcopy(building.record["arrays"]["RNA/counts"])
        document["chunk_grid"]["configuration"]["chunk_shape"] = [16, 16]
        sync(
            building.set(
                "RNA/counts/zarr.json",
                prototype.buffer.from_bytes(json.dumps(document).encode()),
            )
        )
        array = zarr.open_array(store=building, path="RNA/counts", mode="r")
        np.testing.assert_array_equal(array[:], joint.RNA.matrixGroup["counts"][:])
        chunk_bytes = np.prod(array.chunks) * array.dtype.itemsize
        offset = (8 // array.chunks[0]) * (16 // array.chunks[1]) * int(chunk_bytes)
        padding = sync(
            building.get(
                "RNA/counts/c/0/0",
                prototype,
                RangeByteRequest(offset, offset + int(chunk_bytes)),
            )
        )
        assert padding.to_bytes() == bytes(int(chunk_bytes))
        sync(building.delete_dir("RNA/countsT"))
        assert not sync(building.exists("RNA/countsT/zarr.json"))
        with pytest.raises(FileNotFoundError):
            sync(building.getsize("RNA/countsT/zarr.json"))
        sync(building.delete_dir("RNA/counts"))
        building.constructing = False
        with pytest.raises(ValueError, match="array metadata is missing"):
            sync(building.get("RNA/counts/zarr.json", prototype))
    finally:
        building.close()


@pytest.mark.parametrize(
    "problem,message",
    [
        ("record", "Invalid composite record"),
        ("sources", "no source bindings"),
        ("arrays", "Invalid composite array metadata"),
        ("codec", "Invalid composite array codecs"),
        ("layout", "paired sharded layout"),
        ("maps", "Invalid composite mapping fingerprints"),
        ("map_shape", "Invalid composite mapping: rowSources"),
        ("map_dtype", "Invalid composite mapping: rowSources"),
        ("binding", "Invalid composite source binding"),
        ("duplicate", "unique and nonempty"),
        ("empty_name", "unique and nonempty"),
        ("location", "persistent locations"),
        ("mounted", "Mounted datastores cannot be composite sources"),
    ],
)
def test_composite_reopen_rejects_invalid_saved_contract(
    composite_adapter, sources, problem, message
):
    path = composite_adapter.z.store.root
    root = zarr.open_group(path, mode="r+")
    record = copy.deepcopy(root.attrs["composite"])
    if problem == "record":
        record["unexpected"] = True
    elif problem == "sources":
        record["sources"] = []
    elif problem == "arrays":
        record["arrays"] = {}
    elif problem == "codec":
        record["arrays"]["RNA/counts"]["codecs"][0]["configuration"][
            "index_location"
        ] = "start"
    elif problem == "layout":
        record["arrays"]["RNA/counts"]["codecs"][0]["name"] = "bytes"
    elif problem == "maps":
        record["maps"] = {}
    elif problem in {"map_shape", "map_dtype"}:
        mappings = root["_composite"]
        values = mappings["rowSources"][:]
        del mappings["rowSources"]
        mappings.create_array(
            "rowSources",
            data=values[:-1] if problem == "map_shape" else values.astype(np.float64),
        )
    elif problem == "binding":
        record["sources"][1]["unexpected"] = True
    elif problem == "duplicate":
        record["sources"][1]["name"] = record["sources"][0]["name"]
    elif problem == "empty_name":
        record["sources"][1]["name"] = ""
    elif problem == "location":
        record["sources"][1]["location"] = ""
    elif problem == "mounted":
        sources["b"].z.attrs["matrixSource"] = {}
    root.attrs["composite"] = record
    before = _tree_digest(path)
    with pytest.raises(ValueError, match=message):
        DataStore(str(path), nthreads=1)
    assert _tree_digest(path) == before


@pytest.mark.parametrize(
    "problem,error,message",
    [
        ("empty", ValueError, "At least one source"),
        ("features", ValueError, "features must be"),
        ("name", ValueError, "Source names"),
        ("type", TypeError, "open DataStore"),
        ("selection", ValueError, "existing source"),
        ("remote", ValueError, "destination must be a local path"),
        ("existing", FileExistsError, "already exists"),
        ("mounted", ValueError, "Mounted and composite sources"),
        ("assay", ValueError, "has no assay"),
        ("overlap", ValueError, "lies inside the Zarr store"),
        ("overlap_without_root_metadata", ValueError, "must not overlap a source"),
        ("assay_type", ValueError, "different types"),
        ("empty_selection", ValueError, "at least one selected cell"),
        ("budget", MemoryError, "mappings exceed mem_budget"),
    ],
)
def test_composite_invalid_creation_preserves_sources(
    sources, tmp_path, problem, error, message
):
    inputs = sources
    destination = tmp_path / "rejected.zarr"
    options = {"features": "union"}
    if problem == "empty":
        inputs = {}
    elif problem == "features":
        options["features"] = "invalid"
    elif problem == "name":
        inputs = {"bad__name": sources["a"]}
    elif problem == "type":
        inputs = {"a": sources["a"].z}
    elif problem == "selection":
        options["rows"] = {"missing": [0]}
    elif problem == "remote":
        destination = "s3://composite-test/rejected.zarr"
    elif problem == "existing":
        destination.mkdir()
    elif problem == "mounted":
        sources["a"].z.attrs["matrixSource"] = {}
    elif problem == "assay":
        options["assay"] = "missing"
    elif problem in {"overlap", "overlap_without_root_metadata"}:
        destination = sources["a"].z.store.root / "rejected.zarr"
        if problem == "overlap_without_root_metadata":
            (sources["a"].z.store.root / "zarr.json").unlink()
    elif problem == "assay_type":
        inputs = {
            **sources,
            "b": DataStore(
                sources["b"].zarr_loc, assay_types={"RNA": "ATAC"}, nthreads=1
            ),
        }
    elif problem == "empty_selection":
        options["rows"] = {"a": [], "b": []}
    elif problem == "budget":
        options["mem_budget"] = 1
    before = {
        name: _tree_digest(source.z.store.root) for name, source in sources.items()
    }
    with pytest.raises(error, match=message):
        create_composite(inputs, at=destination, **options)
    assert {
        name: _tree_digest(source.z.store.root) for name, source in sources.items()
    } == before
    if problem == "existing":
        assert list(destination.iterdir()) == []
    elif problem != "remote":
        assert not destination.exists()


def test_composite_source_locations_require_persistent_credential_free_paths(
    sources,
):
    from types import SimpleNamespace

    from zarr.storage import MemoryStore, WrapperStore

    from cytearc.composite.create import _source_location

    local = WrapperStore(LocalStore(sources["a"].z.store.root))
    source = SimpleNamespace(zarr_loc=local, z=zarr.open_group(store=local, mode="r"))
    assert _source_location(source) == str(sources["a"].z.store.root.resolve())
    memory = SimpleNamespace(zarr_loc=MemoryStore(), z=zarr.group())
    with pytest.raises(ValueError, match="persistent filesystem location"):
        _source_location(memory)
    for location in (
        "https://user@example.org/counts.zarr",
        "https://example.org/counts.zarr?token=secret",
    ):
        with pytest.raises(
            ValueError, match="cannot contain user information or query parameters"
        ):
            _source_location(SimpleNamespace(zarr_loc=location))


def test_composite_layout_rejects_budget_smaller_than_one_read(sources):
    from cytearc.composite.create import _fit_count_layout

    array = sources["a"].RNA.matrixGroup["counts"]
    with pytest.raises(MemoryError, match="count reads cannot fit within mem_budget"):
        _fit_count_layout(
            *array.shape,
            array.dtype.str,
            {"counts": [array]},
            mapping_width=4,
            memory_bytes=1,
        )


@pytest.mark.parametrize("orientation", ["counts", "countsT"])
def test_reopened_composite_retries_throttled_source_reads(
    composite_sources, tmp_path, monkeypatch, orientation
):
    import httpx2
    from huggingface_hub.errors import HfHubHTTPError

    from cytearc.storage import huggingface

    local_path = str(composite_sources["a"][0])
    source = DataStore(
        huggingface.HfReadStore.from_url(local_path, read_only=True),
        zarr_mode="r",
        min_features_per_cell=-1,
        nthreads=1,
        mem_budget="64M",
    )
    source.zarr_loc = "hf://buckets/test/composite/source.zarr"
    path = tmp_path / "joint.zarr"
    joint = create_composite(
        {"study": source},
        at=path,
        features="intersection",
        rows={"study": [1, 4]},
        nthreads=1,
        mem_budget="64M",
    )
    expected = joint.RNA.matrixGroup[orientation][:]
    from_url = huggingface.HfReadStore.from_url

    def local_source(url, *, storage_options=None, read_only=False):
        assert url == source.zarr_loc
        assert storage_options == {"token": False}
        return from_url(local_path, read_only=read_only)

    monkeypatch.setattr(huggingface.HfReadStore, "from_url", local_source)
    reopened = DataStore(
        str(path),
        nthreads=1,
        mem_budget="64M",
        source_storage_options={"study": {"token": False}},
    )
    store = reopened.z.store._roots["study"].store
    read = store.fs._cat_file
    failed = False
    sleeps = []

    async def throttled(path, *args, **kwargs):
        nonlocal failed
        if f"/RNA/{orientation}/c/" in path and not failed:
            failed = True
            response = httpx2.Response(
                429,
                headers={"Retry-After": "7"},
                request=httpx2.Request("GET", "https://bucket.invalid/object"),
            )
            raise HfHubHTTPError("Rate limited", response=response)
        return await read(path, *args, **kwargs)

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(store.fs, "_cat_file", throttled)
    monkeypatch.setattr(huggingface.asyncio, "sleep", sleep)
    try:
        np.testing.assert_array_equal(
            reopened.RNA.matrixGroup[orientation][:], expected
        )
        assert failed
        assert sleeps == [7.0]
    finally:
        for datastore in (reopened, joint, source):
            datastore.z.store.close()
