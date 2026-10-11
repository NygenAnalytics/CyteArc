import asyncio
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from threading import Event, Lock

import numpy as np
import pytest
import zarr
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.local import LocalFileSystem
from zarr.abc.store import RangeByteRequest
from zarr.core.array import AsyncArray
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, WrapperStore

from cytearc import DataStore
from cytearc.composite import create_composite
from cytearc.composite.source_cache import SourceCache
from cytearc.merge.row_plan import build_row_plan
from cytearc.storage.async_execution import ByteLedger
from cytearc.storage.virtual_arrays import virtual_array_reader
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.feature_stream import (
    map_feature_cell_bands,
    map_feature_read_groups,
)
from cytearc.storage.io_policy import StorageIoPolicy
from cytearc.storage.geometry import array_geometry
from tests.test_composite import _write_source, count_fallbacks


@pytest.fixture(scope="module")
def prefill_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp("prefill_sources")
    sources = {}
    for owner in range(4):
        values = (
            np.arange(64)[:, None] % 3 + np.arange(64)[None, :] % 5 + owner
        ).astype(np.uint16)
        features = np.roll(np.arange(64), owner * 7)
        path = directory / f"source{owner}.zarr"
        source = _write_source(
            path,
            values,
            [f"g{feature:02d}" for feature in features],
            policy=CountMatrixPolicy(8192, 2048),
        )
        sources[f"s{owner}"] = path, values, features
        source.z.store.close()
    return sources


@pytest.fixture
def cache_directories(monkeypatch):
    """Record the directory of every source cache, which its first fetch creates."""
    created = []
    ensure = SourceCache._ensure_directory

    def record(self):
        existed = self._local is not None
        result = ensure(self)
        if result and not existed:
            created.append(Path(self._directory.name))
        return result

    monkeypatch.setattr(SourceCache, "_ensure_directory", record)
    return created


@pytest.fixture
def prefill_reads(prefill_sources, monkeypatch):
    roots = {str(path.resolve()) for path, _, _ in prefill_sources.values()}
    state = {"requests": [], "active": 0, "peak": 0, "enabled": True}
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if (
            not state["enabled"]
            or str(store.root) not in roots
            or not key.startswith("RNA/counts/c/")
        ):
            return await get(store, key, *args, **kwargs)
        state["requests"].append((str(store.root), key))
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            await asyncio.sleep(0.01)
            return await get(store, key, *args, **kwargs)
        finally:
            state["active"] -= 1

    monkeypatch.setattr(LocalStore, "get", record)
    return state


@pytest.mark.parametrize("read_workers", (1, 3))
def test_composite_count_prefill_overlaps_sources_and_preserves_preparation(
    prefill_sources, prefill_reads, read_workers, monkeypatch, tmp_path
):
    create = import_module("cytearc.composite.create")
    roots = {str(path.resolve()) for path, _, _ in prefill_sources.values()}
    for module in (create, import_module("cytearc.composite.store")):
        monkeypatch.setattr(
            module, "is_remote_zarr_location", lambda path: str(path) in roots
        )
    results = []
    for prefill in (False, True):
        sources = {
            name: DataStore(
                str(path),
                mem_budget="64M",
                nthreads=3,
                storageIo=StorageIoPolicy(
                    readWorkers=read_workers if name == "s0" else 3
                ),
            )
            for name, (path, _, _) in prefill_sources.items()
        }
        prefill_reads.update(requests=[], active=0, peak=0, enabled=True)
        with monkeypatch.context() as patch:
            if not prefill:
                patch.setattr(create, "_prefetch_counts", lambda *args, **kwargs: None)
            joint = create_composite(
                sources,
                at=tmp_path / f"joint_{prefill}.zarr",
                features="union",
                mem_budget="64M",
                nthreads=3,
            )
        prefill_reads["enabled"] = False
        try:
            assert prefill_reads["active"] == 0
            assert prefill_reads["peak"] == (read_workers if prefill else 1)
            assert len(prefill_reads["requests"]) == len(roots)
            assert len(set(prefill_reads["requests"])) == len(roots)
            tables = {
                (path, name): array[:]
                for path in ("cellData", "RNA/featureData")
                for name, array in joint.z[path].arrays()
            }
            counts = joint.RNA.matrixGroup["counts"]
            results.append((tables, counts[:], counts.attrs["content_fingerprint"]))
        finally:
            joint.z.store.close()
            for source in sources.values():
                assert source.RNA.matrixGroup["counts"][0, 0] >= 0
                source.z.store.close()
    before, after = results
    assert before[0].keys() == after[0].keys()
    for key in before[0]:
        np.testing.assert_array_equal(before[0][key], after[0][key])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


@pytest.mark.parametrize("admission", ("partial_wave", "capacity", "entries"))
def test_composite_count_prefill_stops_without_eviction_and_deduplicates(
    prefill_sources, prefill_reads, admission, monkeypatch
):
    from cytearc.composite.create import _prefetch_counts

    paths = [path for path, _, _ in prefill_sources.values()]
    prefix_limit = 4096
    stores = [LocalStore(path, read_only=True) for path in paths]
    # Seed a real source document; its presence must survive all admitted waves.
    seed_key = "RNA/featureData/ids/zarr.json"
    seed_body = sync(stores[0].get(seed_key, default_buffer_prototype()))
    assert seed_body is not None
    cache = SourceCache(
        capacity_bytes={"partial_wave": 2, "capacity": 1, "entries": 4}[admission]
        * prefix_limit
        + len(seed_body),
        max_entries=3 if admission == "entries" else 16,
    )
    arrays = [
        zarr.open_array(cache.wrap(store), path="RNA/counts", mode="r")
        for store in stores
    ]
    try:
        assert sync(arrays[0].store.prefetch(seed_key, max_bytes=len(seed_body)))
        plan = build_row_plan([64] * 4, [16] * 4, list(prefill_sources))
        first_owners = list(dict.fromkeys(plan.coordinatesPermutations[:, 0]))
        geometries = [array_geometry(array) for array in arrays]
        assert all(
            2 * geometry.nominalChunkBytes() == prefix_limit for geometry in geometries
        )
        _prefetch_counts(
            arrays,
            plan,
            [np.arange(64) for _ in arrays],
            geometries,
            segment_rows=8,
            read_workers=3,
            memory_bytes=1_000_000,
            cache=cache,
        )
        expected_count = {"partial_wave": 4, "capacity": 1, "entries": 2}[admission]
        expected = [
            (str(paths[owner]), "RNA/counts/c/0/0")
            for owner in first_owners[:expected_count]
        ]
        assert prefill_reads["requests"] == expected
        assert prefill_reads["peak"] == (1 if admission == "capacity" else 2)
        assert prefill_reads["active"] == 0
        for owner in first_owners[:expected_count]:
            before = len(prefill_reads["requests"])
            np.testing.assert_array_equal(
                arrays[owner][:], prefill_sources[f"s{owner}"][1]
            )
            assert len(prefill_reads["requests"]) == before
        get = LocalStore.get

        async def refuse_seed(store, key, *args, **kwargs):
            if str(store.root) == str(paths[0]) and key == seed_key:
                pytest.fail("Count prefill evicted the existing metadata entry")
            return await get(store, key, *args, **kwargs)

        monkeypatch.setattr(LocalStore, "get", refuse_seed)
        result = sync(arrays[0].store.get(seed_key, default_buffer_prototype()))
        assert result is not None and result.to_bytes() == seed_body.to_bytes()
    finally:
        cache.close()


@pytest.mark.parametrize("reason", ("memory", "narrow"))
def test_composite_count_prefill_skips_unadmitted_or_narrow_reads(
    prefill_sources, prefill_reads, reason
):
    from cytearc.composite.create import _prefetch_counts

    cache = SourceCache()
    try:
        arrays = [
            zarr.open_array(
                cache.wrap(LocalStore(path, read_only=True)),
                path="RNA/counts",
                mode="r",
            )
            for path, _, _ in prefill_sources.values()
        ]
        mappings = [np.arange(64) for _ in arrays]
        if reason == "narrow":
            for mapping in mappings:
                mapping[1:] = -1
        _prefetch_counts(
            arrays,
            build_row_plan([64] * 4, [16] * 4, list(prefill_sources)),
            mappings,
            [array_geometry(array) for array in arrays],
            segment_rows=8,
            read_workers=3,
            memory_bytes=10_000 if reason == "memory" else 1_000_000,
            cache=cache,
        )
        assert not prefill_reads["requests"]
    finally:
        cache.close()


@pytest.mark.parametrize("decline", ("oversized", "missing"))
def test_composite_count_prefill_joins_declined_wave_and_stops(
    tmp_path, monkeypatch, decline
):
    from cytearc.composite.create import _prefetch_counts

    paths = [tmp_path / f"source{index}.zarr" for index in range(4)]
    originals = [
        zarr.create_array(
            path,
            data=np.full((64, 64), index + 1, dtype=np.uint16),
            chunks=(64, 16),
            shards=(64, 64),
            compressors=None if index == 0 else "auto",
        )
        for index, path in enumerate(paths)
    ]
    if decline == "missing":
        sync(originals[0].store.delete("c/0/0"))
    requests = []
    active = 0
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        nonlocal active
        if str(store.root) not in {str(path) for path in paths} or key != "c/0/0":
            return await get(store, key, *args, **kwargs)
        requests.append(str(store.root))
        active += 1
        try:
            await asyncio.sleep(0.01 if str(store.root) == str(paths[1]) else 0)
            return await get(store, key, *args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(LocalStore, "get", record)
    cache = SourceCache()
    try:
        arrays = [
            zarr.open_array(cache.wrap(array.store), mode="r") for array in originals
        ]
        _prefetch_counts(
            arrays,
            build_row_plan([64] * 4, [64] * 4, [f"s{index}" for index in range(4)]),
            [np.arange(64) for _ in arrays],
            [array_geometry(array) for array in arrays],
            segment_rows=64,
            read_workers=2,
            memory_bytes=1_000_000,
            cache=cache,
        )
        assert requests == [str(path) for path in paths[:2]]
        assert active == 0
        np.testing.assert_array_equal(arrays[1][:], np.full((64, 64), 2))
        assert len(requests) == 2
    finally:
        cache.close()
    np.testing.assert_array_equal(originals[1][:], np.full((64, 64), 2))


def test_composite_count_prefill_failure_drains_and_preserves_borrowed_sources(
    prefill_sources, cache_directories, monkeypatch, tmp_path
):
    create = import_module("cytearc.composite.create")
    roots = {str(path.resolve()) for path, _, _ in prefill_sources.values()}
    sources = {
        name: DataStore(str(path), mem_budget="64M", nthreads=3)
        for name, (path, _, _) in prefill_sources.items()
    }
    for module in (create, import_module("cytearc.composite.store")):
        monkeypatch.setattr(
            module, "is_remote_zarr_location", lambda path: str(path) in roots
        )
    caches = []
    prefetch = create._prefetch_counts

    def record_cache(*args, **kwargs):
        caches.append(kwargs["cache"])
        return prefetch(*args, **kwargs)

    monkeypatch.setattr(create, "_prefetch_counts", record_cache)
    started = asyncio.Event()
    release = asyncio.Event()
    active = calls = cancelled = completed = 0
    get = LocalStore.get
    cancel = SourceCache._cancel

    def release_cancelled_read(self, entry):
        cancel(self, entry)
        release.set()

    async def fail(store, key, *args, **kwargs):
        nonlocal active, calls, cancelled, completed
        if str(store.root) not in roots or not key.startswith("RNA/counts/c/"):
            return await get(store, key, *args, **kwargs)
        index = calls
        calls += 1
        active += 1
        if active == 3:
            started.set()
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            if index == 0:
                raise OSError("count prefill source failed")
            await asyncio.wait_for(release.wait(), timeout=5)
            result = await get(store, key, *args, **kwargs)
            completed += 1
            return result
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", fail)
        patch.setattr(SourceCache, "_cancel", release_cancelled_read)
        with pytest.raises(ExceptionGroup) as failure:
            create_composite(
                sources,
                at=tmp_path / "joint.zarr",
                features="union",
                mem_budget="64M",
                nthreads=3,
            )
    errors = list(failure.value.exceptions)
    while any(isinstance(error, BaseExceptionGroup) for error in errors):
        errors = [
            nested
            for error in errors
            for nested in (
                error.exceptions if isinstance(error, BaseExceptionGroup) else [error]
            )
        ]
    assert any(
        isinstance(error, OSError) and str(error) == "count prefill source failed"
        for error in errors
    )
    assert (calls, active, cancelled, completed) == (3, 0, 0, 2)
    assert len(caches) == 1 and caches[0]._local is None
    assert cache_directories and not any(path.exists() for path in cache_directories)
    for name, source in sources.items():
        np.testing.assert_array_equal(
            source.RNA.matrixGroup["counts"][:], prefill_sources[name][1]
        )
        source.z.store.close()


@pytest.mark.parametrize("dtype", (np.uint16, np.float32, np.float64))
@pytest.mark.parametrize("order", ("C", "F"))
def test_composite_indexed_copy_preserves_shuffled_and_repeated_values(
    tmp_path, monkeypatch, dtype, order
):
    module = import_module("cytearc.composite.reader")
    width = 1025
    values = (np.arange(width * width).reshape(width, width) % 32749).astype(dtype)
    if values.dtype.kind == "f":
        values = values / dtype(4) - dtype(100)
        values[0, 0] = -0.0
    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=values.shape,
        chunks=values.shape,
        dtype=values.dtype,
    )
    source[:] = values
    rng = np.random.default_rng(83)
    rows, columns = rng.permutation(width), rng.permutation(width)
    rows[-8:], columns[-8:] = rows[:8], columns[:8]
    rows[0], columns[0] = 0, 0
    row_destinations = rng.permutation(width + 6)[:width]
    column_destinations = rng.permutation(width + 8)[:width]
    output = np.full((width + 6, width + 8), 17, dtype=dtype, order=order)
    expected = output.copy()
    expected[np.ix_(row_destinations, column_destinations)] = values[
        np.ix_(rows, columns)
    ]
    copy_values = module._copy_selected_values
    copied_shapes = []

    def record(values, target, selected_rows, selected_columns, *destinations):
        copied_shapes.append((len(selected_rows), len(selected_columns)))
        return copy_values(
            values, target, selected_rows, selected_columns, *destinations
        )

    monkeypatch.setattr(module, "_copy_selected_values", record)
    sync(
        module.copy_source_selection(
            source, rows, columns, output, row_destinations, column_destinations
        )
    )
    assert copied_shapes == [(width, width)]
    np.testing.assert_array_equal(output, expected)
    assert output.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "case",
    (
        "promotion",
        "float16",
        "non_native",
        "source_fortran",
        "noncontiguous",
        "readonly",
        "small",
        "narrow",
    ),
)
def test_composite_indexed_copy_keeps_numpy_fallback(tmp_path, monkeypatch, case):
    module = import_module("cytearc.composite.reader")
    shape = (
        (32, 32)
        if case == "small"
        else (1, 1_000_000)
        if case == "narrow"
        else (1025, 1025)
    )
    source_dtype = np.float16 if case == "float16" else np.uint16
    values = (np.arange(np.prod(shape)).reshape(shape) % 251).astype(source_dtype)
    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=shape,
        chunks=shape,
        dtype=values.dtype,
        config={"order": "F" if case == "source_fortran" else "C"},
    )
    source[:] = values
    rows = np.arange(shape[0])
    columns = np.arange(shape[1])
    output_dtype = (
        np.uint32
        if case == "promotion"
        else np.dtype(np.uint16).newbyteorder("S")
        if case == "non_native"
        else values.dtype
    )
    output = np.zeros(
        (len(rows), len(columns) * (2 if case == "noncontiguous" else 1)),
        dtype=output_dtype,
    )
    if case == "noncontiguous":
        output = output[:, ::2]
    if case == "readonly":
        output.flags.writeable = False
    if case == "source_fortran":
        getitem = AsyncArray.getitem

        async def check_layout(array, selection, *args, **kwargs):
            result = await getitem(array, selection, *args, **kwargs)
            assert result.flags.f_contiguous and not result.flags.c_contiguous
            return result

        monkeypatch.setattr(AsyncArray, "getitem", check_layout)

    def unexpected(*args):
        pytest.fail("Unsupported or small selections must retain the NumPy copy")

    monkeypatch.setattr(module, "_copy_selected_values", unexpected)

    def copy():
        sync(module.copy_source_selection(source, rows, columns, output, rows, columns))

    if case == "readonly":
        with pytest.raises(ValueError, match="read-only"):
            copy()
    else:
        copy()
        expected = values[np.ix_(rows, columns)].astype(output_dtype)
        np.testing.assert_array_equal(output, expected)
        assert output.tobytes() == expected.tobytes()


@pytest.fixture(params=("compiled", "python"))
def indexed_copy(request):
    from cytearc.composite.reader import _copy_selected_values

    return (
        _copy_selected_values
        if request.param == "compiled"
        else _copy_selected_values.py_func
    )


def test_composite_indexed_copy_accepts_negative_indices(indexed_copy):
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    rows, columns = np.array([-1, 0]), np.array([-1, 1])
    row_destinations, column_destinations = np.array([-1, 0]), np.array([-1, 2])
    output = np.zeros((5, 6), dtype=values.dtype)
    expected = output.copy()
    expected[np.ix_(row_destinations, column_destinations)] = values[
        np.ix_(rows, columns)
    ]
    indexed_copy(values, output, rows, columns, row_destinations, column_destinations)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "vector,index",
    ((0, 3), (0, -4), (1, 4), (1, -5), (2, 5), (2, -6), (3, 6), (3, -7)),
)
def test_composite_indexed_copy_rejects_invalid_indices_before_writes(
    indexed_copy, vector, index
):
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    output = np.full((5, 6), 17, dtype=values.dtype)
    indices = [np.array([0, 1]) for _ in range(4)]
    indices[vector][1] = index
    with pytest.raises(IndexError):
        indexed_copy(values, output, *indices)
    assert np.all(output == 17)


@pytest.mark.parametrize("vector", range(4))
def test_composite_indexed_copy_rejects_uint64_max_before_writes(indexed_copy, vector):
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    output = np.full((5, 6), 17, dtype=values.dtype)
    indices = [np.array([0, 1]) for _ in range(4)]
    indices[vector] = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(IndexError):
        indexed_copy(values, output, *indices)
    assert np.all(output == 17)


@pytest.mark.parametrize("axis", (0, 1))
def test_composite_indexed_copy_rejects_mismatched_lengths_before_writes(
    indexed_copy, axis
):
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    output = np.full((5, 6), 17, dtype=values.dtype)
    indices = [np.array([0, 1]) for _ in range(4)]
    indices[axis + 2] = np.array([0])
    with pytest.raises(ValueError, match="lengths must match"):
        indexed_copy(values, output, *indices)
    assert np.all(output == 17)


def test_composite_indexed_copy_accepts_signed_sources_and_unsigned_destinations(
    indexed_copy,
):
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    rows, columns = np.array([-1, 0, -1]), np.array([3, -3])
    row_destinations = np.array([4, 0, 2], dtype=np.uint64)
    column_destinations = np.array([5, 2], dtype=np.uint64)
    output = np.zeros((5, 6), dtype=values.dtype)
    expected = output.copy()
    expected[np.ix_(row_destinations, column_destinations)] = values[
        np.ix_(rows, columns)
    ]
    indexed_copy(values, output, rows, columns, row_destinations, column_destinations)
    np.testing.assert_array_equal(output, expected)


@pytest.fixture(scope="module")
def reader_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reader_sources")
    rng = np.random.default_rng(92)
    sources = {}
    for name, rows, features, dtype in (
        ("a", 24, np.arange(32), np.uint8),
        ("b", 20, rng.permutation(np.arange(1, 33)), np.uint16),
    ):
        values = rng.integers(0, 29, size=(rows, len(features))).astype(dtype)
        if name == "b":
            values *= 13
        names = np.asarray([f"g{feature:02d}" for feature in features])
        path = directory / f"{name}.zarr"
        _write_source(path, values, names, policy=CountMatrixPolicy(1024, 128))
        sources[name] = path, values, names.tolist()
    return sources


@pytest.fixture(scope="module", params=("union", "intersection"))
def reader_joint(reader_sources, tmp_path_factory, request):
    sources = {
        name: DataStore(str(path), mem_budget="64M", nthreads=2)
        for name, (path, _, _) in reader_sources.items()
    }
    directory = tmp_path_factory.mktemp(f"reader_{request.param}")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            import_module("cytearc.composite.create"),
            "DEFAULT_COUNT_MATRIX_POLICY",
            CountMatrixPolicy(512, 128),
        )
        joint = create_composite(
            sources,
            at=directory / "joint.zarr",
            features=request.param,
            rows={
                "a": [23, 0, 19, 2, 7, 11, 16, 5, 20],
                "b": [19, 1, 3, 6, 8, 12, 15, 17],
            },
            mem_budget="64M",
            nthreads=2,
        )
    features = joint.RNA.feats.fetch_all("ids").tolist()
    expected = np.zeros((joint.cells.N, len(features)), dtype=np.uint16)
    for destination, identity in enumerate(joint.cells.fetch_all("ids")):
        name, row = identity.split("__c")
        _, values, source_features = reader_sources[name]
        for column, feature in enumerate(features):
            if feature in source_features:
                expected[destination, column] = values[
                    int(row), source_features.index(feature)
                ]
    return joint, expected


@pytest.fixture
def source_reads(reader_sources, monkeypatch):
    roots = {str(path.resolve()) for path, _, _ in reader_sources.values()}
    requests = []
    getitem = AsyncArray.getitem

    async def record(self, selection, *args, **kwargs):
        root = str(getattr(self.store, "root", ""))
        if root in roots and self.path in {"RNA/counts", "RNA/countsT"}:
            rows, columns = selection
            assert isinstance(rows, slice) and isinstance(columns, slice)
            row_start, row_stop, row_step = rows.indices(self.shape[0])
            column_start, column_stop, column_step = columns.indices(self.shape[1])
            assert row_step == column_step == 1
            assert (row_stop - 1) // self.chunks[0] == row_start // self.chunks[0]
            assert (column_stop - 1) // self.chunks[1] == column_start // self.chunks[1]
            requests.append(
                (
                    root,
                    self.path,
                    row_start // self.chunks[0],
                    column_start // self.chunks[1],
                )
            )
        return await getitem(self, selection, *args, **kwargs)

    monkeypatch.setattr(AsyncArray, "getitem", record)
    return requests


@pytest.fixture
def source_overlap(source_reads, reader_sources, monkeypatch):
    roots = {str(path.resolve()) for path, _, _ in reader_sources.values()}
    counts = {"active": 0, "peak": 0}
    getitem = AsyncArray.getitem

    async def record(self, selection, *args, **kwargs):
        if str(getattr(self.store, "root", "")) not in roots or self.path not in {
            "RNA/counts",
            "RNA/countsT",
        }:
            return await getitem(self, selection, *args, **kwargs)
        counts["active"] += 1
        counts["peak"] = max(counts["peak"], counts["active"])
        try:
            await asyncio.sleep(0)
            return await getitem(self, selection, *args, **kwargs)
        finally:
            counts["active"] -= 1

    monkeypatch.setattr(AsyncArray, "getitem", record)
    return counts


@pytest.mark.parametrize("name", ("counts", "countsT"))
@pytest.mark.parametrize(
    "selection", ("slice", "duplicates", "empty_rows", "empty_columns")
)
def test_composite_array_reader_preserves_selection_and_reads_chunks_once(
    reader_joint, source_reads, name, selection
):
    joint, expected = reader_joint
    array = joint.RNA.matrixGroup[name]
    reference = expected if name == "counts" else expected.T
    reader = virtual_array_reader(array)
    assert reader is not None
    assert reader.shape == array.shape
    assert reader.dtype == np.dtype(np.uint16)
    if selection == "slice":
        rows, columns = slice(None, None, -2), slice(1, None, 3)
    elif selection == "duplicates":
        rows = np.array([-1, 0, 2, 0])
        columns = np.array([-1, 1, 0, 1])
    elif selection == "empty_rows":
        rows, columns = np.array([], dtype=np.int64), slice(None)
    else:
        rows, columns = slice(None), np.array([], dtype=np.int64)
    row_indices = np.arange(array.shape[0])[rows]
    column_indices = np.arange(array.shape[1])[columns]
    actual = sync(reader.read(rows, columns))
    np.testing.assert_array_equal(
        actual, reference[np.ix_(row_indices, column_indices)]
    )
    assert actual.dtype == reference.dtype
    if actual.size:
        assert source_reads
        assert max(Counter(source_reads).values()) == 1
    else:
        assert not source_reads


def test_composite_reader_provider_does_not_leak_through_wrapper(
    reader_joint, reader_sources
):
    class ForwardingStore(WrapperStore):
        def __getattr__(self, name):
            return getattr(self._store, name)

    joint, _ = reader_joint
    wrapped = ForwardingStore(joint.z.store)
    array = zarr.open_array(store=wrapped, path="RNA/counts", mode="r")
    assert callable(wrapped.virtual_array_reader)
    assert virtual_array_reader(array) is None
    # Zarr reads through the wrapper still compute composite chunks, so their
    # scratch stays charged.
    scratch = array_geometry(joint.RNA.matrixGroup["counts"]).readScratchBytes
    assert scratch > 0
    assert array_geometry(array).readScratchBytes == scratch
    for path, _, _ in reader_sources.values():
        assert (
            virtual_array_reader(zarr.open_array(str(path / "RNA/counts"), mode="r"))
            is None
        )


def test_profiled_composite_keeps_its_scratch_and_records_its_reads(reader_joint):
    from profiling.recording_store import RecordingStoreWrapper
    from cytearc.storage.virtual_arrays import read_selection

    joint, expected = reader_joint
    profiled = RecordingStoreWrapper(joint.z.store)
    array = zarr.open_array(store=profiled, path="RNA/counts", mode="r")
    scratch = array_geometry(joint.RNA.matrixGroup["counts"]).readScratchBytes
    assert scratch > 0
    assert array_geometry(array).readScratchBytes == scratch
    # The profiler records every read, so the reader does not bypass it.
    assert virtual_array_reader(array) is None
    profiled.probe.reset()
    np.testing.assert_array_equal(
        read_selection(array, slice(None), slice(None)), expected
    )
    assert profiled.probe.to_json()["gets"] > 0


@pytest.mark.parametrize(
    "selection,error",
    (
        (np.array([[0]]), ValueError),
        (np.array([0.5]), TypeError),
        (np.array([10_000]), IndexError),
        (np.array([-10_000]), IndexError),
    ),
)
def test_composite_array_reader_rejects_invalid_selection(
    reader_joint, source_reads, selection, error
):
    joint, _ = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    with pytest.raises(error):
        sync(reader.read(selection, slice(None)))
    assert not source_reads


def test_composite_array_reader_rejects_invalid_resource_dimensions(
    reader_joint, source_reads
):
    joint, _ = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    with pytest.raises(ValueError, match="workers must be positive"):
        sync(reader.read(slice(None), slice(None), read_workers=0))
    with pytest.raises(ValueError, match="workers must be positive"):
        reader.read_scratch_bytes(1, 1, read_workers=0)
    for shape in ((-1, 1), (1, -1)):
        with pytest.raises(ValueError, match="dimensions cannot be negative"):
            reader.read_scratch_bytes(*shape)
    assert not source_reads


@pytest.mark.parametrize(
    "case",
    ("workers", "empty_rows", "empty_columns", "negative_row", "large_column"),
)
def test_composite_source_copy_validates_before_writing(
    reader_joint, source_reads, case
):
    from cytearc.composite.reader import copy_source_selection

    joint, _ = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    source = reader._sources[0]["counts"]
    rows, columns = np.array([0, 1]), np.array([0, 1])
    if case == "empty_rows":
        rows = rows[:0]
    elif case == "empty_columns":
        columns = columns[:0]
    elif case == "negative_row":
        rows[0] = -1
    elif case == "large_column":
        columns[0] = source.shape[1]
    output = np.full((2, 2), 17, dtype=source.dtype)

    def read():
        sync(
            copy_source_selection(
                source,
                rows,
                columns,
                output,
                np.arange(len(rows)),
                np.arange(len(columns)),
                read_workers=0 if case == "workers" else 1,
            )
        )

    if case == "workers":
        with pytest.raises(ValueError, match="workers must be positive"):
            read()
    elif case in {"negative_row", "large_column"}:
        with pytest.raises(IndexError, match="out-of-range"):
            read()
    else:
        read()
    np.testing.assert_array_equal(output, 17)
    assert not source_reads


@pytest.mark.parametrize("mapping", ("rowSources", "sourceRows", "featureColumns"))
def test_composite_array_reader_rejects_corrupt_mapping_before_source_reads(
    reader_joint, source_reads, mapping, monkeypatch
):
    from zarr.storage import MemoryStore

    joint, _ = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    values = reader._mappings[mapping][:].astype(np.int64)
    index = (
        np.flatnonzero(reader._mappings["rowSources"][:] == 0)[0]
        if mapping == "sourceRows"
        else 0
    )
    values.flat[index] = -2
    monkeypatch.setitem(
        reader._mappings, mapping, zarr.create_array(store=MemoryStore(), data=values)
    )

    with pytest.raises(ValueError, match="invalid source|outside a source matrix"):
        sync(reader.read(slice(None), slice(None)))
    assert not source_reads


def test_composite_array_reader_keeps_absent_source_features_zero(
    reader_joint, source_reads
):
    joint, _ = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    absent = np.flatnonzero(reader._mappings["featureColumns"][0] == -1)
    if not len(absent):
        pytest.skip("Intersection features occur in every source")
    cells = np.flatnonzero(reader._mappings["rowSources"][:] == 0)
    values = sync(reader.read(cells, absent))
    np.testing.assert_array_equal(values, np.zeros((len(cells), len(absent))))
    assert not source_reads


@pytest.mark.parametrize("minority", (False, True))
def test_composite_source_copy_keeps_selective_reads_for_narrow_shards(
    tmp_path, minority
):
    from cytearc.composite.reader import copy_source_selection

    width = 8 if minority else 4
    values = np.arange(width * width, dtype=np.uint16).reshape(width, width)
    source = zarr.create_array(
        store=LocalStore(tmp_path / "source.zarr"),
        data=values,
        chunks=(2, 2),
        shards=(width, width),
    )
    rows = np.array([0, 2]) if minority else np.array([0])
    columns = np.array([0, 2, 4]) if minority else np.array([0])
    cache = SourceCache()
    try:
        cached = zarr.open_array(store=cache.wrap(source.store), mode="r")
        output = np.empty((len(rows), len(columns)), dtype=values.dtype)
        sync(
            copy_source_selection(
                cached,
                rows,
                columns,
                output,
                np.arange(len(rows)),
                np.arange(len(columns)),
            )
        )
        np.testing.assert_array_equal(output, values[np.ix_(rows, columns)])
        assert cache._used_bytes == 0
    finally:
        cache.close()


@pytest.mark.parametrize("workers", (1, 3))
def test_composite_array_reader_bounds_parallel_source_reads(
    reader_joint, source_reads, source_overlap, workers
):
    joint, expected = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    assert reader is not None
    actual = sync(reader.read(slice(None), slice(None), read_workers=workers))
    np.testing.assert_array_equal(actual, expected)
    assert source_overlap == {"active": 0, "peak": workers}
    assert max(Counter(source_reads).values()) == 1


def test_composite_array_reader_drains_siblings_after_source_failure(
    reader_joint, monkeypatch
):
    joint, _ = reader_joint
    reader = virtual_array_reader(joint.RNA.matrixGroup["counts"])
    assert reader is not None
    getitem = AsyncArray.getitem
    started = asyncio.Event()
    active = cancelled = calls = completed = 0

    async def fail(self, selection, *args, **kwargs):
        nonlocal active, cancelled, calls, completed
        if self.path not in {"RNA/counts", "RNA/countsT"}:
            return await getitem(self, selection, *args, **kwargs)
        index = calls
        calls += 1
        active += 1
        if active == 3:
            started.set()
        try:
            await started.wait()
            if index == 0:
                raise OSError("source payload read failed")
            await asyncio.sleep(0.01)
            values = await getitem(self, selection, *args, **kwargs)
            completed += 1
            return values
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1

    monkeypatch.setattr(AsyncArray, "getitem", fail)
    with pytest.raises(ExceptionGroup) as failure:
        sync(reader.read(slice(None), slice(None), read_workers=3))
    assert any(
        isinstance(error, OSError) and str(error) == "source payload read failed"
        for error in failure.value.exceptions
    )
    assert (calls, active, cancelled, completed) == (3, 0, 0, 2)


@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize(
    ("action", "fail_during_drain"),
    (
        ("cancel", False),
        ("repeat_cancel", False),
        ("sibling_failure", False),
        ("cancel", True),
        ("sibling_failure", True),
    ),
)
def test_composite_source_copy_drains_threaded_reads(
    tmp_path, monkeypatch, workers, cached, action, fail_during_drain
):
    from cytearc.composite.reader import copy_source_selection

    values = np.arange(32, dtype=np.uint16).reshape(8, 4)
    path = tmp_path / "source.zarr"
    zarr.create_array(store=str(path), data=values, chunks=(4, 4), compressors=None)
    filesystem = LocalFileSystem(skip_instance_cache=True)
    get = filesystem.cat_file
    entered, release, finished, failed = Event(), Event(), Event(), Event()
    held = []
    enabled = True

    def read_file(key, *args, **kwargs):
        value = get(key, *args, **kwargs)
        if not enabled:
            return value
        if key.endswith("/c/0/0") and not finished.is_set():
            held.append(len(value))
            entered.set()
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("Source read was not released")
                if fail_during_drain:
                    raise OSError("source read failed during drain")
            finally:
                held.clear()
                finished.set()
        elif (
            key.endswith("/c/1/0")
            and action == "sibling_failure"
            and not failed.is_set()
        ):
            assert entered.wait(timeout=5)
            failed.set()
            raise OSError("second source rectangle failed")
        return value

    monkeypatch.setattr(filesystem, "cat_file", read_file)
    store = FsspecStore(
        AsyncFileSystemWrapper(fs=filesystem, asynchronous=True),
        path=str(path),
        read_only=True,
    )
    cache = SourceCache()
    source = zarr.open_array(store=cache.wrap(store) if cached else store, mode="r")
    output = np.zeros_like(values)
    ledger = ByteLedger(256)

    async def read():
        async def copy():
            await ledger.acquire(256)
            try:
                await copy_source_selection(
                    source,
                    np.arange(8),
                    np.arange(4),
                    output,
                    np.arange(8),
                    np.arange(4),
                    read_workers=workers,
                )
            finally:
                await ledger.release(256)

        task = asyncio.create_task(copy())
        try:
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                if action == "sibling_failure" and workers > 1:
                    while not failed.is_set():
                        await asyncio.sleep(0.001)
            if action != "sibling_failure":
                task.cancel()
                if action == "repeat_cancel":
                    await asyncio.sleep(0)
                    task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.05)
            assert not done
            assert not finished.is_set() and held == [32]
            assert ledger.held_bytes() == 256
            release.set()
            if action != "sibling_failure" and not fail_during_drain:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises((OSError, ExceptionGroup)) as failure:
                    await task
                errors = (
                    failure.value.exceptions
                    if isinstance(failure.value, ExceptionGroup)
                    else (failure.value,)
                )
                expected = (
                    "source read failed during drain"
                    if fail_during_drain
                    else "second source rectangle failed"
                )
                assert any(
                    isinstance(error, OSError) and str(error) == expected
                    for error in errors
                )
            assert finished.is_set() and not held
            assert ledger.held_bytes() == 0
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(read())
        enabled = False
        np.testing.assert_array_equal(source[:], values)
    finally:
        release.set()
        cache.close()
        store.close()


@pytest.mark.parametrize("workers", (1, 3))
def test_composite_creation_failure_drains_threaded_source_reads(
    tmp_path, monkeypatch, workers
):
    local = import_module("zarr.storage._local")
    values = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    source_path = tmp_path / "source.zarr"
    source = _write_source(
        source_path,
        values,
        [f"g{index:02d}" for index in range(32)],
        policy=CountMatrixPolicy(1024, 128),
    )
    get = local._get
    entered, release, finished, failed = Event(), Event(), Event(), Event()
    lock = Lock()
    calls = []
    held = []

    def read_file(path, prototype, byte_range):
        value = get(path, prototype, byte_range)
        if not path.is_relative_to(source_path / "RNA/counts/c") or not isinstance(
            byte_range, RangeByteRequest
        ):
            return value
        with lock:
            index = len(calls)
            calls.append(path)
        if index == 0:
            held.append(len(value))
            entered.set()
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("Source read was not released")
            finally:
                held.clear()
                finished.set()
        elif index == 1:
            assert entered.wait(timeout=5)
            failed.set()
            raise OSError("second source rectangle failed")
        return value

    monkeypatch.setattr(local, "_get", read_file)
    destination = tmp_path / "joint.zarr"
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            create_composite,
            {"source": source},
            at=destination,
            features="union",
            mem_budget="64M",
            nthreads=workers,
        )
        try:
            assert entered.wait(timeout=5)
            if workers > 1:
                assert failed.wait(timeout=5)
            with pytest.raises(TimeoutError):
                result.result(timeout=0.05)
            assert not finished.is_set() and sum(held) > 0
        finally:
            release.set()
            assert finished.wait(timeout=5)
        with pytest.raises((OSError, ExceptionGroup)):
            result.result(timeout=5)
        assert finished.is_set() and not held
    incomplete = zarr.open_group(str(destination), mode="r")
    assert not incomplete.attrs["composite"]["complete"]
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], values)


@pytest.mark.parametrize("stream", ("groups", "bands"))
@pytest.mark.parametrize("workers", (1, None))
def test_composite_feature_stream_honors_source_read_worker_limit(
    reader_joint, source_overlap, workers, stream
):
    joint, expected = reader_joint
    mapping = map_feature_read_groups if stream == "groups" else map_feature_cell_bands
    metrics = {}

    def check(group):
        values = expected.T[group.featStart : group.featEnd]
        if stream == "bands":
            values = values[:, group.cellStart : group.cellEnd]
        np.testing.assert_array_equal(group.values, values)

    list(
        mapping(
            joint.RNA.matrixGroup["countsT"],
            check,
            resources=joint.resources,
            io=StorageIoPolicy(readWorkers=workers),
            metrics=metrics,
        )
    )
    assert source_overlap["active"] == 0
    if workers is not None:
        assert source_overlap["peak"] == workers
    else:
        assert (
            1
            < source_overlap["peak"]
            <= metrics["effectiveReadWorkers"] * metrics["innerReads"]
        )


def test_composite_creation_honors_source_read_worker_limit(
    reader_sources, source_overlap, tmp_path
):
    observed = []
    for workers in (1, 3):
        sources = {
            name: DataStore(
                str(path),
                mem_budget="64M",
                nthreads=3,
                storageIo=StorageIoPolicy(readWorkers=workers),
            )
            for name, (path, _, _) in reader_sources.items()
        }
        source_overlap["peak"] = 0
        joint = create_composite(
            sources,
            at=tmp_path / f"joint_{workers}.zarr",
            features="union",
            mem_budget="64M",
            nthreads=3,
        )
        assert source_overlap == {"active": 0, "peak": workers}
        counts = joint.RNA.matrixGroup["counts"]
        observed.append(
            (
                counts[:],
                joint.cells.fetch_all("ids"),
                counts.attrs["content_fingerprint"],
            )
        )
    np.testing.assert_array_equal(observed[0][0], observed[1][0])
    np.testing.assert_array_equal(observed[0][1], observed[1][1])
    assert observed[0][2] == observed[1][2]


def test_composite_creation_caches_metadata_handles_without_caching_payloads(
    reader_sources, monkeypatch, tmp_path
):
    sources = {}
    for name, (path, _, _) in reader_sources.items():
        destination = tmp_path / f"{name}.zarr"
        shutil.copytree(path, destination)
        source = DataStore(str(destination), nthreads=1, mem_budget="64M")
        source.cells.insert(
            "nullable",
            [
                None if index % 4 == 0 else float(index)
                for index in range(source.cells.N)
            ],
            key=None,
        )
        source.cells.insert(
            "status", ["case", "control"] * (source.cells.N // 2), key=None
        )
        source.cells._get_array("status").attrs.update(
            {"levels": ["case", "control"], "ordered": False}
        )
        genes = [int(value[1:]) for value in source.RNA.feats.fetch_all("ids")]
        source.RNA.feats.insert(
            "quality", [None if gene % 5 == 0 else float(gene) for gene in genes]
        )
        source.RNA.feats._get_array("quality").attrs["ordered"] = False
        sources[name] = source

    roots = {str(source.z.store.root) for source in sources.values()}
    requests = {"metadata": Counter(), "payload": Counter()}
    get = LocalStore.get
    recording = False

    async def record(self, key, *args, **kwargs):
        if recording and str(self.root) in roots:
            if key.endswith("/zarr.json") and key.startswith(
                ("cellData/", "RNA/featureData/")
            ):
                requests["metadata"][str(self.root), key] += 1
            elif "/c/" in key:
                requests["payload"][str(self.root), key] += 1
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    results = []
    for cached in (False, True):
        opened = {
            name: DataStore(source.zarr_loc, nthreads=1, mem_budget="64M")
            for name, source in sources.items()
        }
        for counter in requests.values():
            counter.clear()
        with monkeypatch.context() as patch:
            if not cached:
                patch.setattr(
                    import_module("cytearc.composite.create"),
                    "_CachedMetadataTable",
                    lambda table: table,
                )
            recording = True
            try:
                joint = create_composite(
                    opened,
                    at=tmp_path / f"joint_{cached}.zarr",
                    features="union",
                    rows={"a": [23, 0, 7, 12], "b": [19, 1, 4]},
                    mem_budget="64M",
                    nthreads=1,
                )
            finally:
                recording = False
        tables = {}
        for path in ("cellData", "RNA/featureData"):
            for name, array in joint.z[path].arrays():
                tables[path, name] = array[:], dict(array.attrs)
        assert joint.cells._get_missing_mask_array("orig_nullable") is not None
        assert joint.RNA.feats._get_missing_mask_array("quality") is not None
        assert joint.cells._get_array("orig_status").attrs["levels"] == [
            "case",
            "control",
        ]
        counts = joint.RNA.matrixGroup["counts"]
        results.append(
            (
                {name: counter.copy() for name, counter in requests.items()},
                tables,
                counts[:],
                counts.attrs["content_fingerprint"],
            )
        )
    before, after = results
    for prefix in ("cellData/", "RNA/featureData/"):
        totals = [
            sum(
                count
                for (_, key), count in result[0]["metadata"].items()
                if key.startswith(prefix)
            )
            for result in results
        ]
        assert 0 < totals[1] < totals[0]
    assert after[0]["payload"] == before[0]["payload"]
    assert before[0]["payload"]
    assert after[1].keys() == before[1].keys()
    for key in before[1]:
        np.testing.assert_array_equal(after[1][key][0], before[1][key][0])
        assert after[1][key][1] == before[1][key][1]
    np.testing.assert_array_equal(after[2], before[2])
    assert after[3] == before[3]


@pytest.fixture
def segmented_metadata_source(tmp_path):
    values = np.arange(96 * 16, dtype=np.uint16).reshape(96, 16) % 31
    source = _write_source(
        tmp_path / "source.zarr",
        values,
        [f"g{index:02d}" for index in range(16)],
        policy=CountMatrixPolicy(512, 128),
    )
    source.cells.insert(
        "nullable",
        [None if index % 4 == 0 else float(index) for index in range(96)],
        key=None,
    )
    column = source.cells._get_array("nullable")
    mask = source.cells._get_missing_mask_array("nullable")
    assert mask is not None
    for array in (column, mask):
        data, attributes = array[:], dict(array.attrs)
        del source.z[array.path]
        replacement = source.z.create_array(array.path, data=data, chunks=(32,))
        replacement.attrs.update(attributes)
    source.RNA.feats.insert(
        "quality", [None if index % 5 == 0 else float(index) for index in range(16)]
    )
    source.cells._get_array("nullable").attrs["description"] = "Nullable score"
    assert source.RNA.rawData.chunksize[0] < 32
    return source, values


@pytest.mark.parametrize("selected_rows", (None, (35,)))
def test_composite_creation_reuses_only_selected_encoded_metadata_chunks(
    segmented_metadata_source, selected_rows, monkeypatch, tmp_path
):
    source, expected_counts = segmented_metadata_source
    create = import_module("cytearc.composite.create")
    source_root = str(source.z.store.root)
    for module in (create, import_module("cytearc.composite.store")):
        monkeypatch.setattr(
            module, "is_remote_zarr_location", lambda path: str(path) == source_root
        )
    get = LocalStore.get
    requests = Counter()
    recording = False

    async def record(self, key, *args, **kwargs):
        if recording and str(self.root) == source_root and key.startswith("cellData/"):
            if "/c/" in key:
                requests[key] += 1
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    results = []
    for cached in (False, True):
        requests.clear()
        with monkeypatch.context() as patch:
            if not cached:
                patch.setattr(
                    create._CachedMetadataTable,
                    "use_source_cache",
                    lambda *args, **kwargs: None,
                )
            recording = True
            try:
                joint = create_composite(
                    {"source": source},
                    at=tmp_path / f"joint_{cached}.zarr",
                    features="union",
                    rows=None if selected_rows is None else {"source": selected_rows},
                    mem_budget="64M",
                    nthreads=1,
                )
            finally:
                recording = False
        try:
            tables = {
                (path, name): (array[:], dict(array.attrs))
                for path in ("cellData", "RNA/featureData")
                for name, array in joint.z[path].arrays()
            }
            identities = joint.cells.fetch_all("ids")
            order = np.array([int(value.split("__c")[1]) for value in identities])
            expected_rows = np.arange(96) if selected_rows is None else selected_rows
            np.testing.assert_array_equal(np.sort(order), expected_rows)
            counts = joint.RNA.matrixGroup["counts"]
            np.testing.assert_array_equal(counts[:], expected_counts[order])
            assert joint.cells._get_missing_mask_array("orig_nullable") is not None
            assert joint.RNA.feats._get_missing_mask_array("quality") is not None
            results.append(
                (requests.copy(), tables, counts.attrs["content_fingerprint"])
            )
        finally:
            joint.z.store.close()
    before, after = results
    assert after[1].keys() == before[1].keys()
    for key in before[1]:
        np.testing.assert_array_equal(after[1][key][0], before[1][key][0])
        assert after[1][key][1] == before[1][key][1]
    assert after[2] == before[2]
    for column in ("ids", "names"):
        key = f"cellData/{column}/c/0"
        assert before[0][key] > 1
        assert after[0][key] == 1
    for column in ("nullable", "__cytearc_missing__nullable"):
        prefix = f"cellData/{column}/c/"
        prior = {
            key: count for key, count in before[0].items() if key.startswith(prefix)
        }
        current = {
            key: count for key, count in after[0].items() if key.startswith(prefix)
        }
        assert current.keys() == prior.keys()
        if selected_rows is None:
            assert set(current) == {f"{prefix}{index}" for index in range(3)}
            assert sum(current.values()) < sum(prior.values())
        else:
            assert set(current) == {f"{prefix}1"}
        assert max(current.values()) == 1


def test_composite_metadata_dtype_inspection_does_not_fetch_payload(
    segmented_metadata_source, monkeypatch
):
    source, _ = segmented_metadata_source
    handles = import_module("cytearc.composite.create")._CachedMetadataTable(
        source.cells
    )
    source_root = str(source.z.store.root)
    requests = Counter()
    get = LocalStore.get

    async def record(self, key, *args, **kwargs):
        if str(self.root) == source_root and key.startswith("cellData/"):
            if "/c/" in key:
                requests[key] += 1
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    owner = SourceCache()
    try:
        handles.use_source_cache(owner)
        assert handles.get_dtype("nullable") == np.dtype(np.float64)
        array = handles._get_array("nullable")
        missing = handles._get_missing_mask_array("nullable")
        assert missing is not None and missing.dtype == np.dtype(bool)
        assert array.shape == (96,) and array.chunks == (32,)
        assert array.attrs["description"] == "Nullable score"
        assert not requests
        np.testing.assert_array_equal(array[35:36], [35.0])
        np.testing.assert_array_equal(array[37:38], [37.0])
        np.testing.assert_array_equal(missing[35:37], [False, True])
        assert requests == {
            "cellData/nullable/c/1": 1,
            "cellData/__cytearc_missing__nullable/c/1": 1,
        }
    finally:
        owner.close()


@pytest.mark.parametrize(
    ("read_workers", "mem_budget", "expected_overlap"),
    ((1, "64M", 1), (3, "64M", 3), (None, "8M", 1)),
)
def test_composite_metadata_schema_prefetch_bounds_actual_reads(
    segmented_metadata_source,
    read_workers,
    mem_budget,
    expected_overlap,
    monkeypatch,
    tmp_path,
):
    source, _ = segmented_metadata_source
    source.cells.insert(
        "cohort", [f"study-{index % 3}" for index in range(96)], key=None
    )
    source = DataStore(
        source.zarr_loc,
        nthreads=4,
        mem_budget="64M",
        storageIo=StorageIoPolicy(readWorkers=read_workers),
    )
    create = import_module("cytearc.composite.create")
    source_root = str(source.z.store.root)
    for module in (create, import_module("cytearc.composite.store")):
        monkeypatch.setattr(
            module, "is_remote_zarr_location", lambda path: str(path) == source_root
        )
    get = LocalStore.get
    plan = create.plan_cell_metadata
    requests = Counter()
    before_plan = []
    active = peak = 0

    async def record(self, key, *args, **kwargs):
        nonlocal active, peak
        if (
            str(self.root) != source_root
            or not key.startswith("cellData/")
            or "/c/" not in key
        ):
            return await get(self, key, *args, **kwargs)
        requests[key] += 1
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return await get(self, key, *args, **kwargs)
        finally:
            active -= 1

    def record_plan(*args, **kwargs):
        before_plan.append((requests.copy(), peak))
        return plan(*args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    monkeypatch.setattr(create, "plan_cell_metadata", record_plan)
    joint = create_composite(
        {"source": source},
        at=tmp_path / "joint.zarr",
        features="union",
        rows={"source": [35]},
        mem_budget=mem_budget,
        nthreads=4,
    )
    try:
        expected_schema = {
            f"cellData/{column}/c/0": 1 for column in ("ids", "names", "cohort")
        }
        assert before_plan == [(expected_schema, expected_overlap)]
        assert active == 0
        assert all(requests[key] == 1 for key in expected_schema)
        assert requests["cellData/nullable/c/1"] == 1
        assert not requests["cellData/nullable/c/0"]
        assert not requests["cellData/nullable/c/2"]
        np.testing.assert_array_equal(joint.cells.fetch_all("ids"), ["source__c35"])
        np.testing.assert_array_equal(joint.cells.fetch_all("orig_cohort"), ["study-2"])
    finally:
        joint.z.store.close()


def test_composite_creation_with_cache_disabled_keeps_remote_metadata_and_counts(
    segmented_metadata_source, monkeypatch, tmp_path
):
    from cytearc.composite.source_cache import SOURCE_CACHE_BYTES_ENV

    source, values = segmented_metadata_source
    source_root = str(source.z.store.root)
    for name in ("cytearc.composite.create", "cytearc.composite.store"):
        monkeypatch.setattr(
            import_module(name),
            "is_remote_zarr_location",
            lambda path: str(path) == source_root,
        )
    monkeypatch.setenv(SOURCE_CACHE_BYTES_ENV, "0")
    get = LocalStore.get
    requests = Counter()

    async def record(self, key, *args, **kwargs):
        if str(self.root) == source_root and "/c/" in key:
            requests[key] += 1
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    joint = create_composite(
        {"source": source},
        at=tmp_path / "joint.zarr",
        features="union",
        rows={"source": [35]},
        mem_budget="64M",
        nthreads=2,
    )
    try:
        assert joint.z.store.source_cache is None
        assert requests["cellData/ids/c/0"] > 1
        assert requests["cellData/names/c/0"] > 1
        assert requests["cellData/nullable/c/1"] > 0
        assert not requests["cellData/nullable/c/0"]
        assert not requests["cellData/nullable/c/2"]
        np.testing.assert_array_equal(joint.cells.fetch_all("ids"), ["source__c35"])
        np.testing.assert_array_equal(joint.cells.fetch_all("orig_nullable"), [35.0])
        for _ in range(2):
            requests.clear()
            np.testing.assert_array_equal(joint.RNA.rawData[:], values[35:36])
            assert any(key.startswith("RNA/counts") for key in requests)
    finally:
        joint.z.store.close()
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], values)


@pytest.mark.parametrize("limit", ("capacity", "entries"))
def test_composite_metadata_schema_prefetch_skips_working_set_that_cannot_fit(
    segmented_metadata_source, limit, monkeypatch, tmp_path
):
    source, _ = segmented_metadata_source
    source.cells.insert(
        "cohort", [f"study-{index % 3}" for index in range(96)], key=None
    )
    create = import_module("cytearc.composite.create")
    source_root = str(source.z.store.root)
    for module in (create, import_module("cytearc.composite.store")):
        monkeypatch.setattr(
            module, "is_remote_zarr_location", lambda path: str(path) == source_root
        )
    options = (
        {"capacity_bytes": 2_000_000} if limit == "capacity" else {"max_entries": 2}
    )
    monkeypatch.setattr(create, "new_source_cache", lambda: SourceCache(**options))
    get = LocalStore.get
    plan = create.plan_cell_metadata
    requests = Counter()
    before_plan = []

    async def record(self, key, *args, **kwargs):
        if (
            str(self.root) == source_root
            and key.startswith("cellData/")
            and "/c/" in key
        ):
            requests[key] += 1
        return await get(self, key, *args, **kwargs)

    def record_plan(*args, **kwargs):
        before_plan.append(requests.copy())
        return plan(*args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    monkeypatch.setattr(create, "plan_cell_metadata", record_plan)
    observed = []
    for enabled in (False, True):
        requests.clear()
        with monkeypatch.context() as patch:
            if not enabled:
                patch.setattr(
                    create, "_prefetch_metadata_schema", lambda *args, **kwargs: None
                )
            joint = create_composite(
                {"source": source},
                at=tmp_path / f"joint_{enabled}.zarr",
                features="union",
                rows={"source": [35]},
                mem_budget="64M",
                nthreads=3,
            )
        try:
            observed.append(requests.copy())
            np.testing.assert_array_equal(joint.cells.fetch_all("ids"), ["source__c35"])
            np.testing.assert_array_equal(
                joint.cells.fetch_all("orig_cohort"), ["study-2"]
            )
        finally:
            joint.z.store.close()
    assert before_plan == [Counter(), Counter()]
    assert observed[1] == observed[0]
    assert observed[0]


def test_composite_metadata_cache_waits_for_memory_admission(
    segmented_metadata_source, monkeypatch, tmp_path
):
    source, _ = segmented_metadata_source
    source_root = str(source.z.store.root)
    for name in ("cytearc.composite.create", "cytearc.composite.store"):
        monkeypatch.setattr(
            import_module(name),
            "is_remote_zarr_location",
            lambda path: str(path) == source_root,
        )
    get = LocalStore.get
    requests = []

    async def record(self, key, *args, **kwargs):
        if str(self.root) == source_root and key.startswith(
            ("cellData/", "RNA/counts/", "RNA/countsT/")
        ):
            if "/c/" in key:
                requests.append(key)
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    with pytest.raises(MemoryError, match="metadata schema discovery"):
        create_composite(
            {"source": source},
            at=tmp_path / "joint.zarr",
            features="union",
            mem_budget="64K",
            nthreads=1,
        )
    assert not requests


def test_composite_metadata_schema_prefetch_failure_drains_other_reads(
    segmented_metadata_source, monkeypatch, tmp_path
):
    source, expected_counts = segmented_metadata_source
    source.cells.insert(
        "cohort", [f"study-{index % 3}" for index in range(96)], key=None
    )
    source_root = str(source.z.store.root)
    for name in ("cytearc.composite.create", "cytearc.composite.store"):
        monkeypatch.setattr(
            import_module(name),
            "is_remote_zarr_location",
            lambda path: str(path) == source_root,
        )
    get = LocalStore.get
    close = SourceCache.close
    cache_directories = []
    started = asyncio.Event()
    release = asyncio.Event()
    active = cancelled = calls = completed = 0
    cancel = SourceCache._cancel

    def release_cancelled_read(self, entry):
        cancel(self, entry)
        release.set()

    async def fail(self, key, *args, **kwargs):
        nonlocal active, cancelled, calls, completed
        if (
            str(self.root) != source_root
            or not key.startswith("cellData/")
            or "/c/" not in key
        ):
            return await get(self, key, *args, **kwargs)
        active += 1
        calls += 1
        if active == 3:
            started.set()
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            if key == "cellData/ids/c/0":
                raise OSError("metadata schema source read failed")
            await asyncio.wait_for(release.wait(), timeout=5)
            result = await get(self, key, *args, **kwargs)
            completed += 1
            return result
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1

    def record_close(self):
        if self._directory is not None:
            cache_directories.append(Path(self._directory.name))
        close(self)

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", fail)
        patch.setattr(SourceCache, "close", record_close)
        patch.setattr(SourceCache, "_cancel", release_cancelled_read)
        with pytest.raises(ExceptionGroup) as failure:
            create_composite(
                {"source": source},
                at=tmp_path / "joint.zarr",
                features="union",
                mem_budget="64M",
                nthreads=3,
            )
    errors = list(failure.value.exceptions)
    while any(isinstance(error, BaseExceptionGroup) for error in errors):
        errors = [
            nested
            for error in errors
            for nested in (
                error.exceptions if isinstance(error, BaseExceptionGroup) else [error]
            )
        ]
    assert any(
        isinstance(error, OSError)
        and str(error) == "metadata schema source read failed"
        for error in errors
    )
    assert (calls, active, cancelled, completed) == (3, 0, 0, 2)
    assert cache_directories and all(not path.exists() for path in cache_directories)
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], expected_counts)


@pytest.mark.parametrize("failure_phase", ("planning", "constructor", "fetch"))
def test_composite_metadata_cache_failure_closes_cache_and_keeps_source_open(
    segmented_metadata_source, failure_phase, monkeypatch, tmp_path
):
    source, expected_counts = segmented_metadata_source
    source_root = str(source.z.store.root)
    create = import_module("cytearc.composite.create")
    store = import_module("cytearc.composite.store")
    for name in ("cytearc.composite.create", "cytearc.composite.store"):
        monkeypatch.setattr(
            import_module(name),
            "is_remote_zarr_location",
            lambda path: str(path) == source_root,
        )
    get = LocalStore.get
    close = SourceCache.close
    cache_directories = []
    cached_payload_counts = []
    plan = create.plan_cell_metadata
    constructor = store.CompositeStore.__init__

    def fail_plan(*args, **kwargs):
        plan(*args, **kwargs)
        raise OSError("metadata planning failed")

    def fail_constructor(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        raise OSError("metadata constructor failed")

    async def fail(self, key, *args, **kwargs):
        if str(self.root) == source_root and key.startswith("cellData/nullable/c/"):
            raise OSError("metadata source read failed")
        return await get(self, key, *args, **kwargs)

    def record_close(self):
        if self._directory is not None:
            directory = Path(self._directory.name)
            cache_directories.append(directory)
            cached_payload_counts.append(
                sum(path.is_file() for path in directory.rglob("*"))
            )
        close(self)

    with monkeypatch.context() as patch:
        if failure_phase == "planning":
            patch.setattr(create, "plan_cell_metadata", fail_plan)
        elif failure_phase == "constructor":
            patch.setattr(store.CompositeStore, "__init__", fail_constructor)
        else:
            patch.setattr(LocalStore, "get", fail)
        patch.setattr(SourceCache, "close", record_close)
        with pytest.raises(OSError, match="metadata .*failed"):
            create_composite(
                {"source": source},
                at=tmp_path / "joint.zarr",
                features="union",
                mem_budget="64M",
                nthreads=1,
            )
    assert cache_directories
    assert any(cached_payload_counts)
    assert all(not path.exists() for path in cache_directories)
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], expected_counts)
    assert source.cells._get_array("nullable")[:].shape == (96,)


@pytest.mark.parametrize("stream", ("groups", "bands"))
def test_composite_feature_stream_preserves_boundaries_and_retained_buffers(
    reader_joint, source_reads, monkeypatch, stream
):
    joint, expected = reader_joint
    counts_t = joint.RNA.matrixGroup["countsT"]
    cells = np.arange(joint.cells.N)[::2]
    features = np.array([expected.shape[1] - 1, 1, 8, 3])
    mapping = map_feature_read_groups if stream == "groups" else map_feature_cell_bands
    options = dict(cell_idx=cells, feat_idx=features, resources=joint.resources)
    if stream == "bands":
        options.update(cellMajorOrder=True)
    with monkeypatch.context() as patch:
        patch.setattr(
            "cytearc.storage.feature_stream.virtual_array_reader", lambda _: None
        )
        baseline = list(mapping(counts_t, lambda group: group, **options))
    baseline_reads = len(source_reads)
    source_reads.clear()
    actual = list(mapping(counts_t, lambda group: group, **options))
    assert len(actual) == len(baseline)
    for observed, original in zip(actual, baseline, strict=True):
        assert (observed.featStart, observed.featEnd, observed.unitIndex) == (
            original.featStart,
            original.featEnd,
            original.unitIndex,
        )
        owner = observed.values
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert owner.nbytes == observed.values.nbytes
        if stream == "bands":
            assert (observed.cellStart, observed.cellEnd) == (
                original.cellStart,
                original.cellEnd,
            )
            np.testing.assert_array_equal(
                observed.featureRows(), original.featureRows()
            )
            # A composite band reads only its selected cells, in order.
            assert observed.values.shape[1] == len(observed.selectedLocal)
            np.testing.assert_array_equal(
                observed.values[:, observed.selectedLocal],
                original.values[:, original.selectedLocal],
            )
            np.testing.assert_array_equal(
                observed.selectedDestinations, original.selectedDestinations
            )
        else:
            np.testing.assert_array_equal(observed.values, original.values)
    assert 0 < len(source_reads) < baseline_reads


@pytest.mark.parametrize("stream", (map_feature_read_groups, map_feature_cell_bands))
@pytest.mark.parametrize("ordered", (True, False))
@pytest.mark.parametrize("ending", ("failure", "close"))
def test_composite_feature_stream_releases_raw_buffers_after_stop(
    reader_joint, monkeypatch, stream, ordered, ending
):
    import gc
    import weakref

    from cytearc.composite.reader import CompositeArrayReader

    joint, expected = reader_joint
    read = CompositeArrayReader.read
    raw_buffers = []
    failed_items = []
    failure = RuntimeError("feature callback failed")

    async def observe_read(self, *args, **kwargs):
        values = await read(self, *args, **kwargs)
        raw_buffers.append(weakref.ref(values))
        return values

    def process(item):
        if ending == "failure" and item.unitIndex == 0:
            failed_items.append(item)
            raise failure
        return item

    monkeypatch.setattr(CompositeArrayReader, "read", observe_read)
    options = dict(resources=joint.resources, orderedCompute=ordered)
    if stream is map_feature_cell_bands:
        options["cellMajorOrder"] = True
    results = stream(joint.RNA.matrixGroup["countsT"], process, **options)
    if ending == "failure":
        with pytest.raises(RuntimeError, match="feature callback failed") as caught:
            list(results)
        assert caught.value is failure
        item = failed_items[0]
    else:
        item = next(results)
        results.close()

    gc.collect()
    assert raw_buffers
    assert all(reference() is None for reference in raw_buffers)
    cells = (
        slice(item.cellStart, item.cellEnd)
        if stream is map_feature_cell_bands
        else slice(None)
    )
    np.testing.assert_array_equal(
        item.values, expected.T[item.featStart : item.featEnd, cells]
    )


@pytest.mark.parametrize("stream", (map_feature_read_groups, map_feature_cell_bands))
@pytest.mark.parametrize("close_early", (False, True))
def test_composite_feature_stream_reduces_workers_to_fit_real_reads(
    reader_joint, stream, close_early
):
    from cytearc.storage.feature_stream import read_group_layout

    joint, expected = reader_joint
    array = joint.RNA.matrixGroup["countsT"]
    reader = virtual_array_reader(array)
    layout = read_group_layout(array, cell_idx=None, feat_idx=None)
    width = max(end - start for start, end in layout.groups)
    cells = (
        max(end - start for start, end, _local, _dest in layout.selection.bands)
        if stream is map_feature_cell_bands
        else len(layout.selection.cells)
    )
    memory = (
        layout.selection.indexBytes
        + 64 * array.shape[0]
        + 2 * width * cells * reader.dtype.itemsize
        + reader.read_scratch_bytes(width, cells, read_workers=1)
    )
    metrics = {}
    results = stream(
        array,
        lambda item: item,
        resources=ResourceBudget(memory, 2),
        orderedCompute=False,
        metrics=metrics,
    )
    if close_early:
        items = [next(results)]
        results.close()
    else:
        items = list(results)
    assert items
    assert metrics["innerReads"] == 1
    assert metrics["actualComputeWorkers"] == 1
    for item in items:
        cell_selection = (
            slice(item.cellStart, item.cellEnd)
            if stream is map_feature_cell_bands
            else slice(None)
        )
        np.testing.assert_array_equal(
            item.values, expected.T[item.featStart : item.featEnd, cell_selection]
        )


def test_composite_feature_stream_limits_compute_to_available_groups(reader_joint):
    joint, expected = reader_joint
    metrics = {}
    groups = list(
        map_feature_read_groups(
            joint.RNA.matrixGroup["countsT"],
            lambda item: item,
            resources=ResourceBudget(joint.resources.memoryBytes, 8),
            orderedCompute=False,
            metrics=metrics,
        )
    )
    assert 1 < len(groups) < 8
    assert metrics["actualComputeWorkers"] == len(groups)
    for group in groups:
        np.testing.assert_array_equal(
            group.values, expected.T[group.featStart : group.featEnd]
        )


@pytest.mark.parametrize("stream", (map_feature_read_groups, map_feature_cell_bands))
def test_composite_feature_stream_empty_features_read_nothing(
    reader_joint, source_reads, stream
):
    joint, _ = reader_joint
    assert (
        list(
            stream(
                joint.RNA.matrixGroup["countsT"],
                lambda item: item,
                feat_idx=np.array([], dtype=np.int64),
                resources=joint.resources,
            )
        )
        == []
    )
    assert not source_reads


@pytest.mark.parametrize("stream", (map_feature_read_groups, map_feature_cell_bands))
def test_composite_feature_stream_preserves_empty_cell_groups(
    reader_joint, source_reads, monkeypatch, stream
):
    joint, expected = reader_joint
    options = dict(
        cell_idx=np.array([], dtype=np.int64),
        feat_idx=np.array([0, 3, expected.shape[1] - 1]),
        resources=joint.resources,
    )
    counts_t = joint.RNA.matrixGroup["countsT"]
    with monkeypatch.context() as patch:
        patch.setattr(
            "cytearc.storage.feature_stream.virtual_array_reader", lambda _: None
        )
        baseline = list(stream(counts_t, lambda group: group, **options))
    actual = list(stream(counts_t, lambda group: group, **options))
    assert len(actual) == len(baseline)
    for observed, original in zip(actual, baseline, strict=True):
        assert (observed.featStart, observed.featEnd, observed.unitIndex) == (
            original.featStart,
            original.featEnd,
            original.unitIndex,
        )
        np.testing.assert_array_equal(observed.values, original.values)
        assert observed.values.dtype == original.values.dtype
    assert not source_reads


@pytest.mark.parametrize("stream", (map_feature_read_groups, map_feature_cell_bands))
def test_composite_feature_stream_rejects_budget_before_source_payload(
    reader_joint, source_reads, stream
):
    joint, _ = reader_joint
    with pytest.raises(MemoryError):
        list(
            stream(
                joint.RNA.matrixGroup["countsT"],
                lambda group: group,
                resources=ResourceBudget(1, 1),
            )
        )
    assert not source_reads


@pytest.mark.parametrize(
    "operation", ("summary", "logged_summary", "normalization", "markers")
)
def test_composite_rna_consumers_match_virtual_reads_with_less_source_io(
    reader_joint, source_reads, monkeypatch, operation
):
    from cytearc.assay.normalization import _counts_t_renormalized_batches
    from cytearc.features.markers import find_markers_by_rank

    joint, expected = reader_joint
    cells = np.arange(joint.cells.N)[::2]
    features = np.arange(expected.shape[1])

    def run():
        if operation in {"summary", "logged_summary"}:
            return joint.RNA._compute_feature_summary(
                cells, features, log_transform=operation == "logged_summary"
            )
        if operation == "normalization":
            values = list(
                _counts_t_renormalized_batches(
                    joint.RNA,
                    cells,
                    features[::3],
                    scaleFactor=float(joint.RNA.sf),
                    logTransform=True,
                )
            )
            return {"values": np.concatenate(values)}
        result = find_markers_by_rank(
            joint.RNA, np.arange(len(cells)) % 2, cells, features, nthreads=2
        )
        return {
            "statistics": result.statistics,
            "groups": result.group_ids,
            "features": result.feature_index,
        }

    with monkeypatch.context() as patch:
        patch.setattr(
            "cytearc.storage.feature_stream.virtual_array_reader", lambda _: None
        )
        baseline = run()
    baseline_reads = len(source_reads)
    source_reads.clear()
    with count_fallbacks(monkeypatch) as fallbacks:
        actual = run()
    assert fallbacks == []
    assert actual.keys() == baseline.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], baseline[key])
        if actual[key].dtype == np.float64:
            np.testing.assert_array_equal(
                actual[key].view(np.uint64), baseline[key].view(np.uint64)
            )
    assert 0 < len(source_reads) < baseline_reads
