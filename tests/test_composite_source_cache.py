import asyncio
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Thread

import numpy as np
import pytest
import zarr
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.local import LocalFileSystem
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore

from cytearc.composite import create_composite
from cytearc.composite.source_cache import SourceCache
from cytearc.storage.count_matrix import CountMatrixPolicy
from tests.store_probes import RecordingStore
from tests.test_composite import _write_source

PROTOTYPE = default_buffer_prototype()
BODY = bytes(range(64))


def populated_source(values=None, *, store=None):
    source = RecordingStore() if store is None else store

    async def write():
        for key, value in (values or {"shard": BODY}).items():
            await source.set(key, PROTOTYPE.buffer.from_bytes(value))

    asyncio.run(write())
    source.reset()
    return source


def payload_gets(source):
    return [key for kind, key in source.ops if kind == "get"]


def cache_directory(cache):
    """Return the cache's directory, created as its first fetch would create it."""
    assert cache._ensure_directory()
    return Path(cache._directory.name)


@pytest.fixture
def shard_cache():
    owner = SourceCache(capacity_bytes=256, max_entries=4)
    yield owner
    owner.close()


@pytest.mark.parametrize(
    "limits",
    (
        {"capacity_bytes": 0},
        {"capacity_bytes": -1},
        {"max_entries": 0},
        {"max_entries": -1},
    ),
)
def test_source_cache_rejects_nonpositive_capacity(limits):
    with pytest.raises(ValueError, match="limits must be positive"):
        SourceCache(**limits)


@pytest.mark.parametrize("limit", (0, -1))
def test_source_cache_rejects_nonpositive_read_limits(shard_cache, limit):
    source = populated_source()
    cached = shard_cache.wrap(source)
    with pytest.raises(ValueError, match="prefetch byte limit must be positive"):
        asyncio.run(cached.prefetch("shard", limit))
    with pytest.raises(ValueError, match="suffix byte limit must be positive"):
        cached.cache_shard_indexes("", limit)
    assert not payload_gets(source)
    assert not shard_cache._entries and shard_cache._used_bytes == 0


@pytest.mark.parametrize("warm", (False, True))
def test_source_cache_preserves_byte_ranges_and_batch_order(shard_cache, warm):
    source = populated_source()
    cached = shard_cache.wrap(source)
    if warm:
        assert asyncio.run(cached.prefetch("shard", len(BODY)))
        source.reset()
    requests = [
        (None, BODY),
        (RangeByteRequest(3, 11), BODY[3:11]),
        (RangeByteRequest(60, 100), BODY[60:]),
        (RangeByteRequest(70, 100), b""),
        (RangeByteRequest(8, 8), b""),
        (OffsetByteRequest(17), BODY[17:]),
        (OffsetByteRequest(100), b""),
        (SuffixByteRequest(5), BODY[-5:]),
        (SuffixByteRequest(100), BODY),
        (SuffixByteRequest(0), b""),
    ]

    async def read():
        for request, expected in requests:
            result = await cached.get("shard", PROTOTYPE, request)
            assert result is not None and result.to_bytes() == expected
        partial = await cached.get_partial_values(
            PROTOTYPE, [("shard", request) for request, _ in requests]
        )
        assert [value.to_bytes() for value in partial] == [
            expected for _, expected in requests
        ]
        result = {}
        async for batch in cached.get_ranges(
            "shard", [request for request, _ in requests], prototype=PROTOTYPE
        ):
            result.update((index, value.to_bytes()) for index, value in batch)
        assert result == dict(enumerate(expected for _, expected in requests))
        many = [
            (key, value.to_bytes())
            async for key, value in cached._get_many(
                [("shard", PROTOTYPE, request) for request, _ in requests]
            )
        ]
        assert all(key == "shard" for key, _ in many)
        assert sorted(value for _, value in many) == sorted(
            expected for _, expected in requests
        )

    asyncio.run(read())
    assert bool(payload_gets(source)) is not warm


def test_source_cache_declines_oversized_and_missing_objects(shard_cache):
    source = populated_source()
    requests = []
    original = source.get

    async def record(key, prototype, byte_range=None):
        requests.append((key, byte_range))
        return await original(key, prototype, byte_range)

    source.get = record
    cached = shard_cache.wrap(source)
    assert not asyncio.run(cached.prefetch("shard", 8))
    assert requests == [("shard", RangeByteRequest(0, 9))]
    assert not asyncio.run(cached.prefetch("shard", 8))
    assert requests == [("shard", RangeByteRequest(0, 9))]
    result = asyncio.run(cached.get("shard", PROTOTYPE, OffsetByteRequest(32)))
    assert result.to_bytes() == BODY[32:]
    assert requests[-1] == ("shard", OffsetByteRequest(32))
    assert not asyncio.run(cached.prefetch("absent", 8))
    assert asyncio.run(cached.get("absent", PROTOTYPE)) is None
    assert asyncio.run(cached.prefetch("shard", len(BODY)))
    assert requests[-1] == ("shard", RangeByteRequest(0, len(BODY) + 1))
    count = len(requests)
    assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
    assert len(requests) == count


def test_source_cache_evicts_negative_entries_without_files():
    source = populated_source()
    owner = SourceCache(capacity_bytes=64, max_entries=1)
    cached = owner.wrap(source)
    try:
        assert not asyncio.run(cached.prefetch("absent", 64))
        assert owner._used_bytes == 0 and len(owner._entries) == 1
        assert not list(cache_directory(owner).iterdir())
        assert asyncio.run(cached.prefetch("shard", 64))
        source.reset()
        assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
        assert not payload_gets(source)
        assert owner._used_bytes == 64 and len(owner._entries) == 1
    finally:
        owner.close()


def test_source_cache_reads_suffix_larger_than_capacity_without_caching():
    source = populated_source()
    owner = SourceCache(capacity_bytes=4, max_entries=1)
    cached = owner.wrap(source)
    cached.cache_shard_indexes("", 8)
    try:
        for _ in range(2):
            result = asyncio.run(cached.get("shard", PROTOTYPE, SuffixByteRequest(8)))
            assert result.to_bytes() == BODY[-8:]
        assert payload_gets(source) == ["shard", "shard"]
        assert not owner._entries and owner._used_bytes == 0
    finally:
        owner.close()


def test_source_cache_rejects_source_that_exceeds_suffix_bound(
    shard_cache, monkeypatch
):
    source = populated_source()
    cached = shard_cache.wrap(source)
    cached.cache_shard_indexes("", 8)
    get = source.get

    async def exceed_bound(key, prototype, byte_range=None):
        assert byte_range == SuffixByteRequest(8)
        return await get(key, prototype, RangeByteRequest(0, 9))

    with monkeypatch.context() as patch:
        patch.setattr(source, "get", exceed_bound)
        with pytest.raises(ValueError, match="more bytes than the requested suffix"):
            asyncio.run(cached.get("shard", PROTOTYPE, SuffixByteRequest(8)))
    assert not shard_cache._entries and shard_cache._used_bytes == 0
    assert not list(cache_directory(shard_cache).iterdir())
    assert (
        asyncio.run(cached.get("shard", PROTOTYPE, SuffixByteRequest(8))).to_bytes()
        == BODY[-8:]
    )


@pytest.mark.parametrize(
    ("registered_limit", "length", "admitted"),
    (
        (8, 8, True),
        (8, 9, False),
        (100_000, 65_536, True),
        (100_000, 65_537, False),
        (8, 0, False),
    ),
)
def test_source_cache_suffix_registration_bounds(registered_limit, length, admitted):
    body = BODY * 1025
    source = populated_source({"RNA/counts/c/0/0": body})
    owner = SourceCache(capacity_bytes=2**18, max_entries=4)
    cached = owner.wrap(source)
    cached.cache_shard_indexes("RNA/counts", registered_limit)
    try:
        for _ in range(2):
            actual = asyncio.run(
                cached.get("RNA/counts/c/0/0", PROTOTYPE, SuffixByteRequest(length))
            )
            assert actual.to_bytes() == (body[-length:] if length else b"")
        assert payload_gets(source) == ["RNA/counts/c/0/0"] * (1 if admitted else 2)
        assert owner._used_bytes == (length if admitted else 0)
    finally:
        owner.close()


def test_source_cache_suffix_keys_preserve_scope_length_source_and_prototype(
    shard_cache,
):
    from zarr.core.buffer import BufferPrototype

    class TaggedBuffer(PROTOTYPE.buffer):
        pass

    tagged = BufferPrototype(TaggedBuffer, PROTOTYPE.nd_buffer)
    first = populated_source(
        {
            "RNA/counts/c/0/0": BODY,
            "RNA/counts/c/1/0": BODY[::-1],
            "RNA/countsT/c/0/0": BODY,
        }
    )
    second = populated_source({"RNA/counts/c/0/0": b"x" * 64})
    left, right = shard_cache.wrap(first), shard_cache.wrap(second)
    for cached in (left, right):
        cached.cache_shard_indexes("RNA/counts", 16)
    requests = [
        (left, "RNA/counts/c/0/0", 4, BODY[-4:]),
        (left, "RNA/counts/c/0/0", 8, BODY[-8:]),
        (left, "RNA/counts/c/1/0", 4, BODY[::-1][-4:]),
        (right, "RNA/counts/c/0/0", 4, b"xxxx"),
    ]
    for prototype in (PROTOTYPE, tagged):
        for cached, key, length, expected in requests:
            result = asyncio.run(cached.get(key, prototype, SuffixByteRequest(length)))
            assert type(result) is prototype.buffer
            assert result.to_bytes() == expected
    assert payload_gets(first) == [
        "RNA/counts/c/0/0",
        "RNA/counts/c/0/0",
        "RNA/counts/c/1/0",
    ]
    assert payload_gets(second) == ["RNA/counts/c/0/0"]
    for _ in range(2):
        actual = asyncio.run(
            left.get("RNA/countsT/c/0/0", PROTOTYPE, SuffixByteRequest(4))
        )
        assert actual.to_bytes() == BODY[-4:]
    assert payload_gets(first)[-2:] == ["RNA/countsT/c/0/0"] * 2


@pytest.mark.parametrize("limit", ("bytes", "entries"))
def test_source_cache_suffix_admission_never_evicts_existing_entries(limit):
    owner = SourceCache(
        capacity_bytes=24 if limit == "bytes" else 256,
        max_entries=8 if limit == "bytes" else 2,
    )
    source = populated_source({"whole": b"w" * 16, "a": BODY, "b": BODY[::-1]})
    cached = owner.wrap(source)
    cached.cache_shard_indexes("", 8)
    try:
        assert asyncio.run(cached.prefetch("whole", 16))
        assert (
            asyncio.run(cached.get("a", PROTOTYPE, SuffixByteRequest(8))).to_bytes()
            == BODY[-8:]
        )
        source.reset()
        for _ in range(2):
            assert asyncio.run(cached.get("whole", PROTOTYPE)).to_bytes() == b"w" * 16
            assert (
                asyncio.run(cached.get("a", PROTOTYPE, SuffixByteRequest(8))).to_bytes()
                == BODY[-8:]
            )
            assert (
                asyncio.run(cached.get("b", PROTOTYPE, SuffixByteRequest(8))).to_bytes()
                == BODY[::-1][-8:]
            )
        assert payload_gets(source) == ["b", "b"]
        assert owner._used_bytes == 24
        assert len(owner._entries) == 2
    finally:
        owner.close()


def test_source_cache_suffix_yields_to_whole_object_and_preserves_missing(shard_cache):
    source = populated_source()
    cached = shard_cache.wrap(source)
    cached.cache_shard_indexes("", 16)
    for _ in range(2):
        assert (
            asyncio.run(cached.get("absent", PROTOTYPE, SuffixByteRequest(8))) is None
        )
    assert payload_gets(source) == ["absent"]
    assert (
        asyncio.run(cached.get("shard", PROTOTYPE, SuffixByteRequest(8))).to_bytes()
        == BODY[-8:]
    )
    assert asyncio.run(cached.prefetch("shard", len(BODY)))
    source.reset()
    for request in (
        SuffixByteRequest(8),
        SuffixByteRequest(16),
        RangeByteRequest(4, 8),
    ):
        expected = (
            BODY[-request.suffix :]
            if isinstance(request, SuffixByteRequest)
            else BODY[4:8]
        )
        assert (
            asyncio.run(cached.get("shard", PROTOTYPE, request)).to_bytes() == expected
        )
    assert not payload_gets(source)


@pytest.mark.parametrize("limit", ("bytes", "entries"))
def test_source_cache_evicts_idle_entries_across_sources(limit):
    owner = SourceCache(
        capacity_bytes=32 if limit == "bytes" else 256,
        max_entries=8 if limit == "bytes" else 2,
    )
    first = populated_source({"a": b"a" * 16, "b": b"b" * 16})
    second = populated_source({"a": b"c" * 16})
    left, right = owner.wrap(first), owner.wrap(second)
    try:
        assert asyncio.run(left.prefetch("a", 16))
        assert asyncio.run(left.prefetch("b", 16))
        assert asyncio.run(left.get("a", PROTOTYPE)).to_bytes() == b"a" * 16
        assert asyncio.run(right.prefetch("a", 16))
        first.reset()
        second.reset()
        assert asyncio.run(left.get("a", PROTOTYPE)).to_bytes() == b"a" * 16
        assert asyncio.run(right.get("a", PROTOTYPE)).to_bytes() == b"c" * 16
        assert not payload_gets(first) and not payload_gets(second)
        assert asyncio.run(left.get("b", PROTOTYPE)).to_bytes() == b"b" * 16
        assert payload_gets(first) == ["b"]
    finally:
        owner.close()


class GatedSource(RecordingStore):
    def __init__(self):
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.release.set()
        self.attempts = 0
        self.cancelled = 0
        self.failure = False

    async def get(self, key, prototype, byte_range=None):
        self.attempts += 1
        self.entered.set()
        try:
            async with asyncio.timeout(5):
                while not self.release.is_set():
                    await asyncio.sleep(0.001)
            if self.failure:
                raise OSError("source read failed")
            return await super().get(key, prototype, byte_range)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


async def wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("cross_loop", (False, True))
@pytest.mark.parametrize("suffix", (False, True))
def test_source_cache_shares_one_fetch_between_concurrent_waiters(
    shard_cache, cross_loop, suffix
):
    source = populated_source(store=GatedSource())
    source.release.clear()
    cached = shard_cache.wrap(source)
    if suffix:
        cached.cache_shard_indexes("", 8)

    async def prefetch():
        async with asyncio.timeout(5):
            if suffix:
                result = await cached.get("shard", PROTOTYPE, SuffixByteRequest(8))
                assert result.to_bytes() == BODY[-8:]
                return True
            return await cached.prefetch("shard", len(BODY))

    if cross_loop:
        barrier = Barrier(2)

        def read():
            barrier.wait(timeout=5)
            return asyncio.run(prefetch())

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(read) for _ in range(2)]
            try:
                assert source.entered.wait(timeout=5)
                source.release.set()
                assert [future.result(timeout=5) for future in futures] == [True, True]
            finally:
                source.release.set()
    else:

        async def together():
            first, second = [asyncio.create_task(prefetch()) for _ in range(2)]
            try:
                await wait_until(source.entered.is_set)
                source.release.set()
                assert await asyncio.gather(first, second) == [True, True]
            finally:
                source.release.set()

        asyncio.run(together())
    assert source.attempts == 1


def test_source_cache_reserves_capacity_before_starting_another_fetch():
    owner = SourceCache(capacity_bytes=64, max_entries=2)
    source = populated_source({"a": BODY, "b": BODY}, store=GatedSource())
    source.release.clear()
    cached = owner.wrap(source)

    async def read():
        first = asyncio.create_task(cached.prefetch("a", 64))
        try:
            await wait_until(source.entered.is_set)
            assert not await cached.prefetch("b", 64)
            assert source.attempts == 1
            source.release.set()
            assert await first
        finally:
            source.release.set()

    try:
        asyncio.run(read())
    finally:
        owner.close()


@pytest.mark.parametrize("cancel_all", (False, True))
@pytest.mark.parametrize("suffix", (False, True))
def test_source_cache_cancellation_keeps_only_completed_fetches(
    shard_cache, cancel_all, suffix
):
    source = populated_source(store=GatedSource())
    source.release.clear()
    cached = shard_cache.wrap(source)
    if suffix:
        cached.cache_shard_indexes("", 8)

    async def load():
        if suffix:
            result = await cached.get("shard", PROTOTYPE, SuffixByteRequest(8))
            assert result.to_bytes() == BODY[-8:]
            return True
        return await cached.prefetch("shard", 64)

    async def read():
        first = asyncio.create_task(load())
        await wait_until(source.entered.is_set)
        second = asyncio.create_task(load())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        if cancel_all:
            second.cancel()
            await wait_until(
                lambda: any(entry.abandoned for entry in shard_cache._entries.values())
            )
            assert not second.done()
            assert shard_cache._used_bytes == (8 if suffix else 64)
            source.release.set()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert source.cancelled == 0
            assert await load()
            assert source.attempts == 2
        else:
            source.release.set()
            assert await second
            assert source.attempts == 1 and source.cancelled == 0
        assert (await cached.get("shard", PROTOTYPE)).to_bytes() == BODY

    try:
        asyncio.run(read())
    finally:
        source.release.set()


@pytest.mark.parametrize("action", ("cancel", "repeat_cancel", "close"))
@pytest.mark.parametrize("suffix", (False, True))
@pytest.mark.parametrize("source_error", (False, True))
def test_source_cache_drains_threaded_source_before_releasing_reservation(
    tmp_path, monkeypatch, action, suffix, source_error
):
    body = BODY * 1024
    (tmp_path / "shard").write_bytes(body)
    (tmp_path / "other").write_bytes(body)
    filesystem = LocalFileSystem(skip_instance_cache=True)
    original = filesystem.cat_file
    entered, release, finished = Event(), Event(), Event()
    calls = []
    held_bytes = []

    def read_file(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(args[0])
        held_bytes.append(len(value))
        entered.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("Source read was not released")
            if source_error and not finished.is_set():
                raise OSError("Source read failed while draining")
            return value
        finally:
            held_bytes.pop()
            finished.set()

    monkeypatch.setattr(filesystem, "cat_file", read_file)
    source = FsspecStore(
        AsyncFileSystemWrapper(fs=filesystem, asynchronous=True),
        path=str(tmp_path),
        read_only=True,
    )
    owner = SourceCache(capacity_bytes=len(body), max_entries=1)
    cached = owner.wrap(source)
    if suffix:
        cached.cache_shard_indexes("", len(body))
    directory = cache_directory(owner)

    async def load():
        if suffix:
            return await cached.get("shard", PROTOTYPE, SuffixByteRequest(len(body)))
        return await cached.prefetch("shard", len(body))

    async def read():
        task = asyncio.create_task(load())
        try:
            await wait_until(entered.is_set)
            assert owner._used_bytes == len(body)
            assert held_bytes == [len(body)]
            if action == "close":
                owner.close()
            else:
                task.cancel()
            await wait_until(
                lambda: any(entry.abandoned for entry in owner._entries.values())
            )
            if action == "repeat_cancel":
                task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.02)
            assert not done
            assert not finished.is_set()
            assert held_bytes == [len(body)]
            assert owner._active == 1 and owner._used_bytes == len(body)
            assert directory.exists()
            if action != "close":
                assert not await cached.prefetch("other", len(body))
                if not suffix:
                    assert not await cached.prefetch("shard", len(body))
                assert len(calls) == 1
            release.set()
            expected = (
                OSError
                if source_error
                else RuntimeError
                if action == "close"
                else asyncio.CancelledError
            )
            with pytest.raises(expected) as failure:
                await task
            if source_error:
                assert str(failure.value) == "Source read failed while draining"
            assert finished.is_set() and not held_bytes
            assert owner._active == 0 and owner._used_bytes == 0
            assert not owner._entries
            assert len(calls) == 1
            if action == "close":
                assert not directory.exists()
            else:
                assert await cached.prefetch("other", len(body))
            assert (await source.get("shard", PROTOTYPE)).to_bytes() == body
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(read())
    finally:
        owner.close()
        source.close()


@pytest.mark.parametrize("suffix", (False, True))
def test_source_cache_propagates_source_failure_and_allows_retry(shard_cache, suffix):
    source = populated_source(store=GatedSource())
    source.failure = True
    cached = shard_cache.wrap(source)
    if suffix:
        cached.cache_shard_indexes("", 8)

    async def load():
        if suffix:
            result = await cached.get("shard", PROTOTYPE, SuffixByteRequest(8))
            assert result.to_bytes() == BODY[-8:]
            return True
        return await cached.prefetch("shard", 64)

    async def read():
        failures = await asyncio.gather(
            load(),
            load(),
            return_exceptions=True,
        )
        assert all(isinstance(error, OSError) for error in failures)
        assert source.attempts == 1
        with pytest.raises(OSError, match="source read failed"):
            await cached.get("shard", PROTOTYPE, RangeByteRequest(1, 4))
        source.failure = False
        assert await load()
        assert (await cached.get("shard", PROTOTYPE)).to_bytes() == BODY

    asyncio.run(read())


def test_source_cache_turns_off_after_a_failed_local_write(shard_cache, monkeypatch):
    source = populated_source()
    cached = shard_cache.wrap(source)
    directory = cache_directory(shard_cache)
    write = LocalStore.set
    buffers = []

    async def fail(self, key, value, *args, **kwargs):
        if self.root == directory:
            buffers.append(weakref.ref(value))
            await write(self, key, value[:3], *args, **kwargs)
            raise OSError("cache write failed")
        return await write(self, key, value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "set", fail)
        assert not asyncio.run(cached.prefetch("shard", 64))
    gc.collect()
    assert buffers and all(reference() is None for reference in buffers)
    assert shard_cache.is_off
    assert not directory.exists()
    assert not shard_cache._entries and shard_cache._used_bytes == 0
    assert not asyncio.run(cached.prefetch("shard", 64))
    assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
    assert (
        asyncio.run(cached.get("shard", PROTOTYPE, RangeByteRequest(1, 4))).to_bytes()
        == BODY[1:4]
    )
    assert payload_gets(source) == ["shard", "shard", "shard"]


def test_source_cache_turns_off_when_failed_write_cannot_be_deleted(monkeypatch):
    source = populated_source({"shard": b"abcd"})
    owner = SourceCache(capacity_bytes=4, max_entries=1)
    cached = owner.wrap(source)
    directory = cache_directory(owner)
    write = LocalStore.set
    delete = LocalStore.delete
    partial_sizes = []

    async def fail_write(self, key, value, *args, **kwargs):
        await write(self, key, value, *args, **kwargs)
        if self.root == directory:
            raise OSError("cache write failed")

    async def fail_delete(self, key, *args, **kwargs):
        if self.root == directory:
            partial_sizes.append((self.root / key).stat().st_size)
            raise OSError("cache cleanup failed")
        return await delete(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "set", fail_write)
    monkeypatch.setattr(LocalStore, "delete", fail_delete)
    try:
        assert not asyncio.run(cached.prefetch("shard", 4))
        assert partial_sizes == [4]
        assert owner.is_off
        # The directory, with the object it could not delete, is removed whole.
        assert not directory.exists()
        assert not asyncio.run(cached.prefetch("shard", 4))
        assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == b"abcd"
        assert payload_gets(source) == ["shard", "shard"]
    finally:
        owner.close()


def test_source_cache_read_failure_falls_back_to_source(shard_cache, monkeypatch):
    source = populated_source()
    cached = shard_cache.wrap(source)
    assert asyncio.run(cached.prefetch("shard", 64))
    source.reset()
    directory = cache_directory(shard_cache)
    read = LocalStore.get

    async def fail(self, *args, **kwargs):
        if self.root == directory:
            raise OSError("cache read failed")
        return await read(self, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", fail)
    assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
    assert payload_gets(source) == ["shard"]
    assert shard_cache.is_off and shard_cache._active == 0
    assert not directory.exists()


def test_source_cache_missing_cached_file_falls_back_to_source(shard_cache):
    source = populated_source()
    cached = shard_cache.wrap(source)
    assert asyncio.run(cached.prefetch("shard", 64))
    source.reset()
    next(cache_directory(shard_cache).iterdir()).unlink()
    assert (
        asyncio.run(cached.get("shard", PROTOTYPE, RangeByteRequest(1, 4))).to_bytes()
        == BODY[1:4]
    )
    assert payload_gets(source) == ["shard"]
    assert shard_cache.is_off and shard_cache._active == 0


def test_source_cache_clears_shared_exception_frames(shard_cache, monkeypatch):
    source = populated_source()
    get = source.get
    buffers = []

    async def fail(key, prototype, byte_range=None):
        value = await get(key, prototype, byte_range)
        buffers.append(weakref.ref(value))
        try:
            raise ValueError("invalid source object")
        except ValueError as cause:
            try:
                raise OSError("source decode failed") from cause
            except OSError as error:
                raise ExceptionGroup("shared failure", [error, error]) from error

    monkeypatch.setattr(source, "get", fail)
    cached = shard_cache.wrap(source)
    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(cached.prefetch("shard", 64))
    gc.collect()
    first, second = failure.value.exceptions
    assert first is second
    assert str(first) == "source decode failed"
    assert str(first.__cause__) == "invalid source object"
    assert failure.value.__traceback__ is not None
    assert buffers and all(reference() is None for reference in buffers)
    assert not shard_cache._entries and shard_cache._used_bytes == 0


def test_source_cache_read_only_clone_does_not_own_source_or_cache(
    shard_cache, monkeypatch
):
    source = populated_source()
    cached = shard_cache.wrap(source)
    assert asyncio.run(cached.prefetch("shard", 64))
    closed = []
    close = source.close

    def record():
        closed.append(True)
        close()

    monkeypatch.setattr(source, "close", record)
    clone = cached.with_read_only(True)
    clone.close()
    cached.close()
    with cached:
        pass
    assert not closed
    source.reset()
    assert asyncio.run(clone.get("shard", PROTOTYPE)).to_bytes() == BODY
    assert not payload_gets(source)
    directory = cache_directory(shard_cache)
    assert directory.exists()
    shard_cache.close()
    assert not directory.exists()
    assert not closed
    assert asyncio.run(source.get("shard", PROTOTYPE)).to_bytes() == BODY


def test_source_cache_rejects_mutations_and_native_sync_reads(shard_cache):
    source = populated_source()
    cached = shard_cache.wrap(source)
    replacement = PROTOTYPE.buffer.from_bytes(b"replacement")
    assert cached.read_only
    assert not cached._supports_sync_io
    assert not cached.supports_writes
    assert not cached.supports_deletes
    with pytest.raises(ValueError, match="read-only"):
        cached._check_writable()

    async def mutate():
        for operation in (
            lambda: cached.set("shard", replacement),
            lambda: cached.set_if_not_exists("new", replacement),
            lambda: cached._set_many([("shard", replacement)]),
            lambda: cached.delete("shard"),
            lambda: cached.delete_dir(""),
            cached.clear,
        ):
            with pytest.raises(ValueError, match="read-only"):
                await operation()

    asyncio.run(mutate())
    with pytest.raises(ValueError, match="read-only"):
        cached.set_sync("shard", replacement)
    with pytest.raises(ValueError, match="read-only"):
        cached.delete_sync("shard")
    with pytest.raises(ValueError, match="read-only"):
        cached.with_read_only(False)
    with pytest.raises(TypeError, match="asynchronous"):
        cached.get_sync("shard")
    assert asyncio.run(source.get("shard", PROTOTYPE)).to_bytes() == BODY
    assert not asyncio.run(source.exists("new"))


def test_source_cache_close_drains_pending_work_and_removes_files(shard_cache):
    source = populated_source(store=GatedSource())
    source.release.clear()
    cached = shard_cache.wrap(source)
    directory = cache_directory(shard_cache)

    async def read():
        task = asyncio.create_task(cached.prefetch("shard", 64))
        await wait_until(source.entered.is_set)
        shard_cache.close()
        assert not task.done()
        assert shard_cache._used_bytes == 64
        source.release.set()
        with pytest.raises(RuntimeError, match="cache is closed"):
            await task
        await wait_until(lambda: not directory.exists())
        assert source.cancelled == 0

    try:
        asyncio.run(read())
    finally:
        source.release.set()


def test_source_cache_close_before_fetch_starts_avoids_source_io(shard_cache):
    source = populated_source()
    cached = shard_cache.wrap(source)
    directory = cache_directory(shard_cache)

    async def read():
        task = asyncio.create_task(cached.prefetch("shard", 64))
        await asyncio.sleep(0)
        entry = next(iter(shard_cache._entries.values()))
        assert not entry.started
        shard_cache.close()
        with pytest.raises(RuntimeError, match="cache is closed"):
            await task

    asyncio.run(read())
    assert not directory.exists()
    assert not payload_gets(source)
    assert shard_cache._active == 0 and shard_cache._used_bytes == 0


def test_source_cache_propagates_final_directory_cleanup_failure(
    shard_cache, monkeypatch
):
    source = populated_source(store=GatedSource())
    source.release.clear()
    cached = shard_cache.wrap(source)
    directory = cache_directory(shard_cache)

    def fail_cleanup():
        raise PermissionError("cache directory removal denied")

    async def read():
        task = asyncio.create_task(cached.prefetch("shard", 64))
        try:
            await wait_until(source.entered.is_set)
            shard_cache.close()
            source.release.set()
            with pytest.raises(PermissionError, match="cache directory removal denied"):
                await task
            assert shard_cache._active == 0 and shard_cache._used_bytes == 0
            assert directory.exists()
        finally:
            source.release.set()
            await asyncio.gather(task, return_exceptions=True)

    with monkeypatch.context() as patch:
        patch.setattr(shard_cache._directory, "cleanup", fail_cleanup)
        asyncio.run(read())
    shard_cache.close()
    assert not directory.exists()
    assert asyncio.run(source.get("shard", PROTOTYPE)).to_bytes() == BODY


@pytest.mark.parametrize("unlink_error", (False, True))
def test_source_cache_close_between_write_and_publication(
    shard_cache, monkeypatch, unlink_error
):
    source = populated_source()
    cached = shard_cache.wrap(source)
    directory = cache_directory(shard_cache)
    lock = shard_cache._lock
    closed_after_write = []

    class CloseAfterWrite:
        def __enter__(self):
            lock.acquire()

        def __exit__(self, *_):
            lock.release()
            if not closed_after_write and (directory / "0").exists():
                closed_after_write.append(True)
                thread = Thread(target=shard_cache.close)
                thread.start()
                thread.join(timeout=5)
                assert not thread.is_alive()

    unlink = Path.unlink

    def fail_unlink(path, *args, **kwargs):
        if path.parent == directory:
            raise PermissionError("cache object removal denied")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(shard_cache, "_lock", CloseAfterWrite())
    if unlink_error:
        monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(RuntimeError, match="cache is closed"):
        asyncio.run(cached.prefetch("shard", 64))
    assert closed_after_write == [True]
    assert not directory.exists()
    assert shard_cache._active == 0 and shard_cache._used_bytes == 0
    assert asyncio.run(source.get("shard", PROTOTYPE)).to_bytes() == BODY


@pytest.fixture
def cached_composite(tmp_path, monkeypatch):
    import cytearc.composite.store as stores

    values = np.tile(np.arange(1, 65, dtype=np.uint16), (128, 1))
    path = tmp_path / "source.zarr"
    source = _write_source(
        path,
        values,
        [f"g{index:03d}" for index in range(values.shape[1])],
        policy=CountMatrixPolicy(4096, 512),
    )
    monkeypatch.setattr(
        stores, "is_remote_zarr_location", lambda location: str(location) == str(path)
    )
    destination = tmp_path / "joint.zarr"
    joint = create_composite(
        {"source": source},
        at=destination,
        features="intersection",
        nthreads=2,
        mem_budget="64M",
    )
    composite = stores.CompositeStore(
        LocalStore(destination), joint.z.store.record, {"source": source.z}
    )
    joint.z.store.close()
    requests = []
    get = LocalStore.get

    async def record(self, key, prototype, byte_range=None):
        if self.root == path and key.startswith(("RNA/counts/", "RNA/countsT/")):
            if "/c/" in key:
                requests.append((key, byte_range))
        return await get(self, key, prototype, byte_range)

    monkeypatch.setattr(LocalStore, "get", record)
    yield composite, source, values, requests
    composite.close()


def test_composite_narrow_sharded_reads_reuse_only_suffix_indices(monkeypatch):
    from cytearc.composite.reader import copy_source_selection

    expected = np.arange(128 * 64, dtype=np.uint16).reshape(128, 64)
    source = RecordingStore()
    original = zarr.create_array(
        store=source, data=expected, chunks=(16, 16), shards=(64, 64)
    )
    owner = SourceCache()
    requests = []
    get = source.get

    async def record(key, prototype, byte_range=None):
        if key.startswith("c/"):
            requests.append((key, byte_range))
        return await get(key, prototype, byte_range)

    monkeypatch.setattr(source, "get", record)
    try:
        array = zarr.open_array(owner.wrap(source), mode="r")
        rows, columns = np.array([1, 17]), np.array([3])
        for offset in (0, 0, 64):
            selected_rows = rows + offset
            actual = np.zeros((len(rows), len(columns)), dtype=np.uint16)
            sync(
                copy_source_selection(
                    array,
                    selected_rows,
                    columns,
                    actual,
                    np.arange(len(rows)),
                    np.arange(len(columns)),
                    read_workers=2,
                )
            )
            np.testing.assert_array_equal(
                actual, expected[np.ix_(selected_rows, columns)]
            )
        suffixes = [
            (key, request)
            for key, request in requests
            if isinstance(request, SuffixByteRequest)
        ]
        assert [key for key, _ in suffixes] == ["c/0/0", "c/1/0"]
        assert all(
            isinstance(request, (RangeByteRequest, SuffixByteRequest))
            for _, request in requests
        )
        assert len(requests) == 8
        assert owner._used_bytes == sum(request.suffix for _, request in suffixes)
    finally:
        owner.close()
    np.testing.assert_array_equal(original[1:2, 3:4], expected[1:2, 3:4])


def test_composite_broad_reads_reuse_encoded_source_objects(cached_composite):
    composite, _, expected, requests = cached_composite
    reader = composite.virtual_array_reader("RNA/counts")
    assert reader is not None
    first = sync(reader.read(slice(None), slice(None), read_workers=2))
    np.testing.assert_array_equal(first, expected)
    assert requests
    requests.clear()
    second = sync(reader.read(slice(None), slice(None), read_workers=2))
    np.testing.assert_array_equal(second, expected)
    assert not requests


def test_composite_small_reads_keep_selective_source_ranges(cached_composite):
    composite, source, expected, requests = cached_composite
    reader = composite.virtual_array_reader("RNA/counts")
    assert reader is not None
    actual = sync(reader.read(np.array([0]), np.array([3])))
    np.testing.assert_array_equal(actual, expected[:1, 3:4])
    assert requests
    for key, request in requests:
        array = source.RNA.matrixGroup[key.split("/")[1]]
        assert np.prod(array.shards) >= 4 * np.prod(array.chunks)
        assert request is not None
        prefix = RangeByteRequest(0, 2 * int(np.prod(array.chunks)) * 2 + 1)
        assert request != prefix
    requests.clear()
    repeated = sync(reader.read(np.array([0]), np.array([3])))
    np.testing.assert_array_equal(repeated, actual)
    assert requests
    assert all(not isinstance(request, SuffixByteRequest) for _, request in requests)


def test_composite_clone_close_preserves_cache_and_borrowed_sources(
    cached_composite, monkeypatch
):
    composite, source, expected, requests = cached_composite
    reader = composite.virtual_array_reader("RNA/counts")
    assert reader is not None
    sync(reader.read(slice(None), slice(None)))
    directory = cache_directory(composite.source_cache)
    closed_sources = []
    close = LocalStore.close

    def record(self):
        if self.root == source.z.store.root:
            closed_sources.append(self)
        close(self)

    monkeypatch.setattr(LocalStore, "close", record)
    clone = composite.with_read_only(True)
    clone.close()
    assert directory.exists()
    requests.clear()
    np.testing.assert_array_equal(sync(reader.read(slice(None), slice(None))), expected)
    assert not requests
    composite.close()
    assert not directory.exists()
    assert not closed_sources
    np.testing.assert_array_equal(source.RNA.matrixGroup["counts"][:], expected)


def test_composite_close_during_parallel_read_raises_instead_of_returning_zeros(
    cached_composite, monkeypatch
):
    composite, source, _, _ = cached_composite
    reader = composite.virtual_array_reader("RNA/counts")
    assert reader is not None
    entered, release = Event(), Event()
    get = LocalStore.get

    async def block(self, key, *args, **kwargs):
        if self.root == source.z.store.root and "/c/" in key:
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
        return await get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", block)

    async def read():
        task = asyncio.create_task(
            reader.read(slice(None), slice(None), read_workers=2)
        )
        await wait_until(entered.is_set)
        composite.close()
        release.set()
        with pytest.raises(ExceptionGroup) as failure:
            await task
        assert "Composite source cache is closed" in repr(failure.value)

    try:
        asyncio.run(read())
    finally:
        release.set()


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, 2 * 1024**3),
        ("", 2 * 1024**3),
        ("0", 0),
        ("4096", 4096),
        ("512M", 512 * 1024**2),
        ("1.5g", int(1.5 * 1024**3)),
    ),
)
def test_source_cache_size_comes_from_the_environment(monkeypatch, value, expected):
    from cytearc.composite.source_cache import (
        SOURCE_CACHE_BYTES_ENV,
        configured_source_cache_bytes,
        new_source_cache,
    )

    if value is None:
        monkeypatch.delenv(SOURCE_CACHE_BYTES_ENV, raising=False)
    else:
        monkeypatch.setenv(SOURCE_CACHE_BYTES_ENV, value)
    assert configured_source_cache_bytes() == expected
    cache = new_source_cache()
    try:
        if expected:
            assert cache is not None and cache.capacity_bytes == expected
            # The directory waits for the first fetch.
            assert cache._directory is None
        else:
            assert cache is None
    finally:
        if cache is not None:
            cache.close()


@pytest.mark.parametrize("value", ("-1", "lots", "1.5", "G"))
def test_source_cache_size_rejects_values_that_are_not_sizes(monkeypatch, value):
    from cytearc.composite.source_cache import (
        SOURCE_CACHE_BYTES_ENV,
        configured_source_cache_bytes,
    )

    monkeypatch.setenv(SOURCE_CACHE_BYTES_ENV, value)
    with pytest.raises(ValueError, match=SOURCE_CACHE_BYTES_ENV):
        configured_source_cache_bytes()


def test_unused_source_cache_creates_no_directory():
    owner = SourceCache()
    cached = owner.wrap(populated_source())
    assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
    assert owner._directory is None
    owner.close()
    assert owner._directory is None


@pytest.mark.parametrize("asynchronous", (False, True))
def test_composite_store_reuses_one_reader_per_virtual_array(
    cached_composite, asynchronous
):
    def reader_for(store, path):
        if asynchronous:
            return sync(store._reader_async(path))
        return store.virtual_array_reader(path)

    composite, _, expected, _ = cached_composite
    reader = reader_for(composite, "RNA/counts")
    assert reader is not None
    assert reader_for(composite, "RNA/counts") is reader
    assert reader_for(composite, "RNA/countsT") is not reader
    assert reader_for(composite, "cellData/ids") is None
    # A clone opens its own mapping arrays.
    clone = composite.with_read_only(True)
    clone_reader = reader_for(clone, "RNA/counts")
    assert clone_reader is not None and clone_reader is not reader
    np.testing.assert_array_equal(
        sync(clone_reader.read(slice(None), slice(None))), expected
    )


def test_source_cache_closed_owner_rejects_reads_without_touching_source():
    source = populated_source()
    owner = SourceCache()
    cached = owner.wrap(source)
    assert asyncio.run(cached.prefetch("shard", len(BODY)))
    directory = cache_directory(owner)
    source.reset()
    owner.close()
    assert not directory.exists()

    with pytest.raises(RuntimeError, match="cache is closed"):
        owner.wrap(source)
    with pytest.raises(RuntimeError, match="cache is closed"):
        cached.cache_shard_indexes("", len(BODY))
    with pytest.raises(RuntimeError, match="cache is closed"):
        asyncio.run(cached.prefetch("shard", len(BODY)))
    with pytest.raises(RuntimeError, match="cache is closed"):
        asyncio.run(cached.get("shard", PROTOTYPE))
    assert not payload_gets(source)
    assert owner._active == owner._used_bytes == 0
    assert asyncio.run(source.get("shard", PROTOTYPE)).to_bytes() == BODY


def test_source_cache_concurrent_close_removes_populated_directory_once(
    monkeypatch,
):
    source = populated_source()
    owner = SourceCache()
    cached = owner.wrap(source)
    assert asyncio.run(cached.prefetch("shard", len(BODY)))
    directory = cache_directory(owner)
    assert any(directory.iterdir())
    cleanup = owner._directory.cleanup
    entered = Barrier(2)
    first_closed = Event()

    def overlapping_cleanup():
        if entered.wait(timeout=5) == 1:
            assert first_closed.wait(timeout=5)
        cleanup()

    def close():
        owner.close()
        first_closed.set()

    try:
        monkeypatch.setattr(owner._directory, "cleanup", overlapping_cleanup)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(close) for _ in range(2)]
            for future in pending:
                future.result(timeout=5)
    finally:
        first_closed.set()
        monkeypatch.undo()
        owner.close()
    assert not directory.exists()
    assert owner._directory is owner._local is None
    assert not owner._entries and owner._used_bytes == owner._active == 0
    assert asyncio.run(source.get("shard", PROTOTYPE)).to_bytes() == BODY


def test_composite_without_source_cache_reads_sources_directly(
    cached_composite, monkeypatch
):
    import cytearc.composite.store as stores
    from cytearc.composite.source_cache import (
        SOURCE_CACHE_BYTES_ENV,
        CachedSourceStore,
    )

    composite, source, expected, requests = cached_composite
    assert composite.source_cache is not None
    monkeypatch.setenv(SOURCE_CACHE_BYTES_ENV, "0")
    uncached = stores.CompositeStore(
        LocalStore(composite._store.root), composite.record, {"source": source.z}
    )
    try:
        assert uncached.source_cache is None
        assert not any(
            isinstance(array.store, CachedSourceStore)
            for array in uncached.source_arrays("counts")
        )
        reader = uncached.virtual_array_reader("RNA/counts")
        assert reader is not None
        for _ in range(2):
            requests.clear()
            np.testing.assert_array_equal(
                sync(reader.read(slice(None), slice(None))), expected
            )
            assert requests
    finally:
        uncached.close()


def test_source_cache_turned_off_before_a_fetch_starts_declines_it(shard_cache):
    source = populated_source()
    cached = shard_cache.wrap(source)

    async def read():
        task = asyncio.create_task(cached.prefetch("shard", 64))
        await asyncio.sleep(0)
        entry = next(iter(shard_cache._entries.values()))
        assert not entry.started
        shard_cache._turn_off(OSError("local disk failed"))
        assert await task is False
        assert (await cached.get("shard", PROTOTYPE)).to_bytes() == BODY

    asyncio.run(read())
    assert payload_gets(source) == ["shard"]
    assert shard_cache._active == 0 and not shard_cache._entries


def test_source_cache_bounds_large_probes_without_size_requests(monkeypatch):
    sizes = []

    class SizedSource(RecordingStore):
        """A source whose size answers may read the whole object, like HTTP."""

        async def getsize(self, key):
            sizes.append(key)
            return len(self._store_dict[key])

    reservation = 16 * 1024**2
    source = populated_source({"large": BODY * 4}, store=SizedSource())
    requests = []
    get = source.get

    async def record(key, prototype, byte_range=None):
        requests.append(byte_range)
        return await get(key, prototype, byte_range)

    monkeypatch.setattr(source, "get", record)
    owner = SourceCache(capacity_bytes=2 * reservation, max_entries=4)
    cached = owner.wrap(source)
    try:
        assert asyncio.run(cached.prefetch("large", reservation))
        assert sizes == []
        assert requests == [RangeByteRequest(0, reservation + 1)]
    finally:
        owner.close()


def test_source_cache_close_during_failed_write_cleanup_resolves_the_read(
    monkeypatch,
):
    source = populated_source()
    owner = SourceCache(capacity_bytes=256, max_entries=4)
    cached = owner.wrap(source)
    directory = cache_directory(owner)
    write = LocalStore.set
    delete = LocalStore.delete
    deleting = asyncio.Event()
    release = asyncio.Event()

    async def fail_write(self, key, value, *args, **kwargs):
        await write(self, key, value, *args, **kwargs)
        if self.root == directory:
            raise OSError("cache write failed")

    async def slow_delete(self, key, *args, **kwargs):
        if self.root == directory:
            deleting.set()
            await release.wait()
        return await delete(self, key, *args, **kwargs)

    async def read():
        task = asyncio.create_task(cached.prefetch("shard", 64))
        await asyncio.wait_for(deleting.wait(), timeout=5)
        entry = next(iter(owner._entries.values()))
        # Closing cancels the fetch while it deletes its partial write.
        owner.close()
        await asyncio.sleep(0)
        release.set()
        done, _ = await asyncio.wait({task}, timeout=5)
        if task not in done:
            # Release the stranded read so the failure cannot hang the suite.
            entry.future.set_result(False)
            await task
            pytest.fail("Closing during cleanup left the read unresolved")
        with pytest.raises(RuntimeError, match="cache is closed"):
            await task

    monkeypatch.setattr(LocalStore, "set", fail_write)
    monkeypatch.setattr(LocalStore, "delete", slow_delete)
    asyncio.run(read())
    assert owner._active == 0 and owner._used_bytes == 0
    assert not owner._entries
    assert not directory.exists()


def test_source_cache_turns_off_when_its_directory_cannot_be_created(monkeypatch):
    import cytearc.composite.source_cache as source_cache

    def fail(*args, **kwargs):
        raise OSError("no temporary directory")

    monkeypatch.setattr(source_cache, "TemporaryDirectory", fail)
    source = populated_source()
    owner = SourceCache(capacity_bytes=256, max_entries=4)
    cached = owner.wrap(source)
    try:
        assert not asyncio.run(cached.prefetch("shard", 64))
        assert owner.is_off and owner._directory is None
        assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
        assert payload_gets(source) == ["shard"]
    finally:
        owner.close()


def test_source_cache_turns_off_when_eviction_cannot_delete(monkeypatch):
    source = populated_source({"first": BODY, "second": BODY})
    owner = SourceCache(capacity_bytes=64, max_entries=1)
    cached = owner.wrap(source)
    try:
        assert asyncio.run(cached.prefetch("first", 64))
        directory = cache_directory(owner)
        unlink = Path.unlink

        def fail(path, *args, **kwargs):
            if path.parent == directory:
                raise PermissionError("cache object removal denied")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail)
        assert not asyncio.run(cached.prefetch("second", 64))
        assert owner.is_off
        # Removing the whole directory does not need the denied unlink.
        assert not directory.exists()
        source.reset()
        assert asyncio.run(cached.get("first", PROTOTYPE)).to_bytes() == BODY
        assert payload_gets(source) == ["first"]
    finally:
        owner.close()


def test_source_cache_keeps_a_directory_it_cannot_remove_for_close(
    shard_cache, monkeypatch
):
    source = populated_source()
    cached = shard_cache.wrap(source)
    directory = cache_directory(shard_cache)

    def fail_cleanup():
        raise PermissionError("cache directory removal denied")

    with monkeypatch.context() as patch:
        patch.setattr(shard_cache._directory, "cleanup", fail_cleanup)
        shard_cache._turn_off(OSError("local disk failed"))
        assert shard_cache.is_off and directory.exists()
    assert asyncio.run(cached.get("shard", PROTOTYPE)).to_bytes() == BODY
    shard_cache.close()
    assert not directory.exists()


def test_composite_construction_forgets_readers_of_removed_arrays(cached_composite):
    import copy as copy_module

    import cytearc.composite.store as stores

    composite, source, _, _ = cached_composite
    building = stores.CompositeStore(
        LocalStore(composite._store.root),
        copy_module.deepcopy(composite.record),
        {"source": source.z},
        constructing=True,
    )
    try:
        assert building.virtual_array_reader("RNA/counts") is not None
        sync(building.delete("RNA/counts/zarr.json"))
        assert building.virtual_array_reader("RNA/counts") is None
        assert building.virtual_array_reader("RNA/countsT") is not None
        sync(building.delete_dir("RNA/countsT"))
        assert building.virtual_array_reader("RNA/countsT") is None
        assert not building._readers
    finally:
        building.close()
