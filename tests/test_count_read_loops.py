"""Count reads on an event loop of their own, in the thread that asks for them.

``read_selection`` runs each read's coroutine through ``run_read``, on a new
event loop of the calling thread, so the loop's work, such as copying decoded
chunks into the result, runs in that thread and reads from several threads
run at once instead of in turn on Zarr's one loop thread. Store calls and
decodes run on Zarr's codec pool, which each loop borrows and never shuts
down. A loop runs until every task its read started has ended, such as a
shared cache fill that another read waits on, and then closes. A thread that
already runs an event loop reads through Zarr's loop.
"""

import asyncio
import contextlib
import functools
import gc
import http.server
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from typing import Any

import numpy as np
import pytest
import zarr
import zarr.core.array as zarr_array
import zarr.core.sync as zarr_sync
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from zarr.codecs import ZstdCodec
from zarr.storage import (
    FsspecStore,
    LocalStore,
    LoggingStore,
    MemoryStore,
    ObjectStore,
    WrapperStore,
    ZipStore,
)

from cytearc.composite import create_composite
from cytearc.composite.source_cache import SourceCache
from cytearc.matrix import ChunkedArray
from cytearc.storage.async_execution import ensure_zarr_host_ceiling, run_read
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.virtual_arrays import (
    VirtualArrayStore,
    _reads_on_any_loop,
    read_selection,
)
from tests.test_composite import _write_source
from tests.test_composite_source_cache import (
    BODY,
    GatedSource,
    populated_source,
    wait_until,
)

VALUES = np.arange(32 * 18, dtype=np.uint16).reshape(32, 18)
SELECTIONS = {
    "slices": (slice(2, 30), slice(1, 17)),
    "row indices": (np.array([3, 1, 20, 31]), slice(0, 18)),
    "indices on both axes": (np.array([3, 1, 20, 31]), np.array([5, 0, 17, 9])),
}


def _array(*, shards: tuple[int, int] | None = (16, 18)) -> zarr.Array:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array(
        "counts",
        shape=VALUES.shape,
        chunks=(8, 6),
        shards=shards,
        dtype=VALUES.dtype,
        compressors=ZstdCodec(),
    )
    array[:] = VALUES
    return array


def _record_selections(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record the thread, its name and the running loop of each Zarr selection read."""
    seen: list[tuple] = []
    original = zarr_array._get_selection

    async def record(*args, **kwargs):
        seen.append(
            (
                threading.get_ident(),
                threading.current_thread().name,
                asyncio.get_running_loop(),
            )
        )
        return await original(*args, **kwargs)

    monkeypatch.setattr(zarr_array, "_get_selection", record)
    return seen


@pytest.mark.parametrize(("rows", "columns"), SELECTIONS.values(), ids=SELECTIONS)
def test_a_read_runs_on_a_loop_of_the_calling_thread(monkeypatch, rows, columns):
    array = _array()
    seen = _record_selections(monkeypatch)

    np.testing.assert_array_equal(
        read_selection(array, rows, columns), VALUES[rows][:, columns]
    )

    assert seen
    for ident, _, loop in seen:
        assert ident == threading.get_ident()
        assert loop is not zarr_sync._get_loop()
        # The read's loop closed once the read ended.
        assert loop.is_closed()


def test_reads_from_two_threads_run_their_loop_work_at_once(monkeypatch):
    array = _array()
    meeting = threading.Barrier(2, timeout=5)
    original = zarr_array._get_selection

    async def meet(*args, **kwargs):
        # Work that holds its loop: on one shared loop thread, the second read
        # could not reach this point until the first had passed it.
        meeting.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(zarr_array, "_get_selection", meet)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reads = [
            pool.submit(read_selection, array, slice(0, 16), slice(0, 18))
            for _ in range(2)
        ]
        for read in reads:
            np.testing.assert_array_equal(read.result(timeout=20), VALUES[:16])


def test_many_streams_in_a_row_leave_zarrs_codec_pool_running():
    array = _array()
    ensure_zarr_host_ceiling()
    pool = zarr_sync._executor
    counts = ChunkedArray(array, block_size=4)
    for _ in range(25):
        # Each stream reads in threads of its own, one loop per read.
        blocks = list(counts.stream_blocks(nthreads=4, prefetch=4))
        np.testing.assert_array_equal(np.vstack(blocks), VALUES)
    del blocks
    gc.collect()

    assert zarr_sync._executor is pool
    assert not pool._shutdown
    # Zarr's own synchronous reads still decode on the pool.
    np.testing.assert_array_equal(array[:], VALUES)


def test_decodes_run_on_zarrs_codec_pool(monkeypatch):
    array = _array()
    names: list[str] = []
    original = ZstdCodec._decode_sync

    def record(self, *args, **kwargs):
        names.append(threading.current_thread().name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ZstdCodec, "_decode_sync", record)
    np.testing.assert_array_equal(
        read_selection(array, slice(None), slice(None)), VALUES
    )

    assert names
    assert all(name.startswith("zarr_pool") for name in names)


def test_a_failed_read_waits_for_the_shared_fill_it_started():
    source = populated_source(store=GatedSource())
    source.release.clear()
    cache = SourceCache(capacity_bytes=256, max_entries=4)
    cached = cache.wrap(source)
    failures: list[tuple[BaseException, float]] = []

    async def failing_read():
        # Start a fill of the shared cache, as a composite read does, and
        # fail while the fill is in flight.
        asyncio.ensure_future(cached.prefetch("shard", len(BODY)))
        await wait_until(source.entered.is_set)
        raise ValueError("the read failed")

    async def waiting_read():
        await wait_until(source.entered.is_set)
        return await cached.prefetch("shard", len(BODY))

    def fail() -> None:
        try:
            run_read(failing_read())
        except ValueError as error:
            failures.append((error, time.monotonic()))

    failing = threading.Thread(target=fail)
    failing.start()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        waiting = pool.submit(run_read, waiting_read())
        assert source.entered.wait(timeout=5)
        # The failed read keeps its loop running for the fill that the other
        # read waits on.
        failing.join(timeout=0.3)
        assert failing.is_alive()
        released = time.monotonic()
        source.release.set()
        assert waiting.result(timeout=5) is True
    finally:
        # Released before waiting for the readers, so a failure never hangs.
        source.release.set()
        pool.shutdown(wait=True)
        failing.join(timeout=5)
        cache.close()

    [(error, ended)] = failures
    assert str(error) == "the read failed"
    assert ended >= released
    assert source.attempts == 1


@pytest.fixture
def cached_composite(tmp_path, monkeypatch):
    """A composite whose one source reads through a source cache."""
    import cytearc.composite.store as stores

    values = np.arange(128 * 64, dtype=np.uint16).reshape(128, 64) % 251
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
    assert composite.source_cache is not None
    yield composite
    composite.close()


def test_one_composite_read_from_several_threads_at_once(cached_composite):
    counts = zarr.open_array(store=cached_composite, path="RNA/counts", mode="r")
    rows = np.arange(5, 120)
    columns = np.arange(0, 64, 3)
    # Zarr's own read computes the composite's chunks from the same sources.
    expected = np.asarray(counts.get_orthogonal_selection((rows, columns)))
    start = threading.Barrier(4, timeout=5)

    def read() -> np.ndarray:
        start.wait()
        return read_selection(counts, rows, columns)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [pool.submit(read) for _ in range(4)]
        for result in results:
            np.testing.assert_array_equal(result.result(timeout=30), expected)


def test_a_thread_running_an_event_loop_reads_through_zarrs_loop(monkeypatch):
    array = _array()
    seen = _record_selections(monkeypatch)

    async def read_inside_a_loop() -> np.ndarray:
        return read_selection(array, slice(0, 8), slice(0, 18))

    np.testing.assert_array_equal(asyncio.run(read_inside_a_loop()), VALUES[:8])
    assert [loop for _, _, loop in seen] == [zarr_sync._get_loop()]


def test_an_interrupted_read_stops_before_run_read_returns():
    stopped: list[str] = []

    async def read() -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        asyncio.get_running_loop().call_soon(interrupt)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.append("cancelled")
            raise

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_read(read())

    assert stopped == ["cancelled"]
    assert time.monotonic() - started < 5


def _known_stores(tmp_path) -> dict[str, object]:
    """Stores known to work on any event loop, one of each kind."""
    from obstore.store import MemoryStore as ObjectMemory

    return {
        "local": LocalStore(tmp_path / "local.zarr"),
        "memory": MemoryStore(),
        "obstore": ObjectStore(ObjectMemory()),
        "fsspec over a synchronous file system": FsspecStore.from_url(
            f"memory://{tmp_path.name}/fsspec.zarr"
        ),
        "wrapper of a local store": LoggingStore(LocalStore(tmp_path / "wrapped.zarr")),
    }


@pytest.mark.parametrize(
    "kind",
    [
        "local",
        "memory",
        "obstore",
        "fsspec over a synchronous file system",
        "wrapper of a local store",
    ],
)
def test_stores_that_work_on_any_loop_read_on_a_loop_of_their_own(
    tmp_path, monkeypatch, kind
):
    store = _known_stores(tmp_path)[kind]
    array = zarr.create_array(
        store=store, shape=VALUES.shape, chunks=(8, 6), dtype=VALUES.dtype
    )
    array[:] = VALUES
    seen = _record_selections(monkeypatch)

    np.testing.assert_array_equal(
        read_selection(array, slice(0, 16), np.array([4, 0, 17])),
        VALUES[:16][:, [4, 0, 17]],
    )
    assert seen
    assert all(loop is not zarr_sync._get_loop() for _, _, loop in seen)


def test_stores_that_may_be_bound_to_one_loop_read_through_zarrs_loop(tmp_path):
    from fsspec.implementations.memory import MemoryFileSystem

    # Any other store, here a zip file, is not known to work on any loop.
    assert not _reads_on_any_loop(ZipStore(tmp_path / "counts.zip", mode="w"), "")
    # A wrapper that limits its calls with an asyncio.Semaphore binds it to a loop.
    limited = AsyncFileSystemWrapper(
        MemoryFileSystem(), asynchronous=True, semaphore=asyncio.Semaphore(2)
    )
    assert not _reads_on_any_loop(FsspecStore(limited, path="/counts.zarr"), "")
    # A native asynchronous file system binds its client session to a loop.
    pytest.importorskip("aiohttp")
    import fsspec

    native = fsspec.filesystem("http", asynchronous=True)
    assert not _reads_on_any_loop(FsspecStore(native, path="http://127.0.0.1/a"), "")


class _NamedStores(WrapperStore, VirtualArrayStore):
    """A virtual store that names the stores its reads touch."""

    def __init__(self, store, named=None) -> None:
        super().__init__(store)
        self.named = named

    def virtual_array_reader(self, path):
        return None

    def chunk_read_scratch_bytes(self, path):
        return 0

    def stores_read(self, path):
        return self.named if self.named is not None else super().stores_read(path)


def test_a_virtual_array_reads_on_its_own_loop_only_when_every_store_it_names_allows_it(
    tmp_path,
):
    local = LocalStore(tmp_path / "local.zarr")
    # Unnamed stores keep Zarr's loop, even over a store that allows its own.
    assert not _reads_on_any_loop(_NamedStores(local), "counts")
    assert _reads_on_any_loop(_NamedStores(local, (local, MemoryStore())), "counts")
    zipped = ZipStore(tmp_path / "source.zip", mode="w")
    assert not _reads_on_any_loop(_NamedStores(local, (local, zipped)), "counts")


def test_a_composite_names_its_own_store_and_the_stores_of_its_sources(
    cached_composite,
):
    from cytearc.composite.source_cache import CachedSourceStore

    stores = cached_composite.stores_read("RNA/counts")

    assert stores[0] is cached_composite._store
    # Its one source reads ``counts`` and ``countsT`` through the source cache.
    assert len(stores) == 3
    assert all(isinstance(store, CachedSourceStore) for store in stores[1:])
    assert _reads_on_any_loop(cached_composite, "RNA/counts")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files without logging each request."""

    def log_message(self, format, *args):  # noqa: A002 - the base class names it
        pass


@pytest.fixture
def served_counts(tmp_path):
    """The counts as a Zarr array served over HTTP from this machine."""
    zarr.create_array(
        str(tmp_path / "served.zarr"),
        shape=VALUES.shape,
        chunks=(8, 6),
        dtype=VALUES.dtype,
    )[:] = VALUES
    handler = functools.partial(_QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/served.zarr"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_an_aiohttp_fsspec_store_reads_correctly_through_zarrs_loop(
    served_counts, monkeypatch
):
    pytest.importorskip("aiohttp")
    array = zarr.open_array(
        store=FsspecStore.from_url(served_counts, read_only=True), mode="r"
    )
    seen = _record_selections(monkeypatch)

    # The file system's session is bound to the loop it first ran on, so
    # every read, the second and third ones too, runs on Zarr's loop.
    for _ in range(3):
        np.testing.assert_array_equal(
            read_selection(array, slice(4, 20), np.array([3, 0, 11])),
            VALUES[4:20][:, [3, 0, 11]],
        )
    assert [loop for _, _, loop in seen] == [zarr_sync._get_loop()] * 3


def test_an_interrupted_read_finishes_the_shared_fetch_that_another_loop_waits_for():
    from cytearc.storage.huggingface import _SharedFetches

    fetches = _SharedFetches[str]()
    listing = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    async def fetch() -> str:
        # A store's listing, in flight until the test releases it.
        calls.append(1)
        listing.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        return "listing"

    async def interrupted_read() -> str:
        waiting = asyncio.ensure_future(fetches.get("", fetch))
        await wait_until(listing.is_set)

        def interrupt() -> None:
            raise KeyboardInterrupt

        asyncio.get_running_loop().call_soon(interrupt)
        return await waiting

    async def waiting_read() -> str:
        await wait_until(listing.is_set)
        return await fetches.get("", fetch)

    interrupted: list[float] = []

    def interrupt_mid_listing() -> None:
        try:
            run_read(interrupted_read())
        except KeyboardInterrupt:
            interrupted.append(time.monotonic())

    reader = threading.Thread(target=interrupt_mid_listing)
    reader.start()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        waiting = pool.submit(run_read, waiting_read())
        assert listing.wait(timeout=5)
        # The interrupted read keeps its loop running for the listing it
        # started, which the other read waits for.
        reader.join(timeout=0.3)
        assert reader.is_alive()
        released = time.monotonic()
        release.set()
        assert waiting.result(timeout=5) == "listing"
    finally:
        # Released before waiting for the readers, so a failure never hangs.
        release.set()
        pool.shutdown(wait=True)
        reader.join(timeout=5)

    [ended] = interrupted
    assert ended >= released
    # One listing served both reads: none was abandoned and taken again.
    assert calls == [1]


class _FailingStore(MemoryStore):
    """A memory store whose reads of the keys in ``failing`` raise ``OSError``.

    ``whole_values_too`` makes reads of a whole value fail as well as reads of
    a byte range, so a shard's index, which Zarr reads from the shard's end,
    fails only when it is set.
    """

    failing: frozenset[str] = frozenset()
    whole_values_too = True

    async def get(self, key, prototype, byte_range=None):  # type: ignore[no-untyped-def]
        from zarr.abc.store import RangeByteRequest

        if key in self.failing and (
            self.whole_values_too or isinstance(byte_range, RangeByteRequest)
        ):
            raise OSError(f"the store could not read {key}")
        return await super().get(key, prototype, byte_range)


# A read of 4,000 rows by 500 values of uint16 takes 4 MB, and a chunk 400 KB.
FAILED_READ_SHAPE = (4_000, 500)
FAILED_READ_CHUNK_BYTES = 400 * 500 * 2

FAILED_READS = {
    # Zarr fetches each chunk on its own, and the last one fails.
    "a chunk": ((400, 500), None, {"c/9/0"}, True, slice(None)),
    # Zarr fetches the whole of each shard it reads, and the second one fails.
    "a whole shard": ((400, 500), (2_000, 500), {"c/1/0"}, True, slice(None)),
    # A read of half the chunks of each shard fetches their byte ranges once
    # it has the shard's index, merged into one fetch, which fails for the
    # second shard. Zarr's own frames then hold the error group it raises.
    "the chunks of a shard": (
        (400, 250),
        (2_000, 500),
        {"c/1/0"},
        False,
        slice(0, 250),
    ),
}


def _failing_read(layout: str) -> tuple[zarr.Array, tuple[slice, slice]]:
    chunks, shards, failing, whole_values_too, columns = FAILED_READS[layout]
    store = _FailingStore()
    root = zarr.open_group(store=store, mode="w")
    array = root.create_array(
        "counts",
        shape=FAILED_READ_SHAPE,
        chunks=chunks,
        shards=shards,
        dtype=np.uint16,
        compressors=ZstdCodec(),
    )
    array[:] = np.arange(np.prod(FAILED_READ_SHAPE), dtype=np.uint16).reshape(
        FAILED_READ_SHAPE
    )
    # A read of the first rows, which fails nowhere, makes the codecs and
    # pools that every read shares before anything is traced.
    np.asarray(array[:10, :])
    store.failing = frozenset(f"counts/{key}" for key in failing)
    store.whole_values_too = whole_values_too
    return array, (slice(None), columns)


def _traced_bytes_settle_below(limit: int, timeout: float = 10.0) -> int:
    """Return the bytes traced once they fall below ``limit``, or after ``timeout`` seconds.

    On Zarr's loop a failed read's other chunk reads run on after the error
    reaches the caller, holding the output until they end.
    """
    import tracemalloc

    deadline = time.monotonic() + timeout
    while (traced := tracemalloc.get_traced_memory()[0]) >= limit:
        if time.monotonic() > deadline:
            break
        time.sleep(0.02)
    return traced


def _read_errors_left(timeout: float = 10.0) -> list[BaseException]:
    """Return the failed read's errors that are still alive, once none are or after ``timeout`` seconds.

    On Zarr's loop the gather of a failed read's other chunk reads holds the
    error until those reads end, after the error reaches the caller; with
    the collector off, an error that a reference cycle keeps never goes.
    """
    deadline = time.monotonic() + timeout
    while True:
        left = [
            item
            for item in gc.get_objects()
            if isinstance(item, BaseException)
            and "the store could not read" in str(item)
        ]
        if not left or time.monotonic() > deadline:
            return left
        del left
        time.sleep(0.02)


def _failed_read_error(path: str, read: Any) -> str | None:
    """Run the coroutine ``read`` on the read path ``path``, and return the class of its error.

    The error is dropped where it is caught, in the thread that runs an
    event loop on that path, as a cell of a notebook handles it.
    """
    from cytearc.storage.async_execution import run_on_zarr_loop

    def caught() -> str | None:
        try:
            (run_read if path == "own loop" else run_on_zarr_loop)(read)
        except (OSError, ExceptionGroup) as error:
            # Only the error's class outlives this clause.
            return type(error).__name__
        return None

    if path != "inside a running loop":
        return caught()

    async def inside() -> str | None:
        try:
            # A thread that runs an event loop reads through Zarr's loop.
            run_read(read)
        except (OSError, ExceptionGroup) as error:
            return type(error).__name__
        return None

    return asyncio.run(inside())


@pytest.mark.parametrize("layout", FAILED_READS)
@pytest.mark.parametrize("scope", ["under a stream stop", "outside a stream"])
@pytest.mark.parametrize("path", ["own loop", "Zarr's loop", "inside a running loop"])
def test_a_failed_read_frees_its_output_once_its_error_is_dropped(path, scope, layout):
    import tracemalloc

    from cytearc.storage.async_execution import ReadStop, read_stop_scope

    array, selection = _failing_read(layout)
    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        with (
            read_stop_scope(ReadStop())
            if scope == "under a stream stop"
            else contextlib.nullcontext()
        ):
            raised = _failed_read_error(path, array.async_array.getitem(selection))
        # Without the collector, only reference counting frees the read's
        # output, 4 MB, and the shard-level outputs of a sharded read, so
        # anything that keeps the dropped error in a cycle keeps them all.
        traced = _traced_bytes_settle_below(FAILED_READ_CHUNK_BYTES)
        # Nor does any error of the read stay, such as the group that Zarr
        # raised inside a shard read before the one that reached the caller.
        left = _read_errors_left()
    finally:
        tracemalloc.stop()
        gc.enable()
    assert raised == (
        "ExceptionGroup" if layout == "the chunks of a shard" else "OSError"
    )
    assert traced < FAILED_READ_CHUNK_BYTES
    assert left == []


def test_an_interrupted_read_frees_its_output_once_the_interruption_is_dropped():
    import tracemalloc

    async def read() -> None:
        # The read's frame holds 4 MB while the interruption reaches it.
        output = np.ones(FAILED_READ_SHAPE, dtype=np.uint16)

        def interrupt() -> None:
            raise KeyboardInterrupt

        asyncio.get_running_loop().call_soon(interrupt)
        await asyncio.sleep(30)
        del output

    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        try:
            run_read(read())
        except KeyboardInterrupt:
            interrupted = True
        traced = _traced_bytes_settle_below(FAILED_READ_CHUNK_BYTES, timeout=2)
    finally:
        tracemalloc.stop()
        gc.enable()
    # The interruption is raised, and the cancellation that it caused in the
    # read, whose traceback reaches the read's frame, is dropped with it.
    assert interrupted
    assert traced < FAILED_READ_CHUNK_BYTES


@pytest.mark.parametrize("layout", FAILED_READS)
@pytest.mark.parametrize("path", ["own loop", "Zarr's loop"])
def test_a_failed_read_frees_its_output_while_its_error_is_still_held(path, layout):
    import tracemalloc

    from cytearc.storage.async_execution import run_on_zarr_loop

    # A notebook keeps the last error, and its traceback, until the next one,
    # as IPython's sys.last_traceback does, so the read's frames give up what
    # they hold before the error reaches the caller.
    array, selection = _failing_read(layout)
    kept: list[BaseException] = []
    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        try:
            (run_read if path == "own loop" else run_on_zarr_loop)(
                array.async_array.getitem(selection)
            )
        except (OSError, ExceptionGroup) as error:
            kept.append(error)
        traced = _traced_bytes_settle_below(FAILED_READ_CHUNK_BYTES)
    finally:
        tracemalloc.stop()
        gc.enable()
    [error] = kept
    assert "the store could not read" in repr(error)
    # The traceback keeps its lines, from the runner to the store's failure.
    assert error.__traceback__ is not None
    assert traced < FAILED_READ_CHUNK_BYTES


@pytest.mark.parametrize("reached", ["a cause", "a context", "an error of a group"])
@pytest.mark.parametrize("path", ["own loop", "Zarr's loop"])
def test_a_failed_read_frees_the_output_that_only_an_error_inside_its_error_reaches(
    path, reached
):
    import tracemalloc

    from cytearc.storage.async_execution import run_on_zarr_loop

    async def decode() -> None:
        # This frame holds 4 MB when the store's error leaves it, and only the
        # traceback of that error, not of the one the read raises, reaches it.
        output = np.ones(FAILED_READ_SHAPE, dtype=np.uint16)
        raise OSError(
            f"the store could not read counts/c/0/0, for {output.nbytes:,} bytes"
        )

    def fail_while_handling() -> None:
        raise RuntimeError("the read failed")

    async def read() -> None:
        if reached == "an error of a group":
            async with asyncio.TaskGroup() as group:
                group.create_task(decode())
        try:
            await decode()
        except OSError as error:
            if reached == "a context":
                # An error raised while the store's error is handled takes it
                # as its context.
                fail_while_handling()
            failure = error
        # An error raised from the store's error once it is handled takes it
        # as its cause, and has no context.
        raise RuntimeError("the read failed") from failure

    kept: list[BaseException] = []
    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        try:
            (run_read if path == "own loop" else run_on_zarr_loop)(read())
        except (RuntimeError, ExceptionGroup) as error:
            kept.append(error)
        traced = _traced_bytes_settle_below(FAILED_READ_CHUNK_BYTES, timeout=2)
    finally:
        tracemalloc.stop()
        gc.enable()
    [error] = kept
    if reached == "a cause":
        assert error.__context__ is None
    inner = {
        "a cause": error.__cause__,
        "a context": error.__context__,
        "an error of a group": (
            error.exceptions[0] if isinstance(error, ExceptionGroup) else None
        ),
    }[reached]
    assert isinstance(inner, OSError)
    assert inner.__traceback__ is not None
    assert traced < FAILED_READ_CHUNK_BYTES


@pytest.mark.parametrize("held", [False, True], ids=["stop dropped", "stop held"])
@pytest.mark.parametrize("path", ["own loop", "Zarr's loop"])
def test_a_stopped_read_frees_its_output_while_its_stop_is_held_and_once_dropped(
    path, held
):
    import tracemalloc

    from cytearc.storage.async_execution import (
        ReadStop,
        ReadStopped,
        read_stop_scope,
        run_on_zarr_loop,
    )

    stop = ReadStop()

    async def read() -> None:
        # The read's frame holds 4 MB while the stream's stop reaches it.
        output = np.ones(FAILED_READ_SHAPE, dtype=np.uint16)
        asyncio.get_running_loop().call_soon(stop.stop)
        await asyncio.sleep(30)
        del output

    kept: list[BaseException] = []
    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        try:
            with read_stop_scope(stop):
                (run_read if path == "own loop" else run_on_zarr_loop)(read())
        except ReadStopped as error:
            stopped = True
            # A stream drops the stop as it closes; a caller may still hold it.
            if held:
                kept.append(error)
        traced = _traced_bytes_settle_below(FAILED_READ_CHUNK_BYTES, timeout=2)
    finally:
        tracemalloc.stop()
        gc.enable()
    assert stopped
    assert traced < FAILED_READ_CHUNK_BYTES
