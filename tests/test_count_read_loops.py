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
import functools
import gc
import http.server
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
