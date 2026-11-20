"""Reads that a memory ledger charges decode on as many threads of Zarr's codec pool as the ledger counts, on any machine."""

import contextvars
import threading
from collections.abc import Callable, Iterator
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zarr
import zarr.core.sync as zarr_sync
from zarr.codecs import BloscCodec, ZstdCodec

from cytearc.metadata.rows import read_rows, read_whole
from cytearc.storage import async_execution
from cytearc.storage.async_execution import (
    ensure_zarr_host_ceiling,
    zarr_decoding_threads,
    zarr_io_concurrency,
)
from cytearc.utils.memory_budget import MemoryLedger

#: Which reader a decode belongs to, set in each reading thread.
_READER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "reader", default=None
)

#: The host threads of the pools that each test reads beside: a count that
#: no test reads at, so that every count a test reads at has a pool of its
#: own whatever the machine.
_HOST_THREADS = 7


@pytest.fixture(autouse=True)
def _decoding_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[async_execution._DecodingPools]:
    """Make fresh decoding pools of seven host threads Zarr's, and put the process's back after the test."""
    pools = async_execution._DecodingPools(_HOST_THREADS)
    monkeypatch.setattr(zarr_sync._get_loop(), "_default_executor", pools)
    monkeypatch.setattr(zarr_sync, "_executor", pools)
    yield pools
    pools.shutdown(wait=False)


class _Decodes:
    """The thread that ran each zstd or Blosc decode, with the reader it decoded for.

    The first ``together`` decodes wait for each other, so that many threads
    decode at once, and with ``each_reader`` the first decode of each of
    those readers waits for the first of the others, so their reads are in
    flight together.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        together: int = 0,
        each_reader: tuple[str, ...] = (),
    ) -> None:
        self.seen: list[tuple[threading.Thread, str | None]] = []
        lock = threading.Lock()
        waiting = together or len(each_reader)
        barrier = threading.Barrier(waiting, timeout=20) if waiting else None
        started: set[str | None] = set()
        # Decodes since recording began, which clearing ``seen`` keeps.
        decoded = [0]

        def record() -> None:
            reader = _READER.get()
            with lock:
                self.seen.append((threading.current_thread(), reader))
                decoded[0] += 1
                if each_reader:
                    wait = reader in each_reader and reader not in started
                    started.add(reader)
                else:
                    wait = decoded[0] <= together
            if barrier is not None and wait:
                barrier.wait()

        def recording(decode):
            def run(codec, chunk_bytes, chunk_spec):
                record()
                return decode(codec, chunk_bytes, chunk_spec)

            return run

        for codec in (ZstdCodec, BloscCodec):
            monkeypatch.setattr(codec, "_decode_sync", recording(codec._decode_sync))

    def threads(self, reader: str | None = None) -> set[threading.Thread]:
        return {thread for thread, of in self.seen if reader is None or of == reader}


def _values(root: zarr.Group, n_chunks: int = 48) -> np.ndarray:
    values = np.arange(1_000 * n_chunks, dtype=np.int64)
    root.create_array(
        "values", data=values, chunks=(1_000,), compressors=ZstdCodec(level=1)
    )
    return values


def _read_whole(root: zarr.Group, threads: int) -> np.ndarray:
    ledger = MemoryLedger(None, "Reading", codec_threads=threads)
    return read_whole(ledger, root, "values", item="the values")


def _read_rows(root: zarr.Group, threads: int) -> np.ndarray:
    ledger = MemoryLedger(None, "Reading", codec_threads=threads)
    rows = np.arange(0, root["values"].shape[0], 7)
    return read_rows(ledger, root["values"], rows, item="the rows")


@pytest.mark.parametrize("read", [_read_whole, _read_rows], ids=["whole", "rows"])
def test_a_read_decodes_on_its_ledgers_threads_after_a_wider_read(
    tmp_path, monkeypatch, read
) -> None:
    # Forty-eight chunks read whole ten at a time, charged to six codec
    # threads, decode on six at once; each read after it decodes on no more
    # threads than its ledger counts, and gives the same values.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    expected = values if read is _read_whole else values[::7]
    decodes = _Decodes(monkeypatch, together=6)
    with zarr_io_concurrency(10):
        np.testing.assert_array_equal(_read_whole(root, 6), values)
        assert len(decodes.threads()) == 6
        for threads in (1, 2, 3, 1):
            decodes.seen.clear()
            np.testing.assert_array_equal(read(root, threads), expected)
            assert 1 <= len(decodes.threads()) <= threads, decodes.threads()
            assert all(f"_of_{threads}_" in t.name for t in decodes.threads())


def test_a_read_at_the_host_threads_decodes_on_zarrs_pool_as_any_other_read(
    tmp_path, monkeypatch
) -> None:
    # The default nthreads is the host's, whose reads decode on the threads
    # that every other read of the process decodes on.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    host = ensure_zarr_host_ceiling()
    assert host == _HOST_THREADS
    decodes = _Decodes(monkeypatch)
    with zarr_io_concurrency(10):
        np.testing.assert_array_equal(_read_whole(root, host), values)
        np.testing.assert_array_equal(root["values"][...], values)
    pool = zarr_sync._get_loop()._default_executor
    assert decodes.threads() <= set(pool._threads)
    assert not any("_of_" in thread.name for thread in decodes.threads())


@pytest.mark.parametrize(
    "counts",
    [(1, 3), (2, 2), (3, 1, 3)],
    ids=["one-beside-three", "two-beside-two", "three-one-three"],
)
def test_reads_in_flight_together_decode_on_the_pool_of_their_own_count(
    tmp_path, monkeypatch, counts
) -> None:
    # Each reader's first decode waits for the first of the others, so the
    # reads are in flight together. Each decodes on no more threads than its
    # ledger counts; readers of one count share one pool of that many
    # threads, and readers of different counts share no thread.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    readers = tuple(f"reader {index}" for index in range(len(counts)))
    decodes = _Decodes(monkeypatch, each_reader=readers)
    results: dict[str, np.ndarray] = {}

    def reading(reader: str, threads: int) -> None:
        _READER.set(reader)
        with zarr_io_concurrency(4):
            results[reader] = _read_whole(root, threads)

    callers = [
        threading.Thread(target=reading, args=(reader, threads))
        for reader, threads in zip(readers, counts, strict=True)
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=60)
    assert sorted(results) == sorted(readers)
    for reader in readers:
        np.testing.assert_array_equal(results[reader], values)
    by_count: dict[int, set[threading.Thread]] = {}
    for reader, threads in zip(readers, counts, strict=True):
        assert 1 <= len(decodes.threads(reader)) <= threads
        by_count.setdefault(threads, set()).update(decodes.threads(reader))
    for threads, decoding in by_count.items():
        assert len(decoding) <= threads
    pools = list(by_count.values())
    assert all(a.isdisjoint(b) for i, a in enumerate(pools) for b in pools[i + 1 :])


def test_a_pool_that_replaced_cytearcs_is_replaced_before_the_next_charged_read(
    tmp_path, monkeypatch, _decoding_pools
) -> None:
    # A pool set on Zarr's loop after CyteArc's, as a library or Zarr itself
    # can set one, gives way to CyteArc's when a charged read begins.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    ensure_zarr_host_ceiling()
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(loop, "_default_executor", loop._default_executor)
    decodes = _Decodes(monkeypatch)
    with ThreadPoolExecutor(8, thread_name_prefix="foreign") as foreign:
        loop.set_default_executor(foreign)
        with zarr_io_concurrency(10):
            np.testing.assert_array_equal(_read_whole(root, 1), values)
        assert len(decodes.threads()) == 1
        assert not any(t.name.startswith("foreign") for t in decodes.threads())
        assert loop._default_executor is _decoding_pools


def test_the_installed_pool_keeps_the_size_of_the_pool_zarr_ran_before(
    monkeypatch,
) -> None:
    # A pool that Zarr's loop ran before CyteArc's first use sets the host
    # threads, and CyteArc's pools take its place with as many.
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(loop, "_default_executor", None)
    monkeypatch.setattr(zarr_sync, "_executor", None)
    with ThreadPoolExecutor(5) as earlier:
        loop.set_default_executor(earlier)
        assert ensure_zarr_host_ceiling() == 5
        pools = loop._default_executor
        assert isinstance(pools, async_execution._DecodingPools)
        assert pools._max_workers == 5
        assert zarr_sync._executor is pools
        assert ensure_zarr_host_ceiling() == 5
        assert loop._default_executor is pools
        pools.shutdown()


def test_the_pools_on_zarrs_loop_become_zarrs_pool_when_zarr_holds_none(
    monkeypatch, _decoding_pools
) -> None:
    # Zarr holds no pool of its own while CyteArc's pools run on its loop.
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(zarr_sync, "_executor", None)
    assert loop._default_executor is _decoding_pools

    assert ensure_zarr_host_ceiling() == _HOST_THREADS

    assert zarr_sync._executor is _decoding_pools
    # So Zarr makes no pool of its own to replace them on the loop.
    assert zarr_sync._get_executor() is _decoding_pools
    assert loop._default_executor is _decoding_pools


def test_a_reinstall_finds_the_pools_it_made_beside_zarrs_own(monkeypatch) -> None:
    # Where Zarr made its own pool first, the pools that take its place on
    # the loop become Zarr's too, so that once another executor replaces
    # them, the next install finds the same pools instead of making more.
    loop = zarr_sync._get_loop()
    with (
        ThreadPoolExecutor(3) as zarrs,
        ThreadPoolExecutor(3) as foreign,
        ThreadPoolExecutor(3) as later,
    ):
        monkeypatch.setattr(zarr_sync, "_executor", zarrs)
        loop.set_default_executor(foreign)
        assert ensure_zarr_host_ceiling() == 3
        pools = loop._default_executor
        assert isinstance(pools, async_execution._DecodingPools)
        assert zarr_sync._executor is pools
        loop.set_default_executor(later)
        assert ensure_zarr_host_ceiling() == 3
        assert loop._default_executor is pools
        pools.shutdown()


def test_the_repertoire_operations_decode_their_own_charged_reads_on_the_stores_threads(
    tmp_path, monkeypatch
) -> None:
    # The reads that the repertoire operations charge to a ledger outside
    # read_whole and read_rows, of the contigs' text and of the export's
    # passes, decode on the two threads of a store of nthreads=2.
    from cytearc.datastore._operations import repertoire
    from tests.clone_helpers import (
        CELLRANGER_HEADER,
        CLONAL_ROWS,
        receptor_store,
        write_lines,
    )

    def labeled(function, label):
        def run(*args, **kwargs):
            token = _READER.set(label)
            try:
                return function(*args, **kwargs)
            finally:
                _READER.reset(token)

        return run

    for name in ("_decoded_values", "_value_lengths"):
        monkeypatch.setattr(repertoire, name, labeled(getattr(repertoire, name), name))
    for name in ("measure", "scan"):
        method = getattr(repertoire._ExportContigs, name)
        monkeypatch.setattr(repertoire._ExportContigs, name, labeled(method, name))
    store = receptor_store(tmp_path, nthreads=2)
    path = write_lines(tmp_path, "clonal.csv", CELLRANGER_HEADER, CLONAL_ROWS)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    decodes = _Decodes(monkeypatch)
    clonotypes = store.repertoire.clonotypes(receptors, sequence="aa")
    store.repertoire.clonotypes(receptors, metric="hamming", max_distance=1)
    store.repertoire.spectratype(receptors)
    store.repertoire.export_airr(clonotypes, tmp_path / "contigs.tsv")
    for label in ("_decoded_values", "_value_lengths", "measure", "scan"):
        decoding = {thread.name for thread in decodes.threads(label)}
        assert decoding, label
        assert all("_of_2_" in name for name in decoding), (label, decoding)


def test_nested_scopes_decode_on_the_innermost_count_and_restore_the_outer(
    tmp_path, monkeypatch
) -> None:
    # A scope left by an error restores the count of the scope around it,
    # and leaving that restores the host threads.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    decodes = _Decodes(monkeypatch)

    def decoded_on() -> set[str]:
        decodes.seen.clear()
        np.testing.assert_array_equal(root["values"][...], values)
        return {thread.name for thread in decodes.threads()}

    with zarr_io_concurrency(10), zarr_decoding_threads(3):
        with pytest.raises(KeyError), zarr_decoding_threads(1):
            inner = decoded_on()
            raise KeyError("left early")
        outer = decoded_on()
    outside = decoded_on()
    assert inner and all("_of_1_" in name for name in inner)
    assert outer and all("_of_3_" in name for name in outer)
    assert outside and not any("_of_" in name for name in outside)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2", None])
def test_decoding_threads_are_a_whole_number_of_at_least_one(threads) -> None:
    with pytest.raises(ValueError, match="whole number of threads of at least 1"):
        with zarr_decoding_threads(threads):
            pytest.fail("the scope must not open")


def test_shutting_the_pools_down_shuts_down_the_pool_of_every_count() -> None:
    pools = async_execution._DecodingPools(2)
    for threads in (1, 3):
        with zarr_decoding_threads(threads):
            assert (
                pools.submit(threading.current_thread)
                .result(5)
                .name.startswith(f"zarr_pool_of_{threads}_")
            )
    assert (
        pools.submit(threading.current_thread).result(5).name.startswith("zarr_pool_")
    )
    pools.shutdown()
    for threads in (None, 1, 3, 5):
        token = async_execution._DECODING_THREADS.set(threads)
        try:
            with pytest.raises(RuntimeError, match="after shutdown"):
                pools.submit(threading.current_thread)
        finally:
            async_execution._DECODING_THREADS.reset(token)


# A pool thread that has reported a call's result.


class _SetResultPause:
    """Pause the pool thread of an armed call just after it sets the call's result.

    The thread waits inside ``set_result``, once the future's callbacks have
    told the event loop, until the test releases it, as a thread that the OS
    deschedules there would: it still holds the call, its future and the
    result it returned. A call arms the pause of the thread that runs it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import concurrent.futures.thread as pool_thread

        self._armed = threading.local()
        self.paused = threading.Event()
        self.release = threading.Event()
        run = pool_thread._WorkItem.run
        pause = self

        def paused_run(item, *args):  # type: ignore[no-untyped-def]
            future = item.future
            set_result = future.set_result

            def set_result_then_pause(result):  # type: ignore[no-untyped-def]
                set_result(result)
                if getattr(pause._armed, "on", False):
                    pause._armed.on = False
                    pause.paused.set()
                    assert pause.release.wait(timeout=20)

            future.set_result = set_result_then_pause
            try:
                run(item, *args)
            finally:
                del future.set_result

        monkeypatch.setattr(pool_thread._WorkItem, "run", paused_run)

    def arm(self) -> None:
        self._armed.on = True


@pytest.fixture
def without_cyclic_collection() -> Iterator[None]:
    """Free objects by their references alone, so that one a cycle holds stays alive."""
    import gc

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class _Chunk:
    """Stands in for a decoded chunk: an object that a weak reference can follow."""


def _chunk_outlives_its_reader(
    pool: ThreadPoolExecutor, pause: _SetResultPause
) -> bool:
    """Whether a pool thread that pauses after returning a chunk keeps it once its reader let go."""
    import asyncio
    import weakref

    returned: list[weakref.ref[_Chunk]] = []

    def decode() -> _Chunk:
        pause.arm()
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    async def read() -> bool:
        chunk = await asyncio.get_running_loop().run_in_executor(pool, decode)
        assert isinstance(chunk, _Chunk)
        del chunk
        # The loop lets go of the callback that resumed this read, and its
        # future, once the read yields.
        await asyncio.sleep(0)
        # The thread that returned the chunk still waits in set_result.
        assert pause.paused.wait(timeout=20)
        return returned[0]() is not None

    try:
        return asyncio.run(read())
    finally:
        pause.release.set()


@pytest.mark.parametrize(
    ("pool", "keeps"),
    [
        (lambda: ThreadPoolExecutor(2), True),
        (lambda: async_execution._ReleasingPool(2), False),
        (lambda: async_execution._DecodingPools(2), False),
        (lambda: async_execution._DecodingPools(3), False),
    ],
    ids=["a plain pool", "a releasing pool", "the host threads", "a counted pool"],
)
def test_a_pool_thread_paused_after_its_result_keeps_no_chunk_its_reader_dropped(
    monkeypatch, without_cyclic_collection, pool, keeps
) -> None:
    # A plain pool's thread keeps the chunk that it returned until it runs
    # again; CyteArc's pools hand the chunk to the reader's loop, so the
    # paused thread holds nothing that reaches it. The decoding pools of
    # three threads decode within zarr_decoding_threads(2) on a pool of two.
    pause = _SetResultPause(monkeypatch)
    executor = pool()
    try:
        with zarr_decoding_threads(2):
            assert _chunk_outlives_its_reader(executor, pause) is keeps
    finally:
        executor.shutdown(wait=True)


def _read_through_run_read(
    array: zarr.Array, check: Callable[[np.ndarray], None]
) -> None:
    from cytearc.storage.virtual_arrays import read_selection

    with zarr_decoding_threads(2):
        values = read_selection(array, slice(None), slice(None))
    check(values)


def _read_through_zarrs_loop(
    array: zarr.Array, check: Callable[[np.ndarray], None]
) -> None:
    check(np.asarray(array[...]))


def _read_through_a_storage_runner(
    array: zarr.Array, check: Callable[[np.ndarray], None]
) -> None:
    from dataclasses import replace

    from cytearc.storage.async_execution import AsyncStorageRunner
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.execution import WorkShape, plan_operation

    # Two codec threads on any machine, so that one can pause.
    plan = plan_operation(
        ResourceBudget(64 * 1024**2, 2), WorkShape(nUnits=2, unitBytes=1)
    )
    runner = AsyncStorageRunner(operation=replace(plan, codecWorkers=2))

    async def operation(active: AsyncStorageRunner) -> None:
        values = await active.io(array.async_array.getitem((slice(None), slice(None))))
        # The runner waits for its codec threads as it ends, so the read is
        # checked before then.
        check(np.asarray(values))

    runner.run(operation)


@pytest.mark.parametrize(
    "read",
    [_read_through_run_read, _read_through_zarrs_loop, _read_through_a_storage_runner],
    ids=["a read on its own loop", "Zarr's loop", "a storage runner"],
)
def test_a_read_leaves_no_decoded_chunk_with_a_paused_pool_thread(
    tmp_path, monkeypatch, without_cyclic_collection, read
) -> None:
    # The thread of the first decode pauses after it returns the chunk, as
    # a busy machine can deschedule it; the read decodes its other chunks on
    # other threads and copies every chunk into its result. Once the read has
    # returned, no decoded chunk is left, though that thread has not run again.
    import weakref

    values = np.arange(8_000, dtype=np.int64).reshape(80, 100)
    array = zarr.open_group(tmp_path / "store.zarr", mode="w").create_array(
        "values", data=values, chunks=(10, 100), compressors=ZstdCodec(level=1)
    )
    pause = _SetResultPause(monkeypatch)
    decoded: list[weakref.ref[Any]] = []
    lock = threading.Lock()
    decode = ZstdCodec._decode_sync

    def recording(codec, chunk_bytes, chunk_spec):  # type: ignore[no-untyped-def]
        chunk = decode(codec, chunk_bytes, chunk_spec)
        with lock:
            if not decoded:
                pause.arm()
            decoded.append(weakref.ref(chunk))
        return chunk

    monkeypatch.setattr(ZstdCodec, "_decode_sync", recording)
    checked: list[bool] = []

    def check(result: np.ndarray) -> None:
        try:
            np.testing.assert_array_equal(result, values)
            assert pause.paused.wait(timeout=20)
            assert len(decoded) == 8
            assert [chunk() for chunk in decoded] == [None] * 8
            checked.append(True)
        finally:
            pause.release.set()

    try:
        read(array, check)
    finally:
        pause.release.set()
    assert checked == [True]


def _held_call(started: threading.Event, release: threading.Event):  # type: ignore[no-untyped-def]
    """A call that holds its pool thread until ``release`` is set."""

    def call() -> str:
        started.set()
        assert release.wait(timeout=20)
        return "held"

    return call


def test_a_call_that_raises_reaches_its_reader_through_a_releasing_pool() -> None:
    import asyncio

    def failing() -> None:
        raise KeyError("no chunk at 3")

    async def read() -> None:
        loop = asyncio.get_running_loop()
        with pytest.raises(KeyError, match="no chunk at 3"):
            await loop.run_in_executor(pool, failing)
        assert await loop.run_in_executor(pool, lambda: "next") == "next"

    with async_execution._ReleasingPool(1) as pool:
        asyncio.run(read())


def test_a_queued_call_whose_reader_was_cancelled_never_runs() -> None:
    import asyncio

    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    async def read() -> None:
        loop = asyncio.get_running_loop()
        holding = loop.run_in_executor(pool, _held_call(started, release))
        queued = asyncio.ensure_future(
            loop.run_in_executor(pool, lambda: ran.append("queued"))
        )
        assert started.wait(timeout=20)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        # The loop cancels the call as it runs the cancelled future's
        # callbacks, before the held call ends.
        await asyncio.sleep(0)
        release.set()
        assert await holding == "held"
        # The pool runs another call after the one it held, not the cancelled one.
        await loop.run_in_executor(pool, lambda: ran.append("later"))

    with async_execution._ReleasingPool(1) as pool:
        try:
            asyncio.run(read())
        finally:
            release.set()
    assert ran == ["later"]


def test_a_started_call_whose_reader_was_cancelled_runs_on_and_keeps_nothing(
    without_cyclic_collection,
) -> None:
    import asyncio
    import weakref

    started = threading.Event()
    release = threading.Event()
    returned: list[weakref.ref[_Chunk]] = []

    def decode() -> _Chunk:
        started.set()
        assert release.wait(timeout=20)
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    async def read() -> None:
        loop = asyncio.get_running_loop()
        reading = asyncio.ensure_future(loop.run_in_executor(pool, decode))
        assert started.wait(timeout=20)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        # The call has started, so it runs to its end, and the loop drops the
        # chunk it hands back.
        release.set()
        assert await loop.run_in_executor(pool, lambda: "next") == "next"
        await asyncio.sleep(0)

    with async_execution._ReleasingPool(1) as pool:
        try:
            asyncio.run(read())
        finally:
            release.set()
    assert len(returned) == 1
    assert returned[0]() is None


def test_a_pool_that_shuts_down_cancels_the_reader_of_a_queued_call() -> None:
    import asyncio

    started = threading.Event()
    release = threading.Event()

    async def read() -> None:
        loop = asyncio.get_running_loop()
        holding = loop.run_in_executor(pool, _held_call(started, release))
        queued = loop.run_in_executor(pool, lambda: "queued")
        assert started.wait(timeout=20)
        pool.shutdown(wait=False, cancel_futures=True)
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await holding == "held"

    pool = async_execution._ReleasingPool(1)
    try:
        asyncio.run(read())
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_a_call_whose_loop_closed_keeps_nothing_once_it_ends(
    monkeypatch, without_cyclic_collection
) -> None:
    import asyncio
    import weakref

    started = threading.Event()
    release = threading.Event()
    returned: list[weakref.ref[_Chunk]] = []
    pause = _SetResultPause(monkeypatch)

    def decode() -> _Chunk:
        started.set()
        assert release.wait(timeout=20)
        pause.arm()
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    async def submit() -> Any:
        return pool.submit(decode)

    with async_execution._ReleasingPool(1) as pool:
        loop = asyncio.new_event_loop()
        try:
            future = loop.run_until_complete(submit())
            assert started.wait(timeout=20)
        finally:
            loop.close()
        release.set()
        try:
            # Nothing awaits the call once its loop has closed, so its chunk
            # goes as the call ends, while its thread waits in set_result.
            assert pause.paused.wait(timeout=20)
            assert len(returned) == 1
            assert returned[0]() is None
        finally:
            pause.release.set()
    assert not future.done()


def test_a_read_does_not_wait_for_a_call_that_ended_before_it_looked() -> None:
    import asyncio
    import time

    class EndingPool(async_execution._ReleasingPool):
        """Return each future of an event loop's call once the call has ended and gone."""

        def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            future = super().submit(fn, *args, **kwargs)
            deadline = time.monotonic() + 20
            while future.call() is not None:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            return future

    async def read() -> int:
        value = await asyncio.get_running_loop().run_in_executor(lent, lambda: 42)
        assert lent._calls == set()
        lent.wait_for_calls()
        return value

    with EndingPool(1) as pool:
        lent = async_execution._LentPool(pool)
        assert asyncio.run(read()) == 42
