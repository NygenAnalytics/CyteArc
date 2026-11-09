"""Closing a block stream stops its running reads.

``stream_shards`` gives each threaded stream a ``ReadStop``, set in a context
variable in the threads that run its reads. Closing the stream stops them: a
read's task is cancelled on the event loop that runs it, the read's own loop
or Zarr's, and a read stops before the next operation of its block. The stream
still returns from ``close()`` only after its reads have ended, and it drops
the ``ReadStopped`` that they raise, while their real errors still raise. A
cache fill or listing that a stopped read started runs to its end for the
reads that wait for it. A stream started within a read stops with the stream
that runs the read, and a stream running beside it never stops.

The tests hold reads at a ``Gate`` and require ``close()`` to return while the
gate is still closed; an implementation that waits for the reads instead fails
when the deadline passes, after which the gate opens so nothing hangs.
"""

import asyncio
import contextvars
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import zarr
import zarr.core.sync as zarr_sync
from zarr.core.sync import SyncError
from zarr.storage import MemoryStore, ZipStore

from cytearc.composite.source_cache import SourceCache
from cytearc.matrix import ChunkedArray
from cytearc.storage import async_execution
from cytearc.storage.async_execution import (
    ReadStop,
    ReadStopped,
    read_stop_scope,
    run_on_zarr_loop,
    run_read,
)
from cytearc.storage.huggingface import _SharedFetches
from cytearc.storage.parallel import stream_shards
from cytearc.storage.virtual_arrays import read_selection
from cytearc.utils.shutdown import ShutdownRequest, ShutdownRequested
from tests.test_composite_source_cache import (
    BODY,
    GatedSource,
    populated_source,
    wait_until,
)

VALUES = np.arange(32 * 12, dtype=np.uint16).reshape(32, 12)
# Each stream item reads one row chunk of eight rows.
ROWS = 8


class Gate:
    """An await point that holds reads until the test opens it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.threads: list[str] = []

    async def wait(self) -> None:
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        try:
            async with asyncio.timeout(20):
                while not self.release.is_set():
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _close(stream: Iterator[Any], *gates: Gate) -> list[BaseException]:
    """Close ``stream`` while ``gates`` hold its reads; return what ``close()`` raised.

    Fails when ``close()`` does not return while the gates are still closed.
    """
    raised: list[BaseException] = []

    def close() -> None:
        try:
            stream.close()  # type: ignore[attr-defined]
        except BaseException as error:
            raised.append(error)

    closer = threading.Thread(target=close)
    closer.start()
    closer.join(timeout=5)
    returned = not closer.is_alive()
    for gate in gates:
        gate.release.set()
    closer.join(timeout=30)
    assert returned, "close() waited for the held reads to finish"
    return raised


def _gate_chunk_rows(monkeypatch, store_type: type, gate: Gate) -> None:
    """Hold every read of ``store_type`` of a chunk below the first row chunk."""
    original = store_type.get

    async def get(self, key, prototype, byte_range=None):  # type: ignore[no-untyped-def]
        if "/c/" in key and int(key.split("/c/")[1].split("/")[0]) > 0:
            await gate.wait()
        return await original(self, key, prototype, byte_range)

    monkeypatch.setattr(store_type, "get", get)


def _memory_counts() -> zarr.Array:
    return zarr.create_array(
        store=MemoryStore(), name="counts", data=VALUES, chunks=(ROWS, 6)
    )


def _row_chunks(array: zarr.Array) -> Callable[[int], np.ndarray]:
    return lambda item: read_selection(
        array, slice(ROWS * item, ROWS * (item + 1)), slice(None)
    )


def test_close_stops_a_read_on_its_own_event_loop(monkeypatch):
    array = _memory_counts()
    gate = Gate()
    _gate_chunk_rows(monkeypatch, MemoryStore, gate)
    stream = stream_shards(range(4), _row_chunks(array), workers=2)

    np.testing.assert_array_equal(next(stream), VALUES[:ROWS])
    assert gate.entered.wait(timeout=5)

    assert _close(stream, gate) == []
    assert gate.cancelled.is_set()
    # The held read ran on its own loop, in the stream's reader thread.
    assert gate.threads and all(name != "zarr_io" for name in gate.threads)


def test_close_stops_a_read_of_a_store_that_reads_on_zarrs_loop(tmp_path, monkeypatch):
    array = _zip_counts(tmp_path)
    gate = Gate()
    _gate_chunk_rows(monkeypatch, ZipStore, gate)
    stream = stream_shards(range(4), _row_chunks(array), workers=2)

    np.testing.assert_array_equal(next(stream), VALUES[:ROWS])
    assert gate.entered.wait(timeout=5)

    assert _close(stream, gate) == []
    assert gate.cancelled.is_set()
    # A zip store is not known to work on any loop, so the read ran on Zarr's.
    assert gate.threads and set(gate.threads) == {"zarr_io"}


def test_close_stops_a_read_made_from_a_thread_that_runs_an_event_loop(monkeypatch):
    array = _memory_counts()
    gate = Gate()
    _gate_chunk_rows(monkeypatch, MemoryStore, gate)
    read = _row_chunks(array)

    def read_inside_a_loop(item: int) -> np.ndarray:
        async def inside() -> np.ndarray:
            return read(item)

        return asyncio.run(inside())

    stream = stream_shards(range(4), read_inside_a_loop, workers=2)

    np.testing.assert_array_equal(next(stream), VALUES[:ROWS])
    assert gate.entered.wait(timeout=5)

    assert _close(stream, gate) == []
    assert gate.cancelled.is_set()
    # A thread that runs an event loop reads through Zarr's loop.
    assert gate.threads and set(gate.threads) == {"zarr_io"}


def _zip_counts(tmp_path) -> zarr.Array:
    path = tmp_path / "counts.zip"
    with ZipStore(path, mode="w") as writable:
        zarr.create_array(store=writable, name="counts", data=VALUES, chunks=(ROWS, 6))
    return zarr.open_array(store=ZipStore(path, mode="r"), path="counts", mode="r")


@pytest.mark.parametrize("path", ["own loop", "Zarr's loop", "inside a running loop"])
def test_close_stops_a_read_before_the_next_operation_of_its_block(
    path, tmp_path, monkeypatch
):
    first = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    later: list[float] = []
    original_stop = ReadStop.stop

    def stop(self: ReadStop) -> None:
        original_stop(self)
        stopped.set()

    monkeypatch.setattr(ReadStop, "stop", stop)

    def slow_first(values: np.ndarray) -> np.ndarray:
        # An operation that runs cannot be interrupted, so the second block's
        # first operation runs until the test lets it finish.
        if values.flat[0] >= VALUES[ROWS, 0]:
            first.set()
            assert release.wait(timeout=20)
        return values

    def record_second(values: np.ndarray) -> np.ndarray:
        later.append(float(values.flat[0]))
        return values

    # The block's counts arrive through each read path before its operations
    # run; an open gate records the threads that read the later row chunks.
    backing = _zip_counts(tmp_path) if path == "Zarr's loop" else _memory_counts()
    reads = Gate()
    reads.release.set()
    _gate_chunk_rows(monkeypatch, type(backing.store), reads)
    counts = ChunkedArray(backing, block_size=ROWS)
    counts = counts._unary(slow_first)._unary(record_second)
    if path == "inside a running loop":

        def block(item: int) -> np.ndarray:
            async def inside() -> np.ndarray:
                return counts._materialize_range(ROWS * item, ROWS * (item + 1))

            return asyncio.run(inside())

        stream = stream_shards(range(4), block, workers=2)
    else:
        stream = counts.stream_blocks(nthreads=2, prefetch=2)
    np.testing.assert_array_equal(next(stream), VALUES[:ROWS])
    assert first.wait(timeout=5)

    raised: list[BaseException] = []

    def close() -> None:
        try:
            stream.close()
        except BaseException as error:
            raised.append(error)

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert stopped.wait(timeout=5)
    finally:
        release.set()
        closer.join(timeout=30)
    assert not closer.is_alive()
    assert raised == []
    # The second block stopped after its first operation, before its second.
    assert later == [float(VALUES[0, 0])]
    if path == "own loop":
        assert reads.threads and "zarr_io" not in reads.threads
    else:
        assert reads.threads and set(reads.threads) == {"zarr_io"}


def test_a_cancellation_that_no_stop_made_keeps_its_meaning():
    async def cancelled() -> None:
        raise asyncio.CancelledError

    with read_stop_scope(ReadStop()), pytest.raises(asyncio.CancelledError):
        run_read(cancelled())


def test_an_interruption_keeps_its_meaning_under_a_stream_stop():
    async def interrupted() -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        asyncio.get_running_loop().call_soon(interrupt)
        await asyncio.sleep(30)

    with read_stop_scope(ReadStop()), pytest.raises(KeyboardInterrupt):
        run_read(interrupted())


def test_a_stop_cancels_a_read_on_zarrs_loop_and_waits_for_it_to_end(monkeypatch):
    gate = Gate()
    stop = ReadStop()
    unwinding = threading.Event()
    finish_unwinding = threading.Event()
    unwound = threading.Event()
    # Set once the reader waits in Zarr's sync(), or has returned.
    settled = threading.Event()
    outcome: list[tuple[BaseException, bool]] = []
    original_sync = zarr_sync.sync

    def sync(*args: Any, **kwargs: Any) -> Any:
        settled.set()
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(zarr_sync, "sync", sync)

    async def held() -> None:
        try:
            await gate.wait()
        finally:
            # The cancelled read unwinds only when the test lets it.
            unwinding.set()
            async with asyncio.timeout(20):
                while not finish_unwinding.is_set():
                    await asyncio.sleep(0.001)
            unwound.set()

    def read() -> None:
        with read_stop_scope(stop):
            try:
                run_on_zarr_loop(held())
            except BaseException as error:
                outcome.append((error, unwound.is_set()))
            finally:
                settled.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert gate.entered.wait(timeout=5)
        stop.stop()
        assert unwinding.wait(timeout=5)
        assert settled.wait(timeout=5)
        finish_unwinding.set()
        reader.join(timeout=5)
        assert not reader.is_alive()
    finally:
        gate.release.set()
        finish_unwinding.set()
        reader.join(timeout=30)
    [(error, ended)] = outcome
    assert isinstance(error, ReadStopped)
    # The read had ended on Zarr's loop before the reader saw ReadStopped.
    assert ended


def test_a_read_from_zarrs_loop_thread_raises_under_a_stop_instead_of_blocking(
    monkeypatch,
):
    # Zarr's pool exists before its loop is replaced, so Zarr's sync() attaches
    # no pool to the stand-in loop.
    async_execution.ensure_zarr_host_ceiling()
    # A loop of the test's own stands in for Zarr's, so a read that blocked
    # the loop's thread would wedge only this loop.
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(zarr_sync, "_get_loop", lambda: loop)

    async def value() -> int:
        return 1

    async def read_on_the_loop() -> int:
        read = value()
        try:
            return run_read(read)
        finally:
            read.close()

    with read_stop_scope(ReadStop()):
        future = asyncio.run_coroutine_threadsafe(read_on_the_loop(), loop)
    try:
        with pytest.raises(SyncError):
            future.result(timeout=5)
    finally:
        if future.done():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()


def _finishing(monkeypatch) -> threading.Event:
    """Return an event set when a read's loop starts running the tasks its read left."""
    finishing = threading.Event()
    original = async_execution._finish_tasks

    def finish(loop: asyncio.AbstractEventLoop) -> None:
        finishing.set()
        original(loop)

    monkeypatch.setattr(async_execution, "_finish_tasks", finish)
    return finishing


def _closing(stream: Iterator[Any], release: threading.Event, returns: list[bool]):
    """Return a thread that closes ``stream`` and records whether ``release`` was set by then."""

    def close() -> None:
        stream.close()  # type: ignore[attr-defined]
        returns.append(release.is_set())

    return threading.Thread(target=close)


def test_a_stopped_read_lets_the_shared_fill_it_started_finish(monkeypatch):
    finishing = _finishing(monkeypatch)
    source = populated_source(store=GatedSource())
    source.release.clear()
    cache = SourceCache(capacity_bytes=256, max_entries=4)
    cached = cache.wrap(source)

    def prefetch(item: int) -> bool:
        if item == 0:
            return True
        return run_read(cached.prefetch("shard", len(BODY)))

    async def waiting_read() -> bool:
        await wait_until(source.entered.is_set)
        return await cached.prefetch("shard", len(BODY))

    stream = stream_shards(range(3), prefetch, workers=2)
    pool = ThreadPoolExecutor(max_workers=1)
    returns: list[bool] = []
    closer = _closing(stream, source.release, returns)
    try:
        assert next(stream) is True
        assert source.entered.wait(timeout=5)
        waiting = pool.submit(run_read, waiting_read())

        async def joined() -> None:
            await wait_until(
                lambda: any(entry.waiters == 2 for entry in cache._entries.values())
            )

        asyncio.run(joined())
        closer.start()
        # The stopped read has ended, and its loop runs the tasks it left: the
        # fill, which cannot progress until the source opens.
        assert finishing.wait(timeout=5)
        source.release.set()
        assert waiting.result(timeout=5) is True
        closer.join(timeout=10)
        assert not closer.is_alive()
    finally:
        source.release.set()
        pool.shutdown(wait=True)
        if closer.ident is not None:
            closer.join(timeout=10)
        cache.close()
    # The fill was never cancelled, one fetch served both reads, and close()
    # returned only after the fill had ended.
    assert source.cancelled == 0
    assert source.attempts == 1
    assert returns == [True]


def test_a_stopped_read_lets_the_shared_listing_it_started_finish(monkeypatch):
    finishing = _finishing(monkeypatch)
    fetches = _SharedFetches[str]()
    listing = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    async def fetch() -> str:
        calls.append(1)
        listing.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        return "listing"

    def list_store(item: int) -> str:
        if item == 0:
            return "first"
        return run_read(fetches.get("", fetch))

    async def waiting_read() -> str:
        await wait_until(listing.is_set)
        return await fetches.get("", fetch)

    stream = stream_shards(range(3), list_store, workers=2)
    pool = ThreadPoolExecutor(max_workers=1)
    returns: list[bool] = []
    closer = _closing(stream, release, returns)
    try:
        assert next(stream) == "first"
        assert listing.wait(timeout=5)
        waiting = pool.submit(run_read, waiting_read())
        closer.start()
        # The stopped read has ended, and its loop runs the listing it started.
        assert finishing.wait(timeout=5)
        release.set()
        assert waiting.result(timeout=5) == "listing"
        closer.join(timeout=10)
        assert not closer.is_alive()
    finally:
        release.set()
        pool.shutdown(wait=True)
        if closer.ident is not None:
            closer.join(timeout=10)
    # One listing served both reads, and close() returned only after it ended.
    assert calls == [1]
    assert returns == [True]


def test_a_stopped_read_returns_after_the_call_it_left_on_zarrs_pool_ends(monkeypatch):
    array = _memory_counts()
    in_pool = threading.Event()
    release = threading.Event()
    call_ended = threading.Event()
    # Set once the stopped read waits for its call, or has returned.
    settled = threading.Event()
    reader: list[int] = []
    waited: list[bool] = []
    returned: list[bool] = []

    def held_call() -> None:
        in_pool.set()
        try:
            assert release.wait(timeout=20)
        finally:
            call_ended.set()

    original_get = MemoryStore.get

    async def get(self, key, prototype, byte_range=None):  # type: ignore[no-untyped-def]
        # Reading the second row chunk's first chunk holds a thread of the pool.
        if key.endswith("/c/1/0"):
            await asyncio.get_running_loop().run_in_executor(None, held_call)
        return await original_get(self, key, prototype, byte_range)

    monkeypatch.setattr(MemoryStore, "get", get)
    original_wait = async_execution._LentPool.wait_for_calls

    def wait_for_calls(self: Any) -> None:
        holding = bool(reader) and threading.get_ident() == reader[0]
        if holding:
            settled.set()
        original_wait(self)
        if holding:
            waited.append(call_ended.is_set())

    monkeypatch.setattr(async_execution._LentPool, "wait_for_calls", wait_for_calls)
    read = _row_chunks(array)

    def read_rows(item: int) -> np.ndarray:
        if item != 1:
            return read(item)
        reader.append(threading.get_ident())
        try:
            return read(item)
        finally:
            returned.append(call_ended.is_set())
            settled.set()

    stream = stream_shards(range(4), read_rows, workers=2)
    raised: list[BaseException] = []

    def close() -> None:
        try:
            stream.close()
        except BaseException as error:
            raised.append(error)

    closer = threading.Thread(target=close)
    try:
        np.testing.assert_array_equal(next(stream), VALUES[:ROWS])
        # The second read's task awaits a call that runs on the pool.
        assert in_pool.wait(timeout=5)
        closer.start()
        assert settled.wait(timeout=5)
        release.set()
        closer.join(timeout=10)
        assert not closer.is_alive()
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=10)
    assert raised == []
    # The stop cancelled the read's task, and the read returned only after the
    # call it had left running on the pool ended.
    assert waited == [True]
    assert returned == [True]


def _failing_when_stopped(gate: Gate, error: Callable[[], BaseException]):
    def read(item: int) -> int:
        if item == 0:
            return 0

        async def held() -> int:
            try:
                await gate.wait()
            except asyncio.CancelledError:
                raise error() from None
            return item

        return run_read(held())

    return read


def test_a_real_failure_of_a_stopped_read_still_raises_from_close():
    gate = Gate()
    read = _failing_when_stopped(gate, lambda: ValueError("the read failed"))
    stream = stream_shards(range(3), read, workers=2)
    assert next(stream) == 0
    assert gate.entered.wait(timeout=5)

    [error] = _close(stream, gate)
    assert isinstance(error, ValueError)
    assert str(error) == "the read failed"


def test_real_failures_of_stopped_reads_raise_in_one_group_without_the_stops():
    gates = [Gate(), Gate(), Gate()]

    def read(item: int) -> int:
        if item == 0:
            return 0
        gate = gates[item - 1]

        async def held() -> int:
            try:
                await gate.wait()
            except asyncio.CancelledError:
                if item < 3:
                    raise ValueError(f"read {item} failed") from None
                raise
            return item

        return run_read(held())

    stream = stream_shards(range(4), read, workers=4)
    assert next(stream) == 0
    for gate in gates:
        assert gate.entered.wait(timeout=5)

    [group] = _close(stream, *gates)
    assert isinstance(group, BaseExceptionGroup)
    assert group.message == "Block stream failed during cleanup"
    assert sorted(str(error) for error in group.exceptions) == [
        "read 1 failed",
        "read 2 failed",
    ]


def test_a_shutdown_request_from_a_stopped_read_still_raises_from_close():
    gate = Gate()
    request = ShutdownRequest(requested_at_ns=1, reason="SIGTERM received")
    read = _failing_when_stopped(gate, lambda: ShutdownRequested(request))
    stream = stream_shards(range(3), read, workers=2)
    assert next(stream) == 0
    assert gate.entered.wait(timeout=5)

    [error] = _close(stream, gate)
    assert isinstance(error, ShutdownRequested)
    assert error.request == request


def test_closing_a_stream_never_stops_a_stream_beside_it():
    closed_gate = Gate()
    beside_gate = Gate()

    def held(gate: Gate) -> Callable[[int], int]:
        def read(item: int) -> int:
            if item == 0:
                return 0

            async def wait() -> int:
                await gate.wait()
                return item

            return run_read(wait())

        return read

    closed = stream_shards(range(3), held(closed_gate), workers=2)
    beside = stream_shards(range(4), held(beside_gate), workers=2)
    assert next(closed) == 0
    assert next(beside) == 0
    assert closed_gate.entered.wait(timeout=5)
    assert beside_gate.entered.wait(timeout=5)

    assert _close(closed, closed_gate) == []
    beside_gate.release.set()
    assert list(beside) == [1, 2, 3]
    assert not beside_gate.cancelled.is_set()


def test_a_serial_stream_within_a_read_stops_with_the_stream_that_runs_the_read():
    gate = Gate()

    def inner(item: int) -> int:
        async def wait() -> int:
            await gate.wait()
            return item

        return run_read(wait())

    def outer(item: int) -> list[int]:
        if item == 0:
            return []
        # Within a stream's read, a stream runs serially in the reader thread.
        return list(stream_shards(range(2), inner, workers=2))

    stream = stream_shards(range(3), outer, workers=2)
    assert next(stream) == []
    assert gate.entered.wait(timeout=5)

    assert _close(stream, gate) == []
    assert gate.cancelled.is_set()


def test_a_threaded_stream_within_a_read_stops_with_the_stream_that_runs_the_read():
    gate = Gate()
    outer_threads: set[int] = set()
    carrier_threads: set[int] = set()
    inner_threads: set[int] = set()

    def inner(item: int) -> int:
        inner_threads.add(threading.get_ident())

        async def wait() -> int:
            await gate.wait()
            return item

        return run_read(wait())

    def carried() -> list[int]:
        carrier_threads.add(threading.get_ident())
        return list(stream_shards(range(2), inner, workers=2))

    def outer(item: int) -> list[int]:
        if item == 0:
            return []
        outer_threads.add(threading.get_ident())
        # A thread that carries the read's context, as asyncio.to_thread
        # gives one, but is no worker of the stream runs a threaded stream.
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, carried).result()

    stream = stream_shards(range(3), outer, workers=2)
    assert next(stream) == []
    assert gate.entered.wait(timeout=5)

    assert _close(stream, gate) == []
    assert gate.cancelled.is_set()
    assert inner_threads
    assert inner_threads.isdisjoint(outer_threads | carrier_threads)


def test_reads_outside_a_stream_run_without_a_stop():
    assert async_execution.current_read_stop() is None
    array = _memory_counts()
    np.testing.assert_array_equal(
        read_selection(array, slice(0, ROWS), slice(None)), VALUES[:ROWS]
    )
