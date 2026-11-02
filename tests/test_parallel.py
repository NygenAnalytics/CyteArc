import gc
import threading
import weakref

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.matrix import ChunkedArray
from cytearc.storage.parallel import in_shard_context, stream_shards
from tests.thread_helpers import blas_threads, raised_threads


def test_progress_closes_display_after_producer_cleanup_fails(monkeypatch):
    from cytearc.utils import progress

    closed = []

    class Display:
        def update(self):
            pass

        def close(self):
            closed.append("display")
            raise OSError("display cleanup")

    def blocks():
        try:
            yield 1
        finally:
            closed.append("producer")
            raise RuntimeError("producer cleanup")

    monkeypatch.setattr(progress, "tqdmbar", lambda **kwargs: Display())
    stream = progress.iter_progress(blocks())
    assert next(stream) == 1
    with pytest.raises(ExceptionGroup) as caught:
        stream.close()
    assert closed == ["producer", "display"]
    assert [str(error) for error in caught.value.exceptions] == [
        "producer cleanup",
        "display cleanup",
    ]


def test_array_explicit_thread_limit_is_respected(monkeypatch):
    from cytearc.storage.budget import ResourceBudget

    array = ChunkedArray.from_numpy(
        np.ones((32, 4)), block_size=4, nthreads=4, resources=ResourceBudget(1024**2, 4)
    )
    caller = threading.get_ident()
    threads = []
    original = ChunkedArray._materialize_range

    def materialize(self, start, end):
        threads.append(threading.get_ident())
        return original(self, start, end)

    monkeypatch.setattr(ChunkedArray, "_materialize_range", materialize)
    np.testing.assert_array_equal(array.sum(axis=0).compute(nthreads=1), [32.0] * 4)
    assert threads == [caller] * 8


def test_array_budget_includes_storage_retained_by_operand_views():
    from cytearc.storage.budget import ResourceBudget

    backing = np.ones((4, 8))
    large = np.ones((10_000, 8))
    array = ChunkedArray.from_numpy(
        backing, block_size=2, resources=ResourceBudget(16_384, 1)
    )
    with pytest.raises(MemoryError, match="Resident data"):
        (array + large[:1]).compute()
    np.testing.assert_array_equal((array + large[:1].copy()).compute(), backing * 2)


def test_stream_shards_bounds_in_flight():
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0
    # Each item waits for seven others, so the stream finishes only if eight
    # items run at once.
    barrier = threading.Barrier(8, timeout=10)

    def produce(value):
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        barrier.wait()
        with lock:
            in_flight -= 1
        return value

    assert list(stream_shards(range(16), produce, workers=8)) == list(range(16))
    assert max_seen == 8


def test_paused_serial_stream_does_not_mark_its_consumer_as_a_worker():
    contexts = []

    def produce(value):
        contexts.append(in_shard_context())
        return value

    stream = stream_shards([1, 2], produce, workers=1)
    try:
        assert next(stream) == 1
        assert not in_shard_context()
        assert contexts == [True]
        worker_threads = list(
            stream_shards([0, 1], lambda _: threading.get_ident(), workers=2)
        )
        assert all(worker != threading.get_ident() for worker in worker_threads)
    finally:
        stream.close()
    assert not in_shard_context()


def test_stream_shards_serial_backend_runs_inline():
    caller = threading.get_ident()
    threads = []

    def produce(value):
        threads.append(threading.get_ident())
        return value

    out = list(stream_shards([0, 1], produce, workers=8, backend="serial"))
    assert out == [0, 1]
    assert threads == [caller, caller]


def test_nested_stream_shards_run_serial():
    inner_calls = []
    lock = threading.Lock()

    def outer(value):
        assert in_shard_context() is True
        worker = threading.get_ident()

        def inner(item):
            # A nested stream runs inline on the worker of its outer item.
            with lock:
                inner_calls.append(
                    (value, item, in_shard_context(), threading.get_ident() == worker)
                )
            return 10 * value + item

        return list(stream_shards([0, 1, 2], inner, workers=8))

    assert list(stream_shards([0, 1], outer, workers=8)) == [[0, 1, 2], [10, 11, 12]]
    for value in (0, 1):
        assert [call for call in inner_calls if call[0] == value] == [
            (value, item, True, True) for item in (0, 1, 2)
        ]


def test_stream_shards_preserves_order():
    out = list(stream_shards(range(5), lambda x: x * 2, workers=4))
    assert out == [0, 2, 4, 6, 8]


def test_stream_shards_bounds_and_restores_io_concurrency():
    with zarr.config.set({"async.concurrency": 7}):
        seen = []

        def fn(x):
            seen.append(zarr.config.get("async.concurrency"))
            return x

        out = list(stream_shards(range(4), fn, workers=2, io_concurrency=3))
        assert out == [0, 1, 2, 3]
        assert seen == [3] * 4
        assert zarr.config.get("async.concurrency") == 7


def test_stream_shards_config_neutral_without_io():
    with zarr.config.set({"async.concurrency": 5}):
        seen = []

        def fn(x):
            seen.append(zarr.config.get("async.concurrency"))
            return x

        assert list(stream_shards(range(4), fn, workers=2)) == [0, 1, 2, 3]
        # Every item runs under the caller's setting, which stays in place.
        assert seen == [5] * 4
        assert zarr.config.get("async.concurrency") == 5


@pytest.mark.parametrize("workers", [1, 2])
def test_a_paused_stream_holds_no_io_limit(workers):
    from cytearc.storage import async_execution

    before = zarr.config.get("async.concurrency")
    seen = []

    def fn(value):
        seen.append(zarr.config.get("async.concurrency"))
        return value

    stream = stream_shards(range(2), fn, workers=workers, io_concurrency=1)
    try:
        # Taking both items waits for the read-ahead, so no item is running.
        assert [next(stream), next(stream)] == [0, 1]
        # The stream waits for its consumer and holds no limit meanwhile.
        assert zarr.config.get("async.concurrency") == before
        assert async_execution._ACTIVE_IO_LIMITS == {}
    finally:
        stream.close()
    assert seen == [1, 1]


@pytest.mark.parametrize("workers", [1, 2])
def test_a_stream_abandoned_in_a_reference_cycle_leaves_the_io_limit(workers):
    from cytearc.storage import async_execution

    before = zarr.config.get("async.concurrency")

    def refuse_the_last_item() -> weakref.ref:
        stream = stream_shards(
            range(2), lambda value: value, workers=workers, io_concurrency=1
        )
        kept: list[BaseException] = []
        try:
            for value in stream:
                if value == 1:
                    raise ValueError("the last item is refused")
        except ValueError as error:
            # The error's traceback holds this frame, which holds the error and
            # the stream, so the stream outlives the call in a reference cycle.
            kept.append(error)
        return weakref.ref(stream)

    enabled = gc.isenabled()
    gc.disable()
    try:
        stream = refuse_the_last_item()
        # Only the cycle collector can close the stream, and it has not run.
        assert stream() is not None
        assert zarr.config.get("async.concurrency") == before
        assert async_execution._ACTIVE_IO_LIMITS == {}
    finally:
        if enabled:
            gc.enable()
        # The collector frees the stream, so a failure here leaves no limit
        # entered for later tests.
        gc.collect()
    assert stream() is None


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_item_releases_its_io_limit(workers):
    from cytearc.storage import async_execution

    before = zarr.config.get("async.concurrency")

    def fail(value):
        raise RuntimeError(f"item {value} failed")

    # The kept error keeps the frames of the failed call alive, so only the
    # limit's own exit can restore the setting.
    with pytest.raises(RuntimeError, match="^item 0 failed$") as failed:
        list(stream_shards(range(1), fail, workers=workers, io_concurrency=1))
    assert zarr.config.get("async.concurrency") == before
    assert async_execution._ACTIVE_IO_LIMITS == {}
    del failed


@pytest.mark.parametrize("workers", [1, 2])
def test_each_item_holds_the_thread_limit_and_a_paused_stream_holds_none(workers):
    from cytearc.utils import compute

    with raised_threads(blas=2):
        seen = []

        def fn(value):
            seen.append(blas_threads())
            return value

        stream = stream_shards(range(2), fn, workers=workers, within_block_threads=1)
        try:
            # Taking both items waits for the read-ahead, so no item is running.
            assert [next(stream), next(stream)] == [0, 1]
            # The consumer's own work runs under the count outside the stream.
            assert blas_threads() == {2}
            assert compute._active_thread_limits == {}
        finally:
            stream.close()
        assert seen == [{1}, {1}]


@pytest.mark.parametrize("workers", [1, 2])
def test_a_stream_stopped_by_an_error_that_is_kept_leaves_no_thread_limit(workers):
    from cytearc.utils import compute

    def refuse_the_first_item() -> tuple[weakref.ref, list[BaseException]]:
        stream = stream_shards(
            range(3), lambda value: value, workers=workers, within_block_threads=1
        )
        kept: list[BaseException] = []
        try:
            for _ in stream:
                raise ValueError("the first item is refused")
        except ValueError as error:
            # As a notebook keeps the last error: its traceback holds this
            # frame, which holds the stream, so the stream stays alive.
            kept.append(error)
        return weakref.ref(stream), kept

    with raised_threads(blas=2):
        enabled = gc.isenabled()
        gc.disable()
        try:
            stream, kept = refuse_the_first_item()
            assert stream() is not None
            assert blas_threads() == {2}
            assert compute._active_thread_limits == {}
        finally:
            if enabled:
                gc.enable()
            del kept
            gc.collect()


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_item_releases_its_thread_limit(workers):
    from cytearc.utils import compute

    def fail(value):
        raise RuntimeError(f"item {value} failed")

    with raised_threads(blas=2):
        with pytest.raises(RuntimeError, match="^item 0 failed$") as failed:
            list(stream_shards(range(1), fail, workers=workers, within_block_threads=1))
        assert blas_threads() == {2}
        assert compute._active_thread_limits == {}
        del failed


def test_a_stream_inside_an_item_keeps_the_limit_of_the_outer_item():
    def inner(value):
        return blas_threads()

    def outer(value):
        # The nested stream asks for three threads, and its items still run
        # under the one thread of the outer item.
        return list(stream_shards(range(2), inner, workers=4, within_block_threads=3))

    with raised_threads(blas=2):
        assert list(
            stream_shards(range(2), outer, workers=2, within_block_threads=1)
        ) == [[{1}, {1}], [{1}, {1}]]
        assert blas_threads() == {2}


def test_stream_shards_cancels_unconsumed_work():
    started = []
    lock = threading.Lock()

    def work(value):
        with lock:
            started.append(value)
        return value

    stream = stream_shards(range(100), work, workers=2)
    assert next(stream) == 0
    stream.close()
    # Two items are read ahead, and the next is submitted only when the
    # consumer asks for it, so closing leaves the rest of the source unread.
    assert set(started) <= {0, 1}


@pytest.mark.parametrize("workers", [1, 2])
def test_stream_shards_closes_source_iterator(workers):
    closed = False

    def source():
        nonlocal closed
        try:
            yield from range(100)
        finally:
            closed = True

    stream = stream_shards(source(), lambda value: value, workers=workers)
    assert next(stream) == 0
    stream.close()
    assert closed


def test_stream_shards_cancels_pending_work_after_failure():
    started = []
    lock = threading.Lock()

    def work(value):
        with lock:
            started.append(value)
        if value == 0:
            raise RuntimeError("injected worker failure")
        return value

    with pytest.raises(RuntimeError, match="injected worker failure"):
        list(stream_shards(range(100), work, workers=2))
    assert set(started).issubset({0, 1})


def test_overlapping_parallel_runtimes_share_the_lower_io_limit():
    before = zarr.config.get("async.concurrency")
    barrier = threading.Barrier(2)
    seen = {3: [], 7: []}

    def run(io_concurrency):
        def inspect_config(value):
            barrier.wait()
            seen[io_concurrency].append(zarr.config.get("async.concurrency"))
            barrier.wait()
            return value

        list(
            stream_shards(
                [0],
                inspect_config,
                workers=1,
                io_concurrency=io_concurrency,
            )
        )

    first = threading.Thread(target=run, args=(3,))
    second = threading.Thread(target=run, args=(7,))
    first.start()
    second.start()
    first.join()
    second.join()
    assert seen == {3: [3], 7: [3]}
    assert zarr.config.get("async.concurrency") == before


def _toy_chunked(data, chunk_rows, nthreads):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array(
        "d", shape=data.shape, chunks=(chunk_rows, data.shape[1]), dtype=data.dtype
    )
    arr[:] = data
    return ChunkedArray(arr, nthreads=nthreads)


def test_stream_blocks_matches_serial_materialization():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((35, 6)).astype(np.float32)
    serial = np.vstack(list(_toy_chunked(data, 10, 1).stream_blocks(nthreads=1)))
    parallel = np.vstack(list(_toy_chunked(data, 10, 4).stream_blocks(nthreads=4)))
    np.testing.assert_array_equal(serial, data)
    np.testing.assert_array_equal(parallel, serial)


def test_reductions_bit_identical_across_threads():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((37, 5)).astype(np.float32)
    ca1 = _toy_chunked(data, 10, 1)
    ca4 = _toy_chunked(data, 10, 4)
    np.testing.assert_array_equal(
        np.asarray(ca4.sum(axis=0).compute(4)),
        np.asarray(ca1.sum(axis=0).compute(1)),
    )
    np.testing.assert_array_equal(
        np.asarray(ca4.var(axis=0).compute(4)),
        np.asarray(ca1.var(axis=0).compute(1)),
    )
    m1, s1 = ca1.mean_and_std(nthreads=1)
    m4, s4 = ca4.mean_and_std(nthreads=4)
    np.testing.assert_array_equal(m4, m1)
    np.testing.assert_array_equal(s4, s1)


def test_compute_matches_source_across_threads():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((40, 4)).astype(np.float32)
    np.testing.assert_array_equal(_toy_chunked(data, 8, 1).compute(1), data)
    np.testing.assert_array_equal(_toy_chunked(data, 8, 5).compute(5), data)


def test_stream_counts_the_consumed_block_toward_its_limit():
    alive = []
    maximum = 0
    lock = threading.Lock()
    produced = [threading.Event() for _ in range(12)]

    def produce(index):
        nonlocal maximum
        block = np.full(512, index)
        with lock:
            alive.append(weakref.ref(block))
            maximum = max(maximum, sum(ref() is not None for ref in alive))
        produced[index].set()
        return block

    stream = stream_shards(range(12), produce, workers=2)
    try:
        for index in range(12):
            block = next(stream)
            np.testing.assert_array_equal(block, index)
            # Hold the block until the next one exists beside it.
            if index + 1 < 12:
                assert produced[index + 1].wait(10)
            del block
    finally:
        stream.close()
    assert maximum == 2


def test_serial_source_observes_previous_consumption():
    consumed = []

    def source():
        for _ in range(4):
            yield len(consumed)

    for item in stream_shards(source(), lambda value: value, workers=1):
        consumed.append(item)
    assert consumed == [0, 1, 2, 3]


def test_stream_cleanup_reports_the_failure_with_every_cleanup_error():
    closed = []

    def source():
        try:
            yield from range(10)
        finally:
            closed.append("source")
            raise OSError("source cleanup failed")

    def work(value):
        if value == 0:
            raise ValueError("worker failed")
        return value

    with pytest.raises(
        BaseExceptionGroup, match="Block stream failed during cleanup"
    ) as caught:
        list(stream_shards(source(), work, workers=2))
    assert [(type(error), str(error)) for error in caught.value.exceptions] == [
        (ValueError, "worker failed"),
        (OSError, "source cleanup failed"),
    ]
    assert closed == ["source"]


def test_failed_pool_shutdown_still_closes_the_source(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import cytearc.storage.parallel as parallel

    class FailingShutdown(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait, cancel_futures=cancel_futures)
            raise RuntimeError("pool shutdown interrupted")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", FailingShutdown)
    closed = []

    def source():
        try:
            yield from range(4)
        finally:
            closed.append("source")

    stream = stream_shards(source(), lambda value: value, workers=2)
    assert next(stream) == 0
    # Closing early is no failure of its own, so the cleanup error stands alone.
    with pytest.raises(RuntimeError, match="pool shutdown interrupted"):
        stream.close()
    assert closed == ["source"]


def test_progress_stream_reports_a_failure_with_its_cleanup_errors(monkeypatch):
    from cytearc.utils import progress

    class Display:
        def update(self):
            pass

        def close(self):
            raise OSError("display cleanup")

    def failing():
        yield 1
        raise ValueError("producer failed")

    monkeypatch.setattr(progress, "tqdmbar", lambda **kwargs: Display())
    with pytest.raises(BaseExceptionGroup, match="Progress stream cleanup") as caught:
        list(progress.iter_progress(failing()))
    assert [(type(error), str(error)) for error in caught.value.exceptions] == [
        (ValueError, "producer failed"),
        (OSError, "display cleanup"),
    ]

    stream = progress.iter_progress(iter([1, 2]))
    assert next(stream) == 1
    with pytest.raises(OSError, match="display cleanup"):
        stream.close()


def test_progress_bars_let_explicit_settings_replace_cytearc_defaults(monkeypatch):
    import tqdm.auto

    from cytearc.utils.progress import tqdm_params, tqdmbar

    captured: list[dict] = []
    monkeypatch.setattr(
        tqdm.auto, "tqdm", lambda *args, **kwargs: captured.append(kwargs)
    )
    tqdmbar(total=3, bar_format="{n}/{total}")

    assert captured[0]["bar_format"] == "{n}/{total}"
    assert captured[0]["total"] == 3
    assert captured[0]["colour"] == tqdm_params["colour"]
    assert captured[0]["dynamic_ncols"] is True


def test_early_close_surfaces_worker_failure():
    failed = threading.Event()

    def produce(index):
        if index:
            failed.set()
            raise ValueError("producer failed")
        return index

    stream = stream_shards(range(2), produce, workers=2)
    assert next(stream) == 0
    assert failed.wait(5)
    with pytest.raises(ValueError, match="producer failed"):
        stream.close()
