"""One async coordinator for CyteArc-owned Zarr array operations."""

import asyncio
import contextvars
import importlib
import itertools
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from typing import Any, TypeVar, cast

import zarr

from ..utils.memory_budget import MemoryBudgetError
from ..utils.numba import numba_threads
from ..utils.shutdown import ShutdownRequested, shutdown_checkpoint

from .budget import detect_workers
from .execution import OperationPlan

T = TypeVar("T")

# The host threads of Zarr's codec pool, fixed when the pool is installed.
# This is the process thread ceiling, not an operation plan.
_HOST_THREAD_CEILING: int | None = None
_ZARR_CONFIG_LOCK = threading.RLock()
_ACTIVE_IO_LIMITS: dict[object, int] = {}
_IDLE_IO_LIMIT: int | None = None

# The threads that the Zarr reads of the calling context decode on, or None
# for the host threads of Zarr's codec pool.
_DECODING_THREADS: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "zarr_decoding_threads", default=None
)


_IO_TASKS: contextvars.ContextVar[set[asyncio.Future[Any]] | None] = (
    contextvars.ContextVar("storage_io_tasks", default=None)
)


@contextmanager
def zarr_io_concurrency(limit: int) -> Iterator[None]:
    global _IDLE_IO_LIMIT
    token = object()
    with _ZARR_CONFIG_LOCK:
        if not _ACTIVE_IO_LIMITS:
            _IDLE_IO_LIMIT = zarr.config.get("async.concurrency")
        _ACTIVE_IO_LIMITS[token] = max(1, int(limit))
        zarr.config.set({"async.concurrency": min(_ACTIVE_IO_LIMITS.values())})
    try:
        yield
    finally:
        with _ZARR_CONFIG_LOCK:
            del _ACTIVE_IO_LIMITS[token]
            restored = (
                min(_ACTIVE_IO_LIMITS.values()) if _ACTIVE_IO_LIMITS else _IDLE_IO_LIMIT
            )
            zarr.config.set({"async.concurrency": restored})


def _in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _leaves(error: BaseException) -> Iterator[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        for inner in error.exceptions:
            yield from _leaves(inner)
    else:
        yield error


def _reported_error(error: BaseException) -> BaseException:
    """Return the error a caller sees for one failed storage operation.

    Task groups wrap even a single failure in an exception group. A shutdown
    request is reported on its own, and a group with one leaf is replaced by
    that leaf; groups of distinct failures are kept.
    """
    if not isinstance(error, BaseExceptionGroup):
        return error
    leaves = list(_leaves(error))
    for leaf in leaves:
        if isinstance(leaf, ShutdownRequested):
            return leaf
    return leaves[0] if len(leaves) == 1 else error


async def _await_completion(future: asyncio.Future[T]) -> T:
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except BaseException:
            break
    if cancellation is not None:
        try:
            future.result()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                raise BaseExceptionGroup(
                    "Storage work failed during cancellation", [cancellation, exc]
                ) from None
        raise cancellation
    return future.result()


def _scoped_task(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    **kwargs: Any,
) -> asyncio.Future[Any]:
    task = asyncio.Task(coro, loop=loop, **kwargs)
    context = kwargs.get("context")
    scope = _IO_TASKS.get() if context is None else context.get(_IO_TASKS)
    if scope is not None:
        scope.add(task)

        def completed(future: asyncio.Future[Any]) -> None:
            # Successful results belong to their caller. Keep failures and
            # cancelled tasks for _drained, without retaining decoded buffers.
            if not future.cancelled() and future.exception() is None:
                scope.discard(future)

        task.add_done_callback(completed)
    return task


async def _kept(results: list[T], coroutine: Coroutine[Any, Any, T]) -> None:
    """Run ``coroutine`` and leave its result in ``results``."""
    results.append(await coroutine)


async def _released_call(executor: ThreadPoolExecutor, fn: Callable[[], T]) -> T:
    """Run ``fn`` on ``executor`` without its worker keeping ``fn`` or its result.

    A pool worker holds its last work item until it takes the next one, so a
    buffer that ``fn`` captures or returns would otherwise outlive the
    reservation that covers it.
    """
    calls = [fn]
    del fn
    results: list[T] = []
    # The worker runs ``fn`` in the caller's context, as a thread of the
    # operation would, so a pipeline stage's log context, collected warnings,
    # and shutdown checkpoints reach it.
    context = contextvars.copy_context()

    def call() -> None:
        results.append(context.run(calls.pop()))

    await _await_completion(asyncio.get_running_loop().run_in_executor(executor, call))
    return results.pop()


async def _drained(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run one storage coroutine and wait for every task it started."""
    tasks: set[asyncio.Future[Any]] = set()
    token = _IO_TASKS.set(tasks)
    try:
        root = asyncio.create_task(coroutine)
    finally:
        _IO_TASKS.reset(token)
    errors: list[BaseException] = []
    result: Any = None
    try:
        result = await _await_completion(root)
    except BaseException as exc:
        errors.append(exc)
    # The root's outcome is already known. Gathering it again would report a
    # cancelled root twice, as gather creates a new CancelledError for it.
    observed: set[asyncio.Future[Any]] = {root}
    while outstanding := tasks - observed:
        observed.update(outstanding)
        # Nothing cancels this coroutine: io() shields the future it waits on.
        outcomes = await _await_completion(
            asyncio.gather(*outstanding, return_exceptions=True)
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not any(
                outcome is error for error in errors
            ):
                errors.append(outcome)
    tasks.clear()
    if len(errors) > 1:
        raise BaseExceptionGroup("Storage I/O failed while draining work", errors)
    if errors:
        raise errors[0]
    return cast(T, result)


class _IoLoops:
    """Event-loop threads that run storage coroutines.

    Zarr copies decoded chunks into their destination on the thread that runs
    its event loop. Spreading operations over several loops runs those copies
    in parallel, while codec work still uses the shared codec pool.
    """

    def __init__(self, count: int, executor: ThreadPoolExecutor) -> None:
        self._loops: list[asyncio.AbstractEventLoop] = []
        self._threads: list[threading.Thread] = []
        self._turn = itertools.count()
        for index in range(max(1, count)):
            loop = asyncio.new_event_loop()
            loop.set_default_executor(executor)
            loop.set_task_factory(_scoped_task)
            thread = threading.Thread(
                target=loop.run_forever, name=f"cytearc-zarr-io_{index}", daemon=True
            )
            thread.start()
            self._loops.append(loop)
            self._threads.append(thread)

    def submit(self, coroutine: Coroutine[Any, Any, T]) -> "Future[T]":
        loop = self._loops[next(self._turn) % len(self._loops)]
        return asyncio.run_coroutine_threadsafe(_drained(coroutine), loop)

    def close(self) -> None:
        for loop in self._loops:
            loop.call_soon_threadsafe(loop.stop)
        for loop, thread in zip(self._loops, self._threads, strict=True):
            thread.join()
            loop.close()


def _active_zarr_workers() -> int | None:
    sync = importlib.import_module("zarr.core.sync")
    pool = sync._executor
    if pool is None and sync.loop[0] is not None:
        pool = sync.loop[0]._default_executor
    return None if pool is None else int(pool._max_workers)


class _DecodingPools(ThreadPoolExecutor):
    """Zarr's codec pool: the host's threads, and a pool of each other count of threads that reads decode on.

    Zarr's event loop runs each fetch and decode of a read on its default
    executor, in the context of the call that made the read. A call within
    :func:`zarr_decoding_threads` runs them on a pool of exactly that many
    threads, which the reads of every call of that count share; any other
    call runs them on the host's threads.
    """

    def __init__(self, workers: int) -> None:
        super().__init__(max_workers=workers, thread_name_prefix="zarr_pool")
        self._sized: dict[int, ThreadPoolExecutor] = {}
        self._sized_lock = threading.Lock()
        # Set under the lock, so no pool of a new count starts once shutdown has begun.
        self._closing = False

    def submit[**P, R](
        self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> "Future[R]":
        threads = _DECODING_THREADS.get()
        if threads is None or threads == self._max_workers:
            return super().submit(fn, *args, **kwargs)
        with self._sized_lock:
            pool = self._sized.get(threads)
            if pool is None:
                if self._closing:
                    raise RuntimeError("cannot schedule new futures after shutdown")
                pool = ThreadPoolExecutor(
                    max_workers=threads, thread_name_prefix=f"zarr_pool_of_{threads}"
                )
                self._sized[threads] = pool
        return pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._sized_lock:
            self._closing = True
            sized = list(self._sized.values())
        for pool in sized:
            pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def _install_decoding_pools(workers: int) -> None:
    """Make CyteArc's decoding pools the default executor of Zarr's event loop.

    Pools made here also become Zarr's own, which Zarr then makes no pool in
    place of, and which a later install finds when another executor has
    replaced them on the loop. A pool that Zarr or asyncio made before is
    left to finish what it runs, as Zarr leaves a pool that it replaces,
    since the loop may be submitting to it.
    """
    sync: Any = importlib.import_module("zarr.core.sync")
    loop = sync._get_loop()
    pools = loop._default_executor
    if not isinstance(pools, _DecodingPools):
        pools = sync._executor
        if not isinstance(pools, _DecodingPools):
            pools = _DecodingPools(workers)
            sync._executor = pools
        loop.set_default_executor(pools)
    if sync._executor is None:
        sync._executor = pools


def ensure_zarr_host_ceiling() -> int:
    """Install CyteArc's decoding pools as Zarr's codec pool, and return their host threads.

    The host threads are those of a pool that Zarr already runs, or Zarr's
    ``threading.max_workers``, or the host's workers, fixed by the first use.
    """
    global _HOST_THREAD_CEILING
    with _ZARR_CONFIG_LOCK:
        active = _active_zarr_workers()
        if active is not None:
            _HOST_THREAD_CEILING = active
        elif _HOST_THREAD_CEILING is None:
            configured = zarr.config.get("threading.max_workers", None)
            _HOST_THREAD_CEILING = int(configured or max(1, detect_workers()))
            zarr.config.set({"threading.max_workers": _HOST_THREAD_CEILING})
        _install_decoding_pools(_HOST_THREAD_CEILING)
    return _HOST_THREAD_CEILING


@contextmanager
def zarr_decoding_threads(threads: int) -> Iterator[None]:
    """Decode the Zarr reads that the calling thread makes in this context on a pool of exactly ``threads`` threads.

    A thread of Zarr's codec pool can still hold the last chunk that it
    fetched or decoded after the read returns, so reads within leave at most
    ``threads`` such chunks, on any machine. Reads of the same count in
    flight together, from any thread, share one pool of that many threads;
    reads of another count, or outside this context, decode on their own.
    Reads through :class:`AsyncStorageRunner` decode on the runner's pool.

    Raises:
        ValueError: If ``threads`` is not a whole number of at least 1.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError(
            f"Reads decode on a whole number of threads of at least 1, not {threads!r}"
        )
    ensure_zarr_host_ceiling()
    token = _DECODING_THREADS.set(threads)
    try:
        yield
    finally:
        _DECODING_THREADS.reset(token)


class ByteLedger:
    """Admit CyteArc-owned buffers before async tasks are created.

    ``reservedBytes`` is what the memory budget holds beside the ledger's
    ``limitBytes``, such as an operation's resident data, which a refusal
    adds back to name the whole budget.
    """

    def __init__(self, limitBytes: int, *, reservedBytes: int = 0):
        if limitBytes < 1:
            raise ValueError("byte ledger limit must be positive")
        self.limitBytes = int(limitBytes)
        self.reservedBytes = max(0, int(reservedBytes))
        self._held = 0
        self._peak = 0
        self._condition = asyncio.Condition()

    async def acquire(self, nbytes: int) -> None:
        size = int(nbytes)
        if size < 1:
            return
        if size > self.limitBytes:
            raise MemoryBudgetError(
                f"One buffer needs {size} bytes, but the operation limit is "
                f"{self.limitBytes} bytes",
                needed_bytes=self.reservedBytes + size,
                budget_bytes=self.reservedBytes + self.limitBytes,
                item="one buffer",
            )
        async with self._condition:
            while self._held + size > self.limitBytes:
                await self._condition.wait()
            self._held += size
            self._peak = max(self._peak, self._held)

    async def release(self, nbytes: int) -> None:
        size = int(nbytes)
        if size < 0:
            raise ValueError("released byte count must not be negative")
        async with self._condition:
            if size > self._held:
                raise RuntimeError(
                    f"cannot release {size} bytes from a ledger holding {self._held}"
                )
            self._held -= size
            self._condition.notify_all()

    def held_bytes(self) -> int:
        return self._held

    def peak_bytes(self) -> int:
        return self._peak

    def is_empty(self) -> bool:
        return self._held == 0


class AsyncStorageRunner:
    """Own one event loop, codec executor, compute pool, and byte ledger.

    The runner holds Zarr's process-wide I/O limit only while its own reads
    and writes run, and sets no BLAS or OpenMP limit, since its compute calls
    run no such work. A runner that waits on its caller, as a stream whose
    consumer has stopped does, holds no process-wide limit.
    """

    def __init__(
        self,
        *,
        operation: OperationPlan,
    ):
        self.plan = operation
        self.ledger = ByteLedger(
            operation.requestedMemoryBytes - operation.residentBytes,
            reservedBytes=operation.residentBytes,
        )
        self.readerWaitSeconds = 0.0
        self._compute_pool: ThreadPoolExecutor | None = None
        self._codec_pool: ThreadPoolExecutor | None = None
        self._read_slots: asyncio.Semaphore | None = None
        self._commit_slots: asyncio.Semaphore | None = None
        self._io_loops: _IoLoops | None = None
        self._worker_numba: list[AbstractContextManager[None]] = []

    def run(self, operation: Callable[["AsyncStorageRunner"], Awaitable[T]]) -> T:
        """Run ``operation`` to completion and return its result.

        Inside a running event loop, as in a Jupyter cell, the operation runs
        on a helper thread with its own loop. An interruption of the caller,
        such as Ctrl-C, cancels the operation and waits until it has stopped
        reading, computing, and writing before the interruption propagates,
        so no work continues in the background behind the caller's cleanup.
        """
        # Outside an except clause, so a failure is not chained to the error
        # that found no running loop.
        if not _in_event_loop():
            return asyncio.run(self._run(operation))
        error: list[BaseException] = []
        result: list[T] = []
        context = contextvars.copy_context()
        started = threading.Event()
        main: list[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]]] = []

        async def _main() -> T:
            task = asyncio.current_task()
            assert task is not None
            main.append((asyncio.get_running_loop(), task))
            started.set()
            return await self._run(operation)

        def _in_thread() -> None:
            try:
                result.append(context.run(lambda: asyncio.run(_main())))
            except BaseException as exc:
                error.append(exc)
            finally:
                started.set()

        thread = threading.Thread(target=_in_thread, name="cytearc-storage-operation")
        thread.start()
        try:
            thread.join()
        except BaseException:
            # The caller was interrupted, which stops only its wait. Cancel the
            # operation, wait until its work has stopped, then propagate.
            started.wait()
            for loop, task in main:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    # The loop closed: the operation already finished.
                    pass
            thread.join()
            raise
        if error:
            raise error[0]
        return result[0]

    async def _run(
        self,
        operation: Callable[["AsyncStorageRunner"], Awaitable[T]],
    ) -> T:
        loop = asyncio.get_running_loop()
        result: Any = None
        errors: list[BaseException] = []
        # Numba keeps a count for each thread; this sets the count of the
        # thread that runs the operation's own code.
        numba_context = numba_threads(self.plan.threadsPerComputeWorker)
        numba_entered = False
        try:
            ensure_zarr_host_ceiling()
            self._codec_pool = ThreadPoolExecutor(
                max_workers=self.plan.codecWorkers,
                thread_name_prefix="cytearc-zarr-codec",
            )
            loop.set_default_executor(self._codec_pool)
            self._io_loops = _IoLoops(self.plan.codecWorkers, self._codec_pool)
            self._compute_pool = ThreadPoolExecutor(
                max_workers=self.plan.computeWorkers,
                thread_name_prefix="cytearc-compute",
                initializer=self._start_compute_worker,
            )
            self._read_slots = asyncio.Semaphore(
                self.plan.readWorkers * self.plan.innerReads
            )
            self._commit_slots = asyncio.Semaphore(self.plan.writeWorkers)
            numba_context.__enter__()
            numba_entered = True
            shutdown_checkpoint()
            result = await operation(self)
            shutdown_checkpoint()
        except BaseException as exc:
            errors.append(_reported_error(exc))
        finally:
            pending = asyncio.all_tasks(loop) - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            if pending:
                try:
                    outcomes = await _await_completion(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                    errors.extend(
                        item
                        for item in outcomes
                        if isinstance(item, BaseException)
                        and not isinstance(item, asyncio.CancelledError)
                    )
                except BaseException as exc:
                    errors.append(exc)
            for pool in (self._compute_pool, self._codec_pool):
                if pool is not None:
                    try:
                        pool.shutdown(wait=True, cancel_futures=True)
                    except BaseException as exc:
                        errors.append(exc)
            # The workers have finished, so their settings end without a change.
            for context in self._worker_numba:
                try:
                    context.__exit__(None, None, None)
                except BaseException as exc:
                    errors.append(exc)
            self._worker_numba.clear()
            if self._io_loops is not None:
                try:
                    self._io_loops.close()
                except BaseException as exc:
                    errors.append(exc)
                self._io_loops = None
            if numba_entered:
                try:
                    numba_context.__exit__(None, None, None)
                except BaseException as exc:
                    errors.append(exc)
            leftover = self.ledger.held_bytes()
        if leftover:
            errors.append(
                RuntimeError(
                    f"byte ledger still holds {leftover} bytes after the operation"
                )
            )
        if len(errors) > 1:
            raise BaseExceptionGroup(
                "Async storage failed during execution or cleanup",
                errors,
            )
        if errors:
            raise errors[0]
        return cast(T, result)

    async def io(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run a self-contained storage coroutine on one of the I/O loops.

        Zarr's async concurrency is held at the plan's limit, or a lower one
        already in force, until the coroutine has finished. The result comes
        back in a holder, so the I/O loop thread keeps no reference to it once
        the coroutine has finished.
        """
        if self._io_loops is None:
            raise RuntimeError("I/O loops are not installed")
        results: list[T] = []
        with zarr_io_concurrency(self.plan.ioConcurrency):
            await _await_completion(
                asyncio.wrap_future(self._io_loops.submit(_kept(results, coroutine)))
            )
        return results.pop()

    def _start_compute_worker(self) -> None:
        """Run a compute worker's Numba kernels on the plan's threads for its life."""
        context = numba_threads(max(1, int(self.plan.threadsPerComputeWorker)))
        context.__enter__()
        self._worker_numba.append(context)

    async def compute(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` on the compute pool, whose workers run Numba's kernels on
        the plan's threads per worker.

        A call sets no BLAS or OpenMP limit: such a limit applies to the whole
        process, beside the caller's own work, and the runner's calls assemble,
        count and copy without BLAS or OpenMP work. A function that runs such
        work holds its own through ``process_thread_limit``.
        """
        if self._compute_pool is None:
            raise RuntimeError("compute pool is not installed")

        def _limited() -> T:
            from .parallel import _shard_context

            with _shard_context():
                return fn()

        shutdown_checkpoint()
        result = await _released_call(self._compute_pool, _limited)
        shutdown_checkpoint()
        return result

    async def offload(self, fn: Callable[[], T]) -> T:
        """Run a short blocking call on the codec pool, off the event loop."""
        if self._codec_pool is None:
            raise RuntimeError("codec pool is not installed")
        return await _released_call(self._codec_pool, fn)

    async def read_slot(self) -> asyncio.Semaphore:
        if self._read_slots is None:
            raise RuntimeError("read slots are not installed")
        return self._read_slots

    async def commit_slot(self) -> asyncio.Semaphore:
        if self._commit_slots is None:
            raise RuntimeError("commit slots are not installed")
        return self._commit_slots

    @asynccontextmanager
    async def reserve_bytes(self, nbytes: int) -> AsyncIterator[None]:
        """Hold a ledger charge for the complete lifetime of an owned buffer."""
        started = time.perf_counter()
        shutdown_checkpoint()
        await self.ledger.acquire(nbytes)
        try:
            shutdown_checkpoint()
            self.readerWaitSeconds += time.perf_counter() - started
            yield
        finally:
            await _await_completion(asyncio.create_task(self.ledger.release(nbytes)))

    @asynccontextmanager
    async def read_lane(self) -> AsyncIterator[None]:
        """Enter one bounded outer read lane without changing byte ownership."""
        slot = await self.read_slot()
        started = time.perf_counter()
        await slot.acquire()
        self.readerWaitSeconds += time.perf_counter() - started
        try:
            yield
        finally:
            slot.release()

    @asynccontextmanager
    async def commit_lane(self) -> AsyncIterator[None]:
        """Enter one bounded destination commit lane without changing ownership."""
        slot = await self.commit_slot()
        async with slot:
            yield
