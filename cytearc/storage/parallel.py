import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Literal


from ..utils.errors import raise_collected_errors
from ..utils.shutdown import shutdown_checkpoint

__all__ = ["stream_shards", "in_shard_context"]

type Backend = Literal["thread", "serial"]

_shard_ctx = threading.local()

_CLEANUP_FAILED = "Block stream failed during cleanup"


def in_shard_context() -> bool:
    """Return whether the caller is already inside a shard worker."""
    return bool(getattr(_shard_ctx, "active", False))


@contextmanager
def _shard_context() -> Iterator[None]:
    previous = getattr(_shard_ctx, "active", False)
    _shard_ctx.active = True
    try:
        yield
    finally:
        _shard_ctx.active = previous


def _io_limited(fn: Callable[[Any], Any], io: int | None) -> Callable[[Any], Any]:
    """Return ``fn`` holding Zarr's process-wide I/O limit ``io`` while it runs."""
    if io is None:
        return fn
    from .async_execution import zarr_io_concurrency

    def limited(item: Any) -> Any:
        with zarr_io_concurrency(io):
            return fn(item)

    return limited


def _close_iterator(iterator: Iterator[Any]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _imap_ordered(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int,
) -> Iterator[Any]:
    def worker(item: Any) -> Any:
        _shard_ctx.active = True
        return fn(item)

    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending: deque[Future[Any]] = deque()

    def enqueue() -> bool:
        shutdown_checkpoint()
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.append(executor.submit(worker, item))
        return True

    # The error that ends the stream is caught by name rather than read in a
    # ``finally``: Python unbinds the name when the clause ends, so this frame
    # never keeps an error whose traceback refers back to it. That reference
    # cycle would keep the read-ahead blocks until the cycle collector ran.
    try:
        for _ in range(workers):
            if not enqueue():
                break
        while pending:
            result = pending.popleft().result()
            shutdown_checkpoint()
            try:
                yield result
            finally:
                del result
            enqueue()
    except BaseException as error:
        raise_collected_errors(
            _stop_reads(executor, pending, iterator), _CLEANUP_FAILED, ended_by=error
        )
        raise
    raise_collected_errors(_stop_reads(executor, pending, iterator), _CLEANUP_FAILED)


def _stop_reads(
    executor: ThreadPoolExecutor,
    pending: deque[Future[Any]],
    iterator: Iterator[Any],
) -> list[BaseException]:
    """Stop a stream's reads and close its items, returning what failed.

    Reads that have not started are cancelled, and running reads finish before
    this returns. ``pending`` is empty afterwards, so no finished block outlives
    the stream. The caller raises the errors through ``raise_collected_errors``,
    which empties the list, so no frame here keeps one.
    """
    errors: list[BaseException] = []
    for future in pending:
        future.cancel()
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except BaseException as error:
        errors.append(error)
    while pending:
        _collect_failure(pending.popleft(), errors)
    try:
        _close_iterator(iterator)
    except BaseException as error:
        errors.append(error)
    return errors


def _collect_failure(future: Future[Any], errors: list[BaseException]) -> None:
    """Add the error of a stopped read to ``errors``, unless it was cancelled or succeeded."""
    try:
        failure = None if future.cancelled() else future.exception()
    except BaseException as error:
        # An interruption while a read that a failed shutdown left still runs.
        errors.append(error)
        return
    if failure is not None:
        errors.append(failure)


def _progress(
    iterator: Iterator[Any],
    msg: str | None,
    total: int | None,
) -> Iterator[Any]:
    if msg is None:
        yield from iterator
        return
    from ..utils.progress import iter_progress

    yield from iter_progress(iterator, desc=msg, total=total)


def stream_shards(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int,
    io_concurrency: int | None = None,
    msg: str | None = None,
    total: int | None = None,
    backend: Backend = "thread",
) -> Generator[Any, None, None]:
    """Yield transformed items in order with bounded read-ahead.

    Zarr's process-wide async concurrency is set to ``io_concurrency``, or to
    a lower limit already in force, only while a call of ``fn`` runs, in the
    thread that runs it. Between items, the consumer runs under the setting
    outside the stream, or, in a threaded stream, under the stream's while a
    read-ahead call runs. A stream that its consumer stops without closing, as
    an error in the consumer does, leaves the setting as it was once its
    read-ahead calls end, even before the stream is collected. A stream sets
    no BLAS or OpenMP limit, which would apply to the whole process beside its
    consumer; an ``fn`` that runs such work holds its own through
    ``process_thread_limit``.
    """
    workers = max(1, int(workers))
    work = _io_limited(fn, io_concurrency)
    if backend == "serial" or workers <= 1 or in_shard_context():
        iterator = iter(items)
        try:

            def checked() -> Iterator[Any]:
                for item in iterator:
                    shutdown_checkpoint()
                    with _shard_context():
                        result = work(item)
                    shutdown_checkpoint()
                    yield result

            yield from _progress(checked(), msg, total)
        finally:
            _close_iterator(iterator)
        return
    base = _imap_ordered(items, work, workers=workers)
    yield from _progress(base, msg, total)
