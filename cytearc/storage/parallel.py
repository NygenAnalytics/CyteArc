import threading
import sys
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Literal


from ..utils.compute import process_thread_limit
from ..utils.shutdown import shutdown_checkpoint

__all__ = ["stream_shards", "in_shard_context"]

type Backend = Literal["thread", "serial"]

_shard_ctx = threading.local()


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


def _limited(
    fn: Callable[[Any], Any], io: int | None, threads: int | None
) -> Callable[[Any], Any]:
    """Return ``fn`` holding the stream's process-wide limits while it runs.

    ``io`` is Zarr's I/O limit and ``threads`` the native thread-pool limit;
    None leaves one as it is.
    """
    if io is None and threads is None:
        return fn
    from .async_execution import zarr_io_concurrency

    def io_limited(item: Any) -> Any:
        if io is None:
            return fn(item)
        with zarr_io_concurrency(io):
            return fn(item)

    if threads is None:
        return io_limited

    def limited(item: Any) -> Any:
        with process_thread_limit(threads):
            return io_limited(item)

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

    try:
        for _ in range(workers):
            if not enqueue():
                break
        while pending:
            result = pending.popleft().result()
            shutdown_checkpoint()
            yield result
            del result
            enqueue()
    finally:
        original = sys.exception()
        errors: list[BaseException] = []
        for future in pending:
            future.cancel()
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as exc:
            errors.append(exc)
        for future in pending:
            try:
                future.result()
            except CancelledError:
                pass
            except BaseException as exc:
                errors.append(exc)
        try:
            _close_iterator(iterator)
        except BaseException as exc:
            errors.append(exc)
        if errors:
            if original is not None and not isinstance(original, GeneratorExit):
                errors.insert(0, original)
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("Block stream failed during cleanup", errors)


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
    within_block_threads: int | None = None,
    io_concurrency: int | None = None,
    msg: str | None = None,
    total: int | None = None,
    backend: Backend = "thread",
) -> Generator[Any, None, None]:
    """Yield transformed items in order with bounded read-ahead.

    Each call of ``fn`` holds two process-wide limits, in the thread that runs
    it: Zarr's async concurrency at ``io_concurrency``, or a lower limit
    already in force, and the native thread pools at ``within_block_threads``
    threads. A stream inside a shard worker sets no thread limit, so the
    worker's own applies. No limit is held across a yield: between items, the
    consumer runs under the settings outside the stream, or, in a threaded
    stream, under the stream's while a read-ahead call runs. A stream that its
    consumer stops without closing, as an error in the consumer does, leaves
    the settings as they were once its read-ahead calls end, even before the
    stream is collected.
    """
    workers = max(1, int(workers))
    threads = (
        within_block_threads
        if within_block_threads is not None
        and within_block_threads >= 1
        and not in_shard_context()
        else None
    )
    work = _limited(fn, io_concurrency, threads)
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
