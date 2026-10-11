"""Bounded temporary reuse of encoded composite source objects.

The cache keeps whole source objects, such as count shards, and shard-index
suffixes in a temporary directory, so later reads of a remote source do not
download them again. It is an optimization only: a fault of the local
directory turns the cache off and every read goes to its source.
"""

import asyncio
import copy
import operator
import os
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from threading import RLock
from traceback import clear_frames
from typing import Any, Self

from zarr.abc.store import ByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.storage import LocalStore, WrapperStore

from ..utils.logging import logger

SOURCE_CACHE_BYTES_ENV = "CYTEARC_SOURCE_CACHE_BYTES"
DEFAULT_SOURCE_CACHE_BYTES = 2 * 1024**3
_SIZE_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

type _CacheKey = tuple[int, str] | tuple[int, str, int]


def configured_source_cache_bytes() -> int:
    """Return the byte limit of each composite's remote source cache.

    ``CYTEARC_SOURCE_CACHE_BYTES`` sets it as bytes or a binary size such as
    ``512M`` or ``4G``, and ``0`` turns the cache off. Unset, it is 2 GiB.

    Raises:
        ValueError: If the variable is not a size.
    """
    text = os.environ.get(SOURCE_CACHE_BYTES_ENV, "").strip()
    if not text:
        return DEFAULT_SOURCE_CACHE_BYTES
    scale = _SIZE_SUFFIXES.get(text[-1].upper(), 1)
    number = text[:-1] if scale > 1 else text
    try:
        value = float(number)
    except ValueError:
        value = -1.0
    if value < 0 or (scale == 1 and not value.is_integer()):
        raise ValueError(
            f"{SOURCE_CACHE_BYTES_ENV}={text!r} must be bytes, a size such as "
            "'512M' or '4G', or 0 to turn the composite source cache off"
        )
    return int(value * scale)


def new_source_cache() -> "SourceCache | None":
    """Return a cache for remote composite sources, or None when it is off."""
    capacity = configured_source_cache_bytes()
    return SourceCache(capacity_bytes=capacity) if capacity > 0 else None


@dataclass(eq=False)
class _Entry:
    source: Store
    key: _CacheKey
    path: str
    reserved: int
    limit: int
    suffix: int | None = None
    future: Future[bool] = field(default_factory=Future)
    task: asyncio.Task[None] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    waiters: int = 0
    readers: int = 0
    ready: bool = False
    abandoned: bool = False
    started: bool = False


async def _finish_io[T](operation: Awaitable[T]) -> T:
    task = asyncio.ensure_future(operation)
    cancelled = False
    try:
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                # Stores can use worker threads. Repeated cancellation must not
                # release their buffers or files before the I/O has finished.
                cancelled = True
    except BaseException:
        del task, operation
        raise
    if cancelled:
        del result, task, operation
        raise asyncio.CancelledError
    return result


def _release_exception_frames(error: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        clear_frames(current.__traceback__)
        pending.extend(
            item
            for item in (current.__cause__, current.__context__)
            if item is not None
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


class SourceCache:
    """Encoded source objects kept in one temporary directory.

    The directory is created by the first fetch. ``close`` ends the cache for
    its owner, after which reads through it raise. A fault of the directory
    instead turns the cache off: pending and later reads go to their sources.
    """

    def __init__(
        self,
        *,
        capacity_bytes: int = DEFAULT_SOURCE_CACHE_BYTES,
        max_entries: int = 4096,
    ) -> None:
        self._capacity = operator.index(capacity_bytes)
        self._max_entries = operator.index(max_entries)
        if self._capacity < 1 or self._max_entries < 1:
            raise ValueError("Source cache byte and entry limits must be positive")
        self._directory: TemporaryDirectory[str] | None = None
        self._local: LocalStore | None = None
        self._lock = RLock()
        self._entries: OrderedDict[_CacheKey, _Entry] = OrderedDict()
        self._used_bytes = 0
        self._next_path = 0
        self._active = 0
        self._closed = False
        self._off = False

    @property
    def capacity_bytes(self) -> int:
        """Bytes the cache may keep."""
        return self._capacity

    @property
    def max_entries(self) -> int:
        """Objects the cache may keep."""
        return self._max_entries

    @property
    def free_bytes(self) -> int:
        """Bytes not yet reserved by kept or pending objects."""
        with self._lock:
            return self._capacity - self._used_bytes

    @property
    def free_entries(self) -> int:
        """Objects that can still be added without eviction."""
        with self._lock:
            return self._max_entries - len(self._entries)

    @property
    def is_off(self) -> bool:
        """Whether a fault of the local directory turned the cache off."""
        return self._off

    def wrap(self, source: Store) -> "CachedSourceStore":
        with self._lock:
            self._require_open()
        return CachedSourceStore(source, self)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Composite source cache is closed")

    @property
    def _disk(self) -> LocalStore:
        assert self._local is not None
        return self._local

    def _ensure_directory(self) -> bool:
        """Create the directory on first use; turn the cache off if it fails."""
        if self._local is not None:
            return True
        try:
            self._directory = TemporaryDirectory(prefix="cytearc-source-cache-")
            self._local = LocalStore(self._directory.name)
        except Exception as error:
            self._turn_off(error)
            return False
        return True

    def _turn_off(self, error: BaseException) -> None:
        """Stop caching after a local fault and serve reads from the sources.

        Fetches already running finish and are discarded; cancelling them could
        interrupt the bookkeeping of the fetch that hit the fault.
        """
        with self._lock:
            if self._off or self._closed:
                return
            self._off = True
        logger.warning(
            "Composite source cache turned off after a local storage fault; "
            f"reads continue from the sources: {error}"
        )
        self._release_if_idle()

    def _release_if_idle(self) -> None:
        """Remove the directory of a closed or off cache once nothing uses it.

        A failed removal keeps the directory for a later ``close`` to retry. It
        raises from ``close`` and is only logged for a cache that is off.
        """
        with self._lock:
            if not (self._closed or self._off) or self._active:
                return
            self._entries.clear()
            self._used_bytes = 0
            directory = self._directory
        if directory is None:
            return
        try:
            directory.cleanup()
        except Exception as error:
            if self._closed:
                raise
            logger.warning(f"Could not remove the composite source cache: {error}")
            return
        with self._lock:
            if self._directory is directory:
                self._directory = None
                self._local = None

    def _cancel(self, entry: _Entry) -> None:
        if not entry.abandoned and entry.task is not None:
            entry.abandoned = True
            assert entry.loop is not None
            if entry.started:
                entry.loop.call_soon_threadsafe(entry.task.cancel)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for entry in self._entries.values():
                if not entry.ready:
                    self._cancel(entry)
        self._release_if_idle()

    def _make_space(self, size: int) -> bool:
        while (
            self._used_bytes + size > self._capacity
            or len(self._entries) >= self._max_entries
        ):
            idle = next(
                (
                    entry
                    for entry in self._entries.values()
                    if entry.task is None and entry.readers == 0
                ),
                None,
            )
            if idle is None:
                return False
            if idle.ready:
                try:
                    (self._disk.root / idle.path).unlink(missing_ok=True)
                except Exception as error:
                    self._turn_off(error)
                    return False
            del self._entries[idle.key]
            self._used_bytes -= idle.reserved
        return True

    async def _wait(self, entry: _Entry) -> bool:
        future = asyncio.wrap_future(entry.future)
        future.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            return await asyncio.shield(future)
        finally:
            drain = False
            with self._lock:
                entry.waiters -= 1
                if entry.waiters == 0 and not entry.future.done():
                    self._cancel(entry)
                    drain = True
            if drain:
                try:
                    await _finish_io(future)
                except asyncio.CancelledError:
                    # The canceled caller keeps its original cancellation.
                    pass

    async def _populate(self, entry: _Entry) -> None:
        complete = False
        wrote = False
        size = 0
        value: Buffer | None = None
        error: BaseException | None = None
        try:
            with self._lock:
                entry.started = True
                if self._closed or entry.abandoned:
                    raise asyncio.CancelledError
                # A cache turned off since this entry was added declines it.
                off = self._off
            if not off:
                # Whole-object probes ask for one byte more than they keep to
                # detect an object that does not fit, and a suffix request is
                # bounded by itself. A size request is not used: some stores,
                # such as HTTP without a length header, answer it by
                # downloading the object.
                value = await _finish_io(
                    entry.source.get(
                        entry.key[1],
                        default_buffer_prototype(),
                        RangeByteRequest(0, entry.reserved + 1)
                        if entry.suffix is None
                        else SuffixByteRequest(entry.suffix),
                    )
                )
            if (
                entry.suffix is not None
                and value is not None
                and len(value) > entry.limit
            ):
                raise ValueError("Source returned more bytes than the requested suffix")
            if value is not None and len(value) <= entry.reserved:
                size = len(value)
                wrote = True
                try:
                    await _finish_io(self._disk.set(entry.path, value))
                    complete = True
                except Exception as disk_error:
                    self._turn_off(disk_error)
        except BaseException as caught:
            error = caught
        finally:
            value = None
            with self._lock:
                complete = (
                    complete
                    and not self._closed
                    and not self._off
                    and not entry.abandoned
                )
            if wrote and not complete:
                try:
                    await _finish_io(self._disk.delete(entry.path))
                except asyncio.CancelledError as cancelled:
                    # The delete has finished; the bookkeeping below must run.
                    error = error or cancelled
                except Exception as cleanup_error:
                    self._turn_off(cleanup_error)
            with self._lock:
                if complete:
                    self._used_bytes -= entry.reserved - size
                    entry.reserved = size
                    entry.ready = True
                    self._entries.move_to_end(entry.key)
                else:
                    self._used_bytes -= entry.reserved
                    entry.reserved = 0
                    if (
                        error is not None
                        or wrote
                        or self._closed
                        or self._off
                        or entry.abandoned
                    ):
                        self._entries.pop(entry.key, None)
                    else:
                        # Immutable missing or oversized objects need no second probe.
                        self._entries.move_to_end(entry.key)
                entry.task = None
                entry.loop = None
                self._active -= 1
                closed = self._closed
            try:
                self._release_if_idle()
            except BaseException as cleanup_error:
                error = cleanup_error
            if closed and (error is None or isinstance(error, asyncio.CancelledError)):
                error = RuntimeError("Composite source cache is closed")
            if error is not None:
                _release_exception_frames(error)
            if isinstance(error, asyncio.CancelledError):
                entry.future.cancel()
            elif error is not None:
                entry.future.set_exception(error)
            else:
                entry.future.set_result(complete)

    async def _prefetch(
        self,
        source: Store,
        source_key: int,
        key: str,
        max_bytes: int,
        *,
        suffix: int | None = None,
    ) -> bool:
        limit = operator.index(max_bytes)
        if limit < 1:
            raise ValueError("Source prefetch byte limit must be positive")
        if suffix is not None and limit > self._capacity:
            return False
        limit = min(limit, self._capacity)
        with self._lock:
            self._require_open()
            if self._off:
                return False
            cache_key: _CacheKey = (
                (source_key, key) if suffix is None else (source_key, key, suffix)
            )
            entry = self._entries.get(cache_key)
            if entry is not None:
                if entry.ready:
                    self._entries.move_to_end(cache_key)
                    return True
                if entry.abandoned:
                    return False
                if entry.future.done() and limit > entry.limit:
                    del self._entries[cache_key]
                    entry = None
            if entry is None:
                if suffix is not None:
                    if (
                        self._used_bytes + limit > self._capacity
                        or len(self._entries) >= self._max_entries
                    ):
                        return False
                elif not self._make_space(limit):
                    return False
                if not self._ensure_directory():
                    return False
                entry = _Entry(
                    source, cache_key, str(self._next_path), limit, limit, suffix
                )
                self._next_path += 1
                self._entries[cache_key] = entry
                self._used_bytes += limit
                self._active += 1
                entry.loop = asyncio.get_running_loop()
                entry.task = asyncio.create_task(self._populate(entry))
            entry.waiters += 1
        return await self._wait(entry)

    async def _get(
        self,
        source: Store,
        source_key: int,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None,
        suffix_limit: int = 0,
    ) -> Buffer | None:
        suffix = (
            byte_range.suffix
            if isinstance(byte_range, SuffixByteRequest)
            and 0 < byte_range.suffix <= suffix_limit
            else None
        )
        prefetched = False
        while True:
            with self._lock:
                self._require_open()
                if self._off:
                    entry = None
                    break
                entry = self._entries.get((source_key, key))
                if (entry is None or not entry.ready) and suffix is not None:
                    entry = self._entries.get((source_key, key, suffix))
                    if (
                        entry is not None
                        and not entry.ready
                        and entry.future.done()
                        and not entry.future.result()
                    ):
                        self._entries.move_to_end(entry.key)
                        return None
                if entry is not None and entry.ready:
                    entry.readers += 1
                    self._active += 1
                    self._entries.move_to_end(entry.key)
                    break
            if not prefetched and suffix is not None:
                prefetched = True
                await self._prefetch(source, source_key, key, suffix, suffix=suffix)
            else:
                entry = None
                break
        if entry is None:
            return await source.get(key, prototype, byte_range)
        value: Buffer | None = None
        try:
            try:
                value = await _finish_io(
                    self._disk.get(
                        entry.path,
                        prototype,
                        byte_range if entry.suffix is None else None,
                    )
                )
                if value is None:
                    raise FileNotFoundError(
                        f"Cached composite source object is missing: {key}"
                    )
            except Exception as disk_error:
                value = None
                self._turn_off(disk_error)
        finally:
            with self._lock:
                entry.readers -= 1
                self._active -= 1
            self._release_if_idle()
        if value is None:
            return await source.get(key, prototype, byte_range)
        return value


class CachedSourceStore(WrapperStore[Store]):
    """A read-only view of one source store that reads through a ``SourceCache``."""

    def __init__(self, source: Store, cache: SourceCache) -> None:
        super().__init__(source)
        self._cache = cache
        self._source_key = id(source)
        self._suffix_limits: dict[str, int] = {}

    @property
    def cache(self) -> SourceCache:
        """The cache that this view reads through."""
        return self._cache

    @property
    def source_key(self) -> int:
        """Identity of the wrapped source within its cache."""
        return self._source_key

    def cache_shard_indexes(self, path: str, max_bytes: int) -> None:
        """Keep the shard-index suffixes of the array at ``path`` once read.

        Each sharded read fetches a shard's index from its end. Caching index
        suffixes of at most ``max_bytes`` (64 KiB at most) lets later reads of
        the same shard skip that request.
        """
        limit = min(64 * 1024, operator.index(max_bytes))
        if limit < 1:
            raise ValueError("Source suffix byte limit must be positive")
        prefix = f"{path}/" if path else ""
        with self._cache._lock:
            self._cache._require_open()
            self._suffix_limits[prefix] = min(
                limit, self._suffix_limits.get(prefix, limit)
            )

    def _with_store(self, store: Store) -> Self:
        clone = copy.copy(self)
        clone._store = store
        return clone

    def with_read_only(self, read_only: bool = False) -> Self:
        if not read_only:
            raise ValueError("Cached composite sources are read-only")
        return self._with_store(self._store)

    @property
    def read_only(self) -> bool:
        return True

    @property
    def _supports_sync_io(self) -> bool:
        return False

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    async def prefetch(self, key: str, max_bytes: int) -> bool:
        """Fetch one whole object into the cache; return whether it is cached."""
        return await self._cache._prefetch(
            self._store, self._source_key, key, max_bytes
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        with self._cache._lock:
            suffix_limit = next(
                (
                    limit
                    for prefix, limit in self._suffix_limits.items()
                    if key.startswith(prefix)
                ),
                0,
            )
        return await self._cache._get(
            self._store, self._source_key, key, prototype, byte_range, suffix_limit
        )

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, request) for key, request in key_ranges]

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        async for item in Store._get_many(self, requests):
            yield item

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        raise TypeError("Cached composite sources require asynchronous reads")

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        **options: Any,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        async for batch in Store.get_ranges(
            self, key, byte_ranges, prototype=prototype, **options
        ):
            yield batch

    async def set(self, key: str, value: Buffer) -> None:
        raise ValueError("Cached composite sources are read-only")

    def _check_writable(self) -> None:
        raise ValueError("Cached composite sources are read-only")

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        raise ValueError("Cached composite sources are read-only")

    def set_sync(self, key: str, value: Buffer) -> None:
        raise ValueError("Cached composite sources are read-only")

    def delete_sync(self, key: str) -> None:
        raise ValueError("Cached composite sources are read-only")

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        raise ValueError("Cached composite sources are read-only")

    async def delete(self, key: str) -> None:
        raise ValueError("Cached composite sources are read-only")

    async def delete_dir(self, prefix: str) -> None:
        raise ValueError("Cached composite sources are read-only")

    async def clear(self) -> None:
        raise ValueError("Cached composite sources are read-only")
