import asyncio
import concurrent.futures
import functools
import math
import numbers
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast
from urllib.parse import quote

import httpx2
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from huggingface_hub.hf_file_system import (
    HfFileSystem,
    HfFileSystemResolvedBucketPath,
)
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    parse_ratelimit_headers,
)
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore
from zarr.storage._fsspec import ALLOWED_EXCEPTIONS

from .stores import REMOTE_RETRY_TIMEOUT, byte_request_span

# huggingface_hub sets no timeout on a bucket's lookup or its tree listings, so
# a request that the server or the network leaves unanswered would wait
# forever. Each attempt at a request of the store is given this many seconds.
REQUEST_TIMEOUT = 60.0
# The seconds in which a request is retried, as object stores retry theirs.
RETRY_TIMEOUT = REMOTE_RETRY_TIMEOUT.total_seconds()
_TIMEOUT_OPTIONS = ("request_timeout", "retry_timeout")

# Failures that a later attempt may not meet: no answer in time, a dropped or
# refused connection, and a response cut short, as huggingface_hub retries its
# own downloads on them.
_TRANSPORT_ERRORS = (
    TimeoutError,
    httpx2.TimeoutException,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
)


def retry_delay(error: Exception, attempt: int) -> float:
    delay = float(2 ** (attempt + 1))
    response = getattr(error, "response", None)
    if response is None:
        return delay
    headers = response.headers
    raw = headers.get("Retry-After")
    if raw:
        try:
            delay = max(delay, float(raw))
        except ValueError:
            try:
                moment = parsedate_to_datetime(raw)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=UTC)
                delay = max(delay, (moment - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    raw_reset = headers.get("RateLimit-Reset") or headers.get("X-RateLimit-Reset")
    if raw_reset:
        try:
            reset = float(raw_reset)
            delay = max(delay, reset - time.time() if reset > 1_000_000_000 else reset)
        except ValueError:
            pass
    rate_limit = parse_ratelimit_headers(headers)
    if rate_limit is not None and rate_limit.remaining == 0:
        delay = max(delay, float(rate_limit.reset_in_seconds))
    return delay


def _seconds(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number of seconds, not {value!r}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"{name} must be a positive, finite number of seconds, not {value!r}"
        )
    return seconds


async def _names(listing: AsyncIterator[str]) -> list[str]:
    return [name async for name in listing]


@dataclass(frozen=True)
class _StoreListing:
    """Every object of a read-only store, by key, with its size in bytes.

    ``url`` returns the resolver URL that reads a key.
    """

    sizes: dict[str, int]
    url: Callable[[str], str]


def _within(root: str, key: str) -> str:
    return f"{root}/{key}" if root else key


def _list_store(fs: HfFileSystem, path: str) -> _StoreListing:
    """List the store at ``path`` with one recursive tree listing.

    A bucket, repository, revision or path that does not exist lists as empty.
    The listing goes to Hugging Face's API directly rather than through the
    file system's directory cache, which concurrent listings can leave
    half filled.
    """
    try:
        resolved = fs.resolve_path(path)
    except FileNotFoundError:
        return _StoreListing({}, lambda key: "")
    api = HfApi(endpoint=fs.endpoint, token=fs.token)
    url: Callable[[str], str]
    if isinstance(resolved, HfFileSystemResolvedBucketPath):
        bucket, root = resolved.bucket_id, resolved.path
        tree: Iterable[Any] = api.list_bucket_tree(bucket, root or None, recursive=True)

        # As HfFileSystem.url builds a bucket file's URL.
        def url(key: str) -> str:
            return f"{fs.endpoint}/buckets/{bucket}/resolve/{quote(_within(root, key))}"

    else:
        repository = resolved
        root = repository.path_in_repo
        tree = api.list_repo_tree(
            repository.repo_id,
            root or None,
            recursive=True,
            revision=repository.revision,
            repo_type=repository.repo_type,
        )

        def url(key: str) -> str:
            return hf_hub_url(
                repository.repo_id,
                _within(root, key),
                repo_type=repository.repo_type,
                revision=repository.revision,
                endpoint=fs.endpoint,
            )

    prefix = f"{root}/" if root else ""
    sizes: dict[str, int] = {}
    try:
        for entry in tree:
            # Folders carry no size. A prefix matches as text, so "data.zarr"
            # also lists "data.zarr2/..."; only keys below the root are kept.
            size = getattr(entry, "size", None)
            if size is not None and entry.path.startswith(prefix):
                sizes[entry.path[len(prefix) :]] = int(size)
    except EntryNotFoundError:
        sizes = {}
    return _StoreListing(sizes, url)


class _ListingAbandoned(Exception):
    """The listing that a caller waited for ended with its event loop."""


class _SharedListing:
    """One listing of a store, taken once and awaited from any event loop.

    The first caller starts the listing on its event loop, and every caller,
    on that loop or another, waits for the same thread-safe future. Cancelling
    a caller cancels only its wait. A listing that fails is taken again by the
    next caller, and one whose event loop ended before it finished is taken
    again by a caller that was waiting for it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._future: concurrent.futures.Future[_StoreListing] | None = None

    async def get(self, take: Callable[[], Awaitable[_StoreListing]]) -> _StoreListing:
        while True:
            with self._lock:
                future = self._future
                starting = future is None
                if future is None:
                    future = self._future = concurrent.futures.Future()
                    # A running future cannot be cancelled, so a waiter that is
                    # cancelled leaves the listing to the others.
                    future.set_running_or_notify_cancel()
            if starting:
                task = asyncio.ensure_future(take())
                task.add_done_callback(functools.partial(self._settle, future))
            try:
                return await asyncio.wrap_future(future)
            except _ListingAbandoned:
                continue

    def _settle(
        self,
        future: concurrent.futures.Future[_StoreListing],
        task: asyncio.Future[_StoreListing],
    ) -> None:
        error: BaseException | None = (
            _ListingAbandoned() if task.cancelled() else task.exception()
        )
        if error is None:
            future.set_result(task.result())
            return
        with self._lock:
            if self._future is future:
                self._future = None
        future.set_exception(error)

    def clear(self) -> None:
        with self._lock:
            self._future = None


class HfReadStore(FsspecStore):
    """Read one published store with unique listings and metadata cached in RAM.

    A read-only store lists its objects once, recursively, the first time it
    needs them, and keeps each object's size. That listing answers ``exists``,
    ``getsize``, ``list``, ``list_prefix`` and ``list_dir``, a key it does not
    hold reads as missing without a request, and a read of a key it holds is
    one GET of the key's resolver URL for exactly the bytes asked for. The
    listing is a snapshot: the store does not see objects written after it,
    and a listed key that is no longer there raises ``FileNotFoundError``. A
    writable store reads through Hugging Face's file system.

    Each attempt at a request of the store, a read or a listing, is given
    ``request_timeout`` seconds, 60 by default. An attempt that times out, loses
    its connection, or meets a 408 or 5xx response is retried with exponential
    backoff until ``retry_timeout`` seconds, 180 by default, after the request
    began, each attempt cut to the time left; the request then raises
    ``TimeoutError``, which names the request, the store and that time. A 429
    response is retried after the delay its headers ask for, at most three
    times and only for a delay of at most 300 s, and that wait does not count
    toward ``retry_timeout``. ``from_url`` takes both timeouts from
    ``storage_options``.

    Raises:
        TypeError: If a timeout is not a number of seconds.
        ValueError: If a timeout is not positive and finite.
    """

    def __init__(
        self,
        *args: Any,
        request_timeout: float = REQUEST_TIMEOUT,
        retry_timeout: float = RETRY_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.request_timeout = _seconds("request_timeout", request_timeout)
        self.retry_timeout = _seconds("retry_timeout", retry_timeout)
        self._location = self.fs.unstrip_protocol(self.path)
        self._metadata: dict[str, asyncio.Task[bytes | None]] = {}
        self._listing = _SharedListing()

    @classmethod
    def from_url(
        cls,
        url: str,
        storage_options: dict[str, Any] | None = None,
        read_only: bool = False,
        allowed_exceptions: tuple[type[Exception], ...] = ALLOWED_EXCEPTIONS,
    ) -> "HfReadStore":
        """Open the store at ``url``, taking its timeouts from ``storage_options``.

        ``request_timeout`` and ``retry_timeout`` are seconds, as the class
        describes. The other options open Hugging Face's file system.
        """
        options = dict(storage_options or {})
        timeouts = {
            name: _seconds(name, options.pop(name))
            for name in _TIMEOUT_OPTIONS
            if name in options
        }
        store = super().from_url(
            url,
            storage_options=options,
            read_only=read_only,
            allowed_exceptions=allowed_exceptions,
        )
        for name, seconds in timeouts.items():
            setattr(store, name, seconds)
        return cast(HfReadStore, store)

    def with_read_only(self, read_only: bool = False) -> "HfReadStore":
        store = cast(HfReadStore, super().with_read_only(read_only))
        store.request_timeout = self.request_timeout
        store.retry_timeout = self.retry_timeout
        if read_only and self.read_only:
            store._listing = self._listing
        return store

    def _listed_fs(self) -> HfFileSystem | None:
        """Return Hugging Face's file system when the store reads through its listing."""
        fs = getattr(self.fs, "sync_fs", self.fs)
        return fs if self.read_only and isinstance(fs, HfFileSystem) else None

    async def _store_listing(self, fs: HfFileSystem) -> _StoreListing:
        return await self._listing.get(
            functools.partial(
                self._request,
                f"Listing {self._location}",
                lambda: asyncio.to_thread(_list_store, fs, self.path),
            )
        )

    async def _read_listed(
        self,
        fs: HfFileSystem,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None,
    ) -> Buffer | None:
        listing = await self._store_listing(fs)
        size = listing.sizes.get(key)
        if size is None:
            return None
        start, stop = byte_request_span(byte_range, size)
        if stop <= start:
            return prototype.buffer.from_bytes(b"")
        url = listing.url(key)

        def fetch() -> bytes:
            response = get_session().get(
                url,
                headers={
                    **build_hf_headers(token=fs.token),
                    "Range": f"bytes={start}-{stop - 1}",
                },
                timeout=self.request_timeout,
            )
            if response.status_code == 404:
                raise FileNotFoundError(
                    f"{key!r} is listed in {self._location}, but reading it "
                    "answered 404: the store changed after it was listed. Open "
                    "the store again to read what it holds now."
                )
            hf_raise_for_status(response)
            data = response.content
            if response.status_code == 200 and len(data) == size:
                # A server that ignores the range sends the whole object.
                data = data[start:stop]
            if len(data) != stop - start:
                raise OSError(
                    f"Reading bytes {start} to {stop} of {key!r} from "
                    f"{self._location} returned {len(data)} bytes: the store "
                    "changed after it was listed. Open the store again to read "
                    "what it holds now."
                )
            return data

        data = await self._request(
            f"Reading {key!r} from {self._location}",
            lambda: asyncio.to_thread(fetch),
        )
        return prototype.buffer.from_bytes(data)

    async def _request[T](self, request: str, call: Callable[[], Awaitable[T]]) -> T:
        """Return what ``call`` returns, retried within the store's timeouts.

        ``request`` names the request in the error that ends its retries.
        """
        # Hugging Face's file system runs each request on a thread, and
        # huggingface_hub sets no timeout on its listings, so the thread of an
        # attempt abandoned here can keep waiting until the server or the
        # network closes its connection; the request itself moves on.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.retry_timeout
        attempts = throttled = 0
        while True:
            attempts += 1
            allowed = min(self.request_timeout, deadline - loop.time())
            timer = asyncio.timeout(allowed)
            try:
                async with timer:
                    return await call()
            except HfHubHTTPError as error:
                status = error.response.status_code
                if status == 429:
                    if throttled == 3:
                        raise
                    delay = retry_delay(error, throttled)
                    if delay > 300:
                        raise
                    throttled += 1
                    deadline += delay
                    await asyncio.sleep(delay)
                    continue
                if status != 408 and status < 500:
                    raise
                failure: Exception = error
                reason = f"HTTP {status}"
            except _TRANSPORT_ERRORS as error:
                failure = error
                reason = (
                    f"no answer within {max(allowed, 0.0):.3g} s"
                    if timer.expired()
                    else f"{type(error).__name__}: {error}"
                )
            delay = retry_delay(failure, attempts - throttled - 1)
            if loop.time() + delay >= deadline:
                raise TimeoutError(
                    f"{request} did not succeed within {self.retry_timeout:g} s: "
                    f"{attempts} attempt{'s' if attempts > 1 else ''} failed, the "
                    f"last with {reason}. To wait longer, pass more seconds in "
                    "storage_options as request_timeout, the time for each "
                    "attempt, or retry_timeout, the time for the request."
                ) from failure
            await asyncio.sleep(delay)

    async def _read(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        fs = self._listed_fs()
        if fs is not None:
            return await self._read_listed(fs, key, prototype, byte_range)
        read = super().get
        return await self._request(
            f"Reading {key!r} from {self._location}",
            lambda: read(key, prototype, byte_range),
        )

    async def _read_metadata(
        self, key: str, prototype: BufferPrototype
    ) -> bytes | None:
        value = await self._read(key, prototype)
        return None if value is None else value.to_bytes()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if (
            not self.read_only
            or byte_range is not None
            or key.rsplit("/", 1)[-1]
            not in {"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"}
        ):
            return await self._read(key, prototype, byte_range)
        task = self._metadata.get(key)
        if task is None:

            def discard_failed(completed: asyncio.Task[bytes | None]) -> None:
                # Consume errors even if every waiting reader was cancelled.
                if completed.cancelled() or completed.exception() is not None:
                    if self._metadata.get(key) is completed:
                        del self._metadata[key]

            task = asyncio.create_task(self._read_metadata(key, prototype))
            self._metadata[key] = task
            task.add_done_callback(discard_failed)
        # Cancelling one reader must not cancel a fetch shared by other readers.
        if not task.done():
            await asyncio.wait((task,))
        raw = task.result()
        return None if raw is None else prototype.buffer.from_bytes(raw)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self._read(key, prototype, byte_range)
            for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        fs = self._listed_fs()
        if fs is not None:
            return key in (await self._store_listing(fs)).sizes
        exists = super().exists
        return await self._request(
            f"Looking up {key!r} in {self._location}", lambda: exists(key)
        )

    async def getsize(self, key: str) -> int:
        fs = self._listed_fs()
        if fs is not None:
            size = (await self._store_listing(fs)).sizes.get(key)
            if size is None:
                raise FileNotFoundError(key)
            return size
        getsize = super().getsize
        return await self._request(
            f"Reading the size of {key!r} in {self._location}", lambda: getsize(key)
        )

    def close(self) -> None:
        self._metadata.clear()
        self._listing.clear()
        super().close()

    async def list(self) -> AsyncIterator[str]:
        fs = self._listed_fs()
        if fs is not None:
            for key in sorted((await self._store_listing(fs)).sizes):
                yield key
            return
        listing = super().list
        for name in await self._request(
            f"Listing {self._location}", lambda: _names(listing())
        ):
            yield name

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        fs = self._listed_fs()
        if fs is not None:
            # As Hugging Face's file system finds them: the prefix names a
            # directory, or an object.
            directory = prefix.rstrip("/")
            for key in sorted((await self._store_listing(fs)).sizes):
                if not directory or key == directory or key.startswith(f"{directory}/"):
                    yield key
            return
        listing = super().list_prefix
        for name in await self._request(
            f"Listing the keys under {prefix!r} in {self._location}",
            lambda: _names(listing(prefix)),
        ):
            yield name

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        fs = self._listed_fs()
        if fs is not None:
            directory = prefix.rstrip("/")
            start = f"{directory}/" if directory else ""
            children = {
                key[len(start) :].split("/", 1)[0]
                for key in (await self._store_listing(fs)).sizes
                if key.startswith(start)
            }
            for name in sorted(children):
                yield name
            return
        listing = super().list_dir
        names = await self._request(
            f"Listing {prefix!r} in {self._location}",
            lambda: _names(listing(prefix)),
        )
        # Concurrent HF listings can append the same paths to its directory cache.
        seen: set[str] = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                yield name
