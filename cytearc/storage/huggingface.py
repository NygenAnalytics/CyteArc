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
from huggingface_hub import hf_hub_url
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from huggingface_hub.hf_file_system import (
    HfFileSystem,
    HfFileSystemResolvedBucketPath,
    HfFileSystemResolvedRepositoryPath,
)
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    parse_ratelimit_headers,
)
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore
from zarr.storage._fsspec import ALLOWED_EXCEPTIONS
from zarr.storage._utils import _dereference_path

from .async_execution import zarr_host_threads
from .stores import REMOTE_RETRY_TIMEOUT, byte_request_span

# A bucket's lookup sets no timeout, as huggingface_hub sends it, so a request
# that the server or the network leaves unanswered would wait forever. Each
# attempt at a request of the store is given this many seconds.
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
    """The objects of a store that one listing found, by key, with their sizes in bytes.

    ``url`` returns the resolver URL that reads a key.
    """

    sizes: dict[str, int]
    url: Callable[[str], str]


def _within(root: str, key: str) -> str:
    return f"{root}/{key}" if root else key


type _Resolved = HfFileSystemResolvedBucketPath | HfFileSystemResolvedRepositoryPath


def _file_url(fs: HfFileSystem, resolved: _Resolved, path: str) -> str:
    """Return the URL of Hugging Face's resolver for the file at ``path``.

    ``path`` is a path in the bucket or the repository that ``resolved``
    names.
    """
    if isinstance(resolved, HfFileSystemResolvedBucketPath):
        # As HfFileSystem.url builds a bucket file's URL.
        return f"{fs.endpoint}/buckets/{resolved.bucket_id}/resolve/{quote(path)}"
    return hf_hub_url(
        resolved.repo_id,
        path,
        repo_type=resolved.repo_type,
        revision=resolved.revision,
        endpoint=fs.endpoint,
    )


def _list_tree(
    fs: HfFileSystem, resolved: _Resolved, path: str, *, recursive: bool, timeout: float
) -> Iterable[dict[str, Any]]:
    """List the folder at ``path`` of the bucket or repository that ``resolved`` names.

    The entries are those of Hugging Face's API, each with its ``type`` and
    ``path``, and a file's ``size``. The listing goes to the API directly
    rather than through the file system's directory cache, which keeps a
    listing for the life of the process and which concurrent listings can
    leave half filled. A bucket matches a recursive listing's ``path`` as
    text.

    Each page is one GET, the request that huggingface_hub's
    ``list_bucket_tree`` and ``list_repo_tree`` send, without their retries.
    From huggingface_hub 2.1.0 on, those functions retry each page within an
    attempt of the store: their retries would multiply the store's, and a
    rate limit that huggingface_hub waited out would read as an attempt that
    got no answer. The listing takes at most ``timeout`` seconds, the time of
    the store's attempt that sends it: each page waits for the time left, and
    the listing raises TimeoutError instead of asking for a page after it, so
    that a listing whose attempt the store gave up on sends no more.
    """
    folder = "/" + quote(path, safe="") if path else ""
    if isinstance(resolved, HfFileSystemResolvedBucketPath):
        url = f"{fs.endpoint}/api/buckets/{resolved.bucket_id}/tree{folder}"
        params = {"recursive": recursive}
    else:
        revision = quote(resolved.revision, safe="")
        url = (
            f"{fs.endpoint}/api/{resolved.repo_type}s/{resolved.repo_id}"
            f"/tree/{revision}{folder}"
        )
        params = {"recursive": recursive, "expand": False}
    headers = build_hf_headers(token=fs.token)
    deadline = time.monotonic() + timeout

    def left() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError(f"The listing of {url} took more than {timeout:g} s")
        return seconds

    response = get_session().get(url, params=params, headers=headers, timeout=left())
    while True:
        hf_raise_for_status(response)
        yield from response.json()
        # The link to the next page carries the listing's query.
        page = response.links.get("next", {}).get("url")
        if page is None:
            return
        response = get_session().get(page, headers=headers, timeout=left())


def _list_store(
    fs: HfFileSystem,
    path: str,
    within: str = "",
    timeout: float = REQUEST_TIMEOUT,
) -> _StoreListing:
    """List the store at ``path`` with one recursive tree listing.

    ``within`` limits the listing to that key or folder of the store and the
    keys below it. A repository's tree lists folders only, so a key of a
    repository is looked up, in ``timeout`` seconds, when no folder has its
    name. A bucket, repository, revision or path that does not exist lists as
    empty.
    """
    try:
        resolved = fs.resolve_path(path)
    except FileNotFoundError:
        return _StoreListing({}, lambda key: "")
    root = resolved.path
    listed = _within(root, within) if within else root
    tree = _list_tree(fs, resolved, listed, recursive=True, timeout=timeout)
    prefix = f"{root}/" if root else ""
    sizes: dict[str, int] = {}
    try:
        for entry in tree:
            # Only files are keys: a repository's folders list with a size of
            # 0. A prefix matches as text, so "data.zarr" also lists
            # "data.zarr2/..."; only keys below the root, and within the
            # folder asked for, are kept.
            if entry["type"] != "file" or not entry["path"].startswith(prefix):
                continue
            key = entry["path"][len(prefix) :]
            if not within or key == within or key.startswith(f"{within}/"):
                sizes[key] = int(entry["size"])
    except EntryNotFoundError:
        sizes = {}
        if within and not isinstance(resolved, HfFileSystemResolvedBucketPath):
            found = _file_entry(fs, _dereference_path(path, within), timeout)
            if found is not None:
                sizes[within] = int(found["size"])
    return _StoreListing(sizes, lambda key: _file_url(fs, resolved, _within(root, key)))


def _list_folder(fs: HfFileSystem, path: str, timeout: float) -> list[str]:
    """Return the names of the files and folders in the folder at ``path``.

    The folder is listed when asked, within ``timeout`` seconds, as
    ``_list_tree`` lists. A folder that does not exist lists as empty.
    """
    try:
        resolved = fs.resolve_path(path)
    except FileNotFoundError:
        return []
    folder = resolved.path
    start = f"{folder}/" if folder else ""
    try:
        # A bucket matches the folder as text, so listing "c/1" also names
        # "c/10"; only the folder's own entries are kept.
        return [
            entry["path"][len(start) :]
            for entry in _list_tree(
                fs, resolved, folder, recursive=False, timeout=timeout
            )
            if entry["path"].startswith(start)
        ]
    except EntryNotFoundError:
        return []


def _paths_info(
    fs: HfFileSystem,
    url: str,
    timeout: float,
    *,
    json: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Send one lookup of a path to ``url`` and return the entry of the file it names.

    ``json`` or ``data`` is the request's body. The answer lists the path if
    it names a file or a folder, and only a file is a key, so a folder, like
    a missing path, gives None.
    """
    response = get_session().post(
        url,
        headers=build_hf_headers(token=fs.token),
        timeout=timeout,
        json=json,
        data=data,
    )
    hf_raise_for_status(response)
    files = (entry for entry in response.json() if entry.get("type") == "file")
    return next(files, None)


def _file_entry(fs: HfFileSystem, path: str, timeout: float) -> dict[str, Any] | None:
    """Return Hugging Face's entry for the file at ``path`` when asked, or None.

    Hugging Face's file system answers a lookup from the listing of the
    file's folder that it keeps, which misses a file that another process
    wrote after the listing and still holds one that it deleted, and the Hub
    documents a bucket's non-recursive listings as not strongly consistent.
    This lookup asks the bucket's or the repository's ``paths-info`` endpoint
    about the one path instead, with one request whatever the size of its
    folder, and keeps no answer. The entry holds the file's ``size``.

    Raises:
        FileNotFoundError: If the bucket, repository or revision does not
            exist.
    """
    resolved = fs.resolve_path(path)
    if isinstance(resolved, HfFileSystemResolvedBucketPath):
        url = f"{fs.endpoint}/api/buckets/{resolved.bucket_id}/paths-info"
        return _paths_info(fs, url, timeout, json={"paths": [resolved.path]})
    revision = quote(resolved.revision, safe="")
    url = (
        f"{fs.endpoint}/api/{resolved.repo_type}s/{resolved.repo_id}"
        f"/paths-info/{revision}"
    )
    # Form-encoded, as huggingface_hub's HfApi.get_paths_info sends it.
    return _paths_info(
        fs,
        url,
        timeout,
        data={"paths": [resolved.path_in_repo], "expand": False},
    )


def _fetch(
    fs: HfFileSystem, url: str, timeout: float, asked: str | None
) -> httpx2.Response:
    """GET ``url`` with the file system's token, and the ``Range`` ``asked``, if any."""
    headers = build_hf_headers(token=fs.token)
    if asked is not None:
        headers["Range"] = asked
    return get_session().get(url, headers=headers, timeout=timeout)


def _read_file(
    fs: HfFileSystem, path: str, byte_range: ByteRequest | None, timeout: float
) -> bytes | None:
    """Return the bytes of the file at ``path`` that ``byte_range`` asks for, or None.

    The read is one GET of the file's resolver URL, with the ``Range`` that
    asks for the bytes, as a read-only store reads a listed key, and a 404
    reads as a missing file. A read of no bytes only looks the file up. A
    bucket or repository that does not exist holds no file.
    """
    try:
        resolved = fs.resolve_path(path)
    except FileNotFoundError:
        return None
    asked: str | None
    match byte_range:
        case None:
            asked = None
        case RangeByteRequest(start=start, end=end) if end > start:
            asked = f"bytes={start}-{end - 1}"
        case OffsetByteRequest(offset=offset):
            asked = f"bytes={offset}-"
        case SuffixByteRequest(suffix=suffix) if suffix > 0:
            asked = f"bytes=-{suffix}"
        case _:
            # A range of no bytes: an empty one, or a suffix of none.
            return None if _file_entry(fs, path, timeout) is None else b""
    response = _fetch(fs, _file_url(fs, resolved, resolved.path), timeout, asked)
    if response.status_code == 404:
        return None
    if response.status_code == 416:
        # The bytes asked for start past the file's end.
        return b""
    hf_raise_for_status(response)
    data = response.content
    if response.status_code == 200 and asked is not None:
        # A server that ignores the range sends the whole file.
        start, stop = byte_request_span(byte_range, len(data))
        data = data[start:stop]
    return data


class _FetchAbandoned(Exception):
    """The fetch that a reader waited for ended with its event loop."""


class _SharedFetches[T]:
    """Fetches shared by key, each taken once and awaited from any event loop.

    The first reader of a key starts its fetch on its own event loop, and every
    reader of that key, on that loop or another, waits for the same
    thread-safe future. Cancelling a reader cancels only its wait. A fetch
    that fails is dropped, so the next reader fetches again; a reader that
    waited for a fetch whose event loop ended before it finished fetches again
    itself.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: dict[str, concurrent.futures.Future[T]] = {}

    async def get(self, key: str, fetch: Callable[[], Awaitable[T]]) -> T:
        while True:
            with self._lock:
                future = self._futures.get(key)
                starting = future is None
                if future is None:
                    future = self._futures[key] = concurrent.futures.Future()
                    # A running future cannot be cancelled, so a reader that is
                    # cancelled leaves the fetch to the others.
                    future.set_running_or_notify_cancel()
            if starting:
                task = asyncio.ensure_future(fetch())
                task.add_done_callback(functools.partial(self._settle, key, future))
            try:
                return await asyncio.wrap_future(future)
            except _FetchAbandoned:
                continue

    def _settle(
        self,
        key: str,
        future: concurrent.futures.Future[T],
        task: asyncio.Future[T],
    ) -> None:
        # Reading the task's error here also consumes it when every reader
        # that waited for it was cancelled.
        error: BaseException | None = (
            _FetchAbandoned() if task.cancelled() else task.exception()
        )
        if error is None:
            future.set_result(task.result())
            return
        with self._lock:
            if self._futures.get(key) is future:
                del self._futures[key]
        future.set_exception(error)

    def clear(self) -> None:
        with self._lock:
            self._futures.clear()


class HfReadStore(FsspecStore):
    """Read one published store with unique listings and metadata cached in RAM.

    A read-only store keeps each metadata document it reads, shared by its
    readers on any event loop. It lists its objects once, recursively, the
    first time it needs them, and keeps each object's size. That listing
    answers ``exists``, ``getsize``, ``list``, ``list_prefix`` and
    ``list_dir``, a key it does not hold reads as missing without a request,
    and a read of a key it holds is one GET of the key's resolver URL for
    exactly the bytes asked for. The listing is a snapshot: the store does not
    see objects written after it, and a listed key that is no longer there
    raises ``OSError``. A writable store keeps no listing: it reads a key with
    one GET of the key's resolver URL for the bytes asked, a 404 reading as a
    missing key, looks a key up, reads its size or reads no bytes of it with
    one lookup of that key, and lists keys with a listing made when asked, so
    it sees a key that another process wrote or deleted since its last
    request. Zarr's read-only views of a writable store read as it does. It
    reads a key as missing only when the server reports it missing, so a
    lookup that fails never makes ``set_if_not_exists`` write over a key. That call still
    looks the key up and then writes it: a bucket's uploads take no
    condition, nor do the commits that Hugging Face's file system makes to a
    repository, so another process can write the key in between.

    Each attempt at a request of the store, a read, a lookup or a listing, is
    given ``request_timeout`` seconds, 60 by default. An attempt that times
    out, loses its connection, or meets a 408 or 5xx response is retried with
    exponential backoff until ``retry_timeout`` seconds, 180 by default, after
    the request began, each attempt cut to the time left; the request then
    raises ``TimeoutError``, which names the request, the store and that time.
    A 429 response is retried after the delay its headers ask for, at most
    three times and only for a delay of at most 300 s, and that wait does not
    count toward ``retry_timeout``. Other failures, such as a 401 or 403
    response, raise at once. ``from_url`` takes both timeouts from
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
        self._metadata = _SharedFetches[bytes | None]()
        self._listing = _SharedFetches[_StoreListing]()
        # Whether the store reads through one listing, as a store opened
        # read-only does; Zarr's read-only views of a writable store do not.
        self._listed = self.read_only

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
        store._listed = read_only and self._listed
        if store._listed:
            store._listing = self._listing
        return store

    def _hf_fs(self) -> HfFileSystem | None:
        """Return Hugging Face's file system, or None for another file system."""
        fs = getattr(self.fs, "sync_fs", self.fs)
        return fs if isinstance(fs, HfFileSystem) else None

    def _listed_fs(self) -> HfFileSystem | None:
        """Return Hugging Face's file system when the store reads through its listing."""
        return self._hf_fs() if self._listed else None

    async def _store_listing(self, fs: HfFileSystem) -> _StoreListing:
        return await self._listing.get(
            "",
            functools.partial(
                self._request,
                f"Listing {self._location}",
                lambda: asyncio.to_thread(
                    _list_store, fs, self.path, "", self.request_timeout
                ),
            ),
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
            asked = f"bytes={start}-{stop - 1}"
            response = _fetch(fs, url, self.request_timeout, asked)
            if response.status_code == 404:
                # Not FileNotFoundError: Zarr reads that from a partial shard
                # read as a missing shard, which fills its chunks with zeros.
                raise OSError(
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
        # Hugging Face's file system runs each request on a thread, and a
        # bucket's lookup sets no timeout, so the thread of an attempt
        # abandoned here can keep waiting until the server or the network
        # closes its connection; the request itself moves on.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.retry_timeout
        attempts = throttled = 0
        while True:
            attempts += 1
            allowed = min(self.request_timeout, deadline - loop.time())
            timer = asyncio.timeout(allowed)
            try:
                # A request waits on the network, so it runs on the host's
                # threads even where the store's reads decode on fewer.
                async with timer:
                    with zarr_host_threads():
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
        request = f"Reading {key!r} from {self._location}"
        files = self._hf_fs()
        if files is None:
            read = super().get
            return await self._request(
                request, lambda: read(key, prototype, byte_range)
            )
        path = _dereference_path(self.path, key)
        data = await self._request(
            request,
            lambda: asyncio.to_thread(
                _read_file, files, path, byte_range, self.request_timeout
            ),
        )
        return None if data is None else prototype.buffer.from_bytes(data)

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
            not self._listed
            or byte_range is not None
            or key.rsplit("/", 1)[-1]
            not in {"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"}
        ):
            return await self._read(key, prototype, byte_range)
        # Readers of a key share one fetch, from whichever event loop they run on.
        raw = await self._metadata.get(key, lambda: self._read_metadata(key, prototype))
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
        path = _dereference_path(self.path, key)
        files = self._hf_fs()

        async def look_up() -> bool:
            # Hugging Face's exists answers False for any OSError, and the
            # HfHubHTTPError of a failed request is one. A lookup by path
            # raises FileNotFoundError only for a bucket, repository or
            # revision that does not exist, and another file system's info
            # only for a missing key; every other failure reaches _request,
            # which retries it or raises it.
            try:
                if files is not None:
                    entry = await asyncio.to_thread(
                        _file_entry, files, path, self.request_timeout
                    )
                    return entry is not None
                await self.fs._info(path)
            except FileNotFoundError:
                return False
            return True

        return await self._request(f"Looking up {key!r} in {self._location}", look_up)

    async def getsize(self, key: str) -> int:
        fs = self._listed_fs()
        if fs is not None:
            size = (await self._store_listing(fs)).sizes.get(key)
            if size is None:
                raise FileNotFoundError(key)
            return size
        request = f"Reading the size of {key!r} in {self._location}"
        files = self._hf_fs()
        if files is None:
            getsize = super().getsize
            return await self._request(request, lambda: getsize(key))
        path = _dereference_path(self.path, key)
        entry = await self._request(
            request,
            lambda: asyncio.to_thread(_file_entry, files, path, self.request_timeout),
        )
        if entry is None:
            raise FileNotFoundError(key)
        return int(entry["size"])

    def close(self) -> None:
        self._metadata.clear()
        self._listing.clear()
        super().close()

    def _forget_listings(self) -> None:
        """Drop the listings of Hugging Face's file system before it deletes a folder.

        The file system lists a folder before it deletes the keys in it, from
        its cached listings, and a delete drops only the folder's listing and
        its parents', so a key written below a subfolder since an earlier delete
        would stay. The store's reads, lookups and listings do not use these
        listings. Only the listings go: ``invalidate_cache()`` would also drop
        the file system's record of which repositories exist, which it would
        then check again.
        """
        files = self._hf_fs()
        if files is not None:
            files.dircache.clear()

    async def delete_dir(self, prefix: str) -> None:
        self._forget_listings()
        await super().delete_dir(prefix)

    async def _list_now(
        self, fs: HfFileSystem, request: str, within: str = ""
    ) -> list[str]:
        """Return the keys of a writable store, or those within a key or folder, listed now."""
        listing = await self._request(
            request,
            lambda: asyncio.to_thread(
                _list_store, fs, self.path, within, self.request_timeout
            ),
        )
        return sorted(listing.sizes)

    async def list(self) -> AsyncIterator[str]:
        fs = self._listed_fs()
        if fs is not None:
            for key in sorted((await self._store_listing(fs)).sizes):
                yield key
            return
        request = f"Listing {self._location}"
        files = self._hf_fs()
        if files is not None:
            for key in await self._list_now(files, request):
                yield key
            return
        listing = super().list
        for name in await self._request(request, lambda: _names(listing())):
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
        request = f"Listing the keys under {prefix!r} in {self._location}"
        files = self._hf_fs()
        if files is not None:
            for key in await self._list_now(files, request, prefix.rstrip("/")):
                yield key
            return
        listing = super().list_prefix
        for name in await self._request(request, lambda: _names(listing(prefix))):
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
        request = f"Listing {prefix!r} in {self._location}"
        files = self._hf_fs()
        if files is not None:
            folder = _dereference_path(self.path, prefix)
            for name in await self._request(
                request,
                lambda: asyncio.to_thread(
                    _list_folder, files, folder, self.request_timeout
                ),
            ):
                yield name
            return
        listing = super().list_dir
        names = await self._request(request, lambda: _names(listing(prefix)))
        # A file system can list a name twice, as Hugging Face's did when
        # concurrent listings appended to its directory cache.
        seen: set[str] = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                yield name
