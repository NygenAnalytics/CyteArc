import asyncio
import math
import numbers
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast

import httpx2
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import parse_ratelimit_headers
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore
from zarr.storage._fsspec import ALLOWED_EXCEPTIONS

from .stores import REMOTE_RETRY_TIMEOUT

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


class HfReadStore(FsspecStore):
    """Read one published store with unique listings and metadata cached in RAM.

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
        return store

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
        exists = super().exists
        return await self._request(
            f"Looking up {key!r} in {self._location}", lambda: exists(key)
        )

    async def getsize(self, key: str) -> int:
        getsize = super().getsize
        return await self._request(
            f"Reading the size of {key!r} in {self._location}", lambda: getsize(key)
        )

    def close(self) -> None:
        self._metadata.clear()
        super().close()

    async def list(self) -> AsyncIterator[str]:
        listing = super().list
        for name in await self._request(
            f"Listing {self._location}", lambda: _names(listing())
        ):
            yield name

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        listing = super().list_prefix
        for name in await self._request(
            f"Listing the keys under {prefix!r} in {self._location}",
            lambda: _names(listing(prefix)),
        ):
            yield name

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
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
