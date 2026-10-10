import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import parse_ratelimit_headers
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore


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


class HfReadStore(FsspecStore):
    """Read one published store with unique listings and metadata cached in RAM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metadata: dict[str, asyncio.Task[bytes | None]] = {}

    async def _read(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        for attempt in range(4):
            try:
                return await super().get(key, prototype, byte_range)
            except HfHubHTTPError as error:
                if (
                    error.response is None
                    or error.response.status_code != 429
                    or attempt == 3
                ):
                    raise
                delay = retry_delay(error, attempt)
                if delay > 300:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("Unreachable read retry state")

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

    def close(self) -> None:
        self._metadata.clear()
        super().close()

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Concurrent HF listings can append the same paths to its directory cache.
        seen: set[str] = set()
        async for name in super().list_dir(prefix):
            if name not in seen:
                seen.add(name)
                yield name
