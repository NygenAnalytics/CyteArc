"""Offline tests for the cached, rate-limit-aware Hugging Face store reader."""

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import httpx2
import numpy as np
import pytest
import zarr
from huggingface_hub.errors import HfHubHTTPError
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import FsspecStore

from cytearc.storage import huggingface

pytestmark = pytest.mark.usefixtures("hub_offline")

NOW = 1_700_000_000.0


def _status_error(
    status: int, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    response = httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("GET", "https://bucket.invalid/object"),
    )
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=response.request, response=response
    )


def test_read_store_removes_repeated_listing_entries(tmp_path, monkeypatch):
    async def repeated(self, prefix):
        for name in ["a", "b", "a", "c", "b"]:
            yield name

    monkeypatch.setattr(FsspecStore, "list_dir", repeated)
    store = huggingface.HfReadStore.from_url(
        str(tmp_path), read_only=True, storage_options={"token": False}
    )

    async def names() -> list[str]:
        return [name async for name in store.list_dir("")]

    assert asyncio.run(names()) == ["a", "b", "c"]


@pytest.mark.parametrize("value", [b'{"node_type": "group"}', None])
@pytest.mark.parametrize("key", ["zarr.json", "RNA/.zattrs"])
def test_read_store_caches_metadata_per_open(tmp_path, monkeypatch, key, value):
    calls = []
    published = value
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append(key)
        return None if published is None else prototype.buffer.from_bytes(published)

    monkeypatch.setattr(FsspecStore, "get", read)
    first = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    second = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        nonlocal published
        initial = await first.get(key, prototype)
        published = b'{"node_type": "group", "attributes": {"new": true}}'
        cached = await first.get(key, prototype)
        assert (None if initial is None else initial.to_bytes()) == value
        assert (None if cached is None else cached.to_bytes()) == value
        assert (await second.get(key, prototype)).to_bytes() == published
        first.close()
        assert (await first.get(key, prototype)).to_bytes() == published

    try:
        asyncio.run(check())
        assert calls == [key, key, key]
    finally:
        first.close()
        second.close()


def test_read_store_shares_fetches_without_serializing_other_keys(
    tmp_path, monkeypatch
):
    calls = []
    prototype = default_buffer_prototype()
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        started, release = asyncio.Event(), asyncio.Event()

        async def read(self, key, prototype, byte_range=None):
            calls.append(key)
            if len(calls) == 2:
                started.set()
            await release.wait()
            return prototype.buffer.from_bytes(b"metadata")

        monkeypatch.setattr(FsspecStore, "get", read)
        cancelled = asyncio.create_task(store.get("zarr.json", prototype))
        shared = asyncio.create_task(store.get("zarr.json", prototype))
        other = asyncio.create_task(store.get("RNA/zarr.json", prototype))
        await asyncio.wait_for(started.wait(), timeout=5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        results = await asyncio.gather(shared, other)
        assert [result.to_bytes() for result in results] == [b"metadata"] * 2
        assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert sorted(calls) == ["RNA/zarr.json", "zarr.json"]
    finally:
        store.close()


def test_read_store_retries_failed_metadata_fetches(tmp_path, monkeypatch):
    calls = []
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append(key)
        if len(calls) == 1:
            raise OSError("Temporary transfer failure")
        return prototype.buffer.from_bytes(b"metadata")

    monkeypatch.setattr(FsspecStore, "get", read)
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        with pytest.raises(OSError, match="Temporary transfer failure"):
            await store.get("zarr.json", prototype)
        for _ in range(2):
            assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert calls == ["zarr.json", "zarr.json"]
    finally:
        store.close()


def test_read_store_handles_failure_after_its_reader_is_cancelled(
    tmp_path, monkeypatch
):
    prototype = default_buffer_prototype()
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    errors = []

    async def check():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def read(self, key, prototype, byte_range=None):
            if not started.is_set():
                started.set()
                await release.wait()
                loop.call_soon(finished.set)
                raise OSError("Abandoned transfer failed")
            return prototype.buffer.from_bytes(b"metadata")

        monkeypatch.setattr(FsspecStore, "get", read)
        reader = asyncio.create_task(store.get("zarr.json", prototype))
        await asyncio.wait_for(started.wait(), timeout=5)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert errors == []
    finally:
        store.close()


@pytest.mark.parametrize(
    ("key", "byte_range", "read_only"),
    [
        ("RNA/counts/c/0/0", None, True),
        ("RNA/zarr.json", RangeByteRequest(0, 4), True),
        ("zarr.json", None, False),
    ],
)
def test_read_store_does_not_cache_chunks_ranges_or_writable_metadata(
    tmp_path, monkeypatch, key, byte_range, read_only
):
    calls = []
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append((key, byte_range))
        return prototype.buffer.from_bytes(str(len(calls)).encode())

    monkeypatch.setattr(FsspecStore, "get", read)
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=read_only)

    async def check():
        first = await store.get(key, prototype, byte_range)
        second = await store.get(key, prototype, byte_range)
        assert first.to_bytes() == b"1"
        assert second.to_bytes() == b"2"

    try:
        asyncio.run(check())
        assert calls == [(key, byte_range), (key, byte_range)]
    finally:
        store.close()


@pytest.mark.parametrize("failure_at", ["metadata", "shard_index", "shard_values"])
def test_read_store_retries_throttled_array_reads(tmp_path, monkeypatch, failure_at):
    values = np.arange(128, dtype=np.int32).reshape(16, 8)
    zarr.create_array(store=tmp_path, data=values, chunks=(4, 4), shards=(8, 8))
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    read = store.fs._cat_file
    failed = False
    sleeps = []

    async def throttled(path, *args, **kwargs):
        nonlocal failed
        start = kwargs.get("start")
        category = (
            "metadata"
            if path.endswith("zarr.json")
            else "shard_index"
            if start is not None and start < 0
            else "shard_values"
        )
        if category == failure_at and not failed:
            failed = True
            response = httpx2.Response(
                429,
                headers={"Retry-After": "7"},
                request=httpx2.Request("GET", "https://bucket.invalid/object"),
            )
            raise HfHubHTTPError("Rate limited", response=response)
        return await read(path, *args, **kwargs)

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(store.fs, "_cat_file", throttled)
    monkeypatch.setattr(huggingface.asyncio, "sleep", sleep)
    try:
        array = zarr.open_array(store=store, mode="r", zarr_format=3)
        np.testing.assert_array_equal(array[1:3, 2:4], values[1:3, 2:4])
        assert failed
        assert sleeps == [7.0]
    finally:
        store.close()


@pytest.mark.parametrize(
    ("status", "retry_after", "attempts", "delays"),
    [
        (429, None, 4, [2.0, 4.0, 8.0]),
        (429, "301", 1, []),
        (403, "7", 1, []),
        (500, "7", 1, []),
    ],
)
def test_read_store_bounds_retries_and_preserves_failures(
    tmp_path, monkeypatch, status, retry_after, attempts, delays
):
    (tmp_path / "chunk").write_bytes(b"source counts")
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    read = store.fs._cat_file
    response = httpx2.Response(
        status,
        headers={} if retry_after is None else {"Retry-After": retry_after},
        request=httpx2.Request("GET", "https://bucket.invalid/object"),
    )
    failure = HfHubHTTPError("Read failed", response=response)
    calls = 0
    sleeps = []

    async def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise failure

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(store.fs, "_cat_file", failing)
    monkeypatch.setattr(huggingface.asyncio, "sleep", sleep)
    prototype = default_buffer_prototype()
    try:
        with pytest.raises(HfHubHTTPError) as raised:
            asyncio.run(store.get("chunk", prototype))
        assert raised.value is failure
        assert calls == attempts
        assert sleeps == delays
        monkeypatch.setattr(store.fs, "_cat_file", read)
        assert asyncio.run(store.get("chunk", prototype)).to_bytes() == b"source counts"
    finally:
        store.close()


def test_read_store_cancels_rate_limit_backoff(tmp_path, monkeypatch):
    (tmp_path / "chunk").write_bytes(b"source counts")
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    read = store.fs._cat_file
    calls = 0

    async def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        response = httpx2.Response(
            429,
            headers={"Retry-After": "30"},
            request=httpx2.Request("GET", "https://bucket.invalid/object"),
        )
        raise HfHubHTTPError("Rate limited", response=response)

    async def check():
        waiting = asyncio.Event()

        async def sleep(delay):
            assert delay == 30.0
            waiting.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(huggingface.asyncio, "sleep", sleep)
        task = asyncio.create_task(store.get("chunk", default_buffer_prototype()))
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(store.fs, "_cat_file", failing)
    try:
        asyncio.run(check())
        assert calls == 1
        monkeypatch.setattr(store.fs, "_cat_file", read)
        assert (
            asyncio.run(store.get("chunk", default_buffer_prototype())).to_bytes()
            == b"source counts"
        )
    finally:
        store.close()


def test_retry_delay_without_a_response_uses_exponential_backoff():
    assert huggingface.retry_delay(httpx.ConnectError("reset"), 0) == 2.0
    assert huggingface.retry_delay(httpx.ConnectError("reset"), 2) == 8.0


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "10"}, 10.0),
        ({"Retry-After": "1"}, 2.0),
        ({"Retry-After": "soon"}, 2.0),
        ({"RateLimit-Reset": str(int(NOW) + 100)}, 100.0),
        ({"RateLimit-Reset": "30"}, 30.0),
        ({"X-RateLimit-Reset": "25"}, 25.0),
        ({"RateLimit-Reset": "later"}, 2.0),
        ({"RateLimit": '"api";r=0;t=55'}, 55.0),
        ({"RateLimit": '"api";r=3;t=55'}, 2.0),
    ],
)
def test_retry_delay_honours_server_headers(monkeypatch, headers, expected):
    monkeypatch.setattr(huggingface, "time", SimpleNamespace(time=lambda: NOW))
    assert huggingface.retry_delay(_status_error(503, headers), 0) == expected


@pytest.mark.parametrize("aware", [True, False])
def test_retry_delay_accepts_http_dates(aware):
    moment = datetime.now(UTC) + timedelta(seconds=120)
    header = format_datetime(
        moment if aware else moment.replace(tzinfo=None), usegmt=aware
    )
    delay = huggingface.retry_delay(_status_error(503, {"Retry-After": header}), 0)
    assert 100 < delay <= 120
