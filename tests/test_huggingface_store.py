"""Offline tests for the cached, rate-limit-aware Hugging Face store reader."""

import asyncio
import re
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import httpx2
import numpy as np
import pytest
import zarr
from huggingface_hub.errors import HfHubHTTPError
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import GroupMetadata
from zarr.core.sync import sync
from zarr.storage import FsspecStore

from cytearc.storage import huggingface
from cytearc.storage.preparation import (
    PREPARATION_CLAIM,
    _claim_document,
    preparation_claim,
)
from tests.fixtures_hub import BUCKET_ID

pytestmark = pytest.mark.usefixtures("hub_offline")

NOW = 1_700_000_000.0
LOOPBACK_URI = f"hf://buckets/{BUCKET_ID}/data.zarr"
# A store call that its own deadline does not end fails the test after this
# long instead of hanging it.
GUARD_SECONDS = 10


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


def test_read_store_shares_a_metadata_fetch_across_event_loops(tmp_path, monkeypatch):
    calls = []
    started, release = threading.Event(), threading.Event()
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append(key)
        started.set()
        # The fetch waits on a thread, as a network read does.
        assert await asyncio.to_thread(release.wait, GUARD_SECONDS)
        return prototype.buffer.from_bytes(b"metadata")

    monkeypatch.setattr(FsspecStore, "get", read)
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    results = {}

    def reader(name):
        results[name] = asyncio.run(store.get("zarr.json", prototype)).to_bytes()

    first = threading.Thread(target=reader, args=("first",), daemon=True)
    second = threading.Thread(target=reader, args=("second",), daemon=True)
    try:
        first.start()
        assert started.wait(GUARD_SECONDS)
        # The second reader, on its own event loop, waits for the same fetch.
        second.start()
        time.sleep(0.2)
        assert results == {}
        release.set()
        first.join(GUARD_SECONDS)
        second.join(GUARD_SECONDS)
    finally:
        release.set()
        store.close()
    assert results == {"first": b"metadata", "second": b"metadata"}
    assert calls == ["zarr.json"]


def test_read_store_fetches_again_when_a_shared_fetch_loses_its_event_loop(
    tmp_path, monkeypatch
):
    calls = []
    started, finish, release = threading.Event(), threading.Event(), threading.Event()
    prototype = default_buffer_prototype()

    async def read(self, key, prototype, byte_range=None):
        calls.append(key)
        if len(calls) == 1:
            started.set()
            await asyncio.to_thread(release.wait, GUARD_SECONDS)
        return prototype.buffer.from_bytes(b"metadata")

    monkeypatch.setattr(FsspecStore, "get", read)
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    results = {}

    async def leave():
        reader = asyncio.create_task(store.get("zarr.json", prototype))
        assert await asyncio.to_thread(finish.wait, GUARD_SECONDS)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        # Ending this event loop cancels the fetch that the reader started.

    def waiter():
        results["waiter"] = asyncio.run(store.get("zarr.json", prototype)).to_bytes()

    leaving = threading.Thread(target=lambda: asyncio.run(leave()), daemon=True)
    waiting = threading.Thread(target=waiter, daemon=True)
    try:
        leaving.start()
        assert started.wait(GUARD_SECONDS)
        waiting.start()
        time.sleep(0.2)
        finish.set()
        waiting.join(GUARD_SECONDS)
        assert results == {"waiter": b"metadata"}
    finally:
        release.set()
        finish.set()
        leaving.join(GUARD_SECONDS)
        store.close()
    # The waiter fetched the key again on its own event loop.
    assert calls == ["zarr.json", "zarr.json"]


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
        (404, None, 1, []),
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


@pytest.mark.parametrize("status", [408, 500, 503])
def test_read_store_retries_server_errors_until_its_deadline(
    tmp_path, monkeypatch, status
):
    (tmp_path / "chunk").write_bytes(b"source counts")
    store = huggingface.HfReadStore.from_url(
        str(tmp_path), read_only=True, storage_options={"retry_timeout": 20}
    )
    read = store.fs._cat_file
    response = httpx2.Response(
        status,
        headers={"Retry-After": "7"},
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
        with pytest.raises(TimeoutError) as raised:
            asyncio.run(store.get("chunk", prototype))
        # Each wait is the longer of Retry-After and 2, 4, 8, ... seconds, and
        # the fifth would end past the 20 s deadline.
        assert sleeps == [7.0, 7.0, 8.0, 16.0]
        assert calls == 5
        assert raised.value.__cause__ is failure
        assert str(raised.value).startswith(
            f"Reading 'chunk' from file://{tmp_path} did not succeed within 20 s: "
            f"5 attempts failed, the last with HTTP {status}."
        )
        monkeypatch.setattr(store.fs, "_cat_file", read)
        assert asyncio.run(store.get("chunk", prototype)).to_bytes() == b"source counts"
    finally:
        store.close()


def test_read_store_reads_partial_values_through_its_reads(tmp_path):
    (tmp_path / "chunk").write_bytes(b"source counts")
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)
    requests = [("chunk", RangeByteRequest(0, 6)), ("missing", None), ("chunk", None)]
    try:
        values = asyncio.run(
            store.get_partial_values(default_buffer_prototype(), requests)
        )
    finally:
        store.close()
    assert [None if value is None else value.to_bytes() for value in values] == [
        b"source",
        None,
        b"source counts",
    ]


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("request_timeout", 0, ValueError),
        ("request_timeout", -1.5, ValueError),
        ("retry_timeout", float("inf"), ValueError),
        ("retry_timeout", float("nan"), ValueError),
        ("request_timeout", True, TypeError),
        ("retry_timeout", "180", TypeError),
        ("retry_timeout", None, TypeError),
    ],
)
def test_read_store_refuses_timeouts_that_are_not_positive_seconds(
    tmp_path, name, value, error
):
    with pytest.raises(error, match=f"^{name} must be"):
        huggingface.HfReadStore.from_url(
            str(tmp_path), read_only=True, storage_options={name: value}
        )
    fs = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True).fs
    with pytest.raises(error, match=f"^{name} must be"):
        huggingface.HfReadStore(fs, read_only=True, **{name: value})


def test_read_store_takes_its_timeouts_from_storage_options():
    options = {"token": False, "endpoint": "http://127.0.0.1:9"}
    defaults = huggingface.HfReadStore.from_url(LOOPBACK_URI, storage_options=options)
    store = huggingface.HfReadStore.from_url(
        LOOPBACK_URI,
        storage_options={**options, "request_timeout": 5, "retry_timeout": 30.5},
    )
    read_only = store.with_read_only(True)

    assert (defaults.request_timeout, defaults.retry_timeout) == (60.0, 180.0)
    assert (store.request_timeout, store.retry_timeout) == (5.0, 30.5)
    # Hugging Face's file system receives only its own options.
    assert store.fs.sync_fs.storage_options["endpoint"] == options["endpoint"]
    assert "request_timeout" not in store.fs.sync_fs.storage_options
    assert "retry_timeout" not in store.fs.sync_fs.storage_options
    assert read_only.read_only and not store.read_only
    assert (read_only.request_timeout, read_only.retry_timeout) == (5.0, 30.5)


async def _names(listing):
    return [name async for name in listing]


def _loopback_store(hub, *, read_only=True, uri=LOOPBACK_URI, **timeouts):
    """Publish an array in the hub's bucket and open it as an ``hf://`` store."""
    if not (hub.root / "data.zarr").exists():
        zarr.create_array(
            store=hub.root / "data.zarr",
            data=np.arange(64, dtype=np.int32).reshape(8, 8),
            chunks=(4, 4),
        )
    return huggingface.HfReadStore.from_url(
        uri,
        storage_options={"token": False, "endpoint": hub.endpoint, **timeouts},
        read_only=read_only,
    )


def _received(hub, kind):
    """Return the path and detail of each request of ``kind`` that the hub met."""
    return [(path, detail) for got, path, detail in hub.received if got == kind]


def _record_sleeps(monkeypatch):
    sleeps = []
    pause = asyncio.sleep

    async def sleep(delay):
        sleeps.append(delay)
        await pause(0)

    monkeypatch.setattr(huggingface.asyncio, "sleep", sleep)
    return sleeps


_PROTOTYPE = default_buffer_prototype()
_STORE_CALLS = {
    "get": lambda store: store.get("zarr.json", _PROTOTYPE),
    "get_partial_values": lambda store: store.get_partial_values(
        _PROTOTYPE, [("zarr.json", None)]
    ),
    "exists": lambda store: store.exists("zarr.json"),
    "getsize": lambda store: store.getsize("zarr.json"),
    "list": lambda store: _names(store.list()),
    "list_prefix": lambda store: _names(store.list_prefix("")),
    "list_dir": lambda store: _names(store.list_dir("")),
}
# What each call of a writable store requests first, which names the request
# in the error that ends its retries. Every call of a read-only store lists
# the store first.
_REQUESTS = {
    "get": f"Reading 'zarr.json' from {LOOPBACK_URI}",
    "get_partial_values": f"Reading 'zarr.json' from {LOOPBACK_URI}",
    "exists": f"Looking up 'zarr.json' in {LOOPBACK_URI}",
    "getsize": f"Reading the size of 'zarr.json' in {LOOPBACK_URI}",
    "list": f"Listing {LOOPBACK_URI}",
    "list_prefix": f"Listing the keys under '' in {LOOPBACK_URI}",
    "list_dir": f"Listing '' in {LOOPBACK_URI}",
}


def test_read_store_answers_lookups_and_listings_from_the_bucket(loopback_hub):
    store = _loopback_store(loopback_hub)
    published = loopback_hub.root / "data.zarr"
    keys = sorted(
        path.relative_to(published).as_posix()
        for path in published.rglob("*")
        if path.is_file()
    )
    try:
        listed = sync(_names(store.list()), timeout=GUARD_SECONDS)
        under_c = sync(_names(store.list_prefix("c")), timeout=GUARD_SECONDS)
        children = sync(_names(store.list_dir("")), timeout=GUARD_SECONDS)
        found = sync(store.exists("c/1/1"), timeout=GUARD_SECONDS)
        missing = sync(store.exists("c/9/9"), timeout=GUARD_SECONDS)
        size = sync(store.getsize("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert sorted(listed) == keys
    assert sorted(under_c) == [key for key in keys if key.startswith("c/")]
    assert sorted(children) == ["c", "zarr.json"]
    assert found and not missing
    assert size == (published / "zarr.json").stat().st_size


@pytest.mark.parametrize("read_only", [True, False])
@pytest.mark.parametrize(
    ("stalled", "call"),
    [
        ("bucket", "get"),
        ("bucket", "list_dir"),
        *(("tree", call) for call in _STORE_CALLS),
    ],
)
def test_read_store_raises_when_a_request_stalls(
    loopback_hub, stalled, call, read_only
):
    # Hugging Face's lookup of a bucket and its tree listings wait on the
    # server without a timeout of their own.
    store = _loopback_store(
        loopback_hub, read_only=read_only, request_timeout=0.25, retry_timeout=1
    )
    loopback_hub.plan[stalled] = ["stall"]
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(_STORE_CALLS[call](store), timeout=GUARD_SECONDS)
    finally:
        store.close()
    # The next attempt would wait 2 s first, past the 1 s that the request has.
    assert time.monotonic() - started < 1.5
    request = f"Listing {LOOPBACK_URI}" if read_only else _REQUESTS[call]
    assert str(raised.value).startswith(
        f"{request} did not succeed within 1 s: 1 attempt failed, the last "
        "with no answer within 0.25 s. To wait longer, pass more seconds in "
        "storage_options as request_timeout"
    )
    assert [kind for kind, _ in loopback_hub.requests].count(stalled) == 1


@pytest.mark.parametrize("call", ["get", "get_partial_values"])
def test_read_store_raises_when_a_read_stalls(loopback_hub, call):
    store = _loopback_store(loopback_hub, request_timeout=0.25, retry_timeout=1)
    loopback_hub.plan["file"] = ["stall"]
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(_STORE_CALLS[call](store), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert str(raised.value).startswith(
        f"Reading 'zarr.json' from {LOOPBACK_URI} did not succeed within 1 s: 1 "
        "attempt failed, the last with no answer within 0.25 s."
    )
    assert [kind for kind, _ in loopback_hub.requests].count("file") == 1


@pytest.mark.parametrize("stalled", ["bucket", "tree", "file"])
def test_read_store_reads_after_abandoning_a_stalled_attempt(
    loopback_hub, monkeypatch, stalled
):
    # The attempt after the stalled one must answer within the request
    # timeout, so the timeout leaves room for a loaded test machine.
    store = _loopback_store(loopback_hub, request_timeout=1.0)
    loopback_hub.plan[stalled] = ["stall"]
    sleeps = _record_sleeps(monkeypatch)
    try:
        chunk = sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert chunk.to_bytes() == (loopback_hub.root / "data.zarr/c/1/0").read_bytes()
    assert sleeps == [2.0]
    assert [kind for kind, _ in loopback_hub.requests].count(stalled) == 2


def test_read_store_cuts_its_last_attempt_to_the_time_left(loopback_hub):
    store = _loopback_store(loopback_hub, request_timeout=1.0, retry_timeout=3.6)
    loopback_hub.plan["tree"] = ["stall", "stall"]
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(_names(store.list_dir("")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    elapsed = time.monotonic() - started
    # One attempt of 1 s, a wait of 2 s, and an attempt cut to the 0.6 s left.
    # A loaded machine handles the first timeout and the wait late, which
    # shortens the last attempt: by up to 0.5 s, it still runs and is cut.
    assert 3.5 < elapsed < 4.6
    assert re.match(
        rf"Listing {re.escape(LOOPBACK_URI)} did not succeed within 3\.6 s: "
        r"2 attempts failed, the last with no answer within 0\.[1-6]\d* s\.",
        str(raised.value),
    ), str(raised.value)
    assert [kind for kind, _ in loopback_hub.requests].count("tree") == 2


@pytest.mark.parametrize("failing", ["bucket", "tree"])
def test_read_store_retries_a_server_error(loopback_hub, monkeypatch, failing):
    store = _loopback_store(loopback_hub)
    loopback_hub.plan[failing] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        names = sync(_names(store.list_dir("")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert sorted(names) == sorted(
        path.name for path in (loopback_hub.root / "data.zarr").iterdir()
    )
    assert sleeps == [2.0]
    assert [kind for kind, _ in loopback_hub.requests].count(failing) == 2


def test_read_store_waits_out_a_rate_limited_read_as_before(loopback_hub, monkeypatch):
    # A read retries a 429 after the delay its headers ask for, as it always has.
    store = _loopback_store(loopback_hub)
    loopback_hub.plan["bucket"] = [(429, {"Retry-After": "7"})]
    sleeps = _record_sleeps(monkeypatch)
    try:
        chunk = sync(store.get("c/0/1", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert chunk.to_bytes() == (loopback_hub.root / "data.zarr/c/0/1").read_bytes()
    assert sleeps == [7.0]


def test_read_store_waits_out_a_rate_limit_beyond_its_retry_timeout(loopback_hub):
    # The 2 s wait that a 429 asks for does not use up the 1 s retry timeout,
    # so the attempt after it still has the time it needs.
    store = _loopback_store(loopback_hub, retry_timeout=1)
    loopback_hub.plan["tree"] = [429]
    started = time.monotonic()
    try:
        names = sync(_names(store.list_dir("")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert time.monotonic() - started >= 2
    assert sorted(names) == ["c", "zarr.json"]
    assert [kind for kind, _ in loopback_hub.requests].count("tree") == 2


def test_read_store_retries_a_refused_connection(monkeypatch):
    # Nothing listens on a port that was just released.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    store = huggingface.HfReadStore.from_url(
        LOOPBACK_URI,
        storage_options={
            "token": False,
            "endpoint": f"http://127.0.0.1:{port}",
            "retry_timeout": 5,
        },
        read_only=True,
    )
    sleeps = _record_sleeps(monkeypatch)
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(store.get("zarr.json", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert sleeps == [2.0, 4.0]
    assert isinstance(raised.value.__cause__, httpx2.ConnectError)
    # A read-only store lists itself before it reads, so the listing fails.
    assert str(raised.value).startswith(
        f"Listing {LOOPBACK_URI} did not succeed within 5 s: "
        "3 attempts failed, the last with ConnectError: "
    )


def test_read_store_lists_once_and_reads_exactly_the_bytes_asked(loopback_hub):
    store = _loopback_store(loopback_hub)
    published = loopback_hub.root / "data.zarr"
    first = (published / "c/1/0").read_bytes()
    second = (published / "c/0/1").read_bytes()
    reads = [
        ("c/1/0", None, first, f"bytes=0-{len(first) - 1}"),
        ("c/1/0", RangeByteRequest(2, 9), first[2:9], "bytes=2-8"),
        ("c/1/0", OffsetByteRequest(5), first[5:], f"bytes=5-{len(first) - 1}"),
        (
            "c/1/0",
            SuffixByteRequest(4),
            first[-4:],
            f"bytes={len(first) - 4}-{len(first) - 1}",
        ),
        # A range past the end stops at the end, as other stores read it.
        (
            "c/0/1",
            RangeByteRequest(len(second) - 3, len(second) + 50),
            second[-3:],
            f"bytes={len(second) - 3}-{len(second) - 1}",
        ),
        # An empty range and a key that the store does not hold need no request.
        ("c/0/1", RangeByteRequest(3, 3), b"", None),
        ("c/9/9", None, None, None),
    ]

    async def read_all():
        return await asyncio.gather(
            *(store.get(key, _PROTOTYPE, byte_range) for key, byte_range, _, _ in reads)
        )

    try:
        values = sync(read_all(), timeout=GUARD_SECONDS)
        exists = sync(store.exists("c/1/0"), timeout=GUARD_SECONDS)
        size = sync(store.getsize("c/0/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert [None if value is None else value.to_bytes() for value in values] == [
        expected for _, _, expected, _ in reads
    ]
    assert exists and size == len(second)
    assert _received(loopback_hub, "tree") == [("data.zarr", "recursive")]
    assert sorted(_received(loopback_hub, "file")) == sorted(
        (f"data.zarr/{key}", header)
        for key, _, _, header in reads
        if header is not None
    )


def test_read_store_reads_a_zarr_array_without_listing_its_directories(loopback_hub):
    store = _loopback_store(loopback_hub)
    try:
        array = zarr.open_array(store=store, mode="r", zarr_format=3)
        values = array[2:7, 1:6]
    finally:
        store.close()
    np.testing.assert_array_equal(
        values, np.arange(64, dtype=np.int32).reshape(8, 8)[2:7, 1:6]
    )
    # One listing of the store answers every lookup the array makes.
    assert _received(loopback_hub, "tree") == [("data.zarr", "recursive")]
    assert {path for path, _ in _received(loopback_hub, "file")} == {
        "data.zarr/zarr.json",
        *(f"data.zarr/c/{row}/{column}" for row in (0, 1) for column in (0, 1)),
    }


def test_read_store_raises_when_a_listed_key_is_gone(loopback_hub):
    store = _loopback_store(loopback_hub)
    chunk = loopback_hub.root / "data.zarr/c/1/1"
    try:
        assert sync(store.exists("c/1/1"), timeout=GUARD_SECONDS)
        # The store changes after it was listed.
        chunk.rename(chunk.with_name("moved"))
        with pytest.raises(OSError) as raised:
            sync(store.get("c/1/1", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    # Not FileNotFoundError, which Zarr reads as a missing chunk, and not
    # retried.
    assert type(raised.value) is OSError
    assert str(raised.value).startswith(
        f"'c/1/1' is listed in {LOOPBACK_URI}, but reading it answered 404: the "
        "store changed after it was listed."
    )
    assert [kind for kind, _ in loopback_hub.requests].count("file") == 1


def _leaves(error):
    """Return the errors that an exception group holds, or the error itself."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _leaves(inner)]
    return [error]


def test_read_store_raises_when_a_listed_shard_is_gone_after_its_index(loopback_hub):
    uri = f"hf://buckets/{BUCKET_ID}/sharded.zarr"
    zarr.create_array(
        store=loopback_hub.root / "sharded.zarr",
        data=np.arange(1, 65, dtype=np.int32).reshape(8, 8),
        chunks=(2, 2),
        shards=(4, 4),
    )
    store = _loopback_store(loopback_hub, uri=uri)
    try:
        array = zarr.open_array(store=store, mode="r", zarr_format=3)
        # A partial read of a shard reads its index, then its chunks' bytes.
        # Zarr reads a FileNotFoundError from the second as a missing shard and
        # fills its chunks with zeros, so the store must raise something else.
        loopback_hub.plan["file"] = ["serve", 404]
        with pytest.raises((OSError, BaseExceptionGroup)) as raised:
            array[0:2, 0:2]
    finally:
        store.close()
    errors = [leaf for leaf in _leaves(raised.value) if type(leaf) is OSError]
    assert len(errors) == 1
    assert str(errors[0]).startswith(
        f"'c/0/0' is listed in {uri}, but reading it answered 404: the store "
        "changed after it was listed."
    )
    index, values = _received(loopback_hub, "file")[-2:]
    assert index[0] == values[0] == "sharded.zarr/c/0/0"
    assert index[1] != values[1]


def test_read_store_reads_a_missing_store_as_empty(loopback_hub):
    store = _loopback_store(loopback_hub, uri=f"hf://buckets/{BUCKET_ID}/absent.zarr")
    try:
        exists = sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        keys = sync(_names(store.list_prefix("")), timeout=GUARD_SECONDS)
        value = sync(store.get("zarr.json", _PROTOTYPE), timeout=GUARD_SECONDS)
        empty = sync(store.is_empty(""), timeout=GUARD_SECONDS)
        with pytest.raises(FileNotFoundError):
            sync(store.getsize("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert not exists and keys == [] and value is None and empty
    assert _received(loopback_hub, "tree") == [("absent.zarr", "recursive")]
    assert _received(loopback_hub, "file") == []


def test_read_store_shares_its_listing_across_callers_and_event_loops(
    loopback_hub, monkeypatch
):
    store = _loopback_store(loopback_hub)
    started, cancelled, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    calls = []
    list_store = huggingface._list_store

    def held(fs, path):
        calls.append(path)
        started.set()
        assert release.wait(GUARD_SECONDS)
        return list_store(fs, path)

    monkeypatch.setattr(huggingface, "_list_store", held)
    results: dict[str, object] = {}

    async def first_caller_cancelled():
        reader = asyncio.create_task(store.get("c/0/0", _PROTOTYPE))
        while not started.is_set():
            await asyncio.sleep(0.01)
        waiter = asyncio.create_task(store.exists("c/0/0"))
        await asyncio.sleep(0)
        # Cancelling the caller that started the listing leaves it to the others.
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        cancelled.set()
        return await waiter

    def other_loop():
        results["other"] = asyncio.run(store.getsize("zarr.json"))

    try:
        thread = threading.Thread(target=other_loop, daemon=True)
        with_cancel = threading.Thread(
            target=lambda: results.update(first=asyncio.run(first_caller_cancelled())),
            daemon=True,
        )
        with_cancel.start()
        assert cancelled.wait(GUARD_SECONDS)
        # A caller on another event loop waits for the same listing.
        thread.start()
        release.set()
        with_cancel.join(GUARD_SECONDS)
        thread.join(GUARD_SECONDS)
        clone = store.with_read_only(True)
        clone_keys = sync(_names(clone.list()), timeout=GUARD_SECONDS)
    finally:
        store.close()
    published = loopback_hub.root / "data.zarr"
    assert results == {
        "first": True,
        "other": (published / "zarr.json").stat().st_size,
    }
    assert calls == [store.path]
    assert len(clone_keys) == 5
    assert _received(loopback_hub, "tree") == [("data.zarr", "recursive")]


def test_read_store_lists_again_after_a_failed_listing_and_after_closing(
    loopback_hub,
):
    store = _loopback_store(loopback_hub)
    loopback_hub.plan["tree"] = [403]
    try:
        with pytest.raises(HfHubHTTPError):
            sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        found = sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        store.close()
        reopened = sync(store.exists("c/0/0"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert found and reopened
    assert _received(loopback_hub, "tree") == [("data.zarr", "recursive")] * 3


def test_writable_read_store_reads_through_the_file_system(loopback_hub):
    store = _loopback_store(loopback_hub, read_only=False)
    try:
        chunk = sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
        found = sync(store.exists("c/0/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert chunk.to_bytes() == (loopback_hub.root / "data.zarr/c/1/0").read_bytes()
    assert found
    # Hugging Face's file system lists the directory of each key it reads.
    listings = _received(loopback_hub, "tree")
    assert ("data.zarr/c/1", "directory") in listings
    assert all(detail == "directory" for _, detail in listings)


def _accept_uploads(hub, store, monkeypatch):
    """Write each file that ``store`` uploads where ``hub`` serves it, and list them.

    The loopback hub answers reads only, so the bucket's upload call writes
    into the directory that the hub serves instead.
    """
    uploads = []

    def upload(bucket_id, *, add):
        assert bucket_id == BUCKET_ID
        for source, remote in add:
            uploads.append(remote)
            target = hub.root / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())

    monkeypatch.setattr(store.fs.sync_fs._api, "batch_bucket_files", upload)
    return uploads


def _publish_group(hub):
    """Publish a group with one attribute as the hub's ``data.zarr``."""
    zarr.create_group(store=hub.root / "data.zarr", attributes={"title": "kept"})


def _published_attributes(hub):
    return zarr.open_group(hub.root / "data.zarr", mode="r").attrs.asdict()


# The document that Zarr writes for each parent group of a node it creates,
# unless the parent's key exists.
_EMPTY_GROUP = GroupMetadata(zarr_format=3).to_buffer_dict(_PROTOTYPE)["zarr.json"]


@pytest.mark.parametrize("failing", ["bucket", "tree"])
def test_writable_store_retries_a_lookup_that_meets_a_server_error(
    loopback_hub, monkeypatch, failing
):
    # Hugging Face's exists reads a failed request as a missing key.
    store = _loopback_store(loopback_hub, read_only=False)
    loopback_hub.plan[failing] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        found = sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert found
    assert sleeps == [2.0]
    assert [kind for kind, _ in loopback_hub.requests].count(failing) == 2


@pytest.mark.parametrize("key", ["absent.json", "c/9/9"])
def test_writable_store_reads_a_key_as_missing_once_the_server_reports_it(
    loopback_hub, monkeypatch, key
):
    # A key absent from its directory, and a key whose directory is absent.
    store = _loopback_store(loopback_hub, read_only=False)
    loopback_hub.plan["tree"] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        found = sync(store.exists(key), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert not found
    # The answer comes from the listing after the failed one.
    assert sleeps == [2.0]
    assert loopback_hub.plan["tree"] == []


@pytest.mark.parametrize("failing", ["bucket", "tree"])
@pytest.mark.parametrize("status", [401, 403])
def test_writable_store_raises_a_lookup_that_the_server_refuses(
    loopback_hub, monkeypatch, status, failing
):
    store = _loopback_store(loopback_hub, read_only=False)
    loopback_hub.plan[failing] = [status]
    sleeps = _record_sleeps(monkeypatch)
    try:
        with pytest.raises(HfHubHTTPError) as raised:
            sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        # The refusal is not kept as an answer: the next lookup finds the key.
        found = sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert raised.value.response.status_code == status
    assert sleeps == []
    assert found


def test_writable_store_raises_when_the_retries_of_a_lookup_run_out(
    loopback_hub, monkeypatch
):
    store = _loopback_store(loopback_hub, read_only=False, retry_timeout=5)
    loopback_hub.plan["tree"] = [503, 503, 503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    # The waits are 2 and 4 s, and the next, of 8 s, would end past the 5 s.
    assert sleeps == [2.0, 4.0]
    assert isinstance(raised.value.__cause__, HfHubHTTPError)
    assert raised.value.__cause__.response.status_code == 503
    assert str(raised.value).startswith(
        f"Looking up 'zarr.json' in {LOOPBACK_URI} did not succeed within 5 s: "
        "3 attempts failed, the last with HTTP 503."
    )
    assert [kind for kind, _ in loopback_hub.requests].count("tree") == 3


def test_writable_store_does_not_write_over_a_key_whose_lookup_failed(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    loopback_hub.plan["tree"] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        sync(store.set_if_not_exists("zarr.json", _EMPTY_GROUP), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert uploads == []
    assert _published_attributes(loopback_hub) == {"title": "kept"}
    assert sleeps == [2.0]
    assert _received(loopback_hub, "tree") == [("data.zarr", "directory")] * 2


def test_writable_store_keeps_a_parent_group_when_creating_a_child_meets_a_server_error(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
    take = loopback_hub.take
    wrote = False

    def take_during_a_write(kind, path, detail=None):
        nonlocal wrote
        if kind == "file" and path == "data.zarr/zarr.json" and not wrote:
            wrote = True
            # Zarr reads the parent group to check that it is not an array,
            # and another write through the store lands meanwhile. Its upload
            # drops Hugging Face's cached listing of the parent's directory,
            # so Zarr's lookup of the parent before it writes the parent's
            # document reaches the server, and meets a 503.
            asyncio.run(store.set("sibling/zarr.json", _EMPTY_GROUP))
            loopback_hub.plan["tree"] = [503]
        return take(kind, path, detail)

    monkeypatch.setattr(loopback_hub, "take", take_during_a_write)
    try:
        root.create_group("child")
    finally:
        store.close()
    assert _published_attributes(loopback_hub) == {"title": "kept"}
    assert sorted(uploads) == [
        "data.zarr/child/zarr.json",
        "data.zarr/sibling/zarr.json",
    ]
    assert sleeps == [2.0]
    assert _received(loopback_hub, "tree")[-2:] == [("data.zarr", "directory")] * 2


def test_writable_store_keeps_the_preparation_claim_of_another_process(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    held = loopback_hub.root / "data.zarr" / PREPARATION_CLAIM / "zarr.json"
    held.parent.mkdir()
    claim = _claim_document("another process").to_bytes()
    held.write_bytes(claim)
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
    # The first listing of the claim's directory fails.
    loopback_hub.plan["tree"] = [503]
    try:
        with pytest.raises(RuntimeError, match="is being prepared by another process"):
            with preparation_claim(root):
                pass
    finally:
        store.close()
    assert uploads == []
    assert held.read_bytes() == claim
    assert sleeps == [2.0]
    listings = _received(loopback_hub, "tree")
    assert listings.count((f"data.zarr/{PREPARATION_CLAIM}", "directory")) == 2


def test_offline_requests_build_their_headers_without_the_agent_registry(
    monkeypatch, tmp_path
):
    # huggingface_hub fetches its agent registry from the Hub when the cache
    # file is missing or older than a day; under hub_offline no request may.
    from huggingface_hub import constants
    from huggingface_hub.utils import _detect_agent, build_hf_headers

    missing = tmp_path / "agent_harnesses.json"
    monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", str(missing))
    monkeypatch.setattr(_detect_agent, "_registry", None)

    headers = build_hf_headers()

    assert "hf_hub/" in headers["user-agent"]
    assert "agent/" not in headers["user-agent"]
    assert not missing.exists()
