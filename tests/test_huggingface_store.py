"""Offline tests for the cached, rate-limit-aware Hugging Face store reader."""

import asyncio
import json
import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import httpx2
import numpy as np
import pytest
import zarr
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import GroupMetadata
from zarr.core.sync import _get_loop, sync
from zarr.storage import FsspecStore

from cytearc.storage import huggingface
from cytearc.storage.async_execution import zarr_decoding_threads
from cytearc.storage.preparation import (
    PREPARATION_CLAIM,
    _claim_document,
    preparation_claim,
)
from cytearc.utils.logging import logger
from tests.fixtures_hub import BUCKET_ID, REPOSITORY_ID
from tests.storage_helpers import recorded_codec_pools

pytestmark = pytest.mark.usefixtures("hub_offline")

NOW = 1_700_000_000.0
LOOPBACK_URI = f"hf://buckets/{BUCKET_ID}/data.zarr"
REPOSITORY_URI = f"hf://datasets/{REPOSITORY_ID}/data.zarr"
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


def test_a_fetch_that_fails_after_a_close_keeps_the_fetch_that_followed_it(
    tmp_path, monkeypatch
):
    calls = []
    prototype = default_buffer_prototype()
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=True)

    async def check():
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def read(self, key, prototype, byte_range=None):
            calls.append(key)
            fetch = len(calls) - 1
            started[fetch].set()
            await release[fetch].wait()
            if fetch == 0:
                raise OSError("The first transfer failed")
            return prototype.buffer.from_bytes(b"metadata")

        monkeypatch.setattr(FsspecStore, "get", read)
        before = asyncio.create_task(store.get("zarr.json", prototype))
        await asyncio.wait_for(started[0].wait(), timeout=5)
        # Closing forgets the fetch in flight, so the next reader starts its
        # own, which the readers after it share.
        store.close()
        after = asyncio.create_task(store.get("zarr.json", prototype))
        await asyncio.wait_for(started[1].wait(), timeout=5)
        release[0].set()
        with pytest.raises(OSError, match="The first transfer failed"):
            await before
        release[1].set()
        assert (await after).to_bytes() == b"metadata"
        # The failed fetch did not drop the one that replaced it.
        assert (await store.get("zarr.json", prototype)).to_bytes() == b"metadata"

    try:
        asyncio.run(check())
        assert calls == ["zarr.json", "zarr.json"]
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
def test_retry_delay_honors_server_headers(monkeypatch, headers, expected):
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


@pytest.mark.parametrize("read_only", [True, False])
def test_read_store_looks_keys_up_through_another_file_system(tmp_path, read_only):
    # A store that a test opens over a local folder has no listing of its own
    # and asks its file system, whichever it was opened with.
    (tmp_path / "chunk").write_bytes(b"source counts")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "key").write_bytes(b"x")
    store = huggingface.HfReadStore.from_url(str(tmp_path), read_only=read_only)
    try:
        found = asyncio.run(store.exists("chunk"))
        missing = asyncio.run(store.exists("missing"))
        size = asyncio.run(store.getsize("chunk"))
        keys = asyncio.run(_names(store.list()))
        under = asyncio.run(_names(store.list_prefix("sub")))
    finally:
        store.close()
    assert (found, missing, size) == (True, False, len(b"source counts"))
    assert (sorted(keys), under) == (["chunk", "sub/key"], ["sub/key"])


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


def _loopback_store(hub, *, read_only=True, uri=LOOPBACK_URI, **options):
    """Publish an array in the hub's bucket and open it as an ``hf://`` store.

    ``options`` are added to the store's storage options.
    """
    if not (hub.root / "data.zarr").exists():
        zarr.create_array(
            store=hub.root / "data.zarr",
            data=np.arange(64, dtype=np.int32).reshape(8, 8),
            chunks=(4, 4),
        )
    return huggingface.HfReadStore.from_url(
        uri,
        storage_options={"token": False, "endpoint": hub.endpoint, **options},
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


def _recorded_requests(hub, kind: str, *, at_least: int) -> int:
    """Count the hub's requests of ``kind`` once it records ``at_least``, or after a while.

    The store gives up on a stalled request after its timeout, and a loaded
    machine can run the hub's handler, which records the request, only after
    that, or not at all when the store gave up before it sent the request.
    """
    deadline = time.monotonic() + 2
    while True:
        count = [recorded for recorded, _ in hub.requests].count(kind)
        if count >= at_least or time.monotonic() > deadline:
            return count
        time.sleep(0.01)


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
# The kind of request that each call of a writable store sends after the
# bucket's lookup.
_WRITABLE_REQUESTS = {
    "get": "file",
    "get_partial_values": "file",
    "exists": "paths-info",
    "getsize": "paths-info",
    "list": "tree",
    "list_prefix": "tree",
    "list_dir": "tree",
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


def test_a_read_on_one_decoding_thread_fetches_on_the_host_threads(
    loopback_hub, monkeypatch
):
    store = _loopback_store(loopback_hub)
    requests: list[int | None] = []
    session = huggingface.get_session
    list_store = huggingface._list_store

    with recorded_codec_pools(host_threads=4) as pools:

        class _Session:
            def get(self, *args, **kwargs):
                requests.append(pools.current_threads())
                return session().get(*args, **kwargs)

        def listing(*args):
            requests.append(pools.current_threads())
            return list_store(*args)

        monkeypatch.setattr(huggingface, "get_session", _Session)
        monkeypatch.setattr(huggingface, "_list_store", listing)
        pools.label = "read"
        try:
            # As a stage of a one-worker store reads.
            with zarr_decoding_threads(1):
                values = zarr.open_array(store=store, mode="r")[:]
        finally:
            pools.label = None
            store.close()

    np.testing.assert_array_equal(values, np.arange(64).reshape(8, 8))
    # The listing, the metadata and the four chunks are fetched on the host's
    # four threads, and the four chunks decode on the one thread asked for.
    assert len(requests) == 6 and set(requests) == {4}
    assert pools.threads("read") == [1] * 4


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_a_writable_store_sends_its_requests_on_the_host_threads(
    loopback_hub, monkeypatch, uri
):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    requests: list[tuple[str, int | None]] = []

    with recorded_codec_pools(host_threads=4) as pools:
        for name in ("_file_entry", "_read_file", "_list_store", "_list_folder"):
            send = getattr(huggingface, name)

            def recorded(*args, name=name, send=send):
                requests.append((name, pools.current_threads()))
                return send(*args)

            monkeypatch.setattr(huggingface, name, recorded)
        try:
            # As a stage of a one-worker store looks a key up, reads it, reads
            # its size and lists keys.
            with zarr_decoding_threads(1):
                found = sync(store.exists("c/1/1"), timeout=GUARD_SECONDS)
                read = sync(store.get("c/1/1", _PROTOTYPE), timeout=GUARD_SECONDS)
                size = sync(store.getsize("c/1/1"), timeout=GUARD_SECONDS)
                listed = sync(_names(store.list_prefix("c")), timeout=GUARD_SECONDS)
                named = sync(_names(store.list_dir("c")), timeout=GUARD_SECONDS)
        finally:
            store.close()

    held = (loopback_hub.root / "data.zarr/c/1/1").read_bytes()
    assert (found, read.to_bytes(), size) == (True, held, len(held))
    assert (sorted(listed), sorted(named)) == (
        ["c/0/0", "c/0/1", "c/1/0", "c/1/1"],
        ["0", "1"],
    )
    # The requests wait on the network, so they run on the host's four threads.
    assert requests == [
        ("_file_entry", 4),
        ("_read_file", 4),
        ("_file_entry", 4),
        ("_list_store", 4),
        ("_list_folder", 4),
    ]


@pytest.mark.parametrize(
    ("read_only", "stalled", "call"),
    [
        *(
            (read_only, "bucket", call)
            for read_only in (True, False)
            for call in ("get", "list_dir")
        ),
        *((True, "tree", call) for call in _STORE_CALLS),
        # A writable store reads a key by its resolver URL, looks it up or
        # reads its size in the bucket, and lists keys with Hugging Face's
        # API.
        *((False, _WRITABLE_REQUESTS[call], call) for call in _STORE_CALLS),
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
    # The message counts the store's attempts; no other attempt reached the hub.
    assert _recorded_requests(loopback_hub, stalled, at_least=1) <= 1


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
    assert _recorded_requests(loopback_hub, "file", at_least=1) <= 1


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
    assert _recorded_requests(loopback_hub, stalled, at_least=2) <= 2


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
    assert _recorded_requests(loopback_hub, "tree", at_least=2) <= 2


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
    # The retries' deadline reads the clock of Zarr's loop, which the recorded
    # sleeps do not advance. Holding the clock still makes the outcome depend on
    # the recorded delays alone, not on how long the refused connections take:
    # on a loaded machine the first two took more than the 1 s that the second
    # delay leaves of the 5 s, and the request gave up after one delay.
    loop = _get_loop()
    held = loop.time()
    monkeypatch.setattr(loop, "time", lambda: held)
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


@pytest.mark.parametrize(
    ("missing", "uri"),
    [
        ("bucket", f"hf://buckets/{BUCKET_ID}/data.zarr"),
        ("repository", f"hf://datasets/{REPOSITORY_ID}/data.zarr"),
    ],
)
def test_read_store_reads_a_store_whose_bucket_or_repository_is_missing_as_empty(
    loopback_hub, missing, uri
):
    loopback_hub.plan[missing] = [(404, {"X-Error-Code": "RepoNotFound"})]
    store = _loopback_store(loopback_hub, uri=uri)
    try:
        exists = sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        keys = sync(_names(store.list()), timeout=GUARD_SECONDS)
        value = sync(store.get("zarr.json", _PROTOTYPE), timeout=GUARD_SECONDS)
        with pytest.raises(FileNotFoundError):
            sync(store.getsize("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert not exists and keys == [] and value is None
    # The lookup that found nothing is the only request: nothing is listed.
    assert loopback_hub.requests == [(missing, "")]


def test_read_store_lists_and_reads_a_dataset_repository(loopback_hub):
    uri = f"hf://datasets/{REPOSITORY_ID}/data.zarr"
    store = _loopback_store(loopback_hub, uri=uri)
    published = loopback_hub.root / "data.zarr"
    keys = sorted(
        path.relative_to(published).as_posix()
        for path in published.rglob("*")
        if path.is_file()
    )
    chunk = (published / "c/1/0").read_bytes()
    try:
        listed = sync(_names(store.list()), timeout=GUARD_SECONDS)
        children = sync(_names(store.list_dir("c")), timeout=GUARD_SECONDS)
        part = sync(
            store.get("c/1/0", _PROTOTYPE, RangeByteRequest(1, 6)),
            timeout=GUARD_SECONDS,
        )
        array = zarr.open_array(store=store, mode="r", zarr_format=3)
        values = array[2:7, 1:6]
    finally:
        store.close()
    # The repository lists folders too, which the store leaves out.
    assert listed == keys
    assert children == ["0", "1"]
    assert part.to_bytes() == chunk[1:6]
    np.testing.assert_array_equal(
        values, np.arange(64, dtype=np.int32).reshape(8, 8)[2:7, 1:6]
    )
    assert _received(loopback_hub, "repository") == [("", None)]
    assert _received(loopback_hub, "tree") == [("data.zarr", "recursive")]
    assert ("data.zarr/c/1/0", "bytes=1-5") in _received(loopback_hub, "file")


def test_read_store_takes_the_asked_bytes_from_a_server_that_ignores_ranges(
    loopback_hub,
):
    store = _loopback_store(loopback_hub)
    chunk = (loopback_hub.root / "data.zarr/c/1/0").read_bytes()
    try:
        assert sync(store.exists("c/1/0"), timeout=GUARD_SECONDS)
        loopback_hub.plan["file"] = ["whole"]
        part = sync(
            store.get("c/1/0", _PROTOTYPE, RangeByteRequest(2, 9)),
            timeout=GUARD_SECONDS,
        )
    finally:
        store.close()
    assert part.to_bytes() == chunk[2:9]
    assert _received(loopback_hub, "file") == [("data.zarr/c/1/0", "bytes=2-8")]


@pytest.mark.parametrize("change", ["shrunk", "grew under a server ignoring ranges"])
def test_read_store_raises_when_a_listed_key_changed_size(loopback_hub, change):
    store = _loopback_store(loopback_hub)
    chunk = loopback_hub.root / "data.zarr/c/1/0"
    listed = chunk.read_bytes()
    try:
        assert sync(store.getsize("c/1/0"), timeout=GUARD_SECONDS) == len(listed)
        # The object is replaced after the store listed it.
        if change == "shrunk":
            chunk.write_bytes(listed[:-3])
        else:
            chunk.write_bytes(listed + b"more")
            loopback_hub.plan["file"] = ["whole"]
        with pytest.raises(OSError) as raised:
            sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    returned = len(listed) - 3 if change == "shrunk" else len(listed) + 4
    # Not FileNotFoundError, which Zarr reads as a missing chunk, and not
    # retried.
    assert type(raised.value) is OSError
    assert str(raised.value) == (
        f"Reading bytes 0 to {len(listed)} of 'c/1/0' from {LOOPBACK_URI} returned "
        f"{returned} bytes: the store changed after it was listed. Open the store "
        "again to read what it holds now."
    )
    assert _received(loopback_hub, "file") == [
        ("data.zarr/c/1/0", f"bytes=0-{len(listed) - 1}")
    ]


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


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_reads_a_key_with_one_request_of_its_resolver_url(
    loopback_hub, uri
):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    held = (loopback_hub.root / "data.zarr/c/1/0").read_bytes()
    try:
        chunk = sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
        part = sync(
            store.get("c/1/0", _PROTOTYPE, RangeByteRequest(2, 6)),
            timeout=GUARD_SECONDS,
        )
        tail = sync(
            store.get("c/1/0", _PROTOTYPE, SuffixByteRequest(3)),
            timeout=GUARD_SECONDS,
        )
        empty = sync(
            store.get("c/1/0", _PROTOTYPE, RangeByteRequest(4, 4)),
            timeout=GUARD_SECONDS,
        )
        found = sync(store.exists("c/0/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert (chunk.to_bytes(), part.to_bytes(), tail.to_bytes(), empty.to_bytes()) == (
        held,
        held[2:6],
        held[-3:],
        b"",
    )
    assert found
    # Each read is one GET of the key's resolver URL for the bytes asked, and
    # a suffix takes the key's size from a lookup first, as does a read of no
    # bytes, which needs nothing else. Nothing is listed.
    assert _received(loopback_hub, "file") == [
        ("data.zarr/c/1/0", None),
        ("data.zarr/c/1/0", "bytes=2-5"),
        ("data.zarr/c/1/0", f"bytes={len(held) - 3}-{len(held) - 1}"),
    ]
    assert _received(loopback_hub, "paths-info") == [
        ("data.zarr/c/1/0", None),
        ("data.zarr/c/1/0", None),
        ("data.zarr/c/0/1", None),
    ]
    assert _received(loopback_hub, "tree") == []


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
@pytest.mark.parametrize("read_only", [True, False])
def test_read_store_reads_a_directory_as_no_key(loopback_hub, read_only, uri):
    # The Hub names a folder in its answer to a lookup, and a store's keys are
    # its files.
    store = _loopback_store(loopback_hub, read_only=read_only, uri=uri)
    try:
        directory = sync(store.exists("c/1"), timeout=GUARD_SECONDS)
        key = sync(store.exists("c/1/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert (directory, key) == (False, True)


def test_writable_store_lists_its_keys_anew_for_each_call(loopback_hub, monkeypatch):
    # A writable store keeps no listing, so it sees the keys that it writes
    # after it listed.
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    published = loopback_hub.root / "data.zarr"
    keys = sorted(
        path.relative_to(published).as_posix()
        for path in published.rglob("*")
        if path.is_file()
    )
    try:
        listed = sync(_names(store.list()), timeout=GUARD_SECONDS)
        under_c = sync(_names(store.list_prefix("c/1")), timeout=GUARD_SECONDS)
        # A key that the store writes after it listed its keys.
        sync(
            store.set("c/1/9", _PROTOTYPE.buffer.from_bytes(b"late")),
            timeout=GUARD_SECONDS,
        )
        relisted = sync(_names(store.list_prefix("c/1")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert sorted(listed) == keys
    assert sorted(under_c) == [key for key in keys if key.startswith("c/1/")]
    assert uploads == ["data.zarr/c/1/9"]
    assert sorted(relisted) == sorted([*under_c, "c/1/9"])
    # Each call lists the store, or the folder asked for, again.
    assert _received(loopback_hub, "tree") == [
        ("data.zarr", "recursive"),
        ("data.zarr/c/1", "recursive"),
        ("data.zarr/c/1", "recursive"),
    ]


def _accept_uploads(hub, store, monkeypatch):
    """Write each file that ``store`` uploads where ``hub`` serves it, and list them.

    The loopback hub answers reads only, so the bucket's upload call writes
    into the directory that the hub serves instead, and removes from it the
    files that the call deletes, with the folders they leave empty, since a
    bucket holds no empty folder.
    """
    uploads = []

    def upload(bucket_id, *, add=(), delete=()):
        assert bucket_id == BUCKET_ID
        for source, remote in add:
            uploads.append(remote)
            target = hub.root / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
        for remote in delete:
            _delete_from_bucket(hub, remote)

    monkeypatch.setattr(store.fs.sync_fs._api, "batch_bucket_files", upload)
    return uploads


def _delete_from_bucket(hub, remote):
    """Delete a file that ``hub`` serves, with the folders it leaves empty."""
    target = hub.root / remote
    target.unlink()
    for folder in target.parents:
        if folder == hub.root or any(folder.iterdir()):
            break
        folder.rmdir()


def _publish_group(hub):
    """Publish a group with one attribute as the hub's ``data.zarr``."""
    zarr.create_group(store=hub.root / "data.zarr", attributes={"title": "kept"})


def _published_attributes(hub):
    return zarr.open_group(hub.root / "data.zarr", mode="r").attrs.asdict()


@contextmanager
def _warnings_logged() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


# The document that Zarr writes for each parent group of a node it creates,
# unless the parent's key exists.
_EMPTY_GROUP = GroupMetadata(zarr_format=3).to_buffer_dict(_PROTOTYPE)["zarr.json"]


@pytest.mark.parametrize("failing", ["bucket", "paths-info"])
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
    loopback_hub.plan["paths-info"] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        found = sync(store.exists(key), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert not found
    # The answer comes from the lookup after the failed one.
    assert sleeps == [2.0]
    assert loopback_hub.plan["paths-info"] == []


@pytest.mark.parametrize("failing", ["bucket", "paths-info"])
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
    loopback_hub.plan["paths-info"] = [503, 503, 503]
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
    assert [kind for kind, _ in loopback_hub.requests].count("paths-info") == 3


def test_writable_store_ends_the_thread_of_a_lookup_it_abandoned(
    loopback_hub, monkeypatch
):
    ended = threading.Event()
    file_entry = huggingface._file_entry

    def recorded(*args):
        try:
            return file_entry(*args)
        finally:
            ended.set()

    monkeypatch.setattr(huggingface, "_file_entry", recorded)
    store = _loopback_store(
        loopback_hub, read_only=False, request_timeout=0.25, retry_timeout=1
    )
    loopback_hub.plan["paths-info"] = ["stall"]
    try:
        with pytest.raises(TimeoutError):
            sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        # The store abandons the attempt, and the request's own timeout ends
        # the thread that waits on the server, which never answers.
        assert ended.wait(timeout=GUARD_SECONDS)
    finally:
        store.close()


def test_writable_store_does_not_write_over_a_key_whose_lookup_failed(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    loopback_hub.plan["paths-info"] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        sync(store.set_if_not_exists("zarr.json", _EMPTY_GROUP), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert uploads == []
    assert _published_attributes(loopback_hub) == {"title": "kept"}
    assert sleeps == [2.0]
    assert _received(loopback_hub, "paths-info") == [("data.zarr/zarr.json", None)] * 2


def test_writable_store_keeps_a_parent_group_when_creating_a_child_meets_a_server_error(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    sleeps = _record_sleeps(monkeypatch)
    root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
    # Zarr looks up the parent's document before it would write one, and the
    # lookup meets a 503.
    loopback_hub.plan["paths-info"] = [503]
    try:
        root.create_group("child")
    finally:
        store.close()
    assert _published_attributes(loopback_hub) == {"title": "kept"}
    assert uploads == ["data.zarr/child/zarr.json"]
    assert sleeps == [2.0]
    assert _received(loopback_hub, "paths-info") == [("data.zarr/zarr.json", None)] * 2


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
    # The first lookup of the claim fails.
    loopback_hub.plan["paths-info"] = [503]
    try:
        with pytest.raises(
            RuntimeError, match="is being prepared by another process"
        ) as refused:
            with preparation_claim(root):
                pass
    finally:
        store.close()
    assert str(refused.value).endswith(
        f"remove {LOOPBACK_URI}/{PREPARATION_CLAIM} and open the store again."
    )
    assert uploads == []
    assert held.read_bytes() == claim
    assert sleeps == [2.0]
    assert (
        _received(loopback_hub, "paths-info")
        == [(f"data.zarr/{PREPARATION_CLAIM}/zarr.json", None)] * 2
    )


def test_writable_store_finds_a_key_that_another_process_wrote_after_a_lookup(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    directory = loopback_hub.root / "data.zarr" / "sub"
    directory.mkdir()
    (directory / "other").write_bytes(b"x")
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    mine = _PROTOTYPE.buffer.from_bytes(b"mine")
    try:
        # Hugging Face's file system would answer this lookup from a listing
        # of the key's directory, and keep the listing.
        missing = sync(store.exists("sub/key"), timeout=GUARD_SECONDS)
        # Another process writes the key.
        (directory / "key").write_bytes(b"theirs")
        found = sync(store.exists("sub/key"), timeout=GUARD_SECONDS)
        sync(store.set_if_not_exists("sub/key", mine), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert (missing, found) == (False, True)
    assert (directory / "key").read_bytes() == b"theirs"
    assert uploads == []
    # Each lookup asks the bucket about the key alone, without a listing.
    assert _received(loopback_hub, "paths-info") == [("data.zarr/sub/key", None)] * 3
    assert _received(loopback_hub, "tree") == []


def test_writable_store_misses_a_key_that_another_process_deleted_after_a_lookup(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    key = loopback_hub.root / "data.zarr" / "sub" / "key"
    key.parent.mkdir()
    (key.parent / "other").write_bytes(b"x")
    key.write_bytes(b"theirs")
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    mine = _PROTOTYPE.buffer.from_bytes(b"mine")
    try:
        found = sync(store.exists("sub/key"), timeout=GUARD_SECONDS)
        # Another process deletes the key.
        key.unlink()
        missing = sync(store.exists("sub/key"), timeout=GUARD_SECONDS)
        sync(store.set_if_not_exists("sub/key", mine), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert (found, missing) == (True, False)
    assert key.read_bytes() == b"mine"
    assert uploads == ["data.zarr/sub/key"]


def test_writable_store_keeps_a_parent_group_that_another_process_wrote_after_a_lookup(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    published = loopback_hub.root / "data.zarr"
    # Another process has written a child of group "a", but not yet the
    # group's own document.
    (published / "a" / "x").mkdir(parents=True)
    (published / "a" / "x" / "zarr.json").write_bytes(_EMPTY_GROUP.to_bytes())
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
    try:
        missing = sync(store.exists("a/zarr.json"), timeout=GUARD_SECONDS)
        # The other process writes the group's document, with an attribute.
        zarr.create_group(store=published / "a", attributes={"by": "another"})
        # Zarr writes a parent's document only where its lookup finds none.
        root.create_group("a/y")
    finally:
        store.close()
    assert not missing
    assert zarr.open_group(published / "a", mode="r").attrs.asdict() == {
        "by": "another"
    }
    assert uploads == ["data.zarr/a/y/zarr.json"]


def test_writable_repository_store_looks_each_key_up_by_its_path(loopback_hub):
    store = _loopback_store(loopback_hub, read_only=False, uri=REPOSITORY_URI)
    key = loopback_hub.root / "data.zarr" / "c" / "1" / "9"
    try:
        missing = sync(store.exists("c/1/9"), timeout=GUARD_SECONDS)
        # Another process commits the key, and then a commit that deletes it.
        key.write_bytes(b"theirs")
        found = sync(store.exists("c/1/9"), timeout=GUARD_SECONDS)
        key.unlink()
        deleted = sync(store.exists("c/1/9"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert (missing, found, deleted) == (False, True, False)
    assert _received(loopback_hub, "paths-info") == [("data.zarr/c/1/9", None)] * 3
    assert _received(loopback_hub, "tree") == []


def test_writable_repository_store_sends_its_lookup_as_huggingface_hub_does(
    monkeypatch,
):
    # The loopback hub serves revision main only, so the request that the
    # store sends for another revision is compared with the one that
    # huggingface_hub's own lookup of the same paths sends.
    from huggingface_hub import hf_api
    from huggingface_hub.hf_file_system import HfFileSystemResolvedRepositoryPath

    sent = []

    class Session:
        def post(self, url, **kwargs):
            sent.append((url, kwargs.get("data"), kwargs.get("json")))
            return httpx2.Response(200, json=[], request=httpx2.Request("POST", url))

    monkeypatch.setattr(huggingface, "get_session", Session)
    monkeypatch.setattr(hf_api, "get_session", Session)
    endpoint = "http://127.0.0.1:9"
    fs = huggingface.HfFileSystem(
        endpoint=endpoint, token=False, skip_instance_cache=True
    )
    key = "data.zarr/zarr.json"
    resolved = HfFileSystemResolvedRepositoryPath(
        "dataset", "org/name", "refs/pr/1", key
    )
    monkeypatch.setattr(fs, "resolve_path", lambda path: resolved)

    found = huggingface._file_entry(fs, "datasets/org/name@refs%2Fpr%2F1/" + key, 5)
    HfApi(endpoint=endpoint, token=False).get_paths_info(
        "org/name", [key], revision="refs/pr/1", repo_type="dataset"
    )
    assert found is None
    assert sent[0] == sent[1]
    assert sent[0][0] == f"{endpoint}/api/datasets/org/name/paths-info/refs%2Fpr%2F1"


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_looks_a_key_up_with_its_token(loopback_hub, uri):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri, token="hf_test")
    try:
        found = sync(store.exists("c/1/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert found
    assert _received(loopback_hub, "paths-info") == [
        ("data.zarr/c/1/1", "Bearer hf_test")
    ]


# Bytes of distinct values, so that a slice of the wrong bytes cannot pass.
_WRITTEN = bytes(range(256)) * 4


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_reads_a_key_that_another_process_wrote_after_a_read(
    loopback_hub, uri
):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    key = loopback_hub.root / "data.zarr" / "c" / "1" / "9"
    reads = {
        "whole": None,
        "range": RangeByteRequest(10, 20),
        "offset": OffsetByteRequest(1000),
        "suffix": SuffixByteRequest(5),
        "past the end": OffsetByteRequest(2000),
        "empty": RangeByteRequest(7, 7),
    }
    try:
        # Hugging Face's file system would keep its listing of the key's
        # folder from this read.
        sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
        # Another process writes the key.
        key.write_bytes(_WRITTEN)
        read = {
            name: sync(store.get("c/1/9", _PROTOTYPE, request), timeout=GUARD_SECONDS)
            for name, request in reads.items()
        }
        size = sync(store.getsize("c/1/9"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert {name: value.to_bytes() for name, value in read.items()} == {
        "whole": _WRITTEN,
        "range": _WRITTEN[10:20],
        "offset": _WRITTEN[1000:],
        "suffix": _WRITTEN[-5:],
        "past the end": b"",
        "empty": b"",
    }
    assert size == len(_WRITTEN)


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_reads_a_key_that_another_process_deleted_as_missing(
    loopback_hub, uri
):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    key = loopback_hub.root / "data.zarr" / "c" / "1" / "1"
    try:
        held = sync(store.get("c/1/1", _PROTOTYPE), timeout=GUARD_SECONDS)
        # Another process deletes the key.
        key.unlink()
        reads = [
            sync(store.get("c/1/1", _PROTOTYPE, request), timeout=GUARD_SECONDS)
            for request in (None, RangeByteRequest(0, 4), SuffixByteRequest(4))
        ]
        with pytest.raises(FileNotFoundError):
            sync(store.getsize("c/1/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert held is not None
    assert reads == [None, None, None]


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_lists_the_keys_that_another_process_wrote_or_deleted(
    loopback_hub, uri
):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    published = loopback_hub.root / "data.zarr"

    def listings():
        return (
            sorted(sync(_names(store.list()), timeout=GUARD_SECONDS)),
            sorted(sync(_names(store.list_prefix("c/1")), timeout=GUARD_SECONDS)),
            sorted(sync(_names(store.list_dir("c/1")), timeout=GUARD_SECONDS)),
        )

    try:
        before = listings()
        # Another process writes c/1/9 and deletes c/1/1.
        (published / "c" / "1" / "9").write_bytes(b"theirs")
        (published / "c" / "1" / "1").unlink()
        after = listings()
    finally:
        store.close()
    keys = sorted(
        path.relative_to(published).as_posix()
        for path in published.rglob("*")
        if path.is_file()
    )
    assert before == (
        ["c/0/0", "c/0/1", "c/1/0", "c/1/1", "zarr.json"],
        ["c/1/0", "c/1/1"],
        ["0", "1"],
    )
    assert after == (keys, ["c/1/0", "c/1/9"], ["0", "9"])


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_lists_a_prefix_as_a_folder_or_a_key(loopback_hub, uri):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    # A folder whose name starts with the name of another, which a listing
    # matched as text would also name.
    beside = loopback_hub.root / "data.zarr" / "c" / "1x"
    beside.mkdir()
    (beside / "0").write_bytes(b"x")
    try:
        folder = sync(_names(store.list_prefix("c/1")), timeout=GUARD_SECONDS)
        slashed = sync(_names(store.list_prefix("c/1/")), timeout=GUARD_SECONDS)
        key = sync(_names(store.list_prefix("zarr.json")), timeout=GUARD_SECONDS)
        names = sync(_names(store.list_dir("c")), timeout=GUARD_SECONDS)
        absent = sync(_names(store.list_dir("c/9")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert sorted(folder) == sorted(slashed) == ["c/1/0", "c/1/1"]
    assert key == ["zarr.json"]
    assert sorted(names) == ["0", "1", "1x"]
    assert absent == []


def test_writable_store_lists_a_folder_without_the_folders_that_share_its_name(
    monkeypatch,
):
    # huggingface_hub notes that a bucket matches a listed prefix as text, so
    # that listing "logs" also names "logs_existing/...".
    def tree(fs, resolved, path, *, recursive):
        assert (path, recursive) == ("data.zarr/c/1", False)
        return [
            SimpleNamespace(path="data.zarr/c/1/0"),
            SimpleNamespace(path="data.zarr/c/10"),
            SimpleNamespace(path="data.zarr/c/1/1"),
        ]

    monkeypatch.setattr(huggingface, "_list_tree", tree)
    fs = huggingface.HfFileSystem(token=False, skip_instance_cache=True)
    fs._bucket_exists_cache[BUCKET_ID] = (True, None)
    names = huggingface._list_folder(fs, f"buckets/{BUCKET_ID}/data.zarr/c/1")
    assert names == ["0", "1"]


@pytest.mark.parametrize(
    ("missing", "uri"), [("bucket", LOOPBACK_URI), ("repository", REPOSITORY_URI)]
)
def test_writable_store_reads_a_store_whose_bucket_or_repository_is_missing_as_empty(
    loopback_hub, missing, uri
):
    loopback_hub.plan[missing] = [(404, {"X-Error-Code": "RepoNotFound"})]
    store = _loopback_store(loopback_hub, read_only=False, uri=uri)
    try:
        exists = sync(store.exists("zarr.json"), timeout=GUARD_SECONDS)
        value = sync(store.get("zarr.json", _PROTOTYPE), timeout=GUARD_SECONDS)
        keys = sync(_names(store.list()), timeout=GUARD_SECONDS)
        under = sync(_names(store.list_prefix("c")), timeout=GUARD_SECONDS)
        names = sync(_names(store.list_dir("")), timeout=GUARD_SECONDS)
        with pytest.raises(FileNotFoundError):
            sync(store.getsize("zarr.json"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert (exists, value, keys, under, names) == (False, None, [], [], [])
    # The lookup that found nothing is the only request.
    assert loopback_hub.requests == [(missing, "")]


@pytest.mark.parametrize("failing", ["bucket", "file"])
def test_writable_store_retries_a_read_that_meets_a_server_error(
    loopback_hub, monkeypatch, failing
):
    store = _loopback_store(loopback_hub, read_only=False)
    loopback_hub.plan[failing] = [503]
    sleeps = _record_sleeps(monkeypatch)
    try:
        chunk = sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert chunk.to_bytes() == (loopback_hub.root / "data.zarr/c/1/0").read_bytes()
    assert sleeps == [2.0]
    assert [kind for kind, _ in loopback_hub.requests].count(failing) == 2


@pytest.mark.parametrize("status", [401, 403])
def test_writable_store_raises_a_read_that_the_server_refuses(
    loopback_hub, monkeypatch, status
):
    store = _loopback_store(loopback_hub, read_only=False)
    loopback_hub.plan["file"] = [status]
    sleeps = _record_sleeps(monkeypatch)
    try:
        with pytest.raises(HfHubHTTPError) as raised:
            sync(store.get("c/1/0", _PROTOTYPE), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert raised.value.response.status_code == status
    assert sleeps == []


def test_writable_store_takes_the_asked_bytes_from_a_server_that_ignores_ranges(
    loopback_hub,
):
    store = _loopback_store(loopback_hub, read_only=False)
    loopback_hub.plan["file"] = ["whole"] * 3
    held = (loopback_hub.root / "data.zarr/c/1/0").read_bytes()
    reads = (RangeByteRequest(2, 6), OffsetByteRequest(5), SuffixByteRequest(3))
    try:
        values = [
            sync(store.get("c/1/0", _PROTOTYPE, request), timeout=GUARD_SECONDS)
            for request in reads
        ]
    finally:
        store.close()
    assert [value.to_bytes() for value in values] == [held[2:6], held[5:], held[-3:]]
    assert loopback_hub.plan["file"] == []


@pytest.mark.parametrize("uri", [LOOPBACK_URI, REPOSITORY_URI])
def test_writable_store_sends_its_token_with_every_request(loopback_hub, uri):
    store = _loopback_store(loopback_hub, read_only=False, uri=uri, token="hf_test")
    try:
        sync(store.get("c/1/1", _PROTOTYPE), timeout=GUARD_SECONDS)
        sync(
            store.get("c/1/1", _PROTOTYPE, SuffixByteRequest(2)), timeout=GUARD_SECONDS
        )
        sync(store.getsize("c/1/1"), timeout=GUARD_SECONDS)
        sync(_names(store.list()), timeout=GUARD_SECONDS)
        sync(_names(store.list_dir("c")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    kinds = {kind for kind, _ in loopback_hub.authorizations}
    assert kinds >= {"file", "paths-info", "tree"}
    assert {token for _, token in loopback_hub.authorizations} == {"Bearer hf_test"}


def test_writable_store_deletes_a_folder_with_a_key_that_another_process_wrote(
    loopback_hub, monkeypatch
):
    store = _loopback_store(loopback_hub, read_only=False)
    _accept_uploads(loopback_hub, store, monkeypatch)
    folder = loopback_hub.root / "data.zarr" / "c" / "1"
    try:
        listed = sorted(sync(_names(store.list_dir("c/1")), timeout=GUARD_SECONDS))
        # Another process writes a key into the folder.
        (folder / "9").write_bytes(b"theirs")
        sync(store.delete_dir("c/1"), timeout=GUARD_SECONDS)
    finally:
        store.close()
    assert listed == ["0", "1"]
    assert not folder.exists()


def test_writable_store_claims_a_preparation_that_another_process_released(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    held = loopback_hub.root / "data.zarr" / PREPARATION_CLAIM / "zarr.json"
    held.parent.mkdir()
    held.write_bytes(_claim_document("another process").to_bytes())
    store = _loopback_store(loopback_hub, read_only=False)
    uploads = _accept_uploads(loopback_hub, store, monkeypatch)
    root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
    try:
        with pytest.raises(RuntimeError, match="is being prepared by another process"):
            with preparation_claim(root):
                raise AssertionError("a second holder must not start")
        # The other process's preparation ends before the store is prepared,
        # and it removes its claim.
        _delete_from_bucket(loopback_hub, f"data.zarr/{PREPARATION_CLAIM}/zarr.json")
        with _warnings_logged() as warnings:
            with preparation_claim(root):
                during = json.loads(held.read_bytes())
    finally:
        store.close()
    token = during["attributes"]["cytearc:preparationClaim"]["token"]
    assert re.fullmatch("[0-9a-f]{32}", token)
    assert not held.exists()
    assert uploads == [f"data.zarr/{PREPARATION_CLAIM}/zarr.json"]
    # The claim of an hf:// store cannot exclude a second process that opens
    # the store at the same moment, and this open says so once.
    assert warnings == [
        f"Another process that opens {LOOPBACK_URI} at the same moment as this "
        "one could also prepare it, because the store cannot refuse a second "
        "writer. Open it from one process until it is prepared."
    ]


def test_writable_store_names_a_claim_that_it_cannot_remove_by_its_url(
    loopback_hub, monkeypatch
):
    _publish_group(loopback_hub)
    store = _loopback_store(loopback_hub, read_only=False)
    _accept_uploads(loopback_hub, store, monkeypatch)
    upload = store.fs.sync_fs._api.batch_bucket_files

    def refuse_deletes(bucket_id, *, add=(), delete=()):
        if delete:
            raise PermissionError("the bucket refuses deletes")
        upload(bucket_id, add=add)

    monkeypatch.setattr(store.fs.sync_fs._api, "batch_bucket_files", refuse_deletes)
    root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
    try:
        with _warnings_logged() as warnings:
            with preparation_claim(root):
                pass
    finally:
        store.close()
    # The second warning names the claim that the user removes.
    assert warnings[1:] == [
        f"Could not remove the preparation claim {LOOPBACK_URI}/{PREPARATION_CLAIM}: "
        "the bucket refuses deletes. If the store is not prepared, remove the "
        "claim before opening it again."
    ]
    assert warnings[0].startswith(f"Another process that opens {LOOPBACK_URI} ")


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
