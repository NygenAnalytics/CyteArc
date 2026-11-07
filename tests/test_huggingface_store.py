"""Offline tests for the cached, rate-limit-aware Hugging Face store reader."""

import asyncio
import re
import socket
import time
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
from zarr.core.sync import sync
from zarr.storage import FsspecStore

from cytearc.storage import huggingface
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


def _loopback_store(hub, **timeouts):
    """Publish an array in the hub's bucket and open it as an ``hf://`` store."""
    zarr.create_array(
        store=hub.root / "data.zarr",
        data=np.arange(64, dtype=np.int32).reshape(8, 8),
        chunks=(4, 4),
    )
    return huggingface.HfReadStore.from_url(
        LOOPBACK_URI,
        storage_options={"token": False, "endpoint": hub.endpoint, **timeouts},
        read_only=True,
    )


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


@pytest.mark.parametrize(
    ("stalled", "call"),
    [
        ("bucket", "get"),
        ("bucket", "list_dir"),
        *(("tree", call) for call in _STORE_CALLS),
    ],
)
def test_read_store_raises_when_a_request_stalls(loopback_hub, stalled, call):
    # Hugging Face's lookup of a bucket and its tree listings wait on the
    # server without a timeout of their own.
    store = _loopback_store(loopback_hub, request_timeout=0.25, retry_timeout=1)
    loopback_hub.plan[stalled] = ["stall"]
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(_STORE_CALLS[call](store), timeout=GUARD_SECONDS)
    finally:
        store.close()
    # The next attempt would wait 2 s first, past the 1 s that the request has.
    assert time.monotonic() - started < 1.5
    assert str(raised.value).startswith(
        f"{_REQUESTS[call]} did not succeed within 1 s: 1 attempt failed, the last "
        "with no answer within 0.25 s. To wait longer, pass more seconds in "
        "storage_options as request_timeout"
    )
    assert [kind for kind, _ in loopback_hub.requests].count(stalled) == 1


@pytest.mark.parametrize("stalled", ["bucket", "tree", "file"])
def test_read_store_reads_after_abandoning_a_stalled_attempt(
    loopback_hub, monkeypatch, stalled
):
    store = _loopback_store(loopback_hub, request_timeout=0.25)
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
    store = _loopback_store(loopback_hub, request_timeout=0.5, retry_timeout=2.7)
    loopback_hub.plan["tree"] = ["stall", "stall"]
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as raised:
            sync(_names(store.list_dir("")), timeout=GUARD_SECONDS)
    finally:
        store.close()
    elapsed = time.monotonic() - started
    # One attempt of 0.5 s, a wait of 2 s, and an attempt cut to the 0.2 s left.
    assert 2.6 < elapsed < 3.5
    assert re.match(
        rf"Listing '' in {re.escape(LOOPBACK_URI)} did not succeed within 2\.7 s: "
        r"2 attempts failed, the last with no answer within 0\.(1[5-9]|2)\d* s\.",
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
    assert str(raised.value).startswith(
        f"Reading 'zarr.json' from {LOOPBACK_URI} did not succeed within 5 s: "
        "3 attempts failed, the last with ConnectError: "
    )
