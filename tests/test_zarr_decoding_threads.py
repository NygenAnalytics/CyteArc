"""Reads that a memory ledger charges decode on as many threads of Zarr's codec pool as the ledger counts, on any machine."""

import contextvars
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.core.sync as zarr_sync
from zarr.codecs import ZstdCodec

from cytearc.metadata.rows import read_rows, read_whole
from cytearc.storage import async_execution
from cytearc.storage.async_execution import (
    ensure_zarr_host_ceiling,
    zarr_decoding_threads,
    zarr_io_concurrency,
)
from cytearc.utils.memory_budget import MemoryLedger

#: Which reader a decode belongs to, set in each reading thread.
_READER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "reader", default=None
)


class _Decodes:
    """The thread that ran each zstd decode, by name, with the reader it decoded for.

    The first ``together`` decodes wait for each other, so that many threads
    decode at once, and with ``each_reader`` the first decode of each of
    those readers waits for the first of the others, so their reads are in
    flight together.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        together: int = 0,
        each_reader: tuple[str, ...] = (),
    ) -> None:
        self.seen: list[tuple[str, str | None]] = []
        lock = threading.Lock()
        waiting = together or len(each_reader)
        barrier = threading.Barrier(waiting, timeout=20) if waiting else None
        started: set[str | None] = set()
        # Decodes since recording began, which clearing ``seen`` keeps.
        decoded = [0]
        decode = ZstdCodec._decode_sync

        def recording(codec, chunk_bytes, chunk_spec):
            reader = _READER.get()
            with lock:
                self.seen.append((threading.current_thread().name, reader))
                decoded[0] += 1
                if each_reader:
                    wait = reader in each_reader and reader not in started
                    started.add(reader)
                else:
                    wait = decoded[0] <= together
            if barrier is not None and wait:
                barrier.wait()
            return decode(codec, chunk_bytes, chunk_spec)

        monkeypatch.setattr(ZstdCodec, "_decode_sync", recording)

    def threads(self, reader: str | None = None) -> set[str]:
        return {name for name, of in self.seen if reader is None or of == reader}


def _values(root: zarr.Group, n_chunks: int = 48) -> np.ndarray:
    values = np.arange(1_000 * n_chunks, dtype=np.int64)
    root.create_array(
        "values", data=values, chunks=(1_000,), compressors=ZstdCodec(level=1)
    )
    return values


def _read_whole(root: zarr.Group, threads: int) -> np.ndarray:
    ledger = MemoryLedger(None, "Reading", codec_threads=threads)
    return read_whole(ledger, root, "values", item="the values")


def _read_rows(root: zarr.Group, threads: int) -> np.ndarray:
    ledger = MemoryLedger(None, "Reading", codec_threads=threads)
    rows = np.arange(0, root["values"].shape[0], 7)
    return read_rows(ledger, root["values"], rows, item="the rows")


@pytest.mark.parametrize("read", [_read_whole, _read_rows], ids=["whole", "rows"])
def test_a_read_decodes_on_its_ledgers_threads_after_a_wider_read(
    tmp_path, monkeypatch, read
) -> None:
    # Forty-eight chunks read whole ten at a time, charged to six codec
    # threads, decode on six at once; each read after it decodes on no more
    # threads than its ledger counts, and gives the same values.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    expected = values if read is _read_whole else values[::7]
    decodes = _Decodes(monkeypatch, together=6)
    with zarr_io_concurrency(10):
        np.testing.assert_array_equal(_read_whole(root, 6), values)
        assert len(decodes.threads()) == 6
        for threads in (1, 2, 3, 1):
            decodes.seen.clear()
            np.testing.assert_array_equal(read(root, threads), expected)
            assert 1 <= len(decodes.threads()) <= threads, decodes.threads()
            assert all(f"_of_{threads}_" in name for name in decodes.threads())


def test_a_read_at_the_host_threads_decodes_on_zarrs_pool_as_any_other_read(
    tmp_path, monkeypatch
) -> None:
    # The default nthreads is the host's, whose reads decode on the threads
    # that every other read of the process decodes on.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    host = ensure_zarr_host_ceiling()
    decodes = _Decodes(monkeypatch)
    with zarr_io_concurrency(10):
        np.testing.assert_array_equal(_read_whole(root, host), values)
        np.testing.assert_array_equal(root["values"][...], values)
    pool = zarr_sync._get_loop()._default_executor
    assert decodes.threads() <= {thread.name for thread in pool._threads}
    assert not any("_of_" in name for name in decodes.threads())


@pytest.mark.parametrize(
    "counts",
    [(1, 3), (2, 2), (3, 1, 3)],
    ids=["one-beside-three", "two-beside-two", "three-one-three"],
)
def test_reads_in_flight_together_decode_on_the_pool_of_their_own_count(
    tmp_path, monkeypatch, counts
) -> None:
    # Each reader's first decode waits for the first of the others, so the
    # reads are in flight together. Each decodes on no more threads than its
    # ledger counts; readers of one count share one pool of that many
    # threads, and readers of different counts share no thread.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    readers = tuple(f"reader {index}" for index in range(len(counts)))
    decodes = _Decodes(monkeypatch, each_reader=readers)
    results: dict[str, np.ndarray] = {}

    def reading(reader: str, threads: int) -> None:
        _READER.set(reader)
        with zarr_io_concurrency(4):
            results[reader] = _read_whole(root, threads)

    callers = [
        threading.Thread(target=reading, args=(reader, threads))
        for reader, threads in zip(readers, counts, strict=True)
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=60)
    assert sorted(results) == sorted(readers)
    for reader in readers:
        np.testing.assert_array_equal(results[reader], values)
    by_count: dict[int, set[str]] = {}
    for reader, threads in zip(readers, counts, strict=True):
        assert 1 <= len(decodes.threads(reader)) <= threads
        by_count.setdefault(threads, set()).update(decodes.threads(reader))
    for threads, names in by_count.items():
        assert len(names) <= threads
    pools = list(by_count.values())
    assert all(a.isdisjoint(b) for i, a in enumerate(pools) for b in pools[i + 1 :])


def test_a_pool_that_replaced_cytearcs_is_replaced_before_the_next_charged_read(
    tmp_path, monkeypatch
) -> None:
    # A pool set on Zarr's loop after CyteArc's, as a library or Zarr itself
    # can set one, gives way to CyteArc's when a charged read begins.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    ensure_zarr_host_ceiling()
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(loop, "_default_executor", loop._default_executor)
    decodes = _Decodes(monkeypatch)
    with ThreadPoolExecutor(8, thread_name_prefix="foreign") as foreign:
        loop.set_default_executor(foreign)
        with zarr_io_concurrency(10):
            np.testing.assert_array_equal(_read_whole(root, 1), values)
        assert len(decodes.threads()) == 1
        assert not any(name.startswith("foreign") for name in decodes.threads())
        assert isinstance(loop._default_executor, async_execution._DecodingPools)


def test_the_installed_pool_keeps_the_size_of_the_pool_zarr_ran_before(
    monkeypatch,
) -> None:
    # A pool that Zarr's loop ran before CyteArc's first use sets the host
    # threads, and CyteArc's pools take its place with as many.
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(loop, "_default_executor", None)
    monkeypatch.setattr(zarr_sync, "_executor", None)
    with ThreadPoolExecutor(5) as earlier:
        loop.set_default_executor(earlier)
        assert ensure_zarr_host_ceiling() == 5
        pools = loop._default_executor
        assert isinstance(pools, async_execution._DecodingPools)
        assert pools._max_workers == 5
        assert zarr_sync._executor is pools
        assert ensure_zarr_host_ceiling() == 5
        assert loop._default_executor is pools
        pools.shutdown()


def test_nested_scopes_decode_on_the_innermost_count_and_restore_the_outer(
    tmp_path, monkeypatch
) -> None:
    # A scope left by an error restores the count of the scope around it,
    # and leaving that restores the host threads.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    decodes = _Decodes(monkeypatch)

    def decoded_on() -> set[str]:
        decodes.seen.clear()
        np.testing.assert_array_equal(root["values"][...], values)
        return decodes.threads()

    with zarr_io_concurrency(10), zarr_decoding_threads(3):
        with pytest.raises(KeyError), zarr_decoding_threads(1):
            inner = decoded_on()
            raise KeyError("left early")
        outer = decoded_on()
    outside = decoded_on()
    assert inner and all("_of_1_" in name for name in inner)
    assert outer and all("_of_3_" in name for name in outer)
    assert outside and not any("_of_" in name for name in outside)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2", None])
def test_decoding_threads_are_a_whole_number_of_at_least_one(threads) -> None:
    with pytest.raises(ValueError, match="whole number of threads of at least 1"):
        with zarr_decoding_threads(threads):
            pytest.fail("the scope must not open")


def test_shutting_the_pools_down_shuts_down_the_pool_of_every_count() -> None:
    pools = async_execution._DecodingPools(2)
    for threads in (1, 3):
        with zarr_decoding_threads(threads):
            assert (
                pools.submit(threading.current_thread)
                .result(5)
                .name.startswith(f"zarr_pool_of_{threads}_")
            )
    assert (
        pools.submit(threading.current_thread).result(5).name.startswith("zarr_pool_")
    )
    pools.shutdown()
    for threads in (None, 1, 3, 5):
        token = async_execution._DECODING_THREADS.set(threads)
        try:
            with pytest.raises(RuntimeError, match="after shutdown"):
                pools.submit(threading.current_thread)
        finally:
            async_execution._DECODING_THREADS.reset(token)


_LOADED_READS_SCRIPT = r"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr

from cytearc.metadata.rows import read_whole
from cytearc.storage.async_execution import zarr_io_concurrency
from cytearc.utils.memory_budget import MemoryLedger, observed_charges
from tests.memory_helpers import ZARR_READS, audited, hold_finished_reads, traced_peak

# Two CPUs shared with four processes that keep them busy, as a busy runner
# shares them: a pool thread that finished a decode waits to run again.
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
busy = [
    subprocess.Popen(
        [sys.executable, "-c", "import time\nend = time.time() + 120\nwhile time.time() < end: pass"]
    )
    for _ in range(4)
]


def incompressible(root, name, n_values, chunk):
    values = np.random.default_rng(len(name)).integers(0, 2**31 - 1, size=n_values)
    root.create_array(name, shape=(n_values,), chunks=(chunk,), dtype=np.int64)[...] = values


def planned(read):
    plans = []
    with observed_charges(lambda _item, _nbytes, plan: plans.append(plan)):
        read(MemoryLedger(None, "Planning"))
    return max(plans)


try:
    with tempfile.TemporaryDirectory() as tmp:
        root = zarr.open_group(Path(tmp) / "store.zarr", mode="w")
        incompressible(root, "first", 400_000, 100_000)
        incompressible(root, "second", 1_000, 1_000)

        def whole(ledger):
            return read_whole(ledger, root, "first", item="the first array")

        def run(threads):
            ledger = MemoryLedger(None, "Reading", codec_threads=threads)
            values = whole(ledger)
            del values
            ledger.release(8 * 400_000)
            read_whole(ledger, root, "second", item="the second array")

        with zarr_io_concurrency(1):
            whole(MemoryLedger(None, "Warming"))
            run(4)
            estimate = planned(whole)
            peaks = [
                traced_peak(lambda: whole(MemoryLedger(None, "Reading")), estimate)[1]
                for _ in range(int(sys.argv[1]))
            ]
            overruns = []
            for _ in range(int(sys.argv[2])):
                with hold_finished_reads(*ZARR_READS):
                    overruns += audited(lambda: run(4))[1]
finally:
    for process in busy:
        process.kill()
        process.wait()
print(json.dumps({"estimate": estimate, "peaks": peaks, "overruns": overruns}))
"""


@pytest.mark.slow
@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity, as on Linux"
)
def test_reads_one_chunk_at_a_time_stay_within_their_charge_on_busy_cpus() -> None:
    # The reads of test_memory_costs's four-chunk whole read and of its read
    # one chunk at a time, on two CPUs that four other processes keep busy.
    # With the four chunks spread over Zarr's whole pool, from one in twenty
    # such reads on Python 3.13 to a half on 3.14 held up to a chunk more
    # than their charge, 5.7 MB against 4.9 MB; each read decodes on the one
    # thread or four that its ledger counts, so none passes its charge.
    traced = subprocess.run(
        [sys.executable, "-c", _LOADED_READS_SCRIPT, "40", "10"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert traced.returncode == 0, traced.stderr
    result = json.loads(traced.stdout.strip().splitlines()[-1])
    estimate = result["estimate"]
    over = [peak for peak in result["peaks"] if peak > estimate]
    assert over == [], (estimate, over)
    assert result["overruns"] == []
    # Each read holds its result of 3.2 MB and the chunk it decodes.
    assert min(result["peaks"]) > 3_200_000 + 800_000
