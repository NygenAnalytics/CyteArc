"""Each stage record measures what the stage used and what its process had.

The record tests build metrics by hand. The ledger tests run stages on an
in-memory store, and the store tests run the RNA recipe on a copy of the
PBMC fixture.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec
from zarr.storage import MemoryStore

import cytearc.utils.process as process_module
from cytearc import DataStore
from cytearc.datastore._pipeline import ledger as ledger_module
from cytearc.datastore._pipeline.events import PipelineEvent, PipelineEventEmitter
from cytearc.datastore._pipeline.ledger import RunLedger
from cytearc.storage.pipeline_runs import (
    PipelineStageMetrics,
    PipelineStageResources,
    PipelineStageTrace,
    create_pipeline_run_record,
    list_pipeline_run_records,
    load_pipeline_stage_records,
    pipeline_stage_path,
)
from tests.storage_helpers import recorded_codec_pools

# The metric fields that this release added, as a record of an earlier
# release lacks them.
_RESOURCE_FIELDS = (
    "cpuSeconds",
    "threadsPeak",
    "rssHighWaterStartBytes",
    "rssHighWaterEndBytes",
    "networkRxBytes",
    "networkTxBytes",
    "resources",
    "trace",
    "unavailable",
)


def _resources(**changes: Any) -> PipelineStageResources:
    values: dict[str, Any] = {
        "workers": 8,
        "memory_budget_bytes": 2**33,
        "cpus_available": 8,
        "cpu_quota_cores": 7.5,
        "memory_limit_bytes": 2**35,
    }
    return PipelineStageResources(**{**values, **changes})


def _trace(**changes: Any) -> PipelineStageTrace:
    values: dict[str, Any] = {
        "seconds": (0.0, 0.7, 1.4),
        "cpu_seconds": (0.0, 1.0, 2.25),
        "rss_bytes": (100, 140, None),
        "threads": (4, 9, 6),
        "network_rx_bytes": (0, 1_000, 3_000),
        "network_tx_bytes": (0, 10, 30),
    }
    return PipelineStageTrace(**{**values, **changes})


def _metrics(**changes: Any) -> PipelineStageMetrics:
    values: dict[str, Any] = {
        "wall_seconds": 1.5,
        "rss_baseline_bytes": 100,
        "rss_peak_bytes": 140,
        "rss_incremental_peak_bytes": 40,
        "sample_interval_seconds": 0.1,
        "sample_count": 3,
        "sampling_error_count": 1,
        "rss_unavailable_reason": None,
        "cpu_seconds": 2.25,
        "threads_peak": 9,
        "rss_high_water_start_bytes": 1_000,
        "rss_high_water_end_bytes": 5_000,
        "network_rx_bytes": 3_000,
        "network_tx_bytes": 30,
        "resources": _resources(),
        "trace": _trace(),
        "unavailable": {},
    }
    return PipelineStageMetrics(**{**values, **changes})


def test_stage_metrics_round_trip_every_resource_field() -> None:
    metrics = _metrics()

    stored = json.loads(json.dumps(metrics.to_dict()))

    assert set(_RESOURCE_FIELDS) <= set(stored)
    assert stored["resources"] == {
        "workers": 8,
        "memoryBudgetBytes": 2**33,
        "cpusAvailable": 8,
        "cpuQuotaCores": 7.5,
        "memoryLimitBytes": 2**35,
    }
    assert stored["trace"]["rssBytes"] == [100, 140, None]
    assert PipelineStageMetrics.from_dict(stored) == metrics


def test_metrics_of_an_earlier_release_read_without_resource_use() -> None:
    stored = _metrics().to_dict()
    for name in _RESOURCE_FIELDS:
        del stored[name]

    earlier = PipelineStageMetrics.from_dict(stored)

    assert earlier.cpu_seconds is None
    assert (earlier.resources, earlier.trace, earlier.unavailable) == (None, None, None)
    assert (earlier.threads_peak, earlier.network_rx_bytes) == (None, None)
    # Written again, such metrics name every field, each null, and read back.
    again = earlier.to_dict()
    assert {name: again[name] for name in _RESOURCE_FIELDS} == dict.fromkeys(
        _RESOURCE_FIELDS
    )
    assert PipelineStageMetrics.from_dict(again) == earlier


def test_metrics_hold_all_resource_fields_or_none() -> None:
    stored = _metrics().to_dict()
    del stored["trace"]
    with pytest.raises(
        ValueError, match=r"do not match the contract: missing \['trace'\]"
    ):
        PipelineStageMetrics.from_dict(stored)

    stored = _metrics().to_dict()
    stored["gpuSeconds"] = 1.0
    with pytest.raises(ValueError, match=r"extra \['gpuSeconds'\]"):
        PipelineStageMetrics.from_dict(stored)


_LONG_TRACE = {
    "seconds": tuple(0.1 * point for point in range(65)),
    "cpu_seconds": tuple(0.1 * point for point in range(65)),
    "rss_bytes": None,
    "threads": None,
    "network_rx_bytes": None,
    "network_tx_bytes": None,
}


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _metrics(cpu_seconds=-1.0), "cpu_seconds must be finite"),
        (lambda: _trace(**_LONG_TRACE), "holds 1 to 64 points, not 65"),
        (lambda: _trace(seconds=(0.0, 1.0, 0.5)), "seconds must never decrease"),
        (lambda: _trace(cpu_seconds=(0.0, 2.5, 2.25)), "never decrease"),
        (lambda: _trace(seconds=(0.5, 0.7, 1.4)), "must start at 0"),
        (lambda: _trace(network_tx_bytes=(0, 10, -1)), "non-negative integer"),
        (lambda: _trace(threads=(4, 9)), "trace threads holds 2 points"),
        (lambda: _resources(workers=0), "workers must be a positive integer"),
        (lambda: _resources(cpu_quota_cores=-0.5), "finite and non-negative"),
        (
            lambda: _metrics(unavailable={"threadsPeak": "no /proc"}),
            "threadsPeak must be null exactly where unavailable names it",
        ),
        (
            lambda: _metrics(threads_peak=None, trace=_trace(threads=None)),
            "threadsPeak must be null exactly where unavailable names it",
        ),
        (
            lambda: _metrics(
                resources=_resources(memory_limit_bytes=None),
            ),
            "memoryLimitBytes must be null exactly where unavailable names it",
        ),
        (
            lambda: _metrics(unavailable={"gpuSeconds": "no GPU"}),
            "names 'gpuSeconds', which is not one of",
        ),
        (lambda: _metrics(cpu_seconds=3.0), "count the stage's CPU seconds"),
        (lambda: _metrics(network_rx_bytes=2_999), "count the stage's network_rx"),
        (lambda: _metrics(threads_peak=10), "largest thread count of the trace"),
        (lambda: _metrics(rss_peak_bytes=150), "largest RSS of the trace"),
        (lambda: _metrics(sample_count=0), "holds RSS exactly when RSS was sampled"),
        (lambda: _metrics(cpu_seconds=None), "recorded together, with cpu_seconds"),
        (lambda: _metrics(resources=None), "resources must be PipelineStageResources"),
    ],
)
def test_stage_metrics_reject_inconsistent_resource_use(build, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build()


def _null_trace_threads(stored: dict[str, Any]) -> None:
    stored["trace"]["threads"] = None


def _null_trace_network(stored: dict[str, Any]) -> None:
    stored["trace"]["networkRxBytes"] = None


def _text_trace_seconds(stored: dict[str, Any]) -> None:
    stored["trace"]["seconds"] = "0.0 0.7 1.4"


@pytest.mark.parametrize(
    ("edit", "error", "message"),
    [
        (_text_trace_seconds, TypeError, "^trace seconds must be a sequence$"),
        (
            lambda stored: stored.update(trace=None),
            TypeError,
            "^metrics trace must be PipelineStageTrace$",
        ),
        (
            lambda stored: stored.update(unavailable={"threadsPeak": "x" * 513}),
            ValueError,
            "^metrics unavailable threadsPeak exceeds the record limit$",
        ),
        (
            _null_trace_threads,
            ValueError,
            "^The trace holds threads exactly when threads_peak exists$",
        ),
        (
            _null_trace_network,
            ValueError,
            "^The trace holds network_rx_bytes exactly when the stage does$",
        ),
    ],
    ids=["text-seconds", "no-trace", "long-reason", "no-threads", "no-network"],
)
def test_a_stage_record_edited_into_contradicting_metrics_is_refused(
    edit, error, message
) -> None:
    root, ledger = _ledger("work")
    ledger.run("work", lambda: ())
    stage = root[pipeline_stage_path(ledger.run_id, 0)]
    stored = json.loads(json.dumps(dict(stage.attrs)))
    stored["metrics"] = json.loads(json.dumps(_metrics().to_dict()))
    stage.attrs.put(stored)
    # The record reads before the edit.
    (record,) = load_pipeline_stage_records(root, ledger.run_id)
    assert record.metrics == _metrics()

    edit(stored["metrics"])
    stage.attrs.put(stored)

    with pytest.raises(error, match=message):
        load_pipeline_stage_records(root, ledger.run_id)


def test_a_null_beside_its_reason_reads_and_a_value_beside_one_does_not() -> None:
    reason = "OS thread counts require the Linux /proc filesystem"
    metrics = _metrics(
        threads_peak=None,
        trace=_trace(threads=None),
        unavailable={"threadsPeak": reason},
    )
    stored = json.loads(json.dumps(metrics.to_dict()))
    assert PipelineStageMetrics.from_dict(stored).unavailable == {"threadsPeak": reason}

    stored["threadsPeak"] = 9
    with pytest.raises(ValueError, match="null exactly where unavailable names it"):
        PipelineStageMetrics.from_dict(stored)


def _ledger(
    *stages: str, workers: int = 2, memory: int = 2**30
) -> tuple[zarr.Group, RunLedger]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = create_pipeline_run_record(
        root,
        recipe="rna",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=stages,
        cytearc_version="1.0.0",
    )
    events: list[PipelineEvent] = []
    emitter = PipelineEventEmitter(
        [events.append],
        run_id=run.run_id,
        recipe="rna",
        label=None,
        stage_order=stages,
    )
    ledger = RunLedger(
        root, run.run_id, emitter, workers=workers, memory_budget_bytes=memory
    )
    return root, ledger


def _metrics_of(root: zarr.Group, ledger: RunLedger) -> dict[str, PipelineStageMetrics]:
    records = load_pipeline_stage_records(root, ledger.run_id)
    found = {record.stage: record.metrics for record in records}
    assert all(metrics is not None for metrics in found.values())
    return {stage: metrics for stage, metrics in found.items() if metrics is not None}


def test_a_stage_records_at_least_the_cpu_time_its_threads_measured() -> None:
    root, ledger = _ledger("work")
    measured: list[float] = []
    data = b"\x00" * (32 * 2**20)

    def work() -> None:
        started = time.thread_time()
        # Hashing a large buffer releases the GIL, so both threads can run.
        for _ in range(6):
            hashlib.sha256(data).digest()
        measured.append(time.thread_time() - started)

    def action() -> tuple[()]:
        threads = [threading.Thread(target=work) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return ()

    ledger.run("work", action)

    metrics = _metrics_of(root, ledger)["work"]
    assert len(measured) == 2 and min(measured) > 0
    # The process clock holds both threads, whatever else loads the machine.
    assert metrics.cpu_seconds is not None
    assert metrics.cpu_seconds >= sum(measured) - 1e-6
    assert metrics.resources is not None
    assert (metrics.resources.workers, metrics.resources.memory_budget_bytes) == (
        2,
        2**30,
    )


def test_a_stage_that_sleeps_records_almost_no_cpu_time() -> None:
    root, ledger = _ledger("wait")

    def wait() -> tuple[()]:
        time.sleep(0.5)
        return ()

    ledger.run("wait", wait)

    metrics = _metrics_of(root, ledger)["wait"]
    assert metrics.wall_seconds >= 0.5
    assert metrics.cpu_seconds is not None and metrics.cpu_seconds < 0.1


def test_skipped_stages_record_their_resource_use() -> None:
    root, ledger = _ledger("off", workers=3, memory=2**31)

    ledger.skip("off")

    metrics = _metrics_of(root, ledger)["off"]
    assert metrics.cpu_seconds is not None and metrics.trace is not None
    assert metrics.resources is not None and metrics.resources.workers == 3
    assert metrics.resources.memory_budget_bytes == 2**31


def test_records_without_proc_keep_cpu_time_resources_and_the_cpu_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_module, "_PROC_ROOT", tmp_path / "missing")
    root, ledger = _ledger("work")

    def burn() -> tuple[()]:
        until = time.process_time() + 0.05
        while time.process_time() < until:
            pass
        return ()

    ledger.run("work", burn)

    metrics = _metrics_of(root, ledger)["work"]
    assert metrics.rss_peak_bytes is None and metrics.sample_count == 0
    assert metrics.rss_unavailable_reason == (
        "process-tree RSS requires the Linux /proc filesystem"
    )
    assert (metrics.threads_peak, metrics.network_rx_bytes) == (None, None)
    assert metrics.unavailable is not None
    assert {
        name: metrics.unavailable[name]
        for name in ("threadsPeak", "networkRxBytes", "networkTxBytes")
    } == {
        "threadsPeak": "OS thread counts require the Linux /proc filesystem",
        "networkRxBytes": "network counters require the Linux /proc filesystem",
        "networkTxBytes": "network counters require the Linux /proc filesystem",
    }
    assert metrics.cpu_seconds is not None and metrics.cpu_seconds >= 0.05
    assert metrics.resources is not None and metrics.resources.workers == 2
    trace = metrics.trace
    assert trace is not None and len(trace.seconds) >= 2
    assert (trace.rss_bytes, trace.threads, trace.network_rx_bytes) == (None,) * 3


def test_unknown_cpus_and_memory_are_recorded_with_their_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: None)
    monkeypatch.setattr(ledger_module, "detect_memory_limit_bytes", lambda: None)
    root, ledger = _ledger("work")

    ledger.run("work", lambda: ())

    metrics = _metrics_of(root, ledger)["work"]
    assert metrics.resources is not None and metrics.unavailable is not None
    assert (metrics.resources.cpus_available, metrics.resources.memory_limit_bytes) == (
        None,
        None,
    )
    assert metrics.unavailable["cpusAvailable"] == (
        "the logical CPUs that the process may use could not be determined"
    )
    assert metrics.unavailable["memoryLimitBytes"] == (
        "neither physical memory nor a cgroup memory limit could be read"
    )


# Small settings whose Leiden candidates run CyteArc's Numba-parallel Leiden.
_SMALL: dict[str, Any] = {
    "filtering": True,
    "cell_cycle": False,
    "hvg": {"top_n": 50},
    "pca": {"dims": 3},
    "neighbors": {"k": 3},
    "umap": False,
    "leiden": {"partitions": [0.5, 1.0]},
    "doublets": False,
    "markers": False,
}


def _store(source: str, tmp_path: Path, nthreads: int) -> DataStore:
    location = tmp_path / f"workers_{nthreads}.zarr"
    shutil.copytree(source, location)
    return DataStore(str(location), default_assay="RNA", nthreads=nthreads)


# The RNA recipe on a store of one worker, with the settings given as its last
# argument.
_ONE_WORKER_RUN = textwrap.dedent(
    """
    import json
    import sys

    from cytearc import DataStore

    store = DataStore(sys.argv[1], default_assay="RNA", nthreads=1)
    store.pipeline.rna(**json.loads(sys.argv[2]))
    """
)


def _cpu_trace(stage: str, metrics: PipelineStageMetrics) -> str:
    """The stage's CPU time and threads, and the cores it kept busy between samples.

    A record that holds CPU time also holds its resources and trace.
    """
    assert metrics.resources is not None and metrics.trace is not None
    trace = metrics.trace
    lines = [
        f"{stage}: {metrics.cpu_seconds} CPU seconds in {metrics.wall_seconds} "
        f"wall seconds, at most {metrics.threads_peak} threads, "
        f"{metrics.resources.cpus_available} CPUs available"
    ]
    threads = trace.threads or (None,) * len(trace.seconds)
    seconds = cpu = 0.0
    for at, used, sampled in zip(
        trace.seconds, trace.cpu_seconds, threads, strict=True
    ):
        cores = (used - cpu) / (at - seconds) if at > seconds else None
        busy = "" if cores is None else f", {cores:.2f} cores since the last"
        lines.append(f"  {at:.3f} s: {used:.3f} CPU seconds{busy}, {sampled} threads")
        seconds, cpu = at, used
    return "\n".join(lines)


def test_the_one_core_check_reports_the_cores_and_threads_of_each_sample() -> None:
    report = _cpu_trace("work", _metrics())

    # 1.0 and 1.25 CPU seconds in each 0.7 s between the hand-built samples.
    assert report.splitlines() == [
        "work: 2.25 CPU seconds in 1.5 wall seconds, at most 9 threads, "
        "8 CPUs available",
        "  0.000 s: 0.000 CPU seconds, 4 threads",
        "  0.700 s: 1.000 CPU seconds, 1.43 cores since the last, 9 threads",
        "  1.400 s: 2.250 CPU seconds, 1.79 cores since the last, 6 threads",
    ]
    # Without /proc, as on macOS, the samples count no threads.
    unsampled = _metrics(
        trace=_trace(threads=None),
        threads_peak=None,
        unavailable={
            "threadsPeak": "OS thread counts require the Linux /proc filesystem"
        },
    )
    assert _cpu_trace("work", unsampled).splitlines()[2:] == [
        "  0.700 s: 1.000 CPU seconds, 1.43 cores since the last, None threads",
        "  1.400 s: 2.250 CPU seconds, 1.79 cores since the last, None threads",
    ]


@pytest.mark.slow
def test_a_store_of_one_worker_records_it_and_stays_on_one_core(
    datastore_zarr_root: str, tmp_path: Path
) -> None:
    location = tmp_path / "workers_1.zarr"
    shutil.copytree(datastore_zarr_root, location)

    # A process of its own counts only the run's threads and children, none
    # that earlier tests in the same process left running.
    completed = subprocess.run(
        [sys.executable, "-c", _ONE_WORKER_RUN, str(location), json.dumps(_SMALL)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    store = DataStore(str(location), default_assay="RNA", nthreads=1)
    (run,) = list_pipeline_run_records(store.zw)
    records = {
        record.stage: record
        for record in load_pipeline_stage_records(store.zw, run.run_id)
    }
    workers = {
        stage: record.metrics.resources.workers
        for stage, record in records.items()
        if record.metrics is not None and record.metrics.resources is not None
    }
    assert workers == dict.fromkeys(records, 1)
    for stage in ("leiden",):
        metrics = records[stage].metrics
        assert metrics is not None and metrics.cpu_seconds is not None
        # A busy machine can only lower CPU time against wall time.
        assert metrics.cpu_seconds <= 1.1 * metrics.wall_seconds + 0.05, _cpu_trace(
            stage, metrics
        )


@pytest.mark.slow
def test_a_one_worker_stage_encodes_and_decodes_on_one_thread(
    datastore_zarr_root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(datastore_zarr_root, tmp_path, nthreads=1)
    stage_run = RunLedger.run

    with recorded_codec_pools(host_threads=4) as pools:

        def labeled(ledger: RunLedger, stage: str, action: Any) -> Any:
            pools.label = stage
            try:
                return stage_run(ledger, stage, action)
            finally:
                pools.label = None

        monkeypatch.setattr(RunLedger, "run", labeled)
        store.pipeline.rna(**_SMALL)

    in_stages = [(stage, threads) for stage, threads in pools.calls if stage]
    # The encodes and decodes of each stage, the graph load of Leiden among
    # them, run on a pool of one thread, not on the four host threads.
    assert {threads for _stage, threads in in_stages} == {1}, sorted(
        set(in_stages), key=str
    )
    assert {"filtering", "pca", "neighbors", "leiden"} <= {
        stage for stage, _threads in in_stages
    }


@pytest.mark.parametrize("workers", [1, 3])
def test_a_stage_encodes_and_decodes_on_a_pool_of_its_workers_threads(
    tmp_path: Path, workers: int
) -> None:
    _root, ledger = _ledger("store", workers=workers)
    group = zarr.open_group(str(tmp_path / "values.zarr"), mode="w")
    values = np.arange(48_000, dtype=np.int64)

    with recorded_codec_pools(host_threads=7) as pools:

        def store_and_read() -> tuple[()]:
            pools.label = "stage"
            group.create_array(
                "values", data=values, chunks=(1_000,), compressors=ZstdCodec()
            )
            np.testing.assert_array_equal(group["values"][:], values)
            pools.label = None
            return ()

        ledger.run("store", store_and_read)
        pools.label = "after"
        np.testing.assert_array_equal(group["values"][:], values)

    # Forty-eight chunks encode and decode on the stage's workers, and the same
    # read after the stage decodes on the host's seven threads.
    assert pools.threads("stage") == [workers] * 96
    assert pools.threads("after") == [7] * 48


def test_a_store_of_four_workers_records_them_and_the_report_shows_resource_use(
    datastore_zarr_root: str, tmp_path: Path
) -> None:
    store = _store(datastore_zarr_root, tmp_path, nthreads=4)
    run = store.pipeline.rna(**_SMALL)
    records = load_pipeline_stage_records(store.zw, run.run_id)
    assert {
        record.metrics.resources.workers
        for record in records
        if record.metrics is not None and record.metrics.resources is not None
    } == {4}

    # The record of a stage that an earlier release wrote has no CPU time.
    leiden = next(record for record in records if record.stage == "leiden")
    group = store.zw[pipeline_stage_path(run.run_id, leiden.ordinal)]
    attrs = dict(group.attrs)
    attrs["metrics"] = {
        key: value
        for key, value in attrs["metrics"].items()
        if key not in _RESOURCE_FIELDS
    }
    group.attrs.put(attrs)

    markdown = run.report(format="markdown")
    lines = markdown.splitlines()
    header = next(line for line in lines if line.startswith("| # |"))
    assert header == (
        "| # | Stage | Status | Wall seconds | CPU seconds | Peak RSS bytes "
        "| Received bytes | Created | Reused | Warnings |"
    )
    rows = {
        cells[2]: cells
        for line in lines
        if line.startswith("| ") and not line.startswith(("| # |", "|---"))
        for cells in [[cell.strip() for cell in line.split("|")]]
    }
    pca = rows["pca"]
    metrics = next(record.metrics for record in records if record.stage == "pca")
    assert metrics is not None and metrics.cpu_seconds is not None
    assert pca[5] == f"{metrics.cpu_seconds:.3f}"
    assert pca[7] == (
        "unavailable"
        if metrics.network_rx_bytes is None
        else str(metrics.network_rx_bytes)
    )
    # The stage without CPU time and network bytes leaves both cells empty,
    # and its dict report holds each new value as None.
    assert (rows["leiden"][5], rows["leiden"][7]) == ("", "")
    report = run.report()
    stage = next(item for item in report["stages"] if item["stage"] == "leiden")
    assert {name: stage["metrics"][name] for name in _RESOURCE_FIELDS} == dict.fromkeys(
        _RESOURCE_FIELDS
    )
    assert stage["metrics"]["wallSeconds"] == leiden.metrics.wall_seconds
