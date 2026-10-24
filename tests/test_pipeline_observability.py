"""Pipeline runs report their events, progress, warnings, and diagnostics.

A run also stops at its next safe checkpoint once its token is cancelled.
"""

import contextvars
import dataclasses
import inspect
import json
import logging
import pickle
import shutil
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc.utils.logging as cytearc_logging
from cytearc import (
    CancellationToken,
    DataStore,
    PipelineCancelled,
    configure_output,
    pipeline_listener,
)
from cytearc.datastore._pipeline import events as pipeline_events
from cytearc.datastore._pipeline import executor as pipeline_executor
from cytearc.datastore._pipeline.events import PipelineEventEmitter
from cytearc.datastore.pipeline_accessor import (
    PipelineCallback,
    PipelineEvent,
    PipelineProgress,
)
from cytearc.storage.pipeline_runs import (
    STAGE_WARNING_LIMIT,
    PipelineErrorRecord,
    PipelineStageMetrics,
    PipelineStageRecord,
    bounded_record_text,
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_stage_record,
    load_pipeline_stage_records,
    pipeline_stage_path,
    start_pipeline_stage_record,
)
from cytearc.utils.logging import (
    RUN_ID_EXTRA,
    STAGE_EXTRA,
    logger,
    progress_enabled,
    set_verbosity,
)
from cytearc.utils.progress import (
    iter_progress,
    observe_progress,
    progress_requested,
    tqdmbar,
)
from cytearc.utils.shutdown import (
    ShutdownRequested,
    ShutdownToken,
    TemporarySignalGuard,
    cancellation_scope,
    current_shutdown_token,
    shutdown_scope,
)
from cytearc.utils.warnings import warn

_LABEL = "observed"
_PARTITIONS = ("leiden_0.5", "leiden_1.0")
# Small settings that still filter cells and choose between two Leiden
# candidates, so every summarized stage of the recipe runs.
_OPTIONS: dict[str, Any] = {
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
_SKIPPED = frozenset(
    {
        "cell_calling",
        "hashing",
        "cell_cycle",
        "harmony",
        "embedding_initialization",
        "umap",
        "paris",
        "membership_strength",
        "tsne",
        "doublets",
        "markers",
    }
)
_EVENT_KEYS = {
    "kind",
    "runId",
    "recipe",
    "label",
    "stage",
    "ordinal",
    "stages",
    "timeNs",
    "progress",
    "metrics",
    "summary",
    "outputs",
    "warnings",
    "error",
}
# Some tests start a thread and expect it to begin with an empty context, as
# new threads do on the GIL builds that CI uses. Where a new thread inherits
# its creator's context, the free-threaded default since Python 3.14, the
# thread shares the listeners and warning collection of the code that started
# it, so those tests do not apply.
_THREADS_INHERIT_CONTEXT = bool(getattr(sys.flags, "thread_inherit_context", 0))
_INHERITED_CONTEXT = "new threads inherit their creator's context"


def _open_copy(source: str | Path, destination: Path) -> DataStore:
    shutil.copytree(source, destination)
    return DataStore(str(destination), default_assay="RNA")


@pytest.fixture(scope="module")
def observed_base(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Run the recipe once, on a private copy of the PBMC store without artifacts.

    Tests read this run's events and records, or copy the store so that their
    own runs reuse its artifacts.
    """
    location = tmp_path_factory.mktemp("pipeline_observability") / "data.zarr"
    store = _open_copy(datastore_zarr_root, location)
    events: list[PipelineEvent] = []
    # The run's durable status when its run_started and run_completed arrive.
    statuses: dict[str, str] = {}

    def observe(event: PipelineEvent) -> None:
        events.append(event)
        if event.kind in {"run_started", "run_completed"}:
            statuses[event.kind] = store.pipeline.open(run_id=event.run_id).status

    run = store.pipeline.rna(label=_LABEL, callback=observe, **_OPTIONS)
    return SimpleNamespace(
        location=location,
        store=store,
        run=run,
        report=run.report(),
        events=tuple(events),
        statuses=statuses,
    )


@pytest.fixture
def observed_store(observed_base: SimpleNamespace, tmp_path: Path) -> DataStore:
    """A writable copy of the base store, whose artifacts a run reuses."""
    return _open_copy(observed_base.location, tmp_path / "data.zarr")


@pytest.fixture
def restored_output() -> Iterator[None]:
    """Put CyteArc's output settings and its sink back as the test found them.

    The session fixture in conftest removes every handler, so CyteArc's sink
    is usually absent; only the sink that this test installs is removed.
    """
    saved = dataclasses.replace(cytearc_logging._config)
    handler = cytearc_logging._handler_id
    installed = handler is not None and handler in logger._core.handlers
    yield
    cytearc_logging._remove_cytearc_handler()
    for field in dataclasses.fields(saved):
        setattr(cytearc_logging._config, field.name, getattr(saved, field.name))
    if installed:
        cytearc_logging._install_cytearc_handler()
    else:
        cytearc_logging._handler_id = handler


def _assert_stage_sequence(
    events: Sequence[PipelineEvent],
    stage_order: Sequence[str],
) -> None:
    """Check that a completed run's events follow its stage order.

    Every stage either starts, reports progress, and completes, or is skipped
    without starting, and each stage event carries the stage's ordinal.
    """
    assert events[0].kind == "run_started"
    assert events[-1].kind == "run_completed"
    running: str | None = None
    visited: list[str] = []
    for event in events[1:-1]:
        assert event.stage is not None, event
        assert event.ordinal == stage_order.index(event.stage), event
        if event.kind == "stage_started":
            assert running is None, event
            running = event.stage
            visited.append(event.stage)
        elif event.kind == "stage_progress":
            assert event.stage == running, event
        elif event.kind == "stage_completed":
            assert event.stage == running, event
            running = None
        else:
            assert event.kind == "stage_skipped", event
            assert running is None, event
            visited.append(event.stage)
    assert running is None
    assert visited == list(stage_order)


def test_run_events_follow_the_recorded_stages(observed_base: SimpleNamespace) -> None:
    events = observed_base.events
    run = observed_base.run
    stage_order = observed_base.report["run"]["stageOrder"]

    _assert_stage_sequence(events, stage_order)
    assert {(event.run_id, event.recipe, event.label) for event in events} == {
        (run.run_id, "rna", _LABEL)
    }
    assert {event.stages for event in events} == {len(stage_order)}
    assert all(isinstance(event.time_ns, int) and event.time_ns > 0 for event in events)
    # The run record exists when the run starts and is complete when it ends.
    assert observed_base.statuses == {
        "run_started": "running",
        "run_completed": "completed",
    }
    for event in (events[0], events[-1]):
        assert (event.stage, event.ordinal) == (None, None)
        assert (event.metrics, event.summary, event.progress) == (None, None, None)
        assert (event.outputs, event.reused, event.warnings) == ((), (), ())
        assert event.error is None

    terminal = {
        event.stage: event
        for event in events
        if event.kind in {"stage_completed", "stage_skipped"}
    }
    assert list(terminal) == stage_order
    for record in observed_base.report["stages"]:
        event = terminal[record["stage"]]
        expected_kind = (
            "stage_skipped" if record["status"] == "skipped" else "stage_completed"
        )
        assert event.kind == expected_kind
        assert event.ordinal == record["ordinal"]
        assert event.metrics == record["metrics"]
        assert event.summary == record["summary"]
        assert list(event.warnings) == record["warnings"]
        assert [(key, ref.to_dict()) for key, ref in event.outputs] == [
            (output["outputKey"], output["artifact"]) for output in record["outputs"]
        ]
        assert list(event.reused) == [
            output["outputKey"] for output in record["outputs"] if output["reused"]
        ]
        assert (event.progress, event.error) == (None, None)
    skipped = {
        stage for stage, event in terminal.items() if event.kind != "stage_completed"
    }
    assert skipped == _SKIPPED
    assert all(
        (event.outputs, event.summary) == ((), None)
        for stage, event in terminal.items()
        if stage in skipped
    )
    # The store held no artifacts, so the run computed every output.
    assert all(event.reused == () for event in terminal.values())
    assert terminal["pca"].outputs == (("pca", run["pca"]),)
    assert terminal["pca"].metrics["wallSeconds"] > 0


def test_run_events_convert_to_json_and_pickle(observed_base: SimpleNamespace) -> None:
    events = observed_base.events
    converted = json.loads(json.dumps([event.to_dict() for event in events]))

    assert all(set(item) == _EVENT_KEYS for item in converted)
    assert [(item["kind"], item["stage"]) for item in converted] == [
        (event.kind, event.stage) for event in events
    ]
    pca = next(
        index
        for index, event in enumerate(events)
        if (event.kind, event.stage) == ("stage_completed", "pca")
    )
    assert converted[pca]["runId"] == observed_base.run.run_id
    assert converted[pca]["label"] == _LABEL
    assert converted[pca]["timeNs"] == events[pca].time_ns
    assert converted[pca]["outputs"] == [
        {
            "key": "pca",
            "artifact": observed_base.run["pca"].to_dict(),
            "reused": False,
        }
    ]
    assert converted[pca]["summary"] == events[pca].summary
    assert converted[pca]["metrics"] == events[pca].metrics
    progress = next(item for item in converted if item["kind"] == "stage_progress")
    assert set(progress["progress"]) == {"description", "completed", "total"}

    # Events keep the documented import path, which their pickles name.
    assert PipelineEvent.__module__ == "cytearc.datastore.pipeline_accessor"
    assert PipelineProgress.__module__ == "cytearc.datastore.pipeline_accessor"
    assert pickle.loads(pickle.dumps(events)) == events


def test_an_event_error_converts_to_its_type_and_bounded_message() -> None:
    event = PipelineEvent(
        kind="stage_failed",
        run_id="0" * 64,
        recipe="rna",
        label=None,
        stage="pca",
        ordinal=7,
        stages=21,
        time_ns=1,
        progress=PipelineProgress(description="Fitting PCA", completed=2, total=4),
        error=ValueError("x" * 600),
    )

    converted = event.to_dict()

    assert converted["error"] == {
        "type": "builtins.ValueError",
        "message": bounded_record_text("x" * 600),
    }
    assert len(converted["error"]["message"]) == 512
    assert converted["progress"] == {
        "description": "Fitting PCA",
        "completed": 2,
        "total": 4,
    }
    restored = pickle.loads(pickle.dumps(event))
    assert (restored.kind, restored.progress) == (event.kind, event.progress)
    assert type(restored.error) is ValueError
    assert str(restored.error) == "x" * 600


def test_stage_summaries_describe_the_completed_stages(
    observed_base: SimpleNamespace,
) -> None:
    run = observed_base.run
    summaries = {
        stage["stage"]: stage["summary"] for stage in observed_base.report["stages"]
    }

    snapshot = summaries["input_snapshot"]
    filtering = summaries["filtering"]
    assert filtering["inputCells"] == snapshot["input_cell_selection"]["cells"]
    assert 0 < filtering["cells"] < filtering["inputCells"]
    assert filtering["removedCells"] == filtering["inputCells"] - filtering["cells"]
    # Without a sample column, MAD bounds come from one sample.
    assert filtering["samples"] == 1
    assert summaries["normalization"]["cells"] == filtering["cells"]

    pca = summaries["pca"]
    assert pca["cells"] == filtering["cells"]
    assert pca["dims"] == _OPTIONS["pca"]["dims"]
    assert len(pca["varianceRatio"]) == pca["dims"]
    assert all(0 < ratio < 1 for ratio in pca["varianceRatio"])
    assert pca["varianceExplained"] == pytest.approx(
        sum(pca["varianceRatio"]), abs=1e-3
    )

    for key in _PARTITIONS:
        labels, counts = np.unique(run.cells.fetch(key), return_counts=True)
        assert summaries[key] == {
            "clusters": len(labels),
            "largestCluster": int(counts.max()),
            "smallestCluster": int(counts.min()),
        }

    selection = summaries["cluster_selection"]
    assert set(selection["scores"]) == set(_PARTITIONS)
    assert selection["selected"] in _PARTITIONS
    assert run["clusters"] == run[selection["selected"]]
    assert all(summaries[stage] is None for stage in _SKIPPED)


def test_a_run_records_its_signal_protection(observed_base: SimpleNamespace) -> None:
    # The fixture's run ran on the main thread, where a run installs its
    # signal handlers.
    protection = {"signalGuardAvailable": True, "unavailableReason": None}

    assert observed_base.report["summary"]["signalProtection"] == protection
    assert observed_base.report["run"]["config"]["shutdown"] == protection


def test_listeners_receive_the_runs_started_in_their_block(
    observed_store: DataStore,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def recorder(name: str) -> PipelineCallback:
        def record(event: PipelineEvent) -> None:
            calls.append((name, event.run_id, event.kind))

        return record

    callback, outer, inner = recorder("callback"), recorder("outer"), recorder("inner")
    with pipeline_listener(outer):
        with pipeline_listener(inner) as registered:
            assert registered is inner
            nested = observed_store.pipeline.rna(callback=callback, **_OPTIONS)
        outer_only = observed_store.pipeline.rna(callback=callback, **_OPTIONS)
    after = observed_store.pipeline.rna(callback=callback, **_OPTIONS)

    for run, receivers in (
        (nested, ["callback", "outer", "inner"]),
        (outer_only, ["callback", "outer"]),
        (after, ["callback"]),
    ):
        names = [name for name, run_id, _kind in calls if run_id == run.run_id]
        kinds = {
            receiver: [
                kind
                for name, run_id, kind in calls
                if run_id == run.run_id and name == receiver
            ]
            for receiver in receivers
        }
        assert kinds["callback"][0] == "run_started"
        assert kinds["callback"][-1] == "run_completed"
        # Each event reaches the run's callback first, then the listeners in
        # the order they were registered.
        assert names == receivers * len(kinds["callback"])
        assert all(kinds[receiver] == kinds["callback"] for receiver in receivers)


def test_listener_failures_are_logged_and_the_run_completes(
    observed_store: DataStore,
) -> None:
    class ListenerExit(BaseException):
        """A BaseException that no pipeline code expects."""

    failures: list[tuple[str, type[BaseException]]] = []
    received: list[PipelineEvent] = []

    def failing(event: PipelineEvent) -> None:
        if event.kind == "run_started":
            raise ListenerExit("listener exit")
        if (event.kind, event.stage) == ("stage_started", "pca"):
            raise KeyboardInterrupt("listener interrupt")
        if event.kind == "run_completed":
            raise RuntimeError("listener error")

    def capture(message: Any) -> None:
        record = message.record
        if record["message"].startswith("Pipeline callback failed"):
            failures.append((record["message"], record["exception"].type))

    sink = logger.add(capture, level="ERROR", format="{message}")
    try:
        with pipeline_listener(failing), pipeline_listener(received.append):
            run = observed_store.pipeline.rna(**_OPTIONS)
    except KeyboardInterrupt as error:
        # An escaped interrupt would otherwise stop the whole test session.
        pytest.fail(f"A listener's KeyboardInterrupt escaped the run: {error!r}")
    finally:
        logger.remove(sink)

    assert run.status == "completed"
    # A failing listener stops neither the run nor the listeners after it.
    assert received[0].kind == "run_started"
    assert received[-1].kind == "run_completed"
    _assert_stage_sequence(received, run.report()["run"]["stageOrder"])
    assert [error for _message, error in failures] == [
        ListenerExit,
        KeyboardInterrupt,
        RuntimeError,
    ]
    for (message, _error), kind in zip(
        failures, ("run_started", "stage_started", "run_completed"), strict=True
    ):
        assert kind in message


@pytest.mark.skipif(_THREADS_INHERIT_CONTEXT, reason=_INHERITED_CONTEXT)
def test_a_listener_misses_runs_on_another_thread(observed_store: DataStore) -> None:
    leaked: list[PipelineEvent] = []
    own: list[PipelineEvent] = []
    outcome: dict[str, Any] = {}

    def run_on_worker() -> None:
        try:
            outcome["run"] = observed_store.pipeline.rna(
                callback=own.append, **_OPTIONS
            )
        except BaseException as error:
            outcome["error"] = error

    with pipeline_listener(leaked.append):
        worker = threading.Thread(target=run_on_worker)
        worker.start()
        worker.join()

    if "error" in outcome:
        raise outcome["error"]
    assert outcome["run"].status == "completed"
    assert [own[0].kind, own[-1].kind] == ["run_started", "run_completed"]
    assert leaked == []
    # Signal handlers belong to the main thread, so the run records that it
    # ran without them.
    assert outcome["run"].report()["summary"]["signalProtection"] == {
        "signalGuardAvailable": False,
        "unavailableReason": "signal handlers require the main thread",
    }


def test_listeners_and_callbacks_must_be_callable(observed_store: DataStore) -> None:
    runs = [run.run_id for run in observed_store.pipeline.list_runs(limit=100)]

    with pytest.raises(TypeError, match="^listener must be callable$"):
        with pipeline_listener("not callable"):
            pass
    with pytest.raises(TypeError, match="^callback must be callable$"):
        observed_store.pipeline.rna(callback="not callable", **_OPTIONS)

    assert [run.run_id for run in observed_store.pipeline.list_runs(limit=100)] == runs


_BAR = "Observability bar"
_BAR_UNITS = 100
_ITERATION = "Observability iteration"
_ITERATION_UNITS = 50


def test_progress_bars_become_throttled_progress_events(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = DataStore._graph_connectivity
    elapsed: list[float] = []

    def connectivity_with_progress(self: DataStore, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        with tqdmbar(total=_BAR_UNITS, desc=_BAR) as bar:
            for _unit in range(_BAR_UNITS):
                time.sleep(0.004)
                bar.update(1)
        for _item in iter_progress(
            range(_ITERATION_UNITS), desc=_ITERATION, total=_ITERATION_UNITS
        ):
            pass
        elapsed.append(time.perf_counter() - started)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataStore, "_graph_connectivity", connectivity_with_progress)
    previous = progress_enabled()
    events: list[PipelineEvent] = []
    configure_output(progress=False)
    try:
        run = observed_store.pipeline.rna(callback=events.append, **_OPTIONS)
    finally:
        configure_output(progress=previous)

    stage_order = run.report()["run"]["stageOrder"]
    _assert_stage_sequence(events, stage_order)
    (seconds,) = elapsed
    for description, units in ((_BAR, _BAR_UNITS), (_ITERATION, _ITERATION_UNITS)):
        progress = [
            event
            for event in events
            if event.progress is not None and event.progress.description == description
        ]
        assert {(event.kind, event.stage, event.ordinal) for event in progress} == {
            ("stage_progress", "connectivity", stage_order.index("connectivity"))
        }
        completed = [event.progress.completed for event in progress]
        assert {event.progress.total for event in progress} == {units}
        # The first update and the finished bar always arrive, in order; the
        # updates between them arrive at most once per interval.
        assert completed[0] == 1
        assert completed[-1] == units
        assert completed == sorted(set(completed))
        assert len(progress) <= 2 + seconds / pipeline_events.PROGRESS_INTERVAL_SECONDS
        assert len(progress) < units
    # Progress is observed while bars stay hidden.
    assert _BAR not in capsys.readouterr().err


def test_progress_events_are_throttled_for_each_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[PipelineEvent] = []
    emitter = PipelineEventEmitter(
        [events.append],
        run_id="0" * 64,
        recipe="rna",
        label=None,
        stage_order=("pca",),
    )
    monkeypatch.setattr(pipeline_events, "PROGRESS_INTERVAL_SECONDS", 3600.0)
    for completed in range(1, 11):
        emitter.progress("pca", 0, 1, "Fitting PCA", completed, 10)
        emitter.progress("pca", 0, 2, "Writing", completed, None)
    # Updates past a bar's total are throttled like any other.
    for completed in range(11, 14):
        emitter.progress("pca", 0, 1, "Fitting PCA", completed, 10)

    # Each bar reports its first update and the update that finishes it; a
    # bar without a total never finishes.
    assert [
        (event.kind, event.stage, event.progress.description, event.progress.completed)
        for event in events
    ] == [
        ("stage_progress", "pca", "Fitting PCA", 1),
        ("stage_progress", "pca", "Writing", 1),
        ("stage_progress", "pca", "Fitting PCA", 10),
    ]

    # A second bar that shares a description has a throttle of its own.
    events.clear()
    emitter.progress("pca", 0, 3, "Fitting PCA", 1, 10)
    assert [event.progress.completed for event in events] == [1]

    events.clear()
    monkeypatch.setattr(pipeline_events, "PROGRESS_INTERVAL_SECONDS", 0.0)
    for completed in range(11, 14):
        emitter.progress("pca", 0, 2, "Writing", completed, None)
    assert [event.progress.completed for event in events] == [11, 12, 13]


_STAGE_BAR = "Observability stage bar"
_WORKER_BAR = "Observability worker bar"


def test_progress_from_another_thread_never_reaches_the_callbacks(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = DataStore._graph_connectivity
    observed_on_worker: list[bool] = []

    def connectivity_with_a_worker_bar(
        self: DataStore, *args: Any, **kwargs: Any
    ) -> Any:
        with tqdmbar(total=2, desc=_STAGE_BAR) as bar:
            bar.update(2)

        def advance_elsewhere() -> None:
            # A worker that shares the stage's context, as compute workers
            # do, so its bar is observed too.
            observed_on_worker.append(progress_requested())
            with tqdmbar(total=3, desc=_WORKER_BAR) as bar:
                bar.update(3)

        worker = threading.Thread(
            target=contextvars.copy_context().run, args=(advance_elsewhere,)
        )
        worker.start()
        worker.join()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        DataStore, "_graph_connectivity", connectivity_with_a_worker_bar
    )
    events: list[PipelineEvent] = []
    threads: set[int] = set()

    def record(event: PipelineEvent) -> None:
        events.append(event)
        threads.add(threading.get_ident())

    previous = progress_enabled()
    configure_output(progress=False)
    try:
        run = observed_store.pipeline.rna(callback=record, **_OPTIONS)
    finally:
        configure_output(progress=previous)

    assert run.status == "completed"
    assert observed_on_worker == [True]
    # Every callback runs on the thread that runs the pipeline, so the
    # worker's progress is dropped, while the stage's own bar is reported.
    assert threads == {threading.get_ident()}
    descriptions = {
        event.progress.description for event in events if event.progress is not None
    }
    assert _STAGE_BAR in descriptions
    assert _WORKER_BAR not in descriptions


_ISSUED = "observability check: a warning issued during PCA"
_LOGGED = "observability check: a warning logged during PCA"
_ELSEWHERE = "observability check: a warning from another thread"


@pytest.mark.skipif(_THREADS_INHERIT_CONTEXT, reason=_INHERITED_CONTEXT)
def test_stage_warnings_reach_the_record_event_and_report(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = DataStore._reduction_pca

    def warning_pca(self: DataStore, *args: Any, **kwargs: Any) -> Any:
        warn(_ISSUED)
        logger.warning(_LOGGED)

        def elsewhere() -> None:
            warn(_ELSEWHERE)
            logger.warning(_ELSEWHERE)

        thread = threading.Thread(target=elsewhere)
        thread.start()
        thread.join()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataStore, "_reduction_pca", warning_pca)
    events: list[PipelineEvent] = []
    with pytest.warns(UserWarning) as issued:
        run = observed_store.pipeline.rna(callback=events.append, **_OPTIONS)

    # Collecting a warning leaves its display alone.
    assert _ISSUED in [str(warning.message) for warning in issued]
    report = run.report()
    stages = {stage["stage"]: stage for stage in report["stages"]}
    pca = stages["pca"]
    observed = [message for message in pca["warnings"] if "observability" in message]
    assert observed == [_ISSUED, _LOGGED]
    assert all(_ELSEWHERE not in stage["warnings"] for stage in report["stages"])
    completed = next(
        event
        for event in events
        if (event.kind, event.stage) == ("stage_completed", "pca")
    )
    assert list(completed.warnings) == pca["warnings"]

    assert report["summary"]["warnings"] == [
        {"stage": stage["stage"], "message": message}
        for stage in report["stages"]
        for message in stage["warnings"]
    ]
    assert {"stage": "pca", "message": _ISSUED} in report["summary"]["warnings"]
    assert {"stage": "pca", "message": _LOGGED} in report["summary"]["warnings"]

    markdown = run.report(format="markdown")
    row = next(
        line
        for line in markdown.splitlines()
        if line.startswith(f"| {pca['ordinal']} | pca |")
    )
    assert row.endswith(f"| {len(pca['warnings'])} |")
    diagnostics = markdown.split("\n## Diagnostics\n", 1)[1].split("\n## ", 1)[0]
    section = diagnostics.split("### pca\n", 1)[1].split("\n### ", 1)[0]
    assert json.dumps(pca["summary"], sort_keys=True) in section
    assert f"- Warning: {_ISSUED}" in section
    assert f"- Warning: {_LOGGED}" in section
    assert _ELSEWHERE not in markdown


class _Records(logging.Handler):
    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_the_stdlib_bridge_names_the_run_and_stage(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    restored_output: None,
) -> None:
    original = DataStore._reduction_pca

    def logging_pca(self: DataStore, *args: Any, **kwargs: Any) -> Any:
        logger.info("bridge check during PCA")
        logger.warning("bridge warning during PCA")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataStore, "_reduction_pca", logging_pca)
    bridged = logging.getLogger("cytearc")
    handler = _Records()
    level = bridged.level
    bridged.addHandler(handler)
    bridged.setLevel(logging.INFO)
    try:
        # Another test in this process may have raised the level or turned
        # on timestamps; the fixture puts back whatever this test changes.
        configure_output(level="INFO", timestamps=False, stdlib=True)
        run = observed_store.pipeline.rna(**_OPTIONS)
        logger.info("bridge check after the run")
        configure_output(stdlib=False)
        logger.info("stdout check")
    finally:
        bridged.removeHandler(handler)
        bridged.setLevel(level)

    records = {record.getMessage(): record for record in handler.records}
    during = records["bridge check during PCA"]
    assert (during.levelno, during.cytearc_run_id, during.cytearc_stage) == (
        logging.INFO,
        run.run_id,
        "pca",
    )
    warning = records["bridge warning during PCA"]
    assert (warning.levelno, warning.cytearc_run_id, warning.cytearc_stage) == (
        logging.WARNING,
        run.run_id,
        "pca",
    )
    after = records["bridge check after the run"]
    assert (after.cytearc_run_id, after.cytearc_stage) == (None, None)
    stage_order = run.report()["run"]["stageOrder"]
    staged = [record for record in handler.records if record.cytearc_stage is not None]
    assert {record.cytearc_run_id for record in staged} == {run.run_id}
    assert {record.cytearc_stage for record in staged} <= set(stage_order)

    # Without the bridge, records go to stdout again and not to logging.
    assert "stdout check" not in records
    output = capsys.readouterr().out
    assert "INFO: stdout check" in output
    assert "bridge check" not in output


@contextmanager
def _bridged(level: int) -> Iterator[_Records]:
    """Capture every record that reaches the ``cytearc`` logger at ``level``."""
    bridged = logging.getLogger("cytearc")
    handler = _Records(level=logging.NOTSET)
    saved = bridged.level
    bridged.addHandler(handler)
    bridged.setLevel(level)
    try:
        yield handler
    finally:
        bridged.removeHandler(handler)
        bridged.setLevel(saved)


def _next_line() -> int:
    """Return the line after the caller's, where the caller logs next."""
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    return frame.f_back.f_lineno + 1


def _log_one_of_each() -> dict[str, int]:
    """Log one record of each kind and return the line that logged each."""
    lines = {"success": _next_line()}
    logger.success("bridged success")
    lines["trace"] = _next_line()
    logger.trace("bridged trace")
    with logger.contextualize(**{RUN_ID_EXTRA: "a" * 64, STAGE_EXTRA: "pca"}):
        try:
            raise ZeroDivisionError("bridged failure")
        except ZeroDivisionError:
            lines["exception"] = _next_line()
            logger.exception("bridged exception")
    return lines


def test_the_stdlib_bridge_keeps_where_and_how_a_record_was_logged(
    restored_output: None,
) -> None:
    with _bridged(1) as handler:
        configure_output(level="TRACE", stdlib=True)
        lines = _log_one_of_each()

    records = {record.getMessage(): record for record in handler.records}
    module = Path(__file__).stem
    for message, level, number, line in (
        ("bridged success", "SUCCESS", 25, lines["success"]),
        ("bridged trace", "TRACE", 5, lines["trace"]),
        ("bridged exception", "ERROR", logging.ERROR, lines["exception"]),
    ):
        record = records[message]
        # The record keeps loguru's level name, and names the call that
        # logged it rather than the bridge.
        assert (record.name, record.levelname, record.levelno) == (
            "cytearc",
            level,
            number,
        )
        assert (record.funcName, record.lineno, record.module) == (
            "_log_one_of_each",
            line,
            module,
        )
        assert Path(record.pathname).resolve() == Path(__file__).resolve()
    success = records["bridged success"]
    assert success.exc_info is None
    assert (success.cytearc_run_id, success.cytearc_stage) == (None, None)
    failure = records["bridged exception"]
    assert (failure.cytearc_run_id, failure.cytearc_stage) == ("a" * 64, "pca")
    assert failure.exc_info is not None
    error_type, error, _traceback = failure.exc_info
    assert (error_type, str(error)) == (ZeroDivisionError, "bridged failure")
    formatted = logging.Formatter(
        "%(levelname)s %(module)s:%(lineno)d %(message)s"
    ).format(failure)
    assert formatted.startswith(
        f"ERROR {module}:{lines['exception']} bridged exception\nTraceback"
    )
    assert formatted.endswith("ZeroDivisionError: bridged failure")


def test_the_stdlib_bridge_forwards_only_what_the_cytearc_logger_handles(
    restored_output: None,
) -> None:
    with _bridged(logging.WARNING) as handler:
        configure_output(level="DEBUG", stdlib=True)
        logger.info("below the level of the cytearc logger")
        logger.warning("at the level of the cytearc logger")

    # The handler takes every level, so the logger's own level held the first
    # record back.
    assert [record.getMessage() for record in handler.records] == [
        "at the level of the cytearc logger"
    ]


def test_set_verbosity_ends_the_stdlib_bridge(
    restored_output: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cytearc.log"
    with _bridged(logging.DEBUG) as handler:
        configure_output(level="INFO", timestamps=False, stdlib=True)
        logger.info("bridged first")
        set_verbosity("INFO")
        logger.info("on stdout")
        set_verbosity("INFO", str(path))
        configure_output(stdlib=True)
        logger.info("bridged again")
        # Ending the bridge restores the file that set_verbosity chose.
        configure_output(stdlib=False)
        logger.info("in the file")

    assert [record.getMessage() for record in handler.records] == [
        "bridged first",
        "bridged again",
    ]
    assert capsys.readouterr().out == "INFO: on stdout\n"
    assert path.read_text(encoding="utf-8") == "INFO: in the file\n"


def test_stdlib_must_be_a_bool(restored_output: None) -> None:
    stdlib = cytearc_logging._config.stdlib

    with pytest.raises(TypeError, match="^stdlib must be a bool$"):
        configure_output(stdlib=1)  # type: ignore[arg-type]

    assert cytearc_logging._config.stdlib is stdlib


_BEFORE_PCA = (
    "input_snapshot",
    "filtering",
    "highly_variable_features",
    "normalization",
)


def _cancel_at_pca(
    store: DataStore,
    *,
    via: str,
    reason: str,
    options: dict[str, Any],
) -> list[PipelineEvent]:
    """Cancel a run when its PCA stage starts, from its callback or a thread.

    Check the cancellation, the durable records it leaves, and its events,
    and return the events.
    """
    token = CancellationToken()
    events: list[PipelineEvent] = []
    reached = threading.Event()
    cancelled = threading.Event()

    def cancel_from_thread() -> None:
        if reached.wait(60):
            token.cancel(reason)
        cancelled.set()

    def observe(event: PipelineEvent) -> None:
        events.append(event)
        if (event.kind, event.stage) != ("stage_started", "pca"):
            return
        if via == "callback":
            token.cancel(reason)
        else:
            reached.set()
            cancelled.wait(60)

    thread = threading.Thread(target=cancel_from_thread)
    if via == "thread":
        thread.start()
    try:
        with pytest.raises(PipelineCancelled) as caught:
            store.pipeline.rna(callback=observe, cancel=token, **options)
    finally:
        if via == "thread":
            reached.set()
            thread.join()

    error = caught.value
    run_id = events[0].run_id
    assert (error.run_id, error.stage, error.reason) == (run_id, "pca", reason)
    assert run_id in str(error) and reason in str(error)
    assert (token.cancelled, token.reason) == (True, reason)
    run = store.pipeline.open(run_id=run_id)
    report = run.report()
    assert run.status == "interrupted"
    assert report["run"]["interruption"]["kind"] == "cancelled"
    assert report["run"]["interruption"]["message"] == reason
    stage_order = report["run"]["stageOrder"]
    pca = stage_order.index("pca")
    stages = report["stages"]
    assert [stage["stage"] for stage in stages] == stage_order[: pca + 1]
    assert stages[pca]["status"] == "interrupted"
    assert stages[pca]["interruption"]["kind"] == "cancelled"
    assert stages[pca]["outputs"] == []
    earlier = {stage["stage"]: stage["status"] for stage in stages[:pca]}
    assert {stage for stage, status in earlier.items() if status != "skipped"} == set(
        _BEFORE_PCA
    )
    assert set(earlier.values()) == {"completed", "skipped"}

    milestones = [event for event in events if event.kind != "stage_progress"]
    assert [(event.kind, event.stage) for event in milestones[-3:-1]] == [
        ("stage_started", "pca"),
        ("stage_interrupted", "pca"),
    ]
    assert milestones[-1].kind == "run_interrupted"
    interrupted = milestones[-2]
    assert interrupted.metrics == stages[pca]["metrics"]
    assert interrupted.error is not None
    json.dumps(interrupted.to_dict())
    assert pickle.loads(pickle.dumps(interrupted)).kind == "stage_interrupted"
    return events


@pytest.mark.filterwarnings("ignore:Only .* nearest neighbours:UserWarning")
def test_a_callback_cancels_the_running_stage_and_a_rerun_reuses_the_rest(
    observed_store: DataStore,
) -> None:
    # Other HVG settings make the run compute HVGs, normalization, and PCA,
    # and reuse its input snapshot and filtering.
    options = {**_OPTIONS, "hvg": {"top_n": 40}}
    events = _cancel_at_pca(
        observed_store, via="callback", reason="stopped by a callback", options=options
    )

    # The stages that completed keep their artifacts, which a rerun reuses.
    first = {event.stage: event for event in events if event.kind == "stage_completed"}
    assert first["highly_variable_features"].reused == ()
    rerun_events: list[PipelineEvent] = []
    rerun = observed_store.pipeline.rna(callback=rerun_events.append, **options)
    assert rerun.status == "completed"
    completed = {
        event.stage: event for event in rerun_events if event.kind == "stage_completed"
    }
    for stage in _BEFORE_PCA:
        assert completed[stage].outputs == first[stage].outputs
        assert completed[stage].reused == tuple(
            key for key, _ref in first[stage].outputs
        )
    assert completed["pca"].reused == ()


def test_another_thread_cancels_the_running_stage(observed_store: DataStore) -> None:
    # Other PCA settings make the run compute PCA, which the cancellation
    # interrupts, after it reuses the earlier stages' artifacts.
    _cancel_at_pca(
        observed_store,
        via="thread",
        reason="stopped by another thread",
        options={**_OPTIONS, "pca": {"dims": 4}},
    )


def test_a_cancellation_after_the_run_completed_returns_the_run(
    observed_store: DataStore,
) -> None:
    token = CancellationToken()
    events: list[PipelineEvent] = []

    def cancel_on_completion(event: PipelineEvent) -> None:
        events.append(event)
        if event.kind == "run_completed":
            token.cancel("too late")

    run = observed_store.pipeline.rna(
        callback=cancel_on_completion, cancel=token, **_OPTIONS
    )

    # The run was durable before the cancellation, which cannot stop it.
    assert (run.run_id, run.status) == (events[-1].run_id, "completed")
    assert events[-1].kind == "run_completed"
    assert (token.cancelled, token.reason) == (True, "too late")
    assert observed_store.pipeline.open(run_id=run.run_id).status == "completed"


def test_a_cancellation_seen_after_a_stage_finished_names_no_stage(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = CancellationToken()
    reason = "stopped as PCA finished"
    original = DataStore._reduction_pca

    def pca_then_cancel(self: DataStore, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        # The cancellation arrives once the stage's work has returned.
        token.cancel(reason)
        return result

    monkeypatch.setattr(DataStore, "_reduction_pca", pca_then_cancel)
    events: list[PipelineEvent] = []

    with pytest.raises(PipelineCancelled) as caught:
        observed_store.pipeline.rna(callback=events.append, cancel=token, **_OPTIONS)

    run_id = events[0].run_id
    error = caught.value
    # PCA's work finished, so the run stopped between stages and the
    # cancellation names no stage.
    assert (error.run_id, error.stage, error.reason) == (run_id, None, reason)
    assert str(error) == f"Pipeline run {run_id} was cancelled: {reason}"
    milestones = [event for event in events if event.kind != "stage_progress"]
    assert [(event.kind, event.stage) for event in milestones[-3:]] == [
        ("stage_started", "pca"),
        ("stage_completed", "pca"),
        ("run_interrupted", None),
    ]
    completed, interrupted = milestones[-2:]
    assert interrupted.ordinal is None
    assert isinstance(interrupted.error, ShutdownRequested)
    assert interrupted.error.request.cancelled
    run = observed_store.pipeline.open(run_id=run_id)
    report = run.report()
    assert run.status == "interrupted"
    interruption = report["run"]["interruption"]
    assert (interruption["kind"], interruption["message"]) == ("cancelled", reason)
    stages = report["stages"]
    assert [stage["stage"] for stage in stages] == report["run"]["stageOrder"][
        : len(stages)
    ]
    pca = stages[-1]
    assert (pca["stage"], pca["status"], pca["complete"]) == ("pca", "completed", True)
    assert pca["interruption"] is None
    # The completed stage keeps its outputs and diagnostics.
    assert [output["outputKey"] for output in pca["outputs"]] == ["pca"]
    assert pca["summary"] is not None
    assert completed.summary == pca["summary"]


def test_callbacks_run_outside_the_runs_shutdown_scope(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tokens: list[ShutdownToken] = []

    class RecordedToken(ShutdownToken):
        def __init__(self) -> None:
            super().__init__()
            run_tokens.append(self)

    monkeypatch.setattr(pipeline_executor, "ShutdownToken", RecordedToken)
    expected = observed_store.RNA.rawData[:50].compute()
    # This read passes a checkpoint, which a pending stop would interrupt.
    stopped = ShutdownToken()
    stopped.request(reason="stopped")
    with shutdown_scope(stopped), pytest.raises(ShutdownRequested):
        observed_store.RNA.rawData[:50].compute()

    token = CancellationToken()
    seen: list[ShutdownToken | None] = []
    reads: list[Any] = []

    def listener(event: PipelineEvent) -> None:
        seen.append(current_shutdown_token())
        if (event.kind, event.stage) == ("stage_started", "pca"):
            token.cancel("stopped by a listener")
        elif event.kind == "run_interrupted":
            # A host reads data once the run it observes has stopped.
            reads.append(observed_store.RNA.rawData[:50].compute())

    listened: list[ShutdownToken | None] = []

    def registered(_event: PipelineEvent) -> None:
        listened.append(current_shutdown_token())

    with pipeline_listener(registered), pytest.raises(PipelineCancelled):
        observed_store.pipeline.rna(callback=listener, cancel=token, **_OPTIONS)

    (run_token,) = run_tokens
    assert run_token.requested
    # Callbacks and listeners run in the caller's context, which has no
    # shutdown token, so the run's pending stop never interrupts their reads.
    assert seen and set(seen) == {None}
    assert listened and set(listened) == {None}
    (values,) = reads
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize(
    ("failing", "message"),
    [
        ("end_pipeline_run", "Could not record the interruption of pipeline run"),
        ("open_pipeline_run", "Could not read cancelled pipeline run"),
    ],
    ids=["recording_the_run", "reading_the_run"],
)
def test_a_cancellation_reaches_the_caller_when_its_bookkeeping_fails(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
    message: str,
) -> None:
    failure = RuntimeError(f"{failing} failed")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(pipeline_executor, failing, fail)
    token = CancellationToken()
    reason = "stopped between stages"
    events: list[PipelineEvent] = []

    def stop_after_normalization(event: PipelineEvent) -> None:
        events.append(event)
        if (event.kind, event.stage) == ("stage_completed", "normalization"):
            token.cancel(reason)

    logged: list[tuple[str, BaseException | None]] = []

    def capture(entry: Any) -> None:
        exception = entry.record["exception"]
        logged.append(
            (entry.record["message"], None if exception is None else exception.value)
        )

    sink = logger.add(capture, level="ERROR", format="{message}")
    try:
        with pytest.raises(PipelineCancelled) as caught:
            observed_store.pipeline.rna(
                callback=stop_after_normalization, cancel=token, **_OPTIONS
            )
    finally:
        logger.remove(sink)

    run_id = events[0].run_id
    assert (caught.value.run_id, caught.value.stage, caught.value.reason) == (
        run_id,
        None,
        reason,
    )
    # The failure is logged, never raised in place of the cancellation.
    assert (f"{message} {run_id}", failure) in logged
    lifecycle = [
        (event.kind, event.stage) for event in events if event.kind != "stage_progress"
    ]
    status = observed_store.pipeline.open(run_id=run_id).status
    if failing == "end_pipeline_run":
        # The interruption was never written, so no event reports it.
        assert lifecycle[-1] == ("stage_completed", "normalization")
        assert status == "running"
    else:
        assert lifecycle[-2:] == [
            ("stage_completed", "normalization"),
            ("run_interrupted", None),
        ]
        assert status == "interrupted"


def test_a_token_cancelled_before_the_run_writes_nothing(
    observed_store: DataStore,
) -> None:
    token = CancellationToken()
    token.cancel("cancelled early")
    runs = [run.run_id for run in observed_store.pipeline.list_runs(limit=100)]
    events: list[PipelineEvent] = []

    with pytest.raises(PipelineCancelled) as caught:
        observed_store.pipeline.rna(callback=events.append, cancel=token, **_OPTIONS)

    error = caught.value
    assert (error.run_id, error.stage, error.reason) == (None, None, "cancelled early")
    assert str(error) == "Pipeline run was cancelled before it started: cancelled early"
    assert events == []
    assert [run.run_id for run in observed_store.pipeline.list_runs(limit=100)] == runs


@pytest.mark.parametrize(
    "cancel", ["stop", threading.Event()], ids=["string", "threading_event"]
)
def test_cancel_must_be_a_cancellation_token(
    observed_store: DataStore,
    cancel: object,
) -> None:
    runs = [run.run_id for run in observed_store.pipeline.list_runs(limit=100)]

    with pytest.raises(TypeError, match="^cancel must be a CancellationToken or None$"):
        observed_store.pipeline.rna(cancel=cancel, **_OPTIONS)

    assert [run.run_id for run in observed_store.pipeline.list_runs(limit=100)] == runs


def test_a_cancellation_token_cancels_once() -> None:
    token = CancellationToken()
    assert (token.cancelled, token.reason) == (False, None)
    assert repr(token) == "CancellationToken(active)"

    assert token.cancel("first reason") is True
    assert token.cancel("second reason") is False

    assert (token.cancelled, token.reason) == (True, "first reason")
    assert repr(token) == "CancellationToken(cancelled, reason='first reason')"
    with pytest.raises(TypeError, match="^reason must be a non-empty string$"):
        token.cancel("")


def test_concurrent_cancellations_have_one_first() -> None:
    token = CancellationToken()
    threads = 8
    barrier = threading.Barrier(threads)
    results: list[tuple[str, bool]] = []

    def cancel(reason: str) -> None:
        barrier.wait()
        results.append((reason, token.cancel(reason)))

    workers = [
        threading.Thread(target=cancel, args=(f"thread {index}",))
        for index in range(threads)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert len(results) == threads
    assert [reason for reason, first in results if first] == [token.reason]


@pytest.mark.parametrize(
    ("run_id", "stage"),
    [("a" * 64, "pca"), (None, None)],
    ids=["during_a_stage", "before_the_run"],
)
def test_pipeline_cancelled_pickles(run_id: str | None, stage: str | None) -> None:
    error = PipelineCancelled(run_id, stage, "stop")

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is PipelineCancelled
    assert isinstance(restored, RuntimeError)
    assert (restored.run_id, restored.stage, restored.reason) == (run_id, stage, "stop")
    assert str(restored) == str(error)


def test_pipeline_cancelled_names_a_stage_only_when_given_one() -> None:
    run_id = "a" * 64

    assert str(PipelineCancelled(run_id, "pca", "stop")) == (
        f"Pipeline run {run_id} was cancelled during stage 'pca': stop"
    )
    # A run that stopped between stages names none.
    assert str(PipelineCancelled(run_id, None, "stop")) == (
        f"Pipeline run {run_id} was cancelled: stop"
    )


@pytest.mark.parametrize(
    ("run_id", "stage", "reason", "message"),
    [
        ("", None, "stop", "run_id must be non-empty or None"),
        (7, None, "stop", "run_id must be non-empty or None"),
        ("a" * 64, "", "stop", "stage must be non-empty or None"),
        ("a" * 64, 7, "stop", "stage must be non-empty or None"),
        ("a" * 64, None, "", "reason must be non-empty"),
        (None, None, None, "reason must be non-empty"),
    ],
    ids=[
        "empty_run_id",
        "run_id_type",
        "empty_stage",
        "stage_type",
        "empty_reason",
        "reason_type",
    ],
)
def test_pipeline_cancelled_checks_what_it_names(
    run_id: Any, stage: Any, reason: Any, message: str
) -> None:
    with pytest.raises(TypeError, match=f"^Pipeline cancellation {message}$"):
        PipelineCancelled(run_id, stage, reason)


def test_a_signal_after_a_cancellation_is_the_first_signal() -> None:
    token = ShutdownToken()
    assert token.signal_record is None

    assert token.request(reason="stopped by the user", cancelled=True)
    assert not token.request(reason="stopped again", cancelled=True)
    assert token.signal_record is None
    # The first signal is the first of its kind even after a cancellation, so
    # only a second one escalates.
    assert token.request(reason="received SIGTERM", signal_number=int(signal.SIGTERM))
    assert not token.request(reason="received SIGINT", signal_number=int(signal.SIGINT))

    first = token.request_record
    signalled = token.signal_record
    assert first is not None and signalled is not None
    assert (first.reason, first.cancelled, first.signal_number) == (
        "stopped by the user",
        True,
        None,
    )
    assert (signalled.reason, signalled.cancelled, signalled.signal_name) == (
        "received SIGTERM",
        False,
        "SIGTERM",
    )
    # A checkpoint raises the first request, which the run records.
    with pytest.raises(ShutdownRequested) as caught:
        token.checkpoint()
    assert caught.value.request is first
    with pytest.raises(ValueError, match="^a cancellation is not a signal$"):
        token.request(reason="both", signal_number=int(signal.SIGINT), cancelled=True)


def test_a_first_signal_is_also_the_first_request() -> None:
    token = ShutdownToken()

    assert token.request(reason="received SIGINT", signal_number=int(signal.SIGINT))
    assert not token.request(reason="stopped by the user", cancelled=True)

    assert token.request_record is token.signal_record
    assert token.request_record is not None
    assert token.request_record.signal_name == "SIGINT"


def test_propagate_after_a_cancellation_keeps_a_signals_usual_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", raised.append)

    # A cancellation alone returns, and its caller reports it.
    cancelled = ShutdownToken()
    cancelled.request(reason="stopped by the user", cancelled=True)
    cancelled.propagate()

    # Any other request without a signal is raised.
    requested = ShutdownToken()
    requested.request(reason="host shutdown")
    with pytest.raises(ShutdownRequested, match="^host shutdown$"):
        requested.propagate()

    # A signal after a cancellation reaches the handler it interrupted.
    called: list[tuple[int, object]] = []
    frame = object()
    handled = ShutdownToken()
    handled.request(reason="stopped by the user", cancelled=True)
    handled.request(
        reason="received SIGTERM",
        signal_number=int(signal.SIGTERM),
        previous_handler=lambda number, current: called.append((number, current)),
        frame=frame,  # type: ignore[arg-type]
    )
    with pytest.raises(ShutdownRequested) as caught:
        handled.propagate()
    assert called == [(int(signal.SIGTERM), frame)]
    assert caught.value.request is handled.signal_record

    # A signal whose prior behavior was the default is delivered again.
    delivered = ShutdownToken()
    delivered.request(reason="stopped by the user", cancelled=True)
    delivered.request(
        reason="received SIGHUP",
        signal_number=int(signal.SIGHUP),
        previous_handler=signal.SIG_DFL,
    )
    with pytest.raises(ShutdownRequested, match="^received SIGHUP$"):
        delivered.propagate()
    assert raised == [int(signal.SIGHUP)]


def test_the_signal_guard_escalates_only_a_second_signal_after_a_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(threading, "current_thread", threading.main_thread)
    installed: dict[int, Any] = {}
    escalated: list[int] = []

    def prior_handler(number: int, _frame: object) -> None:
        escalated.append(number)

    monkeypatch.setattr(signal, "getsignal", lambda _number: prior_handler)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.__setitem__(int(number), handler),
    )
    token = ShutdownToken()
    cancel = CancellationToken()

    with TemporarySignalGuard(token), cancellation_scope(cancel, token):
        cancel.cancel("stopped by the user")
        handler = installed[int(signal.SIGTERM)]
        handler(int(signal.SIGTERM), None)
        # The first signal waits for the run's next checkpoint.
        assert escalated == []
        assert token.signal_record is not None
        handler(int(signal.SIGTERM), None)
        assert escalated == [int(signal.SIGTERM)]

    # The guard put the prior handler back, and the run records the
    # cancellation, which came first.
    assert installed[int(signal.SIGTERM)] is prior_handler
    assert token.request_record is not None
    assert token.request_record.cancelled


def test_linking_a_cancelled_token_requests_shutdown_at_once() -> None:
    cancel = CancellationToken()
    cancel.cancel("cancelled before the run")
    token = ShutdownToken()

    with cancellation_scope(cancel, token):
        record = token.request_record
        assert record is not None
        assert (record.reason, record.cancelled) == ("cancelled before the run", True)
        with pytest.raises(ShutdownRequested, match="^cancelled before the run$"):
            token.checkpoint()

    # A token that left the scope no longer hears later cancellations.
    later = CancellationToken()
    unlinked = ShutdownToken()
    with cancellation_scope(later, unlinked):
        assert not unlinked.requested
    later.cancel("too late")
    assert not unlinked.requested
    with cancellation_scope(None, unlinked):
        assert not unlinked.requested
    with pytest.raises(TypeError, match="^cancel must be a CancellationToken or None$"):
        with cancellation_scope("stop", unlinked):  # type: ignore[arg-type]
            pass


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.1,
        rss_baseline_bytes=10,
        rss_peak_bytes=20,
        rss_incremental_peak_bytes=10,
        sample_interval_seconds=0.1,
        sample_count=1,
        sampling_error_count=0,
        rss_unavailable_reason=None,
    )


def _stage(**changes: Any) -> PipelineStageRecord:
    values: dict[str, Any] = {
        "stage": "pca",
        "ordinal": 0,
        "started_at_ns": 10,
        "finished_at_ns": 20,
        "status": "completed",
        "complete": True,
        "outputs": (),
        "plans": (),
        "metrics": _metrics(),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineStageRecord(**values)


def _run_record(*stages: str) -> tuple[zarr.Group, str]:
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
    return root, run.run_id


@pytest.mark.parametrize(
    "diagnostics",
    [
        {},
        {
            "summary": {"dims": 3, "varianceRatio": [0.5, 0.25]},
            "warnings": ("first warning", "second warning"),
        },
    ],
    ids=["without_diagnostics", "with_diagnostics"],
)
def test_stage_records_round_trip_their_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    record = _stage(**diagnostics)

    stored = json.loads(json.dumps(record.to_dict()))

    assert stored["summary"] == diagnostics.get("summary")
    assert stored["warnings"] == list(diagnostics.get("warnings", ()))
    assert PipelineStageRecord.from_dict(stored) == record


def test_stage_records_without_diagnostics_fields_still_read() -> None:
    stored = _stage().to_dict()
    del stored["summary"], stored["warnings"]

    record = PipelineStageRecord.from_dict(stored)

    assert (record.summary, record.warnings) == (None, ())

    # A stage that an earlier release wrote has neither field on disk.
    root, run_id = _run_record("pca")
    start_pipeline_stage_record(root, run_id=run_id, ordinal=0, stage="pca")
    finish_pipeline_stage_record(
        root,
        run_id=run_id,
        ordinal=0,
        status="completed",
        metrics=_metrics(),
        summary={"dims": 3},
        warnings=("a warning",),
    )
    group = root[pipeline_stage_path(run_id, 0)]
    attrs = dict(group.attrs)
    del attrs["summary"], attrs["warnings"]
    group.attrs.put(attrs)

    (loaded,) = load_pipeline_stage_records(root, run_id)

    assert (loaded.status, loaded.summary, loaded.warnings) == ("completed", None, ())


def test_finishing_a_stage_bounds_its_warnings() -> None:
    root, run_id = _run_record("pca")
    start_pipeline_stage_record(root, run_id=run_id, ordinal=0, stage="pca")
    long = "w" * 600
    warnings = [long, *(f"warning {index}" for index in range(STAGE_WARNING_LIMIT + 9))]

    record = finish_pipeline_stage_record(
        root,
        run_id=run_id,
        ordinal=0,
        status="completed",
        metrics=_metrics(),
        warnings=warnings,
    )

    assert STAGE_WARNING_LIMIT == 50
    assert len(record.warnings) == STAGE_WARNING_LIMIT
    assert record.warnings[0] == long[:509] + "..."
    assert record.warnings[1:] == tuple(warnings[1:STAGE_WARNING_LIMIT])
    assert load_pipeline_stage_record(root, run_id, 0).warnings == record.warnings


def test_stage_records_hold_diagnostics_only_where_they_apply() -> None:
    running = {
        "status": "running",
        "complete": False,
        "finished_at_ns": None,
        "metrics": None,
    }
    failure = PipelineErrorRecord(type="ValueError", message="boom")

    with pytest.raises(ValueError, match="^running stages cannot contain diagnostics$"):
        _stage(**running, warnings=("too early",))
    with pytest.raises(ValueError, match="^running stages cannot contain diagnostics$"):
        _stage(**running, summary={"cells": 1})
    with pytest.raises(ValueError, match="^only completed stages can hold a summary$"):
        _stage(status="skipped", summary={"cells": 1})
    with pytest.raises(ValueError, match="^only completed stages can hold a summary$"):
        _stage(status="failed", error=failure, summary={"cells": 1})
    # A failed stage keeps the warnings logged before it failed.
    assert _stage(status="failed", error=failure, warnings=("before",)).warnings == (
        "before",
    )
    with pytest.raises(ValueError, match="^stage warnings exceed the record limits$"):
        _stage(warnings=tuple(f"w{index}" for index in range(STAGE_WARNING_LIMIT + 1)))
    with pytest.raises(ValueError, match="^stage warnings exceed the record limits$"):
        _stage(warnings=("x" * 513,))
    with pytest.raises(TypeError, match="non-empty strings"):
        _stage(warnings=("",))

    root, run_id = _run_record("pca")
    start_pipeline_stage_record(root, run_id=run_id, ordinal=0, stage="pca")
    with pytest.raises(ValueError, match="^only completed stages can hold a summary$"):
        finish_pipeline_stage_record(
            root,
            run_id=run_id,
            ordinal=0,
            status="skipped",
            metrics=_metrics(),
            summary={"cells": 1},
        )
    assert load_pipeline_stage_record(root, run_id, 0).status == "running"


def test_finishing_a_stage_drops_empty_warnings() -> None:
    root, run_id = _run_record("pca")
    start_pipeline_stage_record(root, run_id=run_id, ordinal=0, stage="pca")

    record = finish_pipeline_stage_record(
        root,
        run_id=run_id,
        ordinal=0,
        status="completed",
        metrics=_metrics(),
        warnings=("", "kept"),
    )

    # An empty warning carries nothing, so it never blocks the record.
    assert record.warnings == ("kept",)
    assert load_pipeline_stage_record(root, run_id, 0).status == "completed"


def test_streamed_feature_stages_report_progress(
    datastore_zarr_root: str, tmp_path: Path
) -> None:
    # A store without saved artifacts, so the stages compute and draw bars.
    store = _open_copy(datastore_zarr_root, tmp_path / "fresh.zarr")
    events: list[PipelineEvent] = []

    store.pipeline.rna(
        callback=events.append,
        **{**_OPTIONS, "cell_cycle": True, "markers": True},
    )

    # These stages advance their bars on the thread that consumes a feature
    # stream, so their progress reaches the run's callbacks.
    progressed: dict[str | None, set[str | None]] = {}
    for event in events:
        if event.kind == "stage_progress" and event.progress is not None:
            progressed.setdefault(event.stage, set()).add(event.progress.description)
    assert "Calculating feature statistics" in progressed["highly_variable_features"]
    assert "Calculating feature statistics" in progressed["cell_cycle"]
    assert "Finding markers" in progressed["markers"]


_PCA_BAR = "Observability PCA bar"


def test_callbacks_never_change_the_stage_records(
    observed_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = DataStore._reduction_pca

    def pca_with_progress(self: DataStore, *args: Any, **kwargs: Any) -> Any:
        # PCA reuses its artifact here, so it draws a bar of its own to
        # report progress while it runs.
        with tqdmbar(total=2, desc=_PCA_BAR) as bar:
            bar.update(1)
            bar.update(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataStore, "_reduction_pca", pca_with_progress)
    hosted: list[str | None] = []
    hosting: list[bool] = []

    def busy_host(event: PipelineEvent) -> None:
        # A host that logs, plans artifacts, draws its own bar, and fails
        # while a stage reports progress. Were its bar the stage's, its
        # progress would call it again, so it does its work once at a time.
        if event.kind != "stage_progress" or event.stage != "pca" or hosting:
            return
        assert event.progress is not None
        hosting.append(True)
        try:
            hosted.append(event.progress.description)
            logger.warning("host callback warning")
            observed_store.features.snapshot(
                from_assay="RNA",
                mask=np.ones(observed_store.RNA.feats.N, dtype=bool),
            )
            for _ in iter_progress(range(3), desc="host bar"):
                pass
        finally:
            hosting.clear()
        raise RuntimeError("host callback failure")

    events: list[PipelineEvent] = []

    def record(event: PipelineEvent) -> None:
        events.append(event)
        busy_host(event)

    failures: list[str] = []

    def capture(message: Any) -> None:
        if message.record["message"].startswith("Pipeline callback failed"):
            failures.append(message.record["message"])

    sink = logger.add(capture, level="ERROR", format="{message}")
    try:
        run = observed_store.pipeline.rna(callback=record, **_OPTIONS)
    finally:
        logger.remove(sink)

    assert hosted and set(hosted) == {_PCA_BAR}
    # The host's failure is shown, but it is never the stage's warning.
    assert failures
    assert set(failures) == {
        "Pipeline callback failed while handling stage_progress for pca"
    }
    pca = next(stage for stage in run.report()["stages"] if stage["stage"] == "pca")
    assert pca["status"] == "completed"
    assert "host callback warning" not in pca["warnings"]
    assert not any("Pipeline callback failed" in warning for warning in pca["warnings"])
    assert all(
        plan["operation"] not in {"set_feature_selection", "create_all_features"}
        for plan in pca["plans"]
    )
    # The host's own bar is not the stage's progress.
    assert all(
        event.progress is None or event.progress.description != "host bar"
        for event in events
    )


def test_recipes_check_integer_settings_before_writing(
    observed_store: DataStore,
) -> None:
    before = {run.run_id for run in observed_store.pipeline.list_runs()}

    for neighbors in ({"k": 0}, {"k": "15"}):
        with pytest.raises((TypeError, ValueError), match=r"neighbors\['k'\]"):
            observed_store.pipeline.rna(**{**_OPTIONS, "neighbors": neighbors})

    assert {run.run_id for run in observed_store.pipeline.list_runs()} == before


def test_an_observed_bar_behaves_like_its_bar() -> None:
    updates: list[tuple[int, str | None, int, int | None]] = []

    def observe(
        bar: int, description: str | None, done: int, total: int | None
    ) -> None:
        updates.append((bar, description, done, total))

    with observe_progress(observe):
        bar = tqdmbar(desc="settable", total=2)
        bar.total = 4
        assert len(bar) == 4
        assert bool(bar)
        bar.update(4)
        bar.close()
        # A bar over an iterable takes its total from it and reports each item.
        items = tqdmbar(["a", "b"], desc="items")
        assert list(items) == ["a", "b"]
        items.close()

    first = updates[0][0]
    assert updates == [
        (first, "settable", 4, 4),
        (first + 1, "items", 1, 2),
        (first + 1, "items", 2, 2),
    ]


def test_a_failing_progress_observer_is_logged_and_the_work_goes_on() -> None:
    with pytest.raises(TypeError, match="^observer must be callable$"):
        with observe_progress("not callable"):  # type: ignore[arg-type]
            pass

    calls: list[int] = []

    def failing(_bar: int, _description: str | None, done: int, _total: Any) -> None:
        calls.append(done)
        raise ValueError("observer failure")

    failures: list[tuple[str, type[BaseException] | None]] = []

    def capture(message: Any) -> None:
        exception = message.record["exception"]
        failures.append(
            (message.record["message"], None if exception is None else exception.type)
        )

    sink = logger.add(capture, level="ERROR", format="{message}")
    try:
        with observe_progress(failing):
            bar = tqdmbar(total=3, desc="observed")
            bar.update(1)
            bar.update(2)
            bar.close()
    finally:
        logger.remove(sink)

    # Each failure is logged, and the bar keeps counting and reporting.
    assert calls == [1, 3]
    assert failures == [("A progress observer failed", ValueError)] * 2
