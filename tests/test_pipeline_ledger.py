"""Pipeline stages run one at a time and end with exactly one durable outcome."""

import asyncio
import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc.datastore._pipeline.ledger as ledger_module
from cytearc.datastore._pipeline.events import PipelineEvent, PipelineEventEmitter
from cytearc.datastore._pipeline.ledger import (
    RunLedger,
    StageSummarizer,
    interruption_record,
)
from cytearc.datastore.pipeline_run import PipelineExecutionError
from cytearc.storage.pipeline_runs import (
    bounded_record_text,
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_run_record,
    load_pipeline_stage_records,
    start_pipeline_stage_record,
)
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import logger
from cytearc.utils.shutdown import (
    ShutdownRequest,
    ShutdownRequested,
    ShutdownToken,
    shutdown_scope,
)
from tests.test_pipeline_run_record_paths import _captured_warnings, _metrics


def _ledger(
    *stages: str,
    summarize: StageSummarizer | None = None,
) -> tuple[zarr.Group, RunLedger, list[PipelineEvent]]:
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
        root, run.run_id, emitter, summarize, workers=1, memory_budget_bytes=2**30
    )
    return root, ledger, events


def _statuses(root: zarr.Group, ledger: RunLedger) -> dict[str, str]:
    return {
        record.stage: record.status
        for record in load_pipeline_stage_records(root, ledger.run_id)
    }


@pytest.mark.parametrize("name", ["workers", "memory_budget_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, None, "4"])
def test_a_ledger_needs_positive_integer_workers_and_budget(name, value) -> None:
    root, ledger, _events = _ledger("first")
    resources = {"workers": 1, "memory_budget_bytes": 2**30, name: value}
    with pytest.raises(
        ValueError, match=f"^{name} must be a positive integer, not {value!r}$"
    ):
        RunLedger(root, ledger.run_id, ledger.events, **resources)
    # One worker and a budget of one byte are positive integers.
    accepted = RunLedger(
        root, ledger.run_id, ledger.events, workers=1, memory_budget_bytes=1
    )
    assert (accepted.workers, accepted.memory_budget_bytes) == (1, 1)


@pytest.mark.parametrize("value", [-1, True, 1.0, None, "1"])
def test_a_ledger_starts_at_a_non_negative_integer_ordinal(value) -> None:
    root, ledger, _events = _ledger("first", "second")
    resources = {"workers": 1, "memory_budget_bytes": 2**30}
    with pytest.raises(
        ValueError, match=f"^ordinal must be a non-negative integer, not {value!r}$"
    ):
        RunLedger(root, ledger.run_id, ledger.events, ordinal=value, **resources)
    # A ledger that starts at the second stage records it at ordinal 1, after
    # the first stage's record.
    ledger.skip("first")
    resumed = RunLedger(root, ledger.run_id, ledger.events, ordinal=1, **resources)
    resumed.skip("second")
    assert [
        (record.ordinal, record.stage)
        for record in load_pipeline_stage_records(root, ledger.run_id)
    ] == [(0, "first"), (1, "second")]


def test_grouped_interruption_ends_the_stage_as_that_interruption() -> None:
    root, ledger, events = _ledger("first", "second")
    interruption = KeyboardInterrupt("stop")

    def first_stage() -> tuple[()]:
        raise BaseExceptionGroup(
            "workers failed",
            [ValueError("worker"), BaseExceptionGroup("nested", [interruption])],
        )

    with pytest.raises(KeyboardInterrupt) as caught:
        ledger.run("first", first_stage)

    assert caught.value is interruption
    assert load_pipeline_run_record(root, ledger.run_id).status == "interrupted"
    assert _statuses(root, ledger) == {"first": "interrupted"}
    assert [(event.kind, event.stage, event.ordinal) for event in events] == [
        ("stage_started", "first", 0),
        ("stage_interrupted", "first", 0),
        ("run_interrupted", "first", 0),
    ]
    assert events[-1].error is interruption


def test_grouped_failure_without_interruption_fails_the_stage() -> None:
    root, ledger, events = _ledger("first")
    group = ExceptionGroup("workers failed", [ValueError("a"), ValueError("b")])

    def first_stage() -> tuple[()]:
        raise group

    with pytest.raises(PipelineExecutionError) as caught:
        ledger.run("first", first_stage)

    assert (caught.value.run_id, caught.value.stage) == (ledger.run_id, "first")
    assert caught.value.__cause__ is group
    assert load_pipeline_run_record(root, ledger.run_id).status == "failed"
    assert _statuses(root, ledger) == {"first": "failed"}
    # The ledger reports the stage; the executor that owns the run reports
    # run_failed once the error reaches it.
    assert [(event.kind, event.stage) for event in events] == [
        ("stage_started", "first"),
        ("stage_failed", "first"),
    ]
    assert events[-1].error is group


def test_interruption_after_a_completed_action_keeps_the_stage_completed() -> None:
    root, ledger, events = _ledger("first", "second")
    token = ShutdownToken()

    def first_stage() -> tuple[()]:
        token.request(reason="stop after this stage")
        return ()

    with shutdown_scope(token), pytest.raises(ShutdownRequested):
        ledger.run("first", first_stage)

    assert load_pipeline_run_record(root, ledger.run_id).status == "interrupted"
    assert _statuses(root, ledger) == {"first": "completed"}
    # The stage's work finished, so the run stopped between stages and the
    # interruption names no stage.
    assert [(event.kind, event.stage) for event in events] == [
        ("stage_started", "first"),
        ("stage_completed", "first"),
        ("run_interrupted", None),
    ]
    assert ledger.current_stage is None


def test_interruption_pending_before_a_stage_starts_writes_no_stage() -> None:
    root, ledger, events = _ledger("first", "second")
    token = ShutdownToken()
    token.request(reason="stop before the stage")

    with shutdown_scope(token):
        with pytest.raises(ShutdownRequested):
            ledger.run("first", lambda: ())
        with pytest.raises(ShutdownRequested):
            ledger.skip("first")

    # No stage started, so the executor that owns the run ends it.
    assert load_pipeline_run_record(root, ledger.run_id).status == "running"
    assert load_pipeline_stage_records(root, ledger.run_id) == ()
    assert (ledger.ordinal, ledger.current_stage, events) == (0, None, [])


def test_each_completed_stage_records_only_its_own_diagnostics() -> None:
    summarized: list[tuple[str, tuple[tuple[str, ArtifactRef], ...]]] = []

    def summarize(
        stage: str,
        outputs: Sequence[tuple[str, ArtifactRef]],
    ) -> Mapping[str, Any] | None:
        summarized.append((stage, tuple(outputs)))
        return {"cells": 3} if stage == "first" else None

    root, ledger, events = _ledger("first", "second", summarize=summarize)
    running: list[str | None] = []

    def stage_logging(message: str) -> Callable[[], tuple[()]]:
        def action() -> tuple[()]:
            running.append(ledger.current_stage)
            logger.warning(message)
            return ()

        return action

    ledger.run("first", stage_logging("First stage warning"))
    # A warning logged between stages belongs to neither stage.
    logger.warning("Between stages")
    ledger.run("second", stage_logging("Second stage warning"))

    first, second = load_pipeline_stage_records(root, ledger.run_id)
    assert (first.summary, first.warnings) == ({"cells": 3}, ("First stage warning",))
    assert (second.summary, second.warnings) == (None, ("Second stage warning",))
    assert summarized == [("first", ()), ("second", ())]
    # The ledger names the running stage, and none between stages.
    assert running == ["first", "second"]
    assert ledger.current_stage is None
    completed = [event for event in events if event.kind == "stage_completed"]
    assert [(event.summary, event.warnings) for event in completed] == [
        (first.summary, first.warnings),
        (second.summary, second.warnings),
    ]


def test_a_summary_failure_only_warns_and_the_stage_completes() -> None:
    def summarize(
        _stage: str,
        _outputs: Sequence[tuple[str, ArtifactRef]],
    ) -> Mapping[str, Any] | None:
        raise ValueError("no diagnostics")

    root, ledger, events = _ledger("first", summarize=summarize)

    with _captured_warnings() as messages:
        ledger.run("first", lambda: ())

    (stage,) = load_pipeline_stage_records(root, ledger.run_id)
    assert (stage.status, stage.summary) == ("completed", None)
    assert "Could not summarize pipeline stage first: no diagnostics" in messages
    assert load_pipeline_run_record(root, ledger.run_id).status == "running"
    assert [event.kind for event in events] == ["stage_started", "stage_completed"]


@pytest.mark.parametrize(
    "cancellation",
    # Zarr's synchronous API reports a cancelled read with the CancelledError
    # of concurrent.futures, an Exception, which must end the run as asyncio's
    # does rather than be logged as a failed summary.
    [asyncio.CancelledError, concurrent.futures.CancelledError],
    ids=["asyncio", "zarr"],
)
def test_a_cancelled_summary_ends_the_run_after_its_completed_stage(
    cancellation: type[BaseException],
) -> None:
    error = cancellation()

    def summarize(
        _stage: str,
        _outputs: Sequence[tuple[str, ArtifactRef]],
    ) -> Mapping[str, Any] | None:
        raise error

    root, ledger, events = _ledger("first", "second", summarize=summarize)

    with pytest.raises(cancellation) as caught:
        ledger.run("first", lambda: ())

    assert caught.value is error
    assert _statuses(root, ledger) == {"first": "completed"}
    run = load_pipeline_run_record(root, ledger.run_id)
    assert (run.status, run.complete) == ("interrupted", True)
    assert run.interruption is not None
    assert run.interruption.kind == "asyncio_cancelled"
    assert [(event.kind, event.stage) for event in events] == [
        ("stage_started", "first"),
        ("stage_completed", "first"),
        ("run_interrupted", None),
    ]


@pytest.mark.parametrize(
    ("error", "raised", "status", "kind"),
    [
        (
            RuntimeError("stage failed"),
            PipelineExecutionError,
            "failed",
            "stage_failed",
        ),
        (
            KeyboardInterrupt("stop"),
            KeyboardInterrupt,
            "interrupted",
            "stage_interrupted",
        ),
    ],
    ids=["failed", "interrupted"],
)
def test_unfinished_stage_keeps_its_warnings_without_a_summary(
    error: BaseException,
    raised: type[BaseException],
    status: str,
    kind: str,
) -> None:
    summarized: list[str] = []

    def summarize(
        stage: str,
        _outputs: Sequence[tuple[str, ArtifactRef]],
    ) -> Mapping[str, Any] | None:
        summarized.append(stage)
        return {"cells": 3}

    root, ledger, events = _ledger("first", summarize=summarize)

    def first_stage() -> tuple[()]:
        logger.warning("Stopped early")
        raise error

    with pytest.raises(raised):
        ledger.run("first", first_stage)

    (stage,) = load_pipeline_stage_records(root, ledger.run_id)
    assert (stage.status, stage.summary, stage.warnings) == (
        status,
        None,
        ("Stopped early",),
    )
    assert summarized == []
    (terminal,) = [event for event in events if event.kind == kind]
    assert (terminal.summary, terminal.warnings, terminal.error) == (
        None,
        ("Stopped early",),
        error,
    )


@pytest.mark.parametrize(
    ("method", "failing", "statuses", "kinds"),
    [
        ("run", "start_pipeline_stage_record", {}, []),
        (
            "run",
            "finish_pipeline_stage_record",
            {"first": "failed"},
            ["stage_started", "stage_failed"],
        ),
        ("skip", "start_pipeline_stage_record", {}, []),
        (
            "skip",
            "finish_pipeline_stage_record",
            {"first": "failed"},
            ["stage_failed"],
        ),
    ],
    ids=["run_start", "run_finish", "skip_start", "skip_finish"],
)
def test_failure_while_recording_a_stage_fails_the_stage_and_run(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    failing: str,
    statuses: dict[str, str],
    kinds: list[str],
) -> None:
    root, ledger, events = _ledger("first", "second")
    expected = RuntimeError("stage bookkeeping failed")
    record = getattr(ledger_module, failing)
    failures: list[RuntimeError] = []

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        # Only the stage's first write fails, so its failure can be recorded.
        if not failures:
            failures.append(expected)
            raise expected
        return record(*args, **kwargs)

    monkeypatch.setattr(ledger_module, failing, fail_once)

    with pytest.raises(PipelineExecutionError) as caught:
        if method == "run":
            ledger.run("first", lambda: ())
        else:
            ledger.skip("first")

    assert (caught.value.run_id, caught.value.stage) == (ledger.run_id, "first")
    assert caught.value.__cause__ is expected
    run = load_pipeline_run_record(root, ledger.run_id)
    assert (run.status, run.complete) == ("failed", True)
    assert run.error is not None
    assert (run.error.type, run.error.message) == ("RuntimeError", str(expected))
    assert _statuses(root, ledger) == statuses
    assert all(
        stage.complete for stage in load_pipeline_stage_records(root, ledger.run_id)
    )
    assert [event.kind for event in events] == kinds
    if kinds:
        assert events[-1].error is expected


@pytest.mark.parametrize("failing", ["start", "completion after an interruption"])
@pytest.mark.parametrize(
    "interruption",
    # A keyboard interrupt never enters the ledger's handlers of ordinary
    # errors, so it shows how a cancelled record write must pass them.
    [KeyboardInterrupt, concurrent.futures.CancelledError],
    ids=["keyboard", "zarr_cancelled"],
)
def test_a_cancelled_record_write_passes_as_a_keyboard_interrupt_does(
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
    interruption: type[BaseException],
) -> None:
    error = interruption("the record write was cancelled")
    summarize: StageSummarizer | None = None
    if failing == "start":

        def start(*_args: Any, **_kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(ledger_module, "start_pipeline_stage_record", start)
    else:
        record = ledger_module.finish_pipeline_stage_record

        def finish(*args: Any, **kwargs: Any) -> Any:
            if kwargs["status"] == "completed":
                raise error
            return record(*args, **kwargs)

        monkeypatch.setattr(ledger_module, "finish_pipeline_stage_record", finish)

        def summarize(
            _stage: str,
            _outputs: Sequence[tuple[str, ArtifactRef]],
        ) -> Mapping[str, Any] | None:
            # An interruption after the stage's work, which the ledger then
            # records as a completed stage.
            raise asyncio.CancelledError

    root, ledger, events = _ledger("first", "second", summarize=summarize)

    with pytest.raises(interruption) as caught:
        ledger.run("first", lambda: ())

    # The error passes unwrapped, and the executor that owns the run ends it.
    assert caught.value is error
    assert load_pipeline_run_record(root, ledger.run_id).status == "running"
    started = failing != "start"
    assert _statuses(root, ledger) == ({"first": "running"} if started else {})
    assert [event.kind for event in events] == (["stage_started"] if started else [])


def _interruptions() -> list[tuple[BaseException, str]]:
    return [
        (KeyboardInterrupt("stop while skipping"), "keyboard_interrupt"),
        (
            ShutdownRequested(
                ShutdownRequest(
                    requested_at_ns=1, reason="stop while skipping", cancelled=True
                )
            ),
            "cancelled",
        ),
    ]


@pytest.mark.parametrize(
    ("error", "kind"), _interruptions(), ids=["keyboard_interrupt", "cancelled"]
)
@pytest.mark.parametrize(
    ("failing", "statuses"),
    [
        ("start_pipeline_stage_record", {}),
        ("finish_pipeline_stage_record", {"first": "interrupted"}),
    ],
    ids=["while_starting", "while_finishing"],
)
def test_an_interruption_while_a_skip_is_recorded_names_no_stage(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    kind: str,
    failing: str,
    statuses: dict[str, str],
) -> None:
    root, ledger, events = _ledger("first", "second")
    record = getattr(ledger_module, failing)

    def interrupt_the_skip(*args: Any, **kwargs: Any) -> Any:
        # The interruption lands while the skip is written, as the stage
        # starts or as it is finished; recording the interruption then works.
        if kwargs.get("status") in (None, "skipped"):
            raise error
        return record(*args, **kwargs)

    monkeypatch.setattr(ledger_module, failing, interrupt_the_skip)

    with pytest.raises(type(error)) as caught:
        ledger.skip("first")

    assert caught.value is error
    # A skip emits no stage event, so the run stops between stages and its
    # one event names no stage.
    assert [(event.kind, event.stage, event.ordinal) for event in events] == [
        ("run_interrupted", None, None)
    ]
    assert events[-1].error is error
    run = load_pipeline_run_record(root, ledger.run_id)
    assert (run.status, run.complete) == ("interrupted", True)
    assert run.interruption is not None and run.interruption.kind == kind
    assert _statuses(root, ledger) == statuses
    for stage in load_pipeline_stage_records(root, ledger.run_id):
        assert stage.complete
        assert stage.interruption is not None and stage.interruption.kind == kind
    assert (ledger.ordinal, ledger.current_stage) == (0, None)


@pytest.mark.parametrize("method", ["run", "skip"])
def test_the_run_ends_when_its_interrupted_stage_cannot_be_recorded(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    root, ledger, events = _ledger("first", "second")
    interruption = KeyboardInterrupt("stop")
    failure = RuntimeError("cannot record the interrupted stage")
    record = ledger_module.finish_pipeline_stage_record

    def finish(*args: Any, **kwargs: Any) -> Any:
        if kwargs["status"] == "interrupted":
            raise failure
        if kwargs["status"] == "skipped":
            # The interruption lands while a skip is written.
            raise interruption
        return record(*args, **kwargs)

    monkeypatch.setattr(ledger_module, "finish_pipeline_stage_record", finish)

    def first_stage() -> tuple[()]:
        raise interruption

    logged: list[tuple[str, type[BaseException] | None]] = []

    def capture(message: Any) -> None:
        exception = message.record["exception"]
        logged.append(
            (message.record["message"], None if exception is None else exception.type)
        )

    sink = logger.add(capture, level="ERROR", format="{message}")
    # The interruption reaches the caller, and the storage error is logged, so
    # a keyboard interrupt or a cancellation still stops its caller.
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            if method == "run":
                ledger.run("first", first_stage)
            else:
                ledger.skip("first")
    finally:
        logger.remove(sink)

    assert caught.value is interruption
    assert (
        f"Could not record the interruption of stage 0 of pipeline run {ledger.run_id}",
        RuntimeError,
    ) in logged
    # The stage record stays as it was, but the run ends: it is never left
    # running.
    run = load_pipeline_run_record(root, ledger.run_id)
    assert (run.status, run.complete) == ("interrupted", True)
    assert run.interruption is not None
    assert (run.interruption.kind, run.interruption.message) == (
        "keyboard_interrupt",
        "stop",
    )
    assert _statuses(root, ledger) == {"first": "running"}
    expected = (
        [
            ("stage_started", "first"),
            ("stage_interrupted", "first"),
            ("run_interrupted", "first"),
        ]
        if method == "run"
        else [("run_interrupted", None)]
    )
    assert [(event.kind, event.stage) for event in events] == expected


@pytest.mark.parametrize(
    ("method", "error", "status", "kinds"),
    [
        (
            "run",
            OSError("the store failed after the write"),
            "failed",
            [("stage_started", "first"), ("stage_failed", "first")],
        ),
        (
            "skip",
            OSError("the store failed after the write"),
            "failed",
            [("stage_failed", "first")],
        ),
        (
            "skip",
            KeyboardInterrupt("stop after the write"),
            "interrupted",
            [("run_interrupted", None)],
        ),
    ],
    ids=["run-failure", "skip-failure", "skip-interruption"],
)
def test_an_error_after_the_stage_record_is_written_keeps_the_record(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    error: BaseException,
    status: str,
    kinds: list[tuple[str, str | None]],
) -> None:
    root, ledger, events = _ledger("first", "second")
    record = ledger_module.finish_pipeline_stage_record
    written = "completed" if method == "run" else "skipped"

    def write_then_raise(*args: Any, **kwargs: Any) -> Any:
        # The stage's record is written, and the error lands right after it.
        finished = record(*args, **kwargs)
        if kwargs["status"] == written:
            raise error
        return finished

    monkeypatch.setattr(ledger_module, "finish_pipeline_stage_record", write_then_raise)

    expected = KeyboardInterrupt if status == "interrupted" else PipelineExecutionError
    with pytest.raises(expected) as caught:
        if method == "run":
            ledger.run("first", lambda: ())
        else:
            ledger.skip("first")

    cause = caught.value if status == "interrupted" else caught.value.__cause__
    assert cause is error
    # The record that was written stays as it was, and the run ends.
    assert _statuses(root, ledger) == {"first": written}
    run = load_pipeline_run_record(root, ledger.run_id)
    assert (run.status, run.complete) == (status, True)
    assert [(event.kind, event.stage) for event in events] == kinds


def test_stages_start_only_after_every_earlier_stage_succeeded() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = create_pipeline_run_record(
        root,
        recipe="rna",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("a", "b", "c", "d"),
        cytearc_version="1.0.0",
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=0, stage="a")
    with pytest.raises(ValueError, match="sequentially"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="c")
    # "a" is still running, so no later stage may start beside it.
    with pytest.raises(ValueError, match="sequentially"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="b")
    finish_pipeline_stage_record(
        root, run_id=run.run_id, ordinal=0, status="completed", metrics=_metrics()
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="b")
    finish_pipeline_stage_record(
        root, run_id=run.run_id, ordinal=1, status="skipped", metrics=_metrics()
    )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="c")
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=2,
        status="failed",
        metrics=_metrics(),
        error=RuntimeError("deliberate failure"),
    )
    with pytest.raises(ValueError, match="sequentially"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=3, stage="d")


_LONG_TEXT = "stop: " + "x" * 600
_BOUNDED_TEXT = _LONG_TEXT[:509] + "..."


def _long_interruptions() -> list[tuple[BaseException, str]]:
    return [
        (
            ShutdownRequested(
                ShutdownRequest(
                    requested_at_ns=1,
                    reason=_LONG_TEXT,
                    signal_number=15,
                    signal_name="SIGTERM",
                )
            ),
            "signal",
        ),
        (
            ShutdownRequested(
                ShutdownRequest(requested_at_ns=1, reason=_LONG_TEXT, cancelled=True)
            ),
            "cancelled",
        ),
        (
            ShutdownRequested(ShutdownRequest(requested_at_ns=1, reason=_LONG_TEXT)),
            "shutdown_request",
        ),
        (KeyboardInterrupt(_LONG_TEXT), "keyboard_interrupt"),
        (asyncio.CancelledError(_LONG_TEXT), "asyncio_cancelled"),
        # Zarr's synchronous API reports a cancelled call with this class.
        (concurrent.futures.CancelledError(_LONG_TEXT), "asyncio_cancelled"),
    ]


@pytest.mark.parametrize(
    ("error", "kind"),
    _long_interruptions(),
    ids=[
        "signal",
        "cancelled",
        "shutdown_request",
        "keyboard_interrupt",
        "asyncio_cancelled",
        "zarr_cancelled",
    ],
)
def test_long_interruption_messages_are_truncated_and_recorded(
    error: BaseException, kind: str
) -> None:
    record = interruption_record(error)
    assert record is not None
    assert (record.kind, record.message) == (kind, _BOUNDED_TEXT)
    assert len(record.message) == 512
    # A message of exactly the bound is kept whole.
    assert bounded_record_text(_LONG_TEXT[:512]) == _LONG_TEXT[:512]

    root, ledger, events = _ledger("first")

    def first_stage() -> tuple[()]:
        raise error

    with pytest.raises(type(error)) as caught:
        ledger.run("first", first_stage)

    # The caller sees the original interruption with its full message.
    assert caught.value is error
    run = load_pipeline_run_record(root, ledger.run_id)
    assert (run.status, run.complete) == ("interrupted", True)
    assert run.interruption is not None
    assert (run.interruption.kind, run.interruption.message) == (kind, _BOUNDED_TEXT)
    (stage,) = load_pipeline_stage_records(root, ledger.run_id)
    assert (stage.status, stage.complete) == ("interrupted", True)
    assert stage.interruption is not None
    assert stage.interruption.message == _BOUNDED_TEXT
    assert [event.kind for event in events][-2:] == [
        "stage_interrupted",
        "run_interrupted",
    ]
