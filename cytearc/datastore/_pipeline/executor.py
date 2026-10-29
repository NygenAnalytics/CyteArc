from collections.abc import Callable, Mapping, Sequence
from contextvars import Context, copy_context
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ...storage.artifacts import ArtifactRef
from ...storage.pipeline_runs import (
    PipelineFieldDescriptor,
    PipelineOutputRecord,
    PipelineRunRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
)
from ...utils.logging import logger
from ...utils.shutdown import (
    CancellationToken,
    ShutdownRequested,
    ShutdownToken,
    TemporarySignalGuard,
    cancellation_scope,
    shutdown_checkpoint,
    shutdown_scope,
)
from ..pipeline_run import (
    PipelineAxisView,
    PipelineCancelled,
    PipelineExecutionError,
    PipelineRun,
    open_pipeline_run,
)
from .events import PipelineCallback, PipelineEventEmitter, run_callbacks
from .ledger import RunLedger, StageSummarizer, end_pipeline_run, interruption_record

type StageAction = Callable[["RunState"], Sequence[tuple[str, ArtifactRef]]]


@dataclass(slots=True)
class RunState:
    """What the stages of one run share while it executes.

    ``artifacts`` maps every stage output key to its ref, the run's outputs
    and the stages' private inputs alike. The other attributes hold values
    that later stages read: the assay object, the frozen feature names, and
    the coordinates that built the graph.
    """

    store: Any
    assay: Any
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    feature_names: np.ndarray | None = None
    coordinates: ArtifactRef | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannedStage:
    """One stage of a resolved recipe.

    A stage without an action is recorded as skipped; its ``on_skip`` hook
    lets later stages see the inputs they would have read from it.
    """

    name: str
    action: StageAction | None
    on_skip: Callable[[RunState], None] | None = None


@dataclass(frozen=True, slots=True)
class RecipePlan:
    """A recipe resolved and validated against one store, ready to execute.

    Attributes:
        recipe: The recorded recipe name, such as ``"rna"``.
        assay: The assay the run's feature view belongs to.
        label: The requested label, or None.
        config: The JSON settings the run records.
        stages: Every stage in order, skipped ones included.
        output_keys: The run outputs, in record order; a key that no stage
            produced is left out.
        fields: Builds the frozen field descriptors once the stages are done.
        summarize: Computes a completed stage's diagnostics, or None.
        start: Builds the shared state before the first stage.
    """

    recipe: str
    assay: str
    label: str | None
    config: Mapping[str, Any]
    stages: tuple[PlannedStage, ...]
    output_keys: tuple[str, ...]
    fields: Callable[[RunState], tuple[PipelineFieldDescriptor, ...]]
    start: Callable[[Any], RunState]
    summarize: StageSummarizer | None = None

    @property
    def stage_order(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)


def execute_recipe(
    store: Any,
    plan: RecipePlan,
    *,
    callback: PipelineCallback | None,
    cancel: CancellationToken | None,
) -> PipelineRun:
    """Execute a resolved recipe and return its completed durable run.

    The run installs cooperative signal handling on the main thread and
    listens to ``cancel``. A cancellation stops the run at its next safe
    checkpoint and raises ``PipelineCancelled``; a signal is handled as
    before and propagated once durable state is written.
    """
    callbacks = run_callbacks(callback)
    # Callbacks run in the caller's context, outside the run's shutdown and
    # cancellation scopes, so a pending stop never interrupts their own work.
    context = copy_context()
    if cancel is not None:
        if not isinstance(cancel, CancellationToken):
            raise TypeError("cancel must be a CancellationToken or None")
        if cancel.cancelled:
            raise PipelineCancelled(None, None, cancel.reason or "cancelled")
    token = ShutdownToken()
    active: list[tuple[str, PipelineEventEmitter, RunLedger]] = []
    result: PipelineRun | None = None
    try:
        with (
            TemporarySignalGuard(token) as guard,
            shutdown_scope(token),
            cancellation_scope(cancel, token),
        ):
            try:
                result = _execute(
                    store,
                    plan,
                    callbacks,
                    context=context,
                    signal_guard_available=guard.available,
                    signal_guard_unavailable_reason=guard.unavailable_reason,
                    active=active,
                )
            except BaseException as error:
                if active:
                    started_run, events, _ledger = active[0]
                    if interruption_record(error) is not None:
                        if _end_interrupted(store, started_run, error):
                            events.emit("run_interrupted", error=error)
                    elif isinstance(error, PipelineExecutionError):
                        events.emit("run_failed", stage=error.stage, error=error)
                raise
    except ShutdownRequested as error:
        if not error.request.cancelled:
            raise
        cancelled_run: str | None = None
        stage: str | None = None
        if active:
            cancelled_run, _events, ledger = active[0]
            stage = ledger.current_stage
            try:
                finished = open_pipeline_run(store, run_id=cancelled_run)
            except Exception:
                # The cancellation is still reported when the run cannot be read.
                logger.exception(
                    f"Could not read cancelled pipeline run {cancelled_run}"
                )
            else:
                if finished.status == "completed":
                    # A cancellation after the run completed comes too late to
                    # stop it, so the caller receives the completed run.
                    result = finished
        if result is None:
            raise PipelineCancelled(
                cancelled_run, stage, error.request.reason
            ) from error
    finally:
        # A signal takes its usual effect once durable state is written, even
        # when a cancellation stopped the run first.
        token.propagate()
    assert result is not None
    return result


def _end_interrupted(store: Any, run_id: str, error: BaseException) -> bool:
    """End an interrupted run; a failed write is logged, never raised.

    The interruption, not a storage error while recording it, is what the
    caller receives.
    """
    try:
        return end_pipeline_run(store.zw, run_id, error)
    except Exception:
        logger.exception(f"Could not record the interruption of pipeline run {run_id}")
        return False


def _open_views(
    store: Any,
    record: PipelineRunRecord,
    outputs: tuple[PipelineOutputRecord, ...],
    fields: tuple[PipelineFieldDescriptor, ...],
) -> None:
    """Open the run's cell and feature views on the record it will complete.

    The views check every field against the run's selections, so a field
    that they cannot read fails the run at finalize, before the run takes
    its label.
    """
    completed = replace(
        record,
        label=record.requested_label,
        # The views read no times; the record only needs a valid one.
        finished_at_ns=record.started_at_ns,
        status="completed",
        complete=True,
        outputs=outputs,
        fields=fields,
    )
    PipelineAxisView(store, completed, axis="cells")
    PipelineAxisView(store, completed, axis="features")


def _execute(
    store: Any,
    plan: RecipePlan,
    callbacks: Sequence[PipelineCallback],
    *,
    context: Context,
    signal_guard_available: bool,
    signal_guard_unavailable_reason: str | None,
    active: list[tuple[str, PipelineEventEmitter, RunLedger]],
) -> PipelineRun:
    shutdown_checkpoint()
    from ... import __version__

    # The shared state reads no data, so it exists before the run record does.
    state = plan.start(store)
    config = dict(plan.config)
    config["shutdown"] = {
        "signalGuardAvailable": signal_guard_available,
        "unavailableReason": signal_guard_unavailable_reason,
    }
    record = create_pipeline_run_record(
        store.zw,
        recipe=plan.recipe,
        requested_label=plan.label,
        assay=plan.assay,
        config=config,
        stage_order=plan.stage_order,
        cytearc_version=__version__,
    )
    events = PipelineEventEmitter(
        callbacks,
        context=context,
        run_id=record.run_id,
        recipe=plan.recipe,
        label=plan.label,
        stage_order=plan.stage_order,
    )
    ledger = RunLedger(store.zw, record.run_id, events, plan.summarize)
    active.append((record.run_id, events, ledger))
    events.emit("run_started")
    for stage in plan.stages:
        action = stage.action
        if action is None:
            if stage.on_skip is not None:
                try:
                    stage.on_skip(state)
                except Exception as error:
                    # A skipped stage's hook fails before its record exists,
                    # so the run ends failed as when a record cannot start.
                    ledger.terminate(error)
                    raise PipelineExecutionError(
                        record.run_id, stage.name, error
                    ) from error
            ledger.skip(stage.name)
            continue

        def run_stage(
            action: StageAction = action,
        ) -> Sequence[tuple[str, ArtifactRef]]:
            outputs = tuple(action(state))
            state.artifacts.update(outputs)
            return outputs

        ledger.run(stage.name, run_stage)

    outputs = tuple(
        PipelineOutputRecord(key=key, artifact=state.artifacts[key])
        for key in plan.output_keys
        if key in state.artifacts
    )
    try:
        fields = plan.fields(state)
        _open_views(store, record, outputs, fields)
        shutdown_checkpoint()
        complete_pipeline_run_record(
            store.zw,
            run_id=record.run_id,
            outputs=outputs,
            fields=fields,
        )
    except Exception as error:
        ledger.terminate(error)
        raise PipelineExecutionError(record.run_id, "finalize", error) from error
    # The run is complete and durable, so listeners learn it before a pending
    # shutdown, which can no longer change the run, is handled.
    events.emit("run_completed")
    run = open_pipeline_run(store, run_id=record.run_id)
    shutdown_checkpoint()
    return run
