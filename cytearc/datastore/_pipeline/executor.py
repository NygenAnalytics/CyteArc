from collections.abc import Callable, Mapping, Sequence
from contextvars import Context, copy_context
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from ...assay import NormMethod
from ...storage.artifact_writer import (
    ArtifactComputationRequired,
    artifact_forecast_scope,
)
from ...storage.artifacts import ArtifactRef
from ...storage.pipeline_runs import (
    PipelineFieldDescriptor,
    PipelineOutputListRecord,
    PipelineOutputRecord,
    PipelineRunRecord,
    PipelineStageRecord,
    changed_since_read,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    ensure_pipeline_label_claimable,
    load_pipeline_run_record,
    load_pipeline_stage_records,
    pause_pipeline_run_record,
    resume_pipeline_run_record,
)
from ...storage.stores import has_atomic_conditional_writes
from ...utils.logging import logger
from ...utils.shutdown import (
    CancellationToken,
    ShutdownRequested,
    ShutdownToken,
    TemporarySignalGuard,
    cancellation_scope,
    is_interruption,
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
from .normalizers import applied_normalizers

# A stage output names one artifact, or a tuple of artifacts in a fixed order,
# such as the Leiden candidates ordered by resolution.
type StageOutput = tuple[str, ArtifactRef | tuple[ArtifactRef, ...]]
type StageAction = Callable[["RunState"], Sequence[StageOutput]]

# The names under which a stage declares the run state's coordinates and
# frozen feature names among its inputs and outputs, beside artifact keys.
COORDINATES = "coordinates"
FEATURE_NAMES = "feature_names"


@dataclass(slots=True)
class RunState:
    """What the stages of one run share while it executes.

    ``artifacts`` maps every stage output key that names one artifact to its
    ref, the run's outputs and the stages' private inputs alike, and
    ``artifact_lists`` maps each key that names several artifacts to them,
    in order. The other attributes hold values that later stages read: the
    assay object, the frozen feature names, and the coordinates that built
    the graph.
    """

    store: Any
    assay: Any
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    artifact_lists: dict[str, tuple[ArtifactRef, ...]] = field(default_factory=dict)
    feature_names: np.ndarray | None = None
    coordinates: ArtifactRef | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def add(self, outputs: Sequence[StageOutput]) -> None:
        """Add a stage's outputs, each to the mapping that holds its kind."""
        for key, value in outputs:
            if isinstance(value, tuple):
                self.artifact_lists[key] = value
            else:
                self.artifacts[key] = value


@dataclass(frozen=True, slots=True)
class PlannedStage:
    """One stage of a resolved recipe.

    A stage without an action is recorded as skipped; its ``on_skip`` hook
    lets later stages see the inputs they would have read from it.

    ``inputs`` names what the action, or the hook of a skipped stage, reads
    from the run state, and ``outputs`` what it adds there: artifact keys,
    :data:`COORDINATES` and :data:`FEATURE_NAMES` for those values of the
    state, and the keys of its extras. A plan follows them to find the stages
    that wait on a stage that would be computed, and refuses a stage that
    reads or adds an artifact key it does not declare.
    """

    name: str
    action: StageAction | None
    on_skip: Callable[[RunState], None] | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


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
            produced is left out. A key whose stage produced several
            artifacts is recorded among the run's output lists.
        fields: Builds the frozen field descriptors once the stages are done.
        summarize: Computes a completed stage's diagnostics, or None.
        start: Builds the shared state before the first stage.
        normalizers: The normalizer that the run applies to each assay, by
            name, as the assay's ``normMethod`` while the stages run.
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
    normalizers: Mapping[str, NormMethod] = MappingProxyType({})

    @property
    def stage_order(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)


# The next step of a resume that cannot go on, which reuses what still holds.
_RERUN = "ds.pipeline.rerun(run) starts a new run that reuses its finished stages"


def check_run_until(
    plan: RecipePlan,
    run_until: object,
    *,
    after: int = 0,
    run_id: str | None = None,
) -> int | None:
    """Return the ordinal of the stage that ``run_until`` names, or None without one.

    ``after`` is the first stage left to run, of a run that resumes, and
    ``run_id`` that run's ID.

    Raises:
        TypeError: If ``run_until`` is not a stage's name or None.
        ValueError: If it names no stage of the recipe, a stage that the
            run's settings skip, or a stage that the run has already run;
            the message lists the stages that the run runs, or for a run
            that resumes, the stages that it has left to run.
    """
    if run_until is None:
        return None
    runs = [
        stage.name
        for ordinal, stage in enumerate(plan.stages)
        if stage.action is not None and ordinal >= after
    ]
    listed = (
        f"the stages that this run runs are {runs!r}"
        if after == 0
        else f"the stages that run {run_id} has left to run are {runs!r}"
    )
    if not isinstance(run_until, str):
        raise TypeError(
            f"run_until must name a stage of the run or be None, not "
            f"{run_until!r}; {listed}"
        )
    if run_until not in plan.stage_order:
        raise ValueError(
            f"run_until={run_until!r} is not a stage of the {plan.recipe} recipe; "
            f"{listed}"
        )
    ordinal = plan.stage_order.index(run_until)
    if plan.stages[ordinal].action is None:
        raise ValueError(
            f"run_until={run_until!r} names a stage that this run's settings "
            f"skip; {listed}"
        )
    if ordinal < after:
        raise ValueError(
            f"run_until={run_until!r} names a stage that run {run_id} has already "
            f"run; {listed}"
        )
    return ordinal


def check_resumable_record(record: PipelineRunRecord, version: str) -> None:
    """Refuse a run that cannot resume: one that is not paused, or that another CyteArc release recorded.

    Raises:
        ValueError: If the run is not paused, its pause was not written
            completely, or its ``cytearc_version`` is not ``version``.
    """
    run_id = record.run_id
    if record.status == "running":
        # Another process may run it now, so a rerun could duplicate its work.
        raise ValueError(
            f"Pipeline run {run_id} is 'running', and only a paused run resumes: "
            "another process runs it, or stopped without recording its end; open "
            "the run again to follow it"
        )
    if record.status == "completed":
        raise ValueError(
            f"Pipeline run {run_id} is 'completed', and only a paused run resumes; "
            "ds.pipeline.open(run_id=...) reads its outputs"
        )
    if record.status != "paused":
        raise ValueError(
            f"Pipeline run {run_id} is {record.status!r}, and only a paused run "
            f"resumes; {_RERUN}"
        )
    if not record.complete:
        raise ValueError(
            f"Pipeline run {run_id} did not finish writing its pause, so it cannot "
            f"resume; {_RERUN}"
        )
    if record.cytearc_version != version:
        raise ValueError(
            f"Pipeline run {run_id} was recorded by CyteArc "
            f"{record.cytearc_version}, so it resumes only with that release, not "
            f"{version}; resume it with CyteArc {record.cytearc_version}, or "
            "ds.pipeline.rerun(run) runs its settings again with this release"
        )


def _check_resumes_plan(record: PipelineRunRecord, plan: RecipePlan) -> None:
    """Refuse a plan other than the one that the paused run recorded.

    The plan takes the run's recipe, assay and label from its record, so its
    stage order and settings are what can differ.
    """
    recorded = {key: value for key, value in record.config.items() if key != "shutdown"}
    if plan.stage_order != record.stage_order or dict(plan.config) != recorded:
        raise ValueError(
            f"Pipeline run {record.run_id} cannot resume: its recorded settings no "
            "longer resolve to the stages and settings that it recorded; "
            "ds.pipeline.rerun(run) starts a new run with these settings that "
            "reuses the finished stages that still match"
        )


def _finished_stages(
    store: Any, record: PipelineRunRecord
) -> tuple[PipelineStageRecord, ...]:
    """Return the stage records of a paused run, which must be the stages its segments name.

    The run record is read again first: one that holds more segments than
    ``record``, or is no longer paused, was resumed or ended by another
    process after this one read it.

    Raises:
        RuntimeError: If another process resumed or ended the run since
            ``record`` was read.
        ValueError: If the records name other stages, or one did not end
            successfully.
    """
    current = load_pipeline_run_record(store.zw, record.run_id)
    if len(current.segments) > len(record.segments) or current.status != "paused":
        raise changed_since_read(record.run_id)
    finished = load_pipeline_stage_records(store.zw, record.run_id)
    paused_after = [stage for segment in record.segments for stage in segment.stages]
    if [stage.stage for stage in finished] != paused_after or any(
        not stage.complete or stage.status not in {"completed", "skipped"}
        for stage in finished
    ):
        raise ValueError(
            f"Pipeline run {record.run_id} cannot resume: its stage records no "
            f"longer match the stages that it ran before it paused; {_RERUN}"
        )
    return finished


def _recorded_outputs(record: PipelineStageRecord) -> dict[str, Any]:
    """Return a completed stage's outputs as its action returned them, by key."""
    outputs: dict[str, Any] = {
        item.output_key: item.artifact for item in record.outputs
    }
    outputs.update(
        (item.output_key, tuple(item.artifacts)) for item in record.output_lists
    )
    return outputs


def _replay_finished_stages(
    plan: RecipePlan,
    state: RunState,
    finished: Sequence[PipelineStageRecord],
    run_id: str,
) -> None:
    """Restore the state that the finished stages of a paused run left, computing and writing nothing.

    Each stage that completed runs again in an artifact forecast, where it
    reuses its complete artifacts and raises before it would plan a new
    one, and must return exactly the artifacts that its stage record names;
    each skipped stage runs its skip hook.

    Raises:
        ValueError: If a finished stage would compute an artifact again, or
            returns other artifacts than its record names.
    """
    for stage, record in zip(plan.stages, finished, strict=False):
        if record.status == "skipped":
            if stage.action is not None:
                raise ValueError(
                    f"Pipeline run {run_id} cannot resume: it skipped stage "
                    f"{stage.name!r}, which its settings now run; {_RERUN}"
                )
            if stage.on_skip is not None:
                stage.on_skip(state)
            continue
        if stage.action is None:
            raise ValueError(
                f"Pipeline run {run_id} cannot resume: it ran stage {stage.name!r}, "
                f"which its settings now skip; {_RERUN}"
            )
        expected = _recorded_outputs(record)
        try:
            with artifact_forecast_scope():
                outputs = tuple(stage.action(state))
        except ArtifactComputationRequired as signal:
            named = ", ".join(
                f"{ref.kind} {ref.artifact_id[:12]}"
                for value in expected.values()
                for ref in (value if isinstance(value, tuple) else (value,))
            )
            raise ValueError(
                f"Pipeline run {run_id} cannot resume: its finished stage "
                f"{stage.name!r} would compute again instead of reusing the "
                f"artifacts that its stage record names ({named}). "
                f"{signal.computation.describe()}. ds.pipeline.rerun(run) starts "
                "a new run that computes it again and reuses the finished stages "
                "that still match"
            ) from None
        replayed = dict(outputs)
        for key in sorted(set(expected) | set(replayed)):
            if expected.get(key) != replayed.get(key):
                raise ValueError(
                    f"Pipeline run {run_id} cannot resume: its finished stage "
                    f"{stage.name!r} recorded {key!r} as {expected.get(key)!r}, but "
                    f"the store now gives {replayed.get(key)!r}; ds.pipeline.rerun"
                    "(run) starts a new run that reuses the finished stages that "
                    "still match"
                )
        state.add(outputs)


def execute_recipe(
    store: Any,
    plan: RecipePlan,
    *,
    callback: PipelineCallback | None,
    cancel: CancellationToken | None,
    run_until: str | None = None,
    resumed: PipelineRunRecord | None = None,
) -> PipelineRun:
    """Execute a resolved recipe and return its durable run, completed or paused.

    The run installs cooperative signal handling on the main thread and
    listens to ``cancel``. A cancellation stops the run at its next safe
    checkpoint and raises ``PipelineCancelled``; a signal is handled as
    before and propagated once durable state is written. Each assay
    normalizes with the plan's normalizer while the stages run, and with the
    session's again once the run ends, however it ends.

    With ``run_until``, the run pauses once that stage has completed, while
    a stage that runs is left. With ``resumed``, a paused run that
    :func:`check_resumable_record` accepted and that ``plan`` rebuilds from
    its record, the call restores the finished stages from their artifacts
    and runs on under the same run ID.
    """
    after = 0
    if resumed is not None:
        _check_resumes_plan(resumed, plan)
        if resumed.requested_label is not None:
            try:
                ensure_pipeline_label_claimable(store.zw, resumed.requested_label)
            except ValueError as error:
                raise ValueError(
                    f"Pipeline run {resumed.run_id} cannot resume: {error}; "
                    "ds.pipeline.rerun(run, label=...) starts a new run under "
                    "another label that reuses its finished stages"
                ) from error
        after = sum(len(segment.stages) for segment in resumed.segments)
    stop = check_run_until(
        plan,
        run_until,
        after=after,
        run_id=None if resumed is None else resumed.run_id,
    )
    # The run pauses after the stage only while a stage that runs is left.
    pause_after = (
        stop
        if stop is not None
        and any(stage.action is not None for stage in plan.stages[stop + 1 :])
        else None
    )
    if (pause_after is not None or resumed is not None) and not (
        has_atomic_conditional_writes(store.zw.store)
    ):
        # Each resume claims its pause with an atomic write, so a run pauses
        # only where it can resume.
        raise RuntimeError(
            f"{'Resuming' if resumed is not None else 'Pausing'} a pipeline run "
            "requires a Zarr store with atomic set_if_not_exists support, which "
            "local, memory and object stores have; "
            + (
                "open the store as one of them to resume the run"
                if resumed is not None
                else "run without run_until on this store"
            )
        )
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
                with applied_normalizers(store, plan.normalizers):
                    result = _execute(
                        store,
                        plan,
                        callbacks,
                        context=context,
                        signal_guard_available=guard.available,
                        signal_guard_unavailable_reason=guard.unavailable_reason,
                        active=active,
                        pause_after=pause_after,
                        resumed=resumed,
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
                if finished.status in {"completed", "paused"}:
                    # A cancellation after the run completed or paused comes
                    # too late to stop it, so the caller receives the run.
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
        # The views read no times or segments; the record only needs valid ones.
        finished_at_ns=record.started_at_ns,
        status="completed",
        complete=True,
        outputs=outputs,
        fields=fields,
        segments=(),
    )
    PipelineAxisView(store, completed, axis="cells")
    PipelineAxisView(store, completed, axis="features")


def _pause(
    store: Any,
    run_id: str,
    events: PipelineEventEmitter,
    ledger: RunLedger,
    shutdown: dict[str, Any],
) -> PipelineRun:
    """Commit the run paused and return it.

    A failure to write the pause, such as a label that another run has
    committed, ends the run failed and raises ``PipelineExecutionError``
    for the step ``pause``.
    """
    try:
        shutdown_checkpoint()
        pause_pipeline_run_record(store.zw, run_id=run_id, shutdown=shutdown)
    except Exception as error:
        if is_interruption(error):
            # A cancelled Zarr call ends the run interrupted.
            raise
        ledger.terminate(error)
        raise PipelineExecutionError(run_id, "pause", error) from error
    events.emit("run_paused")
    run = open_pipeline_run(store, run_id=run_id)
    shutdown_checkpoint()
    return run


def _execute(
    store: Any,
    plan: RecipePlan,
    callbacks: Sequence[PipelineCallback],
    *,
    context: Context,
    signal_guard_available: bool,
    signal_guard_unavailable_reason: str | None,
    active: list[tuple[str, PipelineEventEmitter, RunLedger]],
    pause_after: int | None,
    resumed: PipelineRunRecord | None,
) -> PipelineRun:
    shutdown_checkpoint()
    from ... import __version__

    # The shared state reads no data, so it exists before the run record does.
    state = plan.start(store)
    shutdown = {
        "signalGuardAvailable": signal_guard_available,
        "unavailableReason": signal_guard_unavailable_reason,
    }
    if resumed is None:
        config = dict(plan.config)
        config["shutdown"] = shutdown
        record = create_pipeline_run_record(
            store.zw,
            recipe=plan.recipe,
            requested_label=plan.label,
            assay=plan.assay,
            config=config,
            stage_order=plan.stage_order,
            cytearc_version=__version__,
        )
        first = 0
    else:
        finished = _finished_stages(store, resumed)
        # The replay writes nothing, so a run that cannot resume stays paused.
        _replay_finished_stages(plan, state, finished, resumed.run_id)
        # The claim names the pause that the replay restored, so a process
        # that resumed the run since this one read it makes the claim fail.
        record = resume_pipeline_run_record(
            store.zw,
            run_id=resumed.run_id,
            segment=len(resumed.segments),
            shutdown=shutdown,
        )
        first = len(finished)
    events = PipelineEventEmitter(
        callbacks,
        context=context,
        run_id=record.run_id,
        recipe=plan.recipe,
        label=plan.label,
        stage_order=plan.stage_order,
    )
    ledger = RunLedger(
        store.zw,
        record.run_id,
        events,
        plan.summarize,
        workers=store.nthreads,
        memory_budget_bytes=store.memoryBytes,
        ordinal=first,
    )
    active.append((record.run_id, events, ledger))
    events.emit("run_started" if resumed is None else "run_resumed")
    for stage in plan.stages[first:]:
        if pause_after is not None and ledger.ordinal > pause_after:
            return _pause(store, record.run_id, events, ledger, shutdown)
        action = stage.action
        if action is None:
            if stage.on_skip is not None:
                try:
                    stage.on_skip(state)
                except Exception as error:
                    if is_interruption(error):
                        # A cancelled Zarr call ends the run interrupted.
                        raise
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
        ) -> Sequence[StageOutput]:
            outputs = tuple(action(state))
            state.add(outputs)
            return outputs

        ledger.run(stage.name, run_stage)

    outputs = tuple(
        PipelineOutputRecord(key=key, artifact=state.artifacts[key])
        for key in plan.output_keys
        if key in state.artifacts
    )
    output_lists = tuple(
        PipelineOutputListRecord(key=key, artifacts=state.artifact_lists[key])
        for key in plan.output_keys
        if key in state.artifact_lists
    )
    try:
        fields = plan.fields(state)
        _open_views(store, record, outputs, fields)
        shutdown_checkpoint()
        complete_pipeline_run_record(
            store.zw,
            run_id=record.run_id,
            outputs=outputs,
            output_lists=output_lists,
            fields=fields,
        )
    except Exception as error:
        if is_interruption(error):
            # A cancelled Zarr call ends the run interrupted.
            raise
        ledger.terminate(error)
        raise PipelineExecutionError(record.run_id, "finalize", error) from error
    # The run is complete and durable, so listeners learn it before a pending
    # shutdown, which can no longer change the run, is handled.
    events.emit("run_completed")
    run = open_pipeline_run(store, run_id=record.run_id)
    shutdown_checkpoint()
    return run
