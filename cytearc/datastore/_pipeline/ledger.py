import asyncio
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from ...storage.artifact_writer import ArtifactPlanReceipt, artifact_plan_scope
from ...storage.artifacts import ArtifactRef
from ...storage.pipeline_runs import (
    STAGE_WARNING_LIMIT,
    PipelineInterruptionRecord,
    PipelinePlanRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    bounded_record_text,
    fail_pipeline_run_record,
    finish_pipeline_stage_record,
    interrupt_pipeline_run_record,
    load_pipeline_run_record,
    load_pipeline_stage_record,
    start_pipeline_stage_record,
)
from ...utils.logging import RUN_ID_EXTRA, STAGE_EXTRA, collect_warnings, logger
from ...utils.process import ProcessTreeRssMeasurement, sample_process_tree_rss
from ...utils.progress import observe_progress
from ...utils.shutdown import ShutdownRequested, shutdown_checkpoint
from ..pipeline_run import PipelineExecutionError
from .events import PipelineEventEmitter
from .settings import parameter_value

type StageOutputs = Sequence[tuple[str, ArtifactRef]]
type StageSummarizer = Callable[[str, StageOutputs], Mapping[str, Any] | None]


def stage_metrics(
    *,
    wall_seconds: float,
    rss: ProcessTreeRssMeasurement,
) -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=wall_seconds,
        rss_baseline_bytes=rss.baseline_bytes,
        rss_peak_bytes=rss.peak_bytes,
        rss_incremental_peak_bytes=rss.incremental_peak_bytes,
        sample_interval_seconds=rss.sample_interval_seconds,
        sample_count=rss.sample_count,
        sampling_error_count=rss.sampling_error_count,
        rss_unavailable_reason=rss.unavailable_reason,
    )


def interruption_record(
    error: BaseException,
) -> PipelineInterruptionRecord | None:
    if isinstance(error, ShutdownRequested):
        request = error.request
        if request.signal_number is not None:
            kind = "signal"
        elif request.cancelled:
            kind = "cancelled"
        else:
            kind = "shutdown_request"
        return PipelineInterruptionRecord(
            kind=kind,
            message=bounded_record_text(request.reason),
            requested_at_ns=request.requested_at_ns,
            signal_number=request.signal_number,
            signal_name=request.signal_name,
        )
    if isinstance(error, KeyboardInterrupt):
        return PipelineInterruptionRecord(
            kind="keyboard_interrupt",
            message=bounded_record_text(str(error) or "keyboard interrupt"),
            requested_at_ns=time.time_ns(),
        )
    if isinstance(error, asyncio.CancelledError):
        return PipelineInterruptionRecord(
            kind="asyncio_cancelled",
            message=bounded_record_text(str(error) or "async operation cancelled"),
            requested_at_ns=time.time_ns(),
        )
    return None


def _as_interruption(error: BaseException) -> BaseException:
    """Return the interruption that an exception group holds, else ``error``.

    Workers that fail together raise an exception group. A group that holds a
    handled interruption leaf (``ShutdownRequested``, ``KeyboardInterrupt``, or
    ``asyncio.CancelledError``) ends its stage as the first such leaf.
    """
    if isinstance(error, BaseExceptionGroup):
        for inner in error.exceptions:
            leaf = _as_interruption(inner)
            if interruption_record(leaf) is not None:
                return leaf
    return error


def end_pipeline_run(root: Any, run_id: str, error: BaseException) -> bool:
    """End a running pipeline run with ``error``; return whether it wrote.

    A handled interruption marks the run interrupted and any other error marks
    it failed. A run that has already ended is left unchanged.
    """
    if load_pipeline_run_record(root, run_id).complete:
        return False
    interruption = interruption_record(error)
    if interruption is None:
        fail_pipeline_run_record(root, run_id=run_id, error=error)
    else:
        interrupt_pipeline_run_record(root, run_id=run_id, interruption=interruption)
    return True


@dataclass(frozen=True, slots=True)
class _StageOutcome:
    outputs: tuple[tuple[str, ArtifactRef], ...]
    completed: bool
    error: BaseException | None
    metrics: PipelineStageMetrics
    plans: tuple[ArtifactPlanReceipt, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any] | None


class RunLedger:
    """Run pipeline stages one at a time and record each one durably.

    Stages run on the calling thread in the persisted order, and each stage's
    record is terminal before the next stage starts, so its wall time, sampled
    memory, artifact receipt, warnings, and summary describe that stage alone.
    While a stage runs, CyteArc's log records carry the run and stage, its
    warnings are collected, and its progress bars become progress events.
    """

    __slots__ = ("current_stage", "events", "ordinal", "root", "run_id", "summarize")

    def __init__(
        self,
        root: Any,
        run_id: str,
        events: PipelineEventEmitter,
        summarize: StageSummarizer | None = None,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.ordinal = 0
        self.events = events
        self.summarize = summarize
        # The stage that started last, which an interruption names.
        self.current_stage: str | None = None

    def _execute_stage(
        self,
        stage: str,
        ordinal: int,
        action: Callable[[], StageOutputs],
        wall_started: float,
    ) -> _StageOutcome:
        outputs: tuple[tuple[str, ArtifactRef], ...] = ()
        completed = False
        summary: Mapping[str, Any] | None = None
        caught: BaseException | None = None
        collected: list[str] = []
        with sample_process_tree_rss() as read_rss:
            with ExitStack() as context:
                context.enter_context(
                    logger.contextualize(
                        **{RUN_ID_EXTRA: self.run_id, STAGE_EXTRA: stage}
                    )
                )
                collected = context.enter_context(collect_warnings())
                if self.events.active:
                    # The compute pool's workers share the stage's context, but
                    # callbacks run on the run's thread, so only that thread's
                    # bars report progress.
                    stage_thread = threading.get_ident()

                    def report(
                        bar: int,
                        description: str | None,
                        done: int,
                        total: int | None,
                    ) -> None:
                        if threading.get_ident() == stage_thread:
                            self.events.progress(
                                stage, ordinal, bar, description, done, total
                            )

                    context.enter_context(observe_progress(report))
                plans = context.enter_context(artifact_plan_scope())
                try:
                    outputs = tuple(action())
                    completed = True
                    # The summary reads the stage's artifacts, so its time,
                    # memory, and warnings belong to the stage.
                    summary = self._summary(stage, outputs)
                    shutdown_checkpoint()
                except BaseException as error:
                    caught = error
        return _StageOutcome(
            outputs=outputs,
            completed=completed,
            error=caught,
            metrics=stage_metrics(
                wall_seconds=time.perf_counter() - wall_started,
                rss=read_rss(),
            ),
            plans=tuple(plans),
            # The warnings that a stage record keeps; its events carry the same.
            warnings=tuple(
                bounded_record_text(str(warning)) for warning in collected if warning
            )[:STAGE_WARNING_LIMIT],
            summary=summary,
        )

    def _records(
        self,
        outputs: StageOutputs,
        plans: Sequence[ArtifactPlanReceipt],
    ) -> tuple[PipelineStageOutputRecord, ...]:
        dispositions: dict[ArtifactRef, set[str]] = {}
        for plan in plans:
            dispositions.setdefault(plan.ref, set()).add(plan.disposition)
        missing = [key for key, ref in outputs if ref not in dispositions]
        if missing:
            raise RuntimeError(
                "Pipeline stage outputs were not observed by artifact planning: "
                f"{missing!r}"
            )
        return tuple(
            PipelineStageOutputRecord(
                output_key=key,
                artifact=ref,
                reused=dispositions[ref] == {"reused"},
            )
            for key, ref in outputs
        )

    @staticmethod
    def _plans(
        plans: Sequence[ArtifactPlanReceipt],
    ) -> tuple[PipelinePlanRecord, ...]:
        return tuple(
            PipelinePlanRecord(
                operation=plan.operation,
                ref=plan.ref,
                disposition=plan.disposition,
            )
            for plan in plans
        )

    def _summary(self, stage: str, outputs: StageOutputs) -> Mapping[str, Any] | None:
        """Return a completed stage's diagnostics; a failure only logs a warning.

        A summary is recorded as JSON values, so one that does not convert,
        such as one holding infinity, is left out rather than failing the run.
        """
        if self.summarize is None:
            return None
        try:
            summary = self.summarize(stage, outputs)
            if summary is None:
                return None
            converted = parameter_value(summary, f"{stage} summary")
            assert isinstance(converted, dict)
            return converted
        except Exception as error:
            logger.warning(f"Could not summarize pipeline stage {stage}: {error}")
            return None

    def terminate(self, error: BaseException) -> bool:
        """End the run with ``error`` unless it already ended."""
        return end_pipeline_run(self.root, self.run_id, error)

    def interrupt_pending(self, error: BaseException, stage: str | None) -> None:
        if interruption_record(error) is None:
            raise TypeError("error is not a handled pipeline interruption")
        self._end_interrupted(error)
        self.events.emit("run_interrupted", stage=stage, error=error)

    def _end_interrupted(self, error: BaseException) -> None:
        """End the run interrupted; a failed write is logged, never raised.

        The interruption, not a storage error while recording it, is what the
        caller receives, so a cancellation or keyboard interrupt still stops
        its caller.
        """
        try:
            self.terminate(error)
        except Exception:
            logger.exception(
                f"Could not record the interruption of pipeline run {self.run_id}"
            )

    @staticmethod
    def _fallback_metrics(wall_started: float) -> PipelineStageMetrics:
        with sample_process_tree_rss() as read_rss:
            pass
        return stage_metrics(
            wall_seconds=time.perf_counter() - wall_started,
            rss=read_rss(),
        )

    def _record_interrupted_stage(
        self,
        *,
        ordinal: int,
        error: BaseException,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        """Mark a running stage and the run interrupted by ``error``.

        The run ends even when the stage record cannot be written, so it never
        stays running. A storage error while either is written is logged, so
        the interruption reaches the caller.
        """
        interruption = interruption_record(error)
        if interruption is None:
            raise TypeError("error is not a handled pipeline interruption")
        try:
            current_stage = load_pipeline_stage_record(self.root, self.run_id, ordinal)
            if current_stage.status == "running":
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=ordinal,
                    status="interrupted",
                    plans=plans,
                    metrics=metrics,
                    interruption=interruption,
                    warnings=warnings,
                )
        except Exception:
            logger.exception(
                f"Could not record the interruption of stage {ordinal} of pipeline "
                f"run {self.run_id}"
            )
        self._end_interrupted(error)

    def _finish_interrupted(
        self,
        *,
        stage: str,
        ordinal: int,
        error: BaseException,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        self._record_interrupted_stage(
            ordinal=ordinal,
            error=error,
            metrics=metrics,
            plans=plans,
            warnings=warnings,
        )
        self.events.emit(
            "stage_interrupted",
            stage=stage,
            ordinal=ordinal,
            metrics=metrics.to_dict(),
            warnings=warnings,
            error=error,
        )
        self.events.emit("run_interrupted", stage=stage, ordinal=ordinal, error=error)

    def _finish_failed(
        self,
        *,
        stage: str,
        ordinal: int,
        error: Exception,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        try:
            current_stage = load_pipeline_stage_record(self.root, self.run_id, ordinal)
            if current_stage.status == "running":
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=ordinal,
                    status="failed",
                    plans=plans,
                    metrics=metrics,
                    error=error,
                    warnings=warnings,
                )
        finally:
            self.terminate(error)
        self.events.emit(
            "stage_failed",
            stage=stage,
            ordinal=ordinal,
            metrics=metrics.to_dict(),
            warnings=warnings,
            error=error,
        )

    def skip(self, stage: str) -> None:
        """Record a stage that the run does not compute.

        A skip does no work and emits no ``stage_started``, so it checks for
        shutdown only before it writes, and an interruption while it is
        recorded ends the run between stages, naming no stage.
        """
        shutdown_checkpoint()
        wall_started = time.perf_counter()
        ordinal = self.ordinal
        stage_started = False
        metrics: PipelineStageMetrics | None = None
        try:
            start_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
                stage=stage,
            )
            stage_started = True
            with sample_process_tree_rss() as read_rss:
                pass
            metrics = stage_metrics(
                wall_seconds=time.perf_counter() - wall_started,
                rss=read_rss(),
            )
            finish_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
                status="skipped",
                metrics=metrics,
            )
        except BaseException as error:
            interruption = interruption_record(error)
            if interruption is not None:
                if stage_started:
                    self._record_interrupted_stage(
                        ordinal=ordinal,
                        error=error,
                        metrics=metrics or self._fallback_metrics(wall_started),
                    )
                self.interrupt_pending(error, None)
                raise
            if not isinstance(error, Exception):
                raise
            if stage_started:
                self._finish_failed(
                    stage=stage,
                    ordinal=ordinal,
                    error=error,
                    metrics=metrics or self._fallback_metrics(wall_started),
                )
            else:
                self.terminate(error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1
        self.events.emit(
            "stage_skipped",
            stage=stage,
            ordinal=ordinal,
            metrics=metrics.to_dict(),
        )

    def run(
        self,
        stage: str,
        action: Callable[[], StageOutputs],
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        shutdown_checkpoint()
        wall_started = time.perf_counter()
        ordinal = self.ordinal
        try:
            start_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
                stage=stage,
            )
        except Exception as error:
            self.terminate(error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1
        self.current_stage = stage
        logger.info(f"Running pipeline stage: {stage.replace('_', ' ')}")
        self.events.emit("stage_started", stage=stage, ordinal=ordinal)
        outputs = self._finish(
            stage,
            ordinal,
            self._execute_stage(stage, ordinal, action, wall_started),
        )
        # Between stages no stage is running, so an interruption names none.
        self.current_stage = None
        return outputs

    def _finish(
        self,
        stage: str,
        ordinal: int,
        outcome: _StageOutcome,
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        """Record a stage's outcome and return its outputs.

        An ``Exception`` from the stage fails the stage and the run and raises
        ``PipelineExecutionError``. A handled interruption, or the first one
        that an exception group holds, ends the run as interrupted and is
        raised; the stage is recorded completed if its action returned, else
        interrupted. Any other ``BaseException``, such as ``SystemExit``, is
        raised without recording the stage or ending the run.
        """
        plan_records = self._plans(outcome.plans)
        caught = None if outcome.error is None else _as_interruption(outcome.error)
        summary = outcome.summary
        records: tuple[PipelineStageOutputRecord, ...] = ()
        if caught is None:
            try:
                records = self._records(outcome.outputs, outcome.plans)
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=ordinal,
                    status="completed",
                    outputs=records,
                    plans=plan_records,
                    metrics=outcome.metrics,
                    summary=summary,
                    warnings=outcome.warnings,
                )
            except Exception as error:
                caught = error
        if caught is None:
            self._emit_completed(stage, ordinal, outcome, records, summary)
            logger.info(f"Completed pipeline stage: {stage.replace('_', ' ')}")
            return outcome.outputs
        interruption = interruption_record(caught)
        if interruption is not None and outcome.completed:
            # The stage's work finished before the interruption was seen, so
            # the stage completed and the run stops between stages.
            try:
                records = self._records(outcome.outputs, outcome.plans)
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=ordinal,
                    status="completed",
                    outputs=records,
                    plans=plan_records,
                    metrics=outcome.metrics,
                    summary=summary,
                    warnings=outcome.warnings,
                )
            except Exception as error:
                self._finish_failed(
                    stage=stage,
                    ordinal=ordinal,
                    error=error,
                    metrics=outcome.metrics,
                    plans=plan_records,
                    warnings=outcome.warnings,
                )
                raise PipelineExecutionError(self.run_id, stage, error) from error
            self._emit_completed(stage, ordinal, outcome, records, summary)
            self.current_stage = None
            self.interrupt_pending(caught, None)
            raise caught
        if interruption is not None:
            self._finish_interrupted(
                stage=stage,
                ordinal=ordinal,
                error=caught,
                metrics=outcome.metrics,
                plans=plan_records,
                warnings=outcome.warnings,
            )
        elif isinstance(caught, Exception):
            self._finish_failed(
                stage=stage,
                ordinal=ordinal,
                error=caught,
                metrics=outcome.metrics,
                plans=plan_records,
                warnings=outcome.warnings,
            )
            raise PipelineExecutionError(self.run_id, stage, caught) from caught
        raise caught

    def _emit_completed(
        self,
        stage: str,
        ordinal: int,
        outcome: _StageOutcome,
        records: Sequence[PipelineStageOutputRecord],
        summary: Mapping[str, Any] | None,
    ) -> None:
        self.events.emit(
            "stage_completed",
            stage=stage,
            ordinal=ordinal,
            metrics=outcome.metrics.to_dict(),
            summary=summary,
            outputs=outcome.outputs,
            reused=tuple(record.output_key for record in records if record.reused),
            warnings=outcome.warnings,
        )
