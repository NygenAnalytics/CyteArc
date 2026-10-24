import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
from typing import Any, Literal

from ...storage.pipeline_runs import bounded_record_text
from ...storage.refs import ArtifactRef
from ...utils.logging import UNCOLLECTED_EXTRA, logger

type PipelineEventKind = Literal[
    "run_started",
    "stage_started",
    "stage_progress",
    "stage_completed",
    "stage_skipped",
    "stage_failed",
    "stage_interrupted",
    "run_completed",
    "run_failed",
    "run_interrupted",
]

# A progress bar emits at most one progress event in this many seconds,
# besides its first update and the update that reaches its total.
PROGRESS_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class PipelineProgress:
    """Progress of the bounded work a stage is doing.

    Attributes:
        description: What the work is, such as ``"Harmony"``, or None.
        completed: Units of the work done so far.
        total: Units the work has in all, or None when it is not known.
    """

    description: str | None
    completed: int
    total: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "completed": self.completed,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One step of a pipeline run, as listeners receive it.

    A run emits ``run_started``, then each stage's events in order, and ends
    with ``run_completed``, ``run_failed``, or ``run_interrupted``. A stage
    emits ``stage_skipped`` alone, or ``stage_started``, any number of
    ``stage_progress`` events, and one of ``stage_completed``,
    ``stage_failed``, or ``stage_interrupted``. An interruption that a stage
    notices just after its work finished records the stage as completed, so
    ``stage_completed`` precedes a ``run_interrupted`` that names no stage.

    Attributes:
        kind: What happened. Run events are ``run_started``, ``run_completed``,
            ``run_failed``, and ``run_interrupted``; stage events are
            ``stage_started``, ``stage_progress``, ``stage_completed``,
            ``stage_skipped``, ``stage_failed``, and ``stage_interrupted``.
        run_id: The durable run identifier.
        recipe: The recipe, such as ``"rna"``.
        label: The label the run requested, or None.
        stage: The stage that the event is about. Stage events name their
            stage, and ``run_started`` and ``run_completed`` name none.
            ``run_failed`` always names one: the stage that failed, or
            ``"finalize"`` when completing the run record failed.
            ``run_interrupted`` names the stage whose work the interruption
            cut short, and none when the run stopped between stages, while a
            skipped stage was recorded, or just after a stage's work finished.
        ordinal: The stage's position in ``run.report()["run"]["stageOrder"]``,
            or None when the event names no stage of that order.
        stages: How many stages the run has, skipped stages included.
        time_ns: When the event happened, in nanoseconds since the epoch.
        progress: The stage's progress, for ``stage_progress`` events.
        metrics: The stage's wall time and sampled memory, as its record holds
            them, for terminal stage events.
        summary: The stage's diagnostics, as its record holds them, for
            ``stage_completed`` events.
        outputs: The ``(key, ref)`` pairs a completed stage produced.
        reused: The output keys whose artifacts were reused, not computed.
        warnings: The warnings logged while the stage ran.
        error: The exception, for failed and interrupted events.
    """

    kind: PipelineEventKind
    run_id: str
    recipe: str
    label: str | None
    stage: str | None
    ordinal: int | None
    stages: int
    time_ns: int
    progress: PipelineProgress | None = None
    metrics: Mapping[str, Any] | None = None
    summary: Mapping[str, Any] | None = None
    outputs: tuple[tuple[str, ArtifactRef], ...] = ()
    reused: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: BaseException | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the event as JSON-compatible values, for a UI or a log."""
        error = self.error
        return {
            "kind": self.kind,
            "runId": self.run_id,
            "recipe": self.recipe,
            "label": self.label,
            "stage": self.stage,
            "ordinal": self.ordinal,
            "stages": self.stages,
            "timeNs": self.time_ns,
            "progress": None if self.progress is None else self.progress.to_dict(),
            "metrics": None if self.metrics is None else dict(self.metrics),
            "summary": None if self.summary is None else dict(self.summary),
            "outputs": [
                {"key": key, "artifact": ref.to_dict(), "reused": key in self.reused}
                for key, ref in self.outputs
            ],
            "warnings": list(self.warnings),
            "error": (
                None
                if error is None
                else {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": bounded_record_text(str(error)),
                }
            ),
        }


type PipelineCallback = Callable[[PipelineEvent], None]

_LISTENERS: ContextVar[tuple[PipelineCallback, ...]] = ContextVar(
    "cytearc_pipeline_listeners",
    default=(),
)


@contextmanager
def pipeline_listener(listener: PipelineCallback) -> Iterator[PipelineCallback]:
    """Send the events of every pipeline run started in this context to ``listener``.

    The registration is context-local: it covers the code in the ``with``
    block, on this thread or in this asyncio task, and nests. A run reads the
    listeners when it starts and calls them, after its own ``callback``, on the
    thread that runs it, so a listener that does slow work should hand each
    event to a queue. A listener's exception is logged and never stops the run.

    Example:
        >>> with cytearc.pipeline_listener(events.append):  # doctest: +SKIP
        ...     ds.pipeline.rna(label="rna")
    """
    if not callable(listener):
        raise TypeError("listener must be callable")
    token = _LISTENERS.set((*_LISTENERS.get(), listener))
    try:
        yield listener
    finally:
        _LISTENERS.reset(token)


def run_callbacks(callback: PipelineCallback | None) -> tuple[PipelineCallback, ...]:
    """Return a run's callback followed by the listeners of the current context."""
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    own = () if callback is None else (callback,)
    return (*own, *_LISTENERS.get())


class PipelineEventEmitter:
    """Deliver one run's events to its callbacks in order.

    Each callback receives every event of the run on the thread that runs it,
    in a copy of ``context``: the caller's context before the run began, or
    the context in which the emitter was made. A callback's work therefore
    never reaches a stage's warnings, artifact receipts, log context, or
    progress, and the run's shutdown requests never interrupt it. A
    callback's exception, ``BaseException`` included, is logged and never
    changes the run.
    """

    __slots__ = (
        "_callbacks",
        "_context",
        "_last_progress",
        "_label",
        "_recipe",
        "_run_id",
        "_stage_order",
    )

    def __init__(
        self,
        callbacks: Sequence[PipelineCallback],
        *,
        run_id: str,
        recipe: str,
        label: str | None,
        stage_order: Sequence[str],
        context: Context | None = None,
    ) -> None:
        self._callbacks = tuple(callbacks)
        self._context = copy_context() if context is None else context
        self._run_id = run_id
        self._recipe = recipe
        self._label = label
        self._stage_order = tuple(stage_order)
        # Each bar's last progress event and whether that event finished it.
        self._last_progress: dict[tuple[str, int], tuple[float, bool]] = {}

    @property
    def active(self) -> bool:
        return bool(self._callbacks)

    def emit(
        self,
        kind: PipelineEventKind,
        *,
        stage: str | None = None,
        ordinal: int | None = None,
        progress: PipelineProgress | None = None,
        metrics: Mapping[str, Any] | None = None,
        summary: Mapping[str, Any] | None = None,
        outputs: Sequence[tuple[str, ArtifactRef]] = (),
        reused: Sequence[str] = (),
        warnings: Sequence[str] = (),
        error: BaseException | None = None,
    ) -> None:
        if not self._callbacks:
            return
        if stage is not None and ordinal is None and stage in self._stage_order:
            # A run event may name a step outside the stage order, "finalize".
            ordinal = self._stage_order.index(stage)
        event = PipelineEvent(
            kind=kind,
            run_id=self._run_id,
            recipe=self._recipe,
            label=self._label,
            stage=stage,
            ordinal=ordinal,
            stages=len(self._stage_order),
            time_ns=time.time_ns(),
            progress=progress,
            metrics=None if metrics is None else dict(metrics),
            summary=None if summary is None else dict(summary),
            outputs=tuple(outputs),
            reused=tuple(reused),
            warnings=tuple(warnings),
            error=error,
        )
        for callback in self._callbacks:
            try:
                self._context.copy().run(callback, event)
            except BaseException:
                # A callback's failure belongs to its host, not to the run's
                # stage records.
                logger.bind(**{UNCOLLECTED_EXTRA: True}).exception(
                    f"Pipeline callback failed while handling {kind}"
                    + ("" if stage is None else f" for {stage}")
                )

    def progress(
        self,
        stage: str,
        ordinal: int,
        bar: int,
        description: str | None,
        completed: int,
        total: int | None,
    ) -> None:
        """Emit a throttled ``stage_progress`` event for one progress bar.

        A bar's first update and the update that reaches its total are always
        emitted; any other comes at least ``PROGRESS_INTERVAL_SECONDS`` after
        the bar's previous event.
        """
        if not self._callbacks:
            return
        now = time.monotonic()
        key = (stage, bar)
        finished = total is not None and completed >= total
        last, was_finished = self._last_progress.get(key, (None, False))
        if (
            last is not None
            and now - last < PROGRESS_INTERVAL_SECONDS
            and (was_finished or not finished)
        ):
            return
        self._last_progress[key] = (now, was_finished or finished)
        self.emit(
            "stage_progress",
            stage=stage,
            ordinal=ordinal,
            progress=PipelineProgress(
                description=description,
                completed=int(completed),
                total=None if total is None else int(total),
            ),
        )
