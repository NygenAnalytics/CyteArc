"""The plan of a recipe call: what each stage would reuse or compute, and why.

A plan runs each stage's action inside an artifact forecast, where planning
an artifact reuses as a run does but raises before it plans one to compute.
A stage whose every artifact exists, complete and current, is reused with
the run's own refs; the first artifact that it would compute ends the stage
with the reason. The stages that read what a computed stage would produce
cannot learn their identities, which depend on artifacts that do not exist
yet, so the plan reports them computed, waiting on that stage.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ...storage.artifact_writer import (
    ArtifactComputationRequired,
    artifact_forecast_scope,
)
from ...storage.artifacts import ArtifactRef
from ...storage.pipeline_runs import (
    PipelineStageMetrics,
    ensure_pipeline_label_claimable,
    list_pipeline_run_records,
    load_pipeline_stage_records,
)
from ...utils.logging import logger
from .executor import COORDINATES, FEATURE_NAMES, PlannedStage, RecipePlan, RunState

type StagePlanStatus = Literal["reused", "computed", "skipped"]
type StagePlanReason = Literal[
    "absent",
    "changed",
    "superseded",
    "incomplete",
    "missing_diagnostics",
    "invalid",
    "invalidate_cache",
    "upstream",
]


@dataclass(frozen=True, slots=True)
class PipelineStageEstimate:
    """What a stage took when the latest earlier run of its recipe computed it.

    A past measurement, not a model: that run's cells, settings, and machine
    may differ from the planned call's, and ``resources`` records the machine.

    Attributes:
        run_id: The run whose stage record holds the measurement.
        wall_seconds: The stage's wall time.
        rss_peak_bytes: The sampled peak RSS of the process tree, or None
            where RSS was not sampled.
        rss_incremental_peak_bytes: That peak above the RSS when the stage
            started, or None.
        cpu_seconds: The CPU time of the process and its child processes
            during the stage, or None from a record that lacks it.
        threads_peak: The most OS threads that the process tree held, or
            None.
        rss_high_water_bytes: The process's peak RSS when the stage raised
            the kernel's mark of it, which includes peaks that no sample saw;
            None when the stage stayed under an earlier mark of its process,
            such as one inherited from the process that started it, or where
            the mark was not read.
        network_rx_bytes: Bytes that the network namespace received during
            the stage, or None.
        network_tx_bytes: Bytes that it sent, or None.
        resources: What the process had: ``workers``, ``memoryBudgetBytes``,
            ``cpusAvailable``, ``cpuQuotaCores`` and ``memoryLimitBytes``, as
            the stage record holds them, or None from a record that lacks
            them.
    """

    run_id: str
    wall_seconds: float
    rss_peak_bytes: int | None
    rss_incremental_peak_bytes: int | None
    cpu_seconds: float | None = None
    threads_peak: int | None = None
    rss_high_water_bytes: int | None = None
    network_rx_bytes: int | None = None
    network_tx_bytes: int | None = None
    # A dict cannot be hashed, so the estimate's hash leaves it out.
    resources: dict[str, Any] | None = field(default=None, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "wallSeconds": self.wall_seconds,
            "rssPeakBytes": self.rss_peak_bytes,
            "rssIncrementalPeakBytes": self.rss_incremental_peak_bytes,
            "cpuSeconds": self.cpu_seconds,
            "threadsPeak": self.threads_peak,
            "rssHighWaterBytes": self.rss_high_water_bytes,
            "networkRxBytes": self.network_rx_bytes,
            "networkTxBytes": self.network_tx_bytes,
            "resources": None if self.resources is None else dict(self.resources),
        }


@dataclass(frozen=True, slots=True)
class PipelineStagePlan:
    """What one stage of a planned call would do.

    Attributes:
        stage: The stage, as the run's ``stageOrder`` names it.
        ordinal: Its position in that order.
        status: ``"reused"`` when every artifact the stage plans exists,
            complete and current; ``"computed"`` when the stage would compute
            one; ``"skipped"`` when the call turns the stage off.
        reason: Why a computed stage computes: ``"absent"``, the store holds
            no artifact of the operation; ``"changed"``, the closest one was
            computed with other parameters or inputs; ``"superseded"``, an
            operation revision supersedes the stored one; ``"incomplete"``,
            the stored one was interrupted; ``"missing_diagnostics"``, the
            stored one lacks the diagnostics that its operation saves;
            ``"invalid"``, the stored one fails its payload checks;
            ``"upstream"``, the stage reads what a stage before it would
            compute. None for other stages.
        detail: The reason in one sentence, or None.
        operation: The operation whose artifact the stage would compute
            first, or None.
        outputs: The ``(key, ref)`` pairs that a reused stage would record,
            the run's own refs; empty for other stages. An output that holds
            several artifacts, such as ``leiden``, gives one pair for each,
            in order.
        existing: The stored artifact that the reason concerns: the one that
            a revision supersedes, that was interrupted, that lacks
            diagnostics, or that fails its checks, or the closest one of the
            operation. None otherwise.
        changed_parameters: For ``"changed"``, the operation's parameters in
            which the closest artifact differs, such as ``("top_n",)``.
        changed_inputs: For ``"changed"``, the inputs in which it differs.
        upstream: For ``"upstream"``, the earlier stages that the stage waits
            on, in stage order.
        estimate: For a computed stage, what it took when the latest earlier
            run of the same recipe and assay computed it, or None.
    """

    stage: str
    ordinal: int
    status: StagePlanStatus
    reason: StagePlanReason | None = None
    detail: str | None = None
    operation: str | None = None
    outputs: tuple[tuple[str, ArtifactRef], ...] = ()
    existing: ArtifactRef | None = None
    changed_parameters: tuple[str, ...] = ()
    changed_inputs: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()
    estimate: PipelineStageEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stage's plan as JSON-compatible values."""
        return {
            "stage": self.stage,
            "ordinal": self.ordinal,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "operation": self.operation,
            "outputs": [
                {"key": key, "artifact": ref.to_dict()} for key, ref in self.outputs
            ],
            "existing": None if self.existing is None else self.existing.to_dict(),
            "changedParameters": list(self.changed_parameters),
            "changedInputs": list(self.changed_inputs),
            "upstream": list(self.upstream),
            "estimate": None if self.estimate is None else self.estimate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """What a recipe call would reuse and compute, stage by stage.

    Attributes:
        recipe: The recipe, such as ``"rna"``.
        assay: The assay that the run would analyze.
        label: The requested label, or None.
        config: The settings that the run would record under ``config``.
        stages: Every stage in run order, skipped stages included.
    """

    recipe: str
    assay: str
    label: str | None
    config: dict[str, Any]
    stages: tuple[PipelineStagePlan, ...]

    def __getitem__(self, stage: str) -> PipelineStagePlan:
        """Return the plan of the stage named ``stage``."""
        for planned in self.stages:
            if planned.stage == stage:
                return planned
        raise KeyError(f"The {self.recipe} plan has no stage {stage!r}")

    @property
    def computed(self) -> tuple[str, ...]:
        """The stages that would compute an artifact, in run order."""
        return tuple(stage.stage for stage in self.stages if stage.status == "computed")

    @property
    def reused(self) -> tuple[str, ...]:
        """The stages that would reuse every artifact, in run order."""
        return tuple(stage.stage for stage in self.stages if stage.status == "reused")

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as JSON-compatible values."""
        return {
            "recipe": self.recipe,
            "assay": self.assay,
            "label": self.label,
            "config": self.config,
            "stages": [stage.to_dict() for stage in self.stages],
        }


class _StageArtifacts[V](dict[str, V]):
    """The run's artifacts, or artifact lists, as one stage of a plan sees them.

    The stage reads only the keys it declares as inputs and adds only those
    it declares as outputs, so a declaration that misses a key fails loudly
    instead of hiding a stage that would wait on another.
    """

    def __init__(self, artifacts: dict[str, V], stage: PlannedStage) -> None:
        super().__init__(artifacts)
        self._stage = stage

    def _require(self, key: str, declared: tuple[str, ...], role: str) -> None:
        if key not in declared:
            raise RuntimeError(
                f"Pipeline stage {self._stage.name!r} uses {key!r}, which it does "
                f"not declare among its {role}"
            )

    def __getitem__(self, key: str) -> V:
        self._require(key, self._stage.inputs, "inputs")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self._require(key, self._stage.inputs, "inputs")
        return super().get(key, default)

    def __setitem__(self, key: str, value: V) -> None:
        self._require(key, self._stage.outputs, "outputs")
        super().__setitem__(key, value)


def _run_declared[T](
    stage: PlannedStage, state: RunState, step: Callable[[RunState], T]
) -> T:
    """Run a stage's action or skip hook on the artifacts that it declares.

    The state keeps the step's additions only when the step returns.
    """
    artifacts, lists = state.artifacts, state.artifact_lists
    seen = _StageArtifacts(artifacts, stage)
    seen_lists = _StageArtifacts(lists, stage)
    state.artifacts, state.artifact_lists = seen, seen_lists
    try:
        result = step(state)
    finally:
        state.artifacts, state.artifact_lists = artifacts, lists
    artifacts.clear()
    artifacts.update(seen)
    lists.clear()
    lists.update(seen_lists)
    return result


def _set_pending(
    state: RunState,
    pending: dict[str, tuple[str, ...]],
    keys: Sequence[str],
    owners: tuple[str, ...],
) -> None:
    """Mark what a stage would add as waiting on ``owners``, which would compute it.

    The value goes from the state, so that no later stage reads an earlier
    stage's value in its place.
    """
    for key in keys:
        pending[key] = owners
        state.artifacts.pop(key, None)
        state.artifact_lists.pop(key, None)
        if key == COORDINATES:
            state.coordinates = None
        elif key == FEATURE_NAMES:
            state.feature_names = None
        else:
            state.extras.pop(key, None)


def _waiting_on(
    stage: PlannedStage,
    pending: dict[str, tuple[str, ...]],
    order: dict[str, int],
) -> tuple[str, ...]:
    """Return the stages whose computed values ``stage`` reads, in stage order."""
    owners = {owner for key in stage.inputs for owner in pending.get(key, ())}
    return tuple(sorted(owners, key=order.__getitem__))


def _estimate(run_id: str, metrics: PipelineStageMetrics) -> PipelineStageEstimate:
    """Return the estimate that a completed stage's metrics give.

    The process's peak RSS counts only when the stage raised it: a stage
    whose peak stayed under an earlier stage's mark has only its sampled
    peak.
    """
    start = metrics.rss_high_water_start_bytes
    end = metrics.rss_high_water_end_bytes
    return PipelineStageEstimate(
        run_id=run_id,
        wall_seconds=metrics.wall_seconds,
        rss_peak_bytes=metrics.rss_peak_bytes,
        rss_incremental_peak_bytes=metrics.rss_incremental_peak_bytes,
        cpu_seconds=metrics.cpu_seconds,
        threads_peak=metrics.threads_peak,
        rss_high_water_bytes=(
            end if start is not None and end is not None and end > start else None
        ),
        network_rx_bytes=metrics.network_rx_bytes,
        network_tx_bytes=metrics.network_tx_bytes,
        resources=None if metrics.resources is None else metrics.resources.to_dict(),
    )


def _estimates(
    store: Any, plan: RecipePlan, stages: set[str]
) -> dict[str, PipelineStageEstimate]:
    """Return what each stage took when the latest run that computed it did.

    Only runs of the plan's recipe and assay count, and only a completed stage
    that created an artifact, so a stage that reused everything does not pass
    for a computation. A run whose stage records cannot be read is skipped
    with a warning, as listing runs skips one whose run record cannot be read.
    """
    found: dict[str, PipelineStageEstimate] = {}
    if not stages:
        return found
    for run in list_pipeline_run_records(store.zw, limit=2**31 - 1):
        wanted = (stages - found.keys()) & set(run.stage_order)
        if run.recipe != plan.recipe or run.assay != plan.assay or not wanted:
            continue
        try:
            records = load_pipeline_stage_records(store.zw, run.run_id)
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(
                f"Skipped the stage records of pipeline run {run.run_id} for "
                f"plan estimates: {error}"
            )
            continue
        for record in records:
            if (
                record.stage in wanted
                and record.stage not in found
                and record.status == "completed"
                and record.metrics is not None
                and any(item.disposition == "created" for item in record.plans)
            ):
                found[record.stage] = _estimate(run.run_id, record.metrics)
        if found.keys() >= stages:
            break
    return found


def plan_recipe(store: Any, plan: RecipePlan) -> PipelinePlan:
    """Report what executing ``plan`` would reuse and compute, writing nothing.

    The label is checked as a run checks it before it writes. Each stage then
    runs in an artifact forecast until it would compute an artifact; a stage
    that reads a value that an earlier computed stage would produce is not
    run and waits on that stage.
    """
    if plan.label is not None:
        ensure_pipeline_label_claimable(store.zw, plan.label)
    state = plan.start(store)
    order = {stage.name: ordinal for ordinal, stage in enumerate(plan.stages)}
    # Each value that a stage of this plan would compute, by the stages it
    # waits on: the stage that would compute it, or for a skipped stage's
    # aliases, the stages that would compute what it aliases.
    pending: dict[str, tuple[str, ...]] = {}
    planned: list[PipelineStagePlan] = []
    for ordinal, stage in enumerate(plan.stages):
        waiting = _waiting_on(stage, pending, order)
        action = stage.action
        if action is None:
            if waiting:
                _set_pending(state, pending, stage.outputs, waiting)
            elif stage.on_skip is not None:
                _run_declared(stage, state, stage.on_skip)
            planned.append(PipelineStagePlan(stage.name, ordinal, "skipped"))
            continue
        if waiting:
            _set_pending(state, pending, stage.outputs, (stage.name,))
            planned.append(
                PipelineStagePlan(
                    stage.name,
                    ordinal,
                    "computed",
                    reason="upstream",
                    detail=(
                        "It reads what "
                        + ", ".join(waiting)
                        + " would compute, so its identity is known only after "
                        "they run"
                    ),
                    upstream=waiting,
                )
            )
            continue
        try:
            with artifact_forecast_scope():
                outputs = tuple(_run_declared(stage, state, action))
        except ArtifactComputationRequired as signal:
            computation = signal.computation
            _set_pending(state, pending, stage.outputs, (stage.name,))
            planned.append(
                PipelineStagePlan(
                    stage.name,
                    ordinal,
                    "computed",
                    reason=computation.reason,
                    detail=computation.describe(),
                    operation=computation.operation,
                    existing=computation.existing,
                    changed_parameters=computation.changed_parameters,
                    changed_inputs=computation.changed_inputs,
                )
            )
            continue
        undeclared = [key for key, _ref in outputs if key not in stage.outputs]
        if undeclared:
            raise RuntimeError(
                f"Pipeline stage {stage.name!r} returned {undeclared!r}, which it "
                "does not declare among its outputs"
            )
        state.add(outputs)
        for key in stage.outputs:
            pending.pop(key, None)
        planned.append(
            PipelineStagePlan(
                stage.name,
                ordinal,
                "reused",
                outputs=tuple(
                    (key, ref)
                    for key, value in outputs
                    for ref in (value if isinstance(value, tuple) else (value,))
                ),
            )
        )
    estimates = _estimates(
        store, plan, {stage.stage for stage in planned if stage.status == "computed"}
    )
    return PipelinePlan(
        recipe=plan.recipe,
        assay=plan.assay,
        label=plan.label,
        config=dict(plan.config),
        stages=tuple(
            replace(stage, estimate=estimates[stage.stage])
            if stage.stage in estimates
            else stage
            for stage in planned
        ),
    )
