"""The plan of a trajectory: what each call of a chain would reuse or compute, and why.

Inside the block of ``ds.trajectory.plan()``, each trajectory call that saves
a result runs in an artifact forecast, where planning an artifact reuses as a
call does but raises before it plans one to compute, as a pipeline plan's
stages do. A call whose result the store holds returns that result's ref. A
call that would compute returns its own plan in place of the ref that it would
return, so the chain goes on as written, and a later call that takes that plan
as an input cannot learn its identity, which depends on a result that does not
exist yet: it is reported computed, waiting on that call, and is not run.
"""

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from ..storage.artifact_writer import (
    ArtifactComputationRequired,
    ComputationReason,
    artifact_forecast_scope,
)
from ..storage.refs import ArtifactRef

__all__ = ["TrajectoryCallPlan", "TrajectoryPlan"]

type CallPlanStatus = Literal["reused", "computed"]
type CallPlanReason = ComputationReason | Literal["upstream"]

# The trajectory calls that a plan covers, in the order the reference lists them.
PLANNED_CALLS = (
    "pseudotime",
    "fate",
    "states",
    "curves",
    "lineage_tree",
    "lineage_membership",
    "branch_cells",
    "markers",
    "aggregation",
    "lineage_markers",
    "drivers",
    "lineage_aggregation",
)


@dataclass(frozen=True, slots=True)
class TrajectoryCallPlan:
    """What one call of a planned trajectory would do.

    Inside the plan's block, a call that would compute returns this plan in
    place of the ref that it would return, and later planned calls take it
    as an input.

    Attributes:
        call: The trajectory method, such as ``"fate"``.
        ordinal: The call's position in the plan, from 0.
        status: ``"reused"`` when the store holds the call's result, complete
            and current; ``"computed"`` when the call would compute it.
        reason: Why a computed call computes: ``"absent"``, the store holds no
            artifact of the operation; ``"changed"``, the closest one was
            computed with other parameters or inputs; ``"superseded"``, an
            operation revision supersedes the stored one; ``"incomplete"``,
            the stored one was interrupted; ``"missing_diagnostics"``, the
            stored one lacks the diagnostics that its operation saves;
            ``"invalid"``, the stored one fails its payload checks;
            ``"invalidate_cache"``, the call asks to compute it again;
            ``"upstream"``, the call takes the result of a call before it
            that would compute. None for a reused call.
        detail: The reason in one sentence, or None.
        operation: The operation whose artifact the call would compute first,
            such as the snapshot of a direction given as a pandas Series, or
            None.
        ref: The result that a reused call returns, or None.
        existing: The stored artifact that the reason concerns: the one that
            a revision supersedes, that was interrupted, that lacks
            diagnostics, or that fails its checks, or the closest one of the
            operation. None otherwise.
        changed_parameters: For ``"changed"``, the operation's parameters in
            which the closest artifact differs.
        changed_inputs: For ``"changed"``, the inputs in which it differs.
        upstream: For ``"upstream"``, the positions of the calls whose results
            the call takes, in order.
    """

    call: str
    ordinal: int
    status: CallPlanStatus
    reason: CallPlanReason | None = None
    detail: str | None = None
    operation: str | None = None
    ref: ArtifactRef | None = None
    existing: ArtifactRef | None = None
    changed_parameters: tuple[str, ...] = ()
    changed_inputs: tuple[str, ...] = ()
    upstream: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the call's plan as JSON-compatible values."""
        return {
            "call": self.call,
            "ordinal": self.ordinal,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "operation": self.operation,
            "ref": None if self.ref is None else self.ref.to_dict(),
            "existing": None if self.existing is None else self.existing.to_dict(),
            "changedParameters": list(self.changed_parameters),
            "changedInputs": list(self.changed_inputs),
            "upstream": list(self.upstream),
        }


_OPEN: ContextVar["TrajectoryPlan | None"] = ContextVar(
    "cytearc_trajectory_plan", default=None
)


class TrajectoryPlan:
    """What the trajectory calls made in a plan's block would reuse and compute.

    ``ds.trajectory.plan()`` returns the plan, a context manager. Its calls
    fill as the block runs and are final once it ends.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._calls: list[TrajectoryCallPlan] = []
        self._used = False
        self._scope: Any = None
        self._token: Any = None

    def __enter__(self) -> "TrajectoryPlan":
        if self._used:
            raise RuntimeError(
                "This trajectory plan has already been used. Open a new one "
                "with ds.trajectory.plan()."
            )
        if _OPEN.get() is not None:
            raise RuntimeError(
                "A trajectory plan is already open, and plans do not nest. End "
                "its with block before you open another plan."
            )
        self._used = True
        # Every artifact that a call in the block would plan is forecast, so
        # a call that the plan does not cover cannot write either.
        self._scope = artifact_forecast_scope()
        self._scope.__enter__()
        self._token = _OPEN.set(self)
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        _OPEN.reset(self._token)
        self._scope.__exit__(None, None, None)
        if isinstance(error, ArtifactComputationRequired):
            raise RuntimeError(
                "A call inside ds.trajectory.plan() would compute a new "
                f"{error.computation.operation} result, and a plan writes "
                "nothing. A plan covers only the trajectory calls "
                f"{', '.join(PLANNED_CALLS)}: make any other call before the "
                "plan or after its with block."
            ) from error

    @property
    def calls(self) -> tuple[TrajectoryCallPlan, ...]:
        """Every planned call, in the order made."""
        return tuple(self._calls)

    def __getitem__(self, key: int | str) -> TrajectoryCallPlan:
        """Return the call at position ``key``, or the only call named ``key``.

        Raises:
            KeyError: If no call has that name, or several do.
            IndexError: If the plan holds no call at that position.
            TypeError: If ``key`` is neither a position nor a name.
        """
        if isinstance(key, bool) or not isinstance(key, int | str):
            raise TypeError(
                f"A plan's calls are found by position or name, not {key!r}"
            )
        if isinstance(key, int):
            return self._calls[key]
        found = [call for call in self._calls if call.call == key]
        if len(found) != 1:
            positions = [call.ordinal for call in found]
            raise KeyError(
                f"The plan holds no {key} call"
                if not found
                else f"The plan holds {len(found)} {key} calls, at positions "
                f"{positions}; take one by its position"
            )
        return found[0]

    @property
    def computed(self) -> tuple[str, ...]:
        """The calls that would compute, in the order made."""
        return tuple(call.call for call in self._calls if call.status == "computed")

    @property
    def reused(self) -> tuple[str, ...]:
        """The calls whose results the store holds, in the order made."""
        return tuple(call.call for call in self._calls if call.status == "reused")

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as JSON-compatible values."""
        return {"calls": [call.to_dict() for call in self._calls]}


def _plans_taken(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> list[TrajectoryCallPlan]:
    """Return the call plans among a call's arguments."""
    return [
        value
        for value in (*args, *kwargs.values())
        if isinstance(value, TrajectoryCallPlan)
    ]


def _plan_passed_on(call: str, taken: TrajectoryCallPlan) -> TypeError:
    """Refuse a call plan passed to a call that cannot plan with it."""
    return TypeError(
        f"{call} was given the plan of {taken.call}, call {taken.ordinal} of a "
        "trajectory plan, which would compute a result that does not exist "
        f"yet. Run {taken.call} outside ds.trajectory.plan() to save its "
        "result, and pass the ref that it returns."
    )


def refuse_call_plans(
    call: str, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> None:
    """Refuse a call plan among the arguments of a call that a plan does not cover.

    Raises:
        TypeError: If an argument is a :class:`TrajectoryCallPlan`.
    """
    taken = _plans_taken(args, kwargs)
    if taken:
        raise _plan_passed_on(call, taken[0])


def planned_call(
    store: Any,
    call: str,
    run: Callable[[], ArtifactRef],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> ArtifactRef | TrajectoryCallPlan:
    """Run a trajectory call, or plan it when a plan of ``store`` is open.

    With no plan open, the call runs as it always does. In a plan, a call that
    takes the plan of an earlier call waits on it; any other call runs in the
    forecast and returns its ref when the store holds its result, or its plan
    when it would compute.

    Raises:
        TypeError: If an argument is the plan of a call in no open plan of
            ``store``.
        RuntimeError: If a plan of another DataStore is open.
    """
    plan = _OPEN.get()
    taken = _plans_taken(args, kwargs)
    if plan is None:
        if taken:
            raise _plan_passed_on(call, taken[0])
        return run()
    if plan._store is not store:
        raise RuntimeError(
            f"A trajectory plan of another DataStore is open, so {call} on this "
            "store would run unplanned inside it. Make this store's calls before "
            "or after that plan's with block, or plan them in a plan of this "
            "store."
        )
    for value in taken:
        if not any(value is made for made in plan._calls):
            raise _plan_passed_on(call, value)
    ordinal = len(plan._calls)
    if taken:
        waiting = tuple(sorted({value.ordinal for value in taken}))
        named = ", ".join(
            f"{plan._calls[index].call} (call {index})" for index in waiting
        )
        planned = TrajectoryCallPlan(
            call,
            ordinal,
            "computed",
            reason="upstream",
            detail=(
                f"It takes what {named} would compute, so its identity is known "
                "only after that runs"
            ),
            upstream=waiting,
        )
        plan._calls.append(planned)
        return planned
    try:
        ref = run()
    except ArtifactComputationRequired as signal:
        computation = signal.computation
        planned = TrajectoryCallPlan(
            call,
            ordinal,
            "computed",
            reason=computation.reason,
            detail=computation.describe(),
            operation=computation.operation,
            existing=computation.existing,
            changed_parameters=computation.changed_parameters,
            changed_inputs=computation.changed_inputs,
        )
        plan._calls.append(planned)
        return planned
    plan._calls.append(TrajectoryCallPlan(call, ordinal, "reused", ref=ref))
    return ref
