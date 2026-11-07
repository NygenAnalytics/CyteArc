from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
import time
from typing import Any, Literal

import numpy as np
import zarr
from zarr.errors import ContainsArrayError, ContainsGroupError

from ..utils.logging import logger
from .artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_path,
    canonical_bytes,
    make_provenance,
    nearest_stored_artifacts,
    new_artifact_id,
    reusable_artifact_groups,
    serialize_artifact_value,
)
from .diagnostics import checked_diagnostics, declared_diagnostics, read_diagnostics
from .operation_revisions import effective_revision
from .stores import metadata_workers, run_concurrently


type ArtifactPlanDisposition = Literal["created", "reused"]


@dataclass(frozen=True, slots=True)
class ArtifactPlanReceipt:
    """One artifact planning decision observed by a pipeline stage."""

    operation: str
    ref: ArtifactRef
    disposition: ArtifactPlanDisposition


_PLAN_COLLECTORS: ContextVar[tuple[list[ArtifactPlanReceipt], ...]] = ContextVar(
    "cytearc_artifact_plan_collectors",
    default=(),
)


@contextmanager
def artifact_plan_scope() -> Any:
    """Collect nested artifact planning decisions without changing producers."""

    receipts: list[ArtifactPlanReceipt] = []
    collectors = _PLAN_COLLECTORS.get()
    token = _PLAN_COLLECTORS.set((*collectors, receipts))
    try:
        yield receipts
    finally:
        _PLAN_COLLECTORS.reset(token)


type ComputationReason = Literal[
    "absent",
    "changed",
    "superseded",
    "incomplete",
    "missing_diagnostics",
    "invalid",
    "invalidate_cache",
]


@dataclass(frozen=True, slots=True)
class ArtifactComputation:
    """An artifact that planning would compute rather than reuse, and why.

    Attributes:
        operation: The operation that would compute it.
        kind: Its artifact kind.
        reason: Why no stored artifact serves it: ``"absent"``, the store holds
            no artifact of the operation; ``"changed"``, the closest one of the
            operation was computed with other parameters or inputs;
            ``"superseded"``, one of this identity is at an earlier revision;
            ``"incomplete"``, one of this identity was interrupted;
            ``"missing_diagnostics"``, one of this identity lacks the
            diagnostics that its operation saves; ``"invalid"``, one of this
            identity fails its payload checks; ``"invalidate_cache"``, the
            call asked to compute it again.
        existing: The stored artifact that the reason concerns, or None.
        revision_changes: For ``"superseded"``, what each newer revision changed.
        changed_parameters: For ``"changed"``, the parameters that differ.
        changed_inputs: For ``"changed"``, the inputs that differ.
    """

    operation: str
    kind: str
    reason: ComputationReason
    existing: ArtifactRef | None = None
    revision_changes: tuple[str, ...] = ()
    changed_parameters: tuple[str, ...] = ()
    changed_inputs: tuple[str, ...] = ()

    def describe(self) -> str:
        """Say in one sentence why the artifact would be computed."""
        operation = self.operation
        if self.reason == "absent":
            return f"The store holds no {operation} artifact"
        if self.reason == "changed":
            differences = []
            if self.changed_parameters:
                differences.append("parameters " + ", ".join(self.changed_parameters))
            if self.changed_inputs:
                differences.append("inputs " + ", ".join(self.changed_inputs))
            return f"The closest {operation} artifact differs in " + " and ".join(
                differences
            )
        if self.reason == "superseded":
            changes = "; ".join(self.revision_changes)
            return f"A revision supersedes the stored {operation} artifact" + (
                f": {changes}" if changes else ""
            )
        if self.reason == "incomplete":
            return f"The stored {operation} artifact is incomplete"
        if self.reason == "missing_diagnostics":
            return (
                f"The stored {operation} artifact lacks the diagnostics that "
                f"{operation} saves"
            )
        if self.reason == "invalid":
            return f"The stored {operation} artifact fails its payload checks"
        return f"The call asks to compute the {operation} artifact again"


class ArtifactComputationRequired(BaseException):
    """Raised by :func:`plan_artifact` in a forecast instead of planning a new artifact.

    It derives from ``BaseException``, as a shutdown request does, so that an
    operation's handlers of ``Exception`` let it pass.
    """

    def __init__(self, computation: ArtifactComputation) -> None:
        super().__init__(computation.describe())
        self.computation = computation


_FORECAST: ContextVar[bool] = ContextVar("cytearc_artifact_forecast", default=False)


@contextmanager
def artifact_forecast_scope() -> Iterator[None]:
    """Make planning raise ``ArtifactComputationRequired`` rather than plan a new artifact.

    Inside the scope, :func:`plan_artifact` reuses as it always does and
    raises before it plans an artifact to compute, so an operation stops
    before it computes or writes anything and its caller learns why.
    """
    token = _FORECAST.set(True)
    try:
        yield
    finally:
        _FORECAST.reset(token)


def _computation(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    assay: str | None,
    kind: str,
    provenance: dict[str, Any],
    rejected: list[tuple[ArtifactRef, ComputationReason]],
    invalidate_cache: bool,
) -> ArtifactComputation:
    """Explain why planning found no stored artifact to reuse."""
    operation = str(provenance["operation"])
    if invalidate_cache:
        return ArtifactComputation(operation, kind, "invalidate_cache")
    if rejected:
        # The candidates are newest first.
        existing, reason = rejected[0]
        return ArtifactComputation(operation, kind, reason, existing)
    found = nearest_stored_artifacts(
        root, scope=scope, assay=assay, kind=kind, provenance=provenance
    )
    if found.superseded is not None:
        return ArtifactComputation(
            operation,
            kind,
            "superseded",
            found.superseded.ref,
            revision_changes=tuple(
                entry.change for entry in found.superseded.superseded_by
            ),
        )
    if found.incomplete is not None:
        return ArtifactComputation(operation, kind, "incomplete", found.incomplete)
    if found.nearest is not None:
        return ArtifactComputation(
            operation,
            kind,
            "changed",
            found.nearest.ref,
            changed_parameters=found.changed_parameters,
            changed_inputs=found.changed_inputs,
        )
    return ArtifactComputation(operation, kind, "absent")


def _record_plan(planned: "PlannedArtifact") -> "PlannedArtifact":
    operation = planned.provenance["operation"]
    receipt = ArtifactPlanReceipt(
        operation=operation,
        ref=planned.ref,
        disposition="reused" if planned.reused else "created",
    )
    for collector in _PLAN_COLLECTORS.get():
        collector.append(receipt)
    return planned


@dataclass(frozen=True, slots=True)
class PlannedArtifact:
    """What ``plan_artifact`` decided: a reused artifact or a new one."""

    ref: ArtifactRef
    provenance: dict[str, Any]
    execution_options: dict[str, Any]
    reused: bool
    required_arrays: tuple[Any, ...]
    required_attributes: tuple[Any, ...]
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None
    #: The keys of the diagnostics that the operation saves with the artifact.
    diagnostics: tuple[str, ...] = ()


def planned_diagnostics(
    planned: PlannedArtifact, values: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Check the diagnostics of a planned artifact against those its operation declares.

    Returns:
        The diagnostics as the artifact saves them, or None for an operation
        that declares none.

    Raises:
        ValueError: If the operation declares diagnostics and ``values`` is
            None or holds other keys or values, or it declares none and
            ``values`` holds some.
    """
    operation = str(planned.provenance["operation"])
    if not planned.diagnostics:
        if values:
            raise ValueError(
                f"{operation} declares no diagnostics, so its artifact saves none, "
                f"not {sorted(values)}"
            )
        return None
    if values is None:
        raise ValueError(
            f"{operation} saves the diagnostics {', '.join(planned.diagnostics)} "
            "with every artifact; pass them"
        )
    return checked_diagnostics(values, planned.diagnostics, operation=operation)


def _holds_diagnostics(group: zarr.Group, operation: str) -> bool:
    """Whether an artifact holds the diagnostics that its operation declares."""
    try:
        read_diagnostics(group.attrs, operation, path=group.path)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ArrayRequirement:
    """A required array of an artifact payload."""

    name: str
    shape: tuple[int | None, ...] | None = None
    dtype_kind: str | None = None
    dtype: Any | None = None

    def matches(self, group: zarr.Group) -> bool:
        from .types import as_zarr_array

        try:
            array = as_zarr_array(group[self.name], name=self.name)
        except (KeyError, TypeError):
            return False
        if self.shape is not None:
            if len(array.shape) != len(self.shape):
                return False
            if any(
                expected is not None and int(actual) != expected
                for actual, expected in zip(array.shape, self.shape, strict=True)
            ):
                return False
        if self.dtype_kind is not None:
            if np.dtype(array.dtype).kind != self.dtype_kind:
                return False
        return self.dtype is None or np.dtype(array.dtype) == np.dtype(self.dtype)


@dataclass(frozen=True, slots=True)
class AttributeRequirement:
    name: str
    expected_types: tuple[type[Any], ...] | None = None
    predicate: Callable[[Any], bool] | None = None

    def matches(self, group: zarr.Group) -> bool:
        if self.name not in group.attrs:
            return False
        value = group.attrs[self.name]
        if self.expected_types is not None and not isinstance(
            value,
            self.expected_types,
        ):
            return False
        return self.predicate(value) if self.predicate is not None else True


def _array_requirements(
    required: tuple[str | ArrayRequirement, ...],
) -> tuple[ArrayRequirement, ...]:
    return tuple(
        requirement
        if isinstance(requirement, ArrayRequirement)
        else ArrayRequirement(str(requirement))
        for requirement in required
    )


def _attribute_requirements(
    required: tuple[str | AttributeRequirement, ...],
) -> tuple[AttributeRequirement, ...]:
    return tuple(
        requirement
        if isinstance(requirement, AttributeRequirement)
        else AttributeRequirement(str(requirement))
        for requirement in required
    )


def _requirements_met(
    group: zarr.Group, requirements: tuple[ArrayRequirement, ...]
) -> list[bool]:
    # Each check opens one array; object stores overlap them.
    return run_concurrently(
        [partial(requirement.matches, group) for requirement in requirements],
        workers=metadata_workers(group),
    )


def plan_artifact(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    execution_options: dict[str, Any],
    assay: str | None = None,
    invalidate_cache: bool = False,
    required_arrays: tuple[str | ArrayRequirement, ...] = (),
    required_attributes: tuple[str | AttributeRequirement, ...] = (),
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
) -> PlannedArtifact:
    """Reuse a complete artifact with this exact provenance or plan a new one.

    An artifact is reused only with the diagnostics that its operation
    declares, so an artifact saved without them is computed anew. Inside
    :func:`artifact_forecast_scope`, a new artifact is never planned: the call
    raises ``ArtifactComputationRequired`` with the reason instead.
    """
    diagnostics = declared_diagnostics(operation)
    provenance = make_provenance(
        operation=operation,
        parameters=parameters,
        inputs=inputs,
    )
    revision = effective_revision(
        operation, kind, provenance["parameters"], provenance["inputs"]
    )
    if revision > 1:
        provenance = make_provenance(
            operation=operation,
            parameters=provenance["parameters"],
            inputs=provenance["inputs"],
            revision=revision,
        )
    stored_execution_options = serialize_artifact_value(execution_options)
    if not isinstance(stored_execution_options, dict):
        raise TypeError("execution_options must serialize to a mapping")
    canonical_bytes(stored_execution_options)
    forecast = _FORECAST.get()
    candidates = reusable_artifact_groups(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        provenance=provenance,
        invalidate_cache=invalidate_cache,
        log_recomputation=not forecast,
    )
    requirements = _array_requirements(required_arrays)
    attribute_requirements = _attribute_requirements(required_attributes)
    reused = None
    # Why each candidate was passed over, which a forecast reports.
    rejected: list[tuple[ArtifactRef, ComputationReason]] = []
    for candidate, group in candidates:
        if not all(_requirements_met(group, requirements)):
            rejected.append((candidate, "invalid"))
            continue
        if any(
            not requirement.matches(group) for requirement in attribute_requirements
        ):
            rejected.append((candidate, "invalid"))
            continue
        if diagnostics and not _holds_diagnostics(group, operation):
            rejected.append((candidate, "missing_diagnostics"))
            continue
        if reuse_validator is not None and not reuse_validator(
            candidate,
            group,
        ):
            rejected.append((candidate, "invalid"))
            continue
        reused = candidate
        break
    if reused is None and forecast:
        raise ArtifactComputationRequired(
            _computation(
                root,
                scope=scope,
                assay=assay,
                kind=kind,
                provenance=provenance,
                rejected=rejected,
                invalidate_cache=invalidate_cache,
            )
        )
    if reused is not None:
        return _record_plan(
            PlannedArtifact(
                ref=reused,
                provenance=provenance,
                execution_options=stored_execution_options,
                reused=True,
                required_arrays=required_arrays,
                required_attributes=required_attributes,
                reuse_validator=reuse_validator,
                diagnostics=diagnostics,
            )
        )
    # A random 256-bit ID does not collide; start_artifact refuses an existing path.
    ref = ArtifactRef(
        scope=scope,
        assay=assay,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    return _record_plan(
        PlannedArtifact(
            ref=ref,
            provenance=provenance,
            execution_options=stored_execution_options,
            reused=False,
            required_arrays=required_arrays,
            required_attributes=required_attributes,
            reuse_validator=reuse_validator,
            diagnostics=diagnostics,
        )
    )


def start_artifact(root: zarr.Group, planned: PlannedArtifact) -> zarr.Group:
    """Create the incomplete group of a planned artifact.

    Raises:
        PermissionError: If ``root`` is read-only. Nothing is written.
    """
    if planned.reused:
        raise ValueError("Cannot start a reused artifact")
    if root.read_only:
        raise PermissionError(
            f"{planned.provenance['operation']} requires a DataStore opened "
            "with zarr_mode='r+'"
        )
    path = artifact_path(planned.ref)
    from .. import __version__

    try:
        # One metadata write creates the group with its record.
        return root.create_group(
            path,
            attributes={
                "artifact_id": planned.ref.artifact_id,
                "kind": planned.ref.kind,
                "provenance": planned.provenance,
                "execution_options": planned.execution_options,
                "created_at_ns": time.time_ns(),
                "cytearc_version": __version__,
                "complete": False,
            },
        )
    except (ContainsArrayError, ContainsGroupError) as exc:
        raise FileExistsError(f"Artifact path already exists: {path}") from exc


def validate_artifact_payload(
    group: zarr.Group,
    planned: PlannedArtifact,
) -> None:
    """Check a started artifact against its plan before it is published."""
    if planned.reused:
        raise ValueError("Cannot finish a reused artifact")
    if group.attrs.get("complete") is not False:
        group.attrs["complete"] = False
    if (
        group.attrs.get("artifact_id") != planned.ref.artifact_id
        or group.attrs.get("kind") != planned.ref.kind
    ):
        raise ValueError("Artifact group does not match its creation plan")
    requirements = _array_requirements(planned.required_arrays)
    for requirement, met in zip(
        requirements, _requirements_met(group, requirements), strict=True
    ):
        if not met:
            raise ValueError(
                f"Artifact array {requirement.name!r} does not satisfy its contract"
            )
    for attribute_requirement in _attribute_requirements(planned.required_attributes):
        if not attribute_requirement.matches(group):
            raise ValueError(
                f"Artifact attribute {attribute_requirement.name!r} "
                "does not satisfy its contract"
            )
    if planned.reuse_validator is not None and not planned.reuse_validator(
        planned.ref,
        group,
    ):
        raise ValueError("Artifact payload does not satisfy its reuse contract")


def finish_artifact(
    group: zarr.Group,
    planned: PlannedArtifact,
) -> None:
    """Validate a started artifact and mark it complete."""
    validate_artifact_payload(group, planned)
    group.attrs["complete"] = True


def discard_artifact(root: zarr.Group, planned: PlannedArtifact) -> None:
    """Delete the incomplete group of a started artifact after a failed write.

    A deletion failure is logged, not raised, so the write's own error is the
    one that propagates.
    """
    path = artifact_path(planned.ref)
    try:
        del root[path]
    except Exception as error:
        logger.warning(f"Could not remove the incomplete artifact at {path}: {error}")


def interrupted_artifact_message(path: str) -> str:
    """Explain why an interrupted write keeps its incomplete artifact."""
    return (
        f"An interruption stopped the write of artifact {path}, which Zarr may "
        "still be running, so the artifact was kept, marked incomplete; nothing "
        "reuses it. Once no process is writing to this store, remove it with "
        "ds.artifacts.prune_incomplete(dry_run=False)."
    )


@contextmanager
def artifact_transaction(
    root: zarr.Group,
    planned: PlannedArtifact,
) -> Iterator[zarr.Group]:
    """Start a planned artifact, yield its group, and finish it after the body.

    If starting the group, the body, or the payload validation raises an
    exception, the group that this call created is deleted before the error
    propagates, so a failed write leaves no incomplete artifact behind. An
    interruption (``KeyboardInterrupt``, ``asyncio.CancelledError``,
    ``SystemExit``, or a shutdown request) keeps the group, marked incomplete,
    and logs how to remove it: Zarr may still be running the write that the
    interruption stopped waiting for, and a deleted group would collect its
    chunks. Nothing is deleted once the publication write is issued.
    """
    group: zarr.Group | None = None
    try:
        group = start_artifact(root, planned)
        yield group
        validate_artifact_payload(group, planned)
    except BaseException as error:
        if not isinstance(error, Exception):
            logger.warning(interrupted_artifact_message(artifact_path(planned.ref)))
        elif group is not None:
            discard_artifact(root, planned)
        raise
    # The publication write follows the try, so nothing is deleted once it is
    # issued.
    group.attrs["complete"] = True


async def _has_keys(store: Any, prefix: str) -> bool:
    async for _ in store.list_prefix(prefix):
        return True
    return False


def prune_incomplete_artifacts(
    root: zarr.Group,
    assays: Iterable[str],
    *,
    dry_run: bool,
) -> list[str]:
    """Find, or remove, what interrupted and killed artifact writes left behind.

    Below the datastore-scoped artifacts of ``root`` and those of each assay
    in ``assays``, this finds the artifact groups that are not complete and
    the directories below a known artifact kind that hold keys but no group
    record: the chunks of a write that was killed before its group existed,
    or that landed after its group was removed. Complete artifacts, groups
    whose record cannot be read, unknown kinds, and other objects are never
    touched. A mounted target is pruned of its own artifacts only; its matrix
    source is read only.

    An artifact that another process is writing is incomplete until it
    finishes, so remove residue only after confirming that no process is
    writing to the store.

    Args:
        root: The root or workspace group of a datastore.
        assays: Names of the assays whose artifacts are checked.
        dry_run: Find the residue without removing anything.

    Returns:
        The paths below ``root`` that were found, or removed, sorted.

    Raises:
        PermissionError: If ``dry_run`` is False and ``root`` is read only.
    """
    from zarr.core.sync import sync

    from .artifacts import (
        ARTIFACT_KINDS,
        _artifact_status,
        _child_groups,
        _member_names,
    )
    from .stores import MountedArtifactStore
    from .types import as_zarr_group

    if not dry_run and root.read_only:
        raise PermissionError(
            "Removing incomplete artifacts requires a DataStore opened with "
            "zarr_mode='r+'"
        )
    if isinstance(root.store, MountedArtifactStore):
        root = zarr.open_group(
            store=root.store._store,
            path=root.path,
            mode="r" if root.read_only else "r+",
            use_consolidated=False,
        )
    scopes: list[tuple[ArtifactScope, str | None, str]] = [
        ("datastore", None, "artifacts"),
        *(("assay", assay, f"{assay}/artifacts") for assay in assays),
    ]
    found: list[str] = []
    for scope, assay, base_path in scopes:
        try:
            base = as_zarr_group(root[base_path], name=base_path)
        except (KeyError, TypeError):
            # No artifacts, or an object that CyteArc did not write.
            continue
        for kind, kind_group in _child_groups(base, "artifact kinds"):
            if kind not in ARTIFACT_KINDS:
                continue
            names = sync(_member_names(kind_group))
            for name in sorted(set(names) - {"zarr.json"}):
                path = f"{base_path}/{kind}/{name}"
                try:
                    ref = ArtifactRef(
                        scope=scope, assay=assay, kind=kind, artifact_id=name
                    )
                except ValueError:
                    # Not an artifact ID, so not something CyteArc wrote.
                    continue
                try:
                    node = kind_group[name]
                except KeyError:
                    # Keys without a group record: chunks of a write without
                    # a group. A plain object of that name holds no keys.
                    prefix = "/".join(filter(None, (root.store_path.path, path)))
                    if sync(_has_keys(root.store, f"{prefix}/")):
                        found.append(path)
                    continue
                except (TypeError, ValueError):
                    continue
                try:
                    if not _artifact_status(
                        as_zarr_group(node, name=path), ref, path
                    ).complete:
                        found.append(path)
                except (KeyError, TypeError, ValueError):
                    continue
    found.sort()
    if not dry_run:
        for path in found:
            sync((root.store_path / path).delete_dir())
    return found


def reused_artifact_group(
    root: zarr.Group,
    planned: PlannedArtifact,
) -> zarr.Group:
    from .artifacts import artifact_group

    if not planned.reused:
        raise ValueError("Artifact is not reused")
    return artifact_group(root, planned.ref)
