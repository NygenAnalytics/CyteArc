"""Operation revisions, the one explicit way to stop reusing stored results.

An artifact's identity is its scope, assay, kind, and canonical provenance:
the producing operation, its parameters, and its inputs. Reuse requires an
exact provenance match, so a fix that changes what an operation computes for
unchanged provenance must also change provenance, or stored results of the
old code would keep being reused. This module records each such fix as an
:class:`OperationRevision`.

Planning computes the effective revision of a new artifact with
:func:`effective_revision` and records it in provenance only when it is 2 or
more. A revision changes only the identities of the artifacts it applies to.
Artifacts that record no revision are revision 1.

The registry is append-only. A released :class:`OperationRevision` is never
edited, renumbered, or removed, and its predicate never changes: stored
artifacts are judged against it by every later release. A new release that
changes the same results again adds the next revision.

See ``docs/source/developers/operation_revisions.md`` for when a change needs
a revision and how to add one.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

type RevisionPredicate = Callable[[str, Mapping[str, Any], Mapping[str, Any]], bool]
"""Predicate over an artifact's kind and its recorded parameters and inputs."""


@dataclass(frozen=True, slots=True)
class OperationRevision:
    """One released change to what an operation computes.

    Attributes:
        revision: The revision number; an operation's first change is 2.
        release: The first CyteArc release that records it, such as ``"1.0.0"``.
        change: One line that says what changed.
        applies: A pure predicate that selects the affected artifacts, or None for all.
    """

    revision: int
    release: str
    change: str
    applies: RevisionPredicate | None


# Released revisions by operation. An operation that is not listed has no
# revisions, so every artifact of it is revision 1. Append a revision to the
# tuple of the operation it changes, adding the operation if needed; never
# edit a released entry.
_OPERATION_REVISIONS: dict[str, tuple[OperationRevision, ...]] = {}


def _validate_registry(
    registry: Mapping[str, tuple[OperationRevision, ...]],
) -> None:
    for operation, revisions in registry.items():
        numbers = [entry.revision for entry in revisions]
        if numbers != list(range(2, len(numbers) + 2)):
            raise ValueError(
                f"Revisions of {operation!r} must be numbered 2, 3, ... in order, "
                f"got {numbers}"
            )


_validate_registry(_OPERATION_REVISIONS)
OPERATION_REVISIONS: Mapping[str, tuple[OperationRevision, ...]] = MappingProxyType(
    _OPERATION_REVISIONS
)
"""Released revisions by operation; an operation that is not listed has none."""


def applicable_revisions(
    operation: str,
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> tuple[OperationRevision, ...]:
    """Return the released revisions that apply to one artifact, oldest first."""
    return tuple(
        entry
        for entry in OPERATION_REVISIONS.get(operation, ())
        if entry.applies is None or entry.applies(kind, parameters, inputs)
    )


def effective_revision(
    operation: str,
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> int:
    """Return the highest applicable revision of ``operation`` for an artifact, or 1."""
    applicable = applicable_revisions(operation, kind, parameters, inputs)
    return applicable[-1].revision if applicable else 1
