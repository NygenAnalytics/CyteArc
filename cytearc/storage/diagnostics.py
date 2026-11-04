"""The diagnostics that an artifact saves about how it was computed.

Diagnostics are the numbers that an operation records about what it read,
kept, dropped and expected, so that a caller can judge a result without
reading its arrays: the barcodes of a file that match no cell of the store,
say, or the share of cells in the largest clone. Each operation declares its
keys once, on its arguments class, with the call that computes its artifact;
the artifact holds them in one ``diagnostics`` attribute, written with its
last write, so a reused artifact reports the numbers of its computation.
The writer refuses a mapping with other keys, and the reader refuses an
artifact whose keys differ, so the keys are a contract of each operation.

Keys are camelCase. A value is an integer, a finite float, a text, None for
a number that was not measured, or, under a key that ends in ``Examples``, a
list of at most five texts. Diagnostics never enter an artifact's identity
and hold nothing that varies between runs on the same inputs, such as
timings. Operations warn from them through :func:`warn_diagnostics` after
they compute or reuse an artifact, so a reused artifact warns as a computed
one does.
"""

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..utils.warnings import DiagnosticWarning, warn

__all__ = [
    "DIAGNOSTICS_ATTRIBUTE",
    "EXAMPLES",
    "DiagnosticRule",
    "checked_diagnostics",
    "declare_diagnostics",
    "declared_diagnostics",
    "read_diagnostics",
    "warn_diagnostics",
]

#: The attribute of an artifact's group that holds its diagnostics.
DIAGNOSTICS_ATTRIBUTE = "diagnostics"
#: The most examples that a diagnostic lists.
EXAMPLES = 5
# A diagnostic's key: camelCase, starting with a lowercase letter.
_KEY = re.compile(r"^[a-z][A-Za-z0-9]*$")
# The suffix of a key whose value lists examples.
_EXAMPLES_SUFFIX = "Examples"

type DiagnosticValue = int | float | str | list[str] | None


@dataclass(frozen=True, slots=True)
class _Declaration:
    keys: tuple[str, ...]
    call: str


# Each operation's declared keys and the call that computes its artifacts, as
# its arguments class declares them when it is defined.
_DECLARED: dict[str, _Declaration] = {}


def declare_diagnostics(operation: str, keys: Sequence[str], *, call: str) -> None:
    """Declare the diagnostics that ``operation`` saves with each artifact.

    Args:
        operation: The operation, as its artifacts' provenance names it.
        keys: The keys of its diagnostics, in the order they are reported.
        call: The public call that computes its artifacts, such as
            ``"ds.clones.import_clones"``, which the refusal of an artifact
            without them names.

    Raises:
        ValueError: If a key is not camelCase or repeats, ``call`` is empty,
            or the operation was declared with other keys or another call.
    """
    resolved = tuple(keys)
    for key in resolved:
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            raise ValueError(
                f"The diagnostic {key!r} of {operation} must be camelCase, such as "
                "'unmatchedBarcodes'"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{operation} declares a diagnostic twice: {list(resolved)}")
    if not isinstance(call, str) or not call:
        raise ValueError(f"{operation} must name the call that computes its artifacts")
    declaration = _Declaration(resolved, call)
    existing = _DECLARED.get(operation)
    if existing is not None and existing != declaration:
        raise ValueError(
            f"{operation} is declared with the diagnostics {list(existing.keys)} "
            f"already, not {list(resolved)}"
        )
    _DECLARED[operation] = declaration


def declared_diagnostics(operation: str | None) -> tuple[str, ...]:
    """Return the keys of the diagnostics that ``operation`` saves, empty for none."""
    if operation is None:
        return ()
    declaration = _DECLARED.get(operation)
    return () if declaration is None else declaration.keys


def _checked_value(operation: str, key: str, value: Any) -> DiagnosticValue:
    if key.endswith(_EXAMPLES_SUFFIX):
        if not isinstance(value, list | tuple) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"The diagnostic {key!r} of {operation} must be a list of texts, "
                f"not {value!r}"
            )
        if len(value) > EXAMPLES:
            raise ValueError(
                f"The diagnostic {key!r} of {operation} lists {len(value)} examples; "
                f"it lists at most {EXAMPLES}"
            )
        return list(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        raise ValueError(
            f"The diagnostic {key!r} of {operation} must be a number, a text or None, "
            f"not the boolean {value!r}"
        )
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not math.isfinite(float(value)):
            raise ValueError(
                f"The diagnostic {key!r} of {operation} must be finite, not {value!r}; "
                "save None for a number that was not measured"
            )
        return float(value)
    raise ValueError(
        f"The diagnostic {key!r} of {operation} must be a number, a text or None, "
        f"not {value!r}"
    )


def checked_diagnostics(
    values: Any, keys: Sequence[str], *, operation: str
) -> dict[str, DiagnosticValue]:
    """Check diagnostics against the keys an operation declares and return them as JSON values.

    NumPy numbers become Python numbers and example tuples become lists; the
    result follows the declared order.

    Raises:
        ValueError: If ``values`` is not a mapping of exactly ``keys``, or a
            value is not one that a diagnostic holds.
    """
    if not isinstance(values, Mapping):
        raise ValueError(
            f"The diagnostics of {operation} must be a mapping, not {values!r}"
        )
    missing = [key for key in keys if key not in values]
    extra = sorted(str(key) for key in values if key not in keys)
    if missing or extra:
        detail = "; ".join(
            part
            for part in (
                f"missing {', '.join(missing)}" if missing else "",
                f"not declared {', '.join(extra)}" if extra else "",
            )
            if part
        )
        raise ValueError(
            f"The diagnostics of {operation} must be exactly {', '.join(keys) or 'none'}: "
            f"{detail}"
        )
    return {key: _checked_value(operation, key, values[key]) for key in keys}


def _recompute(operation: str) -> str:
    """The advice that a refusal of an artifact without its diagnostics gives."""
    return (
        f"call {_DECLARED[operation].call} again with the same arguments, which "
        "computes the artifact anew and leaves this one unused"
    )


def read_diagnostics(
    attributes: Mapping[str, Any], operation: str | None, *, path: str
) -> dict[str, DiagnosticValue]:
    """Read an artifact's diagnostics, checked against those its operation declares.

    An operation that declares none reads none, whatever the artifact holds
    under ``diagnostics``.

    Args:
        attributes: The attributes of the artifact's group.
        operation: The operation that saved the artifact.
        path: The artifact's path in the store, for errors.

    Raises:
        ValueError: If the operation declares diagnostics and the artifact
            holds none, as artifacts saved before CyteArc recorded them, or
            other keys or values; the message names the call that computes
            the artifact anew.
    """
    keys = declared_diagnostics(operation)
    if not keys:
        return {}
    assert operation is not None
    if DIAGNOSTICS_ATTRIBUTE not in attributes:
        raise ValueError(
            f"The artifact at {path} holds no diagnostics, which {operation} saves "
            f"with every artifact; {_recompute(operation)}"
        )
    try:
        return checked_diagnostics(
            attributes[DIAGNOSTICS_ATTRIBUTE], keys, operation=operation
        )
    except ValueError as error:
        raise ValueError(
            f"The artifact at {path} holds other diagnostics than {operation} saves: "
            f"{error}. To read it, {_recompute(operation)}"
        ) from None


@dataclass(frozen=True, slots=True)
class DiagnosticRule:
    """A warning that an operation gives from one of its saved diagnostics.

    Attributes:
        name: The diagnostic that the warning reports.
        applies: Whether the saved diagnostics call for the warning.
        message: The warning's text, from the saved diagnostics.
    """

    name: str
    applies: Callable[[Mapping[str, Any]], bool]
    message: Callable[[Mapping[str, Any]], str]


class _Operation(Protocol):
    @property
    def operation(self) -> str: ...


def warn_diagnostics(
    arguments: _Operation,
    diagnostics: Mapping[str, Any],
    rules: Sequence[DiagnosticRule],
) -> None:
    """Give the warnings that the rules derive from an artifact's saved diagnostics.

    An operation calls this after it computes or reuses its artifact, with
    the diagnostics read back from the artifact, so that both warn alike.

    Raises:
        KeyError: If a rule reports a diagnostic that the artifact does not
            hold.
    """
    for rule in rules:
        if rule.name not in diagnostics:
            raise KeyError(
                f"A warning of {arguments.operation} reports the diagnostic "
                f"{rule.name!r}, which its artifact does not hold"
            )
        if rule.applies(diagnostics):
            warn(
                DiagnosticWarning(
                    rule.message(diagnostics),
                    arguments.operation,
                    rule.name,
                    diagnostics[rule.name],
                )
            )
