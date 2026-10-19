"""Sample-level treatment-coded design matrices and contrast weights.

A design for ``~ v1 + v2 + ...`` has an intercept column and, for every
factor, one indicator column per level other than its first, the
reference level (treatment or dummy coding). A coefficient is then the
difference between its level and the reference with the other factors
held fixed, and a contrast is a weighted sum of coefficients.

Level labels and column names follow R's ``factor`` and ``model.matrix``:
numbers are ordered numerically and labelled as R prints them, logical
values become ``"FALSE"`` and ``"TRUE"``, and a column is named by its
factor and level, such as ``conditiontreated``. Text labels are ordered by
code point, which is R's order in the C locale. R's order in other locales
can differ for labels that mix letter case or punctuation, so pass
``levels`` to fix the order of such labels.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DesignMatrix",
    "contrast_vector",
    "factor_labels",
    "treatment_design",
]

INTERCEPT = "(Intercept)"

_RESERVED_NAMES = frozenset(
    {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "next",
        "break",
        "in",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "NA_complex_",
    }
)

# R prints a double with at most 15 significant digits.
_DOUBLE_DIGITS = 15


@dataclass(frozen=True, slots=True, eq=False)
class DesignMatrix:
    """A samples x columns design and the level order of each factor.

    ``columns`` names the columns of ``matrix``: the intercept first, then
    each factor's non-reference levels. ``levels`` holds every level of each
    factor in order, the reference first.
    """

    matrix: np.ndarray
    columns: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _double_label(value: float) -> str:
    """Label a double the way R's ``as.character`` does.

    The value is rounded to 15 significant digits and written in fixed or
    scientific notation, whichever is shorter, with fixed notation on a tie.
    Integral values below 2**53 keep every digit in fixed notation.
    """
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0:
        return "0"
    mantissa, exponent_text = f"{abs(value):.{_DOUBLE_DIGITS - 1}e}".split("e")
    significand = mantissa.replace(".", "").rstrip("0") or "0"
    exponent = int(exponent_text)
    sign = "-" if value < 0 else ""
    head, tail = significand[0], significand[1:]
    scientific = (
        f"{sign}{head}{'.' if tail else ''}{tail}"
        f"e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    )
    if value.is_integer() and abs(value) < 2**53:
        fixed = str(int(value))
    elif exponent < 0:
        fixed = f"{sign}0.{'0' * (-exponent - 1)}{significand}"
    else:
        whole = significand.ljust(exponent + 1, "0")
        fraction = significand[exponent + 1 :]
        fixed = f"{sign}{whole[: exponent + 1]}{'.' if fraction else ''}{fraction}"
    return fixed if len(fixed) <= len(scientific) else scientific


def _value_kind(values: np.ndarray) -> str:
    """Classify values as ``"logical"``, ``"integer"``, ``"double"``, or ``"text"``."""
    if values.dtype.kind == "b":
        return "logical"
    if values.dtype.kind in "iu":
        return "integer"
    if values.dtype.kind == "f":
        return "double"
    if values.dtype.kind == "O" and values.size:
        items = values.tolist()
        if all(isinstance(item, bool | np.bool_) for item in items):
            return "logical"
        if all(
            isinstance(item, Integral) and not isinstance(item, bool | np.bool_)
            for item in items
        ):
            return "integer"
        if all(
            isinstance(item, Real) and not isinstance(item, bool | np.bool_)
            for item in items
        ):
            return "double"
    return "text"


def _label(value: Any, kind: str) -> str:
    if kind == "logical":
        return "TRUE" if bool(value) else "FALSE"
    if kind == "integer":
        return str(int(value))
    if kind == "double":
        return _double_label(float(value))
    return str(value)


def factor_labels(
    values: Sequence[Any] | np.ndarray, *, name: str = "values"
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return each value's level label and the default level order.

    Numbers are ordered by value and logical values FALSE before TRUE;
    text is ordered by code point.

    Raises:
        ValueError: If ``values`` is not one-dimensional or has a missing
            value.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind in "OfcSU" and any(_is_missing(item) for item in array):
        raise ValueError(f"{name} has missing values")
    kind = _value_kind(array)
    labels = np.asarray([_label(item, kind) for item in array.tolist()], dtype=object)
    if kind == "text":
        order = sorted(set(labels.tolist()))
    else:
        numeric = np.asarray(array, dtype=np.float64)
        order = list(dict.fromkeys(labels[np.argsort(numeric, kind="stable")].tolist()))
    return labels, tuple(order)


def _syntactic(name: str) -> bool:
    """Whether R accepts ``name`` as a variable name without backticks."""
    if not name or name in _RESERVED_NAMES:
        return False
    if name[0] == ".":
        if len(name) > 1 and name[1].isdigit():
            return False
    elif not name[0].isalpha():
        return False
    return all(char.isalnum() or char in "._" for char in name)


def _column_prefix(name: str) -> str:
    return name if _syntactic(name) else f"`{name}`"


def _resolve_levels(
    name: str, observed: tuple[str, ...], requested: Sequence[Any] | None
) -> tuple[str, ...]:
    if requested is None:
        return observed
    requested_array = np.asarray(list(requested))
    kind = _value_kind(requested_array) if requested_array.size else "text"
    resolved = tuple(_label(item, kind) for item in requested_array.tolist())
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"levels of {name!r} repeat a level")
    unknown = sorted(set(observed).difference(resolved))
    if unknown:
        raise ValueError(f"levels of {name!r} do not include {unknown}")
    unused = [level for level in resolved if level not in set(observed)]
    if unused:
        raise ValueError(f"levels of {name!r} name levels without samples: {unused}")
    return resolved


def treatment_design(
    variables: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    levels: Mapping[str, Sequence[Any]] | None = None,
) -> DesignMatrix:
    """Return the treatment-coded design of ``~ v1 + v2 + ...``.

    Every variable is a factor whose first level is the reference. The
    design has an intercept, then the indicators of each variable's other
    levels, with the variables in the order given.

    Args:
        variables: One value per sample for each variable.
        levels: Level order of some variables, the reference first.
            Defaults to :func:`factor_labels` order.

    Raises:
        ValueError: If ``variables`` is empty, the variables differ in
            length, a variable has a missing value or fewer than two
            levels, ``levels`` names an unknown variable or does not match
            the observed levels, or the design is not full rank.
    """
    if not variables:
        raise ValueError("A design needs at least one variable")
    requested = dict(levels or {})
    unknown = sorted(set(requested).difference(variables))
    if unknown:
        raise ValueError(f"levels name unknown variables: {unknown}")
    lengths = {name: len(np.asarray(values)) for name, values in variables.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Design variables differ in length: {lengths}")
    n_samples = next(iter(lengths.values()))
    if n_samples == 0:
        raise ValueError("A design needs at least one sample")

    columns = [INTERCEPT]
    blocks = [np.ones((n_samples, 1))]
    resolved_levels: dict[str, tuple[str, ...]] = {}
    for name, values in variables.items():
        labels, observed = factor_labels(values, name=f"Design variable {name!r}")
        order = _resolve_levels(name, observed, requested.get(name))
        if len(order) < 2:
            raise ValueError(
                f"Design variable {name!r} needs at least two levels, got {order}"
            )
        resolved_levels[name] = order
        prefix = _column_prefix(str(name))
        columns.extend(f"{prefix}{level}" for level in order[1:])
        blocks.append(
            np.column_stack([labels == level for level in order[1:]]).astype(np.float64)
        )
    matrix = np.ascontiguousarray(np.hstack(blocks), dtype=np.float64)
    if len(set(columns)) != len(columns):
        raise ValueError(f"Design columns are not unique: {columns}")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        raise ValueError(
            f"The design of {list(variables)} is not full rank "
            f"(rank {rank} with {matrix.shape[1]} columns); a variable is "
            "confounded with the others or a level combination has no samples"
        )
    return DesignMatrix(matrix=matrix, columns=tuple(columns), levels=resolved_levels)


def contrast_vector(design: DesignMatrix, contrast: Mapping[str, float]) -> np.ndarray:
    """Return contrast weights over ``design.columns``.

    Columns that ``contrast`` leaves out have weight 0.

    Raises:
        ValueError: If ``contrast`` names an unknown column, a weight is not
            finite, or every weight is 0.
    """
    position = {column: index for index, column in enumerate(design.columns)}
    unknown = sorted(set(contrast).difference(position))
    if unknown:
        raise ValueError(
            f"Contrast names unknown design columns {unknown}; the design has "
            f"{list(design.columns)}"
        )
    weights = np.zeros(len(design.columns), dtype=np.float64)
    for column, weight in contrast.items():
        weights[position[column]] = float(weight)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Contrast weights must be finite")
    if not np.any(weights):
        raise ValueError("A contrast needs at least one non-zero weight")
    return weights
