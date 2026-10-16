"""Sample-level design matrices with R's treatment coding.

:func:`treatment_design` builds the matrix that ``model.matrix(~ a + b)``
returns in R: an intercept, one column per non-reference level of each
categorical variable, and one column per numeric variable. R orders factor
levels with the collation of the session locale, so the default here is
Python's code-point order and callers fix any other order with ``levels``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DesignMatrix", "contrast_vector", "treatment_design"]


@dataclass(frozen=True, slots=True, eq=False)
class DesignMatrix:
    """A design matrix, its column names, and the level order of each factor."""

    matrix: np.ndarray
    columns: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]


def _is_numeric(values: np.ndarray) -> bool:
    return values.dtype.kind in "iuf"


def treatment_design(
    variables: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    levels: Mapping[str, Sequence[Any]] | None = None,
) -> DesignMatrix:
    """Return the treatment-coded design of ``~ variables`` with an intercept.

    Args:
        variables: Sample-level values per variable, in formula order.
        levels: Level order of categorical variables. The first level is the
            reference. Variables without an entry use sorted levels.

    Raises:
        ValueError: If variables have different lengths, a level list does
            not cover the observed values, or a variable is constant.
    """
    if not variables:
        raise ValueError("At least one design variable is required")
    lengths = {len(values) for values in variables.values()}
    if len(lengths) != 1:
        raise ValueError("Design variables must have one value per sample")
    n_samples = lengths.pop()
    levels = dict(levels or {})
    unknown = sorted(set(levels).difference(variables))
    if unknown:
        raise ValueError(f"levels names unknown design variables: {unknown}")
    columns: list[np.ndarray] = [np.ones(n_samples)]
    names: list[str] = ["(Intercept)"]
    factor_levels: dict[str, tuple[str, ...]] = {}
    for name, raw in variables.items():
        values = np.asarray(raw)
        if _is_numeric(values) and name not in levels:
            column = values.astype(np.float64)
            if not np.all(np.isfinite(column)):
                raise ValueError(f"Design variable {name!r} has non-finite values")
            columns.append(column)
            names.append(name)
            continue
        labels = np.asarray([str(value) for value in values], dtype=object)
        observed = sorted(set(labels.tolist()))
        if name in levels:
            order = [str(level) for level in levels[name]]
            if len(set(order)) != len(order):
                raise ValueError(f"levels of {name!r} contain duplicates")
            missing = sorted(set(observed).difference(order))
            if missing:
                raise ValueError(f"levels of {name!r} do not cover {missing}")
            order = [level for level in order if level in set(observed)]
        else:
            order = observed
        if len(order) < 2:
            raise ValueError(f"Design variable {name!r} has only one level")
        factor_levels[name] = tuple(order)
        for level in order[1:]:
            columns.append((labels == level).astype(np.float64))
            names.append(f"{name}{level}")
    matrix = np.column_stack(columns)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError(
            "The design matrix is not of full rank; a variable is confounded "
            "with another or with the samples"
        )
    return DesignMatrix(matrix=matrix, columns=tuple(names), levels=factor_levels)


def contrast_vector(design: DesignMatrix, contrast: Mapping[str, float]) -> np.ndarray:
    """Return the contrast weights over ``design``'s columns.

    Raises:
        ValueError: If the contrast names an unknown column or is all zero.
    """
    unknown = sorted(set(contrast).difference(design.columns))
    if unknown:
        raise ValueError(
            f"contrast names unknown design columns {unknown}; "
            f"columns are {list(design.columns)}"
        )
    vector = np.array(
        [float(contrast.get(column, 0.0)) for column in design.columns],
        dtype=np.float64,
    )
    if not np.any(vector):
        raise ValueError("contrast must have a non-zero weight")
    return vector
