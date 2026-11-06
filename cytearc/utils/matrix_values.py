"""What an imported expression matrix holds: counts, or values normalized before import.

The import reference's normalized values section states the rules that
:func:`decide_matrix_values` applies. Logs are undone once, at import, by
:func:`linear_values`, so every stored value is linear and scaled per cell.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import DTypeLike

from .count_values import CountValueRange

MatrixValuesKind = Literal["counts", "normalized", "log_normalized", "log2_normalized"]
MATRIX_VALUES: tuple[MatrixValuesKind, ...] = (
    "counts",
    "normalized",
    "log_normalized",
    "log2_normalized",
)
#: The number of cells, from the first, whose values decide what a matrix holds.
DETECTION_CELLS = 100
# A reading fits when the totals of the sampled cells spread by at most this
# share of their median, or when this share of at least _MULTIPLE_EVIDENCE
# values, besides each cell's smallest, lie within the tolerance of a whole
# multiple of their cell's smallest value. A cell's smallest value is always a
# multiple of itself, so it carries no evidence.
_TOTAL_SPREAD = 0.01
_MULTIPLE_SHARE = 0.99
_MULTIPLE_TOLERANCE = 0.05
_MULTIPLE_EVIDENCE = 50
# The factor by which each log kind's values are multiplied before ``expm1``.
_LOG_SCALES = {"log_normalized": 1.0, "log2_normalized": float(np.log(2.0))}


@dataclass(frozen=True)
class MatrixValues:
    """What one imported assay holds.

    Attributes:
        kind: ``counts``, or the kind of normalized values that the source held.
        cellTotal: The total that the normalized values of every sampled cell
            share, or None.
    """

    kind: MatrixValuesKind
    cellTotal: float | None = None


#: The ``values=`` argument of an import: one kind for every assay, a kind for
#: each assay named, or None to decide every assay from its values.
ImportValues = MatrixValuesKind | Mapping[str, MatrixValuesKind] | None


def import_values_argument(
    values: object, assays: Sequence[str] | None = None
) -> dict[str, MatrixValuesKind | None] | MatrixValuesKind | None:
    """Return a validated ``values=`` argument of an import.

    A mapping must name only ``assays``, when they are given.

    Raises:
        ValueError: If a kind is not one of ``MATRIX_VALUES``, or a mapping
            names an assay that the import does not write.
    """
    if not isinstance(values, Mapping):
        return matrix_values_argument(values)
    unknown = sorted(set(values) - set(assays)) if assays is not None else []
    if unknown:
        raise ValueError(
            f"values names assays that the import does not write: {unknown}; it "
            f"writes {list(assays or ())}"
        )
    return {str(name): matrix_values_argument(kind) for name, kind in values.items()}


def matrix_values_argument(values: object) -> MatrixValuesKind | None:
    """Return a validated ``values=`` argument.

    Raises:
        ValueError: If ``values`` is neither None nor one of ``MATRIX_VALUES``.
    """
    if values is None:
        return None
    if isinstance(values, str) and values in MATRIX_VALUES:
        return values
    choices = ", ".join(repr(kind) for kind in MATRIX_VALUES)
    raise ValueError(f"values must be None or one of {choices}, not {values!r}")


def _reading(data: np.ndarray, kind: str) -> np.ndarray:
    if kind == "normalized":
        return np.asarray(data, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        values: np.ndarray = np.expm1(
            np.multiply(data, _LOG_SCALES[kind], dtype=np.float64)
        )
    return values


def fitting_readings(sample: Any) -> dict[str, float | None]:
    """Return each normalized kind under which ``sample``'s cells fit.

    ``sample`` holds whole cells, one row per cell. Each fitting kind maps to
    the median total of the cells when their totals agree within 1%, and to
    None when only the whole multiples fit.
    """
    from scipy.sparse import csr_matrix

    cells = csr_matrix(sample, dtype=np.float64, copy=True)
    cells.sum_duplicates()
    cells.eliminate_zeros()
    lengths = np.diff(cells.indptr)
    starts = cells.indptr[:-1][lengths > 0]
    fits: dict[str, float | None] = {}
    if starts.size == 0:
        return fits
    for kind in MATRIX_VALUES[1:]:
        values = _reading(cells.data, kind)
        if not (np.isfinite(values).all() and (values > 0).all()):
            continue
        totals = np.add.reduceat(values, starts)
        median = float(np.median(totals))
        shared = starts.size > 1 and float(np.ptp(totals)) <= _TOTAL_SPREAD * median
        smallest = np.repeat(np.minimum.reduceat(values, starts), lengths[lengths > 0])
        ratios = values / smallest
        near = int(
            np.count_nonzero(np.abs(ratios - np.rint(ratios)) <= _MULTIPLE_TOLERANCE)
        )
        # Leave out each cell's smallest value, which is always near.
        evidence = values.size - starts.size
        multiples = (
            evidence >= _MULTIPLE_EVIDENCE
            and (near - starts.size) >= _MULTIPLE_SHARE * evidence
        )
        if shared or multiples:
            fits[kind] = median if shared else None
    return fits


def decide_matrix_values(
    declared: MatrixValuesKind | None,
    value_range: CountValueRange,
    sample: Callable[[], Any],
    *,
    assay: str,
) -> MatrixValues:
    """Return what an assay holds, from its declaration and its values.

    ``sample`` returns the first ``DETECTION_CELLS`` cells, one row per cell;
    it is called only for values that are not whole. A declaration of counts
    is trusted unless the values fit only a log, so that fractional counts
    import as counts; a declaration of normalized values must be one of the
    readings that fit, since a log undone or kept by mistake changes every
    value.

    Raises:
        ValueError: If a value is negative; if undeclared values that are not
            whole fit no single reading, or the sampled cells hold only whole
            numbers; if a declaration is not among the readings that fit; or
            if a log is declared for whole numbers.
    """
    if value_range.negative:
        raise ValueError(
            f"Assay {assay!r} holds negative values, as values centred or scaled "
            "for PCA do. CyteArc imports counts and normalized values, which are "
            "never negative; import the counts or the normalized values instead"
        )
    if value_range.integral:
        if declared in _LOG_SCALES:
            raise ValueError(
                f"Assay {assay!r} was declared values={declared!r}, but every value "
                "is a whole number, which logs of normalized values are not. Pass "
                "values='counts', or values='normalized' for normalized values "
                "rounded to whole numbers"
            )
        return MatrixValues("counts" if declared is None else declared)
    cells = sample()
    fits = fitting_readings(cells)
    data = np.asarray(getattr(cells, "data", cells))
    # Whole sampled values beside fractions elsewhere say nothing about the rest.
    whole_sample = data.size > 0 and bool(np.all(data == np.rint(data)))
    fitting = ", ".join(f"values={kind!r}" for kind in fits)
    if declared == "counts":
        if len(fits) == 1 and next(iter(fits)) in _LOG_SCALES:
            raise ValueError(
                f"Assay {assay!r} was declared values='counts', but its first "
                f"{DETECTION_CELLS} cells fit only {fitting}. Pass {fitting}"
            )
        return MatrixValues("counts")
    if declared is None:
        if whole_sample or len(fits) != 1:
            why = (
                "hold only whole numbers"
                if whole_sample
                else "do not show what they are"
            )
            raise ValueError(
                f"Assay {assay!r} holds values that are not whole numbers, and its "
                f"first {DETECTION_CELLS} cells {why}. Pass values='normalized' for "
                "counts scaled by one factor per cell, values='log_normalized' for "
                "their natural log (log1p, as scanpy and Seurat write it), "
                "values='log2_normalized' for log2(x + 1), or values='counts' for "
                "fractional counts such as alevin's or RSEM's estimates. Values "
                "rounded to whole numbers import as counts"
            )
        decided = cast(MatrixValuesKind, next(iter(fits)))
        return MatrixValues(decided, fits[decided])
    if fits and declared not in fits:
        raise ValueError(
            f"Assay {assay!r} was declared values={declared!r}, but its first "
            f"{DETECTION_CELLS} cells fit only {fitting}. Pass one of them"
        )
    if whole_sample and declared in _LOG_SCALES:
        raise ValueError(
            f"Assay {assay!r} was declared values={declared!r}, but its first "
            f"{DETECTION_CELLS} cells hold only whole numbers, which logs of "
            "normalized values are not"
        )
    return MatrixValues(declared, fits.get(declared))


def linear_values(
    values: np.ndarray, kind: MatrixValuesKind, dtype: DTypeLike, *, assay: str
) -> np.ndarray:
    """Return ``values`` with the log of a log kind undone, in ``dtype``.

    Other kinds return ``values`` unchanged. The log is undone in float64.

    Raises:
        ValueError: If a value is too large to be the log of a normalized
            value, so that undoing it overflows ``dtype``.
    """
    if kind not in _LOG_SCALES:
        return values
    with np.errstate(over="ignore"):
        linear: np.ndarray = np.expm1(
            np.multiply(values, _LOG_SCALES[kind], dtype=np.float64)
        ).astype(dtype, copy=False)
    finite = np.isfinite(linear)
    if not finite.all():
        value = np.asarray(values).ravel()[np.flatnonzero(~finite.ravel())[0]]
        raise ValueError(
            f"Undoing the log of the value {float(value)!r} of assay {assay!r} "
            f"overflows {np.dtype(dtype)}, so it is not the log of a normalized "
            f"value. Check values={kind!r}"
        )
    return linear


def linear_coo(
    matrix: Any, kind: MatrixValuesKind, dtype: DTypeLike, *, assay: str
) -> Any:
    """Return a sparse block as COO with the log of a log kind undone.

    Duplicate coordinates are summed before the log is undone, since the
    summed value is the stored one. Other kinds return ``matrix`` unchanged.
    """
    from scipy.sparse import coo_matrix

    from .arrays import canonicalize_sparse

    if kind not in _LOG_SCALES:
        return matrix
    canonical = canonicalize_sparse(coo_matrix(matrix, copy=True))
    return coo_matrix(
        (
            linear_values(canonical.data, kind, dtype, assay=assay),
            (canonical.row, canonical.col),
        ),
        shape=canonical.shape,
    )
