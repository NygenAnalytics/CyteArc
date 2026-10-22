from typing import Any, Literal

import numpy as np

from ..utils.arrays import has_duplicates
from ..utils.logging import logger

type DuplicateFeatureIds = Literal["error", "make_unique"]
type DuplicateCellIds = Literal["error", "make_unique"]

_DUPLICATE_FEATURE_ID_OPTIONS = ("error", "make_unique")


def as_text(value: Any) -> str:
    """Return ``value`` as text, decoding byte strings as UTF-8."""
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _identifier_texts(values: Any, label: str) -> np.ndarray:
    """Return identifiers as text after rejecting missing and blank ones."""
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    texts = np.asarray(
        ["" if value is None else as_text(value) for value in array],
        dtype=str,
    )
    if texts.size and bool(np.any(np.char.strip(texts) == "")):
        raise ValueError(f"{label} must contain non-empty values")
    return texts


def require_unique_identifiers(values: Any, label: str) -> None:
    """Reject identifiers that are missing, blank, or repeated.

    Args:
        values: One-dimensional identifiers. Byte strings are decoded as UTF-8.
        label: Name of the identifiers used in error messages, such as
            ``Cell IDs``.

    Raises:
        ValueError: If a value is missing or blank, or a value repeats.
    """
    if has_duplicates(_identifier_texts(values, label)):
        raise ValueError(f"{label} must contain unique values")


def duplicate_feature_ids_option(value: Any) -> DuplicateFeatureIds:
    """Return a validated ``duplicate_feature_ids`` reader argument.

    Raises:
        ValueError: If ``value`` is neither ``"error"`` nor ``"make_unique"``.
    """
    return _duplicate_ids_option(value, "duplicate_feature_ids")


def duplicate_cell_ids_option(value: Any) -> DuplicateCellIds:
    """Return a validated ``duplicate_cell_ids`` reader argument.

    Raises:
        ValueError: If ``value`` is neither ``"error"`` nor ``"make_unique"``.
    """
    return _duplicate_ids_option(value, "duplicate_cell_ids")


def _duplicate_ids_option(value: Any, option: str) -> Literal["error", "make_unique"]:
    if isinstance(value, str) and value in _DUPLICATE_FEATURE_ID_OPTIONS:
        return "error" if value == "error" else "make_unique"
    raise ValueError(f"{option} must be 'error' or 'make_unique', not {value!r}")


def _later_occurrences(texts: np.ndarray) -> np.ndarray:
    """Return the positions, in file order, of values that occurred before."""
    # A stable sort keeps equal values in file order, so each value of a run
    # of equal values but the first is a later occurrence.
    order = np.argsort(texts, kind="stable")
    ordered = texts[order]
    later = np.zeros(texts.size, dtype=bool)
    later[order[1:]] = ordered[1:] == ordered[:-1]
    return np.flatnonzero(later)


def unique_feature_identifiers(
    values: np.ndarray,
    label: str,
    duplicates: DuplicateFeatureIds,
) -> np.ndarray:
    """Return feature identifiers that are present and unique.

    With ``duplicates="error"``, a repeated identifier raises ValueError. With
    ``"make_unique"``, the first occurrence of each identifier keeps it, and
    each later occurrence, in file order, takes the identifier followed by
    ``-1``, ``-2``, and so on, as AnnData's ``var_names_make_unique`` names
    them. A suffixed name that is already an
    identifier, or that an earlier occurrence took, is skipped, so the result
    is unique even when the source also holds an identifier such as
    ``GENE-1``. A warning reports how many identifiers were renamed.

    Args:
        values: One-dimensional identifiers. Byte strings are decoded as UTF-8.
        label: Name of the identifiers used in messages, such as
            ``Feature IDs``.
        duplicates: ``"error"`` or ``"make_unique"``.

    Returns:
        ``values`` itself when no identifier repeats. Otherwise an object
        array with the text of every identifier, in which each later
        occurrence of a repeated identifier is renamed.

    Raises:
        ValueError: If a value is missing or blank, or, with
            ``duplicates="error"``, if a value repeats.
    """
    return _unique_identifiers(
        values,
        label,
        duplicates,
        option="duplicate_feature_ids",
        note=" Feature names are unchanged.",
    )


def unique_cell_identifiers(
    values: np.ndarray,
    label: str,
    duplicates: DuplicateCellIds,
) -> np.ndarray:
    """Return cell identifiers that are present and unique.

    Repeated identifiers are renamed as :func:`unique_feature_identifiers`
    renames them, with ``-1``, ``-2``, and so on, as AnnData's
    ``obs_names_make_unique`` names them, when ``duplicates`` is
    ``"make_unique"``.

    Raises:
        ValueError: If a value is missing or blank, or, with
            ``duplicates="error"``, if a value repeats. The message names the
            ``duplicate_cell_ids`` option.
    """
    return _unique_identifiers(
        values, label, duplicates, option="duplicate_cell_ids", note=""
    )


def _unique_identifiers(
    values: np.ndarray,
    label: str,
    duplicates: Literal["error", "make_unique"],
    *,
    option: str,
    note: str,
) -> np.ndarray:
    texts = _identifier_texts(values, label)
    if not has_duplicates(texts):
        return values
    if duplicates == "error":
        raise ValueError(
            f"{label} must contain unique values; pass "
            f"{option}='make_unique' to rename repeated IDs"
        )
    resolved = texts.astype(object)
    taken = set(texts.tolist())
    suffixes: dict[str, int] = {}
    later = _later_occurrences(texts)
    for position in later.tolist():
        base = str(texts[position])
        suffix = suffixes.get(base, 0)
        while True:
            suffix += 1
            name = f"{base}-{suffix}"
            if name not in taken:
                break
        suffixes[base] = suffix
        taken.add(name)
        resolved[position] = name
    examples = ", ".join(
        f"{str(texts[position])!r} to {resolved[position]!r}"
        for position in later[:3].tolist()
    )
    logger.warning(
        f"{label} repeat {len(suffixes)} values: {later.size} later occurrences "
        f"were renamed with a numbered suffix, for example {examples}.{note}"
    )
    return resolved
