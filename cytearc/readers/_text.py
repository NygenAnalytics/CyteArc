import math
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from ..utils.arguments import quoted_value
from ..utils.arrays import has_duplicates
from ..utils.logging import logger

type DuplicateFeatureIds = Literal["error", "make_unique"]
type DuplicateCellIds = Literal["error", "make_unique"]

_DUPLICATE_FEATURE_ID_OPTIONS = ("error", "make_unique")
# The most rows that a refusal names.
_NAMED_ROWS = 5
# The types of IDs that are text already, as most readers hold them.
_TEXT_TYPES = frozenset({str, np.str_})
# The values that a decode converts at once.
_DECODE_ROWS = 65_536


def as_text(value: Any) -> str:
    """Return ``value`` as text, decoding byte strings as UTF-8."""
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


class FieldText:
    """One field of a delimited file, as the file holds it.

    read_csv reads text such as ``NA`` or ``null`` as a missing value, also
    in what a converter returns, unless that is not text: its python engine
    always has, and its C engine has from pandas 3.1 on. A converter that
    wraps each field of a column of IDs in this class keeps every field as
    its text.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def field_texts(fields: Iterable[FieldText]) -> np.ndarray:
    """Return the text of each field, as an object array."""
    return np.asarray([field.text for field in fields], dtype=object)


def dataset_place(path: str, dataset: str) -> str:
    """Name a dataset of an HDF5 file as refusals name it.

    For example ``the dataset 'matrix/barcodes' of 'pbmc.h5'``.
    """
    return f"the dataset {dataset!r} of {quoted_value(path)}"


def _of(place: str | None) -> str:
    """Return the words after a row of a refusal that say where the values are."""
    return "" if place is None else f" of {place}"


def _named_rows(rows: list[int]) -> str:
    """Name rows as refusals name them, counting from 1: the first few and how many more."""
    shown = [str(row + 1) for row in rows[:_NAMED_ROWS]]
    more = len(rows) - len(shown)
    if more:
        return f"rows {', '.join(shown)} and {more:,} more"
    if len(shown) == 1:
        return f"row {shown[0]}"
    return f"rows {', '.join(shown[:-1])} and {shown[-1]}"


def utf8_refusal(
    value: bytes, row: int, expectation: str, place: str | None
) -> ValueError:
    """Return the refusal of bytes in a column that are not UTF-8 text.

    Args:
        value: The bytes.
        row: Their row, counting from 0, which the refusal counts from 1.
        expectation: What the column must hold, which the refusal starts
            with, such as ``"H5AD cell IDs must be UTF-8 text"``.
        place: Where the values are, such as
            ``the obs index '_index' of 'pbmc.h5ad'``, or None to name the
            row alone.
    """
    return ValueError(
        f"{expectation}; row {row + 1}{_of(place)} holds {quoted_value(bytes(value))}"
    )


def check_utf8(values: np.ndarray, expectation: str, place: str | None) -> None:
    """Refuse values that include bytes that are not UTF-8 text.

    Raises:
        ValueError: If a value is such bytes, as :func:`utf8_refusal`
            refuses the first.
    """
    for row, value in enumerate(values.tolist()):
        if isinstance(value, bytes):
            try:
                value.decode("utf-8")
            except UnicodeDecodeError:
                raise utf8_refusal(value, row, expectation, place) from None


def utf8_texts(values: np.ndarray, expectation: str, place: str | None) -> list[str]:
    """Return values as text, as :func:`as_text` converts each.

    Args:
        values: One-dimensional values, such as the bytes of a dataset.
        expectation: What the values must be, which a refusal starts with.
        place: Where the values are, which a refusal names, as
            :func:`utf8_refusal` takes it.

    Raises:
        ValueError: If a value is bytes that are not UTF-8 text, as
            :func:`check_utf8` refuses the first.
    """
    texts: list[str] = []
    # Blocks bound the Python objects of the values held beside the texts.
    for start in range(0, len(values), _DECODE_ROWS):
        try:
            texts += [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values[start : start + _DECODE_ROWS].tolist()
            ]
        except UnicodeDecodeError:
            # The refusal names the first such value, which the decode does not.
            check_utf8(values, expectation, place)
            raise
    return texts


def check_identifiers(values: np.ndarray, label: str, place: str | None) -> None:
    """Refuse IDs that an import cannot store.

    Args:
        values: The IDs as the file holds them.
        label: What the IDs are, such as ``"H5AD cell IDs"``.
        place: Where they are, such as ``the obs index '_index' of
            'pbmc.h5ad'``, or None to name the rows alone.

    Raises:
        ValueError: If an ID is bytes that are not UTF-8 text, naming the
            first row that holds them, or if IDs are missing, as None or as
            NaN in a column of numbers, or blank, naming their rows.
    """
    # One pass, decoding inline: a call per ID would take several times as long.
    missing: list[int] = []
    for row, value in enumerate(values.tolist()):
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                raise utf8_refusal(
                    value, row, f"{label} must be UTF-8 text", place
                ) from None
        elif value is None or (
            isinstance(value, float | np.floating) and math.isnan(value)
        ):
            missing.append(row)
            continue
        # The text that an import stores: the decoded bytes, or the str of
        # another value, such as a number.
        if not str(value).strip():
            missing.append(row)
    if missing:
        state = "is" if len(missing) == 1 else "are"
        raise ValueError(
            f"{label} must contain non-empty values; {_named_rows(missing)}"
            f"{_of(place)} {state} missing or blank"
        )


def _identifier_texts(values: Any, label: str, place: str | None) -> np.ndarray:
    """Return identifiers as the text that an import stores.

    Raises:
        ValueError: If the identifiers are not one-dimensional, or
            :func:`check_identifiers` refuses one.
    """
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    items = array.tolist()
    # IDs that are text and not blank need neither a check nor a conversion
    # of their own, and both tests run at C speed.
    if not (set(map(type, items)) <= _TEXT_TYPES and all(map(str.strip, items))):
        check_identifiers(array, label, place)
        items = utf8_texts(array, f"{label} must be UTF-8 text", place)
    return np.asarray(items, dtype=str)


def require_unique_identifiers(
    values: Any, label: str, *, place: str | None = None
) -> None:
    """Reject identifiers that are missing, blank, or repeated.

    Args:
        values: One-dimensional identifiers. Byte strings are decoded as UTF-8.
        label: Name of the identifiers used in error messages, such as
            ``Cell IDs``.
        place: Where the identifiers are, such as ``column 1 of
            'barcodes.tsv'``, which refusals name with the rows. Without it,
            refusals name the rows alone.

    Raises:
        ValueError: If a value is bytes that are not UTF-8 text, missing,
            as None or NaN, or blank, as :func:`check_identifiers` refuses
            it, or a value repeats.
    """
    if has_duplicates(_identifier_texts(values, label, place)):
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
    *,
    place: str | None = None,
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
        place: Where the identifiers are, which refusals name with the rows,
            as :func:`require_unique_identifiers` takes it.

    Returns:
        ``values`` itself when no identifier repeats. Otherwise an object
        array with the text of every identifier, in which each later
        occurrence of a repeated identifier is renamed.

    Raises:
        ValueError: If :func:`check_identifiers` refuses a value, or, with
            ``duplicates="error"``, if a value repeats.
    """
    return _unique_identifiers(
        values,
        label,
        duplicates,
        option="duplicate_feature_ids",
        note=" Feature names are unchanged.",
        place=place,
    )


def unique_cell_identifiers(
    values: np.ndarray,
    label: str,
    duplicates: DuplicateCellIds,
    *,
    place: str | None = None,
) -> np.ndarray:
    """Return cell identifiers that are present and unique.

    Repeated identifiers are renamed as :func:`unique_feature_identifiers`
    renames them, with ``-1``, ``-2``, and so on, as AnnData's
    ``obs_names_make_unique`` names them, when ``duplicates`` is
    ``"make_unique"``.

    Raises:
        ValueError: If :func:`check_identifiers` refuses a value, or, with
            ``duplicates="error"``, if a value repeats. The message names the
            ``duplicate_cell_ids`` option.
    """
    return _unique_identifiers(
        values, label, duplicates, option="duplicate_cell_ids", note="", place=place
    )


def _unique_identifiers(
    values: np.ndarray,
    label: str,
    duplicates: Literal["error", "make_unique"],
    *,
    option: str,
    note: str,
    place: str | None,
) -> np.ndarray:
    texts = _identifier_texts(values, label, place)
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
