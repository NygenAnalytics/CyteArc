"""Write immune receptor contigs as an AIRR Rearrangement file.

The AIRR Community's Rearrangement schema (Vander Heiden et al. 2018) is the
tab-separated format that Immcantation (Change-O, SHazaM, dowser), dandelion,
scirpy and Cell Ranger read and write. A file holds one row per contig, the
schema's required fields first, empty where a value is unknown; logical
fields are written ``T`` and ``F``.
"""

import gzip
import io
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["AIRR_LOGICAL_FIELDS", "AIRR_REQUIRED_FIELDS", "write_airr_rearrangements"]

#: The fields that every AIRR Rearrangement file holds, in order.
AIRR_REQUIRED_FIELDS = (
    "sequence_id",
    "sequence",
    "rev_comp",
    "productive",
    "v_call",
    "d_call",
    "j_call",
    "sequence_alignment",
    "germline_alignment",
    "junction",
    "junction_aa",
    "v_cigar",
    "d_cigar",
    "j_cigar",
)
# Rows rendered at once, which bounds the text held in memory.
_CHUNK_ROWS = 100_000
#: The logical fields, written ``T`` and ``F``.
AIRR_LOGICAL_FIELDS = frozenset(
    {
        "rev_comp",
        "productive",
        "vj_in_frame",
        "stop_codon",
        "complete_vdj",
        "is_cell",
        "high_confidence",
        "full_length",
    }
)


_TRUE = ("T", "TRUE", "1")
_FALSE = ("F", "FALSE", "0")


def _logical_column(values: pd.Series, name: str) -> np.ndarray:
    """Render a logical field as ``T`` and ``F``, empty when missing.

    Booleans, 0 and 1, and the text ``T``, ``F``, ``TRUE`` and ``FALSE`` in
    any case are logical values, as a table read from an AIRR file holds them.
    """
    out = np.full(len(values), "", dtype=object)
    missing = values.isna().to_numpy()
    present = values[~missing]
    if pd.api.types.is_bool_dtype(present.dtype):
        out[~missing] = np.where(present.to_numpy(dtype=bool), "T", "F")
        return out
    text = (
        (
            present.astype(float).map(lambda number: f"{number:g}")
            if pd.api.types.is_numeric_dtype(present.dtype)
            else present.astype(str)
        )
        .str.strip()
        .str.upper()
    )
    flags = text.isin(_TRUE).to_numpy()
    known = flags | text.isin(_FALSE).to_numpy() | (text == "").to_numpy()
    if not known.all():
        raise ValueError(
            f"{name} must hold logical values, but holds "
            f"{present.iloc[int(np.flatnonzero(~known)[0])]!r}"
        )
    rendered = np.where(flags, "T", "F").astype(object)
    rendered[(text == "").to_numpy()] = ""
    out[~missing] = rendered
    return out


def _whole(values: pd.Series) -> bool:
    """Whether a column of real numbers holds only whole numbers, or none."""
    numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
    present = numbers[~np.isnan(numbers)]
    return bool(np.isfinite(present).all() and (present == np.round(present)).all())


def _text_column(values: pd.Series, name: str, whole: bool) -> np.ndarray:
    """Render one column as text, empty when missing.

    Logical fields are written ``T`` and ``F``. With ``whole``, real numbers
    are written as integers, as AIRR's integer fields need them.
    """
    if name in AIRR_LOGICAL_FIELDS:
        return _logical_column(values, name)
    missing = values.isna().to_numpy()
    if pd.api.types.is_float_dtype(values.dtype):
        numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
        rendered = np.asarray(
            [
                str(int(number)) if whole else repr(float(number))
                for number in np.where(missing, 0.0, numbers)
            ],
            dtype=object,
        )
    else:
        rendered = values.astype("string").fillna("").astype(str).to_numpy()
    out = np.asarray(rendered, dtype=object)
    out[missing] = ""
    return out


def write_airr_rearrangements(
    table: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    columns: Sequence[str] | None = None,
) -> int:
    """Write a table of contigs as an AIRR Rearrangement TSV file.

    The required fields come first, in the schema's order, empty when the
    table lacks them, then the table's other columns in their order, or
    ``columns`` when given. A path ending in ``.gz`` is gzip-compressed.

    Args:
        table: One row per contig, its columns AIRR field names.
        path: The file to write.
        columns: The columns after the required fields, or None for the
            table's other columns.

    Returns:
        The number of rows written.

    Raises:
        TypeError: If ``path`` is not a path.
        ValueError: If ``table`` holds no ``sequence_id``, or names a column
            that it lacks.
    """
    if not isinstance(path, str | os.PathLike):
        raise TypeError("path must be a path")
    if "sequence_id" not in table.columns:
        raise ValueError("The contigs need a sequence_id")
    rest = (
        [column for column in table.columns if column not in AIRR_REQUIRED_FIELDS]
        if columns is None
        else [column for column in columns if column not in AIRR_REQUIRED_FIELDS]
    )
    missing = [column for column in rest if column not in table.columns]
    if missing:
        raise ValueError(f"The contigs have no column {missing}")
    order = [*AIRR_REQUIRED_FIELDS, *rest]
    # Real numbers that are all whole, such as counts that pandas reads as
    # floats because one is missing, are written as integers.
    whole = {
        column
        for column in order
        if column in table.columns
        and pd.api.types.is_float_dtype(table[column].dtype)
        and _whole(table[column])
    }
    target = os.fspath(path)
    binary: Any = (
        gzip.open(target, "wb") if target.endswith(".gz") else open(target, "wb")
    )
    with binary, io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
        handle.write("\t".join(order) + "\n")
        for start in range(0, len(table), _CHUNK_ROWS):
            chunk = table.iloc[start : start + _CHUNK_ROWS]
            rendered = {
                column: (
                    _text_column(chunk[column], column, column in whole)
                    if column in chunk.columns
                    else np.full(len(chunk), "", dtype=object)
                )
                for column in order
            }
            pd.DataFrame(rendered, columns=order).to_csv(
                handle, sep="\t", index=False, header=False, lineterminator="\n"
            )
    return len(table)
