"""Write immune receptor contigs as an AIRR Rearrangement file.

The AIRR Community's Rearrangement schema (Vander Heiden et al. 2018) is the
tab-separated format that Immcantation (Change-O, SHazaM, dowser), dandelion,
scirpy and Cell Ranger read and write. A file holds one row per contig, the
schema's required fields first, empty where a value is unknown; logical
fields are written ``T`` and ``F``.

``AirrRearrangementWriter`` writes a file one block of contigs at a time, so a
caller that streams its contigs holds one block at once, and
``write_airr_rearrangements`` writes a table through it. Each block's values
are rendered as UTF-8 bytes and copied into its rows by a compiled loop; text
given as ``EncodedText`` is copied from its packed bytes without being
decoded. A value holding a tab, a newline, a carriage return or a double
quote is quoted as pandas' ``to_csv`` quotes it.

The schema requires unique ``sequence_id`` values. ``suffix_repeated_ids``
names the later occurrences of a repeated ID from 64-bit hashes of every ID,
reading the text of only the IDs whose hashes repeat.
"""

import csv
import gzip
import io
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from types import TracebackType
from typing import Any, NamedTuple, Self

import numpy as np
import pandas as pd
from numba import njit

from ..repertoire.contigs import EncodedText, decode_text, take_text

__all__ = [
    "AIRR_LOGICAL_FIELDS",
    "AIRR_REQUIRED_FIELDS",
    "AirrRearrangementWriter",
    "sequence_id_hashes",
    "suffix_repeated_ids",
    "write_airr_rearrangements",
]

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
# Rows of a table written at once, which bounds the text held in memory.
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


class _Bytes(NamedTuple):
    """A column as UTF-8 bytes: row ``i`` holds value ``codes[i]``.

    Value ``k`` occupies ``data[offsets[k]:offsets[k + 1]]``; a code of -1 is
    an empty value. Without codes, row ``i`` holds value ``i``.
    """

    data: np.ndarray
    offsets: np.ndarray
    codes: np.ndarray | None


# T and F, the values of a logical field, coded 1 and 0.
_FLAG_DATA = np.frombuffer(b"FT", dtype=np.uint8).copy()
_FLAG_OFFSETS = np.arange(3, dtype=np.int64)
# The bytes of a tab and a newline, and the codes of a column without codes.
_TAB = 9
_NEWLINE = 10
_NO_CODES = np.empty(0, dtype=np.int64)


def _packed(strings: Any) -> _Bytes:
    """Return text, none of it missing, as UTF-8 bytes, one value per row."""
    texts = strings.tolist() if isinstance(strings, np.ndarray) else list(strings)
    joined = "".join(texts)
    data = joined.encode("utf-8")
    if len(data) == len(joined):
        # ASCII text has one byte per character.
        lengths = np.fromiter(map(len, texts), np.int64, len(texts))
    else:
        lengths = np.fromiter(
            (len(text.encode("utf-8")) for text in texts), np.int64, len(texts)
        )
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return _Bytes(np.frombuffer(data, dtype=np.uint8).copy(), offsets, None)


@njit(cache=True, nogil=True)
def _integer_bytes(
    values: np.ndarray, missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Write integers in decimal, as ``str`` does, empty where ``missing``."""
    n_values = values.shape[0]
    offsets = np.zeros(n_values + 1, dtype=np.int64)
    ten = np.uint64(10)
    for index in range(n_values):
        length = 0
        if not missing[index]:
            value = values[index]
            if value < 0:
                length = 1
                # -(value + 1) does not overflow at the smallest int64.
                magnitude = np.uint64(-(value + 1)) + np.uint64(1)
            else:
                magnitude = np.uint64(value)
            length += 1
            while magnitude >= ten:
                magnitude //= ten
                length += 1
        offsets[index + 1] = offsets[index] + length
    data = np.empty(offsets[n_values], dtype=np.uint8)
    for index in range(n_values):
        if missing[index]:
            continue
        value = values[index]
        if value < 0:
            data[offsets[index]] = 45
            magnitude = np.uint64(-(value + 1)) + np.uint64(1)
        else:
            magnitude = np.uint64(value)
        position = offsets[index + 1] - 1
        while True:
            data[position] = magnitude % ten + np.uint64(48)
            magnitude //= ten
            position -= 1
            if magnitude == 0:
                break
    return data, offsets


@njit(cache=True, nogil=True)
def _scatter(
    out: np.ndarray,
    cursor: np.ndarray,
    data: np.ndarray,
    offsets: np.ndarray,
    codes: np.ndarray,
    coded: bool,
    separator: int,
) -> bool:
    """Copy each row's value and then ``separator`` to the row's cursor.

    Returns whether a copied value holds a tab, a newline, a carriage return
    or a double quote, which make a value quoted.
    """
    special = False
    for row in range(cursor.shape[0]):
        code = codes[row] if coded else row
        position = cursor[row]
        if code >= 0:
            for index in range(offsets[code], offsets[code + 1]):
                byte = data[index]
                if byte == 9 or byte == 10 or byte == 13 or byte == 34:
                    special = True
                out[position] = byte
                position += 1
        out[position] = separator
        cursor[row] = position + 1
    return special


@njit(cache=True, nogil=True)
def _separate(cursor: np.ndarray, out: np.ndarray, separator: int) -> None:
    """Write ``separator`` at each row's cursor, after an empty value."""
    for row in range(cursor.shape[0]):
        out[cursor[row]] = separator
        cursor[row] += 1


def _lengths(column: _Bytes) -> np.ndarray:
    """Return the bytes of each row's value, reading only the rows' offsets."""
    codes, offsets = column.codes, column.offsets
    if codes is None:
        sizes: np.ndarray = np.diff(offsets)
        return sizes
    # A code of -1 reads offsets that it does not use.
    lengths: np.ndarray = np.where(codes >= 0, offsets[codes + 1] - offsets[codes], 0)
    return lengths


def _strings(column: _Bytes | None, n_rows: int) -> list[str]:
    """Decode each row's value of a column, each distinct value once."""
    if column is None:
        return [""] * n_rows
    codes = np.arange(n_rows) if column.codes is None else column.codes
    needed, inverse = np.unique(codes, return_inverse=True)
    values = np.full(len(needed), "", dtype=object)
    present = needed >= 0
    if present.any():
        values[present] = take_text(column.data, column.offsets, needed[present])
    texts: list[str] = values[inverse.reshape(-1)].tolist()
    return texts


def _rows(columns: list[_Bytes | None], n_rows: int) -> np.ndarray | bytes:
    """Join the columns of rows with tabs, each row ending with a newline.

    A value that holds a tab, a newline, a carriage return or a double quote
    is quoted, with its double quotes doubled, as Python's csv module and
    pandas' ``to_csv`` quote it; ``None`` is an empty column.
    """
    if not n_rows:
        return b""
    widths = np.full(n_rows, len(columns), dtype=np.int64)
    for column in columns:
        if column is not None:
            widths += _lengths(column)
    cursor = np.zeros(n_rows, dtype=np.int64)
    np.cumsum(widths[:-1], out=cursor[1:])
    out = np.empty(int(cursor[-1] + widths[-1]), dtype=np.uint8)
    special = False
    for position, column in enumerate(columns):
        separator = _NEWLINE if position == len(columns) - 1 else _TAB
        if column is None:
            _separate(cursor, out, separator)
            continue
        codes = column.codes
        special |= _scatter(
            out,
            cursor,
            column.data,
            column.offsets,
            _NO_CODES if codes is None else codes,
            codes is not None,
            separator,
        )
    if not special:
        return out
    del out
    buffer = io.StringIO()
    csv.writer(buffer, delimiter="\t", lineterminator="\n").writerows(
        zip(*(_strings(column, n_rows) for column in columns))
    )
    return buffer.getvalue().encode("utf-8")


def _real_text(numbers: np.ndarray, whole: bool) -> np.ndarray:
    """Render real numbers as ``repr`` writes them, or as integers when whole."""
    out = np.full(len(numbers), "", dtype=object)
    present = ~np.isnan(numbers)
    values = numbers[present].tolist()
    out[present] = np.asarray(
        [str(int(number)) for number in values]
        if whole
        else [repr(number) for number in values],
        dtype=object,
    )
    return out


def _encoded_column(values: EncodedText) -> _Bytes:
    return _Bytes(
        np.ascontiguousarray(values.data, dtype=np.uint8),
        np.ascontiguousarray(values.offsets, dtype=np.int64),
        np.ascontiguousarray(values.codes, dtype=np.int64),
    )


def _bytes_column(values: Any, name: str, whole: bool) -> _Bytes:
    """Render one column as UTF-8 bytes, empty where a value is missing.

    The common dtypes of contig tables are rendered directly: text, integers,
    booleans and float64 numbers; other columns as ``_text_column`` renders
    them.
    """
    if isinstance(values, EncodedText):
        if name in AIRR_LOGICAL_FIELDS:
            decoded = decode_text(values.codes, values.categories)
            return _packed(_logical_column(pd.Series(decoded, dtype=object), name))
        return _encoded_column(values)
    if isinstance(values, pd.Series):
        array = (
            values.to_numpy() if isinstance(values.dtype, np.dtype) else values.array
        )
    else:
        array = values
    if name in AIRR_LOGICAL_FIELDS:
        if isinstance(array, pd.arrays.BooleanArray):
            flags = np.where(
                array.isna(), -1, array.to_numpy(dtype=bool, na_value=False)
            )
            return _Bytes(_FLAG_DATA, _FLAG_OFFSETS, flags.astype(np.int64))
        if isinstance(array, np.ndarray) and array.dtype == np.bool_:
            return _Bytes(_FLAG_DATA, _FLAG_OFFSETS, array.astype(np.int64))
    elif isinstance(array, np.ndarray):
        if array.dtype == np.float64:
            return _packed(_real_text(array, whole))
        if array.dtype.kind == "i" or (
            array.dtype.kind == "u" and array.dtype.itemsize < 8
        ):
            return _integers(array.astype(np.int64), np.zeros(len(array), bool))
        if array.dtype.kind == "U":
            return _packed(array)
        if array.dtype == object and pd.api.types.infer_dtype(array, skipna=True) in (
            "string",
            "empty",
        ):
            return _packed(np.where(pd.isna(array), "", array))
    elif isinstance(array, pd.arrays.IntegerArray) and (
        array.dtype.kind == "i" or array.dtype.itemsize < 8
    ):
        return _integers(
            array.to_numpy(dtype=np.int64, na_value=0), np.asarray(array.isna())
        )
    elif isinstance(getattr(array, "dtype", None), pd.StringDtype):
        return _packed(array.to_numpy(dtype=object, na_value=""))
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return _packed(_text_column(series, name, whole))


def _integers(values: np.ndarray, missing: np.ndarray) -> _Bytes:
    data, offsets = _integer_bytes(
        np.ascontiguousarray(values, dtype=np.int64),
        np.ascontiguousarray(missing, dtype=np.bool_),
    )
    return _Bytes(data, offsets, None)


def _n_values(values: Any) -> int:
    return len(values.codes) if isinstance(values, EncodedText) else len(values)


class AirrRearrangementWriter:
    """Write an AIRR Rearrangement TSV file one block of contigs at a time.

    Opening the writer writes the header: the required fields, in the
    schema's order, then ``columns`` without them. Each block is a table of
    contigs, a DataFrame or a mapping of column names to equal-length values,
    rendered as ``write_airr_rearrangements`` renders a table: logical fields
    as ``T`` and ``F``, missing values empty, and the real-number columns of
    ``whole`` as integers. A column may also be ``EncodedText``, whose code
    for each row names one of its packed values, -1 for a missing value; the
    values' bytes are copied without being decoded, and need not be sorted or
    distinct. A block may lack required fields, which are written empty. A path
    ending in ``.gz`` is gzip-compressed at level 6.

    Close the writer, or use it as a context manager::

        with AirrRearrangementWriter("contigs.tsv.gz", ["cell_id"]) as writer:
            for block in blocks:
                writer.write(block)

    Args:
        path: The file to write.
        columns: The columns after the required fields.
        whole: Real-number columns to write as integers, such as counts that
            pandas reads as floats because one is missing. Decide them from
            every row before writing, so the blocks agree.

    Raises:
        TypeError: If ``path`` is not a path.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        columns: Sequence[str] = (),
        *,
        whole: Collection[str] = (),
    ) -> None:
        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path")
        self._rest = tuple(
            column for column in columns if column not in AIRR_REQUIRED_FIELDS
        )
        self._whole = frozenset(whole)
        self._n_rows = 0
        target = os.fspath(path)
        self._handle: Any = (
            gzip.open(target, "wb", compresslevel=6)
            if target.endswith(".gz")
            else open(target, "wb")
        )
        self._handle.write(("\t".join(self.columns) + "\n").encode("utf-8"))

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns of the file, in order."""
        return (*AIRR_REQUIRED_FIELDS, *self._rest)

    @property
    def n_rows(self) -> int:
        """The number of rows written."""
        return self._n_rows

    @property
    def closed(self) -> bool:
        """Whether the file is closed."""
        return self._handle is None

    def write(self, block: pd.DataFrame | Mapping[str, Any]) -> int:
        """Write a block of contigs.

        Args:
            block: One row per contig, its columns AIRR field names.

        Returns:
            The number of rows written.

        Raises:
            ValueError: If the writer is closed, ``block`` holds no
                ``sequence_id``, lacks a column after the required fields or
                holds columns of different lengths, or a logical field holds
                a value that is not logical.
        """
        if self._handle is None:
            raise ValueError("The writer is closed")
        names = set(block.columns if isinstance(block, pd.DataFrame) else block)
        if "sequence_id" not in names:
            raise ValueError("The contigs need a sequence_id")
        missing = [column for column in self._rest if column not in names]
        if missing:
            raise ValueError(f"The contigs have no column {missing}")
        if isinstance(block, pd.DataFrame):
            n_rows = len(block)
        else:
            n_rows = _n_values(block["sequence_id"])
            if any(
                _n_values(block[column]) != n_rows
                for column in self.columns
                if column in names
            ):
                raise ValueError("The columns of a block differ in length")
        columns = [
            _bytes_column(block[column], column, column in self._whole)
            if column in names
            else None
            for column in self.columns
        ]
        self._handle.write(_rows(columns, n_rows))
        self._n_rows += n_rows
        return n_rows

    def close(self) -> None:
        """Close the file; closing again does nothing."""
        if self._handle is not None:
            handle, self._handle = self._handle, None
            handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


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
    with AirrRearrangementWriter(path, rest, whole=whole) as writer:
        for start in range(0, len(table), _CHUNK_ROWS):
            writer.write(table.iloc[start : start + _CHUNK_ROWS])
    return len(table)


def sequence_id_hashes(ids: Any) -> np.ndarray:
    """Return a 64-bit hash of each contig ID, as ``suffix_repeated_ids`` takes.

    Args:
        ids: Contig IDs as text.

    Returns:
        The hashes as uint64.
    """
    text = np.asarray(ids, dtype=object).reshape(-1)
    return np.asarray(pd.util.hash_array(text, categorize=False), dtype=np.uint64)


def suffix_repeated_ids(
    hashes: np.ndarray, ids_at: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Give each later occurrence of a repeated contig ID a unique suffix.

    The schema requires unique ``sequence_id`` values, but Cell Ranger names
    contigs ``<barcode>_contig_<n>`` in the TCR and the BCR files alike. In
    row order, the second occurrence of an ID gets ``_2``, the third ``_3``
    and so on; a suffixed ID that some row holds, or that an earlier
    occurrence took, takes the next number instead. The IDs are compared by
    their hashes, and only the rows whose hashes repeat, or equal the hash of
    a suffixed ID, are read as text, so a collision never merges two IDs.

    Args:
        hashes: The ``sequence_id_hashes`` of every row's ID, in row order.
        ids_at: Return the IDs of rows, as text, given their positions in
            ascending order.

    Returns:
        The rows whose ID repeats an earlier row's, ascending, and their new
        IDs.
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    none = (np.empty(0, dtype=np.int64), np.empty(0, dtype=object))
    ordered = np.sort(hashes)
    shared = ordered[1:][ordered[1:] == ordered[:-1]]
    if not len(shared):
        return none
    candidates = np.flatnonzero(np.isin(hashes, shared))
    texts = np.asarray(ids_at(candidates), dtype=object)
    codes = pd.factorize(texts)[0]
    occurrence = pd.Series(codes).groupby(codes).cumcount().to_numpy()
    later = occurrence > 0
    if not later.any():
        return none
    rows = candidates[later]
    bases = texts[later].tolist()
    numbers = (occurrence[later] + 1).tolist()
    first = [f"{base}_{number}" for base, number in zip(bases, numbers, strict=True)]
    first_hashes = sequence_id_hashes(first)
    # The IDs of the rows of each hash that a suffixed ID has, read once.
    held: dict[int, set[str]] = {}

    def hold(values: np.ndarray) -> None:
        wanted = [value for value in np.unique(values).tolist() if value not in held]
        if not wanted:
            return
        found = np.flatnonzero(np.isin(hashes, np.asarray(wanted, dtype=np.uint64)))
        held.update((value, set()) for value in wanted)
        for value, text in zip(
            hashes[found].tolist(), np.asarray(ids_at(found)).tolist(), strict=True
        ):
            held[value].add(text)

    def stored(text: str, value: np.uint64) -> bool:
        """Whether a row holds the ID ``text``, whose hash is ``value``."""
        at = int(np.searchsorted(ordered, value))
        if at == len(ordered) or ordered[at] != value:
            return False
        hold(np.asarray([value], dtype=np.uint64))
        return text in held[int(value)]

    hold(first_hashes[np.isin(first_hashes, ordered)])
    taken: set[str] = set()
    names = np.empty(len(rows), dtype=object)
    for index, (base, number, name) in enumerate(
        zip(bases, numbers, first, strict=True)
    ):
        value = first_hashes[index]
        while name in taken or stored(name, value):
            number += 1
            name = f"{base}_{number}"
            value = sequence_id_hashes([name])[0]
        taken.add(name)
        names[index] = name
    return rows.astype(np.int64), names
