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
decoded.

Values are written as they are, unquoted, as the AIRR format's dialect asks.
It forbids tabs and newlines in values, which would end a field or a row, and
readers take a carriage return for a row's end too, so a value that holds any
of the three is refused. The format advises against ``@``, ``#``, ``"`` and
``'``, which readers may take differently, so a file whose values hold them
warns with their number and fields when it is closed; ``quality`` and
``quality_alignment`` hold them by the format's own Phred encoding, and are
not counted. A value that starts with a double quote is counted apart, in
every field: airr's ``read_rearrangement``, scirpy's ``read_airr`` and
pandas' default reader take it for an opening quote, and can join or drop
rows without an error, so it has a warning of its own.

The schema requires unique ``sequence_id`` values. ``suffix_repeated_ids``
names the later occurrences of a repeated ID from 64-bit hashes of every ID,
reading the text of only the IDs whose hashes repeat.
"""

import gzip
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from types import TracebackType
from typing import Any, NamedTuple, Self

import numpy as np
import pandas as pd

from ..metadata.records import EncodedText, decode_text
from ..utils.arrays import sorted_unique
from ..utils.logging import logger
from ..utils.numba import njit
from ..utils.arguments import quoted_value

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
# Rendered bytes handed to the file at once, which bounds what compressing
# them holds.
_WRITE_BYTES = 2**20
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
            f"{quoted_value(present.iloc[int(np.flatnonzero(~known)[0])])}"
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
                str(int(number)) if whole else str(float(number))
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
# The class of each byte: 1 for a byte that values must not hold, a tab, a
# newline or a carriage return; 2 for one that they should avoid, a double
# quote, #, ' or @; 0 for any other.
_BYTE_CLASSES = np.zeros(256, dtype=np.uint8)
_BYTE_CLASSES[[9, 10, 13]] = 1
_BYTE_CLASSES[[34, 35, 39, 64]] = 2
# The fields of Phred scores, which the format writes as ASCII 33 to 126.
_PHRED_FIELDS = frozenset({"quality", "quality_alignment"})


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
    count_discouraged: bool,
) -> tuple[int, int, int, int]:
    """Copy each row's value and then ``separator`` to the row's cursor.

    Returns:
        The number of rows whose value holds a tab, a newline or a carriage
        return, and the first of them or -1; with ``count_discouraged``, the
        number of rows whose value holds @, #, ' or a double quote past its
        first byte; and the number of rows whose value starts with a double
        quote.
    """
    n_forbidden = 0
    first = -1
    n_discouraged = 0
    n_leading = 0
    for row in range(cursor.shape[0]):
        code = codes[row] if coded else row
        position = cursor[row]
        if code >= 0:
            start, end = offsets[code], offsets[code + 1]
            # A double quote that starts a value is counted on its own.
            if start < end and data[start] == 34:
                n_leading += 1
                out[position] = 34
                position += 1
                start += 1
            found = 0
            for index in range(start, end):
                byte = data[index]
                found |= _BYTE_CLASSES[byte]
                out[position] = byte
                position += 1
            if found & 1:
                n_forbidden += 1
                if first < 0:
                    first = row
            if found & 2 and count_discouraged:
                n_discouraged += 1
        out[position] = separator
        cursor[row] = position + 1
    return n_forbidden, first, n_discouraged, n_leading


@njit(cache=True, nogil=True)
def _count_forbidden_values(data: np.ndarray, offsets: np.ndarray) -> int:
    """Count the values of packed text that hold a tab, a newline or a carriage return."""
    count = 0
    for value in range(offsets.shape[0] - 1):
        for index in range(offsets[value], offsets[value + 1]):
            if _BYTE_CLASSES[data[index]] & 1:
                count += 1
                break
    return count


@njit(cache=True, nogil=True)
def _forbidden_values(data: np.ndarray, offsets: np.ndarray, count: int) -> np.ndarray:
    """Return the positions of the ``count`` values that ``_count_forbidden_values`` counts.

    The positions are ascending, as int64.
    """
    found = np.empty(count, dtype=np.int64)
    taken = 0
    for value in range(offsets.shape[0] - 1):
        if taken == count:
            break
        for index in range(offsets[value], offsets[value + 1]):
            if _BYTE_CLASSES[data[index]] & 1:
                found[taken] = value
                taken += 1
                break
    return found


@njit(cache=True, nogil=True)
def _mark_forbidden_rows(
    piece: np.ndarray, start: int, bounds: np.ndarray, marks: np.ndarray
) -> None:
    """Mark each row whose bytes in ``piece`` hold a tab, a newline or a carriage return.

    Row ``i`` holds the bytes ``bounds[i]:bounds[i + 1]`` of a ragged field,
    empty where ``bounds[i + 1]`` equals ``bounds[i]``, and ``piece`` holds
    its bytes from ``start`` on, within those of the rows.
    """
    index = 0
    while index < piece.shape[0]:
        if _BYTE_CLASSES[piece[index]] & 1:
            row = np.searchsorted(bounds, start + index, side="right") - 1
            marks[row] = True
            # The rest of the row's bytes need not be read; the loop moves on
            # by at least a byte whatever the bounds hold.
            index = max(index + 1, bounds[row + 1] - start)
        else:
            index += 1


@njit(cache=True, nogil=True)
def _mark_coded_rows(
    values: np.ndarray,
    marked: np.ndarray,
    through: np.ndarray,
    indirect: bool,
    marks: np.ndarray,
) -> None:
    """Mark each row whose code is among ``marked``, which is sorted.

    Row ``i``'s code is ``values[i]``, or with ``indirect`` ``through[values[i]]``;
    a code of -1 is a missing value.
    """
    for row in range(values.shape[0]):
        code = values[row]
        if indirect and code >= 0:
            code = through[code]
        if code < 0:
            continue
        at = np.searchsorted(marked, code)
        if at < marked.shape[0] and marked[at] == code:
            marks[row] = True


def _forbidden_message(found: Sequence[tuple[str, int, str]], subject: str) -> str:
    """Name each field whose values hold a tab, a newline or a carriage return.

    Args:
        found: Each field, the number of its values that hold one, and the
            ``sequence_id`` of the first.
        subject: What holds the values, such as "the contigs".
    """
    parts = [
        f"{count} value{'s' if count > 1 else ''} of {field}, "
        f"{'the first ' if count > 1 else ''}in contig {quoted_value(contig)}"
        for field, count, contig in found
    ]
    return (
        "AIRR values must not hold a tab, a newline or a carriage return, which "
        f"these values of {subject} do: " + "; ".join(parts)
    )


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


def _rows(
    columns: list[_Bytes | None], n_rows: int, counted: list[bool]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Join the columns of rows with tabs, each row ending with a newline.

    Values are copied as they are; ``None`` is an empty column.

    Args:
        columns: The columns, in order.
        n_rows: The rows of each column.
        counted: Whether to count each column's values that hold @, #, '
            or a double quote past their first byte.

    Returns:
        The rows as one array of bytes, and for each column the number of
        rows whose value holds a tab, a newline or a carriage return, the
        first of them or -1, the number of counted values that hold @, #, '
        or a double quote past their first byte, and the number of values
        that start with a double quote.
    """
    marks = [(0, -1, 0, 0)] * len(columns)
    if not n_rows:
        return np.empty(0, dtype=np.uint8), marks
    widths = np.full(n_rows, len(columns), dtype=np.int64)
    for column in columns:
        if column is not None:
            widths += _lengths(column)
    cursor = np.zeros(n_rows, dtype=np.int64)
    np.cumsum(widths[:-1], out=cursor[1:])
    out = np.empty(int(cursor[-1] + widths[-1]), dtype=np.uint8)
    del widths
    for position, column in enumerate(columns):
        separator = _NEWLINE if position == len(columns) - 1 else _TAB
        if column is None:
            _separate(cursor, out, separator)
            continue
        codes = column.codes
        marks[position] = _scatter(
            out,
            cursor,
            column.data,
            column.offsets,
            _NO_CODES if codes is None else codes,
            codes is not None,
            separator,
            counted[position],
        )
    return out, marks


def _value_text(column: _Bytes, row: int) -> str:
    """Decode one row's value of a column of UTF-8 bytes."""
    code = row if column.codes is None else int(column.codes[row])
    if code < 0:
        return ""
    start, end = int(column.offsets[code]), int(column.offsets[code + 1])
    return bytes(column.data[start:end]).decode("utf-8")


def _real_text(numbers: np.ndarray, whole: bool) -> np.ndarray:
    """Render real numbers as Python writes them, or as integers when whole."""
    out = np.full(len(numbers), "", dtype=object)
    present = ~np.isnan(numbers)
    values = numbers[present].tolist()
    out[present] = np.asarray(
        [str(int(number)) for number in values]
        if whole
        else [str(number) for number in values],
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


def _column_names(columns: Any) -> tuple[str, ...]:
    """Check columns given as a sequence of distinct names.

    A name is a field of the header row, so it may not hold a tab, a newline
    or a carriage return either.
    """
    if isinstance(columns, str) or not isinstance(columns, Sequence):
        raise TypeError("columns must be a sequence of column names")
    names = tuple(columns)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("columns must be a sequence of column names")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"columns name {repeated} more than once")
    broken = [name for name in names if {"\t", "\n", "\r"} & set(name)]
    if broken:
        raise ValueError(
            "Column names must not hold a tab, a newline or a carriage return, "
            f"which {broken} do"
        )
    return names


def _refuse_repeated_labels(frame: pd.DataFrame, what: str) -> None:
    """Refuse a DataFrame whose columns repeat a label, naming the labels."""
    repeated = list(dict.fromkeys(frame.columns[frame.columns.duplicated()].tolist()))
    if repeated:
        raise ValueError(f"The {what}'s columns name {repeated} more than once")


def _n_values(values: Any) -> int:
    return len(values.codes) if isinstance(values, EncodedText) else len(values)


class AirrRearrangementWriter:
    """Write an AIRR Rearrangement TSV file one block of contigs at a time.

    Opening the writer writes the header: the required fields, in the
    schema's order, then ``columns`` without them. Each block is a table of
    contigs, a DataFrame or a mapping of column names to equal-length values,
    rendered as ``write_airr_rearrangements`` renders a table: logical fields
    as ``T`` and ``F``, missing values empty, the real-number columns of
    ``whole`` as integers, and other values as they are, unquoted. A column
    may also be ``EncodedText``, whose code for each row names one of its
    packed values, -1 for a missing value; the values' bytes are copied
    without being decoded, and need not be sorted or distinct. A block may
    lack required fields, which are written empty. A path ending in ``.gz`` is
    gzip-compressed at level 6.

    A block that holds a value with a tab, a newline or a carriage return,
    which the AIRR format forbids, is refused before any of its rows is
    written. Values that hold ``@``, ``#``, ``'`` or a double quote past their
    first character, which the format advises against, are written as they
    are and counted, outside ``quality`` and ``quality_alignment``; values
    that start with a double quote, which readers that take quotes read as an
    opening quote, are counted apart, in every field. Closing the writer warns
    with each count and its fields.

    The schema requires unique ``sequence_id`` values, and the writer writes
    each block as given, so a caller whose contigs' IDs may repeat gives the
    later ones suffixes first, as :func:`suffix_repeated_ids` chooses them.
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
        TypeError: If ``path`` is not a path, or ``columns`` is not a sequence
            of column names.
        ValueError: If ``columns`` names a column more than once, or a name
            holds a tab, a newline or a carriage return.
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
        names = _column_names(columns)
        self._rest = tuple(
            column for column in names if column not in AIRR_REQUIRED_FIELDS
        )
        self._whole = frozenset(whole)
        self._n_rows = 0
        # The values of each field that hold @, #, ' or a double quote past
        # their first character, and those that start with a double quote.
        self._discouraged: dict[str, int] = {}
        self._leading_quotes: dict[str, int] = {}
        target = os.fspath(path)
        self._file: Any = open(target, "wb")
        # A compressed file's header names no file and no time, so the same
        # contigs give the same bytes.
        self._handle: Any = (
            gzip.GzipFile(
                filename="", mode="wb", compresslevel=6, fileobj=self._file, mtime=0
            )
            if target.endswith(".gz")
            else self._file
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
    def n_discouraged(self) -> int:
        """The values written that hold ``@``, ``#``, ``'`` or an inner ``"``.

        The AIRR format advises against these characters. A double quote
        counts here past a value's first character, and values of ``quality``
        and ``quality_alignment`` are not counted.
        """
        return sum(self._discouraged.values())

    @property
    def n_leading_quotes(self) -> int:
        """The number of values written that start with a double quote, in any field."""
        return sum(self._leading_quotes.values())

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
                ``sequence_id``, lacks a column after the required fields,
                holds columns of different lengths or, as a DataFrame, repeats
                a column's label, a logical field holds a value that is not
                logical, or a value holds a tab, a newline or a carriage
                return; the block is then not written.
        """
        if self._handle is None:
            raise ValueError("The writer is closed")
        if isinstance(block, pd.DataFrame):
            _refuse_repeated_labels(block, "block")
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
        rendered, marks = _rows(
            columns, n_rows, [column not in _PHRED_FIELDS for column in self.columns]
        )
        forbidden = sorted(
            (first, position, count)
            for position, (count, first, _, _) in enumerate(marks)
            if count
        )
        if forbidden:
            # The block's sequence_id, the first column, names the contigs.
            ids = columns[0]
            assert ids is not None
            raise ValueError(
                _forbidden_message(
                    [
                        (self.columns[position], count, _value_text(ids, first))
                        for first, position, count in forbidden
                    ],
                    "the block",
                )
            )
        for start in range(0, len(rendered), _WRITE_BYTES):
            self._handle.write(rendered[start : start + _WRITE_BYTES])
        self._n_rows += n_rows
        for name, (_, _, discouraged, leading) in zip(self.columns, marks, strict=True):
            if discouraged:
                self._discouraged[name] = self._discouraged.get(name, 0) + discouraged
            if leading:
                self._leading_quotes[name] = self._leading_quotes.get(name, 0) + leading
        return n_rows

    def close(self) -> None:
        """Close the file; closing again does nothing.

        A file whose values start with a double quote warns with their number
        and fields, and so does one whose values hold ``@``, ``#``, ``'`` or a
        double quote past their first character, which the AIRR format advises
        against.
        """
        if self._handle is not None:
            handle, self._handle = self._handle, None
            handle.close()
            self._file.close()
            if self._leading_quotes:
                logger.warning(
                    self._counted(self._leading_quotes, "start", "starts")
                    + " with a double quote, which airr's read_rearrangement, "
                    "scirpy's read_airr and pandas' default reader take for an "
                    "opening quote, so they can join or drop rows without an "
                    "error; pandas reads every value as written with "
                    "sep='\\t', dtype=str, keep_default_na=False and "
                    "quoting=csv.QUOTE_NONE"
                )
            if self._discouraged:
                logger.warning(
                    self._counted(self._discouraged, "hold", "holds")
                    + " @, #, ' or a double quote past the first character, which "
                    "the AIRR format advises against since readers may take them "
                    "differently; they are written as they are"
                )

    def _counted(self, counts: dict[str, int], verb: str, verb_one: str) -> str:
        """Name how many values of which fields, in the file's order, and the verb."""
        fields = [name for name in self.columns if name in counts]
        named = (
            fields[0]
            if len(fields) == 1
            else ", ".join(fields[:-1]) + " and " + fields[-1]
        )
        count = sum(counts.values())
        return (
            f"{count} value{'s' if count > 1 else ''} of {named} "
            f"{verb if count > 1 else verb_one}"
        )

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
    ``columns`` when given. Values are written as they are, unquoted. A value
    that holds a tab, a newline or a carriage return, which the AIRR format
    forbids, is refused before the file is opened, as is a logical field's
    value that is not logical, so a refused table leaves no file. A path
    ending in ``.gz`` is gzip-compressed.

    Args:
        table: One row per contig, its columns AIRR field names.
        path: The file to write.
        columns: The columns after the required fields, or None for the
            table's other columns.

    Returns:
        The number of rows written.

    Raises:
        TypeError: If ``table`` is not a DataFrame, ``path`` is not a path, or
            ``columns`` is not a sequence of column names.
        ValueError: If ``table`` repeats a column's label, holds no
            ``sequence_id`` or a ``sequence_id`` more than once, which the
            schema forbids, a value or a column's name holds a tab, a newline
            or a carriage return, a logical field holds a value that is not
            logical, or ``columns`` names a column more than once or one that
            the table lacks.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if not isinstance(path, str | os.PathLike):
        raise TypeError("path must be a path")
    _refuse_repeated_labels(table, "table")
    if "sequence_id" not in table.columns:
        raise ValueError("The contigs need a sequence_id")
    ids = table["sequence_id"]
    if not ids.is_unique:
        repeated = ids[ids.duplicated()].iloc[0]
        raise ValueError(
            f"sequence_id {quoted_value(repeated)} repeats, but the schema requires each "
            "contig's ID to be unique"
        )
    names = list(table.columns) if columns is None else list(_column_names(columns))
    rest = [column for column in names if column not in AIRR_REQUIRED_FIELDS]
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
    # Every value is checked before the file is opened, so a refused table
    # leaves no file.
    forbidden = _checked_columns(table, order, whole)
    if forbidden:
        raise ValueError(_forbidden_message(forbidden, "the table"))
    with AirrRearrangementWriter(path, rest, whole=whole) as writer:
        for start in range(0, len(table), _CHUNK_ROWS):
            writer.write(table.iloc[start : start + _CHUNK_ROWS])
    return len(table)


def _checked_columns(
    table: pd.DataFrame, order: Sequence[str], whole: Collection[str]
) -> list[tuple[str, int, str]]:
    """Check a table's values as the writer will render them, before any is written.

    Each column of ``order`` that the table holds is rendered as the writer
    renders it, a chunk of rows at a time. A logical field raises for a value
    that is not logical, and other columns are searched for values that hold a
    tab, a newline or a carriage return. Columns of booleans, and columns of
    numbers outside logical fields, render as ``T`` and ``F``, numbers,
    ``True`` and ``False``, which need no check.

    Returns:
        Each column whose values hold such a character, the number of its
        values that hold one, and the ``sequence_id`` of the first, ordered by
        that row.

    Raises:
        ValueError: If a logical field holds a value that is not logical.
    """
    found = []
    for column in order:
        if column not in table.columns:
            continue
        dtype = table[column].dtype
        logical = column in AIRR_LOGICAL_FIELDS
        if pd.api.types.is_bool_dtype(dtype) or (
            not logical and pd.api.types.is_numeric_dtype(dtype)
        ):
            continue
        count, first = 0, -1
        for start in range(0, len(table), _CHUNK_ROWS):
            values = _bytes_column(
                table[column].iloc[start : start + _CHUNK_ROWS], column, column in whole
            )
            if logical:
                # Rendering a logical field checks its values, and T and F
                # hold no forbidden character.
                continue
            # Only logical fields of booleans render with codes, so each row
            # here holds its own value.
            assert values.codes is None
            n_found = _count_forbidden_values(values.data, values.offsets)
            if n_found and first < 0:
                first = start + int(
                    _forbidden_values(values.data, values.offsets, 1)[0]
                )
            count += n_found
        if count:
            found.append((first, column, count))
    ids = table["sequence_id"]
    # Each first contig is named by its ID as the file would write it.
    return [
        (
            column,
            count,
            _value_text(
                _bytes_column(ids.iloc[first : first + 1], "sequence_id", False), 0
            ),
        )
        for first, column, count in sorted(found, key=lambda item: item[0])
    ]


def sequence_id_hashes(ids: Any) -> np.ndarray:
    """Return a 64-bit hash of each contig ID, as ``suffix_repeated_ids`` takes.

    Args:
        ids: Contig IDs as text.

    Returns:
        The hashes as uint64.
    """
    text = np.asarray(ids, dtype=object).reshape(-1)
    return np.asarray(pd.util.hash_array(text, categorize=False), dtype=np.uint64)


def _found(ordered: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Whether each value is among ``ordered``, which is sorted and not empty.

    Searching the sorted values holds two int64 arrays and a mask the length
    of ``values``, where ``np.isin`` sorts both arrays together.
    """
    at = np.searchsorted(ordered, values)
    np.minimum(at, len(ordered) - 1, out=at)
    found: np.ndarray = ordered[at] == values
    return found


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
    candidates = np.flatnonzero(_found(shared, hashes))
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
        wanted = [
            value for value in sorted_unique(values).tolist() if value not in held
        ]
        if not wanted:
            return
        found = np.flatnonzero(_found(np.asarray(wanted, dtype=np.uint64), hashes))
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

    hold(first_hashes[_found(ordered, first_hashes)])
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
