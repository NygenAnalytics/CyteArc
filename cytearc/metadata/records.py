"""Long tables of records that belong to cells, encoded for storage.

The files of many single-cell libraries list records one per row, each joined
to its cell by barcode: the receptor contigs of a V(D)J library, the molecules
of a lineage barcode library, the alleles of a lineage recorder. These
functions parse the text of such rows chunk by chunk, naming the file lines of
the values they refuse, and encode the parsed records as one table ordered by
cell. Each field of a table has a kind:

- Text (``TEXT``), such as a gene call or a barcode, is stored once per
  distinct value: each record holds the position of its value among the
  field's distinct values, sorted, so equal text has equal codes and the
  order of codes is the order of the text.
- Ragged text (``RAGGED``), which nearly every record has its own of, such as
  an ID or a full sequence, is stored record by record, as UTF-8 bytes with
  offsets.
- Integers (``INTEGER``) are whole numbers that are never negative, such as
  counts and positions. Flags (``FLAG``) are true or false, and numbers
  (``NUMBER``) are real.

A missing value is stored as follows:

- Text has code -1, which sorts before every value, as the empty string sorts
  before every text.
- Ragged text has an offset of -1 and no bytes, so it differs from an empty
  value.
- An integer is -1, since counts and positions are never negative.
- A flag is -1, False is 0 and True is 1.
- A number is NaN.

These functions never touch storage.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.numba import njit

__all__ = [
    "FLAG",
    "INTEGER",
    "MISSING_TEXT",
    "NUMBER",
    "RAGGED",
    "TEXT",
    "EncodedText",
    "FieldKind",
    "RecordEncoder",
    "RecordTable",
    "chunk_rows_for",
    "decode_ragged",
    "decode_text",
    "encode_ragged",
    "encode_text",
    "file_lines",
    "pack_text",
    "parse_columns",
    "parse_flags",
    "parse_integers",
    "parse_numbers",
    "parse_text",
    "take_ragged",
    "take_text",
    "unpack_text",
]

type FieldKind = Literal["text", "ragged", "integer", "flag", "number"]

#: A short text field, stored once per distinct value.
TEXT: FieldKind = "text"
#: Text that nearly every record has its own of, stored record by record.
RAGGED: FieldKind = "ragged"
#: A whole number that is never negative, such as a count or a position.
INTEGER: FieldKind = "integer"
#: A true or false value.
FLAG: FieldKind = "flag"
#: A real number, such as an identity to a reference.
NUMBER: FieldKind = "number"
_KINDS = (TEXT, RAGGED, INTEGER, FLAG, NUMBER)

#: Text that stands for a missing value in the files of single-cell tools:
#: AIRR files write nothing, and Cell Ranger and other tools write ``None`` or
#: ``NA``.
MISSING_TEXT = frozenset({"", "None", "none", "NA", "NaN", "nan", "null", "NULL"})

_TRUE = frozenset({"T", "TRUE", "True", "true", "1"})
_FALSE = frozenset({"F", "FALSE", "False", "false", "0"})


def _checked_fields(fields: Mapping[str, FieldKind]) -> Mapping[str, FieldKind]:
    """Return the kind of each field, refusing kinds this module does not know."""
    if not isinstance(fields, Mapping):
        raise TypeError("fields must map each field name to its kind")
    unknown = {str(name): kind for name, kind in fields.items() if kind not in _KINDS}
    if unknown:
        name, kind = next(iter(unknown.items()))
        raise ValueError(
            f"Field {name!r} has the kind {kind!r}; a field's kind is one of "
            + ", ".join(map(repr, _KINDS))
        )
    return MappingProxyType(dict(fields))


def file_lines(rows: np.ndarray, first_line: int) -> str:
    """Name up to five file lines of chunk rows, such as ``"lines 2, 5"``.

    Args:
        rows: Positions of rows within a chunk.
        first_line: The file line of the chunk's first row.
    """
    lines = [str(int(row) + first_line) for row in np.asarray(rows)[:5]]
    more = len(rows) - len(lines)
    noun = "line" if len(rows) == 1 else "lines"
    return f"{noun} " + ", ".join(lines) + (f" and {more} more" if more > 0 else "")


def _stripped(values: Any) -> tuple[pd.Series, np.ndarray]:
    """Return values as stripped text and a mask of missing-value tokens."""
    raw = np.asarray(values, dtype=object).reshape(-1)
    try:
        stripped = [value.strip() for value in raw]
    except AttributeError:
        # A missing value that is not text, such as None or NaN.
        stripped = [value.strip() if isinstance(value, str) else None for value in raw]
    text = pd.Series(stripped, dtype=object)
    missing = (text.isna() | text.isin(MISSING_TEXT)).to_numpy(dtype=bool)
    return text, missing


def parse_text(values: Any) -> np.ndarray:
    """Return text with every missing-value token as None.

    Args:
        values: One value per record, as read from a file.

    Returns:
        An object array of stripped text, None where a value is missing.
    """
    text, missing = _stripped(values)
    parsed: np.ndarray = text.to_numpy(dtype=object, copy=True)
    parsed[missing] = None
    return parsed


_FLAG_VALUES = {**dict.fromkeys(_TRUE, 1), **dict.fromkeys(_FALSE, 0)}


def parse_flags(values: Any, *, name: str, first_line: int = 2) -> np.ndarray:
    """Parse true or false values as -1 (missing), 0 (False) or 1 (True).

    Args:
        values: One value per record: ``T``, ``F``, ``true``, ``false``,
            ``True``, ``False``, ``TRUE``, ``FALSE``, ``1``, ``0`` or a
            missing-value token.
        name: The field, named in errors.
        first_line: The file line of the first value, named in errors.

    Returns:
        The flags as int8.

    Raises:
        ValueError: If a value is not a recognised true, false or missing token.
    """
    text, missing = _stripped(values)
    mapped = text.map(_FLAG_VALUES)
    unknown = ~missing & mapped.isna().to_numpy(dtype=bool)
    if unknown.any():
        rows = np.flatnonzero(unknown)
        raise ValueError(
            f"{name} must be true or false, but holds {text.iloc[rows[0]]!r} on "
            + file_lines(rows, first_line)
        )
    flags: np.ndarray = mapped.fillna(-1).to_numpy(dtype=np.int8)
    return flags


def _numbers(values: Any) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Return stripped text, its missing mask, and its values as float64."""
    text, missing = _stripped(values)
    numbers = pd.to_numeric(text.where(~missing), errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    return text, missing, numbers


def parse_integers(values: Any, *, name: str, first_line: int = 2) -> np.ndarray:
    """Parse whole numbers of at least 0, with -1 for a missing value.

    Args:
        values: One value per record.
        name: The field, named in errors.
        first_line: The file line of the first value, named in errors.

    Returns:
        The numbers as int64.

    Raises:
        ValueError: If a value is not a whole number of at least 0.
    """
    text, missing, numbers = _numbers(values)
    present = ~missing
    bad = present & (
        ~np.isfinite(numbers) | (numbers < 0) | (numbers != np.floor(numbers))
    )
    if bad.any():
        rows = np.flatnonzero(bad)
        raise ValueError(
            f"{name} must be a whole number of at least 0, but holds "
            f"{text.iloc[rows[0]]!r} on " + file_lines(rows, first_line)
        )
    if present.any() and numbers[present].max() >= 2.0**63:
        raise ValueError(f"{name} holds a number too large for int64")
    parsed = np.full(len(numbers), -1, dtype=np.int64)
    parsed[present] = numbers[present].astype(np.int64)
    return parsed


def parse_numbers(values: Any, *, name: str, first_line: int = 2) -> np.ndarray:
    """Parse finite real numbers, with NaN for a missing value.

    Args:
        values: One value per record.
        name: The field, named in errors.
        first_line: The file line of the first value, named in errors.

    Returns:
        The numbers as float64.

    Raises:
        ValueError: If a value is not a finite number.
    """
    text, missing, numbers = _numbers(values)
    bad = ~missing & ~np.isfinite(numbers)
    if bad.any():
        rows = np.flatnonzero(bad)
        raise ValueError(
            f"{name} must be a finite number, but holds {text.iloc[rows[0]]!r} on "
            + file_lines(rows, first_line)
        )
    numbers[missing] = np.nan
    return numbers


def parse_columns(
    frame: pd.DataFrame, fields: Mapping[str, FieldKind], *, first_line: int
) -> dict[str, np.ndarray]:
    """Parse the columns of a chunk of rows by the kind of each field.

    Args:
        frame: The raw text of one chunk, with a column per field; columns
            outside ``fields`` are not parsed.
        fields: The kind of each field.
        first_line: The file line of the chunk's first row, named in errors.

    Returns:
        Each field that the chunk holds, in the order of ``fields``: text and
        ragged text as Python strings with None where missing, flags as int8,
        integers as int64 and numbers as float64.

    Raises:
        ValueError: If a field's kind is unknown, or a value is of the wrong
            kind.
    """
    parsed: dict[str, np.ndarray] = {}
    for name, kind in _checked_fields(fields).items():
        if name not in frame.columns:
            continue
        raw = frame[name].to_numpy(dtype=object)
        if kind == FLAG:
            parsed[name] = parse_flags(raw, name=name, first_line=first_line)
        elif kind == INTEGER:
            parsed[name] = parse_integers(raw, name=name, first_line=first_line)
        elif kind == NUMBER:
            parsed[name] = parse_numbers(raw, name=name, first_line=first_line)
        else:
            parsed[name] = parse_text(raw)
    return parsed


def pack_text(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pack text as concatenated UTF-8 bytes and offsets.

    Value ``i`` occupies ``data[offsets[i]:offsets[i + 1]]``.

    Args:
        values: Text, none of it missing.

    Returns:
        The bytes as uint8 and ``len(values) + 1`` offsets as int64.
    """
    strings = [str(value) for value in np.asarray(values, dtype=object).reshape(-1)]
    joined = "".join(strings)
    data = joined.encode("utf-8")
    if len(data) == len(joined):
        # ASCII text has one byte per character.
        lengths = np.fromiter(map(len, strings), np.int64, len(strings))
    else:
        lengths = np.fromiter(
            (len(text.encode("utf-8")) for text in strings), np.int64, len(strings)
        )
    offsets = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return np.frombuffer(data, dtype=np.uint8).copy(), offsets


def unpack_text(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return packed text as an object array of strings."""
    buffer = np.asarray(data, dtype=np.uint8).tobytes()
    bounds = np.asarray(offsets, dtype=np.int64)
    starts, ends = bounds[:-1].tolist(), bounds[1:].tolist()
    text = buffer.decode("utf-8")
    values = np.empty(len(starts), dtype=object)
    if len(text) == len(buffer):
        # ASCII text has one byte per character, so the offsets index the text.
        values[:] = [text[start:end] for start, end in zip(starts, ends, strict=True)]
    else:
        values[:] = [
            buffer[start:end].decode("utf-8")
            for start, end in zip(starts, ends, strict=True)
        ]
    return values


@dataclass(frozen=True, slots=True)
class EncodedText:
    """Text stored once per distinct value.

    Encoded records hold their distinct values sorted; packed values that
    others build, such as an export's cell IDs, may come in any order, and
    then ``categories`` holds them in that order.

    Attributes:
        codes: The position of each record's value among the distinct values,
            -1 for a missing value, as int32.
        data: The UTF-8 bytes of the distinct values, concatenated.
        offsets: The offsets of the distinct values in ``data``, one more than
            there are values.
    """

    codes: np.ndarray
    data: np.ndarray = field(repr=False)
    offsets: np.ndarray = field(repr=False)

    @classmethod
    def from_categories(cls, codes: Any, categories: Any) -> "EncodedText":
        """Build encoded text from codes and its distinct values."""
        data, offsets = pack_text(categories)
        return cls(codes=np.asarray(codes, dtype=np.int32), data=data, offsets=offsets)

    @property
    def categories(self) -> np.ndarray:
        """The distinct values, in their stored order, as an object array of text."""
        return unpack_text(self.data, self.offsets)


def encode_text(values: Any) -> EncodedText:
    """Encode text as sorted distinct values and one code per value.

    Args:
        values: Text, with None or another NA for a missing value.

    Returns:
        The codes and the sorted distinct values.
    """
    text = np.asarray(values, dtype=object).reshape(-1)
    present = ~np.asarray(pd.isna(text), dtype=bool)
    codes = np.full(len(text), -1, dtype=np.int32)
    if not present.any():
        return EncodedText.from_categories(codes, [])
    categories, inverse = np.unique(text[present].astype(str), return_inverse=True)
    codes[present] = inverse.astype(np.int32)
    return EncodedText.from_categories(codes, categories)


def decode_text(codes: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Return the text of each code, None for -1."""
    codes = np.asarray(codes)
    values = np.full(codes.shape, None, dtype=object)
    present = codes >= 0
    values[present] = np.asarray(categories, dtype=object)[codes[present]]
    return values


def encode_ragged(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Encode text record by record as concatenated UTF-8 bytes and offsets.

    Value ``i`` occupies ``data[offsets[i]:offsets[i + 1]]``. A missing value
    has an offset of -1 in its slot of ``offsets[1:]`` and no bytes, so it can
    be told from an empty value.

    Args:
        values: Text, with None or another NA for a missing value.

    Returns:
        The bytes as uint8 and ``len(values) + 1`` offsets as int64.
    """
    text = np.asarray(values, dtype=object).reshape(-1)
    missing = np.asarray(pd.isna(text), dtype=bool)
    filled = text.copy()
    filled[missing] = ""
    data, offsets = pack_text(filled)
    ends = offsets[1:].copy()
    ends[missing] = -1
    offsets[1:] = ends
    return data, offsets


def _ragged_bounds(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the start, the end and the missing flag of every ragged value."""
    offsets = np.asarray(offsets, dtype=np.int64)
    missing = offsets[1:] < 0
    ends = np.where(missing, 0, offsets[1:])
    # A missing value's end is -1, so each start is the last end before it.
    filled = np.maximum.accumulate(np.concatenate(([0], ends)))
    return filled[:-1], np.where(missing, filled[:-1], ends), missing


def decode_ragged(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return the text of each ragged value, None where it is missing."""
    starts, ends, missing = _ragged_bounds(offsets)
    # Each value starts where the one before it ends, so the starts and the
    # last end bound every value as packed text does.
    bounds = (
        np.concatenate([starts, ends[-1:]]) if len(starts) else np.zeros(1, np.int64)
    )
    values = unpack_text(data, bounds)
    values[missing] = None
    return values


@njit(cache=True, nogil=True)
def _gather_ragged(
    data: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    new_starts: np.ndarray,
    out: np.ndarray,
) -> None:
    """Copy the bytes of value ``starts[i]:ends[i]`` to ``new_starts[i]`` of ``out``."""
    for index in range(starts.shape[0]):
        target = new_starts[index]
        for position in range(starts[index], ends[index]):
            out[target] = data[position]
            target += 1


def take_ragged(
    data: np.ndarray, offsets: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder or choose ragged values: value ``i`` of the result is ``order[i]``.

    Only the chosen values' bytes are copied, so choosing a few values of a
    long field reads little.

    Returns:
        The bytes and offsets of the chosen values, missing where they are.
    """
    starts, ends, missing = _ragged_bounds(offsets)
    order = np.asarray(order, dtype=np.int64)
    starts, ends, missing = starts[order], ends[order], missing[order]
    new_offsets = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(ends - starts, out=new_offsets[1:])
    new_data = np.empty(int(new_offsets[-1]), dtype=np.uint8)
    _gather_ragged(
        np.asarray(data, dtype=np.uint8), starts, ends, new_offsets[:-1], new_data
    )
    marked = new_offsets[1:].copy()
    marked[missing] = -1
    new_offsets[1:] = marked
    return new_data, new_offsets


def take_text(
    data: np.ndarray, offsets: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Decode some values of packed text, gathering only their bytes.

    Args:
        data: The packed bytes.
        offsets: The offsets of every value in ``data``.
        positions: The values to decode, as positions.

    Returns:
        The text of each position, as an object array.
    """
    gathered, bounds = take_ragged(data, offsets, np.asarray(positions, dtype=np.int64))
    return unpack_text(gathered, bounds)


@dataclass(frozen=True, slots=True)
class RecordTable:
    """Encoded records, ordered by cell, then by source, then by file row.

    Attributes:
        cells: The position of each record's cell among the selected cells, in
            ascending order, as int64.
        sources: The position of each record's source file, as int32.
        kinds: The kind of each field that at least one source provides, in
            the order of the encoder's fields.
        text: The codes and distinct values of each text field.
        ragged: The bytes and offsets of each ragged field.
        values: The int64, int8 or float64 values of each integer, flag and
            number field.
    """

    cells: np.ndarray
    sources: np.ndarray
    kinds: Mapping[str, FieldKind]
    text: Mapping[str, EncodedText] = field(repr=False)
    ragged: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(repr=False)
    values: Mapping[str, np.ndarray] = field(repr=False)

    @property
    def fields(self) -> tuple[str, ...]:
        """The table's fields, in storage order."""
        return tuple(self.kinds)

    @property
    def n_records(self) -> int:
        return len(self.cells)

    def _kind(self, name: str) -> FieldKind:
        kind = self.kinds.get(name)
        if kind is None:
            raise KeyError(f"The table holds no field {name!r}")
        return kind

    def column(self, name: str) -> np.ndarray:
        """Return one field per record, decoded: text, None where missing.

        Raises:
            KeyError: If the table holds no such field.
        """
        kind = self._kind(name)
        if kind == TEXT:
            encoded = self.text[name]
            return decode_text(encoded.codes, encoded.categories)
        if kind == RAGGED:
            return decode_ragged(*self.ragged[name])
        return self.values[name]

    def column_at(self, name: str, rows: np.ndarray) -> np.ndarray:
        """Return one field of chosen records, decoding only those.

        Raises:
            KeyError: If the table holds no such field.
        """
        kind = self._kind(name)
        rows = np.asarray(rows, dtype=np.int64)
        if kind == TEXT:
            encoded = self.text[name]
            return decode_text(np.asarray(encoded.codes)[rows], encoded.categories)
        if kind == RAGGED:
            return decode_ragged(*take_ragged(*self.ragged[name], rows))
        values: np.ndarray = np.asarray(self.values[name])[rows]
        return values


@dataclass(slots=True)
class _Chunk:
    """The encoded fields of one chunk, only those that the chunk holds."""

    cells: np.ndarray
    source: int
    text: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ragged: dict[str, tuple[np.ndarray, np.ndarray]]
    values: dict[str, np.ndarray]
    id_hashes: np.ndarray | None

    @property
    def n_rows(self) -> int:
        return len(self.cells)


class RecordEncoder:
    """Encode records chunk by chunk, holding each distinct text value once.

    Each chunk is a table of parsed records whose columns are field names:
    text as Python strings with None for a missing value, flags as int8,
    integers as int64 and numbers as float64, as :func:`parse_columns` returns
    them. Columns outside ``fields`` are ignored, and a field that a chunk
    lacks is missing for its records and takes no memory. Each chunk's text is
    factorized on its own and its distinct values packed as bytes; ``finish``
    merges the distinct values of all chunks, one field at a time, and sorts
    them.

    Args:
        fields: The kind of each field to keep, in storage order.
        id_field: A ragged field whose values identify each record within its
            source, such as a contig ID, or None. The encoder then counts the
            records without an ID and finds the IDs that repeat within a
            source.

    Raises:
        TypeError: If ``fields`` is not a mapping.
        ValueError: If a field's kind is unknown, or ``id_field`` is not a
            ragged field of ``fields``.
    """

    def __init__(
        self, fields: Mapping[str, FieldKind], *, id_field: str | None = None
    ) -> None:
        self._kinds = _checked_fields(fields)
        if id_field is not None and self._kinds.get(id_field) != RAGGED:
            raise ValueError(
                f"id_field {id_field!r} must name a ragged field of fields, whose "
                "values identify each record"
            )
        self._id_field = id_field
        self._chunks: list[_Chunk] = []
        self._fields: set[str] = set()
        self._nbytes = 0
        self._missing_ids = 0
        self._finished = False

    @property
    def n_records(self) -> int:
        return sum(chunk.n_rows for chunk in self._chunks)

    @property
    def missing_ids(self) -> int:
        """The number of records without an ID.

        Raises:
            ValueError: If the encoder has no ``id_field``.
        """
        self._require_ids()
        return self._missing_ids

    @property
    def nbytes(self) -> int:
        """The bytes of the encoded arrays that the encoder holds."""
        return self._nbytes

    def _require_ids(self) -> None:
        if self._id_field is None:
            raise ValueError("The encoder has no id_field, so its records have no IDs")

    def add(self, chunk: pd.DataFrame, cells: np.ndarray, source: int) -> None:
        """Add a chunk of parsed records of one source.

        Args:
            chunk: Parsed records, one row per record.
            cells: The position of each row's cell among the selected cells.
            source: The position of the chunk's source file.

        Raises:
            ValueError: If ``cells`` does not hold one cell per row.
            RuntimeError: If the encoder has finished.
        """
        if self._finished:
            raise RuntimeError("The encoder has finished; start a new one")
        n_rows = len(chunk)
        cell_positions = np.asarray(cells, dtype=np.int64).reshape(-1)
        if len(cell_positions) != n_rows:
            raise ValueError("A chunk needs one cell per record")
        encoded = _Chunk(
            cells=cell_positions,
            source=int(source),
            text={},
            ragged={},
            values={},
            id_hashes=None,
        )
        for name in map(str, chunk.columns):
            kind = self._kinds.get(name)
            if kind is None:
                continue
            self._fields.add(name)
            if kind == TEXT:
                codes, distinct = pd.factorize(
                    chunk[name].to_numpy(dtype=object), use_na_sentinel=True
                )
                data, offsets = pack_text(distinct)
                encoded.text[name] = (np.asarray(codes, dtype=np.int32), data, offsets)
            elif kind == RAGGED:
                values = chunk[name].to_numpy(dtype=object)
                encoded.ragged[name] = encode_ragged(values)
                if name == self._id_field:
                    missing = np.asarray(pd.isna(values), dtype=bool)
                    self._missing_ids += int(missing.sum())
                    encoded.id_hashes = np.asarray(
                        pd.util.hash_array(values.astype(str)), dtype=np.uint64
                    )
            else:
                encoded.values[name] = _typed_values(kind, chunk[name], n_rows)
        if self._id_field is not None and encoded.id_hashes is None:
            # A chunk without the ID field holds no record with an ID.
            self._missing_ids += n_rows
        self._nbytes += (
            cell_positions.nbytes
            + sum(array.nbytes for parts in encoded.text.values() for array in parts)
            + sum(array.nbytes for parts in encoded.ragged.values() for array in parts)
            + sum(array.nbytes for array in encoded.values.values())
            + (0 if encoded.id_hashes is None else encoded.id_hashes.nbytes)
        )
        self._chunks.append(encoded)

    def repeated_ids(self, limit: int = 5) -> list[tuple[int, str]]:
        """Return up to ``limit`` record IDs that repeat within one source.

        IDs are compared by a 64-bit hash, and the records whose hashes repeat
        are compared by their text, so a hash collision is never reported.
        Check ``missing_ids`` first: missing IDs compare equal. Call before
        ``finish``.

        Returns:
            The source and the ID of each repeat found, by source and ID.

        Raises:
            ValueError: If the encoder has no ``id_field``.
        """
        self._require_ids()
        chunks = [chunk for chunk in self._chunks if chunk.id_hashes is not None]
        if not chunks:
            return []
        hashes = np.concatenate([chunk.id_hashes for chunk in chunks])
        sources = np.concatenate(
            [np.full(chunk.n_rows, chunk.source, dtype=np.int64) for chunk in chunks]
        )
        order = np.lexsort((hashes, sources))
        same = (hashes[order][1:] == hashes[order][:-1]) & (
            sources[order][1:] == sources[order][:-1]
        )
        if not same.any():
            return []
        candidates = np.unique(np.concatenate([order[1:][same], order[:-1][same]]))
        starts = np.cumsum([0] + [chunk.n_rows for chunk in chunks])
        found: set[tuple[int, str]] = set()
        seen: set[tuple[int, str]] = set()
        id_field = str(self._id_field)
        for row in candidates.tolist():
            position = int(np.searchsorted(starts, row, side="right")) - 1
            chunk = chunks[position]
            data, offsets = chunk.ragged[id_field]
            local = np.asarray([row - int(starts[position])])
            text = str(decode_ragged(*take_ragged(data, offsets, local))[0])
            key = (chunk.source, text)
            if key in seen:
                found.add(key)
            seen.add(key)
        return sorted(found)[:limit]

    def finish(self) -> RecordTable:
        """Return the records ordered by cell, then source, then file row.

        The chunks are consumed field by field, so the encoder holds no more
        than the table it returns and one field in transit.

        Raises:
            RuntimeError: If the encoder has finished already.
        """
        if self._finished:
            raise RuntimeError("The encoder has finished; start a new one")
        self._finished = True
        chunks = self._chunks
        lengths = [chunk.n_rows for chunk in chunks]
        cells = np.concatenate(
            [np.empty(0, dtype=np.int64), *(chunk.cells for chunk in chunks)]
        )
        sources = np.concatenate(
            [
                np.empty(0, dtype=np.int32),
                *(
                    np.full(chunk.n_rows, chunk.source, dtype=np.int32)
                    for chunk in chunks
                ),
            ]
        )
        # A stable sort keeps the file rows of each cell and source in order.
        order = np.lexsort((sources, cells))
        # Each row's position in the result, so that the values of every
        # chunk are placed in it as the chunk is released.
        destination = np.empty(len(order), dtype=np.int64)
        destination[order] = np.arange(len(order))
        bounds = np.cumsum([0, *lengths])
        text: dict[str, EncodedText] = {}
        ragged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        values: dict[str, np.ndarray] = {}
        kinds = {
            name: kind for name, kind in self._kinds.items() if name in self._fields
        }
        for name, kind in kinds.items():
            if kind == TEXT:
                parts = [chunk.text.pop(name, None) for chunk in chunks]
                text[name] = _merge_text(parts, lengths, order)
            elif kind == RAGGED:
                ragged_pieces = [
                    chunk.ragged.pop(name)
                    if name in chunk.ragged
                    else _missing_ragged(chunk.n_rows)
                    for chunk in chunks
                ]
                ragged[name] = take_ragged(*_concatenate_ragged(ragged_pieces), order)
            else:
                placed = _typed_values(kind, None, len(order))
                for position, chunk in enumerate(chunks):
                    if name in chunk.values:
                        rows = destination[bounds[position] : bounds[position + 1]]
                        placed[rows] = chunk.values.pop(name)
                values[name] = placed
        self._chunks = []
        self._nbytes = 0
        return RecordTable(
            cells=cells[order],
            sources=sources[order],
            kinds=MappingProxyType(kinds),
            text=MappingProxyType(text),
            ragged=MappingProxyType(ragged),
            values=MappingProxyType(values),
        )


def _missing_ragged(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n_rows`` missing ragged values."""
    offsets = np.full(n_rows + 1, -1, dtype=np.int64)
    offsets[0] = 0
    return np.empty(0, dtype=np.uint8), offsets


def _merge_text(
    parts: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    lengths: Sequence[int],
    order: np.ndarray,
) -> EncodedText:
    """Merge the factorized text of every chunk into sorted distinct values.

    Args:
        parts: Each chunk's codes into its own distinct values, -1 where
            missing, with those values packed; None for a chunk without the
            field.
        lengths: The rows of each chunk.
        order: The final order of the records, as positions in the chunks'
            concatenation.
    """
    every = np.concatenate(
        [np.empty(0, dtype=object)]
        + [unpack_text(part[1], part[2]) for part in parts if part is not None]
    )
    if len(every):
        merged, categories = pd.factorize(every)
        categories = np.asarray(categories, dtype=object)
        # Object text sorts by code point, as fixed-width text would, without
        # a copy padded to the longest value.
        ranks = np.argsort(categories, kind="stable")
        final = np.empty(len(categories), dtype=np.int32)
        final[ranks] = np.arange(len(categories), dtype=np.int32)
        global_codes = final[merged]
        data, offsets = pack_text(categories[ranks])
    else:
        global_codes = np.empty(0, dtype=np.int32)
        data, offsets = pack_text([])
    del every
    pieces = []
    start = 0
    for part, length in zip(parts, lengths, strict=True):
        piece = np.full(length, -1, dtype=np.int32)
        if part is not None:
            codes = part[0]
            present = codes >= 0
            piece[present] = global_codes[start + codes[present]]
            start += len(part[2]) - 1
        pieces.append(piece)
    codes = np.concatenate([np.empty(0, dtype=np.int32), *pieces])
    return EncodedText(codes=codes[order], data=data, offsets=offsets)


def _typed_values(kind: str, column: Any, n_rows: int) -> np.ndarray:
    """Return a chunk's integers, flags or numbers, missing where absent."""
    if kind == INTEGER:
        if column is None:
            return np.full(n_rows, -1, dtype=np.int64)
        return np.asarray(column, dtype=np.int64)
    if kind == FLAG:
        if column is None:
            return np.full(n_rows, -1, dtype=np.int8)
        return np.asarray(column, dtype=np.int8)
    if column is None:
        return np.full(n_rows, np.nan, dtype=np.float64)
    return np.asarray(column, dtype=np.float64)


def _concatenate_ragged(
    chunks: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Join the ragged values of one or more chunks into one ragged array."""
    datas: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    base = 0
    for data, offsets in chunks:
        chunk_ends = np.asarray(offsets[1:], dtype=np.int64)
        datas.append(np.asarray(data, dtype=np.uint8))
        ends.append(np.where(chunk_ends < 0, -1, chunk_ends + base))
        base += len(data)
    return np.concatenate(datas), np.concatenate([np.zeros(1, np.int64), *ends])


def chunk_rows_for(budget_bytes: int, *, columns: int) -> int:
    """Return how many file rows to parse at once within a memory budget.

    Parsing holds about 1,000 bytes per row and column, across the raw text,
    its stripped copy and the parsed values, and a chunk may take a twentieth
    of the budget, within 10,000 and 50,000 rows: smaller chunks would repeat
    the distinct values that each chunk holds.
    """
    rows = int(budget_bytes) // (20 * 1_000 * max(1, int(columns)))
    return int(min(50_000, max(10_000, rows)))
