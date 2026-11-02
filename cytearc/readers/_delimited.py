"""Read delimited text tables that single-cell tools write, in bounded chunks.

The clone tables of lineage tracing tools and the barcode tables of their
extractors share their quirks: plain or gzip-compressed text, whatever the
file's name, separated by tabs, commas or spaces, quoted as R quotes, without
a final newline, a header that starts with ``#`` as TREX writes it, and R's
row-name layout, whose header holds one field fewer than its rows; R's
``write.csv`` writes the row names under an empty first name instead. A
delimiter after each row's last field also gives one field more than the
header, an empty one, so when every row ends in its delimiter with nothing
after it the table is refused, since the two layouts read alike there.
Blank lines hold no rows. These functions find a table's layout from its
first lines and yield its rows as text, chunk by chunk, naming the file in
their errors.
"""

import csv
import gzip
import io
import itertools
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "GZIP_SIGNATURE",
    "HDF5_SIGNATURE",
    "TextLayout",
    "compression",
    "count_rows",
    "delimiter",
    "first_lines",
    "joined_columns",
    "line_chunks",
    "line_list",
    "open_text",
    "parsed_chunks",
    "require_columns",
    "signature",
    "split_fields",
    "text_chunks",
    "text_errors",
    "text_layout",
]

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
GZIP_SIGNATURE = b"\x1f\x8b"
# Rows after the header that the layout samples for their fields and width.
_SAMPLE_ROWS = 100
# What reading a gzip file that is cut short or corrupt raises.
_CORRUPT = (EOFError, zlib.error, gzip.BadGzipFile)


@dataclass(frozen=True, slots=True)
class TextLayout:
    """A delimited text file's columns, delimiter and compression.

    Attributes:
        path: The file.
        names: The name of each column, ``""`` for R's row names.
        delimiter: ``"\\t"``, ``","`` or ``" "``.
        compression: ``"gzip"`` or None.
        row_names: Whether the first field of each row is an R row name.
        header: Whether the first line is a header; a table without one
            names its columns by their position.
        row_chars: The characters of the longest of the first rows.
    """

    path: str
    names: tuple[str, ...]
    delimiter: str
    compression: str | None
    row_names: bool
    header: bool = True
    row_chars: int = 0


def signature(path: str) -> bytes:
    """Return the first eight bytes of a file."""
    with open(path, "rb") as handle:
        return handle.read(8)


def compression(path: str) -> str | None:
    """Return ``"gzip"`` for a gzip-compressed file, by its first bytes, else None."""
    return "gzip" if signature(path).startswith(GZIP_SIGNATURE) else None


def open_text(path: str, compression: str | None) -> io.TextIOBase:
    """Open a plain or gzip-compressed text file as UTF-8."""
    if compression == "gzip":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _corrupt_error(path: str, error: BaseException) -> ValueError:
    return ValueError(f"{path} is cut short or corrupt: {error}")


@contextmanager
def text_errors(path: str) -> Iterator[None]:
    """Name the file in the errors of reading it as UTF-8 text, plain or gzip.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    try:
        yield
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not a text file that CyteArc can read") from None
    except _CORRUPT as error:
        raise _corrupt_error(path, error) from None


@contextmanager
def _parse_errors(path: str) -> Iterator[None]:
    """Name the file in the errors of parsing it with pandas, a header alone aside."""
    try:
        yield
    except pd.errors.EmptyDataError:
        raise
    except _CORRUPT as error:
        raise _corrupt_error(path, error) from None
    except (pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"{path} is malformed: {error}") from None


def first_lines(path: str, compression: str | None, n_lines: int) -> list[str]:
    """Return up to ``n_lines`` first lines of a text file, without line ends.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    lines: list[str] = []
    with text_errors(path), open_text(path, compression) as handle:
        for line in handle:
            lines.append(line.rstrip("\r\n"))
            if len(lines) == n_lines:
                break
    return lines


def _records(lines: Iterable[str], path: str) -> Iterator[str]:
    """Yield the rows of a table's text after its header, each without its line ending.

    Blank lines are skipped, and a row whose quotes are open goes on to the
    next line; each line's quotes are counted once.

    Raises:
        ValueError: If a quote is still open where the file ends.
    """
    parts: list[str] = []
    # The line on which the quote that is still open opened, 0 for none.
    opened = 0
    for number, line in enumerate(lines, start=2):
        text = line.rstrip("\r\n")
        if not parts and not text.strip():
            continue
        parts.append(text)
        if text.count('"') % 2:
            opened = 0 if opened else number
        if not opened:
            yield "\n".join(parts)
            parts = []
    if opened:
        raise ValueError(
            f"{path} is malformed: the quote opened on line {opened} is never closed"
        )


def _header_and_rows(path: str, compression: str | None) -> tuple[str, list[str]]:
    """Return a table's first line and up to ``_SAMPLE_ROWS`` rows after it.

    Blank lines after the first are skipped.
    """
    with text_errors(path), open_text(path, compression) as handle:
        header = next(handle, "").rstrip("\r\n")
        rows = list(itertools.islice(_records(handle, path), _SAMPLE_ROWS))
    return header, rows


def _ends_in_delimiter(row: str, delimiter: str) -> bool:
    """Return whether a row ends in its delimiter, with nothing but spaces after it.

    Space-separated rows lose the spaces at their ends when they are split,
    so none ends in its delimiter.
    """
    if delimiter == " ":
        return False
    return row.rstrip(" \t".replace(delimiter, "")).endswith(delimiter)


def _rows_hold_values_after_their_names(
    path: str, compression: str | None, delimiter: str, n_names: int
) -> bool:
    """Return whether a row of a table holds anything after its named columns.

    Each row is read until one holds a field after the header's ``n_names``
    columns that is not empty: a value, or quoted empty text as R writes it.
    """
    with text_errors(path), open_text(path, compression) as handle:
        next(handle, "")
        for row in _records(handle, path):
            if len(split_fields(row, delimiter)) == n_names + 1 and not (
                _ends_in_delimiter(row, delimiter)
            ):
                return True
    return False


def delimiter(lines: list[str]) -> str:
    """Return the delimiter of a table: tab, comma or space.

    The header decides; a header of one name leaves it to the first row, so
    a quoted barcode that holds a comma or a space stays one field.
    """
    header = lines[0]
    for candidate in ("\t", ","):
        if candidate in header:
            return candidate
    if " " in header.strip():
        return " "
    row = lines[1] if len(lines) > 1 else ""
    for candidate in ("\t", ","):
        if candidate in row:
            return candidate
    return " "


def split_fields(line: str, delimiter: str) -> list[str]:
    """Split one line into its fields, as the table's reader splits it."""
    if delimiter == " ":
        reader = csv.reader([line.strip()], delimiter=" ", skipinitialspace=True)
    else:
        reader = csv.reader([line], delimiter=delimiter)
    return next(reader, [])


def text_layout(path: str, *, what: str = "a clone table") -> TextLayout:
    """Read a table's header and delimiter from its first line and rows.

    A header that starts with ``#`` is a header whose first name loses the
    ``#``. The first row is the first line after the header that holds text.
    A header with one field fewer than that row is R's row-name layout, whose
    first column is named ``""``; a delimiter after each row's last field
    gives one field more than the header too, an empty one, so a table whose
    every row ends in its delimiter with nothing after it fits both layouts
    and is refused. A header whose first name is empty, as R's ``write.csv``
    writes it, holds row names too.

    Args:
        path: The table.
        what: What the table is, such as ``"a clone table"``, for the error
            about an empty file.

    Raises:
        ValueError: If the file is empty, is not UTF-8 text, a gzip file is
            cut short or corrupt, a quote never closes, its first row holds
            another number of fields than its header names, or its rows of one
            field more than the header all end in a delimiter with nothing
            after it.
    """
    found = compression(path)
    first, rows = _header_and_rows(path, found)
    if not first.strip():
        raise ValueError(f"{path} is empty; {what} needs a header line")
    separator = delimiter([first, *rows[:1]])
    header = [name.strip() for name in split_fields(first.lstrip("﻿"), separator)]
    # TREX starts its headers with "#", which marks no comment.
    if header and header[0].startswith("#"):
        header[0] = header[0][1:]
    row_names = False
    if rows:
        fields = split_fields(rows[0], separator)
        if len(fields) == len(header) + 1:
            # R's write.table leaves its row names out of the header, and a
            # delimiter after each row's last field gives one field more too,
            # an empty one. The two differ only where a row holds something
            # after its last named column.
            if _ends_in_delimiter(
                rows[0], separator
            ) and not _rows_hold_values_after_their_names(
                path, found, separator, len(header)
            ):
                raise ValueError(
                    f"The header of {path} names {len(header)} columns, and its "
                    f"rows of {len(header) + 1} fields all end in a delimiter with "
                    "nothing after it: R's row names under a header that leaves "
                    "them out, with an empty last column, or a delimiter after "
                    "each row's last field. Name the row names in the header, as "
                    "R's write.table(..., col.names=NA) writes them, or end the "
                    "header with the delimiter too"
                )
            row_names = True
            header = ["", *header]
        elif len(fields) != len(header):
            raise ValueError(
                f"The header of {path} names {len(header)} columns, but its first "
                f"row holds {len(fields)} fields"
            )
        elif header[0] == "":
            # R's write.csv writes its row names under an empty name.
            row_names = True
    return TextLayout(
        path=path,
        names=tuple(header),
        delimiter=separator,
        compression=found,
        row_names=row_names,
        row_chars=max((len(row) for row in rows), default=0),
    )


def require_columns(layout: TextLayout, names: Sequence[str]) -> None:
    """Refuse a table that lacks a column to read, or names it twice.

    Raises:
        ValueError: If a column is missing or named more than once.
    """
    # Counted once, since a merged table names a column per barcode.
    counts = Counter(layout.names)
    missing = [name for name in names if name not in counts]
    if missing:
        shown = ", ".join(repr(name) for name in layout.names if name)
        raise ValueError(
            f"{layout.path} has no column "
            + ", ".join(repr(name) for name in missing)
            + f"; its columns are {shown}"
        )
    repeated = sorted({name for name in names if counts[name] > 1})
    if repeated:
        raise ValueError(
            f"{layout.path} names the column "
            + ", ".join(repr(name) for name in repeated)
            + " more than once, so which one to read is unclear"
        )


def line_chunks(
    text: TextLayout, chunk_rows: int
) -> Iterator[tuple[np.ndarray, list[str]]]:
    """Yield chunks of a table's lines that hold text, its header aside.

    Each chunk comes with the file line of each of its lines, counting from
    1, so that errors name the lines even when blank lines sit between rows.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    numbers: list[int] = []
    lines: list[str] = []
    with text_errors(text.path), open_text(text.path, text.compression) as handle:
        number = 0
        if text.header:
            handle.readline()
            number = 1
        for line in handle:
            number += 1
            if not line.strip():
                continue
            numbers.append(number)
            lines.append(line)
            if len(lines) == chunk_rows:
                yield np.asarray(numbers, dtype=np.int64), lines
                numbers, lines = [], []
    if lines:
        yield np.asarray(numbers, dtype=np.int64), lines


def count_rows(text: TextLayout) -> int:
    """Count a table's lines that hold text, its header aside.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    count = 0
    with text_errors(text.path), open_text(text.path, text.compression) as handle:
        if text.header:
            handle.readline()
        for line in handle:
            if line.strip():
                count += 1
    return count


def joined_columns(
    frame: pd.DataFrame, positions: Sequence[int], separator: str
) -> np.ndarray:
    """Return the stripped text of columns, joined by ``separator`` row by row."""
    parts = [frame[position].str.strip() for position in positions]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + separator + part
    return np.asarray(joined.to_numpy(dtype=object))


def line_list(rows: np.ndarray, first_line: int, text: TextLayout | None = None) -> str:
    """Name up to five lines of rows of a chunk that starts at ``first_line``.

    ``first_line`` counts rows, as ``text_chunks`` yields it; with ``text``,
    the rows' lines in the file are found by reading it again, past blank
    lines and quoted fields that hold line breaks.
    """
    lines = [int(row) + first_line for row in rows[:5]]
    if text is not None:
        lines = _file_lines(text, lines)
    more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
    return ", ".join(str(line) for line in lines) + more


def _blank_record(fields: list[str], delimiter: str) -> bool:
    """Whether a record is a line that pandas skips as blank.

    A line of whitespace alone is blank, but a tab-separated line of tabs
    holds empty fields.
    """
    if not all(not field.strip() for field in fields):
        return False
    return delimiter == " " or len(fields) <= 1


def _file_lines(text: TextLayout, row_lines: list[int]) -> list[int]:
    """Return the file lines of rows that ``text_chunks`` numbers.

    ``text_chunks`` numbers the first row after the header line 2, or line 1
    in a table without a header, and counts rows from there. Only errors need
    the file lines, so the file is read again from its start.
    """
    first = 2 if text.header else 1
    wanted = {line - first for line in row_lines}
    found: dict[int, int] = {}
    with open_text(text.path, text.compression) as handle:
        if text.header:
            handle.readline()
        if text.delimiter == " ":
            reader = csv.reader(handle, delimiter=" ", skipinitialspace=True)
        else:
            reader = csv.reader(handle, delimiter=text.delimiter)
        row, consumed = -1, 0
        for fields in reader:
            start = consumed + first
            consumed = reader.line_num
            if _blank_record(fields, text.delimiter):
                continue
            row += 1
            if row in wanted:
                found[row] = start
                if len(found) == len(wanted):
                    break
    # A row the second reading misses keeps the line that counts rows.
    return [found.get(line - first, line) for line in row_lines]


def text_chunks(
    text: TextLayout, chunk_rows: int
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield chunks of a table's rows as text, with each chunk's first line.

    Columns are numbered by position. Every field is read, so a row with
    more fields than the first raises. Blank lines hold no rows. The lines
    count rows, the first after the header being line 2, or line 1 in a table
    without a header, so a row's line is its chunk's first line plus its
    position; ``line_list`` with ``text`` finds the line in the file, past
    blank lines and quoted fields that hold line breaks.

    Raises:
        ValueError: If the file is malformed.
    """
    separator = r"\s+" if text.delimiter == " " else text.delimiter
    chunks = parsed_chunks(
        text.path,
        sep=separator,
        header=None,
        skiprows=1 if text.header else 0,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        compression=text.compression,
        quotechar='"',
        chunksize=chunk_rows,
        skip_blank_lines=True,
        encoding="utf-8",
    )
    line = 2 if text.header else 1
    for chunk in chunks:
        yield line, chunk
        line += len(chunk)


def parsed_chunks(path: str, **options: Any) -> Iterator[pd.DataFrame]:
    """Yield the chunks that pandas parses, naming the file in its errors.

    Raises:
        ValueError: If pandas cannot parse the file, or a gzip file is cut
            short or corrupt.
    """
    try:
        with _parse_errors(path):
            reader = pd.read_csv(path, **options)
    except pd.errors.EmptyDataError:
        # A header without rows holds no rows.
        return
    with reader:
        while True:
            with _parse_errors(path):
                chunk = next(reader, None)
            if chunk is None:
                return
            yield chunk
