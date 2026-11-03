"""Read delimited text tables that single-cell tools write, in bounded chunks.

The clone tables of lineage tracing tools, the barcode tables of their
extractors, the allele tables of lineage recorders, receptor contigs and the
tables of variant callers share their quirks: plain or gzip-compressed text,
whatever the file's name, separated by tabs, commas or spaces, quoted as R
quotes, without a final newline, a header that starts with ``#`` as TREX
writes it, and R's row-name layout, whose header holds one field fewer than
its rows; R's ``write.csv`` writes the row names under an empty first name
instead. A delimiter after each row's last field also gives one field more
than the header, an empty one, so when every row ends in its delimiter with
nothing after it the table is refused, since the two layouts read alike
there, unless its reader prefers R's row names because its own checks refuse
the other layout, as a character matrix's do.

Text is UTF-8, and a byte-order mark at its start is no part of the header
(:mod:`cytearc.utils.files`). Blank lines, which hold only whitespace and no
delimiter, hold no rows; a line of delimiters is a row of empty fields.
Every row holds as many fields as its header names, a row name included, and
a row of fewer or more is refused, naming its line, wherever the chunks of
rows end. Errors name each row by its line in the file, past blank lines and
quoted fields that hold line breaks.
"""

import csv
import gzip
import itertools
import re
import zlib
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from ..utils.arguments import quoted_value

from ..utils.files import open_text

__all__ = [
    "TextLayout",
    "count_rows",
    "delimiter",
    "first_lines",
    "joined_columns",
    "line_chunks",
    "parsed_chunks",
    "require_columns",
    "rows_end_in_delimiter",
    "split_fields",
    "text_chunks",
    "text_errors",
    "text_layout",
]

# Rows after the header that the layout samples for their fields and width.
_SAMPLE_ROWS = 100
# What reading a gzip file that is cut short or corrupt raises.
_CORRUPT = (EOFError, zlib.error, gzip.BadGzipFile)
# The characters of text that the rows hand pandas at least at a time.
_READ_CHARACTERS = 1 << 16


@dataclass(frozen=True, slots=True)
class TextLayout:
    """A delimited text file's columns and delimiter.

    Attributes:
        path: The file.
        names: The name of each column, ``""`` for R's row names.
        delimiter: ``"\\t"``, ``","`` or ``" "``.
        row_names: Whether the first field of each row is an R row name.
        header: Whether the first line is a header; a table without one
            names its columns by their position.
        row_chars: The characters of the longest of the first rows.
        quoted: Whether a field may be quoted, as R quotes text; without
            quoting, as in AIRR files, a quote is part of its value.
    """

    path: str
    names: tuple[str, ...]
    delimiter: str
    row_names: bool
    header: bool = True
    row_chars: int = 0
    quoted: bool = True


def _corrupt_error(path: str, error: BaseException) -> ValueError:
    return ValueError(f"{path} is cut short or corrupt: {error}")


def _text_error(path: str) -> ValueError:
    return ValueError(f"{path} is not a text file that CyteArc can read")


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
        raise _text_error(path) from None
    except _CORRUPT as error:
        raise _corrupt_error(path, error) from None


@contextmanager
def _parse_errors(path: str) -> Iterator[None]:
    """Name the file in the errors of parsing it with pandas."""
    try:
        yield
    except _CORRUPT as error:
        raise _corrupt_error(path, error) from None
    except (pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"{path} is malformed: {error}") from None


def first_lines(path: str, n_lines: int) -> list[str]:
    """Return up to ``n_lines`` first lines of a text file, without line ends.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    lines: list[str] = []
    with text_errors(path), open_text(path) as handle:
        for line in handle:
            lines.append(line.rstrip("\r\n"))
            if len(lines) == n_lines:
                break
    return lines


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


def _dialect(delimiter: str) -> dict[str, Any]:
    """The options of Python's csv module that read fields as pandas reads them."""
    if delimiter == " ":
        return {"delimiter": " ", "skipinitialspace": True}
    return {"delimiter": delimiter}


def _for_csv(line: str, delimiter: str) -> str:
    """A line as the csv module splits it, spaces read as pandas reads them."""
    if delimiter == " ":
        return line.strip(" \t\r\n").replace("\t", " ")
    return line


def split_fields(line: str, delimiter: str, *, quoted: bool = True) -> list[str]:
    """Split one row into its fields, as the table's reader splits it.

    Spaces separate fields of a space-separated row however many there are,
    and spaces at its ends separate none.
    """
    if not quoted or '"' not in line:
        return line.split() if delimiter == " " else line.split(delimiter)
    return next(csv.reader([_for_csv(line, delimiter)], **_dialect(delimiter)), [])


def _is_blank(line: str, delimiter: str) -> bool:
    """Whether a line holds no row: only whitespace, and no delimiter."""
    return not line.strip() and (delimiter == " " or delimiter not in line)


# The states of a row's text as the quote machine reads it, as pandas'
# tokenizer reads quoted fields: at the start of a field, in an unquoted
# field, in a quoted field, and just after a quote within a quoted field,
# which closes the field unless another quote follows it.
_FIELD_START, _UNQUOTED, _QUOTED, _QUOTE = 0, 1, 2, 3
# What the quote machine stops at: quotes, and each delimiter or, between
# space-separated fields, each run of spaces and tabs.
_SPECIAL = {
    "\t": re.compile('["\t]'),
    ",": re.compile('[",]'),
    " ": re.compile('"|[ \t]+'),
}


def _scan(text: str, state: int, delimiter: str) -> tuple[int, int]:
    """Read one line of a row's text, without its line end, from ``state``.

    Returns:
        The state at the line's end, and the fields that the line ends: one
        per delimiter outside quotes. A line that ends in ``_QUOTED`` goes on
        with the next line, within the quoted field.
    """
    ended = 0
    field_start = 0
    previous = -2
    for match in _SPECIAL[delimiter].finditer(text):
        at = match.start()
        quote = text[at] == '"'
        if state == _FIELD_START and at > field_start:
            state = _UNQUOTED
        elif state == _QUOTE and at > previous + 1:
            # Text after a closing quote joins the field, as pandas reads it.
            state = _UNQUOTED
        if state == _QUOTED:
            if quote:
                state = _QUOTE
        elif state == _QUOTE:
            if quote:
                state = _QUOTED
            else:
                ended += 1
                state = _FIELD_START
                field_start = match.end()
        elif state == _FIELD_START:
            if quote:
                state = _QUOTED
            else:
                ended += 1
                field_start = match.end()
        elif not quote:
            ended += 1
            state = _FIELD_START
            field_start = match.end()
        previous = match.end() - 1
    return state, ended


def _quoted_record(
    first: str, rest: Iterator[str], delimiter: str
) -> tuple[list[str], int, bool]:
    """Read a row whose text holds a quote, taking lines while a quoted field goes on.

    Returns:
        The row's lines as read, its fields, and whether the file ended within
        a quoted field.
    """
    lines = [first]
    state, ended = _scan(_for_scan(first, delimiter), _FIELD_START, delimiter)
    while state == _QUOTED:
        line = next(rest, None)
        if line is None:
            return lines, ended + 1, True
        lines.append(line)
        state, more = _scan(_for_scan(line, delimiter), state, delimiter)
        ended += more
    return lines, ended + 1, False


def _for_scan(line: str, delimiter: str) -> str:
    """A line without its line end, and a space-separated one without spaces at its ends."""
    if delimiter == " ":
        return line.strip(" \t\r\n")
    return line.rstrip("\r\n")


def _never_closes(path: str, line: int) -> ValueError:
    return ValueError(
        f"{path} is malformed: the quote opened on line {line} is never closed"
    )


def _records(
    handle: Any, path: str, delimiter: str, quoted: bool, first_line: int
) -> Iterator[tuple[int, str]]:
    """Yield the rows of a table's text: each one's first line and text.

    Blank lines are skipped, and the lines of a quoted field that holds a
    line break stay in one row. Fields are not counted.

    Raises:
        ValueError: If a quote never closes.
    """
    lines = iter(handle)
    line = first_line
    for physical in lines:
        start = line
        line += 1
        if quoted and '"' in physical:
            taken, _fields, cut = _quoted_record(physical, lines, delimiter)
            if cut:
                raise _never_closes(path, start)
            line += len(taken) - 1
            yield start, "".join(taken).rstrip("\r\n")
        elif not _is_blank(physical, delimiter):
            yield start, physical.rstrip("\r\n")


class _Rows:
    """A table's rows as text for pandas to parse, each row's fields counted.

    pandas reads the text through ``read``, which hands it whole rows only:
    blank lines are left out, and the lines of a row whose quoted field holds
    a line break stay together. The file line of each row handed on is kept
    until ``take`` hands it out with the row's chunk. The first row of
    another number of fields than the layout names, a quote that never
    closes, or text that cannot be read ends the text, and ``error`` says
    why, so the rows before it are handed on whatever chunks pandas reads.
    """

    def __init__(self, handle: Any, text: TextLayout, first_line: int) -> None:
        self._handle = handle
        self._text = text
        self._n_fields = len(text.names)
        # The last line of the file read so far.
        self._line = first_line - 1
        self._lines = array("q")
        self.error: ValueError | None = None
        self._ended = False

    def read(self, size: int = -1) -> str:
        # Text without rows would end pandas' reading, so lines are read
        # until some hold a row or the file ends.
        while not self._ended:
            try:
                text = self._read(max(int(size), _READ_CHARACTERS))
            except UnicodeDecodeError:
                self._end(_text_error(self._text.path))
            except _CORRUPT as error:
                self._end(_corrupt_error(self._text.path, error))
            else:
                if text:
                    return text
        return ""

    def take(self, n_rows: int) -> np.ndarray:
        """Return the file lines of the next ``n_rows`` rows handed on."""
        lines = np.asarray(self._lines[:n_rows], dtype=np.int64).copy()
        del self._lines[:n_rows]
        return lines

    def count(self) -> int:
        """Hand on the remaining rows, counting them rather than keeping their lines."""
        n_rows = 0
        while self.read(1 << 20):
            n_rows += len(self._lines)
            del self._lines[:]
        return n_rows

    def _end(self, error: ValueError) -> None:
        self.error = error
        self._ended = True

    def _refuse(self, line: int, n_fields: int) -> None:
        text = self._text
        n_names = len(text.names)
        if not text.header:
            expected = f"its rows hold {n_names}"
        elif text.row_names:
            expected = (
                f"its rows hold {n_names}: a row name and the {n_names - 1} columns "
                "that its header names"
            )
        else:
            expected = f"its header names {n_names} columns"
        self._end(
            ValueError(
                f"{text.path} is malformed: line {line} holds {n_fields} "
                f"field{'' if n_fields == 1 else 's'}, but {expected}"
            )
        )

    def _read(self, size: int) -> str:
        batch = self._handle.readlines(size)
        if not batch:
            self._ended = True
            return ""
        first = self._line + 1
        self._line += len(batch)
        joined = "".join(batch)
        delimiter = self._text.delimiter
        if self._text.quoted and '"' in joined:
            return self._read_quoted(batch, first)
        if delimiter == " ":
            widths = np.fromiter(
                map(len, map(str.split, batch)), dtype=np.int64, count=len(batch)
            )
            blank = widths == 0
        else:
            widths = 1 + np.fromiter(
                map(str.count, batch, itertools.repeat(delimiter)),
                dtype=np.int64,
                count=len(batch),
            )
            blank = (widths == 1) & np.fromiter(
                map(str.isspace, batch), dtype=bool, count=len(batch)
            )
        wrong = np.flatnonzero((widths != self._n_fields) & ~blank)
        if len(wrong):
            at = int(wrong[0])
            self._refuse(first + at, int(widths[at]))
            batch, blank = batch[:at], blank[:at]
        kept = np.flatnonzero(~blank)
        self._lines.frombytes((first + kept).astype(np.int64).tobytes())
        if len(kept) == len(batch) and not len(wrong):
            return joined
        return "".join(batch[position] for position in kept.tolist())

    def _read_quoted(self, batch: list[str], first: int) -> str:
        """Hand on the rows of a batch whose lines hold quotes, row by row.

        A row whose quoted field goes on past the batch takes lines from the
        handle until the field closes.
        """
        delimiter = self._text.delimiter
        extra = 0

        def rest() -> Iterator[str]:
            nonlocal extra
            yield from lines
            while line := self._handle.readline():
                extra += 1
                yield line

        lines = iter(batch)
        following = rest()
        parts: list[str] = []
        line = first
        for physical in lines:
            start = line
            if '"' not in physical:
                line += 1
                if _is_blank(physical, delimiter):
                    continue
                n_fields = len(split_fields(physical, delimiter, quoted=False))
                taken = [physical]
            else:
                taken, n_fields, cut = _quoted_record(physical, following, delimiter)
                line += len(taken)
                if cut:
                    self._end(_never_closes(self._text.path, start))
                    break
            if n_fields != self._n_fields:
                self._refuse(start, n_fields)
                break
            self._lines.append(start)
            parts.append("".join(taken))
        self._line += extra
        return "".join(parts)


def text_layout(
    path: str,
    *,
    what: str = "a clone table",
    prefer_row_names: bool = False,
    quoted: bool = True,
) -> TextLayout:
    """Read a table's header and delimiter from its first line and rows.

    A header that starts with ``#`` is a header whose first name loses the
    ``#``. The first row is the first line after the header that holds text.
    A header with one field fewer than that row is R's row-name layout, whose
    first column is named ``""``; a delimiter after each row's last field
    gives one field more than the header too, an empty one, so a table whose
    every row ends in its delimiter with nothing after it fits both layouts
    and is refused, unless ``prefer_row_names`` reads its row names. A header
    whose first name is empty, as R's ``write.csv`` writes it, holds row names
    too.

    Args:
        path: The table.
        what: What the table is, such as ``"a clone table"``, for the error
            about an empty file.
        prefer_row_names: Read rows of one field more than the header as
            R's row names even where they all end in a delimiter with nothing
            after it, without reading the rest of the file to tell, rather
            than refuse them. Only for a table whose reader refuses the other
            layout by its own checks, as a character matrix refuses the
            cells' column read among PEtracer's characters, or an empty last
            state where its format writes none, which a delimiter after each
            row's last field gives.
        quoted: Whether a field may be quoted; without quoting, a quote is
            part of its value.

    Raises:
        ValueError: If the file is empty, is not UTF-8 text, a gzip file is
            cut short or corrupt, a quote never closes, its first row holds
            another number of fields than its header names, or, without
            ``prefer_row_names``, its rows of one field more than the header
            all end in a delimiter with nothing after it.
    """
    first, rows = _header_and_rows(path, quoted)
    if not first.strip():
        raise ValueError(f"{path} is empty; {what} needs a header line")
    separator = rows.delimiter
    header = [name.strip() for name in split_fields(first, separator, quoted=quoted)]
    # TREX starts its headers with "#", which marks no comment.
    if header and header[0].startswith("#"):
        header[0] = header[0][1:]
    row_names = False
    if rows.texts:
        fields = split_fields(rows.texts[0], separator, quoted=quoted)
        if len(fields) == len(header) + 1:
            # R's write.table leaves its row names out of the header, and a
            # delimiter after each row's last field gives one field more too,
            # an empty one. The two differ only where a row holds something
            # after its last named column.
            if (
                not prefer_row_names
                and _ends_in_delimiter(rows.texts[0], separator)
                and not _rows_hold_values_after_their_names(
                    path, separator, quoted, len(header)
                )
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
        row_names=row_names,
        row_chars=max((len(row) for row in rows.texts), default=0),
        quoted=quoted,
    )


@dataclass(frozen=True, slots=True)
class _Sample:
    """A table's delimiter and its first rows' text."""

    delimiter: str
    texts: list[str]


def _header_and_rows(path: str, quoted: bool) -> tuple[str, _Sample]:
    """Return a table's first line, its delimiter, and up to ``_SAMPLE_ROWS`` rows."""
    with text_errors(path), open_text(path) as handle:
        first = handle.readline().rstrip("\r\n")
        row = next((line for line in handle if not _is_blank(line, " ")), "").rstrip(
            "\r\n"
        )
    separator = delimiter([first, row])
    with text_errors(path), open_text(path) as handle:
        handle.readline()
        texts = [
            text
            for _line, text in itertools.islice(
                _records(handle, path, separator, quoted, 2), _SAMPLE_ROWS
            )
        ]
    return first, _Sample(delimiter=separator, texts=texts)


def _ends_in_delimiter(row: str, delimiter: str) -> bool:
    """Return whether a row ends in its delimiter, with nothing but spaces after it.

    Space-separated rows lose the spaces at their ends when they are split,
    so none ends in its delimiter.
    """
    if delimiter == " ":
        return False
    return row.rstrip(" \t".replace(delimiter, "")).endswith(delimiter)


def _rows_hold_values_after_their_names(
    path: str, delimiter: str, quoted: bool, n_names: int
) -> bool:
    """Return whether a row of a table holds anything after its named columns.

    Each row is read until one holds a field after the header's ``n_names``
    columns that is not empty: a value, or quoted empty text as R writes it.
    """
    with text_errors(path), open_text(path) as handle:
        handle.readline()
        for _line, row in _records(handle, path, delimiter, quoted, 2):
            if len(
                split_fields(row, delimiter, quoted=quoted)
            ) == n_names + 1 and not _ends_in_delimiter(row, delimiter):
                return True
    return False


def rows_end_in_delimiter(text: TextLayout) -> bool:
    """Return whether a table's R row names come from rows that all end in a delimiter.

    Such rows hold one field more than the header and nothing after their
    last delimiter, which a delimiter after each row's last field writes
    alike, so ``text_layout`` reads them as row names only for a reader that
    prefers them. The file is read again, so a reader asks only to explain
    an error.

    Raises:
        ValueError: If the file is not UTF-8 text, a gzip file is cut short or
            corrupt, or a quote never closes.
    """
    if not text.row_names or not text.header:
        return False
    first, rows = _header_and_rows(text.path, text.quoted)
    n_names = len(split_fields(first, text.delimiter, quoted=text.quoted))
    return (
        bool(rows.texts)
        and n_names == len(text.names) - 1
        and _ends_in_delimiter(rows.texts[0], text.delimiter)
        and not _rows_hold_values_after_their_names(
            text.path, text.delimiter, text.quoted, n_names
        )
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
        shown = ", ".join(quoted_value(name) for name in layout.names if name)
        raise ValueError(
            f"{layout.path} has no column "
            + ", ".join(quoted_value(name) for name in missing)
            + f"; its columns are {shown}"
        )
    repeated = sorted({name for name in names if counts[name] > 1})
    if repeated:
        raise ValueError(
            f"{layout.path} names the column "
            + ", ".join(quoted_value(name) for name in repeated)
            + " more than once, so which one to read is unclear"
        )


def line_chunks(
    text: TextLayout, chunk_rows: int
) -> Iterator[tuple[np.ndarray, list[str]]]:
    """Yield chunks of a table's lines that hold rows, its header aside.

    Each chunk comes with the file line of each of its lines, counting from
    1. Lines are not joined across quoted line breaks, and their fields are
    not counted, so a reader of a table whose rows may hold more fields than
    its header, as DNA Typewriter's of 2022, checks them itself.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    numbers: list[int] = []
    lines: list[str] = []
    with text_errors(text.path), open_text(text.path) as handle:
        number = 0
        if text.header:
            handle.readline()
            number = 1
        for line in handle:
            number += 1
            if _is_blank(line, text.delimiter):
                continue
            numbers.append(number)
            lines.append(line)
            if len(lines) == chunk_rows:
                yield np.asarray(numbers, dtype=np.int64), lines
                numbers, lines = [], []
    if lines:
        yield np.asarray(numbers, dtype=np.int64), lines


def count_rows(text: TextLayout) -> int:
    """Count a table's rows, its header aside, checking each one's fields.

    Raises:
        ValueError: If the file is not UTF-8 text, a gzip file is cut short
            or corrupt, a quote never closes, or a row holds another number
            of fields than the header names.
    """
    with text_errors(text.path), open_text(text.path) as handle:
        if text.header:
            handle.readline()
        rows = _Rows(handle, text, 2 if text.header else 1)
        n_rows = rows.count()
    if rows.error is not None:
        raise rows.error
    return n_rows


def joined_columns(
    frame: pd.DataFrame, positions: Sequence[int], separator: str
) -> np.ndarray:
    """Return the stripped text of columns, joined by ``separator`` row by row.

    A row whose text in any of the columns is empty joins to ``""``, so a
    cell that lacks a part of its barcode has none.
    """
    parts = [frame[position].str.strip() for position in positions]
    joined = parts[0]
    lacking = (parts[0] == "").to_numpy(dtype=bool, copy=True)
    for part in parts[1:]:
        joined = joined + separator + part
        lacking |= (part == "").to_numpy(dtype=bool)
    values: np.ndarray = joined.to_numpy(dtype=object, copy=True)
    values[lacking] = ""
    return values


def text_chunks(
    text: TextLayout, chunk_rows: int, *, columns: Sequence[int] | None = None
) -> Iterator[tuple[np.ndarray, pd.DataFrame]]:
    """Yield chunks of a table's rows as text, with each row's file line.

    Columns are numbered by position, and every row holds one field per
    column: a row of fewer or more fields is refused, naming its line, once
    the rows before it are yielded, so a chunk of any size meets it at the
    same row. Blank lines hold no rows, and lines count from 1, the header's.

    Args:
        text: The table's layout.
        chunk_rows: The most rows of a chunk.
        columns: The positions of the columns that the chunks hold, or None
            for every column; every field is counted either way.

    Yields:
        The file line of each row of the chunk, as int64, and the chunk.

    Raises:
        ValueError: If the file is not UTF-8 text, a gzip file is cut short
            or corrupt, a quote never closes, or a row holds another number
            of fields than the header names.
    """
    with text_errors(text.path), open_text(text.path) as handle:
        if text.header:
            handle.readline()
        rows = _Rows(handle, text, 2 if text.header else 1)
        options: dict[str, Any] = {
            "sep": r"\s+" if text.delimiter == " " else text.delimiter,
            "header": None,
            "names": list(range(len(text.names))),
            "index_col": False,
            "dtype": str,
            "keep_default_na": False,
            "na_filter": False,
            "skip_blank_lines": False,
            "chunksize": max(1, int(chunk_rows)),
        }
        if columns is not None:
            options["usecols"] = list(columns)
        if text.quoted:
            options["quotechar"] = '"'
        else:
            options["quoting"] = csv.QUOTE_NONE
        for chunk in parsed_chunks(text.path, rows, **options):
            if len(chunk):
                yield rows.take(len(chunk)), chunk
    if rows.error is not None:
        raise rows.error


def parsed_chunks(path: str, source: Any, **options: Any) -> Iterator[pd.DataFrame]:
    """Yield the chunks that pandas parses from a file or stream, naming the file.

    The options name every column, so text without rows parses to an empty
    chunk rather than raising.

    Raises:
        ValueError: If pandas cannot parse the file, or a gzip file is cut
            short or corrupt.
    """
    with _parse_errors(path):
        reader = pd.read_csv(source, **options)
    with reader:
        while True:
            with _parse_errors(path):
                chunk = next(reader, None)
            if chunk is None:
                return
            yield chunk
