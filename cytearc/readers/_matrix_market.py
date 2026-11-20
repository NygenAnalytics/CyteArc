"""Read Matrix Market coordinate matrices, plain or gzip-compressed, in chunks.

Count matrices of 10x Genomics, clone matrices such as LARRY's and the allele
count matrices of variant callers are Matrix Market coordinate files: a
banner, comment lines that start with ``%``, a dimensions line and one entry
per line. ``read_matrix_market_header`` reads the banner and the dimensions,
and ``matrix_market_entries`` streams the entries in chunks, checking each
coordinate against the dimensions and the number of entries against the one
declared. A file's first bytes, not its name, decide whether it is
gzip-compressed (:mod:`cytearc.utils.files`).
"""

import gzip
import os
import re
import sys
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np
import pandas as pd

from ..utils.arguments import quoted_value
from ..utils.files import GZIP_SIGNATURE, open_text

__all__ = [
    "MatrixMarketHeader",
    "matrix_market_entries",
    "matrix_market_entry_object_bytes",
    "matrix_market_header",
    "matrix_market_parse_bytes",
    "read_matrix_market_header",
]

# The buffer through which a parse reads its file.
_FILE_BUFFER_BYTES = 128 * 1024
# What pandas' C reader holds from its first chunk on, whatever the chunk
# size: its buffers, the text of the 256 KiB that it reads at a time, decoded,
# and the buffer through which the parse reads its file. Parsing plain and
# gzip-compressed files of short and wide lines 10 entries at a time,
# tracemalloc traced at most 1.33 MB with pandas 2.2.3, the oldest supported
# release, on Python 3.13, and 1.40 MB with pandas 3.0.6 on Python 3.13 and
# 3.14, the same with a file system that reports a block of 4 MiB. The charge
# leaves room for the zlib builds and platforms that differ from the one
# measured.
_READER_BYTES = 1_500_000
# pandas before 3.0 keeps three floats of 24 bytes for each chunk it reads,
# until the parse ends, in a list of 8-byte slots that grows by up to an
# eighth more.
_CHUNK_BYTES = 3 * (24 + 9)


@dataclass(frozen=True, slots=True)
class MatrixMarketHeader:
    """The banner and dimensions line of a Matrix Market coordinate matrix.

    Attributes:
        path: The file, as errors name it.
        field: ``"integer"``, ``"real"`` or ``"pattern"``.
        n_rows: The matrix's rows.
        n_columns: The matrix's columns.
        n_entries: The entries that the file declares.
        header_lines: The lines before the first entry: the banner, comment
            and blank lines, and the dimensions line.
    """

    path: str
    field: str
    n_rows: int
    n_columns: int
    n_entries: int
    header_lines: int


# What reading a gzip file that is cut short or corrupt raises.
_CORRUPT = (EOFError, zlib.error, gzip.BadGzipFile)
# pandas' refusal of a line of more fields than the names it reads.
_MORE_FIELDS = re.compile(r"Expected \d+ fields in line (\d+), saw (\d+)")


def read_matrix_market_header(
    handle: TextIO, path: str, *, fields: tuple[str, ...] = ("integer", "real")
) -> MatrixMarketHeader:
    """Read the banner and dimensions line of a Matrix Market coordinate matrix.

    Args:
        handle: The file's text, open at its start.
        path: The file, as errors name it.
        fields: The kinds of values the matrix may hold: ``"integer"``,
            ``"real"`` and ``"pattern"``, which holds none.

    Raises:
        ValueError: If the file is not a general coordinate matrix of one of
            ``fields``, or its dimensions line is not three whole numbers of
            at least 0.
    """
    first = handle.readline().strip()
    banner = first.split()
    if len(banner) != 5 or [part.lower() for part in banner[:3]] != [
        "%%matrixmarket",
        "matrix",
        "coordinate",
    ]:
        raise ValueError(
            "Matrix Market input must use coordinate matrix format, but "
            f"{path} begins {quoted_value(first)}"
        )
    field = banner[3].lower()
    symmetry = banner[4].lower()
    if field not in fields:
        allowed = ", ".join(fields[:-1]) + " or " + fields[-1]
        raise ValueError(
            f"Matrix Market input must use {allowed} coordinate values, but "
            f"{path} holds {quoted_value(field)} values"
        )
    if symmetry != "general":
        raise ValueError(
            "Matrix Market input must use general symmetry, but "
            f"{path} is {quoted_value(symmetry)}"
        )
    header_lines = 1
    dimensions = None
    for line in handle:
        header_lines += 1
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            dimensions = stripped
            break
    if dimensions is None:
        raise ValueError(
            "The Matrix Market dimensions line must contain three integers, but "
            f"{path} ends before it"
        )
    sizes = dimensions.split()
    try:
        if len(sizes) != 3:
            raise ValueError
        n_rows, n_columns, n_entries = (int(value) for value in sizes)
    except ValueError:
        raise ValueError(
            "The Matrix Market dimensions line must contain three integers, but "
            f"{path} has {quoted_value(dimensions)}"
        ) from None
    if n_rows < 0 or n_columns < 0 or n_entries < 0:
        raise ValueError(
            f"Matrix Market dimensions cannot be negative, but {path} declares "
            f"{quoted_value(dimensions)}"
        )
    return MatrixMarketHeader(
        path=path,
        field=field,
        n_rows=n_rows,
        n_columns=n_columns,
        n_entries=n_entries,
        header_lines=header_lines,
    )


def matrix_market_header(
    path: str | os.PathLike[str],
    *,
    fields: tuple[str, ...] = ("integer", "real"),
    name: str | None = None,
) -> MatrixMarketHeader:
    """Read the banner and dimensions line of a Matrix Market file, plain or gzip.

    Args:
        path: The file.
        fields: The kinds of values the matrix may hold, as
            :func:`read_matrix_market_header` takes them.
        name: The file as the header and errors name it, such as a ZIP
            archive's member that ``path`` holds extracted; by default
            ``path``.

    Raises:
        ValueError: As :func:`read_matrix_market_header` raises, or if the
            file is not UTF-8 text or a gzip file is cut short or corrupt.
    """
    file = os.fspath(path)
    if name is None:
        name = file
    try:
        with open_text(file) as handle:
            return read_matrix_market_header(handle, name, fields=fields)
    except UnicodeDecodeError:
        raise ValueError(f"{name} is not a text file that CyteArc can read") from None
    except _CORRUPT as error:
        raise ValueError(f"{name} is cut short or corrupt: {error}") from None


def _entry_width(field: str) -> int:
    """The number of fields of each entry of a matrix of ``field`` values."""
    return 2 if field == "pattern" else 3


def matrix_market_entry_object_bytes(header: MatrixMarketHeader) -> int:
    """Return the bytes of the Python objects that pandas 2.2 makes of each
    entry while :func:`matrix_market_entries` parses a chunk.

    Given a dtype for each column, pandas 2.2 builds each chunk through an
    object array of the chunk's columns, filled in one assignment, so numpy
    first turns every column into an array of Python objects: per field, an
    8-byte reference to an int of at most an int64's size, or to a float.
    Later versions hold less while they build a chunk, so these bytes are the
    worst case of the supported versions.

    Args:
        header: The matrix's header, from :func:`matrix_market_header`.
    """
    # The parser reads one column past the entry's fields.
    fields = _entry_width(header.field) + 1
    largest = sys.getsizeof(int(np.iinfo(np.int64).max))
    return fields * (np.dtype(object).itemsize + largest)


def matrix_market_parse_bytes(header: MatrixMarketHeader, chunk_rows: int) -> int:
    """Return what parsing a Matrix Market file ``chunk_rows`` entries at a time
    holds, as a count reader parses it.

    pandas' C reader holds its buffers and the text that it reads at a time
    from the first chunk on, whatever the chunk size, and pandas before 3.0
    keeps three floats for each chunk that it has read until the parse ends. A
    chunk holds at most ``chunk_rows`` entries and never more than the file
    declares: for each, an int64 row and column and a uint64 value in four
    copies, as the parse and a reader that sums duplicate coordinates make
    them, a mask, and the Python objects of
    :func:`matrix_market_entry_object_bytes`. pandas 3.0 keeps no floats and
    makes no such objects, so these bytes are the worst case of the supported
    versions.

    Args:
        header: The matrix's header, from :func:`matrix_market_header`.
        chunk_rows: The most entries of a chunk, at least 1.
    """
    value_bytes = 2 * np.dtype(np.int64).itemsize + np.dtype(np.uint64).itemsize
    entry_bytes = (
        4 * value_bytes
        + np.dtype(np.bool_).itemsize
        + matrix_market_entry_object_bytes(header)
    )
    chunks = -(-header.n_entries // int(chunk_rows))
    entries = min(int(chunk_rows), header.n_entries)
    return _READER_BYTES + _CHUNK_BYTES * chunks + entries * entry_bytes


def _entry_fields(field: str) -> str:
    """The fields of each entry of a matrix of ``field`` values."""
    if field == "pattern":
        return (
            "Each entry of a Matrix Market pattern matrix holds 2 fields, its row "
            "and column"
        )
    return (
        f"Each entry of a Matrix Market {field} matrix holds 3 fields, its row, "
        "column and value"
    )


@contextmanager
def _entry_errors(path: str, field: str) -> Iterator[None]:
    """Name the file in the errors of parsing its entries."""
    try:
        yield
    except _CORRUPT as error:
        raise ValueError(f"{path} is cut short or corrupt: {error}") from None
    except (TypeError, ValueError) as error:
        more = _MORE_FIELDS.search(str(error))
        if more is not None:
            raise ValueError(
                f"{_entry_fields(field)}, but line {more[1]} of {path} holds {more[2]}"
            ) from None
        raise ValueError(
            f"Could not parse Matrix Market coordinates of {path}: {error}"
        ) from None


def matrix_market_entries(
    header: MatrixMarketHeader,
    chunk_rows: int,
    *,
    source: str | os.PathLike[str] | None = None,
    separator: str = " ",
    index_offset: int = -1,
    counts: bool = False,
    remedy: str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Yield the entries of a Matrix Market coordinate matrix in chunks.

    Each chunk holds at least one entry. Lines that start with ``%`` hold no
    entry. Coordinates are parsed as
    whole numbers and values as the matrix's field declares: int64 for
    ``"integer"`` and float64 for ``"real"``, so scipy's ``3.71E2`` in a real
    matrix reads as 371. An entry holds its row and column, and a value
    unless the matrix is a pattern; an entry of more fields is refused, as
    its fields would otherwise be read shifted or cut.

    Args:
        header: The matrix's header, from :func:`matrix_market_header`.
        chunk_rows: The most entries of a chunk.
        source: The file to read, where ``header.path`` only names it, as a
            ZIP archive's member extracted to a temporary file; by default
            ``header.path``.
        separator: The delimiter of an entry's fields; a space stands for any
            whitespace.
        index_offset: Added to each coordinate; the default turns Matrix
            Market's coordinates, which count from 1, into positions from 0.
        counts: Whether every value must be a count: a finite whole number of
            at least 0.
        remedy: Appended to the refusal of a value that is not a count.

    Yields:
        Each entry's row and column, ``index_offset`` added, as int64, and its
        value, or None for a pattern matrix.

    Raises:
        ValueError: If an entry holds more fields than its matrix's field
            gives it, cannot be parsed or lies outside the matrix, a count is
            not a finite whole number of at least 0, the file holds another
            number of entries than it declares, or a gzip file is cut short or
            corrupt.
    """
    path = header.path
    file = path if source is None else os.fspath(source)
    pattern = header.field == "pattern"
    width = _entry_width(header.field)
    dtype: dict[int, Any] = {0: np.int64, 1: np.int64}
    if not pattern:
        dtype[2] = np.int64 if header.field == "integer" else np.float64
    # A column past the entry's fields holds a field of a wider entry: pandas
    # would read the first field of a wider first entry as an index, and
    # drop the last field of one that starts a chunk.
    dtype[width] = np.float32
    seen = 0
    # pandas would open the file with open()'s default buffer, which follows
    # the block size that the file system reports, without a limit on Python
    # 3.13 and up to 8 MiB on 3.14, so a file system that reports a block of
    # megabytes would add megabytes to the parse; the file is read through
    # 128 KiB instead.
    with open(file, "rb", buffering=_FILE_BUFFER_BYTES) as handle:
        # The first bytes, read into that buffer, say whether the file is
        # gzip-compressed, which no second opening of the file has to find.
        gzipped = handle.read(len(GZIP_SIGNATURE)) == GZIP_SIGNATURE
        handle.seek(0)
        with _entry_errors(path, header.field):
            # The names of the columns make a matrix without entries, which
            # ends at its dimensions line, parse to an empty chunk.
            chunks = iter(
                pd.read_csv(
                    handle,
                    comment="%",
                    sep=r"\s+" if separator == " " else separator,
                    header=None,
                    skiprows=header.header_lines,
                    names=list(range(width + 1)),
                    index_col=False,
                    dtype=dtype,
                    chunksize=max(1, int(chunk_rows)),
                    compression="gzip" if gzipped else None,
                )
            )
        while True:
            with _entry_errors(path, header.field):
                frame = next(chunks, None)
            if frame is None:
                break
            if not len(frame):
                continue
            wider = ~np.isnan(frame[width].to_numpy(copy=False))
            if wider.any():
                raise ValueError(
                    f"{_entry_fields(header.field)}, but entry "
                    f"{seen + int(np.argmax(wider)) + 1} of {path} holds more"
                )
            seen += len(frame)
            rows = frame[0].to_numpy(dtype=np.int64, copy=False) + index_offset
            columns = frame[1].to_numpy(dtype=np.int64, copy=False) + index_offset
            if (
                np.any(rows < 0)
                or np.any(rows >= header.n_rows)
                or np.any(columns < 0)
                or np.any(columns >= header.n_columns)
            ):
                raise ValueError(
                    f"A Matrix Market coordinate of {path} is outside the declared "
                    f"dimensions, {header.n_rows} by {header.n_columns}"
                )
            values = None if pattern else frame[2].to_numpy(copy=False)
            if counts and values is not None:
                whole = (
                    np.isfinite(values) & (values >= 0) & (values == np.floor(values))
                )
                if not whole.all():
                    raise ValueError(
                        "Matrix Market counts must be finite non-negative integers, "
                        f"but {path} holds {quoted_value(values[~whole][0].item())}"
                        + ("" if remedy is None else f". {remedy}")
                    )
            yield rows, columns, values
    if seen != header.n_entries:
        raise ValueError(
            f"The Matrix Market header declares {header.n_entries} entries, but "
            f"{seen} were read from {path}"
        )
