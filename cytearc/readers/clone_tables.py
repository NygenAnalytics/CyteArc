"""Read clone assignments from the files that lineage tracing tools write.

Each format lists, for each cell, the clone it belongs to. ``clone_layout``
recognises a file's format from its contents and resolves the columns that
hold each cell's barcode, its clone and, where the format gives one, the
probability of that clone. ``iter_clone_rows`` then yields the rows of a
table or an AnnData file in bounded chunks, and the ``matrix_*`` functions
read a clone matrix with its cell list. Clone labels are text as the file
writes them, and a cell without a clone has the label None. Joining barcodes
to cells and checking the clones belong to the caller.

Formats:

- ``"cospar"``: CoSpar's clone table, ``Cell_ID,Clone_ID``.
- ``"trex"``: TREX's ``clones.txt``: ``clone_nr`` and ``cell_id`` separated
  by tabs, or ``#clone_id,cell_id`` in the 2022 releases.
- ``"celltag_multi"``: CellTag-multi's clone tables: ``clone_id`` or
  ``clone.id`` with ``cell.bc``, after an unnamed index column in the iEP
  table on Zenodo.
- ``"celltagr"``: CellTagR's clone composition, ``clone.id`` and
  ``cell.barcode``.
- ``"sticr"``: STICR's cell metadata, ``cell_Id`` and ``VBC``.
- ``"cassiopeia"``: Cassiopeia's allele table, ``cellBC`` and ``lineageGrp``,
  one row per cell and target site, where lineage group 0 means no clone.
- ``"numbat"``: Numbat's ``clone_post_{i}.tsv``: ``cell``, ``clone_opt`` and
  its posterior ``p_opt``.
- ``"table"``: any delimited table whose columns ``columns`` names, one row
  per cell or several, or one clone column per labelling round, such as the
  Biddy et al. workflow's ``celltag_data.txt``, which takes cells from its R
  row names.
- ``"matrix"``: a Matrix Market clone matrix with a cell list beside it:
  LARRY's ``*_clone_matrix.mtx.gz`` with ``*_metadata.txt.gz`` on GEO, or
  ``cells.tsv`` and ``clones.tsv`` as ``ds.clones.export_clones`` writes.
- ``"cospar_h5ad"``: an AnnData file with CoSpar's ``obsm['X_clone']``.
- ``"mito"``: an AnnData file with MiTo's ``obs['MiTo_clone']``, where
  ``"unassigned"`` means no clone.
- ``"h5ad"``: an AnnData file whose ``obs`` column ``columns`` names.

Text may be plain or gzip-compressed, separated by tabs, commas or spaces,
with R's double quotes and without a final newline. A header that starts
with ``#``, as TREX writes, is a header. A header with one field fewer than
its rows is R's row-name layout, whose first field is the row name; R's
``write.csv`` writes the row names under an empty first name instead. A
delimiter after each row's last field also gives one field more than the
header, an empty one, so when every row ends in its delimiter with nothing
after it the table is refused, since the two layouts read alike there.
Blank lines hold no rows.
"""

import csv
import gzip
import io
import itertools
import os
import re
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..clones.assignments import clone_label_texts
from ._h5ad_columns import (
    column_length,
    index_key,
    is_nullable,
    read_column,
    read_table_column,
    sparse_encoding,
    sparse_shape,
)
from ._text import as_text

__all__ = [
    "CELL_ID_SEPARATOR",
    "CLONE_FORMATS",
    "CloneLayout",
    "CloneRows",
    "MISSING_TEXT",
    "MatrixEntries",
    "clone_layout",
    "iter_clone_rows",
    "matrix_cells",
    "matrix_clone_ids",
    "matrix_entries",
]

#: The values of ``format``.
CLONE_FORMATS: tuple[str, ...] = (
    "cospar",
    "trex",
    "celltag_multi",
    "celltagr",
    "sticr",
    "cassiopeia",
    "numbat",
    "table",
    "matrix",
    "cospar_h5ad",
    "mito",
    "h5ad",
)
#: Joins the columns that together hold a cell's barcode, such as LARRY's
#: ``Library`` and ``Cell barcode``, as a merged store joins sample and cell.
CELL_ID_SEPARATOR = "__"

_TEXT_FORMATS = frozenset(
    {
        "cospar",
        "trex",
        "celltag_multi",
        "celltagr",
        "sticr",
        "cassiopeia",
        "numbat",
        "table",
    }
)
_H5AD_FORMATS = frozenset({"cospar_h5ad", "mito", "h5ad"})
_COLUMN_ROLES = frozenset({"cell", "clone", "probability"})
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_GZIP_SIGNATURE = b"\x1f\x8b"
#: Text that R and pandas write for a missing value, which a table's clone
#: column reads as no clone.
MISSING_TEXT = frozenset({"", "NA", "NaN", "nan"})
_MITO_UNASSIGNED = "unassigned"
# Rows after the header that the layout samples for their fields and width.
_SAMPLE_ROWS = 100
# What reading a gzip file that is cut short or corrupt raises.
_CORRUPT = (EOFError, zlib.error, gzip.BadGzipFile)


@dataclass(frozen=True, slots=True)
class _Preset:
    """The columns of a tool's clone table."""

    cell: str
    clones: tuple[str, ...]
    probability: str | None = None


# Each tool's cell column and the names its clone column has had.
_PRESETS: dict[str, _Preset] = {
    "cospar": _Preset("Cell_ID", ("Clone_ID",)),
    # TREX wrote "#clone_id,cell_id" in 2022, "clone#" briefly in 2023, and
    # "clone_nr" since; the leading "#" of a header is dropped.
    "trex": _Preset("cell_id", ("clone_nr", "clone#", "clone_id")),
    "celltag_multi": _Preset("cell.bc", ("clone_id", "clone.id")),
    "celltagr": _Preset("cell.barcode", ("clone.id",)),
    "sticr": _Preset("cell_Id", ("VBC",)),
    "cassiopeia": _Preset("cellBC", ("lineageGrp",)),
    "numbat": _Preset("cell", ("clone_opt",), probability="p_opt"),
}


@dataclass(frozen=True, slots=True)
class CloneLayout:
    """How to read the clones of one file.

    Attributes:
        format: One of ``CLONE_FORMATS``.
        path: The file.
        cell: The columns whose values, joined by ``CELL_ID_SEPARATOR``,
            are each cell's barcode; empty for R row names or an AnnData
            file's cell names.
        clone: The column of clone labels; None for a matrix.
        probability: The column of clone probabilities, or None.
        cells_path: A matrix's cell list.
        clones_path: A matrix's clone list, or None to label clones by their
            position from 0.
    """

    format: str
    path: str
    cell: tuple[str, ...] = ()
    clone: str | None = None
    probability: str | None = None
    cells_path: str | None = None
    clones_path: str | None = None
    _text: "_TextLayout | None" = field(default=None, repr=False)
    _cells_text: "_TextLayout | None" = field(default=None, repr=False)
    _matrix: "_MatrixHeader | None" = field(default=None, repr=False)

    @property
    def is_matrix(self) -> bool:
        """Whether the clones come as a cells by clones matrix."""
        return self.format in ("matrix", "cospar_h5ad")

    @property
    def n_fields(self) -> int:
        """The fields of each row that a chunk holds, which sizes chunks."""
        text = self._text if self._text is not None else self._cells_text
        return len(text.names) if text is not None else 3

    @property
    def row_chars(self) -> int:
        """The characters of the longest row the layout sampled, 0 for none."""
        text = self._text if self._text is not None else self._cells_text
        return text.row_chars if text is not None else 0

    @property
    def related_paths(self) -> tuple[str, ...]:
        """The files read beside ``path``: a matrix's cell and clone lists."""
        return tuple(path for path in (self.cells_path, self.clones_path) if path)

    def record(self) -> dict[str, Any]:
        """Return the format and columns that identify how the file is read."""
        return {
            "format": self.format,
            "cell": list(self.cell),
            "clone": self.clone,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class _TextLayout:
    """A delimited text file's header, delimiter and compression.

    Attributes:
        row_names: Whether the first field of each row names its cell, under
            an empty name or one that the header leaves out, as R writes it.
        row_chars: The characters of the longest of the first rows.
    """

    path: str
    names: tuple[str, ...]
    delimiter: str
    compression: str | None
    row_names: bool
    row_chars: int = 0


@dataclass(frozen=True, slots=True)
class _MatrixHeader:
    """A Matrix Market clone matrix's shape, cell axis and header lines."""

    n_rows: int
    n_columns: int
    n_entries: int
    pattern: bool
    header_lines: int
    compression: str | None
    cell_axis: int = 0

    @property
    def n_clones(self) -> int:
        return self.n_columns if self.cell_axis == 0 else self.n_rows


@dataclass(frozen=True, slots=True)
class CloneRows:
    """A chunk of rows of a clone table, in file order.

    Attributes:
        first_line: The line of the first row, counting rows: the header is
            line 1 of a table and blank lines hold no row, and an AnnData
            file's first cell is line 1.
        barcodes: Each row's cell barcode, as text.
        clones: Each row's clone label, None for none.
        probabilities: Each row's clone probability, NaN for none, or None
            when the format gives none.
    """

    first_line: int
    barcodes: np.ndarray
    clones: np.ndarray
    probabilities: np.ndarray | None


@dataclass(frozen=True, slots=True)
class MatrixEntries:
    """A chunk of the nonzero entries of a clone matrix.

    Attributes:
        cells: Each entry's cell, as its row in the cell list from 0.
        clones: Each entry's clone, as its position from 0.
    """

    cells: np.ndarray
    clones: np.ndarray


def _signature(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(8)


def _compression(path: str) -> str | None:
    return "gzip" if _signature(path).startswith(_GZIP_SIGNATURE) else None


def _open_text(path: str, compression: str | None) -> io.TextIOBase:
    if compression == "gzip":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _corrupt_error(path: str, error: BaseException) -> ValueError:
    return ValueError(f"{path} is cut short or corrupt: {error}")


@contextmanager
def _text_errors(path: str) -> Iterator[None]:
    """Name the file in the errors of reading it as UTF-8 text, plain or gzip."""
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


def _first_lines(path: str, compression: str | None, n_lines: int) -> list[str]:
    lines: list[str] = []
    with _text_errors(path), _open_text(path, compression) as handle:
        for line in handle:
            lines.append(line.rstrip("\r\n"))
            if len(lines) == n_lines:
                break
    return lines


def _records(lines: Iterable[str]) -> Iterator[str]:
    """Yield the rows of a table's text, each without its line ending.

    Blank lines are skipped, and a row whose quotes are open goes on to the
    next line.
    """
    record = ""
    for line in lines:
        text = line.rstrip("\r\n")
        record = f"{record}\n{text}" if record else text
        if not record.strip():
            record = ""
        elif not record.count('"') % 2:
            yield record
            record = ""


def _header_and_rows(path: str, compression: str | None) -> tuple[str, list[str]]:
    """Return a table's first line and up to ``_SAMPLE_ROWS`` rows after it.

    Blank lines after the first are skipped.
    """
    with _text_errors(path), _open_text(path, compression) as handle:
        header = next(handle, "").rstrip("\r\n")
        rows = list(itertools.islice(_records(handle), _SAMPLE_ROWS))
    return header, rows


def _ends_in_delimiter(row: str, delimiter: str) -> bool:
    """Return whether a row ends in its delimiter, with nothing after it.

    Space-separated rows lose the spaces at their ends when they are split,
    so none ends in its delimiter.
    """
    return delimiter != " " and row.endswith(delimiter)


def _rows_hold_values_after_their_names(
    path: str, compression: str | None, delimiter: str, n_names: int
) -> bool:
    """Return whether a row of a table holds anything after its named columns.

    Each row is read until one holds a field after the header's ``n_names``
    columns that is not empty: a value, or quoted empty text as R writes it.
    """
    with _text_errors(path), _open_text(path, compression) as handle:
        next(handle, "")
        for row in _records(handle):
            if len(_fields(row, delimiter)) == n_names + 1 and not _ends_in_delimiter(
                row, delimiter
            ):
                return True
    return False


def _delimiter(lines: list[str]) -> str:
    """Return the delimiter of a table: tab, comma or space.

    The header decides; a header of one name leaves it to the first row, so
    a quoted barcode that holds a comma or a space stays one field.
    """
    header = lines[0]
    for delimiter in ("\t", ","):
        if delimiter in header:
            return delimiter
    if " " in header.strip():
        return " "
    row = lines[1] if len(lines) > 1 else ""
    for delimiter in ("\t", ","):
        if delimiter in row:
            return delimiter
    return " "


def _fields(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        reader = csv.reader([line.strip()], delimiter=" ", skipinitialspace=True)
    else:
        reader = csv.reader([line], delimiter=delimiter)
    return next(reader, [])


def _text_layout(path: str) -> _TextLayout:
    compression = _compression(path)
    first, rows = _header_and_rows(path, compression)
    if not first.strip():
        raise ValueError(f"{path} is empty; a clone table needs a header line")
    delimiter = _delimiter([first, *rows[:1]])
    header = [name.strip() for name in _fields(first.lstrip("﻿"), delimiter)]
    # TREX starts its headers with "#", which marks no comment.
    if header and header[0].startswith("#"):
        header[0] = header[0][1:]
    row_names = False
    if rows:
        fields = _fields(rows[0], delimiter)
        if len(fields) == len(header) + 1:
            # R's write.table leaves its row names out of the header, and a
            # delimiter after each row's last field gives one field more too,
            # an empty one. The two differ only where a row holds something
            # after its last named column.
            if _ends_in_delimiter(
                rows[0], delimiter
            ) and not _rows_hold_values_after_their_names(
                path, compression, delimiter, len(header)
            ):
                raise ValueError(
                    f"The header of {path} names {len(header)} columns, and each "
                    f"of its rows holds {len(header) + 1} fields, the last one "
                    "empty: R's row names under a header that leaves them out, "
                    "with an empty last column, or a delimiter after each row's "
                    "last field. Name the row names in the header, as R's "
                    "write.table(..., col.names=NA) writes them, or end the header "
                    "with the delimiter too"
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
    return _TextLayout(
        path=path,
        names=tuple(header),
        delimiter=delimiter,
        compression=compression,
        row_names=row_names,
        row_chars=max((len(row) for row in rows), default=0),
    )


def _columns_mapping(columns: Any) -> dict[str, tuple[str, ...]]:
    if columns is None:
        return {}
    if not isinstance(columns, Mapping):
        raise TypeError(
            "columns must map 'cell', 'clone' and 'probability' to column names, "
            "or be None"
        )
    unknown = sorted(str(key) for key in columns if key not in _COLUMN_ROLES)
    if unknown:
        raise ValueError(
            f"columns names unknown roles {', '.join(map(repr, unknown))}; the "
            "roles are 'cell', 'clone' and 'probability'"
        )
    resolved: dict[str, tuple[str, ...]] = {}
    for role, value in columns.items():
        if isinstance(value, str):
            names = (value,)
        elif (
            role == "cell"
            and isinstance(value, Sequence)
            and not isinstance(value, bytes)
            and len(value) > 0
            and all(isinstance(name, str) for name in value)
        ):
            names = tuple(value)
        else:
            what = (
                "a column name or a sequence of names"
                if role == "cell"
                else "a column name"
            )
            raise TypeError(f"columns[{role!r}] must be {what}")
        if any(not name for name in names):
            raise ValueError(f"columns[{role!r}] names an empty column")
        resolved[role] = names
    return resolved


def _require_columns(layout: _TextLayout, names: Sequence[str]) -> None:
    """Refuse a table that lacks a column to read, or names it twice."""
    missing = [name for name in names if name not in layout.names]
    if missing:
        shown = ", ".join(repr(name) for name in layout.names if name)
        raise ValueError(
            f"{layout.path} has no column "
            + ", ".join(repr(name) for name in missing)
            + f"; its columns are {shown}"
        )
    repeated = sorted({name for name in names if layout.names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"{layout.path} names the column "
            + ", ".join(repr(name) for name in repeated)
            + " more than once, so which one to read is unclear"
        )


def _detected_preset(layout: _TextLayout) -> str | None:
    names = set(layout.names)
    matches = [
        name
        for name, preset in _PRESETS.items()
        if preset.cell in names and any(clone in names for clone in preset.clones)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{layout.path} has the columns of several clone tables: "
            + ", ".join(matches)
            + "; pass format= to choose one"
        )
    return matches[0] if matches else None


def _table_layout(
    path: str, format: str | None, columns: dict[str, tuple[str, ...]]
) -> CloneLayout:
    text = _text_layout(path)
    if format is None:
        format = "table" if columns else _detected_preset(text)
        if format is None:
            shown = ", ".join(repr(name) for name in text.names if name)
            raise ValueError(
                f"{path} has the columns {shown}, which match no clone table that "
                "CyteArc knows; name the cell and clone columns with "
                "columns={'cell': ..., 'clone': ...}"
            )
    if format != "table":
        if columns:
            raise ValueError(
                f"format={format!r} reads fixed columns; pass columns with "
                "format='table', 'matrix' or 'h5ad'"
            )
        preset = _PRESETS[format]
        clone = next((name for name in preset.clones if name in text.names), None)
        if clone is None:
            clone = preset.clones[0]
        needed = [preset.cell, clone] + (
            [preset.probability] if preset.probability else []
        )
        _require_columns(text, needed)
        return CloneLayout(
            format=format,
            path=path,
            cell=(preset.cell,),
            clone=clone,
            probability=preset.probability,
            _text=text,
        )
    if "clone" not in columns:
        raise ValueError(
            "format='table' needs the clone column: columns={'clone': ...}"
        )
    clone_name = columns["clone"][0]
    if "cell" in columns:
        cell = columns["cell"]
    elif text.row_names:
        cell = ()
    else:
        raise ValueError(
            f"{path} has no row names, so name its cell column: "
            "columns={'cell': ..., 'clone': ...}"
        )
    probability = columns["probability"][0] if "probability" in columns else None
    _require_columns(text, [*cell, clone_name, *([probability] if probability else [])])
    return CloneLayout(
        format="table",
        path=path,
        cell=cell,
        clone=clone_name,
        probability=probability,
        _text=text,
    )


def _matrix_header(path: str, compression: str | None) -> _MatrixHeader:
    with _text_errors(path), _open_text(path, compression) as handle:
        banner = handle.readline().strip().split()
        if len(banner) < 5 or banner[0].lower() != "%%matrixmarket":
            raise ValueError(f"{path} is not a Matrix Market file")
        kind, layout, field_type, symmetry = (word.lower() for word in banner[1:5])
        if kind != "matrix" or layout != "coordinate":
            raise ValueError(
                f"{path} is a {layout} Matrix Market {kind}; a clone matrix must be "
                "a coordinate matrix"
            )
        if field_type not in ("integer", "real", "pattern") or symmetry != "general":
            raise ValueError(
                f"{path} holds {field_type} {symmetry} values; a clone matrix holds "
                "integer, real or pattern values of a general matrix"
            )
        header_lines = 1
        for line in handle:
            header_lines += 1
            if line.startswith("%") or not line.strip():
                continue
            sizes = line.split()
            break
        else:
            raise ValueError(f"{path} has no size line after its comments")
    try:
        n_rows, n_columns, n_entries = (int(value) for value in sizes)
    except ValueError:
        raise ValueError(
            f"The size line of {path} must hold three whole numbers, not "
            f"{' '.join(sizes)!r}"
        ) from None
    return _MatrixHeader(
        n_rows=n_rows,
        n_columns=n_columns,
        n_entries=n_entries,
        pattern=field_type == "pattern",
        header_lines=header_lines,
        compression=compression,
    )


def _beside(path: str) -> tuple[str, str | None]:
    """Find a clone matrix's cell list, and its clone list, in its folder."""
    matrix = Path(path)
    folder = matrix.parent
    candidates: list[Path] = []
    larry = re.fullmatch(r"(.*)clone_matrix\.mtx(?:\.gz)?", matrix.name)
    if larry is not None:
        candidates += [
            folder / f"{larry.group(1)}metadata.txt{suffix}" for suffix in (".gz", "")
        ]
    exported = [folder / f"cells.tsv{suffix}" for suffix in (".gz", "")]
    found = [candidate for candidate in candidates + exported if candidate.is_file()]
    if not found:
        raise FileNotFoundError(
            f"The clone matrix {path} needs its cell list beside it: cells.tsv, or "
            "<name>metadata.txt.gz beside <name>clone_matrix.mtx.gz, as GEO ships "
            "LARRY's"
        )
    if len(found) > 1:
        raise ValueError(
            f"The clone matrix {path} has several cell lists beside it: "
            + ", ".join(os.fspath(candidate) for candidate in found)
            + "; keep one"
        )
    cells = found[0]
    clones = None
    if cells in exported:
        listed = [
            folder / f"clones.tsv{suffix}"
            for suffix in (".gz", "")
            if (folder / f"clones.tsv{suffix}").is_file()
        ]
        if len(listed) > 1:
            raise ValueError(
                f"The clone matrix {path} has two clone lists beside it; keep one"
            )
        clones = os.fspath(listed[0]) if listed else None
    return os.fspath(cells), clones


def _count_rows(text: _TextLayout) -> int:
    """Count a table's lines that hold text, its header aside."""
    count = 0
    with _text_errors(text.path), _open_text(text.path, text.compression) as handle:
        handle.readline()
        for line in handle:
            if line.strip():
                count += 1
    return count


def _list_cell_columns(
    text: _TextLayout, cell: tuple[str, ...] | None
) -> tuple[str, ...]:
    if cell is not None:
        _require_columns(text, cell)
        return cell
    named = [name for name in text.names if name]
    if text.row_names:
        return ()
    if len(named) == 1:
        return (named[0],)
    if "cell" in named:
        return ("cell",)
    raise ValueError(
        f"The cell list {text.path} has the columns "
        + ", ".join(repr(name) for name in named)
        + "; name the columns of each cell's barcode with columns={'cell': [...]}"
    )


def _matrix_layout(
    path: str, compression: str | None, columns: dict[str, tuple[str, ...]]
) -> CloneLayout:
    extra = sorted(set(columns) - {"cell"})
    if extra:
        raise ValueError(
            "format='matrix' takes only columns={'cell': ...}, the columns of its "
            "cell list"
        )
    header = _matrix_header(path, compression)
    cells_path, clones_path = _beside(path)
    cells_text = _text_layout(cells_path)
    cell = _list_cell_columns(cells_text, columns.get("cell"))
    n_cells = _count_rows(cells_text)
    n_clones = None
    if clones_path is not None:
        n_clones = _count_rows(_text_layout(clones_path))
    axes = [
        axis
        for axis, (cells, clones) in enumerate(
            ((header.n_rows, header.n_columns), (header.n_columns, header.n_rows))
        )
        if cells == n_cells and (n_clones is None or clones == n_clones)
    ]
    if not axes:
        raise ValueError(
            f"The clone matrix {path} is {header.n_rows} by {header.n_columns}, but "
            f"its cell list {cells_path} holds {n_cells} cells"
            + (f" and its clone list {n_clones} clones" if n_clones is not None else "")
        )
    # A matrix that fits both ways, such as a square one, holds its cells in
    # rows, as LARRY's files on GEO are documented and the export writes them.
    return CloneLayout(
        format="matrix",
        path=path,
        cell=cell,
        cells_path=cells_path,
        clones_path=clones_path,
        _cells_text=cells_text,
        _matrix=_MatrixHeader(
            n_rows=header.n_rows,
            n_columns=header.n_columns,
            n_entries=header.n_entries,
            pattern=header.pattern,
            header_lines=header.header_lines,
            compression=header.compression,
            cell_axis=axes[0],
        ),
    )


def _h5ad_layout(
    path: str, format: str | None, columns: dict[str, tuple[str, ...]]
) -> CloneLayout:
    with h5py.File(path, "r") as handle:
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group) or index_key(obs) is None:
            raise ValueError(f"{path} has no AnnData obs table with cell names")
        obsm = handle.get("obsm")
        has_x_clone = isinstance(obsm, h5py.Group) and "X_clone" in obsm
        has_mito = "MiTo_clone" in obs
        obs_names = set(obs.keys())
    if format is None:
        if columns:
            format = "h5ad"
        elif has_x_clone and has_mito:
            raise ValueError(
                f"{path} holds both CoSpar's obsm['X_clone'] and MiTo's "
                "obs['MiTo_clone']; pass format='cospar_h5ad' or format='mito'"
            )
        elif has_x_clone:
            format = "cospar_h5ad"
        elif has_mito:
            format = "mito"
        else:
            raise ValueError(
                f"{path} holds neither obsm['X_clone'] nor obs['MiTo_clone']; name "
                "the obs column of the clones with columns={'clone': ...}"
            )
    if format != "h5ad" and columns:
        raise ValueError(
            f"format={format!r} reads fixed columns; pass columns with "
            "format='table', 'matrix' or 'h5ad'"
        )
    if format == "cospar_h5ad":
        if not has_x_clone:
            raise ValueError(f"{path} has no obsm['X_clone']")
        return CloneLayout(format=format, path=path)
    if format == "mito":
        if not has_mito:
            raise ValueError(f"{path} has no obs['MiTo_clone']")
        return CloneLayout(format=format, path=path, clone="MiTo_clone")
    if "cell" in columns:
        raise ValueError(
            "format='h5ad' reads each cell's name from the obs index, so columns "
            "takes only 'clone' and 'probability'"
        )
    if "clone" not in columns:
        raise ValueError("format='h5ad' needs the clone column: columns={'clone': ...}")
    clone = columns["clone"][0]
    probability = columns["probability"][0] if "probability" in columns else None
    for name in (clone, probability):
        if name is not None and name not in obs_names:
            raise ValueError(f"{path} has no obs column {name!r}")
    return CloneLayout(format=format, path=path, clone=clone, probability=probability)


def clone_layout(
    path: str | os.PathLike[str],
    *,
    format: str | None = None,
    columns: Mapping[str, str | Sequence[str]] | None = None,
) -> CloneLayout:
    """Recognise a clone file's format and resolve the columns to read.

    The format follows from the file's contents: an HDF5 file is AnnData, a
    file that starts with ``%%MatrixMarket`` a clone matrix, and a text file
    a table whose header names its tool's columns. ``columns`` reads a table
    or an AnnData file by the columns it names instead, and ``format`` names
    the format of a file whose contents do not settle it.

    Args:
        path: The clone file.
        format: One of ``CLONE_FORMATS``, or None to recognise it.
        columns: ``"cell"``, ``"clone"`` and ``"probability"`` mapped to
            column names; ``"cell"`` may list several columns, which are
            joined by ``CELL_ID_SEPARATOR``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the format is unknown or does not fit the file, the
            file is empty or malformed, or a column is missing.
        FileNotFoundError: If the file, or a matrix's cell list, is missing.
    """
    if not isinstance(path, str | os.PathLike):
        raise TypeError("path must be a path")
    if format is not None and format not in CLONE_FORMATS:
        raise ValueError(
            f"format must be one of {', '.join(CLONE_FORMATS)}, or None, not {format!r}"
        )
    resolved = _columns_mapping(columns)
    name = os.fspath(path)
    signature = _signature(name)
    if not signature:
        raise ValueError(f"{name} is empty")
    if signature.startswith(_HDF5_SIGNATURE):
        if format is not None and format not in _H5AD_FORMATS:
            raise ValueError(f"{name} is an AnnData file, not format={format!r}")
        return _h5ad_layout(name, format, resolved)
    compression = _compression(name)
    first = _first_lines(name, compression, 1)
    if first and first[0].lower().startswith("%%matrixmarket"):
        if format not in (None, "matrix"):
            raise ValueError(f"{name} is a Matrix Market file, not format={format!r}")
        return _matrix_layout(name, compression, resolved)
    if format is not None and format not in _TEXT_FORMATS:
        raise ValueError(f"{name} is a text table, not format={format!r}")
    return _table_layout(name, format, resolved)


def _joined_barcodes(frame: pd.DataFrame, positions: Sequence[int]) -> np.ndarray:
    parts = [frame[position].str.strip() for position in positions]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + CELL_ID_SEPARATOR + part
    return np.asarray(joined.to_numpy(dtype=object))


def _line_list(
    rows: np.ndarray, first_line: int, text: "_TextLayout | None" = None
) -> str:
    """Name up to five lines of rows of a chunk that starts at ``first_line``.

    ``first_line`` counts rows, as ``_text_chunks`` yields it; with ``text``,
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


def _file_lines(text: _TextLayout, row_lines: list[int]) -> list[int]:
    """Return the file lines of rows that ``_text_chunks`` numbers from line 2.

    Only errors need them, so the file is read again from its start.
    """
    wanted = {line - 2 for line in row_lines}
    found: dict[int, int] = {}
    with _open_text(text.path, text.compression) as handle:
        handle.readline()
        if text.delimiter == " ":
            reader = csv.reader(handle, delimiter=" ", skipinitialspace=True)
        else:
            reader = csv.reader(handle, delimiter=text.delimiter)
        row, consumed = -1, 0
        for fields in reader:
            start = consumed + 2
            consumed = reader.line_num
            if _blank_record(fields, text.delimiter):
                continue
            row += 1
            if row in wanted:
                found[row] = start
                if len(found) == len(wanted):
                    break
    # A row the second reading misses keeps the line that counts rows.
    return [found.get(line - 2, line) for line in row_lines]


def _text_chunks(
    text: _TextLayout, chunk_rows: int
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield chunks of a table's rows as text, with each chunk's first line.

    Every field is read, so a row with more fields than the first raises.
    Blank lines hold no rows. The lines count rows, the first after the
    header being line 2, so a row's line is its chunk's first line plus its
    position; ``_line_list`` with ``text`` finds the line in the file, past
    blank lines and quoted fields that hold line breaks.
    """
    separator = r"\s+" if text.delimiter == " " else text.delimiter
    chunks = _parsed_chunks(
        text.path,
        sep=separator,
        header=None,
        skiprows=1,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        compression=text.compression,
        quotechar='"',
        chunksize=chunk_rows,
        skip_blank_lines=True,
        encoding="utf-8",
    )
    line = 2
    for chunk in chunks:
        yield line, chunk
        line += len(chunk)


def _parsed_chunks(path: str, **options: Any) -> Iterator[pd.DataFrame]:
    """Yield the chunks that pandas parses, naming the file in its errors."""
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


def _probabilities(
    values: pd.Series,
    first_line: int,
    path: str,
    table: "_TextLayout | None" = None,
) -> np.ndarray:
    text = values.str.strip()
    missing = text.isin(MISSING_TEXT).to_numpy()
    numbers = pd.to_numeric(text.where(~missing, None), errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid = np.flatnonzero(
        ~missing & (np.isnan(numbers) | (numbers < 0) | (numbers > 1))
    )
    if len(invalid):
        raise ValueError(
            f"Clone probabilities of {path} must be numbers from 0 to 1, or NA, on "
            f"lines {_line_list(invalid, first_line, table)}"
        )
    return np.asarray(numbers, dtype=np.float64)


def _table_labels(values: pd.Series, format: str) -> np.ndarray:
    text = values.str.strip()
    labels = text.to_numpy(dtype=object)
    missing = text.isin(MISSING_TEXT).to_numpy()
    if format == "cassiopeia":
        # Cassiopeia writes lineage group 0 for a cell without a clone, as an
        # integer or, beside missing groups, as a float, so 4.0 is group 4.
        numbers = pd.to_numeric(text.where(~missing, None), errors="coerce").to_numpy(
            dtype=np.float64
        )
        missing = missing | (numbers == 0)
        whole = ~missing & np.isfinite(numbers) & (numbers == np.floor(numbers))
        labels[whole] = [str(int(number)) for number in numbers[whole].tolist()]
    labels[missing] = None
    return np.asarray(labels)


def _iter_table_rows(layout: CloneLayout, chunk_rows: int) -> Iterator[CloneRows]:
    text = layout._text
    assert text is not None
    index = {name: position for position, name in enumerate(text.names)}
    cell_positions = [index[name] for name in layout.cell] if layout.cell else [0]
    clone_position = index[str(layout.clone)]
    probability_position = (
        index[layout.probability] if layout.probability is not None else None
    )
    for first_line, chunk in _text_chunks(text, chunk_rows):
        barcodes = _joined_barcodes(chunk, cell_positions)
        empty = np.flatnonzero(barcodes == "")
        if len(empty):
            raise ValueError(
                f"{layout.path} has rows without a cell barcode on lines "
                f"{_line_list(empty, first_line, text)}"
            )
        probabilities = None
        if probability_position is not None:
            probabilities = _probabilities(
                chunk[probability_position], first_line, layout.path, text
            )
        yield CloneRows(
            first_line=first_line,
            barcodes=barcodes,
            clones=_table_labels(chunk[clone_position], layout.format),
            probabilities=probabilities,
        )


def _obs_names(handle: h5py.File) -> h5py.Dataset | h5py.Group:
    """Return the node of an AnnData file's cell names.

    AnnData writes a text index as a dataset, or, in current releases, as a
    nullable string array: a group of values and a mask.
    """
    obs = handle["obs"]
    key = index_key(obs)
    names = obs.get(str(key))
    if not (isinstance(names, h5py.Dataset) and names.ndim == 1) and not is_nullable(
        names
    ):
        raise ValueError(
            f"{handle.filename} stores its cell names in a form CyteArc cannot read"
        )
    return names


def _n_obs(names: h5py.Dataset | h5py.Group) -> int:
    return int(column_length(names) or 0)


def _decoded(values: np.ndarray) -> np.ndarray:
    return np.asarray([as_text(value) for value in values.tolist()], dtype=object)


def _obs_name_rows(
    names: h5py.Dataset | h5py.Group, start: int, stop: int, path: str
) -> np.ndarray:
    """Return the names of cells ``[start, stop)`` as text, refusing missing ones."""
    values, missing = read_column(names, start, stop)
    if missing.any():
        raise ValueError(
            f"{path} has cells without a name in its obs index, the first in row "
            f"{start + int(np.flatnonzero(missing)[0]) + 1}"
        )
    return _decoded(np.asarray(values))


def _iter_h5ad_rows(layout: CloneLayout, chunk_rows: int) -> Iterator[CloneRows]:
    with h5py.File(layout.path, "r") as handle:
        names = _obs_names(handle)
        obs = handle["obs"]
        n_cells = _n_obs(names)
        for start in range(0, n_cells, chunk_rows):
            stop = min(start + chunk_rows, n_cells)
            values, missing = read_table_column(
                handle, obs, str(layout.clone), ("__categories",), start, stop
            )
            labels = clone_label_texts(values, missing)
            if layout.format == "mito":
                labels[labels == _MITO_UNASSIGNED] = None
            probabilities = None
            if layout.probability is not None:
                numbers, absent = read_table_column(
                    handle, obs, layout.probability, ("__categories",), start, stop
                )
                if np.asarray(numbers).dtype.kind not in "biuf":
                    raise ValueError(
                        f"The obs column {layout.probability!r} of {layout.path} "
                        "must hold numbers"
                    )
                probabilities = np.asarray(numbers, dtype=np.float64).copy()
                probabilities[np.asarray(absent, dtype=bool)] = np.nan
                bad = np.flatnonzero(
                    ~np.isnan(probabilities)
                    & ((probabilities < 0) | (probabilities > 1))
                )
                if len(bad):
                    raise ValueError(
                        f"The obs column {layout.probability!r} of {layout.path} "
                        "must hold probabilities from 0 to 1"
                    )
            yield CloneRows(
                first_line=start + 1,
                barcodes=_obs_name_rows(names, start, stop, layout.path),
                clones=labels,
                probabilities=probabilities,
            )


def iter_clone_rows(layout: CloneLayout, chunk_rows: int) -> Iterator[CloneRows]:
    """Yield the rows of a clone table or AnnData file in chunks.

    Args:
        layout: The layout of a file whose format is not a matrix.
        chunk_rows: The rows of each chunk.

    Raises:
        ValueError: If the layout is a matrix, a row has no cell barcode, a
            probability is not a number from 0 to 1, or the file is malformed.
    """
    if layout.is_matrix:
        raise ValueError(f"format={layout.format!r} is a matrix; read it by entries")
    if layout.format in _H5AD_FORMATS:
        yield from _iter_h5ad_rows(layout, chunk_rows)
    else:
        yield from _iter_table_rows(layout, chunk_rows)


def matrix_cells(layout: CloneLayout, chunk_rows: int) -> Iterator[np.ndarray]:
    """Yield the barcodes of a clone matrix's cells in chunks, in matrix order.

    Raises:
        ValueError: If the layout is not a matrix, or a cell has no barcode.
    """
    if not layout.is_matrix:
        raise ValueError(f"format={layout.format!r} is not a matrix")
    if layout.format == "cospar_h5ad":
        with h5py.File(layout.path, "r") as handle:
            names = _obs_names(handle)
            n_cells = _n_obs(names)
            for start in range(0, n_cells, chunk_rows):
                stop = min(start + chunk_rows, n_cells)
                yield _obs_name_rows(names, start, stop, layout.path)
        return
    text = layout._cells_text
    assert text is not None
    index = {name: position for position, name in enumerate(text.names)}
    positions = [index[name] for name in layout.cell] if layout.cell else [0]
    for first_line, chunk in _text_chunks(text, chunk_rows):
        barcodes = _joined_barcodes(chunk, positions)
        empty = np.flatnonzero(barcodes == "")
        if len(empty):
            raise ValueError(
                f"The cell list {text.path} has cells without a barcode on lines "
                f"{_line_list(empty, first_line, text)}"
            )
        yield barcodes


def matrix_clone_ids(layout: CloneLayout) -> np.ndarray:
    """Return the labels of a clone matrix's clones, in matrix order.

    A matrix without a clone list labels each clone by its position from 0,
    as SciPy and AnnData number the columns.

    Raises:
        ValueError: If the layout is not a matrix, or a listed clone has no
            label or repeats.
    """
    if not layout.is_matrix:
        raise ValueError(f"format={layout.format!r} is not a matrix")
    if layout.format == "cospar_h5ad":
        with h5py.File(layout.path, "r") as handle:
            n_clones = _x_clone_shape(handle)[1]
        return np.arange(n_clones).astype(str).astype(object)
    header = layout._matrix
    assert header is not None
    if layout.clones_path is None:
        return np.arange(header.n_clones).astype(str).astype(object)
    text = _text_layout(layout.clones_path)
    labels = np.concatenate(
        [
            chunk[0].str.strip().to_numpy(dtype=object)
            for _line, chunk in _text_chunks(text, 1 << 20)
        ]
        or [np.empty(0, dtype=object)]
    )
    if np.any(labels == "") or pd.Index(labels).has_duplicates:
        raise ValueError(
            f"The clone list {layout.clones_path} must give each clone its own label"
        )
    return labels


def _x_clone_shape(handle: h5py.File) -> tuple[int, int]:
    node = handle["obsm"]["X_clone"]
    if isinstance(node, h5py.Dataset):
        if node.ndim != 2:
            raise ValueError(f"obsm['X_clone'] of {handle.filename} is not a matrix")
        return int(node.shape[0]), int(node.shape[1])
    shape = sparse_shape(node)
    if shape is None or sparse_encoding(node) is None:
        raise ValueError(
            f"obsm['X_clone'] of {handle.filename} is neither a dense matrix nor a "
            "CSR or CSC sparse matrix"
        )
    return shape


def _x_clone_entries(layout: CloneLayout, chunk_rows: int) -> Iterator[MatrixEntries]:
    with h5py.File(layout.path, "r") as handle:
        n_obs = _n_obs(_obs_names(handle))
        n_rows, n_columns = _x_clone_shape(handle)
        if n_rows != n_obs:
            raise ValueError(
                f"obsm['X_clone'] of {layout.path} has {n_rows} rows for {n_obs} cells"
            )
        node = handle["obsm"]["X_clone"]
        if isinstance(node, h5py.Dataset):
            # A block holds as many values as a chunk of three-field rows.
            block_rows = max(1, (3 * chunk_rows) // max(n_columns, 1))
            for start in range(0, n_rows, block_rows):
                block = np.asarray(node[start : start + block_rows])
                rows, columns = np.nonzero(block)
                yield MatrixEntries(cells=rows + start, clones=columns)
            return
        encoding = sparse_encoding(node)
        indptr = node["indptr"]
        major = n_rows if encoding == "csr" else n_columns
        for start in range(0, major, chunk_rows):
            stop = min(start + chunk_rows, major)
            pointers = np.asarray(indptr[start : stop + 1], dtype=np.int64)
            first, last = int(pointers[0]), int(pointers[-1])
            indices = np.asarray(node["indices"][first:last], dtype=np.int64)
            data = np.asarray(node["data"][first:last])
            majors = np.repeat(np.arange(start, stop), np.diff(pointers))
            stored = data != 0
            if encoding == "csr":
                yield MatrixEntries(cells=majors[stored], clones=indices[stored])
            else:
                yield MatrixEntries(cells=indices[stored], clones=majors[stored])


def _mtx_entries(layout: CloneLayout, chunk_rows: int) -> Iterator[MatrixEntries]:
    header = layout._matrix
    assert header is not None
    columns = [0, 1] if header.pattern else [0, 1, 2]
    chunks = _parsed_chunks(
        layout.path,
        sep=r"\s+",
        header=None,
        skiprows=header.header_lines,
        usecols=columns,
        compression=header.compression,
        chunksize=chunk_rows,
        dtype={column: np.float64 for column in columns},
    )
    seen = 0
    for chunk in chunks:
        seen += len(chunk)
        coordinates = chunk[[0, 1]].to_numpy(dtype=np.float64)
        whole = np.isfinite(coordinates) & (coordinates == np.floor(coordinates))
        rows = coordinates[:, 0].astype(np.int64)
        cols = coordinates[:, 1].astype(np.int64)
        outside = (
            ~whole.all(axis=1)
            | (rows < 1)
            | (rows > header.n_rows)
            | (cols < 1)
            | (cols > header.n_columns)
        )
        if outside.any():
            raise ValueError(
                f"The clone matrix {layout.path} has entries outside its "
                f"{header.n_rows} by {header.n_columns} shape"
            )
        member = np.ones(len(chunk), dtype=bool)
        if not header.pattern:
            values = chunk[2].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"The clone matrix {layout.path} holds values that are not finite"
                )
            member = values != 0
        cells, clones = (rows, cols) if header.cell_axis == 0 else (cols, rows)
        yield MatrixEntries(cells=cells[member] - 1, clones=clones[member] - 1)
    if seen != header.n_entries:
        raise ValueError(
            f"The clone matrix {layout.path} declares {header.n_entries} entries but "
            f"holds {seen}"
        )


def matrix_entries(layout: CloneLayout, chunk_rows: int) -> Iterator[MatrixEntries]:
    """Yield the nonzero entries of a clone matrix in chunks.

    Raises:
        ValueError: If the layout is not a matrix, an entry lies outside the
            matrix or is not finite, the matrix holds fewer or more entries
            than it declares, or its rows differ from its cells.
    """
    if not layout.is_matrix:
        raise ValueError(f"format={layout.format!r} is not a matrix")
    if layout.format == "cospar_h5ad":
        yield from _x_clone_entries(layout, chunk_rows)
    else:
        yield from _mtx_entries(layout, chunk_rows)
