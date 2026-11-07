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
Blank lines hold no rows, and a row of fewer or more fields than the header
is refused (:mod:`cytearc.utils.delimited`). A clone label that is one of
:data:`cytearc.metadata.records.MISSING_TEXT`, such as ``NA``, ``None`` or
``NULL``, means no clone, and a cell barcode joined from several columns
must have every part.
"""

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..clones.assignments import clone_label_texts
from ..metadata.records import MISSING_TEXT, file_lines
from ..utils.files import HDF5_SIGNATURE, file_signature
from ..utils.delimited import (
    TextLayout,
    count_rows,
    first_lines,
    joined_columns,
    require_columns,
    stripped_text,
    text_chunks,
    text_layout,
)
from ._h5ad_columns import (
    index_key,
    index_length,
    obs_names,
    read_table_column,
    sparse_encoding,
    sparse_shape,
)
from ._matrix_market import (
    MatrixMarketHeader,
    matrix_market_entries,
    matrix_market_header,
)
from ..utils.arguments import quoted_value

__all__ = [
    "CELL_ID_SEPARATOR",
    "CLONE_FORMATS",
    "CloneLayout",
    "CloneRows",
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
_MITO_UNASSIGNED = "unassigned"


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
    _text: "TextLayout | None" = field(default=None, repr=False)
    _cells_text: "TextLayout | None" = field(default=None, repr=False)
    _matrix: "MatrixMarketHeader | None" = field(default=None, repr=False)
    # The matrix axis that holds the cells: 0 for rows, 1 for columns.
    _cell_axis: int = field(default=0, repr=False)

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
class CloneRows:
    """A chunk of rows of a clone table, in file order.

    Attributes:
        lines: Each row's line in the file, the header being line 1, or an
            AnnData file's row of each cell, counting from 1.
        barcodes: Each row's cell barcode, as text.
        clones: Each row's clone label, None for none.
        probabilities: Each row's clone probability, NaN for none, or None
            when the format gives none.
    """

    lines: np.ndarray
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
            raise TypeError(f"columns[{quoted_value(role)}] must be {what}")
        if any(not name for name in names):
            raise ValueError(f"columns[{quoted_value(role)}] names an empty column")
        resolved[role] = names
    return resolved


def _detected_preset(layout: TextLayout) -> str | None:
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
    text = text_layout(path)
    if format is None:
        format = "table" if columns else _detected_preset(text)
        if format is None:
            shown = ", ".join(quoted_value(name) for name in text.names if name)
            raise ValueError(
                f"{path} has the columns {shown}, which match no clone table that "
                "CyteArc knows; name the cell and clone columns with "
                "columns={'cell': ..., 'clone': ...}"
            )
    if format != "table":
        if columns:
            raise ValueError(
                f"format={quoted_value(format)} reads fixed columns; pass columns with "
                "format='table', 'matrix' or 'h5ad'"
            )
        preset = _PRESETS[format]
        clone = next((name for name in preset.clones if name in text.names), None)
        if clone is None:
            clone = preset.clones[0]
        needed = [preset.cell, clone] + (
            [preset.probability] if preset.probability else []
        )
        require_columns(text, needed)
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
    require_columns(text, [*cell, clone_name, *([probability] if probability else [])])
    return CloneLayout(
        format="table",
        path=path,
        cell=cell,
        clone=clone_name,
        probability=probability,
        _text=text,
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


def _list_cell_columns(
    text: TextLayout, cell: tuple[str, ...] | None
) -> tuple[str, ...]:
    if cell is not None:
        require_columns(text, cell)
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
        + ", ".join(quoted_value(name) for name in named)
        + "; name the columns of each cell's barcode with columns={'cell': [...]}"
    )


def _matrix_layout(path: str, columns: dict[str, tuple[str, ...]]) -> CloneLayout:
    extra = sorted(set(columns) - {"cell"})
    if extra:
        raise ValueError(
            "format='matrix' takes only columns={'cell': ...}, the columns of its "
            "cell list"
        )
    header = matrix_market_header(path, fields=("integer", "real", "pattern"))
    cells_path, clones_path = _beside(path)
    cells_text = text_layout(cells_path)
    cell = _list_cell_columns(cells_text, columns.get("cell"))
    n_cells = count_rows(cells_text)
    n_clones = None
    if clones_path is not None:
        n_clones = count_rows(text_layout(clones_path))
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
        _matrix=header,
        _cell_axis=axes[0],
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
            f"format={quoted_value(format)} reads fixed columns; pass columns with "
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
            raise ValueError(f"{path} has no obs column {quoted_value(name)}")
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
            f"format must be one of {', '.join(CLONE_FORMATS)}, or None, not {quoted_value(format)}"
        )
    resolved = _columns_mapping(columns)
    name = os.fspath(path)
    head = file_signature(name)
    if not head:
        raise ValueError(f"{name} is empty")
    if head.startswith(HDF5_SIGNATURE):
        if format is not None and format not in _H5AD_FORMATS:
            raise ValueError(
                f"{name} is an AnnData file, not format={quoted_value(format)}"
            )
        return _h5ad_layout(name, format, resolved)
    first = first_lines(name, 1)
    if first and first[0].lower().startswith("%%matrixmarket"):
        if format not in (None, "matrix"):
            raise ValueError(
                f"{name} is a Matrix Market file, not format={quoted_value(format)}"
            )
        return _matrix_layout(name, resolved)
    if format is not None and format not in _TEXT_FORMATS:
        raise ValueError(f"{name} is a text table, not format={quoted_value(format)}")
    return _table_layout(name, format, resolved)


def _probabilities(values: pd.Series, lines: np.ndarray, path: str) -> np.ndarray:
    text = stripped_text(values)
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
            f"{file_lines(lines[invalid])}"
        )
    return np.asarray(numbers, dtype=np.float64)


def _refuse_cellless(
    barcodes: np.ndarray, lines: np.ndarray, path: str, columns: Sequence[str]
) -> None:
    """Refuse rows without a cell barcode, or without a part of one."""
    empty = np.flatnonzero(barcodes == "")
    if len(empty):
        what = (
            "rows without a cell barcode"
            if len(columns) < 2
            else "rows that lack a part of their cell barcode ("
            + ", ".join(quoted_value(name) for name in columns)
            + ")"
        )
        raise ValueError(f"{path} has {what} on {file_lines(lines[empty])}")


def _table_labels(values: pd.Series, format: str) -> np.ndarray:
    text = stripped_text(values)
    labels = text.to_numpy(dtype=object, copy=True)
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


def _iter_table_rows(
    layout: CloneLayout, chunk_rows: int, chunk_chars: int | None
) -> Iterator[CloneRows]:
    text = layout._text
    assert text is not None
    index = {name: position for position, name in enumerate(text.names)}
    cell_positions = [index[name] for name in layout.cell] if layout.cell else [0]
    clone_position = index[str(layout.clone)]
    probability_position = (
        index[layout.probability] if layout.probability is not None else None
    )
    for lines, chunk in text_chunks(text, chunk_rows, chunk_chars=chunk_chars):
        barcodes = joined_columns(chunk, cell_positions, CELL_ID_SEPARATOR)
        _refuse_cellless(barcodes, lines, layout.path, layout.cell)
        probabilities = None
        if probability_position is not None:
            probabilities = _probabilities(
                chunk[probability_position], lines, layout.path
            )
        yield CloneRows(
            lines=lines,
            barcodes=barcodes,
            clones=_table_labels(chunk[clone_position], layout.format),
            probabilities=probabilities,
        )


def _iter_h5ad_rows(layout: CloneLayout, chunk_rows: int) -> Iterator[CloneRows]:
    with h5py.File(layout.path, "r") as handle:
        obs = handle["obs"]
        n_cells = index_length(obs, layout.path)
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
                        f"The obs column {quoted_value(layout.probability)} of {layout.path} "
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
                        f"The obs column {quoted_value(layout.probability)} of {layout.path} "
                        "must hold probabilities from 0 to 1"
                    )
            yield CloneRows(
                lines=np.arange(start + 1, stop + 1, dtype=np.int64),
                barcodes=obs_names(handle, start, stop, path=layout.path),
                clones=labels,
                probabilities=probabilities,
            )


def iter_clone_rows(
    layout: CloneLayout, chunk_rows: int, *, chunk_chars: int | None = None
) -> Iterator[CloneRows]:
    """Yield the rows of a clone table or AnnData file in chunks.

    Args:
        layout: The layout of a file whose format is not a matrix.
        chunk_rows: The most rows of each chunk.
        chunk_chars: The most characters of a table's chunk, past the row it
            ends in, or None for chunks bounded by their rows alone. An
            AnnData file's chunks are bounded by their rows.

    Raises:
        ValueError: If the layout is a matrix, a row has no cell barcode, a
            probability is not a number from 0 to 1, or the file is malformed.
    """
    if layout.is_matrix:
        raise ValueError(
            f"format={quoted_value(layout.format)} is a matrix; read it by entries"
        )
    if layout.format in _H5AD_FORMATS:
        yield from _iter_h5ad_rows(layout, chunk_rows)
    else:
        yield from _iter_table_rows(layout, chunk_rows, chunk_chars)


def matrix_cells(
    layout: CloneLayout, chunk_rows: int, *, chunk_chars: int | None = None
) -> Iterator[np.ndarray]:
    """Yield the barcodes of a clone matrix's cells in chunks, in matrix order.

    A chunk holds at most ``chunk_rows`` cells, and at most ``chunk_chars``
    characters of a cell list's text, past the row it ends in, when given.

    Raises:
        ValueError: If the layout is not a matrix, or a cell has no barcode.
    """
    if not layout.is_matrix:
        raise ValueError(f"format={quoted_value(layout.format)} is not a matrix")
    if layout.format == "cospar_h5ad":
        with h5py.File(layout.path, "r") as handle:
            n_cells = index_length(handle["obs"], layout.path)
            for start in range(0, n_cells, chunk_rows):
                stop = min(start + chunk_rows, n_cells)
                yield obs_names(handle, start, stop, path=layout.path)
        return
    text = layout._cells_text
    assert text is not None
    index = {name: position for position, name in enumerate(text.names)}
    positions = [index[name] for name in layout.cell] if layout.cell else [0]
    for lines, chunk in text_chunks(text, chunk_rows, chunk_chars=chunk_chars):
        barcodes = joined_columns(chunk, positions, CELL_ID_SEPARATOR)
        _refuse_cellless(barcodes, lines, f"The cell list {text.path}", layout.cell)
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
        raise ValueError(f"format={quoted_value(layout.format)} is not a matrix")
    if layout.format == "cospar_h5ad":
        with h5py.File(layout.path, "r") as handle:
            n_clones = _x_clone_shape(handle)[1]
        return np.arange(n_clones).astype(str).astype(object)
    header = layout._matrix
    assert header is not None
    if layout.clones_path is None:
        n_clones = header.n_columns if layout._cell_axis == 0 else header.n_rows
        return np.arange(n_clones).astype(str).astype(object)
    text = text_layout(layout.clones_path)
    labels = np.concatenate(
        [
            stripped_text(chunk[0]).to_numpy(dtype=object)
            for _lines, chunk in text_chunks(text, 1 << 20)
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
        n_obs = index_length(handle["obs"], layout.path)
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
    for rows, columns, values in matrix_market_entries(header, chunk_rows):
        member = np.ones(len(rows), dtype=bool)
        if values is not None:
            numbers = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(numbers)):
                raise ValueError(
                    f"The clone matrix {layout.path} holds values that are not finite"
                )
            member = numbers != 0
        cells, clones = (rows, columns) if layout._cell_axis == 0 else (columns, rows)
        yield MatrixEntries(cells=cells[member], clones=clones[member])


def matrix_entries(layout: CloneLayout, chunk_rows: int) -> Iterator[MatrixEntries]:
    """Yield the nonzero entries of a clone matrix in chunks.

    Raises:
        ValueError: If the layout is not a matrix, an entry lies outside the
            matrix or is not finite, the matrix holds fewer or more entries
            than it declares, or its rows differ from its cells.
    """
    if not layout.is_matrix:
        raise ValueError(f"format={quoted_value(layout.format)} is not a matrix")
    if layout.format == "cospar_h5ad":
        yield from _x_clone_entries(layout, chunk_rows)
    else:
        yield from _mtx_entries(layout, chunk_rows)
