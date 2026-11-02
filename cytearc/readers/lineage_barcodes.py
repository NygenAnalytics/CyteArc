"""Read the lineage barcodes that extraction tools write, at any level.

Tools that extract expressed lineage barcodes from sequencing reads write
each cell's barcodes at one of three levels: one row per read, one row per
molecule (a cell, a UMI and a barcode), or one row per pair of a cell and a
barcode with its UMIs or reads. ``barcode_layout`` recognises a file's format
from its contents and resolves the columns that hold each row's cell,
barcode, UMI and counts; ``iter_barcode_rows`` then yields the rows in
bounded chunks. Joining cells and collapsing reads into molecules belong to
the caller.

Formats:

- ``"trex"``: TREX's ``molecules.txt`` and ``molecules_corrected.txt`` of
  every release, ``#cell_id``, ``umi``, ``clone_id`` and, since September
  2024, ``read_count``; one row per cell and UMI, whose cloneID may hold
  ``-`` (a base not covered) and ``0`` (a deletion). Smart-seq2 runs write no
  ``umi``, one row per read.
- ``"pycashier"``: pycashier's ``<sample>.umi_cell_labeled.barcode.tsv``,
  ``info``, ``umi``, ``cell`` and ``barcode``, one row per read, with a header
  since release 2024.1001 and without one before.
- ``"celltag"``: CellTag-multi's per-read tables, ``Cell.BC``, ``UMI`` and
  ``Cell.Tag``, without ``UMI`` for scATAC-seq, and the Biddy et al.
  workflow's parsed reads, which add ``Read.ID``, ``Read.Seq`` and ``Gene``.
- ``"fatemap"``: FateMap's ``stepThreeStarcodeShavedReads.txt``, ``cellID``,
  ``UMI``, ``OriginalBarcode``, ``BC50StarcodeD8`` and the barcodes of other
  lengths, and ``SampleNum``, with a header or without one; one row per
  molecule and original barcode, read by its 50-base barcode.
- ``"quicat"``: QuiCAT's per-sample ``<sample>_output.csv``, ``sample``,
  ``cell``, ``barcode`` and ``counts``, one row per cell and barcode.
- ``"quicat_wide"``: QuiCAT's merged ``barcodes_output.csv``, ``sample``,
  ``cell`` and one column per barcode, zeros where a cell lacks it.
- ``"bartab"``: BARtab's ``<sample>.counts.tsv``, ``gene``, ``cell`` and
  ``count``, one row per barcode and cell.
- ``"bartab_annotation"``: BARtab's ``<sample>_cell_barcode_annotation.tsv``,
  one row per cell in R's row-name layout, its barcodes and their UMIs as
  lists joined by ``;``, or by ``,`` in early files separated by spaces.
- ``"molecule_info"``: Cell Ranger's ``molecule_info.h5``, the molecules of
  one feature barcode library.
- ``"h5ad"``: an AnnData file of cells by barcodes, such as QuiCAT's
  ``barcodes_output.h5ad``, whose ``X`` counts UMIs.
- ``"reads"``, ``"molecules"`` and ``"pairs"``: any delimited table at that
  level, read by the columns that ``columns`` names.

QuiCAT counts distinct UMIs for 10x Genomics data and reads where a
technology has no UMIs; its formats read the counts as UMIs, and a table of
reads is read with ``format="pairs"`` and ``columns``. BARtab is GPL-3.0, so
its formats follow its documentation and published files only.
"""

import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NoReturn

import h5py
import numpy as np
import pandas as pd

from ..metadata.records import MISSING_TEXT, file_lines, parse_text
from ..utils.arrays import sorted_unique
from ._delimited import (
    HDF5_SIGNATURE,
    TextLayout,
    compression,
    first_lines,
    joined_columns,
    line_chunks,
    line_list,
    require_columns,
    signature,
    split_fields,
    text_chunks,
    text_layout,
)
from ._h5ad_columns import (
    column_length,
    index_key,
    read_column,
    read_table_column,
    sparse_encoding,
    sparse_shape,
)
from ._text import as_text
from .cellranger import (
    MIN_MOLECULE_ROWS,
    MoleculeInfoLibrary,
    is_molecule_info,
    iter_molecule_info,
    molecule_info_library,
)

__all__ = [
    "BARCODE_FORMATS",
    "COLUMN_SEPARATOR",
    "LIST_SEPARATORS",
    "TABLE_FORMATS",
    "BarcodeLayout",
    "BarcodeRows",
    "RowLevel",
    "barcode_layout",
    "iter_barcode_rows",
]

type RowLevel = Literal["reads", "molecules", "pairs"]

#: The values of ``format``.
BARCODE_FORMATS: tuple[str, ...] = (
    "trex",
    "pycashier",
    "celltag",
    "fatemap",
    "quicat",
    "quicat_wide",
    "bartab",
    "bartab_annotation",
    "molecule_info",
    "h5ad",
    "reads",
    "molecules",
    "pairs",
)
#: The formats of any table, named for what each of its rows is.
TABLE_FORMATS: tuple[RowLevel, ...] = ("reads", "molecules", "pairs")
#: The characters that may join the items of a list field.
LIST_SEPARATORS: tuple[str, ...] = (";", ",", ":", "|")
#: Joins the columns that together hold a cell barcode or a lineage barcode,
#: as a merged store joins sample and cell.
COLUMN_SEPARATOR = "__"

_TEXT_PRESETS = (
    "trex",
    "pycashier",
    "celltag",
    "fatemap",
    "quicat",
    "quicat_wide",
    "bartab",
    "bartab_annotation",
)
_HDF5_FORMATS = ("molecule_info", "h5ad")
_ROLES = ("cell", "barcode", "umi", "umis", "reads", "library", "separator")
# The roles that each level of table may read; pairs may also split lists.
_LEVEL_ROLES: dict[str, frozenset[str]] = {
    "reads": frozenset({"cell", "barcode", "umi", "library"}),
    "molecules": frozenset({"cell", "barcode", "umi", "reads", "library"}),
    "pairs": frozenset({"cell", "barcode", "umis", "reads", "library", "separator"}),
}
_H5AD_ROLES = frozenset({"umis", "reads", "library"})
# A pycashier file without a header, as releases up to v23.1.2 wrote it.
_PYCASHIER_COLUMNS = ("info", "umi", "cell", "barcode")
# FateMap's columns, which some of its files hold without a header.
_FATEMAP_COLUMNS = (
    "cellID",
    "UMI",
    "OriginalBarcode",
    "BC50StarcodeD8",
    "BC40StarcodeD8",
    "BC30StarcodeD8",
    "BC30StarcodeD6",
    "SampleNum",
)
# QuiCAT's columns of each cell, beside the barcodes of a merged table.
_QUICAT_CELL_COLUMNS = ("sample", "cell", "replicate", "condition")
# Columns that only clone tables hold, for the hint that they go to
# import_clones.
_CLONE_COLUMNS = frozenset(
    {"clone_nr", "clone#", "Clone_ID", "clone.id", "VBC", "lineageGrp", "clone_opt"}
)
_NUCLEOTIDES = re.compile(r"^[ACGTN]+$")
# Counts are stored as int64, so each must be below 2**63.
_COUNT_LIMIT = 2.0**63
# A chunk holds at most this many rows of a text table.
_MAX_CHUNK_ROWS = 50_000
# Bytes that a chunk holds per field of a text table while it parses it: the
# raw text, its stripped copy and the parsed values.
_TEXT_FIELD_BYTES = 1_000
# Bytes per field of a wide table of counts: its text, its count as a float
# and the masks that check the count.
_COUNT_FIELD_BYTES = 120
# Bytes per entry of an AnnData matrix read: its position, value, cell and
# barcode.
_ENTRY_BYTES = 200
# Bytes per molecule of a molecule_info.h5 read: its datasets, decoded UMI and
# cell barcode.
_MOLECULE_BYTES = 150
# Names and labels of an AnnData file are decoded this many at a time.
_NAME_BLOCK = 65_536
# Bytes of the reference that an object array holds to each decoded text.
_REFERENCE_BYTES = 8
# Bytes that a distinct library label takes in the table that codes labels,
# beside its text.
_LABEL_ENTRY_BYTES = 120
# Bytes of the cache of chunks that h5py keeps for each dataset it reads.
_CHUNK_CACHE_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class BarcodeLayout:
    """How to read the barcodes of one file.

    Attributes:
        format: One of ``BARCODE_FORMATS``.
        path: The file.
        level: What each row is: ``"reads"``, ``"molecules"`` or ``"pairs"``.
        library: The library of every row, unless ``library_column`` names
            each row's.
        cell: The columns whose values, joined by ``COLUMN_SEPARATOR``, are
            each row's cell barcode; empty for R's row names and AnnData's
            cell names.
        barcode: The columns whose values, joined by ``COLUMN_SEPARATOR``,
            are each row's lineage barcode; empty for a wide table and
            AnnData's barcode names.
        umi: The column of each row's UMI, or None.
        umis: The column of each pair's UMIs, or None.
        reads: The column of each row's reads, or None.
        library_column: The column of each row's library, or None.
        separator: The characters that join the items of list fields, or
            None.
        one_row_per_molecule: Whether the format lists each cell and UMI
            once, as TREX's molecule tables do.
        required: The columns that the format fills in every row, so that an
            empty value marks a row cut short.
    """

    format: str
    path: str
    level: RowLevel
    library: str
    cell: tuple[str, ...] = ()
    barcode: tuple[str, ...] = ()
    umi: str | None = None
    umis: str | None = None
    reads: str | None = None
    library_column: str | None = None
    separator: str | None = None
    one_row_per_molecule: bool = False
    required: tuple[str, ...] = ()
    _text: TextLayout | None = field(default=None, repr=False)
    _molecules: MoleculeInfoLibrary | None = field(default=None, repr=False)
    _matrix: "_Matrix | None" = field(default=None, repr=False)

    def _row_bytes(self) -> int:
        if self._text is not None:
            per_field = (
                _COUNT_FIELD_BYTES
                if self.format == "quicat_wide"
                else _TEXT_FIELD_BYTES
            )
            return per_field * len(self._text.names)
        return _ENTRY_BYTES if self._matrix is not None else _MOLECULE_BYTES

    def chunk_rows(self, chunk_bytes: int) -> int:
        """The rows of a chunk that holds at most about ``chunk_bytes``.

        A chunk of a text table holds rows of the file, of a matrix nonzero
        entries, and of Cell Ranger's molecule info molecules, at least
        65,536 of them.
        """
        rows = max(1, int(chunk_bytes) // self._row_bytes())
        if self._text is not None:
            return min(_MAX_CHUNK_ROWS, rows)
        if self._molecules is not None:
            return max(MIN_MOLECULE_ROWS, rows)
        return rows

    def chunk_bytes(self, chunk_rows: int) -> int:
        """The bytes that a chunk of ``chunk_rows`` holds at most, about."""
        held = self._row_bytes() * int(chunk_rows)
        if self._matrix is not None and self._matrix.encoding == "dense":
            # A dense matrix is read a tile at a time, whose entries are
            # parsed ``chunk_rows`` at a time, while HDF5 inflates each of
            # its chunks into a buffer of its own.
            height, width = self._matrix.tile_shape(chunk_rows)
            held += height * width * self._matrix.itemsize + self._matrix.buffer_bytes
        return held

    def held_bytes(self) -> int:
        """The bytes that reading holds beside its chunks, such as a matrix's names.

        An AnnData file's names of cells and barcodes, and its column of
        libraries, are read once to count them at the size they take
        decoded.
        """
        if self._molecules is not None:
            return self._molecules.barcode_block_bytes
        if self._matrix is None:
            return 0
        matrix = self._matrix
        held = _decoded_bytes_of_names(self.path, self.library_column)
        # A sparse matrix's pointers, their differences as they are checked,
        # and HDF5's buffers for its entries and their positions.
        if matrix.encoding == "csr":
            held += 16 * (matrix.n_cells + 1) + matrix.buffer_bytes
        elif matrix.encoding == "csc":
            held += 16 * (matrix.n_barcodes + 1) + matrix.buffer_bytes
        return held

    def record(self) -> dict[str, Any]:
        """Return the format, columns and library that identify how the file is read."""
        return {
            "format": self.format,
            "level": self.level,
            "library": None if self.library_column is not None else self.library,
            "columns": {
                "cell": list(self.cell),
                "barcode": list(self.barcode),
                "umi": self.umi,
                "umis": self.umis,
                "reads": self.reads,
                "library": self.library_column,
                "separator": self.separator,
            },
        }


@dataclass(frozen=True, slots=True)
class _Matrix:
    """An AnnData file's matrix of counts: its layout, shape and value size.

    ``chunks`` is the shape of the HDF5 chunks of a dense matrix, or None
    for one stored without chunks or a sparse one, and ``buffer_bytes`` the
    bytes of the chunk that HDF5 inflates and caches for each dataset read.
    """

    encoding: str
    n_cells: int
    n_barcodes: int
    itemsize: int
    chunks: tuple[int, int] | None = None
    buffer_bytes: int = 0

    def tile_shape(self, entries: int) -> tuple[int, int]:
        """The rows and columns of the tiles that a dense matrix is read in.

        A tile covers whole HDF5 chunks, so that each chunk is read and
        decompressed once, and holds about ``entries`` values, or one chunk
        if a chunk holds more; a matrix stored without chunks is read in
        whole rows when a row fits.
        """
        entries = max(1, int(entries))
        chunk_rows, chunk_columns = self.chunks or (1, 1)
        chunk_rows = max(1, min(chunk_rows, self.n_cells))
        chunk_columns = max(1, min(chunk_columns, self.n_barcodes))
        width = entries // chunk_rows // chunk_columns * chunk_columns
        width = max(1, min(self.n_barcodes, max(chunk_columns, width)))
        height = entries // width // chunk_rows * chunk_rows
        height = max(1, min(self.n_cells, max(chunk_rows, height)))
        return height, width


@dataclass(frozen=True, slots=True)
class BarcodeRows:
    """A chunk of the rows of a barcode file.

    Attributes:
        first_line: The file line of the chunk's first row, counting from 1.
        cells: Each row's cell barcode, as text.
        barcodes: Each row's lineage barcode, None where a row has none.
        umis: Each row's UMI, or None when the rows have none.
        umi_counts: Each row's UMIs, -1 where unknown, or None.
        reads: Each row's reads, -1 where unknown, or None.
        libraries: Each row's library, or None when the file's library is
            every row's.
    """

    first_line: int
    cells: np.ndarray
    barcodes: np.ndarray
    umis: np.ndarray | None = None
    umi_counts: np.ndarray | None = None
    reads: np.ndarray | None = None
    libraries: np.ndarray | None = None


def _file_library(path: str, library: str | None) -> str:
    """The library of a file's rows: ``library``, or the file's name."""
    if library is not None:
        return library
    stem = Path(path).name.split(".", 1)[0]
    if not stem:
        raise ValueError(
            f"The barcode file {path!r} has no name before its extensions to name "
            "its library; give it a library"
        )
    return stem


def _role_columns(columns: Any) -> dict[str, Any]:
    """Check the roles of ``columns`` and the columns each names."""
    if columns is None:
        return {}
    if not isinstance(columns, Mapping):
        raise TypeError(
            "columns must map roles such as 'cell' and 'barcode' to column names, "
            "or be None"
        )
    unknown = sorted(str(role) for role in columns if role not in _ROLES)
    if unknown:
        raise ValueError(
            f"columns names unknown roles {', '.join(map(repr, unknown))}; the roles "
            "are " + ", ".join(map(repr, _ROLES))
        )
    resolved: dict[str, Any] = {}
    for role, value in columns.items():
        if role == "separator":
            if value not in LIST_SEPARATORS:
                raise ValueError(
                    "columns['separator'] must be one of "
                    + ", ".join(map(repr, LIST_SEPARATORS))
                    + f", not {value!r}"
                )
            resolved[role] = value
            continue
        if isinstance(value, str):
            names: tuple[str, ...] = (value,)
        elif (
            role in ("cell", "barcode")
            and isinstance(value, Sequence)
            and not isinstance(value, bytes)
            and len(value) > 0
            and all(isinstance(name, str) for name in value)
        ):
            names = tuple(value)
        else:
            what = (
                "a column name or a sequence of names"
                if role in ("cell", "barcode")
                else "a column name"
            )
            raise TypeError(f"columns[{role!r}] must be {what}")
        if any(not name for name in names):
            raise ValueError(f"columns[{role!r}] names an empty column")
        resolved[role] = names
    return resolved


def _hint(text: TextLayout) -> str:
    # TREX's clones.txt of 2022 starts with "#clone_id".
    if _CLONE_COLUMNS & set(text.names) or text.names[:1] == ("clone_id",):
        return (
            "; it holds clone IDs, so import it with ds.clones.import_clones, which "
            "reads clone tables"
        )
    return (
        "; read it with format='reads', 'molecules' or 'pairs' and columns that "
        "name its cell and barcode columns"
    )


def _detected_text_format(text: TextLayout) -> str | None:
    names = set(text.names)
    matches = []
    if text.names and text.names[0] == "cell_id" and "clone_id" in names:
        matches.append("trex")
    if set(_PYCASHIER_COLUMNS) <= names:
        matches.append("pycashier")
    if {"Cell.BC", "Cell.Tag"} <= names:
        matches.append("celltag")
    if {"cellID", "UMI", "BC50StarcodeD8"} <= names:
        matches.append("fatemap")
    if {"sample", "cell", "barcode", "counts"} <= names:
        matches.append("quicat")
    elif text.names[:2] == ("sample", "cell"):
        matches.append("quicat_wide")
    if names == {"gene", "cell", "count"}:
        matches.append("bartab")
    if text.names[text.row_names :] == ("barcode", "bc.umi.count") and (
        text.row_names or _header_only(text)
    ):
        matches.append("bartab_annotation")
    if len(matches) > 1:
        raise ValueError(
            f"{text.path} has the columns of several barcode tables: "
            + ", ".join(matches)
            + "; pass format= to choose one"
        )
    return matches[0] if matches else None


def _header_only(text: TextLayout) -> bool:
    """Whether a table holds its header and no row."""
    return len(first_lines(text.path, text.compression, 2)) < 2


def _is_nucleotides(value: str) -> bool:
    return bool(_NUCLEOTIDES.match(value))


def _headerless_format(path: str, found: str | None) -> str | None:
    """Recognise a table without a header by the fields of its first line."""
    lines = first_lines(path, found, 1)
    if not lines:
        return None
    fields = lines[0].split("\t")
    if (
        len(fields) == len(_PYCASHIER_COLUMNS)
        and fields[0].startswith("@")
        and all(_is_nucleotides(value) for value in fields[1:3])
    ):
        return "pycashier"
    if (
        len(fields) == len(_FATEMAP_COLUMNS)
        and all(_is_nucleotides(value) for value in fields[:2])
        and fields[-1].strip().isdigit()
    ):
        return "fatemap"
    return None


def _headerless_layout(path: str, found: str | None, format: str) -> TextLayout:
    names = _PYCASHIER_COLUMNS if format == "pycashier" else _FATEMAP_COLUMNS
    lines = first_lines(path, found, 1)
    n_fields = len(lines[0].split("\t")) if lines else 0
    if n_fields != len(names):
        raise ValueError(
            f"{path} has no header, so it must hold the {len(names)} columns of "
            f"format={format!r} in order, but its first row holds {n_fields} fields"
        )
    return TextLayout(
        path=path,
        names=names,
        delimiter="\t",
        compression=found,
        row_names=False,
        header=False,
    )


def _preset_layout(format: str, text: TextLayout, library: str) -> BarcodeLayout:
    names = set(text.names)

    def preset(level: RowLevel, **columns: Any) -> BarcodeLayout:
        return BarcodeLayout(
            format=format,
            path=text.path,
            level=level,
            library=library,
            _text=text,
            **columns,
        )

    if format == "trex":
        require_columns(text, ["cell_id", "clone_id"])
        umi = "umi" if "umi" in names else None
        reads = "read_count" if "read_count" in names and umi else None
        return preset(
            "molecules" if umi else "reads",
            cell=("cell_id",),
            barcode=("clone_id",),
            umi=umi,
            reads=reads,
            one_row_per_molecule=umi is not None,
            required=tuple(name for name in ("clone_id", reads) if name),
        )
    if format == "pycashier":
        require_columns(text, _PYCASHIER_COLUMNS)
        return preset(
            "reads",
            cell=("cell",),
            barcode=("barcode",),
            umi="umi",
            required=("barcode",),
        )
    if format == "celltag":
        require_columns(text, ["Cell.BC", "Cell.Tag"])
        return preset(
            "reads",
            cell=("Cell.BC",),
            barcode=("Cell.Tag",),
            umi="UMI" if "UMI" in names else None,
            required=("Cell.Tag",),
        )
    if format == "fatemap":
        require_columns(text, ["cellID", "UMI", "BC50StarcodeD8"])
        # FateMap ends each row with its sample, so a row cut short lacks it.
        return preset(
            "molecules",
            cell=("cellID",),
            barcode=("BC50StarcodeD8",),
            umi="UMI",
            required=(
                "BC50StarcodeD8",
                *(("SampleNum",) if "SampleNum" in names else ()),
            ),
        )
    if format == "quicat":
        require_columns(text, ["cell", "barcode", "counts"])
        # A cell without barcodes has a row without a barcode or a count.
        return preset("pairs", cell=("cell",), barcode=("barcode",), umis="counts")
    if format == "quicat_wide":
        if text.row_names or "" in text.names:
            raise ValueError(
                f"{text.path} has an unnamed column, such as an index that pandas "
                "wrote; QuiCAT's merged table has none, so remove it"
            )
        require_columns(text, ["cell"])
        barcodes = [name for name in text.names if name not in _QUICAT_CELL_COLUMNS]
        if not barcodes:
            raise ValueError(f"{text.path} has no barcode columns")
        require_columns(text, barcodes)
        return preset("pairs", cell=("cell",))
    if format == "bartab":
        require_columns(text, ["gene", "cell", "count"])
        return preset(
            "pairs",
            cell=("cell",),
            barcode=("gene",),
            umis="count",
            required=("gene", "count"),
        )
    if not text.row_names:
        if not (text.names == ("barcode", "bc.umi.count") and _header_only(text)):
            raise ValueError(
                f"{text.path} is not in R's row-name layout, as BARtab writes its "
                "barcode annotation: one header field fewer than each row"
            )
        # An annotation without cells holds its header alone.
        text = replace(text, names=("", *text.names), row_names=True)
    require_columns(text, ["barcode", "bc.umi.count"])
    return preset(
        "pairs",
        barcode=("barcode",),
        umis="bc.umi.count",
        separator=";,",
        required=("barcode", "bc.umi.count"),
    )


def _table_layout(
    format: str, text: TextLayout, library: str, roles: dict[str, Any]
) -> BarcodeLayout:
    allowed = _LEVEL_ROLES[format]
    refused = sorted(set(roles) - allowed)
    if refused:
        raise ValueError(
            f"format={format!r} reads no column for "
            + ", ".join(map(repr, refused))
            + "; its roles are "
            + ", ".join(map(repr, sorted(allowed)))
        )
    if "barcode" not in roles:
        raise ValueError(
            f"format={format!r} needs the barcode column: columns={{'barcode': ...}}"
        )
    if format == "molecules" and "umi" not in roles:
        raise ValueError(
            "format='molecules' needs each molecule's UMI: columns={'umi': ...}; a "
            "table without UMIs is read with format='reads' or 'pairs'"
        )
    if "cell" in roles:
        cell = roles["cell"]
    elif text.row_names:
        cell = ()
    else:
        raise ValueError(
            f"{text.path} has no row names, so name its cell column: "
            "columns={'cell': ..., 'barcode': ...}"
        )
    separator = roles.get("separator")
    if separator is not None and len(roles["barcode"]) > 1:
        raise ValueError(
            "columns['separator'] splits one barcode column into lists, so "
            "columns['barcode'] must name one column"
        )
    single = {
        role: roles[role][0]
        for role in ("umi", "umis", "reads", "library")
        if role in roles
    }
    needed = [*cell, *roles["barcode"], *single.values()]
    require_columns(text, needed)
    return BarcodeLayout(
        format=format,
        path=text.path,
        level=format,  # type: ignore[arg-type]
        library=library,
        cell=cell,
        barcode=roles["barcode"],
        umi=single.get("umi"),
        umis=single.get("umis"),
        reads=single.get("reads"),
        library_column=single.get("library"),
        separator=separator,
        _text=text,
    )


def _h5ad_layout(path: str, library: str, roles: dict[str, Any]) -> BarcodeLayout:
    refused = sorted(set(roles) - _H5AD_ROLES)
    if refused:
        raise ValueError(
            "format='h5ad' reads cells and barcodes from the obs and var names, so "
            "columns takes only 'umis' or 'reads', naming 'X', and 'library', an "
            "obs column; not " + ", ".join(map(repr, refused))
        )
    counts = [role for role in ("umis", "reads") if role in roles]
    if len(counts) > 1:
        raise ValueError("X counts UMIs or reads, so name one of them in columns")
    if counts and roles[counts[0]][0] != "X":
        raise ValueError(
            f"columns[{counts[0]!r}] must name 'X', the counts of the file"
        )
    library_column = roles["library"][0] if "library" in roles else None
    with h5py.File(path, "r") as handle:
        obs, var = handle.get("obs"), handle.get("var")
        if not isinstance(obs, h5py.Group) or index_key(obs) is None:
            raise ValueError(f"{path} has no AnnData obs table with cell names")
        if not isinstance(var, h5py.Group) or index_key(var) is None:
            raise ValueError(f"{path} has no AnnData var table with barcode names")
        if "X" not in handle:
            raise ValueError(f"{path} has no X matrix of barcode counts")
        if library_column is not None and library_column not in obs:
            raise ValueError(f"{path} has no obs column {library_column!r}")
        matrix = _matrix(handle, path)
    return BarcodeLayout(
        format="h5ad",
        path=path,
        level="pairs",
        library=library,
        umis="X" if not counts or counts[0] == "umis" else None,
        reads="X" if counts and counts[0] == "reads" else None,
        library_column=library_column,
        _matrix=matrix,
    )


def _names(table: h5py.Group, path: str, what: str) -> Any:
    """Return the node of an AnnData table's names, its index.

    AnnData writes a text index as a dataset, or, in current releases, as a
    nullable string array: a group of values and a mask.

    Raises:
        ValueError: If the index is neither.
    """
    node = table.get(str(index_key(table)))
    if column_length(node) is None or (
        isinstance(node, h5py.Dataset) and node.ndim != 1
    ):
        raise ValueError(
            f"{path} stores its {what} names in a form CyteArc cannot read"
        )
    return node


def _matrix(handle: h5py.File, path: str) -> _Matrix:
    """Read the layout, shape and value size of an AnnData file's ``X``."""
    obs, var = handle["obs"], handle["var"]
    n_cells = int(column_length(_names(obs, path, "cell")) or 0)
    n_barcodes = int(column_length(_names(var, path, "barcode")) or 0)
    node = handle["X"]
    chunks = None
    if isinstance(node, h5py.Dataset):
        encoding, shape = "dense", tuple(int(size) for size in node.shape)
        itemsize = int(node.dtype.itemsize)
        if node.chunks is not None and len(node.chunks) == 2:
            chunks = (int(node.chunks[0]), int(node.chunks[1]))
        buffer_bytes = _buffer_bytes(node)
    else:
        encoding, shape = str(sparse_encoding(node)), sparse_shape(node) or ()
        data = node.get("data")
        itemsize = int(data.dtype.itemsize) if isinstance(data, h5py.Dataset) else 0
        buffer_bytes = sum(
            _buffer_bytes(node[name])
            for name in ("data", "indices")
            if isinstance(node.get(name), h5py.Dataset)
        )
    if encoding not in ("dense", "csr", "csc") or len(shape) != 2 or not itemsize:
        raise ValueError(f"X of {path} is neither a dense nor a sparse matrix")
    if shape != (n_cells, n_barcodes):
        raise ValueError(
            f"X of {path} is {shape[0]} by {shape[1]}, but the file names "
            f"{n_cells} cells and {n_barcodes} barcodes"
        )
    return _Matrix(
        encoding=encoding,
        n_cells=n_cells,
        n_barcodes=n_barcodes,
        itemsize=itemsize,
        chunks=chunks,
        buffer_bytes=buffer_bytes,
    )


def _buffer_bytes(dataset: h5py.Dataset) -> int:
    """The bytes of one inflated chunk of a dataset, and of h5py's cache of chunks."""
    if dataset.chunks is None:
        return _CHUNK_CACHE_BYTES
    elements = int(np.prod(dataset.chunks, dtype=np.int64))
    return elements * int(dataset.dtype.itemsize) + _CHUNK_CACHE_BYTES


def _name_blocks(node: Any, path: str) -> Iterator[list[str]]:
    """Decode an AnnData index of names, ``_NAME_BLOCK`` at a time.

    Raises:
        ValueError: If a name is missing.
    """
    n_names = int(column_length(node) or 0)
    for start in range(0, n_names, _NAME_BLOCK):
        values, missing = read_column(node, start, min(start + _NAME_BLOCK, n_names))
        if np.any(missing):
            row = start + int(np.flatnonzero(missing)[0]) + 1
            raise ValueError(
                f"{path} has a name missing from the index {node.name!r}, the first "
                f"in row {row}"
            )
        yield [as_text(value) for value in np.asarray(values).tolist()]


def _library_blocks(
    handle: h5py.File, obs: h5py.Group, column: str, n_cells: int, path: str
) -> Iterator[list[str]]:
    """Decode an ``obs`` column of libraries, ``_NAME_BLOCK`` cells at a time."""
    for start in range(0, n_cells, _NAME_BLOCK):
        labels, absent = read_table_column(
            handle,
            obs,
            column,
            ("__categories",),
            start,
            min(start + _NAME_BLOCK, n_cells),
        )
        if np.any(np.asarray(absent, dtype=bool)):
            raise ValueError(
                f"The obs column {column!r} of {path} lacks the library of some cells"
            )
        yield [as_text(value) for value in np.asarray(labels).tolist()]


def _texts_bytes(texts: list[str]) -> int:
    """The bytes that texts take in an object array, references included."""
    return sum(map(sys.getsizeof, texts)) + _REFERENCE_BYTES * len(texts)


def _decoded_bytes_of_names(path: str, library_column: str | None) -> int:
    """Count the bytes of an AnnData file's names and libraries, decoded.

    The names of cells and barcodes are held whole, the libraries as a code
    per cell and a table of their distinct labels, and decoding a block
    holds about as much again as the block takes.
    """
    held = 0
    largest = 0
    with h5py.File(path, "r") as handle:
        obs, var = handle["obs"], handle["var"]
        for table in (obs, var):
            what = "cell" if table is obs else "barcode"
            for names in _name_blocks(_names(table, path, what), path):
                size = _texts_bytes(names)
                held += size
                largest = max(largest, size)
        if library_column is not None:
            n_cells = int(column_length(_names(obs, path, "cell")) or 0)
            seen = np.empty(0, dtype=np.uint64)
            held += 4 * n_cells
            for labels in _library_blocks(handle, obs, library_column, n_cells, path):
                largest = max(largest, _texts_bytes(labels))
                hashes = np.asarray(
                    pd.util.hash_array(np.asarray(labels, dtype=object)),
                    dtype=np.uint64,
                )
                distinct, first = np.unique(hashes, return_index=True)
                new = ~np.isin(distinct, seen)
                held += sum(
                    sys.getsizeof(labels[index]) + _LABEL_ENTRY_BYTES
                    for index in first[new].tolist()
                )
                seen = np.union1d(seen, distinct[new])
            held += 8 * len(seen)
    return held + 2 * largest


def barcode_layout(
    path: str | os.PathLike[str],
    *,
    format: str | None = None,
    columns: Mapping[str, Any] | None = None,
    library: str | None = None,
) -> BarcodeLayout:
    """Recognise a barcode file's format and resolve the columns to read.

    The format follows from the file's contents: an HDF5 file is Cell
    Ranger's ``molecule_info.h5`` or AnnData, and a text table is recognised
    by its header, or, without one, by the fields of its first row. A table
    of another tool is read with ``format`` set to ``"reads"``,
    ``"molecules"`` or ``"pairs"`` and ``columns``, which names its columns.

    Args:
        path: The barcode file.
        format: One of ``BARCODE_FORMATS``, or None to recognise it.
        columns: Roles mapped to column names: ``"cell"`` and ``"barcode"``
            (a column, or several joined by ``COLUMN_SEPARATOR``), ``"umi"``,
            ``"umis"``, ``"reads"`` and ``"library"``, and ``"separator"``,
            one of ``LIST_SEPARATORS``, which splits the barcode and count
            fields of pairs into lists.
        library: The library of the file's rows, or None for the file's name
            before its first ``.``; for ``molecule_info``, the type of the
            library to read, which names its rows.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the format is unknown or does not fit the file, the
            file is empty or malformed, or a column is missing.
    """
    if not isinstance(path, str | os.PathLike):
        raise TypeError("path must be a path")
    if format is not None and format not in BARCODE_FORMATS:
        raise ValueError(
            f"format must be one of {', '.join(BARCODE_FORMATS)}, or None, not "
            f"{format!r}"
        )
    if library is not None and not isinstance(library, str):
        raise TypeError("library must be text or None")
    if library == "":
        raise ValueError("library must name a library, not be empty")
    roles = _role_columns(columns)
    name = os.fspath(path)
    head = signature(name)
    if not head:
        raise ValueError(f"{name} is empty")
    if roles and format not in (*TABLE_FORMATS, "h5ad"):
        raise ValueError(
            "columns names the columns of a table read with format='reads', "
            "'molecules' or 'pairs', or of an AnnData file read with format='h5ad'"
            + ("" if format is None else f"; format={format!r} reads fixed columns")
        )
    if roles.get("library") is not None and library is not None:
        raise ValueError(
            "columns names a library column, so library= would name no row; pass one"
        )
    if head.startswith(HDF5_SIGNATURE):
        if format is not None and format not in _HDF5_FORMATS:
            raise ValueError(f"{name} is an HDF5 file, not format={format!r}")
        if format == "molecule_info" or (format is None and is_molecule_info(name)):
            molecules = molecule_info_library(name, library)
            return BarcodeLayout(
                format="molecule_info",
                path=name,
                level="molecules",
                library=molecules.library_type,
                umi="umi",
                reads="count",
                _molecules=molecules,
            )
        return _h5ad_layout(name, _file_library(name, library), roles)
    if format in _HDF5_FORMATS:
        raise ValueError(f"{name} is a text table, not format={format!r}")
    file_library = _file_library(name, library)
    found = compression(name)
    if format in TABLE_FORMATS:
        text = text_layout(name, what="a barcode table")
        return _table_layout(format, text, file_library, roles)
    # A first line of a tool's reads or molecules is no header, so tables
    # without one are recognised before a header is read.
    headerless = _headerless_format(name, found)
    if format is None:
        if headerless is not None:
            text = _headerless_layout(name, found, headerless)
            return _preset_layout(headerless, text, file_library)
        text = text_layout(name, what="a barcode table")
        detected = _detected_text_format(text)
        if detected is None:
            shown = ", ".join(repr(column) for column in text.names if column)
            raise ValueError(
                f"{name} has the columns {shown}, which match no barcode table "
                "that CyteArc knows" + _hint(text)
            )
        return _preset_layout(detected, text, file_library)
    if format in ("pycashier", "fatemap"):
        if headerless == format:
            text = _headerless_layout(name, found, format)
        else:
            text = text_layout(name, what="a barcode table")
            if _detected_text_format(text) != format:
                text = _headerless_layout(name, found, format)
    else:
        text = text_layout(name, what="a barcode table")
    return _preset_layout(format, text, file_library)


def _distinct_text(values: np.ndarray) -> np.ndarray:
    """Parse text as ``parse_text`` does, once per distinct value of the column.

    Rows of reads repeat few cells, UMIs and barcodes, so stripping and
    finding missing tokens per distinct value saves most of the work.
    """
    codes, distinct = pd.factorize(np.asarray(values, dtype=object))
    parsed: np.ndarray = parse_text(np.asarray(distinct, dtype=object))[codes]
    return parsed


def _on_lines(rows: np.ndarray, first_line: int, text: TextLayout) -> str:
    """Name the file lines of rows of a chunk, such as ``"lines 4, 7"``.

    ``first_line`` counts rows, as ``text_chunks`` yields it, so the lines
    are found in the file past blank lines and quoted line breaks.
    """
    noun = "line" if len(rows) == 1 else "lines"
    return f"{noun} {line_list(np.asarray(rows), first_line, text)}"


def _stripped_cells(
    values: np.ndarray, lines: np.ndarray, path: str, text: TextLayout | None = None
) -> np.ndarray:
    """Strip cell barcodes once per distinct value, refusing rows without one.

    A missing-value token is a cell barcode as any other text is. ``lines``
    are the rows' file lines, or with ``text`` the lines that count rows.
    """
    codes, distinct = pd.factorize(np.asarray(values, dtype=object))
    stripped = np.asarray([value.strip() for value in distinct], dtype=object)
    cells: np.ndarray = stripped[codes]
    _refuse_cellless(cells, lines, path, text)
    return cells


def _refuse_cellless(
    cells: np.ndarray, lines: np.ndarray, path: str, text: TextLayout | None = None
) -> None:
    empty = np.flatnonzero(cells == "")
    if len(empty):
        raise ValueError(
            f"{path} has rows without a cell barcode on lines "
            f"{line_list(lines[empty], 0, text)}"
        )


def _cell_texts(
    chunk: pd.DataFrame, positions: Sequence[int], first_line: int, text: TextLayout
) -> np.ndarray:
    lines = first_line + np.arange(len(chunk))
    if len(positions) > 1:
        cells = joined_columns(chunk, positions, COLUMN_SEPARATOR)
        # A row that lacks a part of its cell barcode has none.
        for position in positions:
            cells[(chunk[position].str.strip() == "").to_numpy(dtype=bool)] = ""
        _refuse_cellless(cells, lines, text.path, text)
        return cells
    return _stripped_cells(
        chunk[positions[0] if positions else 0].to_numpy(dtype=object),
        lines,
        text.path,
        text,
    )


def _barcode_texts(chunk: pd.DataFrame, positions: Sequence[int]) -> np.ndarray:
    """Each row's barcode, its parts joined; None where a part is missing."""
    parts = [
        _distinct_text(chunk[position].to_numpy(dtype=object)) for position in positions
    ]
    barcodes = parts[0]
    if len(parts) > 1:
        missing = np.zeros(len(barcodes), dtype=bool)
        for part in parts:
            missing |= np.asarray([value is None for value in part], dtype=bool)
        joined = pd.Series(parts[0], dtype=object).fillna("")
        for part in parts[1:]:
            joined = (
                joined + COLUMN_SEPARATOR + pd.Series(part, dtype=object).fillna("")
            )
        barcodes = joined.to_numpy(dtype=object, copy=True)
        barcodes[missing] = None
    return np.asarray(barcodes, dtype=object)


def _required_text(
    chunk: pd.DataFrame, position: int, name: str, first_line: int, text: TextLayout
) -> np.ndarray:
    """Parse a text column that every row needs, naming the lines that lack it."""
    values = _distinct_text(chunk[position].to_numpy(dtype=object))
    missing = np.flatnonzero(pd.isna(values))
    if len(missing):
        raise ValueError(
            f"{text.path} has rows without {name} on "
            + _on_lines(missing, first_line, text)
        )
    return values


def _parsed_counts(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse counts once per distinct value, as ``parse_integers`` parses them.

    Returns:
        Each value's count, -1 where missing; whether it is missing; whether
        it is no whole number of at least 0; and whether it is one too large
        for int64.
    """
    codes, distinct = pd.factorize(np.asarray(values, dtype=object))
    text = pd.Series(distinct, dtype=object).str.strip()
    missing = (text.isna() | text.isin(MISSING_TEXT)).to_numpy(dtype=bool)
    numbers = pd.to_numeric(text.where(~missing), errors="coerce").to_numpy(
        dtype=np.float64
    )
    bad = ~missing & (
        ~np.isfinite(numbers) | (numbers < 0) | (numbers != np.floor(numbers))
    )
    huge = ~missing & ~bad & (numbers >= _COUNT_LIMIT)
    counts = np.full(len(distinct), -1, dtype=np.int64)
    whole = ~missing & ~bad & ~huge
    counts[whole] = numbers[whole].astype(np.int64)
    return counts[codes], missing[codes], bad[codes], huge[codes]


def _item_counts(
    items: np.ndarray,
    origin: np.ndarray,
    name: str,
    first_line: int,
    text: TextLayout,
) -> np.ndarray:
    """Parse counts, of fields or list items, naming the file lines of bad ones.

    ``origin`` holds the row of the chunk that each item comes from.
    """
    counts, _missing, bad, huge = _parsed_counts(items)
    if bad.any():
        raise ValueError(
            f"{name} must be a whole number of at least 0, but holds "
            f"{str(items[np.flatnonzero(bad)[0]]).strip()!r} on "
            + _on_lines(sorted_unique(origin[bad]), first_line, text)
        )
    if huge.any():
        raise ValueError(
            f"{name} holds a number too large for int64, "
            f"{str(items[np.flatnonzero(huge)[0]]).strip()!r}, on "
            + _on_lines(sorted_unique(origin[huge]), first_line, text)
        )
    return counts


def _split(values: np.ndarray, pattern: str) -> list[list[str]]:
    return [
        [] if value is None else [item.strip() for item in re.split(pattern, value)]
        for value in values.tolist()
    ]


def _exploded_rows(
    layout: BarcodeLayout,
    first_line: int,
    cells: np.ndarray,
    barcodes: np.ndarray,
    counts: dict[str, np.ndarray],
    libraries: np.ndarray | None,
) -> BarcodeRows:
    """Split list fields into one row per item, keeping their counts aligned.

    A row without barcodes keeps one item without a barcode, so that its
    cell is still joined and counted.
    """
    text = layout._text
    assert text is not None
    pattern = "[" + re.escape(str(layout.separator)) + "]"
    listed = np.asarray([value is not None for value in barcodes], dtype=bool)
    barcode_items = [items or [""] for items in _split(barcodes, pattern)]
    lengths = np.asarray([len(items) for items in barcode_items], dtype=np.int64)
    origin = np.repeat(np.arange(len(barcodes)), lengths)
    flat = np.asarray([item for items in barcode_items for item in items], dtype=object)
    parsed: dict[str, np.ndarray] = {}
    for role, raw in counts.items():
        items = _split(parse_text(raw), pattern)
        unaligned = np.flatnonzero(
            listed
            & (np.asarray([len(entry) for entry in items], dtype=np.int64) != lengths)
        )
        if len(unaligned):
            raise ValueError(
                f"{layout.path} lists another number of {role} than of barcodes on "
                + _on_lines(unaligned, first_line, text)
            )
        values = np.asarray(
            [
                item
                for entry, length, present in zip(
                    items, lengths.tolist(), listed.tolist(), strict=True
                )
                for item in (entry if present else [""] * length)
            ],
            dtype=object,
        )
        parsed[role] = _item_counts(values, origin, role, first_line, text)
    flat[np.asarray([item in MISSING_TEXT for item in flat.tolist()], dtype=bool)] = (
        None
    )
    return BarcodeRows(
        first_line=first_line,
        cells=cells[origin],
        barcodes=flat,
        umi_counts=parsed.get("umis"),
        reads=parsed.get("reads"),
        libraries=None if libraries is None else libraries[origin],
    )


def _matches_none(values: np.ndarray) -> np.ndarray:
    """Whether each value is not a nucleotide sequence, tested once per distinct value."""
    codes, distinct = pd.factorize(values)
    flags = np.asarray([not _is_nucleotides(value) for value in distinct], dtype=bool)
    return np.asarray(flags[codes], dtype=bool)


def _check_pycashier(
    cells: np.ndarray, umis: np.ndarray, first_line: int, text: TextLayout
) -> None:
    bad_cells = _matches_none(cells)
    bad_umis = _matches_none(umis)
    bad = np.flatnonzero(bad_cells | bad_umis)
    if len(bad):
        raise ValueError(
            f"pycashier's cell and umi fields must be nucleotide sequences, but "
            f"{text.path} holds {str(cells[bad[0]])!r} and {str(umis[bad[0]])!r} on "
            + _on_lines(bad, first_line, text)
            + "; a read name that holds '_' shifts the fields of its row"
        )


def _refuse_unfilled(
    layout: BarcodeLayout, column: str, missing: np.ndarray, first_line: int
) -> None:
    """Refuse rows that lack a value that the format fills in every row."""
    rows = np.flatnonzero(missing)
    if len(rows):
        assert layout._text is not None
        raise ValueError(
            f"{layout.path} has rows without {column} on "
            + _on_lines(rows, first_line, layout._text)
            + f", which format={layout.format!r} fills in every row; the file may "
            "be cut short"
        )


def _iter_table_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    text = layout._text
    assert text is not None
    index = {name: position for position, name in enumerate(text.names)}
    cell_positions = [index[name] for name in layout.cell]
    barcode_positions = [index[name] for name in layout.barcode]
    samples: set[str] = set()
    for first_line, chunk in text_chunks(text, chunk_rows):
        cells = _cell_texts(chunk, cell_positions, first_line, text)
        # A row cut short lacks the values that its tool writes in every row,
        # which is checked before the rules that such a row would break.
        for column in layout.required:
            values = _distinct_text(chunk[index[column]].to_numpy(dtype=object))
            _refuse_unfilled(layout, column, pd.isna(values), first_line)
        umis = None
        if layout.umi is not None:
            umis = _required_text(
                chunk, index[layout.umi], f"a UMI ({layout.umi})", first_line, text
            )
        if layout.format == "pycashier":
            assert umis is not None
            _check_pycashier(cells, umis, first_line, text)
        if layout.format == "fatemap" and "SampleNum" in index:
            samples.update(chunk[index["SampleNum"]].str.strip().unique().tolist())
            if len(samples) > 1:
                raise ValueError(
                    f"{layout.path} holds the samples "
                    + ", ".join(sorted(samples)[:5])
                    + ", whose equal cell barcodes would name one cell; read it with "
                    "format='molecules' and columns={'cell': ['SampleNum', "
                    "'cellID'], 'umi': 'UMI', 'barcode': 'BC50StarcodeD8'}, which "
                    "joins each cell's sample to its barcode"
                )
        libraries = None
        if layout.library_column is not None:
            libraries = _required_text(
                chunk,
                index[layout.library_column],
                f"a library ({layout.library_column})",
                first_line,
                text,
            )
        if layout.separator is not None:
            raw = {
                role: chunk[index[column]].to_numpy(dtype=object)
                for role, column in (("umis", layout.umis), ("reads", layout.reads))
                if column is not None
            }
            lists = parse_text(chunk[barcode_positions[0]].to_numpy(dtype=object))
            yield _exploded_rows(layout, first_line, cells, lists, raw, libraries)
            continue
        counts = {
            column: _item_counts(
                chunk[index[column]].to_numpy(dtype=object),
                np.arange(len(chunk)),
                column,
                first_line,
                text,
            )
            for column in (layout.umis, layout.reads)
            if column is not None
        }
        barcodes = _barcode_texts(chunk, barcode_positions)
        if layout.format == "quicat":
            # QuiCAT writes a cell without barcodes with neither a barcode nor
            # a count, and a barcode always with its count.
            _refuse_unfilled(
                layout,
                str(layout.umis),
                ~pd.isna(barcodes) & (counts[str(layout.umis)] < 0),
                first_line,
            )
        yield BarcodeRows(
            first_line=first_line,
            cells=cells,
            barcodes=barcodes,
            umis=umis,
            umi_counts=None if layout.umis is None else counts[layout.umis],
            reads=None if layout.reads is None else counts[layout.reads],
            libraries=libraries,
        )


def _text_as_zero(value: str) -> float:
    """Read a field of text as 0, so that numpy parses a row's counts beside it."""
    return 0.0


def _kept_as_zero(kept: list[str]) -> Callable[[str], float]:
    """Read a field of text as 0, keeping its text in ``kept``, row by row."""

    def convert(value: str) -> float:
        kept.append(value)
        return 0.0

    return convert


def _refuse_wide_lines(
    layout: BarcodeLayout,
    lines: list[str],
    numbers: np.ndarray,
    positions: np.ndarray,
    error: str,
) -> NoReturn:
    """Name the lines of a merged table whose fields are not a cell and counts."""
    text = layout._text
    assert text is not None
    rows = [split_fields(line.rstrip("\r\n"), text.delimiter) for line in lines]
    widths = np.asarray([len(fields) for fields in rows], dtype=np.int64)
    uneven = np.flatnonzero(widths != len(text.names))
    if len(uneven):
        raise ValueError(
            f"{layout.path} holds {widths[uneven[0]]} fields on "
            + file_lines(numbers[uneven], 0)
            + f", where its header names {len(text.names)} columns"
        )
    block = np.asarray(
        [[fields[position] for position in positions] for fields in rows], dtype=object
    )
    _counts, missing, bad, huge = _parsed_counts(block.reshape(-1))
    wrong = np.flatnonzero(missing | bad | huge)
    if len(wrong):
        row, column = divmod(int(wrong[0]), len(positions))
        raise ValueError(
            f"{layout.path} holds {str(block[row, column]).strip()!r} as the count "
            f"of barcode {text.names[positions[column]]!r} on "
            + file_lines(numbers[np.unique(wrong // len(positions))], 0)
            + "; every field of QuiCAT's merged table holds a count of UMIs, a "
            "whole number of at least 0 and below 2**63"
        )
    raise ValueError(f"{layout.path} is malformed: {error}")


def _iter_wide_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    """Read a merged table's nonzero counts.

    numpy parses each chunk in one pass in C, keeping each row's cell
    barcode as it goes; pandas works column by column, which costs a merged
    table of thousands of barcodes more than its parsing.
    """
    text = layout._text
    assert text is not None
    positions = np.asarray(
        [
            position
            for position, name in enumerate(text.names)
            if name not in _QUICAT_CELL_COLUMNS
        ],
        dtype=np.intp,
    )
    # The barcode of each column; the columns of each cell read as 0, so no
    # count names them.
    names = np.asarray(
        [None if name in _QUICAT_CELL_COLUMNS else name for name in text.names],
        dtype=object,
    )
    cell_position = text.names.index("cell")
    options: dict[str, Any] = {
        "delimiter": None if text.delimiter == " " else text.delimiter,
        "comments": None,
        "quotechar": '"',
    }
    for numbers, lines in line_chunks(text, chunk_rows):
        cells: list[str] = []
        converters: dict[int | str, Callable[[str], float]] = {
            position: _text_as_zero
            for position, name in enumerate(text.names)
            if name in _QUICAT_CELL_COLUMNS
        }
        converters[cell_position] = _kept_as_zero(cells)
        try:
            # Every field is parsed, so a row of another width raises.
            values = np.loadtxt(
                lines, dtype=np.float64, ndmin=2, converters=converters, **options
            )
        except ValueError as error:
            _refuse_wide_lines(layout, lines, numbers, positions, str(error))
        if (
            values.shape != (len(lines), len(text.names))
            or len(cells) != len(lines)
            or not np.all(np.isfinite(values))
            or not np.all(values >= 0)
            or not np.all(values < _COUNT_LIMIT)
            or not np.all(values == np.floor(values))
        ):
            _refuse_wide_lines(
                layout, lines, numbers, positions, "its rows are not a cell and counts"
            )
        rows, columns = np.nonzero(values > 0)
        yield BarcodeRows(
            first_line=int(numbers[0]),
            cells=_stripped_cells(
                np.asarray(cells, dtype=object), numbers, layout.path
            )[rows],
            barcodes=names[columns],
            umi_counts=values[rows, columns].astype(np.int64),
        )


def _whole_counts(values: np.ndarray, path: str) -> np.ndarray:
    numbers = np.asarray(values, dtype=np.float64)
    if np.any(
        ~np.isfinite(numbers)
        | (numbers < 0)
        | (numbers >= _COUNT_LIMIT)
        | (numbers != np.floor(numbers))
    ):
        raise ValueError(
            f"The counts in X of {path} must be whole numbers of at least 0 and "
            "below 2**63"
        )
    return numbers.astype(np.int64)


def _decoded_names(node: Any, path: str) -> np.ndarray:
    """Decode an AnnData index of names into one array, a block at a time."""
    names = np.empty(int(column_length(node) or 0), dtype=object)
    start = 0
    for block in _name_blocks(node, path):
        names[start : start + len(block)] = block
        start += len(block)
    return names


def _library_codes(
    handle: h5py.File, obs: h5py.Group, column: str, n_cells: int, path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Code each cell's library: a code per cell and the label of each code."""
    codes = np.empty(n_cells, dtype=np.int32)
    lookup: dict[str, int] = {}
    start = 0
    for labels in _library_blocks(handle, obs, column, n_cells, path):
        codes[start : start + len(labels)] = [
            lookup.setdefault(label, len(lookup)) for label in labels
        ]
        start += len(labels)
    return codes, np.asarray(list(lookup), dtype=object)


def _dense_entries(
    node: h5py.Dataset, matrix: _Matrix, entries: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield the rows, columns and values of a dense matrix's nonzero entries.

    The matrix is read in tiles of whole HDF5 chunks, and each tile's
    entries are searched ``entries`` at a time; the nonzero ones gather
    across searches and tiles, and each run yields at most ``entries``.
    """
    height, width = matrix.tile_shape(entries)
    gathered: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_gathered = 0
    for top in range(0, matrix.n_cells, height):
        for left in range(0, matrix.n_barcodes, width):
            tile = np.asarray(
                node[
                    top : min(top + height, matrix.n_cells),
                    left : min(left + width, matrix.n_barcodes),
                ]
            )
            flat = tile.reshape(-1)
            for begin in range(0, len(flat), entries):
                found = np.flatnonzero(flat[begin : begin + entries]) + begin
                if not len(found):
                    continue
                if n_gathered + len(found) > entries:
                    yield _joined_entries(gathered)
                    gathered, n_gathered = [], 0
                rows, columns = np.divmod(found, tile.shape[1])
                gathered.append((rows + top, columns + left, flat[found]))
                n_gathered += len(found)
    if gathered:
        yield _joined_entries(gathered)


def _joined_entries(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns, values = zip(*parts, strict=True)
    return np.concatenate(rows), np.concatenate(columns), np.concatenate(values)


def _sparse_entries(
    node: h5py.Group, matrix: _Matrix, entries: int, path: str
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield the rows, columns and values of a sparse matrix's stored entries.

    Stored entries are read ``entries`` at a time, whatever rows or columns
    they belong to, so a barcode of many cells is read in parts.
    """
    pointers = np.asarray(node["indptr"][:], dtype=np.int64)
    n_major, n_minor = (
        (matrix.n_cells, matrix.n_barcodes)
        if matrix.encoding == "csr"
        else (matrix.n_barcodes, matrix.n_cells)
    )
    n_stored = int(pointers[-1]) if len(pointers) else 0
    if (
        len(pointers) != n_major + 1
        or pointers[0] != 0
        or np.any(np.diff(pointers) < 0)
        or n_stored > min(node["indices"].shape[0], node["data"].shape[0])
    ):
        raise ValueError(f"X of {path} has pointers that do not fit its entries")
    for first in range(0, n_stored, entries):
        last = min(first + entries, n_stored)
        minor = np.asarray(node["indices"][first:last], dtype=np.int64)
        if len(minor) and (minor.min() < 0 or minor.max() >= n_minor):
            raise ValueError(f"X of {path} holds entries beyond its shape")
        majors = np.searchsorted(pointers, np.arange(first, last), side="right") - 1
        rows, columns = (majors, minor) if matrix.encoding == "csr" else (minor, majors)
        yield rows, columns, np.asarray(node["data"][first:last])


def _iter_h5ad_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    """Yield the nonzero entries of an AnnData file's ``X``, ``chunk_rows`` at most.

    A dense matrix is read in tiles of whole HDF5 chunks, and a sparse one
    ``chunk_rows`` stored entries at a time. The names of the cells and
    barcodes are decoded once.
    """
    matrix = layout._matrix
    assert matrix is not None
    counts_role = "umis" if layout.umis is not None else "reads"
    entries = max(1, int(chunk_rows))
    with h5py.File(layout.path, "r") as handle:
        obs, var = handle["obs"], handle["var"]
        cell_names = _decoded_names(_names(obs, layout.path, "cell"), layout.path)
        barcode_names = _decoded_names(_names(var, layout.path, "barcode"), layout.path)
        node = handle["X"]
        library_codes = library_names = None
        if layout.library_column is not None:
            library_codes, library_names = _library_codes(
                handle, obs, layout.library_column, len(cell_names), layout.path
            )
        found = (
            _dense_entries(node, matrix, entries)
            if matrix.encoding == "dense"
            else _sparse_entries(node, matrix, entries, layout.path)
        )
        for rows, columns, values in found:
            stored = values != 0
            rows, columns, values = rows[stored], columns[stored], values[stored]
            counts = _whole_counts(values, layout.path)
            if not len(rows):
                continue
            yield BarcodeRows(
                first_line=int(rows.min()) + 1,
                cells=cell_names[rows],
                barcodes=barcode_names[columns],
                umi_counts=counts if counts_role == "umis" else None,
                reads=counts if counts_role == "reads" else None,
                libraries=None
                if library_codes is None or library_names is None
                else library_names[library_codes[rows]],
            )


def _iter_molecule_rows(
    layout: BarcodeLayout, chunk_rows: int
) -> Iterator[BarcodeRows]:
    molecules = layout._molecules
    assert molecules is not None
    for rows in iter_molecule_info(molecules, chunk_rows=chunk_rows):
        yield BarcodeRows(
            first_line=rows.first_row + 1,
            cells=rows.barcodes,
            barcodes=rows.features,
            umis=rows.umis,
            reads=rows.reads,
        )


def iter_barcode_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    """Yield the rows of a barcode file in chunks of at most ``chunk_rows`` file rows.

    A pair's list fields yield one row per item. A row without a barcode,
    such as an empty list or ``NA``, has the barcode None.

    Raises:
        ValueError: If a row lacks its cell barcode, UMI or library, a count
            is not a whole number of at least 0, a list's counts do not align
            with its barcodes, the file breaks a rule of its format, or it is
            malformed.
    """
    rows = max(1, int(chunk_rows))
    if layout.format == "molecule_info":
        yield from _iter_molecule_rows(layout, rows)
    elif layout.format == "h5ad":
        yield from _iter_h5ad_rows(layout, rows)
    elif layout.format == "quicat_wide":
        yield from _iter_wide_rows(layout, rows)
    else:
        yield from _iter_table_rows(layout, rows)
