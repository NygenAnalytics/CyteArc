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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

from ..metadata.records import MISSING_TEXT, file_lines, parse_integers, parse_text
from ._delimited import (
    HDF5_SIGNATURE,
    TextLayout,
    compression,
    first_lines,
    joined_columns,
    line_list,
    require_columns,
    signature,
    text_chunks,
    text_layout,
)
from ._h5ad_columns import index_key, read_table_column, sparse_encoding, sparse_shape
from ._text import as_text
from .cellranger import (
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
    _text: TextLayout | None = field(default=None, repr=False)
    _molecules: MoleculeInfoLibrary | None = field(default=None, repr=False)

    @property
    def n_columns(self) -> int:
        """The fields of each row that a chunk holds, which sizes chunks."""
        return len(self._text.names) if self._text is not None else 6

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
    if _CLONE_COLUMNS & set(text.names):
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
    if text.row_names and text.names[1:] == ("barcode", "bc.umi.count"):
        matches.append("bartab_annotation")
    if len(matches) > 1:
        raise ValueError(
            f"{text.path} has the columns of several barcode tables: "
            + ", ".join(matches)
            + "; pass format= to choose one"
        )
    return matches[0] if matches else None


def _is_nucleotides(value: str) -> bool:
    return bool(_NUCLEOTIDES.match(value))


def _headerless_format(path: str, found: str | None) -> str | None:
    """Recognise a table without a header by the fields of its first line."""
    # The caller read the file's header, so it has a first line.
    fields = first_lines(path, found, 1)[0].split("\t")
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
        )
    if format == "pycashier":
        require_columns(text, _PYCASHIER_COLUMNS)
        return preset("reads", cell=("cell",), barcode=("barcode",), umi="umi")
    if format == "celltag":
        require_columns(text, ["Cell.BC", "Cell.Tag"])
        return preset(
            "reads",
            cell=("Cell.BC",),
            barcode=("Cell.Tag",),
            umi="UMI" if "UMI" in names else None,
        )
    if format == "fatemap":
        require_columns(text, ["cellID", "UMI", "BC50StarcodeD8"])
        return preset(
            "molecules",
            cell=("cellID",),
            barcode=("BC50StarcodeD8",),
            umi="UMI",
        )
    if format == "quicat":
        require_columns(text, ["cell", "barcode", "counts"])
        return preset("pairs", cell=("cell",), barcode=("barcode",), umis="counts")
    if format == "quicat_wide":
        require_columns(text, ["cell"])
        barcodes = [name for name in text.names if name not in _QUICAT_CELL_COLUMNS]
        if not barcodes:
            raise ValueError(f"{text.path} has no barcode columns")
        require_columns(text, barcodes)
        return preset("pairs", cell=("cell",))
    if format == "bartab":
        require_columns(text, ["gene", "cell", "count"])
        return preset("pairs", cell=("cell",), barcode=("gene",), umis="count")
    if not text.row_names:
        raise ValueError(
            f"{text.path} is not in R's row-name layout, as BARtab writes its "
            "barcode annotation: one header field fewer than each row"
        )
    require_columns(text, ["barcode", "bc.umi.count"])
    return preset(
        "pairs",
        barcode=("barcode",),
        umis="bc.umi.count",
        separator=";,",
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
    return BarcodeLayout(
        format="h5ad",
        path=path,
        level="pairs",
        library=library,
        umis="X" if not counts or counts[0] == "umis" else None,
        reads="X" if counts and counts[0] == "reads" else None,
        library_column=library_column,
    )


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
    if format is None:
        text = text_layout(name, what="a barcode table")
        detected = _detected_text_format(text)
        if detected is None:
            headerless = _headerless_format(name, found)
            if headerless is None:
                shown = ", ".join(repr(column) for column in text.names if column)
                raise ValueError(
                    f"{name} has the columns {shown}, which match no barcode table "
                    "that CyteArc knows" + _hint(text)
                )
            text = _headerless_layout(name, found, headerless)
            detected = headerless
        return _preset_layout(detected, text, file_library)
    if format in ("pycashier", "fatemap"):
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


def _cell_texts(
    chunk: pd.DataFrame, positions: Sequence[int], first_line: int, path: str
) -> np.ndarray:
    if len(positions) > 1:
        cells = joined_columns(chunk, positions, COLUMN_SEPARATOR)
    else:
        # A missing-value token is a cell barcode as any other text is.
        codes, distinct = pd.factorize(
            chunk[positions[0] if positions else 0].to_numpy(dtype=object)
        )
        stripped = np.asarray([value.strip() for value in distinct], dtype=object)
        cells = stripped[codes]
    empty = np.flatnonzero(cells == "")
    if len(empty):
        raise ValueError(
            f"{path} has rows without a cell barcode on lines "
            f"{line_list(empty, first_line)}"
        )
    return cells


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
    chunk: pd.DataFrame, position: int, name: str, first_line: int, path: str
) -> np.ndarray:
    """Parse a text column that every row needs, naming the lines that lack it."""
    values = _distinct_text(chunk[position].to_numpy(dtype=object))
    missing = np.flatnonzero(pd.isna(values))
    if len(missing):
        raise ValueError(
            f"{path} has rows without {name} on {file_lines(missing, first_line)}"
        )
    return values


def _item_counts(
    items: np.ndarray, origin: np.ndarray, name: str, first_line: int
) -> np.ndarray:
    """Parse the counts of list items, naming the file lines of bad ones."""
    text = pd.Series(items, dtype=object).str.strip()
    missing = (text.isna() | text.isin(MISSING_TEXT)).to_numpy(dtype=bool)
    numbers = pd.to_numeric(text.where(~missing), errors="coerce").to_numpy(
        dtype=np.float64
    )
    bad = ~missing & (
        ~np.isfinite(numbers) | (numbers < 0) | (numbers != np.floor(numbers))
    )
    if bad.any():
        rows = origin[bad]
        raise ValueError(
            f"{name} must be a whole number of at least 0, but holds "
            f"{text[bad].iloc[0]!r} on " + file_lines(rows, first_line)
        )
    counts = np.full(len(numbers), -1, dtype=np.int64)
    counts[~missing] = numbers[~missing].astype(np.int64)
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
                + file_lines(unaligned, first_line)
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
        parsed[role] = _item_counts(values, origin, role, first_line)
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
    cells: np.ndarray, umis: np.ndarray, first_line: int, path: str
) -> None:
    bad_cells = _matches_none(cells)
    bad_umis = _matches_none(umis)
    bad = np.flatnonzero(bad_cells | bad_umis)
    if len(bad):
        raise ValueError(
            f"pycashier's cell and umi fields must be nucleotide sequences, but "
            f"{path} holds {str(cells[bad[0]])!r} and {str(umis[bad[0]])!r} on "
            + file_lines(bad, first_line)
            + "; a read name that holds '_' shifts the fields of its row"
        )


def _iter_table_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    text = layout._text
    assert text is not None
    index = {name: position for position, name in enumerate(text.names)}
    cell_positions = [index[name] for name in layout.cell]
    barcode_positions = [index[name] for name in layout.barcode]
    samples: set[str] = set()
    for first_line, chunk in text_chunks(text, chunk_rows):
        cells = _cell_texts(chunk, cell_positions, first_line, layout.path)
        umis = None
        if layout.umi is not None:
            umis = _required_text(
                chunk,
                index[layout.umi],
                f"a UMI ({layout.umi})",
                first_line,
                layout.path,
            )
        if layout.format == "pycashier":
            assert umis is not None
            _check_pycashier(cells, umis, first_line, layout.path)
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
                layout.path,
            )
        if layout.separator is not None:
            raw = {
                role: chunk[index[column]].to_numpy(dtype=object)
                for role, column in (("umis", layout.umis), ("reads", layout.reads))
                if column is not None
            }
            yield _exploded_rows(
                layout,
                first_line,
                cells,
                parse_text(chunk[barcode_positions[0]].to_numpy(dtype=object)),
                raw,
                libraries,
            )
            continue
        counts = {
            role: parse_integers(
                chunk[index[column]].to_numpy(dtype=object),
                name=column,
                first_line=first_line,
            )
            for role, column in (("umis", layout.umis), ("reads", layout.reads))
            if column is not None
        }
        yield BarcodeRows(
            first_line=first_line,
            cells=cells,
            barcodes=_barcode_texts(chunk, barcode_positions),
            umis=umis,
            umi_counts=counts.get("umis"),
            reads=counts.get("reads"),
            libraries=libraries,
        )


def _iter_wide_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    text = layout._text
    assert text is not None
    positions = [
        position
        for position, name in enumerate(text.names)
        if name not in _QUICAT_CELL_COLUMNS
    ]
    names = np.asarray([text.names[position] for position in positions], dtype=object)
    cell_position = text.names.index("cell")
    for first_line, chunk in text_chunks(text, chunk_rows):
        cells = _cell_texts(chunk, [cell_position], first_line, layout.path)
        counts = np.column_stack(
            [
                parse_integers(
                    chunk[position].to_numpy(dtype=object),
                    name=text.names[position],
                    first_line=first_line,
                )
                for position in positions
            ]
        )
        rows, columns = np.nonzero(counts > 0)
        yield BarcodeRows(
            first_line=first_line,
            cells=cells[rows],
            barcodes=names[columns],
            umi_counts=counts[rows, columns],
        )


def _whole_counts(values: np.ndarray, path: str) -> np.ndarray:
    numbers = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(numbers) | (numbers < 0) | (numbers != np.floor(numbers))):
        raise ValueError(
            f"The counts in X of {path} must be whole numbers of at least 0"
        )
    return numbers.astype(np.int64)


def _decoded_names(dataset: Any) -> np.ndarray:
    return np.asarray(
        [as_text(value) for value in np.asarray(dataset).tolist()], dtype=object
    )


def _iter_h5ad_rows(layout: BarcodeLayout, chunk_rows: int) -> Iterator[BarcodeRows]:
    counts_role = "umis" if layout.umis is not None else "reads"
    with h5py.File(layout.path, "r") as handle:
        obs, var = handle["obs"], handle["var"]
        cell_names = obs[str(index_key(obs))]
        barcode_names = _decoded_names(var[str(index_key(var))][:])
        n_cells, n_barcodes = int(cell_names.shape[0]), len(barcode_names)
        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            encoding, shape = "dense", tuple(int(size) for size in node.shape)
        else:
            encoding, shape = str(sparse_encoding(node)), sparse_shape(node) or ()
        if encoding not in ("dense", "csr", "csc") or len(shape) != 2:
            raise ValueError(
                f"X of {layout.path} is neither a dense nor a sparse matrix"
            )
        if shape != (n_cells, n_barcodes):
            raise ValueError(
                f"X of {layout.path} is {shape[0]} by {shape[1]}, but the file names "
                f"{n_cells} cells and {n_barcodes} barcodes"
            )
        major = n_barcodes if encoding == "csc" else n_cells
        for start in range(0, major, chunk_rows):
            stop = min(start + chunk_rows, major)
            if encoding == "dense":
                block = np.asarray(node[start:stop])
                rows, columns = np.nonzero(block)
                values = block[rows, columns]
                rows = rows + start
            else:
                pointers = np.asarray(node["indptr"][start : stop + 1], dtype=np.int64)
                first, last = int(pointers[0]), int(pointers[-1])
                minor = np.asarray(node["indices"][first:last], dtype=np.int64)
                values = np.asarray(node["data"][first:last])
                majors = np.repeat(np.arange(start, stop), np.diff(pointers))
                rows, columns = (
                    (majors, minor) if encoding == "csr" else (minor, majors)
                )
                stored = values != 0
                rows, columns, values = rows[stored], columns[stored], values[stored]
            counts = _whole_counts(values, layout.path)
            if not len(rows):
                continue
            low, high = int(rows.min()), int(rows.max())
            cells = _decoded_names(cell_names[low : high + 1])[rows - low]
            libraries = None
            if layout.library_column is not None:
                labels, missing = read_table_column(
                    handle, obs, layout.library_column, ("__categories",), low, high + 1
                )
                texts = np.asarray(
                    [as_text(value) for value in np.asarray(labels).tolist()],
                    dtype=object,
                )
                if np.any(np.asarray(missing, dtype=bool)):
                    raise ValueError(
                        f"The obs column {layout.library_column!r} of {layout.path} "
                        "lacks the library of some cells"
                    )
                libraries = texts[rows - low]
            yield BarcodeRows(
                first_line=start + 1,
                cells=cells,
                barcodes=barcode_names[columns],
                umi_counts=counts if counts_role == "umis" else None,
                reads=counts if counts_role == "reads" else None,
                libraries=libraries,
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
