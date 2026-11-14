"""Read the allele tables that lineage recorder pipelines write.

``allele_layout`` recognizes a file's format from its contents, resolves the
columns that hold each row's cell, integration, sites and counts, and gives
the format's tokens for each state of a site (:class:`SiteRules`);
``iter_allele_rows`` then yields the rows in bounded chunks and
``classified_rows`` the states of their sites. Joining cells belongs to the
caller. Every format reads text as written, with pandas' conversion of
missing values off, so Cassiopeia's ``None``, which means uncut, stays text.

Formats, with what each token means:

- ``"carlin"``: CARLIN's ``AlleleAnnotations.txt`` and ``AlleleColonies.txt``,
  as CARLIN and Custom_CARLIN write them for DARLIN: line i of the first is
  an allele in CARLIN's notation (``23_268del,55_248del``), line i of the
  second the cell barcodes that carry it, comma-joined. ``[]`` is unedited,
  ``X`` and ``?`` an unknown allele. One file names the pair, or the folder
  that holds it; the files must have equal lines, list each barcode once,
  and come from a single-cell run, as ``Results.txt`` beside them says, or
  by barcodes of at least 14 bases where it is absent: a bulk run lists
  UMIs instead of cells and is refused.
- ``"darlin"``: the ``darlin`` CLI's ``step4_final.tsv``, ``n_UMIs``, ``CR``,
  ``mutation``, ``aligned_LR`` and ``aligned_ref``, one row per cell and
  lineage barcode; the reads of each row are summed over its molecules in
  ``step4_annotated.tsv`` beside it, where that file exists. ``[]`` and, as
  darlin-core before 1.1.0 wrote it, an empty mutation are unedited. No
  output of the CLI is public, so this follows its documented columns.
- ``"cassiopeia"``: Cassiopeia's allele tables before 3.0, ``cellBC``,
  ``intBC``, a column ``r1`` to ``rN`` per cut site, ``UMI`` and
  ``readCount``, with the extra columns of published tables. A site that
  holds ``None`` anywhere, such as ``GCCGA[None]AAAAT``, is uncut; empty,
  ``NaN``, ``nan``, ``NA``, ``NC`` and ``Missing`` are missing.
- ``"cassiopeia_parquet"``: Cassiopeia 3.0's ``<name>.alleles.parquet``,
  ``cellBC``, ``intBC``, its site columns, ``n_umis`` and ``n_reads``; only
  rows whose ``pass`` holds true are read where the table has ``pass``.
  ``None`` and text holding ``[None]`` are uncut, and a null is missing.
  Reading parquet needs polars.
- ``"petracer"``: PEtracer's long tables of 2025, ``intID``, ``cellBC``,
  ``RNF2``, ``HEK3``, ``EMX1``, ``UMI`` and ``readCount``. A site holds the
  inserted 5-mer: one of its eight lineage marks, or another insertion,
  which is an unknown mark. ``None`` is unedited, as is ``CTTGGG`` at
  ``EMX1``, the sequence an unedited cassette reads there, and an empty
  site was not read.
- ``"petracer_characters"``: PEtracer's character matrices of 2026, a cell
  column, then ``intID<N>-<site>`` columns: ``*`` unedited, ``-`` missing,
  ``1`` to ``8`` a lineage mark, ``!`` another insertion and an empty field
  an integration that the cell's embryo lacks.
- ``"typewriter"``: DNA Typewriter's table of 2022, ``Cell``, ``TargetBC``,
  ``Site1`` to ``Site5`` and ``nUMI``: a slot holds a 6-base insertion and
  is blank when unedited. Rows may hold more slot fields than the header,
  as 16 rows of the published table do, if those fields are empty.
- ``"typewriter_chains"``: DNA Typewriter's per-cell tables of 2026, a
  ``cell`` or ``cell_id`` column and one column per 12-base integration
  barcode, beside ``n_loci``, ``n_doublet_loci``, ``mean_dominance`` and
  ``pass_qc``, which are not read. A value is a chain of the symbols
  written, joined by ``-``, the slots after it unedited, ``-`` for no edit
  and empty for no call; or six slots joined by ``|``, with ``U`` unedited
  and ``NA`` no call.
- ``"treedata"``: a TreeData ``.h5td`` file or ``.zarr`` store's
  ``obsm['characters']``, with the states that ``uns['missing_state']`` and
  ``uns['unmodified_state']`` name, or, where the file names none,
  Cassiopeia's: ``-1``, ``-`` and ``NA`` missing, ``0`` and ``*`` unedited.
- ``"laml"``: a delimited character matrix, as LAML reads it and
  ``export_characters(format="laml")`` writes it, and as Cassiopeia's
  ``to_csv`` and ConvexML write it: a cell column, then one column per
  character: ``0`` and ``*`` unedited, ``-1``, ``-``, ``?`` and ``NA``
  missing, other text a state.
- ``"table"``: any table, read by the columns that ``columns`` names.

Ordered slots follow the prefix rule: a slot before an edited slot was
edited, so one that reads as unedited there was not read and is missing,
and a slot at or beyond its integration's length is not applicable. An
integration has the recorder's slots, five for DNA Typewriter's table of
2022 and six for its tables of 2026, unless the caller names fewer for it.
Column names of character matrices split into an integration and a site on
their last hyphen; a name without one is an integration of one site.
"""

import csv
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..clones.barcodes import pair_keys, repeated_keys
from ..clones.recorders import (
    MISSING,
    SITE_KINDS,
    AlleleRows,
    ChunkStates,
    RefusedTokens,
    SiteRules,
    chunk_states,
)
from ..metadata.records import MISSING_TEXT, file_lines, parse_integers
from ..utils.files import HDF5_SIGNATURE, file_signature, open_text
from ..utils.delimited import (
    TextLayout,
    delimiter,
    first_lines,
    line_chunks,
    require_columns,
    rows_end_in_delimiter,
    split_fields,
    text_chunks,
    text_errors,
    text_layout,
)
from ._treedata import CharacterTable, character_table, is_zarr_store
from ..utils.arguments import quoted_value

__all__ = [
    "ALLELE_FORMATS",
    "AlleleLayout",
    "allele_file",
    "allele_layout",
    "classified_rows",
    "iter_allele_rows",
]

#: The values of ``format``.
ALLELE_FORMATS: tuple[str, ...] = (
    "carlin",
    "darlin",
    "cassiopeia",
    "cassiopeia_parquet",
    "petracer",
    "petracer_characters",
    "typewriter",
    "typewriter_chains",
    "treedata",
    "laml",
    "table",
)
# The formats whose files hold one integration, which the caller names.
_ONE_INTEGRATION = ("carlin", "darlin")
# The formats of ordered slots, whose shorter integrations the caller names.
_ORDERED = ("typewriter", "typewriter_chains")
# The formats that number each row by its line in the file, which the
# records keep and errors name: the header is line 1 of a table, and a CARLIN
# pair's row takes the line of its allele, the same in both files. Parquet
# tables and TreeData files number their rows by position from 1.
_FILE_LINES = frozenset(
    {
        "carlin",
        "darlin",
        "cassiopeia",
        "petracer",
        "petracer_characters",
        "laml",
        "typewriter",
        "typewriter_chains",
        "table",
    }
)
_PARQUET_SIGNATURE = b"PAR1"
# CARLIN's notation: deletions, insertions, deletion-insertions and
# single-base substitutions on the array, joined by commas.
_CARLIN_EVENT = (
    r"(?:\d+_\d+del|\d+_\d+ins[ACGTN]+|\d+_\d+delins[ACGTN]+|\d+[ACGTN]>[ACGTN])"
)
_CARLIN_ALLELE = rf"{_CARLIN_EVENT}(?:,{_CARLIN_EVENT})*"
# PEtracer's and DNA Typewriter's insertions are bases. An insertion that
# holds N, an ambiguous base, names no allele, so it is read as missing, as is
# NA; other text is no insertion.
_INSERTION = r"[ACGT]+"
_AMBIGUOUS_INSERTION = r"[ACGTN]*N[ACGTN]*"
# What Cassiopeia 3.0 writes at a site of known insertions for an edit that its
# lookup does not name: the inserted sequence, or a deletion's length after D,
# with the inserted sequence after I where there is one.
_LOOKUP_EDIT = r"[ACGT]+|D\d+(?:I[ACGTN]+)?"
# PEtracer's character matrices of 2026: "*" unedited, "-" missing, a mark
# from 1 to 8, "!" an insertion outside the lookup and nothing for an
# integration that the cell's embryo lacks.
_PETRACER_CHARACTER_RULES = SiteRules(
    kind="independent",
    not_applicable=frozenset({""}),
    missing=frozenset({"-"}),
    unedited=frozenset({"*"}),
    unknown=frozenset({"!"}),
    marks=frozenset(str(mark) for mark in range(1, 9)),
)
# A state of a character matrix: text that pandas or other formats read as
# missing, None among it, and a negative number that the matrix does not name
# missing, are none; the format's own tokens are read before this applies.
_MATRIX_STATE = (
    r"(?!(?:"
    + "|".join(re.escape(token) for token in sorted(MISSING_TEXT) if token)
    + r"|-\d+)$)\S(?:.*\S)?"
)
# A whole number written with a fraction of zeros, as pandas writes the
# integers of a column that holds a missing value.
_WHOLE_FLOAT = re.compile(r"(-?\d+)\.0+")
_NUCLEOTIDE_NAME = re.compile(r"^[ACGTN]+$")
# PEtracer's eight lineage marks at each site, the 5-mers that Koblan, Yost
# et al. (Science 2025) insert, as Cassiopeia's PEtracer preset lists them.
_PETRACER_MARKS = MappingProxyType(
    {
        "RNF2": frozenset(
            {"ACAGT", "ACTTA", "TTCCT", "TATAT", "GTTCA", "TGCCA", "TCCAA", "ACTCC"}
        ),
        "HEK3": frozenset(
            {"GATAG", "AATCG", "GCAAG", "GCGCC", "CTTTG", "ATCAA", "CTCTC", "ATTTA"}
        ),
        "EMX1": frozenset(
            {"GGACA", "ACAAT", "CCCTA", "AGTAC", "CCGAT", "CCTTT", "ATCAA", "ATTCG"}
        ),
    }
)
_PETRACER_SITES = ("RNF2", "HEK3", "EMX1")
_PETRACER_COLUMN = re.compile(r"^.+-(?:RNF2|HEK3|EMX1)$")
# DNA Typewriter's slots per tape in 2026; its table of 2022 names five in
# its header.
_TYPEWRITER_CHAIN_SLOTS = 6
_TYPEWRITER_QC_COLUMNS = frozenset(
    {"n_loci", "n_doublet_loci", "mean_dominance", "pass_qc"}
)
_TYPEWRITER_CELL_COLUMNS = ("cell", "cell_id")
# Cassiopeia's columns besides its sites, before 3.0 and in 3.0.
_CASSIOPEIA_SITE = re.compile(r"^r\d+$")
# The schema key under which pandas saves a table's attrs in parquet.
_PANDAS_ATTRS = "PANDAS_ATTRS"
_PARQUET_RESERVED = frozenset(
    {
        "sample",
        "cellBC",
        "intBC",
        "UMI",
        "clone",
        "n_reads",
        "n_umis",
        "allele_frac",
        "pass",
    }
)
# A CARLIN run lists cell barcodes of at least this many bases: 16 for 10x
# Genomics, 16 to 19 for inDrops; bulk runs list UMIs of 10 or 12.
_MIN_CARLIN_BARCODE = 14
# The roles of columns that a table read through ``columns`` may name.
_TABLE_ROLES = (
    "cell",
    "integration",
    "sites",
    "site",
    "allele",
    "separator",
    "umis",
    "reads",
    "kind",
    "slots",
    "unedited",
    "missing",
    "not_applicable",
    "unknown",
)
_TOKEN_ROLES = ("unedited", "missing", "not_applicable", "unknown")
# A chunk holds at most this many rows of a file.
_MAX_CHUNK_ROWS = 50_000
# Bytes that a chunk holds per field of a row beside the field's text: the
# references and headers of its raw text, its stripped copy and its token.
_FIELD_BYTES = 100
# Fields that the csv module and polars read hold a list entry or an Arrow
# value beside.
_LIST_FIELD_BYTES = 150
_PARQUET_FIELD_BYTES = 300
# Each byte of a row's text is held as raw text, stripped text and a token,
# and a row may be twice as long as the longest of the first lines.
_TEXT_COPIES = 4
_SAMPLED_LINES = 1_000
# Bytes that each record of a chunk takes in flight: its state and allele,
# the columns of the records built from them, and their cells.
_RECORD_BYTES = 100
# A CARLIN pair's row is a cell barcode, held in lists, then as text, its
# line number, its key and its record.
_CARLIN_ROW_BYTES = 400
# Bytes per molecule row of step4_annotated.tsv while its reads are summed:
# its key, its reads, and their sorted copies.
_READS_KEY_BYTES = 80
# Bytes that a CARLIN annotation line and a barcode's key take while held.
_ANNOTATION_BYTES = 120
_BARCODE_KEY_BYTES = 24


@dataclass(frozen=True, slots=True)
class AlleleLayout:
    """How to read the alleles of one file.

    Attributes:
        format: One of ``ALLELE_FORMATS``.
        path: The file that names the cells, such as a CARLIN pair's
            ``AlleleColonies.txt``.
        tool: The tool that wrote the file, such as ``"CARLIN"``.
        pipeline: The pipeline whose notation the alleles follow, which an
            allele bank must share, such as ``"CARLIN (MATLAB)"``.
        rules: The format's tokens for each state.
        integration: The one integration of a file that holds one, or None.
        lengths: For ordered slots, the slots of integrations shorter than
            ``n_slots``.
        n_slots: For ordered slots, the slots of the recorder's arrays.
        rows_hold_sites: Whether each row holds one site of its cell and
            integration, so that the rows of one cell and integration are
            one allele.
        companions: The other files read, as ``(role, path)`` pairs.
        columns: The columns read, by role, as the identity records them.
        n_columns: The fields of each row that reading parses.
        n_sites: The sites that each row gives, its records.
        field_bytes: The bytes that each field of a row holds beside its
            text while it is read.
        line_bytes: The bytes of a row's text that a chunk is sized for:
            twice the longest of the file's first lines.
    """

    format: str
    path: str
    tool: str
    pipeline: str
    rules: SiteRules
    integration: str | None = None
    lengths: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    n_slots: int | None = None
    rows_hold_sites: bool = False
    companions: tuple[tuple[str, str], ...] = ()
    columns: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    n_columns: int = 1
    n_sites: int = 1
    field_bytes: int = _FIELD_BYTES
    line_bytes: int = 0
    _text: TextLayout | None = field(default=None, repr=False)
    _table: CharacterTable | None = field(default=None, repr=False)
    _read: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )

    def _row_bytes(self) -> int:
        return (
            self.field_bytes * self.n_columns
            + _TEXT_COPIES * self.line_bytes
            + _RECORD_BYTES * self.n_sites
        )

    def chunk_rows(self, chunk_bytes: int) -> int:
        """The rows of a chunk that holds at most about ``chunk_bytes``."""
        return max(1, min(_MAX_CHUNK_ROWS, int(chunk_bytes) // self._row_bytes()))

    def chunk_bytes(self, chunk_rows: int) -> int:
        """The bytes that a chunk of ``chunk_rows`` rows holds at most, about."""
        held = self._row_bytes() * int(chunk_rows)
        if self._table is not None:
            held += self._table.block_bytes(chunk_rows)
        return held

    def held_bytes(self) -> int:
        """The bytes that reading holds beside its chunks.

        A CARLIN pair holds its annotation lines and the keys of its
        barcodes, which find a barcode listed twice; a ``darlin`` table the
        reads of each molecule key of ``step4_annotated.tsv``, as they are
        summed; a TreeData file the names of its characters.
        """
        held = 0
        if self.format == "carlin":
            held += _ANNOTATION_BYTES * int(self._read["n_alleles"])
            held += _BARCODE_KEY_BYTES * int(self._read["n_barcodes"])
        if self.format == "darlin" and self._read.get("annotated") is not None:
            held += _READS_KEY_BYTES * int(self._read["n_annotated"])
        if self._table is not None:
            held += self._table.held_bytes()
        return held

    def record(self) -> dict[str, Any]:
        """Return what identifies how the file is read, beside its bytes.

        The bytes of the companion files, such as a CARLIN pair's
        annotations, identify the import too; the caller hashes them.
        """
        return {
            "format": self.format,
            "integration": self.integration,
            "lengths": dict(sorted(self.lengths.items())),
            "columns": dict(self.columns),
        }

    def source(self) -> dict[str, Any]:
        """Return what the artifact records of the file: its tool and notation."""
        return {
            "format": self.format,
            "tool": self.tool,
            "pipeline": self.pipeline,
            "kind": self.rules.kind,
            "integration": self.integration,
            "rows_hold_sites": self.rows_hold_sites,
            **{
                key: value
                for key, value in self._read.items()
                if key
                in ("states_from", "names_from", "encoding", "reads_from", "sites_from")
            },
        }


def _checked_tokens(value: Any, name: str) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        tokens = list(value)
        if all(isinstance(token, str) for token in tokens):
            return frozenset(tokens)
    raise TypeError(f"columns[{quoted_value(name)}] must be text or a sequence of text")


def _carlin_pair(path: Path) -> tuple[Path, Path] | None:
    """The annotations and colonies of a CARLIN pair that a path names, or None."""
    if path.is_dir():
        annotations = path / "AlleleAnnotations.txt"
        colonies = path / "AlleleColonies.txt"
        if annotations.is_file() or colonies.is_file():
            return annotations, colonies
        return None
    name = path.name
    if "AlleleAnnotations" in name:
        return path, path.with_name(name.replace("AlleleAnnotations", "AlleleColonies"))
    if "AlleleColonies" in name:
        return path.with_name(name.replace("AlleleColonies", "AlleleAnnotations")), path
    return None


def allele_file(path: str | os.PathLike[str]) -> str | os.PathLike[str]:
    """The file whose bytes identify the cells that a path names.

    A CARLIN pair, named by either file or by the folder that holds it, is
    identified by its ``AlleleColonies.txt``, which names the cells; every
    other path names itself.

    Raises:
        FileNotFoundError: If the path, or a named pair's file, does not
            exist.
    """
    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"No allele file at {quoted_value(os.fspath(path))}")
    if is_zarr_store(location):
        return path
    pair = _carlin_pair(location)
    if pair is None:
        if location.is_dir():
            raise ValueError(
                f"{quoted_value(os.fspath(path))} is a folder that holds neither a CARLIN pair "
                "nor a Zarr store"
            )
        return path
    for part in pair:
        if not part.is_file():
            raise FileNotFoundError(
                f"The CARLIN pair of {quoted_value(os.fspath(path))} lacks {part.name}, which "
                "must sit beside the other file"
            )
    return os.fspath(pair[1])


def _line_bytes(path: str) -> int:
    """Twice the bytes of the longest of a text file's first lines."""
    return 2 * max(
        (len(line.encode("utf-8")) for line in first_lines(path, _SAMPLED_LINES)),
        default=0,
    )


def _header(path: str) -> tuple[list[str], str]:
    """A text file's header names and its delimiter."""
    lines = first_lines(path, 2)
    if not lines or not lines[0].strip():
        raise ValueError(f"{path} is empty; an allele table needs a header line")
    separator = delimiter(lines)
    names = [name.strip() for name in split_fields(lines[0], separator)]
    return names, separator


def _detected_format(path: str) -> str:
    location = Path(path)
    if is_zarr_store(location):
        return "treedata"
    if _carlin_pair(location) is not None:
        return "carlin"
    head = file_signature(path)
    if head.startswith(HDF5_SIGNATURE):
        return "treedata"
    if head.startswith(_PARQUET_SIGNATURE):
        return "cassiopeia_parquet"
    names, _separator = _header(path)
    present = set(names)
    matches = []
    if {"n_UMIs", "CR", "mutation"} <= present:
        matches.append("darlin")
    if {"cellBC", "intBC"} <= present and any(
        _CASSIOPEIA_SITE.match(name) for name in names
    ):
        matches.append("cassiopeia")
    if {"intID", "cellBC", *_PETRACER_SITES} <= present:
        matches.append("petracer")
    if {"Cell", "TargetBC", "Site1"} <= present:
        matches.append("typewriter")
    if (
        names[:1] in (["cell"], ["cell_id"])
        and len(names) > 1
        and all(
            name in _TYPEWRITER_QC_COLUMNS or _NUCLEOTIDE_NAME.match(name)
            for name in names[1:]
        )
        and any(_NUCLEOTIDE_NAME.match(name) for name in names[1:])
    ):
        matches.append("typewriter_chains")
    if (
        names[:1] in ([""], ["cell_barcode"])
        and len(names) > 1
        and all(_PETRACER_COLUMN.match(name) for name in names[1:])
    ):
        matches.append("petracer_characters")
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            f"{path} has the columns of several formats, "
            + ", ".join(map(repr, matches))
            + "; pass format to choose one"
        )
    raise ValueError(
        f"The allele format of {path} is unknown; its columns are "
        + ", ".join(quoted_value(name) for name in names if name)
        + ". Pass format, one of "
        + ", ".join(map(repr, ALLELE_FORMATS))
        + "; 'laml' reads a character matrix and 'table' any table by the "
        "columns that columns names"
    )


def _integration_option(
    format: str, path: str, integrations: Any, n_slots: int | None
) -> tuple[str | None, dict[str, int]]:
    """Check one file's item of ``integrations`` against its format.

    Returns:
        The file's one integration, or None, and the slots of its shorter
        integrations.
    """
    if format in _ONE_INTEGRATION:
        if integrations is None:
            raise ValueError(
                f"{path} holds the alleles of one array, which the file does not "
                "name; pass integrations, such as 'CA', 'TA' or 'RA' for DARLIN's "
                "arrays, for each file"
            )
        if not isinstance(integrations, str):
            raise TypeError(
                f"integrations must name the array of {path}, as text, not "
                f"{quoted_value(integrations)}"
            )
    if integrations is None:
        return None, {}
    if isinstance(integrations, str) and (
        format in _ONE_INTEGRATION or format == "table"
    ):
        if not integrations:
            raise ValueError("integrations must name an integration, not be empty")
        return integrations, {}
    if format in _ORDERED or format == "table":
        if not isinstance(integrations, Mapping):
            raise TypeError(
                f"For {path}, integrations must map each integration shorter than "
                "the recorder's arrays to its number of slots"
            )
        lengths: dict[str, int] = {}
        for name, slots in integrations.items():
            if (
                not isinstance(name, str)
                or isinstance(slots, bool)
                or not isinstance(slots, int | np.integer)
            ):
                raise TypeError(
                    "integrations must map integration names to numbers of slots"
                )
            if n_slots is not None and not 1 <= int(slots) <= n_slots:
                raise ValueError(
                    f"The integration {quoted_value(name)} has {slots} slots, but the arrays of "
                    f"{path} have from 1 to {n_slots}"
                )
            lengths[name] = int(slots)
        return None, lengths
    raise ValueError(
        f"{path} names each row's integration, so integrations has nothing to "
        "name; it names the array of a file that holds one, such as a CARLIN pair, "
        "or the slots of contracted DNA Typewriter arrays"
    )


def _results_type(annotations: Path) -> str | None:
    """The ``CfgType`` of the CARLIN run that ``Results.txt`` beside a pair records."""
    name = annotations.name.replace("AlleleAnnotations", "Results")
    results = annotations.with_name(name)
    if not results.is_file():
        return None
    location = os.fspath(results)
    with text_errors(location), open_text(location) as handle:
        for line in handle:
            if line.startswith("CfgType:"):
                return line.split(":", 1)[1].strip()
    return None


def _lines(path: Path | str) -> Iterator[str]:
    """Stream a text file's lines, plain or gzip-compressed, without line ends.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    text = os.fspath(path)
    with text_errors(text), open_text(text) as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def _text_lines(path: Path) -> list[str]:
    """The lines of a small text file, without line ends, the last kept if unended.

    Raises:
        ValueError: If the file is not UTF-8 text, or a gzip file is cut
            short or corrupt.
    """
    location = os.fspath(path)
    with text_errors(location), open_text(location) as handle:
        text = handle.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def _carlin_layout(path: str, integration: str) -> AlleleLayout:
    pair = _carlin_pair(Path(path))
    if pair is None:
        raise ValueError(
            f"{path} is no CARLIN pair: format='carlin' reads AlleleAnnotations.txt "
            "and AlleleColonies.txt, named by either file or by their folder"
        )
    annotations, colonies = pair
    run = _results_type(annotations)
    if run is not None and not run.startswith("sc"):
        raise ValueError(
            f"{os.fspath(colonies)} comes from a CARLIN run of type {quoted_value(run)}, which "
            "lists UMIs or reads rather than cells; import single-cell runs, whose "
            "type starts with 'sc'"
        )
    lines = _text_lines(annotations)
    n_barcodes = 0
    shortest = longest = None
    for line in _lines(colonies):
        for barcode in line.split(","):
            barcode = barcode.strip()
            if barcode:
                n_barcodes += 1
                size = len(barcode.encode("utf-8"))
                shortest = size if shortest is None else min(shortest, size)
                longest = size if longest is None else max(longest, size)
    if run is None and shortest is not None and shortest < _MIN_CARLIN_BARCODE:
        raise ValueError(
            f"{os.fspath(colonies)} lists tags of {shortest} bases, as CARLIN's bulk "
            "runs list UMIs rather than cells; import single-cell runs. A run of "
            "shorter cell barcodes reads with its Results.txt beside the pair, "
            "whose CfgType names the run"
        )
    return AlleleLayout(
        format="carlin",
        path=os.fspath(colonies),
        tool="CARLIN",
        pipeline="CARLIN (MATLAB)",
        rules=SiteRules(
            kind="indel",
            unedited=frozenset({"[]"}),
            unknown=frozenset({"X", "?"}),
            pattern=_CARLIN_ALLELE,
        ),
        integration=integration,
        companions=(("annotations", os.fspath(annotations)),),
        n_columns=1,
        n_sites=1,
        field_bytes=_CARLIN_ROW_BYTES,
        line_bytes=2 * (longest or 0),
        _read=MappingProxyType(
            {
                "annotations": os.fspath(annotations),
                "n_alleles": len(lines),
                "n_barcodes": n_barcodes,
            }
        ),
    )


def _annotated_path(final: Path) -> Path | None:
    for name in ("step4_annotated.tsv", "step4_annotated.tsv.gz"):
        candidate = final.with_name(name)
        if candidate.is_file():
            return candidate
    return None


def _count_text_rows(text: TextLayout) -> int:
    count = 0
    for _numbers, lines in line_chunks(text, _MAX_CHUNK_ROWS):
        count += len(lines)
    return count


def _darlin_layout(path: str, integration: str) -> AlleleLayout:
    text = text_layout(path, what="a darlin table")
    require_columns(text, ("n_UMIs", "CR", "mutation"))
    line_bytes = _line_bytes(path)
    annotated = _annotated_path(Path(path))
    read: dict[str, Any] = {"annotated": None, "reads_from": None}
    companions: tuple[tuple[str, str], ...] = ()
    if annotated is not None:
        if "aligned_LR" not in text.names:
            raise ValueError(
                f"{path} has no aligned_LR column, which joins its rows to the "
                f"molecules of {annotated.name}"
            )
        molecules = text_layout(os.fspath(annotated), what="a darlin table")
        require_columns(molecules, ("CR", "reads", "aligned_LR"))
        read = {
            "annotated": molecules,
            "reads_from": annotated.name,
            "n_annotated": _count_text_rows(molecules),
        }
        companions = (("molecules", os.fspath(annotated)),)
        # The molecules are read in chunks of the same rows.
        line_bytes = max(line_bytes, _line_bytes(molecules.path))
    return AlleleLayout(
        format="darlin",
        path=path,
        tool="darlin",
        pipeline="darlin-core",
        rules=SiteRules(
            kind="indel", unedited=frozenset({"[]", ""}), pattern=_CARLIN_ALLELE
        ),
        integration=integration,
        companions=companions,
        n_columns=len(text.names),
        n_sites=1,
        line_bytes=line_bytes,
        _text=text,
        _read=MappingProxyType(read),
    )


def _cassiopeia_layout(path: str) -> AlleleLayout:
    text = text_layout(path, what="an allele table")
    require_columns(text, ("cellBC", "intBC"))
    sites = [name for name in text.names if _CASSIOPEIA_SITE.match(name)]
    if not sites:
        raise ValueError(f"{path} has no site column r1 to rN of an allele table")
    require_columns(text, sites)
    return AlleleLayout(
        format="cassiopeia",
        path=path,
        tool="Cassiopeia",
        pipeline="Cassiopeia 2",
        rules=SiteRules(
            kind="indel",
            missing=frozenset({"", "NaN", "nan", "NA", "NC", "Missing"}),
            unedited_within="None",
        ),
        columns=MappingProxyType({"sites": sites}),
        n_columns=len(text.names),
        n_sites=len(sites),
        line_bytes=_line_bytes(path),
        _text=text,
        _read=MappingProxyType({"sites": tuple(sites)}),
    )


def _polars() -> Any:
    try:
        import polars
    except ImportError as error:
        raise ImportError(
            "Reading Cassiopeia's parquet allele tables needs polars 1.30 or later; "
            "install polars"
        ) from error
    # Cassiopeia keeps its attributes in the file's metadata, which polars
    # reads from 1.30 on.
    if not hasattr(polars, "read_parquet_metadata"):
        raise ImportError(
            "Reading Cassiopeia's parquet allele tables needs polars 1.30 or later, "
            f"which reads a parquet file's metadata, not polars {polars.__version__}; "
            "install a later polars"
        )
    return polars


def _parquet_attributes(path: str, metadata: Mapping[str, str]) -> dict[str, Any]:
    """The ``attrs`` that pandas saves with a table, as Cassiopeia 3.0 writes them.

    Raises:
        ValueError: If the table's attributes are not a JSON object.
    """
    import json

    saved = metadata.get(_PANDAS_ATTRS)
    if saved is None:
        return {}
    try:
        attributes = json.loads(saved)
    except ValueError as error:
        raise ValueError(
            f"The {_PANDAS_ATTRS} of {path} are not JSON: {error}"
        ) from None
    if not isinstance(attributes, dict):
        raise ValueError(f"The {_PANDAS_ATTRS} of {path} are not a JSON object")
    return attributes


def _parquet_rules(
    path: str, sites: Sequence[str], attributes: Mapping[str, Any]
) -> SiteRules:
    """The rules of a table's sites, by the edit model that Cassiopeia 3.0 records."""
    models = attributes.get("edit_models") or {}
    if not isinstance(models, dict):
        raise ValueError(f"The edit_models of {path} are not a mapping of sites")
    found = {str(models.get(site, "indel")) for site in sites}
    if len(found) > 1:
        raise ValueError(
            f"The sites of {path} follow several edit models, "
            + ", ".join(sorted(found))
            + "; CyteArc reads the sites of one table by one model"
        )
    (model,) = found
    if model == "indel":
        # Text that holds None is an uncut site, as Cassiopeia reads it.
        return SiteRules(kind="indel", missing=frozenset({""}), unedited_within="None")
    if model == "insertion_lookup":
        states = attributes.get("site_states") or {}
        if not isinstance(states, dict) or any(
            not isinstance(states.get(site), list) for site in sites
        ):
            raise ValueError(
                f"{path} records sites of insertions of known sequences without "
                "the site_states that name their labels"
            )
        # Insertions in the lookup are written by their label, others by their
        # sequence, deletions as D and their length, with I and the sequence
        # inserted beside them, and the unedited site as None.
        return SiteRules(
            kind="independent",
            missing=frozenset({""}),
            unedited=frozenset({"None"}),
            marks_at=MappingProxyType(
                {site: frozenset(map(str, states[site])) for site in sites}
            ),
            pattern=_LOOKUP_EDIT,
            ambiguous=_AMBIGUOUS_INSERTION,
        )
    if model == "sequential_insertion":
        return SiteRules(
            kind="ordered",
            missing=frozenset({""}),
            unedited=frozenset({"None"}),
            pattern=_INSERTION,
            ambiguous=_AMBIGUOUS_INSERTION,
        )
    raise ValueError(
        f"The sites of {path} follow Cassiopeia's {quoted_value(model)} edit model, which "
        "CyteArc does not read"
    )


def _parquet_layout(path: str) -> AlleleLayout:
    pl = _polars()
    schema = pl.read_parquet_schema(path)
    names = list(schema)
    missing = [name for name in ("cellBC", "intBC") if name not in names]
    if missing:
        raise ValueError(
            f"{path} has no column " + ", ".join(map(repr, missing)) + " of "
            "Cassiopeia's allele tables"
        )
    if "UMI" in names:
        raise ValueError(
            f"{path} holds one row per UMI, as Cassiopeia 3.0 writes a table before "
            "it collapses molecules; import the collapsed table, whose n_umis "
            "count molecules"
        )
    attributes = _parquet_attributes(path, pl.read_parquet_metadata(path))
    listed = attributes.get("sites")
    if listed is not None:
        if not isinstance(listed, list) or not all(isinstance(s, str) for s in listed):
            raise ValueError(f"The sites of {path} are not a list of column names")
        sites = [site for site in listed if site in names]
        sites_from = "attrs"
    else:
        # Cassiopeia's own rule: the categorical columns that no other role
        # names.
        sites = [
            name
            for name in names
            if name not in _PARQUET_RESERVED
            and not name.startswith(("pass_", "doublet"))
            and isinstance(schema[name], pl.Categorical | pl.Enum)
        ]
        sites_from = "columns"
    if not sites:
        raise ValueError(f"{path} names no site columns")
    rules = _parquet_rules(path, sites, attributes)
    sample_values = []
    if "sample" in names:
        sample_values = (
            (pl.scan_parquet(path).select(pl.col("sample").cast(pl.String).unique()))
            .collect()["sample"]
            .to_list()
        )
        if len(sample_values) > 1:
            raise ValueError(
                f"{path} holds {len(sample_values)} samples, whose cell barcodes the "
                "join cannot tell apart; write one table per sample and give each "
                "its barcode_prefix"
            )
    return AlleleLayout(
        format="cassiopeia_parquet",
        path=path,
        tool="Cassiopeia",
        pipeline="Cassiopeia 3",
        rules=rules,
        n_slots=len(sites) if rules.kind == "ordered" else None,
        columns=MappingProxyType({"sites": sites}),
        n_columns=len(sites) + 5,
        n_sites=len(sites),
        field_bytes=_PARQUET_FIELD_BYTES,
        _read=MappingProxyType(
            {
                "sites": tuple(sites),
                "sites_from": sites_from,
                "has_pass": "pass" in names,
                "has_umis": "n_umis" in names,
                "has_reads": "n_reads" in names,
            }
        ),
    )


def _petracer_layout(path: str) -> AlleleLayout:
    text = text_layout(path, what="an allele table")
    require_columns(text, ("intID", "cellBC", *_PETRACER_SITES))
    return AlleleLayout(
        format="petracer",
        path=path,
        tool="PEtracer",
        pipeline="PEtracer 2025",
        rules=SiteRules(
            kind="independent",
            missing=frozenset({""}),
            unedited=frozenset({"None"}),
            unedited_at=MappingProxyType({"EMX1": frozenset({"CTTGGG"})}),
            marks_at=_PETRACER_MARKS,
            pattern=_INSERTION,
            ambiguous=_AMBIGUOUS_INSERTION,
        ),
        n_columns=len(text.names),
        n_sites=len(_PETRACER_SITES),
        line_bytes=_line_bytes(path),
        _text=text,
    )


def _split_character(name: str, path: str) -> tuple[str, str]:
    integration, hyphen, site = name.rpartition("-")
    if not hyphen:
        return name, ""
    if not integration or not site:
        raise ValueError(
            f"The character {quoted_value(name)} of {path} has no integration or no site on "
            "either side of its last hyphen"
        )
    return integration, site


def _matrix_columns(names: Sequence[str], path: str) -> tuple[tuple[str, str], ...]:
    columns = tuple(_split_character(name, path) for name in names)
    if not columns:
        raise ValueError(f"{path} holds no character column")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{path} names a character twice")
    return columns


def _matrix_layout(path: str, format: str) -> AlleleLayout:
    # A matrix's first column holds its cells, so a header one field shorter
    # than its rows lacks the cells' column, as R's write.table leaves it,
    # even where every row ends in an empty state, as PEtracer writes a site
    # that the cell's embryo lacks. A matrix whose rows do end in a delimiter
    # after a named cell column then gives an empty last state, or a cell
    # column among the characters, which the states and names refuse.
    text = text_layout(path, what="a character matrix", prefer_row_names=True)
    if (
        format == "petracer_characters"
        and not text.row_names
        and _PETRACER_COLUMN.match(text.names[0])
    ):
        # A space-separated file loses an empty field at the end of a row
        # unless it is quoted, so a matrix whose rows end in an empty state
        # holds no more fields than its header, and its first character would
        # read as the cells' column.
        raise ValueError(
            f"{path} names its first column {quoted_value(text.names[0])}, as PEtracer names "
            "a character, but its rows hold no more fields than its header, so "
            "that column would read as the cells': a space-separated matrix "
            "loses an empty state at the end of a row unless its fields are "
            "quoted. Quote the fields, as R's write.table does by default, or "
            "separate them by commas or tabs"
        )
    names = list(text.names[1:])
    columns = _matrix_columns(names, path)
    if format == "petracer_characters":
        other = sorted({site for _integration, site in columns} - set(_PETRACER_SITES))
        if other:
            # A first column named among characters read as R's row names is
            # likely a cell column before rows that end in a delimiter.
            hint = (
                f"; its rows hold one field more than its header, so its first "
                f"column, {quoted_value(text.names[1])}, reads as a character, as R's "
                "write.table leaves the cells' column unnamed: write the matrix "
                "without a delimiter at the end of each row"
                if rows_end_in_delimiter(text)
                else ""
            )
            raise ValueError(
                f"{path} names sites {', '.join(map(repr, other))}, which are none "
                "of PEtracer's " + ", ".join(_PETRACER_SITES) + "; PEtracer's "
                "characters are named intID<N>-<site>" + hint
            )
        rules = _PETRACER_CHARACTER_RULES
        tool, pipeline = "PEtracer", "PEtracer 2026"
    else:
        rules = SiteRules(
            kind="independent",
            missing=frozenset({"-1", "-", "?", "NA"}),
            unedited=frozenset({"0", "*"}),
            pattern=_MATRIX_STATE,
        )
        tool, pipeline = "character matrix", "character matrix"
    return AlleleLayout(
        format=format,
        path=path,
        tool=tool,
        pipeline=pipeline,
        rules=rules,
        n_columns=len(text.names),
        n_sites=len(columns),
        line_bytes=_line_bytes(path),
        _text=text,
        _read=MappingProxyType({"columns": columns}),
    )


def _typewriter_layout(path: str, lengths: Mapping[str, int]) -> AlleleLayout:
    names, separator = _header(path)
    slots = [name for name in names if re.fullmatch(r"Site\d+", name)]
    expected = [
        "Cell",
        "TargetBC",
        *(f"Site{slot}" for slot in range(1, len(slots) + 1)),
    ]
    if names[: len(expected)] != expected or not slots:
        raise ValueError(
            f"The header of {path} must name Cell, TargetBC and Site1 to SiteN in "
            f"that order, as DNA Typewriter's table does, not {', '.join(names)}"
        )
    extra = names[len(expected) :]
    if extra not in ([], ["nUMI"]):
        raise ValueError(
            f"{path} has columns {', '.join(map(repr, extra))} after its slots; "
            "DNA Typewriter's table ends with nUMI"
        )
    text = TextLayout(
        path=path,
        names=tuple(names),
        delimiter=separator,
        row_names=False,
    )
    return AlleleLayout(
        format="typewriter",
        path=path,
        tool="DNA Typewriter",
        pipeline="DNA Typewriter 2022",
        rules=SiteRules(
            kind="ordered",
            unedited=frozenset({""}),
            pattern=_INSERTION,
            ambiguous=_AMBIGUOUS_INSERTION,
        ),
        lengths=MappingProxyType(dict(lengths)),
        n_slots=len(slots),
        n_columns=len(names) + 1,
        n_sites=len(slots),
        field_bytes=_LIST_FIELD_BYTES,
        line_bytes=_line_bytes(path),
        _text=text,
        _read=MappingProxyType({"has_umis": bool(extra)}),
    )


def _chains_layout(path: str, lengths: Mapping[str, int]) -> AlleleLayout:
    text = text_layout(path, what="a DNA Typewriter table")
    names = list(text.names)
    if names[:1] not in ([name] for name in _TYPEWRITER_CELL_COLUMNS):
        raise ValueError(
            f"The first column of {path} must be cell or cell_id, as DNA Typewriter "
            f"writes it, not {quoted_value(names[0])}"
        )
    integrations = []
    for name in names[1:]:
        if name in _TYPEWRITER_QC_COLUMNS:
            continue
        if not _NUCLEOTIDE_NAME.match(name):
            raise ValueError(
                f"{path} has a column {quoted_value(name)} that is neither an integration "
                "barcode nor one of DNA Typewriter's quality columns, "
                + ", ".join(sorted(_TYPEWRITER_QC_COLUMNS))
            )
        integrations.append(name)
    if not integrations:
        raise ValueError(f"{path} names no integration barcode")
    if len(set(integrations)) != len(integrations):
        raise ValueError(f"{path} names an integration twice")
    lacking = sorted(set(lengths) - set(integrations))
    if lacking:
        raise ValueError(
            f"integrations gives the slots of {', '.join(map(repr, lacking))}, which "
            f"{path} holds no column of; check the names of the contracted tapes"
        )
    return AlleleLayout(
        format="typewriter_chains",
        path=path,
        tool="DNA Typewriter",
        pipeline="DNA Typewriter 2026",
        rules=SiteRules(
            kind="ordered",
            unedited=frozenset({""}),
            pattern=_INSERTION,
            ambiguous=_AMBIGUOUS_INSERTION,
            absent=MISSING,
        ),
        lengths=MappingProxyType(dict(lengths)),
        n_slots=_TYPEWRITER_CHAIN_SLOTS,
        n_columns=len(names),
        n_sites=len(integrations) * _TYPEWRITER_CHAIN_SLOTS,
        line_bytes=_line_bytes(path),
        _text=text,
        _read=MappingProxyType({"integrations": tuple(integrations)}),
    )


def _treedata_layout(path: str) -> AlleleLayout:
    table = character_table(path)
    columns = _matrix_columns(table.characters, path)
    if table.names_from != "columns":
        # PEtracer's published files hold its character matrices of 2026 as
        # an array, with the matrices' tokens and no states in uns.
        return AlleleLayout(
            format="treedata",
            path=path,
            tool="PEtracer",
            pipeline="PEtracer 2026",
            rules=_PETRACER_CHARACTER_RULES,
            n_columns=len(columns),
            n_sites=len(columns),
            _table=table,
            _read=MappingProxyType(
                {
                    "columns": columns,
                    "states_from": "PEtracer's characters",
                    "names_from": table.names_from,
                    "encoding": table.encoding,
                }
            ),
        )
    if table.missing_state is not None or table.unmodified_state is not None:
        if table.missing_state is None or table.unmodified_state is None:
            raise ValueError(
                f"{path} records only one of uns['missing_state'] and "
                "uns['unmodified_state']; a file names both or neither"
            )
        missing = frozenset({table.missing_state})
        unedited = frozenset({table.unmodified_state})
        states_from = "uns"
    else:
        missing = frozenset({"-1", "-", "NA"})
        unedited = frozenset({"0", "*"})
        states_from = "Cassiopeia's defaults"
    return AlleleLayout(
        format="treedata",
        path=path,
        tool="TreeData",
        pipeline="TreeData",
        rules=SiteRules(
            kind="independent",
            missing=missing,
            unedited=unedited,
            # PEtracer writes an insertion outside its lookup as "!", which
            # no other tool writes as a state.
            unknown=frozenset({"!"}) - missing - unedited,
            pattern=_MATRIX_STATE,
        ),
        n_columns=len(columns),
        n_sites=len(columns),
        _table=table,
        _read=MappingProxyType(
            {
                "columns": columns,
                "states_from": states_from,
                "encoding": table.encoding,
            }
        ),
    )


def _table_roles(columns: Any) -> dict[str, Any]:
    if not isinstance(columns, Mapping):
        raise TypeError(
            "columns must map roles such as 'cell' and 'allele' to column names"
        )
    unknown = sorted(str(role) for role in columns if role not in _TABLE_ROLES)
    if unknown:
        raise ValueError(
            f"columns names unknown roles {', '.join(map(repr, unknown))}; the roles "
            "are " + ", ".join(map(repr, _TABLE_ROLES))
        )
    roles = dict(columns)
    if not isinstance(roles.get("cell"), str):
        raise ValueError("columns must name the 'cell' column, as text")
    for role in ("integration", "site", "allele", "umis", "reads"):
        if role in roles and not (isinstance(roles[role], str) and roles[role]):
            raise TypeError(f"columns[{quoted_value(role)}] must be a column name")
    shapes = [role for role in ("sites", "site") if role in roles]
    if "sites" in roles:
        sites = roles["sites"]
        if (
            isinstance(sites, str)
            or not isinstance(sites, Sequence)
            or not sites
            or not all(isinstance(name, str) and name for name in sites)
        ):
            raise TypeError("columns['sites'] must be a sequence of column names")
        if len(set(sites)) != len(sites):
            raise ValueError("columns['sites'] names a column twice")
        if "allele" in roles or "separator" in roles:
            raise ValueError(
                "columns names either 'sites', a column per site, or 'allele', one "
                "column of every site's allele"
            )
    elif "allele" not in roles:
        raise ValueError(
            "columns must name 'sites', a column per site, or 'allele', the column "
            "that holds each row's allele"
        )
    if len(shapes) > 1:
        raise ValueError("columns names 'sites' or 'site', not both")
    if "separator" in roles and (
        not isinstance(roles["separator"], str) or len(roles["separator"]) != 1
    ):
        raise ValueError("columns['separator'] must be one character")
    if "site" in roles and "separator" in roles:
        raise ValueError(
            "columns names 'site', so each row holds one site, or 'separator', "
            "which splits a row's allele into sites, not both"
        )
    kind = roles.get("kind", "indel")
    if kind not in SITE_KINDS:
        raise ValueError(
            f"columns['kind'] must be one of {', '.join(map(repr, SITE_KINDS))}, not "
            f"{quoted_value(kind)}"
        )
    if kind == "ordered" and "site" in roles:
        raise ValueError(
            "Ordered slots need a row's slots together, in 'sites' or split from "
            "'allele' by 'separator', not one site per row"
        )
    if "slots" in roles:
        slots = roles["slots"]
        if kind != "ordered":
            raise ValueError("columns['slots'] counts the slots of ordered sites only")
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ValueError("columns['slots'] must be a whole number of at least 1")
    for role in _TOKEN_ROLES:
        if role in roles:
            roles[role] = sorted(_checked_tokens(roles[role], role))
    return roles


def _table_layout(
    path: str, columns: Any, integration: str | None, lengths: Mapping[str, int]
) -> AlleleLayout:
    roles = _table_roles(columns)
    text = text_layout(path, what="an allele table")
    named = [
        roles[role]
        for role in ("cell", "integration", "site", "allele", "umis", "reads")
        if role in roles
    ] + list(roles.get("sites", []))
    require_columns(text, named)
    if ("integration" in roles) == (integration is not None):
        raise ValueError(
            "A table read through columns names each row's integration in "
            "columns['integration'], or its one integration in integrations; give "
            "one of them"
        )
    given = {role: frozenset(roles.get(role, ())) for role in _TOKEN_ROLES}
    # The tokens that pandas reads as missing are missing, unless the table's
    # roles give them another meaning.
    others = given["unedited"] | given["not_applicable"] | given["unknown"]
    rules = SiteRules(
        kind=roles.get("kind", "indel"),
        not_applicable=given["not_applicable"],
        missing=(MISSING_TEXT - others) | given["missing"],
        unedited=given["unedited"],
        unknown=given["unknown"],
    )
    if "sites" in roles:
        sites = [str(name) for name in roles["sites"]]
    elif "separator" in roles:
        sites = []
    else:
        sites = [""]
    n_slots = roles.get("slots")
    if rules.kind == "ordered" and n_slots is None and sites:
        n_slots = len(sites)
    if lengths and rules.kind != "ordered":
        raise ValueError("integrations gives slots of ordered sites only")
    split_sites = None
    if "separator" in roles:
        if rules.kind == "ordered":
            if n_slots is None:
                raise ValueError(
                    "columns['slots'] must count the slots of ordered sites that "
                    "'separator' splits from each row's allele"
                )
        else:
            # The sites of every row are those of the file's first row, so a
            # chunk of any size reads them alike.
            split_sites = _first_split_sites(text, roles, rules.missing)
    return AlleleLayout(
        format="table",
        path=path,
        tool="table",
        pipeline="table",
        rules=rules,
        integration=integration,
        lengths=MappingProxyType(dict(lengths)),
        n_slots=n_slots,
        rows_hold_sites="site" in roles,
        columns=MappingProxyType(roles),
        n_columns=len(text.names),
        n_sites=len(sites) if sites else (n_slots or split_sites or 1),
        line_bytes=_line_bytes(path),
        _text=text,
        _read=MappingProxyType({"sites": tuple(sites), "split_sites": split_sites}),
    )


def _first_split_sites(
    text: TextLayout, roles: Mapping[str, Any], missing: frozenset[str]
) -> int:
    """The sites that the first allele of a table splits into, or 1 for none.

    The file is read only until its first row with an allele, most often its
    first row.
    """
    position = text.names.index(roles["allele"])
    for _line, chunk in text_chunks(text, _SAMPLED_LINES):
        for value in _stripped(chunk[position]).tolist():
            if value is not None and value not in missing:
                return len(str(value).split(roles["separator"]))
    return 1


def allele_layout(
    path: str | os.PathLike[str],
    *,
    format: str | None = None,
    columns: Mapping[str, Any] | None = None,
    integrations: Any = None,
) -> AlleleLayout:
    """Recognize a file's format and find what each of its rows holds.

    Args:
        path: The file, a CARLIN pair's file or folder, or a ``.zarr`` store.
        format: One of ``ALLELE_FORMATS``, or None to recognize it.
        columns: For ``format="table"``, the columns to read by role.
        integrations: This file's item of the import's ``integrations``:
            the name of the one integration of a file that holds one, or a
            mapping of integrations shorter than the recorder's arrays to
            their slots.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the format is unknown or the file is not in it.
    """
    text = os.fspath(path)
    if format is not None and format not in ALLELE_FORMATS:
        raise ValueError(
            f"format must be one of {', '.join(map(repr, ALLELE_FORMATS))}, not "
            f"{quoted_value(format)}"
        )
    if columns is not None and format not in (None, "table"):
        raise ValueError(
            "columns reads tables of other tools with format='table', so it does "
            f"not apply to format={quoted_value(format)}"
        )
    found = (
        ("table" if columns is not None else _detected_format(text))
        if format is None
        else format
    )
    if found == "table" and columns is None:
        raise ValueError("format='table' needs columns that name what to read")
    slots = {"typewriter": None, "typewriter_chains": _TYPEWRITER_CHAIN_SLOTS}.get(
        found
    )
    integration, lengths = _integration_option(found, text, integrations, slots)
    if found == "carlin":
        assert integration is not None
        return _carlin_layout(text, integration)
    if found == "darlin":
        assert integration is not None
        return _darlin_layout(text, integration)
    if found == "cassiopeia":
        return _cassiopeia_layout(text)
    if found == "cassiopeia_parquet":
        if not file_signature(text).startswith(_PARQUET_SIGNATURE):
            raise ValueError(
                f"{text} is no parquet file, which format='cassiopeia_parquet' reads"
            )
        return _parquet_layout(text)
    if found == "petracer":
        return _petracer_layout(text)
    if found in ("petracer_characters", "laml"):
        return _matrix_layout(text, found)
    if found == "typewriter":
        layout = _typewriter_layout(text, lengths)
        for name, length in lengths.items():
            if not 1 <= length <= (layout.n_slots or 0):
                raise ValueError(
                    f"The integration {quoted_value(name)} has {length} slots, but the arrays of "
                    f"{text} have from 1 to {layout.n_slots}"
                )
        return layout
    if found == "typewriter_chains":
        return _chains_layout(text, lengths)
    if found == "treedata":
        return _treedata_layout(text)
    return _table_layout(text, columns, integration, lengths)


def _stripped(values: Any) -> np.ndarray:
    """Text stripped of surrounding whitespace, None kept."""
    raw = np.asarray(values, dtype=object).reshape(-1)
    return np.asarray(
        [value.strip() if isinstance(value, str) else value for value in raw.tolist()],
        dtype=object,
    )


def _counts(
    values: Any, name: str, lines: np.ndarray, path: str, *, table: bool
) -> np.ndarray:
    """Counts of a column, -1 where a table read through columns has none.

    A tool's table writes a count in every row, so there a missing value is
    a row cut short. ``lines`` are the rows' lines, which errors name.
    """
    try:
        return parse_integers(
            np.asarray(values, dtype=object),
            name=name,
            missing=MISSING_TEXT if table else frozenset(),
            lines=lines,
        )
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def _cells(values: Any, lines: np.ndarray, path: str) -> np.ndarray:
    """Cell barcodes, or integration names, stripped; an empty one is refused.

    ``lines`` are the rows' lines, which errors name.
    """
    cells = _stripped(values)
    empty = np.asarray([not value for value in cells.tolist()], dtype=bool)
    if empty.any():
        rows = np.flatnonzero(empty)
        raise ValueError(
            f"{path} has rows without a cell barcode or integration on "
            + file_lines(np.asarray(lines)[rows])
        )
    return cells


def _integration_names(values: Any, lines: np.ndarray, path: str) -> np.ndarray:
    """Integration names, stripped; an empty one names no integration, None.

    PEtracer leaves ``intID`` empty where an intBC is not on its whitelist.
    """
    del lines, path
    names = _stripped(values)
    names[np.asarray([not name for name in names.tolist()], dtype=bool)] = None
    return names


def _iter_carlin(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    annotations = _text_lines(Path(layout._read["annotations"]))
    empty = [
        number for number, line in enumerate(annotations, start=1) if not line.strip()
    ]
    if empty:
        raise ValueError(
            f"{layout._read['annotations']} has an empty line {empty[0]}, so the "
            "allele of its cells is unknown"
        )
    keys: list[np.ndarray] = []
    barcodes: list[str] = []
    numbers: list[int] = []
    tokens: list[str] = []
    n_lines = 0

    def flush() -> AlleleRows:
        found = np.asarray(barcodes, dtype=object)
        keys.append(pair_keys(found, np.full(len(found), "", dtype=object)))
        return AlleleRows(
            barcodes=found,
            rows=np.asarray(numbers, dtype=np.int64),
            columns=((layout.integration, ""),),
            tokens=np.asarray(tokens, dtype=object).reshape(-1, 1),
        )

    for number, line in enumerate(_lines(layout.path), start=1):
        n_lines = number
        if number > len(annotations):
            raise ValueError(
                f"{layout.path} has more lines than {layout._read['annotations']}, "
                "which names the allele of each line"
            )
        allele = annotations[number - 1].strip()
        for barcode in line.split(","):
            barcode = barcode.strip()
            if not barcode:
                continue
            barcodes.append(barcode)
            numbers.append(number)
            tokens.append(allele)
            if len(barcodes) == chunk_rows:
                yield flush()
                barcodes, numbers, tokens = [], [], []
    if barcodes:
        yield flush()
    if n_lines != len(annotations):
        raise ValueError(
            f"{layout.path} has {n_lines} lines and {layout._read['annotations']} "
            f"{len(annotations)}, but a CARLIN pair lists one allele per line of both"
        )
    repeated = repeated_keys(keys)
    if repeated:
        raise ValueError(
            f"{layout.path} lists {repeated} cell barcodes on more than one line, but "
            "a single-cell CARLIN run gives each cell one allele"
        )


def _read_sums(
    layout: AlleleLayout, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum the reads and count the molecules of each cell and lineage barcode.

    Returns:
        The distinct keys of ``CR`` and ``aligned_LR`` sorted, as two uint64
        columns, and each key's reads and molecules.
    """
    molecules: TextLayout = layout._read["annotated"]
    positions = {name: position for position, name in enumerate(molecules.names)}
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for lines, chunk in text_chunks(molecules, chunk_rows):
        keys = pair_keys(
            _stripped(chunk[positions["CR"]]), _stripped(chunk[positions["aligned_LR"]])
        )
        reads = _counts(
            chunk[positions["reads"]].to_numpy(dtype=object),
            "reads",
            lines,
            molecules.path,
            table=False,
        )
        parts.append((keys, reads))
    if not parts:
        empty = np.empty((0, 2), dtype=np.uint64)
        return empty, np.empty(0, np.int64), np.empty(0, np.int64)
    keys = np.concatenate([part[0] for part in parts])
    reads = np.concatenate([part[1] for part in parts])
    del parts
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, reads = keys[order], reads[order]
    starts = np.ones(len(keys), dtype=bool)
    starts[1:] = (keys[1:] != keys[:-1]).any(axis=1)
    first = np.flatnonzero(starts)
    sums = np.add.reduceat(reads, first) if len(first) else np.empty(0, np.int64)
    counts = np.diff(np.append(first, len(keys)))
    return keys[first], sums.astype(np.int64), counts.astype(np.int64)


def _lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """The position of each key among sorted distinct keys, -1 where absent."""
    high, low = sorted_keys[:, 0], sorted_keys[:, 1]
    start = np.searchsorted(high, keys[:, 0], side="left")
    stop = np.searchsorted(high, keys[:, 0], side="right")
    found = np.full(len(keys), -1, dtype=np.int64)
    for row in np.flatnonzero(stop > start).tolist():
        candidates = np.arange(start[row], stop[row])
        match = candidates[low[candidates] == keys[row, 1]]
        if len(match):
            found[row] = int(match[0])
    return found


def _iter_darlin(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    positions = {name: position for position, name in enumerate(text.names)}
    sums = None
    if layout._read.get("annotated") is not None:
        sums = _read_sums(layout, chunk_rows)
    for lines, chunk in text_chunks(text, chunk_rows):
        cells = _cells(chunk[positions["CR"]].to_numpy(dtype=object), lines, text.path)
        umis = _counts(
            chunk[positions["n_UMIs"]].to_numpy(dtype=object),
            "n_UMIs",
            lines,
            text.path,
            table=False,
        )
        reads = None
        if sums is not None:
            keys = pair_keys(cells, _stripped(chunk[positions["aligned_LR"]]))
            found = _lookup(sums[0], keys)
            if (found < 0).any():
                rows = np.flatnonzero(found < 0)
                raise ValueError(
                    f"{text.path} names cells and lineage barcodes that "
                    f"{layout._read['reads_from']} holds no molecule of, on "
                    + file_lines(lines[rows])
                )
            molecules = sums[2][found]
            mismatched = np.flatnonzero(molecules != umis)
            if len(mismatched):
                raise ValueError(
                    f"{text.path} counts other UMIs than the molecules that "
                    f"{layout._read['reads_from']} lists for the same cell and lineage "
                    "barcode, on " + file_lines(lines[mismatched])
                )
            reads = sums[1][found]
        yield AlleleRows(
            barcodes=cells,
            rows=lines,
            columns=((layout.integration, ""),),
            tokens=_stripped(chunk[positions["mutation"]]).reshape(-1, 1),
            umis=umis,
            reads=reads,
        )


def _iter_cassiopeia(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    positions = {name: position for position, name in enumerate(text.names)}
    sites = layout._read["sites"]
    for lines, chunk in text_chunks(text, chunk_rows):
        umis = reads = None
        if "UMI" in positions:
            raw = chunk[positions["UMI"]].to_numpy(dtype=object)
            if any(
                isinstance(value, str) and re.fullmatch(r"[ACGTN]+", value.strip())
                for value in raw[: min(len(raw), 100)].tolist()
            ):
                raise ValueError(
                    f"The UMI column of {text.path} holds sequences, as Cassiopeia's "
                    "molecule tables do; import its allele table, whose UMI counts "
                    "molecules"
                )
            umis = _counts(raw, "UMI", lines, text.path, table=False)
        if "readCount" in positions:
            reads = _counts(
                chunk[positions["readCount"]].to_numpy(dtype=object),
                "readCount",
                lines,
                text.path,
                table=False,
            )
        tokens = np.stack([_stripped(chunk[positions[site]]) for site in sites], axis=1)
        yield AlleleRows(
            barcodes=_cells(
                chunk[positions["cellBC"]].to_numpy(dtype=object),
                lines,
                text.path,
            ),
            rows=lines,
            columns=tuple((None, site) for site in sites),
            tokens=tokens,
            integrations=_integration_names(
                chunk[positions["intBC"]].to_numpy(dtype=object), lines, text.path
            ),
            umis=umis,
            reads=reads,
        )


def _iter_parquet(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    pl = _polars()
    sites = list(layout._read["sites"])
    wanted = ["cellBC", "intBC", *sites]
    if layout._read["has_umis"]:
        wanted.append("n_umis")
    if layout._read["has_reads"]:
        wanted.append("n_reads")
    # Each slice of the scan reads only the row groups it needs, and the rows
    # that fail Cassiopeia's filters are dropped once read.
    selected = (
        pl.scan_parquet(layout.path)
        .with_row_index("__row", offset=1)
        .select(
            [pl.col("__row")]
            + ([pl.col("pass").cast(pl.Boolean)] if layout._read["has_pass"] else [])
            + [pl.col(name).cast(pl.String) for name in ("cellBC", "intBC", *sites)]
            + [
                pl.col(name).cast(pl.Int64)
                for name in ("n_umis", "n_reads")
                if name in wanted
            ]
        )
    )
    start = 0
    while True:
        chunk = selected.slice(start, chunk_rows).collect()
        if chunk.height == 0:
            return
        start += chunk.height
        if layout._read["has_pass"]:
            if chunk["pass"].null_count():
                raise ValueError(f"{layout.path} has rows whose pass is missing")
            chunk = chunk.filter(pl.col("pass"))
            if chunk.height == 0:
                continue
        rows = chunk["__row"].to_numpy().astype(np.int64)

        def counts(name: str) -> np.ndarray | None:
            if name not in chunk.columns:
                return None
            values = chunk[name]
            if values.null_count() or (values < 0).any():
                raise ValueError(
                    f"{layout.path} has a {name} that is not a count of at least 0"
                )
            return np.asarray(values.to_numpy(), dtype=np.int64)

        tokens = np.stack(
            [
                _stripped(np.asarray(chunk[site].to_list(), dtype=object))
                for site in sites
            ],
            axis=1,
        )
        yield AlleleRows(
            barcodes=_cells(
                np.asarray(chunk["cellBC"].to_list(), dtype=object), rows, layout.path
            ),
            rows=rows,
            columns=tuple((None, site) for site in sites),
            tokens=tokens,
            integrations=_integration_names(
                np.asarray(chunk["intBC"].to_list(), dtype=object), rows, layout.path
            ),
            umis=counts("n_umis"),
            reads=counts("n_reads"),
        )


def _iter_petracer(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    positions = {name: position for position, name in enumerate(text.names)}
    for lines, chunk in text_chunks(text, chunk_rows):
        tokens = np.stack(
            [_stripped(chunk[positions[site]]) for site in _PETRACER_SITES], axis=1
        )
        yield AlleleRows(
            barcodes=_cells(
                chunk[positions["cellBC"]].to_numpy(dtype=object),
                lines,
                text.path,
            ),
            rows=lines,
            columns=tuple((None, site) for site in _PETRACER_SITES),
            tokens=tokens,
            integrations=_integration_names(
                chunk[positions["intID"]].to_numpy(dtype=object), lines, text.path
            ),
            umis=None
            if "UMI" not in positions
            else _counts(
                chunk[positions["UMI"]].to_numpy(dtype=object),
                "UMI",
                lines,
                text.path,
                table=False,
            ),
            reads=None
            if "readCount" not in positions
            else _counts(
                chunk[positions["readCount"]].to_numpy(dtype=object),
                "readCount",
                lines,
                text.path,
                table=False,
            ),
        )


def _iter_matrix(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    columns = layout._read["columns"]
    for lines, chunk in text_chunks(text, chunk_rows):
        tokens = np.stack(
            [_stripped(chunk[position]) for position in range(1, len(text.names))],
            axis=1,
        )
        if layout.format == "laml":
            tokens = _whole_numbers(tokens)
        yield AlleleRows(
            barcodes=_cells(chunk[0].to_numpy(dtype=object), lines, text.path),
            rows=lines,
            columns=columns,
            tokens=tokens,
        )


def _whole_numbers(tokens: np.ndarray) -> np.ndarray:
    """Tokens of whole numbers written with zero fractions, such as 1.0, as integers."""
    values = np.asarray(tokens, dtype=object)
    codes, distinct = pd.factorize(values.reshape(-1), use_na_sentinel=True)
    written = []
    for token in np.asarray(distinct, dtype=object).tolist():
        found = _WHOLE_FLOAT.fullmatch(str(token))
        written.append(str(int(found.group(1))) if found else token)
    result = np.full(values.size, None, dtype=object)
    present = codes >= 0
    result[present] = np.asarray(written, dtype=object)[codes[present]]
    return result.reshape(values.shape)


def _iter_typewriter(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    n_slots = int(layout.n_slots or 0)
    has_umis = bool(layout._read["has_umis"])
    width = 2 + n_slots + (1 if has_umis else 0)
    for numbers, lines in line_chunks(text, chunk_rows):
        fields = list(csv.reader(lines, delimiter=text.delimiter))
        short = [position for position, row in enumerate(fields) if len(row) < width]
        if short:
            raise ValueError(
                f"{text.path} has rows of fewer fields than its header, cut short, on "
                "lines " + ", ".join(str(int(numbers[row])) for row in short[:5])
            )
        extra = [
            position
            for position, row in enumerate(fields)
            if len(row) > width
            and any(value.strip() for value in row[2 + n_slots : len(row) - has_umis])
        ]
        if extra:
            raise ValueError(
                f"{text.path} has rows with an insertion beyond slot {n_slots}, which "
                "its header lacks, on lines "
                + ", ".join(str(int(numbers[row])) for row in extra[:5])
            )
        tokens = np.asarray(
            [[value.strip() for value in row[2 : 2 + n_slots]] for row in fields],
            dtype=object,
        ).reshape(-1, n_slots)
        umis = None
        if has_umis:
            umis = _counts(
                np.asarray([row[-1] for row in fields], dtype=object),
                "nUMI",
                numbers,
                text.path,
                table=False,
            )
        yield AlleleRows(
            barcodes=_cells(
                np.asarray([row[0] for row in fields], dtype=object), numbers, text.path
            ),
            rows=numbers,
            columns=tuple((None, str(slot)) for slot in range(1, n_slots + 1)),
            tokens=tokens,
            integrations=_integration_names(
                np.asarray([row[1] for row in fields], dtype=object), numbers, text.path
            ),
            umis=umis,
        )


def _expanded_chains(
    values: np.ndarray, n_slots: int, path: str, name: str
) -> np.ndarray:
    """Expand each value of an integration column into the tokens of its slots.

    A chain gives its symbols, then empty tokens for the unedited slots after
    it; ``-`` gives every slot an empty token; an empty value and ``NA`` give
    no token, so every slot is missing. Six slots joined by ``|`` give their
    symbols, ``U`` an empty token and ``NA`` none.

    Raises:
        ValueError: If a chain has more symbols than slots, an empty symbol,
            or a ``|`` value another number of slots.
    """
    codes, distinct = pd.factorize(np.asarray(values, dtype=object))
    expanded = np.full((len(distinct), n_slots), None, dtype=object)
    for position, value in enumerate(np.asarray(distinct, dtype=object).tolist()):
        text = str(value).strip()
        if text in ("", "NA"):
            continue
        if text == "-":
            expanded[position] = ""
            continue
        if "|" in text:
            parts = text.split("|")
            if len(parts) != n_slots:
                raise ValueError(
                    f"{path} holds {quoted_value(text)} for {name}, {len(parts)} slots where its "
                    f"arrays have {n_slots}"
                )
            expanded[position] = [
                "" if part == "U" else None if part == "NA" else part for part in parts
            ]
            continue
        parts = text.split("-")
        if len(parts) > n_slots or any(not part for part in parts):
            raise ValueError(
                f"{path} holds {quoted_value(text)} for {name}, which is no chain of at most "
                f"{n_slots} symbols joined by '-'"
            )
        expanded[position] = parts + [""] * (n_slots - len(parts))
    tokens: np.ndarray = expanded[codes]
    return tokens


def _iter_chains(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    positions = {name: position for position, name in enumerate(text.names)}
    integrations = layout._read["integrations"]
    n_slots = int(layout.n_slots or _TYPEWRITER_CHAIN_SLOTS)
    columns = tuple(
        (integration, str(slot))
        for integration in integrations
        for slot in range(1, n_slots + 1)
    )
    for lines, chunk in text_chunks(text, chunk_rows):
        tokens = np.concatenate(
            [
                _expanded_chains(
                    chunk[positions[integration]].to_numpy(dtype=object),
                    n_slots,
                    text.path,
                    integration,
                )
                for integration in integrations
            ],
            axis=1,
        )
        yield AlleleRows(
            barcodes=_cells(chunk[0].to_numpy(dtype=object), lines, text.path),
            rows=lines,
            columns=columns,
            tokens=tokens,
        )


def _iter_treedata(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    table = layout._table
    assert table is not None
    columns = layout._read["columns"]
    start = 1
    for cells, states in table.blocks(chunk_rows):
        rows = np.arange(start, start + len(cells), dtype=np.int64)
        start += len(cells)
        yield AlleleRows(
            barcodes=np.asarray(cells, dtype=object),
            rows=rows,
            columns=columns,
            tokens=_whole_numbers(_stripped(states).reshape(states.shape)),
        )


def _iter_table(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    text = layout._text
    assert text is not None
    roles = layout.columns
    positions = {name: position for position, name in enumerate(text.names)}
    separator = roles.get("separator")
    sites = list(layout._read["sites"])
    # The keys of each row's cell, integration and site, when rows hold one
    # site each, which find a site that the table lists twice.
    keys: list[np.ndarray] = []
    for lines, chunk in text_chunks(text, chunk_rows):
        if "sites" in roles:
            tokens = np.stack(
                [_stripped(chunk[positions[name]]) for name in roles["sites"]], axis=1
            )
            columns = tuple((layout.integration, site) for site in sites)
        elif separator is not None:
            alleles = _stripped(chunk[positions[roles["allele"]]])
            # A row without an allele reads none of its sites.
            split = [
                None
                if value is None or value in layout.rules.missing
                else str(value).split(separator)
                for value in alleles.tolist()
            ]
            if layout.n_slots is not None:
                n_sites = int(layout.n_slots)
                other = [
                    row
                    for row, parts in enumerate(split)
                    if parts is not None and len(parts) > n_sites
                ]
                what = f"more than {n_sites} sites"
            else:
                n_sites = int(layout._read["split_sites"])
                other = [
                    row
                    for row, parts in enumerate(split)
                    if parts is not None and len(parts) != n_sites
                ]
                what = f"another number of sites than the {n_sites} of its first allele"
            if other:
                raise ValueError(
                    f"{text.path} splits alleles into {what} on "
                    + file_lines(lines[other])
                )
            tokens = np.full((len(split), n_sites), None, dtype=object)
            for row, parts in enumerate(split):
                if parts is not None:
                    tokens[row, : len(parts)] = [part.strip() for part in parts]
            columns = tuple(
                (layout.integration, str(site)) for site in range(1, n_sites + 1)
            )
        else:
            tokens = _stripped(chunk[positions[roles["allele"]]]).reshape(-1, 1)
            columns = ((layout.integration, ""),)
        integrations = None
        if layout.integration is None:
            integrations = _integration_names(
                chunk[positions[roles["integration"]]].to_numpy(dtype=object),
                lines,
                text.path,
            )
            columns = tuple((None, site) for _name, site in columns)
        if "site" in roles:
            site_names = _stripped(chunk[positions[roles["site"]]])
            if any(not value for value in site_names.tolist()):
                raise ValueError(f"{text.path} has rows without a site")
            named = (
                np.full(len(chunk), layout.integration, dtype=object)
                if integrations is None
                else integrations
            )
            keys.append(
                pair_keys(
                    _cells(
                        chunk[positions[roles["cell"]]].to_numpy(dtype=object),
                        lines,
                        text.path,
                    ),
                    np.asarray(
                        [
                            f"{integration}\x1f{site}"
                            for integration, site in zip(
                                named.tolist(), site_names.tolist(), strict=True
                            )
                        ],
                        dtype=object,
                    ),
                )
            )
            yield from _site_rows(
                layout, chunk, positions, lines, tokens, integrations, site_names
            )
            continue
        yield AlleleRows(
            barcodes=_cells(
                chunk[positions[roles["cell"]]].to_numpy(dtype=object),
                lines,
                text.path,
            ),
            rows=lines,
            columns=columns,
            tokens=tokens,
            integrations=integrations,
            umis=_table_counts(layout, chunk, positions, "umis", lines),
            reads=_table_counts(layout, chunk, positions, "reads", lines),
        )
    repeated = repeated_keys(keys) if keys else 0
    if repeated:
        raise ValueError(
            f"{text.path} lists {repeated} sites of one cell and integration on more "
            "than one row, so their alleles conflict"
        )


def _table_counts(
    layout: AlleleLayout,
    chunk: pd.DataFrame,
    positions: Mapping[str, int],
    role: str,
    lines: np.ndarray,
) -> np.ndarray | None:
    name = layout.columns.get(role)
    if name is None:
        return None
    return _counts(
        chunk[positions[name]].to_numpy(dtype=object),
        name,
        lines,
        layout.path,
        table=True,
    )


def _site_rows(
    layout: AlleleLayout,
    chunk: pd.DataFrame,
    positions: Mapping[str, int],
    lines: np.ndarray,
    tokens: np.ndarray,
    integrations: np.ndarray | None,
    site_names: np.ndarray,
) -> Iterator[AlleleRows]:
    """Yield the rows of a table whose rows each hold one site, site by site."""
    roles = layout.columns
    cells = _cells(
        chunk[positions[roles["cell"]]].to_numpy(dtype=object), lines, layout.path
    )
    umis = _table_counts(layout, chunk, positions, "umis", lines)
    reads = _table_counts(layout, chunk, positions, "reads", lines)
    codes, distinct = pd.factorize(site_names)
    for position, site in enumerate(np.asarray(distinct, dtype=object).tolist()):
        rows = np.flatnonzero(codes == position)
        yield AlleleRows(
            barcodes=cells[rows],
            rows=lines[rows],
            columns=(
                (None if integrations is not None else layout.integration, str(site)),
            ),
            tokens=tokens[rows].reshape(-1, 1),
            integrations=None if integrations is None else integrations[rows],
            umis=None if umis is None else umis[rows],
            reads=None if reads is None else reads[rows],
        )


def iter_allele_rows(layout: AlleleLayout, chunk_rows: int) -> Iterator[AlleleRows]:
    """Yield a file's rows in chunks of at most ``chunk_rows`` rows.

    A CARLIN pair yields one row per cell barcode, after it has checked that
    its files have equal lines and list each barcode once.

    Raises:
        ValueError: If the file is malformed.
    """
    readers = {
        "carlin": _iter_carlin,
        "darlin": _iter_darlin,
        "cassiopeia": _iter_cassiopeia,
        "cassiopeia_parquet": _iter_parquet,
        "petracer": _iter_petracer,
        "petracer_characters": _iter_matrix,
        "laml": _iter_matrix,
        "typewriter": _iter_typewriter,
        "typewriter_chains": _iter_chains,
        "treedata": _iter_treedata,
        "table": _iter_table,
    }
    yield from readers[layout.format](layout, chunk_rows)


def classified_rows(layout: AlleleLayout, rows: AlleleRows) -> ChunkStates:
    """Return the state of each site of a chunk, by the format's rules.

    Raises:
        ValueError: If a site holds a token that the format does not read,
            naming the file's lines.
    """
    try:
        return chunk_states(
            rows, layout.rules, lengths=layout.lengths, n_slots=layout.n_slots
        )
    except RefusedTokens as refused:
        shown = ", ".join(
            f"{quoted_value(token)} at {site or 'its allele'}"
            for site, token in refused.tokens
        )
        raise ValueError(
            f"{layout.path} holds tokens that {quoted_value(layout.format)} does not read, "
            f"{shown}, on {_where(layout, rows.rows[list(refused.rows)])}"
        ) from None


def _where(layout: AlleleLayout, rows: np.ndarray) -> str:
    """Name up to five rows of a file: their lines in a text file, else positions."""
    if layout.format in _FILE_LINES:
        return file_lines(rows)
    return "row" + file_lines(rows).removeprefix("line")
