"""Parsing shared by the readers of immune receptor contig files."""

import gzip
import os

import numpy as np
import pandas as pd

from ..metadata.records import file_lines, parse_columns, parse_text
from ..repertoire.chains import infer_loci
from ..repertoire.contigs import FIELDS

__all__ = [
    "compression",
    "first_line",
    "header_fields",
    "parse_contig_columns",
    "read_options",
]

_GENE_CALLS = ("v_call", "d_call", "j_call", "c_call")


def compression(path: str | os.PathLike[str]) -> str | None:
    """Return ``"gzip"`` for a gzip-compressed file, by its first bytes, else None.

    The bytes decide rather than the file's suffix, so a compressed file
    without ``.gz`` reads as well.
    """
    with open(path, "rb") as handle:
        return "gzip" if handle.read(2) == b"\x1f\x8b" else None


def first_line(path: str | os.PathLike[str]) -> str:
    """Return the first line of a plain or gzip-compressed text file.

    A UTF-8 byte order mark, which spreadsheet programs write, is not part of
    the line.
    """
    opener = gzip.open if compression(path) == "gzip" else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as text:
        return text.readline().rstrip("\r\n")


def read_options(path: str | os.PathLike[str]) -> dict[str, object]:
    """Return the options of ``pandas.read_csv`` that every contig reader shares.

    Every value is read as text, empty where a field is empty, and the first
    column is never taken as an index, so a separator at the end of every
    row leaves the columns in place. Blank lines are read as rows, so that
    the rows of a chunk follow the file's lines.
    """
    return {
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "index_col": False,
        "skip_blank_lines": False,
        "compression": compression(path),
    }


def header_fields(path: str | os.PathLike[str], separator: str) -> tuple[str, ...]:
    """Return the stripped fields of a file's header line, repeats included.

    Raises:
        ValueError: If the file has no header line.
    """
    header = first_line(path)
    if not header.strip():
        raise ValueError(f"{os.fspath(path)!r} has no header line")
    return tuple(field.strip() for field in header.split(separator))


def parse_contig_columns(
    frame: pd.DataFrame,
    *,
    first_line: int,
    infer_locus: bool,
) -> pd.DataFrame:
    """Parse the AIRR fields of a chunk of contig rows.

    A row whose every field is empty, as a blank line reads, holds no contig
    and is left out.

    Args:
        frame: Raw text of one chunk, one row per file line, with AIRR field
            names, including ``cell_id``.
        first_line: The file line of the chunk's first row, named in errors.
        infer_locus: Infer ``locus`` from the gene calls; for a file without a
            ``locus`` field.

    Returns:
        ``cell_id`` and each field of ``FIELDS`` that the chunk holds, text as
        Python strings with None where missing, flags as int8, counts and
        positions as int64 and numbers as float64.

    Raises:
        ValueError: If a cell ID is missing, or a field holds a value of the
            wrong kind.
    """
    raw_ids = frame["cell_id"].to_numpy(dtype=object)
    # Only a row without a cell ID can be blank, so the other fields of most
    # chunks are never compared.
    blank = np.zeros(len(frame), dtype=bool)
    candidates = np.flatnonzero(raw_ids == "")
    if len(candidates):
        blank[candidates] = (frame.iloc[candidates] == "").to_numpy().all(axis=1)
    parsed: dict[str, np.ndarray] = {}
    cell_ids = parse_text(raw_ids)
    missing = np.flatnonzero(
        np.fromiter((value is None for value in cell_ids), bool, len(cell_ids)) & ~blank
    )
    if len(missing):
        raise ValueError(f"cell_id is missing on {file_lines(missing, first_line)}")
    parsed["cell_id"] = cell_ids
    parsed.update(parse_columns(frame, FIELDS, first_line=first_line))
    if infer_locus:
        parsed["locus"] = infer_loci(
            {name: parsed[name] for name in _GENE_CALLS if name in parsed}
        )
    # Object columns keep None for missing text whatever string dtype pandas
    # would infer.
    chunk = pd.DataFrame(
        {
            name: pd.Series(values, dtype=object) if values.dtype == object else values
            for name, values in parsed.items()
        }
    )
    if blank.any():
        chunk = chunk.loc[~blank].reset_index(drop=True)
    return chunk
