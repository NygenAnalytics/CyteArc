"""Parsing shared by the readers of immune receptor contig files."""

import gzip
import os

import numpy as np
import pandas as pd

from ..repertoire.chains import infer_loci
from ..repertoire.contigs import (
    FIELDS,
    FLAG,
    INTEGER,
    NUMBER,
    file_lines,
    parse_flags,
    parse_integers,
    parse_numbers,
    parse_text,
)

__all__ = ["first_line", "header_fields", "parse_contig_columns"]

_GENE_CALLS = ("v_call", "d_call", "j_call", "c_call")


def first_line(path: str | os.PathLike[str]) -> str:
    """Return the first line of a plain or gzip-compressed text file."""
    with open(path, "rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as text:
        return text.readline().rstrip("\r\n")


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

    Args:
        frame: Raw text of one chunk, with AIRR field names, including
            ``cell_id``.
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
    parsed: dict[str, np.ndarray] = {}
    cell_ids = parse_text(frame["cell_id"].to_numpy(dtype=object))
    missing = np.flatnonzero([value is None for value in cell_ids])
    if len(missing):
        raise ValueError(f"cell_id is missing on {file_lines(missing, first_line)}")
    parsed["cell_id"] = cell_ids
    for name, kind in FIELDS.items():
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
    if infer_locus:
        parsed["locus"] = infer_loci(
            {name: parsed[name] for name in _GENE_CALLS if name in parsed}
        )
    # Object columns keep None for missing text whatever string dtype pandas
    # would infer.
    return pd.DataFrame(
        {
            name: pd.Series(values, dtype=object) if values.dtype == object else values
            for name, values in parsed.items()
        }
    )
