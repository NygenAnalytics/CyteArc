"""Parsing shared by the readers of immune receptor contig files.

Contig files are delimited tables read by :mod:`cytearc.readers._delimited`,
which refuses a row of fewer or more fields than the header names, so a row
cut short is never read with empty fields.
"""

import os
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from ..metadata.records import file_lines, parse_columns, parse_text
from ..repertoire.chains import infer_loci
from ..repertoire.contigs import FIELDS
from ..utils.arguments import quoted_value
from ._delimited import first_lines, split_fields, text_chunks, text_layout

__all__ = [
    "contig_chunks",
    "header_fields",
    "parse_contig_columns",
]

_GENE_CALLS = ("v_call", "d_call", "j_call", "c_call")


def header_fields(
    path: str | os.PathLike[str], separator: str, *, quoted: bool
) -> tuple[str, ...]:
    """Return the stripped fields of a file's header line, repeats included.

    Raises:
        ValueError: If the file has no header line, or is not UTF-8 text.
    """
    lines = first_lines(os.fspath(path), 1)
    if not lines or not lines[0].strip():
        raise ValueError(f"{quoted_value(os.fspath(path))} has no header line")
    return tuple(
        field.strip() for field in split_fields(lines[0], separator, quoted=quoted)
    )


def contig_chunks(
    path: str | os.PathLike[str],
    *,
    what: str,
    columns: Mapping[str, str],
    quoted: bool,
    chunk_rows: int,
    infer_locus: bool,
) -> Iterator[tuple[np.ndarray, pd.DataFrame]]:
    """Parse the contigs of a table in chunks of rows.

    Args:
        path: The table.
        what: What the table is, for the error about an empty file.
        columns: The file's column names, mapped to the AIRR names they take;
            other columns are not read.
        quoted: Whether a field may be quoted.
        chunk_rows: The most rows to parse at once.
        infer_locus: Infer ``locus`` from the gene calls.

    Yields:
        The file line of each contig of the chunk, and its parsed fields, as
        :func:`parse_contig_columns` returns them.

    Raises:
        ValueError: If the file is malformed, a row holds another number of
            fields than the header names, a cell ID is missing, or a field
            holds a value of the wrong kind.
    """
    text = text_layout(os.fspath(path), what=what, quoted=quoted)
    positions = [
        position for position, name in enumerate(text.names) if name in columns
    ]
    names = [columns[text.names[position]] for position in positions]
    for lines, chunk in text_chunks(text, max(1, int(chunk_rows)), columns=positions):
        chunk.columns = names
        yield parse_contig_columns(chunk, lines=lines, infer_locus=infer_locus)


def parse_contig_columns(
    frame: pd.DataFrame,
    *,
    lines: np.ndarray,
    infer_locus: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Parse the AIRR fields of a chunk of contig rows.

    A row whose every field is empty, as spreadsheets end a table, holds no
    contig and is left out.

    Args:
        frame: Raw text of one chunk, one row per contig row of the file,
            with AIRR field names, including ``cell_id``.
        lines: The file line of each row, named in errors.
        infer_locus: Infer ``locus`` from the gene calls; for a file without a
            ``locus`` field.

    Returns:
        The file line of each contig, and ``cell_id`` and each field of
        ``FIELDS`` that the chunk holds, text as Python strings with None
        where missing, flags as int8, counts and positions as int64 and
        numbers as float64.

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
        raise ValueError(f"cell_id is missing on {file_lines(lines[missing])}")
    parsed["cell_id"] = cell_ids
    parsed.update(parse_columns(frame, FIELDS, lines=lines))
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
        return lines[~blank], chunk.loc[~blank].reset_index(drop=True)
    return lines, chunk
