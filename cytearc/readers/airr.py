"""AIRR Rearrangement files of single-cell immune receptor libraries.

The AIRR Community's Rearrangement schema (Vander Heiden et al. 2018) holds one
row per rearrangement, here one contig of a cell's receptor chain, in a
tab-separated file with a header of field names. True and false are written
``T`` and ``F`` and a missing value is empty. Values are not quoted, so a
quote is read as part of its value, also where a file's writer quoted it, as
the airr library's writer quotes a value that holds a double quote. A
single-cell file joins each row to its cell through ``cell_id``. Cell Ranger writes such a file,
``airr_rearrangement.tsv``, and other pipelines that follow the schema read
the same when they hold the fields that an import needs.
"""

import os
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..repertoire.contigs import FIELDS, REQUIRED_FIELDS
from ._contigs import contig_chunks, header_fields
from ._delimited import first_lines
from ..utils.arguments import quoted_value

__all__ = [
    "AIRR_FORMAT",
    "airr_columns",
    "contig_file_format",
    "iter_airr_rearrangements",
]

#: The format name of AIRR Rearrangement files.
AIRR_FORMAT = "airr"

_GENE_CALLS = ("v_call", "d_call", "j_call", "c_call")


def contig_file_format(path: str | os.PathLike[str]) -> str:
    """Tell an AIRR Rearrangement file from Cell Ranger's contig annotations.

    An AIRR file is tab-separated and its header names ``sequence_id``; Cell
    Ranger's contig annotations are comma-separated and their header names
    ``barcode`` and ``contig_id``.

    Args:
        path: A plain or gzip-compressed contig file.

    Returns:
        ``"airr"`` or ``"cellranger"``.

    Raises:
        ValueError: If the header is neither.
    """
    lines = first_lines(os.fspath(path), 1)
    header = lines[0] if lines else ""
    tab_fields = {field.strip() for field in header.split("\t")}
    comma_fields = {field.strip() for field in header.split(",")}
    if "sequence_id" in tab_fields:
        return AIRR_FORMAT
    if {"barcode", "contig_id"} <= comma_fields:
        return "cellranger"
    raise ValueError(
        f"{quoted_value(os.fspath(path))} is neither an AIRR Rearrangement file, a "
        "tab-separated file whose header names sequence_id, nor Cell Ranger's "
        "contig annotations, a comma-separated file whose header names barcode "
        "and contig_id"
    )


def airr_columns(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Return the header fields of an AIRR Rearrangement file, as written.

    Raises:
        ValueError: If the file has no header line.
    """
    return header_fields(path, "\t", quoted=False)


def iter_airr_rearrangements(
    path: str | os.PathLike[str],
    *,
    chunk_rows: int = 50_000,
) -> Iterator[tuple[np.ndarray, pd.DataFrame]]:
    """Parse an AIRR Rearrangement file in chunks of rows.

    A plain or gzip-compressed file is read, without quoting: the AIRR file
    format quotes no value, so quotes are read as part of their values. Fields
    outside ``FIELDS`` are not kept, blank lines hold no rearrangement, and a
    row of fewer or more fields than the header is refused. A file without
    ``locus`` has each contig's locus inferred from its gene calls
    (:func:`cytearc.repertoire.chains.infer_loci`).

    Args:
        path: The AIRR Rearrangement file.
        chunk_rows: The most rows to parse at once.

    Yields:
        The file line of each rearrangement of a chunk and the chunk's parsed
        fields: ``cell_id`` and each field of ``FIELDS`` that the file holds,
        text as Python strings with None where missing, flags as int8, counts
        and positions as int64 and numbers as float64.

    Raises:
        ValueError: If the file lacks ``cell_id`` or a required field, a field
            is repeated, a row holds another number of fields than the header,
            or a value is of the wrong kind.
    """
    columns = airr_columns(path)
    repeated = sorted({name for name in columns if columns.count(name) > 1})
    if repeated:
        raise ValueError(
            "The AIRR file repeats the fields " + ", ".join(map(repr, repeated))
        )
    if "cell_id" not in columns:
        raise ValueError(
            "The AIRR file has no cell_id field, so its rearrangements cannot be "
            "joined to cells; single-cell AIRR files name each row's cell in "
            "cell_id"
        )
    infer_locus = "locus" not in columns
    required = [name for name in REQUIRED_FIELDS if name != "locus"]
    missing = [name for name in required if name not in columns]
    if infer_locus and not any(name in columns for name in _GENE_CALLS):
        missing.append("locus")
    if missing:
        raise ValueError(
            "The AIRR file lacks the fields "
            + ", ".join(quoted_value(name) for name in missing)
            + "; CyteArc needs "
            + ", ".join(REQUIRED_FIELDS)
            + ", with locus inferred from v_call, d_call, j_call or c_call when "
            "the file has none"
        )
    kept = ("cell_id", *FIELDS)
    # The AIRR file format quotes no value, so a quote is part of its value,
    # also at the start of one.
    yield from contig_chunks(
        path,
        what="an AIRR Rearrangement file",
        columns={name: name for name in kept},
        quoted=False,
        chunk_rows=chunk_rows,
        infer_locus=infer_locus,
    )
