"""Write clone assignments as files that lineage tracing tools read.

``write_clone_table`` writes CoSpar's clone table, ``Cell_ID,Clone_ID``, one
row per cell with a clone. ``write_clone_matrix`` writes a cells by clones
Matrix Market file with ``cells.tsv`` and ``clones.tsv`` beside it, which
name its rows and columns. ``ds.clones.import_clones`` reads both back.
"""

import csv
import gzip
import io
import os
from pathlib import Path

import numpy as np

__all__ = ["write_clone_matrix", "write_clone_table"]

# Rows written to the text at a time.
_BLOCK_ROWS = 100_000


def _open_output(path: str | os.PathLike[str]) -> io.TextIOBase:
    """Open a text file for writing, gzip-compressed when its name ends in .gz."""
    if os.fspath(path).endswith(".gz"):
        return io.TextIOWrapper(
            gzip.open(path, "wb", compresslevel=6), encoding="utf-8", newline=""
        )
    return open(path, "w", encoding="utf-8", newline="")


def _checked(cell_ids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cell_ids, dtype=object)
    clones = np.asarray(labels, dtype=object)
    if cells.ndim != 1 or clones.shape != cells.shape:
        raise ValueError("Clone tables need one clone label per cell")
    return cells, clones


def write_clone_table(
    path: str | os.PathLike[str], cell_ids: np.ndarray, labels: np.ndarray
) -> int:
    """Write CoSpar's clone table: a CSV of ``Cell_ID`` and ``Clone_ID``.

    Args:
        path: The file to write, gzip-compressed when it ends in ``.gz``.
        cell_ids: The cells with a clone.
        labels: Each cell's clone label.

    Returns:
        The number of cells written.

    Raises:
        ValueError: If the cells and labels differ in length.
    """
    cells, clones = _checked(cell_ids, labels)
    with _open_output(path) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Cell_ID", "Clone_ID"])
        for start in range(0, len(cells), _BLOCK_ROWS):
            stop = start + _BLOCK_ROWS
            writer.writerows(
                zip(
                    cells[start:stop].tolist(), clones[start:stop].tolist(), strict=True
                )
            )
    return len(cells)


def _write_list(path: Path, name: str, values: np.ndarray) -> None:
    # Every field is quoted, so a reader finds no delimiter inside a label.
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow([name])
        for start in range(0, len(values), _BLOCK_ROWS):
            writer.writerows(
                [value] for value in values[start : start + _BLOCK_ROWS].tolist()
            )


def write_clone_matrix(
    path: str | os.PathLike[str],
    cell_ids: np.ndarray,
    codes: np.ndarray,
    clone_ids: np.ndarray,
) -> int:
    """Write a cells by clones Matrix Market file with its cell and clone lists.

    The matrix holds a 1 where a cell belongs to a clone. ``cells.tsv`` and
    ``clones.tsv``, written beside it and replacing files of those names,
    hold the matrix's cell IDs and clone labels in order, each under a header.

    Args:
        path: The Matrix Market file, gzip-compressed when it ends in ``.gz``.
        cell_ids: The cells with a clone.
        codes: Each cell's clone, as a position in ``clone_ids``.
        clone_ids: The clone labels.

    Returns:
        The number of cells written.

    Raises:
        ValueError: If the cells and codes differ in length, a code lies
            outside the clones, or a cell ID or clone label holds a line
            break, which the lists cannot hold.
    """
    cells = np.asarray(cell_ids, dtype=object)
    positions = np.asarray(codes, dtype=np.int64)
    labels = np.asarray(clone_ids, dtype=object)
    if cells.ndim != 1 or positions.shape != cells.shape:
        raise ValueError("A clone matrix needs one clone per cell")
    if len(positions) and (positions.min() < 0 or positions.max() >= len(labels)):
        raise ValueError(f"Clone positions must lie in [0, {len(labels)})")
    # The lists hold one name per line, as their readers count them.
    for what, names in (("cell ID", cells), ("clone label", labels)):
        broken = next(
            (
                str(name)
                for name in names.tolist()
                if "\n" in str(name) or "\r" in str(name)
            ),
            None,
        )
        if broken is not None:
            raise ValueError(
                f"The {what} {broken!r} holds a line break, which the matrix's "
                "lists cannot hold; write the clones as a clone table instead"
            )
    folder = Path(os.fspath(path)).parent
    with _open_output(path) as handle:
        handle.write("%%MatrixMarket matrix coordinate integer general\n")
        handle.write("% cells by clones, written by CyteArc\n")
        handle.write(f"{len(cells)} {len(labels)} {len(cells)}\n")
        for start in range(0, len(cells), _BLOCK_ROWS):
            rows = np.arange(start, min(start + _BLOCK_ROWS, len(cells))) + 1
            columns = positions[start : start + _BLOCK_ROWS] + 1
            handle.write(
                "".join(
                    f"{row} {column} 1\n"
                    for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
                )
            )
    _write_list(folder / "cells.tsv", "cell", cells)
    _write_list(folder / "clones.tsv", "clone", labels)
    return len(cells)
