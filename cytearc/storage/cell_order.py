"""Shuffle the cells of an import, and find the order of the imported file again.

An import that shuffles cells writes row ``r`` of every per-cell array from
row ``order[r]`` of the imported file, where :func:`shuffled_order` gives the
permutation ``order``. Any contiguous slice of the store is then a uniformly
random sample of its cells. Each cell keeps its row in the imported file, from
0, in the int64 cell column ``sourceRow``, which carries the role attribute
``source_row``; :func:`source_order_rows` sorts the rows of a selection by it,
for exports that restore the file's order. The cell table records the shuffle
in its ``cellOrder`` attribute, ``{"shuffled": true, "seed": <seed>}``, which
:func:`recorded_shuffle_seed` reads.

A store imported in the file's order has neither the column nor the record,
and its rows are already in that order.
"""

from typing import Any

import numpy as np
import zarr

from ..utils.arguments import integer_argument
from ..utils.digest import mix64_array
from .arrays import MetadataBlock, create_streamed_metadata_column
from .layout import column_chunk_rows
from .profiles import StorageProfile
from .types import as_zarr_array

__all__ = [
    "CELL_ORDER_ATTR",
    "SOURCE_ROW",
    "SOURCE_ROW_ROLE",
    "recorded_shuffle_seed",
    "shuffle_seed_argument",
    "shuffled_order",
    "source_order_rows",
    "write_shuffled_cell_order",
]

SOURCE_ROW = "sourceRow"
"""The cell column that holds each cell's row in the imported file."""

SOURCE_ROW_ROLE = "source_row"
"""``role`` attribute that marks a ``sourceRow`` column as the one CyteArc wrote."""

CELL_ORDER_ATTR = "cellOrder"
"""Attribute of the cell table that records how an import shuffled its cells."""

_MAX_SEED = 2**64 - 1
# SplitMix64's increment, which spaces the row counters of the stream.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
# "CellOrdr" in ASCII, mixed into the seed, so the shuffle's stream differs from the other
# counter-based streams that take the same seed.
_SALT = np.uint64(0x43656C6C4F726472)


def shuffle_seed_argument(value: object) -> int:
    """Return ``shuffle_seed`` as a Python integer from 0 to 2**64 - 1.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is negative or above 2**64 - 1.
    """
    return integer_argument(value, "shuffle_seed", minimum=0, maximum=_MAX_SEED)


def shuffled_order(n_rows: int, seed: int) -> np.ndarray:
    """Return the permutation that shuffles ``n_rows`` cells: row ``r`` holds row ``order[r]``.

    Each row index is hashed with SplitMix64's finalizer, keyed by ``seed``,
    and the rows are ranked by their hashes, ties broken by the index. The
    hashes are exact 64-bit integer arithmetic, so a seed gives the same
    permutation on every platform and NumPy version.

    Args:
        n_rows: Number of cells, at least 0.
        seed: The seed of the shuffle, from 0 to 2**64 - 1.

    Returns:
        The source row of each destination row, as int64.

    Raises:
        TypeError: If ``n_rows`` or ``seed`` is not an integer.
        ValueError: If ``n_rows`` is negative, or ``seed`` lies outside its range.
    """
    n_rows = integer_argument(n_rows, "n_rows", minimum=0)
    seed = shuffle_seed_argument(seed)
    key = mix64_array(np.array([seed], dtype=np.uint64) ^ _SALT)
    hashes = mix64_array(
        key ^ mix64_array(np.arange(n_rows, dtype=np.uint64) + _GOLDEN)
    )
    order: np.ndarray = np.argsort(hashes, kind="stable").astype(np.int64, copy=False)
    return order


def write_shuffled_cell_order(
    cells: zarr.Group,
    order: np.ndarray,
    seed: int,
    *,
    profile: StorageProfile | None = None,
) -> None:
    """Write the ``sourceRow`` column and the ``cellOrder`` record of a shuffled import.

    Args:
        cells: The cell table of the new store.
        order: The source row of each cell, as :func:`shuffled_order` returns it.
        seed: The seed that gave ``order``.
        profile: Zarr encoding profile of the import.

    Raises:
        ValueError: If ``order`` is not a one-dimensional integer array.
    """
    rows = np.asarray(order)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError(
            f"A cell order is one integer row per cell; found dtype {rows.dtype} "
            f"and shape {rows.shape}"
        )
    rows = rows.astype(np.int64, copy=False)
    create_streamed_metadata_column(
        cells,
        SOURCE_ROW,
        shape=int(rows.size),
        dtype=np.int64,
        blocks=(MetadataBlock(0, rows, None),),
        chunkSize=column_chunk_rows(int(rows.size), np.dtype(np.int64).itemsize),
        profile=profile,
        attributes={"role": SOURCE_ROW_ROLE},
    )
    cells.attrs[CELL_ORDER_ATTR] = {
        "shuffled": True,
        "seed": shuffle_seed_argument(seed),
    }


def recorded_shuffle_seed(cells: zarr.Group) -> int | None:
    """Return the seed with which an import shuffled a cell table, or None.

    None means the table records no shuffle: its import kept the file's
    order, or wrote the store before imports could shuffle.

    Raises:
        ValueError: If the ``cellOrder`` record is malformed.
    """
    if CELL_ORDER_ATTR not in cells.attrs:
        return None
    record: Any = cells.attrs[CELL_ORDER_ATTR]
    seed = record.get("seed") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or set(record) != {"shuffled", "seed"}
        or record["shuffled"] is not True
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _MAX_SEED
    ):
        raise ValueError(
            f"The cell table's {CELL_ORDER_ATTR} attribute is malformed: "
            f"{record!r}; it must be a mapping of 'shuffled': true and an "
            f"integer 'seed' from 0 to {_MAX_SEED}"
        )
    return seed


def source_order_rows(cells: zarr.Group, selection: np.ndarray) -> np.ndarray:
    """Return the rows that ``selection`` selects, in the order of the imported file.

    The rows are sorted by their ``sourceRow``, ties kept in store order.
    A table without CyteArc's ``sourceRow`` column, one that carries no
    ``source_row`` role, is in the imported file's order already, and its
    rows keep their store order.

    Args:
        cells: A store's cell table.
        selection: One boolean per row of the table.

    Returns:
        The selected rows, as int64 row indices.

    Raises:
        ValueError: If ``selection`` is not one boolean per row, or the
            ``sourceRow`` column does not hold one int64 per row.
    """
    n_rows = int(as_zarr_array(cells["ids"], name="ids").shape[0])
    mask = np.asarray(selection)
    if mask.dtype != np.bool_ or mask.shape != (n_rows,):
        raise ValueError(
            f"A cell selection is one boolean per cell, {n_rows} of them; found "
            f"dtype {mask.dtype} and shape {mask.shape}"
        )
    rows = np.flatnonzero(mask).astype(np.int64, copy=False)
    column = cells.get(SOURCE_ROW)
    if not isinstance(column, zarr.Array) or column.attrs.get("role") != (
        SOURCE_ROW_ROLE
    ):
        return rows
    if column.dtype != np.int64 or column.shape != (n_rows,):
        raise ValueError(
            f"Cell column {SOURCE_ROW!r} must hold one int64 row of the imported "
            f"file for each of the {n_rows} cells; found dtype {column.dtype} and "
            f"shape {column.shape}"
        )
    source = np.asarray(column.get_orthogonal_selection((rows,)))
    ordered: np.ndarray = rows[np.argsort(source, kind="stable")]
    return ordered
