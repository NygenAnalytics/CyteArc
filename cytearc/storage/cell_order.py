"""Shuffle the cells of an import, and find the order of the imported file again.

An import that shuffles cells writes row ``r`` of every per-cell array from
cell ``order[r]`` of the cells it imports, where :func:`shuffled_order` gives
the permutation ``order``. Any contiguous slice of the store is then a
uniformly random sample of its cells. Each cell keeps its row in the imported
file, from 0, in the int64 cell column ``sourceRow``, which carries the role
attribute ``source_row``. A reader that keeps only some of the file's cells,
such as one that filters barcodes, leaves gaps in those rows.
:func:`source_order_rows` sorts the rows of a selection by them, for exports
that restore the file's order. The cell table records the shuffle
in its ``cellOrder`` attribute, ``{"shuffled": true, "seed": <seed>}``, which
:func:`recorded_shuffle_seed` reads.

A store imported in the file's order has neither the column nor the record,
and its rows are already in that order. A merge writes ``sourceRow`` as each
cell's row in the concatenation of its sources, and a subset copies it, so
exports of either restore the order of the files they came from. Since such
readers and subsets leave gaps in the values, they are only sorted and ranked,
never taken as positions.
"""

from typing import Any, Literal

import numpy as np
import zarr

from ..utils.arguments import integer_argument, quoted_value
from ..utils.digest import mix64_array
from .arrays import MetadataBlock, create_streamed_metadata_column
from .layout import column_chunk_rows
from .profiles import StorageProfile
from .types import as_zarr_array

__all__ = [
    "CELL_ORDERS",
    "CELL_ORDER_ATTR",
    "SOURCE_ROW",
    "SOURCE_ROW_ROLE",
    "CellOrder",
    "cell_order_argument",
    "copy_cell_order",
    "destination_rows",
    "exported_cell_order",
    "is_source_row_column",
    "recorded_shuffle_seed",
    "shuffle_seed_argument",
    "shuffled_order",
    "source_order_positions",
    "source_order_ranks",
    "source_order_rows",
    "write_shuffled_cell_order",
]

type CellOrder = Literal["source", "stored"]
"""The order in which an export writes cells: the imported file's, or the store's."""

CELL_ORDERS: tuple[CellOrder, ...] = ("source", "stored")

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


def cell_order_argument(value: object) -> CellOrder:
    """Return an export's ``cell_order``, ``"source"`` or ``"stored"``.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is another string.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"cell_order must be 'source' or 'stored', not {type(value).__name__}"
        )
    if value == "source":
        return "source"
    if value == "stored":
        return "stored"
    raise ValueError(
        "cell_order must be 'source', the order of the imported file, or "
        f"'stored', the order of the store, not {quoted_value(value)}"
    )


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


def destination_rows(order: np.ndarray) -> np.ndarray:
    """Return the destination row of each source row, the inverse of ``order``.

    Source row ``order[r]`` goes to row ``r``, so an import that reads its
    rows in the file's order sends row ``s`` to ``destination_rows(order)[s]``.

    Raises:
        ValueError: If ``order`` is not a one-dimensional integer array.
    """
    rows = np.asarray(order)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError(
            f"A cell order is one integer row per cell; found dtype {rows.dtype} "
            f"and shape {rows.shape}"
        )
    destinations = np.empty(rows.size, dtype=np.int64)
    destinations[rows] = np.arange(rows.size, dtype=np.int64)
    return destinations


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
        order: Each cell's row in the imported file: the order that
            :func:`shuffled_order` returns when the import keeps every row
            of the file, or the file rows of the cells it keeps, taken in
            that order, when it keeps only some.
        seed: The seed that gave the order.
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


def is_source_row_column(array: object) -> bool:
    """Return whether a cell column is CyteArc's ``sourceRow``, which carries the ``source_row`` role.

    A plain column of that name, such as one a store held before imports
    could shuffle, is not.
    """
    return isinstance(array, zarr.Array) and array.attrs.get("role") == (
        SOURCE_ROW_ROLE
    )


def _source_rows(cells: zarr.Group, rows: np.ndarray) -> np.ndarray | None:
    """Return the ``sourceRow`` of store rows ``rows``, or None without the column.

    Raises:
        ValueError: If the column does not hold one int64 per row.
    """
    n_rows = int(as_zarr_array(cells["ids"], name="ids").shape[0])
    column = cells.get(SOURCE_ROW)
    if not is_source_row_column(column):
        return None
    assert isinstance(column, zarr.Array)
    if column.dtype != np.int64 or column.shape != (n_rows,):
        raise ValueError(
            f"Cell column {SOURCE_ROW!r} must hold one int64 row of the imported "
            f"file for each of the {n_rows} cells; found dtype {column.dtype} and "
            f"shape {column.shape}"
        )
    if rows.size == n_rows:
        # The rows of every cell, in store order.
        return np.asarray(column[...])
    return np.asarray(column.get_orthogonal_selection((rows,)))


def source_order_ranks(source_rows: np.ndarray) -> np.ndarray:
    """Return the rank of each cell in the order of its ``source_rows``, ties in their order.

    The cells are given in store order, so ``ranks[i]`` is where the cell of
    ``source_rows[i]`` lies in an export in source order.

    Raises:
        ValueError: If ``source_rows`` is not a one-dimensional integer array.
    """
    values = np.asarray(source_rows)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(
            f"Source rows are one integer per cell; found dtype {values.dtype} and "
            f"shape {values.shape}"
        )
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size, dtype=np.int64)
    return ranks


def source_order_positions(cells: zarr.Group, rows: np.ndarray) -> np.ndarray | None:
    """Return the order in which an export in source order writes the cells at store rows ``rows``.

    ``rows`` are in store order, and ``rows[positions]`` are the same rows
    sorted by their ``sourceRow``, ties kept in store order. None means that
    store order is already source order: the table has no ``sourceRow``
    column that carries the ``source_row`` role, or the rows' source rows
    never decrease.

    Args:
        cells: A store's cell table.
        rows: Store rows, in increasing order.

    Returns:
        Positions in ``rows``, as int64, or None.

    Raises:
        ValueError: If ``rows`` are not increasing rows of the table, or the
            ``sourceRow`` column does not hold one int64 per row.
    """
    n_rows = int(as_zarr_array(cells["ids"], name="ids").shape[0])
    selected = np.asarray(rows)
    if (
        selected.ndim != 1
        or (selected.size and selected.dtype.kind not in "iu")
        or (selected.size and (int(selected[0]) < 0 or int(selected[-1]) >= n_rows))
        or bool(np.any(np.diff(selected) <= 0))
    ):
        raise ValueError(
            f"Exported rows are increasing integer rows of the {n_rows} cells; "
            f"found dtype {selected.dtype} and shape {selected.shape}"
        )
    source = _source_rows(cells, selected.astype(np.int64, copy=False))
    if source is None or bool(np.all(source[1:] >= source[:-1])):
        return None
    positions: np.ndarray = np.argsort(source, kind="stable").astype(
        np.int64, copy=False
    )
    return positions


def exported_cell_order(
    cells: zarr.Group, rows: np.ndarray, cell_order: object
) -> np.ndarray | None:
    """Return the order in which an export writes the cells at store rows ``rows``.

    With ``cell_order="source"``, the export writes the cells in the order
    of the imported file, as :func:`source_order_positions` gives it; None
    means that the export keeps store order, as it always does with
    ``"stored"``.

    Raises:
        TypeError: If ``cell_order`` is not a string.
        ValueError: If ``cell_order`` is neither ``"source"`` nor
            ``"stored"``, or as :func:`source_order_positions` raises.
    """
    if cell_order_argument(cell_order) == "stored":
        return None
    return source_order_positions(cells, rows)


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
    positions = source_order_positions(cells, rows)
    return rows if positions is None else rows[positions]


def copy_cell_order(source: zarr.Group, target: zarr.Group) -> None:
    """Copy the ``cellOrder`` record of a cell table to a table of some of its rows.

    A subset, repack or mount keeps the store's order of rows, so the rows
    it copies stay in the random order that the record states. A table
    without the record gives none.

    Raises:
        ValueError: If the source's record is malformed.
    """
    seed = recorded_shuffle_seed(source)
    if seed is not None:
        target.attrs[CELL_ORDER_ATTR] = {"shuffled": True, "seed": seed}
