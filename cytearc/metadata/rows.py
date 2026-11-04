import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import zarr

from ..storage.async_execution import zarr_decoding_threads, zarr_io_concurrency
from ..storage.geometry import (
    READ_OBJECT_BYTES,
    ROW_INDEX_BYTES,
    ROW_KEPT_INDEX_BYTES,
    ArrayGeometry,
    array_geometry,
)
from ..storage.partition import (
    checked_indices,
    is_contiguous,
    partition_indices,
    row_band,
)
from ..storage.types import as_zarr_array, read_array
from ..utils.memory_budget import (
    MemoryLedger,
    object_array_bytes,
    python_text_bytes,
)


_SELECTION_INDEX_ARRAYS = 16
# A Python string's header and its slot in an object array, beside up to four
# bytes for each character.
_STRING_BYTES = 64


class _RowReadableMetaData(Protocol):
    N: int

    def _get_array(self, column: str) -> Any: ...

    def _bool_array(self, key: str) -> Any: ...

    def default_block_rows(self, column: str = "I") -> int: ...


def _read_array_rows(array: Any, rows: np.ndarray) -> np.ndarray:
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Metadata row indices must be one-dimensional")
    if indices.size == 0:
        return np.asarray(array[0:0])
    start = int(indices.min())
    stop = int(indices.max()) + 1
    if start < 0 or stop > int(array.shape[0]):
        raise IndexError("Metadata row indices are out of bounds")
    if is_contiguous(indices):
        return np.asarray(array[int(indices[0]) : int(indices[-1]) + 1])

    orthogonal_selection = getattr(array, "get_orthogonal_selection", None)
    if callable(orthogonal_selection):
        try:
            return np.asarray(orthogonal_selection((indices,)))
        except (AttributeError, NotImplementedError, TypeError):
            pass
    coordinate_selection = getattr(array, "get_coordinate_selection", None)
    if callable(coordinate_selection):
        try:
            return np.asarray(coordinate_selection((indices,)))
        except (AttributeError, NotImplementedError, TypeError):
            pass
    return np.asarray(array[indices])


def previous_read_bytes(*stored: int) -> int:
    """Return what the read before a read of stored arrays can still hold beside it.

    ``stored`` holds the bytes of one stored chunk of each array that is read
    one after another, as stored and decoded, as
    :func:`array_row_selection_parts` counts them. A read can hold the largest
    of those chunks beside its own, with the objects that Zarr makes for both
    reads, as ``ArrayGeometry.heldReadBytes`` counts for one array.
    """
    return max(stored, default=0) + READ_OBJECT_BYTES


def array_row_selection_parts(array: Any) -> tuple[int, int]:
    """Return fixed and per-row bytes for one chunk-serial selection."""
    row_values = int(np.prod(tuple(array.shape)[1:], dtype=np.int64))
    row_bytes = max(1, int(np.dtype(array.dtype).itemsize) * row_values)
    index_bytes = np.dtype(np.int64).itemsize
    per_row = 3 * row_bytes + _SELECTION_INDEX_ARRAYS * index_bytes
    geometry = array_geometry(array)
    if geometry is None:
        return 0, int(per_row)
    return geometry.storedChunkBytes(), int(per_row)


def text_width(array: Any) -> int:
    """Return the characters of each value of a stored text array, 0 for other values.

    The store writes text as fixed-width unicode, four bytes per character.
    """
    dtype = np.dtype(array.dtype)
    return int(dtype.itemsize) // 4 if dtype.kind == "U" else 0


def text_row_selection_parts(array: Any) -> tuple[int, int]:
    """Return fixed and per-row bytes of reading stored rows as Python strings.

    The store keeps text at the width of its longest value and decodes whole
    chunks, so one long value widens every row that is read. Each row is held
    as :func:`array_row_selection_parts` counts it, and then becomes a Python
    string of up to four bytes per character.
    """
    fixed, per_row = array_row_selection_parts(array)
    return fixed, per_row + _STRING_BYTES + 4 * text_width(array)


def text_read_bytes(array: Any, n_rows: int) -> int:
    """Return the bytes of reading ``n_rows`` stored rows as Python strings."""
    fixed, per_row = text_row_selection_parts(array)
    return fixed + int(n_rows) * per_row


def read_array_rows_chunkwise(array: Any, rows: np.ndarray) -> np.ndarray:
    """Read distinct selected rows one physical chunk at a time."""
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Metadata row indices must be one-dimensional")
    if indices.size == 0:
        return np.asarray(array[0:0])

    geometry = array_geometry(array)
    if geometry is None:
        checked = checked_indices(indices, limit=int(array.shape[0]), name="rows")
        return _read_array_rows(array, checked)

    row_shape = tuple(int(extent) for extent in array.shape[1:])
    blocks = partition_indices(geometry, 0, indices)
    output = np.empty((indices.size, *row_shape), dtype=np.dtype(array.dtype))
    for block in blocks:
        values = _read_array_rows(array, block.indices)
        if values.shape != (block.indices.size, *row_shape):
            raise ValueError("Metadata row selection returned an invalid shape")
        output[block.destinations] = values
    return output


def operation_concurrency() -> int:
    """The chunks that Zarr decodes at once for the operation under way."""
    return max(1, int(zarr.config.get("async.concurrency")))


def read_whole(
    ledger: MemoryLedger, group: zarr.Group, name: str, *, item: str
) -> np.ndarray:
    """Read the whole array ``name`` of ``group`` within the ledger's room, and keep it charged as ``item``.

    An unsharded array is read in one call, as :func:`~cytearc.storage.types.read_array`
    reads it, at the highest Zarr concurrency up to the operation's whose
    decoded chunks fit the room, and at one chunk at a time when only that
    fits. A sharded array is read a band of inner chunks at a time, so that
    no fetch takes more than one band of a shard. The ledger counts what the
    read holds while it runs, as ``ArrayGeometry.wholeReadBytes`` states, or
    the result beside what the read leaves behind where that is more: a
    chunk on each of the ledger's codec threads, which the read decodes on,
    up to the chunks it decodes. It keeps the result charged and records
    what the read leaves behind.

    Raises:
        MemoryBudgetError: If even a read of one chunk at a time would pass
            the budget.
        TypeError: If ``name`` is a group, not an array.
    """
    array = as_zarr_array(group[name], name=name)
    geometry = array_geometry(array)
    if geometry is None:
        # An array without chunks, such as a scalar, is read as it is.
        values = read_array(group, name) if array.ndim else np.asarray(array[()])
        ledger.hold(item, values.nbytes)
        return values
    n_chunks = -(-geometry.shape[0] // geometry.axisChunk(0)) * geometry.bandChunks()
    threads = ledger.holding_threads(n_chunks)
    held = (
        geometry.heldReadBytes(threads)
        if geometry.shards is None
        else geometry.bandHeldBytes(threads)
    )
    after = geometry.shape[0] * geometry.rowBytes() + held

    def reading(concurrency: int) -> int:
        return max(geometry.wholeReadBytes(concurrency), after)

    concurrency = _fitting_concurrency(ledger, reading)
    ledger.peak(item, reading(concurrency))
    with (
        zarr_io_concurrency(concurrency),
        zarr_decoding_threads(ledger.codec_threads),
    ):
        if geometry.shards is None:
            values = read_array(group, name)
        else:
            values = np.empty(geometry.shape, dtype=array.dtype)
            step = geometry.axisChunk(0)
            for start in range(0, geometry.shape[0], step):
                values[start : start + step] = array[start : start + step]
    ledger.leave(held)
    ledger.hold(item, values.nbytes)
    return values


def _fitting_concurrency(ledger: MemoryLedger, cost: Any) -> int:
    """The highest concurrency, up to the operation's, whose read ``cost(c)`` fits the room; 1 when none does."""
    limit = operation_concurrency()
    room = ledger.room
    if room is None:
        return limit
    for concurrency in range(limit, 1, -1):
        if cost(concurrency) <= room:
            return concurrency
    return 1


def row_read_bytes(geometry: ArrayGeometry | None, rows: np.ndarray) -> int:
    """What reading ``rows``, distinct rows of an array of ``geometry``, a band of chunks at a time holds, with the rows.

    Ordering the rows by band holds their indices as
    ``storage.geometry.ROW_INDEX_BYTES`` counts them. Reading the bands then
    holds each row in the result and the indices it keeps, one band's
    selected rows as Zarr returns them and what reading the band holds; a
    read of several bands holds the band before beside the next, as
    ``ArrayGeometry.bandReadBytes`` counts. An array without chunks is
    counted by its rows' indices alone.
    """
    indices = np.asarray(rows)
    n_rows = int(indices.size)
    if geometry is None:
        return n_rows * ROW_INDEX_BYTES
    step = geometry.axisChunk(0)
    bands = int(indices.max()) // step - int(indices.min()) // step + 1 if n_rows else 1
    return _band_rows_read_bytes(geometry, n_rows, bands)


def _band_rows_read_bytes(geometry: ArrayGeometry, n_rows: int, bands: int) -> int:
    """:func:`row_read_bytes` of ``n_rows`` rows that span ``bands`` bands of chunks."""
    ordering = n_rows * ROW_INDEX_BYTES
    reading = n_rows * (
        geometry.rowBytes() + ROW_KEPT_INDEX_BYTES
    ) + geometry.bandReadBytes(
        min(n_rows, geometry.axisChunk(0)), operation_concurrency(), bands=bands
    )
    return max(ordering, reading)


def rows_read_charge(
    ledger: MemoryLedger,
    array: Any,
    n_rows: int,
    *,
    span: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """What :func:`read_rows` charges ``ledger`` to read ``n_rows`` distinct rows of ``array``, and what that read leaves behind.

    ``span`` gives the first and last of the rows, which set the bands of
    chunks the rows span. Without it, the rows span the fewest bands that
    ``n_rows`` rows can, which no rows of that count are charged less than.
    """
    geometry = array_geometry(array)
    n_rows = int(n_rows)
    if geometry is None:
        return n_rows * ROW_INDEX_BYTES, 0
    step = geometry.axisChunk(0)
    if span is None:
        bands = max(1, -(-n_rows // step))
    else:
        bands = int(span[1]) // step - int(span[0]) // step + 1
    reading = _band_rows_read_bytes(geometry, n_rows, bands)
    if not n_rows:
        return reading, 0
    chunks = min(n_rows, bands) * geometry.bandChunks()
    held = geometry.bandHeldBytes(ledger.holding_threads(chunks))
    return max(reading, n_rows * geometry.rowBytes() + held), held


def read_rows(
    ledger: MemoryLedger, array: Any, rows: np.ndarray, *, item: str
) -> np.ndarray:
    """Read distinct ``rows`` of a stored array a band of chunks at a time, keeping them charged as ``item``.

    The rows are read as :func:`read_array_rows_chunkwise` reads them, in the
    order given. The ledger counts what the read holds, as
    :func:`row_read_bytes` states, or the rows beside what the read leaves
    behind where that is more: a chunk on each of the ledger's codec threads,
    which the read decodes on, up to the chunks of the bands the rows span, as
    :func:`rows_read_charge` states. It keeps the rows charged and records what
    the read leaves behind.

    Raises:
        MemoryBudgetError: If the read would pass the budget.
    """
    indices = np.asarray(rows)
    span = (int(indices.min()), int(indices.max())) if indices.size else None
    reading, held = rows_read_charge(ledger, array, int(indices.size), span=span)
    ledger.peak(item, reading)
    with zarr_decoding_threads(ledger.codec_threads):
        values = read_array_rows_chunkwise(array, indices)
    if held:
        ledger.leave(held)
    ledger.hold(item, values.nbytes)
    return values


def read_text_rows(
    ledger: MemoryLedger, array: Any, rows: np.ndarray, *, item: str
) -> np.ndarray:
    """Read distinct ``rows`` of a stored text array as Python strings, keeping them charged as ``item``.

    The rows are read at their stored width through :func:`read_rows`, then
    measured: their characters and their widest code point give what their
    strings hold, as :func:`~cytearc.utils.memory_budget.python_text_bytes`
    counts, which is charged before they are made. The stored rows are then
    let go, and the strings stay charged.

    Returns:
        An object array of one string for each row.

    Raises:
        MemoryBudgetError: If the read or the strings would pass the budget.
        TypeError: If the array does not store fixed-width text.
    """
    if np.dtype(array.dtype).kind != "U":
        raise TypeError(f"{item} must be stored as text, not as {array.dtype}")
    stored = read_rows(ledger, array, rows, item=item)
    # Each value's length, while the characters are counted.
    ledger.peak(item, 8 * len(stored))
    characters = int(np.char.str_len(stored).sum()) if len(stored) else 0
    widest = int(stored.view(np.uint32).max()) if stored.size else 0
    ledger.hold(
        item,
        object_array_bytes(len(stored))
        + python_text_bytes(len(stored), characters, widest=widest),
    )
    values = stored.astype(object)
    stored_bytes = stored.nbytes
    del stored
    ledger.release(stored_bytes)
    return values


def read_metadata_rows(
    metadata: _RowReadableMetaData,
    column: str,
    rows: np.ndarray,
) -> np.ndarray:
    """Read selected metadata rows without expanding scattered ranges."""
    return _read_array_rows(metadata._get_array(column), rows)


def read_metadata_rows_chunkwise(
    metadata: _RowReadableMetaData,
    column: str,
    rows: np.ndarray,
) -> np.ndarray:
    """Read distinct metadata rows one physical chunk at a time."""
    return read_array_rows_chunkwise(metadata._get_array(column), rows)


def iter_metadata_column_blocks(
    metadata: _RowReadableMetaData,
    column: str,
    *,
    block_rows: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield every value in a metadata column through bounded slices."""
    array = metadata._get_array(column)
    requested_rows = (
        metadata.default_block_rows(column) if block_rows is None else int(block_rows)
    )
    if requested_rows < 1:
        raise ValueError("block_rows must be >= 1")
    chunk_rows = row_band(
        array_geometry(array),
        unit="chunk",
        fallback=metadata.default_block_rows(column),
    )
    resolved_rows = min(requested_rows, chunk_rows)
    for chunk_start in range(0, metadata.N, chunk_rows):
        chunk_stop = min(chunk_start + chunk_rows, metadata.N)
        for start in range(chunk_start, chunk_stop, resolved_rows):
            stop = min(start + resolved_rows, chunk_stop)
            yield np.asarray(array[start:stop])


def metadata_missing_mask(metadata: Any, column: str) -> Any | None:
    """Return a column's internal missing-mask array when one exists."""
    get_mask = getattr(metadata, "_get_missing_mask_array", None)
    if not callable(get_mask):
        return None
    return get_mask(column)


def metadata_column_fingerprint(metadata: _RowReadableMetaData, column: str) -> str:
    """Hash one metadata column in bounded blocks.

    The digest covers the stored dtype, the type and representation of each
    object value, and the linked missing mask, so an edit to any of them
    changes it.
    """
    digest = hashlib.sha256()
    for block in iter_metadata_column_blocks(metadata, column):
        digest.update(str(block.dtype).encode())
        digest.update(str(block.shape).encode())
        digest.update(
            json.dumps(
                [(type(value).__name__, repr(value)) for value in block.tolist()]
            ).encode()
            if block.dtype.hasobject
            else block.tobytes()
        )
    missing = metadata_missing_mask(metadata, column)
    digest.update(b"missing:none" if missing is None else b"missing:present")
    if missing is not None:
        # Zarr arrays have no len(); the mask has one row per column row.
        for start in range(0, int(missing.shape[0]), 65_536):
            digest.update(
                np.asarray(missing[start : start + 65_536], dtype=bool).tobytes()
            )
    return digest.hexdigest()


def read_metadata_missing_rows(
    metadata: Any,
    column: str,
    rows: np.ndarray,
) -> np.ndarray | None:
    """Read a column's internal missing mask for selected rows."""
    mask = metadata_missing_mask(metadata, column)
    if mask is None:
        return None
    return np.asarray(_read_array_rows(mask, rows), dtype=bool)


def read_metadata_missing_rows_chunkwise(
    metadata: Any,
    column: str,
    rows: np.ndarray,
) -> np.ndarray | None:
    """Read a missing mask one physical chunk at a time."""
    mask = metadata_missing_mask(metadata, column)
    if mask is None:
        return None
    return np.asarray(read_array_rows_chunkwise(mask, rows), dtype=bool)


def apply_missing_mask(
    values: np.ndarray,
    missing: np.ndarray | None,
    *,
    labels: bool = False,
) -> np.ndarray:
    """Show rows flagged by a linked missing mask as missing values.

    Nullable columns and artifacts store a placeholder in each masked row. By
    default, masked integer and float rows become NaN in a float64 copy, masked
    boolean rows become False so that boolean filters exclude them, masked
    datetime and timedelta rows become NaT in a copy of their dtype, and other
    masked rows become None in an object copy. With ``labels=True``, every
    masked row becomes None in an object copy, so categorical labels keep their
    values. ``values`` is returned unchanged when no row is masked.
    """
    array = np.asarray(values)
    if missing is None:
        return array
    mask = np.asarray(missing, dtype=bool)
    if mask.shape != array.shape:
        raise ValueError("Missing mask does not align with its values")
    if not mask.any():
        return array
    kind = "O" if labels else array.dtype.kind
    if kind == "b":
        output = array.copy()
        output[mask] = False
    elif kind in {"f", "i", "u"}:
        output = array.astype(np.float64, copy=True)
        output[mask] = np.nan
    elif kind in {"M", "m"}:
        output = array.copy()
        output[mask] = np.array("NaT", dtype=array.dtype)[()]
    else:
        output = array.astype(object, copy=True)
        output[mask] = None
    return output


@dataclass(frozen=True, slots=True)
class MetaDataRowBlock:
    """One contiguous slice of a metadata table for blockwise scans."""

    start: int
    stop: int
    active_global_indices: np.ndarray
    values: dict[str, np.ndarray]


def array_block_rows(array: Any, n_rows: int) -> int:
    """Return a row block size aligned with ``array``'s chunks.

    Arrays without chunk geometry use at most 100,000 of their ``n_rows`` rows.
    """
    return row_band(array_geometry(array), unit="chunk", fallback=min(n_rows, 100_000))


def default_block_rows(metadata: _RowReadableMetaData, column: str = "I") -> int:
    """Return a row block size aligned with the backing Zarr chunks."""
    return array_block_rows(metadata._get_array(column), metadata.N)


def iter_row_blocks(
    metadata: _RowReadableMetaData,
    *,
    cell_key: str = "I",
    columns: Iterable[str] | None = None,
    block_rows: int | None = None,
) -> Iterator[MetaDataRowBlock]:
    """Yield contiguous active row blocks from a metadata table."""
    key_array = metadata._bool_array(cell_key)
    if block_rows is None:
        block_rows = array_block_rows(key_array, metadata.N)
    if block_rows < 1:
        raise ValueError("block_rows must be >= 1")
    column_arrays = {
        column: metadata._get_array(column)
        for column in ([] if columns is None else list(columns))
    }

    for start in range(0, metadata.N, block_rows):
        stop = min(start + block_rows, metadata.N)
        key_slice = np.asarray(key_array[start:stop], dtype=bool)
        local_indices = np.flatnonzero(key_slice)
        active_global_indices = (local_indices + start).astype(
            np.int64,
            copy=False,
        )
        values: dict[str, np.ndarray] = {}
        for column, array in column_arrays.items():
            block = np.asarray(array[start:stop])
            values[column] = block[local_indices]
        yield MetaDataRowBlock(
            start=start,
            stop=stop,
            active_global_indices=active_global_indices,
            values=values,
        )
