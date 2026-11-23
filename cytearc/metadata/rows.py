import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import zarr

from ..storage.async_execution import zarr_decoding_threads, zarr_io_concurrency
from ..storage.categorical import CategoricalColumn
from ..storage.geometry import (
    ROW_INDEX_BYTES,
    ROW_KEPT_INDEX_BYTES,
    ArrayGeometry,
    TextExtent,
    array_geometry,
    text_extent,
)
from ..storage.partition import (
    checked_indices,
    is_contiguous,
    partition_indices,
    read_variable_text_rows,
    row_band,
)
from ..storage.artifacts import fingerprint_text_blocks
from ..storage.content_stamps import (
    stored_content_stamp,
    stored_content_stamp_bytes,
)
from ..storage.layout import column_chunk_rows
from ..storage.types import as_zarr_array, read_array
from ..utils.memory_budget import (
    MemoryLedger,
    object_array_bytes,
    python_text_bytes,
)


_SELECTION_INDEX_ARRAYS = 16
# A Python string's header and its slot in an object array, beside up to four
# bytes for each character.
STRING_BYTES = 64


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
    if getattr(getattr(array, "dtype", None), "kind", None) == "T":
        return read_variable_text_rows(array, indices)

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


def array_row_selection_parts(
    array: Any, *, ledger: MemoryLedger | None = None
) -> tuple[int, int]:
    """Return fixed and per-row bytes for one chunk-serial selection.

    A stored array counts a chunk as stored and decoded, and each row three
    times beside its indices, at the bytes that ``ArrayGeometry`` gives its
    values; variable-width text counts each value as
    :func:`~cytearc.storage.geometry.variable_text_extent` measures the
    longest, and the measure among the fixed bytes, charged to ``ledger``.
    """
    return _row_selection_parts(
        array, array_geometry(array, measure_text=True, ledger=ledger), ledger
    )


def _row_selection_parts(
    array: Any, geometry: ArrayGeometry | None, ledger: MemoryLedger | None
) -> tuple[int, int]:
    """:func:`array_row_selection_parts` of ``array``, whose geometry, its text measured, is ``geometry``."""
    row_values = int(np.prod(tuple(array.shape)[1:], dtype=np.int64))
    extent = (
        text_extent(array, geometry, ledger)
        if np.dtype(array.dtype).kind == "T"
        else None
    )
    if geometry is not None:
        value_bytes = geometry.itemsize
    elif extent is not None:
        value_bytes = extent.valueBytes()
    else:
        value_bytes = int(np.dtype(array.dtype).itemsize)
    row_bytes = max(1, value_bytes * row_values)
    index_bytes = np.dtype(np.int64).itemsize
    per_row = 3 * row_bytes + _SELECTION_INDEX_ARRAYS * index_bytes
    fixed = 0 if geometry is None else geometry.storedChunkBytes()
    if extent is not None:
        fixed = max(fixed, extent.measureBytes)
    return fixed, int(per_row)


def text_width(array: Any, *, ledger: MemoryLedger | None = None) -> int:
    """Return the characters of each value of a stored text array, 0 for other values.

    The store writes text as fixed-width unicode, four bytes per character.
    Text of variable width, which another tool can write, such as Zarr's
    ``dtype=str``, takes the characters of its longest value, as
    :func:`~cytearc.storage.geometry.variable_text_extent` measures them,
    charged to ``ledger``.
    """
    dtype = np.dtype(array.dtype)
    if dtype.kind == "U":
        return int(dtype.itemsize) // 4
    if dtype.kind != "T":
        return 0
    return text_extent(array, None, ledger).characters


def text_row_selection_parts(
    array: Any, *, ledger: MemoryLedger | None = None
) -> tuple[int, int]:
    """Return fixed and per-row bytes of reading stored rows as Python strings.

    The store keeps text at the width of its longest value and decodes whole
    chunks, so one long value widens every row that is read. Each row is held
    as :func:`array_row_selection_parts` counts it, and then becomes a Python
    string of up to four bytes per character, or, for variable-width text, of
    the bytes per character of its widest character.
    """
    if np.dtype(array.dtype).kind != "T":
        fixed, per_row = array_row_selection_parts(array)
        return fixed, per_row + STRING_BYTES + 4 * text_width(array)
    geometry = array_geometry(array, measure_text=True, ledger=ledger)
    fixed, per_row = _row_selection_parts(array, geometry, ledger)
    extent = text_extent(array, geometry, ledger)
    string = object_array_bytes(1) + python_text_bytes(
        1, extent.characters, widest=extent.widest
    )
    return fixed, per_row + string


def text_read_bytes(
    array: Any, n_rows: int, *, ledger: MemoryLedger | None = None
) -> int:
    """Return the bytes of reading ``n_rows`` stored rows as Python strings.

    Variable-width text counts the rows, three times as
    :func:`array_row_selection_parts` counts each row, and their strings, at
    no more than every value of the array holds, however long its longest
    value.
    """
    n_rows = int(n_rows)
    if np.dtype(array.dtype).kind != "T":
        fixed, per_row = text_row_selection_parts(array)
        return fixed + n_rows * per_row
    geometry = array_geometry(array, measure_text=True, ledger=ledger)
    fixed, _ = _row_selection_parts(array, geometry, ledger)
    extent = text_extent(array, geometry, ledger)
    characters = min(n_rows * extent.characters, extent.characterTotal)
    strings = object_array_bytes(n_rows) + python_text_bytes(
        n_rows, characters, widest=extent.widest
    )
    index_bytes = _SELECTION_INDEX_ARRAYS * np.dtype(np.int64).itemsize
    return (
        fixed
        + 3 * _rows_held(array, geometry, extent, n_rows)
        + n_rows * index_bytes
        + strings
    )


def _rows_held(
    array: Any, geometry: ArrayGeometry | None, extent: TextExtent, n_rows: int
) -> int:
    """The bytes of ``n_rows`` rows of variable-width text ``array`` as NumPy keeps them: no more than its longest value for each value, nor than every value."""
    if geometry is not None:
        return geometry.rowsBytes(n_rows)
    row_values = int(np.prod(tuple(array.shape)[1:], dtype=np.int64))
    return min(n_rows * row_values * extent.valueBytes(), extent.keptBytes())


def fingerprint_stored_rows(array: Any, rows: np.ndarray) -> str:
    """Fingerprint a stored text column at ascending ``rows``, one stored chunk at a time.

    The fingerprint is :func:`~cytearc.storage.artifacts.fingerprint_strings`
    of the values at those rows read whole, but the check holds the rows of
    one chunk at a time; values of variable width are read twice, once to
    measure the widest.
    """
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Metadata row indices must be one-dimensional")
    if indices.size > 1 and not bool(np.all(indices[1:] > indices[:-1])):
        raise ValueError("Rows to fingerprint must be distinct and ascending")
    band = row_band(
        array_geometry(array), unit="chunk", fallback=max(1, int(array.shape[0]))
    )
    # Where the rows of each stored chunk end among the rows, in order.
    ends = np.searchsorted(indices, np.arange(band, int(array.shape[0]), band))

    def blocks() -> Iterator[np.ndarray]:
        start = 0
        for stop in [*ends.tolist(), int(indices.size)]:
            if stop > start:
                yield _read_array_rows(array, indices[start:stop])
            start = stop

    return fingerprint_text_blocks(int(indices.size), array.dtype, blocks)


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
        # A band's rows go before the next band is read.
        del values
    return output


def operation_concurrency() -> int:
    """The chunks that Zarr decodes at once for the operation under way."""
    return max(1, int(zarr.config.get("async.concurrency")))


def read_whole(
    ledger: MemoryLedger,
    group: zarr.Group,
    name: str,
    *,
    item: str,
    codes: bool = False,
) -> np.ndarray:
    """Read the whole array ``name`` of ``group`` within the ledger's room, and keep it charged as ``item``.

    An unsharded array is read in one call, as :func:`~cytearc.storage.types.read_array`
    reads it, at the highest Zarr concurrency up to the operation's whose
    decoded chunks fit the room, and at one chunk at a time when only that
    fits. A sharded array is read a band of inner chunks at a time, so that
    no fetch takes more than one band of a shard. The ledger counts what the
    read holds while it runs, as ``ArrayGeometry.wholeReadBytes`` states, and
    keeps the result charged. A categorical column of a metadata table is
    read as its labels, as :func:`read_rows` decodes them, unless ``codes``
    asks for its stored codes.

    Raises:
        MemoryBudgetError: If even a read of one chunk at a time would pass
            the budget.
        TypeError: If ``name`` is a group, not an array.
    """
    array = as_zarr_array(group[name], name=name)
    categorical = None if codes else CategoricalColumn.open(group, name, values=array)
    if categorical is not None:
        # A categorical column reads its codes whole and decodes them.
        stored = read_whole(ledger, group, name, item=item, codes=True)
        return _decoded(ledger, categorical, stored, item)
    values, _ = _read_whole_array(ledger, group, name, array, item=item)
    return values


def _read_whole_array(
    ledger: MemoryLedger, group: zarr.Group, name: str, array: zarr.Array, *, item: str
) -> tuple[np.ndarray, ArrayGeometry | None]:
    """Read stored array ``name`` of ``group``, ``array``, whole as :func:`read_whole` does, and return the geometry that its charges took."""
    geometry = array_geometry(array, measure_text=True, ledger=ledger)
    if geometry is None:
        # An array without chunks, such as a scalar, is read as it is.
        values = read_array(group, name) if array.ndim else np.asarray(array[()])
        ledger.hold(item, _kept_bytes(values, None, ledger))
        return values, None
    concurrency = _fitting_concurrency(ledger, geometry.wholeReadBytes)
    ledger.peak(item, geometry.wholeReadBytes(concurrency))
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
    ledger.hold(item, _kept_bytes(values, geometry, ledger))
    return values, geometry


def _kept_bytes(
    values: np.ndarray, geometry: ArrayGeometry | None, ledger: MemoryLedger
) -> int:
    """The bytes that ``values``, read from an array of ``geometry``, keep.

    NumPy counts a value of variable-width text at its 16-byte element; the
    geometry counts its characters too, as ``ArrayGeometry.rowsBytes`` counts
    rows, beside what the read leaves for Python's next collection, which the
    strings that such a read makes, charged by their characters, leave no
    room for; text read from an array without chunks is measured itself.
    """
    if values.dtype.kind != "T":
        return int(values.nbytes)
    if geometry is not None:
        rows = values.shape[0] if values.ndim else 1
        return geometry.rowsBytes(rows) + geometry.leftoverBytes(rows)
    return text_extent(values, None, ledger).keptBytes()


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


def row_read_bytes(geometry: ArrayGeometry | None, n_rows: int) -> int:
    """What reading ``n_rows`` distinct rows of an array of ``geometry`` a band of chunks at a time holds, with the rows.

    Ordering the rows by band holds their indices as
    ``storage.geometry.ROW_INDEX_BYTES`` counts them. Reading the bands then
    holds each row in the result and the indices it keeps, one band's
    selected rows as Zarr returns them and what reading the band holds, as
    ``ArrayGeometry.bandReadBytes`` counts. An unsharded array's read also
    copies the rows that a decoded chunk holds before it places them, up to
    a chunk's rows; a sharded array's read places them through its
    shard-level copy, which ``bandReadBytes`` counts. Variable-width text is
    read a chunk's span of the rows at a time, which Zarr returns up to a
    chunk's rows, before the rows are taken from it, and one row alone.
    Without a geometry, as
    for an array in memory, a row's bytes are not known, so only the rows'
    indices are counted; :func:`read_rows_bytes` adds the rows from the
    array.
    """
    n_rows = int(n_rows)
    ordering = n_rows * ROW_INDEX_BYTES
    if geometry is None:
        return ordering
    band_rows = min(n_rows, geometry.axisChunk(0))
    # Variable-width text is read a chunk's span of the rows at a time, which
    # Zarr returns up to a chunk of rows; one row is its own span.
    returned_rows = (
        geometry.axisChunk(0) if geometry.text is not None and n_rows > 1 else band_rows
    )
    reading = (
        geometry.rowsBytes(n_rows)
        + n_rows * ROW_KEPT_INDEX_BYTES
        + geometry.bandReadBytes(returned_rows, operation_concurrency())
        + (geometry.rowsBytes(band_rows) if geometry.shards is None else 0)
    )
    return max(ordering, reading)


def read_rows_bytes(
    array: Any, n_rows: int, *, ledger: MemoryLedger | None = None
) -> int:
    """What :func:`read_rows` holds to read ``n_rows`` distinct rows of ``array``, with the rows.

    A stored array holds what :func:`row_read_bytes` states, and variable-width
    text no less than what measuring it holds, as
    :func:`~cytearc.storage.geometry.variable_text_extent` measures it once in
    an operation, the measure charged to ``ledger``. An array without chunks,
    such as a NumPy array, holds the rows it returns beside their indices. A
    categorical column holds the read of its codes, then its
    codes beside the categories, the lookup that decoding builds and the
    labels returned.
    """
    n_rows = int(n_rows)
    if not isinstance(array, CategoricalColumn):
        geometry = array_geometry(array, measure_text=True, ledger=ledger)
        if geometry is not None:
            reading = row_read_bytes(geometry, n_rows)
            # A read of variable-width text first measures it, unless its
            # operation has.
            if geometry.text is not None:
                return max(reading, geometry.text.measureBytes)
            return reading
        if np.dtype(array.dtype).kind == "T":
            extent = text_extent(array, None, ledger)
            rows = _rows_held(array, None, extent, n_rows)
            return row_read_bytes(None, n_rows) + max(rows, extent.measureBytes)
        row_values = int(np.prod(array.shape[1:], dtype=np.int64))
        row_bytes = row_values * int(np.dtype(array.dtype).itemsize)
        return row_read_bytes(None, n_rows) + n_rows * row_bytes
    codes = row_read_bytes(array_geometry(array.codes), n_rows)
    decoding = (
        n_rows * int(np.dtype(array.codes.dtype).itemsize)
        + _categories_bytes(array)
        + n_rows * int(array.dtype.itemsize)
    )
    return max(codes, decoding)


def read_rows(
    ledger: MemoryLedger, array: Any, rows: np.ndarray, *, item: str
) -> np.ndarray:
    """Read distinct ``rows`` of a stored array a band of chunks at a time, keeping them charged as ``item``.

    The rows are read as :func:`read_array_rows_chunkwise` reads them, in the
    order given, on the ledger's codec threads. The ledger counts what the
    read holds, as :func:`row_read_bytes` states, and keeps the rows charged.
    A categorical column reads the codes of the rows that way, then decodes
    them into its categories' dtype beside the categories, so the read holds
    codes and the labels returned rather than a stored label per row.

    Raises:
        MemoryBudgetError: If the read would pass the budget.
    """
    if isinstance(array, CategoricalColumn):
        codes = read_rows(ledger, array.codes, rows, item=item)
        return _decoded(ledger, array, codes, item)
    values, _ = _read_stored_rows(ledger, array, rows, item=item)
    return values


def _read_stored_rows(
    ledger: MemoryLedger, array: Any, rows: np.ndarray, *, item: str
) -> tuple[np.ndarray, ArrayGeometry | None]:
    """Read ``rows`` of an array that is not categorical as :func:`read_rows` does, and return the geometry that its charges took."""
    indices = np.asarray(rows)
    geometry = array_geometry(array, measure_text=True, ledger=ledger)
    ledger.peak(item, row_read_bytes(geometry, int(indices.size)))
    with zarr_decoding_threads(ledger.codec_threads):
        values = read_array_rows_chunkwise(array, indices)
    ledger.hold(item, _kept_bytes(values, geometry, ledger))
    return values, geometry


def read_text_rows(
    ledger: MemoryLedger, array: Any, rows: np.ndarray, *, item: str
) -> np.ndarray:
    """Read distinct ``rows`` of a stored text array as Python strings, keeping them charged as ``item``.

    The rows are read at their stored width through :func:`read_rows`, then
    measured: their characters and their widest code point give what their
    strings hold, as :func:`~cytearc.utils.memory_budget.python_text_bytes`
    counts, which is charged before they are made. The stored rows are then
    let go, and the strings stay charged.

    A categorical column of text categories reads the codes of the rows
    through :func:`read_rows` and returns its categories at them as one
    Python string per category, which the rows share: the read holds the
    codes, the categories' strings and a reference per row, not a string per
    row.

    Text of variable width, which another tool can write with Zarr's
    ``dtype=str``, is read as :func:`read_rows` reads it, and its strings
    are charged at the bytes per character of its widest character, as
    :func:`~cytearc.storage.geometry.variable_text_extent` measures it.

    Returns:
        An object array of one string for each row.

    Raises:
        MemoryBudgetError: If the read or the strings would pass the budget.
        TypeError: If the array does not store text.
    """
    if np.dtype(array.dtype).kind not in "UT":
        raise TypeError(f"{item} must be stored as text, not as {array.dtype}")
    if isinstance(array, CategoricalColumn):
        codes = read_rows(ledger, array.codes, rows, item=item)
        return _decoded_strings(ledger, array, codes, item)
    held = ledger.held_bytes
    stored, geometry = _read_stored_rows(ledger, array, rows, item=item)
    stored_bytes = ledger.held_bytes - held
    values = _strings_of(ledger, stored, item, widest=_widest(stored, geometry, ledger))
    del stored
    ledger.release(stored_bytes)
    return values


def read_whole_text(
    ledger: MemoryLedger, group: zarr.Group, name: str, *, item: str
) -> np.ndarray:
    """Read the whole text array ``name`` of ``group`` as Python strings, keeping them charged as ``item``.

    The array is read at its stored width through :func:`read_whole`, then
    its strings are charged and made as :func:`read_text_rows` makes them,
    and the stored values are let go.

    Returns:
        An object array of one string for each value.

    Raises:
        MemoryBudgetError: If the read or the strings would pass the budget.
        TypeError: If the array does not store text.
    """
    array = as_zarr_array(group[name], name=name)
    categorical = CategoricalColumn.open(group, name, values=array)
    labels = array if categorical is None else categorical
    if np.dtype(labels.dtype).kind not in "UT":
        raise TypeError(f"{item} must be stored as text, not as {labels.dtype}")
    if categorical is not None:
        codes = read_whole(ledger, group, name, item=item, codes=True)
        return _decoded_strings(ledger, categorical, codes, item)
    held = ledger.held_bytes
    stored, geometry = _read_whole_array(ledger, group, name, array, item=item)
    stored_bytes = ledger.held_bytes - held
    values = _strings_of(ledger, stored, item, widest=_widest(stored, geometry, ledger))
    del stored
    ledger.release(stored_bytes)
    return values


def _widest(
    stored: np.ndarray, geometry: ArrayGeometry | None, ledger: MemoryLedger
) -> int:
    """The widest code point of text that a read of an array of ``geometry`` returned, 0 for none.

    Variable-width text takes the widest of its array, which the read's
    charge measured, or of itself, read from an array without chunks.
    """
    if stored.size == 0:
        return 0
    if stored.dtype.kind == "T":
        return text_extent(stored, geometry, ledger).widest
    return int(stored.view(np.uint32).max())


def _strings_of(
    ledger: MemoryLedger, stored: np.ndarray, item: str, *, widest: int
) -> np.ndarray:
    """``stored``, text that the ledger holds, as Python strings charged beside it.

    ``widest`` bounds the code points of its characters. The caller lets go
    of the stored text, and releases its charge, once it drops its own
    reference to it.
    """
    # Each value's length, while the characters are counted.
    ledger.peak(item, 8 * len(stored))
    characters = int(np.strings.str_len(stored).sum()) if len(stored) else 0
    ledger.hold(
        item,
        object_array_bytes(len(stored))
        + python_text_bytes(len(stored), characters, widest=widest),
    )
    return stored.astype(object)


def _categories_bytes(column: CategoricalColumn) -> int:
    """The bytes of a categorical column's categories as read, beside the lookup that decoding builds from them."""
    return 2 * (column.n_categories + 1) * int(column.dtype.itemsize)


def _decoded(
    ledger: MemoryLedger, column: CategoricalColumn, codes: np.ndarray, item: str
) -> np.ndarray:
    """The labels of ``codes``, which the ledger holds, charged in their place.

    The categories and the lookup are held while the labels are made; the
    codes are let go once they are.
    """
    ledger.peak(item, _categories_bytes(column) + len(codes) * column.dtype.itemsize)
    values = column.decode(codes)
    ledger.hold(item, values.nbytes)
    codes_bytes = codes.nbytes
    del codes
    ledger.release(codes_bytes)
    return values


def _decoded_strings(
    ledger: MemoryLedger, column: CategoricalColumn, codes: np.ndarray, item: str
) -> np.ndarray:
    """The labels of ``codes``, which the ledger holds, as references to one string per category, charged in their place.

    Each category becomes one Python string, measured by its characters and
    its widest character, which the rows share; the codes are let go once
    the references are made.
    """
    categories = column.categories
    characters = int(np.char.str_len(categories).sum()) if categories.size else 0
    widest = int(categories.view(np.uint32).max()) if categories.size else 0
    ledger.peak(item, _categories_bytes(column))
    ledger.hold(
        item,
        object_array_bytes(len(codes) + column.n_categories + 1)
        + python_text_bytes(column.n_categories + 1, characters, widest=widest),
    )
    values = column.decode_objects(codes)
    codes_bytes = codes.nbytes
    del codes
    ledger.release(codes_bytes)
    return values


def read_metadata_rows(
    metadata: _RowReadableMetaData,
    column: str,
    rows: np.ndarray,
) -> np.ndarray:
    """Read selected metadata rows without expanding scattered ranges."""
    return _read_array_rows(metadata._get_array(column), rows)


def categorical_rows(column: CategoricalColumn, codes: np.ndarray) -> pd.Categorical:
    """Return the codes of some rows of a categorical column as a pandas Categorical over its categories.

    The codes are checked as decoding checks them and kept as they are, so
    no row becomes its label; pandas widens them only where its own code
    type for that many categories is wider, for 127 and 128 categories.
    """
    return pd.Categorical.from_codes(
        column.check_codes(codes),
        categories=pd.Index(column.categories),
        ordered=column.ordered,
        validate=False,
    )


def read_metadata_label_rows(
    metadata: _RowReadableMetaData,
    column: str,
    rows: np.ndarray,
) -> np.ndarray | pd.Categorical:
    """Read selected rows of a label column: a categorical column as a Categorical of its codes, another as its values.

    A categorical column's codes are read as :func:`read_metadata_rows`
    reads values, and returned by :func:`categorical_rows`.
    """
    array = metadata._get_array(column)
    if isinstance(array, CategoricalColumn):
        return categorical_rows(array, _read_array_rows(array.codes, rows))
    return _read_array_rows(array, rows)


def category_values_bytes(column: CategoricalColumn) -> int:
    """The bytes of a categorical column's categories as stored and as the Python values of a pandas Categorical.

    Text categories are measured by their characters and their widest
    character, as :func:`~cytearc.utils.memory_budget.python_text_bytes`
    counts them, and other categories as numbers.
    """
    categories = column.categories
    n_categories = len(categories)
    if categories.dtype.kind != "U":
        return categories.nbytes + object_array_bytes(n_categories) + 32 * n_categories
    characters = int(np.char.str_len(categories).sum()) if n_categories else 0
    widest = int(categories.view(np.uint32).max()) if categories.size else 0
    return (
        categories.nbytes
        + object_array_bytes(n_categories)
        + python_text_bytes(n_categories, characters, widest=widest)
    )


def read_categorical_rows(
    ledger: MemoryLedger, column: CategoricalColumn, rows: np.ndarray, *, item: str
) -> pd.Categorical:
    """Read distinct ``rows`` of a categorical column as a Categorical of their codes, keeping it charged as ``item``.

    The codes are read as :func:`read_rows` reads them and stay charged,
    beside the categories, as :func:`category_values_bytes` counts them, so
    the read holds a code per row and no label.

    Raises:
        MemoryBudgetError: If the read would pass the budget.
    """
    codes = read_rows(ledger, column.codes, rows, item=item)
    ledger.hold(item, category_values_bytes(column))
    categorical = categorical_rows(column, codes)
    if categorical.codes.dtype != codes.dtype:
        # pandas widened the codes into a copy, which replaces them.
        ledger.hold(item, categorical.codes.nbytes)
        codes_bytes = codes.nbytes
        del codes
        ledger.release(codes_bytes)
    return categorical


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
    changes it. A categorical column is hashed as its labels would be stored
    in a column of their own: at the dtype of
    :meth:`~cytearc.storage.categorical.CategoricalColumn.stored_labels_dtype`,
    in blocks of the chunk rows that a cell or feature table gives such a
    column, so the same labels give the same digest in either layout.
    """
    digest = hashlib.sha256()
    for block in _fingerprint_blocks(metadata, column):
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


def _fingerprint_blocks(
    metadata: _RowReadableMetaData, column: str
) -> Iterator[np.ndarray]:
    """Yield a column's values in the blocks that :func:`metadata_column_fingerprint` hashes."""
    array = metadata._get_array(column)
    if not isinstance(array, CategoricalColumn):
        yield from iter_metadata_column_blocks(metadata, column)
        return
    dtype = array.stored_labels_dtype()
    rows = column_chunk_rows(metadata.N, max(1, dtype.itemsize))
    for start in range(0, metadata.N, rows):
        yield array[start : start + rows].astype(dtype)


def metadata_column_stamp(metadata: _RowReadableMetaData, column: str) -> str:
    """Stamp one metadata column by its stored chunks and those of its missing mask.

    The digest covers the column's stored objects, as
    :func:`~cytearc.storage.content_stamps.stored_content_stamp` reads them
    without decoding them, and whether it has a missing mask and that mask's
    objects, so an edit to the values, their dtype or which are missing
    changes it. A categorical column is stamped by its codes and its
    categories. Values written again with other chunks, codecs or layout
    change it too.
    """
    array = metadata._get_array(column)
    if isinstance(array, CategoricalColumn):
        digest = hashlib.sha256(stored_content_stamp(array.codes))
        digest.update(b"categories:")
        digest.update(stored_content_stamp(array.categories_array))
    else:
        digest = hashlib.sha256(stored_content_stamp(array))
    missing = metadata_missing_mask(metadata, column)
    digest.update(b"missing:none" if missing is None else b"missing:present")
    if missing is not None:
        digest.update(stored_content_stamp(missing))
    return digest.hexdigest()


def metadata_column_stamp_bytes(metadata: _RowReadableMetaData, column: str) -> int:
    """The most bytes that :func:`metadata_column_stamp` holds: the largest of its stamps, made in turn."""
    missing = metadata_missing_mask(metadata, column)
    array = metadata._get_array(column)
    stamped = (
        [array.codes, array.categories_array]
        if isinstance(array, CategoricalColumn)
        else [array]
    )
    return max(
        *(stored_content_stamp_bytes(each) for each in stamped),
        0 if missing is None else stored_content_stamp_bytes(missing),
    )


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
