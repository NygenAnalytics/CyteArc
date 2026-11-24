"""Row bands and chunk-aligned index blocks for streamed arrays."""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..utils.arrays import has_duplicates
from .geometry import _BAND_LEFTOVER_LIMIT_BYTES, ArrayGeometry, array_geometry

type Fits = Callable[[int], bool]

__all__ = [
    "IndexBlock",
    "affordable_width",
    "band_shards",
    "checked_indices",
    "coalesced_gap_bytes",
    "contiguous_ranges",
    "index_band_bytes",
    "indexed_read_parts",
    "is_contiguous",
    "partition_indices",
    "read_variable_text_rows",
    "row_band",
    "row_read_parts",
    "run_shards",
    "scan_band",
    "widest_chunk_span",
]

# One multi-chunk read per band: Zarr fetches a band's chunks concurrently.
SCAN_BAND_BYTES = 64 * 1024**2
#: A read in bands orders the indices of the axis it slices by chunk, and
#: keeps their order, the ordered indices and their chunks while it reads,
#: each band's indices within the band, and where each band starts and
#: stops among them, 16 bytes a band, which a band of one index holds for
#: that index: 48 bytes an index of int64 in all. Its stable sort and masks
#: held less beside them, 26 bytes an index traced.
_BAND_INDEX_BYTES = 48
#: Passes over a selection's indices take at most this many at a time, or,
#: where they take whole bands, one band that holds more, so a pass holds 9
#: to 11 bytes for each index of a piece, 590 to 720 KB for 65,536 of them,
#: not for each index of the selection.
_INDEX_PIECE = 65_536


@dataclass(frozen=True, slots=True)
class IndexBlock:
    """Selected indices that share physical chunks, and where they belong."""

    indices: np.ndarray
    destinations: np.ndarray
    bins: tuple[int, ...]


def row_band(
    geometry: ArrayGeometry | None,
    *,
    unit: Literal["shard", "chunk"] = "shard",
    fallback: int,
) -> int:
    """Return rows per block from stored geometry, or ``fallback`` without it.

    ``unit`` selects the stored extent to follow: ``shard`` for the object a
    write lands in, ``chunk`` for the unit a read decodes.
    """
    if geometry is None:
        return max(1, int(fallback))
    extent = geometry.axisShard(0) if unit == "shard" else geometry.axisChunk(0)
    return max(1, extent)


def scan_band(geometry: ArrayGeometry | None, *, fallback: int) -> int:
    """Return rows per read for a whole-array scan: whole chunks up to 64 MiB.

    Reading one chunk per request leaves object stores idle for a round trip
    per chunk; a band of chunks costs one round trip.
    """
    if geometry is None:
        return max(1, int(fallback))
    chunks = max(1, SCAN_BAND_BYTES // max(1, geometry.nominalChunkBytes()))
    return geometry.axisChunk(0) * chunks


def contiguous_ranges(nRows: int, band: int) -> list[tuple[int, int]]:
    """Split a row axis into consecutive half-open ranges of ``band`` rows."""
    rows = max(0, int(nRows))
    step = max(1, int(band))
    return [(start, min(start + step, rows)) for start in range(0, rows, step)]


def is_contiguous(indices: np.ndarray) -> bool:
    """Return whether indices form a strictly increasing consecutive run."""
    if indices.size == 0:
        return True
    if indices[0] < 0:
        return False
    if indices.size == 1:
        return True
    # A run of n consecutive indices spans n - 1, which most selections of
    # cells do not, so they are told apart without a pass over them, which
    # would hold 9 bytes an index.
    if int(indices[-1]) - int(indices[0]) != indices.size - 1:
        return False
    # The steps of a run are taken a piece at a time, and in int64: in a
    # narrow integer type a descending step, such as 255 to 0 in uint8, would
    # wrap around to a step of one.
    values = np.asarray(indices)
    for start in range(0, values.size - 1, _INDEX_PIECE):
        piece = values[start : start + _INDEX_PIECE + 1].astype(np.int64, copy=False)
        if not np.all(np.diff(piece) == 1):
            return False
    return True


def widest_chunk_span(indices: np.ndarray, chunk: int) -> int:
    """Return how many positions the widest chunk's selected indices span.

    The indices that one stored chunk of extent ``chunk`` holds span from the
    smallest to the largest of them. ``indices`` are non-negative.
    """
    if indices.size == 0:
        return 0
    values = np.sort(np.asarray(indices, dtype=np.int64))
    bins = values // max(1, int(chunk))
    boundary = bins[1:] != bins[:-1]
    first = values[np.r_[True, boundary]]
    last = values[np.r_[boundary, True]]
    return int((last - first).max()) + 1


def band_shards(
    geometry: ArrayGeometry | None, rows: np.ndarray | range, band_rows: int
) -> int:
    """Return the most rows of shards that a band of ``band_rows`` consecutive entries of ``rows`` touches.

    ``rows`` holds rows of the array, or is a ``range`` of them, as a copy
    reads a run of rows a band at a time; an unsharded array counts its rows
    of chunks. A band counts each row of shards it touches once, in whatever
    order its rows come, as Zarr reads each shard of a selection once. The
    entries are counted a piece of whole bands at a time, at most 65,536
    entries or one band, so the count holds about 11 bytes for each entry of
    a piece, not of the selection, and sorts a piece only if its entries are
    not in order already.
    """
    if geometry is None or len(rows) == 0:
        return 1
    extent = geometry.axisShard(0)
    band = max(1, int(band_rows))
    if isinstance(rows, range) and rows.step == 1:
        starts = np.arange(rows.start, rows.stop, band, dtype=np.int64)
        ends = np.minimum(starts + band, rows.stop) - 1
        return int((ends // extent - starts // extent).max()) + 1
    piece = max(band, _INDEX_PIECE // band * band)
    # In int64, since a narrow integer type may not hold the extent.
    return max(
        _most_in_bands(
            np.asarray(rows[start : start + piece], dtype=np.int64) // extent, band
        )
        for start in range(0, len(rows), piece)
    )


def _most_in_bands(values: np.ndarray, band: int) -> int:
    """Return the most distinct values that a band of ``band`` consecutive entries of ``values`` holds.

    ``values`` holds at least one entry, and is sorted band by band in place
    unless its entries are in order already.
    """
    whole = values.size - values.size % band
    groups = (values[:whole].reshape(-1, band), values[whole:].reshape(1, -1))
    if bool(np.any(values[1:] < values[:-1])):
        for group in groups:
            group.sort(axis=1)
    # Sorted, each band's values change once for each one after the first.
    return max(
        int((1 + np.count_nonzero(group[:, 1:] != group[:, :-1], axis=1)).max())
        for group in groups
        if group.size
    )


def run_shards(geometry: ArrayGeometry | None, rows: int) -> int:
    """Return the most rows of shards that a run of ``rows`` consecutive rows touches, wherever it starts.

    An unsharded array counts its rows of chunks, as :func:`band_shards` does.
    """
    if geometry is None or rows <= 1:
        return 1
    return -(-(int(rows) - 1) // geometry.axisShard(0)) + 1


def _column_shards(geometry: ArrayGeometry, columns: np.ndarray) -> int:
    """Return the column shards that ``columns`` touch."""
    chunks_per_shard = -(-geometry.axisShard(1) // geometry.axisChunk(1))
    return int(np.unique(geometry.binOf(1, columns) // chunks_per_shard).size)


def _shard_column_chunks(geometry: ArrayGeometry, columns: np.ndarray) -> int:
    """Return the most inner column chunks that ``columns`` touch in one shard."""
    column_chunks = np.unique(geometry.binOf(1, columns))
    chunks_per_shard = -(-geometry.axisShard(1) // geometry.axisChunk(1))
    _, counts = np.unique(column_chunks // chunks_per_shard, return_counts=True)
    return int(counts.max())


def coalesced_gap_bytes(
    geometry: ArrayGeometry, columns: np.ndarray, *, chunks: int
) -> int:
    """Return the most stored bytes that a read of ``columns`` fetches in one shard beside the ``chunks`` chunks it is charged.

    Zarr fetches the chunks that a read takes from part of a shard as byte
    ranges, and merges ranges up to ``coalesceGapBytes`` apart into one fetch,
    which holds the stored bytes between them too: no more than that many
    bytes for each gap, nor than the chunks in it at the bound of their
    encoded size. A shard of one row of chunks stores them in column order,
    so its gaps lie between the selected column chunks. A shard of several
    rows of chunks may store them in any order, Zarr's default being Morton
    order, so each of the ``chunks - 1`` spaces between the chunks that a
    read takes may hold a gap, up to every chunk of the shard it does not
    take. A read that takes every chunk of a shard fetches the shard whole.
    """
    gap = max(0, int(geometry.coalesceGapBytes))
    if gap == 0 or geometry.shards is None or columns.size == 0:
        return 0
    stored = geometry.storedChunkBytes() - geometry.nominalChunkBytes()
    per_shard = -(-geometry.axisShard(1) // geometry.axisChunk(1))
    in_array = -(-geometry.shape[1] // geometry.axisChunk(1))
    shard_rows = -(-geometry.axisShard(0) // geometry.axisChunk(0))
    column_chunks = np.unique(geometry.binOf(1, columns))
    shard_of = column_chunks // per_shard
    most = 0
    for shard in np.unique(shard_of).tolist():
        if shard_rows == 1:
            between = np.diff(column_chunks[shard_of == shard]) - 1
            fetched = int(np.minimum(gap, between[between > 0] * stored).sum())
        else:
            present = shard_rows * min(per_shard, in_array - shard * per_shard)
            taken = min(max(1, int(chunks)), present)
            fetched = min((taken - 1) * gap, (present - taken) * stored)
        most = max(most, fetched)
    return most


def index_band_bytes(
    geometry: ArrayGeometry,
    columns: np.ndarray,
    *,
    rows: int,
    row_span: int,
    concurrency: int = 1,
    shards: int = 1,
) -> tuple[int, int]:
    """Return what a read of ``rows`` rows by ``columns``, both index arrays, holds read in bands of columns, and in bands of rows.

    ``read_selection`` reads a sharded array's index arrays on both axes one
    stored chunk of one axis at a time, as a band that slices that axis over
    the indices the chunk holds, and holds the widest band beside the result
    while Zarr reads it. A band of columns has ``widest_chunk_span(columns)``
    values per row; a band of rows holds every selected column over the
    ``row_span`` rows that the selected rows of one row chunk span. Each band
    read takes the indices of the axis it does not slice by an index array,
    so it also holds Zarr's indexers of them and its copy of what it takes
    from each chunk it decodes: each row at a column band's columns, or a
    row band's rows at the most selected columns of one chunk, beside the
    reader's own ordering of the indices of the axis it slices. Either read
    also holds the stored bytes between the chunks it takes, as
    :func:`coalesced_gap_bytes` counts them. A band of columns lies in one
    column shard, so its read touches the ``shards`` rows of shards that the
    rows touch, as :func:`band_shards` counts them; a band of rows lies in
    one row of shards, so its read touches the column shards of
    ``columns``. At Zarr's async ``concurrency``, a band read decodes up to
    that many chunks of a shard at once and reads up to that many of the
    shards it touches at once. The reads of the earlier bands leave their
    objects while the last one runs, as
    :meth:`~cytearc.storage.geometry.ArrayGeometry.chunksLeftoverBytes`
    counts them for the chunks those reads took, up to 640 KiB: a chunk in
    each row chunk that the rows lie in for each earlier band of columns, and
    each selected column chunk for each earlier band of rows, one band to a
    row chunk that the rows lie in. A read of no columns reads nothing.
    """
    if columns.size == 0:
        return 0, 0
    rows = max(0, int(rows))
    (column_fixed, column_per_row), (row_fixed, row_per_row) = _index_band_parts(
        geometry, columns, row_span=row_span, concurrency=concurrency, shards=shards
    )
    column_left, row_left = _band_leftovers(geometry, columns, rows=rows, shards=shards)
    return (
        column_fixed + rows * column_per_row + column_left,
        row_fixed + rows * row_per_row + row_left,
    )


def _index_band_parts(
    geometry: ArrayGeometry,
    columns: np.ndarray,
    *,
    row_span: int,
    concurrency: int,
    shards: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the bytes fixed and the bytes a row that a read in bands of columns, and in bands of rows, holds in its last band, beside what the earlier bands leave.

    They are the band's Zarr read and the reader's ordering of the indices
    of the axis that the bands slice, as :func:`index_band_bytes` states.
    """
    row_chunks = -(-geometry.axisShard(0) // geometry.axisChunk(0))
    # A column band touches a column chunk of each row chunk in a shard; a
    # row band touches a row chunk of each selected column chunk.
    touched = max(row_chunks, _shard_column_chunks(geometry, columns))
    gap = coalesced_gap_bytes(geometry, columns, chunks=touched)

    def read(resultBytes: int, indexes: int, takenBytes: int, read_shards: int) -> int:
        return geometry.readBytes(
            resultBytes,
            touched,
            decodes=max(1, int(concurrency)),
            shards=read_shards,
            gapBytes=gap,
            indexes=indexes,
            takenBytes=takenBytes,
        )

    span = widest_chunk_span(columns, geometry.axisChunk(1))

    def column_band(rows: int) -> int:
        return read(
            rows * span * geometry.itemsize,
            rows,
            geometry.takenChunkBytes(rows, span, indexedAxes=1),
            max(1, int(shards)),
        )

    column_fixed = column_band(0) + _BAND_INDEX_BYTES * columns.size
    _, in_chunk = np.unique(geometry.binOf(1, columns), return_counts=True)
    row_band = read(
        columns.size * row_span * geometry.itemsize,
        columns.size,
        geometry.takenChunkBytes(row_span, int(in_chunk.max()), indexedAxes=1),
        _column_shards(geometry, columns),
    )
    return (
        (column_fixed, column_band(1) - column_band(0)),
        (row_band, _BAND_INDEX_BYTES),
    )


def _band_leftovers(
    geometry: ArrayGeometry, columns: np.ndarray, *, rows: int, shards: int
) -> tuple[int, int]:
    """Return what the reads of the earlier bands leave of their objects while the last one runs, read in bands of columns, and in bands of rows.

    ``rows`` rows in ``shards`` rows of shards lie in no more row chunks
    than they have rows, nor than those rows of shards and the array hold. A
    band of columns takes a chunk in each of those row chunks, and a band of
    rows each selected column chunk of one of them, as
    :meth:`~cytearc.storage.geometry.ArrayGeometry.chunksLeftoverBytes`
    counts what they leave, up to the 640 KiB at which what two-dimensional
    reads leave levels off.
    """
    row_chunks = min(
        max(0, int(rows)),
        max(1, int(shards)) * -(-geometry.axisShard(0) // geometry.axisChunk(0)),
        geometry.bandCount(),
    )
    column_chunks = int(np.unique(geometry.binOf(1, columns)).size)
    return (
        geometry.chunksLeftoverBytes(
            (column_chunks - 1) * row_chunks, limit=_BAND_LEFTOVER_LIMIT_BYTES
        ),
        geometry.chunksLeftoverBytes(
            (row_chunks - 1) * column_chunks, limit=_BAND_LEFTOVER_LIMIT_BYTES
        ),
    )


def indexed_read_parts(
    geometry: ArrayGeometry,
    columns: np.ndarray,
    *,
    block_rows: int,
    concurrency: int = 1,
    shards: int = 1,
) -> tuple[int, int]:
    """Return per-decode and per-row bytes beside a read that indexes both axes.

    The reader reads such a block in the bands that hold less, as
    :func:`index_band_bytes` counts them for its rows, and the rows of a row
    chunk span no more than the chunk. So the bands of either axis, at a
    chunk's rows, bound the read of a block of up to ``block_rows`` rows,
    beside what the earlier bands of such a block leave, which a block of
    fewer rows leaves no more of, and the bound that is smaller for
    ``block_rows`` rows is charged. The selected rows are not scanned, so a
    plan holds no copy of them.
    """
    if columns.size == 0:
        return 0, 0
    rows = max(1, int(block_rows))
    (column_fixed, column_per_row), (row_fixed, row_per_row) = _index_band_parts(
        geometry,
        columns,
        row_span=min(geometry.axisChunk(0), geometry.shape[0]),
        concurrency=concurrency,
        shards=shards,
    )
    column_left, row_left = _band_leftovers(geometry, columns, rows=rows, shards=shards)
    column_fixed += column_left
    row_fixed += row_left
    if column_fixed + rows * column_per_row <= row_fixed + rows * row_per_row:
        return int(column_fixed), int(column_per_row)
    return int(row_fixed), int(row_per_row)


def row_read_parts(
    geometry: ArrayGeometry | None,
    columns: np.ndarray,
    *,
    rows_contiguous: bool,
    concurrency: int = 1,
    shards: int = 1,
    band_rows: int | None = None,
) -> tuple[int, int]:
    """Return per-decode and per-row bytes beside a selected row block's result.

    A read of part of a shard also holds the stored bytes between the chunks
    it takes, as :func:`coalesced_gap_bytes` counts them. Zarr reads a block
    at its async ``concurrency``: up to that many of the ``shards`` rows of
    shards that the block's rows touch, as :func:`band_shards` counts them,
    at once in each column shard, and up to that many chunks of each shard at
    once. An unsharded array's chunks are read and decoded each on its own,
    up to that many at once. A block whose rows are not one run, or whose
    columns are not, takes them by an index array, so its read also holds
    Zarr's indexers of them and its copy of what it takes from each chunk it
    decodes: each row of the block at the most selected columns of one chunk.
    A sharded block whose rows and columns are both index arrays is read in
    bands, as :func:`indexed_read_parts` charges, unless its rows form one
    run, which Zarr reads in one request over the rows of shards that a run
    of ``band_rows`` rows touches, as :func:`run_shards` counts them, and no
    more than ``shards``. Without ``band_rows``, the run is charged
    ``shards`` rows of shards and the bands are charged as a block of a row
    chunk's rows.
    """
    if geometry is None or columns.size == 0:
        return 0, 0
    reads = max(1, int(concurrency))
    row_chunks = -(-geometry.axisShard(0) // geometry.axisChunk(0))
    touched = row_chunks * _shard_column_chunks(geometry, columns)
    column_chunks, in_chunk = np.unique(geometry.binOf(1, columns), return_counts=True)
    column_gather = not is_contiguous(columns)
    # A sharded block whose rows and columns are both index arrays is read in
    # bands, below, which leaves the one Zarr request blocks whose rows form
    # one run.
    banded = not rows_contiguous and column_gather and geometry.shards is not None
    rows_indexed = not rows_contiguous and not banded
    request_shards = max(1, int(shards))
    if banded and band_rows is not None:
        request_shards = min(request_shards, run_shards(geometry, band_rows))

    def read(rows: int) -> int:
        """What a read of ``rows`` rows holds beside the rows it returns."""
        result = rows * columns.size * geometry.itemsize
        indexes = (rows if rows_indexed else 0) + (columns.size if column_gather else 0)
        taken = geometry.takenChunkBytes(
            rows,
            int(in_chunk.max()),
            indexedAxes=int(rows_indexed) + int(column_gather),
        )
        if geometry.shards is None:
            chunks = max(1, int(shards)) * int(column_chunks.size)
            held = geometry.readBytes(
                result, chunks, decodes=reads, indexes=indexes, takenBytes=taken
            )
        else:
            held = geometry.readBytes(
                result,
                touched,
                decodes=reads,
                shards=request_shards * _column_shards(geometry, columns),
                gapBytes=coalesced_gap_bytes(geometry, columns, chunks=touched),
                indexes=indexes,
                takenBytes=taken,
            )
        return held - result

    fixed = read(0)
    per_row = read(1) - fixed
    if banded:
        # A block whose rows form one run is read by Zarr in one request, and
        # any other in bands, so both are covered.
        band_fixed, band_per_row = indexed_read_parts(
            geometry,
            columns,
            block_rows=geometry.axisChunk(0) if band_rows is None else band_rows,
            concurrency=reads,
            shards=shards,
        )
        fixed = max(fixed, band_fixed)
        per_row = max(per_row, band_per_row)
    return int(fixed), int(per_row)


def affordable_width(fits: Fits, maxWidth: int) -> int:
    """Return the largest width up to ``maxWidth`` that ``fits`` accepts."""
    limit = max(0, int(maxWidth))
    if limit == 0:
        return 0
    if fits(limit):
        return limit
    low = 0
    high = limit - 1
    while low < high:
        candidate = (low + high + 1) // 2
        if fits(candidate):
            low = candidate
        else:
            high = candidate - 1
    return low


def checked_indices(
    values: Sequence[int] | np.ndarray,
    *,
    limit: int,
    name: str,
) -> np.ndarray:
    """Validate a one-dimensional selection of distinct in-range indices."""
    indexes = np.asarray(values)
    if indexes.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if indexes.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(indexes.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    indexes = indexes.astype(np.int64, copy=False)
    if np.any(indexes < 0) or np.any(indexes >= limit):
        raise IndexError(f"{name} contains an out-of-range index")
    if has_duplicates(indexes):
        raise ValueError(f"{name} cannot contain duplicate indexes")
    return indexes


def _bins(geometry: ArrayGeometry, axis: int, indices: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unique(geometry.binOf(axis, indices)))


def _bin_groups(
    geometry: ArrayGeometry,
    axis: int,
    indices: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Group index positions by chunk, ascending, keeping request order inside."""
    bins = geometry.binOf(axis, indices)
    order = np.argsort(bins, kind="stable")
    ordered = bins[order]
    edges = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            (np.flatnonzero(np.diff(ordered)) + 1).astype(np.int64),
            np.array([order.size], dtype=np.int64),
        )
    )
    return [
        (int(ordered[edges[position]]), order[edges[position] : edges[position + 1]])
        for position in range(edges.size - 1)
    ]


def read_variable_text_rows(array: Any, rows: np.ndarray) -> np.ndarray:
    """Read rows of an array of variable-width text, a stored chunk's span at a time.

    Zarr's selections of scattered rows index each decoded chunk with the
    rows followed by an Ellipsis, which NumPy releases before 2.3.2
    mishandle for variable-width text (numpy/numpy#29279): NumPy 2.1
    corrupts memory and crashes, and NumPy 2.2 to 2.3.1 return other
    strings. This reads, for each stored chunk, the slice that spans its
    selected rows, which decodes no more than Zarr's selection would, and
    takes the rows from it with ``np.take``, which reads the stored strings
    on every supported release. The rows can repeat and come in any order;
    the caller checks that they lie in the array.
    """
    indices = np.asarray(rows, dtype=np.int64)
    if indices.size == 0:
        return np.asarray(array[0:0])
    geometry = array_geometry(array)
    if geometry is None:
        return np.take(np.asarray(array), indices, axis=0)
    output = np.empty((indices.size, *array.shape[1:]), dtype=array.dtype)
    for _chunk, positions in _bin_groups(geometry, 0, indices):
        chunk_rows = indices[positions]
        start = int(chunk_rows.min())
        span = np.asarray(array[start : int(chunk_rows.max()) + 1])
        output[positions] = np.take(span, chunk_rows - start, axis=0)
        # A chunk's span goes before the next is read.
        del span
    return output


def partition_indices(
    geometry: ArrayGeometry,
    axis: int,
    indices: Sequence[int] | np.ndarray,
    *,
    maxWidth: int | None = None,
    fits: Fits | None = None,
) -> list[IndexBlock]:
    """Group selected indices into blocks aligned to physical chunks.

    With ``maxWidth`` the selection is cut into fixed-width blocks in request
    order. With ``fits`` adjacent chunks are packed while the predicate accepts
    the combined width, and a chunk too wide on its own is split. With neither,
    every chunk becomes one block.

    Raises:
        ValueError: If ``fits`` accepts no block of one index. The predicate
            hides what a block holds, so a caller whose predicate is a memory
            budget checks one index first and names the bytes it needs.
    """
    if maxWidth is not None and fits is not None:
        raise ValueError("Pass either maxWidth or fits, not both")
    resolved = checked_indices(indices, limit=geometry.shape[axis], name="indices")
    if resolved.size == 0:
        return []
    positions = np.arange(resolved.size, dtype=np.int64)

    if maxWidth is not None:
        width = max(1, operator.index(maxWidth))
        return [
            IndexBlock(
                indices=resolved[start : start + width],
                destinations=positions[start : start + width],
                bins=_bins(geometry, axis, resolved[start : start + width]),
            )
            for start in range(0, resolved.size, width)
        ]

    groups = _bin_groups(geometry, axis, resolved)
    if fits is None:
        return [
            IndexBlock(
                indices=resolved[member],
                destinations=positions[member],
                bins=(value,),
            )
            for value, member in groups
        ]

    blocks: list[IndexBlock] = []
    pending: list[np.ndarray] = []
    pending_bins: list[int] = []
    pending_width = 0
    previous_bin: int | None = None

    def flush() -> None:
        nonlocal pending_width, previous_bin
        if not pending:
            return
        member = np.concatenate(pending)
        blocks.append(
            IndexBlock(
                indices=resolved[member],
                destinations=positions[member],
                bins=tuple(pending_bins),
            )
        )
        pending.clear()
        pending_bins.clear()
        pending_width = 0
        previous_bin = None

    for value, member in groups:
        if not fits(member.size):
            flush()
            width = affordable_width(fits, member.size)
            if width < 1:
                raise ValueError(
                    "One index of the selection does not fit the operation budget: "
                    "fits accepts no block of one index"
                )
            for start in range(0, member.size, width):
                piece = member[start : start + width]
                blocks.append(
                    IndexBlock(
                        indices=resolved[piece],
                        destinations=positions[piece],
                        bins=(value,),
                    )
                )
            continue
        adjacent = previous_bin is None or value == previous_bin + 1
        if pending and (not adjacent or not fits(pending_width + member.size)):
            flush()
        pending.append(member)
        pending_bins.append(value)
        pending_width += int(member.size)
        previous_bin = value
    flush()
    return blocks
