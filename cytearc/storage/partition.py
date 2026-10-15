"""Row bands and chunk-aligned index blocks for streamed arrays."""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils.arrays import has_duplicates
from .geometry import ArrayGeometry

type Fits = Callable[[int], bool]

__all__ = [
    "IndexBlock",
    "affordable_width",
    "checked_indices",
    "contiguous_ranges",
    "indexed_read_parts",
    "is_contiguous",
    "partition_indices",
    "row_band",
    "row_read_parts",
    "scan_band",
    "widest_chunk_span",
]

# One multi-chunk read per band: Zarr fetches a band's chunks concurrently.
SCAN_BAND_BYTES = 64 * 1024**2


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
    # Steps are taken in int64: in a narrow integer type a descending step,
    # such as 255 to 0 in uint8, would wrap around to a step of one.
    return bool(np.all(np.diff(np.asarray(indices).astype(np.int64, copy=False)) == 1))


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


def _shard_column_chunks(geometry: ArrayGeometry, columns: np.ndarray) -> int:
    """Return the most inner column chunks that ``columns`` touch in one shard."""
    column_chunks = np.unique(geometry.binOf(1, columns))
    chunks_per_shard = -(-geometry.axisShard(1) // geometry.axisChunk(1))
    _, counts = np.unique(column_chunks // chunks_per_shard, return_counts=True)
    return int(counts.max())


def indexed_read_parts(
    geometry: ArrayGeometry,
    columns: np.ndarray,
    *,
    block_rows: int,
) -> tuple[int, int]:
    """Return per-decode and per-row bytes beside a read that indexes both axes.

    ``read_selection`` reads a sharded array's index arrays on both axes one
    stored chunk of one axis at a time, as a band that slices that axis over
    the indices the chunk holds, and holds the widest band beside the result
    while Zarr reads it. A band of columns has ``widest_chunk_span(columns)``
    values per row; a band of rows holds every selected column over at most
    one stored row chunk. The reader slices the axis whose band is narrower,
    so either bound covers its read; the one that is smaller for a block of
    ``block_rows`` rows is charged. The selected rows are not scanned, so a
    plan holds no copy of them.
    """
    if columns.size == 0:
        return 0, 0
    row_chunks = -(-geometry.axisShard(0) // geometry.axisChunk(0))
    # A column band touches a column chunk of each row chunk in a shard; a
    # row band touches a row chunk of each selected column chunk.
    touched = max(row_chunks, _shard_column_chunks(geometry, columns))
    fixed = geometry.readBytes(0, touched, decodes=1)
    column_band = widest_chunk_span(columns, geometry.axisChunk(1)) * geometry.itemsize
    per_row = geometry.readBytes(column_band, touched, decodes=1) - fixed
    row_span = min(geometry.axisChunk(0), geometry.shape[0])
    rows_band = columns.size * row_span * geometry.itemsize
    rows_fixed = geometry.readBytes(rows_band, touched, decodes=1)
    if max(1, int(block_rows)) * per_row <= rows_fixed - fixed:
        return int(fixed), int(per_row)
    return int(rows_fixed), 0


def row_read_parts(
    geometry: ArrayGeometry | None,
    columns: np.ndarray,
    *,
    rows_contiguous: bool,
) -> tuple[int, int]:
    """Return per-decode and per-row bytes beside a selected row block's result."""
    if geometry is None or columns.size == 0:
        return 0, 0
    row_chunks = -(-geometry.axisShard(0) // geometry.axisChunk(0))
    touched = row_chunks * _shard_column_chunks(geometry, columns)
    fixed = geometry.readBytes(0, touched, decodes=1)
    per_row = columns.size * geometry.itemsize if geometry.shards is not None else 0
    column_gather = not is_contiguous(columns)
    if not rows_contiguous and column_gather and geometry.shards is not None:
        # Sharded blocks that index both axes are read in bands. A block whose
        # rows form one run is read by Zarr in one request, so both are covered.
        band_fixed, band_per_row = indexed_read_parts(
            geometry, columns, block_rows=geometry.axisChunk(0)
        )
        fixed = max(fixed, band_fixed)
        per_row = max(per_row, band_per_row)
    if not rows_contiguous or column_gather:
        # Sorted selections, permutations, chunk labels, cumulative counts,
        # and their temporary arrays in Zarr's integer indexers.
        index_bytes = 16 * np.dtype(np.int64).itemsize
        grid = sum(
            -(-extent // geometry.axisShard(axis))
            for axis, extent in enumerate(geometry.shape)
        )
        fixed += index_bytes * (grid + (columns.size if column_gather else 0))
        if not rows_contiguous:
            per_row += index_bytes
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
                raise MemoryError(
                    "One index of the selection does not fit the operation budget"
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
