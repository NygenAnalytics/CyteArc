"""Bounded array selections from composite source matrices."""

import asyncio
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from numba import njit

from ..storage.geometry import ArrayGeometry, array_geometry
from ..storage.virtual_arrays import AxisSelection
from .source_cache import _finish_io

if TYPE_CHECKING:
    from ..storage.budget import ResourceBudget
    from ..storage.feature_stream import FeatureCellBand, FeatureReadGroup
    from ..storage.io_policy import StorageIoPolicy
    from .source_cache import CachedSourceStore


@njit(cache=True, nogil=True)
def _copy_selected_values(
    values: np.ndarray,
    output: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    row_destinations: np.ndarray,
    column_destinations: np.ndarray,
) -> None:
    if len(rows) != len(row_destinations) or len(columns) != len(column_destinations):
        raise ValueError("Source and destination index lengths must match")
    for index in rows:
        if index < -values.shape[0] or index >= values.shape[0]:
            raise IndexError("Source row index is out of bounds")
    for index in columns:
        if index < -values.shape[1] or index >= values.shape[1]:
            raise IndexError("Source column index is out of bounds")
    for index in row_destinations:
        if index < -output.shape[0] or index >= output.shape[0]:
            raise IndexError("Destination row index is out of bounds")
    for index in column_destinations:
        if index < -output.shape[1] or index >= output.shape[1]:
            raise IndexError("Destination column index is out of bounds")
    for row_start in range(0, len(rows), 32):
        for column_start in range(0, len(columns), 32):
            for row in range(row_start, min(row_start + 32, len(rows))):
                source_row = rows[row]
                destination_row = row_destinations[row]
                for column in range(column_start, min(column_start + 32, len(columns))):
                    output[destination_row, column_destinations[column]] = values[
                        source_row, columns[column]
                    ]


def _axis_indices(selection: AxisSelection, extent: int) -> np.ndarray:
    """Return a selection along one axis as validated non-negative indices."""
    if isinstance(selection, slice):
        return np.arange(*selection.indices(extent), dtype=np.int64)
    indices = np.asarray(selection)
    if indices.ndim != 1:
        raise ValueError("Array selections must be one-dimensional")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if indices.dtype.kind not in "iu":
        raise TypeError("Array selections must contain integers")
    if np.any(indices >= extent) or (
        indices.dtype.kind == "i" and np.any(indices < -extent)
    ):
        raise IndexError("Array selection contains an out-of-range index")
    indices = indices.astype(np.int64, copy=True)
    indices[indices < 0] += extent
    return indices


def _chunk_groups(indices: np.ndarray, width: int) -> list[tuple[np.ndarray, int, int]]:
    """Group indices by the chunk of ``width`` they fall in.

    Each group holds the positions of its indices in ``indices``, its
    smallest index, and one past its largest, in chunk order.
    """
    bins = indices // width
    order = np.argsort(bins, kind="stable")
    edges = np.flatnonzero(np.diff(bins[order])) + 1
    groups = []
    for positions in np.split(order, edges):
        selected = indices[positions]
        groups.append((positions, int(selected.min()), int(selected.max()) + 1))
    return groups


def source_index_bytes(geometry: ArrayGeometry, n_rows: int, n_columns: int) -> int:
    """Return the index bookkeeping of one read of a source selection.

    A read keeps a few Python objects for every chunk along each axis that the
    selection touches.
    """
    return 384 * sum(
        min(size, -(-extent // width))
        for size, extent, width in zip(
            (n_rows, n_columns), geometry.shape, geometry.chunks, strict=True
        )
    )


def _mapping_selection(indices: np.ndarray) -> slice | np.ndarray:
    """Read a contiguous run of mapping entries as one slice."""
    from ..storage.partition import is_contiguous

    if len(indices) and is_contiguous(indices):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def selection_scratch_bytes(
    mappings: Mapping[str, ArrayGeometry],
    sources: Sequence[Mapping[str, ArrayGeometry]],
    *,
    cells: int,
    features: int,
    read_workers: int = 1,
) -> int:
    """Return what one composite read of that many cells and features holds.

    That is what the read holds beside its result: the mapping windows of its
    cells and features or ``read_workers`` admitted source rectangles,
    whichever is larger, and the index bookkeeping of its source reads.
    ``mappings`` holds the geometry of each mapping array and ``sources`` the
    ``counts`` and ``countsT`` geometry of each source.
    """
    if not cells or not features:
        return 0
    mapping_scratch = 0
    for name, size in (
        ("rowSources", cells),
        ("sourceRows", cells),
        ("featureColumns", features),
    ):
        geometry = mappings[name]
        axis = len(geometry.shape) - 1
        chunks = min(size, -(-geometry.shape[axis] // geometry.axisChunk(axis)))
        mapping_scratch = max(
            mapping_scratch,
            geometry.readBytes(size * geometry.itemsize, chunks=chunks),
        )
    source_scratch = read_workers * max(
        max(
            geometry.readBytes(geometry.nominalChunkBytes())
            + geometry.nominalChunkBytes(),
            2 * geometry.nominalChunkBytes()
            + 64 * (geometry.axisChunk(0) + geometry.axisChunk(1)),
        )
        for geometries in sources
        for geometry in geometries.values()
    )
    block_index_bytes = max(
        source_index_bytes(
            geometry,
            *((features, cells) if name == "countsT" else (cells, features)),
        )
        for geometries in sources
        for name, geometry in geometries.items()
    )
    # Selection copies and mapping indexers coexist with per-chunk Python
    # descriptors, shard coverage maps and the admitted source rectangles.
    return (
        192 * (cells + features)
        + block_index_bytes
        + max(mapping_scratch, source_scratch)
    )


@dataclass(frozen=True, slots=True)
class ShardPrefetchPlan:
    """The source shards that one selection reads densely enough to fetch whole.

    ``shardKey(row_start, column_start)`` names the object of the shard that
    holds that source position, or is None when the selection reads too few
    of the shard's chunks to fetch it whole. One fetch keeps at most
    ``maxBytes``, two decoded source chunks.
    """

    store: "CachedSourceStore"
    maxBytes: int
    shardKey: Callable[[int, int], str | None]


def shard_prefetch_plan(
    source: zarr.Array,
    rows: np.ndarray,
    columns: np.ndarray,
    geometry: ArrayGeometry,
) -> ShardPrefetchPlan | None:
    """Plan whole-shard fetches of a cached source, or None when none pays off.

    A shard is fetched whole when the selection reads at least half of its
    inner chunks, so its encoded object costs one request instead of one per
    chunk.
    """
    from .source_cache import CachedSourceStore

    store = source.store
    if not isinstance(store, CachedSourceStore) or geometry.shards is None:
        return None

    def bands(indices: np.ndarray, axis: int) -> dict[int, tuple[int, int]]:
        shard_width = geometry.axisShard(axis)
        chunk_width = geometry.axisChunk(axis)
        selected = {}
        for positions, start, _stop in _chunk_groups(indices, shard_width):
            shard = start // shard_width
            touched = len(np.unique(indices[positions] // chunk_width))
            extent = min(shard_width, geometry.shape[axis] - shard * shard_width)
            selected[shard] = (touched, -(-extent // chunk_width))
        return selected

    first_axis = 0 if len(rows) <= len(columns) else 1
    selections = (rows, columns)
    first_bands = bands(selections[first_axis], first_axis)
    if all(2 * touched < total for touched, total in first_bands.values()):
        return None
    second_bands = bands(selections[1 - first_axis], 1 - first_axis)
    row_bands, column_bands = (
        (first_bands, second_bands) if first_axis == 0 else (second_bands, first_bands)
    )

    def shard_key(start_row: int, start_column: int) -> str | None:
        row = start_row // geometry.axisShard(0)
        column = start_column // geometry.axisShard(1)
        row_touches, row_chunks = row_bands[row]
        column_touches, column_chunks = column_bands[column]
        touched = row_touches * column_touches
        if touched < 2 or 2 * touched < row_chunks * column_chunks:
            return None
        return "/".join(
            part
            for part in (source.path, source.metadata.encode_chunk_key((row, column)))
            if part
        )

    return ShardPrefetchPlan(store, 2 * geometry.nominalChunkBytes(), shard_key)


async def copy_source_selection(
    source: zarr.Array,
    rows: np.ndarray,
    columns: np.ndarray,
    output: np.ndarray,
    row_destinations: np.ndarray,
    column_destinations: np.ndarray,
    *,
    geometry: ArrayGeometry | None = None,
    read_workers: int = 1,
) -> None:
    """Copy ``source[rows, columns]`` into ``output`` at the destination indices.

    Each source chunk that the selection touches is read once, as the smallest
    rectangle that holds its selected values, and ``read_workers`` rectangles
    are read at once. ``output`` is two-dimensional, or four-dimensional in
    the encoded chunk order of a virtual shard.
    """
    read_workers = operator.index(read_workers)
    if read_workers < 1:
        raise ValueError("Source read workers must be positive")
    if geometry is None:
        geometry = array_geometry(source)
        assert geometry is not None
    if not rows.size or not columns.size:
        return
    if (
        np.any(rows < 0)
        or np.any(rows >= geometry.shape[0])
        or np.any(columns < 0)
        or np.any(columns >= geometry.shape[1])
    ):
        raise IndexError("Source selection contains an out-of-range index")
    if geometry.shards is not None:
        from .source_cache import CachedSourceStore

        if isinstance(source.store, CachedSourceStore):
            # Index copies precede payload decoding and share its chunk allowance.
            source.store.cache_shard_indexes(source.path, geometry.nominalChunkBytes())
    prefetch = shard_prefetch_plan(source, rows, columns, geometry)
    row_groups = _chunk_groups(rows, geometry.axisChunk(0))
    column_groups = _chunk_groups(columns, geometry.axisChunk(1))
    rectangles = iter((row, column) for row in row_groups for column in column_groups)

    async def copy_rectangles() -> None:
        assert geometry is not None
        for row, column in rectangles:
            row_positions, start_row, stop_row = row
            column_positions, start_column, stop_column = column
            if prefetch is not None:
                key = prefetch.shardKey(start_row, start_column)
                if key is not None:
                    await prefetch.store.prefetch(key, max_bytes=prefetch.maxBytes)
            values = np.asarray(
                await _finish_io(
                    source.async_array.getitem(
                        (slice(start_row, stop_row), slice(start_column, stop_column))
                    )
                )
            )
            # Repeated requested indices must not enlarge the scatter temporary.
            for row_start in range(0, len(row_positions), geometry.axisChunk(0)):
                row_piece = row_positions[row_start : row_start + geometry.axisChunk(0)]
                for column_start in range(
                    0, len(column_positions), geometry.axisChunk(1)
                ):
                    column_piece = column_positions[
                        column_start : column_start + geometry.axisChunk(1)
                    ]
                    source_rows = rows[row_piece] - start_row
                    source_columns = columns[column_piece] - start_column
                    destination_rows = row_destinations[row_piece]
                    destination_columns = column_destinations[column_piece]
                    if (
                        output.ndim == 2
                        and (output.flags.c_contiguous or output.flags.f_contiguous)
                        and output.flags.writeable
                        and values.flags.c_contiguous
                        and values.dtype == output.dtype
                        and values.dtype.isnative
                        and (
                            values.dtype.kind in "biu"
                            and values.dtype.itemsize <= 8
                            or values.dtype.kind == "f"
                            and values.dtype.itemsize in (4, 8)
                        )
                        and len(row_piece) >= 32
                        and len(column_piece) >= 32
                        and len(row_piece) * len(column_piece) >= 1_000_000
                    ):
                        _copy_selected_values(
                            values,
                            output,
                            source_rows,
                            source_columns,
                            destination_rows,
                            destination_columns,
                        )
                        continue
                    destination_rows, destination_columns = np.ix_(
                        destination_rows, destination_columns
                    )
                    destination: tuple[np.ndarray, ...]
                    if output.ndim == 4:
                        chunk_rows, chunk_columns = output.shape[2:]
                        destination = (
                            destination_rows // chunk_rows,
                            destination_columns // chunk_columns,
                            destination_rows % chunk_rows,
                            destination_columns % chunk_columns,
                        )
                    else:
                        destination = (destination_rows, destination_columns)
                    output[destination] = values[np.ix_(source_rows, source_columns)]
            del values

    if read_workers == 1:
        await copy_rectangles()
    else:
        async with asyncio.TaskGroup() as group:
            for _ in range(min(read_workers, len(row_groups) * len(column_groups))):
                group.create_task(copy_rectangles())


class CompositeArrayReader:
    """Read one virtual composite count array from its source matrices.

    A read maps each selected cell to its source and row, and each selected
    feature to its column in every source, then copies each source's values
    from whichever of its ``counts`` and ``countsT`` touches fewer chunk bytes.
    Its feature streams, in ``composite.streams``, batch adjacent feature
    groups into one such read.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        dtype: np.dtype[Any],
        transposed: bool,
        mappings: dict[str, zarr.Array],
        sources: list[dict[str, zarr.Array]],
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self._transposed = transposed
        self._mappings = mappings
        self._sources = sources
        self._mapping_geometry: dict[str, ArrayGeometry] = {}
        self._source_geometry: list[dict[str, ArrayGeometry]] = []
        for name, array in mappings.items():
            geometry = array_geometry(array)
            assert geometry is not None
            self._mapping_geometry[name] = geometry
        for arrays in sources:
            geometries = {}
            for name, array in arrays.items():
                geometry = array_geometry(array)
                assert geometry is not None
                geometries[name] = geometry
            self._source_geometry.append(geometries)

    def read_scratch_bytes(
        self, n_rows: int, n_columns: int, *, read_workers: int = 1
    ) -> int:
        read_workers = operator.index(read_workers)
        if read_workers < 1:
            raise ValueError("Source read workers must be positive")
        if n_rows < 0 or n_columns < 0:
            raise ValueError("Array selection dimensions cannot be negative")
        cells, features = (
            (n_columns, n_rows) if self._transposed else (n_rows, n_columns)
        )
        return selection_scratch_bytes(
            self._mapping_geometry,
            self._source_geometry,
            cells=cells,
            features=features,
            read_workers=read_workers,
        )

    async def read(
        self,
        rows: AxisSelection,
        columns: AxisSelection,
        *,
        read_workers: int = 1,
    ) -> np.ndarray:
        row_indices = _axis_indices(rows, self.shape[0])
        column_indices = _axis_indices(columns, self.shape[1])
        output = np.zeros((len(row_indices), len(column_indices)), dtype=self.dtype)
        await self._fill(row_indices, column_indices, output, read_workers)
        return output

    async def fill(
        self,
        rows: AxisSelection,
        columns: AxisSelection,
        output: np.ndarray,
        *,
        read_workers: int = 1,
    ) -> None:
        """Write ``rows`` by ``columns`` into ``output``.

        ``output`` holds the selection in this array's orientation, either as
        a two-dimensional array or in the encoded chunk order of one virtual
        shard. Features that a source lacks keep their values in ``output``.
        """
        await self._fill(
            _axis_indices(rows, self.shape[0]),
            _axis_indices(columns, self.shape[1]),
            output,
            read_workers,
        )

    async def _fill(
        self,
        row_indices: np.ndarray,
        column_indices: np.ndarray,
        output: np.ndarray,
        read_workers: int,
    ) -> None:
        read_workers = operator.index(read_workers)
        if read_workers < 1:
            raise ValueError("Source read workers must be positive")
        if not len(row_indices) or not len(column_indices):
            return
        cells, features = (
            (column_indices, row_indices)
            if self._transposed
            else (row_indices, column_indices)
        )
        cell_output = output.T if self._transposed else output
        owners = np.asarray(
            await self._mappings["rowSources"].async_array.get_orthogonal_selection(
                (_mapping_selection(cells),)
            )
        )
        source_rows = np.asarray(
            await self._mappings["sourceRows"].async_array.get_orthogonal_selection(
                (_mapping_selection(cells),)
            )
        )
        if np.any(owners < 0) or np.any(owners >= len(self._sources)):
            raise ValueError("Composite row mapping names an invalid source")
        for owner in np.unique(owners):
            source = self._sources[int(owner)]
            geometries = self._source_geometry[int(owner)]
            cell_destinations = np.flatnonzero(owners == owner)
            selected_rows = source_rows[cell_destinations]
            mapped_features = np.asarray(
                await self._mappings[
                    "featureColumns"
                ].async_array.get_orthogonal_selection(
                    (int(owner), _mapping_selection(features))
                )
            )
            feature_destinations = np.flatnonzero(mapped_features >= 0)
            selected_features = mapped_features[feature_destinations]
            if (
                np.any(selected_rows < 0)
                or np.any(selected_rows >= source["counts"].shape[0])
                or np.any(selected_features >= source["counts"].shape[1])
                or np.any(mapped_features < -1)
            ):
                raise ValueError("Composite mapping is outside a source matrix")
            if not len(selected_features):
                continue
            if output.ndim == 2:
                costs = {}
                for name, geometry in geometries.items():
                    first, second = (
                        (selected_features, selected_rows)
                        if name == "countsT"
                        else (selected_rows, selected_features)
                    )
                    costs[name] = (
                        len(np.unique(geometry.binOf(0, first)))
                        * len(np.unique(geometry.binOf(1, second)))
                        * geometry.nominalChunkBytes()
                    )
                name = min(costs, key=costs.__getitem__)
            else:
                # A shard's chunk order fixes its orientation, so it is
                # filled from the source orientation that matches it.
                name = "countsT" if self._transposed else "counts"
            transposed = name == "countsT"
            target = (
                output
                if output.ndim != 2
                else cell_output.T
                if transposed
                else cell_output
            )
            await copy_source_selection(
                source[name],
                selected_features if transposed else selected_rows,
                selected_rows if transposed else selected_features,
                target,
                feature_destinations if transposed else cell_destinations,
                cell_destinations if transposed else feature_destinations,
                geometry=geometries[name],
                read_workers=read_workers,
            )

    def map_feature_read_groups[T](
        self,
        array: zarr.Array,
        process: Callable[["FeatureReadGroup"], T],
        *,
        cell_idx: np.ndarray | None,
        feat_idx: Sequence[int] | np.ndarray | None,
        resources: "ResourceBudget | None",
        progress: str | None,
        io: "StorageIoPolicy | None",
        metrics: dict[str, Any] | None,
        scratchBytes: int,
        orderedCompute: bool,
    ) -> Iterator[T]:
        from .streams import map_read_groups

        return map_read_groups(
            self,
            array,
            process,
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            resources=resources,
            progress=progress,
            io=io,
            metrics=metrics,
            scratchBytes=scratchBytes,
            orderedCompute=orderedCompute,
        )

    def map_feature_cell_bands[T](
        self,
        array: zarr.Array,
        process: Callable[["FeatureCellBand"], T],
        *,
        cell_idx: np.ndarray | None,
        feat_idx: Sequence[int] | np.ndarray | None,
        resources: "ResourceBudget | None",
        progress: str | None,
        io: "StorageIoPolicy | None",
        metrics: dict[str, Any] | None,
        scratchBytes: int,
        orderedCompute: bool,
        cellMajorOrder: bool,
    ) -> Iterator[T]:
        from .streams import map_cell_bands

        return map_cell_bands(
            self,
            array,
            process,
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            resources=resources,
            progress=progress,
            io=io,
            metrics=metrics,
            scratchBytes=scratchBytes,
            orderedCompute=orderedCompute,
            cellMajorOrder=cellMajorOrder,
        )
