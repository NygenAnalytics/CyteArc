"""Arrays that a store computes, and the one seam through which CyteArc reads them.

A virtual array has Zarr metadata, but its chunks exist only as bytes that its
store computes on request, such as composite counts gathered from source
datastores. Reading those bytes through Zarr works, but a store that can
answer a selection directly does less work. Such a store implements
``VirtualArrayStore``, and CyteArc reads every count selection through
``read_selection`` and every feature stream through
``storage.feature_stream``, which hand virtual arrays to the store's reader.
"""

import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import zarr
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from zarr.abc.store import Store
from zarr.storage import (
    FsspecStore,
    LocalStore,
    MemoryStore,
    ObjectStore,
    WrapperStore,
)

from .async_execution import run_on_zarr_loop, run_read
from .types import array_metadata_shards

if TYPE_CHECKING:
    from .budget import ResourceBudget
    from .feature_stream import FeatureCellBand, FeatureReadGroup
    from .io_policy import StorageIoPolicy

type AxisSelection = slice | np.ndarray


class VirtualArrayReader(Protocol):
    """Read one virtual array directly from the data its store computes it from.

    ``read`` returns the same values as a Zarr read of the selection. The
    feature streams have the arguments, items, order, and memory contract of
    ``map_feature_read_groups`` and ``map_feature_cell_bands`` in
    ``storage.feature_stream``, which call them for a virtual array.
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    async def read(
        self, rows: AxisSelection, columns: AxisSelection, *, read_workers: int = 1
    ) -> np.ndarray:
        """Return ``rows`` by ``columns`` in the requested order."""
        ...

    def read_scratch_bytes(
        self, n_rows: int, n_columns: int, *, read_workers: int = 1
    ) -> int:
        """Return what one read of that many rows and columns holds beside its result."""
        ...

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
        """Stream the array's persisted read groups like ``map_feature_read_groups``."""
        ...

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
        """Stream feature groups by cell band like ``map_feature_cell_bands``."""
        ...


class VirtualArrayStore(ABC):
    """A Zarr store that computes some of its arrays from other data."""

    @abstractmethod
    def virtual_array_reader(self, path: str) -> VirtualArrayReader | None:
        """Return the reader of the virtual array at ``path``, or None for others."""

    @abstractmethod
    def chunk_read_scratch_bytes(self, path: str) -> int:
        """Return what one Zarr read of ``path``'s computed chunks holds beside them.

        ``array_geometry`` adds it to every Zarr read of the array, which runs
        when a caller reads the array without ``read_selection`` or through a
        wrapper of this store.
        """

    def stores_read(self, path: str) -> Sequence[Store] | None:
        """Return every store that a read of the array at ``path`` reads from, if known.

        ``read_selection`` gives a read an event loop of its own only when it
        knows each of these stores to work on any event loop. A store that
        returns None, as this default does, is read through Zarr's loop.
        """
        return None


def virtual_array_reader(array: zarr.Array) -> VirtualArrayReader | None:
    """Return the reader of ``array`` when its own store computes it.

    A wrapper around such a store, such as an instrumented one, sees every
    read of the array, so the reader is not used through it.
    """
    store = array.store
    if isinstance(store, VirtualArrayStore):
        return store.virtual_array_reader(array.path)
    return None


def _reads_on_any_loop(store: Any, path: str) -> bool:
    """Return whether every store that reading ``path`` of ``store`` touches works on any event loop.

    Known to: Zarr's local and memory stores, obstore's ``ObjectStore``, and an
    fsspec store over Zarr's wrapper of a synchronous file system, such as
    ``HfReadStore``, which runs each call on a thread. A native asynchronous
    fsspec file system, such as fsspec's HTTP or S3 ones, binds its client
    session to the first loop it runs on, and an unknown store may do the
    same. Wrappers are looked through, and a virtual array's store names the
    stores its reads touch.
    """
    while True:
        if isinstance(store, VirtualArrayStore):
            stores = store.stores_read(path)
            return stores is not None and all(
                _reads_on_any_loop(inner, path) for inner in stores
            )
        if isinstance(store, WrapperStore):
            store = store._store
            continue
        if isinstance(store, LocalStore | MemoryStore | ObjectStore):
            return True
        return (
            isinstance(store, FsspecStore)
            and isinstance(store.fs, AsyncFileSystemWrapper)
            and store.fs.semaphore is None
        )


def _run[T](array: zarr.Array, coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a read of ``array`` on an event loop of its own when its stores allow it.

    Reads that touch a store not known to work on any event loop run on
    Zarr's loop, as Zarr's synchronous API runs them. Either way, the stop of
    the block stream that runs the read cancels it.
    """
    if _reads_on_any_loop(array.store, array.path):
        return run_read(coroutine)
    return run_on_zarr_loop(coroutine)


def chunk_read_scratch_bytes(array: Any) -> int:
    """Return what one Zarr read of ``array``'s computed chunks holds beside them.

    The store that computes a virtual array answers through its
    ``chunk_read_scratch_bytes``. Zarr reads through a wrapper, such as a
    profiling one, still compute those chunks underneath it, so the method is
    found through wrappers that forward attributes and through Zarr's own
    ``WrapperStore``. ``virtual_array_reader`` does not look through them, so
    a wrapper's reads are never bypassed. Arrays of other stores hold none.
    """
    store = getattr(array, "store", None)
    while store is not None:
        scratch = getattr(store, "chunk_read_scratch_bytes", None)
        if callable(scratch):
            return operator.index(scratch(array.path))
        store = store._store if isinstance(store, WrapperStore) else None
    return 0


def _axis_selection(selection: AxisSelection, extent: int) -> AxisSelection:
    """Return a slice for an in-range index run that is provably contiguous.

    Only consecutive ascending integers inside ``extent`` become a slice, so
    it selects exactly the requested positions. Any other selection, an
    out-of-range one included, keeps its order and repeats and is checked by
    orthogonal indexing, which rejects indices outside the axis.
    """
    from .partition import is_contiguous

    if isinstance(selection, slice):
        return selection
    indices = np.asarray(selection)
    if (
        indices.size > 0
        and indices.dtype.kind in "iu"
        and is_contiguous(indices)
        and int(indices[-1]) < extent
    ):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def read_selection(
    array: zarr.Array, rows: AxisSelection, columns: AxisSelection
) -> np.ndarray:
    """Read ``rows`` by ``columns`` of a two-dimensional Zarr array.

    A virtual array is read through its store's reader. Any other array is
    read through Zarr, as a basic slice when both axes are contiguous runs and
    otherwise through orthogonal indexing, which preserves the order of index
    arrays. A sharded array's index arrays on both axes are read in bands;
    see ``_read_index_arrays``. A read whose stores all work on any event
    loop runs on an event loop of its own in the calling thread, through
    ``run_read``, so reads from several threads, such as the readers of a
    block stream, run at once; any other read runs on Zarr's loop.
    """
    reader = virtual_array_reader(array)
    if reader is not None:
        return _run(array, reader.read(rows, columns))
    row_selection = _axis_selection(rows, int(array.shape[0]))
    column_selection = _axis_selection(columns, int(array.shape[1]))
    if isinstance(row_selection, slice) and isinstance(column_selection, slice):
        return np.asarray(
            _run(array, array.async_array.getitem((row_selection, column_selection)))
        )
    if (
        isinstance(row_selection, np.ndarray)
        and isinstance(column_selection, np.ndarray)
        and array_metadata_shards(array) is not None
    ):
        return _read_index_arrays(array, row_selection, column_selection)
    return _orthogonal_selection(array, (row_selection, column_selection))


def _orthogonal_selection(
    array: zarr.Array, selection: tuple[AxisSelection, AxisSelection]
) -> np.ndarray:
    """Read an orthogonal selection of ``array``, on an event loop of its own when its stores allow it."""
    return np.asarray(
        _run(array, array.async_array.get_orthogonal_selection(selection))
    )


def _in_axis_indices(selection: np.ndarray, extent: int) -> np.ndarray | None:
    """Return ``selection`` as positions if it holds integers inside the axis."""
    if selection.ndim != 1 or selection.size == 0 or selection.dtype.kind not in "iu":
        return None
    if selection.min() < 0 or selection.max() >= extent:
        return None
    return selection.astype(np.intp, copy=False)


def _read_index_arrays(
    array: zarr.Array, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Read a sharded array's index arrays on both axes in bands.

    Zarr turns index arrays on both axes into coordinates of every selected
    value, and inside a shard those coordinates and their sorting hold many
    times the result; an unsharded read indexes each chunk directly and holds
    none. Instead, one stored chunk of one axis at a time is read as a band
    that slices the axis over the indices the chunk holds, with the other axis
    indexed by Zarr, and the band's selected positions are copied into the
    result in the requested order, repeats included. The axis whose widest
    band is narrower is sliced, so the read holds its result beside one Zarr
    read of that band, as ``indexed_read_parts`` charges.
    """
    from .partition import widest_chunk_span

    row_indices = _in_axis_indices(rows, int(array.shape[0]))
    column_indices = _in_axis_indices(columns, int(array.shape[1]))
    if row_indices is None or column_indices is None:
        # Zarr reads empty selections, negative indices, and boolean masks,
        # and rejects indices outside the axis before it reads anything.
        return _orthogonal_selection(array, (rows, columns))
    row_chunk, column_chunk = (operator.index(extent) for extent in array.chunks)
    column_band = row_indices.size * widest_chunk_span(column_indices, column_chunk)
    row_band = column_indices.size * widest_chunk_span(row_indices, row_chunk)
    if column_band <= row_band:
        return _read_bands(
            array, row_indices, column_indices, axis=1, chunk=column_chunk
        )
    return _read_bands(array, row_indices, column_indices, axis=0, chunk=row_chunk)


def _read_bands(
    array: zarr.Array,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    axis: int,
    chunk: int,
) -> np.ndarray:
    """Read ``rows`` by ``columns`` one stored chunk of ``axis`` at a time.

    Each band slices ``axis`` from the smallest to the largest index that one
    chunk holds, and Zarr indexes the other axis. A band's selected positions
    are copied in slabs of at most one chunk, so the copy holds no more than a
    decoded chunk beside the band.
    """
    spanned, indexed = (rows, columns) if axis == 0 else (columns, rows)
    order = np.argsort(spanned, kind="stable")
    values = spanned[order]
    bins = values // chunk
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    stops = np.r_[starts[1:], values.size]
    slab = max(1, math.prod(operator.index(extent) for extent in array.chunks))
    result: np.ndarray | None = None
    for first, last in zip(starts.tolist(), stops.tolist(), strict=True):
        start, stop = int(values[first]), int(values[last - 1]) + 1
        positions = order[first:last]
        local = values[first:last] - start
        band = _orthogonal_selection(
            array,
            (slice(start, stop), indexed)
            if axis == 0
            else (indexed, slice(start, stop)),
        )
        if result is None:
            result = np.empty((rows.size, columns.size), dtype=band.dtype)
        if axis == 0:
            step = max(1, slab // columns.size)
            for begin in range(0, positions.size, step):
                end = begin + step
                result[positions[begin:end]] = band[local[begin:end]]
        else:
            step = max(1, slab // positions.size)
            for begin in range(0, rows.size, step):
                end = begin + step
                result[begin:end, positions] = band[begin:end, local]
        del band
    assert result is not None
    return result
