"""Arrays that a store computes, and the one seam through which CyteArc reads them.

A virtual array has Zarr metadata, but its chunks exist only as bytes that its
store computes on request, such as composite counts gathered from source
datastores. Reading those bytes through Zarr works, but a store that can
answer a selection directly does less work. Such a store implements
``VirtualArrayStore``, and CyteArc reads every count selection through
``read_selection`` and every feature stream through
``storage.feature_stream``, which hand virtual arrays to the store's reader.
"""

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import zarr
from zarr.storage import WrapperStore

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


def virtual_array_reader(array: zarr.Array) -> VirtualArrayReader | None:
    """Return the reader of ``array`` when its own store computes it.

    A wrapper around such a store, such as an instrumented one, sees every
    read of the array, so the reader is not used through it.
    """
    store = array.store
    if isinstance(store, VirtualArrayStore):
        return store.virtual_array_reader(array.path)
    return None


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
    arrays. Call it outside Zarr's event loop, like any synchronous Zarr read.
    """
    reader = virtual_array_reader(array)
    if reader is not None:
        from zarr.core.sync import sync

        return sync(reader.read(rows, columns))
    row_selection = _axis_selection(rows, int(array.shape[0]))
    column_selection = _axis_selection(columns, int(array.shape[1]))
    if isinstance(row_selection, slice) and isinstance(column_selection, slice):
        return np.asarray(array[row_selection, column_selection])
    return np.asarray(array.get_orthogonal_selection((row_selection, column_selection)))
