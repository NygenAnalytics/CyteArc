import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore, WrapperStore

from cytearc.matrix import ChunkedArray
from cytearc.storage.geometry import array_geometry
from cytearc.storage.virtual_arrays import (
    VirtualArrayStore,
    read_selection,
    virtual_array_reader,
)

VALUES = np.arange(12 * 10, dtype=np.int32).reshape(12, 10)


class _Reader:
    def __init__(self, values: np.ndarray, reads: list) -> None:
        self.shape = values.shape
        self.dtype = values.dtype
        self._values = values
        self._reads = reads

    async def read(self, rows, columns, *, read_workers=1):
        self._reads.append((rows, columns))
        return self._values[rows][:, columns]

    def read_scratch_bytes(self, n_rows, n_columns, *, read_workers=1):
        return 7 * n_rows * n_columns


class _VirtualStore(WrapperStore, VirtualArrayStore):
    """A store whose array at ``virtual`` is read through a reader."""

    def __init__(self, store, values: np.ndarray | None = None) -> None:
        super().__init__(store)
        self.values = VALUES if values is None else values
        self.reads: list = []

    def _with_store(self, store):
        clone = type(self)(store, self.values)
        clone.reads = self.reads
        return clone

    def virtual_array_reader(self, path):
        return _Reader(self.values, self.reads) if path == "virtual" else None

    def chunk_read_scratch_bytes(self, path):
        return 1234 if path == "virtual" else 0


@pytest.fixture
def plain():
    return zarr.create_array(store=MemoryStore(), data=VALUES, chunks=(4, 3))


@pytest.fixture
def virtual():
    memory = MemoryStore()
    for path in ("virtual", "stored"):
        zarr.create_array(store=memory, name=path, data=VALUES, chunks=(4, 3))
    store = _VirtualStore(memory)
    return store, {
        path: zarr.open_array(store=store, path=path, mode="r")
        for path in ("virtual", "stored")
    }


@pytest.mark.parametrize(
    ("rows", "columns"),
    (
        (slice(None), slice(None)),
        (slice(2, 9), slice(1, 4)),
        (np.array([3, 4, 5]), np.array([0, 1, 2, 3])),
        (np.array([7, 1, 7]), np.array([9, 0])),
        (np.array([], dtype=np.int64), slice(None)),
        (np.array([5]), np.array([6])),
        (slice(0, 12, 3), np.array([2, 3, 4])),
    ),
)
def test_read_selection_matches_numpy_for_plain_arrays(plain, rows, columns):
    expected = VALUES[rows][:, columns]
    actual = read_selection(plain, rows, columns)
    assert actual.dtype == VALUES.dtype
    np.testing.assert_array_equal(actual, expected)


def test_read_selection_dispatches_only_virtual_arrays(virtual):
    store, arrays = virtual
    assert virtual_array_reader(arrays["stored"]) is None
    assert virtual_array_reader(arrays["virtual"]) is not None
    rows, columns = np.array([5, 2]), slice(1, 3)
    np.testing.assert_array_equal(
        read_selection(arrays["stored"], rows, columns), VALUES[rows][:, columns]
    )
    assert store.reads == []
    np.testing.assert_array_equal(
        read_selection(arrays["virtual"], rows, columns), VALUES[rows][:, columns]
    )
    assert len(store.reads) == 1


def test_virtual_store_adds_chunk_read_scratch_to_geometry(plain, virtual):
    _, arrays = virtual
    assert array_geometry(plain).readScratchBytes == 0
    assert array_geometry(arrays["stored"]).readScratchBytes == 0
    assert array_geometry(arrays["virtual"]).readScratchBytes == 1234


def test_chunked_arrays_read_virtual_arrays_through_their_reader(virtual):
    store, arrays = virtual
    chunked = ChunkedArray(arrays["virtual"])
    np.testing.assert_array_equal(
        chunked[[1, 4, 9]][:, [0, 7]].compute(), VALUES[[1, 4, 9]][:, [0, 7]]
    )
    assert store.reads
    store.reads.clear()
    np.testing.assert_array_equal(ChunkedArray(arrays["stored"]).compute(), VALUES)
    assert store.reads == []


@pytest.mark.parametrize(
    ("rows", "columns"),
    (
        (slice(None), np.array([9, 10])),
        (slice(None), np.array([10, 11])),
        (np.array([11, 12]), slice(None)),
    ),
)
def test_read_selection_rejects_out_of_range_index_runs(plain, rows, columns):
    with pytest.raises(IndexError):
        read_selection(plain, rows, columns)


def test_chunked_arrays_reject_out_of_range_columns(plain):
    with pytest.raises(IndexError):
        ChunkedArray(plain, cols=np.array([9, 10])).compute()


BANDED = np.random.default_rng(29).integers(0, 1000, size=(40, 36)).astype(np.uint16)


@pytest.fixture(params=("sharded", "unsharded"))
def banded(request):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array(
        "counts",
        shape=BANDED.shape,
        chunks=(8, 6),
        shards=(16, 18) if request.param == "sharded" else None,
        dtype=BANDED.dtype,
    )
    array[:] = BANDED
    return array


def _mask(size: int, selected: list[int]) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[selected] = True
    return mask


INDEX_ARRAYS = {
    "sorted": (np.array([0, 3, 4, 9, 17, 33, 39]), np.array([1, 2, 8, 20, 35])),
    "unsorted": (np.array([33, 4, 17, 0]), np.array([35, 1, 20, 8])),
    "repeated": (np.array([5, 5, 30, 5]), np.array([7, 7, 0, 30, 7])),
    "negative": (np.array([-1, 2, -40]), np.array([-36, 5, -1])),
    "masks": (_mask(40, [1, 2, 20, 39]), _mask(36, [0, 12, 13, 35])),
    "unsigned": (
        np.array([39, 2, 11], dtype=np.uint8),
        np.array([35, 0], dtype=np.uint64),
    ),
    # Nearly every row and three columns: the rows are read in bands.
    "row bands": (np.delete(np.arange(40), [7, 21]), np.array([0, 17, 35])),
    # Two rows and nearly every column: the columns are read in bands.
    "column bands": (np.array([1, 38]), np.delete(np.arange(36), [5, 30])),
}


@pytest.mark.parametrize(
    ("rows", "columns"), INDEX_ARRAYS.values(), ids=INDEX_ARRAYS.keys()
)
def test_read_selection_reads_index_arrays_on_both_axes_like_zarr(
    banded, rows, columns
):
    given = (rows.copy(), columns.copy())

    actual = read_selection(banded, rows, columns)

    assert actual.dtype == BANDED.dtype
    np.testing.assert_array_equal(actual, BANDED[rows][:, columns])
    np.testing.assert_array_equal(
        actual, banded.get_orthogonal_selection((rows.copy(), columns.copy()))
    )
    # Zarr wraps negative indices in place; the reader leaves its input alone.
    np.testing.assert_array_equal(rows, given[0])
    np.testing.assert_array_equal(columns, given[1])


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize(
    ("rows", "columns"), INDEX_ARRAYS.values(), ids=INDEX_ARRAYS.keys()
)
def test_bands_of_either_axis_read_the_selection(banded, rows, columns, axis):
    from cytearc.storage.virtual_arrays import _read_bands

    row_indices = np.arange(BANDED.shape[0])[rows]
    column_indices = np.arange(BANDED.shape[1])[columns]

    actual = _read_bands(
        banded, row_indices, column_indices, axis=axis, chunk=banded.chunks[axis]
    )

    np.testing.assert_array_equal(actual, BANDED[rows][:, columns])


def test_sharded_reads_never_send_index_arrays_on_both_axes_to_zarr(monkeypatch):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array(
        "counts", shape=BANDED.shape, chunks=(8, 6), shards=(16, 18), dtype=np.uint16
    )
    array[:] = BANDED
    selections = []
    original = zarr.Array.get_orthogonal_selection

    def record(self, selection, *args, **kwargs):
        selections.append(selection)
        return original(self, selection, *args, **kwargs)

    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", record)
    # Zarr reads negative indices and boolean masks itself.
    for name, (rows, columns) in INDEX_ARRAYS.items():
        if name not in ("negative", "masks"):
            read_selection(array, rows, columns)

    assert selections
    assert not any(
        all(isinstance(axis, np.ndarray) for axis in selection)
        for selection in selections
    )


@pytest.mark.parametrize(
    ("rows", "columns"),
    (
        (np.array([0, 40]), np.array([1, 3])),
        (np.array([0, 2]), np.array([3, 36])),
        (np.array([-41, 2]), np.array([1, 3])),
        (np.array([0, 2], dtype=np.uint64), np.array([1, 2**63], dtype=np.uint64)),
        (np.ones(39, dtype=bool), np.array([1, 3])),
    ),
    ids=(
        "row past the end",
        "column past the end",
        "row before the start",
        "unsigned column past the end",
        "short row mask",
    ),
)
def test_read_selection_rejects_index_arrays_outside_the_axes(banded, rows, columns):
    with pytest.raises(IndexError):
        read_selection(banded, rows, columns)


class _Forwarding(WrapperStore):
    def __getattr__(self, name):
        return getattr(self._store, name)


@pytest.mark.parametrize("wrapper", ("forwarding", "zarr", "profiling"))
def test_wrapped_virtual_store_keeps_its_chunk_read_scratch(virtual, wrapper):
    from profiling.recording_store import RecordingStoreWrapper
    from zarr.storage import LoggingStore

    store, _ = virtual
    wrapped = {
        "forwarding": _Forwarding,
        "zarr": LoggingStore,
        "profiling": RecordingStoreWrapper,
    }[wrapper](store)
    array = zarr.open_array(store=wrapped, path="virtual", mode="r")
    # Zarr reads through the wrapper still compute the chunks underneath it.
    assert array_geometry(array).readScratchBytes == 1234
    assert (
        array_geometry(
            zarr.open_array(store=wrapped, path="stored", mode="r")
        ).readScratchBytes
        == 0
    )
    # The wrapper sees every read, so the reader does not bypass it.
    assert virtual_array_reader(array) is None


@pytest.mark.parametrize("dtype", (np.uint8, np.int8, np.uint16, np.int64))
def test_narrow_integer_selections_read_the_requested_columns(dtype):
    values = np.arange(3 * 300, dtype=np.int32).reshape(3, 300)
    array = zarr.create_array(store=MemoryStore(), data=values, chunks=(2, 64))
    for columns in ([127, 0], [5, 6, 7], [126, 127]):
        selected = np.asarray(columns, dtype=dtype)
        expected = values[:, np.asarray(columns)]
        np.testing.assert_array_equal(
            read_selection(array, slice(None), selected), expected
        )
        np.testing.assert_array_equal(
            ChunkedArray(array, cols=selected).compute(), expected
        )
    if np.iinfo(dtype).max >= 255:
        selected = np.asarray([255, 0], dtype=dtype)
        np.testing.assert_array_equal(
            read_selection(array, slice(None), selected), values[:, [255, 0]]
        )
        np.testing.assert_array_equal(
            ChunkedArray(array, cols=selected).compute(), values[:, [255, 0]]
        )
