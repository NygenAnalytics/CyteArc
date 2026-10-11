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
