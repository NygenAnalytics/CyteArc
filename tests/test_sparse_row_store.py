"""The row bucket store returns rows by destination exactly as they were appended.

References are built independently of the store: dense arrays filled entry by
entry in the order the entries were appended, SciPy row permutations, and
sorts of the appended coordinates.
"""

import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numcodecs
import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, vstack

from cytearc.readers import _sparse
from cytearc.readers._sparse import ROW_BYTES, SparseRowStore, encoded_entry_bytes
from cytearc.utils.memory_budget import MemoryBudgetError
from tests.row_store_helpers import asked_bytes, free_disk

_BIG = 1 << 40
# The operation that the stores of these tests name in their errors.
_OPERATION = "sparse row conversion"


def _whole_store(tmp_path, shape, dtype, **options) -> SparseRowStore:
    settings = {"bucket_rows": 7, "buffer_bytes": 4096, "bucket_bytes": _BIG}
    settings.update(options)
    return SparseRowStore(
        shape,
        dtype,
        operation=_OPERATION,
        whole_rows=True,
        temp_dir=tmp_path,
        **settings,
    )


def _loose_store(tmp_path, shape, dtype, **options) -> SparseRowStore:
    settings = {"bucket_rows": 7, "buffer_bytes": 4096, "bucket_bytes": _BIG}
    settings.update(options)
    return SparseRowStore(
        shape,
        dtype,
        operation=_OPERATION,
        whole_rows=False,
        temp_dir=tmp_path,
        **settings,
    )


def _random_rows(rng, n_rows, n_columns, dtype, density=0.3) -> csr_matrix:
    mask = rng.random((n_rows, n_columns)) < density
    values = np.zeros((n_rows, n_columns), dtype=dtype)
    count = int(mask.sum())
    if np.dtype(dtype).kind == "b":
        values[mask] = True
    elif np.dtype(dtype).kind == "f":
        values[mask] = rng.normal(0, 1e3, count).astype(dtype)
    else:
        info = np.iinfo(dtype)
        values[mask] = rng.integers(
            info.min, info.max, count, dtype=dtype, endpoint=True
        )
    matrix = csr_matrix(values)
    return matrix


def _bits(values: np.ndarray) -> np.ndarray:
    """Values as raw bytes, so NaN, infinities and negative zero compare exactly."""
    return np.ascontiguousarray(values).view(np.uint8)


def _assert_same_rows(found: csr_matrix, expected: csr_matrix) -> None:
    assert found.shape == expected.shape
    assert found.dtype == expected.dtype
    expected = csr_matrix(expected)
    expected.sort_indices()
    np.testing.assert_array_equal(found.indptr, expected.indptr)
    np.testing.assert_array_equal(found.indices, expected.indices)
    np.testing.assert_array_equal(_bits(found.data), _bits(expected.data))


DTYPES = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bucket_rows", [1, 7, 60])
def test_whole_rows_return_in_destination_order(tmp_path, dtype, bucket_rows):
    rng = np.random.default_rng(bucket_rows)
    source = _random_rows(rng, 60, 23, dtype)
    order = rng.permutation(60)
    destinations = np.empty(60, dtype=np.int64)
    destinations[order] = np.arange(60)
    store = _whole_store(tmp_path, source.shape, dtype, bucket_rows=bucket_rows)
    try:
        for start in range(0, 60, 11):
            store.append_rows(
                destinations[start : start + 11], source[start : start + 11]
            )
        store.finalize()
        expected = source[order]
        _assert_same_rows(store.read(0, 60), expected)
        # Windows that start and end inside buckets and cross their boundaries.
        for start, stop in ((0, 1), (3, 17), (6, 8), (13, 60), (59, 60), (20, 20)):
            window = store.read(start, stop)
            _assert_same_rows(window, expected[start:stop])
            assert window.has_canonical_format
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "values",
    [
        np.array([np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max]),
        np.array([0, 1, np.iinfo(np.uint64).max], dtype=np.uint64),
        np.array([0, 255, 256, 65535, 65536, 2**32 - 1, 2**32], dtype=np.uint64),
        np.array([-0.0, 0.0, 1.0, np.nan, np.inf, -np.inf, 2.5], dtype=np.float64),
        np.array([2.0**32 - 1, 2.0**32, 2.0**53, 1e300], dtype=np.float64),
        np.array([16777216.0, 4294967040.0, 1.0], dtype=np.float32),
        np.array([1 + 2j, 0, -3j], dtype=np.complex64),
    ],
    ids=[
        "int64-extremes",
        "uint64-max",
        "unsigned-boundaries",
        "float-specials",
        "float-beyond-uint32",
        "float32-large-integers",
        "complex",
    ],
)
@pytest.mark.parametrize("whole_rows", [True, False])
def test_values_round_trip_bit_for_bit(tmp_path, values, whole_rows):
    n = values.size
    matrix = csr_matrix(
        (values, np.arange(n) % 3, np.arange(n + 1)), shape=(n, 3), dtype=values.dtype
    )
    store = SparseRowStore(
        matrix.shape,
        values.dtype,
        operation=_OPERATION,
        bucket_rows=2,
        whole_rows=whole_rows,
        buffer_bytes=0,
        bucket_bytes=_BIG,
        temp_dir=tmp_path,
    )
    try:
        rows = np.arange(n)[::-1].copy()
        if whole_rows:
            store.append_rows(rows, matrix[::-1])
        else:
            store.append_entries(rows, (np.arange(n) % 3)[::-1], values[::-1])
        store.finalize()
        _assert_same_rows(store.read(0, n), matrix)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (np.array([0, 255], dtype=np.int64), 0),
        (np.array([256], dtype=np.int32), 1),
        (np.array([2**32 - 1], dtype=np.uint64), 2),
        (np.array([2**32], dtype=np.uint64), 3),
        (np.array([-1, 5], dtype=np.int16), _sparse._SOURCE_VALUES),
        (np.array([True, False]), 0),
        (np.array([1.0, 70000.0], dtype=np.float32), 2),
        (np.array([2.0**32], dtype=np.float64), _sparse._SOURCE_VALUES),
        (np.array([0.5], dtype=np.float32), _sparse._SOURCE_VALUES),
        (np.array([-0.0, 1.0], dtype=np.float64), _sparse._SOURCE_VALUES),
        (np.array([np.nan], dtype=np.float64), _sparse._SOURCE_VALUES),
        (np.array([0.0, 3.0, 65504.0], dtype=np.float16), 1),
        (np.array([np.inf], dtype=np.float64), _sparse._SOURCE_VALUES),
        (np.array([1 + 0j], dtype=np.complex128), _sparse._SOURCE_VALUES),
        (np.array([], dtype=np.float64), 0),
    ],
)
def test_a_block_narrows_values_only_when_every_value_is_exact(values, code):
    assert _sparse._value_code(values) == code


def _appended_reference(rows, columns, values, shape, dtype) -> np.ndarray:
    """Sum entries in the order they were appended, in 64 bits for integers."""
    accumulate = np.float64 if np.dtype(dtype).kind == "f" else np.int64
    dense = np.zeros(shape, dtype=accumulate)
    np.add.at(dense, (rows, columns), values.astype(accumulate))
    return dense.astype(dtype)


@pytest.mark.parametrize("bucket_rows", [1, 5, 40])
@pytest.mark.parametrize("source", [np.int16, np.uint8, np.float32])
def test_loose_entries_join_pieces_and_sum_duplicates(tmp_path, bucket_rows, source):
    rng = np.random.default_rng(bucket_rows)
    shape = (40, 17)
    count = 900
    rows = rng.integers(0, shape[0], count)
    # Rows 3 and 30 stay empty.
    rows[np.isin(rows, (3, 30))] = 4
    columns = rng.integers(0, shape[1], count)
    values = rng.integers(0, 100, count).astype(source)
    dtype = np.int64 if np.dtype(source).kind in "iu" else np.dtype(source)
    store = _loose_store(
        tmp_path, shape, dtype, bucket_rows=bucket_rows, source_dtype=source
    )
    try:
        # Many appends of unsorted columns, a row in pieces across them.
        for start in range(0, count, 37):
            store.append_entries(
                rows[start : start + 37],
                columns[start : start + 37],
                values[start : start + 37],
            )
        store.finalize()
        found = store.read(0, shape[0])
        assert found.has_canonical_format
        assert found.dtype == np.dtype(dtype)
        expected = _appended_reference(rows, columns, values, shape, dtype)
        np.testing.assert_array_equal(found.toarray(), expected)
        np.testing.assert_array_equal(found.getnnz(axis=1)[[3, 30]], [0, 0])
        # The pointers count appended entries, at least those that a read holds.
        np.testing.assert_array_equal(
            np.diff(store.indptr), np.bincount(rows, minlength=shape[0])
        )
    finally:
        store.close()


def test_a_summed_value_past_its_dtype_is_refused(tmp_path):
    store = _loose_store(tmp_path, (1, 1), np.uint8, source_dtype=np.int64)
    try:
        store.append_entries([0, 0], [0, 0], [200, 100])
        store.finalize()
        with pytest.raises(OverflowError, match="exceed the destination dtype"):
            store.read(0, 1)
    finally:
        store.close()


def test_duplicates_kept_for_a_later_sum_stay_in_the_order_they_arrived(tmp_path):
    rng = np.random.default_rng(5)
    shape = (12, 6)
    count = 400
    rows = rng.integers(0, shape[0], count)
    columns = rng.integers(0, shape[1], count)
    # Distinct values, so the order of a coordinate's duplicates is visible.
    values = rng.permutation(count).astype(np.float32)
    store = _loose_store(tmp_path, shape, np.float32, sum_duplicates=False)
    try:
        for start in range(0, count, 50):
            store.append_entries(
                rows[start : start + 50],
                columns[start : start + 50],
                values[start : start + 50],
            )
        store.finalize()
        found = store.read(0, shape[0])
        order = np.lexsort((columns, rows))
        np.testing.assert_array_equal(found.indices, columns[order])
        np.testing.assert_array_equal(found.data, values[order])
        np.testing.assert_array_equal(
            found.indptr, np.r_[0, np.cumsum(np.bincount(rows, minlength=shape[0]))]
        )
        assert found.has_sorted_indices
        assert not found.has_canonical_format
    finally:
        store.close()


@pytest.mark.parametrize("summed", [True, False])
def test_whole_rows_with_unsorted_or_repeated_columns_are_sorted(tmp_path, summed):
    matrix = csr_matrix(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([2, 0, 2, 1, 1]),
            np.array([0, 3, 3, 5]),
        ),
        shape=(3, 3),
    )
    store = _whole_store(
        tmp_path, matrix.shape, np.int64, source_dtype=np.int32, sum_duplicates=summed
    )
    try:
        store.append_rows([2, 0, 1], matrix)
        store.finalize()
        found = store.read(0, 3)
        if summed:
            np.testing.assert_array_equal(
                found.toarray(), [[0, 0, 0], [0, 9, 0], [2, 0, 4]]
            )
            assert found.has_canonical_format
        else:
            np.testing.assert_array_equal(found.indptr, [0, 0, 2, 5])
            np.testing.assert_array_equal(found.indices, [1, 1, 0, 2, 2])
            np.testing.assert_array_equal(found.data, [4, 5, 2, 1, 3])
    finally:
        store.close()


@pytest.mark.parametrize("whole_rows", [True, False])
@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (5, 3)])
def test_empty_axes_rows_and_buckets_round_trip(tmp_path, whole_rows, shape):
    store = SparseRowStore(
        shape,
        np.float32,
        operation=_OPERATION,
        bucket_rows=2,
        whole_rows=whole_rows,
        buffer_bytes=0,
        bucket_bytes=_BIG,
        temp_dir=tmp_path,
    )
    try:
        if whole_rows and shape[0]:
            store.append_rows(np.arange(shape[0]), csr_matrix(shape, dtype=np.float32))
        store.finalize()
        found = store.read(0, shape[0])
        assert found.shape == shape and found.nnz == 0
        assert [batch.shape[0] for batch in store.iter_rows(0, shape[0], 2)] == [
            min(2, shape[0] - start) for start in range(0, shape[0], 2)
        ]
        assert store.read_bytes() >= 0
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


def test_one_huge_row_fills_a_bucket_of_its_own(tmp_path):
    shape = (6, 5000)
    rows = np.full(5000, 4)
    columns = np.arange(5000)[::-1].copy()
    values = np.arange(1, 5001, dtype=np.int32)
    store = _loose_store(
        tmp_path, shape, np.int64, bucket_rows=3, source_dtype=np.int32
    )
    try:
        for start in range(0, 5000, 700):
            store.append_entries(
                rows[start : start + 700],
                columns[start : start + 700],
                values[start : start + 700],
            )
        store.append_entries([0], [3], [7])
        store.finalize()
        found = store.read(0, 6)
        expected = np.zeros(shape, dtype=np.int64)
        expected[4, columns] = values
        expected[0, 3] = 7
        np.testing.assert_array_equal(found.toarray(), expected)
    finally:
        store.close()


def test_a_bucket_that_one_read_cannot_hold_is_split_at_finalize(tmp_path):
    rng = np.random.default_rng(2)
    shape = (40, 30)
    # Skewed rows: the first five hold almost every entry.
    rows = np.concatenate([rng.integers(0, 5, 2000), rng.integers(5, 40, 40)])
    columns = rng.integers(0, 30, rows.size)
    values = rng.integers(1, 9, rows.size).astype(np.int32)
    widest = int(np.bincount(rows).max())
    # Room for the widest row and a few more entries, far from the bucket.
    limit = _sparse._bucket_bytes(widest + 20, widest + 20, 2, 8, True)
    store = _loose_store(
        tmp_path,
        shape,
        np.int64,
        bucket_rows=40,
        bucket_bytes=limit,
        source_dtype=np.int32,
        buffer_bytes=2048,
    )
    try:
        for start in range(0, rows.size, 300):
            store.append_entries(
                rows[start : start + 300],
                columns[start : start + 300],
                values[start : start + 300],
            )
        store.finalize()
        assert store.bucket_starts.size > 2
        assert store.bucket_starts[0] == 0 and store.bucket_starts[-1] == 40
        assert store.read_bytes() <= limit + store.bucket_starts.nbytes
        expected = _appended_reference(rows, columns, values, shape, np.int64)
        np.testing.assert_array_equal(store.read(0, 40).toarray(), expected)
        # Every bucket directory names its rows; the split one left nothing.
        names = sorted(path.name for path in store.directory.iterdir())
        assert not any(name.startswith("split-") for name in names)
        reopened = SparseRowStore.open(store.directory)
        np.testing.assert_array_equal(reopened.read(0, 40).toarray(), expected)
    finally:
        store.close()


def _sample_rows(seed: int) -> tuple[csr_matrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = _random_rows(rng, 90, 31, np.float32)
    order = rng.permutation(90)
    destinations = np.empty(90, dtype=np.int64)
    destinations[order] = np.arange(90)
    return matrix, destinations


def _write_part(directory: str, part: int, first: int, last: int) -> None:
    matrix, destinations = _sample_rows(11)
    writer = SparseRowStore.open(directory, part=part, buffer_bytes=2048)
    writer.append_rows(destinations[first:last], matrix[first:last])
    writer.finish()


def _write_entries(directory: str, part: int) -> None:
    writer = SparseRowStore.open(directory, part=part, buffer_bytes=1024)
    writer.append_entries([0, 1, 0], [2, 2, 2], [float(part), 1.0, 10.0 * part])
    writer.finish()


def _read_window(directory: str, start: int, stop: int) -> np.ndarray:
    reader = SparseRowStore.open(directory)
    try:
        return reader.read(start, stop).toarray()
    finally:
        reader.close()


def test_writers_in_other_processes_append_and_readers_open_by_path(tmp_path):
    matrix, destinations = _sample_rows(11)
    order = np.argsort(destinations)
    store = _whole_store(tmp_path, matrix.shape, np.float32, bucket_rows=8)
    try:
        directory = str(store.directory)
        with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
            futures = [
                pool.submit(_write_part, directory, part, first, last)
                for part, (first, last) in enumerate(((0, 40), (40, 90)), start=1)
            ]
            for future in futures:
                future.result()
            store.finalize()
            window = pool.submit(_read_window, directory, 5, 77).result()
        np.testing.assert_array_equal(window, matrix[order][5:77].toarray())
        _assert_same_rows(store.read(0, 90), matrix[order])
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


def test_parts_join_in_part_order(tmp_path):
    store = _loose_store(tmp_path, (2, 3), np.float64, sum_duplicates=False)
    try:
        # Part 0, the owner's, then parts 2 and 1 written out of order.
        store.append_entries([0], [2], [-1.0])
        for part in (2, 1):
            _write_entries(str(store.directory), part)
        store.finalize()
        found = store.read(0, 2)
        np.testing.assert_array_equal(
            found.data, [-1.0, 1.0, 10.0, 2.0, 20.0, 1.0, 1.0]
        )
    finally:
        store.close()


def test_whole_rows_must_arrive_exactly_once_across_writers(tmp_path):
    matrix = csr_matrix(np.eye(4, dtype=np.float32))
    store = _whole_store(tmp_path, (4, 4), np.float32)
    try:
        with pytest.raises(ValueError, match="row 1 was appended more than once"):
            store.append_rows([1, 1], matrix[:2])
        store.append_rows([0], matrix[:1])
        with pytest.raises(ValueError, match="row 0 was appended more than once"):
            store.append_rows([0], matrix[:1])
        writer = SparseRowStore.open(store.directory, part=1, buffer_bytes=0)
        writer.append_rows([0, 2], matrix[:2])
        writer.finish()
        with pytest.raises(ValueError, match="row 0 was appended by more than one"):
            store.finalize()
    finally:
        store.close()
    missing = _whole_store(tmp_path, (4, 4), np.float32)
    try:
        missing.append_rows([0, 2], matrix[:2])
        with pytest.raises(
            ValueError, match="2 destination rows were never appended, the first 1"
        ):
            missing.finalize()
    finally:
        missing.close()


def test_a_writer_that_did_not_finish_stops_finalize(tmp_path):
    store = _whole_store(tmp_path, (2, 2), np.float32)
    try:
        writer = SparseRowStore.open(store.directory, part=3)
        writer.append_rows([0, 1], csr_matrix(np.eye(2, dtype=np.float32)))
        with pytest.raises(RuntimeError, match="part-0003 .* did not finish"):
            store.finalize()
        with pytest.raises(RuntimeError, match="already being written"):
            SparseRowStore.open(store.directory, part=3)
    finally:
        store.close()


def test_wrong_inputs_are_refused_with_what_was_expected(tmp_path):
    whole = _whole_store(tmp_path, (3, 4), np.float32)
    loose = _loose_store(tmp_path, (3, 4), np.float32)
    narrow = _loose_store(tmp_path, (3, 4), np.int16)
    try:
        with pytest.raises(TypeError, match="takes whole rows; call append_rows"):
            whole.append_entries([0], [0], [1.0])
        with pytest.raises(TypeError, match="takes loose entries; call append_entries"):
            loose.append_rows([0], csr_matrix((1, 4), dtype=np.float32))
        with pytest.raises(IndexError, match=r"Destination rows must lie in \[0, 3\)"):
            whole.append_rows([3], csr_matrix((1, 4), dtype=np.float32))
        with pytest.raises(IndexError, match=r"Destination rows must lie in \[0, 3\)"):
            loose.append_entries([-1], [0], [1.0])
        with pytest.raises(IndexError, match=r"Column indices must lie in \[0, 4\)"):
            loose.append_entries([0], [4], [1.0])
        with pytest.raises(IndexError, match=r"Column indices must lie in \[0, 4\)"):
            whole.append_rows(
                [0],
                csr_matrix(
                    (np.ones(1, np.float32), np.array([5]), [0, 1]), shape=(1, 4)
                ),
            )
        with pytest.raises(ValueError, match="takes 1 rows of 4 columns"):
            whole.append_rows([0], csr_matrix((2, 4), dtype=np.float32))
        with pytest.raises(TypeError, match="takes a CSR matrix"):
            whole.append_rows([0], coo_matrix((1, 4), dtype=np.float32))
        with pytest.raises(TypeError, match="one-dimensional integer array"):
            whole.append_rows([0.5], csr_matrix((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="of one length"):
            loose.append_entries([0, 1], [0], [1.0, 2.0])
        with pytest.raises(OverflowError, match="cannot be represented"):
            narrow.append_entries([0], [0], [0.5])
        with pytest.raises(OverflowError, match="exceed the destination dtype"):
            narrow.append_entries([0], [0], [70000])
        with pytest.raises(RuntimeError, match="once its owner finalizes it"):
            loose.read(0, 1)
        loose.finalize()
        with pytest.raises(RuntimeError, match="takes no more rows"):
            loose.append_entries([0], [0], [1.0])
        with pytest.raises(RuntimeError, match="finalizes a sparse row store, once"):
            loose.finalize()
        with pytest.raises(IndexError, match=r"window \[2, 4\) is outside"):
            loose.read(2, 4)
        with pytest.raises(IndexError, match=r"window \[2, 1\) is outside"):
            next(loose.iter_rows(2, 1, 1))
        with pytest.raises(ValueError, match="batch_rows must be at least 1"):
            next(loose.iter_rows(0, 1, 0))
    finally:
        whole.close()
        loose.close()
        narrow.close()
    for options, message in (
        ({"bucket_rows": 0}, "bucket_rows must be at least 1"),
        ({"parts": 0}, "parts must be at least 1, not 0"),
        ({"buffer_bytes": -1}, "buffer_bytes must not be negative"),
        ({"bucket_bytes": -1}, "bucket_bytes must not be negative"),
    ):
        with pytest.raises(ValueError, match=message):
            _whole_store(tmp_path, (3, 4), np.float32, **options)
    with pytest.raises(ValueError, match="must not be negative"):
        _whole_store(tmp_path, (-1, 4), np.float32)
    assert list(tmp_path.iterdir()) == []


def test_opening_needs_a_store_and_reading_needs_it_finalized(tmp_path):
    with pytest.raises(RuntimeError, match="holds no sparse row store"):
        SparseRowStore.open(tmp_path)
    store = _whole_store(tmp_path, (2, 2), np.float32)
    try:
        with pytest.raises(RuntimeError, match="is not finalized"):
            SparseRowStore.open(store.directory)
        with pytest.raises(ValueError, match="part must be at least 0"):
            SparseRowStore.open(store.directory, part=-1)
        settings = store.directory / "store.json"
        settings.write_text(settings.read_text().replace('"format": 1', '"format": 9'))
        with pytest.raises(RuntimeError, match="format 9 is not format 1"):
            SparseRowStore.open(store.directory, part=1)
        store.append_rows([0, 1], csr_matrix((2, 2), dtype=np.float32))
        store.finalize()
        with pytest.raises(RuntimeError, match="no part to finish"):
            store.finish()
    finally:
        store.close()


def _finalized_rows(
    tmp_path, n_rows=30, bucket_rows=6
) -> tuple[SparseRowStore, csr_matrix]:
    rng = np.random.default_rng(n_rows)
    matrix = _random_rows(rng, n_rows, 9, np.float32, density=0.5)
    store = _whole_store(tmp_path, matrix.shape, np.float32, bucket_rows=bucket_rows)
    store.append_rows(np.arange(n_rows)[::-1].copy(), matrix[::-1])
    store.finalize()
    return store, matrix


@pytest.mark.parametrize("read_ahead", [False, True])
def test_reading_in_batches_deletes_buckets_inside_the_window(tmp_path, read_ahead):
    store, matrix = _finalized_rows(tmp_path)
    try:
        batches = list(store.iter_rows(4, 25, 5, read_ahead=read_ahead, delete=True))
        assert [batch.shape[0] for batch in batches] == [5, 5, 5, 5, 1]
        _assert_same_rows(vstack(batches, format="csr"), matrix[4:25])
        # Buckets [6, 12), [12, 18) and [18, 24) lie inside the window.
        names = sorted(path.name for path in store.directory.glob("rows-*"))
        assert [name for name in names if name.endswith(".read")] == [
            f"rows-{first:012d}.read" for first in (6, 12, 18)
        ]
        for start, stop in ((7, 8), (18, 24)):
            with pytest.raises(
                RuntimeError, match="deleted once a read had taken them"
            ):
                store.read(start, stop)
        # The buckets at the window's edges are kept for their other rows.
        _assert_same_rows(store.read(0, 6), matrix[:6])
        _assert_same_rows(store.read(24, 30), matrix[24:30])
    finally:
        store.close()


def test_read_ahead_returns_the_same_batches_and_raises_its_errors(
    tmp_path, monkeypatch
):
    store, matrix = _finalized_rows(tmp_path)
    try:
        plain = list(store.iter_rows(0, 30, 4))
        ahead = list(store.iter_rows(0, 30, 4, read_ahead=True))
        for first, second in zip(plain, ahead, strict=True):
            _assert_same_rows(first, second)
        loaded = []
        original = store._load_bucket

        def failing(bucket):
            loaded.append(bucket)
            if bucket == 2:
                raise OSError("injected read failure")
            return original(bucket)

        monkeypatch.setattr(store, "_load_bucket", failing)
        batches = store.iter_rows(0, 30, 6, read_ahead=True)
        _assert_same_rows(next(batches), matrix[:6])
        _assert_same_rows(next(batches), matrix[6:12])
        with pytest.raises(OSError, match="injected read failure"):
            next(batches)
        # Bucket 2 was read ahead while bucket 1 was read, and never again.
        assert loaded == [0, 1, 2]
    finally:
        store.close()


def test_a_read_holds_no_more_than_the_store_reports(tmp_path):
    rng = np.random.default_rng(9)
    shape = (300, 2000)
    rows = rng.integers(0, shape[0], 60_000)
    columns = rng.integers(0, shape[1], rows.size)
    values = rng.integers(0, 50, rows.size).astype(np.float32)
    for summed in (True, False):
        store = _loose_store(
            tmp_path,
            shape,
            np.float32,
            bucket_rows=100,
            buffer_bytes=1 << 20,
            sum_duplicates=summed,
        )
        try:
            store.append_entries(rows, columns, values)
            store.finalize()
            reported = store.read_bytes()
            tracemalloc.start()
            try:
                for bucket in range(store.bucket_starts.size - 1):
                    tracemalloc.reset_peak()
                    base = tracemalloc.get_traced_memory()[0]
                    store._cache = None
                    first, last = store.bucket_starts[bucket : bucket + 2]
                    store.read(int(first), int(last))
                    assert tracemalloc.get_traced_memory()[1] - base <= reported
            finally:
                tracemalloc.stop()
        finally:
            store.close()


def test_pending_pieces_stay_within_the_buffer(tmp_path):
    rng = np.random.default_rng(4)
    store = _loose_store(
        tmp_path, (50, 40), np.float64, bucket_rows=5, buffer_bytes=8000
    )
    try:
        for _ in range(30):
            rows = rng.integers(0, 50, 20)
            store.append_entries(rows, rng.integers(0, 40, 20), rng.random(20))
            assert store._buckets._buffered <= 8000 // 4
        store.finalize()
    finally:
        store.close()


@pytest.mark.parametrize("ragged", [False, True], ids=["sparse", "ragged"])
def test_encoding_a_block_holds_at_most_three_copies_of_its_pieces(ragged):
    # Pending pieces take a quarter of the buffer and the encoding of one
    # block the rest, three copies of it, though ragged rows stay uncompressed.
    rng = np.random.default_rng(13)
    lengths = rng.integers(0, 200, 20_000)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    entries = int(lengths.sum())
    if ragged:
        indices = np.zeros(0, dtype=np.int32)
        values = rng.integers(32, 127, entries).astype(np.uint8)
    else:
        indices = ((np.arange(entries) - starts) * 3).astype(np.int32)
        values = rng.random(entries).astype(np.float32)
    piece = (np.arange(lengths.size, dtype=np.int64), lengths, indices, values)
    held = sum(int(array.nbytes) for array in piece)
    tracemalloc.start()
    try:
        _sparse._encode_block(
            [piece], n_columns=600, bucket_rows=lengths.size, ragged=ragged
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= 3 * held


def _bucket_file(store: SparseRowStore, first_row: int = 0) -> Path:
    return next(store.directory.glob(f"rows-{first_row:012d}/part-*.bin"))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda data: data[:-3], "holds .* bytes where .* were written"),
        (lambda data: b"", "holds 0 bytes where .* were written"),
        (lambda data: b"XXXX" + data[4:], "does not start a block"),
        (lambda data: data[:-1] + bytes([data[-1] ^ 0xFF]), "fails its checksum"),
        # The checksum covers the header too.
        (lambda data: data[:8] + bytes([0xEE]) + data[9:], "fails its checksum"),
    ],
    ids=["truncated", "emptied", "magic", "payload", "header"],
)
def test_damaged_files_are_refused(tmp_path, damage, message):
    store, _ = _finalized_rows(tmp_path)
    try:
        path = _bucket_file(store)
        path.write_bytes(damage(path.read_bytes()))
        with pytest.raises(RuntimeError, match=message):
            store.read(0, 1)
    finally:
        store.close()


def _sign(block: bytes) -> bytes:
    """Return a block whose checksum matches its header and payloads again."""
    size = _sparse._HEADER.size - _sparse._CHECKSUM.size
    payloads = [block[_sparse._HEADER.size :]]
    checksum = _sparse._checksum(block[:size], payloads)
    return block[:size] + _sparse._CHECKSUM.pack(checksum) + payloads[0]


def _replace_header(block: bytes, **fields: int) -> bytes:
    names = (
        "magic flags row length index value pieces entries "
        "rowBytes lengthBytes indexBytes valueBytes checksum"
    ).split()
    values = dict(zip(names, _sparse._HEADER.unpack_from(block), strict=True))
    values.update(fields)
    header = _sparse._HEADER.pack(*(values[name] for name in names))
    return _sign(header + block[_sparse._HEADER.size :])


def test_files_whose_size_matches_but_whose_blocks_end_early_are_refused(tmp_path):
    store, _ = _finalized_rows(tmp_path)
    try:
        path = _bucket_file(store)
        data = path.read_bytes()

        def write(changed: bytes) -> None:
            path.write_bytes(changed)
            store._fileBytes[store._fileBytes[:, 0] > 0, 0] = len(changed)
            store._cache = None

        # A header that claims more payload than the file holds.
        fields = list(_sparse._HEADER.unpack_from(data))
        fields[-2] += 50
        write(_sparse._HEADER.pack(*fields) + data[_sparse._HEADER.size :])
        with pytest.raises(RuntimeError, match="payloads take .* bytes and .* remain"):
            store.read(0, 1)
        # A second block cut off inside its header.
        write(data + data[:10])
        with pytest.raises(RuntimeError, match="Block 1 .* header is incomplete"):
            store.read(0, 1)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"value": 0xEE}, "names an unknown dtype"),
        ({"row": 7}, "names an unknown dtype"),
        ({"pieces": 99}, "decodes to .* values where its header records 99"),
        ({"entries": 1}, "decodes to .* values where its header records 1"),
    ],
)
def test_signed_blocks_that_contradict_their_header_are_refused(
    tmp_path, fields, message
):
    store, _ = _finalized_rows(tmp_path)
    try:
        path = _bucket_file(store)
        data = path.read_bytes()
        changed = _replace_header(data, **fields)
        path.write_bytes(changed)
        with pytest.raises(RuntimeError, match=message):
            store.read(0, 1)
    finally:
        store.close()


def test_a_missing_file_is_refused(tmp_path):
    store, _ = _finalized_rows(tmp_path)
    try:
        _bucket_file(store, 6).unlink()
        with pytest.raises(RuntimeError, match=r"rows \[6, 12\), is missing"):
            store.read(5, 7)
    finally:
        store.close()


def test_blocks_that_disagree_with_the_written_rows_are_refused(tmp_path):
    store, matrix = _finalized_rows(tmp_path)
    try:
        path = _bucket_file(store)

        def write(rows, lengths, indices, values):
            block, _ = _sparse._encode_block(
                [
                    (
                        np.asarray(rows),
                        np.asarray(lengths),
                        np.asarray(indices),
                        np.asarray(values),
                    )
                ],
                n_columns=9,
                bucket_rows=6,
            )
            path.write_bytes(block)
            store._fileBytes[:, 0] = 0
            store._fileBytes[0, 0] = len(block)
            store._cache = None

        entries = int(store.indptr[6])
        # Row 0 twice and row 5 never.
        write(
            [0, 0, 1, 2, 3, 4],
            [entries, 0, 0, 0, 0, 0],
            np.zeros(entries, int),
            np.ones(entries, np.float32),
        )
        with pytest.raises(
            RuntimeError, match="do not hold each of those rows exactly once"
        ):
            store.read(0, 1)
        write(
            [0],
            [entries + 1],
            np.zeros(entries + 1, int),
            np.ones(entries + 1, np.float32),
        )
        with pytest.raises(RuntimeError, match="hold more than the .* entries"):
            store.read(0, 1)
        write([0], [1], [0], np.ones(1, np.float32))
        with pytest.raises(RuntimeError, match="hold 1 entries where"):
            store.read(0, 1)
        write([7], [1], [0], np.ones(1, np.float32))
        with pytest.raises(RuntimeError, match="holds a row outside its bucket"):
            store.read(0, 1)
        # One piece of one entry that holds two.
        write([0], [1], [0, 0], np.ones(2, np.float32))
        with pytest.raises(
            RuntimeError, match="holds pieces of 1 entries where its header records 2"
        ):
            store.read(0, 1)
        write([0], [1], [9], np.ones(1, np.float32))
        with pytest.raises(RuntimeError, match="holds a column outside the matrix"):
            store.read(0, 1)
    finally:
        store.close()


def test_a_full_disk_is_refused_before_and_while_writing(tmp_path, monkeypatch):
    free = {"bytes": 0}
    monkeypatch.setattr(
        "cytearc.readers._sparse.shutil.disk_usage",
        lambda _: SimpleNamespace(free=free["bytes"]),
    )
    asked = {}
    for entries in (10, 20):
        with pytest.raises(
            OSError, match="temporary bytes .* which has 0 free"
        ) as full:
            _loose_store(tmp_path, (5, 3), np.float32, entries=entries)
        asked[entries] = asked_bytes(full.value)
    assert list(tmp_path.iterdir()) == []
    # Ten more float32 values with uint8 column indices, each its own piece
    # of 16 bytes at most: 10 * (4 + 1 + 16) bytes before compression.
    assert asked[20] - asked[10] == 210
    free["bytes"] = asked[10]
    store = _loose_store(tmp_path, (5, 3), np.float32, entries=10, buffer_bytes=0)
    try:
        free["bytes"] = 10
        with pytest.raises(OSError, match="temporary bytes .* which has 10 free"):
            store.append_entries([0], [0], [1.0])
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


def _store_files(directory: Path) -> dict[str, int]:
    """The bytes of a store's files by kind: its settings, its records and its buckets' files."""
    kinds = {"settings": 0, "records": 0, "buckets": 0}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if path.name == "store.json":
            kinds["settings"] += size
        elif path.suffix == ".npz":
            kinds["records"] += size
        elif path.suffix == ".bin":
            kinds["buckets"] += size
        else:
            # The mark of an open part holds nothing.
            assert size == 0, path
    return kinds


@pytest.mark.parametrize(
    ("form", "writers"),
    [("whole", 0), ("loose", 0), ("ragged", 0), ("whole", 2)],
    ids=["whole", "loose", "ragged", "whole-from-two-writers"],
)
def test_the_disk_check_counts_every_file_that_the_store_writes(
    tmp_path, form, writers
):
    rng = np.random.default_rng(41)
    destinations = rng.permutation(300)
    matrix = _random_rows(rng, 300, 40, np.float32, density=0.1)
    values, offsets = _ragged_rows(rng, 300)
    if form == "ragged":
        entries = int(offsets[-1])
        # Each byte once, without a column index, and each row's place and
        # length in two codes of at most 8 bytes.
        rows = entries + 300 * 16
    elif form == "whole":
        entries = int(matrix.nnz)
        # float32 values and uint8 column indices before compression, and
        # each row's place and length in at most 16 bytes.
        rows = entries * (4 + 1) + 300 * 16
    else:
        entries = int(matrix.nnz)
        # Each entry may be a piece of its own, of 16 bytes at most.
        rows = entries * (4 + 1 + 16)
    # Each part's file in each of the six buckets of 50 rows holds at least
    # one block: a header of 64 bytes and four Blosc frames, each with a
    # header of 16 bytes.
    blocks = 6 * (1 + writers) * (64 + 4 * 16)

    def create() -> SparseRowStore:
        # Rows wait until they are finished, so each bucket takes one block.
        options = {
            "bucket_rows": 50,
            "buffer_bytes": 1 << 20,
            "entries": entries,
            "parts": 1 + writers,
        }
        if form == "ragged":
            return _ragged_store(tmp_path, 300, **options)
        builder = _whole_store if form == "whole" else _loose_store
        return builder(tmp_path, matrix.shape, np.float32, **options)

    with free_disk(0), pytest.raises(OSError, match="which has 0 free") as full:
        create()
    asked = asked_bytes(full.value)
    # One byte short, the store is refused before it writes anything.
    with free_disk(asked - 1), pytest.raises(OSError, match=f"needs {asked} "):
        create()
    assert list(tmp_path.iterdir()) == []
    with free_disk(asked):
        store = create()
        try:
            if form == "ragged":
                _append_in_blocks(store, destinations, values, offsets)
            elif form == "loose":
                coo = matrix.tocoo()
                store.append_entries(destinations[coo.row], coo.col, coo.data)
            elif not writers:
                store.append_rows(destinations, matrix)
            else:
                # Other writers send half of the rows each, as the processes
                # of a shuffle do, and the owner's part stays empty.
                halves = np.array_split(np.arange(300), writers)
                for part, share in enumerate(halves, start=1):
                    writer = SparseRowStore.open(
                        store.directory, part=part, buffer_bytes=1 << 20
                    )
                    writer.append_rows(destinations[share], matrix[share])
                    writer.finish()
            store.finalize()
            written = _store_files(store.directory)
        finally:
            store.close()
    # The check asks for the rows as it bounds them, a block for each part
    # in each bucket, and every byte of the store's settings and records:
    # those of its own part, of each other writer's part and the finalized
    # record.
    assert asked == rows + blocks + written["settings"] + written["records"]
    assert sum(written.values()) <= asked


def test_a_bucket_limit_below_one_row_is_refused(tmp_path):
    with pytest.raises(MemoryBudgetError) as raised:
        _whole_store(tmp_path, (2, 2), np.float32, bucket_bytes=10)
    assert raised.value.item == "reading one entry of the sparse row conversion"
    assert raised.value.needed_bytes == _sparse._bucket_bytes(1, 1, 1, 4, True)
    limit = _sparse._bucket_bytes(3, 1, 1, 4, True)
    store = _whole_store(tmp_path, (2, 4), np.float32, bucket_bytes=limit - 1)
    try:
        store.append_rows(
            [0, 1], csr_matrix(np.array([[1, 1, 1, 0], [1, 0, 0, 0]], np.float32))
        )
        with pytest.raises(MemoryBudgetError) as widest:
            store.finalize()
        assert widest.value.item == "one row of the sparse row conversion"
        assert widest.value.needed_bytes == limit
        assert widest.value.budget_bytes == limit - 1
    finally:
        store.close()


class _TooMany(ValueError):
    pass


def test_entries_beyond_max_nnz_raise_the_given_error(tmp_path):
    store = _loose_store(tmp_path, (3, 3), np.float32, max_nnz=4, nnz_error=_TooMany)
    try:
        store.append_entries([0, 1], [0, 1], [1.0, 2.0])
        with pytest.raises(_TooMany, match="exceeds maxNnz=4"):
            store.append_entries([0, 1, 2], [2, 2, 2], [1.0, 2.0, 3.0])
    finally:
        store.close()
    store = _loose_store(tmp_path, (3, 3), np.float32, max_nnz=2, nnz_error=_TooMany)
    try:
        store.append_entries([0, 1], [0, 1], [1.0, 2.0])
        writer = SparseRowStore.open(store.directory, part=1)
        writer.append_entries([2], [2], [3.0])
        writer.finish()
        with pytest.raises(_TooMany, match="exceeds maxNnz=2"):
            store.finalize()
    finally:
        store.close()


def test_closing_the_owner_removes_the_directory_and_a_reader_keeps_it(tmp_path):
    store, matrix = _finalized_rows(tmp_path)
    reader = SparseRowStore.open(store.directory)
    _assert_same_rows(reader.read(0, 30), matrix)
    # Both keep the int64 pointers of 30 rows.
    assert store.resident_bytes == reader.resident_bytes == 31 * 8
    reader.close()
    assert store.directory.exists()
    store.close()
    assert not store.directory.exists()


def test_counts_spill_in_about_one_byte_per_entry(tmp_path):
    rng = np.random.default_rng(8)
    n_rows, n_columns = 400, 20_000
    lengths = rng.poisson(800, n_rows)
    indices = np.concatenate(
        [np.sort(rng.choice(n_columns, size=k, replace=False)) for k in lengths]
    )
    values = rng.geometric(0.6, indices.size).astype(np.float32)
    matrix = csr_matrix(
        (values, indices, np.r_[0, np.cumsum(lengths)]), shape=(n_rows, n_columns)
    )
    store = _whole_store(
        tmp_path, matrix.shape, np.float32, bucket_rows=100, buffer_bytes=64 << 20
    )
    try:
        store.append_rows(rng.permutation(n_rows), matrix)
        store.finalize()
        spilled = sum(
            path.stat().st_size for path in store.directory.glob("rows-*/*.bin")
        )
        # Narrow values and column gaps compress to about a byte an entry;
        # float32 values with int32 indices take eight.
        assert spilled / matrix.nnz < 2
    finally:
        store.close()


def test_conversion_sizes_name_their_parts():
    # One summed float64 entry reads in 3 * 8 + 32 + 2 * 8 + 24 bytes, its
    # piece in 48 and its row's pointers in 2 * 32.
    one_entry = 96 + 48 + 64
    assert (
        SparseRowStore.conversion_bytes(2, entry_bytes=100)
        == 3 * ROW_BYTES + 2 * one_entry
    )
    assert SparseRowStore.conversion_bytes(2, entry_bytes=500) == 3 * ROW_BYTES + 1000
    assert SparseRowStore.conversion_bytes(
        2, entry_bytes=1, row_entries=3, itemsize=4, sum_duplicates=False
    ) == 3 * ROW_BYTES + 2 * (3 * (3 * 4 + 32) + 3 * 48 + 64)
    assert SparseRowStore.chunk_entries(3 * ROW_BYTES + 2000, 2, 100) == 10
    assert SparseRowStore.chunk_entries(0, 2, 100) == 1


def test_from_entries_refuses_chunks_of_the_wrong_shape_and_cleans_up(tmp_path):
    chunks = iter((coo_matrix([[1, 0], [0, 2]]), coo_matrix([[1, 0, 0]])))
    with pytest.raises(ValueError, match="wrong shape"):
        SparseRowStore.from_entries(
            chunks,
            (2, 2),
            np.int64,
            operation=_OPERATION,
            max_bytes=1 << 20,
            entry_bytes=100,
            entries=3,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("entries", [None, 6, 10**9])
def test_from_entries_sizes_buckets_from_the_entries_expected(tmp_path, entries):
    matrix = coo_matrix(np.arange(1, 13, dtype=np.int64).reshape(6, 2))
    store = SparseRowStore.from_entries(
        iter((matrix,)),
        matrix.shape,
        np.int64,
        operation=_OPERATION,
        max_bytes=1 << 20,
        entry_bytes=100,
        entries=entries,
        temp_dir=tmp_path,
    )
    try:
        # A billion entries over six rows asks for buckets of one row.
        assert (store.bucket_starts.size - 1 == 6) == (entries == 10**9)
        np.testing.assert_array_equal(store.read(0, 6).toarray(), matrix.toarray())
    finally:
        store.close()


def test_rewriting_a_split_bucket_keeps_the_order_of_duplicates(tmp_path):
    rows = np.array([0, 1, 0, 1, 0, 1] * 30)
    columns = np.zeros(rows.size, dtype=np.int64)
    values = np.arange(rows.size, dtype=np.float64)
    limit = _sparse._bucket_bytes(90, 90, 1, 8, False) + 100
    store = _loose_store(
        tmp_path,
        (2, 1),
        np.float64,
        bucket_rows=2,
        bucket_bytes=limit,
        sum_duplicates=False,
        buffer_bytes=400,
    )
    try:
        for start in range(0, rows.size, 7):
            store.append_entries(
                rows[start : start + 7],
                columns[start : start + 7],
                values[start : start + 7],
            )
        store.finalize()
        np.testing.assert_array_equal(store.bucket_starts, [0, 1, 2])
        found = store.read(0, 2)
        np.testing.assert_array_equal(
            found.data, np.r_[values[rows == 0], values[rows == 1]]
        )
    finally:
        store.close()


def test_stable_order_matches_numpy_for_every_key_width():
    rng = np.random.default_rng(0)
    edges = np.array([0, 2**16 - 1, 2**16, 2**32 - 1, 2**32, 2**40 - 1])
    for top in (2**8, 2**16, 2**20, 2**40):
        keys = rng.integers(0, top, 5000)
        # Keys at the edges of each 16-bit digit, each twice.
        present = edges[edges < top]
        keys[: 2 * present.size] = np.repeat(present, 2)
        np.testing.assert_array_equal(
            _sparse._stable_order(keys), np.argsort(keys, kind="stable")
        )


def test_empty_appends_change_nothing(tmp_path):
    whole = _whole_store(tmp_path, (2, 3), np.float32)
    loose = _loose_store(tmp_path, (2, 3), np.float32)
    try:
        whole.append_rows(
            np.zeros(0, dtype=np.int64), csr_matrix((0, 3), dtype=np.float32)
        )
        loose.append_entries([], [], [])
        whole.append_rows([1, 0], csr_matrix(np.eye(2, 3, dtype=np.float32)))
        loose.finalize()
        whole.finalize()
        np.testing.assert_array_equal(whole.read(0, 2).toarray(), np.eye(2, 3)[::-1])
        assert loose.read(0, 2).nnz == 0
        assert not list(loose.directory.glob("rows-*"))
    finally:
        whole.close()
        loose.close()


def test_closing_batches_early_stops_the_read_ahead(tmp_path):
    import threading

    store, matrix = _finalized_rows(tmp_path)
    try:
        threads = threading.active_count()
        batches = store.iter_rows(0, 30, 6, read_ahead=True)
        _assert_same_rows(next(batches), matrix[:6])
        assert threading.active_count() == threads + 1
        batches.close()
        assert threading.active_count() == threads
        # Bucket 1, read ahead and dropped, reads again.
        _assert_same_rows(store.read(6, 12), matrix[6:12])
    finally:
        store.close()


def test_buckets_split_to_the_split_size_and_one_wider_row_reads_alone(tmp_path):
    shape = (12, 40)
    # Row 5 holds 30 entries; every other row holds 2.
    rows = np.r_[np.repeat(np.arange(12), 2), np.full(28, 5)]
    columns = np.r_[np.tile([0, 1], 12), np.arange(2, 30)]
    values = np.arange(1, rows.size + 1, dtype=np.int64)
    split = _sparse._bucket_bytes(8, 8, 4, 8, True)
    limit = _sparse._bucket_bytes(30, 30, 1, 8, True)
    with pytest.raises(ValueError, match="split_bytes must lie between 0 and"):
        _loose_store(
            tmp_path, shape, np.int64, bucket_bytes=limit, split_bytes=limit + 1
        )
    store = _loose_store(
        tmp_path,
        shape,
        np.int64,
        bucket_rows=12,
        bucket_bytes=limit,
        split_bytes=split,
        buffer_bytes=1 << 16,
    )
    try:
        store.append_entries(rows, columns, values)
        store.finalize()
        # Bands of four rows of two entries each, and row 5 alone.
        starts = store.bucket_starts.tolist()
        assert [5, 6] == [start for start in starts if start in (5, 6)]
        assert max(np.diff(starts)) <= 4
        expected = np.zeros(shape, dtype=np.int64)
        expected[rows, columns] = values
        np.testing.assert_array_equal(store.read(0, 12).toarray(), expected)
        # The widest read is row 5's band: one piece of 30 entries.
        assert store.read_bytes() - store.bucket_starts.nbytes == (
            _sparse._bucket_bytes(30, 1, 1, 8, True)
        )
    finally:
        store.close()


def test_an_empty_bucket_beyond_the_split_size_splits_without_files(tmp_path):
    store = _loose_store(
        tmp_path,
        (8, 3),
        np.float32,
        bucket_rows=4,
        split_bytes=_sparse._bucket_bytes(0, 0, 2, 4, True),
    )
    try:
        store.append_entries([6], [1], [2.0])
        store.finalize()
        # Rows 0 to 3 had no entry and no folder, and split all the same.
        assert store.bucket_starts.tolist() == [0, 2, 4, 6, 7, 8]
        expected = np.zeros((8, 3), dtype=np.float32)
        expected[6, 1] = 2.0
        np.testing.assert_array_equal(store.read(0, 8).toarray(), expected)
    finally:
        store.close()


def test_a_measured_rate_estimates_the_disk_that_whole_rows_take(tmp_path):
    rng = np.random.default_rng(12)
    counts = csr_matrix(rng.poisson(0.3, (2000, 3000)).astype(np.float32))
    rate = encoded_entry_bytes(counts[:400])
    # Small counts narrow to one byte and their columns to gaps, compressed.
    assert 0.5 < rate < 2.0
    store = _whole_store(
        tmp_path,
        counts.shape,
        np.float32,
        bucket_rows=250,
        buffer_bytes=1 << 20,
        entries=counts.nnz,
        entry_bytes=rate,
    )
    try:
        store.append_rows(np.arange(2000)[::-1], counts)
        store.finalize()
        written = _store_files(store.directory)
        # The estimate asks for a quarter more than the sample's rate gives,
        # a block of 64 + 4 * 16 bytes for each of the 8 buckets, and the
        # store's settings and records. That covers what the store wrote,
        # and is far less than the bound before compression, 4 + 2 bytes for
        # each entry.
        estimate = store._spill_bound(counts.nnz, rate)
        assert estimate == (
            int(np.ceil(counts.nnz * rate * 1.25))
            + 8 * 128
            + written["settings"]
            + written["records"]
        )
        assert sum(written.values()) <= estimate < counts.nnz * 6 / 2
    finally:
        store.close()
    with pytest.raises(TypeError, match="encoded_entry_bytes takes a CSR matrix"):
        encoded_entry_bytes(counts.tocoo())
    assert encoded_entry_bytes(csr_matrix((3, 4), dtype=np.float32)) > 0


def test_store_sizes_for_planners_count_rows_and_buckets():
    assert SparseRowStore.writer_bytes(10, 4) == 10 * 9 + (3 + 1) * 25
    assert SparseRowStore.writer_bytes(0, 4) == 25
    # 100 entries of float32 in 10 rows read in 44 bytes each, and each row
    # 80 more; 5,200 bytes hold one row of 520.
    assert SparseRowStore.whole_row_bucket_rows(10, 100, 5_200, 4) == 10
    assert SparseRowStore.whole_row_bucket_rows(10, 100, 1_040, 4) == 2
    assert SparseRowStore.whole_row_bucket_rows(10, 100, 10, 4) == 1
    assert SparseRowStore.whole_row_bucket_rows(0, 0, 10, 4) == 1


def test_the_operation_a_store_serves_names_its_errors_in_every_process(tmp_path):
    store = SparseRowStore(
        (4, 3),
        np.float32,
        operation="shuffled test import",
        bucket_rows=2,
        whole_rows=True,
        buffer_bytes=0,
        bucket_bytes=_BIG,
        max_nnz=2,
        temp_dir=tmp_path,
    )
    try:
        assert store.operation == "shuffled test import"
        # A writer opened by the store's directory names it too.
        writer = SparseRowStore.open(store.directory, part=1)
        assert writer.operation == "shuffled test import"
        with pytest.raises(ValueError, match="^Shuffled test import exceeds maxNnz=2$"):
            writer.append_rows([0], csr_matrix(np.ones((1, 3), np.float32)))
    finally:
        store.close()
    with pytest.raises(MemoryBudgetError) as refused:
        SparseRowStore(
            (2, 2),
            np.float32,
            operation="shuffled test import",
            bucket_rows=1,
            whole_rows=True,
            buffer_bytes=0,
            bucket_bytes=10,
            temp_dir=tmp_path,
        )
    assert str(refused.value).startswith("Shuffled test import exceeds the memory")
    assert refused.value.item == "reading one entry of the shuffled test import"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["", "   ", None, 3])
def test_a_store_without_an_operation_is_refused(tmp_path, operation):
    with pytest.raises(ValueError, match="operation names what a sparse row store"):
        SparseRowStore(
            (2, 2),
            np.float32,
            operation=operation,
            bucket_rows=1,
            whole_rows=True,
            buffer_bytes=0,
            bucket_bytes=_BIG,
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Ragged rows


def _ragged_rows(rng, n_rows, widest=60) -> tuple[np.ndarray, np.ndarray]:
    """Rows of 0 to ``widest`` printable bytes: their bytes and the offset of each."""
    lengths = rng.integers(0, widest + 1, n_rows)
    values = rng.integers(32, 127, int(lengths.sum())).astype(np.uint8)
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return values, offsets


def _row_bytes(values, offsets) -> list[bytes]:
    return [bytes(values[offsets[i] : offsets[i + 1]]) for i in range(offsets.size - 1)]


def _ragged_store(tmp_path, n_rows, widest=60, **options) -> SparseRowStore:
    settings = {"bucket_rows": 7, "buffer_bytes": 4096, "bucket_bytes": _BIG}
    settings.update(options)
    return SparseRowStore(
        (n_rows, widest),
        np.uint8,
        operation=_OPERATION,
        whole_rows=True,
        ragged=True,
        temp_dir=tmp_path,
        **settings,
    )


def _append_in_blocks(store, destinations, values, offsets, block_rows=41) -> None:
    for first in range(0, destinations.size, block_rows):
        last = min(first + block_rows, destinations.size)
        store.append_ragged_rows(
            destinations[first:last],
            values[offsets[first] : offsets[last]],
            offsets[first : last + 1] - offsets[first],
        )


@pytest.mark.parametrize("bucket_rows", [1, 7, 500])
@pytest.mark.parametrize("read_ahead", [False, True])
def test_ragged_rows_return_in_destination_order(tmp_path, bucket_rows, read_ahead):
    rng = np.random.default_rng(bucket_rows)
    values, offsets = _ragged_rows(rng, 300)
    destinations = rng.permutation(300)
    store = _ragged_store(tmp_path, 300, bucket_rows=bucket_rows)
    try:
        _append_in_blocks(store, destinations, values, offsets)
        store.finalize()
        rows = _row_bytes(values, offsets)
        # Row i of the appended rows is destination row destinations[i].
        expected = [rows[i] for i in np.argsort(destinations)]
        found = []
        for batch_values, batch_offsets in store.iter_ragged_rows(
            0, 300, 23, read_ahead=read_ahead
        ):
            assert batch_values.dtype == np.uint8 and batch_offsets[0] == 0
            assert batch_offsets[-1] == batch_values.size
            found += _row_bytes(batch_values, batch_offsets)
        assert found == expected
        window = [
            row
            for batch in store.iter_ragged_rows(37, 211, 50)
            for row in _row_bytes(*batch)
        ]
        assert window == expected[37:211]
    finally:
        store.close()


def test_ragged_buckets_split_and_go_as_their_rows_are_read(tmp_path):
    rng = np.random.default_rng(3)
    values, offsets = _ragged_rows(rng, 400)
    destinations = rng.permutation(400)
    # One bucket of every row, split at finalize into bands that each read in
    # a sixteenth of what the rows take.
    split = SparseRowStore.ragged_row_read_bytes(int(offsets[-1]) // 16)
    store = _ragged_store(tmp_path, 400, bucket_rows=400, split_bytes=split)
    try:
        _append_in_blocks(store, destinations, values, offsets)
        store.finalize()
        assert store.bucket_starts.size - 1 >= 16
        assert store.read_bytes() <= store.bucket_starts.nbytes + split
        expected = [_row_bytes(values, offsets)[i] for i in np.argsort(destinations)]
        found = []
        held = []
        for batch in store.iter_ragged_rows(0, 400, 25, delete=True):
            found += _row_bytes(*batch)
            held.append(len(list(store.directory.glob("rows-*[0-9]"))))
        assert found == expected
        # Each band's files go once its rows are taken.
        assert held == sorted(held, reverse=True) and held[0] > held[-1]
        assert list(store.directory.glob("rows-*[0-9]")) == []
        with pytest.raises(RuntimeError, match="were deleted once a read had taken"):
            next(store.iter_ragged_rows(0, 1, 1))
    finally:
        store.close()


def test_ragged_rows_are_stored_as_they_are_without_column_indices(tmp_path):
    # Rows of one repeated text, which zstd compresses to a few bytes each.
    rows = [b"TRAV1-2\tCASSLGTDTQYF\tIGHM\n"] * 500
    values = np.frombuffer(b"".join(rows), dtype=np.uint8)
    offsets = np.arange(0, values.size + 1, len(rows[0]), dtype=np.int64)
    store = _ragged_store(tmp_path, 500, bucket_rows=500)
    try:
        store.append_ragged_rows(np.arange(500), values, offsets)
        store.finalize()
        data = _bucket_file(store).read_bytes()
        fields = _sparse._HEADER.unpack_from(data)
        row_bytes, length_bytes, index_bytes, value_bytes = fields[8:12]
        # The block's column indices take no bytes, and its values are a
        # Blosc frame of the bytes as they are: a 16-byte header, then the
        # bytes.
        assert fields[4] == _sparse._RAGGED and index_bytes == 0
        assert value_bytes == values.size + 16
        start = _sparse._HEADER.size + row_bytes + length_bytes + index_bytes + 16
        assert data[start : start + values.size] == values.tobytes()
        # An assembled bucket holds the bytes and the row offsets, without an
        # eight-byte index for each byte.
        assert store.read_bytes(read_ahead=True) - store.read_bytes() == (
            values.size + 501 * 8
        )
    finally:
        store.close()
    # Sparse rows of the same bytes, in one block, still compress with zstd.
    sparse = _whole_store(tmp_path, (500, 26), np.uint8, bucket_rows=500)
    try:
        sparse.append_rows(
            np.arange(500), csr_matrix(values.reshape(500, 26).astype(np.uint8))
        )
        sparse.finalize()
        data = _bucket_file(sparse).read_bytes()
        fields = _sparse._HEADER.unpack_from(data)
        start = _sparse._HEADER.size + sum(fields[8:11])
        frame = data[start : start + fields[11]]
        assert fields[11] < values.size // 5
        assert numcodecs.blosc.cbuffer_complib(frame) == "Zstd"
    finally:
        sparse.close()


def test_a_ragged_read_holds_no_more_than_the_store_reports(tmp_path):
    rng = np.random.default_rng(12)
    values, offsets = _ragged_rows(rng, 3000, widest=400)
    store = _ragged_store(tmp_path, 3000, widest=400, bucket_rows=700)
    try:
        _append_in_blocks(store, rng.permutation(3000), values, offsets, 333)
        store.finalize()
        reported = store.read_bytes()
        tracemalloc.start()
        try:
            for bucket in range(store.bucket_starts.size - 1):
                tracemalloc.reset_peak()
                base = tracemalloc.get_traced_memory()[0]
                store._load_ragged_bucket(bucket)
                assert tracemalloc.get_traced_memory()[1] - base <= reported
        finally:
            tracemalloc.stop()
    finally:
        store.close()


def test_wrong_ragged_rows_are_refused_with_what_was_expected(tmp_path):
    for options in ({"whole_rows": False}, {"dtype": np.int32}):
        settings = {"whole_rows": True, "dtype": np.uint8, **options}
        with pytest.raises(ValueError, match="Ragged rows are bytes that arrive whole"):
            SparseRowStore(
                (3, 5),
                settings.pop("dtype"),
                operation=_OPERATION,
                bucket_rows=2,
                buffer_bytes=0,
                bucket_bytes=_BIG,
                ragged=True,
                temp_dir=tmp_path,
                **settings,
            )
    store = _ragged_store(tmp_path, 3, widest=5)
    sparse = _whole_store(tmp_path, (3, 5), np.uint8)
    values = np.frombuffer(b"abcdef", dtype=np.uint8)
    try:
        with pytest.raises(
            TypeError, match="takes ragged rows; call append_ragged_rows"
        ):
            store.append_rows(np.arange(3), csr_matrix((3, 5), dtype=np.uint8))
        with pytest.raises(TypeError, match="takes whole rows; call append_rows"):
            sparse.append_ragged_rows(np.arange(2), values, np.asarray([0, 3, 6]))
        with pytest.raises(TypeError, match="one-dimensional uint8 bytes"):
            store.append_ragged_rows([0], np.asarray([1, 2], dtype=np.int64), [0, 2])
        for bounds in ([0, 3], [1, 3, 6], [0, 4, 3], [0, 3, 5], [0.0, 3.0, 6.0]):
            with pytest.raises(ValueError, match="integer offsets that rise from 0"):
                store.append_ragged_rows([0, 1], values, np.asarray(bounds))
        with pytest.raises(ValueError, match="holds at most 5 bytes; found one of 6"):
            store.append_ragged_rows([0], values, [0, 6])
        with pytest.raises(IndexError, match=r"must lie in \[0, 3\)"):
            store.append_ragged_rows([3], values[:1], [0, 1])
        store.append_ragged_rows([0, 1], values, [0, 3, 6])
        with pytest.raises(ValueError, match="Destination row 1 was appended more"):
            store.append_ragged_rows([1], values[:2], [0, 2])
        with pytest.raises(ValueError, match="1 destination rows were never appended"):
            store.finalize()
    finally:
        store.close()
        sparse.close()
    store = _ragged_store(tmp_path, 1, widest=5)
    sparse = _whole_store(tmp_path, (1, 5), np.uint8)
    try:
        store.append_ragged_rows([0], values[:2], [0, 2])
        store.finalize()
        sparse.append_rows([0], csr_matrix((1, 5), dtype=np.uint8))
        sparse.finalize()
        with pytest.raises(TypeError, match="holds ragged rows; read them with"):
            next(store.iter_rows(0, 1, 1))
        with pytest.raises(TypeError, match="holds ragged rows; read them with"):
            store.read(0, 1)
        with pytest.raises(TypeError, match="holds sparse rows; read them with"):
            next(sparse.iter_ragged_rows(0, 1, 1))
    finally:
        store.close()
        sparse.close()


@pytest.mark.parametrize("ragged", [False, True])
def test_a_block_of_the_other_form_of_rows_is_refused(tmp_path, ragged):
    if ragged:
        store = _ragged_store(tmp_path, 2, widest=5)
        store.append_ragged_rows([0, 1], np.frombuffer(b"abcde", np.uint8), [0, 2, 5])
    else:
        store = _whole_store(tmp_path, (2, 5), np.float32)
        store.append_rows([0, 1], csr_matrix(np.eye(2, 5, dtype=np.float32)))
    try:
        store.finalize()
        path = _bucket_file(store)
        # The block's index code claims the other form of rows.
        flipped = 0 if ragged else _sparse._RAGGED
        path.write_bytes(_replace_header(path.read_bytes(), index=flipped))
        found, expected = ("sparse", "ragged") if ragged else ("ragged", "sparse")
        with pytest.raises(
            RuntimeError, match=f"holds {found} rows in a store of {expected} rows"
        ):
            if ragged:
                next(store.iter_ragged_rows(0, 2, 2))
            else:
                store.read(0, 2)
    finally:
        store.close()


def test_ragged_rows_spill_their_bytes_and_a_wider_row_reads_alone(tmp_path):
    asked = {}
    for n_rows, entries in ((4, 1000), (4, 2000), (5, 1000)):
        with free_disk(0), pytest.raises(OSError, match="temporary bytes") as full:
            _ragged_store(tmp_path, n_rows, entries=entries)
        asked[n_rows, entries] = asked_bytes(full.value)
    # Without a measured rate, the disk check counts each byte once and no
    # column index. Each row takes 16 bytes for its place and length, and
    # 17 in the records: its count of int64 and mark of uint8 in the part,
    # and its int64 pointer in the finalized record.
    assert asked[4, 2000] - asked[4, 1000] == 1000
    assert asked[5, 1000] - asked[4, 1000] == 16 + 8 + 1 + 8
    with free_disk(asked[4, 1000] - 1), pytest.raises(OSError):
        _ragged_store(tmp_path, 4, entries=1000)
    with free_disk(asked[4, 1000]):
        _ragged_store(tmp_path, 4, entries=1000).close()
    # A row that one read cannot hold is refused at finalize, with the bytes
    # that reading it holds.
    store = _ragged_store(
        tmp_path,
        2,
        bucket_bytes=SparseRowStore.ragged_row_read_bytes(50),
        split_bytes=SparseRowStore.ragged_row_read_bytes(10),
    )
    try:
        # A row of 51 bytes, one more than the limit reads.
        values = np.full(60, ord("A"), dtype=np.uint8)
        store.append_ragged_rows([0, 1], values, [0, 9, 60])
        with pytest.raises(MemoryBudgetError) as refused:
            store.finalize()
        assert refused.value.needed_bytes == SparseRowStore.ragged_row_read_bytes(51)
    finally:
        store.close()
    # Buckets sized for rows of 100 bytes read in about the bytes asked.
    for read in (1, 2_000, 50_000, 10**9):
        rows = SparseRowStore.ragged_bucket_rows(1000, 100, read)
        assert 1 <= rows <= 1000
        if 1 < rows < 1000:
            cost = SparseRowStore.ragged_row_read_bytes(100)
            assert rows * (cost - 32) <= read < (rows + 1) * (cost - 32)
