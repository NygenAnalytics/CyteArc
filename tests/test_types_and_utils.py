import datetime
import decimal
import gc
import weakref

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from zarr.errors import ContainsArrayError, GroupNotFoundError
from zarr.storage import MemoryStore

from cytearc.storage.types import (
    array_metadata_shards,
    as_zarr_array,
    as_zarr_group,
    read_array,
    read_fresh_group,
)
from cytearc.utils import (
    array_digest,
    clean_array,
    configure_output,
    permute_into_chunks,
    rescale_array,
    rolling_window,
    set_verbosity,
    compute_with_progress,
    tqdmbar,
)
from cytearc.utils.arguments import (
    RANDOM_STATE_MAX,
    bool_argument,
    clip_fraction_argument,
    float_argument,
    integer_argument,
    optional_name_argument,
    categorical_label_codes,
    label_array,
    require_labels_match_their_text,
)
from cytearc.utils.digest import mix64, mix64_array
from cytearc.utils.numba import numba_threads
from cytearc.utils.arrays import (
    _rolling_window_kernel,
    assay_feature_ranges,
    canonicalize_sparse,
    checked_sparse_cast,
    cumulative_nnz,
    max_window_nnz,
    read_only_copy,
    sparse_matrix_bytes,
)
from cytearc.utils.errors import raise_collected_errors
from cytearc.utils.progress import iter_progress
from tests.conftest import quiet_cytearc_output


def test_as_zarr_array_accepts_array():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(3,), dtype="f8")
    assert as_zarr_array(arr) is arr


def test_as_zarr_array_rejects_group():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("nested")
    with pytest.raises(TypeError, match="Expected Zarr array"):
        as_zarr_array(group, name="nested")


def test_as_zarr_group_accepts_group():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("nested")
    assert as_zarr_group(group) is group


def test_as_zarr_group_rejects_array():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(2,), dtype="f8")
    with pytest.raises(TypeError, match="Expected Zarr group"):
        as_zarr_group(arr, name="data")


def test_array_metadata_shards_returns_none_without_sharding():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(4,), dtype="f8")
    assert array_metadata_shards(arr) is None


def test_read_fresh_group_reads_the_stored_record(tmp_path):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    handle = root.create_group("child", attributes={"value": 1})
    zarr.open_group(store=root.store, path="child", mode="r+").attrs["value"] = 2
    root.create_array("array", shape=(1,), dtype="i4")

    # An open handle keeps the attributes it read when it was opened.
    assert handle.attrs["value"] == 1
    assert read_fresh_group(root, "child").attrs["value"] == 2
    assert read_fresh_group(handle).attrs["value"] == 2
    # A missing group raises unless the caller expects one.
    with pytest.raises(GroupNotFoundError):
        read_fresh_group(root, "missing")
    assert read_fresh_group(root, "missing", missing_ok=True) is None
    with pytest.raises(ContainsArrayError):
        read_fresh_group(root, "array", missing_ok=True)


def test_read_fresh_group_opens_as_its_parent_unless_told_otherwise(tmp_path):
    path = str(tmp_path / "store.zarr")
    zarr.open_group(path, mode="w").create_group("child")
    writable = zarr.open_group(path, mode="r+")
    read_only = zarr.open_group(path, mode="r")

    assert read_fresh_group(writable, "child").read_only is False
    assert read_fresh_group(read_only, "child").read_only is True
    assert read_fresh_group(writable, "child", mode="r").read_only is True


@pytest.mark.parametrize("fill_val", [0.0, 1.0, -1.0])
def test_clean_array_replaces_nan_inf_and_zero(fill_val):
    raw = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
    cleaned = clean_array(raw, fill_val=fill_val)
    np.testing.assert_array_equal(
        cleaned, [1.0, fill_val, fill_val, fill_val, fill_val]
    )
    np.testing.assert_array_equal(raw, [1.0, np.nan, np.inf, -np.inf, 0.0])


def test_rescale_array_trims_extreme_values():
    from scipy.stats import norm

    values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    # A normal at the median (0) with the values' spread (sqrt(40.4)) puts its
    # 10th and 90th percentiles at -/+ 1.28155 * 6.35610 = -/+ 8.14572.
    bound = norm.ppf(0.9) * np.sqrt(40.4)
    assert bound == pytest.approx(8.14572, rel=1e-5)
    trimmed = rescale_array(values, frac=0.9)
    assert trimmed is values
    np.testing.assert_array_equal(trimmed, [-bound, -1.0, 0.0, 1.0, bound])


def test_rescale_array_requires_an_upper_quantile_above_one_half():
    values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    # frac=1 puts the bounds at -/+ infinity, so nothing is trimmed.
    np.testing.assert_array_equal(rescale_array(values.copy(), frac=1), values)
    for frac in (0.1, 0.5, 1.5):
        with pytest.raises(ValueError, match="frac must be greater than 0.5"):
            rescale_array(values.copy(), frac=frac)
    for frac in (True, "0.9", None):
        with pytest.raises(TypeError, match="frac must be a real number"):
            rescale_array(values.copy(), frac=frac)
    with pytest.raises(ValueError, match="frac must be finite"):
        rescale_array(values.copy(), frac=np.nan)


def test_set_verbosity_rejects_invalid_level():
    with pytest.raises(ValueError, match="Please provide a value for level"):
        set_verbosity("NOT_A_REAL_LEVEL")


@pytest.fixture
def quiet_output_afterwards():
    """Leave the suite's quiet output once a test that changes it ends."""
    yield
    quiet_cytearc_output()


def test_set_verbosity_accepts_valid_level(quiet_output_afterwards):
    from cytearc.utils.logging import _config

    try:
        set_verbosity("ERROR")
        assert (_config.level, _config.filepath) == ("ERROR", None)
        with pytest.raises(
            ValueError, match="^Please provide a value for level recognized by Loguru$"
        ):
            set_verbosity(None)
        assert _config.level == "ERROR"
    finally:
        set_verbosity("INFO")
    assert (_config.level, _config.filepath) == ("INFO", None)


def test_tqdmbar_uses_explicit_progress_independently_of_severity(
    monkeypatch, quiet_output_afterwards
):
    import tqdm.auto as tqdm_auto

    captured: list[bool] = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            captured.append(bool(kwargs.get("disable")))

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(tqdm_auto, "tqdm", FakeTqdm)
    try:
        configure_output(level="ERROR", progress=True)
        list(tqdmbar(range(1), desc="test"))
        set_verbosity("WARNING")
        list(tqdmbar(range(1), desc="test"))
        configure_output(level="DEBUG", progress=False)
        list(tqdmbar(range(1), desc="test"))
        list(tqdmbar(range(1), desc="test", disable=False))
        configure_output(progress=True)
        list(tqdmbar(range(1), desc="test", disable=True))
    finally:
        configure_output(level="INFO", progress=False, timestamps=False)

    assert captured == [False, False, True, True, True]


def test_controlled_compute_bounds_deferred_threads_and_passes_arrays_through():
    from cytearc.utils.compute import controlled_compute

    calls: list[int] = []

    class Deferred:
        def compute(self, nthreads):
            calls.append(nthreads)
            return [1, 2]

    np.testing.assert_array_equal(controlled_compute(Deferred(), 3), [1, 2])
    assert calls == [3]
    values = np.arange(3)
    assert controlled_compute(values, 2) is values
    np.testing.assert_array_equal(controlled_compute([4, 5], 2), [4, 5])


def test_sort_categories_orders_numbers_then_text_then_missing_values():
    import pandas as pd

    from cytearc.utils.arrays import sort_categories

    values = ["b10", np.float32("nan"), "b2", pd.NA, 3, None, "10", [1, 2], pd.NaT, 2.5]
    ordered = sort_categories(values)
    # Numbers by value, text in natural order (a list label sorts as its
    # text), and every missing marker last in input order.
    expected = [9, 4, 6, 7, 2, 0, 1, 3, 5, 8]
    assert [id(value) for value in ordered] == [id(values[index]) for index in expected]


def test_compute_with_progress_uses_explicit_progress_setting():
    calls: list[tuple[int, str | None]] = []

    class Deferred:
        def compute(self, nthreads, msg):
            calls.append((nthreads, msg))
            return np.array([1])

    try:
        configure_output(progress=False)
        compute_with_progress(Deferred(), "Computing", 2)
        configure_output(progress=True)
        compute_with_progress(Deferred(), "Computing", 3)
    finally:
        configure_output(progress=False)

    assert calls == [(2, None), (3, "Computing")]
    np.testing.assert_array_equal(
        compute_with_progress(np.array([2, 3])),
        np.array([2, 3]),
    )


def test_progress_iterator_releases_consumed_values():
    references: list[weakref.ReferenceType[object]] = []

    class Chunk:
        pass

    def source():
        for _ in range(2):
            chunk = Chunk()
            references.append(weakref.ref(chunk))
            yield chunk

    stream = iter_progress(source(), total=2, disable=True)
    first = next(stream)
    second = next(stream)
    del first
    gc.collect()

    assert references[0]() is None
    assert references[1]() is second
    stream.close()


def test_progress_iterator_closes_source_on_early_exit():
    closed = False
    produced: list[object] = []

    def source():
        nonlocal closed
        try:
            for _ in range(2):
                produced.append(object())
                yield produced[-1]
        finally:
            closed = True

    stream = iter_progress(source(), total=2, disable=True)
    assert next(stream) is produced[0]
    assert not closed
    stream.close()

    # Closing the stream closes the source before it produces its second item.
    assert closed
    assert len(produced) == 1


def test_raise_collected_errors_raises_nothing_without_errors():
    errors: list[BaseException] = []
    raise_collected_errors(errors, "cleanup failed")
    raise_collected_errors(errors, "cleanup failed", ended_by=ValueError("ended"))
    assert errors == []


def test_raise_collected_errors_raises_one_error_as_itself():
    error = OSError("close failed")
    errors: list[BaseException] = [error]
    with pytest.raises(OSError) as caught:
        raise_collected_errors(errors, "cleanup failed")
    assert caught.value is error
    assert errors == []


def test_raise_collected_errors_groups_errors_behind_the_one_that_ended_the_work():
    ended = ValueError("read failed")
    first, second = OSError("close failed"), RuntimeError("shutdown failed")
    errors: list[BaseException] = [first, second]
    with pytest.raises(BaseExceptionGroup, match="cleanup failed") as caught:
        raise_collected_errors(errors, "cleanup failed", ended_by=ended)
    assert list(caught.value.exceptions) == [ended, first, second]
    assert errors == []


def test_raise_collected_errors_leaves_out_the_exit_of_a_closed_generator():
    error = OSError("close failed")
    with pytest.raises(OSError) as caught:
        raise_collected_errors([error], "cleanup failed", ended_by=GeneratorExit())
    assert caught.value is error


def test_a_frame_that_raises_collected_errors_is_freed_without_a_collection():
    held: list[weakref.ref] = []

    def clean_up() -> None:
        data = np.ones(1_000)
        held.append(weakref.ref(data))
        errors: list[BaseException] = []
        try:
            raise OSError("close failed")
        except OSError as error:
            errors.append(error)
        # The error's traceback refers to this frame, which holds ``errors``.
        raise_collected_errors(errors, "cleanup failed")

    enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            clean_up()
        except OSError:
            pass
        assert held[0]() is None
    finally:
        if enabled:
            gc.enable()


def test_rolling_window_smoothes_along_rows():
    data = np.arange(10, dtype=float).reshape(5, 2)
    smoothed = rolling_window(data, w=3)
    expected = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [7.0, 8.0],
        ]
    )
    np.testing.assert_array_equal(smoothed, expected)


def test_rolling_window_even_and_oversized_windows():
    data = np.arange(5, dtype=float).reshape(-1, 1)

    np.testing.assert_array_equal(rolling_window(np.array([[7.0]]), w=1), [[7.0]])
    np.testing.assert_array_equal(
        rolling_window(data, w=2), [[0.5], [1.5], [2.5], [3.5], [4.0]]
    )
    np.testing.assert_array_equal(
        rolling_window(data, w=20), [[1.0], [1.5], [2.0], [2.5], [3.0]]
    )
    for window_size in (0, -1):
        with pytest.raises(ValueError, match="greater than zero"):
            rolling_window(data, w=window_size)


def test_python_rolling_window_kernel_matches_public_compiled_path():
    data = np.arange(12, dtype=float).reshape(6, 2)

    np.testing.assert_allclose(
        _rolling_window_kernel(data, 4),
        rolling_window(data, 4),
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        _rolling_window_kernel(np.arange(3), 2)
    with pytest.raises(ValueError, match="greater than zero"):
        _rolling_window_kernel(data, 0)
    with pytest.raises(ValueError, match="at least one row"):
        _rolling_window_kernel(np.empty((0, 2)), 1)


def test_array_digest_is_deterministic_and_shape_sensitive():
    values = np.arange(6, dtype=np.int64)

    assert array_digest(values) == array_digest(values.copy())
    assert array_digest(values) != array_digest(values.reshape(2, 3))


def test_permute_into_chunks_preserves_all_indices():
    chunks = permute_into_chunks(10, 3, seed=7)
    # Each run of three indices, and the short remainder, is shuffled within
    # itself by one generator seeded once.
    rng = np.random.default_rng(7)
    expected = [
        rng.permutation(np.arange(start, min(start + 3, 10))) for start in (0, 3, 6, 9)
    ]
    assert [chunk.tolist() for chunk in chunks] == [
        chunk.tolist() for chunk in expected
    ]
    merged = np.concatenate(chunks)
    np.testing.assert_array_equal(np.sort(merged), np.arange(10))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            np.array([1 + 0j]),
            "Complex sparse values cannot use an integer destination",
        ),
        (
            np.array([1.5]),
            "cannot be represented by the destination dtype",
        ),
        (
            np.array([np.nan]),
            "cannot be represented by the destination dtype",
        ),
    ],
)
def test_checked_sparse_cast_rejects_lossy_integer_conversions(values, message):
    with pytest.raises(OverflowError, match=message):
        checked_sparse_cast(values, np.int32)


def test_canonicalize_sparse_handles_empty_and_already_canonical_inputs():
    empty = coo_matrix(
        (
            np.array([], dtype=np.int64),
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        ),
        shape=(2, 2),
    )
    empty.has_canonical_format = False

    canonical_empty = canonicalize_sparse(empty)

    assert canonical_empty.shape == (2, 2)
    assert canonical_empty.nnz == 0
    assert canonical_empty.has_canonical_format

    canonical = coo_matrix(
        (
            np.array([1.0, 2.0]),
            (
                np.array([0, 1]),
                np.array([0, 1]),
            ),
        ),
        shape=(2, 2),
    )
    canonical.sum_duplicates()

    returned = canonicalize_sparse(canonical, dtype=np.int16)

    assert returned is canonical
    assert returned.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(returned.data, [1, 2])


def test_csr_to_coo_marks_only_rows_sorted_without_repeats():
    from scipy.sparse import csc_matrix, csr_matrix

    from cytearc.utils.arrays import csr_to_coo

    pointers = np.array([0, 2, 3])
    # Row 0 holds (0, 1) = 4 and (0, 3) = 5, and row 1 holds (1, 0) = 6.
    stored = {
        "sorted": (np.array([1, 3, 0]), np.array([4.0, 5.0, 6.0]), True),
        "unsorted": (np.array([3, 1, 0]), np.array([5.0, 4.0, 6.0]), False),
        "repeated": (np.array([1, 1, 0]), np.array([4.0, 5.0, 6.0]), False),
    }
    for name, (indices, data, canonical) in stored.items():
        matrix = csr_matrix((data, indices, pointers), shape=(2, 4))
        coo = csr_to_coo(matrix)
        assert coo.has_canonical_format is canonical, name
        np.testing.assert_array_equal(coo.row, [0, 0, 1])
        np.testing.assert_array_equal(coo.col, indices)
        np.testing.assert_array_equal(coo.data, data)
        summed = canonicalize_sparse(coo)
        if canonical:
            # A marked COO is neither sorted nor summed again.
            assert summed is coo
        expected = {(0, 1): 4.0, (0, 3): 5.0, (1, 0): 6.0}
        if name == "repeated":
            expected = {(0, 1): 9.0, (1, 0): 6.0}
        assert (
            dict(
                zip(
                    zip(summed.row.tolist(), summed.col.tolist(), strict=True),
                    summed.data.tolist(),
                    strict=True,
                )
            )
            == expected
        )
        assert list(zip(summed.row.tolist(), summed.col.tolist(), strict=True)) == (
            sorted(expected)
        )

    for other in (csc_matrix(np.eye(2)), coo_matrix(np.eye(2)), np.eye(2)):
        with pytest.raises(TypeError, match="takes a CSR matrix"):
            csr_to_coo(other)


def _dense_rows_cases() -> list[tuple[str, np.ndarray]]:
    """Dense rows whose CSR form SciPy's conversion gives independently."""
    rng = np.random.default_rng(3)
    values = rng.integers(-4, 5, size=(37, 23))
    values[rng.random(values.shape) < 0.7] = 0
    # A row without values in the middle, and a zero at the very end, which
    # the spare slot of the pass takes.
    values[5] = 0
    values[-1, -1] = 0
    floats = values.astype(np.float32)
    floats[0, 0], floats[1, 1], floats[2, 2], floats[3, 3] = np.nan, -0.0, np.inf, 0.5
    return [
        ("uint8", np.abs(values).astype(np.uint8)),
        ("uint16", np.abs(values).astype(np.uint16)),
        ("int32", values.astype(np.int32)),
        ("int64", values.astype(np.int64)),
        ("float32", floats),
        ("float64", floats.astype(np.float64)),
        ("bool", values != 0),
        ("fortran", np.asfortranarray(values.astype(np.float32))),
        ("column slice", values.astype(np.uint16)[:, 3:17]),
        ("all zero", np.zeros((4, 6), dtype=np.float32)),
        ("all nonzero", np.ones((3, 5), dtype=np.uint16)),
        ("no rows", np.zeros((0, 5), dtype=np.float32)),
        ("no columns", np.zeros((4, 0), dtype=np.float32)),
    ]


@pytest.mark.parametrize(
    "values",
    [values for _, values in _dense_rows_cases()],
    ids=[name for name, _ in _dense_rows_cases()],
)
def test_csr_from_dense_stores_what_scipy_stores(values):
    from cytearc.utils.arrays import csr_from_dense

    converted = csr_from_dense(values)
    expected = csr_matrix(values)
    assert converted.format == "csr"
    assert converted.shape == values.shape
    assert converted.dtype == values.dtype
    assert converted.has_canonical_format
    np.testing.assert_array_equal(converted.indptr, expected.indptr)
    np.testing.assert_array_equal(converted.indices, expected.indices)
    # NaN is stored and a negative zero left out, as SciPy does.
    np.testing.assert_array_equal(converted.data, expected.data)
    assert converted.indices.dtype == converted.indptr.dtype == np.int32
    np.testing.assert_array_equal(converted.toarray(), np.where(values == 0, 0, values))


def test_csr_from_dense_holds_little_beside_its_result():
    import tracemalloc

    from cytearc.utils.arrays import csr_from_dense, sparse_matrix_bytes

    rng = np.random.default_rng(4)
    values = np.zeros((2000, 500), dtype=np.float32)
    values[rng.random(values.shape) < 0.1] = 1.5
    # The first conversion compiles the pass.
    csr_from_dense(values)
    peaks = {}
    for name, convert in (("compiled", csr_from_dense), ("scipy", csr_matrix)):
        tracemalloc.start()
        try:
            converted = convert(values)
            peaks[name] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    stored = sparse_matrix_bytes(converted)
    assert converted.nnz > 90_000
    # Beside its result, the conversion holds one spare value and index and
    # a few objects, where SciPy's holds more than 16 bytes a value.
    assert peaks["compiled"] <= stored + 4 + 4 + 16_384
    assert peaks["scipy"] > stored + 16 * converted.nnz


@pytest.mark.parametrize("dtype", ["uint16", "int32", "float32"])
def test_fill_dense_from_csr_writes_the_rows_that_numpy_slices(dtype):
    from cytearc.utils.arrays import fill_dense_from_csr

    rng = np.random.default_rng(6)
    values = rng.integers(-3, 4, size=(30, 9)).astype(dtype)
    values[rng.random(values.shape) < 0.6] = 0
    matrix = csr_matrix(values)
    # Rows 7 to 18 go to columns 4 to 12 of rows that hold -1 elsewhere,
    # which stay as they are, as do the rows' zeros.
    out = np.full((11, 15), -1 if dtype != "uint16" else 99, dtype=dtype)
    expected = out.copy()
    expected[:, 4:13] = np.where(values[7:18] == 0, expected[:, 4:13], values[7:18])
    fill_dense_from_csr(matrix, 7, out, 4)
    np.testing.assert_array_equal(out, expected)
    # Every row of the matrix, in the first columns of zeroed rows.
    whole = np.zeros((30, 9), dtype=dtype)
    fill_dense_from_csr(matrix, 0, whole)
    np.testing.assert_array_equal(whole, values)
    # No rows, at the end of the matrix.
    fill_dense_from_csr(matrix, 30, np.zeros((0, 9), dtype=dtype))


@pytest.mark.parametrize(
    ("start", "rows", "columns", "column"),
    [(-1, 2, 9, 0), (29, 2, 9, 0), (0, 2, 8, 0), (0, 2, 9, 1), (0, 2, 9, -1)],
)
def test_fill_dense_from_csr_refuses_rows_or_columns_outside(
    start, rows, columns, column
):
    from cytearc.utils.arrays import fill_dense_from_csr

    matrix = csr_matrix(np.eye(30, 9, dtype=np.float32))
    with pytest.raises(ValueError, match="do not fit dense rows of shape"):
        fill_dense_from_csr(matrix, start, np.zeros((rows, columns)), column)


def test_fill_dense_from_csr_kernel_runs_as_python_too():
    from cytearc.utils import arrays

    matrix = csr_matrix(np.asarray([[0, 3, 0], [0, 0, 0], [7, 0, 2]], dtype=np.int16))
    out = np.full((2, 5), -1, dtype=np.int16)
    arrays._fill_dense_kernel(matrix.indptr, matrix.indices, matrix.data, 1, out, 2)
    assert out.tolist() == [[-1, -1, -1, -1, -1], [-1, -1, 7, -1, 2]]


def test_csr_from_dense_kernel_runs_as_python_too():
    from cytearc.utils import arrays

    values = np.asarray([[0, 3, 0], [0, 0, 0], [7, 0, 2]], dtype=np.int16)
    indptr = np.zeros(4, dtype=np.int32)
    indices = np.full(4, -1, dtype=np.int32)
    data = np.full(4, -1, dtype=np.int16)
    arrays._fill_csr_kernel(values, indptr, indices, data)
    assert indptr.tolist() == [0, 1, 1, 3]
    assert indices[:3].tolist() == [1, 0, 2]
    assert data[:3].tolist() == [3, 7, 2]


def test_canonicalize_sparse_detects_int64_duplicate_overflow():
    maximum = np.iinfo(np.int64).max
    duplicated = coo_matrix(
        (
            np.array([maximum, 1], dtype=np.int64),
            (
                np.array([0, 0]),
                np.array([0, 0]),
            ),
        ),
        shape=(1, 1),
    )

    with pytest.raises(OverflowError, match="Duplicate sparse values exceed"):
        canonicalize_sparse(duplicated)


def test_array_digest_rejects_object_values():
    with pytest.raises(TypeError, match="object arrays"):
        array_digest(np.array([object()], dtype=object))


def test_integer_argument_accepts_numpy_integers_and_checks_bounds():
    assert integer_argument(np.int64(3), "count", minimum=1) == 3
    assert type(integer_argument(np.uint8(3), "count")) is int
    for value in (True, np.bool_(True), 2.0, "2", None):
        with pytest.raises(TypeError, match="count must be an integer"):
            integer_argument(value, "count", minimum=1)
    with pytest.raises(ValueError, match="count must be at least 1"):
        integer_argument(0, "count", minimum=1)
    with pytest.raises(ValueError, match="count must be at most 4"):
        integer_argument(5, "count", minimum=1, maximum=4)


def test_random_state_max_is_the_largest_seed_of_numpy_and_scikit_learn():
    values = np.arange(8, dtype=np.float64).reshape(4, 2)

    np.random.RandomState(RANDOM_STATE_MAX)
    for kmeans in (KMeans, MiniBatchKMeans):
        kmeans(n_clusters=2, n_init=1, random_state=RANDOM_STATE_MAX).fit(values)
    with pytest.raises(ValueError, match="Seed must be between 0 and 2\\*\\*32 - 1"):
        np.random.RandomState(RANDOM_STATE_MAX + 1)
    for kmeans in (KMeans, MiniBatchKMeans):
        with pytest.raises(ValueError, match="'random_state' parameter"):
            kmeans(n_clusters=2, n_init=1, random_state=RANDOM_STATE_MAX + 1).fit(
                values
            )


def test_float_argument_shares_one_value_across_numeric_spellings():
    for value in (1, 1.0, np.int64(1), np.float32(1.0)):
        resolved = float_argument(value, "ratio")
        assert type(resolved) is float
        assert resolved == 1.0
    for value in (True, np.bool_(False), "1", None, 1j):
        with pytest.raises(TypeError, match="ratio must be a real number"):
            float_argument(value, "ratio")
    for value in (float("nan"), float("inf"), np.float64(-np.inf)):
        with pytest.raises(ValueError, match="ratio must be finite"):
            float_argument(value, "ratio")


def test_clip_fraction_argument_requires_a_finite_fraction_below_one_half():
    for value, expected in ((0, 0.0), (np.int64(0), 0.0), (np.float32(0.25), 0.25)):
        resolved = clip_fraction_argument(value)
        assert type(resolved) is float
        assert resolved == expected
    for value in (True, np.bool_(False), "0.1", None, np.array(0.1)):
        with pytest.raises(TypeError, match="clip_fraction must be a real number"):
            clip_fraction_argument(value)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="clip_fraction must be finite"):
            clip_fraction_argument(value)
    for value in (-0.1, 0.5, 0.75, 1):
        with pytest.raises(
            ValueError, match="tails must be at least 0 and less than 0.5"
        ):
            clip_fraction_argument(value, "tails")


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["a"],
        # Labels of any types pass that group as their text reads.
        ["a", 2, 2.5, True, b"b"],
        [1, "2"],
        [1, np.int64(1), np.uint8(2), 2**70],
        ["a", np.str_("a")],
        [True, np.bool_(True), False],
        [np.float32(7.5), 7.5],
        [b"a", "a"],
        [pd.Timestamp("2020-01-01"), datetime.datetime(2020, 1, 1)],
        # Missing labels are no labels.
        ["a", None, float("nan"), pd.NA, 1],
        # 0.0 and -0.0 form one label, in any array.
        np.asarray([1.5, -0.0, 0.0, np.nan]),
        np.asarray([0.5, np.float64(1.5), -0.0, 0.0, np.nan], dtype=object),
        ["a", 0.0, -0.0, np.float32(-0.0)],
        # A typed array, whatever NumPy made of the labels.
        np.asarray([1, "1"]),
        np.asarray([b"a", b"b"]),
        np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        # A categorical's unused categories are no labels.
        pd.Categorical([1, 1], categories=[1, "1"]),
        pd.Series(["a", "b"], dtype="category"),
        pd.Series([1, None, 2], dtype="Int64"),
    ],
)
def test_require_labels_match_their_text_passes_labels_that_group_as_they_read(
    labels,
):
    require_labels_match_their_text(labels, name="labels")


@pytest.mark.parametrize(
    ("labels", "found"),
    [
        # Python takes 1, 1.0 and True for one value...
        (
            [1, 1.0, True],
            "values that would form one label but print differently, such as 1, "
            "1.0 and True",
        ),
        # ...and keeps 1 and "1" apart, though they print alike.
        (
            [2, 1, "1"],
            "values that print alike but would form separate labels, such as 1 and '1'",
        ),
        # Two values and two texts, but three pairs of them.
        (
            [1, 1.0, "1"],
            "values that would form one label but print differently, such as 1 "
            "and 1.0, and values that print alike but would form separate "
            "labels, such as 1 and '1'",
        ),
        (
            [np.float32(0.1), 0.1],
            "values that print alike but would form separate labels, such as "
            "np.float32(0.1) and 0.1",
        ),
        (
            [decimal.Decimal("1.0"), decimal.Decimal("1")],
            "values that would form one label but print differently, such as "
            "Decimal('1.0') and Decimal('1')",
        ),
        # A zero of another type than float stays apart from 0.0 in print.
        (
            [0, 0.0, -0.0],
            "values that would form one label but print differently, such as 0, "
            "0.0 and -0.0",
        ),
        (
            pd.Categorical(["b", 1, "1"], categories=["b", 1, "1"]),
            "values that print alike but would form separate labels, such as 1 and '1'",
        ),
        # Each conflict is shown by its group that the labels show first.
        (
            ["x", "1", 1.0, 1, "y", 1.5, "1.5"],
            "values that would form one label but print differently, such as 1.0 "
            "and 1, and values that print alike but would form separate labels, "
            "such as '1' and 1",
        ),
        # A long value is quoted by its start and its end.
        (
            ["1" * 200, int("1" * 200)],
            "values that print alike but would form separate labels, such as '"
            + "1" * 37
            + "..."
            + "1" * 38
            + "' and "
            + "1" * 38
            + "..."
            + "1" * 39,
        ),
    ],
)
def test_require_labels_match_their_text_quotes_one_group_of_each_conflict(
    labels, found
):
    with pytest.raises(ValueError) as raised:
        require_labels_match_their_text(labels, name="labels")

    assert (
        str(raised.value)
        == f"labels holds {found}; convert them to text, or to one type, first"
    )


def test_require_labels_match_their_text_refuses_labels_in_two_dimensions():
    with pytest.raises(ValueError, match="^labels must be one-dimensional$"):
        require_labels_match_their_text(np.zeros((2, 2)), name="labels")


def test_require_labels_match_their_text_reads_no_typed_array(monkeypatch):
    def unread(*args, **kwargs):
        raise AssertionError("the labels were read one by one")

    # Ten million integer labels would make as many Python objects to group.
    monkeypatch.setattr(pd, "factorize", unread)
    monkeypatch.setattr(pd.api.types, "infer_dtype", unread)
    for labels in (
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=bool),
        np.asarray(["a", "b"]),
        np.asarray([b"a", b"b"]),
        np.asarray(["2020-01-01"], dtype="datetime64[D]"),
        np.asarray([0.0, -0.0, 1.0], dtype=np.float32),
        np.asarray([1 + 1j, 2 + 0j, 1 + 1j]),
    ):
        require_labels_match_their_text(labels, name="labels")
    with pytest.raises(AssertionError, match="read one by one"):
        require_labels_match_their_text(
            np.asarray(["a", 1], dtype=object), name="labels"
        )


def test_require_labels_match_their_text_groups_no_labels_of_one_type(monkeypatch):
    def ungrouped(*args, **kwargs):
        raise AssertionError("the labels were grouped")

    # Labels of one type of text, bytes, integers or booleans, or Python
    # floats, cannot group otherwise than their text reads.
    monkeypatch.setattr(pd, "factorize", ungrouped)
    for labels in (
        ["a", np.str_("b")],
        [b"a", b"b"],
        [1, np.int64(2)],
        [True, np.bool_(False)],
        [0.5, np.float64(1.5), 0.0, -0.0],
    ):
        require_labels_match_their_text(labels, name="labels")
    with pytest.raises(AssertionError, match="were grouped"):
        require_labels_match_their_text([np.float32(0.5), 1.5], name="labels")


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        # NumPy would read these lists as two texts of "a", two bytes of
        # b"1", and two floats of 2**53.
        ([b"a", "a"], [b"a", "a"]),
        ([b"1", 1], [b"1", 1]),
        ([2**53 + 1, 2**53, 0.5], [2**53 + 1, 2**53, 0.5]),
    ],
)
def test_label_array_keeps_the_type_of_each_label_of_a_list(labels, expected):
    values = label_array(labels)

    assert values.dtype == object
    assert values.tolist() == expected
    assert [type(value) for value in values] == [type(value) for value in expected]
    # An array, or a pandas Series or Categorical, gives its values as they are.
    assert label_array(np.asarray([1, 2])).dtype == np.int64
    assert label_array(pd.Series([1.5, 2.5])).dtype == np.float64
    assert label_array(pd.Categorical(["a", "b"])).tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("labels", "codes", "count"),
    [
        # A categorical keeps its categories, unused ones included, and their
        # order.
        (pd.Categorical(["a", "b", "a"], categories=["b", "a", "c"]), [1, 0, 1], 3),
        # Other labels are numbered in their sorted order...
        (["b", "a", "b", None], [1, 0, 1, -1], 2),
        ([2**53 + 1, 2**53, 0.5], [2, 1, 0], 3),
        # ...or in order of first appearance where they cannot be sorted.
        ([b"1", 1, b"1", "z"], [0, 1, 0, 2], 3),
    ],
)
def test_categorical_label_codes_number_labels_as_a_categorical_would(
    labels, codes, count
):
    numbers, n_codes = categorical_label_codes(labels)

    assert numbers.tolist() == codes
    assert n_codes == count


@pytest.mark.parametrize(
    "pool",
    [
        np.asarray(["b", "a", "c", "10", "2"]),
        np.asarray([3, 1, 2, 10]),
        np.asarray([0.5, 1.5, -2.0]),
        np.asarray([True, False]),
        # pandas sorts integers before text.
        np.asarray(["a", 2, "c", 1, 2.5], dtype=object),
    ],
)
def test_categorical_label_codes_order_a_contingency_table_as_its_labels_do(pool):
    rng = np.random.default_rng(len(pool))
    for _ in range(20):
        left = pool[rng.integers(0, len(pool), 40)]
        right = np.asarray(["x", "y", "z"])[rng.integers(0, 3, 40)]

        coded = pd.crosstab(
            categorical_label_codes(left)[0], categorical_label_codes(right)[0]
        )

        np.testing.assert_array_equal(
            coded.to_numpy(), pd.crosstab(left, right).to_numpy()
        )


def test_read_only_copy_owns_its_values_and_stays_read_only():
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T
    copied = read_only_copy(source, np.int64)
    source[0, 0] = 99

    assert copied.dtype == np.dtype(np.int64)
    assert copied.flags.c_contiguous
    np.testing.assert_array_equal(copied, [[0, 3], [1, 4], [2, 5]])
    with pytest.raises(ValueError, match="read-only"):
        copied[0, 0] = 1
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        copied.setflags(write=True)


def test_sparse_matrix_bytes_counts_shared_arrays_once():
    matrix = csr_matrix(np.eye(3))
    single = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    sharing = matrix.copy()
    sharing.indices = matrix.indices
    sharing.indptr = matrix.indptr

    assert sparse_matrix_bytes(matrix) == single
    assert sparse_matrix_bytes(matrix, matrix) == single
    assert sparse_matrix_bytes(matrix, sharing) == single + sharing.data.nbytes


def test_cumulative_and_max_window_nnz_follow_row_prefix_sums():
    cumulative = cumulative_nnz(np.asarray([2, 0, 5, 1], dtype=np.int32))
    np.testing.assert_array_equal(cumulative, [0, 2, 2, 7, 8])
    assert cumulative.dtype == np.int64
    assert max_window_nnz(cumulative, 1) == 5
    assert max_window_nnz(cumulative, 2) == 6
    assert max_window_nnz(cumulative, 10) == 8
    assert max_window_nnz(cumulative_nnz(np.empty(0, dtype=np.int64)), 3) == 0
    with pytest.raises(ValueError, match="window_rows must be positive"):
        max_window_nnz(cumulative, 0)


def test_assay_feature_ranges_group_spans_in_first_seen_order():
    import pandas as pd

    table = pd.DataFrame(
        [["RNA", 0, 3], ["ADT", 3, 5], ["RNA", 5, 6]],
        columns=["type", "start", "end"],
        index=["RNA", "ADT", "RNA"],
    ).T
    assert assay_feature_ranges(table) == {
        "RNA": ((0, 3), (5, 6)),
        "ADT": ((3, 5),),
    }


def test_read_array_reads_a_whole_array_and_refuses_a_group():
    group = zarr.open_group(MemoryStore(), mode="w")
    group.create_array("values", data=np.arange(12, dtype=np.int32).reshape(3, 4))
    group.create_group("nested")

    found = read_array(group, "values")

    assert isinstance(found, np.ndarray) and found.dtype == np.int32
    np.testing.assert_array_equal(found, np.arange(12).reshape(3, 4))
    with pytest.raises(TypeError, match="Expected Zarr array at 'nested'"):
        read_array(group, "nested")


@pytest.mark.parametrize("value", [True, False])
def test_bool_argument_returns_a_boolean(value):
    assert bool_argument(value, "flag") is value


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_bool_argument_refuses_anything_but_a_boolean(value):
    with pytest.raises(TypeError, match="^flag must be a boolean$"):
        bool_argument(value, "flag")


@pytest.mark.parametrize(("value", "expected"), [(np.True_, True), (np.False_, False)])
def test_bool_argument_takes_a_numpy_boolean_as_a_python_one(value, expected):
    found = bool_argument(value, "flag")
    assert type(found) is bool and found is expected


def test_optional_name_argument_takes_a_name_or_none():
    assert optional_name_argument("lane", "sample_column") == "lane"
    assert optional_name_argument(None, "sample_column") is None
    for value in ("", 3, ["lane"]):
        with pytest.raises(
            TypeError, match="^sample_column must be a non-empty string or None$"
        ):
            optional_name_argument(value, "sample_column")
    with pytest.raises(
        TypeError, match="^id_column must be a feature column name or None$"
    ):
        optional_name_argument("", "id_column", noun="a feature column name")


def _splitmix64(value: int) -> int:
    """SplitMix64's finalizer in Python integers, from its published constants."""
    mask = (1 << 64) - 1
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def test_mix64_is_splitmix64s_finalizer_compiled_and_on_arrays():
    values = np.array([0, 1, 2**63, 2**64 - 1, 0x9E3779B97F4A7C15], dtype=np.uint64)
    expected = [_splitmix64(int(value)) for value in values]

    mixed = mix64_array(values)

    assert mixed.dtype == np.uint64
    assert mixed.tolist() == expected
    assert [int(mix64(value)) for value in values] == expected
    # The first output of SplitMix64 seeded with 0, from its reference code.
    assert _splitmix64(0x9E3779B97F4A7C15) == 0xE220A8397B1DCDAF


def test_numba_threads_sets_a_bounded_count_and_restores_the_last():
    import numba

    previous = numba.get_num_threads()
    most = numba.config.NUMBA_NUM_THREADS
    with numba_threads(1):
        assert numba.get_num_threads() == 1
        with numba_threads(most + 5):
            assert numba.get_num_threads() == most
        assert numba.get_num_threads() == 1
    with numba_threads(0):
        assert numba.get_num_threads() == 1
    assert numba.get_num_threads() == previous
    with pytest.raises(RuntimeError, match="inside"):
        with numba_threads(1):
            raise RuntimeError("inside")
    assert numba.get_num_threads() == previous
