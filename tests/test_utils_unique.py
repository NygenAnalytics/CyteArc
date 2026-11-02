"""The sort-based replacements of ``np.unique`` equal it on integers."""

import numpy as np
import pytest

from cytearc.utils import arrays
from cytearc.utils.arrays import (
    sorted_unique,
    union_sorted,
    unique_inverse,
    unique_rows,
)
from tests.perturbation_helpers import traced_peak


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("high", [1, 3, 1_000, 2**40])
def test_sorted_unique_and_unique_inverse_equal_numpy(seed: int, high: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.integers(-high, high + 1, size=int(rng.integers(0, 400)))
    assert np.array_equal(sorted_unique(values), np.unique(values))
    distinct, inverse = unique_inverse(values)
    expected, expected_inverse = np.unique(values, return_inverse=True)
    assert np.array_equal(distinct, expected)
    assert np.array_equal(inverse, expected_inverse)
    assert inverse.dtype == np.int64
    assert distinct.dtype == values.dtype


def test_sorted_unique_keeps_the_dtype_of_small_and_unsigned_integers() -> None:
    values = np.asarray([3, 1, 3, 255, 0], dtype=np.uint8)
    assert sorted_unique(values).tolist() == [0, 1, 3, 255]
    assert sorted_unique(values).dtype == np.uint8
    hashes = np.asarray([2**64 - 1, 5, 2**64 - 1], dtype=np.uint64)
    assert sorted_unique(hashes).tolist() == [5, 2**64 - 1]
    assert sorted_unique(np.asarray([True, False, True])).tolist() == [False, True]
    assert sorted_unique(np.asarray([7])).tolist() == [7]
    assert sorted_unique(np.asarray([], dtype=np.int64)).tolist() == []


@pytest.mark.parametrize("values", [np.zeros((2, 2), dtype=int), np.asarray([1.5])])
def test_sorted_unique_refuses_other_arrays(values) -> None:
    with pytest.raises(TypeError, match="one-dimensional array of integers"):
        sorted_unique(values)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("columns", [1, 2, 3])
def test_unique_rows_equals_numpy(seed: int, columns: int) -> None:
    rng = np.random.default_rng(seed)
    n_rows = int(rng.integers(1, 500))
    highs = rng.choice([1, 4, 50, 2**31], size=columns)
    rows = np.column_stack(
        [rng.integers(-1, high, size=n_rows) for high in highs]
    ).astype(np.int64)
    distinct, inverse = unique_rows(rows)
    expected, expected_inverse = np.unique(rows, axis=0, return_inverse=True)
    assert np.array_equal(distinct, expected)
    assert np.array_equal(inverse, expected_inverse.reshape(-1))
    assert np.array_equal(distinct[inverse], rows)


def test_unique_rows_never_overflows_wide_columns() -> None:
    big = np.iinfo(np.int64)
    rows = np.asarray(
        [[big.max, big.min], [big.min, big.max], [big.max, big.min], [0, 0]],
        dtype=np.int64,
    )
    distinct, inverse = unique_rows(rows)
    expected, expected_inverse = np.unique(rows, axis=0, return_inverse=True)
    assert np.array_equal(distinct, expected)
    assert np.array_equal(inverse, expected_inverse.reshape(-1))
    unsigned = np.asarray([[2**64 - 1, 1], [2**64 - 2, 1], [2**64 - 1, 1]], np.uint64)
    distinct, inverse = unique_rows(unsigned)
    assert distinct.tolist() == [[2**64 - 2, 1], [2**64 - 1, 1]]
    assert inverse.tolist() == [1, 0, 1]


def test_unique_rows_and_inverse_of_no_values() -> None:
    distinct, inverse = unique_rows(np.zeros((0, 2), dtype=np.int64))
    assert distinct.shape == (0, 2)
    assert inverse.shape == (0,)
    distinct, inverse = unique_inverse(np.asarray([], dtype=np.int32))
    assert distinct.dtype == np.int32 and distinct.shape == (0,)
    assert inverse.dtype == np.int64 and inverse.shape == (0,)


def test_unique_inverse_and_rows_refuse_other_arrays() -> None:
    with pytest.raises(TypeError, match="unique_inverse needs a one-dimensional"):
        unique_inverse(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(TypeError, match="unique_inverse needs a one-dimensional"):
        unique_inverse(np.asarray([0.5, 1.5]))
    with pytest.raises(TypeError, match="unique_rows needs a two-dimensional"):
        unique_rows(np.asarray([1, 2]))
    with pytest.raises(TypeError, match="unique_rows needs a two-dimensional"):
        unique_rows(np.asarray([[True], [False]]))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("high", [2, 50, 2**63 - 1])
def test_union_sorted_equals_numpy(seed: int, high: int) -> None:
    rng = np.random.default_rng(seed)
    held = np.unique(rng.integers(0, high, size=int(rng.integers(0, 300)))).astype(
        np.uint64
    )
    # Values repeat among themselves and with the held ones, in any order.
    values = rng.integers(0, high, size=int(rng.integers(0, 300))).astype(np.uint64)
    if len(held) and len(values):
        values[: len(values) // 3] = rng.choice(held, size=len(values) // 3)
    merged = union_sorted(held, values)
    assert np.array_equal(merged, np.union1d(held, values))
    assert merged.dtype == np.uint64


def test_union_sorted_of_empty_and_extreme_values() -> None:
    hashes = np.asarray([0, 2**64 - 1], dtype=np.uint64)
    assert union_sorted(
        hashes, np.asarray([2**64 - 1, 5, 5], dtype=np.uint64)
    ).tolist() == [0, 5, 2**64 - 1]
    empty = np.empty(0, dtype=np.uint64)
    assert union_sorted(empty, np.asarray([3, 1, 3], dtype=np.uint64)).tolist() == [
        1,
        3,
    ]
    assert union_sorted(hashes, empty) is hashes
    assert union_sorted(empty, empty).tolist() == []
    signed = np.asarray([-5, 7], dtype=np.int64)
    assert union_sorted(signed, np.asarray([-9, 7], dtype=np.int64)).tolist() == [
        -9,
        -5,
        7,
    ]


def test_union_sorted_holds_the_held_values_twice_at_most() -> None:
    held = np.arange(0, 2_000_000, 2, dtype=np.uint64)
    values = np.arange(1, 20_001, 2, dtype=np.uint64)
    union_sorted(held[:4], values[:4])
    # Beside the held values and the new ones: the result, sized exactly, and
    # the sorted copy of the new ones, where sorting all again holds the held
    # ones three more times.
    allowed = 8 * (len(held) + len(values)) + values.nbytes + 4_096
    merged, peak = traced_peak(lambda: union_sorted(held, values), allowed)
    assert len(merged) == len(held) + len(values)
    assert peak <= allowed, peak


def test_union_sorted_kernel_runs_as_python_too() -> None:
    held = np.asarray([1, 4, 9], dtype=np.int64)
    new = np.asarray([0, 4, 4, 10], dtype=np.int64)
    kernel = arrays._union_kernel
    assert kernel(held, new, held[:0]) == 5
    out = np.empty(5, dtype=np.int64)
    assert kernel(held, new, out) == 5
    assert out.tolist() == [0, 1, 4, 9, 10]
    assert kernel(held[:0], new[:0], out[:0]) == 0


@pytest.mark.parametrize(
    ("held", "values"),
    [
        (np.zeros((2, 2), dtype=np.uint64), np.asarray([1], dtype=np.uint64)),
        (np.asarray([1], dtype=np.uint64), np.asarray([1.5])),
        (np.asarray([1], dtype=np.uint64), np.asarray([1], dtype=np.int64)),
    ],
)
def test_union_sorted_refuses_other_arrays(held, values) -> None:
    with pytest.raises(TypeError, match="union_sorted needs"):
        union_sorted(held, values)
