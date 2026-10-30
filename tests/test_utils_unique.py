"""The sort-based replacement of ``np.unique`` equals it on integers."""

import numpy as np
import pytest

from cytearc.utils.arrays import sorted_unique


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("high", [1, 3, 1_000, 2**40])
def test_sorted_unique_equals_numpy(seed: int, high: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.integers(-high, high + 1, size=int(rng.integers(0, 400)))
    assert np.array_equal(sorted_unique(values), np.unique(values))


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
