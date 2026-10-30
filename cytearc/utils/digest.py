"""Compiled per-row digests and totals for dense count blocks, and SplitMix64.

SplitMix64's finaliser (Steele, Lea and Flood 2014) mixes a 64-bit integer
into one whose bits each depend on all of its bits. The digests mix each
count's column and value with it, and counter-based generators, such as the
random sets of the association tests, key their streams through it.
"""

import numpy as np

from .numba import njit

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_ODD = np.uint64(0xD6E8FEB86659FD93)


@njit(cache=True, inline="always")
def mix64(value: np.uint64) -> np.uint64:
    """Return the SplitMix64 finaliser of a 64-bit unsigned integer."""
    value = (value ^ (value >> np.uint64(30))) * _M1
    value = (value ^ (value >> np.uint64(27))) * _M2
    return value ^ (value >> np.uint64(31))


def mix64_array(values: np.ndarray) -> np.ndarray:
    """Return the SplitMix64 finaliser of each unsigned 64-bit integer.

    Each value equals :func:`mix64` of it; array products wrap around without
    warnings, as the compiled products do.
    """
    mixed = (values ^ (values >> np.uint64(30))) * _M1
    mixed = (mixed ^ (mixed >> np.uint64(27))) * _M2
    finalised: np.ndarray = mixed ^ (mixed >> np.uint64(31))
    return finalised


@njit(cache=True, nogil=True)
def summarize_rows(
    values: np.ndarray,
    bits: np.ndarray,
    digests: np.ndarray,
    row_sums: np.ndarray,
    row_positive: np.ndarray,
    column_positive: np.ndarray,
) -> None:
    """Digest each row's nonzero column-value pairs and total its values.

    ``bits`` is the unsigned-integer view of ``values``. Zero entries are
    skipped, so digests depend only on each row's sparse content. Positive
    entries are counted per row and added to ``column_positive``.

    Raises:
        ValueError: If a value is NaN or infinite. Count matrices hold finite
            values; the check compiles away for integer values.
    """
    for row in range(values.shape[0]):
        first = np.uint64(0x243F6A8885A308D3)
        second = np.uint64(0x13198A2E03707344)
        nonzero = np.uint64(0)
        positive = 0
        total = 0.0
        for column in range(values.shape[1]):
            value = values[row, column]
            if value != 0:
                if not np.isfinite(value):
                    raise ValueError(
                        "Count matrices must hold finite values; found NaN or infinity"
                    )
                key = mix64(
                    ((np.uint64(column) + np.uint64(1)) * _GOLDEN)
                    ^ np.uint64(bits[row, column])
                )
                first = mix64(first ^ key)
                second = (second + key) * _ODD
                nonzero += np.uint64(1)
                total += value
                if value > 0:
                    positive += 1
                    column_positive[column] += 1
        digests[row, 0] = mix64(first ^ nonzero)
        digests[row, 1] = mix64(second ^ (nonzero * _GOLDEN))
        row_sums[row] = total
        row_positive[row] = positive
