"""Draw the random sets of cells that the association tests compare each group with.

Every random number comes from a counter-based generator: SplitMix64's
finaliser of a key plus a counter. A group's key in a round comes from the
seed, the group's code and the round; each set's key from the group's key and
the set's index; and each uniform within a set from the set's key and a
counter. A set therefore depends only on those numbers, so results do not
depend on threads or on how genes are split into bands.

Two kinds of sets are drawn. A uniform set holds a fixed number of positions,
drawn without replacement by Floyd's algorithm. A Bernoulli set takes every
cell independently with its own probability: the cells are sorted once into
buckets of probability within a factor of two, geometric skips propose cells
of a bucket at its largest probability, and each proposed cell is kept with
its probability over that bound, so each cell enters with exactly its
probability at a cost of about twice the set's size. A Bernoulli set that
holds no cell or every cell is drawn again with the set's next uniforms, so
the sets hold some cells but not all, as the groups they stand for do.
"""

import math

import numpy as np
from numba import prange

from cytearc.utils.digest import mix64, mix64_array
from cytearc.utils.numba import njit

__all__ = [
    "N_BUCKETS",
    "draw_bernoulli_set",
    "draw_uniform_range",
    "draw_uniform_set",
    "draw_uniform_sets",
    "probability_buckets",
    "resampling_keys",
    "set_key",
    "unit_uniform",
]

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SALT = np.uint64(0x5CE97E2A55A0C1A7)
_SHIFT_11 = np.uint64(11)
_UNIT_INTERVAL = 2.0**-53
#: The buckets of probability of a Bernoulli set: bucket 0 holds the
#: probabilities of 1/2 or more, bucket k the probabilities from 2^-(k+1) up
#: to 2^-k, and the last bucket every smaller probability, 0 included.
N_BUCKETS = 64


def resampling_keys(seed: int, codes: np.ndarray, stage: int) -> np.ndarray:
    """Return the key of the random sets of each group in one round.

    Args:
        seed: The seed of the test.
        codes: A distinct, non-negative code for each group, such as its
            target's position in the labels.
        stage: The round, 1, 2 or 3.

    Returns:
        One unsigned 64-bit key per group.
    """
    base = mix64_array(np.array([seed % 2**64], dtype=np.uint64) ^ _SALT)
    groups = mix64_array(np.asarray(codes, dtype=np.int64).astype(np.uint64) + _GOLDEN)
    rounds = mix64_array(np.array([stage], dtype=np.uint64) + _GOLDEN)
    keys: np.ndarray = mix64_array(mix64_array(base ^ groups) ^ rounds)
    return keys


@njit(cache=True, nogil=True)
def set_key(key: np.uint64, index: int) -> np.uint64:
    """Return the key of a group's random set ``index`` from the group's key."""
    return mix64(key ^ mix64(np.uint64(index) + _GOLDEN))


@njit(cache=True, nogil=True)
def unit_uniform(key: np.uint64, counter: int) -> float:
    """Return the uniform number ``counter`` of a set's key, above 0 and at most 1.

    It is a multiple of 2^-53, never 0, so its logarithm is finite.
    """
    mixed = mix64(key + np.uint64(counter) * _GOLDEN)
    return (np.float64(mixed >> _SHIFT_11) + 1.0) * _UNIT_INTERVAL


@njit(cache=True, nogil=True)
def draw_uniform_set(
    key: np.uint64, size: int, n_union: int, marked: np.ndarray, out: np.ndarray
) -> None:
    """Draw a uniformly random set of ``size`` of ``n_union`` positions.

    Floyd's algorithm takes one draw per member; ``marked`` is all False
    before and after. A uniform of 53 bits is at most 1 - 2^-53, and its
    product with ``last + 1`` rounds below ``last + 1``, so ``pick`` is at
    most ``last``.
    """
    for count in range(size):
        last = n_union - size + count
        mixed = mix64(key + np.uint64(count) * _GOLDEN)
        pick = int(np.float64(mixed >> _SHIFT_11) * _UNIT_INTERVAL * (last + 1))
        if marked[pick]:
            pick = last
        marked[pick] = True
        out[count] = pick
    for count in range(size):
        marked[out[count]] = False


@njit(cache=True, nogil=True)
def draw_uniform_range(
    key: np.uint64,
    first: int | np.int64,
    step: int,
    size: int,
    n_union: int,
    draws: np.ndarray,
) -> None:
    """Draw the uniform sets ``first``, ``first + step``, ... of a group."""
    marked = np.zeros(n_union, dtype=np.bool_)
    picks = np.empty(size, dtype=np.int64)
    for d in range(first, draws.shape[0], step):
        draw_uniform_set(set_key(key, d), size, n_union, marked, picks)
        for i in range(size):
            draws[d, i] = picks[i]


@njit(cache=True, nogil=True, parallel=True)
def draw_uniform_sets(
    key: np.uint64, n_draws: int, size: int, n_union: int
) -> np.ndarray:
    """Return a group's uniform sets, sets x members, as positions among its cells.

    Set ``d`` depends only on ``key`` and ``d``, so the sets do not depend on
    how the draws are split among threads.
    """
    draws = np.empty((n_draws, size), dtype=np.int32)
    n_chunks = min(n_draws, 64)
    for chunk in prange(n_chunks):
        # prange's index is unsigned; np.int64 keeps every call on one compiled
        # version of the kernel.
        draw_uniform_range(key, np.int64(chunk), n_chunks, size, n_union, draws)
    return draws


@njit(cache=True, nogil=True)
def _bucket(probability: float, n_buckets: int) -> int:
    """Return the bucket of a probability; see :data:`N_BUCKETS`."""
    if probability >= 0.5:
        return 0
    if not probability > 0:
        return n_buckets - 1
    # probability = fraction * 2^exponent with fraction in [1/2, 1), so it
    # lies in [2^(exponent - 1), 2^exponent): bucket -exponent.
    exponent = math.frexp(probability)[1]
    return min(-exponent, n_buckets - 1)


@njit(cache=True, nogil=True)
def probability_buckets(
    probabilities: np.ndarray,
    order: np.ndarray,
    offsets: np.ndarray,
    bounds: np.ndarray,
) -> None:
    """Sort cells into buckets of probability for :func:`draw_bernoulli_set`.

    Args:
        probabilities: Each cell's probability, from 0 to 1.
        order: Receives the cells, bucket after bucket, each bucket in
            ascending order of cell.
        offsets: Receives where each bucket starts in ``order``, and the end
            of the last; one more entry than ``bounds``.
        bounds: Receives each bucket's bound, at least its probabilities: 1
            for bucket 0 and 2^-k for bucket k. Its length sets the buckets.
    """
    n_buckets = len(bounds)
    counts = np.zeros(n_buckets, dtype=np.int64)
    for j in range(len(probabilities)):
        counts[_bucket(probabilities[j], n_buckets)] += 1
    offsets[0] = 0
    for k in range(n_buckets):
        offsets[k + 1] = offsets[k] + counts[k]
        bounds[k] = math.ldexp(1.0, -k)
    fill = offsets[:n_buckets].copy()
    for j in range(len(probabilities)):
        k = _bucket(probabilities[j], n_buckets)
        order[fill[k]] = j
        fill[k] += 1


@njit(cache=True, nogil=True)
def _draw_bernoulli(
    key: np.uint64,
    counter: int,
    probabilities: np.ndarray,
    order: np.ndarray,
    offsets: np.ndarray,
    bounds: np.ndarray,
    out: np.ndarray,
) -> tuple[int, int]:
    """Draw every cell independently with its probability, from uniform ``counter`` on.

    Returns how many cells were drawn and the first uniform left unused; see
    :func:`draw_bernoulli_set`.
    """
    count = 0
    for k in range(len(bounds)):
        start = offsets[k]
        end = offsets[k + 1]
        if start == end:
            continue
        bound = bounds[k]
        if k == 0:
            for position in range(start, end):
                cell = order[position]
                if unit_uniform(key, counter) <= probabilities[cell]:
                    out[count] = cell
                    count += 1
                counter += 1
            continue
        log_miss = math.log1p(-bound)
        position = start - 1
        while True:
            # The cells skipped before the next proposal: at least j with
            # probability (1 - bound)^j. A skip past the bucket ends it, even
            # one too large for an integer.
            skip = math.log(unit_uniform(key, counter)) / log_miss
            counter += 1
            if skip >= end - position - 1:
                break
            position += int(skip) + 1
            cell = order[position]
            # bound is a power of 2, so the product is exact.
            if unit_uniform(key, counter) * bound <= probabilities[cell]:
                out[count] = cell
                count += 1
            counter += 1
    return count, counter


@njit(cache=True, nogil=True)
def draw_bernoulli_set(
    key: np.uint64,
    probabilities: np.ndarray,
    order: np.ndarray,
    offsets: np.ndarray,
    bounds: np.ndarray,
    out: np.ndarray,
) -> int:
    """Draw every cell independently with its probability, given some cells but not all.

    The drawn cells go to the start of ``out``, which holds room for every
    cell, in the order of :func:`probability_buckets`. Each cell of bucket 0
    takes a uniform. In every other bucket, the gaps between proposed cells
    are geometric, so each cell is proposed with the bucket's bound, and a
    proposed cell is kept with its probability over the bound, at least 1/2:
    every cell enters with its probability, to within 2^-53.

    A draw that holds no cell or every cell is drawn again with the uniforms
    that follow it, until a draw holds some cells but not all. The set
    therefore follows the independent draws conditioned on holding at least
    one cell and leaving at least one out. The uniforms are those of ``key``
    and counters 0, 1, ..., so the set depends only on the key and the
    probabilities. The draws repeat until one succeeds, so the probabilities
    must give such a set a fair chance: when they add up to at least 1/2 and
    leave at least 1/2 out, at least one draw in six succeeds.

    Args:
        key: The set's key, from :func:`set_key`.
        probabilities: Each cell's probability.
        order: The cells by bucket, from :func:`probability_buckets`.
        offsets: Where each bucket starts in ``order``.
        bounds: Each bucket's bound.
        out: Receives the drawn cells.

    Returns:
        The number of cells drawn.
    """
    n = len(probabilities)
    # int() gives an int64 counter rather than a literal, as the later draws
    # pass, so every call uses one compiled version of the kernel.
    count, counter = _draw_bernoulli(
        key, int(0), probabilities, order, offsets, bounds, out
    )
    while count == 0 or count == n:
        count, counter = _draw_bernoulli(
            key, counter, probabilities, order, offsets, bounds, out
        )
    return count
