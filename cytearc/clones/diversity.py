"""The diversity of the clones within a group of cells.

A group's clone sizes, the cells of each of its clones, describe its clonal
structure as species abundances describe an ecological sample. The classic
indices summarize them as scRepertoire, immunarch and vegan report them, and
Hill numbers of orders 0, 1 and 2, the effective numbers of clones, count all
clones, weigh them by their frequency, or count only the dominant ones. Every
observed index depends on how many cells were sampled, so groups of
different sizes are compared by rarefying and extrapolating their Hill
numbers to a common number of cells, or to a common sample coverage, the
share of the population's cells whose clone the sample holds (Chao and Jost
2012; Chao et al. 2014, 2023).

The estimators follow the papers: Chao1 (Chao 1984, 1987) and its
bias-corrected form (Chao 2005), ACE (Chao and Lee 1992), the Shannon entropy
estimator of Chao, Wang and Jost (2013), Good's unbiased Simpson estimator,
the rarefaction and extrapolation of Chao et al. (2014) with the Shannon
extrapolation of Chao et al. (2023), and the sample coverage of Chao and Jost
(2012). Rarefied probabilities are products of ratios rather than differences
of log-gamma functions, which keeps them exact for groups of millions of
cells. Compiled loops rarefy a group to many sizes in one call. These
functions work on clone sizes and never touch storage.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import digamma

from ..storage.refs import ArtifactRef
from ..utils.arrays import sorted_unique
from ..utils.numba import njit

__all__ = [
    "ORDERS",
    "RARE_THRESHOLD",
    "CloneSizes",
    "ClonalDiversity",
    "Rarefaction",
    "ace",
    "chao1",
    "chao1_bias_corrected",
    "coverage",
    "default_sizes",
    "diversity_at",
    "diversity_tables",
    "estimated_hill",
    "hill_curve",
    "indices",
    "observed_hill",
    "rarefaction_curves",
    "size_for_coverage",
]

#: The orders of the Hill numbers that rarefaction and extrapolation reach:
#: 0 counts clones, 1 weighs them by frequency (the exponential of
#: Shannon entropy), and 2 counts the dominant ones (the inverse Simpson
#: index).
ORDERS = (0, 1, 2)
#: Clones of at most this many cells are rare for ACE (Chao and Lee 1992).
RARE_THRESHOLD = 10
# Terms of the entropy estimator's series summed at once.
_CHUNK = 1 << 20
# How a Hill number of a sample size was reached, by code.
_METHODS = ("rarefaction", "observed", "extrapolation")
_RAREFACTION, _OBSERVED, _EXTRAPOLATION = range(3)
# The sizes that each round of the search for a coverage evaluates at once.
_COVERAGE_PROBES = 16


@dataclass(frozen=True, slots=True)
class CloneSizes:
    """A group's clone sizes, summarized by how many clones have each size.

    Attributes:
        values: The distinct clone sizes, ascending.
        counts: The clones of each size.
        n: The cells, the sum of the clone sizes.
        richness: The number of clones.
        f1: The clones of one cell, singletons.
        f2: The clones of two cells, doubletons.
    """

    values: np.ndarray
    counts: np.ndarray
    n: int
    richness: int
    f1: int
    f2: int

    @classmethod
    def from_sizes(cls, sizes: Any) -> "CloneSizes":
        """Summarize clone sizes; sizes of 0 are left out."""
        array = np.asarray(sizes, dtype=np.int64).reshape(-1)
        if (array < 0).any():
            raise ValueError("Clone sizes must not be negative")
        # Sorting and comparing neighbours counts each size, as np.unique
        # does, without hashing many distinct sizes.
        ordered = np.sort(array[array > 0])
        starts = np.flatnonzero(np.diff(ordered, prepend=0) != 0)
        values = ordered[starts]
        counts = np.diff(starts, append=len(ordered))
        return cls(
            values=values.astype(np.int64),
            counts=counts.astype(np.int64),
            n=int((values * counts).sum()),
            richness=int(counts.sum()),
            f1=int(counts[values == 1].sum()),
            f2=int(counts[values == 2].sum()),
        )

    def frequencies(self) -> tuple[np.ndarray, np.ndarray]:
        """Return each distinct size's clone frequency and its number of clones."""
        return self.values / self.n, self.counts


def _shannon(clones: CloneSizes) -> float:
    """The observed Shannon entropy, in nats."""
    p, counts = clones.frequencies()
    return float(-np.sum(counts * p * np.log(p)))


def observed_hill(clones: CloneSizes, order: float) -> float:
    """The observed Hill number of a real order, the effective number of clones.

    Order 0 is the number of clones, order 1 the exponential of the Shannon
    entropy, and order 2 the inverse Simpson index.
    """
    if clones.n == 0:
        return float("nan")
    p, counts = clones.frequencies()
    if order == 1:
        return float(np.exp(_shannon(clones)))
    return float(np.sum(counts * p**order) ** (1 / (1 - order)))


def _simpson_pairs(clones: CloneSizes) -> float:
    """Good's unbiased estimate of the Simpson concentration, sum of p squared."""
    if clones.n < 2:
        return float("nan")
    pairs = np.sum(clones.counts * clones.values * (clones.values - 1))
    return float(pairs / (clones.n * (clones.n - 1)))


def coverage(clones: CloneSizes) -> float:
    """The estimated sample coverage of the group's cells (Chao and Jost 2012).

    The coverage is the share of the population's cells whose clone the
    sample holds; one minus it is the chance that one more cell brings a new
    clone. Without singletons the sample is complete.
    """
    if clones.n == 0:
        return float("nan")
    if clones.f1 == 0:
        return 1.0
    return float(1 - clones.f1 / clones.n * _undetected_ratio(clones))


def _undetected_ratio(clones: CloneSizes) -> float:
    """The ratio by which the coverage deficit shrinks with each added cell.

    It is ``(n - 1) f1 / ((n - 1) f1 + 2 f2)`` with doubletons, and with
    ``f1 - 1`` and ``f2 = 1`` without them, as in the entropy estimator of
    Chao, Wang and Jost (2013).
    """
    n, f1, f2 = clones.n, clones.f1, clones.f2
    if f2 > 0:
        return (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
    return (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)


def chao1(clones: CloneSizes) -> float:
    """The Chao1 estimate of the clones in the population (Chao 1984, 1987).

    ``S + (n - 1) / n * f1**2 / (2 f2)``, or ``S + (n - 1) / n * f1 (f1 - 1) /
    2`` without doubletons, as iNEXT and immunarch estimate richness.
    """
    return clones.richness + _undetected(clones)


def _undetected(clones: CloneSizes) -> float:
    """The estimated clones that the sample missed, Chao1's ``f0``."""
    n, f1, f2 = clones.n, clones.f1, clones.f2
    if n == 0:
        return 0.0
    if f2 > 0:
        return (n - 1) / n * f1**2 / (2 * f2)
    return (n - 1) / n * f1 * (f1 - 1) / 2


def chao1_bias_corrected(clones: CloneSizes) -> float:
    """The bias-corrected Chao1 estimate (Chao 2005): ``S + f1 (f1 - 1) / (2 (f2 + 1))``.

    It is the form that vegan's ``estimateR`` and scRepertoire's ``chao1``
    report, without the ``(n - 1) / n`` factor.
    """
    f1, f2 = clones.f1, clones.f2
    return clones.richness + f1 * (f1 - 1) / (2 * (f2 + 1))


def ace(clones: CloneSizes, rare: int = RARE_THRESHOLD) -> float:
    """The abundance-based coverage estimate of the clones (Chao and Lee 1992).

    Clones of at most ``rare`` cells are rare. The estimate is
    ``S_abundant + S_rare / C + f1 / C * gamma**2``, where ``C = 1 - f1 /
    N_rare`` is the coverage of the rare cells and ``gamma**2`` the squared
    coefficient of variation of the rare clone sizes, at least 0. It is NaN
    when every rare clone is a singleton, since then ``C = 0``.
    """
    rare_mask = clones.values <= rare
    values, counts = clones.values[rare_mask], clones.counts[rare_mask]
    abundant = int(clones.counts[~rare_mask].sum())
    rare_clones = int(counts.sum())
    if rare_clones == 0:
        return float(abundant)
    rare_cells = int((values * counts).sum())
    rare_coverage = 1 - clones.f1 / rare_cells
    if rare_coverage <= 0:
        return float("nan")
    pairs = float(np.sum(values * (values - 1) * counts))
    gamma = max(
        rare_clones / rare_coverage * pairs / (rare_cells * (rare_cells - 1)) - 1,
        0.0,
    )
    return float(
        abundant + rare_clones / rare_coverage + clones.f1 / rare_coverage * gamma
    )


def _entropy_tail(n: int, a: float) -> float:
    """``(1 - a) ** (1 - n) * (-log(a) - sum((1 - a) ** r / r, r = 1..n-1))``.

    The bracket is the tail of the series of ``-log(a)`` from ``r = n``, so
    the whole equals ``sum((1 - a) ** j / (j + n - 1), j >= 1)``. When the
    factor ``(1 - a) ** (1 - n)`` is moderate the tail is computed as written,
    and otherwise by its series, which then converges in a few multiples of
    ``n`` terms; either avoids overflow and loss of precision.
    """
    decay = -np.log1p(-a)
    if (n - 1) * decay <= 5:
        partial = 0.0
        for start in range(1, n, _CHUNK):
            r = np.arange(start, min(start + _CHUNK, n), dtype=np.float64)
            partial += float(np.sum(np.exp(r * np.log1p(-a)) / r))
        return float(np.exp((n - 1) * decay) * (-np.log(a) - partial))
    total = 0.0
    start = 1
    while True:
        j = np.arange(start, start + _CHUNK, dtype=np.float64)
        terms = np.exp(-j * decay) / (j + n - 1)
        total += float(terms.sum())
        if terms[-1] < 1e-18 * total:
            return total
        start += _CHUNK


def chao_entropy(clones: CloneSizes) -> float:
    """The Shannon entropy estimate of Chao, Wang and Jost (2013), in nats.

    The first term sums, over clones of 1 to ``n - 1`` cells, their
    frequency times ``1 / X + ... + 1 / (n - 1)``; the second estimates the
    entropy of the clones the sample missed from the singletons and
    doubletons through ``A = 2 f2 / ((n - 1) f1 + 2 f2)``, or ``2 / ((n - 1)
    (f1 - 1) + 2)`` without doubletons.
    """
    n, f1, f2 = clones.n, clones.f1, clones.f2
    if n == 0:
        return float("nan")
    inner = clones.values <= n - 1
    values, counts = clones.values[inner], clones.counts[inner]
    first = float(np.sum(counts * values / n * (digamma(n) - digamma(values))))
    if f1 == 0:
        return first
    if f2 > 0:
        a = 2 * f2 / ((n - 1) * f1 + 2 * f2)
    else:
        a = 2 / ((n - 1) * (f1 - 1) + 2)
    if a >= 1:
        return first
    return first + f1 / n * _entropy_tail(n, a)


def estimated_hill(clones: CloneSizes, order: int) -> float:
    """The asymptotic Hill number of order 0, 1 or 2, as iNEXT's ``AsyEst``.

    Order 0 is Chao1, order 1 the exponential of the entropy estimate of
    Chao, Wang and Jost (2013), and order 2 the inverse of Good's unbiased
    Simpson estimate, infinite when no clone holds two cells.
    """
    _check_order(order)
    if clones.n == 0:
        return float("nan")
    if order == 0:
        return chao1(clones)
    if order == 1:
        return float(np.exp(chao_entropy(clones)))
    pairs = _simpson_pairs(clones)
    if not pairs > 0:
        return float("inf")
    return 1 / pairs


def _check_order(order: Any) -> None:
    if isinstance(order, bool) or order not in ORDERS:
        raise ValueError("order must be 0, 1 or 2")


@njit(cache=True)
def _missing_logs(values: np.ndarray, n: int, sizes: np.ndarray) -> np.ndarray:
    """The log chance that a sample of each size misses each distinct size.

    A clone of ``X`` of the ``n`` cells is missed by a sample of ``m``
    cells with chance ``C(n - X, m) / C(n, m)``, the product of ``1 - m / (n
    - j)`` for ``j`` below ``X``, summed here as logarithms in order of ``j``
    so that the chance keeps its precision when it is close to 1; it is
    ``-inf`` when ``X > n - m``. Row ``s`` holds the sample of ``sizes[s]``
    cells and column ``i`` the distinct size ``values[i]``, ascending.
    """
    missing = np.empty((len(sizes), len(values)))
    for row in range(len(sizes)):
        m = sizes[row]
        total = 0.0
        j = 0
        for column in range(len(values)):
            size = values[column]
            if size > n - m:
                missing[row, column] = -np.inf
                continue
            while j < size:
                total += math.log1p(-m / (n - j))
                j += 1
            missing[row, column] = total
    return missing


@njit(cache=True)
def _expected_entropies(
    values: np.ndarray, counts: np.ndarray, n: int, sizes: np.ndarray
) -> np.ndarray:
    """The expected Shannon entropy of a sample of each size below ``n``.

    For each distinct size ``X`` the cells drawn from one such clone
    follow the hypergeometric distribution, whose probabilities are built
    from the ratio of consecutive terms, outward from the mode, and
    normalized over the support, which keeps them to a few units in the last
    place; terms beyond 40 standard deviations of the mode, below ``e**-800``
    of it, are left out.
    """
    entropies = np.empty(len(sizes))
    logs = np.empty(64)
    for at in range(len(sizes)):
        m = sizes[at]
        total = 0.0
        for index in range(len(values)):
            size = values[index]
            low = max(0, m - (n - size))
            high = min(size, m)
            mean = m * size / n
            spread = math.sqrt(mean * (1 - size / n) * (n - m) / max(n - 1, 1))
            mode = min(max((m + 1) * (size + 1) // (n + 2), low), high)
            reach = int(math.ceil(40 * spread)) + 2
            first = max(low, mode - reach)
            length = min(high, mode + reach) - first + 1
            if length > len(logs):
                logs = np.empty(2 * length)
            # The log of each term over the first, by the ratio of
            # consecutive terms.
            logs[0] = 0.0
            running = 0.0
            for offset in range(1, length):
                k = float(first + offset)
                running += (
                    math.log(size - k + 1)
                    + math.log(m - k + 1)
                    - math.log(k)
                    - math.log(n - size - m + k)
                )
                logs[offset] = running
            shift = logs[mode - first]
            weights = 0.0
            entropy = 0.0
            for offset in range(length):
                weight = math.exp(logs[offset] - shift)
                weights += weight
                drawn = first + offset
                if drawn > 0:
                    share = drawn / m
                    entropy += weight * (-share * math.log(share))
            total += counts[index] * entropy / weights
        entropies[at] = total
    return entropies


def _rarefied(
    clones: CloneSizes,
    sizes: np.ndarray,
    order: int,
    missing: Callable[[], np.ndarray],
) -> np.ndarray:
    """The expected Hill numbers of samples of whole sizes below the ``n`` cells.

    ``missing`` gives the sizes' :func:`_missing_logs`, which order 0 needs.
    """
    if order == 0:
        return np.sum(clones.counts * -np.expm1(missing()), axis=1)
    if order == 2:
        return 1 / (1 / sizes + (1 - 1 / sizes) * _pairs_or_zero(clones))
    return np.exp(_expected_entropies(clones.values, clones.counts, clones.n, sizes))


def _pairs_or_zero(clones: CloneSizes) -> float:
    pairs = _simpson_pairs(clones)
    return 0.0 if np.isnan(pairs) else pairs


def _extrapolated(
    clones: CloneSizes,
    sizes: np.ndarray,
    order: int,
    shannon: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """The expected Hill numbers of samples of more than the ``n`` cells.

    Order 0 adds Chao1's missed clones as more cells find them; order 2
    extends Good's Simpson estimate; order 1 approaches the asymptotic
    estimate geometrically from the observed value, at the rate that makes
    the curve pass through the rarefied value at ``n - 1`` cells, the unified
    form of Chao et al. (2023) that reduces to the richness formula at order
    0 (Chao et al. 2014, 2023). ``shannon`` holds the terms of
    :func:`_shannon_terms`, when known.
    """
    n, f1 = clones.n, clones.f1
    extra = sizes - n
    if order == 0:
        missed = _undetected(clones)
        if missed == 0:
            return np.full(len(sizes), float(clones.richness))
        return clones.richness + missed * (1 - (1 - f1 / (n * missed + f1)) ** extra)
    if order == 2:
        return 1 / (1 / sizes + (1 - 1 / sizes) * _pairs_or_zero(clones))
    observed, asymptote, rate = _shannon_terms(clones) if shannon is None else shannon
    return observed + (asymptote - observed) * (1 - (1 - rate) ** extra)


def _shannon_terms(
    clones: CloneSizes, asymptote: float | None = None
) -> tuple[float, float, float]:
    """The observed and asymptotic Shannon diversity and the extrapolation rate.

    The rate is ``(D(n) - D(n - 1)) / (D(inf) - D(n - 1))``, so the curve passes
    through the rarefied value at ``n - 1`` cells; it is 0, a flat curve, when
    the asymptote does not exceed the observed value. ``asymptote`` passes
    the asymptotic estimate, when known.
    """
    observed = observed_hill(clones, 1)
    if asymptote is None:
        asymptote = estimated_hill(clones, 1)
    if clones.n < 2 or not asymptote > observed:
        return observed, asymptote, 0.0
    entropy = _expected_entropies(
        clones.values, clones.counts, clones.n, np.asarray([clones.n - 1])
    )
    before = float(np.exp(entropy[0]))
    return observed, asymptote, (observed - before) / (asymptote - before)


def _rarefied_coverage(
    clones: CloneSizes, sizes: np.ndarray, missing: np.ndarray | None = None
) -> np.ndarray:
    """The expected coverage of samples of whole sizes below the ``n`` cells.

    Each clone of ``X`` cells contributes its frequency times
    ``C(n - X, m) / C(n - 1, m)``, the chance of missing it divided by the
    chance ``1 - m / n`` of missing one given cell. ``missing`` passes the
    sizes' :func:`_missing_logs`, when known.
    """
    n = clones.n
    if missing is None:
        missing = _missing_logs(clones.values, n, sizes)
    chance = np.exp(missing - np.log1p(-sizes / n)[:, None])
    return 1 - np.sum(clones.counts * clones.values / n * chance, axis=1)


def _extrapolated_coverage(clones: CloneSizes, sizes: np.ndarray) -> np.ndarray:
    """The expected coverage of samples of more than the ``n`` cells (Chao and Jost 2012)."""
    if clones.f1 == 0:
        return np.ones(len(sizes))
    return 1 - clones.f1 / clones.n * _undetected_ratio(clones) ** (
        sizes - clones.n + 1
    )


def _sample_sizes(sizes: Any) -> np.ndarray:
    """Return sample sizes as floats, each at least one cell."""
    array = np.asarray(sizes, dtype=np.float64).reshape(-1)
    if not np.all(array >= 1):
        raise ValueError("A sample holds at least one cell")
    return array


class _Samples:
    """Samples of some sizes of a group's cells, and how each is reached.

    A sample smaller than the ``n`` cells is rarefied, at ``n`` observed, and
    larger extrapolated. A size between two whole numbers below ``n``
    interpolates linearly between them, as iNEXT does, and above ``n - 1``
    toward the observed value. Each whole size below ``n`` is rarefied once,
    for every order and the coverage.
    """

    def __init__(self, clones: CloneSizes, sizes: np.ndarray) -> None:
        n = clones.n
        self.clones = clones
        self.sizes = sizes
        self.above = sizes > n
        self.below = sizes < n
        self.methods = np.full(len(sizes), _OBSERVED, dtype=np.int8)
        self.methods[self.above] = _EXTRAPOLATION
        self.methods[self.below] = _RAREFACTION
        self.any_above = bool(self.above.any())
        self.any_below = bool(self.below.any())
        self._missing: np.ndarray | None = None
        if self.any_below:
            self.low = np.floor(sizes[self.below]).astype(np.int64)
            self.fraction = sizes[self.below] - self.low
            self.between = self.fraction > 0
            upper = self.low + 1
            self.inner = self.between & (upper < n)
            self.whole = sorted_unique(np.concatenate([self.low, upper[self.inner]]))
            self._low_at = np.searchsorted(self.whole, self.low)
            self._upper_at = np.searchsorted(self.whole, upper[self.inner])

    def _missed(self) -> np.ndarray:
        """The :func:`_missing_logs` of the whole sizes, computed once."""
        if self._missing is None:
            self._missing = _missing_logs(self.clones.values, self.clones.n, self.whole)
        return self._missing

    def _combine(
        self,
        at_n: float,
        extrapolate: Callable[[np.ndarray], np.ndarray],
        rarefy: Callable[[], np.ndarray],
    ) -> np.ndarray:
        values = np.full(len(self.sizes), at_n)
        if self.any_above:
            values[self.above] = extrapolate(self.sizes[self.above])
        if self.any_below:
            reached = rarefy()
            below = reached[self._low_at]
            between = self.between
            if between.any():
                upper = np.full(len(below), at_n)
                upper[self.inner] = reached[self._upper_at]
                below[between] += (upper[between] - below[between]) * self.fraction[
                    between
                ]
            values[self.below] = below
        return values

    def hill(
        self,
        order: int,
        shannon: tuple[float, float, float] | None = None,
        observed: float | None = None,
    ) -> np.ndarray:
        """The Hill numbers of an order.

        ``shannon`` passes the terms of :func:`_shannon_terms` and
        ``observed`` the observed Hill number, when known.
        """
        clones = self.clones
        return self._combine(
            observed_hill(clones, order) if observed is None else observed,
            lambda sizes: _extrapolated(clones, sizes, order, shannon),
            lambda: _rarefied(clones, self.whole, order, self._missed),
        )

    def coverage(self) -> np.ndarray:
        """The expected coverage of each sample."""
        clones = self.clones
        return self._combine(
            coverage(clones),
            lambda sizes: _extrapolated_coverage(clones, sizes),
            lambda: _rarefied_coverage(clones, self.whole, self._missed()),
        )


def coverage_at(clones: CloneSizes, m: float) -> float:
    """The expected coverage of a sample of ``m`` cells, rarefied or extrapolated.

    A size between two whole numbers below ``n`` interpolates linearly, as
    iNEXT does.
    """
    return float(_Samples(clones, _sample_sizes(m)).coverage()[0])


def diversity_at(
    clones: CloneSizes,
    m: float,
    order: int,
    *,
    shannon: tuple[float, float, float] | None = None,
) -> tuple[float, str]:
    """The Hill number of a sample of ``m`` cells, and how it was reached.

    Below ``n`` cells the sample is rarefied, interpolating linearly between
    whole numbers of cells as iNEXT does; at ``n`` it is observed; above, it
    is extrapolated. Sizes below 1 are not samples. ``shannon`` passes the
    terms of the Shannon extrapolation computed once for many sizes.

    Returns:
        The Hill number and ``"rarefaction"``, ``"observed"`` or
        ``"extrapolation"``.
    """
    _check_order(order)
    samples = _Samples(clones, _sample_sizes(m))
    return float(samples.hill(order, shannon)[0]), _METHODS[int(samples.methods[0])]


def size_for_coverage(clones: CloneSizes, level: float) -> float:
    """The cells that a sample needs to reach a coverage, as iNEXT's ``invChat``.

    Below the observed coverage the size is found among the rarefied sizes
    and interpolated linearly between whole numbers; above it, the
    extrapolated coverage is inverted. A level below the coverage of one cell
    needs one cell.

    Raises:
        ValueError: If the level is not between 0 and 1.
    """
    if not 0 < level < 1:
        raise ValueError("coverage must be between 0 and 1")
    n = clones.n
    observed = coverage(clones)
    if level == observed:
        return float(n)
    if level > observed:
        # Below full coverage there are singletons, and the deficit shrinks by
        # a ratio strictly between 0 and 1 with each added cell.
        ratio = _undetected_ratio(clones)
        return float(n + np.log((1 - level) * n / clones.f1) / np.log(ratio) - 1)
    low_value = coverage_at(clones, 1)
    if low_value >= level:
        return 1.0
    # The rarefied coverage grows with the size: narrow the whole sizes
    # around the level, several at a time, to the last below it.
    low, high, high_value = 1, n, observed
    while high - low > 1:
        if high - low - 1 <= _COVERAGE_PROBES:
            probes = np.arange(low + 1, high, dtype=np.int64)
        else:
            spaced = np.linspace(low, high, _COVERAGE_PROBES + 2)[1:-1]
            probes = sorted_unique(spaced.astype(np.int64))
        values = _rarefied_coverage(clones, probes)
        reached = values >= level
        first = int(np.argmax(reached)) if reached.any() else len(probes)
        if first > 0:
            low, low_value = int(probes[first - 1]), float(values[first - 1])
        if first < len(probes):
            high, high_value = int(probes[first]), float(values[first])
    return float(low + (level - low_value) / (high_value - low_value))


def hill_curve(
    clones: CloneSizes, sizes: Sequence[float], order: int
) -> tuple[list[float], list[str]]:
    """The Hill numbers of samples of each size, as :func:`diversity_at` gives them.

    The sizes are computed together, and the terms that every extrapolated
    size of order 1 shares once.
    """
    _check_order(order)
    samples = _Samples(clones, _sample_sizes(sizes))
    shannon = _shannon_terms(clones) if order == 1 and samples.any_above else None
    values = samples.hill(order, shannon)
    return values.tolist(), [_METHODS[code] for code in samples.methods.tolist()]


def default_sizes(n: int, endpoint: int | None = None, knots: int = 40) -> np.ndarray:
    """The sizes of iNEXT's default curve: ``knots`` sizes from 1 to ``endpoint``.

    ``endpoint`` defaults to twice the ``n`` cells of the sample. Up to the
    sample's size, about half the knots rarefy evenly from 1 to ``n - 1``,
    then the sample itself, then about half extrapolate evenly from ``n + 1``
    to the endpoint; an endpoint at most ``n`` rarefies only.
    """
    if isinstance(knots, bool) or not isinstance(knots, int) or knots < 2:
        raise ValueError("knots must be an integer of at least 2")
    if endpoint is None:
        endpoint = 2 * n
    if isinstance(endpoint, bool) or not isinstance(endpoint, int) or endpoint < 1:
        raise ValueError("endpoint must be a positive integer")
    if endpoint <= n:
        sizes = np.floor(np.linspace(1, endpoint, knots))
    else:
        rarefied = (
            np.floor(np.linspace(1, n - 1, knots // 2 - 1)) if n > 1 else np.array([])
        )
        extrapolated = np.floor(np.linspace(n + 1, endpoint, knots // 2))
        sizes = np.concatenate([rarefied, [n], extrapolated])
    return sorted_unique(sizes.astype(np.int64))


def _gini(clones: CloneSizes) -> float:
    """The Gini coefficient of the clone sizes, 0 when all are equal."""
    if clones.richness == 0:
        return float("nan")
    sizes = np.repeat(clones.values, clones.counts).astype(np.float64)
    ranks = np.arange(1, len(sizes) + 1)
    total = sizes.sum()
    return float(
        2 * np.sum(ranks * sizes) / (len(sizes) * total) - (len(sizes) + 1) / len(sizes)
    )


def _d50(clones: CloneSizes) -> int:
    """The fewest clones, largest first, that hold at least half the cells."""
    sizes = np.repeat(clones.values[::-1], clones.counts[::-1])
    return int(np.searchsorted(np.cumsum(sizes), clones.n / 2, side="left") + 1)


#: The columns of the table of indices, in order.
INDEX_COLUMNS = (
    "cells",
    "clones",
    "singletons",
    "doubletons",
    "coverage",
    "shannon",
    "pielou",
    "simpson",
    "inverse_simpson",
    "gini_simpson",
    "gini",
    "d50",
    "chao1",
    "chao1_bias_corrected",
    "ace",
)
# The indices that count cells or clones.
_COUNT_INDICES = frozenset({"cells", "clones", "singletons", "doubletons", "d50"})


def indices(clones: CloneSizes) -> dict[str, float]:
    """The observed diversity indices and richness estimates of a group.

    Returns:
        ``cells``, ``clones``, ``singletons``, ``doubletons``,
        ``coverage``; ``shannon`` (entropy in nats), ``pielou`` (Shannon
        entropy over its maximum, the log of the clones; NaN for one
        clone), ``simpson`` (sum of squared frequencies),
        ``inverse_simpson``, ``gini_simpson`` (one minus Simpson), ``gini``
        (the Gini coefficient of the clone sizes), ``d50`` (the fewest
        largest clones that hold half the cells), ``chao1``,
        ``chao1_bias_corrected`` and ``ace``.
    """
    p, counts = clones.frequencies()
    simpson = float(np.sum(counts * p**2))
    shannon = _shannon(clones)
    return {
        "cells": clones.n,
        "clones": clones.richness,
        "singletons": clones.f1,
        "doubletons": clones.f2,
        "coverage": coverage(clones),
        "shannon": shannon,
        "pielou": shannon / np.log(clones.richness)
        if clones.richness > 1
        else float("nan"),
        "simpson": simpson,
        "inverse_simpson": 1 / simpson,
        "gini_simpson": 1 - simpson,
        "gini": _gini(clones),
        "d50": _d50(clones),
        "chao1": chao1(clones),
        "chao1_bias_corrected": chao1_bias_corrected(clones),
        "ace": ace(clones),
    }


@dataclass(frozen=True, slots=True)
class ClonalDiversity:
    """The diversity of the clones of groups of cells.

    The groups are the combinations of the ``group_by`` columns' labels and
    the ``groups`` artifact's labels; each table names a group in ``group``,
    its labels joined by commas, and holds its labels in the key columns.

    Attributes:
        indices: One row per group: ``group``, the key columns and the
            columns of :func:`indices`.
        hill: One row per group and order 0, 1 and 2: ``group``, the key
            columns, ``order``, ``observed`` and ``estimated`` (asymptotic)
            Hill numbers, the Hill number ``at_size`` of the common size, with
            its ``coverage_at_size`` and ``method_at_size``, and the Hill
            number ``at_coverage`` of the common coverage, with its
            ``size_at_coverage`` and ``method_at_coverage``.
        clones: The clone artifact, such as a ``clones`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
        size: The common size in cells.
        coverage: The common coverage.
    """

    indices: pd.DataFrame = field(repr=False)
    hill: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
    size: int
    coverage: float


@dataclass(frozen=True, slots=True)
class Rarefaction:
    """The rarefaction and extrapolation curves of groups of cells.

    Attributes:
        curves: One row per group, order and size: ``group`` (its labels
            joined by commas), the key columns, ``order``, ``size`` (cells),
            ``method`` (``"rarefaction"``, ``"observed"`` or
            ``"extrapolation"``), ``diversity`` (the Hill number) and
            ``coverage``.
        clones: The clone artifact, such as a ``clones`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
        orders: The orders of the curves.
    """

    curves: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
    orders: tuple[int, ...]


def common_size(groups: Sequence[CloneSizes]) -> int:
    """iNEXT's default common size: the smallest of the groups' doubled sizes."""
    return int(min(2 * clones.n for clones in groups))


def common_coverage(groups: Sequence[CloneSizes]) -> float:
    """iNEXT's default common coverage: the lowest coverage at doubled sizes."""
    return float(min(coverage_at(clones, 2 * clones.n) for clones in groups))


def _method_names(codes: np.ndarray) -> np.ndarray:
    """Name each method code."""
    return np.asarray(np.asarray(_METHODS, dtype=object)[codes])


def _keyed(keys: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    """The rows of ``keys`` at ``positions``, numbered from 0."""
    return keys.take(positions).reset_index(drop=True)


def diversity_tables(
    groups: Sequence[CloneSizes],
    keys: pd.DataFrame,
    *,
    size: int | None = None,
    coverage: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    """The indices and Hill numbers of groups, the tables of :class:`ClonalDiversity`.

    Each group's Hill numbers of orders 0, 1 and 2 are observed, estimated,
    and standardized to a common size and a common coverage, as iNEXT's
    ``estimateD`` does.

    Args:
        groups: Each group's clone sizes; every group holds a cell.
        keys: One row per group, in the order of ``groups``: ``group``, its
            name, and its label in each key column.
        size: The common size in cells, or None for :func:`common_size`.
        coverage: The common coverage, between 0 and 1, or None for
            :func:`common_coverage`.

    Returns:
        The table of indices, the table of Hill numbers, the common size and
        the common coverage.

    Raises:
        ValueError: If ``coverage`` is not above 0.
    """
    common = common_size(groups) if size is None else int(size)
    level = common_coverage(groups) if coverage is None else float(coverage)
    n_groups, n_orders = len(groups), len(ORDERS)
    shape = (n_groups, n_orders)
    observed, estimated = np.empty(shape), np.empty(shape)
    at_size, at_coverage = np.empty(shape), np.empty(shape)
    size_methods = np.empty(shape, dtype=np.int8)
    coverage_methods = np.empty(shape, dtype=np.int8)
    coverage_at_size = np.empty(n_groups)
    size_at_coverage = np.empty(n_groups)
    rows = []
    for position, clones in enumerate(groups):
        rows.append(indices(clones))
        # The common coverage is 1 only when every group's coverage is 1,
        # which the group's observed size reaches.
        needed = size_for_coverage(clones, level) if level < 1 else float(clones.n)
        samples = _Samples(clones, np.asarray([common, needed], dtype=np.float64))
        best = [estimated_hill(clones, order) for order in ORDERS]
        shannon = _shannon_terms(clones, best[1]) if samples.any_above else None
        for at, order in enumerate(ORDERS):
            observed[position, at] = observed_hill(clones, order)
            estimated[position, at] = best[at]
            at_size[position, at], at_coverage[position, at] = samples.hill(
                order, shannon, observed[position, at]
            )
        size_methods[position], coverage_methods[position] = samples.methods
        coverage_at_size[position] = samples.coverage()[0]
        size_at_coverage[position] = needed
    table = _keyed(keys, np.arange(n_groups))
    for column in INDEX_COLUMNS:
        table[column] = np.asarray(
            [row[column] for row in rows],
            dtype=np.int64 if column in _COUNT_INDICES else np.float64,
        )
    hill = _keyed(keys, np.repeat(np.arange(n_groups), n_orders))
    hill["order"] = np.tile(np.asarray(ORDERS, dtype=np.int64), n_groups)
    hill["observed"] = observed.reshape(-1)
    hill["estimated"] = estimated.reshape(-1)
    hill["at_size"] = at_size.reshape(-1)
    hill["coverage_at_size"] = np.repeat(coverage_at_size, n_orders)
    hill["method_at_size"] = _method_names(size_methods.reshape(-1))
    hill["at_coverage"] = at_coverage.reshape(-1)
    hill["size_at_coverage"] = np.repeat(size_at_coverage, n_orders)
    hill["method_at_coverage"] = _method_names(coverage_methods.reshape(-1))
    return table, hill, common, level


def rarefaction_curves(
    groups: Sequence[CloneSizes],
    keys: pd.DataFrame,
    orders: Sequence[int] = ORDERS,
    *,
    sizes: Sequence[int] | None = None,
    endpoint: int | None = None,
    knots: int = 40,
) -> pd.DataFrame:
    """The rarefaction and extrapolation curves of groups, as :class:`Rarefaction` holds them.

    Args:
        groups: Each group's clone sizes; every group holds a cell.
        keys: One row per group, in the order of ``groups``: ``group``, its
            name, and its label in each key column.
        orders: The orders of the Hill numbers, from 0, 1 and 2.
        sizes: Positive sizes in cells for every group, or None for each
            group's :func:`default_sizes`.
        endpoint: The largest default size, or None for twice each group's
            size.
        knots: The number of default sizes.

    Returns:
        One row per group, order and size: ``group``, the key columns,
        ``order``, ``size``, ``method``, ``diversity`` and ``coverage``.
    """
    chosen = tuple(orders)
    for order in chosen:
        _check_order(order)
    fixed = None if sizes is None else sorted_unique(np.asarray(sizes, dtype=np.int64))
    curve_sizes = [
        default_sizes(clones.n, endpoint, knots) if fixed is None else fixed
        for clones in groups
    ]
    lengths = np.asarray([len(points) for points in curve_sizes], dtype=np.int64)
    ends = np.cumsum(lengths * len(chosen))
    total = int(ends[-1]) if len(ends) else 0
    order_column = np.empty(total, dtype=np.int64)
    size_column = np.empty(total, dtype=np.int64)
    methods = np.empty(total, dtype=np.int8)
    diversity = np.empty(total)
    covered = np.empty(total)
    for clones, points, end in zip(groups, curve_sizes, ends.tolist(), strict=True):
        samples = _Samples(clones, points.astype(np.float64))
        sample_coverage = samples.coverage()
        shannon = _shannon_terms(clones) if 1 in chosen and samples.any_above else None
        start = end - len(points) * len(chosen)
        for order in chosen:
            stop = start + len(points)
            order_column[start:stop] = order
            size_column[start:stop] = points
            methods[start:stop] = samples.methods
            diversity[start:stop] = samples.hill(order, shannon)
            covered[start:stop] = sample_coverage
            start = stop
    table = _keyed(keys, np.repeat(np.arange(len(groups)), lengths * len(chosen)))
    table["order"] = order_column
    table["size"] = size_column
    table["method"] = _method_names(methods)
    table["diversity"] = diversity
    table["coverage"] = covered
    return table
