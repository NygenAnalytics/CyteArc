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
cells.

Many groups are measured at once, from one long table of each group's clone
sizes and how many clones have each size: the indices are sums over its
rows, and compiled loops rarefy the samples of many groups in one call.
Groups with equal tables get equal measures, so the diversity of many groups
measures each distinct table once. These functions work on clone sizes and
never touch storage.
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
from .groups import checked_group_names
from .groups import pair_counts
from ..utils.arguments import quoted_value

__all__ = [
    "LARGEST_SIZE",
    "ORDERS",
    "RARE_THRESHOLD",
    "CloneSizes",
    "ClonalDiversity",
    "GroupedCloneSizes",
    "Rarefaction",
    "ace",
    "chao1",
    "chao1_bias_corrected",
    "checked_curve_arguments",
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
#: The largest sample size of a curve, 2**53 cells: sizes are computed as
#: floats, which hold every whole number up to it exactly.
LARGEST_SIZE = 2**53
#: Clones of at most this many cells are rare for ACE (Chao and Lee 1992).
RARE_THRESHOLD = 10
# The series of the entropy estimator is summed until the terms it leaves out
# are below this share of its sum.
_TAIL_TOLERANCE = 1e-18
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
        # Sorting and comparing neighbors counts each size, as np.unique
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


def _integer_sums(starts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each group's exact sum of an integer value of its rows.

    ``starts`` holds where each group's rows start, and the number of rows
    last.
    """
    totals = np.zeros(len(values) + 1, dtype=np.int64)
    np.cumsum(values, out=totals[1:])
    return np.asarray(totals[starts[1:]] - totals[starts[:-1]], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class GroupedCloneSizes:
    """Groups' clone sizes, as one long table of group, size and count.

    Each row holds a group, a clone size, and how many of the group's clones
    have that size. A group's rows are consecutive, with its sizes ascending,
    so they are its frequency table as :class:`CloneSizes` holds it; a group
    without a clone has no rows and no cells.

    Attributes:
        starts: Where each group's rows start, and the number of rows last.
        groups: The group of each row.
        values: The clone size of each row.
        counts: The clones of each row's size.
        n: Each group's cells, the sum of its clone sizes.
        richness: Each group's clones.
        f1: Each group's clones of one cell, singletons.
        f2: Each group's clones of two cells, doubletons.
    """

    starts: np.ndarray = field(repr=False)
    groups: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    counts: np.ndarray = field(repr=False)
    n: np.ndarray = field(repr=False)
    richness: np.ndarray = field(repr=False)
    f1: np.ndarray = field(repr=False)
    f2: np.ndarray = field(repr=False)

    @classmethod
    def from_clones(cls, groups: Any, sizes: Any, n_groups: int) -> "GroupedCloneSizes":
        """Summarize the clone sizes of groups from each clone's group and size.

        Args:
            groups: Each clone's group, from 0 to ``n_groups - 1``.
            sizes: Each clone's cells; clones of no cells are left out.
            n_groups: The number of groups.

        Raises:
            ValueError: If ``groups`` and ``sizes`` differ in length, a group
                is not between 0 and ``n_groups - 1``, or a size is negative.
        """
        codes = np.asarray(groups, dtype=np.int64).reshape(-1)
        cells = np.asarray(sizes, dtype=np.int64).reshape(-1)
        if codes.shape != cells.shape:
            raise ValueError("groups and sizes must hold one value per clone")
        if len(codes) and (codes.min() < 0 or codes.max() >= n_groups):
            raise ValueError(f"groups must be between 0 and {n_groups - 1}")
        if (cells < 0).any():
            raise ValueError("Clone sizes must not be negative")
        if not cells.all():
            kept = cells > 0
            codes, cells = codes[kept], cells[kept]
        # Each distinct pair of group and size is a row, sorted by group and
        # size, with its number of clones.
        row_groups, values, counts = pair_counts(codes, cells)
        starts = np.searchsorted(row_groups, np.arange(n_groups + 1))
        return _grouped(starts, values, counts)

    @classmethod
    def from_groups(cls, groups: Sequence[CloneSizes]) -> "GroupedCloneSizes":
        """Join the clone sizes of groups into one table, in their order."""
        lengths = np.fromiter(
            (len(clones.values) for clones in groups),
            dtype=np.int64,
            count=len(groups),
        )
        starts = np.zeros(len(groups) + 1, dtype=np.int64)
        np.cumsum(lengths, out=starts[1:])
        if not len(groups):
            return _grouped(starts, np.empty(0, np.int64), np.empty(0, np.int64))
        return _grouped(
            starts,
            np.concatenate([clones.values for clones in groups], dtype=np.int64),
            np.concatenate([clones.counts for clones in groups], dtype=np.int64),
        )

    def __len__(self) -> int:
        """The number of groups."""
        return len(self.n)

    def group(self, index: int) -> CloneSizes:
        """The clone sizes of one group.

        Raises:
            ValueError: If ``index`` is not between 0 and the number of groups
                less one.
        """
        if not 0 <= index < len(self):
            raise ValueError(
                f"index must be between 0 and {len(self) - 1}, not {quoted_value(index)}"
            )
        rows = slice(int(self.starts[index]), int(self.starts[index + 1]))
        return CloneSizes(
            values=self.values[rows],
            counts=self.counts[rows],
            n=int(self.n[index]),
            richness=int(self.richness[index]),
            f1=int(self.f1[index]),
            f2=int(self.f2[index]),
        )

    def take(self, positions: Any) -> "GroupedCloneSizes":
        """The table of the groups at ``positions``, numbered from 0 in that order.

        Raises:
            ValueError: If a position is not between 0 and the number of
                groups less one.
        """
        chosen = np.asarray(positions, dtype=np.int64).reshape(-1)
        if len(chosen) and (chosen.min() < 0 or chosen.max() >= len(self)):
            raise ValueError(f"positions must be between 0 and {len(self) - 1}")
        lengths = np.diff(self.starts)[chosen]
        starts = np.zeros(len(chosen) + 1, dtype=np.int64)
        np.cumsum(lengths, out=starts[1:])
        rows = np.repeat(self.starts[chosen] - starts[:-1], lengths) + np.arange(
            starts[-1]
        )
        return _grouped(starts, self.values[rows], self.counts[rows])

    def distinct(self) -> tuple["GroupedCloneSizes", np.ndarray]:
        """The distinct frequency tables of the groups, and the one each group holds.

        Returns:
            The table of the distinct frequency tables, in the order of the
            first group that holds each, and each group's position in it.
        """
        n_groups = len(self.n)
        if n_groups < 2:
            return self, np.arange(n_groups)
        lengths = np.diff(self.starts)
        # Groups fall into classes by their number of rows, and then by the
        # size and count of their rows one position at a time, so two groups
        # share a class only when their tables are equal. The classes made at
        # a position are numbered above every class so far, so they stay apart
        # from those of shorter groups. The groups that reach a position are
        # the first of the groups ordered from the longest.
        classes = lengths.copy()
        top = int(classes.max())
        longest = np.argsort(-lengths, kind="stable")
        reaching = n_groups - np.cumsum(np.bincount(lengths))
        for position in range(int(lengths.max(initial=0))):
            active = longest[: reaching[position]]
            rows = self.starts[active] + position
            keys = (self.counts[rows], self.values[rows], classes[active])
            order = np.lexsort(keys)
            new = np.zeros(len(order), dtype=bool)
            new[0] = True
            for key in keys:
                ordered = key[order]
                new[1:] |= ordered[1:] != ordered[:-1]
            classes[active[order]] = top + np.cumsum(new)
            top += int(new.sum())
        # Each class's first group gives its table its place.
        order = np.argsort(classes, kind="stable")
        ordered = classes[order]
        first = np.ones(n_groups, dtype=bool)
        first[1:] = ordered[1:] != ordered[:-1]
        firsts = order[first]
        if len(firsts) == n_groups:
            return self, np.arange(n_groups)
        places = np.empty(len(firsts), dtype=np.int64)
        places[np.argsort(firsts)] = np.arange(len(firsts))
        held = np.empty(n_groups, dtype=np.int64)
        held[order] = places[np.cumsum(first) - 1]
        return self.take(np.sort(firsts)), held


def _grouped(
    starts: np.ndarray, values: np.ndarray, counts: np.ndarray
) -> GroupedCloneSizes:
    """The table of the groups whose rows start at ``starts``, with their totals."""
    lengths = np.diff(starts)
    return GroupedCloneSizes(
        starts=starts,
        groups=np.repeat(np.arange(len(lengths), dtype=np.int64), lengths),
        values=values,
        counts=counts,
        n=_integer_sums(starts, values * counts),
        richness=_integer_sums(starts, counts),
        f1=_integer_sums(starts, np.where(values == 1, counts, 0)),
        f2=_integer_sums(starts, np.where(values == 2, counts, 0)),
    )


def _single(clones: CloneSizes) -> GroupedCloneSizes:
    """The table of one group."""
    return GroupedCloneSizes.from_groups([clones])


def _sums(table: GroupedCloneSizes, values: np.ndarray) -> np.ndarray:
    """Each group's sum of a value of its rows, 0 without rows.

    The sums carry their rounding errors apart, since the extrapolated Shannon
    diversity divides by the small difference of two diversities near the
    group's size, which magnifies an error in the last place many times.
    """
    return _segment_sums(table.starts, np.ascontiguousarray(values, dtype=np.float64))


def _running(table: GroupedCloneSizes, values: np.ndarray) -> np.ndarray:
    """Each row's sum of an integer value over its group's rows up to it."""
    through = np.cumsum(values)
    return np.asarray(through - (through - values)[table.starts[table.groups]])


def _frequencies(table: GroupedCloneSizes) -> np.ndarray:
    """Each row's clone frequency, its clone size over its group's cells."""
    return np.asarray(table.values / table.n[table.groups], dtype=np.float64)


def _shannon(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's observed Shannon entropy, in nats."""
    p = _frequencies(table)
    return -_sums(table, table.counts * p * np.log(p))


def _simpson(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's Simpson concentration, the sum of its squared clone frequencies."""
    return _sums(table, table.counts * _frequencies(table) ** 2)


def _observed(table: GroupedCloneSizes, order: float) -> np.ndarray:
    """Each group's observed Hill number of a real order, NaN without cells."""
    if order == 1:
        hills = np.exp(_shannon(table))
    else:
        weights = _sums(table, table.counts * _frequencies(table) ** order)
        with np.errstate(divide="ignore"):
            hills = weights ** (1 / (1 - order))
    hills[table.n == 0] = np.nan
    return hills


def observed_hill(clones: CloneSizes, order: float) -> float:
    """The observed Hill number of a real order, the effective number of clones.

    Order 0 is the number of clones, order 1 the exponential of the Shannon
    entropy, and order 2 the inverse Simpson index.
    """
    return float(_observed(_single(clones), order)[0])


def _simpson_pairs(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's Good's unbiased estimate of the Simpson concentration.

    It is NaN for groups of fewer than two cells.
    """
    values, n = table.values, table.n
    pairs = _integer_sums(table.starts, table.counts * values * (values - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = pairs / (n * (n - 1))
    estimate[n < 2] = np.nan
    return np.asarray(estimate, dtype=np.float64)


def _undetected_ratio(table: GroupedCloneSizes) -> np.ndarray:
    """The ratio by which each group's coverage deficit shrinks with each added cell.

    It is ``(n - 1) f1 / ((n - 1) f1 + 2 f2)`` with doubletons, and with
    ``f1 - 1`` and ``f2 = 1`` without them, as in the entropy estimator of
    Chao, Wang and Jost (2013); it is read only for groups with singletons.
    """
    n, f1, f2 = table.n, table.f1, table.f2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            f2 > 0,
            (n - 1) * f1 / ((n - 1) * f1 + 2 * f2),
            (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2),
        )
    return np.asarray(ratio, dtype=np.float64)


def _coverage(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's estimated sample coverage, NaN without cells."""
    n, f1 = table.n, table.f1
    with np.errstate(divide="ignore", invalid="ignore"):
        covered = np.where(f1 == 0, 1.0, 1 - f1 / n * _undetected_ratio(table))
    covered[n == 0] = np.nan
    return covered


def coverage(clones: CloneSizes) -> float:
    """The estimated sample coverage of the group's cells (Chao and Jost 2012).

    The coverage is the share of the population's cells whose clone the
    sample holds; one minus it is the chance that one more cell brings a new
    clone. Without singletons the sample is complete.
    """
    return float(_coverage(_single(clones))[0])


def _undetected(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's estimated clones that its sample missed, Chao1's ``f0``."""
    n, f1, f2 = table.n, table.f1, table.f2
    with np.errstate(divide="ignore", invalid="ignore"):
        missed = np.where(
            f2 > 0,
            (n - 1) / n * f1**2 / (2 * f2),
            (n - 1) / n * f1 * (f1 - 1) / 2,
        )
    missed[n == 0] = 0.0
    return np.asarray(missed, dtype=np.float64)


def _chao1(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's Chao1 estimate of the clones in its population."""
    return np.asarray(table.richness + _undetected(table), dtype=np.float64)


def chao1(clones: CloneSizes) -> float:
    """The Chao1 estimate of the clones in the population (Chao 1984, 1987).

    ``S + (n - 1) / n * f1**2 / (2 f2)``, or ``S + (n - 1) / n * f1 (f1 - 1) /
    2`` without doubletons, as iNEXT and immunarch estimate richness.
    """
    return float(_chao1(_single(clones))[0])


def _chao1_bias_corrected(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's bias-corrected Chao1 estimate."""
    f1, f2 = table.f1, table.f2
    return np.asarray(table.richness + f1 * (f1 - 1) / (2 * (f2 + 1)), dtype=np.float64)


def chao1_bias_corrected(clones: CloneSizes) -> float:
    """The bias-corrected Chao1 estimate (Chao 2005): ``S + f1 (f1 - 1) / (2 (f2 + 1))``.

    It is the form that vegan's ``estimateR`` and scRepertoire's ``chao1``
    report, without the ``(n - 1) / n`` factor.
    """
    return float(_chao1_bias_corrected(_single(clones))[0])


def _ace(table: GroupedCloneSizes, rare: int) -> np.ndarray:
    """Each group's abundance-based coverage estimate of its clones, as :func:`ace`."""
    values, counts, starts = table.values, table.counts, table.starts
    is_rare = values <= rare
    abundant = _integer_sums(starts, np.where(is_rare, 0, counts))
    rare_clones = _integer_sums(starts, np.where(is_rare, counts, 0))
    rare_cells = _integer_sums(starts, np.where(is_rare, values * counts, 0))
    pairs = _integer_sums(
        starts, np.where(is_rare, values * (values - 1) * counts, 0)
    ).astype(np.float64)
    f1 = table.f1
    with np.errstate(divide="ignore", invalid="ignore"):
        rare_coverage = 1 - f1 / rare_cells
        gamma = np.maximum(
            rare_clones / rare_coverage * pairs / (rare_cells * (rare_cells - 1)) - 1,
            0.0,
        )
        estimate = abundant + rare_clones / rare_coverage + f1 / rare_coverage * gamma
        estimate[rare_coverage <= 0] = np.nan
    return np.where(rare_clones == 0, abundant, estimate)


def ace(clones: CloneSizes, rare: int = RARE_THRESHOLD) -> float:
    """The abundance-based coverage estimate of the clones (Chao and Lee 1992).

    Clones of at most ``rare`` cells are rare. The estimate is
    ``S_abundant + S_rare / C + f1 / C * gamma**2``, where ``C = 1 - f1 /
    N_rare`` is the coverage of the rare cells and ``gamma**2`` the squared
    coefficient of variation of the rare clone sizes, at least 0. It is NaN
    when every rare clone is a singleton, since then ``C = 0``.
    """
    return float(_ace(_single(clones), rare)[0])


@njit(cache=True)
def _add(total: float, error: float, term: float) -> tuple[float, float]:
    """Add a term to a sum, carrying the rounding error apart (Neumaier)."""
    added = total + term
    if abs(total) >= abs(term):
        error += (total - added) + term
    else:
        error += (term - added) + total
    return added, error


@njit(cache=True)
def _segment_sums(starts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each group's sum of the values of its rows, which start at ``starts``."""
    sums = np.empty(len(starts) - 1)
    for group in range(len(starts) - 1):
        total, error = 0.0, 0.0
        for row in range(starts[group], starts[group + 1]):
            total, error = _add(total, error, values[row])
        sums[group] = total + error
    return sums


@njit(cache=True)
def _entropy_tail(n: int, a: float) -> float:
    """``(1 - a) ** (1 - n) * (-log(a) - sum((1 - a) ** r / r, r = 1..n-1))``.

    The bracket is the tail of the series of ``-log(a)`` from ``r = n``, so
    the whole equals ``sum((1 - a) ** j / (j + n - 1), j >= 1)``. When the
    factor ``(1 - a) ** (1 - n)`` is moderate the tail is computed as written,
    and otherwise by its series, whose terms shrink by a factor of at most
    ``1 - a`` each, until the terms left out are below a share
    :data:`_TAIL_TOLERANCE` of its sum; either avoids overflow and loss of
    precision. Both sums carry their rounding errors apart.
    """
    step = math.log1p(-a)
    decay = -step
    total = 0.0
    error = 0.0
    if (n - 1) * decay <= 5:
        for r in range(1, n):
            total, error = _add(total, error, math.exp(r * step) / r)
        return math.exp((n - 1) * decay) * (-math.log(a) - (total + error))
    # The terms after a term sum to at most the term over e**decay - 1.
    left = 1 / math.expm1(decay)
    j = 1
    while True:
        term = math.exp(-j * decay) / (j + n - 1)
        total, error = _add(total, error, term)
        if term * left < _TAIL_TOLERANCE * (total + error):
            return total + error
        j += 1


@njit(cache=True)
def _entropy_tails(n: np.ndarray, a: np.ndarray) -> np.ndarray:
    """:func:`_entropy_tail` of each pair of ``n`` and ``a``."""
    tails = np.empty(len(n))
    for at in range(len(n)):
        tails[at] = _entropy_tail(n[at], a[at])
    return tails


def _chao_entropy(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's Shannon entropy estimate, as :func:`chao_entropy`."""
    n, f1, f2 = table.n, table.f1, table.f2
    values = table.values
    row_n = n[table.groups]
    # Clones of fewer than n cells.
    inner = values <= row_n - 1
    first = _sums(
        table,
        np.where(
            inner,
            table.counts * values / row_n * (digamma(row_n) - digamma(values)),
            0.0,
        ),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(
            f2 > 0, 2 * f2 / ((n - 1) * f1 + 2 * f2), 2 / ((n - 1) * (f1 - 1) + 2)
        )
    tail = np.flatnonzero((f1 > 0) & (a < 1))
    first[tail] += f1[tail] / n[tail] * _entropy_tails(n[tail], a[tail])
    first[n == 0] = np.nan
    return first


def chao_entropy(clones: CloneSizes) -> float:
    """The Shannon entropy estimate of Chao, Wang and Jost (2013), in nats.

    The first term sums, over clones of 1 to ``n - 1`` cells, their
    frequency times ``1 / X + ... + 1 / (n - 1)``; the second estimates the
    entropy of the clones the sample missed from the singletons and
    doubletons through ``A = 2 f2 / ((n - 1) f1 + 2 f2)``, or ``2 / ((n - 1)
    (f1 - 1) + 2)`` without doubletons.
    """
    return float(_chao_entropy(_single(clones))[0])


def _estimated(table: GroupedCloneSizes, order: int) -> np.ndarray:
    """Each group's asymptotic Hill number of order 0, 1 or 2, as :func:`estimated_hill`."""
    if order == 0:
        hills = _chao1(table)
    elif order == 1:
        hills = np.exp(_chao_entropy(table))
    else:
        pairs = _simpson_pairs(table)
        with np.errstate(divide="ignore", invalid="ignore"):
            hills = np.where(pairs > 0, 1 / pairs, np.inf)
    hills[table.n == 0] = np.nan
    return hills


def estimated_hill(clones: CloneSizes, order: int) -> float:
    """The asymptotic Hill number of order 0, 1 or 2, as iNEXT's ``AsyEst``.

    Order 0 is Chao1, order 1 the exponential of the entropy estimate of
    Chao, Wang and Jost (2013), and order 2 the inverse of Good's unbiased
    Simpson estimate, infinite when no clone holds two cells.
    """
    _check_order(order)
    return float(_estimated(_single(clones), order)[0])


def _check_order(order: Any) -> None:
    if isinstance(order, bool) or order not in ORDERS:
        raise ValueError("order must be 0, 1 or 2")


@njit(cache=True)
def _rarefied_sums(
    starts: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    n: np.ndarray,
    owners: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The expected clones and coverage of samples of whole sizes below their groups' cells.

    Sample ``s`` draws ``sizes[s]`` of the ``n`` cells of group ``owners[s]``.
    A clone of ``X`` cells is missed by a sample of ``m`` cells with chance
    ``C(n - X, m) / C(n, m)``, the product of ``1 - m / (n - j)`` for ``j``
    below ``X``, summed here as logarithms in order of ``j`` so that the
    chance keeps its precision when it is close to 1; it is 0 when ``X > n -
    m``. The expected clones count each clone by its chance of being drawn,
    and the coverage leaves out each clone's frequency times ``C(n - X, m) /
    C(n - 1, m)``, the chance of missing it over the chance ``1 - m / n`` of
    missing one given cell. The sums over a group's sizes carry their
    rounding errors apart.
    """
    richness = np.empty(len(sizes))
    covered = np.empty(len(sizes))
    for sample in range(len(sizes)):
        group = owners[sample]
        m = sizes[sample]
        cells = n[group]
        given = math.log1p(-m / cells)
        missing = 0.0
        j = 0
        drawn, drawn_error = 0.0, 0.0
        lost, lost_error = 0.0, 0.0
        for row in range(starts[group], starts[group + 1]):
            size = values[row]
            if size > cells - m:
                drawn, drawn_error = _add(drawn, drawn_error, float(counts[row]))
                continue
            while j < size:
                missing += math.log1p(-m / (cells - j))
                j += 1
            drawn, drawn_error = _add(
                drawn, drawn_error, counts[row] * -math.expm1(missing)
            )
            lost, lost_error = _add(
                lost, lost_error, counts[row] * size / cells * math.exp(missing - given)
            )
        richness[sample] = drawn + drawn_error
        covered[sample] = 1 - (lost + lost_error)
    return richness, covered


@njit(cache=True)
def _expected_entropies(
    starts: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    n: np.ndarray,
    owners: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    """The expected Shannon entropy of samples of whole sizes below their groups' cells.

    Sample ``s`` draws ``sizes[s]`` of the ``n`` cells of group ``owners[s]``.
    For each distinct size ``X`` the cells drawn from one such clone follow
    the hypergeometric distribution, whose probabilities are built from the
    ratio of consecutive terms, outward from the mode, and normalized over the
    support, which keeps them to a few units in the last place; terms beyond
    40 standard deviations of the mode, below ``e**-800`` of it, are left out.
    """
    entropies = np.empty(len(sizes))
    logs = np.empty(64)
    for at in range(len(sizes)):
        group = owners[at]
        m = sizes[at]
        cells = n[group]
        total = 0.0
        for index in range(starts[group], starts[group + 1]):
            size = values[index]
            low = max(0, m - (cells - size))
            high = min(size, m)
            mean = m * size / cells
            spread = math.sqrt(
                mean * (1 - size / cells) * (cells - m) / max(cells - 1, 1)
            )
            mode = min(max((m + 1) * (size + 1) // (cells + 2), low), high)
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
                    - math.log(cells - size - m + k)
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


@dataclass(frozen=True, slots=True)
class _Terms:
    """Each group's measures that its rarefied and extrapolated samples read.

    Attributes:
        observed: The observed Hill numbers of orders 0, 1 and 2.
        coverage: The observed sample coverage.
        missed: Chao1's estimate of the clones that the sample missed.
        pairs: Good's Simpson estimate, 0 below two cells.
        ratio: The ratio by which the coverage deficit shrinks with each
            added cell.
        asymptote: The asymptotic Shannon diversity, NaN for groups whose
            Shannon diversity is not extrapolated.
        rate: The rate at which the extrapolated Shannon diversity approaches
            the asymptote, 0 for a flat curve.
    """

    observed: tuple[np.ndarray, ...]
    coverage: np.ndarray
    missed: np.ndarray
    pairs: np.ndarray
    ratio: np.ndarray
    asymptote: np.ndarray
    rate: np.ndarray


def _terms(
    table: GroupedCloneSizes,
    shannon: np.ndarray,
    asymptote: np.ndarray | None = None,
) -> _Terms:
    """The measures that samples of a table's groups read.

    The Shannon diversity extrapolates toward the asymptote geometrically
    from the observed value, at the rate ``(D(n) - D(n - 1)) / (D(inf) - D(n -
    1))`` that makes the curve pass through the rarefied value at ``n - 1``
    cells, or 0, a flat curve, below two cells or when the asymptote does not
    exceed the observed value. The asymptote and the rate are measured only
    for the groups at the positions ``shannon``; ``asymptote`` passes every
    group's asymptotic Shannon diversity, when known.
    """
    observed = tuple(_observed(table, order) for order in ORDERS)
    pairs = _simpson_pairs(table)
    pairs[np.isnan(pairs)] = 0.0
    reached = np.full(len(table), np.nan)
    rate = np.zeros(len(table))
    if len(shannon):
        if asymptote is not None:
            reached[shannon] = asymptote[shannon]
        elif len(shannon) == len(table):
            reached[shannon] = np.exp(_chao_entropy(table))[shannon]
        else:
            reached[shannon] = np.exp(_chao_entropy(table.take(shannon)))
        rising = shannon[
            (table.n[shannon] >= 2) & (reached[shannon] > observed[1][shannon])
        ]
        before = np.exp(
            _expected_entropies(
                table.starts,
                table.values,
                table.counts,
                table.n,
                rising,
                table.n[rising] - 1,
            )
        )
        rate[rising] = (observed[1][rising] - before) / (reached[rising] - before)
    return _Terms(
        observed=observed,
        coverage=_coverage(table),
        missed=_undetected(table),
        pairs=pairs,
        ratio=_undetected_ratio(table),
        asymptote=reached,
        rate=rate,
    )


def _extrapolated(
    table: GroupedCloneSizes,
    owners: np.ndarray,
    sizes: np.ndarray,
    order: int,
    terms: _Terms,
) -> np.ndarray:
    """The expected Hill numbers of samples of more than their groups' cells.

    Order 0 adds Chao1's missed clones as more cells find them; order 2
    extends Good's Simpson estimate; order 1 approaches the asymptotic
    estimate geometrically from the observed value, at the rate that makes
    the curve pass through the rarefied value at ``n - 1`` cells, the unified
    form of Chao et al. (2023) that reduces to the richness formula at order
    0 (Chao et al. 2014, 2023).
    """
    n = table.n[owners]
    extra = sizes - n
    if order == 0:
        missed = terms.missed[owners]
        f1 = table.f1[owners]
        richness = table.richness[owners]
        with np.errstate(divide="ignore", invalid="ignore"):
            grown = richness + missed * (1 - (1 - f1 / (n * missed + f1)) ** extra)
        return np.asarray(np.where(missed == 0, richness, grown), dtype=np.float64)
    if order == 2:
        pairs = terms.pairs[owners]
        return np.asarray(1 / (1 / sizes + (1 - 1 / sizes) * pairs), dtype=np.float64)
    observed = terms.observed[1][owners]
    grown = (terms.asymptote[owners] - observed) * (
        1 - (1 - terms.rate[owners]) ** extra
    )
    return np.asarray(observed + grown, dtype=np.float64)


def _extrapolated_coverage(
    table: GroupedCloneSizes, owners: np.ndarray, sizes: np.ndarray, ratio: np.ndarray
) -> np.ndarray:
    """The expected coverage of samples of more than their groups' cells (Chao and Jost 2012).

    ``ratio`` holds each group's :func:`_undetected_ratio`. A group without
    singletons keeps its complete coverage, and its ratio, which can exceed 1,
    is not raised to a power.
    """
    f1, n = table.f1[owners], table.n[owners]
    covered = np.ones(len(owners))
    some = f1 > 0
    covered[some] = 1 - f1[some] / n[some] * ratio[owners[some]] ** (
        sizes[some] - n[some] + 1
    )
    return covered


def _sample_sizes(sizes: Any) -> np.ndarray:
    """Return sample sizes as floats, each at least one cell."""
    array = np.asarray(sizes, dtype=np.float64).reshape(-1)
    if not np.all(array >= 1):
        raise ValueError("A sample holds at least one cell")
    return array


class _Samples:
    """Samples of some sizes of groups' cells, and how each is reached.

    Sample ``i`` holds ``sizes[i]`` of the cells of group ``owners[i]``. A
    sample smaller than its group's ``n`` cells is rarefied, at ``n``
    observed, and larger extrapolated. A size between two whole numbers below
    ``n`` interpolates linearly between them, as iNEXT does, and above ``n -
    1`` toward the observed value. Each whole size below a group's ``n`` is
    rarefied once, for every order and the coverage.
    """

    def __init__(
        self, table: GroupedCloneSizes, owners: np.ndarray, sizes: np.ndarray
    ) -> None:
        n = table.n[owners]
        self.table = table
        self.owners = owners
        self.sizes = sizes
        self.above = sizes > n
        self.below = sizes < n
        self.methods = np.full(len(sizes), _OBSERVED, dtype=np.int8)
        self.methods[self.above] = _EXTRAPOLATION
        self.methods[self.below] = _RAREFACTION
        self.any_above = bool(self.above.any())
        self.any_below = bool(self.below.any())
        self._drawn: tuple[np.ndarray, np.ndarray] | None = None
        if self.any_below:
            below = owners[self.below]
            self.low = np.floor(sizes[self.below]).astype(np.int64)
            self.fraction = sizes[self.below] - self.low
            self.between = self.fraction > 0
            upper = self.low + 1
            self.inner = self.between & (upper < n[self.below])
            # Each pair of a group and a whole size is rarefied once.
            pair_owners = np.concatenate([below, below[self.inner]])
            pair_sizes = np.concatenate([self.low, upper[self.inner]])
            order = np.lexsort((pair_sizes, pair_owners))
            ordered_owners, ordered_sizes = pair_owners[order], pair_sizes[order]
            first = np.ones(len(order), dtype=bool)
            first[1:] = (ordered_owners[1:] != ordered_owners[:-1]) | (
                ordered_sizes[1:] != ordered_sizes[:-1]
            )
            self.whole_owners = ordered_owners[first]
            self.whole = ordered_sizes[first]
            places = np.empty(len(order), dtype=np.int64)
            places[order] = np.cumsum(first) - 1
            self._low_at = places[: len(self.low)]
            self._upper_at = places[len(self.low) :]

    def _drawn_sums(self) -> tuple[np.ndarray, np.ndarray]:
        """The :func:`_rarefied_sums` of the whole sizes, computed once."""
        if self._drawn is None:
            table = self.table
            self._drawn = _rarefied_sums(
                table.starts,
                table.values,
                table.counts,
                table.n,
                self.whole_owners,
                self.whole,
            )
        return self._drawn

    def _combine(
        self,
        at_n: np.ndarray,
        extrapolate: Callable[[np.ndarray, np.ndarray], np.ndarray],
        rarefy: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Each sample's value from its group's value at its cells, ``at_n``."""
        values = at_n[self.owners]
        if self.any_above:
            values[self.above] = extrapolate(
                self.owners[self.above], self.sizes[self.above]
            )
        if self.any_below:
            reached = rarefy()
            below = reached[self._low_at]
            between = self.between
            if between.any():
                upper = at_n[self.owners[self.below]]
                upper[self.inner] = reached[self._upper_at]
                below[between] += (upper[between] - below[between]) * self.fraction[
                    between
                ]
            values[self.below] = below
        return np.asarray(values, dtype=np.float64)

    def _rarefied(self, order: int, terms: _Terms) -> np.ndarray:
        """The expected Hill numbers of the whole sizes below the groups' cells."""
        if order == 0:
            return self._drawn_sums()[0]
        if order == 2:
            whole, pairs = self.whole, terms.pairs[self.whole_owners]
            return np.asarray(
                1 / (1 / whole + (1 - 1 / whole) * pairs), dtype=np.float64
            )
        table = self.table
        return np.exp(
            _expected_entropies(
                table.starts,
                table.values,
                table.counts,
                table.n,
                self.whole_owners,
                self.whole,
            )
        )

    def hill(self, order: int, terms: _Terms) -> np.ndarray:
        """The Hill numbers of an order."""
        table = self.table
        return self._combine(
            terms.observed[order],
            lambda owners, sizes: _extrapolated(table, owners, sizes, order, terms),
            lambda: self._rarefied(order, terms),
        )

    def coverage(self, terms: _Terms) -> np.ndarray:
        """The expected coverage of each sample."""
        table = self.table
        return self._combine(
            terms.coverage,
            lambda owners, sizes: _extrapolated_coverage(
                table, owners, sizes, terms.ratio
            ),
            lambda: self._drawn_sums()[1],
        )


# No group's Shannon diversity is extrapolated.
_NO_GROUPS = np.empty(0, dtype=np.int64)


def coverage_at(clones: CloneSizes, m: float) -> float:
    """The expected coverage of a sample of ``m`` cells, rarefied or extrapolated.

    A size between two whole numbers below ``n`` interpolates linearly, as
    iNEXT does.
    """
    table = _single(clones)
    samples = _Samples(table, np.zeros(1, dtype=np.int64), _sample_sizes(m))
    return float(samples.coverage(_terms(table, _NO_GROUPS))[0])


def diversity_at(clones: CloneSizes, m: float, order: int) -> tuple[float, str]:
    """The Hill number of a sample of ``m`` cells, and how it was reached.

    Below ``n`` cells the sample is rarefied, interpolating linearly between
    whole numbers of cells as iNEXT does; at ``n`` it is observed; above, it
    is extrapolated. Sizes below 1 are not samples.

    Returns:
        The Hill number and ``"rarefaction"``, ``"observed"`` or
        ``"extrapolation"``.
    """
    _check_order(order)
    values, methods = hill_curve(clones, [m], order)
    return values[0], methods[0]


def _sizes_for_coverage(
    table: GroupedCloneSizes, observed: np.ndarray, level: float
) -> np.ndarray:
    """The cells that samples of each group need to reach a coverage, as iNEXT's ``invChat``.

    ``observed`` holds each group's observed coverage, and every group holds
    a cell. Below the observed coverage the size is found among the rarefied
    sizes and interpolated linearly between whole numbers; above it, the
    extrapolated coverage is inverted. A level below the coverage of one
    cell needs one cell.
    """
    n = table.n
    needed = n.astype(np.float64)
    higher = np.flatnonzero(level > observed)
    if len(higher):
        # Below full coverage there are singletons, and the deficit shrinks by
        # a ratio strictly between 0 and 1 with each added cell.
        ratio = _undetected_ratio(table)[higher]
        needed[higher] = (
            n[higher]
            + np.log((1 - level) * n[higher] / table.f1[higher]) / np.log(ratio)
            - 1
        )
    lower = np.flatnonzero(level < observed)
    if not len(lower):
        return needed
    # The coverage of one cell, rarefied below a group's cells.
    one = observed[lower]
    several = n[lower] > 1
    one[several] = _rarefied_sums(
        table.starts,
        table.values,
        table.counts,
        n,
        lower[several],
        np.ones(int(several.sum()), dtype=np.int64),
    )[1]
    needed[lower[one >= level]] = 1.0
    searched = lower[one < level]
    low = np.ones(len(searched), dtype=np.int64)
    low_value = one[one < level]
    high = n[searched]
    high_value = observed[searched]
    # The rarefied coverage grows with the size: each round narrows the whole
    # sizes of every group around the level, several at a time, to the last
    # below it.
    active = np.flatnonzero(high - low > 1)
    while len(active):
        span = high[active] - low[active]
        dense = span - 1 <= _COVERAGE_PROBES
        n_probes = np.where(dense, span - 1, _COVERAGE_PROBES)
        firsts = np.cumsum(n_probes) - n_probes
        local = np.arange(int(n_probes.sum())) - np.repeat(firsts, n_probes)
        prober = np.repeat(active, n_probes)
        # Probes are every size between the bounds when they are few, or
        # otherwise the 16 sizes of np.linspace(low, high, 18) between them.
        spaced = (
            (local + 1).astype(np.float64)
            * np.repeat(span / (_COVERAGE_PROBES + 1), n_probes)
            + low[prober]
        ).astype(np.int64)
        probes = np.where(np.repeat(dense, n_probes), low[prober] + 1 + local, spaced)
        values = _rarefied_sums(
            table.starts, table.values, table.counts, n, searched[prober], probes
        )[1]
        # Each group's first probe that reaches the level, or its number of
        # probes when none does.
        reached = np.minimum.reduceat(
            np.where(values >= level, local, np.repeat(n_probes, n_probes)), firsts
        )
        moved = reached > 0
        at = firsts[moved] + reached[moved] - 1
        low[active[moved]] = probes[at]
        low_value[active[moved]] = values[at]
        capped = reached < n_probes
        at = firsts[capped] + reached[capped]
        high[active[capped]] = probes[at]
        high_value[active[capped]] = values[at]
        active = active[high[active] - low[active] > 1]
    needed[searched] = low + (level - low_value) / (high_value - low_value)
    return needed


def size_for_coverage(clones: CloneSizes, level: float) -> float:
    """The cells that a sample needs to reach a coverage, as iNEXT's ``invChat``.

    Below the observed coverage the size is found among the rarefied sizes
    and interpolated linearly between whole numbers; above it, the
    extrapolated coverage is inverted. A level below the coverage of one cell
    needs one cell.

    Raises:
        ValueError: If the level is not between 0 and 1, or the group holds
            no cell.
    """
    if not 0 < level < 1:
        raise ValueError("coverage must be between 0 and 1")
    if clones.n == 0:
        raise ValueError("A group without cells reaches no coverage")
    table = _single(clones)
    return float(_sizes_for_coverage(table, _coverage(table), level)[0])


def hill_curve(
    clones: CloneSizes, sizes: Sequence[float], order: int
) -> tuple[list[float], list[str]]:
    """The Hill numbers of samples of each size, as :func:`diversity_at` gives them.

    The sizes are computed together, and the terms that every extrapolated
    size of order 1 shares once.
    """
    _check_order(order)
    table = _single(clones)
    points = _sample_sizes(sizes)
    samples = _Samples(table, np.zeros(len(points), dtype=np.int64), points)
    shannon = np.zeros(int(order == 1 and samples.any_above), dtype=np.int64)
    values = samples.hill(order, _terms(table, shannon))
    return values.tolist(), [_METHODS[code] for code in samples.methods.tolist()]


def checked_curve_arguments(endpoint: Any, knots: Any) -> tuple[int | None, int]:
    """Return the ``endpoint`` and ``knots`` of the default curve sizes, checked.

    Python and NumPy integers are accepted; the endpoint may be None.

    Raises:
        ValueError: If ``knots`` is not an integer of at least 2, or
            ``endpoint`` is not a positive integer up to :data:`LARGEST_SIZE`.
    """
    if isinstance(knots, bool) or not isinstance(knots, int | np.integer) or knots < 2:
        raise ValueError("knots must be an integer of at least 2")
    if endpoint is not None and (
        isinstance(endpoint, bool)
        or not isinstance(endpoint, int | np.integer)
        or not 1 <= endpoint <= LARGEST_SIZE
    ):
        raise ValueError(f"endpoint must be a positive integer up to {LARGEST_SIZE}")
    return (None if endpoint is None else int(endpoint)), int(knots)


def default_sizes(n: int, endpoint: int | None = None, knots: int = 40) -> np.ndarray:
    """The sizes of iNEXT's default curve: ``knots`` sizes from 1 to ``endpoint``.

    ``endpoint`` defaults to twice the ``n`` cells of the sample. Up to the
    sample's size, about half the knots rarefy evenly from 1 to ``n - 1``,
    then the sample itself, then about half extrapolate evenly from ``n + 1``
    to the endpoint; an endpoint at most ``n`` rarefies only.
    """
    endpoint, knots = checked_curve_arguments(endpoint, knots)
    if endpoint is None:
        endpoint = 2 * n
    if endpoint <= n:
        sizes = np.floor(np.linspace(1, endpoint, knots))
    else:
        rarefied = (
            np.floor(np.linspace(1, n - 1, knots // 2 - 1)) if n > 1 else np.array([])
        )
        extrapolated = np.floor(np.linspace(n + 1, endpoint, knots // 2))
        sizes = np.concatenate([rarefied, [n], extrapolated])
    return sorted_unique(sizes.astype(np.int64))


def _gini(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's Gini coefficient of its clone sizes, 0 when all are equal.

    With the sizes ascending, a row's clones take the ranks after the clones
    of the rows before it, so the sum of each size times its rank is a sum
    over rows. It is NaN for a group without clones.
    """
    counts = table.counts
    before = _running(table, counts) - counts
    ranked = table.values * (counts * before + counts * (counts + 1) // 2)
    weighted = _integer_sums(table.starts, ranked)
    richness = table.richness
    with np.errstate(divide="ignore", invalid="ignore"):
        gini = 2 * weighted / (richness * table.n) - (richness + 1) / richness
    gini[richness == 0] = np.nan
    return np.asarray(gini, dtype=np.float64)


def _d50(table: GroupedCloneSizes) -> np.ndarray:
    """Each group's fewest clones, largest first, that hold at least half its cells.

    The clones of a row and of every row after it, the larger clones, hold at
    least half the cells for each row up to some row, which the largest
    clones reach half the cells in; a group without cells has 1.
    """
    n = table.n
    cells = table.values * table.counts
    through = _running(table, cells)
    row_n = n[table.groups]
    reaching = (2 * (row_n - (through - cells)) >= row_n).astype(np.int64)
    d50 = np.ones(len(n), dtype=np.int64)
    held = np.flatnonzero(n > 0)
    rows = table.starts[held] + _integer_sums(table.starts, reaching)[held] - 1
    larger_cells = n[held] - through[rows]
    larger_clones = table.richness[held] - _running(table, table.counts)[rows]
    size = table.values[rows]
    d50[held] = larger_clones + (n[held] - 2 * larger_cells + 2 * size - 1) // (
        2 * size
    )
    return d50


#: The columns of the table of indices, in order.
INDEX_COLUMNS = (
    "n_cells",
    "n_clones",
    "n_singletons",
    "n_doubletons",
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
_COUNT_INDICES = frozenset(
    {"n_cells", "n_clones", "n_singletons", "n_doubletons", "d50"}
)


def _indices(table: GroupedCloneSizes) -> dict[str, np.ndarray]:
    """Each group's column of :func:`indices`; every group holds a cell."""
    shannon = _shannon(table)
    simpson = _simpson(table)
    richness = table.richness
    with np.errstate(divide="ignore", invalid="ignore"):
        pielou = np.where(richness > 1, shannon / np.log(richness), np.nan)
    return {
        "n_cells": table.n,
        "n_clones": richness,
        "n_singletons": table.f1,
        "n_doubletons": table.f2,
        "coverage": _coverage(table),
        "shannon": shannon,
        "pielou": pielou,
        "simpson": simpson,
        "inverse_simpson": 1 / simpson,
        "gini_simpson": 1 - simpson,
        "gini": _gini(table),
        "d50": _d50(table),
        "chao1": _chao1(table),
        "chao1_bias_corrected": _chao1_bias_corrected(table),
        "ace": _ace(table, RARE_THRESHOLD),
    }


def indices(clones: CloneSizes) -> dict[str, float]:
    """The observed diversity indices and richness estimates of a group.

    Returns:
        ``n_cells``, ``n_clones``, ``n_singletons`` (clones of one cell),
        ``n_doubletons`` (clones of two cells), ``coverage``; ``shannon``
        (entropy in nats), ``pielou`` (Shannon
        entropy over its maximum, the log of the clones; NaN for one
        clone), ``simpson`` (sum of squared frequencies),
        ``inverse_simpson``, ``gini_simpson`` (one minus Simpson), ``gini``
        (the Gini coefficient of the clone sizes), ``d50`` (the fewest
        largest clones that hold half the cells), ``chao1``,
        ``chao1_bias_corrected`` and ``ace``.

    Raises:
        ValueError: If the group holds no cell.
    """
    if clones.n == 0:
        raise ValueError("A group without cells has no diversity indices")
    columns = _indices(_single(clones))
    return {name: column[0].item() for name, column in columns.items()}


@dataclass(frozen=True, slots=True)
class ClonalDiversity:
    """The diversity of the clones of groups of cells.

    The groups are the combinations of the ``group_by`` columns' labels, the
    ``groups`` artifact's labels and the labels that the clones keep apart;
    each table names a group in ``group``, its labels joined by commas, and
    ``group_keys`` holds its labels, so ``indices.join(group_keys,
    on="group")`` gives each row its group's labels.

    Attributes:
        indices: One row per group: ``group`` and the columns of
            :func:`indices`.
        hill: One row per group and order 0, 1 and 2: ``group``, ``order``,
            ``observed`` and ``estimated`` (asymptotic) Hill numbers, the Hill
            number ``at_size`` of the common size, with its
            ``coverage_at_size`` and ``method_at_size``, and the Hill number
            ``at_coverage`` of the common coverage, with its
            ``size_at_coverage`` and ``method_at_coverage``.
        group_keys: One row per group, indexed by its name, ``group``, with
            its label in each key column: the ``group_by`` columns, the kind
            of the ``groups`` artifact, then the columns that the clones keep
            apart, under their own names.
        clones: The clone artifact, such as a ``clones`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
        size: The common size in cells.
        coverage: The common coverage.
        diagnostics: The cells that the summary leaves out, by name:
            ``cellsWithoutClone``, the cells without a clone among those
            kept, ``cellsWithoutGroup``, those with a clone but no group,
            and the ``groups``.
    """

    indices: pd.DataFrame = field(repr=False)
    hill: pd.DataFrame = field(repr=False)
    group_keys: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
    size: int
    coverage: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Rarefaction:
    """The rarefaction and extrapolation curves of groups of cells.

    The groups are those of :class:`ClonalDiversity`, and ``group_keys``
    holds their labels.

    Attributes:
        curves: One row per group, order and size: ``group`` (its labels
            joined by commas), ``order``, ``size`` (cells), ``method``
            (``"rarefaction"``, ``"observed"`` or ``"extrapolation"``),
            ``diversity`` (the Hill number) and ``coverage``.
        group_keys: One row per group, indexed by its name, ``group``, with
            its label in each key column, as :class:`ClonalDiversity` holds
            them.
        clones: The clone artifact, such as a ``clones`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
        orders: The orders of the curves.
        diagnostics: The cells that the summary leaves out, by name:
            ``cellsWithoutClone``, the cells without a clone among those
            kept, ``cellsWithoutGroup``, those with a clone but no group,
            and the ``groups``.
    """

    curves: pd.DataFrame = field(repr=False)
    group_keys: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
    orders: tuple[int, ...]
    diagnostics: dict[str, Any]


def common_size(groups: GroupedCloneSizes) -> int:
    """iNEXT's default common size: the smallest of the groups' doubled sizes.

    Raises:
        ValueError: If there is no group.
    """
    if not len(groups):
        raise ValueError("The common size of no group is undefined")
    return int(np.min(2 * groups.n))


def common_coverage(groups: GroupedCloneSizes) -> float:
    """iNEXT's default common coverage: the lowest coverage at doubled sizes.

    Every group holds a cell, so each doubled size is extrapolated.

    Raises:
        ValueError: If there is no group.
    """
    if not len(groups):
        raise ValueError("The common coverage of no group is undefined")
    covered = _extrapolated_coverage(
        groups,
        np.arange(len(groups)),
        2.0 * groups.n,
        _undetected_ratio(groups),
    )
    return float(np.min(covered))


def _method_names(codes: np.ndarray) -> np.ndarray:
    """Name each method code."""
    return np.asarray(np.asarray(_METHODS, dtype=object)[codes])


def _named(names: pd.Index, positions: np.ndarray) -> pd.DataFrame:
    """A table of the groups' names at ``positions``, in ``group``."""
    return pd.DataFrame({"group": names.take(positions)})


def diversity_tables(
    groups: GroupedCloneSizes,
    names: Sequence[str],
    *,
    size: int | None = None,
    coverage: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    """The indices and Hill numbers of groups, the tables of :class:`ClonalDiversity`.

    Each group's Hill numbers of orders 0, 1 and 2 are observed, estimated,
    and standardized to a common size and a common coverage, as iNEXT's
    ``estimateD`` does. Groups with equal frequency tables get equal rows, so
    each distinct table is measured once and its rows go to every group that
    holds it.

    Args:
        groups: The groups' clone sizes; every group holds a cell.
        names: Each group's name, in the order of ``groups``.
        size: The common size in cells, or None for :func:`common_size`.
        coverage: The common coverage, between 0 and 1, or None for
            :func:`common_coverage`.

    Returns:
        The table of indices, the table of Hill numbers, the common size and
        the common coverage.

    Raises:
        ValueError: If a group holds no cell, ``names`` does not hold one
            text name for each group, names two groups alike, ``coverage``
            is not above 0, or there is no group and ``size`` or
            ``coverage`` is None.
    """
    named = checked_group_names(names, len(groups))
    if (groups.n == 0).any():
        raise ValueError("Every group must hold a cell")
    tables, held = groups.distinct()
    common = common_size(tables) if size is None else int(size)
    level = common_coverage(tables) if coverage is None else float(coverage)
    if not level > 0:
        raise ValueError("coverage must be between 0 and 1")
    n_tables, n_orders = len(tables), len(ORDERS)
    columns = _indices(tables)
    # The common coverage is 1 only when every group's coverage is 1, which
    # the group's observed size reaches.
    needed = (
        _sizes_for_coverage(tables, columns["coverage"], level)
        if level < 1
        else tables.n.astype(np.float64)
    )
    # Each table is sampled at the common size and at the size that reaches
    # the common coverage.
    samples = _Samples(
        tables,
        np.repeat(np.arange(n_tables), 2),
        np.column_stack([np.full(n_tables, float(common)), needed]).reshape(-1),
    )
    estimated = np.column_stack([_estimated(tables, order) for order in ORDERS])
    terms = _terms(
        tables,
        np.flatnonzero(samples.above.reshape(n_tables, 2).any(axis=1)),
        asymptote=estimated[:, 1],
    )
    reached = np.stack([samples.hill(order, terms) for order in ORDERS], axis=-1)
    reached = reached.reshape(n_tables, 2, n_orders)
    coverage_at_size = samples.coverage(terms)[::2]
    methods = samples.methods.reshape(n_tables, 2)
    # Each group takes the rows of its table.
    table = _named(named, np.arange(len(groups)))
    for column in INDEX_COLUMNS:
        table[column] = np.asarray(
            columns[column][held],
            dtype=np.int64 if column in _COUNT_INDICES else np.float64,
        )
    hill = _named(named, np.repeat(np.arange(len(groups)), n_orders))
    hill["order"] = np.tile(np.asarray(ORDERS, dtype=np.int64), len(groups))
    hill["observed"] = np.column_stack(terms.observed)[held].reshape(-1)
    hill["estimated"] = estimated[held].reshape(-1)
    hill["at_size"] = reached[held, 0].reshape(-1)
    hill["coverage_at_size"] = np.repeat(coverage_at_size[held], n_orders)
    hill["method_at_size"] = _method_names(np.repeat(methods[held, 0], n_orders))
    hill["at_coverage"] = reached[held, 1].reshape(-1)
    hill["size_at_coverage"] = np.repeat(needed[held], n_orders)
    hill["method_at_coverage"] = _method_names(np.repeat(methods[held, 1], n_orders))
    return table, hill, common, level


def rarefaction_curves(
    groups: GroupedCloneSizes,
    names: Sequence[str],
    orders: Sequence[int] = ORDERS,
    *,
    sizes: Sequence[int] | None = None,
    endpoint: int | None = None,
    knots: int = 40,
) -> pd.DataFrame:
    """The rarefaction and extrapolation curves of groups, as :class:`Rarefaction` holds them.

    Args:
        groups: The groups' clone sizes; every group holds a cell.
        names: Each group's name, in the order of ``groups``.
        orders: The orders of the Hill numbers, from 0, 1 and 2.
        sizes: Positive sizes in cells for every group, or None for each
            group's :func:`default_sizes`.
        endpoint: The largest default size, or None for twice each group's
            size.
        knots: The number of default sizes.

    Returns:
        One row per group, order and size: ``group``, ``order``, ``size``,
        ``method``, ``diversity`` and ``coverage``.

    Raises:
        ValueError: If ``names`` does not hold one text name for each group
            or names two groups alike, or an order is not 0, 1 or 2.
    """
    named = checked_group_names(names, len(groups))
    chosen = tuple(orders)
    for order in chosen:
        _check_order(order)
    fixed = None if sizes is None else sorted_unique(np.asarray(sizes, dtype=np.int64))
    curve_sizes = [
        default_sizes(n, endpoint, knots) if fixed is None else fixed
        for n in groups.n.tolist()
    ]
    lengths = np.asarray([len(points) for points in curve_sizes], dtype=np.int64)
    # The Shannon extrapolation is measured for the groups whose curves of
    # order 1 pass their cells.
    largest = np.asarray(
        [points[-1] if len(points) else 0 for points in curve_sizes], dtype=np.int64
    )
    shannon = np.flatnonzero(largest > groups.n) if 1 in chosen else _NO_GROUPS
    terms = _terms(groups, shannon)
    ends = np.cumsum(lengths * len(chosen))
    total = int(ends[-1]) if len(ends) else 0
    order_column = np.empty(total, dtype=np.int64)
    size_column = np.empty(total, dtype=np.int64)
    methods = np.empty(total, dtype=np.int8)
    diversity = np.empty(total)
    covered = np.empty(total)
    for at, (points, end) in enumerate(zip(curve_sizes, ends.tolist(), strict=True)):
        samples = _Samples(
            groups, np.full(len(points), at, dtype=np.int64), points.astype(np.float64)
        )
        sample_coverage = samples.coverage(terms)
        start = end - len(points) * len(chosen)
        for order in chosen:
            stop = start + len(points)
            order_column[start:stop] = order
            size_column[start:stop] = points
            methods[start:stop] = samples.methods
            diversity[start:stop] = samples.hill(order, terms)
            covered[start:stop] = sample_coverage
            start = stop
    table = _named(named, np.repeat(np.arange(len(groups)), lengths * len(chosen)))
    table["order"] = order_column
    table["size"] = size_column
    table["method"] = _method_names(methods)
    table["diversity"] = diversity
    table["coverage"] = covered
    return table
