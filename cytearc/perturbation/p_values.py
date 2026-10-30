"""Turn the statistics of random sets of cells into calibrated p-values.

Both association tests, against the control cells and against all other cells,
compare each pair's statistic with the statistics of random sets of cells. A
first round of sets gives each pair's p-value. A pair whose p-value is then at
or below a threshold takes a second, larger round, and a skew-normal
distribution fitted to that round's statistics by their moments gives p-values
far below the reciprocal of the number of sets, unless its tail is thinner than
the statistics' own; a third round gives the p-value then. The p-values of each
set of pairs are adjusted by Benjamini-Hochberg.

These functions hold statistics in memory and never touch storage.
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "RESAMPLING",
    "STATISTICS_BYTES",
    "PairResults",
    "adjusted_p_values",
    "fit_skew_normals",
    "resampling_p_values",
    "skew_normal_p_values",
    "skew_normal_tail_ratios",
    "staged_p_values",
]

#: The ways of drawing random sets: each cell with its fitted probability, the
#: conditional randomization test, or as many cells as the group uniformly.
RESAMPLING = ("conditional", "permutation")
#: The memory that the second round's statistics may hold at once, by default.
STATISTICS_BYTES = 1 << 28


# Above this skewness the moment equations of the skew-normal have no
# solution; its largest skewness is about 0.9953.
_SKEWNESS_LIMIT = 0.995
# sceptre's fits take this skewness, with its sign, for a sample beyond the
# skew-normal's reach.
_SKEWNESS_FALLBACK = 0.9 * _SKEWNESS_LIMIT
# A fit whose tail at the 99th percentile holds less than this share of the
# statistics' own gives way to the third round.
_TAIL_RATIO_FLOOR = 0.5
# The second round's statistics that its counts, fits and checks work on at
# once, in whole rows: a working copy that stays in a core's L2 cache through
# its passes.
_SPAN_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class PairResults:
    """The test of each pair of a group and a gene.

    Attributes:
        z: The standardized score statistic; positive when the treatment
            cells express the gene more than the fit predicts.
        p_value: The p-value of the side tested.
        stage: The round whose random sets gave the p-value: 1, 2 with a
            skew-normal fit or without the approximation, or 3 when the fit
            failed its check.
        skew_location: The location of the skew-normal fit that gave the
            p-value, NaN when the random sets gave it.
        skew_scale: The scale of that fit.
        skew_shape: The shape of that fit.
        log2_fold_change: log2 of the treatment cells' counts over their
            means under the pair's fit.
        se_log2_fold_change: The standard error of ``log2_fold_change``.
    """

    z: np.ndarray
    p_value: np.ndarray
    stage: np.ndarray
    skew_location: np.ndarray
    skew_scale: np.ndarray
    skew_shape: np.ndarray
    log2_fold_change: np.ndarray
    se_log2_fold_change: np.ndarray


def resampling_p_values(counts: np.ndarray, n_draws: int, side: str) -> np.ndarray:
    """Return the p-values that the random sets give.

    With ``n_draws`` sets, the left p-value is (1 + sets at most the observed
    statistic) / (n_draws + 1) and the right one counts sets at least the
    observed statistic; ``"both"`` doubles the smaller, up to 1.

    Args:
        counts: Per pair, the sets at most and at least the observed
            statistic, from :func:`resampling_counts`.
        n_draws: The number of sets.
        side: ``"both"``, ``"left"`` or ``"right"``.

    Returns:
        Each pair's p-value.
    """
    left = (1.0 + counts[:, 0]) / (n_draws + 1.0)
    right = (1.0 + counts[:, 1]) / (n_draws + 1.0)
    return _sided(left, right, side)


def _sided(left: np.ndarray, right: np.ndarray, side: str) -> np.ndarray:
    """Return the p-value of ``side`` from the two one-sided p-values."""
    if side == "left":
        return left
    if side == "right":
        return right
    both: np.ndarray = np.minimum(1.0, 2.0 * np.minimum(left, right))
    return both


def _span_rows(n_columns: int) -> int:
    """Return how many rows of ``n_columns`` float64 values a span holds, at least 1."""
    return max(1, _SPAN_BYTES // (8 * max(n_columns, 1)))


def _row_spans(values: np.ndarray) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield spans of a matrix's rows and their values as C-contiguous float64.

    The rows of a C-contiguous float64 matrix are views, not copies.
    """
    step = _span_rows(values.shape[1])
    for begin in range(0, values.shape[0], step):
        rows = slice(begin, begin + step)
        yield rows, np.ascontiguousarray(values[rows], dtype=np.float64)


def _working_copy(values: np.ndarray) -> np.ndarray:
    """Return room for one span of a matrix's rows, uninitialized."""
    n_rows, n_columns = values.shape
    return np.empty((min(_span_rows(n_columns), n_rows), n_columns))


def _row_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each row's mean and its second and third central moments, divisor n.

    Each span of rows is centered in one working copy and squared, then
    centered again and cubed, so the moments take one span beyond the values.
    """
    n_rows = values.shape[0]
    mean = np.empty(n_rows)
    second = np.empty(n_rows)
    third = np.empty(n_rows)
    work = _working_copy(values)
    for rows, block in _row_spans(values):
        centered = work[: len(block)]
        middle = block.mean(axis=1, keepdims=True)
        mean[rows] = middle[:, 0]
        np.subtract(block, middle, out=centered)
        np.square(centered, out=centered)
        second[rows] = centered.mean(axis=1)
        np.subtract(block, middle, out=centered)
        np.power(centered, 3, out=centered)
        third[rows] = centered.mean(axis=1)
    return mean, second, third


def _select_ranks(ordered: np.ndarray, ranks: list[int]) -> None:
    """Partially sort each row in place so that it holds its sorted values at ``ranks``.

    ``ranks`` are sorted and distinct. NumPy selects one rank much faster than
    several at once, so the rows are split at the middle rank, and each side
    then selects the ranks it holds.
    """
    if ranks:
        middle = len(ranks) // 2
        rank = ranks[middle]
        ordered.partition(rank, axis=1)
        _select_ranks(ordered[:, :rank], ranks[:middle])
        _select_ranks(
            ordered[:, rank + 1 :], [other - rank - 1 for other in ranks[middle + 1 :]]
        )


def _order_statistics(values: np.ndarray, ranks: list[int]) -> np.ndarray:
    """Return the values at ``ranks`` of each row sorted, rows x ranks.

    A partial sort of one working copy of a span of rows at a time finds the
    same values as a full sort, NaN last, in one span beyond the values.
    """
    found = np.empty((values.shape[0], len(ranks)))
    work = _working_copy(values)
    for rows, block in _row_spans(values):
        ordered = work[: len(block)]
        ordered[...] = block
        _select_ranks(ordered, sorted(set(ranks)))
        found[rows] = ordered[:, ranks]
    return found


def _counts_beyond(statistics: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return each row's statistics at most and at least its observed one, rows x 2.

    The rows are compared a span at a time.
    """
    counts = np.empty((len(observed), 2), dtype=np.int64)
    for rows, block in _row_spans(statistics):
        point = observed[rows, None]
        counts[rows, 0] = (block <= point).sum(axis=1)
        counts[rows, 1] = (block >= point).sum(axis=1)
    return counts


def fit_skew_normals(
    statistics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a skew-normal distribution to each row of statistics by its moments.

    From the mean m, the standard deviation s with divisor n and the skewness
    g, the fit takes |delta| = sqrt(pi/2 |g|^(2/3) / (|g|^(2/3) + ((4 -
    pi)/2)^(2/3))) with the sign of g, shape alpha = delta / sqrt(1 -
    delta^2), scale omega = s / sqrt(1 - 2 delta^2 / pi) and location
    xi = m - omega delta sqrt(2 / pi). A skewness beyond 0.995 in size, more
    than a skew-normal reaches, becomes 0.9 times 0.995 with its sign, as in
    sceptre's fits.

    The moments are taken a span of rows at a time, so besides a few values
    per row the fit holds one working copy of about 1 MiB of the statistics,
    or of one row when a row is larger, never more than the statistics
    themselves; statistics that are not C-contiguous float64 take a float64
    copy of each span too.

    Args:
        statistics: Statistics, pairs x sets.

    Returns:
        Each row's location, scale and shape; a row without spread or with a
        missing statistic gets NaN.
    """
    values = np.asarray(statistics)
    n_rows = values.shape[0]
    location = np.full(n_rows, np.nan)
    scale = np.full(n_rows, np.nan)
    shape = np.full(n_rows, np.nan)
    mean, second, third = _row_moments(values)
    usable = np.isfinite(second) & (second > 0)
    if not usable.any():
        return location, scale, shape
    deviation = np.sqrt(second[usable])
    skewness = third[usable] / deviation**3
    skewness = np.where(
        np.abs(skewness) > _SKEWNESS_LIMIT,
        np.sign(skewness) * _SKEWNESS_FALLBACK,
        skewness,
    )
    power = np.abs(skewness) ** (2.0 / 3.0)
    delta = np.sign(skewness) * np.sqrt(
        np.pi / 2.0 * power / (power + ((4.0 - np.pi) / 2.0) ** (2.0 / 3.0))
    )
    fitted_scale = deviation / np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    location[usable] = mean[usable] - fitted_scale * delta * np.sqrt(2.0 / np.pi)
    scale[usable] = fitted_scale
    shape[usable] = delta / np.sqrt(1.0 - delta**2)
    return location, scale, shape


def skew_normal_tail_ratios(
    statistics: np.ndarray,
    z: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    shape: np.ndarray,
    side: str,
) -> np.ndarray:
    """Return how each fitted tail compares with its statistics' own.

    On the side tested, or with ``side="both"`` the side of the statistics'
    median that ``z`` lies on, the fitted tail probability beyond the
    statistics' 99th percentile, or 1st for the left side, is divided by the
    share of the statistics beyond it. Below 1, the fitted tail is thinner
    there than the statistics', and the p-values it gives farther out are too
    small.

    The percentiles and the median are the statistics at those positions of
    each row sorted, found by a partial sort of a span of rows at a time, so
    besides a few values per row the check holds one working copy of about
    1 MiB of the statistics, or of one row when a row is larger, never more
    than the statistics themselves; statistics that are not C-contiguous
    float64 take a float64 copy of each span too.

    Args:
        statistics: Statistics, pairs x sets.
        z: Each pair's observed statistic.
        location: Each pair's fitted location.
        scale: Each pair's fitted scale.
        shape: Each pair's fitted shape.
        side: ``"both"``, ``"left"`` or ``"right"``.

    Returns:
        Each pair's ratio, NaN without a fit.
    """
    values = np.asarray(statistics)
    n_draws = values.shape[1]
    observed = np.asarray(z, dtype=np.float64)
    median, upper, lower = _order_statistics(
        values,
        [n_draws // 2, math.ceil(0.99 * n_draws) - 1, math.floor(0.01 * n_draws)],
    ).T
    above = np.empty(len(upper))
    below = np.empty(len(lower))
    for rows, block in _row_spans(values):
        above[rows] = (block >= upper[rows, None]).mean(axis=1)
        below[rows] = (block <= lower[rows, None]).mean(axis=1)
    if side == "both":
        right = observed >= median
    else:
        right = np.full(len(observed), side == "right")
    point = np.where(right, upper, lower)
    beyond = np.where(right, above, below)
    standardized = (point - location) / scale
    fitted = np.where(
        right,
        stats.skewnorm.sf(standardized, shape),
        stats.skewnorm.cdf(standardized, shape),
    )
    ratios: np.ndarray = fitted / beyond
    return ratios


def skew_normal_p_values(
    z: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    shape: np.ndarray,
    side: str,
) -> np.ndarray:
    """Return the p-values of statistics under fitted skew-normal distributions.

    The tails are computed in log space, so they stay accurate far beyond the
    random sets.

    Args:
        z: Each pair's statistic.
        location: Each pair's fitted location.
        scale: Each pair's fitted scale.
        shape: Each pair's fitted shape.
        side: ``"both"``, ``"left"`` or ``"right"``.

    Returns:
        Each pair's p-value.
    """
    standardized = (np.asarray(z, dtype=np.float64) - location) / scale
    left = np.exp(stats.skewnorm.logcdf(standardized, shape))
    right = np.exp(stats.skewnorm.logsf(standardized, shape))
    return _sided(left, right, side)


def staged_p_values(
    z: np.ndarray,
    counts: np.ndarray,
    second_statistics: Callable[[np.ndarray, int], np.ndarray],
    third_counts: Callable[[np.ndarray, int], np.ndarray],
    *,
    n_resamples: tuple[int, int, int],
    stage_threshold: float,
    approximation: str,
    side: str,
    statistics_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return each pair's p-value from up to three rounds of random sets.

    The first round's counts give each pair's p-value. A pair whose p-value
    is then at most ``stage_threshold`` is compared with ``n_resamples[1]``
    fresh sets, and with ``approximation="skew_normal"`` a skew-normal fit to
    their statistics gives the p-value, unless its tail beyond their 99th
    percentile, on the observed statistic's side, holds less than half of the
    statistics' own share there; then ``n_resamples[2]`` fresh sets give the
    p-value. With ``approximation="none"``, the second round's sets give it.
    The second round takes its pairs in batches whose statistics fit
    ``statistics_bytes``. Both tests of this package, against control cells
    and against all other cells, take their p-values this way.

    Args:
        z: Each pair's statistic, NaN for a pair without one.
        counts: Per pair, the first round's sets with a statistic at most and
            at least ``z``, pairs x 2.
        second_statistics: Returns the statistics of the second round's sets
            of the pairs it is given, pairs x sets, from their positions and
            the number of sets.
        third_counts: Returns the third round's counts of the pairs it is
            given, pairs x 2, likewise.
        n_resamples: The random sets of the three rounds.
        stage_threshold: The p-value at or below which a pair takes the
            second round.
        approximation: ``"skew_normal"`` or ``"none"``.
        side: ``"both"``, ``"left"`` or ``"right"``.
        statistics_bytes: The memory that the second round's statistics may
            hold at once, at least one pair's.

    Returns:
        Each pair's p-value, NaN without a statistic; the round that gave it;
        and the location, scale and shape of the skew-normal fit that gave
        it, NaN where random sets gave it.
    """
    first_draws, second_draws, third_draws = n_resamples
    p_value = resampling_p_values(counts, first_draws, side)
    p_value[np.isnan(z)] = np.nan
    n_pairs = len(z)
    stage = np.ones(n_pairs, dtype=np.int8)
    location = np.full(n_pairs, np.nan)
    scale = np.full(n_pairs, np.nan)
    shape = np.full(n_pairs, np.nan)
    second = np.flatnonzero(p_value <= stage_threshold)
    batch = max(1, statistics_bytes // (8 * second_draws))
    third: list[np.ndarray] = []
    for begin in range(0, len(second), batch):
        chosen = second[begin : begin + batch]
        statistics = second_statistics(chosen, second_draws)
        empirical = _counts_beyond(statistics, z[chosen])
        p_value[chosen] = resampling_p_values(empirical, second_draws, side)
        stage[chosen] = 2
        if approximation == "skew_normal":
            fit_location, fit_scale, fit_shape = fit_skew_normals(statistics)
            ratios = skew_normal_tail_ratios(
                statistics, z[chosen], fit_location, fit_scale, fit_shape, side
            )
            good = ratios >= _TAIL_RATIO_FLOOR
            p_value[chosen[good]] = skew_normal_p_values(
                z[chosen][good],
                fit_location[good],
                fit_scale[good],
                fit_shape[good],
                side,
            )
            location[chosen[good]] = fit_location[good]
            scale[chosen[good]] = fit_scale[good]
            shape[chosen[good]] = fit_shape[good]
            third.append(chosen[~good])
    if third:
        poor = np.concatenate(third)
        if len(poor):
            counted = third_counts(poor, third_draws)
            p_value[poor] = resampling_p_values(counted, third_draws, side)
            stage[poor] = 3
    return p_value, stage, location, scale, shape


def adjusted_p_values(
    p_values: np.ndarray, tested: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return Benjamini-Hochberg adjusted p-values and the pairs significant at ``alpha``.

    Args:
        p_values: Each pair's p-value.
        tested: Whether each pair was tested; the others get NaN and are not
            significant.
        alpha: The false discovery rate.

    Returns:
        The adjusted p-values and whether each is at most ``alpha``.
    """
    adjusted = np.full(len(p_values), np.nan)
    chosen = np.flatnonzero(np.asarray(tested, dtype=bool) & ~np.isnan(p_values))
    if len(chosen):
        adjusted[chosen] = stats.false_discovery_control(
            np.asarray(p_values, dtype=np.float64)[chosen], method="bh"
        )
    significant = np.zeros(len(p_values), dtype=bool)
    significant[chosen] = adjusted[chosen] <= alpha
    return adjusted, significant
