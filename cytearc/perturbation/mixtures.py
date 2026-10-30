"""Mixture models that set the UMI threshold of each guide.

A guide is called in a cell when the cell's UMIs of the guide reach the guide's
threshold in the cell's sample. Two methods fit a two-component mixture to the
UMI counts of each guide in each sample, one component for the ambient
molecules that most cells hold and one for the cells that carry the guide, and
set the threshold from the fitted parameters:

- ``poisson_gaussian`` follows Replogle et al. 2022 as crispat and pertpy fit
  it. It models x = log2(count) of the cells with at least one UMI of the guide
  as a Poisson background, extended to non-integer x through the Gamma
  function and left unnormalised, plus a Gaussian. The parameters are maximum
  a posteriori estimates under crispat's priors, found by expectation
  maximisation from 15 starting points as pertpy does, and the admissible start
  with the lowest loss wins. The threshold is the smallest count, from 1 to the
  guide's largest, whose Gaussian posterior exceeds one half.
- ``gaussian`` follows Cell Ranger's documentation. It fits two Gaussians with
  one shared variance to x = log10(count + 1) of every cell, including cells
  with no UMI of the guide, by expectation maximisation from several splits of
  the values into two groups, and the fit with the highest likelihood wins.
  The threshold is the smallest count whose higher-mean component has a
  posterior above one half.

Every fit reads a histogram from
:func:`~cytearc.perturbation.histograms.guide_count_histograms`, which makes it
equal to the fit on every cell. Each guide in each sample gets a status from
:data:`FIT_STATUSES`, and a guide whose status is not ``"fitted"`` calls
nothing. The Poisson-Gaussian fit adapts pertpy's MIT-licensed implementation,
as ``THIRD_PARTY_NOTICES.md`` records.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.special import gammaln

from cytearc.perturbation.histograms import GuideCountHistograms

__all__ = [
    "FIT_STATUSES",
    "GUIDE_FIT_FITTED",
    "GUIDE_FIT_NO_CELLS",
    "GUIDE_FIT_NO_FIT",
    "GUIDE_FIT_NO_POSITIVE",
    "GUIDE_FIT_TOO_SPARSE",
    "MAX_ITERATIONS",
    "GaussianFits",
    "PoissonGaussianFits",
    "fit_gaussian",
    "fit_poisson_gaussian",
    "gaussian_posterior",
    "poisson_gaussian_posterior",
]

#: Fit statuses, stored as positions in this tuple.
FIT_STATUSES = ("fitted", "no_cells", "too_sparse", "no_fit", "no_positive")
#: The fit is admissible and calls the guide's largest count.
GUIDE_FIT_FITTED = 0
#: No cell of the sample has a UMI of the guide.
GUIDE_FIT_NO_CELLS = 1
#: The guide has too few cells or counts to fit: for ``poisson_gaussian``
#: fewer than 2 cells with a UMI of the guide or a largest count below 2, and
#: for ``gaussian`` a single distinct count.
GUIDE_FIT_TOO_SPARSE = 2
#: No starting point led to an admissible fit.
GUIDE_FIT_NO_FIT = 3
#: The fit is admissible but calls no count up to the guide's largest.
GUIDE_FIT_NO_POSITIVE = 4
#: The most EM iterations of one start, as in pertpy.
MAX_ITERATIONS = 1000

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# Poisson-Gaussian settings, as in pertpy.
_TOLERANCE = 1e-7
_SCALE_FLOOR = 1e-3
_WEIGHT_FLOOR = 1e-6
_NEWTON_STEPS = 12
# Each fit runs from 15 starts: the 10 prior starts below and 5 starts that
# split each guide's values at a quantile of its cells, Poisson below and
# Gaussian above.
_SPLIT_QUANTILES = np.array([0.5, 0.75, 0.9, 0.97, 0.99])
# Each prior start holds the medians of 15 draws from each prior, as crispat
# starts Pyro's optimiser with init_to_median, as (Poisson rate, Gaussian mean,
# Gaussian standard deviation) with Gaussian weight 0.5. They were drawn once
# with numpy.random.default_rng(2024) and are fixed, so every guide starts from
# the same points and its fit depends on its own counts only.
_PRIOR_STARTS = np.array(
    [
        (1.896025, 4.129306, 7.340522),
        (0.649694, 2.663801, 7.998683),
        (0.678659, 2.624041, 2.419733),
        (1.483196, 1.854345, 5.034174),
        (0.636920, 3.612308, 6.188498),
        (1.156050, 3.764632, 9.558820),
        (0.951677, 2.993959, 6.724229),
        (0.977971, 3.352989, 10.926269),
        (0.955492, 3.002857, 6.436256),
        (1.096023, 3.238447, 6.989896),
    ]
)

# Gaussian settings. The variance gets the same small floor that scikit-learn
# adds, so a component on one distinct value keeps a positive variance.
_GAUSSIAN_TOLERANCE = 1e-10
_VARIANCE_FLOOR = 1e-6
# Shares of the cells, counted from the largest count down, that start the
# higher component, besides the best two-group split and the widest gap. In a
# library of n guides about 1 / n of the cells carry each guide, so the shares
# cover libraries from a few guides to genome scale.
_GAUSSIAN_TAILS = np.array([0.5, 0.1, 0.03, 0.01, 0.003, 0.001, 1e-4, 1e-5])


@dataclass(frozen=True, slots=True)
class PoissonGaussianFits:
    """Poisson-Gaussian fits of every guide in every sample.

    Every array is samples x guides. The parameters are NaN where no admissible
    fit exists.

    Attributes:
        status: The position of each fit's status in :data:`FIT_STATUSES`.
        threshold: The smallest called UMI count, 0 where the guide calls
            nothing.
        poisson_rate: The rate of the Poisson background on log2 counts.
        gaussian_mean: The mean of the Gaussian on log2 counts.
        gaussian_std: The standard deviation of the Gaussian.
        gaussian_weight: The weight of the Gaussian, the share of cells it
            explains.
        loss: The negative log joint density at the estimate, without the
            constant terms of the priors.
        iterations: The EM iterations of the winning start, at most 1,000.
    """

    status: np.ndarray
    threshold: np.ndarray
    poisson_rate: np.ndarray
    gaussian_mean: np.ndarray
    gaussian_std: np.ndarray
    gaussian_weight: np.ndarray
    loss: np.ndarray
    iterations: np.ndarray


@dataclass(frozen=True, slots=True)
class GaussianFits:
    """Two-Gaussian fits of every guide in every sample.

    Every array is samples x guides. The parameters are NaN where no admissible
    fit exists.

    Attributes:
        status: The position of each fit's status in :data:`FIT_STATUSES`.
        threshold: The smallest called UMI count, 0 where the guide calls
            nothing.
        low_mean: The mean of the lower component on log10(count + 1).
        high_mean: The mean of the higher component.
        std: The standard deviation the two components share.
        high_weight: The weight of the higher component.
        log_likelihood: The mean log likelihood per cell at the estimate.
        iterations: The EM iterations of the fit, at most 1,000.
    """

    status: np.ndarray
    threshold: np.ndarray
    low_mean: np.ndarray
    high_mean: np.ndarray
    std: np.ndarray
    high_weight: np.ndarray
    log_likelihood: np.ndarray
    iterations: np.ndarray


@njit(cache=True, nogil=True, inline="always")
def _logaddexp(a: float, b: float) -> float:
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


@njit(cache=True, nogil=True)
def _poisson_gaussian_loss(
    values: np.ndarray,
    cells: np.ndarray,
    log_gamma: np.ndarray,
    rate: float,
    mean: float,
    std: float,
    weight: float,
    gauss: np.ndarray,
) -> float:
    """Return the negative log joint and write each value's Gaussian posterior."""
    log_rate = math.log(rate)
    log_std = math.log(std)
    log_weight = math.log(weight)
    log_rest = math.log1p(-weight)
    total = 0.0
    for index in range(values.shape[0]):
        value = values[index]
        z = (value - mean) / std
        normal = log_weight - _LOG_SQRT_2PI - log_std - 0.5 * z * z
        poisson = log_rest + value * log_rate - rate - log_gamma[index]
        mixture = _logaddexp(normal, poisson)
        total += cells[index] * mixture
        gauss[index] = math.exp(normal - mixture)
    prior = (
        -0.1 * log_rest
        - 0.9 * log_weight
        - 0.5 * ((mean - 3.0) / 2.0) ** 2
        - log_std
        - 0.5 * (log_std - 2.0) ** 2
        - log_rate
        - 0.5 * log_rate**2
    )
    return -(total + prior)


@njit(cache=True, nogil=True)
def _solve_log_std(sum_squares: float, n_gauss: float) -> float:
    """Return the MAP log standard deviation given the Gaussian's cells.

    It is the root of a convex decreasing function, found by Newton steps.
    """
    log_floor = math.log(_SCALE_FLOOR)
    value = 0.5 * math.log(max(sum_squares, 1e-12) / max(n_gauss, 1e-12))
    value = min(max(value, log_floor), 10.0)
    for _ in range(_NEWTON_STEPS):
        scaled = sum_squares * math.exp(-2.0 * value)
        step = (scaled - (value - 2.0) - n_gauss - 1.0) / (-2.0 * scaled - 1.0)
        value = min(max(value - step, log_floor), 10.0)
    return value


@njit(cache=True, nogil=True)
def _solve_log_rate(sum_values: float, n_poisson: float) -> float:
    """Return the MAP log Poisson rate given the background's cells.

    It is the root of a concave decreasing function, found by Newton steps.
    """
    value = math.log(max(sum_values, 1e-12) / max(n_poisson, 1e-12))
    value = min(max(value, -20.0), 10.0)
    for _ in range(_NEWTON_STEPS):
        scaled = n_poisson * math.exp(value)
        step = (sum_values - 1.0 - value - scaled) / (-1.0 - scaled)
        value = min(max(value - step, -20.0), 10.0)
    return value


@njit(cache=True, nogil=True)
def _poisson_gaussian_update(
    values: np.ndarray, cells: np.ndarray, gauss: np.ndarray, std: float
) -> tuple[float, float, float, float]:
    """Return the parameters of one MAP-EM step from the Gaussian posteriors."""
    n_gauss = 0.0
    n_poisson = 0.0
    sum_gauss = 0.0
    sum_poisson = 0.0
    for index in range(values.shape[0]):
        in_gauss = cells[index] * gauss[index]
        rest = cells[index] - in_gauss
        n_gauss += in_gauss
        n_poisson += rest
        sum_gauss += in_gauss * values[index]
        sum_poisson += rest * values[index]
    # Dirichlet(0.9, 0.1) over (Poisson, Gaussian) puts mass 0.1 - 1 on the
    # Gaussian, and N(3, 2) on the mean adds 3 / 4 and 1 / 4.
    weight = (n_gauss - 0.9) / (n_gauss + n_poisson - 1.0)
    weight = min(max(weight, _WEIGHT_FLOOR), 1.0 - _WEIGHT_FLOOR)
    precision = 1.0 / (std * std)
    mean = (sum_gauss * precision + 0.75) / (n_gauss * precision + 0.25)
    sum_squares = 0.0
    for index in range(values.shape[0]):
        deviation = values[index] - mean
        sum_squares += cells[index] * gauss[index] * deviation * deviation
    std = max(math.exp(_solve_log_std(sum_squares, n_gauss)), _SCALE_FLOOR)
    rate = math.exp(_solve_log_rate(sum_poisson, n_poisson))
    return rate, mean, std, weight


@njit(cache=True, nogil=True)
def _poisson_gaussian_run(
    values: np.ndarray,
    cells: np.ndarray,
    log_gamma: np.ndarray,
    gauss: np.ndarray,
    rate: float,
    mean: float,
    std: float,
    weight: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[float, float, float, float, float, int]:
    """Run EM from one start until the loss stops changing."""
    previous = math.inf
    current = 1e30
    iterations = 0
    while iterations < max_iterations and abs(previous - current) > tolerance * abs(
        current
    ):
        loss = _poisson_gaussian_loss(
            values, cells, log_gamma, rate, mean, std, weight, gauss
        )
        rate, mean, std, weight = _poisson_gaussian_update(values, cells, gauss, std)
        previous = current
        current = loss
        iterations += 1
    loss = _poisson_gaussian_loss(
        values, cells, log_gamma, rate, mean, std, weight, gauss
    )
    return rate, mean, std, weight, loss, iterations


@njit(cache=True, nogil=True)
def _split_start(
    values: np.ndarray, cells: np.ndarray, quantile: float
) -> tuple[float, float, float, float]:
    """Return a start that splits the values at a quantile of the cells."""
    total = 0.0
    for index in range(values.shape[0]):
        total += cells[index]
    total = max(total, 1.0)
    n_upper = 0.0
    n_lower = 0.0
    sum_upper = 0.0
    sum_lower = 0.0
    cumulative = 0.0
    for index in range(values.shape[0]):
        cumulative += cells[index]
        if cumulative / total > quantile and cells[index] > 0:
            n_upper += cells[index]
            sum_upper += cells[index] * values[index]
        else:
            n_lower += cells[index]
            sum_lower += cells[index] * values[index]
    rate = max(sum_lower / max(n_lower, 1.0), 0.05)
    mean = sum_upper / n_upper if n_upper > 0 else values[values.shape[0] - 1]
    variance = 0.0
    cumulative = 0.0
    for index in range(values.shape[0]):
        cumulative += cells[index]
        if cumulative / total > quantile and cells[index] > 0:
            variance += cells[index] * (values[index] - mean) ** 2
    std = max(math.sqrt(variance / max(n_upper, 1.0)), 0.5)
    weight = min(max(n_upper / total, 0.01), 0.9)
    return rate, mean, std, weight


@njit(cache=True, nogil=True)
def _poisson_gaussian_group(
    counts: np.ndarray,
    cells: np.ndarray,
    prior_starts: np.ndarray,
    quantiles: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[float, float, float, float, float, int]:
    """Fit one guide in one sample from every start and keep the best admissible fit."""
    size = counts.shape[0]
    values = np.empty(size)
    weights = np.empty(size)
    log_gamma = np.empty(size)
    gauss = np.empty(size)
    for index in range(size):
        values[index] = math.log2(counts[index])
        weights[index] = cells[index]
        log_gamma[index] = math.lgamma(values[index] + 1.0)
    best = (math.nan, math.nan, math.nan, math.nan, math.inf, 0)
    n_prior = prior_starts.shape[0]
    for start in range(n_prior + quantiles.shape[0]):
        if start < n_prior:
            rate = prior_starts[start, 0]
            mean = prior_starts[start, 1]
            std = prior_starts[start, 2]
            weight = 0.5
        else:
            rate, mean, std, weight = _split_start(
                values, weights, quantiles[start - n_prior]
            )
        fit = _poisson_gaussian_run(
            values,
            weights,
            log_gamma,
            gauss,
            rate,
            mean,
            std,
            weight,
            max_iterations,
            tolerance,
        )
        rate, mean, std, weight, loss = fit[0], fit[1], fit[2], fit[3], fit[4]
        # Label-switched optima, with the Gaussian below the Poisson, and
        # optima pinned at the weight or scale floor are inadmissible.
        admissible = (
            math.isfinite(loss)
            and mean > rate
            and weight > 2 * _WEIGHT_FLOOR
            and std > 2 * _SCALE_FLOOR
        )
        if admissible and loss < best[4]:
            best = fit
    return best


@njit(cache=True, nogil=True, parallel=True)
def _fit_poisson_gaussian_groups(
    offsets: np.ndarray,
    counts: np.ndarray,
    cells: np.ndarray,
    fit: np.ndarray,
    prior_starts: np.ndarray,
    quantiles: np.ndarray,
    max_iterations: int,
    tolerance: float,
    out: np.ndarray,
) -> None:
    """Fit every selected histogram; ``out`` rows get rate, mean, std, weight, loss, iterations."""
    for entry in prange(fit.shape[0]):
        if not fit[entry]:
            continue
        start = offsets[entry]
        stop = offsets[entry + 1]
        best = _poisson_gaussian_group(
            counts[start:stop],
            cells[start:stop],
            prior_starts,
            quantiles,
            max_iterations,
            tolerance,
        )
        out[entry, 0] = best[0]
        out[entry, 1] = best[1]
        out[entry, 2] = best[2]
        out[entry, 3] = best[3]
        out[entry, 4] = best[4]
        out[entry, 5] = best[5]


@njit(cache=True, nogil=True)
def _poisson_gaussian_threshold(
    rate: float, mean: float, std: float, weight: float, max_count: int
) -> int:
    """Return the smallest count up to ``max_count`` with Gaussian posterior above one half, or 0."""
    log_rate = math.log(rate)
    log_std = math.log(std)
    log_weight = math.log(weight)
    log_rest = math.log1p(-weight)
    for count in range(1, max_count + 1):
        value = math.log2(count)
        z = (value - mean) / std
        normal = log_weight - _LOG_SQRT_2PI - log_std - 0.5 * z * z
        poisson = log_rest + value * log_rate - rate - math.lgamma(value + 1.0)
        if normal > poisson:
            return count
    return 0


@njit(cache=True, nogil=True, parallel=True)
def _poisson_gaussian_thresholds(
    parameters: np.ndarray, fitted: np.ndarray, max_counts: np.ndarray, out: np.ndarray
) -> None:
    for entry in prange(fitted.shape[0]):
        if fitted[entry]:
            out[entry] = _poisson_gaussian_threshold(
                parameters[entry, 0],
                parameters[entry, 1],
                parameters[entry, 2],
                parameters[entry, 3],
                max_counts[entry],
            )


def fit_poisson_gaussian(histograms: GuideCountHistograms) -> PoissonGaussianFits:
    """Fit the Poisson-Gaussian mixture to every guide in every sample.

    Args:
        histograms: The UMI count histograms of every guide in every sample.

    Returns:
        The status, threshold and parameters of every fit, samples x guides.
    """
    shape = (histograms.n_samples, histograms.n_guides)
    nonzero = histograms.nonzero_cells().reshape(-1)
    max_counts = histograms.max_counts().reshape(-1)
    fit = (nonzero >= 2) & (max_counts >= 2)
    status = np.where(
        nonzero == 0,
        GUIDE_FIT_NO_CELLS,
        np.where(fit, GUIDE_FIT_FITTED, GUIDE_FIT_TOO_SPARSE),
    )
    out = np.full((len(fit), 6), np.nan)
    out[:, 5] = 0
    _fit_poisson_gaussian_groups(
        histograms.offsets,
        histograms.counts,
        histograms.cells,
        fit,
        _PRIOR_STARTS,
        _SPLIT_QUANTILES,
        MAX_ITERATIONS,
        _TOLERANCE,
        out,
    )
    admissible = fit & np.isfinite(out[:, 4])
    status[fit & ~admissible] = GUIDE_FIT_NO_FIT
    out[~admissible] = np.nan
    out[~admissible, 5] = 0
    threshold = np.zeros(len(fit), dtype=np.int64)
    _poisson_gaussian_thresholds(out, admissible, max_counts, threshold)
    status[admissible & (threshold == 0)] = GUIDE_FIT_NO_POSITIVE
    return PoissonGaussianFits(
        status=status.astype(np.int8).reshape(shape),
        threshold=threshold.reshape(shape),
        poisson_rate=out[:, 0].reshape(shape),
        gaussian_mean=out[:, 1].reshape(shape),
        gaussian_std=out[:, 2].reshape(shape),
        gaussian_weight=out[:, 3].reshape(shape),
        loss=out[:, 4].reshape(shape),
        iterations=out[:, 5].astype(np.int32).reshape(shape),
    )


@njit(cache=True, nogil=True)
def _add_split(splits: np.ndarray, n_splits: int, split: int) -> int:
    """Append ``split`` unless it is already among the first ``n_splits``."""
    for index in range(n_splits):
        if splits[index] == split:
            return n_splits
    splits[n_splits] = split
    return n_splits + 1


@njit(cache=True, nogil=True)
def _gaussian_splits(
    values: np.ndarray, weights: np.ndarray, total: float, tails: np.ndarray
) -> np.ndarray:
    """Return the distinct starting splits of sorted values.

    Split ``k`` starts the lower component on ``values[: k + 1]`` and the higher
    on the rest. The starts are the split with the smallest within-group sum of
    squares, which solves two-means exactly, the split at the widest gap
    between neighbouring values, and the splits that give the higher component
    the top ``tails`` share of the cells.
    """
    size = values.shape[0]
    splits = np.empty(2 + tails.shape[0], dtype=np.int64)
    total_sum = 0.0
    total_squares = 0.0
    for index in range(size):
        total_sum += weights[index] * values[index]
        total_squares += weights[index] * values[index] ** 2
    best_cost = math.inf
    best = 0
    left = left_sum = left_squares = 0.0
    for index in range(size - 1):
        left += weights[index]
        left_sum += weights[index] * values[index]
        left_squares += weights[index] * values[index] ** 2
        right = total - left
        right_sum = total_sum - left_sum
        cost = (left_squares - left_sum**2 / left) + (
            total_squares - left_squares - right_sum**2 / right
        )
        if cost < best_cost:
            best_cost = cost
            best = index
    n_splits = _add_split(splits, 0, best)
    widest = -1.0
    gap = 0
    for index in range(size - 1):
        width = values[index + 1] - values[index]
        if width > widest:
            widest = width
            gap = index
    n_splits = _add_split(splits, n_splits, gap)
    for tail in tails:
        target = tail * total
        upper = 0.0
        split = 0
        for index in range(size - 1, 0, -1):
            upper += weights[index]
            if upper >= target:
                split = index - 1
                break
        n_splits = _add_split(splits, n_splits, split)
    return splits[:n_splits]


@njit(cache=True, nogil=True)
def _gaussian_em(
    values: np.ndarray,
    weights: np.ndarray,
    total: float,
    split: int,
    max_iterations: int,
    tolerance: float,
    high: np.ndarray,
) -> tuple[float, float, float, float, float, int]:
    """Run EM from one split until the mean log likelihood stops changing.

    Returns the low mean, high mean, variance, high weight, mean log
    likelihood and iterations, with NaN parameters when the fit degenerates.
    """
    size = values.shape[0]
    n_low = sum_low = sum_high = 0.0
    for index in range(size):
        if index <= split:
            n_low += weights[index]
            sum_low += weights[index] * values[index]
        else:
            sum_high += weights[index] * values[index]
    n_high = total - n_low
    low_mean = sum_low / n_low
    high_mean = sum_high / n_high
    squares = 0.0
    for index in range(size):
        mean = low_mean if index <= split else high_mean
        squares += weights[index] * (values[index] - mean) ** 2
    variance = squares / total + _VARIANCE_FLOOR
    high_weight = n_high / total
    previous = -math.inf
    log_likelihood = -math.inf
    iterations = 0
    while iterations < max_iterations:
        log_low = math.log1p(-high_weight)
        log_high = math.log(high_weight)
        half_precision = 0.5 / variance
        log_likelihood = 0.0
        n_high = sum_high = n_low = sum_low = 0.0
        for index in range(size):
            value = values[index]
            a = log_low - (value - low_mean) ** 2 * half_precision
            b = log_high - (value - high_mean) ** 2 * half_precision
            mixture = _logaddexp(a, b)
            share = math.exp(b - mixture)
            high[index] = share
            log_likelihood += weights[index] * mixture
            n_high += weights[index] * share
            sum_high += weights[index] * share * value
            n_low += weights[index] * (1.0 - share)
            sum_low += weights[index] * (1.0 - share) * value
        log_likelihood = log_likelihood / total - 0.5 * math.log(
            2.0 * math.pi * variance
        )
        iterations += 1
        if not (n_high > 0.0 and n_low > 0.0):
            return math.nan, math.nan, math.nan, math.nan, math.nan, iterations
        if abs(log_likelihood - previous) <= tolerance:
            break
        previous = log_likelihood
        low_mean = sum_low / n_low
        high_mean = sum_high / n_high
        high_weight = n_high / total
        squares = 0.0
        for index in range(size):
            share = high[index]
            squares += weights[index] * (
                (1.0 - share) * (values[index] - low_mean) ** 2
                + share * (values[index] - high_mean) ** 2
            )
        variance = squares / total + _VARIANCE_FLOOR
    # The higher component's posterior grows with the value, so each step
    # keeps the higher mean at or above the lower one; equal means degenerate.
    if not (
        math.isfinite(log_likelihood)
        and high_mean > low_mean
        and 0.0 < high_weight < 1.0
    ):
        return math.nan, math.nan, math.nan, math.nan, math.nan, iterations
    return low_mean, high_mean, variance, high_weight, log_likelihood, iterations


@njit(cache=True, nogil=True)
def _gaussian_group(
    counts: np.ndarray,
    cells: np.ndarray,
    n_zero: int,
    tails: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[float, float, float, float, float, int]:
    """Fit two Gaussians with a shared variance to log10(count + 1) of one guide.

    EM runs from every start of :func:`_gaussian_splits`, and the admissible
    fit with the highest likelihood wins, the earlier start on a tie. Returns
    the low mean, high mean, variance, high weight, mean log likelihood and
    iterations, with NaN parameters and a likelihood of minus infinity when no
    start gives an admissible fit.
    """
    lead = 1 if n_zero > 0 else 0
    size = counts.shape[0] + lead
    values = np.empty(size)
    weights = np.empty(size)
    if lead:
        values[0] = 0.0
        weights[0] = n_zero
    for index in range(counts.shape[0]):
        values[lead + index] = math.log10(counts[index] + 1.0)
        weights[lead + index] = cells[index]
    total = 0.0
    for index in range(size):
        total += weights[index]
    high = np.empty(size)
    best = (math.nan, math.nan, math.nan, math.nan, -math.inf, 0)
    for split in _gaussian_splits(values, weights, total, tails):
        fit = _gaussian_em(
            values, weights, total, split, max_iterations, tolerance, high
        )
        if math.isfinite(fit[4]) and fit[4] > best[4]:
            best = fit
    return best


@njit(cache=True, nogil=True, inline="always")
def _gaussian_calls(
    count: int, low_mean: float, high_mean: float, variance: float, high_weight: float
) -> bool:
    """Return whether the higher component's posterior at ``count`` exceeds one half."""
    value = math.log10(count + 1.0)
    half_precision = 0.5 / variance
    low = math.log1p(-high_weight) - (value - low_mean) ** 2 * half_precision
    high = math.log(high_weight) - (value - high_mean) ** 2 * half_precision
    return high > low


@njit(cache=True, nogil=True)
def _gaussian_threshold(
    low_mean: float,
    high_mean: float,
    variance: float,
    high_weight: float,
    max_count: int,
) -> int:
    """Return the smallest called count up to ``max_count``, or 0.

    With a shared variance, the log odds of the higher component grow linearly
    with the value, so the called counts are those from one count upward, and
    a binary search finds it.
    """
    if not _gaussian_calls(max_count, low_mean, high_mean, variance, high_weight):
        return 0
    low = 1
    high = max_count
    while low < high:
        middle = (low + high) // 2
        if _gaussian_calls(middle, low_mean, high_mean, variance, high_weight):
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True, nogil=True, parallel=True)
def _fit_gaussian_groups(
    offsets: np.ndarray,
    counts: np.ndarray,
    cells: np.ndarray,
    zero_cells: np.ndarray,
    fit: np.ndarray,
    tails: np.ndarray,
    max_iterations: int,
    tolerance: float,
    out: np.ndarray,
) -> None:
    """Fit every selected histogram; ``out`` rows get the six fit values and the threshold."""
    for entry in prange(fit.shape[0]):
        if not fit[entry]:
            continue
        start = offsets[entry]
        stop = offsets[entry + 1]
        result = _gaussian_group(
            counts[start:stop],
            cells[start:stop],
            zero_cells[entry],
            tails,
            max_iterations,
            tolerance,
        )
        out[entry, 0] = result[0]
        out[entry, 1] = result[1]
        out[entry, 2] = result[2]
        out[entry, 3] = result[3]
        out[entry, 4] = result[4]
        out[entry, 5] = result[5]
        if math.isfinite(result[4]):
            out[entry, 6] = _gaussian_threshold(
                result[0], result[1], result[2], result[3], counts[stop - 1]
            )


def fit_gaussian(histograms: GuideCountHistograms) -> GaussianFits:
    """Fit two Gaussians with a shared variance to every guide in every sample.

    Args:
        histograms: The UMI count histograms of every guide in every sample.

    Returns:
        The status, threshold and parameters of every fit, samples x guides.
    """
    shape = (histograms.n_samples, histograms.n_guides)
    nonzero = histograms.nonzero_cells().reshape(-1)
    zero = histograms.zero_cells().reshape(-1)
    distinct = histograms.distinct_counts().reshape(-1) + (zero > 0)
    fit = (nonzero > 0) & (distinct >= 2)
    status = np.where(
        nonzero == 0,
        GUIDE_FIT_NO_CELLS,
        np.where(fit, GUIDE_FIT_FITTED, GUIDE_FIT_TOO_SPARSE),
    )
    out = np.full((len(fit), 7), np.nan)
    out[:, 5:] = 0
    _fit_gaussian_groups(
        histograms.offsets,
        histograms.counts,
        histograms.cells,
        zero,
        fit,
        _GAUSSIAN_TAILS,
        MAX_ITERATIONS,
        _GAUSSIAN_TOLERANCE,
        out,
    )
    admissible = fit & np.isfinite(out[:, 4])
    status[fit & ~admissible] = GUIDE_FIT_NO_FIT
    out[fit & ~admissible, :5] = np.nan
    threshold = out[:, 6].astype(np.int64)
    status[admissible & (threshold == 0)] = GUIDE_FIT_NO_POSITIVE
    return GaussianFits(
        status=status.astype(np.int8).reshape(shape),
        threshold=threshold.reshape(shape),
        low_mean=out[:, 0].reshape(shape),
        high_mean=out[:, 1].reshape(shape),
        std=np.sqrt(out[:, 2]).reshape(shape),
        high_weight=out[:, 3].reshape(shape),
        log_likelihood=out[:, 4].reshape(shape),
        iterations=out[:, 5].astype(np.int32).reshape(shape),
    )


def poisson_gaussian_posterior(
    counts: np.ndarray,
    *,
    poisson_rate: float,
    gaussian_mean: float,
    gaussian_std: float,
    gaussian_weight: float,
) -> np.ndarray:
    """Return the Gaussian component's posterior at each nonzero UMI count."""
    values = np.log2(np.asarray(counts, dtype=np.float64))
    normal = (
        np.log(gaussian_weight)
        - _LOG_SQRT_2PI
        - np.log(gaussian_std)
        - 0.5 * ((values - gaussian_mean) / gaussian_std) ** 2
    )
    poisson = (
        np.log1p(-gaussian_weight)
        + values * np.log(poisson_rate)
        - poisson_rate
        - gammaln(values + 1.0)
    )
    posterior: np.ndarray = np.exp(normal - np.logaddexp(normal, poisson))
    return posterior


def gaussian_posterior(
    counts: np.ndarray,
    *,
    low_mean: float,
    high_mean: float,
    std: float,
    high_weight: float,
) -> np.ndarray:
    """Return the higher component's posterior at each UMI count."""
    values = np.log10(np.asarray(counts, dtype=np.float64) + 1.0)
    half_precision = 0.5 / std**2
    low = np.log1p(-high_weight) - (values - low_mean) ** 2 * half_precision
    high = np.log(high_weight) - (values - high_mean) ** 2 * half_precision
    posterior: np.ndarray = np.exp(high - np.logaddexp(low, high))
    return posterior
