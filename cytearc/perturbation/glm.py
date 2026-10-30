"""Fit the regressions of the association tests and score sets of cells under them.

A gene's null model is a negative binomial regression of its counts on the cell
covariates: Poisson coefficients fitted by Newton's method, and the size that
maximizes the likelihood with the Poisson means held fixed. Under that fit,
each cell has a score and a weight, and the standardized score statistic of a
set of cells, the statistic of SCEPTRE (Barry et al. 2024), adds them up. The
fold change of a set of cells compares its counts with their fitted means. A
logistic regression of a group's indicator on the same covariates gives each
cell its probability of belonging to the group, which the conditional
randomization test of high-MOI screens draws its random sets from.

Every kernel works on arrays that the caller allocates, so the engines that
call them choose how to hold and share them.
"""

import math

import numpy as np

from cytearc.utils.numba import njit

__all__ = [
    "cholesky",
    "cholesky_solve",
    "count_array",
    "digamma",
    "fit_logistic",
    "fit_negative_binomial",
    "fit_poisson",
    "forward_substitution",
    "log2_fold_change",
    "log_linear_means",
    "logistic_probabilities",
    "negative_binomial_information",
    "pearson_dispersion",
    "score_statistic",
    "set_test",
    "size_derivatives",
    "size_dispersion",
    "trigamma",
]

# Linear predictors are kept within exp's range; exp(-700) is still above 0.
_LINEAR_LIMIT = 700.0
# A Cholesky pivot this small against its diagonal marks a direction that the
# cells do not determine, such as a covariate that is 0 on all of them.
_PIVOT_TOLERANCE = 1e-10
# Above this log size, about 1.6e15, the negative binomial is Poisson to
# double precision at the means of counts.
_LOG_SIZE_LIMIT = 35.0
_FIT_ITERATIONS = 100
_FIT_TOLERANCE = 1e-12
# Newton's method reaches a logistic fit in a few steps; a direction that
# separates the response takes about one step per factor of e in its
# probabilities, so this many leaves them near 0 or 1.
_LOGISTIC_ITERATIONS = 50
# Counts up to this sum their digamma differences term by term.
_FINITE_SUM_COUNTS = 8


def count_array(values: np.ndarray) -> np.ndarray:
    """Return counts as a C-contiguous float32 array, without a copy if they are one.

    float32 holds every count below 2^24 exactly, at half the memory of
    float64, and with one type each kernel compiles once.
    """
    return np.ascontiguousarray(values, dtype=np.float32)


@njit(cache=True, nogil=True)
def cholesky(matrix: np.ndarray, out: np.ndarray) -> None:
    """Write the lower Cholesky factor of a symmetric positive semidefinite matrix.

    A direction that the matrix does not determine gets an infinite pivot, so
    solves give it 0.
    """
    p = matrix.shape[0]
    for r in range(p):
        for c in range(r + 1):
            total = matrix[r, c]
            for k in range(c):
                total -= out[r, k] * out[c, k]
            if r == c:
                if total > _PIVOT_TOLERANCE * matrix[r, r] and total > 0:
                    out[r, r] = math.sqrt(total)
                else:
                    out[r, r] = np.inf
            else:
                out[r, c] = total / out[c, c]
        for c in range(r + 1, p):
            out[r, c] = 0.0


@njit(cache=True, nogil=True)
def forward_substitution(
    factor: np.ndarray, vector: np.ndarray, out: np.ndarray
) -> None:
    """Solve ``factor @ out = vector`` for a lower triangular factor."""
    for r in range(len(vector)):
        total = vector[r]
        for k in range(r):
            total -= factor[r, k] * out[k]
        out[r] = total / factor[r, r]


@njit(cache=True, nogil=True)
def cholesky_solve(factor: np.ndarray, vector: np.ndarray, out: np.ndarray) -> None:
    """Solve ``factor @ factor.T @ out = vector``."""
    p = len(vector)
    forward_substitution(factor, vector, out)
    for r in range(p - 1, -1, -1):
        total = out[r]
        for k in range(r + 1, p):
            total -= factor[k, r] * out[k]
        out[r] = total / factor[r, r]


@njit(cache=True, nogil=True)
def log_linear_means(
    covariates: np.ndarray, coefficients: np.ndarray, eta: np.ndarray, mu: np.ndarray
) -> None:
    """Write each cell's linear predictor and mean."""
    n, p = covariates.shape
    for i in range(n):
        total = 0.0
        for c in range(p):
            total += covariates[i, c] * coefficients[c]
        total = min(max(total, -_LINEAR_LIMIT), _LINEAR_LIMIT)
        eta[i] = total
        mu[i] = math.exp(total)


@njit(cache=True, nogil=True)
def negative_binomial_information(
    covariates: np.ndarray, mu: np.ndarray, dispersion: float, out: np.ndarray
) -> None:
    """Write the negative binomial information of the coefficients."""
    n, p = covariates.shape
    out[:, :] = 0.0
    for i in range(n):
        weight = mu[i] / (1.0 + dispersion * mu[i])
        for r in range(p):
            weighted = weight * covariates[i, r]
            for c in range(r + 1):
                out[r, c] += weighted * covariates[i, c]
    for r in range(p):
        for c in range(r + 1, p):
            out[r, c] = out[c, r]


@njit(cache=True, nogil=True)
def fit_poisson(
    y: np.ndarray,
    covariates: np.ndarray,
    coefficients: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
) -> None:
    """Fit Poisson coefficients by Newton's method, from ``coefficients``.

    Each step halves until the log-likelihood does not fall, and the fit stops
    when a step raises it by less than the tolerance. ``eta`` and ``mu`` end
    at the fit.
    """
    n, p = covariates.shape
    information = np.empty((p, p))
    gradient = np.empty(p)
    factor = np.empty((p, p))
    step = np.empty(p)
    trial = np.empty(p)
    log_linear_means(covariates, coefficients, eta, mu)
    current = 0.0
    for i in range(n):
        current += y[i] * eta[i] - mu[i]
    for _ in range(_FIT_ITERATIONS):
        negative_binomial_information(covariates, mu, 0.0, information)
        gradient[:] = 0.0
        for i in range(n):
            residual = y[i] - mu[i]
            for c in range(p):
                gradient[c] += residual * covariates[i, c]
        cholesky(information, factor)
        cholesky_solve(factor, gradient, step)
        scale = 1.0
        candidate = current
        for _half in range(60):
            for c in range(p):
                trial[c] = coefficients[c] + scale * step[c]
            log_linear_means(covariates, trial, eta, mu)
            candidate = 0.0
            for i in range(n):
                candidate += y[i] * eta[i] - mu[i]
            if candidate >= current:
                break
            scale *= 0.5
        coefficients[:] = trial
        change = candidate - current
        current = candidate
        if change <= _FIT_TOLERANCE * (abs(current) + 1.0):
            break


@njit(cache=True, nogil=True)
def pearson_dispersion(y: np.ndarray, mu: np.ndarray, degrees: float) -> float:
    """Return the dispersion that solves the Pearson moment equation.

    The sum of (y - mu)^2 / (mu + dispersion mu^2) equals ``degrees``, the
    cells less the coefficients; without overdispersion the dispersion is 0.
    The sum is convex and falling in the dispersion, so Newton's method from 0
    rises to the root without passing it.
    """
    excess = 0.0
    for i in range(len(y)):
        excess += (y[i] - mu[i]) ** 2 / mu[i]
    if excess <= degrees:
        return 0.0
    dispersion = 0.0
    for _ in range(200):
        value = -degrees
        slope = 0.0
        for i in range(len(y)):
            pearson = (y[i] - mu[i]) ** 2 / mu[i]
            scale = 1.0 + dispersion * mu[i]
            value += pearson / scale
            slope -= pearson * mu[i] / (scale * scale)
        following = dispersion - value / slope
        settled = following - dispersion <= 1e-12 * following
        dispersion = following
        if settled:
            break
    return dispersion


@njit(cache=True, nogil=True)
def digamma(x: float) -> float:
    """Return the digamma function of a positive number.

    The recurrence psi(x) = psi(x + 1) - 1 / x carries x to 10 or more, where
    the asymptotic series is accurate to about 1e-15.
    """
    result = 0.0
    while x < 10.0:
        result -= 1.0 / x
        x += 1.0
    inverse = 1.0 / x
    square = inverse * inverse
    series = square * (
        1.0 / 12.0
        - square
        * (
            1.0 / 120.0
            - square
            * (
                1.0 / 252.0
                - square
                * (
                    1.0 / 240.0
                    - square
                    * (1.0 / 132.0 - square * (691.0 / 32760.0 - square / 12.0))
                )
            )
        )
    )
    return result + math.log(x) - 0.5 * inverse - series


@njit(cache=True, nogil=True)
def trigamma(x: float) -> float:
    """Return the trigamma function of a positive number, as :func:`digamma` does."""
    result = 0.0
    while x < 10.0:
        result += 1.0 / (x * x)
        x += 1.0
    inverse = 1.0 / x
    square = inverse * inverse
    series = (
        inverse
        + 0.5 * square
        + inverse
        * square
        * (
            1.0 / 6.0
            - square
            * (
                1.0 / 30.0
                - square
                * (
                    1.0 / 42.0
                    - square
                    * (
                        1.0 / 30.0
                        - square
                        * (5.0 / 66.0 - square * (691.0 / 2730.0 - square * 7.0 / 6.0))
                    )
                )
            )
        )
    )
    return result + series


@njit(cache=True, nogil=True)
def size_derivatives(y: np.ndarray, mu: np.ndarray, size: float) -> tuple[float, float]:
    """Return the first two derivatives of the log-likelihood in log(size).

    The means are held fixed. For an integer count y, psi(size + y) -
    psi(size) is the sum of 1 / (size + k) for k below y.
    """
    first = 0.0
    second = 0.0
    base_first = digamma(size)
    base_second = trigamma(size)
    for i in range(len(y)):
        count = y[i]
        mean = mu[i]
        gap_first = 0.0
        gap_second = 0.0
        if count > 0:
            if count <= _FINITE_SUM_COUNTS and count == math.floor(count):
                for k in range(int(count)):
                    inverse = 1.0 / (size + k)
                    gap_first += inverse
                    gap_second -= inverse * inverse
            else:
                gap_first = digamma(size + count) - base_first
                gap_second = trigamma(size + count) - base_second
        total = size + mean
        first += gap_first + math.log(size / total) + (mean - count) / total
        second += (
            gap_second + 1.0 / size - 1.0 / total - (mean - count) / (total * total)
        )
    return size * first, size * first + size * size * second


@njit(cache=True, nogil=True)
def size_dispersion(y: np.ndarray, mu: np.ndarray, degrees: float) -> float:
    """Return the dispersion, 1 / size, of negative binomial counts with known means.

    The size is the maximum likelihood estimate with the means held fixed,
    found by Newton's method in log(size) from the moment estimate. When the
    counts show no overdispersion that the likelihood can see, the sum of
    (y - mu)^2 - y being at most 0, the likelihood has no finite maximum and
    the dispersion solves the Pearson moment equation instead, or is 0.
    """
    excess = 0.0
    squares = 0.0
    for i in range(len(y)):
        excess += (y[i] - mu[i]) ** 2 - y[i]
        squares += mu[i] * mu[i]
    if excess <= 0.0:
        return pearson_dispersion(y, mu, degrees)
    log_size = math.log(squares / excess)
    for _ in range(100):
        gradient, curvature = size_derivatives(y, mu, math.exp(log_size))
        # Newton's step where the likelihood curves down enough, and otherwise
        # a step of at most 1 uphill.
        step = -gradient / min(curvature, -abs(gradient) - 1e-300)
        step = min(max(step, -3.0), 3.0)
        log_size += step
        if log_size > _LOG_SIZE_LIMIT:
            return 0.0
        if log_size < -_LOG_SIZE_LIMIT:
            # Counts without a positive count: no finite maximum either.
            log_size = -_LOG_SIZE_LIMIT
            break
        if abs(step) < 1e-10:
            break
    return math.exp(-log_size)


@njit(cache=True, nogil=True)
def score_statistic(
    score_sum: float,
    weight_sum: float,
    weighted: np.ndarray,
    factor: np.ndarray,
    solved: np.ndarray,
) -> float:
    """Return the standardized score statistic of one set of cells.

    ``weighted`` sums each cell's weight times its covariates, and ``factor``
    is the Cholesky factor of the information of the fit. A set without
    spread, such as an empty set, has no statistic: NaN.
    """
    forward_substitution(factor, weighted, solved)
    variance = weight_sum
    for c in range(len(solved)):
        variance -= solved[c] * solved[c]
    if variance > 0:
        return score_sum / math.sqrt(variance)
    return np.nan


@njit(cache=True, nogil=True)
def log2_fold_change(
    count_sum: float,
    mean_sum: float,
    square_sum: float,
    tangent: np.ndarray,
    factor: np.ndarray,
    dispersion: float,
    solved: np.ndarray,
) -> tuple[float, float]:
    """Return a set of cells' log2 fold change under a fit and its standard error.

    The fold change is the set's counts over their fitted means, the
    estimator of the SCEPTRE book, and its standard error comes from the delta
    method, with the negative binomial variance of the counts and the
    information of the fit, whose Cholesky factor is ``factor``.

    Args:
        count_sum: The set's counts.
        mean_sum: The set's fitted means.
        square_sum: The set's squared fitted means.
        tangent: The set's means times their covariates, summed; divided by
            ``mean_sum`` in place.
        factor: The Cholesky factor of the fit's information.
        dispersion: The fit's dispersion, 1 / size.
        solved: Room for one covariate vector.

    Returns:
        The log2 fold change, -inf for a set without counts, and its standard
        error.
    """
    p = len(tangent)
    ratio = count_sum / mean_sum
    for c in range(p):
        tangent[c] /= mean_sum
    forward_substitution(factor, tangent, solved)
    variance = 0.0
    for c in range(p):
        variance += solved[c] * solved[c]
    if count_sum > 0:
        spread = count_sum + dispersion * ratio * ratio * square_sum
        variance += spread / (count_sum * count_sum)
    fold = math.log2(ratio) if ratio > 0 else -np.inf
    return fold, math.sqrt(variance) / math.log(2.0)


@njit(cache=True, nogil=True)
def fit_negative_binomial(
    y: np.ndarray,
    covariates: np.ndarray,
    coefficients: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
    factor: np.ndarray,
) -> float:
    """Fit a negative binomial regression of counts under the null, from ``coefficients``.

    The fit takes Poisson coefficients by Newton's method, then the size that
    maximizes the likelihood with the Poisson means held fixed: the null fit
    of a pair in the test against control cells and of a gene in the test
    against all other cells. ``coefficients``, ``eta`` and ``mu`` end at the
    fit, and ``factor`` receives the lower Cholesky factor of its negative
    binomial information. Returns the dispersion, 1 / size.
    """
    n, p = covariates.shape
    fit_poisson(y, covariates, coefficients, eta, mu)
    dispersion = size_dispersion(y, mu, float(n - p))
    information = np.empty((p, p))
    negative_binomial_information(covariates, mu, dispersion, information)
    cholesky(information, factor)
    return dispersion


@njit(cache=True, nogil=True)
def set_test(
    y: np.ndarray,
    mu: np.ndarray,
    covariates: np.ndarray,
    dispersion: float,
    cells: np.ndarray,
    factor: np.ndarray,
    solved: np.ndarray,
) -> tuple[float, float, float]:
    """Return a set of cells' score statistic, log2 fold change and its standard error.

    Under a fit with means ``mu``, dispersion phi and the Cholesky factor
    ``factor`` of its information, a cell's score is (y - mu) / (1 + phi mu)
    and its weight mu / (1 + phi mu). The statistic standardizes the set's
    summed score, as :func:`score_statistic` does, and the fold change is
    :func:`log2_fold_change` of the set. ``cells`` lists at least one cell,
    and they add up in its order; ``solved`` is room for one covariate vector.
    """
    p = covariates.shape[1]
    sums = np.zeros(p + 1)
    tangent = np.zeros(p)
    count_sum = 0.0
    mean_sum = 0.0
    square_sum = 0.0
    for j in cells:
        scale = 1.0 + dispersion * mu[j]
        weight = mu[j] / scale
        sums[0] += (y[j] - mu[j]) / scale
        for c in range(p):
            sums[1 + c] += weight * covariates[j, c]
        count_sum += y[j]
        mean_sum += mu[j]
        square_sum += mu[j] * mu[j]
        for c in range(p):
            tangent[c] += mu[j] * covariates[j, c]
    statistic = score_statistic(sums[0], sums[1], sums[1:], factor, solved)
    fold, error = log2_fold_change(
        count_sum, mean_sum, square_sum, tangent, factor, dispersion, solved
    )
    return statistic, fold, error


@njit(cache=True, nogil=True)
def logistic_probabilities(
    covariates: np.ndarray,
    coefficients: np.ndarray,
    eta: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Write each cell's linear predictor and its probability under a logistic model."""
    n, p = covariates.shape
    for i in range(n):
        total = 0.0
        for c in range(p):
            total += covariates[i, c] * coefficients[c]
        total = min(max(total, -_LINEAR_LIMIT), _LINEAR_LIMIT)
        eta[i] = total
        probabilities[i] = 1.0 / (1.0 + math.exp(-total))


@njit(cache=True, nogil=True)
def _logistic_likelihood(y: np.ndarray, eta: np.ndarray) -> float:
    """Return the logistic log-likelihood, y eta - log(1 + e^eta) summed.

    Each term is computed without overflow on either side of 0.
    """
    total = 0.0
    for i in range(len(y)):
        value = eta[i]
        if value > 0:
            total += (y[i] - 1.0) * value - math.log1p(math.exp(-value))
        else:
            total += y[i] * value - math.log1p(math.exp(value))
    return total


@njit(cache=True, nogil=True)
def fit_logistic(
    y: np.ndarray,
    covariates: np.ndarray,
    coefficients: np.ndarray,
    eta: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Fit logistic coefficients of responses from 0 to 1 by Newton's method.

    The fit starts from ``coefficients``. Each step halves until the
    log-likelihood does not fall. The fit stops after 50 steps, or once a
    step would raise the log-likelihood by less than 1e-12 of its size,
    after taking that step whole. A direction that the cells do not determine
    keeps its coefficient. Under separation, such as a sample whose cells all
    have responses of 0, the likelihood has no maximum: the steps drive those
    cells' probabilities towards 0 until the gains fall below the tolerance,
    the limit of the fit. ``eta`` and ``probabilities`` end at the fit.
    """
    n, p = covariates.shape
    information = np.empty((p, p))
    gradient = np.empty(p)
    factor = np.empty((p, p))
    step = np.empty(p)
    trial = np.empty(p)
    logistic_probabilities(covariates, coefficients, eta, probabilities)
    current = _logistic_likelihood(y, eta)
    for _ in range(_LOGISTIC_ITERATIONS):
        information[:, :] = 0.0
        gradient[:] = 0.0
        for i in range(n):
            probability = probabilities[i]
            weight = probability * (1.0 - probability)
            residual = y[i] - probability
            for r in range(p):
                weighted = weight * covariates[i, r]
                gradient[r] += residual * covariates[i, r]
                for c in range(r + 1):
                    information[r, c] += weighted * covariates[i, c]
        for r in range(p):
            for c in range(r + 1, p):
                information[r, c] = information[c, r]
        cholesky(information, factor)
        cholesky_solve(factor, gradient, step)
        # The step's predicted gain is half of g' J^-1 g. Gains this small
        # are lost in the likelihood's rounding, so the last step is taken
        # whole, which lands within about its square of the maximum.
        decrement = 0.0
        for c in range(p):
            decrement += gradient[c] * step[c]
        if decrement <= 2.0 * _FIT_TOLERANCE * (abs(current) + 1.0):
            for c in range(p):
                coefficients[c] += step[c]
            logistic_probabilities(covariates, coefficients, eta, probabilities)
            break
        scale = 1.0
        candidate = current
        for _half in range(60):
            for c in range(p):
                trial[c] = coefficients[c] + scale * step[c]
            logistic_probabilities(covariates, trial, eta, probabilities)
            candidate = _logistic_likelihood(y, eta)
            if candidate >= current:
                break
            scale *= 0.5
        coefficients[:] = trial
        current = candidate
