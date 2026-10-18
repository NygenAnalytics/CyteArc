"""Linear models with empirical Bayes moderated statistics.

Rows are features (here, categories) and columns are samples. Each row is
fitted by least squares on one design matrix, and the residual variances are
moderated by an empirical Bayes prior as in Smyth (2004): the variances are
modelled as a scaled F-distribution whose scale and second degrees of
freedom are estimated from the moments of the log variances. The robust
estimator Winsorizes the log variances and assigns smaller prior degrees of
freedom to outlying rows, and a trended prior lets the prior scale depend on
the average value of each row (Phipson, Lee, Majewski, Alexander and Smyth
2016).

The module was implemented from those papers. Agreement with limma's
``lmFit``, ``contrasts.fit``, ``squeezeVar`` and ``eBayes`` is checked by
comparing outputs; the behaviour at boundaries (infinite prior degrees of
freedom, zero variances, few rows, the trend smoother and its span) was
established by calling limma on constructed inputs. Only complete data
without observation weights is supported, so every row has the same
residual degrees of freedom.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma, polygamma

__all__ = [
    "FDistFit",
    "LinearModelFit",
    "ModeratedFit",
    "SqueezedVariances",
    "contrasts_fit",
    "e_bayes",
    "fit_f_dist",
    "fit_f_dist_robustly",
    "lm_fit",
    "squeeze_var",
    "subset_coefficients",
    "trigamma_inverse",
]

# Variances below this fraction of their median are raised to it, so that
# log variances stay finite. The robust estimator Winsorizes the small tail
# and uses a much smaller floor.
_ZERO_VARIANCE_OFFSET = 1e-5
_ROBUST_ZERO_VARIANCE_OFFSET = 1e-12
# The robust estimator needs enough rows to Winsorize; with fewer it falls
# back to the moment estimator.
_ROBUST_MIN_ROWS = 10
# Span of the robust lowess trend of the log variances.
_ROBUST_TREND_SPAN = 0.4
# Gauss-Legendre nodes for the Winsorized moments (Phipson et al. 2016,
# Section 10.4).
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = leggauss(128)


@dataclass(frozen=True, slots=True, eq=False)
class LinearModelFit:
    """Least-squares fits of every row on one design.

    ``coefficients`` and ``stdev_unscaled`` are rows x coefficients, and
    ``cov_coefficients`` is the unscaled coefficient covariance shared by
    all rows. ``amean`` is each row's mean value.
    """

    coefficients: np.ndarray
    stdev_unscaled: np.ndarray
    sigma: np.ndarray
    df_residual: np.ndarray
    cov_coefficients: np.ndarray
    design: np.ndarray
    amean: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class ModeratedFit:
    """Moderated t- and F-statistics of a :class:`LinearModelFit`.

    ``df_prior``, ``s2_prior``, ``s2_post``, ``df_total``, ``f_statistic``,
    ``f_df2`` and ``f_p_value`` hold one value per row; ``t`` and
    ``p_value`` are rows x coefficients.
    """

    fit: LinearModelFit
    df_prior: np.ndarray
    s2_prior: np.ndarray
    s2_post: np.ndarray
    t: np.ndarray
    df_total: np.ndarray
    p_value: np.ndarray
    f_statistic: np.ndarray
    f_df1: int
    f_df2: np.ndarray
    f_p_value: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class FDistFit:
    """A scaled F-distribution fitted to variances.

    ``scale`` has one value per variance, constant without a covariate.
    ``df2_shrunk`` is the robust estimator's per-variance second degrees of
    freedom, reduced for outliers, and is ``None`` for the moment estimator.
    """

    scale: np.ndarray
    df2: float
    df2_shrunk: np.ndarray | None = None


@dataclass(frozen=True, slots=True, eq=False)
class SqueezedVariances:
    """Empirical Bayes prior and posterior variances, one value per row."""

    df_prior: np.ndarray
    var_prior: np.ndarray
    var_post: np.ndarray


# Linear models ------------------------------------------------------------


def lm_fit(values: np.ndarray, design: np.ndarray) -> LinearModelFit:
    """Fit every row of ``values`` (rows x samples) on ``design``.

    Raises:
        ValueError: If values are not finite, the shapes disagree, or the
            design is not of full column rank.
    """
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    if y.ndim != 2 or x.ndim != 2 or x.shape[0] != y.shape[1]:
        raise ValueError("design must have one row per column of values")
    if not np.all(np.isfinite(y)):
        raise ValueError("values must be finite")
    n_samples, n_coefficients = x.shape
    if np.linalg.matrix_rank(x) < n_coefficients:
        raise ValueError("The design matrix is not of full column rank")
    q, r = np.linalg.qr(x)
    coefficients = np.linalg.solve(r, q.T @ y.T).T
    residuals = y - coefficients @ x.T
    df = n_samples - n_coefficients
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt((residuals**2).sum(axis=1) / df) if df > 0 else None
    if sigma is None:
        sigma = np.full(y.shape[0], np.nan)
    r_inverse = np.linalg.inv(r)
    cov = r_inverse @ r_inverse.T
    return LinearModelFit(
        coefficients=coefficients,
        stdev_unscaled=np.tile(np.sqrt(np.diag(cov)), (y.shape[0], 1)),
        sigma=sigma,
        df_residual=np.full(y.shape[0], float(df)),
        cov_coefficients=cov,
        design=x,
        amean=y.mean(axis=1),
    )


def contrasts_fit(fit: LinearModelFit, contrasts: np.ndarray) -> LinearModelFit:
    """Re-express a fit in terms of contrasts of its coefficients.

    ``contrasts`` is coefficients x contrasts, or one contrast vector.
    """
    weights = np.asarray(contrasts, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or weights.shape[0] != fit.coefficients.shape[1]:
        raise ValueError("contrasts must have one row per coefficient")
    cov = weights.T @ fit.cov_coefficients @ weights
    return replace(
        fit,
        coefficients=fit.coefficients @ weights,
        stdev_unscaled=np.tile(np.sqrt(np.diag(cov)), (fit.coefficients.shape[0], 1)),
        cov_coefficients=cov,
    )


def subset_coefficients(fit: LinearModelFit, columns: list[int]) -> LinearModelFit:
    """Keep only the given coefficients, with their covariance."""
    index = np.asarray(columns, dtype=np.int64)
    return replace(
        fit,
        coefficients=fit.coefficients[:, index],
        stdev_unscaled=fit.stdev_unscaled[:, index],
        cov_coefficients=fit.cov_coefficients[np.ix_(index, index)],
    )


# Hyperparameter estimation ------------------------------------------------


def trigamma_inverse(x: np.ndarray | float) -> np.ndarray:
    """Return ``y`` with ``trigamma(y) == x``.

    Uses the monotonically convergent Newton iteration of Smyth (2004,
    appendix) on ``1 / trigamma``, started at ``0.5 + 1 / x``. Very large
    and very small arguments use the asymptotes ``1 / sqrt(x)`` and
    ``1 / x``. Non-positive arguments give infinity.
    """
    values = np.asarray(x, dtype=np.float64)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    result[finite & (values <= 0)] = np.inf
    result[values == np.inf] = 0.0
    large = finite & (values > 1e7)
    small = finite & (values > 0) & (values < 1e-6)
    result[large] = 1 / np.sqrt(values[large])
    result[small] = 1 / values[small]
    middle = finite & (values >= 1e-6) & (values <= 1e7)
    target = values[middle]
    y = 0.5 + 1 / target
    active = np.ones(target.shape, dtype=bool)
    for _ in range(50):
        tri = polygamma(1, y[active])
        step = tri * (1 - tri / target[active]) / polygamma(2, y[active])
        y[active] += step
        done = -step / y[active] < 1e-8
        active[np.flatnonzero(active)[done]] = False
        if not active.any():
            break
    result[middle] = y
    return result


def _natural_spline_basis(x: np.ndarray, df: int) -> np.ndarray:
    """Natural cubic spline basis with an intercept.

    The knots are the range of ``x`` and ``df - 1`` interior knots at the
    quantiles ``1 / df, ..., (df - 1) / df``; the basis spans cubic splines
    that are linear beyond the boundary knots (Hastie, Tibshirani and
    Friedman, *The Elements of Statistical Learning*, eq. 5.4-5.5).
    """
    low, high = float(x.min()), float(x.max())
    width = high - low if high > low else 1.0
    t = (x - low) / width
    interior = np.quantile(t, np.arange(1, df) / df) if df > 1 else np.empty(0)
    knots = np.concatenate([[0.0], interior, [(high - low) / width]])
    last = knots[-1]

    def truncated(index: int) -> np.ndarray:
        cubes = np.clip(t - knots[index], 0, None) ** 3
        tail = np.clip(t - last, 0, None) ** 3
        return np.asarray((cubes - tail) / (last - knots[index]))

    columns = [np.ones_like(t), t]
    columns.extend(
        truncated(index) - truncated(knots.size - 2) for index in range(knots.size - 2)
    )
    return np.column_stack(columns)


def _trend_spline_df(n: int) -> int:
    """Spline degrees of freedom, besides the intercept, of the trend."""
    return 1 + int(n >= 6) + int(n >= 30)


def _positive_variances(
    x: np.ndarray, offset: float = _ZERO_VARIANCE_OFFSET
) -> np.ndarray:
    values = np.maximum(x, 0.0)
    median = float(np.median(values))
    floor = offset * median if median > 0 else 1.0
    return np.asarray(np.maximum(values, floor))


def _ok_rows(
    x: np.ndarray, df1: np.ndarray, covariate: np.ndarray | None
) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(df1) & (df1 > 0)
    if covariate is not None:
        ok &= np.isfinite(covariate)
    return ok


def _broadcast(
    x: np.ndarray | float, df1: np.ndarray | float, covariate: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    values = np.atleast_1d(np.asarray(x, dtype=np.float64))
    df = np.broadcast_to(np.asarray(df1, dtype=np.float64), values.shape).copy()
    if covariate is None:
        return values, df, None
    cov = np.asarray(covariate, dtype=np.float64)
    if cov.shape != values.shape:
        raise ValueError("covariate must have one value per variance")
    return values, df, cov


def fit_f_dist(
    x: np.ndarray, df1: np.ndarray | float, covariate: np.ndarray | None = None
) -> FDistFit:
    """Fit ``x ~ scale * F(df1, df2)`` by moments of ``log(x)`` (Smyth 2004).

    With a covariate the log scale is a natural cubic spline in it, and the
    spread is measured about that trend (Phipson et al. 2016, Section 10.1).
    When the log variances vary no more than sampling alone explains,
    ``df2`` is infinite and the scale is the mean variance (or the trend).
    Missing values are left out of the fit, and the covariate is ignored
    with fewer than three variances.
    """
    values, df, cov = _broadcast(x, df1, covariate)
    scale = np.full(values.shape, np.nan)
    ok = _ok_rows(values, df, cov)
    n = int(ok.sum())
    if n == 0:
        return FDistFit(scale=scale, df2=np.nan)
    if n == 1:
        scale[:] = float(values[ok][0])
        return FDistFit(scale=scale, df2=0.0)
    positive = _positive_variances(values[ok])
    d1 = df[ok]
    e = np.log(positive) - digamma(d1 / 2) + np.log(d1 / 2)
    if cov is None or n < 3:
        mean_e: np.ndarray | float = float(e.mean())
        spread = float(((e - e.mean()) ** 2).sum() / (n - 1))
    else:
        basis = _natural_spline_basis(cov[ok], _trend_spline_df(n))
        coef, *_ = np.linalg.lstsq(basis, e, rcond=None)
        mean_e = basis @ coef
        spread = float(((e - mean_e) ** 2).sum() / (n - basis.shape[1]))
    excess = spread - float(polygamma(1, d1 / 2).mean())
    if excess > 0:
        df2 = float(2 * trigamma_inverse(excess))
        fitted = np.exp(mean_e + digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        fitted = np.exp(mean_e) if np.ndim(mean_e) else float(positive.mean())
    if np.ndim(fitted) == 0:
        scale[:] = fitted
    else:
        scale[ok] = fitted
    return FDistFit(scale=scale, df2=df2)


def _lowess(
    x: np.ndarray, y: np.ndarray, span: float, iterations: int = 3
) -> np.ndarray:
    """Robust locally linear smoother of Cleveland (1979).

    Each point is fitted by weighted least squares on its ``floor(span * n)``
    nearest neighbours with tricube weights, then refitted ``iterations``
    times with bisquare robustness weights built from six times the median
    absolute residual. Points within one hundredth of the range of ``x`` of
    the last fitted point are interpolated linearly, and weights within
    0.001 of one or zero are rounded to one or zero. Returns the fitted
    values in the input order.
    """
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = xs.size
    neighbours = int(max(min(np.floor(span * n), n), 2))
    delta = 0.01 * (xs[-1] - xs[0])
    robustness = np.ones(n)
    fitted = np.empty(n)
    for iteration in range(iterations + 1):
        last = -1
        point = 0
        while True:
            distance = np.abs(xs - xs[point])
            bandwidth = np.partition(distance, neighbours - 1)[neighbours - 1]
            if bandwidth > 0:
                u = distance / bandwidth
                weights = np.where(u < 1, (1 - u**3) ** 3, 0.0)
                weights[u <= 0.001] = 1.0
                weights[u > 0.999] = 0.0
            else:
                weights = (distance == 0).astype(np.float64)
            weights *= robustness
            total = weights.sum()
            if total <= 0:
                fitted[point] = ys[point]
            else:
                weights /= total
                centre = float(weights @ xs)
                spread = float(weights @ (xs - centre) ** 2)
                value = float(weights @ ys)
                if spread > 0:
                    slope = float(weights @ ((xs - centre) * ys)) / spread
                    value += slope * (xs[point] - centre)
                fitted[point] = value
            if last >= 0 and point - last > 1:
                fraction = (xs[last + 1 : point] - xs[last]) / (xs[point] - xs[last])
                fitted[last + 1 : point] = fitted[last] + fraction * (
                    fitted[point] - fitted[last]
                )
            last = point
            while last + 1 < n and xs[last + 1] == xs[point]:
                last += 1
                fitted[last] = fitted[point]
            if last >= n - 1:
                break
            within = int(np.searchsorted(xs, xs[last] + delta, side="right")) - 1
            point = max(within, last + 1)
        if iteration == iterations:
            break
        residuals = ys - fitted
        scale = float(np.median(np.abs(residuals)))
        if scale <= 1e-12 * float(np.abs(ys).max(initial=0.0)):
            break
        u = np.abs(residuals) / (6 * scale)
        robustness = np.where(u < 1, (1 - u**2) ** 2, 0.0)
        robustness[u <= 0.001] = 1.0
        robustness[u > 0.999] = 0.0
    result = np.empty(n)
    result[order] = fitted
    return result


def _null_distribution(df1: float, df2: float) -> stats.rv_continuous:
    if np.isinf(df2):
        return stats.chi2(df1, scale=1 / df1)
    return stats.f(df1, df2)


def _winsorized_log_moments(
    df1: float, df2: float, tail_p: tuple[float, float]
) -> tuple[float, float]:
    """Mean and variance of ``log`` of a Winsorized ``F(df1, df2)`` variable.

    The tails beyond the ``tail_p`` quantiles are set to those quantiles.
    The central part is integrated by Gauss-Legendre quadrature after the
    change of variable ``u = f / (1 + f)`` (Phipson et al. 2016, Section
    10.3).
    """
    lower_p, upper_p = tail_p
    distribution = _null_distribution(df1, df2)
    lower, upper = float(distribution.ppf(lower_p)), float(distribution.isf(upper_p))
    a, b = lower / (1 + lower), upper / (1 + upper)
    u = (b - a) / 2 * _QUADRATURE_NODES + (a + b) / 2
    f = u / (1 - u)
    density = distribution.pdf(f) / (1 - u) ** 2 * (_QUADRATURE_WEIGHTS / 2)
    log_f = np.log(f)
    mean = (
        lower_p * np.log(lower)
        + (b - a) * float(density @ log_f)
        + upper_p * np.log(upper)
    )
    variance = (
        lower_p * (np.log(lower) - mean) ** 2
        + (b - a) * float(density @ (log_f - mean) ** 2)
        + upper_p * (np.log(upper) - mean) ** 2
    )
    return float(mean), float(variance)


def _robust_df2(
    df1: float, spread: float, lower_bound: float, tail_p: tuple[float, float]
) -> float | None:
    """Solve ``var(log win(F(df1, df2))) == spread`` for ``df2``.

    The moment estimate ``lower_bound`` bounds the answer from below.
    Returns ``None`` when the Winsorized spread calls for a finite ``df2``
    but the moment estimate is infinite; the moment fit is then kept whole.
    """
    if spread <= _winsorized_log_moments(df1, np.inf, tail_p)[1]:
        return np.inf
    if np.isinf(lower_bound):
        return None
    start = 1 / lower_bound

    def excess(inverse_df2: float) -> float:
        df2 = np.inf if inverse_df2 == 0 else 1 / inverse_df2
        variance = _winsorized_log_moments(df1, df2, tail_p)[1]
        # The variance grows without bound as df2 falls to zero; quadrature
        # overflows first, so report a large excess there.
        return variance - spread if np.isfinite(variance) else 1e300

    if excess(start) <= 0:
        return lower_bound
    return float(1 / brentq(excess, 0.0, start, xtol=1e-300, rtol=1e-15))


def _outlier_df2(
    ratio: np.ndarray, df1: float, df2: float, total_df: float
) -> np.ndarray:
    """Per-variance prior df, reduced for outliers (Phipson et al. 2016, Section 4)."""
    n = ratio.size
    tail = _null_distribution(df1, df2).sf(ratio)
    rank = stats.rankdata(-ratio)
    prob = np.minimum(tail / ((rank - 0.5) / n), 1.0)
    if np.all(prob >= 1):
        return np.full(n, df2)
    order = np.argsort(tail, kind="stable")
    ordered = prob[order]
    if np.isinf(df2):
        # With an infinite prior df the outliers' df scale with the pooled
        # df, and the probabilities are only made monotone.
        prob[order] = np.maximum.accumulate(ordered)
        return np.where(prob < 1, prob * total_df, np.inf)
    # Lift the leading probabilities to their smallest running mean, then
    # make them non-decreasing in the tail probability.
    running = np.cumsum(ordered) / np.arange(1, n + 1)
    first_min = int(np.argmin(running))
    ordered[: first_min + 1] = running[first_min]
    prob[order] = np.maximum.accumulate(ordered)
    # Two steps of the monotone iteration of Section 10.5 for the df at which
    # the largest ratio is the median.
    outlier_df = df2
    largest = float(ratio.max())
    for _ in range(2):
        tail_max = float(stats.f.sf(largest, df1, outlier_df))
        outlier_df *= np.log(0.5) / np.log(tail_max)
    return np.asarray(prob * df2 + (1 - prob) * outlier_df)


def fit_f_dist_robustly(
    x: np.ndarray,
    df1: np.ndarray | float,
    covariate: np.ndarray | None = None,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> FDistFit:
    """Fit a scaled F-distribution robustly (Phipson et al. 2016).

    The log variances, detrended by a robust lowess curve when a covariate is
    given, are Winsorized at the ``winsor_tail_p`` lower and upper tail
    proportions, and ``df2`` is chosen so that the variance of a Winsorized
    log F variable matches theirs. ``df2`` is never smaller than the moment
    estimate of :func:`fit_f_dist`, and when the moment estimate is
    infinite while the Winsorized spread asks for a finite value, the moment
    fit is returned. Outlying variances receive smaller values in
    ``df2_shrunk``. With fewer than ten variances, or unequal ``df1``, this
    returns the moment estimate.
    """
    values, df, cov = _broadcast(x, df1, covariate)
    moments = fit_f_dist(values, df, cov)
    ok = _ok_rows(values, df, cov)
    n = int(ok.sum())
    d1 = df[ok]
    if n < _ROBUST_MIN_ROWS or np.ptp(d1) > 0:
        return replace(moments, df2_shrunk=np.full(values.shape, moments.df2))
    df1_value = float(d1[0])
    log_values = np.log(_positive_variances(values[ok], _ROBUST_ZERO_VARIANCE_OFFSET))
    trend = (
        _lowess(cov[ok], log_values, _ROBUST_TREND_SPAN)
        if cov is not None
        else np.zeros(n)
    )
    detrended = log_values - trend
    lower, upper = np.quantile(detrended, [winsor_tail_p[0], 1 - winsor_tail_p[1]])
    winsorized = np.clip(detrended, lower, upper)
    df2 = _robust_df2(
        df1_value, float(winsorized.var(ddof=1)), moments.df2, winsor_tail_p
    )
    if df2 is None:
        return replace(moments, df2_shrunk=np.full(values.shape, np.inf))
    log_scale = (
        float(winsorized.mean())
        - _winsorized_log_moments(df1_value, df2, winsor_tail_p)[0]
    )
    scale = np.full(values.shape, np.nan)
    scale[ok] = np.exp(log_scale + trend)
    shrunk = np.full(values.shape, np.nan)
    shrunk[ok] = _outlier_df2(
        np.exp(log_values - trend - log_scale), df1_value, df2, float(d1.sum())
    )
    return FDistFit(scale=scale, df2=df2, df2_shrunk=shrunk)


def squeeze_var(
    var: np.ndarray,
    df: np.ndarray | float,
    covariate: np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> SqueezedVariances:
    """Return empirical Bayes posterior variances.

    The posterior variance is the average of the prior and observed
    variances weighted by their degrees of freedom; it equals the prior
    when the prior degrees of freedom are infinite and for rows without
    residual degrees of freedom. With fewer than three variances no prior
    is estimated and the variances are returned unchanged.

    Raises:
        ValueError: If the positive residual degrees of freedom differ
            between rows.
    """
    values = np.asarray(var, dtype=np.float64)
    df_values = np.broadcast_to(np.asarray(df, dtype=np.float64), values.shape)
    positive = df_values[df_values > 0]
    if positive.size and np.ptp(positive) > 0:
        raise ValueError(
            "squeeze_var needs equal residual degrees of freedom in every row"
        )
    if positive.size < 3:
        # Too few variances to estimate a prior: leave them unmoderated.
        return SqueezedVariances(
            df_prior=np.zeros(values.shape),
            var_prior=values.copy(),
            var_post=values.copy(),
        )
    fit = (
        fit_f_dist_robustly(values, df_values, covariate, winsor_tail_p)
        if robust
        else fit_f_dist(values, df_values, covariate)
    )
    df_prior = (
        np.asarray(fit.df2_shrunk, dtype=np.float64)
        if robust and fit.df2_shrunk is not None
        else np.full(values.shape, fit.df2)
    )
    var_prior = np.asarray(fit.scale, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        var_post = (df_values * values + df_prior * var_prior) / (df_values + df_prior)
    use_prior = np.isinf(df_prior) | (df_values == 0)
    var_post = np.where(use_prior, var_prior, var_post)
    return SqueezedVariances(df_prior=df_prior, var_prior=var_prior, var_post=var_post)


# Moderated statistics ---------------------------------------------------------


def e_bayes(
    fit: LinearModelFit,
    *,
    trend: bool = False,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> ModeratedFit:
    """Moderated t-statistics per coefficient and an F-statistic per row.

    The residual variances are squeezed with :func:`squeeze_var`, using each
    row's mean value as the covariate when ``trend`` is set. The
    t-statistics have the residual plus prior degrees of freedom, but no
    more than the residual degrees of freedom pooled over rows. The
    F-statistic tests all coefficients of ``fit`` equal to zero: the
    quadratic form of the moderated t-statistics in the inverse of their
    correlation matrix, divided by its rank (Smyth 2004, Section 7). Its
    denominator degrees of freedom are the residual plus prior degrees of
    freedom without that cap, so an infinite prior gives a chi-square
    test.
    """
    squeezed = squeeze_var(
        fit.sigma**2,
        fit.df_residual,
        covariate=fit.amean if trend else None,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    df_pooled = float(fit.df_residual.sum())
    df_total = np.minimum(fit.df_residual + squeezed.df_prior, df_pooled)
    s_post = np.sqrt(squeezed.var_post)
    t = fit.coefficients / fit.stdev_unscaled / s_post[:, None]
    p_value = 2 * stats.t.sf(np.abs(t), df_total[:, None])

    stdev = np.sqrt(np.diag(fit.cov_coefficients))
    correlation = fit.cov_coefficients / np.outer(stdev, stdev)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    keep = eigenvalues > eigenvalues.max() * 1e-8
    rank = int(keep.sum())
    projected = t @ eigenvectors[:, keep]
    f_statistic = (projected**2 / eigenvalues[keep]).sum(axis=1) / rank
    f_df2 = fit.df_residual + squeezed.df_prior
    finite = np.isfinite(f_df2)
    f_p_value = np.where(
        finite,
        stats.f.sf(f_statistic, rank, np.where(finite, f_df2, 1.0)),
        stats.chi2.sf(f_statistic * rank, rank),
    )
    return ModeratedFit(
        fit=fit,
        df_prior=squeezed.df_prior,
        s2_prior=squeezed.var_prior,
        s2_post=squeezed.var_post,
        t=t,
        df_total=df_total,
        p_value=p_value,
        f_statistic=f_statistic,
        f_df1=rank,
        f_df2=f_df2,
        f_p_value=f_p_value,
    )
