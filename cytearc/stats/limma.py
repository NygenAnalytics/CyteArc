"""Linear models with empirical Bayes moderation, ported from limma 3.66.0.

The functions follow ``lmFit``, ``contrasts.fit``, ``eBayes``,
``squeezeVar``, ``fitFDist``, ``fitFDistRobustly`` and ``trigammaInverse``.
Rows are features (here categories) and columns are samples. Only complete
data without observation weights is supported, which is the case that
propeller uses, so every row has the same residual df and ``squeezeVar``
takes its legacy path.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats
from scipy.special import polygamma

from ._rmath import (
    logmdigamma,
    lowess,
    natural_spline_basis,
    trimmed_mean,
    zeroin,
)

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


@dataclass(frozen=True, slots=True, eq=False)
class LinearModelFit:
    """Row-wise least-squares fits that share one design matrix."""

    coefficients: np.ndarray
    stdev_unscaled: np.ndarray
    sigma: np.ndarray
    df_residual: np.ndarray
    cov_coefficients: np.ndarray
    design: np.ndarray
    amean: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class ModeratedFit:
    """Moderated t and F statistics of :func:`e_bayes`."""

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
    """Scaled F distribution fitted to sample variances."""

    scale: np.ndarray
    df2: float
    df2_shrunk: np.ndarray | None = None


@dataclass(frozen=True, slots=True, eq=False)
class SqueezedVariances:
    """Posterior variances and the prior of :func:`squeeze_var`."""

    df_prior: np.ndarray
    var_prior: np.ndarray
    var_post: np.ndarray


def _as_vector(value: np.ndarray | float, n: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n, float(array))
    return array


def lm_fit(values: np.ndarray, design: np.ndarray) -> LinearModelFit:
    """Fit ``lmFit(values, design)`` for complete data.

    Raises:
        ValueError: If values are not finite or the design is not of full rank.
    """
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    if y.ndim != 2 or x.ndim != 2 or y.shape[1] != x.shape[0]:
        raise ValueError("values must be features x samples and match the design rows")
    if not np.all(np.isfinite(y)):
        raise ValueError("lm_fit requires finite values")
    n_samples, n_coef = x.shape
    q, r = np.linalg.qr(x)
    rank = int(np.linalg.matrix_rank(x))
    if rank < n_coef:
        raise ValueError("design matrix is not of full rank")
    coefficients = np.linalg.solve(r, q.T @ y.T).T
    residuals = y - coefficients @ x.T
    df_residual = n_samples - n_coef
    if df_residual > 0:
        sigma = np.sqrt(np.sum(residuals**2, axis=1) / df_residual)
    else:
        sigma = np.full(y.shape[0], np.nan)
    r_inv = np.linalg.inv(r)
    cov = r_inv @ r_inv.T
    stdev = np.tile(np.sqrt(np.diag(cov)), (y.shape[0], 1))
    return LinearModelFit(
        coefficients=coefficients,
        stdev_unscaled=stdev,
        sigma=sigma,
        df_residual=np.full(y.shape[0], float(df_residual)),
        cov_coefficients=cov,
        design=x,
        amean=np.mean(y, axis=1),
    )


def _cov_to_cor(cov: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(cov))
    return np.asarray(cov / np.outer(scale, scale))


def contrasts_fit(fit: LinearModelFit, contrasts: np.ndarray) -> LinearModelFit:
    """Return ``contrasts.fit(fit, contrasts)``."""
    matrix = np.asarray(contrasts, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    n_coef = fit.coefficients.shape[1]
    if matrix.shape[0] != n_coef:
        raise ValueError("contrast rows must match the number of coefficients")
    coefficients = fit.coefficients
    stdev = fit.stdev_unscaled
    cov = fit.cov_coefficients
    cormatrix = _cov_to_cor(cov)
    all_zero = np.sum(np.abs(matrix), axis=1) == 0
    if np.any(all_zero):
        keep = ~all_zero
        matrix = matrix[keep]
        coefficients = coefficients[:, keep]
        stdev = stdev[:, keep]
        cov = cov[np.ix_(keep, keep)]
        cormatrix = cormatrix[np.ix_(keep, keep)]
    new_coefficients = coefficients @ matrix
    if cormatrix.size < 2:
        orthogonal = True
    else:
        lower = np.tril_indices_from(cormatrix, k=-1)
        orthogonal = bool(np.all(np.abs(cormatrix[lower]) < 1e-14))
    chol = np.linalg.cholesky(cov).T
    new_cov = (chol @ matrix).T @ (chol @ matrix)
    if orthogonal:
        new_stdev = np.sqrt(stdev**2 @ matrix**2)
    else:
        chol_cor = np.linalg.cholesky(cormatrix).T
        scaled = stdev[:, :, None] * matrix[None, :, :]
        projected = np.einsum("ij,gjc->gic", chol_cor, scaled)
        new_stdev = np.sqrt(np.sum(projected**2, axis=1))
    return replace(
        fit,
        coefficients=new_coefficients,
        stdev_unscaled=new_stdev,
        cov_coefficients=new_cov,
    )


def subset_coefficients(fit: LinearModelFit, columns: list[int]) -> LinearModelFit:
    """Return ``fit[, columns]`` with its coefficient covariance."""
    index = np.asarray(columns, dtype=np.int64)
    return replace(
        fit,
        coefficients=fit.coefficients[:, index],
        stdev_unscaled=fit.stdev_unscaled[:, index],
        cov_coefficients=fit.cov_coefficients[np.ix_(index, index)],
    )


def trigamma_inverse(x: np.ndarray | float) -> np.ndarray:
    """Return limma's ``trigammaInverse``."""
    values = np.atleast_1d(np.asarray(x, dtype=np.float64)).copy()
    result = np.full(values.shape, np.nan)
    finite = ~np.isnan(values)
    large = finite & (values > 1e7)
    small = finite & (values < 1e-6) & (values >= 0)
    newton = finite & (values >= 1e-6) & (values <= 1e7)
    result[large] = 1.0 / np.sqrt(values[large])
    result[small] = 1.0 / values[small]
    if np.any(newton):
        target = values[newton]
        y = 0.5 + 1.0 / target
        iterations = 0
        while True:
            iterations += 1
            tri = polygamma(1, y)
            dif = tri * (1.0 - tri / target) / polygamma(2, y)
            y = y + dif
            if np.max(-dif / y) < 1e-8 or iterations > 50:
                break
        result[newton] = y
    return result


def _pf_upper(f: np.ndarray, df1: np.ndarray | float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        d1 = np.asarray(df1, dtype=np.float64)
        return np.asarray(stats.chi2.sf(f * d1, d1))
    return np.asarray(stats.f.sf(f, df1, df2))


def _pf_lower(f: np.ndarray, df1: np.ndarray | float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        d1 = np.asarray(df1, dtype=np.float64)
        return np.asarray(stats.chi2.cdf(f * d1, d1))
    return np.asarray(stats.f.cdf(f, df1, df2))


def _log_pf_upper(f: np.ndarray, df1: np.ndarray | float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        d1 = np.asarray(df1, dtype=np.float64)
        return np.asarray(stats.chi2.logsf(f * d1, d1))
    return np.asarray(stats.f.logsf(f, df1, df2))


def _log_pf_lower(f: np.ndarray, df1: np.ndarray | float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        d1 = np.asarray(df1, dtype=np.float64)
        return np.asarray(stats.chi2.logcdf(f * d1, d1))
    return np.asarray(stats.f.logcdf(f, df1, df2))


def _qf(p: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        return np.asarray(stats.chi2.ppf(p, df1) / df1)
    return np.asarray(stats.f.ppf(p, df1, df2))


def _f_density(x: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        return np.asarray(df1 * stats.chi2.pdf(x * df1, df1))
    return np.asarray(stats.f.pdf(x, df1, df2))


def fit_f_dist(
    x: np.ndarray,
    df1: np.ndarray | float,
    covariate: np.ndarray | None = None,
) -> FDistFit:
    """Return ``fitFDist(x, df1, covariate)``."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    if n == 0:
        return FDistFit(scale=np.array([np.nan]), df2=np.nan)
    if n == 1:
        return FDistFit(scale=x.copy(), df2=0.0)
    df1_array = np.asarray(df1, dtype=np.float64)
    scalar_df1 = df1_array.ndim == 0
    if scalar_df1:
        if not (np.isfinite(df1_array) and df1_array > 1e-15):
            return FDistFit(scale=np.array([np.nan]), df2=np.nan)
        ok = np.ones(n, dtype=bool)
    else:
        if df1_array.size != n:
            raise ValueError("x and df1 have different lengths")
        ok = np.isfinite(df1_array) & (df1_array > 1e-15)
    cov = None
    if covariate is not None:
        cov = np.asarray(covariate, dtype=np.float64).copy()
        if cov.size != n:
            raise ValueError("x and covariate must be of same length")
        if np.any(np.isnan(cov)):
            raise ValueError("NA covariate values not allowed")
        finite = np.isfinite(cov)
        if not np.all(finite):
            if np.any(finite):
                low, high = cov[finite].min(), cov[finite].max()
                cov[cov == -np.inf] = low - 1
                cov[cov == np.inf] = high + 1
            else:
                cov = np.sign(cov)
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 1:
        return FDistFit(scale=x[ok], df2=0.0)
    notallok = nok < n
    cov_notok = None
    if notallok:
        x = x[ok]
        if not scalar_df1:
            df1_array = df1_array[ok]
        if cov is not None:
            cov_notok = cov[~ok]
            cov = cov[ok]
    splinedf = 1
    if cov is not None:
        splinedf = 1 + int(nok >= 3) + int(nok >= 6) + int(nok >= 30)
        splinedf = min(splinedf, int(np.unique(cov).size))
        if splinedf < 2:
            out = fit_f_dist(x, df1_array)
            return FDistFit(scale=np.full(n, float(out.scale[0])), df2=out.df2)
    x = np.maximum(x, 0.0)
    m = float(np.median(x))
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)
    z = np.log(x)
    e = z + logmdigamma(df1_array / 2)
    if cov is None:
        emean_value = float(np.mean(e))
        evar = float(np.sum((e - emean_value) ** 2) / (nok - 1))
        emean = np.array([emean_value])
    else:
        basis, predicted = natural_spline_basis(
            cov, splinedf, new_x=cov_notok if notallok else None
        )
        coef, _, rank, _ = np.linalg.lstsq(basis, e, rcond=None)
        fitted = basis @ coef
        evar = float(np.sum((e - fitted) ** 2) / (e.size - rank))
        if notallok:
            assert predicted is not None
            emean = np.zeros(n)
            emean[ok] = fitted
            emean[~ok] = predicted @ coef
        else:
            emean = fitted
    evar = evar - float(np.mean(polygamma(1, df1_array / 2)))
    if evar > 0:
        df2 = float(2 * trigamma_inverse(evar)[0])
        s20 = np.exp(emean - logmdigamma(df2 / 2))
    else:
        df2 = math.inf
        s20 = np.array([float(np.mean(x))]) if cov is None else np.exp(emean)
    return FDistFit(scale=np.asarray(s20, dtype=np.float64), df2=df2)


def _winsorized_moments(
    df1: float, df2: float, winsor_tail_p: np.ndarray
) -> tuple[float, float]:
    nodes, weights = np.polynomial.legendre.leggauss(128)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    fq = _qf(np.array([winsor_tail_p[0], 1 - winsor_tail_p[1]]), df1, df2)
    zq = np.log(fq)
    q = fq / (1 + fq)
    q_nodes = q[0] + (q[1] - q[0]) * nodes
    f_nodes = q_nodes / (1 - q_nodes)
    z_nodes = np.log(f_nodes)
    density = _f_density(f_nodes, df1, df2) / (1 - q_nodes) ** 2
    q21 = q[1] - q[0]
    mean = q21 * np.sum(weights * density * z_nodes) + np.sum(zq * winsor_tail_p)
    var = q21 * np.sum(weights * density * (z_nodes - mean) ** 2) + np.sum(
        (zq - mean) ** 2 * winsor_tail_p
    )
    return float(mean), float(var)


def _r_quantile(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(np.quantile(values, probabilities))


def fit_f_dist_robustly(
    x: np.ndarray,
    df1: np.ndarray | float,
    covariate: np.ndarray | None = None,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> FDistFit:
    """Return ``fitFDistRobustly(x, df1, covariate, winsor.tail.p)``."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    if n < 2:
        return FDistFit(scale=np.array([np.nan]), df2=np.nan, df2_shrunk=None)
    if n == 2:
        return fit_f_dist(x, df1, covariate)
    df1_array = np.asarray(df1, dtype=np.float64)
    cov = None if covariate is None else np.asarray(covariate, dtype=np.float64)
    ok = ~np.isnan(x) & np.isfinite(df1_array) & (df1_array > 1e-6)
    ok = np.broadcast_to(ok, x.shape).copy()
    if not np.all(ok):
        df2_shrunk = x.copy()
        sub_df1 = df1_array if df1_array.ndim == 0 else df1_array[ok]
        sub_cov = None if cov is None else cov[ok]
        fit = fit_f_dist_robustly(x[ok], sub_df1, sub_cov, winsor_tail_p)
        assert fit.df2_shrunk is not None
        df2_shrunk[ok] = fit.df2_shrunk
        df2_shrunk[~ok] = fit.df2
        if cov is None:
            scale = fit.scale
        else:
            scale = x.copy()
            scale[ok] = fit.scale
            order = np.argsort(cov[ok], kind="stable")
            scale[~ok] = np.exp(
                np.interp(cov[~ok], cov[ok][order], np.log(fit.scale)[order])
            )
        return FDistFit(scale=scale, df2=fit.df2, df2_shrunk=df2_shrunk)
    m = float(np.median(x))
    if m <= 0:
        raise ValueError("Variances are mostly <= 0")
    x = np.where(x < m * 1e-12, m * 1e-12, x)
    non_robust = fit_f_dist(x, df1_array, cov)
    tail = np.resize(np.asarray(winsor_tail_p, dtype=np.float64), 2)
    prob = np.array([tail[0], 1 - tail[1]])
    if np.all(tail < 1 / n):
        return replace(non_robust, df2_shrunk=np.full(n, non_robust.df2))
    if df1_array.ndim > 0 and df1_array.size > 1:
        df1max = float(np.max(df1_array))
        lower_df = df1_array < (df1max - 1e-14)
        if np.any(lower_df):
            s = non_robust.scale if cov is None else non_robust.scale[lower_df]
            f = x[lower_df] / s
            df2 = non_robust.df2
            pupper = _log_pf_upper(f, df1_array[lower_df], df2)
            plower = _log_pf_lower(f, df1_array[lower_df], df2)
            up = pupper < plower
            f = f.copy()
            if np.any(up):
                f[up] = _qf_from_log_upper(pupper[up], df1max, df2)
            if np.any(~up):
                f[~up] = _qf_from_log_lower(plower[~up], df1max, df2)
            x = x.copy()
            x[lower_df] = f * s
            df1_value = df1max
        else:
            df1_value = float(df1_array[0])
    else:
        df1_value = float(df1_array)
    z = np.log(x)
    if cov is None:
        ztrend: np.ndarray | float = trimmed_mean(z, tail[1])
        zresid = z - ztrend
    else:
        ztrend = lowess(cov, z, f=0.4, iterations=3)
        zresid = z - ztrend
    zrq = _r_quantile(zresid, prob)
    zwins = np.minimum(np.maximum(zresid, zrq[0]), zrq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.mean((zwins - zwmean) ** 2) * n / (n - 1))

    def link_inverse(value: float) -> float:
        return math.inf if value >= 1 else value / (1 - value)

    mom_mean, mom_var = _winsorized_moments(df1_value, math.inf, tail)
    funval_inf = math.log(zwvar / mom_var)
    if funval_inf <= 0:
        ztrend_corrected = ztrend + zwmean - mom_mean
        s20 = np.exp(np.broadcast_to(ztrend_corrected, (n,)))
        fstat = np.exp(z - ztrend_corrected)
        tail_p = stats.chi2.sf(fstat * df1_value, df1_value)
        ranks = stats.rankdata(fstat)
        empirical = (n - ranks + 0.5) / n
        prob_not_outlier = np.minimum(tail_p / empirical, 1)
        df_pooled = n * df1_value
        df2_shrunk = np.full(n, math.inf)
        outlier = prob_not_outlier < 1
        if np.any(outlier):
            df2_shrunk[outlier] = prob_not_outlier[outlier] * df_pooled
            order = np.argsort(tail_p, kind="stable")
            df2_shrunk[order] = np.maximum.accumulate(df2_shrunk[order])
        return FDistFit(
            scale=np.asarray(s20, dtype=np.float64),
            df2=math.inf,
            df2_shrunk=df2_shrunk,
        )

    def objective(value: float) -> float:
        _, variance = _winsorized_moments(df1_value, link_inverse(value), tail)
        return math.log(zwvar / variance)

    if math.isinf(non_robust.df2):
        return replace(non_robust, df2_shrunk=np.full(n, non_robust.df2))
    rbx = non_robust.df2 / (1 + non_robust.df2)
    funval_low = objective(rbx)
    if funval_low >= 0:
        df2 = non_robust.df2
    else:
        root = zeroin(
            objective,
            rbx,
            1.0,
            f_lower=funval_low,
            f_upper=funval_inf,
            tol=1e-8,
        )
        df2 = link_inverse(root)
    mom_mean, _ = _winsorized_moments(df1_value, df2, tail)
    ztrend_corrected = ztrend + zwmean - mom_mean
    s20 = np.exp(np.broadcast_to(ztrend_corrected, (n,)))
    zresid = z - ztrend_corrected
    fstat = np.exp(zresid)
    log_tail_p = _log_pf_upper(fstat, df1_value, df2)
    ranks = stats.rankdata(fstat)
    log_empirical = np.log(n - ranks + 0.5) - math.log(n)
    log_not_outlier = np.minimum(log_tail_p - log_empirical, 0)
    prob_not_outlier = np.exp(log_not_outlier)
    prob_outlier = -np.expm1(log_not_outlier)
    if np.any(log_not_outlier < 0):
        min_log_tail_p = float(np.min(log_tail_p))
        if min_log_tail_p == -math.inf:
            df2_shrunk = prob_not_outlier * df2
        else:
            df2_outlier = math.log(0.5) / min_log_tail_p * df2
            new_log_tail_p = float(
                _log_pf_upper(np.array([np.max(fstat)]), df1_value, df2_outlier)[0]
            )
            df2_outlier = math.log(0.5) / new_log_tail_p * df2_outlier
            df2_shrunk = prob_not_outlier * df2 + prob_outlier * df2_outlier
        order = np.argsort(log_tail_p, kind="stable")
        ordered = df2_shrunk[order]
        running = np.cumsum(ordered) / np.arange(1, n + 1)
        imin = int(np.argmin(running))
        ordered[: imin + 1] = running[imin]
        df2_shrunk = df2_shrunk.copy()
        df2_shrunk[order] = np.maximum.accumulate(ordered)
    else:
        df2_shrunk = np.full(n, df2)
    return FDistFit(
        scale=np.asarray(s20, dtype=np.float64), df2=df2, df2_shrunk=df2_shrunk
    )


def _qf_from_log_upper(log_p: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        return np.asarray(stats.chi2.isf(np.exp(log_p), df1) / df1)
    return np.asarray(stats.f.isf(np.exp(log_p), df1, df2))


def _qf_from_log_lower(log_p: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if math.isinf(df2):
        return np.asarray(stats.chi2.ppf(np.exp(log_p), df1) / df1)
    return np.asarray(stats.f.ppf(np.exp(log_p), df1, df2))


def _squeeze(
    var: np.ndarray, df: np.ndarray, var_prior: np.ndarray, df_prior: np.ndarray
) -> np.ndarray:
    if np.all(np.isfinite(df_prior)):
        return np.asarray((df * var + df_prior * var_prior) / (df + df_prior))
    n = var.size
    var_post = np.broadcast_to(var_prior, (n,)).astype(np.float64).copy()
    if np.min(df_prior) > 1e100:
        return var_post
    finite = np.isfinite(df_prior)
    d = df[finite] if df.size > 1 else df
    var_post[finite] = (d * var[finite] + df_prior[finite] * var_post[finite]) / (
        d + df_prior[finite]
    )
    return var_post


def squeeze_var(
    var: np.ndarray,
    df: np.ndarray | float,
    covariate: np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> SqueezedVariances:
    """Return ``squeezeVar(var, df, covariate, robust=robust)``.

    limma takes its legacy path when every positive df is equal, which is the
    only case implemented here.

    Raises:
        ValueError: If the positive residual df differ between rows.
    """
    values = np.asarray(var, dtype=np.float64).copy()
    n = values.size
    if n == 0:
        raise ValueError("var is empty")
    df_values = np.asarray(df, dtype=np.float64)
    if n < 3:
        return SqueezedVariances(
            df_prior=np.zeros(n), var_prior=values.copy(), var_post=values.copy()
        )
    if df_values.ndim > 0 and df_values.size > 1:
        values[df_values == 0] = 0
    positive = df_values[df_values > 0]
    if np.min(positive) != np.max(positive):
        raise ValueError(
            "squeeze_var supports equal residual df only; limma's unequal-df "
            "method (fitFDistUnequalDF1) is not ported"
        )
    if robust:
        fit = fit_f_dist_robustly(values, df_values, covariate, winsor_tail_p)
        assert fit.df2_shrunk is not None
        df_prior = np.asarray(fit.df2_shrunk, dtype=np.float64)
    else:
        fit = fit_f_dist(values, df_values, covariate)
        df_prior = np.asarray(fit.df2, dtype=np.float64)
    if np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior df")
    df_prior_vector = np.broadcast_to(df_prior, (n,)).astype(np.float64)
    var_prior = np.broadcast_to(fit.scale, (n,)).astype(np.float64)
    var_post = _squeeze(
        values,
        np.broadcast_to(df_values, (n,)).astype(np.float64),
        var_prior,
        df_prior_vector,
    )
    return SqueezedVariances(
        df_prior=df_prior_vector, var_prior=var_prior, var_post=var_post
    )


def _f_statistic(t: np.ndarray, cov_coefficients: np.ndarray) -> tuple[np.ndarray, int]:
    n_tests = t.shape[1]
    if n_tests == 1:
        return t[:, 0] ** 2, 1
    cov = cov_coefficients.copy()
    diagonal = np.diag(cov).copy()
    zero = diagonal == 0
    if np.any(zero):
        cov[np.diag_indices_from(cov)] = np.where(zero, 1.0, diagonal)
    cor = _cov_to_cor(cov)
    eigenvalues, eigenvectors = np.linalg.eigh(cor)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    rank = int(np.sum(eigenvalues / eigenvalues[0] > 1e-8))
    q = eigenvectors[:, :rank] / np.sqrt(eigenvalues[:rank]) / math.sqrt(rank)
    return np.sum((t @ q) ** 2, axis=1), rank


def e_bayes(
    fit: LinearModelFit,
    *,
    trend: bool = False,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> ModeratedFit:
    """Return the moderated statistics of ``eBayes(fit, trend, robust)``."""
    if np.max(fit.df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    if not np.any(np.isfinite(fit.sigma)):
        raise ValueError("No finite residual standard deviations")
    covariate = fit.amean if trend else None
    squeezed = squeeze_var(
        fit.sigma**2,
        fit.df_residual,
        covariate=covariate,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    t = fit.coefficients / fit.stdev_unscaled / np.sqrt(squeezed.var_post)[:, None]
    df_total = fit.df_residual + squeezed.df_prior
    df_pooled = float(np.nansum(fit.df_residual))
    df_total = np.minimum(df_total, df_pooled)
    p_value = 2 * stats.t.sf(np.abs(t), df_total[:, None])
    f_statistic, df1 = _f_statistic(t, fit.cov_coefficients)
    df2 = squeezed.df_prior + fit.df_residual
    f_p_value = np.where(
        np.isinf(df2),
        stats.chi2.sf(df1 * f_statistic, df1),
        stats.f.sf(f_statistic, df1, np.where(np.isinf(df2), 1.0, df2)),
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
        f_df1=df1,
        f_df2=df2,
        f_p_value=f_p_value,
    )
