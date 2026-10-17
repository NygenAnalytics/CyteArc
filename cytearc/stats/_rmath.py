"""Exact ports of the R numerical routines that limma calls.

limma's results depend on the stopping rule of R's root finder, on R's
``lowess``, and on R's natural spline basis. Matching them closely needs the
same algorithms, not merely equivalent ones, so this module ports each
routine from the R sources:

- :func:`zeroin` is ``R_zeroin2`` behind ``stats::uniroot``.
- :func:`lowess` is ``clowess`` behind ``stats::lowess``.
- :func:`natural_spline_basis` is ``splines::ns`` with an intercept.
"""

import math
from collections.abc import Callable

import numpy as np
from numba import njit
from scipy.interpolate import BSpline
from scipy.special import digamma

__all__ = [
    "logmdigamma",
    "lowess",
    "natural_spline_basis",
    "optimize_tolerance",
    "trimmed_mean",
    "zeroin",
]

_DBL_EPSILON = float(np.finfo(np.float64).eps)


def optimize_tolerance() -> float:
    """Return the default tolerance of ``stats::optimize``."""
    return float(_DBL_EPSILON**0.25)


def logmdigamma(x: np.ndarray | float) -> np.ndarray:
    """Return ``log(x) - digamma(x)`` as limma's ``logmdigamma`` does."""
    values = np.asarray(x, dtype=np.float64)
    return np.asarray(np.log(values) - digamma(values), dtype=np.float64)


def zeroin(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    f_lower: float | None = None,
    f_upper: float | None = None,
    tol: float | None = None,
    maxiter: int = 1000,
) -> float:
    """Find a root of ``f`` in ``[lower, upper]`` exactly as ``stats::uniroot``.

    Raises:
        ValueError: If the function values at the ends do not have opposite signs.
    """
    if tol is None:
        tol = optimize_tolerance()
    a = float(lower)
    b = float(upper)
    fa = float(f(a)) if f_lower is None else float(f_lower)
    fb = float(f(b)) if f_upper is None else float(f_upper)
    if fa * fb > 0.0:
        raise ValueError("f() values at end points not of opposite sign")
    c = a
    fc = fa
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    remaining = maxiter + 1
    while remaining > 0:
        remaining -= 1
        prev_step = b - a
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol_act = 2.0 * _DBL_EPSILON * abs(b) + tol / 2.0
        new_step = (c - b) / 2.0
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2.0) and p < abs(
                prev_step * q / 2.0
            ):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act
        a = b
        fa = fb
        b += new_step
        fb = float(f(b))
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa
    return b


def trimmed_mean(values: np.ndarray, trim: float) -> float:
    """Return ``mean(values, trim=trim)`` with R's trimming rule."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if trim <= 0 or n == 0:
        return float(np.mean(x))
    if trim >= 0.5:
        return float(np.median(x))
    lo = int(math.floor(n * trim))
    return float(np.mean(x[lo : n - lo]))


@njit(cache=True)
def _lowest(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    xs: float,
    nleft: int,
    nright: int,
    w: np.ndarray,
    use_rw: bool,
    rw: np.ndarray,
) -> tuple[float, bool]:
    # Indices are zero based; nleft and nright are inclusive.
    rng = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h
    a = 0.0
    j = nleft
    while j < n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                q = r / h
                q = 1.0 - q * q * q
                w[j] = q * q * q
            if use_rw:
                w[j] *= rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1
    nrt = j - 1
    if a <= 0.0:
        return 0.0, False
    for j in range(nleft, nrt + 1):
        w[j] /= a
    if h > 0.0:
        a = 0.0
        for j in range(nleft, nrt + 1):
            a += w[j] * x[j]
        b = xs - a
        c = 0.0
        for j in range(nleft, nrt + 1):
            c += w[j] * (x[j] - a) * (x[j] - a)
        if math.sqrt(c) > 0.001 * rng:
            b /= c
            for j in range(nleft, nrt + 1):
                w[j] *= b * (x[j] - a) + 1.0
    ys = 0.0
    for j in range(nleft, nrt + 1):
        ys += w[j] * y[j]
    return ys, True


@njit(cache=True)
def _clowess(
    x: np.ndarray, y: np.ndarray, f: float, nsteps: int, delta: float
) -> np.ndarray:
    n = x.shape[0]
    ys = np.zeros(n)
    if n < 2:
        ys[0] = y[0]
        return ys
    rw = np.zeros(n)
    res = np.zeros(n)
    ns = max(2, min(n, int(f * n + 1e-7)))
    iteration = 1
    while iteration <= nsteps + 1:
        nleft = 0
        nright = ns - 1
        last = -1
        i = 0
        while True:
            if nright < n - 1:
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                    continue
            value, ok = _lowest(x, y, n, x[i], nleft, nright, res, iteration > 1, rw)
            ys[i] = value if ok else y[i]
            if last < i - 1:
                # The first estimated point has last == -1 only when i == 0.
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        for k in range(n):
            res[k] = y[k] - ys[k]
        sc = 0.0
        for k in range(n):
            sc += abs(res[k])
        sc /= n
        if iteration > nsteps:
            break
        for k in range(n):
            rw[k] = abs(res[k])
        sorted_rw = np.sort(rw)
        m1 = n // 2
        if n % 2 == 0:
            m2 = n - m1 - 1
            cmad = 3.0 * (sorted_rw[m1] + sorted_rw[m2])
        else:
            cmad = 6.0 * sorted_rw[m1]
        if cmad < 1e-7 * sc:
            break
        c9 = 0.999 * cmad
        c1 = 0.001 * cmad
        for k in range(n):
            r = abs(res[k])
            if r <= c1:
                rw[k] = 1.0
            elif r <= c9:
                q = r / cmad
                q = 1.0 - q * q
                rw[k] = q * q
            else:
                rw[k] = 0.0
        iteration += 1
    return ys


def lowess(
    x: np.ndarray,
    y: np.ndarray,
    *,
    f: float = 2.0 / 3.0,
    iterations: int = 3,
    delta: float | None = None,
) -> np.ndarray:
    """Return ``stats::lowess(x, y, f, iter)$y`` in the original order of ``x``."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    order = np.argsort(x_values, kind="stable")
    xs = np.ascontiguousarray(x_values[order])
    if delta is None:
        delta = 0.01 * float(xs[-1] - xs[0]) if xs.size else 0.0
    fitted_sorted = _clowess(
        xs, np.ascontiguousarray(y_values[order]), float(f), int(iterations), delta
    )
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return fitted


def _bspline_basis(knots: np.ndarray, x: np.ndarray, derivative: int) -> np.ndarray:
    n_basis = knots.size - 4
    basis = np.empty((x.size, n_basis), dtype=np.float64)
    for column in range(n_basis):
        coefficients = np.zeros(n_basis)
        coefficients[column] = 1.0
        spline = BSpline(knots, coefficients, 3, extrapolate=True)
        if derivative:
            spline = spline.derivative(derivative)
        basis[:, column] = spline(x)
    # R's splineDesign closes the right boundary on the last basis function.
    if derivative == 0:
        at_right = x == knots[-1]
        if np.any(at_right):
            basis[at_right] = 0.0
            basis[at_right, -1] = 1.0
    return basis


def natural_spline_basis(
    x: np.ndarray,
    df: int,
    *,
    new_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``splines::ns(x, df=df, intercept=TRUE)`` and, optionally, its prediction.

    The basis spans the same space as R's, so least-squares fits on it match
    R's fitted values. Points of ``new_x`` beyond the boundary knots are
    extrapolated linearly, as ``predict.ns`` does.
    """
    values = np.asarray(x, dtype=np.float64)
    n_inner = int(df) - 2
    if n_inner < 0:
        n_inner = 0
    boundary = np.array([values.min(), values.max()])
    if n_inner > 0:
        probabilities = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        inner = np.quantile(values, probabilities)
    else:
        inner = np.empty(0)
    knots = np.sort(np.concatenate([np.repeat(boundary, 4), inner]))
    const = _bspline_basis(knots, boundary, 2)
    q_full, _ = np.linalg.qr(const.T, mode="complete")
    null_space = q_full[:, 2:]

    def evaluate(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        basis = np.empty((points.size, knots.size - 4))
        below = points < boundary[0]
        above = points > boundary[1]
        inside = ~(below | above)
        if np.any(inside):
            basis[inside] = _bspline_basis(knots, points[inside], 0)
        for mask, edge in ((below, boundary[0]), (above, boundary[1])):
            if np.any(mask):
                edge_point = np.array([edge])
                value = _bspline_basis(knots, edge_point, 0)
                slope = _bspline_basis(knots, edge_point, 1)
                basis[mask] = value + (points[mask] - edge)[:, None] * slope
        return basis @ null_space

    fitted_basis = evaluate(values)
    predicted = None if new_x is None else evaluate(np.asarray(new_x))
    return fitted_basis, predicted
