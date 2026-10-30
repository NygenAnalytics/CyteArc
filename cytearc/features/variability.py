"""Highly variable feature selection.

``features.hvgs`` scores how variable each feature is among the selected
cells, keeps the candidates that pass its filters, and selects the ``top_n``
candidates with the highest scores. Three flavors score a feature:

- ``"cytearc"``: the variance of library-size normalized values divided by an
  expression-dependent background, the lower quartile of the variances of
  features with a similar mean (:func:`fit_lowess`).
- ``"seurat_v3"``: the variance-stabilizing transformation of Stuart et al.
  (2019), Cell 177:1888-1902. A local quadratic regression of log10 variance
  on log10 mean of the raw counts (:func:`fit_loess`) gives each feature an
  expected variance; counts are standardized with it, clipped at
  ``sqrt(n)`` standard deviations above the mean, and the score is the
  variance of the standardized values (:func:`seurat_v3_expected_variance`,
  :func:`seurat_v3_standardized_variance`).
- ``"pearson_residuals"``: the variance of analytic Pearson residuals of a
  negative binomial model with a fixed overdispersion ``theta``, Lause,
  Berens and Kobak (2021), Genome Biology 22:258
  (:func:`pearson_residuals`).

With batches, every batch is scored on its own, each batch ranks the
candidates, and :func:`combine_batch_rankings` selects features by the
number of batches that rank them among their ``top_n`` and then by their
median rank, the order of Seurat's ``SelectIntegrationFeatures``.
"""

from typing import Any, Literal, NamedTuple, cast

import numpy as np
import pandas as pd

from ..utils.arguments import float_argument, integer_argument
from ..utils.arrays import regex_match_mask, within_bounds
from ..utils.logging import logger
from ..utils.numba import njit
from .gene_families import GENE_FAMILY_PATTERNS

__all__ = [
    "DEFAULT_HVG_BLACKLIST",
    "DEFAULT_LOESS_SPAN",
    "DEFAULT_PEARSON_THETA",
    "HVG_FLAVORS",
    "HVG_FLAVOR_OPTION_NAMES",
    "HVG_MIN_CELLS_CAP",
    "HVG_MIN_CELLS_FLOOR",
    "HVG_OPTION_NAMES",
    "BatchRanking",
    "HvgFlavorOptions",
    "HvgOptions",
    "combine_batch_rankings",
    "default_hvg_min_cells",
    "default_hvg_ubiquity_limit",
    "fit_loess",
    "fit_lowess",
    "hvg_candidates",
    "hvg_flavor_options",
    "hvg_options",
    "pearson_residual_clip",
    "pearson_residuals",
    "resolve_hvg_detection_bounds",
    "select_batch_highly_variable_features",
    "select_highly_variable_features",
    "seurat_v3_clip_values",
    "seurat_v3_expected_variance",
    "seurat_v3_standardized_variance",
]

_ADAPTIVE_MIN_SUPPORT = 50
_ADAPTIVE_QUANTILE = 0.25
# Largest gradient entry at which a local fit whose line search stopped counts as
# converged. Gradient entries of the weighted data term never exceed 0.75, so one
# absolute bound suits every window.
_STALLED_FIT_GRADIENT = float(np.sqrt(np.finfo(np.float64).eps))

# Excludes every registered gene family. Patterns match names case-insensitively
# unless a scoped regex flag overrides it.
DEFAULT_HVG_BLACKLIST = "|".join(GENE_FAMILY_PATTERNS.values())

# The flavors of features.hvgs; the first is the default.
HVG_FLAVORS: tuple[str, ...] = ("cytearc", "seurat_v3", "pearson_residuals")
# The span of the seurat_v3 trend and the overdispersion of Pearson residuals.
DEFAULT_LOESS_SPAN = 0.3
DEFAULT_PEARSON_THETA = 100.0
# The default min_cells of the cytearc flavor is 1% of the selected cells,
# at least HVG_MIN_CELLS_FLOOR and at most HVG_MIN_CELLS_CAP.
HVG_MIN_CELLS_CAP = 20
HVG_MIN_CELLS_FLOOR = 3
_MIN_CELLS_DIVISOR = 100
# The default max_cells of the cytearc flavor excludes features detected in
# more than 99.5% of the selected cells, computed in integers as 199/200.
_UBIQUITY_NUMERATOR = 199
_UBIQUITY_DENOMINATOR = 200


def _fit_local_quantile(
    log_means: np.ndarray,
    log_variances: np.ndarray,
    noise_residual: np.ndarray,
    center: float,
    bandwidth: float,
    minimum_bandwidth: float,
) -> float:
    from scipy.optimize import minimize

    reference: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(2):
        keep = np.abs(log_means - center) < bandwidth
        relative = (log_means[keep] - center) / bandwidth
        weight = 0.5 + 0.5 * (1 - np.abs(relative) ** 3) ** 3
        if reference is not None:
            weight *= np.interp(log_means[keep], reference[0], reference[1])
        weight /= weight.sum()
        degree = min(2, max(1, len(np.unique(relative)) - 3))
        design = np.stack([relative**power for power in range(degree + 1)], axis=1)

        # Limit the influence of isolated genes at the ends of a fitting window.
        for _ in range(3):
            inverse = np.linalg.pinv(design.T @ (weight[:, None] * design))
            leverage = weight * np.sum((design @ inverse) * design, axis=1)
            limit = 1.25 * (degree + 1) * (weight @ weight)
            weight *= np.minimum(1, limit / np.maximum(leverage, 1e-15))
            weight /= weight.sum()

        origin = np.median(log_variances[keep])
        observed = log_variances[keep] - origin
        initial = np.linalg.lstsq(
            design * np.sqrt(weight[:, None]), observed * np.sqrt(weight), rcond=None
        )[0]
        local_noise = noise_residual[keep]
        noise = np.median(local_noise[np.isfinite(local_noise)]) if degree == 2 else 0.0
        penalty = noise * np.sqrt(weight @ weight)

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            residual = observed - design @ coefficients
            absolute = np.sqrt(residual**2 + 1e-8)
            value = weight @ (0.5 * absolute + (_ADAPTIVE_QUANTILE - 0.5) * residual)
            gradient = -design.T @ (
                weight * (0.5 * residual / absolute + _ADAPTIVE_QUANTILE - 0.5)
            )
            if degree == 2:
                value += penalty * coefficients[2] ** 2
                gradient[2] += 2 * penalty * coefficients[2]
            return float(value), gradient

        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 300, "maxls": 50},
        )
        # The line search stops (status 2) when the fit already sits at the optimum
        # to machine precision, which depends on the scipy version and BLAS
        # threading. Accept that stop only once the gradient has vanished.
        stalled = (
            result.status == 2 and np.max(np.abs(result.jac)) <= _STALLED_FIT_GRADIENT
        )
        if not (result.success or stalled) or not np.all(np.isfinite(result.x)):
            raise ValueError(f"Adaptive variance trend fit failed: {result.message}")

        # Continue linearly outside the interior support instead of extrapolating
        # poorly constrained curvature across an isolated endpoint.
        edge = np.clip(0.0, *np.quantile(relative, [0.1, 0.9]))
        prediction = result.x[0] + origin
        if degree == 2:
            prediction -= result.x[2] * edge**2
        if reference is not None or minimum_bandwidth >= bandwidth or degree < 2:
            break

        residual = observed - design @ result.x
        model_noise = np.median(np.abs(residual - np.median(residual)))
        if model_noise <= max(2 * noise, 1e-4):
            break

        # Narrow only when smooth curvature exceeds local scatter. Carry an
        # outlier guard from the wider window into the smaller fit.
        unique, groups = np.unique(log_means[keep], return_inverse=True)
        values = np.array(
            [np.median(observed[groups == index]) for index in range(len(unique))]
        )
        slopes = np.diff(values) / np.diff(unique)
        curvature = np.diff(slopes) / (unique[2:] - unique[:-2])
        curvature = np.abs(np.r_[curvature[0], curvature, curvature[-1]])
        scale = max(3 * np.median(curvature), 1e-8)
        robust_weight = np.minimum(1, (scale / np.maximum(curvature, 1e-15)) ** 2)
        reference = (log_means[keep], robust_weight[groups])
        bandwidth = minimum_bandwidth

    return float(prediction)


def _trend_inputs(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float means and variances and where both are finite and positive.

    Only those genes enter a trend fit; the others receive a corrected value of
    zero.
    """
    means = np.asarray(a, dtype=float)
    variances = np.asarray(b, dtype=float)
    if means.ndim != 1 or variances.ndim != 1 or means.shape != variances.shape:
        raise ValueError("LOWESS inputs must be one-dimensional arrays of equal length")
    valid = np.isfinite(means) & np.isfinite(variances) & (means > 0) & (variances > 0)
    return means, variances, valid


def _fit_lowess_adaptive(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int,
    lowess_frac: float,
) -> np.ndarray:
    """Fit the adaptive trend; ``fit_lowess`` has checked the arguments."""
    means, variances, valid = _trend_inputs(a, b)
    corrected = np.zeros(means.shape, dtype=float)
    if not valid.any():
        return corrected
    if valid.sum() < 3:
        raise ValueError(
            "At least three genes with positive finite means and variances "
            "are needed to estimate an adaptive variance trend"
        )

    log_means = np.log(means[valid])
    log_variances = np.log(variances[valid])
    order = np.argsort(log_means, kind="stable")
    sorted_means = log_means[order]
    sorted_variances = log_variances[order]

    unique, starts, counts = np.unique(
        sorted_means, return_index=True, return_counts=True
    )
    if len(unique) == 1:
        correction = np.full(
            log_means.shape, np.quantile(log_variances, _ADAPTIVE_QUANTILE)
        )
    else:
        span = np.ptp(unique)
        rank = (starts + (counts - 1) / 2) / (len(order) - 1)
        coordinate = 0.5 * rank + 0.5 * (unique - unique[0]) / span
        coordinate = (coordinate - coordinate[0]) / np.ptp(coordinate)
        labels = np.repeat(
            np.minimum(np.floor(n_bins * coordinate).astype(int), n_bins - 1), counts
        )
        boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(order)]
        centers = [
            np.median(sorted_means[start:end])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        # Fit both endpoints instead of clamping the sparse tail to an inner bin.
        anchors = np.unique(np.r_[unique[0], centers, unique[-1]])

        median_variance = sorted_variances[starts].copy()
        for index in np.flatnonzero(counts > 1):
            start = starts[index]
            median_variance[index] = np.median(
                sorted_variances[start : start + counts[index]]
            )
        neighbor_prediction = np.full(len(unique), np.nan)
        if len(unique) > 2:
            position = (unique[1:-1] - unique[:-2]) / (unique[2:] - unique[:-2])
            neighbor_prediction[1:-1] = median_variance[:-2] + position * (
                median_variance[2:] - median_variance[:-2]
            )
        noise_residual = np.abs(
            sorted_variances - np.repeat(neighbor_prediction, counts)
        )

        fitted = np.empty(len(anchors))
        support = min(_ADAPTIVE_MIN_SUPPORT, len(order))
        minimum_support = min(8, len(order))
        for anchor_index, center in enumerate(anchors):
            distances = np.abs(sorted_means - center)
            bandwidth = max(
                lowess_frac * span,
                np.partition(distances, support - 1)[support - 1] * 1.05,
            )
            if bandwidth == 0:
                bandwidth = distances[distances > 0].min() * 1.05
            minimum_bandwidth = max(
                lowess_frac * span,
                np.partition(distances, minimum_support - 1)[minimum_support - 1]
                * 1.05,
            )
            if minimum_bandwidth == 0:
                minimum_bandwidth = distances[distances > 0].min() * 1.05
            fitted[anchor_index] = _fit_local_quantile(
                sorted_means,
                sorted_variances,
                noise_residual,
                center,
                bandwidth,
                minimum_bandwidth,
            )
        correction = np.interp(log_means, anchors, fitted)

    with np.errstate(over="ignore", under="ignore"):
        corrected[valid] = np.exp(log_variances - correction)
    if not np.all(np.isfinite(corrected[valid]) & (corrected[valid] > 0)):
        raise ValueError(
            "Adaptive variance correction produced nonfinite or zero scores"
        )
    return corrected


_BIN_STRATEGIES = ("adaptive", "fixed")


def _lowess_frac(value: object) -> float:
    """Return a checked ``lowess_frac`` as a float."""
    resolved = float_argument(value, "lowess_frac")
    if not 0 <= resolved <= 1:
        raise ValueError("lowess_frac must be between 0 and 1")
    return resolved


def _bin_strategy(value: object) -> Literal["fixed", "adaptive"]:
    """Return a checked ``bin_strategy`` as a Python string."""
    if not isinstance(value, str) or value not in _BIN_STRATEGIES:
        raise ValueError("bin_strategy must be either 'fixed' or 'adaptive'")
    return cast(Literal["fixed", "adaptive"], str(value))


def fit_lowess(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int,
    lowess_frac: float,
    *,
    bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
) -> np.ndarray:
    """Divide variance by an expression-dependent background.

    Adaptive fits use local quantile regression: ``n_bins`` controls evaluation
    density and ``lowess_frac`` is the window radius as a fraction of log-mean
    range. Minimum support is 50 genes, or 8 where smooth curvature warrants it.
    Smaller inputs use all available genes. The background is the lower quartile.
    Fixed fits use LOWESS over the minimum-variance gene in each equal-width bin.
    Both strategies fit only genes whose mean and variance are finite and
    positive; every other gene receives zero.
    """
    n_bins = integer_argument(n_bins, "n_bins", minimum=1)
    lowess_frac = _lowess_frac(lowess_frac)
    if _bin_strategy(bin_strategy) == "adaptive":
        from threadpoolctl import threadpool_limits

        # The adaptive fit solves many tiny least-squares problems, where extra
        # BLAS threads cost far more in synchronization than they save.
        with threadpool_limits(limits=1, user_api="blas"):
            return _fit_lowess_adaptive(a, b, n_bins, lowess_frac)

    from statsmodels.nonparametric.smoothers_lowess import lowess

    means, variances, valid = _trend_inputs(a, b)
    corrected = np.zeros(means.shape, dtype=float)
    if not valid.any():
        return corrected
    stats = pd.DataFrame({"a": means[valid], "b": variances[valid]}).apply(np.log)
    bin_edges = np.histogram(stats.a, bins=n_bins)[1]
    bin_edges[-1] += 0.1
    bin_idx: list[list[Any]] = []
    for index in range(n_bins):
        idx = pd.Series(
            (stats.a >= bin_edges[index]) & (stats.a < bin_edges[index + 1])
        )
        if sum(idx) > 0:
            bin_idx.append(list(idx[idx].index))
    bin_vals: list[list[float]] = []
    for idx in bin_idx:
        temp_stat = stats.reindex(idx)
        temp_gene = temp_stat.idxmin().b
        bin_vals.append([temp_stat.b[temp_gene], temp_stat.a[temp_gene]])
    bin_array = np.array(bin_vals).T
    bin_cor_fac = lowess(
        bin_array[0],
        bin_array[1],
        return_sorted=False,
        frac=lowess_frac,
        it=100,
    ).T
    fixed_var: dict[Any, float] = {}
    for correction, indices in zip(bin_cor_fac, bin_idx):
        for idx in indices:
            fixed_var[idx] = np.e ** (stats.b[idx] - correction)
    corrected[valid] = [fixed_var[index] for index in range(len(stats))]
    return corrected


@njit(cache=True, nogil=True)
def _fit_loess_points(
    x: np.ndarray,
    y: np.ndarray,
    q: int,
    degree: int,
    points: np.ndarray,
    values: np.ndarray,
    slopes: np.ndarray,
) -> None:
    """Write the local polynomial fit and its slope at each ascending point.

    ``x`` is sorted ascending. The ``q`` data points nearest to a point
    ``x0`` are a contiguous run of them, and ``h`` is the distance to the
    farthest. A data point at distance ``d < h`` has the tricube weight
    ``(1 - (d / h)**3)**3``; the others weigh zero. The weighted
    least-squares polynomial of ``degree`` in ``(x - x0) / h`` is built from
    orthogonal polynomials under the weights; its value and derivative at
    ``x0`` are the fit and the slope. A polynomial whose weighted norm
    vanishes, as when the weighted points take fewer distinct values than the
    degree needs, is left out, so the fit lowers its degree instead of
    failing. When all ``q`` points sit at ``x0``, the fit is the mean of every
    data point at ``x0`` and the slope is 0.
    """
    n = x.shape[0]
    left = 0
    for i in range(points.shape[0]):
        x0 = points[i]
        if i > 0 and x0 == points[i - 1]:
            # The fit depends on x0 alone.
            values[i] = values[i - 1]
            slopes[i] = slopes[i - 1]
            continue
        while left + q < n and x[left + q] - x0 < x0 - x[left]:
            left += 1
        h = max(x0 - x[left], x[left + q - 1] - x0)
        if h <= 0.0:
            # The window never starts past the first data point at or above
            # x0, so here it starts at the first point tied with x0.
            end = left + q
            while end < n and x[end] == x0:
                end += 1
            total = 0.0
            for j in range(left, end):
                total += y[j]
            values[i] = total / (end - left)
            slopes[i] = 0.0
            continue
        sum_w = 0.0
        sum_wu = 0.0
        sum_wy = 0.0
        for j in range(left, left + q):
            ratio = abs(x[j] - x0) / h
            if ratio >= 1.0:
                continue
            weight = (1.0 - ratio * ratio * ratio) ** 3
            sum_w += weight
            sum_wu += weight * (x[j] - x0) / h
            sum_wy += weight * y[j]
        if sum_w <= 0.0:
            # Every point of the window sits at distance h, which happens only
            # with fewer than three points or a vertex midway between two.
            total = 0.0
            for j in range(left, left + q):
                total += y[j]
            values[i] = total / q
            slopes[i] = 0.0
            continue
        value = sum_wy / sum_w
        slope = 0.0
        if degree >= 1:
            alpha1 = sum_wu / sum_w
            norm1 = 0.0
            dot1 = 0.0
            moment1 = 0.0
            for j in range(left, left + q):
                ratio = abs(x[j] - x0) / h
                if ratio >= 1.0:
                    continue
                weight = (1.0 - ratio * ratio * ratio) ** 3
                u = (x[j] - x0) / h
                p1 = u - alpha1
                norm1 += weight * p1 * p1
                dot1 += weight * p1 * y[j]
                moment1 += weight * u * p1 * p1
            if norm1 > 1e-12 * sum_w:
                value -= alpha1 * dot1 / norm1
                slope += dot1 / norm1
                if degree >= 2:
                    alpha2 = moment1 / norm1
                    beta1 = norm1 / sum_w
                    norm2 = 0.0
                    dot2 = 0.0
                    for j in range(left, left + q):
                        ratio = abs(x[j] - x0) / h
                        if ratio >= 1.0:
                            continue
                        weight = (1.0 - ratio * ratio * ratio) ** 3
                        u = (x[j] - x0) / h
                        p2 = (u - alpha2) * (u - alpha1) - beta1
                        norm2 += weight * p2 * p2
                        dot2 += weight * p2 * y[j]
                    if norm2 > 1e-12 * sum_w:
                        value += (alpha2 * alpha1 - beta1) * dot2 / norm2
                        slope -= (alpha1 + alpha2) * dot2 / norm2
        values[i] = value
        slopes[i] = slope / h


# Cells of the k-d tree that interpolation evaluates hold at most this fraction
# of span * n points, the default ``cell`` of R's ``loess.control``.
_LOESS_CELL = 0.2


def _loess_vertices(x: np.ndarray, max_points: int) -> np.ndarray:
    """Return the ascending vertices of a one-dimensional k-d tree over sorted ``x``.

    The root cell spans the data widened by 0.5% of their range on each side.
    A cell with more than ``max_points`` points splits at the median point,
    moved to the nearest position where it does not separate tied values, so
    that the lower cell ends at that value; a cell whose split would fall on
    one of its vertices stays whole.
    """
    low, high = float(x[0]), float(x[-1])
    margin = 0.005 * max(high - low, 1e-10 * max(abs(low), abs(high)) + 1e-30)
    vertices = [low - margin, high + margin]
    cells = [(0, len(x) - 1, low - margin, high + margin)]
    while cells:
        first, last, lower, upper = cells.pop()
        if last - first + 1 <= max_points:
            continue
        middle = (first + last) // 2
        offset = 0
        while first <= middle + offset < last:
            if x[middle + offset] != x[middle + offset + 1]:
                middle += offset
                break
            offset = -offset
            if offset >= 0:
                offset += 1
        split = float(x[middle])
        if split in (lower, upper):
            continue
        vertices.append(split)
        cells.append((first, middle, lower, split))
        cells.append((middle + 1, last, split, upper))
    return np.unique(np.asarray(vertices, dtype=np.float64))


def fit_loess(
    x: np.ndarray,
    y: np.ndarray,
    *,
    span: float = DEFAULT_LOESS_SPAN,
    degree: int = 2,
    surface: Literal["interpolate", "direct"] = "interpolate",
) -> np.ndarray:
    """Fit a local polynomial regression of ``y`` on ``x`` at every ``x``.

    This is local regression in the sense of Cleveland and Devlin (1988),
    Journal of the American Statistical Association 83:596-610, without
    robustness iterations: the ``floor(span * n)`` points nearest to a point,
    at least ``degree + 1`` of them, carry tricube weights of their distance
    relative to the farthest of them, and a weighted least-squares polynomial
    of ``degree`` gives the fitted value. ``surface="direct"`` computes this
    fit at every point. ``surface="interpolate"``, the default of R's
    ``loess`` and of scikit-misc, follows Cleveland and Grosse (1991),
    Statistics and Computing 1:47-62: it computes the fit and its slope at
    the vertices of a k-d tree whose cells hold at most
    ``floor(0.2 * span * n)`` points, and blends them between vertices with
    cubic Hermite polynomials.

    Args:
        x: One predictor value per point.
        y: One response value per point.
        span: The fraction of the points that each local fit uses, greater
            than 0 and at most 1.
        degree: The local polynomial degree, 0, 1, or 2.
        surface: ``"interpolate"`` or ``"direct"``.

    Returns:
        The float64 fitted value at each point, in the order of ``x``.

    Raises:
        ValueError: If the inputs are not finite one-dimensional arrays of
            equal length, or ``span``, ``degree``, or ``surface`` is out of
            range.
    """
    predictor = np.asarray(x, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    if predictor.ndim != 1 or predictor.shape != response.shape:
        raise ValueError("LOESS inputs must be one-dimensional arrays of equal length")
    if not (np.isfinite(predictor).all() and np.isfinite(response).all()):
        raise ValueError("LOESS inputs must be finite")
    span = _loess_span(span)
    degree = integer_argument(degree, "degree", minimum=0, maximum=2)
    if surface not in ("interpolate", "direct"):
        raise ValueError("surface must be 'interpolate' or 'direct'")
    n_points = predictor.shape[0]
    fitted = np.empty(n_points, dtype=np.float64)
    if n_points == 0:
        return fitted
    neighbours = min(n_points, max(int(np.floor(span * n_points)), degree + 1))
    order = np.argsort(predictor, kind="stable")
    sorted_x = np.ascontiguousarray(predictor[order])
    sorted_y = np.ascontiguousarray(response[order])
    points = (
        sorted_x
        if surface == "direct"
        else _loess_vertices(
            sorted_x, max(1, int(np.floor(n_points * span * _LOESS_CELL)))
        )
    )
    values = np.empty(points.shape[0], dtype=np.float64)
    slopes = np.empty(points.shape[0], dtype=np.float64)
    _fit_loess_points(sorted_x, sorted_y, neighbours, degree, points, values, slopes)
    if surface == "direct":
        fitted[order] = values
        return fitted
    # Cubic Hermite blending of the vertex values and slopes in each cell.
    cell = np.clip(
        np.searchsorted(points, predictor, side="right") - 1, 0, len(points) - 2
    )
    width = points[cell + 1] - points[cell]
    t = (predictor - points[cell]) / width
    fitted[:] = (
        (1 - t) ** 2 * (1 + 2 * t) * values[cell]
        + t**2 * (3 - 2 * t) * values[cell + 1]
        + (t * (1 - t) ** 2 * slopes[cell] - t**2 * (1 - t) * slopes[cell + 1]) * width
    )
    return fitted


def _loess_span(value: object) -> float:
    """Return a checked LOESS span as a float."""
    resolved = float_argument(value, "span")
    if not 0 < resolved <= 1:
        raise ValueError("span must be greater than 0 and at most 1")
    return resolved


def seurat_v3_expected_variance(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    span: float = DEFAULT_LOESS_SPAN,
) -> np.ndarray:
    """Return the count variance that the seurat_v3 trend expects per feature.

    The trend is :func:`fit_loess` of degree 2 of ``log10(variance)`` on
    ``log10(mean)`` over the features whose variance is positive. A feature
    with zero variance is constant, so its expected variance is 0 and its
    standardized variance is 0 too.

    Args:
        means: The mean count of each feature over the cells.
        variances: The variance of each feature's counts, with one degree of
            freedom removed.
        span: The fraction of the features in each local fit.

    Returns:
        The float64 expected variance of each feature.

    Raises:
        ValueError: If a varying feature has a mean that is not positive.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    expected = np.zeros(means.shape, dtype=np.float64)
    varying = variances > 0
    if varying.any():
        if not (means[varying] > 0).all():
            raise ValueError("Features with a positive variance need a positive mean")
        trend = fit_loess(
            np.log10(means[varying]), np.log10(variances[varying]), span=span
        )
        expected[varying] = np.power(10.0, trend)
    return expected


def seurat_v3_clip_values(
    means: np.ndarray, expected_variance: np.ndarray, n_cells: int | np.ndarray
) -> np.ndarray:
    """Return the count at which seurat_v3 clips each feature.

    A count is clipped at ``sqrt(n_cells)`` expected standard deviations above
    the feature's mean, so no standardized value exceeds ``sqrt(n_cells)``.
    ``n_cells`` broadcasts against the means, such as one count per batch
    row.
    """
    clips: np.ndarray = np.asarray(means, dtype=np.float64) + np.sqrt(
        np.asarray(expected_variance, dtype=np.float64)
    ) * np.sqrt(np.asarray(n_cells, dtype=np.float64))
    return clips


def seurat_v3_standardized_variance(
    clipped_squares: np.ndarray, expected_variance: np.ndarray, n_cells: int
) -> np.ndarray:
    """Return the variance of each feature's standardized, clipped counts.

    ``clipped_squares`` holds, for each feature, the sum over the cells of
    ``(min(count, clip) - mean)**2`` with the clip of
    :func:`seurat_v3_clip_values`. Dividing by ``(n_cells - 1)`` and the
    expected variance gives the variance of the standardized values. A
    feature whose expected variance is 0 is constant and scores 0.

    Raises:
        ValueError: If there are fewer than two cells.
    """
    if n_cells < 2:
        raise ValueError("A standardized variance needs at least two cells")
    expected = np.asarray(expected_variance, dtype=np.float64)
    standardized = np.zeros(expected.shape, dtype=np.float64)
    np.divide(
        np.asarray(clipped_squares, dtype=np.float64),
        (n_cells - 1) * expected,
        out=standardized,
        where=expected > 0,
    )
    return standardized


def pearson_residual_clip(clip: float | None, n_cells: int) -> float:
    """Return the clip of Pearson residuals: ``clip``, or ``sqrt(n_cells)`` if None."""
    return float(np.sqrt(n_cells)) if clip is None else float(clip)


def pearson_residuals(
    counts: np.ndarray,
    cell_totals: np.ndarray,
    feature_fractions: np.ndarray,
    *,
    theta: float = DEFAULT_PEARSON_THETA,
    clip: float,
) -> np.ndarray:
    """Return analytic Pearson residuals of counts, clipped to ``[-clip, clip]``.

    The expected count of a feature in a cell is ``mu = n_c * p_g``, the
    cell's total times the feature's fraction of all counts, and the residual
    is ``(x - mu) / sqrt(mu + mu**2 / theta)``, Lause, Berens and Kobak
    (2021). A residual whose expected count is 0 is 0, its limit as the
    expected count goes to 0. Every step runs in float64.

    Args:
        counts: Counts with one row per cell and one column per feature.
        cell_totals: The total count of each cell.
        feature_fractions: Each feature's share of all counts.
        theta: The negative binomial overdispersion.
        clip: The largest absolute residual.

    Returns:
        Float64 residuals with the shape of ``counts``.
    """
    values = np.asarray(counts, dtype=np.float64)
    expected = np.multiply.outer(
        np.asarray(cell_totals, dtype=np.float64),
        np.asarray(feature_fractions, dtype=np.float64),
    )
    residuals = np.zeros(values.shape, dtype=np.float64)
    positive = expected > 0
    residuals[positive] = (values[positive] - expected[positive]) / np.sqrt(
        expected[positive] + expected[positive] ** 2 / theta
    )
    return np.clip(residuals, -clip, clip, out=residuals)


# The ``select_hvgs`` keywords that ``hvg_options`` checks.
HVG_OPTION_NAMES = (
    "min_cells",
    "top_n",
    "n_bins",
    "lowess_frac",
    "keep_bounds",
    "bin_strategy",
)


class HvgOptions(NamedTuple):
    """Checked ``select_hvgs`` options, each as an HVG selection records it.

    ``min_cells`` is None when the flavor's default applies.
    """

    min_cells: int | None
    top_n: int
    n_bins: int
    lowess_frac: float
    keep_bounds: bool
    bin_strategy: Literal["fixed", "adaptive"]


def hvg_options(
    *,
    min_cells: object,
    top_n: object,
    n_bins: object,
    lowess_frac: object,
    keep_bounds: object,
    bin_strategy: object,
) -> HvgOptions:
    """Return the checked counts, trend options, and flag of ``select_hvgs``."""
    checked_min_cells = (
        None
        if min_cells is None
        else integer_argument(min_cells, "min_cells", minimum=0)
    )
    checked_top_n = integer_argument(top_n, "top_n", minimum=1)
    checked_n_bins = integer_argument(n_bins, "n_bins", minimum=1)
    _lowess_frac(lowess_frac)
    if not isinstance(keep_bounds, bool | np.bool_):
        raise TypeError("keep_bounds must be a boolean")
    return HvgOptions(
        min_cells=checked_min_cells,
        top_n=checked_top_n,
        n_bins=checked_n_bins,
        lowess_frac=cast(float, lowess_frac),
        keep_bounds=bool(keep_bounds),
        bin_strategy=_bin_strategy(bin_strategy),
    )


# The ``select_hvgs`` keywords that ``hvg_flavor_options`` checks.
HVG_FLAVOR_OPTION_NAMES = (
    "flavor",
    "max_cells",
    "min_mean",
    "max_mean",
    "min_var",
    "max_var",
    "n_bins",
    "lowess_frac",
    "bin_strategy",
    "span",
    "theta",
    "clip",
)
# The options of the cytearc flavor and the defaults that other flavors keep.
_CYTEARC_DEFAULTS: dict[str, object] = {
    "min_mean": -np.inf,
    "max_mean": np.inf,
    "min_var": -np.inf,
    "max_var": np.inf,
    "n_bins": 200,
    "lowess_frac": 0.1,
    "bin_strategy": "adaptive",
}


class HvgFlavorOptions(NamedTuple):
    """The checked flavor of ``select_hvgs`` and its own options.

    ``span`` is a float for ``"seurat_v3"`` and ``theta`` a float for
    ``"pearson_residuals"``; ``clip`` is None when residuals are clipped at
    the square root of each batch's cell count.
    """

    flavor: Literal["cytearc", "seurat_v3", "pearson_residuals"]
    max_cells: float | None
    span: float
    theta: float
    clip: float | None


def _bound(value: object, name: str) -> float:
    """Return a log2 HVG bound: a real number, which may be infinite."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if np.isnan(resolved):
        raise ValueError(f"{name} must not be NaN")
    return resolved


def hvg_flavor_options(
    *,
    flavor: object,
    max_cells: object,
    min_mean: object,
    max_mean: object,
    min_var: object,
    max_var: object,
    n_bins: object,
    lowess_frac: object,
    bin_strategy: object,
    span: object,
    theta: object,
    clip: object,
) -> HvgFlavorOptions:
    """Check the flavor of ``select_hvgs`` and the options that depend on it.

    The mean and variance bounds and the trend options belong to the cytearc
    flavor, ``span`` to seurat_v3, and ``theta`` and ``clip`` to
    pearson_residuals; another flavor refuses a value other than the default.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If an option is out of range or does not apply to the
            flavor.
    """
    if not isinstance(flavor, str) or flavor not in HVG_FLAVORS:
        raise ValueError(
            "flavor must be one of " + ", ".join(repr(name) for name in HVG_FLAVORS)
        )
    if max_cells is not None:
        if isinstance(max_cells, bool | np.bool_) or not isinstance(
            max_cells, int | float | np.integer | np.floating
        ):
            raise TypeError("max_cells must be a number, infinity, or None")
        if np.isnan(max_cells) or max_cells < 0:
            raise ValueError("max_cells must be at least 0")
    bounds = {
        "min_mean": _bound(min_mean, "min_mean"),
        "max_mean": _bound(max_mean, "max_mean"),
        "min_var": _bound(min_var, "min_var"),
        "max_var": _bound(max_var, "max_var"),
    }
    checked_span = _loess_span(span)
    checked_theta = float_argument(theta, "theta")
    if checked_theta <= 0:
        raise ValueError("theta must be greater than 0")
    checked_clip = None if clip is None else float_argument(clip, "clip")
    if checked_clip is not None and checked_clip < 0:
        raise ValueError("clip must be at least 0")
    if flavor != "cytearc":
        given = {
            **bounds,
            "n_bins": n_bins,
            "lowess_frac": lowess_frac,
            "bin_strategy": bin_strategy,
        }
        changed = [
            name for name, value in given.items() if value != _CYTEARC_DEFAULTS[name]
        ]
        if changed:
            raise ValueError(
                ", ".join(changed)
                + f" apply only to flavor='cytearc'; leave them at their defaults "
                f"for flavor={flavor!r}"
            )
    if flavor != "seurat_v3" and checked_span != DEFAULT_LOESS_SPAN:
        raise ValueError("span applies only to flavor='seurat_v3'")
    if flavor != "pearson_residuals" and (
        checked_theta != DEFAULT_PEARSON_THETA or checked_clip is not None
    ):
        raise ValueError("theta and clip apply only to flavor='pearson_residuals'")
    return HvgFlavorOptions(
        flavor=cast(Literal["cytearc", "seurat_v3", "pearson_residuals"], flavor),
        max_cells=None if max_cells is None else float(cast(float, max_cells)),
        span=checked_span,
        theta=checked_theta,
        clip=checked_clip,
    )


def default_hvg_min_cells(n_selected: int) -> int:
    """Return the default ``min_cells`` of the cytearc flavor for ``n_selected`` cells.

    It is 1% of the selected cells rounded up, at least 3 and at most 20, so
    selections of 2,000 cells or more keep the earlier fixed 20. It never
    exceeds a fifth of the cells, at least 1, so a tiny selection keeps a
    band of detection counts to choose from.
    """
    scaled = -(-int(n_selected) // _MIN_CELLS_DIVISOR)
    return min(
        HVG_MIN_CELLS_CAP,
        max(HVG_MIN_CELLS_FLOOR, scaled),
        max(1, int(n_selected) // 5),
    )


def default_hvg_ubiquity_limit(n_selected: int) -> int:
    """Return the most cells a feature may be detected in under the default cap.

    Features detected in more than 99.5% of the selected cells are ubiquitous,
    so the limit is ``floor(0.995 * n_selected)``.
    """
    return int(n_selected) * _UBIQUITY_NUMERATOR // _UBIQUITY_DENOMINATOR


def resolve_hvg_detection_bounds(
    min_cells: int | None,
    max_cells: float | None,
    *,
    n_selected: int,
    keep_bounds: bool,
    flavor: str = "cytearc",
) -> tuple[int, int | float]:
    """Return the detection bounds that HVG selection applies and records.

    A feature is a candidate when it is detected in at least ``min_cells``
    selected cells and in fewer than ``max_cells``, or in at most
    ``max_cells`` with ``keep_bounds``. For the cytearc flavor, a ``min_cells``
    of None is :func:`default_hvg_min_cells`, and a ``max_cells`` of None
    excludes features detected in more than 99.5% of the selected cells:
    :func:`default_hvg_ubiquity_limit` plus one, or the limit itself with
    ``keep_bounds``. Other flavors bound detection only when asked, so their
    None bounds are 0 and infinity. Infinity sets no upper bound, and any
    other ``max_cells`` is taken as an integer.
    """
    if min_cells is None:
        min_cells = default_hvg_min_cells(n_selected) if flavor == "cytearc" else 0
    if max_cells is None:
        if flavor != "cytearc":
            return min_cells, np.inf
        limit = default_hvg_ubiquity_limit(n_selected)
        return min_cells, limit if keep_bounds else limit + 1
    if max_cells == np.inf:
        return min_cells, np.inf
    return min_cells, int(max_cells)


def _linear_threshold(value: float, unbounded_value: float) -> float:
    if value == unbounded_value:
        return value
    return float(2**value)


def hvg_candidates(
    corrected_variance: np.ndarray,
    normalized_cell_counts: np.ndarray,
    mean_nonzero: np.ndarray,
    *,
    allowed: np.ndarray,
    min_cells: int,
    max_cells: int | float,
    min_var: float,
    max_var: float,
    min_mean: float,
    max_mean: float,
    keep_bounds: bool,
) -> np.ndarray:
    """Return the features that pass the HVG candidate filters.

    A candidate is ``allowed``, is detected in at least ``min_cells`` cells
    and in fewer than ``max_cells`` (at most with ``keep_bounds``), and has
    a mean over the cells where it is detected and a corrected variance
    within the bounds. The mean and variance bounds are log2 values, so a
    bound ``b`` compares with ``2**b``, and an infinite bound leaves its side
    open. Bounds exclude equal values unless ``keep_bounds`` is set, and NaN
    never lies within them.
    """
    detected = np.asarray(normalized_cell_counts)
    candidates = np.asarray(allowed, dtype=bool) & (detected >= min_cells)
    candidates &= detected <= max_cells if keep_bounds else detected < max_cells
    candidates &= within_bounds(
        np.asarray(mean_nonzero),
        _linear_threshold(min_mean, -np.inf),
        _linear_threshold(max_mean, np.inf),
        keep_bounds=keep_bounds,
    )
    candidates &= within_bounds(
        np.asarray(corrected_variance),
        _linear_threshold(min_var, -np.inf),
        _linear_threshold(max_var, np.inf),
        keep_bounds=keep_bounds,
    )
    return candidates


def _top_candidates(
    scores: np.ndarray, candidates: np.ndarray, top_n: int
) -> np.ndarray:
    """Return the ``top_n`` candidates by score, ties broken by lower index."""
    candidate_idx = np.flatnonzero(candidates)
    order = np.lexsort((candidate_idx, -scores[candidate_idx]))
    return candidate_idx[order[:top_n]]


def _no_candidates_message(
    *,
    min_cells: int,
    max_cells: int | float,
    min_var: float,
    max_var: float,
    min_mean: float,
    max_mean: float,
    where: str = "",
) -> str:
    """Return the error of an HVG selection that no feature passes."""
    return (
        "HVG selection contains no features. No features passed HVG candidate "
        f"filters{where} (min_cells={min_cells}, max_cells={max_cells}, log2 "
        f"bounds min_mean={min_mean}, max_mean={max_mean}, min_var={min_var}, "
        f"max_var={max_var}). Widen the bounds or change the blacklist."
    )


def _warn_few_candidates(n_candidates: int, top_n: int) -> None:
    """Warn that fewer than ``top_n`` features passed the candidate filters."""
    logger.warning(
        f"Only {n_candidates} features passed the HVG candidate filters, fewer "
        f"than top_n={top_n}; all of them are selected"
    )


def select_highly_variable_features(
    corrected_variance: np.ndarray,
    normalized_cell_counts: np.ndarray,
    mean_nonzero: np.ndarray,
    active_features: np.ndarray,
    feature_names: np.ndarray,
    *,
    min_cells: int,
    max_cells: int | float,
    top_n: int,
    min_var: float,
    max_var: float,
    min_mean: float,
    max_mean: float,
    blacklist: str,
    keep_bounds: bool,
) -> np.ndarray:
    """Select highly variable features from precomputed feature statistics.

    Every filter applies to the candidates before they are ranked: the
    blacklist, the detection bounds ``min_cells`` and ``max_cells``, the
    log2 bounds on the mean over detected cells ``min_mean`` and
    ``max_mean``, and the log2 bounds on the corrected variance ``min_var``
    and ``max_var`` (see :func:`hvg_candidates`). The ``top_n`` candidates
    with the highest corrected variance are selected, ties broken by lower
    feature index, or every candidate when fewer pass.

    Raises:
        ValueError: If the inputs are misaligned, ``top_n`` is below 1, or no
            feature passes the filters.
    """
    corrected_variance = np.asarray(corrected_variance)
    normalized_cell_counts = np.asarray(normalized_cell_counts)
    mean_nonzero = np.asarray(mean_nonzero)
    active_features = np.asarray(active_features, dtype=bool)
    feature_names = np.asarray(feature_names)
    size = corrected_variance.shape[0]
    if any(
        values.shape != (size,)
        for values in (
            normalized_cell_counts,
            mean_nonzero,
            active_features,
            feature_names,
        )
    ):
        raise ValueError("HVG inputs must be one-dimensional arrays of equal length")
    if top_n < 1:
        raise ValueError(
            "ERROR: Please provide a value greater than 0 for `top_n` parameter"
        )
    allowed = active_features.copy()
    if blacklist:
        allowed &= ~regex_match_mask(feature_names, blacklist)
    candidates = hvg_candidates(
        corrected_variance,
        normalized_cell_counts,
        mean_nonzero,
        allowed=allowed,
        min_cells=min_cells,
        max_cells=max_cells,
        min_var=min_var,
        max_var=max_var,
        min_mean=min_mean,
        max_mean=max_mean,
        keep_bounds=keep_bounds,
    )
    n_candidates = int(candidates.sum())
    if n_candidates == 0:
        raise ValueError(
            _no_candidates_message(
                min_cells=min_cells,
                max_cells=max_cells,
                min_var=min_var,
                max_var=max_var,
                min_mean=min_mean,
                max_mean=max_mean,
            )
        )
    if top_n > n_candidates:
        _warn_few_candidates(n_candidates, top_n)
    selected = np.zeros(size, dtype=bool)
    selected[_top_candidates(corrected_variance, candidates, top_n)] = True
    return selected


class BatchRanking(NamedTuple):
    """Highly variable features chosen across batches.

    Attributes:
        selected: Whether each feature is selected.
        batch_count: The number of batches that rank each feature among their
            ``top_n``.
        median_rank: Each feature's median rank over those batches, counted
            from 0 for the most variable feature, and NaN for a feature that
            no batch ranks among its ``top_n``.
    """

    selected: np.ndarray
    batch_count: np.ndarray
    median_rank: np.ndarray


def combine_batch_rankings(
    scores: np.ndarray,
    candidates: np.ndarray,
    *,
    top_n: int,
    batch_names: list[str] | None = None,
) -> BatchRanking:
    """Select features that many batches rank as highly variable.

    Each batch ranks its candidates by score, highest first and ties broken by
    lower feature index, and its first ``top_n`` are highly variable in it.
    Features are then ordered by the number of batches in which they are
    highly variable, most first, then by their median rank over those
    batches, then by feature index, and the first ``top_n`` are selected. This
    is the order of Seurat's ``SelectIntegrationFeatures``; scanpy's
    ``seurat_v3`` flavor sorts by median rank first, and its
    ``seurat_v3_paper`` flavor by batch count first, as here. With one batch
    the selection is the ``top_n`` candidates by score.

    A warning names each of several batches with fewer than ``top_n``
    candidates, and another reports a selection of fewer than ``top_n``
    features.

    Args:
        scores: Each batch's score of each feature, one row per batch.
        candidates: Whether each feature is a candidate in each batch.
        top_n: The number of features that each batch ranks as highly
            variable, and the number selected.
        batch_names: The name of each batch for messages.

    Returns:
        The selection with each feature's batch count and median rank.

    Raises:
        ValueError: If no batch has a candidate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=bool)
    n_batches, n_features = scores.shape
    names = batch_names or [str(index) for index in range(n_batches)]
    ranks = np.full((n_batches, n_features), np.nan)
    for batch in range(n_batches):
        n_candidates = int(candidates[batch].sum())
        if n_batches > 1 and n_candidates < top_n:
            logger.warning(
                f"Batch {names[batch]!r} has {n_candidates} HVG candidates, fewer "
                f"than top_n={top_n}; all of them count as highly variable in it"
            )
        top = _top_candidates(scores[batch], candidates[batch], top_n)
        ranks[batch, top] = np.arange(len(top), dtype=np.float64)
    ranked = ~np.isnan(ranks)
    batch_count = ranked.sum(axis=0).astype(np.int64)
    eligible = np.flatnonzero(batch_count > 0)
    if eligible.size == 0:
        raise ValueError(
            "HVG selection contains no features. No features passed HVG candidate "
            "filters in any batch. Widen the bounds or change the blacklist."
        )
    median_rank = np.full(n_features, np.nan)
    median_rank[eligible] = np.nanmedian(ranks[:, eligible], axis=0)
    order = eligible[
        np.lexsort((eligible, median_rank[eligible], -batch_count[eligible]))
    ]
    selected = np.zeros(n_features, dtype=bool)
    selected[order[:top_n]] = True
    if len(order) < top_n and n_batches == 1:
        _warn_few_candidates(len(order), top_n)
    elif len(order) < top_n:
        logger.warning(
            f"Only {len(order)} features are highly variable in some batch, fewer "
            f"than top_n={top_n}; all of them are selected"
        )
    return BatchRanking(selected, batch_count, median_rank)


def select_batch_highly_variable_features(
    scores: np.ndarray,
    mean_nonzero: np.ndarray,
    detected: np.ndarray,
    feature_names: np.ndarray,
    *,
    min_cells: int,
    max_cells: int | float,
    top_n: int,
    min_var: float,
    max_var: float,
    min_mean: float,
    max_mean: float,
    blacklist: str,
    keep_bounds: bool,
    batch_names: list[str] | None = None,
) -> BatchRanking:
    """Select highly variable features from the scores of each batch.

    Each batch's candidates pass the blacklist, the detection bounds over
    every selected cell, and the log2 bounds on that batch's mean over
    detected cells and on its score (see :func:`hvg_candidates`);
    :func:`combine_batch_rankings` then ranks them. With one batch this is
    the ``top_n`` candidates by score.

    Args:
        scores: Each batch's score of each feature, one row per batch.
        mean_nonzero: Each batch's mean over the cells where a feature is
            detected, one row per batch.
        detected: The number of selected cells where each feature is detected.
        feature_names: The name of each feature, matched against ``blacklist``.
        min_cells: The fewest cells a candidate is detected in.
        max_cells: The detection cutoff, exclusive unless ``keep_bounds``.
        top_n: The number of features each batch ranks and the selection keeps.
        min_var: The log2 lower bound of the score.
        max_var: The log2 upper bound of the score.
        min_mean: The log2 lower bound of the mean over detected cells.
        max_mean: The log2 upper bound of the mean over detected cells.
        blacklist: A case-insensitive regular expression of excluded names.
        keep_bounds: Whether bounds keep equal values.
        batch_names: The name of each batch for messages.

    Returns:
        The selection with each feature's batch count and median rank.

    Raises:
        ValueError: If no batch has a candidate; the message names the
            filters.
    """
    scores = np.asarray(scores, dtype=np.float64)
    allowed = np.ones(scores.shape[1], dtype=bool)
    if blacklist:
        allowed &= ~regex_match_mask(np.asarray(feature_names), blacklist)
    candidates = np.stack(
        [
            hvg_candidates(
                scores[batch],
                detected,
                np.asarray(mean_nonzero)[batch],
                allowed=allowed,
                min_cells=min_cells,
                max_cells=max_cells,
                min_var=min_var,
                max_var=max_var,
                min_mean=min_mean,
                max_mean=max_mean,
                keep_bounds=keep_bounds,
            )
            for batch in range(scores.shape[0])
        ]
    )
    if not candidates.any():
        raise ValueError(
            _no_candidates_message(
                min_cells=min_cells,
                max_cells=max_cells,
                min_var=min_var,
                max_var=max_var,
                min_mean=min_mean,
                max_mean=max_mean,
                where=" in any batch" if scores.shape[0] > 1 else "",
            )
        )
    return combine_batch_rankings(
        scores, candidates, top_n=top_n, batch_names=batch_names
    )
