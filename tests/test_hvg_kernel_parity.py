"""The Numba kernels of HVG flavors, compiled and as Python, against NumPy references.

The flavor tests exercise these kernels compiled; these tests also run each
one's ``py_func`` on inputs that reach every branch, and check both against
direct NumPy statements of what the kernels' docstrings promise.
"""

import numpy as np
import pytest

from cytearc.assay.hvg_statistics import (
    _band_clipped_squares_kernel,
    _band_moments_kernel,
    _band_pearson_kernel,
)
from cytearc.features.variability import _fit_loess_points

_KERNELS = ["jit", "python"]


def _kernel(kernel, mode: str):
    return kernel if mode == "jit" else kernel.py_func


def _band() -> dict[str, np.ndarray]:
    """Counts of 3 feature rows for 7 cells of 2 batches, with zeros and a gap.

    Row 2 is never read, column 6 of the values is not a selected cell, and
    batch 1 has a cell without any count in row 0.
    """
    values = np.array(
        [
            [0, 3, 0, 5, 1, 0, 9],
            [2, 0, 0, 4, 0, 7, 9],
            [9, 9, 9, 9, 9, 9, 9],
        ],
        dtype=np.uint16,
    )
    return {
        "values": values,
        "value_rows": np.array([1, 0], dtype=np.int64),
        "local": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        "groups": np.array([0, 0, 1, 1, 0, 1], dtype=np.int64),
        # Batch sizes include selected cells outside this band of cells.
        "sizes": np.array([4, 5], dtype=np.int64),
    }


def _reference_columns(band: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Each written column's counts of the band's selected cells."""
    return [
        band["values"][row, band["local"]].astype(np.float64)
        for row in band["value_rows"]
    ]


@pytest.mark.parametrize("mode", _KERNELS)
def test_band_moments_kernel_matches_moments_with_implicit_zeros(mode) -> None:
    band = _band()
    scale = 1000.0
    inverse_totals = np.array([0.5, 0.25, 1.0, 0.125, 0.2, 0.1])
    n_groups, n_columns = 2, 2
    total = np.zeros((n_groups, n_columns))
    m2 = np.zeros((n_groups, n_columns))
    detected = np.zeros((n_groups, n_columns), dtype=np.int64)
    # A maximum carried from an earlier band stays when no value exceeds it.
    maximum = np.array([[0.0, 0.0], [10_000.0, 0.0]])
    means = np.zeros(n_groups)
    counts = np.zeros(n_groups, dtype=np.int64)

    _kernel(_band_moments_kernel, mode)(
        band["values"],
        band["value_rows"],
        band["local"],
        band["groups"],
        scale,
        inverse_totals,
        band["sizes"],
        total,
        m2,
        detected,
        maximum,
        means,
        counts,
    )

    for k, column in enumerate(_reference_columns(band)):
        for group in range(n_groups):
            cells = band["groups"] == group
            values = scale * column[cells] * inverse_totals[cells]
            stored = values[column[cells] != 0]
            padded = np.concatenate(
                [stored, np.zeros(band["sizes"][group] - stored.size)]
            )
            assert total[group, k] == pytest.approx(stored.sum(), rel=1e-12)
            assert detected[group, k] == np.count_nonzero(stored > 0)
            assert m2[group, k] == pytest.approx(
                ((padded - padded.mean()) ** 2).sum(), rel=1e-12
            )
            expected_maximum = max(
                stored.max(initial=0.0), 10_000.0 if (group, k) == (1, 0) else 0.0
            )
            assert maximum[group, k] == expected_maximum


@pytest.mark.parametrize("mode", _KERNELS)
def test_band_clipped_squares_kernel_matches_clipped_deviations(mode) -> None:
    band = _band()
    means = np.array([[1.5, 0.5], [2.0, 1.0]])
    # A clip below some counts and one above every count.
    clips = np.array([[3.0, 2.0], [100.0, 4.0]])
    out = np.zeros((2, 2))
    stored = np.zeros(2, dtype=np.int64)

    _kernel(_band_clipped_squares_kernel, mode)(
        band["values"],
        band["value_rows"],
        band["local"],
        band["groups"],
        means,
        clips,
        band["sizes"],
        out,
        stored,
    )

    for k, column in enumerate(_reference_columns(band)):
        for group in range(2):
            counts = column[band["groups"] == group]
            counts = counts[counts != 0]
            clipped = np.minimum(counts, clips[group, k])
            padded = np.concatenate(
                [clipped, np.zeros(band["sizes"][group] - counts.size)]
            )
            assert out[group, k] == pytest.approx(
                ((padded - means[group, k]) ** 2).sum(), rel=1e-12
            )


@pytest.mark.parametrize("mode", _KERNELS)
def test_band_pearson_kernel_matches_clipped_residual_moments(mode) -> None:
    band = _band()
    totals = np.array([10.0, 4.0, 0.0, 12.0, 6.0, 8.0])
    # Batch 1 has no counts of the second written column: residuals there are 0.
    fractions = np.array([[0.2, 0.3], [0.25, 0.0]])
    theta = 100.0
    clips = np.array([1.0, 50.0])
    total = np.zeros((2, 2))
    m2 = np.zeros((2, 2))
    means = np.zeros(2)
    counts = np.zeros(2, dtype=np.int64)

    _kernel(_band_pearson_kernel, mode)(
        band["values"],
        band["value_rows"],
        band["local"],
        band["groups"],
        totals,
        fractions,
        theta,
        clips,
        total,
        m2,
        means,
        counts,
    )

    for k, column in enumerate(_reference_columns(band)):
        for group in range(2):
            cells = band["groups"] == group
            expected_counts = totals[cells] * fractions[group, k]
            residuals = np.zeros(expected_counts.shape)
            positive = expected_counts > 0
            mu = expected_counts[positive]
            residuals[positive] = np.clip(
                (column[cells][positive] - mu) / np.sqrt(mu + mu * mu / theta),
                -clips[group],
                clips[group],
            )
            assert total[group, k] == pytest.approx(residuals.sum(), abs=1e-12)
            assert m2[group, k] == pytest.approx(
                ((residuals - residuals.mean()) ** 2).sum(), abs=1e-12
            )
    # The clip of batch 0 binds: an unclipped residual of the band exceeds it.
    column = _reference_columns(band)[1]
    cells = band["groups"] == 0
    mu = totals[cells] * fractions[0, 1]
    unclipped = (column[cells] - mu) / np.sqrt(mu + mu * mu / theta)
    assert np.abs(unclipped).max() > clips[0]


def _reference_loess_point(
    x: np.ndarray, y: np.ndarray, q: int, degree: int, x0: float
) -> tuple[float, float]:
    """Weighted least squares of degree ``degree`` over the q nearest points."""
    order = np.argsort(np.abs(x - x0), kind="stable")[:q]
    h = np.abs(x[order] - x0).max()
    if h <= 0.0:
        tied = x == x0
        return float(y[tied].mean()), 0.0
    ratio = np.abs(x - x0) / h
    weights = np.where(ratio < 1.0, (1.0 - ratio**3) ** 3, 0.0)
    window = np.zeros(x.shape, dtype=bool)
    window[order] = True
    weights = np.where(window, weights, 0.0)
    if weights.sum() <= 0.0:
        return float(y[window].mean()), 0.0
    u = (x - x0) / h
    used = weights > 0
    distinct = np.unique(u[used]).size
    fit_degree = min(degree, distinct - 1)
    design = np.vander(u[used], fit_degree + 1, increasing=True)
    sqrt_w = np.sqrt(weights[used])
    coefficients, *_ = np.linalg.lstsq(
        design * sqrt_w[:, None], y[used] * sqrt_w, rcond=None
    )
    slope = coefficients[1] / h if fit_degree >= 1 else 0.0
    return float(coefficients[0]), float(slope)


@pytest.mark.parametrize("mode", _KERNELS)
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_loess_point_kernel_matches_weighted_least_squares(mode, degree) -> None:
    rng = np.random.default_rng(11)
    x = np.sort(rng.normal(size=40))
    y = 1.0 + 2.0 * x - 0.5 * x**2 + rng.normal(scale=0.1, size=40)
    # Repeated points reuse the previous fit; the points move the window right.
    points = np.array([x[0], x[3], x[3], 0.0, x[20], x[-1]])
    values = np.empty(points.shape)
    slopes = np.empty(points.shape)

    _kernel(_fit_loess_points, mode)(x, y, 12, degree, points, values, slopes)

    for i, point in enumerate(points):
        value, slope = _reference_loess_point(x, y, 12, degree, point)
        assert values[i] == pytest.approx(value, rel=1e-9, abs=1e-12)
        assert slopes[i] == pytest.approx(slope, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("mode", _KERNELS)
def test_loess_point_kernel_handles_ties_and_degenerate_windows(mode) -> None:
    kernel = _kernel(_fit_loess_points, mode)

    # Every point of the window sits at x0: the fit is the mean of the tie.
    x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 5.0])
    y = np.array([3.0, 1.0, 2.0, 3.0, 6.0, 9.0])
    values, slopes = np.empty(1), np.empty(1)
    kernel(x, y, 3, 2, np.array([1.0]), values, slopes)
    assert values[0] == pytest.approx(3.0) and slopes[0] == 0.0

    # A vertex midway between two points: both sit at distance h and weigh 0.
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 5.0])
    kernel(x, y, 2, 2, np.array([1.0]), values, slopes)
    assert values[0] == pytest.approx(3.0) and slopes[0] == 0.0

    # Weighted points of one distinct value cannot fit a slope: the degree
    # lowers to a weighted mean, with the far point weighing zero.
    x = np.array([0.0, 0.0, 0.0, 3.0])
    y = np.array([2.0, 4.0, 6.0, 100.0])
    kernel(x, y, 4, 2, np.array([0.0]), values, slopes)
    assert values[0] == pytest.approx(4.0) and slopes[0] == 0.0

    # Two distinct weighted values fit a line, and the quadratic term vanishes.
    x = np.array([0.0, 0.0, 1.0, 1.0, 3.0])
    y = np.array([1.0, 1.0, 3.0, 3.0, 50.0])
    kernel(x, y, 5, 2, np.array([0.0]), values, slopes)
    value, slope = _reference_loess_point(x, y, 5, 2, 0.0)
    assert values[0] == pytest.approx(value, rel=1e-9)
    assert slopes[0] == pytest.approx(slope, rel=1e-9)
