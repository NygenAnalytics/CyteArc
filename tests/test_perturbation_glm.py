import warnings

import numpy as np
import pytest
import statsmodels.api as sm

from cytearc.perturbation import glm
from tests.perturbation_helpers import dense_pair, use_python_kernels


def _python_kernels(monkeypatch: pytest.MonkeyPatch, python_kernels: bool) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, glm)


def _binomial_fit(y: np.ndarray, design: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(y, design, family=sm.families.Binomial()).fit(tol=1e-14)


def _indicator(n: int, seed: int, coefficients: np.ndarray):
    """A design with an intercept, two covariates and a sample, and its 0/1 responses."""
    rng = np.random.default_rng(seed)
    design = np.column_stack(
        [np.ones(n), rng.normal(size=(n, 2)), (np.arange(n) % 3 == 0).astype(float)]
    )
    probabilities = 1 / (1 + np.exp(-design @ coefficients))
    return design, (rng.random(n) < probabilities).astype(np.float64)


def _fit(y: np.ndarray, design: np.ndarray, start: np.ndarray | None = None):
    n, p = design.shape
    coefficients = np.zeros(p)
    if start is None:
        share = y.mean()
        coefficients[0] = np.log(share / (1 - share))
    else:
        coefficients[:] = start
    eta = np.empty(n)
    probabilities = np.empty(n)
    glm.fit_logistic(y, design, coefficients, eta, probabilities)
    return coefficients, eta, probabilities


@pytest.mark.parametrize("python_kernels", [False, True])
def test_logistic_fits_match_statsmodels(monkeypatch, python_kernels) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n = 3000 if not python_kernels else 300
    design, y = _indicator(n, 1, np.array([-2.5, 0.8, -0.4, 0.6]))

    coefficients, eta, probabilities = _fit(y, design)

    expected = _binomial_fit(y, design)
    np.testing.assert_allclose(coefficients, expected.params, rtol=0, atol=1e-8)
    np.testing.assert_allclose(probabilities, expected.mu, rtol=1e-8)
    np.testing.assert_allclose(eta, design @ coefficients, rtol=1e-12)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_logistic_fits_halve_steps_that_lower_the_likelihood(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    design, y = _indicator(400, 2, np.array([-1.0, 1.2, 0.0, -0.5]))
    # Far from the fit, every cell's probability is near 1 and its weight near
    # 0, so a full Newton step overshoots.
    start = np.array([9.0, -4.0, 3.0, 2.0])

    coefficients, _, _ = _fit(y, design, start)

    np.testing.assert_allclose(
        coefficients, _binomial_fit(y, design).params, rtol=0, atol=1e-8
    )


@pytest.mark.parametrize("python_kernels", [False, True])
def test_logistic_fits_keep_directions_the_cells_do_not_determine(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    design, y = _indicator(500, 3, np.array([-1.5, 0.7, 0.3, 0.0]))
    # A covariate that is 0 on every cell.
    widened = np.column_stack([design, np.zeros(len(y))])

    coefficients, _, _ = _fit(y, widened)

    assert coefficients[-1] == 0.0
    np.testing.assert_allclose(
        coefficients[:-1], _binomial_fit(y, design).params, rtol=0, atol=1e-8
    )


@pytest.mark.parametrize("python_kernels", [False, True])
def test_separated_cells_fall_towards_probability_zero(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    design, y = _indicator(900, 4, np.array([-1.0, 0.9, -0.6, 0.0]))
    # No cell of the sample of the last column responds: separation.
    sample = design[:, -1] == 1
    y[sample] = 0.0

    coefficients, _, probabilities = _fit(y, design)

    assert probabilities[sample].max() < 1e-6
    assert coefficients[-1] < -10
    # In the limit those cells drop out, so the other coefficients are the
    # fit on the other cells without the sample's column.
    rest = _binomial_fit(y[~sample], design[~sample, :-1])
    np.testing.assert_allclose(coefficients[:-1], rest.params, rtol=0, atol=1e-6)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_logistic_probabilities_stay_within_the_unit_interval(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    design = np.column_stack([np.ones(5), [-2000.0, -1.0, 0.0, 1.0, 2000.0]])
    eta = np.empty(5)
    probabilities = np.empty(5)

    glm.logistic_probabilities(design, np.array([0.0, 1.0]), eta, probabilities)

    np.testing.assert_array_equal(eta, [-700.0, -1.0, 0.0, 1.0, 700.0])
    assert 0 < probabilities[0] < 1e-300 and probabilities[-1] == 1.0
    np.testing.assert_allclose(probabilities[1:4], 1 / (1 + np.exp(-eta[1:4])))


@pytest.mark.parametrize("python_kernels", [False, True])
def test_a_set_test_under_a_negative_binomial_fit_matches_a_dense_one(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(4)
    n = 400 if not python_kernels else 120
    design = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    mu = np.exp(design @ np.array([1.0, 0.4, -0.3]))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(np.float64)
    cells = np.flatnonzero(rng.random(n) < 0.3)
    treated = np.zeros(n, dtype=bool)
    treated[cells] = True
    coefficients = np.zeros(3)
    coefficients[0] = np.log(y.mean())
    eta, fitted, factor = np.empty(n), np.empty(n), np.empty((3, 3))

    dispersion = glm.fit_negative_binomial(y, design, coefficients, eta, fitted, factor)
    statistic, fold, error = glm.set_test(
        y, fitted, design, dispersion, cells, factor, np.empty(3)
    )

    # The fit's means are those of the Poisson fit, its factor that of its
    # information, and the set's statistic, fold change and standard error
    # those of a dense computation with statsmodels' fit and SciPy's size.
    expected = dense_pair(y, design, treated)
    np.testing.assert_allclose(
        fitted, np.exp(design @ coefficients), rtol=1e-12, atol=0
    )
    weights = fitted / (1 + dispersion * fitted)
    np.testing.assert_allclose(
        factor @ factor.T, design.T @ (weights[:, None] * design), rtol=1e-10
    )
    np.testing.assert_allclose((statistic, fold, error), expected, rtol=1e-6)
