import math

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.mixture import GaussianMixture

from cytearc.perturbation import mixtures
from cytearc.perturbation.histograms import GuideCountHistograms, guide_count_histograms
from cytearc.perturbation.mixtures import (
    FIT_STATUSES,
    GUIDE_FIT_FITTED,
    GUIDE_FIT_NO_CELLS,
    GUIDE_FIT_NO_FIT,
    GUIDE_FIT_NO_POSITIVE,
    GUIDE_FIT_TOO_SPARSE,
    fit_gaussian,
    fit_poisson_gaussian,
    gaussian_posterior,
    poisson_gaussian_posterior,
)
from tests.perturbation_helpers import screen_counts, use_python_kernels


def _guides(
    guides: list[tuple[list[int], list[int]]], n_cells: int
) -> GuideCountHistograms:
    """Histograms of one sample whose guides hold the given counts and cells."""
    sizes = [len(counts) for counts, _ in guides]
    offsets = np.zeros(len(guides) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return GuideCountHistograms(
        n_cells=np.asarray([n_cells]),
        n_guides=len(guides),
        offsets=offsets,
        counts=np.asarray([c for counts, _ in guides for c in counts], dtype=np.int64),
        cells=np.asarray([n for _, cells in guides for n in cells], dtype=np.int64),
    )


def _per_cell(found: GuideCountHistograms) -> GuideCountHistograms:
    """The same histograms with every cell listed as its own count of weight 1."""
    sizes = np.add.reduceat(found.cells, found.offsets[:-1]) * (
        np.diff(found.offsets) > 0
    )
    offsets = np.zeros(len(found.offsets), dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    counts = np.repeat(found.counts, found.cells)
    return GuideCountHistograms(
        n_cells=found.n_cells,
        n_guides=found.n_guides,
        offsets=offsets,
        counts=counts,
        cells=np.ones(len(counts), dtype=np.int64),
    )


def _screen(seed: int, *, n_cells: int = 4_000, n_guides: int = 4, n_samples: int = 2):
    counts = screen_counts(n_cells, n_guides, seed=seed)
    samples = np.random.default_rng(seed + 1).integers(0, n_samples, size=n_cells)
    found = guide_count_histograms(
        [counts], samples, n_samples=n_samples, n_guides=n_guides
    )
    return counts, samples, found


# Guides of one sample of 6,000 cells, one per status of a Poisson-Gaussian fit.
_STATUS_GUIDES = [
    ([], []),  # no cell holds the guide
    ([50], [1]),  # one cell
    ([1], [100]),  # many cells, but none above 1 UMI
    ([1, 2], [1, 1]),  # no start gives an admissible fit
    ([2, 7, 10, 12, 18, 31, 34], [3, 5_000, 1, 10, 1, 100, 100]),  # no count is called
    ([1, 2, 3, 40, 60, 80], [400, 60, 5, 30, 50, 20]),  # carriers above ambient counts
]
_STATUS_EXPECTED = [
    GUIDE_FIT_NO_CELLS,
    GUIDE_FIT_TOO_SPARSE,
    GUIDE_FIT_TOO_SPARSE,
    GUIDE_FIT_NO_FIT,
    GUIDE_FIT_NO_POSITIVE,
    GUIDE_FIT_FITTED,
]


def _poisson_gaussian_loss(
    params: np.ndarray, values: np.ndarray, cells: np.ndarray
) -> float:
    """The negative log joint written from the model, without prior constants."""
    rate, mean, std, weight = params
    normal = (
        np.log(weight)
        - 0.5 * np.log(2 * np.pi)
        - np.log(std)
        - 0.5 * ((values - mean) / std) ** 2
    )
    poisson = np.log1p(-weight) + values * np.log(rate) - rate - gammaln(values + 1)
    likelihood = np.sum(cells * np.logaddexp(normal, poisson))
    prior = (
        -0.1 * np.log1p(-weight)
        - 0.9 * np.log(weight)
        - 0.5 * ((mean - 3.0) / 2.0) ** 2
        - np.log(std)
        - 0.5 * (np.log(std) - 2.0) ** 2
        - np.log(rate)
        - 0.5 * np.log(rate) ** 2
    )
    return float(-(likelihood + prior))


def test_poisson_gaussian_separates_carriers_from_ambient_counts() -> None:
    counts, samples, found = _screen(10)
    fits = fit_poisson_gaussian(found)

    assert fits.status.tolist() == [[GUIDE_FIT_FITTED] * 4] * 2
    assert fits.threshold.dtype == np.int64
    assert ((fits.threshold > 3) & (fits.threshold < 30)).all()
    np.testing.assert_allclose(fits.gaussian_mean, 6.0, atol=0.3)
    np.testing.assert_allclose(fits.gaussian_std, 1.0, atol=0.2)
    assert (fits.iterations > 0).all()
    for sample in range(2):
        for guide in range(4):
            threshold = fits.threshold[sample, guide]
            posterior = poisson_gaussian_posterior(
                np.arange(1, threshold + 1),
                poisson_rate=fits.poisson_rate[sample, guide],
                gaussian_mean=fits.gaussian_mean[sample, guide],
                gaussian_std=fits.gaussian_std[sample, guide],
                gaussian_weight=fits.gaussian_weight[sample, guide],
            )
            # The threshold is the first count where the Gaussian wins.
            assert posterior[-1] > 0.5
            assert (posterior[:-1] <= 0.5).all()
            column = counts[samples == sample, guide]
            share = (column >= 8).sum() / (column > 0).sum()
            assert abs(fits.gaussian_weight[sample, guide] - share) < 0.02


def test_poisson_gaussian_estimate_is_a_maximum_of_the_posterior() -> None:
    _, _, found = _screen(11, n_samples=1, n_guides=2)
    fits = fit_poisson_gaussian(found)
    for guide in range(2):
        counts, cells = found.histogram(0, guide)
        values = np.log2(counts.astype(np.float64))
        estimate = np.asarray(
            [
                fits.poisson_rate[0, guide],
                fits.gaussian_mean[0, guide],
                fits.gaussian_std[0, guide],
                fits.gaussian_weight[0, guide],
            ]
        )
        loss = _poisson_gaussian_loss(estimate, values, cells)
        assert loss == pytest.approx(fits.loss[0, guide], rel=1e-9)

        def unconstrained(point, values=values, cells=cells):
            rate, mean, std, weight = (
                math.exp(point[0]),
                point[1],
                math.exp(point[2]),
                1.0 / (1.0 + math.exp(-point[3])),
            )
            return _poisson_gaussian_loss(
                np.asarray([rate, mean, std, weight]), values, cells
            )

        start = np.asarray(
            [
                math.log(estimate[0]),
                estimate[1],
                math.log(estimate[2]),
                math.log(estimate[3] / (1 - estimate[3])),
            ]
        )
        better = minimize(
            unconstrained,
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
        )
        assert better.fun > loss - 1e-6 * abs(loss)
        np.testing.assert_allclose(better.x, start, atol=2e-3)


def test_poisson_gaussian_fit_of_a_histogram_equals_the_fit_of_every_cell() -> None:
    _, _, found = _screen(12, n_cells=1_500)
    binned = fit_poisson_gaussian(found)
    every_cell = fit_poisson_gaussian(_per_cell(found))
    np.testing.assert_array_equal(binned.threshold, every_cell.threshold)
    np.testing.assert_array_equal(binned.status, every_cell.status)
    for name in ("poisson_rate", "gaussian_mean", "gaussian_std", "gaussian_weight"):
        np.testing.assert_allclose(
            getattr(binned, name), getattr(every_cell, name), rtol=1e-3
        )


def test_poisson_gaussian_fits_each_sample_on_its_own() -> None:
    noisy = screen_counts(3_000, 2, seed=13, ambient=2.0)
    quiet = screen_counts(3_000, 2, seed=14, ambient=0.1)
    samples = np.repeat([0, 1], 3_000)
    found = guide_count_histograms([noisy, quiet], samples, n_samples=2, n_guides=2)
    fits = fit_poisson_gaussian(found)
    assert (fits.threshold[0] > fits.threshold[1]).all()
    alone = fit_poisson_gaussian(
        guide_count_histograms(
            [quiet], np.zeros(3_000, dtype=np.int64), n_samples=1, n_guides=2
        )
    )
    np.testing.assert_array_equal(alone.threshold[0], fits.threshold[1])


def test_each_fit_status_of_the_poisson_gaussian() -> None:
    fits = fit_poisson_gaussian(_guides(_STATUS_GUIDES, n_cells=6_000))
    assert fits.status[0].tolist() == _STATUS_EXPECTED
    assert [FIT_STATUSES[code] for code in fits.status[0]] == [
        "no_cells",
        "too_sparse",
        "too_sparse",
        "no_fit",
        "no_positive",
        "fitted",
    ]
    assert fits.threshold[0, :5].tolist() == [0, 0, 0, 0, 0]
    assert 3 < fits.threshold[0, 5] <= 40
    unfitted = fits.status[0] != GUIDE_FIT_FITTED
    assert np.isnan(fits.gaussian_mean[0, :4]).all()
    assert fits.iterations[0, :4].tolist() == [0, 0, 0, 0]
    assert np.isfinite(fits.gaussian_mean[0, 4:]).all()
    assert unfitted.sum() == 5


def test_gaussian_matches_scikit_learn() -> None:
    counts = screen_counts(3_000, 3, seed=15)
    found = guide_count_histograms(
        [counts], np.zeros(3_000, dtype=np.int64), n_samples=1, n_guides=3
    )
    fits = fit_gaussian(found)
    assert fits.status.tolist() == [[GUIDE_FIT_FITTED] * 3]
    for guide in range(3):
        values = np.log10(counts[:, guide].astype(np.float64) + 1).reshape(-1, 1)
        model = max(
            (
                GaussianMixture(
                    2,
                    covariance_type="tied",
                    tol=1e-12,
                    max_iter=10_000,
                    random_state=seed,
                ).fit(values)
                for seed in range(3)
            ),
            key=lambda fitted, values=values: fitted.score(values),
        )
        order = np.argsort(model.means_.ravel())
        low, high = model.means_.ravel()[order]
        assert fits.low_mean[0, guide] == pytest.approx(low, abs=1e-4)
        assert fits.high_mean[0, guide] == pytest.approx(high, abs=1e-4)
        assert fits.std[0, guide] == pytest.approx(
            math.sqrt(model.covariances_.ravel()[0]), abs=1e-4
        )
        assert fits.high_weight[0, guide] == pytest.approx(
            model.weights_[order[1]], abs=1e-4
        )
        assert fits.log_likelihood[0, guide] == pytest.approx(
            model.score(values), abs=1e-8
        )
        grid = np.arange(1, counts[:, guide].max() + 1)
        posterior = model.predict_proba(np.log10(grid + 1.0).reshape(-1, 1))[
            :, order[1]
        ]
        assert fits.threshold[0, guide] == grid[np.flatnonzero(posterior > 0.5)[0]]


def test_gaussian_finds_carriers_beside_many_ambient_cells() -> None:
    # Most cells hold no UMI of the guide, many hold one or two ambient UMIs,
    # and ten cells carry it. The best two-group split puts the ambient cells
    # in the higher group, so only the other starts find the carriers.
    found = _guides([([1, 2, 150, 170, 200], [300, 50, 4, 3, 3])], n_cells=5_360)
    fits = fit_gaussian(found)
    assert fits.status.tolist() == [[GUIDE_FIT_FITTED]]
    assert fits.high_mean[0, 0] == pytest.approx(math.log10(171), abs=0.1)
    assert fits.high_weight[0, 0] == pytest.approx(10 / 5_360, rel=1e-3)
    assert 2 < fits.threshold[0, 0] <= 150


def test_gaussian_threshold_is_where_the_higher_component_wins() -> None:
    _, _, found = _screen(16, n_samples=1, n_guides=2)
    fits = fit_gaussian(found)
    for guide in range(2):
        threshold = fits.threshold[0, guide]
        posterior = gaussian_posterior(
            np.asarray([threshold - 1, threshold]),
            low_mean=fits.low_mean[0, guide],
            high_mean=fits.high_mean[0, guide],
            std=fits.std[0, guide],
            high_weight=fits.high_weight[0, guide],
        )
        assert posterior[0] <= 0.5 < posterior[1]


def test_gaussian_statuses_without_cells_or_with_one_count() -> None:
    found = _guides([([], []), ([3], [10]), ([1, 40], [100, 5])], n_cells=10)
    # The third guide holds 105 cells of a 10-cell sample only in this stub,
    # which keeps every cell of the second guide at one count.
    fits = fit_gaussian(found)
    assert fits.status[0].tolist() == [
        GUIDE_FIT_NO_CELLS,
        GUIDE_FIT_TOO_SPARSE,
        GUIDE_FIT_FITTED,
    ]
    assert np.isnan(fits.high_mean[0, :2]).all()


def test_gaussian_statuses_of_degenerate_fits(monkeypatch) -> None:
    found = _guides([([1, 40], [100, 5]), ([1, 40], [100, 5])], n_cells=500)
    use_python_kernels(monkeypatch, mixtures)
    degenerate = (math.nan, math.nan, math.nan, math.nan, -math.inf, 0)
    group = mixtures._gaussian_group
    calls = iter([degenerate, None])

    def first_degenerate(*args):
        result = next(calls)
        return group(*args) if result is None else result

    monkeypatch.setattr(mixtures, "_gaussian_group", first_degenerate)
    monkeypatch.setattr(mixtures, "_gaussian_threshold", lambda *args: 0)
    fits = fit_gaussian(found)
    assert fits.status[0].tolist() == [GUIDE_FIT_NO_FIT, GUIDE_FIT_NO_POSITIVE]
    assert fits.threshold[0].tolist() == [0, 0]
    assert np.isnan(fits.low_mean[0, 0])
    assert np.isfinite(fits.low_mean[0, 1])


@pytest.mark.parametrize("compiled", [True, False])
def test_threshold_kernels_return_zero_when_no_count_is_called(compiled) -> None:
    poisson = mixtures._poisson_gaussian_threshold
    gaussian = mixtures._gaussian_threshold
    if not compiled:
        poisson, gaussian = poisson.py_func, gaussian.py_func
    # A Gaussian far above every count, holding almost no cells.
    assert poisson(1.0, 12.0, 0.5, 1e-5, 20) == 0
    assert poisson(0.2, 3.0, 1.0, 0.3, 20) > 0
    # The higher component, near 100 UMIs, wins only far above the largest count.
    assert gaussian(0.0, 2.0, 0.01, 1e-9, 5) == 0
    for args in [(0.1, 1.5, 0.05, 0.02), (0.0, 2.0, 0.2, 0.3), (0.2, 0.9, 0.01, 0.5)]:
        found = gaussian(*args, 500)
        linear = next(
            count
            for count in range(1, 501)
            if gaussian_posterior(
                np.asarray([count]),
                low_mean=args[0],
                high_mean=args[1],
                std=math.sqrt(args[2]),
                high_weight=args[3],
            )[0]
            > 0.5
        )
        assert found == linear


def test_gaussian_em_gives_up_on_a_degenerate_fit() -> None:
    weights = np.asarray([5.0, 1.0, 1.0])
    for em in (mixtures._gaussian_em, mixtures._gaussian_em.py_func):
        # A value that is not a number leaves no cell in either component.
        lost = em(
            np.asarray([0.0, math.nan, 1.0]), weights, 7.0, 0, 100, 1e-10, np.empty(3)
        )
        # Without an iteration, no likelihood is ever computed.
        unrun = em(np.asarray([0.0, 0.5, 1.0]), weights, 7.0, 0, 0, 1e-10, np.empty(3))
        for result in (lost, unrun):
            assert all(math.isnan(value) for value in result[:5])


def test_python_kernels_fit_like_the_compiled_ones(monkeypatch) -> None:
    found = _guides(_STATUS_GUIDES, n_cells=6_000)
    expected = (fit_poisson_gaussian(found), fit_gaussian(found))
    use_python_kernels(monkeypatch, mixtures)
    for fits, reference in zip(
        (fit_poisson_gaussian(found), fit_gaussian(found)), expected
    ):
        np.testing.assert_array_equal(fits.status, reference.status)
        np.testing.assert_array_equal(fits.threshold, reference.threshold)
        for name in fits.__slots__:
            np.testing.assert_allclose(
                getattr(fits, name), getattr(reference, name), rtol=1e-6
            )
