"""Benchmarks of the random sets of high-MOI association tests.

A high-MOI test compares each group of treatment cells with all other cells,
so the first round's 499 random sets of every group are drawn from every cell
and scored on every gene of a band. With conditional resampling each cell
enters a set with its own probability. One benchmark times a group's sets,
from its detection model to the drawn cells; the other times the scoring of
those sets on a band of 64 genes, held cell by cell, the work that grows with
the cells, the treated cells and the genes. A group holds about 1% of the
cells. Each benchmark checks its result, so it doubles as a test at the smoke
size.
"""

from functools import lru_cache

import numpy as np
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

LADDER = Ladder(sizes=(50_000, 100_000, 200_000, 400_000), smoke=2_000)
N_SETS = 499
N_BAND = 64
# A detection model whose probabilities average about 1%, with the spread
# over cells that sequencing depth gives.
MODEL = np.array([-4.8, 0.8, -0.4, 0.2, 0.0, 0.0, 0.0])


@lru_cache(maxsize=1)
def _covariates(n_cells: int) -> np.ndarray:
    """An intercept and six standard normal covariates per cell."""
    rng = np.random.default_rng(inputs.SEED)
    return np.column_stack([np.ones(n_cells), rng.standard_normal((n_cells, 6))])


def _keys() -> list[np.uint64]:
    """The keys of a group's first-round sets, as uint64 for compiled calls."""
    from cytearc.perturbation.random_sets import resampling_keys, set_key

    key = resampling_keys(0, np.array([0]), 1)[0]
    return [np.uint64(set_key(key, d)) for d in range(N_SETS)]


@lru_cache(maxsize=1)
def _band(n_cells: int) -> tuple[np.ndarray, ...]:
    """A band's cell-major scores and weights, and the Cholesky factors of its fits."""
    rng = np.random.default_rng([inputs.SEED, 1])
    covariates = _covariates(n_cells)
    scores = rng.standard_normal((n_cells, N_BAND))
    weights = rng.uniform(0.1, 2.0, (n_cells, N_BAND))
    information = np.einsum("ik,ic,id->kcd", weights, covariates, covariates)
    factors = np.ascontiguousarray(np.linalg.cholesky(information))
    observed = rng.normal(0.0, 1.0, N_BAND)
    return scores, weights, factors, observed


def test_conditional_sets(bench) -> None:
    from cytearc.perturbation.high_moi import _sampler
    from cytearc.perturbation.random_sets import draw_bernoulli_set

    def make(n_cells: int):
        covariates = _covariates(n_cells)
        out = np.empty(n_cells, dtype=np.int64)
        keys = _keys()

        def call():
            probabilities, order, offsets, bounds = _sampler(covariates, MODEL, True)
            sizes = np.array(
                [
                    draw_bernoulli_set(key, probabilities, order, offsets, bounds, out)
                    for key in keys
                ]
            )
            return probabilities, sizes

        return call

    def check(n_cells: int, value) -> None:
        probabilities, sizes = value
        expected = 1 / (1 + np.exp(-_covariates(n_cells) @ MODEL))
        np.testing.assert_allclose(probabilities, expected, rtol=1e-12)
        # A set's size is a sum of independent Bernoulli draws.
        mean = expected.sum()
        variance = (expected * (1 - expected)).sum()
        assert abs(sizes.mean() - mean) < 5 * np.sqrt(variance / N_SETS)
        assert 0.7 < sizes.var() / variance < 1.3

    bench("perturbation.conditional_sets", make, LADDER, check=check)


def test_band_gather(bench) -> None:
    from cytearc.perturbation.high_moi import _sampler, _score_sets
    from cytearc.perturbation.random_sets import draw_bernoulli_set, resampling_keys

    key = resampling_keys(0, np.array([0]), 1)[0]
    pairs = np.arange(N_BAND)

    def make(n_cells: int):
        covariates = _covariates(n_cells)
        scores, weights, factors, observed = _band(n_cells)
        sampler = _sampler(covariates, MODEL, True)
        no_statistics = np.empty((0, N_SETS))

        def call():
            counts = np.zeros((N_BAND, 2), dtype=np.int64)
            _score_sets(
                0,
                1,
                N_SETS,
                key,
                True,
                *sampler,
                0,
                scores,
                weights,
                factors,
                0,
                N_BAND,
                covariates,
                pairs,
                pairs,
                observed,
                counts,
                no_statistics,
            )
            return counts

        return call

    def check(n_cells: int, counts) -> None:
        # The same sets, scored by the formula with NumPy's sums and solves.
        covariates = _covariates(n_cells)
        scores, weights, _, observed = _band(n_cells)
        sampler = _sampler(covariates, MODEL, True)
        out = np.empty(n_cells, dtype=np.int64)
        information = np.einsum("ik,ic,id->kcd", weights, covariates, covariates)
        expected = np.zeros((N_BAND, 2), dtype=np.int64)
        for set_key in _keys():
            cells = out[: draw_bernoulli_set(set_key, *sampler, out)]
            weighted = np.einsum("ik,ic->kc", weights[cells], covariates[cells])
            solved = np.linalg.solve(information, weighted[:, :, None])[:, :, 0]
            variance = weights[cells].sum(0) - (weighted * solved).sum(1)
            statistics = scores[cells].sum(0) / np.sqrt(variance)
            expected[:, 0] += statistics <= observed
            expected[:, 1] += statistics >= observed
        np.testing.assert_array_equal(counts, expected)

    bench("perturbation.band_gather", make, LADDER, check=check)
