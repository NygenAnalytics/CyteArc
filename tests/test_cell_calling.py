"""EmptyDrops cell calling (Lun et al. 2019) against independent references.

The references below are written from the papers with SciPy: Simple
Good-Turing from Gale and Sampson (1995), the Dirichlet-multinomial
likelihood from ``scipy.stats``, the precision estimate from the digamma
score of every ambient count vector, and Monte Carlo p-values from the
closed-form probability of each simulated count vector.
"""

import math

import numpy as np
import pytest
from loguru import logger
from scipy import sparse
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.stats import dirichlet_multinomial, multinomial
from statsmodels.stats.multitest import multipletests

from cytearc.quality_control import (
    BarcodeRanks,
    EmptyDropsResult,
    barcode_ranks,
    empty_drops,
    good_turing_proportions,
)
from cytearc.quality_control import cell_calling as cc

# References -----------------------------------------------------------------


def _reference_good_turing(counts: list[int]) -> np.ndarray:
    """Simple Good-Turing, step by step as Gale and Sampson (1995) give it."""
    seen = sorted({count for count in counts if count > 0})
    species = {count: counts.count(count) for count in seen}
    total = sum(counts)
    unseen_share = species.get(1, 0) / total
    averaged = {}
    for index, count in enumerate(seen):
        before = seen[index - 1] if index else 0
        after = seen[index + 1] if index + 1 < len(seen) else 2 * count - before
        averaged[count] = species[count] / (0.5 * (after - before))
    xs = [math.log(count) for count in seen]
    ys = [math.log(averaged[count]) for count in seen]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x

    def smoothed(count: float) -> float:
        return math.exp(intercept + slope * math.log(count))

    adjusted = {}
    switched = False
    for count in seen:
        estimate = (count + 1) * smoothed(count + 1) / smoothed(count)
        following = species.get(count + 1)
        if not switched and following:
            turing = (count + 1) * following / species[count]
            spread = 1.96 * math.sqrt(
                (count + 1) ** 2
                * following
                / species[count] ** 2
                * (1 + following / species[count])
            )
            if abs(estimate - turing) > spread:
                adjusted[count] = turing
                continue
        switched = True
        adjusted[count] = estimate
    normalizer = sum(species[count] * adjusted[count] for count in seen)
    n_unseen = counts.count(0)
    values = np.array(
        [
            unseen_share / n_unseen
            if count == 0
            else (1 - unseen_share) * adjusted[count] / normalizer
            for count in counts
        ]
    )
    return values if n_unseen else values / values.sum()


def _dm_log_probability(counts: np.ndarray, proportions: np.ndarray, alpha: float):
    total = int(counts.sum())
    if math.isinf(alpha):
        return multinomial.logpmf(counts, total, proportions)
    return dirichlet_multinomial.logpmf(counts, alpha * proportions, total)


def _reference_alpha(ambient: np.ndarray, proportions: np.ndarray) -> float:
    """Root of the digamma score of the full ambient likelihood in log(alpha)."""
    counts = ambient[:, proportions > 0]
    p = proportions[proportions > 0]
    totals = counts.sum(axis=1)

    def score(log_alpha: float) -> float:
        alpha = math.exp(log_alpha)
        per_barcode = (
            digamma(alpha)
            - digamma(totals + alpha)
            + (p * (digamma(counts + alpha * p) - digamma(alpha * p))).sum(axis=1)
        )
        return alpha * float(per_barcode.sum())

    return math.exp(brentq(score, math.log(1e-2), math.log(1e7), xtol=1e-14))


def _reference_p_values(
    totals: np.ndarray,
    counts: np.ndarray,
    proportions: np.ndarray,
    alpha: float,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count simulated count vectors at most as probable as each barcode."""
    sampler = cc._sampler(proportions, alpha)
    observed = np.array(
        [_dm_log_probability(row, proportions, alpha) for row in counts]
    )
    threshold = observed + np.where(
        np.isfinite(observed), 1e-9 * np.maximum(1.0, np.abs(observed)), 0.0
    )
    largest = int(totals.max())
    exceeded = np.zeros(len(totals), dtype=np.int64)
    for iteration in range(n_iter):
        features = cc._simulated_features(
            iteration,
            np.uint64(seed),
            largest,
            sampler.alpha,
            sampler.multinomial,
            sampler.cumulative,
            sampler.guide,
        )
        simulated = np.zeros(len(proportions), dtype=np.int64)
        for drawn, feature in enumerate(features, start=1):
            simulated[feature] += 1
            same_total = totals == drawn
            if same_total.any():
                value = _dm_log_probability(simulated, proportions, alpha)
                exceeded[same_total & (value <= threshold)] += 1
    return (exceeded + 1) / (n_iter + 1), exceeded == 0


def _dirichlet_multinomial_molecules(
    rng: np.random.Generator,
    totals: np.ndarray,
    proportions: np.ndarray,
    precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw Dirichlet-multinomial count vectors one molecule at a time.

    The Polya urn draws them exactly: molecule m of a droplet, counting from
    0, is a fresh draw from ``proportions`` with probability precision /
    (precision + m), and otherwise takes the feature of one of the droplet's
    m earlier molecules, chosen uniformly. It is feature j with probability
    (precision * p_j + c_j) / (precision + m), where c_j of the earlier
    molecules are feature j, as the Dirichlet-multinomial gives. The cost is
    a few draws per molecule, not a draw per feature and droplet.

    Returns:
        The droplet and the feature of each molecule.
    """
    owner = np.repeat(np.arange(len(totals)), totals)
    first = np.repeat(np.cumsum(totals) - totals, totals)
    earlier = np.arange(len(owner)) - first
    fresh = rng.random(len(owner)) * (precision + earlier) < precision
    features = rng.choice(len(proportions), size=len(owner), p=proportions)
    # A copy points at an earlier molecule, which may be a copy itself.
    source = np.where(
        fresh,
        np.arange(len(owner)),
        first + (rng.random(len(owner)) * earlier).astype(np.int64),
    )
    while not np.array_equal(source[source], source):
        source = source[source]
    return owner, features[source]


def _droplets(
    seed: int = 0,
    *,
    n_features: int = 400,
    n_empty: int = 4000,
    n_cells: int = 60,
    n_small: int = 30,
    precision: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate empty droplets of an ambient profile and cells of four types.

    Empty droplets draw their counts from the Dirichlet-multinomial of the
    ambient profile and the given precision, as the model assumes. Cells of
    four marker-enriched profiles hold 1,000 to 6,000 counts, and small cells
    150 to 400, near the ambient level.
    """
    rng = np.random.default_rng(seed)
    ambient = rng.gamma(0.3, 1.0, n_features)
    ambient /= ambient.sum()
    totals = np.maximum(1, rng.lognormal(math.log(12), 0.9, n_empty)).astype(int)
    counts = np.zeros((n_empty + n_cells + n_small, n_features), dtype=np.int64)
    molecules = _dirichlet_multinomial_molecules(rng, totals, ambient, precision)
    np.add.at(counts, molecules, 1)
    profiles = []
    for _ in range(4):
        profile = ambient.copy()
        markers = rng.choice(n_features, 25, replace=False)
        profile[markers] += rng.gamma(2.0, 1.0, 25) * 0.03
        profiles.append(profile / profile.sum())
    for index, row in enumerate(range(n_empty, n_empty + n_cells)):
        total = int(rng.integers(1000, 6000))
        counts[row] = rng.multinomial(total, profiles[index % 4])
    for index, row in enumerate(range(n_empty + n_cells, len(counts))):
        total = int(rng.integers(150, 400))
        counts[row] = rng.multinomial(total, profiles[index % 4])
    truth = np.r_[np.zeros(n_empty, dtype=bool), np.ones(n_cells + n_small, dtype=bool)]
    return counts, truth, ambient


@pytest.fixture(scope="module")
def droplets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _droplets()


# Good-Turing ----------------------------------------------------------------


def test_good_turing_matches_gale_and_sampson_with_turing_estimates() -> None:
    frequencies = {1: 3000, 2: 600, 3: 300, 4: 200, 5: 90, 6: 70, 8: 40, 11: 20, 30: 10}
    counts = [0] * 300 + [
        count for count, n_count in frequencies.items() for _ in range(n_count)
    ]
    # The Turing estimate of r = 1, 2 * N2 / N1, differs from the smoothed
    # estimate by more than 1.96 standard deviations, so it is used.
    seen = sorted(frequencies)
    edges = [0, *seen[:-1]], [*seen[1:], 2 * seen[-1] - seen[-2]]
    x = np.log(seen)
    y = np.log(
        [
            frequencies[count] / (0.5 * (after - before))
            for count, before, after in zip(seen, *edges, strict=True)
        ]
    )
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    spread = 1.96 * math.sqrt(4 * 600 / 3000**2 * (1 + 600 / 3000))
    assert abs(2 * 2**slope - 2 * 600 / 3000) > spread

    observed = good_turing_proportions(np.array(counts))

    np.testing.assert_allclose(observed, _reference_good_turing(counts), rtol=1e-12)
    assert observed.sum() == pytest.approx(1.0, abs=1e-12)
    # Unseen species share N1 / N equally.
    np.testing.assert_allclose(observed[:300], 3000 / sum(counts) / 300, rtol=1e-12)


def test_good_turing_smooths_from_the_first_count_by_hand() -> None:
    # N1 = 3, N2 = 2, N3 = 1 and N = 10: Z = 3, 2, 1 at r = 1, 2, 3.
    counts = np.array([1, 1, 1, 2, 2, 3, 0, 0])
    x = np.log([1.0, 2.0, 3.0])
    y = np.log([3.0, 2.0, 1.0])
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    adjusted = np.array([(r + 1) * (1 + 1 / r) ** slope for r in (1, 2, 3)])
    # The Turing estimate 2 * 2 / 3 is within 1.96 deviations of the smoothed
    # estimate, so every count is smoothed.
    assert abs(adjusted[0] - 4 / 3) <= 1.96 * math.sqrt(4 * 2 / 9 * (1 + 2 / 3))
    seen_share = 1 - 3 / 10
    normalizer = 3 * adjusted[0] + 2 * adjusted[1] + adjusted[2]
    expected = np.r_[
        np.repeat(seen_share * adjusted / normalizer, [3, 2, 1]), [0.15, 0.15]
    ]

    np.testing.assert_allclose(good_turing_proportions(counts), expected, rtol=1e-12)
    np.testing.assert_allclose(
        good_turing_proportions(counts), _reference_good_turing(list(counts))
    )


def test_good_turing_without_unseen_species_sums_to_one() -> None:
    counts = [1] * 20 + [2] * 6 + [3] * 3 + [5] * 2 + [9]
    observed = good_turing_proportions(np.array(counts, dtype=np.uint16))

    np.testing.assert_allclose(observed, _reference_good_turing(counts), rtol=1e-12)
    assert observed.sum() == pytest.approx(1.0, abs=1e-12)


def test_good_turing_warns_about_a_rising_frequency_fit() -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        observed = good_turing_proportions(np.array([1, 2, 2, 2, 2, 2, 0]))
    finally:
        logger.remove(sink)

    assert any("slope 2.322" in message for message in messages)
    np.testing.assert_allclose(
        observed, _reference_good_turing([1, 2, 2, 2, 2, 2, 0]), rtol=1e-12
    )


@pytest.mark.parametrize(
    ("counts", "error", "message"),
    [
        (np.array([[1, 2], [3, 4]]), ValueError, "one-dimensional"),
        (np.array([1.0, 2.0]), TypeError, "integers"),
        (np.array([1, -1, 2]), ValueError, "non-negative"),
        (np.array([0, 3, 3, 3]), ValueError, "two distinct positive"),
        (np.array([0, 0]), ValueError, "two distinct positive"),
    ],
)
def test_good_turing_rejects_invalid_counts(counts, error, message) -> None:
    with pytest.raises(error, match=message):
        good_turing_proportions(counts)


# Barcode ranks --------------------------------------------------------------


def test_barcode_ranks_average_tied_ranks_and_need_five_totals() -> None:
    ranks = barcode_ranks(np.array([10, 5, 5, 1, 0, 120, 130]), lower=100)

    assert isinstance(ranks, BarcodeRanks)
    np.testing.assert_array_equal(ranks.total, [130, 120, 10, 5, 1, 0])
    np.testing.assert_array_equal(ranks.rank, [1, 2, 3, 4.5, 6, 7])
    assert np.isnan(ranks.fitted).all()
    assert math.isnan(ranks.knee) and math.isnan(ranks.inflection)
    assert ranks.lower == 100
    table = ranks.to_pandas()
    assert list(table.columns) == ["rank", "total", "fitted"]
    np.testing.assert_array_equal(table["total"], ranks.total)


def test_barcode_ranks_find_the_knee_of_a_cliff_and_its_steepest_point() -> None:
    # log10(total) falls with slope -0.25 over ranks 1 to 1,000 and with
    # slope -6 after them: the curve bends at rank 1,000 (total 1,778).
    rank = np.arange(1, 4001)
    log_total = np.where(
        rank <= 1000,
        4 - 0.25 * np.log10(rank),
        3.25 - 6 * (np.log10(rank) - 3),
    )
    totals = np.round(10.0**log_total)
    ranks = barcode_ranks(totals)

    assert 1778 <= ranks.knee <= 2500
    assert 100 < ranks.inflection < 1778
    fitted = ranks.total > 100
    assert np.isfinite(ranks.fitted[fitted]).all()
    assert np.isnan(ranks.fitted[~fitted]).all()
    # The smoothed curve follows the totals it fits, and rounds the hard
    # corner of the bend by less than 15%.
    relative = np.abs(ranks.fitted[fitted] / ranks.total[fitted] - 1)
    assert np.median(relative) < 0.01
    assert relative.max() < 0.15
    # The knee moves with the bend.
    later = barcode_ranks(totals * 2.0, lower=200)
    assert later.knee == pytest.approx(2 * ranks.knee, rel=0.2)


@pytest.mark.parametrize(
    ("totals", "lower", "error", "message"),
    [
        (np.ones((2, 2)), 100, ValueError, "one-dimensional"),
        (np.array(["a", "b"]), 100, TypeError, "real numbers"),
        (np.array([1.0, np.nan]), 100, ValueError, "finite and non-negative"),
        (np.array([1.0, -1.0]), 100, ValueError, "finite and non-negative"),
        (np.array([1, 2]), -1, ValueError, "lower must be at least 0"),
        (np.array([1, 2]), True, TypeError, "lower must be an integer"),
    ],
)
def test_barcode_ranks_reject_invalid_arguments(totals, lower, error, message) -> None:
    with pytest.raises(error, match=message):
        barcode_ranks(totals, lower=lower)


# Ambient pool and alpha -----------------------------------------------------


def _ambient_counts(seed: int, precision: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    proportions = rng.gamma(0.6, 1.0, 40)
    proportions /= proportions.sum()
    totals = rng.integers(2, 60, 3000)
    rows = [
        rng.multinomial(total, rng.dirichlet(proportions * precision))
        for total in totals
    ]
    return np.asarray(rows, dtype=np.int64), proportions


def _pool(counts: np.ndarray, lower: int) -> cc._AmbientPool:
    totals = counts.sum(axis=1)
    keep = totals > 0
    return cc._collect_ambient(
        [counts[keep][:700], counts[keep][700:]],
        totals[keep],
        counts.shape[1],
        lower,
        keep_tested=False,
    )


def test_alpha_matches_the_maximum_of_the_full_ambient_likelihood() -> None:
    counts, truth = _ambient_counts(4, precision=30.0)
    pool = _pool(counts, lower=100)
    proportions = cc._ambient_proportions(pool)

    estimate = cc._estimate_alpha(cc._alpha_terms(pool, proportions))

    assert estimate == pytest.approx(_reference_alpha(counts, proportions), rel=1e-9)
    # The estimate recovers the simulated precision.
    assert 20 < estimate < 45
    np.testing.assert_array_equal(pool.counts, counts.sum(axis=0))


def test_alpha_takes_the_multinomial_without_overdispersion() -> None:
    # Barcodes that count no feature twice look underdispersed: the
    # likelihood rises with alpha up to the multinomial.
    rng = np.random.default_rng(2)
    counts = np.zeros((400, 60), dtype=np.int64)
    for row in counts:
        row[rng.choice(60, int(rng.integers(2, 9)), replace=False)] = 1
    pool = _pool(counts, lower=100)
    terms = cc._alpha_terms(pool, cc._ambient_proportions(pool))

    assert terms.weight.size == 0
    assert cc._estimate_alpha(terms) == math.inf


def test_alpha_of_single_count_barcodes_is_the_multinomial() -> None:
    # Barcodes with one count each do not inform alpha, so the likelihood is
    # flat and the multinomial wins the tie.
    counts = np.zeros((30, 6), dtype=np.int64)
    counts[np.arange(30), np.arange(30) % 6] = 1
    terms = cc._alpha_terms(_pool(counts, lower=100), np.full(6, 1 / 6))

    assert terms.score(0.0) == 0.0
    assert terms.log_likelihood(1e-3) == terms.log_likelihood(math.inf) == 0.0
    assert cc._estimate_alpha(terms) == math.inf


def test_alpha_of_single_feature_barcodes_is_the_smallest_precision() -> None:
    # Every barcode counts one feature several times: overdispersion without
    # bound, so the likelihood falls with alpha everywhere.
    counts = np.zeros((60, 5), dtype=np.int64)
    counts[np.arange(60), np.arange(60) % 5] = np.arange(60) % 7 + 2
    pool = _pool(counts, lower=100)
    terms = cc._alpha_terms(pool, cc._ambient_proportions(pool))

    assert terms.score(math.log(1e-3)) < 0
    assert cc._estimate_alpha(terms) == pytest.approx(1e-3)


def test_ambient_pool_reads_dense_and_sparse_blocks_alike() -> None:
    counts, _ = _ambient_counts(5, precision=10.0)
    counts[::50] *= 4  # Some barcodes above lower, which count as expressed only.
    totals = counts.sum(axis=1)
    dense = cc._collect_ambient(
        [counts[:1000], counts[1000:]], totals, counts.shape[1], 60, keep_tested=True
    )
    sparse_pool = cc._collect_ambient(
        [sparse.csr_matrix(counts[:1500]), sparse.csr_matrix(counts[1500:])],
        totals,
        counts.shape[1],
        60,
        keep_tested=True,
    )
    ambient = counts[totals <= 60]

    for pool in (dense, sparse_pool):
        np.testing.assert_array_equal(pool.counts, ambient.sum(axis=0))
        np.testing.assert_array_equal(pool.expressed, counts.sum(axis=0) > 0)
        np.testing.assert_array_equal(pool.totals, totals[totals <= 60])
        expected = np.zeros((counts.shape[1], 61), dtype=np.int64)
        for row in ambient:
            for feature in np.flatnonzero(row):
                expected[feature, row[feature]] += 1
        np.testing.assert_array_equal(pool.histogram, expected)
        assert pool.tested is not None
        np.testing.assert_array_equal(
            sparse.vstack(pool.tested).toarray(), counts[totals > 60]
        )


def test_ambient_pool_refuses_counts_that_disagree_with_totals() -> None:
    counts = np.array([[1, 2, 0], [0, 5, 1]])
    for block in (counts, sparse.csr_matrix(counts)):
        with pytest.raises(RuntimeError, match="barcode 1 add up to 6, not .* 5"):
            cc._collect_ambient([block], np.array([3, 5]), 3, 100, keep_tested=False)
        # A count above lower cannot belong to an ambient barcode.
        with pytest.raises(RuntimeError, match="barcode 1 add up to 6, not .* 4"):
            cc._collect_ambient([block], np.array([3, 4]), 3, 4, keep_tested=False)
    with pytest.raises(RuntimeError, match="Read 2 barcodes .* expected 3"):
        cc._collect_ambient([counts], np.array([3, 6, 1]), 3, 100, keep_tested=False)


def test_ambient_profile_needs_counts_from_one_to_lower() -> None:
    counts = np.array([[200, 1, 0], [0, 300, 5]])
    pool = cc._collect_ambient([counts], counts.sum(axis=1), 3, 100, keep_tested=False)
    with pytest.raises(ValueError, match="filtering_cutoff=0"):
        cc._ambient_proportions(pool)


# Likelihood -----------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.7, 25.0, math.inf])
def test_log_probabilities_match_the_dirichlet_multinomial(alpha) -> None:
    rng = np.random.default_rng(8)
    proportions = rng.dirichlet(np.ones(9))
    counts = rng.poisson(rng.uniform(0.5, 30, (12, 1)) * proportions)
    counts[0, :] = 0
    counts[0, 3] = 40
    counts = counts[counts.sum(axis=1) > 0]
    totals = counts.sum(axis=1)
    sampler = cc._sampler(proportions, alpha)

    parts, read_totals = cc._feature_parts(
        [counts[:5], sparse.csr_matrix(counts[5:])], len(counts), sampler
    )
    observed = cc._total_parts(read_totals, alpha) + parts

    np.testing.assert_array_equal(read_totals, totals)
    expected = [_dm_log_probability(row, proportions, alpha) for row in counts]
    np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-10)


def test_a_count_of_a_feature_without_proportion_is_impossible() -> None:
    proportions = np.array([0.5, 0.5, 0.0])
    sampler = cc._sampler(proportions, 4.0)
    parts, _ = cc._feature_parts([np.array([[1, 1, 0], [1, 0, 2]])], 2, sampler)

    assert np.isfinite(parts[0])
    assert parts[1] == -np.inf
    with pytest.raises(RuntimeError, match="Read 2 barcodes to test, expected 3"):
        cc._feature_parts([np.array([[1, 1, 0], [1, 0, 2]])], 3, sampler)


# Monte Carlo p-values -------------------------------------------------------


@pytest.mark.parametrize("alpha", [3.0, math.inf])
def test_monte_carlo_p_values_match_the_closed_form_reference(alpha) -> None:
    proportions = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    counts = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 2],
            [3, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [2, 2, 1, 0, 0],
            [0, 0, 0, 0, 5],
            [3, 2, 1, 1, 1],
            [0, 1, 0, 1, 0],
        ]
    )
    totals = counts.sum(axis=1)
    sampler = cc._sampler(proportions, alpha)
    parts, _ = cc._feature_parts([counts], len(counts), sampler)
    log_probabilities = cc._total_parts(totals, alpha) + parts

    p_values, limited = cc._monte_carlo_p_values(
        totals,
        parts,
        log_probabilities,
        sampler,
        n_iter=300,
        random_seed=11,
        nthreads=2,
    )
    expected, expected_limited = _reference_p_values(
        totals, counts, proportions, alpha, 300, 11
    )

    np.testing.assert_array_equal(p_values, expected)
    np.testing.assert_array_equal(limited, expected_limited)
    assert p_values.min() >= 1 / 301


def test_monte_carlo_counts_draws_as_probable_as_the_barcode() -> None:
    # Two equally likely features: every count vector of total 2 is at most
    # as probable as [1, 1], and half of them are as probable as [2, 0].
    proportions = np.array([0.5, 0.5])
    counts = np.array([[1, 1], [2, 0]])
    totals = counts.sum(axis=1)
    sampler = cc._sampler(proportions, math.inf)
    parts, _ = cc._feature_parts([counts], 2, sampler)
    p_values, limited = cc._monte_carlo_p_values(
        totals,
        parts,
        cc._total_parts(totals, math.inf) + parts,
        sampler,
        n_iter=999,
        random_seed=3,
        nthreads=1,
    )

    assert p_values[0] == 1.0
    assert 0.4 < p_values[1] < 0.6
    assert not limited.any()


def test_simulated_molecules_follow_the_dirichlet_multinomial() -> None:
    proportions = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    total, n_iter = 12, 6000
    for alpha, inflation in ((3.0, (12 + 3.0) / (1 + 3.0)), (math.inf, 1.0)):
        sampler = cc._sampler(proportions, alpha)
        vectors = np.array(
            [
                np.bincount(
                    cc._simulated_features(
                        iteration,
                        np.uint64(5),
                        total,
                        sampler.alpha,
                        sampler.multinomial,
                        sampler.cumulative,
                        sampler.guide,
                    ),
                    minlength=5,
                )
                for iteration in range(n_iter)
            ]
        )
        variance = total * proportions * (1 - proportions) * inflation
        standard_error = np.sqrt(variance / n_iter)
        assert np.all(
            np.abs(vectors.mean(axis=0) - total * proportions) < 5 * standard_error
        )
        np.testing.assert_allclose(vectors.var(axis=0), variance, rtol=0.12)


def test_draw_feature_is_the_exact_inverse_cdf_for_any_guide() -> None:
    proportions = np.array([0.0, 0.2, 0.0, 0.5, 0.3, 0.0])
    sampler = cc._sampler(proportions, math.inf)
    draws = np.r_[np.linspace(0, 1, 2001, endpoint=False), 0.2, 0.7, 1 - 2**-53]
    expected = np.searchsorted(sampler.cumulative, draws, side="right")
    assert set(expected) == {1, 3, 4}
    rng = np.random.default_rng(1)
    guides = [sampler.guide, np.zeros(6, np.int64), np.full(6, 5, np.int64)]
    guides.append(rng.integers(0, 6, 6))
    for guide in guides:
        for kernel in (cc._draw_feature, cc._draw_feature.py_func):
            observed = [kernel(sampler.cumulative, guide, draw) for draw in draws]
            np.testing.assert_array_equal(observed, expected)


def test_first_at_least_is_a_lower_bound_search() -> None:
    values = np.array([-np.inf, 1.0, 1.0, 2.0, 5.0, 0.0, 9.0])
    for kernel in (cc._first_at_least, cc._first_at_least.py_func):
        for target in (-np.inf, 0.5, 1.0, 1.5, 5.0, 6.0):
            expected = 0 + np.searchsorted(values[0:5], target, side="left")
            assert kernel(values, 0, 5, target) == expected
        assert kernel(values, 5, 7, 0.0) == 5
        assert kernel(values, 5, 7, 10.0) == 7


# In-memory EmptyDrops -------------------------------------------------------


@pytest.mark.slow
def test_empty_drops_calls_simulated_cells_and_keeps_empty_droplets_out() -> None:
    counts, truth, _ = _droplets(
        seed=0,
        n_features=2000,
        n_empty=30_000,
        n_cells=400,
        n_small=200,
        precision=2000.0,
    )
    result = empty_drops(counts, random_seed=1, nthreads=2)
    called = result.is_cell()

    # Every cell is called, the small ones near the ambient level included.
    assert called[truth].all()
    tested_empty = ~truth & (result.total > 100)
    assert tested_empty.sum() > 200
    # No empty droplet is called, within the default FDR of 0.1%.
    assert not called[~truth].any()
    assert result.alpha == pytest.approx(2000.0, rel=0.25)
    assert 100 < result.inflection < result.knee == result.retain
    assert result.n_ambient == int(
        ((counts.sum(axis=1) > 0) & (counts.sum(axis=1) <= 100)).sum()
    )


def test_empty_drops_reports_every_barcode(droplets) -> None:
    counts, truth, _ = droplets
    result = empty_drops(counts, n_iter=2000, random_seed=4, nthreads=2)
    totals = counts.sum(axis=1)
    tested = totals > 100
    retained = totals >= result.retain

    assert isinstance(result, EmptyDropsResult)
    np.testing.assert_array_equal(result.total, totals)
    assert np.isnan(result.log_probability[~tested]).all()
    assert np.isnan(result.p_value[~tested]).all()
    assert np.isnan(result.fdr[~tested]).all()
    assert not result.limited[~tested].any()
    # Exactly the barcodes at or above retain are cells with p-value 0.
    assert retained.any()
    np.testing.assert_array_equal(result.p_value == 0, retained)
    assert not result.limited[retained].any()
    assert np.isfinite(result.log_probability[tested]).all()
    simulated = tested & ~retained
    assert (
        (result.p_value[simulated] >= 1 / 2001) & (result.p_value[simulated] <= 1)
    ).all()
    np.testing.assert_array_equal(
        result.limited[simulated], result.p_value[simulated] == 1 / 2001
    )
    expected_fdr = multipletests(result.p_value[tested], method="fdr_bh")[1]
    np.testing.assert_allclose(result.fdr[tested], expected_fdr, rtol=1e-12)
    assert result.ambient.sum() == pytest.approx(1.0)
    assert result.barcodes is None
    assert (result.lower, result.n_iter, result.random_seed) == (100, 2000, 4)
    assert result.is_cell()[truth].mean() > 0.9
    table = result.to_pandas()
    assert table.index.name == "position"
    np.testing.assert_array_equal(table.index, np.flatnonzero(tested))
    assert list(table.columns) == [
        "total",
        "log_probability",
        "p_value",
        "fdr",
        "limited",
    ]


def test_empty_drops_is_reproducible_on_any_number_of_threads(droplets) -> None:
    counts, _, _ = droplets
    first = empty_drops(counts, n_iter=500, random_seed=9, nthreads=1)
    again = empty_drops(
        sparse.csr_matrix(counts), n_iter=500, random_seed=9, nthreads=3
    )
    other = empty_drops(counts, n_iter=500, random_seed=10, nthreads=1)

    for name in ("log_probability", "p_value", "fdr", "limited", "ambient"):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
    assert first.alpha == again.alpha
    assert not np.array_equal(first.p_value, other.p_value, equal_nan=True)


def test_empty_drops_takes_alpha_retain_and_barcodes(droplets) -> None:
    counts, _, _ = droplets
    barcodes = np.array([f"bc{index}" for index in range(len(counts))])
    result = empty_drops(
        counts.astype(np.float32),
        n_iter=200,
        alpha=50.0,
        retain=500,
        lower=60,
        random_seed=2,
        nthreads=1,
        barcodes=barcodes,
    )
    totals = counts.sum(axis=1)

    assert (result.alpha, result.retain, result.lower) == (50.0, 500, 60)
    # Barcodes above lower are tested, and those at or above retain are cells.
    assert (totals >= 500).any()
    np.testing.assert_array_equal(result.p_value == 0, totals >= 500)
    np.testing.assert_array_equal(np.isfinite(result.p_value), totals > 60)
    table = result.to_pandas()
    assert table.index.name == "barcode"
    assert table.index.tolist() == barcodes[totals > 60].tolist()


def test_empty_drops_sums_repeated_sparse_entries_without_changing_input() -> None:
    counts, _, _ = _droplets(seed=3, n_empty=600, n_cells=12, n_small=0)
    # Split every count into two entries of the same row and feature.
    indptr, indices, data = [0], [], []
    for row in counts:
        features = np.flatnonzero(row)
        indices.extend(np.repeat(features, 2))
        data.extend(
            np.ravel(np.c_[row[features] - row[features] // 2, row[features] // 2])
        )
        indptr.append(len(indices))
    doubled = sparse.csr_matrix((data, indices, indptr), shape=counts.shape)
    assert not doubled.has_canonical_format
    entries = doubled.data.copy()

    split = empty_drops(doubled, n_iter=100, retain=1000, random_seed=0, nthreads=1)
    whole = empty_drops(counts, n_iter=100, retain=1000, random_seed=0, nthreads=1)

    np.testing.assert_array_equal(doubled.data, entries)
    np.testing.assert_array_equal(split.p_value, whole.p_value)
    np.testing.assert_array_equal(split.log_probability, whole.log_probability)


def test_empty_drops_requires_barcodes_to_test_and_an_ambient_pool() -> None:
    with pytest.raises(ValueError, match="No barcode has more than lower=100"):
        empty_drops(np.array([[3, 4], [0, 9]]))
    # Four distinct totals above lower leave the knee undefined.
    counts = np.array([[150, 0], [0, 160], [170, 1], [90, 90], [1, 2], [0, 3]])
    with pytest.raises(ValueError, match="knee cannot be located; pass retain"):
        empty_drops(counts)
    result = empty_drops(counts, retain=175, n_iter=50, nthreads=1)
    assert result.retain == 175 and math.isnan(result.knee)
    with pytest.raises(ValueError, match="no ambient pool"):
        empty_drops(np.array([[150, 0], [0, 160], [170, 1], [0, 0]]), retain=500)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"counts": np.ones((2, 2, 2))}, ValueError, "two-dimensional"),
        ({"counts": np.ones((2, 2), dtype=bool)}, TypeError, "numeric"),
        ({"counts": np.array([[1, -1]])}, ValueError, "non-negative integers"),
        ({"counts": np.array([[1.5, 1]])}, ValueError, "non-negative integers"),
        ({"counts": np.array([[np.nan, 1]])}, ValueError, "non-negative integers"),
        ({"lower": 0}, ValueError, "lower must be at least 1"),
        ({"lower": 1.5}, TypeError, "lower must be an integer"),
        ({"retain": 100}, ValueError, "retain must be at least 101"),
        ({"retain": 200.0}, TypeError, "retain must be an integer"),
        ({"n_iter": 0}, ValueError, "n_iter must be at least 1"),
        ({"alpha": 0.0}, ValueError, "alpha must be positive"),
        ({"alpha": math.inf}, ValueError, "alpha must be finite"),
        ({"alpha": True}, TypeError, "alpha must be a real number"),
        ({"random_seed": -1}, ValueError, "random_seed must be at least 0"),
        ({"nthreads": 0}, ValueError, "nthreads must be at least 1"),
        ({"barcodes": ["a"]}, ValueError, "one identifier per row"),
    ],
)
def test_empty_drops_rejects_invalid_arguments(arguments, error, message) -> None:
    keywords = {"counts": np.array([[1, 2], [3, 4]]), **arguments}
    with pytest.raises(error, match=message):
        empty_drops(**keywords)


def test_is_cell_checks_its_threshold(droplets) -> None:
    counts, _, _ = droplets
    result = empty_drops(counts, n_iter=50, retain=300, nthreads=1)

    np.testing.assert_array_equal(result.is_cell(1.0), np.isfinite(result.fdr))
    for value, error, message in (
        (0.0, ValueError, "fdr must be greater than 0 and at most 1"),
        (1.5, ValueError, "fdr must be greater than 0 and at most 1"),
        ("0.1", TypeError, "fdr must be a real number"),
    ):
        with pytest.raises(error, match=message):
            result.is_cell(value)


def test_empty_drops_retains_every_barcode_above_lower_plus_one(droplets) -> None:
    counts, _, _ = droplets
    result = empty_drops(counts, retain=101, n_iter=10, nthreads=1)
    tested = counts.sum(axis=1) > 100

    assert result.retain == 101
    # Every tested barcode is retained, and the others are not tested.
    expected = np.where(tested, 0.0, np.nan)
    np.testing.assert_array_equal(result.p_value, expected)
    np.testing.assert_array_equal(result.fdr, expected)
    assert not result.limited.any()
    np.testing.assert_array_equal(result.is_cell(), tested)
    np.testing.assert_array_equal(np.isfinite(result.log_probability), tested)


def test_run_empty_drops_refuses_rows_that_change_between_reads() -> None:
    counts, _, _ = _droplets(seed=5, n_empty=500, n_cells=8, n_small=0)
    totals = counts.sum(axis=1)
    reads = []

    def read_rows(positions: np.ndarray):
        reads.append(len(positions))
        rows = counts[positions]
        if len(reads) == 2:
            rows = rows.copy()
            rows[0, 0] += 1
        yield rows

    settings = cc.validated_settings(
        lower=100, retain=10_000, n_iter=10, alpha=None, random_seed=0
    )
    with pytest.raises(RuntimeError, match="changed while read"):
        cc.run_empty_drops(totals, read_rows, counts.shape[1], settings, nthreads=1)
    assert reads == [int((totals > 0).sum()), int((totals > 100).sum())]


# Kernels run as Python, for coverage, against their compiled versions --------


def _with_stored_zeros(counts: np.ndarray) -> sparse.csr_matrix:
    """Return CSR counts that also store a zero entry in every row."""
    indptr, indices, data = [0], [], []
    for row in counts:
        features = np.r_[np.flatnonzero(row), np.flatnonzero(row == 0)[:1]]
        features.sort()
        indices.extend(features)
        data.extend(row[features])
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((data, indices, indptr), shape=counts.shape)
    assert (matrix.data == 0).any()
    return matrix


def test_ambient_kernels_match_their_python_versions() -> None:
    # Barcodes of 3, 8, 0, and 4 counts; with lower=5 the second is tested.
    counts = np.array([[1, 0, 2, 0], [0, 7, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    totals = counts.sum(axis=1)
    csr = _with_stored_zeros(counts)
    kernels = (
        (cc._accumulate_ambient, (counts,)),
        (cc._accumulate_ambient.py_func, (counts,)),
        (cc._accumulate_ambient_sparse, (csr.indptr, csr.indices, csr.data)),
        (cc._accumulate_ambient_sparse.py_func, (csr.indptr, csr.indices, csr.data)),
    )
    for kernel, arguments in kernels:
        sums = (
            np.zeros(4, np.int64),
            np.zeros(4, np.bool_),
            np.zeros((4, 6), np.int64),
        )
        assert kernel(*arguments, totals, 5, *sums) == -1
        ambient, expressed, histogram = sums
        np.testing.assert_array_equal(ambient, [2, 1, 3, 1])
        np.testing.assert_array_equal(expressed, [True] * 4)
        expected = np.zeros((4, 6), np.int64)
        expected[[0, 2, 0, 1, 2, 3], [1, 2, 1, 1, 1, 1]] = 1
        expected[0, 1] = 2
        np.testing.assert_array_equal(histogram, expected)
        fresh = (
            np.zeros(4, np.int64),
            np.zeros(4, np.bool_),
            np.zeros((4, 6), np.int64),
        )
        assert kernel(*arguments, totals + np.array([0, 0, 0, 1]), 5, *fresh) == 3

    sums = (np.zeros(2, np.int64), np.zeros(2, np.bool_), np.zeros((2, 4), np.int64))
    cc._add_ambient_count.py_func(1, np.int64(3), True, 3, *sums)
    cc._add_ambient_count.py_func(0, np.int64(5), True, 3, *sums)
    cc._add_ambient_count.py_func(0, np.int64(2), False, 3, *sums)
    np.testing.assert_array_equal(sums[0], [0, 3])
    np.testing.assert_array_equal(sums[1], [True, True])
    assert sums[2].sum() == 1 and sums[2][1, 3] == 1


def test_likelihood_kernels_match_their_python_versions() -> None:
    proportions = np.array([0.5, 0.3, 0.2, 0.0])
    counts = np.array([[3, 1, 0, 0], [0, 0, 4, 0], [1, 0, 0, 1]])
    csr = _with_stored_zeros(counts)
    for alpha in (2.5, math.inf):
        sampler = cc._sampler(proportions, alpha)
        results = []
        for kernel, arguments in (
            (cc._barcode_terms, (counts,)),
            (cc._barcode_terms.py_func, (counts,)),
            (cc._barcode_terms_sparse, (csr.indptr, csr.indices, csr.data)),
            (cc._barcode_terms_sparse.py_func, (csr.indptr, csr.indices, csr.data)),
        ):
            parts = np.empty(3)
            totals = np.empty(3, np.int64)
            kernel(
                *arguments, sampler.log_proportion, sampler.inverse_scale, parts, totals
            )
            results.append((parts, totals))
        # Python's math.lgamma and the compiled one may differ in the last bit.
        for parts, totals in results[1:]:
            np.testing.assert_allclose(parts, results[0][0], rtol=1e-14)
            np.testing.assert_array_equal(totals, [4, 4, 2])
        assert results[0][0][2] == -np.inf
        for feature, count in ((0, 3), (2, 4), (1, 1)):
            term = cc._count_term.py_func(
                feature, np.int64(count), sampler.log_proportion, sampler.inverse_scale
            )
            assert term == pytest.approx(
                cc._count_term(
                    feature,
                    np.int64(count),
                    sampler.log_proportion,
                    sampler.inverse_scale,
                ),
                rel=1e-14,
            )


def test_random_number_kernels_match_their_python_versions() -> None:
    with np.errstate(over="ignore"):
        for value in (0, 1, 2**63 + 5, 2**64 - 1):
            assert cc._mix64.py_func(np.uint64(value)) == cc._mix64(np.uint64(value))
        for seed, iteration in ((0, 0), (7, 3), (2**64 - 1, 10_000)):
            key = cc._iteration_key.py_func(np.uint64(seed), iteration)
            assert key == cc._iteration_key(np.uint64(seed), iteration)
            # Compiled code keeps keys unsigned; Python callers must too.
            key = np.uint64(key)
            draws = [cc._uniform.py_func(key, counter) for counter in range(50)]
            assert draws == [cc._uniform(key, counter) for counter in range(50)]
            assert all(0.0 <= draw < 1.0 for draw in draws)
    keys = {
        int(cc._iteration_key(np.uint64(1), iteration)) for iteration in range(1000)
    }
    assert len(keys) == 1000


def test_simulation_kernels_match_their_python_versions() -> None:
    proportions = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
    for alpha in (1.5, math.inf):
        sampler = cc._sampler(proportions, alpha)
        with np.errstate(over="ignore"):
            for iteration in range(4):
                features = cc._simulated_features(
                    iteration,
                    np.uint64(3),
                    9,
                    sampler.alpha,
                    sampler.multinomial,
                    sampler.cumulative,
                    sampler.guide,
                )
                python = cc._simulated_features.py_func(
                    iteration,
                    np.uint64(3),
                    9,
                    sampler.alpha,
                    sampler.multinomial,
                    sampler.cumulative,
                    sampler.guide,
                )
                np.testing.assert_array_equal(features, python)
                drawn = np.zeros(9, np.int64)
                key = np.uint64(cc._iteration_key(np.uint64(3), iteration))
                for count in range(9):
                    drawn[count] = cc._next_feature.py_func(
                        key,
                        2 * count,
                        count,
                        sampler.alpha,
                        sampler.multinomial,
                        sampler.cumulative,
                        sampler.guide,
                        drawn,
                    )
                np.testing.assert_array_equal(drawn, features)
            group_totals = np.array([2, 5, 9])
            group_starts = np.array([0, 2, 3, 5])
            thresholds = np.array([-4.0, -1.0, -6.0, -12.0, -8.0])
            arguments = (
                group_totals,
                group_starts,
                thresholds,
                sampler.log_proportion,
                sampler.inverse_scale,
                sampler.alpha,
                sampler.multinomial,
                sampler.cumulative,
                sampler.guide,
                np.r_[0.0, np.log(np.arange(1, 10))],
            )
            hits = []
            for kernel in (cc._simulate, cc._simulate.py_func):
                scratch = (
                    np.zeros((2, 8), np.int64),
                    np.zeros((2, 9), np.int64),
                    np.zeros((2, 5), np.int64),
                )
                kernel(7, np.uint64(4), *arguments, *scratch)
                hits.append(scratch[0].sum(axis=0))
                assert not scratch[2].any()
            single = (
                np.zeros(8, np.int64),
                np.zeros(9, np.int64),
                np.zeros(5, np.int64),
            )
            for iteration in range(7):
                cc._simulate_iteration.py_func(
                    iteration, np.uint64(4), *arguments, *single
                )
            np.testing.assert_array_equal(hits[0], hits[1])
            np.testing.assert_array_equal(hits[0], single[0])
            assert hits[0].sum() == 7 * 3
