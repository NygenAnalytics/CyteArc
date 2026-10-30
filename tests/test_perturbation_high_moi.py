import warnings

import numba
import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from cytearc.perturbation import associations as associations_module
from cytearc.perturbation import glm, high_moi, random_sets
from cytearc.perturbation.associations import (
    TreatmentGroups,
    association_covariates,
    fit_null_models,
    fit_skew_normals,
    pair_tests,
    skew_normal_p_values,
)
from cytearc.perturbation.high_moi import (
    ComplementGroups,
    complement_tests,
    fit_detection_models,
    fit_genes,
    nonzero_group_counts,
)
from cytearc.perturbation.random_sets import N_BUCKETS, resampling_keys
from tests.perturbation_helpers import dense_pair, size_root, use_python_kernels


def _python_kernels(monkeypatch: pytest.MonkeyPatch, python_kernels: bool) -> None:
    """Run the kernels uncompiled, whose 64-bit products wrap as compiled ones do."""
    if python_kernels:
        use_python_kernels(monkeypatch, high_moi, glm, random_sets, associations_module)
        np.seterr(over="ignore")


@pytest.fixture(autouse=True)
def _restore_floating_point_errors():
    previous = np.geterr()
    yield
    np.seterr(**previous)


def _groups(members: list[np.ndarray]) -> ComplementGroups:
    return ComplementGroups(
        members=np.concatenate(members).astype(np.int64),
        offsets=np.r_[0, np.cumsum([len(m) for m in members])].astype(np.int64),
        # Codes other than the groups' positions, as a screen's targets give.
        codes=3 * np.arange(len(members)) + 1,
    )


def _screen(
    n_cells: int = 300,
    n_genes: int = 4,
    sizes: tuple[int, ...] = (40, 30, 25),
    *,
    seed: int = 0,
    effect: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, ComplementGroups, list[np.ndarray]]:
    """Counts, covariates and groups of a small high-MOI screen.

    Each group's cells are drawn at random, so groups share cells. The cells
    come from two samples, and the first group lowers gene 0 by ``effect``.
    """
    rng = np.random.default_rng(seed)
    library = rng.lognormal(np.log(3000), 0.4, n_cells)
    detected = library**0.8 * rng.lognormal(0, 0.05, n_cells)
    design = association_covariates(library, detected, np.arange(n_cells) % 2)
    members = [np.sort(rng.choice(n_cells, size, replace=False)) for size in sizes]
    base = rng.normal(-6.5, 0.8, n_genes)
    theta = rng.uniform(1.0, 6.0, n_genes)
    mu = np.exp(base[:, None] + np.log(library)[None, :])
    mu[0, members[0]] *= effect
    counts = rng.negative_binomial(
        theta[:, None], theta[:, None] / (theta[:, None] + mu)
    ).astype(np.float64)
    return counts, design, _groups(members), members


def _same(first, second) -> None:
    """Two results of the tests are identical, bit for bit."""
    for name in first.__slots__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def _dense_null(
    fits: high_moi.GeneFits,
    design: np.ndarray,
    members: np.ndarray,
    key: np.uint64,
    column: int,
    n_draws: int,
    model: np.ndarray | None,
) -> np.ndarray:
    """Each random set's statistic of one group and gene, by the score formula.

    The sets come from the group's key, as the tests draw them; the sums, the
    information and the solve are NumPy's.
    """
    n = design.shape[0]
    out = np.empty(n, dtype=np.int64)
    if model is not None:
        probabilities = np.empty(n)
        glm.logistic_probabilities(design, model, np.empty(n), probabilities)
        order = np.empty(n, dtype=np.int64)
        offsets = np.empty(N_BUCKETS + 1, dtype=np.int64)
        bounds = np.empty(N_BUCKETS)
        random_sets.probability_buckets(probabilities, order, offsets, bounds)
    weights = fits.weights[:, column]
    information = design.T @ (weights[:, None] * design)
    statistics = np.empty(n_draws)
    for d in range(n_draws):
        set_key = np.uint64(random_sets.set_key(key, d))
        if model is not None:
            count = random_sets.draw_bernoulli_set(
                set_key, probabilities, order, offsets, bounds, out
            )
        else:
            count = len(members)
            random_sets.draw_uniform_set(set_key, count, n, np.zeros(n, bool), out)
        cells = out[:count]
        weighted = (weights[cells, None] * design[cells]).sum(axis=0)
        variance = weights[cells].sum() - weighted @ np.linalg.solve(
            information, weighted
        )
        statistics[d] = (
            fits.scores[cells, column].sum() / np.sqrt(variance)
            if variance > 1e-12
            else np.nan
        )
    return statistics


def _sided(statistics: np.ndarray, z: float, side: str) -> float:
    n_draws = len(statistics)
    left = (1 + (statistics <= z).sum()) / (n_draws + 1)
    right = (1 + (statistics >= z).sum()) / (n_draws + 1)
    return {"left": left, "right": right, "both": min(1.0, 2 * min(left, right))}[side]


# Checks -----------------------------------------------------------------------

_MEMBERS = np.array([0, 3, 5, 9, 12, 1, 2, 5, 20, 30, 39])
_OFFSETS = np.array([0, 5, 11])


@pytest.mark.parametrize(
    ("members", "offsets", "codes", "message"),
    [
        (_MEMBERS, [1, 5, 11], [0, 1], "start at 0 and end"),
        (_MEMBERS, [0, 5, 10], [0, 1], "start at 0 and end"),
        (_MEMBERS.reshape(1, -1), _OFFSETS, [0, 1], "start at 0 and end"),
        (_MEMBERS, [0, 0, 11], [0, 1], "at least one of the 40 cells"),
        (np.arange(40), [0, 40], [0], "must leave at least one out"),
        (_MEMBERS, _OFFSETS, [0, -1], "non-negative code for each group"),
        (_MEMBERS, _OFFSETS, [0], "non-negative code for each group"),
        (_MEMBERS, _OFFSETS, [2, 2], "must be distinct"),
        (np.r_[_MEMBERS[:-1], 40], _OFFSETS, [0, 1], "positions among the 40"),
        (np.r_[0, 3, 5, 12, 9, _MEMBERS[5:]], _OFFSETS, [0, 1], "distinct and"),
        (np.r_[0, 3, 3, 9, 12, _MEMBERS[5:]], _OFFSETS, [0, 1], "distinct and"),
    ],
)
def test_groups_must_be_distinct_ascending_cells_that_leave_some_out(
    members, offsets, codes, message
) -> None:
    groups = ComplementGroups(
        np.asarray(members), np.asarray(offsets), np.asarray(codes)
    )

    with pytest.raises(ValueError, match=message):
        nonzero_group_counts(np.ones((2, 40)), groups)


def test_covariates_need_more_cells_and_the_intercept_first() -> None:
    counts, design, groups, _ = _screen(n_cells=40, n_genes=2, sizes=(5, 6))

    with pytest.raises(ValueError, match="cells x covariates"):
        fit_detection_models(design[:, 1], groups)
    with pytest.raises(ValueError, match="must outnumber the 4 covariates"):
        fit_genes(counts[:, :4], design[:4])
    with pytest.raises(ValueError, match="intercept"):
        fit_genes(counts, design[:, ::-1])
    with pytest.raises(ValueError, match="genes x cells"):
        fit_genes(counts[:, :-1], design)
    with pytest.raises(ValueError, match="genes x cells"):
        complement_tests(counts[0], design, groups, [0], [0], resampling="permutation")
    with pytest.raises(ValueError, match="features x cells"):
        nonzero_group_counts(counts[0], groups)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_covariates_must_be_finite(value) -> None:
    counts, design, groups, _ = _screen(n_cells=40, n_genes=2, sizes=(5, 6))
    design[[3, 7], 1] = value
    design[7, 2] = value
    message = "covariates must be finite, but those of 2 of the 40 cells are not"

    with pytest.raises(ValueError, match=message):
        fit_genes(counts, design)
    with pytest.raises(ValueError, match=message):
        fit_detection_models(design, groups)
    with pytest.raises(ValueError, match=message):
        complement_tests(counts, design, groups, [0], [0], resampling="permutation")
    with pytest.raises(ValueError, match=message):
        complement_tests(
            counts, design, groups, [0], [0], detection_models=np.zeros((2, 4))
        )


def test_detection_models_must_expect_half_a_cell_in_a_set_and_half_out() -> None:
    counts, design, groups, members = _screen(n_cells=40, n_genes=2, sizes=(5, 6))
    fitted = fit_detection_models(design, groups)
    probabilities = np.empty(40)
    # A fitted model expects its group's own cells in a set, far from the
    # limits.
    for group, cells in enumerate(members):
        glm.logistic_probabilities(design, fitted[group], np.empty(40), probabilities)
        assert probabilities.sum() == pytest.approx(len(cells), abs=1e-6)

    for intercept, expected in [(-50.0, "7.71e-21"), (50.0, "40")]:
        models = fitted.copy()
        models[1] = [intercept, 0.0, 0.0, 0.0]
        with pytest.raises(
            ValueError, match=f"group 1 expects {expected} of the 40 cells in a set"
        ):
            complement_tests(
                counts, design, groups, [0, 1], [0, 1], detection_models=models
            )
        # A group without pairs draws no sets, so its model is not checked.
        result = complement_tests(
            counts,
            design,
            groups,
            [0],
            [1],
            detection_models=models,
            n_resamples=(9, 19, 29),
        )
        assert np.isfinite(result.p_value).all()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"pair_groups": [0, 1]}, "one entry per pair"),
        ({"pair_groups": [2]}, "pair_groups must be groups from 0 to 1"),
        ({"pair_groups": [-1]}, "pair_groups must be groups from 0 to 1"),
        ({"pair_rows": [2]}, "pair_rows must be rows of values, from 0 to 1"),
        ({"resampling": "crt"}, "resampling must be 'conditional' or 'permutation'"),
        ({"resampling": "conditional"}, "needs each group's detection model"),
        ({"detection_models": np.zeros((2, 4))}, "go with resampling='conditional'"),
        (
            {"resampling": "conditional", "detection_models": np.zeros((2, 3))},
            "finite coefficients, 2 groups x 4 covariates",
        ),
        (
            {"resampling": "conditional", "detection_models": np.full((2, 4), np.nan)},
            "finite coefficients",
        ),
    ],
)
def test_complement_tests_refuse_pairs_and_resampling_they_cannot_use(
    arguments, message
) -> None:
    counts, design, groups, _ = _screen(n_cells=40, n_genes=2, sizes=(5, 6))
    settings: dict[str, object] = {
        "pair_groups": [0],
        "pair_rows": [1],
        "resampling": "permutation",
    }
    settings.update(arguments)

    with pytest.raises(ValueError, match=message):
        complement_tests(counts, design, groups, **settings)


# QC counts, detection models and gene fits --------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_nonzero_counts_of_overlapping_groups_and_their_complements(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(3)
    values = rng.poisson(0.6, size=(6, 50)).astype(np.float32)
    members = [np.sort(rng.choice(50, k, replace=False)) for k in (10, 25, 49, 1)]

    in_groups, totals = nonzero_group_counts(values, _groups(members))

    np.testing.assert_array_equal(totals, (values > 0).sum(axis=1))
    for group, cells in enumerate(members):
        outside = np.setdiff1d(np.arange(50), cells)
        np.testing.assert_array_equal(in_groups[group], (values[:, cells] > 0).sum(1))
        np.testing.assert_array_equal(
            totals - in_groups[group], (values[:, outside] > 0).sum(1)
        )


@pytest.mark.parametrize("python_kernels", [False, True])
def test_detection_models_are_logistic_fits_of_each_groups_cells(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 400 if not python_kernels else 100
    _, design, groups, members = _screen(n_cells=n_cells, sizes=(60, 45, 30))

    models = fit_detection_models(design, groups)

    for group, cells in enumerate(members):
        y = np.zeros(n_cells)
        y[cells] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = sm.GLM(y, design, family=sm.families.Binomial()).fit(tol=1e-14)
        np.testing.assert_allclose(models[group], expected.params, rtol=0, atol=1e-8)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_gene_fits_on_all_cells_are_the_null_fits_of_the_low_moi_test(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 300 if not python_kernels else 80
    counts, design, _, _ = _screen(n_cells=n_cells, n_genes=3)
    counts[2] = 0.0  # a gene without counts
    p = design.shape[1]

    fits = fit_genes(counts.astype(np.float32), design)

    # The same Newton steps from the log of the mean as the low-MOI null fits
    # on control cells that are all cells.
    np.testing.assert_array_equal(
        fits.coefficients[:2], fit_null_models(counts, design, n_cells)[:2]
    )
    for gene in (0, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poisson = sm.GLM(counts[gene], design, family=sm.families.Poisson()).fit(
                tol=1e-14
            )
        mu = poisson.mu
        dispersion = size_root(counts[gene], mu, n_cells - p)
        np.testing.assert_allclose(fits.coefficients[gene], poisson.params, atol=1e-6)
        np.testing.assert_allclose(fits.dispersion[gene], dispersion, rtol=1e-6)
        scale = 1 + dispersion * mu
        np.testing.assert_allclose(fits.weights[:, gene], mu / scale, rtol=1e-6)
        np.testing.assert_allclose(
            fits.scores[:, gene], (counts[gene] - mu) / scale, rtol=1e-6, atol=1e-9
        )
        factor = fits.factors[gene]
        information = design.T @ (fits.weights[:, gene, None] * design)
        np.testing.assert_allclose(factor @ factor.T, information, rtol=1e-10)
        assert (np.triu(factor, 1) == 0).all()
    assert np.isnan(fits.coefficients[2]).all() and np.isnan(fits.dispersion[2])
    assert np.isnan(fits.factors[2]).all()
    assert not fits.scores[:, 2].any() and not fits.weights[:, 2].any()


# Statistics ---------------------------------------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_statistics_match_a_dense_evaluation_of_the_score_formula(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 300 if not python_kernels else 80
    sizes = (40, 30, 25) if not python_kernels else (12, 9, 7)
    counts, design, groups, members = _screen(n_cells=n_cells, n_genes=4, sizes=sizes)
    counts[2, members[1]] = 0.0  # group 1 holds no counts of gene 2
    counts[3] = 0.0  # gene 3 has no counts at all
    pair_groups = np.array([0, 1, 2, 1, 0, 2, 1, 2])
    pair_rows = np.array([0, 1, 2, 2, 1, 0, 1, 3])

    result = complement_tests(
        counts,
        design,
        groups,
        pair_groups,
        pair_rows,
        resampling="permutation",
        n_resamples=(9, 19, 29),
    )

    for pair, (group, row) in enumerate(zip(pair_groups[:-1], pair_rows[:-1])):
        expected = dense_pair(counts[row], design, members[group])
        found = [result.z[pair], result.log2_fold_change[pair]]
        np.testing.assert_allclose(
            [*found, result.se_log2_fold_change[pair]], expected, rtol=1e-6
        )
    assert result.log2_fold_change[3] == -np.inf
    # A repeated pair is the same test, and a gene without counts has none.
    for name in result.__slots__:
        values = getattr(result, name)
        np.testing.assert_array_equal(values[1], values[6])
    assert np.isnan([result.z[7], result.p_value[7], result.log2_fold_change[7]]).all()
    assert result.stage[7] == 1


def test_permutation_tests_are_the_low_moi_test_when_every_cell_is_a_control() -> None:
    counts, design, groups, members = _screen(
        n_cells=500, n_genes=6, sizes=(80, 60, 50), seed=5, effect=0.3
    )
    pair_groups = np.repeat(np.arange(3), 6)
    pair_rows = np.tile(np.arange(6), 3)
    # Each group is compared with every other cell when every cell is a
    # control cell and the groups lie among them.
    controls = TreatmentGroups(
        members=groups.members,
        offsets=groups.offsets,
        among_controls=np.ones(3, dtype=bool),
        codes=groups.codes,
        n_controls=500,
    )

    low = pair_tests(counts, design, controls, pair_groups, pair_rows, seed=2)
    high = complement_tests(
        counts, design, groups, pair_groups, pair_rows, resampling="permutation", seed=2
    )

    # The low-MOI pair fit starts again from the null fit, so they differ in
    # the last bits; the random sets are the same uniform sets.
    for name in ("z", "log2_fold_change", "se_log2_fold_change"):
        np.testing.assert_allclose(getattr(high, name), getattr(low, name), rtol=1e-9)
    np.testing.assert_array_equal(high.stage, low.stage)
    assert (high.stage == 2).any()
    counted = high.stage != 2
    np.testing.assert_array_equal(high.p_value[counted], low.p_value[counted])
    np.testing.assert_allclose(high.p_value, low.p_value, rtol=1e-9)
    for name in ("skew_location", "skew_scale", "skew_shape"):
        np.testing.assert_allclose(getattr(high, name), getattr(low, name), rtol=1e-6)


# Random sets and rounds ------------------------------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
@pytest.mark.parametrize("resampling", ["conditional", "permutation"])
def test_rounds_count_and_fit_the_statistics_of_their_random_sets(
    monkeypatch, python_kernels, resampling
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 300 if not python_kernels else 60
    sizes = (40, 30) if not python_kernels else (12, 9)
    draws = (49, 199, 299) if not python_kernels else (9, 19, 29)
    counts, design, groups, members = _screen(
        n_cells=n_cells, n_genes=2, sizes=sizes, seed=7, effect=0.5
    )
    models = fit_detection_models(design, groups)
    conditional = resampling == "conditional"
    pair_groups = np.array([0, 1, 0, 1])
    pair_rows = np.array([0, 0, 1, 1])
    fits = fit_genes(counts, design)
    settings = {
        "resampling": resampling,
        "detection_models": models if conditional else None,
        "seed": 4,
        "n_resamples": draws,
        "side": "left",
    }

    def dense(pair: int, stage: int, n_draws: int) -> np.ndarray:
        group = pair_groups[pair]
        key = resampling_keys(4, groups.codes, stage)[group]
        model = models[group] if conditional else None
        return _dense_null(
            fits, design, members[group], key, pair_rows[pair], n_draws, model
        )

    first = complement_tests(
        counts, design, groups, pair_groups, pair_rows, stage_threshold=0.0, **settings
    )
    empirical = complement_tests(
        counts,
        design,
        groups,
        pair_groups,
        pair_rows,
        stage_threshold=1.0,
        approximation="none",
        **settings,
    )
    # Every fit passes its check, or every fit fails it.
    monkeypatch.setattr(associations_module, "_TAIL_RATIO_FLOOR", -np.inf)
    fitted = complement_tests(
        counts, design, groups, pair_groups, pair_rows, stage_threshold=1.0, **settings
    )
    monkeypatch.setattr(associations_module, "_TAIL_RATIO_FLOOR", np.inf)
    third = complement_tests(
        counts, design, groups, pair_groups, pair_rows, stage_threshold=1.0, **settings
    )

    assert (first.stage == 1).all() and (empirical.stage == 2).all()
    assert (third.stage == 3).all() and np.isnan(third.skew_shape).all()
    for pair in range(4):
        z = first.z[pair]
        assert first.p_value[pair] == _sided(dense(pair, 1, draws[0]), z, "left")
        second = dense(pair, 2, draws[1])
        assert empirical.p_value[pair] == _sided(second, z, "left")
        assert third.p_value[pair] == _sided(dense(pair, 3, draws[2]), z, "left")
        location, scale, shape = fit_skew_normals(second[None, :])
        np.testing.assert_allclose(
            [
                fitted.skew_location[pair],
                fitted.skew_scale[pair],
                fitted.skew_shape[pair],
            ],
            [location[0], scale[0], shape[0]],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            fitted.p_value[pair],
            skew_normal_p_values(np.array([z]), location, scale, shape, "left"),
            rtol=1e-9,
        )
    assert (fitted.stage == 2).all()


def _second_round_statistics(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Record the statistics of every second round of :func:`complement_tests`."""
    recorded: list[np.ndarray] = []
    staged = high_moi.staged_p_values

    def recording(z, counts, second_statistics, third_counts, **settings):
        def second(chosen: np.ndarray, n_draws: int) -> np.ndarray:
            statistics = second_statistics(chosen, n_draws)
            recorded.append(statistics.copy())
            return statistics

        return staged(z, counts, second, third_counts, **settings)

    monkeypatch.setattr(high_moi, "staged_p_values", recording)
    return recorded


@pytest.mark.parametrize("python_kernels", [False, True])
def test_conditional_sets_of_a_small_group_all_hold_cells(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 200 if not python_kernels else 60
    rng = np.random.default_rng(9)
    depth = rng.normal(size=n_cells)
    design = np.column_stack([np.ones(n_cells), depth])
    counts = rng.poisson(np.exp(1.0 + 0.3 * depth), size=(1, n_cells)).astype(float)
    # Two cells of middling depth, so the logistic fit has a finite maximum.
    middle = np.argsort(np.abs(depth))[:2]
    groups = _groups([np.sort(middle)])
    models = fit_detection_models(design, groups)
    probabilities = np.empty(n_cells)
    glm.logistic_probabilities(design, models[0], np.empty(n_cells), probabilities)
    n_draws = 199 if not python_kernels else 39

    result = complement_tests(
        counts,
        design,
        groups,
        [0],
        [0],
        detection_models=models,
        seed=1,
        n_resamples=(n_draws, 99, 99),
        stage_threshold=0.0,
    )

    # About one draw in seven holds no cell; it is drawn again, so every set
    # has a statistic and the p-value counts every set.
    assert np.prod(1 - probabilities) > 0.1
    key = resampling_keys(1, groups.codes, 1)[0]
    statistics = _dense_null(
        fit_genes(counts, design), design, middle, key, 0, n_draws, models[0]
    )
    assert np.isfinite(statistics).all()
    assert result.p_value[0] == _sided(statistics, result.z[0], "both")


def test_small_groups_reach_p_values_far_below_the_third_rounds_floor() -> None:
    rng = np.random.default_rng(3)
    n_cells = 2000
    library = rng.lognormal(np.log(3000), 0.4, n_cells)
    detected = library**0.8 * rng.lognormal(0, 0.05, n_cells)
    design = association_covariates(library, detected)
    members = [np.sort(rng.choice(n_cells, size, replace=False)) for size in (7, 8)]
    groups = _groups(members)
    # Each group lowers two genes 50-fold.
    mu = np.exp(np.log(library)[None, :] - 5.0 + rng.normal(0, 0.3, (4, 1)))
    mu[:2, members[0]] *= 0.02
    mu[2:, members[1]] *= 0.02
    counts = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(np.float64)

    result = complement_tests(
        counts,
        design,
        groups,
        np.array([0, 0, 1, 1]),
        np.arange(4),
        detection_models=fit_detection_models(design, groups),
        seed=1,
        side="left",
    )

    # A draw of a group of 7 cells holds no cell about once in 1,100, so the
    # second round's 4,999 sets would hold several empty ones without a
    # statistic, fail the skew-normal fit, and leave the third round's
    # 24,999 sets to give p-values of 1 / 25,000 at best.
    assert (result.stage == 2).all()
    assert (result.p_value < 1e-7).all()


@pytest.mark.parametrize("resampling", ["conditional", "permutation"])
def test_a_group_of_nearly_every_cell_has_null_statistics_of_the_score_formula(
    monkeypatch, resampling
) -> None:
    n_cells = 300
    counts, design, _, _ = _screen(n_cells=n_cells, n_genes=3, sizes=(5,), seed=13)
    members = np.sort(np.random.default_rng(13).choice(n_cells, n_cells - 3, False))
    groups = _groups([members])
    models = fit_detection_models(design, groups)
    conditional = resampling == "conditional"
    recorded = _second_round_statistics(monkeypatch)

    complement_tests(
        counts,
        design,
        groups,
        np.zeros(3, dtype=int),
        np.arange(3),
        resampling=resampling,
        detection_models=models if conditional else None,
        seed=5,
        n_resamples=(99, 999, 999),
        stage_threshold=1.0,
    )

    # About one conditional draw in twenty holds every cell, whose statistic
    # has a variance of 0 that rounding leaves as NaN or as a tiny number,
    # giving a statistic of 1e6 or more. Such draws are drawn again, and a
    # permutation set always leaves three cells out.
    probabilities = np.empty(n_cells)
    glm.logistic_probabilities(design, models[0], np.empty(n_cells), probabilities)
    assert np.prod(probabilities) > 0.02
    (statistics,) = recorded
    key = resampling_keys(5, groups.codes, 2)[0]
    fits = fit_genes(counts, design)
    for gene in range(3):
        expected = _dense_null(
            fits, design, members, key, gene, 999, models[0] if conditional else None
        )
        np.testing.assert_allclose(statistics[gene], expected, rtol=1e-6)
    assert np.isfinite(statistics).all() and np.abs(statistics).max() < 10


@pytest.mark.parametrize("python_kernels", [False, True])
def test_separated_cells_never_enter_a_conditional_set(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 300 if not python_kernels else 80
    counts, design, _, _ = _screen(n_cells=n_cells, n_genes=1)
    # Every cell of the group comes from the first sample, the even cells.
    groups = _groups([np.arange(0, n_cells // 2, 2)])
    models = fit_detection_models(design, groups)
    probabilities = np.empty(n_cells)
    glm.logistic_probabilities(design, models[0], np.empty(n_cells), probabilities)

    result = complement_tests(
        counts,
        design,
        groups,
        [0],
        [0],
        detection_models=models,
        n_resamples=(19, 39, 59),
    )

    assert probabilities[1::2].max() < 1e-6
    key = resampling_keys(0, groups.codes, 1)[0]
    fits = fit_genes(counts, design)
    out = np.empty(n_cells, dtype=np.int64)
    order = np.empty(n_cells, dtype=np.int64)
    offsets = np.empty(N_BUCKETS + 1, dtype=np.int64)
    bounds = np.empty(N_BUCKETS)
    random_sets.probability_buckets(probabilities, order, offsets, bounds)
    for d in range(19):
        set_key = np.uint64(random_sets.set_key(key, d))
        count = random_sets.draw_bernoulli_set(
            set_key, probabilities, order, offsets, bounds, out
        )
        assert (out[:count] % 2 == 0).all()
    assert np.isfinite(result.p_value[0]) and np.isfinite(fits.coefficients).all()


@pytest.mark.parametrize("python_kernels", [False, True])
@pytest.mark.parametrize("resampling", ["conditional", "permutation"])
def test_results_depend_neither_on_bands_nor_threads_nor_memory(
    monkeypatch, python_kernels, resampling
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    n_cells = 200 if not python_kernels else 50
    sizes = (30, 20, 25) if not python_kernels else (9, 6, 8)
    counts, design, groups, _ = _screen(
        n_cells=n_cells, n_genes=5, sizes=sizes, seed=11, effect=0.4
    )
    # Group 0 with every gene, group 1 with two genes far apart, and group 2
    # with two genes, one of them twice.
    pair_groups = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
    pair_rows = np.array([0, 1, 2, 3, 4, 0, 4, 1, 2, 2])
    models = fit_detection_models(design, groups)
    settings = {
        "resampling": resampling,
        "detection_models": models if resampling == "conditional" else None,
        "seed": 3,
        "n_resamples": (19, 39, 59) if not python_kernels else (9, 19, 29),
        "stage_threshold": 0.5,
    }

    def run(**extra) -> high_moi.PairResults:
        return complement_tests(
            counts, design, groups, pair_groups, pair_rows, **settings, **extra
        )

    whole = run()
    # Bands of genes 0 and 1, 2, and 3 and 4.
    banded = {}
    for band in ([0, 1], [2], [3, 4]):
        chosen = np.flatnonzero(np.isin(pair_rows, band))
        part = complement_tests(
            counts[band],
            design,
            groups,
            pair_groups[chosen],
            np.searchsorted(band, pair_rows[chosen]),
            **settings,
        )
        for name in whole.__slots__:
            banded.setdefault(name, np.empty_like(getattr(whole, name)))[chosen] = (
                getattr(part, name)
            )
    # Threads that take whole groups or split a group's sets; with room to copy
    # each group's genes together, or one gene at a time, so that dense genes
    # are read in place, and one pair's second-round statistics at a time.
    others = []
    for threads in (1, 64):
        monkeypatch.setattr(
            high_moi.numba, "get_num_threads", lambda threads=threads: threads
        )
        others.append(run())
        others.append(run(gather_bytes=16 * n_cells, statistics_bytes=1))

    assert (whole.stage >= 2).any()
    for other in others:
        _same(whole, other)
    for name in whole.__slots__:
        np.testing.assert_array_equal(banded[name], getattr(whole, name))


def test_results_do_not_depend_on_the_number_of_threads() -> None:
    counts, design, groups, _ = _screen(n_cells=400, n_genes=6, seed=12, effect=0.4)
    pair_groups = np.repeat(np.arange(3), 6)
    pair_rows = np.tile(np.arange(6), 3)
    models = fit_detection_models(design, groups)
    settings = {"detection_models": models, "n_resamples": (99, 499, 999)}
    previous = numba.get_num_threads()

    many = complement_tests(counts, design, groups, pair_groups, pair_rows, **settings)
    numba.set_num_threads(1)
    try:
        one = complement_tests(
            counts, design, groups, pair_groups, pair_rows, **settings
        )
        one_model = fit_detection_models(design, groups)
    finally:
        numba.set_num_threads(previous)

    _same(many, one)
    np.testing.assert_array_equal(models, one_model)


def test_units_copy_small_or_scattered_genes_and_read_dense_ones_in_place() -> None:
    # Pairs sorted by group and gene: a dense group, a scattered group, and a
    # group of one gene twice.
    columns = np.array([0, 1, 2, 3, 4, 0, 4, 9, 7, 7])
    bounds = np.array([0, 5, 8, 10])

    units = high_moi._plan_units(columns, bounds, 2)

    np.testing.assert_array_equal(units.group_offsets, [0, 1, 3, 4])
    np.testing.assert_array_equal(units.compact, [False, True, True, True])
    np.testing.assert_array_equal(units.first[:1], [0])
    np.testing.assert_array_equal(units.width, [5, 2, 1, 1])
    np.testing.assert_array_equal(units.columns, [0, 4, 9, 7])
    np.testing.assert_array_equal(units.column_offsets, [0, 0, 2, 3, 4])
    np.testing.assert_array_equal(units.pair_offsets, [0, 5, 7, 8, 10])
    np.testing.assert_array_equal(units.local, [0, 1, 2, 3, 4, 0, 1, 0, 0, 0])
    # With room for every gene, each group is copied whole.
    whole = high_moi._plan_units(columns, bounds, 8)
    assert whole.compact.all() and whole.width.tolist() == [5, 3, 1]


def test_no_pairs_make_no_tests() -> None:
    counts, design, groups, _ = _screen(n_cells=40, n_genes=2, sizes=(5, 6))
    empty = np.empty(0, dtype=np.int64)

    result = complement_tests(
        counts, design, groups, empty, empty, resampling="permutation"
    )

    assert all(len(getattr(result, name)) == 0 for name in result.__slots__)


# The whole test ------------------------------------------------------------------


def test_complement_tests_find_a_planted_effect_and_keep_null_pairs_uniform() -> None:
    counts, design, groups, _ = _screen(
        n_cells=1500, n_genes=30, sizes=(150, 120, 140), seed=8, effect=0.3
    )
    pair_groups = np.repeat(np.arange(3), 30)
    pair_rows = np.tile(np.arange(30), 3)
    models = fit_detection_models(design, groups)

    result = complement_tests(
        counts, design, groups, pair_groups, pair_rows, detection_models=models, seed=1
    )

    assert result.stage[0] == 2 and result.p_value[0] < 1e-8
    assert result.z[0] < -5 and result.log2_fold_change[0] < -1
    assert np.isfinite(result.skew_shape[0])
    null = np.delete(result.p_value, 0)
    assert stats.kstest(null, "uniform").pvalue > 0.01
    once = result.stage == 1
    assert np.isnan(result.skew_location[once]).all()
    assert (result.p_value[once] > 0.02).all()


def test_complement_tests_take_one_side_or_both() -> None:
    counts, design, groups, _ = _screen(
        n_cells=1000, n_genes=2, sizes=(150, 50), seed=2, effect=0.3
    )
    models = fit_detection_models(design, groups)

    left, right, both = (
        complement_tests(
            counts,
            design,
            groups,
            np.zeros(2, int),
            np.arange(2),
            detection_models=models,
            side=side,
            seed=2,
        )
        for side in ("left", "right", "both")
    )

    assert left.p_value[0] < 1e-8 and right.p_value[0] > 0.99
    assert both.p_value[0] == 2 * left.p_value[0]


def _calibration_replicate(
    rng: np.random.Generator, *, confounded: bool, misspecified: bool, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The p-values of both resampling ways on one simulated null screen.

    A confounder d, such as depth, drives the detection of the group's guides
    when ``confounded``, and the counts' log mean holds d^2, which the
    covariates miss, when ``misspecified``. The counts never depend on the
    group, so every p-value is null.
    """
    n_cells = 2000
    depth = rng.standard_normal(n_cells)
    design = np.column_stack([np.ones(n_cells), depth])
    slope = 1.5 if confounded else 0.0
    detection = 1 / (1 + np.exp(-(-3.0 + slope * depth)))
    groups = _groups([np.flatnonzero(rng.random(n_cells) < detection)])
    mean = np.exp(0.5 + 0.3 * depth + (0.4 * depth**2 if misspecified else 0.0))
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mean), size=(2, n_cells))
    pairs = (np.zeros(2, int), np.arange(2))
    settings = {"n_resamples": (199, 999, 999), "stage_threshold": 0.0, "seed": seed}
    conditional = complement_tests(
        counts,
        design,
        groups,
        *pairs,
        detection_models=fit_detection_models(design, groups),
        **settings,
    )
    permutation = complement_tests(
        counts, design, groups, *pairs, resampling="permutation", **settings
    )
    return conditional.p_value, permutation.p_value


@pytest.mark.parametrize(
    ("confounded", "misspecified", "permutations_hold"),
    [(False, True, True), (True, False, True), (True, True, False)],
    ids=["unconfounded", "confounded", "confounded-and-misspecified"],
)
def test_conditional_sets_stay_calibrated_where_permutations_fail(
    confounded, misspecified, permutations_hold
) -> None:
    rng = np.random.default_rng([20, confounded, misspecified])

    found = [
        _calibration_replicate(
            rng, confounded=confounded, misspecified=misspecified, seed=replicate
        )
        for replicate in range(60)
    ]

    conditional = np.concatenate([p_values for p_values, _ in found])
    permutation = np.concatenate([p_values for _, p_values in found])
    # Measured on six other draws of the 120 pairs: at most 0.067 for the
    # conditional sets in every case, and for permutations at most 0.083
    # where they hold and 0.57 to 0.69 where they fail.
    assert (conditional <= 0.05).mean() < 0.15
    if permutations_hold:
        assert (permutation <= 0.05).mean() < 0.15
    else:
        assert (permutation <= 0.05).mean() > 0.4
