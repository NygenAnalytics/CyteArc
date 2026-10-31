import math
import re
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import special, stats

from cytearc.datastore._operations import (
    perturbation_associations as perturbation_operations,
)
from cytearc.datastore._operations.perturbation_store import _label_rows
from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import PerturbationAssociations
from cytearc.perturbation import associations as associations_module
from cytearc.perturbation import glm, random_sets
from cytearc.perturbation import p_values as p_values_module
from cytearc.perturbation.associations import (
    ASSOCIATION_SETS,
    CALIBRATION_STATUSES,
    RECORD_BYTES,
    TreatmentGroups,
    association_covariates,
    association_pairs,
    calibration_grade,
    calibration_group_size,
    calibration_pairs,
    calibration_runs,
    calibration_status,
    fit_null_models,
    most_calibration_pairs,
    negative_control_groups,
    nonzero_counts,
    pair_table_rows,
    pair_tests,
    record_batch,
    resampled_statistics,
    resampling_counts,
)
from cytearc.perturbation.p_values import (
    adjusted_p_values,
    fit_skew_normals,
    resampling_p_values,
    skew_normal_p_values,
    skew_normal_tail_ratios,
)
from cytearc.perturbation.random_sets import resampling_keys
from cytearc.storage.artifacts import ArtifactRef
from cytearc.utils.logging import logger
from tests.perturbation_helpers import (
    check_budget,
    dense_pair,
    fixed_labels,
    named_budget,
    open_store,
    record_cell_reads,
    size_root,
    use_python_kernels,
    write_screen,
)


def _python_kernels(monkeypatch: pytest.MonkeyPatch, python_kernels: bool) -> None:
    """Run the kernels uncompiled, whose 64-bit products wrap as compiled ones do."""
    if python_kernels:
        use_python_kernels(monkeypatch, associations_module, glm, random_sets)
        np.seterr(over="ignore")


@pytest.fixture(autouse=True)
def _restore_floating_point_errors():
    previous = np.geterr()
    yield
    np.seterr(**previous)


def _groups(
    n_controls: int,
    members: list[np.ndarray],
    among: list[bool],
) -> TreatmentGroups:
    return TreatmentGroups(
        members=np.concatenate(members).astype(np.int64),
        offsets=np.concatenate([[0], np.cumsum([len(m) for m in members])]).astype(
            np.int64
        ),
        among_controls=np.asarray(among, dtype=bool),
        codes=np.arange(len(members)),
        n_controls=n_controls,
    )


def _screen(
    n_controls: int = 120,
    sizes: tuple[int, ...] = (40, 30),
    n_genes: int = 12,
    *,
    seed: int = 0,
    effect: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, TreatmentGroups]:
    """Counts, covariates and groups of a small screen, the controls first.

    The first group lowers gene 0 by ``effect``; every other pair is null.
    """
    rng = np.random.default_rng(seed)
    n_cells = n_controls + sum(sizes)
    library = rng.lognormal(np.log(3000), 0.4, n_cells)
    detected = library**0.8 * rng.lognormal(0, 0.05, n_cells)
    base = rng.normal(-6.5, 0.8, n_genes)
    theta = rng.uniform(1.0, 6.0, n_genes)
    mu = np.exp(base[:, None] + np.log(library)[None, :])
    mu[0, n_controls : n_controls + sizes[0]] *= effect
    counts = rng.negative_binomial(
        theta[:, None], theta[:, None] / (theta[:, None] + mu)
    ).astype(np.float64)
    bounds = n_controls + np.concatenate([[0], np.cumsum(sizes)])
    members = [np.arange(bounds[k], bounds[k + 1]) for k in range(len(sizes))]
    groups = _groups(n_controls, members, [False] * len(sizes))
    return counts, association_covariates(library, detected), groups


# Covariates and keys -----------------------------------------------------------


def test_covariates_hold_an_intercept_scaled_logs_samples_and_extra_columns() -> None:
    umis = np.array([100.0, 400.0, 900.0, 1600.0])
    detected = np.array([50.0, 100.0, 300.0, 400.0])
    extra = np.array([[1.0], [2.0], [4.0], [9.0]])
    kept = extra.copy()

    design = association_covariates(
        umis, detected, np.array([0, 2, 1, 2]), extra, names=["mito"]
    )

    np.testing.assert_array_equal(extra, kept)
    assert design.shape == (4, 6)
    assert design.flags.c_contiguous
    np.testing.assert_array_equal(design[:, 0], 1.0)
    for column, raw in ((1, np.log(detected)), (2, np.log(umis)), (5, extra[:, 0])):
        np.testing.assert_allclose(design[:, column], stats.zscore(raw))
    np.testing.assert_array_equal(design[:, 3], [0, 0, 1, 0])
    np.testing.assert_array_equal(design[:, 4], [0, 1, 0, 1])
    np.testing.assert_array_equal(
        association_covariates(umis, detected, None, extra)[:, 3], design[:, 5]
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((np.ones(3), np.ones(2)), "one value per cell"),
        ((np.array([1.0, 0.0]), np.ones(2)), "needs counts"),
        ((np.ones(2) * 5, np.array([2.0, 3.0]), np.array([0, -1])), "one code from 0"),
        ((np.ones(2) * 5, np.array([2.0, 3.0]), np.array([0])), "one code from 0"),
        ((np.ones(2) * 5, np.array([2.0, 3.0]), np.array([0, 2])), "Every sample"),
        (
            (np.array([4.0, 5.0]), np.array([2.0, 3.0]), None, [[1.0], [np.nan]]),
            "finite",
        ),
        ((np.array([4.0, 5.0]), np.array([3.0, 3.0])), "'log detected features'"),
        (
            (np.array([4.0, 5.0]), np.array([2.0, 3.0]), None, [[1.0], [1.0]]),
            "column 0",
        ),
        # One value repeated, whose standard deviation rounds above 0 for 10
        # cells.
        (
            (np.arange(10.0) + 10, np.arange(10.0) + 5, None, np.full((10, 1), 0.3)),
            "^The covariate 'column 0' is constant over the tested cells$",
        ),
    ],
)
def test_covariates_refuse_cells_and_columns_they_cannot_use(
    arguments, message
) -> None:
    with pytest.raises(ValueError, match=message):
        association_covariates(*arguments)


def test_covariates_add_log_guide_counts_after_the_expression_columns() -> None:
    umis = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    detected = np.array([50.0, 100.0, 300.0, 400.0, 700.0])
    guide_umis = np.array([3.0, 40.0, 8.0, 120.0, 15.0])
    guide_detected = np.array([1.0, 3.0, 2.0, 6.0, 2.0])
    samples = np.array([0, 1, 1, 0, 1])

    design = association_covariates(
        umis,
        detected,
        samples,
        guide_umis=guide_umis,
        guide_detected=guide_detected,
    )

    # sceptre's high-MOI formula: the expression terms, the guide terms, then
    # the samples.
    assert design.shape == (5, 6)
    np.testing.assert_array_equal(
        design[:, [0, 1, 2, 5]], association_covariates(umis, detected, samples)
    )
    np.testing.assert_allclose(design[:, 3], stats.zscore(np.log(guide_detected)))
    np.testing.assert_allclose(design[:, 4], stats.zscore(np.log(guide_umis)))


@pytest.mark.parametrize(
    ("guides", "message"),
    [
        ({"guide_umis": np.ones(3)}, "go together"),
        ({"guide_detected": np.ones(3)}, "go together"),
        (
            {"guide_umis": np.ones(2), "guide_detected": np.ones(2)},
            "one value per cell",
        ),
        (
            {"guide_umis": np.array([4.0, 0.0, 6.0]), "guide_detected": np.ones(3)},
            "1 of the 3 tested cells has no guide UMIs",
        ),
        (
            {
                "guide_umis": np.array([0.0, 0.0, 6.0]),
                "guide_detected": np.array([0.0, 0.0, 2.0]),
            },
            "2 of the 3 tested cells have no guide UMIs, and the test adjusts for "
            "the log of each cell's guide UMIs; select cells with guide UMIs",
        ),
    ],
)
def test_covariates_refuse_guide_counts_they_cannot_use(guides, message) -> None:
    with pytest.raises(ValueError, match=message):
        association_covariates(
            np.array([10.0, 20.0, 30.0]), np.array([5.0, 6.0, 9.0]), **guides
        )


def _settings(**changes: object) -> tuple:
    """Check association settings at the defaults of ``associations`` but ``changes``."""
    settings: dict[str, object] = {
        "positive_controls": "targets",
        "id_column": "target_gene_id",
        "integration": "union",
        "side": "both",
        "approximation": "skew_normal",
        "n_resamples": (499, 4999, 24999),
        "stage_threshold": 0.02,
        "min_nonzero": 7,
        "n_calibration_pairs": None,
        "alpha": 0.1,
        "seed": 0,
        "moi": "low",
        "resampling": None,
        "sample_column": None,
        "covariates": None,
    }
    settings.update(changes)
    return associations_module.association_settings(**settings)


def test_association_settings_resolve_the_resampling_of_each_moi() -> None:
    assert _settings()[-1] == "permutation"
    assert _settings(moi="high")[-1] == "conditional"
    assert _settings(moi="high", resampling="permutation")[-1] == "permutation"
    assert _settings(resampling="permutation")[-1] == "permutation"
    for changes, message in (
        ({"resampling": "conditional"}, "goes with moi='high'"),
        ({"moi": "medium"}, "moi must be 'low' or 'high', not 'medium'"),
        ({"moi": None}, "moi must be"),
        ({"moi": "high", "resampling": "crt"}, "not 'crt'"),
        ({"moi": "high", "resampling": 1}, "resampling must be"),
    ):
        with pytest.raises(ValueError, match=message):
            _settings(**changes)


@pytest.mark.parametrize(
    ("side", "short", "enough"), [("both", 98, 99), ("left", 48, 49), ("right", 48, 49)]
)
def test_settings_refuse_first_rounds_that_cannot_reach_the_stage_threshold(
    side, short, enough
) -> None:
    # The smallest first-round p-value is 2 / (n + 1) for both sides and
    # 1 / (n + 1) for one, which reaches 0.02 from 99 and 49 sets.
    assert _settings(side=side, n_resamples=(enough, 4999, 24999))[0][0] == enough
    for first in (short, 1):
        smallest = min(1.0, (2.0 if side == "both" else 1.0) / (first + 1))
        with pytest.raises(ValueError) as refusal:
            _settings(side=side, n_resamples=(first, 4999, 24999))
        message = str(refusal.value)
        assert f"smallest first-round p-value is {smallest:.3g}" in message
        assert f"n_resamples[0] of at least {enough}, or stage_threshold=0" in message
    # A threshold of 0 keeps the first round's p-values, and a larger one
    # needs fewer sets.
    _settings(side=side, n_resamples=(short, 4999, 24999), stage_threshold=0)
    _settings(side=side, n_resamples=(short, 4999, 24999), stage_threshold=0.05)


def test_the_stage_threshold_rule_names_the_fewest_sets_that_reach_it() -> None:
    # The last two thresholds sit where dividing 2 by them rounds to the
    # other side of a whole number of sets.
    for threshold in (0.3, 0.01, 1e-4, 0.0037, 2 / 7, 2 / 49, 0.0007202016564638098):
        with pytest.raises(ValueError, match="at least") as refusal:
            _settings(n_resamples=(1, 4999, 24999), stage_threshold=threshold)
        found = re.search(r"at least (\d+)", str(refusal.value))
        assert found is not None
        needed = int(found.group(1))
        # The first round's p-values reach the threshold from that many sets.
        for n_draws, reaches in ((needed, True), (needed - 1, False)):
            smallest = resampling_p_values(np.array([[0, n_draws, n_draws]]), "both")
            assert (smallest[0] <= threshold) == reaches
        _settings(n_resamples=(needed, 4999, 24999), stage_threshold=threshold)
        with pytest.raises(ValueError):
            _settings(n_resamples=(needed - 1, 4999, 24999), stage_threshold=threshold)


def test_resampling_keys_differ_by_seed_group_and_stage() -> None:
    keys = resampling_keys(0, np.arange(3), 1)

    assert keys.dtype == np.uint64
    np.testing.assert_array_equal(keys, resampling_keys(0, np.arange(3), 1))
    assert len(set(keys.tolist())) == 3
    assert not set(keys.tolist()) & set(resampling_keys(0, np.arange(3), 2).tolist())
    assert not set(keys.tolist()) & set(resampling_keys(1, np.arange(3), 1).tolist())


def test_counts_are_held_as_contiguous_float32() -> None:
    single = np.ones((2, 3), dtype=np.float32)
    # Every count below 2**24 keeps its value.
    counts = np.array([[0, 1, 2**24 - 1], [7, 65_535, 2**20 + 3]], dtype=np.uint32)
    strided = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]

    assert glm.count_array(single) is single
    for values in (counts, strided):
        held = glm.count_array(values)
        assert held.dtype == np.float32 and held.flags.c_contiguous
        assert held.tolist() == values.tolist()


# Random sets -------------------------------------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_random_sets_are_uniform_sets_that_do_not_depend_on_threads(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    key = resampling_keys(3, np.array([7]), 1)[0]
    n_draws = 4000 if not python_kernels else 600

    draws = random_sets.draw_uniform_sets(key, n_draws, 3, 9)

    serial = np.empty_like(draws)
    random_sets.draw_uniform_range(key, 0, 1, 3, 9, serial)
    np.testing.assert_array_equal(draws, serial)
    assert draws.dtype == np.int32
    assert all(len(set(row)) == 3 for row in draws.tolist())
    hits = np.bincount(draws.ravel(), minlength=9)
    assert stats.chisquare(hits).pvalue > 0.001


def test_a_set_of_every_position_holds_them_all() -> None:
    marked = np.zeros(5, dtype=bool)
    out = np.empty(5, dtype=np.int64)

    random_sets.draw_uniform_set(np.uint64(11), 5, 5, marked, out)

    assert sorted(out.tolist()) == [0, 1, 2, 3, 4]
    assert not marked.any()


# Linear algebra and special functions -----------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_cholesky_solves_and_drops_directions_without_information(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(6, 3))
    matrix = basis.T @ basis
    vector = rng.normal(size=3)
    factor = np.empty((3, 3))
    solved = np.empty(3)

    glm.cholesky(matrix, factor)
    glm.cholesky_solve(factor, vector, solved)

    np.testing.assert_allclose(factor, np.linalg.cholesky(matrix))
    np.testing.assert_allclose(solved, np.linalg.solve(matrix, vector))
    singular = np.zeros((3, 3))
    singular[:2, :2] = matrix[:2, :2]
    glm.cholesky(singular, factor)
    glm.cholesky_solve(factor, vector, solved)
    assert np.isinf(factor[2, 2])
    np.testing.assert_allclose(solved[:2], np.linalg.solve(matrix[:2, :2], vector[:2]))
    assert solved[2] == 0.0


@pytest.mark.parametrize("python_kernels", [False, True])
def test_digamma_and_trigamma_match_scipy(monkeypatch, python_kernels) -> None:
    _python_kernels(monkeypatch, python_kernels)
    x = np.r_[np.geomspace(1e-5, 1e5, 60), 9.99, 10.0, 10.01]

    digamma = np.array([glm.digamma(v) for v in x])
    trigamma = np.array([glm.trigamma(v) for v in x])

    np.testing.assert_allclose(digamma, special.digamma(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(trigamma, special.polygamma(1, x), rtol=1e-13)


# Negative binomial sizes ---------------------------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_sizes_maximize_the_likelihood_with_the_means_held_fixed(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(5)
    for size, scale in ((0.3, 2.0), (2.0, 0.4), (25.0, 30.0)):
        mu = rng.lognormal(np.log(scale), 0.5, 400)
        y = rng.negative_binomial(size, size / (size + mu)).astype(np.float64)

        dispersion = glm.size_dispersion(y, mu, 397.0)

        np.testing.assert_allclose(dispersion, size_root(y, mu, 397.0), rtol=1e-8)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_sizes_without_overdispersion_fall_back_to_the_pearson_equation(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    mu = np.full(6, 1.5)
    # sum((y - mu)^2 - y) is negative, but the Pearson sum exceeds 6 - 2.
    pearson = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 3.0])
    poisson = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

    dispersion = glm.size_dispersion(pearson, mu, 4.0)

    np.testing.assert_allclose(dispersion, size_root(pearson, mu, 4.0), rtol=1e-10)
    assert dispersion > 0
    assert glm.size_dispersion(poisson, mu, 4.0) == 0.0


def test_sizes_stop_at_their_limits(monkeypatch) -> None:
    # Compiled kernels fix the limit when they compile, so this runs as Python.
    _python_kernels(monkeypatch, True)
    monkeypatch.setattr(glm, "_LOG_SIZE_LIMIT", 1.0)
    rng = np.random.default_rng(6)
    mu = rng.lognormal(0.5, 0.3, 300)
    # A size of about 5, above the limit of e: Poisson.
    y = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(np.float64)

    assert glm.size_dispersion(y, mu, 298.0) == 0.0
    # Without a count, the likelihood rises as the size falls: the floor.
    zeros = np.zeros(300)
    assert glm.size_dispersion(zeros, mu, 298.0) == pytest.approx(np.e)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_large_counts_take_the_digamma_function(monkeypatch, python_kernels) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(7)
    mu = rng.lognormal(np.log(40.0), 0.3, 200)
    y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(np.float64) + 0.5

    first, second = glm.size_derivatives(y, mu, 2.5)

    expected = (
        special.digamma(y + 2.5)
        - special.digamma(2.5)
        + np.log(2.5 / (2.5 + mu))
        + (mu - y) / (2.5 + mu)
    ).sum()
    np.testing.assert_allclose(first, 2.5 * expected, rtol=1e-10)
    assert np.isfinite(second)


# Null fits ---------------------------------------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_null_fits_are_poisson_fits_on_the_control_cells(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    counts, design, _ = _screen(n_controls=80, sizes=(20,), n_genes=3)
    counts[2, :80] = 0.0  # no control counts

    coefficients = fit_null_models(counts.astype(np.float32), design, 80)

    assert np.isnan(coefficients[2]).all()
    for gene in (0, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poisson = sm.GLM(
                counts[gene, :80], design[:80], family=sm.families.Poisson()
            ).fit(tol=1e-14)
        np.testing.assert_allclose(coefficients[gene], poisson.params, atol=1e-6)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_poisson_fits_halve_steps_that_lower_the_likelihood(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(6)
    design = np.column_stack([np.ones(50), rng.normal(size=50)])
    y = rng.poisson(np.exp(3.0 + 0.5 * design[:, 1])).astype(np.float64)
    fitted = np.array([-8.0, 3.0])  # far from the fit, so a full step overshoots
    eta = np.empty(50)
    mu = np.empty(50)

    glm.fit_poisson(y, design, fitted, eta, mu)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = sm.GLM(y, design, family=sm.families.Poisson()).fit(tol=1e-14)
    np.testing.assert_allclose(fitted, expected.params, atol=1e-7)
    np.testing.assert_allclose(mu, expected.mu, rtol=1e-6)


def test_null_fits_refuse_shapes_they_cannot_use() -> None:
    counts, design, _ = _screen(n_controls=20, sizes=(5,), n_genes=2)

    with pytest.raises(ValueError, match="genes x cells"):
        fit_null_models(counts[:, :-1], design, 20)
    with pytest.raises(ValueError, match="outnumber"):
        fit_null_models(counts, design, 3)


# Pair statistics -----------------------------------------------------------------


@pytest.mark.parametrize("python_kernels", [False, True])
def test_pairs_are_scored_by_the_fit_on_their_treatment_and_control_cells(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    counts, design, targets = _screen(n_controls=60, sizes=(20, 15), n_genes=3)
    # Two calibration groups of control cells that share cells 10 to 19.
    calibration = [np.arange(0, 20), np.r_[10:20, 40:50]]
    members = [
        targets.members[targets.offsets[k] : targets.offsets[k + 1]] for k in range(2)
    ]
    groups = _groups(60, members + calibration, [False, False, True, True])
    pair_groups = np.array([0, 1, 2, 3, 0, 3])
    pair_rows = np.array([0, 1, 2, 0, 2, 1])
    coefficients = fit_null_models(counts, design, 60)
    keys = resampling_keys(0, groups.codes, 1)

    z, counted, fold, se = resampling_counts(
        counts, design, groups, keys, pair_groups, pair_rows, coefficients, 40
    )
    statistics = resampled_statistics(
        counts, design, groups, keys, pair_groups, pair_rows, coefficients, 40
    )

    for pair, (group, row) in enumerate(zip(pair_groups, pair_rows, strict=True)):
        cells = groups.members[groups.offsets[group] : groups.offsets[group + 1]]
        if groups.among_controls[group]:
            union, treated = np.arange(60), cells
        else:
            union = np.r_[np.arange(60), cells]
            treated = np.arange(60, len(union))
        expected = dense_pair(counts[row, union], design[union], treated)
        np.testing.assert_allclose([z[pair], fold[pair], se[pair]], expected, rtol=1e-6)
    np.testing.assert_array_equal(
        counted,
        np.column_stack(
            [
                (statistics <= z[:, None]).sum(1),
                (statistics >= z[:, None]).sum(1),
                np.full(len(z), 40),
            ]
        ),
    )


@pytest.mark.parametrize("python_kernels", [False, True])
def test_a_sample_without_control_counts_still_fits_the_pair(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(13)
    n_controls, n_treated = 300, 100
    n_cells = n_controls + n_treated
    library = rng.lognormal(np.log(3000), 0.3, n_cells)
    samples = np.arange(n_cells) % 2
    detected = library**0.8 * rng.lognormal(0, 0.05, n_cells)
    design = association_covariates(library, detected, samples)
    counts = rng.poisson(0.02 * library / 3000, size=(1, n_cells)).astype(float)
    counts[0, n_controls:] += rng.poisson(0.5, n_treated)
    # The control cells of sample 1 hold no count, so the gene's fit on the
    # control cells drives sample 1's coefficient far out.
    counts[0, :n_controls][samples[:n_controls] == 1] = 0.0
    groups = _groups(n_controls, [np.arange(n_controls, n_cells)], [False])
    coefficients = fit_null_models(counts, design, n_controls)

    z, _, fold, se = resampling_counts(
        counts,
        design,
        groups,
        resampling_keys(0, groups.codes, 1),
        np.zeros(1, int),
        np.zeros(1, int),
        coefficients,
        5,
    )

    assert np.abs(coefficients[0]).max() > 10
    expected = dense_pair(counts[0], design, np.arange(n_controls, n_cells))
    np.testing.assert_allclose([z[0], fold[0], se[0]], expected, rtol=1e-6)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_pairs_score_alike_whether_threads_take_groups_or_pairs(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    # The fourth group has no pair.
    counts, design, groups = _screen(n_controls=50, sizes=(12, 10, 8, 5), n_genes=4)
    pair_groups = np.array([2, 0, 1, 0, 2, 1, 1])
    pair_rows = np.array([3, 1, 0, 2, 0, 3, 3])
    coefficients = fit_null_models(counts, design, 50)
    keys = resampling_keys(9, groups.codes, 2)
    arguments = (counts, design, groups, keys, pair_groups, pair_rows, coefficients, 25)

    monkeypatch.setattr(associations_module.numba, "get_num_threads", lambda: 1)
    by_group = resampled_statistics(*arguments)
    monkeypatch.setattr(associations_module.numba, "get_num_threads", lambda: 64)
    by_pair = resampled_statistics(*arguments)
    monkeypatch.setattr(associations_module.numba, "get_num_threads", lambda: 1)
    small = resampled_statistics(*arguments, draw_bytes=1)

    np.testing.assert_array_equal(by_group, by_pair)
    np.testing.assert_array_equal(by_group, small)
    np.testing.assert_array_equal(by_pair[5], by_pair[6])


@pytest.mark.parametrize("python_kernels", [False, True])
@pytest.mark.parametrize("threads", [1, 64])
def test_statistics_land_in_the_rows_of_their_pairs(
    monkeypatch, python_kernels, threads
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    counts, design, groups = _screen(n_controls=50, sizes=(12, 10, 8), n_genes=4)
    pair_groups = np.array([2, 0, 1, 0, 2, 1])
    pair_rows = np.array([3, 1, 0, 2, 0, 3])
    coefficients = fit_null_models(counts, design, 50)
    keys = resampling_keys(4, groups.codes, 2)
    order = np.argsort(pair_groups, kind="stable")
    # One thread takes whole groups; 64 split each group's pairs among them.
    monkeypatch.setattr(associations_module.numba, "get_num_threads", lambda: threads)

    shuffled = resampled_statistics(
        counts, design, groups, keys, pair_groups, pair_rows, coefficients, 25
    )
    grouped = resampled_statistics(
        counts,
        design,
        groups,
        keys,
        pair_groups[order],
        pair_rows[order],
        coefficients,
        25,
    )

    np.testing.assert_array_equal(shuffled[order], grouped)
    assert not np.array_equal(shuffled, grouped)


def test_record_batches_fill_the_record_budget_within_their_bounds() -> None:
    # A pair's records hold 6 values per cell with 5 covariates.
    assert record_batch(3_000, 5) == RECORD_BYTES // (8 * 3_000 * 6)
    assert 2 < record_batch(3_000, 5) < 32
    assert record_batch(100, 5) == 32
    assert record_batch(10_000_000, 5) == 2
    assert record_batch(0, 5) == 32


@pytest.mark.parametrize("python_kernels", [False, True])
def test_pairs_of_genes_without_a_fit_have_no_statistic(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    counts, design, groups = _screen(n_controls=40, sizes=(10,), n_genes=2)
    counts[1, :40] = 0.0
    coefficients = fit_null_models(counts, design, 40)
    keys = resampling_keys(0, groups.codes, 1)

    z, counted, fold, se = resampling_counts(
        counts,
        design,
        groups,
        keys,
        np.zeros(2, int),
        np.array([0, 1]),
        coefficients,
        10,
    )

    assert np.isfinite(z[0]) and np.isnan(z[1])
    # Every set of the gene with a fit has a statistic, and none of the other.
    assert counted[0, 2] == 10
    np.testing.assert_array_equal(counted[1], [0, 0, 0])
    assert np.isnan(fold[1]) and np.isnan(se[1])


@pytest.mark.parametrize("python_kernels", [False, True])
def test_a_treatment_group_without_counts_has_no_fold_change(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    counts, design, groups = _screen(n_controls=40, sizes=(10,), n_genes=1)
    counts[0, 40:] = 0.0
    coefficients = fit_null_models(counts, design, 40)
    keys = resampling_keys(0, groups.codes, 1)

    _, _, fold, se = resampling_counts(
        counts,
        design,
        groups,
        keys,
        np.zeros(1, int),
        np.zeros(1, int),
        coefficients,
        5,
    )

    assert fold[0] == -np.inf and np.isfinite(se[0])


@pytest.mark.parametrize("python_kernels", [False, True])
def test_a_set_without_spread_has_no_statistic(monkeypatch, python_kernels) -> None:
    _python_kernels(monkeypatch, python_kernels)
    solved = np.empty(1)

    # One cell's weight is all the information of the fit.
    value = glm.score_statistic(1.0, 1.0, np.array([1.0]), np.array([[1.0]]), solved)

    assert np.isnan(value)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_a_variance_that_rounding_could_leave_counts_as_none(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    solved = np.empty(1)
    factor = np.array([[1.0]])

    def statistic(share: float) -> float:
        # The covariates explain all but ``share`` of a weight sum of 1.
        weighted = np.array([math.sqrt(1.0 - share)])
        return glm.score_statistic(1.0, 1.0, weighted, factor, solved)

    # Up to 1e-8 of the weight sum is rounding's reach: no statistic.
    for share in (1e-15, 1e-12, 1e-9, 0.9e-8):
        assert np.isnan(statistic(share))
    # Beyond it, the score over the root of the variance.
    for share in (1.1e-8, 1e-6, 0.5):
        np.testing.assert_allclose(statistic(share), 1 / math.sqrt(share), rtol=1e-6)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_a_set_that_the_covariates_determine_has_no_estimate(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(6)
    # The second covariate is the indicator of the first 10 of 50 cells.
    design = np.column_stack([np.ones(50), (np.arange(50) < 10).astype(float)])
    y = rng.poisson(5.0, 50).astype(np.float64)
    coefficients = np.array([math.log(y.mean()), 0.0])
    eta, mu, factor = np.empty(50), np.empty(50), np.empty((2, 2))
    dispersion = glm.fit_negative_binomial(y, design, coefficients, eta, mu, factor)

    def test(cells: np.ndarray) -> tuple[float, float, float]:
        return glm.set_test(y, mu, design, dispersion, cells, factor, np.empty(2))

    # The fit gives those cells their own counts, so their fold change of
    # about 0 is no estimate, and neither is its standard error.
    assert np.isnan(test(np.arange(10))).all()
    assert np.isfinite(test(np.arange(11))).all()


def test_a_target_of_exactly_the_cells_of_a_sample_without_controls_has_no_statistic() -> (
    None
):
    rng = np.random.default_rng(0)
    n_controls, size, n_genes = 300, 40, 10
    n_cells = n_controls + 2 * size
    library = rng.lognormal(np.log(3000), 0.4, n_cells)
    detected = library**0.8 * rng.lognormal(0, 0.05, n_cells)
    # The first target's cells are the second sample, which holds no control.
    samples = np.zeros(n_cells, dtype=int)
    samples[n_controls : n_controls + size] = 1
    design = association_covariates(library, detected, samples)
    mu = np.exp(-6.0 + np.log(library))
    counts = rng.negative_binomial(3.0, 3.0 / (3.0 + mu), size=(n_genes, n_cells))
    groups = _groups(
        n_controls,
        [
            np.arange(n_controls, n_controls + size),
            np.arange(n_controls + size, n_cells),
        ],
        [False, False],
    )

    result = pair_tests(
        counts.astype(np.float64),
        design,
        groups,
        np.repeat([0, 1], n_genes),
        np.tile(np.arange(n_genes), 2),
        n_resamples=(99, 199, 299),
    )

    # Rounding left about 1e-14 of the weight sum as the first target's
    # variance, which gave statistics near 1e7 and p-values of 0; the fit
    # matches its cells' counts, which gave fold changes of about 0.
    for values in (
        result.z,
        result.p_value,
        result.log2_fold_change,
        result.se_log2_fold_change,
    ):
        assert np.isnan(values[:n_genes]).all()
        assert np.isfinite(values[n_genes:]).all()
    assert (np.abs(result.z[n_genes:]) < 5).all()


# P-values ----------------------------------------------------------------------


def test_resampling_p_values_count_the_observed_set_on_each_side() -> None:
    counts = np.array([[0, 10, 9], [10, 0, 9], [4, 6, 9]])

    left = resampling_p_values(counts, "left")
    right = resampling_p_values(counts, "right")
    both = resampling_p_values(counts, "both")

    np.testing.assert_allclose(left, [0.1, 1.1, 0.5])
    np.testing.assert_allclose(right, [1.1, 0.1, 0.7])
    np.testing.assert_allclose(both, [0.2, 0.2, 1.0])


def test_resampling_p_values_leave_out_the_sets_without_a_statistic() -> None:
    rng = np.random.default_rng(3)
    statistics = rng.normal(size=(4, 99))
    # A third of the second row's sets and every set of the last row have no
    # statistic; the third row's do.
    statistics[1, ::3] = np.nan
    statistics[3] = np.nan
    z = np.array([-10.0, -10.0, 0.2, 1.0])

    counted = p_values_module._counts_beyond(statistics, z)

    for row in range(4):
        valid = statistics[row][~np.isnan(statistics[row])]
        np.testing.assert_array_equal(
            counted[row], [(valid <= z[row]).sum(), (valid >= z[row]).sum(), len(valid)]
        )
    # Over the 66 sets with a statistic, none at most -10 gives 1 / 67, not
    # the 1 / 100 that all 99 sets would give.
    left = resampling_p_values(counted, "left")
    assert left[1] == 1 / 67 and left[0] == 1 / 100
    both = resampling_p_values(counted, "both")
    assert both[1] == 2 / 67
    # Without any set with a statistic, the observed one alone gives 1.
    assert left[3] == 1.0 and both[3] == 1.0


@pytest.mark.parametrize("approximation", ["skew_normal", "none"])
def test_every_round_counts_only_the_sets_with_a_statistic(approximation) -> None:
    rng = np.random.default_rng(5)
    z = np.array([-6.0, -6.0, 0.0])
    # The first round's 99 sets: all of the first pair's have a statistic, 49
    # of the second's and 79 of the third's.
    first = np.array([[0, 99, 99], [0, 49, 49], [50, 30, 79]])
    second = rng.normal(size=(2, 999))
    second[1, :333] = np.nan
    asked = []

    def second_statistics(chosen: np.ndarray, n_draws: int) -> np.ndarray:
        assert n_draws == 999
        return second[chosen]

    def third_counts(chosen: np.ndarray, n_draws: int) -> np.ndarray:
        assert n_draws == 1999
        asked.append(chosen.tolist())
        # 1,000 of the 1,999 sets have a statistic, none at most z.
        return np.tile([0, 1000, 1000], (len(chosen), 1))

    p_value, stage, location, *_ = p_values_module.staged_p_values(
        z,
        first,
        second_statistics,
        third_counts,
        n_resamples=(99, 999, 1999),
        stage_threshold=0.05,
        approximation=approximation,
        side="both",
        statistics_bytes=1 << 20,
    )

    # The first round: 2 / 100 and 2 / 50 take the second round, while the
    # third pair keeps 2 x 31 / 80.
    assert stage[2] == 1 and p_value[2] == 2 * 31 / 80
    if approximation == "none":
        # The second round's 666 sets with a statistic, none at most -6.
        assert stage.tolist() == [2, 2, 1] and asked == []
        assert p_value[0] == 2 / 1000 and p_value[1] == 2 / 667
    else:
        # The fit takes only statistics without gaps, so the second pair
        # takes the third round, whose sets with a statistic give 2 / 1001.
        assert stage.tolist() == [2, 3, 1] and asked == [[1]]
        assert np.isfinite(location[0]) and np.isnan(location[1])
        assert p_value[1] == 2 / 1001


def test_skew_normal_fits_recover_the_parameters_of_their_statistics() -> None:
    rng = np.random.default_rng(7)
    shapes = np.array([-3.0, 0.0, 4.0])
    rows = np.vstack(
        [
            stats.skewnorm.rvs(a, loc=0.5, scale=1.5, size=20_000, random_state=rng)
            for a in shapes
        ]
    )
    # An exponential sample's skewness of about 2 is beyond the skew-normal's.
    exponential = rng.exponential(size=20_000)

    location, scale, shape = fit_skew_normals(
        np.vstack([rows, exponential, np.full(20_000, 2.0), np.r_[np.nan, rows[0, 1:]]])
    )

    # Near no skewness the moments trade location against shape, so only the
    # skewed rows recover their location.
    np.testing.assert_allclose(location[[0, 2]], 0.5, atol=0.08)
    np.testing.assert_allclose(scale[[0, 2]], 1.5, rtol=0.05)
    assert shape[0] < -1.5 and abs(shape[1]) < 1.0 and shape[2] > 2.0
    # The fallback skewness of 0.9 x 0.995 gives sceptre's shape of 6.141481.
    assert shape[3] == pytest.approx(6.141481, abs=1e-6)
    fitted = stats.skewnorm(shape[3], loc=location[3], scale=scale[3])
    np.testing.assert_allclose(
        [fitted.mean(), fitted.std()],
        [exponential.mean(), exponential.std()],
        rtol=1e-10,
    )
    assert np.isnan(location[4:]).all() and np.isnan(shape[4:]).all()
    empty = fit_skew_normals(np.full((1, 3), 1.0))
    assert np.isnan(empty[0][0])


def test_tail_ratios_compare_each_fit_with_its_statistics_on_one_side() -> None:
    rng = np.random.default_rng(8)
    skewed = stats.skewnorm.rvs(3.0, size=(1, 4999), random_state=rng)
    # A heavy right tail that a skew-normal's moments cannot follow.
    heavy = stats.t.rvs(2.5, size=(1, 4999), random_state=rng)
    statistics = np.vstack([skewed, heavy])
    location, scale, shape = fit_skew_normals(statistics)

    ratios = skew_normal_tail_ratios(
        statistics, np.array([5.0, 6.0]), location, scale, shape, "both"
    )
    right = skew_normal_tail_ratios(
        statistics[:1], np.array([-5.0]), location[:1], scale[:1], shape[:1], "right"
    )
    left = skew_normal_tail_ratios(
        statistics[:1], np.array([5.0]), location[:1], scale[:1], shape[:1], "left"
    )

    assert 0.7 < ratios[0] < 1.4
    assert ratios[1] < 0.5
    np.testing.assert_allclose(right, ratios[:1])
    upper = np.sort(skewed[0])[math.floor(0.01 * 4999)]
    expected = stats.skewnorm.cdf((upper - location[0]) / scale[0], shape[0]) / np.mean(
        skewed[0] <= upper
    )
    np.testing.assert_allclose(left, [expected])


@pytest.mark.parametrize("span_rows", [1, 3])
def test_fits_and_their_checks_do_not_depend_on_their_spans(
    monkeypatch, span_rows
) -> None:
    rng = np.random.default_rng(10)
    statistics = stats.skewnorm.rvs(2.0, size=(8, 101), random_state=rng)
    statistics[1] = np.round(statistics[1])  # ties
    statistics[2] = 1.0  # no spread
    statistics[3, ::7] = np.nan
    z = rng.normal(0.0, 2.0, 8)
    whole = fit_skew_normals(statistics)
    whole_ratios = skew_normal_tail_ratios(statistics, z, *whole, "both")
    # Spans of one row, or of three rows and a last one of two.
    monkeypatch.setattr(p_values_module, "_SPAN_BYTES", 8 * 101 * span_rows)

    fitted = fit_skew_normals(statistics)
    ratios = skew_normal_tail_ratios(statistics, z, *fitted, "both")
    counted = p_values_module._counts_beyond(statistics, z)

    for value, expected in zip(fitted, whole, strict=True):
        np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(ratios, whole_ratios)
    np.testing.assert_array_equal(
        counted,
        np.column_stack(
            [
                (statistics <= z[:, None]).sum(1),
                (statistics >= z[:, None]).sum(1),
                [101, 101, 101, 101 - 15, 101, 101, 101, 101],
            ]
        ),
    )
    # The median and percentiles are those of a full sort, NaN last.
    for n_draws in (1, 2, 3, 101):
        ranks = [
            n_draws // 2,
            math.ceil(0.99 * n_draws) - 1,
            math.floor(0.01 * n_draws),
        ]
        found = p_values_module._order_statistics(statistics[:, :n_draws], ranks)
        ordered = np.sort(statistics[:, :n_draws], axis=1)
        np.testing.assert_array_equal(found, ordered[:, ranks])


def test_skew_normal_p_values_are_its_tails() -> None:
    z = np.array([-3.0, -1.0, 2.0, 4.0])
    location = np.array([0.1, 0.0, -0.2, 0.3])
    scale = np.array([1.1, 1.0, 0.9, 1.2])
    shape = np.array([2.0, -1.0, 0.5, -3.0])
    x = (z - location) / scale

    left = skew_normal_p_values(z, location, scale, shape, "left")
    right = skew_normal_p_values(z, location, scale, shape, "right")
    both = skew_normal_p_values(z, location, scale, shape, "both")

    np.testing.assert_allclose(left, stats.skewnorm.cdf(x, shape), rtol=1e-9)
    np.testing.assert_allclose(right, stats.skewnorm.sf(x, shape), rtol=1e-9)
    np.testing.assert_allclose(both, np.minimum(1, 2 * np.minimum(left, right)))
    far = skew_normal_p_values(
        np.array([-30.0]), np.zeros(1), np.ones(1), np.array([-2.0]), "both"
    )
    assert 0 < far[0] < 1e-150


def test_calibration_grades_follow_the_sceptre_book() -> None:
    assert [calibration_grade(n) for n in (0, 1, 2, 6, 7, 14, 15, 90)] == [
        "excellent",
        "excellent",
        "mild",
        "mild",
        "moderate",
        "moderate",
        "severe",
        "severe",
    ]


def test_adjusted_p_values_adjust_the_tested_pairs() -> None:
    p_values = np.array([0.001, np.nan, 0.04, 0.5, 0.02])
    tested = np.array([True, True, True, False, True])

    adjusted, significant = adjusted_p_values(p_values, tested, 0.05)

    expected = stats.false_discovery_control([0.001, 0.04, 0.02], method="bh")
    np.testing.assert_allclose(adjusted[[0, 2, 4]], expected)
    assert np.isnan(adjusted[[1, 3]]).all()
    assert significant.tolist() == [True, False, True, False, True]
    none = adjusted_p_values(np.array([0.2]), np.array([False]), 0.1)
    assert np.isnan(none[0][0]) and not none[1][0]


# Calibration groups and pairs ----------------------------------------------------


def _pairs_of(rows_in_reference: bool, n_calibration_pairs: int | None):
    """The pairs of three label rows, two of them tested, and one calibration group.

    Row 0 is tested group 0 and row 2 group 1; row 0's pair with feature 1
    has 2 expressing cells of 5, row 2's with feature 3 has 1 of 4, and the
    calibration group of 3 cells expresses features 0 and 2 in 3 and 4 cells.
    """
    return association_pairs(
        (np.array([0, 2]), np.array([1, 3])),
        (np.array([2]), np.array([3])),
        row_groups=np.array([0, -1, 1]),
        sizes=np.array([5, 4, 3]),
        row_counts=np.array([[2, 9], [4, 1]], dtype=np.int32),
        row_features=np.array([1, 3]),
        calibration_counts=np.array([[3, 0, 4, 1]], dtype=np.int64),
        checked=np.arange(4),
        reference=np.array([10, 6, 8, 5], dtype=np.int64),
        n_reference=20,
        rows_in_reference=rows_in_reference,
        min_nonzero=2,
        n_calibration_pairs=n_calibration_pairs,
        seed=0,
    )


def test_pairs_count_their_cells_against_the_control_cells_or_all_others() -> None:
    against_controls = _pairs_of(False, 5)
    against_others = _pairs_of(True, 5)

    for pairs in (against_controls, against_others):
        assert pairs.sets.tolist() == [0, 0, 1, 1, 2]
        assert pairs.groups.tolist() == [0, 1, 2, 2, 1]
        assert pairs.features.tolist() == [1, 3, 0, 2, 3]
        assert all(counts.dtype == np.int64 for counts in pairs.qc)
        # Only the first discovery pair has 2 expressing treatment cells.
        assert pairs.passing_discovery == 1
    # Tested cells are compared with all 20 control cells, or with the other
    # cells; a calibration group always with the other cells.
    assert [counts.tolist() for counts in against_controls.qc] == [
        [5, 4, 3, 3, 4],
        [20, 20, 17, 17, 20],
        [2, 1, 3, 4, 1],
        [6, 5, 7, 4, 5],
    ]
    assert [counts.tolist() for counts in against_others.qc] == [
        [5, 4, 3, 3, 4],
        [15, 16, 17, 17, 16],
        [2, 1, 3, 4, 1],
        [4, 4, 7, 4, 4],
    ]
    assert against_controls.pass_qc.tolist() == [True, False, True, True, False]


def test_calibration_pairs_are_drawn_among_those_that_pass_qc() -> None:
    # By default as many as the discovery pairs that pass QC, one here.
    drawn = _pairs_of(False, None)
    calibration = drawn.sets == 1
    assert calibration.sum() == 1
    assert drawn.features[calibration][0] in (0, 2)
    assert drawn.pass_qc[calibration].all()
    # None when the check is off or no group passes.
    assert not (_pairs_of(False, 0).sets == 1).any()
    none = association_pairs(
        (np.array([0]), np.array([1])),
        (np.empty(0, np.int64), np.empty(0, np.int64)),
        row_groups=np.array([0]),
        sizes=np.array([5]),
        row_counts=np.array([[2, 9]], dtype=np.int32),
        row_features=np.array([1, 3]),
        calibration_counts=np.zeros((0, 4), dtype=np.int64),
        checked=np.arange(4),
        reference=np.array([10, 6, 8, 5]),
        n_reference=20,
        rows_in_reference=False,
        min_nonzero=2,
        n_calibration_pairs=None,
        seed=0,
    )
    assert none.sets.tolist() == [0] and none.pass_qc.tolist() == [True]


@pytest.mark.parametrize(
    ("arguments", "status"),
    [
        ((3, 3, None, 0, 0, "low"), "checked"),
        ((3, 1, 5, 2, 0, "low"), "checked"),
        # Pairs drawn, but none with a statistic: no check ran.
        ((3, 0, None, 4, 9, "low"), "no_calibration_statistic"),
        ((3, 0, 5, 4, 9, "high"), "no_calibration_statistic"),
        ((0, 0, 0, 5, 9, "low"), "off"),
        ((0, 0, None, 1, 9, "low"), "few_control_guides"),
        ((0, 0, None, 0, 9, "high"), "few_control_guides"),
        ((0, 0, None, 1, 0, "high"), "no_passing_discovery"),
        ((0, 0, None, 2, 0, "low"), "no_passing_discovery"),
        ((0, 0, 5, 2, 0, "low"), "no_passing_calibration"),
        ((0, 0, None, 2, 3, "low"), "no_passing_calibration"),
    ],
)
def test_the_calibration_status_says_why_no_check_ran(arguments, status) -> None:
    assert calibration_status(*arguments) == status
    assert status in CALIBRATION_STATUSES


def test_a_calibration_check_runs_unless_off_or_without_discovery_pairs() -> None:
    assert not calibration_runs(0, 10)
    assert not calibration_runs(None, 0)
    assert calibration_runs(None, 3)
    assert calibration_runs(5, 0)
    assert most_calibration_pairs(2, 10, 5, None) == 5
    assert most_calibration_pairs(2, 10, 50, None) == 20
    assert most_calibration_pairs(2, 10, 50, 7) == 7


def test_calibration_groups_hold_the_median_tested_targets_guides() -> None:
    guide_targets = np.array([0, 0, 1, -1, 2, 2, 2])

    assert calibration_group_size("target", guide_targets, 3, np.array([1])) == 1
    assert calibration_group_size("target", guide_targets, 3, np.array([0, 2])) == 2
    assert calibration_group_size("target", guide_targets, 3, np.array([2])) == 3
    assert calibration_group_size("guide", guide_targets, 3, np.array([2])) == 1


def test_negative_control_groups_are_distinct_random_sets_of_guides() -> None:
    few = negative_control_groups(5, 2, 100, seed=2)
    many = negative_control_groups(30, 3, 12, seed=2)

    assert sorted(tuple(g) for g in few) == sorted(
        (a, b) for a in range(5) for b in range(a + 1, 5)
    )
    assert len(many) == 12 and len({tuple(g) for g in many}) == 12
    assert all((np.diff(g) > 0).all() and len(g) == 3 for g in many)
    assert [g.tolist() for g in many] == [
        g.tolist() for g in negative_control_groups(30, 3, 12, seed=2)
    ]
    shared = negative_control_groups(9, 4, 50, seed=3)
    assert len(shared) == 50
    assert len(np.unique(np.concatenate(shared))) == 9


@pytest.mark.parametrize("python_kernels", [False, True])
def test_nonzero_counts_count_each_units_expressing_cells(
    monkeypatch, python_kernels
) -> None:
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(14)
    values = rng.poisson(0.7, size=(5, 40)).astype(np.float32)
    units = rng.integers(-1, 3, size=40)

    counted = nonzero_counts(values, units, 3)

    expected = np.stack(
        [(values[:, units == unit] > 0).sum(axis=1) for unit in range(3)]
    )
    np.testing.assert_array_equal(counted, expected)


def test_calibration_pairs_draw_among_the_pairs_that_pass_qc() -> None:
    passing = np.array([[True, False, True, True], [False, True, True, False]])

    chosen = calibration_pairs(passing, 3, seed=4)
    every = calibration_pairs(passing, 10, seed=4)

    assert len(chosen[0]) == 3
    assert passing[chosen].all()
    keys = chosen[0] * 4 + chosen[1]
    assert (np.diff(keys) > 0).all()
    assert sorted(zip(*every, strict=True)) == [(0, 0), (0, 2), (0, 3), (1, 1), (1, 2)]


# The whole test ------------------------------------------------------------------


def test_pair_tests_find_a_planted_effect_and_keep_null_pairs_uniform() -> None:
    counts, design, groups = _screen(
        n_controls=400, sizes=(150, 150), n_genes=40, seed=8, effect=0.3
    )
    pair_groups = np.repeat([0, 1], 40)
    pair_rows = np.tile(np.arange(40), 2)

    result = pair_tests(counts, design, groups, pair_groups, pair_rows, seed=1)

    assert result.stage[0] == 2
    assert result.p_value[0] < 1e-8
    assert result.z[0] < -5 and result.log2_fold_change[0] < -1
    assert np.isfinite(
        [result.skew_location[0], result.skew_scale[0], result.skew_shape[0]]
    ).all()
    null = np.delete(result.p_value, 0)
    assert stats.kstest(null, "uniform").pvalue > 0.01
    once = result.stage == 1
    assert np.isnan(result.skew_location[once]).all()
    assert (result.p_value[once] > 0.02).all()


@pytest.mark.parametrize("side", ["left", "right"])
def test_pair_tests_take_one_side(side) -> None:
    counts, design, groups = _screen(n_controls=200, sizes=(80,), n_genes=2, effect=0.3)

    result = pair_tests(
        counts, design, groups, np.zeros(2, int), np.arange(2), side=side, seed=2
    )

    if side == "left":
        assert result.p_value[0] < 1e-6
    else:
        assert result.p_value[0] > 0.99


def test_pair_tests_without_the_approximation_keep_the_sets_p_values() -> None:
    counts, design, groups = _screen(n_controls=200, sizes=(80,), n_genes=2, effect=0.3)
    # The planted pair twice, so that a small budget splits the second round.
    arguments = (counts, design, groups, np.zeros(3, int), np.array([0, 0, 1]))
    settings = {
        "approximation": "none",
        "n_resamples": (49, 99, 199),
        "stage_threshold": 0.05,
    }

    exact = pair_tests(*arguments, **settings)
    batched = pair_tests(*arguments, **settings, statistics_bytes=8)

    assert exact.stage[:2].tolist() == [2, 2]
    assert exact.p_value[0] == pytest.approx(2 / 100)
    assert np.isnan(exact.skew_location[0])
    np.testing.assert_array_equal(exact.p_value, batched.p_value)


def test_pair_tests_take_a_third_round_when_the_fit_fails(monkeypatch) -> None:
    counts, design, groups = _screen(n_controls=200, sizes=(80,), n_genes=2, effect=0.3)
    monkeypatch.setattr(p_values_module, "_TAIL_RATIO_FLOOR", np.inf)

    result = pair_tests(
        counts,
        design,
        groups,
        np.zeros(2, int),
        np.arange(2),
        n_resamples=(99, 199, 999),
    )

    assert result.stage[0] == 3
    assert result.p_value[0] == pytest.approx(2 / 1000)
    assert np.isnan(result.skew_shape[0])


def test_the_second_round_holds_its_statistics_once() -> None:
    # 200 pairs whose 4,999 sets of the second round fill one block of 8 MB.
    counts, design, groups = _screen(n_controls=100, sizes=(10,), n_genes=200, seed=3)
    pairs = (np.zeros(200, int), np.arange(200))
    block = 8 * 4999 * 200
    settings = {
        "n_resamples": (99, 4999, 999),
        "stage_threshold": 1.0,
        "statistics_bytes": block,
    }
    # The kernels compile and the fits load SciPy's functions first.
    pair_tests(counts[:2], design, groups, pairs[0][:2], pairs[1][:2], **settings)
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    start = tracemalloc.get_traced_memory()[0]
    try:
        result = pair_tests(counts, design, groups, *pairs, **settings)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        if not tracing:
            tracemalloc.stop()

    assert (result.stage >= 2).all()
    # Besides the statistics, their random sets and records take under 1 MB,
    # and the fits a working copy of about 1 MiB, never all of them again.
    assert peak < 1.5 * block


def test_pair_tests_of_genes_without_a_fit_have_no_p_value() -> None:
    counts, design, groups = _screen(n_controls=60, sizes=(20,), n_genes=1)
    counts[0, :60] = 0.0

    result = pair_tests(counts, design, groups, np.zeros(1, int), np.zeros(1, int))

    assert np.isnan(result.z[0]) and np.isnan(result.p_value[0])
    assert result.stage[0] == 1


def test_the_result_type_is_exported() -> None:
    assert PerturbationAssociations is associations_module.PerturbationAssociations


# The operation -------------------------------------------------------------------

N_OPERATION_GENES = 30
# Guides 0 and 1 target RNA0, guides 2 and 3 target RNA1, and guides 4 to 7 are
# non-targeting.
GUIDE_TARGETS = ["RNA0", "RNA0", "RNA1", "RNA1", "NT", "NT", "NT", "NT"]
GUIDE_GENE_IDS = ["RNA0", "RNA0", "RNA1", "RNA1", "", "", "", ""]
PLANTED = {("RNA0", "RNA0"): -1, ("RNA0", "RNA5"): -1, ("RNA1", "RNA6"): 1}
FAST = {"n_resamples": (199, 999, 1999)}


def _operation_screen() -> tuple[np.ndarray, ...]:
    """Counts of 60 cells per targeting guide and 75 per control guide.

    RNA0 lowers its own gene and RNA5, and RNA1 raises RNA6. Three more cells
    carry no guide, or guides of both targets.
    """
    rng = np.random.default_rng(11)
    guide_of = np.r_[np.repeat(np.arange(4), 60), np.repeat(np.arange(4, 8), 75)]
    n_cells = len(guide_of) + 3
    library = rng.lognormal(np.log(2000), 0.4, n_cells)
    base = rng.normal(-5.0, 0.7, N_OPERATION_GENES)
    mu = np.exp(base[None, :] + np.log(library)[:, None])
    target = np.full(n_cells, -1)
    target[: len(guide_of)] = np.where(guide_of < 4, guide_of // 2, -1)
    mu[target == 0, 0] *= 0.15
    mu[target == 0, 5] *= 0.3
    mu[target == 1, 6] *= 3.0
    rna = rng.negative_binomial(3.0, 3.0 / (3.0 + mu))
    guides = np.zeros((n_cells, 8), dtype=np.uint16)
    guides[np.arange(len(guide_of)), guide_of] = 20
    guides[-2, [0, 2]] = 20
    guides[-1, [1, 3]] = 20
    runs = np.where(np.arange(n_cells) % 3 == 0, "run1", "run2")
    mito = rng.uniform(0.0, 5.0, n_cells)
    return rna.astype(np.uint16), guides, runs, mito


OPERATION_RNA, OPERATION_GUIDES, OPERATION_RUNS, OPERATION_MITO = _operation_screen()


def _operation_store(
    tmp_path: Path,
    rna: np.ndarray | None = None,
    guides: np.ndarray | None = None,
    name: str = "store.zarr",
    targets: list[str] = GUIDE_TARGETS,
    gene_ids: list[str] = GUIDE_GENE_IDS,
) -> DataStore:
    runs = np.asarray(OPERATION_RUNS, dtype=object)
    return write_screen(
        tmp_path / name,
        OPERATION_RNA if rna is None else rna,
        OPERATION_GUIDES if guides is None else guides,
        targets=targets,
        target_ids=gene_ids,
        cell_columns={"run": runs, "mito": OPERATION_MITO, "label": runs},
        nthreads=2,
    )


def _every_gene(store: DataStore) -> ArtifactRef:
    return store.features.snapshot(feature_indexes=np.arange(N_OPERATION_GENES))


def _warned_associations(
    store: DataStore, *args, **kwargs
) -> tuple[ArtifactRef, list[str]]:
    """Run association tests, and return them with the messages they warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tests = store.perturbation.associations(*args, **kwargs)
    return tests, [
        str(entry.message) for entry in caught if entry.category is UserWarning
    ]


@dataclass(frozen=True)
class _Shared:
    path: Path
    store: DataStore
    labels: ArtifactRef


@pytest.fixture(scope="module")
def shared(tmp_path_factory) -> _Shared:
    """One store of the screen for the tests that keep its data."""
    folder = tmp_path_factory.mktemp("associations")
    store = _operation_store(folder)
    return _Shared(folder / "store.zarr", store, fixed_labels(store, ["NT"]))


def test_associations_find_the_planted_effects(shared) -> None:
    store, labels = shared.store, shared.labels

    tests = store.perturbation.associations(labels, features=_every_gene(store), **FAST)

    assert (tests.kind, tests.assay) == ("perturbation_associations", "RNA")
    result = store.perturbation.load_associations(tests)
    assert isinstance(result, PerturbationAssociations)
    assert (result.alpha, result.side, result.integration, result.from_assay) == (
        0.1,
        "both",
        "union",
        "RNA",
    )
    discovery = result.discovery.set_index(["target", "feature"])
    assert len(discovery) == 2 * N_OPERATION_GENES
    assert (discovery["n_cells"] == 120).all()
    assert (discovery["n_control_cells"] == 300).all()
    assert (discovery["feature_name"] == discovery.index.get_level_values(1)).all()
    for pair, sign in PLANTED.items():
        row = discovery.loc[pair]
        assert row["significant"] and np.sign(row["z"]) == sign
        assert row["stage"] == 2 and row["p_value"] < 1e-6
        assert np.sign(row["log2_fold_change"]) == sign
        assert np.isfinite(row["skew_shape"])
    assert set(discovery.index[discovery["significant"]]) == set(PLANTED)
    untested = discovery[~discovery["pass_qc"]]
    assert untested["p_value"].isna().all() and (untested["stage"] == 0).all()
    calibration = result.calibration
    assert len(calibration) > 0
    assert set(calibration["target"]) <= {f"control group {k}" for k in range(1, 7)}
    assert all(guides.count("+") == 1 for guides in calibration["control_guides"])
    assert (calibration["n_cells"] == 150).all()
    assert (calibration["n_control_cells"] == 150).all()
    assert result.calibration_grade == "excellent"
    assert result.n_false_discoveries == int(calibration["significant"].sum())
    positive = result.positive.set_index("target")
    assert positive["feature"].to_dict() == {"RNA0": "RNA0", "RNA1": "RNA1"}
    assert positive.loc["RNA0", "significant"]
    assert "guide" not in result.discovery


def test_associations_store_the_names_of_each_group_once(shared) -> None:
    from cytearc.storage.artifacts import artifact_group

    store, labels = shared.store, shared.labels
    tests = store.perturbation.associations(labels, features=_every_gene(store), **FAST)

    stored = artifact_group(store.zw, tests)
    result = store.perturbation.load_associations(tests)
    # Each pair names its group of treatment cells, a tested target or a
    # calibration group, whose target and guides are stored once.
    n_pairs = len(result.discovery) + len(result.calibration) + len(result.positive)
    assert stored["groups"].shape == (n_pairs,)
    assert "targets" not in stored and "guides" not in stored
    targets = np.asarray(stored["group_targets"][:]).astype(str).tolist()
    guides = np.asarray(stored["group_guides"][:]).astype(str).tolist()
    assert targets == ["RNA0", "RNA1"] + [f"control group {k}" for k in range(1, 7)]
    assert guides[:2] == ["", ""]
    assert sorted(guides[2:]) == [
        "CRISPR4+CRISPR5",
        "CRISPR4+CRISPR6",
        "CRISPR4+CRISPR7",
        "CRISPR5+CRISPR6",
        "CRISPR5+CRISPR7",
        "CRISPR6+CRISPR7",
    ]


def test_associations_test_tables_of_pairs_one_guide_at_a_time(shared) -> None:
    store, labels = shared.store, shared.labels
    pairs = pd.DataFrame({"target": ["RNA0", "RNA1"], "feature": ["RNA0", "RNA6"]})
    positive = pd.DataFrame({"target": ["RNA1"], "feature": ["RNA6"]})

    tests = store.perturbation.associations(
        labels,
        pairs=pairs,
        positive_controls=positive,
        integration="singleton",
        n_calibration_pairs=3,
        **FAST,
    )

    result = store.perturbation.load_associations(tests)
    assert result.integration == "singleton"
    assert result.discovery[["guide", "target", "feature"]].values.tolist() == [
        ["CRISPR0", "RNA0", "RNA0"],
        ["CRISPR1", "RNA0", "RNA0"],
        ["CRISPR2", "RNA1", "RNA6"],
        ["CRISPR3", "RNA1", "RNA6"],
    ]
    assert (result.discovery["n_cells"] == 60).all()
    assert result.discovery["significant"].all()
    assert len(result.calibration) == 3
    assert not any("+" in guides for guides in result.calibration["control_guides"])
    assert (result.calibration["n_cells"] == 75).all()
    assert result.positive["guide"].tolist() == ["CRISPR2", "CRISPR3"]


def test_association_pairs_match_features_by_id_then_name(shared) -> None:
    store = shared.store
    rows = _label_rows(store, shared.labels, "target")
    ids = np.array(["ENSG0", "ENSG1", "ENSG2", "ENSG3"])
    names = np.array(["A", "B", "B", "ENSG0"])
    frame = pd.DataFrame(
        {"target": ["RNA1", "RNA0", "RNA0"], "feature": ["A", "ENSG0", "ENSG3"]}
    )

    matched = pair_table_rows(frame, "pairs", rows.names, ids, names)

    assert [values.tolist() for values in matched] == [[0, 0, 1], [0, 3, 0]]
    for wrong, message in (
        ([("RNA0", "A")], "must be a DataFrame"),
        (pd.DataFrame({"target": ["RNA0"]}), "columns 'target' and 'feature'"),
        (pd.DataFrame({"target": ["GENE9"], "feature": ["A"]}), "'GENE9'"),
        (pd.DataFrame({"target": ["RNA0"], "feature": ["B"]}), "'B'"),
        (pd.DataFrame({"target": ["RNA0"], "feature": ["Z"]}), "'Z'"),
    ):
        error = TypeError if isinstance(wrong, list) else ValueError
        with pytest.raises(error, match=message):
            pair_table_rows(wrong, "pairs", rows.names, ids, names)
    # Repeated IDs match nothing, so names decide.
    repeated = pair_table_rows(
        pd.DataFrame({"target": ["RNA0"], "feature": ["A"]}),
        "pairs",
        rows.names,
        np.array(["X", "X", "Y", "Z"]),
        names,
    )
    assert [values.tolist() for values in repeated] == [[0], [0]]


def test_cell_totals_read_only_the_cells_they_return(shared, monkeypatch) -> None:
    store = shared.store
    every = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=np.float64)
    reads = record_cell_reads(monkeypatch, store, {"RNA_nCounts"})
    for rows in (
        np.array([3, 4, 5, 90, 91, 400], dtype=np.int64),
        np.array([400, 3, 91, 4], dtype=np.int64),
        np.array([91, 3, 91, 3, 0], dtype=np.int64),
    ):
        reads["RNA_nCounts"].clear()

        totals = store.RNA._cell_count_totals(rows)

        assert totals.dtype == np.float64
        np.testing.assert_array_equal(totals, every[rows])
        # Only the rows asked for are read, never the whole column.
        read = np.concatenate(reads["RNA_nCounts"])
        assert set(read.tolist()) == set(rows.tolist())
    empty = store.RNA._cell_count_totals(np.empty(0, dtype=np.int64))
    assert empty.dtype == np.float64 and empty.shape == (0,)


def test_association_tests_read_the_totals_of_their_tested_cells_only(
    shared, monkeypatch
) -> None:
    store, labels = shared.store, shared.labels
    reads = record_cell_reads(monkeypatch, store, {"RNA_nCounts"})

    store.perturbation.associations(
        labels,
        pairs=pd.DataFrame({"target": ["RNA0"], "feature": ["RNA0"]}),
        positive_controls=None,
        n_calibration_pairs=0,
        invalidate_cache=True,
        **FAST,
    )

    # The control cells and the cells of RNA0 alone, not those of RNA1, of
    # two targets or without a guide.
    read = set(np.concatenate(reads["RNA_nCounts"]).tolist())
    guides = OPERATION_GUIDES > 0
    tested = guides[:, 4:].any(axis=1) & ~guides[:, :4].any(axis=1)
    tested |= guides[:, :2].any(axis=1) & ~guides[:, 2:].any(axis=1)
    assert read == set(np.flatnonzero(tested).tolist())


def test_associations_are_reused_read_only_and_refused_when_new(shared) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    tests = store.perturbation.associations(labels, features=features, **FAST)
    reader = DataStore(
        str(shared.path),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )

    assert reader.perturbation.associations(labels, features=features, **FAST) == tests
    with pytest.raises(PermissionError, match="test_perturbation_associations"):
        reader.perturbation.associations(labels, features=features, seed=3, **FAST)
    again = store.perturbation.associations(
        labels, features=features, invalidate_cache=True, **FAST
    )
    assert again != tests
    pd.testing.assert_frame_equal(
        store.perturbation.load_associations(again).discovery,
        store.perturbation.load_associations(tests).discovery,
    )


def test_associations_adjust_for_samples_and_covariates(shared) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    plain = store.perturbation.associations(labels, features=features, **FAST)

    adjusted = store.perturbation.associations(
        labels, features=features, sample_column="run", covariates=["mito"], **FAST
    )

    assert adjusted != plain
    result = store.perturbation.load_associations(adjusted).discovery
    found = result.set_index(["target", "feature"])["significant"]
    assert set(found.index[found]) == set(PLANTED)
    plain_z = store.perturbation.load_associations(plain).discovery["z"].to_numpy()
    assert not np.array_equal(result["z"].to_numpy(), plain_z, equal_nan=True)


def test_associations_ignore_missing_values_of_untested_cells(shared) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    # The last three cells carry no guide, or guides of both targets, so no
    # test reads them.
    runs = np.asarray(OPERATION_RUNS, dtype=object)
    runs[-3:] = None
    store.cells.insert("run_untested_missing", runs, overwrite=True)
    mito = pd.array(OPERATION_MITO, dtype="Float64")
    mito[-3:] = pd.NA
    store.cells.insert("mito_untested_missing", mito, overwrite=True)
    complete = store.perturbation.load_associations(
        store.perturbation.associations(
            labels, features=features, sample_column="run", covariates=["mito"], **FAST
        )
    )

    partial = store.perturbation.load_associations(
        store.perturbation.associations(
            labels,
            features=features,
            sample_column="run_untested_missing",
            covariates=["mito_untested_missing"],
            **FAST,
        )
    )

    for name in ("discovery", "calibration", "positive"):
        pd.testing.assert_frame_equal(getattr(partial, name), getattr(complete, name))
    # A tested cell still needs its sample label.
    runs[0] = None
    store.cells.insert("run_tested_missing", runs, overwrite=True)
    with pytest.raises(ValueError, match="has no label for 1 selected cells"):
        store.perturbation.associations(
            labels, features=features, sample_column="run_tested_missing", **FAST
        )


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("Int64", np.arange(len(OPERATION_RUNS)) % 5),
        ("boolean", np.arange(len(OPERATION_RUNS)) % 2 == 0),
    ],
)
def test_associations_refuse_missing_covariates_of_tested_cells(
    shared, dtype, values
) -> None:
    store, labels = shared.store, shared.labels
    column = f"missing_{dtype.lower()}"
    nullable = pd.array(values, dtype=dtype)
    # Cell 0 carries a guide of RNA0, so the tests would read its missing
    # value, which is stored as 0 or False.
    nullable[0] = pd.NA
    store.cells.insert(column, nullable, overwrite=True)

    with pytest.raises(
        ValueError, match=f"The covariate '{column}' has values that are not finite"
    ):
        store.perturbation.associations(
            labels, features=_every_gene(store), covariates=[column], **FAST
        )


def test_associations_refuse_a_covariate_of_one_value(shared) -> None:
    store, labels = shared.store, shared.labels
    # The tests read the 300 control cells and the 240 cells of the two
    # targets, over which 1.1 repeated has a standard deviation above 0.
    assert np.full(540, 1.1).std() > 0
    store.cells.insert("constant", np.full(len(OPERATION_RUNS), 1.1), overwrite=True)

    with pytest.raises(
        ValueError,
        match="^The covariate 'constant' is constant over the tested cells$",
    ):
        store.perturbation.associations(
            labels, features=_every_gene(store), covariates=["constant"], **FAST
        )


def test_associations_held_and_streamed_counts_agree(shared, monkeypatch) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    held = store.perturbation.load_associations(
        store.perturbation.associations(labels, features=features, **FAST)
    )

    # Counts that may not be held are read again in bands of 4 features.
    monkeypatch.setattr(perturbation_operations, "_ASSOCIATION_HELD_SHARE", 0.0)
    monkeypatch.setattr(perturbation_operations, "_BAND_BYTES", 8 * 540 * 4)
    streamed = store.perturbation.associations(
        labels, features=features, invalidate_cache=True, **FAST
    )

    loaded = store.perturbation.load_associations(streamed)
    for name in ("discovery", "calibration", "positive"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(held, name))


def test_association_reads_leave_room_for_the_held_counts(shared, monkeypatch) -> None:
    from cytearc.assay import normalization

    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    reads: list[tuple[int, int]] = []
    read_blocks = normalization.feature_value_blocks

    def recorded(assay, band, *args, **kwargs):
        reads.append((len(band), kwargs["resident_bytes"]))
        return read_blocks(assay, band, *args, **kwargs)

    monkeypatch.setattr(normalization, "feature_value_blocks", recorded)
    store.perturbation.associations(
        labels, features=features, invalidate_cache=True, **FAST
    )
    held = reads[0]
    reads.clear()
    monkeypatch.setattr(perturbation_operations, "_ASSOCIATION_HELD_SHARE", 0.0)
    store.perturbation.associations(
        labels, features=features, invalidate_cache=True, **FAST
    )

    # The first read counts every feature's cells for the QC, with the held
    # float32 counts of the 540 tested cells on top when they are held.
    assert held[0] == reads[0][0] == N_OPERATION_GENES
    assert held[1] - reads[0][1] == 4 * 540 * N_OPERATION_GENES


def test_associations_refuse_a_budget_without_room_for_one_feature(shared) -> None:
    store, labels = shared.store, shared.labels
    # The second round's random sets of the 150 cells of a group fill it.
    budgeted = open_store(store.zarr_loc, mem_budget=8_000_000)

    with pytest.raises(MemoryError, match="one feature over 540 tested cells"):
        budgeted.perturbation.associations(
            labels, features=_every_gene(store), n_resamples=(200, 1000, 100_000)
        )


def test_association_refusals_name_the_budget_that_runs_the_tests(shared) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    arguments = {"features": features, "invalidate_cache": True, **FAST}
    default = store.perturbation.load_associations(
        store.perturbation.associations(labels, **arguments)
    )

    def run(budgeted: DataStore) -> ArtifactRef:
        return budgeted.perturbation.associations(labels, **arguments)

    # The tests hold far more than this budget, the pairs' arrays alone.
    with pytest.raises(MemoryError, match="one feature over 540 tested cells") as low:
        run(open_store(store.zarr_loc, nthreads=2, mem_budget=1000))
    needed = named_budget(low.value)

    # A budget one byte short asks for the same budget, which runs the tests
    # with the results of the default budget; the wide count chunk below
    # checks the traced peak.
    with pytest.raises(MemoryError) as short:
        run(open_store(store.zarr_loc, nthreads=2, mem_budget=needed - 1))
    assert named_budget(short.value) == needed
    tight = store.perturbation.load_associations(
        run(open_store(store.zarr_loc, nthreads=2, mem_budget=needed))
    )
    for name in ("discovery", "calibration", "positive"):
        pd.testing.assert_frame_equal(getattr(tight, name), getattr(default, name))


@pytest.mark.slow
def test_association_refusals_count_the_read_of_a_wide_count_chunk(tmp_path) -> None:
    # 1,000 more genes widen the stored chunk of counts, which a read of one
    # gene decodes whole, beyond the slack of the tests' other arrays; the
    # tests take the first 5 genes, without calibration pairs.
    rng = np.random.default_rng(23)
    wide = np.concatenate(
        [OPERATION_RNA, rng.poisson(2.0, size=(len(OPERATION_RNA), 1000))], axis=1
    ).astype(np.uint16)
    store = _operation_store(tmp_path, rna=wide)
    labels = fixed_labels(store, ["NT"])
    arguments = {
        "features": store.features.snapshot(feature_indexes=np.arange(5)),
        "n_calibration_pairs": 0,
        "invalidate_cache": True,
        **FAST,
    }
    default = store.perturbation.load_associations(
        store.perturbation.associations(labels, **arguments)
    )

    def run(budgeted: DataStore) -> ArtifactRef:
        return budgeted.perturbation.associations(labels, **arguments)

    # The named budget runs the tests within it; it once left out the read,
    # which then failed in the stream planner.
    _, _, tests = check_budget(store.zarr_loc, run, check_writes=False, nthreads=2)
    tight = store.perturbation.load_associations(tests)
    for name in ("discovery", "positive"):
        pd.testing.assert_frame_equal(getattr(tight, name), getattr(default, name))


def test_loading_associations_checks_the_memory_budget(shared, monkeypatch) -> None:
    store, labels = shared.store, shared.labels
    tests = store.perturbation.associations(labels, features=_every_gene(store), **FAST)
    loaded = store.perturbation.load_associations(tests)
    n_pairs = sum(len(getattr(loaded, name)) for name in ASSOCIATION_SETS)

    budgeted = open_store(store.zarr_loc, mem_budget=1000)
    with pytest.raises(
        MemoryError, match=f"^The association tests of {n_pairs} pairs hold about"
    ):
        budgeted.perturbation.load_associations(tests)


def test_associations_plan_only_the_calibration_pairs_a_check_can_draw(
    tmp_path, monkeypatch
) -> None:
    # Two control guides make two calibration groups of one guide, which can
    # draw at most 2 x 30 pairs, while the four guides tested one at a time
    # make 4 x 30 trans pairs and 4 positive controls.
    guides = OPERATION_GUIDES.copy()
    guides[:, [6, 7]] = 0
    store = _operation_store(tmp_path, guides=guides)
    arguments = {
        "labels": fixed_labels(store, ["NT"]),
        "features": _every_gene(store),
        "integration": "singleton",
        **FAST,
    }
    mib = 1 << 20
    # A pair holds a MiB here, so the pairs' arrays decide the budget: 184
    # pairs fit in 214 MiB, which 244 pairs, with as many calibration pairs as
    # discovery pairs, would not.
    monkeypatch.setattr(perturbation_operations, "_ASSOCIATION_PAIR_BYTES", mib)
    with pytest.raises(MemoryError, match="one feature over"):
        open_store(store.zarr_loc, mem_budget=183 * mib).perturbation.associations(
            **arguments
        )

    tests = open_store(store.zarr_loc, mem_budget=214 * mib).perturbation.associations(
        **arguments
    )

    result = store.perturbation.load_associations(tests)
    assert (len(result.discovery), len(result.positive)) == (120, 4)
    assert 0 < len(result.calibration) <= 60
    assert set(result.calibration["control_guides"]) == {"CRISPR4", "CRISPR5"}


def test_associations_without_a_calibration_check(shared, tmp_path) -> None:
    store, labels = shared.store, shared.labels

    tests, messages = _warned_associations(
        store, labels, features=_every_gene(store), n_calibration_pairs=0, **FAST
    )

    skipped = store.perturbation.load_associations(tests)
    assert skipped.calibration.empty and skipped.calibration_grade == "none"
    # A check that is turned off raises no warning.
    assert not [message for message in messages if "calibration" in message]
    # One control guide makes no group of two.
    guides = OPERATION_GUIDES.copy()
    guides[:, 4] = guides[:, 4:].max(axis=1)
    guides[:, 5:] = 0
    single = _operation_store(tmp_path, guides=guides, name="single.zarr")
    arguments = {"features": _every_gene(single), **FAST}
    single_labels = fixed_labels(single, ["NT"])
    tests, messages = _warned_associations(single, single_labels, **arguments)
    assert len(messages) == 1 and messages[0].startswith(
        "Control cells with one guide carry 1 control guide, fewer than the 2"
    )
    assert messages[0].endswith("so no calibration check was run")
    loaded = single.perturbation.load_associations(tests)
    assert loaded.calibration.empty and loaded.calibration_grade == "none"
    # Saved tests warn again, alike.
    assert _warned_associations(single, single_labels, **arguments) == (
        tests,
        messages,
    )


def test_associations_check_the_calibration_without_discovery_pairs(shared) -> None:
    store, labels = shared.store, shared.labels
    no_pairs = pd.DataFrame({"target": [], "feature": []})

    checked = store.perturbation.load_associations(
        store.perturbation.associations(
            labels, pairs=no_pairs, n_calibration_pairs=5, **FAST
        )
    )

    assert checked.discovery.empty and len(checked.positive) == 2
    # The calibration pairs take features of the whole assay.
    assert len(checked.calibration) == 5 and checked.calibration["pass_qc"].all()
    assert checked.calibration_grade == "excellent"
    # By default the check matches the discovery pairs that pass QC.
    with pytest.warns(UserWarning, match="No discovery pair passes QC"):
        unchecked = store.perturbation.associations(labels, pairs=no_pairs, **FAST)
    loaded = store.perturbation.load_associations(unchecked)
    assert loaded.calibration.empty and loaded.calibration_grade == "none"


@pytest.mark.parametrize(
    ("n_calibration_pairs", "message"),
    [
        (None, "No discovery pair passes QC, and by default the calibration check"),
        (5, "No pair of a calibration group and a feature passes QC"),
    ],
)
def test_associations_say_why_no_calibration_check_ran(
    shared, n_calibration_pairs, message
) -> None:
    store, labels = shared.store, shared.labels
    # No feature is expressed by 1,000 of the screen's cells.
    arguments = {
        "features": _every_gene(store),
        "min_nonzero": 1000,
        "n_calibration_pairs": n_calibration_pairs,
        **FAST,
    }

    tests, messages = _warned_associations(store, labels, **arguments)

    assert len(messages) == 1 and messages[0].startswith(message)
    result = store.perturbation.load_associations(tests)
    assert not result.discovery["pass_qc"].any()
    assert result.calibration.empty and result.calibration_grade == "none"
    # Saved tests warn again, alike.
    assert _warned_associations(store, labels, **arguments) == (tests, messages)


def test_associations_warn_about_targets_without_a_gene(tmp_path) -> None:
    # GENE9 has neither an ID nor a feature of its name.
    store = _operation_store(
        tmp_path,
        targets=["RNA0", "RNA0", "GENE9", "GENE9", "NT", "NT", "NT", "NT"],
        gene_ids=["RNA0", "RNA0", "", "", "", "", "", ""],
    )
    labels = fixed_labels(store, ["NT"])
    arguments = {"features": _every_gene(store), **FAST}

    tests, messages = _warned_associations(store, labels, **arguments)

    assert len(messages) == 1
    assert messages[0].startswith("1 of 2 targets have no feature of assay 'RNA'")
    assert store.perturbation.load_associations(tests).positive["target"].tolist() == [
        "RNA0"
    ]
    # Tested one at a time, the guides of GENE9 have no gene.
    singles, single_messages = _warned_associations(
        store, labels, integration="singleton", **arguments
    )
    assert single_messages == [
        "2 of 4 guides have no feature of assay 'RNA' for a positive control, for "
        "example 'CRISPR2', 'CRISPR3'. A target that is not a gene, such as an "
        "enhancer, has none; for a gene, check id_column, or the target names "
        "against the feature names."
    ]
    positive = store.perturbation.load_associations(singles).positive
    assert positive["guide"].tolist() == ["CRISPR0", "CRISPR1"]
    # Saved tests warn again, alike.
    assert _warned_associations(store, labels, **arguments) == (tests, messages)
    assert _warned_associations(
        store, labels, integration="singleton", **arguments
    ) == (singles, single_messages)


def test_associations_warn_about_miscalibration(tmp_path) -> None:
    rng = np.random.default_rng(12)
    rna = OPERATION_RNA.astype(np.int64)
    # The cells of control guide 4 express RNA7 to RNA16 four times as much and
    # RNA17 to RNA26 a quarter as much, which their totals cannot absorb.
    guide_four = np.flatnonzero(OPERATION_GUIDES[:, 4] > 0)
    rna[np.ix_(guide_four, np.arange(7, 17))] *= 4
    lowered = np.ix_(guide_four, np.arange(17, 27))
    rna[lowered] = rng.binomial(rna[lowered], 0.25)
    store = _operation_store(tmp_path, rna=rna.astype(np.uint16))
    labels = fixed_labels(store, ["NT"])
    arguments = {"features": _every_gene(store), "n_calibration_pairs": 60, **FAST}

    tests, messages = _warned_associations(store, labels, **arguments)

    result = store.perturbation.load_associations(tests)
    assert result.calibration_grade in ("moderate", "severe")
    assert result.n_false_discoveries >= 7
    assert messages == [
        f"The calibration check found {result.n_false_discoveries} false "
        f"discoveries among {len(result.calibration)} negative-control pairs, a "
        f"{result.calibration_grade} miscalibration; check the covariates and the "
        "control guides before trusting the discoveries"
    ]
    # Saved tests warn again, alike.
    assert _warned_associations(store, labels, **arguments) == (tests, messages)


def _batch_screen(path: Path, batches: dict[int, str]) -> DataStore:
    """A null screen whose cells' batches follow their guides.

    Guides 0 and 1 target RNA0, guides 2 and 3 target RNA1, and guides 4 to
    7 are non-targeting, each the one guide of 60 or 75 cells. ``batches``
    gives the batch of some guides' cells, and the other cells are batch a.
    No gene responds to any guide, and every cell holds ambient guide UMIs
    below the floor of the calls.
    """
    rng = np.random.default_rng(11)
    guide_of = np.r_[np.repeat(np.arange(4), 60), np.repeat(np.arange(4, 8), 75)]
    n_cells = len(guide_of)
    library = rng.lognormal(np.log(2000), 0.4, n_cells)
    mu = np.exp(rng.normal(-5.0, 0.7, 20)[None, :] + np.log(library)[:, None])
    rna = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(np.uint16)
    guides = rng.poisson(0.5, size=(n_cells, 8)).astype(np.uint16)
    guides[np.arange(n_cells), guide_of] = 20
    batch = np.asarray([batches.get(int(guide), "a") for guide in guide_of])
    return write_screen(
        path,
        rna,
        guides,
        targets=GUIDE_TARGETS,
        target_ids=GUIDE_GENE_IDS,
        cell_columns={"batch": batch.astype(object)},
        nthreads=2,
    )


@pytest.mark.parametrize(
    ("moi", "determined"),
    [
        # At low MOI a calibration group is compared with the other control
        # cells, of batch c or of batch a alone.
        ("low", {"CRISPR4+CRISPR5", "CRISPR6+CRISPR7"}),
        # At high MOI with all other cells, which batch a shares with RNA0.
        ("high", {"CRISPR4+CRISPR5"}),
    ],
)
def test_pairs_whose_cells_the_covariates_determine_have_no_statistic(
    tmp_path, moi, determined
) -> None:
    # RNA1 was sequenced alone in batch b, and guides 4 and 5 make batch c.
    store = _batch_screen(tmp_path / "store.zarr", {2: "b", 3: "b", 4: "c", 5: "c"})
    labels = fixed_labels(store, ["NT"])
    arguments = {
        "features": store.features.snapshot(feature_indexes=np.arange(20)),
        "sample_column": "batch",
        "moi": moi,
        "n_calibration_pairs": 1000,
        **FAST,
    }
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]))
    try:
        tests, messages = _warned_associations(store, labels, **arguments)
    finally:
        logger.remove(sink)

    # Within each pair of RNA1, batch b's indicator is RNA1's, so its null
    # pairs had statistics of rounding noise, near 1e7, and p-values of 0,
    # and fold changes of about 0, which the fit gives such cells by
    # construction.
    result = store.perturbation.load_associations(tests)
    discovery = result.discovery
    estimates = ["z", "p_value", "log2_fold_change", "se_log2_fold_change"]
    rna1 = discovery[discovery["target"] == "RNA1"]
    assert rna1["pass_qc"].all() and len(rna1) == 20
    assert rna1[estimates].isna().all(axis=None)
    assert (rna1["stage"] == 0).all() and not rna1["significant"].any()
    rna0 = discovery[discovery["target"] == "RNA0"]
    assert rna0[estimates].notna().all(axis=None) and (rna0["stage"] > 0).all()
    positive = result.positive.set_index("target")
    assert positive.loc["RNA1", estimates].isna().all()
    assert positive.loc["RNA1", "stage"] == 0
    # Only RNA0's pairs count as tested.
    tested = [line for line in logged if line.startswith("Tested ")]
    assert len(tested) == 1
    assert tested[0].startswith(
        "Tested 20 of 40 discovery pairs, and 20 more passed QC without a statistic: "
    )
    # The calibration groups that a batch determines give no false discovery.
    calibration = result.calibration
    without = calibration["control_guides"].isin(determined).to_numpy()
    assert set(calibration.loc[without, "control_guides"]) == determined
    assert calibration["pass_qc"].all() and (~without).any()
    assert calibration.loc[without, estimates].isna().all(axis=None)
    assert (calibration.loc[without, "stage"] == 0).all()
    assert not calibration.loc[without, "significant"].any()
    assert calibration.loc[~without, estimates].notna().all(axis=None)
    assert result.n_false_discoveries == int(calibration["significant"].sum())
    names = sorted(set(calibration.loc[without, "target"]))
    named = [message for message in messages if "no statistic" in message]
    holding = (
        "samples, or of a binary covariate, that hold no control cell"
        if moi == "low"
        else "samples or of a binary covariate"
    )
    assert named == [
        "The pairs of 1 target pass QC but have no statistic, so they have no "
        "p-value or fold change and are not significant: 'RNA1'. The covariates "
        "determine which of each pair's cells carry the target, as they do when "
        f"its cells are exactly the cells of {holding}; test such a target "
        "without that sample column or covariate.",
        f"The pairs of {len(names)} calibration group"
        + ("s" if len(names) > 1 else "")
        + " pass QC but have no statistic, so they have no p-value or fold change "
        "and are not counted as false discoveries: "
        + ", ".join(repr(name) for name in names)
        + ". The covariates determine which of each pair's cells carry the "
        "group's guides, as they do when those are exactly the pair's cells of "
        "samples or of a binary covariate.",
    ]
    # Saved tests warn again, alike.
    again, repeated = _warned_associations(store, labels, **arguments)
    assert again == tests
    assert [message for message in repeated if "no statistic" in message] == named


def test_a_calibration_check_without_a_statistic_does_not_run(tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group

    # Guides 4, 5 and 6 each make a batch, and guide 7 shares batch a with
    # the targets, so among the control cells each guide's cells are a
    # batch's: every calibration group of one guide is determined.
    store = _batch_screen(tmp_path / "store.zarr", {4: "c4", 5: "c5", 6: "c6"})
    labels = fixed_labels(store, ["NT"])
    arguments = {
        "features": store.features.snapshot(feature_indexes=np.arange(20)),
        "sample_column": "batch",
        "integration": "singleton",
        "n_calibration_pairs": 1000,
        **FAST,
    }

    tests, messages = _warned_associations(store, labels, **arguments)

    # The check had pairs, but none with a statistic, so it graded nothing,
    # where it was graded "excellent" from no tested pair.
    result = store.perturbation.load_associations(tests)
    assert len(result.calibration) == 80
    assert result.calibration["z"].isna().all()
    assert result.discovery["z"].notna().all()
    assert (result.calibration_grade, result.n_false_discoveries) == ("none", 0)
    status = artifact_group(store.zw, tests).attrs["calibration_status"]
    assert status == "no_calibration_statistic"
    assert "No calibration pair has a statistic, so no calibration check was run" in (
        messages
    )
    # Saved tests warn again, alike.
    assert _warned_associations(store, labels, **arguments) == (tests, messages)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"labels": "x"}, TypeError, "labels must be an ArtifactRef"),
        ({"labels": "features"}, ValueError, "perturbation_labels artifact"),
        ({"from_assay": "ADT"}, ValueError, "from_assay must name an assay"),
        ({"pairs": "cis"}, ValueError, "'trans' or a DataFrame"),
        ({"features": None}, ValueError, "pass features"),
        (
            {"pairs": pd.DataFrame({"target": [], "feature": []})},
            ValueError,
            "goes with",
        ),
        ({"features": "labels"}, ValueError, "feature_selection artifact"),
        ({"positive_controls": "genes"}, ValueError, "positive_controls must be"),
        ({"id_column": 3}, TypeError, "id_column must be"),
        ({"id_column": "missing"}, ValueError, "is not a feature column"),
        ({"integration": "pooled"}, ValueError, "integration must be"),
        ({"side": "up"}, ValueError, "side must be"),
        ({"approximation": "normal"}, ValueError, "approximation must be"),
        ({"n_resamples": "499"}, TypeError, "sequence of three integers"),
        ({"n_resamples": [499, 4999]}, TypeError, "sequence of three integers"),
        ({"n_resamples": 499}, TypeError, "sequence of three integers"),
        ({"n_resamples": (0, 4999, 24999)}, ValueError, "n_resamples\\[0\\]"),
        ({"n_resamples": (499, 1, 24999)}, ValueError, "n_resamples\\[1\\]"),
        ({"n_resamples": (499, 4999, 0)}, ValueError, "n_resamples\\[2\\]"),
        ({"stage_threshold": 2.0}, ValueError, "stage_threshold must be"),
        (
            {"n_resamples": (50, 4999, 24999)},
            ValueError,
            "smallest first-round p-value is 0.0392, above stage_threshold=0.02",
        ),
        ({"min_nonzero": 0}, ValueError, "min_nonzero"),
        ({"n_calibration_pairs": -1}, ValueError, "n_calibration_pairs"),
        ({"alpha": 1.0}, ValueError, "alpha must be"),
        ({"seed": -1}, ValueError, "seed"),
        ({"sample_column": 4}, TypeError, "sample_column must be"),
        ({"sample_column": "batch"}, ValueError, "sample_column 'batch'"),
        ({"covariates": "mito"}, TypeError, "covariates must be"),
        (
            {"covariates": 3},
            TypeError,
            "^covariates must be a sequence of column names or None$",
        ),
        # The pipeline recipe refuses a missing column with the same error.
        (
            {"covariates": ["depth"]},
            ValueError,
            "^covariate 'depth' not found in cell metadata$",
        ),
        ({"covariates": ["label"]}, ValueError, "must be numeric"),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache must be"),
        (
            {
                "pairs": pd.DataFrame({"target": [], "feature": []}),
                "features": None,
                "positive_controls": None,
            },
            ValueError,
            "no pair to test",
        ),
    ],
)
def test_associations_refuse_arguments_they_cannot_use(
    shared, arguments, error, message
) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    settings: dict[str, object] = {"labels": labels, "features": features}
    settings.update(arguments)
    if isinstance(settings["labels"], str) and settings["labels"] == "features":
        settings["labels"] = features
    if isinstance(settings.get("features"), str):
        settings["features"] = labels

    with pytest.raises(error, match=message):
        store.perturbation.associations(**settings)


def test_association_records_leave_their_tables_out_of_their_repr() -> None:
    table = pd.DataFrame({"target": ["GENE_SHOWN_NOWHERE"], "feature": ["RNA0"]})
    record = PerturbationAssociations(
        discovery=table,
        calibration=table,
        positive=table,
        alpha=0.1,
        side="both",
        integration="union",
        moi="low",
        resampling="permutation",
        from_assay="RNA",
        n_false_discoveries=0,
        calibration_grade="none",
    )

    text = repr(record)

    assert "GENE_SHOWN_NOWHERE" not in text
    assert text == (
        "PerturbationAssociations(alpha=0.1, side='both', integration='union', "
        "moi='low', resampling='permutation', from_assay='RNA', "
        "n_false_discoveries=0, calibration_grade='none')"
    )


def test_association_settings_check_the_sample_column() -> None:
    # The pipeline recipe runs the same check on its settings.
    for column in (4, "", ["lane"]):
        with pytest.raises(
            TypeError, match="^sample_column must be a non-empty string or None$"
        ):
            _settings(sample_column=column)
    # Whether the column exists is checked against the store.
    assert _settings(sample_column="lane") == _settings()


def test_association_settings_check_the_covariates() -> None:
    # The pipeline recipe runs the same check on its settings, so a stage
    # setting such as {"covariates": "depth"} is refused before a run starts.
    for covariates in ("depth", 3, [3], ["depth", None], b"depth"):
        with pytest.raises(
            TypeError, match="^covariates must be a sequence of column names or None$"
        ):
            _settings(covariates=covariates)
    # Whether each column exists and holds numbers is checked against the store.
    assert _settings(covariates=["depth", "mito"]) == _settings()
    assert associations_module.covariate_columns(None) == ()
    # Any iterable of names, read once, such as the columns of a table.
    names = (name for name in ["depth", "mito"])
    assert associations_module.covariate_columns(names) == ("depth", "mito")
    assert associations_module.covariate_columns(pd.Index(["depth"])) == ("depth",)


def test_associations_take_the_resamples_as_any_sequence(shared) -> None:
    store, labels = shared.store, shared.labels
    features = _every_gene(store)
    tests = store.perturbation.associations(labels, features=features, **FAST)

    # A list, as a JSON configuration gives it, names the same tests.
    listed = store.perturbation.associations(
        labels, features=features, n_resamples=list(FAST["n_resamples"])
    )

    assert listed == tests


def test_trans_features_must_belong_to_the_tested_assay(shared) -> None:
    store, labels = shared.store, shared.labels
    guides = store.features.snapshot(feature_indexes=np.arange(3), from_assay="CRISPR")

    with pytest.raises(ValueError, match="feature selection of assay 'RNA'"):
        store.perturbation.associations(labels, features=guides)


def test_loading_associations_refuses_other_artifacts(shared) -> None:
    store, labels = shared.store, shared.labels

    with pytest.raises(TypeError, match="tests must be an ArtifactRef"):
        store.perturbation.load_associations("tests")
    with pytest.raises(ValueError, match="perturbation_associations artifact"):
        store.perturbation.load_associations(labels)
