"""Harmony2's numerical steps checked against direct dense calculations.

Every reference here is written from the equations of the two papers (the
normal equations of each cluster's ridge regression, the closed-form soft
assignment, the clustering objective, the cosine centroids and k-means++
sampling), never by calling the code under test.
"""

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import cytearc.embeddings.harmony.clustering as clustering_module
import cytearc.embeddings.harmony.correction as correction_module
from cytearc.embeddings.harmony import fit_harmony
from cytearc.embeddings.harmony.clustering import (
    ClusteringWorkspace,
    cluster_round,
    clustering_pass,
    cosine_centroids,
    initial_assignments,
    update_inverse_norms,
)
from cytearc.embeddings.harmony.correction import (
    FIXED_PENALTY_MASS_RATIO_LIMIT,
    RegressionStatistics,
    _require_accurate_fixed_penalties,
    _solve_by_elimination,
    active_levels,
    ridge_penalties,
    solve_cluster_regressions,
)
from cytearc.embeddings.harmony.design import BatchDesign, encode_batch_design
from cytearc.embeddings.harmony.fit import _require_finite_centroids
from cytearc.embeddings.harmony.kernels import assign_cells
from cytearc.embeddings.harmony.parameters import resolve_harmony_parameters
from cytearc.embeddings.harmony.seeding import kmeans_plus_plus_seeds, next_seed

# Small fits with few rounds often stop before the objective settles.
pytestmark = pytest.mark.filterwarnings("ignore:Harmony stopped before convergence")


# Dense references


def _indicators(codes: np.ndarray, n_levels: int) -> np.ndarray:
    """The ``(levels, cells)`` one-hot batch design of global level codes."""
    indicators = np.zeros((n_levels, codes.shape[-1]))
    for row in np.atleast_2d(codes):
        indicators[row, np.arange(row.shape[0])] = 1.0
    return indicators


def _dense_statistics(
    design: BatchDesign,
    indicators: np.ndarray,
    assignments: np.ndarray,
    coordinates: np.ndarray,
) -> RegressionStatistics:
    """Sufficient statistics summed directly over the cells."""
    combination_mass = np.empty((design.n_combinations, assignments.shape[1]))
    for combination, levels in enumerate(design.combination_levels):
        members = np.all(indicators[levels] == 1.0, axis=0)
        combination_mass[combination] = assignments[members].sum(axis=0)
    return RegressionStatistics(
        combination_mass=combination_mass,
        level_sums=np.einsum("gi,ik,id->gkd", indicators, assignments, coordinates),
        intercept_sums=assignments.T @ coordinates,
        level_mass=indicators @ assignments,
        totals=assignments.sum(axis=0),
    )


def _dense_coefficients(
    indicators: np.ndarray,
    weights: np.ndarray,
    coordinates: np.ndarray,
    penalties: np.ndarray,
    kept: np.ndarray,
) -> np.ndarray:
    """Solve ``(Phi* diag(R_k) Phi*^T + diag(0, lambda)) W = Phi* diag(R_k) Z``.

    Only the intercept and the ``kept`` levels enter the system; the returned
    rows are the coefficients of the kept levels.
    """
    design = np.vstack([np.ones((1, indicators.shape[1])), indicators[kept]])
    system = (design * weights) @ design.T
    system[1:, 1:] += np.diag(penalties[kept])
    responses = (design * weights) @ coordinates
    return np.linalg.solve(system, responses)[1:]


def _global_codes(columns: list[np.ndarray]) -> np.ndarray:
    """Global level codes of integer columns whose levels are 0, 1, ... in each column."""
    offsets = np.cumsum([0] + [int(column.max()) + 1 for column in columns])
    return np.stack([column + offset for column, offset in zip(columns, offsets)])


def _exact_coefficients(
    codes: np.ndarray,
    weights: np.ndarray,
    coordinates: np.ndarray,
    penalties: np.ndarray,
    kept: np.ndarray,
) -> np.ndarray:
    """Solve one cluster's normal equations exactly over the rationals.

    The equations are summed from the cells with Fraction arithmetic, so every
    identity of the design holds exactly (the intercept equals the sum of each
    column's indicators), and Gauss-Jordan elimination solves them without
    rounding. Only the intercept and the ``kept`` levels enter the system; the
    returned rows are the coefficients of the kept levels, rounded to float64.
    """
    dims = coordinates.shape[1]
    position = {int(level): 1 + index for index, level in enumerate(kept)}
    size = 1 + len(kept)
    # Cells with the same levels share a design row, so their sums are formed first.
    groups: dict[tuple[int, ...], list[Fraction]] = {}
    for cell in range(coordinates.shape[0]):
        weight = Fraction(float(weights[cell]))
        sums = groups.setdefault(
            tuple(int(level) for level in codes[:, cell]), [Fraction(0)] * (1 + dims)
        )
        sums[0] += weight
        for dim in range(dims):
            sums[1 + dim] += weight * Fraction(float(coordinates[cell, dim]))
    system = [[Fraction(0)] * size for _ in range(size)]
    rhs = [[Fraction(0)] * dims for _ in range(size)]
    for levels, sums in groups.items():
        rows = [0] + [position[level] for level in levels if level in position]
        for row in rows:
            for column in rows:
                system[row][column] += sums[0]
            for dim in range(dims):
                rhs[row][dim] += sums[1 + dim]
    for level in kept:
        system[position[int(level)]][position[int(level)]] += Fraction(
            float(penalties[level])
        )
    for pivot in range(size):
        head = system[pivot][pivot]
        system[pivot] = [value / head for value in system[pivot]]
        rhs[pivot] = [value / head for value in rhs[pivot]]
        for row in range(size):
            factor = system[row][pivot]
            if row != pivot and factor != 0:
                system[row] = [
                    a - factor * b for a, b in zip(system[row], system[pivot])
                ]
                rhs[row] = [a - factor * b for a, b in zip(rhs[row], rhs[pivot])]
    return np.array([[float(value) for value in row] for row in rhs[1:]])


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _cosine_centroids(corrected: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """``y_k = S_k / ||S_k||`` with ``S_k = sum_i R_ik z_i / ||z_i||``, zero for a zero sum."""
    sums = _unit_rows(corrected).T @ assignments
    norms = np.linalg.norm(sums, axis=0)
    return np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 0)


def _single_column_case(seed: int, n_cells: int = 300, n_levels: int = 7):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_levels, size=n_cells)
    labels[:n_levels] = np.arange(n_levels)
    design = encode_batch_design(pd.DataFrame({"batch": labels}))
    # A numeric column orders its levels numerically, so level g is label g.
    assert design.levels == (tuple(str(level) for level in range(n_levels)),)
    return rng, labels, design, _indicators(labels, n_levels)


# Ridge regression


def test_arrowhead_solve_matches_dense_solve() -> None:
    rng, _labels, design, indicators = _single_column_case(1)
    n_cells, dims, n_clusters = 300, 4, 3
    # Positive weights; the regression does not need rows that sum to 1.
    assignments = rng.uniform(0.05, 1.0, size=(n_cells, n_clusters))
    coordinates = rng.normal(size=(n_cells, dims)) + rng.normal(size=dims)
    statistics = _dense_statistics(design, indicators, assignments, coordinates)
    proportions = indicators.sum(axis=1) / n_cells
    penalties = 0.2 * proportions[:, None] * assignments.sum(axis=0)[None, :]
    active = np.ones((design.n_levels, n_clusters), dtype=bool)

    coefficients = solve_cluster_regressions(statistics, penalties, active, design)

    every_level = np.arange(design.n_levels)
    for cluster in range(n_clusters):
        expected = _dense_coefficients(
            indicators,
            assignments[:, cluster],
            coordinates,
            penalties[:, cluster],
            every_level,
        )
        np.testing.assert_allclose(
            coefficients[:, cluster, :], expected, rtol=1e-10, atol=1e-12
        )


def test_general_solve_equals_the_arrowhead_with_one_column() -> None:
    rng, _labels, design, indicators = _single_column_case(2)
    assignments = rng.uniform(0.05, 1.0, size=(300, 3))
    coordinates = rng.normal(size=(300, 4))
    penalties = rng.uniform(0.1, 2.0, size=(design.n_levels, 3))
    active = np.ones((design.n_levels, 3), dtype=bool)
    active[3, 1] = False

    arrowhead = solve_cluster_regressions(
        _dense_statistics(design, indicators, assignments, coordinates),
        penalties,
        active,
        design,
    )
    general = _dense_statistics(design, indicators, assignments, coordinates)
    _solve_by_elimination(general, penalties, active, design)

    np.testing.assert_allclose(general.level_sums, arrowhead, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("layout", ["nested", "crossed"])
def test_block_elimination_matches_dense_solve(layout: str) -> None:
    rng = np.random.default_rng(3)
    n_cells, dims, n_clusters = 500, 3, 3
    first = rng.integers(0, 12, size=n_cells)
    first[:12] = np.arange(12)
    if layout == "nested":
        # Each level of the largest column lies within one level of the second.
        second = first % 4
    else:
        second = rng.integers(0, 4, size=n_cells)
        second[:4] = np.arange(4)
    third = rng.integers(0, 3, size=n_cells)
    third[:3] = np.arange(3)
    metadata = pd.DataFrame({"sample": first, "dataset": second, "lane": third})
    design = encode_batch_design(metadata)
    assert design.column_sizes == (12, 4, 3)
    codes = np.stack([first, 12 + second, 16 + third])
    indicators = _indicators(codes, design.n_levels)
    assignments = rng.uniform(0.05, 1.0, size=(n_cells, n_clusters))
    # Level 5 of the largest column has almost no mass in cluster 1.
    assignments[first == 5, 1] *= 1e-8
    coordinates = rng.normal(size=(n_cells, dims))
    statistics = _dense_statistics(design, indicators, assignments, coordinates)
    level_mass = indicators @ assignments
    level_counts = indicators.sum(axis=1)
    penalties = 0.2 * (level_counts / n_cells)[:, None] * assignments.sum(axis=0)
    active = (level_mass > 0) & (level_mass >= 1e-5 * level_counts[:, None])
    assert not active[5, 1] and active.sum() == design.n_levels * n_clusters - 1
    np.testing.assert_array_equal(active_levels(statistics, design, 1e-5), active)

    coefficients = solve_cluster_regressions(statistics, penalties, active, design)

    for cluster in range(n_clusters):
        kept = np.flatnonzero(active[:, cluster])
        expected = _dense_coefficients(
            indicators,
            assignments[:, cluster],
            coordinates,
            penalties[:, cluster],
            kept,
        )
        np.testing.assert_allclose(
            coefficients[kept, cluster, :], expected, rtol=1e-10, atol=1e-12
        )
        dropped = np.flatnonzero(~active[:, cluster])
        assert np.all(coefficients[dropped, cluster, :] == 0.0)


def test_pruned_levels_match_the_restricted_dense_solve() -> None:
    rng, labels, design, indicators = _single_column_case(4)
    n_cells, dims, n_clusters = 300, 4, 2
    assignments = rng.uniform(0.05, 1.0, size=(n_cells, n_clusters))
    level_sizes = np.bincount(labels)
    # Level 2 keeps a mass of 1e-4 in every cluster, and level 5 a mass of
    # twice 1e-5 N_5. Harmony2 compares O_kg with the threshold times the
    # level's cell count N_g: level 2 falls between 1e-5 and 1e-5 N_2, so it is
    # pruned although its mass exceeds the bare threshold, and level 5 stays.
    for level, mass in ((2, 1e-4), (5, 2e-5 * level_sizes[5])):
        members = labels == level
        assignments[members] *= mass / assignments[members].sum(axis=0)
    coordinates = rng.normal(size=(n_cells, dims))
    statistics = _dense_statistics(design, indicators, assignments, coordinates)
    observed = statistics.level_mass
    assert np.all((observed[2] > 1e-5) & (observed[2] < 1e-5 * level_sizes[2]))
    assert np.all(observed[5] >= 1e-5 * level_sizes[5])
    penalties = np.full((design.n_levels, n_clusters), 0.7)

    active = active_levels(statistics, design, 1e-5)
    coefficients = solve_cluster_regressions(statistics, penalties, active, design)

    assert not active[2].any() and np.delete(active, 2, axis=0).all()
    assert np.all(coefficients[2] == 0.0)
    kept = np.delete(np.arange(design.n_levels), 2)
    for cluster in range(n_clusters):
        expected = _dense_coefficients(
            indicators,
            assignments[:, cluster],
            coordinates,
            penalties[:, cluster],
            kept,
        )
        np.testing.assert_allclose(coefficients[kept, cluster, :], expected, rtol=1e-10)


def test_a_level_without_mass_has_a_zero_dense_coefficient() -> None:
    rng, labels, design, indicators = _single_column_case(5)
    n_cells, dims = 300, 4
    weights = rng.uniform(0.05, 1.0, size=n_cells)
    weights[labels == 4] = 0.0
    coordinates = rng.normal(size=(n_cells, dims))
    penalties = np.full(design.n_levels, 0.7)
    every_level = np.arange(design.n_levels)
    kept = np.delete(every_level, 4)

    full = _dense_coefficients(indicators, weights, coordinates, penalties, every_level)
    restricted = _dense_coefficients(indicators, weights, coordinates, penalties, kept)

    # Row 4 of the full system reads 0.7 w_4 = 0, so leaving the level out is exact.
    assert np.max(np.abs(full[4])) < 1e-12
    np.testing.assert_allclose(np.delete(full, 4, axis=0), restricted, rtol=1e-10)
    statistics = _dense_statistics(design, indicators, weights[:, None], coordinates)
    active = active_levels(statistics, design, 1e-5)
    assert active[:, 0].tolist() == [level != 4 for level in range(design.n_levels)]
    coefficients = solve_cluster_regressions(
        statistics, penalties[:, None], active, design
    )
    np.testing.assert_allclose(coefficients[:, 0, :], full, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("penalty", [1e-6, 1e-9, 1e-12, 1e-30])
def test_one_column_solve_is_exact_for_tiny_fixed_penalties(penalty: float) -> None:
    rng, labels, design, indicators = _single_column_case(21, n_cells=240, n_levels=6)
    n_cells, dims, n_clusters = 240, 3, 2
    assignments = rng.uniform(0.05, 1.0, size=(n_cells, n_clusters))
    # Level 2 is pruned in cluster 0, so its cells enter only the intercept there.
    assignments[labels == 2, 0] *= 1e-9
    coordinates = (
        rng.normal(size=(n_cells, dims)) + 5.0 + rng.normal(size=(6, dims))[labels]
    )
    statistics = _dense_statistics(design, indicators, assignments, coordinates)
    penalties = np.full((design.n_levels, n_clusters), penalty)
    active = active_levels(statistics, design, 1e-5)
    assert not active[2, 0] and active[2, 1]

    coefficients = solve_cluster_regressions(statistics, penalties, active, design)

    # A penalty far below the level masses leaves the coefficients of the size
    # of the batch offsets, and the solve must not lose them to cancellation.
    for cluster in range(n_clusters):
        kept = np.flatnonzero(active[:, cluster])
        exact = _exact_coefficients(
            labels[None, :],
            assignments[:, cluster],
            coordinates,
            penalties[:, cluster],
            kept,
        )
        error = np.max(np.abs(coefficients[kept, cluster, :] - exact))
        assert error <= 1e-12 * np.max(np.abs(exact))
        assert np.all(coefficients[~active[:, cluster], cluster, :] == 0.0)


def test_one_column_fit_with_a_tiny_fixed_penalty_keeps_the_input_scale() -> None:
    values, metadata, _batch = _three_batches(seed=11)

    result = fit_harmony(
        values, metadata, n_clusters=3, ridge_penalty=1e-30, max_iter_harmony=3
    )

    assert np.all(np.isfinite(result.corrected))
    assert np.max(np.abs(result.corrected)) <= 2.0 * np.max(np.abs(values))


@pytest.mark.parametrize("n_columns", [2, 3])
@pytest.mark.parametrize("layout", ["nested", "crossed"])
def test_several_column_solve_is_accurate_down_to_the_penalty_limit(
    n_columns: int, layout: str
) -> None:
    rng = np.random.default_rng(22)
    n_cells, dims, n_clusters = 240, 2, 2
    largest = rng.integers(0, 8, size=n_cells)
    largest[:8] = np.arange(8)
    if layout == "nested":
        # Each level of the largest column lies within one level of each other.
        others = [largest % 4, largest % 2]
    else:
        others = []
        for size in (4, 3):
            column = rng.integers(0, size, size=n_cells)
            column[:size] = np.arange(size)
            others.append(column)
    columns = [largest, *others[: n_columns - 1]]
    design = encode_batch_design(
        pd.DataFrame({f"column{index}": column for index, column in enumerate(columns)})
    )
    codes = _global_codes(columns)
    np.testing.assert_array_equal(design.level_codes, codes)
    indicators = _indicators(codes, design.n_levels)
    assignments = rng.uniform(0.05, 1.0, size=(n_cells, n_clusters))
    # Level 3 of the eliminated column is pruned in cluster 0.
    assignments[largest == 3, 0] *= 1e-9
    coordinates = rng.normal(size=(n_cells, dims)) + 5.0
    for column_codes in codes:
        coordinates += rng.normal(size=(design.n_levels, dims))[column_codes]
    active = active_levels(
        _dense_statistics(design, indicators, assignments, coordinates), design, 1e-5
    )
    assert not active[3, 0] and active[:, 1].all()
    masses = assignments.sum(axis=0)
    # The fit refuses a fixed penalty below 1e-6 of a cluster's mass with several
    # columns; at that limit the solve still keeps eight digits.
    for penalty in (1.0, 1e-2, float(masses.max()) / FIXED_PENALTY_MASS_RATIO_LIMIT):
        penalties = np.full((design.n_levels, n_clusters), penalty)
        statistics = _dense_statistics(design, indicators, assignments, coordinates)
        coefficients = solve_cluster_regressions(statistics, penalties, active, design)
        for cluster in range(n_clusters):
            kept = np.flatnonzero(active[:, cluster])
            exact = _exact_coefficients(
                codes,
                assignments[:, cluster],
                coordinates,
                penalties[:, cluster],
                kept,
            )
            error = np.max(np.abs(coefficients[kept, cluster, :] - exact))
            assert error <= 1e-8 * np.max(np.abs(exact)), (penalty, cluster)


def test_fixed_penalties_too_small_for_several_columns_are_refused() -> None:
    values, metadata, _batch = _three_batches(seed=12)
    lanes = np.random.default_rng(12).integers(0, 2, size=len(metadata))
    lanes[:2] = [0, 1]
    two_columns = metadata.assign(lane=lanes)

    with pytest.raises(
        ValueError,
        match=(
            r"ridge_penalty is too small for an accurate regression with several "
            r"batch columns.*leave ridge_penalty out to use the dynamic penalty"
        ),
    ):
        fit_harmony(values, two_columns, n_clusters=3, ridge_penalty=1e-9)

    # One column is solved accurately at any positive penalty, and two columns
    # at penalties of at least 1e-6 times every cluster's mass, at most 120 here.
    fit_harmony(values, metadata, n_clusters=3, ridge_penalty=1e-9)
    fit_harmony(values, two_columns, n_clusters=3, ridge_penalty=2e-4)
    # The dynamic penalty grows with each cluster's mass.
    fit_harmony(values, two_columns, n_clusters=3)


def test_the_fixed_penalty_limit_is_relative_to_each_cluster_mass() -> None:
    def statistics(totals: list[float]) -> RegressionStatistics:
        return RegressionStatistics(
            combination_mass=np.zeros((1, 2)),
            level_sums=np.zeros((2, 2, 1)),
            intercept_sums=np.zeros((2, 1)),
            level_mass=np.zeros((2, 2)),
            totals=np.asarray(totals),
        )

    penalties = np.array([[1e-4, 1e-6], [1e-3, 1e-3]])
    # Cluster 1's penalty of 1e-6 belongs to a pruned level and does not count.
    active = np.array([[True, False], [True, True]])

    _require_accurate_fixed_penalties(statistics([100.0, 1000.0]), penalties, active)
    with pytest.raises(
        ValueError,
        match=(
            r"cluster 1 holds an assignment mass of 1001, more than 1e6 times its "
            r"smallest penalty \(0\.001\)\. Pass a ridge_penalty of at least 0\.0011,"
        ),
    ):
        _require_accurate_fixed_penalties(
            statistics([100.0, 1001.0]), penalties, active
        )


def _three_batches(n_cells: int = 120, seed: int = 6):
    rng = np.random.default_rng(seed)
    batch = np.repeat(np.arange(3), n_cells // 3)
    values = rng.normal(size=(4, n_cells)) + rng.normal(size=(3, 4))[batch].T
    return values, pd.DataFrame({"batch": [f"b{level}" for level in batch]}), batch


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_dynamic_penalty_is_alpha_times_expected_mass(alpha: float | None) -> None:
    values, metadata, batch = _three_batches()
    n_cells = values.shape[1]

    result = fit_harmony(
        values, metadata, n_clusters=3, ridge_alpha=alpha, max_iter_harmony=2
    )

    scale = 0.2 if alpha is None else alpha
    level_counts = np.bincount(batch).astype(float)
    # The last correction used the final assignments: lambda_kb = alpha E_kb.
    expected = (
        scale
        * result.assignments.sum(axis=1)[:, None]
        * level_counts[None, :]
        / n_cells
    )
    np.testing.assert_allclose(result.ridge_penalty, expected, rtol=1e-12)
    assert result.parameters["ridge"] == "dynamic"
    assert result.parameters["ridgeAlpha"] == scale
    assert result.parameters["ridgePenalty"] is None


def test_fixed_penalty_expands_per_column_and_level() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(3, 60))
    metadata = pd.DataFrame(
        {
            "donor": np.tile(["d1", "d2"], 30),
            "lane": np.repeat(["x", "y", "z"], 20),
        }
    )
    for ridge_penalty, expected in (
        ([1.0, 2.0], [1.0, 1.0, 2.0, 2.0, 2.0]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        (0.5, [0.5] * 5),
    ):
        result = fit_harmony(
            values,
            metadata,
            n_clusters=2,
            ridge_penalty=ridge_penalty,
            max_iter_harmony=1,
        )
        assert result.ridge_penalty.shape == (2, 5)
        for row in result.ridge_penalty:
            assert row.tolist() == expected
        assert result.parameters["ridge"] == "fixed"
        assert result.parameters["ridgePenalty"] == expected
        assert result.parameters["ridgeAlpha"] is None
        assert result.active_levels.shape == (2, 5)
        assert result.active_levels.dtype == np.bool_


# Soft assignments


def _assignment_case(seed: int, n_columns: int):
    rng = np.random.default_rng(seed)
    n_cells, dims, n_clusters, levels_per_column = 200, 5, 6, 4
    n_levels = n_columns * levels_per_column
    coordinates = rng.normal(size=(n_cells, dims)) * rng.uniform(
        0.5, 3.0, size=(n_cells, 1)
    )
    centroids = rng.normal(size=(dims, n_clusters))
    centroids /= np.linalg.norm(centroids, axis=0)
    sigma = rng.uniform(0.05, 1.0, size=n_clusters)
    theta = rng.uniform(0.0, 4.0, size=n_levels)
    observed = rng.uniform(0.0, 50.0, size=(n_levels, n_clusters))
    totals = rng.uniform(1.0, 100.0, size=n_clusters)
    proportions = np.concatenate(
        [rng.dirichlet(np.ones(levels_per_column)) for _ in range(n_columns)]
    )
    expected_mass = proportions[:, None] * totals[None, :]
    codes = np.stack(
        [
            column * levels_per_column
            + rng.integers(0, levels_per_column, size=n_cells)
            for column in range(n_columns)
        ]
    ).astype(np.int32)
    return (
        coordinates,
        centroids,
        sigma,
        theta,
        observed,
        expected_mass,
        codes,
    )


def _run_assignment_kernel(coordinates, centroids, sigma, penalty, codes):
    n_cells = coordinates.shape[0]
    norms = np.linalg.norm(coordinates, axis=1)
    similarity = coordinates @ centroids
    assignments = np.full((n_cells, centroids.shape[1]), np.nan)
    entropies = np.empty(n_cells)
    assign_cells(
        similarity,
        np.arange(n_cells, dtype=np.int64),
        1.0 / norms,
        2.0 / sigma,
        penalty,
        codes,
        assignments,
        entropies,
    )
    return assignments, entropies


@pytest.mark.parametrize("n_columns", [1, 2])
def test_assignment_update_is_the_minimiser_of_the_cell_objective(
    n_columns: int,
) -> None:
    coordinates, centroids, sigma, theta, observed, expected_mass, codes = (
        _assignment_case(8 + n_columns, n_columns)
    )
    # The diversity penalty of Harmony2, theta log((E + 1) / (O + E + 1)) <= 0.
    penalty = -theta[:, None] * np.log1p(observed / (expected_mass + 1.0))

    assignments, entropies = _run_assignment_kernel(
        coordinates, centroids, sigma, penalty, codes
    )

    unit = _unit_rows(coordinates)
    distance = 2.0 * (1.0 - unit @ centroids) / sigma[None, :]
    factor = np.ones_like(distance)
    for column_codes in codes:
        # Harmony2's update: the product over batch columns of
        # ((E + 1) / (O + E + 1))^theta, times exp(-2 (1 - y.z) / sigma).
        ratio = (expected_mass + 1.0) / (observed + expected_mass + 1.0)
        factor *= ratio[column_codes] ** theta[column_codes][:, None]
    closed_form = factor * np.exp(-distance)
    closed_form /= closed_form.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(assignments, closed_form, rtol=1e-12)
    np.testing.assert_allclose(assignments.sum(axis=1), 1.0, atol=1e-14)
    with np.errstate(divide="ignore", invalid="ignore"):
        dense_entropy = np.where(
            assignments > 0, assignments * np.log(assignments), 0.0
        )
    np.testing.assert_allclose(entropies, dense_entropy.sum(axis=1), atol=1e-12)

    # The per-cell objective sum_k r_k a_k + sum_k r_k log r_k is minimal at
    # the returned row among points of the simplex.
    linear = distance.copy()
    for column_codes in codes:
        linear += (
            theta[column_codes][:, None]
            * np.log1p(observed / (expected_mass + 1.0))[column_codes]
        )

    def objective(cell: int, row: np.ndarray) -> float:
        positive = row > 0
        return float(row @ linear[cell] + np.sum(row[positive] * np.log(row[positive])))

    rng = np.random.default_rng(12)
    for cell in range(0, coordinates.shape[0], 10):
        best = objective(cell, assignments[cell])
        tolerance = 1e-12 * max(1.0, abs(best))
        for other in rng.dirichlet(np.ones(centroids.shape[1]), size=50):
            assert best <= objective(cell, other) + tolerance
            nearby = 0.999 * assignments[cell] + 0.001 * other
            assert best <= objective(cell, nearby) + tolerance


def test_two_covariate_penalties_multiply() -> None:
    coordinates, centroids, sigma, theta, observed, expected_mass, codes = (
        _assignment_case(20, 2)
    )
    penalty = -theta[:, None] * np.log1p(observed / (expected_mass + 1.0))

    assignments, _entropies = _run_assignment_kernel(
        coordinates, centroids, sigma, penalty, codes
    )

    unit = _unit_rows(coordinates)
    first, second = codes
    ratio = (expected_mass + 1.0) / (observed + expected_mass + 1.0)
    first_factor = ratio[first] ** theta[first][:, None]
    second_factor = ratio[second] ** theta[second][:, None]
    closed_form = (
        first_factor
        * second_factor
        * np.exp(-2.0 * (1.0 - unit @ centroids) / sigma[None, :])
    )
    closed_form /= closed_form.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(assignments, closed_form, rtol=1e-12)


# The clustering objective


def _shared_type_data():
    """Two cell types present in every one of three shifted batches."""
    rng = np.random.default_rng(0)
    batch = np.repeat(np.arange(3), 60)
    cell_type = np.tile(np.repeat(np.arange(2), 30), 3)
    type_centers = np.zeros((2, 6))
    type_centers[0, 0] = type_centers[1, 1] = 5.0
    batch_shift = rng.normal(scale=1.5, size=(3, 6))
    values = (
        rng.normal(scale=0.5, size=(180, 6))
        + type_centers[cell_type]
        + batch_shift[batch]
    )
    return values, pd.DataFrame({"batch": [f"batch_{level}" for level in batch]})


def _first_round(values: np.ndarray, metadata: pd.DataFrame, **options):
    """Run round 0's clustering as fit_harmony does, with the package's steps."""
    design = encode_batch_design(metadata)
    parameters = resolve_harmony_parameters(options, design)
    n_cells, dims = values.shape
    block_cells = max(1, math.ceil(parameters.block_size * n_cells))
    chunk = min(8192, block_cells)
    corrected = np.ascontiguousarray(values, dtype=np.float64)
    inverse_norms = np.empty(n_cells)
    update_inverse_norms(corrected, inverse_norms, np.empty(chunk))
    rng = np.random.default_rng(0)
    order = rng.permutation(n_cells).astype(np.int64)
    centroids = kmeans_plus_plus_seeds(
        corrected, inverse_norms, parameters.n_clusters, rng
    )
    workspace = ClusteringWorkspace.allocate(
        chunk, parameters.n_clusters, dims, design.n_levels
    )
    assignments = np.empty((n_cells, parameters.n_clusters))
    two_over_sigma = 2.0 / parameters.sigma
    initial_assignments(
        corrected,
        inverse_norms,
        centroids,
        two_over_sigma,
        design,
        assignments,
        workspace,
    )
    summary = cluster_round(
        corrected,
        inverse_norms,
        centroids,
        assignments,
        design,
        order,
        block_cells,
        two_over_sigma,
        parameters.theta,
        parameters.max_iter_cluster,
        parameters.epsilon_cluster,
        workspace,
        0,
    )
    return design, parameters, corrected, assignments, summary


@pytest.mark.parametrize("block_size", [0.05, 0.3, 1.0])
def test_objective_does_not_increase_without_diversity_penalty(
    block_size: float,
) -> None:
    values, metadata = _shared_type_data()

    *_state, summary = _first_round(
        values,
        metadata,
        n_clusters=4,
        theta=0.0,
        block_size=block_size,
        max_iter_cluster=15,
        epsilon_cluster=0.0,
    )

    # With theta = 0 every block update and every centroid update minimises
    # the objective exactly, so it never rises.
    objectives = summary.objectives
    assert len(objectives) >= 2
    for previous, current in zip(objectives, objectives[1:]):
        assert current <= previous + 1e-9 * abs(previous)


@pytest.mark.parametrize("theta", [0.0, 2.0])
def test_objective_matches_the_dense_formula(theta: float) -> None:
    values, metadata = _shared_type_data()

    design, parameters, corrected, assignments, summary = _first_round(
        values,
        metadata,
        n_clusters=4,
        theta=theta,
        sigma=[0.1, 0.2, 0.15, 0.3],
        max_iter_cluster=3,
        epsilon_cluster=0.0,
    )

    # J = sum_ik R_ik [2 (1 - y_k . z_i) / sigma_k + log R_ik]
    #     + sum_kb theta_b O_kb log((O_kb + E_kb + 1) / (E_kb + 1)).
    centroids = summary.centroids
    np.testing.assert_allclose(
        centroids, _cosine_centroids(corrected, assignments), atol=1e-12
    )
    distance = 2.0 * (1.0 - _unit_rows(corrected) @ centroids) / parameters.sigma
    batch = metadata["batch"].to_numpy()
    indicators = np.stack(
        [batch == level for level in ("batch_0", "batch_1", "batch_2")]
    )
    observed = indicators.astype(float) @ assignments
    expected_mass = (indicators.sum(axis=1) / len(batch))[:, None] * assignments.sum(
        axis=0
    )
    diversity = theta * np.sum(
        observed * np.log((observed + expected_mass + 1.0) / (expected_mass + 1.0))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(assignments > 0, assignments * np.log(assignments), 0.0)
    dense = np.sum(assignments * distance) + np.sum(entropy) + diversity
    np.testing.assert_allclose(summary.objectives[-1], dense, rtol=1e-10)


def _dense_held_out_pass(
    cells: np.ndarray,
    inverse_norms: np.ndarray,
    centroids: np.ndarray,
    start: np.ndarray,
    codes: np.ndarray,
    n_levels: int,
    theta: np.ndarray,
    sigma: np.ndarray,
    order: np.ndarray,
    block_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One clustering pass of the 2019 Algorithm 2, written densely from its equations.

    For each block of the fixed cell order, the block's assignments are held out
    of O and of the cluster masses, E is formed from the held-out masses and the
    global level proportions Pr_b, each cell of the block takes the Harmony2
    update R_ik proportional to prod_c ((E + 1) / (O + E + 1))^theta times
    exp(-2 (1 - y_k . z_i) / sigma_k), and the new assignments are added back.
    """
    n_cells = cells.shape[0]
    indicators = _indicators(codes, n_levels)
    proportions = indicators.sum(axis=1) / n_cells
    unit = cells * inverse_norms[:, None]
    assignments = start.copy()
    observed = indicators @ assignments
    masses = assignments.sum(axis=0)
    for begin in range(0, n_cells, block_cells):
        block = order[begin : begin + block_cells]
        observed -= indicators[:, block] @ assignments[block]
        masses -= assignments[block].sum(axis=0)
        expected = proportions[:, None] * masses[None, :]
        log_factor = np.log((expected + 1.0) / (observed + expected + 1.0))
        logits = -2.0 * (1.0 - unit[block] @ centroids) / sigma[None, :]
        for column_codes in codes:
            levels = column_codes[block]
            logits += theta[levels][:, None] * log_factor[levels]
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        assignments[block] = weights / weights.sum(axis=1, keepdims=True)
        observed += indicators[:, block] @ assignments[block]
        masses += assignments[block].sum(axis=0)
    return assignments, observed, masses


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("block_cells", [1, 7, 60])
def test_clustering_pass_equals_a_dense_held_out_pass(
    n_columns: int, block_cells: int
) -> None:
    rng = np.random.default_rng(40 + n_columns)
    n_cells, dims, n_clusters = 60, 4, 3
    columns = []
    for size in (4, 3)[:n_columns]:
        column = rng.integers(0, size, size=n_cells)
        column[:size] = np.arange(size)
        columns.append(column)
    design = encode_batch_design(
        pd.DataFrame({f"column{index}": column for index, column in enumerate(columns)})
    )
    codes = _global_codes(columns)
    np.testing.assert_array_equal(design.level_codes, codes)
    cells = rng.normal(size=(n_cells, dims))
    inverse_norms = 1.0 / np.linalg.norm(cells, axis=1)
    centroids = rng.normal(size=(dims, n_clusters))
    centroids /= np.linalg.norm(centroids, axis=0)
    start = rng.dirichlet(np.ones(n_clusters), size=n_cells)
    theta = rng.uniform(1.0, 4.0, size=design.n_levels)
    sigma = rng.uniform(0.1, 0.4, size=n_clusters)
    order = rng.permutation(n_cells).astype(np.int64)

    expected, observed, masses = _dense_held_out_pass(
        cells,
        inverse_norms,
        centroids,
        start,
        codes,
        design.n_levels,
        theta,
        sigma,
        order,
        block_cells,
    )
    assignments = start.copy()
    # Chunks of 16 cells split the larger blocks, which must not change anything.
    workspace = ClusteringWorkspace.allocate(
        min(16, block_cells), n_clusters, dims, design.n_levels
    )
    entropy = clustering_pass(
        cells,
        inverse_norms,
        centroids,
        assignments,
        design,
        order,
        block_cells,
        2.0 / sigma,
        theta,
        workspace,
    )

    np.testing.assert_allclose(assignments, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(workspace.level_mass, observed, rtol=0, atol=1e-12)
    np.testing.assert_allclose(workspace.totals, masses, rtol=0, atol=1e-12)
    np.testing.assert_allclose(entropy, np.sum(expected * np.log(expected)), rtol=1e-12)


# k-means++ seeding


def test_next_seed_draws_proportionally_to_squared_distance() -> None:
    squared = np.array([0.0, 1.0, 4.0, 5.0])
    race = np.empty(4)
    keys = np.empty(4)
    rng = np.random.default_rng(5)

    draws = np.array([next_seed(squared, [0], race, keys, rng) for _ in range(20_000)])

    assert not np.any(draws == 0)
    frequencies = np.bincount(draws, minlength=4) / draws.size
    np.testing.assert_allclose(frequencies[1:], [0.1, 0.4, 0.5], atol=0.02)


def test_next_seed_falls_back_to_unchosen_cells() -> None:
    squared = np.zeros(3)
    race = np.empty(3)
    keys = np.empty(3)
    rng = np.random.default_rng(6)

    draws = np.array([next_seed(squared, [1], race, keys, rng) for _ in range(4_000)])

    assert set(draws.tolist()) == {0, 2}
    np.testing.assert_allclose(np.mean(draws == 0), 0.5, atol=0.03)


def test_seeding_handles_repeated_cells() -> None:
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    cells = directions[[0, 1, 0, 1, 0, 1]] * np.array(
        [[1.0], [1.0], [2.0], [3.0], [0.5], [4.0]]
    )
    inverse_norms = 1.0 / np.linalg.norm(cells, axis=1)

    seeds = kmeans_plus_plus_seeds(cells, inverse_norms, 4, np.random.default_rng(0))

    assert seeds.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(seeds, axis=0), 1.0)
    units = _unit_rows(cells)
    for column in seeds.T:
        assert np.any(np.all(np.isclose(units, column), axis=1))
    # Both directions are seeded before any direction repeats.
    assert len({tuple(np.round(column, 12)) for column in seeds.T}) == 2


def test_third_seed_is_drawn_by_distance_to_the_nearest_seed() -> None:
    # Two pairs of nearly coincident directions, at 0 and 1 degrees and at 180
    # and 181 degrees, with cells of different lengths.
    angles = np.deg2rad([0.0, 1.0, 180.0, 181.0])
    cells = np.stack([np.cos(angles), np.sin(angles)], axis=1) * np.array(
        [[1.0], [2.0], [3.0], [0.5]]
    )
    inverse_norms = 1.0 / np.linalg.norm(cells, axis=1)
    units = cells * inverse_norms[:, None]
    pair = np.array([0, 0, 1, 1])
    runs = 2_000
    same_pair = 0
    for seed in range(runs):
        seeds = kmeans_plus_plus_seeds(
            cells, inverse_norms, 3, np.random.default_rng(seed)
        )
        chosen = [
            int(np.argmin(np.linalg.norm(units - column, axis=1))) for column in seeds.T
        ]
        assert len(set(chosen)) == 3
        same_pair += int(pair[chosen[2]] == pair[chosen[0]])

    # The second seed almost surely comes from the other pair. The two remaining
    # cells then lie equally close to their nearest seed, one beside each seed,
    # so the third seed joins the first seed's pair half of the time. Distances
    # to the first seed alone would almost never put it there (about 8e-5).
    assert abs(same_pair / runs - 0.5) < 0.05


# Centroids


def test_cells_that_cancel_exactly_leave_a_zero_centroid() -> None:
    rng = np.random.default_rng(0)
    n_cells, dims, n_clusters = 500, 7, 5
    cells = rng.normal(size=(n_cells, dims)) * rng.uniform(0.1, 10.0, size=(n_cells, 1))
    assignments = rng.dirichlet(np.ones(n_clusters), size=n_cells)
    assignments[:, 4] = 0.0
    assignments /= assignments.sum(axis=1, keepdims=True)
    # Cluster 4 holds only two opposite cells. Their unit vectors cancel
    # exactly, but a blocked BLAS sum over the chunk that holds them can leave
    # a rounding residue of about 1e-17, whose direction is arbitrary.
    assignments[[10, 11]] = 0.0
    assignments[[10, 11], 4] = 1.0
    cells[11] = -cells[10]
    inverse_norms = np.empty(n_cells)
    update_inverse_norms(cells, inverse_norms, np.empty(64))
    assert inverse_norms[10] == inverse_norms[11]
    assert np.all(cells[10] * inverse_norms[10] + cells[11] * inverse_norms[11] == 0.0)
    workspace = ClusteringWorkspace.allocate(64, n_clusters, dims, 2)

    centroids, norms = cosine_centroids(cells, inverse_norms, assignments, workspace)

    # A zero centroid and a zero norm, so the objective sees y . S = 0.
    assert np.all(centroids[:, 4] == 0.0)
    assert norms[4] == 0.0
    np.testing.assert_allclose(
        centroids[:, :4], _cosine_centroids(cells, assignments)[:, :4], atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(centroids[:, :4], axis=0), 1.0)
    np.testing.assert_allclose(
        norms[:4],
        np.linalg.norm(_unit_rows(cells).T @ assignments[:, :4], axis=0),
        rtol=1e-12,
    )


# Fits


def test_zero_norm_cells_are_assigned() -> None:
    values, metadata, _batch = _three_batches(seed=9)
    values[:, 7] = 0.0

    result = fit_harmony(values, metadata, n_clusters=3, max_iter_harmony=2)

    assert np.all(np.isfinite(result.corrected))
    assert np.all(np.isfinite(result.assignments))
    np.testing.assert_allclose(result.assignments[:, 7].sum(), 1.0, atol=1e-12)


def test_rows_of_assignments_sum_to_one_and_centroids_match() -> None:
    values, metadata, _batch = _three_batches(seed=10)

    result = fit_harmony(values, metadata, n_clusters=4, max_iter_harmony=3)

    np.testing.assert_allclose(result.assignments.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        result.centroids,
        _cosine_centroids(result.corrected.T, result.assignments.T),
        atol=1e-12,
    )
    assert result.centroids.shape == (4, 4)
    assert result.sigma.tolist() == [0.1] * 4


# End to end: batches that share no cell types


def _normalized_entropy(neighbors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Shannon entropy of each cell's neighbour labels over log(min(k, levels))."""
    codes = np.unique(labels, return_inverse=True)[1]
    n_labels = int(codes.max()) + 1
    k = neighbors.shape[1]
    counts = np.zeros((neighbors.shape[0], n_labels))
    np.add.at(
        counts,
        (np.repeat(np.arange(neighbors.shape[0]), k), codes[neighbors].ravel()),
        1.0,
    )
    frequencies = counts / k
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(frequencies > 0, frequencies * np.log(frequencies), 0.0), axis=1
        )
    return entropy / np.log(min(k, n_labels))


def _mixing_and_purity(
    coordinates: np.ndarray, batches: np.ndarray, types: np.ndarray, k: int = 30
) -> tuple[float, float]:
    _distances, neighbors = cKDTree(coordinates).query(coordinates, k=k + 1)
    # The nearest point of each cell is the cell itself.
    neighbors = neighbors[:, 1:]
    mixing = float(np.mean(_normalized_entropy(neighbors, batches)))
    purity = 1.0 - float(np.mean(_normalized_entropy(neighbors, types)))
    return mixing, purity


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harmony_mixes_batches_without_merging_non_overlapping_cell_types(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    dims = 12
    basis = np.linalg.qr(rng.normal(size=(dims, dims)))[0]
    angle = np.deg2rad(60.0)
    centers = np.empty((6, dims))
    for kind in range(3):
        centers[kind] = 6.0 * basis[:, kind]
        # Each type of the second group is a related type of the first,
        # 60 degrees away.
        centers[kind + 3] = 6.0 * (
            np.cos(angle) * basis[:, kind] + np.sin(angle) * basis[:, kind + 3]
        )
    # Batches 0 to 3 hold types 0, 1, 2 and batches 4 to 7 types 3, 4, 5,
    # 50 cells of each type in each of its batches.
    batches = np.repeat(np.arange(8), 150)
    types = np.concatenate(
        [np.repeat(np.arange(3) + 3 * (batch // 4), 50) for batch in range(8)]
    )
    shift = rng.normal(scale=0.6, size=(8, dims))
    values = (
        centers[types]
        + shift[batches]
        + rng.normal(scale=0.5, size=(batches.size, dims))
    )
    values -= values.mean(axis=0)

    result = fit_harmony(
        values.T,
        pd.DataFrame({"batch": [f"s{batch}" for batch in batches]}),
        random_state=seed,
    )

    mixing_before, purity_before = _mixing_and_purity(values, batches, types)
    mixing_after, purity_after = _mixing_and_purity(result.corrected.T, batches, types)
    assert purity_before >= 0.98
    assert purity_after >= 0.98
    # A correct neighbourhood holds the four batches of its own group, so
    # mixing cannot exceed about log 4 / log 8 = 0.667.
    assert mixing_after >= 0.5
    assert mixing_after >= mixing_before + 0.3
    assert result.parameters["converged"]


# Failure paths


def _regression_case(seed: int = 2):
    rng, _labels, design, indicators = _single_column_case(seed)
    assignments = rng.uniform(0.05, 1.0, size=(300, 3))
    coordinates = rng.normal(size=(300, 4))
    penalties = rng.uniform(0.1, 2.0, size=(design.n_levels, 3))

    def statistics() -> RegressionStatistics:
        return _dense_statistics(design, indicators, assignments, coordinates)

    return design, statistics, penalties


def test_the_dynamic_penalty_needs_an_alpha() -> None:
    design, statistics, _penalties = _regression_case()

    with pytest.raises(ValueError, match="The dynamic ridge penalty needs ridge_alpha"):
        ridge_penalties(
            statistics(), design, None, None, np.empty((design.n_levels, 3))
        )


def test_the_arrowhead_refuses_penalties_that_vanish_against_the_masses() -> None:
    design, statistics, penalties = _regression_case()
    # With every level active and no penalty, the intercept keeps no mass of its
    # own, so the Schur complement of the arrowhead is 0.
    active = np.ones((design.n_levels, 3), dtype=bool)

    with pytest.raises(
        np.linalg.LinAlgError,
        match="ridge regression for cluster 0 is not positive definite",
    ):
        solve_cluster_regressions(
            statistics(), np.zeros_like(penalties), active, design
        )


def test_elimination_zeroes_a_cluster_without_active_levels() -> None:
    design, statistics, penalties = _regression_case()
    active = np.ones((design.n_levels, 3), dtype=bool)
    active[:, 1] = False

    general = statistics()
    _solve_by_elimination(general, penalties, active, design)
    arrowhead = solve_cluster_regressions(statistics(), penalties, active, design)

    np.testing.assert_array_equal(general.level_sums[:, 1, :], 0.0)
    np.testing.assert_allclose(general.level_sums, arrowhead, rtol=1e-10, atol=1e-12)


def test_elimination_reports_a_system_that_is_not_positive_definite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, statistics, penalties = _regression_case()
    active = np.ones((design.n_levels, 3), dtype=bool)

    def refuse(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("1-th leading minor not positive definite")

    monkeypatch.setattr(correction_module, "cho_factor", refuse)
    with pytest.raises(
        np.linalg.LinAlgError,
        match="ridge regression for cluster 0 is not positive definite",
    ) as raised:
        _solve_by_elimination(statistics(), penalties, active, design)
    assert isinstance(raised.value.__cause__, np.linalg.LinAlgError)


def test_a_non_finite_objective_stops_the_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    values, metadata, _batch = _three_batches()
    monkeypatch.setattr(
        clustering_module, "clustering_objective", lambda *args: math.nan
    )

    with pytest.raises(ValueError, match="non-finite objective in round 0"):
        fit_harmony(values, metadata, n_clusters=3, max_iter_harmony=1)


def test_non_finite_centroids_stop_the_fit() -> None:
    _require_finite_centroids(np.array([[0.6, 0.8]]), 4)

    with pytest.raises(ValueError, match="non-finite centroids in round 4"):
        _require_finite_centroids(np.array([[0.6, np.nan]]), 4)
