import math

import numpy as np
import pytest
from scipy import stats

from cytearc.trajectory.curves import (
    CURVE_FIT_BINS,
    CURVE_SPLINE_BASIS,
    LineageCurveFit,
    _accumulate_rank_bins,
    _project_rows,
    _PenalizedSpline,
    curve_values_at,
    fit_lineage_curves,
    lineage_ranks,
    project_onto_curve,
    projection_scratch_bytes,
    smooth_against_progress,
)


def _helix(t: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t), 1.5 * t))


def _helix_cells(seed: int, n_cells: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Cells at evenly spaced helix parameters in random order, with noise."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n_cells) + 0.5) / n_cells
    t = t[rng.permutation(n_cells)]
    return t, _helix(t) + rng.normal(scale=0.05, size=(n_cells, 3))


def _midpoint_ranks(ordering: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weight before each cell plus half its own, over the total, cell by cell."""
    order = sorted(range(len(ordering)), key=lambda cell: (ordering[cell], cell))
    ranks = np.full(weights.shape, np.nan)
    for lineage in range(weights.shape[1]):
        total = weights[:, lineage].sum()
        before = 0.0
        for cell in order:
            weight = weights[cell, lineage]
            if weight > 0.0:
                ranks[cell, lineage] = (before + 0.5 * weight) / total
            before += weight
    return ranks


def _nearest_on_polyline(
    row: np.ndarray, points: np.ndarray, progress: np.ndarray
) -> tuple[float, float]:
    """Project one row onto each segment with explicit differences."""
    best_distance, best_position = math.inf, math.nan
    for index in range(points.shape[0] - 1):
        start, vector = points[index], points[index + 1] - points[index]
        length = float(vector @ vector)
        fraction = 0.0
        if length > 0.0:
            fraction = min(max(float((row - start) @ vector) / length, 0.0), 1.0)
        distance = float(np.linalg.norm(row - (start + fraction * vector)))
        if distance < best_distance:
            best_distance = distance
            best_position = progress[index] + fraction * (
                progress[index + 1] - progress[index]
            )
    return best_position, best_distance


def _fit(coordinates, ranks, weights, ordering, blocks=None, **options):
    if blocks is None:
        blocks = [(0, coordinates)]
    return fit_lineage_curves(
        blocks,
        ranks,
        weights,
        ordering,
        n_dims=coordinates.shape[1],
        n_points=options.pop("n_points", 101),
        degrees_of_freedom=options.pop("degrees_of_freedom", 20.0),
        **options,
    )


def test_lineage_ranks_are_weighted_midpoints_along_pseudotime():
    # Cells 0 and 2 tie and keep their positions.
    ordering = np.array([0.5, 0.1, 0.5, 0.9])
    weights = np.array([[1.0, 0.5], [1.0, 0.0], [2.0, 0.5], [0.0, 1.0]])

    ranks = lineage_ranks(ordering, weights)

    np.testing.assert_array_equal(
        ranks,
        [[0.375, 0.125], [0.125, np.nan], [0.75, 0.375], [np.nan, 0.75]],
    )


def test_lineage_ranks_match_a_cell_by_cell_reference():
    rng = np.random.default_rng(0)
    ordering = rng.integers(0, 6, size=40).astype(np.float64)
    weights = rng.dirichlet(np.ones(3), size=40)
    weights[rng.random(weights.shape) < 0.2] = 0.0

    ranks = lineage_ranks(ordering, weights)

    np.testing.assert_allclose(
        ranks, _midpoint_ranks(ordering, weights), rtol=0.0, atol=1e-14
    )
    assert np.nanmin(ranks) > 0.0 and np.nanmax(ranks) < 1.0
    np.testing.assert_array_equal(np.isnan(ranks), weights == 0.0)


@pytest.mark.parametrize(
    ("ordering", "weights", "message"),
    [
        (np.zeros(3), np.ones(3), "ordering must have one value per weight row"),
        (np.zeros(2), np.ones((3, 1)), "ordering must have one value per weight row"),
        (np.array([0.0, np.inf, 1.0]), np.ones((3, 1)), "ordering must be finite"),
        (
            np.zeros(3),
            np.array([[1.0], [np.nan], [1.0]]),
            "weights must be finite and non-negative",
        ),
        (
            np.zeros(3),
            np.array([[1.0], [-0.5], [1.0]]),
            "weights must be finite and non-negative",
        ),
        (
            np.zeros(3),
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            "Every lineage needs a positive total weight",
        ),
    ],
)
def test_lineage_ranks_reject_invalid_inputs(ordering, weights, message):
    with pytest.raises(ValueError, match=message):
        lineage_ranks(ordering, weights)


def test_penalized_spline_basis_is_a_cubic_partition_of_unity_on_the_unit_interval():
    spline = _PenalizedSpline(12)
    x = np.linspace(0.0, 1.0, 37)

    design = spline.design(x)

    assert spline.n_basis == 12 and spline.knots.shape == (16,)
    assert design.shape == (37, 12)
    np.testing.assert_allclose(design.sum(axis=1), 1.0, rtol=1e-14)
    np.testing.assert_array_equal(spline.design([-0.5, 1.7]), spline.design([0.0, 1.0]))
    # The penalty is zero exactly on straight lines, its null space.
    line = np.linspace(-1.0, 2.0, 12)
    assert abs(line @ spline.penalty @ line) < 1e-12
    assert np.allclose(spline.penalty, spline.penalty.T)


@pytest.mark.parametrize("degrees_of_freedom", [2.5, 4.0, 7.5, 11.0])
def test_penalized_spline_reproduces_a_straight_line(degrees_of_freedom):
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(size=50))
    weights = rng.uniform(0.2, 3.0, size=50)
    line = np.column_stack((3.0 * x - 1.0, -2.0 * x + 0.5, np.full(50, 4.0)))
    spline = _PenalizedSpline(12)

    coefficients = spline.fit(x, weights, line, degrees_of_freedom)

    grid = np.linspace(0.0, 1.0, 21)
    expected = np.column_stack((3.0 * grid - 1.0, -2.0 * grid + 0.5, np.full(21, 4.0)))
    np.testing.assert_allclose(spline.design(grid) @ coefficients, expected, atol=1e-10)


@pytest.mark.parametrize("degrees_of_freedom", [2.2, 3.0, 6.5, 10.0, 15.0])
def test_penalized_spline_spends_the_requested_degrees_of_freedom(degrees_of_freedom):
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(size=60))
    weights = rng.uniform(0.5, 2.0, size=60)
    spline = _PenalizedSpline(16)

    # The penalty does not depend on the values, so fitting the identity
    # returns the smoother, whose trace is the effective degrees of freedom.
    smoother = spline.design(x) @ spline.fit(x, weights, np.eye(60), degrees_of_freedom)

    assert np.trace(smoother) == pytest.approx(degrees_of_freedom, rel=1e-9)


def test_penalized_spline_rejects_more_degrees_of_freedom_than_distinct_ranks():
    spline = _PenalizedSpline(12)
    x = np.array([0.1, 0.1, 0.5, 0.9, 0.9])

    with pytest.raises(
        ValueError,
        match="too few distinct pseudotime ranks for 5 degrees of freedom; lower df",
    ):
        spline.fit(x, np.ones(5), x[:, np.newaxis], 5.0)


def test_fit_lineage_curves_recover_a_noisy_helix():
    t, coordinates = _helix_cells(3)
    # Lineage 1 weighs later cells more: its weighted pseudotime distribution
    # is F(t) = (t + t^2) / 2, so its rank p sits at t = (sqrt(1 + 8p) - 1) / 2.
    weights = np.column_stack((np.ones(t.size), 0.5 + t))
    ranks = lineage_ranks(t, weights)

    fit = _fit(coordinates, ranks, weights, t)

    assert isinstance(fit, LineageCurveFit)
    assert fit.points.shape == (2, 101, 3) and fit.ordering.shape == (2, 101)
    np.testing.assert_array_equal(fit.progress, np.linspace(0.0, 1.0, 101))
    inverse = (np.sqrt(1.0 + 8.0 * fit.progress) - 1.0) / 2.0
    for lineage, truth in ((0, fit.progress), (1, inverse)):
        error = np.linalg.norm(fit.points[lineage] - _helix(truth), axis=1)
        assert error.max() < 0.04
        assert np.median(error) < 0.01
        assert np.all(np.diff(fit.ordering[lineage]) >= 0.0)
        position, distance = project_onto_curve(
            coordinates, fit.points[lineage], fit.progress
        )
        assert stats.spearmanr(position, t).statistic > 0.99
        assert np.median(distance) < 0.1
    np.testing.assert_allclose(fit.ordering[0], fit.progress, atol=1e-12)
    np.testing.assert_allclose(fit.ordering[1], inverse, atol=0.005)


def test_fit_lineage_curves_do_not_depend_on_the_block_partition():
    t, coordinates = _helix_cells(4, n_cells=900)
    weights = np.column_stack((np.ones(t.size), 0.5 + t, np.where(t < 0.6, 1.0, 0.0)))
    ranks = lineage_ranks(t, weights)
    whole = _fit(coordinates, ranks, weights, t, degrees_of_freedom=9.0)

    for cuts in ((0, 1, 8, 8, 400, 401, 900), tuple(range(0, 901, 7)) + (900,)):
        blocks = [
            (start, coordinates[start:stop])
            for start, stop in zip(cuts[:-1], cuts[1:], strict=True)
        ]
        split = _fit(
            coordinates, ranks, weights, t, blocks=blocks, degrees_of_freedom=9.0
        )
        np.testing.assert_allclose(split.points, whole.points, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(split.ordering, whole.ordering, rtol=0.0, atol=1e-10)


def test_fit_lineage_curves_keep_the_fitted_pseudotime_non_decreasing():
    t, coordinates = _helix_cells(5, n_cells=1200)
    # A jump in pseudotime makes the smoothed pseudotime overshoot.
    ordering = np.where(t < 0.5, t, t + 10.0)
    weights = np.ones((t.size, 1))

    fit = _fit(coordinates, lineage_ranks(ordering, weights), weights, ordering)

    assert np.all(np.diff(fit.ordering[0]) >= 0.0)
    assert fit.ordering[0, 0] == pytest.approx(0.0, abs=0.05)
    assert fit.ordering[0, -1] == pytest.approx(11.0, abs=0.05)


def _bad_blocks(name: str, coordinates: np.ndarray) -> list:
    n_cells = coordinates.shape[0]
    if name == "gap":
        return [(0, coordinates[:3]), (4, coordinates[4:])]
    if name == "late start":
        return [(1, coordinates)]
    if name == "dimensions":
        return [(0, coordinates[:, :2])]
    if name == "flat":
        return [(0, coordinates[:, 0])]
    if name == "beyond":
        return [(0, coordinates), (n_cells, coordinates[:1])]
    if name == "non-finite":
        broken = coordinates.copy()
        broken[2, 1] = np.nan
        return [(0, broken)]
    return [(0, coordinates[:-2])]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("gap", "Coordinate blocks must cover the cells in order"),
        ("late start", "Coordinate blocks must cover the cells in order"),
        ("dimensions", "Coordinate blocks must cover the cells in order"),
        ("flat", "Coordinate blocks must cover the cells in order"),
        ("beyond", "Coordinate blocks extend beyond the cells"),
        ("non-finite", "Coordinates must be finite"),
        ("short", "Coordinate blocks must cover every cell"),
    ],
)
def test_fit_lineage_curves_reject_blocks_that_do_not_cover_the_cells(name, message):
    t, coordinates = _helix_cells(6, n_cells=40)
    weights = np.ones((40, 1))

    with pytest.raises(ValueError, match=message):
        _fit(
            coordinates,
            lineage_ranks(t, weights),
            weights,
            t,
            blocks=_bad_blocks(name, coordinates),
            degrees_of_freedom=4.0,
        )


def test_fit_lineage_curves_reject_lineages_with_too_few_distinct_ranks():
    t, coordinates = _helix_cells(7, n_cells=4)
    weights = np.ones((4, 1))

    with pytest.raises(ValueError, match="too few distinct pseudotime ranks"):
        _fit(coordinates, lineage_ranks(t, weights), weights, t, degrees_of_freedom=6.0)


def test_curve_fit_constants_describe_the_default_fit():
    assert CURVE_FIT_BINS == 512
    assert CURVE_SPLINE_BASIS == 32


def test_accumulate_rank_bins_agrees_with_its_python_source():
    rng = np.random.default_rng(8)
    block = rng.normal(size=(30, 2))
    weights = rng.uniform(size=(30, 3))
    weights[rng.random(weights.shape) < 0.3] = 0.0
    ranks = rng.uniform(size=(30, 3))
    # A rank of one belongs to the last bin.
    ranks[:4] = 1.0
    ordering = rng.uniform(size=30)
    outputs = []
    for kernel in (_accumulate_rank_bins, _accumulate_rank_bins.py_func):
        mass = np.zeros((3, 8))
        rank_sums = np.zeros((3, 8))
        ordering_sums = np.zeros((3, 8))
        sums = np.zeros((3, 8, 2))
        kernel(block, ranks, weights, ordering, 8, mass, rank_sums, ordering_sums, sums)
        outputs.append((mass, rank_sums, ordering_sums, sums))

    for compiled, python in zip(*outputs, strict=True):
        np.testing.assert_array_equal(compiled, python)
    mass, rank_sums, ordering_sums, sums = outputs[0]
    bins = np.minimum((ranks * 8).astype(np.int64), 7)
    for lineage in range(3):
        for bin_index in range(8):
            cells = (bins[:, lineage] == bin_index) & (weights[:, lineage] > 0.0)
            weight = weights[cells, lineage]
            assert mass[lineage, bin_index] == pytest.approx(weight.sum())
            assert rank_sums[lineage, bin_index] == pytest.approx(
                weight @ ranks[cells, lineage]
            )
            assert ordering_sums[lineage, bin_index] == pytest.approx(
                weight @ ordering[cells]
            )
            np.testing.assert_allclose(
                sums[lineage, bin_index], weight @ block[cells], atol=1e-14
            )


def test_projection_scratch_bytes_count_two_values_per_row():
    # The fused kernel holds no per-segment arrays.
    assert projection_scratch_bytes(2) == 16
    assert projection_scratch_bytes(101) == 16


def test_projection_kernel_matches_its_python_source_and_a_dense_search():
    rng = np.random.default_rng(31)
    points = np.cumsum(rng.normal(size=(12, 3)), axis=0)
    # A repeated point makes a zero-length segment, measured from its start.
    points[5] = points[4]
    rows = rng.normal(scale=3.0, size=(40, 3))
    progress = np.linspace(0.0, 1.0, points.shape[0])
    vectors = np.diff(points, axis=0)
    lengths = np.einsum("ij,ij->i", vectors, vectors)
    inverse = np.zeros_like(lengths)
    inverse[lengths > 0] = 1.0 / lengths[lengths > 0]
    arguments = (rows, points[:-1].copy(), vectors, inverse, progress)
    compiled = _project_rows(*arguments)
    python = _project_rows.py_func(*arguments)
    np.testing.assert_allclose(compiled[0], python[0], atol=1e-12)
    np.testing.assert_allclose(compiled[1], python[1], atol=1e-12)
    # No point sampled densely along the polyline lies nearer than reported.
    steps = np.linspace(0.0, 1.0, 401)
    samples = (
        points[:-1, None, :] * (1.0 - steps[None, :, None])
        + points[1:, None, :] * steps[None, :, None]
    ).reshape(-1, 3)
    nearest = np.sqrt(((rows[:, None, :] - samples[None]) ** 2).sum(axis=2)).min(axis=1)
    assert (compiled[1] <= nearest + 1e-12).all()
    assert (compiled[1] >= nearest - 0.05).all()
    position, distance = project_onto_curve(rows, points, progress, nthreads=2)
    np.testing.assert_allclose(position, compiled[0], atol=0.0)
    np.testing.assert_allclose(distance, compiled[1], atol=0.0)


def test_project_onto_curve_places_rows_on_a_hand_made_polyline_exactly():
    # The curve's mean is (4, 2), so every centred value is a small integer.
    points = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [8.0, 4.0]])
    progress = np.array([0.0, 0.5, 0.75, 1.0])
    rows = np.array(
        [
            [2.0, -3.0],  # Beside the first segment.
            [5.0, 1.0],  # Nearest the second segment.
            [6.0, 4.0],  # On the third segment.
            [-3.0, 0.0],  # Before the start.
            [11.0, 8.0],  # After the end.
            [5.0, -1.0],  # Nearest the corner between two segments.
        ]
    )

    position, distance = project_onto_curve(rows, points, progress)

    np.testing.assert_array_equal(position, [0.25, 0.5625, 0.875, 0.0, 1.0, 0.5])
    np.testing.assert_array_equal(distance, [3.0, 1.0, 0.0, 3.0, 5.0, math.sqrt(2.0)])


def test_project_onto_curve_handles_zero_length_segments():
    # Repeated points at both ends give zero-length first and last segments.
    points = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 4.0], [2.0, 4.0]])
    progress = np.array([0.0, 0.25, 0.75, 1.0])
    rows = np.array([[2.0, -1.0], [3.0, 1.0], [2.0, 5.0], [0.0, 4.0], [2.0, 2.0]])
    middle = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 4.0]])

    position, distance = project_onto_curve(rows, points, progress)
    middle_position, middle_distance = project_onto_curve(
        np.array([[3.0, 0.0], [1.0, 0.5], [2.0, 2.0]]), middle, progress
    )

    assert np.isfinite(position).all() and np.isfinite(distance).all()
    # Rows nearest a repeated point take the first segment that reaches it.
    np.testing.assert_array_equal(position, [0.0, 0.375, 0.75, 0.75, 0.5])
    np.testing.assert_array_equal(distance, [1.0, 1.0, 1.0, 2.0, 0.0])
    np.testing.assert_array_equal(middle_position, [0.25, 0.125, 0.875])
    np.testing.assert_array_equal(middle_distance, [1.0, 0.5, 0.0])


def test_project_onto_curve_matches_a_segment_by_segment_reference():
    rng = np.random.default_rng(9)
    points = np.cumsum(rng.normal(size=(8, 3)), axis=0) + 50.0
    progress = np.concatenate(([0.0], np.sort(rng.uniform(size=6)), [1.0]))
    near = points[rng.integers(0, 8, size=60)] + rng.normal(scale=0.4, size=(60, 3))
    far = rng.normal(scale=6.0, size=(20, 3)) + points.mean(axis=0)
    rows = np.vstack((near, far))

    position, distance = project_onto_curve(rows, points, progress)

    expected = np.array([_nearest_on_polyline(row, points, progress) for row in rows])
    np.testing.assert_allclose(position, expected[:, 0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(distance, expected[:, 1], rtol=1e-9, atol=1e-9)


def test_project_onto_curve_keeps_rows_on_the_curve_within_rounding():
    points = np.array([[0.1, 0.3], [1.7, 0.2], [2.9, 1.3]])
    progress = np.array([0.0, 0.4, 1.0])
    fractions = np.linspace(0.0, 1.0, 9)
    on_first = points[0] + fractions[:, np.newaxis] * (points[1] - points[0])
    on_second = points[1] + fractions[:, np.newaxis] * (points[2] - points[1])

    position, distance = project_onto_curve(
        np.vstack((on_first, on_second)), points, progress
    )

    np.testing.assert_allclose(
        position, np.concatenate((0.4 * fractions, 0.4 + 0.6 * fractions)), atol=1e-12
    )
    # The squared distances are expanded into dot products, which leaves a
    # rounding floor of about sqrt(eps) times the curve's extent.
    assert distance.max() < 1e-7


@pytest.mark.parametrize(
    ("rows", "points", "message"),
    [
        (np.zeros((2, 3)), np.zeros((4, 2)), "must share their dimensions"),
        (np.zeros(2), np.zeros((4, 2)), "must share their dimensions"),
        (np.zeros((2, 2)), np.zeros(4), "must share their dimensions"),
        (np.zeros((2, 2)), np.ones((1, 2)), "A curve needs at least two points"),
        (np.zeros((2, 2)), np.ones((3, 2)), "A curve must not collapse to one point"),
    ],
)
def test_project_onto_curve_rejects_invalid_curves(rows, points, message):
    with pytest.raises(ValueError, match=message):
        project_onto_curve(rows, points, np.linspace(0.0, 1.0, points.shape[0]))


@pytest.mark.parametrize("progress", [np.linspace(0.0, 1.0, 3), np.zeros((4, 1))])
def test_project_onto_curve_rejects_progress_of_another_shape(progress):
    points = np.column_stack((np.linspace(0.0, 3.0, 4), np.zeros(4)))

    with pytest.raises(ValueError, match="one value per curve point"):
        project_onto_curve(np.zeros((2, 2)), points, progress)


def test_curve_values_at_interpolate_between_curve_points_and_clamp():
    values = curve_values_at(
        np.array([-0.5, 0.0, 0.25, 0.5, 0.9, 1.0, 2.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([10.0, 20.0, 40.0]),
    )

    assert values.dtype == np.float64
    np.testing.assert_allclose(values, [10.0, 10.0, 15.0, 20.0, 36.0, 40.0, 40.0])


def test_smooth_against_progress_draws_a_layout_curve_from_weighted_cells():
    rng = np.random.default_rng(10)
    position = rng.uniform(size=3000)
    layout = np.column_stack((2.0 * np.cos(np.pi * position), np.sin(np.pi * position)))
    layout += rng.normal(scale=0.02, size=layout.shape)
    weights = rng.uniform(0.2, 1.0, size=3000)
    # Cells without weight, without a position, or with missing coordinates
    # would ruin the curve if they were used.
    position[:20] = np.nan
    layout[20:40] = 1e6
    weights[20:40] = 0.0
    layout[40:60, 1] = np.nan

    curve = smooth_against_progress(
        layout, position, weights, n_points=51, degrees_of_freedom=10.0
    )

    grid = np.linspace(0.0, 1.0, 51)
    truth = np.column_stack((2.0 * np.cos(np.pi * grid), np.sin(np.pi * grid)))
    assert curve.shape == (51, 2)
    assert np.abs(curve - truth).max() < 0.03


def test_smooth_against_progress_reproduces_a_straight_layout():
    position = np.concatenate(([0.0, 1.0], np.linspace(0.01, 0.99, 300)))
    layout = np.column_stack((2.0 * position - 1.0, 3.0 - position))
    weights = np.linspace(0.5, 1.5, position.size)

    curve = smooth_against_progress(
        layout,
        position,
        weights,
        n_points=11,
        degrees_of_freedom=6.0,
        n_bins=64,
        n_basis=16,
    )

    grid = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(
        curve, np.column_stack((2.0 * grid - 1.0, 3.0 - grid)), atol=1e-10
    )


def test_smooth_against_progress_requires_weighted_cells_on_the_curve():
    layout = np.ones((5, 2))

    with pytest.raises(ValueError, match="No weighted cells lie on the curve"):
        smooth_against_progress(
            layout,
            np.linspace(0, 1, 5),
            np.zeros(5),
            n_points=5,
            degrees_of_freedom=3.0,
        )
    with pytest.raises(ValueError, match="No weighted cells lie on the curve"):
        smooth_against_progress(
            layout, np.full(5, np.nan), np.ones(5), n_points=5, degrees_of_freedom=3.0
        )
