import math

import numba
import numpy as np
import pytest
from scipy import stats
from scipy.special import betainc
from statsmodels.stats.multitest import multipletests

import cytearc.trajectory.lineages as lineages_module
from cytearc.trajectory.lineages import (
    LINEAGE_NONFINITE,
    LINEAGE_OK,
    LINEAGE_SCRATCH_ITEMSIZE,
    LINEAGE_UNTESTED,
    _bin_overlaps,
    _bin_sums,
    _centered_regressors,
    _check_block,
    _weighted_correlation_kernel,
    adjust_lineage_p_values,
    bin_operator_bytes,
    bin_smoothing_matrix,
    equal_mass_bin_operator,
    finalize_lineage_profiles,
    fisher_z_statistics,
    kish_effective_sizes,
    kish_p_values,
    lineage_profiles,
    select_lineages,
    stream_lineage_profiles,
    stream_weighted_correlations,
    weighted_correlations,
)


def _lineage_block(
    seed: int,
    n_cells: int = 48,
    n_features: int = 6,
    n_lineages: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse non-negative values, fate-like weights with zeros, and regressors."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(2.0, 1.0, size=(n_cells, n_features))
    values[rng.random(values.shape) < 0.3] = 0.0
    weights = rng.dirichlet(np.ones(n_lineages), size=n_cells)
    weights[rng.random(weights.shape) < 0.15] = 0.0
    regressors = rng.normal(size=(n_cells, n_lineages))
    return values, weights, regressors


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson r from weighted means and weighted central moments."""
    total = w.sum()
    x_mean = (w * x).sum() / total
    y_mean = (w * y).sum() / total
    cross = (w * (x - x_mean) * (y - y_mean)).sum()
    x_squares = (w * (x - x_mean) ** 2).sum()
    y_squares = (w * (y - y_mean) ** 2).sum()
    return float(cross / math.sqrt(x_squares * y_squares))


def _weighted_pearson_table(
    values: np.ndarray, weights: np.ndarray, regressors: np.ndarray
) -> np.ndarray:
    return np.array(
        [
            [
                _weighted_pearson(
                    values[:, feature], regressors[:, lineage], weights[:, lineage]
                )
                for lineage in range(weights.shape[1])
            ]
            for feature in range(values.shape[1])
        ]
    )


def _correlate(
    values: np.ndarray,
    weights: np.ndarray,
    regressors: np.ndarray,
    min_mass: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    return weighted_correlations(
        values,
        weights,
        regressors,
        min_mass=min_mass,
        nthreads=2,
        feature_labels=[f"g{index}" for index in range(np.shape(values)[1])],
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Step-up BH: the running minimum of m * p_(k) / k from the largest p down."""
    count = p_values.size
    order = np.argsort(p_values)
    scaled = p_values[order] * count / np.arange(1, count + 1)
    stepped = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(count)
    adjusted[order] = stepped
    return adjusted


def _brute_force_bin_operator(
    ordering: np.ndarray, weights: np.ndarray, n_bins: int
) -> np.ndarray:
    """Each cell's share of every equal-mass bin, from interval overlaps in mass.

    Cells are laid end to end in pseudotime order, ties by position, each
    occupying an interval as long as its weight; bin ``b`` of a lineage is the
    mass interval ``[b, b + 1)`` times the lineage's total over ``n_bins``.
    """
    n_cells, n_lineages = weights.shape
    order = sorted(range(n_cells), key=lambda cell: (ordering[cell], cell))
    dense = np.zeros((n_lineages * n_bins, n_cells))
    for lineage in range(n_lineages):
        width = weights[:, lineage].sum() / n_bins
        before = 0.0
        for cell in order:
            after = before + weights[cell, lineage]
            for bin_index in range(n_bins):
                overlap = min(after, (bin_index + 1) * width) - max(
                    before, bin_index * width
                )
                if overlap > 0.0:
                    dense[lineage * n_bins + bin_index, cell] = overlap / width
            before = after
    return dense


def _truncated_gaussian_smoother(n_bins: int, sigma: float) -> np.ndarray:
    radius = math.ceil(4.0 * sigma)
    kernel = np.zeros((n_bins, n_bins))
    for row in range(n_bins):
        for col in range(n_bins):
            if abs(row - col) <= radius:
                kernel[row, col] = math.exp(-0.5 * ((row - col) / sigma) ** 2)
        kernel[row] /= kernel[row].sum()
    return kernel


def _status_block() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eight cells, three lineages, and features that hit each untested rule.

    Lineage 0 weights cells 0-3, lineage 1 cells 4-7, and lineage 2 every cell
    with a constant regressor.
    """
    weights = np.zeros((8, 3))
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0
    weights[:, 2] = 0.5
    ramp = np.arange(8.0)
    regressors = np.column_stack((ramp, ramp, np.full(8, 3.0)))
    eps = np.finfo(np.float64).eps
    tail = [1.0, 2.0, 3.0, 4.0]
    values = np.array(
        [
            [0.0] * 8,
            [0.0, 0.0, 0.0, 0.0, *tail],
            [5.0, 5.0, 5.0, 5.0, *tail],
            [0.0, 0.0, 2.0, 3.0, *tail],
            [0.0, 0.0, 0.0, 3.0, *tail],
            [1.0, 1.0, 1.0, 1.0 + eps, *tail],
            [1.0, 1.0, 1.0, 1.0 + 2.0 * eps, *tail],
        ]
    ).T
    return values, weights, regressors


def test_select_lineages_defaults_to_every_sink_in_stored_order():
    labels, columns = select_lineages(("A", "B", "C"), None)

    assert labels == ("A", "B", "C")
    np.testing.assert_array_equal(columns, [0, 1, 2])
    assert columns.dtype == np.int64


def test_select_lineages_name_where_a_missing_label_was_sought():
    with pytest.raises(ValueError, match="Lineage 'D' is not a sink of the fate map"):
        select_lineages(("A", "B"), ["D"])
    with pytest.raises(ValueError, match="Lineage 'D' is not in the result"):
        select_lineages(("A", "B"), ["D"], where="in the result")


def test_select_lineages_follow_the_requested_order_and_return_stored_labels():
    sinks = (np.int64(2), np.int64(3), np.int64(4))

    labels, columns = select_lineages(sinks, [4, 2])
    tuple_labels, tuple_columns = select_lineages(["A", "B", "C"], ("C",))

    assert labels == (4, 2)
    assert all(type(label) is np.int64 for label in labels)
    np.testing.assert_array_equal(columns, [2, 0])
    assert tuple_labels == ("C",)
    np.testing.assert_array_equal(tuple_columns, [2])


@pytest.mark.parametrize(
    ("lineages", "error", "message"),
    [
        ("A", TypeError, "lineages must be a list of lineage labels or None"),
        (
            np.array(["A"]),
            TypeError,
            "lineages must be a list of lineage labels or None",
        ),
        ([], ValueError, "lineages must name at least one lineage"),
        (["A", "D"], ValueError, "Lineage 'D' is not a sink of the fate map"),
        (["B", "A", "B"], ValueError, "lineages must not repeat a lineage"),
    ],
)
def test_select_lineages_reject_invalid_requests(lineages, error, message):
    with pytest.raises(error, match=message):
        select_lineages(("A", "B", "C"), lineages)


def test_kish_effective_sizes_follow_the_definition():
    rng = np.random.default_rng(0)
    random = rng.exponential(size=20)
    weights = np.column_stack(
        (np.ones(20), np.eye(20)[3] * 0.4, np.zeros(20), random, np.full(20, 0.05))
    )

    sizes = kish_effective_sizes(weights)

    expected_random = sum(random) ** 2 / sum(value * value for value in random)
    np.testing.assert_allclose(
        sizes, [20.0, 1.0, 0.0, expected_random, 20.0], rtol=1e-13, atol=0.0
    )
    assert LINEAGE_SCRATCH_ITEMSIZE == 8


def test_centered_regressors_are_weighted_centred_after_a_power_of_two_scale():
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.0, 1.0, size=(30, 3))
    regressors = np.column_stack(
        (rng.normal(size=30) * 37.0, np.zeros(30), rng.normal(size=30) * 1e300)
    )

    totals, centered, squares = _centered_regressors(weights, regressors)

    np.testing.assert_allclose(totals, weights.sum(axis=0), rtol=1e-15)
    assert np.isfinite(centered).all() and np.isfinite(squares).all()
    for column in (0, 2):
        magnitude = np.abs(regressors[:, column]).max()
        scale = 2.0 ** -math.frexp(magnitude)[1]
        assert 0.5 <= np.abs(regressors[:, column] * scale).max() < 1.0
        scaled = regressors[:, column] * scale
        mean = (weights[:, column] * scaled).sum() / totals[column]
        np.testing.assert_allclose(centered[:, column], scaled - mean, atol=1e-15)
        assert abs((weights[:, column] * centered[:, column]).sum()) < 1e-13
        np.testing.assert_allclose(
            squares[column],
            (weights[:, column] * (scaled - mean) ** 2).sum(),
            rtol=1e-13,
        )
    np.testing.assert_array_equal(centered[:, 1], 0.0)
    assert squares[1] == 0.0


def test_centered_regressors_need_positive_lineage_weight():
    weights = np.array([[1.0, 0.0], [2.0, 0.0]])

    with pytest.raises(ValueError, match="Every lineage needs a positive total weight"):
        _centered_regressors(weights, np.ones((2, 2)))


def test_weighted_correlations_match_a_dense_weighted_pearson_reference():
    values, weights, regressors = _lineage_block(2)
    expected = _weighted_pearson_table(values, weights, regressors)

    r_values, status = _correlate(values, weights, regressors)
    fortran_r, fortran_status = _correlate(
        np.asfortranarray(values), weights, regressors
    )

    np.testing.assert_array_equal(status, LINEAGE_OK)
    assert status.dtype == np.int8
    np.testing.assert_allclose(r_values, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(fortran_r, r_values)
    np.testing.assert_array_equal(fortran_status, status)


def test_weighted_correlations_with_unit_weights_match_pearsonr():
    rng = np.random.default_rng(3)
    regressor = rng.normal(size=40)
    values = np.column_stack(
        [
            regressor * slope + rng.normal(size=40)
            for slope in (-2.0, -0.1, 0.0, 0.5, 3.0)
        ]
    )

    r_values, status = _correlate(values, np.ones((40, 1)), regressor[:, np.newaxis])

    expected = [stats.pearsonr(column, regressor).statistic for column in values.T]
    np.testing.assert_array_equal(status, LINEAGE_OK)
    np.testing.assert_allclose(r_values[:, 0], expected, rtol=0.0, atol=1e-12)


def test_weighted_correlations_do_not_depend_on_the_scale_of_values_or_regressors():
    values, weights, regressors = _lineage_block(4, n_features=3)
    reference = _weighted_pearson_table(values, weights, regressors)
    # Subnormal values lose low bits; the exact power-of-two rescale of what
    # remains is the reference for them.
    tiny = np.ldexp(values, -1040)
    tiny_reference = _weighted_pearson_table(np.ldexp(tiny, 1040), weights, regressors)

    huge_r, huge_status = _correlate(
        np.ldexp(values, 1000), weights, regressors * 1e300
    )
    tiny_r, tiny_status = _correlate(tiny, weights, regressors)

    np.testing.assert_array_equal(huge_status, LINEAGE_OK)
    np.testing.assert_array_equal(tiny_status, LINEAGE_OK)
    np.testing.assert_allclose(huge_r, reference, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(tiny_r, tiny_reference, rtol=0.0, atol=1e-12)


def test_weighted_correlations_mark_untested_features_per_lineage():
    values, weights, regressors = _status_block()

    r_values, status = _correlate(values, weights, regressors, min_mass=2.0)

    ok, untested = LINEAGE_OK, LINEAGE_UNTESTED
    np.testing.assert_array_equal(
        status,
        [
            # Zero everywhere.
            [untested, untested, untested],
            # Not expressed on lineage 0's cells.
            [untested, ok, untested],
            # Constant on lineage 0's cells.
            [untested, ok, untested],
            # Expressed with exactly min_mass in lineage 0.
            [ok, ok, untested],
            # Expressed with less than min_mass in lineage 0.
            [untested, ok, untested],
            # Spanning one rounding step of its magnitude.
            [untested, ok, untested],
            # Spanning two rounding steps.
            [ok, ok, untested],
        ],
    )
    np.testing.assert_array_equal(r_values[status == untested], 0.0)
    checked = status == ok
    # A span of two rounding steps is tested, but its r is rounding noise.
    checked[6, 0] = False
    for feature, lineage in zip(*np.nonzero(checked), strict=True):
        expected = _weighted_pearson(
            values[:, feature], regressors[:, lineage], weights[:, lineage]
        )
        assert r_values[feature, lineage] == pytest.approx(expected, abs=1e-12)


def test_weighted_correlations_clip_perfect_correlations():
    rng = np.random.default_rng(5)
    regressor = rng.normal(size=25) * 1e-3 + 7.0
    weights = rng.uniform(0.1, 1.0, size=(25, 1))
    values = np.column_stack((regressor * 3.0 + 1.0, -regressor))

    r_values, status = _correlate(values, weights, regressor[:, np.newaxis])

    np.testing.assert_array_equal(status, LINEAGE_OK)
    assert np.all(np.abs(r_values) <= 1.0)
    np.testing.assert_allclose(r_values[:, 0], [1.0, -1.0], rtol=0.0, atol=1e-12)


def _invalid_correlation_inputs(name: str) -> dict:
    values, weights, regressors = _lineage_block(6, n_cells=10, n_features=3)
    labels = ["a", "b", "c"]
    if name == "flat values":
        values = values[:, 0]
    elif name == "flat weights":
        weights = weights[:, 0]
        regressors = regressors[:, 0]
    elif name == "short weights":
        weights = weights[:-1]
        regressors = regressors[:-1]
    elif name == "regressor columns":
        regressors = regressors[:, :2]
    elif name == "labels":
        labels = ["a", "b"]
    elif name == "nan weight":
        weights = weights.copy()
        weights[3, 1] = np.nan
    elif name == "negative weight":
        weights = weights.copy()
        weights[3, 1] = -0.1
    elif name == "infinite regressor":
        regressors = regressors.copy()
        regressors[2, 0] = np.inf
    elif name == "empty lineage":
        weights = weights.copy()
        weights[:, 2] = 0.0
    elif name == "nan value":
        values = values.copy()
        values[4, 1] = np.nan
    return {
        "values": values,
        "weights": weights,
        "regressors": regressors,
        "min_mass": 0.0,
        "nthreads": 1,
        "feature_labels": labels,
    }


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("flat values", "values and weights must be two-dimensional"),
        ("flat weights", "values and weights must be two-dimensional"),
        ("short weights", "weights must have one row per cell"),
        ("regressor columns", "regressors must align with the weights"),
        ("labels", "feature_labels must name every feature"),
        ("nan weight", "weights must be finite and non-negative"),
        ("negative weight", "weights must be finite and non-negative"),
        ("infinite regressor", "regressors must be finite"),
        ("empty lineage", "Every lineage needs a positive total weight"),
        ("nan value", "Feature 'b' contains non-finite normalized values"),
    ],
)
def test_weighted_correlations_reject_invalid_inputs(name, message):
    with pytest.raises(ValueError, match=message):
        weighted_correlations(**_invalid_correlation_inputs(name))


def test_weighted_correlations_restore_the_numba_thread_count():
    values, weights, regressors = _lineage_block(7, n_cells=12, n_features=2)
    previous = numba.get_num_threads()
    start = min(2, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(start)
    try:
        for nthreads in (0, 1, 10_000):
            weighted_correlations(
                values,
                weights,
                regressors,
                min_mass=0.0,
                nthreads=nthreads,
                feature_labels=["a", "b"],
            )
            assert numba.get_num_threads() == start
    finally:
        numba.set_num_threads(previous)


def test_weighted_correlation_kernel_agrees_with_its_python_source():
    values, weights, regressors = _status_block()
    reversed_tail = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    values = np.column_stack((values, reversed_tail, np.full(8, np.nan)))
    totals, centered, squares = _centered_regressors(weights, regressors)

    for regressor_squares in (squares, squares / 4.0):
        # A quarter of the regressor's sum of squares doubles every r, so
        # strong correlations leave [-1, 1] and must be clipped.
        arguments = (
            np.ascontiguousarray(values),
            weights,
            np.ascontiguousarray(centered),
            regressor_squares,
            totals,
            2.0,
            float(np.finfo(np.float64).eps),
        )
        compiled_r, compiled_status = _weighted_correlation_kernel(*arguments)
        python_r, python_status = _weighted_correlation_kernel.py_func(*arguments)
        np.testing.assert_array_equal(compiled_status, python_status)
        np.testing.assert_array_equal(compiled_status[-1], LINEAGE_NONFINITE)
        np.testing.assert_allclose(compiled_r, python_r, rtol=1e-15, atol=0.0)
        assert np.all(np.abs(compiled_r) <= 1.0)
    assert compiled_r[1, 1] == 1.0 and compiled_r[7, 1] == -1.0


def _feature_blocks(
    values: np.ndarray, order: np.ndarray, cuts: tuple[int, ...]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Column blocks laid out as transposed feature-major batches, as streamed."""
    return [
        (np.asfortranarray(values[:, start:stop]), order[start:stop])
        for start, stop in zip(cuts[:-1], cuts[1:], strict=True)
    ]


def test_stream_weighted_correlations_match_one_shot_for_any_partition():
    values, weights, regressors = _lineage_block(8, n_features=7)
    order = np.array([10, 11, 12, 13, 14, 15, 16])
    whole_r, whole_status = _correlate(values, weights, regressors, min_mass=1.0)

    for cuts in ((0, 7), (0, 1, 3, 3, 7), tuple(range(8))):
        r_values, status = stream_weighted_correlations(
            _feature_blocks(values, order, cuts),
            order,
            weights,
            regressors,
            min_mass=1.0,
            nthreads=2,
        )
        np.testing.assert_array_equal(r_values, whole_r)
        np.testing.assert_array_equal(status, whole_status)


def test_stream_weighted_correlations_require_ordered_complete_blocks():
    values, weights, regressors = _lineage_block(9, n_cells=10, n_features=3)
    order = np.array([5, 6, 7])

    def stream(blocks):
        return stream_weighted_correlations(
            blocks, order, weights, regressors, min_mass=0.0, nthreads=1
        )

    first_two = np.asfortranarray(values[:, :2])
    with pytest.raises(RuntimeError, match="must follow the requested features"):
        stream([(first_two, np.array([6, 5]))])
    with pytest.raises(RuntimeError, match="must follow the requested features"):
        stream([(values, order), (np.asfortranarray(values[:, :1]), np.array([8]))])
    with pytest.raises(RuntimeError, match="must cover every feature"):
        stream([(first_two, order[:2])])
    nan_values = values.copy()
    nan_values[0, 2] = np.nan
    with pytest.raises(ValueError, match="Feature 7 contains non-finite"):
        stream(_feature_blocks(nan_values, order, (0, 2, 3)))


def test_check_block_returns_the_next_start():
    order = np.array(["a", "b", "c"])

    assert _check_block(np.array(["b", "c"]), order, 1) == 3
    assert _check_block(np.array([], dtype=str), order, 3) == 3
    with pytest.raises(RuntimeError, match="must follow the requested features"):
        _check_block(np.array(["c"]), order, 1)


def test_kish_p_values_with_unit_weights_match_linregress():
    rng = np.random.default_rng(10)
    regressor = rng.uniform(size=30)
    values = np.column_stack(
        [regressor * slope + rng.normal(size=30) for slope in (0.0, 0.8, -1.5, 4.0)]
    )
    weights = np.ones((30, 1))
    r_values, status = _correlate(values, weights, regressor[:, np.newaxis])

    p_values, updated = kish_p_values(r_values, status, kish_effective_sizes(weights))

    expected = [stats.linregress(regressor, column).pvalue for column in values.T]
    np.testing.assert_array_equal(updated, LINEAGE_OK)
    np.testing.assert_allclose(p_values[:, 0], expected, rtol=1e-10, atol=0.0)


def test_kish_p_values_test_weighted_correlations_on_the_effective_size():
    rng = np.random.default_rng(11)
    r_values = rng.uniform(-0.95, 0.95, size=(6, 2))
    status = np.full((6, 2), LINEAGE_OK, dtype=np.int8)
    sizes = np.array([7.3, 41.0])

    p_values, _ = kish_p_values(r_values, status, sizes)

    # The two-sided t-test of r on d degrees of freedom has p = I_{1 - r^2}(d/2, 1/2).
    expected = betainc((sizes - 2.0) / 2.0, 0.5, 1.0 - r_values**2)
    np.testing.assert_allclose(p_values, expected, rtol=1e-10, atol=0.0)


def test_kish_p_values_skip_small_lineages_and_untested_entries():
    r_values = np.array([[0.5, 0.5, 0.5, 1.0], [0.2, 0.2, 0.2, -1.0]])
    status = np.array(
        [[LINEAGE_OK] * 4, [LINEAGE_OK, LINEAGE_OK, LINEAGE_UNTESTED, LINEAGE_OK]],
        dtype=np.int8,
    )
    original = status.copy()

    p_values, updated = kish_p_values(
        r_values, status, np.array([2.0, 1.5, 10.0, 10.0])
    )

    np.testing.assert_array_equal(status, original)
    np.testing.assert_array_equal(updated[:, :2], LINEAGE_UNTESTED)
    np.testing.assert_array_equal(updated[:, 2], [LINEAGE_OK, LINEAGE_UNTESTED])
    assert np.isnan(p_values[:, :2]).all()
    assert np.isnan(p_values[1, 2]) and np.isfinite(p_values[0, 2])
    assert np.all((p_values[:, 3] >= 0.0) & (p_values[:, 3] < 1e-60))


def test_fisher_z_intervals_match_pearsonr_fisher_intervals():
    rng = np.random.default_rng(12)
    n_cells = 35
    regressor = rng.normal(size=n_cells)
    values = np.column_stack(
        [regressor * slope + rng.normal(size=n_cells) for slope in (-1.0, 0.1, 2.0)]
    )
    r_values, status = _correlate(
        values, np.ones((n_cells, 1)), regressor[:, np.newaxis]
    )

    for level in (0.8, 0.95, 0.99):
        _, low, high, updated = fisher_z_statistics(r_values, status, n_cells, level)
        np.testing.assert_array_equal(updated, LINEAGE_OK)
        for feature, column in enumerate(values.T):
            interval = stats.pearsonr(column, regressor).confidence_interval(
                confidence_level=level, method=None
            )
            assert low[feature, 0] == pytest.approx(interval.low, abs=1e-12)
            assert high[feature, 0] == pytest.approx(interval.high, abs=1e-12)


def test_fisher_z_p_values_follow_the_normal_approximation():
    r_values = np.array([[-0.9, 0.0], [0.05, 0.3], [0.6, 0.999]])
    status = np.full(r_values.shape, LINEAGE_OK, dtype=np.int8)
    n_cells = 28

    p_values, _, _, _ = fisher_z_statistics(r_values, status, n_cells, 0.95)

    expected = [
        [
            math.erfc(abs(math.atanh(r)) * math.sqrt(n_cells - 3) / math.sqrt(2.0))
            for r in row
        ]
        for row in r_values
    ]
    np.testing.assert_allclose(p_values, expected, rtol=1e-12, atol=1e-300)


def test_fisher_z_statistics_skip_untested_entries_tiny_samples_and_clip_ones():
    r_values = np.array([[1.0, -1.0, 0.4]])
    status = np.array([[LINEAGE_OK, LINEAGE_OK, LINEAGE_UNTESTED]], dtype=np.int8)

    p_values, low, high, updated = fisher_z_statistics(r_values, status, 50, 0.9)
    tiny = fisher_z_statistics(r_values, status, 3, 0.9)

    np.testing.assert_array_equal(updated, status)
    assert np.isfinite(p_values[0, :2]).all() and np.isfinite(low[0, :2]).all()
    assert 0.99 < low[0, 0] <= high[0, 0] <= 1.0
    assert -1.0 <= low[0, 1] <= high[0, 1] < -0.99
    assert np.isnan([p_values[0, 2], low[0, 2], high[0, 2]]).all()
    for array in tiny[:3]:
        assert np.isnan(array).all()
    np.testing.assert_array_equal(tiny[3], LINEAGE_UNTESTED)


def test_adjust_lineage_p_values_apply_benjamini_hochberg_within_each_lineage():
    rng = np.random.default_rng(13)
    p_values = rng.uniform(size=(12, 3)) ** 2
    status = np.full(p_values.shape, LINEAGE_OK, dtype=np.int8)
    status[[1, 5], 0] = LINEAGE_UNTESTED
    status[2, 1] = LINEAGE_NONFINITE
    p_values[7, 1] = np.nan
    status[:, 2] = LINEAGE_UNTESTED

    adjusted = adjust_lineage_p_values(p_values, status)

    for lineage in range(2):
        tested = (status[:, lineage] == LINEAGE_OK) & np.isfinite(p_values[:, lineage])
        expected = _benjamini_hochberg(p_values[tested, lineage])
        _, statsmodels_expected, _, _ = multipletests(
            p_values[tested, lineage], method="fdr_bh"
        )
        np.testing.assert_allclose(adjusted[tested, lineage], expected, rtol=1e-13)
        np.testing.assert_allclose(adjusted[tested, lineage], statsmodels_expected)
        assert np.isnan(adjusted[~tested, lineage]).all()
    assert np.isnan(adjusted[:, 2]).all()


def test_equal_mass_bin_operator_shares_cells_between_equal_mass_bins():
    # Cells 0 and 2 tie and keep their positions; cell 3 of lineage 0 and
    # cells 1 and 2 of lineage 1 straddle cuts between bins of mass two.
    ordering = np.array([2.0, 1.0, 2.0, 3.0])
    weights = np.array([[1.0, 0.0], [1.0, 3.0], [1.0, 1.5], [3.0, 1.5]])

    operator, bin_ordering, totals = equal_mass_bin_operator(ordering, weights, 3)

    expected = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    np.testing.assert_allclose(operator.toarray(), expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        bin_ordering, [[1.5, 2.5, 3.0], [1.0, 1.5, 2.75]], rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(totals, [6.0, 6.0])
    assert operator.shape == (6, 4)


def test_equal_mass_bin_operator_spreads_a_heavy_cell_over_several_bins():
    ordering = np.array([0.0, 1.0, 2.0])
    weights = np.array([[1.0], [5.0], [1.0]])

    operator, _, _ = equal_mass_bin_operator(ordering, weights, 7)

    dense = operator.toarray()
    np.testing.assert_allclose(dense[1:6, 1], 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(dense[[0, 6], [0, 2]], 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(dense.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_equal_mass_bin_operator_matches_brute_force_overlaps(seed):
    rng = np.random.default_rng(100 + seed)
    n_cells = int(rng.integers(6, 40))
    n_lineages = int(rng.integers(1, 4))
    weights = rng.exponential(size=(n_cells, n_lineages))
    weights[rng.random(weights.shape) < 0.3] = 0.0
    weights[0] = 1.0
    # Few distinct values, so many cells tie.
    ordering = rng.integers(0, 5, size=n_cells).astype(np.float64)
    n_bins = int(rng.integers(1, int(weights.sum(axis=0).min()) + 1))
    features = rng.normal(size=(n_cells, 3))

    operator, bin_ordering, totals = equal_mass_bin_operator(ordering, weights, n_bins)

    expected = _brute_force_bin_operator(ordering, weights, n_bins)
    np.testing.assert_allclose(operator.toarray(), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(operator.sum(axis=1)).ravel(), 1.0, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        bin_ordering, (expected @ ordering).reshape(n_lineages, n_bins), atol=1e-12
    )
    np.testing.assert_allclose(operator @ features, expected @ features, atol=1e-12)
    np.testing.assert_allclose(totals, weights.sum(axis=0), rtol=1e-15)
    assert not operator[:, np.flatnonzero((weights == 0).all(axis=1))].count_nonzero()


@pytest.mark.parametrize(
    ("ordering", "weights", "message"),
    [
        (np.zeros(3), np.ones((4, 1)), "ordering must have one value per cell"),
        (np.array([0.0, np.nan, 1.0]), np.ones((3, 1)), "ordering must be finite"),
        (
            np.zeros(3),
            np.array([[1.0], [np.inf], [1.0]]),
            "weights must be finite and non-negative",
        ),
        (
            np.zeros(3),
            np.array([[1.0], [-1.0], [3.0]]),
            "weights must be finite and non-negative",
        ),
        (
            np.zeros(3),
            np.array([[3.0, 1.0], [3.0, 1.0], [3.0, 0.5]]),
            r"Lineage 1 carries a total weight of 2\.5, below n_bins=3; "
            "request fewer bins",
        ),
    ],
)
def test_equal_mass_bin_operator_rejects_invalid_inputs(ordering, weights, message):
    with pytest.raises(ValueError, match=message):
        equal_mass_bin_operator(ordering, weights, 3)


def test_bin_overlaps_agrees_with_its_python_source():
    rng = np.random.default_rng(14)
    weights = rng.exponential(size=(9, 2))
    weights[[2, 5], 0] = 0.0
    weights[4, 1] = 6.0
    totals = weights.sum(axis=0)
    order = np.argsort(rng.normal(size=9), kind="stable").astype(np.int64)
    outputs = []
    for kernel in (_bin_overlaps, _bin_overlaps.py_func):
        counts = np.zeros(9, dtype=np.int64)
        kernel(
            order, weights, totals, 4, counts, np.empty(0, np.int64), np.empty(0), False
        )
        indptr = np.concatenate(([0], np.cumsum(counts)))
        cursor = indptr[:-1].copy()
        indices = np.full(indptr[-1], -1, dtype=np.int32)
        data = np.zeros(indptr[-1])
        kernel(order, weights, totals, 4, cursor, indices, data, True)
        outputs.append((counts, cursor, indices, data))
        # Every slot is written, and each cursor ends where the next column
        # starts.
        np.testing.assert_array_equal(cursor, indptr[1:])
        assert (indices >= 0).all()

    for compiled_array, python_array in zip(*outputs, strict=True):
        np.testing.assert_array_equal(compiled_array, python_array)
    counts, _, indices, _ = outputs[0]
    # Cells without weight in the first lineage enter only the second one,
    # and the heavy cell covers several of the second lineage's bins.
    assert counts[2] == counts[5] == 1
    assert counts[4] >= 3
    assert set(indices.tolist()) == set(range(8))


def test_equal_mass_bin_operator_is_canonical_with_32_bit_indices():
    rng = np.random.default_rng(26)
    ordering = rng.uniform(size=60)
    weights = rng.dirichlet(np.ones(3), size=60)

    operator, _, _ = equal_mass_bin_operator(ordering, weights, 7)

    assert operator.indices.dtype == operator.indptr.dtype == np.int32
    assert operator.has_canonical_format
    expected = _brute_force_bin_operator(ordering, weights, 7)
    np.testing.assert_allclose(operator.toarray(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("n_cells", "n_lineages", "n_bins"), [(60, 3, 7), (500, 1, 50), (1, 4, 1)]
)
def test_bin_operator_bytes_bound_the_arrays_built(n_cells, n_lineages, n_bins):
    rng = np.random.default_rng(n_cells)
    weights = rng.uniform(1.0, 2.0, size=(n_cells, n_lineages))

    operator, _, _ = equal_mass_bin_operator(rng.uniform(size=n_cells), weights, n_bins)

    built = operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes
    # The order, the cursor, and the int64 column pointers live alongside.
    assert built + 2 * n_cells * 8 <= bin_operator_bytes(n_cells, n_lineages, n_bins)
    assert bin_operator_bytes(n_cells, n_lineages, n_bins) == (
        3 * (n_cells + 1) * 8 + n_lineages * (n_cells + n_bins) * 16
    )


def test_stream_weighted_correlations_prepare_the_regressors_once(monkeypatch):
    values, weights, regressors = _lineage_block(19, n_features=6)
    order = np.arange(6)
    calls = []
    original = lineages_module._centered_regressors

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(lineages_module, "_centered_regressors", counted)

    r_values, status = stream_weighted_correlations(
        _feature_blocks(values, order, (0, 2, 4, 6)),
        order,
        weights,
        regressors,
        min_mass=1.0,
        nthreads=1,
    )

    assert len(calls) == 1
    whole_r, whole_status = _correlate(values, weights, regressors, min_mass=1.0)
    np.testing.assert_array_equal(status, whole_status)
    np.testing.assert_allclose(r_values, whole_r, rtol=1e-13, atol=1e-15)


def test_bin_smoothing_matrix_is_a_truncated_row_normalized_gaussian():
    for n_bins, sigma in ((12, 0.6), (30, 2.5), (5, 0.7)):
        kernel = bin_smoothing_matrix(n_bins, sigma)
        np.testing.assert_allclose(
            kernel, _truncated_gaussian_smoother(n_bins, sigma), rtol=1e-13, atol=1e-16
        )
        np.testing.assert_allclose(kernel.sum(axis=1), 1.0, rtol=1e-14)
    narrow = bin_smoothing_matrix(12, 0.6)
    # The radius is ceil(4 * 0.6) = 3 bins.
    assert narrow[0, 3] > 0.0 and narrow[0, 4] == 0.0
    assert narrow[6, 9] > 0.0 and narrow[6, 10] == 0.0


def test_bin_smoothing_without_smoothing_leaves_bin_means_unchanged():
    rng = np.random.default_rng(15)
    ordering = rng.uniform(size=20)
    weights = rng.dirichlet(np.ones(2), size=20)
    operator, _, _ = equal_mass_bin_operator(ordering, weights, 5)
    values = rng.normal(size=(20, 3))

    assert bin_smoothing_matrix(5, 0.0) is None
    np.testing.assert_allclose(
        lineage_profiles(values, operator, None, 2),
        lineage_profiles(values, operator, np.eye(5), 2),
        rtol=0.0,
        atol=1e-15,
    )


def test_lineage_profiles_average_then_smooth_bins_of_each_lineage():
    rng = np.random.default_rng(16)
    ordering = rng.uniform(size=30)
    weights = rng.dirichlet(np.ones(3), size=30)
    n_bins = 6
    operator, _, _ = equal_mass_bin_operator(ordering, weights, n_bins)
    smoothing = bin_smoothing_matrix(n_bins, 1.2)
    values = rng.gamma(1.0, size=(30, 4))

    profiles = lineage_profiles(values, operator, smoothing, 3)

    means = operator.toarray() @ values
    expected = np.zeros((4, 3, n_bins))
    for feature in range(4):
        for lineage in range(3):
            for bin_index in range(n_bins):
                expected[feature, lineage, bin_index] = sum(
                    smoothing[bin_index, other]
                    * means[lineage * n_bins + other, feature]
                    for other in range(n_bins)
                )
    assert profiles.flags.c_contiguous
    np.testing.assert_allclose(profiles, expected, rtol=1e-13, atol=1e-15)


def test_stream_lineage_profiles_match_one_shot_profiles_and_means():
    rng = np.random.default_rng(17)
    ordering = rng.uniform(size=25)
    weights = rng.dirichlet(np.ones(2), size=25)
    operator, _, _ = equal_mass_bin_operator(ordering, weights, 4)
    smoothing = bin_smoothing_matrix(4, 0.8)
    values = rng.gamma(1.0, size=(25, 5))
    order = np.array(["g0", "g1", "g2", "g3", "g4"])
    whole = lineage_profiles(values, operator, smoothing, 2)

    for cuts in ((0, 5), (0, 2, 2, 5), tuple(range(6))):
        blocks = [
            (values[:, start:stop], order[start:stop])
            for start, stop in zip(cuts[:-1], cuts[1:], strict=True)
        ]
        profiles, means = stream_lineage_profiles(blocks, order, operator, smoothing, 2)
        np.testing.assert_allclose(profiles, whole, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(means, values.mean(axis=0), rtol=1e-14)


def test_stream_lineage_profiles_require_finite_ordered_complete_blocks():
    operator, _, _ = equal_mass_bin_operator(np.arange(6.0), np.ones((6, 1)), 2)
    values = np.ones((6, 3))
    order = np.array(["g0", "g1", "g2"])

    def stream(blocks):
        return stream_lineage_profiles(blocks, order, operator, None, 1)

    with pytest.raises(RuntimeError, match="must follow the requested features"):
        stream([(values[:, :1], np.array(["g1"]))])
    with pytest.raises(RuntimeError, match="must cover every feature"):
        stream([(values[:, :2], order[:2])])
    nan_values = values.copy()
    nan_values[2, 2] = np.inf
    with pytest.raises(ValueError, match="Feature .*g2.* contains non-finite"):
        stream([(nan_values[:, :1], order[:1]), (nan_values[:, 1:], order[1:])])


def test_stream_lineage_profiles_name_the_non_finite_feature_like_correlations():
    operator, _, _ = equal_mass_bin_operator(np.arange(6.0), np.ones((6, 1)), 2)
    values = np.ones((6, 3))
    values[2, 2] = np.nan

    for order, label in ((np.array(["g0", "g1", "g2"]), "'g2'"), (np.arange(3), "2")):
        with pytest.raises(ValueError) as caught:
            stream_lineage_profiles([(values, order)], order, operator, None, 1)
        assert str(caught.value) == (
            f"Feature {label} contains non-finite normalized values"
        )


def test_streams_name_non_finite_features_by_their_names():
    operator, _, _ = equal_mass_bin_operator(np.arange(6.0), np.ones((6, 1)), 2)
    values = np.ones((6, 3))
    values[2, 1] = np.inf
    order = np.array([10, 11, 12])
    names = np.array(["Gcg", "Ins1", "Sst"])
    blocks = [(values[:, :1], order[:1]), (values[:, 1:], order[1:])]
    _, weights, regressors = _lineage_block(27, n_cells=6, n_features=3)

    with pytest.raises(ValueError, match="Feature 'Ins1' contains non-finite"):
        stream_lineage_profiles(blocks, order, operator, None, 1, feature_names=names)
    with pytest.raises(ValueError, match="Feature 'Ins1' contains non-finite"):
        stream_weighted_correlations(
            blocks,
            order,
            weights,
            regressors,
            min_mass=0.0,
            nthreads=1,
            feature_names=names,
        )


def test_streams_need_one_name_per_feature():
    operator, _, _ = equal_mass_bin_operator(np.arange(6.0), np.ones((6, 1)), 2)
    _, weights, regressors = _lineage_block(28, n_cells=6, n_features=3)
    order = np.arange(3)

    with pytest.raises(ValueError, match="feature_names must name every feature"):
        stream_lineage_profiles(
            [], order, operator, None, 1, feature_names=np.array(["a", "b"])
        )
    with pytest.raises(ValueError, match="feature_names must name every feature"):
        stream_weighted_correlations(
            [],
            order,
            weights,
            regressors,
            min_mass=0.0,
            nthreads=1,
            feature_names=["a"],
        )


def test_bin_sums_kernel_matches_its_python_source_and_a_dense_product():
    rng = np.random.default_rng(23)
    ordering = rng.uniform(size=40)
    weights = rng.dirichlet(np.ones(3), size=40)
    operator, _, _ = equal_mass_bin_operator(ordering, weights, 5)
    values = rng.gamma(1.0, size=(40, 6))
    values[rng.uniform(size=values.shape) < 0.6] = 0.0
    values[:, 1] = 0.0
    values[7, 3] = np.inf
    values[9, 4] = np.nan
    feature_major = np.ascontiguousarray(values.T)
    arguments = (
        feature_major,
        operator.indptr,
        operator.indices,
        operator.data,
        operator.shape[0],
    )

    compiled = _bin_sums(*arguments)
    python = _bin_sums.py_func(*arguments)

    finite = np.array([True, True, True, False, False, True])
    for products, totals, flags in (compiled, python):
        np.testing.assert_array_equal(flags, finite)
        np.testing.assert_allclose(
            products[finite], (operator.toarray() @ values[:, finite]).T, rtol=1e-13
        )
        np.testing.assert_allclose(
            totals[finite], values[:, finite].sum(axis=0), rtol=1e-13
        )
        np.testing.assert_array_equal(products[1], 0.0)
        assert totals[1] == 0.0
    np.testing.assert_allclose(compiled[0][finite], python[0][finite], rtol=1e-15)


def test_lineage_profiles_agree_across_layouts_formats_and_threads():
    rng = np.random.default_rng(24)
    ordering = rng.uniform(size=50)
    weights = rng.dirichlet(np.ones(2), size=50)
    operator, _, _ = equal_mass_bin_operator(ordering, weights, 5)
    smoothing = bin_smoothing_matrix(5, 1.0)
    values = rng.gamma(1.0, size=(50, 7))
    values[rng.uniform(size=values.shape) < 0.5] = 0.0
    expected = lineage_profiles(values, operator, smoothing, 2)
    previous = numba.get_num_threads()

    for layout in (np.ascontiguousarray(values), np.asfortranarray(values)):
        for matrix in (operator, operator.tocsr(), operator.tocoo()):
            for nthreads in (1, 2, 10_000):
                profiles = lineage_profiles(
                    layout, matrix, smoothing, 2, nthreads=nthreads
                )
                np.testing.assert_allclose(profiles, expected, rtol=1e-14, atol=0.0)
                assert numba.get_num_threads() == previous
    np.testing.assert_allclose(
        lineage_profiles(values.astype(np.float32), operator, smoothing, 2),
        lineage_profiles(
            values.astype(np.float32).astype(np.float64), operator, smoothing, 2
        ),
        rtol=1e-14,
    )


def test_lineage_profiles_require_one_value_row_per_operator_column():
    operator, _, _ = equal_mass_bin_operator(np.arange(6.0), np.ones((6, 1)), 2)

    for values in (np.ones((5, 2)), np.ones(6)):
        with pytest.raises(ValueError, match="one row per operator column"):
            lineage_profiles(values, operator, None, 1)
    with pytest.raises(ValueError, match="one row per operator column"):
        stream_lineage_profiles(
            [(np.ones((5, 1)), np.array(["g0"]))],
            np.array(["g0"]),
            operator,
            None,
            1,
        )


def test_stream_lineage_profiles_keep_the_operator_and_restore_threads():
    rng = np.random.default_rng(25)
    ordering = rng.uniform(size=30)
    weights = rng.dirichlet(np.ones(2), size=30)
    operator, _, _ = equal_mass_bin_operator(ordering, weights, 3)
    values = rng.gamma(1.0, size=(30, 4))
    order = np.arange(4)
    blocks = _feature_blocks(values, order, (0, 3, 4))
    previous = numba.get_num_threads()

    profiles, means = stream_lineage_profiles(
        blocks, order, operator, None, 2, nthreads=2
    )

    assert numba.get_num_threads() == previous
    np.testing.assert_allclose(
        profiles, lineage_profiles(values, operator, None, 2), rtol=1e-14
    )
    np.testing.assert_allclose(means, values.mean(axis=0), rtol=1e-14)


def test_equal_mass_bin_operator_is_stored_by_cell():
    operator, _, _ = equal_mass_bin_operator(np.arange(8.0), np.ones((8, 2)), 4)

    assert operator.format == "csc"
    assert operator.shape == (8, 8)


def test_finalize_lineage_profiles_z_scale_each_feature_over_all_lineages():
    rng = np.random.default_rng(18)
    profiles = (
        rng.normal(size=(5, 3, 4))
        * np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, np.newaxis, np.newaxis]
    )
    profiles[2] = 7.0
    original = profiles.copy()
    mean_expression = np.array([0.5, 0.05, 0.5, 0.5, 0.11])

    scaled, valid = finalize_lineage_profiles(
        profiles, mean_expression, min_expression=0.1, z_scale=True
    )

    np.testing.assert_array_equal(valid, [True, False, False, True, True])
    np.testing.assert_array_equal(profiles, original)
    for feature in np.flatnonzero(valid):
        flat = original[feature].ravel()
        expected = (flat - flat.mean()) / flat.std()
        np.testing.assert_allclose(scaled[feature].ravel(), expected, rtol=1e-13)
        assert scaled[feature].mean() == pytest.approx(0.0, abs=1e-14)
        assert scaled[feature].std() == pytest.approx(1.0, rel=1e-13)
    np.testing.assert_array_equal(scaled[~valid], 0.0)


def test_finalize_lineage_profiles_without_scaling_keep_valid_rows():
    profiles = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0 + 1e-12]],
        ]
    )

    kept, valid = finalize_lineage_profiles(
        profiles, np.array([1.0, 1.0, 1.0]), min_expression=0.0, z_scale=False
    )
    none_kept, none_valid = finalize_lineage_profiles(
        profiles[1:2], np.array([1.0]), min_expression=0.0, z_scale=True
    )

    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_array_equal(kept[0], profiles[0])
    np.testing.assert_array_equal(kept[2], profiles[2])
    np.testing.assert_array_equal(kept[1], 0.0)
    assert not none_valid.any()
    np.testing.assert_array_equal(none_kept, 0.0)


def test_finalize_lineage_profiles_reject_non_finite_profiles():
    profiles = np.zeros((2, 1, 3))
    profiles[1, 0, 2] = np.nan

    with pytest.raises(ValueError, match="Lineage profiles contain non-finite values"):
        finalize_lineage_profiles(
            profiles, np.ones(2), min_expression=0.0, z_scale=False
        )
