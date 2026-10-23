"""Ambient RNA computations checked against hand-computed and independent references.

The references follow the description of SoupX in Young and Behjati (2020),
GigaScience 9:giaa151, and its Supplementary Methods, and the rounding follows
systematic sampling with probabilities proportional to size (Madow 1949,
doi:10.1214/aoms/1177729988); R is not installed here, so SoupX itself is not
run.
"""

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests

from cytearc.quality_control import cell_calling
from cytearc.quality_control.ambient import (
    _round_up,
    _row_uniforms,
    _splitmix64,
    estimate_contamination,
    estimation_cells,
    expressing_cells,
    gene_set_evidence,
    remove_ambient_counts,
    soup_fractions,
)

_MASK = 2**64 - 1


def _mix(value: int) -> int:
    """The SplitMix64 finaliser of Steele, Lea and Flood (2014), on Python integers."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return value ^ (value >> 31)


def _draw(seed: int, key: int) -> float:
    """The rounding draw of a row: the first draw of its SplitMix64 stream.

    The stream mixes the seed, the salt "AmbientR" in ASCII, and the row key,
    as cell calling's streams mix a seed and an iteration.
    """
    stream = _mix(
        _mix(seed ^ 0x416D6269656E7452) ^ _mix((key + 0x9E3779B97F4A7C15) & _MASK)
    )
    return (_mix(stream) >> 11) / 2**53


def _poisson_upper_tail(observed: int, mean: float) -> float:
    """P(X >= observed) for X ~ Poisson(mean), from the probability mass function."""
    return 1.0 - sum(
        math.exp(-mean) * mean**count / math.factorial(count)
        for count in range(observed)
    )


def _continuous_removal(counts: np.ndarray, fractions, soup) -> np.ndarray:
    """Solve sum(min(n_g, t * b_g)) = min(rho * N, S) for each row by bisection.

    This is the allocation that the Supplementary Methods describe: genes are
    emptied in order of n_g / b_g and the rest is shared in proportion to
    b_g, written independently of the sorted water-filling under test.
    """
    soup = np.asarray(soup, dtype=np.float64)
    removal = np.zeros(counts.shape, dtype=np.float64)
    for row in range(counts.shape[0]):
        held = counts[row].astype(np.float64)
        genes = (held > 0) & (soup > 0)
        removable = held[genes].sum()
        target = min(fractions[row] * held.sum(), removable)
        if target <= 0:
            continue
        if target >= removable:
            removal[row, genes] = held[genes]
            continue

        def excess(level: float, row_held=held[genes], weights=soup[genes]) -> float:
            return float(np.minimum(row_held, level * weights).sum() - target)

        upper = float((held[genes] / soup[genes]).max())
        level = brentq(excess, 0.0, upper, xtol=1e-13, rtol=1e-15)
        removal[row, genes] = np.minimum(held[genes], level * soup[genes])
    return removal


def _systematic_removal(continuous: np.ndarray, seed: int, keys) -> np.ndarray:
    """Round each row's removal by systematic sampling, from its definition.

    Gene j of a row, in gene order, loses floor(t_j) + floor(C_j + u) -
    floor(C_(j-1) + u), where C_j sums the fractional parts of t up to j and
    u is the row's draw.
    """
    removal = np.zeros(continuous.shape, dtype=np.int64)
    for row, key in enumerate(keys):
        u = _draw(seed, int(key))
        running = 0.0
        for gene in np.flatnonzero(continuous[row] > 0):
            whole = math.floor(continuous[row, gene])
            before = running
            running += continuous[row, gene] - whole
            removal[row, gene] = (
                whole + math.floor(running + u) - math.floor(before + u)
            )
    return removal


def _removed(counts: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    return counts.astype(np.int64) - corrected.astype(np.int64)


def test_soup_fractions_are_each_genes_share_of_the_empty_droplet_counts():
    fractions = soup_fractions(np.array([6, 0, 2, 0, 0, 12], dtype=np.uint64))

    np.testing.assert_array_equal(fractions, [0.3, 0.0, 0.1, 0.0, 0.0, 0.6])
    assert fractions.dtype == np.float64


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.zeros(4), "hold no counts"),
        (np.array([1.0, -1.0]), "finite and non-negative"),
        (np.array([1.0, np.nan]), "finite and non-negative"),
        (np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_soup_fractions_reject_invalid_counts(counts, message):
    with pytest.raises(ValueError, match=message):
        soup_fractions(counts)


def test_gene_set_evidence_is_the_poisson_tail_at_the_largest_contamination():
    observed = np.array([[0, 3], [5, 1], [2, 0]])
    totals = np.array([100.0, 40.0, 10.0])
    set_soup = np.array([0.02, 0.05])

    expected, p_values = gene_set_evidence(
        observed, totals, set_soup, max_contamination=0.5
    )

    np.testing.assert_allclose(expected, [[2.0, 5.0], [0.8, 2.0], [0.2, 0.5]])
    reference = [
        [_poisson_upper_tail(int(n), 0.5 * mean) for n, mean in zip(row, means)]
        for row, means in zip(observed, expected)
    ]
    np.testing.assert_allclose(p_values, reference, rtol=1e-12, atol=1e-15)
    # A cell without counts of a set is certainly consistent with the soup.
    assert p_values[0, 0] == 1.0 and p_values[2, 1] == 1.0


def test_gene_set_evidence_defaults_to_pure_soup():
    expected, p_values = gene_set_evidence([[4]], [10.0], [0.3])

    assert expected[0, 0] == pytest.approx(3.0)
    assert p_values[0, 0] == pytest.approx(_poisson_upper_tail(4, 3.0), rel=1e-12)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (([[1]], [1.0], [0.1]), None, None),
        (([[1]], [1.0], [0.1], 0.0), ValueError, "greater than 0"),
        (([[1]], [1.0], [0.1], 1.5), ValueError, "at most 1"),
        (([[1]], [1.0], [0.1], True), TypeError, "real number"),
        (([1], [1.0], [0.1]), ValueError, "cells by gene sets"),
        (([[1]], [1.0, 2.0], [0.1]), ValueError, "cells by gene sets"),
        (([[1]], [1.0], [0.1, 0.2]), ValueError, "cells by gene sets"),
        (([[-1]], [1.0], [0.1]), ValueError, "observed must be finite"),
        (([[1]], [np.inf], [0.1]), ValueError, "totals must be finite"),
        (([[1]], [1.0], [-0.1]), ValueError, "set_soup must be finite"),
    ],
)
def test_gene_set_evidence_validates_its_arguments(arguments, error, message):
    observed, totals, set_soup, *rest = arguments
    keywords = {"max_contamination": rest[0]} if rest else {}
    if error is None:
        gene_set_evidence(observed, totals, set_soup, **keywords)
        return
    with pytest.raises(error, match=message):
        gene_set_evidence(observed, totals, set_soup, **keywords)


def test_expressing_cells_apply_benjamini_hochberg_within_each_gene_set():
    # Adjusted values of the first set: 0.004, 0.04, 0.04, 0.5.
    p_values = np.array(
        [[0.001, 0.9], [0.02, 0.9], [0.03, 0.01], [0.5, 0.9]], dtype=np.float64
    )

    np.testing.assert_array_equal(
        expressing_cells(p_values, fdr=0.05),
        [[True, False], [True, False], [True, True], [False, False]],
    )
    np.testing.assert_array_equal(
        expressing_cells(p_values, fdr=0.03)[:, 0], [True, False, False, False]
    )
    # The bound is strict: adjusted values 0.5 and 0.5 are not below 0.5.
    np.testing.assert_array_equal(
        expressing_cells([[0.25], [0.5]], fdr=0.5), [[False], [False]]
    )
    for column in range(2):
        reference = multipletests(p_values[:, column], method="fdr_bh")[1] < 0.05
        np.testing.assert_array_equal(expressing_cells(p_values)[:, column], reference)


def test_expressing_cells_of_no_cells_are_empty():
    expressing = expressing_cells(np.zeros((0, 2)))

    # A boolean mask, as callers invert and index with it.
    assert expressing.shape == (0, 2) and expressing.dtype == np.bool_


@pytest.mark.parametrize(
    ("p_values", "fdr", "message"),
    [
        ([[0.5]], 0.0, "greater than 0"),
        ([[0.5]], 1.0, "less than 1"),
        ([0.5], 0.05, "cells by gene sets"),
        ([[1.5]], 0.05, "between 0 and 1"),
        ([[np.nan]], 0.05, "between 0 and 1"),
    ],
)
def test_expressing_cells_validate_their_arguments(p_values, fdr, message):
    with pytest.raises(ValueError, match=message):
        expressing_cells(p_values, fdr=fdr)


def test_estimation_cells_leave_out_expressing_cells_and_their_clusters():
    expressing = np.array(
        [[False, False], [True, False], [False, False], [False, False], [False, True]]
    )
    clusters = np.array(["a", "a", "b", "b", "c"])

    np.testing.assert_array_equal(estimation_cells(expressing), ~expressing)
    np.testing.assert_array_equal(
        estimation_cells(expressing, clusters),
        [[False, True], [False, True], [True, True], [True, True], [True, False]],
    )


def test_estimation_cells_of_no_cells_are_empty():
    used = estimation_cells(np.zeros((0, 1), dtype=bool), np.array([], dtype=str))
    assert used.shape == (0, 1) and used.dtype == np.bool_


def test_estimation_cells_validate_their_arguments():
    with pytest.raises(ValueError, match="cells by gene sets"):
        estimation_cells(np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="one label per cell"):
        estimation_cells(np.zeros((3, 1), dtype=bool), ["a", "b"])


def test_estimate_contamination_is_the_ratio_of_observed_to_pure_soup_counts():
    observed = np.array([[2.0, 30.0], [1.0, 5.0], [4.0, 1.0], [3.0, 0.0]])
    expected = np.array([[20.0, 40.0], [10.0, 50.0], [40.0, 10.0], [30.0, 0.0]])
    used = np.array([[True, False], [True, True], [True, True], [True, False]])

    fit = estimate_contamination(observed, expected, used)

    # (2 + 1 + 4 + 3 + 5 + 1) / (20 + 10 + 40 + 30 + 50 + 10)
    assert fit.fraction == pytest.approx(16 / 160, rel=1e-15)
    np.testing.assert_allclose(fit.set_fractions, [10 / 100, 6 / 60])
    np.testing.assert_array_equal(fit.cell_fractions, np.full(4, fit.fraction))
    assert fit.cluster_labels is None and fit.cluster_fractions is None


def test_estimate_contamination_per_cluster_uses_each_clusters_own_cells():
    observed = np.array([[2.0, 30.0], [1.0, 5.0], [4.0, 1.0], [3.0, 0.0]])
    expected = np.array([[20.0, 40.0], [10.0, 50.0], [40.0, 10.0], [30.0, 0.0]])
    used = np.array([[True, False], [True, True], [True, True], [True, False]])
    clusters = np.array([10, 2, 10, 2])

    fit = estimate_contamination(
        observed, expected, used, clusters=clusters, level="cluster"
    )

    # Clusters in natural order: 2 holds rows 1 and 3, 10 holds rows 0 and 2.
    np.testing.assert_array_equal(fit.cluster_labels, [2, 10])
    np.testing.assert_allclose(
        fit.cluster_fractions,
        [(1 + 5 + 3) / (10 + 50 + 30), (2 + 4 + 1) / (20 + 40 + 10)],
    )
    np.testing.assert_allclose(
        fit.cell_fractions,
        fit.cluster_fractions[[1, 0, 1, 0]],
    )
    assert fit.fraction == pytest.approx(16 / 160, rel=1e-15)


def test_estimate_contamination_marks_a_gene_set_without_evidence():
    # Cells without counts expect no soup counts of the second set.
    fit = estimate_contamination(
        [[1.0, 0.0], [2.0, 0.0]],
        [[10.0, 0.0], [10.0, 0.0]],
        [[True, True], [True, False]],
    )

    assert fit.fraction == pytest.approx(0.15)
    assert fit.set_fractions[0] == pytest.approx(0.15)
    assert np.isnan(fit.set_fractions[1])


def test_estimate_contamination_refuses_missing_evidence():
    with pytest.raises(ValueError, match="No cell can measure the contamination"):
        estimate_contamination([[3.0]], [[10.0]], [[False]])
    with pytest.raises(ValueError, match="clusters have no cell .*: b"):
        estimate_contamination(
            [[3.0], [4.0]],
            [[10.0], [10.0]],
            [[True], [False]],
            clusters=["a", "b"],
            level="cluster",
        )


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"level": "cell"}, "'global' or 'cluster'"),
        ({"level": "cluster"}, "needs the cluster of each cell"),
        ({"level": "cluster", "clusters": ["a"]}, "one label per cell"),
        ({"set_names": ["a", "b"]}, "one name per gene set"),
    ],
)
def test_estimate_contamination_validates_its_arguments(keywords, message):
    with pytest.raises(ValueError, match=message):
        estimate_contamination(
            [[1.0], [1.0]], [[2.0], [2.0]], [[True], [True]], **keywords
        )
    with pytest.raises(ValueError, match="cells by gene sets"):
        estimate_contamination([[1.0]], [[2.0], [2.0]], [[True]])


def _zero_estimate_message(
    subject: str, cells: str, genes: str, expected: str, bound: str, consequence: str
) -> str:
    return (
        f"{subject} is 0: no count of {genes} was observed in the {cells} "
        f"measuring it, where pure soup with their totals would give {expected}. "
        f"A count of 0 puts the fraction below {bound} (the one-sided 95% "
        f"Poisson bound, 3 / {expected}), not at 0, so {consequence}. Check that "
        "the gene sets and the soup profile suit these cells."
    )


def test_estimate_contamination_warns_about_each_estimate_of_zero():
    # Pure soup would give the measuring cells 30 counts of the first set,
    # cluster b 40 counts of both sets, and cluster c 4, but they hold none.
    observed = np.array([[0.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    expected = np.array([[10.0, 20.0], [12.0, 10.0], [8.0, 10.0], [4.0, 4.0]])
    used = np.array([[True, True], [True, True], [True, True], [False, True]])

    with pytest.warns(UserWarning) as caught:
        fit = estimate_contamination(
            observed,
            expected,
            used,
            clusters=np.array(["a", "b", "b", "c"]),
            level="cluster",
            set_names=["haemoglobin", "immunoglobulin"],
        )

    # The bounds are -log(0.05) / 30, / 40, and / 4.
    assert [str(warning.message) for warning in caught] == [
        _zero_estimate_message(
            "The contamination estimate from gene set 'haemoglobin'",
            "3 cells",
            "the set",
            "30",
            "0.0999",
            "a correction with an estimate from this gene set alone would leave "
            "4 cells uncorrected",
        ),
        _zero_estimate_message(
            "The contamination estimate of cluster 'b'",
            "2 cells",
            "the measured gene sets",
            "40",
            "0.0749",
            "a correction at the cluster level would leave its 2 cells uncorrected",
        ),
        _zero_estimate_message(
            "The contamination estimate of cluster 'c'",
            "1 cell",
            "the measured gene sets",
            "4",
            "0.749",
            "a correction at the cluster level would leave its 1 cell uncorrected",
        ),
    ]
    # The warnings point at the caller, and the estimates are kept.
    assert {warning.filename for warning in caught} == {__file__}
    assert fit.fraction == pytest.approx(4 / 74, rel=1e-15)
    np.testing.assert_array_equal(fit.set_fractions, [0.0, 4 / 44])
    np.testing.assert_array_equal(fit.cluster_fractions, [4 / 30, 0.0, 0.0])
    np.testing.assert_array_equal(fit.cell_fractions, [4 / 30, 0.0, 0.0, 0.0])

    # Unnamed gene sets are named by their positions.
    with pytest.warns(UserWarning, match="from gene set '0' is 0: no count"):
        estimate_contamination(observed, expected, used)


def test_estimate_contamination_warns_once_for_a_global_estimate_of_zero():
    # One gene set: its estimate is the global one, so one warning names it.
    with pytest.warns(UserWarning) as caught:
        fit = estimate_contamination(
            [[0.0], [0.0], [5.0]],
            [[10.0], [34.8], [10.0]],
            [[True], [True], [False]],
            set_names=["haemoglobin"],
        )

    assert [str(warning.message) for warning in caught] == [
        _zero_estimate_message(
            "The global contamination estimate",
            "2 cells",
            "the measured gene sets",
            "44.8",
            "0.0669",
            "a correction with this estimate would leave 3 cells uncorrected",
        )
    ]
    assert fit.fraction == 0.0 and fit.set_fractions.tolist() == [0.0]
    np.testing.assert_array_equal(fit.cell_fractions, np.zeros(3))


def test_estimate_contamination_does_not_warn_about_positive_or_missing_estimates():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The second set has no evidence, and every other estimate is positive.
        fit = estimate_contamination(
            [[1.0, 0.0], [2.0, 0.0]],
            [[10.0, 0.0], [10.0, 0.0]],
            [[True, True], [True, False]],
            clusters=["a", "b"],
            level="cluster",
            set_names=["haemoglobin", "immunoglobulin"],
        )
    assert fit.fraction == pytest.approx(0.15)
    assert np.isnan(fit.set_fractions[1])
    np.testing.assert_allclose(fit.cluster_fractions, [0.1, 0.2])


def test_rounding_draws_are_splitmix64_streams_of_the_seed_and_row_key():
    keys = np.array([0, 1, 2, 10**6, 2**63, 2**64 - 1], dtype=np.uint64)
    for seed in (0, 1, 2**64 - 1):
        np.testing.assert_array_equal(
            _row_uniforms(seed, keys), [_draw(seed, int(key)) for key in keys]
        )
    # Seeds wrap around at 2**64, as cell calling's do.
    np.testing.assert_array_equal(
        _row_uniforms(2**64 + 3, keys), _row_uniforms(3, keys)
    )
    # The finaliser is cell calling's, applied to whole arrays.
    values = np.array([0, 1, 2**63 + 5, 2**64 - 1], dtype=np.uint64)
    with np.errstate(over="ignore"):
        compiled = [
            int(cell_calling._mix64.py_func(np.uint64(value))) for value in values
        ]
    assert (
        _splitmix64(values).tolist()
        == compiled
        == [_mix(int(value)) for value in values]
    )


def test_removal_takes_the_cells_share_in_proportion_to_the_soup():
    counts = np.tile(np.array([5, 3, 0, 2], dtype=np.uint16), (4000, 1))

    corrected = remove_ambient_counts(
        counts, np.full(4000, 0.4), [0.5, 0.25, 0.25, 0.0]
    )

    # 0.4 * 10 = 4 counts at level 16/3: 8/3 and 4/3. Their fractional parts
    # sum to 1, so each cell rounds one of the two up: the first when its draw
    # u is at least 1/3, which happens with probability 2/3, and otherwise the
    # second. The fourth gene is absent from the soup and keeps its counts.
    draws = np.array([_draw(0, key) for key in range(4000)])
    np.testing.assert_array_equal(
        corrected, np.where(draws[:, None] >= 1 / 3, [2, 2, 0, 2], [3, 1, 0, 2])
    )
    assert corrected.dtype == np.uint16
    assert abs((corrected[:, 0] == 2).mean() - 2 / 3) < 4 * math.sqrt(2 / 9 / 4000)


def test_removal_empties_genes_in_order_of_count_over_soup_fraction():
    counts = np.tile(np.array([1, 10], dtype=np.uint8), (200, 1))

    corrected = remove_ambient_counts(
        counts, np.full(200, 0.5), [0.5, 0.5], random_seed=3
    )

    # 5.5 counts: the first gene (ratio 2) empties, and the second loses 4.5,
    # 4 counts and a fifth when the cell's draw is at least 0.5.
    draws = np.array([_draw(3, key) for key in range(200)])
    np.testing.assert_array_equal(corrected[:, 0], 0)
    np.testing.assert_array_equal(corrected[:, 1], np.where(draws >= 0.5, 5, 6))
    assert set(corrected[:, 1].tolist()) == {5, 6}


def test_systematic_rounding_rounds_up_the_part_that_the_draw_falls_in():
    # Two parts of 0.25: the first is rounded up when 0.25 + u passes 1, the
    # second when 0.5 + u passes 1 and 0.25 + u does not.
    uniforms = np.array([0.0, 0.49, 0.5, 0.74, 0.75, 0.999])

    up = _round_up(
        np.repeat(np.arange(6), 2), np.full(12, 0.25), np.full(6, 0.5), uniforms
    )

    np.testing.assert_array_equal(
        up.reshape(6, 2), [[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0]]
    )
    # The counts follow the row key's draw and keep their dtype.
    counts = np.array([[1, 1]], dtype=np.int64)
    for low, high, expected in (
        (0.0, 0.5, [1, 1]),
        (0.5, 0.75, [1, 0]),
        (0.75, 1.0, [0, 1]),
    ):
        key = next(key for key in range(1000) if low <= _draw(0, key) < high)
        corrected = remove_ambient_counts(counts, [0.25], [1, 1], row_keys=[key])
        np.testing.assert_array_equal(corrected, [expected])
        assert corrected.dtype == np.int64


def test_systematic_rounding_rounds_each_part_up_for_its_share_of_the_draws():
    parts = [
        np.full(10, 0.1),
        np.array([0.0, 0.73, 0.999999, 1e-9, 0.5, 0.5, 0.25, 0.0, 0.31]),
        np.random.default_rng(3).random(40),
        np.zeros(2),
    ]
    totals = np.array([math.fsum(row) for row in parts])
    grid = np.arange(10_000) / 10_000
    # One row for each list of parts and each draw of an even grid on [0, 1).
    owner = np.concatenate(
        [np.full(len(row), row_index) for row_index, row in enumerate(parts)]
    )
    fractions = np.concatenate(parts)
    rows = (np.arange(len(grid))[:, None] * len(parts) + owner).reshape(-1)

    up = _round_up(
        rows,
        np.tile(fractions, len(grid)),
        np.tile(totals, len(grid)),
        np.repeat(grid, len(parts)),
    ).reshape(len(grid), -1)

    assert set(np.unique(up).tolist()) <= {0, 1}
    # The draws that round a part up form an interval of the part's length, so
    # an even grid of 10,000 draws rounds it up for its share to within 1e-4,
    # and a part without a fractional part never.
    np.testing.assert_allclose(up.mean(axis=0), fractions, rtol=0, atol=1e-4 + 1e-12)
    assert not up[:, fractions == 0].any()
    # Each row rounds up its total of parts rounded down or up: on average the
    # total itself.
    for row_index, total in enumerate(totals):
        rounded = up[:, owner == row_index].sum(axis=1)
        assert np.isin(rounded, [math.floor(total), math.ceil(total)]).all()
        assert rounded.mean() == pytest.approx(total, abs=1e-4)


def test_systematic_rounding_aligns_each_row_to_its_total():
    last = 1.0 - 2**-53
    # Ten parts of 0.1 sum to 1 but count 4 units of 2**-40 more; the excess
    # comes off the last part, so even the largest draw rounds up one part.
    up = _round_up(
        np.zeros(10, dtype=np.int64),
        np.full(10, 0.1),
        np.array([1.0]),
        np.array([last]),
    )
    np.testing.assert_array_equal(up, [1] + [0] * 9)
    # Parts that fall 1,025 units short of their total of 2 take the shortfall
    # in their first part with a fractional part and room for it: not the part
    # of 1 - 2**-50, which counts 2**40 - 1 units, nor a part without a
    # fractional part. Every draw then rounds up exactly two parts, where the
    # parts alone would round up one for the draw 0.
    fractions = np.array([0.0, 1.0 - 2**-50, 0.5 - 2**-30, 0.0, 0.5])
    uniforms = np.array([0.0, 2**-53, 0.25, 0.5, last])
    up = _round_up(
        np.repeat(np.arange(5), 5), np.tile(fractions, 5), np.full(5, 2.0), uniforms
    ).reshape(5, 5)
    np.testing.assert_array_equal(
        up,
        [
            [0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
    )
    # A total below 0, where whole parts exceed the target by rounding, rounds
    # nothing up.
    up = _round_up(
        np.zeros(3, dtype=np.int64),
        np.array([0.2, 0.5, 0.0]),
        np.array([-1e-12]),
        np.array([last]),
    )
    np.testing.assert_array_equal(up, [0, 0, 0])


def test_removal_takes_every_soup_count_of_a_cell_with_fewer_than_its_share():
    counts = np.array([[2, 1, 7], [4, 0, 0]], dtype=np.uint32)

    for seed in range(3):
        corrected = remove_ambient_counts(
            counts, [0.9, 1.5], [0.4, 0.6, 0.0], random_seed=seed
        )

        # The first cell holds 3 soup counts but owes 9; the second owes 6 of 4.
        np.testing.assert_array_equal(corrected, [[0, 0, 7], [0, 0, 0]])


def test_removal_without_contamination_or_soup_counts_changes_nothing():
    counts = np.array([[3, 0, 2], [0, 0, 0], [0, 4, 0]], dtype=np.uint16)

    for seed in (0, 7):
        np.testing.assert_array_equal(
            remove_ambient_counts(
                counts, [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], random_seed=seed
            ),
            [[3, 0, 2], [0, 0, 0], [0, 4, 0]],
        )
    unchanged = remove_ambient_counts(counts, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(unchanged, counts)
    assert unchanged is not counts
    assert remove_ambient_counts(counts[:0], [], [1, 1, 1]).shape == (0, 3)


@pytest.mark.parametrize("seed", range(4))
def test_removal_rounds_the_continuous_allocation_by_systematic_sampling(seed):
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 60, 45
    counts = rng.poisson(rng.gamma(0.4, 6.0, size=n_genes), size=(n_cells, n_genes))
    counts = counts.astype(np.uint32)
    soup = rng.dirichlet(np.ones(n_genes)) * (rng.random(n_genes) > 0.25)
    fractions = rng.random(n_cells) * 0.6
    # No removal, every soup count, and half of each cell's counts; a cell
    # without counts and one without counts of genes in the soup.
    fractions[:3] = [0.0, 1.0, 0.5]
    counts[30] = 0
    counts[31, soup > 0] = 0
    keys = rng.permutation(10_000)[:n_cells]

    corrected = remove_ambient_counts(
        counts, fractions, soup, random_seed=seed, row_keys=keys
    )

    removal = _removed(counts, corrected)
    continuous = _continuous_removal(counts, fractions, soup)
    # No gene gains counts or loses more than the cell holds.
    assert removal.min() >= 0 and (removal <= counts).all()
    # Each gene loses its continuous removal rounded down or up, and each cell
    # its target rounded down or up: floor(target + u) for its draw u, as the
    # sum of its genes' losses telescopes.
    assert np.abs(removal - continuous).max() < 1
    soup_counts = (counts * (soup > 0)).sum(axis=1)
    targets = np.minimum(fractions * counts.sum(axis=1), soup_counts)
    lost = removal.sum(axis=1)
    draws = np.array([_draw(seed, int(key)) for key in keys])
    np.testing.assert_array_equal(lost, np.floor(targets + draws))
    np.testing.assert_array_equal(removal[:, soup == 0], 0)
    np.testing.assert_array_equal(lost[:2], [0, soup_counts[1]])
    np.testing.assert_array_equal(corrected[30:32], counts[30:32])
    # The rounding is the systematic sample of each row's draw.
    np.testing.assert_array_equal(removal, _systematic_removal(continuous, seed, keys))
    # Rows are independent given their keys, so any row blocks give the same
    # counts.
    split = n_cells // 3
    blocks = np.vstack(
        [
            remove_ambient_counts(
                counts[:split],
                fractions[:split],
                soup,
                random_seed=seed,
                row_keys=keys[:split],
            ),
            remove_ambient_counts(
                counts[split:],
                fractions[split:],
                soup,
                random_seed=seed,
                row_keys=keys[split:],
            ),
        ]
    )
    np.testing.assert_array_equal(blocks, corrected)


def test_removal_draws_depend_only_on_the_seed_and_the_row_keys():
    rng = np.random.default_rng(11)
    counts = rng.poisson(3.0, size=(200, 30)).astype(np.uint16)
    soup = rng.dirichlet(np.ones(30))
    fractions = np.full(200, 0.07)

    first = remove_ambient_counts(counts, fractions, soup, random_seed=1)

    # A seed gives the same counts, and the keys default to the row positions,
    # whatever their integer type.
    for keys in (None, np.arange(200), np.arange(200, dtype=np.uint32)):
        np.testing.assert_array_equal(
            remove_ambient_counts(
                counts, fractions, soup, random_seed=1, row_keys=keys
            ),
            first,
        )
    np.testing.assert_array_equal(
        _removed(counts, first),
        _systematic_removal(
            _continuous_removal(counts, fractions, soup), 1, range(200)
        ),
    )
    # Another seed or other keys give other draws, and so other counts.
    assert (
        remove_ambient_counts(counts, fractions, soup, random_seed=2) != first
    ).any()
    assert (
        remove_ambient_counts(
            counts, fractions, soup, random_seed=1, row_keys=np.arange(200) + 1000
        )
        != first
    ).any()
    # Rows follow their keys in any order and in any blocks.
    order = rng.permutation(200)
    np.testing.assert_array_equal(
        remove_ambient_counts(
            counts[order], fractions[order], soup, random_seed=1, row_keys=order
        ),
        first[order],
    )
    for edges in ([0, 1, 200], [0, 57, 58, 130, 200], list(range(0, 201, 25))):
        parts = [
            remove_ambient_counts(
                counts[start:stop],
                fractions[start:stop],
                soup,
                random_seed=1,
                row_keys=np.arange(start, stop),
            )
            for start, stop in zip(edges[:-1], edges[1:], strict=True)
        ]
        np.testing.assert_array_equal(np.vstack(parts), first)


def test_removal_gives_genes_with_small_soup_shares_their_expected_removal():
    # A head-to-head with SoupX on 10x Genomics PBMC data found that rounding
    # each cell's removal to the genes with the largest fractional parts gave
    # every extra count to the most abundant soup genes: genes beyond soup rank
    # 200 lost none of the thousands of counts they were due. Here the soup's
    # gene shares fall as a power of their rank, in random gene order, so most
    # genes are due far less than one count in each cell.
    rng = np.random.default_rng(2020)
    n_cells, n_genes = 400, 3000
    shares = 1.0 / np.arange(1, n_genes + 1) ** 1.1
    soup = rng.permutation(shares / shares.sum())
    expression = rng.dirichlet(np.full(n_genes, 0.5))
    totals = rng.integers(2000, 6000, n_cells)
    counts = np.array(
        [rng.multinomial(total, 0.97 * expression + 0.03 * soup) for total in totals]
    ).astype(np.uint32)
    fractions = np.full(n_cells, 0.03)

    removal = _removed(
        counts, remove_ambient_counts(counts, fractions, soup, random_seed=5)
    )

    # A gene is rounded up in each cell independently, with a probability equal
    # to its fractional part f, so its removal summed over the cells has the
    # continuous removal as its mean and sum(f (1 - f)) as its variance. In
    # random gene order the roundings of different genes in a cell are close to
    # independent too, so a group of genes has about the summed variance.
    continuous = _continuous_removal(counts, fractions, soup)
    part = continuous - np.floor(continuous)
    variance = part * (1.0 - part)
    rank = np.empty(n_genes, dtype=np.int64)
    rank[np.argsort(-soup)] = np.arange(1, n_genes + 1)
    for low, high in ((1, 10), (11, 50), (51, 200), (201, 1000), (1001, n_genes)):
        genes = (rank >= low) & (rank <= high)
        due = continuous[:, genes].sum()
        assert due > 3000
        assert abs(removal[:, genes].sum() - due) < 5 * math.sqrt(
            variance[:, genes].sum()
        )
    expected = continuous.sum(axis=0)
    removed = removal.sum(axis=0)
    # A gene due at least 15 counts loses none with probability below exp(-15).
    due = expected >= 15
    assert (rank[due] > 200).sum() >= 50
    assert (removed[due] > 0).all()
    common = expected >= 20
    z_scores = (removed[common] - expected[common]) / np.sqrt(
        variance[:, common].sum(axis=0)
    )
    assert np.abs(z_scores).max() < 5


@pytest.mark.parametrize(
    ("counts", "fractions", "soup", "keywords", "error", "message"),
    [
        (np.ones((1, 2)), [0.1], [1, 1], {}, TypeError, "integers"),
        (np.ones(2, dtype=np.uint8), [0.1], [1, 1], {}, ValueError, "cells by genes"),
        (
            np.zeros((0, 2**23), dtype=np.uint8),
            [],
            [1],
            {},
            ValueError,
            "fewer than 8388608 genes",
        ),
        (np.array([[1, -1]]), [0.1], [1, 1], {}, ValueError, "not be negative"),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1, 0.2],
            [1, 1],
            {},
            ValueError,
            "one fraction per cell",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [np.nan],
            [1, 1],
            {},
            ValueError,
            "fractions must be finite",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [-0.1],
            [1, 1],
            {},
            ValueError,
            "fractions must be finite",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1],
            {},
            ValueError,
            "one fraction per gene",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, -1],
            {},
            ValueError,
            "soup must be finite",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, 1],
            {"random_seed": -1},
            ValueError,
            "random_seed must be at least 0",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, 1],
            {"random_seed": 1.0},
            TypeError,
            "random_seed must be an integer",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, 1],
            {"random_seed": True},
            TypeError,
            "random_seed must be an integer",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, 1],
            {"row_keys": [0.0]},
            TypeError,
            "row_keys must hold integers",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, 1],
            {"row_keys": [0, 1]},
            ValueError,
            "one key per row",
        ),
        (
            np.ones((1, 2), dtype=np.uint8),
            [0.1],
            [1, 1],
            {"row_keys": [-1]},
            ValueError,
            "row_keys must not be negative",
        ),
    ],
)
def test_removal_validates_its_arguments(
    counts, fractions, soup, keywords, error, message
):
    with pytest.raises(error, match=message):
        remove_ambient_counts(counts, fractions, soup, **keywords)
