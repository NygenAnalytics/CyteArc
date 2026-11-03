"""Diversity indices and Hill numbers, against iNEXT, vegan and exact sums."""

import warnings
from fractions import Fraction
from itertools import combinations
from math import comb, fsum, log, ulp

import numpy as np
import pandas as pd
import pytest

from cytearc.clones import diversity as dv
from cytearc.clones.diversity import CloneSizes

# Two abundance vectors, used as clone sizes here, whose outputs below come
# from running iNEXT 3.0.2 and vegan 2.7.6 on them.
A = [10, 5, 5, 3, 2, 2, 1, 1, 1, 1, 1, 1]
B = [50, 20, 4, 2, 2, 1, 1, 1, 1]


def _clones(sizes) -> CloneSizes:
    return CloneSizes.from_sizes(sizes)


def test_clone_sizes_are_summarized_by_size() -> None:
    clones = _clones([3, 0, 1, 2, 1, 1])
    assert clones.values.tolist() == [1, 2, 3]
    assert clones.counts.tolist() == [3, 1, 1]
    assert (clones.n, clones.richness, clones.f1, clones.f2) == (8, 5, 3, 1)
    frequencies, counts = clones.frequencies()
    np.testing.assert_allclose(frequencies, [1 / 8, 2 / 8, 3 / 8])
    with pytest.raises(ValueError, match="must not be negative"):
        _clones([2, -1])


@pytest.mark.parametrize(
    ("sizes", "observed", "estimated"),
    [
        (
            A,
            (12, 8.39095516943926, 6.29479768786127),
            (20.7272727272727, 11.6060325357083, 7.54285714285715),
        ),
        (
            B,
            (9, 3.2845734146719, 2.29644808743169),
            (12.9512195121951, 3.56798065100181, 2.33380182712578),
        ),
        (
            [10, 5, 3, 1, 1, 1, 1],
            (7, 4.61273407138212, 3.50724637681159),
            (12.7272727272727, 6.16958053858778, 3.98275862068966),
        ),
        ([1] * 20, (20, 20, 20), (200.5, 244.132970379159, np.inf)),
        ([1, 1], (2, 2, 2), (2.5, 3.37500000000171, np.inf)),
        (
            [5, 3, 1],
            (3, 2.55202792428089, 2.31428571428571),
            (3, 2.88766938456823, 2.76923076923077),
        ),
        (
            [4, 3, 2, 2],
            (4, 3.82712236060196, 3.66666666666667),
            (4, 4.43520711407059, 5),
        ),
        ([7], (1, 1, 1), (1, 1, 1)),
    ],
)
def test_hill_numbers_match_inext(sizes, observed, estimated) -> None:
    clones = _clones(sizes)
    for order in dv.ORDERS:
        assert dv.observed_hill(clones, order) == pytest.approx(
            observed[order], rel=1e-12, abs=0
        )
        assert dv.estimated_hill(clones, order) == pytest.approx(
            estimated[order], rel=1e-10, abs=0
        )


def test_coverage_matches_inext_with_and_without_doubletons() -> None:
    # (n - 1) f1 / ((n - 1) f1 + 2 f2) with doubletons; with f1 - 1 and f2 = 1
    # without them, which makes a lone singleton without doubletons complete.
    assert dv.coverage(_clones(A)) == pytest.approx(
        1 - 6 / 33 * (32 * 6 / (32 * 6 + 4))
    )
    assert round(dv.coverage(_clones(A)), 4) == 0.8219
    assert round(dv.coverage(_clones(B)), 4) == 0.9518
    assert round(dv.coverage(_clones([10, 5, 3, 1, 1, 1, 1])), 4) == 0.8238
    assert round(dv.coverage(_clones([1] * 20)), 4) == 0.0055
    assert round(dv.coverage(_clones([1, 1])), 4) == 0.6667
    assert dv.coverage(_clones([5, 3, 1])) == 1.0
    assert dv.coverage(_clones([4, 3, 2, 2])) == 1.0
    assert np.isnan(dv.coverage(_clones([])))


@pytest.mark.parametrize(
    ("sizes", "m", "method", "values", "coverage"),
    [
        (
            A,
            16,
            "rarefaction",
            (8.21695037134414, 6.6098071433199, 5.35361216730039),
            0.720121122234581,
        ),
        (
            A,
            50,
            "extrapolation",
            (14.5805132036704, 9.37988957393188, 6.6700353713997),
            0.874555928089749,
        ),
        (
            A,
            66,
            "extrapolation",
            (16.3078174812882, 10.030943287946, 6.86254430878299),
            0.909807035796235,
        ),
        (
            B,
            23,
            "rarefaction",
            (4.83087682300689, 2.86343404471686, 2.20587980477662),
            0.898496221999323,
        ),
        (
            B,
            164,
            "extrapolation",
            (11.5065556459352, 3.45205826596542, 2.3149742848642),
            0.98238214797244,
        ),
    ],
)
def test_rarefaction_and_extrapolation_match_inext(
    sizes, m, method, values, coverage
) -> None:
    clones = _clones(sizes)
    for order in dv.ORDERS:
        value, got = dv.diversity_at(clones, m, order)
        assert got == method
        assert value == pytest.approx(values[order], rel=1e-11, abs=0)
    assert dv.coverage_at(clones, m) == pytest.approx(coverage, rel=1e-11, abs=0)


def test_standardizing_by_coverage_matches_inext() -> None:
    a, b = _clones(A), _clones(B)
    # Above the observed coverage the extrapolated coverage is inverted.
    assert dv.size_for_coverage(a, 0.9) == pytest.approx(
        60.9940672961916, rel=1e-9, abs=0
    )
    for order, value in zip(
        dv.ORDERS, (15.8272727272727, 9.85088615975206, 6.81211855586043), strict=True
    ):
        assert dv.diversity_at(a, 60.9940672961916, order)[0] == pytest.approx(
            value, rel=1e-10, abs=0
        )
    # Below it, rarefied sizes are interpolated linearly, as iNEXT does.
    # iNEXT's root finding stops at a coverage of 0.89999999; the size here
    # reaches 0.9 itself.
    m = dv.size_for_coverage(b, 0.9)
    assert m == pytest.approx(23.9610607098077, rel=1e-6, abs=0)
    assert dv.coverage_at(b, m) == pytest.approx(0.9, rel=1e-14, abs=0)
    for order, value in zip(
        dv.ORDERS, (4.92842811594038, 2.87982535549222, 2.21073264251641), strict=True
    ):
        assert dv.diversity_at(b, 23.9610607098077, order)[0] == pytest.approx(
            value, rel=1e-10, abs=0
        )
    assert dv.size_for_coverage(a, dv.coverage(a)) == 33
    # A level below the coverage of one cell needs one cell.
    assert dv.size_for_coverage(_clones([10, 5, 3, 1, 1, 1, 1]), 0.2) == 1.0
    for level in (0, 1, 1.5, -0.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            dv.size_for_coverage(a, level)


def _subsamples(sizes, m):
    """Every subsample of m cells, as the clone sizes it holds."""
    cells = [clone for clone, size in enumerate(sizes) for _ in range(size)]
    for chosen in combinations(range(len(cells)), m):
        counts: dict[int, int] = {}
        for cell in chosen:
            counts[cells[cell]] = counts.get(cells[cell], 0) + 1
        yield chosen, cells, counts


@pytest.mark.parametrize("sizes", [[3, 2, 1, 1], [4, 1, 1, 1, 1], [2, 2, 2], [5, 2]])
def test_rarefaction_equals_every_subsample(sizes) -> None:
    clones = _clones(sizes)
    n = sum(sizes)
    for m in range(1, n):
        draws = list(_subsamples(sizes, m))
        richness = sum(len(counts) for _, _, counts in draws) / len(draws)
        entropy = sum(
            -sum(c / m * log(c / m) for c in counts.values()) for _, _, counts in draws
        ) / len(draws)
        squares = sum(
            sum((c / m) ** 2 for c in counts.values()) for _, _, counts in draws
        ) / len(draws)
        assert dv.diversity_at(clones, m, 0)[0] == pytest.approx(
            richness, rel=1e-12, abs=0
        )
        assert dv.diversity_at(clones, m, 1)[0] == pytest.approx(
            np.exp(entropy), rel=1e-12, abs=0
        )
        assert dv.diversity_at(clones, m, 2)[0] == pytest.approx(
            1 / squares, rel=1e-12, abs=0
        )
        # The coverage of m cells: the chance that one more cell, drawn from
        # the rest, belongs to a clone the m cells hold.
        seen = Fraction(0)
        for chosen, cells, counts in draws:
            rest = [cell for cell in range(n) if cell not in chosen]
            seen += Fraction(sum(cells[cell] in counts for cell in rest), len(rest))
        assert dv.coverage_at(clones, m) == pytest.approx(
            float(seen / len(draws)), rel=1e-12, abs=0
        )


def test_coverage_is_exact_for_large_samples() -> None:
    rng = np.random.default_rng(3)
    sizes = np.minimum(rng.zipf(1.8, size=3_000), 400)
    clones = _clones(sizes)
    n = clones.n
    for m in (2, n // 3, n - 2):
        missed = sum(
            Fraction(
                int(value) * int(count) * comb(n - int(value), m), n * comb(n - 1, m)
            )
            for value, count in zip(clones.values, clones.counts, strict=True)
            if n - value >= m
        )
        assert dv.coverage_at(clones, m) == pytest.approx(
            float(1 - missed), rel=1e-13, abs=0
        )
        richness = sum(
            int(count) * (1 - Fraction(comb(n - int(value), m), comb(n, m)))
            for value, count in zip(clones.values, clones.counts, strict=True)
        )
        assert dv.diversity_at(clones, m, 0)[0] == pytest.approx(
            float(richness), rel=1e-13, abs=0
        )


def test_sizes_between_whole_numbers_interpolate() -> None:
    clones = _clones(B)
    low, high = dv.diversity_at(clones, 81, 1)[0], dv.observed_hill(clones, 1)
    assert dv.diversity_at(clones, 81.25, 1)[0] == pytest.approx(
        low + 0.25 * (high - low)
    )
    low, high = dv.coverage_at(clones, 81), dv.coverage(clones)
    assert dv.coverage_at(clones, 81.5) == pytest.approx((low + high) / 2)
    assert dv.diversity_at(clones, 82, 0) == (9.0, "observed")
    for m in (0, 0.5):
        with pytest.raises(ValueError, match="at least one cell"):
            dv.diversity_at(clones, m, 0)
        with pytest.raises(ValueError, match="at least one cell"):
            dv.coverage_at(clones, m)
    with pytest.raises(ValueError, match="order must be"):
        dv.diversity_at(clones, 3, 3)
    with pytest.raises(ValueError, match="order must be"):
        dv.estimated_hill(clones, True)


def test_curves_share_the_shannon_terms() -> None:
    clones = _clones(A)
    sizes = dv.default_sizes(clones.n)
    for order in dv.ORDERS:
        values, methods = dv.hill_curve(clones, sizes.tolist(), order)
        expected = [dv.diversity_at(clones, float(m), order) for m in sizes]
        assert values == pytest.approx(
            [value for value, _ in expected], rel=1e-15, abs=0
        )
        assert methods == [method for _, method in expected]


def test_extrapolation_without_singletons() -> None:
    clones = _clones([4, 3, 2, 2])
    # No clone is missed, so richness and coverage stay, while the
    # Shannon estimate still exceeds the observed entropy, as in iNEXT.
    assert dv.diversity_at(clones, 30, 0)[0] == 4
    assert dv.coverage_at(clones, 30) == 1.0
    assert dv.diversity_at(clones, 21, 1)[0] == pytest.approx(
        4.20839264819087, rel=1e-11, abs=0
    )
    assert dv.diversity_at(clones, 22, 1)[0] == pytest.approx(
        4.22969311355331, rel=1e-11, abs=0
    )
    single = _clones([7])
    assert dv.diversity_at(single, 14, 1)[0] == 1.0
    one_cell = _clones([1])
    assert dv.diversity_at(one_cell, 3, 1)[0] == 1.0
    assert dv.diversity_at(one_cell, 3, 2)[0] == pytest.approx(3.0)


def test_default_sizes_follow_inext_knots() -> None:
    assert dv.default_sizes(33).tolist() == [
        1,
        2,
        4,
        6,
        7,
        9,
        11,
        13,
        14,
        16,
        18,
        19,
        21,
        23,
        25,
        26,
        28,
        30,
        32,
        33,
        34,
        35,
        37,
        39,
        40,
        42,
        44,
        45,
        47,
        49,
        50,
        52,
        54,
        55,
        57,
        59,
        60,
        62,
        64,
        66,
    ]
    sizes = dv.default_sizes(1348)
    assert sizes[:4].tolist() == [1, 75, 150, 225] and sizes[18:22].tolist() == [
        1347,
        1348,
        1349,
        1419,
    ]
    assert sizes[-1] == 2696 and len(sizes) == 40
    assert dv.default_sizes(100, endpoint=50, knots=5).tolist() == [1, 13, 25, 37, 50]
    assert dv.default_sizes(1).tolist() == [1, 2]
    for kwargs in ({"knots": 1}, {"knots": True}, {"endpoint": 0}, {"endpoint": 2.5}):
        with pytest.raises(ValueError, match="knots|endpoint"):
            dv.default_sizes(10, **kwargs)


def test_default_sizes_take_numpy_integers_up_to_the_largest_size() -> None:
    assert dv.default_sizes(100, endpoint=np.int64(50), knots=np.int32(5)).tolist() == [
        1,
        13,
        25,
        37,
        50,
    ]
    assert dv.checked_curve_arguments(None, np.int64(40)) == (None, 40)
    # Sizes are floats while they are spaced, exact up to 2**53.
    assert dv.LARGEST_SIZE == 2**53
    assert dv.default_sizes(10, endpoint=2**53, knots=4)[-1] == 2**53
    with pytest.raises(ValueError, match="up to 9007199254740992"):
        dv.default_sizes(10, endpoint=2**53 + 1)


def test_entropy_tail_is_stable_in_both_regimes() -> None:
    for n, a in ((33, 0.0204), (1_000, 0.1), (200_000, 1e-7), (5, 0.9)):
        j = np.arange(1, 400_000, dtype=np.float64)
        direct = float(np.sum((1 - a) ** j / (j + n - 1)))
        if (1 - a) ** j[-1] > 1e-12:
            continue
        assert dv._entropy_tail(n, a) == pytest.approx(direct, rel=1e-11, abs=0)
    # Tails summed as written over more than one chunk, and as a series that
    # needs more than one chunk of terms.
    for n, a in ((2_000_000, 2e-6), (1_000_000, 1e-5)):
        direct = 0.0
        # Terms beyond 60 / a are below e**-60 of the first.
        for start in range(1, int(60 / a), 4_000_000):
            j = np.arange(start, min(start + 4_000_000, int(60 / a)), dtype=np.float64)
            direct += float(np.sum(np.exp(j * np.log1p(-a)) / (j + n - 1)))
        assert dv._entropy_tail(n, a) == pytest.approx(direct, rel=1e-10, abs=0)


def test_indices_of_a_few_clones() -> None:
    clones = _clones([3, 2, 1, 1])
    got = dv.indices(clones)
    p = np.array([3, 2, 1, 1]) / 7
    assert list(got) == list(dv.INDEX_COLUMNS)
    assert (
        got["n_cells"],
        got["n_clones"],
        got["n_singletons"],
        got["n_doubletons"],
    ) == (
        7,
        4,
        2,
        1,
    )
    assert got["shannon"] == pytest.approx(-np.sum(p * np.log(p)))
    assert got["pielou"] == pytest.approx(got["shannon"] / np.log(4))
    assert got["simpson"] == pytest.approx(np.sum(p**2))
    assert got["inverse_simpson"] == pytest.approx(1 / np.sum(p**2))
    assert got["gini_simpson"] == pytest.approx(1 - np.sum(p**2))
    pairs = sum(abs(x - y) for x in (3, 2, 1, 1) for y in (3, 2, 1, 1))
    assert got["gini"] == pytest.approx(pairs / (2 * 16 * 7 / 4))
    assert got["d50"] == 2
    assert got["chao1"] == pytest.approx(4 + 6 / 7 * 4 / 2)
    assert got["chao1_bias_corrected"] == pytest.approx(4 + 2 * 1 / (2 * 2))
    coverage = 1 - 2 / 7
    gamma = max(4 / coverage * 8 / (7 * 6) - 1, 0)
    assert got["ace"] == pytest.approx(4 / coverage + 2 / coverage * gamma)


def test_d50_counts_a_clone_that_holds_exactly_half() -> None:
    # The clone of 2 of 4 cells holds half of them, as do two clones of 3 of
    # 12 cells.
    assert dv.indices(_clones([2, 1, 1]))["d50"] == 1
    assert dv.indices(_clones([3, 3, 2, 2, 1, 1]))["d50"] == 2
    # Just under half needs one clone more.
    assert dv.indices(_clones([3, 3, 2, 2, 2, 1]))["d50"] == 3


def test_indices_of_one_clone_and_richness_estimators() -> None:
    got = dv.indices(_clones([7]))
    assert np.isnan(got["pielou"]) and got["gini"] == 0 and got["d50"] == 1
    assert got["inverse_simpson"] == 1 and got["coverage"] == 1
    # vegan 2.7.6 estimateR.
    assert dv.chao1_bias_corrected(_clones([10, 5, 3, 1, 1, 1, 1])) == 13
    assert dv.ace(_clones([10, 5, 3, 1, 1, 1, 1])) == pytest.approx(
        14.1687242798354, rel=1e-12, abs=0
    )
    assert dv.ace(_clones([5, 3, 1])) == pytest.approx(3.62109375, rel=1e-12, abs=0)
    assert dv.chao1_bias_corrected(_clones([1] * 20)) == 210
    # Every rare clone a singleton leaves ACE undefined, as vegan's NA.
    assert np.isnan(dv.ace(_clones([1] * 20)))
    assert np.isnan(dv.ace(_clones([20, 1, 1])))
    # Without rare clones, ACE counts the abundant ones.
    assert dv.ace(_clones([40, 20, 11])) == 3
    assert dv.chao1(_clones([])) == 0
    assert np.isnan(dv.estimated_hill(_clones([]), 1))
    assert np.isnan(dv.observed_hill(_clones([]), 0))
    assert np.isnan(dv.chao_entropy(_clones([])))
    assert np.isnan(dv._gini(dv.GroupedCloneSizes.from_groups([_clones([])]))[0])


def test_common_levels_follow_inext_defaults() -> None:
    clones = [_clones(A), _clones(B)]
    groups = dv.GroupedCloneSizes.from_groups(clones)
    assert dv.common_size(groups) == 66
    assert dv.common_coverage(groups) == pytest.approx(
        min(dv.coverage_at(clones[0], 66), dv.coverage_at(clones[1], 164))
    )


def _exact_entropy(sizes, m) -> float:
    """The expected entropy of m of the cells, from exact hypergeometric terms."""
    n = sum(sizes)
    total = comb(n, m)
    terms = []
    for size in sizes:
        for k in range(max(1, m - (n - size)), min(size, m) + 1):
            chance = Fraction(comb(size, k) * comb(n - size, m - k), total)
            terms.append(float(chance) * -(k / m) * log(k / m))
    return float(np.sum(np.sort(terms)))


@pytest.mark.parametrize(
    ("sizes", "m"),
    [
        ([200, 150, 80, 40, 40, 20, 9, 5, 3, 2, 1, 1, 1], 300),
        ([200, 150, 80, 40, 40, 20, 9, 5, 3, 2, 1, 1, 1], 551),
        ([90, 3, 3, 2, 1, 1], 7),
        ([5, 4, 1], 1),
    ],
)
def test_rarefied_entropy_equals_exact_hypergeometric_sums(sizes, m) -> None:
    # Large clones and samples need windows of more than 64 terms each.
    clones = _clones(sizes)
    expected = np.exp(_exact_entropy(sizes, m))
    assert dv.diversity_at(clones, m, 1)[0] == pytest.approx(expected, rel=1e-13, abs=0)
    values, methods = dv.hill_curve(clones, [m, m, 1, clones.n], 1)
    assert values[:2] == pytest.approx([expected, expected], rel=1e-13, abs=0)
    assert methods == ["rarefaction", "rarefaction", "rarefaction", "observed"]


@pytest.mark.parametrize("seed", range(6))
def test_size_for_coverage_finds_the_last_whole_size_below_the_level(seed) -> None:
    rng = np.random.default_rng(seed)
    clones = _clones(np.minimum(rng.zipf(1.6, size=int(rng.integers(30, 300))), 60))
    n = clones.n
    # The coverage of every whole size, each computed on its own.
    whole = [dv.coverage_at(clones, m) for m in range(1, n + 1)]
    for level in rng.uniform(whole[0], whole[-1], size=6):
        low = max(m for m in range(1, n) if whole[m - 1] < level)
        expected = low + (level - whole[low - 1]) / (whole[low] - whole[low - 1])
        assert dv.size_for_coverage(clones, level) == pytest.approx(
            expected, rel=1e-13, abs=0
        )


def test_diversity_tables_match_inext_for_every_group() -> None:
    groups = dv.GroupedCloneSizes.from_groups([_clones(A), _clones(B)])
    names = ["a, s1", "b, s2"]
    table, hill, size, level = dv.diversity_tables(groups, names)
    # iNEXT's defaults: the smallest doubled size and the lowest coverage at
    # doubled sizes.
    assert size == 66
    assert level == pytest.approx(0.909807035796235, rel=1e-11, abs=0)
    assert table.columns.tolist() == ["group", *dv.INDEX_COLUMNS]
    assert table["group"].tolist() == ["a, s1", "b, s2"]
    assert table[
        ["n_cells", "n_clones", "n_singletons", "n_doubletons"]
    ].values.tolist() == [
        [33, 12, 6, 2],
        [82, 9, 4, 2],
    ]
    assert table["coverage"].round(4).tolist() == [0.8219, 0.9518]
    assert hill.columns.tolist() == [
        "group",
        "order",
        "observed",
        "estimated",
        "at_size",
        "coverage_at_size",
        "method_at_size",
        "at_coverage",
        "size_at_coverage",
        "method_at_coverage",
    ]
    assert hill["group"].tolist() == ["a, s1"] * 3 + ["b, s2"] * 3
    assert hill["order"].tolist() == [0, 1, 2] * 2
    a, b = hill.iloc[:3], hill.iloc[3:]
    np.testing.assert_allclose(
        a["observed"], [12, 8.39095516943926, 6.29479768786127], rtol=1e-12
    )
    np.testing.assert_allclose(
        b["estimated"],
        [12.9512195121951, 3.56798065100181, 2.33380182712578],
        rtol=1e-10,
    )
    # A at 66 cells, twice its size; B rarefied to them.
    np.testing.assert_allclose(
        a["at_size"], [16.3078174812882, 10.030943287946, 6.86254430878299], rtol=1e-11
    )
    assert a["coverage_at_size"].tolist() == pytest.approx([0.909807035796235] * 3)
    assert a["method_at_size"].tolist() == ["extrapolation"] * 3
    assert b["method_at_size"].tolist() == ["rarefaction"] * 3
    # The common coverage is A's own at 66 cells.
    assert a["size_at_coverage"].tolist() == pytest.approx([66.0] * 3, rel=1e-12, abs=0)
    np.testing.assert_allclose(a["at_coverage"], a["at_size"], rtol=1e-11)
    at_coverage = dv.diversity_tables(groups, names, size=23, coverage=0.9)[1]
    a, b = at_coverage.iloc[:3], at_coverage.iloc[3:]
    np.testing.assert_allclose(
        a["at_coverage"],
        [15.8272727272727, 9.85088615975206, 6.81211855586043],
        rtol=1e-10,
    )
    assert b["size_at_coverage"].iloc[0] == pytest.approx(
        23.9610607098077, rel=1e-6, abs=0
    )
    np.testing.assert_allclose(
        b["at_size"], [4.83087682300689, 2.86343404471686, 2.20587980477662], rtol=1e-11
    )
    assert b["coverage_at_size"].iloc[0] == pytest.approx(
        0.898496221999323, rel=1e-11, abs=0
    )
    assert a["method_at_coverage"].tolist() == ["extrapolation"] * 3
    assert b["method_at_coverage"].tolist() == ["rarefaction"] * 3


def test_rarefaction_curves_hold_each_group_order_and_size() -> None:
    groups = dv.GroupedCloneSizes.from_groups([_clones(A), _clones(B)])
    names = ["a", "b"]
    curves = dv.rarefaction_curves(
        groups, names, (2, 0, 1), sizes=[164, 16, 23, 50, 66, 16]
    )
    assert curves.columns.tolist() == [
        "group",
        "order",
        "size",
        "method",
        "diversity",
        "coverage",
    ]
    sizes = [16, 23, 50, 66, 164]
    assert curves["group"].tolist() == ["a"] * 15 + ["b"] * 15
    assert curves["order"].tolist() == ([2] * 5 + [0] * 5 + [1] * 5) * 2
    assert curves["size"].tolist() == sizes * 6
    # A holds 33 cells and B 82.
    assert curves["method"].tolist()[:5] == ["rarefaction"] * 2 + ["extrapolation"] * 3
    assert curves["method"].tolist()[15:20] == ["rarefaction"] * 4 + ["extrapolation"]
    by_row = curves.set_index(["group", "order", "size"])
    for group, m, values, sample_coverage in (
        (
            "a",
            16,
            (8.21695037134414, 6.6098071433199, 5.35361216730039),
            0.720121122234581,
        ),
        (
            "a",
            50,
            (14.5805132036704, 9.37988957393188, 6.6700353713997),
            0.874555928089749,
        ),
        (
            "a",
            66,
            (16.3078174812882, 10.030943287946, 6.86254430878299),
            0.909807035796235,
        ),
        (
            "b",
            23,
            (4.83087682300689, 2.86343404471686, 2.20587980477662),
            0.898496221999323,
        ),
        (
            "b",
            164,
            (11.5065556459352, 3.45205826596542, 2.3149742848642),
            0.98238214797244,
        ),
    ):
        for order, value in zip(dv.ORDERS, values, strict=True):
            row = by_row.loc[(group, order, m)]
            assert row["diversity"] == pytest.approx(value, rel=1e-11, abs=0)
            assert row["coverage"] == pytest.approx(sample_coverage, rel=1e-11, abs=0)
    defaults = dv.rarefaction_curves(groups, names, (0,), endpoint=40, knots=4)
    # A's 33 cells extrapolate to the endpoint; B's 82 rarefy to it.
    assert defaults["size"].tolist() == [1, 33, 34, 40, 1, 14, 27, 40]
    assert (
        dv.rarefaction_curves(
            dv.GroupedCloneSizes.from_groups([]), names[:0]
        ).columns.tolist()
        == curves.columns.tolist()
    )
    with pytest.raises(ValueError, match="order must be"):
        dv.rarefaction_curves(groups, names, (3,))
    # Each group needs one name, which its rows hold.
    with pytest.raises(ValueError, match="names must name each of the 2 groups, not 1"):
        dv.rarefaction_curves(groups, names[:1])
    with pytest.raises(ValueError, match="names must name each of the 2 groups, not 3"):
        dv.rarefaction_curves(groups, [*names, "c"])
    with pytest.raises(ValueError, match="one text name for each group"):
        dv.rarefaction_curves(groups, pd.DataFrame({"group": names}))
    with pytest.raises(ValueError, match="'a' names two"):
        dv.rarefaction_curves(groups, ["a", "a"])


def _grouped(*sizes) -> dv.GroupedCloneSizes:
    return dv.GroupedCloneSizes.from_groups([_clones(each) for each in sizes])


def test_grouped_clone_sizes_hold_each_groups_frequency_table() -> None:
    # Clones of groups 2, 0, 2, 2, 0, 3, 1 and 3 of four, one of no cells: group
    # 0 holds two singletons, group 1 nothing, group 2 clones of 3, 1 and 3
    # cells and group 3 clones of 5 and 2 cells.
    grouped = dv.GroupedCloneSizes.from_clones(
        [2, 0, 2, 2, 0, 3, 1, 3], [3, 1, 1, 3, 1, 5, 0, 2], 4
    )
    assert len(grouped) == 4
    assert grouped.starts.tolist() == [0, 1, 1, 3, 5]
    assert grouped.groups.tolist() == [0, 2, 2, 3, 3]
    assert grouped.values.tolist() == [1, 1, 3, 2, 5]
    assert grouped.counts.tolist() == [2, 1, 2, 1, 1]
    assert grouped.n.tolist() == [2, 0, 7, 7]
    assert grouped.richness.tolist() == [2, 0, 3, 2]
    assert grouped.f1.tolist() == [2, 0, 1, 0]
    assert grouped.f2.tolist() == [0, 0, 0, 1]
    tables = ([1, 1], [], [3, 1, 3], [5, 2])
    joined = _grouped(*tables)
    for name in ("starts", "groups", "values", "counts", "n", "richness", "f1", "f2"):
        assert getattr(joined, name).tolist() == getattr(grouped, name).tolist()
    for at, sizes in enumerate(tables):
        one, expected = grouped.group(at), _clones(sizes)
        assert one.values.tolist() == expected.values.tolist()
        assert one.counts.tolist() == expected.counts.tolist()
        assert (one.n, one.richness, one.f1, one.f2) == (
            expected.n,
            expected.richness,
            expected.f1,
            expected.f2,
        )
    taken = grouped.take([3, 1, 0, 3])
    assert taken.starts.tolist() == [0, 2, 2, 3, 5]
    assert taken.values.tolist() == [2, 5, 1, 2, 5]
    assert taken.counts.tolist() == [1, 1, 2, 1, 1]
    assert taken.n.tolist() == [7, 0, 2, 7]
    assert len(_grouped()) == 0 and _grouped().starts.tolist() == [0]


@pytest.mark.parametrize(
    ("groups", "sizes", "message"),
    [
        ([0, 1], [1], "one value per clone"),
        ([0, 4], [1, 1], "between 0 and 3"),
        ([-1], [1], "between 0 and 3"),
        ([0, 1], [2, -1], "must not be negative"),
    ],
)
def test_grouped_clone_sizes_refuse_clones_outside_the_groups(
    groups, sizes, message
) -> None:
    with pytest.raises(ValueError, match=message):
        dv.GroupedCloneSizes.from_clones(groups, sizes, 4)


def test_distinct_tables_keep_the_order_of_their_first_groups() -> None:
    grouped = _grouped(
        # Table a: two singletons and a clone of 3 cells.
        [3, 1, 1],
        [1, 3, 1],
        # b: the sizes of a, with one singleton fewer.
        [3, 1],
        # c: the counts of a, with a clone of 4 cells for its 3.
        [1, 1, 4],
        # d: no cells.
        [],
        # e: the rows of a and a clone of 5 cells after them.
        [3, 1, 1, 5],
        [5, 1, 3, 1],
        [],
        [1, 3],
    )
    tables, held = grouped.distinct()
    assert held.tolist() == [0, 0, 1, 2, 3, 4, 4, 3, 1]
    assert tables.starts.tolist() == [0, 2, 4, 6, 6, 9]
    assert tables.values.tolist() == [1, 3, 1, 3, 1, 4, 1, 3, 5]
    assert tables.counts.tolist() == [2, 1, 1, 1, 2, 1, 2, 1, 1]
    assert tables.n.tolist() == [5, 4, 6, 0, 10]
    # Groups that all differ are their own distinct tables.
    alone, places = _grouped([2, 1], [1], [2, 2]).distinct()
    assert places.tolist() == [0, 1, 2]
    assert alone.values.tolist() == [1, 2, 1, 2] and alone.counts.tolist() == [
        1,
        1,
        1,
        2,
    ]
    empty, none = _grouped().distinct()
    assert len(empty) == 0 and none.tolist() == []


def test_groups_with_equal_tables_get_equal_rows() -> None:
    sizes = [
        A,
        [1],
        B,
        [1, 1, 3, 1, 10, 2, 1, 5, 1, 2, 5, 1],
        [6, 3, 1, 1, 2],
        [5, 4, 1, 1, 2],
        [1],
        [1, 2, 50, 1, 4, 1, 20, 1, 2],
        [2, 1, 3, 6, 1],
    ]
    groups = _grouped(*sizes)
    assert groups.distinct()[1].tolist() == [0, 1, 2, 0, 3, 4, 1, 2, 3]
    names = [f"g{at}" for at in range(len(sizes))]
    table, hill, size, level = dv.diversity_tables(groups, names)
    # The groups of one cell set the common size; A's coverage at twice its
    # cells, iNEXT's 0.9098, is the lowest at doubled sizes.
    assert size == 2
    assert level == pytest.approx(0.909807035796235, rel=1e-11, abs=0)
    indices = table.drop(columns="group").to_numpy(dtype=np.float64)
    numbers = hill.drop(columns=["group", "method_at_size", "method_at_coverage"])
    measures = numbers.to_numpy(dtype=np.float64).reshape(len(sizes), -1)
    methods = hill[["method_at_size", "method_at_coverage"]].to_numpy()
    methods = methods.reshape(len(sizes), -1)
    for first, again in ((0, 3), (1, 6), (2, 7), (4, 8)):
        np.testing.assert_array_equal(indices[again], indices[first])
        np.testing.assert_array_equal(measures[again], measures[first])
        assert methods[again].tolist() == methods[first].tolist()
    # Groups 4 and 5 hold 13 cells in five clones, two singletons and a
    # doubleton each, and D50 of 2, but sizes 6 and 3 against 5 and 4.
    counted = ["n_cells", "n_clones", "n_singletons", "n_doubletons", "d50"]
    assert table.loc[4, counted].tolist() == table.loc[5, counted].tolist()
    assert table.loc[4, counted].tolist() == [13, 5, 2, 1, 2]
    assert table.loc[4, "gini"] == pytest.approx(2 * 51 / (5 * 13) - 6 / 5)
    assert table.loc[5, "gini"] == pytest.approx(2 * 50 / (5 * 13) - 6 / 5)
    # A's and B's rows hold iNEXT's numbers in each place.
    by_group = hill.set_index("group")
    for at in (0, 3):
        rows = by_group.loc[f"g{at}"]
        np.testing.assert_allclose(
            rows["observed"], [12, 8.39095516943926, 6.29479768786127], rtol=1e-12
        )
        np.testing.assert_allclose(
            rows["estimated"],
            [20.7272727272727, 11.6060325357083, 7.54285714285715],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            rows["at_coverage"],
            [16.3078174812882, 10.030943287946, 6.86254430878299],
            rtol=1e-11,
        )
        np.testing.assert_allclose(rows["size_at_coverage"], 66, rtol=1e-12)
    for at in (2, 7):
        rows = by_group.loc[f"g{at}"]
        np.testing.assert_allclose(
            rows["observed"], [9, 3.2845734146719, 2.29644808743169], rtol=1e-12
        )
        np.testing.assert_allclose(
            rows["estimated"],
            [12.9512195121951, 3.56798065100181, 2.33380182712578],
            rtol=1e-10,
        )
    # Each group's rows are those it gets alone at the same size and coverage.
    for at in range(len(sizes)):
        alone, alone_hill, _, _ = dv.diversity_tables(
            groups.take([at]),
            names[at : at + 1],
            size=size,
            coverage=level,
        )
        np.testing.assert_array_equal(
            alone.drop(columns="group").to_numpy(dtype=np.float64)[0], indices[at]
        )
        alone_numbers = alone_hill.drop(
            columns=["group", "method_at_size", "method_at_coverage"]
        )
        np.testing.assert_array_equal(
            alone_numbers.to_numpy(dtype=np.float64).reshape(-1), measures[at]
        )


def test_a_group_of_one_cell_and_a_group_of_every_cell() -> None:
    # Each of A's 33 cells as a group of its own, and a group of all of them.
    groups = _grouped(*[[1]] * 33, A)
    names = [f"cell {at}" for at in range(33)] + ["every"]
    table, hill, size, level = dv.diversity_tables(groups, names)
    # One cell doubled is the common size; A's coverage at 66 cells is the
    # lowest coverage at doubled sizes.
    assert size == 2
    assert level == pytest.approx(0.909807035796235, rel=1e-11, abs=0)
    by_group = table.set_index("group")
    rows = hill.set_index(["group", "order"])
    # One cell is one clone of one cell: complete, with no entropy, and no
    # rare clone of more than one cell for ACE.
    for at in range(33):
        cell = by_group.loc[f"cell {at}"]
        assert cell[
            ["n_cells", "n_clones", "n_singletons", "n_doubletons", "d50"]
        ].tolist() == [
            1,
            1,
            1,
            0,
            1,
        ]
        assert (cell["coverage"], cell["shannon"], cell["gini"]) == (1, 0, 0)
        assert (cell["simpson"], cell["inverse_simpson"], cell["gini_simpson"]) == (
            1,
            1,
            0,
        )
        assert (cell["chao1"], cell["chao1_bias_corrected"]) == (1, 1)
        assert np.isnan(cell["pielou"]) and np.isnan(cell["ace"])
        for order, estimated, doubled in zip(
            dv.ORDERS, (1, 1, np.inf), (1, 1, 2), strict=True
        ):
            row = rows.loc[(f"cell {at}", order)]
            assert (row["observed"], row["estimated"]) == (1, estimated)
            # Two cells extrapolated from one keep its clone and its entropy,
            # and count two dominant clones, since no pair shares a clone.
            assert (row["at_size"], row["coverage_at_size"]) == (doubled, 1)
            assert row["method_at_size"] == "extrapolation"
            # The coverage of one cell is 1, so it reaches the common one.
            assert (row["at_coverage"], row["size_at_coverage"]) == (1, 1)
            assert row["method_at_coverage"] == "observed"
    # The group of every cell holds A's indices and iNEXT's numbers, its
    # coverage at 66 cells for the common coverage, and its rarefaction to two
    # cells, counted over every pair of its cells.
    every = by_group.loc["every"]
    assert every[["n_cells", "n_clones", "n_singletons", "n_doubletons"]].tolist() == [
        33,
        12,
        6,
        2,
    ]
    assert every["coverage"] == pytest.approx(1 - 6 / 33 * (32 * 6 / (32 * 6 + 4)))
    draws = list(_subsamples(A, 2))
    richness = sum(len(counts) for _, _, counts in draws) / len(draws)
    entropy = sum(
        -sum(c / 2 * log(c / 2) for c in counts.values()) for _, _, counts in draws
    ) / len(draws)
    squares = sum(sum((c / 2) ** 2 for c in counts.values()) for _, _, counts in draws)
    seen = Fraction(0)
    for chosen, cells, counts in draws:
        rest = [cell for cell in range(33) if cell not in chosen]
        seen += Fraction(sum(cells[cell] in counts for cell in rest), len(rest))
    whole = hill[hill["group"] == "every"]
    np.testing.assert_allclose(
        whole["at_size"],
        [richness, np.exp(entropy), len(draws) / squares],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        whole["coverage_at_size"], float(seen / len(draws)), rtol=1e-12
    )
    assert whole["method_at_size"].tolist() == ["rarefaction"] * 3
    np.testing.assert_allclose(
        whole["observed"], [12, 8.39095516943926, 6.29479768786127], rtol=1e-12
    )
    np.testing.assert_allclose(
        whole["estimated"],
        [20.7272727272727, 11.6060325357083, 7.54285714285715],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        whole["at_coverage"],
        [16.3078174812882, 10.030943287946, 6.86254430878299],
        rtol=1e-11,
    )
    np.testing.assert_allclose(whole["size_at_coverage"], 66, rtol=1e-12)
    assert whole["method_at_coverage"].tolist() == ["extrapolation"] * 3


def test_diversity_tables_refuse_groups_without_cells_and_misaligned_names() -> None:
    names = ["a", "b"]
    with pytest.raises(ValueError, match="Every group must hold a cell"):
        dv.diversity_tables(_grouped(A, []), names)
    with pytest.raises(ValueError, match="names must name each of the 2 groups, not 1"):
        dv.diversity_tables(_grouped(A, B), names[:1])
    with pytest.raises(ValueError, match="names must name each of the 2 groups, not 3"):
        dv.diversity_tables(_grouped(A, B), ["a", "b", "c"])
    # A frame of names and labels names no group, nor do repeated names.
    frame = pd.DataFrame({"group": ["a", "b"], "sample": ["a", "b"]})
    with pytest.raises(ValueError, match="one text name for each group"):
        dv.diversity_tables(_grouped(A, B), frame)
    with pytest.raises(ValueError, match="'a' names two"):
        dv.diversity_tables(_grouped(A, B), ["a", "a"])
    for level in (0.0, -0.5, float("nan")):
        with pytest.raises(ValueError, match="coverage must be between 0 and 1"):
            dv.diversity_tables(_grouped(A, B), names, coverage=level)
    with pytest.raises(ValueError, match="common size of no group"):
        dv.diversity_tables(_grouped(), names[:0])
    with pytest.raises(ValueError, match="common coverage of no group"):
        dv.diversity_tables(_grouped(), names[:0], size=5)
    table, hill, size, level = dv.diversity_tables(
        _grouped(), names[:0], size=5, coverage=0.5
    )
    assert (len(table), len(hill), size, level) == (0, 0, 5, 0.5)
    assert table.columns.tolist() == ["group", *dv.INDEX_COLUMNS]
    # The measures of one group need a cell.
    with pytest.raises(ValueError, match="no diversity indices"):
        dv.indices(_clones([]))
    with pytest.raises(ValueError, match="reaches no coverage"):
        dv.size_for_coverage(_clones([]), 0.5)


def test_shannon_entropy_is_its_terms_summed_exactly() -> None:
    # The extrapolated Shannon diversity divides by the small difference of
    # two diversities near a group's size, so an error in the last place of the
    # observed entropy grows many times; each group's terms are summed within
    # a unit in the last place of their exactly rounded sum.
    rng = np.random.default_rng(7)
    groups = [_clones(np.minimum(rng.zipf(2.2, size=n), 500)) for n in (23_263, 80_000)]
    expected = []
    for clones in groups:
        p = clones.values / clones.n
        expected.append(-fsum((clones.counts * p * np.log(p)).tolist()))
    for clones, exact in zip(groups, expected, strict=True):
        assert abs(dv.indices(clones)["shannon"] - exact) <= ulp(exact)
    names = ["a", "b"]
    table = dv.diversity_tables(dv.GroupedCloneSizes.from_groups(groups), names)[0]
    for got, exact in zip(table["shannon"], expected, strict=True):
        assert abs(got - exact) <= ulp(exact)


def _rows_alone(groups, names, size, coverage) -> list[np.ndarray]:
    """Each group's numbers in the Hill table when it is measured alone."""
    rows = []
    for at in range(len(groups)):
        hill = dv.diversity_tables(
            groups.take([at]),
            names[at : at + 1],
            size=size,
            coverage=coverage,
        )[1]
        numbers = hill.drop(columns=["group", "method_at_size", "method_at_coverage"])
        rows.append(numbers.to_numpy(dtype=np.float64))
    return rows


def test_groups_interpolate_and_search_their_own_coverage() -> None:
    # B and a group of 52 cells both reach coverage 0.9 between two whole
    # sizes, and each interpolates toward its own next size: B at iNEXT's
    # 23.96 cells, within the 1e-7 that iNEXT's root finding leaves.
    names = ["b", "other"]
    groups = _grouped(B, [3, 1, 1, 2, 9, 1, 1, 30, 2, 2])
    hill = dv.diversity_tables(groups, names, size=10, coverage=0.9)[1]
    b = hill[hill["group"] == "b"]
    np.testing.assert_allclose(
        b["at_coverage"],
        [4.92842811594038, 2.87982535549222, 2.21073264251641],
        rtol=1e-7,
    )
    assert b["size_at_coverage"].iloc[0] == pytest.approx(
        23.9610607098077, rel=1e-6, abs=0
    )
    assert hill["method_at_coverage"].tolist() == ["rarefaction"] * 6
    numbers = hill.drop(columns=["group", "method_at_size", "method_at_coverage"])
    for at, alone in enumerate(_rows_alone(groups, names, 10, 0.9)):
        np.testing.assert_array_equal(
            numbers.to_numpy(dtype=np.float64)[3 * at : 3 * at + 3], alone
        )
    # Groups of 9 and 6 cells search from their own coverage at one cell, 11/36
    # and 4/15, and reach 0.35 before two cells, where theirs are 31/63 and
    # 29/60: at 1 + (7/20 - 11/36) / (31/63 - 11/36) = 291/235 and
    # 1 + (7/20 - 4/15) / (29/60 - 4/15) = 18/13 cells.
    names = ["nine", "six"]
    groups = _grouped([1, 1, 2, 5], [3, 1, 2])
    hill = dv.diversity_tables(groups, names, size=3, coverage=0.35)[1]
    reached = hill.groupby("group", sort=False)["size_at_coverage"].first()
    assert reached["nine"] == pytest.approx(291 / 235, rel=1e-14, abs=0)
    assert reached["six"] == pytest.approx(18 / 13, rel=1e-14, abs=0)
    assert hill["method_at_coverage"].tolist() == ["rarefaction"] * 6


def test_a_group_without_singletons_extrapolates_without_warnings() -> None:
    # One clone of 4 cells has neither singleton nor doubleton, so its
    # coverage stays complete at every size; the ratio by which a deficit
    # would shrink exceeds 1 there, and a power of it would overflow.
    clones = _clones([4])
    groups = dv.GroupedCloneSizes.from_groups([clones])
    names = ["a"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dv.coverage_at(clones, 5_000) == 1.0
        hill = dv.diversity_tables(groups, names, size=5_000)[1]
        assert hill["coverage_at_size"].tolist() == [1.0] * 3
        assert hill["at_size"].tolist() == [1.0] * 3
        curves = dv.rarefaction_curves(groups, names, sizes=[10, 5_000])
        assert curves["coverage"].tolist() == [1.0] * 6


def test_grouped_clone_sizes_refuse_positions_outside_the_groups() -> None:
    grouped = _grouped([1, 1], [3, 2, 2])
    for index in (-1, -2, 2):
        with pytest.raises(ValueError, match="index must be between 0 and 1"):
            grouped.group(index)
    for positions in ([-1], [0, 2]):
        with pytest.raises(ValueError, match="positions must be between 0 and 1"):
            grouped.take(positions)


@pytest.mark.parametrize(
    ("n", "a"), [(100, 0.3), (60, 0.5), (16, 0.44), (100_000, 1e-4)]
)
def test_entropy_tail_series_is_summed_exactly(n, a) -> None:
    # Where the factor (1 - a) ** (1 - n) is large the tail is summed as its
    # series, sum((1 - a) ** j / (j + n - 1), j >= 1), here exactly over its
    # terms until they fall below e**-50 of the first; a series that shrinks
    # slowly needs the terms that each stopping point leaves out.
    decay = -np.log1p(-a)
    assert (n - 1) * decay > 5
    j = np.arange(1, int(np.ceil(50 / decay)) + 1, dtype=np.float64)
    expected = fsum((np.exp(-j * decay) / (j + n - 1)).tolist())
    assert dv._entropy_tail(n, a) == pytest.approx(expected, rel=1e-15, abs=0)


def test_two_singletons_extrapolate_their_shannon_diversity() -> None:
    # Two singletons hold D(1) = 1 and D(2) = 2, and their asymptote is
    # exp(1 + 3 log 1.5 - 1) = 1.5**3; the rate (2 - 1) / (3.375 - 1) = 8/19
    # leaves 11/19 per cell, so 4 cells reach 2 + 1.375 (1 - (11/19)**2) =
    # 1052/361.
    clones = _clones([1, 1])
    assert dv.estimated_hill(clones, 1) == pytest.approx(3.375, rel=1e-14, abs=0)
    assert dv.diversity_at(clones, 4, 1) == (
        pytest.approx(1052 / 361, rel=1e-13, abs=0),
        "extrapolation",
    )


def test_curves_extrapolate_shannon_diversity_where_they_pass_a_groups_cells() -> None:
    # B's 82 cells rarefy to 66, and A's 33 extrapolate to them, iNEXT's
    # 10.0309, so only A's Shannon extrapolation is measured.
    groups = dv.GroupedCloneSizes.from_groups([_clones(B), _clones(A)])
    names = ["b", "a"]
    curves = dv.rarefaction_curves(groups, names, (1,), sizes=[66])
    assert curves["method"].tolist() == ["rarefaction", "extrapolation"]
    assert curves["diversity"].iloc[0] == pytest.approx(
        dv.diversity_at(_clones(B), 66, 1)[0], rel=1e-15, abs=0
    )
    assert curves["diversity"].iloc[1] == pytest.approx(
        10.030943287946, rel=1e-11, abs=0
    )
