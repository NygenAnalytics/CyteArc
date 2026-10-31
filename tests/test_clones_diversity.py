"""Diversity indices and Hill numbers, against iNEXT, vegan and exact sums."""

from fractions import Fraction
from itertools import combinations
from math import comb, log

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
            observed[order], rel=1e-12
        )
        assert dv.estimated_hill(clones, order) == pytest.approx(
            estimated[order], rel=1e-10
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
        assert value == pytest.approx(values[order], rel=1e-11)
    assert dv.coverage_at(clones, m) == pytest.approx(coverage, rel=1e-11)


def test_standardizing_by_coverage_matches_inext() -> None:
    a, b = _clones(A), _clones(B)
    # Above the observed coverage the extrapolated coverage is inverted.
    assert dv.size_for_coverage(a, 0.9) == pytest.approx(60.9940672961916, rel=1e-9)
    for order, value in zip(
        dv.ORDERS, (15.8272727272727, 9.85088615975206, 6.81211855586043), strict=True
    ):
        assert dv.diversity_at(a, 60.9940672961916, order)[0] == pytest.approx(
            value, rel=1e-10
        )
    # Below it, rarefied sizes are interpolated linearly, as iNEXT does.
    # iNEXT's root finding stops at a coverage of 0.89999999; the size here
    # reaches 0.9 itself.
    m = dv.size_for_coverage(b, 0.9)
    assert m == pytest.approx(23.9610607098077, rel=1e-6)
    assert dv.coverage_at(b, m) == pytest.approx(0.9, rel=1e-14)
    for order, value in zip(
        dv.ORDERS, (4.92842811594038, 2.87982535549222, 2.21073264251641), strict=True
    ):
        assert dv.diversity_at(b, 23.9610607098077, order)[0] == pytest.approx(
            value, rel=1e-10
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
        assert dv.diversity_at(clones, m, 0)[0] == pytest.approx(richness, rel=1e-12)
        assert dv.diversity_at(clones, m, 1)[0] == pytest.approx(
            np.exp(entropy), rel=1e-12
        )
        assert dv.diversity_at(clones, m, 2)[0] == pytest.approx(1 / squares, rel=1e-12)
        # The coverage of m cells: the chance that one more cell, drawn from
        # the rest, belongs to a clone the m cells hold.
        seen = Fraction(0)
        for chosen, cells, counts in draws:
            rest = [cell for cell in range(n) if cell not in chosen]
            seen += Fraction(sum(cells[cell] in counts for cell in rest), len(rest))
        assert dv.coverage_at(clones, m) == pytest.approx(
            float(seen / len(draws)), rel=1e-12
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
        assert dv.coverage_at(clones, m) == pytest.approx(float(1 - missed), rel=1e-13)
        richness = sum(
            int(count) * (1 - Fraction(comb(n - int(value), m), comb(n, m)))
            for value, count in zip(clones.values, clones.counts, strict=True)
        )
        assert dv.diversity_at(clones, m, 0)[0] == pytest.approx(
            float(richness), rel=1e-13
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
        assert values == pytest.approx([value for value, _ in expected], rel=1e-15)
        assert methods == [method for _, method in expected]


def test_extrapolation_without_singletons() -> None:
    clones = _clones([4, 3, 2, 2])
    # No clone is missed, so richness and coverage stay, while the
    # Shannon estimate still exceeds the observed entropy, as in iNEXT.
    assert dv.diversity_at(clones, 30, 0)[0] == 4
    assert dv.coverage_at(clones, 30) == 1.0
    assert dv.diversity_at(clones, 21, 1)[0] == pytest.approx(
        4.20839264819087, rel=1e-11
    )
    assert dv.diversity_at(clones, 22, 1)[0] == pytest.approx(
        4.22969311355331, rel=1e-11
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
        assert dv._entropy_tail(n, a) == pytest.approx(direct, rel=1e-11)
    # Tails summed as written over more than one chunk, and as a series that
    # needs more than one chunk of terms.
    for n, a in ((2_000_000, 2e-6), (1_000_000, 1e-5)):
        direct = 0.0
        # Terms beyond 60 / a are below e**-60 of the first.
        for start in range(1, int(60 / a), 4_000_000):
            j = np.arange(start, min(start + 4_000_000, int(60 / a)), dtype=np.float64)
            direct += float(np.sum(np.exp(j * np.log1p(-a)) / (j + n - 1)))
        assert dv._entropy_tail(n, a) == pytest.approx(direct, rel=1e-10)


def test_indices_of_a_few_clones() -> None:
    clones = _clones([3, 2, 1, 1])
    got = dv.indices(clones)
    p = np.array([3, 2, 1, 1]) / 7
    assert list(got) == list(dv.INDEX_COLUMNS)
    assert (got["cells"], got["clones"], got["singletons"], got["doubletons"]) == (
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
        14.1687242798354, rel=1e-12
    )
    assert dv.ace(_clones([5, 3, 1])) == pytest.approx(3.62109375, rel=1e-12)
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
    assert np.isnan(dv._gini(_clones([])))


def test_common_levels_follow_inext_defaults() -> None:
    groups = [_clones(A), _clones(B)]
    assert dv.common_size(groups) == 66
    assert dv.common_coverage(groups) == pytest.approx(
        min(dv.coverage_at(groups[0], 66), dv.coverage_at(groups[1], 164))
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
    assert dv.diversity_at(clones, m, 1)[0] == pytest.approx(expected, rel=1e-13)
    values, methods = dv.hill_curve(clones, [m, m, 1, clones.n], 1)
    assert values[:2] == pytest.approx([expected, expected], rel=1e-13)
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
        assert dv.size_for_coverage(clones, level) == pytest.approx(expected, rel=1e-13)


def test_diversity_tables_match_inext_for_every_group() -> None:
    groups = [_clones(A), _clones(B)]
    keys = pd.DataFrame({"group": ["a, s1", "b, s2"], "sample": ["s1", "s2"]})
    table, hill, size, level = dv.diversity_tables(groups, keys)
    # iNEXT's defaults: the smallest doubled size and the lowest coverage at
    # doubled sizes.
    assert size == 66
    assert level == pytest.approx(0.909807035796235, rel=1e-11)
    assert table.columns.tolist() == ["group", "sample", *dv.INDEX_COLUMNS]
    assert table["group"].tolist() == ["a, s1", "b, s2"]
    assert table[["cells", "clones", "singletons", "doubletons"]].values.tolist() == [
        [33, 12, 6, 2],
        [82, 9, 4, 2],
    ]
    assert table["coverage"].round(4).tolist() == [0.8219, 0.9518]
    assert hill.columns.tolist() == [
        "group",
        "sample",
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
    assert a["size_at_coverage"].tolist() == pytest.approx([66.0] * 3, rel=1e-12)
    np.testing.assert_allclose(a["at_coverage"], a["at_size"], rtol=1e-11)
    at_coverage = dv.diversity_tables(groups, keys, size=23, coverage=0.9)[1]
    a, b = at_coverage.iloc[:3], at_coverage.iloc[3:]
    np.testing.assert_allclose(
        a["at_coverage"],
        [15.8272727272727, 9.85088615975206, 6.81211855586043],
        rtol=1e-10,
    )
    assert b["size_at_coverage"].iloc[0] == pytest.approx(23.9610607098077, rel=1e-6)
    np.testing.assert_allclose(
        b["at_size"], [4.83087682300689, 2.86343404471686, 2.20587980477662], rtol=1e-11
    )
    assert b["coverage_at_size"].iloc[0] == pytest.approx(0.898496221999323, rel=1e-11)
    assert a["method_at_coverage"].tolist() == ["extrapolation"] * 3
    assert b["method_at_coverage"].tolist() == ["rarefaction"] * 3


def test_rarefaction_curves_hold_each_group_order_and_size() -> None:
    groups = [_clones(A), _clones(B)]
    keys = pd.DataFrame({"group": ["a", "b"]})
    curves = dv.rarefaction_curves(
        groups, keys, (2, 0, 1), sizes=[164, 16, 23, 50, 66, 16]
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
            assert row["diversity"] == pytest.approx(value, rel=1e-11)
            assert row["coverage"] == pytest.approx(sample_coverage, rel=1e-11)
    defaults = dv.rarefaction_curves(groups, keys, (0,), endpoint=40, knots=4)
    # A's 33 cells extrapolate to the endpoint; B's 82 rarefy to it.
    assert defaults["size"].tolist() == [1, 33, 34, 40, 1, 14, 27, 40]
    assert (
        dv.rarefaction_curves([], keys.iloc[:0]).columns.tolist()
        == curves.columns.tolist()
    )
    with pytest.raises(ValueError, match="order must be"):
        dv.rarefaction_curves(groups, keys, (3,))
