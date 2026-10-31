"""Fate coupling, its null and the fate bias of early cells, against values derived by hand.

The references compute each coupling from a table of each clone's cells in
each group, written from the published definitions, and enumerate every
arrangement of a small experiment's labels for the exact null.
"""

from collections import Counter
from itertools import permutations
from math import sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from cytearc.clones.fate import (
    COUPLING_METHODS,
    clone_coupling,
    coupling_bytes,
    coupling_linkage,
    coupling_plan,
    dominant_fates,
    fate_bias,
    fate_bias_tables,
    permuted_labels,
)
from cytearc.clones.groups import pair_counts
from cytearc.clones.overlap import overlap_pairs
from tests.clone_helpers import benjamini_hochberg, coupling_reference


def _pairs(groups, clones):
    """Count each distinct pair of group and clone, as the summaries count them."""
    return pair_counts(np.asarray(groups), np.asarray(clones))


def _table(groups, clones) -> dict[int, Counter]:
    """Each group's cells of each clone."""
    table: dict[int, Counter] = {}
    for group, clone in zip(groups, clones, strict=True):
        table.setdefault(int(group), Counter())[int(clone)] += 1
    return table


# Group 0 holds clone 0 twice and clone 1 three times; group 1 holds clone 0
# once and clone 2 four times: the groups share exactly clone 0.
SHARED_ONE = (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 2]), np.array([2, 3, 1, 4]))


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # Shares: clone 0 is 2/5 of group 0 and 1/5 of group 1, so 2/3 and
        # 1/3 of its shares; clones 1 and 2 are wholly in one group each.
        # (2/9) / sqrt((4/9 + 1) (1/9 + 1)) = 2 / sqrt(130).
        ("share_cosine", 2 / sqrt(130)),
        # 2 * 1 / sqrt((4 + 9) (1 + 16)).
        ("abundance_cosine", 2 / sqrt(221)),
        # Three clones, (2, 3, 0) and (1, 0, 4), means 5/3: covariance -57/18
        # over 25/9.
        ("normalized_covariance", -57 / 50),
        # One clone of three.
        ("jaccard", 1 / 3),
    ],
)
def test_two_groups_that_share_one_clone_couple_as_derived_by_hand(
    method, expected
) -> None:
    pairs, tree = clone_coupling(*SHARED_ONE, 2, method=method, permutations=0)
    assert pairs.columns.tolist() == [
        "first",
        "second",
        "coupling",
        "null_mean",
        "null_sd",
        "z_score",
    ]
    assert pairs[["first", "second"]].values.tolist() == [[0, 1]]
    assert pairs["coupling"].item() == pytest.approx(expected, rel=1e-14)
    assert pairs[["null_mean", "null_sd", "z_score"]].isna().all(axis=None)
    assert tree.shape == (1, 4)


def test_a_clone_confined_to_one_group_adds_nothing_to_the_shared_terms() -> None:
    # Two groups whose clones are each confined to one group share nothing,
    # and their covariance over two clones is -2 whatever the sizes.
    disjoint = (np.array([0, 1]), np.array([0, 1]), np.array([5, 3]))
    for method, expected in (
        ("share_cosine", 0.0),
        ("abundance_cosine", 0.0),
        ("normalized_covariance", -2.0),
        ("jaccard", 0.0),
    ):
        pairs, _tree = clone_coupling(*disjoint, 2, method=method, permutations=0)
        assert pairs["coupling"].item() == pytest.approx(expected, abs=1e-15)
    # Clone 3 of four cells, confined to group 0, leaves the product of the
    # shared clone and only adds to group 0's own squares: 2 / sqrt(29 * 17).
    groups, clones, cells = SHARED_ONE
    pairs, _tree = clone_coupling(
        np.append(groups, 0),
        np.append(clones, 3),
        np.append(cells, 4),
        2,
        method="abundance_cosine",
        permutations=0,
    )
    assert pairs["coupling"].item() == pytest.approx(2 / sqrt(493), rel=1e-14)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("method", COUPLING_METHODS)
def test_couplings_match_a_reference_from_the_counts(seed, method) -> None:
    rng = np.random.default_rng(seed)
    n_cells, n_groups = int(rng.integers(30, 300)), int(rng.integers(2, 6))
    clones = np.minimum(rng.zipf(1.6, size=n_cells), 30) - 1
    groups = rng.integers(0, n_groups, size=n_cells)
    groups[:n_groups] = np.arange(n_groups)
    pairs, _tree = clone_coupling(
        *_pairs(groups, clones), n_groups, method=method, permutations=0
    )
    table = _table(groups, clones)
    assert len(pairs) == n_groups * (n_groups - 1) // 2
    for row in pairs.itertuples(index=False):
        assert row.coupling == pytest.approx(
            coupling_reference(table, row.first, row.second, method),
            rel=1e-12,
            abs=1e-14,
        )


def test_the_cosine_of_clone_sizes_and_the_jaccard_index_equal_overlaps() -> None:
    rng = np.random.default_rng(11)
    clones = np.minimum(rng.zipf(1.5, size=400), 60) - 1
    groups = rng.integers(0, 5, size=400)
    counted = _pairs(groups, clones)
    overlap = overlap_pairs(*counted, 5)
    for method in ("abundance_cosine", "jaccard"):
        pairs, _tree = clone_coupling(*counted, 5, method=method, permutations=0)
        # Both sum whole numbers, so they are equal to the last bit.
        np.testing.assert_array_equal(
            pairs["coupling"].to_numpy(), overlap[method].to_numpy()
        )


def test_the_shuffles_keep_each_strata_labels_and_vary_with_the_permutation() -> None:
    labels = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4], dtype=np.int32)
    bounds = np.array([0, 3, 6, 9])
    seen = set()
    for permutation in range(100):
        shuffled = permuted_labels(labels, bounds, 7, permutation)
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            assert sorted(shuffled[start:end]) == sorted(labels[start:end])
        seen.add(tuple(shuffled.tolist()))
        np.testing.assert_array_equal(
            permuted_labels(labels, bounds, 7, permutation), shuffled
        )
    # Three orders of the first stratum, one of the second, three of the third.
    assert len(seen) == 9
    assert not np.array_equal(
        permuted_labels(labels, bounds, 7, 0), permuted_labels(labels, bounds, 8, 0)
    )


def test_the_shuffles_draw_every_arrangement_equally_often() -> None:
    # Fisher and Yates's shuffle draws each of the 24 orders of four labels
    # with chance 1/24; a shuffle that never leaves a label in place, as
    # Sattolo's does, would draw 6 of them and never the original order.
    labels = np.arange(4, dtype=np.int32)
    n_draws = 24_000
    counts = Counter(
        tuple(permuted_labels(labels, np.array([0, 4]), 3, k).tolist())
        for k in range(n_draws)
    )
    assert set(counts) == set(permutations(range(4)))
    expected = n_draws / 24
    chi_square = sum((count - expected) ** 2 / expected for count in counts.values())
    # The 0.999 quantile of the chi-square distribution with 23 degrees of
    # freedom is 49.7.
    assert chi_square < 49.7


def _exact_null(groups, clone_sizes, method):
    """Each pair's null mean and spread, over every arrangement of the labels."""
    cells_of_clone = np.repeat(np.arange(len(clone_sizes)), clone_sizes)
    values: dict[tuple[int, int], list[float]] = {}
    for arrangement in set(permutations(groups)):
        table = _table(arrangement, cells_of_clone)
        for first in range(max(groups) + 1):
            for second in range(first + 1, max(groups) + 1):
                values.setdefault((first, second), []).append(
                    coupling_reference(table, first, second, method)
                )
    return {pair: (np.mean(found), np.std(found)) for pair, found in values.items()}


@pytest.mark.parametrize("method", COUPLING_METHODS)
def test_the_null_matches_every_arrangement_of_the_labels(method) -> None:
    # Six cells in groups of 3, 2 and 1 cells and clones of 3, 2 and 1 cells:
    # 60 arrangements of the labels, each as likely under shuffling.
    groups = (0, 0, 0, 1, 1, 2)
    clones = np.repeat(np.arange(3), (3, 2, 1))
    n_permutations = 20_000
    pairs, _tree = clone_coupling(
        *_pairs(groups, clones),
        3,
        method=method,
        permutations=n_permutations,
        seed=5,
    )
    exact = _exact_null(groups, (3, 2, 1), method)
    for row in pairs.itertuples(index=False):
        mean, spread = exact[(row.first, row.second)]
        assert abs(row.null_mean - mean) < 4 * spread / sqrt(n_permutations)
        assert row.null_sd == pytest.approx(spread, rel=0.05)
        assert row.z_score == pytest.approx(
            (row.coupling - row.null_mean) / row.null_sd, rel=1e-12
        )


@pytest.mark.parametrize("method", COUPLING_METHODS)
def test_the_null_is_the_coupling_of_the_arrangements_permuted_labels_draws(
    method,
) -> None:
    rng = np.random.default_rng(6)
    clones = np.minimum(rng.zipf(1.8, size=60), 9) - 1
    groups = rng.integers(0, 3, size=60)
    groups[:3] = np.arange(3)
    n_permutations = 5
    pairs, _tree = clone_coupling(
        *_pairs(groups, clones),
        3,
        method=method,
        permutations=n_permutations,
        seed=12,
        batch=2,
    )
    # The cells with a clone, clone after clone and each clone's groups in
    # ascending order, as the null lays them out, and each permutation of
    # their labels by permuted_labels.
    order = np.lexsort((groups, clones))
    labels, cell_clones = groups[order], clones[order]
    found = []
    for permutation in range(n_permutations):
        shuffled = permuted_labels(labels, np.array([0, 60]), 12, permutation)
        table = _table(shuffled, cell_clones)
        found.append(
            [
                coupling_reference(table, first, second, method)
                for first, second in ((0, 1), (0, 2), (1, 2))
            ]
        )
    values = np.asarray(found)
    np.testing.assert_allclose(pairs["null_mean"], values.mean(axis=0), rtol=1e-12)
    # The spread has n_permutations - 1 degrees of freedom.
    np.testing.assert_allclose(pairs["null_sd"], values.std(axis=0, ddof=1), rtol=1e-12)


def test_the_null_mean_of_the_normalized_covariance_is_its_expectation() -> None:
    # Shuffling places two cells of a clone of s cells in groups f and g with
    # chance n_f n_g / (N (N - 1)) for each of its s (s - 1) ordered pairs of
    # cells, so the mean of sum(x y) is n_f n_g S2 / (N (N - 1)) with S2 the
    # sum of s (s - 1), and the coupling, affine in it, has the mean
    # C (C S2 / (N (N - 1)) - 1) / (C - 1) for every pair.
    rng = np.random.default_rng(3)
    clones = np.minimum(rng.zipf(1.7, size=300), 25) - 1
    groups = rng.integers(0, 4, size=300)
    sizes = np.bincount(clones)
    sizes = sizes[sizes > 0]
    n_clones, n_cells = len(sizes), int(sizes.sum())
    pair_sum = float((sizes * (sizes - 1)).sum())
    expected = n_clones * (n_clones * pair_sum / (n_cells * (n_cells - 1)) - 1)
    expected /= n_clones - 1
    n_permutations = 4_000
    pairs, _tree = clone_coupling(
        *_pairs(groups, clones),
        4,
        method="normalized_covariance",
        permutations=n_permutations,
        seed=1,
    )
    for row in pairs.itertuples(index=False):
        assert abs(row.null_mean - expected) < 4 * row.null_sd / sqrt(n_permutations)


def test_clones_of_one_cell_give_a_null_without_spread() -> None:
    # Clones of one cell lie in one group, so every arrangement couples alike:
    # no shared clone, and a covariance of -C / (C - 1).
    groups, clones = np.array([0, 0, 1, 1, 2]), np.arange(5)
    for method, expected in (
        ("share_cosine", 0.0),
        ("abundance_cosine", 0.0),
        ("normalized_covariance", -5 / 4),
        ("jaccard", 0.0),
    ):
        pairs, _tree = clone_coupling(
            *_pairs(groups, clones), 3, method=method, permutations=50
        )
        np.testing.assert_allclose(pairs["coupling"], expected, atol=1e-15)
        np.testing.assert_array_equal(pairs["null_mean"], pairs["coupling"])
        np.testing.assert_array_equal(pairs["null_sd"], 0.0)
        assert pairs["z_score"].isna().all()


@pytest.mark.parametrize("method", COUPLING_METHODS)
def test_the_null_depends_neither_on_workers_nor_on_batches(method) -> None:
    rng = np.random.default_rng(4)
    clones = np.minimum(rng.zipf(1.6, size=2_000), 80) - 1
    groups = rng.integers(0, 6, size=2_000)
    counted = _pairs(groups, clones)
    runs = [
        clone_coupling(
            *counted, 6, method=method, permutations=101, workers=w, batch=b
        )[0].to_numpy()
        for w, b in ((1, None), (1, 1), (3, 7), (4, 101), (8, 16))
    ]
    for run in runs[1:]:
        np.testing.assert_array_equal(run, runs[0])
    # Another seed draws other permutations, and the same coupling.
    other = clone_coupling(*counted, 6, method=method, permutations=101, seed=1)[0]
    np.testing.assert_array_equal(other["coupling"].to_numpy(), runs[0][:, 2])
    assert not np.array_equal(other["null_mean"].to_numpy(), runs[0][:, 3])


def test_strata_compare_and_shuffle_only_the_groups_of_one_stratum() -> None:
    # Groups 0 and 1 hold clones 0 to 2, groups 2 and 3 clones 3 to 5.
    rng = np.random.default_rng(8)
    clones = rng.integers(0, 6, size=200)
    groups = np.where(clones < 3, 0, 2) + rng.integers(0, 2, size=200)
    counted = _pairs(groups, clones)
    strata = np.array(["a", "a", "b", "b"])
    pairs, tree = clone_coupling(
        *counted,
        4,
        method="normalized_covariance",
        strata=strata,
        permutations=200,
    )
    assert pairs[["first", "second"]].values.tolist() == [[0, 1], [2, 3]]
    # Each stratum is measured on its own clones, as if alone.
    for row, kept in zip(pairs.itertuples(index=False), (clones < 3, clones >= 3)):
        alone, _tree = clone_coupling(
            *_pairs(groups[kept] % 2, clones[kept] % 3),
            2,
            method="normalized_covariance",
            permutations=0,
        )
        assert row.coupling == pytest.approx(alone["coupling"].item(), rel=1e-12)
    # Labels shuffled within strata keep the clones of a stratum within it,
    # so the null is that of each stratum alone.
    assert pairs["null_sd"].gt(0).all()
    # The two strata join last.
    assert tree[-1, 2] > tree[:-1, 2].max()


def test_no_label_crosses_into_another_stratum() -> None:
    # Groups 0 and 1, of stratum a, hold clones of one cell each, which
    # couple alike however their labels are shuffled; groups 2 and 3, of b,
    # hold clones of four cells. A label of b shuffled into a would bring
    # cells of those clones into groups 0 and 1 and spread their null.
    groups = np.array([0, 0, 1, 1] + [2, 3] * 8)
    clones = np.array([0, 1, 2, 3] + [4 + at // 4 for at in range(16)])
    for method in COUPLING_METHODS:
        pairs, _tree = clone_coupling(
            *_pairs(groups, clones),
            4,
            method=method,
            strata=np.array(["a", "a", "b", "b"]),
            permutations=200,
        )
        first = pairs.iloc[0]
        assert (first["first"], first["second"]) == (0, 1)
        assert first["null_sd"] == 0.0 and first["null_mean"] == first["coupling"]
        assert pairs.iloc[1]["null_sd"] > 0


def test_coupling_refuses_what_it_cannot_measure() -> None:
    groups, clones, cells = SHARED_ONE
    with pytest.raises(ValueError, match="method must be one of"):
        clone_coupling(groups, clones, cells, 2, method="cosine")
    with pytest.raises(ValueError, match="whole numbers"):
        clone_coupling(groups, clones, cells + 0.5, 2, permutations=0)
    with pytest.raises(ValueError, match="Every group needs a cell"):
        clone_coupling(groups, clones, cells, 3, permutations=0)
    with pytest.raises(ValueError, match="one stratum per group"):
        clone_coupling(groups, clones, cells, 2, strata=np.zeros(3), permutations=0)
    with pytest.raises(ValueError, match="No two groups share a stratum"):
        clone_coupling(groups, clones, cells, 2, strata=np.arange(2), permutations=0)
    # Clone 0 lies in groups of two strata.
    with pytest.raises(ValueError, match="groups of two strata"):
        clone_coupling(
            np.array([0, 1, 2, 2]),
            np.array([0, 1, 0, 2]),
            np.ones(4),
            3,
            strata=np.array([0, 0, 1]),
            permutations=0,
        )
    with pytest.raises(ValueError, match="at least two clones"):
        clone_coupling(
            np.array([0, 1]),
            np.array([0, 0]),
            np.array([1, 2]),
            2,
            method="normalized_covariance",
        )


def test_the_tree_joins_the_most_coupled_groups_first() -> None:
    # Groups 0 and 1 couple most, then 2 and 3; every other pair little.
    first = np.array([0, 0, 0, 1, 1, 2])
    second = np.array([1, 2, 3, 2, 3, 3])
    coupling = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.8])
    tree = coupling_linkage(first, second, coupling, 4)
    # Distances from the largest coupling: 0 for (0, 1), 0.1 for (2, 3) and
    # 0.8 between the two pairs.
    np.testing.assert_allclose(tree[:, 2], [0.0, 0.1, 0.8], atol=1e-15)
    assert {tuple(sorted(tree[0, :2].astype(int)))} == {(0, 1)}
    assert {tuple(sorted(tree[1, :2].astype(int)))} == {(2, 3)}
    # Average linkage: groups 0 and 1 join first, and group 2 joins them at
    # the mean of its distances to both, (0.9 - 0.5 + 0.9 - 0.1) / 2 = 0.6,
    # where complete linkage would join at 0.8 and single linkage at 0.4.
    three = coupling_linkage(
        np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([0.9, 0.5, 0.1]), 3
    )
    np.testing.assert_allclose(three[:, 2], [0.0, 0.6], atol=1e-15)
    # A pair of two strata, not compared, joins after every compared pair.
    split = coupling_linkage(
        np.array([0, 2]), np.array([1, 3]), np.array([0.2, -1.0]), 4
    )
    assert split[-1, 2] == pytest.approx(0.2 - -1.0 + 1.0)
    with pytest.raises(ValueError, match="finite coupling"):
        coupling_linkage(first, second, np.append(coupling[:-1], np.nan), 4)


def test_the_plan_fits_the_null_into_the_memory_left() -> None:
    sizes = (10, 45, 50_000, 30_000)
    roomy = coupling_plan(*sizes, permutations=1000, nthreads=4, available=10**9)
    assert roomy[:2] == (4, 64)
    assert roomy[2] == coupling_bytes(*sizes, workers=4, batch=64)
    # Without permutations the null needs neither workers nor rows.
    assert coupling_plan(*sizes, permutations=0, nthreads=4, available=0) == (
        0,
        0,
        coupling_bytes(*sizes, workers=0, batch=0),
    )
    # Tighter budgets measure fewer permutations at once, then use fewer
    # workers, and at last one worker and one permutation.
    one = coupling_bytes(*sizes, workers=1, batch=1)
    four = coupling_bytes(*sizes, workers=4, batch=4)
    assert coupling_plan(*sizes, permutations=1000, nthreads=4, available=four) == (
        4,
        4,
        four,
    )
    assert (
        coupling_plan(*sizes, permutations=1000, nthreads=4, available=four - 1)[0] == 2
    )
    assert coupling_plan(*sizes, permutations=1000, nthreads=4, available=one) == (
        1,
        1,
        one,
    )
    assert coupling_plan(*sizes, permutations=1000, nthreads=4, available=one - 1) == (
        1,
        1,
        one,
    )
    # Never more workers or rows than permutations.
    assert coupling_plan(*sizes, permutations=3, nthreads=8, available=10**9)[:2] == (
        3,
        3,
    )
    # With millions of pairs, the rows measured at once hold at most 64 MiB
    # beyond one per worker.
    many = (2_000, 1_999_000, 1_000_000, 300_000)
    assert coupling_plan(*many, permutations=1000, nthreads=8, available=10**12)[
        :2
    ] == (8, 8)
    assert coupling_plan(
        *(10, 45_000, 50, 30), permutations=1000, nthreads=1, available=10**12
    )[:2] == (1, 16)
    assert 64 * 1024**2 // (8 * 200_000) == 41
    assert coupling_plan(
        *(10, 200_000, 50, 30), permutations=1000, nthreads=8, available=10**12
    )[:2] == (8, 41)


# Ten late cells: clone 0 holds three neutrophils, clone 1 two monocytes and
# clone 2 two monocytes; one neutrophil and two monocytes have no clone.
LATE_FATES = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
LATE_CLONES = np.array([0, 0, 0, -1, 1, 1, 2, -1, -1, 2])


def _late_pairs():
    counted = LATE_CLONES >= 0
    return (
        *_pairs(LATE_FATES[counted], LATE_CLONES[counted]),
        np.bincount(LATE_FATES, minlength=2),
    )


def test_fate_bias_follows_each_early_cell_to_its_clones_late_cells() -> None:
    # Early cells of clones 0, 0, 1 and 3, one without a clone, one of 2.
    result = fate_bias(np.array([0, 0, 1, 3, -1, 2]), *_late_pairs())
    assert result.clones.tolist() == [0, 1, 2]
    assert result.clone_cells.tolist() == [[3, 0], [0, 2], [0, 2]]
    assert result.clone_early_cells.tolist() == [2, 1, 1]
    np.testing.assert_array_equal(
        result.fractions,
        [[1, 0], [1, 0], [0, 1], [np.nan, np.nan], [np.nan, np.nan], [0, 1]],
    )
    # Clone 3 has no late cell and no cell has clone -1: no bias.
    assert result.late_cells.tolist() == [3, 3, 2, 0, 0, 2]
    assert result.dominant.tolist() == [0, 0, 1, -1, -1, 1]
    assert result.fate_cells.tolist() == [4, 6]
    # Three neutrophils of clone 0 among four of ten late cells: the chance
    # of three in a draw of three is C(4, 3) / C(10, 3) = 1/30; two monocytes
    # of two among six: C(6, 2) / C(10, 2) = 1/3; no cell in a fate: 1.
    np.testing.assert_allclose(
        result.p_values, [[1 / 30, 1], [1, 1 / 3], [1, 1 / 3]], rtol=1e-12
    )
    # Benjamini and Hochberg within each fate: 1/30 * 3 / 1, and 1/3 * 3 / 2
    # for both monocyte clones.
    np.testing.assert_allclose(
        result.q_values, [[0.1, 1], [1, 0.5], [1, 0.5]], rtol=1e-12
    )


def test_an_early_cell_of_a_clone_between_tested_clones_has_no_bias() -> None:
    # Clones 0 and 2 have late cells and clone 1 none: an early cell of clone
    # 1 falls between them among the tested clones and has no bias.
    fates, clones = np.array([0, 1, 1]), np.array([0, 2, 2])
    result = fate_bias(np.array([1, 2]), *_pairs(fates, clones), np.array([1, 2]))
    assert result.clones.tolist() == [0, 2]
    assert np.isnan(result.fractions[0]).all()
    assert result.fractions[1].tolist() == [0.0, 1.0]
    assert (result.late_cells.tolist(), result.dominant.tolist()) == ([0, 2], [-1, 1])
    assert result.clone_early_cells.tolist() == [0, 1]


def test_fate_bias_p_values_are_one_sided_fisher_tests() -> None:
    rng = np.random.default_rng(2)
    fates = rng.integers(0, 4, size=500)
    clones = np.where(rng.random(500) < 0.8, rng.integers(0, 40, size=500), -1)
    clones = np.where(clones >= 0, clones + (fates == 1) * (clones % 3 == 0), -1)
    counted = clones >= 0
    result = fate_bias(
        np.arange(-1, 41),
        *_pairs(fates[counted], clones[counted]),
        np.bincount(fates, minlength=4),
    )
    late = result.clone_cells.sum(axis=1)
    for row, clone in enumerate(result.clones.tolist()):
        for fate in range(4):
            x = int(result.clone_cells[row, fate])
            in_fate = int((fates == fate).sum())
            table = [[x, late[row] - x], [in_fate - x, 500 - in_fate - late[row] + x]]
            expected = fisher_exact(table, alternative="greater").pvalue
            assert result.p_values[row, fate] == pytest.approx(expected, rel=1e-9)
    # The adjustment within each fate, step by step from the largest p-value.
    for fate in range(4):
        np.testing.assert_allclose(
            result.q_values[:, fate],
            benjamini_hochberg(result.p_values[:, fate]),
            rtol=1e-12,
        )


def test_an_early_cell_of_a_tied_clone_has_no_dominant_fate() -> None:
    # Clone 0 holds one late cell in each fate.
    fates, clones = np.array([0, 1]), np.array([0, 0])
    result = fate_bias(np.array([0]), *_pairs(fates, clones), np.array([1, 1]))
    np.testing.assert_array_equal(result.fractions, [[0.5, 0.5]])
    assert result.dominant.tolist() == [-1]
    assert dominant_fates(np.array([[1, 1], [0, 0], [2, 1], [0, 3]])).tolist() == [
        -1,
        -1,
        0,
        1,
    ]
    assert dominant_fates(np.zeros((2, 0))).tolist() == [-1, -1]
    with pytest.raises(ValueError, match="one row per clone"):
        dominant_fates(np.zeros(3))


def test_fate_bias_tests_each_clone_within_its_stratum() -> None:
    # Fates 0 and 1 are of stratum a, fate 2 of b; clone 0 of a, clone 1 of b.
    fates = np.array([0, 0, 1, 2, 2, 1, 2])
    clones = np.array([0, 0, 0, 1, 1, -1, -1])
    counted = clones >= 0
    result = fate_bias(
        np.array([0, 1]),
        *_pairs(fates[counted], clones[counted]),
        np.bincount(fates),
        np.array(["a", "a", "b"]),
    )
    # Clone 0 is tested in fates 0 and 1 among the four late cells of a: two
    # of its three in fate 0, of two such cells, has the chance C(2, 2) C(2, 1)
    # / C(4, 3) = 1/2, and one in fate 1 at least 1. Clone 1 is tested in fate
    # 2 alone, among the three late cells of b, all of them in fate 2.
    assert np.isnan(result.p_values[0, 2]) and np.isnan(result.p_values[1, :2]).all()
    assert result.p_values[0, 0] == pytest.approx(0.5, rel=1e-12)
    assert result.p_values[0, 1] == pytest.approx(1.0, rel=1e-12)
    assert result.p_values[1, 2] == pytest.approx(1.0, rel=1e-12)
    assert np.isnan(result.q_values[0, 2])
    with pytest.raises(ValueError, match="fates of two strata"):
        fate_bias(
            np.array([0]),
            *_pairs(np.array([0, 2]), np.array([0, 0])),
            np.array([1, 0, 1]),
            np.array(["a", "a", "b"]),
        )


def test_fate_bias_refuses_inconsistent_counts() -> None:
    pairs = _late_pairs()
    with pytest.raises(ValueError, match="fewer late cells than its pairs"):
        fate_bias(np.array([0]), *pairs[:3], np.array([1, 1]))
    with pytest.raises(ValueError, match="a fate below"):
        fate_bias(np.array([0]), *pairs[:3], np.array([4]))
    with pytest.raises(ValueError, match="one-dimensional"):
        fate_bias(np.array([[0]]), *pairs)
    with pytest.raises(ValueError, match="one stratum per fate"):
        fate_bias(np.array([0]), *pairs, np.array([0]))


def test_the_tables_name_cells_clones_and_fates() -> None:
    result = fate_bias(np.array([0, -1, 1]), *_late_pairs())
    fates = pd.DataFrame({"group": ["Neu", "Mono"], "cell_type": ["Neu", "Mono"]})
    cells, table = fate_bias_tables(
        np.array(["e0", "e1", "e2"]),
        codes=np.array([0, -1, 1]),
        dominant=result.dominant,
        late_cells=result.late_cells,
        fractions=result.fractions,
        clone_ids=np.array(["c0", "c1", "c2"], dtype=object),
        clones=result.clones,
        clone_cells=result.clone_cells,
        clone_early_cells=result.clone_early_cells,
        p_values=result.p_values,
        q_values=result.q_values,
        fates=fates,
    )
    assert cells.index.tolist() == ["e0", "e1", "e2"]
    assert cells.columns.tolist() == ["clone", "late_cells", "fate", "Neu", "Mono"]
    assert cells["clone"].tolist()[0] == "c0" and pd.isna(cells["clone"].iloc[1])
    assert [
        name if isinstance(name, str) else None for name in cells["fate"].tolist()
    ] == ["Neu", None, "Mono"]
    assert table.columns.tolist() == [
        "clone",
        "fate",
        "cell_type",
        "early_cells",
        "late_cells",
        "cells",
        "fraction",
        "p_value",
        "q_value",
    ]
    assert table[["clone", "fate", "cells"]].astype(object).values.tolist() == [
        ["c0", "Neu", 3],
        ["c0", "Mono", 0],
        ["c1", "Neu", 0],
        ["c1", "Mono", 2],
        ["c2", "Neu", 0],
        ["c2", "Mono", 2],
    ]
    assert table["early_cells"].tolist() == [1, 1, 1, 1, 0, 0]
    assert table["fraction"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_a_clones_cells_in_any_order_couple_to_the_last_bit() -> None:
    from cytearc.clones.fate import _arrangement_coupling

    # Groups of 3, 7 and 11 cells; clone 0 holds 2, 1 and 5 of them, whose
    # shares 2/3, 1/7 and 5/11 sum to different last bits in different
    # orders, and clones 1 to 3 hold the rest.
    bounds = np.array([0, 8, 9, 15, 21])
    rest = [0] + [1] * 6 + [2] * 6
    pairs = (np.array([0, 0, 1]), np.array([1, 2, 2]))

    def coupling(first_clone: list[int]) -> np.ndarray:
        out = np.empty(3)
        _arrangement_coupling(
            np.asarray(first_clone + rest, dtype=np.int32),
            bounds,
            0,
            np.array([3.0, 7.0, 11.0]),
            np.zeros(3, dtype=np.int64),
            np.array([4.0]),
            *pairs,
            np.zeros(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
            np.zeros(3),
            np.zeros((3, 3)),
            np.zeros(3),
            out,
        )
        return out

    np.testing.assert_array_equal(
        coupling([0, 0, 1, 2, 2, 2, 2, 2]), coupling([2, 2, 1, 0, 2, 0, 2, 2])
    )
