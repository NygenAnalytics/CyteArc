"""Overlap indices and clone tracking, against set-by-set references by hand."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.overlap import (
    OVERLAP_INDICES,
    group_clone_matrix,
    overlap_extent,
    overlap_pairs,
    top_clones,
    tracking_cells,
    tracking_reach,
)
from tests.clone_helpers import overlap_reference


def _pairs(groups: np.ndarray, clones: np.ndarray):
    """Count each distinct pair of group and clone."""
    counts = Counter(zip(groups.tolist(), clones.tolist(), strict=True))
    keys = sorted(counts)
    return (
        np.asarray([group for group, _ in keys]),
        np.asarray([clone for _, clone in keys]),
        np.asarray([counts[key] for key in keys]),
    )


def _reference(groups, clones, first, second):
    """The indices of one pair, from dictionaries of clone sizes."""
    x = Counter(c for g, c in zip(groups, clones, strict=True) if g == first)
    y = Counter(c for g, c in zip(groups, clones, strict=True) if g == second)
    return overlap_reference(x, y)


@pytest.mark.parametrize("seed", range(10))
def test_pairs_match_a_set_by_set_reference(seed) -> None:
    rng = np.random.default_rng(seed)
    n_cells, n_groups = int(rng.integers(20, 400)), int(rng.integers(2, 6))
    clones = np.minimum(rng.zipf(1.7, size=n_cells), 40) - 1
    groups = rng.integers(0, n_groups, size=n_cells)
    groups[:n_groups] = np.arange(n_groups)  # every group holds a cell
    table = overlap_pairs(*_pairs(groups, clones), n_groups)
    assert table.columns.tolist() == [
        "first",
        "second",
        *OVERLAP_INDICES,
        "shared_fraction_1",
        "shared_fraction_2",
    ]
    assert len(table) == n_groups * (n_groups - 1) // 2
    for row in table.itertuples(index=False):
        expected = _reference(groups.tolist(), clones.tolist(), row.first, row.second)
        for column, value in expected.items():
            assert getattr(row, column) == pytest.approx(value, rel=1e-12)


def test_strata_keep_pairs_within_one_stratum() -> None:
    groups = np.array([0, 0, 1, 1, 2, 2, 3])
    clones = np.array([0, 1, 0, 2, 3, 3, 3])
    table = overlap_pairs(
        *_pairs(groups, clones), 4, strata=np.array(["T", "T", "B", "B"])
    )
    assert table[["first", "second"]].values.tolist() == [[0, 1], [2, 3]]
    assert table["shared"].tolist() == [1, 1]
    assert table["morisita_horn"].tolist() == pytest.approx(
        [2 * 1 / ((2 / 4 + 2 / 4) * 4), 1.0]
    )


def test_disjoint_groups_share_nothing() -> None:
    table = overlap_pairs(np.array([0, 1]), np.array([0, 1]), np.array([3, 2]), 2)
    row = table.iloc[0]
    assert row["shared"] == 0
    for column in (
        "jaccard",
        "sorensen",
        "overlap",
        "cosine",
        "morisita_horn",
        "abundance_cosine",
    ):
        assert row[column] == 0
    # A group without cells has undefined indices that divide by it.
    empty = overlap_pairs(np.array([0]), np.array([0]), np.array([3]), 2).iloc[0]
    assert empty["jaccard"] == 0
    assert np.isnan(empty["overlap"]) and np.isnan(empty["morisita_horn"])
    assert np.isnan(empty["shared_fraction_2"])


def test_identical_groups_overlap_completely() -> None:
    # Two groups with the same clone sizes: every index is 1 and every cell
    # of each group belongs to a clone of the other.
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clones = np.array([0, 0, 0, 5, 0, 0, 0, 5])
    row = overlap_pairs(*_pairs(groups, clones), 2).iloc[0]
    assert row["shared"] == 2
    for column in OVERLAP_INDICES[1:]:
        assert row[column] == pytest.approx(1.0, rel=1e-15)
    assert row["shared_fraction_1"] == row["shared_fraction_2"] == 1.0


def test_top_clones_take_each_groups_largest() -> None:
    groups = np.array([0, 0, 0, 1, 1, 1])
    clones = np.array([0, 1, 2, 1, 3, 4])
    cells = np.array([5, 5, 1, 2, 9, 2])
    # Ties go to the lower code: group 0 takes 0 and 1, group 1 takes 3 and 1.
    assert top_clones(groups, clones, cells, 2, 2).tolist() == [0, 1, 3]
    assert top_clones(groups, clones, cells, 2, 1).tolist() == [0, 3]
    assert top_clones(groups, clones, cells, 2, 10).tolist() == [0, 1, 2, 3, 4]


def test_top_clones_do_not_depend_on_the_order_of_the_pairs() -> None:
    rng = np.random.default_rng(3)
    groups = rng.integers(0, 5, size=300)
    clones = np.minimum(rng.zipf(1.5, size=300), 30) - 1
    pair_groups, pair_clones, pair_cells = _pairs(groups, clones)
    expected = top_clones(pair_groups, pair_clones, pair_cells, 5, 3)
    shuffled = rng.permutation(len(pair_cells))
    found = top_clones(
        pair_groups[shuffled], pair_clones[shuffled], pair_cells[shuffled], 5, 3
    )
    assert found.tolist() == expected.tolist()
    # Each group's three largest, ties to the lower code, by hand.
    by_hand: set[int] = set()
    for group in range(5):
        sizes = Counter(clones[groups == group].tolist())
        by_hand |= {
            c for c, _ in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        }
    assert found.tolist() == sorted(by_hand)


@pytest.mark.parametrize("seed", range(8))
def test_strata_pairs_match_the_reference_in_the_order_of_groups(seed) -> None:
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(3, 9))
    n_cells = int(rng.integers(30, 300))
    clones = np.minimum(rng.zipf(1.7, size=n_cells), 30) - 1
    groups = rng.integers(0, n_groups, size=n_cells)
    groups[:n_groups] = np.arange(n_groups)
    # Strata interleave, so the pairs of one stratum are not adjacent.
    strata = rng.choice(np.asarray(["TCR", "BCR", "ambiguous"]), size=n_groups)
    table = overlap_pairs(*_pairs(groups, clones), n_groups, strata=strata)
    expected = [
        [first, second]
        for first in range(n_groups)
        for second in range(first + 1, n_groups)
        if strata[first] == strata[second]
    ]
    assert table[["first", "second"]].values.tolist() == expected
    for row in table.itertuples(index=False):
        reference = _reference(groups.tolist(), clones.tolist(), row.first, row.second)
        for column, value in reference.items():
            assert getattr(row, column) == pytest.approx(value, rel=1e-12)
    sizes = Counter(strata.tolist())
    assert overlap_extent(n_groups, strata) == (len(expected), max(sizes.values()))
    assert overlap_extent(n_groups) == (n_groups * (n_groups - 1) // 2, n_groups)


def test_a_stratum_of_one_group_has_no_pairs() -> None:
    groups = np.array([0, 1, 2])
    clones = np.array([0, 0, 1])
    table = overlap_pairs(*_pairs(groups, clones), 3, strata=np.array(["T", "B", "T"]))
    assert table[["first", "second", "shared"]].values.tolist() == [[0, 2, 0]]
    none = overlap_pairs(*_pairs(groups, clones), 3, strata=np.array(["T", "B", "X"]))
    assert none.empty and none.columns.tolist() == table.columns.tolist()
    assert none["first"].dtype == np.int64 and none["jaccard"].dtype == np.float64


def test_group_clone_matrix_holds_each_clones_cells_in_each_group() -> None:
    groups = np.array([0, 0, 0, 2, 2, 1])
    clones = np.array([4, 4, 1, 4, 0, 0])
    matrix = group_clone_matrix(*_pairs(groups, clones), 4)
    # Groups are rows, clone codes up to the largest are columns, and a group
    # without cells keeps an empty row.
    assert matrix.shape == (4, 5) and matrix.dtype == np.float64
    assert matrix.toarray().tolist() == [
        [0, 1, 0, 0, 2],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    empty = group_clone_matrix(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        3,
    )
    assert empty.shape == (3, 0) and empty.nnz == 0


def test_the_matrix_sums_repeated_pairs_and_stores_only_held_clones() -> None:
    # Group 0 holds clone 1 twice, 2 and 3 cells, and clone 2 with no cell.
    matrix = group_clone_matrix(
        np.array([0, 0, 0, 1]), np.array([1, 1, 2, 0]), np.array([2, 3, 0, 4]), 2
    )
    assert matrix.toarray().tolist() == [[0, 5, 0], [4, 0, 0]]
    assert matrix.nnz == 2
    # Columns for every clone of an artifact, beyond the largest code held.
    wide = group_clone_matrix(np.array([1]), np.array([2]), np.array([7]), 2, 6)
    assert wide.shape == (2, 6) and wide[1, 2] == 7.0 and wide.nnz == 1


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (([0], [0], [-1], 1), ValueError, "non-negative, finite counts"),
        (([0], [0], [np.nan], 1), ValueError, "non-negative, finite counts"),
        (([0], [0], [np.inf], 1), ValueError, "non-negative, finite counts"),
        (([0, 1], [0], [1], 2), ValueError, "one-dimensional arrays of one length"),
        (([[0]], [[0]], [[1]], 1), ValueError, "one-dimensional arrays of one"),
        (([1], [0], [1], 1), ValueError, "exceeds matrix dimension"),
        (([0], [-1], [1], 1), ValueError, "negative axis 1 index"),
        (([0], [3], [1], 1, 3), ValueError, "n_clones is 3, below the clone code 3"),
        (([0], [0], [1], 1, 1.5), TypeError, "n_clones must be an integer"),
        (([0], [0], [1], 1, True), TypeError, "n_clones must be an integer"),
    ],
)
def test_the_matrix_refuses_invalid_pairs(arguments, error, message) -> None:
    arrays = [np.asarray(values) for values in arguments[:3]]
    with pytest.raises(error, match=message):
        group_clone_matrix(*arrays, *arguments[3:])


def test_pairs_given_per_cell_or_of_no_cells_count_as_distinct_pairs() -> None:
    rng = np.random.default_rng(11)
    groups = rng.integers(0, 4, size=200)
    clones = np.minimum(rng.zipf(1.6, size=200), 30) - 1
    groups[:4] = np.arange(4)
    distinct = overlap_pairs(*_pairs(groups, clones), 4)
    # One pair of one cell per cell, in the cells' order, and pairs of no
    # cells for clones that no group holds.
    per_cell = overlap_pairs(groups, clones, np.ones(200, dtype=np.int64), 4)
    pd.testing.assert_frame_equal(per_cell, distinct, check_exact=True)
    padded = overlap_pairs(
        np.concatenate([groups, [0, 1, 2]]),
        np.concatenate([clones, [40, 41, 42]]),
        np.concatenate([np.ones(200, dtype=np.int64), [0, 0, 0]]),
        4,
    )
    pd.testing.assert_frame_equal(padded, distinct, check_exact=True)
    for row in distinct.itertuples(index=False):
        expected = _reference(groups.tolist(), clones.tolist(), row.first, row.second)
        assert row.jaccard == pytest.approx(expected["jaccard"], rel=1e-12)


def _tracked_rows(groups, clones, cells, chosen, strata, fallback):
    """Follow each chosen clone through every group, one row at a time."""
    rows = []
    for position, code in enumerate(chosen.tolist()):
        mine = {
            group: count
            for group, clone, count in zip(groups, clones, cells, strict=True)
            if clone == code
        }
        reach = {strata[group] for group in mine}
        if not reach and fallback[position] >= 0:
            reach = {fallback[position]}
        rows.extend(
            (position, group, mine.get(group, 0))
            for group in range(len(strata))
            if strata[group] in reach
        )
    return rows


@pytest.mark.parametrize("seed", range(8))
def test_tracking_follows_each_clone_through_its_strata(seed) -> None:
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(1, 7))
    n_cells = int(rng.integers(5, 200))
    groups = rng.integers(0, n_groups, size=n_cells)
    clones = np.minimum(rng.zipf(1.6, size=n_cells), 25) - 1
    pair_groups, pair_clones, pair_cells = _pairs(groups, clones)
    # The pairs may come in any order.
    shuffled = rng.permutation(len(pair_cells))
    pair_groups, pair_clones, pair_cells = (
        pair_groups[shuffled],
        pair_clones[shuffled],
        pair_cells[shuffled],
    )
    strata = rng.integers(0, 3, size=n_groups)
    # Some chosen clones have no cells, beyond the codes of the pairs too.
    chosen = np.unique(rng.integers(0, 30, size=int(rng.integers(1, 12))))
    fallback = rng.integers(-1, int(strata.max()) + 1, size=len(chosen))
    reach = tracking_reach(pair_groups, pair_clones, chosen, strata, fallback)
    positions, rows, cells = tracking_cells(
        pair_groups, pair_clones, pair_cells, chosen, strata, reach
    )
    expected = _tracked_rows(
        pair_groups.tolist(),
        pair_clones.tolist(),
        pair_cells.tolist(),
        chosen,
        strata.tolist(),
        fallback.tolist(),
    )
    got = list(zip(positions.tolist(), rows.tolist(), cells.tolist(), strict=True))
    assert got == expected
    assert int((reach * np.bincount(strata, minlength=reach.shape[1])).sum()) == len(
        expected
    )


def test_tracking_without_cells_follows_the_fallback_stratum() -> None:
    empty = np.array([], dtype=np.int64)
    chosen = np.array([0, 3])
    strata = np.array([0, 1, 0])
    reach = tracking_reach(empty, empty, chosen, strata, np.array([0, -1]))
    assert reach.tolist() == [[True, False], [False, False]]
    positions, rows, cells = tracking_cells(empty, empty, empty, chosen, strata, reach)
    assert positions.tolist() == [0, 0] and rows.tolist() == [0, 2]
    assert cells.tolist() == [0, 0]
