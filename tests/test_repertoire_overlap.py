from collections import Counter
from math import sqrt

import numpy as np
import pytest

from cytearc.repertoire.groups import group_names
from cytearc.repertoire.overlap import (
    OVERLAP_INDICES,
    group_frame,
    overlap_extent,
    overlap_pairs,
    pair_counts,
    top_clonotypes,
    tracking_cells,
    tracking_reach,
)


def _pairs(groups: np.ndarray, clonotypes: np.ndarray):
    """Count each distinct pair of group and clonotype."""
    counts = Counter(zip(groups.tolist(), clonotypes.tolist(), strict=True))
    keys = sorted(counts)
    return (
        np.asarray([group for group, _ in keys]),
        np.asarray([clonotype for _, clonotype in keys]),
        np.asarray([counts[key] for key in keys]),
    )


def _reference(groups, clonotypes, first, second):
    """The indices of one pair, from dictionaries of clone sizes."""
    x = Counter(c for g, c in zip(groups, clonotypes, strict=True) if g == first)
    y = Counter(c for g, c in zip(groups, clonotypes, strict=True) if g == second)
    shared = set(x) & set(y)
    s1, s2, both = len(x), len(y), len(shared)
    n1, n2 = sum(x.values()), sum(y.values())
    dot = sum(x[c] * y[c] for c in shared)
    q1, q2 = sum(v * v for v in x.values()), sum(v * v for v in y.values())
    return {
        "shared": both,
        "jaccard": both / (s1 + s2 - both),
        "sorensen": 2 * both / (s1 + s2),
        "overlap": both / min(s1, s2),
        "cosine": both / sqrt(s1 * s2),
        "morisita_horn": 2 * dot / ((q1 / n1**2 + q2 / n2**2) * n1 * n2),
        "abundance_cosine": dot / sqrt(q1 * q2),
        "shared_fraction_1": sum(x[c] for c in shared) / n1,
        "shared_fraction_2": sum(y[c] for c in shared) / n2,
    }


@pytest.mark.parametrize("seed", range(10))
def test_pairs_match_a_set_by_set_reference(seed) -> None:
    rng = np.random.default_rng(seed)
    n_cells, n_groups = int(rng.integers(20, 400)), int(rng.integers(2, 6))
    clonotypes = np.minimum(rng.zipf(1.7, size=n_cells), 40) - 1
    groups = rng.integers(0, n_groups, size=n_cells)
    groups[:n_groups] = np.arange(n_groups)  # every group holds a cell
    table = overlap_pairs(*_pairs(groups, clonotypes), n_groups)
    assert table.columns.tolist() == [
        "first",
        "second",
        *OVERLAP_INDICES,
        "shared_fraction_1",
        "shared_fraction_2",
    ]
    assert len(table) == n_groups * (n_groups - 1) // 2
    for row in table.itertuples(index=False):
        expected = _reference(
            groups.tolist(), clonotypes.tolist(), row.first, row.second
        )
        for column, value in expected.items():
            assert getattr(row, column) == pytest.approx(value, rel=1e-12)


def test_strata_keep_pairs_within_one_stratum() -> None:
    groups = np.array([0, 0, 1, 1, 2, 2, 3])
    clonotypes = np.array([0, 1, 0, 2, 3, 3, 3])
    table = overlap_pairs(
        *_pairs(groups, clonotypes), 4, strata=np.array(["T", "T", "B", "B"])
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


def test_top_clonotypes_take_each_groups_largest() -> None:
    groups = np.array([0, 0, 0, 1, 1, 1])
    clonotypes = np.array([0, 1, 2, 1, 3, 4])
    cells = np.array([5, 5, 1, 2, 9, 2])
    # Ties go to the lower code: group 0 takes 0 and 1, group 1 takes 3 and 1.
    assert top_clonotypes(groups, clonotypes, cells, 2, 2).tolist() == [0, 1, 3]
    assert top_clonotypes(groups, clonotypes, cells, 2, 1).tolist() == [0, 3]
    assert top_clonotypes(groups, clonotypes, cells, 2, 10).tolist() == [0, 1, 2, 3, 4]


def test_group_names_join_labels() -> None:
    assert group_names([{"sample": "s1", "receptor_type": "TCR"}, {}]) == [
        "s1, TCR",
        "all cells",
    ]


def test_pairs_count_each_clonotypes_cells_in_each_group() -> None:
    rng = np.random.default_rng(4)
    groups = rng.integers(0, 6, size=500)
    clonotypes = np.minimum(rng.zipf(1.5, size=500), 80) - 1
    pair_groups, pair_clonotypes, cells = pair_counts(groups, clonotypes)
    expected = _pairs(groups, clonotypes)
    assert pair_groups.tolist() == expected[0].tolist()
    assert pair_clonotypes.tolist() == expected[1].tolist()
    assert cells.tolist() == expected[2].tolist()
    assert cells.dtype == np.int64
    empty = pair_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert [len(part) for part in empty] == [0, 0, 0]


@pytest.mark.parametrize("seed", range(8))
def test_strata_pairs_match_the_reference_in_the_order_of_groups(seed) -> None:
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(3, 9))
    n_cells = int(rng.integers(30, 300))
    clonotypes = np.minimum(rng.zipf(1.7, size=n_cells), 30) - 1
    groups = rng.integers(0, n_groups, size=n_cells)
    groups[:n_groups] = np.arange(n_groups)
    # Strata interleave, so the pairs of one stratum are not adjacent.
    strata = rng.choice(np.asarray(["TCR", "BCR", "ambiguous"]), size=n_groups)
    table = overlap_pairs(*_pairs(groups, clonotypes), n_groups, strata=strata)
    expected = [
        [first, second]
        for first in range(n_groups)
        for second in range(first + 1, n_groups)
        if strata[first] == strata[second]
    ]
    assert table[["first", "second"]].values.tolist() == expected
    for row in table.itertuples(index=False):
        reference = _reference(
            groups.tolist(), clonotypes.tolist(), row.first, row.second
        )
        for column, value in reference.items():
            assert getattr(row, column) == pytest.approx(value, rel=1e-12)
    sizes = Counter(strata.tolist())
    assert overlap_extent(n_groups, strata) == (len(expected), max(sizes.values()))
    assert overlap_extent(n_groups) == (n_groups * (n_groups - 1) // 2, n_groups)


def test_a_stratum_of_one_group_has_no_pairs() -> None:
    groups = np.array([0, 1, 2])
    clonotypes = np.array([0, 0, 1])
    table = overlap_pairs(
        *_pairs(groups, clonotypes), 3, strata=np.array(["T", "B", "T"])
    )
    assert table[["first", "second", "shared"]].values.tolist() == [[0, 2, 0]]
    none = overlap_pairs(
        *_pairs(groups, clonotypes), 3, strata=np.array(["T", "B", "X"])
    )
    assert none.empty and none.columns.tolist() == table.columns.tolist()
    assert none["first"].dtype == np.int64 and none["jaccard"].dtype == np.float64


def _tracked_rows(groups, clonotypes, cells, chosen, strata, fallback):
    """Follow each chosen clonotype through every group, one row at a time."""
    rows = []
    for position, code in enumerate(chosen.tolist()):
        mine = {
            group: count
            for group, clonotype, count in zip(groups, clonotypes, cells, strict=True)
            if clonotype == code
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
def test_tracking_follows_each_clonotype_through_its_strata(seed) -> None:
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(1, 7))
    n_cells = int(rng.integers(5, 200))
    groups = rng.integers(0, n_groups, size=n_cells)
    clonotypes = np.minimum(rng.zipf(1.6, size=n_cells), 25) - 1
    pair_groups, pair_clonotypes, pair_cells = _pairs(groups, clonotypes)
    # The pairs may come in any order.
    shuffled = rng.permutation(len(pair_cells))
    pair_groups, pair_clonotypes, pair_cells = (
        pair_groups[shuffled],
        pair_clonotypes[shuffled],
        pair_cells[shuffled],
    )
    strata = rng.integers(0, 3, size=n_groups)
    # Some chosen clonotypes have no cells, beyond the codes of the pairs too.
    chosen = np.unique(rng.integers(0, 30, size=int(rng.integers(1, 12))))
    fallback = rng.integers(-1, int(strata.max()) + 1, size=len(chosen))
    reach = tracking_reach(pair_groups, pair_clonotypes, chosen, strata, fallback)
    positions, rows, cells = tracking_cells(
        pair_groups, pair_clonotypes, pair_cells, chosen, strata, reach
    )
    expected = _tracked_rows(
        pair_groups.tolist(),
        pair_clonotypes.tolist(),
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


def test_group_frame_names_each_group() -> None:
    frame = group_frame(
        ("sample", "receptor_type"),
        [
            {"sample": "s1", "receptor_type": "TCR"},
            {"sample": "s2", "receptor_type": "BCR"},
        ],
    )
    assert frame.to_dict("list") == {
        "group": ["s1, TCR", "s2, BCR"],
        "sample": ["s1", "s2"],
        "receptor_type": ["TCR", "BCR"],
    }
    assert group_frame((), [{}]).to_dict("list") == {"group": ["all cells"]}
    with pytest.raises(ValueError, match="both named"):
        group_frame(("a", "b"), [{"a": "x, y", "b": "z"}, {"a": "x", "b": "y, z"}])
