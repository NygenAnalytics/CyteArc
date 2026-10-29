from collections import Counter
from math import sqrt

import numpy as np
import pytest

from cytearc.repertoire.overlap import (
    OVERLAP_INDICES,
    group_names,
    overlap_pairs,
    top_clonotypes,
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
