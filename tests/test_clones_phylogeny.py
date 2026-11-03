"""Node times, depths, cuts and clone labels of phylogenies, derived by hand.

Each tree is written out node by node in preorder; the expected times,
depths and clades come from reading the tree, never from the functions under
test.
"""

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.phylogeny import (
    clade_roots,
    clone_labels,
    collapsed_depths,
    leaf_time_ranges,
    node_times,
    phylogeny_cut,
    phylogeny_frames,
    tree_codes,
)

NAN = np.nan
# A tree in preorder:
#   0 root (stem 0.5)
#   +- 1 X (1.0)          one child
#   |  +- 2 Y (2.0)       polytomy
#   |     +- 3 a (1.0)
#   |     +- 4 b (1.0)
#   |     +- 5 c (2.0)
#   +- 6 d (3.0)
PARENT = np.asarray([-1, 0, 1, 2, 2, 2, 0])
LENGTH = np.asarray([0.5, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0])
OFFSETS = np.asarray([0, 7])


def test_node_times_add_lengths_from_the_root_and_its_stem() -> None:
    times = node_times(PARENT, LENGTH, np.full(7, NAN), OFFSETS)
    assert times.tolist() == [0.5, 1.5, 3.5, 4.5, 4.5, 5.5, 3.5]


def test_a_nodes_own_time_wins_over_its_branch_length() -> None:
    given = np.full(7, NAN)
    given[2] = 10.0
    times = node_times(PARENT, LENGTH, given, OFFSETS)
    # Y's own time, then its children's lengths from there.
    assert times.tolist() == [0.5, 1.5, 10.0, 11.0, 11.0, 12.0, 3.5]


def test_a_root_without_time_or_stem_starts_at_zero() -> None:
    length = LENGTH.copy()
    length[0] = NAN
    assert node_times(PARENT, length, np.full(7, NAN), OFFSETS)[:2].tolist() == [
        0.0,
        1.0,
    ]


def test_a_branch_without_a_length_leaves_its_subtree_without_times() -> None:
    length = LENGTH.copy()
    length[2] = NAN
    times = node_times(PARENT, length, np.full(7, NAN), OFFSETS)
    assert times[:2].tolist() == [0.5, 1.5] and times[6] == 3.5
    assert np.isnan(times[2:6]).all()


def test_trees_without_any_length_or_time_have_no_times() -> None:
    # Two trees: the first without lengths, the second with one.
    parent = np.asarray([-1, 0, 0, -1, 3])
    length = np.asarray([NAN, NAN, NAN, NAN, 2.0])
    times = node_times(parent, length, np.full(5, NAN), np.asarray([0, 3, 5]))
    assert np.isnan(times[:3]).all()
    assert times[3:].tolist() == [0.0, 2.0]


def test_depths_count_branchings_with_single_child_nodes_merged() -> None:
    # root branches (to X and d), X has one child, Y branches.
    assert collapsed_depths(PARENT).tolist() == [0, 1, 1, 2, 2, 2, 1]
    # 0 has one child, 1, which merges with it; 1 branches to 2 and 5, and 2
    # to 3 and 4.
    chain = np.asarray([-1, 0, 1, 2, 2, 1])
    assert collapsed_depths(chain).tolist() == [0, 0, 1, 2, 2, 1]
    assert collapsed_depths(np.asarray([-1])).tolist() == [0]


def test_clades_start_at_the_first_node_at_or_past_the_cut() -> None:
    times = node_times(PARENT, LENGTH, np.full(7, NAN), OFFSETS)
    # At 1.5, X is the first past the cut below the root; d reaches it too.
    assert clade_roots(PARENT, times >= 1.5).tolist() == [-1, 1, 1, 1, 1, 1, 6]
    # Just past X, the polytomy Y holds a, b and c.
    assert clade_roots(PARENT, times >= 1.6).tolist() == [-1, -1, 2, 2, 2, 2, 6]
    # Past Y each child of the polytomy is a clade, and d ends before 4.
    assert clade_roots(PARENT, times >= 4.0).tolist() == [-1, -1, -1, 3, 4, 5, -1]
    # The root's own time cuts the tree as one clade.
    assert clade_roots(PARENT, times >= 0.5).tolist() == [0] * 7
    assert clade_roots(PARENT, times >= 99).tolist() == [-1] * 7


def test_a_path_that_crosses_the_cut_twice_belongs_to_the_first_crossing() -> None:
    # root 0 -> a 5 -> b 4 -> c 6, with a negative length below a.
    parent = np.asarray([-1, 0, 1, 2])
    times = node_times(
        parent, np.asarray([0.0, 5.0, -1.0, 2.0]), np.full(4, NAN), np.asarray([0, 4])
    )
    assert times.tolist() == [0.0, 5.0, 4.0, 6.0]
    assert clade_roots(parent, times >= 4.5).tolist() == [-1, 1, 1, 1]


def test_nodes_without_a_time_are_never_past_a_cut() -> None:
    parent = np.asarray([-1, 0, 0])
    assert clade_roots(parent, np.asarray([NAN, NAN, 2.0]) >= 1.0).tolist() == [
        -1,
        -1,
        2,
    ]


def test_clones_are_named_by_their_roots_or_their_positions() -> None:
    offsets = np.asarray([0, 3, 6])
    named = clone_labels(
        np.asarray([1, 4]),
        tree_offsets=offsets,
        tree_names=np.asarray(["first", "second"], dtype=object),
        unique_names=np.asarray([True, False]),
        names=["X", "dup"],
    )
    # The first tree names its nodes; the second does not, so its clone is
    # named by its root's position in that tree.
    assert named.tolist() == ["first/X", "second/node1"]
    alone = clone_labels(
        np.asarray([0, 2]),
        tree_offsets=np.asarray([0, 3]),
        tree_names=np.asarray(["only"], dtype=object),
        unique_names=np.asarray([False]),
        names=["", ""],
    )
    assert alone.tolist() == ["node0", "node2"]


def test_leaf_time_ranges_show_how_far_trees_are_from_ultrametric() -> None:
    times = node_times(PARENT, LENGTH, np.full(7, NAN), OFFSETS)
    leaf = np.asarray([False, False, False, True, True, True, True])
    earliest, latest = leaf_time_ranges(times, leaf, OFFSETS)
    assert (earliest.tolist(), latest.tolist()) == ([3.5], [5.5])
    unknown = times.copy()
    unknown[3] = NAN
    earliest, latest = leaf_time_ranges(unknown, leaf, OFFSETS)
    assert np.isnan(earliest).all() and np.isnan(latest).all()


@pytest.mark.parametrize(
    ("time", "depth", "error", "message"),
    [
        (None, None, ValueError, "Pass exactly one of time"),
        (1.0, 2, ValueError, "Pass exactly one of time"),
        (True, None, TypeError, "time must be a real number"),
        ("1", None, TypeError, "time must be a real number"),
        (float("nan"), None, ValueError, "time must be finite"),
        (float("inf"), None, ValueError, "time must be finite"),
        (None, 1.5, TypeError, "depth must be an integer"),
        (None, True, TypeError, "depth must be an integer"),
        (None, -1, ValueError, "depth must be at least 0"),
    ],
)
def test_a_cut_takes_a_finite_time_or_a_depth(time, depth, error, message) -> None:
    with pytest.raises(error, match=message):
        phylogeny_cut(time, depth)


def test_numpy_numbers_cut_like_python_numbers() -> None:
    assert phylogeny_cut(np.float32(1.5), None).time == 1.5
    assert phylogeny_cut(None, np.int64(3)).depth == 3
    assert phylogeny_cut(np.int64(2), None).by == "time"
    assert phylogeny_cut(None, 0).by == "depth"


def test_tree_codes_follow_the_offsets() -> None:
    assert tree_codes(np.asarray([0, 2, 2, 5])).tolist() == [0, 0, 2, 2, 2]


def test_the_tables_of_a_forest_hold_each_node_and_tree() -> None:
    # Two trees: r(a, b) and s(t(c)).
    nodes, trees = phylogeny_frames(
        parent=np.asarray([-1, 0, 0, -1, 3, 4]),
        times=np.asarray([0.0, 1.0, 2.0, 0.5, 1.0, 3.0]),
        names=np.asarray(["r", "a", "b", "s", "t", "c"], dtype=object),
        cell_ids=np.asarray([None, "cell_a", None, None, None, "cell_c"], dtype=object),
        tree_offsets=np.asarray([0, 3, 6]),
        tree_names=np.asarray(["first", "second"], dtype=object),
    )
    assert nodes["tree"].tolist() == ["first"] * 3 + ["second"] * 3
    assert nodes["node"].tolist() == [0, 1, 2, 0, 1, 2]
    assert nodes["parent"].tolist() == [-1, 0, 0, -1, 0, 1]
    assert nodes["depth"].tolist() == [0, 1, 1, 0, 0, 0]
    assert nodes["is_leaf"].tolist() == [False, True, True, False, False, True]
    assert nodes["ids"].tolist() == [pd.NA, "cell_a", pd.NA, pd.NA, pd.NA, "cell_c"]
    assert trees.to_dict("records") == [
        {
            "tree": "first",
            "n_nodes": 3,
            "n_leaves": 2,
            "n_cells": 1,
            "root_time": 0.0,
            "earliest_leaf_time": 1.0,
            "latest_leaf_time": 2.0,
        },
        {
            "tree": "second",
            "n_nodes": 3,
            "n_leaves": 1,
            "n_cells": 1,
            "root_time": 0.5,
            "earliest_leaf_time": 3.0,
            "latest_leaf_time": 3.0,
        },
    ]
