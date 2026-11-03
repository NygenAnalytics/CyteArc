"""Group names and the cells of each clone in each group, against counts by hand."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.groups import group_key_table, group_names, pair_counts


def _pairs(groups: np.ndarray, clones: np.ndarray):
    """Count each distinct pair of group and clone."""
    counts = Counter(zip(groups.tolist(), clones.tolist(), strict=True))
    keys = sorted(counts)
    return (
        np.asarray([group for group, _ in keys]),
        np.asarray([clone for _, clone in keys]),
        np.asarray([counts[key] for key in keys]),
    )


def test_group_names_join_labels() -> None:
    assert group_names([{"sample": "s1", "cell_type": "B"}, {}]) == [
        "s1, B",
        "all cells",
    ]
    assert group_names([]) == []
    # Equal keys share their name without a clash.
    assert group_names([{"a": "x"}, {"a": "x"}]) == ["x", "x"]


def test_pairs_count_each_clones_cells_in_each_group() -> None:
    rng = np.random.default_rng(4)
    groups = rng.integers(0, 6, size=500)
    clones = np.minimum(rng.zipf(1.5, size=500), 80) - 1
    pair_groups, pair_clones, cells = pair_counts(groups, clones)
    expected = _pairs(groups, clones)
    assert pair_groups.tolist() == expected[0].tolist()
    assert pair_clones.tolist() == expected[1].tolist()
    assert cells.tolist() == expected[2].tolist()
    assert cells.dtype == np.int64
    empty = pair_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert [len(part) for part in empty] == [0, 0, 0]


def test_pairs_of_one_group_and_one_clone() -> None:
    # Group 0 and clone 0 give the key 0, which still starts a pair.
    pair_groups, pair_clones, cells = pair_counts(np.zeros(3), np.zeros(3))
    assert (pair_groups.tolist(), pair_clones.tolist(), cells.tolist()) == (
        [0],
        [0],
        [3],
    )


def test_group_key_table_holds_each_groups_labels_under_their_columns() -> None:
    keys = group_key_table(
        ("sample", "cell_type"),
        [
            {"sample": "s1", "cell_type": "B"},
            {"sample": "s2", "cell_type": "T"},
        ],
    )
    assert keys.index.name == "group"
    assert keys.index.tolist() == ["s1, B", "s2, T"]
    assert keys.to_dict("list") == {"sample": ["s1", "s2"], "cell_type": ["B", "T"]}
    # Without key columns, the one group of every cell has no labels.
    alone = group_key_table((), [{}])
    assert (alone.index.tolist(), alone.columns.tolist()) == (["all cells"], [])
    assert group_key_table(("sample",), []).columns.tolist() == ["sample"]
    assert group_key_table(
        ("cell_type",), [{"cell_type": "B"}], index="fate"
    ).index.name == ("fate")
    with pytest.raises(ValueError, match="both named"):
        group_key_table(("a", "b"), [{"a": "x, y", "b": "z"}, {"a": "x", "b": "y, z"}])


def test_key_columns_may_take_the_names_of_the_index_and_of_a_tables_columns() -> None:
    # A key column named "group" sits beside the index of that name, and the
    # labels of columns named like a summary's own columns stay the user's.
    keys = group_key_table(
        ("group", "n_cells"),
        [{"group": "g1", "n_cells": "x"}, {"group": "g2", "n_cells": "y"}],
    )
    assert keys.index.tolist() == ["g1, x", "g2, y"]
    assert keys.columns.tolist() == ["group", "n_cells"]
    assert keys["group"].tolist() == ["g1", "g2"]
    table = pd.DataFrame({"group": ["g2, y", "g1, x", "g2, y"], "n_cells": [3, 1, 2]})
    joined = table.join(keys, on="group", rsuffix="_label")
    assert joined.to_dict("list") == {
        "group": ["g2, y", "g1, x", "g2, y"],
        "n_cells": [3, 1, 2],
        "group_label": ["g2", "g1", "g2"],
        "n_cells_label": ["y", "x", "y"],
    }
