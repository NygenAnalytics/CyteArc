"""Group names and the cells of each clone in each group, against counts by hand."""

from collections import Counter

import numpy as np
import pytest

from cytearc.clones.groups import group_frame, group_names, pair_counts


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


def test_group_frame_names_each_group() -> None:
    frame = group_frame(
        ("sample", "cell_type"),
        [
            {"sample": "s1", "cell_type": "B"},
            {"sample": "s2", "cell_type": "T"},
        ],
    )
    assert frame.to_dict("list") == {
        "group": ["s1, B", "s2, T"],
        "sample": ["s1", "s2"],
        "cell_type": ["B", "T"],
    }
    assert group_frame((), [{}]).to_dict("list") == {"group": ["all cells"]}
    assert group_frame(("sample",), []).columns.tolist() == ["group", "sample"]
    with pytest.raises(ValueError, match="both named"):
        group_frame(("a", "b"), [{"a": "x, y", "b": "z"}, {"a": "x", "b": "y, z"}])
