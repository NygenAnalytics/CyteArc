"""Cut phylogenies collapsed to a cap of wedges, as a figure of their clones draws them.

Hand-made trees check the cells and clones of each wedge, the order in which
clades open and the cap against counts worked out by hand. A recursive
reference written here, from the rules alone, checks every column on 200
random forests, and a caterpillar 30,000 levels deep, which recursion could
not walk, collapses like a short one.
"""

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.phylogeny import PhylogenyCut, collapse_phylogeny
from tests.memory_helpers import budgeted

AT_TIME = PhylogenyCut(time=0.5, depth=None)

# The columns of a collapsed phylogeny, in order.
COLUMNS = [
    "tree",
    "node",
    "parent",
    "wedge",
    "n_cells",
    "n_clones",
    "clone",
    "n_clades",
    "n_trees",
    "x",
    "start",
    "end",
    "y",
    "low",
    "high",
]


def _forest(
    trees: list[list[tuple[int, float, bool]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arrays of a forest of trees, each given as (parent in its tree, time, names a cell) per node in preorder.

    Cells are numbered in the order of their leaves.
    """
    parent, times, cells, offsets = [], [], [], [0]
    n_cells = 0
    for nodes in trees:
        start = offsets[-1]
        for above, time, has_cell in nodes:
            parent.append(above + start if above >= 0 else -1)
            times.append(time)
            cells.append(n_cells if has_cell else -1)
            n_cells += has_cell
        offsets.append(start + len(nodes))
    return (
        np.asarray(parent, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
        np.asarray(cells, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
    )


def _wedges(frame: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    """Each wedge's node, cells, clones and clades, in drawing order."""
    wedges = frame[frame["wedge"]]
    return list(
        zip(
            wedges["node"].tolist(),
            wedges["n_cells"].tolist(),
            wedges["n_clones"].tolist(),
            wedges["n_clades"].tolist(),
            strict=True,
        )
    )


# A polytomy: a root at time 0 whose children at time 1 are clade A of three
# leaves, B of two, the leaf c and D of four, every leaf at time 3. Cut at
# time 0.5, each child is a clone: D the largest, then A, B and c.
POLYTOMY = [
    (-1, 0.0, False),  # 0 root
    (0, 1.0, False),  # 1 A
    (1, 3.0, True),
    (1, 3.0, True),
    (1, 3.0, True),
    (0, 1.0, False),  # 5 B
    (5, 3.0, True),
    (5, 3.0, True),
    (0, 3.0, True),  # 8 c
    (0, 1.0, False),  # 9 D
    (9, 3.0, True),
    (9, 3.0, True),
    (9, 3.0, True),
    (9, 3.0, True),
]
# Each cell's clone: A's three cells are clone 1, B's clone 2, c's clone 3 and
# D's clone 0.
POLYTOMY_CODES = np.asarray([1, 1, 1, 2, 2, 3, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("max_leaves", "wedges"),
    [
        # The root alone: ten cells of four clones.
        (1, [(0, 10, 4, 1)]),
        # The root opens: D, its largest child, alone, and A, B and c as one
        # wedge of six cells and three clones.
        (2, [(9, 4, 1, 1), (-1, 6, 3, 3)]),
        # The group, of more cells than D, opens: A alone, B and c together.
        (3, [(9, 4, 1, 1), (1, 3, 1, 1), (-1, 3, 2, 2)]),
        # The last group opens into its two clades.
        (4, [(9, 4, 1, 1), (1, 3, 1, 1), (5, 2, 1, 1), (8, 1, 1, 1)]),
        # Each wedge is a clone of the cut, which never opens: four wedges.
        (50, [(9, 4, 1, 1), (1, 3, 1, 1), (5, 2, 1, 1), (8, 1, 1, 1)]),
    ],
)
def test_a_polytomy_opens_one_child_at_a_time_largest_first(max_leaves, wedges) -> None:
    parent, times, cells, offsets = _forest([POLYTOMY])
    frame = collapse_phylogeny(
        parent,
        times,
        cells,
        POLYTOMY_CODES,
        offsets,
        cut=AT_TIME,
        max_leaves=max_leaves,
    )
    assert list(frame.columns) == COLUMNS
    assert _wedges(frame) == wedges
    # Exactly the cap, or every clade when the tree holds fewer.
    assert frame["wedge"].sum() == min(max_leaves, 4)
    # Each wedge is as tall as its cells, stacked from the top.
    shown = frame[frame["wedge"]]
    assert shown["low"].tolist() == [0.0, *np.cumsum(shown["n_cells"])[:-1].tolist()]
    assert (shown["high"] - shown["low"]).tolist() == shown["n_cells"].tolist()


def test_the_wedges_of_a_polytomy_hold_their_clones_and_extents() -> None:
    parent, times, cells, offsets = _forest([POLYTOMY])
    frame = collapse_phylogeny(
        parent, times, cells, POLYTOMY_CODES, offsets, cut=AT_TIME, max_leaves=3
    )
    root, d, a, rest = frame.to_dict("records")
    assert (root["wedge"], root["parent"], root["node"]) == (False, -1, 0)
    # The root opened stands between its first and last children's middles.
    assert root["y"] == (d["y"] + rest["y"]) / 2 == (2.0 + 8.5) / 2
    assert (root["low"], root["high"]) == (0.0, 10.0)
    assert (d["clone"], a["clone"], rest["clone"]) == (0, 1, -1)
    assert (d["x"], d["start"], d["end"]) == (1.0, 0.0, 3.0)
    # The group of B and c runs from the earlier of them, B at time 1.
    assert (rest["x"], rest["start"], rest["end"]) == (1.0, 0.0, 3.0)
    assert (rest["tree"], rest["node"], rest["parent"], rest["n_trees"]) == (
        0,
        -1,
        0,
        1,
    )


# A caterpillar: n0 at time 0 holds the leaf e and n1 at time 1, which holds
# the leaf d and n2 at time 2, which holds c and n3 at time 3, of a and b; every
# leaf at time 4. Cut at time 1.5, the clones are n2's three cells, d and e.
CATERPILLAR = [
    (-1, 0.0, False),  # 0 n0
    (0, 1.0, False),  # 1 n1
    (1, 2.0, False),  # 2 n2
    (2, 3.0, False),  # 3 n3
    (3, 4.0, True),  # 4 a
    (3, 4.0, True),  # 5 b
    (2, 4.0, True),  # 6 c
    (1, 4.0, True),  # 7 d
    (0, 4.0, True),  # 8 e
]
CATERPILLAR_CODES = np.asarray([0, 0, 0, 1, 2])


@pytest.mark.parametrize(
    ("max_leaves", "wedges"),
    [
        (1, [(0, 5, 3, 1)]),
        # n0 opens into n1, of four cells and two clones, and e.
        (2, [(1, 4, 2, 1), (8, 1, 1, 1)]),
        # n1 opens into the clone n2 and d; nothing else opens.
        (3, [(2, 3, 1, 1), (7, 1, 1, 1), (8, 1, 1, 1)]),
        (10, [(2, 3, 1, 1), (7, 1, 1, 1), (8, 1, 1, 1)]),
    ],
)
def test_a_caterpillar_opens_down_its_spine(max_leaves, wedges) -> None:
    parent, times, cells, offsets = _forest([CATERPILLAR])
    frame = collapse_phylogeny(
        parent,
        times,
        cells,
        CATERPILLAR_CODES,
        offsets,
        cut=PhylogenyCut(time=1.5, depth=None),
        max_leaves=max_leaves,
    )
    assert _wedges(frame) == wedges
    # A clone of the cut starts at or past it.
    clones = frame[frame["wedge"] & (frame["n_clones"] == 1)]
    assert (clones["x"] >= 1.5).all()


# PEtracer's trees: a root with one child, s at time 0.5, which splits into A of
# two leaves and B of three at time 1.
ONE_CHILD_ROOT = [
    (-1, 0.0, False),  # 0 root
    (0, 0.5, False),  # 1 s
    (1, 1.0, False),  # 2 A
    (2, 2.0, True),
    (2, 2.0, True),
    (1, 1.0, False),  # 5 B
    (5, 2.0, True),
    (5, 2.0, True),
    (5, 2.0, True),
]


def test_a_root_with_one_child_is_drawn_from_its_first_branching() -> None:
    parent, times, cells, offsets = _forest([ONE_CHILD_ROOT])
    # Depth 1 cuts A and B: the root merges with its one child at depth 0.
    frame = collapse_phylogeny(
        parent,
        None,
        cells,
        np.asarray([1, 1, 0, 0, 0]),
        offsets,
        cut=PhylogenyCut(time=None, depth=1),
        max_leaves=2,
    )
    top = frame.iloc[0]
    # The tree is drawn from s, its first branching, with the stem from the
    # root, both at depth 0 on an axis of branchings.
    assert (top["node"], top["x"], top["start"], top["wedge"]) == (1, 0.0, 0.0, False)
    assert _wedges(frame) == [(5, 3, 1, 1), (2, 2, 1, 1)]
    assert frame["x"].tolist() == [0.0, 1.0, 1.0]
    # Each clade's latest leaf is two branchings deep.
    assert frame["end"].tolist() == [2.0, 2.0, 2.0]
    # At a time, the stem runs from the root at 0 to s at 0.5.
    timed = collapse_phylogeny(
        parent,
        times,
        cells,
        np.asarray([1, 1, 0, 0, 0]),
        offsets,
        cut=PhylogenyCut(time=0.8, depth=None),
        max_leaves=1,
    )
    assert timed[["node", "x", "start", "wedge"]].values.tolist() == [
        [1, 0.5, 0.0, True]
    ]
    # A cut at depth 0 makes the tree one clone, drawn from its root.
    whole = collapse_phylogeny(
        parent,
        None,
        cells,
        np.zeros(5, dtype=np.int64),
        offsets,
        cut=PhylogenyCut(time=None, depth=0),
        max_leaves=10,
    )
    assert whole[["node", "n_clones", "clone", "wedge"]].values.tolist() == [
        [0, 1, 0, True]
    ]


# Three trees: T1 of five cells, T2 of two and T3 of three, each a root over
# its leaves, cut at depth 0, so each tree is one clone.
FOREST = [
    [(-1, 0.0, False), *[(0, 1.0, True)] * 5],
    [(-1, 0.0, False), *[(0, 1.0, True)] * 2],
    [(-1, 0.0, False), *[(0, 1.0, True)] * 3],
]
FOREST_CODES = np.asarray([0] * 5 + [2] * 2 + [1] * 3)


@pytest.mark.parametrize(
    ("max_leaves", "rows"),
    [
        # The whole forest as one wedge of three trees.
        (1, [(-1, -1, 10, 3, 3)]),
        # The largest tree alone, the two others together.
        (2, [(0, 0, 5, 1, 1), (-1, -1, 5, 2, 2)]),
        # Each tree alone, the most cells first; each is one clone, so
        # nothing opens.
        (3, [(0, 0, 5, 1, 1), (2, 0, 3, 1, 1), (1, 0, 2, 1, 1)]),
        (9, [(0, 0, 5, 1, 1), (2, 0, 3, 1, 1), (1, 0, 2, 1, 1)]),
    ],
)
def test_a_forest_of_more_trees_than_the_cap_groups_its_smallest(
    max_leaves, rows
) -> None:
    parent, times, cells, offsets = _forest(FOREST)
    frame = collapse_phylogeny(
        parent,
        times,
        cells,
        FOREST_CODES,
        offsets,
        cut=PhylogenyCut(time=None, depth=0),
        max_leaves=max_leaves,
    )
    assert (
        list(
            zip(
                frame["tree"].tolist(),
                frame["node"].tolist(),
                frame["n_cells"].tolist(),
                frame["n_clones"].tolist(),
                frame["n_trees"].tolist(),
                strict=True,
            )
        )
        == rows
    )
    assert frame["wedge"].all() and (frame["parent"] == -1).all()


def test_the_clade_of_the_most_cells_opens_first() -> None:
    # Below the root, X of six cells, its clades X1 and X2 of three each, and
    # Y of four, its clades Y1 and Y2 of two each, all above the cut at 0.5
    # until their clades at time 1.
    tree = [
        (-1, 0.0, False),  # 0 root
        (0, 0.2, False),  # 1 Y
        (1, 1.0, False),  # 2 Y1
        (2, 2.0, True),
        (2, 2.0, True),
        (1, 1.0, False),  # 5 Y2
        (5, 2.0, True),
        (5, 2.0, True),
        (0, 0.2, False),  # 8 X
        (8, 1.0, False),  # 9 X1
        *[(9, 2.0, True)] * 3,
        (8, 1.0, False),  # 13 X2
        *[(13, 2.0, True)] * 3,
    ]
    codes = np.asarray([2, 2, 3, 3, 0, 0, 0, 1, 1, 1])
    parent, times, cells, offsets = _forest([tree])

    def opened(max_leaves: int) -> list[int]:
        frame = collapse_phylogeny(
            parent, times, cells, codes, offsets, cut=AT_TIME, max_leaves=max_leaves
        )
        return frame.loc[~frame["wedge"], "node"].tolist()

    assert opened(1) == []
    assert opened(2) == [0]
    # X, of six cells, opens before Y, of four, though Y comes first.
    assert opened(3) == [0, 8]
    assert opened(4) == [0, 8, 1]
    frame = collapse_phylogeny(
        parent, times, cells, codes, offsets, cut=AT_TIME, max_leaves=4
    )
    # X is drawn first, the larger child, with its clades in its slot.
    assert frame["node"].tolist() == [0, 8, 9, 13, 1, 2, 5]
    assert frame["parent"].tolist() == [-1, 0, 1, 1, 0, 4, 4]


def test_a_group_ties_by_its_largest_clade_not_by_the_node_opened() -> None:
    # Below the root, X at time 0.1 holds X1, above the cut with the leaves
    # X1a and X1b, and the leaves X2 and X3; Y at time 0.1 holds two leaves.
    # Every leaf ends at time 2, past the cut at 0.5.
    tree = [
        (-1, 0.0, False),  # 0 root
        (0, 0.1, False),  # 1 X
        (1, 0.2, False),  # 2 X1
        (2, 2.0, True),  # 3 X1a
        (2, 2.0, True),  # 4 X1b
        (1, 2.0, True),  # 5 X2
        (1, 2.0, True),  # 6 X3
        (0, 0.1, False),  # 7 Y
        (7, 2.0, True),
        (7, 2.0, True),
    ]
    parent, times, cells, offsets = _forest([tree])
    codes = np.arange(6)
    # At three wedges X1 is drawn alone, X2 and X3 as one group, and Y: each
    # of two cells. The fourth opens X1, which ties the group and Y and comes
    # first in the tree; the group ties by X2, its largest clade, which comes
    # after X1, not by X, which comes before it.
    frame = collapse_phylogeny(
        parent, times, cells, codes, offsets, cut=AT_TIME, max_leaves=4
    )
    assert frame.loc[~frame["wedge"], "node"].tolist() == [0, 1, 2]
    assert _wedges(frame) == [(3, 1, 1, 1), (4, 1, 1, 1), (-1, 2, 2, 2), (7, 2, 2, 1)]


def test_a_cap_far_above_the_cells_collapses_within_a_small_budget() -> None:
    parent, times, cells, offsets = _forest([POLYTOMY])
    # A cap far above the ten cells sizes the opening by the cells: a
    # billion wedges would need 178 GB, and a megabyte draws every wedge
    # that a cap of 50 draws.
    found = collapse_phylogeny(
        parent,
        times,
        cells,
        POLYTOMY_CODES,
        offsets,
        cut=AT_TIME,
        max_leaves=10**9,
        ledger=budgeted(10**6),
    )
    expected = collapse_phylogeny(
        parent, times, cells, POLYTOMY_CODES, offsets, cut=AT_TIME, max_leaves=50
    )
    pd.testing.assert_frame_equal(found, expected)


def test_the_last_clade_of_a_group_opens_in_its_turn() -> None:
    # Below the root, the clade A of five cells and B of three at time 1, and
    # X at time 0.2, above the cut, over the clades X1 and X2 of one cell.
    tree = [
        (-1, 0.0, False),  # 0 root
        (0, 1.0, False),  # 1 A
        *[(1, 2.0, True)] * 5,
        (0, 1.0, False),  # 7 B
        *[(7, 2.0, True)] * 3,
        (0, 0.2, False),  # 11 X
        (11, 1.0, True),  # 12 X1
        (11, 1.0, True),  # 13 X2
    ]
    codes = np.asarray([0] * 5 + [1] * 3 + [2, 3])
    parent, times, cells, offsets = _forest([tree])
    # The root opens into A alone and B and X together; the group opens into
    # B alone and X, its last clade, which then opens into X1 and X2.
    frame = collapse_phylogeny(
        parent, times, cells, codes, offsets, cut=AT_TIME, max_leaves=10
    )
    assert _wedges(frame) == [(1, 5, 1, 1), (7, 3, 1, 1), (12, 1, 1, 1), (13, 1, 1, 1)]
    assert frame.loc[~frame["wedge"], "node"].tolist() == [0, 11]
    # At three wedges X stays one wedge of two clones, drawn last.
    capped = collapse_phylogeny(
        parent, times, cells, codes, offsets, cut=AT_TIME, max_leaves=3
    )
    assert _wedges(capped) == [(1, 5, 1, 1), (7, 3, 1, 1), (11, 2, 2, 1)]


def test_arguments_of_other_types_collapse_as_their_int64_and_float64_copies() -> None:
    parent, times, cells, offsets = _forest([POLYTOMY])
    expected = collapse_phylogeny(
        parent, times, cells, POLYTOMY_CODES, offsets, cut=AT_TIME, max_leaves=3
    )
    # Lists and narrower integers are copied as int64 and float64.
    found = collapse_phylogeny(
        parent.tolist(),
        times.astype(np.float32),
        cells.astype(np.int32),
        POLYTOMY_CODES.astype(np.int16).tolist(),
        offsets.tolist(),
        cut=AT_TIME,
        max_leaves=3,
        ledger=budgeted(10**9),
    )
    pd.testing.assert_frame_equal(found, expected)
    with pytest.raises(TypeError, match="ledger must be a MemoryLedger or None"):
        collapse_phylogeny(
            parent,
            times,
            cells,
            POLYTOMY_CODES,
            offsets,
            cut=AT_TIME,
            max_leaves=3,
            ledger=10**9,
        )


def test_cells_that_end_before_the_cut_are_wedges_without_a_clone() -> None:
    # Below the root, a leaf at time 0.2, before the cut at 0.5, and a clade of
    # two leaves from time 1; a leaf that names no cell is left out.
    tree = [
        (-1, 0.0, False),
        (0, 0.2, True),
        (0, 1.0, False),
        (2, 2.0, True),
        (2, 2.0, True),
        (0, 3.0, False),
    ]
    parent, times, cells, offsets = _forest([tree])
    frame = collapse_phylogeny(
        parent, times, cells, np.asarray([-1, 0, 0]), offsets, cut=AT_TIME, max_leaves=5
    )
    assert _wedges(frame) == [(2, 2, 1, 1), (1, 1, 0, 1)]
    early = frame.iloc[2]
    assert (early["clone"], early["x"], early["end"]) == (-1, 0.2, 0.2)
    assert frame.iloc[0]["n_cells"] == 3


@pytest.mark.parametrize(
    ("codes", "match"),
    [
        # A cell in a clade without a clone.
        (np.asarray([-1, -1, 1]), "not those of the cut"),
        # A cell outside every clade with a clone.
        (np.asarray([0, 0, 0]), "not those of the cut"),
        # Two clones in one clade.
        (np.asarray([-1, 0, 1]), "not those of the cut"),
    ],
)
def test_clones_that_are_not_the_cuts_are_refused(codes, match) -> None:
    tree = [
        (-1, 0.0, False),
        (0, 0.2, True),
        (0, 1.0, False),
        (2, 2.0, True),
        (2, 2.0, True),
    ]
    parent, times, cells, offsets = _forest([tree])
    with pytest.raises(ValueError, match=match):
        collapse_phylogeny(
            parent, times, cells, codes, offsets, cut=AT_TIME, max_leaves=3
        )


def test_wrong_arguments_are_refused() -> None:
    parent, times, cells, offsets = _forest([POLYTOMY])
    options = {"cut": AT_TIME}
    for value, error, match in (
        (0, ValueError, "max_leaves must be at least 1"),
        (-3, ValueError, "max_leaves must be at least 1"),
        (2.0, TypeError, "max_leaves must be an integer"),
        (True, TypeError, "max_leaves must be an integer"),
        ("5", TypeError, "max_leaves must be an integer"),
    ):
        with pytest.raises(error, match=match):
            collapse_phylogeny(
                parent,
                times,
                cells,
                POLYTOMY_CODES,
                offsets,
                max_leaves=value,
                **options,
            )
    with pytest.raises(ValueError, match="arrays disagree"):
        collapse_phylogeny(
            parent, times, cells[:-1], POLYTOMY_CODES, offsets, max_leaves=2, **options
        )
    with pytest.raises(ValueError, match="arrays disagree"):
        collapse_phylogeny(
            parent, times, cells, POLYTOMY_CODES, [0, 5], max_leaves=2, **options
        )
    # Trees that do not start at a root, or hold no node.
    for offsets_given in ([0, 1, 14], [0, 0, 14]):
        with pytest.raises(ValueError, match="start at its root"):
            collapse_phylogeny(
                parent,
                times,
                cells,
                POLYTOMY_CODES,
                offsets_given,
                max_leaves=2,
                **options,
            )
    with pytest.raises(ValueError, match="needs a time or a depth"):
        collapse_phylogeny(
            parent,
            times,
            cells,
            POLYTOMY_CODES,
            offsets,
            cut=PhylogenyCut(time=None, depth=None),
            max_leaves=2,
        )
    with pytest.raises(ValueError, match="clones code 9 cells"):
        collapse_phylogeny(
            parent, times, cells, POLYTOMY_CODES[:-1], offsets, max_leaves=2, **options
        )
    with pytest.raises(ValueError, match="needs each node's time"):
        collapse_phylogeny(
            parent, None, cells, POLYTOMY_CODES, offsets, max_leaves=2, **options
        )
    with pytest.raises(ValueError, match="has no time"):
        collapse_phylogeny(
            parent,
            np.where(np.arange(len(times)) == 3, np.nan, times),
            cells,
            POLYTOMY_CODES,
            offsets,
            max_leaves=2,
            **options,
        )
    with pytest.raises(ValueError, match="so there is nothing to plot$"):
        collapse_phylogeny(
            parent,
            times,
            np.full(len(parent), -1),
            POLYTOMY_CODES,
            offsets,
            max_leaves=2,
            **options,
        )
    # An internal node that names a cell.
    named = cells.copy()
    named[1] = 0
    with pytest.raises(ValueError, match="names a cell"):
        collapse_phylogeny(
            parent, times, named, POLYTOMY_CODES, offsets, max_leaves=2, **options
        )


# The recursive reference: each rule computed again by walking the trees from
# their roots, with lists sorted afresh at each step in place of heaps.


@dataclass
class _Item:
    """A row of the reference: a node, or a group of sibling clades or of trees."""

    kind: str
    node: int = -1
    top: int = -1
    members: list[int] = field(default_factory=list)
    parent: "_Item | None" = None
    children: list["_Item"] = field(default_factory=list)
    opened: bool = False


def _reference(parent, times, cells, codes, offsets, cut, max_leaves) -> pd.DataFrame:
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for node, above in enumerate(parent.tolist()):
        if above >= 0:
            children[above].append(node)
    roots = offsets[:-1].tolist()
    x = [0.0] * n

    def place(node: int, depth: int) -> None:
        x[node] = float(depth)
        for child in children[node]:
            place(child, depth + (len(children[node]) >= 2))

    if cut.time is not None:
        x = [float(value) for value in times]
    else:
        for root in roots:
            place(root, 0)
    past = [value >= (cut.time if cut.time is not None else cut.depth) for value in x]
    in_clade = [False] * n

    def mark(node: int, inside: bool) -> None:
        in_clade[node] = inside or past[node]
        for child in children[node]:
            mark(child, in_clade[node])

    for root in roots:
        mark(root, False)
    cell_leaves: list[list[int]] = [[] for _ in range(n)]

    def gather(node: int) -> list[int]:
        found = [node] if cells[node] >= 0 else []
        for child in children[node]:
            found += gather(child)
        cell_leaves[node] = found
        return found

    for root in roots:
        gather(root)

    def n_cells(node: int) -> int:
        return len(cell_leaves[node])

    def clone_codes(nodes: list[int]) -> set[int]:
        return {
            int(codes[cells[leaf]])
            for node in nodes
            for leaf in cell_leaves[node]
            if codes[cells[leaf]] >= 0
        }

    def key(node: int) -> tuple[int, int]:
        return (-n_cells(node), node)

    def held(node: int) -> list[int]:
        return [child for child in children[node] if n_cells(child) > 0]

    def item(top: int, above: "_Item | None") -> _Item:
        node = top
        while not in_clade[node] and len(held(node)) == 1:
            node = held(node)[0]
        return _Item("node", node=node, top=top, parent=above)

    def cells_of(row: _Item) -> int:
        return (
            n_cells(row.node) if row.kind == "node" else sum(map(n_cells, row.members))
        )

    def first_of(row: _Item) -> int:
        return row.node if row.kind == "node" else min(row.members, key=key)

    trees = sorted((root for root in roots if n_cells(root) > 0), key=key)
    if len(trees) <= max_leaves:
        top_rows = [item(root, None) for root in trees]
    else:
        top_rows = [item(root, None) for root in trees[: max_leaves - 1]]
        top_rows.append(_Item("trees", members=trees[max_leaves - 1 :]))
    rows = list(top_rows)
    wedges = len(top_rows)
    while wedges < max_leaves:
        candidates = [
            row
            for row in rows
            if not row.opened
            and (
                row.kind == "siblings"
                or (
                    row.kind == "node" and not in_clade[row.node] and children[row.node]
                )
            )
        ]
        if not candidates:
            break
        row = min(candidates, key=lambda row: (-cells_of(row), first_of(row)))
        if row.kind == "node":
            row.opened = True
            kids = sorted(held(row.node), key=key)
            row.children = [item(kids[0], row)]
            if len(kids) == 2:
                row.children.append(item(kids[1], row))
            else:
                row.children.append(_Item("siblings", members=kids[1:], parent=row))
            rows += row.children
        else:
            kids = sorted(row.members, key=key)
            siblings = row.parent.children
            at = siblings.index(row)
            alone = item(kids[0], row.parent)
            siblings.insert(at, alone)
            rows.append(alone)
            if len(kids) == 2:
                last = item(kids[1], row.parent)
                siblings[at + 1] = last
                rows.remove(row)
                rows.append(last)
            else:
                row.members = kids[1:]
        wedges += 1
    drawn: list[_Item] = []

    def walk(row: _Item) -> None:
        drawn.append(row)
        for child in row.children:
            walk(child)

    for row in top_rows:
        walk(row)
    position = {id(row): at for at, row in enumerate(drawn)}
    low: dict[int, float] = {}
    high: dict[int, float] = {}
    middle: dict[int, float] = {}
    cursor = 0.0
    for row in drawn:
        if not row.opened:
            low[id(row)], high[id(row)] = cursor, cursor + cells_of(row)
            middle[id(row)] = cursor + cells_of(row) / 2
            cursor += cells_of(row)

    def settle(row: _Item) -> None:
        if row.opened:
            for child in row.children:
                settle(child)
            first, last = row.children[0], row.children[-1]
            low[id(row)], high[id(row)] = low[id(first)], high[id(last)]
            middle[id(row)] = (middle[id(first)] + middle[id(last)]) / 2

    for row in top_rows:
        settle(row)
    records = []
    for row in drawn:
        nodes = [row.node] if row.kind == "node" else row.members
        found = clone_codes(nodes)
        reach = [x[leaf] for node in nodes for leaf in cell_leaves[node]]
        if row.kind == "trees":
            tree, node = -1, -1
        else:
            anchor = row.node if row.kind == "node" else row.parent.node
            tree = int(np.searchsorted(offsets, anchor, side="right") - 1)
            node = row.node - int(offsets[tree]) if row.kind == "node" else -1
        if row.parent is not None:
            start = x[row.parent.node]
        elif row.kind == "node":
            start = x[row.top]
        else:
            start = min(x[member] for member in row.members)
        records.append(
            {
                "tree": tree,
                "node": node,
                "parent": position[id(row.parent)] if row.parent is not None else -1,
                "wedge": not row.opened,
                "n_cells": cells_of(row),
                "n_clones": len(found),
                "clone": next(iter(found)) if len(found) == 1 else -1,
                "n_clades": 1 if row.kind == "node" else len(row.members),
                "n_trees": len(row.members) if row.kind == "trees" else 1,
                "x": x[row.node]
                if row.kind == "node"
                else min(x[m] for m in row.members),
                "start": start,
                "end": max(reach),
                "y": middle[id(row)],
                "low": low[id(row)],
                "high": high[id(row)],
            }
        )
    return pd.DataFrame(records, columns=COLUMNS)


def _random_tree(
    rng: np.random.Generator, n_nodes: int
) -> list[tuple[int, float, bool]]:
    """A random tree in preorder: each node hangs from the last node or one of its ancestors.

    Lengths are mostly positive, now and then 0 or negative, and three in four
    leaves name a cell.
    """
    nodes: list[tuple[int, float, bool]] = [(-1, float(rng.integers(0, 3)), False)]
    path = [0]
    for node in range(1, n_nodes):
        del path[len(path) - int(rng.integers(0, len(path))) :]
        path = path or [0]
        above = path[-1]
        length = float(rng.choice([0.0, -0.5, 0.5, 1.0, 1.0, 2.0, 3.0]))
        nodes.append((above, nodes[above][1] + length, False))
        path.append(node)
    has_children = {above for above, _time, _cell in nodes if above >= 0}
    return [
        (above, time, node not in has_children and bool(rng.random() < 0.75))
        for node, (above, time, _cell) in enumerate(nodes)
    ]


def _cut_codes(parent, times, cells, offsets, cut) -> np.ndarray:
    """Each cell's clone at the cut, numbered by clade in node order, from first principles."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for node, above in enumerate(parent.tolist()):
        if above >= 0:
            children[above].append(node)
    codes = np.full(int(cells.max()) + 1, -1, dtype=np.int64)
    clade: dict[int, int] = {}
    stack = [(root, -1, 0) for root in offsets[:-1].tolist()]
    while stack:
        node, root, depth = stack.pop()
        value = times[node] if cut.time is not None else depth
        limit = cut.time if cut.time is not None else cut.depth
        if root < 0 and value >= limit:
            root = node
        if cells[node] >= 0 and root >= 0:
            codes[cells[node]] = clade.setdefault(root, len(clade))
        for child in children[node]:
            stack.append((child, root, depth + (len(children[node]) >= 2)))
    return codes


@pytest.mark.parametrize("seed", range(200))
def test_the_passes_equal_a_recursive_reference_on_random_forests(seed) -> None:
    rng = np.random.default_rng(seed)
    trees = []
    for _tree in range(int(rng.integers(1, 4))):
        tree = _random_tree(rng, int(rng.integers(1, 40)))
        trees.append(tree)
    parent, times, cells, offsets = _forest(trees)
    if not (cells >= 0).any():
        # The last node in preorder is a leaf; it names the one cell.
        cells[-1] = 0
    if rng.random() < 0.5:
        cut = PhylogenyCut(time=float(rng.choice(np.unique(times))), depth=None)
    else:
        cut = PhylogenyCut(time=None, depth=int(rng.integers(0, 6)))
    codes = _cut_codes(parent, times, cells, offsets, cut)
    max_leaves = int(rng.choice([1, 2, 3, 5, 8, 13, 100]))
    found = collapse_phylogeny(
        parent,
        times if cut.time is not None else None,
        cells,
        codes,
        offsets,
        cut=cut,
        max_leaves=max_leaves,
    )
    expected = _reference(parent, times, cells, codes, offsets, cut, max_leaves)
    pd.testing.assert_frame_equal(found, expected, check_dtype=False)
    # The cap: as many wedges as it allows, or every wedge the cut allows.
    assert found["wedge"].sum() <= max_leaves
    assert found["n_cells"][found["parent"] < 0].sum() == (cells >= 0).sum()


def _caterpillar(depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A caterpillar ``depth`` levels deep, in preorder: spine nodes, then leaves.

    Spine node k, at position k and time k, holds spine node k + 1 and a leaf;
    the last holds two leaves. In preorder the spine comes first, then the
    last spine node's two leaves, then each earlier spine node's leaf from
    the deepest up. Every leaf is at time ``depth + 1`` and names a cell.
    """
    parent = np.empty(2 * depth + 1, dtype=np.int64)
    parent[0] = -1
    parent[1:depth] = np.arange(depth - 1)
    parent[depth : depth + 2] = depth - 1
    parent[depth + 2 :] = np.arange(depth - 2, -1, -1)
    times = np.concatenate(
        [np.arange(depth, dtype=np.float64), np.full(depth + 1, depth + 1.0)]
    )
    cells = np.concatenate([np.full(depth, -1), np.arange(depth + 1)]).astype(np.int64)
    return parent, times, cells


def test_a_caterpillar_30000_levels_deep_collapses_without_recursion() -> None:
    depth = 30_000
    parent, times, cells = _caterpillar(depth)
    n_cells = int((cells >= 0).sum())
    assert n_cells == depth + 1
    offsets = np.asarray([0, len(parent)])
    # Cut past the spine: each leaf is a clone of its own.
    cut = PhylogenyCut(time=depth + 0.5, depth=None)
    codes = np.arange(n_cells, dtype=np.int64)
    limit = sys.getrecursionlimit()
    assert depth > 10 * limit
    capped = collapse_phylogeny(
        parent, times, cells, codes, offsets, cut=cut, max_leaves=100
    )
    # Each step opens the next spine node: its leaf alone, the rest below.
    assert capped["wedge"].sum() == 100
    assert capped["n_cells"][capped["parent"] < 0].tolist() == [n_cells]
    every = collapse_phylogeny(
        parent, times, cells, codes, offsets, cut=cut, max_leaves=n_cells
    )
    assert every["wedge"].sum() == n_cells
    assert (every.loc[every["wedge"], "n_cells"] == 1).all()
    assert every.loc[~every["wedge"], "n_cells"].iloc[0] == n_cells
    # The deepest spine node opened stands at time depth - 1.
    assert every.loc[~every["wedge"], "x"].max() == depth - 1
