"""Cell phylogenies and the clones that cutting them defines.

A cell phylogeny is a forest: each tree's leaves are cells, and its internal
nodes their reconstructed ancestors. A forest is stored as arrays over its
nodes, the trees one after another and each in preorder, so a parent always
comes before its children: each node's parent, its time and its name.

A node's time runs from its tree's root, plus any length that the file gives
above the root, in a unit that the user declares. Files give times directly,
as TreeData's node ``time`` does, or branch lengths, as Newick does, and a
node's own time is preferred over the time its branch length implies.

Cutting a tree at a time makes a clone of each clade whose root is the first
node at or past the cut on the way down from the tree's root: the cells that
descend from one lineage alive at that time. Cutting at a depth counts
branchings instead, after merging each node that has one child with that
child, so a root with one child, as PEtracer's trees have, does not shift the
depths. Each child of a polytomy that the cut passes is a clone of its own. A
leaf whose path never reaches the cut, one that ends before the time or
fewer branchings than the depth below its root, is in no clone. These
functions never touch storage.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.numba import njit
from ..utils.arguments import quoted_value

__all__ = [
    "TREE_SEPARATOR",
    "CellPhylogeny",
    "PhylogenyCut",
    "clade_roots",
    "clone_labels",
    "collapsed_depths",
    "leaf_time_ranges",
    "node_times",
    "phylogeny_frames",
    "phylogeny_cut",
    "tree_codes",
]

#: The text between a tree's name and a node's in the labels of clones cut
#: from a forest of several trees.
TREE_SEPARATOR = "/"
# Clones of a tree whose nodes do not all have names of their own are named
# by this text and the position of their root in the tree.
_POSITION_PREFIX = "node"


@dataclass(frozen=True, slots=True)
class PhylogenyCut:
    """Where to cut a phylogeny: at a time or at a depth.

    Attributes:
        time: The time of the cut from each tree's root, or None.
        depth: The branchings below each tree's root, or None.
    """

    time: float | None
    depth: int | None

    @property
    def by(self) -> str:
        """``"time"`` or ``"depth"``."""
        return "time" if self.time is not None else "depth"


def phylogeny_cut(time: Any, depth: Any) -> PhylogenyCut:
    """Check where to cut a phylogeny; exactly one of the two must be given.

    Raises:
        TypeError: If ``time`` is not a real number or ``depth`` not an
            integer.
        ValueError: If both or neither are given, ``time`` is not finite, or
            ``depth`` is negative.
    """
    if (time is None) == (depth is None):
        raise ValueError(
            "Pass exactly one of time, a time from each tree's root, and depth, "
            "a number of branchings below it"
        )
    if time is not None:
        if isinstance(time, bool) or not isinstance(time, int | float | np.number):
            raise TypeError("time must be a real number")
        value = float(time)
        if not np.isfinite(value):
            raise ValueError(f"time must be finite, not {quoted_value(time)}")
        return PhylogenyCut(time=value, depth=None)
    if isinstance(depth, bool) or not isinstance(depth, int | np.integer):
        raise TypeError("depth must be an integer")
    if int(depth) < 0:
        raise ValueError(f"depth must be at least 0, not {quoted_value(depth)}")
    return PhylogenyCut(time=None, depth=int(depth))


def tree_codes(tree_offsets: np.ndarray) -> np.ndarray:
    """Each node's tree, as a position among the trees, as int64."""
    offsets = np.asarray(tree_offsets, dtype=np.int64)
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))


@njit(cache=True)
def _resolved_times(
    parent: np.ndarray,
    length: np.ndarray,
    given: np.ndarray,
    timed: np.ndarray,
    tree: np.ndarray,
) -> np.ndarray:
    times = np.empty(len(parent), dtype=np.float64)
    for node in range(len(parent)):
        if not timed[tree[node]]:
            times[node] = np.nan
        elif not np.isnan(given[node]):
            times[node] = given[node]
        elif parent[node] < 0:
            # A root starts at its own length, or at 0.
            times[node] = 0.0 if np.isnan(length[node]) else length[node]
        else:
            times[node] = times[parent[node]] + length[node]
    return times


def node_times(
    parent: np.ndarray,
    length: np.ndarray,
    time: np.ndarray,
    tree_offsets: np.ndarray,
) -> np.ndarray:
    """Each node's time from its tree's root, plus any length above the root.

    A node's own time is preferred; otherwise its parent's time plus the
    length of its branch. A root without a time starts at its own length, the
    stem above it, or at 0. A tree without any time or length has no times,
    and a node below a branch without a length has none either.

    Args:
        parent: Each node's parent, -1 for a root; parents come first.
        length: The length of the branch above each node, NaN for none.
        time: Each node's own time, NaN for none.
        tree_offsets: Where each tree's nodes start, and the end.

    Returns:
        Each node's time, NaN where it is unknown, as float64.
    """
    parent = np.asarray(parent, dtype=np.int64)
    length = np.asarray(length, dtype=np.float64)
    given = np.asarray(time, dtype=np.float64)
    trees = tree_codes(tree_offsets)
    n_trees = len(tree_offsets) - 1
    known = ~np.isnan(length) | ~np.isnan(given)
    timed = np.bincount(trees[known], minlength=n_trees) > 0
    return _resolved_times(parent, length, given, timed, trees)


@njit(cache=True)
def _depths(parent: np.ndarray, branches: np.ndarray) -> np.ndarray:
    depths = np.zeros(len(parent), dtype=np.int64)
    for node in range(len(parent)):
        above = parent[node]
        if above >= 0:
            depths[node] = depths[above] + (1 if branches[above] else 0)
    return depths


def collapsed_depths(parent: np.ndarray) -> np.ndarray:
    """Each node's branchings below its root, single-child nodes merged.

    A node one branching below its parent is one deeper; the child of a node
    with one child is that node merged with it, at the same depth. A root is
    at depth 0.
    """
    parent = np.asarray(parent, dtype=np.int64)
    n_children = np.bincount(parent[parent >= 0], minlength=len(parent))
    return _depths(parent, n_children >= 2)


@njit(cache=True)
def _clade_roots(parent: np.ndarray, past: np.ndarray) -> np.ndarray:
    roots = np.full(len(parent), -1, dtype=np.int64)
    for node in range(len(parent)):
        above = parent[node]
        if above >= 0 and roots[above] >= 0:
            roots[node] = roots[above]
        elif past[node]:
            roots[node] = node
    return roots


def clade_roots(parent: np.ndarray, past_cut: np.ndarray) -> np.ndarray:
    """Each node's clade: the first node at or past the cut from its root.

    Args:
        parent: Each node's parent, -1 for a root; parents come first.
        past_cut: Whether each node is at or past the cut.

    Returns:
        The root of each node's clade, as a node, -1 for a node whose path
        from its root never reaches the cut, as int64.
    """
    return _clade_roots(
        np.asarray(parent, dtype=np.int64), np.asarray(past_cut, dtype=np.bool_)
    )


def clone_labels(
    roots: np.ndarray,
    *,
    tree_offsets: np.ndarray,
    tree_names: np.ndarray,
    unique_names: np.ndarray,
    names: np.ndarray,
) -> np.ndarray:
    """Label the clades that clones are cut from, by their root nodes.

    A clade is named by its root's name when every node of its tree has a
    name of its own, as PEtracer's and Cassiopeia's trees do, and otherwise by
    ``node`` and its root's position in the tree, from 0 at the tree's root.
    With several trees, each label starts with its tree's name and ``/``.

    Args:
        roots: The roots of the clades, as nodes.
        tree_offsets: Where each tree's nodes start, and the end.
        tree_names: Each tree's name.
        unique_names: Whether each tree's nodes all have names of their own.
        names: The names of ``roots``, in their order.

    Returns:
        Each clade's label, as an object array of text.
    """
    roots = np.asarray(roots, dtype=np.int64)
    offsets = np.asarray(tree_offsets, dtype=np.int64)
    trees = np.searchsorted(offsets, roots, side="right") - 1
    positions = roots - offsets[trees]
    named = np.asarray(unique_names, dtype=bool)[trees]
    several = len(offsets) > 2
    prefixes = np.asarray(tree_names, dtype=object)
    labels = np.empty(len(roots), dtype=object)
    for index, (tree, position, use_name, name) in enumerate(
        zip(
            trees.tolist(), positions.tolist(), named.tolist(), list(names), strict=True
        )
    ):
        node = str(name) if use_name else f"{_POSITION_PREFIX}{position}"
        labels[index] = f"{prefixes[tree]}{TREE_SEPARATOR}{node}" if several else node
    return labels


def leaf_time_ranges(
    times: np.ndarray, leaf: np.ndarray, tree_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The earliest and latest leaf time of each tree, NaN where unknown.

    The gap between them is how far a tree is from ultrametric, in which
    every leaf ends at one time.
    """
    times = np.asarray(times, dtype=np.float64)
    trees = tree_codes(tree_offsets)
    n_trees = len(tree_offsets) - 1
    leaves = np.asarray(leaf, dtype=bool)
    leaf_trees, leaf_times = trees[leaves], times[leaves]
    known = ~np.isnan(leaf_times)
    earliest = np.full(n_trees, np.inf)
    latest = np.full(n_trees, -np.inf)
    np.minimum.at(earliest, leaf_trees[known], leaf_times[known])
    np.maximum.at(latest, leaf_trees[known], leaf_times[known])
    # A tree with a leaf of unknown time has no range.
    unknown = np.bincount(leaf_trees[~known], minlength=n_trees) > 0
    unknown |= ~np.isfinite(earliest)
    earliest[unknown] = np.nan
    latest[unknown] = np.nan
    return earliest, latest


@dataclass(frozen=True, slots=True)
class CellPhylogeny:
    """Cell phylogenies, read from a ``cell_phylogeny`` artifact.

    Attributes:
        nodes: One row per node, the trees one after another in the natural
            order of their names and each in preorder: ``tree`` (its tree's name, as a categorical), ``node``
            (its position in its tree, 0 at the root), ``name`` (its name,
            ``""`` for none), ``parent`` (its parent's position in the tree,
            -1 for the root), ``time`` (its time from the root plus any length
            above the root, NaN where unknown), ``depth`` (its branchings below
            the root, single-child nodes merged), ``is_leaf`` and ``ids`` (the
            ID of the selected cell that a leaf names, NA for none).
        trees: One row per tree: ``tree``, ``n_nodes``, ``n_leaves``,
            ``n_cells`` (its leaves that name a selected cell), ``root_time``,
            ``earliest_leaf_time`` and ``latest_leaf_time``.
        time_unit: The unit of the times, as declared at import, or None.
    """

    nodes: pd.DataFrame = field(repr=False)
    trees: pd.DataFrame = field(repr=False)
    time_unit: str | None


def phylogeny_frames(
    *,
    parent: np.ndarray,
    times: np.ndarray,
    names: np.ndarray,
    cell_ids: np.ndarray,
    tree_offsets: np.ndarray,
    tree_names: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the tables of nodes and trees of stored phylogenies.

    Args:
        parent: Each node's parent as a node of the forest, -1 for a root.
        times: Each node's time.
        names: Each node's name.
        cell_ids: The ID of the cell that each node names, None for none.
        tree_offsets: Where each tree's nodes start, and the end.
        tree_names: Each tree's name.
    """
    parent = np.asarray(parent, dtype=np.int64)
    offsets = np.asarray(tree_offsets, dtype=np.int64)
    trees = tree_codes(offsets)
    starts = offsets[trees]
    is_leaf = np.ones(len(parent), dtype=bool)
    is_leaf[parent[parent >= 0]] = False
    nodes = pd.DataFrame(
        {
            "tree": pd.Categorical.from_codes(
                trees, categories=pd.Index(np.asarray(tree_names, dtype=object))
            ),
            "node": np.arange(len(parent), dtype=np.int64) - starts,
            "name": np.asarray(names, dtype=object),
            "parent": np.where(parent >= 0, parent - starts, -1),
            "time": np.asarray(times, dtype=np.float64),
            "depth": collapsed_depths(parent),
            "is_leaf": is_leaf,
            "ids": pd.array(np.asarray(cell_ids, dtype=object), dtype="string"),
        }
    )
    earliest, latest = leaf_time_ranges(times, is_leaf, offsets)
    with_cell = np.asarray(pd.notna(cell_ids), dtype=bool)
    n_trees = len(offsets) - 1
    table = pd.DataFrame(
        {
            "tree": np.asarray(tree_names, dtype=object),
            "n_nodes": np.diff(offsets),
            "n_leaves": np.bincount(trees[is_leaf], minlength=n_trees),
            "n_cells": np.bincount(trees[with_cell], minlength=n_trees),
            "root_time": np.asarray(times, dtype=np.float64)[offsets[:-1]],
            "earliest_leaf_time": earliest,
            "latest_leaf_time": latest,
        }
    )
    return nodes, table
