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

from ..storage.refs import ArtifactRef
from ..utils.arguments import integer_argument, quoted_value
from ..utils.memory_budget import NUMPY_STEP_BYTES, MemoryLedger
from ..utils.numba import njit

__all__ = [
    "TREE_SEPARATOR",
    "CellPhylogeny",
    "CollapsedPhylogeny",
    "PhylogenyCut",
    "clade_roots",
    "clade_labels",
    "collapse_phylogeny",
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


def clade_labels(
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


# A row of a collapsed phylogeny is a node of a tree, the sibling clades of
# an opened node that the cap keeps together, or the trees that it keeps
# together.
_NODE = 0
_SIBLINGS = 1
_TREES = 2
# What collapsing holds, beside the arrays it is given. For a cut at a depth,
# each node's depth as a position, made from collapsed depths, which hold 17
# bytes a node at once. Each node's flag of its clade, made beside the cut's
# flag and the clades' roots. From the pass back, each node's subtree size,
# cells, clones, clone and latest leaf.
_DEPTH_PEAK_BYTES_PER_NODE = 17
_CLADE_PEAK_BYTES_PER_NODE = 10
_COUNT_BYTES_PER_NODE = 40
# Opening the clades holds each node's place among the clades opened, and for
# each row that the cap allows eleven numbers and a flag, of which the
# opening's own four go when it ends and three more once the rows' values are
# found.
_POOL_BYTES_PER_NODE = 8
_OPENING_BYTES_PER_SLOT = 89
_OPENING_SCRATCH_PER_SLOT = 32
_OPENED_SCRATCH_PER_SLOT = 24
# Each row's values: its cells, clones, clone, clades, trees, position, start
# and end; its place: whether it is a wedge, its order, extent and middle,
# found beside three more numbers a row; and its tree, node and parent in
# drawing order, found beside one more.
_VALUE_BYTES_PER_ROW = 64
_PLACE_BYTES_PER_ROW = 33
_PLACING_BYTES_PER_ROW = 24
_ORDERED_BYTES_PER_ROW = 24
# The table's fourteen columns, eight of integers, a flag and six of
# numbers, made in drawing order; pandas 2 copies them into the table's
# blocks, and pandas 3 keeps them.
_TABLE_BYTES_PER_ROW = 113


@njit(cache=True)
def _clade_counts(
    parent: np.ndarray,
    cells: np.ndarray,
    codes: np.ndarray,
    in_clade: np.ndarray,
    x: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(parent)
    size = np.ones(n, dtype=np.int64)
    n_cells = np.zeros(n, dtype=np.int64)
    n_clones = np.zeros(n, dtype=np.int64)
    clone = np.full(n, -1, dtype=np.int64)
    reach = np.full(n, -np.inf)
    # Children come after their parent, so going back from the last node
    # finishes each node's children before the node.
    for node in range(n - 1, -1, -1):
        cell = cells[node]
        if cell >= 0:
            code = codes[cell]
            # Only a leaf names a cell, and its cell has a clone exactly when
            # the leaf is in a clade of the cut.
            if size[node] > 1 or (code >= 0) != in_clade[node]:
                return node, size, n_cells, n_clones, clone, reach
            n_cells[node] = 1
            clone[node] = code
            reach[node] = x[node]
        if in_clade[node] and n_cells[node] > 0:
            n_clones[node] = 1
        above = parent[node]
        if above < 0:
            continue
        size[above] += size[node]
        if n_cells[node] == 0:
            continue
        n_cells[above] += n_cells[node]
        reach[above] = max(reach[above], reach[node])
        if in_clade[above]:
            # Every cell of a clade is in the clade's one clone.
            if clone[above] >= 0 and clone[above] != clone[node]:
                return node, size, n_cells, n_clones, clone, reach
            clone[above] = clone[node]
        else:
            n_clones[above] += n_clones[node]
            clone[above] = max(clone[above], clone[node])
    return -1, size, n_cells, n_clones, clone, reach


@njit(cache=True)
def _first_branching(
    node: int, size: np.ndarray, n_cells: np.ndarray, in_clade: np.ndarray
) -> int:
    """Follow the one child that holds cells down from ``node`` until a node with two, a leaf or a clade of the cut."""
    while not in_clade[node] and size[node] > 1:
        only = -1
        count = 0
        child = node + 1
        end = node + size[node]
        while child < end and count < 2:
            if n_cells[child] > 0:
                count += 1
                only = child
            child += size[child]
        if count != 1:
            return node
        node = only
    return node


@njit(cache=True)
def _before(cells_a: int, node_a: int, cells_b: int, node_b: int) -> bool:
    """Whether a clade of ``cells_a`` cells at ``node_a`` opens before one of ``cells_b`` at ``node_b``: more cells first, then the earlier node."""
    return cells_a > cells_b or (cells_a == cells_b and node_a < node_b)


@njit(cache=True)
def _sift_clades(
    pool: np.ndarray, start: int, count: int, at: int, n_cells: np.ndarray
) -> None:
    """Restore the heap of the ``count`` clades at ``pool[start]``, the clade of the most cells first, below position ``at``."""
    while True:
        best = 2 * at + 1
        if best >= count:
            return
        if best + 1 < count:
            left, right = pool[start + best], pool[start + best + 1]
            if _before(n_cells[right], right, n_cells[left], left):
                best += 1
        above, below = pool[start + at], pool[start + best]
        if not _before(n_cells[below], below, n_cells[above], above):
            return
        pool[start + at], pool[start + best] = below, above
        at = best


@njit(cache=True)
def _heapify_clades(
    pool: np.ndarray, start: int, count: int, n_cells: np.ndarray
) -> None:
    for at in range(count // 2 - 1, -1, -1):
        _sift_clades(pool, start, count, at, n_cells)


@njit(cache=True)
def _pop_clade(pool: np.ndarray, start: int, count: int, n_cells: np.ndarray) -> int:
    """Take the clade of the most cells from the heap of the ``count`` clades at ``pool[start]``."""
    top = pool[start]
    pool[start] = pool[start + count - 1]
    pool[start + count - 1] = top
    _sift_clades(pool, start, count - 1, 0, n_cells)
    return int(top)


@njit(cache=True)
def _row_before(a: int, b: int, cells: np.ndarray, first: np.ndarray) -> bool:
    return _before(cells[a], first[a], cells[b], first[b])


@njit(cache=True)
def _push_row(
    heap: np.ndarray, n_heap: int, row: int, cells: np.ndarray, first: np.ndarray
) -> int:
    """Add ``row`` to the heap of rows to open, the row of the most cells first."""
    at = n_heap
    heap[at] = row
    while at > 0:
        above = (at - 1) // 2
        if not _row_before(heap[at], heap[above], cells, first):
            break
        heap[at], heap[above] = heap[above], heap[at]
        at = above
    return n_heap + 1


@njit(cache=True)
def _pop_row(
    heap: np.ndarray, n_heap: int, cells: np.ndarray, first: np.ndarray
) -> tuple[int, int]:
    """Take the row of the most cells from the heap of rows to open."""
    top = heap[0]
    n_heap -= 1
    heap[0] = heap[n_heap]
    at = 0
    while True:
        best = 2 * at + 1
        if best >= n_heap:
            break
        if best + 1 < n_heap and _row_before(heap[best + 1], heap[best], cells, first):
            best += 1
        if not _row_before(heap[best], heap[at], cells, first):
            break
        heap[at], heap[best] = heap[best], heap[at]
        at = best
    return top, n_heap


@njit(cache=True)
def _add_clade(
    clade: int,
    parent_row: int,
    rank: int,
    n_rows: int,
    n_heap: int,
    rows: tuple[np.ndarray, ...],
    heap: np.ndarray,
    size: np.ndarray,
    n_cells: np.ndarray,
    in_clade: np.ndarray,
) -> tuple[int, int]:
    """Draw ``clade`` as a row below ``parent_row``, from its first branching, and queue it to open if it can."""
    kind, node, top, parent, ranks, cells, first = rows[:7]
    row = n_rows
    branching = _first_branching(clade, size, n_cells, in_clade)
    kind[row] = _NODE
    node[row] = branching
    top[row] = clade
    parent[row] = parent_row
    ranks[row] = rank
    cells[row] = n_cells[branching]
    first[row] = branching
    # Only a clade above the cut with children opens, never one of the cut.
    if not in_clade[branching] and size[branching] > 1:
        n_heap = _push_row(heap, n_heap, row, cells, first)
    return n_rows + 1, n_heap


@njit(cache=True)
def _collapse(
    size: np.ndarray,
    n_cells: np.ndarray,
    in_clade: np.ndarray,
    tree_roots: np.ndarray,
    max_leaves: int,
    capacity: int,
) -> tuple[np.ndarray, ...]:
    n = len(size)
    # The trees that hold cells, and the children that hold cells of each
    # node opened, each set a heap of the clade of the most cells first.
    pool = np.empty(n, dtype=np.int64)
    n_trees = 0
    for root in tree_roots:
        if n_cells[root] > 0:
            pool[n_trees] = root
            n_trees += 1
    kind = np.zeros(capacity, dtype=np.int64)
    node = np.full(capacity, -1, dtype=np.int64)
    top = np.full(capacity, -1, dtype=np.int64)
    parent = np.full(capacity, -1, dtype=np.int64)
    ranks = np.zeros(capacity, dtype=np.int64)
    cells = np.zeros(capacity, dtype=np.int64)
    # The node by which a row's ties open: its own, or its group's largest.
    first = np.zeros(capacity, dtype=np.int64)
    opened = np.zeros(capacity, dtype=np.bool_)
    start = np.zeros(capacity, dtype=np.int64)
    members = np.zeros(capacity, dtype=np.int64)
    revealed = np.zeros(capacity, dtype=np.int64)
    heap = np.empty(capacity, dtype=np.int64)
    rows = (kind, node, top, parent, ranks, cells, first)
    n_rows = 0
    n_heap = 0
    used = n_trees
    _heapify_clades(pool, 0, n_trees, n_cells)
    # Each tree is a wedge while the trees fit the cap; past it, the
    # smallest share one.
    shown = n_trees if n_trees <= max_leaves else max_leaves - 1
    left = n_trees
    for rank in range(shown):
        root = _pop_clade(pool, 0, left, n_cells)
        left -= 1
        n_rows, n_heap = _add_clade(
            root, -1, rank, n_rows, n_heap, rows, heap, size, n_cells, in_clade
        )
    wedges = shown
    if left > 0:
        kind[n_rows] = _TREES
        ranks[n_rows] = shown
        members[n_rows] = left
        n_rows += 1
        wedges += 1
    while wedges < max_leaves and n_heap > 0:
        row, n_heap = _pop_row(heap, n_heap, cells, first)
        if kind[row] == _NODE:
            # Open the clade: its children that hold cells, the largest drawn
            # alone and the others together until they open in turn.
            here = node[row]
            begin = used
            child = here + 1
            end = here + size[here]
            while child < end:
                if n_cells[child] > 0:
                    pool[used] = child
                    used += 1
                child += size[child]
            count = used - begin
            _heapify_clades(pool, begin, count, n_cells)
            opened[row] = True
            largest = _pop_clade(pool, begin, count, n_cells)
            count -= 1
            n_rows, n_heap = _add_clade(
                largest, row, 0, n_rows, n_heap, rows, heap, size, n_cells, in_clade
            )
            revealed[row] = 1
            if count == 1:
                n_rows, n_heap = _add_clade(
                    pool[begin],
                    row,
                    1,
                    n_rows,
                    n_heap,
                    rows,
                    heap,
                    size,
                    n_cells,
                    in_clade,
                )
                revealed[row] = 2
            else:
                group = n_rows
                n_rows += 1
                kind[group] = _SIBLINGS
                node[group] = here
                parent[group] = row
                start[group] = begin
                members[group] = count
                cells[group] = n_cells[here] - n_cells[largest]
                first[group] = pool[begin]
                n_heap = _push_row(heap, n_heap, group, cells, first)
        else:
            # Draw the largest clade of a group alone.
            above = parent[row]
            begin = start[row]
            count = members[row]
            largest = _pop_clade(pool, begin, count, n_cells)
            count -= 1
            n_rows, n_heap = _add_clade(
                largest,
                above,
                revealed[above],
                n_rows,
                n_heap,
                rows,
                heap,
                size,
                n_cells,
                in_clade,
            )
            revealed[above] += 1
            if count == 1:
                # The last clade of the group is drawn in the group's row.
                last = pool[begin]
                branching = _first_branching(last, size, n_cells, in_clade)
                kind[row] = _NODE
                node[row] = branching
                top[row] = last
                ranks[row] = revealed[above]
                revealed[above] += 1
                cells[row] = n_cells[branching]
                first[row] = branching
                members[row] = 0
                if not in_clade[branching] and size[branching] > 1:
                    n_heap = _push_row(heap, n_heap, row, cells, first)
            else:
                members[row] = count
                cells[row] -= n_cells[largest]
                first[row] = pool[begin]
                n_heap = _push_row(heap, n_heap, row, cells, first)
        wedges += 1
    # A group of sibling clades comes after the clades drawn alone.
    for row in range(n_rows):
        if kind[row] == _SIBLINGS:
            ranks[row] = revealed[parent[row]]
    return (
        kind[:n_rows],
        node[:n_rows],
        top[:n_rows],
        parent[:n_rows],
        ranks[:n_rows],
        opened[:n_rows],
        start[:n_rows],
        members[:n_rows],
        pool,
    )


@njit(cache=True)
def _row_values(
    kind: np.ndarray,
    node: np.ndarray,
    top: np.ndarray,
    parent: np.ndarray,
    start: np.ndarray,
    members: np.ndarray,
    pool: np.ndarray,
    n_cells: np.ndarray,
    n_clones: np.ndarray,
    clone: np.ndarray,
    x: np.ndarray,
    reach: np.ndarray,
) -> tuple[np.ndarray, ...]:
    n_rows = len(kind)
    cells_of = np.zeros(n_rows, dtype=np.int64)
    clones_of = np.zeros(n_rows, dtype=np.int64)
    clone_of = np.full(n_rows, -1, dtype=np.int64)
    clades = np.ones(n_rows, dtype=np.int64)
    trees = np.ones(n_rows, dtype=np.int64)
    x_of = np.zeros(n_rows, dtype=np.float64)
    end_of = np.zeros(n_rows, dtype=np.float64)
    start_of = np.zeros(n_rows, dtype=np.float64)
    for row in range(n_rows):
        if kind[row] == _NODE:
            here = node[row]
            cells_of[row] = n_cells[here]
            clones_of[row] = n_clones[here]
            if n_clones[here] == 1:
                clone_of[row] = clone[here]
            x_of[row] = x[here]
            end_of[row] = reach[here]
            continue
        # A group holds its clades' cells and clones, from the earliest of
        # them to the latest leaf of any.
        total = 0
        clones = 0
        code = -1
        earliest = np.inf
        latest = -np.inf
        for at in range(start[row], start[row] + members[row]):
            member = pool[at]
            total += n_cells[member]
            clones += n_clones[member]
            code = max(code, clone[member])
            earliest = min(earliest, x[member])
            latest = max(latest, reach[member])
        cells_of[row] = total
        clones_of[row] = clones
        if clones == 1:
            clone_of[row] = code
        clades[row] = members[row]
        if kind[row] == _TREES:
            trees[row] = members[row]
        x_of[row] = earliest
        end_of[row] = latest
    for row in range(n_rows):
        # A branch runs from the node opened above, or a tree's from its root.
        if parent[row] >= 0:
            start_of[row] = x[node[parent[row]]]
        elif kind[row] == _NODE:
            start_of[row] = x[top[row]]
        else:
            start_of[row] = x_of[row]
    return cells_of, clones_of, clone_of, clades, trees, x_of, start_of, end_of


@njit(cache=True)
def _place(
    parent: np.ndarray, ranks: np.ndarray, wedge: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(parent)
    # Each row's children in their ranks, the trees' under a last slot.
    offsets = np.zeros(n_rows + 2, dtype=np.int64)
    for row in range(n_rows):
        offsets[(parent[row] if parent[row] >= 0 else n_rows) + 1] += 1
    for slot in range(n_rows + 1):
        offsets[slot + 1] += offsets[slot]
    children = np.empty(n_rows, dtype=np.int64)
    for row in range(n_rows):
        slot = parent[row] if parent[row] >= 0 else n_rows
        children[offsets[slot] + ranks[row]] = row
    # The rows in drawing order, each before its children, by a stack.
    order = np.empty(n_rows, dtype=np.int64)
    stack = np.empty(n_rows, dtype=np.int64)
    n_stack = 0
    for at in range(offsets[n_rows + 1] - 1, offsets[n_rows] - 1, -1):
        stack[n_stack] = children[at]
        n_stack += 1
    n_order = 0
    while n_stack > 0:
        n_stack -= 1
        row = stack[n_stack]
        order[n_order] = row
        n_order += 1
        for at in range(offsets[row + 1] - 1, offsets[row] - 1, -1):
            stack[n_stack] = children[at]
            n_stack += 1
    # Each wedge takes a share of the height as large as its cells, from the
    # top; a node opened stands between its first and last children.
    low = np.zeros(n_rows, dtype=np.float64)
    high = np.zeros(n_rows, dtype=np.float64)
    y = np.zeros(n_rows, dtype=np.float64)
    cursor = 0.0
    for at in range(n_rows):
        row = order[at]
        if wedge[row]:
            low[row] = cursor
            cursor += cells[row]
            high[row] = cursor
            y[row] = (low[row] + high[row]) / 2
    for at in range(n_rows - 1, -1, -1):
        row = order[at]
        if not wedge[row]:
            above = children[offsets[row]]
            below = children[offsets[row + 1] - 1]
            low[row] = low[above]
            high[row] = high[below]
            y[row] = (y[above] + y[below]) / 2
    return order, low, high, y


@njit(cache=True)
def _drawn_positions(
    order: np.ndarray,
    kind: np.ndarray,
    node: np.ndarray,
    parent: np.ndarray,
    tree_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each row's tree, node in its tree and parent row, in drawing order."""
    n_rows = len(order)
    drawn_at = np.empty(n_rows, dtype=np.int64)
    for at in range(n_rows):
        drawn_at[order[at]] = at
    trees = np.full(n_rows, -1, dtype=np.int64)
    nodes = np.full(n_rows, -1, dtype=np.int64)
    parents = np.full(n_rows, -1, dtype=np.int64)
    for at in range(n_rows):
        row = order[at]
        if parent[row] >= 0:
            parents[at] = drawn_at[parent[row]]
        if kind[row] == _TREES:
            continue
        # A group of sibling clades is in the tree of the node opened.
        tree = np.searchsorted(tree_offsets, node[row], side="right") - 1
        trees[at] = tree
        if kind[row] == _NODE:
            nodes[at] = node[row] - tree_offsets[tree]
    return trees, nodes, parents


def _ledger_of(ledger: MemoryLedger | None) -> MemoryLedger:
    if ledger is None:
        return MemoryLedger(None, "Collapsing a phylogeny")
    if not isinstance(ledger, MemoryLedger):
        raise TypeError("ledger must be a MemoryLedger or None")
    return ledger


def _given(
    ledger: MemoryLedger, values: Any, dtype: type, item: str
) -> tuple[np.ndarray, int]:
    """``values`` as an array of ``dtype``, and the bytes charged for a copy where one is made."""
    if isinstance(values, np.ndarray) and values.dtype == dtype:
        return values, 0
    copied = np.asarray(values)
    charged = ledger.hold(item, copied.size * np.dtype(dtype).itemsize)
    return copied.astype(dtype), charged


def collapse_phylogeny(
    parent: np.ndarray,
    times: np.ndarray | None,
    cells: np.ndarray,
    clone_codes: np.ndarray,
    tree_offsets: np.ndarray,
    *,
    cut: PhylogenyCut,
    max_leaves: int,
    ledger: MemoryLedger | None = None,
) -> pd.DataFrame:
    """Collapse cut phylogenies to at most ``max_leaves`` wedges, placed for a figure.

    Each tree starts as one wedge, and the clade that holds the most cells
    opens until ``max_leaves`` wedges are drawn: its child of the most cells
    is drawn alone and its other children together, until they open in
    turn, so each step draws one more wedge and a polytomy opens one child
    at a time. A clade of the cut, one clone, never opens, nor does a leaf,
    so fewer wedges are drawn when the trees hold fewer clades. A forest of
    more trees than ``max_leaves`` draws its largest trees alone and the
    others as one wedge. A clade is drawn from its first branching, below
    any nodes of one child that holds cells, and leaves that name no cell
    are left out. Ties open the earlier node first.

    Wedges are stacked from the top, the trees and each node's children in
    the order they open, the most cells first and a group last, each wedge
    as tall as its cells. Three passes over the nodes, none recursive, find
    each node's cells and the clones cut below it from the last node back,
    open the clades through a heap, and place the rows.

    Each step is charged to ``ledger`` before it allocates, and what a step
    no longer needs is released once it goes; the table returned stays
    charged, at its ``memory_usage``.

    Args:
        parent: Each node's parent, -1 for a root; parents come first, and
            each tree's nodes start at its root.
        times: Each node's time, for a cut at a time; None for a depth.
        cells: The cell, as a position among ``clone_codes``, that each node
            names, -1 for none; only leaves name cells.
        clone_codes: Each cell's clone as a code of the cut, -1 for none.
        tree_offsets: Where each tree's nodes start, and the end.
        cut: Where the clones were cut.
        max_leaves: The most wedges drawn, at least 1.
        ledger: The caller's memory ledger, or None for no budget.

    Returns:
        One row per node or group drawn, in drawing order, each after the
        node opened above it: ``tree`` (its tree's position, -1 for a group
        of trees), ``node`` (its position in the tree, -1 for a group),
        ``parent`` (the row of the node opened above it, -1 for none),
        ``wedge`` (whether it is drawn collapsed), ``n_cells``, ``n_clones``
        (the clones cut in it), ``clone`` (the code of its one clone, -1
        for none or several), ``n_clades`` (the clades a group holds, 1
        otherwise), ``n_trees`` (the trees it holds), ``x`` (where it
        starts on the x axis: its node's time or depth, a group's earliest
        clade's), ``start`` (where the branch to it starts), ``end`` (its
        latest leaf), and ``y``, ``low`` and ``high``, its middle and extent
        in cells from the top.

    Raises:
        TypeError: If ``max_leaves`` is not an integer, or ``ledger`` not a
            MemoryLedger.
        ValueError: If ``max_leaves`` is below 1, the arrays disagree, a
            node has no time for a cut at a time, no leaf names a cell, an
            internal node names a cell, or the clones are not those of the
            cut: a cell in a clade without a clone or outside one with a
            clone, or two clones in one clade.
        MemoryBudgetError: If a step does not fit the ledger's budget.
    """
    max_leaves = integer_argument(max_leaves, "max_leaves", minimum=1)
    ledger = _ledger_of(ledger)
    parent, parent_bytes = _given(ledger, parent, np.int64, "each node's parent")
    cells, cell_bytes = _given(ledger, cells, np.int64, "each node's cell")
    codes, code_bytes = _given(ledger, clone_codes, np.int64, "each cell's clone")
    offsets = np.asarray(tree_offsets, dtype=np.int64)
    n_nodes = len(parent)
    if (
        len(cells) != n_nodes
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != n_nodes
    ):
        raise ValueError(
            f"The phylogeny's arrays disagree: {n_nodes} parents, {len(cells)} "
            f"cells and trees that end at {offsets[-1] if len(offsets) else None}"
        )
    roots = offsets[:-1]
    if (np.diff(offsets) <= 0).any() or (parent[roots] >= 0).any():
        raise ValueError(
            "Each tree of the phylogeny must hold a node and start at its root"
        )
    if n_nodes and int(cells.max()) >= len(codes):
        raise ValueError(
            f"A leaf names cell {int(cells.max())}, but the clones code {len(codes)} cells"
        )
    depth_bytes = 0
    if cut.time is not None:
        if times is None or len(times) != n_nodes:
            raise ValueError("A cut at a time needs each node's time")
        x, time_bytes = _given(ledger, times, np.float64, "each node's time")
        ledger.peak("checking each node's time", n_nodes + NUMPY_STEP_BYTES)
        if np.isnan(x).any():
            raise ValueError("A node of the phylogeny has no time; cut it at a depth")
        limit = cut.time
    elif cut.depth is not None:
        time_bytes = 0
        ledger.peak(
            "each node's depth", _DEPTH_PEAK_BYTES_PER_NODE * n_nodes + NUMPY_STEP_BYTES
        )
        x = collapsed_depths(parent).astype(np.float64)
        depth_bytes = ledger.hold("each node's depth", 8 * n_nodes)
        limit = float(cut.depth)
    else:
        raise ValueError("The cut needs a time or a depth")
    ledger.peak(
        "each node's clade", _CLADE_PEAK_BYTES_PER_NODE * n_nodes + NUMPY_STEP_BYTES
    )
    past = x >= limit
    in_clade = clade_roots(parent, past) >= 0
    del past
    clade_bytes = ledger.hold("each node's clade", n_nodes)
    count_bytes = ledger.hold(
        "each node's cells and clones", _COUNT_BYTES_PER_NODE * n_nodes
    )
    bad, size, n_cells, n_clones, clone, reach = _clade_counts(
        parent, cells, codes, in_clade, x
    )
    if bad >= 0:
        raise ValueError(
            f"The clones are not those of the cut at node {bad}: an internal node "
            "names a cell, a cell in a clade of the cut has no clone or one "
            "outside has, or a clade holds two clones"
        )
    total = int(n_cells[roots].sum())
    if not total:
        raise ValueError(
            "No leaf of the trees names a cell, so there is nothing to plot"
        )
    # Each step draws one more wedge and at most two more rows, and a wedge
    # holds a cell at least.
    capacity = 2 * min(max_leaves, total) + 2
    opening_bytes = ledger.hold(
        f"opening the clades to {max_leaves} wedges",
        _POOL_BYTES_PER_NODE * n_nodes + _OPENING_BYTES_PER_SLOT * capacity,
    )
    kind, node, top, row_parent, ranks, opened, start, members, pool = _collapse(
        size, n_cells, in_clade, roots, max_leaves, capacity
    )
    opening_bytes -= _OPENING_SCRATCH_PER_SLOT * capacity
    ledger.release(_OPENING_SCRATCH_PER_SLOT * capacity)
    n_rows = len(kind)
    value_bytes = ledger.hold(
        "each row's cells, clones and extent", _VALUE_BYTES_PER_ROW * n_rows
    )
    values = _row_values(
        kind,
        node,
        top,
        row_parent,
        start,
        members,
        pool,
        n_cells,
        n_clones,
        clone,
        x,
        reach,
    )
    row_cells, row_clones, row_clone, clades, trees, row_x, row_start, row_end = values
    del (
        values,
        pool,
        size,
        n_cells,
        n_clones,
        clone,
        reach,
        in_clade,
        top,
        start,
        members,
        x,
    )
    let_go = (
        count_bytes
        + clade_bytes
        + depth_bytes
        + _POOL_BYTES_PER_NODE * n_nodes
        + _OPENED_SCRATCH_PER_SLOT * capacity
    )
    opening_bytes -= (
        _POOL_BYTES_PER_NODE * n_nodes + _OPENED_SCRATCH_PER_SLOT * capacity
    )
    ledger.release(let_go)
    place_bytes = ledger.hold("each row's place", _PLACE_BYTES_PER_ROW * n_rows)
    ledger.peak("placing the rows", _PLACING_BYTES_PER_ROW * (n_rows + 1))
    wedge = ~opened
    order, low, high, y = _place(row_parent, ranks, wedge, row_cells)
    ordered_bytes = ledger.hold(
        "each row's tree, node and parent", _ORDERED_BYTES_PER_ROW * n_rows
    )
    ledger.peak("each row's tree, node and parent", 8 * n_rows)
    tree, position, above = _drawn_positions(order, kind, node, row_parent, offsets)
    del kind, node, row_parent, ranks, opened
    ledger.release(opening_bytes)
    # The columns in drawing order, then the table's blocks that copy them.
    ledger.peak(
        "the table of the rows", 2 * _TABLE_BYTES_PER_ROW * n_rows + NUMPY_STEP_BYTES
    )
    frame = pd.DataFrame(
        {
            "tree": tree,
            "node": position,
            "parent": above,
            "wedge": wedge[order],
            "n_cells": row_cells[order],
            "n_clones": row_clones[order],
            "clone": row_clone[order],
            "n_clades": clades[order],
            "n_trees": trees[order],
            "x": row_x[order],
            "start": row_start[order],
            "end": row_end[order],
            "y": y[order],
            "low": low[order],
            "high": high[order],
        },
        copy=False,
    )
    ledger.hold("the table of the rows", int(frame.memory_usage(index=True).sum()))
    del tree, position, above, wedge, order, low, high, y, parent, cells, codes
    del row_cells, row_clones, row_clone, clades, trees, row_x, row_start, row_end
    ledger.release(
        value_bytes
        + place_bytes
        + ordered_bytes
        + parent_bytes
        + cell_bytes
        + code_bytes
        + time_bytes
    )
    return frame


@dataclass(frozen=True, slots=True)
class CollapsedPhylogeny:
    """What a figure of clones cut from a phylogeny draws: the trees collapsed to a cap of wedges.

    Attributes:
        nodes: One row per node or group drawn, in drawing order, as
            :func:`collapse_phylogeny` returns it: ``tree`` holds each row's
            tree as a position and ``clone`` its one clone as a code, its rank
            by size among the cut's clones.
        tree_names: Each row's tree's name, None for a group of trees.
        clone_ids: Each row's one clone's label, None for none or several.
        clones: The clones drawn.
        phylogeny: The phylogeny they were cut from.
        cut: Where the clones were cut.
        time_unit: The unit of the trees' times, or None.
        tree: The one tree drawn, or None for every tree.
        n_trees: The trees drawn that hold cells.
        n_clones: The clones cut in the trees drawn.
        max_leaves: The most wedges the figure draws.
    """

    nodes: pd.DataFrame = field(repr=False)
    tree_names: np.ndarray = field(repr=False)
    clone_ids: np.ndarray = field(repr=False)
    clones: ArtifactRef
    phylogeny: ArtifactRef
    cut: PhylogenyCut
    time_unit: str | None
    tree: str | None
    n_trees: int
    n_clones: int
    max_leaves: int


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
