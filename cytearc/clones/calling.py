"""Clones from per-cell sets of heritable marks.

Cells that share heritable marks share an ancestor: expressed lineage
barcodes, the mutant calls of genotyped sites, rare recorder alleles. Each
cell's marks are a set of integer item codes, and clones are called from the
sets in steps that every source shares. A cell may also hold items unknown,
such as the sites that no read covered, which neither hold nor lack the
mark:

- identical sets: cells whose sets are equal, and whose unknown items are,
  within one group such as a mouse, form one node, so a node holds one cell
  or many;
- a Jaccard graph: two nodes of a group whose sets share an item are linked
  when the Jaccard index of their sets, the items they share over the items
  either holds that both know, exceeds a similarity, or reaches it when
  ``inclusive``. Each node meets only the nodes that share an item of its
  prefix, the items of its set that the fewest nodes hold, as many as a
  linked pair's overlap allows, so an item that most nodes hold, such as the
  barcode that correction merged many into, is followed almost never. With
  unknown items, the intersections come from the sparse product of the
  nodes-by-items matrix with its transpose, row by row. Either way, nodes
  that share no item are never compared;
- or links through any item that two nodes share, which joins the nodes of
  each item without listing every pair of them;
- pruning, as TREX does: edges whose two ends share no neighbour are removed
  when either end has another neighbour;
- doublets, as TREX finds them: nodes of one cell whose neighbours fall apart
  into two or more components once the node is removed, local cut vertices,
  are removed with their cells;
- connected components of what remains, which are the clones.

:func:`set_clones` runs the steps; each also stands alone. Nodes are numbered
by their first cell, components by their first node, and clones by size,
largest first, then by their first cell, so every result follows from the
sets and the order of the cells alone, whatever order the items arrived in
and however many threads run. Each step charges the bytes it allocates to
the caller's memory ledger before allocating them, which refuses them with a
:class:`~cytearc.utils.memory_budget.MemoryBudgetError` beyond its budget.
These functions never touch storage.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils.arguments import bool_argument, quoted_value
from ..utils.arrays import unique_inverse
from ..utils.digest import mix64
from ..utils.memory_budget import NUMPY_STEP_BYTES, MemoryLedger
from ..utils.numba import njit

__all__ = [
    "CALLING_METHODS",
    "CellSets",
    "Edges",
    "SetClones",
    "SetNodes",
    "cell_sets",
    "connected_components",
    "identical_sets",
    "jaccard_edges",
    "ambiguous_nodes",
    "local_cut_vertices",
    "pruned_edges",
    "set_clones",
]

type CallingMethod = Literal["sets", "jaccard", "shared"]

#: The ways of grouping sets into clones: identical sets alone, a Jaccard
#: graph over them, or links through any item two sets share.
CALLING_METHODS: tuple[str, ...] = ("sets", "jaccard", "shared")
# Bytes of each item of a set while sets are built from pairs: the order of
# the pairs, their sorted cells and items and the flags of new pairs, 19 to
# 25 as traced on 60,000 to 2,000,000 pairs; and while a cell's items and
# unknown items are joined, the pairs of both joined, their order and sorted
# copies.
_PAIR_BYTES = 28
_JOINED_PAIR_BYTES = 48
# Bytes of each cell of the sets built: its pointer, and its count of items
# while the pointers are summed.
_SET_CELL_BYTES = 16
# Bytes of each cell while identical sets are found: its two hashes, size and
# group with their sorted copies and order, the comparisons of neighbours in
# that order, its run, its node and their order, and its node's first cell,
# size, group and pointer, 167 as traced on 5,000 to 50,000 cells; and of
# each item of a node, the item and the checks of its order, 12 as traced.
_IDENTITY_CELL_BYTES = 180
_NODE_ITEM_BYTES = 16
# Bytes that the nodes hold once found: each cell's node, each node's first
# cell, size, group and pointer and its flags of whether it remains and
# whether it is ambiguous, and each of its items.
_HELD_CELL_BYTES = 8
_NODE_BYTES = 42
_HELD_ITEM_BYTES = 8
# Bytes of each item of a node while the Jaccard graph is built: the group of
# its entry, its code within the group, the transposed lists and their order;
# of each unknown item, the same without a place in the lists, and the marks
# of the codes; and of each node, its intersections, the nodes it touched and
# its size.
_GRAPH_ITEM_BYTES = 64
_GRAPH_UNKNOWN_BYTES = 72
_GRAPH_NODE_BYTES = 40
# Without unknown items, the graph is built through the nodes' prefixes. Bytes
# of each item of a node while the items are coded within their groups: the
# group of its entry, its code, the lists of each code's nodes and their
# order, 43 as traced on 50,000 to 190,000 nodes of 3 to 20 items; and of each
# node, the pointers that place its entries.
_ITEM_CODE_BYTES = 48
_ITEM_CODE_NODE_BYTES = 16
# A pair linked at an index of at least t shares at least t times the items
# of each of its nodes; the bound is taken this much lower, so that rounding
# the product never makes a prefix too short.
_PREFIX_MARGIN = 1e-9
# Bytes of each edge kept: its two ends; and while the graph is pruned or its
# cut vertices found, each edge twice in the lists of neighbours with their
# order, and each node's pointer, marks and stack.
_EDGE_BYTES = 16
_ADJACENCY_EDGE_BYTES = 80
_ADJACENCY_NODE_BYTES = 48
# Bytes of each node while components are found: its root, the roots kept,
# their order and numbering, and its component, 66 as traced; and of each
# cell, its doublet flag, component, clone and the masks that place them.
_COMPONENT_NODE_BYTES = 80
_CLONE_CELL_BYTES = 40
# Bytes of each component while the clones are numbered: its size, whether it
# is kept, its order and its rank.
_CLONE_COMPONENT_BYTES = 48
# Item codes with unknown items stay below this, so that their doubled codes
# fit 64 bits.
_MAX_UNKNOWN_CODE = 1 << 61


#: What a refusal of a calling step suggests beside a larger budget.
_REMEDY = "call clones within smaller groups"


def _ledger(ledger: MemoryLedger | None) -> MemoryLedger:
    """The caller's ledger, or one without a budget for a call without one."""
    if ledger is None:
        return MemoryLedger(None, "Calling clones from sets")
    if not isinstance(ledger, MemoryLedger):
        raise TypeError("ledger must be a MemoryLedger or None")
    return ledger


def _peak(ledger: MemoryLedger, item: str, nbytes: int) -> None:
    """Check that a step's ``nbytes``, and NumPy's objects and sort buffers, fit beside what the ledger holds."""
    ledger.peak(item, nbytes + NUMPY_STEP_BYTES, remedy=_REMEDY)


@dataclass(frozen=True, slots=True)
class CellSets:
    """Each cell's set of item codes, as the rows of a sparse cells-by-items matrix.

    Attributes:
        indptr: Where each cell's items start and end, ``n_cells + 1``
            int64 offsets.
        items: The item codes of every cell in turn, int64, ascending and
            distinct within each cell.
    """

    indptr: np.ndarray
    items: np.ndarray

    def __post_init__(self) -> None:
        indptr = np.asarray(self.indptr)
        items = np.asarray(self.items)
        if indptr.ndim != 1 or len(indptr) < 1 or indptr.dtype.kind not in "iu":
            raise ValueError("indptr must hold n_cells + 1 integer offsets")
        if items.ndim != 1 or (len(items) and items.dtype.kind not in "iu"):
            raise ValueError("items must be one-dimensional integer codes")
        if indptr[0] != 0 or indptr[-1] != len(items) or np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must rise from 0 to the number of items")
        if len(items) and items.min() < 0:
            raise ValueError("item codes must be at least 0")
        # A row starts at each pointer; within a row, items must rise.
        starts = np.zeros(len(items) + 1, dtype=bool)
        starts[indptr[:-1]] = True
        if len(items) > 1 and np.any(
            ~starts[1 : len(items)] & (items[1:] <= items[:-1])
        ):
            raise ValueError("each cell's items must be ascending and distinct")

    @property
    def n_cells(self) -> int:
        return len(self.indptr) - 1

    @property
    def nbytes(self) -> int:
        """The bytes of the sets' pointers and items."""
        return int(np.asarray(self.indptr).nbytes + np.asarray(self.items).nbytes)

    @property
    def sizes(self) -> np.ndarray:
        """The items of each cell, as int64."""
        return np.diff(np.asarray(self.indptr, dtype=np.int64))


def cell_sets(
    cells: np.ndarray,
    items: np.ndarray,
    n_cells: int,
    *,
    ledger: MemoryLedger | None = None,
) -> CellSets:
    """Build each cell's set from pairs of a cell and an item, repeats merged.

    Args:
        cells: Each pair's cell, a position below ``n_cells``.
        items: Each pair's item code, at least 0.
        n_cells: The cells, including those without an item.
        ledger: The caller's memory ledger, which the build is charged to;
            None for no limit.

    Raises:
        ValueError: If the arrays differ in length or hold a cell or item out
            of range.
        MemoryBudgetError: If the build does not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    given_cells = np.asarray(cells).reshape(-1)
    given_items = np.asarray(items).reshape(-1)
    if len(given_cells) != len(given_items):
        raise ValueError("Each pair needs one cell and one item")
    # The pairs become int64 where they are not, and the sets keep each
    # cell's pointer beside the count of its items.
    copies = sum(
        8 * len(codes)
        for codes in (given_cells, given_items)
        if codes.dtype != np.int64
    )
    _peak(
        ledger,
        "the sets of the cells",
        _PAIR_BYTES * len(given_cells) + copies + _SET_CELL_BYTES * (n_cells + 1),
    )
    cell_codes = given_cells.astype(np.int64, copy=False)
    item_codes = given_items.astype(np.int64, copy=False)
    del given_cells, given_items
    if len(cell_codes) and (cell_codes.min() < 0 or cell_codes.max() >= n_cells):
        raise ValueError(f"cells must be positions below {n_cells}")
    if len(item_codes) and item_codes.min() < 0:
        raise ValueError("item codes must be at least 0")
    order = np.lexsort((item_codes, cell_codes))
    cell_codes = cell_codes[order]
    item_codes = item_codes[order]
    del order
    new = np.ones(len(cell_codes), dtype=bool)
    new[1:] = (cell_codes[1:] != cell_codes[:-1]) | (item_codes[1:] != item_codes[:-1])
    cell_codes = cell_codes[new]
    item_codes = item_codes[new]
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell_codes, minlength=n_cells), out=indptr[1:])
    return CellSets(indptr=indptr, items=item_codes)


@dataclass(frozen=True, slots=True)
class SetNodes:
    """The distinct sets of each group: the nodes of the cells' graph.

    Attributes:
        cell_nodes: Each cell's node, -1 for a cell without an item.
        first_cells: Each node's first cell; nodes are numbered in its order.
        cell_counts: The cells of each node.
        groups: Each node's group.
        sets: Each node's set.
        unknown: Each node's unknown items, or None when the cells had none
            given.
    """

    cell_nodes: np.ndarray
    first_cells: np.ndarray
    cell_counts: np.ndarray
    groups: np.ndarray
    sets: CellSets
    unknown: CellSets | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.first_cells)


@njit(cache=True)
def _set_hashes(indptr: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two 64-bit hashes of each set, folded over its items in order."""
    n_sets = len(indptr) - 1
    first = np.empty(n_sets, dtype=np.uint64)
    second = np.empty(n_sets, dtype=np.uint64)
    for row in range(n_sets):
        low = np.uint64(0x243F6A8885A308D3)
        high = np.uint64(0x13198A2E03707344)
        for at in range(indptr[row], indptr[row + 1]):
            item = np.uint64(items[at])
            low = mix64(low ^ (item + np.uint64(0x9E3779B97F4A7C15)))
            high = mix64((high + item) * np.uint64(0xD6E8FEB86659FD93))
        first[row] = low
        second[row] = high
    return first, second


@njit(cache=True)
def _equal_sets(indptr: np.ndarray, items: np.ndarray, first: int, second: int) -> bool:
    """Whether two sets of one size hold the same items, as runs of equal keys do."""
    start = indptr[first]
    other = indptr[second]
    for at in range(indptr[first + 1] - start):
        if items[start + at] != items[other + at]:
            return False
    return True


@njit(cache=True)
def _verified_runs(
    order: np.ndarray, starts: np.ndarray, indptr: np.ndarray, items: np.ndarray
) -> np.ndarray:
    """Whether every cell of each run of equal hashes holds the run's first set."""
    n_runs = len(starts) - 1
    verified = np.ones(n_runs, dtype=np.bool_)
    for run in range(n_runs):
        head = order[starts[run]]
        for at in range(starts[run] + 1, starts[run + 1]):
            if not _equal_sets(indptr, items, head, order[at]):
                verified[run] = False
                break
    return verified


def _unknown_sets(sets: CellSets, unknown: CellSets, ledger: MemoryLedger) -> CellSets:
    """Each cell's items and unknown items as one set, of even and odd codes.

    An item ``i`` that the cell holds is coded ``2 * i`` and one unknown
    ``2 * i + 1``, so two cells hold equal sets of these codes exactly when
    their items and their unknown items are equal.
    """
    n_cells = sets.n_cells
    if unknown.n_cells != n_cells:
        raise ValueError("unknown must hold one set per cell of sets")
    n_held = len(sets.items)
    n_unknown = len(unknown.items)
    for name, codes in (("item", sets.items), ("unknown item", unknown.items)):
        if len(codes) and int(np.max(codes)) >= _MAX_UNKNOWN_CODE:
            raise ValueError(
                f"{name} codes must stay below {_MAX_UNKNOWN_CODE} with unknown items"
            )
    _peak(
        ledger,
        "the cells' items and unknown items",
        _JOINED_PAIR_BYTES * (n_held + n_unknown) + _SET_CELL_BYTES * (n_cells + 1),
    )
    cells = np.concatenate(
        [
            np.repeat(np.arange(n_cells, dtype=np.int64), sets.sizes),
            np.repeat(np.arange(n_cells, dtype=np.int64), unknown.sizes),
        ]
    )
    codes = np.concatenate(
        [
            np.asarray(sets.items, dtype=np.int64) * 2,
            np.asarray(unknown.items, dtype=np.int64) * 2 + 1,
        ]
    )
    order = np.lexsort((codes, cells))
    cells = cells[order]
    codes = codes[order]
    del order
    if len(codes) > 1 and np.any(
        (cells[1:] == cells[:-1]) & (codes[1:] >> 1 == codes[:-1] >> 1)
    ):
        raise ValueError("a cell cannot both hold an item and hold it unknown")
    indptr = np.asarray(sets.indptr, dtype=np.int64) + np.asarray(
        unknown.indptr, dtype=np.int64
    )
    return CellSets(indptr=indptr, items=codes)


def identical_sets(
    sets: CellSets,
    groups: np.ndarray | None = None,
    *,
    unknown: CellSets | None = None,
    ledger: MemoryLedger | None = None,
) -> SetNodes:
    """Group the cells of each group whose sets are equal into nodes.

    Each set is hashed twice in 64 bits, the cells are sorted by group, size
    and hashes, and every cell of a run of equal keys is compared item by
    item with the run's first, so two sets share a node only when they are
    equal; colliding hashes are split apart. With ``unknown``, two cells
    share a node only when their unknown items are equal too. A cell without
    an item joins no node, whatever it holds unknown.

    Args:
        sets: Each cell's items.
        groups: Each cell's group as an integer code, such as its mouse; sets
            of two groups never share a node. None puts every cell in one.
        unknown: Each cell's unknown items, such as the sites that no read
            covered, none of them among its items; None for none.
        ledger: The caller's memory ledger, which the grouping is charged to;
            None for no limit.

    Raises:
        ValueError: If ``groups`` does not hold one integer code per cell, or
            ``unknown`` one set per cell without the cell's items.
        MemoryBudgetError: If the grouping does not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    n_cells = sets.n_cells
    # Sets of other integer types are copied as int64, and the joined sets of
    # items and unknown items are held while the nodes are found.
    copied = 0
    if (
        np.asarray(sets.indptr).dtype != np.int64
        or np.asarray(sets.items).dtype != np.int64
    ):
        copied = 8 * (len(sets.items) + n_cells + 1)
    keyed = sets
    if unknown is not None:
        keyed = _unknown_sets(sets, unknown, ledger)
        copied = 8 * (len(keyed.items) + n_cells + 1)
    _peak(ledger, "the sets' hashes", _IDENTITY_CELL_BYTES * n_cells + copied)
    group_codes = _group_codes(groups, n_cells)
    held_sizes = sets.sizes
    sizes = keyed.sizes
    indptr = np.asarray(keyed.indptr, dtype=np.int64)
    items = np.asarray(keyed.items, dtype=np.int64)
    first_hash, second_hash = _set_hashes(indptr, items)
    present = np.flatnonzero(held_sizes > 0)
    order = present[
        np.lexsort(
            (
                second_hash[present],
                first_hash[present],
                sizes[present],
                group_codes[present],
            )
        )
    ]
    keys = (group_codes, sizes, first_hash, second_hash)
    new = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        new[1:] = np.any([key[order[1:]] != key[order[:-1]] for key in keys], axis=0)
    del first_hash, second_hash
    starts = np.append(np.flatnonzero(new), len(order)).astype(np.int64)
    verified = _verified_runs(order, starts, indptr, items)
    run_of = np.cumsum(new) - 1
    # Each run is a node unless two of its sets differ, which only colliding
    # hashes give; such a run is split by its sets, compared whole.
    node_keys = run_of.astype(np.int64)
    if not np.all(verified):
        node_keys = _split_collisions(node_keys, order, starts, verified, indptr, items)
    # Within a run the cells keep their order, so a node's first member is
    # its first cell; nodes are numbered in the order of their first cells.
    cell_nodes = np.full(n_cells, -1, dtype=np.int64)
    distinct, inverse = unique_inverse(node_keys)
    firsts = np.full(len(distinct), n_cells, dtype=np.int64)
    np.minimum.at(firsts, inverse, order)
    rank = np.empty(len(distinct), dtype=np.int64)
    by_first = np.argsort(firsts, kind="stable")
    rank[by_first] = np.arange(len(distinct))
    cell_nodes[order] = rank[inverse]
    first_cells = firsts[by_first]
    n_nodes = len(first_cells)
    _peak(
        ledger,
        "the nodes' sets",
        _IDENTITY_CELL_BYTES * n_cells
        + copied
        + _NODE_ITEM_BYTES * int(sizes[first_cells].sum()),
    )
    del indptr, items, keyed
    node_sets = _node_rows(sets, first_cells)
    node_unknown = None if unknown is None else _node_rows(unknown, first_cells)
    return SetNodes(
        cell_nodes=cell_nodes,
        first_cells=first_cells,
        cell_counts=np.bincount(cell_nodes[cell_nodes >= 0], minlength=n_nodes).astype(
            np.int64
        ),
        groups=group_codes[first_cells],
        sets=node_sets,
        unknown=node_unknown,
    )


def _node_rows(sets: CellSets, first_cells: np.ndarray) -> CellSets:
    """The rows of each node's first cell, as int64."""
    indptr = np.asarray(sets.indptr, dtype=np.int64)
    sizes = indptr[first_cells + 1] - indptr[first_cells]
    node_indptr = np.zeros(len(first_cells) + 1, dtype=np.int64)
    np.cumsum(sizes, out=node_indptr[1:])
    items = _gathered_rows(
        indptr, np.asarray(sets.items, dtype=np.int64), first_cells, node_indptr
    )
    return CellSets(indptr=node_indptr, items=items)


def _split_collisions(
    node_keys: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    verified: np.ndarray,
    indptr: np.ndarray,
    items: np.ndarray,
) -> np.ndarray:
    """Give the distinct sets of each run whose hashes collided keys of their own."""
    keys = node_keys.copy()
    next_key = int(keys.max()) + 1 if len(keys) else 0
    for run in np.flatnonzero(~verified).tolist():
        seen: dict[tuple[int, ...], int] = {}
        for at in range(int(starts[run]), int(starts[run + 1])):
            cell = int(order[at])
            members = tuple(items[indptr[cell] : indptr[cell + 1]].tolist())
            if not seen:
                seen[members] = int(keys[at])
            elif members not in seen:
                seen[members] = next_key
                next_key += 1
            keys[at] = seen[members]
    return keys


@njit(cache=True)
def _gathered_rows(
    indptr: np.ndarray, items: np.ndarray, rows: np.ndarray, out_indptr: np.ndarray
) -> np.ndarray:
    gathered = np.empty(out_indptr[-1], dtype=np.int64)
    for at in range(len(rows)):
        row = rows[at]
        start = indptr[row]
        length = indptr[row + 1] - start
        target = out_indptr[at]
        for offset in range(length):
            gathered[target + offset] = items[start + offset]
    return gathered


def _group_codes(groups: np.ndarray | None, n_cells: int) -> np.ndarray:
    if groups is None:
        return np.zeros(n_cells, dtype=np.int64)
    codes = np.asarray(groups)
    if codes.shape != (n_cells,) or (len(codes) and codes.dtype.kind not in "iu"):
        raise ValueError("groups must hold one integer code per cell")
    if len(codes) and codes.min() < 0:
        raise ValueError("group codes must be at least 0")
    return codes.astype(np.int64)


@dataclass(frozen=True, slots=True)
class Edges:
    """The edges of a graph of nodes, each once, ordered by their two ends.

    Attributes:
        first: Each edge's lower end, int64.
        second: Each edge's higher end, int64.
    """

    first: np.ndarray
    second: np.ndarray

    def __len__(self) -> int:
        return len(self.first)

    def kept(self, keep: np.ndarray) -> "Edges":
        return Edges(first=self.first[keep], second=self.second[keep])

    @property
    def nbytes(self) -> int:
        return int(self.first.nbytes + self.second.nbytes)


@njit(cache=True)
def _jaccard_pairs(
    node_indptr: np.ndarray,
    node_items: np.ndarray,
    item_indptr: np.ndarray,
    item_nodes: np.ndarray,
    sizes: np.ndarray,
    unknown_indptr: np.ndarray,
    unknown_items: np.ndarray,
    within: float,
    across: float,
    modalities: np.ndarray,
    inclusive: bool,
    first: np.ndarray,
    second: np.ndarray,
    fill: bool,
) -> int:
    """Count, or fill, the pairs of nodes linked by their Jaccard index.

    Each node's intersections with the higher nodes that share an item are
    accumulated through the items' lists of nodes, the row of the product
    of the nodes-by-items matrix with its transpose; with ``fill`` the pairs
    are written in order of their two ends. With unknown items, the items
    that one node of a pair holds and the other holds unknown are left out
    of the items either holds.
    """
    n_nodes = len(node_indptr) - 1
    n_items = len(item_indptr) - 1
    shared = np.zeros(n_nodes, dtype=np.int64)
    touched = np.empty(n_nodes, dtype=np.int64)
    with_unknown = len(unknown_items) > 0
    # The node that last marked each item as held, and as unknown.
    held_by = np.full(n_items if with_unknown else 0, -1, dtype=np.int64)
    unknown_to = np.full(n_items if with_unknown else 0, -1, dtype=np.int64)
    count = 0
    for node in range(n_nodes):
        n_touched = 0
        for at in range(node_indptr[node], node_indptr[node + 1]):
            item = node_items[at]
            for position in range(item_indptr[item], item_indptr[item + 1]):
                other = item_nodes[position]
                if other <= node:
                    continue
                if shared[other] == 0:
                    touched[n_touched] = other
                    n_touched += 1
                shared[other] += 1
        if fill and n_touched > 1:
            touched[:n_touched] = np.sort(touched[:n_touched])
        if with_unknown:
            for at in range(node_indptr[node], node_indptr[node + 1]):
                held_by[node_items[at]] = node
            for at in range(unknown_indptr[node], unknown_indptr[node + 1]):
                unknown_to[unknown_items[at]] = node
        for at in range(n_touched):
            other = touched[at]
            common = shared[other]
            shared[other] = 0
            # The items of either that the other holds unknown.
            uncovered = 0
            if with_unknown:
                for position in range(node_indptr[other], node_indptr[other + 1]):
                    if unknown_to[node_items[position]] == node:
                        uncovered += 1
                for position in range(unknown_indptr[other], unknown_indptr[other + 1]):
                    if held_by[unknown_items[position]] == node:
                        uncovered += 1
            index = common / (sizes[node] + sizes[other] - common - uncovered)
            same = modalities[node] >= 0 and modalities[node] == modalities[other]
            limit = within if same else across
            linked = index >= limit if inclusive else index > limit
            if linked:
                if fill:
                    first[count] = node
                    second[count] = other
                count += 1
    return count


@njit(cache=True)
def _prefix_length(size: int, lowest: float) -> int:
    """The items of a node of ``size`` items that its prefix holds.

    A pair linked at an index of at least ``lowest`` shares at least
    ``lowest`` times the items of each of its nodes, and one item at least.
    A node that shares ``overlap`` items with another holds at most
    ``size - overlap`` items before the first of them, in any order of the
    items, so its first ``size - overlap + 1`` items in that order hold the
    first item that it shares with any node it is linked to.
    """
    overlap = max(int(np.ceil((lowest - _PREFIX_MARGIN) * size)), 1)
    return min(max(size - overlap + 1, 0), size)


@njit(cache=True)
def _prefix_shape(
    indptr: np.ndarray, item_indptr: np.ndarray, lowest: float
) -> tuple[int, int, int]:
    """The items of every node's prefix, the most items of a node, and the most nodes of an item."""
    n_prefix = 0
    largest = 0
    for node in range(len(indptr) - 1):
        size = indptr[node + 1] - indptr[node]
        n_prefix += _prefix_length(size, lowest)
        largest = max(largest, size)
    most = 0
    for code in range(len(item_indptr) - 1):
        most = max(most, item_indptr[code + 1] - item_indptr[code])
    return n_prefix, largest, most


@njit(cache=True)
def _item_ranks(item_indptr: np.ndarray, most: int) -> np.ndarray:
    """Each item code's rank by how many nodes hold it, then by code.

    A counting sort by the nodes of each code, which ``most`` bounds, keeps
    codes held by equally many nodes in the order of their codes.
    """
    n_codes = len(item_indptr) - 1
    starts = np.zeros(most + 2, dtype=np.int64)
    for code in range(n_codes):
        starts[item_indptr[code + 1] - item_indptr[code] + 1] += 1
    for held in range(most + 1):
        starts[held + 1] += starts[held]
    ranks = np.empty(n_codes, dtype=np.int64)
    for code in range(n_codes):
        held = item_indptr[code + 1] - item_indptr[code]
        ranks[code] = starts[held]
        starts[held] += 1
    return ranks


@njit(cache=True)
def _set_prefixes(
    indptr: np.ndarray,
    items: np.ndarray,
    ranks: np.ndarray,
    lowest: float,
    n_prefix: int,
    largest: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each node's prefix, and the nodes whose prefix holds each item code.

    A node's prefix is its items of lowest rank, as many as
    :func:`_prefix_length` gives for its size. ``n_prefix`` is the items of
    every prefix and ``largest`` the most items of a node.

    Returns:
        Whether each item of each node lies in its prefix, and the pointers
        and nodes of each code's list of the nodes whose prefix holds it,
        ascending.
    """
    n_nodes = len(indptr) - 1
    n_codes = len(ranks)
    in_prefix = np.zeros(len(items), dtype=np.bool_)
    scratch = np.empty(largest, dtype=np.int64)
    for node in range(n_nodes):
        start = indptr[node]
        size = indptr[node + 1] - start
        length = _prefix_length(size, lowest)
        if length == size:
            for at in range(start, start + size):
                in_prefix[at] = True
            continue
        for at in range(size):
            scratch[at] = ranks[items[start + at]]
        # The prefix holds the items up to the rank of its last, the
        # length-th lowest of the node's distinct ranks.
        if size <= 16:
            for at in range(1, size):
                value = scratch[at]
                place = at
                while place > 0 and scratch[place - 1] > value:
                    scratch[place] = scratch[place - 1]
                    place -= 1
                scratch[place] = value
            last = scratch[length - 1]
        else:
            last = np.sort(scratch[:size])[length - 1]
        for at in range(start, start + size):
            if ranks[items[at]] <= last:
                in_prefix[at] = True
    prefix_indptr = np.zeros(n_codes + 1, dtype=np.int64)
    for at in range(len(items)):
        if in_prefix[at]:
            prefix_indptr[items[at] + 1] += 1
    for code in range(n_codes):
        prefix_indptr[code + 1] += prefix_indptr[code]
    prefix_nodes = np.empty(n_prefix, dtype=np.int64)
    filled = prefix_indptr[:-1].copy()
    for node in range(n_nodes):
        for at in range(indptr[node], indptr[node + 1]):
            if in_prefix[at]:
                code = items[at]
                prefix_nodes[filled[code]] = node
                filled[code] += 1
    return in_prefix, prefix_indptr, prefix_nodes


@njit(cache=True)
def _prefix_jaccard_pairs(
    node_indptr: np.ndarray,
    node_items: np.ndarray,
    in_prefix: np.ndarray,
    prefix_indptr: np.ndarray,
    prefix_nodes: np.ndarray,
    within: float,
    across: float,
    modalities: np.ndarray,
    inclusive: bool,
    first: np.ndarray,
    second: np.ndarray,
    fill: bool,
) -> tuple[int, int]:
    """Count, or fill, the pairs of nodes linked by their Jaccard index, met through their prefixes.

    Two linked nodes share an item of both their prefixes, the first item
    they share in the order of the ranks, so each node meets every higher
    node it may be linked to through the lists of its own prefix's items.
    Each pair met is compared over the two whole sets, whose ascending codes
    are merged, and linked as :func:`_jaccard_pairs` links it; with ``fill``
    the pairs are written in order of their two ends.

    Returns:
        The pairs linked, and the most nodes that one node met.
    """
    n_nodes = len(node_indptr) - 1
    seen = np.full(n_nodes, -1, dtype=np.int64)
    touched = np.empty(n_nodes, dtype=np.int64)
    count = 0
    most = 0
    for node in range(n_nodes):
        n_touched = 0
        start = node_indptr[node]
        end = node_indptr[node + 1]
        for at in range(start, end):
            if not in_prefix[at]:
                continue
            code = node_items[at]
            for position in range(prefix_indptr[code], prefix_indptr[code + 1]):
                other = prefix_nodes[position]
                if other <= node or seen[other] == node:
                    continue
                seen[other] = node
                touched[n_touched] = other
                n_touched += 1
        most = max(most, n_touched)
        if fill and n_touched > 1:
            touched[:n_touched] = np.sort(touched[:n_touched])
        for at in range(n_touched):
            other = touched[at]
            other_start = node_indptr[other]
            other_end = node_indptr[other + 1]
            # The items both hold, by merging their ascending codes.
            left = start
            right = other_start
            common = 0
            while left < end and right < other_end:
                if node_items[left] == node_items[right]:
                    common += 1
                    left += 1
                    right += 1
                elif node_items[left] < node_items[right]:
                    left += 1
                else:
                    right += 1
            index = common / ((end - start) + (other_end - other_start) - common)
            same = modalities[node] >= 0 and modalities[node] == modalities[other]
            limit = within if same else across
            linked = index >= limit if inclusive else index > limit
            if linked:
                if fill:
                    first[count] = node
                    second[count] = other
                count += 1
    return count, most


def _similarity(value: float, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a number from 0 to 1")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return number


def jaccard_edges(
    nodes: SetNodes,
    similarity: float,
    *,
    inclusive: bool,
    across: float | None = None,
    modalities: np.ndarray | None = None,
    ledger: MemoryLedger | None = None,
) -> Edges:
    """Link the nodes of each group whose sets are similar enough.

    Two nodes of a group that share an item are linked when the Jaccard
    index of their sets, computed as the float division of the items they
    share by the items either holds, exceeds ``similarity``, or reaches it
    with ``inclusive``. Nodes that share no item are never linked, so a
    similarity of 0 links every pair that shares an item, and of 1 none,
    since equal sets share a node already. With the nodes' unknown items,
    an item that one node holds and the other holds unknown counts in
    neither, so the index compares the items both know, and a similarity
    of 1 with ``inclusive`` links the nodes that agree wherever both know.

    Without unknown items, each node meets only the nodes that share an item
    of its prefix: its items that the fewest nodes of the group hold, as
    many as the lower of the two similarities allows a linked pair to leave
    unshared, the prefix filtering of exact set-similarity joins. Every pair
    met is compared over the two whole sets, so the edges are those of every
    pair that shares an item, in the same order. With unknown items, the
    index's denominator leaves out items unknown to the other node, so that
    bound does not hold, and every node meets every node that shares an item.

    Args:
        nodes: The nodes of :func:`identical_sets`.
        similarity: The index two nodes must pass, from 0 to 1.
        inclusive: Link at an index equal to the similarity too.
        across: Another similarity, for nodes of different modalities, such
            as the scRNA-seq and scATAC-seq cells of CellTag-multi; None uses
            ``similarity`` for every pair.
        modalities: With ``across``, each node's modality as an integer
            code, -1 for a node whose cells come from several: ``similarity``
            applies between two nodes of one and the same modality, and
            ``across`` between any others.
        ledger: The caller's memory ledger, which the graph is charged to;
            None for no limit.

    Raises:
        ValueError: If a similarity lies outside 0 to 1, or ``modalities``
            is missing or does not hold one code per node.
        MemoryBudgetError: If the graph does not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    within = _similarity(similarity, "similarity")
    inclusive = bool_argument(inclusive, "inclusive")
    n_nodes = nodes.n_nodes
    # Each node's modality as int64, held while the graph is built.
    _peak(ledger, "the nodes' modalities", 8 * n_nodes)
    if across is None:
        if modalities is not None:
            raise ValueError("modalities apply only with a similarity across them")
        other = within
        node_modalities = np.zeros(n_nodes, dtype=np.int64)
    else:
        other = _similarity(across, "across")
        if modalities is None:
            raise ValueError(
                "A similarity across modalities needs each node's modality"
            )
        node_modalities = np.asarray(modalities)
        if node_modalities.shape != (n_nodes,) or (
            n_nodes and node_modalities.dtype.kind not in "iu"
        ):
            raise ValueError("modalities must hold one integer code per node")
        node_modalities = node_modalities.astype(np.int64)
    indptr = np.asarray(nodes.sets.indptr, dtype=np.int64)
    n_items = len(nodes.sets.items)
    if nodes.unknown is None or not len(nodes.unknown.items):
        return _prefix_edges(
            nodes, indptr, within, other, node_modalities, bool(inclusive), ledger
        )
    unknown_indptr = np.asarray(nodes.unknown.indptr, dtype=np.int64)
    n_unknown = len(nodes.unknown.items)
    graph_bytes = (
        _GRAPH_ITEM_BYTES * n_items
        + _GRAPH_UNKNOWN_BYTES * n_unknown
        + (_GRAPH_NODE_BYTES + 8) * n_nodes
    )
    _peak(ledger, "the lists of each item's nodes", graph_bytes)
    item_indptr, item_nodes, codes = _items_by_group(nodes, n_unknown)
    local = codes[:n_items]
    unknown_local = codes[n_items:]
    del codes
    sizes = np.diff(indptr)
    empty = np.empty(0, dtype=np.int64)
    arguments = (
        indptr,
        local,
        item_indptr,
        item_nodes,
        sizes,
        unknown_indptr,
        unknown_local,
        within,
        other,
        node_modalities,
        bool(inclusive),
    )
    n_edges = _jaccard_pairs(*arguments, empty, empty, False)
    _peak(
        ledger, f"the {n_edges} edges of the graph", graph_bytes + _EDGE_BYTES * n_edges
    )
    first = np.empty(n_edges, dtype=np.int64)
    second = np.empty(n_edges, dtype=np.int64)
    _jaccard_pairs(*arguments, first, second, True)
    return Edges(first=first, second=second)


def _prefix_edges(
    nodes: SetNodes,
    indptr: np.ndarray,
    within: float,
    across: float,
    modalities: np.ndarray,
    inclusive: bool,
    ledger: MemoryLedger,
) -> Edges:
    """The edges of :func:`jaccard_edges` between nodes without unknown items, met through their prefixes.

    Each step is charged beside the nodes' modalities, which the caller made
    and holds until the edges are returned.
    """
    n_nodes = nodes.n_nodes
    n_items = len(nodes.sets.items)
    modality_bytes = 8 * n_nodes
    _peak(
        ledger,
        "the lists of each item's nodes",
        modality_bytes + _ITEM_CODE_BYTES * n_items + _ITEM_CODE_NODE_BYTES * n_nodes,
    )
    item_indptr, item_nodes, codes = _items_by_group(nodes, 0)
    del item_nodes
    n_codes = len(item_indptr) - 1
    lowest = min(within, across)
    n_prefix, largest, most_holders = _prefix_shape(indptr, item_indptr, lowest)
    # The codes and their pointers stay while each code's rank is counted,
    # and the ranks while the prefixes are chosen: each node's flags, the
    # lists of each code's nodes with their pointers and the places they are
    # filled at, and one node's ranks with their sorted copy.
    coded = modality_bytes + 8 * n_items + 8 * (n_codes + 1)
    prefixes = n_items + 8 * (n_codes + 1) + 8 * n_prefix
    choosing = 8 * n_codes + 8 * largest * (2 if largest > 16 else 1)
    _peak(
        ledger,
        "the prefixes of the nodes' sets",
        coded + 8 * n_codes + max(8 * (most_holders + 2), prefixes + choosing),
    )
    ranks = _item_ranks(item_indptr, most_holders)
    in_prefix, prefix_indptr, prefix_nodes = _set_prefixes(
        indptr, codes, ranks, lowest, n_prefix, largest
    )
    del ranks, item_indptr
    # Each node then meets the nodes its prefix reaches, marking each and
    # listing those it touched.
    kept = modality_bytes + 8 * n_items + prefixes
    _peak(ledger, "the nodes that the prefixes reach", kept + 16 * n_nodes)
    empty = np.empty(0, dtype=np.int64)
    arguments = (
        indptr,
        codes,
        in_prefix,
        prefix_indptr,
        prefix_nodes,
        within,
        across,
        modalities,
        inclusive,
    )
    n_edges, most_met = _prefix_jaccard_pairs(*arguments, empty, empty, False)
    # Filling the edges sorts the nodes each node met, a copy of the most.
    _peak(
        ledger,
        f"the {n_edges} edges of the graph",
        kept + 16 * n_nodes + 8 * most_met + _EDGE_BYTES * n_edges,
    )
    first = np.empty(n_edges, dtype=np.int64)
    second = np.empty(n_edges, dtype=np.int64)
    _prefix_jaccard_pairs(*arguments, first, second, True)
    return Edges(first=first, second=second)


def _items_by_group(
    nodes: SetNodes, n_unknown: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each item's nodes, with the items coded within each group.

    Items are coded within each group, so that nodes of two groups never
    meet through an item they both hold; the nodes' unknown items take the
    codes of their groups' items, or codes of their own.

    Returns:
        The pointers and nodes of each code's list of the nodes that hold
        it, ascending, and the code of each node's item in turn, followed by
        those of its unknown items.
    """
    n_nodes = nodes.n_nodes
    indptr = np.asarray(nodes.sets.indptr, dtype=np.int64)
    n_items = len(nodes.sets.items)
    entry_nodes = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    items = np.asarray(nodes.sets.items, dtype=np.int64)
    if n_unknown:
        assert nodes.unknown is not None
        unknown_indptr = np.asarray(nodes.unknown.indptr, dtype=np.int64)
        entry_nodes = np.concatenate(
            [
                entry_nodes,
                np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(unknown_indptr)),
            ]
        )
        items = np.concatenate([items, np.asarray(nodes.unknown.items, dtype=np.int64)])
    n_entries = len(items)
    entry_groups = nodes.groups[entry_nodes]
    by_item = np.lexsort((entry_nodes, items, entry_groups))
    new = np.ones(n_entries, dtype=bool)
    if n_entries > 1:
        new[1:] = (items[by_item][1:] != items[by_item][:-1]) | (
            entry_groups[by_item][1:] != entry_groups[by_item][:-1]
        )
    del entry_groups, items
    codes = np.empty(n_entries, dtype=np.int64)
    codes[by_item] = np.cumsum(new) - 1
    n_codes = int(np.count_nonzero(new))
    del new
    # Only the items held, not those unknown, list their nodes.
    held = by_item[by_item < n_items] if n_unknown else by_item
    item_nodes = entry_nodes[held]
    del entry_nodes, by_item, held
    item_indptr = np.zeros(n_codes + 1, dtype=np.int64)
    np.cumsum(np.bincount(codes[:n_items], minlength=n_codes), out=item_indptr[1:])
    return item_indptr, item_nodes, codes


def _shared_item_edges(nodes: SetNodes, *, ledger: MemoryLedger) -> Edges:
    """Edges that join the nodes of each item of a group into one component.

    Each node of an item's list is linked to the next, so the components are
    those of the graph that links every two nodes sharing an item, without
    the pairs of that graph.
    """
    n_items = len(nodes.sets.items)
    _peak(
        ledger,
        "the lists of each item's nodes",
        _GRAPH_ITEM_BYTES * n_items
        + _GRAPH_NODE_BYTES * nodes.n_nodes
        + _EDGE_BYTES * n_items,
    )
    item_indptr, item_nodes, _codes = _items_by_group(nodes, 0)
    del _codes
    following = np.ones(len(item_nodes), dtype=bool)
    following[item_indptr[:-1][np.diff(item_indptr) > 0]] = False
    later = np.flatnonzero(following)
    return Edges(first=item_nodes[later - 1], second=item_nodes[later])


def _neighbours(n_nodes: int, edges: Edges) -> tuple[np.ndarray, np.ndarray]:
    """Each node's neighbours, ascending, as CSR pointers and positions."""
    ends = np.concatenate([edges.first, edges.second])
    others = np.concatenate([edges.second, edges.first])
    order = np.lexsort((others, ends))
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(ends, minlength=n_nodes), out=indptr[1:])
    return indptr, others[order]


def _adjacency_bytes(n_nodes: int, n_edges: int) -> int:
    return _ADJACENCY_EDGE_BYTES * n_edges + _ADJACENCY_NODE_BYTES * n_nodes


@njit(cache=True)
def _bridges(
    first: np.ndarray, second: np.ndarray, indptr: np.ndarray, neighbours: np.ndarray
) -> np.ndarray:
    """Whether each edge's ends share no neighbour while either has another."""
    removed = np.zeros(len(first), dtype=np.bool_)
    for edge in range(len(first)):
        left = first[edge]
        right = second[edge]
        left_start = indptr[left]
        left_end = indptr[left + 1]
        right_start = indptr[right]
        right_end = indptr[right + 1]
        if left_end - left_start < 2 and right_end - right_start < 2:
            continue
        common = False
        while left_start < left_end and right_start < right_end:
            a = neighbours[left_start]
            b = neighbours[right_start]
            if a == b:
                common = True
                break
            if a < b:
                left_start += 1
            else:
                right_start += 1
        removed[edge] = not common
    return removed


def pruned_edges(
    n_nodes: int, edges: Edges, *, ledger: MemoryLedger | None = None
) -> tuple[Edges, int]:
    """Remove the edges whose ends share no neighbour, as TREX removes bridges.

    An edge is removed when its two ends have no common neighbour and either
    end has a neighbour besides the other, TREX's rule for an edge that
    joins two groups of cells which a doublet or a shared barcode bridged.
    The rule is local: it does not test whether removing the edge splits a
    component. Every edge is judged on the graph as given, then the removed
    ones go together.

    Args:
        n_nodes: The nodes.
        edges: The edges.
        ledger: The caller's memory ledger, which the lists of neighbours are
            charged to; None for no limit.

    Returns:
        The edges kept, in their order, and the number removed.

    Raises:
        MemoryBudgetError: If the lists of neighbours do not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    _peak(ledger, "the lists of neighbours", _adjacency_bytes(n_nodes, len(edges)))
    indptr, neighbours = _neighbours(n_nodes, edges)
    removed = _bridges(edges.first, edges.second, indptr, neighbours)
    return edges.kept(~removed), int(np.count_nonzero(removed))


@njit(cache=True)
def _cut_vertices(
    indptr: np.ndarray, neighbours: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    """Whether each eligible node's neighbours fall apart without it."""
    n_nodes = len(indptr) - 1
    cut = np.zeros(n_nodes, dtype=np.bool_)
    marked = np.full(n_nodes, -1, dtype=np.int64)
    seen = np.full(n_nodes, -1, dtype=np.int64)
    stack = np.empty(n_nodes, dtype=np.int64)
    for node in range(n_nodes):
        start = indptr[node]
        end = indptr[node + 1]
        if not eligible[node] or end - start < 2:
            continue
        for at in range(start, end):
            marked[neighbours[at]] = node
        head = neighbours[start]
        seen[head] = node
        stack[0] = head
        depth = 1
        reached = 1
        while depth:
            depth -= 1
            current = stack[depth]
            for at in range(indptr[current], indptr[current + 1]):
                other = neighbours[at]
                if marked[other] == node and seen[other] != node:
                    seen[other] = node
                    stack[depth] = other
                    depth += 1
                    reached += 1
        cut[node] = reached < end - start
    return cut


def local_cut_vertices(
    n_nodes: int,
    edges: Edges,
    eligible: np.ndarray,
    *,
    ledger: MemoryLedger | None = None,
) -> np.ndarray:
    """Find the nodes whose neighbours fall apart once the node is removed.

    A node with two or more neighbours is a local cut vertex when the graph
    that its neighbours and the edges between them form has two or more
    components, TREX's test for a doublet: a cell that carries the marks of
    two clones links cells that are otherwise unrelated. Only ``eligible``
    nodes are tested, such as those of a single cell, as TREX tests them.

    Returns:
        The local cut vertices among the eligible nodes, ascending.

    Raises:
        ValueError: If ``eligible`` does not hold one flag per node.
        MemoryBudgetError: If the lists of neighbours do not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    flags = np.asarray(eligible, dtype=bool)
    if flags.shape != (n_nodes,):
        raise ValueError("eligible must hold one flag per node")
    _peak(ledger, "the lists of neighbours", _adjacency_bytes(n_nodes, len(edges)))
    indptr, neighbours = _neighbours(n_nodes, edges)
    return np.flatnonzero(_cut_vertices(indptr, neighbours, flags)).astype(np.int64)


@njit(cache=True)
def _holds(indptr: np.ndarray, items: np.ndarray, node: int, item: int) -> bool:
    """Whether a node's ascending items hold ``item``."""
    low = indptr[node]
    high = indptr[node + 1]
    while low < high:
        middle = (low + high) // 2
        if items[middle] < item:
            low = middle + 1
        else:
            high = middle
    return bool(low < indptr[node + 1] and items[low] == item)


@njit(cache=True)
def _ambiguous(
    indptr: np.ndarray,
    neighbours: np.ndarray,
    set_indptr: np.ndarray,
    set_items: np.ndarray,
    unknown_indptr: np.ndarray,
    unknown_items: np.ndarray,
) -> np.ndarray:
    """Whether two of each node's neighbours disagree at one of its unknown items."""
    n_nodes = len(indptr) - 1
    flags = np.zeros(n_nodes, dtype=np.bool_)
    for node in range(n_nodes):
        if indptr[node + 1] - indptr[node] < 2:
            continue
        for position in range(unknown_indptr[node], unknown_indptr[node + 1]):
            item = unknown_items[position]
            held = False
            absent = False
            for at in range(indptr[node], indptr[node + 1]):
                other = neighbours[at]
                if _holds(set_indptr, set_items, other, item):
                    held = True
                elif not _holds(unknown_indptr, unknown_items, other, item):
                    absent = True
                if held and absent:
                    break
            if held and absent:
                flags[node] = True
                break
    return flags


def ambiguous_nodes(
    nodes: SetNodes, edges: Edges, *, ledger: MemoryLedger | None = None
) -> np.ndarray:
    """Find the nodes whose sets agree with two nodes that disagree.

    With unknown items, a node is linked to the nodes it agrees with wherever
    both know. When two of its neighbours disagree at an item that the node
    holds unknown, one holding the item and the other knowing it absent, the
    node cannot tell which of them its cells belong with: they are ambiguous,
    however many there are. Neighbours that share no known item do not
    disagree, so a node whose cells were read where its partly read
    neighbours were not stays. A node without unknown items is never
    ambiguous.

    Returns:
        Whether each node is ambiguous.

    Raises:
        MemoryBudgetError: If the lists of neighbours do not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    n_nodes = nodes.n_nodes
    if nodes.unknown is None or not len(edges):
        return np.zeros(n_nodes, dtype=bool)
    _peak(ledger, "the lists of neighbours", _adjacency_bytes(n_nodes, len(edges)))
    indptr, neighbours = _neighbours(n_nodes, edges)
    return np.asarray(
        _ambiguous(
            indptr,
            neighbours,
            np.asarray(nodes.sets.indptr, dtype=np.int64),
            np.asarray(nodes.sets.items, dtype=np.int64),
            np.asarray(nodes.unknown.indptr, dtype=np.int64),
            np.asarray(nodes.unknown.items, dtype=np.int64),
        ),
        dtype=bool,
    )


@njit(cache=True)
def _roots(n_nodes: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Each node's component as its lowest node, by union-find."""
    parent = np.arange(n_nodes, dtype=np.int64)
    for edge in range(len(first)):
        left = first[edge]
        while parent[left] != left:
            parent[left] = parent[parent[left]]
            left = parent[left]
        right = second[edge]
        while parent[right] != right:
            parent[right] = parent[parent[right]]
            right = parent[right]
        if left < right:
            parent[right] = left
        elif right < left:
            parent[left] = right
    for node in range(n_nodes):
        root = node
        while parent[root] != root:
            root = parent[root]
        parent[node] = root
    return parent


@njit(cache=True)
def _touches_removed(kept: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    """Whether an edge touches a node that is not kept."""
    for edge in range(len(first)):
        if not kept[first[edge]] or not kept[second[edge]]:
            return True
    return False


def connected_components(
    n_nodes: int,
    edges: Edges,
    present: np.ndarray | None = None,
    *,
    ledger: MemoryLedger | None = None,
) -> np.ndarray:
    """Number the connected components of a graph by their lowest node.

    Args:
        n_nodes: The nodes.
        edges: The edges, between present nodes.
        present: The nodes that remain, or None for all; a node removed,
            such as a doublet, belongs to no component.
        ledger: The caller's memory ledger, which the components is charged to;
            None for no limit.

    Returns:
        Each node's component as int64, numbered in the order of each
        component's lowest node, -1 for a node removed.

    Raises:
        ValueError: If ``present`` does not hold one flag per node, or an edge
            touches a node removed.
        MemoryBudgetError: If the components do not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    kept = np.ones(n_nodes, dtype=bool) if present is None else np.asarray(present)
    if kept.shape != (n_nodes,):
        raise ValueError("present must hold one flag per node")
    _peak(ledger, "the components", _COMPONENT_NODE_BYTES * n_nodes)
    kept = kept.astype(bool)
    if _touches_removed(kept, edges.first, edges.second):
        raise ValueError("An edge touches a node that was removed")
    roots = _roots(n_nodes, edges.first, edges.second)
    components = np.full(n_nodes, -1, dtype=np.int64)
    _distinct, numbered = unique_inverse(roots[kept])
    components[kept] = numbered
    return components


@dataclass(frozen=True, slots=True)
class SetClones:
    """The clones called from the cells' sets.

    Attributes:
        codes: Each cell's clone, numbered by size, largest first, then by
            first cell; -1 for a cell without one.
        sizes: The cells of each clone.
        doublets: Whether each cell was removed as a doublet.
        ambiguous: Whether each cell was removed as ambiguous, its set
            agreeing with two clones that disagree.
        n_nodes: The distinct sets.
        n_edges: The edges of the Jaccard graph, before pruning.
        n_pruned: The edges that pruning removed, over both rounds.
        n_doublet_nodes: The doublets that each round removed.
        n_small_cells: The cells left without a clone because theirs held
            fewer cells than the smallest size.
        n_ambiguous_nodes: The nodes removed as ambiguous.
    """

    codes: np.ndarray
    sizes: np.ndarray
    doublets: np.ndarray
    ambiguous: np.ndarray
    n_nodes: int
    n_edges: int
    n_pruned: int
    n_doublet_nodes: tuple[int, ...]
    n_small_cells: int
    n_ambiguous_nodes: int

    @property
    def n_clones(self) -> int:
        return len(self.sizes)

    @property
    def nbytes(self) -> int:
        """The bytes of the clones' arrays, which :func:`set_clones` leaves charged to its ledger."""
        return sum(
            array.nbytes
            for array in (self.codes, self.sizes, self.doublets, self.ambiguous)
        )


def set_clones(
    sets: CellSets,
    *,
    unknown: CellSets | None = None,
    groups: np.ndarray | None = None,
    method: str = "sets",
    similarity: float | None = None,
    inclusive: bool | None = None,
    across: float | None = None,
    modalities: np.ndarray | None = None,
    prune: bool = False,
    doublets: bool = False,
    ambiguous: bool = False,
    min_size: int = 1,
    ledger: MemoryLedger | None = None,
) -> SetClones:
    """Call clones from the cells' sets, within each group.

    ``method="sets"`` makes each node of identical sets a clone.
    ``method="jaccard"`` links the nodes by :func:`jaccard_edges`, then, as
    TREX does, ``prune`` removes the edges of :func:`pruned_edges` and
    ``doublets`` the single-cell nodes of :func:`local_cut_vertices` with
    their cells, in TREX's order: pruning, doublets, pruning again and
    doublets again. Before them, ``ambiguous`` removes the nodes of
    :func:`ambiguous_nodes`, whatever their cells, which sets with unknown
    items need, such as genotypes: a node of cells that agree with two
    clones that disagree joins neither. ``method="shared"`` links the nodes that share any item,
    as DARLIN links cells that share a rare allele, joining the nodes of
    each item without listing every pair of them. The connected components
    of what remains are the clones, and a clone of fewer than ``min_size``
    cells leaves its cells without one.

    Args:
        sets: Each cell's items; a cell without one gets no clone.
        unknown: Each cell's unknown items, such as the sites that no read
            covered, for ``"sets"`` and ``"jaccard"``; None for none.
        groups: Each cell's group, within which clones are called, or None.
        method: ``"sets"``, ``"jaccard"`` or ``"shared"``.
        similarity: For ``"jaccard"``, the index that links two nodes.
        inclusive: For ``"jaccard"``, whether an index equal to the
            similarity links.
        across: For ``"jaccard"``, the similarity between nodes of different
            modalities, or None.
        modalities: Each cell's modality as an integer code, with ``across``.
        prune: For ``"jaccard"``, remove TREX's bridges.
        doublets: For ``"jaccard"``, remove TREX's doublets.
        ambiguous: For ``"jaccard"``, remove the ambiguous nodes.
        min_size: The fewest cells of a clone.
        ledger: The caller's memory ledger, which each step is charged to;
            None for no limit. The clones returned stay charged to it, their
            ``nbytes``, for the caller to release once it lets them go.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If an option is unknown, missing for the method or given
            to a method that does not use it.
        MemoryBudgetError: If a step does not fit the ledger's budget.
    """
    ledger = _ledger(ledger)
    if method not in CALLING_METHODS:
        raise ValueError(f"method must be one of {', '.join(CALLING_METHODS)}")
    flags = tuple(
        (name, bool_argument(flag, name))
        for name, flag in (
            ("prune", prune),
            ("doublets", doublets),
            ("ambiguous", ambiguous),
        )
    )
    prune, doublets, ambiguous = (flag for _name, flag in flags)
    if isinstance(min_size, bool | np.bool_) or not isinstance(
        min_size, int | np.integer
    ):
        raise TypeError("min_size must be an integer")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if method != "jaccard":
        given = [
            name
            for name, value in (
                ("similarity", similarity),
                ("inclusive", inclusive),
                ("across", across),
                ("modalities", modalities),
            )
            if value is not None
        ] + [name for name, flag in flags if flag]
        if method == "shared" and unknown is not None:
            given.append("unknown")
        if given:
            rule = (
                "groups identical sets alone"
                if method == "sets"
                else "links the sets that share any item"
            )
            raise ValueError(
                f"method={quoted_value(method)} {rule}, so {', '.join(given)} would change "
                "nothing; use method='jaccard'"
            )
    elif similarity is None or inclusive is None:
        raise ValueError("method='jaccard' needs a similarity and inclusive")
    n_cells = sets.n_cells
    nodes = identical_sets(sets, groups, unknown=unknown, ledger=ledger)
    n_nodes = nodes.n_nodes
    # The nodes are held until the clones are numbered: each cell's node, and
    # each node's first cell, size, group, set and whether it remains and is
    # ambiguous.
    nodes_bytes = ledger.hold(
        "the nodes",
        _HELD_CELL_BYTES * n_cells
        + _NODE_BYTES * n_nodes
        + _HELD_ITEM_BYTES * len(nodes.sets.items)
        + (
            0
            if nodes.unknown is None
            else _HELD_ITEM_BYTES * (len(nodes.unknown.items) + n_nodes + 1)
        ),
    )
    present = np.ones(n_nodes, dtype=bool)
    unclear = np.zeros(n_nodes, dtype=bool)
    edges = Edges(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    edge_bytes = 0
    n_edges = n_pruned = 0
    removed_rounds: list[int] = []
    if method == "shared":
        edges = _shared_item_edges(nodes, ledger=ledger)
        edge_bytes = ledger.hold("the graph's edges", edges.nbytes)
    if method == "jaccard":
        assert similarity is not None and inclusive is not None
        node_modalities = None
        if across is not None:
            # Each cell's node and modality, and each node's lowest and
            # highest, while the nodes' modalities are found.
            _peak(ledger, "the nodes' modalities", 17 * n_cells + 24 * n_nodes)
            node_modalities = _node_modalities(nodes, modalities, n_cells)
            ledger.hold("the nodes' modalities", node_modalities.nbytes)
        edges = jaccard_edges(
            nodes,
            similarity,
            inclusive=bool(inclusive),
            across=across,
            modalities=node_modalities,
            ledger=ledger,
        )
        if node_modalities is not None:
            modality_bytes = node_modalities.nbytes
            del node_modalities
            ledger.release(modality_bytes)
        else:
            del node_modalities
        n_edges = len(edges)
        edge_bytes = ledger.hold("the graph's edges", edges.nbytes)
        if ambiguous:
            unclear = ambiguous_nodes(nodes, edges, ledger=ledger)
            present &= ~unclear
            edges, kept_bytes = _kept_edges(ledger, edges, present)
            ledger.release(edge_bytes)
            edge_bytes = kept_bytes
        rounds = 2 if doublets else 1
        for _round in range(rounds):
            if prune:
                pruned, removed = pruned_edges(n_nodes, edges, ledger=ledger)
                kept_bytes = ledger.hold("the graph's edges", pruned.nbytes)
                edges = pruned
                del pruned
                ledger.release(edge_bytes)
                edge_bytes = kept_bytes
                n_pruned += removed
            if doublets:
                cut = local_cut_vertices(
                    n_nodes, edges, present & (nodes.cell_counts == 1), ledger=ledger
                )
                removed_rounds.append(len(cut))
                present[cut] = False
                del cut
                edges, kept_bytes = _kept_edges(ledger, edges, present)
                ledger.release(edge_bytes)
                edge_bytes = kept_bytes
    components = connected_components(n_nodes, edges, present, ledger=ledger)
    component_bytes = ledger.hold("the nodes' components", components.nbytes)
    del edges
    ledger.release(edge_bytes)
    n_components = int(components.max()) + 1 if np.any(components >= 0) else 0
    # Numbering the clones holds the arrays of each cell, and the sizes,
    # order and rank of each component.
    _peak(
        ledger,
        "the cells' clones",
        _CLONE_CELL_BYTES * n_cells + _CLONE_COMPONENT_BYTES * n_components,
    )
    doublet = np.zeros(n_cells, dtype=bool)
    removed_unclear = np.zeros(n_cells, dtype=bool)
    has_node = nodes.cell_nodes >= 0
    doublet[has_node] = ~present[nodes.cell_nodes[has_node]]
    removed_unclear[has_node] = unclear[nodes.cell_nodes[has_node]]
    doublet &= ~removed_unclear
    cell_components = np.full(n_cells, -1, dtype=np.int64)
    cell_components[has_node] = components[nodes.cell_nodes[has_node]]
    in_component = cell_components >= 0
    component_sizes = np.bincount(cell_components[in_component], minlength=n_components)
    # Components are numbered by their lowest node, which holds their first
    # cell, so ranking by size and then component orders ties by first cell.
    kept = np.flatnonzero(component_sizes >= min_size)
    order = kept[np.argsort(-component_sizes[kept], kind="stable")]
    rank = np.full(n_components, -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[in_component] = rank[cell_components[in_component]]
    n_small = int(np.count_nonzero(in_component & (codes < 0)))
    called = SetClones(
        codes=codes,
        sizes=component_sizes[order].astype(np.int64),
        doublets=doublet,
        ambiguous=removed_unclear,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_pruned=n_pruned,
        n_doublet_nodes=tuple(removed_rounds),
        n_small_cells=n_small,
        n_ambiguous_nodes=int(np.count_nonzero(unclear)),
    )
    # The clones are held once what numbering them built is let go.
    del has_node, cell_components, in_component, component_sizes, kept, order
    del rank, codes, doublet, removed_unclear
    ledger.hold("the cells' clones", called.nbytes)
    del nodes, components, present, unclear
    ledger.release(component_bytes)
    ledger.release(nodes_bytes)
    return called


def _kept_edges(
    ledger: MemoryLedger, edges: Edges, present: np.ndarray
) -> tuple[Edges, int]:
    """The edges between nodes that remain, held on ``ledger`` beside ``edges``, which the caller lets go."""
    # Each edge's two flags and their conjunction, and the edges kept.
    ledger.peak("the edges kept", (_EDGE_BYTES + 3) * len(edges), remedy=_REMEDY)
    kept = edges.kept(present[edges.first] & present[edges.second])
    return kept, ledger.hold("the graph's edges", kept.nbytes)


def _node_modalities(
    nodes: SetNodes, modalities: np.ndarray | None, n_cells: int
) -> np.ndarray:
    """Each node's modality: its cells' one modality, or -1 for several."""
    if modalities is None:
        raise ValueError("A similarity across modalities needs each cell's modality")
    codes = np.asarray(modalities)
    if codes.shape != (n_cells,) or (n_cells and codes.dtype.kind not in "iu"):
        raise ValueError("modalities must hold one integer code per cell")
    codes = codes.astype(np.int64)
    if n_cells and codes.min() < 0:
        raise ValueError("modality codes must be at least 0")
    member = nodes.cell_nodes >= 0
    node_codes = nodes.cell_nodes[member]
    lowest = np.full(nodes.n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.full(nodes.n_nodes, -1, dtype=np.int64)
    np.minimum.at(lowest, node_codes, codes[member])
    np.maximum.at(highest, node_codes, codes[member])
    return np.where(lowest == highest, lowest, -1)
