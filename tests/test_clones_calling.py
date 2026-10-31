"""Clones from per-cell sets: identical sets, the Jaccard graph, TREX's pruning and doublets.

Graphs are built by hand, or from sets whose shared items give exactly the
edges wanted under a similarity of 0, and their expected nodes, edges,
bridges, cut vertices and clones are derived by hand or by networkx from
TREX's documented rules.
"""

import itertools

import networkx as nx
import numpy as np
import pytest

from cytearc.clones import calling
from cytearc.clones.calling import (
    CellSets,
    Edges,
    cell_sets,
    connected_components,
    identical_sets,
    jaccard_edges,
    local_cut_vertices,
    pruned_edges,
    set_clones,
)
from tests.clone_helpers import smallest_budget, traced_peak


def _sets(rows: list[list[int]]) -> CellSets:
    cells = [cell for cell, row in enumerate(rows) for _ in row]
    items = [item for row in rows for item in row]
    return cell_sets(np.asarray(cells), np.asarray(items), len(rows))


def _edges(pairs: list[tuple[int, int]]) -> Edges:
    ordered = sorted((min(a, b), max(a, b)) for a, b in pairs)
    first = np.asarray([a for a, _ in ordered], dtype=np.int64)
    second = np.asarray([b for _, b in ordered], dtype=np.int64)
    return Edges(first=first, second=second)


def _pairs(edges: Edges) -> list[tuple[int, int]]:
    return list(zip(edges.first.tolist(), edges.second.tolist(), strict=True))


def _graph_sets(n_nodes: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Sets whose shared items are exactly the edges, with an item of each node's own.

    Under a similarity of 0 two such sets are linked exactly when they hold
    the item of their edge.
    """
    rows = [[node] for node in range(n_nodes)]
    for offset, (a, b) in enumerate(pairs):
        rows[a].append(n_nodes + offset)
        rows[b].append(n_nodes + offset)
    return [sorted(row) for row in rows]


def _partition(codes: np.ndarray) -> set[frozenset[int]]:
    return {
        frozenset(np.flatnonzero(codes == clone).tolist())
        for clone in range(int(codes.max()) + 1 if len(codes) else 0)
    }


# Sets built from pairs.


def test_sets_merge_repeated_pairs_and_order_each_cells_items() -> None:
    sets = cell_sets(np.asarray([2, 0, 2, 2, 0]), np.asarray([5, 3, 1, 5, 3]), 4)
    assert sets.indptr.tolist() == [0, 1, 1, 3, 3]
    assert sets.items.tolist() == [3, 1, 5]
    assert sets.sizes.tolist() == [1, 0, 2, 0]


@pytest.mark.parametrize(
    ("indptr", "items", "message"),
    [
        ([0, 2], [3, 3], "ascending and distinct"),
        ([0, 2], [4, 3], "ascending and distinct"),
        ([0, 1, 3], [1, 2], "rise from 0"),
        ([1, 2], [1, 2], "rise from 0"),
        ([0, 2, 1], [1, 2], "rise from 0"),
        ([0, 1], [-1], "at least 0"),
        ([0, 1], [0.5], "integer codes"),
        ([[0, 1]], [1], "n_cells \\+ 1 integer offsets"),
        ([0.0, 1.0], [1], "n_cells \\+ 1 integer offsets"),
        ([], [], "n_cells \\+ 1 integer offsets"),
    ],
)
def test_sets_refuse_pointers_and_items_they_cannot_hold(
    indptr, items, message
) -> None:
    with pytest.raises(ValueError, match=message):
        CellSets(indptr=np.asarray(indptr), items=np.asarray(items))


def test_a_cells_items_may_repeat_those_of_the_cell_before() -> None:
    # Each row starts afresh, so its first item may be lower than the last
    # of the row before, and trailing empty rows point at the end.
    sets = CellSets(indptr=np.asarray([0, 2, 3, 3, 3]), items=np.asarray([4, 9, 1]))
    assert sets.n_cells == 4


def test_sets_refuse_cells_and_items_out_of_range() -> None:
    with pytest.raises(ValueError, match="positions below 2"):
        cell_sets(np.asarray([2]), np.asarray([1]), 2)
    with pytest.raises(ValueError, match="at least 0"):
        cell_sets(np.asarray([0]), np.asarray([-1]), 2)
    with pytest.raises(ValueError, match="one cell and one item"):
        cell_sets(np.asarray([0, 1]), np.asarray([1]), 2)


# Identical sets.


def test_identical_sets_of_a_group_share_a_node_numbered_by_its_first_cell() -> None:
    sets = _sets([[4], [1, 2], [], [2, 1], [1, 2], [4], [7]])
    nodes = identical_sets(sets, groups=np.asarray([0, 0, 0, 0, 1, 0, 0]))
    # {4} first at cell 0, {1, 2} of group 0 at cell 1, {1, 2} of group 1 at
    # cell 4, {7} at cell 6; the empty set joins no node.
    assert nodes.cell_nodes.tolist() == [0, 1, -1, 1, 2, 0, 3]
    assert nodes.first_cells.tolist() == [0, 1, 4, 6]
    assert nodes.n_cells.tolist() == [2, 2, 1, 1]
    assert nodes.groups.tolist() == [0, 0, 1, 0]
    assert nodes.sets.items.tolist() == [4, 1, 2, 1, 2, 7]


def test_colliding_hashes_never_merge_different_sets(monkeypatch) -> None:
    rows = [[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]]
    honest = identical_sets(_sets(rows))

    def colliding(indptr, items):
        n_sets = len(indptr) - 1
        return np.zeros(n_sets, dtype=np.uint64), np.zeros(n_sets, dtype=np.uint64)

    monkeypatch.setattr(calling, "_set_hashes", colliding)
    collided = identical_sets(_sets(rows))
    assert collided.cell_nodes.tolist() == honest.cell_nodes.tolist() == [0, 1, 0, 2, 1]
    assert collided.n_cells.tolist() == [2, 2, 1]


def test_hashes_tell_apart_sets_of_one_size_and_shared_items() -> None:
    # Every 3-item set of 8 items is its own node, whatever items it shares.
    rows = [list(row) for row in itertools.combinations(range(8), 3)]
    nodes = identical_sets(_sets(rows + rows[::-1]))
    assert nodes.n_nodes == len(rows)
    assert nodes.n_cells.tolist() == [2] * len(rows)


def test_identical_sets_refuse_groups_that_do_not_align() -> None:
    sets = _sets([[1], [2]])
    with pytest.raises(ValueError, match="one integer code per cell"):
        identical_sets(sets, groups=np.asarray([0]))
    with pytest.raises(ValueError, match="one integer code per cell"):
        identical_sets(sets, groups=np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="at least 0"):
        identical_sets(sets, groups=np.asarray([0, -1]))


# The Jaccard graph.


def _jaccard(rows, similarity, inclusive, groups=None, **options) -> list:
    nodes = identical_sets(_sets(rows), groups=groups)
    return _pairs(jaccard_edges(nodes, similarity, inclusive=inclusive, **options))


def test_an_index_equal_to_the_similarity_links_only_when_inclusive() -> None:
    # 7 shared of 10 items is 0.7 exactly in floating point, as TREX's
    # len(a & b) / len(a | b) computes it; 2 of 3 is the float 2 / 3.
    seven = [list(range(7)), list(range(10))]
    assert _jaccard(seven, 0.7, False) == []
    assert _jaccard(seven, 0.7, True) == [(0, 1)]
    assert _jaccard(seven, 0.69, False) == [(0, 1)]
    two_thirds = [[1, 2], [1, 2, 3]]
    assert _jaccard(two_thirds, 2 / 3, False) == []
    assert _jaccard(two_thirds, 2 / 3, True) == [(0, 1)]


def test_sets_that_share_no_item_or_another_group_never_link() -> None:
    rows = [[1], [2], [1, 3], [1]]
    # A similarity of 0 links every pair that shares an item, inclusive or
    # not, and nothing else.
    assert _jaccard(rows, 0.0, False) == [(0, 2)]
    assert _jaccard(rows, 0.0, True) == [(0, 2)]
    # In another group, cell 2 meets neither cell 0 nor cell 3, which share a
    # node; with cell 3 in its group instead, the two link.
    assert _jaccard(rows, 0.0, False, groups=np.asarray([0, 0, 1, 0])) == []
    assert _jaccard(rows, 0.0, False, groups=np.asarray([0, 0, 1, 1])) == [(2, 3)]


def test_a_similarity_of_one_links_nothing_since_equal_sets_share_a_node() -> None:
    rows = [[1, 2], [1, 2], [1, 2, 3]]
    assert _jaccard(rows, 1.0, True) == []
    assert _jaccard(rows, 1.0, False) == []


def test_edges_list_every_pair_once_in_order_of_their_ends() -> None:
    # Five sets that all share item 0 link in all ten pairs.
    rows = [[0, 10 + node] for node in range(5)]
    assert _jaccard(rows, 0.0, False) == list(itertools.combinations(range(5), 2))


def test_two_similarities_apply_within_and_across_modalities() -> None:
    # The two sets share 11 of 20 items, 0.55: above 0.5 but not 0.6.
    rows = [list(range(0, 15)), list(range(4, 20))]
    nodes = identical_sets(_sets(rows))
    assert len(set(rows[0]) & set(rows[1])) / len(set(rows[0]) | set(rows[1])) == 0.55

    def linked(modalities: list[int]) -> list:
        return _pairs(
            jaccard_edges(
                nodes,
                0.6,
                inclusive=False,
                across=0.5,
                modalities=np.asarray(modalities),
            )
        )

    assert linked([0, 0]) == []
    assert linked([0, 1]) == [(0, 1)]
    # A node whose cells come from several modalities is across from all.
    assert linked([-1, 0]) == [(0, 1)]
    assert linked([-1, -1]) == [(0, 1)]


def test_the_jaccard_graph_refuses_settings_it_cannot_use() -> None:
    nodes = identical_sets(_sets([[1], [1, 2]]))
    with pytest.raises(ValueError, match="from 0 to 1"):
        jaccard_edges(nodes, 1.5, inclusive=False)
    with pytest.raises(TypeError, match="from 0 to 1"):
        jaccard_edges(nodes, True, inclusive=False)
    with pytest.raises(TypeError, match="inclusive must be a boolean"):
        jaccard_edges(nodes, 0.5, inclusive=1)
    with pytest.raises(ValueError, match="needs each node's modality"):
        jaccard_edges(nodes, 0.5, inclusive=False, across=0.4)
    with pytest.raises(ValueError, match="only with a similarity across"):
        jaccard_edges(nodes, 0.5, inclusive=False, modalities=np.asarray([0, 0]))
    with pytest.raises(ValueError, match="one integer code per node"):
        jaccard_edges(
            nodes, 0.5, inclusive=False, across=0.4, modalities=np.asarray([0])
        )


def test_the_graph_matches_every_pair_compared_by_python_sets() -> None:
    rng = np.random.default_rng(3)
    rows = [
        sorted(rng.choice(30, size=int(rng.integers(1, 6)), replace=False).tolist())
        for _ in range(300)
    ]
    groups = rng.integers(0, 3, size=len(rows))
    nodes = identical_sets(_sets(rows), groups=groups)
    found = set(_pairs(jaccard_edges(nodes, 0.4, inclusive=True)))
    expected = set()
    node_rows = [set(rows[cell]) for cell in nodes.first_cells.tolist()]
    for a, b in itertools.combinations(range(nodes.n_nodes), 2):
        if nodes.groups[a] != nodes.groups[b]:
            continue
        shared = len(node_rows[a] & node_rows[b])
        if shared and shared / len(node_rows[a] | node_rows[b]) >= 0.4:
            expected.add((a, b))
    assert found == expected and expected


# TREX's pruning.


def test_pruning_judges_every_edge_on_the_graph_as_given() -> None:
    # On a path a-b-c, each edge's ends share no neighbour while b has two,
    # so both go together; removed one at a time, the second would stay.
    kept, removed = pruned_edges(3, _edges([(0, 1), (1, 2)]))
    assert _pairs(kept) == [] and removed == 2


@pytest.mark.parametrize(
    ("n_nodes", "pairs", "kept"),
    [
        # One edge whose ends have no other neighbour stays.
        (2, [(0, 1)], [(0, 1)]),
        # Every edge of a triangle has a common neighbour.
        (3, [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 2)]),
        # The edge between two triangles has none, and goes.
        (
            6,
            [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)],
            [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
        ),
        # A square has no triangle, so every edge goes.
        (4, [(0, 1), (1, 2), (2, 3), (0, 3)], []),
        # A pendant edge off a triangle: its ends share nothing and the
        # triangle's end has others.
        (4, [(0, 1), (0, 2), (1, 2), (2, 3)], [(0, 1), (0, 2), (1, 2)]),
    ],
)
def test_pruning_removes_edges_whose_ends_share_no_neighbour(
    n_nodes, pairs, kept
) -> None:
    result, removed = pruned_edges(n_nodes, _edges(pairs))
    assert _pairs(result) == kept
    assert removed == len(pairs) - len(kept)


# TREX's doublets: local cut vertices.


@pytest.mark.parametrize(
    ("n_nodes", "pairs", "cut"),
    [
        # The middle of a path holds two neighbours that nothing else links.
        (3, [(0, 1), (1, 2)], [1]),
        # A triangle's nodes see their neighbours linked.
        (3, [(0, 1), (0, 2), (1, 2)], []),
        # A star's centre, whose three leaves are apart.
        (4, [(0, 1), (0, 2), (0, 3)], [0]),
        # A node joining two triangles by two neighbours on each side.
        (
            7,
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (3, 4),
                (3, 5),
                (4, 5),
                (6, 0),
                (6, 1),
                (6, 3),
                (6, 4),
            ],
            [6],
        ),
        # The neighbours of node 4 are linked through each other, so it is
        # not a cut vertex though removing it changes the graph.
        (5, [(0, 1), (1, 2), (0, 4), (1, 4), (2, 4), (2, 3)], [2]),
    ],
)
def test_cut_vertices_are_nodes_whose_neighbours_fall_apart(
    n_nodes, pairs, cut
) -> None:
    found = local_cut_vertices(n_nodes, _edges(pairs), np.ones(n_nodes, dtype=bool))
    assert found.tolist() == cut


def test_only_eligible_nodes_are_tested_for_cutting() -> None:
    edges = _edges([(0, 1), (0, 2), (0, 3)])
    eligible = np.asarray([False, True, True, True])
    assert local_cut_vertices(4, edges, eligible).tolist() == []
    with pytest.raises(ValueError, match="one flag per node"):
        local_cut_vertices(4, edges, np.ones(3, dtype=bool))


# Connected components.


def test_components_are_numbered_by_their_lowest_node() -> None:
    edges = _edges([(3, 5), (1, 4), (0, 3)])
    assert connected_components(6, edges).tolist() == [0, 1, 2, 0, 1, 0]


def test_removed_nodes_belong_to_no_component() -> None:
    present = np.asarray([True, False, True, True])
    components = connected_components(4, _edges([(2, 3)]), present)
    assert components.tolist() == [0, -1, 1, 1]
    with pytest.raises(ValueError, match="node that was removed"):
        connected_components(4, _edges([(1, 2)]), present)
    with pytest.raises(ValueError, match="one flag per node"):
        connected_components(4, _edges([]), np.ones(3, dtype=bool))


def test_components_match_networkx_on_random_graphs() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        n_nodes = int(rng.integers(1, 60))
        pairs = {
            tuple(sorted(rng.choice(n_nodes, size=2, replace=False).tolist()))
            for _ in range(int(rng.integers(0, 50)))
            if n_nodes > 1
        }
        components = connected_components(n_nodes, _edges(sorted(pairs)))
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
        graph.add_edges_from(pairs)
        expected = {frozenset(part) for part in nx.connected_components(graph)}
        assert _partition(components) == expected
        # Each component's number follows its lowest node.
        lowest = [
            min(np.flatnonzero(components == c)) for c in range(components.max() + 1)
        ]
        assert lowest == sorted(lowest)


# Calling clones from sets.


def test_sets_alone_make_each_node_a_clone_ranked_by_size_then_first_cell() -> None:
    rows = [[9], [1], [1], [], [2], [9], [1], [3]]
    calls = set_clones(_sets(rows), method="sets")
    # {1} holds three cells, {9} two, and {2} and {3} one each, the first of
    # which comes earlier.
    assert calls.codes.tolist() == [1, 0, 0, -1, 2, 1, 0, 3]
    assert calls.sizes.tolist() == [3, 2, 1, 1]
    assert calls.n_nodes == 4 and calls.n_edges == 0


def test_small_clones_leave_their_cells_without_one() -> None:
    rows = [[9], [1], [1], [2], [9], [1]]
    calls = set_clones(_sets(rows), method="sets", min_size=2)
    assert calls.codes.tolist() == [1, 0, 0, -1, 1, 0]
    assert calls.n_small_cells == 1


def test_the_jaccard_graph_joins_similar_sets_into_one_clone() -> None:
    # A clone of four barcodes, and cells that lost one of them.
    rows = [[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4], [7, 8], [1, 2, 3]]
    calls = set_clones(_sets(rows), method="jaccard", similarity=0.7, inclusive=False)
    assert _partition(calls.codes) == {frozenset({0, 1, 2, 4}), frozenset({3})}


def test_trex_removes_a_doublet_that_links_two_clones() -> None:
    # Two triangles of nodes, 0-1-2 and 3-4-5, and a cell (node 6) that shares
    # items with two nodes of each, so pruning keeps its edges and it is a
    # local cut vertex; every node is one cell.
    pairs = [
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
        (6, 0),
        (6, 1),
        (6, 3),
        (6, 4),
    ]
    rows = _graph_sets(7, pairs)
    calls = set_clones(
        _sets(rows),
        method="jaccard",
        similarity=0.0,
        inclusive=False,
        prune=True,
        doublets=True,
    )
    assert _partition(calls.codes) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}
    assert calls.doublets.tolist() == [False] * 6 + [True]
    assert calls.n_doublet_nodes == (1, 0)
    # Without the doublet rule the cell joins the two clones into one.
    joined = set_clones(
        _sets(rows), method="jaccard", similarity=0.0, inclusive=False, prune=True
    )
    assert _partition(joined.codes) == {frozenset(range(7))}
    assert not joined.doublets.any() and joined.n_doublet_nodes == ()


def test_a_cut_vertex_of_several_cells_is_kept() -> None:
    # The middle of a path is two cells with one set: TREX tests only nodes
    # of one cell.
    rows = _graph_sets(3, [(0, 1), (1, 2)])
    rows.insert(2, rows[1])
    calls = set_clones(
        _sets(rows), method="jaccard", similarity=0.0, inclusive=False, doublets=True
    )
    assert not calls.doublets.any()
    assert _partition(calls.codes) == {frozenset(range(4))}


def test_pruning_isolates_a_doublet_whose_ends_share_no_neighbour() -> None:
    # A cell linked to one node of each triangle loses both edges to pruning
    # first, and stays a clone of its own rather than a doublet.
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (6, 2), (6, 3)]
    calls = set_clones(
        _sets(_graph_sets(7, pairs)),
        method="jaccard",
        similarity=0.0,
        inclusive=False,
        prune=True,
        doublets=True,
    )
    assert _partition(calls.codes) == {
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
        frozenset({6}),
    }
    assert calls.n_pruned == 2 and not calls.doublets.any()


def _reference_trex(n_nodes: int, pairs, single, *, prune: bool, doublets: bool):
    """TREX's sequence as its documentation states it, with networkx."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(pairs)

    def bridges():
        removed = [
            (a, b)
            for a, b in graph.edges()
            if not set(graph[a]) & set(graph[b])
            and (graph.degree(a) > 1 or graph.degree(b) > 1)
        ]
        graph.remove_edges_from(removed)

    def cut():
        found = [
            node
            for node in graph.nodes()
            if single[node]
            and graph.degree(node) > 1
            and nx.number_connected_components(graph.subgraph(graph[node])) > 1
        ]
        graph.remove_nodes_from(found)
        return found

    removed = []
    for _ in range(2 if doublets else 1):
        if prune:
            bridges()
        if doublets:
            removed += cut()
    return {frozenset(part) for part in nx.connected_components(graph)}, set(removed)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    ("prune", "doublets"), [(True, True), (True, False), (False, True)]
)
def test_the_calls_follow_trexs_documented_sequence_on_random_graphs(
    seed, prune, doublets
) -> None:
    rng = np.random.default_rng(seed)
    n_nodes = 40
    pairs = sorted(
        {
            tuple(sorted(rng.choice(n_nodes, size=2, replace=False).tolist()))
            for _ in range(70)
        }
    )
    rows = _graph_sets(n_nodes, pairs)
    # Every third node holds two cells, which TREX never removes.
    cells = [node for node in range(n_nodes) for _ in range(1 + (node % 3 == 0))]
    calls = set_clones(
        _sets([rows[node] for node in cells]),
        method="jaccard",
        similarity=0.0,
        inclusive=False,
        prune=prune,
        doublets=doublets,
    )
    single = [node % 3 != 0 for node in range(n_nodes)]
    expected, removed = _reference_trex(
        n_nodes, pairs, single, prune=prune, doublets=doublets
    )
    cells_of = {
        frozenset(index for index, node in enumerate(cells) if node in part)
        for part in expected
    }
    assert _partition(calls.codes) == cells_of
    assert {cells[index] for index in np.flatnonzero(calls.doublets)} == removed


def test_calls_do_not_depend_on_the_order_items_arrive_in() -> None:
    rng = np.random.default_rng(9)
    rows = [
        rng.choice(40, size=int(rng.integers(1, 5)), replace=False) for _ in range(200)
    ]
    cells = np.concatenate([[cell] * len(row) for cell, row in enumerate(rows)])
    items = np.concatenate(rows)
    options = {
        "method": "jaccard",
        "similarity": 0.3,
        "inclusive": True,
        "prune": True,
        "doublets": True,
    }
    reference = set_clones(cell_sets(cells, items, len(rows)), **options)
    for permutation in range(3):
        order = np.random.default_rng(permutation).permutation(len(cells))
        again = set_clones(cell_sets(cells[order], items[order], len(rows)), **options)
        np.testing.assert_array_equal(again.codes, reference.codes)
        np.testing.assert_array_equal(again.doublets, reference.doublets)


def test_set_clones_refuses_options_of_the_other_method() -> None:
    sets = _sets([[1], [1]])
    with pytest.raises(ValueError, match="similarity, inclusive"):
        set_clones(sets, method="sets", similarity=0.5, inclusive=True)
    with pytest.raises(ValueError, match="prune, doublets"):
        set_clones(sets, method="sets", prune=True, doublets=True)
    with pytest.raises(ValueError, match="needs a similarity and inclusive"):
        set_clones(sets, method="jaccard", similarity=0.5)
    with pytest.raises(ValueError, match="method must be one of"):
        set_clones(sets, method="graph")
    with pytest.raises(ValueError, match="min_size must be at least 1"):
        set_clones(sets, min_size=0)
    with pytest.raises(TypeError, match="min_size must be an integer"):
        set_clones(sets, min_size=2.0)
    with pytest.raises(TypeError, match="prune must be a boolean"):
        set_clones(sets, method="jaccard", similarity=0.5, inclusive=True, prune=1)
    with pytest.raises(ValueError, match="needs each cell's modality"):
        set_clones(sets, method="jaccard", similarity=0.5, inclusive=True, across=0.4)


def test_modalities_must_be_one_code_of_at_least_zero_per_cell() -> None:
    sets = _sets([[1], [1, 2]])
    options = {
        "method": "jaccard",
        "similarity": 0.5,
        "inclusive": False,
        "across": 0.4,
    }
    for modalities, message in (
        (np.asarray([0]), "one integer code per cell"),
        (np.asarray([0.0, 1.0]), "one integer code per cell"),
        (np.asarray([0, -1]), "at least 0"),
    ):
        with pytest.raises(ValueError, match=message):
            set_clones(sets, modalities=modalities, **options)


def test_modalities_of_cells_give_their_nodes_one_modality_or_several() -> None:
    # Cells 0 and 1 share a set from two modalities; cell 2 holds its own.
    rows = [list(range(15)), list(range(15)), list(range(4, 20))]
    one = set_clones(
        _sets(rows),
        method="jaccard",
        similarity=0.6,
        inclusive=False,
        across=0.5,
        modalities=np.asarray([0, 0, 0]),
    )
    assert _partition(one.codes) == {frozenset({0, 1}), frozenset({2})}
    several = set_clones(
        _sets(rows),
        method="jaccard",
        similarity=0.6,
        inclusive=False,
        across=0.5,
        modalities=np.asarray([0, 1, 0]),
    )
    assert _partition(several.codes) == {frozenset({0, 1, 2})}


# Memory.


def _random_sets(n_cells: int, seed: int = 0) -> CellSets:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, size=n_cells)
    cells = np.repeat(np.arange(n_cells), sizes)
    # Clones of about 20 cells share items, so the graph has edges.
    items = cells // 20 * 8 + rng.integers(0, 8, size=len(cells))
    return cell_sets(cells, items, n_cells)


@pytest.mark.parametrize(
    "options",
    [
        {"method": "sets"},
        {"method": "jaccard", "similarity": 0.3, "inclusive": False},
        {
            "method": "jaccard",
            "similarity": 0.3,
            "inclusive": False,
            "prune": True,
            "doublets": True,
        },
    ],
)
def test_calling_holds_no_more_than_the_smallest_budget_it_accepts(options) -> None:
    sets = _random_sets(20_000)
    groups = np.arange(20_000) % 4

    def call(budget: int) -> object:
        return set_clones(sets, groups=groups, max_bytes=budget, **options)

    budget = smallest_budget(call, 1 << 30)
    assert traced_peak(lambda: call(budget)) <= budget


def test_a_budget_too_small_refuses_before_allocating() -> None:
    sets = _random_sets(2_000)
    with pytest.raises(MemoryError, match="Raise the memory budget"):
        set_clones(sets, method="sets", max_bytes=10_000)
    with pytest.raises(MemoryError, match="the sets of the cells"):
        cell_sets(np.arange(1000), np.arange(1000), 1000, max_bytes=1000)
    with pytest.raises(ValueError, match="max_bytes must be at least 0"):
        set_clones(sets, method="sets", max_bytes=-1)
    with pytest.raises(TypeError, match="max_bytes must be an integer"):
        set_clones(sets, method="sets", max_bytes=1.5)
