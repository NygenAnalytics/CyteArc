from collections.abc import Iterator
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ..utils.logging import logger
from ..utils.progress import iter_progress


def make_digraph(d: np.ndarray) -> nx.DiGraph:
    """Convert a scipy linkage matrix into a directed tree graph.

    Each node records ``nleaves``, the leaves below an internal node and zero
    for a leaf.
    """
    graph = nx.DiGraph()
    node = d.shape[0] + 1
    for row_values in iter_progress(
        d,
        desc="Constructing graph from dendrogram",
        total=d.shape[0],
    ):
        row = row_values.astype(int)
        graph.add_node(node, nleaves=row[3])
        if row[0] <= d.shape[0]:
            graph.add_node(row[0], nleaves=0)
        if row[1] <= d.shape[0]:
            graph.add_node(row[1], nleaves=0)
        graph.add_edge(node, row[0])
        graph.add_edge(node, row[1])
        node += 1
    if graph.number_of_edges() != d.shape[0] * 2:
        logger.warning(
            "Number of edges in directed graph not twice the dendrogram shape"
        )
    return graph


def CoalesceTree(graph: nx.DiGraph, clusters: np.ndarray) -> nx.DiGraph:
    """Coalesce a hierarchy graph to each cluster's holding node."""

    def calc_steps_to_top(g: nx.DiGraph, c: np.ndarray) -> pd.Series:
        steps: dict[int, int] = {}
        for node in range(len(c)):
            steps[node] = 0
            queue = [node]
            while len(queue) > 0:
                for predecessor in g.predecessors(queue.pop(0)):
                    steps[node] += 1
                    queue.append(predecessor)
        return pd.Series(steps).sort_values()

    def iter_predecessors(g: nx.DiGraph, node: int) -> Iterator[int]:
        queue = [node]
        while len(queue) > 0:
            for predecessor in g.predecessors(queue.pop(0)):
                yield predecessor
                queue.append(predecessor)

    def aggregate_leaves(g: nx.DiGraph, node: int) -> list[int]:
        queue = [node]
        leaves: list[int] = []
        while len(queue) > 0:
            for successor in g.successors(queue.pop(0)):
                if g.nodes[successor]["nleaves"] == 0:
                    leaves.append(successor)
                else:
                    queue.append(successor)
        return leaves

    def get_holding_nodes(g: nx.DiGraph, c: np.ndarray) -> dict[int, int]:
        holding_nodes: dict[int, int] = {}
        steps = calc_steps_to_top(g, c)
        for cluster in iter_progress(
            set(c),
            desc="Identifying the top node for cluster",
        ):
            cluster_nodes = set(np.where(c == cluster)[0])
            n_cluster_nodes = len(cluster_nodes)
            start_node = steps.reindex(cluster_nodes).idxmin()
            if n_cluster_nodes == 1:
                holding_nodes[start_node] = cluster
                continue
            for predecessor in iter_predecessors(g, start_node):
                if g.nodes[predecessor]["nleaves"] >= n_cluster_nodes:
                    leaves = set(aggregate_leaves(g, predecessor))
                    if cluster_nodes.issubset(leaves):
                        if leaves != cluster_nodes or predecessor in holding_nodes:
                            raise ValueError(
                                "Cluster labels are incompatible with the hierarchy: "
                                f"cluster {cluster!r} is not monophyletic"
                            )
                        holding_nodes[predecessor] = cluster
                        break
            else:
                raise ValueError(
                    "Cluster labels are incompatible with the hierarchy: "
                    f"cluster {cluster!r} is not monophyletic"
                )
        return holding_nodes

    def aggregate_predecessors(g: nx.DiGraph, node: int) -> list[int]:
        return list(iter_predecessors(g, node))

    def make_subgraph(g: nx.DiGraph, nodes: dict[int, int]) -> nx.DiGraph:
        subgraph_nodes: list[int] = list(nodes.keys())
        for node in nodes:
            subgraph_nodes.extend(aggregate_predecessors(g, node))
        subgraph = nx.DiGraph(nx.subgraph(g, list(set(subgraph_nodes))))
        for node in nodes:
            subgraph.nodes[node]["partition_id"] = nodes[node]
        return subgraph

    return make_subgraph(graph, get_holding_nodes(graph, clusters))


def coalesce_dendrogram(dendrogram: np.ndarray, clusters: np.ndarray) -> nx.DiGraph:
    """Coalesce a linkage matrix to each cluster's holding node.

    The tree equals ``CoalesceTree(make_digraph(dendrogram), clusters)``: the
    same nodes, edges, and attributes, in the same order, with leaves keyed
    by NumPy integers as ``make_digraph`` keys them. No graph of every node
    is built. A cluster's holding node is its highest node whose leaves all
    carry its label, and the tree keeps the holding nodes and their
    ancestors.

    Raises:
        ValueError: If the dendrogram is not a binary tree whose rows merge
            two earlier nodes into one with their leaf count, or if a cluster
            is not the leaf set of one node.
    """
    linkage = np.asarray(dendrogram)
    if linkage.ndim != 2 or linkage.shape[1] != 4 or linkage.shape[0] == 0:
        raise ValueError("dendrogram must have shape (n_leaves - 1, 4)")
    rows = linkage.astype(int)
    n_leaves = rows.shape[0] + 1
    n_nodes = 2 * n_leaves - 1
    root = n_nodes - 1
    labels = np.asarray(clusters)
    if labels.shape != (n_leaves,):
        raise ValueError("clusters must hold one label per dendrogram leaf")
    children = rows[:, :2]
    merged = np.arange(n_leaves, n_nodes)
    if (
        np.any(children < 0)
        or np.any(children >= merged[:, np.newaxis])
        or np.any(np.bincount(children.ravel(), minlength=n_nodes) > 1)
    ):
        raise ValueError("dendrogram rows must merge two earlier nodes, each once")
    sizes = np.ones(n_nodes, dtype=np.int64)
    sizes[n_leaves:] = rows[:, 3]
    if np.any(sizes[n_leaves:] != sizes[children[:, 0]] + sizes[children[:, 1]]):
        raise ValueError("dendrogram leaf counts do not match its merges")
    parents = np.empty(n_nodes, dtype=np.int64)
    parents[children.ravel()] = np.repeat(merged, 2)
    parents[root] = root

    # Each node's leaves fill one range of the left-to-right leaf order. Its
    # start sums the leaf counts of the left siblings on its path to the root,
    # which pointer jumping adds up in logarithmically many passes.
    starts = np.zeros(n_nodes, dtype=np.int64)
    starts[children[:, 1]] = sizes[children[:, 0]]
    jumps = parents.copy()
    while np.any(jumps != root):
        starts += starts[jumps]
        jumps = jumps[jumps]
    order = np.empty(n_leaves, dtype=np.int64)
    order[starts[:n_leaves]] = np.arange(n_leaves)
    values, codes = np.unique(labels, return_inverse=True)
    ordered_codes = codes[order]
    label_changes = np.zeros(n_leaves, dtype=np.int64)
    label_changes[1:] = ordered_codes[1:] != ordered_codes[:-1]
    np.cumsum(label_changes, out=label_changes)
    pure = label_changes[starts + sizes - 1] == label_changes[starts]
    node_codes = ordered_codes[starts]
    highest = pure & ~pure[parents]
    highest[root] = pure[root]
    holders = np.flatnonzero(highest)
    holder_counts = np.bincount(node_codes[holders], minlength=values.size)
    holder_of_code = np.zeros(values.size, dtype=np.int64)
    holder_of_code[node_codes[holders]] = holders

    # Clusters are visited, and their holding nodes keyed, as CoalesceTree
    # visits and keys them, so the tree's sets iterate in the same order.
    code_of_value = {value: code for code, value in enumerate(values.tolist())}
    holding: dict[Any, Any] = {}
    for cluster in set(labels):
        code = code_of_value[cluster]
        if holder_counts[code] != 1:
            raise ValueError(
                "Cluster labels are incompatible with the hierarchy: "
                f"cluster {cluster!r} is not monophyletic"
            )
        node = int(holder_of_code[code])
        holding[np.int64(node) if node < n_leaves else node] = cluster

    # Holding nodes and their ancestors, the impure nodes once every cluster
    # has one holding node.
    kept = ~pure
    kept[holders] = True
    if 2 * int(np.count_nonzero(kept)) < n_nodes:
        # A subgraph of fewer than half the nodes iterates its node set. That
        # set is built from the holding nodes and then each one's ancestors;
        # an ancestor seen before adds nothing, and nor do the ones above it.
        parent_list = parents.tolist()
        listed: list[Any] = list(holding)
        seen = set(holding)
        for holder in holding:
            node = parent_list[int(holder)]
            while node not in seen:
                listed.append(node)
                seen.add(node)
                node = parent_list[node]
        # networkx deduplicates the nodes through a set, then keeps the set
        # of that list.
        deduplicated = list(set(listed))
        shown = set(node for node in deduplicated)
        nodes: list[Any] = list(shown)
    else:
        # A larger one iterates the full graph's nodes in their order: each
        # merged node, then the leaves it merges.
        full_order = np.full((n_leaves - 1, 3), -1, dtype=np.int64)
        full_order[:, 0] = merged
        leaf_children = children < n_leaves
        full_order[:, 1:][leaf_children] = children[leaf_children]
        full_order = full_order[full_order >= 0]
        nodes = [
            int(node) if node >= n_leaves else np.int64(node)
            for node in full_order[kept[full_order]].tolist()
        ]
    tree = nx.DiGraph()
    tree.add_nodes_from(nodes)
    tree.add_edges_from(
        (node, child)
        for node in nodes
        if node >= n_leaves
        for child in rows[int(node) - n_leaves, :2]
        if kept[child]
    )
    for node in nodes:
        tree.nodes[node]["nleaves"] = (
            rows[int(node) - n_leaves, 3] if node >= n_leaves else 0
        )
    for node, cluster in holding.items():
        tree.nodes[node]["partition_id"] = cluster
    return tree
