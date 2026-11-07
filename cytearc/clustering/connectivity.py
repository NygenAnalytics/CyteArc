"""The graph weight between clusters of cells.

A cell graph joins two clusters through the edges between their cells. A
pair's raw weight sums the weights of those edges, each undirected edge once,
and its normalized weight divides the raw weight by the geometric mean of the
two clusters' incident weights, the summed weighted degrees of their cells.
These functions work on a sparse graph and each cell's cluster code and never
touch storage: ``DataStore.clusters.connectivity`` returns the tables, and
``plots.cluster_connectivity`` draws them on a layout.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

__all__ = [
    "ClusterConnectivity",
    "check_edge_filters",
    "cluster_connectivity_tables",
    "filter_cluster_edges",
    "label_codes",
]


@dataclass(frozen=True, slots=True, eq=False)
class ClusterConnectivity:
    """The graph weight between each pair of clusters.

    Attributes:
        nodes: One row per cluster, in the order of its categories:
            ``category``, the cluster's label; ``nCells``, its number of
            cells; and ``incidentWeight``, the summed weighted degree of its
            cells, edges within the cluster included.
        edges: One row per pair of clusters that the graph joins: ``source``
            and ``target``, the clusters' labels, the source first in the
            order of ``nodes``; ``rawWeight``, the summed weight of the edges
            between their cells, each undirected edge once; and
            ``normalizedWeight``, ``rawWeight`` divided by
            ``sqrt(incidentWeight[source] * incidentWeight[target])``, or 0
            when either incident weight is 0. Rows sort by source, then
            target, in the order of ``nodes``.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame


def label_codes(labels: Any, categories: Sequence[Any]) -> NDArray[np.intp]:
    """Return the position of each label among ``categories``.

    Raises:
        ValueError: If a label is not one of the categories.
    """
    category_index = {category: index for index, category in enumerate(categories)}
    try:
        return np.fromiter(
            (category_index[value] for value in labels),
            dtype=np.intp,
            count=len(labels),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Could not map every group value to a category") from exc


def check_edge_filters(
    minimum_edge_weight: float, max_edges_per_node: int | None
) -> None:
    """Check the filters of :func:`filter_cluster_edges` before anything is read.

    Raises:
        ValueError: If ``minimum_edge_weight`` is negative or not finite, or
            ``max_edges_per_node`` is neither None nor a non-negative integer.
    """
    if not np.isfinite(minimum_edge_weight) or minimum_edge_weight < 0:
        raise ValueError("minimum_edge_weight must be finite and non-negative")
    if max_edges_per_node is not None and (
        isinstance(max_edges_per_node, bool)
        or not isinstance(max_edges_per_node, int)
        or max_edges_per_node < 0
    ):
        raise ValueError("max_edges_per_node must be a non-negative integer or None")


def _checked_graph(graph: Any, n_cells: int) -> sparse.csr_matrix:
    """Return the graph's weights as float64 CSR, summed, without self-loops.

    Raises:
        TypeError: If ``graph`` is not a scipy sparse matrix.
        ValueError: If its shape is not one row and column per cell, or a
            weight is negative or not finite.
    """
    if not sparse.issparse(graph):
        raise TypeError("graph must be a scipy sparse matrix")
    if graph.shape != (n_cells, n_cells):
        raise ValueError(
            "Graph shape must match the selected cells; "
            f"expected {(n_cells, n_cells)}, got {graph.shape}"
        )
    matrix = graph.tocsr(copy=True).astype(np.float64, copy=False)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Graph contains non-finite edge weights")
    if (matrix.data < 0).any():
        raise ValueError("Graph edge weights must be non-negative")
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


def cluster_connectivity_tables(
    graph: Any,
    codes: Any,
    categories: Sequence[Any],
) -> ClusterConnectivity:
    """Sum a symmetric cell graph's weight between each pair of clusters.

    Duplicate entries of ``graph`` are summed and self-loops left out. The
    graph must be symmetric, as ``DataStore.graph.load(symmetric=True)``
    returns it: each undirected edge is counted once, from the upper
    triangle, and a cell's weighted degree sums its row.

    Args:
        graph: Sparse cell graph, one row and column per cell.
        codes: Each cell's cluster, its position among ``categories``.
        categories: The clusters' labels, each once, in the order of the
            ``nodes`` table.

    Returns:
        The clusters and every pair that the graph joins, as
        :class:`ClusterConnectivity`.

    Raises:
        TypeError: If ``graph`` is not a scipy sparse matrix or ``codes`` are
            not integers.
        ValueError: If ``codes`` are not one per graph row, a code is not a
            position among ``categories``, a category repeats, or a weight
            is negative or not finite.
    """
    cluster_codes = np.asarray(codes)
    if cluster_codes.ndim != 1:
        raise ValueError("codes must be one-dimensional")
    if cluster_codes.dtype.kind not in "iu":
        raise TypeError("codes must be integers")
    n_categories = len(categories)
    category_values = np.empty(n_categories, dtype=object)
    category_values[:] = list(categories)
    if len(set(category_values.tolist())) != n_categories:
        raise ValueError("categories must name each cluster once")
    if len(cluster_codes) and (
        int(cluster_codes.min()) < 0 or int(cluster_codes.max()) >= n_categories
    ):
        raise ValueError("codes must be positions among the categories")
    cluster_codes = cluster_codes.astype(np.intp, copy=False)
    matrix = _checked_graph(graph, len(cluster_codes))

    incident_by_cell = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    incident_by_category = np.bincount(
        cluster_codes,
        weights=incident_by_cell,
        minlength=n_categories,
    )
    nodes = pd.DataFrame(
        {
            "category": category_values,
            "nCells": np.bincount(cluster_codes, minlength=n_categories),
            "incidentWeight": incident_by_category,
        }
    )

    upper = sparse.triu(matrix, k=1, format="coo")
    source_codes = cluster_codes[upper.row]
    target_codes = cluster_codes[upper.col]
    intercluster = source_codes != target_codes
    first = np.minimum(source_codes[intercluster], target_codes[intercluster])
    second = np.maximum(source_codes[intercluster], target_codes[intercluster])
    pair_keys = first.astype(np.int64) * n_categories + second
    unique_keys, inverse = np.unique(pair_keys, return_inverse=True)
    # bincount gives integers when it has no edge to weigh.
    raw_weights = np.bincount(
        inverse,
        weights=np.asarray(upper.data[intercluster], dtype=np.float64),
        minlength=len(unique_keys),
    ).astype(np.float64, copy=False)
    source = (unique_keys // n_categories).astype(np.intp, copy=False)
    target = (unique_keys % n_categories).astype(np.intp, copy=False)
    denominator = np.sqrt(incident_by_category[source] * incident_by_category[target])
    normalized = np.divide(
        raw_weights,
        denominator,
        out=np.zeros_like(raw_weights),
        where=denominator > 0,
    )
    edges = pd.DataFrame(
        {
            "source": category_values[source],
            "target": category_values[target],
            "rawWeight": raw_weights,
            "normalizedWeight": normalized,
        }
    )
    return ClusterConnectivity(nodes=nodes, edges=edges)


def filter_cluster_edges(
    connectivity: ClusterConnectivity,
    *,
    minimum_edge_weight: float,
    max_edges_per_node: int | None,
) -> ClusterConnectivity:
    """Keep the pairs of clusters whose normalized weight reaches a floor, up to a number per cluster.

    A pair is kept when its ``normalizedWeight`` is at least
    ``minimum_edge_weight``. With ``max_edges_per_node``, the kept pairs are
    then taken from the largest normalized weight down, ties broken by the
    larger raw weight and then by the order of the nodes, and a pair is
    skipped once either of its clusters holds that many. The nodes do not
    change.

    Raises:
        ValueError: If a filter is invalid, as :func:`check_edge_filters`
            checks.
    """
    check_edge_filters(minimum_edge_weight, max_edges_per_node)
    edges = connectivity.edges
    positions = pd.Index(connectivity.nodes["category"].to_numpy(dtype=object))
    source = positions.get_indexer(edges["source"].to_numpy(dtype=object))
    target = positions.get_indexer(edges["target"].to_numpy(dtype=object))
    raw_weights = edges["rawWeight"].to_numpy(dtype=np.float64)
    normalized_weights = edges["normalizedWeight"].to_numpy(dtype=np.float64)

    keep = normalized_weights >= minimum_edge_weight
    if max_edges_per_node is not None and keep.any():
        ranked = np.lexsort((target, source, -raw_weights, -normalized_weights))
        degree = np.zeros(len(positions), dtype=np.intp)
        capped = np.zeros(len(edges), dtype=bool)
        for edge_index in ranked[keep[ranked]]:
            left = source[edge_index]
            right = target[edge_index]
            if (
                degree[left] >= max_edges_per_node
                or degree[right] >= max_edges_per_node
            ):
                continue
            capped[edge_index] = True
            degree[left] += 1
            degree[right] += 1
        keep = capped
    kept = np.flatnonzero(keep)
    ordered = kept[np.lexsort((target[kept], source[kept]))]
    return ClusterConnectivity(
        nodes=connectivity.nodes,
        edges=edges.iloc[ordered].reset_index(drop=True),
    )
