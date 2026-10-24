"""Minimum spanning trees over cluster centres, a diagnostic for lineages.

Following Slingshot (Street et al. 2018, doi:10.1186/s12864-018-4772-0),
clusters are joined by a minimum spanning tree whose edge lengths compare
cluster means relative to their spread, and the paths from a root cluster to
the tree's leaves propose lineages. A tree must join every cluster, so
transcriptionally similar terminal clusters can be chained through one
another; the leaves are candidates for sinks to check against markers, not
discovered fates.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ClusterTree",
    "cluster_distances",
    "cluster_moments",
    "spanning_tree",
    "tree_lineages",
]


@dataclass(frozen=True, slots=True)
class ClusterTree:
    """A spanning tree over clusters and the lineages it proposes.

    Attributes:
        edges: Cluster index pairs, smaller index first, sorted.
        lengths: The distance of each edge.
        leaves: Clusters joined to exactly one other cluster, sorted.
        lineages: Cluster index paths from the root to each leaf other than
            the root, ordered by leaf; empty without a root.
    """

    edges: np.ndarray
    lengths: np.ndarray
    leaves: np.ndarray
    lineages: tuple[tuple[int, ...], ...]


def cluster_moments(
    blocks: Iterable[tuple[int, np.ndarray]],
    codes: np.ndarray,
    *,
    n_clusters: int,
    n_dims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each cluster's size, mean, and sample covariance from row blocks.

    ``codes`` gives each cell's cluster index. Each block's moments about its
    own cluster means are merged into the running moments with the pairwise
    update of Chan, Golub, and LeVeque (1983), so one pass is exact and stays
    stable. A cluster of one cell has a zero covariance.
    """
    cell_codes = np.asarray(codes, dtype=np.int64)
    if (
        cell_codes.ndim != 1
        or np.any(cell_codes < 0)
        or np.any(cell_codes >= n_clusters)
    ):
        raise ValueError("codes must give a cluster index for every cell")
    if np.unique(cell_codes).size != n_clusters:
        raise ValueError("Every cluster needs at least one cell")
    counts = np.zeros(n_clusters, dtype=np.int64)
    means = np.zeros((n_clusters, n_dims), dtype=np.float64)
    squares = np.zeros((n_clusters, n_dims, n_dims), dtype=np.float64)
    covered = 0
    for start, raw_block in blocks:
        block = np.asarray(raw_block, dtype=np.float64)
        if start != covered or block.ndim != 2 or block.shape[1] != n_dims:
            raise ValueError("Coordinate blocks must cover the cells in order")
        stop = start + block.shape[0]
        if stop > cell_codes.shape[0]:
            raise ValueError("Coordinate blocks extend beyond the cells")
        if not np.isfinite(block).all():
            raise ValueError("Coordinates must be finite")
        block_codes = cell_codes[start:stop]
        order = np.argsort(block_codes, kind="stable")
        present, first, sizes = np.unique(
            block_codes[order], return_index=True, return_counts=True
        )
        for cluster, offset, size in zip(present, first, sizes, strict=True):
            rows = block[order[offset : offset + size]]
            block_mean = rows.mean(axis=0)
            centered = rows - block_mean
            block_squares = centered.T @ centered
            previous = int(counts[cluster])
            total = previous + int(size)
            delta = block_mean - means[cluster]
            means[cluster] += delta * (size / total)
            squares[cluster] += block_squares + np.outer(delta, delta) * (
                previous * size / total
            )
            counts[cluster] = total
        covered = stop
    if covered != cell_codes.shape[0]:
        raise ValueError("Coordinate blocks must cover every cell")
    # The sums of squares become covariances in place, one cluster at a time,
    # so no second clusters-by-dimensions-squared array is held. A single
    # cell's sums of squares are already zero.
    for cluster in np.flatnonzero(counts > 1):
        squares[cluster] /= counts[cluster] - 1
    return counts, means, squares


def cluster_distances(means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Return Slingshot's distance between every pair of clusters.

    The distance between clusters i and j is
    ``(m_i - m_j)^T (S_i + S_j)^{-1} (m_i - m_j)``, a multivariate t-statistic
    of their means ``m`` and covariances ``S``. Where ``S_i + S_j`` is
    singular, its diagonal is used, as Street et al. suggest. A dimension
    without variance in either cluster then adds nothing where the means
    agree and makes the distance infinite where they differ, as the
    t-statistic does; two single-cell clusters at different positions are
    infinitely far apart, not identical. A pooled covariance is singular
    when its smallest eigenvalue is at most the dimension count times
    machine precision times its largest, the precision at which an inverse
    holds no information; such a matrix can still factor through rounding,
    so a Cholesky factorization alone would not detect it.
    """
    n_clusters, n_dims = means.shape
    tolerance = n_dims * np.finfo(np.float64).eps
    distances = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for first in range(n_clusters):
        for second in range(first + 1, n_clusters):
            difference = means[first] - means[second]
            pooled = covariances[first] + covariances[second]
            eigenvalues = np.linalg.eigvalsh(pooled)
            if eigenvalues[0] > tolerance * max(eigenvalues[-1], 0.0):
                factor = np.linalg.cholesky(pooled)
                solved = np.linalg.solve(factor, difference)
                value = float(solved @ solved)
            else:
                variance = np.diag(pooled)
                usable = variance > 0.0
                if np.any(difference[~usable] != 0.0):
                    value = np.inf
                else:
                    value = float(
                        np.sum(np.square(difference[usable]) / variance[usable])
                    )
            distances[first, second] = distances[second, first] = value
    return distances


def _prim_tree(distances: np.ndarray) -> list[tuple[int, int]]:
    """Return a minimum spanning tree of a complete graph by Prim's algorithm.

    Every pair of nodes is joined, including pairs at distance zero. The tree
    grows from node 0, and ties go to the edge of the smallest node index.
    When every node left is infinitely far from the tree, the first of them
    joins across an infinite edge, which the caller rejects.
    """
    n_nodes = distances.shape[0]
    in_tree = np.zeros(n_nodes, dtype=bool)
    in_tree[0] = True
    best = distances[0].astype(np.float64, copy=True)
    parent = np.zeros(n_nodes, dtype=np.int64)
    edges: list[tuple[int, int]] = []
    for _ in range(n_nodes - 1):
        candidates = np.where(in_tree, np.inf, best)
        node = int(np.argmin(candidates))
        if in_tree[node]:
            node = int(np.flatnonzero(~in_tree)[0])
        edges.append((min(node, int(parent[node])), max(node, int(parent[node]))))
        in_tree[node] = True
        closer = ~in_tree & (distances[node] < best)
        best[closer] = distances[node][closer]
        parent[closer] = node
    return edges


def spanning_tree(
    distances: np.ndarray,
    *,
    root: int | None = None,
    ends: Sequence[int] = (),
    labels: Sequence[Any] | None = None,
) -> ClusterTree:
    """Join clusters by a minimum spanning tree and trace lineages from a root.

    With ``ends``, the tree spans the other clusters first and each end
    cluster then joins its nearest other cluster, so ends are always leaves,
    as in Slingshot's supervised mode. With one other cluster, every end
    joins it. ``labels`` name the clusters in errors.

    Raises:
        ValueError: If an end repeats, the root is also an end, an index is
            out of range, every cluster is an end, or the tree can reach a
            cluster only across an infinite distance.
    """
    n_clusters = distances.shape[0]
    end_set = sorted(set(int(end) for end in ends))
    if len(end_set) != len(ends):
        raise ValueError("ends must not repeat a cluster")
    if any(end < 0 or end >= n_clusters for end in end_set):
        raise ValueError("ends must index clusters")
    if root is not None and not 0 <= root < n_clusters:
        raise ValueError("root must index a cluster")
    if root is not None and root in end_set:
        raise ValueError("The root cannot also be an end")
    inner = [cluster for cluster in range(n_clusters) if cluster not in end_set]
    if not inner:
        raise ValueError("At least one cluster must not be an end")
    edges = [
        (min(inner[a], inner[b]), max(inner[a], inner[b]))
        for a, b in _prim_tree(distances[np.ix_(inner, inner)])
    ]
    for end in end_set:
        nearest = inner[int(np.argmin(distances[end, inner]))]
        edges.append((min(end, nearest), max(end, nearest)))
    edges.sort()
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    lengths = distances[edge_array[:, 0], edge_array[:, 1]]
    infinite = np.flatnonzero(~np.isfinite(lengths))
    if infinite.size:
        pair = [int(index) for index in edge_array[infinite[0]]]
        first, second = pair if labels is None else [labels[index] for index in pair]
        raise ValueError(
            f"The tree can join clusters {first!r} and {second!r} only across an "
            "infinite distance: their means differ in a dimension where neither "
            "cluster varies. Merge clusters without spread, such as single-cell "
            "clusters, into neighbouring clusters"
        )
    leaves, lineages = tree_lineages(n_clusters, edge_array, root)
    return ClusterTree(
        edges=edge_array,
        lengths=np.asarray(lengths, dtype=np.float64),
        leaves=leaves,
        lineages=lineages,
    )


def tree_lineages(
    n_clusters: int,
    edges: np.ndarray,
    root: int | None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return a tree's leaves and, with a root, the paths from it to each leaf.

    Paths are ordered by leaf index, and a root that is itself a leaf starts
    paths but is not one. Without a root there are no paths.
    """
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    degree = np.bincount(edge_array.ravel(), minlength=n_clusters)
    leaves = np.flatnonzero(degree == 1).astype(np.int64)
    if root is None:
        return leaves, ()
    neighbours: dict[int, list[int]] = {cluster: [] for cluster in range(n_clusters)}
    for first, second in edge_array.tolist():
        neighbours[first].append(second)
        neighbours[second].append(first)
    parent = {root: -1}
    queue = [root]
    for cluster in queue:
        for neighbour in sorted(neighbours[cluster]):
            if neighbour not in parent:
                parent[neighbour] = cluster
                queue.append(neighbour)
    paths = []
    for leaf in leaves.tolist():
        if leaf == root:
            continue
        path = [leaf]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        paths.append(tuple(reversed(path)))
    return leaves, tuple(paths)
