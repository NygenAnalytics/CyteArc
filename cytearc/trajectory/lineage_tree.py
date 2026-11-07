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
import pandas as pd

from ..utils.arrays import sort_categories
from .results import ClusterTree

__all__ = [
    "SpanningTree",
    "build_cluster_tree",
    "cluster_codes",
    "cluster_distances",
    "cluster_moments",
    "labelled_tree",
    "spanning_tree",
    "tree_ends",
    "tree_lineages",
]


@dataclass(frozen=True, slots=True)
class SpanningTree:
    """A spanning tree over cluster indices and the lineages it proposes.

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
) -> SpanningTree:
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
    return SpanningTree(
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


def cluster_codes(labels: Any) -> tuple[list[Any], np.ndarray]:
    """Return the clusters of the cells' labels in natural order, and each cell's cluster index.

    The clusters follow the order in which CyteArc's plots and tables show
    labels, so ``"c2"`` comes before ``"c10"``.

    Raises:
        ValueError: If ``labels`` does not hold one label per cell.
    """
    raw = np.asarray(labels)
    if raw.ndim != 1:
        raise ValueError("labels must hold one label per cell")
    # Factorize the cells in compiled code; only the distinct labels are
    # sorted and remapped in Python.
    first_codes, uniques = pd.factorize(raw, use_na_sentinel=False)
    native = [
        value.item() if isinstance(value, np.generic) else value
        for value in np.asarray(uniques, dtype=object)
    ]
    clusters = sort_categories(native)
    index = {cluster: code for code, cluster in enumerate(clusters)}
    remap = np.asarray([index[value] for value in native], dtype=np.int64)
    return clusters, remap[np.asarray(first_codes, dtype=np.int64)]


def tree_ends(
    clusters: Sequence[Any], root: Any, ends: Sequence[Any]
) -> tuple[int | None, list[int], tuple[Any, ...]]:
    """Return the root's cluster index, and the ends' indices and labels in cluster order.

    The tree does not depend on the order of the ends, so they are put in
    cluster order and any order of them builds one tree.

    Raises:
        ValueError: If the root or an end is not a cluster label.
    """

    def position(label: Any, name: str) -> int:
        for index, cluster in enumerate(clusters):
            if bool(cluster == label):
                return index
        raise ValueError(f"{name} {label!r} is not a cluster label")

    root_index = None if root is None else position(root, "root")
    pairs = sorted(
        ((position(end, "End"), end) for end in ends), key=lambda pair: pair[0]
    )
    return root_index, [index for index, _ in pairs], tuple(end for _, end in pairs)


def labelled_tree(
    clusters: Sequence[Any],
    sizes: np.ndarray,
    means: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    *,
    root: Any,
    ends: Sequence[Any],
) -> ClusterTree:
    """Name a tree over cluster indices by the cluster labels.

    ``edges`` holds cluster index pairs and ``lengths`` their distances; the
    leaves and, with a root, the lineages are traced from them as
    :func:`spanning_tree` traces them.
    """
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    root_index, _, _ = tree_ends(clusters, root, ())
    leaves, paths = tree_lineages(len(clusters), edge_array, root_index)
    return ClusterTree(
        clusters=tuple(clusters),
        sizes=np.asarray(sizes, dtype=np.int64),
        means=np.asarray(means, dtype=np.float64),
        edges=pd.DataFrame(
            {
                "source": [clusters[first] for first in edge_array[:, 0]],
                "target": [clusters[second] for second in edge_array[:, 1]],
                "length": np.asarray(lengths, dtype=np.float64),
            }
        ),
        root=root,
        ends=tuple(ends),
        leaves=tuple(clusters[leaf] for leaf in leaves),
        lineages=tuple(tuple(clusters[cluster] for cluster in path) for path in paths),
    )


def build_cluster_tree(
    coordinates: Any,
    labels: Any,
    *,
    root: Any = None,
    ends: list[Any] | None = None,
) -> ClusterTree:
    """Build the cluster tree that ``ds.trajectory.lineage_tree`` saves, without a store.

    Use it to see the tree before saving it, or on a store opened
    read-only, where ``lineage_tree`` cannot save one. The clusters are the
    distinct labels in their natural order, and the tree is built as
    ``lineage_tree`` builds it: the distance between two clusters is
    ``(m_i - m_j)^T (S_i + S_j)^{-1} (m_i - m_j)`` for their mean
    coordinates ``m`` and covariances ``S``, using the diagonal where
    ``S_i + S_j`` is singular, a minimum spanning tree joins the clusters,
    and the paths from ``root`` to the other leaves propose lineages
    (Street et al. 2018). ``ends`` makes clusters leaves: the tree first
    spans the others, then each end joins its nearest cluster. Given the
    coordinates and labels of a saved tree's cells, the tree equals the one
    that ``load_lineage_tree`` returns.

    Args:
        coordinates: Coordinates of the cells, cells by dimensions, such as
            the PCA that a graph was built on.
        labels: One cluster label per cell, such as the values of a
            clustering. Every cell needs a label.
        root: Cluster label that starts the lineages, or None for an
            unrooted tree without lineages.
        ends: Cluster labels to hold as leaves.

    Returns:
        The tree, as a :class:`~cytearc.trajectory.ClusterTree`.

    Raises:
        TypeError: If ``coordinates`` are not numbers, ``ends`` is not a
            list, or the root or an end is not a scalar label.
        ValueError: If ``coordinates`` are not finite cells by dimensions,
            ``labels`` does not hold one label for each of their rows, a
            label is missing, fewer than two clusters remain, the root or an
            end is not a cluster label or repeats, the root is also an end,
            every cluster is an end, or the tree can join two clusters only
            across an infinite distance.
    """
    from .artifacts import validate_lineage_tree_parameters

    if ends is not None and not isinstance(ends, list):
        raise TypeError("ends must be a list of cluster labels or None")
    validated = validate_lineage_tree_parameters(
        {"root": root, "ends": [] if ends is None else ends}
    )
    rows = np.asarray(coordinates)
    if rows.dtype.kind not in "iuf":
        raise TypeError("coordinates must hold numbers")
    if rows.ndim != 2 or rows.shape[1] < 1:
        raise ValueError("coordinates must be cells by dimensions")
    rows = rows.astype(np.float64, copy=False)
    if not np.isfinite(rows).all():
        raise ValueError("Coordinates must be finite")
    cell_labels = np.asarray(labels)
    if cell_labels.ndim != 1 or cell_labels.shape[0] != rows.shape[0]:
        raise ValueError("labels must hold one label for each row of coordinates")
    if bool(np.any(pd.isna(cell_labels))):
        raise ValueError(
            "Every cell needs a label; leave out the cells without one, from "
            "both coordinates and labels"
        )
    clusters, codes = cluster_codes(cell_labels)
    if len(clusters) < 2:
        raise ValueError("A lineage tree needs at least two clusters")
    root_index, end_indices, ordered_ends = tree_ends(
        clusters, validated["root"], validated["ends"]
    )
    counts, means, covariances = cluster_moments(
        [(0, rows)], codes, n_clusters=len(clusters), n_dims=rows.shape[1]
    )
    tree = spanning_tree(
        cluster_distances(means, covariances),
        root=root_index,
        ends=end_indices,
        labels=clusters,
    )
    return labelled_tree(
        clusters,
        counts,
        means,
        tree.edges,
        tree.lengths,
        root=validated["root"],
        ends=ordered_ends,
    )
