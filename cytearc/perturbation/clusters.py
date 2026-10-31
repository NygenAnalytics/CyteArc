"""Clusters of perturbations whose profiles change expression alike.

Replogle et al. (2022) grouped the perturbations of a genome-scale screen by
the correlation of their mean expression profiles and recovered protein
complexes and pathways. These functions cluster the profiles of
:mod:`cytearc.perturbation.profiles` the same way:

- The distance between two profiles is 1 minus their Pearson correlation over
  the features, which ignores how strongly each perturbation acts, as
  Replogle et al. chose it, or their Euclidean distance.
- HDBSCAN (Campello et al. 2013), scikit-learn's implementation, finds the
  clusters on the precomputed distances and leaves the perturbations it
  cannot place in a cluster unassigned, -1. Its ``min_samples`` counts the
  point itself, so the ``min_samples=1`` of the ``hdbscan`` package, which
  Replogle et al. ran, is 2 here.
- The leaf order of the average-linkage tree of the same distances, SciPy's
  plain order without optimal leaf ordering, orders the perturbations of a
  heatmap, so that each subtree's perturbations sit together.
- A UMAP layout of each perturbation's nearest perturbations by the same
  distances places them in two dimensions, through CyteArc's UMAP layout.

These functions read profiles that a caller passes and never touch storage.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cytearc.utils.arguments import integer_argument

__all__ = [
    "METRICS",
    "SELECTIONS",
    "ClusterArrays",
    "PerturbationClusters",
    "cluster_memory_bytes",
    "cluster_profiles",
    "cluster_settings",
    "embed_profiles",
    "hdbscan_clusters",
    "leaf_order",
    "nearest_profiles",
    "perturbation_names",
    "profile_distances",
]

#: The distances between profiles.
METRICS = ("correlation", "euclidean")
#: How HDBSCAN selects clusters from its tree: excess of mass or leaves.
SELECTIONS = ("eom", "leaf")
#: The largest seed of the layout: scikit-learn's PCA, which starts it,
#: accepts 32-bit unsigned random states only.
MAX_SEED = 2**32 - 1

# The layout's settings, umap-learn's defaults for a few thousand points: the
# nearest perturbations of each, the spread and smallest distance of the
# layout, and its epochs.
_EMBEDDING_NEIGHBORS = 15
_EMBEDDING_SPREAD = 1.0
_EMBEDDING_MIN_DIST = 0.1
_EMBEDDING_EPOCHS = 500
# The starting layout spans this many units from the origin, with noise of
# this standard deviation that separates equal profiles, as umap-learn starts.
_START_SPAN = 10.0
_START_NOISE = 1e-4
# The rows of distances that one step of the neighbour search sorts at once.
_SORT_ROWS = 256
# The columns of the random matrices of scikit-learn's randomized PCA of the
# starting layout: its 2 components and 10 more.
_PCA_SKETCH_COLUMNS = 12


@dataclass(frozen=True, slots=True)
class ClusterArrays:
    """One entry per perturbation, in the order of the profiles.

    Attributes:
        cluster: Each perturbation's cluster, -1 when HDBSCAN leaves it
            unassigned.
        probability: The strength of its membership, from 0 to 1, 0 when
            unassigned.
        order: The perturbations' positions in the leaf order of average
            linkage.
        embedding: The two-dimensional layout, perturbations x 2, float32.
    """

    cluster: np.ndarray
    probability: np.ndarray
    order: np.ndarray
    embedding: np.ndarray


def cluster_settings(
    metric: object,
    min_cluster_size: object,
    min_samples: object,
    selection: object,
    seed: object,
    perturbations: object,
) -> tuple[int, int, int, list[str] | None]:
    """Check the settings of perturbation clusters, as ``cluster`` takes them.

    The pipeline recipe checks the same settings before its run starts.
    Whether each of ``perturbations`` has a profile is checked against the
    profiles.

    Returns:
        ``min_cluster_size``, ``min_samples`` and ``seed`` as Python
        integers, and the names of ``perturbations`` as a list, or None.

    Raises:
        TypeError: If a number is not an integer, or ``perturbations`` is
            neither a sequence of names nor None.
        ValueError: If ``metric`` or ``selection`` is unknown, a number is out
            of range, or ``perturbations`` names fewer than 3 distinct
            perturbations or fewer than ``min_samples``.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be 'correlation' or 'euclidean', not {metric!r}")
    cluster_size = integer_argument(min_cluster_size, "min_cluster_size", minimum=2)
    core_size = integer_argument(min_samples, "min_samples", minimum=1)
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be 'eom' or 'leaf', not {selection!r}")
    resolved_seed = integer_argument(seed, "seed", minimum=0, maximum=MAX_SEED)
    requested = None if perturbations is None else perturbation_names(perturbations)
    if requested is not None:
        # The clustering takes each named perturbation once.
        n_named = len(set(requested))
        if n_named < 3:
            raise ValueError(
                "Clustering needs at least 3 distinct perturbations, but "
                f"perturbations names {n_named}"
            )
        if core_size > n_named:
            raise ValueError(
                f"min_samples={core_size} must be at most the {n_named} "
                "perturbations clustered"
            )
    return cluster_size, core_size, resolved_seed, requested


def perturbation_names(perturbations: object) -> list[str]:
    """Check the perturbations to cluster, and return their names as a list.

    Raises:
        TypeError: If ``perturbations`` is not a sequence of names.
    """
    if isinstance(perturbations, str | bytes) or not isinstance(
        perturbations, Iterable
    ):
        raise TypeError(
            "perturbations must be a sequence of perturbation names or None"
        )
    requested = list(perturbations)
    if not all(isinstance(name, str) for name in requested):
        raise TypeError(
            "perturbations must be a sequence of perturbation names or None"
        )
    return requested


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Return each row centred and scaled to length 1, or 0 when it is constant.

    The dot product of two such rows is the Pearson correlation of the rows.
    """
    data = np.asarray(values, dtype=np.float64)
    if not data.shape[1]:
        return np.zeros(data.shape)
    rows = data - data.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->i", rows, rows))
    # A row of one value has no correlation, whatever the rounding of its
    # mean leaves.
    norms[data.max(axis=1) == data.min(axis=1)] = 0.0
    np.divide(rows, norms[:, None], out=rows, where=norms[:, None] > 0)
    rows[norms == 0] = 0.0
    return rows


def profile_distances(values: np.ndarray, metric: str) -> np.ndarray:
    """Return the distances between every two profiles.

    Args:
        values: The profiles, perturbations x features.
        metric: ``"correlation"`` for 1 minus the Pearson correlation of two
            profiles over the features, or ``"euclidean"``. A profile with one
            value in every feature correlates with no other, so it sits at
            distance 1 from each.

    Returns:
        The distances, perturbations x perturbations, as float64: symmetric,
        with a zero diagonal.

    Raises:
        ValueError: If ``metric`` is unknown, or the profiles are not
            finite and two-dimensional.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("The profiles must be finite, perturbations x features")
    distances: np.ndarray
    if metric == "correlation":
        rows = _unit_rows(data)
        distances = rows @ rows.T
        np.subtract(1.0, distances, out=distances)
        np.clip(distances, 0.0, 2.0, out=distances)
    elif metric == "euclidean":
        squares = np.einsum("ij,ij->i", data, data)
        distances = data @ data.T
        distances *= -2.0
        distances += squares[:, None]
        distances += squares[None, :]
        np.maximum(distances, 0.0, out=distances)
        np.sqrt(distances, out=distances)
    else:
        raise ValueError(f"metric must be 'correlation' or 'euclidean', not {metric!r}")
    # Equal in both directions to the last bit, with a zero diagonal.
    distances += distances.T
    distances *= 0.5
    np.fill_diagonal(distances, 0.0)
    return distances


def hdbscan_clusters(
    distances: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster perturbations by HDBSCAN on their precomputed distances.

    scikit-learn's ``HDBSCAN`` runs with ``metric="precomputed"`` and its
    default ``alpha`` of 1 and ``allow_single_cluster=False``. Its
    ``min_samples`` counts the point itself, so the ``hdbscan`` package's
    ``min_samples=1``, which Replogle et al. ran, is 2 here.

    Args:
        distances: The distances, perturbations x perturbations, symmetric
            with a zero diagonal.
        min_cluster_size: The fewest perturbations of a cluster, at least 2.
        min_samples: The perturbations, itself included, within a
            perturbation's core distance, at least 1 and at most the
            perturbations.
        selection: ``"eom"`` to select clusters by excess of mass, or
            ``"leaf"`` to take the leaves of the cluster tree.

    Returns:
        Each perturbation's cluster, -1 when unassigned, and the strength of
        its membership.
    """
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="precomputed",
        cluster_selection_method=selection,
        copy=True,
    ).fit(np.asarray(distances, dtype=np.float64))
    return (
        np.asarray(model.labels_, dtype=np.int64),
        np.asarray(model.probabilities_, dtype=np.float64),
    )


def leaf_order(distances: np.ndarray) -> np.ndarray:
    """Return the leaf order of the average-linkage tree of the distances.

    The order is SciPy's ``leaves_list`` of ``linkage(method="average")``, as
    a clustered heatmap orders its rows. SciPy's optimal leaf ordering is not
    applied: it holds several copies of the distances and its time grows with
    the cube of the perturbations.

    Args:
        distances: The distances, perturbations x perturbations, symmetric
            with a zero diagonal.

    Returns:
        The perturbations' positions, in leaf order.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    tree = linkage(
        squareform(np.asarray(distances, dtype=np.float64), checks=False),
        method="average",
    )
    return np.asarray(leaves_list(tree), dtype=np.int64)


def nearest_profiles(
    distances: np.ndarray, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return each perturbation's nearest other perturbations, nearest first.

    Of two equally near perturbations the first in order comes first.

    Args:
        distances: The distances, perturbations x perturbations.
        n_neighbors: The neighbours of each perturbation, fewer than the
            perturbations.

    Returns:
        The neighbours' positions and their distances, perturbations x
        ``n_neighbors``.
    """
    n = len(distances)
    indices = np.empty((n, n_neighbors), dtype=np.int64)
    nearest = np.empty((n, n_neighbors), dtype=np.float32)
    for start in range(0, n, _SORT_ROWS):
        rows = np.arange(start, min(start + _SORT_ROWS, n))
        band = np.array(distances[rows], dtype=np.float64)
        band[np.arange(len(rows)), rows] = np.inf
        order = np.argsort(band, axis=1, kind="stable")[:, :n_neighbors]
        indices[rows] = order
        nearest[rows] = np.take_along_axis(band, order, axis=1)
    return indices, nearest


def embed_profiles(
    values: np.ndarray, distances: np.ndarray, *, metric: str, seed: int
) -> np.ndarray:
    """Lay the perturbations out in two dimensions by UMAP on their distances.

    Each perturbation's 15 nearest perturbations by ``distances``, or all
    others when there are fewer, form the graph whose fuzzy union CyteArc's
    UMAP layout optimizes, with umap-learn's default spread of 1, smallest
    distance of 0.1 and 500 epochs, on one thread so that the layout depends
    only on ``seed``. The layout starts from the first two principal
    components of the profiles as the metric sees them, centred and scaled
    rows for ``"correlation"``, whose Euclidean distances follow 1 - r, scaled
    to span 10 units with noise of standard deviation 1e-4, as umap-learn
    starts from a PCA.

    Args:
        values: The profiles, perturbations x features.
        distances: Their distances, perturbations x perturbations.
        metric: The metric of the distances.
        seed: The seed of the starting noise and of the layout.

    Returns:
        The layout, perturbations x 2, as float32.
    """
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import PCA

    from ..embeddings.umap import fit_transform
    from ..neighbors.graph import build_connectivity_arrays

    n = len(distances)
    indices, nearest = nearest_profiles(distances, min(_EMBEDDING_NEIGHBORS, n - 1))
    edges, weights = build_connectivity_arrays(
        indices, nearest, local_connectivity=1.0, bandwidth=1.0
    )
    graph = coo_matrix(
        (weights, (edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64))),
        shape=(n, n),
    )
    rows = (
        _unit_rows(values)
        if metric == "correlation"
        else np.asarray(values, dtype=np.float64)
    )
    # Equal profiles have no variance for the PCA to explain.
    with np.errstate(divide="ignore", invalid="ignore"):
        start = PCA(n_components=2, random_state=seed).fit_transform(rows)
    span = float(np.abs(start).max())
    if span > 0:
        start *= _START_SPAN / span
    start += np.random.default_rng(seed).normal(scale=_START_NOISE, size=start.shape)
    embedding, _, _ = fit_transform(
        graph=graph,
        ini_embed=np.ascontiguousarray(start, dtype=np.float32),
        spread=_EMBEDDING_SPREAD,
        min_dist=_EMBEDDING_MIN_DIST,
        n_epochs=_EMBEDDING_EPOCHS,
        random_seed=seed,
        repulsion_strength=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5.0,
        densmap_kwds={},
        parallel=False,
        nthreads=1,
        verbose=False,
    )
    return np.asarray(embedding, dtype=np.float32)


def cluster_memory_bytes(n_perturbations: int, n_features: int) -> int:
    """Estimate the bytes that :func:`cluster_profiles` holds at its peak.

    The float64 profiles stay while each step runs. The distances take the
    centred profiles and a copy of the distances that makes them symmetric;
    then, beside the distances, HDBSCAN holds its copy of them with the core
    distances and its tree, measured at 2.2 copies on Python 3.13 and 2.6
    on 3.15 and budgeted at 3.2, the linkage holds the
    condensed distances and its copy, and the layout's PCA up to 5.3 copies
    of the profiles, measured, with scikit-learn's randomized solver's
    float64 matrices of features x 12, its 2 components and 10 more,
    measured at 3.1 copies beside 2 copies of the profiles and budgeted at
    4; they outweigh the profiles below 12 perturbations.

    Args:
        n_perturbations: The perturbations clustered.
        n_features: The features of each profile.

    Returns:
        The estimated peak in bytes.
    """
    pairs = 8 * n_perturbations * n_perturbations
    profiles = 8 * n_perturbations * n_features
    sketches = 8 * _PCA_SKETCH_COLUMNS * n_features
    return max(
        2 * pairs + 2 * profiles,
        9 * pairs // 2 + profiles,
        pairs + 7 * profiles + 4 * sketches,
    )


def cluster_profiles(
    values: np.ndarray,
    *,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
    selection: str,
    seed: int,
) -> ClusterArrays:
    """Cluster the perturbations, order them and lay them out.

    Args:
        values: The profiles, perturbations x features.
        metric: ``"correlation"`` or ``"euclidean"``.
        min_cluster_size: The fewest perturbations of a cluster.
        min_samples: The perturbations, itself included, within a
            perturbation's core distance.
        selection: ``"eom"`` or ``"leaf"``.
        seed: The seed of the layout.

    Returns:
        The clusters, their strengths, the leaf order and the layout.

    Raises:
        ValueError: If there are fewer than 3 perturbations or 2 features.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 2:
        raise ValueError(
            "Clustering needs at least 3 perturbations and 2 features, not "
            f"{data.shape}"
        )
    distances = profile_distances(data, metric)
    cluster, probability = hdbscan_clusters(
        distances,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        selection=selection,
    )
    order = leaf_order(distances)
    embedding = embed_profiles(data, distances, metric=metric, seed=seed)
    return ClusterArrays(
        cluster=cluster, probability=probability, order=order, embedding=embedding
    )


@dataclass(frozen=True, slots=True)
class PerturbationClusters:
    """Clusters of perturbations whose profiles change expression alike.

    Attributes:
        table: One row per clustered perturbation, indexed by its name, in
            the order of the profiles, with ``cluster`` (-1 when HDBSCAN
            leaves it unassigned), ``probability`` (the strength of its
            membership, 0 when unassigned), ``n_cells`` (the cells of its
            profile), and ``x`` and ``y``, its place in the two-dimensional
            layout.
        order: The perturbations' names in the leaf order of average
            linkage, the order of a heatmap's rows and columns.
        distances: The distances between the profiles, perturbations x
            perturbations, in the order of ``table``; with
            ``metric="correlation"``, ``1 - distances`` holds their Pearson
            correlations.
        metric: ``"correlation"`` or ``"euclidean"``.
        min_cluster_size: The fewest perturbations of a cluster.
        min_samples: HDBSCAN's ``min_samples``, which counts the point
            itself.
        selection: ``"eom"`` or ``"leaf"``.
        seed: The seed of the layout.
        from_assay: The expression assay of the profiles.
    """

    table: pd.DataFrame = field(repr=False)
    order: np.ndarray = field(repr=False)
    distances: pd.DataFrame = field(repr=False)
    metric: str
    min_cluster_size: int
    min_samples: int
    selection: str
    seed: int
    from_assay: str
