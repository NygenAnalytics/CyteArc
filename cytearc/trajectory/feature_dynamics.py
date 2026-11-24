from typing import Any

import numpy as np
from numba import prange

import cytearc

from ..utils.arguments import integer_argument
from ..utils.arrays import rolling_window
from ..utils.logging import logger
from ..utils.numba import njit, numba_threads
from .parameters import resolve_aggregation_ann_params

__all__ = [
    "aggregate_feature_profiles",
    "knn_clustering",
    "nearest_feature_profiles",
    "scatter_feature_clusters",
    "validate_pseudotime_regressor",
]

# Scratch bytes per input value in aggregate_feature_profiles: the ordered
# copy, its smoothed replacement, and the temporary used by the standard
# deviation, with room for the boolean finiteness checks.
AGGREGATION_SCRATCH_ITEMSIZE = 3 * np.dtype(np.float64).itemsize


def validate_pseudotime_regressor(
    values: object,
    expected_size: int,
    pseudotime_key: str,
    cell_key: str,
    *,
    has_validity_column: bool,
) -> np.ndarray:
    try:
        pseudotime = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Pseudotime column '{pseudotime_key}' must be numeric"
        ) from exc

    if pseudotime.ndim != 1:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' must be one-dimensional"
        )
    if pseudotime.shape[0] != expected_size:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' has {pseudotime.shape[0]} values, "
            f"but cell_key '{cell_key}' selects {expected_size} cells"
        )
    if not np.isfinite(pseudotime).all():
        validity_key = f"{pseudotime_key}__valid"
        if has_validity_column:
            raise ValueError(
                f"Pseudotime column '{pseudotime_key}' contains unscored cells. "
                f"Use cell_key='{validity_key}' for downstream analysis"
            )
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' contains non-finite values"
        )
    if pseudotime.size < 2 or np.unique(pseudotime).size < 2:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' must contain at least two distinct values"
        )
    return pseudotime


def scatter_feature_clusters(
    n_features: int,
    feature_indices: np.ndarray,
    clusters: np.ndarray,
    unassigned_value: int,
) -> np.ndarray:
    feature_indices = np.asarray(feature_indices, dtype=int)
    clusters = np.asarray(clusters, dtype=int)
    if feature_indices.shape != clusters.shape:
        raise ValueError("Feature indices and cluster assignments are misaligned")
    if unassigned_value in clusters:
        raise ValueError("unassigned_value conflicts with an assigned feature cluster")
    values = np.full(n_features, unassigned_value, dtype=int)
    values[feature_indices] = clusters
    return values


def aggregate_feature_profiles(
    values: np.ndarray,
    ordering_indices: np.ndarray,
    feature_indices: np.ndarray,
    *,
    min_expression: float,
    window_size: int,
    n_bins: int,
    smooth: bool,
    z_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Order, smooth, filter, and bin normalized feature profiles.

    ``values`` is cells by features and is not modified. Besides the input,
    the calculation holds at most :data:`AGGREGATION_SCRATCH_ITEMSIZE` bytes
    per value, because smoothing and scaling reuse one ordered copy.
    """
    values = np.asarray(values, dtype=float)
    mean_expression = values.mean(axis=0)
    # Fancy indexing returns a new C-order array, which can be updated in
    # place and gives the column reductions a fixed summation order.
    ordered = np.ascontiguousarray(values[ordering_indices])
    finite_columns = np.isfinite(ordered).all(axis=0)
    if not finite_columns.all():
        invalid_columns = np.asarray(feature_indices)[~finite_columns]
        raise ValueError(
            f"Normalized features contain non-finite values: {invalid_columns.tolist()}"
        )
    if smooth:
        ordered = rolling_window(ordered, window_size)
        if not np.isfinite(ordered).all():
            raise ValueError("Smoothed feature profiles contain non-finite values")

    standard_deviation = ordered.std(axis=0)
    valid_features = (
        (mean_expression > min_expression)
        & np.isfinite(standard_deviation)
        & (standard_deviation > np.finfo(float).eps)
    )
    if z_scale:
        # Scale in place. Reducing only the retained columns keeps the
        # established floating-point summation order of their means.
        column_mean = np.zeros(ordered.shape[1], dtype=np.float64)
        if valid_features.any():
            column_mean[valid_features] = ordered[:, valid_features].mean(axis=0)
        ordered -= column_mean
        ordered /= np.where(valid_features, standard_deviation, 1.0)
    ordered[:, ~valid_features] = 0.0

    binned = np.stack(
        [
            bin_values.mean(axis=0)
            for bin_values in np.array_split(ordered, n_bins, axis=0)
        ],
        axis=1,
    )
    if not np.isfinite(binned).all():
        raise ValueError("Binned feature profiles contain non-finite values")
    return binned, valid_features


# The distances between feature profiles that ann_params' "space" names.
_FEATURE_SPACES = ("l2", "ip", "cosine")


@njit(cache=True, nogil=True)
def _unit_profiles(profiles: np.ndarray) -> np.ndarray:
    """Divide each profile by its Euclidean norm, summed over the bins in order."""
    n_profiles, n_bins = profiles.shape
    unit = np.empty_like(profiles)
    for row in range(n_profiles):
        total = 0.0
        for position in range(n_bins):
            total += profiles[row, position] * profiles[row, position]
        norm = np.sqrt(total)
        for position in range(n_bins):
            unit[row, position] = profiles[row, position] / norm
    return unit


@njit(cache=True, nogil=True, parallel=True)
def _nearest_profiles(
    profiles: np.ndarray,
    columns: np.ndarray,
    n_neighbors: int,
    inner_product: bool,
    n_workers: int,
) -> np.ndarray:
    """Return each profile's ``n_neighbors`` nearest other profiles, nearest first.

    The distance is the squared Euclidean distance, or one minus the inner
    product, summed over the bins in order, one multiplication and one
    addition at a time, so that it is the same on every platform; of two
    equally near profiles the earlier comes first. ``columns`` holds the
    profiles transposed, in C order, so that each bin's sums run along
    contiguous memory. Each row is computed alone, so the result does not
    depend on ``n_workers``.
    """
    n_profiles, n_bins = profiles.shape
    neighbors = np.empty((n_profiles, n_neighbors), dtype=np.int64)
    for worker in prange(n_workers):
        distances = np.empty(n_profiles, dtype=np.float64)
        nearest = np.empty(n_neighbors, dtype=np.float64)
        for row in range(worker, n_profiles, n_workers):
            distances[:] = 0.0
            for position in range(n_bins):
                value = profiles[row, position]
                column = columns[position]
                if inner_product:
                    for other in range(n_profiles):
                        distances[other] += column[other] * value
                else:
                    for other in range(n_profiles):
                        offset = column[other] - value
                        distances[other] += offset * offset
            if inner_product:
                for other in range(n_profiles):
                    distances[other] = 1.0 - distances[other]
            found = 0
            for other in range(n_profiles):
                if other == row:
                    continue
                distance = distances[other]
                if found == n_neighbors and not distance < nearest[n_neighbors - 1]:
                    continue
                slot = found if found < n_neighbors else n_neighbors - 1
                # A profile moves ahead only of farther ones, so equally near
                # profiles keep their order.
                while slot > 0 and distance < nearest[slot - 1]:
                    nearest[slot] = nearest[slot - 1]
                    neighbors[row, slot] = neighbors[row, slot - 1]
                    slot -= 1
                nearest[slot] = distance
                neighbors[row, slot] = other
                if found < n_neighbors:
                    found += 1
    return neighbors


def nearest_feature_profiles(
    profiles: np.ndarray,
    n_neighbors: int,
    *,
    space: str = "l2",
    nthreads: int = 1,
) -> np.ndarray:
    """Return each feature's nearest other features by its profile, nearest first.

    ``space`` names the distance: ``"l2"`` the squared Euclidean distance,
    ``"ip"`` one minus the inner product and ``"cosine"`` one minus the
    cosine similarity. Every distance is computed, each summed over the bins
    in order, so the neighbors are exact and the same on every platform and
    for any ``nthreads``; of two equally near features the earlier comes
    first.

    Args:
        profiles: The features' profiles, features by bins, finite.
        n_neighbors: The neighbors of each feature, at least 1 and fewer than
            the features.
        space: The distance between profiles.
        nthreads: The threads that compute the distances.

    Returns:
        The positions of each feature's neighbors, features by
        ``n_neighbors``, as int64.

    Raises:
        TypeError: If ``n_neighbors`` or ``nthreads`` is not an integer.
        ValueError: If ``profiles`` is not a finite matrix, ``n_neighbors``
            is out of range, ``nthreads`` is below 1, ``space`` is not one of
            the three, or a profile is all zeros under ``"cosine"``.
    """
    values = np.ascontiguousarray(profiles, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("profiles must be a matrix of features by bins")
    if not np.isfinite(values).all():
        raise ValueError("Feature profiles must be finite")
    n_profiles = values.shape[0]
    n_neighbors = integer_argument(n_neighbors, "n_neighbors")
    if not 1 <= n_neighbors < n_profiles:
        raise ValueError(f"n_neighbors must satisfy 1 <= n_neighbors < {n_profiles}")
    nthreads = integer_argument(nthreads, "nthreads", minimum=1)
    if space not in _FEATURE_SPACES:
        raise ValueError(f"space must be one of {_FEATURE_SPACES}, not {space!r}")
    if space == "cosine":
        zero = np.flatnonzero(~values.any(axis=1))
        if zero.size:
            raise ValueError(
                f"Cosine distances need profiles that are not all zeros; the "
                f"profiles at {zero[:5].tolist()} are"
            )
        values = _unit_profiles(values)
    workers = min(nthreads, n_profiles)
    with numba_threads(workers):
        return np.asarray(
            _nearest_profiles(
                values,
                np.ascontiguousarray(values.T),
                n_neighbors,
                space != "l2",
                workers,
            )
        )


def knn_clustering(
    d_array: "cytearc.matrix.ChunkedArray",
    n_neighbors: int,
    n_clusters: int,
    nthreads: int,
    ann_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Cluster pseudotime-ordered feature profiles into modules.

    Each feature's ``n_neighbors`` nearest other features, found exactly by
    :func:`nearest_feature_profiles` with the distance that
    ``ann_params["space"]`` names, form a graph whose Paris hierarchy is cut
    into ``n_clusters`` modules. The modules are numbered from 1 by the median
    bin of their features' peaks, and modules whose medians tie by the cut's
    order, the larger first. The modules therefore depend only on the
    profiles, not on the platform or ``nthreads``.
    """
    import pandas as pd
    from scipy.sparse import csr_matrix

    from ..utils.compute import compute_with_progress

    if len(d_array.shape) != 2:
        raise ValueError("d_array must be two-dimensional")
    n_genes = int(d_array.shape[0])
    if n_genes < 2:
        raise ValueError("At least two retained genes are required for clustering")
    if isinstance(n_neighbors, (bool, np.bool_)) or not isinstance(
        n_neighbors,
        (int, np.integer),
    ):
        raise TypeError("n_neighbors must be an integer")
    n_neighbors = int(n_neighbors)
    if not 1 <= n_neighbors < n_genes:
        raise ValueError(f"n_neighbors must satisfy 1 <= n_neighbors < {n_genes}")
    if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(
        n_clusters,
        (int, np.integer),
    ):
        raise TypeError("n_clusters must be an integer")
    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= n_genes:
        raise ValueError(f"n_clusters must satisfy 1 <= n_clusters <= {n_genes}")

    def make_knn_mat(profiles: np.ndarray, k: int) -> "csr_matrix":
        logger.debug("Pseudotime modules: finding feature neighbors")
        indices_mat = nearest_feature_profiles(
            profiles,
            k,
            space=str(resolved_ann_params["space"]),
            nthreads=nthreads,
        )
        return csr_matrix(
            (
                np.ones(indices_mat.shape[0] * indices_mat.shape[1]),
                (
                    np.repeat(
                        np.arange(indices_mat.shape[0]),
                        indices_mat.shape[1],
                    ),
                    indices_mat.flatten(),
                ),
            ),
            shape=(indices_mat.shape[0], indices_mat.shape[0]),
        )

    def make_clusters(matrix: "csr_matrix", n_cluster: int) -> np.ndarray:
        from ..clustering.paris import fit_paris_hierarchy, fixed_cut

        logger.debug("Pseudotime modules: clustering modules")
        hierarchy = fit_paris_hierarchy(
            matrix,
            nthreads=nthreads,
        )
        n_components = int(np.count_nonzero(hierarchy.synthetic_joins)) + 1
        if 1 < n_cluster < n_components:
            raise ValueError(
                f"The feature KNN graph has {n_components} disconnected "
                f"components, which cannot form n_clusters={n_cluster} modules. "
                "Increase n_neighbors or n_clusters."
            )
        return fixed_cut(hierarchy, n_cluster)

    def fix_cluster_order(
        data: "cytearc.matrix.ChunkedArray",
        clusters: np.ndarray,
        threads: int,
    ) -> np.ndarray:
        idxmax = compute_with_progress(
            data.argmax(axis=1),
            "Sorting clusters",
            threads,
        )
        # The cut labels its clusters by size, so a stable sort numbers
        # modules of equal median peaks the same way on every platform.
        cluster_medians = (
            pd.DataFrame([idxmax, clusters])
            .T.groupby(1)
            .median()[0]
            .sort_values(kind="stable")
        )
        return np.asarray(
            pd.Series(clusters)
            .replace(
                dict(
                    zip(
                        cluster_medians.index,
                        range(1, 1 + len(cluster_medians)),
                        strict=True,
                    )
                )
            )
            .values
        )

    resolved_ann_params = resolve_aggregation_ann_params(
        ann_params,
        dim=int(d_array.shape[1]),
    )
    profiles = compute_with_progress(d_array, None, nthreads)
    return fix_cluster_order(
        d_array,
        make_clusters(
            make_knn_mat(profiles, n_neighbors),
            n_clusters,
        ),
        nthreads,
    )
