"""k-means++ seeding of the Harmony2 clusters on unit-normalised cells.

Harmony2 replaces the k-means initialisation of the 2019 method with k-means++
seeding (Arthur and Vassilvitskii, 2007): the first seed is a uniformly random
cell, and every further seed is a cell drawn with probability proportional to its
squared distance from the nearest seed chosen so far. The clustering compares
unit vectors, so the distances are those between unit-normalised cells, for which
``||a - b||^2 = 2 (1 - a.b)`` is the clustering's own distance term.

The draw is an exponential race: with ``e_j ~ Exp(1)``, the smallest
``e_j / D2_j`` falls on cell ``j`` with probability ``D2_j / sum_l D2_l``. The
seeding keeps one vector of nearest squared distances, so it costs ``O(K N d)``
time and ``O(N)`` memory. The seeds' unit vectors are the initial centroids; no
Lloyd iterations follow, because the clustering that comes next is itself a soft
spherical k-means.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import numpy as np


def next_seed(
    squared_distances: np.ndarray,
    chosen: list[int],
    race: np.ndarray,
    keys: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Draw the next k-means++ seed by an exponential race over squared distances to the nearest seed.

    A cell that is already a seed, or that coincides with one, is never drawn
    while another cell has a positive distance. When no unchosen cell has one,
    the seed is uniform over the unchosen cells.

    Args:
        squared_distances: ``(cells,)`` squared distance of each cell to its
            nearest seed.
        chosen: Indices of the seeds chosen so far.
        race: ``(cells,)`` buffer that receives one exponential draw per cell.
        keys: ``(cells,)`` buffer that receives the race keys.
        rng: The fit's random generator.

    Returns:
        The index of the next seed.
    """
    rng.standard_exponential(out=race)
    keys.fill(np.inf)
    np.divide(race, squared_distances, out=keys, where=squared_distances > 0)
    keys[chosen] = np.inf
    seed = int(np.argmin(keys))
    if keys[seed] == np.inf:
        race[chosen] = np.inf
        seed = int(np.argmin(race))
    return seed


def _squared_distances_to(
    cells: np.ndarray,
    inverse_norms: np.ndarray,
    unit_squares: np.ndarray,
    seed: int,
    out: np.ndarray,
) -> None:
    """Write ``||ztilde_i - ztilde_seed||^2`` of every cell into ``out``."""
    center = cells[seed] * inverse_norms[seed]
    np.matmul(cells, center, out=out)
    out *= inverse_norms
    out *= -2.0
    out += unit_squares
    out += float(center @ center)
    np.maximum(out, 0.0, out=out)


def kmeans_plus_plus_seeds(
    cells: np.ndarray,
    inverse_norms: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (dims, n_clusters) unit centroids seeded by k-means++ on unit-normalised cells.

    Args:
        cells: ``(cells, dims)`` cell-major coordinates.
        inverse_norms: ``(cells,)`` inverse Euclidean norm of each cell, or 0 for
            a cell at the origin, whose unit vector is the zero vector.
        n_clusters: The number of seeds, at most the number of cells.
        rng: The fit's random generator.

    Returns:
        ``(dims, n_clusters)`` float64 unit vectors of the seed cells; a seed at
        the origin gives a zero column.
    """
    n_cells = cells.shape[0]
    # Squared norm of each unit-normalised cell: 1, or 0 at the origin.
    unit_squares = np.sign(inverse_norms)
    nearest = np.empty(n_cells, dtype=np.float64)
    distances = np.empty(n_cells, dtype=np.float64)
    race = np.empty(n_cells, dtype=np.float64)
    keys = np.empty(n_cells, dtype=np.float64)
    chosen = [int(rng.integers(n_cells))]
    _squared_distances_to(cells, inverse_norms, unit_squares, chosen[0], nearest)
    nearest[chosen[0]] = 0.0
    for _ in range(n_clusters - 1):
        seed = next_seed(nearest, chosen, race, keys, rng)
        chosen.append(seed)
        _squared_distances_to(cells, inverse_norms, unit_squares, seed, distances)
        np.minimum(nearest, distances, out=nearest)
        nearest[seed] = 0.0
    seeds = np.asarray(chosen, dtype=np.int64)
    return np.ascontiguousarray((cells[seeds] * inverse_norms[seeds, None]).T)
