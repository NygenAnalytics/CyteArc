"""Numba kernels of the Harmony2 clustering.

``assign_cells`` turns precomputed cosine similarities into soft assignments,
one cell per parallel iteration, so its result does not depend on the thread
count. ``accumulate_cells`` adds or removes cells' assignment mass from the
per-level and per-cluster tables, serially and in the order of its index array.
Neither allocates; the Python wrappers in ``clustering.py`` own every buffer.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def assign_cells(
    similarity: np.ndarray,
    cells: np.ndarray,
    inverse_norms: np.ndarray,
    two_over_sigma: np.ndarray,
    penalty: np.ndarray,
    level_codes: np.ndarray,
    assignments: np.ndarray,
    entropies: np.ndarray,
) -> None:
    """Write the soft assignments of a chunk of cells and their ``sum_k R log R``.

    The logit of cell ``i`` and cluster ``k`` is
    ``(y_k . ztilde_i - 1) * 2 / sigma_k + sum_c penalty[g_c(i), k]``, and the
    assignment row is its softmax, computed after subtracting the row maximum.
    ``sum_k R_ik log R_ik`` follows from the shifted logits without any logarithm
    of an assignment, so assignments that underflow to 0 contribute exactly 0.

    Args:
        similarity: ``(chunk, clusters)`` products ``y_k . Z_i`` of the chunk's
            unnormalised cells with the centroids; overwritten as scratch.
        cells: ``(chunk,)`` int64 indices of the chunk's cells.
        inverse_norms: ``(cells,)`` inverse norm of every cell, 0 at the origin.
        two_over_sigma: ``(clusters,)`` values ``2 / sigma_k``.
        penalty: ``(levels, clusters)`` diversity penalty of each level and cluster.
        level_codes: ``(columns, cells)`` int32 global level of each cell.
        assignments: ``(cells, clusters)`` assignments; the chunk's rows are written.
        entropies: ``(chunk,)`` receives ``sum_k R_ik log R_ik`` of each cell.
    """
    n_rows = cells.shape[0]
    n_clusters = similarity.shape[1]
    n_columns = level_codes.shape[0]
    for row in prange(n_rows):
        cell = cells[row]
        scale = inverse_norms[cell]
        top = -np.inf
        for cluster in range(n_clusters):
            value = (similarity[row, cluster] * scale - 1.0) * two_over_sigma[cluster]
            for column in range(n_columns):
                value += penalty[level_codes[column, cell], cluster]
            similarity[row, cluster] = value
            if value > top:
                top = value
        total = 0.0
        weighted = 0.0
        for cluster in range(n_clusters):
            shifted = similarity[row, cluster] - top
            weight = math.exp(shifted)
            similarity[row, cluster] = weight
            total += weight
            weighted += weight * shifted
        for cluster in range(n_clusters):
            assignments[cell, cluster] = similarity[row, cluster] / total
        entropies[row] = weighted / total - math.log(total)


@njit(cache=True, nogil=True)
def accumulate_cells(
    assignments: np.ndarray,
    cells: np.ndarray,
    level_codes: np.ndarray,
    level_mass: np.ndarray,
    totals: np.ndarray,
    sign: float,
) -> None:
    """Add ``sign`` times the assignment rows of ``cells`` to the level and cluster tables.

    Args:
        assignments: ``(cells, clusters)`` soft assignments.
        cells: ``(n,)`` int64 indices of the cells to add or remove, in summation order.
        level_codes: ``(columns, cells)`` int32 global level of each cell.
        level_mass: ``(levels, clusters)`` table of ``O``, updated in place.
        totals: ``(clusters,)`` table of cluster masses, updated in place.
        sign: ``1.0`` to add the cells, ``-1.0`` to remove them.
    """
    n_rows = cells.shape[0]
    n_clusters = assignments.shape[1]
    n_columns = level_codes.shape[0]
    for row in range(n_rows):
        cell = cells[row]
        for cluster in range(n_clusters):
            value = sign * assignments[cell, cluster]
            totals[cluster] += value
            for column in range(n_columns):
                level_mass[level_codes[column, cell], cluster] += value
