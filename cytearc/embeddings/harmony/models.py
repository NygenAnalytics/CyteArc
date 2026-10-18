"""The result record of a Harmony2 batch correction.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonyResult:
    """Final Harmony2 state: corrected coordinates and the soft clustering that Symphony reference mapping needs.

    The state is that of the last round, whether or not the objective settled;
    ``parameters["converged"]`` records which.

    Attributes:
        original: ``(dims, cells)`` float64, an independent copy of the input coordinates.
        corrected: ``(dims, cells)`` float64, the batch-corrected coordinates.
        assignments: ``(clusters, cells)`` float64 soft cluster assignments; each column sums to 1.
        centroids: ``(dims, clusters)`` float64 cosine centroids of the corrected cells, unit or zero columns.
        sigma: ``(clusters,)`` float64 entropy weight of each cluster.
        ridge_penalty: ``(clusters, levels)`` float64 ridge penalty of the last correction round.
        active_levels: ``(clusters, levels)`` bool, the batch levels that entered each cluster's last regression.
        batch_columns: The batch metadata columns, in the order of the design.
        batch_levels: The levels of each batch column, as text, in design order.
        parameters: JSON-compatible record of the resolved options and of the fit's progress.
    """

    original: np.ndarray
    corrected: np.ndarray
    assignments: np.ndarray
    centroids: np.ndarray
    sigma: np.ndarray
    ridge_penalty: np.ndarray
    active_levels: np.ndarray
    batch_columns: tuple[str, ...]
    batch_levels: tuple[tuple[str, ...], ...]
    parameters: dict[str, object]
