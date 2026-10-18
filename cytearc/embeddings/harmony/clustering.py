"""Maximum-diversity soft clustering of Harmony2.

The clustering compares unit-normalised cells with unit centroids and minimises

    J(R, Y) = sum_i sum_k R_ik [2 (1 - y_k . ztilde_i) / sigma_k + log R_ik]
            + sum_k sum_g theta_g O_kg log((O_kg + E_kg + 1) / (E_kg + 1))

over soft assignments ``R`` (rows on the simplex) and centroids ``Y``, where
``O_kg`` is the assignment mass of level ``g`` in cluster ``k`` and
``E_kg = Pr_g n_k`` the mass expected if batch and cluster were independent.
This is the stabilised objective of Harmony2 divided by ``sigma_k`` per cluster,
which makes a per-cluster ``sigma`` exact. The penalties of several batch columns
add. With ``O`` and ``E`` held fixed, the assignment of one cell is the softmax of
``(y_k . ztilde_i - 1) 2 / sigma_k - sum_c theta log1p(O / (E + 1))``, the
exact minimiser of the cell's part of ``J``.

As in the 2019 method, cells are updated in blocks, with ``O`` and ``E`` computed
on the cells outside the block, and the centroids are the normalised
assignment-weighted sums of the unit cells (spherical k-means). The paper leaves
the order of the cells open. Here the blocks follow one random order, drawn once
per fit, because the input order could put a whole batch in one block, where its
held-out ``O`` would be zero.

Every array that holds one row per cell is cell-major, and chunk temporaries come
from the preallocated :class:`ClusteringWorkspace`.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import math
from dataclasses import dataclass

import numpy as np

from ...utils.shutdown import shutdown_checkpoint
from .design import BatchDesign
from .kernels import accumulate_cells, assign_cells

CANCELLED_SUM_TOLERANCE = 1e-12
"""Largest ``||S_k|| / n_k`` that a centroid sum may have and still count as cancelled.

Each coordinate of ``S_k`` sums ``n_k``-weighted unit-vector coordinates, so its
rounding error is a modest multiple of ``eps n_k`` (eps is about 2.2e-16); 1e-12
leaves a margin of several thousand while staying far below any sum of real cells
that do not cancel.
"""


@dataclass(slots=True)
class ClusteringWorkspace:
    """Preallocated buffers of the clustering and correction steps.

    Attributes:
        similarity: ``(chunk, clusters)`` similarities, logits or gathered assignments.
        coordinates: ``(chunk, dims)`` gathered coordinate rows.
        product: ``(chunk, dims)`` corrected rows of a chunk.
        entropies: ``(chunk,)`` ``sum_k R log R`` of each cell of a chunk.
        norm_scratch: ``(chunk,)`` norms of a chunk of cells.
        level_mass: ``(levels, clusters)`` table of ``O`` during a pass.
        delta: ``(levels, clusters)`` exact ``O`` of a correction round.
        table: ``(levels, clusters)`` penalties of a block, or the ridge penalties.
        dk: ``(dims, clusters)`` per-chunk centroid sums.
        kd: ``(clusters, dims)`` per-chunk regression sums.
        totals: ``(clusters,)`` cluster masses ``n_k`` during a pass.
    """

    similarity: np.ndarray
    coordinates: np.ndarray
    product: np.ndarray
    entropies: np.ndarray
    norm_scratch: np.ndarray
    level_mass: np.ndarray
    delta: np.ndarray
    table: np.ndarray
    dk: np.ndarray
    kd: np.ndarray
    totals: np.ndarray

    @classmethod
    def allocate(
        cls,
        chunk_cells: int,
        n_clusters: int,
        dims: int,
        n_levels: int,
    ) -> "ClusteringWorkspace":
        """Allocate the buffers for chunks of ``chunk_cells`` cells."""
        return cls(
            similarity=np.empty((chunk_cells, n_clusters), dtype=np.float64),
            coordinates=np.empty((chunk_cells, dims), dtype=np.float64),
            product=np.empty((chunk_cells, dims), dtype=np.float64),
            entropies=np.empty(chunk_cells, dtype=np.float64),
            norm_scratch=np.empty(chunk_cells, dtype=np.float64),
            level_mass=np.zeros((n_levels, n_clusters), dtype=np.float64),
            delta=np.zeros((n_levels, n_clusters), dtype=np.float64),
            table=np.zeros((n_levels, n_clusters), dtype=np.float64),
            dk=np.empty((dims, n_clusters), dtype=np.float64),
            kd=np.empty((n_clusters, dims), dtype=np.float64),
            totals=np.zeros(n_clusters, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class RoundSummary:
    """Objectives and final centroids of one round's clustering.

    Attributes:
        objectives: The objective after each pass, in order.
        centroids: ``(dims, clusters)`` centroids after the last pass.
    """

    objectives: tuple[float, ...]
    centroids: np.ndarray


def update_inverse_norms(
    cells: np.ndarray, out: np.ndarray, scratch: np.ndarray
) -> None:
    """Write ``1 / ||cells_i||`` into ``out``, or 0 for a cell at the origin.

    Args:
        cells: ``(cells, dims)`` cell-major coordinates.
        out: ``(cells,)`` receives the inverse norms.
        scratch: ``(chunk,)`` buffer; its length sets the chunk size.
    """
    n_cells = cells.shape[0]
    chunk = scratch.shape[0]
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        rows = cells[start:stop]
        norms = scratch[: stop - start]
        np.einsum("ij,ij->i", rows, rows, out=norms)
        np.sqrt(norms, out=norms)
        inverse = out[start:stop]
        inverse.fill(0.0)
        np.divide(1.0, norms, out=inverse, where=norms > 0)


def cosine_centroids(
    corrected: np.ndarray,
    inverse_norms: np.ndarray,
    assignments: np.ndarray,
    workspace: ClusteringWorkspace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit centroids ``Y`` and the norms of the weighted sums they normalise.

    ``S_k = sum_i R_ik ztilde_i`` and ``y_k = S_k / ||S_k||``. A cluster whose cells
    cancel keeps a zero centroid and a zero norm. Cancellation is judged against
    the cluster's weight ``n_k = sum_i R_ik``: a sum with
    ``||S_k|| <= CANCELLED_SUM_TOLERANCE * n_k`` is the rounding residue of an exact
    cancellation, whose direction depends on the order of the floating-point
    additions, and is treated as zero. The objective then sees ``y_k . S_k = 0``.

    Args:
        corrected: ``(cells, dims)`` current corrected coordinates.
        inverse_norms: ``(cells,)`` inverse norms of the corrected cells.
        assignments: ``(cells, clusters)`` soft assignments.
        workspace: Chunk buffers.

    Returns:
        ``(dims, clusters)`` centroids and ``(clusters,)`` norms ``||S_k||``.
    """
    n_cells, dims = corrected.shape
    n_clusters = assignments.shape[1]
    chunk = workspace.similarity.shape[0]
    sums = np.zeros((dims, n_clusters), dtype=np.float64)
    weight = np.zeros(n_clusters, dtype=np.float64)
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        weights = workspace.similarity[: stop - start]
        weight += assignments[start:stop].sum(axis=0)
        np.multiply(
            assignments[start:stop], inverse_norms[start:stop, None], out=weights
        )
        np.matmul(corrected[start:stop].T, weights, out=workspace.dk)
        sums += workspace.dk
    norms = np.sqrt(np.einsum("ij,ij->j", sums, sums))
    # Underflowed squares also give a zero norm, so the test covers them too.
    cancelled = norms <= CANCELLED_SUM_TOLERANCE * weight
    norms[cancelled] = 0.0
    sums[:, cancelled] = 0.0
    np.divide(sums, norms, out=sums, where=~cancelled)
    return sums, norms


def _fill_penalties(
    level_mass: np.ndarray,
    totals: np.ndarray,
    proportions: np.ndarray,
    theta: np.ndarray,
    table: np.ndarray,
) -> None:
    """Write ``-theta_g log1p(O_kg / (E_kg + 1))`` into ``table`` in place."""
    np.multiply.outer(proportions, totals, out=table)
    table += 1.0
    np.divide(level_mass, table, out=table)
    np.log1p(table, out=table)
    table *= theta[:, None]
    np.negative(table, out=table)


def initial_assignments(
    corrected: np.ndarray,
    inverse_norms: np.ndarray,
    centroids: np.ndarray,
    two_over_sigma: np.ndarray,
    design: BatchDesign,
    assignments: np.ndarray,
    workspace: ClusteringWorkspace,
) -> None:
    """Write the penalty-free assignments of every cell to the seeded centroids.

    Args:
        corrected: ``(cells, dims)`` corrected coordinates.
        inverse_norms: ``(cells,)`` inverse norms of the corrected cells.
        centroids: ``(dims, clusters)`` unit centroids.
        two_over_sigma: ``(clusters,)`` values ``2 / sigma_k``.
        design: The batch design.
        assignments: ``(cells, clusters)`` receives the assignments.
        workspace: Chunk buffers; its ``table`` is cleared.
    """
    n_cells = corrected.shape[0]
    chunk = workspace.similarity.shape[0]
    workspace.table.fill(0.0)
    # Indices of the current chunk, advanced in place from chunk to chunk.
    cells = np.arange(chunk, dtype=np.int64)
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        rows = stop - start
        similarity = workspace.similarity[:rows]
        np.matmul(corrected[start:stop], centroids, out=similarity)
        assign_cells(
            similarity,
            cells[:rows],
            inverse_norms,
            two_over_sigma,
            workspace.table,
            design.level_codes,
            assignments,
            workspace.entropies[:rows],
        )
        cells += chunk


def clustering_pass(
    corrected: np.ndarray,
    inverse_norms: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    design: BatchDesign,
    update_order: np.ndarray,
    block_cells: int,
    two_over_sigma: np.ndarray,
    theta: np.ndarray,
    workspace: ClusteringWorkspace,
) -> float:
    """Update every cell's assignment once, block by block, and return ``sum R log R``.

    ``O`` and the cluster masses are summed exactly at the start of the pass.
    Each block of ``update_order`` is then held out of them, its cells are
    assigned against the held-out penalties, and the new rows are added back.
    On return, the workspace's ``level_mass`` and ``totals`` hold the pass's
    final ``O`` and ``n``.

    Args:
        corrected: ``(cells, dims)`` corrected coordinates.
        inverse_norms: ``(cells,)`` inverse norms of the corrected cells.
        centroids: ``(dims, clusters)`` centroids.
        assignments: ``(cells, clusters)`` assignments, updated in place.
        design: The batch design.
        update_order: ``(cells,)`` int64 fixed random order of the cells.
        block_cells: Cells per update block.
        two_over_sigma: ``(clusters,)`` values ``2 / sigma_k``.
        theta: ``(levels,)`` diversity penalty of each level.
        workspace: Buffers.

    Returns:
        ``sum_ik R_ik log R_ik`` of the pass's assignments.
    """
    level_codes = design.level_codes
    level_mass = workspace.level_mass
    totals = workspace.totals
    level_mass.fill(0.0)
    totals.fill(0.0)
    accumulate_cells(assignments, update_order, level_codes, level_mass, totals, 1.0)
    chunk = workspace.similarity.shape[0]
    entropy = 0.0
    for block_start in range(0, update_order.shape[0], block_cells):
        block = update_order[block_start : block_start + block_cells]
        accumulate_cells(assignments, block, level_codes, level_mass, totals, -1.0)
        _fill_penalties(
            level_mass, totals, design.level_proportions, theta, workspace.table
        )
        for start in range(0, block.shape[0], chunk):
            cells = block[start : start + chunk]
            rows = cells.shape[0]
            coordinates = workspace.coordinates[:rows]
            similarity = workspace.similarity[:rows]
            entropies = workspace.entropies[:rows]
            np.take(corrected, cells, axis=0, out=coordinates, mode="clip")
            np.matmul(coordinates, centroids, out=similarity)
            assign_cells(
                similarity,
                cells,
                inverse_norms,
                two_over_sigma,
                workspace.table,
                level_codes,
                assignments,
                entropies,
            )
            entropy += float(entropies.sum())
        accumulate_cells(assignments, block, level_codes, level_mass, totals, 1.0)
    return entropy


def clustering_objective(
    norms: np.ndarray,
    two_over_sigma: np.ndarray,
    entropy: float,
    theta: np.ndarray,
    design: BatchDesign,
    workspace: ClusteringWorkspace,
) -> float:
    """Return ``J`` after a pass and the centroid update that follows it.

    The distance term is ``sum_k (2 / sigma_k) (n_k - ||S_k||)``, because
    ``y_k . S_k = ||S_k||`` right after the centroid update; the entropy term is
    the pass's ``sum R log R``; the diversity term uses the pass's final ``O``
    and ``n`` in the workspace.

    Args:
        norms: ``(clusters,)`` norms ``||S_k||`` from :func:`cosine_centroids`.
        two_over_sigma: ``(clusters,)`` values ``2 / sigma_k``.
        entropy: ``sum_ik R_ik log R_ik`` of the pass.
        theta: ``(levels,)`` diversity penalty of each level.
        design: The batch design.
        workspace: Buffers holding the pass's ``level_mass`` and ``totals``.

    Returns:
        The objective.
    """
    totals = workspace.totals
    table = workspace.table
    distance = float(two_over_sigma @ (totals - norms))
    np.multiply.outer(design.level_proportions, totals, out=table)
    table += 1.0
    np.divide(workspace.level_mass, table, out=table)
    np.log1p(table, out=table)
    table *= workspace.level_mass
    diversity = float(theta @ table.sum(axis=1))
    return distance + entropy + diversity


def cluster_round(
    corrected: np.ndarray,
    inverse_norms: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    design: BatchDesign,
    update_order: np.ndarray,
    block_cells: int,
    two_over_sigma: np.ndarray,
    theta: np.ndarray,
    max_passes: int,
    epsilon: float,
    workspace: ClusteringWorkspace,
    round_index: int,
) -> RoundSummary:
    """Run clustering passes until the objective settles or the pass limit is reached.

    After pass ``t`` with objective ``J_t``, the round stops when ``t >= 2`` and
    ``|J_{t-1} - J_t| <= epsilon |J_{t-1}|``, or when ``t == max_passes``.

    Args:
        corrected: ``(cells, dims)`` corrected coordinates.
        inverse_norms: ``(cells,)`` inverse norms of the corrected cells.
        centroids: ``(dims, clusters)`` centroids that the first pass uses.
        assignments: ``(cells, clusters)`` assignments, updated in place.
        design: The batch design.
        update_order: ``(cells,)`` int64 fixed random order of the cells.
        block_cells: Cells per update block.
        two_over_sigma: ``(clusters,)`` values ``2 / sigma_k``.
        theta: ``(levels,)`` diversity penalty of each level.
        max_passes: Passes at most.
        epsilon: Relative objective change that ends the round.
        workspace: Buffers.
        round_index: The 0-based round, for error messages.

    Returns:
        The objective of every pass and the centroids after the last one.

    Raises:
        ValueError: If the assignments, centroids or objective become non-finite.
    """
    objectives: list[float] = []
    for _pass in range(max_passes):
        shutdown_checkpoint()
        entropy = clustering_pass(
            corrected,
            inverse_norms,
            centroids,
            assignments,
            design,
            update_order,
            block_cells,
            two_over_sigma,
            theta,
            workspace,
        )
        if not math.isfinite(entropy):
            raise ValueError(
                "Harmony produced non-finite cluster assignments in round "
                f"{round_index}"
            )
        centroids, norms = cosine_centroids(
            corrected, inverse_norms, assignments, workspace
        )
        if not np.isfinite(centroids).all():
            raise ValueError(
                f"Harmony produced non-finite centroids in round {round_index}"
            )
        objective = clustering_objective(
            norms, two_over_sigma, entropy, theta, design, workspace
        )
        if not math.isfinite(objective):
            raise ValueError(
                f"Harmony produced non-finite objective in round {round_index}"
            )
        objectives.append(objective)
        if len(objectives) >= 2:
            previous = objectives[-2]
            if abs(previous - objective) <= epsilon * abs(previous):
                break
    return RoundSummary(objectives=tuple(objectives), centroids=centroids)
