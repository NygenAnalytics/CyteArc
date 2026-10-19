"""Mixture-of-experts ridge correction of Harmony2.

Each cluster ``k`` is an expert: a ridge regression of the ORIGINAL coordinates on
an unpenalised intercept and one indicator per batch level, with every cell
weighted by its assignment ``R_ik``. For the active levels ``A_k`` of cluster
``k`` it minimises

    sum_i R_ik ||Z_i - w_0 - sum_{g in A_k} [i in g] w_g||^2 + sum_{g in A_k} lambda_kg ||w_g||^2.

The corrected cells subtract the batch terms of every cluster, weighted by the
assignments, and keep the intercepts, which carry the cell-type positions:
``Zhat_i = Z_i - sum_k R_ik sum_c W_{k, g_c(i)}``.

Harmony2 changes three things relative to the 2019 method. The default penalty is
dynamic, ``lambda_kg = alpha E_kg``, so that a batch needs real support in a
cluster before its coefficient escapes the shrinkage. Levels with almost no mass
in a cluster (``O_kg < prune_threshold * N_g``) are pruned from its regression.
And the normal equations are solved through their structure instead of a dense
inverse: with one batch column the system is an arrowhead matrix with a closed
form, and with several columns the column with the most levels is eliminated
through its diagonal block, leaving a small Schur complement that is factored by
Cholesky.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from ...utils.shutdown import shutdown_checkpoint
from .clustering import ClusteringWorkspace
from .design import BatchDesign
from .parameters import ResolvedHarmonyParameters

FIXED_PENALTY_MASS_RATIO_LIMIT = 1e6
"""Largest cluster mass per unit of fixed ridge penalty accepted with several batch columns.

With several batch columns the Schur complement holds entries of the size of a
cluster's mass ``n_k``, while its smallest eigenvalues are of the size of the
penalties, so rounding its entries perturbs the coefficients by about
``c eps n_k / lambda`` relative to their size. Against exact rational solves of
nested and crossed designs with two to four columns, ``c`` was at most 5.4; at
this limit the relative error therefore stays near 1e-9. The dynamic penalty
grows with the cluster's mass and stays far from the limit, and the one-column
solve has no such loss.
"""


@dataclass(slots=True)
class RegressionStatistics:
    """Sufficient statistics of every cluster's regression in one round.

    Attributes:
        combination_mass: ``(combinations, clusters)`` assignment mass of the cells
            of each level combination.
        level_sums: ``(levels, clusters, dims)`` sums ``r_g = sum_{i in g} R_ik Z_i``;
            the solve overwrites them with the coefficients ``W``.
        intercept_sums: ``(clusters, dims)`` sums ``r_0 = sum_i R_ik Z_i``.
        level_mass: ``(levels, clusters)`` exact ``O``, held in the workspace's
            ``delta`` table.
        totals: ``(clusters,)`` cluster masses ``n_k``.
    """

    combination_mass: np.ndarray
    level_sums: np.ndarray
    intercept_sums: np.ndarray
    level_mass: np.ndarray
    totals: np.ndarray


def regression_statistics(
    original: np.ndarray,
    assignments: np.ndarray,
    design: BatchDesign,
    workspace: ClusteringWorkspace,
) -> RegressionStatistics:
    """Sum the assignment-weighted coordinates of every level combination.

    Every entry of the normal equations is a sum over the cells of some level
    combinations, so one pass over the cells grouped by combination gives them all.

    Args:
        original: ``(cells, dims)`` original coordinates.
        assignments: ``(cells, clusters)`` soft assignments.
        design: The batch design.
        workspace: Chunk buffers; ``delta`` receives the exact ``O``.

    Returns:
        The statistics of every cluster.
    """
    n_clusters = assignments.shape[1]
    dims = original.shape[1]
    combinations = design.combination_levels
    starts = design.combination_starts
    order = design.cells_by_combination
    chunk = workspace.similarity.shape[0]
    products = workspace.kd
    combination_mass = np.zeros((design.n_combinations, n_clusters), dtype=np.float64)
    level_sums = np.zeros((design.n_levels, n_clusters, dims), dtype=np.float64)
    intercept_sums = np.zeros((n_clusters, dims), dtype=np.float64)
    for combination in range(design.n_combinations):
        members = order[starts[combination] : starts[combination + 1]]
        for start in range(0, members.shape[0], chunk):
            cells = members[start : start + chunk]
            rows = cells.shape[0]
            weights = workspace.similarity[:rows]
            coordinates = workspace.coordinates[:rows]
            np.take(assignments, cells, axis=0, out=weights, mode="clip")
            np.take(original, cells, axis=0, out=coordinates, mode="clip")
            np.matmul(weights.T, coordinates, out=products)
            combination_mass[combination] += weights.sum(axis=0)
            intercept_sums += products
            for level in combinations[combination]:
                level_sums[level] += products
    level_mass = workspace.delta
    level_mass.fill(0.0)
    for column in range(design.n_columns):
        np.add.at(level_mass, combinations[:, column], combination_mass)
    return RegressionStatistics(
        combination_mass=combination_mass,
        level_sums=level_sums,
        intercept_sums=intercept_sums,
        level_mass=level_mass,
        totals=combination_mass.sum(axis=0),
    )


def ridge_penalties(
    statistics: RegressionStatistics,
    design: BatchDesign,
    ridge_penalty: np.ndarray | None,
    ridge_alpha: float | None,
    out: np.ndarray,
) -> np.ndarray:
    """Write the ``(levels, clusters)`` ridge penalties of one round into ``out``.

    The dynamic penalty is ``lambda_kg = alpha E_kg = alpha Pr_g n_k`` from this
    round's assignments; a fixed penalty repeats ``lambda_g`` over the clusters.

    Args:
        statistics: The round's statistics.
        design: The batch design.
        ridge_penalty: ``(levels,)`` fixed penalties, or None for the dynamic one.
        ridge_alpha: The dynamic penalty's scale, used when ``ridge_penalty`` is None.
        out: ``(levels, clusters)`` buffer.

    Returns:
        ``out``.
    """
    if ridge_penalty is None:
        if ridge_alpha is None:
            raise ValueError("The dynamic ridge penalty needs ridge_alpha")
        np.multiply.outer(design.level_proportions, statistics.totals, out=out)
        out *= ridge_alpha
    else:
        out[...] = ridge_penalty[:, None]
    return out


def active_levels(
    statistics: RegressionStatistics,
    design: BatchDesign,
    prune_threshold: float,
) -> np.ndarray:
    """Return the ``(levels, clusters)`` levels that enter each cluster's regression.

    Level ``g`` is active in cluster ``k`` when ``O_kg > 0`` and
    ``O_kg >= prune_threshold * N_g``. A level without mass in a cluster drops out
    exactly: all of its data terms vanish, so with a positive penalty its
    coefficient would be zero.
    """
    mass = statistics.level_mass
    active = mass > 0.0
    active &= mass >= (prune_threshold * design.level_counts)[:, None]
    return active


def _solve_arrowheads(
    statistics: RegressionStatistics,
    penalties: np.ndarray,
    active: np.ndarray,
) -> None:
    """Solve every cluster's regression for one batch column in closed form.

    With one column the normal equations form an arrowhead matrix: the intercept
    row and column couple to every level, and the levels are uncoupled. Row ``g``
    reads ``O_kg w_0 + D_g w_g = r_g`` with ``D_g = O_kg + lambda_kg``, and
    substituting it into the intercept row ``n_k w_0 + sum_g O_kg w_g = r_0``
    leaves ``s w_0 = r_0 - sum_{active g} (O_kg / D_g) r_g`` with
    ``s = n_k - sum_{active g} O_kg^2 / D_g``.

    Both sides lose every digit to cancellation when a penalty is far below the
    level masses, so both are computed from the partition of the cells by level:
    ``n_k`` and ``r_0`` are the sums of every level's ``O_kg`` and ``r_g``, which
    gives ``s = sum_{pruned g} O_kg + sum_{active g} (lambda_kg / D_g) O_kg`` and
    ``s w_0 = sum_{pruned g} r_g + sum_{active g} (lambda_kg / D_g) r_g``, sums of
    terms without subtraction.
    """
    # One call per cluster, so that a cluster's temporaries are released before
    # the next cluster allocates its own.
    for cluster in range(statistics.level_sums.shape[1]):
        _solve_arrowhead(statistics, penalties, active, cluster)


def _solve_arrowhead(
    statistics: RegressionStatistics,
    penalties: np.ndarray,
    active: np.ndarray,
    cluster: int,
) -> None:
    level_sums = statistics.level_sums
    mass = statistics.level_mass[:, cluster]
    selected = active[:, cluster]
    kept = np.flatnonzero(selected)
    if kept.size == 0:
        level_sums[:, cluster, :] = 0.0
        return
    kept_mass = mass[kept]
    kept_penalties = penalties[kept, cluster]
    diagonal = kept_mass + kept_penalties
    # The share of each active level that its own coefficient leaves to the
    # intercept: lambda / D, between 0 and 1.
    shrinkage = kept_penalties / diagonal
    schur = float(mass[~selected].sum()) + float(np.sum(kept_mass * shrinkage))
    if not schur > 0.0:
        # Only a penalty that underflows against the level masses gets here.
        raise np.linalg.LinAlgError(
            f"Harmony's ridge regression for cluster {cluster} is not positive definite"
        )
    responses = level_sums[kept, cluster, :]
    intercept = shrinkage @ responses
    intercept += level_sums[~selected, cluster, :].sum(axis=0)
    intercept /= schur
    coefficients = np.multiply.outer(kept_mass, intercept)
    np.subtract(responses, coefficients, out=coefficients)
    coefficients /= diagonal[:, None]
    level_sums[:, cluster, :] = 0.0
    level_sums[kept, cluster, :] = coefficients


@dataclass(frozen=True, slots=True)
class _Elimination:
    """The column that the elimination removes and the index arrays it reuses.

    Attributes:
        column: Index of the batch column with the most levels, ``c*``.
        first: Global index of its first level.
        last: One past the global index of its last level.
        in_column: ``(levels,)`` whether each level belongs to ``c*``.
        other_columns: The other batch columns, in order.
        positions: ``(levels,)`` int32 scratch for each cluster's level positions.
    """

    column: int
    first: int
    last: int
    in_column: np.ndarray
    other_columns: tuple[int, ...]
    positions: np.ndarray


def _solve_by_elimination(
    statistics: RegressionStatistics,
    penalties: np.ndarray,
    active: np.ndarray,
    design: BatchDesign,
) -> None:
    """Solve every cluster's regression by eliminating the largest batch column.

    The unknowns split into the levels ``A1`` of the column ``c*`` with the most
    levels and ``u2 = [w_0; w_{A2}]``, the intercept and the levels of every other
    column. The ``A1`` block of the normal equations is diagonal,
    ``D1 = O + lambda``, so ``w_{A1} = D1^-1 (r_{A1} - F u2)`` with the sparse
    coupling ``F``, and ``u2`` solves the Schur complement system
    ``(A22 - F^T D1^-1 F) u2 = r_2 - F^T D1^-1 r_{A1}``. The complement is
    positive definite for positive penalties and is factored by Cholesky. Its
    intercept row and column, and the intercept entry of the right-hand side,
    are computed from the partition of the cells by the levels of ``c*`` as sums
    without cancellation. The other entries still cancel when the penalties are
    far below the cluster's mass, which loses about ``eps n_k / lambda`` of
    relative accuracy; ``correct_embedding`` therefore refuses fixed penalties
    below ``1 / FIXED_PENALTY_MASS_RATIO_LIMIT`` of a cluster's mass.
    """
    column = design.largest_column
    first = int(design.level_offsets[column])
    last = int(design.level_offsets[column + 1])
    in_column = np.zeros(design.n_levels, dtype=bool)
    in_column[first:last] = True
    elimination = _Elimination(
        column=column,
        first=first,
        last=last,
        in_column=in_column,
        other_columns=tuple(
            other for other in range(design.n_columns) if other != column
        ),
        positions=np.empty(design.n_levels, dtype=np.int32),
    )
    # One call per cluster, so that a cluster's dense Schur complement is
    # released before the next cluster allocates its own.
    for cluster in range(statistics.level_sums.shape[1]):
        _eliminate_cluster(statistics, penalties, active, design, elimination, cluster)


def _eliminate_cluster(
    statistics: RegressionStatistics,
    penalties: np.ndarray,
    active: np.ndarray,
    design: BatchDesign,
    elimination: _Elimination,
    cluster: int,
) -> None:
    level_sums = statistics.level_sums
    dims = level_sums.shape[2]
    combinations = design.combination_levels
    mass = statistics.level_mass[:, cluster]
    cluster_penalties = penalties[:, cluster]
    selected = active[:, cluster]
    if not selected.any():
        level_sums[:, cluster, :] = 0.0
        return
    star = np.flatnonzero(selected & elimination.in_column)
    others = np.flatnonzero(selected & ~elimination.in_column)
    n_star = star.size
    size = 1 + others.size
    # Positions of the active levels: rows of F for c*, and 1, 2, ... in u2 for
    # the other columns, whose position 0 is the intercept.
    positions = elimination.positions
    positions.fill(-1)
    positions[star] = np.arange(n_star, dtype=np.int32)
    positions[others] = np.arange(1, size, dtype=np.int32)
    weights = statistics.combination_mass[:, cluster]

    first, last = elimination.first, elimination.last
    star_mass = mass[star]
    star_penalties = cluster_penalties[star]
    star_diagonal = star_mass + star_penalties
    # Every cell lies in exactly one level of c*, so the intercept's entries split
    # over those levels: O_h = sum_g X(g, h) and r_0 = sum_g r_g. Eliminating an
    # active level g removes the share O_g / D_g of its terms, so the intercept
    # row and right-hand side of the Schur complement keep the share
    # lambda_g / D_g of each active level and all of each pruned one. Computed
    # this way they are sums without cancellation, however small the penalties.
    star_share = np.ones(last - first, dtype=np.float64)
    star_share[star - first] = star_penalties / star_diagonal
    combination_share = star_share[combinations[:, elimination.column] - first]
    combination_share *= weights

    # A22 over the intercept and the other columns' levels; Fortran order lets
    # the Cholesky factorisation work in place.
    schur = np.zeros((size, size), dtype=np.float64, order="F")
    diagonal_index = np.arange(1, size)
    schur[diagonal_index, diagonal_index] = mass[others] + cluster_penalties[others]
    other_positions = [
        positions[combinations[:, other]] for other in elimination.other_columns
    ]
    for index, left in enumerate(other_positions):
        present = left > 0
        np.add.at(schur[0], left[present], combination_share[present])
        for right in other_positions[index + 1 :]:
            both = (left > 0) & (right > 0)
            np.add.at(schur, (left[both], right[both]), weights[both])
            np.add.at(schur, (right[both], left[both]), weights[both])
    del combination_share
    schur[1:, 0] = schur[0, 1:]
    pruned_star_mass = float(mass[first:last][~selected[first:last]].sum())
    schur[0, 0] = pruned_star_mass + float(
        np.sum(star_mass * star_penalties / star_diagonal)
    )
    responses = np.empty((size, dims), dtype=np.float64, order="F")
    responses[0] = star_share @ level_sums[first:last, cluster, :]
    responses[1:] = level_sums[others, cluster, :]

    # The coupling F of the c* levels with the other columns' levels; its
    # intercept column, the masses O_g, enters only the back-substitution.
    coupling = None
    star_responses = None
    if n_star:
        star_positions = positions[combinations[:, elimination.column]]
        rows = [np.empty(0, dtype=np.int32)]
        columns = [np.empty(0, dtype=np.int32)]
        values = [np.empty(0, dtype=np.float64)]
        for other in other_positions:
            both = (star_positions >= 0) & (other > 0)
            rows.append(star_positions[both])
            columns.append(other[both])
            values.append(weights[both])
        del star_positions
        coupling = sparse.csr_array(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(n_star, size),
        )
        del rows, columns, values
        inverse_diagonal = 1.0 / star_diagonal
        scaled = (
            sparse.dia_array((inverse_diagonal[None, :], [0]), shape=(n_star, n_star))
            @ coupling
        )
        schur -= (coupling.T @ scaled).toarray()
        del scaled
        star_responses = level_sums[star, cluster, :]
        responses -= coupling.T @ (star_responses * inverse_diagonal[:, None])
    del other_positions
    try:
        factor = cho_factor(schur, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            f"Harmony's ridge regression for cluster {cluster} is not positive definite"
        ) from error
    solution = cho_solve(factor, responses, overwrite_b=True, check_finite=False)
    level_sums[:, cluster, :] = 0.0
    if coupling is not None and star_responses is not None:
        # w_{A1} = D1^-1 (r_{A1} - O_{A1} w_0 - F u2).
        star_coefficients = star_responses - coupling @ solution
        star_coefficients -= np.multiply.outer(star_mass, solution[0])
        star_coefficients /= star_diagonal[:, None]
        level_sums[star, cluster, :] = star_coefficients
    level_sums[others, cluster, :] = solution[1:]


def _require_accurate_fixed_penalties(
    statistics: RegressionStatistics,
    penalties: np.ndarray,
    active: np.ndarray,
) -> None:
    """Refuse fixed penalties too small for an accurate solve with several batch columns.

    Raises:
        ValueError: If a cluster's mass exceeds ``FIXED_PENALTY_MASS_RATIO_LIMIT``
            times the smallest penalty among its active levels.
    """
    smallest = np.where(active, penalties, np.inf).min(axis=0)
    excess = statistics.totals > FIXED_PENALTY_MASS_RATIO_LIMIT * smallest
    if not excess.any():
        return
    cluster = int(np.flatnonzero(excess)[0])
    mass = float(statistics.totals[cluster])
    # The smallest accepted penalty, rounded up to two significant digits so
    # that the suggested value itself passes.
    needed = mass / FIXED_PENALTY_MASS_RATIO_LIMIT
    step = 10.0 ** (math.floor(math.log10(needed)) - 1)
    suggestion = math.ceil(needed / step) * step
    raise ValueError(
        "ridge_penalty is too small for an accurate regression with several batch "
        f"columns: cluster {cluster} holds an assignment mass of {mass:.4g}, more "
        f"than 1e6 times its smallest penalty ({float(smallest[cluster]):.3g}). Pass "
        f"a ridge_penalty of at least {suggestion:.2g}, or leave ridge_penalty out to "
        "use the dynamic penalty."
    )


def solve_cluster_regressions(
    statistics: RegressionStatistics,
    penalties: np.ndarray,
    active: np.ndarray,
    design: BatchDesign,
) -> np.ndarray:
    """Solve every cluster's ridge regression and return its batch coefficients.

    The coefficients overwrite ``statistics.level_sums``; every inactive level's
    coefficient is zero, and the intercepts are solved for but not returned.

    Args:
        statistics: The round's statistics.
        penalties: ``(levels, clusters)`` ridge penalties.
        active: ``(levels, clusters)`` levels that enter each regression.
        design: The batch design.

    Returns:
        ``(levels, clusters, dims)`` coefficients ``W``.

    Raises:
        numpy.linalg.LinAlgError: If a regression system is not positive definite.
    """
    if design.n_columns == 1:
        _solve_arrowheads(statistics, penalties, active)
    else:
        _solve_by_elimination(statistics, penalties, active, design)
    return statistics.level_sums


def apply_correction(
    original: np.ndarray,
    assignments: np.ndarray,
    coefficients: np.ndarray,
    design: BatchDesign,
    corrected: np.ndarray,
    workspace: ClusteringWorkspace,
    round_index: int,
) -> None:
    """Write ``Zhat_i = Z_i - sum_k R_ik sum_c W_{k, g_c(i)}`` for every cell.

    The cells of one level combination share their batch terms, so the terms are
    summed once per combination.

    Args:
        original: ``(cells, dims)`` original coordinates.
        assignments: ``(cells, clusters)`` soft assignments.
        coefficients: ``(levels, clusters, dims)`` batch coefficients.
        design: The batch design.
        corrected: ``(cells, dims)`` receives the corrected coordinates.
        workspace: Chunk buffers.
        round_index: The 0-based round, for error messages.

    Raises:
        ValueError: If a corrected coordinate is not finite.
    """
    combinations = design.combination_levels
    starts = design.combination_starts
    order = design.cells_by_combination
    chunk = workspace.similarity.shape[0]
    terms = np.empty(coefficients.shape[1:], dtype=np.float64)
    for combination in range(design.n_combinations):
        levels = combinations[combination]
        terms[...] = coefficients[levels[0]]
        for level in levels[1:]:
            terms += coefficients[level]
        members = order[starts[combination] : starts[combination + 1]]
        for start in range(0, members.shape[0], chunk):
            cells = members[start : start + chunk]
            rows = cells.shape[0]
            weights = workspace.similarity[:rows]
            coordinates = workspace.coordinates[:rows]
            product = workspace.product[:rows]
            np.take(assignments, cells, axis=0, out=weights, mode="clip")
            np.take(original, cells, axis=0, out=coordinates, mode="clip")
            np.matmul(weights, terms, out=product)
            np.subtract(coordinates, product, out=product)
            if not np.isfinite(product).all():
                raise ValueError(
                    "Harmony produced non-finite corrected coordinates in round "
                    f"{round_index}"
                )
            corrected[cells] = product


def correct_embedding(
    original: np.ndarray,
    assignments: np.ndarray,
    corrected: np.ndarray,
    design: BatchDesign,
    parameters: ResolvedHarmonyParameters,
    workspace: ClusteringWorkspace,
    round_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one round's correction: statistics, penalties, pruning, solves and update.

    Args:
        original: ``(cells, dims)`` original coordinates.
        assignments: ``(cells, clusters)`` soft assignments of this round.
        corrected: ``(cells, dims)`` receives the corrected coordinates.
        design: The batch design.
        parameters: The fit's resolved options.
        workspace: Buffers; ``table`` receives the penalties and ``delta`` the
            exact level masses.
        round_index: The 0-based round, for error messages.

    Returns:
        ``(clusters, levels)`` copies of the round's ridge penalties and of its
        activity mask.

    Raises:
        ValueError: If a corrected coordinate is not finite, or, with several
            batch columns, a fixed ridge penalty is below
            ``1 / FIXED_PENALTY_MASS_RATIO_LIMIT`` of a cluster's mass.
        numpy.linalg.LinAlgError: If a regression system is not positive definite.
    """
    shutdown_checkpoint()
    statistics = regression_statistics(original, assignments, design, workspace)
    penalties = ridge_penalties(
        statistics,
        design,
        parameters.ridge_penalty,
        parameters.ridge_alpha,
        workspace.table,
    )
    active = active_levels(statistics, design, parameters.prune_threshold)
    if parameters.ridge_penalty is not None and design.n_columns > 1:
        _require_accurate_fixed_penalties(statistics, penalties, active)
    coefficients = solve_cluster_regressions(statistics, penalties, active, design)
    apply_correction(
        original,
        assignments,
        coefficients,
        design,
        corrected,
        workspace,
        round_index,
    )
    # Release the regression sums before the results are copied.
    del statistics, coefficients
    return penalties.T.copy(), active.T.copy()
