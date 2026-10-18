"""The Harmony2 fit: alternate maximum-diversity clustering and ridge correction.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ...utils.arguments import integer_argument
from ...utils.logging import logger
from ...utils.numba import restore_numba_threads
from ...utils.progress import tqdmbar
from ...utils.warnings import warn
from .clustering import (
    ClusteringWorkspace,
    cluster_round,
    cosine_centroids,
    initial_assignments,
    update_inverse_norms,
)
from .correction import correct_embedding
from .design import BatchDesign, check_batch_metadata, encode_batch_design
from .models import HarmonyResult
from .parameters import (
    ResolvedHarmonyParameters,
    check_harmony_options,
    harmony_work_sizes,
    resolve_harmony_parameters,
)
from .seeding import kmeans_plus_plus_seeds


@restore_numba_threads
def fit_harmony(
    values: np.ndarray,
    metadata: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    theta: float | Sequence[float] | np.ndarray = 2.0,
    sigma: float | Sequence[float] | np.ndarray = 0.1,
    tau: float = 0.0,
    ridge_penalty: float | Sequence[float] | np.ndarray | None = None,
    ridge_alpha: float | None = None,
    prune_threshold: float = 1e-5,
    block_size: float = 0.05,
    max_iter_cluster: int = 200,
    max_iter_harmony: int = 40,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
    random_state: int = 4466,
    nthreads: int = 1,
) -> HarmonyResult:
    """Correct batch effects in reduced coordinates with Harmony2.

    Harmony alternates two steps. A soft clustering of the unit-normalised cells
    favours clusters in which every batch is represented in proportion to its
    size; then, within each cluster, a ridge regression of the original
    coordinates on the batch design estimates each batch's offset, and every
    cell subtracts the offsets of its batches, weighted by its cluster
    assignments. The regression keeps an unpenalised intercept per cluster, so
    the cell-type structure stays. The two steps repeat until the clustering
    objective settles between rounds. This is CyteArc's implementation of
    Harmony2: k-means++ seeding, the stabilised diversity penalty, a dynamic
    ridge penalty and batch pruning, with the online block updates of the 2019
    method.

    Args:
        values: ``(dims, cells)`` real coordinates, usually principal
            components, one column per cell.
        metadata: One row per cell and one column per batch variable. A
            categorical column keeps the declared order of its used categories;
            any other column orders its values with Python's ``sorted``.
        n_clusters: Number of soft clusters. Default: the cell count over 30,
            rounded half to even, kept between 1 and 100.
        theta: Diversity penalty, non-negative: a scalar, one value per batch
            column, or one value per batch level in design order. Larger values
            mix batches more strongly. Default 2.
        sigma: Entropy weight of the soft assignments, positive: a scalar or
            one value per cluster. Larger values give softer assignments.
            Default 0.1.
        tau: Discounting of theta for small batches: each level's theta is
            scaled by ``1 - exp(-(N_b / (n_clusters * tau))^2)``. Default 0,
            no discounting.
        ridge_penalty: A fixed, positive ridge penalty of the batch
            coefficients: a scalar, one value per batch column, or one value
            per batch level. Default None: the dynamic penalty
            ``ridge_alpha * E_kb`` of Harmony2, the mass that batch ``b`` would
            have in cluster ``k`` if batch and cluster were independent.
            ``ridge_penalty=1.0`` gives the fixed penalty of the 2019 method.
            With several batch columns, a fixed penalty must be at least 1e-6
            times each cluster's assignment mass (the summed assignments of
            its cells); a smaller one raises ``ValueError``, because float64
            cannot solve those regressions accurately. With one batch column
            any positive penalty is solved accurately.
        ridge_alpha: Scale of the dynamic ridge penalty, positive. Default
            None, which means 0.2. Cannot be combined with ``ridge_penalty``.
        prune_threshold: A batch level enters cluster ``k``'s regression only
            when its mass there, ``O_kb``, is positive and at least this
            fraction of its cell count ``N_b``; in ``[0, 1)``. Default 1e-5.
        block_size: Fraction of the cells updated together in each online
            block of the clustering, in ``(0, 1]``. Default 0.05.
        max_iter_cluster: Clustering passes per round, at most; at least 1.
            Default 200.
        max_iter_harmony: Rounds of clustering and correction, at most; at
            least 1. The fit stops as soon as it converges, so this only bounds
            fits that settle slowly. Default 40, where the 2019 paper used 10:
            large, heterogeneous datasets often need more than 10 rounds.
        epsilon_cluster: Relative change of the objective between consecutive
            passes that ends a round's clustering, after at least two passes.
            Default 1e-5.
        epsilon_harmony: Relative change of the objective per round that ends
            the fit. The fit has converged when the objective changed by at
            most this fraction of its value in each of the last two rounds (in
            the one round when only two have run), or, from five rounds on,
            when the average of two consecutive round objectives changes by at
            most this fraction per round. Clustering and correction can settle
            into a cycle that returns to nearly the same state every other
            round, so consecutive objectives alternate around a settled level;
            the second test ends such fits, and asking for two small changes
            in a row keeps an alternating objective from stopping the fit by
            chance. Default 1e-4.
        random_state: Non-negative seed of the fit's own random generator; the
            global NumPy state is never used. Default 4466.
        nthreads: Threads for BLAS and the Numba kernels. Results do not
            depend on it, up to the last bits of BLAS sums. Default 1.

    Returns:
        The corrected coordinates with the soft clustering, centroids and
        penalties of the final round, and a record of the resolved options and
        the fit's progress in ``parameters``.

    Raises:
        TypeError: If ``values`` does not hold real numbers, ``metadata`` is
            not a DataFrame, or an option has the wrong type.
        ValueError: If an argument is out of range, the metadata does not
            describe the cells, the values are not finite, a fixed
            ``ridge_penalty`` is too small for several batch columns, or the
            fit produces non-finite values or clusters without cells.
        numpy.linalg.LinAlgError: If a regression system is not positive
            definite.

    References:
        Korsunsky I, et al. Fast, sensitive and accurate integration of
        single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019).
        doi:10.1038/s41592-019-0619-0

        Patikas N, Yao H, et al. Integration of large, complex single-cell
        datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(
            "values must be two-dimensional with shape (dims, cells); got "
            f"{values.ndim} dimensions"
        )
    if values.dtype.kind not in "iuf":
        raise TypeError("values must hold real numbers")
    dims, n_cells = (int(size) for size in values.shape)
    if n_cells < 2:
        raise ValueError(f"Harmony needs at least two cells; got {n_cells}")
    if dims < 1:
        raise ValueError("values must have at least one dimension")
    check_batch_metadata(metadata, n_cells)
    threads = integer_argument(nthreads, "nthreads", minimum=1)
    # The cluster count is checked against the cell count once the design is known.
    if n_clusters is not None:
        integer_argument(n_clusters, "n_clusters")
    options = check_harmony_options(
        theta=theta,
        sigma=sigma,
        tau=tau,
        ridge_penalty=ridge_penalty,
        ridge_alpha=ridge_alpha,
        prune_threshold=prune_threshold,
        block_size=block_size,
        max_iter_cluster=max_iter_cluster,
        max_iter_harmony=max_iter_harmony,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
        random_state=random_state,
    )
    design = encode_batch_design(metadata)
    parameters = resolve_harmony_parameters(
        {**options, "n_clusters": n_clusters}, design
    )
    del options
    import numba

    numba.set_num_threads(min(threads, int(numba.config.NUMBA_NUM_THREADS)))
    with threadpool_limits(limits=threads):
        return _fit(values, design, parameters)


def _fit(
    values: np.ndarray,
    design: BatchDesign,
    parameters: ResolvedHarmonyParameters,
) -> HarmonyResult:
    dims, n_cells = values.shape
    n_clusters = parameters.n_clusters
    block_cells, chunk = harmony_work_sizes(n_cells, parameters.block_size)

    original = np.empty((n_cells, dims), dtype=np.float64)
    for start in range(0, n_cells, chunk):
        rows = original[start : start + chunk]
        rows[...] = values[:, start : start + chunk].T
        if not np.isfinite(rows).all():
            raise ValueError("values contains non-finite entries (NaN or infinity)")
    corrected = original.copy()
    inverse_norms = np.empty(n_cells, dtype=np.float64)
    norm_scratch = np.empty(chunk, dtype=np.float64)
    update_inverse_norms(corrected, inverse_norms, norm_scratch)
    del norm_scratch

    rng = np.random.default_rng(parameters.random_state)
    update_order = rng.permutation(n_cells).astype(np.int64, copy=False)
    centroids = kmeans_plus_plus_seeds(corrected, inverse_norms, n_clusters, rng)

    assignments = np.empty((n_cells, n_clusters), dtype=np.float64)
    workspace = ClusteringWorkspace.allocate(chunk, n_clusters, dims, design.n_levels)
    two_over_sigma = 2.0 / parameters.sigma
    initial_assignments(
        corrected,
        inverse_norms,
        centroids,
        two_over_sigma,
        design,
        assignments,
        workspace,
    )

    max_rounds = parameters.max_iter_harmony
    epsilon = parameters.epsilon_harmony
    objectives: list[float] = []
    passes: list[int] = []
    converged = False
    penalties: np.ndarray | None = None
    active: np.ndarray | None = None
    progress = tqdmbar(total=max_rounds, desc="Harmony")
    try:
        for round_index in range(max_rounds):
            if round_index > 0:
                update_inverse_norms(corrected, inverse_norms, workspace.norm_scratch)
                centroids, _norms = cosine_centroids(
                    corrected, inverse_norms, assignments, workspace
                )
                _require_finite_centroids(centroids, round_index)
            summary = cluster_round(
                corrected,
                inverse_norms,
                centroids,
                assignments,
                design,
                update_order,
                block_cells,
                two_over_sigma,
                parameters.theta,
                parameters.max_iter_cluster,
                parameters.epsilon_cluster,
                workspace,
                round_index,
            )
            centroids = summary.centroids
            objective = summary.objectives[-1]
            objectives.append(objective)
            passes.append(len(summary.objectives))
            logger.debug(
                f"Harmony round {round_index}: objective {objective:.10g} after "
                f"{passes[-1]} clustering passes"
            )
            # The previous round's copies are released before the correction.
            penalties = active = None
            penalties, active = correct_embedding(
                original,
                assignments,
                corrected,
                design,
                parameters,
                workspace,
                round_index,
            )
            progress.update(1)
            if rounds_converged(objectives, epsilon):
                converged = True
                remaining = max_rounds - (round_index + 1)
                if remaining:
                    progress.update(remaining)
                break
    finally:
        progress.close()

    # max_iter_harmony is at least 1, so the loop corrected the cells at least once.
    assert penalties is not None and active is not None
    rounds = len(objectives)
    update_inverse_norms(corrected, inverse_norms, workspace.norm_scratch)
    centroids, _norms = cosine_centroids(
        corrected, inverse_norms, assignments, workspace
    )
    _require_finite_centroids(centroids, rounds - 1)
    mass = assignments.sum(axis=0)
    empty = int(np.count_nonzero(mass == 0.0))
    if empty:
        raise ValueError(
            f"Harmony left {empty} of {n_clusters} clusters without assigned cells; "
            "use a larger sigma or fewer clusters"
        )
    if not converged and rounds >= 2:
        warn(_unconverged_message(objectives, epsilon))
    logger.info(
        f"Harmony {'converged' if converged else 'stopped'} after {rounds} "
        f"round{'s' if rounds != 1 else ''}"
    )
    return HarmonyResult(
        original=original.T,
        corrected=corrected.T,
        assignments=assignments.T,
        centroids=centroids,
        sigma=parameters.sigma,
        ridge_penalty=penalties,
        active_levels=active,
        batch_columns=design.columns,
        batch_levels=design.levels,
        parameters=_parameter_record(parameters, design, objectives, passes, converged),
    )


CYCLE_SLOPE_WEIGHTS = (-1.0, -2.0, 0.0, 2.0, 1.0)
"""Weights, divided by 8, of the last five round objectives in their cycle-robust slope."""


def objective_cycle_slope(objectives: Sequence[float]) -> float:
    """Return the per-round slope of the last five round objectives, free of a two-round cycle.

    The alternation of clustering and correction can settle into a cycle that
    returns to nearly the same state every other round, so consecutive round
    objectives alternate around a common level. The slope is the least-squares
    slope of the last five objectives under a model of a linear trend plus such
    an alternation, whose amplitude may also change linearly:
    ``(-J_{r-4} - 2 J_{r-3} + 2 J_{r-1} + J_r) / 8``. It is the mean of the
    changes of the two-round average ``(J_{t-1} + J_t) / 2`` over the last two
    windows, halved because those averages lie two rounds apart.

    Args:
        objectives: Round objectives, at least five, oldest first.

    Returns:
        The slope per round, in units of the objective.
    """
    window = objectives[-len(CYCLE_SLOPE_WEIGHTS) :]
    if len(window) < len(CYCLE_SLOPE_WEIGHTS):
        raise ValueError("The cycle slope needs at least five round objectives")
    return (
        sum(weight * value for weight, value in zip(CYCLE_SLOPE_WEIGHTS, window)) / 8.0
    )


def _changed_within(previous: float, current: float, epsilon: float) -> bool:
    """Whether ``current`` differs from ``previous`` by at most ``epsilon`` of it."""
    return abs(previous - current) <= epsilon * abs(previous)


def rounds_converged(objectives: Sequence[float], epsilon: float) -> bool:
    """Return whether Harmony's rounds have converged after the last round objective.

    ``epsilon`` is a relative tolerance per round on the objective, as in the
    2019 method. The fit has converged when either:

    - the objective changed by at most ``epsilon`` of its value between each of
      the last three rounds (between the two rounds when only two have run): a
      settled objective repeats, and two small changes in a row cannot be a
      chance crossing of an alternating objective; or
    - from five rounds on, the cycle-robust slope of the last five objectives
      (:func:`objective_cycle_slope`) is at most ``epsilon`` of the two-round
      average of the objective per round: the clustering and correction
      alternate between two states whose average objective has settled.

    The comparisons multiply instead of dividing, so an objective that stays
    exactly zero converges without a floating-point warning.

    Args:
        objectives: Round objectives, oldest first.
        epsilon: Relative tolerance per round, ``epsilon_harmony``.

    Returns:
        True when the rounds have converged.
    """
    rounds = len(objectives)
    if rounds < 2:
        return False
    if _changed_within(objectives[-2], objectives[-1], epsilon) and (
        rounds == 2 or _changed_within(objectives[-3], objectives[-2], epsilon)
    ):
        return True
    if rounds >= len(CYCLE_SLOPE_WEIGHTS):
        level = abs(objectives[-1] + objectives[-2]) / 2.0
        return abs(objective_cycle_slope(objectives)) <= epsilon * level
    return False


def _relative(change: float, value: float) -> float:
    return abs(change) / abs(value) if value != 0 else math.inf


def _unconverged_message(objectives: Sequence[float], epsilon: float) -> str:
    """Describe why the rounds did not converge, for the warning."""
    rounds = len(objectives)
    last = _relative(objectives[-1] - objectives[-2], objectives[-2])
    message = f"Harmony stopped before convergence after {rounds} rounds: the objective"
    if rounds == 2:
        return (
            f"{message} changed by {last:.3g} of its value in the last round, above "
            f"epsilon_harmony={epsilon}"
        )
    before = _relative(objectives[-2] - objectives[-3], objectives[-3])
    message += (
        f" changed by {before:.3g} and {last:.3g} of its value in the last two rounds"
    )
    if rounds < len(CYCLE_SLOPE_WEIGHTS):
        return f"{message}, and convergence needs both to be at most epsilon_harmony={epsilon}"
    slope = _relative(
        objective_cycle_slope(objectives), (objectives[-1] + objectives[-2]) / 2.0
    )
    return (
        f"{message}, and its two-round average by {slope:.3g} per round; convergence "
        f"needs both changes, or that average's change, to be at most "
        f"epsilon_harmony={epsilon}"
    )


def _require_finite_centroids(centroids: np.ndarray, round_index: int) -> None:
    if not np.isfinite(centroids).all():
        raise ValueError(
            f"Harmony produced non-finite centroids in round {round_index}"
        )


def _parameter_record(
    parameters: ResolvedHarmonyParameters,
    design: BatchDesign,
    objectives: list[float],
    passes: list[int],
    converged: bool,
) -> dict[str, object]:
    """Return the JSON-compatible record of a fit's options and progress."""
    fixed = parameters.ridge_penalty
    return {
        "algorithm": "harmony2",
        "nClusters": parameters.n_clusters,
        "theta": parameters.theta.tolist(),
        "tau": parameters.tau,
        "sigma": parameters.sigma.tolist(),
        "ridge": "dynamic" if fixed is None else "fixed",
        "ridgeAlpha": parameters.ridge_alpha,
        "ridgePenalty": None if fixed is None else fixed.tolist(),
        "pruneThreshold": parameters.prune_threshold,
        "blockSize": parameters.block_size,
        "maxIterCluster": parameters.max_iter_cluster,
        "maxIterHarmony": parameters.max_iter_harmony,
        "epsilonCluster": parameters.epsilon_cluster,
        "epsilonHarmony": parameters.epsilon_harmony,
        "randomState": parameters.random_state,
        "initialization": "kmeans++",
        "phiColumns": list(design.phi_columns),
        "rounds": len(objectives),
        "converged": converged,
        "objective": [float(value) for value in objectives],
        "clusterPasses": [int(count) for count in passes],
    }
