"""CyteArc's native, multi-threaded and deterministic Leiden backend.

This module drives the Leiden algorithm of Traag, Waltman and van Eck (2019)
for modularity with a resolution parameter ``gamma`` (the configuration null
model). The input matrix ``W`` stands for the symmetric adjacency
``A = W + W^T`` including its diagonal, and the objective is

    Q = (1/2m) * sum_ij [A_ij - gamma * k_i * k_j / 2m] * delta(c_i, c_j),

with ``k_i = sum_j A_ij`` and ``2m = sum_ij A_ij``. The algorithm runs three
times, each iteration starting from the previous partition (the first from
singletons). An iteration repeats, level by level: fast local moving of nodes,
refinement of every community into well connected parts, and aggregation of
the graph by the refined partition, with the unrefined partition as the
starting partition of the aggregate graph. A level loop ends when local moving
leaves every node in its own community. If refinement merges nothing, which
is rare (it has not been seen on kNN graphs but occurs on some small graphs),
the paper keeps iterating; here the refinement of that level is retried up to
``REFINEMENT_RETRIES`` times, retry ``r`` drawing from the stream
``derive_key(refine_key, r)``. Re-running local moving on the unchanged,
converged partition would make no move, so a retry is the same as the paper's
next round. If every retry merges nothing as well, the communities are split
into connected components and the level loop ends (a safety stop that
guarantees termination). The labels are numbered ``1..r`` by decreasing
community size, ties broken by the smallest node index.

Every phase runs in Numba kernels that release the GIL. Python regains
control between phases, where ``shutdown_checkpoint`` runs. Results never
depend on the number of threads: floating-point sums have fixed orders,
community totals are exact integers, parallel decisions read snapshots, and
random numbers come from counter-based streams keyed by the seed, the
iteration, the level, the phase and the entity (a sweep or a community).
Seeds are reduced modulo ``2**64``. Bitwise reproducibility holds on one
platform build; ``exp`` comes from the platform's math library.

Constants that define the labels (frozen at release; changing one needs an
operation revision scoped to ``backend == "cytearc"``):

==============================  =======  ======================================
Name                            Value    Role
==============================  =======  ======================================
``N_ITERATIONS``                3        iterations of the whole algorithm
``THETA``                       0.01     refinement randomness, in units of the
                                         mean off-diagonal edge weight
``MOVE_TOLERANCE``              1e-10    a move must gain more than
                                         ``MOVE_TOLERANCE * k_v``
``GROUPS``                      64       random groups per parallel sweep
``SEQUENTIAL_LEVEL_ENTRIES``    2**21    levels with fewer stored entries run
                                         the sequential queue of the paper
``TAIL_DIVISOR``                64       parallel sweeps stop once fewer than
                                         ``n / TAIL_DIVISOR`` nodes are active
``MAX_PARALLEL_SWEEPS``         64       cap on parallel sweeps per level
``REFINEMENT_RETRIES``          8        refinement retries on a level where
                                         refinement merged nothing
``QUEUE_EVALUATIONS_PER_NODE``  1000     the sequential queue of a level stops
                                         after this many evaluations per node
==============================  =======  ======================================

``SEQUENTIAL_LEVEL_ENTRIES``, ``TAIL_DIVISOR`` and ``MAX_PARALLEL_SWEEPS``
were measured once on 100,000-node kNN graphs at 1 and 4 threads, across their
allowed ranges (``2**19`` to ``2**23``, 16 to 256 and 32 to 128). No other
value was clearly faster or better, so they keep the values above. The last
two only matter in degenerate cases: retries happen only when a refinement
merges nothing, and the evaluation cap is a hang guard that accepted moves
(each gaining more than ``MOVE_TOLERANCE * k_v``) never reach.

Memory: besides the ``O(n + m)`` arrays of the graph and its first aggregate,
every worker owns scratch sized by the largest degree ``D`` of the current
level: a hash table of ``next_pow2(2 * D + 2)`` slots of 16 bytes and
candidate buffers of ``20 * (D + 1)`` bytes (aggregation sizes its tables by
the largest aggregate row, at most the number of aggregate nodes). This is
``O(T * D)`` for ``T`` threads; it is small on kNN graphs but grows with a
hub, for example about 4.3 GB of hash tables and 1.6 GB of candidate buffers
for a node of degree ``10**7`` at 8 threads.

Further fixed values live next to the code that uses them: the random number
constants (``GOLDEN``, the two ``mix64`` multipliers, ``SEED_SALT`` and the
phase ids) and ``FLOAT_SUM_BLOCK = 65536`` in ``_leiden_primitives``, and
``MAX_NODES = 2**30 - 1`` in ``_leiden_graph``. ``PARALLEL_MIN_ITEMS = 256``
and the other block sizes only decide between loops that compute the same
values, so they do not change results.

References:
    Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9, 5233 (2019).
    doi:10.1038/s41598-019-41695-z

    Lu H, Halappanavar M, Kalyanaraman A. Parallel heuristics for scalable
    community detection. Parallel Computing 47, 19-37 (2015).
    doi:10.1016/j.parco.2015.03.003

    Sahu S, Kothapalli K, Banerjee DS. Fast Leiden algorithm for community
    detection in shared memory setting. Proceedings of the 53rd International
    Conference on Parallel Processing (ICPP 2024), 11-20.
    doi:10.1145/3673038.3673146 (preprint arXiv:2312.13936)
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import sparray, spmatrix

from ..utils.logging import logger
from ..utils.numba import restore_numba_threads
from ..utils.shutdown import shutdown_checkpoint
from ._leiden_aggregate import (
    aggregate_counts,
    aggregate_fill,
    aggregate_node_info,
    aggregation_order,
    compact_communities,
    compact_refined,
    final_labels,
    next_level_state,
    renumber_partition,
    split_into_components,
)
from ._leiden_graph import MAX_NODES, LeidenGraph, build_leiden_graph
from ._leiden_moving import move_nodes_level
from ._leiden_primitives import (
    EMPTY_SLOT,
    FLOAT_SUM_BLOCK,
    PARALLEL_MIN_ITEMS,
    PHASE_REFINE,
    derive_key,
    exclusive_scan,
    gather,
    hash_capacity,
    stable_bucket_order,
    stream_key,
)
from ._leiden_refine import community_order, refine_partition

__all__ = [
    "FLOAT_SUM_BLOCK",
    "GROUPS",
    "MAX_NODES",
    "MAX_PARALLEL_SWEEPS",
    "MOVE_TOLERANCE",
    "N_ITERATIONS",
    "PARALLEL_MIN_ITEMS",
    "QUEUE_EVALUATIONS_PER_NODE",
    "REFINEMENT_RETRIES",
    "SEQUENTIAL_LEVEL_ENTRIES",
    "TAIL_DIVISOR",
    "THETA",
    "native_leiden_membership",
]

N_ITERATIONS = 3
THETA = 0.01
MOVE_TOLERANCE = 1e-10
GROUPS = 64
SEQUENTIAL_LEVEL_ENTRIES = 2**21
TAIL_DIVISOR = 64
MAX_PARALLEL_SWEEPS = 64
REFINEMENT_RETRIES = 8
QUEUE_EVALUATIONS_PER_NODE = 1000

# GROUPS must be a power of two, which tests/test_leiden_native.py checks.
_GROUP_BITS = GROUPS.bit_length() - 1


@dataclass(frozen=True, slots=True)
class _RunSettings:
    """Per-run values shared by every level."""

    seed: np.uint64
    null_coef: float
    eps_coef: float
    theta_w: float
    n_workers: int
    sequential_level_entries: int
    parallel_min_items: int


def _hash_scratch(
    max_distinct: int, n_workers: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.uint64]:
    """Allocate one empty hash accumulator per worker."""
    capacity, shift = hash_capacity(max_distinct)
    keys = np.full((n_workers, capacity), EMPTY_SLOT, dtype=np.int32)
    vals = np.zeros((n_workers, capacity), dtype=np.float64)
    used = np.empty((n_workers, capacity), dtype=np.int32)
    return keys, vals, used, shift


def _log_level(
    iteration: int,
    level: int,
    n_nodes: int,
    n_entries: int,
    stats: np.ndarray,
    n_comm: int,
    n_refined: int,
    retries: int,
    seconds: tuple[float, float, float],
    note: str,
) -> None:
    logger.debug(
        f"native Leiden iteration={iteration} level={level} nodes={n_nodes} "
        f"entries={n_entries} path={'groups' if stats[4] else 'queue'} "
        f"evaluations={int(stats[0])} moves={int(stats[1])} sweeps={int(stats[2])} "
        f"queued={int(stats[3])} communities={n_comm} refined={n_refined} "
        f"retries={retries} move_s={seconds[0]:.6f} refine_s={seconds[1]:.6f} "
        f"aggregate_s={seconds[2]:.6f} end={note}"
    )


def _run_iteration(
    graph: LeidenGraph, flat: np.ndarray, iteration: int, settings: _RunSettings
) -> np.ndarray:
    """Run one iteration of the Leiden algorithm from the partition ``flat``."""
    n_workers = settings.n_workers
    min_items = settings.parallel_min_items
    indptr = graph.indptr
    indices = graph.indices
    weights = graph.weights
    strengths = graph.strengths
    max_degree = graph.max_degree
    labels, totals, sizes, n_comm = renumber_partition(
        flat, strengths, n_workers, min_items
    )
    base_map: np.ndarray | None = None
    level = 0
    while True:
        shutdown_checkpoint()
        started = time.perf_counter()
        n_level = int(strengths.shape[0])
        n_entries = int(indptr[n_level])
        keys, vals, used, shift = _hash_scratch(max_degree, n_workers)
        next_free = np.array([n_comm], dtype=np.int64)
        stats = move_nodes_level(
            indptr,
            indices,
            weights,
            strengths,
            labels,
            totals,
            sizes,
            next_free,
            settings.seed,
            iteration,
            level,
            settings.null_coef,
            settings.eps_coef,
            n_workers,
            settings.sequential_level_entries,
            TAIL_DIVISOR,
            MAX_PARALLEL_SWEEPS,
            _GROUP_BITS,
            min_items,
            QUEUE_EVALUATIONS_PER_NODE,
            keys,
            vals,
            used,
            shift,
        )
        if stats[5]:
            logger.debug(
                f"native Leiden iteration={iteration} level={level} the sequential "
                f"queue stopped at its cap of {QUEUE_EVALUATIONS_PER_NODE * n_level} "
                "evaluations"
            )
        labels, compact_totals, _compact_sizes, n_comm = compact_communities(
            labels, totals, sizes, n_workers, min_items
        )
        del totals, sizes, _compact_sizes
        moved = time.perf_counter()
        if n_comm == n_level:
            _log_level(
                iteration,
                level,
                n_level,
                n_entries,
                stats,
                n_comm,
                n_level,
                0,
                (moved - started, 0.0, 0.0),
                "converged",
            )
            break
        shutdown_checkpoint()
        member_start, members = stable_bucket_order(
            labels, n_comm, n_workers, min_items
        )
        comm_order = community_order(member_start)
        refined = np.empty(n_level, dtype=np.int32)
        rsize = np.empty(n_level, dtype=np.int32)
        kref = np.empty(n_level, dtype=np.int64)
        eext = np.empty(n_level, dtype=np.float64)
        rlist = np.empty(n_level, dtype=np.int32)
        n_refined = np.empty(n_comm, dtype=np.int32)
        cand_ids = np.empty((n_workers, max_degree + 1), dtype=np.int32)
        cand_gain = np.empty((n_workers, max_degree + 1), dtype=np.float64)
        cand_a = np.empty((n_workers, max_degree + 1), dtype=np.float64)
        refine_key = np.uint64(
            stream_key(settings.seed, iteration, level, PHASE_REFINE)
        )
        retries = 0
        while True:
            # Retry r draws from derive_key(refine_key, r); attempt 0 uses the
            # level's key itself.
            attempt_key = (
                refine_key
                if retries == 0
                else np.uint64(derive_key(refine_key, retries))
            )
            refine_partition(
                indptr,
                indices,
                weights,
                strengths,
                labels,
                n_comm,
                member_start,
                members,
                comm_order,
                settings.null_coef,
                settings.theta_w,
                attempt_key,
                n_workers,
                refined,
                rsize,
                kref,
                eext,
                rlist,
                n_refined,
                keys,
                vals,
                used,
                shift,
                cand_ids,
                cand_gain,
                cand_a,
                min_items,
            )
            ragg, n_aggregate = compact_refined(refined, rsize, n_workers, min_items)
            if retries > 0:
                logger.debug(
                    f"native Leiden iteration={iteration} level={level} refinement "
                    f"retry={retries} of {REFINEMENT_RETRIES} nodes={n_level} "
                    f"refined={n_aggregate}"
                )
            if n_aggregate < n_level or retries == REFINEMENT_RETRIES:
                break
            retries += 1
            shutdown_checkpoint()
        del member_start, members, comm_order, kref, eext, rlist
        del cand_ids, cand_gain, cand_a, keys, vals, used, refined, rsize
        refined_at = time.perf_counter()
        if n_aggregate == n_level:
            labels = split_into_components(indptr, indices, labels)
            _log_level(
                iteration,
                level,
                n_level,
                n_entries,
                stats,
                n_comm,
                n_aggregate,
                retries,
                (moved - started, refined_at - moved, 0.0),
                "safety-stop",
            )
            break
        shutdown_checkpoint()
        agg_start, agg_members = stable_bucket_order(
            ragg, n_aggregate, n_workers, min_items
        )
        new_strengths, new_labels, row_bounds = aggregate_node_info(
            strengths, labels, indptr, agg_start, agg_members, n_workers
        )
        agg_order = aggregation_order(row_bounds, n_workers, min_items)
        agg_keys, agg_vals, agg_used, agg_shift = _hash_scratch(
            min(int(row_bounds.max()), n_aggregate), n_workers
        )
        del row_bounds
        row_lengths = aggregate_counts(
            indptr,
            indices,
            ragg,
            agg_start,
            agg_members,
            agg_order,
            n_workers,
            agg_keys,
            agg_used,
            agg_shift,
            min_items,
        )
        new_indptr = exclusive_scan(row_lengths, n_workers, min_items)
        max_degree = int(row_lengths.max())
        del row_lengths
        new_indices = np.empty(int(new_indptr[n_aggregate]), dtype=np.int32)
        new_weights = np.empty(int(new_indptr[n_aggregate]), dtype=np.float64)
        aggregate_fill(
            indptr,
            indices,
            weights,
            ragg,
            agg_start,
            agg_members,
            agg_order,
            new_indptr,
            new_indices,
            new_weights,
            n_workers,
            agg_keys,
            agg_vals,
            agg_used,
            agg_shift,
            min_items,
        )
        del agg_start, agg_members, agg_order, agg_keys, agg_vals, agg_used
        totals, sizes = next_level_state(compact_totals, n_refined, n_aggregate)
        del compact_totals, n_refined
        base_map = ragg if base_map is None else gather(ragg, base_map, n_workers)
        del ragg
        indptr = new_indptr
        indices = new_indices
        weights = new_weights
        strengths = new_strengths
        labels = new_labels
        aggregated = time.perf_counter()
        _log_level(
            iteration,
            level,
            n_level,
            n_entries,
            stats,
            n_comm,
            n_aggregate,
            retries,
            (moved - started, refined_at - moved, aggregated - refined_at),
            "aggregated",
        )
        level += 1
    if base_map is None:
        return labels
    return gather(labels, base_map, n_workers)


@restore_numba_threads
def native_leiden_membership(
    graph: spmatrix | sparray,
    resolution: float,
    random_seed: int,
    nthreads: int | None,
    *,
    _sequential_level_entries: int = SEQUENTIAL_LEVEL_ENTRIES,
    _parallel_min_items: int = PARALLEL_MIN_ITEMS,
) -> np.ndarray:
    """Cluster a sparse graph with CyteArc's native Leiden implementation.

    Args:
        graph: Square SciPy sparse matrix or array. Entry ``(i, j, w)`` is an
            undirected edge of weight ``w``; the adjacency is ``W + W^T``.
        resolution: Validated, finite and positive resolution ``gamma``.
        random_seed: Validated non-negative seed, reduced modulo ``2**64``.
        nthreads: Number of worker threads, or ``None`` for Numba's current
            thread count. It never changes the labels.
        _sequential_level_entries: Test hook; levels with fewer stored entries
            run the sequential queue. Changes results; only tests pass it.
        _parallel_min_items: Test hook; inputs smaller than this use
            single-thread loops. Does not change results.

    Returns:
        One int64 label per row, ``1..r`` by decreasing community size, ties
        broken by the smallest node index.
    """
    import numba

    requested = numba.get_num_threads() if nthreads is None else int(nthreads)
    threads = max(1, min(requested, int(numba.config.NUMBA_NUM_THREADS)))
    numba.set_num_threads(threads)
    started = time.perf_counter()
    leiden_graph = build_leiden_graph(graph, threads, _parallel_min_items)
    n_nodes = leiden_graph.n_nodes
    if n_nodes == 0:
        return np.empty(0, dtype=np.int64)
    if not leiden_graph.has_offdiagonal:
        return np.arange(1, n_nodes + 1, dtype=np.int64)
    logger.debug(
        f"native Leiden graph nodes={n_nodes} entries={int(leiden_graph.indptr[-1])} "
        f"threads={threads} build_s={time.perf_counter() - started:.6f}"
    )
    settings = _RunSettings(
        seed=np.uint64(int(random_seed) % 2**64),
        null_coef=float(resolution)
        / (leiden_graph.strength_scale * float(leiden_graph.total_strength)),
        eps_coef=MOVE_TOLERANCE / leiden_graph.strength_scale,
        theta_w=THETA * leiden_graph.mean_weight,
        n_workers=threads,
        sequential_level_entries=int(_sequential_level_entries),
        parallel_min_items=int(_parallel_min_items),
    )
    flat = np.arange(n_nodes, dtype=np.int32)
    for iteration in range(N_ITERATIONS):
        shutdown_checkpoint()
        flat = _run_iteration(leiden_graph, flat, iteration, settings)
    labels = final_labels(flat, n_nodes, threads)
    logger.debug(
        f"native Leiden finished clusters={int(labels.max())} "
        f"seconds={time.perf_counter() - started:.6f}"
    )
    return labels
