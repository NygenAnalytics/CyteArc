"""Local moving of nodes for CyteArc's native Leiden backend.

This is the fast local moving phase (MoveNodesFast) of the Leiden algorithm of
Traag, Waltman and van Eck (2019), Algorithm A.2 lines 13-25, for modularity
with a resolution parameter. A node moves only when the move strictly raises
the quality, to the best neighbouring community or to an empty one, and the
neighbours outside its new community are queued again.

Notation (all weights are the normalised weights of ``_leiden_graph``): for a
node ``v`` and a set ``X`` not containing ``v``, ``a(v, X)`` is the weight from
``v`` to ``X`` (self-loops excluded) and ``f(v, X) = a(v, X) - c_v * K_X`` with
``c_v = gamma * k_v / 2m`` and ``K_X`` the total strength of ``X``. Moving
``v`` from ``C`` to ``D`` changes the quality by ``f(v, D) - f(v, C - v)``
(in units of edge weight). Strengths and community totals are int64
fixed-point numbers, so ``K_X`` never drifts.

Two schedules apply the same moves rule:

- The sequential queue (``move_nodes_queue``) is the paper's procedure: nodes
  are visited from a first-in first-out queue that starts in random order. It
  runs small levels and the tail of large ones.
- On large levels (``move_nodes_level``) each sweep splits the active nodes
  into random groups. All nodes of a group choose their moves from the state
  at the start of the group, in parallel; the moves are then applied
  together and the neighbours of moved nodes are activated. Decisions read
  only that snapshot and moves are ordered by node id, so the result does
  not depend on the number of threads. To stop two singletons from swapping
  communities in the same group, the singlet minimum-label rule of Lu,
  Halappanavar and Kalyanaraman (2015) lets a singleton join another
  singleton only when the other one's community id is smaller.

Tie rules: staying wins every tie (a margin of ``1e-10 * k_v`` absorbs
rounding noise), among existing communities the smallest id wins, and an
existing community wins a tie with an empty one.

References:
    Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9, 5233 (2019).
    doi:10.1038/s41598-019-41695-z

    Lu H, Halappanavar M, Kalyanaraman A. Parallel heuristics for scalable
    community detection. Parallel Computing 47, 19-37 (2015).
    doi:10.1016/j.parco.2015.03.003
"""

import numpy as np
from numba import njit, prange

from ._leiden_primitives import (
    BUCKET_CHUNK,
    COMPACT_BLOCK,
    GOLDEN,
    PHASE_GROUPS,
    PHASE_MOVE_ORDER,
    WORKER_BLOCK,
    acc_add,
    acc_reset,
    check_worker_rows,
    derive_key,
    mix64,
    scatter_add,
    stream_key,
    uniform01,
)

# Move targets that are not community ids.
STAY = -1
NEW_COMMUNITY = -2


@njit(cache=True, nogil=True)
def best_move(
    v: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    strengths: np.ndarray,
    null_coef: float,
    eps_coef: float,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
) -> tuple[int, float]:
    """Return the best move of node ``v`` and its quality gain.

    The target is a community id, ``STAY`` or ``NEW_COMMUNITY``; the gain is
    ``f(v, D) - f(v, C - v)`` for a move and 0 for ``STAY``. ``keys``, ``vals``
    and ``used`` are one worker's hash accumulator (empty on entry and on
    return).
    """
    current = labels[v]
    n_used = int(np.int64(0))
    for position in range(indptr[v], indptr[v + 1]):
        neighbour = indices[position]
        if neighbour != v:
            n_used = acc_add(
                keys, vals, used, n_used, labels[neighbour], weights[position], shift
            )
    node_strength = strengths[v]
    c_v = null_coef * np.float64(node_strength)
    eps_v = eps_coef * np.float64(node_strength)
    weight_to_current = 0.0
    best_target = -1
    best_score = 0.0
    for index in range(n_used):
        slot = used[index]
        community = keys[slot]
        if community == current:
            weight_to_current = vals[slot]
            continue
        score = vals[slot] - c_v * np.float64(totals[community])
        if (
            best_target < 0
            or score > best_score
            or (score == best_score and community < best_target)
        ):
            best_target = community
            best_score = score
    acc_reset(keys, used, n_used)
    alone = sizes[current] == 1
    if alone:
        stay_score = 0.0
    else:
        stay_score = weight_to_current - c_v * np.float64(
            totals[current] - node_strength
        )
    if best_target >= 0 and (alone or best_score >= 0.0):
        option = best_target
        option_score = best_score
    elif not alone:
        option = NEW_COMMUNITY
        option_score = 0.0
    else:
        return STAY, 0.0
    if option_score > stay_score + eps_v:
        return option, option_score - stay_score
    return STAY, 0.0


@njit(cache=True, nogil=True)
def random_permutation(n: int, key: np.uint64) -> np.ndarray:
    """Return a Fisher-Yates permutation of ``0..n-1`` drawn from stream ``key``.

    Position ``i`` (from ``n - 1`` down to 1) swaps with
    ``floor(uniform01(key, i) * (i + 1))``, clamped to ``i``.
    """
    permutation = np.arange(n).astype(np.int32)
    for position in range(n - 1, 0, -1):
        other = int(uniform01(key, position) * (position + 1))
        if other > position:
            other = position
        held = permutation[position]
        permutation[position] = permutation[other]
        permutation[other] = held
    return permutation


@njit(cache=True, nogil=True)
def move_nodes_queue(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    strengths: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    next_free: np.ndarray,
    initial: np.ndarray,
    inq: np.ndarray,
    queue: np.ndarray,
    null_coef: float,
    eps_coef: float,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
    max_evaluations: int,
) -> tuple[int, int, bool]:
    """Move nodes from a first-in first-out queue until it is empty.

    ``initial`` lists distinct nodes in queue order; ``inq`` flags queued
    nodes and ``queue`` is a buffer of one slot per node. A move to an empty
    community takes the fresh id ``next_free[0]``. Every accepted move gains
    more than ``eps_v``, so the queue empties; as a guard, the loop also stops
    after ``max_evaluations`` evaluations, leaving every applied move complete.
    Returns the number of evaluations, the number of moves, and whether the
    guard stopped the loop.
    """
    n = strengths.shape[0]
    capacity = totals.shape[0]
    head = 0
    count = 0
    for index in range(initial.shape[0]):
        node = initial[index]
        queue[count] = node
        inq[node] = 1
        count += 1
    evaluations = 0
    moves = 0
    capped = False
    while count > 0:
        if evaluations >= max_evaluations:
            capped = True
            break
        v = queue[head]
        head += 1
        if head == n:
            head = 0
        count -= 1
        inq[v] = 0
        evaluations += 1
        target, _gain = best_move(
            v,
            indptr,
            indices,
            weights,
            labels,
            totals,
            sizes,
            strengths,
            null_coef,
            eps_coef,
            keys,
            vals,
            used,
            shift,
        )
        if target == STAY:
            continue
        if target == NEW_COMMUNITY:
            if next_free[0] >= capacity:
                continue
            target = next_free[0]
            next_free[0] += 1
        source = labels[v]
        totals[source] -= strengths[v]
        sizes[source] -= 1
        totals[target] += strengths[v]
        sizes[target] += 1
        labels[v] = target
        moves += 1
        for position in range(indptr[v], indptr[v + 1]):
            neighbour = indices[position]
            if neighbour != v and labels[neighbour] != target and inq[neighbour] == 0:
                tail = head + count
                if tail >= n:
                    tail -= n
                queue[tail] = neighbour
                inq[neighbour] = 1
                count += 1
    return evaluations, moves, capped


@njit(cache=True, nogil=True)
def _group_of(sweep_key: np.uint64, node: int, group_shift: np.uint64) -> int:
    """Return the group of ``node`` in one sweep (the top bits of a hash)."""
    return int(np.int64(mix64(sweep_key + np.uint64(node) * GOLDEN) >> group_shift))


@njit(cache=True, nogil=True, parallel=True)
def _bucket_active_by_group(
    active: np.ndarray,
    sweep_key: np.uint64,
    group_shift: np.uint64,
    n_groups: int,
    n_workers: int,
    parallel_min_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Order the active nodes by group, ascending node id within a group."""
    n = active.shape[0]
    n_chunks = (n + BUCKET_CHUNK - 1) // BUCKET_CHUNK
    n_run = n_workers if n_workers > 1 and n >= 16 * parallel_min_items else 1
    counts = np.zeros((n_chunks, n_groups), dtype=np.int64)
    for worker in prange(n_run):
        for chunk in range(worker, n_chunks, n_run):
            row = counts[chunk]
            for node in range(chunk * BUCKET_CHUNK, min(n, (chunk + 1) * BUCKET_CHUNK)):
                if active[node] != 0:
                    row[_group_of(sweep_key, node, group_shift)] += 1
    group_start = np.zeros(n_groups + 1, dtype=np.int64)
    running = np.int64(0)
    for group in range(n_groups):
        group_start[group] = running
        for scan_chunk in range(n_chunks):
            count = counts[scan_chunk, group]
            counts[scan_chunk, group] = running
            running += count
    group_start[n_groups] = running
    order = np.empty(running, dtype=np.int32)
    for worker in prange(n_run):
        for chunk in range(worker, n_chunks, n_run):
            row = counts[chunk]
            for node in range(chunk * BUCKET_CHUNK, min(n, (chunk + 1) * BUCKET_CHUNK)):
                if active[node] != 0:
                    group_index = _group_of(sweep_key, node, group_shift)
                    order[row[group_index]] = node
                    row[group_index] += 1
    return order, group_start


@njit(cache=True, nogil=True)
def _decide_node(
    position: int,
    nodes: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    strengths: np.ndarray,
    null_coef: float,
    eps_coef: float,
    active: np.ndarray,
    target: np.ndarray,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
) -> None:
    """Decide the move of one group member from the group-start state."""
    v = nodes[position]
    active[v] = 0
    choice, _gain = best_move(
        v,
        indptr,
        indices,
        weights,
        labels,
        totals,
        sizes,
        strengths,
        null_coef,
        eps_coef,
        keys,
        vals,
        used,
        shift,
    )
    if choice >= 0:
        own = labels[v]
        if sizes[own] == 1 and sizes[choice] == 1 and choice > own:
            choice = STAY
    target[position] = choice


@njit(cache=True, nogil=True, parallel=True)
def _decide_group(
    nodes: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    strengths: np.ndarray,
    null_coef: float,
    eps_coef: float,
    active: np.ndarray,
    target: np.ndarray,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
    n_workers: int,
    parallel_min_items: int,
) -> None:
    """Decide every move of a group; writes only ``active`` and ``target``."""
    n_nodes = nodes.shape[0]
    if n_workers <= 1 or n_nodes < parallel_min_items:
        row_keys = keys[0]
        row_vals = vals[0]
        row_used = used[0]
        for position in range(n_nodes):
            _decide_node(
                position,
                nodes,
                indptr,
                indices,
                weights,
                labels,
                totals,
                sizes,
                strengths,
                null_coef,
                eps_coef,
                active,
                target,
                row_keys,
                row_vals,
                row_used,
                shift,
            )
        return
    n_blocks = (n_nodes + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        worker_keys = keys[worker]
        worker_vals = vals[worker]
        worker_used = used[worker]
        for block in range(worker, n_blocks, n_workers):
            for position in range(
                block * WORKER_BLOCK, min(n_nodes, (block + 1) * WORKER_BLOCK)
            ):
                _decide_node(
                    position,
                    nodes,
                    indptr,
                    indices,
                    weights,
                    labels,
                    totals,
                    sizes,
                    strengths,
                    null_coef,
                    eps_coef,
                    active,
                    target,
                    worker_keys,
                    worker_vals,
                    worker_used,
                    shift,
                )


@njit(cache=True, nogil=True)
def _count_block_moves(target: np.ndarray, low: int, high: int) -> tuple[int, int]:
    """Count the moves to existing communities and the new-community requests."""
    existing = 0
    requests = 0
    for position in range(low, high):
        choice = target[position]
        if choice >= 0:
            existing += 1
        elif choice == NEW_COMMUNITY:
            requests += 1
    return existing, requests


@njit(cache=True, nogil=True)
def _write_block_moves(
    low: int,
    high: int,
    nodes: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    strengths: np.ndarray,
    first_mover: int,
    first_new_id: int,
    allow_new: bool,
    n_moves: int,
    movers: np.ndarray,
    item_keys: np.ndarray,
    item_dq: np.ndarray,
    item_ds: np.ndarray,
) -> None:
    """Record the moves of one block in group order and relabel the movers."""
    mover = first_mover
    new_id = first_new_id
    for position in range(low, high):
        choice = target[position]
        if choice == STAY:
            continue
        if choice == NEW_COMMUNITY:
            if not allow_new:
                continue
            choice = new_id
            new_id += 1
        v = nodes[position]
        movers[mover] = v
        item_keys[mover] = labels[v]
        item_dq[mover] = -strengths[v]
        item_ds[mover] = -1
        item_keys[n_moves + mover] = choice
        item_dq[n_moves + mover] = strengths[v]
        item_ds[n_moves + mover] = 1
        labels[v] = choice
        mover += 1


@njit(cache=True, nogil=True, parallel=True)
def _compact_group(
    nodes: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    strengths: np.ndarray,
    next_free: int,
    capacity: int,
    n_workers: int,
    parallel_min_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect the moves of a group in group order and relabel the movers.

    Returns the moved nodes, the scatter items that update the community
    totals (``-q`` from the old community, ``+q`` to the new one), and the
    number of fresh community ids handed out. The ``j``-th new-community
    request of the group gets id ``next_free + j``; if the group would exceed
    the id capacity, all of its new-community requests stay instead.
    """
    n_nodes = nodes.shape[0]
    n_blocks = (n_nodes + COMPACT_BLOCK - 1) // COMPACT_BLOCK
    existing_counts = np.zeros(n_blocks, dtype=np.int64)
    request_counts = np.zeros(n_blocks, dtype=np.int64)
    small = n_workers <= 1 or n_nodes < parallel_min_items
    if small:
        for serial_block in range(n_blocks):
            serial_existing, serial_requests = _count_block_moves(
                target,
                serial_block * COMPACT_BLOCK,
                min(n_nodes, (serial_block + 1) * COMPACT_BLOCK),
            )
            existing_counts[serial_block] = serial_existing
            request_counts[serial_block] = serial_requests
    else:
        for worker in prange(n_workers):
            for block in range(worker, n_blocks, n_workers):
                block_existing, block_requests = _count_block_moves(
                    target,
                    block * COMPACT_BLOCK,
                    min(n_nodes, (block + 1) * COMPACT_BLOCK),
                )
                existing_counts[block] = block_existing
                request_counts[block] = block_requests
    total_requests = int(np.int64(0))
    for counted in range(n_blocks):
        total_requests += request_counts[counted]
    allow_new = next_free + total_requests <= capacity
    first_movers = np.empty(n_blocks, dtype=np.int64)
    first_new_ids = np.empty(n_blocks, dtype=np.int64)
    n_moves = int(np.int64(0))
    granted = int(np.int64(0))
    for scanned in range(n_blocks):
        first_movers[scanned] = n_moves
        first_new_ids[scanned] = next_free + granted
        n_moves += existing_counts[scanned]
        if allow_new:
            n_moves += request_counts[scanned]
            granted += request_counts[scanned]
    movers = np.empty(n_moves, dtype=np.int32)
    item_keys = np.empty(2 * n_moves, dtype=np.int32)
    item_dq = np.empty(2 * n_moves, dtype=np.int64)
    item_ds = np.empty(2 * n_moves, dtype=np.int32)
    if small:
        for written in range(n_blocks):
            _write_block_moves(
                written * COMPACT_BLOCK,
                min(n_nodes, (written + 1) * COMPACT_BLOCK),
                nodes,
                target,
                labels,
                strengths,
                first_movers[written],
                first_new_ids[written],
                allow_new,
                n_moves,
                movers,
                item_keys,
                item_dq,
                item_ds,
            )
    else:
        for worker in prange(n_workers):
            for block in range(worker, n_blocks, n_workers):
                _write_block_moves(
                    block * COMPACT_BLOCK,
                    min(n_nodes, (block + 1) * COMPACT_BLOCK),
                    nodes,
                    target,
                    labels,
                    strengths,
                    first_movers[block],
                    first_new_ids[block],
                    allow_new,
                    n_moves,
                    movers,
                    item_keys,
                    item_dq,
                    item_ds,
                )
    return movers, item_keys, item_dq, item_ds, granted


@njit(cache=True, nogil=True)
def _activate_from(
    v: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
) -> None:
    """Activate the neighbours of a moved node outside its new community."""
    own = labels[v]
    for position in range(indptr[v], indptr[v + 1]):
        neighbour = indices[position]
        if neighbour != v and labels[neighbour] != own:
            active[neighbour] = 1


@njit(cache=True, nogil=True, parallel=True)
def _activate_neighbours(
    movers: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    n_workers: int,
    parallel_min_items: int,
) -> None:
    """Activate the neighbours of every mover after all labels are updated.

    Several workers may store the constant 1 into the same flag; every store
    writes the same value, so the final flags do not depend on the order.
    """
    n_movers = movers.shape[0]
    if n_workers <= 1 or n_movers < parallel_min_items:
        for index in range(n_movers):
            _activate_from(movers[index], indptr, indices, labels, active)
        return
    n_blocks = (n_movers + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            for position in range(
                block * WORKER_BLOCK, min(n_movers, (block + 1) * WORKER_BLOCK)
            ):
                _activate_from(movers[position], indptr, indices, labels, active)


@njit(cache=True, nogil=True)
def move_nodes_level(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    strengths: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    next_free: np.ndarray,
    seed: np.uint64,
    iteration: int,
    level: int,
    null_coef: float,
    eps_coef: float,
    n_workers: int,
    sequential_level_entries: int,
    tail_divisor: int,
    max_parallel_sweeps: int,
    group_bits: int,
    parallel_min_items: int,
    queue_evaluations_per_node: int,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
) -> np.ndarray:
    """Run the local moving phase of one level in place.

    ``labels`` holds community ids in ``[0, capacity)`` with ``totals`` and
    ``sizes`` indexed by id; ``next_free[0]`` is the first unused id. Levels
    with fewer than ``sequential_level_entries`` stored entries run the
    sequential queue from a random permutation. Larger levels run up to
    ``max_parallel_sweeps`` sweeps of ``2**group_bits`` random groups and
    finish with the sequential queue once fewer than
    ``n / tail_divisor`` nodes are active. The sequential queue of a level
    stops after ``queue_evaluations_per_node * n`` evaluations at most (a
    hang guard that well-behaved inputs never reach).

    Returns int64 statistics: evaluations, moves, parallel sweeps, nodes in
    the final sequential queue, 1 if the group scheme ran (else 0), and 1 if
    the sequential queue stopped at its evaluation cap (else 0).
    """
    n = strengths.shape[0]
    check_worker_rows(keys.shape[0], n_workers)
    check_worker_rows(vals.shape[0], n_workers)
    check_worker_rows(used.shape[0], n_workers)
    stats = np.zeros(6, dtype=np.int64)
    queue = np.empty(max(n, 1), dtype=np.int32)
    row_keys = keys[0]
    row_vals = vals[0]
    row_used = used[0]
    max_evaluations = queue_evaluations_per_node * n
    if indptr[n] < sequential_level_entries:
        order = random_permutation(
            n, stream_key(seed, iteration, level, np.int64(PHASE_MOVE_ORDER))
        )
        inq = np.zeros(n, dtype=np.uint8)
        evaluations, moves, capped = move_nodes_queue(
            indptr,
            indices,
            weights,
            strengths,
            labels,
            totals,
            sizes,
            next_free,
            order,
            inq,
            queue,
            null_coef,
            eps_coef,
            row_keys,
            row_vals,
            row_used,
            shift,
            max_evaluations,
        )
        stats[0] = evaluations
        stats[1] = moves
        stats[3] = n
        stats[5] = 1 if capped else 0
        return stats
    stats[4] = 1
    capacity = totals.shape[0]
    active = np.zeros(n, dtype=np.uint8)
    for node in range(n):
        for position in range(indptr[node], indptr[node + 1]):
            if indices[position] != node:
                active[node] = 1
                break
    n_groups = 1 << group_bits
    group_shift = np.uint64(64 - group_bits)
    groups_key = stream_key(seed, iteration, level, np.int64(PHASE_GROUPS))
    target_buffer = np.empty(max(n, 1), dtype=np.int32)
    for sweep in range(max_parallel_sweeps):
        sweep_key = derive_key(groups_key, sweep)
        order, group_start = _bucket_active_by_group(
            active, sweep_key, group_shift, n_groups, n_workers, parallel_min_items
        )
        n_active = group_start[n_groups]
        if n_active == 0:
            return stats
        if tail_divisor * n_active < n:
            break
        for group in range(n_groups):
            low = group_start[group]
            high = group_start[group + 1]
            if low == high:
                continue
            nodes = order[low:high]
            target = target_buffer[: high - low]
            _decide_group(
                nodes,
                indptr,
                indices,
                weights,
                labels,
                totals,
                sizes,
                strengths,
                null_coef,
                eps_coef,
                active,
                target,
                keys,
                vals,
                used,
                shift,
                n_workers,
                parallel_min_items,
            )
            movers, item_keys, item_dq, item_ds, granted = _compact_group(
                nodes,
                target,
                labels,
                strengths,
                next_free[0],
                capacity,
                n_workers,
                parallel_min_items,
            )
            next_free[0] += granted
            stats[0] += high - low
            stats[1] += movers.shape[0]
            if movers.shape[0] > 0:
                scatter_add(
                    item_keys,
                    item_dq,
                    item_ds,
                    totals,
                    sizes,
                    n_workers,
                    parallel_min_items,
                )
                _activate_neighbours(
                    movers,
                    indptr,
                    indices,
                    labels,
                    active,
                    n_workers,
                    parallel_min_items,
                )
        stats[2] += 1
    n_tail = 0
    for node in range(n):
        if active[node] != 0:
            n_tail += 1
    tail = np.empty(n_tail, dtype=np.int32)
    filled = 0
    for node in range(n):
        if active[node] != 0:
            tail[filled] = node
            filled += 1
    stats[3] = n_tail
    evaluations, moves, capped = move_nodes_queue(
        indptr,
        indices,
        weights,
        strengths,
        labels,
        totals,
        sizes,
        next_free,
        tail,
        active,
        queue,
        null_coef,
        eps_coef,
        row_keys,
        row_vals,
        row_used,
        shift,
        max_evaluations,
    )
    stats[0] += evaluations
    stats[1] += moves
    stats[5] = 1 if capped else 0
    return stats
