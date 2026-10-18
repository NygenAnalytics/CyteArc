"""Refinement phase of CyteArc's native Leiden backend.

This is RefinePartition and MergeNodesSubset of the Leiden algorithm of
Traag, Waltman and van Eck (2019), Algorithm A.2 lines 26-43, for modularity.
Inside every community ``S`` of the partition left by local moving, the
refined partition starts from singletons. Nodes that are well connected to
the rest of ``S`` are visited in random order; a node that is still a
singleton may join a refined community ``D`` of a neighbour in ``S`` when
``D`` is itself well connected to ``S`` and the join does not lower the
quality. Among the eligible communities, and staying alone, one is drawn
with probability proportional to ``exp(gain / theta)``.

In the notation of ``_leiden_moving`` (normalised weights, fixed-point
strengths, ``c_v = gamma * k_v / 2m``):

- node ``v`` is well connected when ``a(v, S - v) >= c_v * (K_S - k_v)``;
- refined community ``D`` is well connected when
  ``E(D, S - D) >= (gamma * K_D / 2m) * (K_S - K_D)``;
- the gain of moving the singleton ``v`` into ``D`` is
  ``a(v, D) - c_v * K_D``, the paper's quality difference for modularity in
  units of edge weight;
- ``theta`` is 0.01 in units of the mean off-diagonal edge weight.

``E(D, S - D)`` is kept per refined community and updated exactly when a
singleton joins. Only edges inside ``S`` count towards ``a`` and ``E``, while
strengths and ``2m`` are global. Candidates are ordered by id with staying
first, so each draw is reproducible. Communities are independent of each
other: each one runs sequentially with its own random stream, keyed by its
id, and the communities are spread over the workers, so the result does not
depend on the number of threads.

Reference:
    Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9, 5233 (2019).
    doi:10.1038/s41598-019-41695-z
"""

import numpy as np
from numba import njit, prange

from ._leiden_primitives import (
    acc_add,
    acc_find,
    acc_reset,
    check_worker_rows,
    derive_key,
    sort_ids,
    uniform01,
)


@njit(cache=True, nogil=True)
def merge_nodes_subset(
    c: int,
    low: int,
    high: int,
    members: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    strengths: np.ndarray,
    labels: np.ndarray,
    null_coef: float,
    theta_w: float,
    key: np.uint64,
    refined: np.ndarray,
    rsize: np.ndarray,
    kref: np.ndarray,
    eext: np.ndarray,
    rlist: np.ndarray,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
    cand_ids: np.ndarray,
    cand_gain: np.ndarray,
    cand_a: np.ndarray,
) -> int:
    """Refine community ``c`` and return its number of refined communities.

    ``members[low:high]`` lists the nodes of ``c`` in ascending order. Refined
    community ids are founder node ids. Writes ``refined``, ``rsize``,
    ``kref``, ``eext`` and ``rlist`` only at the members of ``c``.
    """
    community_total = np.int64(0)
    for position in range(low, high):
        v = members[position]
        refined[v] = v
        rsize[v] = 1
        kref[v] = strengths[v]
        community_total += strengths[v]
    if high - low == 1:
        eext[members[low]] = 0.0
        return 1
    n_connected = 0
    for position in range(low, high):
        v = members[position]
        inside = 0.0
        for edge in range(indptr[v], indptr[v + 1]):
            neighbour = indices[edge]
            if neighbour != v and labels[neighbour] == c:
                inside += weights[edge]
        eext[v] = inside
        c_v = null_coef * np.float64(strengths[v])
        if inside >= c_v * np.float64(community_total - strengths[v]):
            rlist[low + n_connected] = v
            n_connected += 1
    counter = int(np.int64(0))
    for position in range(n_connected - 1, 0, -1):
        other = int(uniform01(key, counter) * (position + 1))
        counter += 1
        if other > position:
            other = position
        held = rlist[low + position]
        rlist[low + position] = rlist[low + other]
        rlist[low + other] = held
    merges = 0
    for visit in range(n_connected):
        v = rlist[low + visit]
        if refined[v] != v or rsize[v] != 1:
            continue
        n_used = int(np.int64(0))
        for edge in range(indptr[v], indptr[v + 1]):
            neighbour = indices[edge]
            if neighbour != v and labels[neighbour] == c:
                n_used = acc_add(
                    keys, vals, used, n_used, refined[neighbour], weights[edge], shift
                )
        c_v = null_coef * np.float64(strengths[v])
        n_candidates = int(np.int64(0))
        for index in range(n_used):
            slot = used[index]
            community = keys[slot]
            community_strength = kref[community]
            threshold = (null_coef * np.float64(community_strength)) * np.float64(
                community_total - community_strength
            )
            if eext[community] >= threshold:
                if vals[slot] - c_v * np.float64(community_strength) >= 0.0:
                    cand_ids[n_candidates] = community
                    n_candidates += 1
        if n_candidates == 0:
            acc_reset(keys, used, n_used)
            continue
        sort_ids(cand_ids, n_candidates)
        best_gain = 0.0
        for index in range(n_candidates):
            community = cand_ids[index]
            weight_to = vals[acc_find(keys, community, shift)]
            gain = weight_to - c_v * np.float64(kref[community])
            cand_a[index] = weight_to
            cand_gain[index] = gain
            if gain > best_gain:
                best_gain = gain
        acc_reset(keys, used, n_used)
        stay_weight = np.exp((0.0 - best_gain) / theta_w)
        total = stay_weight
        for index in range(n_candidates):
            weight = np.exp((cand_gain[index] - best_gain) / theta_w)
            cand_gain[index] = weight
            total += weight
        draw = uniform01(key, counter) * total
        counter += 1
        cumulative = stay_weight
        chosen = -1
        if draw >= cumulative:
            for index in range(n_candidates):
                cumulative += cand_gain[index]
                chosen = index
                if draw < cumulative:
                    break
        if chosen < 0:
            continue
        joined = cand_ids[chosen]
        refined[v] = joined
        rsize[joined] += 1
        rsize[v] = 0
        kref[joined] += strengths[v]
        eext[joined] = eext[joined] + eext[v] - 2.0 * cand_a[chosen]
        merges += 1
    return high - low - merges


@njit(cache=True, nogil=True)
def community_order(member_start: np.ndarray) -> np.ndarray:
    """Order communities by size descending, ties by id ascending."""
    n_communities = member_start.shape[0] - 1
    largest = 0
    for community in range(n_communities):
        size = member_start[community + 1] - member_start[community]
        if size > largest:
            largest = size
    cursor = np.zeros(largest + 2, dtype=np.int64)
    for community in range(n_communities):
        size = member_start[community + 1] - member_start[community]
        cursor[largest - size + 1] += 1
    for slot in range(largest + 1):
        cursor[slot + 1] += cursor[slot]
    order = np.empty(n_communities, dtype=np.int32)
    for community in range(n_communities):
        size = member_start[community + 1] - member_start[community]
        order[cursor[largest - size]] = community
        cursor[largest - size] += 1
    return order


@njit(cache=True, nogil=True, parallel=True)
def refine_partition(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    strengths: np.ndarray,
    labels: np.ndarray,
    n_comm: int,
    member_start: np.ndarray,
    members: np.ndarray,
    comm_order: np.ndarray,
    null_coef: float,
    theta_w: float,
    refine_key: np.uint64,
    n_workers: int,
    refined: np.ndarray,
    rsize: np.ndarray,
    kref: np.ndarray,
    eext: np.ndarray,
    rlist: np.ndarray,
    n_refined: np.ndarray,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
    cand_ids: np.ndarray,
    cand_gain: np.ndarray,
    cand_a: np.ndarray,
    parallel_min_items: int,
) -> None:
    """Refine every community; communities run in parallel, largest first.

    Worker ``w`` takes positions ``w, w + n_workers, ...`` of ``comm_order``.
    Community ``c`` draws from the stream ``derive_key(refine_key, c)`` and
    writes ``n_refined[c]`` and the refinement arrays of its members only.
    """
    check_worker_rows(keys.shape[0], n_workers)
    check_worker_rows(vals.shape[0], n_workers)
    check_worker_rows(used.shape[0], n_workers)
    check_worker_rows(cand_ids.shape[0], n_workers)
    check_worker_rows(cand_gain.shape[0], n_workers)
    check_worker_rows(cand_a.shape[0], n_workers)
    n = strengths.shape[0]
    if n_workers <= 1 or n < parallel_min_items:
        for position in range(n_comm):
            community = comm_order[position]
            n_refined[community] = merge_nodes_subset(
                community,
                member_start[community],
                member_start[community + 1],
                members,
                indptr,
                indices,
                weights,
                strengths,
                labels,
                null_coef,
                theta_w,
                derive_key(refine_key, np.int64(community)),
                refined,
                rsize,
                kref,
                eext,
                rlist,
                keys[0],
                vals[0],
                used[0],
                shift,
                cand_ids[0],
                cand_gain[0],
                cand_a[0],
            )
        return
    for worker in prange(n_workers):
        worker_keys = keys[worker]
        worker_vals = vals[worker]
        worker_used = used[worker]
        worker_ids = cand_ids[worker]
        worker_gain = cand_gain[worker]
        worker_a = cand_a[worker]
        for position in range(worker, n_comm, n_workers):
            community = comm_order[position]
            n_refined[community] = merge_nodes_subset(
                community,
                member_start[community],
                member_start[community + 1],
                members,
                indptr,
                indices,
                weights,
                strengths,
                labels,
                null_coef,
                theta_w,
                derive_key(refine_key, np.int64(community)),
                refined,
                rsize,
                kref,
                eext,
                rlist,
                worker_keys,
                worker_vals,
                worker_used,
                shift,
                worker_ids,
                worker_gain,
                worker_a,
            )
