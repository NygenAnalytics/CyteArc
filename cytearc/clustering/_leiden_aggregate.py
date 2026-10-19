"""Aggregation and bookkeeping of CyteArc's native Leiden backend.

In the Leiden algorithm of Traag, Waltman and van Eck (2019), Algorithm A.2
lines 6-8 and 44-48, every refined community becomes one node of the next
level's graph, while the unrefined partition becomes that graph's initial
partition. With the membership matrix ``S`` of the refined partition the
aggregate adjacency is ``A' = S^T A S`` with its diagonal (internal edges
become self-loops that count every internal edge twice) and the aggregate
strengths are ``S^T q``, so the quality of every partition is preserved.

The aggregate graph is built as a compressed sparse row matrix with prefix
sums and per-worker hash tables, the approach of GVE-Leiden (Sahu, Kothapalli
and Banerjee, 2024), but with an exact counting pass instead of over-allocated
rows: the first pass counts the distinct neighbours of each aggregate node, a
prefix sum turns the counts into row offsets, and the second pass accumulates
the weights and writes each row sorted by column. Each row is summed in a
fixed order (members ascending, then CSR order), so the result does not
depend on the number of threads.

This module also holds the renumbering steps, the safety-stop split into
connected components and the final numbering of the labels (largest
community first, ties by smallest node index).

References:
    Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9, 5233 (2019).
    doi:10.1038/s41598-019-41695-z

    Sahu S, Kothapalli K, Banerjee DS. Fast Leiden algorithm for community
    detection in shared memory setting. Proceedings of the 53rd International
    Conference on Parallel Processing (ICPP 2024), 11-20.
    doi:10.1145/3673038.3673146 (preprint arXiv:2312.13936)
"""

import numpy as np
from numba import njit, prange

from ._leiden_primitives import (
    WORKER_BLOCK,
    acc_add,
    acc_find,
    acc_insert,
    acc_reset,
    bit_length,
    check_worker_rows,
    compact_ids,
    gather,
    scatter_add,
    sort_ids,
    stable_bucket_order,
)


@njit(cache=True, nogil=True)
def renumber_partition(
    flat: np.ndarray, strengths: np.ndarray, n_workers: int, parallel_min_items: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Renumber a partition of the base nodes and compute its community state.

    ``flat`` holds int32 ids in ``[0, n)``. Used ids get new ids ``0..C-1`` in
    ascending order of the old id. Returns the new labels and the community
    totals and sizes, sized for ``2n`` community ids, and ``C``.
    """
    n = flat.shape[0]
    sizes_by_id = np.zeros(n, dtype=np.int32)
    totals_by_id = np.zeros(n, dtype=np.int64)
    ones = np.ones(n, dtype=np.int32)
    scatter_add(
        flat, strengths, ones, totals_by_id, sizes_by_id, n_workers, parallel_min_items
    )
    new_ids, n_comm = compact_ids(sizes_by_id, n_workers, parallel_min_items)
    labels = gather(new_ids, flat, n_workers)
    totals = np.zeros(2 * n, dtype=np.int64)
    sizes = np.zeros(2 * n, dtype=np.int32)
    for old in range(n):
        if sizes_by_id[old] > 0:
            totals[new_ids[old]] = totals_by_id[old]
            sizes[new_ids[old]] = sizes_by_id[old]
    return labels, totals, sizes, n_comm


@njit(cache=True, nogil=True)
def compact_communities(
    labels: np.ndarray,
    totals: np.ndarray,
    sizes: np.ndarray,
    n_workers: int,
    parallel_min_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Renumber the used community ids after local moving.

    Used ids (``sizes > 0``) get new ids ``0..C-1`` in ascending order of the
    old id; totals and sizes move along. Returns the new labels, the compact
    totals and sizes (length ``C``) and ``C``.
    """
    new_ids, n_comm = compact_ids(sizes, n_workers, parallel_min_items)
    new_labels = gather(new_ids, labels, n_workers)
    compact_totals = np.empty(n_comm, dtype=np.int64)
    compact_sizes = np.empty(n_comm, dtype=np.int32)
    for old in range(sizes.shape[0]):
        if sizes[old] > 0:
            compact_totals[new_ids[old]] = totals[old]
            compact_sizes[new_ids[old]] = sizes[old]
    return new_labels, compact_totals, compact_sizes, n_comm


@njit(cache=True, nogil=True)
def compact_refined(
    refined: np.ndarray, rsize: np.ndarray, n_workers: int, parallel_min_items: int
) -> tuple[np.ndarray, int]:
    """Number the refined communities ``0..R-1`` by ascending founder id.

    Returns the aggregate node of every node and ``R``.
    """
    new_ids, n_refined_total = compact_ids(rsize, n_workers, parallel_min_items)
    ragg = gather(new_ids, refined, n_workers)
    return ragg, n_refined_total


@njit(cache=True, nogil=True, parallel=True)
def aggregate_node_info(
    strengths: np.ndarray,
    labels: np.ndarray,
    indptr: np.ndarray,
    agg_start: np.ndarray,
    agg_members: np.ndarray,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the strength, initial label and row-length bound of each aggregate node.

    The strength is the exact sum of the members' strengths, the initial label
    is the community of the first member (all members share it), and the
    bound is the sum of the members' row lengths.
    """
    n_agg = agg_start.shape[0] - 1
    new_strengths = np.empty(n_agg, dtype=np.int64)
    new_labels = np.empty(n_agg, dtype=np.int32)
    row_bounds = np.empty(n_agg, dtype=np.int64)
    n_blocks = (n_agg + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            for node in range(
                block * WORKER_BLOCK, min(n_agg, (block + 1) * WORKER_BLOCK)
            ):
                strength_total = np.int64(0)
                bound = np.int64(0)
                for position in range(agg_start[node], agg_start[node + 1]):
                    member = agg_members[position]
                    strength_total += strengths[member]
                    bound += indptr[member + 1] - indptr[member]
                new_strengths[node] = strength_total
                row_bounds[node] = bound
                new_labels[node] = labels[agg_members[agg_start[node]]]
    return new_strengths, new_labels, row_bounds


@njit(cache=True, nogil=True)
def aggregation_order(
    row_bounds: np.ndarray, n_workers: int, parallel_min_items: int
) -> np.ndarray:
    """Order aggregate nodes by decreasing ``bit_length`` of their row bound.

    Rows with the most work come first, which balances the workers. Ties keep
    ascending order. The aggregate graph does not depend on this order.
    """
    n_agg = row_bounds.shape[0]
    bucket_keys = np.empty(n_agg, dtype=np.int32)
    for node in range(n_agg):
        bucket_keys[node] = 63 - bit_length(row_bounds[node])
    _start, order = stable_bucket_order(
        bucket_keys, np.int64(64), n_workers, parallel_min_items
    )
    return order


@njit(cache=True, nogil=True)
def _count_row(
    node: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    ragg: np.ndarray,
    agg_start: np.ndarray,
    agg_members: np.ndarray,
    keys: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
) -> int:
    """Return the number of distinct aggregate neighbours of one aggregate node."""
    n_used = int(np.int64(0))
    for position in range(agg_start[node], agg_start[node + 1]):
        member = agg_members[position]
        for edge in range(indptr[member], indptr[member + 1]):
            n_used = acc_insert(keys, used, n_used, ragg[indices[edge]], shift)
    acc_reset(keys, used, n_used)
    return n_used


@njit(cache=True, nogil=True, parallel=True)
def aggregate_counts(
    indptr: np.ndarray,
    indices: np.ndarray,
    ragg: np.ndarray,
    agg_start: np.ndarray,
    agg_members: np.ndarray,
    agg_order: np.ndarray,
    n_workers: int,
    keys: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
    parallel_min_items: int,
) -> np.ndarray:
    """First aggregation pass: the row length of every aggregate node."""
    check_worker_rows(keys.shape[0], n_workers)
    check_worker_rows(used.shape[0], n_workers)
    n_agg = agg_start.shape[0] - 1
    row_lengths = np.zeros(n_agg, dtype=np.int64)
    if n_workers <= 1 or n_agg < parallel_min_items:
        row_keys = keys[0]
        row_used = used[0]
        for serial_position in range(n_agg):
            node = agg_order[serial_position]
            row_lengths[node] = _count_row(
                node,
                indptr,
                indices,
                ragg,
                agg_start,
                agg_members,
                row_keys,
                row_used,
                shift,
            )
        return row_lengths
    n_blocks = (n_agg + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        worker_keys = keys[worker]
        worker_used = used[worker]
        for block in range(worker, n_blocks, n_workers):
            for position in range(
                block * WORKER_BLOCK, min(n_agg, (block + 1) * WORKER_BLOCK)
            ):
                aggregate = agg_order[position]
                row_lengths[aggregate] = _count_row(
                    aggregate,
                    indptr,
                    indices,
                    ragg,
                    agg_start,
                    agg_members,
                    worker_keys,
                    worker_used,
                    shift,
                )
    return row_lengths


@njit(cache=True, nogil=True)
def _fill_row(
    node: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    ragg: np.ndarray,
    agg_start: np.ndarray,
    agg_members: np.ndarray,
    new_indptr: np.ndarray,
    new_indices: np.ndarray,
    new_weights: np.ndarray,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
) -> None:
    """Accumulate and write the sorted row of one aggregate node."""
    n_used = int(np.int64(0))
    for position in range(agg_start[node], agg_start[node + 1]):
        member = agg_members[position]
        for edge in range(indptr[member], indptr[member + 1]):
            n_used = acc_add(
                keys, vals, used, n_used, ragg[indices[edge]], weights[edge], shift
            )
    start = new_indptr[node]
    row = new_indices[start : start + n_used]
    for index in range(n_used):
        row[index] = keys[used[index]]
    sort_ids(row, n_used)
    for index in range(n_used):
        new_weights[start + index] = vals[acc_find(keys, row[index], shift)]
    acc_reset(keys, used, n_used)


@njit(cache=True, nogil=True, parallel=True)
def aggregate_fill(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    ragg: np.ndarray,
    agg_start: np.ndarray,
    agg_members: np.ndarray,
    agg_order: np.ndarray,
    new_indptr: np.ndarray,
    new_indices: np.ndarray,
    new_weights: np.ndarray,
    n_workers: int,
    keys: np.ndarray,
    vals: np.ndarray,
    used: np.ndarray,
    shift: np.uint64,
    parallel_min_items: int,
) -> None:
    """Second aggregation pass: write the rows of ``A' = S^T A S``."""
    check_worker_rows(keys.shape[0], n_workers)
    check_worker_rows(vals.shape[0], n_workers)
    check_worker_rows(used.shape[0], n_workers)
    n_agg = agg_start.shape[0] - 1
    if n_workers <= 1 or n_agg < parallel_min_items:
        row_keys = keys[0]
        row_vals = vals[0]
        row_used = used[0]
        for serial_position in range(n_agg):
            _fill_row(
                agg_order[serial_position],
                indptr,
                indices,
                weights,
                ragg,
                agg_start,
                agg_members,
                new_indptr,
                new_indices,
                new_weights,
                row_keys,
                row_vals,
                row_used,
                shift,
            )
        return
    n_blocks = (n_agg + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        worker_keys = keys[worker]
        worker_vals = vals[worker]
        worker_used = used[worker]
        for block in range(worker, n_blocks, n_workers):
            for position in range(
                block * WORKER_BLOCK, min(n_agg, (block + 1) * WORKER_BLOCK)
            ):
                _fill_row(
                    agg_order[position],
                    indptr,
                    indices,
                    weights,
                    ragg,
                    agg_start,
                    agg_members,
                    new_indptr,
                    new_indices,
                    new_weights,
                    worker_keys,
                    worker_vals,
                    worker_used,
                    shift,
                )


@njit(cache=True, nogil=True)
def next_level_state(
    compact_totals: np.ndarray, n_refined: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the community totals and sizes of the next level.

    The communities keep their ids and exact totals; the size of a community
    is its number of refined communities, the next level's nodes in it.
    """
    totals = np.zeros(2 * n_nodes, dtype=np.int64)
    sizes = np.zeros(2 * n_nodes, dtype=np.int32)
    for community in range(compact_totals.shape[0]):
        totals[community] = compact_totals[community]
        sizes[community] = n_refined[community]
    return totals, sizes


@njit(cache=True, nogil=True)
def _find_root(parent: np.ndarray, node: int) -> int:
    """Return the root of ``node`` in a union-find forest, halving the path."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True, nogil=True)
def split_into_components(
    indptr: np.ndarray, indices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Split every community into its connected components.

    Only edges whose two ends share a label are kept. Each component is
    labelled by its smallest node (a union-find in which the smaller root
    always wins). Used as a safety stop when refinement merges nothing.
    """
    n = labels.shape[0]
    parent = np.arange(n).astype(np.int32)
    for node in range(n):
        for edge in range(indptr[node], indptr[node + 1]):
            neighbour = indices[edge]
            if neighbour > node and labels[neighbour] == labels[node]:
                root_a = _find_root(parent, node)
                root_b = _find_root(parent, neighbour)
                if root_a < root_b:
                    parent[root_b] = root_a
                elif root_b < root_a:
                    parent[root_a] = root_b
    components = np.empty(n, dtype=np.int32)
    for node in range(n):
        components[node] = _find_root(parent, node)
    return components


@njit(cache=True, nogil=True)
def final_labels(flat: np.ndarray, n_comm: int, n_workers: int) -> np.ndarray:
    """Number communities ``1..r`` by size descending, ties by smallest member.

    ``flat`` holds int32 community ids in ``[0, n_comm)``. Returns int64
    labels. This runs once per call and is linear, so it stays sequential;
    ``n_workers`` is accepted for a uniform kernel interface.
    """
    n = flat.shape[0]
    sizes = np.zeros(n_comm, dtype=np.int64)
    by_first_member = np.empty(n_comm, dtype=np.int32)
    n_found = 0
    for node in range(n):
        community = flat[node]
        if sizes[community] == 0:
            by_first_member[n_found] = community
            n_found += 1
        sizes[community] += 1
    largest = np.int64(0)
    for index in range(n_found):
        size = sizes[by_first_member[index]]
        if size > largest:
            largest = size
    cursor = np.zeros(largest + 2, dtype=np.int64)
    for index in range(n_found):
        cursor[largest - sizes[by_first_member[index]] + 1] += 1
    for slot in range(largest + 1):
        cursor[slot + 1] += cursor[slot]
    rank = np.zeros(n_comm, dtype=np.int64)
    for index in range(n_found):
        community = by_first_member[index]
        slot = largest - sizes[community]
        rank[community] = cursor[slot]
        cursor[slot] += 1
    labels = np.empty(n, dtype=np.int64)
    for node in range(n):
        labels[node] = rank[flat[node]] + 1
    return labels
