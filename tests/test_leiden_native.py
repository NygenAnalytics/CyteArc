"""Tests of CyteArc's native Leiden backend.

None of these tests needs igraph or leidenalg. Reference quantities (modularity,
connectivity, aggregation and move gains) are computed here by brute force with
NumPy and SciPy, independently of the implementation.
"""

import contextlib
import functools
import math
import shutil
import threading
import time
from collections.abc import Iterator

import numba
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import cytearc.clustering._leiden_aggregate as aggregate_module
import cytearc.clustering._leiden_moving as moving_module
import cytearc.clustering._leiden_native as native_module
import cytearc.clustering._leiden_primitives as primitives_module
import cytearc.clustering._leiden_refine as refine_module
from cytearc.clustering._leiden_aggregate import (
    aggregate_counts,
    aggregate_fill,
    aggregate_node_info,
    aggregation_order,
    compact_refined,
    final_labels,
    renumber_partition,
    split_into_components,
)
from cytearc.clustering._leiden_graph import build_leiden_graph
from cytearc.clustering._leiden_moving import (
    NEW_COMMUNITY,
    STAY,
    _bucket_active_by_group,
    best_move,
)
from cytearc.clustering._leiden_native import THETA, native_leiden_membership
from cytearc.clustering._leiden_primitives import (
    BUCKET_DIRECT_KEYS,
    EMPTY_SLOT,
    FLOAT_SUM_BLOCK,
    GOLDEN,
    INSERTION_SORT_LIMIT,
    PHASE_GROUPS,
    PHASE_MOVE_ORDER,
    PHASE_REFINE,
    compact_ids,
    derive_key,
    exclusive_scan,
    fixed_block_sum,
    hash_capacity,
    scatter_add,
    stable_bucket_order,
    stream_key,
    uniform01,
)
from cytearc.clustering._leiden_refine import merge_nodes_subset
from cytearc.clustering.leiden import leiden_membership
from cytearc.neighbors.graph import build_connectivity_arrays
from cytearc.utils.logging import logger
from cytearc.utils.shutdown import ShutdownRequested, ShutdownToken, shutdown_scope

# Forces the random-group scheme on every level and the multi-worker loop of
# every kernel, even on small graphs.
FORCED = {"_sequential_level_entries": 0, "_parallel_min_items": 1}
PATHS = [pytest.param({}, id="default"), pytest.param(FORCED, id="forced-parallel")]


# ---------------------------------------------------------------------------
# Reference helpers (independent of the implementation)
# ---------------------------------------------------------------------------


def _symmetric(W) -> sp.csr_matrix:
    """``A = W + W^T`` in float64 with duplicates summed."""
    matrix = sp.csr_matrix(sp.coo_matrix(W).astype(np.float64))
    matrix.sum_duplicates()
    return sp.csr_matrix(matrix + matrix.T)


def _dense_modularity(A: np.ndarray, labels: np.ndarray, gamma: float) -> float:
    """Modularity of a dense symmetric adjacency, diagonal included."""
    labels = np.asarray(labels)
    strengths = A.sum(axis=1)
    two_m = strengths.sum()
    same = labels[:, None] == labels[None, :]
    expected = gamma * np.outer(strengths, strengths) / two_m
    return float(((A - expected) * same).sum() / two_m)


def reference_modularity(W, labels: np.ndarray, gamma: float) -> float:
    """Modularity of ``labels`` for ``A = W + W^T`` (the backend's objective)."""
    A = _symmetric(W)
    labels = np.asarray(labels)
    strengths = np.asarray(A.sum(axis=1)).ravel()
    two_m = strengths.sum()
    coo = A.tocoo()
    inner = coo.data[labels[coo.row] == labels[coo.col]].sum()
    _, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=strengths)
    return float((inner - gamma * np.dot(totals, totals) / two_m) / two_m)


def communities_connected(W, labels: np.ndarray) -> bool:
    """Whether every community induces a connected subgraph of ``W + W^T``."""
    labels = np.asarray(labels)
    coo = _symmetric(W).tocoo()
    keep = (labels[coo.row] == labels[coo.col]) & (coo.data != 0)
    n = labels.shape[0]
    inside = sp.coo_matrix(
        (np.ones(int(keep.sum())), (coo.row[keep], coo.col[keep])), shape=(n, n)
    )
    n_components, _ = connected_components(inside, directed=False)
    return n_components == np.unique(labels).size


def valid_labelling(labels: np.ndarray, n: int) -> bool:
    """Whether ``labels`` is an int64 labelling ``1..r`` of ``n`` nodes."""
    return (
        labels.dtype == np.int64
        and labels.shape == (n,)
        and np.array_equal(np.unique(labels), np.arange(1, labels.max() + 1))
    )


def ring_of_cliques(r: int, s: int) -> sp.csr_matrix:
    """``r`` cliques of ``s`` nodes, consecutive cliques joined by one edge."""
    rows, cols = [], []
    for clique in range(r):
        base = clique * s
        for i in range(s):
            for j in range(i + 1, s):
                rows.append(base + i)
                cols.append(base + j)
        rows.append(base + s - 1)
        cols.append(((clique + 1) % r) * s)
    n = r * s
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def planted(
    blocks: int, size: int, p_in: float, p_out: float, seed: int
) -> tuple[sp.csr_matrix, np.ndarray]:
    """A planted partition graph, each edge stored once (upper triangle)."""
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(blocks), size)
    probability = np.where(truth[:, None] == truth[None, :], p_in, p_out)
    n = blocks * size
    adjacency = np.triu(rng.random((n, n)) < probability, 1)
    return sp.csr_matrix(adjacency.astype(np.float64)), truth


def cliques_graph(sizes: list[int], n_isolated: int) -> sp.csr_matrix:
    """Disjoint cliques on consecutive nodes, followed by isolated nodes."""
    rows, cols = [], []
    start = 0
    for size in sizes:
        for i in range(start, start + size):
            for j in range(i + 1, start + size):
                rows.append(i)
                cols.append(j)
        start += size
    n = start + n_isolated
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


@functools.lru_cache(maxsize=None)
def _knn_graph_cached(n: int, k: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(10, 8))
    points = centers[rng.integers(0, 10, size=n)] + rng.normal(size=(n, 8))
    distances, indices = (
        NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    )
    edges, weights = build_connectivity_arrays(
        indices[:, 1:].astype(np.int64),
        distances[:, 1:],
        local_connectivity=1.0,
        bandwidth=1.5,
    )
    graph = sp.csr_matrix(
        (weights, (edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64))),
        shape=(n, n),
        dtype=np.float32,
    )
    graph.sort_indices()
    return graph


def knn_graph(n: int, k: int, seed: int = 0) -> sp.csr_matrix:
    """A directed kNN graph with CyteArc's UMAP membership weights (float32)."""
    return _knn_graph_cached(n, k, seed).copy()


def _random_small_graph(rng: np.random.Generator, n: int) -> sp.coo_matrix:
    """A random weighted graph with self-loops, duplicates and explicit zeros."""
    density = rng.uniform(0.2, 0.7)
    rows, cols = np.nonzero(rng.random((n, n)) < density)
    # A path keeps at least one off-diagonal edge.
    rows = np.concatenate([rows, np.arange(n - 1), rows[:3], rng.integers(0, n, 3)])
    cols = np.concatenate([cols, np.arange(1, n), cols[:3], rng.integers(0, n, 3)])
    data = rng.uniform(0.05, 3.0, size=rows.shape[0])
    data[-3:] = 0.0
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n))


def _hash_scratch(max_distinct: int, n_workers: int = 1):
    capacity, shift = hash_capacity(max_distinct)
    keys = np.full((n_workers, capacity), EMPTY_SLOT, dtype=np.int32)
    vals = np.zeros((n_workers, capacity), dtype=np.float64)
    used = np.empty((n_workers, capacity), dtype=np.int32)
    return keys, vals, used, shift


def _dense_level_adjacency(graph) -> np.ndarray:
    n = graph.n_nodes
    return sp.csr_matrix(
        (graph.weights, graph.indices, graph.indptr), shape=(n, n)
    ).toarray()


# ---------------------------------------------------------------------------
# 7.1 Formulas and building blocks
# ---------------------------------------------------------------------------


def _csr_variants(seed: int) -> list[tuple[str, object]]:
    rng = np.random.default_rng(seed)
    n = 12
    rows = rng.integers(0, n, size=40)
    cols = rng.integers(0, n, size=40)
    data = rng.uniform(0.1, 2.0, size=40)
    rows = np.concatenate([rows, rows[:6], np.arange(4), rng.integers(0, n, 4)])
    cols = np.concatenate([cols, cols[:6], np.arange(4), rng.integers(0, n, 4)])
    data = np.concatenate([data, data[:6] * 0.5, rng.uniform(0.1, 2.0, 4), np.zeros(4)])
    coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    canonical = coo.tocsr()
    canonical.sum_duplicates()
    order = np.lexsort((-cols, rows))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])
    unsorted = sp.csr_matrix((data[order], cols[order], indptr), shape=(n, n))
    wide_index = canonical.copy()
    wide_index.indices = wide_index.indices.astype(np.int64)
    wide_index.indptr = wide_index.indptr.astype(np.int64)
    integer = sp.coo_matrix(
        (np.round(data * 4).astype(np.int64), (rows, cols)), shape=(n, n)
    )
    boolean = sp.coo_matrix((data > 0.7, (rows, cols)), shape=(n, n))
    return [
        ("csr", canonical),
        ("csr-unsorted-duplicates", unsorted),
        ("csc", canonical.tocsc()),
        ("coo-duplicates", coo),
        ("lil", canonical.tolil()),
        ("csr_array", sp.csr_array(canonical)),
        ("csr-int64-indices", wide_index),
        ("csr-float32", canonical.astype(np.float32)),
        ("coo-float32-duplicates", coo.astype(np.float32)),
        ("coo-int64-data", integer),
        ("coo-bool-data", boolean),
    ]


def _expected_symmetric(variant) -> sp.csr_matrix:
    """SciPy's canonical ``W + W^T`` of the stored values, in float64."""
    W = sp.coo_matrix(variant).astype(np.float64).tocsr()
    W.sum_duplicates()
    A = sp.csr_matrix(W + W.T)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    return A


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_symmetric_csr_matches_scipy(seed):
    for name, variant in _csr_variants(seed):
        graph = build_leiden_graph(variant, 1)
        expected = _expected_symmetric(variant)
        assert graph.indptr.dtype == np.int64, name
        assert graph.indices.dtype == np.int32, name
        np.testing.assert_array_equal(graph.indptr, expected.indptr, err_msg=name)
        np.testing.assert_array_equal(graph.indices, expected.indices, err_msg=name)
        restored = np.ldexp(graph.weights, graph.weight_exponent)
        assert np.array_equal(restored, expected.data), name
        assert graph.has_offdiagonal, name
        assert 1.0 <= graph.mean_weight < 2.0, name
        assert graph.total_strength == int(graph.strengths.sum()), name
        assert graph.max_degree == int(np.diff(expected.indptr).max()), name
        # The same arrays for several workers and forced parallel loops.
        parallel = build_leiden_graph(variant, 4, 1)
        np.testing.assert_array_equal(parallel.indices, graph.indices, err_msg=name)
        assert np.array_equal(parallel.weights, graph.weights), name
        np.testing.assert_array_equal(parallel.strengths, graph.strengths)


def _snapshot(matrix) -> dict:
    state = {
        "type": type(matrix),
        "format": matrix.format,
        "dtype": matrix.dtype,
        "shape": matrix.shape,
    }
    for name in ("data", "indices", "indptr", "row", "col"):
        value = getattr(matrix, name, None)
        if isinstance(value, np.ndarray) and value.dtype != object:
            state[name] = value.copy()
    if matrix.format == "lil":
        state["lil_rows"] = [list(row) for row in matrix.rows]
        state["lil_data"] = [list(row) for row in matrix.data]
    if matrix.format in {"csr", "csc"}:
        state["canonical"] = matrix.has_canonical_format
        state["sorted"] = matrix.has_sorted_indices
    return state


def _assert_same_state(before: dict, after: dict, name: str) -> None:
    assert before.keys() == after.keys(), name
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert after[key].dtype == value.dtype, (name, key)
            np.testing.assert_array_equal(after[key], value, err_msg=f"{name} {key}")
        else:
            assert after[key] == value, (name, key)


def test_graph_construction_does_not_modify_the_input():
    for name, variant in _csr_variants(3):
        before = _snapshot(variant)
        build_leiden_graph(variant, 2, 1)
        native_leiden_membership(variant, 1.0, 1, 2)
        _assert_same_state(before, _snapshot(variant), name)


def _brute_force_options(
    A: np.ndarray, labels: np.ndarray, v: int, gamma: float
) -> dict[int, float]:
    """Quality gain (``Delta Q * m``) of every possible move of node ``v``."""
    two_m = A.sum()
    base = _dense_modularity(A, labels, gamma)
    current = labels[v]
    options: dict[int, float] = {}
    for community in np.unique(labels):
        if community == current:
            continue
        moved = labels.copy()
        moved[v] = community
        options[int(community)] = (
            (_dense_modularity(A, moved, gamma) - base) * two_m / 2
        )
    if np.count_nonzero(labels == current) > 1:
        moved = labels.copy()
        moved[v] = labels.max() + 1
        options[NEW_COMMUNITY] = (_dense_modularity(A, moved, gamma) - base) * two_m / 2
    return options


@pytest.mark.parametrize("seed", range(30))
def test_best_move_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 13))
    gamma = float(rng.choice([0.05, 0.5, 1.0, 2.0, 7.0]))
    graph = build_leiden_graph(_random_small_graph(rng, n), 1)
    A = _dense_level_adjacency(graph)
    labels = rng.integers(0, n, size=n).astype(np.int32)
    totals = np.zeros(2 * n, dtype=np.int64)
    sizes = np.zeros(2 * n, dtype=np.int32)
    np.add.at(totals, labels, graph.strengths)
    np.add.at(sizes, labels, 1)
    null_coef = gamma / (graph.strength_scale * float(graph.total_strength))
    eps_coef = 1e-10 / graph.strength_scale
    keys, vals, used, shift = _hash_scratch(graph.max_degree)
    strengths = A.sum(axis=1)
    for v in range(n):
        target, gain = best_move(
            v,
            graph.indptr,
            graph.indices,
            graph.weights,
            labels,
            totals,
            sizes,
            graph.strengths,
            null_coef,
            eps_coef,
            keys[0],
            vals[0],
            used[0],
            shift,
        )
        options = _brute_force_options(A, labels, v, gamma)
        scale = max(1.0, strengths[v])
        if sizes[labels[v]] == 1:
            assert target != NEW_COMMUNITY
        if not options or max(options.values()) <= 1e-10 * strengths[v] + 1e-12 * scale:
            assert target == STAY
            assert gain == 0.0
            continue
        best = max(options.values())
        tied = [key for key, value in options.items() if value >= best - 1e-12 * scale]
        existing = sorted(key for key in tied if key >= 0)
        expected = existing[0] if existing else NEW_COMMUNITY
        assert target == expected
        assert gain == pytest.approx(best, rel=1e-9, abs=1e-12 * scale)
    assert np.all(keys == EMPTY_SLOT)


def _move_arrays(edges: list[tuple[int, int, float]], n: int) -> tuple:
    rows = [i for i, _, _ in edges] + [j for _, j, _ in edges]
    cols = [j for _, j, _ in edges] + [i for i, _, _ in edges]
    weights = [w for _, _, w in edges] * 2
    A = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    A.sort_indices()
    return (
        A.indptr.astype(np.int64),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
    )


def _call_best_move(v, edges, n, labels, strengths, null_coef=0.25):
    indptr, indices, weights = _move_arrays(edges, n)
    labels = np.asarray(labels, dtype=np.int32)
    strengths = np.asarray(strengths, dtype=np.int64)
    totals = np.zeros(2 * n, dtype=np.int64)
    sizes = np.zeros(2 * n, dtype=np.int32)
    np.add.at(totals, labels, strengths)
    np.add.at(sizes, labels, 1)
    keys, vals, used, shift = _hash_scratch(n)
    return best_move(
        v,
        indptr,
        indices,
        weights,
        labels,
        totals,
        sizes,
        strengths,
        null_coef,
        1e-12,
        keys[0],
        vals[0],
        used[0],
        shift,
    )


def test_best_move_tie_rules():
    # Exact dyadic arithmetic: c_v = 0.25 * q_v.
    # Staying wins an exact tie: f(stay) = 1 - 1 * 1 = 0 = f({2}) = 1 - 1 * 1.
    target, gain = _call_best_move(
        1, [(0, 1, 1.0), (1, 2, 1.0)], 3, [0, 0, 1], [1, 4, 1]
    )
    assert (target, gain) == (STAY, 0.0)
    # The smallest of two equally good communities wins: f = 2 - 1 * 1 = 1 for both.
    target, gain = _call_best_move(
        0, [(0, 1, 2.0), (0, 2, 2.0)], 3, [2, 1, 0], [4, 1, 1]
    )
    assert (target, gain) == (0, 1.0)
    target, gain = _call_best_move(
        0, [(0, 1, 2.0), (0, 2, 2.0)], 3, [5, 1, 3], [4, 1, 1]
    )
    assert (target, gain) == (1, 1.0)
    # An existing community wins a tie with an empty one: f({3}) = 2 - 1 * 2 = 0.
    target, gain = _call_best_move(0, [(0, 3, 2.0)], 4, [0, 0, 1, 2], [4, 1, 5, 2])
    assert target == 2
    assert gain == 1.0
    # Without that neighbour the node leaves for an empty community.
    target, gain = _call_best_move(0, [(1, 3, 2.0)], 4, [0, 0, 1, 2], [4, 1, 5, 2])
    assert (target, gain) == (NEW_COMMUNITY, 1.0)
    # A singleton never moves to an empty community.
    target, gain = _call_best_move(0, [(0, 1, 0.5)], 2, [0, 1], [4, 8])
    assert (target, gain) == (STAY, 0.0)


def _brute_force_merge_check(
    graph, labels, members, null_coef, refined, rsize, kref, eext
):
    A = _dense_level_adjacency(graph)
    q = graph.strengths
    member_set = set(members.tolist())
    total_s = int(q[members].sum())
    founders = sorted({int(refined[v]) for v in members})
    assert all(f in member_set and refined[f] == f for f in founders)
    for founder in founders:
        part = members[refined[members] == founder]
        assert rsize[founder] == part.size
        assert kref[founder] == int(q[part].sum())
        rest = np.array(sorted(member_set - set(part.tolist())), dtype=np.int64)
        external = A[np.ix_(part, rest)].sum() if rest.size else 0.0
        assert eext[founder] == pytest.approx(external, rel=1e-9, abs=1e-12)
        inside = sp.csr_matrix(A[np.ix_(part, part)])
        assert connected_components(inside, directed=False)[0] == 1
    for v in members:
        if refined[v] != v:
            assert rsize[v] == 0
        inside_weight = 0.0
        for u in members:
            if u != v:
                inside_weight += A[v, u]
        c_v = null_coef * float(q[v])
        if not inside_weight >= c_v * float(total_s - q[v]):
            assert refined[v] == v and rsize[v] == 1
    return len(founders)


@pytest.mark.parametrize("seed", range(25))
def test_merge_nodes_subset_matches_brute_force(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(3, 13))
    gamma = float(rng.choice([0.05, 0.5, 1.0, 2.0, 7.0]))
    graph = build_leiden_graph(_random_small_graph(rng, n), 1)
    in_subset = rng.random(n) < 0.7
    in_subset[:2] = True
    labels = np.where(in_subset, 0, 1).astype(np.int32)
    members = np.flatnonzero(in_subset).astype(np.int32)
    null_coef = gamma / (graph.strength_scale * float(graph.total_strength))
    theta_w = THETA * graph.mean_weight
    key = np.uint64(stream_key(np.uint64(seed), 0, 0, PHASE_REFINE))
    results = []
    for _repeat in range(2):
        refined = np.full(n, -5, dtype=np.int32)
        rsize = np.full(n, -5, dtype=np.int32)
        kref = np.full(n, -5, dtype=np.int64)
        eext = np.full(n, np.nan)
        rlist = np.full(n, -5, dtype=np.int32)
        keys, vals, used, shift = _hash_scratch(graph.max_degree)
        width = graph.max_degree + 1
        count = merge_nodes_subset(
            0,
            0,
            members.size,
            members,
            graph.indptr,
            graph.indices,
            graph.weights,
            graph.strengths,
            labels,
            null_coef,
            theta_w,
            key,
            refined,
            rsize,
            kref,
            eext,
            rlist,
            keys[0],
            vals[0],
            used[0],
            shift,
            np.empty(width, dtype=np.int32),
            np.empty(width, dtype=np.float64),
            np.empty(width, dtype=np.float64),
        )
        outside = ~in_subset
        assert np.all(refined[outside] == -5) and np.all(rsize[outside] == -5)
        assert count == _brute_force_merge_check(
            graph, labels, members, null_coef, refined, rsize, kref, eext
        )
        results.append((refined.copy(), rsize.copy(), kref.copy(), eext.copy()))
    for first, second in zip(results[0], results[1]):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("seed", range(12))
def test_aggregation_matches_membership_products(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(4, 40))
    graph = build_leiden_graph(_random_small_graph(rng, n), 1)
    A = _dense_level_adjacency(graph)
    blocks = rng.integers(0, max(2, n // 3), size=n)
    founders = np.array(
        [np.flatnonzero(blocks == b).min() for b in blocks], dtype=np.int32
    )
    rsize = np.zeros(n, dtype=np.int32)
    np.add.at(rsize, founders, 1)
    labels = blocks.astype(np.int32)
    outputs = []
    for n_workers in (1, 2, 4):
        ragg, n_agg = compact_refined(founders, rsize, n_workers, 1)
        agg_start, agg_members = stable_bucket_order(ragg, n_agg, n_workers, 1)
        new_q, new_labels, bounds = aggregate_node_info(
            graph.strengths, labels, graph.indptr, agg_start, agg_members, n_workers
        )
        order = aggregation_order(bounds, n_workers, 1)
        keys, vals, used, shift = _hash_scratch(
            min(int(bounds.max()), n_agg), n_workers
        )
        lengths = aggregate_counts(
            graph.indptr,
            graph.indices,
            ragg,
            agg_start,
            agg_members,
            order,
            n_workers,
            keys,
            used,
            shift,
            1,
        )
        indptr = exclusive_scan(lengths, n_workers, 1)
        indices = np.empty(indptr[-1], dtype=np.int32)
        weights = np.empty(indptr[-1], dtype=np.float64)
        aggregate_fill(
            graph.indptr,
            graph.indices,
            graph.weights,
            ragg,
            agg_start,
            agg_members,
            order,
            indptr,
            indices,
            weights,
            n_workers,
            keys,
            vals,
            used,
            shift,
            1,
        )
        outputs.append((ragg, new_q, new_labels, indptr, indices, weights))
    for other in outputs[1:]:
        for first, second in zip(outputs[0], other):
            assert np.array_equal(first, second)
    ragg, new_q, new_labels, indptr, indices, weights = outputs[0]
    n_agg = indptr.shape[0] - 1
    for row in range(n_agg):
        assert np.all(np.diff(indices[indptr[row] : indptr[row + 1]]) > 0)
    membership = np.zeros((n, n_agg))
    membership[np.arange(n), ragg] = 1.0
    aggregated = sp.csr_matrix(
        (weights, indices, indptr), shape=(n_agg, n_agg)
    ).toarray()
    np.testing.assert_allclose(
        aggregated, membership.T @ A @ membership, rtol=1e-12, atol=0
    )
    expected_q = np.zeros(n_agg, dtype=np.int64)
    np.add.at(expected_q, ragg, graph.strengths)
    np.testing.assert_array_equal(new_q, expected_q)
    np.testing.assert_array_equal(
        new_labels,
        labels[np.argsort(ragg, kind="stable")][
            np.searchsorted(np.sort(ragg), np.arange(n_agg))
        ],
    )
    for _ in range(5):
        partition = rng.integers(0, max(1, n_agg // 2), size=n_agg)
        assert _dense_modularity(A, partition[ragg], 1.3) == pytest.approx(
            _dense_modularity(aggregated, partition, 1.3), abs=1e-12
        )


def test_split_into_components_and_final_numbering():
    # Community 0 holds two separate paths, community 1 one path.
    W = sp.csr_matrix(
        ([1.0, 1.0, 1.0, 1.0], ([0, 1, 3, 5], [1, 2, 4, 6])), shape=(7, 7)
    )
    graph = build_leiden_graph(W, 1)
    labels = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int32)
    components = split_into_components(graph.indptr, graph.indices, labels)
    np.testing.assert_array_equal(components, [0, 0, 0, 3, 3, 5, 5])
    flat = np.array([4, 4, 1, 1, 1, 0, 2, 2], dtype=np.int32)
    np.testing.assert_array_equal(final_labels(flat, 5, 2), [2, 2, 1, 1, 1, 4, 3, 3])


def test_counter_based_random_numbers():
    seed = np.uint64(7)
    key = stream_key(seed, 0, 0, PHASE_REFINE)
    assert stream_key(seed, 0, 0, PHASE_REFINE) == key
    assert stream_key(np.uint64(8), 0, 0, PHASE_REFINE) != key
    keys = {
        stream_key(seed, iteration, level, phase)
        for iteration in range(3)
        for level in range(3)
        for phase in (PHASE_MOVE_ORDER, PHASE_GROUPS, PHASE_REFINE)
    }
    assert len(keys) == 27
    assert len({derive_key(np.uint64(key), entity) for entity in range(200)}) == 200
    draws = np.array([uniform01(np.uint64(key), counter) for counter in range(20_000)])
    assert np.array_equal(
        draws, [uniform01(np.uint64(key), counter) for counter in range(20_000)]
    )
    assert draws.min() >= 0.0 and draws.max() < 1.0
    assert abs(draws.mean() - 0.5) < 0.01
    assert np.unique(draws).size == draws.size
    sweep_key = np.uint64(
        derive_key(np.uint64(stream_key(seed, 0, 0, PHASE_GROUPS)), 0)
    )
    n = 65_536
    counts = {}
    for n_workers, min_items in ((1, 1024), (4, 1)):
        order, start = _bucket_active_by_group(
            np.ones(n, dtype=np.uint8),
            sweep_key,
            np.uint64(58),
            64,
            n_workers,
            min_items,
        )
        counts[n_workers] = np.diff(start)
        assert np.array_equal(np.sort(order), np.arange(n))
        for group in range(64):
            assert np.all(np.diff(order[start[group] : start[group + 1]]) > 0)
    np.testing.assert_array_equal(counts[1], counts[4])
    sigma = math.sqrt(n * (1 / 64) * (63 / 64))
    assert np.all(np.abs(counts[1] - n / 64) <= 5 * sigma)


@pytest.mark.parametrize("min_items", [1, 1024])
def test_parallel_primitives_match_numpy(min_items):
    rng = np.random.default_rng(11)
    counts = rng.integers(0, 50, size=300_000).astype(np.int64)
    expected_scan = np.concatenate([[0], np.cumsum(counts)])
    values = rng.standard_normal(300_000)
    block_sums = [
        np.cumsum(values[start : start + FLOAT_SUM_BLOCK])[-1]
        for start in range(0, values.size, FLOAT_SUM_BLOCK)
    ]
    expected_sum = np.cumsum(block_sums)[-1]
    sizes = rng.integers(0, 3, size=100_000).astype(np.int32)
    used = sizes > 0
    for n_workers in (1, 2, 4):
        np.testing.assert_array_equal(
            exclusive_scan(counts, n_workers, min_items), expected_scan
        )
        assert fixed_block_sum(values, n_workers, min_items) == expected_sum
        for n_keys in (50, 4096, 50_000):
            keys = rng.integers(0, n_keys, size=100_000).astype(np.int32)
            start, order = stable_bucket_order(keys, n_keys, n_workers, min_items)
            np.testing.assert_array_equal(order, np.argsort(keys, kind="stable"))
            np.testing.assert_array_equal(
                start,
                np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=n_keys))]),
            )
        scatter_keys = rng.integers(0, 3000, size=50_000).astype(np.int32)
        dq = rng.integers(-1000, 1000, size=50_000).astype(np.int64)
        ds = rng.integers(-3, 4, size=50_000).astype(np.int32)
        totals = np.zeros(3000, dtype=np.int64)
        group_sizes = np.zeros(3000, dtype=np.int32)
        scatter_add(scatter_keys, dq, ds, totals, group_sizes, n_workers, min_items)
        expected_totals = np.zeros(3000, dtype=np.int64)
        expected_sizes = np.zeros(3000, dtype=np.int32)
        np.add.at(expected_totals, scatter_keys, dq)
        np.add.at(expected_sizes, scatter_keys, ds)
        np.testing.assert_array_equal(totals, expected_totals)
        np.testing.assert_array_equal(group_sizes, expected_sizes)
        new_ids, n_used = compact_ids(sizes, n_workers, min_items)
        assert n_used == int(used.sum())
        np.testing.assert_array_equal(new_ids[used], np.arange(n_used))


# ---------------------------------------------------------------------------
# 7.2 Behaviour on small graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ring_of_cliques_at_high_resolution(path, seed):
    W = ring_of_cliques(30, 5)
    cliques = np.repeat(np.arange(30), 5)
    labels = native_leiden_membership(W, 2.0, seed, None, **path)
    assert adjusted_rand_score(cliques, labels) == 1.0
    assert round(reference_modularity(W, labels, 2.0), 6) == 0.842424


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ring_of_cliques_at_lower_resolutions(path, seed):
    W = ring_of_cliques(30, 5)
    cliques = np.repeat(np.arange(30), 5)
    counts = {}
    for gamma, clique_quality in ((1.0, 0.875758), (0.5, 0.892424)):
        labels = native_leiden_membership(W, gamma, seed, None, **path)
        for clique in range(30):
            assert np.unique(labels[cliques == clique]).size == 1
        assert reference_modularity(W, labels, gamma) >= clique_quality - 1e-6
        counts[gamma] = int(labels.max())
    assert counts[0.5] <= counts[1.0] <= 30


@pytest.mark.parametrize("path", PATHS)
def test_planted_partition_is_recovered(path):
    W, truth = planted(4, 50, 0.3, 0.01, 5)
    for seed in range(1, 6):
        labels = native_leiden_membership(W, 1.0, seed, None, **path)
        assert adjusted_rand_score(truth, labels) == 1.0


@pytest.mark.parametrize("path", PATHS)
def test_edge_weights_decide_a_complete_graph(path):
    weights = np.full((8, 8), 0.001)
    weights[:4, :4] = 1.0
    weights[4:, 4:] = 1.0
    np.fill_diagonal(weights, 0)
    labels = native_leiden_membership(sp.csr_matrix(weights), 1.0, 11, None, **path)
    np.testing.assert_array_equal(labels, [1, 1, 1, 1, 2, 2, 2, 2])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seed", [1, 2])
def test_disconnected_cliques_and_isolated_nodes(path, seed):
    W = cliques_graph([5, 4, 3], 4)
    labels = native_leiden_membership(W, 1.0, seed, None, **path)
    np.testing.assert_array_equal(
        labels, [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7]
    )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seed", [1, 2])
def test_labels_are_numbered_by_size_then_first_node(path, seed):
    W = cliques_graph([3, 5, 3, 4], 1)
    labels = native_leiden_membership(W, 1.0, seed, None, **path)
    np.testing.assert_array_equal(
        labels, [3, 3, 3, 1, 1, 1, 1, 1, 4, 4, 4, 2, 2, 2, 2, 5]
    )


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_communities_are_connected(gamma):
    W = knn_graph(2000, 10)
    for seed in (1, 2, 3):
        labels = native_leiden_membership(W, gamma, seed, None)
        assert valid_labelling(labels, 2000)
        assert communities_connected(W, labels)
    forced = native_leiden_membership(W, gamma, 1, None, **FORCED)
    assert communities_connected(W, forced)


def test_labels_do_not_depend_on_the_thread_count():
    W = knn_graph(4000, 15, seed=1)
    runs = [
        native_leiden_membership(W, 1.0, 4444, threads, **FORCED)
        for threads in (1, 2, 3, 4)
        for _repeat in range(2)
    ]
    for labels in runs[1:]:
        np.testing.assert_array_equal(labels, runs[0])
    default = [
        native_leiden_membership(W, 1.0, 4444, threads) for threads in (1, 2, 3, 4)
    ]
    for labels in default[1:]:
        np.testing.assert_array_equal(labels, default[0])
    public = leiden_membership(W, 1.0, 4444, nthreads=3)
    np.testing.assert_array_equal(public, default[0])


def test_forced_parallel_path_matches_sequential_quality():
    W = knn_graph(4000, 15, seed=1)
    forced = []
    default = []
    for seed in (1, 2, 3):
        forced.append(
            reference_modularity(
                W, native_leiden_membership(W, 1.0, seed, 4, **FORCED), 1.0
            )
        )
        default.append(
            reference_modularity(W, native_leiden_membership(W, 1.0, seed, 4), 1.0)
        )
    assert np.mean(forced) >= np.mean(default) - 0.005


def test_seeds_are_repeatable_and_valid():
    W = knn_graph(2000, 10)
    for seed in (5, 2**40):
        first = leiden_membership(W, 1.0, seed)
        np.testing.assert_array_equal(leiden_membership(W, 1.0, seed), first)
        assert valid_labelling(first, 2000)
        assert communities_connected(W, first)
    # Seeds are reduced modulo 2**64.
    np.testing.assert_array_equal(
        leiden_membership(W, 1.0, 2**64 + 5), leiden_membership(W, 1.0, 5)
    )


def _grouped_graph(groups: list[list[int]]) -> sp.csr_matrix:
    n = sum(len(group) for group in groups)
    rows, cols = [], []
    for group in groups:
        for cell in group:
            for neighbour in group:
                if neighbour != cell:
                    rows.append(cell)
                    cols.append(neighbour)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


@pytest.mark.parametrize("path", PATHS)
def test_explicit_zeros_change_nothing(path):
    solid = _grouped_graph([[0, 1, 2, 3], [4, 5, 6, 7]]).tocoo()
    padded = sp.coo_matrix(
        (
            np.concatenate([solid.data, np.zeros(4)]),
            (
                np.concatenate([solid.row, [0, 1, 2, 3]]),
                np.concatenate([solid.col, [4, 5, 6, 7]]),
            ),
        ),
        shape=solid.shape,
    )
    for seed in (1, 4444):
        np.testing.assert_array_equal(
            native_leiden_membership(padded, 1.0, seed, None, **path),
            native_leiden_membership(solid, 1.0, seed, None, **path),
        )
    W = knn_graph(1500, 10)
    coo = W.tocoo()
    rng = np.random.default_rng(0)
    extra_rows = rng.integers(0, 1500, 300)
    extra_cols = rng.integers(0, 1500, 300)
    zero_padded = sp.coo_matrix(
        (
            np.concatenate([coo.data, np.zeros(300, dtype=np.float32)]),
            (
                np.concatenate([coo.row, extra_rows]),
                np.concatenate([coo.col, extra_cols]),
            ),
        ),
        shape=W.shape,
    )
    np.testing.assert_array_equal(
        native_leiden_membership(zero_padded, 1.0, 3, None, **path),
        native_leiden_membership(W, 1.0, 3, None, **path),
    )


def test_duplicate_entries_are_summed():
    W = knn_graph(1500, 10).astype(np.float64)
    coo = W.tocoo()
    halves = sp.coo_matrix(
        (
            np.concatenate([coo.data / 2, coo.data / 2]),
            (np.concatenate([coo.row, coo.row]), np.concatenate([coo.col, coo.col])),
        ),
        shape=W.shape,
    )
    np.testing.assert_array_equal(
        leiden_membership(halves, 1.0, 9), leiden_membership(W, 1.0, 9)
    )


def test_labels_are_invariant_to_representation():
    W = knn_graph(1500, 10)
    expected = leiden_membership(W, 1.0, 21)
    wide = W.astype(np.float64)
    wide.indices = wide.indices.astype(np.int64)
    wide.indptr = wide.indptr.astype(np.int64)
    variants = [
        W.astype(np.float64),
        wide,
        W.tocsc(),
        W.tocoo(),
        sp.csr_array(W),
        W * 2,
        W * 0.5,
    ]
    for variant in variants:
        np.testing.assert_array_equal(leiden_membership(variant, 1.0, 21), expected)


@pytest.mark.parametrize("path", PATHS)
def test_edge_cases(path):
    empty = native_leiden_membership(sp.csr_matrix((0, 0)), 1.0, 1, None, **path)
    assert empty.dtype == np.int64 and empty.shape == (0,)
    single = sp.csr_matrix((1, 1))
    np.testing.assert_array_equal(native_leiden_membership(single, 1.0, 1, None), [1])
    looped = sp.csr_matrix(np.array([[2.5]]))
    np.testing.assert_array_equal(native_leiden_membership(looped, 1.0, 1, None), [1])
    edgeless = native_leiden_membership(sp.csr_matrix((5, 5)), 1.0, 1, None, **path)
    np.testing.assert_array_equal(edgeless, [1, 2, 3, 4, 5])
    assert edgeless.dtype == np.int64
    loops = sp.diags([1.0, 2.0, 0.5, 3.0]).tocsr()
    np.testing.assert_array_equal(
        native_leiden_membership(loops, 1.0, 1, None), [1, 2, 3, 4]
    )
    zeros = sp.coo_matrix((np.zeros(3), ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
    np.testing.assert_array_equal(
        native_leiden_membership(zeros, 1.0, 1, None), [1, 2, 3]
    )


def _graph_with(values) -> sp.csr_matrix:
    return sp.csr_matrix((np.asarray(values), ([0, 1], [1, 0])), shape=(2, 2))


_RING = ring_of_cliques(4, 3)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"graph": _graph_with([1.0, -1.0])}, ValueError, "must be non-negative"),
        ({"graph": _graph_with([1.0, np.nan])}, ValueError, "must be finite"),
        ({"graph": _graph_with([1.0, np.inf])}, ValueError, "must be finite"),
        ({"graph": _graph_with([-np.inf, -1.0])}, ValueError, "must be finite"),
        ({"graph": _graph_with([1.0 + 1j, 1.0])}, TypeError, "must be real numbers"),
        ({"graph": sp.csr_matrix((2, 3))}, ValueError, "must be a square matrix"),
        ({"graph": np.eye(3)}, TypeError, "SciPy sparse matrix or array"),
        ({"graph": sp.coo_matrix((2**30, 2**30))}, ValueError, "at most 1073741823"),
        (
            {"graph": _graph_with([1e308, 1e308])},
            ValueError,
            "symmetrized total is not finite",
        ),
        ({"resolution": 0}, ValueError, "finite and positive"),
        ({"resolution": -1}, ValueError, "finite and positive"),
        ({"resolution": math.inf}, ValueError, "finite and positive"),
        ({"resolution": math.nan}, ValueError, "finite and positive"),
        ({"resolution": True}, TypeError, "resolution must be a number"),
        ({"resolution": "1"}, TypeError, "resolution must be a number"),
        ({"random_seed": None}, TypeError, "unseeded run is not supported"),
        ({"random_seed": -1}, ValueError, "random_seed must be at least 0"),
        ({"random_seed": 1.5}, TypeError, "random_seed must be an integer"),
        ({"nthreads": 0}, ValueError, "nthreads must be at least 1"),
        ({"nthreads": -1}, ValueError, "nthreads must be at least 1"),
        ({"nthreads": 1.5}, TypeError, "nthreads must be an integer"),
        ({"nthreads": True}, TypeError, "nthreads must be an integer"),
        (
            {"backend": "unknown"},
            ValueError,
            "backend must be 'cytearc', 'igraph' or 'leidenalg'",
        ),
    ],
)
def test_invalid_arguments_are_rejected(arguments, error, message):
    call = {"graph": _RING, "resolution": 1.0, "random_seed": 1, **arguments}
    graph = call.pop("graph")
    with pytest.raises(error, match=message):
        leiden_membership(graph, **call)


def test_validation_order():
    bad_graph = np.eye(3)
    with pytest.raises(ValueError, match="backend"):
        leiden_membership(bad_graph, -1.0, None, backend="unknown", nthreads=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resolution"):
        leiden_membership(bad_graph, -1.0, None, nthreads=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="random_seed"):
        leiden_membership(bad_graph, 1.0, None, nthreads=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nthreads"):
        leiden_membership(bad_graph, 1.0, 1, nthreads=0)
    # A negative stored value is an error even when a duplicate cancels it.
    cancelling = sp.coo_matrix(([-1.0, 2.0, 1.0], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        leiden_membership(cancelling, 1.0, 1)
    # -0.0 is a zero, not a negative weight.
    signed_zero = sp.coo_matrix(([-0.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))
    np.testing.assert_array_equal(leiden_membership(signed_zero, 1.0, 1), [2, 1, 1])


def test_thread_count_is_restored():
    previous = numba.get_num_threads()
    target = min(2, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(target)
    try:
        leiden_membership(_RING, 1.0, 1, nthreads=1)
        assert numba.get_num_threads() == target
        with pytest.raises(ValueError, match="non-negative"):
            leiden_membership(_graph_with([1.0, -1.0]), 1.0, 1, nthreads=1)
        assert numba.get_num_threads() == target
        leiden_membership(_RING, 1.0, 1, nthreads=64)
        assert numba.get_num_threads() == target
    finally:
        numba.set_num_threads(previous)


def test_shutdown_request_stops_the_native_backend():
    token = ShutdownToken()
    token.request(reason="test")
    before = numba.get_num_threads()
    with shutdown_scope(token):
        with pytest.raises(ShutdownRequested):
            leiden_membership(_RING, 1.0, 1, nthreads=1)
    assert numba.get_num_threads() == before


def test_gpl_backends_ignore_nthreads(monkeypatch):
    import cytearc.clustering.leiden as leiden_module

    calls = []

    def fake_igraph_membership(graph, resolution, random_seed):
        # CyteArc does not install igraph; only the dispatch is tested.
        calls.append((resolution, random_seed))
        return np.ones(graph.shape[0], dtype=np.int64)

    monkeypatch.setattr(leiden_module, "_igraph_membership", fake_igraph_membership)
    labels = leiden_membership(_RING, 1.0, 1, backend="igraph", nthreads=2)
    assert labels.shape == (12,)
    assert calls == [(1.0, 1)]


# ---------------------------------------------------------------------------
# Paper guarantees, refinement retries, hang guard and robustness
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _debug_messages() -> Iterator[list[str]]:
    """Collect CyteArc's debug log messages while the block runs."""
    messages: list[str] = []
    handler = logger.add(
        lambda message: messages.append(str(message)), level="DEBUG", format="{message}"
    )
    try:
        yield messages
    finally:
        logger.remove(handler)


# A 4-cycle on which, for seed 0, refinement merges nothing on the second level
# of the second iteration: the two aggregate nodes there have a merge gain of
# exactly 0, so each visit stays alone with probability 1/2.
_STALLING_CYCLE = sp.csr_matrix(
    ([2.0, 1.0, 3.0, 2.0], ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4)
)


def test_refinement_is_retried_when_it_merges_nothing():
    with _debug_messages() as messages:
        runs = [
            native_leiden_membership(_STALLING_CYCLE, 1.0, 0, threads)
            for threads in (1, 2, 3, 4)
        ]
    for labels in runs[1:]:
        np.testing.assert_array_equal(labels, runs[0])
    assert valid_labelling(runs[0], 4)
    assert communities_connected(_STALLING_CYCLE, runs[0])
    assert any("refinement retry=1 of 8" in message for message in messages)
    assert not any("end=safety-stop" in message for message in messages)


def test_safety_stop_splits_when_refinement_never_merges(monkeypatch):
    monkeypatch.setattr(native_module, "REFINEMENT_RETRIES", 0)
    with _debug_messages() as messages:
        runs = [
            native_leiden_membership(_STALLING_CYCLE, 1.0, 0, threads)
            for threads in (1, 2, 3, 4)
        ]
    for labels in runs[1:]:
        np.testing.assert_array_equal(labels, runs[0])
    assert valid_labelling(runs[0], 4)
    assert communities_connected(_STALLING_CYCLE, runs[0])
    assert any("end=safety-stop" in message for message in messages)


def _merge_gains(W, labels: np.ndarray, gamma: float) -> tuple:
    """Quality gain ``E(C, D) - gamma K_C K_D / 2m`` of merging each pair."""
    A = _symmetric(W)
    strengths = np.asarray(A.sum(axis=1)).ravel()
    two_m = strengths.sum()
    _, community = np.unique(labels, return_inverse=True)
    membership = sp.csr_matrix(
        (np.ones(labels.size), (np.arange(labels.size), community)),
        shape=(labels.size, community.max() + 1),
    )
    between = (membership.T @ A @ membership).toarray()
    totals = membership.T @ strengths
    gains = between - gamma * np.outer(totals / two_m, totals)
    np.fill_diagonal(gains, -np.inf)
    return gains, totals, two_m


def _assert_gamma_separated(W, labels: np.ndarray, gamma: float) -> None:
    gains, totals, two_m = _merge_gains(W, labels, gamma)
    # A merge gain may only exceed 0 by the move tolerance (1e-10 times the
    # community's strength) and rounding.
    tolerance = 1e-9 * np.maximum.outer(totals, totals) + 1e-12 * two_m
    assert np.all(gains <= tolerance)


def _weighted_random_graph(n: int, p: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(np.triu(rng.random((n, n)) < p, 1))
    return sp.csr_matrix((rng.uniform(0.1, 3.0, rows.size), (rows, cols)), shape=(n, n))


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_communities_are_gamma_separated(gamma):
    graphs = [
        ring_of_cliques(30, 5),
        planted(4, 50, 0.3, 0.01, 5)[0],
        planted(8, 60, 0.12, 0.02, 3)[0],
        _weighted_random_graph(300, 0.03, 4),
        knn_graph(1500, 10),
    ]
    for W in graphs:
        for seed in (1, 2, 3):
            labels = native_leiden_membership(W, gamma, seed, None)
            _assert_gamma_separated(W, labels, gamma)


@pytest.mark.parametrize("log_odds", [0.0, math.log(3.0), math.log(9.0)])
def test_refinement_draw_matches_the_closed_form(log_odds):
    # Two nodes joined by one edge of weight 1: k = (1, 1), 2m = 2 and the mean
    # weight is 1, so the merge gain is g = 1 - gamma / 2 and theta is 0.01. A
    # visit of a singleton merges with probability 1 / (1 + exp(-g / theta)),
    # because staying alone has gain 0 (A.2 line 38). The first visited node
    # merges, or stays and the second one merges: P(merge) = 1 - stay**2.
    gamma = 2.0 * (1.0 - log_odds * THETA)
    graph = build_leiden_graph(sp.csr_matrix(([1.0], ([0], [1])), shape=(2, 2)), 1)
    null_coef = gamma / (graph.strength_scale * float(graph.total_strength))
    theta_w = THETA * graph.mean_weight
    gain = 1.0 - (null_coef * float(graph.strengths[0])) * float(graph.strengths[1])
    stay = 1.0 / (1.0 + math.exp(gain / theta_w))
    expected = 1.0 - stay**2
    labels = np.zeros(2, dtype=np.int32)
    members = np.arange(2, dtype=np.int32)
    keys, vals, used, shift = _hash_scratch(graph.max_degree)
    state = (
        np.empty(2, dtype=np.int32),
        np.empty(2, dtype=np.int32),
        np.empty(2, dtype=np.int64),
        np.empty(2, dtype=np.float64),
        np.empty(2, dtype=np.int32),
    )
    buffers = (
        np.empty(2, dtype=np.int32),
        np.empty(2, dtype=np.float64),
        np.empty(2, dtype=np.float64),
    )
    base = np.uint64(stream_key(np.uint64(2024), 0, 0, PHASE_REFINE))
    draws = 20_000
    merged = 0
    for index in range(draws):
        count = merge_nodes_subset(
            0,
            0,
            2,
            members,
            graph.indptr,
            graph.indices,
            graph.weights,
            graph.strengths,
            labels,
            null_coef,
            theta_w,
            np.uint64(derive_key(base, index)),
            *state,
            keys[0],
            vals[0],
            used[0],
            shift,
            *buffers,
        )
        merged += count == 1
    # The draws are fixed by the keys, so this cannot flake; six binomial
    # standard deviations leave room for another platform's exp.
    tolerance = 6.0 * math.sqrt(expected * (1.0 - expected) / draws)
    assert abs(merged / draws - expected) <= tolerance


def _replay_refinement(graph, labels, members, community, null_coef, theta_w, key):
    """Replay MergeNodesSubset (A.2 lines 33-43, SPEC 3.3) and record each join."""
    strengths = [int(value) for value in graph.strengths]
    indptr, indices, weights = graph.indptr, graph.indices, graph.weights
    total = sum(strengths[v] for v in members)
    refined = {int(v): int(v) for v in members}
    rsize = dict.fromkeys(refined, 1)
    kref = {v: strengths[v] for v in refined}
    eext = {}
    order = []
    for v in refined:
        inside = 0.0
        for edge in range(indptr[v], indptr[v + 1]):
            u = int(indices[edge])
            if u != v and labels[u] == community:
                inside += weights[edge]
        eext[v] = inside
        if inside >= (null_coef * float(strengths[v])) * float(total - strengths[v]):
            order.append(v)
    well_connected = set(order)
    counter = 0
    for position in range(len(order) - 1, 0, -1):
        other = min(int(uniform01(key, counter) * (position + 1)), position)
        counter += 1
        order[position], order[other] = order[other], order[position]
    joins = []
    for v in order:
        if refined[v] != v or rsize[v] != 1:
            continue
        weight_to: dict[int, float] = {}
        for edge in range(indptr[v], indptr[v + 1]):
            u = int(indices[edge])
            if u != v and labels[u] == community:
                target = refined[u]
                weight_to[target] = weight_to.get(target, 0.0) + weights[edge]
        c_v = null_coef * float(strengths[v])
        candidates = []
        for target in sorted(weight_to):
            threshold = (null_coef * float(kref[target])) * float(total - kref[target])
            gain = weight_to[target] - c_v * float(kref[target])
            if eext[target] >= threshold and gain >= 0.0:
                candidates.append(
                    (target, gain, weight_to[target], eext[target] - threshold)
                )
        if not candidates:
            continue
        best = max(0.0, max(candidate[1] for candidate in candidates))
        stay = math.exp((0.0 - best) / theta_w)
        total_weight = stay
        chances = []
        for candidate in candidates:
            chance = math.exp((candidate[1] - best) / theta_w)
            chances.append(chance)
            total_weight += chance
        draw = uniform01(key, counter) * total_weight
        counter += 1
        cumulative = stay
        chosen = None
        if draw >= cumulative:
            for candidate, chance in zip(candidates, chances):
                cumulative += chance
                chosen = candidate
                if draw < cumulative:
                    break
        if chosen is None:
            continue
        target, gain, weight_to_target, margin = chosen
        joins.append((v in well_connected, gain, margin))
        refined[v] = target
        rsize[target] += 1
        rsize[v] = 0
        kref[target] += strengths[v]
        eext[target] = eext[target] + eext[v] - 2.0 * weight_to_target
    return refined, joins


@pytest.mark.parametrize("seed", range(20))
def test_refined_communities_grow_by_valid_joins(seed):
    # Gamma-connectivity (Theorem 5) rests on every join adding a well-connected
    # singleton v to a well-connected refined community D (D in T) with
    # f(v, D) >= 0. The replay shows that the kernel's refined communities are
    # produced by such joins.
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(4, 15))
    graph = build_leiden_graph(_random_small_graph(rng, n), 1)
    in_subset = rng.random(n) < 0.8
    in_subset[:3] = True
    labels = np.where(in_subset, 0, 1).astype(np.int32)
    members = np.flatnonzero(in_subset).astype(np.int32)
    theta_w = THETA * graph.mean_weight
    keys, vals, used, shift = _hash_scratch(graph.max_degree)
    width = graph.max_degree + 1
    n_joins = 0
    for attempt, gamma in enumerate((0.1, 0.5, 1.0, 2.0) * 2):
        null_coef = gamma / (graph.strength_scale * float(graph.total_strength))
        key = np.uint64(stream_key(np.uint64(seed), attempt, 0, PHASE_REFINE))
        refined = np.empty(n, dtype=np.int32)
        merge_nodes_subset(
            0,
            0,
            members.size,
            members,
            graph.indptr,
            graph.indices,
            graph.weights,
            graph.strengths,
            labels,
            null_coef,
            theta_w,
            key,
            refined,
            np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.int64),
            np.empty(n, dtype=np.float64),
            np.empty(n, dtype=np.int32),
            keys[0],
            vals[0],
            used[0],
            shift,
            np.empty(width, dtype=np.int32),
            np.empty(width, dtype=np.float64),
            np.empty(width, dtype=np.float64),
        )
        replayed, joins = _replay_refinement(
            graph, labels, members, 0, null_coef, theta_w, key
        )
        assert {int(v): int(refined[v]) for v in members} == replayed
        for in_r, gain, margin in joins:
            assert in_r
            assert gain >= 0.0
            assert margin >= 0.0
        n_joins += len(joins)
    assert n_joins > 0


def _block_graph(n: int, block: int, intra: int, seed: int) -> sp.csr_matrix:
    """Dense blocks of ``block`` nodes joined by one random edge per node."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), intra + 1)
    cols = (rows // block) * block + rng.integers(0, block, size=rows.size)
    outside = np.arange(rows.size) % (intra + 1) == intra
    cols[outside] = rng.integers(0, n, size=int(outside.sum()))
    weights = rng.uniform(0.2, 1.0, size=rows.size)
    return sp.csr_matrix((weights, (rows, np.minimum(cols, n - 1))), shape=(n, n))


def test_native_backend_releases_the_gil():
    # Warm up with the same dtypes first: compiling or loading kernels holds the
    # GIL, the kernels themselves must not.
    leiden_membership(_block_graph(2000, 100, 10, 0), 1.0, 1, nthreads=2)
    W = _block_graph(300_000, 100, 10, 1)
    ticks: list[float] = []
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.01)

    thread = threading.Thread(target=heartbeat)
    thread.start()
    try:
        started = time.perf_counter()
        leiden_membership(W, 1.0, 1, nthreads=2)
        finished = time.perf_counter()
    finally:
        stop.set()
        thread.join()
    inside = [tick for tick in ticks if started <= tick <= finished]
    gaps = np.diff([started, *inside, finished])
    # The call takes about a second; level-0 local moving alone runs for some
    # 0.3 s of it, and a kernel holding the GIL would stop the heartbeat that
    # long. With the GIL released the gaps stay near the 10 ms period.
    assert inside
    assert gaps.max() < 0.2


def test_sequential_queue_cap_stops_local_moving(monkeypatch):
    W = ring_of_cliques(10, 5)
    monkeypatch.setattr(native_module, "QUEUE_EVALUATIONS_PER_NODE", 0)
    with _debug_messages() as messages:
        labels = native_leiden_membership(W, 1.0, 1, 2)
    # Without evaluations no node moves, so every node stays alone.
    np.testing.assert_array_equal(labels, np.arange(1, 51))
    assert any("stopped at its cap" in message for message in messages)
    monkeypatch.setattr(native_module, "QUEUE_EVALUATIONS_PER_NODE", 1)
    with _debug_messages() as messages:
        runs = [native_leiden_membership(W, 1.0, 1, threads) for threads in (1, 3)]
    np.testing.assert_array_equal(runs[1], runs[0])
    assert valid_labelling(runs[0], 50)
    assert communities_connected(W, runs[0])
    assert any("stopped at its cap" in message for message in messages)


def test_guards_stay_silent_on_ordinary_graphs():
    with _debug_messages() as messages:
        native_leiden_membership(knn_graph(2000, 10), 1.0, 1, None)
        native_leiden_membership(planted(4, 50, 0.3, 0.01, 5)[0], 1.0, 1, None)
        native_leiden_membership(ring_of_cliques(30, 5), 1.0, 1, None, **FORCED)
    assert messages
    assert not any("stopped at its cap" in message for message in messages)
    assert not any("refinement retry" in message for message in messages)


def test_adversarial_star():
    n_leaves = 200_000
    W = sp.csr_matrix(
        (
            np.ones(n_leaves),
            (np.zeros(n_leaves, dtype=np.int64), np.arange(1, n_leaves + 1)),
        ),
        shape=(n_leaves + 1, n_leaves + 1),
    )
    labels = native_leiden_membership(W, 1.0, 1, 4)
    assert valid_labelling(labels, n_leaves + 1)
    assert communities_connected(W, labels)
    np.testing.assert_array_equal(native_leiden_membership(W, 1.0, 1, 1), labels)


def test_adversarial_weight_range():
    W, _ = planted(8, 60, 0.12, 0.02, 3)
    W.data = 10.0 ** np.random.default_rng(8).uniform(-150.0, 150.0, size=W.nnz)
    labels = native_leiden_membership(W, 1.0, 1, None)
    assert valid_labelling(labels, W.shape[0])
    assert communities_connected(W, labels)
    _assert_gamma_separated(W, labels, 1.0)
    for factor in (2.0**-200, 2.0**200):
        np.testing.assert_array_equal(
            native_leiden_membership(W * factor, 1.0, 1, None), labels
        )


def test_adversarial_zero_strength_node():
    # Node 10 hangs on node 0 by a weight of 1e-40, so its fixed-point strength
    # rounds to 0 (no null-model term and no move margin). Joining its only
    # neighbour's community is still the one move with a positive gain.
    W = cliques_graph([10], 1).tolil()
    W[0, 10] = 1e-40
    W = W.tocsr()
    assert build_leiden_graph(W, 1).strengths[10] == 0
    labels = native_leiden_membership(W, 1.0, 1, None)
    assert labels[10] == labels[0]
    assert valid_labelling(labels, 11)
    assert communities_connected(W, labels)


def _random_connected_graph(n: int, extra: int, seed: int) -> sp.csr_matrix:
    """A random tree on ``n`` nodes plus ``extra`` random edges."""
    rng = np.random.default_rng(seed)
    rows = np.concatenate([np.arange(1, n), rng.integers(0, n, extra)])
    cols = np.concatenate(
        [[rng.integers(0, node) for node in range(1, n)], rng.integers(0, n, extra)]
    )
    weights = rng.uniform(0.5, 2.0, size=rows.size)
    return sp.csr_matrix((weights, (rows, cols)), shape=(n, n))


def test_extreme_resolutions():
    parts = [_random_connected_graph(size, 2 * size, size) for size in (30, 40, 50)]
    W = sp.block_diag([*parts, sp.csr_matrix((2, 2))], format="csr")
    n_components, components = connected_components(_symmetric(W), directed=False)
    coarse = native_leiden_membership(W, 1e-8, 1, None)
    assert np.unique(coarse).size == n_components
    assert adjusted_rand_score(components, coarse) == 1.0
    fine = native_leiden_membership(W, 1e8, 1, None)
    np.testing.assert_array_equal(fine, np.arange(1, W.shape[0] + 1))


def test_subnormal_weights_give_identical_labels():
    W, _ = planted(4, 50, 0.3, 0.01, 5)
    tiny = W * 2.0**-1040
    assert tiny.data.max() < np.finfo(np.float64).tiny
    np.testing.assert_array_equal(
        native_leiden_membership(tiny, 1.0, 3, None),
        native_leiden_membership(W, 1.0, 3, None),
    )


# ---------------------------------------------------------------------------
# 7.3 DataStore integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def leiden_store_template(datastore_zarr_root, tmp_path_factory):
    """A copy of the 1K PBMC store with a small-k graph, built once."""
    from cytearc.datastore.datastore import DataStore
    from tests.fixtures_datastore import build_neighbourhood_graph

    path = tmp_path_factory.mktemp("native_leiden") / "store.zarr"
    shutil.copytree(datastore_zarr_root, path)
    datastore = DataStore(str(path), default_assay="RNA")
    cells = datastore.qc.auto_filter()
    features = datastore.features.hvgs(
        cells,
        from_assay="RNA",
        top_n=100,
        show_plot=False,
        min_cells=int(0.01 * datastore.cells.N),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )
    graph = build_neighbourhood_graph(
        datastore,
        from_assay="RNA",
        cell_selection=cells,
        features=features,
        dims=5,
        k=3,
        n_centroids=10,
        local_cache=False,
    )
    return path, graph


@pytest.fixture
def leiden_store(leiden_store_template, tmp_path):
    from cytearc.datastore.datastore import DataStore

    template, graph = leiden_store_template
    target = tmp_path / "store.zarr"
    shutil.copytree(template, target)
    return DataStore(str(target), default_assay="RNA"), graph


def _stored_labels(datastore, ref) -> np.ndarray:
    from cytearc.metadata.artifacts import artifact_values
    from cytearc.storage.artifacts import artifact_group

    return np.asarray(artifact_values(artifact_group(datastore.zw, ref), "values"))


def test_datastore_uses_the_native_backend_by_default(leiden_store):
    datastore, graph = leiden_store
    ref = datastore.clusters.leiden(graph)
    assert datastore.artifacts.inspect(ref).parameters["backend"] == "cytearc"
    labels = _stored_labels(datastore, ref)
    n_cells = datastore.graph.load(graph).shape[0]
    assert labels.shape == (n_cells,)
    assert labels.dtype.kind == "i"
    assert labels.min() == 1
    assert np.array_equal(np.unique(labels), np.arange(1, labels.max() + 1))


def test_datastore_thread_count_is_not_part_of_the_identity(leiden_store):
    datastore, graph = leiden_store
    first = datastore.clusters.leiden(graph)
    expected = _stored_labels(datastore, first)
    original = datastore.nthreads
    try:
        datastore.nthreads = 1
        assert datastore.clusters.leiden(graph) == first
        single = datastore.clusters.leiden(graph, invalidate_cache=True)
        np.testing.assert_array_equal(_stored_labels(datastore, single), expected)
        datastore.nthreads = original
        again = datastore.clusters.leiden(graph, invalidate_cache=True)
        np.testing.assert_array_equal(_stored_labels(datastore, again), expected)
    finally:
        datastore.nthreads = original


def test_datastore_rejects_an_unknown_backend(leiden_store):
    datastore, graph = leiden_store
    before = datastore.artifacts.list(kind="cluster_labels", complete_only=False)
    with pytest.raises(ValueError, match="backend"):
        datastore.clusters.leiden(graph, backend="unknown")  # type: ignore[arg-type]
    assert (
        datastore.artifacts.list(kind="cluster_labels", complete_only=False) == before
    )


# ---------------------------------------------------------------------------
# 7.4 Slow tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fifty_thousand_node_graph():
    W = knn_graph(50_000, 11, seed=7)
    default = native_leiden_membership(W, 1.0, 1, 1)
    assert communities_connected(W, default)
    for threads in (3, 4):
        np.testing.assert_array_equal(
            native_leiden_membership(W, 1.0, 1, threads), default
        )
    # This graph is below SEQUENTIAL_LEVEL_ENTRIES, so the group scheme is forced
    # to compare it with the sequential queue.
    groups = native_leiden_membership(W, 1.0, 1, 4, _sequential_level_entries=0)
    for threads in (1, 3):
        np.testing.assert_array_equal(
            native_leiden_membership(W, 1.0, 1, threads, _sequential_level_entries=0),
            groups,
        )
    assert communities_connected(W, groups)
    sequential = native_leiden_membership(W, 1.0, 1, 4, _sequential_level_entries=2**62)
    assert reference_modularity(W, groups, 1.0) >= (
        reference_modularity(W, sequential, 1.0) - 0.002
    )


# ---------------------------------------------------------------------------
# Kernels run as plain Python
# ---------------------------------------------------------------------------


_KERNEL_MODULES = (
    "cytearc.clustering._leiden_primitives",
    "cytearc.clustering._leiden_graph",
    "cytearc.clustering._leiden_moving",
    "cytearc.clustering._leiden_refine",
    "cytearc.clustering._leiden_aggregate",
    "cytearc.clustering._leiden_native",
)


def _run_kernels_as_python(monkeypatch) -> int:
    """Bind every compiled Leiden kernel, wherever it is imported, to its Python source."""
    import importlib

    replaced = 0
    for name in _KERNEL_MODULES:
        module = importlib.import_module(name)
        for attribute, value in list(vars(module).items()):
            if isinstance(value, numba.core.dispatcher.Dispatcher):
                monkeypatch.setattr(module, attribute, value.py_func)
                replaced += 1
    return replaced


def _isolated_and_zero_strength_graph() -> sp.csr_matrix:
    # Two triangles, an isolated node, and a node whose only edge has weight 0.
    rows = [0, 1, 2, 3, 4, 5, 7]
    cols = [1, 2, 0, 4, 5, 3, 0]
    data = [1.0, 2.0, 1.5, 1.0, 1.0, 3.0, 0.0]
    return sp.csr_matrix((data, (rows, cols)), shape=(8, 8))


_PYTHON_KERNEL_CASES = [
    pytest.param(lambda: knn_graph(150, 8, 3), 1.0, 7, 0, id="knn"),
    pytest.param(
        lambda: _random_small_graph(np.random.default_rng(4), 24),
        0.7,
        2,
        0,
        id="random",
    ),
    pytest.param(_isolated_and_zero_strength_graph, 1.0, 1, 0, id="isolated"),
    pytest.param(lambda: _STALLING_CYCLE, 1.0, 0, 8, id="retry"),
    pytest.param(lambda: _STALLING_CYCLE, 1.0, 0, 0, id="safety-stop"),
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("make_graph", "gamma", "seed", "retries"), _PYTHON_KERNEL_CASES
)
def test_kernels_run_as_python_give_the_compiled_labels(
    monkeypatch, path, make_graph, gamma, seed, retries
):
    # Coverage cannot see inside compiled kernels, and the compiled backend must
    # compute exactly what their Python source says, bit for bit.
    graph = make_graph()
    if retries == 0:
        monkeypatch.setattr(native_module, "REFINEMENT_RETRIES", 0)
    expected = native_leiden_membership(graph, gamma, seed, 2, **path)

    assert _run_kernels_as_python(monkeypatch) > 50
    # The random streams wrap uint64 products on purpose; NumPy warns about it.
    with np.errstate(over="ignore"):
        actual = native_leiden_membership(graph, gamma, seed, 2, **path)

    np.testing.assert_array_equal(actual, expected)
    assert valid_labelling(actual, graph.shape[0])


def _one_community_level(n_workers: int, parallel_min_items: int, last_id: bool):
    """The first level of an 80-node kNN graph with every node in one community.

    At resolution 3 every node gains by leaving it, so local moving opens new
    communities. With ``last_id`` only the last community id is still free.
    """
    graph = build_leiden_graph(knn_graph(80, 6, 5), n_workers, parallel_min_items)
    labels, totals, sizes, n_comm = renumber_partition(
        np.zeros(graph.n_nodes, dtype=np.int32),
        graph.strengths,
        n_workers,
        parallel_min_items,
    )
    first_free = totals.shape[0] - 1 if last_id else n_comm
    return graph, (labels, totals, sizes, np.array([first_free], dtype=np.int64))


def _move_level(graph, state, n_workers, parallel_min_items, *schedule):
    """Run ``move_nodes_level`` on a copy of ``state`` and return the result.

    ``schedule`` holds the sequential level entries, the tail divisor and the
    queue evaluations per node.
    """
    sequential_level_entries, tail_divisor, queue_evaluations_per_node = schedule
    labels, totals, sizes, next_free = (array.copy() for array in state)
    keys, vals, used, shift = _hash_scratch(graph.max_degree, n_workers)
    stats = moving_module.move_nodes_level(
        graph.indptr,
        graph.indices,
        graph.weights,
        graph.strengths,
        labels,
        totals,
        sizes,
        next_free,
        np.uint64(3),
        0,
        0,
        3.0 / (graph.strength_scale * float(graph.total_strength)),
        native_module.MOVE_TOLERANCE / graph.strength_scale,
        n_workers,
        sequential_level_entries,
        tail_divisor,
        native_module.MAX_PARALLEL_SWEEPS,
        2,
        parallel_min_items,
        queue_evaluations_per_node,
        keys,
        vals,
        used,
        shift,
    )
    return labels, totals, sizes, next_free, stats


def _move_level_compiled_and_as_python(
    monkeypatch, n_workers, parallel_min_items, *schedule, last_id=False
):
    """Check that one level moves identically compiled and as Python.

    Returns the first free id before the level, the one after it, and the
    level statistics.
    """
    graph, state = _one_community_level(n_workers, parallel_min_items, last_id)
    compiled = _move_level(graph, state, n_workers, parallel_min_items, *schedule)
    _run_kernels_as_python(monkeypatch)
    with np.errstate(over="ignore"):
        python = _move_level(graph, state, n_workers, parallel_min_items, *schedule)

    for actual, expected in zip(python, compiled):
        np.testing.assert_array_equal(actual, expected)
    labels, totals, sizes, next_free, stats = python
    # The community state still describes the labels, which use handed-out ids.
    expected_totals = np.zeros_like(totals)
    np.add.at(expected_totals, labels, graph.strengths)
    np.testing.assert_array_equal(totals, expected_totals)
    np.testing.assert_array_equal(sizes, np.bincount(labels, minlength=sizes.size))
    assert labels.max() < next_free[0] <= totals.size
    return int(state[3][0]), int(next_free[0]), stats


@pytest.mark.parametrize(
    ("queue_evaluations_per_node", "last_id"),
    [(1000, False), (1000, True), (1, False)],
    ids=["free-ids", "last-free-id", "evaluation-cap"],
)
def test_the_sequential_queue_runs_as_python(
    monkeypatch, queue_evaluations_per_node, last_id
):
    first_free, next_free, stats = _move_level_compiled_and_as_python(
        monkeypatch, 1, 256, 2**62, 64, queue_evaluations_per_node, last_id=last_id
    )

    assert stats[4] == 0
    if queue_evaluations_per_node == 1:
        # Each node is evaluated once before the cap stops the queue.
        assert stats[0] == 80 and stats[5] == 1
    else:
        assert stats[5] == 0
    if last_id:
        # The first node to leave takes the last id; later leavers stay.
        assert next_free == first_free + 1
    else:
        assert next_free > first_free


@pytest.mark.parametrize("parallel_min_items", [256, 1], ids=["serial", "parallel"])
def test_group_sweeps_hand_over_to_the_tail_queue_as_python(
    monkeypatch, parallel_min_items
):
    # With a tail divisor of 1 the queue takes over once a sweep leaves any
    # node inactive.
    first_free, next_free, stats = _move_level_compiled_and_as_python(
        monkeypatch, 2, parallel_min_items, 0, 1, 1000
    )

    assert stats[4] == 1 and stats[2] == 1 and stats[3] > 0
    assert next_free > first_free


def test_a_group_asking_for_more_ids_than_remain_stays_as_python(monkeypatch):
    # Every node asks for a new community and every group needs more than the
    # one id left, so no node moves.
    first_free, next_free, stats = _move_level_compiled_and_as_python(
        monkeypatch, 2, 256, 0, 1, 1000, last_id=True
    )

    assert stats[4] == 1 and stats[1] == 0
    assert next_free == first_free


@pytest.fixture(params=["compiled", "python"])
def kernel_mode(request, monkeypatch):
    """Run a test once with the compiled kernels and once with their source."""
    if request.param == "python":
        _run_kernels_as_python(monkeypatch)
    # The random streams and hashes wrap uint64 products on purpose.
    with np.errstate(over="ignore"):
        yield request.param


@pytest.mark.usefixtures("kernel_mode")
def test_hash_probing_steps_past_a_colliding_key():
    capacity, shift = hash_capacity(8)
    candidates = np.arange(1, 400, dtype=np.uint64)
    home = ((candidates * GOLDEN) >> shift).astype(np.int64)
    slot = int(np.bincount(home).argmax())
    first, second, absent = (int(key) for key in candidates[home == slot][:3])
    keys = np.full(capacity, EMPTY_SLOT, dtype=np.int32)
    used = np.empty(capacity, dtype=np.int32)

    n_used = primitives_module.acc_insert(keys, used, 0, first, shift)
    n_used = primitives_module.acc_insert(keys, used, n_used, second, shift)
    n_used = primitives_module.acc_insert(keys, used, n_used, second, shift)

    assert n_used == 2
    assert primitives_module.acc_find(keys, first, shift) == slot
    assert primitives_module.acc_find(keys, second, shift) == (slot + 1) % capacity
    assert primitives_module.acc_find(keys, absent, shift) == -1


@pytest.mark.usefixtures("kernel_mode")
def test_sort_ids_sorts_short_and_long_prefixes():
    rng = np.random.default_rng(6)
    for count in (0, 1, 5, INSERTION_SORT_LIMIT, INSERTION_SORT_LIMIT + 1, 300):
        ids = rng.permutation(1000)[: count + 4].astype(np.int32)
        expected = np.concatenate([np.sort(ids[:count]), ids[count:]])
        primitives_module.sort_ids(ids, count)
        np.testing.assert_array_equal(ids, expected)


@pytest.mark.usefixtures("kernel_mode")
def test_scratch_rows_are_checked_against_the_worker_count():
    primitives_module.check_worker_rows(2, 2)
    with pytest.raises(ValueError, match="n_workers must be at least 1"):
        primitives_module.check_worker_rows(2, 0)
    with pytest.raises(ValueError, match="one row per worker"):
        primitives_module.check_worker_rows(1, 2)


@pytest.mark.usefixtures("kernel_mode")
@pytest.mark.parametrize(("n_workers", "min_items"), [(1, 256), (2, 1)])
def test_stable_bucket_order_sorts_many_keys_range_by_range(n_workers, min_items):
    rng = np.random.default_rng(13)
    n_keys = 3 * BUCKET_DIRECT_KEYS + 5
    keys = rng.integers(0, n_keys, size=20_000).astype(np.int32)

    start, order = primitives_module.stable_bucket_order(
        keys, n_keys, n_workers, min_items
    )

    np.testing.assert_array_equal(order, np.argsort(keys, kind="stable"))
    np.testing.assert_array_equal(
        start, np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=n_keys))])
    )


@pytest.mark.usefixtures("kernel_mode")
def test_split_into_components_joins_a_root_found_through_a_later_node():
    # Edges 0-2 and 1-2: node 2 joins root 0 first, so node 1 reaches the
    # smaller root 0 through it.
    graph = build_leiden_graph(
        sp.csr_matrix(([1.0, 1.0], ([0, 1], [2, 2])), shape=(3, 3)), 1
    )
    labels = np.zeros(3, dtype=np.int32)

    components = aggregate_module.split_into_components(
        graph.indptr, graph.indices, labels
    )

    np.testing.assert_array_equal(components, [0, 0, 0])


def _merge_subset(merge, graph, labels, members, gamma: float):
    """Refine the community of ``members`` and return its count and labels."""
    n = graph.n_nodes
    refined = np.full(n, -5, dtype=np.int32)
    rsize = np.full(n, -5, dtype=np.int32)
    kref = np.full(n, -5, dtype=np.int64)
    eext = np.full(n, np.nan)
    rlist = np.full(n, -5, dtype=np.int32)
    keys, vals, used, shift = _hash_scratch(graph.max_degree)
    width = graph.max_degree + 1
    null_coef = gamma / (graph.strength_scale * float(graph.total_strength))
    count = merge(
        int(labels[members[0]]),
        0,
        members.size,
        members,
        graph.indptr,
        graph.indices,
        graph.weights,
        graph.strengths,
        labels,
        null_coef,
        THETA * graph.mean_weight,
        np.uint64(stream_key(np.uint64(0), 0, 0, PHASE_REFINE)),
        refined,
        rsize,
        kref,
        eext,
        rlist,
        keys[0],
        vals[0],
        used[0],
        shift,
        np.empty(width, dtype=np.int32),
        np.empty(width, dtype=np.float64),
        np.empty(width, dtype=np.float64),
    )
    assert count == _brute_force_merge_check(
        graph, labels, members, null_coef, refined, rsize, kref, eext
    )
    return count, refined[members]


@pytest.mark.usefixtures("kernel_mode")
def test_a_member_whose_neighbours_are_poorly_connected_stays_alone():
    # Node 0 is well connected to its community {0, 1, 2}, but nodes 1 and 2
    # lean on nodes outside it, so neither is a community node 0 may join.
    W = sp.csr_matrix(
        ([1.0, 1.0, 50.0, 50.0], ([0, 1, 1, 2], [1, 2, 3, 4])), shape=(5, 5)
    )
    graph = build_leiden_graph(W, 1)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)

    count, refined = _merge_subset(
        refine_module.merge_nodes_subset,
        graph,
        labels,
        np.arange(3, dtype=np.int32),
        1.0,
    )

    assert count == 3
    np.testing.assert_array_equal(refined, [0, 1, 2])


def test_a_draw_of_one_keeps_every_shuffle_swap_in_range(monkeypatch):
    # Draws are multiples of 2**-53 below 1, so floor(u * (i + 1)) never exceeds
    # i. A stub that returns 1.0 reaches the guard that clamps the swap to i.
    def draw_one(key, counter):
        return 1.0

    monkeypatch.setattr(moving_module, "uniform01", draw_one)
    monkeypatch.setattr(refine_module, "uniform01", draw_one)
    triangle = build_leiden_graph(sp.csr_matrix(1.0 - np.eye(3)), 1)

    permutation = moving_module.random_permutation.py_func(6, np.uint64(3))
    count, refined = _merge_subset(
        refine_module.merge_nodes_subset.py_func,
        triangle,
        np.zeros(3, dtype=np.int32),
        np.arange(3, dtype=np.int32),
        0.05,
    )

    np.testing.assert_array_equal(permutation, np.arange(6))
    # A draw of 1 also picks the last candidate, so every node joins node 2.
    assert count == 1
    np.testing.assert_array_equal(refined, [2, 2, 2])


def test_the_group_count_is_a_power_of_two():
    # Parallel sweeps pick a node's group from the top bits of a hash.
    assert native_module.GROUPS == 1 << native_module._GROUP_BITS


@pytest.mark.filterwarnings("ignore:overflow encountered in ldexp:RuntimeWarning")
def test_strengths_that_overflow_after_normalisation_are_rejected():
    # The symmetrized weights have a finite total, but scaling them by the tiny
    # mean off-diagonal weight sends the self-loop's strength to infinity.
    graph = sp.csr_matrix(
        ([1e300, 1e-300, 1e-300], ([0, 0, 1], [0, 1, 2])), shape=(3, 3)
    )

    with pytest.raises(ValueError, match="symmetrized total is not finite"):
        native_leiden_membership(graph, 1.0, 0, 1)
