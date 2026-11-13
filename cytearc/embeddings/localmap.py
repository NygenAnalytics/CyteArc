"""Lay out cells with LocalMAP, from a weighted neighbor graph and coordinates.

LocalMAP (Wang et al., AAAI 2025) extends PaCMAP (Wang et al., JMLR 2021).
Both place each cell by three kinds of pairs: its neighbors, which attract
it; mid-near cells, the second closest of six random cells, which attract it
weakly and hold the global arrangement; and further pairs, random cells that
repel it. Adam moves the cells over three phases of iterations. The mid-near
weight falls from 1000 to 3 over the first phase, stays at 3 in the second,
and is 0 in the third. In the third phase LocalMAP scales each neighbor's
attraction by ``max_distance / 2 / sqrt(d)`` and, every tenth iteration, draws
each cell's further pairs again from the cells within ``max_distance`` of it
in the layout, which separates clusters that sit next to each other.

PaCMAP's neighbors are 10 unweighted pairs per cell. Here they are the edges
of a graph, such as a connectivity map, and each edge attracts in proportion
to its weight. The weights are scaled so that the attraction of all edges
sums to that of 10 unweighted pairs per cell, which keeps PaCMAP's balance of
attraction and repulsion for a graph of any number of neighbors.

Each cell's gradient sums the forces of every pair it belongs to, its own
pairs and the pairs that name it, so cells are computed independently and the
layout does not depend on the number of threads. Every random draw comes from
SplitMix64's finaliser of a key that the seed, the kind of draw and the cell
give, plus a counter, so pairs do not depend on threads either.

Adapted from the PaCMAP package 0.9.1 (Apache-2.0), which holds the LocalMAP
reference implementation. The reference draws from one random state shared
by its threads; here, as in brisc (MIT), each cell draws from its own key.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import prange
from scipy.sparse import csr_matrix, issparse

from ..utils.arguments import float_argument, integer_argument
from ..utils.digest import mix64, mix64_array
from ..utils.numba import njit, numba_threads
from ..utils.progress import iter_progress

__all__ = [
    "LOCALMAP_ITERATIONS",
    "NEIGHBOR_PAIRS",
    "LocalmapSettings",
    "further_pairs",
    "initial_layout",
    "localmap_layout",
    "localmap_peak_bytes",
    "localmap_settings",
    "mid_near_pairs",
    "neighbor_partners",
    "pair_partners",
]

#: Iterations of the three phases in the reference implementation.
LOCALMAP_ITERATIONS = (100, 100, 250)
#: PaCMAP's neighbor pairs per cell, whose attraction a graph's edges share.
NEIGHBOR_PAIRS = 10

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SALT = np.uint64(0x2545F4914F6CDD1D)
_SHIFT_32 = np.uint64(32)
# Kinds of random draws, each with its own keys.
_MID_NEAR_DRAWS = 1
_FURTHER_DRAWS = 2
_NEARBY_DRAWS = 3
_MID_NEAR_CANDIDATES = 6
_MID_NEAR_START_WEIGHT = 1000.0
_NEARBY_TRIALS = 100
_REDRAW_INTERVAL = 10
_BETA1 = 0.9
_BETA2 = 0.999
_ADAM_EPSILON = 1e-7
# An initialization is centered and scaled to this standard deviation, as
# the reference scales coordinates that it is given.
_INITIAL_SCALE = 1e-4


@dataclass(frozen=True, slots=True)
class LocalmapSettings:
    """LocalMAP settings as canonical Python numbers."""

    localmap_dims: int
    n_mid_near: int
    n_further: int
    iterations: tuple[int, int, int]
    learning_rate: float
    max_distance: float
    random_seed: int


def localmap_settings(
    *,
    localmap_dims: object,
    n_mid_near: object,
    n_further: object,
    iterations: object,
    learning_rate: object,
    max_distance: object,
    random_seed: object,
) -> LocalmapSettings:
    """Validate LocalMAP settings and return them as canonical Python numbers.

    Raises:
        TypeError: If a setting has the wrong type, or ``iterations`` is not
            three values.
        ValueError: If a setting is out of range.
    """
    dims = integer_argument(localmap_dims, "localmap_dims", minimum=1)
    mid_near = integer_argument(n_mid_near, "n_mid_near", minimum=1)
    further = integer_argument(n_further, "n_further", minimum=1)
    if not isinstance(iterations, tuple | list) or len(iterations) != 3:
        raise TypeError("iterations must be three integers, one per phase")
    phases = [
        integer_argument(value, f"iterations[{phase}]", minimum=1)
        for phase, value in enumerate(iterations)
    ]
    rate = float_argument(learning_rate, "learning_rate")
    distance = float_argument(max_distance, "max_distance")
    for name, value in (("learning_rate", rate), ("max_distance", distance)):
        if value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return LocalmapSettings(
        localmap_dims=dims,
        n_mid_near=mid_near,
        n_further=further,
        iterations=(phases[0], phases[1], phases[2]),
        learning_rate=rate,
        max_distance=distance,
        random_seed=integer_argument(random_seed, "random_seed", minimum=0),
    )


def draw_key(random_seed: int, kind: int, round_: int = 0) -> np.uint64:
    """Return the key of one kind of random draws of a layout."""
    words = np.array([random_seed % 2**64, kind, round_], dtype=np.uint64)
    mixed = mix64_array(words + np.array([_SALT, _GOLDEN, _GOLDEN], dtype=np.uint64))
    return np.uint64(mix64_array(mixed[:1] ^ mixed[1:2] ^ mix64_array(mixed[2:]))[0])


@njit(cache=True, nogil=True, inline="always")
def _cell_key(key: np.uint64, cell: int) -> np.uint64:
    return mix64(key ^ mix64(np.uint64(cell) + _GOLDEN))


@njit(cache=True, nogil=True, inline="always")
def _random_below(key: np.uint64, counter: int, bound: int) -> int:
    """Return draw ``counter`` of a cell's key, an integer from 0 to ``bound - 1``."""
    mixed = mix64(key + np.uint64(counter) * _GOLDEN)
    return int(((mixed >> _SHIFT_32) * np.uint64(bound)) >> _SHIFT_32)


def _chunks(n_cells: int) -> int:
    """Return how many blocks of cells a parallel kernel splits the cells into.

    Every kernel computes each cell independently, so results do not depend
    on this count, which only balances the threads.
    """
    import numba

    return max(1, min(n_cells, 8 * numba.get_num_threads()))


def localmap_peak_bytes(
    n_cells: int,
    n_edges: int,
    *,
    dims: int,
    n_mid_near: int,
    n_further: int,
) -> int:
    """Estimate the bytes a LocalMAP layout holds besides its graph.

    Each cell's partners list every pair in both directions: the graph's
    edges with float32 weights, and the mid-near and further pairs, whose
    rows are kept as well. Building a partner list holds each pair's two
    cells once more. The layout, its gradient and Adam's two moments are
    float32.

    Args:
        n_cells: Cells of the graph.
        n_edges: Stored edges of the graph.
        dims: Dimensions of the layout.
        n_mid_near: Mid-near pairs per cell.
        n_further: Further pairs per cell.

    Returns:
        The estimated peak in bytes.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    edges = integer_argument(n_edges, "n_edges", minimum=0)
    layout_dims = integer_argument(dims, "dims", minimum=1)
    mid_near = integer_argument(n_mid_near, "n_mid_near", minimum=0)
    further = integer_argument(n_further, "n_further", minimum=0)
    neighbor_partners_bytes = 2 * edges * (4 + 4) + 8 * (cells + 1)
    pair_rows = 4 * cells * (mid_near + further)
    pair_partners_bytes = 2 * pair_rows + 16 * (cells + 1)
    building = 2 * 4 * cells * max(mid_near, further, 1)
    optimizer = 4 * 4 * cells * layout_dims
    return (
        neighbor_partners_bytes + pair_rows + pair_partners_bytes + building + optimizer
    )


@njit(cache=True, nogil=True, parallel=True)
def _partners(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    n_chunks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """List each cell's pairs in both directions; ``weights`` may be empty."""
    n_cells = indptr.shape[0] - 1
    weighted = weights.shape[0] > 0
    block = (n_cells + n_chunks - 1) // n_chunks
    # Bucket the pairs by the block of their second cell, in the order of
    # their first cell: count them, lay out the buckets, then fill them.
    counts = np.zeros((n_chunks, n_chunks), dtype=np.int64)
    for unsigned_chunk in prange(n_chunks):
        chunk = np.int64(unsigned_chunk)
        for i in range(chunk * block, min(n_cells, (chunk + 1) * block)):
            for p in range(indptr[i], indptr[i + 1]):
                counts[chunk, indices[p] // block] += 1
    offsets = np.empty((n_chunks, n_chunks), dtype=np.int64)
    bucket_start = np.empty(n_chunks + 1, dtype=np.int64)
    total = 0
    for target_block in range(n_chunks):
        bucket_start[target_block] = total
        for source_block in range(n_chunks):
            offsets[source_block, target_block] = total
            total += counts[source_block, target_block]
    bucket_start[n_chunks] = total
    sources = np.empty(total, dtype=np.int32)
    positions = np.empty(total, dtype=np.int64)
    for unsigned_chunk in prange(n_chunks):
        chunk = np.int64(unsigned_chunk)
        cursor = offsets[chunk].copy()
        for i in range(chunk * block, min(n_cells, (chunk + 1) * block)):
            for p in range(indptr[i], indptr[i + 1]):
                bucket = indices[p] // block
                position = cursor[bucket]
                cursor[bucket] = position + 1
                sources[position] = i
                positions[position] = p
    degrees = np.empty(n_cells, dtype=np.int64)
    for i in prange(n_cells):
        degrees[i] = indptr[i + 1] - indptr[i]
    for unsigned_target in prange(n_chunks):
        target = np.int64(unsigned_target)
        for position in range(bucket_start[target], bucket_start[target + 1]):
            degrees[indices[positions[position]]] += 1
    out_indptr = np.empty(n_cells + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_cells):
        out_indptr[i + 1] = out_indptr[i] + degrees[i]
    partners = np.empty(out_indptr[n_cells], dtype=np.int32)
    out_weights = np.empty(out_indptr[n_cells] if weighted else 0, dtype=np.float32)
    for unsigned_target in prange(n_chunks):
        target = np.int64(unsigned_target)
        first = target * block
        fill = np.empty(block, dtype=np.int64)
        for i in range(first, min(n_cells, first + block)):
            start = out_indptr[i]
            for p in range(indptr[i], indptr[i + 1]):
                partners[start + p - indptr[i]] = indices[p]
                if weighted:
                    out_weights[start + p - indptr[i]] = weights[p]
            fill[i - first] = start + indptr[i + 1] - indptr[i]
        for position in range(bucket_start[target], bucket_start[target + 1]):
            p = positions[position]
            j = indices[p]
            partners[fill[j - first]] = sources[position]
            if weighted:
                out_weights[fill[j - first]] = weights[p]
            fill[j - first] += 1
    return out_indptr, partners, out_weights


def pair_partners(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """List every cell's partners in a set of pairs, in both directions.

    Row i of ``pairs`` holds the pairs (i, j). A cell's partners are its own
    pairs' cells in row order, then the cells that name it, in ascending order,
    once for each time they name it.

    Returns:
        The CSR offsets and the int32 partner cells.
    """
    rows = np.ascontiguousarray(pairs, dtype=np.int32)
    n_cells, width = rows.shape
    indptr = np.arange(n_cells + 1, dtype=np.int64) * width
    out_indptr, partners, _ = _partners(
        indptr, rows.reshape(-1), np.empty(0, dtype=np.float32), _chunks(n_cells)
    )
    return out_indptr, partners


def _checked_graph(graph: csr_matrix) -> csr_matrix:
    """Return the graph as CSR with float32 weights, rejecting what LocalMAP cannot use.

    A CSR graph with float32 weights and no stored zeros is returned as it
    is; any other is copied, so the caller's matrix is never changed.
    """
    if not issparse(graph):
        raise TypeError("The neighbor graph must be a scipy sparse matrix")
    matrix = csr_matrix(graph)
    n_cells = matrix.shape[0]
    if matrix.shape != (n_cells, n_cells):
        raise ValueError(f"The neighbor graph must be square, not {matrix.shape}")
    if n_cells >= np.iinfo(np.int32).max:
        raise ValueError("LocalMAP lays out fewer than 2**31 - 1 cells")
    weights = np.asarray(matrix.data)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Neighbor graph weights must be finite and non-negative")
    if np.any(matrix.diagonal() != 0):
        raise ValueError("The neighbor graph must not join a cell to itself")
    if (
        isinstance(graph, csr_matrix)
        and weights.dtype == np.float32
        and np.all(weights > 0)
    ):
        return graph
    matrix = csr_matrix(
        (weights.astype(np.float32), matrix.indices.copy(), matrix.indptr.copy()),
        shape=matrix.shape,
    )
    matrix.eliminate_zeros()
    return matrix


def neighbor_partners(
    graph: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """List each cell's neighbors in a graph, in both directions, with scaled weights.

    An edge (i, j) attracts i and j with its weight, so a cell's partners are
    its own edges' cells and the cells whose edges name it, as in the graph
    plus its transpose. The weights are scaled so that they sum to
    ``2 * NEIGHBOR_PAIRS`` per cell over all cells.

    Args:
        graph: Cells by cells, the weight of each directed edge, such as a
            connectivity map.

    Returns:
        The CSR offsets, the int32 partner cells and their float32 weights.

    Raises:
        TypeError: If ``graph`` is not a sparse matrix.
        ValueError: If the graph is not square, has a negative, non-finite or
            self-joining weight, or leaves a cell without a weighted edge.
    """
    matrix = _checked_graph(graph)
    n_cells = matrix.shape[0]
    indptr, partners, weights = _partners(
        matrix.indptr, matrix.indices, matrix.data, _chunks(n_cells)
    )
    edgeless = np.flatnonzero(np.diff(indptr) == 0)
    if edgeless.size:
        raise ValueError(
            f"LocalMAP needs every cell to have a weighted edge; {edgeless.size} "
            f"cells have none, such as cell {int(edgeless[0])}"
        )
    total = float(np.sum(matrix.data, dtype=np.float64))
    weights *= np.float32(NEIGHBOR_PAIRS * n_cells / total)
    return indptr, partners, weights


@njit(cache=True, nogil=True, parallel=True)
def _mid_near_pairs(
    coordinates: np.ndarray, n_mid_near: int, key: np.uint64, n_chunks: int
) -> np.ndarray:
    n_cells, dims = coordinates.shape
    pairs = np.empty((n_cells, n_mid_near), dtype=np.int32)
    for unsigned_chunk in prange(n_chunks):
        chunk = np.int64(unsigned_chunk)
        candidates = np.empty(_MID_NEAR_CANDIDATES, dtype=np.int64)
        for i in range(chunk * n_cells // n_chunks, (chunk + 1) * n_cells // n_chunks):
            cell_key = _cell_key(key, i)
            counter = 0
            for pair in range(n_mid_near):
                for slot in range(_MID_NEAR_CANDIDATES):
                    while True:
                        draw = _random_below(cell_key, counter, n_cells)
                        counter += 1
                        if draw == i:
                            continue
                        taken = False
                        for earlier in range(slot):
                            if candidates[earlier] == draw:
                                taken = True
                                break
                        if not taken:
                            for earlier in range(pair):
                                if pairs[i, earlier] == draw:
                                    taken = True
                                    break
                        if not taken:
                            candidates[slot] = draw
                            break
                best = np.inf
                second = np.inf
                best_cell = -1
                second_cell = -1
                for slot in range(_MID_NEAR_CANDIDATES):
                    cell = candidates[slot]
                    distance = 0.0
                    for d in range(dims):
                        difference = (
                            np.float64(coordinates[i, d]) - coordinates[cell, d]
                        )
                        distance += difference * difference
                    if distance < best:
                        second, second_cell = best, best_cell
                        best, best_cell = distance, cell
                    elif distance < second:
                        second, second_cell = distance, cell
                pairs[i, pair] = second_cell
    return pairs


def mid_near_pairs(
    coordinates: np.ndarray, n_mid_near: int, random_seed: int
) -> np.ndarray:
    """Draw each cell's mid-near pairs: the second closest of six random cells.

    Args:
        coordinates: Cells by dimensions, the space of the graph. Coordinates
            of a cosine graph should be scaled to unit length first, which
            orders squared Euclidean distances as cosine distances.
        n_mid_near: Mid-near pairs per cell.
        random_seed: Seed of the draws.

    Returns:
        Cells by ``n_mid_near`` int32 cell indices, distinct within a row and
        never the row's cell.

    Raises:
        ValueError: If there are not more than ``n_mid_near + 6`` cells, or a
            coordinate is not finite.
    """
    values = np.asarray(coordinates)
    if values.ndim != 2:
        raise ValueError("Coordinates must be a matrix of cells by dimensions")
    count = integer_argument(n_mid_near, "n_mid_near", minimum=1)
    seed = integer_argument(random_seed, "random_seed", minimum=0)
    n_cells = values.shape[0]
    if n_cells <= count + _MID_NEAR_CANDIDATES:
        raise ValueError(
            f"LocalMAP draws {count} mid-near pairs from six random cells each, "
            f"which needs more than {count + _MID_NEAR_CANDIDATES} cells; "
            f"there are {n_cells}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Coordinates must be finite")
    return _mid_near_pairs(
        np.ascontiguousarray(values, dtype=np.float32),
        count,
        draw_key(seed, _MID_NEAR_DRAWS),
        _chunks(n_cells),
    )


@njit(cache=True, nogil=True, inline="always")
def _is_neighbor(
    indptr: np.ndarray, indices: np.ndarray, cell: int, other: int
) -> bool:
    for p in range(indptr[cell], indptr[cell + 1]):
        if indices[p] == other:
            return True
    return False


@njit(cache=True, nogil=True, parallel=True)
def _further_pairs(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_further: int,
    key: np.uint64,
    n_chunks: int,
) -> np.ndarray:
    n_cells = indptr.shape[0] - 1
    pairs = np.empty((n_cells, n_further), dtype=np.int32)
    for unsigned_chunk in prange(n_chunks):
        chunk = np.int64(unsigned_chunk)
        for i in range(chunk * n_cells // n_chunks, (chunk + 1) * n_cells // n_chunks):
            cell_key = _cell_key(key, i)
            counter = 0
            for pair in range(n_further):
                while True:
                    draw = _random_below(cell_key, counter, n_cells)
                    counter += 1
                    if draw == i or _is_neighbor(indptr, indices, i, draw):
                        continue
                    taken = False
                    for earlier in range(pair):
                        if pairs[i, earlier] == draw:
                            taken = True
                            break
                    if not taken:
                        pairs[i, pair] = draw
                        break
    return pairs


def further_pairs(graph: csr_matrix, n_further: int, random_seed: int) -> np.ndarray:
    """Draw each cell's further pairs: random cells that are not its neighbors.

    Args:
        graph: Cells by cells; a cell's neighbors are the cells its row names.
        n_further: Further pairs per cell.
        random_seed: Seed of the draws.

    Returns:
        Cells by ``n_further`` int32 cell indices, distinct within a row and
        neither the row's cell nor one of its neighbors.

    Raises:
        ValueError: If a cell has fewer than ``n_further`` other cells that
            are not its neighbors.
    """
    matrix = _checked_graph(graph)
    count = integer_argument(n_further, "n_further", minimum=1)
    seed = integer_argument(random_seed, "random_seed", minimum=0)
    n_cells = matrix.shape[0]
    most = int(np.diff(matrix.indptr).max(initial=0))
    if n_cells < most + count + 1:
        raise ValueError(
            f"LocalMAP draws {count} further pairs outside up to {most} neighbors, "
            f"which needs at least {most + count + 1} cells; there are {n_cells}"
        )
    return _further_pairs(
        matrix.indptr,
        matrix.indices,
        count,
        draw_key(seed, _FURTHER_DRAWS),
        _chunks(n_cells),
    )


@njit(cache=True, nogil=True, parallel=True)
def _redraw_nearby_further_pairs(
    embedding: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    pairs: np.ndarray,
    key: np.uint64,
    max_distance_squared: float,
    n_chunks: int,
) -> None:
    n_cells, dims = embedding.shape
    n_further = pairs.shape[1]
    for unsigned_chunk in prange(n_chunks):
        chunk = np.int64(unsigned_chunk)
        # A cell's new pairs, -1 where 100 draws found none; only these are
        # checked for repeats, and a cell keeps its old pair in place of -1.
        drawn = np.empty(n_further, dtype=np.int64)
        for i in range(chunk * n_cells // n_chunks, (chunk + 1) * n_cells // n_chunks):
            cell_key = _cell_key(key, i)
            counter = 0
            for pair in range(n_further):
                drawn[pair] = -1
                for _trial in range(_NEARBY_TRIALS):
                    draw = _random_below(cell_key, counter, n_cells)
                    counter += 1
                    if draw == i:
                        continue
                    distance = 0.0
                    for d in range(dims):
                        difference = np.float64(embedding[i, d]) - embedding[draw, d]
                        distance += difference * difference
                    if distance > max_distance_squared:
                        continue
                    if _is_neighbor(indptr, indices, i, draw):
                        continue
                    taken = False
                    for earlier in range(pair):
                        if drawn[earlier] == draw:
                            taken = True
                            break
                    if not taken:
                        drawn[pair] = draw
                        break
            for pair in range(n_further):
                if drawn[pair] >= 0:
                    pairs[i, pair] = drawn[pair]


@njit(cache=True, nogil=True, parallel=True)
def _gradient(
    embedding: np.ndarray,
    neighbor_indptr: np.ndarray,
    neighbor_partners: np.ndarray,
    neighbor_weights: np.ndarray,
    mid_near_indptr: np.ndarray,
    mid_near_partners: np.ndarray,
    further_indptr: np.ndarray,
    further_partners: np.ndarray,
    w_neighbors: float,
    w_mid_near: float,
    w_further: float,
    neighbor_scale: float,
    gradient: np.ndarray,
    n_chunks: int,
) -> None:
    n_cells, dims = embedding.shape
    for unsigned_chunk in prange(n_chunks):
        chunk = np.int64(unsigned_chunk)
        total = np.empty(dims, dtype=np.float64)
        difference = np.empty(dims, dtype=np.float64)
        for i in range(chunk * n_cells // n_chunks, (chunk + 1) * n_cells // n_chunks):
            for d in range(dims):
                total[d] = 0.0
            for p in range(neighbor_indptr[i], neighbor_indptr[i + 1]):
                j = neighbor_partners[p]
                distance = 1.0
                for d in range(dims):
                    difference[d] = np.float64(embedding[i, d]) - embedding[j, d]
                    distance += difference[d] * difference[d]
                weight = (
                    w_neighbors
                    * neighbor_weights[p]
                    * 20.0
                    / ((10.0 + distance) * (10.0 + distance))
                )
                if neighbor_scale > 0.0:
                    weight *= neighbor_scale / math.sqrt(distance)
                for d in range(dims):
                    total[d] += weight * difference[d]
            if w_mid_near > 0.0:
                for p in range(mid_near_indptr[i], mid_near_indptr[i + 1]):
                    j = mid_near_partners[p]
                    distance = 1.0
                    for d in range(dims):
                        difference[d] = np.float64(embedding[i, d]) - embedding[j, d]
                        distance += difference[d] * difference[d]
                    weight = (
                        w_mid_near
                        * 20000.0
                        / ((10000.0 + distance) * (10000.0 + distance))
                    )
                    for d in range(dims):
                        total[d] += weight * difference[d]
            for p in range(further_indptr[i], further_indptr[i + 1]):
                j = further_partners[p]
                distance = 1.0
                for d in range(dims):
                    difference[d] = np.float64(embedding[i, d]) - embedding[j, d]
                    distance += difference[d] * difference[d]
                weight = w_further * 2.0 / ((1.0 + distance) * (1.0 + distance))
                for d in range(dims):
                    total[d] -= weight * difference[d]
            for d in range(dims):
                gradient[i, d] = total[d]


@njit(cache=True, nogil=True, parallel=True)
def _adam_step(
    embedding: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: float,
) -> None:
    n_cells, dims = embedding.shape
    for i in prange(n_cells):
        for d in range(dims):
            g = gradient[i, d]
            first_moment[i, d] += (1.0 - _BETA1) * (g - first_moment[i, d])
            second_moment[i, d] += (1.0 - _BETA2) * (g * g - second_moment[i, d])
            embedding[i, d] -= (
                step
                * first_moment[i, d]
                / (math.sqrt(second_moment[i, d]) + _ADAM_EPSILON)
            )


def phase_weights(
    iteration: int, iterations: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Return the mid-near, neighbor and further-pair weights of an iteration."""
    first, second, _ = iterations
    if iteration < first:
        fraction = iteration / first
        return (1 - fraction) * _MID_NEAR_START_WEIGHT + fraction * 3.0, 2.0, 1.0
    if iteration < first + second:
        return 3.0, 3.0, 1.0
    return 0.0, 1.0, 1.0


def initial_layout(values: np.ndarray) -> np.ndarray:
    """Center an initialization and scale it to a standard deviation of 1e-4.

    One scale for every dimension keeps the initialization's proportions.
    Adam moves each cell by about the learning rate in its first step, so the
    layout keeps the initialization's arrangement and not its size.

    Raises:
        ValueError: If the values are not finite or every cell has the same
            coordinates.
    """
    layout = np.array(values, dtype=np.float64, order="C", copy=True)
    if layout.ndim != 2:
        raise ValueError("An initialization must be a matrix of cells by dimensions")
    if not np.all(np.isfinite(layout)):
        raise ValueError("An initialization must hold only finite values")
    layout -= layout.mean(axis=0)
    spread = float(np.sqrt(np.mean(layout * layout)))
    if spread == 0.0:
        raise ValueError(
            "An initialization must place the cells apart; every cell has the "
            "same coordinates"
        )
    return np.ascontiguousarray(layout * (_INITIAL_SCALE / spread), dtype=np.float32)


def localmap_layout(
    initialization: np.ndarray,
    graph: csr_matrix,
    mid_near: np.ndarray,
    further: np.ndarray,
    *,
    iterations: tuple[int, int, int] = LOCALMAP_ITERATIONS,
    learning_rate: float = 1.0,
    max_distance: float = 10.0,
    random_seed: int = 0,
    nthreads: int = 1,
    verbose: bool = False,
    on_iteration: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Optimize a LocalMAP layout from its initialization, graph and pairs.

    Args:
        initialization: Cells by dimensions, as :func:`initial_layout` scales
            it. It is copied, not changed.
        graph: Cells by cells, the weight of each directed neighbor edge.
        mid_near: Cells by mid-near pairs, from :func:`mid_near_pairs`.
        further: Cells by further pairs, from :func:`further_pairs`.
        iterations: Iterations of the three phases.
        learning_rate: Adam's learning rate.
        max_distance: The layout distance within which the third phase draws
            further pairs again, and the scale of its neighbor attraction.
        random_seed: Seed of the third phase's draws.
        nthreads: Threads of the parallel kernels; the layout does not depend
            on it.
        verbose: Whether to show a progress bar.
        on_iteration: Called with each iteration's index after its update.

    Returns:
        The float32 layout, cells by dimensions.

    Raises:
        ValueError: If the pairs do not have one row per cell, or the graph
            is not one :func:`neighbor_partners` accepts.
    """
    embedding = np.array(initialization, dtype=np.float32, order="C", copy=True)
    if embedding.ndim != 2:
        raise ValueError("The initialization must be a matrix of cells by dimensions")
    n_cells = embedding.shape[0]
    for name, pairs in (("mid_near", mid_near), ("further", further)):
        if pairs.ndim != 2 or pairs.shape[0] != n_cells:
            raise ValueError(f"{name} must hold one row of pairs per cell")
    if graph.shape[0] != n_cells:
        raise ValueError("The graph and the initialization must have the same cells")
    matrix = _checked_graph(graph)
    further_now = np.array(further, dtype=np.int32, order="C", copy=True)
    first, second, third = iterations
    total = first + second + third
    gradient = np.zeros_like(embedding)
    first_moment = np.zeros_like(embedding)
    second_moment = np.zeros_like(embedding)
    max_distance_squared = float(max_distance) ** 2
    with numba_threads(nthreads):
        n_chunks = _chunks(n_cells)
        neighbors = neighbor_partners(matrix)
        mid_near_csr = pair_partners(mid_near)
        further_csr = pair_partners(further_now)
        for iteration in iter_progress(
            range(total), desc="Training LocalMAP", total=total, disable=not verbose
        ):
            w_mid_near, w_neighbors, w_further = phase_weights(iteration, iterations)
            local = iteration > first + second
            _gradient(
                embedding,
                *neighbors,
                *mid_near_csr,
                *further_csr,
                w_neighbors,
                w_mid_near,
                w_further,
                max_distance / 2.0 if local else 0.0,
                gradient,
                n_chunks,
            )
            step = (
                learning_rate
                * math.sqrt(1.0 - _BETA2 ** (iteration + 1))
                / (1.0 - _BETA1 ** (iteration + 1))
            )
            _adam_step(embedding, gradient, first_moment, second_moment, step)
            if local and iteration % _REDRAW_INTERVAL == 0:
                _redraw_nearby_further_pairs(
                    embedding,
                    matrix.indptr,
                    matrix.indices,
                    further_now,
                    draw_key(random_seed, _NEARBY_DRAWS, iteration),
                    max_distance_squared,
                    n_chunks,
                )
                further_csr = pair_partners(further_now)
            if on_iteration is not None:
                on_iteration(iteration)
    return embedding
