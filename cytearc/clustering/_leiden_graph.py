"""Graph preparation for CyteArc's native Leiden backend.

The Leiden algorithm of Traag, Waltman and van Eck (2019) optimises
modularity on an undirected weighted graph. CyteArc reads its input as
follows. Every stored entry ``(i, j, w)`` of the canonical matrix ``W``
(duplicates summed) is one undirected edge of weight ``w``, so the effective
symmetric adjacency is ``A = W + W^T`` including the diagonal: a stored
self-loop ``W_ii = w`` gives ``A_ii = 2w`` and adds ``2w`` to the strength of
node ``i``. Aggregating ``A`` by a partition then keeps exactly the same
convention (``A' = S^T A S``).

``build_leiden_graph`` validates the input, merges the sorted rows of ``W``
and ``W^T`` into one canonical symmetric CSR matrix (float64 weights), scales
all weights by a power of two so that the mean off-diagonal weight lies in
``[1, 2)``, and stores node strengths as int64 fixed-point numbers, so that
community totals are exact integer sums whatever order they are added in.
Scaling by a power of two is exact, so it changes no decision.

Reference:
    Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
    well-connected communities. Sci Rep 9, 5233 (2019).
    doi:10.1038/s41598-019-41695-z
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange
from scipy import sparse

from ._leiden_primitives import (
    FLOAT_SUM_BLOCK,
    PARALLEL_MIN_ITEMS,
    WORKER_BLOCK,
    exclusive_scan,
    fixed_block_sum,
)

# Node ids are int32 and community ids range over [0, 2n), so n < 2**30.
MAX_NODES = 2**30 - 1

# Stored values are validated in chunks to bound the temporary memory.
_VALIDATION_CHUNK = 1 << 22


@dataclass(frozen=True, slots=True)
class LeidenGraph:
    """Symmetric graph and fixed-point strengths of one native Leiden run.

    Attributes:
        indptr: int64 row offsets of the symmetric CSR matrix ``A``.
        indices: int32 column indices, ascending within each row.
        weights: float64 weights of ``A`` scaled by ``2**-weight_exponent``.
        strengths: int64 fixed-point strengths ``rint(k * strength_scale)``.
        strength_scale: The power of two ``s`` of the fixed-point strengths.
        total_strength: Exact sum of ``strengths`` (``2m`` in fixed point).
        mean_weight: Mean scaled off-diagonal weight, in ``[1, 2)``.
        weight_exponent: The exponent ``E`` of the weight scaling.
        has_offdiagonal: Whether ``A`` has an off-diagonal entry.
        max_degree: The longest row of ``A``.
    """

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    strengths: np.ndarray
    strength_scale: float
    total_strength: int
    mean_weight: float
    weight_exponent: int
    has_offdiagonal: bool
    max_degree: int

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.indptr.shape[0] - 1)


def _validated_stored_values(graph: object) -> Any:
    """Validate the graph and return an object whose ``data`` holds its values.

    The type, shape and dtype checks come first, so a huge empty matrix fails
    before any conversion. Stored values are checked before duplicates are
    summed, so a negative value is an error even when a duplicate cancels it.
    """
    if not sparse.issparse(graph):
        raise TypeError("Leiden graph must be a SciPy sparse matrix or array")
    matrix: Any = graph
    shape = tuple(matrix.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Leiden graph must be a square matrix")
    if shape[0] > MAX_NODES:
        raise ValueError(
            "Leiden graph has too many nodes for the 'cytearc' backend "
            f"(at most {MAX_NODES})"
        )
    if np.dtype(matrix.dtype).kind not in "biuf":
        raise TypeError("Leiden graph weights must be real numbers")
    source = matrix if matrix.format in {"csr", "csc", "coo", "bsr"} else matrix.tocoo()
    values = np.asarray(source.data).reshape(-1)
    kind = values.dtype.kind
    non_finite = False
    negative = False
    for low in range(0, values.size, _VALIDATION_CHUNK):
        chunk = values[low : low + _VALIDATION_CHUNK]
        if kind == "f" and not bool(np.isfinite(chunk).all()):
            non_finite = True
        if kind in "if" and bool((chunk < 0).any()):
            negative = True
    if non_finite:
        raise ValueError("Leiden graph weights must be finite")
    if negative:
        raise ValueError("Leiden graph weights must be non-negative")
    return source


def _canonical_arrays(
    graph: Any, source: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the CSR arrays of the canonical ``W`` and of ``W^T``.

    Canonical means sorted column indices and no duplicate entries. A
    canonical float32 or float64 CSR (or CSC) input is used without a copy.
    Every other input is converted to float64 before duplicates are summed,
    so duplicate sums are float64 sums for every input dtype. The caller's
    object is never modified.
    """
    native_dtype = source.dtype in (np.dtype(np.float32), np.dtype(np.float64))
    if graph.format == "csr" and native_dtype and graph.has_canonical_format:
        w_matrix = graph
        t_matrix = graph.T.tocsr()
        t_arrays = (t_matrix.indptr, t_matrix.indices, t_matrix.data)
    elif graph.format == "csc" and native_dtype and graph.has_canonical_format:
        w_matrix = graph.tocsr()
        t_arrays = (graph.indptr, graph.indices, graph.data)
    else:
        if source.dtype == np.dtype(np.float64):
            w_matrix = source.tocsr(copy=True)
        else:
            w_matrix = source.astype(np.float64).tocsr()
        if not w_matrix.has_canonical_format:
            w_matrix.sum_duplicates()
        t_matrix = w_matrix.T.tocsr()
        t_arrays = (t_matrix.indptr, t_matrix.indices, t_matrix.data)
    arrays = [w_matrix.indptr, w_matrix.indices, w_matrix.data, *t_arrays]
    index_dtype = (
        np.int32
        if all(np.dtype(arrays[i].dtype) == np.dtype(np.int32) for i in (0, 1, 3, 4))
        else np.int64
    )
    for i in (0, 1, 3, 4):
        arrays[i] = np.ascontiguousarray(arrays[i], dtype=index_dtype)
    for i in (2, 5):
        arrays[i] = np.ascontiguousarray(arrays[i])
    return arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]


@njit(cache=True, nogil=True, parallel=True)
def _symmetric_row_lengths(
    n: int,
    w_indptr: np.ndarray,
    w_indices: np.ndarray,
    w_data: np.ndarray,
    t_indptr: np.ndarray,
    t_indices: np.ndarray,
    t_data: np.ndarray,
    n_workers: int,
) -> np.ndarray:
    """Count the non-zero entries of every row of ``W + W^T``."""
    lengths = np.zeros(n, dtype=np.int64)
    n_blocks = (n + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            for row in range(block * WORKER_BLOCK, min(n, (block + 1) * WORKER_BLOCK)):
                w_pos = w_indptr[row]
                w_end = w_indptr[row + 1]
                t_pos = t_indptr[row]
                t_end = t_indptr[row + 1]
                count = 0
                while w_pos < w_end or t_pos < t_end:
                    if t_pos >= t_end or (
                        w_pos < w_end and w_indices[w_pos] < t_indices[t_pos]
                    ):
                        value = np.float64(w_data[w_pos])
                        w_pos += 1
                    elif w_pos >= w_end or t_indices[t_pos] < w_indices[w_pos]:
                        value = np.float64(t_data[t_pos])
                        t_pos += 1
                    else:
                        value = np.float64(w_data[w_pos]) + np.float64(t_data[t_pos])
                        w_pos += 1
                        t_pos += 1
                    if value != 0.0:
                        count += 1
                lengths[row] = count
    return lengths


@njit(cache=True, nogil=True, parallel=True)
def _symmetric_fill(
    n: int,
    w_indptr: np.ndarray,
    w_indices: np.ndarray,
    w_data: np.ndarray,
    t_indptr: np.ndarray,
    t_indices: np.ndarray,
    t_data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    n_workers: int,
) -> None:
    """Write the non-zero entries of ``W + W^T``, columns ascending.

    An entry present in both ``W`` and ``W^T`` is one float64 addition, which
    is commutative, so the result is a canonical function of ``W``.
    """
    n_blocks = (n + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            for row in range(block * WORKER_BLOCK, min(n, (block + 1) * WORKER_BLOCK)):
                w_pos = w_indptr[row]
                w_end = w_indptr[row + 1]
                t_pos = t_indptr[row]
                t_end = t_indptr[row + 1]
                out = indptr[row]
                while w_pos < w_end or t_pos < t_end:
                    if t_pos >= t_end or (
                        w_pos < w_end and w_indices[w_pos] < t_indices[t_pos]
                    ):
                        column = w_indices[w_pos]
                        value = np.float64(w_data[w_pos])
                        w_pos += 1
                    elif w_pos >= w_end or t_indices[t_pos] < w_indices[w_pos]:
                        column = t_indices[t_pos]
                        value = np.float64(t_data[t_pos])
                        t_pos += 1
                    else:
                        column = w_indices[w_pos]
                        value = np.float64(w_data[w_pos]) + np.float64(t_data[t_pos])
                        w_pos += 1
                        t_pos += 1
                    if value != 0.0:
                        indices[out] = column
                        weights[out] = value
                        out += 1


@njit(cache=True, nogil=True, parallel=True)
def _offdiagonal_sum(
    indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, n_workers: int
) -> tuple[float, int]:
    """Return the sum and count of the off-diagonal entries.

    The sum runs over fixed blocks of ``FLOAT_SUM_BLOCK`` consecutive stored
    entries in CSR order, each block in order, then over the block sums.
    """
    n_entries = indptr[indptr.shape[0] - 1]
    n_blocks = (n_entries + FLOAT_SUM_BLOCK - 1) // FLOAT_SUM_BLOCK
    block_sums = np.zeros(n_blocks, dtype=np.float64)
    block_counts = np.zeros(n_blocks, dtype=np.int64)
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            low = block * FLOAT_SUM_BLOCK
            high = min(n_entries, low + FLOAT_SUM_BLOCK)
            row = np.searchsorted(indptr, low, side="right") - 1
            block_total = 0.0
            block_count = 0
            for position in range(low, high):
                while indptr[row + 1] <= position:
                    row += 1
                if indices[position] != row:
                    block_total += weights[position]
                    block_count += 1
            block_sums[block] = block_total
            block_counts[block] = block_count
    total = 0.0
    count = 0
    for summed in range(n_blocks):
        total += block_sums[summed]
        count += block_counts[summed]
    return total, count


@njit(cache=True, nogil=True, parallel=True)
def _row_sums(indptr: np.ndarray, weights: np.ndarray, n_workers: int) -> np.ndarray:
    """Return the strength of every node, each row summed in CSR order."""
    n = indptr.shape[0] - 1
    sums = np.empty(n, dtype=np.float64)
    n_blocks = (n + WORKER_BLOCK - 1) // WORKER_BLOCK
    for worker in prange(n_workers):
        for block in range(worker, n_blocks, n_workers):
            for row in range(block * WORKER_BLOCK, min(n, (block + 1) * WORKER_BLOCK)):
                row_total = 0.0
                for position in range(indptr[row], indptr[row + 1]):
                    row_total += weights[position]
                sums[row] = row_total
    return sums


def build_leiden_graph(
    graph: object, n_workers: int, parallel_min_items: int = PARALLEL_MIN_ITEMS
) -> LeidenGraph:
    """Validate ``graph`` and build the symmetric graph of a native Leiden run.

    Args:
        graph: Square SciPy sparse matrix or array with non-negative, finite,
            real weights. It is never modified.
        n_workers: Number of worker threads of the parallel kernels.
        parallel_min_items: Inputs smaller than this use single-thread loops
            (results are identical).

    Returns:
        The symmetric CSR matrix ``A = W + W^T`` with normalised weights and
        fixed-point node strengths.

    Raises:
        TypeError: If ``graph`` is not sparse or its weights are not real.
        ValueError: If ``graph`` is not square, has too many nodes, or has a
            negative, NaN or infinite weight, or if the symmetrised total
            weight overflows.
    """
    source = _validated_stored_values(graph)
    matrix: Any = graph
    n = int(matrix.shape[0])
    w_indptr, w_indices, w_data, t_indptr, t_indices, t_data = _canonical_arrays(
        matrix, source
    )
    del source
    lengths = _symmetric_row_lengths(
        n, w_indptr, w_indices, w_data, t_indptr, t_indices, t_data, n_workers
    )
    indptr = exclusive_scan(lengths, n_workers, parallel_min_items)
    max_degree = int(lengths.max()) if n else 0
    del lengths
    n_entries = int(indptr[n])
    indices = np.empty(n_entries, dtype=np.int32)
    weights = np.empty(n_entries, dtype=np.float64)
    _symmetric_fill(
        n,
        w_indptr,
        w_indices,
        w_data,
        t_indptr,
        t_indices,
        t_data,
        indptr,
        indices,
        weights,
        n_workers,
    )
    del w_indptr, w_indices, w_data, t_indptr, t_indices, t_data
    too_large = (
        "Leiden graph weights are too large: their symmetrized total is not finite"
    )
    if not math.isfinite(fixed_block_sum(weights, n_workers, parallel_min_items)):
        raise ValueError(too_large)
    offdiagonal_total, offdiagonal_count = _offdiagonal_sum(
        indptr, indices, weights, n_workers
    )
    if offdiagonal_count == 0:
        return LeidenGraph(
            indptr=indptr,
            indices=indices,
            weights=weights,
            strengths=np.zeros(n, dtype=np.int64),
            strength_scale=1.0,
            total_strength=0,
            mean_weight=0.0,
            weight_exponent=0,
            has_offdiagonal=False,
            max_degree=max_degree,
        )
    raw_mean = float(offdiagonal_total) / int(offdiagonal_count)
    weight_exponent = math.frexp(raw_mean)[1] - 1
    np.ldexp(weights, -weight_exponent, out=weights)
    mean_weight = math.ldexp(raw_mean, -weight_exponent)
    strengths_float = _row_sums(indptr, weights, n_workers)
    two_m = fixed_block_sum(strengths_float, n_workers, parallel_min_items)
    if not math.isfinite(two_m):
        raise ValueError(too_large)
    mantissa, exponent = math.frexp(two_m)
    ceil_log2 = exponent - 1 if mantissa == 0.5 else exponent
    scale_exponent = 61 - ceil_log2
    strengths = np.rint(np.ldexp(strengths_float, scale_exponent)).astype(np.int64)
    del strengths_float
    return LeidenGraph(
        indptr=indptr,
        indices=indices,
        weights=weights,
        strengths=strengths,
        strength_scale=math.ldexp(1.0, scale_exponent),
        total_strength=int(strengths.sum()),
        mean_weight=mean_weight,
        weight_exponent=weight_exponent,
        has_offdiagonal=True,
        max_degree=max_degree,
    )
