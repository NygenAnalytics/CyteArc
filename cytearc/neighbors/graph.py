from dataclasses import dataclass
from numbers import Real

import numpy as np
from numba import jit, njit, prange
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arguments import integer_argument
from ..utils.numba import enable_dispatcher_cache
from ..utils.progress import iter_progress

# Python objects a graph load or a connectivity conversion holds whatever the
# graph size, such as the sparse-matrix wrappers, measured below 64 KiB.
_GRAPH_OBJECT_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class GraphLoadMemory:
    """Bytes of loading a stored graph as a SciPy CSR matrix.

    Attributes:
        peakBytes: The most bytes the load holds.
        heldBytes: Bytes of the CSR matrix that the load returns.
        maxEntries: The most entries that matrix stores.
    """

    peakBytes: int
    heldBytes: int
    maxEntries: int


def _sparse_index_bytes(n_cells: int, n_entries: int) -> int:
    """Return the itemsize of SciPy's index dtype for such a sparse matrix."""
    return 4 if max(n_cells, n_entries) <= np.iinfo(np.int32).max else 8


def graph_load_memory(
    n_cells: int,
    n_edges: int,
    *,
    symmetric: bool,
    upper_only: bool = False,
    edge_itemsize: int = 4,
    weight_itemsize: int = 4,
) -> GraphLoadMemory:
    """Estimate the bytes of loading a stored graph of ``n_edges`` directed edges.

    A load reads the stored edge pairs and weights, builds a COO matrix whose
    coordinates are copies in SciPy's index dtype, and converts it to CSR.
    ``symmetric`` then forms the fuzzy union ``A + A^T - A * A^T``. SciPy sizes
    a sum or an elementwise product for the entries of both operands, so the
    union holds the directed matrix, a CSR copy of its transpose for each
    operation, a sum and a product sized for ``2 * n_edges`` entries, and a
    compact copy of the product: its result keeps the capacity of
    ``2 * n_edges`` entries. ``upper_only`` keeps the upper triangle, which
    holds at most ``n_edges`` entries, through a COO copy of the union.

    Args:
        n_cells: Cells of the graph, its rows and columns.
        n_edges: Stored directed edges.
        symmetric: Whether the load forms the fuzzy union.
        upper_only: Whether it then keeps the upper triangle.
        edge_itemsize: Bytes of one stored edge endpoint.
        weight_itemsize: Bytes of one stored weight.

    Returns:
        The estimated peak, the bytes of the loaded matrix, and its most
        entries.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    edges = integer_argument(n_edges, "n_edges", minimum=0)
    edge_bytes = integer_argument(edge_itemsize, "edge_itemsize", minimum=1)
    weight_bytes = integer_argument(weight_itemsize, "weight_itemsize", minimum=1)
    index_bytes = _sparse_index_bytes(cells, 2 * edges)
    entry_bytes = index_bytes + weight_bytes
    pointer_bytes = (cells + 1) * index_bytes
    directed_bytes = edges * entry_bytes + pointer_bytes
    # The stored pairs and weights, the COO coordinates, and the CSR matrix.
    loading = edges * (2 * edge_bytes + weight_bytes + 2 * index_bytes) + directed_bytes
    if not symmetric:
        return GraphLoadMemory(
            peakBytes=loading + _GRAPH_OBJECT_BYTES,
            heldBytes=directed_bytes,
            maxEntries=edges,
        )
    union = 7 * edges * entry_bytes + 5 * pointer_bytes
    union_bytes = 2 * edges * entry_bytes + pointer_bytes
    if not upper_only:
        return GraphLoadMemory(
            peakBytes=max(loading, union) + _GRAPH_OBJECT_BYTES,
            heldBytes=union_bytes,
            maxEntries=2 * edges,
        )
    # The union, the COO rows of its entries and their mask, the upper
    # entries, and their CSR matrix.
    upper = (
        union_bytes
        + 2 * edges * (index_bytes + 1)
        + edges * (2 * index_bytes + weight_bytes)
        + directed_bytes
    )
    return GraphLoadMemory(
        peakBytes=max(loading, union, upper) + _GRAPH_OBJECT_BYTES,
        heldBytes=directed_bytes,
        maxEntries=edges,
    )


def connectivity_peak_bytes(n_cells: int, n_neighbors: int) -> int:
    """Estimate the peak bytes of ``build_connectivity_arrays``, with its inputs.

    The conversion holds the uint32 neighbor indices and float32 distances.
    The kernel fit holds a float32 copy padded with each cell's zero distance
    and float32 bandwidths and offsets per cell; the memberships hold int32
    rows and columns and float32 weights and distances; then the uint32 edge
    pairs are filled beside the rows, columns, and weights, and checks hold
    one boolean per edge.

    Args:
        n_cells: Cells of the neighbor matrix.
        n_neighbors: Neighbors per cell.

    Returns:
        The estimated peak in bytes.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below one.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    k = integer_argument(n_neighbors, "n_neighbors", minimum=1)
    edges = cells * k
    inputs = 8 * edges
    smoothing = 4 * cells * (k + 1) + 8 * cells
    memberships = 16 * edges + 8 * cells
    pairs = 12 * edges + 8 * edges + edges
    return inputs + max(smoothing, memberships, pairs) + _GRAPH_OBJECT_BYTES


def smooth_knn_chunk(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    local_connectivity: float,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one KNN distance block into fuzzy graph edges."""
    from umap.umap_ import compute_membership_strengths, smooth_knn_dist

    # umap-learn compiles both kernels without an on-disk cache.
    enable_dispatcher_cache(smooth_knn_dist, compute_membership_strengths)
    indices_array = np.asarray(indices)
    distance_array = np.asarray(distances, dtype=np.float32)
    if not distance_array.flags.c_contiguous:
        distance_array = np.ascontiguousarray(distance_array)
    n_rows, n_neighbors = distance_array.shape
    with_self = np.zeros((n_rows, n_neighbors + 1), dtype=np.float32)
    with_self[:, 1:] = distance_array
    sigmas, rhos = smooth_knn_dist(
        with_self,
        k=float(n_neighbors + 1),
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
    )
    # Free the padded copy before the edge arrays are allocated, so it does
    # not raise the peak memory of the conversion.
    del with_self
    rows, columns, values, _ = compute_membership_strengths(
        indices_array,
        distance_array,
        sigmas,
        rhos,
    )
    return np.asarray(rows), np.asarray(columns), np.asarray(values)


def validate_connectivity_parameters(
    local_connectivity: float,
    bandwidth: float,
) -> tuple[float, float]:
    if isinstance(local_connectivity, bool) or not isinstance(
        local_connectivity,
        Real,
    ):
        raise TypeError("local_connectivity must be a real number")
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, Real):
        raise TypeError("bandwidth must be a real number")
    resolved_local_connectivity = float(local_connectivity)
    resolved_bandwidth = float(bandwidth)
    if not np.isfinite(resolved_local_connectivity) or resolved_local_connectivity < 0:
        raise ValueError("local_connectivity must be finite and non-negative")
    if not np.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0:
        raise ValueError("bandwidth must be finite and greater than zero")
    return resolved_local_connectivity, resolved_bandwidth


def build_connectivity_arrays(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    local_connectivity: float,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge and weight arrays retaining every KNN entry."""
    local_connectivity, bandwidth = validate_connectivity_parameters(
        local_connectivity,
        bandwidth,
    )
    neighbor_indices = np.asarray(indices)
    neighbor_distances = np.asarray(distances)
    if neighbor_indices.ndim != 2 or neighbor_distances.shape != neighbor_indices.shape:
        raise ValueError("Neighbor indices and distances must have matching matrices")
    if not np.issubdtype(neighbor_indices.dtype, np.integer):
        raise TypeError("Neighbor indices must be integers")
    if np.any(neighbor_indices < 0) or np.any(
        neighbor_indices >= neighbor_indices.shape[0]
    ):
        raise ValueError("Neighbor indices are outside the cell range")
    if not np.all(np.isfinite(neighbor_distances)) or np.any(neighbor_distances < 0):
        raise ValueError("Neighbor distances must be finite and non-negative")
    rows, columns, values = smooth_knn_chunk(
        neighbor_indices,
        neighbor_distances,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
    )
    expected = int(neighbor_indices.size)
    if (
        rows.shape != (expected,)
        or columns.shape != (expected,)
        or values.shape != (expected,)
    ):
        raise ValueError("UMAP membership output does not match the KNN matrix")
    if (
        np.any(rows < 0)
        or np.any(columns < 0)
        or np.any(rows > np.iinfo(np.uint32).max)
        or np.any(columns > np.iinfo(np.uint32).max)
    ):
        raise ValueError("Connectivity endpoints exceed uint32 bounds")
    edges = np.empty((expected, 2), dtype=np.uint32)
    edges[:, 0] = rows
    edges[:, 1] = columns
    weights = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Connectivity weights must be finite")
    return edges, weights


def take_nearest_per_row(
    weights: np.ndarray,
    edges: np.ndarray,
    n_cells: int,
    use_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``use_k`` nearest stored edges of every cell.

    Edges are stored row major and ordered by increasing distance inside a cell,
    so a row's leading entries are its nearest neighbors. Rows with fewer than
    ``use_k`` stored edges keep all their entries.
    """
    sources = np.asarray(edges[:, 0], dtype=np.intp)
    if np.any(np.diff(sources) < 0):
        raise ValueError("Graph edges are not grouped by source cell")
    counts = np.bincount(sources, minlength=n_cells)
    row_starts = np.repeat(np.cumsum(counts) - counts, counts)
    rank_in_row = np.arange(sources.size, dtype=np.intp) - row_starts
    keep = rank_in_row < use_k
    return weights[keep], edges[keep]


@njit(cache=True, nogil=True, parallel=True)
def _count_sorted_shared_neighbors(
    indices: np.ndarray,
    ordered: np.ndarray,
    out: np.ndarray,
) -> None:
    """Write the neighbors each cell shares with each of its neighbors to ``out``.

    ``ordered`` holds each row of ``indices`` sorted. Two sorted rows are
    merged, and a value repeated within a row counts once, as in the
    intersection of the rows as sets. Each cell is its own parallel
    iteration, so the counts do not depend on the thread count.
    """
    n_cells, n_neighbors = indices.shape
    for cell in prange(n_cells):
        for slot in range(n_neighbors):
            neighbor = indices[cell, slot]
            shared = 0
            left = 0
            right = 0
            while left < n_neighbors and right < n_neighbors:
                value = ordered[cell, left]
                other = ordered[neighbor, right]
                if value < other:
                    left += 1
                elif other < value:
                    right += 1
                else:
                    shared += 1
                    while left < n_neighbors and ordered[cell, left] == value:
                        left += 1
                    while right < n_neighbors and ordered[neighbor, right] == value:
                        right += 1
            out[cell, slot] = shared


@njit(cache=True, nogil=True)
def _count_shared_neighbors(indices: np.ndarray, out: np.ndarray) -> None:
    """Write the neighbors each cell shares with each of its neighbors to ``out``.

    A neighbor listed twice in a row counts once, as in a set intersection.
    """
    ordered = np.empty(indices.shape, dtype=indices.dtype)
    for cell in range(indices.shape[0]):
        ordered[cell] = indices[cell]
        ordered[cell].sort()
    _count_sorted_shared_neighbors(indices, ordered, out)


@jit(nopython=True, cache=True)
def calc_snn(indices: np.ndarray) -> np.ndarray:
    """Calculate shared-neighbor fractions for a KNN index matrix."""
    ncells, nk = indices.shape
    snn = np.zeros((ncells, nk))
    _count_shared_neighbors(indices, snn)
    return np.asarray(snn / (nk - 1))


def _shared_neighbor_fractions(indices: np.ndarray) -> np.ndarray:
    """Return :func:`calc_snn` of ``indices`` in one NumPy-allocated array.

    The fractions are computed in place, so merging holds one float64 matrix
    per graph, and it holds it and the sorted copy of the rows that the
    counts read as NumPy memory, which a memory trace sees.
    """
    shared = np.empty(indices.shape, dtype=np.float64)
    ordered = np.sort(indices, axis=1)
    _count_sorted_shared_neighbors(indices, ordered, shared)
    del ordered
    shared /= indices.shape[1] - 1
    return shared


@njit(cache=True, nogil=True, parallel=True)
def _merge_ranked_rows(
    candidates: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray,
    kept_columns: np.ndarray,
    kept_weights: np.ndarray,
    kept_counts: np.ndarray,
) -> None:
    """Keep the best distinct candidates of each row, as ``weight_sort_indices``.

    ``order`` sorts the ranks of each row's candidates in ascending order, as
    ``np.argsort`` does, so each row is read from its end, and a candidate is
    kept at its first and best-ranked copy until the row of ``kept_columns``
    is full. ``kept_counts`` receives how many candidates each row kept, fewer
    than the width of ``kept_columns`` when a row has fewer distinct ones.
    """
    n_rows, width = candidates.shape
    n_keep = kept_columns.shape[1]
    for row in prange(n_rows):
        kept = 0
        position = width - 1
        while position >= 0 and kept < n_keep:
            slot = order[row, position]
            column = candidates[row, slot]
            seen = False
            for previous in range(kept):
                if kept_columns[row, previous] == column:
                    seen = True
                    break
            if not seen:
                kept_columns[row, kept] = column
                kept_weights[row, kept] = weights[row, slot]
                kept += 1
            position -= 1
        kept_counts[row] = kept


# Python objects a merge holds whatever the data size, such as the per-row
# candidates and the progress bars, measured below 64 KiB.
_SNN_OBJECT_BYTES = 256 * 1024
# Rows whose ranked candidates a merge holds at once.
_MERGE_BLOCK_ROWS = 4096


def snn_merge_peak_bytes(
    n_cells: int,
    n_neighbors: int,
    n_graphs: int,
    *,
    index_itemsize: int = 4,
    weight_itemsize: int = 4,
) -> int:
    """Estimate the peak bytes of ``merge_graphs``, including its input graphs."""
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    k = integer_argument(n_neighbors, "n_neighbors", minimum=1)
    graphs = integer_argument(n_graphs, "n_graphs", minimum=1)
    index_bytes = integer_argument(index_itemsize, "index_itemsize", minimum=1)
    weight_bytes = integer_argument(weight_itemsize, "weight_itemsize", minimum=1)
    edges = cells * k
    # Each graph's indices, weights, and shared-neighbor fractions, its row
    # pointers and row counts, and the merged columns and weights.
    held = (
        edges * (graphs * (index_bytes + weight_bytes + 8) + index_bytes + weight_bytes)
        + cells * graphs * 2 * index_bytes
    )
    # A block of rows holds two float64 copies of every graph's candidate
    # ranks, or the ranks and their int64 order, or the order with the
    # candidates and their weights, beside the int64 count of the candidates
    # each row keeps.
    block_rows = min(cells, _MERGE_BLOCK_ROWS)
    merging = block_rows * (graphs * k * max(16, 8 + index_bytes + weight_bytes) + 8)
    # The int64 rows of the COO graph, with their copy in its index dtype, and
    # the int64 row numbers that they repeat.
    graph = edges * (8 + index_bytes) + cells * 8
    return held + max(merging, graph) + _SNN_OBJECT_BYTES


def weight_sort_indices(
    i: np.ndarray,
    w: np.ndarray,
    wn: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort neighbors by adjusted weight and remove duplicates."""
    idx = np.argsort(wn)[::-1]
    i = i[idx]
    w = w[idx]
    _, unique_idx = np.unique(i, return_index=True)
    unique_idx_arr = np.asarray(sorted(unique_idx))
    return i[unique_idx_arr][:n], w[unique_idx_arr][:n]


def merge_graphs(csr_mats: list[csr_matrix]) -> coo_matrix:
    """Merge regular graphs using edge weights and shared neighbors."""
    if not csr_mats:
        raise ValueError("At least one graph is required")
    if len({matrix.shape for matrix in csr_mats}) != 1:
        raise ValueError("ERROR: All graphs do not have the same shape.")
    row_counts = [np.diff(matrix.indptr) for matrix in csr_mats]
    if any(
        len(counts) == 0 or not np.all(counts == counts[0]) for counts in row_counts
    ):
        raise ValueError("ERROR: All graphs must have a regular neighbor count")
    neighbor_counts = {int(counts[0]) for counts in row_counts}
    if len(neighbor_counts) != 1:
        raise ValueError("ERROR: All graphs do not have the same number of edges")
    nk = neighbor_counts.pop()
    if nk < 2:
        raise ValueError("SNN integration requires at least two neighbors per cell")
    n_cells = csr_mats[0].shape[0]
    # Regular rows let each graph be read as one cell-by-neighbor matrix.
    neighbor_rows = [matrix.indices.reshape((n_cells, nk)) for matrix in csr_mats]
    weight_rows = [matrix.data.reshape((n_cells, nk)) for matrix in csr_mats]
    snns = [
        _shared_neighbor_fractions(neighbors)
        for neighbors in iter_progress(
            neighbor_rows,
            desc="Identifying SNNs in graphs",
        )
    ]
    columns = np.empty(
        n_cells * nk,
        dtype=np.result_type(*(neighbors.dtype for neighbors in neighbor_rows)),
    )
    data = np.empty(
        n_cells * nk,
        dtype=np.result_type(*(weights.dtype for weights in weight_rows)),
    )
    kept_counts = np.empty(min(n_cells, _MERGE_BLOCK_ROWS), dtype=np.int64)
    for start in iter_progress(
        range(0, n_cells, _MERGE_BLOCK_ROWS),
        desc="Merging graph edges",
        total=-(-n_cells // _MERGE_BLOCK_ROWS),
    ):
        stop = min(start + _MERGE_BLOCK_ROWS, n_cells)
        # Each row's candidates are its edges in every graph, in graph order.
        # A candidate ranks by its weight plus its shared-neighbor fraction,
        # and np.argsort sorts each row of a block as it sorts the row alone,
        # so tied candidates keep the order that weight_sort_indices gives them.
        ranks = np.concatenate(
            [
                graph_weights[start:stop] + fractions[start:stop]
                for graph_weights, fractions in zip(weight_rows, snns, strict=True)
            ],
            axis=1,
        )
        order = np.argsort(ranks, axis=1)
        del ranks
        counts = kept_counts[: stop - start]
        _merge_ranked_rows(
            np.concatenate(
                [neighbors[start:stop] for neighbors in neighbor_rows], axis=1
            ),
            np.concatenate(
                [graph_weights[start:stop] for graph_weights in weight_rows], axis=1
            ),
            order,
            columns[start * nk : stop * nk].reshape(stop - start, nk),
            data[start * nk : stop * nk].reshape(stop - start, nk),
            counts,
        )
        del order
        short = np.flatnonzero(counts < nk)
        if len(short):
            row = int(short[0])
            raise ValueError(
                f"Row {start + row} of the merged graphs has {int(counts[row])} "
                f"distinct neighbors, but SNN integration keeps {nk}"
            )
    rows = np.repeat(np.arange(n_cells), nk)
    return coo_matrix((data, (rows, columns)), shape=csr_mats[0].shape)


@njit(cache=True, nogil=True)
def csr_symmetry_error(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray | None,
    allow_self_loops: bool,
) -> int:
    """Return the first symmetry defect of a CSR matrix with sorted rows.

    Returns 0 for a symmetric matrix, 1 for a self-loop when
    ``allow_self_loops`` is false, 2 for an entry without its transpose, and 3
    for an entry whose transpose holds another value. Passing ``None`` as
    ``data`` checks only the sparsity structure.
    """
    for source in range(indptr.shape[0] - 1):
        for offset in range(indptr[source], indptr[source + 1]):
            target = indices[offset]
            if target == source:
                if allow_self_loops:
                    continue
                return 1
            lower = indptr[target]
            upper = indptr[target + 1]
            while lower < upper:
                middle = (lower + upper) // 2
                if indices[middle] < source:
                    lower = middle + 1
                else:
                    upper = middle
            if lower >= indptr[target + 1] or indices[lower] != source:
                return 2
            if data is not None:
                if data[lower] != data[offset]:
                    return 3
    return 0
