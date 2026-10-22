from collections.abc import Sequence
from typing import Any

import numpy as np
from numba import config, njit, prange, set_num_threads
from scipy.sparse import coo_matrix

from ..utils.arguments import integer_argument
from ..utils.logging import logger
from ..utils.numba import restore_numba_threads
from ..utils.progress import iter_progress

# Rows that the input checks of WNN integration scan at once.
_VALIDATION_ROWS = 100_000
# Coordinate values whose squared row norms are summed at once; the sum
# holds two float64 copies of them.
_NORM_BAND_VALUES = 1 << 20
# Python objects an integration holds whatever the data size, such as the
# per-cell candidate arrays and the progress bar, measured below 64 KiB.
_WNN_OBJECT_BYTES = 256 * 1024
# Cells whose candidates, distances, and predictions are scored at once.
_WNN_BLOCK_CELLS = 2048
# Coordinate dtypes that the compiled kernels read.
_KERNEL_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
# NumPy's pairwise summation adds runs of up to this many values with eight
# partial sums and splits longer runs.
_PAIRWISE_BLOCK = 128
# Halving any run of an int64 count leaves at most this many pending runs.
_PAIRWISE_DEPTH = 64
# A bandwidth at most this many nearest-neighbour distances is degenerate.
_TOLERANCE_SCALE = 8.0 * np.finfo(np.float64).eps
# The smallest stored graph weight, the smallest positive normal float32.
_TINY_WEIGHT = float(np.finfo(np.float32).tiny)


def wnn_peak_bytes(
    n_cells: int,
    neighbor_counts: Sequence[int],
    dims: Sequence[int],
    *,
    index_itemsize: int = 4,
    coordinate_itemsize: int = 4,
) -> int:
    """Estimate the most bytes that WNN integration holds, including its inputs."""
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    counts = [
        integer_argument(count, "neighbor_counts", minimum=1)
        for count in neighbor_counts
    ]
    widths = [integer_argument(width, "dims", minimum=1) for width in dims]
    if len(counts) != len(widths) or len(counts) < 2:
        raise ValueError(
            "neighbor_counts and dims must describe the same two or more modalities"
        )
    index_bytes = integer_argument(index_itemsize, "index_itemsize", minimum=1)
    value_bytes = integer_argument(
        coordinate_itemsize, "coordinate_itemsize", minimum=1
    )
    modalities = len(counts)
    edges = min(counts)
    inputs = cells * sum(
        count * index_bytes + width * value_bytes
        for count, width in zip(counts, widths, strict=True)
    )
    checked_rows = min(cells, _VALIDATION_ROWS)
    norm_rows = max(
        min(cells, _norm_band_rows(width)) * (2 * width + 1) for width in widths
    )
    # A block scores its cells in float64. Per cell, its candidates fill
    # int64 rows: the stacked neighbour rows, their sort order, the
    # candidates, and the neighbours' positions among them. Every modality's
    # distances to the candidates follow, then the offsets from each
    # modality's predictions with a row of scratch, the modality scores, and
    # the affinities that combine the distances.
    candidates = sum(counts)
    distances = 8 * modalities * candidates
    scratch = 8 * max(2 * max(widths), max(counts))
    scores = 8 * modalities * modalities
    block = min(cells, _WNN_BLOCK_CELLS) * max(
        32 * candidates + 8,
        # Each modality's own distances are gathered and sorted.
        24 * candidates + distances + max(scratch, 24 * max(counts)),
        16 * candidates
        + distances
        + 8 * modalities * max(widths)
        + scratch
        + scores
        + 32 * modalities,
        8 * candidates + distances + 5 * scores,
        32 * candidates + distances,
    )
    # The inverse norms, the float32 modality weights, and the uint32 columns
    # and float32 weights of the graph.
    held = cells * (8 * modalities + 4 * modalities + 8 * edges)
    phases = (
        # Each row band of neighbor checks holds the row numbers, a sorted
        # copy of the band, and its comparisons.
        checked_rows * (8 + max(counts) * (index_bytes + 2)),
        # The finite check of a band of coordinates.
        checked_rows * max(widths),
        # The float64 inverse norms of every modality, the squared norms of
        # the last one, and the float64 copies of one band of its rows.
        cells * 8 * (modalities + 1) + 8 * norm_rows,
        held + block,
        # As the graph is built, its uint32 row indices and the int32 copies
        # of its rows and columns.
        held + cells * 12 * edges,
    )
    return inputs + max(phases) + _WNN_OBJECT_BYTES


def _validate_neighbor_indices(
    name: str,
    values: np.ndarray,
    expected_cells: int | None = None,
) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 2 or indices.shape[0] == 0:
        raise ValueError(f"WNN neighbors for {name} must be a non-empty matrix")
    if indices.shape[1] < 2:
        raise ValueError(
            f"WNN neighbors for {name} must contain at least two neighbors per cell"
        )
    if expected_cells is not None and indices.shape[0] != expected_cells:
        raise ValueError("WNN neighbor matrices must have the same number of cells")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"WNN neighbors for {name} must contain integer indices")

    n_cells = indices.shape[0]
    if int(indices.min()) < 0 or int(indices.max()) >= n_cells:
        raise ValueError(f"WNN neighbors for {name} contain indices outside cell range")
    for start in range(0, n_cells, 100_000):
        stop = min(start + 100_000, n_cells)
        block = indices[start:stop]
        rows = np.arange(start, stop)[:, np.newaxis]
        if np.any(block == rows):
            raise ValueError(f"WNN neighbors for {name} must exclude self")
        ordered = np.sort(block, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError(f"WNN neighbors for {name} must be unique within each row")
    return indices


def _validate_embedding(
    name: str,
    values: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    embedding = np.asarray(values)
    if embedding.ndim != 2 or embedding.shape[1] == 0:
        raise ValueError(f"WNN embedding for {name} must be a non-empty matrix")
    if embedding.shape[0] != n_cells:
        raise ValueError(f"WNN embedding for {name} must have one row per graph cell")
    if not np.issubdtype(embedding.dtype, np.number) or np.issubdtype(
        embedding.dtype,
        np.complexfloating,
    ):
        raise TypeError(f"WNN embedding for {name} must contain real numeric values")
    if not np.issubdtype(embedding.dtype, np.floating):
        embedding = embedding.astype(np.float64)
    for start in range(0, n_cells, 100_000):
        if not np.all(np.isfinite(embedding[start : start + 100_000])):
            raise ValueError(f"WNN embedding for {name} contains non-finite values")
    return embedding


def _norm_band_rows(dims: int) -> int:
    """Return the rows of a band whose squared norms are summed at once."""
    return max(1, _NORM_BAND_VALUES // max(1, int(dims)))


def _inverse_row_norms(values: np.ndarray) -> np.ndarray:
    """Return the inverse float64 L2 norm of each row, zero for a zero row.

    The squared norms are summed in float64 over bands of rows, because the
    sum casts both of its operands to float64 copies; each row's sum does not
    depend on the band.
    """
    norms = np.empty(values.shape[0], dtype=np.float64)
    band = _norm_band_rows(values.shape[1])
    for start in range(0, values.shape[0], band):
        rows = values[start : start + band]
        if not np.can_cast(rows.dtype, np.float64):
            # NumPy 2.1's einsum refuses to cast wider floats, such as long
            # double, to float64; later releases cast them as this does.
            rows = rows.astype(np.float64)
        norms[start : start + band] = np.einsum(
            "ij,ij->i",
            rows,
            rows,
            dtype=np.float64,
            optimize=True,
        )
    np.sqrt(norms, out=norms)
    inverse = np.zeros(norms.shape, dtype=np.float64)
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    return inverse


def _kernel_affinity(
    distances: np.ndarray,
    nearest_distance: float,
    bandwidth: float,
) -> np.ndarray:
    adjusted = np.maximum(distances - nearest_distance, 0.0)
    # A bandwidth this close to zero is indistinguishable from rounding noise in
    # the distance reduction, so the tolerance tracks the local distance scale
    # to keep the result invariant to a rescaling of the embedding.
    tolerance = 8.0 * np.finfo(np.float64).eps * nearest_distance
    if bandwidth <= tolerance:
        return (adjusted <= tolerance).astype(np.float64)
    return np.exp(-(adjusted / bandwidth))


def _prediction_affinity(
    point: np.ndarray,
    estimate: np.ndarray,
    nearest_distance: float,
    bandwidth: float,
) -> float:
    estimate_distance = float(np.linalg.norm(point - estimate))
    return float(
        _kernel_affinity(
            np.asarray([estimate_distance]),
            nearest_distance,
            bandwidth,
        )[0]
    )


@njit(cache=True, nogil=True)
def _pairwise_run(
    values: np.ndarray,
    start: int,
    count: int,
    zero: np.floating[Any],
) -> np.floating[Any]:
    """Sum at most 128 values from ``start`` as NumPy's ``add.reduce`` does.

    NumPy adds fewer than eight values in turn, from ``-zero``, and more with
    eight interleaved partial sums. ``zero`` sets the precision of the sum.
    """
    total: np.floating[Any] = -zero
    if count < 8:
        for index in range(start, start + count):
            total += values[index]
        return total
    sum0 = values[start]
    sum1 = values[start + 1]
    sum2 = values[start + 2]
    sum3 = values[start + 3]
    sum4 = values[start + 4]
    sum5 = values[start + 5]
    sum6 = values[start + 6]
    sum7 = values[start + 7]
    index = start + 8
    stop = start + count - count % 8
    while index < stop:
        sum0 += values[index]
        sum1 += values[index + 1]
        sum2 += values[index + 2]
        sum3 += values[index + 3]
        sum4 += values[index + 4]
        sum5 += values[index + 5]
        sum6 += values[index + 6]
        sum7 += values[index + 7]
        index += 8
    total = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7))
    while index < start + count:
        total += values[index]
        index += 1
    return total


@njit(cache=True, nogil=True)
def _pairwise_sum(
    values: np.ndarray,
    start: int,
    count: int,
    zero: np.floating[Any],
) -> np.floating[Any]:
    """Sum ``count`` values from ``start`` as NumPy's ``add.reduce`` does.

    NumPy sums a run of more than 128 values as the sum of its two halves,
    split at a multiple of eight. A stack of the runs whose left half is
    summed stands in for that recursion, which Numba cannot cache.
    """
    if count <= _PAIRWISE_BLOCK:
        return _pairwise_run(values, start, count, zero)
    starts = np.empty(_PAIRWISE_DEPTH, dtype=np.int64)
    counts = np.empty(_PAIRWISE_DEPTH, dtype=np.int64)
    lefts = np.full(_PAIRWISE_DEPTH, zero)
    split = np.zeros(_PAIRWISE_DEPTH, dtype=np.bool_)
    top = 0
    starts[0] = start
    counts[0] = count
    while True:
        # Descend to the first run of the newest subtree.
        while counts[top] > _PAIRWISE_BLOCK:
            half = counts[top] // 2
            half -= half % 8
            split[top] = False
            starts[top + 1] = starts[top]
            counts[top + 1] = half
            top += 1
        total = _pairwise_run(values, starts[top], counts[top], zero)
        top -= 1
        # A summed right half completes its run.
        while top >= 0 and split[top]:
            total = lefts[top] + total
            top -= 1
        if top < 0:
            return total
        # A summed left half leaves its run's right half to sum.
        lefts[top] = total
        split[top] = True
        half = counts[top] // 2
        half -= half % 8
        starts[top + 1] = starts[top] + half
        counts[top + 1] = counts[top] - half
        top += 1


@njit(cache=True, nogil=True, parallel=True)
def _candidate_pools(
    stacked: np.ndarray,
    order: np.ndarray,
    pools: np.ndarray,
    sizes: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Write each cell's candidates and where its neighbours sit among them.

    ``stacked`` holds each cell's neighbour rows side by side, and ``order``
    the order that sorts each row. A row of ``pools`` receives the sorted
    distinct values of the row, the union of the cell's neighbour rows, and
    ``sizes`` their count; ``positions`` receives the position of every
    stacked neighbour among them. Entries of ``pools`` past a row's size are
    not written.
    """
    n_rows, width = stacked.shape
    for row in prange(n_rows):
        size = 0
        for rank in range(width):
            column = order[row, rank]
            value = stacked[row, column]
            if size == 0 or value != pools[row, size - 1]:
                pools[row, size] = value
                size += 1
            positions[row, column] = size - 1
        sizes[row] = size


@njit(cache=True, nogil=True, parallel=True)
def _candidate_distances(
    embedding: np.ndarray,
    inverse: np.ndarray | None,
    first_cell: int,
    pools: np.ndarray,
    sizes: np.ndarray,
    zero: np.floating[Any],
    scratch: np.ndarray,
    distances: np.ndarray,
) -> None:
    """Write the distance from each cell of a block to each of its candidates.

    Row ``i`` of the block is cell ``first_cell + i``. Coordinates are scaled
    by ``inverse`` in float64 unless it is None, and the squared differences
    of a pair are summed in the precision of ``zero`` as ``np.linalg.norm``
    sums a row. Entries past a row's size are ``zero``. Each row of
    ``scratch`` holds one cell's coordinates and then its squared differences
    from a candidate.
    """
    n_rows, width = pools.shape
    dims = embedding.shape[1]
    for row in prange(n_rows):
        cell = first_cell + row
        values = scratch[row]
        for axis in range(dims):
            if inverse is None:
                values[axis] = embedding[cell, axis]
            else:
                values[axis] = np.float64(embedding[cell, axis]) * inverse[cell]
        size = sizes[row]
        for slot in range(size):
            candidate = pools[row, slot]
            for axis in range(dims):
                if inverse is None:
                    difference = values[axis] - embedding[candidate, axis]
                else:
                    difference = values[axis] - (
                        np.float64(embedding[candidate, axis]) * inverse[candidate]
                    )
                values[dims + axis] = difference * difference
            distances[row, slot] = np.sqrt(
                zero + _pairwise_sum(values, dims, dims, zero)
            )
        for slot in range(size, width):
            distances[row, slot] = zero


@njit(cache=True, nogil=True, parallel=True)
def _prediction_offsets(
    embedding: np.ndarray,
    inverse: np.ndarray | None,
    first_cell: int,
    stacked: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    zero: np.floating[Any],
    scratch: np.ndarray,
    offsets: np.ndarray,
) -> None:
    """Write each cell's offset from the mean of every modality's neighbours.

    ``offsets[i, source]`` is cell ``first_cell + i`` minus the mean of the
    neighbours that columns ``starts[source]`` onward of ``stacked`` list for
    modality ``source``, both in this embedding's coordinates, scaled by
    ``inverse`` unless it is None. As ``ndarray.mean`` does, each coordinate
    adds the neighbours in their listed order, or pairwise when there is a
    single coordinate, in the precision of ``zero``, and the mean is rounded
    to it before it is subtracted. Each row of ``scratch`` holds one cell's
    sums.
    """
    n_rows = stacked.shape[0]
    n_sources = starts.shape[0]
    dims = embedding.shape[1]
    for row in prange(n_rows):
        cell = first_cell + row
        values = scratch[row]
        for source in range(n_sources):
            start = starts[source]
            count = counts[source]
            if dims == 1:
                for neighbor in range(count):
                    member = stacked[row, start + neighbor]
                    if inverse is None:
                        values[neighbor] = embedding[member, 0]
                    else:
                        values[neighbor] = (
                            np.float64(embedding[member, 0]) * inverse[member]
                        )
                values[0] = (zero + _pairwise_sum(values, 0, count, zero)) / count
            else:
                for axis in range(dims):
                    values[axis] = zero
                for neighbor in range(count):
                    member = stacked[row, start + neighbor]
                    for axis in range(dims):
                        if inverse is None:
                            values[axis] += embedding[member, axis]
                        else:
                            values[axis] += (
                                np.float64(embedding[member, axis]) * inverse[member]
                            )
                for axis in range(dims):
                    values[axis] = values[axis] / count
            for axis in range(dims):
                if inverse is None:
                    offsets[row, source, axis] = embedding[cell, axis] - values[axis]
                else:
                    offsets[row, source, axis] = (
                        np.float64(embedding[cell, axis]) * inverse[cell] - values[axis]
                    )


@njit(cache=True, nogil=True)
def _precedes(value: float, slot: int, other: float, other_slot: int) -> bool:
    """Whether a candidate ranks before another one.

    Higher affinities rank first and NaN ranks last; ties keep the order of
    the candidates, as ``np.lexsort((candidates, -affinity))`` sorts them.
    """
    if value != value:
        return other != other and slot < other_slot
    if other != other:
        return True
    if value != other:
        return value > other
    return slot < other_slot


@njit(cache=True, nogil=True, parallel=True)
def _select_candidates(
    combined: np.ndarray,
    pools: np.ndarray,
    sizes: np.ndarray,
    ranking: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Keep each cell's candidates of highest combined affinity.

    A row of ``columns`` receives as many candidates as it has entries, in
    rank order, and ``weights`` their affinities clipped to between the
    smallest positive float32 and one, as ``np.clip`` and ``np.maximum`` clip
    them; NaN stays NaN. Each row of ``ranking`` holds one cell's best
    candidates while they are found.
    """
    n_rows = combined.shape[0]
    n_keep = columns.shape[1]
    for row in prange(n_rows):
        best = ranking[row]
        kept = 0
        for slot in range(sizes[row]):
            value = combined[row, slot]
            if kept == n_keep and not _precedes(
                value, slot, combined[row, best[kept - 1]], best[kept - 1]
            ):
                continue
            position = min(kept, n_keep - 1)
            while position > 0 and _precedes(
                value, slot, combined[row, best[position - 1]], best[position - 1]
            ):
                best[position] = best[position - 1]
                position -= 1
            best[position] = slot
            kept = min(kept + 1, n_keep)
        for rank in range(n_keep):
            slot = best[rank]
            value = combined[row, slot]
            if value == value:
                if value < 0.0:
                    value = 0.0
                elif value > 1.0:
                    value = 1.0
                if value < _TINY_WEIGHT:
                    value = _TINY_WEIGHT
            columns[row, rank] = pools[row, slot]
            weights[row, rank] = value


def _affinities(
    values: np.ndarray,
    nearest: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    """Return ``_kernel_affinity`` of each row of ``values`` in float64.

    Row ``i`` decays from ``nearest[i]`` over ``bandwidth[i]``, both in the
    precision of ``values``, so each entry is computed as the kernel of one
    cell computes it. Operations run in place where they can, which changes
    no value.
    """
    adjusted = values - nearest[:, np.newaxis]
    np.maximum(adjusted, 0.0, out=adjusted)
    tolerance = _TOLERANCE_SCALE * nearest.astype(np.float64)
    degenerate = bandwidth.astype(np.float64) <= tolerance
    affinity = np.empty(values.shape, dtype=np.float64)
    affinity[degenerate] = adjusted[degenerate] <= tolerance[degenerate, np.newaxis]
    decaying = ~degenerate
    exponents = adjusted[decaying]
    del adjusted
    np.divide(exponents, bandwidth[decaying, np.newaxis], out=exponents)
    np.negative(exponents, out=exponents)
    affinity[decaying] = np.exp(exponents)
    return affinity


def _integrate_blocks(
    indices: list[np.ndarray],
    embeddings: list[np.ndarray],
    inverse_norms: list[np.ndarray | None],
    columns: np.ndarray,
    data: np.ndarray,
    modality_weights: np.ndarray,
) -> None:
    """Score blocks of cells with the compiled kernels.

    Each block computes what the per-cell loop of ``_integrate_cells``
    computes, value for value: the kernels repeat NumPy's arithmetic and
    summation order, and the exponentials and the dot products of the
    predictions are NumPy's own, which the CPU's instruction set may change
    in the last bit.
    """
    n_cells = indices[0].shape[0]
    for start in iter_progress(
        range(0, n_cells, _WNN_BLOCK_CELLS),
        desc="Building WNN graph",
        total=-(-n_cells // _WNN_BLOCK_CELLS),
    ):
        stop = min(start + _WNN_BLOCK_CELLS, n_cells)
        _score_block(
            indices,
            embeddings,
            inverse_norms,
            start,
            columns[start:stop],
            data[start:stop],
            modality_weights[start:stop],
        )


def _score_block(
    indices: list[np.ndarray],
    embeddings: list[np.ndarray],
    inverse_norms: list[np.ndarray | None],
    start: int,
    columns: np.ndarray,
    data: np.ndarray,
    modality_weights: np.ndarray,
) -> None:
    """Score the cells of one block, from cell ``start`` on.

    The rows of ``columns``, ``data``, and ``modality_weights`` are the
    block's cells; every array the block allocates is released on return.
    """
    rows = columns.shape[0]
    n_modalities = len(indices)
    counts = np.asarray([values.shape[1] for values in indices], dtype=np.int64)
    starts = np.zeros(n_modalities, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    width = int(counts.sum())
    # L2 normalization multiplies by float64 inverse norms, so it scores in
    # float64; without it a modality scores in its coordinates' precision.
    zeros = [
        embedding.dtype.type(0) if inverse is None else np.float64(0)
        for embedding, inverse in zip(embeddings, inverse_norms, strict=True)
    ]
    # A row of scratch holds a cell's coordinates and squared differences, or
    # the neighbour coordinates whose mean it takes.
    scratch_width = max(
        2 * max(embedding.shape[1] for embedding in embeddings), int(counts.max())
    )
    stacked = np.empty((rows, width), dtype=np.int64)
    for first, count, values in zip(starts, counts, indices, strict=True):
        stacked[:, first : first + count] = values[start : start + rows]
    pools = np.empty((rows, width), dtype=np.int64)
    sizes = np.empty(rows, dtype=np.int64)
    positions = np.empty((rows, width), dtype=np.int64)
    _candidate_pools(stacked, np.argsort(stacked, axis=1), pools, sizes, positions)

    distances: list[np.ndarray] = []
    nearest: list[np.ndarray] = []
    bandwidths: list[np.ndarray] = []
    for embedding, inverse, zero, first, count in zip(
        embeddings, inverse_norms, zeros, starts, counts, strict=True
    ):
        values = np.empty((rows, width), dtype=zero.dtype)
        _candidate_distances(
            embedding,
            inverse,
            start,
            pools,
            sizes,
            zero,
            np.empty((rows, scratch_width), dtype=zero.dtype),
            values,
        )
        # The nearest and the k-th of the modality's own neighbours, with
        # NaN sorted last.
        ranked = np.sort(
            np.take_along_axis(values, positions[:, first : first + count], 1),
            axis=1,
        )
        distances.append(values)
        nearest.append(ranked[:, 0].copy())
        bandwidths.append(ranked[:, -1] - ranked[:, 0])
        del ranked
    del positions

    # A modality's score against another is how much better its own
    # neighbours predict the cell than the other modality's neighbours.
    scores = np.full((rows, n_modalities, n_modalities), -np.inf)
    for target, (embedding, inverse, zero) in enumerate(
        zip(embeddings, inverse_norms, zeros, strict=True)
    ):
        offsets = np.empty((rows, n_modalities, embedding.shape[1]), zero.dtype)
        _prediction_offsets(
            embedding,
            inverse,
            start,
            stacked,
            starts,
            counts,
            zero,
            np.empty((rows, scratch_width), dtype=zero.dtype),
            offsets,
        )
        # np.vecdot takes each dot product as np.linalg.norm takes it.
        predicted = np.sqrt(np.vecdot(offsets, offsets)).astype(np.float64)
        del offsets
        predictions = _affinities(
            predicted,
            nearest[target].astype(np.float64),
            bandwidths[target].astype(np.float64),
        )
        for source in range(n_modalities):
            if source != target:
                scores[:, target, source] = np.clip(
                    predictions[:, target] / (predictions[:, source] + 1e-4),
                    0,
                    200,
                )
    del stacked
    finite = np.isfinite(scores)
    if not finite.any(axis=(1, 2)).all():
        raise FloatingPointError("WNN integration produced non-finite modality weights")
    top = np.where(finite, scores, -np.inf).max(axis=(1, 2))
    strengths = np.zeros(scores.shape, dtype=np.float64)
    strengths[finite] = np.exp((scores - top[:, np.newaxis, np.newaxis])[finite])
    totals = strengths.sum(axis=2)
    weights = totals / totals.sum(axis=1, keepdims=True)
    modality_weights[:] = weights
    del scores, finite, strengths

    # Each modality's affinities are weighted and added in modality order,
    # and its distances are released once they are used.
    combined = _affinities(distances.pop(0), nearest[0], bandwidths[0])
    combined *= weights[:, :1]
    for modality in range(1, n_modalities):
        affinity = _affinities(
            distances.pop(0), nearest[modality], bandwidths[modality]
        )
        affinity *= weights[:, modality : modality + 1]
        combined += affinity
        del affinity
    _select_candidates(
        combined,
        pools,
        sizes,
        np.empty((rows, columns.shape[1]), dtype=np.int64),
        columns,
        data,
    )


def _integrate_cells(
    validated_indices: list[np.ndarray],
    validated_embeddings: list[np.ndarray],
    inverse_norms: list[np.ndarray | None],
    nk: int,
    merged_columns: np.ndarray,
    merged_data: np.ndarray,
    modality_weights: np.ndarray,
) -> None:
    """Score one cell at a time with NumPy.

    This loop defines the values that ``_integrate_blocks`` computes, and it
    scores coordinates that the compiled kernels do not read, such as float16
    or extended precision, in their own precision.
    """
    n_cells = validated_indices[0].shape[0]
    n_modalities = len(validated_indices)
    for cell_idx in iter_progress(
        range(n_cells),
        desc="Building WNN graph",
        total=n_cells,
    ):
        neighbor_rows = [indices[cell_idx] for indices in validated_indices]
        mixed_k = neighbor_rows[0]
        for neighbors in neighbor_rows[1:]:
            mixed_k = np.union1d(mixed_k, neighbors)

        candidate_data: list[np.ndarray] = []
        points: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        positions = [np.searchsorted(mixed_k, neighbors) for neighbors in neighbor_rows]
        nearest_distances: list[float] = []
        bandwidths: list[float] = []
        for embedding, inverse, own_positions in zip(
            validated_embeddings,
            inverse_norms,
            positions,
            strict=True,
        ):
            candidates = embedding[mixed_k]
            point = embedding[cell_idx]
            if inverse is not None:
                candidates = candidates * inverse[mixed_k, np.newaxis]
                point = point * inverse[cell_idx]
            modality_distances = np.linalg.norm(point - candidates, axis=1)
            ranked_distances = np.sort(modality_distances[own_positions])
            nearest_distance = float(ranked_distances[0])
            candidate_data.append(candidates)
            points.append(point)
            distances.append(modality_distances)
            nearest_distances.append(nearest_distance)
            bandwidths.append(float(ranked_distances[-1] - nearest_distance))

        within_affinities = [
            _prediction_affinity(
                point,
                candidates[own_positions].mean(axis=0),
                nearest_distance,
                bandwidth,
            )
            for point, candidates, own_positions, nearest_distance, bandwidth in zip(
                points,
                candidate_data,
                positions,
                nearest_distances,
                bandwidths,
                strict=True,
            )
        ]
        scores = np.full(
            (n_modalities, n_modalities),
            -np.inf,
            dtype=np.float64,
        )
        for target_idx, (
            point,
            candidates,
            nearest_distance,
            bandwidth,
            within_affinity,
        ) in enumerate(
            zip(
                points,
                candidate_data,
                nearest_distances,
                bandwidths,
                within_affinities,
                strict=True,
            )
        ):
            for source_idx, source_positions in enumerate(positions):
                if source_idx == target_idx:
                    continue
                cross_affinity = _prediction_affinity(
                    point,
                    candidates[source_positions].mean(axis=0),
                    nearest_distance,
                    bandwidth,
                )
                scores[target_idx, source_idx] = np.clip(
                    within_affinity / (cross_affinity + 1e-4),
                    0,
                    200,
                )

        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            # Every cross-modality prediction is non-finite, as when the
            # distances of all modalities overflow, so the cell has no
            # modality weights.
            raise FloatingPointError(
                "WNN integration produced non-finite modality weights"
            )
        max_score = float(finite_scores.max())
        pairwise_strengths = np.zeros(scores.shape, dtype=np.float64)
        finite = np.isfinite(scores)
        pairwise_strengths[finite] = np.exp(scores[finite] - max_score)
        modality_strengths = pairwise_strengths.sum(axis=1)
        weights = modality_strengths / modality_strengths.sum()
        modality_weights[cell_idx] = weights

        combined_affinity = weights[0] * _kernel_affinity(
            distances[0],
            nearest_distances[0],
            bandwidths[0],
        )
        for weight, modality_distances, nearest_distance, bandwidth in zip(
            weights[1:],
            distances[1:],
            nearest_distances[1:],
            bandwidths[1:],
            strict=True,
        ):
            combined_affinity += weight * _kernel_affinity(
                modality_distances,
                nearest_distance,
                bandwidth,
            )
        selected = np.lexsort((mixed_k, -combined_affinity))[:nk]
        output_slice = slice(cell_idx * nk, (cell_idx + 1) * nk)
        merged_columns[output_slice] = mixed_k[selected]
        merged_data[output_slice] = np.maximum(
            np.clip(combined_affinity[selected], 0.0, 1.0),
            np.finfo(np.float32).tiny,
        )


@restore_numba_threads
def _wnn_integration_many(
    modalities: list[tuple[str, np.ndarray, np.ndarray]],
    nthreads: int,
    *,
    l2_normalize: bool = True,
) -> tuple[coo_matrix, np.ndarray]:
    """Build an N-modality WNN graph and per-cell modality weights.

    Candidates are the union of each modality's self-free neighbour row. Each
    modality turns distances into affinities with its own nearest and k-th
    neighbour distances, and rows are L2-normalized during scoring by default.
    This bounded pool and bandwidth differ from Seurat's wider search and
    SNN-far bandwidth. The COO graph stores blended affinities as float32
    edge weights, and the second array holds float32 modality weights.

    Compiled kernels score blocks of cells on up to ``nthreads`` threads, each
    cell on its own, so the result does not depend on the thread count. They
    read float32 and float64 coordinates; other floating types, such as
    float16, are scored one cell at a time in their own precision.
    """
    if not isinstance(l2_normalize, bool | np.bool_):
        raise TypeError("l2_normalize must be a boolean")
    if len(modalities) < 2:
        raise ValueError("WNN integration requires at least two modalities")

    names = [name for name, _indices, _embedding in modalities]
    if len(set(names)) != len(names):
        raise ValueError("WNN modality names must be unique")

    first_name, first_indices, first_embedding = modalities[0]
    validated_indices = [_validate_neighbor_indices(first_name, first_indices)]
    n_cells = validated_indices[0].shape[0]
    validated_embeddings = [_validate_embedding(first_name, first_embedding, n_cells)]
    for name, indices, embedding in modalities[1:]:
        validated_indices.append(
            _validate_neighbor_indices(name, indices, expected_cells=n_cells)
        )
        validated_embeddings.append(_validate_embedding(name, embedding, n_cells))

    neighbor_counts = [indices.shape[1] for indices in validated_indices]
    nk = min(neighbor_counts)
    if len(set(neighbor_counts)) != 1:
        counts = ", ".join(
            f"{name}: {count}"
            for name, count in zip(names, neighbor_counts, strict=True)
        )
        logger.warning(
            f"WNN graphs have different neighbor counts ({counts}). "
            f"The integrated graph will retain {nk} "
            "neighbors per cell."
        )

    from threadpoolctl import threadpool_limits

    inverse_norms: list[np.ndarray | None] = (
        [_inverse_row_norms(embedding) for embedding in validated_embeddings]
        if l2_normalize
        else [None] * len(modalities)
    )
    index_dtype = np.uint32 if n_cells < 2**32 else np.uint64
    output_size = n_cells * nk
    merged_columns = np.empty(output_size, dtype=index_dtype)
    merged_data = np.empty(output_size, dtype=np.float32)
    modality_weights = np.empty((n_cells, len(modalities)), dtype=np.float32)

    # Numba sets its thread cap at import, an attribute no stub declares.
    cap = int(getattr(config, "NUMBA_NUM_THREADS"))
    set_num_threads(min(max(1, int(nthreads)), cap))
    with threadpool_limits(limits=nthreads):
        if all(embedding.dtype in _KERNEL_DTYPES for embedding in validated_embeddings):
            _integrate_blocks(
                validated_indices,
                validated_embeddings,
                inverse_norms,
                merged_columns.reshape(n_cells, nk),
                merged_data.reshape(n_cells, nk),
                modality_weights,
            )
        else:
            _integrate_cells(
                validated_indices,
                validated_embeddings,
                inverse_norms,
                nk,
                merged_columns,
                merged_data,
                modality_weights,
            )

    if not np.all(np.isfinite(merged_data)):
        raise FloatingPointError("WNN integration produced non-finite graph weights")
    rows = np.repeat(np.arange(n_cells, dtype=index_dtype), nk)
    graph = coo_matrix(
        (merged_data, (rows, merged_columns)),
        shape=(n_cells, n_cells),
    )
    return graph, modality_weights
