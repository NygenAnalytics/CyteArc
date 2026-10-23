"""Doublet detection from simulated doublets and their nearest neighbors.

Each sample's doublets are simulated by adding the counts of two of its cells.
They are normalized like the observed cells and projected onto the observed
cells' PCA. Every observed cell and every simulated doublet is then scored by
the simulated doublets among its nearest neighbors in its own sample, and the
score is turned into the probability that the cell is a doublet, given an
expected doublet rate, following Scrublet (Wolock, Lopez and Klein, Cell
Systems 2019, doi:10.1016/j.cels.2018.11.005).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, vstack
from threadpoolctl import threadpool_limits

from ..assay import RNAassay
from ..assay.persistence import _read_block
from ..mapping.features import normalize_reference_counts
from ..mapping.reference import MappingReference
from ..mapping.symphony import project_pca
from ..matrix import ChunkedArray
from ..metadata.rows import read_array_rows_chunkwise
from ..neighbors.diffusion import transition_matrix
from ..neighbors.index import (
    ann_index_peak_bytes,
    ann_query_block_bytes,
    fix_knn_query,
    instantiate_knn_index,
)
from ..storage.artifacts import artifact_group
from ..storage.budget import ResourceBudget
from ..storage.execution import admit_stream
from ..storage.geometry import array_geometry
from ..storage.parallel import stream_shards
from ..storage.partition import (
    affordable_width,
    is_contiguous,
    row_band,
    row_read_parts,
)
from ..storage.types import as_zarr_array
from ..utils.arguments import float_argument, integer_argument
from ..utils.arrays import sparse_matrix_bytes

__all__ = [
    "DoubletIndexSettings",
    "DoubletSamplePlan",
    "DoubletSampleResult",
    "DoubletScores",
    "count_simulated_neighbors",
    "doublet_likelihood",
    "doublet_threshold",
    "histogram_minimum_threshold",
    "minmax_doublet_scores",
    "otsu_threshold",
    "plan_doublet_samples",
    "sample_cluster_pool",
    "score_synthetic_doublets",
    "simulate_doublet_pairs",
    "smooth_doublet_scores",
    "sum_doublet_pairs",
]

# Histograms of simulated-doublet scores have this many equal-width bins, and
# the minimum method smooths one at most this many times.
THRESHOLD_BINS = 256
THRESHOLD_SMOOTHING_STEPS = 10_000


@dataclass(frozen=True, slots=True)
class DoubletIndexSettings:
    """The hnswlib settings of the index that one sample is scored on.

    Attributes:
        metric: ``"l2"`` or ``"cosine"``.
        m: The HNSW ``M``; at least two.
        ef_construction: The search depth that builds the index.
        ef: The least search depth of a query; a query searches at least as
            deep as the neighbors it returns.
        parallel: Whether the index is built on several threads, which is
            faster but does not give the same index on every run.
    """

    metric: str
    m: int
    ef_construction: int
    ef: int
    parallel: bool

    def __post_init__(self) -> None:
        if self.metric not in {"l2", "cosine"}:
            raise ValueError("The doublet index metric must be 'l2' or 'cosine'")
        integer_argument(self.m, "m", minimum=2)
        integer_argument(self.ef_construction, "ef_construction", minimum=1)
        integer_argument(self.ef, "ef", minimum=1)
        if not isinstance(self.parallel, bool):
            raise TypeError("parallel must be a boolean")


@dataclass(frozen=True, slots=True)
class DoubletSamplePlan:
    """The sizes that one sample's simulation and scoring use.

    Attributes:
        cells: Selected cells of the sample.
        simulated: Doublets simulated from them.
        neighbors: Neighbors ``k`` that the score of a cell counts, at most
            one fewer than the cells.
        adjusted_neighbors: Neighbors that each cell and simulated doublet is
            scored on: ``k`` scaled by the observed and simulated cells per
            observed cell, rounded to the nearest integer.
    """

    cells: int
    simulated: int
    neighbors: int
    adjusted_neighbors: int


@dataclass(frozen=True, slots=True)
class DoubletSampleResult:
    """One sample's threshold, calls, and simulation.

    Attributes:
        plan: The sizes that the sample used.
        heterotypic_fraction: The share of simulated doublets forced to pair
            two clusters; 0 when the sample's parents hold one cluster.
        threshold: The doublet score above which a cell is called a doublet.
        threshold_method: ``"minimum"`` or ``"otsu"`` for an automatic
            threshold, ``"manual"`` for a given one.
        predicted: Cells called doublets.
        detectable_fraction: Simulated doublets scored above the threshold.
    """

    plan: DoubletSamplePlan
    heterotypic_fraction: float
    threshold: float
    threshold_method: str
    predicted: int
    detectable_fraction: float

    @property
    def estimated_doublet_rate(self) -> float | None:
        """Cells called doublets per detectable doublet, or None if none is.

        Scrublet's estimate of the sample's doublet rate: the fraction of
        cells called doublets divided by the fraction of simulated doublets
        that the threshold detects.
        """
        if self.detectable_fraction == 0:
            return None
        return self.predicted / self.plan.cells / self.detectable_fraction


@dataclass(frozen=True, slots=True)
class DoubletScores:
    """Doublet scores of the selected cells and of their simulated doublets.

    Attributes:
        scores: The doublet score of each selected cell, the probability that
            it is a doublet.
        raw_scores: The fraction of simulated doublets among each selected
            cell's neighbors.
        predicted: Whether each selected cell scores above its sample's
            threshold.
        simulated_scores: The doublet score of each simulated doublet. The
            doublets of each sample follow those of the samples before it.
        simulated_parents: The positions of each simulated doublet's two
            parents among the selected cells.
        samples: Each sample's result, in sample code order.
    """

    scores: NDArray[np.float64]
    raw_scores: NDArray[np.float64]
    predicted: NDArray[np.bool_]
    simulated_scores: NDArray[np.float64]
    simulated_parents: NDArray[np.int64]
    samples: tuple[DoubletSampleResult, ...]


def sample_cluster_pool(
    clusters: NDArray,
    fraction: float,
    max_per_cluster: int | None,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw a per-cluster subsample of cell positions to seed doublet simulation.

    Each cluster contributes ``ceil(fraction * size)`` cells, at most
    ``max_per_cluster`` of them unless that is None.
    """
    pool = []
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        n = min(int(np.ceil(len(idx) * fraction)), len(idx))
        if max_per_cluster is not None:
            n = min(n, max_per_cluster)
        if n <= 0:
            continue
        pool.append(rng.choice(idx, size=n, replace=False))
    if len(pool) == 0:
        raise ValueError("ERROR: No cells could be sampled to simulate doublets")
    return np.sort(np.concatenate(pool))


def simulate_doublet_pairs(
    pool_clusters: NDArray,
    n_sim: int,
    heterotypic_fraction: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Generate index pairs into the candidate pool for simulated doublets.

    Raises:
        ValueError: If ``heterotypic_fraction`` is above 0 with one cluster.
    """
    n_sim = integer_argument(n_sim, "n_sim", minimum=0)
    fraction = float_argument(heterotypic_fraction, "heterotypic_fraction")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("heterotypic_fraction must be from 0 to 1")
    pool_clusters = np.asarray(pool_clusters)
    if pool_clusters.ndim != 1 or len(pool_clusters) == 0:
        raise ValueError("pool_clusters must be a non-empty one-dimensional array")
    _, codes, counts = np.unique(pool_clusters, return_inverse=True, return_counts=True)
    if fraction > 0 and len(counts) < 2:
        raise ValueError(
            "Heterotypic doublets pair cells from two clusters, but the pool "
            "holds one cluster; pass heterotypic_fraction=0 to pair any two cells"
        )
    pool_size = len(pool_clusters)
    left = rng.integers(0, pool_size, size=n_sim)
    if fraction == 0:
        return left, rng.integers(0, pool_size, size=n_sim)
    forced = rng.random(n_sim) < fraction
    own = codes[left]
    # A forced doublet draws its partner among the cells outside its first
    # parent's cluster; any other doublet draws it from the whole pool.
    high = np.full(n_sim, pool_size, dtype=np.int64)
    high[forced] -= counts[own[forced]]
    right = rng.integers(0, high)
    del high
    # Ordered by cluster, the pool holds each cluster as one block. A forced
    # draw indexes that order with its first parent's block left out, so a
    # draw at or past the block's start moves past the block's end.
    rows = np.flatnonzero(forced)
    blocks = own[rows]
    positions = right[rows]
    past = positions >= (np.cumsum(counts) - counts)[blocks]
    positions[past] += counts[blocks[past]]
    right[rows] = np.argsort(codes, kind="stable")[positions]
    return left, right


def sum_doublet_pairs(
    pool_counts: csr_matrix,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
) -> csr_matrix:
    """Add the count rows of each sampled pair.

    Integer sums are exact: bool counts add as uint8, integers narrower than 64
    bits widen to twice their width, and 64-bit integer sums are checked against
    the range of their dtype. Floating-point counts add in float64.

    Raises:
        OverflowError: If a 64-bit integer sum is outside the range of its
            dtype.
    """
    dtype = pool_counts.dtype
    if dtype.kind == "b":
        dtype = np.dtype("uint8")
    elif dtype.kind in "iu" and dtype.itemsize < 8:
        dtype = np.dtype(f"{dtype.kind}{dtype.itemsize * 2}")
    elif dtype.kind == "f":
        dtype = np.dtype(np.float64)
    first = pool_counts[left].astype(dtype, copy=False)
    second = pool_counts[right].astype(dtype, copy=False)
    if dtype.kind in "iu" and pool_counts.dtype.itemsize == 8 and first.nnz:
        entries = first.tocoo(copy=False)
        other_values = np.asarray(second[entries.row, entries.col]).ravel()
        limits = np.iinfo(dtype)
        positive = other_values > 0
        overflow = np.any(entries.data[positive] > limits.max - other_values[positive])
        if dtype.kind == "i":
            negative = other_values < 0
            overflow |= np.any(
                entries.data[negative] < limits.min - other_values[negative]
            )
        if overflow:
            raise OverflowError(f"Synthetic doublet counts exceed the range of {dtype}")
    return (first + second).tocsr()


def plan_doublet_samples(
    cell_counts: Sequence[int] | NDArray[np.integer],
    *,
    simulation_ratio: float,
    n_neighbors: int,
) -> tuple[DoubletSamplePlan, ...]:
    """Return the sizes that each sample's simulation and scoring use.

    A sample of ``n`` cells simulates ``max(1, round(simulation_ratio * n))``
    doublets, ``s``. Its cells count ``k = min(n_neighbors, n - 1)`` neighbors,
    and every cell and simulated doublet is scored on ``round(k * (1 + s / n))``
    neighbors among the sample's other cells and doublets, so that ``k`` of
    them are observed cells when the two are evenly mixed. Rounding takes the
    nearest integer, and a tie the even one.

    Raises:
        ValueError: If a sample has fewer than two cells, which cannot pair.
    """
    ratio = float_argument(simulation_ratio, "simulation_ratio")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("simulation_ratio must be a finite number > 0")
    neighbors = integer_argument(n_neighbors, "n_neighbors", minimum=1)
    plans = []
    for raw_count in cell_counts:
        cells = integer_argument(raw_count, "cell count", minimum=0)
        if cells < 2:
            raise ValueError(
                f"A sample of {cells} selected cells cannot pair cells into "
                "doublets; every sample needs at least two"
            )
        simulated = max(1, int(round(ratio * cells)))
        k = min(neighbors, cells - 1)
        plans.append(
            DoubletSamplePlan(
                cells=cells,
                simulated=simulated,
                neighbors=k,
                adjusted_neighbors=int(round(k * (1 + simulated / cells))),
            )
        )
    return tuple(plans)


def _doublet_rate(value: float) -> float:
    rate = float_argument(value, "expected_doublet_rate")
    if not 0 < rate < 1:
        raise ValueError("expected_doublet_rate must be above 0 and below 1")
    return rate


def doublet_likelihood(
    simulated_neighbors: NDArray[np.integer],
    adjusted_neighbors: int,
    *,
    expected_doublet_rate: float,
    simulation_ratio: float,
) -> NDArray[np.float64]:
    """Return the probability that a cell is a doublet from its neighbors.

    A cell whose ``adjusted_neighbors`` nearest neighbors hold ``d`` simulated
    doublets estimates the fraction of simulated doublets around it with the
    rule of succession, ``q = (d + 1) / (adjusted_neighbors + 2)``. With ``r``
    simulated doublets per observed cell, ``q / ((1 - q) r)`` estimates how
    much denser doublets than singlets are around the cell, so Bayes' rule
    with the prior doublet rate ``rho`` gives the probability that it is a
    doublet:

    ``q rho / r / (q rho / r + (1 - q)(1 - rho))``

    This is Scrublet's doublet score (Wolock, Lopez and Klein, 2019).

    Raises:
        ValueError: If an argument is out of range.
    """
    counts = np.asarray(simulated_neighbors)
    if counts.dtype.kind not in "iu":
        raise TypeError("simulated_neighbors must hold integer counts")
    total = integer_argument(adjusted_neighbors, "adjusted_neighbors", minimum=1)
    if counts.size and (counts.min() < 0 or counts.max() > total):
        raise ValueError("simulated_neighbors must be from 0 to adjusted_neighbors")
    rate = _doublet_rate(expected_doublet_rate)
    ratio = float_argument(simulation_ratio, "simulation_ratio")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("simulation_ratio must be a finite number > 0")
    fraction = (counts.astype(np.float64) + 1) / (total + 2)
    doublet = fraction * rate / ratio
    return doublet / (doublet + (1 - fraction) * (1 - rate))


def _finite_values(values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Threshold values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("Threshold values must be finite")
    return array


def _histogram(
    values: NDArray[np.float64], bins: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    counts, edges = np.histogram(values, bins=bins)
    return counts.astype(np.float64), (edges[:-1] + edges[1:]) / 2


def _local_maxima(heights: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return the middle bin of each run of equal heights above its neighbors.

    A run at either end of the histogram needs to be above its one neighbor.
    """
    starts = np.flatnonzero(np.r_[True, heights[1:] != heights[:-1]])
    stops = np.r_[starts[1:] - 1, len(heights) - 1]
    runs = heights[starts]
    rises = np.r_[True, runs[1:] > runs[:-1]]
    falls = np.r_[runs[:-1] > runs[1:], True]
    peaks = rises & falls
    return np.asarray((starts[peaks] + stops[peaks]) // 2, dtype=np.int64)


def histogram_minimum_threshold(
    values: NDArray[np.floating],
    *,
    bins: int = THRESHOLD_BINS,
    max_steps: int = THRESHOLD_SMOOTHING_STEPS,
) -> float | None:
    """Return the minimum between the two modes of a bimodal histogram.

    The method of Prewitt and Mendelsohn (1966), as Glasbey (1993) describes
    it: the values' histogram of ``bins`` equal-width bins is smoothed with a
    running mean of three bins, the end bins repeated beyond each end, until it
    has exactly two local maxima. A maximum is a run of bins of equal height
    above the bins beside it. The threshold is the center of the lowest bin
    between the two maxima, the first one if several are lowest.

    Returns:
        The threshold, or None when the values are all equal or the histogram
        has fewer than two maxima, or still more than two after ``max_steps``
        smoothing steps.
    """
    array = _finite_values(values)
    integer_argument(bins, "bins", minimum=3)
    steps = integer_argument(max_steps, "max_steps", minimum=0)
    if array.min() == array.max():
        return None
    heights, centers = _histogram(array, bins)
    kernel = np.full(3, 1 / 3)
    maxima = _local_maxima(heights)
    for _ in range(steps):
        if len(maxima) <= 2:
            break
        heights = np.convolve(np.pad(heights, 1, mode="edge"), kernel, mode="valid")
        maxima = _local_maxima(heights)
    if len(maxima) != 2:
        return None
    low, high = maxima
    return float(centers[low + int(np.argmin(heights[low : high + 1]))])


def otsu_threshold(
    values: NDArray[np.floating],
    *,
    bins: int = THRESHOLD_BINS,
) -> float:
    """Return Otsu's threshold of the values' histogram.

    Otsu (1979) splits the histogram of ``bins`` equal-width bins into a low
    and a high class where the variance between the two classes is largest.
    The threshold is the center of the highest bin of the low class, the first
    such bin if several splits tie. Equal values have no split and return
    their value.
    """
    array = _finite_values(values)
    integer_argument(bins, "bins", minimum=2)
    if array.min() == array.max():
        return float(array[0])
    counts, centers = _histogram(array, bins)
    # Splitting after bin i makes bins 0..i the low class. The first and last
    # bins hold the smallest and largest values, so no class is ever empty.
    low_weight = np.cumsum(counts)[:-1]
    high_weight = counts.sum() - low_weight
    low_sum = np.cumsum(counts * centers)[:-1]
    high_sum = float(np.dot(counts, centers)) - low_sum
    between = (
        low_weight * high_weight * (low_sum / low_weight - high_sum / high_weight) ** 2
    )
    return float(centers[int(np.argmax(between))])


def doublet_threshold(simulated_scores: NDArray[np.floating]) -> tuple[float, str]:
    """Choose the doublet-score threshold of one sample automatically.

    Simulated doublets of two cells of one type score low and those of two
    types score high, so their scores are bimodal. The threshold is the
    minimum between the two modes (:func:`histogram_minimum_threshold`), or
    Otsu's threshold (:func:`otsu_threshold`) when the histogram does not
    become bimodal.

    Returns:
        The threshold and the method that chose it: ``"minimum"`` or ``"otsu"``.
    """
    threshold = histogram_minimum_threshold(simulated_scores)
    if threshold is not None:
        return threshold, "minimum"
    return otsu_threshold(simulated_scores), "otsu"


def _doublet_batch_rows(
    resources: ResourceBudget,
    preferred: int,
    row_bytes: int,
    resident_bytes: int,
    decode_bytes: int = 0,
) -> int:
    rows = affordable_width(
        lambda n: (
            resident_bytes + n * row_bytes + decode_bytes <= resources.memoryBytes
        ),
        preferred,
    )
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=max(1, rows) * row_bytes,
        decodeBytes=decode_bytes,
        residentBytes=resident_bytes,
        requested=1,
    )
    return rows


def _load_parent_counts(
    raw: ChunkedArray,
    cell_indices: np.ndarray,
    feature_indices: np.ndarray,
    resources: ResourceBudget,
    resident_bytes: int,
) -> csr_matrix:
    """Read the selected features of parent rows in admitted blocks."""
    backing = raw._backing
    geometry = array_geometry(backing)
    preferred = row_band(geometry, unit="chunk", fallback=1_000)
    columns = np.asarray(feature_indices, dtype=np.int64)
    decode_bytes, read_row_bytes = row_read_parts(
        geometry, columns, rows_contiguous=is_contiguous(cell_indices)
    )
    parts: list[csr_matrix] = []
    retained = 0
    # Dense read, sparse conversion scratch, and the eventual concatenation copy.
    row_bytes = len(columns) * (4 * (raw.dtype.itemsize + 8)) + 32 + read_row_bytes

    def boundaries() -> Iterator[tuple[int, int]]:
        start = 0
        while start < len(cell_indices):
            rows = _doublet_batch_rows(
                resources,
                min(preferred, len(cell_indices) - start),
                row_bytes,
                resident_bytes + columns.nbytes + 2 * retained,
                decode_bytes,
            )
            yield start, start + rows
            start += rows

    def read(interval: tuple[int, int]) -> csr_matrix:
        start, end = interval
        rows = cell_indices[start:end]
        values = (
            backing[np.ix_(rows, columns)]
            if isinstance(backing, np.ndarray)
            else _read_block(backing, rows, columns)
        )
        return csr_matrix(values)

    for part in stream_shards(
        boundaries(),
        read,
        workers=1,
        io_concurrency=1,
        msg="Reading doublet parents",
    ):
        parts.append(part)
        # Concatenation may promote sparse indices to int64.
        retained += part.nnz * (raw.dtype.itemsize + 8) + 8 * (part.shape[0] + 1)
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=max(1, retained),
        residentBytes=resident_bytes + retained,
        requested=1,
    )
    return vstack(parts, format="csr")


def _pair_row_bytes(pool: csr_matrix) -> int:
    largest = int(np.diff(pool.indptr).max(initial=0))
    # Row gathers, widening, addition, and checked 64-bit overflow scratch.
    return int(6 * (2 * largest * (max(8, pool.dtype.itemsize) + 8) + 16))


def count_simulated_neighbors(
    points: NDArray[np.floating],
    n_observed: int,
    adjusted_neighbors: int,
    settings: DoubletIndexSettings,
    *,
    random_seed: int,
    workers: int,
    resources: ResourceBudget,
    resident_bytes: int = 0,
) -> NDArray[np.int64]:
    """Count the simulated doublets among each point's nearest neighbors.

    ``points`` holds one sample's observed cells, its first ``n_observed``
    rows, and then its simulated doublets. An hnswlib index over all of them
    is queried for each point's ``adjusted_neighbors`` nearest other points.
    The index is built on one thread unless ``settings.parallel`` is set, so
    the counts are the same on every run; queries use ``workers`` threads.

    Raises:
        MemoryError: If the index and one query block exceed the memory budget.
    """
    values = np.ascontiguousarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Doublet neighbor points must hold at least two rows")
    n_points, dims = values.shape
    observed = integer_argument(n_observed, "n_observed", minimum=1)
    if observed >= n_points:
        raise ValueError("Doublet neighbor points must hold a simulated doublet")
    k = integer_argument(adjusted_neighbors, "adjusted_neighbors", minimum=1)
    if k >= n_points:
        raise ValueError("adjusted_neighbors must be below the number of points")
    threads = integer_argument(workers, "workers", minimum=1)
    build_threads = threads if settings.parallel else 1
    ef = max(settings.ef, k + 1)
    index_bytes = ann_index_peak_bytes(
        n_points, dims, settings.m, nthreads=max(build_threads, threads)
    )
    resident = int(resident_bytes) + values.nbytes + 8 * n_points
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=index_bytes,
        residentBytes=resident,
        requested=1,
    )
    resident += index_bytes
    # Beside hnswlib's own query memory, each query row holds its labels and
    # distances, their copies without the point itself, and a doublet mask.
    own_row_bytes = 2 * 12 * (k + 1) + (k + 1)

    def block_bytes(rows: int) -> int:
        return (
            ann_query_block_bytes(max(1, rows), k + 1, ef=ef, nthreads=threads)
            + rows * own_row_bytes
        )

    batch_rows = max(
        1,
        affordable_width(
            lambda rows: resident + block_bytes(rows) <= resources.memoryBytes,
            n_points,
        ),
    )
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=block_bytes(batch_rows),
        residentBytes=resident,
        requested=1,
    )
    index = instantiate_knn_index(
        settings.metric,
        dims,
        n_points,
        settings.ef_construction,
        settings.m,
        random_seed,
        ef,
        build_threads,
    )
    index.add_items(values, np.arange(n_points))
    index.set_num_threads(threads)
    counts = np.empty(n_points, dtype=np.int64)
    for start in range(0, n_points, batch_rows):
        stop = min(start + batch_rows, n_points)
        labels, distances = index.knn_query(values[start:stop], k=k + 1)
        neighbors, _, _ = fix_knn_query(labels, distances, np.arange(start, stop))
        counts[start:stop] = np.count_nonzero(neighbors >= observed, axis=1)
    return counts


def _simulate_sample(
    assay: RNAassay,
    reference: MappingReference,
    rng: np.random.Generator,
    members: np.ndarray,
    plan: DoubletSamplePlan,
    *,
    active_indices: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    cell_totals: np.ndarray | None,
    cluster_sample_fraction: float,
    max_cells_per_cluster: int | None,
    heterotypic_fraction: float,
    resources: ResourceBudget,
    resident_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate one sample's doublets and project them onto the reference PCA.

    Returns:
        The float32 coordinates of the doublets, the positions of their two
        parents among the selected cells, and the heterotypic fraction used:
        0 when every parent has the same cluster label.
    """
    model = reference.model
    member_labels = labels[members]
    if cluster_sample_fraction >= 1 and max_cells_per_cluster is None:
        pool = np.arange(len(members), dtype=np.int64)
    else:
        pool = sample_cluster_pool(
            member_labels, cluster_sample_fraction, max_cells_per_cluster, rng
        )
    pool_labels = member_labels[pool]
    # A sample of one cluster can form only homotypic doublets.
    fraction = heterotypic_fraction if len(np.unique(pool_labels)) > 1 else 0.0
    # Pair generation holds both parent indices, the forced-doublet mask, and
    # integer scratch for the forced doublets, at most 72 bytes per doublet,
    # and sorts the pool labels into cluster codes and a cluster order.
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=72 * plan.simulated
        + (2 * pool_labels.dtype.itemsize + 48) * len(pool),
        residentBytes=resident_bytes + pool_labels.nbytes + 8 * len(pool),
        requested=1,
    )
    left, right = simulate_doublet_pairs(pool_labels, plan.simulated, fraction, rng)
    pool_positions = members[pool]
    parents = np.column_stack((pool_positions[left], pool_positions[right]))
    del pool, pool_labels
    resident = resident_bytes + left.nbytes + right.nbytes + parents.nbytes
    resident += pool_positions.nbytes
    counts = _load_parent_counts(
        assay.rawData,
        active_indices[pool_positions],
        feature_indices,
        resources,
        resident,
    )
    totals = None if cell_totals is None else cell_totals[pool_positions]
    coordinates = np.empty((plan.simulated, model.n_dims), dtype=np.float32)
    resident += coordinates.nbytes + sparse_matrix_bytes(counts)
    resident += 0 if totals is None else totals.nbytes
    row_bytes = (
        _pair_row_bytes(counts)
        + model.n_features * (max(8, counts.dtype.itemsize) + 3 * 8)
        + model.n_dims * 32
        + 64
    )
    preferred = row_band(
        array_geometry(assay.rawData._backing), unit="chunk", fallback=1_000
    )
    batch_rows = _doublet_batch_rows(
        resources, min(preferred, plan.simulated), row_bytes, resident
    )
    parameters = reference.normalization_parameters
    with threadpool_limits(limits=resources.workers):
        for start in range(0, plan.simulated, batch_rows):
            stop = min(start + batch_rows, plan.simulated)
            first, second = left[start:stop], right[start:stop]
            simulated = sum_doublet_pairs(counts, first, second)
            raw = simulated.toarray()
            del simulated
            normalized = normalize_reference_counts(
                raw,
                size_factor=parameters["size_factor"],
                log_transform=parameters["log_transform"],
                source=f"The doublet simulation of {assay.name}",
                # A doublet's library is the libraries of its two parents.
                denominator=None if totals is None else totals[first] + totals[second],
            )
            del raw
            coordinates[start:stop] = project_pca(normalized, model)
            del normalized
    return coordinates, parents, fraction


def score_synthetic_doublets(
    assay: RNAassay,
    reference: MappingReference,
    active_indices: np.ndarray,
    labels: np.ndarray,
    sample_codes: np.ndarray,
    feature_indices: np.ndarray,
    cell_totals: np.ndarray | None,
    *,
    index: DoubletIndexSettings,
    cluster_sample_fraction: float,
    max_cells_per_cluster: int | None,
    simulation_ratio: float,
    heterotypic_fraction: float,
    n_neighbors: int,
    expected_doublet_rate: float,
    threshold: float | None,
    random_seed: int,
    resources: ResourceBudget,
) -> DoubletScores:
    """Simulate doublets within each sample and score cells by their neighbors.

    Each sample draws parent pairs from its own cells, every cell by default,
    with its own random generator: the sample's child of
    ``numpy.random.SeedSequence(random_seed)``, spawned in sample code order.
    Parent counts are added, normalized like the reference, with each
    doublet's library size the sum of its parents' ``cell_totals`` unless the
    reference renormalizes the selected features, and projected onto the
    reference PCA. The sample's observed coordinates are its rows of the
    reference reduction. Each observed cell and simulated doublet is scored
    by :func:`doublet_likelihood` on its neighbors in the sample
    (:func:`count_simulated_neighbors`), and a cell is called a doublet when
    its score is above the sample's threshold: ``threshold``, or the one
    :func:`doublet_threshold` chooses from the sample's simulated doublets.

    Args:
        assay: The RNA assay whose counts the parents are read from.
        reference: The mapping reference of the scored cells' PCA.
        active_indices: Physical rows of the selected cells, in selection order.
        labels: The cluster label of each selected cell.
        sample_codes: The sample of each selected cell, from 0 to the number of
            samples minus 1; every code must occur.
        feature_indices: Physical rows of the reference features.
        cell_totals: Library sizes of the selected cells, or None when the
            reference renormalizes its features.
        index: Settings of each sample's neighbor index.
        cluster_sample_fraction: The fraction of each cluster of a sample that
            may parent doublets.
        max_cells_per_cluster: The most cells of a cluster of a sample that may
            parent doublets, or None for no limit.
        simulation_ratio: Simulated doublets per selected cell.
        heterotypic_fraction: The share of simulated doublets forced to pair
            cells of two clusters.
        n_neighbors: The neighbors ``k`` of :func:`plan_doublet_samples`.
        expected_doublet_rate: The prior doublet rate of every sample.
        threshold: The doublet score above which a cell is called a doublet,
            or None to choose one per sample.
        random_seed: The seed of the samples' generators and indexes.
        resources: The memory and worker budget.

    Raises:
        ValueError: If the inputs disagree in size, a sample has fewer than two
            cells, or an argument is out of range.
        MemoryError: If a step exceeds the memory budget.
    """
    n_cells = len(active_indices)
    codes = np.asarray(sample_codes)
    if codes.dtype.kind not in "iu" or codes.shape != (n_cells,):
        raise ValueError("Sample codes must hold one integer per selected cell")
    if np.shape(labels) != (n_cells,):
        raise ValueError("Cluster labels must hold one label per selected cell")
    if n_cells == 0 or codes.min() < 0:
        raise ValueError("Sample codes must be non-negative, one per selected cell")
    sizes = np.bincount(codes)
    if not sizes.all():
        raise ValueError("Every sample code up to the largest must hold a cell")
    # Settings are checked before the reference or any count is read.
    plans = plan_doublet_samples(
        sizes, simulation_ratio=simulation_ratio, n_neighbors=n_neighbors
    )
    _doublet_rate(expected_doublet_rate)
    if threshold is not None and not np.isfinite(float(threshold)):
        raise ValueError("threshold must be a finite number or None")
    parameters = reference.normalization_parameters
    if bool(parameters["renormalize_subset"]) != (cell_totals is None):
        raise ValueError(
            "Cell totals are needed exactly when the reference normalizes by "
            "library size over every feature"
        )
    if cell_totals is not None and np.shape(cell_totals) != (n_cells,):
        raise ValueError("Cell totals must hold one value per selected cell")
    model = reference.model
    coordinates = as_zarr_array(
        artifact_group(reference.datastore.zw, reference.reduction)["data"],
        name="data",
    )
    if coordinates.shape != (n_cells, model.n_dims):
        raise ValueError("Reference coordinates do not match the selected cells")
    n_simulated = sum(plan.simulated for plan in plans)
    resident = sum(
        np.asarray(values).nbytes
        for values in (
            active_indices,
            labels,
            codes,
            feature_indices,
            model.loadings,
            model.feature_means,
            model.feature_scales,
            model.center,
            *(() if cell_totals is None else (cell_totals,)),
        )
    )
    # Two float64 scores and a call per cell; a float64 score and two int64
    # parents per simulated doublet.
    output_bytes = 17 * n_cells + 24 * n_simulated
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=output_bytes,
        residentBytes=resident,
        requested=1,
    )
    resident += output_bytes
    scores = np.empty(n_cells, dtype=np.float64)
    raw_scores = np.empty(n_cells, dtype=np.float64)
    predicted = np.zeros(n_cells, dtype=bool)
    simulated_scores = np.empty(n_simulated, dtype=np.float64)
    simulated_parents = np.empty((n_simulated, 2), dtype=np.int64)
    results = []
    offset = 0
    seeds = np.random.SeedSequence(random_seed).spawn(len(plans))
    for code, (plan, seed) in enumerate(zip(plans, seeds, strict=True)):
        members = np.flatnonzero(codes == code)
        simulated, parents, fraction = _simulate_sample(
            assay,
            reference,
            np.random.default_rng(seed),
            members,
            plan,
            active_indices=active_indices,
            labels=labels,
            feature_indices=feature_indices,
            cell_totals=cell_totals,
            cluster_sample_fraction=cluster_sample_fraction,
            max_cells_per_cluster=max_cells_per_cluster,
            heterotypic_fraction=heterotypic_fraction,
            resources=resources,
            resident_bytes=resident + members.nbytes,
        )
        observed_bytes = plan.cells * model.n_dims * 4
        admit_stream(
            resources,
            nBlocks=1,
            blockBytes=2 * observed_bytes + simulated.nbytes,
            residentBytes=resident + members.nbytes + simulated.nbytes,
            requested=1,
        )
        observed = np.asarray(
            read_array_rows_chunkwise(coordinates, members), dtype=np.float32
        )
        points = np.concatenate((observed, simulated))
        del observed, simulated
        counts = count_simulated_neighbors(
            points,
            plan.cells,
            plan.adjusted_neighbors,
            index,
            random_seed=random_seed,
            workers=resources.workers,
            resources=resources,
            resident_bytes=resident + members.nbytes,
        )
        del points
        likelihood = doublet_likelihood(
            counts,
            plan.adjusted_neighbors,
            expected_doublet_rate=expected_doublet_rate,
            simulation_ratio=plan.simulated / plan.cells,
        )
        observed_scores = likelihood[: plan.cells]
        sample_simulated = likelihood[plan.cells :]
        if threshold is None:
            cutoff, method = doublet_threshold(sample_simulated)
        else:
            cutoff, method = float(threshold), "manual"
        calls = observed_scores > cutoff
        scores[members] = observed_scores
        raw_scores[members] = counts[: plan.cells] / plan.adjusted_neighbors
        predicted[members] = calls
        end = offset + plan.simulated
        simulated_scores[offset:end] = sample_simulated
        simulated_parents[offset:end] = parents
        offset = end
        results.append(
            DoubletSampleResult(
                plan=plan,
                heterotypic_fraction=fraction,
                threshold=cutoff,
                threshold_method=method,
                predicted=int(np.count_nonzero(calls)),
                detectable_fraction=float(np.mean(sample_simulated > cutoff)),
            )
        )
    return DoubletScores(
        scores=scores,
        raw_scores=raw_scores,
        predicted=predicted,
        simulated_scores=simulated_scores,
        simulated_parents=simulated_parents,
        samples=tuple(results),
    )


def minmax_doublet_scores(scores: np.ndarray) -> np.ndarray:
    """Scale scores linearly to [0, 1]; equal scores all become 0."""
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def smooth_doublet_scores(
    graph: csr_matrix,
    scores: np.ndarray,
    *,
    power: int,
    normalize: bool,
) -> np.ndarray:
    """Diffuse scores ``power`` steps over the row-normalized graph.

    With ``normalize``, the result is scaled by :func:`minmax_doublet_scores`.
    """
    if isinstance(power, bool) or not isinstance(power, int | np.integer):
        raise TypeError("t must be a positive integer")
    if power < 1:
        raise ValueError("t must be a positive integer")
    transition = transition_matrix(graph)
    for _ in range(power):
        scores = np.asarray(transition.dot(scores), dtype=np.float64)
    if normalize:
        scores = minmax_doublet_scores(scores)
    return scores
