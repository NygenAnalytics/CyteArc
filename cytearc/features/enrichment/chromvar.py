"""chromVAR motif (or peak set) accessibility deviations.

The numerics follow chromVAR 1.34 (``getBackgroundPeaks`` and
``computeDeviations``): background peaks are drawn from a Gaussian kernel over
a grid in the whitened space of log10 peak counts and GC content; each cell's
deviation for a peak set is its observed minus expected fragments in the set,
relative to the expected, corrected by the same deviation of background sets;
and the z-score divides the corrected deviation by the background deviations'
standard deviation. Cells whose expected count in a set is below one have no
value for it.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.sparse import csc_matrix, csr_matrix

from ...utils.arguments import float_argument, integer_argument

__all__ = [
    "ChromvarModel",
    "background_peaks",
    "build_chromvar_model",
    "chromvar_model_bytes",
    "score_chromvar_block",
]


def background_peaks(
    fragments_per_peak: np.ndarray,
    gc: np.ndarray,
    *,
    n_iterations: int = 50,
    kernel_width: float = 0.1,
    bins: int = 50,
    seed: int,
) -> np.ndarray:
    """Draw GC- and accessibility-matched background peaks, as chromVAR does.

    Peaks are placed by ``log10`` fragment count and GC content, whitened with
    the Cholesky factor of their covariance, and assigned to the nearest point
    of a ``bins`` by ``bins`` grid. Each peak then draws ``n_iterations``
    background peaks with replacement: every occupied grid bin receives
    probability from a Gaussian kernel of width ``kernel_width`` on the
    distance between bins, shared equally by the peaks in that bin. A peak
    can draw itself.

    Args:
        fragments_per_peak: Total fragments of each peak; all must be positive.
        gc: GC content of each peak, as from ``gc_content``.
        n_iterations: Background peaks per peak.
        kernel_width: Kernel standard deviation in whitened units.
        bins: Grid points per axis.
        seed: Seed of the random generator.

    Returns:
        A peaks-by-iterations array of zero-based background peak indices.
    """
    counts = np.asarray(fragments_per_peak, dtype=np.float64)
    bias = np.asarray(gc, dtype=np.float64)
    if counts.ndim != 1 or bias.shape != counts.shape:
        raise ValueError("fragments_per_peak and gc must be aligned vectors")
    if len(counts) < 2:
        raise ValueError("Background peaks need at least two peaks")
    if np.any(counts <= 0) or not np.isfinite(counts).all():
        raise ValueError("Every peak needs at least one fragment")
    if not np.isfinite(bias).all():
        raise ValueError("GC content must be finite; a peak has no A, C, G, or T base")
    n_iterations = integer_argument(n_iterations, "n_iterations", minimum=2)
    bins = integer_argument(bins, "bins", minimum=2)
    seed = integer_argument(seed, "seed", minimum=0)
    if not float_argument(kernel_width, "kernel_width") > 0:
        raise ValueError("kernel_width must be positive")
    from scipy.linalg import solve_triangular

    points = np.column_stack([np.log10(counts), bias])
    try:
        lower = np.linalg.cholesky(np.cov(points.T))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Peak counts and GC content are collinear or constant, so they "
            "cannot be whitened"
        ) from exc
    whitened = solve_triangular(lower, points.T, lower=True).T
    axis1 = np.linspace(whitened[:, 0].min(), whitened[:, 0].max(), bins)
    axis2 = np.linspace(whitened[:, 1].min(), whitened[:, 1].max(), bins)
    centers = np.column_stack([np.repeat(axis1, bins), np.tile(axis2, bins)])
    membership = _nearest_center(whitened, axis1, axis2, bins)
    density = np.bincount(membership, minlength=bins * bins).astype(np.float64)

    rng = np.random.default_rng(seed)
    result = np.empty((len(counts), n_iterations), dtype=np.int64)
    weight_per_peak = 1.0 / density[membership]
    for grid_bin in np.flatnonzero(density > 0):
        members = np.flatnonzero(membership == grid_bin)
        # Kernel weight of each grid point from this bin; the normal density's
        # constant cancels when the probabilities are normalized.
        offsets = centers - centers[grid_bin]
        distances = np.sqrt((offsets * offsets).sum(axis=1))
        kernel = np.exp(-0.5 * (distances / kernel_width) ** 2)
        probability = kernel[membership] * weight_per_peak
        probability = probability / probability.sum()
        draws = rng.choice(
            len(counts), size=(len(members), n_iterations), replace=True, p=probability
        )
        result[members] = draws
    return result


def _nearest_center(
    points: np.ndarray, axis1: np.ndarray, axis2: np.ndarray, bins: int
) -> np.ndarray:
    """Index of the nearest grid center of each point, lowest index on ties."""
    best = np.zeros(len(points), dtype=np.int64)
    best_distance = np.full(len(points), np.inf)
    for first in range(bins):
        dx = points[:, 0] - axis1[first]
        for second in range(bins):
            dy = points[:, 1] - axis2[second]
            distance = dx * dx + dy * dy
            closer = distance < best_distance
            best[closer] = first * bins + second
            best_distance[closer] = distance[closer]
    return best


@dataclass(frozen=True, slots=True, eq=False)
class ChromvarModel:
    """Peak sets, expectation, and background prepared for scoring cells.

    Attributes:
        annotation: Peaks-by-sets CSR matrix of peak multiplicities.
        backgroundIndex: Peaks-by-iterations background peak indices.
        expectation: Each peak's expected fraction of a cell's fragments.
        expectedSet: ``(sets,)`` expected fraction of fragments in each set.
        expectedBackground: ``(iterations, sets)`` expected fraction in each
            background set.
        invertedIndptr: Row pointers over peaks of the inverted background.
        invertedIndices: For each peak ``q``, one entry ``t * sets + m`` per
            member of set ``m`` whose background ``t`` is ``q``, repeated by
            the member's multiplicity.
    """

    annotation: csr_matrix
    backgroundIndex: np.ndarray
    expectation: np.ndarray
    expectedSet: np.ndarray
    expectedBackground: np.ndarray
    invertedIndptr: np.ndarray
    invertedIndices: np.ndarray

    @property
    def n_sets(self) -> int:
        """Number of peak sets."""
        return int(self.annotation.shape[1])

    @property
    def n_iterations(self) -> int:
        """Number of background iterations."""
        return int(self.backgroundIndex.shape[1])

    @property
    def nbytes(self) -> int:
        """Bytes held by the model's arrays."""
        annotation = self.annotation
        return int(
            annotation.data.nbytes
            + annotation.indices.nbytes
            + annotation.indptr.nbytes
            + self.backgroundIndex.nbytes
            + self.expectation.nbytes
            + self.expectedSet.nbytes
            + self.expectedBackground.nbytes
            + self.invertedIndptr.nbytes
            + self.invertedIndices.nbytes
        )


def chromvar_model_bytes(
    annotation_entries: int, n_peaks: int, n_iterations: int
) -> int:
    """Estimate the bytes of a model whose sets hold ``annotation_entries`` peaks.

    The inverted background index dominates: four bytes per member, per
    iteration.
    """
    return int(
        4 * annotation_entries * n_iterations
        + 8 * (n_peaks + 1) * 2
        + 8 * n_peaks * n_iterations
        + 24 * annotation_entries
    )


@njit(cache=True)
def _invert_background(
    indptr: np.ndarray,
    sets: np.ndarray,
    multiplicity: np.ndarray,
    background: np.ndarray,
    n_sets: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_peaks, n_iterations = background.shape
    sizes = np.zeros(n_peaks + 1, dtype=np.int64)
    for peak in range(n_peaks):
        for entry in range(indptr[peak], indptr[peak + 1]):
            for iteration in range(n_iterations):
                sizes[background[peak, iteration] + 1] += multiplicity[entry]
    pointers = np.cumsum(sizes)
    indices = np.empty(pointers[-1], dtype=np.int32)
    fill = pointers[:-1].copy()
    for peak in range(n_peaks):
        for entry in range(indptr[peak], indptr[peak + 1]):
            for iteration in range(n_iterations):
                target = background[peak, iteration]
                column = iteration * n_sets + sets[entry]
                for _ in range(multiplicity[entry]):
                    indices[fill[target]] = column
                    fill[target] += 1
    return pointers, indices


def build_chromvar_model(
    annotation: csc_matrix | csr_matrix,
    fragments_per_peak: np.ndarray,
    background_index: np.ndarray,
) -> ChromvarModel:
    """Prepare peak sets for ``score_chromvar_block``.

    Args:
        annotation: Peaks-by-sets matrix whose non-zero entries mark set
            members; values are used as multiplicities, as chromVAR counts a
            repeated peak once per occurrence.
        fragments_per_peak: Total fragments of each peak over the scored cells.
        background_index: Output of ``background_peaks``.

    Returns:
        The scoring model.
    """
    matches = csr_matrix(annotation)
    counts = np.asarray(fragments_per_peak, dtype=np.float64)
    background = np.ascontiguousarray(background_index, dtype=np.int64)
    n_peaks = len(counts)
    if (
        matches.shape[0] != n_peaks
        or background.ndim != 2
        or len(background) != n_peaks
    ):
        raise ValueError("annotation, fragments_per_peak, and background must align")
    if np.any(counts <= 0):
        raise ValueError("Every peak needs at least one fragment")
    if np.any(background < 0) or np.any(background >= n_peaks):
        raise ValueError("Background indices are out of range")
    matches.sum_duplicates()
    if matches.nnz and (
        np.any(matches.data <= 0) or np.any(matches.data != np.round(matches.data))
    ):
        raise ValueError("annotation values must be positive integer multiplicities")
    expectation = counts / counts.sum()
    n_sets = matches.shape[1]
    n_iterations = background.shape[1]
    by_set = csc_matrix(matches)
    weights = np.asarray(by_set.data, dtype=np.float64)
    set_index = np.repeat(np.arange(n_sets, dtype=np.int64), np.diff(by_set.indptr))
    members = np.asarray(by_set.indices, dtype=np.int64)
    expected_set = np.bincount(
        set_index, weights=weights * expectation[members], minlength=n_sets
    )
    expected_background = np.zeros((n_iterations, n_sets), dtype=np.float64)
    for iteration in range(n_iterations):
        expected_background[iteration] = np.bincount(
            set_index,
            weights=weights * expectation[background[members, iteration]],
            minlength=n_sets,
        )
    pointers, indices = _invert_background(
        matches.indptr.astype(np.int64),
        matches.indices.astype(np.int64),
        matches.data.astype(np.int64),
        background,
        n_sets,
    )
    return ChromvarModel(
        annotation=matches,
        backgroundIndex=background,
        expectation=expectation,
        expectedSet=expected_set,
        expectedBackground=expected_background,
        invertedIndptr=pointers,
        invertedIndices=indices,
    )


@njit(cache=True, parallel=True)
def _score_block(
    counts: np.ndarray,
    annotation_indptr: np.ndarray,
    annotation_indices: np.ndarray,
    annotation_data: np.ndarray,
    inverted_indptr: np.ndarray,
    inverted_indices: np.ndarray,
    expected_set: np.ndarray,
    expected_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_peaks = counts.shape
    n_iterations, n_sets = expected_background.shape
    deviations = np.full((n_cells, n_sets), np.nan)
    scores = np.full((n_cells, n_sets), np.nan)
    for cell in prange(n_cells):
        observed = np.zeros(n_sets)
        sampled = np.zeros(n_iterations * n_sets)
        depth = 0.0
        for peak in range(n_peaks):
            value = np.float64(counts[cell, peak])
            if value == 0.0:
                continue
            depth += value
            for entry in range(annotation_indptr[peak], annotation_indptr[peak + 1]):
                observed[annotation_indices[entry]] += value * annotation_data[entry]
            for entry in range(inverted_indptr[peak], inverted_indptr[peak + 1]):
                sampled[inverted_indices[entry]] += value
        background = np.empty(n_iterations)
        for peak_set in range(n_sets):
            expected = expected_set[peak_set] * depth
            if not expected >= 1.0:
                continue
            observed_deviation = (observed[peak_set] - expected) / expected
            mean = 0.0
            low = np.inf
            high = -np.inf
            for iteration in range(n_iterations):
                expected_sample = expected_background[iteration, peak_set] * depth
                value = (
                    sampled[iteration * n_sets + peak_set] - expected_sample
                ) / expected_sample
                background[iteration] = value
                mean += value
                low = min(low, value)
                high = max(high, value)
            mean /= n_iterations
            spread = 0.0
            for iteration in range(n_iterations):
                difference = background[iteration] - mean
                spread += difference * difference
            sd = np.sqrt(spread / (n_iterations - 1))
            corrected = observed_deviation - mean
            deviations[cell, peak_set] = corrected
            # Equal background deviations have no spread, whatever rounding
            # leaves in the computed one.
            if high > low and sd > 0.0:
                scores[cell, peak_set] = corrected / sd
    return deviations, scores


def score_chromvar_block(
    counts: np.ndarray,
    model: ChromvarModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bias-corrected deviations and z-scores of a block of cells.

    Args:
        counts: Cells-by-peaks fragment counts in the model's peak order.
        model: Output of ``build_chromvar_model``.

    Returns:
        ``(deviations, z)``, each cells-by-sets. A cell whose expected count
        in a set is below one, and a z-score whose background deviations do
        not vary, are NaN.
    """
    values = np.asarray(counts)
    if values.ndim != 2 or values.shape[1] != model.annotation.shape[0]:
        raise ValueError("counts must be cells by the model's peaks")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("counts must be finite numbers")
    # Integer counts are always finite, and unsigned ones never negative.
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise ValueError("counts must be finite numbers")
    if values.dtype.kind in "fi" and np.any(values < 0):
        raise ValueError("counts must be non-negative")
    annotation = model.annotation
    return _score_block(
        np.ascontiguousarray(values),
        annotation.indptr,
        annotation.indices,
        annotation.data,
        model.invertedIndptr,
        model.invertedIndices,
        model.expectedSet,
        model.expectedBackground,
    )
