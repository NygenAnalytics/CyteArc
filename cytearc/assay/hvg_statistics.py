"""Streamed per-batch feature statistics for highly variable feature selection.

Batch-aware selection and the seurat_v3 and pearson_residuals flavors need
statistics of each batch rather than the persisted feature summary of all
selected cells. These functions stream the feature-major ``countsT`` bands of
the selected cells once per call and keep, for every batch and feature,
statistics that merge exactly across cell bands: detections and sums add,
centered sums of squares merge with the update of Chan, Golub, and LeVeque,
and maxima take the larger value. The bands of a feature group are merged in
band order with a fixed pairwise tree once all of them have arrived, so the
results do not depend on which thread finished first, and the group's
statistics are released.

Each selected cell belongs to one batch, given as a code from 0 to
``n_groups - 1`` aligned with the selected cells.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

from ..storage.geometry import array_geometry
from ..utils.compute import pairwise_merge_tree
from ..utils.moments import add_implicit_zeros, welford_add
from .normalization import library_size_divisors

if TYPE_CHECKING:
    from .rna import RNAassay

__all__ = [
    "BatchHvgScores",
    "GroupedMoments",
    "grouped_clipped_squares",
    "grouped_moments",
    "grouped_pearson_residual_moments",
]

# The statistics of one cell band of one feature group: the cells of each
# batch in the band, then arrays with a row per batch and a column per
# selected feature of the group.
type _BandStatistics = tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class BatchHvgScores:
    """The HVG scores of every feature in every batch.

    Attributes:
        sizes: The number of selected cells in each batch.
        scores: Each batch's score of each feature, one row per batch.
        mean_nonzero: Each batch's mean value over the cells where a feature
            is detected: normalized values for the cytearc flavor, counts for
            the others.
        detected: The number of selected cells where each feature is detected.
        pooled_mean_nonzero: The mean value of each feature over every
            selected cell where it is detected.
    """

    sizes: np.ndarray
    scores: np.ndarray
    mean_nonzero: np.ndarray
    detected: np.ndarray
    pooled_mean_nonzero: np.ndarray


@dataclass(frozen=True, slots=True)
class GroupedMoments:
    """Per-batch statistics of each feature, one row per batch.

    Attributes:
        sizes: The number of selected cells in each batch.
        detected: The number of cells with a positive value.
        total: The sum of the values.
        m2: The sum of squared deviations from the batch mean.
        maximum: The largest value, at least 0.
    """

    sizes: np.ndarray
    detected: np.ndarray
    total: np.ndarray
    m2: np.ndarray
    maximum: np.ndarray


@njit(cache=True, nogil=True)
def _band_moments_kernel(
    values: np.ndarray,
    value_rows: np.ndarray,
    local: np.ndarray,
    groups: np.ndarray,
    scale: float,
    inverse_totals: np.ndarray,
    sizes: np.ndarray,
    total: np.ndarray,
    m2: np.ndarray,
    detected: np.ndarray,
    maximum: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Write each batch's moments of ``scale * count * inverse_total``.

    Row ``value_rows[k]`` of ``values`` writes column ``k``. Stored counts
    update a running mean and ``m2`` per batch, and each batch's cells
    without a count then join as zeros, as the summary kernel of all selected
    cells does.
    """
    n_groups = sizes.shape[0]
    for k in range(value_rows.shape[0]):
        row = value_rows[k]
        for group in range(n_groups):
            means[group] = 0.0
            counts[group] = 0
        for cell in range(local.shape[0]):
            count = values[row, local[cell]]
            if count == 0:
                continue
            group = groups[cell]
            value = scale * np.float64(count) * inverse_totals[cell]
            detected[group, k] += value > 0.0
            total[group, k] += value
            if value > maximum[group, k]:
                maximum[group, k] = value
            counts[group] += 1
            means[group], m2[group, k] = welford_add(
                counts[group], means[group], m2[group, k], value
            )
        for group in range(n_groups):
            m2[group, k] = add_implicit_zeros(
                m2[group, k], means[group], counts[group], sizes[group]
            )


@njit(cache=True, nogil=True)
def _band_clipped_squares_kernel(
    values: np.ndarray,
    value_rows: np.ndarray,
    local: np.ndarray,
    groups: np.ndarray,
    means: np.ndarray,
    clips: np.ndarray,
    sizes: np.ndarray,
    out: np.ndarray,
    stored: np.ndarray,
) -> None:
    """Write each batch's sum of ``(min(count, clip) - mean)**2``.

    Row ``value_rows[k]`` of ``values`` writes column ``k``, with the means
    and clips of column ``k``. The cells of a batch without a count add
    ``mean**2`` each.
    """
    n_groups = sizes.shape[0]
    for k in range(value_rows.shape[0]):
        row = value_rows[k]
        for group in range(n_groups):
            stored[group] = 0
        for cell in range(local.shape[0]):
            count = values[row, local[cell]]
            if count == 0:
                continue
            group = groups[cell]
            deviation = min(np.float64(count), clips[group, k]) - means[group, k]
            out[group, k] += deviation * deviation
            stored[group] += 1
        for group in range(n_groups):
            mean = means[group, k]
            out[group, k] += (sizes[group] - stored[group]) * (mean * mean)


@njit(cache=True, nogil=True)
def _band_pearson_kernel(
    values: np.ndarray,
    value_rows: np.ndarray,
    local: np.ndarray,
    groups: np.ndarray,
    totals: np.ndarray,
    fractions: np.ndarray,
    theta: float,
    clips: np.ndarray,
    total: np.ndarray,
    m2: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Write each batch's sum and ``m2`` of clipped Pearson residuals.

    Row ``value_rows[k]`` of ``values`` writes column ``k``, with the
    fractions of column ``k``. A cell's expected count is its total times the
    feature's fraction in the cell's batch, and its residual
    ``(x - mu) / sqrt(mu + mu**2 / theta)`` is clipped to the batch's clip;
    an expected count of 0 gives a residual of 0. Every selected cell
    contributes, with or without a count.
    """
    n_groups = clips.shape[0]
    for k in range(value_rows.shape[0]):
        row = value_rows[k]
        for group in range(n_groups):
            means[group] = 0.0
            counts[group] = 0
        for cell in range(local.shape[0]):
            group = groups[cell]
            expected = totals[cell] * fractions[group, k]
            residual = 0.0
            if expected > 0.0:
                residual = (np.float64(values[row, local[cell]]) - expected) / np.sqrt(
                    expected + expected * expected / theta
                )
                residual = min(max(residual, -clips[group]), clips[group])
            total[group, k] += residual
            counts[group] += 1
            means[group], m2[group, k] = welford_add(
                counts[group], means[group], m2[group, k], residual
            )


def _merge_moments(first: _BandStatistics, second: _BandStatistics) -> _BandStatistics:
    """Merge the per-batch moments of two cell bands of one feature group.

    The statistics are the cells of each batch, the sums, the ``m2``, and
    optionally the detections and the maxima. Each batch merges its sums and
    ``m2`` as ``ColumnMoments.merge`` merges columns, so a batch without
    cells in a band keeps the other band's values exactly.
    """
    sizes_a, total_a, m2_a = first[:3]
    sizes_b, total_b, m2_b = second[:3]
    m2 = m2_a + m2_b
    for group in np.flatnonzero((sizes_a > 0) & (sizes_b > 0)):
        count_a, count_b = int(sizes_a[group]), int(sizes_b[group])
        merged = np.divide(total_b[group], count_b)
        merged -= total_a[group] / count_a
        np.square(merged, out=merged)
        merged *= count_a * count_b / (count_a + count_b)
        merged += m2_a[group]
        merged += m2_b[group]
        m2[group] = merged
    merged_statistics: _BandStatistics = (sizes_a + sizes_b, total_a + total_b, m2)
    if len(first) > 3:
        merged_statistics += (first[3] + second[3], np.maximum(first[4], second[4]))
    return merged_statistics


def _merge_sums(first: _BandStatistics, second: _BandStatistics) -> _BandStatistics:
    """Add the per-batch sums of two cell bands of one feature group."""
    return tuple(a + b for a, b in zip(first, second, strict=True))


def _stream_groups(
    assay: "RNAassay",
    cell_idx: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    feat_idx: np.ndarray,
    *,
    compute: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], _BandStatistics],
    merge: Callable[[_BandStatistics, _BandStatistics], _BandStatistics],
    outputs: tuple[np.ndarray, ...],
    resident_bytes: int,
    progress: str,
) -> None:
    """Stream the bands of the selected cells and features into ``outputs``.

    ``compute(band, value_rows, columns, band_groups)`` returns one band's
    statistics: the cells of each batch in the band, then one array per
    output with a row per batch and a column per entry of ``value_rows``, the
    rows of the band's values that hold selected features. ``columns`` are
    their columns in the outputs, which have a row per batch and a column per
    feature of ``feat_idx``. ``merge`` combines two bands' statistics.
    """
    from ..storage.feature_stream import map_feature_cell_bands, persisted_read_group

    counts_t = assay.rawDataT
    # An RNA assay opens only with a complete countsT.
    assert counts_t is not None
    n_feats = int(counts_t.shape[0])
    dest_of = np.full(n_feats, -1, dtype=np.int64)
    dest_of[feat_idx] = np.arange(len(feat_idx), dtype=np.int64)
    geometry = array_geometry(counts_t)
    assert geometry is not None
    cell_chunk = int(geometry.axisChunk(1))
    n_bands = int(np.unique(cell_idx // cell_chunk).size)
    widest = min(
        len(feat_idx), max(geometry.axisChunk(0), persisted_read_group(counts_t)[0])
    )
    # A band's statistics hold a row per batch, and a column index, for every
    # selected feature of its group. As in the summary of all selected cells,
    # every band of every group may wait for its group to complete; each
    # compute worker also holds one band's statistics and batch codes.
    feature_bytes = (len(outputs) * n_groups + 1) * 8
    scratch = (
        resident_bytes
        + dest_of.nbytes
        + sum(output.nbytes for output in outputs)
        + n_bands * (feature_bytes * len(feat_idx) + 8 * n_groups)
        + assay.resources.workers
        * (feature_bytes * widest + 16 * min(cell_chunk, len(cell_idx)))
    )
    pending: dict[tuple[int, int], list[tuple[int, _BandStatistics]]] = {}

    def process(band: Any) -> tuple[int, tuple[int, int], np.ndarray, _BandStatistics]:
        destinations = dest_of[band.featStart + band.featureRows()]
        value_rows = np.flatnonzero(destinations >= 0)
        columns = destinations[value_rows]
        statistics = compute(
            band, value_rows, columns, groups[band.selectedDestinations]
        )
        return int(band.unitIndex), (band.featStart, band.featEnd), columns, statistics

    for unit, key, columns, statistics in map_feature_cell_bands(
        counts_t,
        process,
        cell_idx=cell_idx,
        feat_idx=feat_idx,
        resources=assay.resources,
        progress=progress,
        io=getattr(assay, "storageIo", None),
        scratchBytes=scratch,
        orderedCompute=False,
    ):
        pending.setdefault(key, []).append((unit, statistics))
        if len(pending[key]) < n_bands:
            continue
        # Every band of a group selects the same features, so the last band's
        # columns are the group's.
        items = sorted(pending.pop(key), key=lambda item: item[0])
        merged = pairwise_merge_tree([item[1] for item in items], merge)
        for output, values in zip(outputs, merged[1:], strict=True):
            output[:, columns] = values
    # Every feature group has a band in every cell band of the selection.
    assert not pending


def _checked_inputs(
    cell_idx: np.ndarray, groups: np.ndarray, n_groups: int, feat_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the inputs as int64 arrays and the number of cells of each batch."""
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    feat_idx = np.asarray(feat_idx, dtype=np.int64)
    if cell_idx.ndim != 1 or groups.shape != cell_idx.shape:
        raise ValueError("groups must hold one batch code per selected cell")
    if groups.size and (int(groups.min()) < 0 or int(groups.max()) >= n_groups):
        raise ValueError(f"Batch codes must lie from 0 to {n_groups - 1}")
    sizes = np.bincount(groups, minlength=n_groups).astype(np.int64)
    return cell_idx, groups, feat_idx, sizes


def _band_sizes(band_groups: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(band_groups, minlength=n_groups).astype(np.int64)


def grouped_moments(
    assay: "RNAassay",
    cell_idx: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    feat_idx: np.ndarray,
    *,
    library_size: bool,
) -> GroupedMoments:
    """Return each batch's detections, sums, ``m2``, and maxima of the features.

    With ``library_size`` the values are those of ``norm_lib_size``,
    ``sf * count / total`` over each cell's prepared total, as the feature
    summary of all selected cells has them; otherwise they are the counts.

    Args:
        assay: An RNA assay, which normalizes with ``norm_lib_size`` and a
            size factor when ``library_size`` is set.
        cell_idx: The sorted, unique selected cells.
        groups: The batch code of each selected cell.
        n_groups: The number of batches.
        feat_idx: The features to summarize.
        library_size: Whether to summarize library-size normalized values.

    Returns:
        The statistics, each with a row per batch and a column per feature.
    """
    cell_idx, groups, feat_idx, sizes = _checked_inputs(
        cell_idx, groups, n_groups, feat_idx
    )
    shape = (n_groups, len(feat_idx))
    # The sums, m2, detections, and maxima.
    outputs = tuple(np.zeros(shape, dtype=np.float64) for _ in range(4))
    if library_size:
        assert assay.sf is not None
        scale = float(assay.sf)
        inverse_totals = library_size_divisors(
            assay._cell_count_totals(cell_idx), source=assay._totals_name, copy=False
        )
        np.reciprocal(inverse_totals, out=inverse_totals)
    else:
        scale = 1.0
        inverse_totals = np.ones(len(cell_idx), dtype=np.float64)

    def compute(
        band: Any, value_rows: np.ndarray, _columns: np.ndarray, band_groups: np.ndarray
    ) -> _BandStatistics:
        band_sizes = _band_sizes(band_groups, n_groups)
        total, m2, detected, maximum = (
            np.zeros((n_groups, len(value_rows)), dtype=np.float64) for _ in range(4)
        )
        _band_moments_kernel(
            band.values,
            value_rows,
            band.selectedLocal,
            band_groups,
            scale,
            inverse_totals[band.selectedDestinations],
            band_sizes,
            total,
            m2,
            detected,
            maximum,
            np.zeros(n_groups, dtype=np.float64),
            np.zeros(n_groups, dtype=np.int64),
        )
        return band_sizes, total, m2, detected, maximum

    if len(cell_idx) and len(feat_idx):
        _stream_groups(
            assay,
            cell_idx,
            groups,
            n_groups,
            feat_idx,
            compute=compute,
            merge=_merge_moments,
            outputs=outputs,
            resident_bytes=inverse_totals.nbytes + groups.nbytes,
            progress="Calculating batch feature statistics",
        )
    total, m2, detected, maximum = outputs
    return GroupedMoments(sizes, detected, total, m2, maximum)


def grouped_clipped_squares(
    assay: "RNAassay",
    cell_idx: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    feat_idx: np.ndarray,
    *,
    means: np.ndarray,
    clips: np.ndarray,
) -> np.ndarray:
    """Return each batch's sum of ``(min(count, clip) - mean)**2`` per feature.

    Args:
        assay: An RNA assay.
        cell_idx: The sorted, unique selected cells.
        groups: The batch code of each selected cell.
        n_groups: The number of batches.
        feat_idx: The features to summarize.
        means: Each batch's mean count of each feature of ``feat_idx``.
        clips: Each batch's clip of each feature of ``feat_idx``.

    Returns:
        The sums, with a row per batch and a column per feature.
    """
    cell_idx, groups, feat_idx, _sizes = _checked_inputs(
        cell_idx, groups, n_groups, feat_idx
    )
    means = np.asarray(means, dtype=np.float64)
    clips = np.asarray(clips, dtype=np.float64)
    squares = np.zeros((n_groups, len(feat_idx)), dtype=np.float64)

    def compute(
        band: Any, value_rows: np.ndarray, columns: np.ndarray, band_groups: np.ndarray
    ) -> _BandStatistics:
        band_sizes = _band_sizes(band_groups, n_groups)
        out = np.zeros((n_groups, len(value_rows)), dtype=np.float64)
        _band_clipped_squares_kernel(
            band.values,
            value_rows,
            band.selectedLocal,
            band_groups,
            np.ascontiguousarray(means[:, columns]),
            np.ascontiguousarray(clips[:, columns]),
            band_sizes,
            out,
            np.zeros(n_groups, dtype=np.int64),
        )
        return band_sizes, out

    if len(cell_idx) and len(feat_idx):
        _stream_groups(
            assay,
            cell_idx,
            groups,
            n_groups,
            feat_idx,
            compute=compute,
            merge=_merge_sums,
            outputs=(squares,),
            resident_bytes=means.nbytes + clips.nbytes + groups.nbytes,
            progress="Calculating clipped feature variances",
        )
    return squares


def grouped_pearson_residual_moments(
    assay: "RNAassay",
    cell_idx: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    feat_idx: np.ndarray,
    *,
    totals: np.ndarray,
    fractions: np.ndarray,
    theta: float,
    clips: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each batch's sum and ``m2`` of clipped Pearson residuals.

    Args:
        assay: An RNA assay.
        cell_idx: The sorted, unique selected cells.
        groups: The batch code of each selected cell.
        n_groups: The number of batches.
        feat_idx: The features to summarize.
        totals: The total count of each selected cell.
        fractions: Each batch's fraction of its counts in each feature of
            ``feat_idx``.
        theta: The negative binomial overdispersion.
        clips: The residual clip of each batch.

    Returns:
        The sums and ``m2``, each with a row per batch and a column per
        feature, over every cell of the batch.
    """
    cell_idx, groups, feat_idx, _sizes = _checked_inputs(
        cell_idx, groups, n_groups, feat_idx
    )
    totals = np.asarray(totals, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    clips = np.asarray(clips, dtype=np.float64)
    shape = (n_groups, len(feat_idx))
    outputs = (np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64))

    def compute(
        band: Any, value_rows: np.ndarray, columns: np.ndarray, band_groups: np.ndarray
    ) -> _BandStatistics:
        band_sizes = _band_sizes(band_groups, n_groups)
        total = np.zeros((n_groups, len(value_rows)), dtype=np.float64)
        m2 = np.zeros((n_groups, len(value_rows)), dtype=np.float64)
        _band_pearson_kernel(
            band.values,
            value_rows,
            band.selectedLocal,
            band_groups,
            totals[band.selectedDestinations],
            np.ascontiguousarray(fractions[:, columns]),
            float(theta),
            clips,
            total,
            m2,
            np.zeros(n_groups, dtype=np.float64),
            np.zeros(n_groups, dtype=np.int64),
        )
        return band_sizes, total, m2

    if len(cell_idx) and len(feat_idx):
        _stream_groups(
            assay,
            cell_idx,
            groups,
            n_groups,
            feat_idx,
            compute=compute,
            merge=_merge_moments,
            outputs=outputs,
            resident_bytes=totals.nbytes + fractions.nbytes + groups.nbytes,
            progress="Calculating Pearson residual variances",
        )
    return outputs
