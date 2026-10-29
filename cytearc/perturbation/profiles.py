"""Perturbation profiles: each perturbation's mean expression against control cells.

Replogle et al. (2022) compare perturbations by their mean expression
profiles, with each feature measured in standard deviations of the control
cells of the cell's own sample, such as its GEM group:

1. Each cell's counts are scaled so that its total, the sum of its counts
   over every feature of the assay, equals the median total of the control
   cells, ``x = count / total * target``.
2. Within each sample, each feature's scaled values are centred on the mean
   of the sample's control cells and divided by their standard deviation with
   the n - 1 divisor, ``z = (x - mean) / sd``, for every cell, control cells
   included.
3. A perturbation's profile is the mean z-score of its cells, whatever their
   sample, computed in float64.

This recipe reproduces the published pseudo-bulk profiles of Replogle et al.'s
K562 essential screen from its cells to float32 rounding. A feature whose
control cells in a sample hold one value has no spread there; its z-scores in
that sample are 0, the control mean, which is CyteArc's choice: the published
genome-wide profiles hold infinite values there. The choices of the recipe sit
in one place: :func:`depth_target` sets the total that every cell is scaled
to, and :data:`CONTROL_SD_DDOF` the divisor of the control cells' variance.

The kernels add the cells of a block one after another in cell order, each
thread taking a tile of features, so the profiles depend neither on how the
cells are split into blocks nor on the threads.

These functions read counts that a caller passes and never touch storage.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit, prange

from ..utils.arguments import integer_argument
from ..utils.moments import welford_add
from .efficiency import id_column_setting, rows_setting

__all__ = [
    "CONTROL_SD_DDOF",
    "PerturbationProfiles",
    "control_moments",
    "depth_target",
    "profile_settings",
    "profile_values",
    "z_score_sums",
]

#: The control cells' variance divides their centred sum of squares by their
#: number minus this, the n - 1 divisor of Replogle et al.'s profiles.
CONTROL_SD_DDOF = 1
# The features that one thread of a kernel takes at a time.
_TILE = 64


def profile_settings(
    by: object,
    min_cells: object,
    mask_targets: object,
    id_column: object,
    sample_column: object,
) -> int:
    """Check the settings of perturbation profiles, as ``profiles`` takes them.

    The pipeline recipe checks the same settings before its run starts.
    Whether the columns exist is checked against the store.

    Returns:
        ``min_cells`` as a Python integer.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If ``by`` is unknown, or ``min_cells`` is below 1.
    """
    rows_setting(by)
    floor = integer_argument(min_cells, "min_cells", minimum=1)
    if not isinstance(mask_targets, bool):
        raise TypeError("mask_targets must be a boolean")
    id_column_setting(id_column)
    if sample_column is not None and (
        not isinstance(sample_column, str) or not sample_column
    ):
        raise TypeError("sample_column must be a non-empty string or None")
    return floor


def depth_target(totals: np.ndarray, control: np.ndarray) -> float:
    """Return the total count that every cell is scaled to.

    It is the median total of the control cells, as Replogle et al. scale
    every cell to the median total of their control cells.

    Args:
        totals: Each cell's total counts over every feature of the assay.
        control: Whether each cell is a control cell.

    Returns:
        The median total of the control cells.

    Raises:
        ValueError: If no cell is a control cell.
    """
    flags = np.asarray(control, dtype=bool)
    if not flags.any():
        raise ValueError("There is no control cell to scale the cells to")
    return float(np.median(np.asarray(totals, dtype=np.float64)[flags]))


@njit(cache=True, nogil=True, parallel=True)
def _add_control_moments(
    counts: np.ndarray,
    totals: np.ndarray,
    target: float,
    samples: np.ndarray,
    seen: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
) -> None:
    """Add a block of control cells to each sample's running moments.

    ``seen`` counts each sample's cells before the block; each tile of
    features keeps its own running count, so the threads share nothing.
    """
    n_cells, n_features = counts.shape
    for tile in prange((n_features + _TILE - 1) // _TILE):
        first = tile * _TILE
        last = min(first + _TILE, n_features)
        count = seen.copy()
        for cell in range(n_cells):
            sample = samples[cell]
            count[sample] += 1
            total = totals[cell]
            for feature in range(first, last):
                value = counts[cell, feature] / total * target if total > 0 else 0.0
                means[sample, feature], m2[sample, feature] = welford_add(
                    count[sample], means[sample, feature], m2[sample, feature], value
                )


@njit(cache=True, nogil=True, parallel=True)
def _add_z_score_sums(
    counts: np.ndarray,
    totals: np.ndarray,
    target: float,
    samples: np.ndarray,
    groups: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    sums: np.ndarray,
) -> None:
    """Add each cell's z-scores to the sums of its perturbation."""
    n_cells, n_features = counts.shape
    for tile in prange((n_features + _TILE - 1) // _TILE):
        first = tile * _TILE
        last = min(first + _TILE, n_features)
        for cell in range(n_cells):
            sample = samples[cell]
            group = groups[cell]
            total = totals[cell]
            for feature in range(first, last):
                sd = sds[sample, feature]
                if sd > 0.0:
                    value = counts[cell, feature] / total * target if total > 0 else 0.0
                    sums[group, feature] += (value - means[sample, feature]) / sd


def _cell_blocks(
    blocks: Iterable[np.ndarray], n_cells: int, n_features: int
) -> Iterable[tuple[np.ndarray, int, int]]:
    """Yield each block as a C-ordered array with the cells it holds.

    Raises:
        ValueError: If a block does not hold ``n_features`` columns, or the
            blocks do not hold ``n_cells`` cells.
    """
    offset = 0
    for block in blocks:
        values = np.ascontiguousarray(block)
        if values.ndim != 2 or values.shape[1] != n_features:
            raise ValueError(
                f"Each block must hold cells x {n_features} features, not "
                f"{values.shape}"
            )
        stop = offset + len(values)
        if stop > n_cells:
            raise ValueError(f"The blocks hold more than the {n_cells} cells")
        yield values, offset, stop
        offset = stop
    if offset != n_cells:
        raise ValueError(f"The blocks hold {offset} of the {n_cells} cells")


def control_moments(
    blocks: Iterable[np.ndarray],
    totals: np.ndarray,
    target: float,
    samples: np.ndarray,
    n_samples: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each sample's control cells and the mean and spread of their features.

    Args:
        blocks: The control cells' counts, cells x features, in blocks of
            cells in the order of ``totals``.
        totals: Each control cell's total counts over every feature of the
            assay; a cell without counts has values of 0.
        target: The total that every cell is scaled to.
        samples: Each control cell's sample, from 0 to ``n_samples - 1``.
        n_samples: The samples.
        n_features: The features of each block.

    Returns:
        The control cells of each sample, and the mean and standard deviation
        of each feature's scaled values over them, samples x features. The
        standard deviation divides by the cells minus :data:`CONTROL_SD_DDOF`,
        and is 0 in a sample with too few cells for that divisor.

    Raises:
        ValueError: If the blocks do not hold one row per cell of ``totals``
            with ``n_features`` columns, or a sample is out of range.
    """
    sums = np.asarray(totals, dtype=np.float64)
    codes = np.asarray(samples, dtype=np.int64)
    if codes.shape != sums.shape or (
        len(codes) and (codes.min() < 0 or codes.max() >= n_samples)
    ):
        raise ValueError(
            f"samples must hold one sample from 0 to {n_samples - 1} per cell"
        )
    seen = np.zeros(n_samples, dtype=np.int64)
    means = np.zeros((n_samples, n_features))
    m2 = np.zeros((n_samples, n_features))
    for values, start, stop in _cell_blocks(blocks, len(sums), n_features):
        _add_control_moments(
            values, sums[start:stop], float(target), codes[start:stop], seen, means, m2
        )
        seen += np.bincount(codes[start:stop], minlength=n_samples)
    divisor = (seen - CONTROL_SD_DDOF).astype(np.float64)[:, None]
    sds = np.zeros((n_samples, n_features))
    np.divide(m2, divisor, out=sds, where=divisor > 0)
    np.sqrt(sds, out=sds)
    return seen, means, sds


def z_score_sums(
    blocks: Iterable[np.ndarray],
    totals: np.ndarray,
    target: float,
    samples: np.ndarray,
    groups: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Return the sum of each perturbation's z-scores over its cells.

    A feature whose standard deviation is 0 in a cell's sample adds 0.

    Args:
        blocks: The perturbed cells' counts, cells x features, in blocks of
            cells in the order of ``totals``.
        totals: Each cell's total counts over every feature of the assay.
        target: The total that every cell is scaled to.
        samples: Each cell's sample, a row of ``means`` and ``sds``.
        groups: Each cell's perturbation, from 0 to ``n_groups - 1``.
        means: Each sample's control mean of each feature.
        sds: Each sample's control standard deviation of each feature.
        n_groups: The perturbations.

    Returns:
        The sums, perturbations x features, as float64.

    Raises:
        ValueError: If the arrays do not align or a code is out of range.
    """
    sums = np.asarray(totals, dtype=np.float64)
    codes = np.asarray(samples, dtype=np.int64)
    rows = np.asarray(groups, dtype=np.int64)
    centre = np.ascontiguousarray(means, dtype=np.float64)
    spread = np.ascontiguousarray(sds, dtype=np.float64)
    if centre.ndim != 2 or spread.shape != centre.shape:
        raise ValueError("means and sds must both be samples x features")
    n_samples, n_features = centre.shape
    if codes.shape != sums.shape or rows.shape != sums.shape:
        raise ValueError("samples and groups must hold one entry per cell")
    if len(codes) and (
        codes.min() < 0
        or codes.max() >= n_samples
        or rows.min() < 0
        or rows.max() >= n_groups
    ):
        raise ValueError("A cell's sample or perturbation is out of range")
    out = np.zeros((n_groups, n_features))
    for values, start, stop in _cell_blocks(blocks, len(sums), n_features):
        _add_z_score_sums(
            values,
            sums[start:stop],
            float(target),
            codes[start:stop],
            rows[start:stop],
            centre,
            spread,
            out,
        )
    return out


def profile_values(
    sums: np.ndarray,
    n_cells: np.ndarray,
    masked_columns: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return each perturbation's mean z-score, with its masked feature at 0.

    Each sum is divided by the perturbation's cells in float64 and rounded to
    float32 as it is written, so no full-size copy of the sums is made.

    Args:
        sums: The sum of each perturbation's z-scores, perturbations x
            features.
        n_cells: Each perturbation's cells, at least 1.
        masked_columns: The column of each perturbation's own target, set to
            0, the control mean, or -1 to mask none.
        out: A float32 array of the shape of ``sums`` that receives the
            profiles, such as a band of columns of all the profiles, or None
            for a new array.

    Returns:
        The profiles, perturbations x features, as float32: ``out`` when it
        is given.

    Raises:
        ValueError: If the arrays do not align, a perturbation has no cell,
            a masked column is out of range, or ``out`` is not a float32
            array of the shape of ``sums``.
    """
    totals = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(n_cells, dtype=np.int64)
    columns = np.asarray(masked_columns, dtype=np.int64)
    if (
        totals.ndim != 2
        or counts.shape != (len(totals),)
        or columns.shape != (len(totals),)
    ):
        raise ValueError(
            "sums must be perturbations x features, with one cell count and one "
            "masked column per perturbation"
        )
    if (counts < 1).any():
        raise ValueError("Every perturbation needs at least one cell")
    if (columns >= totals.shape[1]).any() or (columns < -1).any():
        raise ValueError("A masked column is out of range")
    if out is None:
        out = np.empty(totals.shape, dtype=np.float32)
    elif (
        not isinstance(out, np.ndarray)
        or out.dtype != np.float32
        or out.shape != totals.shape
    ):
        raise ValueError("out must be a float32 array of the shape of sums")
    # NumPy divides in float64 and rounds each quotient to float32 through
    # its small buffers, as dividing and then converting would round it.
    np.divide(totals, counts[:, None], out=out, casting="same_kind")
    rows = np.flatnonzero(columns >= 0)
    out[rows, columns[rows]] = 0.0
    return out


@dataclass(frozen=True, slots=True)
class PerturbationProfiles:
    """Each perturbation's mean z-normalized expression.

    Attributes:
        values: One row per profiled perturbation, a target or a guide,
            indexed by its name or guide ID, and one column per feature, named
            by feature ID: the mean z-score of the perturbation's cells,
            float32.
        perturbations: One row per profiled perturbation, indexed as
            ``values``, with ``target``, its target, ``is_control``, whether
            it is a control guide, ``n_cells``, the cells averaged, and
            ``masked_feature``, the ID of its own target feature that was set
            to 0, or None.
        left_out: One row per perturbation of the labels with fewer than
            ``min_cells`` cells, with ``perturbation``, ``target`` and
            ``n_cells``.
        features: One row per feature, indexed by feature ID, with
            ``feature_name`` and ``n_zero_sd``, the samples whose control
            cells hold one value of the feature, where its z-scores are 0.
        samples: One row per sample, indexed by its label (None for one
            sample), with ``n_control_cells``.
        control_mean: Each sample's control mean of each feature, samples x
            features.
        control_sd: Each sample's control standard deviation of each
            feature, samples x features.
        by: ``"target"`` or ``"guide"``.
        from_assay: The expression assay.
        sample_column: The cell metadata column of the samples, or None.
        min_cells: The fewest cells a profiled perturbation has.
        mask_targets: Whether each perturbation's own target is set to 0.
        labels_kind: ``"perturbation_labels"``, or ``"mixscape_classes"`` when
            only the knocked-out cells of each perturbation that is not a
            control count.
    """

    values: pd.DataFrame = field(repr=False)
    perturbations: pd.DataFrame = field(repr=False)
    left_out: pd.DataFrame = field(repr=False)
    features: pd.DataFrame = field(repr=False)
    samples: pd.DataFrame = field(repr=False)
    control_mean: pd.DataFrame = field(repr=False)
    control_sd: pd.DataFrame = field(repr=False)
    by: str
    from_assay: str
    sample_column: str | None
    min_cells: int
    mask_targets: bool
    labels_kind: str
