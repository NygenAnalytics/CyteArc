"""Bulk aggregation of cell groups over stored or normalized matrices.

It also builds the bulk profiles that ``ds.compare.compute_bulk`` returns: the
cells of each bulk column, labelled by its group, sub-group and
pseudo-replicate, each column's mean or summed feature profile, and the
number of its cells that detect each feature.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import zarr
from numba import prange
from numpy.typing import NDArray

from ..assay import Assay, RNAassay, lib_size_feature_stream_eligible
from ..assay.normalization import library_size_divisors
from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..metadata.rows import read_metadata_rows_chunkwise
from ..storage.budget import ResourceBudget
from ..storage.feature_stream import FeatureCellBand, map_feature_cell_bands
from ..storage.geometry import array_geometry
from ..storage.io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy
from ..utils.arguments import integer_argument
from ..utils.numba import njit

BULK_AGGREGATIONS = ("mean", "sum")
BULK_FEATURE_LABELS = ("index", "id", "name")
# Features that one parallel task of a raw row aggregation adds.
_RAW_FEATURE_CHUNK = 256


@dataclass(frozen=True, slots=True, eq=False)
class BulkProfiles:
    """The bulk profiles of groups of cells, as ``compare.compute_bulk`` returns them.

    Each bulk column holds the cells of one group, of one group and
    sub-group, or of one pseudo-replicate of either, and is labelled by them:
    by its group label alone, or by a MultiIndex of the levels ``group``,
    ``subgroup``, and ``replicate`` (numbered from 1) that the call uses. The
    columns of ``profiles`` and ``detected`` and the rows of ``columns``
    carry the same labels, in the same order. Rows of ``profiles`` and
    ``detected`` are features, labelled as ``feature_label`` asks.

    Attributes:
        profiles: Each bulk column's mean or summed feature profile. A column
            without cells, kept with ``remove_empty_columns=False``, holds
            NaN.
        columns: One row per bulk column, with ``n_cells``, its number of
            cells, as int64.
        detected: The number of each bulk column's cells whose count of each
            feature is above zero, as int64, or None unless the call passed
            ``count_detected=True``.
    """

    profiles: pd.DataFrame
    columns: pd.DataFrame
    detected: pd.DataFrame | None

    @property
    def detected_fraction(self) -> pd.DataFrame:
        """The share of each column's cells that detect each feature.

        It divides ``detected`` by ``n_cells``, so a column without cells
        holds NaN.

        Raises:
            ValueError: If the cells that detect each feature were not
                counted.
        """
        if self.detected is None:
            raise ValueError(
                "The bulk profiles hold no detected counts; pass "
                "count_detected=True to compare.compute_bulk"
            )
        return self.detected.div(self.columns["n_cells"], axis=1)


@njit(cache=True, nogil=True, parallel=True, error_model="numpy")
def _accumulate_group_counts(
    raw: np.ndarray,
    selected: np.ndarray,
    group_codes: np.ndarray,
    scalars: np.ndarray | None,
    size_factor: float,
    values: np.ndarray,
    detected: np.ndarray | None,
) -> None:
    # Features own disjoint output entries; cell order stays fixed within each.
    for feature in prange(raw.shape[0]):  # type: ignore[no-untyped-call, attr-defined]
        for i in range(len(selected)):
            group = group_codes[i]
            count = raw[feature, selected[i]]
            if scalars is None:
                values[feature, group] += count
            else:
                values[feature, group] += np.float64(count) * size_factor / scalars[i]
            if detected is not None and count > 0:
                detected[feature, group] += 1


@njit(cache=True, nogil=True, parallel=True)
def _accumulate_group_rows(
    block: np.ndarray,
    group_codes: np.ndarray,
    sums: np.ndarray | None,
    positives: np.ndarray | None,
) -> None:
    """Add each row of ``block`` to its group's sums and positive-value counts.

    ``sums`` and ``positives`` are groups-by-features. Chunks of features run
    in parallel and own disjoint entries, and each entry adds the rows in
    order, so the results do not depend on the thread count.
    """
    n_rows, n_features = block.shape
    n_chunks = (n_features + _RAW_FEATURE_CHUNK - 1) // _RAW_FEATURE_CHUNK
    for chunk in prange(n_chunks):  # type: ignore[no-untyped-call, attr-defined]
        first = chunk * _RAW_FEATURE_CHUNK
        last = min(first + _RAW_FEATURE_CHUNK, n_features)
        for row in range(n_rows):
            group = group_codes[row]
            for feature in range(first, last):
                value = block[row, feature]
                if sums is not None:
                    sums[group, feature] += value
                if positives is not None and value > 0:
                    positives[group, feature] += 1


def _kernel_counts(values: np.ndarray) -> np.ndarray:
    """Return a block of counts with the same values in a dtype Numba reads.

    Numba reads native byte order and no float16; float32 holds every
    float16 value exactly.
    """
    dtype = values.dtype.newbyteorder("=")
    if dtype == np.float16:
        dtype = np.dtype(np.float32)
    return np.asarray(values, dtype=dtype)


def aggregate_raw_groups(
    raw: ChunkedArray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    sums: bool,
    positives: bool,
    nthreads: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sum each group's raw rows and count their positive values in one pass.

    ``raw`` holds one row per entry of ``group_codes``, the group of the row.
    Each group adds its rows in order: integer counts in the integer dtype
    that NumPy gives their sum, which is exact, and floating counts in
    float64.

    Args:
        raw: Raw counts with cells as rows.
        group_codes: The group of each row, from 0 to ``n_groups - 1``.
        n_groups: Number of groups.
        sums: Whether to sum the rows of each group.
        positives: Whether to count each group's positive values.
        nthreads: Worker count for streaming row blocks.

    Returns:
        Groups-by-features sums and int64 counts of positive values, each
        None when it is not requested.
    """
    codes = np.asarray(group_codes, dtype=np.int64)
    if codes.shape != (raw.shape[0],) or np.any((codes < 0) | (codes >= n_groups)):
        raise ValueError("Bulk group codes must align with raw rows")
    dtype = np.empty(0, dtype=raw.dtype).sum().dtype
    if dtype.kind == "f":
        dtype = np.dtype(np.float64)
    shape = (n_groups, raw.shape[1])
    totals = np.zeros(shape, dtype=dtype) if sums else None
    counts = np.zeros(shape, dtype=np.int64) if positives else None
    held = sum(array.nbytes for array in (totals, counts) if array is not None)
    start = 0
    for block in raw._stream_blocks(
        nthreads=nthreads,
        msg="Aggregating raw groups",
        prefetch=None,
        row_mask=None,
        resident_bytes=held + codes.nbytes,
    ):
        values = _kernel_counts(np.asarray(block))
        del block
        stop = start + len(values)
        _accumulate_group_rows(values, codes[start:stop], totals, counts)
        start = stop
        # The stream only reserves the blocks it reads, not one kept here.
        del values
    return totals, counts


def _require_finite_groups(
    values: np.ndarray,
    group_names: Sequence[Any] | None,
    *,
    statistic: str,
) -> None:
    """Raise unless every value of a groups-by-features result is finite."""
    if values.dtype.kind != "f":
        return
    finite = np.isfinite(values)
    if finite.all():
        return
    group, feature = (int(index) for index in np.argwhere(~finite)[0])
    label = repr(group_names[group]) if group_names is not None else str(group)
    raise ValueError(
        f"Bulk {statistic} of feature {feature} in group {label} is not finite"
    )


def aggregate_rna_groups(
    counts_t: zarr.Array,
    cell_indices: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    scalars: np.ndarray | None,
    size_factor: float,
    count_detected: bool,
    resources: ResourceBudget,
    io: StorageIoPolicy | None = None,
    group_names: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return features-by-groups count sums, or means normalized by ``scalars``.

    With ``count_detected``, also the int64 number of each group's cells
    whose count of each feature is above zero.
    """
    if group_codes.shape != cell_indices.shape or np.any(
        (group_codes < 0) | (group_codes >= n_groups)
    ):
        raise ValueError("Bulk group codes must align with selected cells")
    if scalars is not None:
        if scalars.shape != cell_indices.shape:
            raise ValueError(
                "Bulk normalization scalars must align with selected cells"
            )
        if not (np.isfinite(scalars) & (scalars > 0)).all():
            raise ValueError(
                "Bulk normalization scalars must be finite and positive; "
                "library_size_divisors maps a zero total to 1"
            )
    geometry = array_geometry(counts_t)
    assert geometry is not None
    n_features = int(counts_t.shape[0])
    # Raw sums are exact: integer counts add in the integer dtype NumPy gives
    # their sum, floating counts in float64. Means are float64.
    dtype = np.empty(0, dtype=counts_t.dtype).sum().dtype
    if scalars is not None or dtype.kind == "f":
        dtype = np.dtype(np.float64)
    band_cells = min(int(counts_t.shape[1]), geometry.axisChunk(1))
    output_bytes = n_features * n_groups * (dtype.itemsize + 8 * count_detected)
    # Include DataFrame construction/filter copies and selected-cell bookkeeping.
    resident_bytes = 4 * output_bytes + 64 * len(cell_indices) + 16 * n_groups
    # Reserve band-local group codes and normalization scalars.
    scratch_bytes = resident_bytes + min(len(cell_indices), band_cells) * (
        group_codes.dtype.itemsize + (0 if scalars is None else scalars.dtype.itemsize)
    )

    def accumulate(band: FeatureCellBand) -> None:
        columns = slice(band.featStart, band.featEnd)
        _accumulate_group_counts(
            band.values,
            band.selectedLocal,
            group_codes[band.selectedDestinations],
            None if scalars is None else scalars[band.selectedDestinations],
            size_factor,
            values[:, columns].T,
            None if detected is None else detected[:, columns].T,
        )

    # The stream plans when it is created, so a budget that cannot also hold
    # the output fails before the output is allocated.
    bands = map_feature_cell_bands(
        counts_t,
        accumulate,
        cell_idx=cell_indices,
        resources=resources,
        io=replace(io or DEFAULT_STORAGE_IO_POLICY, computeWorkers=1),
        progress="Aggregating RNA groups",
        scratchBytes=scratch_bytes,
        orderedCompute=True,
    )
    values = np.zeros((n_groups, n_features), dtype=dtype)
    detected = (
        np.zeros((n_groups, n_features), dtype=np.int64) if count_detected else None
    )
    group_sizes = np.bincount(group_codes, minlength=n_groups)
    for _ in bands:
        pass
    if scalars is not None:
        values /= np.maximum(group_sizes, 1)[:, None]
    _require_finite_groups(
        values, group_names, statistic="sum" if scalars is None else "mean"
    )
    return values.T, None if detected is None else detected.T


def aggregate_normalized_groups(
    normalized: ChunkedArray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    nthreads: int,
    group_names: Sequence[Any] | None = None,
) -> np.ndarray:
    """Average normalized rows within each cell group in one streaming pass.

    ``normalized`` holds one row per entry of ``group_codes``. Its
    normalization was fitted once over all of its rows, so a group's profile
    does not depend on which other groups are requested. Rows coded ``-1``
    contribute to the fit but to no group.

    Args:
        normalized: Lazy normalized matrix with cells as rows.
        group_codes: Group code of each row, or ``-1``.
        n_groups: Number of groups.
        nthreads: Worker count for streaming row blocks.
        group_names: Optional group names used in errors.

    Returns:
        A features-by-groups array of means. Empty groups are zero.
    """
    codes = np.asarray(group_codes, dtype=np.int64)
    if codes.shape != (normalized.shape[0],) or np.any(
        (codes < -1) | (codes >= n_groups)
    ):
        raise ValueError("Bulk group codes must align with normalized rows")
    sums = np.zeros((n_groups, normalized.shape[1]), dtype=np.float64)
    rows, n_features = normalized.chunksize
    # The sums stay for the whole pass. A block is copied to float64 unless it
    # already is, and once more for the rows of one group at a time.
    block_copies = 1 if normalized.dtype == np.float64 else 2
    start = 0
    for block in normalized._stream_blocks(
        nthreads=nthreads,
        msg="Aggregating normalized groups",
        prefetch=None,
        row_mask=None,
        resident_bytes=sums.nbytes + (block_copies * rows + 1) * n_features * 8,
    ):
        values = np.asarray(block, dtype=np.float64)
        del block
        block_codes = codes[start : start + len(values)]
        for code in np.unique(block_codes[block_codes >= 0]):
            sums[code] += values[block_codes == code].sum(axis=0)
        start += len(values)
        # The stream only reserves the blocks it reads, not one kept here.
        del values
    sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
    sums /= np.maximum(sizes, 1)[:, None]
    _require_finite_groups(sums, group_names, statistic="mean")
    return sums.T


def validate_bulk_options(
    *,
    aggr_type: object,
    feature_label: object,
    pseudo_reps: object,
) -> int:
    """Check the options of a bulk aggregation before any cell is read.

    Returns:
        ``pseudo_reps`` as a Python integer.

    Raises:
        TypeError: If ``pseudo_reps`` is not an integer.
        ValueError: If ``aggr_type`` is not ``"mean"`` or ``"sum"``,
            ``feature_label`` is not ``"index"``, ``"id"``, or ``"name"``, or
            ``pseudo_reps`` is less than 1.
    """
    if aggr_type not in BULK_AGGREGATIONS:
        raise ValueError("ERROR: `aggr_type` can only be either 'sum' or 'mean'")
    if feature_label not in BULK_FEATURE_LABELS:
        raise ValueError(
            f"feature_label must be 'index', 'id', or 'name'; got {feature_label!r}"
        )
    return integer_argument(pseudo_reps, "pseudo_reps", minimum=1)


def _bulk_replicates(cells: NDArray[Any], n_reps: int, seed: int) -> list[NDArray[Any]]:
    """Split one group's cells into ``n_reps`` pseudo-replicates.

    Each group draws its permutation from a new ``RandomState(seed)``, and
    each replicate lists its cells in ascending order.
    """
    v_list = list(cells)
    random_state = np.random.RandomState(seed)
    shuffled_idx = random_state.choice(v_list, len(v_list), replace=False)
    rep_idx = np.array_split(shuffled_idx, n_reps)
    return [np.array(sorted(x)) for x in rep_idx]


def _index_label(value: Any) -> Any:
    """Return a label as the Python value that a pandas index holds, such as an int for a NumPy int64."""
    return value.item() if isinstance(value, np.generic) else value


def bulk_column_rows(
    group_values: NDArray[Any],
    group_order: Sequence[Any],
    labelled: NDArray[np.bool_],
    cell_idx: NDArray[np.int64],
    *,
    secondary: tuple[NDArray[Any], Sequence[Any]] | None,
    null_vals: Sequence[Any] | None,
    secondary_null_vals: Sequence[Any] | None,
    pseudo_reps: int,
    random_seed: int,
) -> tuple[pd.Index, list[NDArray[Any]]]:
    """Return the labels of each bulk column and the global cell indices of its cells.

    Every combination of a group with a sub-group, and every pseudo-replicate,
    gets a column, so a column holds no cells when a group has no cells of a
    sub-group or fewer cells than ``pseudo_reps``. A column is labelled by
    its group alone, or by a MultiIndex of the levels ``group``,
    ``subgroup``, and ``replicate`` (numbered from 1) that the call uses, so
    labels never join into a name that two columns could share.
    """
    if null_vals is None:
        null_vals = []
    if secondary_null_vals is None:
        secondary_null_vals = []
    level_names = ["group"]
    if secondary is not None:
        level_names.append("subgroup")
    if pseudo_reps > 1:
        level_names.append("replicate")
    levels: dict[str, list[Any]] = {name: [] for name in level_names}
    column_rows: list[NDArray[Any]] = []
    subgroups: Sequence[Any] = [None] if secondary is None else secondary[1]
    for g in group_order:
        if g in null_vals:
            continue
        in_group = (group_values == g) & labelled
        for sg in subgroups:
            if secondary is None:
                selected = in_group
            elif sg in secondary_null_vals:
                continue
            else:
                selected = in_group & (secondary[0] == sg)
            replicates = _bulk_replicates(
                cell_idx[np.flatnonzero(selected)], pseudo_reps, random_seed
            )
            for number, rows in enumerate(replicates, start=1):
                levels["group"].append(_index_label(g))
                if secondary is not None:
                    levels["subgroup"].append(_index_label(sg))
                if pseudo_reps > 1:
                    levels["replicate"].append(number)
                column_rows.append(rows)
    if len(level_names) == 1:
        return pd.Index(levels["group"], name="group"), column_rows
    index = pd.MultiIndex.from_arrays(
        [levels[name] for name in level_names], names=level_names
    )
    return index, column_rows


def bulk_codes(
    cell_idx: NDArray[np.int64],
    column_rows: Sequence[NDArray[Any]],
) -> NDArray[np.int64]:
    """Return the bulk column of each selected cell, or -1 for cells in none.

    Codes number the columns in the order of ``column_rows``.
    """
    codes = np.full(len(cell_idx), -1, dtype=np.int64)
    for code, idx in enumerate(column_rows):
        codes[np.searchsorted(cell_idx, idx)] = code
    return codes


def bulk_read_cells(
    assay: Assay,
    aggr_type: str,
    cell_idx: NDArray[np.int64],
    column_rows: Sequence[NDArray[Any]],
) -> NDArray[np.int64]:
    """Return the global indices of the cells that ``aggregate_bulk_profiles`` reads."""
    if not column_rows or len(cell_idx) == 0:
        return np.zeros(0, dtype=np.int64)
    stream_rna = isinstance(assay, RNAassay) and (
        aggr_type == "sum"
        or (aggr_type == "mean" and lib_size_feature_stream_eligible(assay))
    )
    if stream_rna or aggr_type == "sum":
        return cell_idx[bulk_codes(cell_idx, column_rows) >= 0]
    if aggr_type == "mean":
        return cell_idx
    return np.zeros(0, dtype=np.int64)


def aggregate_bulk_profiles(
    assay: Assay,
    column_rows: Sequence[NDArray[Any]],
    cell_idx: NDArray[np.int64],
    *,
    aggr_type: str,
    count_detected: bool,
    cells: MetaData,
    resources: ResourceBudget,
    nthreads: int,
    column_names: Sequence[Any] | None = None,
) -> tuple[list[NDArray[Any]], list[NDArray[np.int64]] | None]:
    """Return each bulk column's feature profile, and on request its detecting cells.

    Every column of ``column_rows`` holds at least one cell. With
    ``count_detected``, the second list holds, for each column, the int64
    number of its cells whose raw count of each feature is above zero;
    without it, the second value is None. ``column_names`` name the columns
    in errors.
    """
    n_columns = len(column_rows)
    profiles: list[NDArray[Any]] = []
    detected: list[NDArray[np.int64]] | None = [] if count_detected else None
    codes = bulk_codes(cell_idx, column_rows)
    stream_rna = isinstance(assay, RNAassay) and (
        aggr_type == "sum"
        or (aggr_type == "mean" and lib_size_feature_stream_eligible(assay))
    )
    if stream_rna and column_rows:
        assert isinstance(assay, RNAassay)
        included = codes >= 0
        selected = cell_idx[included]
        # The divisors of ``normed``: a cell without counts averages in as
        # zeros rather than dividing zero by zero.
        scalars = (
            library_size_divisors(
                read_metadata_rows_chunkwise(cells, f"{assay.name}_nCounts", selected),
                source=f"{assay.name}_nCounts",
            )
            if aggr_type == "mean"
            else None
        )
        # An RNA assay opens only with a complete countsT and a size factor.
        values, counts = aggregate_rna_groups(
            cast(zarr.Array, assay.rawDataT),
            selected,
            codes[included],
            n_columns,
            scalars=scalars,
            size_factor=cast(float, assay.sf),
            count_detected=count_detected,
            resources=resources,
            io=assay.storageIo,
            group_names=column_names,
        )
        profiles = [values[:, i] for i in range(n_columns)]
        if counts is not None:
            detected = [counts[:, i] for i in range(n_columns)]
    else:
        if aggr_type not in ("sum", "mean"):
            raise ValueError("ERROR: `aggr_type` can only be either 'sum' or 'mean'")
        if aggr_type == "mean" and column_rows and len(cell_idx):
            # Fit the normalization once over every selected cell.
            means = aggregate_normalized_groups(
                assay.normed(
                    cell_idx=cell_idx,
                    feat_idx=np.arange(assay.feats.N, dtype=np.int64),
                ),
                codes,
                n_columns,
                nthreads=nthreads,
                group_names=column_names,
            )
            profiles = [means[:, i] for i in range(n_columns)]
        if column_rows and (aggr_type == "sum" or count_detected):
            # One pass over the raw rows of the selected cells sums every
            # column and counts its positive values. Raw sums are exact:
            # integer counts keep NumPy's integer accumulator.
            included = codes >= 0
            sums, positives = aggregate_raw_groups(
                assay.rawData[cell_idx[included]],
                codes[included],
                n_columns,
                sums=aggr_type == "sum",
                positives=count_detected,
                nthreads=nthreads,
            )
            if sums is not None:
                profiles = [sums[i] for i in range(n_columns)]
            if positives is not None:
                detected = [positives[i] for i in range(n_columns)]
    return profiles, detected


def aligned_feature_labels(values: np.ndarray, index: pd.Index) -> np.ndarray:
    """Return feature labels as a hashable NumPy array aligned to ``index``."""
    labels = np.asarray(values, dtype=object).reshape(-1)
    return labels[np.asarray(index, dtype=np.intp)]


def bulk_profiles(
    profiles: Sequence[NDArray[Any]],
    detected: Sequence[NDArray[np.int64]] | None,
    features: MetaData,
    *,
    columns: pd.Index,
    n_cells: NDArray[np.int64],
    remove_empty_features: bool,
    remove_empty_columns: bool,
    feature_label: str,
) -> BulkProfiles:
    """Return the bulk profiles, labelled by feature and by bulk column, as a record.

    ``columns`` labels every bulk column and ``n_cells`` gives each one's
    number of cells, in the same order; ``profiles`` and ``detected`` hold
    the profiles and detected counts of the columns with cells, in that
    order. Columns without cells are left out, or with
    ``remove_empty_columns=False`` kept, with NaN profiles and no detecting
    cells. ``remove_empty_features`` leaves out the features whose profile
    sums to zero over the returned columns, in ``detected`` too.
    """
    counts = np.asarray(n_cells, dtype=np.int64)
    if counts.shape != (len(columns),):
        raise ValueError("Bulk cell counts must align with the bulk columns")
    filled = counts > 0
    kept = np.flatnonzero(filled | (not remove_empty_columns))
    # The position of each column with cells among the aggregated profiles.
    aggregated = np.cumsum(filled) - 1

    def frame(arrays: Sequence[NDArray[Any]], empty: NDArray[Any]) -> pd.DataFrame:
        values = pd.DataFrame(
            {
                position: arrays[aggregated[position]] if filled[position] else empty
                for position in kept
            }
        )
        values.columns = columns[kept]
        return values

    profile_frame = frame(profiles, np.full(features.N, np.nan))
    detected_frame = (
        None
        if detected is None
        else frame(detected, np.zeros(features.N, dtype=np.int64))
    )
    if remove_empty_features:
        expressed = profile_frame.sum(axis=1) != 0
        profile_frame = profile_frame.loc[expressed]
        if detected_frame is not None:
            detected_frame = detected_frame.loc[expressed]
    if feature_label in ("id", "name"):
        profile_frame = profile_frame.set_index(
            aligned_feature_labels(
                features.fetch_all("ids" if feature_label == "id" else "names"),
                profile_frame.index,
            )
        )
    if detected_frame is not None:
        detected_frame.index = profile_frame.index
    return BulkProfiles(
        profiles=profile_frame,
        columns=pd.DataFrame({"n_cells": counts[kept]}, index=columns[kept]),
        detected=detected_frame,
    )
