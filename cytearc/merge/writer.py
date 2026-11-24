from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr
from scipy.sparse import coo_matrix

from ..assay.rna import RNAassay, library_size_factor
from ..matrix.chunked import ChunkedArray
from ..metadata.rows import operation_concurrency
from ..storage.budget import ResourceBudget
from ..storage.identity import (
    CountSummary,
    finalize_counts,
    row_nonzero_counts,
)
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.geometry import array_geometry
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import StorageProfile
from ..storage.layout import array_shard_rows
from ..storage.partition import affordable_width
from ..storage.schema import (
    TF_DENOMINATOR_ATTR,
    _assay_paths,
    create_zarr_count_assay,
    load_count_array,
    merged_normalized_values,
)
from ..storage.sharding import (
    SparseImportPlan,
    accumulate_sparse_to_shards,
    fit_count_layout,
    plan_dense_write,
    resolve_sparse_import_batch,
    row_band_task_count,
    sparse_counts_admission,
    write_counts_t,
    write_dense_from_row_batches,
)
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.arrays import (
    canonicalize_sparse,
    checked_sparse_cast,
    csr_from_dense,
    fill_dense_from_csr,
    max_window_nnz,
)
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from ..utils.memory_budget import MemoryBudgetError, memory_refusal
from ..utils.progress import iter_progress
from .features import FeatureAlignment
from .row_plan import (
    WINDOW_BYTES_PER_ROW,
    RowPlan,
    RowPlanWindow,
    iter_row_plan_windows,
)


CountsTReuseOutcome = Literal["reusable", "incomplete", "invalid"]


@dataclass(frozen=True, slots=True)
class CountsTReuseAssessment:
    """Structured merge decision for an existing ``countsT`` component."""

    outcome: CountsTReuseOutcome
    reason: str | None = None


def _matrix_group_path(assay_name: str, workspace: str | None) -> str:
    logical, matrix = _assay_paths(assay_name, workspace)
    return logical if matrix is None else matrix


def _assay_metadata_path(assay_name: str, workspace: str | None) -> str:
    return _assay_paths(assay_name, workspace)[0]


@dataclass(frozen=True, slots=True)
class _MergeImportRequirements:
    maxWindowNnz: Callable[[int], int]
    sourceDtype: np.dtype[Any]
    residentBytes: int
    extraProducerBytes: Callable[[int], int]
    # The bytes of one source row of a batch, which extraProducerBytes counts
    # for each row beside the remap's staging and a read's decode.
    sourceRowBytes: int
    # The most nonzero values that a stored shard of each source holds, and
    # those of each source's head, 0 for a source without the assay.
    shardValues: list[int]
    headValues: list[int]


def remap_block_to_coo(
    block: Any,
    order_map: np.ndarray,
    n_feats: int,
    nthreads: int,
    destination_dtype: np.dtype[Any],
) -> coo_matrix:
    """Dense-or-chunked block to COO with feature remapping and summation."""
    computed = controlled_compute(block, nthreads)
    source = coo_matrix(computed)
    mapped = coo_matrix(
        (source.data, (source.row, order_map[source.col])),
        shape=(computed.shape[0], n_feats),
    )
    return canonicalize_sparse(mapped, destination_dtype)


# Column maps of fewer runs than this are copied run by run.
_SCATTER_RUNS = 64


def scatter_block_dense(
    block: np.ndarray,
    order_map: np.ndarray,
    n_feats: int,
    destination_dtype: np.dtype[Any],
    *,
    out: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Place source columns at their merged positions in rows ``rows`` of dense ``out``.

    ``order_map`` must give every source column its own merged column, of
    the ``n_feats`` columns of ``out``. Values are checked and cast to
    ``destination_dtype`` as ``remap_block_to_coo`` casts its coordinates;
    as these hold no zero, the caller turns the negative zeros of ``out`` to
    zeros once it holds every row.
    """
    values = np.asarray(block)
    if np.can_cast(values.dtype, destination_dtype, casting="safe"):
        values = values.astype(destination_dtype, copy=False)
    else:
        values = checked_sparse_cast(values, destination_dtype)
    target = np.asarray(rows)
    # Merged columns mostly follow each other in runs, which copy as slices.
    breaks = np.flatnonzero(np.diff(order_map) != 1) + 1
    if breaks.size < _SCATTER_RUNS:
        for start, stop in zip(
            np.r_[0, breaks].tolist(), np.r_[breaks, order_map.size].tolist()
        ):
            first = int(order_map[start])
            out[target, first : first + stop - start] = values[:, start:stop]
    else:
        out[target[:, None], order_map] = values


def shared_normalization(assay_name: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    """Return the attributes that tell how a combined assay normalizes, as its sources do.

    A combined assay keeps the size factor that its sources normalize with,
    the session's ``sf`` of each source or, when a session sets none, the one
    that the source stores (:func:`~cytearc.assay.rna.library_size_factor`),
    and the ``tfDenominator`` of melded TF-IDF values, which an RNA class
    passes through, so that it normalizes their values as they do. Sources
    whose class normalizes without a size factor, such as ADT assays, give
    none, whatever their sessions set.

    Args:
        assay_name: The combined assay, which the errors name.
        sources: Each source's assay, by the source's name.

    Raises:
        ValueError: If a session's ``sf`` of a source is not a finite positive
            number, two sources normalize with different size factors, or
            some hold melded TF-IDF values and others do not.
    """
    factors: dict[str, float] = {}
    for name, source in sources.items():
        if not isinstance(source, RNAassay):
            continue
        try:
            factors[name] = library_size_factor(source)
        except ValueError as error:
            raise ValueError(f"Source {name!r}: {error}") from error
    if len(set(factors.values())) > 1:
        listed = ", ".join(f"{name!r} {factor!r}" for name, factor in factors.items())
        raise ValueError(
            f"Sources normalize assay {assay_name!r} with different size factors: "
            f"{listed}. A combined assay keeps the one size factor that its sources "
            f"share; set it on each source's assay, such as ds.{assay_name}.sf = "
            "1000, then combine them again."
        )
    denominators = {
        name: source.attrs.get(TF_DENOMINATOR_ATTR) for name, source in sources.items()
    }
    if len(set(denominators.values())) > 1:
        melded = [name for name, value in denominators.items() if value is not None]
        others = [name for name, value in denominators.items() if value is None]
        raise ValueError(
            f"Sources of assay {assay_name!r} hold different values: melded TF-IDF "
            f"values in {', '.join(map(repr, melded))} and other values in "
            f"{', '.join(map(repr, others))}. A combined assay normalizes all of "
            "its values one way, so combine assays that hold one kind of values."
        )
    kept: dict[str, Any] = {}
    if factors:
        kept["size_factor"] = next(iter(factors.values()))
    denominator = next(iter(denominators.values()), None)
    if denominator is not None:
        kept[TF_DENOMINATOR_ATTR] = denominator
    return kept


def create_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    alignment: FeatureAlignment,
    dtype: str,
    *,
    profile: StorageProfile,
    policy: CountMatrixPolicy,
    sources: Sequence[Any],
    normalization: Mapping[str, Any],
) -> zarr.Array:
    """Create the counts of a combined assay, which holds normalized values when a source does.

    ``normalization`` holds the attributes that its sources share, from
    :func:`shared_normalization`, which the assay keeps.
    """
    counts = create_zarr_count_assay(
        z=root,
        assay_name=assay_name,
        workspace=workspace,
        n_cells=n_cells,
        feat_ids=np.array(alignment.mergedFeatsMap["ids"]),
        feat_names=np.array(alignment.mergedFeatsMap["names"]),
        dtype=dtype,
        profile=profile,
        policy=policy,
        normalized_values=merged_normalized_values(
            source.normalizedValues for source in sources
        ),
    )
    if normalization:
        path = _assay_metadata_path(assay_name, workspace)
        as_zarr_group(root[path], name=path).attrs.update(dict(normalization))
    matrix_group = as_zarr_group(
        root[_matrix_group_path(assay_name, workspace)],
        name=_matrix_group_path(assay_name, workspace),
    )
    matrix_group.attrs["complete"] = False
    return counts


def _count_scratch_bytes(columns: int) -> int:
    """Return what ``count_nonzero`` makes of each row of a block of ``columns`` columns.

    That is a boolean copy of the row and the row's count.
    """
    return int(columns) * np.dtype(np.bool_).itemsize + np.dtype(np.intp).itemsize


def _column_run(counts: ChunkedArray, start: int, stop: int) -> ChunkedArray:
    """Return the columns ``start`` to ``stop`` of ``counts``, which are all of them or a slice."""
    return (
        counts if (start, stop) == (0, int(counts.shape[1])) else counts[:, start:stop]
    )


def _column_runs(counts: ChunkedArray, width: int) -> list[ChunkedArray]:
    """Return ``counts`` split into runs of ``width`` columns, the last of the rest."""
    n_columns = int(counts.shape[1])
    return [
        _column_run(counts, start, min(n_columns, start + width))
        for start in range(0, n_columns, width)
    ]


def _count_read_runs(counts: ChunkedArray, room: int) -> list[ChunkedArray] | None:
    """Return runs of whole chunk columns of ``counts`` whose streams fit in ``room`` bytes.

    The runs are the widest that fit, down to one chunk column, and each is
    streamed in whole blocks of a stored band, so that the read decodes each
    stored chunk once however small ``room`` is. Counts without rows or
    columns need no run; None when a run of one chunk column does not fit.
    """
    n_rows, n_columns = (int(value) for value in counts.shape)
    if n_rows == 0 or n_columns == 0:
        return []
    width = max(1, counts.chunk_columns())
    for chunks in range(-(-n_columns // width), 0, -1):
        columns = min(n_columns, chunks * width)
        run = _column_run(counts, 0, columns)
        if run.stream_bytes(_count_scratch_bytes(columns)) <= room:
            return _column_runs(counts, columns)
    return None


def _smallest_count_read_bytes(counts: ChunkedArray) -> int:
    """Return what the smallest read of ``counts`` holds.

    That is a stored band of one chunk column, its decode and what
    ``count_nonzero`` makes of it, or nothing for counts without rows or
    columns.
    """
    n_rows, n_columns = (int(value) for value in counts.shape)
    if n_rows == 0 or n_columns == 0:
        return 0
    columns = min(n_columns, max(1, counts.chunk_columns()))
    return _column_run(counts, 0, columns).stream_bytes(_count_scratch_bytes(columns))


def _row_nnz_load_bytes(assay: Any, source: ChunkedArray | zarr.Array) -> int:
    """Bound what taking one source's nonzero counts per row holds beside the profile.

    ``source`` is the summary that gives the counts, which is read whole, as
    its geometry charges such a read, or else the reader of the counts, which
    holds a count of each row beside the smallest read of the counts.
    """
    if isinstance(source, ChunkedArray):
        n_rows = int(assay.rawData.shape[0])
        return np.dtype(np.int64).itemsize * n_rows + _smallest_count_read_bytes(source)
    geometry = array_geometry(source)
    # A Zarr array always has a chunk layout.
    assert geometry is not None
    return geometry.wholeReadBytes(operation_concurrency())


def _counting_refusal(
    assay_name: str | None, item: str, needed_bytes: int, budget_bytes: int
) -> MemoryBudgetError:
    """Return the refusal of the step that counts the nonzero values of each merged cell."""
    of_assay = "" if assay_name is None else f" of assay {assay_name!r}"
    return memory_refusal(
        f"Counting the nonzero values of each merged cell{of_assay}",
        item,
        needed_bytes,
        budget_bytes,
    )


def _read_item(source: str) -> str:
    """Name what a read of a source's counts holds, for a refusal."""
    return f"each merged cell's count and a read of the counts of source {source!r}"


def _counted_row_values(
    assay: Any,
    source: str,
    reader: ChunkedArray,
    resources: ResourceBudget,
    *,
    held_bytes: int,
) -> np.ndarray:
    """Count the nonzero values of each row of a source's counts in a read.

    ``reader`` plans the read within the merge's ``resources``, beside
    ``held_bytes``, what the merge holds while it reads, and the counts that
    it returns. The read streams the widest runs of whole chunk columns that
    fit (:func:`_count_read_runs`), so it decodes each stored chunk once.

    Raises:
        MemoryBudgetError: If a stored band of one chunk column does not fit.
    """
    row_values = np.zeros(int(reader.shape[0]), dtype=np.int64)
    resident = int(held_bytes) + row_values.nbytes
    budget = int(resources.memoryBytes)
    runs = _count_read_runs(reader, budget - resident)
    if runs is None:
        raise _counting_refusal(
            assay.name,
            _read_item(source),
            resident + _smallest_count_read_bytes(reader),
            budget,
        )
    for run in runs:
        start = 0
        for block in run._stream_blocks(
            nthreads=resources.workers,
            msg=f"Counting the nonzero values of {assay.name} in source {source}",
            prefetch=None,
            row_mask=None,
            resident_bytes=resident,
            scratch_bytes_per_row=_count_scratch_bytes(run.shape[1]),
        ):
            stop = start + int(block.shape[0])
            row_values[start:stop] += np.count_nonzero(block, axis=1)
            start = stop
            # Release the block before the stream reads the next one.
            del block
    return row_values


def _cumulative_row_nnz(
    assays: list[Any | None],
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Return cumulative nonzero counts over merged rows, the most that a stored shard of each source holds, and what each source's head holds.

    A source's head is the cells of its shard before the cell that it gives
    first (:class:`~cytearc.merge.row_plan.RowPlan`), which it gives last.

    Each row count is the number of nonzero values of the source row, taken
    once per source from its count summaries
    (:func:`~cytearc.storage.identity.row_nonzero_counts`). The summaries
    of stores written by earlier CyteArc versions give it only for unsigned
    counts, so the values of other counts are counted in a read within the
    merge's budget. Missing sources contribute zeros, and hold no shard.

    Raises:
        MemoryBudgetError: If the profile and the summaries or the smallest
            read of a source do not fit, before any source is read.
    """
    sources = []
    for index, assay in enumerate(assays):
        if assay is None:
            continue
        counts = as_zarr_array(assay.matrixGroup["counts"], name="counts")
        summarized = row_nonzero_counts(assay.matrixGroup, counts)
        source: zarr.Array | ChunkedArray = (
            summarized
            if summarized is not None
            else assay.rawData._with_resources(resources)
        )
        sources.append((index, assay, source))
    profile_bytes = (row_plan.nCells + 1) * np.dtype(np.int64).itemsize
    held = max(0, int(resident_bytes)) + profile_bytes
    budget = int(resources.memoryBytes)
    needed, item, assay_name = held, "each merged cell's count", None
    for index, assay, source in sources:
        # Placing a source's counts holds them, a flag for each merged cell
        # and the merged rows of the source's cells, in merged and in the
        # source's order (RowPlan.destination_rows).
        place = row_plan.nCells * np.dtype(np.bool_).itemsize + 3 * np.dtype(
            np.int64
        ).itemsize * int(row_plan.sourceCells[index])
        load = max(_row_nnz_load_bytes(assay, source), place)
        if held + load > needed:
            name = row_plan.sourceNames[index]
            needed, assay_name = held + load, assay.name
            item = (
                _read_item(name)
                if isinstance(source, ChunkedArray)
                else f"each merged cell's count and the count summaries of source {name!r}"
            )
    if needed > budget:
        raise _counting_refusal(assay_name, item, needed, budget)
    cumulative = np.zeros(row_plan.nCells + 1, dtype=np.int64)
    row_nnz = cumulative[1:]
    shard_values = [0] * len(assays)
    head_values = [0] * len(assays)
    for source_idx, assay, source in sources:
        if isinstance(source, ChunkedArray):
            row_values = _counted_row_values(
                assay,
                row_plan.sourceNames[source_idx],
                source,
                resources,
                held_bytes=held,
            )
        else:
            row_values = np.asarray(source[:])
        # A merge takes every cell of each source.
        row_nnz[row_plan.destination_rows(source_idx)] = row_values
        if row_values.size:
            shard_rows = max(1, int(assay.rawData.chunksize[0]))
            shard_values[source_idx] = int(
                np.add.reduceat(
                    row_values,
                    np.arange(0, row_values.size, shard_rows),
                    dtype=np.int64,
                ).max()
            )
            first = row_plan.firstCells[source_idx]
            head_values[source_idx] = int(
                row_values[first - first % shard_rows : first].sum(dtype=np.int64)
            )
        del row_values
    np.cumsum(row_nnz, out=row_nnz)
    return cumulative, shard_values, head_values


def _merge_import_requirements(
    assays: list[Any | None],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    destination_dtype: Any,
    *,
    resources: ResourceBudget,
    additionalResidentBytes: int = 0,
) -> _MergeImportRequirements:
    # The destination CountSummary exists for the whole write, so planning and
    # execution both count it through this one function.
    base_resident = (
        row_plan.resident_bytes()
        + alignment.resident_bytes()
        + CountSummary.nbytes_for(row_plan.nCells, alignment.nFeats)
        + max(0, int(additionalResidentBytes))
    )
    cumulative, shard_values, head_values = _cumulative_row_nnz(
        assays,
        row_plan,
        resources,
        resident_bytes=base_resident,
    )
    present = [assay for assay in assays if assay is not None]
    source_dtype = (
        np.result_type(*(assay.rawData.dtype for assay in present))
        if present
        else np.dtype(destination_dtype)
    )
    source_n_feats = max(
        (int(assay.rawData.shape[1]) for assay in present),
        default=0,
    )
    max_decode_bytes = max(
        (
            max(0, int(getter()))
            for assay in present
            if callable(getter := getattr(assay.rawData, "_max_decode_bytes", None))
        ),
        default=0,
    )
    value_bytes = max(
        [
            np.dtype(destination_dtype).itemsize,
            *(np.dtype(assay.rawData.dtype).itemsize for assay in present),
        ]
    )
    resident_bytes = base_resident + int(cumulative.nbytes)

    source_row_bytes = int(source_n_feats * value_bytes)

    def extra_producer_bytes(width: int) -> int:
        rows = max(0, int(width))
        dense_bytes = rows * source_row_bytes
        # Remap/canonicalize staging for source-width COO indices and values.
        remap_bytes = (
            rows
            * max(1, int(source_n_feats))
            * (value_bytes + 2 * np.dtype(np.int32).itemsize)
        )
        # The batch's window of merged rows indexes its pieces.
        return int(
            dense_bytes + remap_bytes + max_decode_bytes + rows * WINDOW_BYTES_PER_ROW
        )

    return _MergeImportRequirements(
        maxWindowNnz=lambda rows: max_window_nnz(cumulative, rows),
        sourceDtype=np.dtype(source_dtype),
        residentBytes=resident_bytes,
        extraProducerBytes=extra_producer_bytes,
        sourceRowBytes=source_row_bytes,
        shardValues=shard_values,
        headValues=head_values,
    )


def fit_assay_counts_layout(
    assays: list[Any | None],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    dtype: str,
    *,
    profile: StorageProfile,
    resources: ResourceBudget,
    writeCountsT: bool,
    additionalResidentBytes: int = 0,
    countsTResidentBytes: int = 0,
    requested: CountMatrixPolicy | None = None,
) -> CountMatrixPolicy:
    """Return the layout of merged counts whose write and ``countsT`` fit.

    ``requested`` is used exactly. Otherwise the default layout is fitted to
    the budget, before the destination arrays are created.
    """
    requirements = _merge_import_requirements(
        assays,
        row_plan,
        alignment,
        dtype,
        resources=resources,
        additionalResidentBytes=additionalResidentBytes,
    )

    return fit_count_layout(
        {"counts": (alignment.nFeats, dtype)},
        nCells=row_plan.nCells,
        profile=profile,
        memoryBytes=resources.memoryBytes,
        transposed=("counts",) if writeCountsT else (),
        admitCounts=sparse_counts_admission(
            nRows=row_plan.nCells,
            maxWindowNnz=requirements.maxWindowNnz,
            sourceDtype=requirements.sourceDtype,
            residentBytes=requirements.residentBytes,
            extraProducerBytes=requirements.extraProducerBytes,
        ),
        requested=requested,
        countsTResidentBytes=countsTResidentBytes,
        subject="The merge",
    )


def _kept_columns(counts: ChunkedArray, columns: np.ndarray) -> ChunkedArray:
    """Return the ``columns`` of ``counts``, which are all of them or a selection."""
    return counts if columns.size == int(counts.shape[1]) else counts[:, columns]


def write_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    assays: list[Any | None],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    *,
    resources: ResourceBudget,
    additionalResidentBytes: int = 0,
    io: StorageIoPolicy | None = None,
) -> int:
    """Stream remapped source rows into the destination counts array.

    Each batch of merged rows takes the next rows of every source, so the
    write holds a window of the next rows of each source, read a chunk
    column at a time, and cuts each batch's rows from them. A window never
    crosses a source's shard, so with room for a shard of every source each
    chunk is decoded once. The write holds a stored shard of each source
    dense, or as CSR when dense shards do not fit or would leave the write
    fewer bands at once (:func:`_held_windows`); the count summaries bound
    the values of a CSR shard. When neither fits, it holds as many dense
    rows as fit, and when no window of a batch's rows fits, each batch reads
    its rows of every source as it comes. When every source column has a
    merged column of its own, as with ``feature_key="ids"``, batches are
    scattered into dense destination rows; otherwise they pass through
    sparse coordinates, which sum the features that share a merged column.
    Every path writes the same values in the same bands.
    """
    destination = load_count_array(root, assay_name, workspace)
    summary = CountSummary(destination)
    requirements = _merge_import_requirements(
        assays,
        row_plan,
        alignment,
        destination.dtype,
        resources=resources,
        additionalResidentBytes=additionalResidentBytes,
    )
    plan = resolve_sparse_import_batch(
        (destination,),
        nRows=row_plan.nCells,
        resources=resources,
        maxWindowNnz=requirements.maxWindowNnz,
        sourceDtype=requirements.sourceDtype,
        residentBytes=requirements.residentBytes,
        extraProducerBytes=requirements.extraProducerBytes,
    )
    batch_rows = max(1, int(plan.batchRows))
    destination_dtype = np.dtype(destination.dtype)
    present = [
        assay for assay in assays if assay is not None and int(assay.feats.N) > 0
    ]
    source_n_feats = max((int(assay.rawData.shape[1]) for assay in present), default=0)
    value_bytes = max(
        [
            destination_dtype.itemsize,
            *(np.dtype(assay.rawData.dtype).itemsize for assay in present),
        ]
    )

    # A source feature that the merged features leave out, as an
    # intersection does, maps to -1, and the read leaves its column out.
    kept_columns = [
        np.flatnonzero(order_map >= 0) for order_map in alignment.featOrderMap
    ]
    kept_maps = [
        order_map[columns]
        for order_map, columns in zip(alignment.featOrderMap, kept_columns, strict=True)
    ]
    # The sources' counts are read within the merge's budget, not each
    # source store's own. A source without the assay, or without features,
    # has no rows to read, and its cells keep zero counts.
    readers = [
        None
        if assay is None or int(assay.feats.N) == 0
        else _kept_columns(assay.rawData._with_resources(resources), columns)
        for assay, columns in zip(assays, kept_columns, strict=True)
    ]
    column_runs = [
        None if reader is None else _chunk_column_runs(reader) for reader in readers
    ]
    for index, reader in enumerate(readers):
        if reader is not None and row_plan.selectedRows[index] is not None:
            raise ValueError("A merge writes every cell of each source")
    shard_rows = max(
        (int(reader.chunksize[0]) for reader in readers if reader is not None),
        default=0,
    )
    # The cells that a batch takes from the largest source read, on average.
    part_rows = -(
        -batch_rows
        * max(
            (
                cells
                for cells, reader in zip(row_plan.sourceCells, readers, strict=True)
                if reader is not None
            ),
            default=0,
        )
        // max(1, row_plan.nCells)
    )
    # The most cells that a batch takes from one source, which bound a piece.
    piece_rows = row_plan.most_source_cells(batch_rows)

    # CSR windows of a stored shard of each source: what they hold, and the
    # most that one refill and one piece taken from a window hold beside them.
    csr_windows = [
        _csr_window_bytes(reader, runs, values, piece_rows)
        for reader, runs, values in zip(
            readers, column_runs, requirements.shardValues, strict=True
        )
        if reader is not None and runs is not None
    ]
    # A source of more than one shard gives the cells before its first cell
    # in their shard last. A window of whole shards keeps these heads apart
    # once it moves on, rather than read their shard again.
    head_rows = [
        0
        if reader is None or int(reader.shape[0]) <= int(reader.chunksize[0])
        else row_plan.firstCells[index] % int(reader.chunksize[0])
        for index, reader in enumerate(readers)
    ]

    def held_bytes(rows: int, sparse: bool, heads: bool = False) -> int:
        """Return the bytes of windows of ``rows`` rows of every source, or of ``sparse`` ones, with their ``heads``."""
        if sparse:
            held = sum(window for window, _, _ in csr_windows)
            kept = sum(
                _csr_rows_bytes(reader, runs, values, head)
                for reader, runs, values, head in zip(
                    readers,
                    column_runs,
                    requirements.headValues,
                    head_rows,
                    strict=True,
                )
                if reader is not None and runs is not None and head
            )
        else:
            held = int(
                rows
                * sum(int(reader.shape[1]) for reader in readers if reader is not None)
                * value_bytes
            )
            kept = sum(
                head * int(reader.shape[1]) * value_bytes
                for reader, head in zip(readers, head_rows, strict=True)
                if reader is not None and rows >= int(reader.chunksize[0])
            )
        return held + (kept if heads else 0)

    def held_read_bytes(rows: int, sparse: bool) -> int:
        """Return the most that filling a window of ``rows`` rows of a source holds beside the windows, or filling a ``sparse`` one and taking a piece from it."""
        if sparse:
            return max((refill for _, refill, _ in csr_windows), default=0) + max(
                (take for _, _, take in csr_windows), default=0
            )
        return max(
            (_held_read_bytes(runs[0], rows) for runs in column_runs if runs),
            default=0,
        )

    def batch_pieces(
        window_rows: int, sparse: bool, heads: bool
    ) -> Iterator[tuple[RowPlanWindow, Iterator[tuple[int, np.ndarray, np.ndarray]]]]:
        """Yield each batch of merged rows with its pieces of source rows.

        A piece is a source, the rows of the batch that it fills, and the
        source rows for them. With ``window_rows``, each source's rows come
        from a window of that many held rows, or with ``sparse`` from a CSR
        window of a stored shard of the source, which keep their ``heads``
        apart; otherwise each batch reads its rows of each source as it
        comes. A piece's rows may be a view of a window that the next piece
        refills, so each piece is used before the next is asked for.
        """
        held = (
            [
                None
                if reader is None or runs is None
                else _HeldSourceRows(
                    reader,
                    runs,
                    int(reader.chunksize[0]) if sparse else window_rows,
                    resources.workers,
                    sparse=sparse,
                    first_row=first,
                    keep_head=heads,
                )
                for reader, runs, first in zip(
                    readers, column_runs, row_plan.firstCells, strict=True
                )
            ]
            if window_rows
            else None
        )

        def pieces(
            window: RowPlanWindow,
        ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
            for part in window.parts:
                reader = readers[part.sourceIdx]
                if reader is None:
                    continue
                first = int(part.localRows[0])
                count = int(part.localRows.size)
                if held is None:
                    values = controlled_compute(
                        reader[first : first + count, :], resources.workers
                    )
                    yield part.sourceIdx, part.positions, values
                    del values
                    continue
                window_source = held[part.sourceIdx]
                assert window_source is not None
                for offset, values in window_source.take(first, count):
                    yield (
                        part.sourceIdx,
                        part.positions[offset : offset + int(values.shape[0])],
                        values,
                    )
                    del values

        for window in iter_row_plan_windows(row_plan, batch_rows):
            yield window, pieces(window)

    destination_shard_rows = array_shard_rows(destination)

    def band_writes(resident: int) -> int:
        """Return the dense bands that the write writes at once beside ``resident`` bytes, or 0 when one does not fit."""
        try:
            return plan_dense_write(
                destination,
                resources,
                row_band_task_count(row_plan.nCells, destination_shard_rows),
                io=io,
                residentBytes=resident,
            ).writeWorkers
        except MemoryBudgetError:
            return 0

    if all(
        np.unique(order_map).size == order_map.size
        for reader, order_map in zip(readers, kept_maps, strict=True)
        if reader is not None
    ):
        mask_bytes = np.dtype(bool).itemsize if destination_dtype.kind in "fc" else 0
        cast_bytes = (
            destination_dtype.itemsize
            if any(
                reader.dtype != destination_dtype
                for reader in readers
                if reader is not None
            )
            else 0
        )
        # A batch holds the merged rows that it assembles, a float
        # destination's zero mask and its window's index arrays, and, when a
        # source's counts take another type, the cast of a piece's source
        # rows, beside the band that the batch is copied into.
        batch_reserve = (
            piece_rows * source_n_feats * cast_bytes
            + batch_rows * alignment.nFeats * (destination_dtype.itemsize + mask_bytes)
            + destination_shard_rows * alignment.nFeats * destination_dtype.itemsize
            + batch_rows * WINDOW_BYTES_PER_ROW
        )

        def dense_write(
            window_rows: int, sparse: bool
        ) -> tuple[int, int, int, bool] | None:
            """Return the bytes that a dense write holding windows of ``window_rows`` rows, or ``sparse`` ones, keeps and reserves for its producer, the bands it writes at once, and whether the windows keep their heads, or None when they do not fit.

            The windows keep their heads unless that leaves the write fewer
            bands at once.
            """
            if not window_rows:
                # A batch read alone also holds a piece's source rows and what
                # its read decodes.
                reserve = (
                    batch_reserve
                    + piece_rows * source_n_feats * value_bytes
                    + requirements.extraProducerBytes(0)
                )
                writes = band_writes(requirements.residentBytes + reserve)
                return (0, reserve, writes, False) if writes else None
            reserve = batch_reserve + held_read_bytes(window_rows, sparse)
            best = None
            for heads in (True, False):
                held = held_bytes(window_rows, sparse, heads)
                writes = band_writes(requirements.residentBytes + held + reserve)
                if writes and (best is None or writes > best[2]):
                    best = (held, reserve, writes, heads)
            return best

        def dense_batches(
            window_rows: int, sparse: bool, heads: bool
        ) -> Iterator[np.ndarray]:
            for window, pieces in batch_pieces(window_rows, sparse, heads):
                dense = np.zeros(
                    (window.nRows, alignment.nFeats), dtype=destination_dtype
                )
                for source_idx, positions, values in pieces:
                    scatter_block_dense(
                        values,
                        kept_maps[source_idx],
                        alignment.nFeats,
                        destination_dtype,
                        out=dense,
                        rows=positions,
                    )
                    # The source rows may be a view of a held window.
                    del values
                if destination_dtype.kind in "fc":
                    # Sparse coordinates hold no negative zero.
                    np.copyto(dense, 0, where=dense == 0)
                yield dense
                # The next batch is made without this one beside it.
                del dense

        window_rows, sparse = _held_windows(
            shard_rows,
            part_rows,
            lambda rows, sparse: (
                0 if (planned := dense_write(rows, sparse)) is None else planned[2]
            ),
        )
        planned = dense_write(window_rows, sparse)
        if planned is not None:
            held, reserve, _, heads = planned
            counter = write_dense_from_row_batches(
                destination,
                dense_batches(window_rows, sparse, heads),
                msg=f"Writing merged assay {assay_name}",
                resources=resources,
                io=io,
                producerReserveBytes=reserve,
                residentBytes=requirements.residentBytes + held,
                countSummary=summary,
            )
            return _finish_assay_counts(
                root, assay_name, workspace, destination, summary, counter
            )

    def sparse_plan(
        window_rows: int, sparse: bool, heads: bool = False
    ) -> SparseImportPlan:
        """Plan the sparse write of batches cut from windows of ``window_rows`` held rows, or from ``sparse`` ones, with their ``heads``."""
        # The windows' reads replace what a batch's read decodes, and a
        # piece takes its source rows as a view of a dense window.
        decode = requirements.extraProducerBytes(0)
        return resolve_sparse_import_batch(
            (destination,),
            nRows=row_plan.nCells,
            resources=resources,
            maxWindowNnz=requirements.maxWindowNnz,
            sourceDtype=requirements.sourceDtype,
            batchRows=batch_rows,
            residentBytes=requirements.residentBytes
            + held_bytes(window_rows, sparse, heads),
            extraProducerBytes=lambda width: (
                requirements.extraProducerBytes(width)
                - decode
                - width * requirements.sourceRowBytes
                + held_read_bytes(window_rows, sparse)
            ),
        )

    def sparse_writes(window_rows: int, sparse: bool) -> tuple[int, bool]:
        """Return the bands that the sparse write writes at once beside the windows, or 0 when they do not fit, and whether the windows keep their heads, unless that leaves fewer bands at once."""
        best = (0, False)
        for heads in (True, False) if window_rows else (False,):
            try:
                plan = sparse_plan(window_rows, sparse, heads)
            except MemoryBudgetError:
                continue
            # Each band the sparse write writes is a dense band, which the
            # plan admitted beside the windows and the producer once.
            writes = max(
                1,
                band_writes(
                    requirements.residentBytes
                    + held_bytes(window_rows, sparse, heads)
                    + plan.producerReserveBytes
                ),
            )
            if writes > best[0]:
                best = (writes, heads)
        return best

    window_rows, sparse = _held_windows(
        shard_rows, part_rows, lambda rows, sparse: sparse_writes(rows, sparse)[0]
    )
    heads = sparse_writes(window_rows, sparse)[1]
    held = 0
    if window_rows:
        plan = sparse_plan(window_rows, sparse, heads)
        held = held_bytes(window_rows, sparse, heads)

    def remapped_batches() -> Iterator[coo_matrix]:
        for window, pieces in batch_pieces(window_rows, sparse, heads):
            # A batch of cells of sources without the assay has no coordinates.
            rows = [np.empty(0, dtype=np.int64)]
            columns = [np.empty(0, dtype=np.int64)]
            data = [np.empty(0, dtype=destination_dtype)]
            for source_idx, positions, values in pieces:
                coordinates = remap_block_to_coo(
                    values,
                    kept_maps[source_idx],
                    alignment.nFeats,
                    resources.workers,
                    destination_dtype,
                )
                # The source rows may be a view of a held window.
                del values
                rows.append(positions[coordinates.row])
                columns.append(np.asarray(coordinates.col))
                data.append(np.asarray(coordinates.data))
                del coordinates
            # Each source's coordinates are canonical and the sources fill
            # different rows; the shard buffer sorts the rows of the batch.
            yield coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                shape=(window.nRows, alignment.nFeats),
            )
            del rows, columns, data

    counter = accumulate_sparse_to_shards(
        destination,
        iter_progress(
            remapped_batches(),
            total=-(-row_plan.nCells // batch_rows),
            desc=f"Writing merged assay {assay_name}",
        ),
        resources=resources,
        residentBytes=requirements.residentBytes + held,
        producerReserveBytes=plan.producerReserveBytes,
        io=io,
        countSummary=summary,
    )
    return _finish_assay_counts(
        root, assay_name, workspace, destination, summary, counter
    )


def _chunk_column_runs(counts: ChunkedArray) -> list[ChunkedArray]:
    """Return the runs of ``counts`` that each hold one stored chunk column.

    A read of some rows of one run decodes only that column's chunks at
    those rows. Counts that are not stored in chunks give all of their
    columns as one run.
    """
    return _column_runs(counts, max(1, counts.chunk_columns()))


def _held_read_bytes(run: ChunkedArray, rows: int) -> int:
    """Return what a read of ``rows`` rows of ``run``, one chunk column, holds beside the rows held.

    A window reads its rows as a slice of ``run``, which keeps their
    positions in an index array. The charge is that of a read of the rows
    by an index array, which counts Zarr's indexers of them and its copy of
    the rows that it takes from each chunk, and so holds the positions too.
    """
    selected = int(rows) * int(run.shape[1]) * run.dtype.itemsize
    return run._read_bytes(selected, rows_indexed=True)


def _csr_rows_bytes(
    reader: ChunkedArray, runs: list[ChunkedArray], values: int, rows: int
) -> int:
    """Return what CSR pieces of ``rows`` rows of a source, one a chunk column of ``runs``, hold with ``values`` values: each value, with a spare one, its column and the row offsets, indexed as SciPy keeps them."""
    index_bytes = (
        4 if max(values, int(reader.shape[1])) <= np.iinfo(np.int32).max else 8
    )
    pieces = len(runs)
    return (values + pieces) * (reader.dtype.itemsize + index_bytes) + pieces * (
        rows + 1
    ) * index_bytes


def _csr_window_bytes(
    reader: ChunkedArray, runs: list[ChunkedArray], values: int, piece_rows: int
) -> tuple[int, int, int]:
    """Return what a CSR window of a stored shard of a source holds, and the most that its refill and a piece taken from it hold beside it.

    ``values`` bounds the nonzero values of a shard, and ``runs`` are the
    source's chunk columns. The window holds the CSR rows of each chunk
    column: the values, each with a spare one, their columns and the row
    offsets, indexed as SciPy keeps them. A refill holds the read of a chunk
    column beside them, and a piece of up to ``piece_rows`` rows its dense
    rows.
    """
    rows, columns = int(reader.chunksize[0]), int(reader.shape[1])
    take = min(rows, int(piece_rows)) * columns * reader.dtype.itemsize
    return (
        _csr_rows_bytes(reader, runs, values, rows),
        _held_read_bytes(runs[0], rows),
        take,
    )


def _held_windows(
    shard_rows: int, part_rows: int, writes: Callable[[int, bool], int]
) -> tuple[int, bool]:
    """Return the rows of each source that a merge holds at once, or 0 to read batch by batch, and whether it holds them as CSR.

    ``writes`` gives the bands that the write writes at once beside windows
    of the rows given of every source, dense or as CSR, or 0 when they do
    not fit. A merge holds dense windows of ``shard_rows``, a stored shard
    of the sources, when they leave the write two bands at once, or as many
    as CSR windows of a stored shard of each source would, and otherwise
    the CSR windows, when they fit. A write of one band at a time runs in
    the producer's thread (:func:`~cytearc.storage.parallel.stream_shards`),
    so reading and writing take turns; from two bands on they overlap, and
    more bands gain less than the CSR windows' conversion costs. When
    neither fits, the merge holds the most dense rows that fit.
    ``part_rows`` are the cells that a batch takes from the largest source;
    dense windows of fewer rows than these, or than the shard when that is
    less, would decode a shard more often than batches that read their own
    rows, so the write then reads each batch's rows as the batch comes.
    """
    least = min(int(part_rows), int(shard_rows))
    if least <= 0:
        return 0, False
    dense = writes(int(shard_rows), False)
    csr = writes(int(shard_rows), True)
    if dense and dense >= min(csr, 2):
        return int(shard_rows), False
    if csr:
        return int(shard_rows), True
    rows = affordable_width(lambda rows: writes(rows, False) > 0, int(shard_rows))
    return (rows if rows >= least else 0), False


class _HeldSourceRows:
    """The rows of one merge source, held a window at a time as batches take them in order.

    A window holds up to ``window_rows`` rows, read a chunk column at a time,
    and never crosses the edge of a stored shard, so a window of a whole
    shard or more decodes each chunk of its shards once. With ``sparse``, a
    window holds each chunk column's rows as CSR, and gives the rows taken
    from it as dense copies. The source gives its rows from ``first_row``
    on, then from its first row up to that one
    (:class:`~cytearc.merge.row_plan.RowPlan`). The rows before
    ``first_row`` in its shard, the head, come last: with ``keep_head``, a
    window of whole shards keeps them apart once it moves on, rather than
    read their shard again.
    """

    def __init__(
        self,
        reader: ChunkedArray,
        runs: list[ChunkedArray],
        window_rows: int,
        workers: int,
        *,
        sparse: bool = False,
        first_row: int = 0,
        keep_head: bool = False,
    ) -> None:
        self._reader = reader
        self._runs = runs
        self._windowRows = max(1, int(window_rows))
        self._shardRows = max(1, int(reader.chunksize[0]))
        self._workers = workers
        self._sparse = sparse
        self._start = self._stop = 0
        self._values: Any = None
        self._headStart = int(first_row) - int(first_row) % self._shardRows
        self._headStop = int(first_row) if keep_head else self._headStart
        self._head: Any = None

    def take(self, first: int, count: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield the source rows ``first`` to ``first + count``, in views of the windows that hold them or dense copies of CSR ones, each with its offset among the rows."""
        offset = 0
        while offset < count:
            row = first + offset
            if self._head is not None and self._headStart <= row < self._headStop:
                width = min(count - offset, self._headStop - row)
                yield offset, self._rows(self._head, row - self._headStart, width)
                offset += width
                continue
            if not self._start <= row < self._stop:
                self._refill(row)
            width = min(count - offset, self._stop - row)
            # No reference to the window outlives the yield, so a refill lets
            # it go before it reads the next.
            yield offset, self._rows(self._values, row - self._start, width)
            offset += width

    def _rows(self, values: Any, start: int, width: int) -> np.ndarray:
        """Return ``width`` rows of held ``values`` from row ``start``: a view of dense rows, or a dense copy of CSR ones."""
        if not self._sparse:
            dense: np.ndarray = values[start : start + width]
            return dense
        rows = np.zeros((width, int(self._reader.shape[1])), self._reader.dtype)
        column = 0
        for piece in values:
            fill_dense_from_csr(piece, start, rows, column)
            column += int(piece.shape[1])
        return rows

    def _refill(self, row: int) -> None:
        """Hold the window that holds source row ``row``.

        A window of part of a shard starts at the row; a window of whole
        shards starts at the shard that holds it, so the window of the first
        row also holds the head. The head is kept apart when that window
        moves on.
        """
        if (
            self._head is None
            and self._values is not None
            and self._start <= self._headStart < self._headStop <= self._stop
            and not self._headStart <= row < self._headStop
        ):
            begin, end = self._headStart - self._start, self._headStop - self._start
            self._head = (
                [piece[begin:end] for piece in self._values]
                if self._sparse
                else self._values[begin:end].copy()
            )
        # The previous window goes before the next is read.
        self._values = None
        n_rows = int(self._reader.shape[0])
        shard = self._shardRows
        if self._windowRows < shard:
            begin = row
            stop = min(n_rows, row + self._windowRows, (row // shard + 1) * shard)
        else:
            begin = row - row % shard
            stop = min(n_rows, begin + self._windowRows // shard * shard)
        if self._sparse:
            values: Any = [
                csr_from_dense(controlled_compute(run[begin:stop, :], self._workers))
                for run in self._runs
            ]
        else:
            values = np.empty(
                (stop - begin, int(self._reader.shape[1])), dtype=self._reader.dtype
            )
            column = 0
            for run in self._runs:
                end = column + int(run.shape[1])
                values[:, column:end] = controlled_compute(
                    run[begin:stop, :], self._workers
                )
                column = end
        self._start, self._stop, self._values = begin, stop, values


def _finish_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    destination: zarr.Array,
    summary: CountSummary,
    counter: int,
) -> int:
    matrix_path = _matrix_group_path(assay_name, workspace)
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    finalize_counts(destination, summary=summary)
    matrix_group.attrs["complete"] = True
    return counter


def write_assay_counts_t(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    profile: StorageProfile,
    resources: ResourceBudget,
    residentBytes: int = 0,
    io: StorageIoPolicy | None = None,
) -> zarr.Array:
    """Write ``countsT``, which replays the layout persisted with the counts."""
    counts = load_count_array(root, assay_name, workspace)
    group_path = _matrix_group_path(assay_name, workspace)
    group = as_zarr_group(root[group_path], name=group_path)
    from ..assay.classification import default_feature_sets

    metadata_path = _assay_metadata_path(assay_name, workspace)
    result = write_counts_t(
        counts,
        group,
        profile=profile,
        resources=resources,
        residentBytes=residentBytes,
        io=io,
        overwrite=True,
        featureSets=default_feature_sets(
            as_zarr_group(root[metadata_path], name=metadata_path)
        ),
    )
    logger.debug(f"Wrote countsT for assay {assay_name}")
    return result


def matrix_group_complete(
    root: zarr.Group, assay_name: str, workspace: str | None
) -> bool:
    path = _matrix_group_path(assay_name, workspace)
    if path not in root:
        return False
    group = as_zarr_group(root[path], name=path)
    return group.attrs.get("complete") is True


def validate_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    n_cells: int,
    alignment: FeatureAlignment,
    dtype: str,
    chunks: tuple[int, int],
) -> str | None:
    """Return why a completed counts component cannot be reused.

    The matrix group is complete, and its assay metadata group exists: merge
    containment refuses a workspace matrix that the workspace does not claim.
    """
    assay_path = _assay_metadata_path(assay_name, workspace)
    matrix_path = _matrix_group_path(assay_name, workspace)
    assay_group = as_zarr_group(root[assay_path], name=assay_path)
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    if "counts" not in matrix_group:
        return f"counts array is missing from {matrix_path!r}"
    counts = as_zarr_array(matrix_group["counts"], name=f"{matrix_path}/counts")
    from ..storage.counts_t_contract import validate_count_matrix

    try:
        validate_count_matrix(matrix_group, require_transpose=False)
    except ValueError as error:
        return str(error)
    expected_shape = (int(n_cells), int(alignment.nFeats))
    if tuple(int(value) for value in counts.shape) != expected_shape:
        return (
            f"counts shape for {assay_name!r} is {tuple(counts.shape)}, "
            f"expected {expected_shape}"
        )
    if np.dtype(counts.dtype) != np.dtype(dtype):
        return (
            f"counts dtype for {assay_name!r} is {np.dtype(counts.dtype)}, "
            f"expected {np.dtype(dtype)}"
        )
    # The validated layout replays the shards from the chunks, shape, and dtype.
    actual_chunks = tuple(int(value) for value in counts.chunks)
    if actual_chunks != tuple(chunks):
        return (
            f"counts chunks for {assay_name!r} are {actual_chunks}, "
            f"expected {tuple(chunks)}"
        )

    if "featureData" not in assay_group:
        return f"featureData is missing from {assay_path!r}"
    feature_group = as_zarr_group(
        assay_group["featureData"],
        name=f"{assay_path}/featureData",
    )
    expected_ids = np.asarray(alignment.mergedFeatsMap["ids"], dtype=str)
    expected_names = np.asarray(alignment.mergedFeatsMap["names"], dtype=str)
    for column, expected in (("ids", expected_ids), ("names", expected_names)):
        if column not in feature_group:
            return f"featureData/{column} is missing for {assay_name!r}"
        feature_array = as_zarr_array(
            feature_group[column],
            name=f"{assay_path}/featureData/{column}",
        )
        actual = np.asarray(feature_array[:], dtype=str)
        if not np.array_equal(actual, expected):
            return f"featureData/{column} does not match for {assay_name!r}"
    if "I" not in feature_group:
        return f"featureData/I is missing for {assay_name!r}"
    included = as_zarr_array(
        feature_group["I"],
        name=f"{assay_path}/featureData/I",
    )
    if tuple(int(value) for value in included.shape) != (alignment.nFeats,):
        return f"featureData/I has the wrong shape for {assay_name!r}"
    if np.dtype(included.dtype) != np.dtype(bool):
        return f"featureData/I has the wrong dtype for {assay_name!r}"
    if not bool(np.asarray(included[:], dtype=bool).all()):
        return f"featureData/I is not fully selected for {assay_name!r}"
    return None


def assess_counts_t_reuse(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    n_cells: int,
    n_features: int,
    dtype: str,
) -> CountsTReuseAssessment:
    """Classify whether an existing ``countsT`` can be reused by merge.

    Outcomes:
    - ``reusable``: complete paired layout matching the planned geometry
    - ``incomplete``: missing or ``complete`` is not True
    - ``invalid``: complete array whose shape, dtype, or paired layout
      disagrees with the merge plan
    """
    from ..storage.counts_t_contract import validate_count_matrix

    matrix_path = _matrix_group_path(assay_name, workspace)
    if matrix_path not in root:
        return CountsTReuseAssessment(
            outcome="incomplete",
            reason=f"matrix group {matrix_path!r} is missing",
        )
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    if "countsT" not in matrix_group:
        return CountsTReuseAssessment(
            outcome="incomplete",
            reason=f"countsT is missing for {assay_name!r}",
        )
    counts_t = as_zarr_array(
        matrix_group["countsT"],
        name=f"{matrix_path}/countsT",
    )
    if counts_t.attrs.get("complete") is not True:
        return CountsTReuseAssessment(
            outcome="incomplete",
            reason=f"countsT is not complete for {assay_name!r}",
        )
    expected_shape = (int(n_features), int(n_cells))
    actual_shape = tuple(int(value) for value in counts_t.shape)
    if actual_shape != expected_shape:
        return CountsTReuseAssessment(
            outcome="invalid",
            reason=(
                f"countsT shape for {assay_name!r} is {actual_shape}, "
                f"expected {expected_shape}"
            ),
        )
    if np.dtype(counts_t.dtype) != np.dtype(dtype):
        return CountsTReuseAssessment(
            outcome="invalid",
            reason=(
                f"countsT dtype for {assay_name!r} is {np.dtype(counts_t.dtype)}, "
                f"expected {np.dtype(dtype)}"
            ),
        )
    try:
        validate_count_matrix(matrix_group, require_transpose=True)
    except ValueError as exc:
        return CountsTReuseAssessment(outcome="invalid", reason=str(exc))
    return CountsTReuseAssessment(outcome="reusable", reason=None)
