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
from ..storage.schema import (
    TF_DENOMINATOR_ATTR,
    _assay_paths,
    create_zarr_count_assay,
    load_count_array,
    merged_normalized_values,
)
from ..storage.sharding import (
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
from ..utils.arrays import canonicalize_sparse, checked_sparse_cast, max_window_nnz
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from ..utils.memory_budget import MemoryBudgetError, memory_refusal
from ..utils.progress import iter_progress
from .features import FeatureAlignment
from .row_plan import RowPlan, iter_row_plan_segments, max_row_plan_block_rows


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


def empty_block_coo(n_rows: int, n_feats: int) -> coo_matrix:
    return coo_matrix((n_rows, n_feats))


# Column maps of fewer runs than this are copied run by run.
_SCATTER_RUNS = 64


def scatter_block_dense(
    block: np.ndarray,
    order_map: np.ndarray,
    n_feats: int,
    destination_dtype: np.dtype[Any],
) -> np.ndarray:
    """Place source columns at their merged positions in dense destination rows.

    ``order_map`` must give every source column its own merged column. The
    rows equal the dense bands that ``remap_block_to_coo`` coordinates fill:
    values are checked and cast in the same way, and as sparse coordinates
    hold no zero, a negative zero is written as zero.
    """
    values = np.asarray(block)
    if np.can_cast(values.dtype, destination_dtype, casting="safe"):
        values = values.astype(destination_dtype, copy=False)
    else:
        values = checked_sparse_cast(values, destination_dtype)
    dense = np.zeros((values.shape[0], n_feats), dtype=destination_dtype)
    # Merged columns mostly follow each other in runs, which copy as slices.
    breaks = np.flatnonzero(np.diff(order_map) != 1) + 1
    if breaks.size < _SCATTER_RUNS:
        for start, stop in zip(
            np.r_[0, breaks].tolist(), np.r_[breaks, order_map.size].tolist()
        ):
            first = int(order_map[start])
            dense[:, first : first + stop - start] = values[:, start:stop]
    else:
        dense[:, order_map] = values
    if destination_dtype.kind in "fc":
        np.copyto(dense, 0, where=dense == 0)
    return dense


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
            return [
                _column_run(counts, start, min(n_columns, start + columns))
                for start in range(0, n_columns, columns)
            ]
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
) -> np.ndarray:
    """Return cumulative nonzero counts over merged rows.

    Each row count is the number of nonzero values of the source row, taken
    once per source from its count summaries
    (:func:`~cytearc.storage.identity.row_nonzero_counts`). The summaries
    of stores written by earlier CyteArc versions give it only for unsigned
    counts, so the values of other counts are counted in a read within the
    merge's budget. Missing sources contribute zeros.

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
        load = _row_nnz_load_bytes(assay, source)
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
        for segment in iter_row_plan_segments(row_plan):
            if segment.sourceIdx != source_idx:
                continue
            stop = segment.destStart + int(segment.localRows.size)
            row_nnz[segment.destStart : stop] = row_values[segment.localRows]
        del row_values
    np.cumsum(row_nnz, out=row_nnz)
    return cumulative


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
    cumulative = _cumulative_row_nnz(
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

    def extra_producer_bytes(width: int) -> int:
        rows = max(0, int(width))
        dense_bytes = rows * source_n_feats * value_bytes
        # Remap/canonicalize staging for source-width COO indices and values.
        remap_bytes = (
            rows
            * max(1, int(source_n_feats))
            * (value_bytes + 2 * np.dtype(np.int32).itemsize)
        )
        return int(dense_bytes + remap_bytes + max_decode_bytes)

    return _MergeImportRequirements(
        maxWindowNnz=lambda rows: max_window_nnz(cumulative, rows),
        sourceDtype=np.dtype(source_dtype),
        residentBytes=resident_bytes,
        extraProducerBytes=extra_producer_bytes,
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
    """Stream remapped source blocks into the destination counts array.

    A row-plan block holds rows of one source chunk in a permuted order, so a
    batch of some of its rows decodes the whole chunk. When a block spans
    several batches and the budget holds it beside the write, each block is
    read once and its batches are cut from memory. When every source column
    has a merged column of its own, as with ``feature_key="ids"``, batches are
    scattered into dense destination rows; otherwise they pass through sparse
    coordinates, which sum the features that share a merged column. Every
    path writes the same values in the same bands.
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
    block_rows = max_row_plan_block_rows(row_plan)
    block_bytes = (
        block_rows * source_n_feats * value_bytes if block_rows > batch_rows else 0
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
    # source store's own.
    readers = [
        None
        if assay is None
        else _kept_columns(assay.rawData._with_resources(resources), columns)
        for assay, columns in zip(assays, kept_columns, strict=True)
    ]

    def source_batches(
        hold_blocks: bool,
    ) -> Iterator[tuple[int, np.ndarray | None, int]]:
        """Yield the source, dense source rows, and row count of each batch.

        Batches follow the destination rows; a source without the assay has
        no rows to read.
        """
        for block in iter_row_plan_segments(row_plan):
            assay = assays[block.sourceIdx]
            reader = readers[block.sourceIdx]
            rows = np.asarray(block.localRows, dtype=np.int64)
            if assay is None or reader is None or int(assay.feats.N) == 0:
                for offset in range(0, rows.size, batch_rows):
                    yield block.sourceIdx, None, min(batch_rows, rows.size - offset)
                continue
            held = (
                controlled_compute(reader[rows, :], resources.workers)
                if hold_blocks
                else None
            )
            for offset in range(0, rows.size, batch_rows):
                batch = rows[offset : offset + batch_rows]
                values = (
                    controlled_compute(reader[batch, :], resources.workers)
                    if held is None
                    else held[offset : offset + batch_rows]
                )
                yield block.sourceIdx, values, batch.size
            del held

    if all(
        np.unique(order_map).size == order_map.size
        for assay, order_map in zip(assays, kept_maps, strict=True)
        if assay is not None and int(assay.feats.N) > 0
    ):
        shard_rows = array_shard_rows(destination)
        mask_bytes = np.dtype(bool).itemsize if destination_dtype.kind in "fc" else 0
        # A batch holds its source rows and their cast, its destination rows,
        # and a float destination's zero mask, beside the band it is copied
        # into and what its read decodes.
        dense_reserve = (
            batch_rows * source_n_feats * (value_bytes + destination_dtype.itemsize)
            + batch_rows * alignment.nFeats * (destination_dtype.itemsize + mask_bytes)
            + shard_rows * alignment.nFeats * destination_dtype.itemsize
            + requirements.extraProducerBytes(0)
        )

        def dense_batches(hold_blocks: bool) -> Iterator[np.ndarray]:
            for source_idx, values, n_rows in source_batches(hold_blocks):
                if values is None:
                    yield np.zeros((n_rows, alignment.nFeats), dtype=destination_dtype)
                else:
                    yield scatter_block_dense(
                        values,
                        kept_maps[source_idx],
                        alignment.nFeats,
                        destination_dtype,
                    )

        for held_bytes in (block_bytes, 0) if block_bytes else (0,):
            try:
                plan_dense_write(
                    destination,
                    resources,
                    row_band_task_count(row_plan.nCells, shard_rows),
                    io=io,
                    residentBytes=requirements.residentBytes
                    + held_bytes
                    + dense_reserve,
                )
            except MemoryBudgetError:
                continue
            counter = write_dense_from_row_batches(
                destination,
                dense_batches(held_bytes > 0),
                msg=f"Writing merged assay {assay_name}",
                resources=resources,
                io=io,
                producerReserveBytes=dense_reserve,
                residentBytes=requirements.residentBytes + held_bytes,
                countSummary=summary,
            )
            return _finish_assay_counts(
                root, assay_name, workspace, destination, summary, counter
            )

    held_bytes = 0
    if block_bytes:
        try:
            plan = resolve_sparse_import_batch(
                (destination,),
                nRows=row_plan.nCells,
                resources=resources,
                maxWindowNnz=requirements.maxWindowNnz,
                sourceDtype=requirements.sourceDtype,
                batchRows=batch_rows,
                residentBytes=requirements.residentBytes + block_bytes,
                extraProducerBytes=requirements.extraProducerBytes,
            )
            held_bytes = block_bytes
        except MemoryBudgetError:
            pass

    def block_stream() -> Iterator[coo_matrix]:
        # Split each row-plan block into planner-admitted batch widths so peak
        # residency matches resolve_sparse_import_batch without restacking.
        total_batches = sum(
            1
            for _ in iter_row_plan_segments(
                row_plan,
                segment_rows=batch_rows,
            )
        )
        batches = (
            empty_block_coo(n_rows, alignment.nFeats)
            if values is None
            else remap_block_to_coo(
                values,
                kept_maps[source_idx],
                alignment.nFeats,
                resources.workers,
                destination_dtype,
            )
            for source_idx, values, n_rows in source_batches(held_bytes > 0)
        )
        yield from iter_progress(
            batches,
            total=total_batches,
            desc=f"Writing merged assay {assay_name}",
        )

    counter = accumulate_sparse_to_shards(
        destination,
        block_stream(),
        resources=resources,
        residentBytes=requirements.residentBytes + held_bytes,
        producerReserveBytes=plan.producerReserveBytes,
        io=io,
        countSummary=summary,
    )
    return _finish_assay_counts(
        root, assay_name, workspace, destination, summary, counter
    )


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
