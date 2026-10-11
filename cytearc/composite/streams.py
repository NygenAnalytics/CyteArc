"""Feature streams over virtual composite counts.

They keep the contract of ``map_feature_read_groups`` and
``map_feature_cell_bands`` in ``storage.feature_stream``: the same groups,
bands, order, and owned buffers. They read differently because a composite
read fans out to every source: adjacent feature groups are batched into one
read when the budget allows, so each source chunk is decoded once per batch
rather than once per group. A band reads only its selected cells, so its
``selectedLocal`` counts them in order, and a consumer that indexes a band by
``selectedLocal`` reads the values that the core stream gives.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from threading import Event
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from ..storage.async_execution import AsyncStorageRunner
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.execution import (
    ExecutionReport,
    auto_read_width,
    record_execution_report,
)
from ..storage.feature_stream import (
    FeatureCellBand,
    FeatureReadGroup,
    cell_band_selection,
    feature_group_ranges,
    iter_bounded_handoff,
    persisted_read_group,
    plan_feature_consume,
    read_group_layout,
    sparse_group_rows,
    stream_units,
)
from ..storage.geometry import array_geometry
from ..storage.io_policy import DEFAULT_STORAGE_IO_POLICY, StorageIoPolicy
from ..utils.progress import tqdmbar

if TYPE_CHECKING:
    from .reader import CompositeArrayReader

# A batched group: its stream index, its feature range, and its selected rows
# when the stream reads only some rows of the group.
type _Group = tuple[int, int, int, np.ndarray | None]


def _group_batches(
    groups: list[tuple[int, int]],
    rows: dict[tuple[int, int], np.ndarray],
    fits: Callable[[int], bool],
) -> list[list[_Group]]:
    """Batch consecutive groups while the features of one read fit."""
    batches: list[list[_Group]] = []
    width = 0
    for index, (start, end) in enumerate(groups):
        selected = rows.get((start, end))
        size = end - start if selected is None else len(selected)
        if not fits(size):
            raise MemoryError("One composite feature group exceeds the memory budget")
        if not batches or not fits(width + size):
            batches.append([])
            width = 0
        batches[-1].append((index, start, end, selected))
        width += size
    return batches


def _batch_features(batch: list[_Group]) -> np.ndarray:
    """Return the features that one batched read gathers, in group order."""
    return np.concatenate(
        [
            np.arange(start, end, dtype=np.int64) if rows is None else start + rows
            for _index, start, end, rows in batch
        ]
    )


class _BandBatch:
    """One batched read of a cell band and the groups it holds."""

    def __init__(self, band: FeatureCellBand, groups: list[_Group]) -> None:
        self.band: FeatureCellBand | None = band
        self.groups = groups
        # ``stream_units`` reads the timing of each loaded item.
        self.readSec = band.readSec


def map_cell_bands[T](
    reader: "CompositeArrayReader",
    array: zarr.Array,
    process: Callable[[FeatureCellBand], T],
    *,
    cell_idx: np.ndarray | None,
    feat_idx: Sequence[int] | np.ndarray | None,
    resources: ResourceBudget | None,
    progress: str | None,
    io: StorageIoPolicy | None,
    metrics: dict[str, Any] | None,
    scratchBytes: int,
    orderedCompute: bool,
    cellMajorOrder: bool,
) -> Iterator[T]:
    """Stream feature groups by cell band, like ``map_feature_cell_bands``.

    A band's ``values`` hold only its selected cells, so ``selectedLocal``
    counts them in order. Groups are batched when the stream is cell-major or
    unordered, where a batch does not change the order of the results.
    """
    geometry = array_geometry(array)
    assert geometry is not None
    feature_width, read_group_bytes = persisted_read_group(array)
    groups = feature_group_ranges(array, feat_idx=feat_idx, featureWidth=feature_width)
    selection = cell_band_selection(geometry, cell_idx)
    if not groups or not selection.bands:
        return iter(())
    sparse = sparse_group_rows(groups, feat_idx, n_feats=geometry.shape[0])
    budget = resources or resolve_budget()
    policy = io or DEFAULT_STORAGE_IO_POLICY
    read_workers = min(2, budget.workers, policy.readWorkers or 2)
    max_cells = max(len(local) for _start, _end, local, _dest in selection.bands)
    resident = (
        scratchBytes
        + selection.indexBytes
        + 64 * geometry.shape[0]
        + sum(rows.nbytes for rows in sparse.values())
    )

    def held(width: int) -> int:
        # The batched read and the owned copy of each group's values.
        return (
            2 * width * max_cells * reader.dtype.itemsize
            + reader.read_scratch_bytes(width, max_cells, read_workers=read_workers)
        )

    minimum_width = max(
        end - start if (start, end) not in sparse else len(sparse[(start, end)])
        for start, end in groups
    )
    while read_workers > 1 and held(minimum_width) + resident > budget.memoryBytes:
        read_workers -= 1
    coalesce = cellMajorOrder or not orderedCompute
    batches = _group_batches(
        groups,
        sparse,
        lambda width: held(width) + resident <= budget.memoryBytes,
    )
    if not coalesce:
        batches = [[group] for batch in batches for group in batch]
    feature_batches = [_batch_features(batch) for batch in batches]
    work = [
        (band_index, band, batch, features)
        for band_index, band in enumerate(selection.bands)
        for batch, features in zip(batches, feature_batches, strict=True)
    ]
    if not coalesce:
        work.sort(key=lambda unit: (unit[2][0][0], unit[0]))
    plan = plan_feature_consume(
        budget,
        io=replace(
            policy,
            readWorkers=max(
                1,
                (policy.readWorkers or auto_read_width(budget.workers)) // read_workers,
            ),
        ),
        nUnits=len(work),
        unitBytes=max(held(len(features)) for features in feature_batches),
        scratchBytes=resident,
        ordered=orderedCompute,
    )
    plan = replace(plan, innerReads=read_workers, ioConcurrency=read_workers)

    @asynccontextmanager
    async def load(
        runner: AsyncStorageRunner, _index: int, unit: Any
    ) -> AsyncIterator[_BandBatch]:
        band_index, (start, end, local, destinations), batch, features = unit
        async with runner.reserve_bytes(held(len(features))):
            async with runner.read_lane():
                started = time.perf_counter()
                raw = await runner.io(
                    reader.read(features, start + local, read_workers=read_workers)
                )
                read_seconds = time.perf_counter() - started
            # The read holds only the selected cells, counted from zero.
            selected_local = np.arange(len(local), dtype=np.int64)
            item = _BandBatch(
                FeatureCellBand(
                    featStart=0,
                    featEnd=len(features),
                    cellStart=start,
                    cellEnd=end,
                    values=raw,
                    selectedLocal=selected_local,
                    selectedDestinations=destinations,
                    readSec=read_seconds,
                    blockBytes=raw.nbytes,
                    unitIndex=band_index,
                ),
                batch,
            )
            del raw
            try:
                yield item
            finally:
                item.band = None
                del item

    def process_batch(item: _BandBatch) -> list[T]:
        band = item.band
        assert band is not None
        offset = 0
        results = []
        try:
            for group_index, start, end, rows in item.groups:
                width = end - start if rows is None else len(rows)
                values = band.values[offset : offset + width].copy(order="C")
                logical_index = (
                    band.unitIndex * len(groups) + group_index
                    if cellMajorOrder
                    else group_index * len(selection.bands) + band.unitIndex
                )
                results.append(
                    process(
                        FeatureCellBand(
                            featStart=start,
                            featEnd=end,
                            cellStart=band.cellStart,
                            cellEnd=band.cellEnd,
                            values=values,
                            selectedLocal=band.selectedLocal,
                            selectedDestinations=band.selectedDestinations,
                            readSec=band.readSec if offset == 0 else 0.0,
                            blockBytes=values.nbytes,
                            unitIndex=logical_index,
                            rows=rows,
                        )
                    )
                )
                offset += width
            return results
        finally:
            del band

    stream = stream_units(
        work,
        load,
        process_batch,
        plan=plan,
        unitKind="compositeCellBand",
        orderedCompute=orderedCompute,
        progress=progress,
        metrics=metrics,
        details={
            "readGroupBytes": read_group_bytes,
            "featureWidth": feature_width,
            "featureGroupCount": len(groups),
            "cellBandCount": len(selection.bands),
            "cellMajorOrder": cellMajorOrder,
            "groupsPerRead": max(len(batch) for batch in batches),
        },
    )
    return (result for batch_results in stream for result in batch_results)


def map_read_groups[T](
    reader: "CompositeArrayReader",
    array: zarr.Array,
    process: Callable[[FeatureReadGroup], T],
    *,
    cell_idx: np.ndarray | None,
    feat_idx: Sequence[int] | np.ndarray | None,
    resources: ResourceBudget | None,
    progress: str | None,
    io: StorageIoPolicy | None,
    metrics: dict[str, Any] | None,
    scratchBytes: int,
    orderedCompute: bool,
) -> Iterator[T]:
    """Stream the persisted read groups, like ``map_feature_read_groups``.

    Batches of groups are read one at a time over every selected cell, and
    the groups of a batch are processed by up to the compute width at once,
    each from its own copy of its values.
    """
    layout = read_group_layout(array, cell_idx=cell_idx, feat_idx=feat_idx)
    if layout is None:
        return iter(())
    budget = resources or resolve_budget()
    policy = io or DEFAULT_STORAGE_IO_POLICY
    read_workers = min(2, budget.workers, policy.readWorkers or 2)
    cells = layout.selection.cells
    resident = scratchBytes + layout.selection.indexBytes + 64 * array.shape[0]
    widest = max(end - start for start, end in layout.groups)
    group_bytes = widest * len(cells) * reader.dtype.itemsize
    compute_width = (
        1
        if orderedCompute
        else min(budget.workers, policy.computeWorkers or budget.workers)
    )
    # Preserve the caller's compute width when the budget permits it.
    while compute_width > 1 and (
        resident
        + (compute_width + 1) * group_bytes
        + reader.read_scratch_bytes(widest, len(cells), read_workers=read_workers)
        > budget.memoryBytes
    ):
        compute_width -= 1

    def fits(width: int) -> bool:
        return (
            resident
            + compute_width * group_bytes
            + width * len(cells) * reader.dtype.itemsize
            + reader.read_scratch_bytes(width, len(cells), read_workers=read_workers)
            <= budget.memoryBytes
        )

    while read_workers > 1 and not fits(widest):
        read_workers -= 1
    batches = _group_batches(layout.groups, {}, fits)
    usable_compute_width = max(len(batch) for batch in batches)
    if compute_width > usable_compute_width:
        compute_width = usable_compute_width
        batches = _group_batches(layout.groups, {}, fits)
    feature_batches = [_batch_features(batch) for batch in batches]
    max_width = max(len(features) for features in feature_batches)
    fixed = (
        resident
        + max_width * len(cells) * reader.dtype.itemsize
        + reader.read_scratch_bytes(max_width, len(cells), read_workers=read_workers)
    )
    plan = plan_feature_consume(
        budget,
        io=policy,
        nUnits=len(layout.groups),
        unitBytes=group_bytes,
        scratchBytes=fixed,
        maxUnitsInFlight=compute_width,
        ordered=orderedCompute,
    )
    # One batch is read at a time, with ``read_workers`` source reads inside it.
    plan = replace(
        plan,
        readWorkers=1,
        prefetch=1,
        innerReads=read_workers,
        ioConcurrency=read_workers,
    )
    details = {
        "readGroupBytes": layout.readGroupBytes,
        "featureWidth": layout.featureWidth,
        "featureGroupCount": len(layout.groups),
        "cellBandCount": len(layout.selection.bands),
        "groupsPerRead": max(len(batch) for batch in batches),
    }
    if metrics is not None:
        metrics.update(plan.as_metrics())
        metrics.update(details)

    def run(deliver: Callable[[T], Awaitable[None]], stop: Event) -> None:
        fetch_seconds = 0.0
        compute_seconds = 0.0
        completed = 0
        bar = tqdmbar(desc=progress, total=len(layout.groups)) if progress else None
        runner = AsyncStorageRunner(operation=plan)

        async def operation(active: AsyncStorageRunner) -> None:
            nonlocal fetch_seconds, compute_seconds, completed
            slots = asyncio.Semaphore(plan.computeWorkers)
            for batch, features in zip(batches, feature_batches, strict=True):
                if stop.is_set():
                    break
                async with active.read_lane():
                    started = time.perf_counter()
                    raw = await active.io(
                        reader.read(features, cells, read_workers=read_workers)
                    )
                    fetch_seconds += time.perf_counter() - started

                async def one(
                    group: _Group, offset: int, source_values: np.ndarray
                ) -> None:
                    nonlocal compute_seconds, completed
                    index, start, end, _rows = group
                    try:
                        async with slots:
                            if stop.is_set():
                                return
                            async with active.reserve_bytes(
                                (end - start) * len(cells) * reader.dtype.itemsize
                            ):
                                values = source_values[
                                    offset : offset + end - start
                                ].copy(order="C")
                                item = FeatureReadGroup(
                                    featStart=start,
                                    featEnd=end,
                                    values=values,
                                    readSec=0.0,
                                    blockBytes=values.nbytes,
                                    unitIndex=index,
                                )
                                started = time.perf_counter()
                                result = await active.compute(partial(process, item))
                                compute_seconds += time.perf_counter() - started
                                await deliver(result)
                                del result, item, values
                                completed += 1
                                if bar is not None:
                                    bar.update(1)
                    finally:
                        del source_values

                indexed_groups = []
                offset = 0
                for group in batch:
                    indexed_groups.append((group, offset))
                    offset += group[2] - group[1]
                pending = iter(indexed_groups)

                async def consume(source_values: np.ndarray) -> None:
                    try:
                        for group, offset in pending:
                            await one(group, offset, source_values)
                    finally:
                        del source_values

                try:
                    if orderedCompute:
                        await consume(raw)
                    else:
                        async with asyncio.TaskGroup() as tasks:
                            for _ in range(min(plan.computeWorkers, len(batch))):
                                tasks.create_task(consume(raw))
                finally:
                    del raw

        try:
            runner.run(operation)
        finally:
            if bar is not None:
                bar.close()
            report = record_execution_report(
                ExecutionReport(
                    plan=plan,
                    unitKind="compositeReadGroup",
                    actualReadWorkers=plan.readWorkers,
                    actualComputeWorkers=plan.computeWorkers,
                    actualWriteWorkers=1,
                    fetchSeconds=fetch_seconds,
                    computeSeconds=compute_seconds,
                    readerWaitSeconds=runner.readerWaitSeconds,
                    unitsCompleted=completed,
                    peakHeldBytes=runner.ledger.peak_bytes(),
                    extra=details,
                )
            )
            if metrics is not None:
                metrics.update(report.as_metrics())

    return iter_bounded_handoff(in_flight=plan.computeWorkers, run=run)
