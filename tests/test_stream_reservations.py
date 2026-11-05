"""Streams and the operations on them reserve what they hold within the budget."""

import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from cytearc.matrix.chunked import ChunkedArray
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.execution import execution_report_scope
from cytearc.storage.geometry import ArrayGeometry
from tests.thread_helpers import blas_threads, raised_threads


def test_read_bytes_follow_what_a_zarr_read_holds() -> None:
    sharded = ArrayGeometry(
        shape=(100, 40), chunks=(10, 4), shards=(50, 40), itemsize=2
    )
    nominal = 10 * 4 * 2
    # The result and Zarr's shard-level copy of it, the compressed bytes of
    # the chunks touched in a shard, and the chunks decoded at once.
    assert sharded.readBytes(1_000, 5) == 2 * 1_000 + (5 + 5) * nominal
    assert sharded.readBytes(1_000, 5, decodes=2) == 2 * 1_000 + (5 + 2) * nominal
    assert sharded.readBytes(1_000, 5, decodes=9) == sharded.readBytes(1_000, 5)
    # An unsharded read counts its result and the chunks it decodes.
    unsharded = ArrayGeometry(shape=(100, 40), chunks=(10, 4), shards=None, itemsize=2)
    assert unsharded.readBytes(1_000, 5) == 1_000 + 5 * nominal
    assert unsharded.readBytes(1_000, 5, decodes=1) == 1_000 + nominal


def test_array_geometry_rejects_inconsistent_layouts() -> None:
    from types import SimpleNamespace

    from cytearc.storage.geometry import array_geometry

    def layout(
        shape: tuple[int, ...], chunks: tuple[int, ...], shards: Any = None
    ) -> Any:
        metadata = SimpleNamespace(shards=shards)
        return SimpleNamespace(
            shape=shape, chunks=chunks, dtype=np.uint16, metadata=metadata
        )

    with pytest.raises(ValueError, match="chunks .* do not match shape"):
        array_geometry(layout((4, 4), (2,)))
    with pytest.raises(ValueError, match="chunks must be positive"):
        array_geometry(layout((4, 4), (0, 2)))
    with pytest.raises(ValueError, match="cannot be negative"):
        array_geometry(layout((-1, 4), (2, 2)))
    with pytest.raises(ValueError, match="shards .* do not match shape"):
        array_geometry(layout((4, 4), (2, 2), shards=(4,)))
    # The store that computes an array reports what one read holds beside it.
    computed = layout((4, 4), (2, 2))
    computed.path = "computed"
    computed.store = SimpleNamespace(chunk_read_scratch_bytes=lambda _path: -1)
    with pytest.raises(ValueError, match="read scratch cannot be negative"):
        array_geometry(computed)


@pytest.mark.parametrize("source_budget", [None, 288, 320])
def test_document_frequency_fits_blocks_beside_retained_arrays(
    source_budget: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cytearc.assay.normalization import stream_document_frequency

    values = np.arange(48, dtype=np.uint16).reshape(12, 4)
    matrix = ChunkedArray(
        values,
        rows=np.arange(12),
        block_size=12,
        resources=None if source_budget is None else ResourceBudget(source_budget, 2),
    )
    reads = []
    materialize = matrix._materialize_range

    def read(start: int, end: int) -> np.ndarray:
        reads.append(end - start)
        return materialize(start, end)

    monkeypatch.setattr(matrix, "_materialize_range", read)
    with execution_report_scope() as reports:
        frequency, _ = stream_document_frequency(
            matrix, memory_bytes=320, nthreads=2, msg="", operation="Frequency"
        )

    np.testing.assert_array_equal(frequency, np.count_nonzero(values, axis=0))
    assert sum(reads) == len(values)
    assert max(reads) == (2 if source_budget == 288 else 5)
    assert matrix.chunksize == (12, 4)
    (report,) = reports
    assert report.plan.reservedBytes <= min(source_budget or 320, 320)
    assert report.plan.readWorkers == report.actualReadWorkers == 1


@pytest.mark.parametrize("budget", [480, 600])
def test_document_frequency_fits_mask_and_term_frequency_scratch(budget: int) -> None:
    from cytearc.assay.normalization import stream_document_frequency

    values = np.arange(48, dtype=np.uint16).reshape(12, 4)
    selected = np.arange(12) % 2 == 0
    totals = values[selected].sum(axis=1, dtype=np.float64)
    matrix = ChunkedArray(values, rows=np.arange(12), block_size=12)
    with execution_report_scope() as reports:
        frequency, term_sums = stream_document_frequency(
            matrix,
            memory_bytes=budget,
            nthreads=2,
            msg="",
            operation="Frequency",
            row_mask=selected,
            term_totals=totals,
        )

    np.testing.assert_array_equal(frequency, np.count_nonzero(values[selected], axis=0))
    np.testing.assert_allclose(
        term_sums, (values[selected] / totals[:, None]).sum(axis=0)
    )
    (report,) = reports
    assert report.plan.reservedBytes <= budget
    assert report.unitsCompleted == (6 if budget == 480 else 3)


def test_document_frequency_reserves_decode_bytes_when_fitting_blocks() -> None:
    from cytearc.assay.normalization import stream_document_frequency

    values = np.arange(48, dtype=np.uint16).reshape(12, 4)
    stored = zarr.array(values, chunks=(4, 4))

    def frequency(budget: int) -> tuple[np.ndarray, Any]:
        matrix = ChunkedArray(stored, rows=np.arange(12), block_size=12)
        with execution_report_scope() as reports:
            counted, _ = stream_document_frequency(
                matrix, memory_bytes=budget, nthreads=1, msg="", operation="Frequency"
            )
        (report,) = reports
        return counted, report

    # A block of every row reads once; a byte less than it reserves reads the
    # rows in several blocks, each reserved with its decode within the budget.
    _, whole = frequency(1 << 20)
    assert whole.unitsCompleted == 1
    budget = whole.plan.reservedBytes - 1
    counted, report = frequency(budget)
    np.testing.assert_array_equal(counted, np.count_nonzero(values, axis=0))
    assert report.plan.reservedBytes <= budget
    assert report.unitsCompleted > 1


@pytest.mark.parametrize("budget", [255, 267])
def test_document_frequency_rejects_impossible_row_before_read(
    budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.assay.normalization import stream_document_frequency

    matrix = ChunkedArray(
        np.arange(48, dtype=np.uint16).reshape(12, 4), rows=np.arange(12), block_size=12
    )

    def unexpected_read(*_args: Any) -> np.ndarray:
        pytest.fail("An impossible row must fail admission before reading")

    monkeypatch.setattr(matrix, "_materialize_range", unexpected_read)
    with pytest.raises(
        MemoryError, match="Frequency could not allocate the required memory"
    ) as caught:
        stream_document_frequency(
            matrix, memory_bytes=budget, nthreads=1, msg="", operation="Frequency"
        )
    assert "retained arrays and the smallest one-row block" in str(caught.value)
    cause = caught.value.__cause__
    assert isinstance(cause, MemoryError)
    assert "256" in str(cause)
    assert f"operation limit is {budget} bytes" in str(cause)


def test_fixed_row_blocks_with_one_prefetch_leave_the_consumer_its_threads() -> None:
    values = np.arange(12).reshape(6, 2)
    matrix = ChunkedArray(values, block_size=2, resources=ResourceBudget(1 << 20, 2))
    rows = 0
    with raised_threads(blas=2), execution_report_scope() as reports:
        for block in matrix._stream_blocks(
            nthreads=2, msg=None, prefetch=1, row_mask=None
        ):
            # The reads hold no thread limit, so the consumer's own work on
            # each block runs under the count outside the stream, not under
            # the plan's one thread per reader.
            assert blas_threads() == {2}
            np.testing.assert_array_equal(block, values[rows : rows + len(block)])
            rows += len(block)
            del block
    assert rows == len(values)
    (report,) = reports
    assert report.plan.threadsPerComputeWorker == 1
    assert report.plan.unitBytes == values[:2].nbytes


def _sharded_counts(tmp_path: Any, values: np.ndarray) -> zarr.Array:
    """Store counts on disk in row shards of 20,000 cells and 16 chunks."""
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    counts = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(20_000, 4),
        shards=(20_000, values.shape[1]),
        dtype=values.dtype,
        fill_value=0,
    )
    counts[:] = values
    return counts


def test_row_block_streams_of_a_sharded_array_within_a_small_budget_give_its_values(
    tmp_path: Any,
) -> None:
    rng = np.random.default_rng(0)
    values = rng.poisson(0.3, size=(40_000, 64)).astype(np.uint16)
    matrix = ChunkedArray(
        _sharded_counts(tmp_path, values),
        nthreads=2,
        resources=ResourceBudget(24_000_000, 2),
    )
    np.testing.assert_array_equal(matrix.compute(), values)


def test_compute_drops_each_block_before_it_asks_for_the_next() -> None:
    matrix = ChunkedArray.from_numpy(np.arange(12.0).reshape(6, 2), block_size=2)
    given: list[weakref.ref[np.ndarray]] = []

    def stream(**_kwargs: Any) -> Iterator[np.ndarray]:
        for start in range(0, 6, 2):
            # The block the caller was given last must be gone by now.
            assert all(ref() is None for ref in given)
            block = np.arange(12.0).reshape(6, 2)[start : start + 2].copy()
            given.append(weakref.ref(block))
            yield block
            del block

    matrix._stream_blocks = stream  # type: ignore[method-assign]
    np.testing.assert_array_equal(matrix.compute(), np.arange(12.0).reshape(6, 2))
    assert len(given) == 3


def test_row_block_streams_keep_no_block_they_have_yielded() -> None:
    values = np.arange(24.0).reshape(12, 2)
    matrix = ChunkedArray(values, block_size=2, resources=ResourceBudget(1 << 20, 1))
    yielded: list[weakref.ref[np.ndarray]] = []
    materialize = matrix._materialize_range

    def read_once_released(start: int, end: int) -> np.ndarray:
        # One reader runs, and the caller dropped every block it was given, so
        # the stream must not hold one while it reads the next.
        assert all(ref() is None for ref in yielded)
        return materialize(start, end)

    matrix._materialize_range = read_once_released  # type: ignore[method-assign]
    total = 0.0
    for block in matrix.stream_blocks():
        total += float(block.sum())
        yielded.append(weakref.ref(block))
        del block
    assert total == values.sum()
    assert len(yielded) == 6


def test_sharded_feature_blocks_reserve_their_read() -> None:
    from cytearc.storage.feature_stream import plan_feature_stream

    root = zarr.open_group(mode="w")
    array = root.create_array(
        "counts", shape=(20, 40), chunks=(5, 4), shards=(10, 40), dtype=np.uint16
    )
    chunk = 5 * 4 * 2
    cells = np.arange(20)

    def plan(budget: int) -> Any:
        return plan_feature_stream(
            array,
            featureAxis=1,
            cellAxis=0,
            featureIndices=np.arange(8),
            cellIndices=cells,
            resources=ResourceBudget(budget, 1),
            blockBytes=lambda width: width * cells.size * 2,
            requestedBatchSize=8,
        )

    # A block of eight features holds its 320-byte result, Zarr's copy of it,
    # and, per decode, the compressed bytes of the two row chunks of each of
    # its two feature chunks in a shard, beside one decoded chunk.
    needed = 320 + 320 + (2 * 2 + 1) * chunk
    planned = plan(needed)
    assert [block.bins for block in planned.blocks] == [(0, 1)]
    assert (planned.readWorkers, planned.ioConcurrency) == (1, 1)
    with pytest.raises(MemoryError, match="affordable width is 7"):
        plan(needed - 1)
