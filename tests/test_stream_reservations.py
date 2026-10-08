"""Streams and the operations on them reserve what they hold within the budget."""

import tracemalloc
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from cytearc.matrix.chunked import ChunkedArray
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.execution import ExecutionReport, execution_report_scope
from cytearc.storage.geometry import ArrayGeometry


def _traced_peak(
    run: Callable[[], Any],
) -> tuple[int, list[ExecutionReport], Any]:
    """Return the traced peak of ``run`` after a warm-up, its reports and result."""
    run()
    with execution_report_scope() as reports:
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            result = run()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return peak - base, reports, result


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
    matrix = ChunkedArray(stored, rows=np.arange(12), block_size=12)
    with execution_report_scope() as reports:
        frequency, _ = stream_document_frequency(
            matrix, memory_bytes=228, nthreads=1, msg="", operation="Frequency"
        )
    np.testing.assert_array_equal(frequency, np.count_nonzero(values, axis=0))
    (report,) = reports
    assert report.plan.reservedBytes <= 228
    assert report.unitsCompleted == 4


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


def test_fixed_row_blocks_preserve_compute_threads_with_one_prefetch() -> None:
    from threadpoolctl import threadpool_info

    values = np.arange(12).reshape(6, 2)
    matrix = ChunkedArray(values, block_size=2, resources=ResourceBudget(1 << 20, 2))
    rows = 0
    with execution_report_scope() as reports:
        for block in matrix._stream_blocks(
            nthreads=2, msg=None, prefetch=1, row_mask=None
        ):
            pools = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
            assert all(pool["num_threads"] == 1 for pool in pools)
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


@pytest.mark.parametrize("consumer", ["stream_blocks", "compute"])
def test_row_block_streams_reserve_what_a_sharded_read_holds(
    tmp_path: Any, consumer: str
) -> None:
    rng = np.random.default_rng(0)
    values = rng.poisson(0.3, size=(40_000, 64)).astype(np.uint16)
    matrix = ChunkedArray(
        _sharded_counts(tmp_path, values),
        nthreads=2,
        resources=ResourceBudget(24_000_000, 2),
    )

    def run() -> np.ndarray | None:
        if consumer == "compute":
            return matrix.compute()
        for block in matrix.stream_blocks():
            del block
        return None

    traced, reports, computed = _traced_peak(run)

    if consumer == "compute":
        np.testing.assert_array_equal(computed, values)
    (plan,) = [report.plan for report in reports]
    block = 20_000 * 64 * 2
    chunk = 20_000 * 4 * 2
    # A block, Zarr's copy of it, the compressed bytes of the 16 chunks of its
    # shard, and the one chunk decoded at a time.
    assert plan.unitBytes == 2 * block + (16 + 1) * chunk
    assert plan.readWorkers == 2
    assert traced <= plan.reservedBytes <= 24_000_000


def test_row_block_reads_count_the_chunks_of_their_columns() -> None:
    root = zarr.open_group(mode="w")
    counts = root.create_array(
        "counts", shape=(100, 40), chunks=(10, 4), shards=(50, 40), dtype=np.uint16
    )
    chunk = 10 * 4 * 2
    full = ChunkedArray(counts, resources=ResourceBudget(1 << 20, 1))
    # A block of one 50-row shard: five row chunks of ten column chunks each.
    assert full._max_decode_bytes() == (5 * 10 + 1) * chunk
    assert full._block_owned_bytes() == 2 * 50 * 40 * 2
    # Five selected columns fall in two column chunks.
    narrow = full[:, [0, 1, 2, 5, 6]]
    assert narrow._max_decode_bytes() == (5 * 2 + 1) * chunk
    assert narrow._block_task_bytes() == 2 * 50 * 5 * 2 + (5 * 2 + 1) * chunk
    assert ChunkedArray.from_numpy(np.ones((4, 3)))._block_task_bytes() == 4 * 3 * 8


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


def _hvg_store(tmp_path: Any, values: np.ndarray, **options: Any) -> Any:
    """Write uint16 RNA counts in read groups of 32 features and open them."""
    from cytearc import DataStore
    from cytearc.storage.count_matrix import CountMatrixPolicy
    from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
    from cytearc.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    path = str(tmp_path / "hvg.zarr")
    root = zarr.open_group(path, mode="w")
    n_cells, n_features = values.shape
    cell_ids = np.array([f"c{index}" for index in range(n_cells)])
    feature_ids = np.array([f"f{index}" for index in range(n_features)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        n_cells,
        feature_ids,
        feature_ids,
        dtype="uint16",
        policy=CountMatrixPolicy(unitBytes=n_cells * 32 * 2, chunkBytes=32 * 2_500 * 2),
    )
    counts[:] = values
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    return DataStore(
        path,
        default_assay="RNA",
        min_features_per_cell=0,
        mito_pattern="",
        ribo_pattern="",
        **options,
    )


def test_hvg_feature_statistics_reserve_their_totals_and_partials(
    tmp_path: Any,
) -> None:
    rng = np.random.default_rng(0)
    values = rng.poisson(0.15, size=(100_000, 64)).astype(np.uint16)
    assay = _hvg_store(tmp_path, values, nthreads=4, mem_budget="6M").RNA
    cells = np.arange(values.shape[0], dtype=np.int64)
    features = np.arange(values.shape[1], dtype=np.int64)

    traced, reports, stats = _traced_peak(
        lambda: assay._streaming_feature_stats(cells, features)
    )

    (plan,) = [report.plan for report in reports]
    # The inverse totals of 100,000 cells, the partial statistics of the 40
    # bands of 2,500 cells, and the stream's band indices stay for the whole
    # stream, about two fifths of the budget.
    assert plan.residentBytes > 100_000 * 8 + 40 * 64 * 3 * 8 + 100_000 * 16
    assert traced <= plan.reservedBytes <= assay.resources.memoryBytes
    normalized = values / values.sum(axis=1, keepdims=True).clip(1) * assay.sf
    np.testing.assert_allclose(stats["normed_tot"], normalized.sum(axis=0))
    np.testing.assert_allclose(stats["sigmas"], normalized.var(axis=0))
