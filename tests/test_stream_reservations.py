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
    # the chunks touched in a shard, each at the bound of its encoded size,
    # here 1,024 bytes beyond the 80 it decodes to, and the chunks decoded at
    # once.
    encoded = 80 + 1_024
    assert sharded.readBytes(1_000, 5) == 2 * 1_000 + 5 * encoded + 5 * nominal
    assert (
        sharded.readBytes(1_000, 5, decodes=2) == 2 * 1_000 + 5 * encoded + 2 * nominal
    )
    assert sharded.readBytes(1_000, 5, decodes=9) == sharded.readBytes(1_000, 5)
    # An unsharded read holds its result and each chunk it decodes at once as
    # stored and decoded: the decoded chunk beside the bound of its compressed
    # bytes, the decoded size plus 1/128 of it, rounded down, and 1,024 bytes.
    unsharded = ArrayGeometry(shape=(100, 40), chunks=(10, 4), shards=None, itemsize=2)
    stored = nominal + nominal + 1_024
    assert unsharded.readBytes(1_000, 5) == 1_000 + 5 * stored
    assert unsharded.readBytes(1_000, 5, decodes=1) == 1_000 + stored
    assert unsharded.readBytes(1_000, 5, decodes=9) == unsharded.readBytes(1_000, 5)
    # A chunk of 2,048 bytes is bounded at 16 bytes more than itself, and the
    # scratch a store reports is charged once per decode.
    scratched = ArrayGeometry(
        shape=(640, 8), chunks=(64, 8), shards=None, itemsize=4, readScratchBytes=100
    )
    assert scratched.readBytes(500, 3, decodes=2) == 500 + 2 * (
        2_048 + 2_048 + 16 + 1_024 + 100
    )
    assert scratched.readBytes(0, 1) == 2_048 + 2_048 + 16 + 1_024 + 100


@pytest.mark.parametrize(
    ("shape", "chunks"),
    [((100, 40), (10, 4)), ((1_000_000,), (100_000,)), ((5_000, 300), (512, 128))],
)
@pytest.mark.parametrize("concurrency", [1, 2, 64])
def test_unsharded_read_bytes_agree_with_whole_and_band_reads(
    shape: tuple[int, ...], chunks: tuple[int, ...], concurrency: int
) -> None:
    # The three costs price the same Zarr read of an unsharded array, so a
    # plan and the memory ledger reserve the same bytes for it.
    geometry = ArrayGeometry(shape=shape, chunks=chunks, shards=None, itemsize=8)
    n_chunks = -(-shape[0] // chunks[0]) * geometry.bandChunks()
    whole = shape[0] * geometry.rowBytes()
    assert geometry.readBytes(
        whole, n_chunks, decodes=concurrency
    ) == geometry.wholeReadBytes(concurrency)
    rows = min(shape[0], chunks[0])
    assert geometry.readBytes(
        rows * geometry.rowBytes(), geometry.bandChunks(), decodes=concurrency
    ) == geometry.bandReadBytes(rows, concurrency)


@pytest.mark.parametrize("profile", ["fast_local", "cloud"])
def test_a_row_block_read_of_an_unsharded_array_holds_no_more_than_its_plan(
    tmp_path: Any, profile: str
) -> None:
    import tracemalloc

    from cytearc.storage.layout import normed_array_spec

    # Normalized values are stored unsharded. Random bits do not compress, so
    # each stored chunk is as large as it is decoded, the most a read holds.
    spec = normed_array_spec(
        2_048, 2_048, profile=profile, targetChunkBytes=2_048 * 2_048 * 4
    )
    assert spec.shards is None and spec.chunks == (2_048, 2_048)
    bits = np.random.default_rng(0).integers(
        0, 1 << 32, size=spec.shape, dtype=np.uint32
    )
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    stored = root.create_array(
        "normed",
        shape=spec.shape,
        chunks=spec.chunks,
        dtype=spec.dtype,
        compressors=spec.compressors,
        fill_value=spec.fillValue,
    )
    stored[:] = bits.view(np.float32)
    matrix = ChunkedArray(stored)
    np.testing.assert_array_equal(
        matrix._materialize_range(0, 2_048).view(np.uint32), bits
    )
    del bits

    tracemalloc.start()
    try:
        block = matrix._materialize_range(0, 2_048)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    decoded = 2_048 * 2_048 * 4
    # The read held the compressed chunk beside its result and the decoded
    # chunk, more than those two alone, and the plan covers all of it.
    assert block.nbytes == decoded
    assert peak > block.nbytes + decoded
    assert matrix.stream_bytes() >= peak


@pytest.mark.parametrize("profile", ["fast_local", "cloud"])
def test_a_band_read_of_a_sharded_array_holds_no_more_than_its_charge(
    tmp_path: Any, profile: str
) -> None:
    import tracemalloc

    from cytearc.storage.arrays import create_numeric_array
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.layout import row_sharded_array_spec

    # Random bits do not compress, so each inner chunk is stored a few bytes
    # larger than the 4 MiB it decodes to.
    spec = row_sharded_array_spec(
        (8_192, 512),
        np.float32,
        profile=profile,
        band_rows=8_192,
        target_chunk_bytes=4 * 1024**2,
    )
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    stored = create_numeric_array(root, "data", spec)
    assert (stored.shards, stored.chunks) == ((8_192, 512), (2_048, 512))
    bits = np.random.default_rng(0).integers(
        0, 1 << 32, size=spec.shape, dtype=np.uint32
    )
    stored[:] = bits.view(np.float32)
    del bits
    geometry = array_geometry(stored)
    assert geometry is not None

    with zarr.config.set({"async.concurrency": 1}):
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            band = stored[:, :]
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    # The read held its result, Zarr's shard-level copy of it, the shard's
    # four compressed chunks and one decoded chunk, more than those alone,
    # and the charge for one decode at a time covers all of it.
    decoded = 2_048 * 512 * 4
    assert peak - base > 2 * band.nbytes + 5 * decoded
    assert geometry.readBytes(band.nbytes, 4, decodes=1) >= peak - base


def test_a_row_block_stream_of_an_unsharded_array_reserves_each_stored_chunk() -> None:
    values = np.arange(48, dtype=np.uint16).reshape(12, 4)
    stored = zarr.array(values, chunks=(6, 4))
    # A block of one chunk's six rows, 48 bytes, decodes that chunk, which is
    # held as stored and decoded: 48 + 48 + 1,024 bytes.
    needed = 48 + (48 + 48 + 1_024)
    matrix = ChunkedArray(stored, resources=ResourceBudget(needed, 1))
    assert matrix._block_size == 6
    np.testing.assert_array_equal(np.vstack(list(matrix.stream_blocks())), values)

    short = ChunkedArray(stored, resources=ResourceBudget(needed - 1, 1))
    with pytest.raises(MemoryError, match=f"One unit needs about {needed} bytes"):
        next(short.stream_blocks())


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

    def frequency(chunks: tuple[int, int], budget: int) -> Any:
        stored = zarr.array(values, chunks=chunks)
        matrix = ChunkedArray(stored, rows=np.arange(12), block_size=12)
        with execution_report_scope() as reports:
            counted, _ = stream_document_frequency(
                matrix, memory_bytes=budget, nthreads=1, msg="", operation="Frequency"
            )
        np.testing.assert_array_equal(counted, np.count_nonzero(values, axis=0))
        (report,) = reports
        return report

    # Stored in chunks of six rows, a block's read holds a chunk of three
    # times the values it holds in chunks of two rows, so the budget of one
    # block of every row in the smaller chunks reads the larger ones in
    # several blocks, each reserved with its decode within the budget.
    budget = frequency((2, 4), 1 << 20).plan.reservedBytes
    assert frequency((2, 4), budget).unitsCompleted == 1
    larger = frequency((6, 4), budget)
    assert larger.unitsCompleted > 1
    assert larger.plan.reservedBytes <= budget


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
    # its two feature chunks in a shard, each at the bound of its encoded
    # size, 1,024 bytes beyond the 40 it decodes to, beside one decoded chunk.
    needed = 320 + 320 + 2 * 2 * (chunk + 1_024) + chunk
    planned = plan(needed)
    assert [block.bins for block in planned.blocks] == [(0, 1)]
    assert (planned.readWorkers, planned.ioConcurrency) == (1, 1)
    with pytest.raises(MemoryError, match="affordable width is 7"):
        plan(needed - 1)
