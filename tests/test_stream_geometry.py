import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.assay.persistence import _read_block
from cytearc.matrix.chunked import ChunkedArray
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.async_execution import zarr_io_concurrency
from cytearc.storage.execution import admit_stream
from cytearc.storage.feature_stream import plan_feature_stream
from cytearc.storage.geometry import ArrayGeometry, array_geometry
from cytearc.storage.parallel import stream_shards
from cytearc.storage.partition import (
    affordable_width,
    checked_indices,
    coalesced_gap_bytes,
    contiguous_ranges,
    index_band_bytes,
    indexed_read_parts,
    is_contiguous,
    partition_indices,
    row_band,
    row_read_parts,
    scan_band,
    widest_chunk_span,
)
from cytearc.storage.virtual_arrays import read_selection

from .store_probes import RecordingStore


def _array(
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None = None,
    dtype: type[np.generic] = np.uint32,
    compressors: None | str = "auto",
) -> zarr.Array:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    return root.create_array(
        "counts",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        fill_value=0,
        compressors=compressors,
    )


def _geometry(
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None = None,
    itemsize: int = 4,
) -> ArrayGeometry:
    return ArrayGeometry(
        shape=shape,
        chunks=chunks,
        shards=shards,
        itemsize=itemsize,
    )


def test_nominal_chunk_bytes_counts_the_whole_chunk_at_an_array_edge() -> None:
    array = _array(shape=(12, 7), chunks=(5, 3))
    geometry = array_geometry(array)

    assert geometry is not None
    # The trailing chunk covers two rows and one column of the array, but Zarr
    # stores and decodes the full 5 x 3 chunk.
    assert geometry.nominalChunkBytes() == 5 * 3 * 4


def test_an_edge_chunk_decodes_to_the_nominal_size() -> None:
    array = _array(shape=(12, 7), chunks=(5, 3), compressors=None)
    array[10:12, 6:7] = 1
    geometry = array_geometry(array)

    uncompressed = [
        len(array.store._store_dict[key].to_bytes())
        for key in array.store._store_dict
        if "/c/" in key
    ]

    assert geometry is not None
    assert uncompressed == [geometry.nominalChunkBytes()]


def test_array_geometry_is_absent_for_in_memory_arrays() -> None:
    assert array_geometry(np.zeros((4, 4), dtype=np.uint32)) is None


_FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"


@pytest.mark.parametrize(
    ("pipeline", "max_workers", "pool_size", "workers"),
    [
        ("zarr.core.codec_pipeline.BatchedCodecPipeline", None, 0, 0),
        ("zarr.core.codec_pipeline.BatchedCodecPipeline", 4, 0, 0),
        (_FUSED, 3, 0, 3),
        (_FUSED, 3, 6, 6),
        (_FUSED, 1, 6, 1),
        (_FUSED, None, 0, "cpus"),
        (_FUSED, "many", 0, "cpus"),
    ],
    ids=[
        "Zarr's default pipeline",
        "the default pipeline with workers set",
        "the opt-in pipeline of three workers",
        "the opt-in pipeline on a pool that an earlier read grew",
        "the opt-in pipeline of one worker, which reads inline",
        "the opt-in pipeline of a worker per CPU",
        "the opt-in pipeline of an unreadable count",
    ],
)
def test_array_geometry_records_whether_zarr_decodes_on_a_pool_of_its_own(
    monkeypatch, pipeline: str, max_workers: object, pool_size: int, workers: object
) -> None:
    import os

    import zarr.core.codec_pipeline as codec_pipeline
    from zarr.errors import ZarrUserWarning

    # The pipeline is the one the array was opened with; its worker count is
    # read when the plan is made. Zarr's fused pipeline reads on one pool of
    # the process, which grows to the most workers any read asked for and
    # never shrinks.
    monkeypatch.setattr(codec_pipeline, "_pool_size", pool_size)
    with zarr.config.set({"codec_pipeline.path": pipeline}):
        array = _array(shape=(20, 20), chunks=(5, 5), shards=(10, 10))
    with zarr.config.set({"codec_pipeline.max_workers": max_workers}):
        if pipeline == _FUSED and max_workers == "many":
            # Zarr warns of a count it cannot read and takes one for each CPU.
            with pytest.warns(ZarrUserWarning, match="Ignoring invalid"):
                geometry = array_geometry(array)
        else:
            geometry = array_geometry(array)
    assert geometry is not None
    expected = max(1, os.cpu_count() or 1) if workers == "cpus" else workers
    # A memory store serves Zarr's synchronous I/O.
    assert (geometry.fusedPoolWorkers, geometry.fusedReadsAsync) == (expected, False)
    assert geometry.poolKeepsDecodedChunk is (expected > 1)
    # An array opened after the setting is gone keeps Zarr's default pipeline.
    reopened = zarr.open_array(array.store, path=array.path, mode="r")
    assert array_geometry(reopened).fusedPoolWorkers == 0


def test_a_fused_pipeline_that_a_user_configures_reads_on_its_own_pool_whatever_the_stores_threads(
    tmp_path, monkeypatch
) -> None:
    import zarr.core.codec_pipeline as codec_pipeline
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from fsspec.implementations.local import LocalFileSystem
    from zarr.storage import FsspecStore, LocalStore

    from cytearc.storage.async_execution import (
        ensure_zarr_host_ceiling,
        register_store_threads,
    )
    from tests.memory_helpers import decodes_on_threads_held_together

    # A pool of the test's own, which Zarr makes afresh and the test then
    # gives back, so that no other test reads on a pool sized here.
    monkeypatch.setattr(codec_pipeline, "_pool", None)
    monkeypatch.setattr(codec_pipeline, "_pool_size", 0)
    values = np.arange(30 * 10, dtype=np.uint16).reshape(30, 10)
    with zarr.config.set(
        {"codec_pipeline.path": _FUSED, "codec_pipeline.max_workers": 3}
    ):
        # CyteArc's codec pipeline takes the place of Zarr's default only, so
        # the configured fused pipeline stays.
        ensure_zarr_host_ceiling()
        assert zarr.config.get("codec_pipeline.path") == _FUSED
        store = LocalStore(str(tmp_path / "local.zarr"))
        register_store_threads(store, 1)
        local = zarr.open_group(store=store, mode="w").create_array(
            "counts", data=values, chunks=(10, 10)
        )
        # A DataStore's registration of one thread for the store does not
        # bound the pool: each of the read's three decodes holds its chunk
        # until the other two hold theirs, which only three threads at once
        # can reach.
        (*_, compressor) = local.metadata.codecs
        with decodes_on_threads_held_together(type(compressor), 3, timeout=5.0):
            np.testing.assert_array_equal(local[:], values)
        files = AsyncFileSystemWrapper(
            LocalFileSystem(auto_mkdir=True), asynchronous=True
        )
        remote = zarr.open_group(
            store=FsspecStore(files, path=str(tmp_path / "async.zarr")), mode="w"
        ).create_array("counts", shape=(20, 20), chunks=(5, 5), dtype=np.uint16)
        assert (
            type(local.async_array.codec_pipeline) is codec_pipeline.FusedCodecPipeline
        )
        local_geometry = array_geometry(local)
        remote_geometry = array_geometry(remote)
    assert local_geometry is not None and remote_geometry is not None
    assert (local_geometry.fusedPoolWorkers, local_geometry.fusedReadsAsync) == (
        3,
        False,
    )
    # A store without Zarr's synchronous I/O is read asynchronously.
    assert (remote_geometry.fusedPoolWorkers, remote_geometry.fusedReadsAsync) == (
        3,
        True,
    )


def test_array_geometry_records_shards_and_falls_back_to_chunks() -> None:
    sharded = array_geometry(_array(shape=(20, 20), chunks=(5, 5), shards=(10, 10)))
    plain = array_geometry(_array(shape=(20, 20), chunks=(5, 5)))

    assert sharded is not None and plain is not None
    assert sharded.shards == (10, 10)
    assert sharded.axisShard(0) == 10
    assert sharded.axisChunk(0) == 5
    assert plain.shards is None
    assert plain.axisShard(0) == plain.axisChunk(0) == 5


def test_bin_of_maps_indices_to_their_chunk() -> None:
    geometry = _geometry(shape=(12, 20), chunks=(5, 5))

    np.testing.assert_array_equal(
        geometry.binOf(1, np.array([0, 4, 5, 17])),
        np.array([0, 0, 1, 3]),
    )


def test_row_band_follows_the_requested_stored_unit() -> None:
    geometry = _geometry(shape=(20, 20), chunks=(5, 5), shards=(10, 10))

    assert row_band(geometry, fallback=999) == 10
    assert row_band(geometry, unit="chunk", fallback=999) == 5


def test_row_band_uses_the_fallback_without_geometry() -> None:
    assert row_band(None, fallback=10_000) == 10_000
    assert row_band(None, fallback=0) == 1


def test_scan_band_reads_whole_chunks_up_to_64_mib_or_the_fallback() -> None:
    # Chunks of 100 rows of 1,000 four-byte values hold 400,000 bytes, and
    # 167 of them fit in 64 MiB.
    narrow = _geometry(shape=(100_000, 1_000), chunks=(100, 1_000))
    assert scan_band(narrow, fallback=1) == 167 * 100
    # A chunk larger than 64 MiB is still read whole.
    wide = _geometry(shape=(100_000, 1_000), chunks=(20_000, 1_000))
    assert scan_band(wide, fallback=1) == 20_000
    # An array without stored chunks, such as one in memory, reads the
    # fallback's rows, at least one.
    assert scan_band(None, fallback=100_000) == 100_000
    assert scan_band(None, fallback=0) == 1
    assert scan_band(None, fallback=-5) == 1


def test_a_block_read_of_sharded_rows_and_columns_in_reverse_gives_them() -> None:
    values = np.random.default_rng(73).integers(
        0, 30, size=(1024, 512), dtype=np.uint16
    )
    array = _array(
        shape=values.shape,
        chunks=(128, 64),
        shards=(256, 256),
        dtype=np.uint16,
    )
    array[:] = values
    rows = np.arange(1023, -1, -2)
    columns = np.arange(511, -1, -3)
    with zarr_io_concurrency(1):
        actual = _read_block(array, rows, columns)
    np.testing.assert_array_equal(actual, values[np.ix_(rows, columns)])


def _sharded_counts() -> tuple[np.ndarray, zarr.Array]:
    values = np.random.default_rng(11).integers(
        0, 30, size=(1024, 2048), dtype=np.uint16
    )
    array = _array(
        shape=values.shape, chunks=(256, 256), shards=(1024, 2048), dtype=np.uint16
    )
    array[:] = values
    return values, array


def test_sharded_reads_that_index_both_axes_give_their_selection() -> None:
    values, array = _sharded_counts()
    rows = np.flatnonzero(np.arange(1024) % 7 != 0)
    columns = np.flatnonzero(np.arange(2048) % 4 != 0)
    actual = read_selection(array, rows, columns)
    np.testing.assert_array_equal(actual, values[np.ix_(rows, columns)])


@pytest.mark.parametrize(
    ("rows", "columns"),
    (
        (np.arange(1024)[np.arange(1024) % 7 != 0], np.arange(2048)[::3]),
        (np.arange(1024)[np.arange(1024) % 7 != 0], np.arange(0, 2048, 17)),
        (np.arange(5, 1024, 11), np.arange(2048)[np.arange(2048) % 4 != 0]),
    ),
    ids=("dense", "few columns", "few rows"),
)
def test_row_blocks_that_index_both_axes_give_their_selection(
    rows: np.ndarray, columns: np.ndarray
) -> None:
    values, array = _sharded_counts()
    matrix = ChunkedArray(array, rows=rows, cols=columns, block_size=rows.size)
    block = matrix._materialize_range(0, rows.size)
    np.testing.assert_array_equal(block, values[np.ix_(rows, columns)])


@pytest.mark.parametrize(
    ("rows", "columns", "axis"),
    (
        # 500 of the 1,024 rows in a shuffled order and three columns, one in
        # each of three column chunks. Bands of columns would index the 500
        # rows, 64 bytes each, beside 500 values of each band twice and its
        # copy of them, 35,000 bytes and 96 to order the three columns;
        # bands of rows index the three columns and order the 500 rows, 32
        # bytes each, beside up to 256 rows of three values twice and the
        # copy of one, under 20,000 bytes. The narrower bands by values alone
        # are those of columns, 500 values against 3 x 255.
        (np.random.default_rng(3).permutation(1024)[:500], np.array([0, 300, 600]), 0),
        # Two rows 255 apart and every other one of the first 128 columns, in
        # one chunk: bands of columns hold the two rows' span of 127 values
        # twice and copied, index the two rows and order the 64 columns,
        # under 4,000 bytes, where a band of rows would hold 256 rows of the
        # 64 columns twice.
        (np.array([0, 255]), np.arange(0, 128, 2), 1),
    ),
    ids=("index three columns", "index two rows"),
)
def test_sharded_reads_that_index_both_axes_slice_the_axis_whose_bands_hold_less(
    monkeypatch: pytest.MonkeyPatch, rows: np.ndarray, columns: np.ndarray, axis: int
) -> None:
    from cytearc.storage import virtual_arrays

    values, array = _sharded_counts()
    sliced: list[int] = []
    read_bands = virtual_arrays._read_bands

    def recording(*args, axis, **kwargs):
        sliced.append(axis)
        return read_bands(*args, axis=axis, **kwargs)

    monkeypatch.setattr(virtual_arrays, "_read_bands", recording)
    with zarr_io_concurrency(1):
        actual = read_selection(array, rows, columns)
    np.testing.assert_array_equal(actual, values[np.ix_(rows, columns)])
    assert sliced == [axis]


def test_indexed_read_parts_charge_the_narrower_band() -> None:
    geometry = ArrayGeometry((4096, 4096), (1024, 1024), (4096, 4096), 2)
    touched = 4  # four row chunks per shard, or four column chunks of the columns
    decode = geometry.readBytes(0, touched, decodes=1)
    # Two columns in each column chunk span all 1,024 positions, so a block
    # of 1,024 rows is cheaper to read in bands of rows. A band takes its
    # eight columns by an index array, so it also holds Zarr's indexers of
    # them, 64 bytes a column in a sharded read, and Zarr's copy of the two
    # columns of 1,024 rows that it takes from the chunk it decodes; the
    # reader orders each of the block's rows by row chunk and keeps where
    # each band starts and stops, 48 bytes a row. The rows may lie in all
    # four row chunks, a band each, and the reads of the three earlier bands
    # leave the objects of a read, 32 KiB, and of the four column chunks
    # that each took, 4 KiB a chunk.
    spread = np.array([0, 1023, 1024, 2047, 2048, 3071, 3072, 4095])
    assert widest_chunk_span(spread, 1024) == 1024
    assert indexed_read_parts(geometry, spread, block_rows=1024) == (
        geometry.readBytes(8 * 1024 * 2, touched, decodes=1)
        + 8 * 64
        + 1024 * 2 * 2
        + 32 * 1024
        + 3 * 4 * 4 * 1024,
        48,
    )
    # A dense run in each column chunk spans 16 positions: bands of columns,
    # which take each row by an index array, 64 bytes, and copy its 16 values
    # of the chunk, 32 bytes, beside the reader's ordering of the 32 columns,
    # 48 bytes each, and what the read of the first of the two bands leaves
    # of its four row chunks' objects. Bands of rows would hold 2 * 65,536
    # bytes of the block, the indexers of 32 columns, the copy of 16 columns
    # of 1,024 rows from a chunk, what the three earlier bands leave of the
    # two column chunks each took, and the ordering of the 1,024 rows,
    # 272,384 bytes, more than the 214,528 of column bands.
    dense = np.concatenate([np.arange(start, start + 16) for start in (0, 1024)])
    assert widest_chunk_span(dense, 1024) == 16
    assert indexed_read_parts(geometry, dense, block_rows=1024) == (
        decode + 32 * 48 + 32 * 1024 + 4 * 4 * 1024,
        geometry.readBytes(16 * 2, touched, decodes=1) - decode + 64 + 32,
    )
    assert 32 * 48 + 32 * 1024 + 4 * 4 * 1024 + 1024 * (2 * 16 * 2 + 64 + 32) == (
        214_528
    )
    assert (
        2 * 65_536 + 32 * 64 + 1024 * 16 * 2 + 32 * 1024 + 3 * 2 * 4 * 1024 + 1024 * 48
        == 272_384
    )


def test_reads_in_bands_are_charged_the_shards_each_band_touches_and_what_the_bands_before_it_leave() -> (
    None
):
    from cytearc.storage.partition import run_shards

    # 30,000 rows by 20 columns in shards of 100 rows of two 10-column
    # chunks, 4 bytes a value: 100 rows, one from each of 100 shards, by
    # every other column, five in each chunk.
    geometry = ArrayGeometry((30_000, 20), (100, 10), (100, 20), 4)
    columns = np.arange(0, 20, 2)
    decoded = 100 * 10 * 4
    stored = decoded + decoded // 128 + 1_024
    shard = 2 * stored + decoded + 32_768 + 2 * 4_096 + 16_384
    column_bands, row_bands = index_band_bytes(
        geometry, columns, rows=100, row_span=1, shards=100
    )
    # A band of rows lies in one shard, and holds its 10 values twice, the
    # shard's read, the indexers of the 10 columns, 64 bytes each, and Zarr's
    # copy of a row's five columns of a chunk. The reads of the 99 bands
    # before it took two chunks each and leave up to 192 KiB of their
    # objects, and the reader orders the 100 rows, 48 bytes each.
    assert row_bands == 2 * 40 + shard + 10 * 64 + 5 * 4 + 192 * 1024 + 100 * 48
    # A band of columns takes the 100 rows of a chunk's nine columns in one
    # read of the 100 shards, one at a time while the task of each of the
    # other 99 holds the objects of a chunk, 4 KiB. It indexes the 100 rows
    # and copies their nine values of the chunk, and the read of the band
    # before it took a chunk in each of 100 row chunks, which leaves up to
    # 192 KiB; the reader orders the 10 columns.
    assert column_bands == (
        2 * 3_600 + shard + 99 * 4_096 + 100 * 64 + 3_600 + 192 * 1024 + 10 * 48
    )
    # Ten rows in ten shards: the nine earlier bands of rows leave the objects
    # of a read and of their 18 chunks, and the band of columns before the
    # last those of its ten chunks.
    column_ten, row_ten = index_band_bytes(
        geometry, columns, rows=10, row_span=1, shards=10
    )
    assert row_ten == 2 * 40 + shard + 10 * 64 + 5 * 4 + 32_768 + 18 * 4_096 + 480
    assert column_ten == (
        2 * 360 + shard + 9 * 4_096 + 10 * 64 + 360 + 32_768 + 10 * 4_096 + 480
    )
    # One row is one band, which leaves no band before it.
    assert index_band_bytes(geometry, columns, rows=1, row_span=1, shards=1)[1] == (
        2 * 40 + shard + 10 * 64 + 5 * 4 + 48
    )
    # In shards of one 10-column chunk, a band of columns takes one column
    # shard of each of the 100 rows of shards, and a band of rows the two
    # column shards of its row, the second waiting while the first is read.
    halves = ArrayGeometry((30_000, 20), (100, 10), (100, 10), 4)
    half = stored + decoded + 32_768 + 4_096 + 16_384
    column_halves, row_halves = index_band_bytes(
        halves, columns, rows=100, row_span=1, shards=100
    )
    assert row_halves == 2 * 40 + half + 4_096 + 10 * 64 + 5 * 4 + 192 * 1024 + 4_800
    assert column_halves == (
        2 * 3_600 + half + 99 * 4_096 + 100 * 64 + 3_600 + 192 * 1024 + 10 * 48
    )
    # A run of rows touches a row of shards for every 100 rows, and one more
    # where it does not start at a shard's first row.
    assert [run_shards(geometry, rows) for rows in (0, 1, 2, 100, 101, 102)] == [
        1,
        1,
        2,
        2,
        2,
        3,
    ]
    assert run_shards(None, 1_000) == 1


def test_a_read_of_part_of_a_shard_is_charged_the_chunks_between_its_columns() -> None:
    from dataclasses import replace

    # Shards of one row of ten 40-gene chunks store them in column order.
    plain = ArrayGeometry((2_000, 400), (1_000, 40), (1_000, 400), 4)
    geometry = replace(plain, coalesceGapBytes=1 << 20)
    # Each chunk decodes to 160,000 bytes and is stored in at most 1/128 of
    # that and 1,024 bytes more.
    stored = 160_000 + 160_000 // 128 + 1_024
    # Genes 5, 0, 41, 42, 120, 200, 300 and 397 fall in chunks 0, 1, 3, 5, 7
    # and 9: a merged fetch also takes chunks 2, 4, 6 and 8 between them.
    genes = np.array([5, 397, 41, 42, 200, 120, 0, 300])
    assert coalesced_gap_bytes(geometry, genes, chunks=6) == 4 * stored
    # Zarr merges no gap wider than its setting, so a gap counts at most that.
    assert (
        coalesced_gap_bytes(
            replace(geometry, coalesceGapBytes=100_000), genes, chunks=6
        )
        == 4 * 100_000
    )
    # Two gaps of two chunks each, between chunks 0, 3 and 6.
    assert coalesced_gap_bytes(geometry, np.array([0, 120, 240]), chunks=3) == 2 * min(
        1 << 20, 2 * stored
    )
    # Adjacent chunks, one chunk, every chunk of the shard, and no merging.
    assert coalesced_gap_bytes(geometry, np.arange(0, 120), chunks=3) == 0
    assert coalesced_gap_bytes(geometry, np.array([7]), chunks=1) == 0
    assert coalesced_gap_bytes(geometry, np.arange(400), chunks=10) == 0
    assert coalesced_gap_bytes(plain, genes, chunks=6) == 0
    # The charge of a run of rows over those genes holds the chunks between,
    # and so does a banded read of scattered rows over them.
    with_gaps, per_row = row_read_parts(geometry, genes, rows_contiguous=True)
    without_gaps, same_per_row = row_read_parts(plain, genes, rows_contiguous=True)
    assert with_gaps - without_gaps == 4 * stored
    assert per_row == same_per_row
    banded = indexed_read_parts(geometry, genes, block_rows=1_000)
    banded_apart = indexed_read_parts(plain, genes, block_rows=1_000)
    assert banded[0] - banded_apart[0] == 4 * stored
    # Of shards of five chunks, the one whose chunks lie furthest apart counts:
    # chunks 0, 2 and 4 of the first leave two gaps, 5 and 6 of the second none.
    halves = replace(geometry, shards=(1_000, 200))
    apart_first = np.array([0, 80, 160, 200, 240])
    assert coalesced_gap_bytes(halves, apart_first, chunks=3) == 2 * stored
    assert coalesced_gap_bytes(halves, apart_first[::-1], chunks=3) == 2 * stored


def test_a_read_of_part_of_a_shard_of_several_rows_of_chunks_is_bounded_in_any_order() -> (
    None
):
    # Shards of two rows of four chunks, which Zarr writes in Morton order.
    geometry = ArrayGeometry((64, 32), (8, 8), (16, 32), 4, coalesceGapBytes=10_000)
    stored = 256 + 256 // 128 + 1_024
    # A read of one column chunk in both rows of chunks takes two of the eight
    # chunks; up to the six others may lie between them, but there is one gap.
    assert coalesced_gap_bytes(geometry, np.arange(8), chunks=2) == min(
        1 * 10_000, 6 * stored
    )
    # Three column chunks in both rows: six chunks taken, five gaps at most,
    # each no wider than the setting, and no more than the two chunks left.
    assert coalesced_gap_bytes(geometry, np.arange(24), chunks=6) == min(
        5 * 10_000, 2 * stored
    )
    # Every chunk of the shard is fetched whole.
    assert coalesced_gap_bytes(geometry, np.arange(32), chunks=8) == 0


def test_band_shards_count_the_rows_of_shards_that_a_band_touches() -> None:
    from cytearc.storage.partition import band_shards

    geometry = ArrayGeometry((8_000, 400), (1_000, 40), (1_000, 400), 4)
    # Bands of a run of rows: aligned with the shards, inside them, across
    # one boundary at most, and from a run that starts inside a shard.
    assert band_shards(geometry, range(8_000), 1_000) == 1
    assert band_shards(geometry, range(8_000), 250) == 1
    assert band_shards(geometry, range(8_000), 1_500) == 2
    assert band_shards(geometry, range(8_000), 2_000) == 2
    assert band_shards(geometry, range(500, 7_500), 1_000) == 2
    assert band_shards(geometry, range(8_000), 10_000) == 8
    # Every fourth row: a band of 1,000 rows spans 4,000 rows, four shards.
    assert band_shards(geometry, np.arange(0, 8_000, 4), 1_000) == 4
    # A band of selected rows that starts inside the shard where the band
    # before it ends counts that shard too: rows 600 to 1,199 touch two.
    assert band_shards(geometry, np.arange(3_000), 600) == 2
    assert band_shards(geometry, range(0, 8_000, 4), 1_000) == 4
    # Rows that come back to a shard count it once, as Zarr reads it once.
    assert band_shards(geometry, np.array([0, 5_000, 1, 5_001]), 4) == 2
    assert band_shards(geometry, np.array([0, 1, 5_000, 5_001]), 4) == 2
    assert band_shards(geometry, np.array([0, 5_000, 1, 5_001]), 2) == 2
    # No rows, or no geometry, touch one row of shards at most.
    assert band_shards(geometry, np.array([], dtype=np.int64), 4) == 1
    assert band_shards(None, range(10), 4) == 1


def test_band_shards_count_the_distinct_shards_of_a_selection_a_piece_at_a_time() -> (
    None
):
    import tracemalloc

    from cytearc.storage.partition import band_shards

    # 1,000 shards of 100 rows, and bands of a selection longer than the
    # 65,536 indices that each piece takes, so bands lie in several pieces.
    geometry = ArrayGeometry((100_000, 4), (100, 4), (100, 4), 2)
    rng = np.random.default_rng(6)
    spread = np.sort(rng.choice(100_000, size=60_000, replace=False))
    shuffled = rng.permutation(100_000)

    def touched(rows: np.ndarray, band: int) -> int:
        """Count the distinct shards of each band, band by band."""
        return max(
            len(set((rows[start : start + band] // 100).tolist()))
            for start in range(0, rows.size, band)
        )

    for band in (7, 1_000, 40_000, 70_000):
        assert band_shards(geometry, spread, band) == touched(spread, band)
        assert band_shards(geometry, shuffled, band) == touched(shuffled, band)
    # Shuffled rows of a few shards count those shards, however often they
    # come back to them.
    few = rng.permutation(300)
    assert band_shards(geometry, few, 300) == 3
    assert band_shards(geometry, few, 150) == 3
    # The count of a million rows holds a piece of them, where all of them at
    # once would hold 9 bytes a row. Each band of 1,000 takes a row of ten
    # shards 100 times each, in a shuffled order.
    million = ArrayGeometry((1_000_000, 4), (100, 4), (100, 4), 2)
    rows = rng.permuted(
        np.repeat(np.arange(10_000) * 100, 100).reshape(1_000, 1_000), axis=1
    )
    rows = rows.reshape(-1)
    tracemalloc.start()
    try:
        assert band_shards(million, rows, 1_000) == 10
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
def test_band_shards_count_rows_of_a_narrow_integer_type(dtype: type) -> None:
    from cytearc.storage.partition import band_shards

    # Shards of 200,000 rows, more than uint16 holds: every eighth of the
    # first 60,000 rows lies in the first shard, and of four rows around the
    # boundary two shards each, whatever integer type holds them.
    geometry = ArrayGeometry((400_000, 20), (200_000, 5), (200_000, 20), 2)
    rows = np.arange(0, 60_000, 8)
    assert band_shards(geometry, rows.astype(dtype), 1_000) == 1
    if np.iinfo(dtype).max >= 400_000:
        around = np.array([7, 250_000, 199_999, 200_000, 9, 399_999])
        assert band_shards(geometry, around.astype(dtype), 3) == 2
        assert band_shards(geometry, around.astype(dtype), 6) == 2
    # A stream of those rows of such an array charges its blocks as a stream
    # of their int64 values does, and keeps their 2 or 4 bytes a row, not 8.
    array = zarr.create_array(
        store=MemoryStore(),
        shape=(200_000, 20),
        chunks=(200_000, 5),
        shards=(200_000, 20),
        dtype=np.uint16,
        fill_value=0,
    )
    budget = ResourceBudget(1 << 30, 1)
    narrow = ChunkedArray(array, rows=rows.astype(dtype), resources=budget)
    wide = ChunkedArray(array, rows=rows, resources=budget)
    assert narrow._read_bytes(12_345) == wide._read_bytes(12_345)
    kept = rows.size * (8 - np.dtype(dtype).itemsize)
    assert narrow.stream_bytes() == wide.stream_bytes() - kept


def test_is_contiguous_checks_a_run_a_piece_at_a_time() -> None:
    import tracemalloc

    # A run of a million rows, whose steps are checked 65,536 at a time.
    run = np.arange(5, 1_000_005)
    tracemalloc.start()
    try:
        assert is_contiguous(run)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The steps of the whole run would hold 9 bytes a row.
    assert peak < 1 << 20
    # Two rows swapped where a piece ends, or inside a piece, leave the ends
    # of a run in place but break it.
    for index in (65_535, 65_536, 131_071, 100_000):
        swapped = np.arange(200_000)
        swapped[[index, index + 1]] = swapped[[index + 1, index]]
        assert not is_contiguous(swapped)
    # So do a step of two and a step of none exactly between pieces, which
    # only a check of the step from each piece to the next sees.
    shifted = np.arange(200_000)
    shifted[65_536:131_072] += 1
    assert not is_contiguous(shifted)
    assert is_contiguous(np.arange(65_537).astype(np.uint32))


def test_row_reads_at_a_concurrency_hold_the_shards_and_decodes_they_run_at_once() -> (
    None
):
    from dataclasses import replace

    geometry = ArrayGeometry((8_000, 400), (1_000, 40), (1_000, 400), 4)
    every_gene = np.arange(400)
    decoded = 160_000
    encoded = decoded + decoded // 128 + 1_024
    # One read at a time: the ten compressed chunks of a shard, one decoded
    # chunk, and the objects of the read: 32 KiB, 4 KiB for each chunk it
    # touches and 16 KiB for each chunk it decodes at once.
    one, per_row = row_read_parts(geometry, every_gene, rows_contiguous=True)
    assert one == 10 * encoded + decoded + (32_768 + 10 * 4_096 + 16_384)
    assert per_row == 400 * 4
    # Ten at once over a band of one shard decode its ten chunks at once,
    # beside the objects of ten decodes.
    ten, ten_per_row = row_read_parts(
        geometry, every_gene, rows_contiguous=True, concurrency=10
    )
    shard = 10 * encoded + 10 * decoded + 32_768 + 10 * 4_096 + 10 * 16_384
    assert (ten, ten_per_row) == (shard, per_row)
    # Over a band of four shards, ten at once read all four together; two at
    # once read two shards, each decoding two chunks at once, while the task
    # that Zarr makes for each of the other two holds the objects of a chunk,
    # 4 KiB.
    assert row_read_parts(
        geometry, every_gene, rows_contiguous=True, concurrency=10, shards=4
    ) == (4 * shard, per_row)
    two = 10 * encoded + 2 * decoded + 32_768 + 10 * 4_096 + 2 * 16_384
    assert row_read_parts(
        geometry, every_gene, rows_contiguous=True, concurrency=2, shards=4
    ) == (2 * two + 2 * 4_096, per_row)
    # At one read at a time the shards that a band touches are read in turn.
    assert row_read_parts(geometry, every_gene, rows_contiguous=True, shards=4) == (
        one + 3 * 4_096,
        per_row,
    )
    # Shards of half the genes are read side by side, ten at once.
    halves = replace(geometry, shards=(1_000, 200))
    half = 5 * encoded + 5 * decoded + 32_768 + 5 * 4_096 + 5 * 16_384
    assert row_read_parts(halves, every_gene, rows_contiguous=True, concurrency=10) == (
        2 * half,
        per_row,
    )
    # Zarr's fused pipeline, whose pool threads keep the chunk they decoded,
    # decodes a shard's chunks one at a time whatever the concurrency, beside
    # the chunk decoded before.
    own_pool = replace(geometry, fusedPoolWorkers=2)
    assert row_read_parts(
        own_pool, every_gene, rows_contiguous=True, concurrency=2
    ) == (
        10 * encoded + decoded + decoded + 32_768 + 10 * 4_096 + 16_384,
        per_row,
    )
    # Scattered rows over genes of wide spans in two chunks are read a row
    # chunk at a time, a band of their 4 x 1,000 values twice over beside the
    # one shard that the band's row chunk lies in, decoding its two chunks at
    # once. The band takes its four genes by an index array, so it also holds
    # Zarr's indexers of them, 64 bytes a gene, and Zarr's copy of the two
    # genes of 1,000 rows that it takes from a chunk, which the read's event
    # loop makes a chunk at a time, beside what the read of the band before
    # it leaves of the two chunks it took, and the reader orders each row by
    # row chunk, 48 bytes. A block whose rows form one run is read in one
    # request over the two rows of shards that the rows touch, both at once,
    # which holds more beside the rows, and for each row its shard-level
    # copy, 16 bytes, and the copy of its two genes of a chunk, 8 bytes, less
    # than that ordering.
    wide = np.array([0, 39, 40, 79])
    shard_read = 2 * encoded + 2 * decoded + 32_768 + 2 * 4_096 + 2 * 16_384
    band = 2 * 4 * 1_000 * 4 + shard_read + 4 * 64 + 1_000 * 2 * 4
    left = 32_768 + 2 * 4_096
    run = 2 * shard_read + 4 * 64
    assert run > band + left
    assert row_read_parts(
        geometry, wide, rows_contiguous=False, concurrency=4, shards=2
    ) == (run, 48)
    assert 4 * 4 + 2 * 4 < 48
    assert indexed_read_parts(
        geometry, wide, block_rows=1_000, concurrency=4, shards=2
    ) == (band + left, 48)
    # A band of one row is a run of one row of shards and one band, which
    # leaves no band before it.
    assert row_read_parts(
        geometry, wide, rows_contiguous=False, concurrency=4, shards=2, band_rows=1
    ) == (band, 48)
    assert band > shard_read + 4 * 64
    # An unsharded array's chunks are each read on their own, so a band over
    # three of them holds up to the concurrency of them at once, beside the
    # objects of a read of the three.
    unsharded = ArrayGeometry((8_000, 400), (1_000, 400), None, 4)
    stored = 2 * 1_600_000 + 1_600_000 // 128 + 1_024
    assert row_read_parts(unsharded, every_gene, rows_contiguous=True, shards=3) == (
        stored + 32_768 + 3 * 4_096 + 16_384,
        0,
    )
    assert row_read_parts(
        unsharded, every_gene, rows_contiguous=True, concurrency=8, shards=3
    ) == (3 * stored + 32_768 + 3 * 4_096 + 3 * 16_384, 0)


def test_row_reads_by_index_arrays_charge_their_indexers_and_copies() -> None:
    unsharded = ArrayGeometry((8_000, 400), (1_000, 400), None, 4)
    stored = 2 * 1_600_000 + 1_600_000 // 128 + 1_024
    objects = 32_768 + 4_096 + 16_384
    three = np.array([0, 5, 9])
    # Rows that form one run, over a run of genes, are read as slices, whose
    # chunks Zarr places as views.
    assert row_read_parts(unsharded, np.arange(3), rows_contiguous=True) == (
        stored + objects,
        0,
    )
    # Rows by an index array hold Zarr's indexers of each row, 40 bytes in an
    # unsharded read, and its copy of the row's three genes, 12 bytes.
    assert row_read_parts(unsharded, np.arange(3), rows_contiguous=False) == (
        stored + objects,
        40 + 12,
    )
    # Gathered genes hold the indexers of the three genes, and the copy of
    # each row's three genes.
    assert row_read_parts(unsharded, three, rows_contiguous=True) == (
        stored + objects + 3 * 40,
        12,
    )
    # Both also hold NumPy's buffers of 8,192 elements of the two index
    # arrays, 8 bytes each.
    assert row_read_parts(unsharded, three, rows_contiguous=False) == (
        stored + objects + 3 * 40 + 8_192 * 16,
        40 + 12,
    )


def test_band_costs_are_zero_for_empty_selections() -> None:
    geometry = ArrayGeometry((4096, 4096), (1024, 1024), (4096, 4096), 2)
    empty = np.array([], dtype=np.int64)
    # Callers skip empty selections, but the exported helpers answer them.
    assert widest_chunk_span(empty, 1024) == 0
    assert indexed_read_parts(geometry, empty, block_rows=1024) == (0, 0)
    assert index_band_bytes(geometry, empty, rows=1024, row_span=1024) == (0, 0)
    assert row_read_parts(geometry, empty, rows_contiguous=False) == (0, 0)
    assert row_read_parts(None, np.array([0, 2]), rows_contiguous=False) == (0, 0)


def test_selected_row_cost_admits_only_touched_chunks_within_each_shard() -> None:
    geometry = ArrayGeometry((32, 24), (4, 2), (8, 8), 2, 10_000)
    narrow, _ = row_read_parts(geometry, np.array([0, 8, 16]), rows_contiguous=True)
    wide, _ = row_read_parts(geometry, np.array([0, 2, 4]), rows_contiguous=True)
    # Both selections touch three feature chunks, but only the second puts
    # them in one shard alongside its two row chunks: four more compressed
    # chunks, each at the bound of its encoded size, 1,024 bytes beyond the
    # 16 it decodes to, and the 4 KiB of objects that Zarr's read makes for
    # each. The first spreads them over three shards, read one at a time, and
    # the task of each of the two waiting holds the objects of a chunk.
    assert wide - narrow == (
        4 * (geometry.nominalChunkBytes() + 1_024 + 4_096) - 2 * 4_096
    )
    minimum = narrow + 1
    admitted = admit_stream(
        ResourceBudget(minimum, 1), nBlocks=1, blockBytes=1, decodeBytes=narrow
    )
    assert admitted.reservedBytes == minimum
    with pytest.raises(
        MemoryError,
        match=f"needs about {minimum} bytes, but only {minimum - 1} bytes are available",
    ):
        admit_stream(
            ResourceBudget(minimum - 1, 1),
            nBlocks=1,
            blockBytes=1,
            decodeBytes=narrow,
        )


def test_contiguous_ranges_cover_the_axis_without_overlap() -> None:
    assert contiguous_ranges(12, 5) == [(0, 5), (5, 10), (10, 12)]
    assert contiguous_ranges(0, 5) == []


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([], True),
        ([7], True),
        ([2, 3, 4], True),
        ([2, 4], False),
        ([3, 2], False),
        ([2, 2], False),
        # The ends of a run, but not in order.
        ([0, 2, 1, 3], False),
        ([0, 1, 1, 3], False),
        # A negative start would wrap around as a slice, so it is no run.
        ([-1, 0, 1], False),
        ([-1], False),
    ],
)
def test_is_contiguous_accepts_only_increasing_consecutive_runs(
    indices, expected
) -> None:
    assert is_contiguous(np.asarray(indices, dtype=np.int64)) is expected


@pytest.mark.parametrize(
    ("indices", "dtype", "expected"),
    [
        # A descending step would wrap around to one in the narrow type.
        ([255, 0], np.uint8, False),
        ([127, -128], np.int8, False),
        ([65535, 0], np.uint16, False),
        ([3, 4, 5], np.uint8, True),
        ([125, 126, 127], np.int8, True),
    ],
)
def test_is_contiguous_does_not_wrap_narrow_integer_steps(
    indices, dtype, expected
) -> None:
    assert is_contiguous(np.asarray(indices, dtype=dtype)) is expected


def test_is_contiguous_tells_a_selection_that_spans_more_than_a_run_apart_without_a_pass() -> (
    None
):
    import tracemalloc

    # A shuffled million rows, whose ends lie further apart than a run's.
    rows = np.random.default_rng(0).permutation(2_000_000)[:1_000_000]
    tracemalloc.start()
    try:
        assert not is_contiguous(rows)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # A pass would hold the steps and their comparison, 9 bytes a row.
    assert peak < 64 * 1024
    assert is_contiguous(np.arange(5, 1_000_005))


def test_affordable_width_finds_the_largest_accepted_width() -> None:
    assert affordable_width(lambda width: width <= 3, 10) == 3
    assert affordable_width(lambda width: width <= 20, 10) == 10
    assert affordable_width(lambda _width: False, 10) == 0


@pytest.mark.parametrize("limit", [0, 1, 2, 10, 1000])
def test_affordable_width_preserves_budget_boundaries(limit):
    for budget in range(limit + 1):
        assert affordable_width(lambda rows: rows <= budget, limit) == budget


def test_partition_indices_emits_one_block_per_chunk_by_default() -> None:
    geometry = _geometry(shape=(12, 20), chunks=(5, 5))

    blocks = partition_indices(geometry, 1, np.array([0, 1, 5, 6, 17]))

    assert [block.bins for block in blocks] == [(0,), (1,), (3,)]
    np.testing.assert_array_equal(blocks[0].indices, np.array([0, 1]))
    np.testing.assert_array_equal(blocks[2].destinations, np.array([4]))


def test_partition_indices_packs_only_adjacent_chunks() -> None:
    geometry = _geometry(shape=(12, 20), chunks=(5, 5))

    blocks = partition_indices(
        geometry,
        1,
        np.array([0, 1, 5, 6, 17]),
        fits=lambda width: width <= 4,
    )

    assert [block.bins for block in blocks] == [(0, 1), (3,)]
    np.testing.assert_array_equal(blocks[0].destinations, np.array([0, 1, 2, 3]))


def test_partition_indices_preserves_destinations_for_unsorted_indices() -> None:
    geometry = _geometry(shape=(12, 15), chunks=(5, 5))
    requested = np.array([12, 1, 6])

    blocks = partition_indices(
        geometry,
        1,
        requested,
        fits=lambda width: width <= 3,
    )

    destinations = np.concatenate([block.destinations for block in blocks])
    indices = np.concatenate([block.indices for block in blocks])
    restored = np.empty(requested.size, dtype=np.int64)
    restored[destinations] = indices

    np.testing.assert_array_equal(restored, requested)


def test_partition_indices_splits_a_chunk_that_is_too_wide_alone() -> None:
    geometry = _geometry(shape=(12, 5), chunks=(5, 5))

    blocks = partition_indices(
        geometry,
        1,
        np.arange(5),
        fits=lambda width: width <= 2,
    )

    assert [block.indices.size for block in blocks] == [2, 2, 1]
    assert [block.bins for block in blocks] == [(0,), (0,), (0,)]


def test_partition_indices_cuts_fixed_width_blocks_in_request_order() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    blocks = partition_indices(geometry, 1, np.arange(10), maxWidth=4)

    assert [block.indices.size for block in blocks] == [4, 4, 2]
    np.testing.assert_array_equal(blocks[1].indices, np.arange(4, 8))
    np.testing.assert_array_equal(blocks[1].destinations, np.arange(4, 8))
    assert blocks[1].bins == (0, 1)


def test_partition_indices_rejects_an_index_that_never_fits() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    with pytest.raises(ValueError, match="does not fit the operation budget"):
        partition_indices(geometry, 1, np.arange(10), fits=lambda _width: False)


def test_partition_indices_rejects_both_width_and_predicate() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    with pytest.raises(ValueError, match="either maxWidth or fits"):
        partition_indices(
            geometry,
            1,
            np.arange(4),
            maxWidth=2,
            fits=lambda _width: True,
        )


def test_partition_indices_returns_nothing_for_an_empty_selection() -> None:
    geometry = _geometry(shape=(12, 10), chunks=(5, 5))

    assert partition_indices(geometry, 1, np.array([], dtype=np.int64)) == []


def test_checked_indices_rejects_malformed_selections() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        checked_indices(np.zeros((2, 2), dtype=np.int64), limit=4, name="rows")
    with pytest.raises(TypeError, match="must contain integers"):
        checked_indices(np.array([0.5]), limit=4, name="rows")
    with pytest.raises(IndexError, match="out-of-range"):
        checked_indices(np.array([4]), limit=4, name="rows")
    with pytest.raises(ValueError, match="duplicate"):
        checked_indices(np.array([1, 1]), limit=4, name="rows")


def test_admit_stream_charges_every_concurrent_chunk_decode() -> None:
    resources = ResourceBudget(1_000, 8)

    admission = admit_stream(
        resources,
        nBlocks=8,
        blockBytes=400,
        decodeBytes=100,
    )

    assert (admission.readWorkers, admission.ioConcurrency) == (2, 1)
    peak = admission.readWorkers * (400 + admission.ioConcurrency * 100)
    assert peak <= resources.memoryBytes


def test_admit_stream_without_decode_bytes_matches_a_flat_task_cost() -> None:
    resources = ResourceBudget(1_000, 8)

    admission = admit_stream(resources, nBlocks=8, blockBytes=250)

    assert admission.readWorkers == 4
    assert admission.readWorkers * 250 <= resources.memoryBytes


def test_admit_stream_honors_resident_bytes_and_a_requested_depth() -> None:
    resources = ResourceBudget(1_000, 8)

    admission = admit_stream(
        resources,
        nBlocks=8,
        blockBytes=100,
        residentBytes=600,
        requested=2,
    )

    assert admission.readWorkers == 2
    assert 600 + admission.readWorkers * 100 <= resources.memoryBytes


def test_admit_stream_reports_when_one_block_cannot_fit() -> None:
    with pytest.raises(MemoryError, match="One unit needs"):
        admit_stream(ResourceBudget(100, 4), nBlocks=4, blockBytes=1_000)


def test_row_band_reproduces_every_expression_it_replaced() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    sharded = root.create_array(
        "sharded",
        shape=(200, 8),
        chunks=(10, 8),
        shards=(50, 8),
        dtype=np.uint32,
    )
    chunked = root.create_array("chunked", shape=(200, 8), chunks=(10, 8), dtype="u4")
    scalar = root.create_array("scalar", shape=(), dtype="u4")
    in_memory = np.zeros((200, 8), dtype=np.uint32)

    for array in (sharded, chunked, scalar, in_memory):
        geometry = array_geometry(array)
        chunks = getattr(array, "chunks", None)
        rows = int(array.shape[0]) if array.ndim else 0

        # cytearc.storage.layout.array_shard_rows
        shards = getattr(getattr(array, "metadata", None), "shards", None)
        if shards is not None and len(shards) > 0:
            expected_shard = int(shards[0])
        elif chunks is not None and len(chunks) > 0:
            expected_shard = int(chunks[0])
        else:
            expected_shard = max(rows, 1)
        assert row_band(geometry, fallback=max(rows, 1)) == expected_shard

        # cytearc.metadata.rows.default_block_rows
        if chunks and len(chunks) > 0 and int(chunks[0]) > 0:
            expected_metadata = int(chunks[0])
        else:
            expected_metadata = max(1, min(rows, 100_000))
        assert (
            row_band(geometry, unit="chunk", fallback=min(rows, 100_000))
            == expected_metadata
        )

        # mapping._projection_block_size, confidence.py and hashing.py
        if chunks is not None and len(chunks) > 0:
            expected_projection = int(chunks[0])
        else:
            expected_projection = min(max(rows, 1), 10_000)
        assert (
            row_band(geometry, unit="chunk", fallback=min(max(rows, 1), 10_000))
            == expected_projection
        )

        # cytearc.storage.artifacts._stored_array_chunk_rows
        if array.ndim > 0 and chunks is not None:
            expected_artifact = max(int(chunks[0]), 1)
        else:
            expected_artifact = 1
        assert row_band(geometry, unit="chunk", fallback=1) == expected_artifact


# A block of four features of 25 cells holds 400 bytes of uint32 values, and
# reading it decodes one chunk of 25 by 4 values, held as stored and decoded:
# the 400 decoded bytes beside the bound of their compressed bytes, 400 + 3 +
# 1,024, and the objects of Zarr's read of one chunk, 32 KiB, 4 KiB for the
# chunk and 16 KiB for its decode.
_BLOCK = 400
_DECODE = 400 + 400 + 3 + 1_024 + 32_768 + 4_096 + 16_384


def _plan_with(
    *,
    resources: ResourceBudget,
    features: np.ndarray,
    batch: int | None = 4,
):
    array = _array(shape=(25, 32), chunks=(25, 4))
    cells = np.arange(25)
    return plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=features,
        cellIndices=cells,
        resources=resources,
        blockBytes=lambda width: width * cells.size * 4,
        requestedBatchSize=batch,
    )


def test_a_stream_that_cannot_be_admitted_reads_one_chunk_at_a_time() -> None:
    # Room for one block beside its decode, but not for a second read on top,
    # so admission fails and the plan must not leave read-ahead concurrency on.
    plan = _plan_with(
        resources=ResourceBudget(_BLOCK + _DECODE + 200, 8),
        features=np.arange(32),
    )

    assert len(plan.blocks) == 8
    assert (plan.readWorkers, plan.ioConcurrency) == (1, 1)


def test_a_stream_without_read_ahead_uses_current_decode_concurrency() -> None:
    # A second block cannot fit, but the current block can afford two decoded
    # chunks. Preserve that inner concurrency without admitting read-ahead.
    plan = _plan_with(
        resources=ResourceBudget(_BLOCK + 2 * _DECODE, 8),
        features=np.arange(32),
    )

    assert len(plan.blocks) == 8
    assert (plan.readWorkers, plan.ioConcurrency) == (1, 2)


def test_a_single_block_stream_budgets_its_own_decode_concurrency() -> None:
    budget = _BLOCK + 3 * _DECODE
    plan = _plan_with(
        resources=ResourceBudget(budget, 8),
        features=np.arange(4),
    )

    assert len(plan.blocks) == 1
    assert plan.readWorkers == 1
    # One block plus three decodes exhausts the limit, so the plan admits
    # three concurrent decodes rather than one per worker.
    assert plan.ioConcurrency == 3
    assert _BLOCK + plan.ioConcurrency * _DECODE <= budget


def test_an_empty_selection_plans_no_read_concurrency() -> None:
    plan = _plan_with(
        resources=ResourceBudget(1_000_000, 8),
        features=np.array([], dtype=np.int64),
    )

    assert plan.blocks == ()
    assert (plan.readWorkers, plan.ioConcurrency) == (1, 1)


def test_streamed_row_blocks_decode_only_as_many_chunks_as_budgeted() -> None:
    store = RecordingStore(delay=0.005)
    root = zarr.open_group(store=store, mode="w")
    array = root.create_array(
        "counts",
        shape=(64, 32),
        chunks=(16, 8),
        dtype=np.uint32,
        compressors=None,
        fill_value=0,
    )
    expected = np.arange(64 * 32, dtype=np.uint32).reshape(64, 32)
    array[:] = expected

    # A row block spans the four column chunks of one chunk row, so an
    # unbounded read can hold all four decodes at once. The budget affords one
    # 2048 byte block beside a single chunk of 512 bytes, held as stored and
    # decoded, 512 + 4 + 1,024 bytes beside it, and the objects of Zarr's read
    # of the four chunks, one decoded at a time, and no more.
    objects = 32_768 + 4 * 4_096 + 16_384
    budgeted = ChunkedArray(
        array,
        resources=ResourceBudget(2_048 + 512 + 512 + 4 + 1_024 + objects + 440, 4),
    )
    store.reset()
    blocks = list(budgeted.stream_blocks())

    np.testing.assert_array_equal(np.vstack(blocks), expected)
    assert store.max_in_flight_for("get") == 1

    unbudgeted = ChunkedArray(array)
    store.reset()
    list(unbudgeted.stream_blocks())

    assert store.max_in_flight_for("get") > 1, (
        "without a budget Zarr uses its own concurrency, so the probe should "
        "observe the overlap this test bounds"
    )


def test_a_stream_that_shrinks_its_blocks_charges_each_size_the_shards_it_touches() -> (
    None
):
    from cytearc.storage.execution import execution_report_scope

    # 20,000 cells by 50 features in shards of one chunk of 100 cells, and a
    # stream of one cell from each of 200 shards in blocks of a shard's 100
    # rows. A block of r of these cells reads r shards, one at a time, so its
    # decode holds the objects of a chunk, 4 KiB, for each of the r - 1 that
    # wait.
    array = zarr.create_array(
        store=MemoryStore(),
        shape=(20_000, 50),
        chunks=(100, 50),
        shards=(100, 50),
        dtype=np.uint16,
    )
    values = (np.arange(20_000 * 50) % 1_000).astype(np.uint16).reshape(20_000, 50)
    array[:] = values
    cells = np.arange(200) * 100 + 7
    matrix = ChunkedArray(array, rows=cells)
    assert matrix._block_size == 100
    one = matrix._with_block_size(1)._max_decode_bytes()
    assert matrix._max_decode_bytes() - one == 99 * 4_096
    resident = matrix._resident_bytes()
    per_row = matrix._block_owned_bytes() // 100
    # A budget of blocks of ten cells, which holds less than the decode of a
    # whole block: each block of ten is charged its own ten shards.
    budget = resident + one + 9 * 4_096 + 10 * per_row
    assert budget < resident + matrix._max_decode_bytes()
    with execution_report_scope() as reports:
        blocks = list(
            matrix._stream_blocks(
                nthreads=1,
                msg=None,
                prefetch=None,
                row_mask=None,
                memory_bytes=budget,
            )
        )
    assert [len(block) for block in blocks] == [10] * 20
    np.testing.assert_array_equal(np.vstack(blocks), values[cells])
    (report,) = reports
    assert report.plan.reservedBytes <= budget


def test_planned_reads_never_decode_more_chunks_than_budgeted() -> None:
    # Budget chosen so admission has to trade read depth against per-read
    # decodes: charging one decode per read admits four concurrent chunk
    # decodes, which overruns the limit once they are actually in flight.
    # Each block holds 1,024 bytes, and each decode a chunk of 256 bytes as
    # stored and decoded, 256 + 2 + 1,024 bytes beside it, and the objects of
    # Zarr's read of that chunk.
    store = RecordingStore(delay=0.005)
    root = zarr.open_group(store=store, mode="w")
    array = root.create_array(
        "counts",
        shape=(64, 32),
        chunks=(16, 4),
        dtype=np.uint32,
        compressors=None,
        fill_value=0,
    )
    array[:] = np.arange(64 * 32, dtype=np.uint32).reshape(64, 32)

    cells = np.arange(64)
    decode = 256 + 256 + 2 + 1_024 + 32_768 + 4_096 + 16_384
    resources = ResourceBudget(3 * (1_024 + decode) + 440, 8)
    block_bytes = 4 * cells.size * 4
    plan = plan_feature_stream(
        array,
        featureAxis=1,
        cellAxis=0,
        featureIndices=np.arange(32),
        cellIndices=cells,
        resources=resources,
        blockBytes=lambda width: width * cells.size * 4,
        requestedBatchSize=4,
    )

    assert len(plan.blocks) == 8
    assert (plan.readWorkers, plan.ioConcurrency) == (3, 1)

    store.reset()
    blocks = list(
        stream_shards(
            range(len(plan.blocks)),
            lambda index: _read_block(array, cells, plan.blocks[index].indices),
            workers=plan.readWorkers,
            io_concurrency=plan.ioConcurrency,
        )
    )

    peak = store.max_in_flight_for("get")
    live = plan.readWorkers * block_bytes + peak * decode

    assert len(blocks) == 8
    assert peak >= 2, "the probe never observed overlapping reads"
    assert peak <= plan.readWorkers * plan.ioConcurrency
    assert live <= resources.memoryBytes
