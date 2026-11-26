"""Streams and the operations on them reserve what they hold within the budget."""

import weakref
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from cytearc.matrix.chunked import ChunkedArray
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.execution import execution_report_scope
from cytearc.storage.geometry import (
    _BAND_LEFTOVER_LIMIT_BYTES,
    ArrayGeometry,
    array_geometry,
)
from cytearc.storage.partition import row_read_parts
from tests.memory_helpers import (
    decodes_held_together,
    held_within_charge,
    traced_charge,
)
from tests.thread_helpers import blas_threads, raised_threads


def _read_objects(chunks: int, decodes: int) -> int:
    """The objects of one Zarr read: 32 KiB, 4 KiB a chunk and 16 KiB a decode at once."""
    return 32_768 + chunks * 4_096 + decodes * 16_384


def test_read_bytes_follow_what_a_zarr_read_holds() -> None:
    sharded = ArrayGeometry(
        shape=(100, 40), chunks=(10, 4), shards=(50, 40), itemsize=2
    )
    nominal = 10 * 4 * 2
    # The result and Zarr's shard-level copy of it, the compressed bytes of
    # the chunks touched in a shard, each at the bound of its encoded size,
    # here 1,024 bytes beyond the 80 it decodes to, and the chunks decoded at
    # once, beside the objects of Zarr's read.
    encoded = 80 + 1_024
    assert sharded.readBytes(1_000, 5) == (
        2 * 1_000 + 5 * encoded + 5 * nominal + _read_objects(5, 5)
    )
    assert sharded.readBytes(1_000, 5, decodes=2) == (
        2 * 1_000 + 5 * encoded + 2 * nominal + _read_objects(5, 2)
    )
    # Zarr's opt-in fused pipeline decodes a shard's chunks one at a time on
    # a plain pool, whatever the concurrency, and the pool's thread may keep
    # the chunk it decoded before.
    plain = replace(sharded, fusedPoolWorkers=2)
    assert plain.poolKeepsDecodedChunk
    assert plain.readBytes(1_000, 5, decodes=2) == (
        2 * 1_000 + 5 * encoded + nominal + nominal + _read_objects(5, 1)
    )
    assert sharded.readBytes(1_000, 5, decodes=9) == sharded.readBytes(1_000, 5)
    # An unsharded read holds its result and each chunk it decodes at once as
    # stored and decoded: the decoded chunk beside the bound of its compressed
    # bytes, the decoded size plus 1/128 of it, rounded down, and 1,024 bytes.
    unsharded = ArrayGeometry(shape=(100, 40), chunks=(10, 4), shards=None, itemsize=2)
    stored = nominal + nominal + 1_024
    assert unsharded.readBytes(1_000, 5) == 1_000 + 5 * stored + _read_objects(5, 5)
    assert unsharded.readBytes(1_000, 5, decodes=1) == (
        1_000 + stored + _read_objects(5, 1)
    )
    # The fused pipeline decodes two of the five chunks at once on a pool of
    # two threads, whose threads may keep the two chunks they decoded before;
    # four at once leave one chunk to decode after them.
    assert replace(unsharded, fusedPoolWorkers=2).readBytes(1_000, 5, decodes=1) == (
        1_000 + 2 * stored + 2 * nominal + _read_objects(5, 2)
    )
    assert replace(unsharded, fusedPoolWorkers=4).readBytes(1_000, 5, decodes=1) == (
        1_000 + 4 * stored + 1 * nominal + _read_objects(5, 4)
    )
    assert unsharded.readBytes(1_000, 1, decodes=1) == (
        1_000 + stored + _read_objects(1, 1)
    )
    assert unsharded.readBytes(1_000, 5, decodes=9) == unsharded.readBytes(1_000, 5)
    # A whole read of the 100 chunks one or ten at a time holds the objects of
    # that read, and on the fused pipeline's pool of ten threads as many
    # previous chunks as it decodes at once, whatever the plan's concurrency.
    assert unsharded.wholeReadBytes(1) == 100 * 40 * 2 + stored + _read_objects(100, 1)
    assert unsharded.wholeReadBytes(10) == (
        100 * 40 * 2 + 10 * stored + _read_objects(100, 10)
    )
    assert replace(unsharded, fusedPoolWorkers=10).wholeReadBytes(1) == (
        100 * 40 * 2 + 10 * (stored + nominal) + _read_objects(100, 10)
    )
    # A chunk of 2,048 bytes is bounded at 16 bytes more than itself, and the
    # scratch a store reports is charged once per decode.
    scratched = ArrayGeometry(
        shape=(640, 8), chunks=(64, 8), shards=None, itemsize=4, readScratchBytes=100
    )
    assert scratched.readBytes(500, 3, decodes=2) == (
        500 + 2 * (2_048 + 2_048 + 16 + 1_024 + 100) + _read_objects(3, 2)
    )
    assert scratched.readBytes(0, 1) == (
        2_048 + 2_048 + 16 + 1_024 + 100 + _read_objects(1, 1)
    )


def test_a_read_on_cytearcs_pools_holds_its_objects_where_a_plain_pool_keeps_a_chunk() -> (
    None
):
    # Chunks of 1,000 by 1,000 uint32 values decode to 4,000,000 bytes and
    # are stored in at most 4,000,000 // 128 + 1,024 more.
    sharded = ArrayGeometry(
        shape=(4_000, 10_000), chunks=(1_000, 1_000), shards=(1_000, 10_000), itemsize=4
    )
    decoded = 4_000_000
    encoded = decoded + decoded // 128 + 1_024
    # Ten chunks decoded one at a time hold no previous chunk on CyteArc's
    # pools, only the objects of Zarr's read: 32 KiB, 4 KiB for each of the
    # ten chunks and 16 KiB for the one decode under way.
    objects = 32_768 + 10 * 4_096 + 16_384
    assert sharded.heldDecodeBytes(10, 1) == objects
    assert sharded.readBytes(0, 10, decodes=1) == 10 * encoded + decoded + objects
    # Four at a time hold 48 KiB more for the three more decodes.
    assert sharded.heldDecodeBytes(10, 4) == objects + 3 * 16_384
    # A plain pool's threads, as Zarr's opt-in fused pipeline runs, also keep
    # the chunk that each decoded before, one for each decode under way.
    own_pool = replace(sharded, fusedPoolWorkers=4)
    assert own_pool.heldDecodeBytes(10, 1) == decoded + objects
    assert own_pool.heldDecodeBytes(10, 4) == 4 * decoded + objects + 3 * 16_384
    # A read that decodes every chunk at once has no previous chunk either way.
    every = 32_768 + 10 * 4_096 + 10 * 16_384
    assert sharded.heldDecodeBytes(10, 10) == own_pool.heldDecodeBytes(10, 10) == every
    # A whole read of an unsharded array of four 40,000,000-byte chunks, two
    # at a time, as the memory ledger plans one, holds the 160,000,000-byte
    # result and the two chunks under way, each as stored and decoded, beside
    # the read's objects, where a plain pool also keeps one earlier chunk for
    # each decode under way.
    # A whole read of the sharded array, four decodes at a time, reads a band
    # of 1,000 rows at a time: the 160,000,000-byte result, the band's
    # 40,000,000 bytes twice over, each of its ten chunks as stored and
    # decoded, and the four chunks under way, beside the objects of that
    # read; the ten chunks cover any chunk that a plain pool keeps. The reads
    # of the three earlier bands leave their objects: 32 KiB, and 4 KiB for
    # each of their thirty chunks.
    whole = (
        160_000_000
        + 2 * 40_000_000
        + 10 * (encoded + decoded)
        + 4 * decoded
        + (32_768 + 10 * 4_096 + 4 * 16_384)
        + (32_768 + 30 * 4_096)
    )
    assert sharded.wholeReadBytes(4) == own_pool.wholeReadBytes(4) == whole
    unsharded = replace(sharded, shards=None, chunks=(1_000, 10_000))
    stored = 40_000_000 + 40_000_000 + 40_000_000 // 128 + 1_024
    read_objects = 32_768 + 4 * 4_096 + 2 * 16_384
    assert unsharded.wholeReadBytes(2) == 160_000_000 + 2 * stored + read_objects
    assert replace(unsharded, fusedPoolWorkers=2).wholeReadBytes(2) == (
        160_000_000 + 2 * stored + 2 * 40_000_000 + read_objects
    )
    # An unsharded array of a local store reads each chunk as a file, so a
    # buffer of 4,096 bytes is held for each of the chunks read at once: two
    # in a whole read, and four of a band of ten chunks.
    buffered = replace(unsharded, readScratchBytes=4_096)
    assert buffered.wholeReadBytes(2) == unsharded.wholeReadBytes(2) + 2 * 4_096
    band = replace(sharded, shards=None)
    assert replace(band, readScratchBytes=4_096).bandReadBytes(1_000, 4) == (
        band.bandReadBytes(1_000, 4) + 4 * 4_096
    )


def test_a_read_through_zarrs_fused_pipeline_is_charged_at_the_reads_it_runs_at_once() -> (
    None
):
    # Chunks of 256 by 1,024 uint32 values decode to 1 MiB and are stored in
    # at most 1 MiB // 128 + 1,024 more.
    decoded = 1_048_576
    stored = 2 * decoded + decoded // 128 + 1_024
    result = 8 * decoded
    unsharded = ArrayGeometry((2_048, 1_024), (256, 1_024), None, 4)
    # On a store that reads synchronously, Zarr's fused pipeline reads as
    # many of the eight chunks at once as its pool has threads, four,
    # whatever Zarr's async concurrency, and its threads may keep the chunk
    # that each decoded before.
    fused = replace(unsharded, fusedPoolWorkers=4)
    assert fused.readsAtOnce(8, 1) == (4, 1)
    assert fused.readBytes(result, 8, decodes=1) == (
        result + 4 * stored + _read_objects(8, 4) + 4 * decoded
    )
    assert fused.readBytes(result, 8, decodes=10) == fused.readBytes(result, 8, 1)
    assert fused.wholeReadBytes(1) == fused.readBytes(result, 8, decodes=1)
    # One worker reads inline, a chunk at a time.
    inline = replace(unsharded, fusedPoolWorkers=1)
    assert inline.readBytes(result, 8, decodes=4) == (
        result + stored + _read_objects(8, 1)
    )
    # On a store that reads asynchronously, it decodes all eight before it
    # places any.
    gathering = replace(fused, fusedReadsAsync=True)
    assert gathering.readsAtOnce(8, 1) == (8, 1)
    assert gathering.readBytes(result, 8, decodes=1) == (
        result + 8 * stored + _read_objects(8, 8)
    )
    # Shards of four chunks, of which a read touches four: a synchronous
    # store reads as many shards at once as the pool has threads, each
    # decoding its chunks one at a time, beside one shard-level copy of the
    # selection.
    sharded = ArrayGeometry((4_096, 1_024), (256, 1_024), (1_024, 1_024), 4)
    shard_result = 16 * decoded
    fused_shards = replace(sharded, fusedPoolWorkers=4)
    assert fused_shards.readsAtOnce(4, 1, shards=4) == (1, 4)
    # Zarr's async concurrency changes neither.
    assert fused_shards.readsAtOnce(4, 10, shards=4) == (1, 4)
    one_shard = 4 * (stored - decoded) + decoded + _read_objects(4, 1) + decoded
    assert fused_shards.readBytes(shard_result, 4, decodes=1, shards=4) == (
        2 * shard_result + 4 * one_shard
    )
    # A plan of row blocks passes the shards that a block touches, and the
    # charge reads as many at once as the pool has threads.
    assert row_read_parts(
        fused_shards, np.arange(1_024), rows_contiguous=True, concurrency=1, shards=4
    ) == (4 * one_shard, 1_024 * 4)
    # A band of four 256 KiB chunks of an unsharded array decodes all four at
    # once on a pool of four threads.
    small = 2 * 262_144 + 262_144 // 128 + 1_024
    banded = ArrayGeometry((2_048, 1_024), (256, 256), None, 4, fusedPoolWorkers=4)
    assert banded.bandReadBytes(256, 1) == (
        256 * 1_024 * 4 + 4 * small + _read_objects(4, 4)
    )
    # An asynchronous store reads the shards at Zarr's async concurrency,
    # each decoding as many chunks at once as the pool has threads, while
    # the task that Zarr makes for each shard it does not read yet holds the
    # objects of a chunk, 4 KiB.
    gathering_shards = replace(fused_shards, fusedReadsAsync=True)
    gathering_shard = 4 * (stored - decoded) + 4 * decoded + _read_objects(4, 4)
    assert gathering_shards.readsAtOnce(4, 1, shards=4) == (4, 1)
    assert gathering_shards.readBytes(shard_result, 4, decodes=1, shards=4) == (
        2 * shard_result + gathering_shard + 3 * 4_096
    )
    assert gathering_shards.readsAtOnce(4, 2, shards=4) == (4, 2)
    assert gathering_shards.readBytes(shard_result, 4, decodes=2, shards=4) == (
        2 * shard_result + 2 * gathering_shard + 2 * 4_096
    )
    # CyteArc's pipelines read the shards a read touches, and the chunks of
    # each, at Zarr's async concurrency.
    assert sharded.readsAtOnce(4, 2, shards=4) == (2, 2)
    assert sharded.readBytes(shard_result, 4, decodes=2, shards=4) == (
        2 * shard_result
        + 2 * (4 * (stored - decoded) + 2 * decoded + _read_objects(4, 2))
        + 2 * 4_096
    )
    assert sharded.readsAtOnce(4, None, shards=3) == (4, 3)


@pytest.mark.parametrize(
    ("store", "workers", "concurrency"),
    [
        ("synchronous", 1, 1),
        ("synchronous", 4, 1),
        ("asynchronous", 4, 1),
        ("asynchronous", 8, 2),
    ],
    ids=[
        "local store, one worker",
        "local store, four workers",
        "async store, four workers",
        "async store, eight workers, two reads at once",
    ],
)
@pytest.mark.parametrize("sharded", [False, True], ids=["unsharded", "sharded"])
def test_a_read_through_zarrs_fused_pipeline_holds_no_more_than_its_charge(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    store: str,
    workers: int,
    concurrency: int,
    sharded: bool,
) -> None:
    import gc
    import tracemalloc

    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from fsspec.implementations.local import LocalFileSystem
    from zarr.storage import FsspecStore

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.virtual_arrays import read_selection
    from tests.memory_helpers import decodes_on_threads_held_together

    import zarr.core.codec_pipeline as codec_pipeline

    # Sixteen chunks of 64 by 1,024 uint32 random bits, 256 KiB each, in
    # shards of four or each on its own, read whole by a plan of one or two
    # reads at once while Zarr's fused pipeline is configured, on a pool of
    # the test's own, which Zarr makes afresh and the test then gives back.
    monkeypatch.setattr(codec_pipeline, "_pool", None)
    monkeypatch.setattr(codec_pipeline, "_pool_size", 0)
    settings = {
        "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
        "codec_pipeline.max_workers": workers,
    }
    location = str(tmp_path / "values.zarr")
    if store == "synchronous":
        opened: Any = LocalStore(location)
    else:
        files = AsyncFileSystemWrapper(
            LocalFileSystem(auto_mkdir=True), asynchronous=True
        )
        opened = FsspecStore(files, path=location)
    values = np.random.default_rng(3).integers(
        0, 1 << 32, size=(1_024, 1_024), dtype=np.uint32
    )
    with zarr.config.set(settings):
        array = zarr.open_group(store=opened, mode="w").create_array(
            "values",
            shape=values.shape,
            chunks=(64, 1_024),
            shards=(256, 1_024) if sharded else None,
            dtype=np.uint32,
        )
        array[:] = values
        geometry = array_geometry(array)
        assert geometry is not None and geometry.fusedPoolWorkers == workers
        assert geometry.fusedReadsAsync is (store == "asynchronous")
        (outer, *_) = array.metadata.codecs
        codecs = outer.codecs if sharded else array.metadata.codecs
        # Each decode on the pool holds its chunk until as many decodes as
        # the pool has workers hold theirs, so the read holds that many at
        # once on any machine: two shards of four chunks at two reads at once.
        with (
            zarr_io_concurrency(concurrency),
            decodes_on_threads_held_together(type(codecs[-1]), workers),
        ):
            read_selection(array, slice(None), slice(None))
            gc.collect()
            tracemalloc.start()
            try:
                base, _ = tracemalloc.get_traced_memory()
                block = read_selection(array, slice(None), slice(None))
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
    np.testing.assert_array_equal(block, values)

    def charge(of: ArrayGeometry) -> int:
        if sharded:
            # The read touches four shards, four chunks in each.
            return of.readBytes(block.nbytes, 4, decodes=concurrency, shards=4)
        return of.readBytes(block.nbytes, 16, decodes=concurrency)

    assert held_within_charge(peak - base, charge(geometry))
    if workers > 1:
        # A charge that followed the plan's reads at once falls short.
        earlier = replace(geometry, fusedPoolWorkers=0, fusedReadsAsync=False)
        assert peak - base > charge(earlier)


@pytest.mark.parametrize("concurrency", [1, 10])
@pytest.mark.parametrize("n_chunks", [1, 10, 40])
@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "unsharded"])
def test_the_objects_of_a_zarr_read_lie_within_what_object_bytes_charges(
    tmp_path: Any, sharded: bool, n_chunks: int, concurrency: int
) -> None:
    import gc
    import os
    import sys
    import tracemalloc

    from zarr.codecs import ZstdCodec

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.virtual_arrays import read_selection

    if sys.gettrace() is not None:
        pytest.skip(
            "a trace function, such as the C tracer that coverage installs, "
            "allocates inside the traced window"
        )
    # A row of chunks of 1 by 4 uint32 values, 16 bytes each, so that what a
    # read of the row holds beyond its buffers is the objects of Zarr's read.
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    array = root.create_array(
        "counts",
        shape=(2, 4 * n_chunks),
        chunks=(1, 4),
        shards=(1, 4 * n_chunks) if sharded else None,
        dtype=np.uint32,
        compressors=ZstdCodec(),
    )
    values = np.arange(8 * n_chunks, dtype=np.uint32).reshape(2, 4 * n_chunks)
    array[:] = values
    stored = sum(
        os.path.getsize(tmp_path / "counts" / "c" / "0" / name)
        for name in (["0"] if sharded else [str(chunk) for chunk in range(n_chunks)])
    )
    geometry = array_geometry(array)
    assert geometry is not None
    decodes = min(concurrency, n_chunks)
    charged = geometry.objectBytes(n_chunks, decodes)
    traced = []
    with zarr_io_concurrency(concurrency):
        # Two reads make what every read shares.
        for _ in range(2):
            read_selection(array, slice(0, 1), slice(None))
        for _ in range(3):
            # A full collection empties CPython's free lists, so that the
            # traced read allocates each of its objects anew.
            gc.collect()
            tracemalloc.start()
            try:
                start = tracemalloc.get_traced_memory()[0]
                block = read_selection(array, slice(0, 1), slice(None))
                traced.append(tracemalloc.get_traced_memory()[1] - start)
            finally:
                tracemalloc.stop()
            np.testing.assert_array_equal(block, values[:1])
            del block
    # tracemalloc counts what every thread allocates, which can only add
    # bytes, so the smallest of the three reads is the one that other threads
    # disturbed least.
    held = min(traced)
    # The buffers: the row, and Zarr's shard-level copy of it, the stored
    # chunks, and the chunks decoded at once, under 2 KB in all.
    buffers = (2 if sharded else 1) * values[:1].nbytes + stored + decodes * 16
    # Within half of the charge, a charge that drifted up from Zarr's objects
    # fails too, as one that falls short of them does.
    assert charged / 2 <= held - buffers
    assert held_within_charge(held - buffers, charged)


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
    assert held_within_charge(peak, matrix.stream_bytes())


@pytest.mark.parametrize("profile", ["fast_local", "cloud"])
def test_a_band_read_of_a_sharded_array_holds_no_more_than_its_charge(
    tmp_path: Any, profile: str
) -> None:
    from cytearc.storage.arrays import create_numeric_array
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.layout import row_sharded_array_spec

    # Random bits do not compress, so each of the shard's two inner chunks is
    # stored a few bytes larger than the 8 MiB it decodes to.
    spec = row_sharded_array_spec(
        (8_192, 512),
        np.float32,
        profile=profile,
        band_rows=8_192,
        target_chunk_bytes=8 * 1024**2,
    )
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    stored = create_numeric_array(root, "data", spec)
    assert (stored.shards, stored.chunks) == ((8_192, 512), (4_096, 512))
    bits = np.random.default_rng(0).integers(
        0, 1 << 32, size=spec.shape, dtype=np.uint32
    )
    stored[:] = bits.view(np.float32)
    del bits
    geometry = array_geometry(stored)
    assert geometry is not None

    with zarr.config.set({"async.concurrency": 1}):
        held = traced_charge(lambda _ledger: stored[:, :]).held

    # The read held its result, Zarr's shard-level copy of it, the shard's
    # two compressed chunks and one decoded chunk, more than those alone. No
    # pool thread keeps the first chunk once the read has copied it, so the
    # charge for one decode at a time covers all of it with the read's
    # objects, 88 KiB, in place of the decoded chunk that it reserved before.
    decoded = 4_096 * 512 * 4
    band_bytes = 8_192 * 512 * 4
    assert held > 2 * band_bytes + 3 * decoded
    assert held_within_charge(held, geometry.readBytes(band_bytes, 2, decodes=1))
    assert geometry.readBytes(band_bytes, 2, decodes=1) - decoded < held


def test_a_whole_read_of_an_unsharded_array_holds_no_decoded_chunk_beside_its_decodes(
    tmp_path: Any,
) -> None:
    import tracemalloc

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.geometry import array_geometry

    # Four chunks of random bits, 1 MiB each, which a whole read decodes two
    # at a time, as the memory ledger plans whole reads of normalized values.
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    stored = root.create_array(
        "values", shape=(1_024, 1_024), chunks=(256, 1_024), dtype=np.uint32
    )
    stored[:] = np.random.default_rng(1).integers(
        0, 1 << 32, size=stored.shape, dtype=np.uint32
    )
    geometry = array_geometry(stored)
    assert geometry is not None and not geometry.poolKeepsDecodedChunk
    # How many decodes overlap depends on the machine's threads, so each of
    # the two decodes under way holds its chunk until the other holds its own.
    compressor = type(stored.metadata.codecs[-1])
    with zarr_io_concurrency(2), decodes_held_together(compressor, 2):
        stored[:, :]
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            values = stored[:, :]
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    decoded = 256 * 1_024 * 4
    # The result and two chunks under way, each compressed and decoded, and
    # no more than the read's objects beside them: the plan covers it without
    # the earlier decoded chunk that it reserved for each decode before.
    assert values.nbytes == 4 * decoded
    assert peak - base > values.nbytes + 2 * 2 * decoded
    assert held_within_charge(peak - base, geometry.wholeReadBytes(2))
    assert geometry.wholeReadBytes(2) - decoded < peak - base


def test_read_bytes_charge_the_indexers_and_copies_of_reads_by_index_arrays() -> None:
    sharded = ArrayGeometry(
        shape=(100, 40), chunks=(10, 4), shards=(50, 40), itemsize=2
    )
    unsharded = replace(sharded, shards=None)
    # Zarr's indexers hold 64 bytes for each position of a sharded read, its
    # own and its shard's, and 40 for each of an unsharded one.
    assert sharded.indexBytes(1_000) == 64_000
    assert unsharded.indexBytes(1_000) == 40_000
    assert sharded.indexBytes(-5) == 0
    # A read of slices places views of its chunks. One index array copies
    # the values it takes from a chunk, here 7 rows of 3 two-byte values;
    # two also hold NumPy's buffers of 8,192 elements of each array, 8 bytes
    # each.
    assert sharded.takenChunkBytes(7, 3, indexedAxes=0) == 0
    assert sharded.takenChunkBytes(7, 3, indexedAxes=1) == 42
    assert unsharded.takenChunkBytes(7, 3, indexedAxes=2) == 42 + 8_192 * (8 + 8)
    assert unsharded.takenChunkBytes(0, 3, indexedAxes=2) == 8_192 * 16
    # Every row of a block counts, beyond a chunk's 10 rows, so that a stream
    # that shrinks its blocks to a budget keeps each block's share of the
    # charge no less than its copy.
    assert sharded.takenChunkBytes(25, 3, indexedAxes=1) == 25 * 3 * 2
    # The indexers are held once, and so is the copy: CyteArc's pipelines
    # copy each decoded chunk's values on the read's event loop, a chunk at a
    # time, however many chunks they decode at once.
    nominal = 10 * 4 * 2
    encoded = 80 + 1_024
    assert sharded.readBytes(1_000, 5, decodes=2, indexes=30, takenBytes=42) == (
        2 * 1_000 + 5 * encoded + 2 * nominal + _read_objects(5, 2) + 42 + 30 * 64
    )
    stored = nominal + nominal + 1_024
    assert unsharded.readBytes(1_000, 5, decodes=2, indexes=30, takenBytes=42) == (
        1_000 + 2 * stored + _read_objects(5, 2) + 42 + 30 * 40
    )
    # Two of three shards are read at once, and the task of the third holds
    # the objects of a chunk, 4 KiB, until it is read.
    assert sharded.readBytes(
        1_000, 5, decodes=2, shards=3, indexes=30, takenBytes=42
    ) == (
        2 * 1_000
        + 2 * (5 * encoded + 2 * nominal + _read_objects(5, 2))
        + 4_096
        + 42
        + 30 * 64
    )
    # Zarr's fused pipeline copies them on its pool's threads: on a store
    # that reads synchronously, a pool of two threads reads two of the three
    # shards at once, each decoding a chunk at a time beside the chunk that
    # its thread decoded before.
    fused = replace(sharded, fusedPoolWorkers=2)
    assert fused.readBytes(
        1_000, 5, decodes=2, shards=3, indexes=30, takenBytes=42
    ) == (
        2 * 1_000
        + 2 * (5 * encoded + nominal + nominal + _read_objects(5, 1))
        + 4_096
        + 2 * 42
        + 30 * 64
    )
    assert sharded.readBytes(1_000, 5, decodes=2) == sharded.readBytes(
        1_000, 5, decodes=2, indexes=0, takenBytes=0
    )


def _random_bits_store(
    tmp_path: Any,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
    dtype: Any,
) -> tuple[zarr.Array, np.ndarray]:
    """Store random bits, which no compressor shrinks, so the stored chunks are as large as they decode to."""
    bits = np.random.default_rng(0).integers(0, 1 << 32, size=shape, dtype=np.uint32)
    values = bits.view(np.float32) if dtype == np.float32 else bits.astype(dtype)
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    stored = root.create_array(
        "values", shape=shape, chunks=chunks, shards=shards, dtype=dtype
    )
    stored[:] = values
    return stored, values


def _traced_block_read(block: ChunkedArray) -> tuple[np.ndarray, int]:
    """Read ``block`` whole at one decode at a time, as row-block streams read, once to warm Zarr and once traced."""
    import tracemalloc

    from cytearc.storage.async_execution import zarr_io_concurrency

    with zarr_io_concurrency(1):
        block._read(0, block.shape[0])
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            values = block._read(0, block.shape[0])
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return values, peak - base


@pytest.mark.parametrize("selected", ["rows", "columns", "both"])
@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "unsharded"])
def test_a_read_of_selected_cells_holds_no_more_than_its_charge(
    tmp_path: Any, sharded: bool, selected: str
) -> None:
    # 4,000 cells by 256 features in chunks of 2,000 cells and 128 features,
    # 1 MB each, in shards of a band of cells or none. The block reads every
    # cell of the first chunk band in a shuffled order, every other feature,
    # or both.
    stored, values = _random_bits_store(
        tmp_path,
        (4_000, 256),
        (2_000, 128),
        (2_000, 256) if sharded else None,
        np.float32,
    )
    matrix = ChunkedArray(stored)
    cells = np.random.default_rng(1).permutation(2_000)
    features = np.arange(0, 256, 2)
    block = {
        "rows": matrix[cells, :],
        "columns": matrix[:2_000, features],
        "both": matrix[cells, :][:, features],
    }[selected]
    read, peak = _traced_block_read(block)
    expected = values[
        np.ix_(
            np.arange(2_000) if selected == "columns" else cells,
            np.arange(256) if selected == "rows" else features,
        )
    ]
    np.testing.assert_array_equal(read.view(np.uint32), expected.view(np.uint32))

    # Beside a read of the same block of a run of cells and of as many
    # features in the same chunks, the read holds Zarr's indexers of the
    # cells or features that it takes by an index array, 64 bytes each in a
    # sharded read and 40 in an unsharded one, and its copy of what it takes
    # from each chunk: 2,000 cells of 128 features, or of the 64 selected
    # features of a chunk. Two index arrays on an unsharded read also hold
    # NumPy's buffers of 8,192 elements of each array.
    charge = block._read_bytes(read.nbytes)
    assert held_within_charge(peak, charge)
    run = matrix[:2_000, :] if selected == "rows" else matrix[:2_000, 64:192]
    copy = 2_000 * (128 if selected == "rows" else 64) * 4
    if selected == "both" and sharded:
        # A sharded read of both is read in bands, charged beside the read of
        # a block of a run of cells.
        assert charge >= run._read_bytes(read.nbytes) + 128 * 64 + copy
        return
    indexes = {"rows": 2_000, "columns": 128, "both": 2_128}[selected]
    buffers = 8_192 * (8 + 8) if selected == "both" else 0
    assert charge - run._read_bytes(read.nbytes) == (
        indexes * (64 if sharded else 40) + copy + buffers
    )
    if selected == "rows":
        # A plan of a read of some of a matrix's rows by an index array, as
        # a merge plans the cells it holds, charges the same.
        assert run._read_bytes(read.nbytes, rows_indexed=True) == charge
    # Without the copy, the charge would not cover the read.
    assert peak > charge - copy


@pytest.mark.parametrize(
    ("chunks", "shards", "touched"),
    [
        pytest.param((100, 10), (100, 20), 100, id="one-column-shard-100"),
        pytest.param(
            (100, 10), (100, 20), 300, id="one-column-shard-300", marks=pytest.mark.slow
        ),
        pytest.param((100, 4), (100, 4), 40, id="five-column-shards-40"),
        pytest.param(
            (100, 4), (100, 4), 300, id="five-column-shards-300", marks=pytest.mark.slow
        ),
    ],
)
def test_a_read_of_cells_spread_over_shards_holds_a_task_for_each_shard(
    tmp_path: Any,
    touched: int,
    chunks: tuple[int, int],
    shards: tuple[int, int],
) -> None:
    # 100 cells for each row of shards that the block reads, by 20 features
    # in shards of 100 cells, of every feature in two chunks or of four
    # features in one; a block of one cell from each of 40, 100 or 300 rows
    # of shards, and a block of as many shuffled cells of the first one to
    # three rows.
    stored, values = _random_bits_store(
        tmp_path, (touched * 100, 20), chunks, shards, np.float32
    )
    column_shards = 20 // shards[1]
    spread = np.arange(touched) * 100 + 7
    block = ChunkedArray(stored, rows=spread, block_size=touched)
    read, peak = _traced_block_read(block)
    np.testing.assert_array_equal(read.view(np.uint32), values[spread].view(np.uint32))
    clustered = np.random.default_rng(7).permutation(touched)
    fewer = ChunkedArray(stored, rows=clustered, block_size=touched)
    # The read takes the shards one at a time, and the task that Zarr makes
    # for each of the others, in every column shard, holds the objects of a
    # chunk, 4 KiB, until it is read; as many cells of fewer rows of shards
    # hold the same indexers and copies.
    charge = block._read_bytes(read.nbytes)
    filled = -(-touched // 100)
    assert charge - fewer._read_bytes(read.nbytes) == (
        column_shards * (touched - filled) * 4_096
    )
    # A stream of blocks of ten of these cells charges each block the shards
    # of ten rows, where blocks of one cell charge those of one row.
    tens = ChunkedArray(stored, rows=spread, block_size=10)
    ones = ChunkedArray(stored, rows=spread, block_size=1)
    assert tens._read_bytes(0) - ones._read_bytes(0) == 9 * column_shards * 4_096
    assert held_within_charge(peak, charge)
    if column_shards == 1:
        # Without those tasks the charge would not cover the read, even with
        # the file buffer that it charges a read of part of a shard, which a
        # read of whole shards, as here, holds no longer from Python 3.14.
        geometry = array_geometry(stored)
        assert geometry is not None
        assert peak > charge - geometry.readScratchBytes - (touched - 1) * 4_096
    else:
        # A charge of one shard in each row of shards, without the tasks of
        # the other four column shards, falls short.
        assert peak > charge - (column_shards - 1) * touched * 4_096


@pytest.mark.parametrize(
    ("chunk", "touched", "left", "short"),
    [
        # Two chunks a band, 100 bands: the reads of the 99 bands before the
        # last took 198 chunks, whose objects pass the limit at which what
        # reads in bands leave levels off, where as many cells of one shard
        # are one band and leave none; a charge without them falls short.
        pytest.param(
            10,
            100,
            _BAND_LEFTOVER_LIMIT_BYTES,
            _BAND_LEFTOVER_LIMIT_BYTES,
            id="two-chunks-100-bands",
        ),
        # Five chunks a band, 1,000 bands, against as many cells of the first
        # ten shards, whose nine bands before the last leave the objects of a
        # read, 32 KiB, and of their 45 chunks, 4 KiB each; a charge of what
        # whole reads of one-dimensional arrays leave, up to 192 KiB, falls
        # short.
        pytest.param(
            4,
            1_000,
            _BAND_LEFTOVER_LIMIT_BYTES - (32 * 1024 + 45 * 4_096),
            _BAND_LEFTOVER_LIMIT_BYTES - 192 * 1024,
            id="five-chunks-1000-bands",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_a_read_in_bands_of_cells_spread_over_shards_holds_what_the_bands_before_it_leave(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    chunk: int,
    touched: int,
    left: int,
    short: int,
) -> None:
    import sys

    from cytearc.storage import virtual_arrays
    from tests.memory_helpers import traced_selection_read

    # 100 cells for each shard that the block reads, by 20 features in shards
    # of 100 cells of 10-feature or 4-feature chunks; a block of one cell
    # from each of 100 or 1,000 shards by every other feature, which a
    # sharded read of both takes in bands of rows, each in one shard, two or
    # five chunks of it, and a block of as many shuffled cells of the first
    # shards.
    stored, values = _random_bits_store(
        tmp_path, (touched * 100, 20), (100, chunk), (100, 20), np.float32
    )
    columns = np.arange(0, 20, 2)
    sliced: list[int] = []
    read_bands = virtual_arrays._read_bands

    def recording(*args: Any, axis: int, **kwargs: Any) -> np.ndarray:
        sliced.append(axis)
        return read_bands(*args, axis=axis, **kwargs)

    monkeypatch.setattr(virtual_arrays, "_read_bands", recording)
    spread = np.arange(touched) * 100 + 7
    block = ChunkedArray(stored, rows=spread, cols=columns, block_size=touched)
    # Each traced read starts after a full collection, which empties CPython's
    # free lists, so the reads of the earlier bands fill them again.
    read, peak = traced_selection_read(stored, spread, columns)
    np.testing.assert_array_equal(
        read.view(np.uint32), values[np.ix_(spread, columns)].view(np.uint32)
    )
    assert set(sliced) == {0}
    charge = block._read_bytes(read.nbytes)
    assert held_within_charge(peak, charge)
    # No band waits on another shard, so the block is charged as many cells
    # of the first shards are but for what the earlier bands leave.
    clustered = np.random.default_rng(7).permutation(touched)
    fewer = ChunkedArray(stored, rows=clustered, cols=columns, block_size=touched)
    assert charge - fewer._read_bytes(read.nbytes) == left
    if sys.version_info < (3, 14):
        # On Python 3.13 the read holds those objects beyond the rest of its
        # charge; the reads leave few of them on Python 3.14.
        assert peak > charge - short


@pytest.mark.parametrize(
    ("layout", "shape", "chunks", "shards"),
    [
        ("unsharded, four column chunks", (2_000, 512), (2_000, 128), None),
        ("unsharded, four row chunks", (8_000, 128), (2_000, 128), None),
        ("sharded, four one-chunk shards", (8_000, 128), (2_000, 128), (2_000, 128)),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_read_of_cells_on_zarrs_fused_pipeline_is_charged_its_pools_reads(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
) -> None:
    import gc
    import tracemalloc

    import zarr.core.codec_pipeline as codec_pipeline

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.geometry import array_geometry
    from tests.memory_helpers import decodes_on_threads_held_together

    # Shuffled cells of four chunks, or of four shards, read as one block on
    # Zarr's fused pipeline with a pool of four threads of the test's own,
    # each decode holding its chunk until the four hold theirs.
    monkeypatch.setattr(codec_pipeline, "_pool", None)
    monkeypatch.setattr(codec_pipeline, "_pool_size", 0)
    settings = {
        "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
        "codec_pipeline.max_workers": 4,
    }
    cells = np.random.default_rng(8).permutation(shape[0])
    with zarr.config.set(settings):
        stored, values = _random_bits_store(tmp_path, shape, chunks, shards, np.float32)
        geometry = array_geometry(stored)
        assert geometry is not None and geometry.fusedPoolWorkers == 4
        (outer, *_) = stored.metadata.codecs
        codecs = outer.codecs if shards is not None else stored.metadata.codecs
        block = ChunkedArray(stored, rows=cells, block_size=shape[0])
        with (
            zarr_io_concurrency(1),
            decodes_on_threads_held_together(type(codecs[-1]), 4),
        ):
            block._read(0, shape[0])
            gc.collect()
            tracemalloc.start()
            try:
                base, _ = tracemalloc.get_traced_memory()
                read = block._read(0, shape[0])
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        np.testing.assert_array_equal(
            read.view(np.uint32), values[cells].view(np.uint32)
        )
        charge = block._read_bytes(read.nbytes)
        # A charge of one decode at a time, as CyteArc's pipelines read, falls
        # short of the pool's four.
        monkeypatch.setattr(
            block,
            "_geometry",
            lambda: replace(geometry, fusedPoolWorkers=0, fusedReadsAsync=False),
        )
        one_at_a_time = block._read_bytes(read.nbytes)
    assert held_within_charge(peak - base, charge)
    assert peak - base > one_at_a_time


def test_a_read_of_a_run_of_cells_given_as_an_index_array_holds_its_charge(
    tmp_path: Any,
) -> None:
    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.partition import band_shards
    from cytearc.storage.virtual_arrays import read_selection

    # 100,000 cells by 4 uint16 features in a shard of four one-feature
    # chunks: a run of every cell given as an index array, by one feature,
    # which the read takes as a slice once it finds the run.
    stored, values = _random_bits_store(
        tmp_path, (100_000, 4), (100_000, 1), (100_000, 4), np.uint16
    )
    geometry = array_geometry(stored)
    cells = np.arange(100_000)
    feature = np.array([2])
    with zarr_io_concurrency(1):
        read = read_selection(stored, cells, feature)
        held = traced_charge(
            lambda _ledger: read_selection(stored, cells, feature)
        ).held
    np.testing.assert_array_equal(read, values[:, [2]])
    fixed, per_row = row_read_parts(
        geometry,
        feature,
        rows_contiguous=True,
        shards=band_shards(geometry, cells, cells.size),
    )
    # Finding the run takes its steps 65,536 at a time; all at once they
    # would hold 9 bytes a cell, 900 KB, beyond the read's charge.
    assert held_within_charge(held, read.nbytes + fixed + cells.size * per_row)


@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "unsharded"])
def test_reads_of_selected_cells_copy_one_chunk_at_a_time_however_many_they_decode(
    tmp_path: Any, sharded: bool
) -> None:
    import tracemalloc

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.virtual_arrays import read_selection

    # 2,000 cells by 512 features in four chunks of 128 features, which a
    # read at a concurrency of 4 decodes at once.
    stored, values = _random_bits_store(
        tmp_path,
        (2_000, 512),
        (2_000, 128),
        (2_000, 512) if sharded else None,
        np.float32,
    )
    geometry = array_geometry(stored)
    cells = np.random.default_rng(4).permutation(2_000)
    features = np.arange(512)
    # How many decodes overlap depends on how the machine schedules their
    # threads, so the four decodes are made to hold their chunks together.
    codecs = stored.metadata.codecs[0].codecs if sharded else stored.metadata.codecs
    with zarr_io_concurrency(4), decodes_held_together(type(codecs[-1]), 4):
        read_selection(stored, cells, features)
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            read = read_selection(stored, cells, features)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    np.testing.assert_array_equal(read.view(np.uint32), values[cells].view(np.uint32))
    fixed, per_row = row_read_parts(
        geometry, features, rows_contiguous=False, concurrency=4
    )
    charge = read.nbytes + fixed + cells.size * per_row
    # The read's event loop copies the cells' values of one decoded chunk at
    # a time, 2,000 x 128 x 4 bytes, so the charge holds one such copy, and
    # without it would not cover the read.
    copy = 2_000 * 128 * 4
    assert held_within_charge(peak - base, charge)
    assert peak - base > charge - copy


@pytest.mark.parametrize("selected", ["rows", "both"])
@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "unsharded"])
def test_a_read_of_many_cells_of_a_narrow_matrix_holds_their_indexers(
    tmp_path: Any, sharded: bool, selected: str
) -> None:
    # 100,000 cells by 4 uint16 features in chunks of every cell and one
    # feature: a read of every cell in a shuffled order holds far more
    # integers indexing the cells than values.
    stored, values = _random_bits_store(
        tmp_path,
        (100_000, 4),
        (100_000, 1),
        (100_000, 4) if sharded else None,
        np.uint16,
    )
    cells = np.random.default_rng(2).permutation(100_000)
    features = np.array([0, 2, 3])
    matrix = ChunkedArray(stored)
    block = matrix[cells, :] if selected == "rows" else matrix[cells, :][:, features]
    read, peak = _traced_block_read(block)
    np.testing.assert_array_equal(
        read, values[cells] if selected == "rows" else values[np.ix_(cells, features)]
    )
    charge = block._read_bytes(read.nbytes)
    assert held_within_charge(peak, charge)
    if selected == "both" and sharded:
        # A sharded read of both is read in bands of the one chunk band of
        # cells, which index the three features, rather than in bands of a
        # feature, which would index every cell at 64 bytes each. Ordering
        # the cells by chunk band keeps 32 bytes of each.
        assert peak < 64 * 100_000
        assert peak > charge - 32 * 100_000
    else:
        # Zarr's indexers of the cells, 64 bytes a cell in a sharded read and
        # 40 in an unsharded one, outweigh everything else that it holds.
        assert peak > charge - (64 if sharded else 40) * 100_000


def test_a_row_block_stream_of_an_unsharded_array_reserves_each_stored_chunk() -> None:
    values = np.arange(48, dtype=np.uint16).reshape(12, 4)
    stored = zarr.array(values, chunks=(6, 4))
    # A block of one chunk's six rows, 48 bytes, decodes that chunk, which is
    # held as stored and decoded, 48 + 48 + 1,024 bytes, beside the objects
    # of Zarr's read of one chunk.
    needed = 48 + (48 + 48 + 1_024) + _read_objects(1, 1)
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

    values = (np.arange(1_200 * 400) % 7).astype(np.uint16).reshape(1_200, 400)

    def frequency(chunks: tuple[int, int], budget: int) -> Any:
        stored = zarr.array(values, chunks=chunks)
        matrix = ChunkedArray(stored, rows=np.arange(1_200), block_size=1_200)
        with execution_report_scope() as reports:
            counted, _ = stream_document_frequency(
                matrix, memory_bytes=budget, nthreads=1, msg="", operation="Frequency"
            )
        np.testing.assert_array_equal(counted, np.count_nonzero(values, axis=0))
        (report,) = reports
        return report

    # Stored in chunks of 600 rows, a block's read holds a chunk of three
    # times the values it holds in chunks of 200 rows, 640 KB more as stored
    # and decoded, beyond the objects of the four more chunks that the
    # smaller ones take, so the budget of one block of every row in the
    # smaller chunks reads the larger ones in several blocks, each reserved
    # with its decode within the budget.
    budget = frequency((200, 400), 1 << 24).plan.reservedBytes
    assert frequency((200, 400), budget).unitsCompleted == 1
    larger = frequency((600, 400), budget)
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
    # size, 1,024 bytes beyond the 40 it decodes to, beside one decoded chunk
    # and the objects of Zarr's read of those four chunks.
    needed = 320 + 320 + 2 * 2 * (chunk + 1_024) + chunk + _read_objects(4, 1)
    planned = plan(needed)
    assert [block.bins for block in planned.blocks] == [(0, 1)]
    assert (planned.readWorkers, planned.ioConcurrency) == (1, 1)
    with pytest.raises(MemoryError, match="affordable width is 7"):
        plan(needed - 1)


def test_a_run_of_rows_over_scattered_columns_holds_the_chunks_between_them(
    tmp_path: Any,
) -> None:
    import tracemalloc

    from cytearc.storage.async_execution import zarr_io_concurrency
    from cytearc.storage.count_matrix import (
        CountMatrixPolicy,
        create_product_counts_array,
    )
    from cytearc.storage.geometry import array_geometry
    from cytearc.storage.partition import row_read_parts
    from cytearc.storage.virtual_arrays import read_selection

    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    counts = create_product_counts_array(
        root.require_group("RNA"),
        2_000,
        400,
        np.uint32,
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=1_600_000, chunkBytes=160_000),
    )
    assert (counts.chunks, counts.shards) == ((1_000, 40), (1_000, 400))
    # Random bits do not compress, so each chunk is stored a few bytes larger
    # than the 160,000 bytes it decodes to.
    counts[:] = np.random.default_rng(6).integers(
        0, 1 << 32, size=(2_000, 400), dtype=np.uint32
    )
    cells = np.arange(100, 433)
    # These genes fall in chunks 0, 1, 3, 5, 7 and 9 of the shard's ten, so
    # Zarr's merged fetch also takes chunks 2, 4, 6 and 8.
    genes = np.array([5, 397, 41, 42, 200, 120, 0, 300])
    geometry = array_geometry(counts)
    assert geometry is not None and geometry.coalesceGapBytes == 1 << 20
    fixed, per_row = row_read_parts(geometry, genes, rows_contiguous=True)
    apart, _ = row_read_parts(
        replace(geometry, coalesceGapBytes=0), genes, rows_contiguous=True
    )
    with zarr_io_concurrency(1):
        read_selection(counts, cells, genes)
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            block = read_selection(counts, cells, genes)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    held = peak - base - block.nbytes
    # The read held more than the six chunks it decodes from, which a charge
    # of those alone, the charge before, left out, and no more than a charge
    # of the four chunks between them too, as row selections and row-block
    # streams over those genes plan it.
    assert held > apart + cells.size * per_row
    assert held_within_charge(held, fixed + cells.size * per_row)
    assert held_within_charge(
        peak - base, ChunkedArray(counts, cols=genes)._read_bytes(block.nbytes)
    )
