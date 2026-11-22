"""Stored chunk geometry of a streamed array."""

import importlib
import math
import operator
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .types import array_metadata_shards

__all__ = [
    "ROW_INDEX_BYTES",
    "ROW_KEPT_INDEX_BYTES",
    "ArrayGeometry",
    "array_geometry",
]

#: A read of selected rows a band of chunks at a time first orders the rows
#: by band, which holds each row's position, chunk and place in their order
#: and a stable sort's buffer; while it reads the bands it keeps each row's
#: position, place in that order, index in its band and place in the result,
#: as int64.
ROW_INDEX_BYTES = 48
ROW_KEPT_INDEX_BYTES = 32

#: What one Zarr read holds beside its buffers, whatever their size: the
#: objects of the read of one shard, or of an unsharded read, those of each
#: chunk it takes, since Zarr makes a task for every chunk before it reads
#: any, and those of each chunk it decodes at once. Measured at about 45 KB,
#: 2.5 KB and 11 KB on Zarr 3.4 after a full collection has emptied CPython's
#: free lists, so that each object is allocated anew.
_READ_OBJECT_BYTES = 64 * 1024
_READ_CHUNK_OBJECT_BYTES = 4 * 1024
_READ_DECODE_OBJECT_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """Chunk and shard extents an array is physically stored in."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    itemsize: int
    readScratchBytes: int = 0
    #: Whether Zarr decodes the array's chunks on a pool of its own, whose
    #: threads keep the last chunk they decoded: Zarr's opt-in
    #: ``FusedCodecPipeline`` with more than one worker. Every other read
    #: decodes on CyteArc's codec pools, which keep none.
    poolKeepsDecodedChunk: bool = False
    #: The most bytes apart that Zarr merges two byte ranges of a read of part
    #: of a shard into one fetch, ``sharding_coalesce_max_gap_bytes`` of the
    #: array's configuration; none for a geometry of no opened array.
    coalesceGapBytes: int = 0

    def axisChunk(self, axis: int) -> int:
        """Return the chunk extent along one axis."""
        return max(1, self.chunks[axis])

    def axisShard(self, axis: int) -> int:
        """Return the shard extent along one axis, or its chunk extent."""
        if self.shards is None:
            return self.axisChunk(axis)
        return max(1, self.shards[axis])

    def nominalChunkBytes(self) -> int:
        """Return the bytes one whole chunk decodes to.

        Zarr stores and decodes a full chunk even where the chunk overhangs the
        array edge, so this deliberately ignores the clipped extent.
        """
        elements = 1
        for extent in self.chunks:
            elements *= max(1, extent)
        return elements * self.itemsize

    def storedChunkBytes(self) -> int:
        """Return the bytes of one whole chunk as stored and decoded.

        That is the decoded chunk beside the bound of its encoded bytes for
        the supported compressors.
        """
        from .layout import _encoded_chunk_bound

        decoded = self.nominalChunkBytes()
        return decoded + _encoded_chunk_bound(decoded)

    def rowBytes(self) -> int:
        """Return the bytes of one row: one value of every later axis."""
        return math.prod(self.shape[1:]) * self.itemsize

    def bandChunks(self) -> int:
        """Return the chunks of one band of rows, those that tile the later axes."""
        return math.prod(
            -(-extent // max(1, chunk))
            for extent, chunk in zip(self.shape[1:], self.chunks[1:], strict=True)
        )

    def objectBytes(self, chunks: int, decodes: int) -> int:
        """Return the objects that one read of ``chunks`` chunks, ``decodes`` at a time, holds beside its buffers.

        They are tens of kilobytes a read and a few kilobytes a chunk, which
        matter only beside chunks of tens of kilobytes.
        """
        touched = max(1, int(chunks))
        decoding = min(touched, max(1, int(decodes)))
        return (
            _READ_OBJECT_BYTES
            + touched * _READ_CHUNK_OBJECT_BYTES
            + decoding * _READ_DECODE_OBJECT_BYTES
        )

    def heldDecodeBytes(self, chunks: int, decodes: int) -> int:
        """Return what a read of ``chunks`` chunks, ``decodes`` at a time, holds beside the chunks it decodes at once.

        A thread of a plain thread pool keeps the last chunk it decoded until
        it runs again, which on a busy machine can be after the read has moved
        on, so a read on such a pool holds one previous decoded chunk for each
        decode under way, and no more than the decodes that follow the first
        ones. Only Zarr's opt-in ``FusedCodecPipeline``, with
        ``codec_pipeline.max_workers`` above 1, decodes on such a pool, as
        ``poolKeepsDecodedChunk`` records. CyteArc's codec pools hand each
        decoded chunk back to the read's event loop, so a read on them holds
        no previous chunk, only the objects of Zarr's read that
        :meth:`objectBytes` counts, which are charged up to those chunks: a
        read of chunks smaller than its objects is charged the chunks, as
        before.
        """
        touched = max(1, int(chunks))
        decoding = min(touched, max(1, int(decodes)))
        previous = min(decoding, touched - decoding) * self.nominalChunkBytes()
        if self.poolKeepsDecodedChunk:
            return previous
        return min(previous, self.objectBytes(touched, decoding))

    def wholeReadBytes(self, concurrency: int) -> int:
        """Return what reading the whole array at ``concurrency`` holds while it runs, with the result.

        An unsharded array is read in one call, which decodes up to
        ``concurrency`` chunks at once, each as stored and decoded, beside any
        previous chunks that :meth:`heldDecodeBytes` counts. A sharded array is read a
        band of inner chunks at a time, as :meth:`bandReadBytes` counts, so
        that no fetch takes more than one band of a shard.
        """
        result = self.shape[0] * self.rowBytes()
        if self.shards is not None:
            return result + self.bandReadBytes(self.axisChunk(0), concurrency)
        n_chunks = max(1, -(-self.shape[0] // self.axisChunk(0)) * self.bandChunks())
        decoding = min(max(1, int(concurrency)), n_chunks)
        return (
            result
            + decoding * self.storedChunkBytes()
            + self.heldDecodeBytes(n_chunks, decoding)
        )

    def bandReadBytes(self, rows: int, concurrency: int) -> int:
        """Return what reading ``rows`` rows of one band of chunks holds beside the result it fills.

        The read holds those rows as Zarr returns them and the band's chunks
        that it decodes at once, each as stored and decoded, beside any
        previous chunks that :meth:`heldDecodeBytes` counts; a sharded array also fills a
        shard-level copy of the rows and holds the stored bytes of every chunk
        of the band it fetched, which covers those.
        """
        rows_bytes = max(0, int(rows)) * self.rowBytes()
        touched = self.bandChunks()
        decoding = min(max(1, int(concurrency)), touched)
        if self.shards is None:
            band = decoding * self.storedChunkBytes() + self.heldDecodeBytes(
                touched, decoding
            )
        else:
            band = (
                rows_bytes
                + touched * self.storedChunkBytes()
                + decoding * self.nominalChunkBytes()
            )
        return rows_bytes + band + self.readScratchBytes

    def readBytes(
        self,
        resultBytes: int,
        chunks: int = 1,
        decodes: int | None = None,
        *,
        gapBytes: int = 0,
    ) -> int:
        """Return the bytes one Zarr read holds while it runs.

        A sharded read fills a shard-level copy of the selection before it
        copies that into the result, and holds the compressed bytes of every
        inner chunk it touches in a shard while it decodes them. Those are
        charged at the bound that :meth:`storedChunkBytes` adds to a decoded
        chunk, since data that does not compress is stored a few bytes larger
        than it decodes to. An unsharded read holds its result and each chunk
        it decodes at once as stored and decoded, as :meth:`storedChunkBytes`
        counts, since Zarr keeps a chunk's compressed bytes beside it while it
        decodes it. Either read that decodes more chunks than it decodes at
        once also holds what :meth:`heldDecodeBytes` counts. ``chunks``
        counts the inner chunks the read touches in one shard and ``decodes``
        the chunks Zarr decodes at once, all of them unless given.
        Storage-provided scratch covers one shard request, including its
        sequential inner-chunk reads. Independent shard requests each need
        this allowance. Unsharded reads charge scratch per decode.
        ``gapBytes`` are the stored bytes that a sharded read of part of a
        shard also fetches between the chunks it touches, as
        :func:`~cytearc.storage.partition.coalesced_gap_bytes` counts them.
        """
        touched = max(1, int(chunks))
        decoding = touched if decodes is None else min(touched, max(1, int(decodes)))
        result = max(0, int(resultBytes))
        held = self.heldDecodeBytes(touched, decoding)
        if self.shards is None:
            return (
                result
                + decoding * (self.storedChunkBytes() + self.readScratchBytes)
                + held
            )
        decoded = self.nominalChunkBytes()
        return (
            2 * result
            + touched * (self.storedChunkBytes() - decoded)
            + decoding * decoded
            + held
            + self.readScratchBytes
            + max(0, int(gapBytes))
        )

    def binOf(self, axis: int, indices: np.ndarray) -> np.ndarray:
        """Map indices along one axis to the chunk they live in."""
        return np.asarray(indices, dtype=np.int64) // self.axisChunk(axis)


def array_geometry(array: Any) -> ArrayGeometry | None:
    """Read stored geometry, or ``None`` when the array has no chunk layout.

    In-memory arrays have no chunks, so callers get one explicit place to decide
    what an unchunked array should mean for them.
    """
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    resolved_chunks = tuple(operator.index(value) for value in chunks)
    resolved_shape = tuple(operator.index(value) for value in array.shape)
    if not resolved_chunks:
        return None
    if len(resolved_chunks) != len(resolved_shape):
        raise ValueError(
            f"Array chunks {resolved_chunks} do not match shape {resolved_shape}"
        )
    if any(value < 1 for value in resolved_chunks):
        raise ValueError(f"Array chunks must be positive, got {resolved_chunks}")
    if any(value < 0 for value in resolved_shape):
        raise ValueError(f"Array dimensions cannot be negative, got {resolved_shape}")

    raw_shards = array_metadata_shards(array)
    shards = None
    if raw_shards is not None:
        shards = tuple(operator.index(value) for value in raw_shards)
        if len(shards) != len(resolved_shape):
            raise ValueError(
                f"Array shards {shards} do not match shape {resolved_shape}"
            )

    from .virtual_arrays import chunk_read_scratch_bytes

    scratch = chunk_read_scratch_bytes(array)
    if scratch < 0:
        raise ValueError(f"Array read scratch cannot be negative, got {scratch}")

    return ArrayGeometry(
        shape=resolved_shape,
        chunks=resolved_chunks,
        shards=shards,
        itemsize=max(1, int(np.dtype(array.dtype).itemsize)),
        readScratchBytes=scratch,
        poolKeepsDecodedChunk=_decodes_on_zarrs_own_pool(array),
        coalesceGapBytes=_coalesce_gap_bytes(array),
    )


def _coalesce_gap_bytes(array: Any) -> int:
    """Return the most bytes apart that Zarr merges two byte ranges of a partial shard read of ``array``.

    A Zarr array keeps the setting of the configuration it was opened with;
    an array of another library, which Zarr does not read, merges none.
    """
    config = getattr(getattr(array, "async_array", array), "config", None)
    return max(0, int(getattr(config, "sharding_coalesce_max_gap_bytes", 0)))


def _decodes_on_zarrs_own_pool(array: Any) -> bool:
    """Return whether Zarr decodes ``array``'s chunks on a pool of its own.

    An array opened while ``codec_pipeline.path`` names Zarr's opt-in
    ``FusedCodecPipeline`` keeps that pipeline, which decodes on a plain
    thread pool of ``codec_pipeline.max_workers`` threads, by default one for
    each CPU, when that is more than one, and inline otherwise.
    """
    pipeline = getattr(getattr(array, "async_array", array), "codec_pipeline", None)
    fused = getattr(
        importlib.import_module("zarr.core.codec_pipeline"),
        "FusedCodecPipeline",
        None,
    )
    if fused is None or not isinstance(pipeline, fused):
        return False
    configured = zarr.config.get("codec_pipeline.max_workers", None)
    try:
        workers = (os.cpu_count() or 1) if configured is None else int(configured)
    except (TypeError, ValueError):
        # Zarr falls back to one worker for each CPU from a value it cannot read.
        workers = os.cpu_count() or 1
    return workers > 1
