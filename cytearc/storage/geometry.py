"""Stored chunk geometry of a streamed array."""

import functools
import importlib
import io
import math
import operator
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ..utils.memory_budget import MemoryLedger, object_array_bytes, python_text_bytes
from .types import array_metadata_shards
from .validation_scope import store_key, validated_once

__all__ = [
    "ROW_BLOCK_BYTES",
    "ROW_INDEX_BYTES",
    "ROW_KEPT_INDEX_BYTES",
    "ArrayGeometry",
    "TextExtent",
    "array_geometry",
    "variable_text_extent",
]

#: A read of selected rows a band of chunks at a time first orders the rows
#: by band, which holds each row's position, chunk and place in their order
#: and a stable sort's buffer; while it reads the bands it keeps each row's
#: position, place in that order, index in its band and place in the result,
#: as int64.
ROW_INDEX_BYTES = 48
ROW_KEPT_INDEX_BYTES = 32
#: Ordering the rows also makes a block of each band's rows, which it keeps
#: while it reads: an object and two arrays of the rows' indices, 400 to 420
#: bytes beside the indices, and 70 to 90 bytes more of each band while the
#: sort groups them.
ROW_BLOCK_BYTES = 512

#: What one Zarr read holds beside its buffers, whatever their size: the
#: objects of the read of one shard, or of an unsharded read, those of each
#: chunk it takes, since Zarr makes a task for every chunk before it reads
#: any, and those of each chunk it decodes at once. A read of one chunk
#: traced 31 to 45 KB, each further chunk 2.2 to 2.4 KB and each further
#: decode at once 12.5 KB, on Zarr 3.4 after a full collection had emptied
#: CPython's free lists, so that each object was allocated anew; reads of
#: tiny chunks are held between half of these charges and the whole of them.
#: Once a read returns, the objects of the read and of its chunks stay held
#: until its caller lets go of them, CPython reuses them from its free lists
#: or Python's next collection frees the reference cycles of Zarr's event
#: loop among them: a read of one chunk left up to 23 KB, and each further
#: chunk up to 2.4 KB more, traced after reads of 1 to 100 chunks, in shards
#: and not, on Python 3.13 to 3.15.
_READ_OBJECT_BYTES = 32 * 1024
_READ_CHUNK_OBJECT_BYTES = 4 * 1024
_READ_DECODE_OBJECT_BYTES = 16 * 1024
#: What the reads of a band each leave of their objects levels off as CPython
#: reuses the objects it freed for the next reads: 140 to 150 KB after 100 to
#: 2,000 reads of a band, on Python 3.13 and 3.14.
_READ_LEFTOVER_LIMIT_BYTES = 192 * 1024

#: Zarr's orthogonal indexer orders the positions of an index array by chunk
#: in arrays of int64. While it builds them it holds each position's chunk,
#: the order of the chunks, the positions in that order, the chunks in that
#: order and their steps, 40 bytes a position; it keeps the order and the
#: ordered positions, and makes each chunk's positions within the chunk for
#: the read, 24 bytes a position while the read runs. A sharded read builds
#: such an indexer again in each shard, over the shard's positions and its
#: inner chunks, and makes their positions within the chunks twice, so it
#: holds 64 bytes a position while it builds that indexer beside the
#: array's. Zarr 3.4.2 traced 40 and 64 bytes a position for shuffled
#: positions, and less for sorted ones.
_INDEX_BYTES = 40
_SHARD_INDEX_BYTES = 64
#: NumPy indexes a chunk by two index arrays at once, as Zarr indexes an
#: unsharded chunk whose rows and columns are both index arrays, through a
#: buffer of up to 8,192 elements of each array, 8 bytes each. Zarr takes
#: the values from a contiguous chunk and places a contiguous copy, which
#: NumPy does not buffer: taking and placing values of 2, 4 and 8 bytes each
#: held 133 to 135 KB on NumPy 2.1 and 2.5, the rest within a read's objects.
_INDEX_BUFFER_ELEMENTS = 8192

#: Python's ``open`` buffers a binary file at its file system's block size,
#: or at ``io.DEFAULT_BUFFER_SIZE`` where the block size is unknown, as on
#: Windows, and ``Path.read_bytes`` reads a whole file through it. From
#: Python 3.14 ``open`` buffers at least ``io.DEFAULT_BUFFER_SIZE``, 128 KiB,
#: and at most 8 MiB, and ``Path.read_bytes`` reads unbuffered.
_FILE_BUFFER_LIMIT_BYTES = 8 * 1024 * 1024
_WHOLE_FILES_UNBUFFERED = sys.version_info >= (3, 14)

#: NumPy keeps each value of variable-width text in a 16-byte element, and a
#: value of more than 15 bytes in UTF-8 in an arena beside the array, behind
#: its size, a byte for a value of up to 255 bytes and 8 for a longer one, in
#: an arena that NumPy grows by a quarter at a time (NEP 55). Zarr's
#: ``vlen-utf8`` codec stores each value's UTF-8 bytes behind a 4-byte length,
#: and each chunk behind a 4-byte count.
_TEXT_ELEMENT_BYTES = 16
_TEXT_INLINE_BYTES = 15
_TEXT_SHORT_SIZE_LIMIT = 255
_TEXT_SHORT_SIZE_BYTES = 1
_TEXT_SIZE_BYTES = 8
_TEXT_LENGTH_BYTES = 4
#: The most bytes of fixed-width text, and the most values, that a measure of
#: variable-width text casts its values to at once, and the most characters
#: of a value that it casts: NumPy's cast holds a buffer of up to its buffer
#: size of values at the width cast to, whatever the values, so a longer
#: value is measured alone, as a Python string.
_TEXT_MEASURE_BYTES = 1024 * 1024
_TEXT_MEASURE_VALUES = 65_536
_TEXT_CAST_CHARACTERS = 64
#: What a measure keeps for each value of the chunk it measures beside the
#: chunk: the value's characters, sorted, its place in that order and its
#: UTF-8 bytes, 8 bytes each.
_TEXT_MEASURE_VALUE_BYTES = 32


def _decode_bytes(
    values: int, utf8: int, characters: int, widest: int, arena: int
) -> int:
    """What decoding a chunk of ``values`` values of variable-width text holds.

    Zarr's codec holds the chunk's stored bytes and makes a Python string of
    each value, referenced from an object array, of which NumPy makes its
    values; NumPy asks each string for its UTF-8 bytes, which a string beyond
    ASCII then keeps beside its characters. ``utf8``, ``characters`` and
    ``arena`` are the values' UTF-8 bytes, characters and arena bytes in all,
    and ``widest`` the widest code point of their characters.
    """
    stored = _TEXT_LENGTH_BYTES * (values + 1) + utf8
    copies = utf8 + values if widest > 0x7F else 0
    kept = _TEXT_ELEMENT_BYTES * values + -(-5 * arena // 4)
    return (
        stored
        + object_array_bytes(values)
        + python_text_bytes(values, characters, widest=widest)
        + copies
        + kept
    )


def _arena_bytes(utf8: np.ndarray) -> int:
    """The bytes of NumPy's arena that values of these UTF-8 lengths take before it grows.

    A value of up to 15 bytes takes none; a longer value takes its bytes
    behind its size.
    """
    sized = np.where(
        utf8 <= _TEXT_SHORT_SIZE_LIMIT, _TEXT_SHORT_SIZE_BYTES, _TEXT_SIZE_BYTES
    )
    return int(np.where(utf8 > _TEXT_INLINE_BYTES, utf8 + sized, 0).sum())


@dataclass(frozen=True, slots=True)
class TextExtent:
    """The extent of the values of an array of variable-width text.

    ``characters`` and ``utf8Bytes`` are the most characters and the most
    UTF-8 bytes of any value, and ``widest`` the widest code point of any
    value's characters, 0 for none. ``values``, ``characterTotal`` and
    ``arenaBytes`` count every value, its characters, and the bytes that
    NumPy's arena takes for them before it grows. ``chunkDecodeBytes`` and
    ``chunkStoredBytes`` are the most that decoding one stored chunk holds and
    the most bytes of one chunk as Zarr stores it before it compresses it,
    each chunk counted by its own values. ``measureBytes`` is the most that
    the measure held for one chunk.
    """

    characters: int
    utf8Bytes: int
    widest: int
    values: int
    characterTotal: int
    arenaBytes: int
    chunkDecodeBytes: int
    chunkStoredBytes: int
    measureBytes: int

    def valueBytes(self) -> int:
        """Return the most bytes that NumPy holds for one value, its element and its arena share."""
        if self.utf8Bytes <= _TEXT_INLINE_BYTES:
            return _TEXT_ELEMENT_BYTES
        size = (
            _TEXT_SHORT_SIZE_BYTES
            if self.utf8Bytes <= _TEXT_SHORT_SIZE_LIMIT
            else _TEXT_SIZE_BYTES
        )
        return _TEXT_ELEMENT_BYTES + -(-5 * (self.utf8Bytes + size) // 4)

    def keptBytes(self) -> int:
        """Return the most bytes that NumPy holds for every value: their elements and their arena."""
        return _TEXT_ELEMENT_BYTES * self.values + -(-5 * self.arenaBytes // 4)


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """Chunk and shard extents an array is physically stored in."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    itemsize: int
    readScratchBytes: int = 0
    #: The threads of Zarr's own pool that its opt-in ``FusedCodecPipeline``
    #: reads the array on, ``codec_pipeline.max_workers`` or more, as many
    #: chunks or shards at once as it has threads, whatever Zarr's async
    #: concurrency, the store's threads or ``zarr_decoding_threads``; 0 for
    #: every other pipeline.
    fusedPoolWorkers: int = 0
    #: Whether that pipeline reads the array's store asynchronously, as it
    #: does a store without Zarr's synchronous I/O: it then decodes every
    #: chunk of an unsharded read before it places any.
    fusedReadsAsync: bool = False
    #: The most bytes apart that Zarr merges two byte ranges of a read of part
    #: of a shard into one fetch, ``sharding_coalesce_max_gap_bytes`` of the
    #: array's configuration; none for a geometry of no opened array.
    coalesceGapBytes: int = 0
    #: The extent of variable-width text, whose ``itemsize`` is what NumPy
    #: keeps for its longest value: its chunks hold what decoding and storing
    #: its chunk of most bytes held, a read of its rows no more than every
    #: value holds, and a read of scattered rows a chunk's span of them; None
    #: for values of fixed size.
    text: TextExtent | None = None

    @property
    def poolKeepsDecodedChunk(self) -> bool:
        """Whether Zarr decodes the array's chunks on a pool of its own, whose threads keep the last chunk they decoded.

        That is Zarr's opt-in ``FusedCodecPipeline`` with more than one
        worker, as ``fusedPoolWorkers`` counts them. Every other read decodes
        on CyteArc's codec pools, which keep none.
        """
        return self.fusedPoolWorkers > 1

    def axisChunk(self, axis: int) -> int:
        """Return the chunk extent along one axis."""
        return max(1, self.chunks[axis])

    def axisShard(self, axis: int) -> int:
        """Return the shard extent along one axis, or its chunk extent."""
        if self.shards is None:
            return self.axisChunk(axis)
        return max(1, self.shards[axis])

    def nominalChunkBytes(self) -> int:
        """Return the bytes one whole chunk holds while Zarr decodes it, its values decoded.

        Zarr stores and decodes a full chunk even where the chunk overhangs the
        array edge, so this deliberately ignores the clipped extent. A chunk
        of variable-width text holds what decoding the chunk of most bytes
        held, as ``text`` counts it.
        """
        if self.text is not None:
            return self.text.chunkDecodeBytes
        return math.prod(max(1, extent) for extent in self.chunks) * self.itemsize

    def storedChunkBytes(self) -> int:
        """Return the bytes of one whole chunk as stored and decoded.

        That is the decoded chunk beside the bound of its encoded bytes for
        the supported compressors.
        """
        from .layout import _encoded_chunk_bound

        encoded = (
            self.nominalChunkBytes()
            if self.text is None
            else self.text.chunkStoredBytes
        )
        return self.nominalChunkBytes() + _encoded_chunk_bound(encoded)

    def rowBytes(self) -> int:
        """Return the bytes of one row: one value of every later axis."""
        return math.prod(self.shape[1:]) * self.itemsize

    def leftoverBytes(self, rows: int) -> int:
        """Return what a read of ``rows`` rows leaves of its objects once it returns.

        It reads no more bands than it has rows, and they leave what
        :meth:`bandsLeftoverBytes` counts.
        """
        return self.bandsLeftoverBytes(min(max(0, int(rows)), self.bandCount()))

    def bandsLeftoverBytes(self, bands: int) -> int:
        """Return what the reads of ``bands`` bands of chunks leave of their objects once they return.

        The reads take every chunk of the bands, as :meth:`chunksLeftoverBytes`
        counts them.
        """
        return self.chunksLeftoverBytes(max(0, int(bands)) * self.bandChunks())

    def chunksLeftoverBytes(self, chunks: int) -> int:
        """Return what reads that took ``chunks`` chunks in all leave of their objects once they return.

        They are counted as :meth:`objectBytes` counts the objects of a read
        and of each chunk that it takes while it runs, for every chunk,
        whether one read takes them or a read of each band does, up to
        192 KiB, where what the reads leave levels off as CPython reuses the
        objects it freed.
        """
        chunks = max(0, int(chunks))
        if chunks == 0:
            return 0
        return min(
            _READ_LEFTOVER_LIMIT_BYTES,
            _READ_OBJECT_BYTES + chunks * _READ_CHUNK_OBJECT_BYTES,
        )

    def bandCount(self) -> int:
        """Return the bands of chunks along the first axis."""
        return -(-self.shape[0] // self.axisChunk(0))

    def rowsBytes(self, rows: int) -> int:
        """Return the bytes of ``rows`` rows as a read returns them.

        Variable-width text holds no more than every value holds, however
        long its longest value.
        """
        held = max(0, int(rows)) * self.rowBytes()
        return held if self.text is None else min(held, self.text.keptBytes())

    def bandChunks(self) -> int:
        """Return the chunks of one band of rows, those that tile the later axes."""
        return math.prod(
            -(-extent // max(1, chunk))
            for extent, chunk in zip(self.shape[1:], self.chunks[1:], strict=True)
        )

    def readsAtOnce(
        self, chunks: int, decodes: int | None, shards: int = 1
    ) -> tuple[int, int]:
        """Return the chunks that a read decodes at once in each shard it reads, and the shards it reads at once.

        ``chunks`` counts the chunks that the read takes in one shard, or in
        the array when it is unsharded, ``shards`` the shards it touches,
        and ``decodes`` Zarr's async concurrency, which bounds both, or None
        for all of them. CyteArc's codec pipelines follow that concurrency.
        Zarr's ``FusedCodecPipeline``, which a user who configures it gets in
        place of CyteArc's, follows the threads of its own pool instead, as
        ``fusedPoolWorkers`` counts them: on a store that reads synchronously
        it reads that many chunks, or shards, at once, each shard decoding
        its chunks one at a time; on a store that reads asynchronously it
        decodes every chunk of an unsharded read before it places any, and
        reads the shards of a sharded one at Zarr's async concurrency, each
        decoding as many chunks at once as the pool has threads.
        """
        touched = max(1, int(chunks))
        touched_shards = max(1, int(shards))
        planned = None if decodes is None else max(1, int(decodes))
        workers = self.fusedPoolWorkers
        if not workers:
            if planned is None:
                return touched, touched_shards
            return min(touched, planned), min(touched_shards, planned)
        if self.shards is None:
            return (touched if self.fusedReadsAsync else min(touched, workers)), 1
        if not self.fusedReadsAsync:
            return 1, min(touched_shards, workers)
        reading = touched_shards if planned is None else min(touched_shards, planned)
        return min(touched, workers), reading

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

        Every read holds the objects of Zarr's read that :meth:`objectBytes`
        counts. A thread of a plain thread pool also keeps the last chunk it
        decoded until it runs again, which on a busy machine can be after the
        read has moved on, so a read on such a pool also holds one previous
        decoded chunk for each decode under way, and no more than the decodes
        that follow the first ones. Only Zarr's opt-in ``FusedCodecPipeline``,
        with ``codec_pipeline.max_workers`` above 1, decodes on such a pool,
        as ``poolKeepsDecodedChunk`` records; CyteArc's codec pools hand each
        decoded chunk back to the read's event loop and keep none.
        """
        touched = max(1, int(chunks))
        decoding = min(touched, max(1, int(decodes)))
        objects = self.objectBytes(touched, decoding)
        if not self.poolKeepsDecodedChunk:
            return objects
        return objects + min(decoding, touched - decoding) * self.nominalChunkBytes()

    def wholeReadBytes(self, concurrency: int) -> int:
        """Return what reading the whole array at ``concurrency`` holds while it runs, with the result.

        An unsharded array is read in one call, which decodes up to
        ``concurrency`` chunks at once, each as stored and decoded, beside
        what :meth:`heldDecodeBytes` counts. A sharded array is read a band of
        inner chunks at a time, as :meth:`bandReadBytes` counts, so that no
        fetch takes more than one band of a shard.
        """
        result = self.rowsBytes(self.shape[0])
        if self.shards is not None:
            # The reads of the earlier bands leave their objects while the
            # last one runs.
            return (
                result
                + self.bandReadBytes(self.axisChunk(0), concurrency)
                + self.bandsLeftoverBytes(self.bandCount() - 1)
            )
        n_chunks = max(1, -(-self.shape[0] // self.axisChunk(0)) * self.bandChunks())
        decoding, _ = self.readsAtOnce(n_chunks, concurrency)
        return (
            result
            + decoding * (self.storedChunkBytes() + self.readScratchBytes)
            + self.heldDecodeBytes(n_chunks, decoding)
        )

    def bandReadBytes(self, rows: int, concurrency: int) -> int:
        """Return what reading ``rows`` rows of one band of chunks holds beside the result it fills.

        The read holds those rows as Zarr returns them and the band's chunks
        that it decodes at once, each as stored and decoded, beside what
        :meth:`heldDecodeBytes` counts; a sharded array also fills a
        shard-level copy of the rows and holds the stored bytes of every chunk
        of the band it fetched, which cover any previous decoded chunks,
        beside the objects of Zarr's read.
        """
        rows_bytes = self.rowsBytes(rows)
        touched = self.bandChunks()
        if self.shards is None:
            # Each chunk decoded at once is a read of its own scratch.
            decoding, _ = self.readsAtOnce(touched, concurrency)
            band = decoding * (
                self.storedChunkBytes() + self.readScratchBytes
            ) + self.heldDecodeBytes(touched, decoding)
            return rows_bytes + band
        decoding = min(max(1, int(concurrency)), touched)
        band = (
            rows_bytes
            + touched * self.storedChunkBytes()
            + decoding * self.nominalChunkBytes()
            + self.objectBytes(touched, decoding)
        )
        return rows_bytes + band + self.readScratchBytes

    def indexBytes(self, positions: int) -> int:
        """Return what Zarr's indexers hold for ``positions`` positions that a read takes by index arrays.

        Zarr orders the positions of an index array by chunk, in integer
        arrays of 40 bytes a position, and a sharded read orders each shard's
        positions again by inner chunk, 64 bytes a position in all.
        """
        per_position = _SHARD_INDEX_BYTES if self.shards is not None else _INDEX_BYTES
        return max(0, int(positions)) * per_position

    def takenChunkBytes(self, rows: int, columns: int, *, indexedAxes: int) -> int:
        """Return what copying ``rows`` by ``columns`` values that a read takes from one chunk holds.

        Zarr copies the values that a read takes from a decoded chunk by
        index arrays on ``indexedAxes`` of its axes before it places them,
        while a read of slices places a view of the chunk. Every row counts,
        even beyond a chunk's rows, so that the charge of a block shrunk to
        fewer rows is no less than its share of the charge of more. With
        index arrays on two axes, NumPy also buffers up to 8,192 elements of
        each array, charged whole for the same reason.
        """
        if indexedAxes <= 0:
            return 0
        held = max(0, int(rows)) * max(0, int(columns)) * self.itemsize
        if indexedAxes > 1:
            held += _INDEX_BUFFER_ELEMENTS * 2 * np.dtype(np.intp).itemsize
        return held

    def readBytes(
        self,
        resultBytes: int,
        chunks: int = 1,
        decodes: int | None = None,
        *,
        shards: int = 1,
        gapBytes: int = 0,
        indexes: int = 0,
        takenBytes: int = 0,
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
        decodes it. Either read also holds what :meth:`heldDecodeBytes`
        counts: the objects of Zarr's read, and the previous decoded chunks
        that a plain pool keeps. ``chunks``
        counts the inner chunks the read touches in one shard and ``decodes``
        the chunks Zarr decodes at once, all of them unless given.
        Storage-provided scratch covers one shard request, including its
        sequential inner-chunk reads. Independent shard requests each need
        this allowance. Unsharded reads charge scratch per decode.
        ``gapBytes`` are the stored bytes that a sharded read of part of a
        shard also fetches between the chunks it touches, as
        :func:`~cytearc.storage.partition.coalesced_gap_bytes` counts them.
        ``shards`` counts the shards that a sharded read touches, of which
        Zarr reads as many at once as :meth:`readsAtOnce` gives: each holds
        the compressed bytes of the chunks it touches, its decodes, what
        :meth:`heldDecodeBytes` counts, scratch and gap, while their
        shard-level copies together hold the selection once. Zarr makes a
        task for every shard before it reads any, as it does for every chunk
        of an unsharded read, so the shards it does not read at once hold
        the objects of a chunk each meanwhile. The chunks that
        a read decodes at once, in the array or in each shard, also follow
        :meth:`readsAtOnce`. A read that takes ``indexes`` positions by index
        arrays rather than slices, on any of its axes, also holds Zarr's
        indexers of them, as :meth:`indexBytes` counts, and ``takenBytes``,
        what copying the values that it takes from one chunk holds, as
        :meth:`takenChunkBytes` counts. CyteArc's codec pipelines hand each
        decoded chunk back to the read's event loop, which copies and places
        its values before it takes the next, so a read holds one such copy
        however many chunks it decodes at once; Zarr's fused pipeline copies
        them on its pool's threads, one for each chunk it decodes at once.
        """
        touched = max(1, int(chunks))
        touched_shards = max(1, int(shards))
        decoding, reading = self.readsAtOnce(touched, decodes, touched_shards)
        result = max(0, int(resultBytes))
        held = self.heldDecodeBytes(touched, decoding)
        copies = reading * decoding if self.fusedPoolWorkers else 1
        indexed = self.indexBytes(indexes) + copies * max(0, int(takenBytes))
        if self.shards is None:
            return (
                result
                + decoding * (self.storedChunkBytes() + self.readScratchBytes)
                + held
                + indexed
            )
        decoded = self.nominalChunkBytes()
        shard = (
            touched * (self.storedChunkBytes() - decoded)
            + decoding * decoded
            + held
            + self.readScratchBytes
            + max(0, int(gapBytes))
        )
        waiting = (touched_shards - reading) * _READ_CHUNK_OBJECT_BYTES
        return 2 * result + reading * shard + waiting + indexed

    def binOf(self, axis: int, indices: np.ndarray) -> np.ndarray:
        """Map indices along one axis to the chunk they live in."""
        return np.asarray(indices, dtype=np.int64) // self.axisChunk(axis)


def array_geometry(
    array: Any, *, measure_text: bool = False, ledger: MemoryLedger | None = None
) -> ArrayGeometry | None:
    """Read stored geometry, or ``None`` when the array has no chunk layout.

    In-memory arrays have no chunks, so callers get one explicit place to decide
    what an unchunked array should mean for them. ``measure_text`` gives
    variable-width text the extent of its values, which a charge of a read
    of them needs, as :func:`variable_text_extent` measures it, charged to
    ``ledger``; without it such text counts the 16-byte elements that NumPy
    keeps for its values, as callers that use only the chunks do, since the
    measure reads every stored chunk.
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
    # A store that computes the array's chunks reads no file. A shard of
    # several chunks can be read in part, as byte ranges of its file; every
    # other read takes whole files, chunks or shards of one chunk.
    if not scratch:
        ranged = shards is not None and any(
            shard > chunk for shard, chunk in zip(shards, resolved_chunks, strict=True)
        )
        scratch = _file_buffer_bytes(array, ranged=ranged)
    fused_workers, fused_async = _fused_pool(array)

    text = (
        variable_text_extent(array, ledger)
        if measure_text and np.dtype(array.dtype).kind == "T"
        else None
    )
    return ArrayGeometry(
        shape=resolved_shape,
        chunks=resolved_chunks,
        shards=shards,
        itemsize=(
            max(1, int(np.dtype(array.dtype).itemsize))
            if text is None
            else text.valueBytes()
        ),
        readScratchBytes=scratch,
        coalesceGapBytes=_coalesce_gap_bytes(array),
        fusedPoolWorkers=fused_workers,
        fusedReadsAsync=fused_async,
        text=text,
    )


def variable_text_extent(array: Any, ledger: MemoryLedger | None = None) -> TextExtent:
    """Measure the values of an array of variable-width text, a stored chunk at a time.

    Zarr's ``dtype=str`` stores text this way, which reads as NumPy's
    ``StringDType``. A Zarr array is measured once in an operation, while a
    validation scope is open, as each public call of a DataStore opens one,
    and its stored bytes cannot change, so later reads of it in the operation
    reuse the measure; an array in memory is measured each time. The measure
    charges ``ledger`` as it reads, as :func:`_measured_text_extent` states.
    """
    if isinstance(array, zarr.Array):
        return validated_once(
            ("variable_text_extent", *store_key(array)),
            lambda: _measured_text_extent(array, ledger),
        )
    return _measured_text_extent(array, ledger)


def text_extent(
    array: Any,
    geometry: ArrayGeometry | None = None,
    ledger: MemoryLedger | None = None,
) -> TextExtent:
    """Return the extent of variable-width text ``array``: that of ``geometry`` where a read's charge measured it, or else :func:`variable_text_extent`'s."""
    if geometry is not None and geometry.text is not None:
        return geometry.text
    return variable_text_extent(array, ledger)


def _measured_text_extent(array: Any, ledger: MemoryLedger | None) -> TextExtent:
    """The extent of ``array``'s values, read a stored chunk at a time and cast to fixed-width text in parts.

    Each chunk's values are cast to fixed-width text in the order of their
    characters, as :func:`_utf8_lengths` casts them, which gives each value's
    UTF-8 bytes and the widest character. A stored chunk counts what decoding
    it holds, and its rows beyond the array's edge as empty values, as Zarr
    decodes a whole chunk, with the bound of its compressed bytes and the
    objects of Zarr's read. Before it reads a stored chunk, the measure
    charges ``ledger`` what decoding the chunk holds at the least, its values
    as empty strings, beside what the measure keeps for each value; once the
    chunk is measured, it charges what reading the chunk held, from the
    chunk's own values, and what the measure held, before it reads the next
    chunk. An array in memory charges only what the measure holds.
    """
    from .layout import _encoded_chunk_bound

    n_rows = int(array.shape[0])
    chunks = getattr(array, "chunks", None)
    rows = max(1, int(chunks[0])) if chunks else max(1, n_rows)
    chunk_values = rows * math.prod(int(extent) for extent in array.shape[1:])
    name = getattr(array, "path", None)
    item = (
        f"a measure of the text of {name!r}"
        if name
        else "a measure of variable-width text"
    )
    keeping = _TEXT_MEASURE_VALUE_BYTES * chunk_values
    objects = _READ_OBJECT_BYTES + _READ_CHUNK_OBJECT_BYTES + _READ_DECODE_OBJECT_BYTES
    least = keeping + (
        _decode_bytes(chunk_values, 0, 0, 0, 0) + objects if chunks else 0
    )
    characters = utf8_most = widest = values = character_total = arena = 0
    chunk_decode = chunk_stored = measure = 0
    for start in range(0, n_rows, rows):
        if ledger is not None:
            ledger.peak(item, least)
        block = np.asarray(array[start : start + rows]).reshape(-1)
        lengths = np.strings.str_len(block).astype(np.int64, copy=False)
        utf8, chunk_widest, part_bytes = _utf8_lengths(block, lengths)
        utf8_total = int(utf8.sum())
        chunk_characters = int(lengths.sum())
        chunk_arena = _arena_bytes(utf8)
        decode = _decode_bytes(
            chunk_values, utf8_total, chunk_characters, chunk_widest, chunk_arena
        )
        stored = _TEXT_LENGTH_BYTES * (chunk_values + 1) + utf8_total
        reading = decode + _encoded_chunk_bound(stored) + objects
        held = (reading if chunks else 0) + keeping + part_bytes
        if ledger is not None:
            ledger.peak(item, held)
        characters = max(characters, int(lengths.max(initial=0)))
        utf8_most = max(utf8_most, int(utf8.max(initial=0)))
        widest = max(widest, chunk_widest)
        values += int(block.size)
        character_total += chunk_characters
        arena += chunk_arena
        chunk_decode = max(chunk_decode, decode)
        chunk_stored = max(chunk_stored, stored)
        measure = max(measure, held)
        del block, lengths, utf8
    return TextExtent(
        characters=characters,
        utf8Bytes=utf8_most,
        widest=widest,
        values=values,
        characterTotal=character_total,
        arenaBytes=arena,
        chunkDecodeBytes=chunk_decode,
        chunkStoredBytes=chunk_stored,
        measureBytes=measure,
    )


def _utf8_lengths(
    block: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Return each value's UTF-8 bytes, the widest code point of the values, and the most bytes that measuring a part held.

    ``lengths`` are the values' characters. Values of up to
    ``_TEXT_CAST_CHARACTERS`` characters are cast to fixed-width text in the
    order of their characters, in parts of at most ``_TEXT_MEASURE_VALUES``
    values and ``_TEXT_MEASURE_BYTES`` of code points, four bytes a character
    of the part's longest value, beside the buffer of NumPy's cast; the code
    points give a value's UTF-8 bytes, a byte for each character and one more
    for each from U+0080, from U+0800 and from U+10000. A longer value is
    encoded alone, as a Python string. Values without characters need
    neither.
    """
    utf8 = lengths.copy()
    order = np.argsort(lengths, kind="stable")
    ordered = lengths[order]
    widest = part_bytes = 0
    start = int(np.searchsorted(ordered, 1))
    cast_stop = int(np.searchsorted(ordered, _TEXT_CAST_CHARACTERS, side="right"))
    while start < cast_stop:
        window = ordered[start : min(cast_stop, start + _TEXT_MEASURE_VALUES)]
        # The values whose cast, as wide as the longest of them, fits.
        fits = np.arange(1, window.size + 1) * window
        count = int(np.searchsorted(fits, _TEXT_MEASURE_BYTES // 4, side="right"))
        stop = start + max(1, count)
        longest = int(ordered[stop - 1])
        positions = order[start:stop]
        # np.take reads variable-width text on every supported NumPy release,
        # as read_variable_text_rows notes.
        part = np.take(block, positions)
        points = part.astype(f"U{longest}").view(np.uint32).reshape(-1, longest)
        most = int(points.max())
        widest = max(widest, most)
        if most > 0x7F:
            utf8[positions] += sum(
                np.count_nonzero(points >= limit, axis=1)
                for limit in (0x80, 0x800, 0x10000)
            )
        # The part as NumPy keeps it, its code points and the buffer of the
        # cast that made them, a comparison of them and the counts it gives,
        # and the fits of the part's values.
        kept = 16 * part.size + 5 * (int(utf8[positions].sum()) + 8 * part.size) // 4
        cast = points.nbytes * 5 // 4 + np.getbufsize() * 4 * longest
        part_bytes = max(part_bytes, kept + cast + 24 * window.size)
        del part, points, fits
        start = stop
    # Longer values become Python strings in parts of at most
    # ``_TEXT_MEASURE_BYTES`` of characters, four bytes each, and each is
    # encoded alone.
    longer = order[cast_stop:]
    reach = np.cumsum(ordered[cast_stop:]) * 4
    first = 0
    while first < longer.size:
        base = int(reach[first - 1]) if first else 0
        stop = int(np.searchsorted(reach, base + _TEXT_MEASURE_BYTES, side="right"))
        stop = max(first + 1, min(first + _TEXT_MEASURE_VALUES, stop))
        positions = longer[first:stop]
        strings = np.take(block, positions).tolist()
        encoded = [value.encode("utf-8") for value in strings]
        utf8[positions] = [len(value) for value in encoded]
        # An ASCII string's widest character is its largest byte.
        ascii_bytes = b"".join(
            value
            for string, value in zip(strings, encoded, strict=True)
            if string.isascii()
        )
        most = int(np.frombuffer(ascii_bytes, dtype=np.uint8).max(initial=0))
        # Another string's widest is its largest code point in UTF-32.
        for string in strings:
            if not string.isascii():
                code_points = np.frombuffer(string.encode("utf-32-le"), dtype=np.uint32)
                most = max(most, int(code_points.max()))
                del code_points
        widest = max(widest, most)
        # The part's Python strings, their UTF-8 bytes, those of its ASCII
        # strings joined, their lists, and one string's code points.
        utf8_total = int(utf8[positions].sum())
        held = (
            2 * object_array_bytes(len(strings))
            + python_text_bytes(
                len(strings), (int(reach[stop - 1]) - base) // 4, widest=most
            )
            + 33 * len(strings)
            + 2 * utf8_total
            + 4 * int(ordered[cast_stop + stop - 1])
        )
        part_bytes = max(part_bytes, held)
        del strings, encoded, ascii_bytes
        first = stop
    return utf8, widest, part_bytes


def _file_buffer_bytes(array: Any, *, ranged: bool) -> int:
    """Return the buffer that each fetch of ``array``'s stored bytes holds, 0 for a store that reads no file.

    Zarr's local store reads a whole file, a chunk of an unsharded array or a
    whole shard, through ``Path.read_bytes``, and a byte range of a file,
    part of a shard of several chunks, through ``open``. Before Python 3.14
    both buffer the file at its file system's block size, or at
    ``io.DEFAULT_BUFFER_SIZE`` where that is unknown. From Python 3.14 a
    whole file is read unbuffered, and a byte range is buffered at the block
    size, at least ``io.DEFAULT_BUFFER_SIZE``, 128 KiB, and at most 8 MiB, so
    only an array that is ``ranged``, whose shards hold several chunks, is
    charged that buffer. Zarr's wrappers of the store are looked through.
    """
    store = getattr(array, "store", None)
    storage: Any = importlib.import_module("zarr.storage")
    while isinstance(store, storage.WrapperStore):
        store = store._store
    if not isinstance(store, storage.LocalStore):
        return 0
    block = _block_size(str(store.root))
    if not _WHOLE_FILES_UNBUFFERED:
        return block if block > 1 else io.DEFAULT_BUFFER_SIZE
    if not ranged:
        return 0
    return max(min(block, _FILE_BUFFER_LIMIT_BYTES), io.DEFAULT_BUFFER_SIZE)


@functools.cache
def _block_size(root: str) -> int:
    """Return the block size of the file system that holds ``root``, 0 where the platform reports none, as Windows does not.

    A store's files share its root's file system, so each root is asked once
    in a process.
    """
    return int(getattr(os.stat(root), "st_blksize", 0))


def _coalesce_gap_bytes(array: Any) -> int:
    """Return the most bytes apart that Zarr merges two byte ranges of a partial shard read of ``array``.

    A Zarr array keeps the setting of the configuration it was opened with;
    an array of another library, which Zarr does not read, merges none.
    """
    config = getattr(getattr(array, "async_array", array), "config", None)
    return max(0, int(getattr(config, "sharding_coalesce_max_gap_bytes", 0)))


def _fused_pool(array: Any) -> tuple[int, bool]:
    """Return the threads of the pool that Zarr's ``FusedCodecPipeline`` reads ``array`` on, and whether it reads the store asynchronously.

    An array opened while ``codec_pipeline.path`` names Zarr's opt-in
    ``FusedCodecPipeline`` keeps that pipeline in place of CyteArc's
    ``StoreThreadsCodecPipeline``, which ``ensure_zarr_host_ceiling``
    installs only in place of Zarr's default. The pipeline reads on a plain
    pool of the workers that Zarr resolves from ``codec_pipeline.max_workers``,
    by default one for each CPU, a pool that grows to the most workers any
    read asked for and never shrinks, and inline with one worker. It reads
    synchronously when every codec of the array decodes synchronously and the
    store serves Zarr's synchronous I/O, as local and memory stores do, and
    asynchronously otherwise. Every other pipeline gives (0, False).
    """
    async_array = getattr(array, "async_array", array)
    pipeline = getattr(async_array, "codec_pipeline", None)
    codec_pipeline: Any = importlib.import_module("zarr.core.codec_pipeline")
    if not isinstance(pipeline, codec_pipeline.FusedCodecPipeline):
        return 0, False
    workers = int(codec_pipeline._resolve_max_workers())
    if workers > 1:
        # A read of several chunks runs on Zarr's one pool, as many at once
        # as it has threads; one worker reads inline.
        workers = max(workers, int(codec_pipeline._pool_size))
    stores: Any = importlib.import_module("zarr.abc.store")
    reads_sync = pipeline.sync_transform is not None and bool(
        stores._store_supports_sync_io(async_array.store_path.store)
    )
    return workers, not reads_sync
