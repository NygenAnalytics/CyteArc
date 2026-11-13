"""Stored chunk geometry of a streamed array."""

import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

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


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """Chunk and shard extents an array is physically stored in."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    itemsize: int
    readScratchBytes: int = 0

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

    def wholeReadBytes(self, concurrency: int) -> int:
        """Return what reading the whole array at ``concurrency`` holds while it runs, with the result.

        An unsharded array is read in one call, which decodes up to
        ``concurrency`` chunks at once, each as stored and decoded. A sharded
        array is read a band of inner chunks at a time, as
        :meth:`bandReadBytes` counts, so that no fetch takes more than one
        band of a shard.
        """
        result = self.shape[0] * self.rowBytes()
        if self.shards is not None:
            return result + self.bandReadBytes(self.axisChunk(0), concurrency)
        n_chunks = -(-self.shape[0] // self.axisChunk(0)) * self.bandChunks()
        decoding = min(max(1, int(concurrency)), max(1, n_chunks))
        return result + decoding * self.storedChunkBytes()

    def bandReadBytes(self, rows: int, concurrency: int) -> int:
        """Return what reading ``rows`` rows of one band of chunks holds beside the result it fills.

        The read holds those rows as Zarr returns them and the band's chunks
        that it decodes at once, each as stored and decoded; a sharded array
        also fills a shard-level copy of the rows and holds the stored bytes
        of every chunk of the band it fetched.
        """
        rows_bytes = max(0, int(rows)) * self.rowBytes()
        touched = self.bandChunks()
        decoding = min(max(1, int(concurrency)), touched)
        if self.shards is None:
            band = decoding * self.storedChunkBytes()
        else:
            band = (
                rows_bytes
                + touched * self.storedChunkBytes()
                + decoding * self.nominalChunkBytes()
            )
        return rows_bytes + band + self.readScratchBytes

    def readBytes(
        self, resultBytes: int, chunks: int = 1, decodes: int | None = None
    ) -> int:
        """Return the bytes one Zarr read holds while it runs.

        A sharded read fills a shard-level copy of the selection before it
        copies that into the result, and holds the compressed bytes of every
        inner chunk it touches in a shard, which never exceed the decoded
        chunk, while it decodes them. An unsharded read holds its result and
        each chunk it decodes at once as stored and decoded, as
        :meth:`storedChunkBytes` counts, since Zarr keeps a chunk's compressed
        bytes beside it while it decodes it. ``chunks`` counts the inner
        chunks the read touches in one shard and ``decodes`` the chunks Zarr
        decodes at once, all of them unless given. Storage-provided scratch
        covers one shard request, including its sequential inner-chunk reads.
        Independent shard requests each need this allowance. Unsharded reads
        charge scratch per decode.
        """
        touched = max(1, int(chunks))
        decoding = touched if decodes is None else min(touched, max(1, int(decodes)))
        result = max(0, int(resultBytes))
        if self.shards is None:
            return result + decoding * (self.storedChunkBytes() + self.readScratchBytes)
        return (
            2 * result
            + (touched + decoding) * self.nominalChunkBytes()
            + self.readScratchBytes
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
    )
