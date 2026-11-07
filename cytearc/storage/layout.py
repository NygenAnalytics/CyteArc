from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

from .geometry import array_geometry
from .partition import contiguous_ranges, row_band
from .profiles import StorageProfile
from .types import array_metadata_shards

PROFILE_METADATA_CHUNK = 100_000
#: The most bytes of values that one stored chunk of a cell or feature column
#: holds, so that reading one row decodes at most this much: fixed-width
#: text, which stores every value at four bytes a character of the widest,
#: holds 100,000 rows to a chunk only up to 41 characters.
COLUMN_CHUNK_BYTES = 16 << 20
_CODEC_MAX_BYTES = 2_147_483_647
DEFAULT_TARGET_CHUNK_BYTES = 128 * 1024 * 1024

type ChunkRows = Callable[[int, int], int]
"""The rows of each stored chunk of a column, from its rows and the bytes of each value as stored.

A writer that encodes a column before it knows the column's stored width,
such as text measured at its widest value, takes the rule its caller passes,
which for a column of a cell or feature table is :func:`column_chunk_rows`.
"""


def column_chunk_rows(
    n_rows: int, itemsize: int, *, max_rows: int = PROFILE_METADATA_CHUNK
) -> int:
    """Return the rows of each stored chunk of a column of a cell or feature table.

    A chunk holds at most ``max_rows`` rows, no more rows than the column's
    ``n_rows``, and no more than :data:`COLUMN_CHUNK_BYTES` of its values,
    each ``itemsize`` bytes as stored, so a small table's column is one chunk
    of its rows and a wide text column takes fewer rows to a chunk. A chunk
    holds at least one row, so a value wider than the bound is a chunk of its
    own. A saved phylogeny chunks its node names the same way.

    Raises:
        ValueError: If ``n_rows`` is negative, or ``itemsize`` or
            ``max_rows`` is less than one.
    """
    if n_rows < 0:
        raise ValueError(f"A column cannot have {n_rows} rows")
    if itemsize < 1:
        raise ValueError(
            f"A column's values must take at least one byte each, not {itemsize}"
        )
    if max_rows < 1:
        raise ValueError(f"A chunk must hold at least one row, not {max_rows}")
    return max(1, min(int(max_rows), int(n_rows), COLUMN_CHUNK_BYTES // int(itemsize)))


def _encoded_chunk_bound(rawBytes: int) -> int:
    """Conservative encoded-size bound for the supported compressors."""
    raw_bytes = max(0, int(rawBytes))
    return raw_bytes + raw_bytes // 128 + 1024


def _divisors(value: int) -> tuple[int, ...]:
    small: list[int] = []
    large: list[int] = []
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            small.append(candidate)
            paired = value // candidate
            if paired != candidate:
                large.append(paired)
        candidate += 1
    return tuple(small + large[::-1])


@dataclass(frozen=True, slots=True)
class ZarrArraySpec:
    """Specification for creating a numeric Zarr array."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: Any
    compressors: tuple[Any, ...]
    shards: tuple[int, ...] | None = None
    fillValue: Any | None = None
    overwrite: bool = True


def get_compressors(
    profile: StorageProfile,
    zarrFormat: int = 3,
) -> tuple[Any, ...]:
    """Return codecs for a storage profile and Zarr format."""
    if zarrFormat == 2:
        from numcodecs import Blosc

        return (Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE),)
    if profile == "cloud":
        return (ZstdCodec(level=3),)
    return (BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle"),)


def _group_zarr_format(group: zarr.Group) -> int:
    return int(group.metadata.zarr_format)


def normalize_chunks(
    chunks: tuple[int, ...] | int,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Map a chunk specification to the dimensions in ``shape``."""

    def bounded_chunk(chunk: int, dimension: int) -> int:
        return max(1, min(max(1, int(chunk)), max(1, int(dimension))))

    if isinstance(chunks, int):
        chunks = (chunks,)
    if len(chunks) == len(shape):
        return tuple(
            bounded_chunk(chunk, dimension)
            for chunk, dimension in zip(chunks, shape, strict=True)
        )
    if len(chunks) == 1 and len(shape) > 1:
        return (bounded_chunk(chunks[0], shape[0]),) + tuple(
            max(1, int(dimension)) for dimension in shape[1:]
        )
    raise ValueError(
        f"Cannot map chunks {chunks} to array shape {shape}. "
        "Provide one chunk size per dimension."
    )


def count_array_spec(
    nCells: int,
    nFeats: int,
    dtype: Any = "uint32",
    *,
    profile: StorageProfile,
    policy: Any | None = None,
) -> ZarrArraySpec:
    """Build the paired rotateOnce U/Q specification for an assay count matrix."""
    from .count_matrix import DEFAULT_COUNT_MATRIX_POLICY, plan_count_matrix_pair

    return plan_count_matrix_pair(
        nCells,
        nFeats,
        dtype,
        policy=policy or DEFAULT_COUNT_MATRIX_POLICY,
        profile=profile,
    ).counts


def normed_array_spec(
    nCells: int,
    nFeats: int,
    *,
    profile: StorageProfile,
    targetChunkBytes: int | None = None,
    zarrFormat: int = 3,
) -> ZarrArraySpec:
    """Build an array specification for normalized expression data."""
    n_cells = max(1, int(nCells))
    n_feats = max(1, int(nFeats))
    row_bytes = n_feats * np.dtype("float32").itemsize
    if row_bytes > _CODEC_MAX_BYTES:
        raise ValueError(
            f"One full-width row requires {row_bytes} bytes, exceeding "
            f"the codec input limit of {_CODEC_MAX_BYTES} bytes"
        )
    requested_target = (
        DEFAULT_TARGET_CHUNK_BYTES
        if targetChunkBytes is None
        else int(targetChunkBytes)
    )
    if requested_target <= 0:
        raise ValueError("Chunk target must be positive")
    target = min(requested_target, _CODEC_MAX_BYTES)
    row_chunk = max(1, min(n_cells, target // row_bytes))
    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=(row_chunk, n_feats),
        shards=None,
        dtype="float32",
        compressors=get_compressors(profile, zarrFormat=zarrFormat),
        fillValue=0.0,
    )


def row_sharded_array_spec(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    profile: StorageProfile,
    band_rows: int,
    target_chunk_bytes: int | None = None,
    zarr_format: int = 3,
    fill_value: Any | None = 0,
) -> ZarrArraySpec:
    """Build a full-width array specification sharded along the first axis."""
    if not shape:
        raise ValueError("Row-sharded arrays require at least one dimension")
    dimensions = tuple(int(value) for value in shape)
    if any(value < 0 for value in dimensions):
        raise ValueError("Array dimensions cannot be negative")
    if int(band_rows) < 1:
        raise ValueError("band_rows must be positive")
    trailing = dimensions[1:]
    trailing_values = int(np.prod(trailing, dtype=np.int64)) if trailing else 1
    row_bytes = trailing_values * int(np.dtype(dtype).itemsize)
    if row_bytes < 1 or row_bytes > _CODEC_MAX_BYTES:
        raise ValueError(
            f"One full-width row requires {row_bytes} bytes, exceeding "
            f"the codec input limit of {_CODEC_MAX_BYTES} bytes"
        )
    requested_target = (
        DEFAULT_TARGET_CHUNK_BYTES
        if target_chunk_bytes is None
        else int(target_chunk_bytes)
    )
    if requested_target < 1:
        raise ValueError("Chunk target must be positive")
    n_rows = max(1, dimensions[0])
    shard_rows = min(n_rows, int(band_rows))
    codec_rows = max(1, _CODEC_MAX_BYTES // row_bytes)
    target_rows = max(
        1,
        min(
            shard_rows,
            requested_target // row_bytes,
            codec_rows,
        ),
    )
    if zarr_format >= 3:
        candidates = tuple(
            value for value in _divisors(shard_rows) if value <= codec_rows
        )
        chunk_rows = min(
            candidates,
            key=lambda value: (abs(value - target_rows), value > target_rows),
            default=1,
        )
        # A shard extent without a divisor near the target, such as a prime
        # or twice a prime row count, would split into tiny chunks.
        if 2 * chunk_rows < target_rows:
            shard_rows = target_rows
            chunk_rows = target_rows
    else:
        chunk_rows = target_rows
    shard_shape = (shard_rows, *tuple(max(1, value) for value in trailing))
    chunk_shape = (
        chunk_rows,
        *tuple(max(1, value) for value in trailing),
    )
    return ZarrArraySpec(
        shape=dimensions,
        chunks=chunk_shape,
        shards=shard_shape if zarr_format >= 3 else None,
        dtype=dtype,
        compressors=get_compressors(profile, zarrFormat=zarr_format),
        fillValue=fill_value,
    )


def array_shard_rows(array: zarr.Array) -> int:
    """Return the row extent of a shard, chunk, or full array."""
    return row_band(
        array_geometry(array),
        unit="shard",
        fallback=max(int(array.shape[0]), 1),
    )


def iter_shard_row_slices(
    n_rows: int,
    shard_rows: int,
) -> Iterator[tuple[int, int]]:
    """Yield row slices aligned to a shard height."""
    yield from contiguous_ranges(n_rows, shard_rows)


def array_info(array: zarr.Array) -> str:
    """Return a compact description of a Zarr array."""
    parts = [f"shape={array.shape}", f"dtype={array.dtype}", f"chunks={array.chunks}"]
    shards_meta = array_metadata_shards(array)
    if shards_meta is not None:
        parts.append(f"shards={shards_meta}")
    return ", ".join(parts)
