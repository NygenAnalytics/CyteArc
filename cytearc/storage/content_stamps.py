"""Reuse values computed from a stored array while its stored bytes are unchanged.

A value that depends only on what an array holds, such as the fingerprint of
its decoded text, is cached for the process under a stamp: a digest of the
array's metadata document, its runtime configuration, and the stored bytes of
each of its chunk or shard objects, or their absence. A stamp reads the encoded
bytes once, which costs far less than decoding and hashing every value again.
Any change to what the array holds, whether CyteArc, another process, or a raw
Zarr write made it, changes the stamp, so a cached value is never served for
other contents.
"""

import asyncio
import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from itertools import product

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype

from .partition import SCAN_BAND_BYTES

# Objects fetched at once: whole objects of up to SCAN_BAND_BYTES decoded,
# and at most this many.
_STAMP_BATCH_OBJECTS = 32
# Cached values; each is a short digest.
_CACHE_ENTRIES = 1024

_CACHE: OrderedDict[tuple[Hashable, bytes], object] = OrderedDict()
_LOCK = threading.Lock()


def _object_keys(array: zarr.Array) -> list[str]:
    """Return the keys of the array's chunk or shard objects, in grid order."""
    grid = array.shards or array.chunks
    counts = [-(-int(size) // int(step)) for size, step in zip(array.shape, grid)]
    return [
        array.metadata.encode_chunk_key(coordinates)
        for coordinates in product(*(range(count) for count in counts))
    ]


def stored_content_stamp(array: zarr.Array) -> bytes:
    """Return a digest of everything that decides the values ``array`` reads.

    It covers the metadata document that the array object decodes with, its
    runtime configuration, which decides whether a missing chunk raises, and
    each of its chunk or shard objects as stored, or that it is missing.
    """
    from zarr.core.sync import sync

    digest = hashlib.blake2b(digest_size=32, person=b"cytearc-stamp")

    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    prototype = default_buffer_prototype()
    documents = array.metadata.to_buffer_dict(prototype)
    for name in sorted(documents):
        add(name.encode("utf-8"))
        add(documents[name].to_bytes())
    add(repr(array.config).encode("utf-8"))
    grid = array.shards or array.chunks
    object_bytes = (
        math.prod(int(size) for size in grid) * np.dtype(array.dtype).itemsize
    )
    batch_size = max(
        1, min(_STAMP_BATCH_OBJECTS, SCAN_BAND_BYTES // max(1, object_bytes))
    )

    async def fetch(batch: list[str]) -> list[bytes | None]:
        buffers = await asyncio.gather(
            *((array.store_path / key).get(prototype=prototype) for key in batch)
        )
        return [None if buffer is None else buffer.to_bytes() for buffer in buffers]

    keys = _object_keys(array)
    for start in range(0, len(keys), batch_size):
        batch = keys[start : start + batch_size]
        for key, payload in zip(batch, sync(fetch(batch)), strict=True):
            add(key.encode("utf-8"))
            if payload is None:
                digest.update(b"\x00")
            else:
                digest.update(b"\x01")
                add(payload)
    return digest.digest()


def cached_for_content[T](
    purpose: Hashable,
    array: zarr.Array,
    compute: Callable[[], T],
) -> T:
    """Return ``compute()``, reused while the stored contents of ``array`` hold.

    ``compute`` must depend only on what ``array`` holds, and ``purpose`` names
    what it computes. A value is cached only when the array's stamp after
    ``compute`` equals the stamp before it, so a value that a concurrent write
    could have changed is returned but never reused.
    """
    key = (purpose, stored_content_stamp(array))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]  # type: ignore[return-value]
    value = compute()
    if stored_content_stamp(array) == key[1]:
        with _LOCK:
            _CACHE[key] = value
            while len(_CACHE) > _CACHE_ENTRIES:
                _CACHE.popitem(last=False)
    return value


def clear_content_cache() -> None:
    """Forget every reused value, so each is computed again on its next use."""
    with _LOCK:
        _CACHE.clear()
