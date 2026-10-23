from typing import Literal, overload

import zarr
from zarr.errors import GroupNotFoundError


type ZarrMode = Literal["r", "r+", "a", "w", "w-"]


def as_zarr_array(
    node: zarr.Array | zarr.Group,
    *,
    name: str = "",
) -> zarr.Array:
    if isinstance(node, zarr.Array):
        return node
    label = f" at {name!r}" if name else ""
    raise TypeError(f"Expected Zarr array{label}, got {type(node).__name__}")


def writable(array: zarr.Array) -> zarr.Array:
    """Return ``array`` configured to write every chunk it receives.

    Zarr otherwise compares each chunk against the fill value before encoding
    it. CyteArc matrices rarely contain an all-fill chunk, and Zstd stores one in
    a few bytes, so that full scan costs more than it saves.
    """
    return array.with_config({"write_empty_chunks": True})


def complete_chunks(array: zarr.Array) -> zarr.Array:
    """Return ``array`` configured for a count matrix, which stores every chunk.

    Writes store every chunk, one of zeros too, as with :func:`writable`. A
    missing chunk object then means the store lost data, as an interrupted
    copy or upload can, so a read of one raises Zarr's ``ChunkNotFoundError``,
    a ``ValueError`` that names the array's store path and the missing chunk
    key, instead of reading fill values, which are zero counts. Zarr checks
    the objects that it fetches, the shards of a sharded array, so a chunk
    missing inside a stored shard still reads as fill values.
    """
    return array.with_config({"write_empty_chunks": True, "read_missing_chunks": False})


def as_zarr_group(
    node: zarr.Array | zarr.Group,
    *,
    name: str = "",
) -> zarr.Group:
    if isinstance(node, zarr.Group):
        return node
    label = f" at {name!r}" if name else ""
    raise TypeError(f"Expected Zarr group{label}, got {type(node).__name__}")


def array_metadata_shards(array: zarr.Array) -> tuple[int, ...] | None:
    return getattr(array.metadata, "shards", None)


@overload
def read_fresh_group(
    parent: zarr.Group,
    path: str = "",
    *,
    mode: Literal["r", "r+"] | None = None,
    missing_ok: Literal[False] = False,
) -> zarr.Group: ...


@overload
def read_fresh_group(
    parent: zarr.Group,
    path: str = "",
    *,
    mode: Literal["r", "r+"] | None = None,
    missing_ok: bool,
) -> zarr.Group | None: ...


def read_fresh_group(
    parent: zarr.Group,
    path: str = "",
    *,
    mode: Literal["r", "r+"] | None = None,
    missing_ok: bool = False,
) -> zarr.Group | None:
    """Open the group at ``path`` below ``parent`` from its stored record.

    Consolidated metadata is never read: it is a copy that writes to the
    store do not update.
    """
    try:
        return zarr.open_group(
            store=parent.store,
            path=(parent.store_path / path).path,
            mode=("r" if parent.read_only else "r+") if mode is None else mode,
            zarr_format=parent.metadata.zarr_format,
            use_consolidated=False,
        )
    except GroupNotFoundError:
        if missing_ok:
            return None
        raise
