import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import numpy as np
import zarr

from .types import as_zarr_array
from .metadata_keys import SOURCE_COLUMN_ATTRIBUTE
from .arrays import (
    MISSING_MASK_PREFIX,
    _decode_metadata_values,
    create_metadata_column,
    create_numeric_array,
    linked_missing_mask,
    MetadataBlock,
    create_streamed_metadata_column,
    stored_metadata_dtype,
)
from .budget import ResourceBudget
from .categorical import (
    CATEGORIES_LINK,
    CATEGORIES_PREFIX,
    ORDERED_ATTRIBUTE,
    categories_name,
    category_count,
    linked_categories,
    stored_categories,
)
from .geometry import array_geometry
from .layout import (
    PROFILE_METADATA_CHUNK,
    ChunkRows,
    column_chunk_rows,
    array_shard_rows,
    normed_array_spec,
)
from .partition import (
    affordable_width,
    band_shards,
    read_variable_text_rows,
    row_band,
    row_read_parts,
    scan_band,
)
from .profiles import StorageProfile
from .sharding import write_dense_in_shard_rows
from ..utils.memory_budget import MemoryBudgetError

COLUMN_METADATA_ATTRIBUTES = (
    "display",
    "feature_selection_fingerprint",
    "levels",
    "ordered",
    "role",
    "assay",
    "source",
    "description",
    "unit",
    SOURCE_COLUMN_ATTRIBUTE,
)


def band_read_bytes(src: zarr.Array, rows: int) -> Callable[[int], int]:
    """Return what reading a band of ``rows`` rows of every column of ``src`` holds beside the band, at each Zarr I/O concurrency.

    The bands are the runs of ``rows`` rows from the first, as a dense write
    of row bands of that many rows reads them, and each is read at the
    write's I/O concurrency, as
    :func:`~cytearc.storage.partition.row_read_parts` prices it: for a
    sharded source, the shard-level copy of the band's rows, and, for up to
    that many of the shards it touches at once, the compressed bytes of the
    inner chunks it touches and up to that many decoded chunks; for an
    unsharded source, up to that many chunks as stored and decoded.
    """
    geometry = array_geometry(src)
    columns = np.arange(int(src.shape[1]))
    shards = band_shards(geometry, range(int(src.shape[0])), rows)

    def held(concurrency: int) -> int:
        fixed, per_row = row_read_parts(
            geometry,
            columns,
            rows_contiguous=True,
            concurrency=concurrency,
            shards=shards,
        )
        return fixed + int(rows) * per_row

    return held


def copy_zarr_array(
    src: zarr.Array,
    dst: zarr.Array,
    msg: str | None = None,
    resources: ResourceBudget | None = None,
) -> None:
    """Stream-copy a 2D Zarr array in row blocks.

    Each block is one row band of ``dst``, read from ``src`` whole at the
    write's I/O concurrency. Its plan charges what that read holds beside the
    block, as :func:`band_read_bytes` states.
    """
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src {src.shape} vs dst {dst.shape}")
    if len(src.shape) != 2:
        raise ValueError("copy_zarr_array only supports 2D arrays")
    band_rows = min(array_shard_rows(dst), int(src.shape[0]))
    write_dense_in_shard_rows(
        dst,
        lambda start, end: np.asarray(src[start:end, :]),
        msg=msg or "Copying Zarr array",
        resources=resources,
        producerBytes=band_read_bytes(src, band_rows),
    )


def _metadata_block_rows(array: zarr.Array) -> int:
    return row_band(
        array_geometry(array),
        unit="chunk",
        fallback=PROFILE_METADATA_CHUNK,
    )


def _copy_chunk_rows(n_rows: int, itemsize: int, chunk_rows: ChunkRows | None) -> int:
    """The rows of each stored chunk of a copied column: the caller's rule, or 100,000."""
    if chunk_rows is None:
        return PROFILE_METADATA_CHUNK
    return chunk_rows(n_rows, max(1, itemsize))


def _metadata_copy_bytes(
    array: zarr.Array,
    rows: int,
    missing: zarr.Array | None,
    chunk_rows: ChunkRows | None = None,
) -> int:
    rows = min(rows, int(array.shape[0]))
    if rows == 0:
        return 0
    dtype = stored_metadata_dtype(array.dtype, lambda: ())
    # The copy and its missing mask share the chunks of the copied values.
    destination_rows = _copy_chunk_rows(int(array.shape[0]), dtype.itemsize, chunk_rows)
    destination_chunks = min(
        -(-int(array.shape[0]) // destination_rows),
        -(-rows // destination_rows) + 1,
    )
    total = 0
    sources = [(array, dtype.itemsize)]
    if missing is not None:
        sources.append((missing, 1))
    for source, output_size in sources:
        geometry = array_geometry(source)
        assert geometry is not None
        chunks = min(
            -(-int(source.shape[0]) // geometry.axisChunk(0)),
            -(-rows // geometry.axisChunk(0)) + 1,
        )
        # Include the previous yielded block, conversion, and destination
        # chunk buffers while Zarr decodes or encodes the current block.
        total += geometry.readBytes(rows * geometry.itemsize, chunks)
        total += 2 * rows * output_size
        total += 2 * destination_chunks * destination_rows * output_size
    return total


def _copy_metadata_array(
    src: zarr.Array,
    dst: zarr.Group,
    name: str,
    *,
    overwrite: bool,
    profile: StorageProfile | None = None,
    row_indices: np.ndarray | None = None,
    missing: zarr.Array | None = None,
    copy_attributes: bool = True,
    block_rows: int | None = None,
    chunk_rows: ChunkRows | None = None,
    categories: zarr.Array | None = None,
) -> None:
    """Stream-copy one metadata column, its rows ``row_indices`` or all.

    With ``copy_attributes``, the attributes in ``COLUMN_METADATA_ATTRIBUTES``
    that ``src`` carries, such as the role and assay of a membership column,
    are part of the copy's first metadata write. ``chunk_rows`` gives the
    copy's chunks from its rows and the bytes of each value as stored, such
    as :func:`~cytearc.storage.layout.column_chunk_rows` for a column of a
    cell or feature table; without it, chunks hold 100,000 rows. A
    categorical column's codes pass its ``categories``, which are copied
    whole, with their ``ordered`` attribute, before the codes that link them;
    a reserved entry follows them when they number exactly the copied rows.
    """
    if src.ndim != 1:
        raise ValueError(
            f"Metadata column {name!r} has {src.ndim} dimensions; metadata columns "
            "must be one-dimensional"
        )
    # Whole numeric and Unicode columns can overlap reads within a bounded band.
    # Selected rows and text needing conversion keep their smaller read blocks.
    if block_rows is None:
        block_rows = (
            scan_band(array_geometry(src), fallback=PROFILE_METADATA_CHUNK)
            if row_indices is None and np.dtype(src.dtype).kind in "biufcmMU"
            else _metadata_block_rows(src)
        )
    n_source = int(src.shape[0])
    dtype = stored_metadata_dtype(
        src.dtype,
        lambda: (
            src[start : start + block_rows] for start in range(0, n_source, block_rows)
        ),
    )
    n_rows = n_source if row_indices is None else len(row_indices)
    # Zarr's selection of scattered rows of variable-width text fails on some
    # NumPy releases, which read_variable_text_rows names.
    select = (
        read_variable_text_rows
        if np.dtype(src.dtype).kind == "T"
        else lambda array, rows: array.oindex[rows]
    )

    def blocks() -> Iterator[MetadataBlock]:
        for start in range(0, n_rows, block_rows):
            stop = min(start + block_rows, n_rows)
            rows = (
                slice(start, stop) if row_indices is None else row_indices[start:stop]
            )
            values = src[rows] if isinstance(rows, slice) else select(src, rows)
            mask = (
                None
                if missing is None
                else (
                    missing[rows] if isinstance(rows, slice) else missing.oindex[rows]
                )
            )
            yield MetadataBlock(
                start=start,
                values=np.asarray(_decode_metadata_values(values)).astype(
                    dtype, copy=False
                ),
                missing=None if mask is None else np.asarray(mask, dtype=bool),
            )

    attributes = (
        {key: src.attrs[key] for key in COLUMN_METADATA_ATTRIBUTES if key in src.attrs}
        if copy_attributes
        else {}
    )
    if categories is not None:
        # The copy holds its own rows, so its categories take a reserved entry
        # when they number exactly those rows, whatever the source held.
        labels, padding = stored_categories(
            np.asarray(categories[: category_count(categories)]), n_rows
        )
        create_metadata_column(
            dst,
            categories_name(name),
            data=labels,
            dtype=labels.dtype,
            overwrite=overwrite,
            chunkSize=_copy_chunk_rows(len(labels), labels.dtype.itemsize, chunk_rows),
            profile=profile,
            attributes={
                ORDERED_ATTRIBUTE: bool(categories.attrs[ORDERED_ATTRIBUTE]),
                **padding,
            },
        )
        attributes[CATEGORIES_LINK] = categories_name(name)
    create_streamed_metadata_column(
        dst,
        name,
        dtype=dtype,
        overwrite=overwrite,
        chunkSize=_copy_chunk_rows(n_rows, dtype.itemsize, chunk_rows),
        shape=n_rows,
        profile=profile,
        blocks=blocks(),
        hasMissing=missing is not None,
        attributes=attributes or None,
    )


def copy_metadata_array(
    src: zarr.Array,
    dst: zarr.Group,
    name: str,
    *,
    overwrite: bool = True,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Stream-copy one metadata vector without carrying presentation attrs.

    The copy is chunked as a column of a cell or feature table, by its rows
    and the bytes of each value as stored
    (:func:`~cytearc.storage.layout.column_chunk_rows`), so a short column is
    one chunk of its own rows.
    """
    _copy_metadata_array(
        src,
        dst,
        name,
        overwrite=overwrite,
        profile=profile,
        copy_attributes=False,
        chunk_rows=column_chunk_rows,
    )
    return as_zarr_array(dst[name], name=name)


def copy_zarr_group_tree(
    src: zarr.Group,
    dst: zarr.Group,
    *,
    overwrite: bool = True,
    exclude_members: set[str] | frozenset[str] | None = None,
    row_indices: np.ndarray | None = None,
    profile: StorageProfile | None = None,
    resources: ResourceBudget | None = None,
    chunk_rows: ChunkRows | None = None,
) -> None:
    """Recursively copy a Zarr group tree.

    ``exclude_members`` applies only to immediate members of ``src``. Child
    groups are copied recursively without inheriting the parent exclusions.
    ``chunk_rows`` gives each copied column's chunks, as
    :func:`_copy_metadata_array` takes it, and child groups take it too. A
    categorical column keeps its layout: the codes of the copied rows, and
    every category.
    """
    if row_indices is None:
        from .stores import prefetch_metadata_listing

        prefetch_metadata_listing(src)
    masks = validate_metadata_dependencies(src, exclude_members=exclude_members)
    from .stores import metadata_workers, run_concurrently

    members = list(src.members())
    opened = dict(members)
    arrays: list[tuple[str, zarr.Array]] = []
    groups: list[tuple[str, zarr.Group]] = []
    linked: dict[str, zarr.Array] = {}
    for name, node in members:
        if exclude_members is not None and name in exclude_members:
            continue
        # A mask and categories are copied together with the column that
        # links them.
        if name.startswith((MISSING_MASK_PREFIX, CATEGORIES_PREFIX)):
            continue
        if isinstance(node, zarr.Group):
            groups.append((name, node))
            continue
        array = as_zarr_array(node, name=name)
        arrays.append((name, array))
        categories = linked_categories(src, name, values=array, opened=opened)
        if categories is not None:
            linked[name] = categories

    planned_rows: dict[str, int] = {}
    workers = 1
    if resources is not None and row_indices is None:
        # Byte and variable-width strings create temporary Python objects
        # during conversion, so they retain their serial chunk-sized reads.
        fixed = [
            (name, array)
            for name, array in arrays
            if array.ndim == 1 and np.dtype(array.dtype).kind in "biufcmMU"
        ]
        chunk_bytes = {
            name: _metadata_copy_bytes(
                array, _metadata_block_rows(array), masks.get(name), chunk_rows
            )
            for name, array in fixed
        }
        minimum = max(chunk_bytes.values(), default=0)
        if minimum > resources.memoryBytes:
            # Every block holds at least one chunk of its column, so no plan,
            # serial or not, stays within a budget below one chunk.
            column = max(chunk_bytes, key=chunk_bytes.__getitem__)
            raise MemoryBudgetError(
                f"Copying one chunk of metadata column {column!r} needs {minimum} "
                f"bytes, but the memory budget is {resources.memoryBytes} bytes.",
                needed_bytes=minimum,
                budget_bytes=int(resources.memoryBytes),
                item=f"one chunk of metadata column {column!r}",
            )
        workers = max(
            1,
            min(
                metadata_workers(src),
                resources.workers,
                len(fixed),
                resources.memoryBytes // max(1, minimum),
            ),
        )
        for name, array in fixed:
            block = _metadata_block_rows(array)
            preferred = scan_band(array_geometry(array), fallback=block)
            chunks = affordable_width(
                lambda width: (
                    _metadata_copy_bytes(
                        array, width * block, masks.get(name), chunk_rows
                    )
                    <= resources.memoryBytes // workers
                ),
                max(1, preferred // block),
            )
            planned_rows[name] = max(1, chunks) * block

    def copy_column(name: str, array: zarr.Array) -> None:
        _copy_metadata_array(
            array,
            dst,
            name,
            overwrite=overwrite,
            row_indices=row_indices,
            missing=masks.get(name),
            profile=profile,
            block_rows=planned_rows.get(name),
            chunk_rows=chunk_rows,
            categories=linked.get(name),
        )

    run_concurrently(
        [
            partial(copy_column, name, array)
            for name, array in arrays
            if name in planned_rows
        ],
        workers=workers,
    )
    for name, array in arrays:
        if name not in planned_rows:
            copy_column(name, array)
    for name, node in groups:
        child = dst.create_group(name, overwrite=overwrite)
        copy_zarr_group_tree(
            node,
            child,
            overwrite=overwrite,
            row_indices=row_indices,
            profile=profile,
            resources=resources,
            chunk_rows=chunk_rows,
        )


def validate_metadata_dependencies(
    group: zarr.Group,
    *,
    exclude_members: set[str] | frozenset[str] | None = None,
) -> dict[str, zarr.Array]:
    """Return each column's missing mask, checking the masks and categories that columns link.

    Raises:
        ValueError: If a column links a mask or categories in a malformed way.
    """
    masks = {}
    for name, array in group.arrays():
        if exclude_members is not None and name in exclude_members:
            continue
        label = f"Column {name!r}"
        mask = linked_missing_mask(group, name, label=label, values=array)
        linked_categories(group, name, label=label, values=array)
        if mask is not None:
            masks[name] = mask
    return masks


def stage_normed_array(
    cache_path: str,
    source: zarr.Array,
    *,
    resources: ResourceBudget | None = None,
) -> zarr.Array:
    """Return the complete local copy of normalized ``source`` at ``cache_path``.

    A copy is made in a folder of its own beside ``cache_path`` and renamed
    into place once complete, so processes that share a cache folder never
    write into, or remove, a copy that another reads. Of processes that stage
    one array at once, the first rename places its copy, and the others read
    it and remove their own.

    Raises:
        RuntimeError: If ``cache_path`` holds an incomplete copy, as an
            interrupted staging of an earlier release left one in place.
    """
    shape = (int(source.shape[0]), int(source.shape[1]))
    if not _holds_normed_array(cache_path, shape):
        folder = f"{cache_path}.{uuid.uuid4().hex}.staging"
        try:
            root = zarr.open_group(Path(folder), mode="w-", use_consolidated=False)
            spec = normed_array_spec(shape[0], shape[1], profile="fast_local")
            staged = create_numeric_array(root, "data", spec)
            copy_zarr_array(
                source,
                staged,
                msg="Staging normalized data locally",
                resources=resources,
            )
            staged.attrs["complete"] = True
            try:
                _rename_once_free(folder, cache_path)
            except OSError:
                # Another process placed its copy first, which is read instead.
                if not _holds_normed_array(cache_path, shape):
                    raise
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    root = zarr.open_group(Path(cache_path), mode="r", use_consolidated=False)
    return as_zarr_array(root["data"], name="data")


# Windows refuses to rename a folder while another program, such as a virus
# scanner, holds a file in it, with access denied (5) or a sharing violation
# (32). Both clear within moments, so the rename is tried again after each of
# these delays, then once more. Off Windows an OSError has no winerror.
_BUSY_WINDOWS_ERRORS = frozenset({5, 32})
_RENAME_RETRY_DELAYS = (0.01, 0.05, 0.1, 0.25, 0.5)


def _rename_once_free(folder: str, target: str) -> None:
    """Rename ``folder`` to ``target``, trying again while Windows holds a file."""
    for delay in _RENAME_RETRY_DELAYS:
        try:
            os.rename(folder, target)
        except OSError as error:
            if getattr(error, "winerror", None) not in _BUSY_WINDOWS_ERRORS:
                raise
            time.sleep(delay)
        else:
            return
    os.rename(folder, target)


def _holds_normed_array(cache_path: str, shape: tuple[int, int]) -> bool:
    """Return whether ``cache_path`` holds a complete copy of this shape."""
    path = Path(cache_path)
    if not (path / "zarr.json").exists():
        return False
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    array = as_zarr_array(root["data"], name="data") if "data" in root else None
    if array is None or array.shape != shape or array.attrs.get("complete") is not True:
        raise RuntimeError(
            f"The local cache folder {cache_path} holds an incomplete copy, as an "
            "interrupted staging of an earlier CyteArc release leaves one. Remove "
            "it once no process uses this cache, and call again."
        )
    return True
