"""Computed count arrays backed by immutable source datastores."""

import copy
import itertools
import json
import math
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import zarr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.codecs import Crc32cCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.dtype import UInt64
from zarr.storage import WrapperStore

from ..storage.artifacts import fingerprint_stored_arrays
from ..storage.geometry import ArrayGeometry, array_geometry
from ..storage.identity import count_fingerprint, validate_preparation
from ..storage.partition import partition_indices
from ..storage.types import as_zarr_array, as_zarr_group

COMPOSITE_ATTR = "composite"
_MAP_NAMES = ("rowSources", "sourceRows", "featureColumns")
_RECORD_KEYS = {"assay", "sources", "arrays", "maps", "complete"}
_SOURCE_KEYS = {
    "name",
    "location",
    "workspace",
    "datasetFingerprint",
    "countsFingerprint",
}


def refuse_composite(root: zarr.Group, *, operation: str) -> None:
    if COMPOSITE_ATTR in root.attrs:
        raise ValueError(f"A composite datastore cannot be {operation}")


def _bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(document, allow_nan=False, separators=(",", ":")).encode()


def _limits(request: ByteRequest | None, size: int) -> tuple[int, int]:
    match request:
        case None:
            return 0, size
        case RangeByteRequest(start=start, end=end):
            return min(start, size), min(end, size)
        case OffsetByteRequest(offset=start):
            return min(start, size), size
        case SuffixByteRequest(suffix=suffix):
            return max(0, size - suffix), size
    raise TypeError(f"Unsupported byte request: {request!r}")


def _array_spec(
    document: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], np.dtype[Any]]:
    shards = tuple(document["chunk_grid"]["configuration"]["chunk_shape"])
    codec = document["codecs"][0]
    if codec["name"] != "sharding_indexed":
        raise ValueError("Composite count arrays must use the paired sharded layout")
    chunks = tuple(codec["configuration"]["chunk_shape"])
    return chunks, shards, np.dtype(document["data_type"]).newbyteorder("<")


def _uncompressed(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    _array_spec(result)
    codec = result["codecs"][0]["configuration"]
    codec["codecs"] = [{"name": "bytes", "configuration": {"endian": "little"}}]
    codec["index_codecs"] = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "crc32c"},
    ]
    codec["index_location"] = "end"
    return result


def _read_scratch_bytes(
    geometry: ArrayGeometry,
    mappings: dict[str, ArrayGeometry],
    sources: Sequence[ArrayGeometry],
    *,
    transposed: bool,
) -> int:
    assert geometry.shards is not None
    cells, features = geometry.shards[::-1] if transposed else geometry.shards
    mapping_bytes = 0
    for name, result_bytes in (
        ("rowSources", cells * 4),
        ("sourceRows", cells * 8),
        ("featureColumns", features * 8),
    ):
        mapping = mappings[name]
        touched = max(1, math.ceil(result_bytes / mapping.nominalChunkBytes()) + 1)
        mapping_bytes += mapping.readBytes(result_bytes, chunks=touched)
    source_bytes = max(
        source.readBytes(source.nominalChunkBytes()) for source in sources
    )
    index_bytes = (
        32
        * math.prod(
            s // c for s, c in zip(geometry.shards, geometry.chunks, strict=True)
        )
        + 4
    )
    return (
        source_bytes
        + mapping_bytes
        + 2 * geometry.nominalChunkBytes()
        + 96 * (cells + features)
        + index_bytes
    )


async def _copy_source(
    source: zarr.Array,
    rows: np.ndarray,
    columns: np.ndarray,
    output: np.ndarray,
    row_destinations: np.ndarray,
    column_destinations: np.ndarray,
) -> None:
    """Copy selected values through rectangular reads within source chunks."""
    geometry = array_geometry(source)
    assert geometry is not None
    row_blocks = partition_indices(geometry, 0, rows)
    column_blocks = partition_indices(geometry, 1, columns)
    for row in row_blocks:
        row_start, row_stop = int(row.indices.min()), int(row.indices.max()) + 1
        for column in column_blocks:
            col_start, col_stop = (
                int(column.indices.min()),
                int(column.indices.max()) + 1,
            )
            values = np.asarray(
                await source.async_array.getitem(
                    (slice(row_start, row_stop), slice(col_start, col_stop))
                )
            )
            dest_rows, dest_columns = np.ix_(
                row_destinations[row.destinations],
                column_destinations[column.destinations],
            )
            destination: tuple[np.ndarray, ...]
            if output.ndim == 4:
                chunk_rows, chunk_columns = output.shape[2:]
                destination = (
                    dest_rows // chunk_rows,
                    dest_columns // chunk_columns,
                    dest_rows % chunk_rows,
                    dest_columns % chunk_columns,
                )
            else:
                destination = (dest_rows, dest_columns)
            output[destination] = values[
                np.ix_(row.indices - row_start, column.indices - col_start)
            ]
            del values, destination, dest_rows, dest_columns


class CompositeStore(WrapperStore[Store]):
    """Expose virtual counts while keeping ordinary metadata and results writable."""

    def __init__(
        self,
        store: Store,
        record: dict[str, Any],
        source_roots: dict[str, zarr.Group],
        *,
        constructing: bool = False,
        owns_sources: bool = False,
    ) -> None:
        super().__init__(store)
        self.record = record
        self.constructing = constructing
        self._roots = source_roots
        self._owns_sources = owns_sources
        self._maps: dict[str, Any] = {}
        self._scratch: dict[str, int] = {}
        self._sources: list[dict[str, zarr.Array]] = []
        for source in record["sources"]:
            root = source_roots[source["name"]]
            matrix_path = (
                f"matrices/{record['assay']}"
                if source["workspace"] is not None
                else record["assay"]
            )
            matrix = as_zarr_group(root[matrix_path], name=matrix_path)
            arrays = {"counts": as_zarr_array(matrix["counts"])}
            if "countsT" in matrix:
                arrays["countsT"] = as_zarr_array(matrix["countsT"])
            self._sources.append(arrays)

    def _with_store(self, store: Store) -> Self:
        clone = copy.copy(self)
        clone._store = store
        clone._owns_sources = False
        return clone

    @property
    def root(self) -> Any:
        return getattr(self._store, "root", None)

    @property
    def _supports_sync_io(self) -> bool:
        return False

    @property
    def supports_consolidated_metadata(self) -> bool:
        return False

    def _path(self, key: str) -> str | None:
        for name in ("counts", "countsT"):
            path = f"{self.record['assay']}/{name}"
            if key == path or key.startswith(path + "/"):
                return path
        return None

    def _shard(self, path: str, key: str) -> tuple[int, int] | None:
        suffix = key.removeprefix(path + "/")
        parts = suffix.split("/")
        if (
            len(parts) != 3
            or parts[0] != "c"
            or any(not value.isdigit() for value in parts[1:])
        ):
            return None
        coords = tuple(int(value) for value in parts[1:])
        document = self.record["arrays"][path]
        _, shards, _ = _array_spec(document)
        if any(
            value < 0 or value * width >= extent
            for value, width, extent in zip(
                coords, shards, document["shape"], strict=True
            )
        ):
            return None
        return coords[0], coords[1]

    async def _map(self, name: str) -> Any:
        if name not in self._maps:
            self._maps[name] = await zarr.api.asynchronous.open_array(
                store=self._store, path=f"_composite/{name}", mode="r"
            )
        return self._maps[name]

    async def _chunk(self, path: str, lower: tuple[int, int]) -> memoryview:
        document = self.record["arrays"][path]
        chunks, _, dtype = _array_spec(document)
        output = np.zeros(chunks, dtype=dtype)
        await self._fill(path, lower, chunks, output)
        return memoryview(output).cast("B")

    async def _fill(
        self,
        path: str,
        lower: tuple[int, int],
        shape: tuple[int, ...],
        output: np.ndarray,
    ) -> None:
        document = self.record["arrays"][path]
        upper = tuple(
            min(start + width, size)
            for start, width, size in zip(lower, shape, document["shape"], strict=True)
        )
        if any(stop <= start for start, stop in zip(lower, upper, strict=True)):
            return
        transposed = path.endswith("/countsT")
        cell_axis, feature_axis = (1, 0) if transposed else (0, 1)
        cell_slice = slice(lower[cell_axis], upper[cell_axis])
        feature_slice = slice(lower[feature_axis], upper[feature_axis])
        owners = np.asarray(await (await self._map("rowSources")).getitem(cell_slice))
        rows = np.asarray(await (await self._map("sourceRows")).getitem(cell_slice))
        if np.any(owners < 0) or np.any(owners >= len(self._sources)):
            raise ValueError("Composite row mapping names an invalid source")
        for owner in np.unique(owners):
            cell_dest = np.flatnonzero(owners == owner)
            source_rows = rows[cell_dest]
            columns = np.asarray(
                await (await self._map("featureColumns")).getitem(
                    (int(owner), feature_slice)
                )
            )
            feature_dest = np.flatnonzero(columns >= 0)
            source_features = columns[feature_dest]
            source = self._sources[int(owner)]["countsT" if transposed else "counts"]
            if (
                np.any(source_rows < 0)
                or np.any(source_rows >= source.shape[cell_axis])
                or np.any(source_features >= source.shape[feature_axis])
            ):
                raise ValueError("Composite mapping is outside a source matrix")
            await _copy_source(
                source,
                source_features if transposed else source_rows,
                source_rows if transposed else source_features,
                output,
                feature_dest if transposed else cell_dest,
                cell_dest if transposed else feature_dest,
            )

    async def _index(self, path: str) -> bytes:
        chunks, shards, dtype = _array_spec(self.record["arrays"][path])
        grid = tuple(s // c for s, c in zip(shards, chunks, strict=True))
        count = math.prod(grid)
        chunk_bytes = math.prod(chunks) * dtype.itemsize
        entries = np.empty((*grid, 2), dtype="<u8")
        entries[..., 0] = np.arange(count, dtype=np.uint64).reshape(grid) * chunk_bytes
        entries[..., 1] = chunk_bytes
        prototype = default_buffer_prototype()
        spec = ArraySpec(
            shape=(*grid, 2),
            dtype=UInt64(endianness="little"),
            fill_value=2**64 - 1,
            config=ArrayConfig(order="C", write_empty_chunks=False),
            prototype=prototype,
        )
        encoded = await Crc32cCodec().encode(
            [(prototype.buffer.from_bytes(entries.tobytes(order="C")), spec)]
        )
        value = next(iter(encoded))
        assert value is not None
        return value.to_bytes()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        path = self._path(key)
        if path is None:
            return await self._store.get(key, prototype, byte_range)
        document = self.record["arrays"].get(path)
        if document is None:
            if self.constructing or path.endswith("/countsT"):
                return None
            raise ValueError(f"Composite array metadata is missing: {path}")
        if key == path + "/zarr.json":
            metadata = _bytes(document)
            start, stop = _limits(byte_range, len(metadata))
            return prototype.buffer.from_bytes(metadata[start:stop])
        shard = self._shard(path, key)
        if shard is None:
            return None
        chunks, shards, dtype = _array_spec(document)
        grid = tuple(s // c for s, c in zip(shards, chunks, strict=True))
        count = math.prod(grid)
        chunk_bytes = math.prod(chunks) * dtype.itemsize
        data_bytes = count * chunk_bytes
        start, stop = _limits(byte_range, data_bytes + count * 16 + 4)
        result = bytearray(max(0, stop - start))
        data_stop = min(stop, data_bytes)
        if start == 0 and data_stop == data_bytes:
            # Fill the requested shard directly in its encoded chunk order.
            # Every participating source chunk is decoded once for this request.
            output = np.frombuffer(
                result, dtype=dtype, count=data_bytes // dtype.itemsize
            )
            output = output.reshape(*grid, *chunks)
            await self._fill(
                path,
                (shard[0] * shards[0], shard[1] * shards[1]),
                shards,
                output,
            )
        elif start < data_stop:
            for flat in range(start // chunk_bytes, (data_stop - 1) // chunk_bytes + 1):
                local = np.unravel_index(flat, grid)
                lower = tuple(
                    int(shard[i] * shards[i] + local[i] * chunks[i]) for i in range(2)
                )
                raw = await self._chunk(path, (lower[0], lower[1]))
                left, right = (
                    max(start, flat * chunk_bytes),
                    min(data_stop, (flat + 1) * chunk_bytes),
                )
                memoryview(result)[left - start : right - start] = raw[
                    left - flat * chunk_bytes : right - flat * chunk_bytes
                ]
                del raw
        if stop > data_bytes:
            index = await self._index(path)
            left = max(start, data_bytes)
            result[left - start : stop - start] = index[
                left - data_bytes : stop - data_bytes
            ]
        return prototype.buffer.from_bytes(result)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, request) for key, request in key_ranges]

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        async for item in Store._get_many(self, requests):
            yield item

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        **options: Any,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        if self._path(key) is not None:
            for index, request in enumerate(byte_ranges):
                yield [(index, await self.get(key, prototype, request))]
        else:
            async for group in self._store.get_ranges(
                key, byte_ranges, prototype=prototype, **options
            ):
                yield group

    async def exists(self, key: str) -> bool:
        path = self._path(key)
        if path is None:
            return await self._store.exists(key)
        if path not in self.record["arrays"]:
            return False
        return key == path + "/zarr.json" or self._shard(path, key) is not None

    async def getsize(self, key: str) -> int:
        path = self._path(key)
        if path is None:
            return await self._store.getsize(key)
        document = self.record["arrays"].get(path)
        if document is None:
            raise FileNotFoundError(key)
        if key == path + "/zarr.json":
            return len(_bytes(document))
        if self._shard(path, key) is None:
            raise FileNotFoundError(key)
        chunks, shards, dtype = _array_spec(document)
        count = math.prod(s // c for s, c in zip(shards, chunks, strict=True))
        return math.prod(shards) * dtype.itemsize + count * 16 + 4

    def _virtual_keys(self) -> Iterable[str]:
        for path, document in self.record["arrays"].items():
            yield path + "/zarr.json"
            _, shards, _ = _array_spec(document)
            for coords in itertools.product(
                *(
                    range(math.ceil(n / s))
                    for n, s in zip(document["shape"], shards, strict=True)
                )
            ):
                yield path + "/c/" + "/".join(map(str, coords))

    async def list(self) -> AsyncIterator[str]:
        async for key in self._store.list():
            yield key
        for key in self._virtual_keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self._store.list_prefix(prefix):
            yield key
        for key in self._virtual_keys():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        seen = set()
        async for name in self._store.list_dir(prefix):
            seen.add(name)
            yield name
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        for path, document in self.record["arrays"].items():
            for key in (path + "/zarr.json", path + "/c/"):
                if key.startswith(prefix) and key != prefix:
                    name = key[len(prefix) :].split("/", 1)[0]
                    if name and name not in seen:
                        seen.add(name)
                        yield name
            chunk_prefix = path + "/c/"
            if not prefix.startswith(chunk_prefix):
                continue
            tail = prefix[len(chunk_prefix) :].strip("/")
            coordinates = [] if not tail else tail.split("/")
            _, shards, _ = _array_spec(document)
            grid = tuple(
                math.ceil(n / s) for n, s in zip(document["shape"], shards, strict=True)
            )
            if len(coordinates) >= 2 or any(
                not value.isdigit() or int(value) >= grid[i]
                for i, value in enumerate(coordinates)
            ):
                continue
            for value in range(grid[len(coordinates)]):
                name = str(value)
                if name not in seen:
                    seen.add(name)
                    yield name

    async def is_empty(self, prefix: str) -> bool:
        async for _ in self.list_prefix(prefix):
            return False
        return True

    def _protected(self, key: str) -> bool:
        return (
            self._path(key) is not None
            or key == "_composite"
            or key.startswith("_composite/")
        )

    async def set(self, key: str, value: Buffer) -> None:
        if self.read_only:
            raise PermissionError("Composite store is read-only")
        path = self._path(key)
        if path is not None:
            if not self.constructing or key != path + "/zarr.json":
                raise PermissionError("Composite counts are immutable")
            self.record["arrays"][path] = _uncompressed(json.loads(value.to_bytes()))
            self._scratch.clear()
            return
        if not self.constructing and self._protected(key):
            raise PermissionError("Composite mappings are immutable")
        if key.endswith("zarr.json"):
            document = json.loads(value.to_bytes())
            if document.get("consolidated_metadata") is not None:
                raise PermissionError("Composite array metadata cannot be consolidated")
            if key == "zarr.json" and not self.constructing:
                if document.get("attributes", {}).get(COMPOSITE_ATTR) != self.record:
                    raise PermissionError("Composite bindings are immutable")
        await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if not await self.exists(key):
            await self.set(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        for key, value in values:
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        if self.read_only or key == "zarr.json":
            raise PermissionError(
                "Composite root cannot be deleted through its adapter"
            )
        if self._protected(key):
            if self.constructing and self._path(key) is not None:
                self.record["arrays"].pop(self._path(key), None)
                return
            raise PermissionError("Composite counts and mappings are immutable")
        await self._store.delete(key)

    async def delete_dir(self, prefix: str) -> None:
        path = prefix.rstrip("/")
        if self.read_only or not path:
            raise PermissionError(
                "Composite root cannot be deleted through its adapter"
            )
        if self._protected(path):
            if self.constructing and self._path(path) is not None:
                self.record["arrays"].pop(self._path(path), None)
                return
            raise PermissionError("Composite counts and mappings are immutable")
        if any(name.startswith(path + "/") for name in self.record["arrays"]):
            raise PermissionError(
                "A group containing composite counts cannot be deleted"
            )
        await self._store.delete_dir(prefix)

    async def clear(self) -> None:
        raise PermissionError("Composite root cannot be cleared through its adapter")

    def get_sync(self, *args: Any, **kwargs: Any) -> Buffer | None:
        raise TypeError("Composite stores use asynchronous routed IO")

    def set_sync(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Composite stores use asynchronous routed IO")

    def delete_sync(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Composite stores use asynchronous routed IO")

    async def _node_stores(self, path: str) -> tuple[Store, ...]:
        if self._path(path) is not None:
            return tuple(root.store for root in self._roots.values())
        return (self._store,)

    def read_scratch_bytes(self, path: str) -> int:
        if path not in self.record["arrays"]:
            return 0
        if path in self._scratch:
            return self._scratch[path]
        chunks, shards, dtype = _array_spec(self.record["arrays"][path])
        transposed = path.endswith("/countsT")
        # Mapping windows and source chunk reads are sequential; no decoder cache.
        mappings = {}
        for name in _MAP_NAMES:
            array = zarr.open_array(
                store=self._store, path=f"_composite/{name}", mode="r"
            )
            self._maps[name] = array.async_array
            geometry = array_geometry(array)
            assert geometry is not None
            mappings[name] = geometry
        sources = []
        for arrays in self._sources:
            array = arrays["countsT" if transposed else "counts"]
            geometry = array_geometry(array)
            assert geometry is not None
            sources.append(geometry)
        self._scratch[path] = _read_scratch_bytes(
            ArrayGeometry(
                tuple(self.record["arrays"][path]["shape"]),
                chunks,
                shards,
                dtype.itemsize,
            ),
            mappings,
            sources,
            transposed=transposed,
        )
        return self._scratch[path]

    def seal(self, root: zarr.Group) -> None:
        if not self.constructing:
            raise RuntimeError("Composite store is already finalized")
        self.record["complete"] = True
        root.attrs[COMPOSITE_ATTR] = self.record
        self.constructing = False

    def close(self) -> None:
        self._store.close()
        if self._owns_sources:
            for root in self._roots.values():
                root.store.close()

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()


def open_composite(
    root: zarr.Group, *, source_storage_options: dict[str, dict[str, Any]] | None = None
) -> zarr.Group:
    if isinstance(root.store, CompositeStore):
        return root
    record = root.attrs[COMPOSITE_ATTR]
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise ValueError("Invalid composite record")
    if record["complete"] is not True:
        raise ValueError("Composite creation is incomplete")
    assay = record["assay"]
    from ..storage.schema import validate_assay_name
    from ..storage.stores import make_store

    validate_assay_name(assay)
    sources = record["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("Composite has no source bindings")
    arrays = record["arrays"]
    if (
        not isinstance(arrays, dict)
        or f"{assay}/counts" not in arrays
        or set(arrays) - {f"{assay}/counts", f"{assay}/countsT"}
    ):
        raise ValueError("Invalid composite array metadata")
    for document in arrays.values():
        if _uncompressed(document) != document:
            raise ValueError("Invalid composite array codecs")
    maps = as_zarr_group(root["_composite"])
    if not isinstance(record["maps"], dict) or set(record["maps"]) != set(_MAP_NAMES):
        raise ValueError("Invalid composite mapping fingerprints")
    shape = arrays[f"{assay}/counts"]["shape"]
    expected_shapes = {
        "rowSources": (shape[0],),
        "sourceRows": (shape[0],),
        "featureColumns": (len(sources), shape[1]),
    }
    for name, expected in expected_shapes.items():
        array = as_zarr_array(maps[name])
        if array.shape != expected or array.dtype.kind not in "iu":
            raise ValueError(f"Invalid composite mapping: {name}")
        if fingerprint_stored_arrays(maps, [name]) != record["maps"][name]:
            raise ValueError(f"Composite mapping was modified: {name}")
    opened: dict[str, zarr.Group] = {}
    try:
        for binding in sources:
            if not isinstance(binding, dict) or set(binding) != _SOURCE_KEYS:
                raise ValueError("Invalid composite source binding")
            name = binding["name"]
            if not isinstance(name, str) or not name or name in opened:
                raise ValueError("Composite source names must be unique and nonempty")
            location = binding["location"]
            if not isinstance(location, str) or not location:
                raise ValueError("Composite sources need persistent locations")
            store = make_store(
                location,
                storage_options=(source_storage_options or {}).get(name),
                read_only=True,
            )
            source = (
                zarr.open_group(Path(store), mode="r")
                if isinstance(store, str)
                else zarr.open_group(store=store, mode="r")
            )
            opened[name] = source
            refuse_composite(source, operation="used as a composite source")
            if "matrixSource" in source.attrs:
                raise ValueError("Mounted datastores cannot be composite sources")
            workspace = binding["workspace"]
            logical = source if workspace is None else as_zarr_group(source[workspace])
            matrix = as_zarr_group(
                source[assay if workspace is None else f"matrices/{assay}"]
            )
            fingerprint = validate_preparation(
                as_zarr_group(logical[assay]),
                as_zarr_group(logical["cellData"]),
                matrix,
                require_transpose=f"{assay}/countsT" in arrays,
            )
            if (
                fingerprint != binding["datasetFingerprint"]
                or count_fingerprint(as_zarr_array(matrix["counts"]))
                != binding["countsFingerprint"]
            ):
                raise ValueError(f"Composite source identity changed: {name}")
        unknown = set(source_storage_options or {}) - set(opened)
        if unknown:
            raise ValueError(f"Unknown composite source options: {sorted(unknown)}")
        wrapped = CompositeStore(root.store, record, opened, owns_sources=True)
        return zarr.open_group(
            store=wrapped, mode="r" if root.read_only else "r+", use_consolidated=False
        )
    except BaseException:
        for source in opened.values():
            source.store.close()
        raise
