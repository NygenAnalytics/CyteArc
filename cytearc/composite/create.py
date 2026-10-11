import asyncio
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import numpy as np
import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, WrapperStore

from ..assay.classification import default_feature_sets, declared_assay_type
from ..merge.features import FeatureAlignment, align_features, resolve_merge_dtype
from ..merge.metadata import (
    SOURCE_MEASURES_ALL,
    admit_cell_metadata_plan,
    metadata_chunk_rows,
    plan_cell_metadata,
    resolve_metadata_schema_scan_rows,
    write_cell_metadata,
    write_feature_metadata,
)
from ..merge.row_plan import (
    RowPlan,
    build_row_plan,
    iter_row_plan_segments,
    max_row_plan_block_rows,
)
from ..merge.writer import create_assay_counts
from ..metadata.membership import resolve_assay_membership
from ..metadata.rows import default_block_rows
from ..storage.artifacts import fingerprint_stored_arrays
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPairPlan,
    CountMatrixPolicy,
    create_count_matrix_array,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from ..storage.destinations import (
    check_destination,
    create_destination,
    refuse_pending_assays,
)
from ..storage.execution import admitted_worker_split
from ..storage.geometry import ArrayGeometry, array_geometry
from ..storage.identity import (
    CountSummary,
    count_fingerprint,
    feature_sums_key,
    finalize_counts,
    generated_cell_columns,
    validate_preparation,
    write_feature_sums,
)
from ..storage.metadata_keys import assay_membership_column
from ..storage.partition import checked_indices
from ..storage.profiles import is_remote_zarr_location, local_zarr_path
from ..storage.schema import validate_assay_name
from ..storage.stores import (
    MATRIX_SOURCE_ATTR,
    locations_overlap,
    prefetch_metadata_listing,
)
from ..storage.types import as_zarr_array, as_zarr_group
from .source_cache import CachedSourceStore, SourceCache, new_source_cache

if TYPE_CHECKING:
    from ..datastore.datastore import DataStore
    from ..metadata.table import MetaData


class _PrefetchingMetadataStore(CachedSourceStore):
    """Read one metadata column's chunks through the cache, each fetched whole.

    Creation reads every chunk of a remote column more than once, for its
    schema scan and for its copy, so each chunk object is fetched once.
    """

    def __init__(self, array: zarr.Array, cache: SourceCache) -> None:
        super().__init__(array.store, cache)
        geometry = array_geometry(array)
        assert geometry is not None
        self._prefix = f"{array.path}/" if array.path else ""
        self._max_bytes = geometry.nominalChunkBytes()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if key.startswith(self._prefix) and key.rsplit("/", 1)[-1] not in {
            "zarr.json",
            ".zarray",
            ".zgroup",
            ".zattrs",
            ".zmetadata",
        }:
            # The two encoded copies precede decoding and fit the metadata
            # read allowance for one decoded and one encoded source chunk.
            await self.prefetch(key, max_bytes=self._max_bytes)
        return await super().get(key, prototype, byte_range)


class _CachedMetadataTable:
    """A source metadata table whose columns read through a ``SourceCache``.

    It offers the table methods that metadata planning and writing use, and
    opens each column once.
    """

    def __init__(self, table: "MetaData") -> None:
        self._table = table
        self.N = table.N
        self.columns = table.columns
        self._arrays: dict[str, zarr.Array] = {}
        self._masks: dict[str, zarr.Array | None] = {}
        self._source_cache: SourceCache | None = None

    @property
    def source_cache(self) -> SourceCache | None:
        """The cache that the columns read through, if any."""
        return self._source_cache

    def use_source_cache(self, cache: SourceCache) -> None:
        self._source_cache = cache

    def _cache_array(self, array: zarr.Array) -> zarr.Array:
        if self._source_cache is None or isinstance(array.store, CachedSourceStore):
            return array
        store = _PrefetchingMetadataStore(array, self._source_cache)
        return zarr.open_array(store=store, path=array.path, mode="r")

    def _get_array(self, column: str) -> zarr.Array:
        if column not in self._arrays:
            self._arrays[column] = self._table._get_array(column)
        self._arrays[column] = self._cache_array(self._arrays[column])
        return self._arrays[column]

    def get_dtype(self, column: str) -> np.dtype[Any]:
        return self._get_array(column).dtype

    def _get_missing_mask_array(self, column: str) -> zarr.Array | None:
        if column not in self._masks:
            self._masks[column] = self._table._get_missing_mask_array(column)
        array = self._masks[column]
        if array is not None:
            self._masks[column] = self._cache_array(array)
        return self._masks[column]

    def _bool_array(self, key: str) -> zarr.Array:
        array = self._get_array(key)
        if array.dtype != np.dtype(bool):
            raise TypeError(f"Metadata column {key!r} must have boolean dtype")
        return array

    def default_block_rows(self, column: str = "I") -> int:
        return default_block_rows(self, column)


def _prefetch_metadata_schema(
    tables: list[_CachedMetadataTable],
    excluded_columns: list[frozenset[str]],
    resources: ResourceBudget,
    *,
    cache: SourceCache,
    resident_bytes: int,
    read_workers: int,
) -> None:
    arrays = []
    for table, excluded in zip(tables, excluded_columns, strict=True):
        if table.source_cache is not cache:
            continue
        for column in table.columns:
            identity = column in {"ids", "names"}
            if not identity and (column == "I" or column in excluded):
                continue
            array = table._get_array(column)
            if not identity and array.dtype.kind not in {"U", "S", "O"}:
                continue
            geometry = array_geometry(array)
            assert geometry is not None
            objects = -(-array.shape[0] // geometry.axisShard(0))
            if objects:
                arrays.append((array, objects, geometry.nominalChunkBytes()))
    objects = sum(count for _array, count, _size in arrays)
    nominal_bytes = sum(count * size for _array, count, size in arrays)
    # Warming more than the fresh cache can hold would evict objects before
    # the schema scan reaches them and cause a second source read.
    if (
        not objects
        or objects > cache.max_entries
        or nominal_bytes > cache.capacity_bytes
    ):
        return
    workers, _ = admitted_worker_split(
        resources,
        nTasks=objects,
        residentBytes=resident_bytes,
        taskBytes=lambda _: 2 * (max(size for _array, _count, size in arrays) + 1),
        requested=read_workers,
    )
    requests = iter(
        (array, coordinate, size)
        for array, count, size in arrays
        for coordinate in range(count)
    )

    async def prefetch() -> None:
        async def worker() -> None:
            for array, coordinate, size in requests:
                store = array.store
                assert isinstance(store, CachedSourceStore)
                key = "/".join(
                    part
                    for part in (
                        array.path,
                        array.metadata.encode_chunk_key((coordinate,)),
                    )
                    if part
                )
                await store.prefetch(key, max_bytes=size)

        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())

    sync(prefetch())


def _prefetch_counts(
    arrays: list[zarr.Array],
    row_plan: RowPlan,
    feature_maps: list[np.ndarray],
    geometries: list[ArrayGeometry | None],
    *,
    segment_rows: int,
    read_workers: int,
    memory_bytes: int,
    cache: SourceCache,
) -> None:
    from .reader import shard_prefetch_plan

    if memory_bytes <= 4096:
        return

    def requests() -> Iterator[tuple[CachedSourceStore, str, int]]:
        for segment in iter_row_plan_segments(row_plan, segment_rows=segment_rows):
            owner = segment.sourceIdx
            geometry = geometries[owner]
            assert geometry is not None
            columns = np.flatnonzero(feature_maps[owner] >= 0)
            if not columns.size:
                continue
            plan = shard_prefetch_plan(
                arrays[owner], segment.localRows, columns, geometry
            )
            if plan is None:
                continue
            assert plan.store.cache is cache
            column_shards = np.unique(columns // geometry.axisShard(1))
            for row in np.unique(segment.localRows // geometry.axisShard(0)):
                for column in column_shards:
                    key = plan.shardKey(
                        int(row) * geometry.axisShard(0),
                        int(column) * geometry.axisShard(1),
                    )
                    if key is not None:
                        yield plan.store, key, plan.maxBytes

    async def prefetch() -> None:
        seen: set[tuple[int, str]] = set()
        descriptor_bytes = 4096
        wave: list[tuple[CachedSourceStore, str, int]] = []
        wave_bytes = 0
        wave_memory = 0

        async def flush() -> bool:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(store.prefetch(key, max_bytes=limit))
                    for store, key, limit in wave
                ]
            return all(task.result() for task in tasks)

        for store, key, limit in requests():
            identity = (store.source_key, key)
            if identity in seen:
                continue
            if len(seen) >= cache.max_entries:
                break
            retained_bytes = 4 * len(key) + 512
            task_bytes = 2 * (limit + 1) + 1024

            def fits() -> bool:
                # Creation owns the idle cache exclusively between joined waves.
                return (
                    wave_bytes + limit <= cache.free_bytes
                    and len(wave) + 1 <= cache.free_entries
                    and descriptor_bytes + retained_bytes + wave_memory + task_bytes
                    <= memory_bytes
                )

            if wave and (len(wave) == read_workers or not fits()):
                if not await flush():
                    return
                wave.clear()
                wave_bytes = wave_memory = 0
            if not fits():
                break
            seen.add(identity)
            descriptor_bytes += retained_bytes
            wave.append((store, key, limit))
            wave_bytes += limit
            wave_memory += task_bytes
        if wave:
            await flush()

    sync(prefetch())


def _source_location(source: Any) -> str:
    location = source.zarr_loc
    if not isinstance(location, str):
        store = source.z.store
        while isinstance(store, WrapperStore):
            store = store._store
        if isinstance(store, LocalStore):
            location = str(store.root)
        elif isinstance(store, FsspecStore):
            location = str(store.fs.unstrip_protocol(store.path))
        else:
            raise ValueError(
                "Composite sources must have a persistent filesystem location"
            )
    if not is_remote_zarr_location(location):
        return str(Path(local_zarr_path(location)).resolve())
    parsed = urlsplit(location)
    if parsed.username is not None or parsed.query:
        raise ValueError(
            "Composite source URLs cannot contain user information or query "
            "parameters; provide credentials through storage_options"
        )
    return location


def _feature_alignment(
    assays: list[Any], names: list[str], features: Literal["union", "intersection"]
) -> tuple[FeatureAlignment, np.ndarray, str]:
    union = align_features(assays, names, require_overlap=False)
    dtype = resolve_merge_dtype(assays, union.featOrderMap)
    presence = np.zeros(union.nFeats, dtype=np.int64)
    for mapping in union.featOrderMap:
        presence[mapping] += 1
    selected = np.flatnonzero(
        presence == len(assays) if features == "intersection" else presence > 0
    )
    if selected.size == 0:
        raise ValueError("The composite feature space is empty")
    positions = np.full(union.nFeats, -1, dtype=np.int64)
    positions[selected] = np.arange(selected.size)
    mappings = [positions[mapping] for mapping in union.featOrderMap]
    frame = union.mergedFeatsMap.iloc[selected].copy().reset_index(drop=True)
    frame["idx"] = np.arange(selected.size)
    alignment = FeatureAlignment(frame, mappings, selected.size, union.overlapFraction)
    columns = np.full((len(assays), selected.size), -1, dtype=np.int64)
    for index, mapping in enumerate(mappings):
        source_columns = np.flatnonzero(mapping >= 0)
        columns[index, mapping[source_columns]] = source_columns
    return alignment, columns, dtype


def _fit_count_layout(
    n_cells: int,
    n_features: int,
    dtype: str,
    source_arrays: dict[str, list[zarr.Array]],
    *,
    mapping_width: int,
    memory_bytes: int,
) -> CountMatrixPairPlan:
    from .reader import selection_scratch_bytes
    from .store import computed_shard_scratch_bytes

    mappings = {
        "rowSources": ArrayGeometry((n_cells,), (mapping_width,), None, 4),
        "sourceRows": ArrayGeometry((n_cells,), (mapping_width,), None, 8),
        "featureColumns": ArrayGeometry(
            (len(source_arrays["counts"]), n_features),
            (1, min(n_features, mapping_width)),
            None,
            8,
        ),
    }
    sources: list[dict[str, ArrayGeometry]] = [{} for _ in source_arrays["counts"]]
    for name, arrays in source_arrays.items():
        for geometries, array in zip(sources, arrays, strict=True):
            geometry = array_geometry(array)
            assert geometry is not None
            geometries[name] = geometry
    itemsize = np.dtype(dtype).itemsize
    policy = DEFAULT_COUNT_MATRIX_POLICY
    while True:
        pair = plan_count_matrix_pair(
            n_cells, n_features, dtype, policy=policy, profile="fast_local"
        )
        peak = 0
        for name, spec in (("counts", pair.counts), ("countsT", pair.countsT)):
            if name not in source_arrays:
                continue
            geometry = ArrayGeometry(spec.shape, spec.chunks, spec.shards, itemsize)
            assert geometry.shards is not None
            # A shard fill, the read the layout must admit, through the reader.
            cells, features = (
                min(s, n)
                for s, n in zip(
                    geometry.shards[::-1] if name == "countsT" else geometry.shards,
                    (n_cells, n_features),
                    strict=True,
                )
            )
            geometry = replace(
                geometry,
                readScratchBytes=computed_shard_scratch_bytes(
                    selection_scratch_bytes(
                        mappings, sources, cells=cells, features=features
                    ),
                    geometry,
                ),
            )
            if name == "counts":
                assert geometry.shards is not None
                touched = math.prod(
                    s // c
                    for s, c in zip(geometry.shards, geometry.chunks, strict=True)
                )
                band_bytes = min(n_cells, spec.chunks[0]) * n_features * itemsize
                cost = geometry.readBytes(band_bytes, touched, decodes=1)
            else:
                width = min(n_features, pair.readGroup.featureWidth)
                band_bytes = width * min(n_cells, spec.chunks[1]) * itemsize
                cost = (
                    width * n_cells * itemsize
                    + geometry.readBytes(band_bytes, math.ceil(width / spec.chunks[0]))
                    + 3 * n_cells * np.dtype(np.int64).itemsize
                )
            peak = max(peak, cost)
        if peak <= memory_bytes:
            return pair
        if policy.unitBytes == 1:
            raise MemoryError("Composite count reads cannot fit within mem_budget")
        policy = CountMatrixPolicy(
            max(1, policy.unitBytes // 2), max(1, policy.chunkBytes // 2)
        )


def create_composite(
    sources: Mapping[str, "DataStore"],
    *,
    at: str | Path,
    features: Literal["union", "intersection"],
    assay: str = "RNA",
    rows: Mapping[str, Sequence[int] | np.ndarray] | None = None,
    mem_budget: int | str | None = None,
    nthreads: int | None = None,
) -> "DataStore":
    """Create a writable datastore that reads its counts from frozen sources."""
    from ..datastore.datastore import DataStore
    from .reader import copy_source_selection, source_index_bytes
    from .store import CompositeStore

    if not sources:
        raise ValueError("At least one source datastore is required")
    if features not in {"union", "intersection"}:
        raise ValueError("features must be 'union' or 'intersection'")
    validate_assay_name(assay)
    if assay == "_composite":
        raise ValueError("Assay name '_composite' is reserved for composite mappings")
    names = list(sources)
    if any(
        not isinstance(name, str) or not name.strip() or "__" in name for name in names
    ):
        raise ValueError("Source names must be nonempty strings without '__'")
    datasets = list(sources.values())
    if any(not isinstance(source, DataStore) for source in datasets):
        raise TypeError("Every source must be an open DataStore")
    selections = dict(rows or {})
    if set(selections) - set(names):
        raise ValueError("Selections must name an existing source")
    target = str(at)
    if is_remote_zarr_location(target):
        raise ValueError("The composite destination must be a local path")
    destination = Path(local_zarr_path(target)).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Composite destination already exists: {destination}")
    check_destination(str(destination))
    resources = resolve_budget(
        mem_budget
        if mem_budget is not None
        else min(ds.memoryBytes for ds in datasets),
        nthreads if nthreads is not None else min(ds.nthreads for ds in datasets),
    )
    assays: list[Any] = []
    selected_rows = []
    bindings = []
    source_roots = {}
    membership = []
    for name, source in zip(names, datasets, strict=True):
        if MATRIX_SOURCE_ATTR in source.z.attrs or "composite" in source.z.attrs:
            raise ValueError("Mounted and composite sources are not supported")
        refuse_pending_assays(
            source.z, operation="composited", subject=f"Source {name!r}"
        )
        if assay not in source.assay_names:
            raise ValueError(f"Source {name!r} has no assay {assay!r}")
        source_assay = source.get_assay(assay)
        fingerprint = validate_preparation(
            source_assay.z,
            source.cells.locations["primary"],
            source_assay.matrixGroup,
            require_transpose=source_assay.requiresCountsT,
        )
        location = _source_location(source)
        if locations_overlap(str(destination), location):
            raise ValueError("The composite destination must not overlap a source")
        selection = selections.get(name)
        selected_rows.append(
            None
            if selection is None
            else np.sort(
                checked_indices(selection, limit=source.cells.N, name=f"rows[{name!r}]")
            )
        )
        bindings.append(
            {
                "name": name,
                "location": location,
                "workspace": source.workspace,
                "datasetFingerprint": fingerprint,
                "countsFingerprint": count_fingerprint(
                    as_zarr_array(source_assay.matrixGroup["counts"], name="counts")
                ),
            }
        )
        source_roots[name] = zarr.open_group(store=source.z.store, mode="r")
        membership.append(
            resolve_assay_membership(source.cells, assay) or SOURCE_MEASURES_ALL
        )
        assays.append(source_assay)
    assay_types = {declared_assay_type(source) for source in assays}
    if len(assay_types) != 1:
        raise ValueError(f"Sources declare different types for assay {assay!r}")
    requires_transpose = assays[0].requiresCountsT
    for source, source_assay in zip(datasets, assays, strict=True):
        prefetch_metadata_listing(source.cells._group)
        prefetch_metadata_listing(source_assay.feats._group)
    alignment, feature_columns, dtype = _feature_alignment(assays, names, features)
    row_plan = build_row_plan(
        [source.cells.N for source in datasets],
        [int(source.rawData.chunksize[0]) for source in assays],
        names,
        selected_rows=selected_rows,
    )
    del selected_rows
    if row_plan.nCells == 0:
        raise ValueError("The composite must contain at least one selected cell")
    resident = (
        row_plan.resident_bytes() + alignment.resident_bytes() + feature_columns.nbytes
    )
    if resident >= resources.memoryBytes:
        raise MemoryError("Composite row and feature mappings exceed mem_budget")
    cell_tables = [_CachedMetadataTable(source.cells) for source in datasets]
    scan_rows = resolve_metadata_schema_scan_rows(
        cell_tables,
        resources,
        resident_bytes=resident,
        preferred_rows=max(
            min(table.N, table._get_array("ids").chunks[0]) for table in cell_tables
        ),
    )
    # Remote cell metadata reads through its own cache, which is closed once
    # the metadata is copied; the composite store keeps a cache for counts.
    metadata_cache: SourceCache | None = None
    wrapper: CompositeStore | None = None
    try:
        schema_read_workers = resources.workers
        for binding, table, source in zip(bindings, cell_tables, datasets, strict=True):
            binding_location = binding["location"]
            assert binding_location is not None
            if is_remote_zarr_location(binding_location):
                if metadata_cache is None:
                    metadata_cache = new_source_cache()
                if metadata_cache is not None:
                    table.use_source_cache(metadata_cache)
                if source.storageIo is not None:
                    schema_read_workers = min(
                        schema_read_workers,
                        source.storageIo.readWorkers or resources.workers,
                    )
        del table
        excluded_columns = [
            frozenset(
                column
                for source_assay in source.assay_names
                for column in (
                    assay_membership_column(source_assay),
                    *generated_cell_columns(
                        source_assay,
                        source.get_assay(source_assay)._percent_features(),
                    ),
                )
            )
            for source in datasets
        ]
        if metadata_cache is not None:
            _prefetch_metadata_schema(
                cell_tables,
                excluded_columns,
                resources,
                cache=metadata_cache,
                resident_bytes=resident,
                read_workers=schema_read_workers,
            )
        metadata_plan = plan_cell_metadata(
            cell_tables,
            names,
            prepend_text="orig",
            reset_cell_filter=True,
            source_column="dataset",
            membership={assay: membership},
            block_rows=metadata_chunk_rows(row_plan),
            scan_rows=scan_rows,
            excluded_columns=excluded_columns,
        )
        metadata_plan = admit_cell_metadata_plan(
            metadata_plan, row_plan, resources, resident_bytes=resident
        )
        source_arrays = {
            name: [
                as_zarr_array(source.matrixGroup[name], name=name) for source in assays
            ]
            for name in (("counts", "countsT") if requires_transpose else ("counts",))
        }
        width = min(metadata_chunk_rows(row_plan), max(1, resources.memoryBytes // 128))
        pair = _fit_count_layout(
            row_plan.nCells,
            alignment.nFeats,
            dtype,
            source_arrays,
            mapping_width=width,
            memory_bytes=resources.memoryBytes,
        )
        root = create_destination(str(destination))
        record: dict[str, Any] = {
            "assay": assay,
            "sources": bindings,
            "arrays": {},
            "maps": {},
            "complete": False,
        }
        root.attrs["composite"] = record
        root.attrs["cytearc:import_source"] = "composite"
        root.attrs["cytearc:import_complete"] = False
        maps = root.create_group("_composite")
        row_sources = maps.create_array(
            "rowSources", shape=(row_plan.nCells,), dtype="int32", chunks=(width,)
        )
        source_rows = maps.create_array(
            "sourceRows", shape=(row_plan.nCells,), dtype="int64", chunks=(width,)
        )
        for segment in iter_row_plan_segments(row_plan):
            destination_rows = slice(
                segment.destStart, segment.destStart + len(segment.localRows)
            )
            row_sources[destination_rows] = segment.sourceIdx
            source_rows[destination_rows] = segment.localRows
        maps.create_array(
            "featureColumns",
            data=feature_columns,
            chunks=(1, min(alignment.nFeats, width)),
        )
        record["maps"] = {
            name: fingerprint_stored_arrays(maps, [name])
            for name in ("rowSources", "sourceRows", "featureColumns")
        }
        wrapper = CompositeStore(root.store, record, source_roots, constructing=True)
        root = zarr.open_group(store=wrapper, mode="r+")
        cells = write_cell_metadata(
            root,
            None,
            row_plan,
            cell_tables,
            metadata_plan,
            profile="fast_local",
            reset_cell_filter=True,
            source_column="dataset",
            membership={assay: membership},
        )
        del cell_tables
        if metadata_cache is not None:
            metadata_cache.close()
            metadata_cache = None
        ids = as_zarr_array(cells["ids"], name="ids")
        if resident + 3 * ids.size * ids.dtype.itemsize > resources.memoryBytes:
            raise MemoryError("Composite cell identity validation exceeds mem_budget")
        values = np.asarray(ids[:])
        if np.unique(values).size != values.size:
            raise ValueError("Composite cell IDs are not unique")
        del values
        create_assay_counts(
            root,
            assay,
            None,
            row_plan.nCells,
            alignment,
            dtype,
            profile="fast_local",
            policy=pair.policy,
        )
        matrix = as_zarr_group(root[assay], name=assay)
        counts = as_zarr_array(matrix["counts"], name="counts")
        write_feature_metadata(
            [_CachedMetadataTable(source.feats) for source in assays],
            alignment.featOrderMap,
            as_zarr_group(matrix["featureData"], name="featureData"),
            alignment.nFeats,
            resources=resources,
            resident_bytes=resident,
            profile="fast_local",
        )
        counts_t = None
        if requires_transpose:
            create_count_matrix_array(matrix, "countsT", pair.countsT)
            counts_t = as_zarr_array(matrix["countsT"], name="countsT")
            persist_count_matrix_plan(counts_t, pair)
        feature_sets = default_feature_sets(matrix)
        family_bytes = len(feature_sets) * row_plan.nCells * 8
        geometries = [array_geometry(array) for array in source_arrays["counts"]]
        read_bytes = max(
            geometry.readBytes(geometry.nominalChunkBytes())
            + geometry.nominalChunkBytes()
            + 96 * sum(geometry.chunks)
            for geometry in geometries
            if geometry is not None
        )
        available = (
            resources.memoryBytes
            - resident
            - CountSummary.nbytes_for(row_plan.nCells, alignment.nFeats)
            - family_bytes
            - 96 * max(array.shape[1] for array in source_arrays["counts"])
            - max(
                source_index_bytes(
                    geometry, max_row_plan_block_rows(row_plan), geometry.shape[1]
                )
                for geometry in geometries
                if geometry is not None
            )
        )
        row_bytes = 3 * alignment.nFeats * np.dtype(dtype).itemsize + 96
        block_rows = min(
            max_row_plan_block_rows(row_plan), (available - read_bytes) // row_bytes
        )
        if block_rows < 1:
            raise MemoryError(
                "Composite count preparation cannot fit one row within mem_budget"
            )
        read_workers = min(
            resources.workers,
            (available - block_rows * row_bytes) // read_bytes,
            *(
                source.storageIo.readWorkers or resources.workers
                for source in datasets
                if source.storageIo is not None
            ),
        )
        count_arrays = wrapper.source_arrays("counts")
        count_cache = wrapper.source_cache
        if count_cache is not None:
            _prefetch_counts(
                count_arrays,
                row_plan,
                alignment.featOrderMap,
                geometries,
                segment_rows=block_rows,
                read_workers=schema_read_workers,
                memory_bytes=available - 96 * block_rows,
                cache=count_cache,
            )
        summary = CountSummary(counts)
        sums = {
            feature_sums_key(indices): (indices, np.zeros(row_plan.nCells))
            for indices in feature_sets
        }
        for segment in iter_row_plan_segments(row_plan, segment_rows=block_rows):
            source_array = count_arrays[segment.sourceIdx]
            mapping = alignment.featOrderMap[segment.sourceIdx]
            block = np.zeros((len(segment.localRows), alignment.nFeats), dtype=dtype)
            columns = np.flatnonzero(mapping >= 0)
            sync(
                copy_source_selection(
                    source_array,
                    segment.localRows,
                    columns,
                    block,
                    np.arange(len(segment.localRows)),
                    mapping[columns],
                    read_workers=read_workers,
                )
            )
            summary.update(segment.destStart, block)
            target_rows = slice(
                segment.destStart, segment.destStart + len(segment.localRows)
            )
            for indices, totals in sums.values():
                totals[target_rows] = block[:, indices].sum(axis=1, dtype=np.float64)
        fingerprint = finalize_counts(counts, summary)
        write_feature_sums(matrix, counts, sums)
        if counts_t is not None:
            counts_t.attrs.update({"complete": True, "source_fingerprint": fingerprint})
        matrix.attrs["complete"] = True
        root.attrs.update(
            {
                "assayTypes": {assay: assay_types.pop()},
                "defaultAssay": assay,
                "cytearc:import_complete": True,
            }
        )
        del summary, sums, block, alignment, feature_columns, row_plan
        result = DataStore(
            wrapper,
            default_assay=assay,
            min_features_per_cell=-1,
            mem_budget=resources.memoryBytes,
            nthreads=resources.workers,
        )
        wrapper.seal(result.z)
        return result
    except BaseException:
        if metadata_cache is not None:
            metadata_cache.close()
        if wrapper is not None:
            wrapper.close()
        raise
