import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import numpy as np
import zarr
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
from ..merge.row_plan import build_row_plan, iter_row_plan_segments
from ..merge.writer import create_assay_counts
from ..metadata.membership import resolve_assay_membership
from ..storage.artifacts import fingerprint_stored_arrays
from ..storage.budget import resolve_budget
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
from ..storage.stores import MATRIX_SOURCE_ATTR, locations_overlap
from ..storage.types import as_zarr_array, as_zarr_group

if TYPE_CHECKING:
    from ..datastore.datastore import DataStore


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
    from .store import _read_scratch_bytes

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
    sources = {}
    for name, arrays in source_arrays.items():
        geometries = []
        for array in arrays:
            geometry = array_geometry(array)
            assert geometry is not None
            geometries.append(geometry)
        sources[name] = geometries
    itemsize = np.dtype(dtype).itemsize
    policy = DEFAULT_COUNT_MATRIX_POLICY
    while True:
        pair = plan_count_matrix_pair(
            n_cells, n_features, dtype, policy=policy, profile="fast_local"
        )
        peak = 0
        for name, spec in (("counts", pair.counts), ("countsT", pair.countsT)):
            if name not in sources:
                continue
            geometry = ArrayGeometry(spec.shape, spec.chunks, spec.shards, itemsize)
            geometry = replace(
                geometry,
                readScratchBytes=_read_scratch_bytes(
                    geometry, mappings, sources[name], transposed=name == "countsT"
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
    from .store import CompositeStore, _copy_source

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
    cell_tables = [source.cells for source in datasets]
    scan_rows = resolve_metadata_schema_scan_rows(
        cell_tables,
        resources,
        resident_bytes=resident,
        preferred_rows=max(
            min(table.N, table._get_array("ids").chunks[0]) for table in cell_tables
        ),
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
        excluded_columns=[
            frozenset(
                column
                for source_assay in source.assay_names
                for column in (
                    assay_membership_column(source_assay),
                    *generated_cell_columns(
                        source_assay, source.get_assay(source_assay)._percent_features()
                    ),
                )
            )
            for source in datasets
        ],
    )
    metadata_plan = admit_cell_metadata_plan(
        metadata_plan, row_plan, resources, resident_bytes=resident
    )
    source_arrays = {
        name: [as_zarr_array(source.matrixGroup[name], name=name) for source in assays]
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
        "featureColumns", data=feature_columns, chunks=(1, min(alignment.nFeats, width))
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
        [source.feats for source in assays],
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
    decode_bytes = max(
        geometry.readBytes(geometry.nominalChunkBytes())
        for geometry in geometries
        if geometry is not None
    )
    available = (
        resources.memoryBytes
        - resident
        - CountSummary.nbytes_for(row_plan.nCells, alignment.nFeats)
        - family_bytes
        - decode_bytes
        - 96 * max(array.shape[1] for array in source_arrays["counts"])
    )
    block_rows = available // (3 * alignment.nFeats * np.dtype(dtype).itemsize + 96)
    if block_rows < 1:
        raise MemoryError(
            "Composite count preparation cannot fit one row within mem_budget"
        )
    summary = CountSummary(counts)
    sums = {
        feature_sums_key(indices): (indices, np.zeros(row_plan.nCells))
        for indices in feature_sets
    }
    for segment in iter_row_plan_segments(row_plan, segment_rows=block_rows):
        source_array = source_arrays["counts"][segment.sourceIdx]
        mapping = alignment.featOrderMap[segment.sourceIdx]
        block = np.zeros((len(segment.localRows), alignment.nFeats), dtype=dtype)
        columns = np.flatnonzero(mapping >= 0)
        sync(
            _copy_source(
                source_array,
                segment.localRows,
                columns,
                block,
                np.arange(len(segment.localRows)),
                mapping[columns],
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
