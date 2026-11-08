"""Storage helpers for tests that build CyteArc stores directly."""

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from numpy.typing import DTypeLike

from cytearc.storage import async_execution
from cytearc.storage.identity import CountSummary, finalize_counts

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore


def open_store(path: str | Path, **options: Any) -> "DataStore":
    """Open a store with RNA as its default assay, on one thread by default.

    No floor of features per cell applies, so every cell of a small test
    store is measured.
    """
    # Imported here, so that the domain tests do not import the datastore.
    from cytearc.datastore.datastore import DataStore

    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        **{"nthreads": 1, **options},
    )


def copy_store(template: Path, folder: Path, **options: Any) -> "DataStore":
    """Copy a template store into ``folder`` and open the copy."""
    target = folder / template.name
    shutil.copytree(template, target)
    return open_store(target, **options)


def written(store: "DataStore") -> tuple[Any, ...]:
    """Return what a call can write: run records, and every artifact of the store."""
    return (
        [run.run_id for run in store.pipeline.list_runs(limit=1000)],
        store.artifacts.list(scope="datastore"),
        *(store.artifacts.list(from_assay=assay) for assay in store.assay_names),
    )


def finalize_test_counts(counts: zarr.Array) -> str:
    """Summarize values already written to ``counts`` and finalize them.

    Writers fill the summary while they write. Tests that assign counts
    directly summarize the stored values here instead.
    """
    summary = CountSummary(counts)
    rows = max(1, int(counts.chunks[0]))
    for start in range(0, int(counts.shape[0]), rows):
        summary.update(start, counts[start : start + rows])
    return finalize_counts(counts, summary=summary)


def write_count_store(
    zarr_loc: str,
    counts: Mapping[str, np.ndarray],
    dtype: DTypeLike,
    *,
    feature_ids: Mapping[str, Sequence[str]] | None = None,
    cell_ids: Sequence[str] | None = None,
    feature_names: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Write a store whose count matrices have the storage dtype ``dtype``.

    ``counts`` maps assay names to cells-by-features matrices over the same
    cells. Each assay takes the preset type of its name, such as RNA, ADT,
    or ATAC, and RNA assays also get ``countsT``. Features are named
    ``f"{assay}{index}"``, or by ``feature_ids`` for the assays it names, and
    cells ``f"cell{index}"`` unless ``cell_ids`` names them. A feature's name
    is its ID unless ``feature_names`` names the features of its assay. The
    values must be exactly representable in ``dtype``, so the stored counts
    equal them.
    """
    from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
    from cytearc.storage.stores import load_zarr
    from cytearc.writers.counts_t import finalize_writer_counts_t

    storage_dtype = np.dtype(dtype)
    root = load_zarr(zarr_loc=zarr_loc, mode="w")
    n_cells = {len(values) for values in counts.values()}
    if len(n_cells) != 1:
        raise ValueError("Every assay must have the same cells")
    n_rows = n_cells.pop()
    if cell_ids is None:
        cells = np.array([f"cell{index}" for index in range(n_rows)])
    else:
        cells = np.asarray(list(cell_ids), dtype=str)
        if len(cells) != n_rows:
            raise ValueError("cell_ids must name every cell")
    create_cell_data(root, None, ids=cells, names=cells)
    for assay, values in counts.items():
        stored = np.asarray(values).astype(storage_dtype)
        if not np.array_equal(stored, values):
            raise ValueError(f"{assay} counts do not fit {storage_dtype}")
        named = (feature_ids or {}).get(assay)
        ids = np.array(
            [f"{assay}{index}" for index in range(stored.shape[1])]
            if named is None
            else list(named)
        )
        names = (feature_names or {}).get(assay)
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            len(cells),
            feat_ids=ids,
            feat_names=ids if names is None else np.array(list(names)),
            dtype=storage_dtype.name,
        )
        array[:] = stored
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)


def rechunk_column(group: zarr.Group, name: str, rows: int = 100_000) -> None:
    """Store column ``name`` of ``group`` again in chunks of ``rows`` rows, with its attributes.

    Stores written before cell columns were chunked by their rows and widths
    keep them in chunks of 100,000 rows, which this writes for a test of how
    such a store is read.
    """
    from cytearc.storage.arrays import create_metadata_column

    array = group[name]
    values, attributes = np.asarray(array[:]), dict(array.attrs)
    create_metadata_column(
        group,
        name,
        data=values,
        dtype=values.dtype,
        chunkSize=rows,
        overwrite=True,
        attributes=attributes,
    )


def rechunk_feature_tables(location: str | Path) -> None:
    """Store every column of each assay's feature table at ``location`` again in one chunk of its rows.

    Stores written before feature tables were chunked by their rows keep
    chunks of 100,000 rows, so a read of a few guides' IDs decodes a chunk of
    100,000 of them; in chunks of their rows, as a new store holds them, the
    reads hold what the rows do, and a memory test sees what each step of an
    operation holds besides.
    """
    root = zarr.open_group(str(location), mode="r+")
    for assay in root.group_keys():
        if "featureData" not in root[assay]:
            continue
        table = root[assay]["featureData"]
        for name in list(table.array_keys()):
            rechunk_column(table, name, rows=max(1, int(table[name].shape[0])))


def insert_nullable_cell_column(
    datastore: Any,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
) -> None:
    """Add a cell metadata column whose ``missing`` rows carry no value."""
    cell_data = datastore.zw["cellData"]
    missing_name = f"__cytearc_missing__{name}"
    cell_data.create_array(name, data=np.asarray(values))
    cell_data.create_array(missing_name, data=np.asarray(missing, dtype=bool))
    cell_data[name].attrs["missing_mask"] = missing_name


def reset_zarr_runtime() -> None:
    """Restore Zarr's process defaults so the next storage call plans afresh.

    Values reused while stored bytes are unchanged are forgotten too, so each
    test computes them as a new process would.
    """
    from cytearc.storage.content_stamps import clear_content_cache

    zarr.config.set({"threading.max_workers": None, "async.concurrency": 10})
    async_execution._HOST_THREAD_CEILING = None
    clear_content_cache()


def record_normalized_values(
    zarr_loc: str | Path,
    assay: str,
    imported_as: str | None = "log_normalized",
) -> None:
    """Record an assay of a legacy-layout store as holding normalized values.

    Steps that need counts read only this record, so a store of counts that
    carries it tests their refusals on the same data that their own tests use.
    """
    group = zarr.open_group(str(zarr_loc), mode="r+")[assay]
    group.attrs["normalizedValues"] = {"importedAs": imported_as, "cellTotal": None}


def serve_uris(
    monkeypatch: Any,
    served: Mapping[str, tuple[Any, dict[str, Any] | None]],
) -> None:
    """Serve each URI from a store that opens only with the options it expects.

    ``served`` maps a URI to its store, a local path or a Zarr store, and the
    options that open it; a store opened with any other options raises
    ``PermissionError`` that names the URI. Every other location opens as usual.
    """
    import cytearc.storage.stores as stores_module
    from zarr.storage import LocalStore

    make = stores_module.make_store

    def serve(
        location: Any,
        *,
        storage_options: dict[str, Any] | None = None,
        read_only: bool = False,
    ) -> Any:
        if not isinstance(location, str) or location not in served:
            return make(location, storage_options=storage_options, read_only=read_only)
        store, expected = served[location]
        if storage_options != expected:
            raise PermissionError(
                f"{location} refuses the options {sorted(storage_options or {})}"
            )
        if isinstance(store, Path):
            return LocalStore(store, read_only=read_only)
        return store.with_read_only(read_only)

    monkeypatch.setattr(stores_module, "make_store", serve)


def object_store_copy(directory: Path) -> Any:
    """Return an in-memory object store that holds the files of a local store."""
    import obstore
    from obstore.store import MemoryStore as ObjectMemoryStore
    from zarr.storage import ObjectStore

    memory = ObjectMemoryStore()
    for file in sorted(directory.rglob("*")):
        if file.is_file():
            obstore.put(
                memory, file.relative_to(directory).as_posix(), file.read_bytes()
            )
    return ObjectStore(store=memory)


def stored_bytes(store: Any) -> dict[str, bytes]:
    """Return every key of a Zarr store with the bytes it holds."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    async def read() -> dict[str, bytes]:
        values = {}
        async for key in store.list():
            value = await store.get(key, default_buffer_prototype())
            assert value is not None
            values[key] = value.to_bytes()
        return values

    return sync(read())
