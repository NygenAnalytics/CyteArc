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
) -> None:
    """Write a store whose count matrices have the storage dtype ``dtype``.

    ``counts`` maps assay names to cells-by-features matrices over the same
    cells. Each assay takes the preset type of its name, such as RNA, ADT,
    or ATAC, and RNA assays also get ``countsT``. Features are named
    ``f"{assay}{index}"``, or by ``feature_ids`` for the assays it names, and
    cells ``f"cell{index}"`` unless ``cell_ids`` names them. The values must
    be exactly representable in ``dtype``, so the stored counts equal them.
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
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            len(cells),
            feat_ids=ids,
            feat_names=ids,
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

    Feature tables keep the chunks of 100,000 rows that their writers give
    them, so a read of a few guides' IDs decodes a chunk of 100,000 of them;
    in chunks of their rows, the reads hold what the rows do, and a memory
    test sees what each step of a screen operation holds besides.
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
