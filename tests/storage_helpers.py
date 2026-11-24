"""Storage helpers for tests that build CyteArc stores directly."""

import shutil
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import zarr
from numpy.typing import DTypeLike

from cytearc.storage import async_execution
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.identity import CountSummary, finalize_counts

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore
    from cytearc.storage.artifacts import ArtifactStatus
    from cytearc.storage.refs import ArtifactRef


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


def rewrite_as_saved_by_1_4(
    store: "DataStore", ref: "ArtifactRef", *, revision: int, without: str
) -> "ArtifactStatus":
    """Rewrite an artifact's record as CyteArc 1.4.0 saved it for the same call.

    Its provenance records ``revision``, none for revision 1, its diagnostics
    lose ``without``, a diagnostic that a later revision added, and its
    ``cytearc_version`` is 1.4.0.

    Returns:
        The artifact's status as it now reads.
    """
    from cytearc.storage.artifacts import artifact_group

    group = artifact_group(store.zw, ref)
    provenance = dict(group.attrs["provenance"])
    provenance.pop("revision", None)
    if revision > 1:
        provenance["revision"] = revision
    group.attrs["provenance"] = provenance
    group.attrs["diagnostics"] = {
        name: value
        for name, value in group.attrs["diagnostics"].items()
        if name != without
    }
    group.attrs["cytearc_version"] = "1.4.0"
    return store.artifacts.inspect(ref)


@contextmanager
def recomputations() -> Iterator[list[str]]:
    """Collect the INFO lines that name an artifact computed anew for a revision."""
    from cytearc.utils.logging import logger

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        yield messages
    finally:
        logger.remove(sink)
    messages[:] = [message for message in messages if message.startswith("Recomputing")]


def recomputation(
    operation: str, ref: "ArtifactRef", revision: int, current: int, change: str
) -> str:
    """The INFO line of ``ref``, of ``revision``, that a call computes anew at ``current``, as the revision guide documents it."""
    return (
        f"Recomputing {operation}: artifact {ref.artifact_id[:12]} is revision "
        f"{revision}, current {current}: {change}"
    )


def computed_anew(
    store: "DataStore",
    saved: "ArtifactRef",
    call: Any,
    *,
    operation: str,
    revision: int,
    current: int,
    change: str,
) -> "ArtifactRef":
    """Check that ``call``, which saved ``saved`` of ``revision``, computes it anew at ``current`` and logs why.

    Returns:
        What ``call`` computes, of revision ``current``.
    """
    with recomputations() as lines:
        computed = call()
    assert computed != saved
    assert lines == [recomputation(operation, saved, revision, current, change)]
    assert store.artifacts.inspect(computed).revision == current
    return computed


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


def drop_row_nonzero(location: Any, assay: str = "RNA") -> None:
    """Remove the ``rowNonzero`` count summary of an assay at ``location``, a path or a store.

    Stores written before the summaries counted each row's nonzero values
    hold their summaries without it.
    """
    from cytearc.storage.stores import load_zarr

    matrix = load_zarr(location, mode="r+")[assay]
    del matrix["countSummaries/rowNonzero"]


def write_count_store(
    zarr_loc: str,
    counts: Mapping[str, np.ndarray],
    dtype: DTypeLike,
    *,
    feature_ids: Mapping[str, Sequence[str]] | None = None,
    cell_ids: Sequence[str] | None = None,
    feature_names: Mapping[str, Sequence[str]] | None = None,
    policy: CountMatrixPolicy | None = None,
) -> None:
    """Write a store whose count matrices have the storage dtype ``dtype``.

    ``counts`` maps assay names to cells-by-features matrices over the same
    cells. Each assay takes the preset type of its name, such as RNA, ADT,
    or ATAC, and RNA assays also get ``countsT``. Features are named
    ``f"{assay}{index}"``, or by ``feature_ids`` for the assays it names, and
    cells ``f"cell{index}"`` unless ``cell_ids`` names them. A feature's name
    is its ID unless ``feature_names`` names the features of its assay. The
    counts take the layout of ``policy``, or the default layout. The values
    must be exactly representable in ``dtype``, so the stored counts equal
    them.
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
            policy=policy,
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


def local_read_buffer(path: str | Path, *, ranged: bool) -> int:
    """The buffer that a read of the local file ``path`` holds, as Python gives it.

    A read of part of a shard of several chunks, ``ranged``, fetches a byte
    range through ``open``, whose buffer the file's size reports beside that
    of a file opened with a buffer of two bytes. A whole file, a chunk of an
    unsharded array or a shard of one chunk, is read through
    ``Path.read_bytes``, which opens it so before Python 3.14 and unbuffered
    from 3.14.
    """
    import sys

    if not ranged and sys.version_info >= (3, 14):
        return 0
    with open(path, "rb") as file, open(path, "rb", 2) as small:
        return sys.getsizeof(file) - sys.getsizeof(small) + 2


def store_as_variable_width_text(group: zarr.Group, name: str) -> None:
    """Store text column ``name`` of ``group`` again as variable-width text, in its chunks and with its attributes.

    Zarr's ``dtype=str`` stores text as variable-width UTF-8, which reads back
    as NumPy's ``StringDType``, so a store that another tool wrote can hold its
    columns that way.
    """
    array = group[name]
    group.create_array(
        name,
        data=np.asarray(array[:]).astype(np.dtypes.StringDType()),
        chunks=array.chunks,
        overwrite=True,
        attributes=array.attrs.asdict(),
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


def by_cell_id(
    cells: Any, values: Any, key: str = "I", *, rows: Any = None
) -> pd.Series | pd.DataFrame:
    """Key per-cell values in store order by cell ID, as public calls take them.

    Values for every cell take the IDs of every cell. With ``rows``, values
    for those store rows take their IDs; otherwise other values are for the
    cells that ``key`` selects, as :meth:`MetaData.insert` reads them, and
    take those cells' IDs in store order. Two-dimensional values and a
    DataFrame become a DataFrame indexed by the IDs, other values a Series.
    A Series is refused, since it already carries an index of its own.
    """
    if isinstance(values, pd.Series):
        raise TypeError("by_cell_id keys values in store order, not a Series")
    # A list reaches a call as the array that NumPy makes of it, as a list
    # passed directly did, so that None stays missing among numbers.
    if isinstance(values, list | tuple):
        values = np.asarray(values)
    ids = cells.fetch_all("ids")
    if rows is not None:
        ids = ids[np.asarray(rows)]
    elif len(values) != cells.N:
        ids = cells.fetch("ids", key)
    if isinstance(values, pd.DataFrame):
        return values.set_axis(pd.Index(ids), axis=0)
    if np.ndim(values) == 2:
        return pd.DataFrame(values, index=ids)
    return pd.Series(values, index=ids)


def graph_rows(store: Any, graph: Any) -> np.ndarray:
    """Return the store rows of a graph's cells, in store order."""
    from cytearc.graph.feature_projection import graph_cell_selection

    selection = graph_cell_selection(store.zw, graph)
    return np.flatnonzero(np.asarray(store.artifacts.load(selection)["values"][:]))


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


@dataclass
class CodecPools:
    """The threads of the pool that ran each Zarr encode and decode, under the label set when it ran.

    A call that ran on no CyteArc codec pool records None.
    """

    label: str | None = None
    calls: list[tuple[str | None, int | None]] = field(default_factory=list)
    running: threading.local = field(default_factory=threading.local)

    def threads(self, label: str | None) -> list[int | None]:
        """The pool threads of each call under ``label``, in order."""
        return [threads for of, threads in self.calls if of == label]

    def current_threads(self) -> int | None:
        """The threads of the pool that runs the calling thread's call, or None outside one."""
        return getattr(self.running, "threads", None)


@contextmanager
def recorded_codec_pools(host_threads: int) -> Iterator[CodecPools]:
    """Record the threads of the pool that runs each Blosc, Zstd or gzip encode and decode, on host pools of ``host_threads`` threads.

    Zarr runs those calls on its event loop's default executor: the decoding
    pools that ``storage.async_execution`` installs on Zarr's loop, which a
    read on a loop of its own borrows, or an ``AsyncStorageRunner``'s codec
    pool. Each pool hands the call to ``_ReleasingPool.submit``, which is
    wrapped here to mark the thread that runs it with the pool's size. Fresh
    decoding pools of ``host_threads`` threads replace Zarr's for the block,
    so a call that ignores a store's threads runs on a pool of a count that
    the test's stores do not ask for.
    """
    import zarr.core.sync as zarr_sync
    from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
    from zarr.codecs._v2 import V2Codec

    recorded = CodecPools()
    running = recorded.running
    submit = async_execution._ReleasingPool.submit

    def submitting(pool: Any, call: Any, /, *args: Any, **kwargs: Any) -> Any:
        threads = pool._max_workers

        def run(*call_args: Any, **call_kwargs: Any) -> Any:
            running.threads = threads
            try:
                return call(*call_args, **call_kwargs)
            finally:
                running.threads = None

        return submit(pool, run, *args, **kwargs)

    def recording(codec_call: Any) -> Any:
        def run(codec: Any, *args: Any, **kwargs: Any) -> Any:
            recorded.calls.append((recorded.label, recorded.current_threads()))
            return codec_call(codec, *args, **kwargs)

        return run

    pools = async_execution._DecodingPools(host_threads)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(zarr_sync._get_loop(), "_default_executor", pools)
            patch.setattr(zarr_sync, "_executor", pools)
            patch.setattr(async_execution._ReleasingPool, "submit", submitting)
            for codec in (BloscCodec, GzipCodec, ZstdCodec, V2Codec):
                for name in ("_decode_sync", "_encode_sync"):
                    patch.setattr(codec, name, recording(getattr(codec, name)))
            yield recorded
    finally:
        pools.shutdown(wait=False)


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


# The cells of each group in the files of import_grouped_cells: source a
# lists group x, then group y; source b lists group z.
GROUPED_CELLS = {"a": (("x", 150), ("y", 150)), "b": (("z", 200),)}


def import_grouped_cells(
    folder: Path,
    *,
    shuffled: Mapping[str, bool],
    seeds: Mapping[str, int] | None = None,
) -> dict[str, "DataStore"]:
    """Import sources ``a`` and ``b`` of ``GROUPED_CELLS``, shuffled or in their file's order, and open them.

    A cell's ID is its group and its row among the group's cells, such as
    ``y12``, and its counts are its row in its file, plus one, and 1. Each
    shuffle takes its seed from ``seeds``, or 1 for ``a`` and 2 for ``b``.
    """
    from scipy.sparse import csr_matrix

    from cytearc import SparseToZarr

    stores = {}
    for seed, (name, groups) in enumerate(GROUPED_CELLS.items(), start=1):
        ids = [f"{group}{row}" for group, size in groups for row in range(size)]
        counts = np.ones((len(ids), 2), dtype=np.uint32)
        counts[:, 0] = np.arange(1, len(ids) + 1)
        location = folder / f"{name}.zarr"
        SparseToZarr(
            csr_matrix(counts),
            str(location),
            cell_ids=ids,
            feature_ids=["f0", "f1"],
            assay_name="RNA",
            nthreads=1,
            shuffle_cells=shuffled[name],
            shuffle_seed=seed if seeds is None else seeds[name],
        ).dump()
        stores[name] = open_store(location)
    return stores


def grouped_file_rows(ids: Sequence[str]) -> np.ndarray:
    """Return each merged cell's row in its source's file, from IDs of ``GROUPED_CELLS`` such as ``a__y12``."""
    rows = []
    for identity in ids:
        name, cell = identity.split("__")
        offset = 0
        for group, size in GROUPED_CELLS[name]:
            if cell[0] == group:
                rows.append(offset + int(cell[1:]))
            offset += size
    return np.asarray(rows, dtype=np.int64)


def assert_random_sample_order(ids: Sequence[str]) -> None:
    """Check merged IDs of ``GROUPED_CELLS`` against the laws of a uniformly random order of the cells.

    In a uniformly random order the sources follow each other in as many
    runs as the Wald-Wolfowitz law expects, each source's cells lie in no
    relation to their rows in its file (Spearman's rho near 0), and a
    window of cells holds group x as a random sample does (hypergeometric).
    The thresholds, 4.5 standard errors and 1e-6 quantiles, keep a correct
    order passing; orders of runs of one source or of a file fail by far.
    """
    from scipy import stats

    sources = np.asarray([identity.startswith("b__") for identity in ids], dtype=int)
    n_cells = sources.size
    n_b = int(sources.sum())
    n_a = n_cells - n_b
    runs = 1 + int(np.count_nonzero(np.diff(sources)))
    mean = 1 + 2 * n_a * n_b / n_cells
    variance = 2 * n_a * n_b * (2 * n_a * n_b - n_cells) / (n_cells**2 * (n_cells - 1))
    assert abs(runs - mean) / np.sqrt(variance) < 4.5, f"{runs} runs of sources"
    file_rows = grouped_file_rows(ids)
    for source in (0, 1):
        positions = np.flatnonzero(sources == source)
        rho = stats.spearmanr(positions, file_rows[positions]).statistic
        assert abs(rho) < 4.5 / np.sqrt(positions.size - 1), f"rho = {rho}"
    in_x = np.asarray([identity.startswith("a__x") for identity in ids])
    law = stats.hypergeom(n_cells, int(in_x.sum()), 50)
    for window in (slice(0, 50), slice(n_cells - 50, n_cells)):
        count = int(in_x[window].sum())
        assert law.ppf(1e-6) <= count <= law.isf(1e-6), f"{count} cells of x"
