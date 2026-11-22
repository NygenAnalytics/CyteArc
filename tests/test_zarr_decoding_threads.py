"""Reads that a memory ledger charges, and every public call of a store, decode on as many threads of Zarr's codec pool as the ledger or the store counts, on any machine."""

import contextvars
import shutil
import threading
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zarr
import zarr.core.sync as zarr_sync
from zarr.codecs import BloscCodec, ZstdCodec

import cytearc.plotting as plotting
from cytearc import DataStore, to_h5ad, to_mtx
from cytearc.metadata.rows import read_rows, read_whole
from cytearc.storage import async_execution
from cytearc.storage.async_execution import (
    ensure_zarr_host_ceiling,
    zarr_decoding_threads,
    zarr_io_concurrency,
)
from zarr.core.codec_pipeline import BatchedCodecPipeline
from cytearc.utils.memory_budget import MemoryLedger
from tests.storage_helpers import recorded_codec_pools

#: Which reader a decode belongs to, set in each reading thread.
_READER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "reader", default=None
)

#: The host threads of the pools that each test reads beside: a count that
#: no test reads at, so that every count a test reads at has a pool of its
#: own whatever the machine.
_HOST_THREADS = 7


@pytest.fixture(autouse=True)
def _decoding_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[async_execution._DecodingPools]:
    """Make fresh decoding pools of seven host threads Zarr's, and put the process's back after the test."""
    pools = async_execution._DecodingPools(_HOST_THREADS)
    monkeypatch.setattr(zarr_sync._get_loop(), "_default_executor", pools)
    monkeypatch.setattr(zarr_sync, "_executor", pools)
    yield pools
    pools.shutdown(wait=False)


class _Decodes:
    """The thread that ran each zstd or Blosc decode, with the reader it decoded for.

    The first ``together`` decodes wait for each other, so that many threads
    decode at once, and with ``each_reader`` the first decode of each of
    those readers waits for the first of the others, so their reads are in
    flight together.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        together: int = 0,
        each_reader: tuple[str, ...] = (),
    ) -> None:
        self.seen: list[tuple[threading.Thread, str | None]] = []
        lock = threading.Lock()
        waiting = together or len(each_reader)
        barrier = threading.Barrier(waiting, timeout=20) if waiting else None
        started: set[str | None] = set()
        # Decodes since recording began, which clearing ``seen`` keeps.
        decoded = [0]

        def record() -> None:
            reader = _READER.get()
            with lock:
                self.seen.append((threading.current_thread(), reader))
                decoded[0] += 1
                if each_reader:
                    wait = reader in each_reader and reader not in started
                    started.add(reader)
                else:
                    wait = decoded[0] <= together
            if barrier is not None and wait:
                barrier.wait()

        def recording(decode):
            def run(codec, chunk_bytes, chunk_spec):
                record()
                return decode(codec, chunk_bytes, chunk_spec)

            return run

        for codec in (ZstdCodec, BloscCodec):
            monkeypatch.setattr(codec, "_decode_sync", recording(codec._decode_sync))

    def threads(self, reader: str | None = None) -> set[threading.Thread]:
        return {thread for thread, of in self.seen if reader is None or of == reader}


def _values(root: zarr.Group, n_chunks: int = 48) -> np.ndarray:
    values = np.arange(1_000 * n_chunks, dtype=np.int64)
    root.create_array(
        "values", data=values, chunks=(1_000,), compressors=ZstdCodec(level=1)
    )
    return values


def _read_whole(root: zarr.Group, threads: int) -> np.ndarray:
    ledger = MemoryLedger(None, "Reading", codec_threads=threads)
    return read_whole(ledger, root, "values", item="the values")


def _read_rows(root: zarr.Group, threads: int) -> np.ndarray:
    ledger = MemoryLedger(None, "Reading", codec_threads=threads)
    rows = np.arange(0, root["values"].shape[0], 7)
    return read_rows(ledger, root["values"], rows, item="the rows")


@pytest.mark.parametrize("read", [_read_whole, _read_rows], ids=["whole", "rows"])
def test_a_read_decodes_on_its_ledgers_threads_after_a_wider_read(
    tmp_path, monkeypatch, read
) -> None:
    # Forty-eight chunks read whole ten at a time, charged to six codec
    # threads, decode on six at once; each read after it decodes on no more
    # threads than its ledger counts, and gives the same values.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    expected = values if read is _read_whole else values[::7]
    decodes = _Decodes(monkeypatch, together=6)
    with zarr_io_concurrency(10):
        np.testing.assert_array_equal(_read_whole(root, 6), values)
        assert len(decodes.threads()) == 6
        for threads in (1, 2, 3, 1):
            decodes.seen.clear()
            np.testing.assert_array_equal(read(root, threads), expected)
            assert 1 <= len(decodes.threads()) <= threads, decodes.threads()
            assert all(f"_of_{threads}_" in t.name for t in decodes.threads())


def test_a_read_at_the_host_threads_decodes_on_zarrs_pool_as_any_other_read(
    tmp_path, monkeypatch
) -> None:
    # The default nthreads is the host's, whose reads decode on the threads
    # that every other read of the process decodes on.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    host = ensure_zarr_host_ceiling()
    assert host == _HOST_THREADS
    decodes = _Decodes(monkeypatch)
    with zarr_io_concurrency(10):
        np.testing.assert_array_equal(_read_whole(root, host), values)
        np.testing.assert_array_equal(root["values"][...], values)
    pool = zarr_sync._get_loop()._default_executor
    assert decodes.threads() <= set(pool._threads)
    assert not any("_of_" in thread.name for thread in decodes.threads())


@pytest.mark.parametrize(
    "counts",
    [(1, 3), (2, 2), (3, 1, 3)],
    ids=["one-beside-three", "two-beside-two", "three-one-three"],
)
def test_reads_in_flight_together_decode_on_the_pool_of_their_own_count(
    tmp_path, monkeypatch, counts
) -> None:
    # Each reader's first decode waits for the first of the others, so the
    # reads are in flight together. Each decodes on no more threads than its
    # ledger counts; readers of one count share one pool of that many
    # threads, and readers of different counts share no thread.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    readers = tuple(f"reader {index}" for index in range(len(counts)))
    decodes = _Decodes(monkeypatch, each_reader=readers)
    results: dict[str, np.ndarray] = {}

    def reading(reader: str, threads: int) -> None:
        _READER.set(reader)
        with zarr_io_concurrency(4):
            results[reader] = _read_whole(root, threads)

    callers = [
        threading.Thread(target=reading, args=(reader, threads))
        for reader, threads in zip(readers, counts, strict=True)
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=60)
    assert sorted(results) == sorted(readers)
    for reader in readers:
        np.testing.assert_array_equal(results[reader], values)
    by_count: dict[int, set[threading.Thread]] = {}
    for reader, threads in zip(readers, counts, strict=True):
        assert 1 <= len(decodes.threads(reader)) <= threads
        by_count.setdefault(threads, set()).update(decodes.threads(reader))
    for threads, decoding in by_count.items():
        assert len(decoding) <= threads
    pools = list(by_count.values())
    assert all(a.isdisjoint(b) for i, a in enumerate(pools) for b in pools[i + 1 :])


def test_a_pool_that_replaced_cytearcs_is_replaced_before_the_next_charged_read(
    tmp_path, monkeypatch, _decoding_pools
) -> None:
    # A pool set on Zarr's loop after CyteArc's, as a library or Zarr itself
    # can set one, gives way to CyteArc's when a charged read begins.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    ensure_zarr_host_ceiling()
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(loop, "_default_executor", loop._default_executor)
    decodes = _Decodes(monkeypatch)
    with ThreadPoolExecutor(8, thread_name_prefix="foreign") as foreign:
        loop.set_default_executor(foreign)
        with zarr_io_concurrency(10):
            np.testing.assert_array_equal(_read_whole(root, 1), values)
        assert len(decodes.threads()) == 1
        assert not any(t.name.startswith("foreign") for t in decodes.threads())
        assert loop._default_executor is _decoding_pools


def test_the_installed_pool_keeps_the_size_of_the_pool_zarr_ran_before(
    monkeypatch,
) -> None:
    # A pool that Zarr's loop ran before CyteArc's first use sets the host
    # threads, and CyteArc's pools take its place with as many.
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(loop, "_default_executor", None)
    monkeypatch.setattr(zarr_sync, "_executor", None)
    with ThreadPoolExecutor(5) as earlier:
        loop.set_default_executor(earlier)
        assert ensure_zarr_host_ceiling() == 5
        pools = loop._default_executor
        assert isinstance(pools, async_execution._DecodingPools)
        assert pools._max_workers == 5
        assert zarr_sync._executor is pools
        assert ensure_zarr_host_ceiling() == 5
        assert loop._default_executor is pools
        pools.shutdown()


def test_the_pools_on_zarrs_loop_become_zarrs_pool_when_zarr_holds_none(
    monkeypatch, _decoding_pools
) -> None:
    # Zarr holds no pool of its own while CyteArc's pools run on its loop.
    loop = zarr_sync._get_loop()
    monkeypatch.setattr(zarr_sync, "_executor", None)
    assert loop._default_executor is _decoding_pools

    assert ensure_zarr_host_ceiling() == _HOST_THREADS

    assert zarr_sync._executor is _decoding_pools
    # So Zarr makes no pool of its own to replace them on the loop.
    assert zarr_sync._get_executor() is _decoding_pools
    assert loop._default_executor is _decoding_pools


def test_a_reinstall_finds_the_pools_it_made_beside_zarrs_own(monkeypatch) -> None:
    # Where Zarr made its own pool first, the pools that take its place on
    # the loop become Zarr's too, so that once another executor replaces
    # them, the next install finds the same pools instead of making more.
    loop = zarr_sync._get_loop()
    with (
        ThreadPoolExecutor(3) as zarrs,
        ThreadPoolExecutor(3) as foreign,
        ThreadPoolExecutor(3) as later,
    ):
        monkeypatch.setattr(zarr_sync, "_executor", zarrs)
        loop.set_default_executor(foreign)
        assert ensure_zarr_host_ceiling() == 3
        pools = loop._default_executor
        assert isinstance(pools, async_execution._DecodingPools)
        assert zarr_sync._executor is pools
        loop.set_default_executor(later)
        assert ensure_zarr_host_ceiling() == 3
        assert loop._default_executor is pools
        pools.shutdown()


def test_the_repertoire_operations_decode_their_own_charged_reads_on_the_stores_threads(
    tmp_path, monkeypatch
) -> None:
    # The reads that the repertoire operations charge to a ledger outside
    # read_whole and read_rows, of the contigs' text and of the export's
    # passes, decode on the two threads of a store of nthreads=2.
    from cytearc.datastore._operations import repertoire
    from tests.clone_helpers import (
        CELLRANGER_HEADER,
        CLONAL_ROWS,
        receptor_store,
        write_lines,
    )

    def labeled(function, label):
        def run(*args, **kwargs):
            token = _READER.set(label)
            try:
                return function(*args, **kwargs)
            finally:
                _READER.reset(token)

        return run

    for name in ("_decoded_values", "_value_lengths"):
        monkeypatch.setattr(repertoire, name, labeled(getattr(repertoire, name), name))
    for name in ("measure", "scan"):
        method = getattr(repertoire._ExportContigs, name)
        monkeypatch.setattr(repertoire._ExportContigs, name, labeled(method, name))
    store = receptor_store(tmp_path, nthreads=2)
    path = write_lines(tmp_path, "clonal.csv", CELLRANGER_HEADER, CLONAL_ROWS)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    decodes = _Decodes(monkeypatch)
    clonotypes = store.repertoire.clonotypes(receptors, sequence="aa")
    store.repertoire.clonotypes(receptors, metric="hamming", max_distance=1)
    store.repertoire.spectratype(receptors)
    store.repertoire.export_airr(clonotypes, tmp_path / "contigs.tsv")
    for label in ("_decoded_values", "_value_lengths", "measure", "scan"):
        decoding = {thread.name for thread in decodes.threads(label)}
        assert decoding, label
        assert all("_of_2_" in name for name in decoding), (label, decoding)


def test_nested_scopes_decode_on_the_innermost_count_and_restore_the_outer(
    tmp_path, monkeypatch
) -> None:
    # A scope left by an error restores the count of the scope around it,
    # and leaving that restores the host threads.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = _values(root)
    decodes = _Decodes(monkeypatch)

    def decoded_on() -> set[str]:
        decodes.seen.clear()
        np.testing.assert_array_equal(root["values"][...], values)
        return {thread.name for thread in decodes.threads()}

    with zarr_io_concurrency(10), zarr_decoding_threads(3):
        with pytest.raises(KeyError), zarr_decoding_threads(1):
            inner = decoded_on()
            raise KeyError("left early")
        outer = decoded_on()
    outside = decoded_on()
    assert inner and all("_of_1_" in name for name in inner)
    assert outer and all("_of_3_" in name for name in outer)
    assert outside and not any("_of_" in name for name in outside)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2", None])
def test_decoding_threads_are_a_whole_number_of_at_least_one(threads) -> None:
    with pytest.raises(ValueError, match="whole number of threads of at least 1"):
        with zarr_decoding_threads(threads):
            pytest.fail("the scope must not open")


def test_shutting_the_pools_down_shuts_down_the_pool_of_every_count() -> None:
    pools = async_execution._DecodingPools(2)
    for threads in (1, 3):
        with zarr_decoding_threads(threads):
            assert (
                pools.submit(threading.current_thread)
                .result(5)
                .name.startswith(f"zarr_pool_of_{threads}_")
            )
    assert (
        pools.submit(threading.current_thread).result(5).name.startswith("zarr_pool_")
    )
    pools.shutdown()
    for threads in (None, 1, 3, 5):
        token = async_execution._DECODING_THREADS.set(threads)
        try:
            with pytest.raises(RuntimeError, match="after shutdown"):
                pools.submit(threading.current_thread)
        finally:
            async_execution._DECODING_THREADS.reset(token)


# The public calls of a store decode on the store's threads.


def _copied_store(source: str, folder: Path, *, nthreads: int) -> Any:
    location = folder / f"threads_{nthreads}.zarr"
    shutil.copytree(source, location)
    return DataStore(str(location), default_assay="RNA", nthreads=nthreads)


@pytest.fixture(scope="module")
def one_thread_analysis(datastore_zarr_root, tmp_path_factory) -> dict[str, Any]:
    """A copy of the PBMC fixture opened with nthreads=1, with a graph and its clusters."""
    from cytearc.utils.warnings import DiagnosticWarning

    store = _copied_store(
        datastore_zarr_root, tmp_path_factory.mktemp("one_thread"), nthreads=1
    )
    cells = store.snapshot_cell_selection("I")
    hvgs = store.features.hvgs(
        cells, top_n=50, min_cells=0, max_cells=np.inf, show_plot=False
    )
    normalized = store.features.normalize(cells, hvgs)
    pca = store.reduction.pca(normalized, dims=4)
    ann = store.graph.ann_index(pca, ann_efc=40, ann_ef=40, ann_m=8)
    with warnings.catch_warnings():
        # The small index misses neighbors; the warning test checks the warning.
        warnings.simplefilter("ignore", DiagnosticWarning)
        neighbors = store.graph.neighbors(ann, k=5)
    graph = store.graph.connectivity(neighbors)
    leiden = store.clusters.leiden(graph, resolution=0.5)
    return {
        "store": store,
        "location": str(store.zarr_loc),
        "source": datastore_zarr_root,
        "cells": cells,
        "normalized": normalized,
        "pca": pca,
        "ann": ann,
        "graph": graph,
        "leiden": leiden,
    }


# Each call reads data that the fixture's calls left unread or computes a
# result that they did not, so that each one decodes chunks.
_ONE_THREAD_CALLS = (
    "opening the store",
    "snapshot_cell_selection",
    "cells.fetch",
    "cells.to_pandas_dataframe",
    "cells.iter_row_blocks",
    "RNA.feats.fetch_all",
    "qc.auto_filter",
    "features.hvgs",
    "reduction.pca",
    "embeddings.initialization",
    "graph.load",
    "clusters.leiden",
    "artifacts.load_values",
    "to_anndata",
    "to_h5ad",
    "to_mtx",
    "plots.distribution",
    "cytearc.plotting.distribution",
    "RNA.rawData compute",
    "RNA.normed compute",
    "RNA.score_features",
    "artifacts.load group read",
    "a caller's own open of the store",
)


def _call(name: str, analysis: dict[str, Any], folder: Path) -> None:
    store = analysis["store"]
    match name:
        case "opening the store":
            # The first writable open of a store applies its cell filter.
            _copied_store(analysis["source"], folder, nthreads=1)
        case "snapshot_cell_selection":
            store.snapshot_cell_selection("I")
        case "cells.fetch":
            store.cells.fetch("RNA_nCounts")
        case "cells.to_pandas_dataframe":
            store.cells.to_pandas_dataframe(["RNA_nFeatures", "RNA_percentMito"])
        case "cells.iter_row_blocks":
            list(store.cells.iter_row_blocks(columns=["RNA_percentRibo"]))
        case "RNA.feats.fetch_all":
            store.RNA.feats.fetch_all("names")
        case "qc.auto_filter":
            store.qc.auto_filter(method="gaussian")
        case "features.hvgs":
            store.features.hvgs(
                analysis["cells"],
                top_n=40,
                min_cells=0,
                max_cells=np.inf,
                show_plot=False,
            )
        case "reduction.pca":
            store.reduction.pca(analysis["normalized"], dims=3)
        case "embeddings.initialization":
            store.embeddings.initialization(
                analysis["pca"], n_centroids=50, rand_state=7
            )
        case "graph.load":
            store.graph.load(analysis["graph"])
        case "clusters.leiden":
            store.clusters.leiden(analysis["graph"], resolution=0.7)
        case "artifacts.load_values":
            store.artifacts.load_values(analysis["leiden"])
        case "to_anndata":
            store.to_anndata(
                cell_selection=analysis["cells"], obs={"leiden": analysis["leiden"]}
            )
        case "to_h5ad":
            to_h5ad(store.RNA, str(folder / "export.h5ad"))
        case "to_mtx":
            to_mtx(store.RNA, str(folder / "export_mtx"))
        case "plots.distribution":
            store.plots.distribution(
                "RNA_nCounts", grouping=analysis["leiden"], show=False
            ).close()
        case "cytearc.plotting.distribution":
            plotting.distribution(store, "RNA_nFeatures", show=False).close()
        case "RNA.rawData compute":
            store.RNA.rawData[:300].compute()
        case "RNA.normed compute":
            store.RNA.normed()[:300].compute()
        case "RNA.score_features":
            store.RNA.score_features(["CD3D", "LYZ", "CD79A"], "I", 20, 5, 7)
        case "artifacts.load group read":
            store.artifacts.load(analysis["cells"])["values"][:]
        case "a caller's own open of the store":
            # It equals the store's own, so it decodes on the store's threads.
            zarr.open_group(str(store.zarr_loc), mode="r")["cellData/ids"][:]


@pytest.mark.slow
@pytest.mark.parametrize("call", list(_ONE_THREAD_CALLS))
def test_each_call_of_a_one_thread_store_decodes_on_one_thread(
    one_thread_analysis, tmp_path, call
) -> None:
    # Opening the store, its own methods, an operation of each accessor
    # family that the fixture holds data for, its cell and feature tables,
    # its exports, its plots, the arrays it hands out and a caller's own open
    # of its location each decode and encode on one thread, not on the
    # host's seven. Before, all but the reads that a memory ledger charges
    # decoded on the host's threads.
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = call
        _call(call, one_thread_analysis, tmp_path)
        pools.label = None
    assert pools.threads(call), f"{call} decoded no chunk"
    assert set(pools.threads(call)) == {1}, pools.threads(call)
    assert pools.threads(None) == []


@pytest.mark.slow
def test_a_store_of_the_hosts_threads_decodes_on_zarrs_pool(
    datastore_zarr_root, tmp_path
) -> None:
    # A store whose nthreads equal the host threads, as a store opened with
    # CYTEARC_WORKERS unset has, opens, operates, plots and exports on Zarr's
    # pool itself, as before, and makes no pool of its own count. The storage
    # runner of a write still encodes on a pool of its own, as before.
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        host = zarr_sync._executor
        store = _copied_store(datastore_zarr_root, tmp_path, nthreads=_HOST_THREADS)
        cells = store.snapshot_cell_selection("I")
        store.cells.fetch("RNA_nCounts")
        hvgs = store.features.hvgs(
            cells, top_n=20, min_cells=0, max_cells=np.inf, show_plot=False
        )
        store.features.normalize(cells, hvgs)
        store.plots.distribution("RNA_nCounts", show=False).close()
        to_mtx(store.RNA, str(tmp_path / "export_mtx"))
        sized = dict(host._sized)
    assert pools.threads(None).count(_HOST_THREADS) > 40
    assert sized == {}


def test_a_decode_error_reaches_the_caller_and_the_next_read_decodes_as_before(
    tmp_path,
) -> None:
    # A chunk whose stored bytes are not zstd fails to decode on the store's
    # one thread; the codec's error reaches the caller through CyteArc's codec
    # pipeline unchanged, the pipeline leaves no threads behind in the
    # caller's context, and the next read decodes on the store's thread.
    import traceback

    root, values = _registered_values(tmp_path, 1)
    (tmp_path / "registered.zarr" / "values" / "c" / "1").write_bytes(b"not zstd" * 8)
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "corrupt"
        with pytest.raises(RuntimeError, match="Zstd decompression error") as failure:
            root["values"][1_000:2_000]
        pools.label = "next"
        np.testing.assert_array_equal(root["values"][2_000:3_000], values[2_000:3_000])
    frames = [
        (Path(frame.filename).name, frame.name)
        for frame in traceback.extract_tb(failure.value.__traceback__)
    ]
    assert ("async_execution.py", "read") in frames, frames
    assert pools.threads("corrupt") == [1]
    assert pools.threads("next") == [1]
    assert async_execution._DECODING_THREADS.get() is None


def test_a_read_leaves_its_threads_to_no_later_read_of_its_task(tmp_path) -> None:
    # One coroutine reads a store registered with one thread, then a store
    # that no one registered: the first read's threads end with it, so the
    # second decodes on the host's threads.
    registered, values = _registered_values(tmp_path, 1)
    other = zarr.open_group(str(tmp_path / "other.zarr"), mode="w")
    other_values = _values(other, n_chunks=3)

    arrays = (registered["values"].async_array, other["values"].async_array)

    async def read_both() -> tuple[Any, Any]:
        first = await arrays[0].getitem(slice(None))
        second = await arrays[1].getitem(slice(None))
        return first, second

    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "both"
        first, second = zarr_sync.sync(read_both())
    np.testing.assert_array_equal(first, values)
    np.testing.assert_array_equal(second, other_values)
    assert pools.threads("both") == [1] * 4 + [_HOST_THREADS] * 3


@pytest.mark.parametrize("any_loop", [True, False], ids=["own loop", "zarr's loop"])
def test_an_array_read_decodes_the_chunks_it_reads_on_the_arrays_threads(
    tmp_path, monkeypatch, any_loop
) -> None:
    # A composite's reads read its sources' chunks through stores that no
    # DataStore of the composite registered. run_array_read gives them the
    # threads of the array it reads, on the read's own event loop and on
    # Zarr's alike: here an unregistered store's chunks, read for an array of
    # a store registered with one thread.
    from cytearc.storage import virtual_arrays

    registered, _ = _registered_values(tmp_path, 1)
    other = zarr.open_group(str(tmp_path / "other.zarr"), mode="w")
    values = _values(other, n_chunks=3)
    monkeypatch.setattr(virtual_arrays, "_reads_on_any_loop", lambda *_: any_loop)
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "read"
        result = virtual_arrays.run_array_read(
            registered["values"], other["values"].async_array.getitem(slice(None))
        )
    np.testing.assert_array_equal(result, values)
    assert pools.threads("read") == [1] * 3


@pytest.mark.parametrize("workspace", [None, "analysis"])
def test_the_feature_and_cell_tables_of_a_one_thread_store_read_on_one_thread(
    tmp_path, workspace
) -> None:
    # The tables of a store with a workspace, whose features live below it,
    # read on the store's thread as those of a store without one do.
    from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
    from cytearc.storage.stores import load_zarr
    from cytearc.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    location = str(tmp_path / "tables.zarr")
    root = load_zarr(zarr_loc=location, mode="w")
    counts = np.random.default_rng(5).poisson(4.0, size=(30, 6)).astype(np.uint32) + 1
    cell_ids = np.asarray([f"cell{index}" for index in range(30)])
    feature_ids = np.asarray([f"gene{index}" for index in range(6)])
    create_cell_data(root, workspace, ids=cell_ids, names=cell_ids)
    array = create_zarr_count_assay(
        root,
        "RNA",
        workspace,
        30,
        feat_ids=feature_ids,
        feat_names=feature_ids,
        dtype="uint32",
    )
    array[:] = counts
    finalize_test_counts(array)
    finalize_writer_counts_t(root, "RNA", workspace)
    store = DataStore(
        location,
        default_assay="RNA",
        workspace=workspace,
        min_features_per_cell=-1,
        nthreads=1,
    )
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "features"
        np.testing.assert_array_equal(store.RNA.feats.fetch_all("ids"), feature_ids)
        pools.label = "cells"
        np.testing.assert_array_equal(store.cells.fetch_all("ids"), cell_ids)
    for label in ("features", "cells"):
        assert pools.threads(label) and set(pools.threads(label)) == {1}, label


def test_stores_of_different_threads_read_at_once_on_their_own_threads(
    datastore_zarr_root, tmp_path, monkeypatch
) -> None:
    # Two stores read from two threads at once: each decodes on a pool of its
    # own nthreads, since each call holds its store's threads in its own
    # thread's context.
    one = _copied_store(datastore_zarr_root, tmp_path, nthreads=1)
    two = _copied_store(datastore_zarr_root, tmp_path, nthreads=2)
    decodes = _Decodes(monkeypatch, each_reader=("one", "two"))

    def read(store: Any, reader: str) -> np.ndarray:
        token = _READER.set(reader)
        try:
            return np.asarray(store.cells.fetch_all("RNA_nCounts"))
        finally:
            _READER.reset(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        reads = [pool.submit(read, one, "one"), pool.submit(read, two, "two")]
        first, second = (read.result(timeout=60) for read in reads)
    np.testing.assert_array_equal(first, second)
    for reader, threads in (("one", 1), ("two", 2)):
        names = {thread.name for thread in decodes.threads(reader)}
        assert names and all(f"_of_{threads}_" in name for name in names), names


def _imported(folder: Path, tag: str) -> str:
    """Import the 1K PBMC Cell Ranger file into a fresh store, and return its location."""
    from cytearc import CrH5Reader, CrToZarr
    from tests import full_path

    location = str(folder / f"imported_{tag}.zarr")
    CrToZarr(CrH5Reader(full_path("1K_pbmc_citeseq.h5")), location).dump()
    return location


@pytest.mark.slow
def test_the_first_writable_open_of_an_import_prepares_it_on_the_stores_thread(
    tmp_path,
) -> None:
    # The first writable open of a fresh import computes its cell properties
    # and filter, reading its counts and writing columns; with nthreads=1 all
    # of it decodes and encodes on one thread.
    location = _imported(tmp_path, "open")
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "open"
        DataStore(location, default_assay="RNA", nthreads=1)
        pools.label = None
    assert len(pools.threads("open")) > 20
    assert set(pools.threads("open")) == {1}
    assert pools.threads(None) == []


def _prepare(name: str, analysis: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Open the sources that a writer reads, with more threads than the writer's one."""
    source = analysis["source"]
    match name:
        case "h5ad import":
            path = folder / "export.h5ad"
            to_h5ad(analysis["store"].RNA, str(path))
            return {"h5ad": str(path)}
        case "subset" | "subset_assay_zarr" | "mount" | "repack_store":
            store = _copied_store(source, folder / "source", nthreads=3)
            return {
                "store": store,
                "cell_ids": store.cells.fetch_all("ids")[:200],
                "shape": store.RNA.rawData.shape,
            }
        case "merge" | "composite":
            return {
                "stores": [
                    _copied_store(source, folder / "a", nthreads=2),
                    _copied_store(source, folder / "b", nthreads=3),
                ]
            }
    return {}


def _write(name: str, prepared: dict[str, Any], folder: Path) -> None:
    from scipy.sparse import random as sparse_random

    from cytearc import (
        CrH5Reader,
        CrToZarr,
        CSVReader,
        CSVtoZarr,
        DataStoreMerge,
        H5adReader,
        H5adToZarr,
        SeuratReader,
        SeuratToZarr,
        SparseToZarr,
        SubsetZarr,
        subset_assay_zarr,
    )
    from cytearc.composite.create import create_composite
    from cytearc.datastore.datastore import mount_datastore
    from cytearc.tools.repack_zarr import repack_store
    from tests import full_path

    match name:
        case "cell ranger import":
            CrToZarr(
                CrH5Reader(full_path("1K_pbmc_citeseq.h5")),
                str(folder / "imported.zarr"),
                nthreads=1,
            ).dump()
        case "h5ad import":
            reader = H5adReader(prepared["h5ad"])
            try:
                H5adToZarr(reader, str(folder / "h5ad.zarr"), nthreads=1).dump()
            finally:
                reader.close()
        case "seurat import":
            with SeuratReader(
                Path(full_path("seurat_assay5_synthetic.rds")), reductions=[]
            ) as reader:
                SeuratToZarr(reader, str(folder / "seurat.zarr"), nthreads=1).dump()
        case "csv import":
            table = folder / "counts.csv"
            rows = [f"c{row},{row % 3},{row % 5},{(row * 7) % 4}" for row in range(60)]
            table.write_text("cell,g0,g1,g2\n" + "\n".join(rows) + "\n")
            CSVtoZarr(
                CSVReader(str(table), id_column=0, batch_size=20),
                str(folder / "csv.zarr"),
                assay_name="RNA",
                nthreads=1,
            ).dump()
        case "sparse import":
            counts = sparse_random(
                120, 40, density=0.2, format="csr", random_state=3
            ).astype("float32")
            counts.data = (counts.data * 9).round() + 1
            SparseToZarr(
                counts,
                str(folder / "sparse.zarr"),
                cell_ids=[f"c{row}" for row in range(120)],
                feature_ids=[f"g{column}" for column in range(40)],
                nthreads=1,
            ).dump()
        case "subset":
            SubsetZarr(
                str(folder / "subset.zarr"),
                [prepared["store"].RNA],
                cell_ids=prepared["cell_ids"],
                nthreads=1,
            ).dump()
        case "subset_assay_zarr":
            cells, features = prepared["shape"]
            subset_assay_zarr(
                str(prepared["store"].zarr_loc),
                "RNA/counts",
                "RNA_subset/counts",
                np.arange(0, cells, 2),
                np.arange(0, features, 3),
                nthreads=1,
            )
        case "merge":
            DataStoreMerge(
                prepared["stores"], str(folder / "merged.zarr"), ["a", "b"], nthreads=1
            ).dump()
        case "mount":
            mount_datastore(
                str(prepared["store"].zarr_loc),
                str(folder / "mounted.zarr"),
                default_assay="RNA",
                nthreads=1,
            )
        case "composite":
            first, second = prepared["stores"]
            create_composite(
                {"a": first, "b": second},
                at=folder / "composite.zarr",
                features="union",
                nthreads=1,
            )
        case "repack_store":
            repack_store(
                str(prepared["store"].zarr_loc),
                str(folder / "repacked.zarr"),
                nthreads=1,
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "cell ranger import",
        "h5ad import",
        "seurat import",
        "csv import",
        "sparse import",
        "subset",
        "subset_assay_zarr",
        "merge",
        "mount",
        "composite",
        "repack_store",
    ],
)
def test_a_writer_of_one_thread_writes_its_new_store_on_one_thread(
    one_thread_analysis, tmp_path, name
) -> None:
    # Each import, both subsets, a merge, a mount, a composite and a repack
    # given nthreads=1 read their sources and write their new data,
    # construction included, on one thread, although the DataStores that
    # opened their sources have two or three.
    prepared = _prepare(name, one_thread_analysis, tmp_path)
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = name
        _write(name, prepared, tmp_path)
        pools.label = None
    assert len(pools.threads(name)) >= 5, pools.threads(name)
    assert set(pools.threads(name)) == {1}, pools.threads(name)


@pytest.mark.slow
def test_a_mount_reads_its_sources_counts_on_its_threads(
    datastore_zarr_root, tmp_path
) -> None:
    # A mount's counts live in its source store, which the mount registers
    # with the mount's threads beside its own, so they decode on the fewer of
    # the mount's one thread and the three of the DataStore that opened the
    # source. A caller's own open of the mount's target reads on one thread
    # too.
    from cytearc.datastore.datastore import mount_datastore

    source = _copied_store(datastore_zarr_root, tmp_path, nthreads=3)
    mounted = mount_datastore(
        str(source.zarr_loc), str(tmp_path / "mounted.zarr"), nthreads=1
    )
    # The mount's store wraps its target's, which a caller can open itself.
    target = zarr.open_group(str(tmp_path / "mounted.zarr"), mode="r")
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "counts"
        mounted.RNA.rawData[:300].compute()
        pools.label = "cells"
        mounted.cells.fetch_all("ids")
        pools.label = "target"
        target["cellData/ids"][:]
    for label in ("counts", "cells", "target"):
        assert pools.threads(label) and set(pools.threads(label)) == {1}, label


@pytest.mark.slow
def test_a_composite_reads_its_sources_on_its_threads(
    datastore_zarr_root, tmp_path
) -> None:
    # A composite reads its sources' stores on its own thread, whatever the
    # threads of the DataStores that opened them: as it is created, and as it
    # reads its counts from them afterwards.
    from cytearc.composite.create import create_composite

    sources = {
        name: _copied_store(datastore_zarr_root, tmp_path / name, nthreads=2)
        for name in ("a", "b")
    }
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "creation"
        composite = create_composite(
            sources, at=tmp_path / "composite.zarr", features="union", nthreads=1
        )
        pools.label = "counts"
        composite.RNA.rawData[:200].compute()
    for label in ("creation", "counts"):
        assert pools.threads(label) and set(pools.threads(label)) == {1}, label


def test_a_mount_refuses_a_store_object_before_it_opens_it(
    datastore_zarr_root, tmp_path
) -> None:
    # A mount takes its source as a location; it refuses a store object
    # before opening it, so the store registers no threads and no target is
    # created.
    from zarr.storage import LocalStore

    from cytearc.datastore.datastore import mount_datastore

    location = tmp_path / "source.zarr"
    shutil.copytree(datastore_zarr_root, location)
    store = LocalStore(str(location), read_only=True)
    with pytest.raises(TypeError, match="Matrix source location must be"):
        mount_datastore(store, str(tmp_path / "mounted.zarr"), nthreads=1)
    assert _entries_of(store) == []
    assert not (tmp_path / "mounted.zarr").exists()


# Which threads a store's reads decode on: the caller's, else the store's,
# else the host's.


def _registered_values(folder: Path, threads: int) -> tuple[zarr.Group, np.ndarray]:
    root = zarr.open_group(str(folder / "registered.zarr"), mode="w")
    values = _values(root, n_chunks=4)
    async_execution.register_store_threads(root.store, threads)
    return root, values


def test_a_registered_store_decodes_on_its_threads_unless_its_caller_sets_them(
    tmp_path,
) -> None:
    # A store registered with two threads decodes on two, wherever no caller
    # sets threads; a caller's three win, and a store that no one registered
    # decodes on the host's seven.
    root, values = _registered_values(tmp_path, 2)
    other = zarr.open_group(str(tmp_path / "other.zarr"), mode="w")
    _values(other, n_chunks=4)
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "store"
        np.testing.assert_array_equal(root["values"][:], values)
        pools.label = "caller"
        with zarr_decoding_threads(3):
            np.testing.assert_array_equal(root["values"][:], values)
        pools.label = "after"
        np.testing.assert_array_equal(root["values"][:], values)
        pools.label = "unregistered"
        other["values"][:]
        pools.label = "write"
        root["values"][:1_000] = values[:1_000]
    assert pools.threads("store") == [2] * 4
    assert pools.threads("caller") == [3] * 4
    assert pools.threads("after") == [2] * 4
    assert pools.threads("unregistered") == [_HOST_THREADS] * 4
    assert pools.threads("write") == [2]


def test_a_copy_or_another_open_of_a_registered_location_takes_its_threads(
    tmp_path,
) -> None:
    # The read-only copy that zarr.open_group(store, mode="r") makes, and a
    # caller's own open of the same location, equal the registered store and
    # decode on its threads.
    root, values = _registered_values(tmp_path, 2)
    copy = zarr.open_group(root.store, mode="r")
    reopened = zarr.open_group(str(tmp_path / "registered.zarr"), mode="r")
    assert copy.store is not root.store and reopened.store is not root.store
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "copy"
        np.testing.assert_array_equal(copy["values"][:], values)
        pools.label = "reopened"
        np.testing.assert_array_equal(reopened["values"][:], values)
    assert pools.threads("copy") == [2] * 4
    assert pools.threads("reopened") == [2] * 4


def test_the_stores_of_one_location_decode_on_the_fewest_threads_registered_for_it(
    tmp_path,
) -> None:
    # Two local stores of one location registered with one and three threads,
    # another open of the location and a wrapper of it all decode on one, so
    # no store exceeds either bound. Once the store of one thread is
    # collected, the others decode on three.
    import gc

    from zarr.storage import LocalStore, LoggingStore

    root = zarr.open_group(str(tmp_path / "shared.zarr"), mode="w")
    values = _values(root, n_chunks=2)
    first = zarr.open_group(str(tmp_path / "shared.zarr"), mode="r")
    second = zarr.open_group(str(tmp_path / "shared.zarr"), mode="r")
    async_execution.register_store_threads(first.store, 1)
    async_execution.register_store_threads(second.store, 3)
    wrapped = zarr.open_group(
        store=LoggingStore(LocalStore(str(tmp_path / "shared.zarr"), read_only=True)),
        mode="r",
    )
    groups = {"first": first, "second": second, "other": root, "wrapped": wrapped}
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        for label, group in groups.items():
            pools.label = label
            np.testing.assert_array_equal(group["values"][:], values)
        del groups["first"], first
        gc.collect()
        for label, group in groups.items():
            pools.label = f"{label} later"
            np.testing.assert_array_equal(group["values"][:], values)
    for label in ("first", "second", "other", "wrapped"):
        assert pools.threads(label) == [1, 1], label
    for label in ("second", "other", "wrapped"):
        assert pools.threads(f"{label} later") == [3, 3], label


def test_a_store_registered_again_keeps_the_fewest_threads(tmp_path) -> None:
    # One store object that two DataStores registered with three and with one
    # thread decodes on one, whatever the order, and registering it with
    # three again changes nothing.
    root, values = _registered_values(tmp_path, 3)
    async_execution.register_store_threads(root.store, 1)
    async_execution.register_store_threads(root.store, 3)
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "store"
        np.testing.assert_array_equal(root["values"][:], values)
    assert pools.threads("store") == [1] * 4
    assert sorted(_entries_of(root.store)) == [1, 3]


def test_a_memory_store_takes_only_its_own_threads(monkeypatch) -> None:
    # A memory store's equality compares its keys and bytes, not a location,
    # so another memory store that holds the same bytes takes no threads of a
    # registered one, and a lookup never compares them.
    from zarr.core.buffer import cpu
    from zarr.storage import MemoryStore

    memory = MemoryStore()
    root = zarr.open_group(store=memory, mode="w")
    values = _values(root, n_chunks=2)
    async_execution.register_store_threads(memory, 2)
    twin = MemoryStore(
        {
            key: cpu.Buffer.from_bytes(value.to_bytes())
            for key, value in memory._store_dict.items()
        }
    )
    assert twin == memory

    def compared(self: Any, other: object) -> bool:
        raise AssertionError("a lookup compared two memory stores")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryStore, "__eq__", compared)
        assert async_execution._registered_threads(twin) is None
        assert async_execution._registered_threads(memory) == 2
    with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
        pools.label = "registered"
        np.testing.assert_array_equal(root["values"][:], values)
        pools.label = "twin"
        np.testing.assert_array_equal(
            zarr.open_group(store=twin, mode="r")["values"][:], values
        )
    assert pools.threads("registered") == [2, 2]
    assert pools.threads("twin") == [_HOST_THREADS] * 2


def test_a_lookup_compares_a_store_only_with_registered_stores_of_its_type(
    tmp_path, caplog
) -> None:
    # A LoggingStore logs every call, its equality checks included; reads of
    # stores of other types never ask a registered one whether it equals them.
    import logging

    from zarr.storage import LocalStore, LoggingStore

    logged = LoggingStore(LocalStore(str(tmp_path / "logged.zarr")))
    async_execution.register_store_threads(logged, 2)
    root = zarr.open_group(str(tmp_path / "plain.zarr"), mode="w")
    values = _values(root, n_chunks=2)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        np.testing.assert_array_equal(root["values"][:], values)
    assert not [record for record in caplog.records if "__eq__" in record.getMessage()]
    assert async_execution._registered_threads(logged) == 2


def _entries_of(store: Any) -> list[int]:
    return [
        threads
        for reference, threads in async_execution._STORE_THREADS
        if reference() is store
    ]


@pytest.mark.parametrize("closed", [False, True], ids=["dropped", "closed"])
def test_the_registry_keeps_no_store_alive_and_forgets_a_collected_one(
    tmp_path, closed
) -> None:
    # The registry holds a store weakly: once nothing else holds the store,
    # closed or not, it is collected, its registration no longer bounds
    # another open of its location, and the next registration drops its
    # entry.
    import gc
    import weakref

    from zarr.storage import LocalStore

    store = LocalStore(str(tmp_path / "dropped.zarr"))
    async_execution.register_store_threads(store, 2)
    reopened = LocalStore(str(tmp_path / "dropped.zarr"))
    assert async_execution._registered_threads(reopened) == 2
    (entry,) = [
        reference
        for reference, _ in async_execution._STORE_THREADS
        if reference() is store
    ]
    held = weakref.ref(store)
    if closed:
        store.close()
    del store
    gc.collect()
    assert held() is None
    assert async_execution._registered_threads(reopened) is None
    later = LocalStore(str(tmp_path / "later.zarr"))
    async_execution.register_store_threads(later, 3)
    assert all(
        reference is not entry for reference, _ in async_execution._STORE_THREADS
    )


def test_registrations_from_many_threads_all_hold(tmp_path) -> None:
    # Eight threads register 150 stores each at once, switching between
    # threads every microsecond so that registrations interleave; every store
    # keeps its own threads and none is lost.
    import sys

    from zarr.storage import LocalStore

    stores = [
        [LocalStore(str(tmp_path / f"s{worker}_{index}.zarr")) for index in range(150)]
        for worker in range(8)
    ]
    barrier = threading.Barrier(8, timeout=20)

    def register(worker: int) -> None:
        barrier.wait()
        for store in stores[worker]:
            async_execution.register_store_threads(store, worker + 1)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(register, range(8)))
    finally:
        sys.setswitchinterval(interval)
    for worker, of_worker in enumerate(stores):
        for store in of_worker:
            assert _entries_of(store) == [worker + 1]
            assert async_execution._registered_threads(store) == worker + 1


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2", None])
def test_a_stores_threads_are_a_whole_number_of_at_least_one(tmp_path, threads) -> None:
    from zarr.storage import LocalStore

    store = LocalStore(str(tmp_path / "refused.zarr"))
    with pytest.raises(ValueError, match="whole number of threads of at least 1"):
        async_execution.register_store_threads(store, threads)
    assert _entries_of(store) == []


def test_registering_a_store_installs_cytearcs_codec_pipeline_first(tmp_path) -> None:
    # In a new interpreter whose first CyteArc call registers a store, the
    # store's arrays, opened after it, read through CyteArc's pipeline.
    import subprocess
    import sys

    code = (
        "import zarr, numpy as np\n"
        "from cytearc.storage import async_execution\n"
        f"group = zarr.open_group({str(tmp_path / 'fresh.zarr')!r}, mode='w')\n"
        "group.create_array('values', data=np.arange(10))\n"
        "async_execution.register_store_threads(group.store, 1)\n"
        "array = zarr.open_group(group.store)['values']\n"
        "print(type(array.async_array.codec_pipeline).__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == "StoreThreadsCodecPipeline"


@pytest.mark.slow
def test_the_first_datastore_of_a_process_is_freed_once_dropped(
    datastore_zarr_root, tmp_path
) -> None:
    # The first DataStore of a new interpreter starts Zarr's event loop, which
    # keeps the context it starts in for as long as it runs. It starts in an
    # empty one, so the DataStore's store and arrays go once the DataStore is
    # dropped, and so does its registration.
    import subprocess
    import sys

    location = str(tmp_path / "first.zarr")
    shutil.copytree(datastore_zarr_root, location)
    code = (
        "import gc, weakref, zarr\n"
        "from zarr.storage import LocalStore\n"
        "from cytearc import DataStore\n"
        "from cytearc.storage import async_execution\n"
        f"store = DataStore({location!r}, default_assay='RNA', nthreads=2)\n"
        "held = weakref.ref(store.z.store)\n"
        "del store\n"
        "gc.collect()\n"
        "arrays = [o for o in gc.get_objects() if isinstance(o, zarr.Array)]\n"
        f"print(held() is None, len([a for a in arrays if {location!r} in str(a.store)]),"
        f" async_execution._registered_threads(LocalStore({location!r})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-3:] == ["True", "0", "None"]


class _OwnPipeline(BatchedCodecPipeline):
    """A codec pipeline that a user configures in place of Zarr's default."""


def test_cytearcs_codec_pipeline_replaces_only_zarrs_default(tmp_path) -> None:
    # With Zarr's default pipeline configured, the host ceiling installs
    # CyteArc's; a pipeline that a user configured stays, and a registered
    # store's arrays opened under it decode on the host's threads.
    from zarr.registry import register_pipeline

    default = "zarr.core.codec_pipeline.BatchedCodecPipeline"
    own = f"{__name__}._OwnPipeline"
    register_pipeline(_OwnPipeline)
    with zarr.config.set({"codec_pipeline.path": default}):
        ensure_zarr_host_ceiling()
        assert zarr.config.get("codec_pipeline.path") == (
            "cytearc.storage.async_execution.StoreThreadsCodecPipeline"
        )
    root, values = _registered_values(tmp_path, 2)
    with zarr.config.set({"codec_pipeline.path": own}):
        ensure_zarr_host_ceiling()
        assert zarr.config.get("codec_pipeline.path") == own
        opened = zarr.open_group(root.store, mode="r")["values"]
        assert type(opened.async_array.codec_pipeline) is _OwnPipeline
        with recorded_codec_pools(host_threads=_HOST_THREADS) as pools:
            pools.label = "own"
            np.testing.assert_array_equal(opened[:], values)
    assert pools.threads("own") == [_HOST_THREADS] * 4


# A pool thread that has reported a call's result.


class _SetResultPause:
    """Pause the pool thread of an armed call just after it sets the call's result.

    The thread waits inside ``set_result``, once the future's callbacks have
    told the event loop, until the test releases it, as a thread that the OS
    deschedules there would: it still holds the call, its future and the
    result it returned. A call arms the pause of the thread that runs it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import concurrent.futures.thread as pool_thread

        self._armed = threading.local()
        self.paused = threading.Event()
        self.release = threading.Event()
        run = pool_thread._WorkItem.run
        pause = self

        def paused_run(item, *args):  # type: ignore[no-untyped-def]
            future = item.future
            set_result = future.set_result

            def set_result_then_pause(result):  # type: ignore[no-untyped-def]
                set_result(result)
                if getattr(pause._armed, "on", False):
                    pause._armed.on = False
                    pause.paused.set()
                    assert pause.release.wait(timeout=20)

            future.set_result = set_result_then_pause
            try:
                run(item, *args)
            finally:
                del future.set_result

        monkeypatch.setattr(pool_thread._WorkItem, "run", paused_run)

    def arm(self) -> None:
        self._armed.on = True


class _LoopWakePause:
    """Pause the pool thread of an armed call just after it wakes an event loop.

    The thread waits once its write to the loop's wake-up socket has
    returned, as a thread that the OS deschedules after that write would,
    which releases the GIL: it is still inside the callback that handed the
    loop its result. A call arms the pause of the thread that runs it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio.selector_events as selector_events

        self._armed = threading.local()
        self.paused = threading.Event()
        self.release = threading.Event()
        write = selector_events.BaseSelectorEventLoop._write_to_self
        pause = self

        def paused_write(loop):  # type: ignore[no-untyped-def]
            write(loop)
            if getattr(pause._armed, "on", False):
                pause._armed.on = False
                pause.paused.set()
                assert pause.release.wait(timeout=20)

        monkeypatch.setattr(
            selector_events.BaseSelectorEventLoop, "_write_to_self", paused_write
        )

    def arm(self) -> None:
        self._armed.on = True


@pytest.fixture
def without_cyclic_collection() -> Iterator[None]:
    """Free objects by their references alone, so that one a cycle holds stays alive."""
    import gc

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class _Chunk:
    """Stands in for a decoded chunk: an object that a weak reference can follow."""


def _chunk_outlives_its_reader(
    pool: ThreadPoolExecutor, pause: _SetResultPause | _LoopWakePause
) -> bool:
    """Whether a pool thread that pauses after returning a chunk keeps it once its reader let go."""
    import asyncio
    import weakref

    returned: list[weakref.ref[_Chunk]] = []
    awaited = threading.Event()

    def decode() -> _Chunk:
        # The call ends once its reader awaits it, so that the pool thread,
        # not the loop, hands the chunk over.
        assert awaited.wait(timeout=20)
        pause.arm()
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    async def read() -> bool:
        reading = asyncio.get_running_loop().run_in_executor(pool, decode)
        awaited.set()
        chunk = await reading
        assert isinstance(chunk, _Chunk)
        # The reader lets go of the chunk and of the future that holds it.
        del chunk, reading
        # The loop lets go of the callback that resumed this read, and its
        # future, once the read yields.
        await asyncio.sleep(0)
        # The thread that returned the chunk still waits in set_result.
        assert pause.paused.wait(timeout=20)
        return returned[0]() is not None

    try:
        return asyncio.run(read())
    finally:
        pause.release.set()


@pytest.mark.parametrize(
    "pause_kind",
    [_SetResultPause, _LoopWakePause],
    ids=["after setting the result", "after waking the loop"],
)
@pytest.mark.parametrize(
    ("pool", "keeps"),
    [
        (lambda: ThreadPoolExecutor(2), True),
        (lambda: async_execution._ReleasingPool(2), False),
        (lambda: async_execution._DecodingPools(2), False),
        (lambda: async_execution._DecodingPools(3), False),
    ],
    ids=["a plain pool", "a releasing pool", "the host threads", "a counted pool"],
)
def test_a_pool_thread_paused_after_its_result_keeps_no_chunk_its_reader_dropped(
    monkeypatch, without_cyclic_collection, pool, keeps, pause_kind
) -> None:
    # A plain pool's thread keeps the chunk that it returned until it runs
    # again; CyteArc's pools hand the chunk to the reader's loop, so the
    # paused thread holds nothing that reaches it, whether it pauses once it
    # has set the call's result or while it wakes the loop. The decoding
    # pools of three threads decode within zarr_decoding_threads(2) on a pool
    # of two.
    pause = pause_kind(monkeypatch)
    executor = pool()
    try:
        with zarr_decoding_threads(2):
            assert _chunk_outlives_its_reader(executor, pause) is keeps
    finally:
        executor.shutdown(wait=True)


def _read_through_run_read(
    array: zarr.Array, check: Callable[[np.ndarray], None]
) -> None:
    from cytearc.storage.virtual_arrays import read_selection

    with zarr_decoding_threads(2):
        values = read_selection(array, slice(None), slice(None))
    check(values)


def _read_through_zarrs_loop(
    array: zarr.Array, check: Callable[[np.ndarray], None]
) -> None:
    check(np.asarray(array[...]))


def _read_through_a_storage_runner(
    array: zarr.Array, check: Callable[[np.ndarray], None]
) -> None:
    from dataclasses import replace

    from cytearc.storage.async_execution import AsyncStorageRunner
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.execution import WorkShape, plan_operation

    # Two codec threads on any machine, so that one can pause.
    plan = plan_operation(
        ResourceBudget(64 * 1024**2, 2), WorkShape(nUnits=2, unitBytes=1)
    )
    runner = AsyncStorageRunner(operation=replace(plan, codecWorkers=2))

    async def operation(active: AsyncStorageRunner) -> None:
        values = await active.io(array.async_array.getitem((slice(None), slice(None))))
        # The runner waits for its codec threads as it ends, so the read is
        # checked before then.
        check(np.asarray(values))

    runner.run(operation)


@pytest.mark.parametrize(
    "read",
    [_read_through_run_read, _read_through_zarrs_loop, _read_through_a_storage_runner],
    ids=["a read on its own loop", "Zarr's loop", "a storage runner"],
)
def test_a_read_leaves_no_decoded_chunk_with_a_paused_pool_thread(
    tmp_path, monkeypatch, without_cyclic_collection, read
) -> None:
    # The thread of the first decode pauses after it returns the chunk, as
    # a busy machine can deschedule it; the read decodes its other chunks on
    # other threads and copies every chunk into its result. Once the read has
    # returned, no decoded chunk is left, nor the stored bytes it was decoded
    # from, though that thread has not run again.
    import weakref

    values = np.arange(8_000, dtype=np.int64).reshape(80, 100)
    array = zarr.open_group(tmp_path / "store.zarr", mode="w").create_array(
        "values", data=values, chunks=(10, 100), compressors=ZstdCodec(level=1)
    )
    pause = _SetResultPause(monkeypatch)
    decoded: list[weakref.ref[Any]] = []
    stored: list[weakref.ref[Any]] = []
    lock = threading.Lock()
    decode = ZstdCodec._decode_sync

    def recording(codec, chunk_bytes, chunk_spec):  # type: ignore[no-untyped-def]
        chunk = decode(codec, chunk_bytes, chunk_spec)
        with lock:
            if not decoded:
                pause.arm()
            decoded.append(weakref.ref(chunk))
            stored.append(weakref.ref(chunk_bytes))
        return chunk

    monkeypatch.setattr(ZstdCodec, "_decode_sync", recording)
    checked: list[bool] = []

    def check(result: np.ndarray) -> None:
        try:
            np.testing.assert_array_equal(result, values)
            assert pause.paused.wait(timeout=20)
            assert len(decoded) == 8
            assert [chunk() for chunk in decoded] == [None] * 8
            # Nor are the stored bytes that each decode read.
            assert [chunk() for chunk in stored] == [None] * 8
            checked.append(True)
        finally:
            pause.release.set()

    try:
        read(array, check)
    finally:
        pause.release.set()
    assert checked == [True]


def test_a_stopped_read_leaves_its_last_chunk_with_no_paused_pool_thread(
    monkeypatch, without_cyclic_collection
) -> None:
    # A block stream stops a read on its own loop while a decode runs. The
    # read waits for the decode to end and closes its loop, which never runs
    # the call's hand-back; the decode's thread pauses after it returns the
    # chunk, as a busy machine can deschedule it. Once the read has ended,
    # the chunk is gone, though that thread has not run again.
    import asyncio
    import weakref

    from cytearc.storage.async_execution import (
        ReadStop,
        ReadStopped,
        read_stop_scope,
        run_read,
    )

    pause = _SetResultPause(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    returned: list[weakref.ref[_Chunk]] = []

    def decode() -> _Chunk:
        started.set()
        assert release.wait(timeout=20)
        pause.arm()
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    wait_for_calls = async_execution._LentPool.wait_for_calls

    def waiting_for_calls(lent: Any) -> None:
        waiting.set()
        wait_for_calls(lent)

    monkeypatch.setattr(async_execution._LentPool, "wait_for_calls", waiting_for_calls)

    async def read() -> _Chunk:
        return await asyncio.to_thread(decode)

    stop = ReadStop()
    outcome: list[str] = []

    def reader() -> None:
        with read_stop_scope(stop):
            try:
                run_read(read())
            except ReadStopped:
                outcome.append("stopped")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert started.wait(timeout=20)
        stop.stop()
        # The stopped read waits for its decode, which then ends.
        assert waiting.wait(timeout=20)
        release.set()
        thread.join(timeout=20)
        assert not thread.is_alive()
        assert pause.paused.wait(timeout=20)
        assert outcome == ["stopped"]
        assert len(returned) == 1
        assert returned[0]() is None
    finally:
        release.set()
        pause.release.set()
        thread.join(timeout=20)


def _held_call(started: threading.Event, release: threading.Event):  # type: ignore[no-untyped-def]
    """A call that holds its pool thread until ``release`` is set."""

    def call() -> str:
        started.set()
        assert release.wait(timeout=20)
        return "held"

    return call


def test_a_call_that_raises_reaches_its_reader_through_a_releasing_pool() -> None:
    import asyncio

    def failing() -> None:
        raise KeyError("no chunk at 3")

    async def read() -> None:
        loop = asyncio.get_running_loop()
        with pytest.raises(KeyError, match="no chunk at 3"):
            await loop.run_in_executor(pool, failing)
        assert await loop.run_in_executor(pool, lambda: "next") == "next"

    with async_execution._ReleasingPool(1) as pool:
        asyncio.run(read())


def test_a_queued_call_whose_reader_was_cancelled_never_runs() -> None:
    import asyncio

    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    async def read() -> None:
        loop = asyncio.get_running_loop()
        holding = loop.run_in_executor(pool, _held_call(started, release))
        queued = asyncio.ensure_future(
            loop.run_in_executor(pool, lambda: ran.append("queued"))
        )
        assert started.wait(timeout=20)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        # The loop cancels the call as it runs the cancelled future's
        # callbacks, before the held call ends.
        await asyncio.sleep(0)
        release.set()
        assert await holding == "held"
        # The pool runs another call after the one it held, not the cancelled one.
        await loop.run_in_executor(pool, lambda: ran.append("later"))

    with async_execution._ReleasingPool(1) as pool:
        try:
            asyncio.run(read())
        finally:
            release.set()
    assert ran == ["later"]


def test_a_started_call_whose_reader_was_cancelled_runs_on_and_keeps_nothing(
    without_cyclic_collection,
) -> None:
    import asyncio
    import weakref

    started = threading.Event()
    release = threading.Event()
    returned: list[weakref.ref[_Chunk]] = []

    def decode() -> _Chunk:
        started.set()
        assert release.wait(timeout=20)
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    async def read() -> None:
        loop = asyncio.get_running_loop()
        reading = asyncio.ensure_future(loop.run_in_executor(pool, decode))
        assert started.wait(timeout=20)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        # The call has started, so it runs to its end, and the loop drops the
        # chunk it hands back.
        release.set()
        assert await loop.run_in_executor(pool, lambda: "next") == "next"
        await asyncio.sleep(0)

    with async_execution._ReleasingPool(1) as pool:
        try:
            asyncio.run(read())
        finally:
            release.set()
    assert len(returned) == 1
    assert returned[0]() is None


def test_a_pool_that_shuts_down_cancels_the_reader_of_a_queued_call() -> None:
    import asyncio

    started = threading.Event()
    release = threading.Event()

    async def read() -> None:
        loop = asyncio.get_running_loop()
        holding = loop.run_in_executor(pool, _held_call(started, release))
        queued = loop.run_in_executor(pool, lambda: "queued")
        assert started.wait(timeout=20)
        pool.shutdown(wait=False, cancel_futures=True)
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await holding == "held"

    pool = async_execution._ReleasingPool(1)
    try:
        asyncio.run(read())
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_a_call_whose_loop_closed_keeps_nothing_once_it_ends(
    monkeypatch, without_cyclic_collection
) -> None:
    import asyncio
    import weakref

    started = threading.Event()
    release = threading.Event()
    returned: list[weakref.ref[_Chunk]] = []
    pause = _SetResultPause(monkeypatch)

    def decode() -> _Chunk:
        started.set()
        assert release.wait(timeout=20)
        pause.arm()
        chunk = _Chunk()
        returned.append(weakref.ref(chunk))
        return chunk

    async def submit() -> Any:
        return pool.submit(decode)

    with async_execution._ReleasingPool(1) as pool:
        loop = asyncio.new_event_loop()
        try:
            future = loop.run_until_complete(submit())
            assert started.wait(timeout=20)
        finally:
            loop.close()
        release.set()
        try:
            # Nothing awaits the call once its loop has closed, so its chunk
            # goes as the call ends, while its thread waits in set_result.
            assert pause.paused.wait(timeout=20)
            assert len(returned) == 1
            assert returned[0]() is None
        finally:
            pause.release.set()
    assert not future.done()


def test_a_read_does_not_wait_for_a_call_that_ended_before_it_looked() -> None:
    import asyncio
    import time

    class EndingPool(async_execution._ReleasingPool):
        """Return each future of an event loop's call once the call has ended and gone."""

        def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            future = super().submit(fn, *args, **kwargs)
            deadline = time.monotonic() + 20
            while future.call() is not None:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            return future

    async def read() -> int:
        value = await asyncio.get_running_loop().run_in_executor(lent, lambda: 42)
        assert lent._calls == set()
        lent.wait_for_calls()
        return value

    with EndingPool(1) as pool:
        lent = async_execution._LentPool(pool)
        assert asyncio.run(read()) == 42
