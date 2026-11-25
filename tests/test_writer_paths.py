"""Edge paths of the H5AD, Seurat, sparse, Cell Ranger, subset, and CSV writers."""

import re
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.storage import LocalStore, MemoryStore

from cytearc import DataStore
from cytearc.readers import CSVReader, H5adReader
from cytearc.readers.cellranger import CrReader
from cytearc.readers._rds._types import RType
from cytearc.readers.seurat import SeuratReader
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.count_matrix import CountMatrixPolicy, plan_count_matrix_pair
from cytearc.utils.count_values import CountValueRange
from cytearc.writers import (
    CSVtoZarr,
    CrToZarr,
    H5adToZarr,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
)
from tests.memory_helpers import HandedArrays, received_messages
from tests.test_seurat_reader import _Wire, _legacy_assay
from tests.test_writers import _write_h5ad
from tests.storage_helpers import by_cell_id

_CELLS = ["c1", "c2", "c3"]


def _sentinel_store() -> MemoryStore:
    store = MemoryStore()
    zarr.open_group(store=store, mode="w").create_group("sentinel")
    return store


def _untouched(store: MemoryStore) -> bool:
    return set(zarr.open_group(store=store, mode="r").group_keys()) == {"sentinel"}


def _store_bytes(store: MemoryStore) -> dict[str, bytes]:
    return {key: bytes(value.to_bytes()) for key, value in store._store_dict.items()}


def test_h5ad_analysis_assay_must_own_a_selected_output(tmp_path) -> None:
    source = _write_h5ad(tmp_path / "cells.h5ad", np.eye(3, dtype=np.uint8))
    with h5py.File(source, mode="r+") as h5:
        h5["obs"].create_dataset("clusters", data=np.array([0, 1, 0], dtype=np.int8))
    plain = H5adReader(str(source), feature_name_key="feature_name")
    clustered = H5adReader(
        str(source), feature_name_key="feature_name", cluster_keys=("clusters",)
    )
    destination = _sentinel_store()
    try:
        with pytest.raises(ValueError, match="requires at least one explicitly"):
            H5adToZarr(plain, zarr_loc=destination, analysis_assay="RNA")
        with pytest.raises(ValueError, match="must name an imported assay"):
            H5adToZarr(clustered, zarr_loc=destination, analysis_assay="ADT")
    finally:
        plain.close()
        clustered.close()
    assert _untouched(destination)


def test_h5ad_numeric_category_clusters_keep_missing_cells(tmp_path) -> None:
    counts = np.array([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.uint8)
    source = _write_h5ad(tmp_path / "scores.h5ad", counts)
    with h5py.File(source, mode="r+") as h5:
        score = h5["obs"].create_group("score")
        score.create_dataset("codes", data=np.array([0, -1, 1, 0], dtype=np.int8))
        score.create_dataset("categories", data=np.array([0.5, 1.5]))
    reader = H5adReader(
        str(source), feature_name_key="feature_name", cluster_keys=("score",)
    )
    destination = MemoryStore()
    try:
        # One-cell blocks leave the block of the unlabeled cell without values.
        result = H5adToZarr(
            reader, zarr_loc=destination, nthreads=1, shuffle_cells=False
        ).dump(batch_size=1)
    finally:
        reader.close()

    labels = artifact_group(
        zarr.open_group(store=destination, mode="r"),
        result.clusterArtifacts["score"],
    )
    assert labels["values"].dtype == np.float64
    np.testing.assert_array_equal(labels["values"][:], [0.5, 0.0, 1.5, 0.5])
    np.testing.assert_array_equal(
        labels[labels["values"].attrs["missing_mask"]][:],
        [False, True, False, False],
    )


def test_h5ad_repeated_dump_reuses_its_artifacts(tmp_path) -> None:
    source = _write_h5ad(tmp_path / "twice.h5ad", np.eye(3, dtype=np.uint8))
    with h5py.File(source, mode="r+") as h5:
        leiden = h5["obs"].create_group("leiden")
        leiden.create_dataset("codes", data=np.array([0, 1, 1], dtype=np.int8))
        leiden.create_dataset("categories", data=np.array([b"a", b"b"]))
        h5.create_group("obsm").create_dataset(
            "X_umap", data=np.arange(6, dtype=np.float32).reshape(3, 2)
        )
    reader = H5adReader(
        str(source),
        feature_name_key="feature_name",
        embedding_roles={"X_umap": "umap"},
        cluster_keys=("leiden",),
    )
    try:
        writer = H5adToZarr(
            reader, zarr_loc=MemoryStore(), nthreads=1, shuffle_cells=False
        )
        first = writer.dump()
        second = writer.dump()
    finally:
        reader.close()

    assert second.cellSelection == first.cellSelection
    assert dict(second.clusterArtifacts) == dict(first.clusterArtifacts)
    assert dict(second.embeddingArtifacts) == dict(first.embeddingArtifacts)


def test_h5ad_dump_rejects_a_non_positive_batch_size(tmp_path) -> None:
    source = _write_h5ad(tmp_path / "cells.h5ad", np.eye(3, dtype=np.uint8))
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(reader, zarr_loc=MemoryStore(), nthreads=1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            writer.dump(batch_size=0)
    finally:
        reader.close()


def test_h5ad_cell_id_source_reads_ids_by_position(tmp_path) -> None:
    from cytearc.writers.h5ad import _H5adCellIdSource

    def text(values) -> list[str]:
        return [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in values
        ]

    source = _write_h5ad(tmp_path / "ids.h5ad", np.eye(3, dtype=np.uint8))
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        ids = _H5adCellIdSource(reader)
        assert len(ids) == 3
        assert text([ids[0], ids[-1]]) == ["c0", "c2"]
        assert text(ids[1:3]) == ["c1", "c2"]
        assert text(ids) == ["c0", "c1", "c2"]
        with pytest.raises(IndexError, match="^range object index out of range$"):
            ids[3]
        with pytest.raises(ValueError, match="only contiguous reads"):
            ids[::2]
        # Through a shuffled import's order, position r reads file row
        # order[r].
        shuffled = _H5adCellIdSource(reader, np.array([2, 0, 1]))
        assert text([shuffled[0], shuffled[-1]]) == ["c2", "c1"]
        assert text(shuffled[0:2]) == ["c2", "c0"]
        assert text(shuffled) == ["c2", "c0", "c1"]
        with pytest.raises(ValueError, match="only contiguous reads"):
            shuffled[::2]
    finally:
        reader.close()


class _Connection:
    """One end of a pipe that records what a process sends through it."""

    def __init__(self, on_send: Callable[[], object] | None = None) -> None:
        self.sent: list[tuple[str, object]] = []
        self.closed = False
        self._onSend = on_send

    def send(self, message: tuple[str, object]) -> None:
        self.sent.append(("message", message))
        if self._onSend is not None:
            self._onSend()

    def send_bytes(self, buffer: Any) -> None:
        self.sent.append(("bytes", bytes(buffer)))

    @property
    def messages(self) -> list[tuple[str, Any]]:
        """The messages as the import reads them, with their arrays."""
        return received_messages(self.sent)

    def close(self) -> None:
        self.closed = True


def test_h5ad_producer_sends_no_band_after_the_parent_stops(
    tmp_path, monkeypatch
) -> None:
    from cytearc.writers import h5ad as h5ad_writer

    source = _write_h5ad(
        tmp_path / "bands.h5ad", np.arange(24, dtype=np.uint8).reshape(8, 3)
    )
    # 12 bytes of uint8 counts make bands of four cells.
    plan = plan_count_matrix_pair(
        8, 3, np.uint8, policy=CountMatrixPolicy(unitBytes=12, chunkBytes=12)
    )
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:

        def produce(connection: _Connection, stop: threading.Event) -> None:
            h5ad_writer._read_h5ad_process_window(
                reader._clone_kwargs(),
                None,
                4,
                0,
                8,
                {"RNA": plan.counts},
                ("RNA",),
                None,
                connection,
                stop,
                {},
            )

        # A stop that arrives while a band is sent ends the window after it.
        stop = threading.Event()
        connection = _Connection(on_send=stop.set)
        produce(connection, stop)
        assert [kind for kind, _payload in connection.messages] == ["band"]
        assert connection.closed

        # A stop that arrives while a band is cut keeps that band unsent.
        stop = threading.Event()
        cut_bands = h5ad_writer._count_matrix_bands

        def stop_while_cutting(*args):
            stop.set()
            yield from cut_bands(*args)

        monkeypatch.setattr(h5ad_writer, "_count_matrix_bands", stop_while_cutting)
        connection = _Connection()
        produce(connection, stop)
        assert connection.messages == []
        assert connection.closed
    finally:
        reader.close()


def test_dense_h5ad_producer_sends_no_band_after_the_parent_stops(
    tmp_path, monkeypatch
) -> None:
    from cytearc.writers import h5ad as h5ad_writer

    values = np.arange(24, dtype=np.uint8).reshape(8, 3)
    source = _write_h5ad(tmp_path / "dense.h5ad", values, encoding="dense")
    # 12 bytes of uint8 counts make bands of four cells.
    plan = plan_count_matrix_pair(
        8, 3, np.uint8, policy=CountMatrixPolicy(unitBytes=12, chunkBytes=12)
    )
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:

        def produce(stop: threading.Event, connection: _Connection) -> list:
            # Batches of four cells, each of which fills a band.
            h5ad_writer._read_dense_h5ad_process_window(
                reader._clone_kwargs(),
                4,
                0,
                8,
                {"RNA": plan.counts},
                None,
                connection,
                stop,
                {},
            )
            assert connection.closed
            return [
                (payload[1].start, payload[1].end)
                for kind, payload in connection.messages
                if kind == "band"
            ]

        # A window that is not stopped sends each band of its rows, then done.
        complete = _Connection()
        assert produce(threading.Event(), complete) == [(0, 4), (4, 8)]
        assert [kind for kind, _payload in complete.messages] == [
            "band",
            "band",
            "done",
        ]
        np.testing.assert_array_equal(
            np.vstack([band.values for _kind, (_name, band) in complete.messages[:2]]),
            values,
        )

        # A stop that arrives while a band is sent ends the window after it,
        # before the window reads its next batch.
        stop = threading.Event()
        read_batches = H5adReader.consume_dense_rows
        batch_rows: list[int] = []

        def count_batches(self, *args, **kwargs):
            for batch in read_batches(self, *args, **kwargs):
                batch_rows.append(len(batch))
                yield batch

        with monkeypatch.context() as patch:
            patch.setattr(H5adReader, "consume_dense_rows", count_batches)
            connection = _Connection(on_send=stop.set)
            assert produce(stop, connection) == [(0, 4)]
        assert [kind for kind, _payload in connection.messages] == ["band"]
        assert batch_rows == [4]

        # A stop that arrives while the next batch is read ends the window
        # before that batch fills a band.
        stop = threading.Event()

        def stop_after_the_first_batch(self, *args, **kwargs):
            for batch in read_batches(self, *args, **kwargs):
                yield batch
                stop.set()

        with monkeypatch.context() as patch:
            patch.setattr(H5adReader, "consume_dense_rows", stop_after_the_first_batch)
            connection = _Connection()
            assert produce(stop, connection) == [(0, 4)]
        assert [kind for kind, _payload in connection.messages] == ["band"]

        # A stop that arrives while a band is cut keeps that band unsent.
        stop = threading.Event()
        cut_bands = h5ad_writer._dense_count_bands

        def stop_while_cutting(*args):
            for item in cut_bands(*args):
                stop.set()
                yield item

        monkeypatch.setattr(h5ad_writer, "_dense_count_bands", stop_while_cutting)
        connection = _Connection()
        assert produce(stop, connection) == []
        assert connection.messages == []
    finally:
        reader.close()


def test_dense_h5ad_writer_reports_done_only_when_the_parent_did_not_stop(
    tmp_path,
) -> None:
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.sharding import plan_dense_band_writes
    from cytearc.writers import h5ad as h5ad_writer

    values = np.arange(1, 25, dtype=np.uint8).reshape(8, 3)
    source = _write_h5ad(tmp_path / "dense.h5ad", values, encoding="dense")
    location = str(tmp_path / "dense.zarr")
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        # Constructing the writer creates the counts that the window fills.
        counts = H5adToZarr(reader, location, nthreads=1, shuffle_cells=False).z[
            "RNA/counts"
        ]
        plan = plan_dense_band_writes((counts,), ResourceBudget(1024**2, 1), 2)

        def write(stop: Any) -> list:
            counts[:] = 0
            connection = _Connection()
            h5ad_writer._write_dense_h5ad_process_window(
                reader._clone_kwargs(),
                4,
                0,
                8,
                location,
                None,
                None,
                ("RNA",),
                None,
                plan,
                connection,
                stop,
                {},
            )
            assert connection.closed
            return connection.messages

        # A window that is not stopped writes its rows, then reports done.
        ((kind, _payload),) = write(threading.Event())
        assert kind == "done"
        np.testing.assert_array_equal(counts[:], values)

        class _StopOnceWritten:
            """A stop that the parent sets once every row is written."""

            def is_set(self) -> bool:
                return bool(np.array_equal(counts[:], values))

        # A stop that arrives after the last write suppresses done.
        assert write(_StopOnceWritten()) == []
        np.testing.assert_array_equal(counts[:], values)
    finally:
        reader.close()


def _watch_dense_bands(monkeypatch) -> HandedArrays:
    """Watch the values of the bands that dense band buffers hand over.

    Whenever a buffer takes another batch, it first waits until the bands
    handed over so far are freed.
    """
    from cytearc.storage import sharding

    handed = HandedArrays()
    add = sharding.DenseBandBuffer.add

    def hand(band: Any) -> Any:
        handed.hand(band.values)
        return band

    def watching_add(self, values):
        handed.wait_until_freed()
        # map hands each band on and keeps none of them.
        yield from map(hand, add(self, values))

    monkeypatch.setattr(sharding.DenseBandBuffer, "add", watching_add)
    return handed


def test_a_dense_h5ad_import_frees_each_band_it_wrote_before_it_reads_more(
    tmp_path, monkeypatch
) -> None:
    values = np.arange(1, 33, dtype=np.uint8).reshape(8, 4)
    source = _write_h5ad(
        tmp_path / "dense.h5ad",
        values,
        encoding="dense",
        feature_types=[b"Gene Expression", b"Antibody Capture"] * 2,
    )
    handed = _watch_dense_bands(monkeypatch)
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        # 4 bytes of uint8 counts of two features make bands of two cells in
        # each assay, which the batches that the write reads fill.
        H5adToZarr(
            reader,
            str(tmp_path / "dense.zarr"),
            assay_split_key="feature_types",
            nthreads=1,
            shuffle_cells=False,
            policy=CountMatrixPolicy(unitBytes=4, chunkBytes=4),
        ).dump(batch_size=2)
    finally:
        reader.close()
    root = zarr.open_group(str(tmp_path / "dense.zarr"), mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, [0, 2]])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, [1, 3]])
    assert len(handed) == 8
    # Each band was written, and freed, before the next part of a batch was
    # buffered, in the other assay or in the next batch.
    assert len(handed.freed) >= len(handed)
    assert handed.freed == [True] * len(handed.freed)


class _PicklingConnection(_Connection):
    """A connection that keeps copies of what it sends, as a pipe pickles it."""

    def send(self, message: tuple[str, object]) -> None:
        import pickle

        super().send(pickle.loads(pickle.dumps(message)))


@pytest.mark.parametrize("window", ["writes its bands", "sends its bands"])
def test_a_dense_h5ad_producer_process_frees_each_band_before_its_next_batch(
    tmp_path, monkeypatch, window
) -> None:
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.sharding import plan_dense_band_writes
    from cytearc.writers import h5ad as h5ad_writer

    values = np.arange(1, 25, dtype=np.uint8).reshape(8, 3)
    source = _write_h5ad(tmp_path / "dense.h5ad", values, encoding="dense")
    location = str(tmp_path / "dense.zarr")
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        # 12 bytes of uint8 counts make bands of four cells, and constructing
        # the writer creates the counts that the window fills.
        counts = H5adToZarr(
            reader,
            location,
            nthreads=1,
            shuffle_cells=False,
            policy=CountMatrixPolicy(unitBytes=12, chunkBytes=12),
        ).z["RNA/counts"]
        handed = _watch_dense_bands(monkeypatch)
        connection = _PicklingConnection()
        # Each process runs here, reading the window's batches of four cells.
        if window == "writes its bands":
            h5ad_writer._write_dense_h5ad_process_window(
                reader._clone_kwargs(),
                4,
                0,
                8,
                location,
                None,
                None,
                ("RNA",),
                None,
                plan_dense_band_writes((counts,), ResourceBudget(1024**2, 1), 2),
                connection,
                threading.Event(),
                {},
            )
            np.testing.assert_array_equal(counts[:], values)
        else:
            h5ad_writer._read_dense_h5ad_process_window(
                reader._clone_kwargs(),
                4,
                0,
                8,
                {"RNA": counts},
                None,
                connection,
                threading.Event(),
                {},
            )
            np.testing.assert_array_equal(
                np.vstack(
                    [band.values for kind, (_name, band) in connection.messages[:2]]
                ),
                values,
            )
    finally:
        reader.close()
    assert [kind for kind, _payload in connection.messages][-1] == "done"
    assert len(handed) == 2
    assert handed.freed == [True, True]


@pytest.mark.parametrize("encoding", ["dense", "csr"])
def test_an_h5ad_import_frees_each_band_it_received_once_its_write_has_it(
    tmp_path, monkeypatch, encoding
) -> None:
    from multiprocessing import connection as pipes

    from cytearc.storage import sharding
    from cytearc.storage.io_policy import StorageIoPolicy

    values = np.arange(1, 4_001, dtype=np.float32).reshape(1_000, 4)
    source = _write_h5ad(tmp_path / "counts.h5ad", values, encoding=encoding)
    dense = encoding == "dense"
    handed = HandedArrays()
    name = "write_dense_bands" if dense else "write_sparse_bands"
    write = getattr(sharding, name)

    class Handing:
        """Hands each band that the import received to the writes, keeping none."""

        def __init__(self, writes: Any) -> None:
            self.writes = iter(writes)

        def __iter__(self) -> "Handing":
            return self

        def __next__(self) -> Any:
            item = next(self.writes)
            handed.hand(item.band.values if dense else item.band.row)
            return item

    receive = pipes.Connection.recv

    def watching_receive(self):
        # The import receives the next message from a producer process once
        # the bands handed to the writes so far are freed.
        handed.wait_until_freed()
        return receive(self)

    monkeypatch.setattr(
        sharding, name, lambda writes, **options: write(Handing(writes), **options)
    )
    monkeypatch.setattr(pipes.Connection, "recv", watching_receive)
    reader = H5adReader(str(source), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        # A destination in memory has producer processes send their bands to
        # this process; of three workers two read and one writes, each band
        # before the next is pulled. 400 bytes of the uint16 counts of four
        # features make bands of 50 cells, one for each batch.
        writer = H5adToZarr(
            reader,
            store,
            mem_budget="1G",
            nthreads=3,
            io=StorageIoPolicy(readWorkers=2),
            shuffle_cells=False,
            policy=CountMatrixPolicy(unitBytes=400, chunkBytes=400),
        )
        writer.dump(batch_size=50)
    finally:
        reader.close()
    assert (writer._lastImportProducerCount, writer._lastImportWriteWorkers) == (2, 1)
    np.testing.assert_array_equal(
        zarr.open_group(store=store, mode="r")["RNA/counts"][:], values
    )
    assert len(handed) == 20
    assert len(handed.freed) > len(handed)
    assert handed.freed == [True] * len(handed.freed)


def _complex_matrix(wire: _Wire, values: list[float]) -> bytes:
    """Encode an R complex matrix of two genes and the three test cells."""
    return (
        wire.integer(wire.flags(RType.COMPLEX, attributes=True))
        + wire.integer(len(values))
        + b"".join(wire.real(value) + wire.real(0.0) for value in values)
        + wire.attributes(
            [
                ("dim", wire.integer_vector([2, len(_CELLS)])),
                ("dimnames", wire.dimnames(["g1", "g2"], _CELLS)),
                ("class", wire.string_vector(["matrix", "array"])),
            ]
        )
    )


def _write_seurat(
    path: Path,
    wire: _Wire,
    *,
    assay: bytes | None = None,
    cell_columns: list[tuple[str, bytes]] | None = None,
    reductions: list[tuple[str, bytes]] | None = None,
) -> Path:
    selected = reductions or []
    root = wire.s4(
        [
            ("assays", wire.vector([assay or _legacy_assay(wire)], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    cell_columns or [("group", wire.string_vector(["a", "b", "c"]))],
                    _CELLS,
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1, 2, 1], ["x", "y"], names=_CELLS)),
            (
                "reductions",
                wire.vector(
                    [node for _name, node in selected],
                    names=[name for name, _node in selected],
                ),
            ),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _assay(wire: _Wire, counts: bytes, feature_column: str = "symbol") -> bytes:
    return wire.s4(
        [
            ("counts", counts),
            (
                "meta.features",
                wire.data_frame(
                    [(feature_column, wire.string_vector(["G1", "G2"]))],
                    ["g1", "g2"],
                ),
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )


def test_seurat_complex_counts_fail_before_the_destination_exists(tmp_path) -> None:
    wire = _Wire()
    counts = _complex_matrix(wire, [1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
    path = _write_seurat(tmp_path / "complex.rds", wire, assay=_assay(wire, counts))
    destination = _sentinel_store()
    with SeuratReader(path, reductions=[]) as reader:
        with pytest.raises(TypeError, match="counts use unsupported dtype complex128"):
            SeuratToZarr(reader, destination, nthreads=1)
    assert _untouched(destination)


@pytest.mark.parametrize(
    ("cell_column", "feature_column", "message"),
    [
        (".", "symbol", "cell metadata column '.' cannot name a Zarr array"),
        ("group", "..", "RNA feature metadata column '..' cannot name a Zarr array"),
    ],
)
def test_seurat_metadata_names_must_name_an_array(
    tmp_path, cell_column, feature_column, message
) -> None:
    wire = _Wire()
    counts = wire.matrix([1, 0, 0, 2, 3, 0], (2, 3), rows=["g1", "g2"], columns=_CELLS)
    path = _write_seurat(
        tmp_path / "names.rds",
        wire,
        assay=_assay(wire, counts, feature_column),
        cell_columns=[(cell_column, wire.string_vector(["a", "b", "c"]))],
    )
    destination = _sentinel_store()
    with SeuratReader(path, reductions=[]) as reader:
        with pytest.raises(ValueError, match=re.escape(message)):
            SeuratToZarr(reader, destination, nthreads=1)
    assert _untouched(destination)


def test_seurat_repeated_dump_reuses_its_artifacts(tmp_path) -> None:
    wire = _Wire()
    pca = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    (3, 2),
                    rows=_CELLS,
                    columns=["PC_1", "PC_2"],
                    real=True,
                ),
            ),
            ("feature.loadings", wire.matrix([], (0, 0), real=True)),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([0])),
            ("stdev", wire.real_vector([])),
            ("key", wire.string_vector(["PC_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    path = _write_seurat(tmp_path / "twice.rds", wire, reductions=[("pca", pca)])
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        writer = SeuratToZarr(reader, destination, nthreads=1, shuffle_cells=False)
        first = writer.dump()
        second = writer.dump()

    assert second.cellSelection == first.cellSelection
    assert second.activeIdentity == first.activeIdentity
    assert dict(second.reductionArtifacts) == dict(first.reductionArtifacts)
    # A reduction without stdev stores none.
    stored = artifact_group(
        zarr.open_group(store=destination, mode="r"),
        first.reductionArtifacts["pca"],
    )
    np.testing.assert_array_equal(stored["data"][:], [[1, 4], [2, 5], [3, 6]])
    assert "stdev" not in stored


def test_seurat_sparse_assay_without_cells_imports(tmp_path) -> None:
    wire = _Wire()
    counts = wire.s4(
        [
            ("i", wire.integer_vector([])),
            ("p", wire.integer_vector([0])),
            ("Dim", wire.integer_vector([2, 0])),
            ("Dimnames", wire.dimnames(["g1", "g2"], [])),
            ("x", wire.real_vector([])),
            ("factors", wire.vector([])),
            ("class", wire.string_vector(["dgCMatrix"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([_assay(wire, counts)], names=["RNA"])),
            ("meta.data", wire.data_frame([("group", wire.string_vector([]))], [])),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([], ["cells"], names=[])),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path = tmp_path / "empty.rds"
    path.write_bytes(wire.document(root))
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        assert reader.get_assay("RNA").counts.is_sparse
        SeuratToZarr(reader, destination, nthreads=1).dump()

    counts_array = zarr.open_group(store=destination, mode="r")["RNA/counts"]
    assert (counts_array.shape, counts_array.dtype) == ((0, 2), np.uint8)


def test_sparse_writer_rejects_ids_that_do_not_match_the_matrix() -> None:
    matrix = csr_matrix(np.eye(3, dtype=np.uint8))
    destination = _sentinel_store()
    with pytest.raises(ValueError, match="Number of cell ids"):
        SparseToZarr(matrix, destination, ["c1", "c2"], ["f1", "f2", "f3"])
    with pytest.raises(ValueError, match="Number of feature ids"):
        SparseToZarr(matrix, destination, ["c1", "c2", "c3"], ["f1"])
    assert _untouched(destination)


def test_sparse_writer_batches_write_every_cell() -> None:
    values = np.arange(1, 16, dtype=np.uint8).reshape(5, 3)
    store = MemoryStore()
    writer = SparseToZarr(
        csr_matrix(values),
        store,
        [f"c{index}" for index in range(5)],
        ["f1", "f2", "f3"],
        nthreads=1,
        shuffle_cells=False,
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        writer.dump(batch_size=0)
    writer.dump(batch_size=2)

    # Batches of two cells leave one cell for the last batch.
    assert writer._lastImportPlan.batchRows == 2
    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    np.testing.assert_array_equal(counts[:], values)


class _CountsReader:
    """A Cell Ranger style reader of small in-memory counts."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        assay: str = "RNA",
        cell_columns: tuple[tuple[str, np.ndarray], ...] = (),
        feature_columns: tuple[tuple[str, np.ndarray], ...] = (),
    ) -> None:
        self._values = values
        self.nCells, self.nFeatures = values.shape
        self.matrix_dtype = values.dtype
        self.assayFeats = pd.DataFrame(
            {assay: ["Gene Expression", 0, self.nFeatures, self.nFeatures]},
            index=["type", "start", "end", "nFeatures"],
        )
        self._cellColumns = cell_columns
        self._featureColumns = feature_columns

    def cell_names(self) -> list[str]:
        return [f"c{index}" for index in range(self.nCells)]

    # A cell table's write reads the names that the reader makes.
    _cell_names_source = CrReader._cell_names_source

    def feature_ids(self, assay_name: str) -> list[str]:
        return [f"f{index}" for index in range(self.nFeatures)]

    def feature_names(self, assay_name: str) -> list[str]:
        return [f"g{index}" for index in range(self.nFeatures)]

    def consume(self, batch_size: int, lines_in_mem: int) -> Iterator[coo_matrix]:
        for start in range(0, self.nCells, batch_size):
            yield coo_matrix(self._values[start : start + batch_size])

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        value_range = CountValueRange()
        value_range.update(self._values)
        return [value_range]

    def max_window_nnz(self, window_rows: int) -> int:
        return int(np.count_nonzero(self._values))

    def producer_staging_bytes(self, batch_size: int, lines_in_mem: int) -> int:
        return 0

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._cellColumns

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._featureColumns


def test_cellranger_assay_types_must_name_imported_assays_and_presets() -> None:
    reader = _CountsReader(np.eye(3, dtype=np.uint8))
    destination = _sentinel_store()
    with pytest.raises(ValueError, match="names assays that are not imported: GEX"):
        CrToZarr(reader, zarr_loc=destination, assay_types={"GEX": "RNA"})
    # The error names the assay, as every writer's does.
    with pytest.raises(ValueError, match="assay_type 'rna' of assay 'RNA' is not a"):
        CrToZarr(reader, zarr_loc=destination, assay_types={"RNA": "rna"})
    assert _untouched(destination)


def test_cellranger_assay_type_makes_a_custom_assay_rna() -> None:
    values = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint8)
    store = MemoryStore()
    writer = CrToZarr(
        _CountsReader(values, assay="GEX"),
        zarr_loc=store,
        nthreads=1,
        assay_types={"GEX": "RNA"},
        shuffle_cells=False,
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        writer.dump(batch_size=0)
    writer.dump()

    root = zarr.open_group(store=store, mode="r")
    assert root.attrs["assayTypes"] == {"GEX": "RNA"}
    np.testing.assert_array_equal(root["GEX/countsT"][:], values.T)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (
            {"cell_columns": (("batch", np.array(["a", "b"])),)},
            r"Cell metadata column 'batch' has shape \(2,\); expected \(3,\)",
        ),
        (
            {"feature_columns": (("tag", np.zeros((3, 2))),)},
            r"Feature metadata column 'tag' has shape \(3, 2\); expected \(3,\)",
        ),
    ],
)
def test_cellranger_reader_metadata_must_align_with_its_axis(columns, message) -> None:
    reader = _CountsReader(np.eye(3, dtype=np.uint8), **columns)
    with pytest.raises(ValueError, match=message):
        CrToZarr(reader, zarr_loc=MemoryStore(), nthreads=1, shuffle_cells=False)


_IMPORTED = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint8)

type _Importer = Callable[[Any, ExitStack, Path, bool], Any]


def _sparse(destination: Any, _stack: ExitStack, _tmp: Path, overwrite: bool) -> Any:
    return SparseToZarr(
        csr_matrix(_IMPORTED),
        destination,
        _CELLS,
        ["f1", "f2", "f3"],
        nthreads=1,
        overwrite=overwrite,
        shuffle_cells=False,
    )


def _csv(destination: Any, _stack: ExitStack, tmp: Path, overwrite: bool) -> Any:
    path = tmp / "counts.csv"
    path.write_text("g1,g2,g3\n1,0,2\n0,3,0\n4,0,5\n")
    return CSVtoZarr(
        CSVReader(str(path)),
        destination,
        assay_name="RNA",
        nthreads=1,
        overwrite=overwrite,
        shuffle_cells=False,
    )


def _cellranger(
    destination: Any, _stack: ExitStack, _tmp: Path, overwrite: bool
) -> Any:
    return CrToZarr(
        _CountsReader(_IMPORTED),
        zarr_loc=destination,
        nthreads=1,
        overwrite=overwrite,
        shuffle_cells=False,
    )


def _h5ad(destination: Any, stack: ExitStack, tmp: Path, overwrite: bool) -> Any:
    reader = H5adReader(
        str(_write_h5ad(tmp / "counts.h5ad", _IMPORTED)),
        feature_name_key="feature_name",
    )
    stack.callback(reader.close)
    return H5adToZarr(
        reader,
        zarr_loc=destination,
        nthreads=1,
        overwrite=overwrite,
        shuffle_cells=False,
    )


def _seurat(destination: Any, stack: ExitStack, tmp: Path, overwrite: bool) -> Any:
    path = _write_seurat(tmp / "counts.rds", _Wire())
    reader = stack.enter_context(SeuratReader(path, reductions=[]))
    return SeuratToZarr(
        reader, destination, nthreads=1, overwrite=overwrite, shuffle_cells=False
    )


_IMPORTERS: dict[str, _Importer] = {
    "sparse": _sparse,
    "csv": _csv,
    "cellranger": _cellranger,
    "h5ad": _h5ad,
    "seurat": _seurat,
}


def _unread(*_args: Any) -> Any:
    raise AssertionError("the source counts were read before the destination check")


@pytest.mark.parametrize("importer", sorted(_IMPORTERS))
def test_importers_replace_a_store_only_with_overwrite(
    tmp_path, monkeypatch, importer
) -> None:
    # Before, an import opened its destination with mode "w", which deleted
    # whatever it held.
    destination = MemoryStore()
    with ExitStack() as stack:
        _sparse(destination, stack, tmp_path, False).dump()
    before = _store_bytes(destination)
    with (
        ExitStack() as stack,
        monkeypatch.context() as patch,
        pytest.raises(FileExistsError, match="is not empty"),
    ):
        # Importers that pass over their source refuse the destination first.
        for writer in ("cellranger", "csv", "h5ad", "seurat", "sparse"):
            patch.setattr(f"cytearc.writers.{writer}.count_storage_dtype", _unread)
        _IMPORTERS[importer](destination, stack, tmp_path, False)
    assert _store_bytes(destination) == before

    with ExitStack() as stack:
        _IMPORTERS[importer](destination, stack, tmp_path, True).dump()
    root = zarr.open_group(store=destination, mode="r")
    assert root["RNA/counts"].attrs["complete"] is True


_SUBSET_COUNTS = np.array(
    [[1, 4, 9], [2, 20, 3], [12, 2, 2], [5, 0, 1]], dtype=np.uint16
)


def _subset_source(location: str | MemoryStore) -> DataStore:
    SparseToZarr(
        csr_matrix(_SUBSET_COUNTS),
        location,
        ["c1", "c2", "c3", "c4"],
        ["f1", "f2", "f3"],
        nthreads=1,
        shuffle_cells=False,
    ).dump()
    return DataStore(location, default_assay="RNA", min_features_per_cell=0, nthreads=1)


def test_subset_refuses_its_source_even_when_it_is_not_prepared(tmp_path) -> None:
    """The overlap check protects a source that no DataStore prepared.

    An assay built directly over imported counts is not prepared, so the
    refusal to replace a prepared store does not protect it.
    """
    on_disk = str(tmp_path / "source.zarr")
    dataset = _subset_source(on_disk)
    dataset.z["RNA"].attrs["prepared"] = False
    for destination in (on_disk, LocalStore(on_disk)):
        with pytest.raises(ValueError, match="overlaps a source store"):
            SubsetZarr(
                destination,
                assays=[dataset.RNA],
                cell_ids=["c1"],
                overwrite_existing_file=True,
                nthreads=1,
            )
    np.testing.assert_array_equal(dataset.RNA.rawData.compute(), _SUBSET_COUNTS)


def test_subset_refuses_a_destination_that_overlaps_its_source(tmp_path) -> None:
    on_disk = str(tmp_path / "source.zarr")
    dataset = _subset_source(on_disk)
    with pytest.raises(ValueError, match="overlaps a source store"):
        SubsetZarr(
            f"{on_disk}/nested.zarr",
            assays=[dataset.RNA],
            cell_ids=["c1"],
            nthreads=1,
        )

    in_memory = MemoryStore()
    dataset = _subset_source(in_memory)
    with pytest.raises(ValueError, match="overlaps a source store"):
        SubsetZarr(
            in_memory,
            assays=[dataset.RNA],
            cell_ids=["c1"],
            overwrite_existing_file=True,
            nthreads=1,
        )
    # Before, a second store object over the source's keys was replaced, which
    # emptied the source. The source is prepared, so overwrite refuses it.
    with pytest.raises(FileExistsError, match=r"holds the prepared assays \['RNA'\]"):
        SubsetZarr(
            MemoryStore(in_memory._store_dict),
            assays=[dataset.RNA],
            cell_ids=["c1"],
            overwrite_existing_file=True,
            nthreads=1,
        )
    np.testing.assert_array_equal(dataset.RNA.rawData.compute(), _SUBSET_COUNTS)


@pytest.mark.parametrize(
    ("cell_ids", "error", "message"),
    [
        # A repeated cell would write a store whose cell IDs repeat.
        (["c2", "c3", "c2"], ValueError, "cell_ids repeats 1 cell IDs, such as 'c2'"),
        (np.array([], dtype=str), ValueError, "cell_ids is empty"),
        ([], ValueError, "cell_ids is empty"),
        (
            ["c1", "c9", "c8"],
            ValueError,
            "2 of the 3 values of cell_ids are not cell IDs of the store, such as 'c9'",
        ),
        # Row positions, which the replaced cell_idx took, are not cell IDs.
        (np.array([0, 2]), ValueError, "such as 0; pass IDs such as those of"),
        ("c1", TypeError, "one-dimensional sequence of cell IDs"),
        ([["c1"], ["c2"]], TypeError, r"of shape \(2, 1\)"),
    ],
    ids=["repeated", "empty", "empty-list", "unknown", "positions", "text", "2d"],
)
def test_subset_rejects_invalid_cell_ids(cell_ids, error, message) -> None:
    dataset = _subset_source(MemoryStore())
    destination = _sentinel_store()
    with pytest.raises(error, match=message):
        SubsetZarr(
            destination,
            assays=[dataset.RNA],
            cell_ids=cell_ids,
            overwrite_existing_file=True,
            nthreads=1,
        )
    assert _untouched(destination)


def test_subset_rejects_cell_key_with_cell_ids() -> None:
    # One selector would otherwise be ignored without a word.
    dataset = _subset_source(MemoryStore())
    destination = _sentinel_store()
    with pytest.raises(ValueError, match="by cell_key or by cell_ids, not both"):
        SubsetZarr(
            destination,
            assays=[dataset.RNA],
            cell_key="I",
            cell_ids=["c1"],
            overwrite_existing_file=True,
            nthreads=1,
        )
    assert _untouched(destination)


@pytest.mark.parametrize(
    "cell_ids",
    [["c4", "c1", "c3"], np.array(["c3", "c4", "c1"]), pd.Index(["c1", "c3", "c4"])],
    ids=["reversed-list", "shuffled-array", "sorted-index"],
)
def test_subset_by_cell_ids_keeps_store_order(cell_ids) -> None:
    dataset = _subset_source(MemoryStore())
    destination = MemoryStore()

    SubsetZarr(destination, assays=[dataset.RNA], cell_ids=cell_ids, nthreads=1).dump()

    root = zarr.open_group(store=destination, mode="r")
    # Whatever order the IDs come in, the subset holds rows 0, 2 and 3.
    np.testing.assert_array_equal(root["cellData/ids"][:], ["c1", "c3", "c4"])
    np.testing.assert_array_equal(root["RNA/counts"][:], _SUBSET_COUNTS[[0, 2, 3]])
    np.testing.assert_array_equal(root["RNA/countsT"][:], _SUBSET_COUNTS[[0, 2, 3]].T)


def test_subset_cell_ids_refuse_ids_that_the_store_repeats() -> None:
    dataset = _subset_source(MemoryStore())
    ids = dataset.cells.locations["primary"]["ids"]
    ids[:] = np.array(["c1", "c2", "c1", "c4"])
    destination = _sentinel_store()

    with pytest.raises(ValueError, match="holds cell ID 'c1' in more than one row"):
        SubsetZarr(destination, assays=[dataset.RNA], cell_ids=["c1"], nthreads=1)
    assert _untouched(destination)
    # IDs that the store holds once still select their cells.
    writer = SubsetZarr(MemoryStore(), assays=[dataset.RNA], cell_ids=["c4", "c2"])
    np.testing.assert_array_equal(writer.cellIdx, [1, 3])


def test_subset_by_a_cell_key_without_selected_cells_writes_no_cells() -> None:
    # A key is data, not an index argument: one that selects no cells writes
    # a store without cells, as an import of a source without cells does.
    dataset = _subset_source(MemoryStore())
    dataset.cells.insert("none", by_cell_id(dataset.cells, np.zeros(4, dtype=bool)))
    destination = MemoryStore()
    SubsetZarr(destination, assays=[dataset.RNA], cell_key="none", nthreads=1).dump()

    root = zarr.open_group(store=destination, mode="r")
    assert root["cellData/ids"].shape == (0,)
    assert root["RNA/counts"].shape == (0, 3)
    assert root["RNA/countsT"].shape == (3, 0)
    # The features are kept, and both orientations are complete, so the store
    # opens as one without cells.
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], ["f1", "f2", "f3"])
    assert root["RNA/counts"].attrs["complete"] is True
    assert root["RNA/countsT"].attrs["complete"] is True
    subset = DataStore(
        destination, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    assert (subset.cells.N, subset.assay_names, subset.RNA.feats.N) == (0, ["RNA"], 3)


@pytest.mark.parametrize(
    ("reset", "expected"),
    [(True, [True, True, True]), (False, [True, False, True])],
)
def test_subset_keeps_the_source_cell_filter_unless_reset(reset, expected) -> None:
    dataset = _subset_source(MemoryStore())
    dataset.cells.update_key(
        by_cell_id(dataset.cells, np.array([True, False, True, False])), key="I"
    )
    destination = MemoryStore()
    SubsetZarr(
        destination,
        assays=[dataset.RNA],
        cell_ids=["c1", "c2", "c3"],
        reset_cell_filter=reset,
        nthreads=1,
    ).dump()

    cells = zarr.open_group(store=destination, mode="r")["cellData"]
    np.testing.assert_array_equal(cells["I"][:], expected)
    np.testing.assert_array_equal(cells["ids"][:], ["c1", "c2", "c3"])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (2, "Dense stream contains 2 rows, expected 4"),
        (5, "Dense stream contains more rows than its destination"),
    ],
)
def test_csv_dump_refuses_a_file_whose_rows_changed(tmp_path, rows, message) -> None:
    path = tmp_path / "counts.csv"

    def write(n_rows: int) -> None:
        path.write_text(
            "g1,g2\n" + "".join(f"{row},{row + 1}\n" for row in range(n_rows))
        )

    write(4)
    destination = str(tmp_path / "counts.zarr")
    writer = CSVtoZarr(
        CSVReader(str(path), batch_size=2),
        destination,
        assay_name="RNA",
        nthreads=1,
        shuffle_cells=False,
    )
    write(rows)
    with pytest.raises(ValueError, match=message):
        writer.dump()
    counts = zarr.open_group(destination, mode="r")["RNA/counts"]
    assert counts.attrs["complete"] is False
