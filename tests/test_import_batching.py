from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from zarr.storage import MemoryStore

from cytearc.readers import CrH5Reader, H5adReader
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.layout import ZarrArraySpec, array_shard_rows, get_compressors
from cytearc.storage.sharding import (
    resolve_sparse_import_batch,
    resolve_sparse_import_spec,
    row_band_task_count,
)
from cytearc.storage.types import array_metadata_shards
from cytearc.utils.count_values import CountValueRange
from cytearc.writers import CrToZarr, H5adToZarr, SparseToZarr


def sparse_write_task_count(destinations, n_rows: int) -> int:
    """Count the destination row bands an import writes, as the planner should."""
    if n_rows <= 0:
        return 0
    return sum(
        row_band_task_count(n_rows, array_shard_rows(destination))
        for destination in destinations
    )


def _planner_destinations() -> tuple[zarr.Array, zarr.Array]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    first = root.create_array(
        "first",
        shape=(12, 4),
        chunks=(2, 4),
        shards=(4, 4),
        dtype=np.uint32,
    )
    second = root.create_array(
        "second",
        shape=(12, 2),
        chunks=(3, 2),
        shards=(6, 2),
        dtype=np.uint32,
    )
    return first, second


def test_sparse_import_spec_matches_physical_array_plan() -> None:
    first, second = _planner_destinations()
    resources = ResourceBudget(memoryBytes=2_000_000, workers=2)

    def max_window_nnz(rows: int) -> int:
        return max(0, int(rows)) * 3

    array_plan = resolve_sparse_import_batch(
        (first, second),
        nRows=12,
        resources=resources,
        maxWindowNnz=max_window_nnz,
        sourceDtype=np.uint32,
    )
    specs = []
    for array in (first, second):
        shards = array_metadata_shards(array)
        specs.append(
            ZarrArraySpec(
                shape=tuple(int(value) for value in array.shape),
                chunks=tuple(int(value) for value in array.chunks),
                dtype=array.dtype,
                compressors=get_compressors("fast_local"),
                shards=(
                    None if shards is None else tuple(int(value) for value in shards)
                ),
                fillValue=0,
            )
        )
    spec_plan = resolve_sparse_import_spec(
        tuple(specs),
        nRows=12,
        resources=resources,
        maxWindowNnz=max_window_nnz,
        sourceDtype=np.uint32,
    )
    assert spec_plan == array_plan


def test_sparse_import_planner_prefers_geometry_and_shrinks_monotonically() -> None:
    destinations = _planner_destinations()
    large = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(13_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 4,
        sourceDtype=np.uint32,
        producerStagingBytes=lambda rows: rows * 1_000,
    )
    small = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(10_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 4,
        sourceDtype=np.uint32,
        producerStagingBytes=lambda rows: rows * 1_000,
    )
    wider = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(13_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 8,
        sourceDtype=np.uint32,
        producerStagingBytes=lambda rows: rows * 1_000,
    )

    assert large.batchRows == 4
    assert large.producerReserveBytes == 8_272
    assert large.writeTasks == 5
    assert small.batchRows == 2
    assert small.producerReserveBytes < large.producerReserveBytes
    assert wider.batchRows < large.batchRows


def test_sparse_admission_checks_the_pull_and_write_phases_separately() -> None:
    from cytearc.storage.sharding import (
        _band_geometry,
        _row_band_task_peak,
        sparse_producer_peak_bytes,
    )

    destination = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "counts", shape=(64, 8), chunks=(16, 8), shards=(16, 8), dtype=np.float32
    )

    def window_nnz(rows: int) -> int:
        return min(rows, 64) * 8

    def admits(memory: int) -> bool:
        try:
            resolve_sparse_import_batch(
                (destination,),
                nRows=64,
                resources=ResourceBudget(memory, 1),
                maxWindowNnz=window_nnz,
                sourceDtype=np.float32,
                batchRows=1,
            )
        except MemoryError:
            return False
        return True

    low, high = 0, 1_000_000
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if admits(middle) else (middle, high)

    # The in-process producer buffers one source row and the next band while
    # no band is written, and holds only its source row while a band is.
    buffering = sparse_producer_peak_bytes(window_nnz(1 + 16), window_nnz(1), 4)
    held = sparse_producer_peak_bytes(0, window_nnz(1), 4)
    band = window_nnz(16) * (4 + 2 * 8)
    dense, inner, chunks = _band_geometry(destination)
    task = _row_band_task_peak(
        sourceBytes=0,
        denseBytes=dense,
        innerChunkBytes=inner,
        nChunks=chunks,
        innerConcurrency=1,
    )
    assert high == max(buffering + 1, held + band + task)
    # Summing the buffering peak into the write task would refuse this budget.
    assert high < buffering + band + task


@pytest.mark.parametrize(
    ("columns", "dtype", "memory_bytes", "reserve_bytes"),
    [
        # The producer's buffering peak leaves no room for a second pending
        # band.
        (3, np.uint16, 60_000, 59_900),
        # A second pending band fits beside the buffering peak, but no band
        # write would fit beside both. Each dense band holds 20,400 sparse
        # bytes.
        (300, np.uint8, 40_800, 20_400),
    ],
)
def test_sparse_writer_writes_pending_bands_before_a_pull_that_would_not_fit(
    columns, dtype, memory_bytes, reserve_bytes
):
    from cytearc.storage.sharding import (
        SparseRowBand,
        SparseWriteBand,
        write_sparse_bands,
    )

    destination = zarr.open_group(store=MemoryStore(), mode="w").create_array(
        "counts",
        shape=(12, columns),
        chunks=(4, columns),
        shards=(4, columns),
        dtype=dtype,
        fill_value=0,
    )
    expected = (np.arange(12 * columns) % 50 + 1).astype(dtype).reshape(12, columns)
    written_before_pull: list[bool] = []

    def writes():
        for start in range(0, 12, 4):
            written_before_pull.append(
                np.array_equal(destination[:start], expected[:start])
            )
            band = coo_matrix(expected[start : start + 4])
            yield SparseWriteBand(
                destination=destination,
                band=SparseRowBand(
                    start=start,
                    end=start + 4,
                    nColumns=columns,
                    row=band.row.astype(np.int64),
                    column=band.col.astype(np.int64),
                    data=band.data,
                    dtype=dtype,
                ),
            )

    # Four workers have room to write several bands at once.
    write_sparse_bands(
        writes(),
        resources=ResourceBudget(memory_bytes, 4),
        producerReserveBytes=reserve_bytes,
    )
    # Every earlier band is written before the producer is pulled again.
    assert written_before_pull == [True, True, True]
    np.testing.assert_array_equal(destination[:], expected)


def test_sparse_import_planner_keeps_an_explicit_batch_up_to_one_band() -> None:
    destinations = _planner_destinations()

    def explicit(rows: int) -> int:
        return resolve_sparse_import_batch(
            destinations,
            nRows=12,
            resources=ResourceBudget(1_000_000, 2),
            maxWindowNnz=lambda width: min(width, 12) * 4,
            sourceDtype=np.uint32,
            batchRows=rows,
        ).batchRows

    # The shortest destination band holds four rows; a wider batch would
    # only buffer more source rows than the layout fit admitted.
    assert explicit(3) == 3
    assert explicit(7) == 4
    with pytest.raises(ValueError, match="batch_size must be positive"):
        resolve_sparse_import_batch(
            destinations,
            nRows=12,
            resources=ResourceBudget(1_000_000, 2),
            maxWindowNnz=lambda rows: min(rows, 12) * 4,
            sourceDtype=np.uint32,
            batchRows=0,
        )
    with pytest.raises(MemoryError, match="one source row"):
        resolve_sparse_import_batch(
            destinations,
            nRows=12,
            resources=ResourceBudget(1_000, 1),
            maxWindowNnz=lambda rows: min(rows, 12) * 4,
            sourceDtype=np.uint32,
        )


class _PlanningReader:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.nCells, self.nFeatures = values.shape
        self.matrix_dtype = values.dtype
        self.assayFeats = pd.DataFrame(
            {
                "RNA": [
                    "Gene Expression",
                    0,
                    self.nFeatures,
                    self.nFeatures,
                ]
            },
            index=["type", "start", "end", "nFeatures"],
        )
        self.consumed: list[int] = []
        self.windowRequests: list[int] = []

    def cell_names(self) -> list[str]:
        return [f"cell-{index}" for index in range(self.nCells)]

    def feature_ids(self, assay: str) -> list[str]:
        return [f"feature-{index}" for index in range(self.nFeatures)]

    def feature_names(self, assay: str) -> list[str]:
        return [f"gene-{index}" for index in range(self.nFeatures)]

    def consume(self, batch_size: int, lines_in_mem: int):
        self.consumed.append(batch_size)
        for start in range(0, self.nCells, batch_size):
            yield coo_matrix(self.values[start : start + batch_size])

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        value_range = CountValueRange()
        value_range.update(self.values)
        return [value_range]

    def max_window_nnz(self, window_rows: int) -> int:
        self.windowRequests.append(window_rows)
        width = min(window_rows, self.nCells)
        return max(
            np.count_nonzero(self.values[start : start + width])
            for start in range(self.nCells - width + 1)
        )

    def producer_staging_bytes(self, batch_size: int, lines_in_mem: int) -> int:
        return 0


def test_crtozarr_automatic_rows_and_preflight_before_consume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.storage import sharding

    original_write = sharding.write_sparse_bands
    observed: dict[str, int] = {}

    def recording_write(writes, **kwargs):
        count = 0

        def recording_source():
            nonlocal count
            for write in writes:
                count += 1
                yield write

        original_write(recording_source(), **kwargs)
        observed["writes"] = count
        observed["total"] = int(kwargs["total"])

    monkeypatch.setattr(sharding, "write_sparse_bands", recording_write)
    values = np.eye(8, 4, dtype=np.uint16)
    reader = _PlanningReader(values)
    store = MemoryStore()
    # The counts store as uint8, so 16 bytes give four-row shards.
    writer = CrToZarr(
        reader,
        store,
        mem_budget="64M",
        policy=CountMatrixPolicy(unitBytes=16, chunkBytes=8),
    )
    writer.dump()

    assert reader.consumed == [4]
    assert writer._lastImportPlan.batchRows == 4
    assert 4 in reader.windowRequests
    assert observed == {"writes": 2, "total": 2}
    assert writer._lastImportPlan.writeTasks == 2
    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert array_shard_rows(counts) == 4
    np.testing.assert_array_equal(counts[:], values)

    rejected = _PlanningReader(values)
    destination = MemoryStore()
    with pytest.raises(MemoryError, match="requested count-matrix policy"):
        CrToZarr(
            rejected,
            destination,
            mem_budget=1,
            policy=CountMatrixPolicy(unitBytes=16, chunkBytes=8),
        )
    assert rejected.consumed == []
    with pytest.raises(FileNotFoundError):
        zarr.open_group(store=destination, mode="r")


class _TrackingCsr(csr_matrix):
    def __init__(self, values: np.ndarray) -> None:
        super().__init__(values)
        self.sliceCount = 0

    def __getitem__(self, key):
        self.sliceCount += 1
        return super().__getitem__(key)


def test_h5ad_and_sparse_preflight_before_source_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.eye(8, 4, dtype=np.uint16)
    path = tmp_path / "preflight.h5ad"
    _write_h5ad(path, values, "csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    h5ad_writer = H5adToZarr(
        reader,
        MemoryStore(),
        mem_budget="64M",
        policy=CountMatrixPolicy(unitBytes=32, chunkBytes=16),
    )
    consumed = False

    def unexpected_consume(batch_size: int):
        nonlocal consumed
        consumed = True
        yield coo_matrix(values[:batch_size])

    monkeypatch.setattr(reader, "consume", unexpected_consume)
    h5ad_writer.resources = ResourceBudget(1, 1)
    try:
        with pytest.raises(MemoryError, match="one source row"):
            h5ad_writer.dump()
    finally:
        reader.h5.close()
    assert not consumed

    tracked = _TrackingCsr(values)
    sparse_writer = SparseToZarr(
        tracked,
        MemoryStore(),
        cell_ids=[f"cell-{index}" for index in range(values.shape[0])],
        feature_ids=[f"feature-{index}" for index in range(values.shape[1])],
        mem_budget="64M",
        policy=CountMatrixPolicy(unitBytes=32, chunkBytes=16),
    )
    sparse_writer.resources = ResourceBudget(1, 1)
    with pytest.raises(MemoryError, match="one source row"):
        sparse_writer.dump()
    assert sparse_writer.mat.sliceCount == 0


def _write_cr_h5(path: Path, values: np.ndarray) -> None:
    matrix = csr_matrix(values)
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset(
            "barcodes",
            data=np.array(
                [f"cell-{index}".encode() for index in range(values.shape[0])]
            ),
        )
        features = group.create_group("features")
        features.create_dataset(
            "id",
            data=np.array(
                [f"feature-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        features.create_dataset(
            "name",
            data=np.array(
                [f"gene-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        features.create_dataset(
            "feature_type",
            data=np.array(
                [
                    (
                        "Gene Expression" if index % 2 == 0 else "Antibody Capture"
                    ).encode()
                    for index in range(values.shape[1])
                ]
            ),
        )


def _write_h5ad(path: Path, values: np.ndarray, encoding: str) -> None:
    with h5py.File(path, mode="w") as handle:
        if encoding == "dense":
            handle.create_dataset("X", data=values)
        else:
            matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values)
            group = handle.create_group("X")
            group.attrs["encoding-type"] = f"{encoding}_matrix"
            group.attrs["shape"] = values.shape
            group.create_dataset("data", data=matrix.data)
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array(
                [f"cell-{index}".encode() for index in range(values.shape[0])]
            ),
        )
        var = handle.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array(
                [f"feature-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        var.create_dataset(
            "feature_name",
            data=np.array(
                [f"gene-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        var.create_dataset(
            "feature_types",
            data=np.array(
                [
                    (
                        "Gene Expression" if index % 2 == 0 else "Antibody Capture"
                    ).encode()
                    for index in range(values.shape[1])
                ]
            ),
        )
        handle.create_group("obsm")


def _assay_counts(store: MemoryStore) -> dict[str, np.ndarray]:
    root = zarr.open_group(store=store, mode="r")
    return {name: np.asarray(root[f"{name}/counts"][:]) for name in ("RNA", "ADT")}


def _count_arrays(store: MemoryStore) -> tuple[zarr.Array, zarr.Array]:
    root = zarr.open_group(store=store, mode="r")
    return root["RNA/counts"], root["ADT/counts"]


def test_cellranger_h5_automatic_matches_explicit_and_caches_planning(
    tmp_path: Path,
) -> None:
    values = np.arange(48, dtype=np.uint16).reshape(8, 6) % 5
    path = tmp_path / "counts.h5"
    _write_cr_h5(path, values)
    stores = [MemoryStore(), MemoryStore()]
    readers = [CrH5Reader(str(path)), CrH5Reader(str(path))]
    try:
        # The uint16 source stores as uint8, so half the byte targets of the
        # uint16 layout give the same geometry.
        automatic = CrToZarr(
            readers[0],
            stores[0],
            mem_budget="64M",
            policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
        )
        explicit = CrToZarr(
            readers[1],
            stores[1],
            mem_budget="64M",
            policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
        )
        automatic.dump()
        explicit.dump(batch_size=3)
        cached = readers[0]._indptrCache
        assert cached is not None
        readers[0].max_window_nnz(2)
        assert readers[0]._indptrCache is cached
        automatic_arrays = _count_arrays(stores[0])
        explicit_arrays = _count_arrays(stores[1])
        assert automatic._lastImportPlan.batchRows == min(
            array_shard_rows(array) for array in automatic_arrays
        )
        assert automatic._lastImportPlan.writeTasks == sparse_write_task_count(
            automatic_arrays,
            values.shape[0],
        )
        without_projection = resolve_sparse_import_batch(
            automatic_arrays,
            nRows=values.shape[0],
            resources=automatic.resources,
            maxWindowNnz=readers[0].max_window_nnz,
            sourceDtype=readers[0].matrix_dtype,
            batchRows=automatic._lastImportPlan.batchRows,
            producerStagingBytes=lambda rows: readers[0].producer_staging_bytes(
                rows,
                100_000,
            ),
        )
        assert (
            automatic._lastImportPlan.producerReserveBytes
            > without_projection.producerReserveBytes
        )
        assert explicit._lastImportPlan.batchRows == 3
        for automatic_array, explicit_array in zip(
            automatic_arrays,
            explicit_arrays,
            strict=True,
        ):
            assert automatic_array.dtype == explicit_array.dtype == np.dtype(np.uint8)
            assert automatic_array.chunks == explicit_array.chunks
            assert array_shard_rows(automatic_array) == array_shard_rows(explicit_array)
    finally:
        for reader in readers:
            reader.close()

    # Even source features are Gene Expression and odd ones Antibody Capture.
    for store in stores:
        counts = _assay_counts(store)
        np.testing.assert_array_equal(counts["RNA"], values[:, ::2])
        np.testing.assert_array_equal(counts["ADT"], values[:, 1::2])


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_h5ad_automatic_matches_explicit_for_split_assays(
    tmp_path: Path,
    encoding: str,
) -> None:
    values = np.arange(48, dtype=np.uint16).reshape(8, 6) % 7
    path = tmp_path / f"{encoding}.h5ad"
    _write_h5ad(path, values, encoding)
    stores = [MemoryStore(), MemoryStore()]
    readers = [
        H5adReader(str(path), feature_name_key="feature_name"),
        H5adReader(str(path), feature_name_key="feature_name"),
    ]
    try:
        # These counts store as uint8, so half the byte targets of the uint16
        # imports in this module give the same geometry.
        automatic = H5adToZarr(
            readers[0],
            stores[0],
            assay_split_key="feature_types",
            mem_budget="64M",
            policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
        )
        explicit = H5adToZarr(
            readers[1],
            stores[1],
            assay_split_key="feature_types",
            mem_budget="64M",
            policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
        )
        automatic.dump()
        explicit.dump(batch_size=3)
        automatic_arrays = _count_arrays(stores[0])
        explicit_arrays = _count_arrays(stores[1])
        assert automatic._lastImportPlan.batchRows == min(
            array_shard_rows(array) for array in automatic_arrays
        )
        assert automatic._lastImportPlan.writeTasks == sparse_write_task_count(
            automatic_arrays,
            values.shape[0],
        )
        # Both encodings yield canonical batches, which the split keeps.
        without_projection = resolve_sparse_import_batch(
            automatic_arrays,
            nRows=values.shape[0],
            resources=automatic.resources,
            maxWindowNnz=readers[0].max_batch_nnz,
            sourceDtype=readers[0].sourceMatrixDtype,
            batchRows=automatic._lastImportPlan.batchRows,
            producerStagingBytes=readers[0].producer_batch_staging_bytes,
            canonicalBatches=True,
        )
        assert (
            automatic._lastImportPlan.producerReserveBytes
            > without_projection.producerReserveBytes
        )
        assert explicit._lastImportPlan.batchRows == 3
        for automatic_array, explicit_array in zip(
            automatic_arrays,
            explicit_arrays,
            strict=True,
        ):
            # Counts up to 6 store as uint8 whatever the source dtype.
            assert automatic_array.dtype == explicit_array.dtype == np.dtype(np.uint8)
            assert automatic_array.chunks == explicit_array.chunks
            assert array_shard_rows(automatic_array) == array_shard_rows(explicit_array)
    finally:
        for reader in readers:
            reader.h5.close()

    # Even source features are Gene Expression and odd ones Antibody Capture.
    for store in stores:
        counts = _assay_counts(store)
        np.testing.assert_array_equal(counts["RNA"], values[:, ::2])
        np.testing.assert_array_equal(counts["ADT"], values[:, 1::2])


def test_dense_h5ad_split_writes_dense_bands_of_each_assay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cytearc.storage import sharding

    values = (np.arange(240, dtype=np.uint16).reshape(40, 6) % 7) * 40
    path = tmp_path / "dense.h5ad"
    _write_h5ad(path, values, "dense")

    def no_sparse_band(*_args, **_kwargs):
        raise AssertionError("a dense matrix was cut into sparse bands")

    monkeypatch.setattr(sharding.SparseShardBuffer, "add", no_sparse_band)
    batches: list[tuple[int, int]] = []
    consume = H5adReader.consume_dense_rows

    def recorded(self, batch_size, row_start=0, row_end=None):
        for batch in consume(self, batch_size, row_start, row_end):
            batches.append((batch_size, batch.shape[0]))
            yield batch

    monkeypatch.setattr(H5adReader, "consume_dense_rows", recorded)
    stores = {"automatic": MemoryStore(), "explicit": MemoryStore()}
    writers = {}
    for name, store in stores.items():
        reader = H5adReader(str(path), feature_name_key="feature_name")
        try:
            # Counts up to 240 store as uint8, so each assay's band of 40
            # cells holds 120 bytes.
            writers[name] = H5adToZarr(
                reader,
                store,
                assay_split_key="feature_types",
                mem_budget="64M",
                nthreads=1,
                policy=CountMatrixPolicy(unitBytes=120, chunkBytes=60),
            )
            batches.clear()
            writers[name].dump(batch_size=None if name == "automatic" else 3)
        finally:
            reader.close()
        # Each batch takes the planned rows, or the rows that remain.
        planned = writers[name]._lastDenseBatchRows
        assert {size for size, _rows in batches} == {planned}
        assert sum(rows for _size, rows in batches) == values.shape[0]
        assert max(rows for _size, rows in batches) == planned
    # A row of six uint16 values takes 12 bytes, a copy of one assay's three
    # columns 6 more, and their checked cast to uint8 3 bytes a value, 27 in
    # all; 240 bytes of bands hold 8 such rows.
    assert writers["automatic"]._lastDenseBatchRows == 8
    assert writers["explicit"]._lastDenseBatchRows == 3
    for store in stores.values():
        counts = _assay_counts(store)
        np.testing.assert_array_equal(counts["RNA"], values[:, ::2])
        np.testing.assert_array_equal(counts["ADT"], values[:, 1::2])
        for array in _count_arrays(store):
            assert array.dtype == np.dtype(np.uint8)
            assert array_metadata_shards(array) == (40, 3)


def test_dense_h5ad_batches_hold_whole_rows_of_hdf5_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.arange(360, dtype=np.uint16).reshape(60, 6) % 7
    paths = {
        "contiguous": tmp_path / "contiguous.h5ad",
        "chunked": tmp_path / "chunked.h5ad",
    }
    for path in paths.values():
        _write_h5ad(path, values, "dense")
    with h5py.File(paths["chunked"], mode="r+") as handle:
        del handle["X"]
        handle.create_dataset("X", data=values, chunks=(7, 3))
    batches: list[int] = []
    consume = H5adReader.consume_dense_rows

    def recorded(self, batch_size, row_start=0, row_end=None):
        for batch in consume(self, batch_size, row_start, row_end):
            batches.append(batch.shape[0])
            yield batch

    monkeypatch.setattr(H5adReader, "consume_dense_rows", recorded)
    read = {}
    for name, path in paths.items():
        store = MemoryStore()
        reader = H5adReader(str(path), feature_name_key="feature_name")
        try:
            # Counts up to 6 store as uint8, so 360 bytes make one band of
            # every cell.
            writer = H5adToZarr(
                reader,
                store,
                mem_budget="64M",
                nthreads=1,
                policy=CountMatrixPolicy(unitBytes=360, chunkBytes=180),
            )
            batches.clear()
            writer.dump()
        finally:
            reader.close()
        read[name] = list(batches)
        np.testing.assert_array_equal(
            zarr.open_group(store=store, mode="r")["RNA/counts"][:], values
        )
    # A row of six uint16 values takes 12 bytes and their checked cast to
    # uint8 18 more, so the band's 360 bytes hold 12 rows. A chunked matrix
    # reads whole rows of its 7-row chunks within them.
    assert read["contiguous"] == [12] * 5
    assert read["chunked"] == [7] * 8 + [4]


def test_dense_rows_come_in_the_source_dtype_and_window(tmp_path: Path) -> None:
    values = _tenth_nonzero()
    dense = tmp_path / "dense.h5ad"
    _write_h5ad(dense, values, "dense")
    reader = H5adReader(str(dense), feature_name_key="feature_name")
    try:
        batches = list(reader.consume_dense_rows(7, 3, 30))
        assert [batch.shape for batch in batches] == [(7, 12)] * 3 + [(6, 12)]
        assert {batch.dtype for batch in batches} == {np.dtype(np.float32)}
        np.testing.assert_array_equal(np.vstack(batches), values[3:30])
        assert list(reader.consume_dense_rows(5, 9, 9)) == []
        with pytest.raises(ValueError, match="batch_size must be positive"):
            next(reader.consume_dense_rows(0))
        with pytest.raises(ValueError, match="outside the matrix"):
            next(reader.consume_dense_rows(5, 30, 41))
    finally:
        reader.close()
    csr = tmp_path / "csr.h5ad"
    _write_h5ad(csr, values, "csr")
    reader = H5adReader(str(csr), feature_name_key="feature_name")
    try:
        with pytest.raises(TypeError, match="consume yields its rows"):
            next(reader.consume_dense_rows(5))
    finally:
        reader.close()


def _largest_window_nonzeros(values: np.ndarray, rows: int) -> int:
    """The most nonzero values of any ``rows`` consecutive rows, by NumPy."""
    rows = min(rows, values.shape[0])
    return max(
        int(np.count_nonzero(values[start : start + rows]))
        for start in range(values.shape[0] - rows + 1)
    )


def _tenth_nonzero(n_cells: int = 40, n_features: int = 12) -> np.ndarray:
    rng = np.random.default_rng(3)
    values = np.where(
        rng.random((n_cells, n_features)) < 0.1,
        rng.integers(1, 9, (n_cells, n_features)),
        0,
    ).astype(np.float32)
    # Signed zeros are zero, as the sparse batches drop them.
    values[values == 0] = np.where(
        rng.random(int((values == 0).sum())) < 0.5, -0.0, 0.0
    )
    return values


@pytest.mark.parametrize("scanned", [True, False], ids=["scanned", "unscanned"])
def test_dense_h5ad_plans_each_window_from_its_nonzero_values(
    tmp_path: Path, scanned: bool
) -> None:
    values = _tenth_nonzero()
    path = tmp_path / "dense.h5ad"
    _write_h5ad(path, values, "dense")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        if scanned:
            reader.count_value_ranges(1 << 20)
        # Before the scan the reader counts the rows itself.
        for rows in (1, 7, values.shape[0], 10 * values.shape[0]):
            expected = _largest_window_nonzeros(values, rows)
            assert reader.max_batch_nnz(rows) == expected
            assert expected < min(rows, values.shape[0]) * values.shape[1]
            # No batch that the import reads holds more than its plan.
            assert max(batch.nnz for batch in reader.consume(rows)) <= expected
        # The import holds the cumulative counts, one 64-bit integer per row
        # and one more.
        reader._prepare_sparse_import()
        assert reader._sparse_import_resident_bytes() == 8 * (values.shape[0] + 1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            reader.max_batch_nnz(0)
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("values", "per_row"),
    [
        (np.arange(1, 61, dtype=np.uint16).reshape(10, 6), 6),
        (np.zeros((10, 6), dtype=np.float32), 0),
    ],
    ids=["no-zeros", "empty-rows"],
)
def test_dense_h5ad_plans_every_value_only_when_none_is_zero(
    tmp_path: Path, values: np.ndarray, per_row: int
) -> None:
    path = tmp_path / "dense.h5ad"
    _write_h5ad(path, values, "dense")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        reader.count_value_ranges(1 << 20)
        for rows in (1, 7, 10, 11):
            assert reader.max_batch_nnz(rows) == min(rows, 10) * per_row
    finally:
        reader.close()


def test_sparse_automatic_matches_explicit() -> None:
    values = csr_matrix(np.arange(48, dtype=np.uint16).reshape(8, 6) % 5)
    stores = [MemoryStore(), MemoryStore()]
    writers = [
        SparseToZarr(
            values,
            store,
            cell_ids=[f"cell-{index}" for index in range(values.shape[0])],
            feature_ids=[f"feature-{index}" for index in range(values.shape[1])],
            mem_budget="64M",
            policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
        )
        for store in stores
    ]
    writers[0].dump()
    writers[1].dump(batch_size=3)

    automatic_array = zarr.open_group(store=stores[0], mode="r")["RNA/counts"]
    explicit_array = zarr.open_group(store=stores[1], mode="r")["RNA/counts"]
    assert writers[0]._lastImportPlan.batchRows == array_shard_rows(automatic_array)
    assert writers[0]._lastImportPlan.writeTasks == sparse_write_task_count(
        (automatic_array,),
        values.shape[0],
    )
    assert writers[1]._lastImportPlan.batchRows == 3
    assert automatic_array.dtype == explicit_array.dtype == np.dtype(np.uint8)
    assert automatic_array.chunks == explicit_array.chunks
    assert array_shard_rows(automatic_array) == array_shard_rows(explicit_array)
    np.testing.assert_array_equal(automatic_array[:], values.toarray())
    np.testing.assert_array_equal(explicit_array[:], values.toarray())


def _record_sorts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the values of each batch that an import sorts and sums.

    The shard buffer, and the undoing of a log, sort a batch and sum its
    repeated coordinates through ``canonicalize_sparse`` unless the batch is
    marked canonical.
    """
    from cytearc.storage import sharding
    from cytearc.utils import arrays

    canonicalize = arrays.canonicalize_sparse
    sorted_values: list[int] = []

    def recorded(coo, dtype=None):
        if not coo.has_canonical_format:
            sorted_values.append(int(coo.nnz))
        return canonicalize(coo, dtype)

    monkeypatch.setattr(sharding, "canonicalize_sparse", recorded)
    monkeypatch.setattr(arrays, "canonicalize_sparse", recorded)
    return sorted_values


def _counts() -> np.ndarray:
    """Eight cells of six features, each cell with four or five nonzero counts."""
    return np.arange(48, dtype=np.float32).reshape(8, 6) % 5


def _rows_out_of_order(values: np.ndarray) -> csr_matrix:
    """CSR rows of ``values`` with row 1's indices reversed and row 4's first
    value split into two entries at the same index."""
    data: list[float] = []
    indices: list[int] = []
    pointers = [0]
    for row, line in enumerate(values):
        columns = np.flatnonzero(line).tolist()
        stored = line[columns].tolist()
        if row == 1:
            columns.reverse()
            stored.reverse()
        if row == 4:
            columns[0:1] = [columns[0], columns[0]]
            stored[0:1] = [stored[0] - 1, 1.0]
        data += stored
        indices += columns
        pointers.append(len(data))
    matrix = csr_matrix(
        (np.array(data, dtype=values.dtype), np.array(indices), np.array(pointers)),
        shape=values.shape,
    )
    assert not matrix.has_canonical_format
    return matrix


def _row_is_canonical(matrix: csr_matrix, row: int) -> bool:
    """Whether one CSR row holds its indices sorted without repeats, by NumPy."""
    indices = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
    return bool(np.all(np.diff(indices) > 0))


def _write_rows(path: Path, matrix: csr_matrix) -> None:
    """Replace the CSR matrix of an h5ad file with ``matrix``'s rows as given."""
    with h5py.File(path, mode="r+") as handle:
        group = handle["X"]
        for name in ("data", "indices", "indptr"):
            del group[name]
            group.create_dataset(name, data=getattr(matrix, name))


def _log_normalized(counts: np.ndarray) -> np.ndarray:
    """Counts scaled to 10,000 per cell and logged, as scanpy writes them."""
    return np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_h5ad_reader_marks_the_batches_of_canonical_rows(
    tmp_path: Path, encoding: str
) -> None:
    values = _counts()
    path = tmp_path / f"{encoding}.h5ad"
    _write_h5ad(path, values, encoding)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        # CSR rows count as not canonical until the value scan reads them.
        assert reader._canonical_batches() is (encoding != "csr")
        reader.count_value_ranges(1 << 20)
        assert reader._canonical_batches() is True
        for rows in (1, 3, 8):
            batches = list(reader.consume(rows))
            # SciPy 1.16 does not mark the COO that it converts from CSR.
            assert all(batch.has_canonical_format is True for batch in batches)
            np.testing.assert_array_equal(
                np.vstack([batch.toarray() for batch in batches]), values
            )
    finally:
        reader.close()


def test_h5ad_reader_leaves_the_batches_of_rows_out_of_order_unmarked(
    tmp_path: Path,
) -> None:
    values = _counts()
    matrix = _rows_out_of_order(values)
    path = tmp_path / "unsorted.h5ad"
    _write_h5ad(path, values, "csr")
    _write_rows(path, matrix)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        reader.count_value_ranges(1 << 20)
        assert reader._canonical_batches() is False
        for rows in (1, 3, 8):
            marks = [batch.has_canonical_format for batch in reader.consume(rows)]
            assert marks == [
                all(
                    _row_is_canonical(matrix, row)
                    for row in range(start, min(start + rows, 8))
                )
                for start in range(0, 8, rows)
            ]
        # Only rows 1 and 4 are out of order.
        assert [batch.has_canonical_format for batch in reader.consume(1)] == [
            row not in (1, 4) for row in range(8)
        ]
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("encoding", "split", "kind"),
    [
        ("csr", False, "counts"),
        ("csr", True, "counts"),
        ("csc", True, "counts"),
        ("dense", True, "counts"),
        ("csr", False, "log_normalized"),
    ],
    ids=["csr", "csr-split", "csc-split", "dense-split", "csr-log"],
)
def test_an_h5ad_import_sorts_no_batch_of_canonical_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    split: bool,
    kind: str,
) -> None:
    counts = _counts()
    values = counts if kind == "counts" else _log_normalized(counts)
    path = tmp_path / f"{encoding}.h5ad"
    _write_h5ad(path, values, encoding)
    sorts = _record_sorts(monkeypatch)
    store = MemoryStore()
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        H5adToZarr(
            reader,
            store,
            assay_split_key="feature_types" if split else None,
            mem_budget="64M",
            nthreads=1,
            policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
            values=None if kind == "counts" else kind,
        ).dump()
    finally:
        reader.close()
    assert sorts == []
    if split:
        stored = _assay_counts(store)
        np.testing.assert_array_equal(stored["RNA"], values[:, ::2])
        np.testing.assert_array_equal(stored["ADT"], values[:, 1::2])
    elif kind == "counts":
        root = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(root["RNA/counts"][:], values)
    else:
        # Undoing the log stores each cell's counts scaled to 10,000.
        root = zarr.open_group(store=store, mode="r")
        np.testing.assert_allclose(
            root["RNA/counts"][:],
            counts / counts.sum(axis=1, keepdims=True) * 1e4,
            rtol=1e-5,
        )


def test_an_h5ad_import_sorts_and_sums_the_batches_of_rows_out_of_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _counts()
    matrix = _rows_out_of_order(values)
    paths = {"canonical": tmp_path / "canonical.h5ad", "out": tmp_path / "out.h5ad"}
    for path in paths.values():
        _write_h5ad(path, values, "csr")
    _write_rows(paths["out"], matrix)
    counts = {}
    sorts: dict[str, list[int]] = {}
    for name, path in paths.items():
        with monkeypatch.context() as patch:
            sorts[name] = _record_sorts(patch)
            store = MemoryStore()
            reader = H5adReader(str(path), feature_name_key="feature_name")
            try:
                H5adToZarr(
                    reader,
                    store,
                    mem_budget="64M",
                    nthreads=1,
                    policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
                ).dump()
            finally:
                reader.close()
        counts[name] = zarr.open_group(store=store, mode="r")["RNA/counts"]
    # Batches of four rows: row 1 is in the first and row 4 in the second, so
    # both are sorted and summed, each with every value it holds.
    assert sorts["out"] == [
        int(matrix.indptr[4] - matrix.indptr[0]),
        int(matrix.indptr[8] - matrix.indptr[4]),
    ]
    assert sorts["canonical"] == []
    for array in counts.values():
        np.testing.assert_array_equal(array[:], values)
    assert (
        counts["out"].attrs["content_fingerprint"]
        == counts["canonical"].attrs["content_fingerprint"]
    )


@pytest.mark.parametrize("kind", ["counts", "log_normalized"])
def test_sparse_to_zarr_sorts_no_batch_of_canonical_rows(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    counts = _counts()
    values = counts if kind == "counts" else _log_normalized(counts)
    sorts = _record_sorts(monkeypatch)
    store = MemoryStore()
    SparseToZarr(
        csr_matrix(values),
        store,
        cell_ids=[f"cell-{index}" for index in range(values.shape[0])],
        feature_ids=[f"feature-{index}" for index in range(values.shape[1])],
        mem_budget="64M",
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
        values=None if kind == "counts" else kind,
    ).dump()
    assert sorts == []
    stored = zarr.open_group(store=store, mode="r")["RNA/counts"][:]
    if kind == "counts":
        np.testing.assert_array_equal(stored, values)
    else:
        np.testing.assert_allclose(
            stored, counts / counts.sum(axis=1, keepdims=True) * 1e4, rtol=1e-5
        )


def test_sparse_to_zarr_sorts_and_sums_the_batches_of_rows_out_of_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _counts()
    matrix = _rows_out_of_order(values)
    sorts = _record_sorts(monkeypatch)
    store = MemoryStore()
    SparseToZarr(
        matrix,
        store,
        cell_ids=[f"cell-{index}" for index in range(values.shape[0])],
        feature_ids=[f"feature-{index}" for index in range(values.shape[1])],
        mem_budget="64M",
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=24, chunkBytes=12),
    ).dump()
    assert sorts == [
        int(matrix.indptr[4] - matrix.indptr[0]),
        int(matrix.indptr[8] - matrix.indptr[4]),
    ]
    stored = zarr.open_group(store=store, mode="r")["RNA/counts"][:]
    np.testing.assert_array_equal(stored, values)


def test_the_shard_buffer_sorts_and_sums_only_a_batch_not_marked_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.storage.sharding import SparseShardBuffer
    from cytearc.utils.arrays import csr_to_coo

    spec = ZarrArraySpec(
        shape=(3, 4), chunks=(3, 4), shards=(3, 4), dtype=np.uint16, compressors=()
    )
    sorts = _record_sorts(monkeypatch)
    # Coordinates out of order, with (0, 1) and (2, 3) each stored twice.
    rows = np.array([2, 0, 1, 0, 2, 2])
    columns = np.array([3, 1, 0, 1, 0, 3])
    data = np.array([5, 100, 7, 100, 1, 60], dtype=np.uint16)
    expected = np.zeros((3, 4), dtype=np.int64)
    np.add.at(expected, (rows, columns), data.astype(np.int64))
    buffer = SparseShardBuffer(spec)
    (band,) = [
        *buffer.add(coo_matrix((data, (rows, columns)), shape=(3, 4))),
        *buffer.finish(),
    ]
    assert sorts == [6]
    np.testing.assert_array_equal(band.dense(), expected)
    # The band holds each coordinate once, in row-major order.
    assert np.all(np.diff(band.row * 4 + band.column) > 0)

    # A batch marked canonical is buffered as it is.
    buffer = SparseShardBuffer(spec)
    marked = csr_to_coo(csr_matrix(expected.astype(np.uint16)))
    (band,) = [*buffer.add(marked), *buffer.finish()]
    assert sorts == [6]
    np.testing.assert_array_equal(band.dense(), expected)


def test_canonical_batches_leave_out_a_charge_for_each_source_value_only() -> None:
    from cytearc.storage.sharding import sparse_producer_peak_bytes

    def saved(buffered: int, source: int, itemsize: int = 4) -> int:
        return sparse_producer_peak_bytes(
            buffered, source, itemsize
        ) - sparse_producer_peak_bytes(
            buffered, source, itemsize, canonicalBatches=True
        )

    # Without source values nothing is sorted, and the saving grows with the
    # source values alone, whatever the producer buffers.
    assert saved(1_000, 0) == 0
    assert saved(0, 10) > 0
    assert saved(1_000, 10) == saved(0, 10)
    assert saved(0, 30) == 3 * saved(0, 10)
    assert all(
        sparse_producer_peak_bytes(5, 7, itemsize, canonicalBatches=True)
        < sparse_producer_peak_bytes(5, 7, itemsize)
        for itemsize in (1, 4, 8)
    )
