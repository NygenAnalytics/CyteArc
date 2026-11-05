"""Every memory refusal outside the variant operations is a MemoryBudgetError.

A refusal's ``needed_bytes`` is the smallest memory budget that passes the
step it refused, in the terms of the call's whole budget, ``budget_bytes``
is that budget and ``item`` names the step. The planners, given the bytes of
their work, name the budget that runs them, and one byte less is refused
again. Each family of master's checks refuses a budget too small for it,
naming its step, and a guard keeps the built-in ``MemoryError`` out of every
module but those of the variant operations.
"""

import ast
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from zarr.storage import MemoryStore

import cytearc
from cytearc.utils.memory_budget import MemoryBudgetError, memory_refusal

CYTEARC_ROOT = Path(cytearc.__file__).parent

#: The modules of the variant operations that still build the built-in
#: MemoryError, with the number that each builds, so that a module whose
#: refusals move to MemoryBudgetError lowers its count and leaves the list at
#: none. Every other module, the clone and screen operations' included,
#: refuses with MemoryBudgetError.
PLAIN_MEMORY_ERRORS = {
    # Variant counts, genotypes and frequency clones, with the shared-neighbour
    # graph that frequency clones build.
    "datastore/_operations/variants.py": 10,
    "neighbors/graph.py": 2,
    "readers/tapestri.py": 1,
    "variants/frequencies.py": 5,
}


def plain_memory_errors(source: str) -> list[int]:
    """The lines of ``source`` that build or raise the built-in MemoryError itself."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        builds = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MemoryError"
        )
        raises = (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Name)
            and node.exc.id == "MemoryError"
        )
        if builds or raises:
            lines.append(node.lineno)
    return sorted(lines)


@functools.cache
def _modules() -> dict[str, str]:
    """The source of every module of the package, by its path within it."""
    return {
        path.relative_to(CYTEARC_ROOT).as_posix(): path.read_text()
        for path in sorted(CYTEARC_ROOT.rglob("*.py"))
    }


def test_the_guard_finds_a_memory_error_built_or_raised_and_nothing_else() -> None:
    source = """
def refuse(size):
    if size > 1:
        raise MemoryError(f"{size} bytes")
    error = MemoryError("built and returned")
    if size:
        raise MemoryError
    raise MemoryBudgetError("named", needed_bytes=2, budget_bytes=1, item="a")


def catch(call):
    try:
        call()
    except MemoryError as error:
        return isinstance(error, MemoryError)
"""
    assert plain_memory_errors(source) == [4, 5, 7]


def test_refusals_outside_the_variants_raise_memory_budget_error() -> None:
    found = {
        module: lines
        for module, source in _modules().items()
        if (lines := plain_memory_errors(source))
    }
    unlisted = {
        module: lines
        for module, lines in found.items()
        if module not in PLAIN_MEMORY_ERRORS
    }
    assert unlisted == {}, "raise MemoryBudgetError, through memory_refusal or a ledger"
    changed = {
        module: (len(found.get(module, [])), listed)
        for module, listed in PLAIN_MEMORY_ERRORS.items()
        if len(found.get(module, [])) != listed
    }
    assert changed == {}, "PLAIN_MEMORY_ERRORS lists each module's count exactly"
    # Each listed module exists, so a renamed one cannot hide its refusals.
    for module in PLAIN_MEMORY_ERRORS:
        assert (CYTEARC_ROOT / module).is_file(), module


def test_memory_budget_error_is_the_one_class_that_extends_memory_error() -> None:
    direct = []
    for module, source in _modules().items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "MemoryError"
                for base in node.bases
            ):
                direct.append((module, node.name))
    assert direct == [("utils/memory_budget.py", "MemoryBudgetError")]
    from cytearc.perturbation.histograms import HistogramMemoryError
    from cytearc.storage.sharding import CountLayoutMemoryError

    assert issubclass(CountLayoutMemoryError, MemoryBudgetError)
    assert issubclass(HistogramMemoryError, MemoryBudgetError)


def refused(call: Callable[[], object]) -> MemoryBudgetError:
    """The MemoryBudgetError that ``call`` raises; any other error fails."""
    with pytest.raises(MemoryError) as raised:
        call()
    assert isinstance(raised.value, MemoryBudgetError), raised.value
    return raised.value


def named_budget_runs(
    at_budget: Callable[[int], object], budget: int, *, item: str
) -> MemoryBudgetError:
    """Check the refusal of ``at_budget(budget)`` and that the budget it names passes its step.

    The refusal names ``budget`` and ``item`` and more bytes than the budget;
    at the named budget ``at_budget`` runs, and one byte less is refused
    naming the same budget and step.
    """
    refusal = refused(lambda: at_budget(budget))
    assert refusal.budget_bytes == budget
    assert refusal.needed_bytes > budget
    assert refusal.item == item
    at_budget(refusal.needed_bytes)
    below = refused(lambda: at_budget(refusal.needed_bytes - 1))
    assert (below.needed_bytes, below.item) == (refusal.needed_bytes, item)
    return refusal


# The storage planner.


def test_planning_units_names_the_budget_of_the_resident_data_and_one_unit() -> None:
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.execution import WorkShape, plan_operation

    one_unit = "one unit beside the resident data"
    with_read = "one unit and one inner read beside the resident data"
    for shape, budget, needed, item in (
        # One unit of 2,048 bytes beside 100 resident bytes.
        (WorkShape(nUnits=4, unitBytes=2048, residentBytes=100), 1024, 2148, one_unit),
        # Resident data alone passes the budget.
        (WorkShape(nUnits=2, unitBytes=4, residentBytes=16), 8, 20, one_unit),
        # A unit and its inner read, each of 16 bytes.
        (
            WorkShape(nUnits=4, unitBytes=16, innerReadBytes=16, chunksPerShard=4),
            20,
            32,
            with_read,
        ),
        # Resident and scratch bytes, a unit with its decode, and an inner read.
        (
            WorkShape(
                nUnits=3,
                unitBytes=10,
                residentBytes=5,
                scratchBytes=7,
                decodeBytes=3,
                innerReadBytes=4,
                chunksPerShard=2,
            ),
            20,
            29,
            with_read,
        ),
    ):
        refusal = named_budget_runs(
            lambda memory, shape=shape: plan_operation(
                ResourceBudget(memory, 2), shape
            ),
            budget,
            item=item,
        )
        assert refusal.needed_bytes == needed


def test_splitting_workers_names_the_budget_of_one_task_beside_the_resident_data() -> (
    None
):
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.execution import admitted_worker_split

    def split(memory: int) -> tuple[int, int]:
        return admitted_worker_split(
            ResourceBudget(memory, 4),
            nTasks=3,
            taskBytes=lambda inner: 10 * inner,
            residentBytes=5,
        )

    for budget in (4, 12):
        refusal = named_budget_runs(
            split, budget, item="one task beside the resident data"
        )
        assert refusal.needed_bytes == 15
    assert split(15) == (1, 1)


def test_an_async_buffer_names_the_whole_budget_beside_the_resident_data() -> None:
    import asyncio

    from cytearc.storage.async_execution import AsyncStorageRunner, ByteLedger
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.execution import WorkShape, plan_operation

    refusal = refused(lambda: asyncio.run(ByteLedger(8).acquire(32)))
    assert (refusal.needed_bytes, refusal.budget_bytes, refusal.item) == (
        32,
        8,
        "one buffer",
    )
    beside = refused(lambda: asyncio.run(ByteLedger(8, reservedBytes=100).acquire(32)))
    assert (beside.needed_bytes, beside.budget_bytes) == (132, 108)
    asyncio.run(ByteLedger(32, reservedBytes=100).acquire(32))
    # A runner's ledger holds what its plan leaves beside the resident data.
    plan = plan_operation(
        ResourceBudget(4096, 1), WorkShape(nUnits=2, unitBytes=64, residentBytes=1000)
    )
    runner = AsyncStorageRunner(operation=plan)
    assert runner.ledger.reservedBytes == 1000
    assert runner.ledger.limitBytes + runner.ledger.reservedBytes == 4096


def _array(shape: tuple[int, int], chunks: tuple[int, int]) -> zarr.Array:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    return root.create_array(
        "counts", shape=shape, chunks=chunks, dtype=np.uint32, fill_value=0
    )


def test_feature_streams_name_the_budget_of_their_blocks() -> None:
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.feature_stream import plan_feature_stream

    edge = _array((12, 7), (5, 3))

    def one_feature(memory: int, resident: int = 0) -> object:
        return plan_feature_stream(
            edge,
            featureAxis=1,
            cellAxis=0,
            featureIndices=np.array([6]),
            cellIndices=np.array([10, 11]),
            resources=ResourceBudget(memory, 1),
            blockBytes=lambda width: width,
            residentBytes=resident,
        )

    # A 60-byte chunk decoded beside the block's one byte, which 61 bytes hold.
    refusal = named_budget_runs(one_feature, 60, item="one feature of the selection")
    assert refusal.needed_bytes == 61
    # Beside 200 resident bytes the same block needs 261.
    resident = refused(lambda: one_feature(150, 200))
    assert (resident.needed_bytes, resident.item) == (
        261,
        "one feature beside the resident data",
    )
    one_feature(261, 200)

    wide = _array((12, 10), (5, 5))
    refusal = named_budget_runs(
        lambda memory: plan_feature_stream(
            wide,
            featureAxis=1,
            cellAxis=0,
            featureIndices=np.arange(10),
            cellIndices=np.arange(12),
            resources=ResourceBudget(memory, 1),
            blockBytes=lambda width: 100 * width,
            requestedBatchSize=5,
        ),
        550,
        item="feature batches of 5 features",
    )
    # Five features of 100 bytes each beside one decoded 100-byte chunk.
    assert refusal.needed_bytes == 600
    assert f"It needs at least {refusal.needed_bytes} bytes." in str(refusal)
    # Beside 300 resident bytes the same batches need 900.
    refusal = named_budget_runs(
        lambda memory: plan_feature_stream(
            wide,
            featureAxis=1,
            cellAxis=0,
            featureIndices=np.arange(10),
            cellIndices=np.arange(12),
            resources=ResourceBudget(memory, 1),
            blockBytes=lambda width: 100 * width,
            requestedBatchSize=5,
            residentBytes=300,
        ),
        850,
        item="feature batches of 5 features",
    )
    assert refusal.needed_bytes == 900


def test_a_predicate_that_accepts_no_index_cannot_partition_a_selection() -> None:
    from cytearc.storage.geometry import ArrayGeometry
    from cytearc.storage.partition import partition_indices

    geometry = ArrayGeometry(shape=(12, 10), chunks=(5, 5), shards=None, itemsize=4)
    with pytest.raises(ValueError, match="fits accepts no block of one index"):
        partition_indices(geometry, 1, np.arange(10), fits=lambda width: width < 1)
    # A predicate that takes one index at a time splits every chunk into ones.
    blocks = partition_indices(geometry, 1, np.arange(10), fits=lambda width: width < 2)
    assert [block.indices.size for block in blocks] == [1] * 10


# The count layout and the sparse and dense writes.


def _admits_rows_within(cost_per_row: int, fixed: int) -> Callable[..., None]:
    """An admission that writes counts of ``fixed`` bytes and ``cost_per_row`` per shard row."""

    def admit(specs: tuple[Any, ...], resources: Any) -> None:
        needed = fixed + cost_per_row * int(specs[0].shards[0])
        if needed > resources.memoryBytes:
            raise memory_refusal(
                "Writing the counts", "the counts write", needed, resources.memoryBytes
            )

    return admit


def _fit(admit: Callable[..., None], memory: int, requested: Any = None) -> Any:
    from cytearc.storage.sharding import fit_count_layout

    return fit_count_layout(
        {"RNA": (1_000, np.uint16)},
        nCells=1_000_000,
        profile="fast_local",
        memoryBytes=memory,
        transposed=(),
        admitCounts=admit,
        requested=requested,
    )


def test_the_count_layout_names_the_budget_of_one_row_shards() -> None:
    from cytearc.storage.count_matrix import CountMatrixPolicy

    admit = _admits_rows_within(10, 1_000)
    # Shards of one row write 1,010 bytes, the least of any layout.
    refusal = named_budget_runs(
        lambda memory: _fit(admit, memory), 500, item="count shards of one row"
    )
    assert refusal.needed_bytes == 1_010
    assert f"It needs at least {refusal.needed_bytes} bytes." in str(refusal)
    # A requested layout names the budget that writes it as requested.
    from cytearc.storage.sharding import CountLayoutMemoryError

    requested = CountMatrixPolicy(unitBytes=64_000, chunkBytes=16_000)
    layout = refused(lambda: _fit(admit, 500, requested))
    assert isinstance(layout, CountLayoutMemoryError)
    assert layout.item == "the requested count-matrix policy"
    assert layout.budget_bytes == 500
    assert _fit(admit, layout.needed_bytes, requested) == requested
    refused(lambda: _fit(admit, layout.needed_bytes - 1, requested))


def test_an_allocation_that_fails_is_not_read_as_a_layout_that_does_not_fit() -> None:
    calls: list[int] = []
    failure = MemoryError("the allocation failed")

    def admit(specs: tuple[Any, ...], resources: Any) -> None:
        calls.append(int(specs[0].shards[0]))
        raise failure

    with pytest.raises(MemoryError) as raised:
        _fit(admit, 1 << 40)
    # The failure reaches the caller as it was raised, after one attempt,
    # rather than halving the layout down to shards of one row.
    assert raised.value is failure
    assert len(calls) == 1


def test_a_sparse_band_write_does_not_read_an_allocation_failure_as_a_full_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cytearc.storage.sharding as sharding
    from cytearc.storage.budget import ResourceBudget

    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts", shape=(4, 3), chunks=(2, 3), shards=(2, 3), dtype=np.uint16
    )
    failure = MemoryError("the allocation failed")
    monkeypatch.setattr(
        sharding,
        "_sparse_batch_plan",
        _fails_once(sharding._sparse_batch_plan, failure),
    )
    with pytest.raises(MemoryError) as raised:
        sharding.accumulate_sparse_to_shards(
            destination,
            iter((coo_matrix(np.ones((4, 3), dtype=np.uint16)),)),
            resources=ResourceBudget(10**8, 1),
            producerReserveBytes=0,
        )
    # The failure reaches the caller rather than a smaller batch, which the
    # plans after it would admit.
    assert raised.value is failure


def test_a_sparse_producer_names_the_budget_of_the_bands_it_buffers() -> None:
    import cytearc.storage.sharding as sharding
    from cytearc.storage.budget import ResourceBudget

    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts", shape=(4, 3), chunks=(2, 3), shards=(2, 3), dtype=np.uint16
    )
    bands = list(
        sharding.SparseShardBuffer(destination).add(
            coo_matrix(np.ones((4, 3), dtype=np.uint16))
        )
    )

    def write(budget: int) -> None:
        sharding.write_sparse_bands(
            iter(sharding.SparseWriteBand(destination, band) for band in bands),
            resources=ResourceBudget(budget, 1),
            residentBytes=10_000,
        )

    # The producer reserves the resident data before it hands over a band.
    refusal = refused(lambda: write(9_999))
    assert refusal.item == "the bands that the sparse producer buffers"
    assert refusal.needed_bytes == 10_000


def _fails_once(
    function: Callable[..., Any], failure: BaseException
) -> Callable[..., Any]:
    """``function``, raising ``failure`` at its first call only."""
    calls = iter(range(1_000_000))

    def call(*args: Any, **kwargs: Any) -> Any:
        if next(calls) == 0:
            raise failure
        return function(*args, **kwargs)

    return call


def test_pending_bands_do_not_read_an_allocation_failure_as_a_batch_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import deque

    import cytearc.storage.sharding as sharding
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.io_policy import StorageIoPolicy

    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "counts", shape=(4, 3), chunks=(2, 3), shards=(2, 3), dtype=np.uint16
    )
    buffer = sharding.SparseShardBuffer(destination)
    bands = list(buffer.add(coo_matrix(np.ones((4, 3), dtype=np.uint16))))
    pending = deque(sharding.SparseWriteBand(destination, band) for band in bands)
    assert len(pending) == 2
    failure = MemoryError("the allocation failed")
    original = sharding._sparse_batch_plan

    def plan(batch: Any, resources: Any, resident: int, size: int, io: Any) -> Any:
        if size > 1:
            raise failure
        return original(batch, resources, resident, size, io)

    monkeypatch.setattr(sharding, "_sparse_batch_plan", plan)
    with pytest.raises(MemoryError) as raised:
        sharding._writable_batch_plan(
            pending, ResourceBudget(10**8, 2), 0, StorageIoPolicy()
        )
    assert raised.value is failure


def test_import_widths_do_not_read_an_allocation_failure_as_a_width_too_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cytearc.storage.sharding as sharding
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.geometry import ArrayGeometry

    failure = MemoryError("the allocation failed")
    monkeypatch.setattr(
        sharding,
        "admitted_worker_split",
        _fails_once(sharding.admitted_worker_split, failure),
    )
    geometry = ArrayGeometry(shape=(4, 3), chunks=(2, 3), shards=(2, 3), itemsize=2)
    with pytest.raises(MemoryError) as raised:
        sharding._resolve_sparse_import_geometries(
            ((geometry, np.dtype(np.uint16)),),
            nRows=4,
            resources=ResourceBudget(10**8, 1),
            maxWindowNnz=lambda width: 3 * width,
            sourceDtype=np.uint16,
        )
    assert raised.value is failure


def test_reduction_writes_do_not_read_an_allocation_failure_as_a_block_too_tall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.datastore._operations import graph
    from cytearc.storage.budget import ResourceBudget

    root = zarr.open_group(store=MemoryStore(), mode="w")
    normalized = root.create_array(
        "data", shape=(40, 30), chunks=(10, 30), dtype=np.float32, fill_value=0
    )
    failure = MemoryError("the allocation failed")
    monkeypatch.setattr(
        graph,
        "_reduction_write_bytes",
        _fails_once(graph._reduction_write_bytes, failure),
    )
    with pytest.raises(MemoryError) as raised:
        graph._lsi_write_block_rows(
            normalized, root, ResourceBudget(10**8, 1), dims=4, limit=10
        )
    assert raised.value is failure


def test_document_frequency_lets_an_allocation_failure_through_as_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.assay.normalization import stream_document_frequency
    from cytearc.matrix import ChunkedArray

    failure = MemoryError("the allocation failed")
    counts = ChunkedArray.from_numpy(np.ones((12, 4), dtype=np.uint16), block_size=3)
    monkeypatch.setattr(
        ChunkedArray,
        "_stream_blocks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(MemoryError) as raised:
        stream_document_frequency(
            counts, memory_bytes=10**8, nthreads=1, msg="", operation="Frequency"
        )
    assert raised.value is failure


def test_a_seurat_row_store_lets_an_allocation_failure_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cytearc.readers._seurat.sources as sources

    matrix = csc_matrix(np.ones((2, 3), dtype=np.uint16))
    source = sources.TransposeMatrixSource(
        sources.CscMatrixSource(
            matrix.data, matrix.indices, matrix.indptr, matrix.shape
        )
    )
    failure = MemoryError("the allocation failed")
    monkeypatch.setattr(
        sources,
        "SparseRowStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(MemoryError) as raised:
        source.read_cells(0, 1)
    # Only a refusal becomes the reader's resource limit.
    assert raised.value is failure


def test_a_merge_lets_an_allocation_failure_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cytearc.merge.metadata as merge_metadata
    import cytearc.merge.writer as merge_writer
    from cytearc.merge import DataStoreMerge

    rng = np.random.default_rng(11)
    left = _count_store(tmp_path / "left.zarr", rng.poisson(1.0, (20, 12)), "l")
    right = _count_store(tmp_path / "right.zarr", rng.poisson(1.0, (16, 12)), "r")

    def merge(name: str) -> None:
        DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(tmp_path / f"{name}.zarr"),
            names=["left", "right"],
            prepend_text="",
            seed=0,
            overwrite=True,
            nthreads=1,
        ).dump()

    for module, name in (
        (merge_writer, "plan_dense_write"),
        (merge_metadata, "admitted_worker_split"),
    ):
        failure = MemoryError("the allocation failed")
        with monkeypatch.context() as patched:
            patched.setattr(module, name, _fails_once(getattr(module, name), failure))
            with pytest.raises(MemoryError) as raised:
                merge(name)
        assert raised.value is failure


def _h5ad(path: Path, values: np.ndarray, *, encoding: str) -> Path:
    """Write a minimal AnnData file whose matrix has ``encoding``."""
    matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values)
    with h5py.File(path, mode="w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = f"{encoding}_matrix"
        group.attrs["shape"] = values.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([f"c{i}".encode() for i in range(values.shape[0])])
        )
        h5.create_group("var").create_dataset(
            "_index", data=np.array([f"f{i}".encode() for i in range(values.shape[1])])
        )
    return path


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_an_h5ad_import_refuses_a_budget_of_one_byte_at_its_scan(
    tmp_path: Path, encoding: str
) -> None:
    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    values = np.random.default_rng(3).poisson(1.5, size=(40, 30)).astype(np.uint16)
    source = _h5ad(tmp_path / f"{encoding}.h5ad", values, encoding=encoding)
    reader = H5adReader(str(source))
    try:
        refusal = refused(
            lambda: H5adToZarr(
                reader, zarr_loc=str(tmp_path / "import.zarr"), mem_budget=1, nthreads=1
            ).dump()
        )
    finally:
        reader.close()
    # The count scan names the longer vector it meets.
    assert refusal.item == "scanning one compressed vector"
    assert refusal.budget_bytes == 1


def test_a_csc_conversion_names_the_budget_of_its_pointers(tmp_path: Path) -> None:
    from cytearc.readers import H5adReader

    values = np.random.default_rng(3).poisson(1.5, size=(40, 30)).astype(np.uint16)
    reader = H5adReader(str(_h5ad(tmp_path / "csc.h5ad", values, encoding="csc")))
    try:
        refusal = refused(lambda: reader.materialize_csc(maxBytes=1))
    finally:
        reader.close()
    assert refusal.item == "the pointers and one entry of the CSC row conversion"
    assert refusal.needed_bytes > refusal.budget_bytes == 1


def test_an_h5ad_import_lets_an_allocation_failure_of_a_producer_split_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cytearc.storage.sharding as sharding
    from cytearc.readers import H5adReader
    from cytearc.storage.count_matrix import CountMatrixPolicy
    from cytearc.writers import H5adToZarr

    values = np.random.default_rng(3).poisson(1.5, size=(400, 30)).astype(np.uint16)
    source = _h5ad(tmp_path / "csr.h5ad", values, encoding="csr")
    failure = MemoryError("the allocation failed")
    original = sharding.resolve_sparse_import_batch
    calls: list[int] = []

    def plan(*args: Any, **kwargs: Any) -> Any:
        # The import's own plan, then the plan of its widest producer split.
        calls.append(len(calls))
        if len(calls) == 2:
            raise failure
        return original(*args, **kwargs)

    # The writer imports the planner when it plans.
    monkeypatch.setattr(sharding, "resolve_sparse_import_batch", plan)
    reader = H5adReader(str(source))
    try:
        with pytest.raises(MemoryError) as raised:
            H5adToZarr(
                reader,
                zarr_loc=str(tmp_path / "split.zarr"),
                mem_budget="256M",
                nthreads=4,
                policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
            ).dump()
    finally:
        reader.close()
    # The failure reaches the caller rather than a narrower split.
    assert raised.value is failure
    assert len(calls) == 2


def test_a_sparse_matrix_import_refuses_a_budget_of_one_byte_at_its_matrix(
    tmp_path: Path,
) -> None:
    from cytearc.writers import SparseToZarr

    matrix = csr_matrix(
        np.random.default_rng(5).poisson(2.0, size=(30, 20)).astype(np.uint32)
    )
    refusal = refused(
        lambda: SparseToZarr(
            matrix,
            zarr_loc=str(tmp_path / "sparse.zarr"),
            cell_ids=[f"c{i}" for i in range(30)],
            feature_ids=[f"f{i}" for i in range(20)],
            mem_budget=1,
            nthreads=1,
        ).dump()
    )
    # The matrix that the writer holds refuses first.
    assert refusal.item == "the sparse matrix"


def test_a_dense_batch_beyond_its_reservation_names_the_write_it_would_reserve() -> (
    None
):
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.sharding import write_dense_from_row_batches

    def oversized(rows: int) -> MemoryBudgetError:
        root = zarr.open_group(store=MemoryStore(), mode="w")
        destination = root.create_array(
            "values", shape=(8, 2), chunks=(2, 2), dtype=np.float64
        )
        return refused(
            lambda: write_dense_from_row_batches(
                destination,
                iter((np.ones((1, 2)), np.ones((rows, 2)))),
                resources=ResourceBudget(10**8, 1),
            )
        )

    from cytearc.storage.execution import execution_report_scope

    five, seven = oversized(5), oversized(7)
    assert five.item == seven.item == "a source batch beyond its producer reservation"
    assert five.budget_bytes == 10**8
    # The write reserves the batch's bytes in place of the reservation that
    # the first batch of one row set: what it names grows with the batch.
    assert seven.needed_bytes - five.needed_bytes == (7 - 5) * 2 * 8
    # A write reserved for the five-row batch from the start reserves that.
    root = zarr.open_group(store=MemoryStore(), mode="w")
    destination = root.create_array(
        "values", shape=(8, 2), chunks=(2, 2), dtype=np.float64
    )
    with execution_report_scope() as reports:
        write_dense_from_row_batches(
            destination,
            iter((np.ones((5, 2)), np.ones((3, 2)))),
            resources=ResourceBudget(10**8, 1),
            producerReserveBytes=5 * 2 * 8,
        )
    (report,) = [report for report in reports if report.unitKind == "denseRowBand"]
    assert five.needed_bytes == report.plan.reservedBytes


def test_the_counts_t_write_names_the_budget_that_admits_it() -> None:
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.layout import count_array_spec
    from cytearc.storage.sharding import preflight_counts_t_spec

    spec = count_array_spec(200, 50, np.uint16, profile="fast_local")
    named_budget_runs(
        lambda memory: preflight_counts_t_spec(
            spec, profile="fast_local", resources=ResourceBudget(memory, 1)
        ),
        100,
        item="the countsT write",
    )


# Copies, composites and merges.


def test_a_metadata_copy_names_the_budget_of_one_chunk() -> None:
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.copy import copy_zarr_group_tree

    source = zarr.open_group(store=MemoryStore(), mode="w")
    source.create_array("values", data=np.arange(12, dtype=np.float64), chunks=(12,))

    def copy(memory: int) -> None:
        target = zarr.open_group(store=MemoryStore(), mode="w")
        copy_zarr_group_tree(source, target, resources=ResourceBudget(memory, 4))
        np.testing.assert_array_equal(target["values"][:], np.arange(12))

    refusal = named_budget_runs(copy, 64, item="one chunk of metadata column 'values'")
    assert f"needs {refusal.needed_bytes} bytes" in str(refusal)


def test_composite_identity_checks_and_feature_groups_name_their_bytes() -> None:
    from cytearc.composite.create import _require_unique_cell_ids
    from cytearc.composite.streams import _group_batches
    from cytearc.storage.budget import ResourceBudget

    root = zarr.open_group(store=MemoryStore(), mode="w")
    ids = root.create_array(
        "ids", data=np.array([f"c{i}" for i in range(10)]), chunks=(4,)
    )
    # Two hashes and a flag per cell, and a block of four IDs and their hashes.
    needed = 100 + 10 * 17 + 4 * (ids.dtype.itemsize + 16)
    refusal = named_budget_runs(
        lambda memory: _require_unique_cell_ids(
            ids, ResourceBudget(memory, 1), resident_bytes=100
        ),
        200,
        item="the composite cell identity validation",
    )
    assert refusal.needed_bytes == needed

    groups = [(0, 3), (3, 8), (8, 9)]
    refusal = named_budget_runs(
        lambda memory: _group_batches(groups, {}, lambda width: 7 * width, memory),
        30,
        item="one composite feature group of 5 features",
    )
    assert refusal.needed_bytes == 35
    assert _group_batches(groups, {}, lambda width: 7 * width, 63) == [
        [(0, 0, 3, None), (1, 3, 8, None), (2, 8, 9, None)]
    ]


def _count_store(path: Path, counts: np.ndarray, cell_prefix: str) -> Any:
    from cytearc.datastore.datastore import DataStore
    from tests.storage_helpers import write_count_store

    write_count_store(
        str(path),
        {"RNA": counts},
        "uint16",
        cell_ids=[f"{cell_prefix}{index}" for index in range(len(counts))],
    )
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
    )
    return store


def test_a_composite_refuses_a_budget_of_one_byte_at_its_mappings(
    tmp_path: Path,
) -> None:
    from cytearc.composite import create_composite

    rng = np.random.default_rng(0)
    first = _count_store(tmp_path / "first.zarr", rng.poisson(1.0, (300, 64)), "a")
    second = _count_store(tmp_path / "second.zarr", rng.poisson(1.0, (300, 64)), "b")
    refusal = refused(
        lambda: create_composite(
            {"first": first, "second": second},
            at=tmp_path / "joint.zarr",
            features="union",
            mem_budget=1,
            nthreads=1,
        )
    )
    assert refusal.item == "the composite row and feature mappings"


def test_a_merge_refuses_a_budget_of_one_byte_at_its_schema_discovery(
    tmp_path: Path,
) -> None:
    from cytearc.merge import DataStoreMerge

    rng = np.random.default_rng(11)
    left = _count_store(tmp_path / "left.zarr", rng.poisson(1.0, (20, 12)), "l")
    right = _count_store(tmp_path / "right.zarr", rng.poisson(1.0, (16, 12)), "r")
    refusal = refused(
        lambda: DataStoreMerge(
            datasets=[left, right],
            zarr_path=str(tmp_path / "merged.zarr"),
            names=["left", "right"],
            prepend_text="",
            seed=0,
            overwrite=True,
            mem_budget=1,
            nthreads=1,
        ).plan()
    )
    assert refusal.item == "one source row of merged cell metadata schema discovery"


# Readers and converters.


def test_the_sparse_row_conversion_names_its_pointers_and_its_widest_row(
    tmp_path: Path,
) -> None:
    from cytearc.readers._sparse import SparseRowStore

    def convert(matrix: coo_matrix, memory: int) -> None:
        SparseRowStore(
            lambda: iter((matrix,)),
            matrix.shape,
            matrix.dtype,
            max_bytes=memory,
            temp_dir=tmp_path,
        ).close()

    # Three row pointers of 32 bytes and one entry of 192 bytes.
    diagonal = coo_matrix([[1, 0], [0, 2]])
    pointers = named_budget_runs(
        lambda memory: convert(diagonal, memory),
        100,
        item="the row pointers and one entry of the sparse row conversion",
    )
    assert pointers.needed_bytes == 3 * 32 + 192
    # A bucket holds whole rows, so the first row's two entries need two.
    matrix = coo_matrix([[1, 2], [0, 0]])
    widest = named_budget_runs(
        lambda memory: convert(matrix, memory),
        300,
        item="one sparse row of the conversion",
    )
    assert widest.needed_bytes == 3 * 32 + 2 * 192
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="exceeds maxNnz=1") as nnz:
        SparseRowStore(
            lambda: iter((matrix,)),
            matrix.shape,
            matrix.dtype,
            max_bytes=1024,
            max_nnz=1,
            temp_dir=tmp_path,
        )
    assert not isinstance(nnz.value, MemoryError)
    assert list(tmp_path.iterdir()) == []


def test_seurat_source_preparation_refuses_a_budget_of_one_byte() -> None:
    from cytearc.readers._seurat.sources import (
        CscMatrixSource,
        prepare_matrix_sources,
        release_temporary_storage,
    )

    matrix = csc_matrix(np.arange(12, dtype=np.uint16).reshape(3, 4))
    prepared = CscMatrixSource(matrix.data, matrix.indices, matrix.indptr, matrix.shape)
    try:
        refusal = refused(lambda: prepare_matrix_sources(prepared, max_bytes=1))
    finally:
        release_temporary_storage(prepared)
    assert refusal.item == "the resident data of the Seurat sources"


# Normalization and value scans.


def test_document_frequency_refuses_a_small_budget_and_streams_within_another() -> None:
    from cytearc.assay.normalization import stream_document_frequency
    from cytearc.matrix import ChunkedArray

    values = np.arange(48, dtype=np.uint16).reshape(12, 4)

    def count(memory: int) -> np.ndarray:
        frequency, _ = stream_document_frequency(
            ChunkedArray.from_numpy(values, block_size=3),
            memory_bytes=memory,
            nthreads=1,
            msg="",
            operation="Frequency",
        )
        return frequency

    refusal = refused(lambda: count(100))
    assert "could not allocate" in str(refusal)
    assert str(refusal).endswith(f"It needs at least {refusal.needed_bytes} bytes.")
    # 500 bytes count the four blocks of three rows one at a time.
    np.testing.assert_array_equal(count(500), np.count_nonzero(values, axis=0))


def test_count_value_scans_name_the_bytes_of_their_longest_vector() -> None:
    from cytearc.utils.count_values import compressed_count_ranges, dense_count_ranges

    matrix = csr_matrix(np.array([[1, 2, 3, 0], [0, 0, 0, 4]], dtype=np.uint16))
    refusal = named_budget_runs(
        lambda memory: compressed_count_ranges(
            matrix.indptr, matrix.indices, matrix.data, minorSize=4, maxBytes=memory
        ),
        10,
        item="scanning one compressed vector",
    )
    # The longest vector holds three values, scanned at 72 bytes each.
    assert refusal.needed_bytes == 3 * 72
    dense = np.ones((5, 3), dtype=np.uint16)
    refusal = named_budget_runs(
        lambda memory: dense_count_ranges(
            lambda start, stop: dense[start:stop], 5, 3, maxBytes=memory
        ),
        0,
        item="scanning one dense row",
    )
    assert refusal.needed_bytes == 3 * 72


# Algorithm checks that state their own peak.


def test_graph_fits_and_loads_name_their_planned_peaks() -> None:
    from cytearc.datastore._operations import graph
    from cytearc.matrix import ChunkedArray
    from cytearc.storage.budget import ResourceBudget

    data = ChunkedArray.from_numpy(np.ones((50, 8)), block_size=10)
    pca = named_budget_runs(
        lambda memory: graph._require_pca_fit_budget(
            data, ResourceBudget(memory, 1), dims=3
        ),
        100,
        item="the PCA fit",
    )
    assert f"needs about {pca.needed_bytes} bytes" in str(pca)
    for kwargs, item in (
        ({"n_cells": 50, "batch_size": 50}, "the in-memory k-means fit"),
        ({"n_cells": 50, "batch_size": 10}, "the streamed k-means fit"),
    ):
        from cytearc.neighbors.stages import KMeansFitMemory

        kmeans = named_budget_runs(
            lambda memory, kwargs=kwargs: graph._require_kmeans_fit_budget(
                KMeansFitMemory(peakBytes=5_000, streamResidentBytes=10),
                data,
                ResourceBudget(memory, 1),
                **kwargs,
            ),
            100,
            item=item,
        )
        assert f"needs about {kmeans.needed_bytes} bytes" in str(kmeans)
    named_budget_runs(
        lambda memory: graph._require_ann_index_budget(
            ResourceBudget(memory, 1),
            required=900,
            index_bytes=600,
            work="Building the ANN index of 50 cells",
            m=8,
            remedy="Lower ann_m.",
        ),
        899,
        item="the hnswlib index with ann_m=8 and its work",
    )
    named_budget_runs(
        lambda memory: graph._require_integration_budget(
            ResourceBudget(memory, 1), 700, "SNN integration of 2 graphs"
        ),
        10,
        item="the integration's inputs and work",
    )
    named_budget_runs(
        lambda memory: graph._require_graph_budget(
            ResourceBudget(memory, 1),
            required=4_000,
            work="Leiden",
            n_cells=50,
            n_edges=500,
        ),
        3_999,
        item="the graph of 500 edges and its work",
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    normalized = root.create_array(
        "data", shape=(40, 30), chunks=(10, 30), dtype=np.float32, fill_value=0
    )
    lsi = named_budget_runs(
        lambda memory: graph._streaming_lsi_block_rows(
            normalized, ResourceBudget(memory, 1), n_components=4, n_oversamples=2
        ),
        1_000,
        item="one row of streaming LSI beside its accumulators",
    )
    assert f"needs about {lsi.needed_bytes} bytes" in str(lsi)


def test_materialized_lsi_melding_and_paris_name_their_peaks() -> None:
    from cytearc.datastore._operations.paris_persistence import _raise_if_over_budget
    from cytearc.embeddings.reduction import require_materialized_lsi_budget
    from cytearc.features.genomic.melding import _max_meld_band_rows
    from cytearc.storage.budget import ResourceBudget

    lsi = named_budget_runs(
        lambda memory: require_materialized_lsi_budget(
            n_rows=100,
            n_features=40,
            itemsize=4,
            n_components=5,
            n_oversamples=3,
            memory_bytes=memory,
        ),
        1_000,
        item="the matrix and its SVD workspace",
    )
    assert f"needs about {lsi.needed_bytes} bytes" in str(lsi)
    meld = named_budget_runs(
        lambda memory: _max_meld_band_rows(
            memoryBytes=memory,
            nDocs=50,
            nSourceFeatures=40,
            nTargetFeatures=10,
            sourceItemsize=4,
            storeItemsize=8,
            mappingBytes=100,
            decodeBytes=50,
            extraResidentBytes=0,
            preferredRows=10,
            maxRows=50,
        ),
        150,
        item="one band of melded cells beside the mapping",
    )
    assert f"needs about {meld.needed_bytes} bytes" in str(meld)
    paris = named_budget_runs(
        lambda memory: _raise_if_over_budget(
            3_000_000, ResourceBudget(memory, 1), "Paris hierarchy fit"
        ),
        1_000,
        item="Paris hierarchy fit",
    )
    assert paris.needed_bytes == 3_000_000
    assert "It needs at least 3000000 bytes." in str(paris)


def test_one_aligned_row_names_its_bytes_beside_the_resident_data() -> None:
    from cytearc.mapping import features as mapping_features
    from cytearc.storage.budget import ResourceBudget

    def plan(memory: int) -> object:
        stream = object.__new__(mapping_features.AlignedFeatureStream)
        stream._query_cell_indices = np.array([0, 1])
        stream._resident_bytes = 10
        stream._decode_bytes = 4
        stream._resources = ResourceBudget(memory, 1)
        stream._source_geometry = None
        return stream._plan_rows(bytes_per_row=3)

    refusal = named_budget_runs(
        plan, 10, item="one aligned row beside the resident data"
    )
    assert refusal.needed_bytes == 10 + 3 + 4


def test_a_diffusion_step_and_its_load_refuse_budgets_too_small_for_them(
    tmp_path: Path,
) -> None:
    from cytearc.neighbors.diffusion import bounded_diffusion_operator
    from cytearc.trajectory.artifacts import (
        load_diffusion_payload,
        write_diffusion_payload,
    )

    rng = np.random.default_rng(2)
    dense = rng.random((12, 12))
    graph = csr_matrix(np.where(dense > 0.6, dense + dense.T, 0.0))
    step = refused(lambda: bounded_diffusion_operator(graph, 3, memory_bytes=200))
    assert step.budget_bytes == 200
    # The step refuses a budget that its bytes reach, so one byte more passes.
    assert f"needs about {step.needed_bytes - 1} bytes" in str(step)
    operator = bounded_diffusion_operator(graph, 3, memory_bytes=10**9)

    group = zarr.open_group(str(tmp_path / "diffusion.zarr"), mode="w")
    write_diffusion_payload(group, operator)
    group.attrs.update(
        {
            "artifact_id": "a" * 64,
            "kind": "diffusion_operator",
            "provenance": {},
            "execution_options": {},
            "complete": True,
            "created_at_ns": 1,
            "cytearc_version": "test",
        }
    )
    load = refused(lambda: load_diffusion_payload(group, n_cells=12, memory_bytes=100))
    assert load.item == "loading the diffusion operator beside the imputed output"
    assert load.needed_bytes > load.budget_bytes == 100


def test_the_marker_search_names_the_budget_of_one_thread(tmp_path: Path) -> None:
    from cytearc.features.markers.search import _gene_major_schedule
    from tests.storage_helpers import write_count_store

    path = tmp_path / "markers.zarr"
    write_count_store(str(path), {"RNA": np.ones((30, 20), dtype=np.uint16)}, "uint16")
    counts_t = zarr.open_group(str(path), mode="r")["RNA/countsT"]

    def schedule(memory: int) -> tuple[int, int]:
        return _gene_major_schedule(
            counts_t,
            np.arange(30),
            np.arange(20),
            n_groups=3,
            group_features=20,
            resident_bytes=500,
            memory_bytes=memory,
            max_threads=4,
            max_calls=2,
        )

    refusal = named_budget_runs(
        schedule, 1_000, item="ranking one read group with one thread"
    )
    assert f"needs at least {refusal.needed_bytes} bytes" in str(refusal)
    assert schedule(refusal.needed_bytes) == (1, 1)


# The lineage curves' placement writes.


def test_cells_in_one_band_and_cells_over_many_read_their_values(
    tmp_path: Path,
) -> None:
    from cytearc.metadata.artifacts import plan_cell_data_artifact
    from cytearc.metadata.selection import resolve_cell_aligned_artifact
    from cytearc.storage.artifact_writer import artifact_transaction
    from cytearc.utils.memory_budget import MemoryLedger

    n_cells, width = 8_000, 50
    store = _count_store(tmp_path / "bands.zarr", np.ones((n_cells, 1)), "c")
    store.cells.insert("tail", np.arange(n_cells) >= n_cells - 3)
    store.cells.insert("spread", np.isin(np.arange(n_cells), [0, 4_000, n_cells - 1]))
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="quality_metric",
        operation="test_memory_refusals",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=store.snapshot_cell_selection("I"),
        arrays={"values": ((n_cells, width), None)},
    )
    values = np.arange(n_cells * width, dtype=np.float64).reshape(n_cells, width)
    with artifact_transaction(store.zw, planned) as group:
        # Eight bands of 1,000 cells.
        group.create_array("values", data=values, chunks=(1_000, width))
    for name, rows in (
        ("tail", [n_cells - 3, n_cells - 2, n_cells - 1]),
        ("spread", [0, 4_000, n_cells - 1]),
    ):
        resolved = resolve_cell_aligned_artifact(
            store.zw,
            planned.ref,
            cell_selection=store.snapshot_cell_selection(name),
            ndim=2,
            ledger=MemoryLedger(None, "Reading", codec_threads=4),
        )
        np.testing.assert_array_equal(resolved.values, values[rows])


def test_lineage_curves_refuse_a_budget_of_one_byte_before_placing_the_cells(
    datastore, lineage_fate, tmp_path: Path
) -> None:
    from tests.storage_helpers import copy_store

    store = copy_store(Path(datastore.zw.store.root), tmp_path)
    store.memoryBytes = 1
    refusal = refused(
        lambda: store.trajectory.curves(
            lineage_fate, lineages=[2], n_points=24, batch_size=97
        )
    )
    # The first check counts the placement writes beside the weights and ranks.
    assert refusal.item == "the lineages' weights, ranks and placement writes"


def test_the_producer_budget_beside_a_growing_writer_is_the_least_that_leaves_it() -> (
    None
):
    from cytearc.assay.normalization import _budget_beside_writer

    # A writer that takes half of the budget, and at least 100 bytes.
    def reserve(budget: int) -> int:
        return max(100, budget // 2)

    needed = _budget_beside_writer(reserve, 150, 300)
    assert needed == 599
    assert needed - reserve(needed) >= 300
    assert (needed - 1) - reserve(needed - 1) < 300
    # A writer of a fixed reserve adds it once.
    assert _budget_beside_writer(lambda _budget: 100, 150, 300) == 400
    # A budget that already leaves the producer its bytes is kept.
    assert _budget_beside_writer(reserve, 1_000, 300) == 1_000


def test_lineage_curves_place_their_cells_in_whole_bands_counted_up_front(
    datastore, lineage_fate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cytearc.datastore._operations.trajectory as trajectory
    from tests.storage_helpers import copy_store

    store = copy_store(Path(datastore.zw.store.root), tmp_path)

    def curves() -> Any:
        return store.trajectory.curves(
            lineage_fate,
            lineages=[2, 3],
            n_points=24,
            batch_size=97,
            invalidate_cache=True,
        )

    def placed(ref: Any) -> tuple[np.ndarray, np.ndarray]:
        group = store.artifacts.load(ref)
        return np.asarray(group["progress"][:]), np.asarray(group["distance"][:])

    whole = placed(curves())
    passes: list[int] = []
    writes: list[tuple[int, int]] = []
    blocks = trajectory._TrajectoryOperationsMixin._lineage_coordinate_blocks
    setitem = zarr.Array.__setitem__

    def recorded_blocks(self: Any, data: Any, **kwargs: Any) -> Any:
        passes.append(kwargs["resident_bytes"])
        return blocks(self, data, **kwargs)

    def recorded_write(array: Any, selection: Any, value: Any) -> Any:
        if array.path.rsplit("/", 1)[-1] == "progress":
            writes.append((selection.start, selection.stop))
        return setitem(array, selection, value)

    # Chunks of 300 rows, so a last band holds fewer.
    monkeypatch.setattr(trajectory, "_PLACEMENT_CHUNK_ROWS", 300)
    monkeypatch.setattr(
        trajectory._TrajectoryOperationsMixin,
        "_lineage_coordinate_blocks",
        recorded_blocks,
    )
    monkeypatch.setattr(zarr.Array, "__setitem__", recorded_write)
    banded = placed(curves())
    n_cells = whole[0].shape[0]
    assert n_cells % 300
    np.testing.assert_array_equal(banded[0], whole[0])
    np.testing.assert_array_equal(banded[1], whole[1])
    assert writes == [
        (start, min(start + 300, n_cells)) for start in range(0, n_cells, 300)
    ]
    # The placement plans beside the bytes of its writes, which the first
    # check counted with the weights and ranks that the fit plans beside.
    fitting, placing = passes
    assert placing - fitting == trajectory._placement_write_bytes(n_cells, 2, 300)
    for budget in (1, placing):
        store.memoryBytes = budget
        with pytest.raises(MemoryBudgetError) as refused:
            curves()
        assert refused.value.item == "the lineages' weights, ranks and placement writes"
        assert refused.value.needed_bytes == placing + 1


@pytest.mark.parametrize(
    "operation",
    ["fate", "lineage_tree", "diffusion", "imputation", "lineage_aggregation"],
)
def test_trajectory_operations_refuse_a_budget_of_one_byte(
    datastore,
    pseudotime_scoring,
    leiden_clustering,
    connectivity_graph,
    lineage_fate,
    mark_hvgs,
    tmp_path: Path,
    operation: str,
) -> None:
    from tests.fixtures_lineages import LINEAGE_SINKS
    from tests.storage_helpers import copy_store

    store = copy_store(Path(datastore.zw.store.root), tmp_path)
    diffusion = (
        store.imputation.diffusion(connectivity_graph, t=2)
        if operation == "imputation"
        else None
    )
    runs = {
        "fate": lambda: store.trajectory.fate(
            pseudotime_scoring,
            leiden_clustering,
            sinks=LINEAGE_SINKS,
            invalidate_cache=True,
        ),
        "lineage_tree": lambda: store.trajectory.lineage_tree(
            connectivity_graph, leiden_clustering, root=1, invalidate_cache=True
        ),
        "diffusion": lambda: store.imputation.diffusion(
            connectivity_graph, t=2, invalidate_cache=True
        ),
        # Forty features, whose imputed values outweigh the operator.
        "imputation": lambda: store.imputation.compute_imputed(
            list(store.get_assay("RNA").feats.fetch_all("names")[:40]),
            diffusion,
            from_assay="RNA",
        ),
        "lineage_aggregation": lambda: store.trajectory.lineage_aggregation(
            lineage_fate,
            features=mark_hvgs,
            n_bins=12,
            n_clusters=5,
            invalidate_cache=True,
        ),
    }

    # Each operation's own check refuses one byte, naming its step.
    store.memoryBytes = 1
    refusal = refused(runs[operation])
    assert refusal.budget_bytes == 1
    assert refusal.needed_bytes > 1
    assert refusal.item.startswith(
        {
            "fate": "the fate solver",
            "lineage_tree": "the cluster moments of the lineage tree",
            "diffusion": "a diffusion step of ",
            "imputation": "loading the diffusion operator beside the imputed output",
            "lineage_aggregation": "the bin operator beside the fate weights",
        }[operation]
    ), refusal.item


def test_a_row_band_writer_writes_whole_bands_and_a_last_partial_one() -> None:
    from cytearc.datastore._operations.enrichment_store import _RowBandWriter

    root = zarr.open_group(store=MemoryStore(), mode="w")
    writes: list[tuple[int, int]] = []
    setitem = zarr.Array.__setitem__

    def recorded(n_rows: int) -> tuple[zarr.Array, _RowBandWriter]:
        array = root.create_array(
            f"values{n_rows}", shape=(n_rows, 2), chunks=(16, 2), dtype=np.float64
        )

        class Recorded:
            shape, dtype, chunks = array.shape, array.dtype, array.chunks
            metadata = array.metadata

            def __setitem__(self, selection: slice, value: np.ndarray) -> None:
                writes.append((selection.start, selection.stop))
                setitem(array, selection, value)

        writer = _RowBandWriter(array)
        writer.array = Recorded()  # type: ignore[assignment]
        return array, writer

    array, writer = recorded(40)
    values = np.arange(80.0).reshape(40, 2)
    # Blocks of five rows fill fifteen rows of a band of sixteen, then cross it.
    for start in range(0, 40, 5):
        writer.add(values[start : start + 5])
    assert writes == [(0, 16), (16, 32)]
    writer.finish()
    assert writes == [(0, 16), (16, 32), (32, 40)]
    np.testing.assert_array_equal(array[:], values)

    # A last band of one row is written too.
    writes.clear()
    array, writer = recorded(33)
    values = np.arange(66.0).reshape(33, 2)
    writer.add(values)
    writer.finish()
    assert writes == [(0, 16), (16, 32), (32, 33)]
    np.testing.assert_array_equal(array[:], values)


def _banded_values(
    tmp_path: Path,
    name: str,
    n_cells: int,
    columns: dict[str, np.ndarray],
    *,
    width: int = 1,
) -> tuple[Path, Any, Any]:
    """A store of ``n_cells`` cells, with ``columns`` of flags, and a cell-aligned artifact of float64 values in bands of 1,000 cells.

    The values are one a cell, or rows of ``width`` values when it is more.
    """
    from cytearc.metadata.artifacts import plan_cell_data_artifact
    from cytearc.storage.artifact_writer import artifact_transaction

    path = tmp_path / f"{name}.zarr"
    shape = (n_cells,) if width == 1 else (n_cells, width)
    store = _count_store(path, np.ones((n_cells, 1)), "c")
    for column, flags in columns.items():
        store.cells.insert(column, flags)
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="quality_metric",
        operation="test_memory_refusals",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=store.snapshot_cell_selection("I"),
        arrays={"values": (shape, None)},
    )
    with artifact_transaction(store.zw, planned) as group:
        group.create_array(
            "values",
            data=np.arange(n_cells * width, dtype=np.float64).reshape(shape),
            chunks=(1_000, *shape[1:]),
        )
    return path, store, planned.ref


def test_cells_over_more_bands_than_the_fewest_are_refused_before_any_value_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cytearc.metadata.selection as selection_module
    from cytearc.metadata.selection import resolve_cell_aligned_artifact
    from cytearc.utils.memory_budget import MemoryLedger

    n_cells = 8_000
    spread = np.isin(np.arange(n_cells), [0, 4_000, n_cells - 1])
    _path, store, ref = _banded_values(
        tmp_path, "spread", n_cells, {"spread": spread}, width=50
    )
    selection = store.snapshot_cell_selection("spread")
    reads: list[int] = []
    rows = selection_module.read_rows

    def recorded(*args: Any, **kwargs: Any) -> Any:
        reads.append(1)
        return rows(*args, **kwargs)

    # 100 kB cannot hold the bands that three cells over the array span.
    monkeypatch.setattr(selection_module, "read_rows", recorded)
    refusal = refused(
        lambda: resolve_cell_aligned_artifact(
            store.zw,
            ref,
            cell_selection=selection,
            ndim=2,
            ledger=MemoryLedger(100_000, "Reading", codec_threads=4),
        )
    )
    assert str(refusal).endswith(
        "Read fewer cells with cell_selection, or raise mem_budget."
    ), refusal
    # No value or mask was read for the refusal.
    assert reads == []
