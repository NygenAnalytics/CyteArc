"""Fitting or naming the count-matrix layout for a memory budget."""

import pickle
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from cytearc.readers import H5adReader
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPolicy,
    load_count_matrix_plan,
    policy_from_payload,
)
from cytearc.storage.sharding import CountLayoutMemoryError, fit_count_layout
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import H5adToZarr, SparseToZarr


def _refusal(message: str, resources: ResourceBudget) -> MemoryBudgetError:
    """A planner's refusal of the counts write, one byte beyond the budget."""
    return MemoryBudgetError(
        message,
        needed_bytes=resources.memoryBytes + 1,
        budget_bytes=resources.memoryBytes,
        item="the counts write",
    )


def _fit(admitCounts, *, transposed=(), memoryBytes=1 << 40, requested=None):
    return fit_count_layout(
        {"RNA": (1_000, np.uint16)},
        nCells=1_000_000,
        profile="fast_local",
        memoryBytes=memoryBytes,
        transposed=transposed,
        admitCounts=admitCounts,
        requested=requested,
    )


def test_fit_uses_an_explicit_policy_exactly():
    explicit = CountMatrixPolicy(unitBytes=64_000, chunkBytes=16_000)
    seen: list[int] = []

    def admit(specs, resources: ResourceBudget) -> None:
        seen.append(specs[0].shards[0])

    assert _fit(admit, requested=explicit) == explicit
    assert seen == [32]

    def refuse(specs, resources: ResourceBudget) -> None:
        raise _refusal("does not fit", resources)

    with pytest.raises(
        CountLayoutMemoryError, match="requested count-matrix policy"
    ) as refused:
        _fit(refuse, requested=explicit)
    assert isinstance(refused.value, MemoryError)
    assert type(refused.value.__cause__) is MemoryBudgetError
    assert str(refused.value.__cause__) == "does not fit"


def test_fit_halves_the_default_policy_until_the_counts_write_fits():
    seen: list[int] = []
    workers: set[int] = set()

    def admit(specs, resources: ResourceBudget) -> None:
        workers.add(resources.workers)
        rows = specs[0].shards[0]
        seen.append(rows)
        if rows > 100_000:
            raise _refusal("band too tall", resources)

    fitted = _fit(admit)
    assert fitted == CountMatrixPolicy(
        unitBytes=DEFAULT_COUNT_MATRIX_POLICY.unitBytes // 8,
        chunkBytes=DEFAULT_COUNT_MATRIX_POLICY.chunkBytes // 8,
    )
    assert fitted.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    assert seen == [500_000, 250_000, 125_000, 62_500]
    # Admission plans one worker, so the layout never depends on the worker count.
    assert workers == {1}


def test_fit_names_the_largest_halving_of_a_requested_policy_that_fits():
    def admit(specs, resources: ResourceBudget) -> None:
        if specs[0].shards[0] > 100_000:
            raise _refusal("band too tall", resources)

    with pytest.raises(
        CountLayoutMemoryError, match="default count-matrix policy"
    ) as refused:
        _fit(admit, requested=DEFAULT_COUNT_MATRIX_POLICY)
    assert isinstance(refused.value, MemoryError)
    assert type(refused.value.__cause__) is MemoryBudgetError
    assert str(refused.value.__cause__) == "band too tall"
    # The same halving that a fit without a policy chooses.
    assert _named_policy(refused.value) == _fit(admit)


def _named_policy(error: BaseException) -> CountMatrixPolicy:
    ((unit, chunk),) = re.findall(
        r"policy=CountMatrixPolicy\(unitBytes=(\d+), chunkBytes=(\d+)\)", str(error)
    )
    return CountMatrixPolicy(unitBytes=int(unit), chunkBytes=int(chunk))


def test_fit_also_admits_the_counts_t_transpose():
    def admit(specs, resources: ResourceBudget) -> None:
        pass

    untransposed = _fit(admit, memoryBytes=64 * 1024**2)
    transposed = _fit(admit, transposed=("RNA",), memoryBytes=64 * 1024**2)
    assert untransposed == DEFAULT_COUNT_MATRIX_POLICY
    assert transposed.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes


def test_fit_stops_at_count_shards_of_one_row():
    seen: list[int] = []

    def refuse(specs, resources: ResourceBudget) -> None:
        seen.append(specs[0].shards[0])
        raise _refusal("does not fit", resources)

    with pytest.raises(MemoryError, match="count shards of one row") as refused:
        _fit(refuse)
    assert type(refused.value) is MemoryBudgetError
    assert seen[-1] == 1
    assert seen.count(1) == 1


def _write_h5ad(
    path: Path,
    values: np.ndarray,
    *,
    dense: bool = False,
    feature_types: list[bytes] | None = None,
) -> Path:
    with h5py.File(path, "w") as h5:
        if dense:
            h5.create_dataset("X", data=values)
        else:
            matrix = csr_matrix(values)
            group = h5.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = values.shape
            group.create_dataset("data", data=matrix.data)
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([f"c{i}".encode() for i in range(values.shape[0])])
        )
        var = h5.create_group("var")
        names = np.array([f"g{i}".encode() for i in range(values.shape[1])])
        var.create_dataset("_index", data=names)
        var.create_dataset("feature_name", data=names)
        if feature_types is not None:
            var.create_dataset("feature_types", data=np.array(feature_types))
        h5.create_group("obsm")
    return path


@pytest.fixture(scope="module")
def wide_counts(tmp_path_factory) -> tuple[Path, np.ndarray]:
    values = np.random.default_rng(0).poisson(0.3, size=(2_000, 600))
    # One count past uint16 keeps uint32 storage and wide count rows.
    values[0, 0] = 70_000
    path = tmp_path_factory.mktemp("layout") / "wide.h5ad"
    return _write_h5ad(path, values.astype(np.float32)), values


def _build(path: Path, destination: Path, **options) -> zarr.Group:
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        H5adToZarr(reader, str(destination), **options).dump()
    finally:
        reader.close()
    return zarr.open_group(str(destination), mode="r")


def _policy(root: zarr.Group, assay: str = "RNA") -> CountMatrixPolicy:
    return policy_from_payload(load_count_matrix_plan(root[f"{assay}/counts"]))


def test_h5ad_import_names_the_count_layout_that_fits_its_budget(wide_counts, tmp_path):
    path, values = wide_counts
    budget = 12 * 1024**2
    # The import keeps the default layout, which writes the whole matrix as
    # one band. That does not fit this budget, so it fails before the
    # destination exists and names the layout that fits.
    with pytest.raises(
        CountLayoutMemoryError, match="default count-matrix policy"
    ) as refused:
        _build(path, tmp_path / "default.zarr", mem_budget=budget, nthreads=1)
    assert isinstance(refused.value, MemoryError)
    assert not (tmp_path / "default.zarr").exists()
    named = _named_policy(refused.value)

    explicit = CountMatrixPolicy(
        unitBytes=DEFAULT_COUNT_MATRIX_POLICY.unitBytes // 2,
        chunkBytes=DEFAULT_COUNT_MATRIX_POLICY.chunkBytes // 2,
    )
    with pytest.raises(
        CountLayoutMemoryError, match="requested count-matrix policy"
    ) as refused_explicit:
        _build(
            path,
            tmp_path / "explicit.zarr",
            mem_budget=budget,
            nthreads=1,
            policy=explicit,
        )
    assert isinstance(refused_explicit.value, MemoryError)
    assert not (tmp_path / "explicit.zarr").exists()
    assert _named_policy(refused_explicit.value) == named

    roomy = _build(path, tmp_path / "roomy.zarr", mem_budget="1G", nthreads=1)
    fitted = _build(
        path, tmp_path / "fitted.zarr", mem_budget=budget, nthreads=1, policy=named
    )
    assert _policy(roomy) == DEFAULT_COUNT_MATRIX_POLICY
    policy = _policy(fitted)
    assert policy == named
    assert policy.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes
    assert policy.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    assert fitted["RNA/counts"].dtype == np.uint32
    assert fitted["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(fitted["RNA/counts"][:], values)
    np.testing.assert_array_equal(fitted["RNA/countsT"][:], values.T)
    # Identity does not depend on the layout.
    assert (
        fitted["RNA/counts"].attrs["content_fingerprint"]
        == roomy["RNA/counts"].attrs["content_fingerprint"]
    )

    # The named layout does not depend on the worker count.
    with pytest.raises(MemoryError, match="default count-matrix policy") as parallel:
        _build(path, tmp_path / "parallel.zarr", mem_budget=budget, nthreads=4)
    assert _named_policy(parallel.value) == named


def test_a_count_layout_refusal_states_the_producer_apart_from_resident_data(
    wide_counts, tmp_path
):
    path, _values = wide_counts
    with pytest.raises(CountLayoutMemoryError) as refused:
        _build(path, tmp_path / "default.zarr", mem_budget=12 * 1024**2, nthreads=1)
    found = re.search(
        r"Resident data needs about (\d+) bytes and the producer that buffers a "
        r"band of source rows about (\d+) bytes, but the operation limit is",
        str(refused.value),
    )
    assert found, str(refused.value)
    resident, producer = (int(number) for number in found.groups())
    # The matrix's resident data is a small part of what the plan refuses.
    assert 0 < resident < producer
    # The budget the refusal names is unchanged: both parts and one task.
    assert refused.value.needed_bytes == resident + producer + 1


@pytest.fixture(scope="module")
def tenth_nonzero(tmp_path_factory) -> tuple[Path, Path, np.ndarray]:
    """One matrix with a tenth of its values nonzero, as dense and CSR h5ad."""
    rng = np.random.default_rng(1)
    values = np.where(
        rng.random((2_000, 600)) < 0.1, rng.integers(1, 30, (2_000, 600)), 0
    ).astype(np.float32)
    directory = tmp_path_factory.mktemp("tenth")
    return (
        _write_h5ad(directory / "dense.h5ad", values, dense=True),
        _write_h5ad(directory / "csr.h5ad", values),
        values,
    )


def _nonzero_share(share: float) -> np.ndarray:
    """2,000 cells by 600 features, with ``share`` of the values nonzero."""
    rng = np.random.default_rng(1)
    return np.where(
        rng.random((2_000, 600)) < share, rng.integers(1, 30, (2_000, 600)), 0
    ).astype(np.float32)


@pytest.mark.parametrize("split", [False, True], ids=["one-assay", "split"])
@pytest.mark.parametrize("share", [0.1, 0.42], ids=["tenth", "screen"])
def test_a_dense_h5ad_writes_the_default_layout_below_the_sparse_plan(
    share, split, tmp_path
):
    values = _nonzero_share(share)
    # The last 40 features are antibodies, which a split stores apart.
    types = [b"Gene Expression"] * 560 + [b"Antibody Capture"] * 40 if split else None
    dense = _write_h5ad(
        tmp_path / "dense.h5ad", values, dense=True, feature_types=types
    )
    csr = _write_h5ad(tmp_path / "csr.h5ad", values, feature_types=types)
    options = {"nthreads": 1, "assay_split_key": "feature_types" if split else None}
    # Through the sparse writer, the default layout needs 9 to 15 MiB at a
    # tenth nonzero and 30 to 50 MiB at 42%, as the screen holds, for the CSR
    # file and before for the dense one. Through dense row bands, the dense
    # file needs about 8 MiB whatever its share of zeros.
    budget = 17 * 1024**2 // 2
    roots = {
        "dense": _build(dense, tmp_path / "dense.zarr", mem_budget=budget, **options)
    }
    with pytest.raises(CountLayoutMemoryError, match="default count-matrix policy"):
        _build(csr, tmp_path / "csr.zarr", mem_budget=budget, **options)
    roots["csr"] = _build(
        csr, tmp_path / "csr.zarr", mem_budget=96 * 1024**2, **options
    )
    expected = (
        {"RNA": values[:, :560], "ADT": values[:, 560:]} if split else {"RNA": values}
    )
    for assay, counts in expected.items():
        for root in roots.values():
            assert _policy(root, assay) == DEFAULT_COUNT_MATRIX_POLICY
            np.testing.assert_array_equal(root[f"{assay}/counts"][:], counts)
        stored = [root[f"{assay}/counts"] for root in roots.values()]
        assert len({array.attrs["content_fingerprint"] for array in stored}) == 1
        assert len({(array.dtype, array.chunks, array.shards) for array in stored}) == 1
        assert load_count_matrix_plan(stored[0]) == load_count_matrix_plan(stored[1])


def _reversed_rows(matrix: csr_matrix) -> csr_matrix:
    """The same values with each row's indices in descending order."""
    order = np.concatenate(
        [
            np.arange(stop - 1, start - 1, -1)
            for start, stop in zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True)
        ]
    )
    return csr_matrix(
        (matrix.data[order], matrix.indices[order], matrix.indptr),
        shape=matrix.shape,
    )


@pytest.fixture(scope="module")
def reversed_rows(tenth_nonzero, tmp_path_factory) -> Path:
    """The tenth-nonzero matrix as a CSR h5ad whose rows hold descending indices."""
    _dense, _csr, values = tenth_nonzero
    path = _write_h5ad(tmp_path_factory.mktemp("reversed") / "reversed.h5ad", values)
    with h5py.File(path, "r+") as h5:
        group = h5["X"]
        matrix = _reversed_rows(
            csr_matrix(
                (group["data"][:], group["indices"][:], group["indptr"][:]),
                shape=values.shape,
            )
        )
        assert not matrix.has_canonical_format
        group["data"][...] = matrix.data
        group["indices"][...] = matrix.indices
    return path


def test_an_h5ad_import_charges_sorting_only_for_rows_out_of_order(
    tenth_nonzero, reversed_rows, tmp_path
):
    dense, csr, values = tenth_nonzero
    # The CSR file's batches arrive canonical, so the default layout needs
    # about 9 MiB for it, and the dense file, written through dense row bands,
    # about 8 MiB. The same rows with their indices reversed are sorted and
    # summed, which needs about 17 MiB.
    budget = 14 * 1024**2
    roots = {
        name: _build(path, tmp_path / f"{name}.zarr", mem_budget=budget, nthreads=1)
        for name, path in (("dense", dense), ("csr", csr))
    }
    with pytest.raises(CountLayoutMemoryError, match="default count-matrix policy"):
        _build(reversed_rows, tmp_path / "reversed.zarr", mem_budget=budget, nthreads=1)
    roots["reversed"] = _build(
        reversed_rows, tmp_path / "reversed.zarr", mem_budget=19 * 1024**2, nthreads=1
    )
    for root in roots.values():
        assert _policy(root) == DEFAULT_COUNT_MATRIX_POLICY
        np.testing.assert_array_equal(root["RNA/counts"][:], values)
    fingerprints = {
        root["RNA/counts"].attrs["content_fingerprint"] for root in roots.values()
    }
    assert len(fingerprints) == 1


def _sparse_import(matrix: csr_matrix, budget: int) -> zarr.Group:
    store = MemoryStore()
    SparseToZarr(
        matrix,
        store,
        cell_ids=[f"c{i}" for i in range(matrix.shape[0])],
        feature_ids=[f"g{j}" for j in range(matrix.shape[1])],
        mem_budget=budget,
        nthreads=1,
    ).dump()
    return zarr.open_group(store=store, mode="r")


def test_sparse_to_zarr_charges_sorting_only_for_rows_out_of_order(tenth_nonzero):
    _dense, _csr, values = tenth_nonzero
    canonical = csr_matrix(values)
    reversed_matrix = _reversed_rows(canonical)
    # As from an h5ad file, the canonical rows need about 10 MiB, and the
    # same rows with their indices reversed, which are sorted, about 18 MiB.
    budget = 14 * 1024**2
    roots = {"canonical": _sparse_import(canonical, budget)}
    with pytest.raises(CountLayoutMemoryError, match="default count-matrix policy"):
        _sparse_import(reversed_matrix, budget)
    roots["reversed"] = _sparse_import(reversed_matrix, 20 * 1024**2)
    for root in roots.values():
        assert _policy(root) == DEFAULT_COUNT_MATRIX_POLICY
        np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert (
        roots["canonical"]["RNA/counts"].attrs["content_fingerprint"]
        == roots["reversed"]["RNA/counts"].attrs["content_fingerprint"]
    )


def test_h5ad_import_below_one_row_shards_fails_before_the_destination_exists(
    wide_counts, tmp_path
):
    path, _values = wide_counts
    destination = tmp_path / "tiny.zarr"
    # The count summaries alone need more than this budget.
    with pytest.raises(MemoryError, match="count shards of one row"):
        _build(path, destination, mem_budget=128 * 1024)
    assert not destination.exists()


def test_a_refused_layout_holds_the_largest_halving_that_fits():
    def admit(specs, resources: ResourceBudget) -> None:
        if specs[0].shards[0] > 100_000:
            raise _refusal("band too tall", resources)

    with pytest.raises(CountLayoutMemoryError) as default:
        _fit(admit, requested=DEFAULT_COUNT_MATRIX_POLICY)
    # The policy that the message names, which a fit without a policy chooses.
    assert default.value.fittingPolicy == _named_policy(default.value) == _fit(admit)

    # Shards of 150,000 rows, three chunks each, fit once halved.
    explicit = CountMatrixPolicy(unitBytes=300_000_000, chunkBytes=100_000_000)
    with pytest.raises(CountLayoutMemoryError, match="requested") as requested:
        _fit(admit, requested=explicit)
    assert requested.value.fittingPolicy == _named_policy(requested.value)
    assert requested.value.fittingPolicy == CountMatrixPolicy(
        unitBytes=150_000_000, chunkBytes=50_000_000
    )

    def refuse(specs, resources: ResourceBudget) -> None:
        raise _refusal("does not fit", resources)

    with pytest.raises(
        CountLayoutMemoryError, match="even count shards of one row do not fit"
    ) as nothing:
        _fit(refuse, requested=DEFAULT_COUNT_MATRIX_POLICY)
    assert nothing.value.fittingPolicy is None

    # Errors cross process boundaries, so the policy survives pickling.
    for error in (default.value, requested.value, nothing.value):
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is CountLayoutMemoryError
        assert restored.fittingPolicy == error.fittingPolicy
        assert str(restored) == str(error)
        assert (restored.needed_bytes, restored.budget_bytes, restored.item) == (
            error.needed_bytes,
            error.budget_bytes,
            error.item,
        )


def test_an_import_policy_requests_the_default_an_exact_layout_or_a_fit():
    from cytearc.storage.sharding import count_layout_request

    explicit = CountMatrixPolicy(unitBytes=64_000, chunkBytes=16_000)
    assert count_layout_request(None) == DEFAULT_COUNT_MATRIX_POLICY
    assert count_layout_request(explicit) is explicit
    assert count_layout_request("fit") is None
    with pytest.raises(
        ValueError, match="CountMatrixPolicy, \"fit\" or None, not 'auto'"
    ):
        count_layout_request("auto")
    with pytest.raises(TypeError, match='CountMatrixPolicy, "fit" or None, not int'):
        count_layout_request(3)


@dataclass(frozen=True)
class _Import:
    """An import of a source whose default layout ``budget`` refuses.

    ``build(destination, policy, mem_budget, dump)`` constructs the writer,
    writes the store when ``dump`` is true, and closes the reader.
    """

    build: Callable[[str, Any, int, bool], Any]
    budget: int
    counts: Mapping[str, np.ndarray] | None


def _write_csv(path: Path, values: np.ndarray) -> Path:
    header = "cell," + ",".join(f"g{j}" for j in range(values.shape[1]))
    rows = (f"c{i}," + ",".join(map(str, row)) for i, row in enumerate(values))
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def _write_10x_h5(path: Path, values: np.ndarray) -> Path:
    matrix = csr_matrix(values)
    n_cells, n_features = values.shape
    with h5py.File(path, "w") as h5:
        group = h5.create_group("matrix")
        group.create_dataset("data", data=matrix.data.astype(np.int32))
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset("shape", data=np.array([n_features, n_cells], np.int32))
        group.create_dataset(
            "barcodes", data=np.array([f"c{i}".encode() for i in range(n_cells)])
        )
        features = group.create_group("features")
        names = np.array([f"g{j}".encode() for j in range(n_features)])
        features.create_dataset("id", data=names)
        features.create_dataset("name", data=names)
        features.create_dataset(
            "feature_type", data=np.array([b"Gene Expression"] * n_features)
        )
    return path


def _write_seurat(path: Path, values: np.ndarray) -> Path:
    """Write a Seurat object whose RNA assay holds ``values`` as dense counts."""
    from tests.test_seurat_reader import _Wire

    wire = _Wire()
    cells = [f"c{i}" for i in range(values.shape[0])]
    genes = [f"g{j}" for j in range(values.shape[1])]
    # An R matrix of genes by cells, stored column by column.
    counts = wire.matrix(
        values.reshape(-1).tolist(), values.T.shape, rows=genes, columns=cells
    )
    assay = wire.s4(
        [
            ("counts", counts),
            (
                "meta.features",
                wire.data_frame([("symbol", wire.string_vector(genes))], genes),
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    [("group", wire.string_vector(["a"] * len(cells)))], cells
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1] * len(cells), ["cells"], names=cells)),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _writing(writer: Any, dump: bool) -> Any:
    if dump:
        writer.dump()
    return writer


@pytest.fixture(scope="module")
def imports(wide_counts, tmp_path_factory) -> dict[str, _Import]:
    """Each import writer, at a budget that refuses its default layout.

    The budgets are fixed for the sources' sizes: each lies below what the
    default layout needs and above what a halving of it needs.
    """
    from cytearc.readers import CrH5Reader, CSVReader, TapestriReader
    from cytearc.readers.seurat import SeuratReader
    from cytearc.writers import CrToZarr, CSVtoZarr, SeuratToZarr, TapestriToZarr
    from tests.variant_truth import tapestri_truth, write_tapestri

    path, values = wide_counts
    directory = tmp_path_factory.mktemp("imports")
    dense = _write_h5ad(directory / "dense.h5ad", values.astype(np.float32), dense=True)
    csv = _write_csv(directory / "counts.csv", values)
    h5 = _write_10x_h5(directory / "counts.h5", values)
    tapestri = write_tapestri(tapestri_truth(), directory / "run.dna+protein.h5")
    seurat_values = values[:600, :300]
    seurat = _write_seurat(directory / "counts.rds", seurat_values)
    sparse = csr_matrix(values.astype(np.float32))

    def h5ad(source: Path) -> Callable[[str, Any, int, bool], Any]:
        def build(destination: str, policy: Any, budget: int, dump: bool) -> Any:
            reader = H5adReader(str(source), feature_name_key="feature_name")
            try:
                return _writing(
                    H5adToZarr(
                        reader,
                        destination,
                        mem_budget=budget,
                        nthreads=1,
                        policy=policy,
                    ),
                    dump,
                )
            finally:
                reader.close()

        return build

    def sparse_build(destination: str, policy: Any, budget: int, dump: bool) -> Any:
        writer = SparseToZarr(
            sparse,
            destination,
            cell_ids=[f"c{i}" for i in range(sparse.shape[0])],
            feature_ids=[f"g{j}" for j in range(sparse.shape[1])],
            mem_budget=budget,
            nthreads=1,
            policy=policy,
        )
        return _writing(writer, dump)

    def csv_build(destination: str, policy: Any, budget: int, dump: bool) -> Any:
        # Reader batches of 100 rows keep the file's chunk below the budget.
        reader = CSVReader(str(csv), id_column=0, batch_size=100)
        writer = CSVtoZarr(
            reader,
            destination,
            assay_name="RNA",
            mem_budget=budget,
            nthreads=1,
            policy=policy,
        )
        return _writing(writer, dump)

    def reader_build(open_reader: Callable[[], Any], writer_class: Any) -> Any:
        def build(destination: str, policy: Any, budget: int, dump: bool) -> Any:
            reader = open_reader()
            try:
                return _writing(
                    writer_class(
                        reader,
                        destination,
                        mem_budget=budget,
                        nthreads=1,
                        policy=policy,
                    ),
                    dump,
                )
            finally:
                reader.close()

        return build

    def seurat_build(destination: str, policy: Any, budget: int, dump: bool) -> Any:
        with SeuratReader(str(seurat), assays=["RNA"], reductions=[]) as reader:
            writer = SeuratToZarr(
                reader, destination, mem_budget=budget, nthreads=1, policy=policy
            )
            return _writing(writer, dump)

    return {
        "h5ad_csr": _Import(h5ad(path), 12 * 1024**2, {"RNA": values}),
        "h5ad_dense": _Import(h5ad(dense), 3 * 1024**2, {"RNA": values}),
        "sparse": _Import(sparse_build, 12 * 1024**2, {"RNA": values}),
        "csv": _Import(csv_build, 8 * 1024**2, {"RNA": values}),
        "cellranger_h5": _Import(
            reader_build(lambda: CrH5Reader(str(h5)), CrToZarr),
            12 * 1024**2,
            {"RNA": values},
        ),
        "tapestri": _Import(
            reader_build(lambda: TapestriReader(str(tapestri)), TapestriToZarr),
            64 * 1024,
            None,
        ),
        "seurat": _Import(seurat_build, 2 * 1024**2, {"RNA": seurat_values}),
    }


_IMPORT_NAMES = (
    "h5ad_csr",
    "h5ad_dense",
    "sparse",
    "csv",
    "cellranger_h5",
    "tapestri",
    "seurat",
)


def _layouts(writer: Any) -> set[CountMatrixPolicy]:
    """The layouts that a writer reports, one per assay or one for all."""
    policy = writer.policy
    return set(policy.values()) if isinstance(policy, dict) else {policy}


def _count_assays(root: zarr.Group) -> list[str]:
    return sorted(name for name in root.group_keys() if "counts" in root[name])


@pytest.mark.parametrize("name", _IMPORT_NAMES)
def test_policy_fit_writes_the_layout_that_the_default_refusal_names(
    name, imports, tmp_path
):
    case = imports[name]
    with pytest.raises(
        CountLayoutMemoryError, match="default count-matrix policy"
    ) as refused:
        case.build(str(tmp_path / "default.zarr"), None, case.budget, False)
    assert not (tmp_path / "default.zarr").exists()
    fitting = refused.value.fittingPolicy
    assert fitting == _named_policy(refused.value)
    assert fitting.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes

    fitted = case.build(str(tmp_path / "fit.zarr"), "fit", case.budget, True)
    explicit = case.build(str(tmp_path / "explicit.zarr"), fitting, case.budget, True)
    assert _layouts(fitted) == _layouts(explicit) == {fitting}
    roots = [
        zarr.open_group(str(tmp_path / f"{store}.zarr"), mode="r")
        for store in ("fit", "explicit")
    ]
    assays = _count_assays(roots[0])
    assert assays and assays == _count_assays(roots[1])
    for assay in assays:
        stored = [root[f"{assay}/counts"] for root in roots]
        # The same counts, identity and layout record as the explicit import.
        assert {_policy(root, assay) for root in roots} == {fitting}
        assert load_count_matrix_plan(stored[0]) == load_count_matrix_plan(stored[1])
        np.testing.assert_array_equal(stored[0][:], stored[1][:])
        assert (
            stored[0].attrs["content_fingerprint"]
            == stored[1].attrs["content_fingerprint"]
        )
        if case.counts is not None:
            np.testing.assert_array_equal(stored[0][:], case.counts[assay])


@pytest.mark.parametrize("name", _IMPORT_NAMES)
def test_policy_fit_keeps_the_default_layout_when_it_fits(name, imports, tmp_path):
    writer = imports[name].build(
        str(tmp_path / "roomy.zarr"), "fit", 256 * 1024**2, False
    )
    assert _layouts(writer) == {DEFAULT_COUNT_MATRIX_POLICY}


def test_policy_fit_below_count_shards_of_one_row_refuses_as_subset_does(
    wide_counts, tmp_path
):
    path, _values = wide_counts
    destination = tmp_path / "tiny.zarr"
    with pytest.raises(MemoryBudgetError, match="count shards of one row") as fitted:
        _build(path, destination, mem_budget=128 * 1024, policy="fit")
    assert type(fitted.value) is MemoryBudgetError
    with pytest.raises(CountLayoutMemoryError, match="even count shards") as default:
        _build(path, destination, mem_budget=128 * 1024)
    assert default.value.fittingPolicy is None
    assert not destination.exists()


def test_policy_fit_fits_each_assay_of_a_sparse_import_on_its_own(wide_counts):
    _path, values = wide_counts
    narrow = np.random.default_rng(2).poisson(0.3, size=(values.shape[0], 40))
    matrices = {
        "RNA": csr_matrix(values.astype(np.float32)),
        "ADT": csr_matrix(narrow.astype(np.float32)),
    }
    options = {
        "cell_ids": [f"c{i}" for i in range(values.shape[0])],
        "feature_ids": {
            name: [f"{name}{j}" for j in range(matrix.shape[1])]
            for name, matrix in matrices.items()
        },
        "mem_budget": 12 * 1024**2,
        "nthreads": 1,
    }
    with pytest.raises(CountLayoutMemoryError) as refused:
        SparseToZarr(matrices, MemoryStore(), **options)
    store = MemoryStore()
    writer = SparseToZarr(matrices, store, policy="fit", **options)

    # Only the wide assay needs smaller shards than the default.
    assert writer.policy == {
        "RNA": refused.value.fittingPolicy,
        "ADT": DEFAULT_COUNT_MATRIX_POLICY,
    }
    writer.dump()
    root = zarr.open_group(store=store, mode="r")
    for name, matrix in matrices.items():
        assert _policy(root, name) == writer.policy[name]
        np.testing.assert_array_equal(root[f"{name}/counts"][:], matrix.toarray())


@pytest.mark.parametrize(("policy", "error"), [("auto", ValueError), (3, TypeError)])
def test_an_import_refuses_a_policy_it_cannot_read_before_the_destination_exists(
    policy, error, wide_counts, tmp_path
):
    path, _values = wide_counts
    with pytest.raises(
        error, match='policy must be a CountMatrixPolicy, "fit" or None'
    ):
        _build(path, tmp_path / "unread.zarr", policy=policy)
    assert not (tmp_path / "unread.zarr").exists()
