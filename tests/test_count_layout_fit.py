"""Fitting or naming the count-matrix layout for a memory budget."""

import re
from pathlib import Path

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


def _write_h5ad(path: Path, values: np.ndarray, *, dense: bool = False) -> Path:
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


def _policy(root: zarr.Group) -> CountMatrixPolicy:
    return policy_from_payload(load_count_matrix_plan(root["RNA/counts"]))


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


def test_a_dense_h5ad_plans_the_memory_of_its_nonzero_values(tenth_nonzero, tmp_path):
    dense, csr, values = tenth_nonzero
    # Planned as if every value were nonzero, the dense file's default layout
    # needed about 224 MiB. Its nonzero values need what the CSR file's do,
    # about 14 MiB, beside its dense rows.
    budget = 64 * 1024**2
    stores = {
        name: _build(path, tmp_path / f"{name}.zarr", mem_budget=budget, nthreads=1)
        for name, path in (("dense", dense), ("csr", csr))
    }
    for root in stores.values():
        assert _policy(root) == DEFAULT_COUNT_MATRIX_POLICY
        np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert (
        stores["dense"]["RNA/counts"].attrs["content_fingerprint"]
        == stores["csr"]["RNA/counts"].attrs["content_fingerprint"]
    )


def test_a_dense_h5ad_still_plans_its_dense_rows(tenth_nonzero, tmp_path):
    dense, csr, _values = tenth_nonzero
    # The CSR file fits 17 MiB. A batch of the dense file also holds its
    # rows' 2,000 by 600 float32 values, 4.6 MiB, so the same nonzero values
    # do not.
    budget = 17 * 1024**2
    root = _build(csr, tmp_path / "csr.zarr", mem_budget=budget, nthreads=1)
    assert _policy(root) == DEFAULT_COUNT_MATRIX_POLICY
    with pytest.raises(CountLayoutMemoryError, match="default count-matrix policy"):
        _build(dense, tmp_path / "dense.zarr", mem_budget=budget, nthreads=1)


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
    # Sorting each batch and summing its repeated coordinates was charged for
    # every source, so the default layout needed about 24 MiB for the CSR
    # file and 29 MiB for the dense one. Their batches arrive canonical, so
    # they need about 14 and 19 MiB. The same rows with their indices
    # reversed are sorted, and still need about 24 MiB.
    budget = 22 * 1024**2
    roots = {
        name: _build(path, tmp_path / f"{name}.zarr", mem_budget=budget, nthreads=1)
        for name, path in (("dense", dense), ("csr", csr))
    }
    with pytest.raises(CountLayoutMemoryError, match="default count-matrix policy"):
        _build(reversed_rows, tmp_path / "reversed.zarr", mem_budget=budget, nthreads=1)
    roots["reversed"] = _build(
        reversed_rows, tmp_path / "reversed.zarr", mem_budget=26 * 1024**2, nthreads=1
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
    # As from an h5ad file, the canonical rows need about 15 MiB instead of
    # 25 MiB, and the same rows with their indices reversed still need 25 MiB.
    budget = 20 * 1024**2
    roots = {"canonical": _sparse_import(canonical, budget)}
    with pytest.raises(CountLayoutMemoryError, match="default count-matrix policy"):
        _sparse_import(reversed_matrix, budget)
    roots["reversed"] = _sparse_import(reversed_matrix, 28 * 1024**2)
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
