"""Exports write cells in the order of the imported file; merges, subsets, repacks and mounts keep that order.

A shuffled H5AD import writes row ``r`` from file row ``order[r]`` and keeps
each cell's file row in ``sourceRow``. Exports of it are checked against the
file itself and against the exports of a plain import of the same file,
which the other export tests pin. Run exports in source order are checked
against the same exports in store order, gathered in the order of the
cells' file rows, which the cell IDs, ``cell-<row>``, also give.
"""

import gzip
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from loguru import logger
from scipy.io import mmread
from scipy.sparse import csr_matrix

import cytearc.readers._sparse as sparse_module
import cytearc.writers.export as export
from cytearc import DataStore
from cytearc.composite import create_composite
from cytearc.datastore.datastore import mount_datastore
from cytearc.merge import DataStoreMerge
from cytearc.readers import H5adReader
from cytearc.readers._sparse import SparseRowStore
from cytearc.storage.cell_order import (
    CELL_ORDER_ATTR,
    SOURCE_ROW,
    SOURCE_ROW_ROLE,
    copy_cell_order,
    exported_cell_order,
    recorded_shuffle_seed,
    shuffled_order,
    source_order_positions,
    source_order_ranks,
    write_shuffled_cell_order,
)
from cytearc.tools.repack_zarr import repack_store
from cytearc.utils.memory_budget import MemoryBudgetError
from cytearc.writers import H5adToZarr, SubsetZarr, to_h5ad, to_mtx
from cytearc.writers.export import (
    H5adMatrix,
    in_memory_reordered_matrix,
    iter_h5ad_blocks,
    materialize_h5ad_matrix,
    reorder_held_bytes,
    reordered_matrix,
)
from tests.row_store_helpers import asked_bytes, finalized_bytes, free_disk
from tests.storage_helpers import by_cell_id

anndata = pytest.importorskip("anndata")

_N_CELLS = 157
_N_FEATURES = 12
_SEED = 0
_ORDER = shuffled_order(_N_CELLS, _SEED)


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _counts() -> np.ndarray:
    """Counts with empty cells, an empty feature, and a count past uint16."""
    counts = np.random.default_rng(1).poisson(0.9, (_N_CELLS, _N_FEATURES))
    counts = counts.astype(np.float32)
    counts[[3, 17]] = 0
    counts[:, 6] = 0
    counts[40, 3] = 70_000
    return counts


_COUNTS = _counts()


def _obs() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    score = rng.normal(size=_N_CELLS)
    score[[5, 50]] = np.nan
    stage = pd.Categorical(
        rng.choice(["early", "mid", "late"], _N_CELLS),
        categories=["early", "mid", "late", "unused"],
        ordered=True,
    )
    stage[[1, 8]] = np.nan
    nullable = pd.array(rng.integers(0, 9, _N_CELLS), dtype="Int64")
    nullable[[0, 4]] = pd.NA
    return pd.DataFrame(
        {
            "score": score,
            "count": rng.integers(0, 1000, _N_CELLS),
            "flag": rng.random(_N_CELLS) < 0.5,
            "label": [f"L{value}" for value in rng.integers(0, 50, _N_CELLS)],
            "stage": stage,
            "nullable": nullable,
            "keep": np.arange(_N_CELLS) % 8 != 5,
        },
        index=pd.Index([f"cell-{row}" for row in range(_N_CELLS)]),
    )


def _write_source(path: Path) -> None:
    rng = np.random.default_rng(3)
    anndata.AnnData(
        csr_matrix(_COUNTS),
        obs=_obs(),
        var=pd.DataFrame(
            {"feature_name": [f"gene-{f}" for f in range(_N_FEATURES)]},
            index=[f"feature-{f}" for f in range(_N_FEATURES)],
        ),
        obsm={
            "X_umap": rng.normal(size=(_N_CELLS, 2)).astype(np.float32),
            "X_pca": rng.normal(size=(_N_CELLS, 5)),
        },
    ).write_h5ad(path)


def _import(source: Path, destination: Path, **options: Any) -> Any:
    reader = H5adReader(
        str(source),
        feature_name_key="feature_name",
        embedding_roles={"X_umap": "umap", "X_pca": "coordinates"},
    )
    try:
        return H5adToZarr(
            reader, str(destination), mem_budget="256M", nthreads=1, **options
        ).dump()
    finally:
        reader.close()


def _open(path: Path, **options: Any) -> DataStore:
    return DataStore(str(path), min_features_per_cell=0, nthreads=1, **options)


@pytest.fixture(scope="module")
def imported(tmp_path_factory) -> SimpleNamespace:
    """A file, its import in the file's order, and its shuffled import, kept read only."""
    folder = tmp_path_factory.mktemp("source_order")
    source = folder / "source.h5ad"
    _write_source(source)
    plain = _import(source, folder / "plain.zarr", shuffle_cells=False)
    shuffled = _import(
        source, folder / "shuffled.zarr", shuffle_cells=True, shuffle_seed=_SEED
    )
    # The first open prepares each store, so later opens write nothing.
    _open(folder / "plain.zarr")
    _open(folder / "shuffled.zarr")
    return SimpleNamespace(
        folder=folder,
        source=source,
        plain=folder / "plain.zarr",
        shuffled=folder / "shuffled.zarr",
        plainImport=plain,
        shuffledImport=shuffled,
    )


def _copy(path: Path, folder: Path) -> Path:
    target = folder / path.name
    shutil.copytree(path, target)
    return target


def _h5_tree(path: Path, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Every dataset's dtype and bytes and every attribute of an HDF5 file, by path."""
    tree: dict[str, Any] = {}

    def attributes(name: str, node: Any) -> None:
        tree[f"{name}@"] = {key: repr(value) for key, value in node.attrs.items()}

    def visit(name: str, node: Any) -> None:
        if any(name == part or name.startswith(f"{part}/") for part in skip):
            return
        attributes(name, node)
        if isinstance(node, h5py.Dataset):
            values = node[()]
            if node.dtype.kind == "O":
                tree[name] = ("O", [repr(value) for value in np.ravel(values)])
            else:
                tree[name] = (str(node.dtype), np.asarray(values).tobytes())

    with h5py.File(path, "r") as h5:
        attributes("/", h5)
        h5.visititems(visit)
    return tree


def _file_rows(ids: Any) -> np.ndarray:
    """The file row of each cell, from its ID ``cell-<row>``, after any merge prefix."""
    return np.asarray([int(str(value).rsplit("cell-", 1)[1]) for value in ids])


# ---------------------------------------------------------------------------
# The cell order helpers


def _table(source_rows: np.ndarray | None, *, role: bool = True) -> zarr.Group:
    group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    n = 0 if source_rows is None else source_rows.size
    group.create_array("ids", data=np.asarray([f"c{i}" for i in range(n)]))
    if source_rows is not None:
        array = group.create_array(SOURCE_ROW, data=np.asarray(source_rows, np.int64))
        if role:
            array.attrs["role"] = SOURCE_ROW_ROLE
    return group


def test_positions_sort_rows_by_their_source_rows_and_keep_ties() -> None:
    table = _table(np.asarray([4, 0, 7, 0, 2, 9]))
    np.testing.assert_array_equal(
        source_order_positions(table, np.arange(6)), [1, 3, 4, 0, 2, 5]
    )
    # Positions are among the given rows: rows 0, 2 and 4 hold 4, 7 and 2.
    np.testing.assert_array_equal(
        source_order_positions(table, np.asarray([0, 2, 4])), [2, 0, 1]
    )
    # Rows whose source rows never decrease, ties included, keep store order.
    assert source_order_positions(table, np.asarray([1, 3, 4, 5])) is None
    assert source_order_positions(table, np.zeros(0, dtype=np.int64)) is None
    # A plain column, without CyteArc's role, does not order the rows.
    plain = _table(np.asarray([3, 2, 1]), role=False)
    assert source_order_positions(plain, np.arange(3)) is None
    assert source_order_positions(_table(None), np.zeros(0, dtype=np.int64)) is None


@pytest.mark.parametrize(
    "rows",
    [
        np.asarray([2, 1]),
        np.asarray([0, 0]),
        np.asarray([-1, 2]),
        np.asarray([1, 6]),
        np.asarray([[0, 1]]),
        np.asarray([0.0, 1.0]),
    ],
    ids=["decreasing", "repeated", "negative", "beyond", "2d", "float"],
)
def test_positions_refuse_rows_that_are_not_increasing_rows_of_the_table(rows):
    with pytest.raises(ValueError, match="increasing integer rows of the 6 cells"):
        source_order_positions(_table(np.arange(6)), rows)


def test_positions_refuse_a_malformed_source_row_column() -> None:
    table = _table(np.arange(4))
    table.create_array(SOURCE_ROW, data=np.arange(4, dtype=np.int32), overwrite=True)
    table[SOURCE_ROW].attrs["role"] = SOURCE_ROW_ROLE
    with pytest.raises(ValueError, match="one int64 row of the imported file"):
        source_order_positions(table, np.arange(4))


def test_ranks_place_each_cell_in_source_order() -> None:
    np.testing.assert_array_equal(
        source_order_ranks(np.asarray([40, 10, 30, 10])), [3, 0, 2, 1]
    )
    assert source_order_ranks(np.zeros(0, dtype=np.int64)).tolist() == []
    with pytest.raises(ValueError, match="one integer per cell"):
        source_order_ranks(np.asarray([0.5, 1.0]))


@pytest.mark.parametrize(
    ("value", "error"),
    [(None, TypeError), (1, TypeError), ("Source", ValueError), ("file", ValueError)],
)
def test_cell_order_takes_source_or_stored(value, error) -> None:
    table = _table(np.asarray([1, 0]))
    with pytest.raises(error, match="cell_order must be 'source'"):
        exported_cell_order(table, np.arange(2), value)
    assert exported_cell_order(table, np.arange(2), "stored") is None
    np.testing.assert_array_equal(
        exported_cell_order(table, np.arange(2), "source"), [1, 0]
    )


def test_the_cell_order_record_is_copied_checked_or_not_at_all() -> None:
    source, target = _table(np.arange(3)), _table(np.arange(3))
    copy_cell_order(source, target)
    assert CELL_ORDER_ATTR not in target.attrs
    source.attrs[CELL_ORDER_ATTR] = {"shuffled": True, "seed": 7}
    copy_cell_order(source, target)
    assert recorded_shuffle_seed(target) == 7
    source.attrs[CELL_ORDER_ATTR] = {"shuffled": True}
    with pytest.raises(ValueError, match="malformed"):
        copy_cell_order(source, _table(np.arange(3)))


# ---------------------------------------------------------------------------
# Reordering streamed rows


def _blocks(values: np.ndarray, rows: int) -> Iterator[np.ndarray]:
    for start in range(0, values.shape[0], rows):
        yield values[start : start + rows]


def _matrix(values: np.ndarray, rows: int = 7, **options: Any) -> H5adMatrix:
    return H5adMatrix(
        shape=values.shape,
        dtype=values.dtype,
        blocks=lambda: _blocks(values, rows),
        **options,
    )


@pytest.mark.parametrize(
    "dtype", [np.uint16, np.int64, np.float32, np.float64, np.bool_]
)
def test_reordered_rows_equal_the_rows_gathered_in_memory(dtype, tmp_path) -> None:
    rng = np.random.default_rng(4)
    values = (rng.random((211, 9)) < 0.3) * rng.integers(1, 2**15, (211, 9))
    values = values.astype(dtype)
    if np.dtype(dtype).kind == "f":
        values[5, 2] = np.nan
        values[6, 3] = 1e30 if dtype == np.float64 else 3.5
        values[7, 4] = -2.25
    if dtype == np.int64:
        values[8, 1] = np.iinfo(np.int64).max
        values[9, 0] = np.iinfo(np.int64).min
    values[10] = 0
    order = np.random.default_rng(5).permutation(211)

    matrix = reordered_matrix(
        _matrix(values), order, max_bytes=64 << 20, temp_dir=tmp_path
    )
    # The rows go through the temporary files only once they are read.
    assert list(tmp_path.iterdir()) == []
    exported = materialize_h5ad_matrix(matrix)

    expected = csr_matrix(values[order])
    assert exported.dtype == expected.dtype
    np.testing.assert_array_equal(exported.indptr, expected.indptr)
    np.testing.assert_array_equal(exported.indices, expected.indices)
    np.testing.assert_array_equal(
        exported.data.view(np.uint8), expected.data.view(np.uint8)
    )
    assert list(tmp_path.iterdir()) == []
    in_memory = materialize_h5ad_matrix(
        in_memory_reordered_matrix(_matrix(values), order)
    )
    np.testing.assert_array_equal(in_memory.toarray(), values[order])


def test_reordered_rows_cross_many_buckets_and_are_deleted_as_read(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(6)
    values = rng.poisson(2.0, (500, 40)).astype(np.uint32)
    order = rng.permutation(500)
    # Buckets of a few kilobytes, so the rows spread over many of them.
    monkeypatch.setattr(export, "_REORDER_READ_BYTES", 4096)
    matrix = reordered_matrix(
        _matrix(values, rows=33),
        order,
        max_bytes=64 << 20,
        temp_dir=tmp_path,
        entries=int(np.count_nonzero(values)),
    )
    pieces = []
    remaining = []
    for piece in iter_h5ad_blocks(matrix):
        pieces.append(piece.toarray())
        (directory,) = tmp_path.iterdir()
        remaining.append(
            sum(
                path.is_dir() and path.name.startswith("rows-")
                for path in directory.iterdir()
            )
        )
    np.testing.assert_array_equal(np.vstack(pieces), values[order])
    # Each bucket's files are deleted once its rows are taken.
    assert remaining[0] > 10
    assert remaining == sorted(remaining, reverse=True)
    assert remaining[-1] < remaining[0]
    assert list(tmp_path.iterdir()) == []


def test_a_reorder_of_no_rows_writes_an_empty_matrix(tmp_path) -> None:
    values = np.zeros((0, 4), dtype=np.int32)
    exported = materialize_h5ad_matrix(
        reordered_matrix(
            _matrix(values),
            np.zeros(0, dtype=np.int64),
            max_bytes=1 << 20,
            temp_dir=tmp_path,
        )
    )
    assert exported.shape == (0, 4)
    assert exported.dtype == np.int32
    assert list(tmp_path.iterdir()) == []


def test_reordered_rows_keep_the_export_dtype_cast(tmp_path) -> None:
    values = np.asarray([[1, 0], [0, 70_000], [3, 4]], dtype=np.uint32)
    order = np.asarray([2, 0, 1])
    matrix = reordered_matrix(
        _matrix(values, export_dtype=np.dtype(np.float32)),
        order,
        max_bytes=64 << 20,
        temp_dir=tmp_path,
    )
    exported = materialize_h5ad_matrix(matrix)
    assert exported.dtype == np.float32
    np.testing.assert_array_equal(exported.toarray(), values[order])
    overflow = reordered_matrix(
        _matrix(values, export_dtype=np.dtype(np.uint16)),
        order,
        max_bytes=64 << 20,
        temp_dir=tmp_path,
    )
    with pytest.raises(OverflowError):
        materialize_h5ad_matrix(overflow)
    # The failed export leaves no temporary files.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "order",
    [
        np.asarray([0, 1]),
        np.asarray([0, 1, 1]),
        np.asarray([0, 1, 3]),
        np.asarray([-1, 0, 1]),
        np.asarray([0.0, 1.0, 2.0]),
    ],
    ids=["short", "repeated", "beyond", "negative", "float"],
)
def test_a_reorder_takes_only_a_permutation_of_the_rows(order) -> None:
    values = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="permutation of its 3 rows"):
        reordered_matrix(_matrix(values), order, max_bytes=64 << 20)


def test_a_reorder_refuses_a_budget_below_spilling_the_rows(tmp_path) -> None:
    values = np.random.default_rng(10).poisson(0.5, (1000, 50)).astype(np.float32)
    order = np.arange(1000)[::-1].copy()
    # Hand-computed. The order and destination of 1,000 rows, 16 bytes each;
    # the bucket writer's 9 bytes per row and 25 bytes for each of its
    # bucket and the end, since 1,000 dense rows of 50 float32 values read
    # in one bucket (44 bytes per value, 80 per row, within 256 MiB), and 8
    # bytes for the bucket and the end; 64 bytes for each of the 50,000
    # values of one conversion step, all 1,000 rows, gathered and appended;
    # and the conversion of one row: 50 values of 28 bytes and 2 row offsets
    # of 20.
    spill = 16_000 + (9_000 + 50) + 16 + 64 * 50_000 + (50 * 28 + 2 * 20)
    # Reading them back holds 1,001 row offsets and three reads of a row of
    # 50 values: 50 values of 44 bytes, its piece's 48 and two rows' 32.
    read = 8 * 1001 + 3 * (50 * 44 + 48 + 64)
    assert spill > read
    with pytest.raises(MemoryBudgetError) as refused:
        reordered_matrix(_matrix(values), order, max_bytes=spill - 1)
    assert refused.value.needed_bytes == spill
    assert "cell_order='stored'" in str(refused.value)
    # At that budget the rows go one conversion step at a time.
    exported = materialize_h5ad_matrix(
        reordered_matrix(_matrix(values), order, max_bytes=spill, temp_dir=tmp_path)
    )
    np.testing.assert_array_equal(exported.toarray(), values[order])


def test_a_reorder_refuses_a_budget_that_cannot_read_back_its_widest_row(
    tmp_path,
) -> None:
    values = np.ones((2, 20_000), dtype=np.float64)
    order = np.asarray([1, 0])
    # Hand-computed. Two rows of 20,000 float64 values form one step of
    # 40,000 values, and converting one row holds 20,000 values of 36 bytes
    # and 2 row offsets of 20; reading a row back holds 20,000 values of 56
    # bytes, its piece's 48 and two rows' 32, three times, beside 3 row
    # offsets.
    spill = 16 * 2 + (9 * 2 + 2 * 25) + 8 * 2 + 64 * 40_000 + (20_000 * 36 + 2 * 20)
    read = 8 * 3 + 3 * (20_000 * 56 + 48 + 64)
    assert read > spill
    with pytest.raises(MemoryBudgetError) as refused:
        reordered_matrix(_matrix(values), order, max_bytes=read - 1)
    assert refused.value.needed_bytes == read
    # At that budget the store reads the row back.
    exported = materialize_h5ad_matrix(
        reordered_matrix(_matrix(values), order, max_bytes=read, temp_dir=tmp_path)
    )
    np.testing.assert_array_equal(exported.toarray(), values[order])


def test_the_bytes_held_while_rows_stream_grow_with_the_budget() -> None:
    shape, dtype = (1000, 50), np.dtype(np.float32)
    fixed = 16_000 + (9_000 + 50) + 16 + 64 * 50_000
    conversion = 50 * 28 + 2 * 20
    # Batches take up to half of what the budget leaves, at 64 bytes per
    # value, no more values than the matrix's 50,000; the pending rows take
    # half of what is then left.
    spare = 6_000_000 - fixed - conversion
    batch = spare // 128
    assert batch < 50_000
    assert reorder_held_bytes(shape, dtype, 6_000_000) == (
        fixed + 64 * batch + (spare - 64 * batch) // 2
    )
    # A larger budget gathers every value and keeps 256 MiB pending.
    assert reorder_held_bytes(shape, dtype, 4 << 30) == (
        fixed + 64 * 50_000 + (256 << 20)
    )
    # At the least budget, neither.
    assert reorder_held_bytes(shape, dtype, fixed + conversion) == fixed


def test_a_large_budget_gathers_at_most_two_million_values_per_append(
    monkeypatch, tmp_path
) -> None:
    values = np.random.default_rng(11).poisson(3.0, (4000, 30)).astype(np.uint16)
    order = np.random.default_rng(12).permutation(4000)
    monkeypatch.setattr(export, "_REORDER_BATCH_ENTRIES", 5_000)
    appended: list[int] = []
    original = SparseRowStore.append_rows

    def recorded(self, rows, matrix):
        appended.append(int(matrix.nnz))
        original(self, rows, matrix)

    monkeypatch.setattr(SparseRowStore, "append_rows", recorded)
    exported = materialize_h5ad_matrix(
        reordered_matrix(_matrix(values, rows=100), order, max_bytes=1 << 30)
    )
    np.testing.assert_array_equal(exported.toarray(), values[order])
    # Pieces of 100 rows, about 2,850 values each, are gathered until they
    # hold 5,000 values: two pieces per append, the last batch alone.
    assert sum(appended) == np.count_nonzero(values)
    assert all(5_000 <= count < 5_000 + 3_000 for count in appended[:-1])
    assert len(appended) == 20


def test_a_reorder_refuses_a_disk_without_room_and_leaves_no_files(
    tmp_path, monkeypatch
) -> None:
    values = np.random.default_rng(8).poisson(3.0, (300, 20)).astype(np.float32)
    order = np.random.default_rng(9).permutation(300)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        sparse_module.shutil,
        "disk_usage",
        lambda path: usage._replace(free=100),
    )
    matrix = reordered_matrix(
        _matrix(values), order, max_bytes=64 << 20, temp_dir=tmp_path
    )
    with pytest.raises(OSError, match="temporary bytes"):
        materialize_h5ad_matrix(matrix)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Reordering rows of text


def _text_rows(n_rows: int, seed: int) -> list[bytes]:
    """Rows of a text file: printable bytes of 0 to 80 bytes, each ending with a newline."""
    rng = np.random.default_rng(seed)
    return [
        bytes(rng.integers(32, 127, int(rng.integers(0, 81))).astype(np.uint8)) + b"\n"
        for _ in range(n_rows)
    ]


def _text_blocks(
    rows: list[bytes], block_rows: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The rows in blocks: each block's bytes and the offset of each row in them."""
    for first in range(0, len(rows), block_rows):
        block = rows[first : first + block_rows]
        offsets = np.zeros(len(block) + 1, dtype=np.int64)
        np.cumsum([len(row) for row in block], out=offsets[1:])
        yield np.frombuffer(b"".join(block), dtype=np.uint8).copy(), offsets


def _text_plan(
    n_rows: int, block_bytes: int = 1 << 20, max_bytes: int = 64 << 20
) -> export.TextReorder:
    """A plan for rows of at most 81 bytes, in blocks of 50 rows and about ``block_bytes``."""
    return export.plan_text_reorder(n_rows, 81, 50, block_bytes, max_bytes)


def _batch_rows(batches) -> list[bytes]:
    return [
        bytes(values[offsets[i] : offsets[i + 1]])
        for values, offsets in batches
        for i in range(offsets.size - 1)
    ]


@pytest.mark.parametrize("block_bytes", [1 << 20, 2048])
def test_text_rows_reach_their_places_and_go_as_they_are_read(
    tmp_path, block_bytes
) -> None:
    rows = _text_rows(700, 1)
    destinations = np.random.default_rng(2).permutation(700)
    # Blocks of a few kilobytes read back from buckets of a few rows, so the
    # rows spread over many of them.
    batches = export.reordered_text_rows(
        _text_blocks(rows, 50),
        destinations,
        _text_plan(700, block_bytes),
        operation="text reorder",
        temp_dir=tmp_path,
    )
    # The rows go to the temporary files only once the batches are read.
    assert list(tmp_path.iterdir()) == []
    found = []
    remaining = []
    for batch in batches:
        found += _batch_rows([batch])
        (directory,) = tmp_path.iterdir()
        remaining.append(len(list(directory.glob("rows-*[0-9]"))))
    expected = [b""] * 700
    for row, place in zip(rows, destinations.tolist(), strict=True):
        expected[place] = row
    assert found == expected
    # Each bucket's files are deleted once its rows are taken, and the
    # store's folder once the last batch is.
    assert remaining == sorted(remaining, reverse=True)
    assert (remaining[0] > 10) == (block_bytes < 1 << 20)
    assert list(tmp_path.iterdir()) == []


def test_text_rows_wait_in_their_files_as_they_are(tmp_path, monkeypatch) -> None:
    # Rows that repeat most of their text, which zstd would shrink to a fifth.
    rows = [
        f"cell{k:04d}_contig_1\tTRAV1-2\tCASSLGTDTQYF\n".encode() for k in range(600)
    ]
    finalize = SparseRowStore.finalize
    spilled = []

    def recorded(store):
        finalize(store)
        spilled.append(
            sum(path.stat().st_size for path in store.directory.glob("rows-*/*.bin"))
        )

    monkeypatch.setattr(SparseRowStore, "finalize", recorded)
    places = np.random.default_rng(7).permutation(600)
    batches = export.reordered_text_rows(
        _text_blocks(rows, 50),
        places,
        _text_plan(600),
        operation="text reorder",
        temp_dir=tmp_path,
    )
    expected = [b""] * 600
    for row, place in zip(rows, places.tolist(), strict=True):
        expected[place] = row
    assert _batch_rows(batches) == expected
    # The files hold every row's bytes, and little beside them: each row's
    # place and length, and each block's headers.
    total = sum(len(row) for row in rows)
    assert total <= spilled[0] <= total + 8 * 600 + 4096


def test_a_text_reorder_of_no_rows_yields_none(tmp_path) -> None:
    batches = export.reordered_text_rows(
        iter([]),
        np.zeros(0, dtype=np.int64),
        _text_plan(0),
        operation="text reorder",
        temp_dir=tmp_path,
    )
    assert _batch_rows(batches) == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("n_rows", "places", "message"),
    [
        (10, np.arange(9), "has places for 9 rows and was sent more"),
        (9, np.arange(10), "has places for 10 rows and was sent 9"),
        (10, np.r_[np.arange(9), 3], "Destination row 3 was appended more than once"),
    ],
)
def test_a_text_reorder_refuses_rows_that_miss_their_places(
    tmp_path, n_rows, places, message
) -> None:
    batches = export.reordered_text_rows(
        _text_blocks(_text_rows(n_rows, 3), 4),
        places,
        _text_plan(places.size),
        operation="text reorder",
        temp_dir=tmp_path,
    )
    with pytest.raises(ValueError, match=message):
        list(batches)
    assert list(tmp_path.iterdir()) == []


def test_closing_a_text_reorder_early_removes_its_files(tmp_path) -> None:
    batches = export.reordered_text_rows(
        _text_blocks(_text_rows(300, 4), 50),
        np.random.default_rng(5).permutation(300),
        _text_plan(300, 2048),
        operation="text reorder",
        temp_dir=tmp_path,
    )
    next(batches)
    assert len(list(tmp_path.iterdir())) == 1
    batches.close()
    assert list(tmp_path.iterdir()) == []


def test_a_text_reorder_asks_for_the_disk_that_its_files_take(tmp_path) -> None:
    # Rows of one length, so the first block's rows give every row's bytes.
    rows = [
        f"cell{k:04d}_contig_1\tTRAV1-2\tCASSLGTDTQYF\n".encode() for k in range(600)
    ]
    places = np.random.default_rng(10).permutation(600)

    def reorder() -> list[bytes]:
        return _batch_rows(
            export.reordered_text_rows(
                _text_blocks(rows, 50),
                places,
                _text_plan(600),
                operation="text reorder",
                temp_dir=tmp_path,
            )
        )

    with finalized_bytes() as peaks:
        found = reorder()
    (peak,) = peaks
    with free_disk(0), pytest.raises(OSError, match="Text reorder needs") as full:
        reorder()
    asked = asked_bytes(full.value)
    # The check asks for at least every byte that the files take, the
    # store's records among them, and at most 16 bytes more for each row.
    assert peak <= asked <= peak + 16 * 600
    # One byte short, the reorder is refused before a row is sent.
    with free_disk(asked - 1), pytest.raises(OSError, match=f"needs {asked} "):
        reorder()
    assert list(tmp_path.iterdir()) == []
    with free_disk(asked):
        assert reorder() == found
    expected = [b""] * 600
    for row, place in zip(rows, places.tolist(), strict=True):
        expected[place] = row
    assert found == expected
    assert list(tmp_path.iterdir()) == []


def test_a_text_reorder_plan_fits_its_bytes_or_names_bytes_that_fit_it() -> None:
    widest = 81

    def plan(max_bytes: int, block_bytes: int = 1 << 20) -> export.TextReorder:
        return export.plan_text_reorder(1000, widest, 50, block_bytes, max_bytes)

    # The plan of the smallest buckets reads one row of the widest at a time
    # and keeps no rows waiting.
    smallest = plan(0)
    assert smallest.read_bytes == SparseRowStore.ragged_row_read_bytes(widest)
    assert smallest.bucket_bytes == 2 * smallest.read_bytes
    assert smallest.buffer_bytes == 0
    least = smallest.held_bytes
    for max_bytes in (least, least + 1, 10 * least, 64 << 20, 1 << 30):
        fitted = plan(max_bytes)
        assert fitted.held_bytes <= max_bytes
        assert smallest.read_bytes <= fitted.read_bytes <= 1 << 20
        assert fitted.bucket_bytes == 2 * fitted.read_bytes
        assert fitted.buffer_bytes <= 1 << 20
    # Below it, a plan fits, or holds more, and a plan within what it holds
    # fits, so the budget that a refusal names passes.
    for max_bytes in (0, 1_000, 20_000, least // 2, least - 1):
        held = plan(max_bytes).held_bytes
        assert held <= max_bytes or plan(held).held_bytes <= held
    # With room, buckets read in about a block, and a block of rows waits.
    roomy = plan(1 << 30)
    assert roomy.read_bytes == 1 << 20 and roomy.buffer_bytes == 1 << 20
    # A block of a few bytes still reads a row of the widest.
    assert plan(1 << 30, block_bytes=10).read_bytes == smallest.read_bytes
    # A block holds no more rows than the reorder.
    assert export.plan_text_reorder(10, widest, 10**6, 1 << 20, 1 << 30) == (
        export.plan_text_reorder(10, widest, 10, 1 << 20, 1 << 30)
    )


# ---------------------------------------------------------------------------
# H5AD and AnnData exports of a shuffled import


def test_the_shuffled_store_holds_the_order_that_exports_undo(imported) -> None:
    store = _open(imported.shuffled)
    np.testing.assert_array_equal(store.cells.fetch_all(SOURCE_ROW), _ORDER)
    np.testing.assert_array_equal(_file_rows(store.cells.fetch_all("ids")), _ORDER)
    assert SOURCE_ROW not in _open(imported.plain).cells.columns


def test_a_shuffled_import_exports_the_file_it_imported(imported, tmp_path) -> None:
    shuffled, plain = _open(imported.shuffled), _open(imported.plain)
    obsm = {
        name: imported.shuffledImport.embeddingArtifacts[name]
        for name in ("X_umap", "X_pca")
    }
    to_h5ad(shuffled.RNA, str(tmp_path / "shuffled.h5ad"), obsm=obsm)
    to_h5ad(
        plain.RNA,
        str(tmp_path / "plain.h5ad"),
        obsm={
            name: imported.plainImport.embeddingArtifacts[name]
            for name in ("X_umap", "X_pca")
        },
    )

    # The plain store's export, which the other export tests pin, file for
    # file, apart from the identifiers of the exported artifacts.
    skip = ("uns/cytearc/artifacts",)
    assert _h5_tree(tmp_path / "shuffled.h5ad", skip) == _h5_tree(
        tmp_path / "plain.h5ad", skip
    )
    # And the file's own values.
    exported = anndata.read_h5ad(tmp_path / "shuffled.h5ad")
    source = anndata.read_h5ad(imported.source)
    assert exported.obs_names.tolist() == source.obs_names.tolist()
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS)
    for column in ("score", "count", "flag", "label", "stage", "nullable", "keep"):
        exported_values, source_values = exported.obs[column], source.obs[column]
        np.testing.assert_array_equal(
            exported_values.isna().to_numpy(), source_values.isna().to_numpy()
        )
        present = ~source_values.isna().to_numpy()
        if pd.api.types.is_numeric_dtype(source_values.dtype):
            # A nullable integer column exports as float64 with NaN.
            np.testing.assert_array_equal(
                exported_values[present].to_numpy(dtype=np.float64),
                source_values[present].to_numpy(dtype=np.float64),
            )
        else:
            assert (
                exported_values[present].astype(str).tolist()
                == source_values[present].astype(str).tolist()
            ), column
    for name in ("X_umap", "X_pca"):
        np.testing.assert_array_equal(exported.obsm[name], source.obsm[name])
    assert SOURCE_ROW not in exported.obs


def test_stored_order_exports_the_store_as_it_holds_the_cells(
    imported, tmp_path
) -> None:
    store = _open(imported.shuffled)
    to_h5ad(store.RNA, str(tmp_path / "stored.h5ad"), cell_order="stored")

    exported = anndata.read_h5ad(tmp_path / "stored.h5ad")
    assert exported.obs_names.tolist() == [f"cell-{row}" for row in _ORDER]
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS[_ORDER])
    np.testing.assert_array_equal(
        exported.obs["score"].to_numpy(), _obs()["score"].to_numpy()[_ORDER]
    )
    assert SOURCE_ROW not in exported.obs


def test_a_store_without_source_rows_exports_alike_in_either_order(
    imported, tmp_path
) -> None:
    store = _open(imported.plain)
    to_h5ad(store.RNA, str(tmp_path / "source.h5ad"), temp_dir=tmp_path / "none")
    to_h5ad(store.RNA, str(tmp_path / "stored.h5ad"), cell_order="stored")
    assert _h5_tree(tmp_path / "source.h5ad") == _h5_tree(tmp_path / "stored.h5ad")
    # No temporary files were needed, so the directory was never made.
    assert not (tmp_path / "none").exists()
    source = store.to_anndata()
    stored = store.to_anndata(cell_order="stored")
    assert source.obs_names.tolist() == stored.obs_names.tolist()
    np.testing.assert_array_equal(source.X.toarray(), stored.X.toarray())


def test_selected_cells_export_in_the_files_order(imported, tmp_path) -> None:
    store = _open(_copy(imported.shuffled, tmp_path))
    rows = _file_rows(store.cells.fetch_all("ids"))
    chosen = (rows % 3 == 1) | (rows > 140)
    store.cells.insert("chosen", by_cell_id(store.cells, chosen))
    selection = store.snapshot_cell_selection("chosen")
    expected = np.sort(rows[chosen])

    to_h5ad(store.RNA, str(tmp_path / "chosen.h5ad"), cell_selection=selection)
    exported = anndata.read_h5ad(tmp_path / "chosen.h5ad")
    assert _file_rows(exported.obs_names).tolist() == expected.tolist()
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS[expected])
    in_memory = store.to_anndata(cell_selection=selection)
    assert _file_rows(in_memory.obs_names).tolist() == expected.tolist()
    np.testing.assert_array_equal(in_memory.X.toarray(), _COUNTS[expected])
    by_key = store.to_anndata(cell_key="chosen")
    assert _file_rows(by_key.obs_names).tolist() == expected.tolist()


def test_to_anndata_holds_the_files_order_and_no_source_rows(imported) -> None:
    store = _open(imported.shuffled)
    # to_anndata holds the cells that I selects, cells with counts.
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    rows = np.sort(_ORDER[active])
    assert rows.size == _N_CELLS - 2
    source = store.to_anndata()
    assert _file_rows(source.obs_names).tolist() == rows.tolist()
    np.testing.assert_array_equal(source.X.toarray(), _COUNTS[rows])
    np.testing.assert_array_equal(
        source.obs["score"].to_numpy(), _obs()["score"].to_numpy()[rows]
    )
    assert SOURCE_ROW not in source.obs
    stored = store.to_anndata(cell_order="stored")
    assert SOURCE_ROW not in stored.obs
    np.testing.assert_array_equal(stored.X.toarray(), _COUNTS[_ORDER[active]])

    # Normalized values, chosen features and layers follow the same order.
    features = [5, 0, 3]
    normed = store.to_anndata(
        matrix="normed", feature_indexes=features, layers={"counts": "RNA"}
    )
    stored_normed = store.to_anndata(
        matrix="normed",
        feature_indexes=features,
        layers={"counts": "RNA"},
        cell_order="stored",
    )
    back = np.argsort(_ORDER[active])
    np.testing.assert_array_equal(normed.X.toarray(), stored_normed.X.toarray()[back])
    np.testing.assert_array_equal(
        normed.layers["counts"].toarray(), _COUNTS[rows][:, features]
    )
    assert normed.var_names.tolist() == [f"feature-{f}" for f in features]


def test_a_plain_source_row_column_exports_as_an_ordinary_column(
    imported, tmp_path
) -> None:
    path = _copy(imported.plain, tmp_path)
    # A column of that name that no import wrote, as a store may hold from
    # before imports could shuffle; inserts refuse the name.
    zarr.open_group(str(path), mode="r+")["cellData"].create_array(
        SOURCE_ROW, data=np.arange(_N_CELLS, dtype=np.int64)[::-1].copy()
    )
    store = _open(path)
    exported = store.to_anndata(cell_key="keep")
    # It neither orders the export nor stays out of it.
    keep = _obs()["keep"].to_numpy()
    assert _file_rows(exported.obs_names).tolist() == np.flatnonzero(keep).tolist()
    np.testing.assert_array_equal(
        exported.obs[SOURCE_ROW].to_numpy(), np.arange(_N_CELLS)[::-1][keep]
    )
    to_h5ad(store.RNA, str(tmp_path / "plain.h5ad"))
    with h5py.File(tmp_path / "plain.h5ad", "r") as h5:
        assert SOURCE_ROW in h5["obs"]


@pytest.mark.parametrize("call", ["to_h5ad", "to_mtx", "to_anndata"])
def test_exports_refuse_an_unknown_cell_order_before_writing(
    imported, tmp_path, call
) -> None:
    store = _open(imported.shuffled)
    target = tmp_path / "out"
    export_call = {
        "to_h5ad": lambda order: to_h5ad(store.RNA, str(target), cell_order=order),
        "to_mtx": lambda order: to_mtx(store.RNA, str(target), cell_order=order),
        "to_anndata": lambda order: store.to_anndata(cell_order=order),
    }[call]
    with pytest.raises(ValueError, match="cell_order must be 'source'"):
        export_call("file")
    with pytest.raises(TypeError, match="cell_order must be 'source' or 'stored'"):
        export_call(None)
    assert not target.exists()
    if call != "to_anndata":
        with pytest.raises(TypeError, match="temp_dir must be a path"):
            {"to_h5ad": to_h5ad, "to_mtx": to_mtx}[call](
                store.RNA, str(target), temp_dir=3
            )
        assert not target.exists()


def test_an_export_refuses_a_budget_below_its_reorder_before_writing(
    imported, tmp_path
) -> None:
    store = _open(imported.shuffled, mem_budget=40_000)
    with pytest.raises(MemoryBudgetError, match="cell_order='stored'"):
        to_h5ad(store.RNA, str(tmp_path / "small.h5ad"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("call", ["to_h5ad", "to_mtx"])
def test_an_export_asks_for_the_disk_that_its_temporary_files_take(
    imported, tmp_path, call
) -> None:
    store = _open(imported.shuffled)
    spill = tmp_path / "spill"
    spill.mkdir()
    write = {"to_h5ad": to_h5ad, "to_mtx": to_mtx}[call]

    def export_to(name: str) -> Path:
        target = tmp_path / (f"{name}.h5ad" if call == "to_h5ad" else name)
        write(store.RNA, str(target), temp_dir=spill)
        return target

    with finalized_bytes() as peaks:
        export_to("roomy")
    (peak,) = peaks
    with free_disk(0), pytest.raises(OSError, match="which has 0 free") as full:
        export_to("full")
    asked = asked_bytes(full.value)
    # The cells hold few counts, so the store's record of each cell takes
    # more than its counts, and the check still asks for every byte of the
    # files.
    assert peak <= asked
    # One byte short, the export is refused and removes its temporary files,
    # and to_h5ad leaves no file.
    with free_disk(asked - 1), pytest.raises(OSError, match=f"needs {asked} "):
        export_to("short")
    assert list(spill.iterdir()) == []
    assert not (tmp_path / "short.h5ad").exists()
    with free_disk(asked):
        assert export_to("enough").exists()
    assert list(spill.iterdir()) == []


def test_an_export_leaves_no_temporary_files(imported, tmp_path) -> None:
    store = _open(imported.shuffled)
    spill = tmp_path / "spill"
    spill.mkdir()
    created: list[Path] = []
    original = SparseRowStore.__init__

    def recorded(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self.directory)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SparseRowStore, "__init__", recorded)
        to_h5ad(store.RNA, str(tmp_path / "out.h5ad"), temp_dir=spill)
        to_mtx(store.RNA, str(tmp_path / "mtx"), temp_dir=spill)
    assert [path.parent for path in created] == [spill, spill]
    assert list(spill.iterdir()) == []


# ---------------------------------------------------------------------------
# Matrix Market


def _mtx_files(folder: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(folder.iterdir()):
        data = path.read_bytes()
        files[path.name] = gzip.decompress(data) if path.suffix == ".gz" else data
    return files


@pytest.mark.parametrize("compress", [False, True])
def test_matrix_market_exports_the_files_cells_in_its_order(
    imported, tmp_path, compress
) -> None:
    to_mtx(_open(imported.shuffled).RNA, str(tmp_path / "shuffled"), compress)
    to_mtx(_open(imported.plain).RNA, str(tmp_path / "plain"), compress)

    # A Matrix Market file written from the file in its order: the plain
    # store's export, which tests/test_writers.py pins.
    assert _mtx_files(tmp_path / "shuffled") == _mtx_files(tmp_path / "plain")
    name = "matrix.mtx.gz" if compress else "matrix.mtx"
    matrix = mmread(tmp_path / "shuffled" / name)
    np.testing.assert_array_equal(np.asarray(matrix.todense()).T, _COUNTS)
    lines = _mtx_files(tmp_path / "shuffled")[name]
    entries = np.asarray(
        [line.split()[:2] for line in lines.decode().splitlines()[3:]], dtype=int
    )
    # Entries are sorted by cell, then feature.
    assert np.array_equal(
        np.lexsort((entries[:, 0], entries[:, 1])), np.arange(len(entries))
    )
    barcodes = "barcodes.tsv.gz" if compress else "barcodes.tsv"
    assert _mtx_files(tmp_path / "shuffled")[barcodes].decode().split() == [
        f"cell-{row}" for row in range(_N_CELLS)
    ]


def test_matrix_market_in_stored_order_follows_the_store(imported, tmp_path) -> None:
    to_mtx(_open(imported.shuffled).RNA, str(tmp_path / "stored"), cell_order="stored")
    matrix = mmread(tmp_path / "stored" / "matrix.mtx")
    np.testing.assert_array_equal(np.asarray(matrix.todense()).T, _COUNTS[_ORDER])
    assert (tmp_path / "stored" / "barcodes.tsv").read_text().split() == [
        f"cell-{row}" for row in _ORDER
    ]


def test_matrix_market_refuses_a_budget_before_writing(imported, tmp_path) -> None:
    store = _open(imported.shuffled, mem_budget=40_000)
    with pytest.raises(MemoryBudgetError, match="the counts in source order"):
        to_mtx(store.RNA, str(tmp_path / "mtx"))
    assert not (tmp_path / "mtx").exists()


# ---------------------------------------------------------------------------
# Merges and composites


def test_a_merge_writes_each_cells_row_in_the_concatenation_of_its_sources(
    imported, tmp_path
) -> None:
    shuffled, plain = _open(imported.shuffled), _open(imported.plain)
    DataStoreMerge(
        [shuffled, plain], str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
    ).dump()
    merged = _open(tmp_path / "merged.zarr")
    cells = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")["cellData"]

    ids = merged.cells.fetch_all("ids").astype(str)
    expected = _file_rows(ids) + np.where(np.char.startswith(ids, "b__"), _N_CELLS, 0)
    np.testing.assert_array_equal(merged.cells.fetch_all(SOURCE_ROW), expected)
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    assert cells[SOURCE_ROW].dtype == np.int64
    # The merge's block shuffle is no uniform permutation, so no record.
    assert CELL_ORDER_ATTR not in cells.attrs
    assert "orig_sourceRow" not in cells

    # Its export holds the sources' exports, one after the other.
    to_h5ad(merged.RNA, str(tmp_path / "merged.h5ad"))
    concatenated = anndata.read_h5ad(tmp_path / "merged.h5ad")
    assert concatenated.obs_names.tolist() == [
        f"{name}__cell-{row}" for name in ("a", "b") for row in range(_N_CELLS)
    ]
    np.testing.assert_array_equal(
        concatenated.X.toarray(), np.vstack([_COUNTS, _COUNTS])
    )
    assert SOURCE_ROW not in concatenated.obs


def test_a_merge_ranks_a_subset_and_replaces_a_plain_source_row_column(
    imported, tmp_path
) -> None:
    shuffled = _open(imported.shuffled)
    keep = _file_rows(shuffled.cells.fetch_all("ids")) % 4 != 1
    SubsetZarr(
        str(tmp_path / "subset.zarr"),
        [shuffled.RNA],
        cell_ids=shuffled.cells.fetch_all("ids")[keep],
        nthreads=1,
    ).dump()
    subset = _open(tmp_path / "subset.zarr")
    plain_path = _copy(imported.plain, tmp_path)
    zarr.open_group(str(plain_path), mode="r+")["cellData"].create_array(
        SOURCE_ROW, data=np.arange(_N_CELLS, dtype=np.int64)[::-1].copy()
    )
    plain = _open(plain_path)

    with _warnings() as messages:
        DataStoreMerge(
            [plain, subset], str(tmp_path / "merged.zarr"), ["b", "a"], nthreads=1
        ).dump()
    assert any(
        "Source 'b' holds a cell column 'sourceRow' that no import wrote" in message
        for message in messages
    )
    merged = _open(tmp_path / "merged.zarr")
    ids = merged.cells.fetch_all("ids").astype(str)
    rows = _file_rows(ids)
    kept = np.sort(_ORDER[keep])
    # The subset's cells rank by their file rows, which have gaps; the
    # plain source's column is replaced by its store order.
    expected = np.where(
        np.char.startswith(ids, "a__"),
        _N_CELLS + np.searchsorted(kept, rows),
        rows,
    )
    np.testing.assert_array_equal(merged.cells.fetch_all(SOURCE_ROW), expected)
    exported = merged.to_anndata(cell_key="I")
    order = np.argsort(merged.cells.fetch_all(SOURCE_ROW)[merged.cells.fetch_all("I")])
    assert (
        exported.obs_names.tolist() == ids[merged.cells.fetch_all("I")][order].tolist()
    )


def test_a_composite_writes_the_rows_of_its_selected_cells(imported, tmp_path) -> None:
    shuffled, plain = _open(imported.shuffled), _open(imported.plain)
    selected = np.arange(0, _N_CELLS, 3)
    with pytest.raises(ValueError, match="not ids, I, names, or sourceRow"):
        create_composite(
            {"a": shuffled, "b": plain},
            at=tmp_path / "refused.zarr",
            features="intersection",
            source_column=SOURCE_ROW,
            nthreads=1,
        )
    joint = create_composite(
        {"a": shuffled, "b": plain},
        at=tmp_path / "joint.zarr",
        features="intersection",
        rows={"a": selected},
        nthreads=1,
    )
    cells = zarr.open_group(str(tmp_path / "joint.zarr"), mode="r")["cellData"]
    ids = joint.cells.fetch_all("ids").astype(str)
    rows = _file_rows(ids)
    # The selected store rows of the shuffled source hold these file rows.
    chosen = np.sort(_ORDER[selected])
    expected = np.where(
        np.char.startswith(ids, "a__"),
        np.searchsorted(chosen, rows),
        chosen.size + rows,
    )
    np.testing.assert_array_equal(joint.cells.fetch_all(SOURCE_ROW), expected)
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    assert CELL_ORDER_ATTR not in cells.attrs
    # The composite prefixes source columns with orig_, but not sourceRow.
    assert "orig_sourceRow" not in cells
    exported = joint.to_anndata(cell_key="I")
    assert exported.obs_names.tolist() == [
        *(f"a__cell-{row}" for row in chosen),
        *(f"b__cell-{row}" for row in range(_N_CELLS)),
    ]
    np.testing.assert_array_equal(
        exported.X.toarray(), np.vstack([_COUNTS[chosen], _COUNTS])
    )


# ---------------------------------------------------------------------------
# Subsets, repacks and mounts


def test_a_subset_keeps_store_order_its_source_rows_and_the_record(
    imported, tmp_path
) -> None:
    shuffled = _open(imported.shuffled)
    ids = shuffled.cells.fetch_all("ids")
    keep = _file_rows(ids) % 5 != 2
    SubsetZarr(
        str(tmp_path / "subset.zarr"), [shuffled.RNA], cell_ids=ids[keep], nthreads=1
    ).dump()
    subset = _open(tmp_path / "subset.zarr")
    cells = zarr.open_group(str(tmp_path / "subset.zarr"), mode="r")["cellData"]

    np.testing.assert_array_equal(subset.cells.fetch_all("ids"), ids[keep])
    np.testing.assert_array_equal(subset.cells.fetch_all(SOURCE_ROW), _ORDER[keep])
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    assert recorded_shuffle_seed(cells) == _SEED
    to_h5ad(subset.RNA, str(tmp_path / "subset.h5ad"))
    exported = anndata.read_h5ad(tmp_path / "subset.h5ad")
    rows = np.sort(_ORDER[keep])
    assert _file_rows(exported.obs_names).tolist() == rows.tolist()
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS[rows])


@pytest.mark.parametrize("copy", ["repack", "repack_data_only", "mount"])
def test_repacks_and_mounts_keep_the_source_rows_and_the_record(
    imported, tmp_path, copy
) -> None:
    source = _copy(imported.shuffled, tmp_path / "source")
    target = tmp_path / "target.zarr"
    if copy == "mount":
        mount_datastore(
            str(source), at=str(target), min_features_per_cell=0, nthreads=1
        )
    else:
        repack_store(str(source), str(target), data_only=copy == "repack_data_only")
    cells = zarr.open_group(str(target), mode="r")["cellData"]
    np.testing.assert_array_equal(cells[SOURCE_ROW][:], _ORDER)
    assert cells[SOURCE_ROW].attrs["role"] == SOURCE_ROW_ROLE
    assert recorded_shuffle_seed(cells) == _SEED
    store = _open(target)
    to_h5ad(store.RNA, str(tmp_path / "copy.h5ad"))
    exported = anndata.read_h5ad(tmp_path / "copy.h5ad")
    assert exported.obs_names.tolist() == [f"cell-{row}" for row in range(_N_CELLS)]
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS)


def test_copies_of_a_store_without_the_record_write_none(imported, tmp_path) -> None:
    plain = _open(imported.plain)
    SubsetZarr(
        str(tmp_path / "subset.zarr"),
        [plain.RNA],
        cell_ids=plain.cells.fetch_all("ids")[:20],
        nthreads=1,
    ).dump()
    mount_datastore(
        str(imported.plain), at=str(tmp_path / "mount.zarr"), min_features_per_cell=0
    )
    repack_store(str(imported.plain), str(tmp_path / "repack.zarr"), data_only=True)
    for name in ("subset.zarr", "mount.zarr", "repack.zarr"):
        cells = zarr.open_group(str(tmp_path / name), mode="r")["cellData"]
        assert CELL_ORDER_ATTR not in cells.attrs
        assert SOURCE_ROW not in cells


# ---------------------------------------------------------------------------
# Graphs, artifacts and pipeline runs


@pytest.fixture(scope="module")
def shuffled_run(imported, tmp_path_factory) -> SimpleNamespace:
    """A completed RNA run over the cells that ``keep`` selects, on the shuffled store."""
    path = _copy(imported.shuffled, tmp_path_factory.mktemp("shuffled_run"))
    store = _open(path)
    run = store.pipeline.rna(
        label="export",
        cells="keep",
        filtering=False,
        cell_cycle=False,
        hvg={"top_n": 8},
        pca={"dims": 3},
        neighbors={"k": 5},
        doublets=False,
        markers=False,
        snapshot_columns=["score", "label"],
    )
    assert run.status == "completed"
    rows = np.flatnonzero(store.cells.fetch_all("keep"))
    # The run's cells in store order, and where each lies in the file.
    return SimpleNamespace(
        store=store, run=run, rows=rows, perm=np.argsort(_ORDER[rows], kind="stable")
    )


def _assert_permuted(source: Any, stored: Any, perm: np.ndarray) -> None:
    """Assert that ``source`` holds the cells of ``stored`` at ``perm``, everything else alike."""
    assert source.obs_names.tolist() == stored.obs_names[perm].tolist()
    assert source.var_names.tolist() == stored.var_names.tolist()
    np.testing.assert_array_equal(source.X.toarray(), stored.X.toarray()[perm])
    assert list(source.obs.columns) == list(stored.obs.columns)
    for column in source.obs.columns:
        pd.testing.assert_series_equal(
            source.obs[column].reset_index(drop=True),
            stored.obs[column].iloc[perm].reset_index(drop=True),
        )
    assert sorted(source.obsm) == sorted(stored.obsm)
    for name in source.obsm:
        np.testing.assert_array_equal(source.obsm[name], stored.obsm[name][perm])
    for name in stored.layers:
        np.testing.assert_array_equal(
            source.layers[name].toarray(), stored.layers[name].toarray()[perm]
        )
    assert sorted(source.obsp) == sorted(stored.obsp)
    for name in source.obsp:
        np.testing.assert_array_equal(
            source.obsp[name].toarray(),
            stored.obsp[name].toarray()[np.ix_(perm, perm)],
        )


def _fuzzy_union(store: DataStore, graph: Any) -> np.ndarray:
    directed = store.graph.load(graph).toarray()
    return directed + directed.T - directed * directed.T


def test_run_exports_hold_the_runs_cells_in_the_files_order(
    shuffled_run, tmp_path
) -> None:
    store, run, perm = shuffled_run.store, shuffled_run.run, shuffled_run.perm
    options = {"obs": {"leiden": run["clusters"]}, "graph": run["connectivity_map"]}
    source = store.to_anndata(run=run, **options)
    stored = store.to_anndata(run=run, cell_order="stored", **options)

    _assert_permuted(source, stored, perm)
    keep = _obs()["keep"].to_numpy()
    assert _file_rows(source.obs_names).tolist() == np.flatnonzero(keep).tolist()
    assert {"X_pca", "X_umap"} <= set(source.obsm)
    features = [int(name.split("-")[1]) for name in source.var_names]
    np.testing.assert_array_equal(
        source.X.toarray(), _COUNTS[np.flatnonzero(keep)][:, features]
    )
    np.testing.assert_allclose(
        source.obsp["connectivities"].toarray(),
        _fuzzy_union(store, run["connectivity_map"])[np.ix_(perm, perm)],
        rtol=1e-6,
    )

    # The file streams the counts through temporary files and holds what
    # the object holds.
    spill = tmp_path / "spill"
    spill.mkdir()
    to_h5ad(store.RNA, str(tmp_path / "run.h5ad"), run=run, temp_dir=spill, **options)
    streamed = anndata.read_h5ad(tmp_path / "run.h5ad")
    assert streamed.obs_names.tolist() == source.obs_names.tolist()
    np.testing.assert_array_equal(streamed.X.toarray(), source.X.toarray())
    for name in source.obsm:
        np.testing.assert_array_equal(streamed.obsm[name], source.obsm[name])
    np.testing.assert_array_equal(
        streamed.obsp["connectivities"].toarray(),
        source.obsp["connectivities"].toarray(),
    )
    assert streamed.obs["leiden"].tolist() == source.obs["leiden"].tolist()
    assert list(spill.iterdir()) == []


def test_normed_run_exports_hold_the_normalized_rows_in_the_files_order(
    shuffled_run, tmp_path
) -> None:
    store, run, perm = shuffled_run.store, shuffled_run.run, shuffled_run.perm
    source = store.to_anndata(run=run, matrix="normed", layers={"counts": "RNA"})
    stored = store.to_anndata(
        run=run, matrix="normed", layers={"counts": "RNA"}, cell_order="stored"
    )
    _assert_permuted(source, stored, perm)
    to_h5ad(store.RNA, str(tmp_path / "normed.h5ad"), run=run, matrix="normed")
    streamed = anndata.read_h5ad(tmp_path / "normed.h5ad")
    np.testing.assert_array_equal(streamed.X.toarray(), source.X.toarray())
    assert streamed.obs_names.tolist() == source.obs_names.tolist()


def test_a_run_export_refuses_a_budget_below_its_reorder(shuffled_run, tmp_path):
    store, run = shuffled_run.store, shuffled_run.run
    small = _open(store.zarr_loc, mem_budget=40_000)
    with pytest.raises(MemoryBudgetError, match="cell_order='stored'"):
        to_h5ad(
            small.RNA, str(tmp_path / "run.h5ad"), run=small.pipeline.list_runs()[0]
        )
    assert not (tmp_path / "run.h5ad").exists()
    del run


def test_live_exports_permute_artifacts_and_graphs(shuffled_run, tmp_path) -> None:
    store, run, perm = shuffled_run.store, shuffled_run.run, shuffled_run.perm
    options = {
        "cell_selection": run["analysis_cell_selection"],
        "obsm": {"X_pca": run["pca"]},
        "obs": {"leiden": run["clusters"]},
        "graph": run["connectivity_map"],
    }
    source = store.to_anndata(**options)
    stored = store.to_anndata(cell_order="stored", **options)
    _assert_permuted(source, stored, perm)
    np.testing.assert_allclose(
        source.obsp["connectivities"].toarray(),
        _fuzzy_union(store, run["connectivity_map"])[np.ix_(perm, perm)],
        rtol=1e-6,
    )
    to_h5ad(store.RNA, str(tmp_path / "live.h5ad"), **options)
    streamed = anndata.read_h5ad(tmp_path / "live.h5ad")
    assert streamed.obs_names.tolist() == source.obs_names.tolist()
    np.testing.assert_array_equal(streamed.X.toarray(), source.X.toarray())
    np.testing.assert_array_equal(streamed.obsm["X_pca"], source.obsm["X_pca"])
    np.testing.assert_array_equal(
        streamed.obsp["distances"].toarray(), source.obsp["distances"].toarray()
    )


# ---------------------------------------------------------------------------
# Clone tables and character matrices


@pytest.fixture
def clone_store(tmp_path):
    """The clone truth's store, its rows claiming file rows of a shuffle with seed 3."""
    from tests.clone_helpers import write_truth_store
    from tests.clone_truth import clone_truth

    truth = clone_truth()
    path = tmp_path / "clones.zarr"
    write_truth_store(path, truth)
    order = shuffled_order(len(truth.cell_ids), 3)
    write_shuffled_cell_order(
        zarr.open_group(str(path), mode="r+")["cellData"], order, 3
    )
    return SimpleNamespace(
        store=DataStore(str(path), nthreads=1), truth=truth, order=order
    )


def test_clone_tables_export_in_the_files_order(clone_store, tmp_path) -> None:
    from tests.clone_helpers import every_cell
    from tests.clone_truth import write_cospar

    store, truth, order = clone_store.store, clone_store.truth, clone_store.order
    ref = store.clones.import_clones(
        write_cospar(truth, tmp_path / "clones.csv"), cell_selection=every_cell(store)
    )
    ids = np.asarray(store.cells.fetch_all("ids")).astype(str)
    with_clone = truth.clones >= 0
    rank = np.argsort(order[with_clone], kind="stable")
    expected = ids[with_clone][rank]

    store.clones.export_clones(ref, tmp_path / "source.csv")
    store.clones.export_clones(ref, tmp_path / "stored.csv", cell_order="stored")
    source = pd.read_csv(tmp_path / "source.csv")
    stored = pd.read_csv(tmp_path / "stored.csv")
    assert source["Cell_ID"].tolist() == expected.tolist()
    assert stored["Cell_ID"].tolist() == ids[with_clone].tolist()
    # Each cell keeps its clone.
    assert sorted(zip(source["Cell_ID"], source["Clone_ID"])) == sorted(
        zip(stored["Cell_ID"], stored["Clone_ID"])
    )
    store.clones.export_clones(ref, tmp_path / "matrix.mtx", format="matrix")
    assert pd.read_csv(tmp_path / "cells.tsv", sep="\t").iloc[:, 0].tolist() == (
        expected.tolist()
    )
    with pytest.raises(ValueError, match="cell_order must be 'source'"):
        store.clones.export_clones(ref, tmp_path / "x.csv", cell_order="file")
    assert not (tmp_path / "x.csv").exists()


def test_character_matrices_export_in_the_files_order(clone_store, tmp_path) -> None:
    from tests.clone_helpers import every_cell
    from tests.clone_truth import recorder_truth, write_petracer_characters

    store, truth, order = clone_store.store, clone_store.truth, clone_store.order
    recorders = recorder_truth(truth)
    path = write_petracer_characters(recorders, truth, tmp_path / "characters.csv.gz")
    alleles = store.clones.import_alleles(path, cell_selection=every_cell(store))
    store.clones.export_characters(alleles, tmp_path / "source.csv", format="laml")
    store.clones.export_characters(
        alleles, tmp_path / "stored.csv", format="laml", cell_order="stored"
    )
    source = pd.read_csv(tmp_path / "source.csv", index_col=0)
    stored = pd.read_csv(tmp_path / "stored.csv", index_col=0)
    ids = np.asarray(store.cells.fetch_all("ids")).astype(str)
    position = {cell: row for row, cell in enumerate(ids)}
    stored_rows = np.asarray([position[cell] for cell in stored.index])
    assert stored_rows.tolist() == sorted(stored_rows.tolist())
    expected = stored.index[np.argsort(order[stored_rows], kind="stable")]
    assert source.index.tolist() == expected.tolist()
    pd.testing.assert_frame_equal(source, stored.loc[source.index])
    store.clones.export_characters(alleles, tmp_path / "source.zarr")
    obs = anndata.io.read_elem(
        zarr.open_group(str(tmp_path / "source.zarr"), mode="r")["obs"]
    )
    assert obs.index.tolist() == expected.tolist()
    with pytest.raises(TypeError, match="cell_order must be 'source' or 'stored'"):
        store.clones.export_characters(alleles, tmp_path / "x.zarr", cell_order=1)
    assert not (tmp_path / "x.zarr").exists()


def test_a_store_whose_source_rows_have_gaps_exports_and_merges_by_their_sort(
    imported, tmp_path
) -> None:
    """Source rows with gaps, as a subset or an import that drops cells leaves them, order by their sort."""
    shuffled = _open(imported.shuffled)
    ids = shuffled.cells.fetch_all("ids")
    keep = (_ORDER % 3 != 0) & (_ORDER < 150)
    SubsetZarr(
        str(tmp_path / "gaps.zarr"), [shuffled.RNA], cell_ids=ids[keep], nthreads=1
    ).dump()
    gaps = _open(tmp_path / "gaps.zarr")
    source_rows = gaps.cells.fetch_all(SOURCE_ROW)
    kept = np.sort(_ORDER[keep])
    # The values skip every third row and the last rows of the file.
    np.testing.assert_array_equal(np.sort(source_rows), kept)
    assert kept[-1] > kept.size and kept[0] == 1

    # Every export writes the kept cells in the file's order.
    to_h5ad(gaps.RNA, str(tmp_path / "gaps.h5ad"))
    exported = anndata.read_h5ad(tmp_path / "gaps.h5ad")
    assert _file_rows(exported.obs_names).tolist() == kept.tolist()
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS[kept])
    to_mtx(gaps.RNA, str(tmp_path / "gaps_mtx"))
    np.testing.assert_array_equal(
        np.asarray(mmread(tmp_path / "gaps_mtx" / "matrix.mtx").todense()).T,
        _COUNTS[kept],
    )
    assert (
        _file_rows(
            (tmp_path / "gaps_mtx" / "barcodes.tsv").read_text().split()
        ).tolist()
        == kept.tolist()
    )
    in_memory = gaps.to_anndata(cell_key="keep")
    expected = kept[_obs()["keep"].to_numpy()[kept]]
    assert _file_rows(in_memory.obs_names).tolist() == expected.tolist()

    # A merge ranks the kept cells by their sort: its rows have no gaps.
    DataStoreMerge(
        [gaps, _open(imported.plain)],
        str(tmp_path / "merged.zarr"),
        ["a", "b"],
        nthreads=1,
    ).dump()
    merged = _open(tmp_path / "merged.zarr")
    merged_ids = merged.cells.fetch_all("ids").astype(str)
    rows = _file_rows(merged_ids)
    expected_rows = np.where(
        np.char.startswith(merged_ids, "a__"),
        np.searchsorted(kept, rows),
        kept.size + rows,
    )
    np.testing.assert_array_equal(merged.cells.fetch_all(SOURCE_ROW), expected_rows)
    np.testing.assert_array_equal(
        np.sort(merged.cells.fetch_all(SOURCE_ROW)), np.arange(kept.size + _N_CELLS)
    )


# ---------------------------------------------------------------------------
# Round trips through other shuffled imports


def _cr_store(make_reader: Any, destination: Path, **options: Any) -> DataStore:
    from cytearc.writers import CrToZarr

    reader = make_reader()
    try:
        CrToZarr(
            reader, str(destination), mem_budget="256M", nthreads=1, **options
        ).dump()
    finally:
        reader.close()
    return _open(destination, default_assay="RNA")


def test_a_shuffled_matrix_market_import_exports_its_file(imported, tmp_path) -> None:
    from cytearc.readers import MtxReader, inspect_mtx

    # The file: the counts in the file's order, as to_mtx writes them.
    source = tmp_path / "source"
    to_mtx(_open(imported.plain).RNA, str(source))

    def reader() -> MtxReader:
        return MtxReader(inspect_mtx(str(source))[0], temp_dir=str(tmp_path))

    shuffled = _cr_store(reader, tmp_path / "shuffled.zarr", shuffle_cells=True)
    plain = _cr_store(reader, tmp_path / "plain.zarr", shuffle_cells=False)
    rows = _file_rows(shuffled.cells.fetch_all("ids"))
    assert rows.tolist() != sorted(rows.tolist())
    np.testing.assert_array_equal(shuffled.cells.fetch_all(SOURCE_ROW), rows)

    # Matrix Market gives back the file byte for byte.
    to_mtx(shuffled.RNA, str(tmp_path / "exported"))
    assert _mtx_files(tmp_path / "exported") == _mtx_files(source)
    # H5AD holds what the export of an import in the file's order holds.
    to_h5ad(shuffled.RNA, str(tmp_path / "shuffled.h5ad"))
    to_h5ad(plain.RNA, str(tmp_path / "plain.h5ad"))
    assert _h5_tree(tmp_path / "shuffled.h5ad") == _h5_tree(tmp_path / "plain.h5ad")
    exported = anndata.read_h5ad(tmp_path / "shuffled.h5ad")
    assert exported.obs_names.tolist() == [f"cell-{row}" for row in range(_N_CELLS)]
    np.testing.assert_array_equal(exported.X.toarray(), _COUNTS)


def test_a_filtered_10x_import_exports_its_kept_cells_in_the_files_order(
    tmp_path,
) -> None:
    from cytearc.readers import CrH5Reader
    from tests.test_shuffled_imports import _CUTOFF, _counts, _kept, _write_10x

    counts = _counts()
    path = _write_10x(tmp_path / "matrix.h5", counts)
    kept = _kept(counts, cutoff=True)
    genes = np.arange(9)

    def reader() -> CrH5Reader:
        return CrH5Reader(
            str(path),
            is_filtered=False,
            filtering_cutoff=_CUTOFF,
            temp_dir=str(tmp_path),
        )

    shuffled = _cr_store(reader, tmp_path / "shuffled.zarr", shuffle_cells=True)
    plain = _cr_store(reader, tmp_path / "plain.zarr", shuffle_cells=False)
    source_rows = shuffled.cells.fetch_all(SOURCE_ROW)
    # The reader dropped cells, so the file rows that sourceRow holds have
    # gaps straight from the import.
    assert kept.size < counts.shape[0]
    np.testing.assert_array_equal(np.sort(source_rows), kept)
    barcodes = [f"BC-{row}" for row in kept]

    to_h5ad(shuffled.RNA, str(tmp_path / "shuffled.h5ad"))
    exported = anndata.read_h5ad(tmp_path / "shuffled.h5ad")
    assert exported.obs_names.tolist() == barcodes
    np.testing.assert_array_equal(exported.X.toarray(), counts[kept][:, genes])
    to_h5ad(plain.RNA, str(tmp_path / "plain.h5ad"))
    assert _h5_tree(tmp_path / "shuffled.h5ad") == _h5_tree(tmp_path / "plain.h5ad")

    to_mtx(shuffled.ADT, str(tmp_path / "shuffled_mtx"))
    to_mtx(plain.ADT, str(tmp_path / "plain_mtx"))
    assert _mtx_files(tmp_path / "shuffled_mtx") == _mtx_files(tmp_path / "plain_mtx")
    assert (tmp_path / "shuffled_mtx" / "barcodes.tsv").read_text().split() == barcodes
    np.testing.assert_array_equal(
        np.asarray(mmread(tmp_path / "shuffled_mtx" / "matrix.mtx").todense()).T,
        counts[kept][:, 9:],
    )
