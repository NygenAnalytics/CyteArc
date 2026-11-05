"""Clone calling's steps refuse budgets too small for them, and the clone operations give their results.

Calling's steps refuse a budget below what they would build before they
build it, the clones of a metadata column read in blocks at a small budget
are the clones of one read, and a column's stamp follows its labels and its
missing mask.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.clones import calling
from cytearc.clones.assignments import ranked_clone_labels
from cytearc.clones.calling import cell_sets, identical_sets, set_clones
from cytearc.metadata.rows import metadata_column_stamp
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
)
from tests.clone_helpers import every_cell, write_cells_store
from tests.storage_helpers import insert_nullable_cell_column


def test_identical_sets_refuse_a_budget_below_the_sets_hashes() -> None:
    # A megabyte cannot hold the hashes of 400,000 cells' sets.
    n_cells = 400_000
    rng = np.random.default_rng(3)
    cells = np.repeat(np.arange(n_cells), 3)
    sets = cell_sets(cells, rng.integers(0, 3_000, size=len(cells)), n_cells)
    with pytest.raises(MemoryBudgetError, match="the sets' hashes"):
        identical_sets(sets, ledger=MemoryLedger(1_000_000, "Calling clones"))


def test_keeping_the_edges_between_remaining_nodes_refuses_a_small_budget() -> None:
    # A megabyte cannot hold a copy of 400,000 edges.
    n_edges = 400_000
    edges = calling.Edges(np.arange(n_edges), np.arange(n_edges) + 1)
    present = np.ones(n_edges + 1, dtype=bool)
    with pytest.raises(MemoryBudgetError, match="the edges kept"):
        calling._kept_edges(MemoryLedger(1_000_000, "Calling clones"), edges, present)


def test_removing_doublets_from_a_chain_of_nodes_keeps_its_edges() -> None:
    # 200,000 cells in 100,000 nodes of two cells, each node's set sharing
    # one of three items with the next: 99,999 edges at a similarity of a
    # third, which removing doublets keeps.
    n_cells = 200_000
    cells = np.repeat(np.arange(n_cells), 2)
    first = np.arange(n_cells) // 2
    items = np.stack([first, first + 1], axis=1).ravel()
    sets = cell_sets(cells, items, n_cells)

    clones = set_clones(
        sets,
        method="jaccard",
        similarity=0.3,
        inclusive=True,
        doublets=True,
        ledger=MemoryLedger(None, "x"),
    )
    assert clones.n_edges == 99_999


# The labels of clones ranked by size.


@pytest.mark.parametrize("n_clones", [0, 1, 9, 10, 99, 100, 12_345])
def test_ranked_labels_name_each_clone_by_its_rank(n_clones) -> None:
    labels = ranked_clone_labels(n_clones)
    assert labels.tolist() == [f"clone_{rank}" for rank in range(1, n_clones + 1)]


# Clones of a cell metadata column.


@pytest.fixture(scope="module")
def column_store(tmp_path_factory) -> Any:
    """A store of 60,000 cells with columns of short, long and four-byte clone labels."""
    path = tmp_path_factory.mktemp("from_column") / "store.zarr"
    n_cells = 60_000
    store = write_cells_store(path, [f"c{i:07d}" for i in range(n_cells)])
    rng = np.random.default_rng(11)
    clone_of = rng.integers(0, 6_000, n_cells)
    store.cells.insert("short", np.asarray([f"k{c}" for c in clone_of]))
    store.cells.insert("long", np.asarray([f"{'ACGT' * 40}_{c}" for c in clone_of]))
    store.cells.insert(
        "four_byte", np.asarray(["\U0001f600" * 12 + str(c) for c in clone_of])
    )
    store.cells.insert("numbers", clone_of.astype(np.int64))
    store.cells.insert("few", np.arange(n_cells) < 48)
    return store


def test_the_stamp_of_a_column_follows_its_stored_chunks(column_store) -> None:
    store = column_store
    cells = every_cell(store)
    first = store.clones.from_column("short", cell_selection=cells)
    stamp = metadata_column_stamp(store.cells, "short")
    # The same column gives the same stamp and reuses the clones.
    assert metadata_column_stamp(store.cells, "short") == stamp
    assert store.clones.from_column("short", cell_selection=cells) == first
    # An edited label changes the stamp and gives new clones.
    labels = store.cells.fetch_all("short").copy()
    labels[7] = "edited"
    store.cells.insert("short", labels, overwrite=True)
    assert metadata_column_stamp(store.cells, "short") != stamp
    assert store.clones.from_column("short", cell_selection=cells) != first


def test_cells_whose_labels_are_missing_join_no_clone_until_the_mask_changes(
    tmp_path,
) -> None:
    # The stored labels of cells 0 and 4 are labels like the others, which
    # only the missing mask hides: those cells join no clone. Marking cell 2
    # missing edits the mask alone, which changes the stamp and gives new
    # clones without it.
    store = write_cells_store(tmp_path / "store.zarr", [f"c{i}" for i in range(8)])
    labels = np.asarray(["k1", "k1", "k2", "k2", "k3", "k3", "k1", "k2"])
    missing = np.isin(np.arange(8), [0, 4])
    insert_nullable_cell_column(store, "partial", labels, missing)
    cells = every_cell(store)
    first = store.clones.from_column("partial", cell_selection=cells)
    loaded = store.clones.load_clones(first).cells
    assert loaded["clone"].isna().tolist() == missing.tolist()
    assert loaded["clone_size"].tolist() == [0, 2, 3, 3, 0, 1, 2, 3]
    stamp = metadata_column_stamp(store.cells, "partial")
    store.zw["cellData"][MISSING_MASK_PREFIX + "partial"][:] = np.isin(
        np.arange(8), [0, 2, 4]
    )
    assert metadata_column_stamp(store.cells, "partial") != stamp
    second = store.clones.from_column("partial", cell_selection=cells)
    assert second != first
    assert store.clones.load_clones(second).cells["clone_size"].tolist() == [
        0,
        2,
        0,
        2,
        0,
        1,
        2,
        2,
    ]


@pytest.mark.parametrize(
    ("column", "bytes_per_cell"), [("short", 300), ("four_byte", 500), ("numbers", 200)]
)
def test_clones_of_a_column_read_in_blocks_are_the_clones_of_one_read(
    column_store, column, bytes_per_cell
) -> None:
    # A few hundred bytes a cell read the labels of 60,000 cells in more and
    # smaller blocks than 16 GiB does, and give the same clones.
    store = column_store
    cells = every_cell(store)
    small = bytes_per_cell * store.cells.N
    loaded = {}
    for budget in (small, 1 << 34):
        store.memoryBytes = budget
        try:
            ref = store.clones.from_column(
                column, cell_selection=cells, invalidate_cache=True
            )
        finally:
            store.memoryBytes = 1 << 34
        loaded[budget] = store.clones.load_clones(ref)
    pd.testing.assert_frame_equal(loaded[small].cells, loaded[1 << 34].cells)
    pd.testing.assert_frame_equal(loaded[small].clones, loaded[1 << 34].clones)


# The groups' labels of clonal expansion.


def test_the_expansions_group_labels_read_back_as_the_grouping_gave_them(
    column_store,
) -> None:
    store = column_store
    store.cells.insert(
        "sample", np.asarray([f"s{i % 3}" for i in range(store.cells.N)])
    )
    store.cells.insert(
        "donor", np.asarray([f"d\U0001f600{i % 2}" for i in range(store.cells.N)])
    )
    clones = store.clones.from_column("short", cell_selection=every_cell(store))
    saved = store.clones.expansion(clones, group_by=["sample", "donor"])
    group = store.artifacts.load(saved)
    # The labels are an array of each group's label in each key column, and
    # no attribute holds them.
    assert "group_keys" not in group.attrs and group.attrs["n_groups"] == 6
    assert group["group_labels"].shape == (12,)
    keys = store.clones.load_expansion(saved).group_keys
    assert keys.columns.tolist() == ["sample", "donor"]
    assert keys.values.tolist() == [
        [sample, donor]
        for sample in ("s0", "s1", "s2")
        for donor in ("d\U0001f6000", "d\U0001f6001")
    ]
    # Without key columns, the one group has no labels to store.
    alone = store.clones.expansion(clones)
    assert "group_labels" not in store.artifacts.load(alone)
    assert store.clones.load_expansion(alone).group_keys.shape == (1, 0)
