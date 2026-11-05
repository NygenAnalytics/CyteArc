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


# Clonotypes across receptor types, and the steps of fate bias and tracking
# that a busy machine's held chunks showed short at 40,000 cells.


def _receptor_type_input(n_cells: int) -> Any:
    """Clonotypes of three cells each, of four receptor types in turn, and their groups' strata."""
    import pandas as pd

    from cytearc.storage.refs import ArtifactRef

    labels = np.asarray(["BCR", "TCR", "ambiguous", "multichain"], dtype=object)
    unsaved = ArtifactRef(scope="datastore", kind="clonotypes", artifact_id="f" * 64)
    codes = np.where(np.arange(n_cells) % 9 == 0, -1, np.arange(n_cells) // 3)
    source = clone_operations._CloneInput(
        ref=unsaved,
        cell_selection=unsaved,
        codes=codes,
        clone_ids=None,
        kept_apart=(("receptor_type", labels[np.arange(n_cells) % 4]),),
        nbytes=0,
    )
    keys = pd.DataFrame(
        {"receptor_type": labels.tolist()}, index=[f"g{i}" for i in range(4)]
    )
    return source, keys, np.arange(4, dtype=np.int64)


def _parts_of_clonotypes(n_cells: int, per_part: int, spanning: float) -> tuple:
    """Cells of clonotypes within types, a share of which pair up into clonotypes of two types."""
    rng = np.random.default_rng(per_part)
    within = rng.integers(0, n_cells // per_part, n_cells)
    within[::50] = -1
    present = np.unique(within[within >= 0])
    within = np.where(within >= 0, np.searchsorted(present, within), -1)
    n_parts = len(present)
    n_pairs = int(spanning * n_parts / 2)
    part_clonotype = np.concatenate(
        [np.repeat(np.arange(n_pairs), 2), np.arange(n_pairs, n_parts - n_pairs)]
    )
    part_type = np.concatenate(
        [np.tile([0, 1], n_pairs), np.ones(n_parts - 2 * n_pairs, dtype=np.int64)]
    )
    shuffled = rng.permutation(n_parts)
    part_clonotype, part_type = part_clonotype[shuffled], part_type[shuffled]
    at = np.maximum(within, 0)
    codes = np.where(within >= 0, part_clonotype[at], -1)
    types = np.where(within >= 0, part_type[at], 2)
    return codes, within, types


@pytest.mark.parametrize(
    ("n_cells", "per_part", "spanning"),
    [(40_000, 3, 0.1), (400_000, 1, 0.0), (400_000, 10, 0.5), (200_000, 1, 1.0)],
    ids=["a tenth spanning", "none spanning", "half spanning", "all spanning"],
)
def test_naming_clonotypes_within_types_holds_what_each_step_charges(
    n_cells, per_part, spanning
) -> None:
    from cytearc.repertoire.clonotypes import within_type_parts
    from cytearc.utils.memory_budget import MemoryLedger
    from tests.memory_helpers import interval_peaks

    codes, within, types = _parts_of_clonotypes(n_cells, per_part, spanning)

    def run() -> tuple:
        ledger = MemoryLedger(1 << 40, "Naming clonotypes within types")
        # A charge of nothing before the call shows what it allocates before
        # its own first charge.
        ledger.hold("the call", 0)
        return within_type_parts(
            codes, within, types, ["BCR", "TCR", "no IR"], ledger=ledger
        )

    clonotypes, added = run()
    # Each clonotype within a type takes its cells' clonotype, and adds a
    # text where its clonotype holds another.
    assert (clonotypes[within[within >= 0]] == codes[codes >= 0]).all()
    shared = np.bincount(clonotypes)[clonotypes] > 1
    assert [bool(text) for text in added.tolist()] == shared.tolist()
    _result, intervals = interval_peaks(run)
    (_call, _nothing, before), *steps = intervals
    # Before its first charge, the call holds its own objects alone, 1.6 KB
    # as traced.
    assert before <= 16 * 1024, before
    assert len(steps) == (3 if shared.any() else 2)
    # Each step plans 1.18 to 1.62 times what it traced.
    for item, planned, traced in steps:
        assert traced <= planned <= 2 * traced, (item, planned, traced)


@pytest.mark.parametrize("n_cells", [40_000, 400_000])
def test_clearing_the_labels_of_cells_without_a_clone_holds_what_fate_bias_charges(
    n_cells,
) -> None:
    source, _keys, _strata = _receptor_type_input(n_cells)
    labels = source.kept_apart[0][1]
    cleared = clone_operations._cleared_labels(source.codes, source.kept_apart)
    # A cell without a clone loses its label, and the labels given stay.
    expected = np.where(source.codes < 0, None, labels)
    assert cleared[0][1].tolist() == expected.tolist()
    assert labels[0] == "BCR" and source.codes[0] == -1
    stated = clone_operations._cleared_label_bytes(n_cells, 1)
    peak = _warm_peak(
        lambda: clone_operations._cleared_labels(source.codes, source.kept_apart)
    )
    # The charge is 1.34 and 1.40 times the traced peak.
    assert peak <= stated <= 1.6 * peak, (peak, stated)


@pytest.mark.parametrize("n_cells", [40_000, 400_000])
def test_finding_the_first_cells_of_named_clones_holds_what_tracking_charges(
    n_cells,
) -> None:
    source, keys, strata = _receptor_type_input(n_cells)
    # Clone 5 starts at cell 15, of BCR, clone 77 at cell 231, of TCR, and the
    # last clone at a cell of its type.
    chosen = np.asarray([5, 77, n_cells // 3 - 2], dtype=np.int64)
    first = 3 * chosen
    first[first % 9 == 0] += 1
    found = clone_operations._fallback_strata(source, keys, strata, chosen)
    assert found.tolist() == (first % 4).tolist()
    stated = clone_operations._FIRST_CELL_BYTES * n_cells + NUMPY_STEP_BYTES
    peak = _warm_peak(
        lambda: clone_operations._fallback_strata(source, keys, strata, chosen)
    )
    # The charge is 1.25 and 1.29 times the traced peak.
    assert peak <= stated <= 1.5 * peak, (peak, stated)


@pytest.mark.parametrize(("n_cells", "n_groups"), [(40_000, 24), (400_000, 2_000)])
def test_choosing_and_finding_the_clones_followed_hold_what_tracking_charges(
    n_cells, n_groups
) -> None:
    from cytearc.clones.groups import pair_counts
    from cytearc.clones.overlap import top_clones, tracking_reach

    rng = np.random.default_rng(0)
    pair_groups, pair_codes, pair_cells = pair_counts(
        rng.integers(0, n_groups, n_cells), rng.integers(0, n_cells // 4, n_cells)
    )
    chosen = top_clones(pair_groups, pair_codes, pair_cells, n_groups, 5)
    strata = np.arange(n_groups) % 2
    unheld = np.full(len(chosen), -1, dtype=np.int64)
    stated = (
        clone_operations._BYTES_PER_CLONE_COUNT * len(pair_cells) + NUMPY_STEP_BYTES
    )
    # The charge is 1.21 times choosing's traced peak and 1.54 to 1.57 times
    # finding the chosen clones' counts.
    for step in (
        lambda: top_clones(pair_groups, pair_codes, pair_cells, n_groups, 5),
        lambda: tracking_reach(pair_groups, pair_codes, chosen, strata, unheld),
    ):
        peak = _warm_peak(step)
        assert peak <= stated <= 1.8 * peak, (peak, stated)


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
