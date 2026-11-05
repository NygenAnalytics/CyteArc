"""The costs that clone calling and the clone operations charge to their ledger, bounded from both sides.

Each named cost is traced on shapes that it was measured on and shapes that
it was not, and must bound what the step holds from above, and from below
within a slack. The clone import and ``from_column`` run at the smallest
budgets they accept on long, four-byte and few-cell shapes, with the
budget at most a bounded multiple of the traced peak.
"""

import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.clones import calling
from cytearc.clones.assignments import ranked_clone_label_bytes, ranked_clone_labels
from cytearc.clones.calling import cell_sets, identical_sets, set_clones
from cytearc.clones.sequences import neighbor_pairs
from cytearc.datastore._operations import clones as clone_operations
from cytearc.datastore._operations.clone_steps import text_file_write_bytes
from cytearc.metadata.rows import metadata_column_stamp
from cytearc.storage.arrays import MISSING_MASK_PREFIX, zarr_write_buffer_bytes
from cytearc.storage.content_stamps import (
    stored_content_stamp,
    stored_content_stamp_bytes,
)
from cytearc.utils.memory_budget import (
    NUMPY_STEP_BYTES,
    MemoryBudgetError,
    MemoryLedger,
)
from cytearc.writers.clone_tables import write_clone_matrix, write_clone_table
from tests.clone_helpers import every_cell, write_cells_store
from tests.storage_helpers import insert_nullable_cell_column
from tests.memory_helpers import (
    at_budget,
    budget_that_runs,
    charges,
    interval_peaks,
    smallest_budget,
    traced_peak,
)


def _warm_peak(call: Callable[[], Any]) -> int:
    """The traced peak of ``call`` after a first run warms its caches."""
    call()
    return traced_peak(call, 1 << 40)[1]


# NumPy's fixed allocations in a step of clone calling.


def _tiny_steps(n_cells: int) -> dict[str, Callable[[], object]]:
    cells = np.arange(n_cells, dtype=np.int64).repeat(2)
    items = np.arange(2 * n_cells, dtype=np.int64) % 3
    sets = cell_sets(cells, items, n_cells)
    nodes = identical_sets(sets)
    edges = calling.jaccard_edges(nodes, 0.5, inclusive=True)
    return {
        "cell_sets": lambda: cell_sets(cells, items, n_cells),
        "identical_sets": lambda: identical_sets(sets),
        "jaccard_edges": lambda: calling.jaccard_edges(nodes, 0.5, inclusive=True),
        "pruned_edges": lambda: calling.pruned_edges(nodes.n_nodes, edges),
        "components": lambda: calling.connected_components(nodes.n_nodes, edges),
        "search": lambda: neighbor_pairs(["ACGTA", "ACGTT", "TTTTT"], "hamming", 1),
    }


def test_numpy_step_bytes_bound_each_tiny_step_from_both_sides() -> None:
    # On 1 to 50 cells a step's arrays are a few hundred bytes, so what it
    # holds is NumPy's objects and its sorts' fixed buffers: at most
    # NUMPY_STEP_BYTES, and the largest of them at least two thirds of it,
    # so an allowance that grew past what steps hold fails too.
    peaks = {}
    for n_cells in (1, 4, 50):
        for name, call in _tiny_steps(n_cells).items():
            peaks[name, n_cells] = _warm_peak(call)
    assert max(peaks.values()) <= NUMPY_STEP_BYTES, peaks
    assert max(peaks.values()) >= NUMPY_STEP_BYTES * 2 // 3, peaks


# 12's undercounts in clone calling.


def test_building_sets_counts_each_cells_pointer_and_count() -> None:
    # 400,000 cells of one item each: the sets keep each cell's pointer, and
    # sum each cell's count of items into them, 16 bytes a cell beside the
    # pairs; a charge of 8 bytes a cell fell short.
    n_cells = 400_000
    cells = np.arange(n_cells, dtype=np.int64)
    items = np.zeros(n_cells, dtype=np.int64)
    _result, taken = charges(
        lambda: cell_sets(cells, items, n_cells, ledger=MemoryLedger(None, "x"))
    )
    (stated,) = [nbytes for item, nbytes, _ in taken if item == "the sets of the cells"]
    peak = _warm_peak(lambda: cell_sets(cells, items, n_cells))
    assert peak <= stated <= 1.5 * peak + NUMPY_STEP_BYTES, (peak, stated)


def test_building_sets_of_many_distinct_pairs_holds_no_more_than_it_plans() -> None:
    # 2,000,000 distinct pairs of 10 cells: the pairs, their sorted cells and
    # items and the flags of new pairs held 25 bytes a pair, against the 24
    # charged, 2 MB beyond the plan.
    rng = np.random.default_rng(1)
    n_pairs = 2_000_000
    cells = rng.integers(0, 10, n_pairs)
    items = np.arange(n_pairs)

    def call() -> object:
        return cell_sets(cells, items, 10, ledger=MemoryLedger(None, "x"))

    call()
    _result, intervals = interval_peaks(call)
    ((planned, traced),) = [
        (planned, traced)
        for item, planned, traced in intervals
        if item == "the sets of the cells"
    ]
    assert traced <= planned + 65_536, (planned, traced)


def test_identical_sets_refuse_before_they_allocate_the_cells_sizes() -> None:
    # A budget below the sets' hashes is refused before the step makes its
    # two arrays of each cell's size: the refused call holds next to nothing.
    n_cells = 400_000
    rng = np.random.default_rng(3)
    cells = np.repeat(np.arange(n_cells), 3)
    sets = cell_sets(cells, rng.integers(0, 3_000, size=len(cells)), n_cells)

    def refused() -> object:
        with pytest.raises(MemoryBudgetError, match="the sets' hashes"):
            identical_sets(sets, ledger=MemoryLedger(1_000_000, "Calling clones"))
        return None

    assert _warm_peak(refused) < 8 * n_cells // 4


@pytest.mark.parametrize(
    ("n_cells", "per_cell", "distinct"),
    [
        # Single-cell clones: each component's size, flag, order and rank
        # are numbered beside each cell's arrays.
        (400_000, 1, None),
        # 3,000 edits shared by 400,000 cells: few nodes and many cells.
        (400_000, 3, 3_000),
    ],
)
def test_each_step_of_set_clones_holds_no_more_than_it_plans(
    n_cells, per_cell, distinct
) -> None:
    rng = np.random.default_rng(5)
    cells = np.repeat(np.arange(n_cells), per_cell)
    items = (
        np.arange(len(cells))
        if distinct is None
        else rng.integers(0, distinct, size=len(cells))
    )
    sets = cell_sets(cells, items, n_cells)

    def call() -> object:
        return set_clones(sets, method="shared", ledger=MemoryLedger(None, "x"))

    call()
    _result, intervals = interval_peaks(call)
    over = [entry for entry in intervals if entry[2] > entry[1] + 65_536]
    assert over == [], over
    # Numbering the clones plans no more than half again what it holds.
    ((planned, traced),) = [
        (planned, traced)
        for item, planned, traced in intervals
        if item == "the cells' clones" and planned > traced
    ][:1]
    assert planned <= 1.5 * traced + NUMPY_STEP_BYTES, (planned, traced)


def test_a_similarity_across_modalities_charges_the_nodes_modalities_first() -> None:
    # 400,000 cells of one to three items each, of two modalities: finding
    # each node's modality and its flags among the edges held 13.6 MB beyond
    # the plan when they were charged after they were made.
    rng = np.random.default_rng(0)
    n_cells = 400_000
    cells = np.repeat(np.arange(n_cells), rng.integers(1, 4, size=n_cells))
    sets = cell_sets(cells, np.arange(len(cells)), n_cells)
    modalities = rng.integers(0, 2, size=n_cells)

    def call() -> object:
        return set_clones(
            sets,
            method="jaccard",
            similarity=0.3,
            inclusive=False,
            across=0.5,
            modalities=modalities,
            ledger=MemoryLedger(None, "x"),
        )

    call()
    _result, intervals = interval_peaks(call)
    assert any(item == "the nodes' modalities" for item, _p, _t in intervals)
    over = [entry for entry in intervals if entry[2] > entry[1] + 65_536]
    assert over == [], over


def test_ambiguous_nodes_keep_their_flags_charged_with_the_nodes() -> None:
    # With unknown items, each node's flags of whether it remains and whether
    # it is ambiguous are part of what the nodes hold, beside their sets.
    rng = np.random.default_rng(9)
    n_cells = 50_000
    cells = np.repeat(np.arange(n_cells), 2)
    sets = cell_sets(cells, rng.integers(0, 40, size=len(cells)), n_cells)
    unknown = cell_sets(
        np.arange(0, n_cells, 2), 40 + rng.integers(0, 8, n_cells // 2), n_cells
    )

    def call() -> object:
        return set_clones(
            sets,
            unknown=unknown,
            method="jaccard",
            similarity=1.0,
            inclusive=True,
            ambiguous=True,
            ledger=MemoryLedger(None, "x"),
        )

    call()
    _result, intervals = interval_peaks(call)
    assert [entry for entry in intervals if entry[2] > entry[1] + 65_536] == []
    _result, taken = charges(call)
    held = next(nbytes for item, nbytes, _ in taken if item == "the nodes")
    nodes = identical_sets(sets, unknown=unknown)
    assert nodes.unknown is not None
    assert held == (
        8 * n_cells
        + 42 * nodes.n_nodes
        + 8 * len(nodes.sets.items)
        + 8 * (len(nodes.unknown.items) + nodes.n_nodes + 1)
    )


def test_keeping_the_edges_between_remaining_nodes_refuses_before_it_copies() -> None:
    # A budget below the copy of 400,000 edges is refused before the edges
    # kept and their flags are made: the refused call holds next to nothing.
    n_edges = 400_000
    edges = calling.Edges(np.arange(n_edges), np.arange(n_edges) + 1)
    present = np.ones(n_edges + 1, dtype=bool)

    def refused() -> object:
        with pytest.raises(MemoryBudgetError, match="the edges kept"):
            calling._kept_edges(
                MemoryLedger(1_000_000, "Calling clones"), edges, present
            )
        return None

    assert _warm_peak(refused) < n_edges


def test_removing_doublets_holds_each_step_within_its_plan() -> None:
    # 200,000 cells in 100,000 nodes of two cells, each node's set sharing
    # one of three items with the next: 100,000 edges at a similarity of a
    # third. Removing doublets keeps the edges between the nodes that remain
    # twice, each a copy of the edges.
    n_cells = 200_000
    cells = np.repeat(np.arange(n_cells), 2)
    first = np.arange(n_cells) // 2
    items = np.stack([first, first + 1], axis=1).ravel()
    sets = cell_sets(cells, items, n_cells)

    def call() -> object:
        return set_clones(
            sets,
            method="jaccard",
            similarity=0.3,
            inclusive=True,
            doublets=True,
            ledger=MemoryLedger(None, "x"),
        )

    clones = call()
    assert clones.n_edges == 99_999
    _result, intervals = interval_peaks(call)
    kept = [entry for entry in intervals if entry[0] == "the edges kept"]
    assert len(kept) == 2
    over = [entry for entry in intervals if entry[2] > entry[1] + 65_536]
    assert over == [], over


def test_the_search_leaves_the_pairs_it_returns_charged() -> None:
    # 2,000 texts of four letters, each within one substitution of others:
    # the search lets go of what it held but the pairs, 16 bytes each.
    rng = np.random.default_rng(4)
    texts = ["".join(rng.choice(list("ACGT"), 4)) for _ in range(2_000)]
    ledger = MemoryLedger(None, "Searching")
    ledger.hold("the caller's texts", 1_000)
    first, second = neighbor_pairs(texts, "hamming", 1, ledger=ledger)
    assert len(first) > 10_000
    assert ledger.held_bytes == 1_000 + 16 * len(first)
    assert first.nbytes + second.nbytes == 16 * len(first)


def test_the_search_lets_go_of_its_charges_once_the_memory_goes() -> None:
    # 20,000 texts of 12 bases: the texts and the last length's members, 16
    # bytes a text, outlived the release of what the search held, 322 KB
    # beyond the plan as the search returned.
    rng = np.random.default_rng(8)
    texts = ["".join(rng.choice(list("ACGT"), 12)) for _ in range(20_000)]

    def call() -> object:
        return neighbor_pairs(
            texts, "hamming", 1, ledger=MemoryLedger(None, "Searching")
        )

    call()
    _result, intervals = interval_peaks(call)
    over = [entry for entry in intervals if entry[2] > entry[1] + 65_536]
    assert over == [], over


# The labels of clones ranked by size.


@pytest.mark.parametrize("n_clones", [0, 1, 9, 10, 99, 100, 12_345])
def test_ranked_labels_cost_their_strings_and_references_exactly(n_clones) -> None:
    labels = ranked_clone_labels(n_clones)
    assert labels.tolist() == [f"clone_{rank}" for rank in range(1, n_clones + 1)]
    assert ranked_clone_label_bytes(n_clones) == 8 * n_clones + sum(
        map(sys.getsizeof, labels.tolist())
    )


# What assigning rows to clones works with.


@pytest.mark.parametrize("n_rows", [200_000, 1_000_000])
@pytest.mark.parametrize("kind", ["one label", "6,000 labels", "half without a label"])
def test_assigning_rows_holds_what_its_cost_states(n_rows, kind) -> None:
    from cytearc.clones.assignments import CloneAssigner, clone_assignment_bytes

    rng = np.random.default_rng(0)
    if kind == "one label":
        labels = np.full(n_rows, "k1", dtype=object)
    else:
        labels = np.asarray(
            [f"k{c}" for c in rng.integers(0, 6_000, n_rows)], dtype=object
        )
        if kind == "half without a label":
            labels[::2] = None
    cells = np.arange(n_rows, dtype=np.int64)
    ids = np.arange(n_rows).astype(str).astype(object)

    def assigned() -> int:
        # What add holds beside its arguments, less what it charges for the
        # labels it keeps, which the strings given already hold.
        ledger = MemoryLedger(None, "Assigning")
        assigner = CloneAssigner(ids, multiple="error", ledger=ledger)
        held = ledger.held_bytes
        _result, peak = traced_peak(lambda: assigner.add(cells, labels), 1 << 40)
        return peak - (ledger.held_bytes - held)

    assigned()
    traced = min(assigned() for _ in range(2))
    stated = clone_assignment_bytes(n_rows)
    # Rows of one label or of 6,000 hold 81 to 84 bytes a row with pandas'
    # hash table, within its bound; half of them without a label hold less.
    assert traced <= stated, (traced, stated)
    if kind != "half without a label":
        assert stated <= 1.5 * traced, (traced, stated)


# A stamp of stored chunks.


@pytest.mark.parametrize(
    ("kind", "rows", "chunk"),
    [
        ("random text", 300_000, 100_000),
        ("repeated text", 300_000, 100_000),
        ("integers", 2_000_000, 20_000),
        ("a few values", 48, 100_000),
    ],
)
def test_a_stamp_holds_what_its_cost_states(tmp_path, kind, rows, chunk) -> None:
    rng = np.random.default_rng(1)
    values = {
        "random text": lambda: np.asarray(
            ["".join(rng.choice(list("ACGT"), 40)) for _ in range(rows)]
        ),
        "repeated text": lambda: np.asarray([f"c{i % 50}" for i in range(rows)]),
        "integers": lambda: rng.integers(0, 2**60, rows),
        "a few values": lambda: np.asarray([f"x{i}" for i in range(rows)]),
    }[kind]()
    root = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    array = root.create_array(
        "values", shape=values.shape, chunks=(min(chunk, rows),), dtype=values.dtype
    )
    array[:] = values
    stated = stored_content_stamp_bytes(array)
    peak = _warm_peak(lambda: stored_content_stamp(array))
    # Twice the stored bytes of the largest batch, read from the file sizes,
    # beside the stamp's objects: within its objects' allowance of the peak.
    assert peak <= stated <= 1.3 * peak + 40 * 1024, (peak, stated)


def test_a_store_that_reports_no_sizes_counts_decoded_chunks(tmp_path) -> None:
    root = zarr.open_group(MemoryStore(), mode="w")
    values = np.random.default_rng(2).integers(0, 2**60, 400_000)
    array = root.create_array(
        "values", shape=values.shape, chunks=(100_000,), dtype=values.dtype
    )
    array[:] = values
    stated = stored_content_stamp_bytes(array)
    # Four chunks of 800,000 bytes fetch at once, each counted decoded and a
    # sixty-fourth more, twice.
    assert stated == 2 * 4 * (800_000 + 800_000 // 64) + 32 * 1024 + 4 * 6 * 1024
    assert _warm_peak(lambda: stored_content_stamp(array)) <= stated


# Writing clone files.


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
def test_the_export_charges_the_writers_buffers(tmp_path, compressed) -> None:
    n_cells = 8
    cells = np.asarray([f"cell{i:08d}" for i in range(n_cells)], dtype=object)
    codes = np.arange(n_cells) % 3
    ids = np.asarray([f"clone_{i}" for i in range(3)], dtype=object)
    suffix = ".gz" if compressed else ""
    stated = (
        text_file_write_bytes(compressed)
        + clone_operations._EXPORT_BYTES_PER_CELL * n_cells
    )
    peaks = [
        _warm_peak(
            lambda: write_clone_table(tmp_path / f"t.csv{suffix}", cells, ids[codes])
        ),
        _warm_peak(
            lambda: write_clone_matrix(tmp_path / f"m.mtx{suffix}", cells, codes, ids)
        ),
    ]
    # The text writer's buffers, and zlib's state with gzip, hold a few
    # cells' file; the matrix writes its lists beside it.
    assert max(peaks) <= stated <= 1.6 * max(peaks), (peaks, stated)


# The labels that clonotypes keep apart.


@pytest.mark.parametrize(("n_pairs", "n_early"), [(1_000, 500), (100_000, 50_000)])
def test_checking_the_labels_kept_apart_holds_what_fate_bias_charges(
    n_pairs, n_early
) -> None:
    import pandas as pd

    from cytearc.utils.memory_budget import hash_table_bytes

    fates = pd.DataFrame(
        {"receptor_type": np.where(np.arange(16) % 2, "TCR", "BCR")},
        index=[f"f{i}" for i in range(16)],
    )
    pair_codes = np.arange(n_pairs) // 2
    pair_fates = (pair_codes % 8) * 2
    early_codes = np.random.default_rng(0).integers(-1, n_pairs // 2, n_early)
    kept = (("receptor_type", np.full(n_early, "BCR", dtype=object)),)
    stated = (
        hash_table_bytes(n_pairs)
        + clone_operations._KEPT_LABEL_PAIR_BYTES * n_pairs
        + clone_operations._KEPT_LABEL_EARLY_BYTES * n_early
        + NUMPY_STEP_BYTES
    )
    peak = _warm_peak(
        lambda: clone_operations._refuse_clones_across_kept_labels(
            fates, pair_fates, pair_codes, early_codes, kept
        )
    )
    # The charge is 1.24 to 1.36 times the peak as traced, whether or not
    # pandas stores text with pyarrow, since the labels stay Python strings.
    assert peak <= stated <= 1.5 * peak, (peak, stated)


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


@pytest.mark.parametrize(
    ("column", "selection", "at_most"),
    [
        # The budgets were 1.47, 1.60, 1.70 and 1.30 times the traced peaks:
        # beside what the assigner works with, each block's read keeps
        # charged the chunk of the column that a codec thread may still hold,
        # up to the next read, 8 MB of four-byte labels.
        ("short", "every_cell", 1.7),
        ("long", "every_cell", 1.85),
        ("four_byte", "every_cell", 1.95),
        ("numbers", "every_cell", 1.5),
        # Few cells of many: the stamp reads every stored chunk of the
        # column, 1.95 times the traced peak.
        ("long", "few", 2.25),
    ],
)
def test_clones_of_a_column_run_at_the_budget_that_their_refusals_name(
    column_store, column, selection, at_most
) -> None:
    store = column_store
    cells = store.snapshot_cell_selection(selection)

    def save(reopened: Any) -> object:
        return reopened.clones.from_column(
            column, cell_selection=cells, invalidate_cache=True
        )

    # Each block takes half of the room, so a larger budget reads larger
    # blocks: the call runs at every budget from the one its refusals name,
    # which a byte less may run too, and holds no more than it. Numbers in
    # blocks of about 1,900 labels climb through 112 refusals to the budget
    # that the save needs, which counts a buffer for each write that Zarr
    # may run at once, 1.25 MiB from Python 3.14, beside the bound.
    budget, _refusals, ref, peak = budget_that_runs(store, save, max_refusals=200)
    assert peak is not None
    assert peak <= budget <= at_most * peak + zarr_write_buffer_bytes(), (budget, peak)
    assert ref.kind == "clones"
    # The labels are read a block at a time, after the stamp of the column.
    _result, taken = charges(lambda: save(store))
    items = [item for item, _nbytes, _planned in taken]
    assert items[0] == "a stamp of the column's stored chunks"
    assert any(item.startswith("a block of") for item in items)


def test_the_stamp_of_a_column_follows_its_stored_chunks(column_store) -> None:
    store = column_store
    cells = every_cell(store)
    first = store.clones.from_column("short", cell_selection=cells)
    stamp = metadata_column_stamp(store.cells, "short")
    # The same column gives the same stamp and reuses the clones, without
    # reading a label.
    assert metadata_column_stamp(store.cells, "short") == stamp
    _result, taken = charges(
        lambda: store.clones.from_column("short", cell_selection=cells)
    )
    assert [item for item, _nbytes, _planned in taken] == [
        "a stamp of the column's stored chunks"
    ]
    # An edited label changes the stamp and gives new clones.
    labels = store.cells.fetch_all("short").copy()
    labels[7] = "edited"
    store.cells.insert("short", labels, overwrite=True)
    assert metadata_column_stamp(store.cells, "short") != stamp
    assert store.clones.from_column("short", cell_selection=cells) != first


# 48 MB reads four blocks; the save alone needs 39.2 MB on Python 3.13 and
# 40.4 MB from 3.14, whose write buffers are larger.
@pytest.mark.parametrize("budget", [None, 48_000_000], ids=["one block", "blocks"])
def test_clones_of_a_column_of_many_cells_hold_each_step_within_its_plan(
    tmp_path, budget
) -> None:
    # 200,000 cells in 25,000 clones whose sizes follow a power law, saved
    # whole or in blocks: from each charge to the next, what is traced stays
    # within what the ledger planned, beside the audit's allowance. The
    # assigner's work on a block, its masks, positions, codes and sorted
    # copies, went uncharged, 6.6 MB past the plan in one block.
    rng = np.random.default_rng(3)
    n_cells = 200_000
    store = write_cells_store(
        tmp_path / "store.zarr", [f"cell{i:07d}-1" for i in range(n_cells)]
    )
    sizes = np.minimum(rng.zipf(1.8, size=n_cells), 300)
    clone_of = np.repeat(np.arange(len(sizes)), sizes)[:n_cells]
    rng.shuffle(clone_of)
    store.cells.insert("clone", np.asarray([f"clone{c}" for c in clone_of]))
    cells = every_cell(store)
    if budget is not None:
        store.memoryBytes = budget

    def save() -> object:
        return store.clones.from_column(
            "clone", cell_selection=cells, invalidate_cache=True
        )

    from cytearc.storage.validation_scope import validation_scope

    with validation_scope():
        save()
        _result, intervals = interval_peaks(save)
    blocks = [item for item, _p, _t in intervals if item.startswith("assigning")]
    assert (len(blocks) == 1) == (budget is None), blocks
    over = [entry for entry in intervals if entry[2] > entry[1] + 65_536]
    assert over == [], over


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


# The clone import at the shapes that traced past their budgets.


def _write_long_table(folder: Path) -> tuple[Any, Path]:
    """A store of 2,000 cells and a table of 400,000 rows that names them and 398,000 barcodes of no cell."""
    ids = [f"c{index:06d}" for index in range(2_000)]
    store = write_cells_store(folder / "store.zarr", ids)
    others = [f"x{index:07d}" for index in range(398_000)]
    table = folder / "clones.csv"
    table.write_text(
        "Cell_ID,Clone_ID\n"
        + "".join(f"{cell},{index % 97}\n" for index, cell in enumerate(ids + others))
    )
    return store, table


def _write_matrix(
    folder: Path, n_listed: int = 300_000, n_cells: int = 2_000
) -> tuple[Any, Path]:
    """A store of the first ``n_cells`` of a clone matrix's ``n_listed`` listed cells, one clone of 50 each."""
    names = [f"c{i:07d}" for i in range(n_listed)]
    matrix = folder / "matrix"
    matrix.mkdir()
    (matrix / "cells.tsv").write_text("cell\n" + "".join(n + "\n" for n in names))
    (matrix / "clones.tsv").write_text(
        "clone\n" + "".join(f"k{i}\n" for i in range(50))
    )
    (matrix / "clones.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        + f"{n_listed} 50 {n_listed}\n"
        + "".join(f"{i + 1} {i % 50 + 1} 1\n" for i in range(n_listed))
    )
    store = write_cells_store(folder / "store.zarr", names[:n_cells])
    return store, matrix / "clones.mtx"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("shape", "at_most"),
    [
        # The budgets accepted were 1.30 and 1.55 times the traced peaks.
        ("long table", 1.45),
        # The matrix's cell list is parsed in chunks of one field, of ASCII
        # IDs that hold one byte a character where each character of the
        # longest sampled row is counted at four, the most that any holds.
        ("matrix of many cells", 1.75),
    ],
)
def test_the_import_holds_its_parse_and_join_within_its_budget(
    tmp_path, shape, at_most
) -> None:
    # On C7's and C8's tips these imports traced 1.09 and 1.25 times the
    # smallest budgets they accepted, since their charges left out what
    # parsing a chunk and joining it hold. Their charges now cover the parse
    # and the join, and the budgets they accept stay within a bound of the
    # traced peaks: a file's chunk takes half of the room, which leaves the
    # other half to the hashes of the barcodes that match no selected cell.
    writer = _write_long_table if shape == "long table" else _write_matrix
    store, path = writer(tmp_path)
    cells = every_cell(store)

    def run() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return store.clones.import_clones(
                path, cell_selection=cells, invalidate_cache=True
            )

    # A larger budget parses larger chunks, so the budgets that refusals
    # name climb slowly; the smallest budget accepted is found by bisection,
    # as the measurements on C7 and C8 found it.
    budget = smallest_budget(at_budget(store, run), 400_000_000)
    store.memoryBytes = budget
    ref, peak = traced_peak(run, budget)
    assert peak <= budget <= at_most * peak, (budget, peak)
    assert ref.kind == "clones"


@pytest.mark.slow
def test_rows_longer_than_the_sampled_rows_make_chunks_of_fewer_rows(
    tmp_path,
) -> None:
    # 2,000 short rows that name the cells, then 58,000 rows of 206
    # characters that name no cell. Chunks sized by the short rows sampled
    # held 1.54 times the smallest budget accepted at 200,000 rows; a chunk
    # now holds at most its rows' characters, so the call holds no more than
    # its budget.
    ids = [f"c{index:06d}" for index in range(2_000)]
    store = write_cells_store(tmp_path / "store.zarr", ids)
    path = tmp_path / "clones.csv"
    path.write_text(
        "Cell_ID,Clone_ID\n"
        + "".join(f"{cell},k{index % 97}\n" for index, cell in enumerate(ids))
        + "".join(f"x{index:0200d},k{index % 97}\n" for index in range(58_000))
    )
    cells = every_cell(store)

    def run() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return store.clones.import_clones(
                path, cell_selection=cells, invalidate_cache=True
            )

    budget = smallest_budget(at_budget(store, run), 400_000_000)
    store.memoryBytes = budget
    ref, peak = traced_peak(run, budget)
    assert peak <= budget, (budget, peak)
    assert ref.kind == "clones"


def test_a_matrix_lets_go_of_its_cells_and_labels_once_it_is_read(tmp_path) -> None:
    # A matrix of 5,000 listed cells and 50 clones: each listed cell's
    # position among the selected cells, 8 bytes a cell, and the clones'
    # labels are held while its entries are assigned, and let go with its
    # chunk once they are, so the steps after the file plan with the room
    # that the file no longer takes.
    store, path = _write_matrix(tmp_path, n_listed=5_000)
    cells = every_cell(store)

    def run() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return store.clones.import_clones(
                path, cell_selection=cells, invalidate_cache=True
            )

    _result, taken = charges(run)
    items = [item for item, _nbytes, _planned in taken]
    chunk = next(n for item, n, _p in taken if item.startswith("a chunk of"))
    (labels,) = [n for item, n, _p in taken if item == "the clone matrix's labels"]
    assert labels >= sum(sys.getsizeof(f"k{i}") for i in range(50))
    released = items.index("the clone matrix's labels") + 1
    released += next(
        at for at, item in enumerate(items[released:]) if item == "a release"
    )
    assert taken[released][1] == chunk + 8 * 5_000 + labels, taken[released]


def test_a_matrix_of_selected_cells_holds_each_step_within_its_plan(
    tmp_path,
) -> None:
    # 50,000 listed cells, every one selected, in 50 clones: the listed
    # cells' positions and the clones' labels stay charged while the
    # entries are assigned, 400 KB beyond the plan were they let go first.
    store, path = _write_matrix(tmp_path, n_listed=50_000, n_cells=50_000)
    cells = every_cell(store)

    def run() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return store.clones.import_clones(
                path, cell_selection=cells, invalidate_cache=True
            )

    from cytearc.storage.validation_scope import validation_scope

    with validation_scope():
        run()
        _result, intervals = interval_peaks(run)
    over = [entry for entry in intervals if entry[2] > entry[1] + 65_536]
    assert over == [], over


def test_each_block_of_a_column_takes_at_most_half_of_the_room(column_store) -> None:
    # At a budget of a few blocks, each block's labels, their text and what
    # assigning them holds take at most half of what the ledger leaves when
    # the block is planned, beside what a step of NumPy calls holds and the
    # hash table's growth to its next power of two.
    store = column_store
    cells = every_cell(store)
    store.memoryBytes = 12_000_000
    try:
        _result, taken = charges(
            lambda: store.clones.from_column(
                "short", cell_selection=cells, invalidate_cache=True
            )
        )
    finally:
        store.memoryBytes = 1 << 34
    starts = [
        at for at, (item, _n, _p) in enumerate(taken) if item.startswith("a block of")
    ]
    blocks = 0
    for at, (item, _nbytes, planned) in enumerate(taken):
        if not item.startswith("assigning a block of"):
            continue
        n_rows = int(item.split()[4])
        start = max(index for index in starts if index < at)
        before = taken[start - 1][2]
        assert planned - before <= (12_000_000 - before) / 2 + 128 * 1024, (
            n_rows,
            planned - before,
            12_000_000 - before,
        )
        blocks += 1
    assert blocks > 1, blocks


def test_a_chunk_of_a_clone_table_charges_what_parsing_it_holds(tmp_path) -> None:
    # 200,000 rows of a barcode and a clone, short and four-byte: a parsed
    # row holds 117 to 121 bytes beside one byte a character of ASCII text
    # and four of text beyond U+FFFF, which the chunk charges at four bytes
    # a character of the longest sampled row. The rows are parsed as Python
    # strings whether or not pandas stores text with pyarrow, which held
    # 336 bytes a four-byte row against the 332 charged when it did.
    ids = [f"c{index:06d}" for index in range(2_000)]
    store = write_cells_store(tmp_path / "store.zarr", ids)
    store.memoryBytes = 1 << 34
    cells = every_cell(store)
    for name, barcode in (
        ("ascii", lambda i: f"c{i:06d}" if i < 2_000 else f"x{i:039d}"),
        # The four-byte barcodes come first, among the rows that the layout
        # samples for the longest row.
        ("four-byte", lambda i: f"c{i:06d}" if i >= 198_000 else f"\U0001f600{i:039d}"),
    ):
        path = tmp_path / f"{name}.csv"
        path.write_text(
            "Cell_ID,Clone_ID\n"
            + "".join(f"{barcode(i)},{i % 97}\n" for i in range(200_000)),
            encoding="utf-8",
        )

        def run() -> object:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return store.clones.import_clones(
                    path, cell_selection=cells, invalidate_cache=True
                )

        run()
        _result, intervals = interval_peaks(run)
        chunk = next(
            at
            for at, entry in enumerate(intervals)
            if entry[0].startswith("a chunk of")
        )
        before = intervals[chunk - 1][1]
        # Without a budget a chunk plans for its most rows, and the whole
        # file is one chunk of its 200,000.
        chunk_rows = int(intervals[chunk][0].split()[3])
        planned = (intervals[chunk][1] - before) / chunk_rows
        traced = (intervals[chunk][2] - before) / 200_000
        assert traced <= planned <= 1.6 * traced, (name, planned, traced)


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
