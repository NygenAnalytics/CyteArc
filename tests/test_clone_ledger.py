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
from cytearc.metadata.rows import metadata_column_stamp
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


# The labels of clones ranked by size.


@pytest.mark.parametrize("n_clones", [0, 1, 9, 10, 99, 100, 12_345])
def test_ranked_labels_cost_their_strings_and_references_exactly(n_clones) -> None:
    labels = ranked_clone_labels(n_clones)
    assert labels.tolist() == [f"clone_{rank}" for rank in range(1, n_clones + 1)]
    assert ranked_clone_label_bytes(n_clones) == 8 * n_clones + sum(
        map(sys.getsizeof, labels.tolist())
    )


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
    fixed = (
        clone_operations._EXPORT_GZIP_BYTES
        if compressed
        else clone_operations._EXPORT_TEXT_BYTES
    )
    stated = fixed + clone_operations._EXPORT_BYTES_PER_CELL * n_cells
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
    assert peak <= stated <= 1.4 * peak, (peak, stated)


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
        ("short", "every_cell", 1.6),
        ("long", "every_cell", 1.6),
        ("four_byte", "every_cell", 1.6),
        ("numbers", "every_cell", 1.6),
        # Few cells of many: the stamp reads every stored chunk of the column.
        ("long", "few", 2.6),
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
    # which a byte less may run too, and holds no more than it.
    budget, _refusals, ref, peak = budget_that_runs(store, save, max_refusals=100)
    assert peak is not None and peak <= budget <= at_most * peak, (budget, peak)
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


def _write_matrix(folder: Path) -> tuple[Any, Path]:
    """A store of the first 2,000 of a clone matrix's 300,000 listed cells, one clone of 50 each."""
    n_listed = 300_000
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
    store = write_cells_store(folder / "store.zarr", names[:2_000])
    return store, matrix / "clones.mtx"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("shape", "at_most"),
    [
        ("long table", 1.4),
        # The matrix's cell list is parsed in chunks of one field, of ASCII
        # IDs that hold one byte a character where each character of the
        # longest sampled row is counted at four, the most that any holds.
        ("matrix of many cells", 1.6),
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


def test_a_chunk_of_a_clone_table_charges_what_parsing_it_holds(tmp_path) -> None:
    # 200,000 rows of a barcode and a clone, short and four-byte: a parsed
    # row holds 117 to 121 bytes beside one byte a character of ASCII text
    # and four of text beyond U+FFFF, which the chunk charges at four bytes
    # a character of the longest sampled row.
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
