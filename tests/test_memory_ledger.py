"""The memory ledger and its refusals, and the reads and steps that charge it.

A ledger refuses a charge beyond its budget naming the step, the bytes it
needs and the budget. The reads of stored arrays, cell selections and label
artifacts, and the steps that code labels and number samples, give with a
ledger what they give without one, and refuse a budget too small for them
before they read.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from cytearc.metadata.rows import read_rows, read_text_rows, read_whole
from cytearc.metadata.selection import (
    category_codes,
    plan_cell_selection_read,
    read_cell_selection,
    sample_codes,
)
from cytearc.storage.async_execution import zarr_io_concurrency
from cytearc.storage.geometry import array_geometry
from cytearc.storage.selections import resolve_generated_selection_artifact
from cytearc.utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
    memory_refusal,
    python_text_bytes,
)
from tests.memory_helpers import (
    named_budget,
)


# The ledger


def test_a_refusal_names_the_step_the_bytes_it_needs_and_the_budget() -> None:
    ledger = MemoryLedger(1_000, "Importing contigs for 8 selected cells")
    ledger.hold("each cell's slots", 600)
    with pytest.raises(MemoryBudgetError) as refused:
        ledger.peak("the cells' IDs", 401)
    error = refused.value
    assert str(error) == (
        "Importing contigs for 8 selected cells holds about 1001 bytes for the "
        "cells' IDs, more than the memory budget of 1000 bytes. Raise mem_budget "
        "to at least 1001 bytes."
    )
    assert (error.needed_bytes, error.budget_bytes, error.item) == (
        1001,
        1000,
        "the cells' IDs",
    )
    assert isinstance(error, MemoryError)
    assert named_budget(error) == 1001
    # 400 more fit exactly, and nothing was charged by the refusal.
    ledger.peak("the cells' IDs", 400)
    assert ledger.held_bytes == 600


def test_a_refusal_names_the_remedy_of_the_charge_or_of_the_ledger() -> None:
    ledger = MemoryLedger(10, "Exporting", remedy="pass fields= with fewer fields")
    with pytest.raises(MemoryBudgetError) as refused:
        ledger.hold("a block", 11)
    assert str(refused.value).endswith(
        "Raise mem_budget to at least 11 bytes, or pass fields= with fewer fields."
    )
    with pytest.raises(MemoryBudgetError) as refused:
        ledger.peak("the IDs", 12, remedy="label fewer clonotypes")
    assert str(refused.value).endswith(", or label fewer clonotypes.")
    assert named_budget(refused.value) == 12


def test_the_refusal_builder_reads_as_the_ledger_refuses() -> None:
    built = memory_refusal("Planning count shards", "one row band", 5_000, 100)
    ledger = MemoryLedger(100, "Planning count shards")
    with pytest.raises(MemoryBudgetError) as refused:
        ledger.peak("one row band", 5_000)
    assert str(built) == str(refused.value)
    assert named_budget(built) == 5_000


@pytest.mark.parametrize(
    ("needed", "budget", "error", "message"),
    [
        (True, 10, TypeError, "needed_bytes must be a number of bytes, not True"),
        (11.0, 10, TypeError, "needed_bytes must be a number of bytes, not 11.0"),
        (11, "10", TypeError, "budget_bytes must be a number of bytes, not '10'"),
        (11, -1, ValueError, "budget_bytes must be at least 0, not -1"),
        (10, 10, ValueError, "needs more than the budget of 10 bytes, not 10"),
    ],
)
def test_the_refusal_builder_refuses_bytes_that_name_no_refusal(
    needed, budget, error, message
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        memory_refusal("Planning", "one band", needed, budget)


def test_the_refusal_builder_takes_numpy_integers_as_bytes() -> None:
    built = memory_refusal("Planning", "one band", np.int64(11), np.uint32(10))
    assert (built.needed_bytes, built.budget_bytes) == (11, 10)
    assert type(built.needed_bytes) is int and type(built.budget_bytes) is int
    assert named_budget(built) == 11


def test_a_refusal_keeps_its_fields_through_a_pickle() -> None:
    error = memory_refusal("Defining", "the chains", 9_000, 8_000, remedy="go on")
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is MemoryBudgetError
    assert (restored.needed_bytes, restored.budget_bytes, restored.item) == (
        9_000,
        8_000,
        "the chains",
    )
    assert str(restored) == str(error)


def test_releases_and_charges_refuse_what_was_never_charged() -> None:
    ledger = MemoryLedger(100, "Counting")
    ledger.hold("a table", 10)
    with pytest.raises(ValueError, match="more than the 10 bytes charged"):
        ledger.release(11)
    with pytest.raises(ValueError, match="non-negative number of bytes"):
        ledger.peak("a table", -1)
    with pytest.raises(ValueError, match="non-negative number of bytes or None"):
        MemoryLedger(-1, "Counting")
    with pytest.raises(ValueError, match="non-negative number of bytes or None"):
        MemoryLedger(True, "Counting")  # type: ignore[arg-type]


def test_a_ledger_without_a_budget_refuses_nothing_and_has_no_room() -> None:
    ledger = MemoryLedger(None, "Domain call")
    ledger.hold("a lot", 2**62)
    ledger.peak("more", 2**62)
    assert ledger.room is None


# Steps that charge the ledger


_KINDS = {
    "ASCII": ("a", 0x7F),
    "two-byte Latin-1": ("é", 0xFF),
    "BMP": ("ā", 0xFFFF),
    "four-byte": ("\U0001f600", 0x10FFFF),
}


def test_python_text_bytes_refuses_a_character_beyond_unicode() -> None:
    with pytest.raises(ValueError, match="cannot exceed U"):
        python_text_bytes(1, 1, widest=0x110000)


def test_coding_labels_refuses_before_it_hashes_them() -> None:
    # 10 kB cannot hold the hashes of 1,000 labels.
    values = np.asarray([f"label{index}" for index in range(1_000)], dtype=object)
    ledger = MemoryLedger(10_000, "Coding labels")
    with pytest.raises(MemoryBudgetError, match="Raise mem_budget") as refused:
        category_codes(values, ledger=ledger)
    assert refused.value.item == "coding 1000 labels"
    assert ledger.charged == 0
    # The codes and names are the same with a ledger as without one.
    codes, names = category_codes(values[::-1], ledger=MemoryLedger(None, "Coding"))
    expected_codes, expected_names = category_codes(values[::-1])
    assert names == expected_names
    assert codes.tolist() == expected_codes.tolist()


@pytest.mark.parametrize(
    ("n_cells", "width", "n_samples", "columns"),
    [
        (100_000, 1, 2, ("lane",)),
        (100_000, 30, 2, ("lane",)),
        (100_000, 30, 50_000, ("lane",)),
        (100_000, 2, 2, ("a_very_long_sample_column_name",)),
        (100_000, 60, 1_000, ("lane",)),
        (100_000, 4, 30, ("lane", "batch")),
        (200_000, 0, 8, ("gem_group",)),
        (200_000, 0, 50_000, ("gem_group",)),
    ],
)
def test_numbering_samples_gives_the_same_codes_with_a_ledger_as_without(
    n_cells, width, n_samples, columns
) -> None:
    # Labels of ``width`` characters, ``n_samples`` of them in each column,
    # or numbers, as an integer column holds them, for a width of 0.
    labels = [
        np.asarray(
            [
                f"{(index + shift) % n_samples:0{width}d}"
                if width
                else (index + shift) % n_samples
                for index in range(n_cells)
            ],
            dtype=object,
        )
        for shift in range(len(columns))
    ]

    def numbered(ledger=None):
        return sample_codes(columns, labels, n_cells, reason="x", ledger=ledger)

    ledger = MemoryLedger(None, "Numbering samples")
    codes, samples, fingerprint = numbered(ledger)
    expected = numbered()
    assert (codes.tolist(), samples, fingerprint) == (
        expected[0].tolist(),
        expected[1],
        expected[2],
    )


def test_numbering_samples_refuses_before_it_sorts_the_labels(monkeypatch) -> None:
    from cytearc.metadata import selection

    labels = np.asarray([f"lane{index % 3}" for index in range(1_000)], dtype=object)
    sorted_labels = []
    unique = np.unique

    def recorded(*args, **kwargs):
        sorted_labels.append(args[0])
        return unique(*args, **kwargs)

    monkeypatch.setattr(selection.np, "unique", recorded)
    ledger = MemoryLedger(10_000, "Numbering samples")
    with pytest.raises(MemoryBudgetError) as refused:
        sample_codes(("lane",), [labels], 1_000, reason="x", ledger=ledger)
    assert refused.value.item == "numbering the cells' labels in 'lane'"
    assert sorted_labels == []
    assert ledger.charged == 0
    codes, samples, _ = sample_codes(
        ("lane",), [labels], 1_000, reason="x", ledger=MemoryLedger(None, "x")
    )
    assert samples == [{"lane": "lane0"}, {"lane": "lane1"}, {"lane": "lane2"}]
    assert codes.tolist() == [index % 3 for index in range(1_000)]


# Reads of stored arrays


def _incompressible(root: zarr.Group, name: str, shape, chunks, dtype=np.int64):
    """An array of random values, which a codec cannot shrink."""
    rng = np.random.default_rng(len(name))
    if np.dtype(dtype).kind == "U":
        width = np.dtype(dtype).itemsize // 4
        codes = rng.integers(
            0x21, 0x7F, size=(int(np.prod(shape)), width), dtype=np.uint32
        )
        values = codes.view(f"U{width}").reshape(shape)
    else:
        values = rng.integers(0, np.iinfo(np.int32).max, size=shape).astype(dtype)
    array = root.create_array(name, shape=shape, dtype=values.dtype, chunks=chunks)
    array[...] = values
    return array, values


def test_a_whole_read_of_an_array_without_chunks_gives_its_value(tmp_path) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    root.create_array("count", data=np.asarray(7, dtype=np.int64))
    with pytest.raises(MemoryBudgetError, match="for the count"):
        read_whole(MemoryLedger(0, "Reading"), root, "count", item="the count")
    value = read_whole(MemoryLedger(None, "Reading"), root, "count", item="the count")
    assert value.shape == () and int(value) == 7


def test_a_whole_read_lowers_its_concurrency_to_the_room_left(tmp_path) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array, values = _incompressible(root, "values", (1_000_000,), (100_000,))
    geometry = array_geometry(array)
    one, ten = geometry.wholeReadBytes(1), geometry.wholeReadBytes(10)
    seen = []
    getitem = zarr.Array.__getitem__

    def read(self, selection):
        seen.append(zarr.config.get("async.concurrency"))
        return getitem(self, selection)

    # A budget that holds what a read decodes at one, ten or five chunks at
    # once reads them that many at once, and gives the same values. Five at
    # a time also hold the five chunks decoded before them, which ten at a
    # time, all the chunks, do not.
    five = geometry.wholeReadBytes(5)
    assert one < five < ten
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(zarr.Array, "__getitem__", read)
        for budget, expected in ((one, 1), (ten, 10), (five, 5), (five - 1, 4)):
            ledger = MemoryLedger(budget, "Reading")
            with zarr_io_concurrency(10):
                result = read_whole(ledger, root, "values", item="the values")
            assert np.array_equal(result, values)
            assert seen.pop() == expected
        with pytest.raises(MemoryBudgetError, match="for the values"):
            read_whole(
                MemoryLedger(1_000, "Reading"), root, "values", item="the values"
            )


@pytest.mark.parametrize("n_rows", [48, 1_000, 60_000, 250_000])
@pytest.mark.parametrize(
    ("shape", "chunks", "dtype"),
    [
        ((300_000,), (100_000,), np.int64),
        ((300_000,), (2_000,), np.int32),
        ((300_000, 4), (100_000, 4), np.int32),
    ],
    ids=["int64", "small-chunks", "two-axes"],
)
def test_a_row_read_gives_its_rows(tmp_path, n_rows, shape, chunks, dtype) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array, values = _incompressible(root, "values", shape, chunks, dtype)
    rows = np.sort(np.random.default_rng(2).choice(shape[0], n_rows, replace=False))
    result = read_rows(MemoryLedger(None, "Reading"), array, rows, item="the rows")
    assert np.array_equal(result, values[rows])


@pytest.mark.parametrize("kind", ["ASCII", "four-byte"])
def test_a_text_row_read_gives_its_rows_as_python_strings(tmp_path, kind) -> None:
    lead, _widest = _KINDS[kind]
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    stored = np.asarray([lead + f"{index:023d}" for index in range(200_000)])
    array = root.create_array("ids", data=stored, chunks=(2_000,))
    rows = np.arange(0, 200_000, 2)
    result = read_text_rows(MemoryLedger(None, "Reading"), array, rows, item="the IDs")
    assert result.dtype == object and result.tolist() == stored[rows].tolist()


def test_a_text_row_read_refuses_values_that_are_not_text(tmp_path) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array = root.create_array("codes", data=np.arange(5), chunks=(5,))
    with pytest.raises(TypeError, match="the codes must be stored as text"):
        read_text_rows(MemoryLedger(None, "x"), array, np.arange(2), item="the codes")


def test_a_sharded_array_reads_whole(tmp_path) -> None:
    # Shards of eight chunks of 400 KB that do not compress, read a band at a
    # time.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    values = np.random.default_rng(3).integers(0, 2**62, 1_600_000)
    array = root.create_array(
        "sharded",
        shape=values.shape,
        dtype=values.dtype,
        chunks=(50_000,),
        shards=(400_000,),
    )
    array[:] = values
    read = read_whole(MemoryLedger(None, "Reading"), root, "sharded", item="x")
    assert np.array_equal(read, values)


def _selection_store(path: Path, flags: np.ndarray):
    """A cell table of IDs and a cell selection of the cells that ``flags`` marks."""
    root = zarr.open_group(path, mode="w")
    n_cells = len(flags)
    ids = np.asarray([f"cell-{index}" for index in range(n_cells)])
    root.create_group("cellData").create_array(
        "ids", data=ids, chunks=(min(n_cells, 100_000),)
    )
    ref = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=flags,
        row_ids=ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]
    return root, ref


@pytest.mark.parametrize("every", [1, 3])
@pytest.mark.parametrize("n_cells", [48, 250_000])
def test_reading_a_cell_selection_gives_its_cells_positions(
    tmp_path, n_cells, every
) -> None:
    from cytearc.storage.validation_scope import validation_scope

    flags = np.arange(n_cells) % every == 0
    root, ref = _selection_store(tmp_path / "store.zarr", flags)
    # The selection's identity check runs once in the operation's scope, when
    # the read is planned.
    with validation_scope():
        planned = plan_cell_selection_read(root, ref)
        assert planned.cells == int(flags.sum())
        positions = read_cell_selection(
            MemoryLedger(None, "Reading"), root, planned, item="the cells"
        )
    assert positions.tolist() == np.flatnonzero(flags).tolist()


def test_a_selection_read_planned_with_its_cells_known_skips_the_check(
    tmp_path,
) -> None:
    # A caller that knows the cells skips the check when the read is planned,
    # so the check reads the flags within the read.
    flags = np.arange(250_000) % 3 == 0
    root, ref = _selection_store(tmp_path / "store.zarr", flags)
    checked = plan_cell_selection_read(root, ref)
    unchecked = plan_cell_selection_read(root, ref, cells=int(flags.sum()))
    assert unchecked.cells == checked.cells
    for plan in (checked, unchecked):
        positions = read_cell_selection(
            MemoryLedger(None, "Reading"), root, plan, item="the cells"
        )
        assert positions.tolist() == np.flatnonzero(flags).tolist()


def test_a_cell_selection_beyond_the_budget_is_refused_before_it_is_read(
    tmp_path, monkeypatch
) -> None:
    import cytearc.metadata.selection as selection

    root, ref = _selection_store(tmp_path / "store.zarr", np.ones(1_000, dtype=bool))
    planned = plan_cell_selection_read(root, ref)

    def unread(*_arguments, **_options):
        raise AssertionError("the selection was read")

    # 5 kB cannot hold the positions of 1,000 cells.
    monkeypatch.setattr(selection, "cell_selection_indices", unread)
    with pytest.raises(MemoryBudgetError, match="for the cells, .* Raise mem_budget"):
        read_cell_selection(
            MemoryLedger(5_000, "Reading"), root, planned, item="the cells"
        )


@pytest.fixture(scope="module")
def labelled_store(tmp_path_factory):
    """A store of 50,000 cells with labels of 30 clusters, frozen for every cell."""
    from tests.storage_helpers import open_store, write_count_store

    path = tmp_path_factory.mktemp("labelled") / "store.zarr"
    n_cells = 50_000
    write_count_store(str(path), {"RNA": np.ones((n_cells, 1), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    store.cells.insert("half", np.arange(n_cells) % 2 == 0)
    store.cells.insert(
        "cluster", np.asarray([f"cluster_{i % 30:03d}" for i in range(n_cells)])
    )
    every = store.snapshot_cell_selection("every_cell")
    return store, store.snapshot_cluster_labels("cluster", cell_selection=every)


@pytest.mark.parametrize("selection", ["every_cell", "half"])
def test_reading_a_label_artifact_gives_the_cells_of_its_selection(
    labelled_store, selection
) -> None:
    from cytearc.metadata.selection import resolve_cell_aligned_artifact

    # 50,000 labelled cells, read for each of them or for half.
    store, labels = labelled_store
    cells = (
        None if selection == "every_cell" else store.snapshot_cell_selection(selection)
    )
    resolved = resolve_cell_aligned_artifact(
        store.zw, labels, cell_selection=cells, ledger=MemoryLedger(None, "Reading")
    )
    assert len(resolved.cell_idx) == (50_000 if cells is None else 25_000)


@pytest.fixture(scope="module")
def sparse_labels(tmp_path_factory):
    """A store of 500,000 cells, with labels frozen for every cell, for the first 1,000 and for the first 100."""
    from tests.storage_helpers import open_store, write_count_store

    path = tmp_path_factory.mktemp("sparse_labels") / "store.zarr"
    n_cells = 500_000
    write_count_store(str(path), {"RNA": np.ones((n_cells, 1), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    store.cells.insert("first", np.arange(n_cells) < 100)
    store.cells.insert("first_thousand", np.arange(n_cells) < 1_000)
    store.cells.insert(
        "cluster", np.asarray([f"cluster_{i % 30:03d}" for i in range(n_cells)])
    )
    first = store.snapshot_cell_selection("first")
    labels = {
        selection: store.snapshot_cluster_labels(
            "cluster", cell_selection=store.snapshot_cell_selection(selection)
        )
        for selection in ("every_cell", "first_thousand")
    }
    labels["first"] = store.snapshot_cluster_labels("cluster", cell_selection=first)
    return store, first, labels


@pytest.mark.parametrize(
    ("source", "requested"),
    [("first", False), ("every_cell", True), ("first_thousand", True)],
    ids=[
        "labels of a few cells",
        "a few cells of every cell",
        "a few cells of a thousand",
    ],
)
def test_a_read_of_a_few_cells_of_many_gives_those_cells(
    sparse_labels, source, requested
) -> None:
    from cytearc.metadata.selection import resolve_cell_aligned_artifact

    # 100 of 500,000 cells, from labels of those cells, of every cell or of
    # the first thousand.
    store, first, labels = sparse_labels
    resolved = resolve_cell_aligned_artifact(
        store.zw,
        labels[source],
        cell_selection=first if requested else None,
        ledger=MemoryLedger(None, "Reading labels"),
    )
    assert len(resolved.cell_idx) == 100
