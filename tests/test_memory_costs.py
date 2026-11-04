"""The memory ledger's refusals and the shared costs that its charges count, each bounded from both sides."""

import dataclasses
import functools
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc.datastore._operations.cell_barcodes import CellBarcodeJoin, lookup_bytes
from cytearc.metadata.artifacts import (
    cell_data_write_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from cytearc.metadata.rows import read_rows, read_text_rows, read_whole
from cytearc.metadata.selection import (
    category_codes,
    plan_cell_selection_read,
    read_cell_selection,
)
from cytearc.storage.arrays import zarr_write_buffer_bytes
from cytearc.storage.async_execution import zarr_io_concurrency
from cytearc.storage.geometry import READ_OBJECT_BYTES, array_geometry
from cytearc.storage.selections import resolve_generated_selection_artifact
from cytearc.utils.arrays import sort_categories, sort_categories_bytes
from cytearc.utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
    fixed_text_bytes,
    hash_table_bytes,
    memory_refusal,
    object_array_bytes,
    observed_charges,
    python_text_bytes,
    sort_bytes,
)
from tests.memory_helpers import (
    ZARR_READS,
    ZARR_WRITES,
    audited,
    charges,
    cost_bound,
    hold_finished_reads,
    named_budget,
    traced_peak,
)

#: The bytes of a NumPy array's object and of the call's own frames, beside
#: an array's values, which a traced allocation of an array holds too.
ARRAY_HEADER = 8 * 1024


def _planned(read) -> int:
    """The most bytes that a ledger without a budget plans while ``read(ledger)`` runs."""
    plans: list[int] = []
    with observed_charges(lambda _item, _nbytes, planned: plans.append(planned)):
        read(MemoryLedger(None, "Planning"))
    return max(plans)


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


def test_what_a_read_leaves_behind_counts_until_the_next_read_replaces_it() -> None:
    ledger = MemoryLedger(1_000, "Reading")
    ledger.hold("the slots", 300)
    ledger.leave(500)
    assert ledger.held_bytes == 800 and ledger.room == 200
    with pytest.raises(MemoryBudgetError, match="holds about 1001 bytes"):
        ledger.peak("the codes", 201)
    # The next read's leftover replaces this one's; releases leave it be.
    ledger.leave(100)
    assert ledger.held_bytes == 400
    ledger.release(300)
    assert (ledger.charged, ledger.left, ledger.room) == (0, 100, 900)


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


def test_observed_charges_report_each_item_its_bytes_and_the_plan() -> None:
    def run() -> None:
        ledger = MemoryLedger(None, "Observing")
        ledger.hold("the slots", 30)
        ledger.peak("a sort", 7)
        ledger.leave(5)
        ledger.release(30)

    _result, taken = charges(run)
    # What a read leaves behind counts from the next charge on.
    assert taken == [("the slots", 30, 30), ("a sort", 7, 37), ("a release", 30, 5)]


def test_an_audit_names_the_charge_whose_interval_held_more_than_it_planned() -> None:
    def run(extra: int) -> None:
        ledger = MemoryLedger(None, "Auditing")
        ledger.peak("a small sort", 1_000_000)
        values = np.ones(1_000_000 + extra, dtype=np.uint8)
        del values
        ledger.hold("the result", 100)

    run(0)
    _result, overruns = audited(lambda: run(0))
    assert overruns == []
    _result, overruns = audited(lambda: run(200_000))
    assert [item for item, _planned, _traced in overruns] == ["a small sort"]
    ((_item, planned, traced),) = overruns
    assert planned == 1_000_000 and traced >= 1_200_000


# Shared costs


_KINDS = {
    "ASCII": ("a", 0x7F),
    "two-byte Latin-1": ("é", 0xFF),
    "BMP": ("ā", 0xFFFF),
    "four-byte": ("\U0001f600", 0x10FFFF),
}


@pytest.mark.parametrize("width", [3, 24, 200])
@pytest.mark.parametrize("kind", list(_KINDS))
def test_python_strings_cost_their_header_and_their_widest_character(
    kind, width
) -> None:
    lead, widest = _KINDS[kind]
    # Strings of exactly ``width`` characters, one of them the widest and the
    # rest digits; each value becomes a string of its own.
    stored = np.asarray(
        [lead + f"{index:0{width - 1}d}"[-(width - 1) :] for index in range(20_000)]
    )
    stored.astype(object)
    estimate = object_array_bytes(len(stored)) + python_text_bytes(
        len(stored), len(stored) * width, widest=widest
    )
    cost_bound(
        estimate + ARRAY_HEADER,
        lambda: stored.astype(object),
        slack=1.02,
        allowance=ARRAY_HEADER,
    )


def test_strings_are_counted_at_the_kind_of_the_widest_of_them() -> None:
    assert python_text_bytes(10, 50) == 10 * 41 + 50
    assert python_text_bytes(10, 50, widest=0xE9) == 10 * 57 + 50
    assert python_text_bytes(10, 50, widest=0x101) == 10 * 58 + 100
    assert python_text_bytes(10, 50, widest=0x1F600) == 10 * 60 + 200
    with pytest.raises(ValueError, match="cannot exceed U"):
        python_text_bytes(1, 1, widest=0x110000)


@pytest.mark.parametrize("width", [1, 12, 80])
def test_fixed_width_text_takes_four_bytes_a_character_of_the_widest(width) -> None:
    values = [f"{index:0{width}d}"[:width] for index in range(50_000)]
    estimate = fixed_text_bytes(len(values), width)
    cost_bound(
        estimate + ARRAY_HEADER,
        lambda: np.asarray(values, dtype=f"U{width}"),
        slack=1.02,
        allowance=ARRAY_HEADER,
    )


def test_object_arrays_take_a_reference_per_value() -> None:
    cost_bound(
        object_array_bytes(100_000) + ARRAY_HEADER,
        lambda: np.empty(100_000, dtype=object),
        slack=1.02,
        allowance=ARRAY_HEADER,
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint8])
@pytest.mark.parametrize(
    ("stable", "order"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["sort", "stable sort", "argsort", "stable argsort"],
)
def test_sorts_hold_their_copy_and_their_merge_buffers(dtype, stable, order) -> None:
    values = np.random.default_rng(0).integers(0, 200, 1_000_000).astype(dtype)
    kind = "stable" if stable else None
    sort = np.argsort if order else np.sort
    sort(values, kind=kind)
    estimate = sort_bytes(len(values), values.itemsize, stable=stable, order=order)
    traced = estimate
    if stable and not _sort_buffers_traced():
        # This NumPy takes a stable sort's merge or radix buffer from the C
        # allocator, which tracemalloc does not see, so only the copy or the
        # order that the sort returns is traced.
        traced = sort_bytes(len(values), values.itemsize, order=order)
        assert estimate > traced
    cost_bound(traced + ARRAY_HEADER, lambda: sort(values, kind=kind), slack=1.3)


@functools.cache
def _sort_buffers_traced() -> bool:
    """Whether tracemalloc sees the buffer of a stable sort, as it does in NumPy 2.5 and not in NumPy 2.1."""
    values = np.random.default_rng(0).integers(0, 200, 100_000).astype(np.int64)
    _sorted, peak = traced_peak(
        lambda: np.sort(values, kind="stable"), 2 * values.nbytes
    )
    return peak > 1.25 * values.nbytes


@pytest.mark.parametrize(
    "n_values",
    [1_000, 100_000, 1_000_000, pytest.param(3_000_000, marks=pytest.mark.slow)],
)
def test_hash_tables_are_sized_for_every_value_hashed(n_values) -> None:
    # pandas sizes the table for up to about a million values whether or not
    # they repeat; beside it, factorize returns a code for each value.
    values = np.random.default_rng(1).integers(0, 50, n_values).astype(np.int64)
    pd.factorize(values)
    estimate = hash_table_bytes(n_values, n_distinct=50) + 8 * n_values
    cost_bound(estimate + n_values // 2 + ARRAY_HEADER, lambda: pd.factorize(values))


@pytest.mark.parametrize(
    "n_values", [100_000, 1_000_000, pytest.param(3_000_000, marks=pytest.mark.slow)]
)
def test_an_index_of_distinct_values_doubles_its_table_past_a_million(
    n_values,
) -> None:
    # An index's lookup hashes each of its distinct values into a table that
    # doubles past about a million of them, and a probe's positions take 8
    # bytes each. pandas 3 copies the values into the index, and pandas 2
    # keeps the array it is given.
    values = np.random.default_rng(4).permutation(10 * n_values)[:n_values]
    probe = values[:10_000]

    def lookup() -> np.ndarray:
        return pd.Index(values).get_indexer(probe)

    lookup()
    copied = not np.shares_memory(pd.Index(values).to_numpy(), values)
    estimate = hash_table_bytes(n_values) + 8 * len(probe) + copied * 8 * n_values
    cost_bound(estimate + n_values // 2 + 64 * 1024, lookup)


def test_a_hash_table_counts_each_doubling_past_its_first_size() -> None:
    # A million values fill 2**21 buckets of 16 bytes and a bit at most 77%
    # full, and
    # pandas sizes no table larger before it hashes; three million distinct
    # values double it once.
    def table(buckets: int) -> int:
        return 16 * buckets + buckets // 8

    assert hash_table_bytes(1_000) == table(2**11)
    assert hash_table_bytes(100_000) == table(2**17)
    assert hash_table_bytes(1_000_000) == table(2**21)
    assert hash_table_bytes(3_000_000, n_distinct=50) == table(2**21)
    assert hash_table_bytes(3_000_000) == table(2**22)
    assert hash_table_bytes(10) == table(16)


def test_text_is_hashed_as_the_strings_it_becomes() -> None:
    # Fixed-width text becomes an object array of strings before pandas
    # hashes it, beside the table, the codes and an object's slot each.
    labels = np.asarray([f"label{code:02d}" for code in range(50)])
    values = labels[np.random.default_rng(2).integers(0, 50, 200_000)]
    pd.factorize(values)
    estimate = (
        object_array_bytes(len(values))
        + python_text_bytes(len(values), 7 * len(values))
        + hash_table_bytes(len(values), n_distinct=50)
        + 16 * len(values)
    )
    cost_bound(estimate + 64 * 1024, lambda: pd.factorize(values), slack=1.3)


# Labels of one to eight tokens of the natural order, the runs of digits and
# of other characters, in ASCII and beyond it.
_LABELS = {
    "ASCII of two tokens": lambda index: f"donor_{index:06d}",
    "ASCII of one token": lambda index: "".join(
        chr(97 + (index // 26**place) % 26) for place in range(5)
    ),
    "ASCII of eight tokens": lambda index: (
        f"a{index % 7}b{index % 11}c{index % 13}d{index}"
    ),
    "two-byte": lambda index: "\u0100" * 10 + f"{index:05d}",
    "four-byte": lambda index: "\U0001f600" * 40 + f"{index:06d}",
}


def _fresh_labels(kind: str, n_labels: int) -> list[str]:
    """Labels made anew, so that no cache that an earlier call left in them counts."""
    return [_LABELS[kind](index) for index in range(n_labels)]


@pytest.mark.parametrize("kind", list(_LABELS))
def test_sorting_labels_naturally_holds_their_keys(kind) -> None:
    # Each label's sort key holds a tuple of its tokens and a casefolded copy
    # of its text; Python reuses up to 2,000 freed tuples of each size, so
    # 20,000 labels keep that out of the count.
    sort_categories(_fresh_labels(kind, 10))
    labels = _fresh_labels(kind, 20_000)
    cost_bound(
        sort_categories_bytes(labels),
        lambda: sort_categories(labels),
        slack=1.3,
        allowance=64 * 1024,
    )


def test_the_natural_sort_counts_each_token_and_the_widest_character() -> None:
    import sys

    # By hand: a label's key and the tuple of its tokens beside 6 bytes, a
    # token's tuple and its slot beside 26, at this Python's sizes of tuples,
    # 110 and 90 bytes on Python 3.13 and 8 more a tuple since 3.14, and the
    # text at its kind.
    label = sys.getsizeof((0, (), "")) + sys.getsizeof(()) + 6
    token = sys.getsizeof((0, 0)) + 8 + 26
    tuple_growth = 8 if sys.version_info >= (3, 14) else 0
    assert (label, token) == (110 + 2 * tuple_growth, 90 + tuple_growth)
    assert sort_categories_bytes(["g12", "s2_rep10"]) == (
        2 * label + (2 + 4) * token + python_text_bytes(2, 11)
    )
    assert sort_categories_bytes(["\U0001f600x"]) == (
        label + token + python_text_bytes(1, 2, widest=0x1F600)
    )
    assert sort_categories_bytes([]) == 0


def _charged(call) -> int:
    """The most bytes that the charges of ``call`` planned on a ledger without a budget."""
    _result, taken = charges(call)
    return max(planned for _item, _nbytes, planned in taken)


@pytest.mark.parametrize("kind", ["ASCII of two tokens", "four-byte"])
@pytest.mark.parametrize(
    ("n_values", "n_distinct"),
    [
        (20_000, 20_000),
        (200_000, 50),
        pytest.param(200_000, 200_000, marks=pytest.mark.slow),
    ],
)
def test_coding_labels_holds_what_its_charges_plan(kind, n_values, n_distinct) -> None:
    # The values of each call are made anew, since pandas keeps in each
    # string beyond ASCII that it hashes the string's UTF-8 form.
    def values() -> np.ndarray:
        labels = np.asarray(_fresh_labels(kind, n_distinct), dtype=object)
        return np.asarray(labels[np.arange(n_values) % n_distinct], dtype=object)

    category_codes(values()[:10])
    planned = _charged(
        lambda: category_codes(values(), ledger=MemoryLedger(None, "Coding labels"))
    )
    given = iter([values(), values()])
    cost_bound(
        planned, lambda: category_codes(next(given)), slack=1.3, allowance=64 * 1024
    )


def test_coding_labels_refuses_before_it_hashes_them() -> None:
    values = np.asarray([f"label{index}" for index in range(1_000)], dtype=object)
    factorized = hash_table_bytes(1_000) + 16 * 1_000 + object_array_bytes(1_000)
    ledger = MemoryLedger(factorized - 1, "Coding labels")
    with pytest.raises(MemoryBudgetError) as refused:
        category_codes(values, ledger=ledger)
    assert refused.value.item == "coding 1000 labels"
    assert refused.value.needed_bytes == factorized
    # Numbers gather no references to their distinct values.
    numbers = MemoryLedger(None, "Coding labels")
    _result, taken = charges(lambda: category_codes(np.arange(1_000), ledger=numbers))
    assert taken[0] == (
        "coding 1000 labels",
        hash_table_bytes(1_000) + 16 * 1_000,
        hash_table_bytes(1_000) + 16 * 1_000,
    )
    # The codes and names are the same with a ledger as without one.
    codes, names = category_codes(values[::-1], ledger=MemoryLedger(None, "Coding"))
    expected_codes, expected_names = category_codes(values[::-1])
    assert names == expected_names
    assert codes.tolist() == expected_codes.tolist()


@pytest.mark.parametrize("n_cells", [20_000, 200_000])
def test_the_joins_lookup_holds_an_index_and_a_hash_table(n_cells) -> None:
    # pandas hashes the IDs into a table on its first use, and pandas 3
    # copies their references into the index, beside a flag for each cell;
    # the IDs' strings are the caller's.
    ids = np.asarray([f"cell{index:08d}-1" for index in range(n_cells)], dtype=object)
    probe = np.asarray(ids[:1_000], dtype=object)

    def join() -> CellBarcodeJoin:
        joined = CellBarcodeJoin(ids, ledger=MemoryLedger(None, "Joining"))
        joined._lookup.get_indexer(pd.Index(probe, dtype=object))
        return joined

    join()
    # Beside the probe's index and positions, the index, its engine and the
    # join are objects of a few kilobytes.
    cost_bound(
        lookup_bytes(n_cells) + 16 * len(probe) + 16 * 1024,
        join,
        slack=1.3,
        allowance=64 * 1024,
    )


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


def _whole_read(root, name):
    return lambda ledger: read_whole(ledger, root, name, item="the array")


@pytest.mark.parametrize(
    ("shape", "chunks", "concurrency"),
    [
        ((400_000,), (100_000,), 1),
        ((400_000,), (4_000,), 1),
        ((400_000,), (4_000,), 10),
        ((100_000, 4), (2_500, 4), 1),
        ((100_000, 4), (500, 4), 10),
    ],
    # At Zarr's default concurrency, how many chunks the pool decodes at once
    # depends on the machine, so those reads have chunks small beside their
    # result, whose bound holds however few decode at once.
    ids=[
        "four-chunks",
        "a-hundred-chunks",
        "a-hundred-chunks-default",
        "two-axes",
        "two-axes-default",
    ],
)
def test_a_whole_read_holds_what_its_concurrency_decodes(
    tmp_path, shape, chunks, concurrency
) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    _incompressible(root, "values", shape, chunks)
    read = _whole_read(root, "values")
    with zarr_io_concurrency(concurrency):
        read(MemoryLedger(None, "Warming"))
        estimate = _planned(read)
        _result, peak = cost_bound(
            estimate, lambda: read(MemoryLedger(None, "Reading")), slack=1.3
        )
    assert peak > 8 * int(np.prod(shape))


def test_a_whole_read_of_an_array_without_chunks_holds_its_value(tmp_path) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    root.create_array("count", data=np.asarray(7, dtype=np.int64))
    ledger = MemoryLedger(0, "Reading")
    with pytest.raises(MemoryBudgetError, match="holds about 8 bytes for the count"):
        read_whole(ledger, root, "count", item="the count")
    ledger = MemoryLedger(8, "Reading")
    value = read_whole(ledger, root, "count", item="the count")
    assert value.shape == () and int(value) == 7
    assert (ledger.charged, ledger.left) == (8, 0)


def test_a_whole_read_lowers_its_concurrency_to_the_room_left(tmp_path) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array, values = _incompressible(root, "values", (1_000_000,), (100_000,))
    geometry = array_geometry(array)
    one, ten = geometry.wholeReadBytes(1), geometry.wholeReadBytes(10)
    assert ten - one >= 9 * 2 * 800_000
    seen = []
    getitem = zarr.Array.__getitem__

    def read(self, selection):
        seen.append(zarr.config.get("async.concurrency"))
        return getitem(self, selection)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(zarr.Array, "__getitem__", read)
        for budget, expected in ((one, 1), (ten, 10), (one + (ten - one) // 2, 5)):
            ledger = MemoryLedger(budget, "Reading")
            with zarr_io_concurrency(10):
                result, taken = charges(
                    lambda ledger=ledger: read_whole(
                        ledger, root, "values", item="the values"
                    )
                )
            assert np.array_equal(result, values)
            assert seen.pop() == expected
            # The read is charged what it holds at the concurrency it runs at.
            peak = geometry.wholeReadBytes(expected)
            assert taken[0] == ("the values", peak, peak)
            assert ledger.charged == values.nbytes
            # The ledger's one codec thread can still hold one chunk, whatever
            # the concurrency.
            assert ledger.left == geometry.heldReadBytes(1)
        with pytest.raises(MemoryBudgetError, match=f"holds about {one} bytes"):
            read_whole(MemoryLedger(one - 1, "Reading"), root, "values", item="x")


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
def test_a_row_read_holds_its_rows_their_indices_and_a_band_of_chunks(
    tmp_path, n_rows, shape, chunks, dtype
) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array, values = _incompressible(root, "values", shape, chunks, dtype)
    rows = np.sort(np.random.default_rng(2).choice(shape[0], n_rows, replace=False))

    def read(ledger):
        return read_rows(ledger, array, rows, item="the rows")

    read(MemoryLedger(None, "Warming"))
    estimate = _planned(read)
    # Each read's work stays held until the next ends, as on a busy machine.
    # A read of a few rows traces about half of the READ_OBJECT_BYTES that its
    # cost counts for Zarr's objects.
    with hold_finished_reads(*ZARR_READS):
        result, _peak = cost_bound(
            estimate,
            lambda: read(MemoryLedger(None, "Reading")),
            slack=1.4,
            allowance=READ_OBJECT_BYTES // 2,
        )
    assert np.array_equal(result, values[rows])


@pytest.mark.parametrize("kind", ["ASCII", "four-byte"])
def test_a_text_row_read_counts_its_strings_by_their_widest_character(
    tmp_path, kind
) -> None:
    lead, _widest = _KINDS[kind]
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    stored = np.asarray([lead + f"{index:023d}" for index in range(200_000)])
    array = root.create_array("ids", data=stored, chunks=(2_000,))
    rows = np.arange(0, 200_000, 2)

    def read(ledger):
        return read_text_rows(ledger, array, rows, item="the IDs")

    read(MemoryLedger(None, "Warming"))
    estimate = _planned(read)
    with hold_finished_reads(*ZARR_READS):
        result, _peak = cost_bound(
            estimate, lambda: read(MemoryLedger(None, "Reading")), slack=1.3
        )
    assert result.dtype == object and result.tolist() == stored[rows].tolist()
    ledger = MemoryLedger(None, "Reading")
    read(ledger)
    # The stored rows are let go; the strings stay charged.
    strings = object_array_bytes(len(rows)) + python_text_bytes(
        len(rows), 24 * len(rows), widest=_widest
    )
    assert ledger.charged == strings


def test_a_text_row_read_refuses_values_that_are_not_text(tmp_path) -> None:
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array = root.create_array("codes", data=np.arange(5), chunks=(5,))
    with pytest.raises(TypeError, match="the codes must be stored as text"):
        read_text_rows(MemoryLedger(None, "x"), array, np.arange(2), item="the codes")


def test_the_chunk_a_read_leaves_behind_counts_beside_the_next_read(
    tmp_path, monkeypatch
) -> None:
    # A whole read of one chunk of 2 MB that does not compress, then a small
    # read: with each read's work held until the next read ends, as a busy
    # machine can hold it, the first chunk as stored and decoded stays beside
    # the second read, which the ledger plans from what the first left.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    _incompressible(root, "first", (250_000,), (250_000,))
    _incompressible(root, "second", (1_000,), (1_000,))

    def run(ledger: MemoryLedger) -> None:
        values = read_whole(ledger, root, "first", item="the first array")
        del values
        ledger.release(8 * 250_000)
        read_whole(ledger, root, "second", item="the second array")

    run(MemoryLedger(None, "Warming"))
    with zarr_io_concurrency(1), hold_finished_reads(*ZARR_READS):
        _result, overruns = audited(lambda: run(MemoryLedger(None, "Reading")))
    assert overruns == []
    ledger = MemoryLedger(None, "Reading")
    run(ledger)
    assert ledger.left == array_geometry(root["second"]).heldReadBytes()
    # A ledger that forgot what the first read left behind plans too little
    # for the second read.
    monkeypatch.setattr(MemoryLedger, "leave", lambda self, nbytes: None)
    with zarr_io_concurrency(1), hold_finished_reads(*ZARR_READS):
        _result, overruns = audited(lambda: run(MemoryLedger(None, "Reading")))
    assert "the second array" in [item for item, _planned, _traced in overruns]


@pytest.mark.parametrize("threads", [1, 4, 16])
def test_a_read_leaves_a_chunk_on_each_codec_thread_up_to_its_chunks(
    tmp_path, threads
) -> None:
    # Ten chunks read whole, and rows of each of them, leave one chunk on each
    # codec thread, up to the ten; a read of one chunk leaves one chunk.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    many, _values = _incompressible(root, "many", (100_000,), (10_000,))
    one, _values = _incompressible(root, "one", (10_000,), (10_000,))
    rows = np.arange(0, 100_000, 5_000)
    for array, name, chunks in ((many, "many", 10), (one, "one", 1)):
        geometry = array_geometry(array)
        held = geometry.heldReadBytes(min(threads, chunks))
        ledger = MemoryLedger(None, "Reading", codec_threads=threads)
        with zarr_io_concurrency(1):
            read_whole(ledger, root, name, item="the values")
        assert ledger.left == held
        ledger = MemoryLedger(None, "Reading", codec_threads=threads)
        read_rows(ledger, array, rows[rows < array.shape[0]], item="the rows")
        assert ledger.left == held
    assert MemoryLedger(None, "Reading", codec_threads=threads).holding_threads(0) == 1


def test_a_read_is_charged_its_result_beside_the_chunks_it_leaves(tmp_path) -> None:
    # Read one chunk at a time, ten chunks of 80 KB hold one chunk while they
    # are read, but leave one on each of four codec threads beside the
    # result, which the read's charge counts before it reads.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    array, _values = _incompressible(root, "values", (100_000,), (10_000,))
    geometry = array_geometry(array)
    after = 800_000 + geometry.heldReadBytes(4)
    assert after > geometry.wholeReadBytes(1)
    with zarr_io_concurrency(1):
        _values, taken = charges(
            lambda: read_whole(
                MemoryLedger(None, "Reading", codec_threads=4),
                root,
                "values",
                item="the values",
            )
        )
    assert taken[0] == ("the values", after, after)
    with pytest.raises(MemoryBudgetError, match=f"holds about {after} bytes"):
        read_whole(
            MemoryLedger(after - 1, "Reading", codec_threads=4),
            root,
            "values",
            item="the values",
        )


def test_a_read_one_chunk_at_a_time_leaves_its_chunks_on_the_codec_threads(
    tmp_path,
) -> None:
    # Zarr's pool spreads the four chunks of a read one at a time over its
    # threads. With each read's work held until the next read ends, the four
    # chunks of 1.6 MB as stored and decoded stay beside the next read, which
    # a ledger of four codec threads plans; one that counts a chunk for the
    # read's one decode at a time plans too little.
    root = zarr.open_group(tmp_path / "store.zarr", mode="w")
    _incompressible(root, "first", (400_000,), (100_000,))
    _incompressible(root, "second", (1_000,), (1_000,))

    def run(threads: int) -> None:
        ledger = MemoryLedger(None, "Reading", codec_threads=threads)
        values = read_whole(ledger, root, "first", item="the first array")
        del values
        ledger.release(8 * 400_000)
        read_whole(ledger, root, "second", item="the second array")

    with zarr_io_concurrency(1):
        run(4)
        with hold_finished_reads(*ZARR_READS):
            _result, overruns = audited(lambda: run(4))
        assert overruns == []
        with hold_finished_reads(*ZARR_READS):
            _result, overruns = audited(lambda: run(1))
    assert "the second array" in [item for item, _planned, _traced in overruns]


def test_a_sharded_read_fetches_one_band_of_chunks_at_a_time(
    tmp_path, monkeypatch
) -> None:
    # Shards of eight chunks of 400 KB that do not compress. Read a band at
    # a time, as the ledger reads a sharded array, a read keeps one chunk's
    # fetch beside the next read; read in one call, as an unsharded array is
    # read, it keeps the whole shard that it fetched.
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
    root.create_array("next", data=np.arange(10), chunks=(10,))

    def run() -> None:
        ledger = MemoryLedger(None, "Reading")
        read = read_whole(ledger, root, "sharded", item="the counts")
        del read
        ledger.release(values.nbytes)
        read_whole(ledger, root, "next", item="the next array")

    read = read_whole(MemoryLedger(None, "Reading"), root, "sharded", item="x")
    assert np.array_equal(read, values)
    del read
    run()
    with zarr_io_concurrency(1), hold_finished_reads(*ZARR_READS):
        _result, overruns = audited(run)
    assert overruns == []
    from cytearc.metadata import rows as metadata_rows

    geometry = array_geometry(array)
    unsharded = dataclasses.replace(geometry, shards=None)
    monkeypatch.setattr(
        metadata_rows,
        "array_geometry",
        lambda stored: (
            unsharded if stored.shape == values.shape else array_geometry(stored)
        ),
    )
    with zarr_io_concurrency(1), hold_finished_reads(*ZARR_READS):
        _result, overruns = audited(run)
    assert "the next array" in [item for item, _planned, _traced in overruns]


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
def test_reading_a_cell_selection_holds_its_flags_and_its_cells_positions(
    tmp_path, n_cells, every
) -> None:
    from cytearc.storage.validation_scope import validation_scope

    flags = np.arange(n_cells) % every == 0
    root, ref = _selection_store(tmp_path / "store.zarr", flags)

    def read(ledger: MemoryLedger) -> np.ndarray:
        return read_cell_selection(ledger, root, planned, item="the cells")

    # The selection's identity check, outside the budget, runs once in the
    # operation's scope, when the read is planned; the read then holds the
    # flags a band of chunks at a time beside the cells' positions, with each
    # read's work held until the next ends, as on a busy machine. A read of a
    # few flags traces less than half of the READ_OBJECT_BYTES that its cost
    # counts for Zarr's objects.
    with validation_scope():
        planned = plan_cell_selection_read(root, ref)
        assert planned.cells == int(flags.sum())
        read(MemoryLedger(None, "Warming"))
        estimate = _planned(read)
        with hold_finished_reads(*ZARR_READS):
            positions, _peak = cost_bound(
                estimate,
                lambda: read(MemoryLedger(None, "Reading")),
                slack=1.3,
                allowance=READ_OBJECT_BYTES,
            )
        assert positions.tolist() == np.flatnonzero(flags).tolist()
        # The positions stay charged, and the flags' last chunk on the codec
        # thread counts until the next read.
        ledger = MemoryLedger(None, "Reading")
        read(ledger)
        geometry = planned.geometry
        assert geometry is not None
        assert ledger.charged == 8 * planned.cells
        assert ledger.left == geometry.heldReadBytes(1)


def test_a_selection_checked_as_its_cells_are_read_counts_the_checks_chunk(
    tmp_path,
) -> None:
    # A caller that knows the cells skips the check when the read is planned,
    # so the check reads the flags within the read, and what that first read
    # leaves on the codec thread is held beside the read of the positions.
    flags = np.arange(250_000) % 3 == 0
    root, ref = _selection_store(tmp_path / "store.zarr", flags)
    checked = plan_cell_selection_read(root, ref)
    unchecked = plan_cell_selection_read(root, ref, cells=int(flags.sum()))
    assert (checked.checked, unchecked.checked) == (True, False)
    assert unchecked.cells == checked.cells

    def planned(plan) -> int:
        return _planned(
            lambda ledger: read_cell_selection(ledger, root, plan, item="the cells")
        )

    geometry = checked.geometry
    assert geometry is not None
    assert planned(unchecked) - planned(checked) == geometry.heldReadBytes(1)


def test_a_cell_selection_beyond_the_budget_is_refused_before_it_is_read(
    tmp_path, monkeypatch
) -> None:
    import cytearc.metadata.selection as selection

    root, ref = _selection_store(tmp_path / "store.zarr", np.ones(1_000, dtype=bool))
    planned = plan_cell_selection_read(root, ref)
    estimate = _planned(
        lambda ledger: read_cell_selection(ledger, root, planned, item="the cells")
    )

    def unread(*_arguments, **_options):
        raise AssertionError("the selection was read")

    monkeypatch.setattr(selection, "cell_selection_indices", unread)
    with pytest.raises(
        MemoryBudgetError, match=f"holds about {estimate} bytes for the cells"
    ):
        read_cell_selection(
            MemoryLedger(estimate - 1, "Reading"), root, planned, item="the cells"
        )


# Writes


def _store(path: Path, n_cells: int):
    root = zarr.open_group(path, mode="w")
    cell_data = root.create_group("cellData")
    ids = np.asarray([f"cell-{index}" for index in range(n_cells)])
    cell_data.create_array("ids", data=ids, chunks=(min(n_cells, 100_000),))
    selection = np.ones(n_cells, dtype=bool)
    cell_data.create_array("I", data=selection, chunks=(min(n_cells, 100_000),))
    ref = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]
    return root, ref


def _random_text(rows: int, width: int, seed: int) -> np.ndarray:
    codes = np.random.default_rng(seed).integers(
        0x10000, 0x10FFFF, size=(rows, width), dtype=np.uint32
    )
    codes[(codes >= 0xD800) & (codes < 0xE000)] = 0x41
    return codes.view(f"U{width}").reshape(rows).astype(object)


def test_a_write_counts_what_the_array_before_it_left_behind(tmp_path) -> None:
    # Two text arrays of one chunk each, of four-byte characters that do not
    # compress: with each write's work held until the next write ends, the
    # labels' block and encoded chunk stay beside the names' write, which the
    # cost counts; the costliest array alone falls a third short.
    n_cells = 60_000
    root, selection = _store(tmp_path / "store.zarr", n_cells)
    arrays = {
        "labels": _random_text(n_cells, 40, 1),
        "names": _random_text(n_cells, 20, 2),
    }
    counter = iter(range(100))

    def write():
        planned = plan_cell_data_artifact(
            root,
            scope="datastore",
            kind="clones",
            operation="probe_writes",
            parameters={"write": next(counter)},
            inputs={},
            execution_options={},
            cell_selection=selection,
            arrays={"codes": ((n_cells,), "i")},
        )
        with hold_finished_reads(*ZARR_WRITES):
            write_cell_data_artifact(
                root,
                planned,
                {"codes": np.zeros(n_cells, dtype=np.int32), **arrays},
            )

    specs = {
        "codes": ((n_cells,), np.int32),
        "labels": ((n_cells,), "U40"),
        "names": ((n_cells,), "U20"),
    }
    stated = cell_data_write_bytes(specs)
    write()
    _result, peak = cost_bound(stated, write, slack=1.3)
    alone = max(cell_data_write_bytes({name: spec}) for name, spec in specs.items())
    assert peak > 1.3 * alone, (peak, alone)
    # The order counts: after the codes, the labels leave little behind them.
    assert (
        cell_data_write_bytes(
            {name: specs[name] for name in ("names", "codes", "labels")}
        )
        < stated - 4 * 20 * n_cells
    )


def test_numbers_written_after_text_hold_the_text_chunk_left_behind() -> None:
    # A million int64 sizes are ten chunks of 800,000 bytes that Zarr encodes
    # at once, with its objects for each write, beside the chunk of 100,000
    # labels of 40 characters, 16 MB, and its encoded copy, which the labels'
    # write left behind.
    with zarr_io_concurrency(10):
        stated = cell_data_write_bytes(
            {"labels": ((100_000,), "U40"), "sizes": ((1_000_000,), np.int64)}
        )
        fixed = 64 * 1024 + zarr_write_buffer_bytes()
    assert stated == fixed + 10 * 800_000 + 10 * 16 * 1024 + 2 * 16_000_000
