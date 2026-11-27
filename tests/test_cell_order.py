"""The cell order of shuffled imports: the permutation, its record, and sourceRow.

The permutation is checked against a SplitMix64 written here in Python
integers, independently of NumPy, and against an order frozen for seed 0, so
a change of algorithm fails a test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.storage.cell_order import (
    destination_rows,
    CELL_ORDER_ATTR,
    SHUFFLE_BYTES_PER_CELL,
    SOURCE_ROW,
    SOURCE_ROW_ROLE,
    check_shuffle_fits,
    recorded_shuffle_seed,
    shuffle_seed_argument,
    shuffled_order,
    source_order_positions,
    source_order_ranks,
    source_order_rows,
    write_shuffled_cell_order,
)
from cytearc.utils.memory_budget import MemoryBudgetError
from tests.storage_helpers import by_cell_id, open_store
from tests.memory_helpers import held_within_charge

_MASK = (1 << 64) - 1


def _mix(value: int) -> int:
    """SplitMix64's finalizer in Python integers."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return value ^ (value >> 31)


def _reference_order(n_rows: int, seed: int) -> list[int]:
    """Rank the rows by their keyed hashes, ties by index, without NumPy."""
    key = _mix(seed ^ int.from_bytes(b"CellOrdr", "big"))
    hashes = [
        _mix(key ^ _mix((row + 0x9E3779B97F4A7C15) & _MASK)) for row in range(n_rows)
    ]
    return sorted(range(n_rows), key=lambda row: (hashes[row], row))


def test_the_order_of_seed_zero_is_frozen() -> None:
    assert shuffled_order(10, 0).tolist() == [6, 8, 0, 5, 2, 4, 7, 9, 3, 1]
    assert shuffled_order(10, 1).tolist() == [1, 8, 0, 7, 6, 3, 2, 4, 9, 5]


@pytest.mark.parametrize(
    ("n_rows", "seed"), [(0, 0), (1, 0), (2, 5), (997, 3), (1000, 2**64 - 1)]
)
def test_the_order_is_a_permutation_equal_to_the_reference(n_rows, seed) -> None:
    order = shuffled_order(n_rows, seed)
    assert order.dtype == np.int64
    assert order.shape == (n_rows,)
    np.testing.assert_array_equal(np.sort(order), np.arange(n_rows))
    assert order.tolist() == _reference_order(n_rows, seed)


@pytest.mark.parametrize("trial", range(6))
def test_the_order_is_the_stable_sort_of_distinct_hashes(trial) -> None:
    from cytearc.utils.digest import mix64_array

    # The hashes of the rows are distinct, so the sort that ranks them needs
    # no rule for ties and gives the stable sort's order.
    rng = np.random.default_rng(trial)
    n_rows = int(rng.integers(0, 300_000))
    seed = int(rng.integers(0, 2**63)) * 2 + int(rng.integers(0, 2))
    key = mix64_array(
        np.array([seed], dtype=np.uint64)
        ^ np.uint64(int.from_bytes(b"CellOrdr", "big"))
    )
    hashes = mix64_array(
        key
        ^ mix64_array(
            np.arange(n_rows, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        )
    )
    assert np.unique(hashes).size == n_rows
    np.testing.assert_array_equal(
        shuffled_order(n_rows, seed), np.argsort(hashes, kind="stable")
    )


def test_a_large_order_is_a_permutation_unlike_its_neighbours() -> None:
    order = shuffled_order(200_003, 0)
    np.testing.assert_array_equal(np.sort(order), np.arange(200_003))
    np.testing.assert_array_equal(order, shuffled_order(200_003, 0))
    # A shift or rotation of the rows would keep most rows beside the row
    # that followed them in the file.
    assert np.count_nonzero(np.diff(order) == 1) < 20
    assert abs(np.corrcoef(order, np.arange(order.size))[0, 1]) < 0.01


def test_seeds_give_different_orders() -> None:
    orders = [
        shuffled_order(1000, seed).tolist() for seed in (0, 1, 2, 2**32, 2**64 - 1)
    ]
    assert len({tuple(order) for order in orders}) == len(orders)
    # The same seed as a NumPy integer gives the same order.
    assert shuffled_order(1000, np.uint64(2**64 - 1)).tolist() == orders[-1]


def test_each_row_is_equally_likely_at_each_destination() -> None:
    # Over 4,000 seeds, row 0 of eight rows lands at each destination about
    # 500 times; a standard deviation is about 21.
    destinations = np.array(
        [int(np.flatnonzero(shuffled_order(8, seed) == 0)[0]) for seed in range(4000)]
    )
    counts = np.bincount(destinations, minlength=8)
    assert counts.min() > 400 and counts.max() < 600


@pytest.mark.parametrize(
    ("seed", "error", "message"),
    [
        (True, TypeError, "shuffle_seed must be an integer"),
        (1.0, TypeError, "shuffle_seed must be an integer"),
        ("1", TypeError, "shuffle_seed must be an integer"),
        (-1, ValueError, r"^shuffle_seed must be from 0 to 2\*\*64 - 1, not -1$"),
        (
            2**64,
            ValueError,
            r"^shuffle_seed must be from 0 to 2\*\*64 - 1, not 18446744073709551616$",
        ),
    ],
)
def test_a_bad_seed_is_refused(seed, error, message) -> None:
    with pytest.raises(error, match=message):
        shuffle_seed_argument(seed)
    with pytest.raises(error, match=message):
        shuffled_order(3, seed)


def test_a_bad_row_count_is_refused() -> None:
    with pytest.raises(ValueError, match="n_rows must be at least 0"):
        shuffled_order(-1, 0)
    with pytest.raises(TypeError, match="n_rows must be an integer"):
        shuffled_order(True, 0)


def _cell_table(n_rows: int) -> zarr.Group:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cells = root.create_group("cellData")
    cells.create_array("ids", data=np.array([f"c{row}" for row in range(n_rows)]))
    return cells


@pytest.mark.parametrize("n_rows", [0, 1, 2, 997])
def test_destination_rows_send_each_source_row_to_its_row_of_the_order(n_rows) -> None:
    order = shuffled_order(n_rows, 3)
    destinations = destination_rows(order)
    assert destinations.dtype == np.int64
    # Row r holds source row order[r], so source row s lands where order holds s.
    for row, source in enumerate(order.tolist()):
        assert destinations[source] == row
    np.testing.assert_array_equal(destinations, np.argsort(order))
    np.testing.assert_array_equal(
        destination_rows(order.astype(np.uint32)), destinations
    )


@pytest.mark.parametrize(
    "order", [np.zeros((2, 2), dtype=np.int64), np.array([0.0, 1.0]), [[0]]]
)
def test_destination_rows_refuse_what_is_not_one_integer_row_per_cell(order) -> None:
    with pytest.raises(ValueError, match="one integer row per cell"):
        destination_rows(order)


def test_drawing_a_shuffle_holds_no_more_than_its_charge() -> None:
    from tests.memory_helpers import traced_charge

    # 30,000 cells of 8 bytes stay below the 256 KiB from which NumPy reuses
    # the temporaries of an expression, which only some Python builds allow,
    # so the trace counts what every build holds. Three arrays of 8 bytes a
    # cell are held at the peak: the hashes as they are mixed and the two
    # steps of the mixing, and later the order, its inverse and their
    # positions. A charge without one of them is passed beyond the slack.
    n_cells = 30_000
    charge = SHUFFLE_BYTES_PER_CELL * n_cells
    order_peak = traced_charge(lambda _ledger: shuffled_order(n_cells, 5), 3).held
    assert order_peak > 2 * 8 * n_cells
    assert held_within_charge(order_peak, charge)
    # An import draws the order and then its inverse, holding the order.
    both_peak = traced_charge(
        lambda _ledger: destination_rows(shuffled_order(n_cells, 5)), 3
    ).held
    assert both_peak > 2 * 8 * n_cells
    assert held_within_charge(both_peak, charge)
    order = shuffled_order(n_cells, 5)
    inverse_peak = (
        order.nbytes + traced_charge(lambda _ledger: destination_rows(order), 3).held
    )
    assert inverse_peak > 2 * 8 * n_cells
    assert held_within_charge(inverse_peak, charge)


def test_a_shuffle_beyond_the_budget_is_refused_naming_the_budget_that_holds_it() -> (
    None
):
    # 1,000 cells at 24 bytes each beside 5,000 held bytes.
    check_shuffle_fits(1_000, 5_000, 29_000)
    with pytest.raises(MemoryBudgetError) as caught:
        check_shuffle_fits(1_000, 5_000, 28_999)
    item = "the cells' random order beside the 5000 bytes that the import holds"
    assert (caught.value.needed_bytes, caught.value.budget_bytes) == (29_000, 28_999)
    assert caught.value.item == item
    assert str(caught.value) == (
        f"Shuffling 1000 cells holds about 29000 bytes for {item}, more than the "
        "memory budget of 28999 bytes. Raise mem_budget to at least 29000 bytes."
    )
    # An import that holds nothing else names the shuffle alone.
    check_shuffle_fits(3, 0, 72)
    with pytest.raises(MemoryBudgetError) as alone:
        check_shuffle_fits(3, 0, 71)
    refusal = alone.value
    assert (refusal.needed_bytes, refusal.item) == (72, "the cells' random order")
    assert str(refusal).startswith(
        "Shuffling 3 cells holds about 72 bytes for the cells' random order, more than"
    )


def test_a_shuffle_of_no_cells_is_never_refused() -> None:
    # Drawing the order of no cells holds nothing, so what an import holds
    # beside it is refused by the import's own steps, not as a shuffle.
    check_shuffle_fits(0, 5_000, 1)
    check_shuffle_fits(0, 0, 0)
    with pytest.raises(MemoryBudgetError):
        check_shuffle_fits(1, 0, 23)


def test_the_record_and_source_rows_round_trip() -> None:
    cells = _cell_table(5)
    assert recorded_shuffle_seed(cells) is None
    write_shuffled_cell_order(
        cells, np.array([3, 0, 4, 1, 2], dtype=np.int32), 2**64 - 1
    )
    assert recorded_shuffle_seed(cells) == 2**64 - 1
    assert dict(cells.attrs) == {CELL_ORDER_ATTR: {"shuffled": True, "seed": 2**64 - 1}}
    column = cells[SOURCE_ROW]
    assert column.dtype == np.int64
    assert dict(column.attrs) == {"role": SOURCE_ROW_ROLE}
    np.testing.assert_array_equal(column[:], [3, 0, 4, 1, 2])
    with pytest.raises(ValueError, match="one integer row per cell"):
        write_shuffled_cell_order(cells, np.zeros(5, dtype=float), 0)
    with pytest.raises(ValueError, match="one integer row per cell"):
        write_shuffled_cell_order(cells, np.zeros((5, 1), dtype=int), 0)


@pytest.mark.parametrize(
    "record",
    [
        "shuffled",
        {"shuffled": True},
        {"shuffled": False, "seed": 0},
        {"shuffled": True, "seed": True},
        {"shuffled": True, "seed": -1},
        {"shuffled": True, "seed": 2**64},
        {"shuffled": True, "seed": 1.0},
        {"shuffled": True, "seed": 0, "extra": 1},
    ],
)
def test_a_malformed_record_is_refused(record) -> None:
    cells = _cell_table(2)
    cells.attrs[CELL_ORDER_ATTR] = record
    with pytest.raises(ValueError, match="cellOrder attribute is malformed"):
        recorded_shuffle_seed(cells)


def test_source_order_rows_sort_the_selected_rows_by_their_source_rows() -> None:
    cells = _cell_table(6)
    selection = np.array([True, False, True, True, True, True])
    # Without the column the rows keep their store order.
    np.testing.assert_array_equal(source_order_rows(cells, selection), [0, 2, 3, 4, 5])
    write_shuffled_cell_order(cells, np.array([5, 2, 0, 4, 1, 3]), 0)
    rows = source_order_rows(cells, selection)
    assert rows.dtype == np.int64
    # Selected source rows 5, 0, 4, 1, 3 sorted: 0, 1, 3, 4, 5.
    np.testing.assert_array_equal(rows, [2, 4, 5, 3, 0])
    np.testing.assert_array_equal(
        source_order_rows(cells, np.zeros(6, dtype=bool)), np.zeros(0)
    )


def test_source_order_rows_keep_ties_and_ignore_a_plain_column() -> None:
    cells = _cell_table(4)
    # A merge of two shuffled stores could hold equal source rows.
    cells.create_array(SOURCE_ROW, data=np.array([1, 0, 1, 0], dtype=np.int64))
    # Without the role, the column is a user's, and rows keep store order.
    np.testing.assert_array_equal(
        source_order_rows(cells, np.ones(4, bool)), [0, 1, 2, 3]
    )
    cells[SOURCE_ROW].attrs["role"] = SOURCE_ROW_ROLE
    np.testing.assert_array_equal(
        source_order_rows(cells, np.ones(4, bool)), [1, 3, 0, 2]
    )


def test_ranks_and_positions_keep_ties_in_store_order_among_many_cells() -> None:
    # 2,000 cells over ten source rows, as merges of shuffled stores could
    # give them; NumPy's unstable sorts reorder ties among this many values.
    source_rows = np.random.default_rng(7).integers(0, 10, 2_000)
    # Python's sort, keyed by the source row and then the store row.
    expected = sorted(range(source_rows.size), key=lambda row: (source_rows[row], row))
    ranks = source_order_ranks(source_rows)
    assert ranks.dtype == np.int64
    for rank, row in enumerate(expected):
        assert ranks[row] == rank
    cells = _cell_table(source_rows.size)
    write_shuffled_cell_order(cells, source_rows, 0)
    positions = source_order_positions(cells, np.arange(source_rows.size))
    assert positions is not None
    assert positions.tolist() == expected


def test_source_order_rows_refuse_a_bad_selection_or_column() -> None:
    cells = _cell_table(3)
    for selection in (np.ones(2, bool), np.ones(3, int), np.ones((3, 1), bool)):
        with pytest.raises(ValueError, match="one boolean per cell, 3 of them"):
            source_order_rows(cells, selection)
    column = cells.create_array(SOURCE_ROW, data=np.array([2.0, 0.0, 1.0]))
    column.attrs["role"] = SOURCE_ROW_ROLE
    with pytest.raises(ValueError, match="must hold one int64 row of the imported"):
        source_order_rows(cells, np.ones(3, dtype=bool))


def _shuffled_store(tmp_path: Path):
    """Import a small shuffled H5AD file and open it."""
    import anndata
    from scipy.sparse import csr_matrix

    from cytearc.readers import H5adReader
    from cytearc.writers import H5adToZarr

    counts = np.random.default_rng(0).poisson(1.0, (6, 4)).astype(np.float32)
    counts[:, 0] += 1
    source = tmp_path / "cells.h5ad"
    anndata.AnnData(
        csr_matrix(counts),
        obs=pd.DataFrame({"flag": [True] * 6}, index=[f"c{row}" for row in range(6)]),
        var=pd.DataFrame(index=[f"g{column}" for column in range(4)]),
    ).write_h5ad(source)
    reader = H5adReader(str(source))
    try:
        H5adToZarr(
            reader, str(tmp_path / "cells.zarr"), nthreads=1, shuffle_cells=True
        ).dump()
    finally:
        reader.close()
    return open_store(tmp_path / "cells.zarr")


def test_source_row_cannot_be_inserted_restricted_reset_or_dropped(tmp_path) -> None:
    store = _shuffled_store(tmp_path)
    cells = store.cells
    before = cells.fetch_all(SOURCE_ROW)
    np.testing.assert_array_equal(np.sort(before), np.arange(6))
    reserved = "Cell column 'sourceRow' is reserved for each cell's row"
    with pytest.raises(ValueError, match=reserved):
        cells.insert(SOURCE_ROW, by_cell_id(cells, np.arange(6)), overwrite=True)
    with pytest.raises(ValueError, match=reserved):
        cells.update_key(by_cell_id(cells, np.ones(6, dtype=bool)), SOURCE_ROW)
    with pytest.raises(ValueError, match=reserved):
        cells.reset_key(SOURCE_ROW)
    with pytest.raises(ValueError, match="cannot be dropped: exports sort by it"):
        cells.drop(SOURCE_ROW)
    np.testing.assert_array_equal(cells.fetch_all(SOURCE_ROW), before)
    # Other columns still change.
    cells.drop("flag")
    assert "flag" not in cells.columns


def test_a_plain_source_row_column_may_be_dropped_but_not_written(tmp_path) -> None:
    store = _shuffled_store(tmp_path)
    group = zarr.open_group(str(tmp_path / "cells.zarr"), mode="r+")["cellData"]
    # A store written before imports shuffled could hold a user's column.
    group[SOURCE_ROW].attrs.put({})
    store = open_store(tmp_path / "cells.zarr")
    store.cells.drop(SOURCE_ROW)
    assert SOURCE_ROW not in store.cells.columns
    with pytest.raises(ValueError, match="is reserved for each cell's row"):
        store.cells.insert(SOURCE_ROW, by_cell_id(store.cells, np.arange(6)))
