"""Cells grouped by columns, label artifacts and kept-apart labels, checked by hand."""

import sys
from pathlib import Path

import numpy as np
import pytest

from cytearc.datastore._operations.cell_groups import (
    GROUPING_BYTES_PER_GROUP,
    GROUPING_BYTES_PER_GROUP_COLUMN,
    cell_groups,
    group_columns,
    within_groups,
)
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.artifacts import plan_cell_data_artifact
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.artifact_writer import artifact_transaction
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from cytearc.storage.validation_scope import validation_scope
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.memory_helpers import (
    ZARR_READS,
    audited,
    budget_that_runs,
    charges,
    hold_finished_reads,
    traced_peak,
)
from tests.storage_helpers import (
    insert_nullable_cell_column,
    open_store,
    write_count_store,
)

N_CELLS = 8
# Natural order puts s2 before s10.
SAMPLES = np.asarray(["s2", "s10", "s2", "s10", "s2", "s10", "s2", "s10"])
BATCHES = np.asarray(["a", "a", "b", "b", "a", "a", "b", "b"])
CLUSTERS = np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"])


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> DataStore:
    path = tmp_path_factory.mktemp("cell_groups") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_CELLS, 3), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("sample", SAMPLES.astype(object))
    store.cells.insert("batch", BATCHES.astype(object))
    store.cells.insert("cluster", CLUSTERS.astype(object))
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.cells.insert("first_four", np.arange(N_CELLS) < 4)
    store.cells.insert("odd", np.arange(N_CELLS) % 2 == 1)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[[0, 5]] = True
    insert_nullable_cell_column(store, "partial", SAMPLES, missing)
    return store


def _group(store: DataStore, rows=None, *, selection: str = "every_cell", **options):
    cells = store.snapshot_cell_selection(selection)
    n_cells = int(np.count_nonzero(np.asarray(store.cells.fetch_all(selection))))
    options = {
        "group_by": None,
        "groups": None,
        "cell_selection": None,
        "ledger": MemoryLedger(None, "Grouping cells"),
        **options,
    }
    mask = np.ones(n_cells, dtype=bool) if rows is None else np.asarray(rows)
    return cell_groups(store.zw, store.cells, cells, mask, **options)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (3, TypeError, "group_by must be a column name"),
        (b"sample", TypeError, "group_by must be a column name"),
        ([3], TypeError, "group_by must be a column name"),
        ([""], TypeError, "group_by must be a column name"),
        (["a", "a"], ValueError, "group_by names a column twice"),
        (["group", "group"], ValueError, "group_by names a column twice"),
    ],
)
def test_group_columns_are_checked(value, error, message) -> None:
    with pytest.raises(error, match=message):
        group_columns(value)


def test_group_columns_may_take_the_names_of_any_tables_columns() -> None:
    # Results hold the groups' labels in a table of their own, so no name is
    # reserved for the columns of a result.
    names = ["group", "clone", "n_cells", "fraction", "locus", "fate", "label"]
    assert group_columns(names) == tuple(names)
    for name in names:
        assert group_columns(name) == (name,)


def test_group_columns_take_one_name_or_several() -> None:
    assert group_columns(None) == ()
    assert group_columns("sample") == ("sample",)
    assert group_columns(["sample", "batch"]) == ("sample", "batch")
    assert group_columns(("x",), "by") == ("x",)
    with pytest.raises(TypeError, match="^by must be"):
        group_columns(1, "by")


def test_columns_group_cells_by_their_labels_in_natural_order(store) -> None:
    grouped = _group(store, group_by=["sample", "batch"])
    assert grouped.key_columns == ("sample", "batch")
    # s2 sorts before s10, and a before b within each sample.
    assert grouped.keys == [
        {"sample": "s2", "batch": "a"},
        {"sample": "s2", "batch": "b"},
        {"sample": "s10", "batch": "a"},
        {"sample": "s10", "batch": "b"},
    ]
    assert grouped.codes.tolist() == [0, 2, 1, 3, 0, 2, 1, 3]
    assert grouped.codes.dtype == np.int64


def test_cells_outside_the_rows_or_without_a_label_have_no_group(store) -> None:
    rows = np.ones(N_CELLS, dtype=bool)
    rows[3] = False
    grouped = _group(store, rows, group_by="partial")
    # cell0 and cell5 have no label, and cell3 is not a row.
    assert grouped.codes.tolist() == [-1, 1, 0, -1, 0, -1, 0, 1]
    assert grouped.keys == [{"partial": "s2"}, {"partial": "s10"}]


def test_a_label_artifact_of_fewer_cells_leaves_the_others_ungrouped(store) -> None:
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("first_four")
    )
    grouped = _group(store, group_by="sample", groups=labels)
    assert grouped.key_columns == ("sample", "cluster_labels")
    assert grouped.keys == [
        {"sample": "s2", "cluster_labels": "x"},
        {"sample": "s10", "cluster_labels": "y"},
    ]
    assert grouped.codes.tolist() == [0, 1, 0, 1, -1, -1, -1, -1]


def test_kept_apart_labels_split_the_groups(store) -> None:
    kinds = np.asarray(["T", "T", "B", "B", "T", "B", None, "T"], dtype=object)
    grouped = _group(store, group_by="batch", kept_apart=(("receptor_type", kinds),))
    assert grouped.key_columns == ("batch", "receptor_type")
    assert grouped.keys == [
        {"batch": "a", "receptor_type": "B"},
        {"batch": "a", "receptor_type": "T"},
        {"batch": "b", "receptor_type": "B"},
        {"batch": "b", "receptor_type": "T"},
    ]
    # cell6 has no label of its own, so no group.
    assert grouped.codes.tolist() == [1, 1, 2, 2, 1, 0, -1, 3]


def test_a_cell_selection_keeps_its_cells(store) -> None:
    odd = store.snapshot_cell_selection("odd")
    grouped = _group(store, group_by="batch", cell_selection=odd)
    assert grouped.codes.tolist() == [-1, 0, -1, 1, -1, 0, -1, 1]
    # The artifact's cells are matched to the selection by their store rows.
    first = _group(store, selection="first_four", cell_selection=odd)
    assert first.codes.tolist() == [-1, 0, -1, 0]
    assert first.keys == [{}]


def test_the_grouping_counts_the_cells_it_leaves_out(store) -> None:
    rows = np.ones(N_CELLS, dtype=bool)
    rows[[2, 3]] = False
    # cell2 and cell3 are not rows; of the others, cell0 and cell5 have no
    # label; the rest fall in two groups.
    grouped = _group(store, rows, group_by="partial")
    assert grouped.diagnostics == {
        "cellsWithoutClone": 2,
        "cellsWithoutGroup": 2,
        "groups": 2,
    }
    # Within the odd cells, cell3 is no row, cell5 has no label, and cell1
    # and cell7 are both of s10.
    odd = store.snapshot_cell_selection("odd")
    within = _group(store, rows, group_by="partial", cell_selection=odd)
    assert within.diagnostics == {
        "cellsWithoutClone": 1,
        "cellsWithoutGroup": 1,
        "groups": 1,
    }
    # Without a column, every row is in the one group.
    assert _group(store, rows, cell_selection=odd).diagnostics == {
        "cellsWithoutClone": 1,
        "cellsWithoutGroup": 0,
        "groups": 1,
    }


def test_without_a_grouping_the_rows_form_one_group(store) -> None:
    rows = np.arange(N_CELLS) % 3 == 0
    grouped = _group(store, rows)
    assert grouped.key_columns == ()
    assert grouped.keys == [{}]
    assert grouped.codes.tolist() == [0, -1, -1, 0, -1, -1, 0, -1]
    # Only the combinations that a cell holds are groups.
    none = _group(store, np.zeros(N_CELLS, dtype=bool))
    assert none.keys == [] and (none.codes == -1).all()
    keyed = _group(store, np.zeros(N_CELLS, dtype=bool), group_by="sample")
    assert keyed.keys == [] and (keyed.codes == -1).all()


def test_key_columns_named_like_a_results_columns_group_as_any_other(store) -> None:
    # Columns named like the columns of the summaries' tables hold the same
    # labels as the sample and batch, and group the cells as those do.
    store.cells.insert("group", SAMPLES.astype(object), overwrite=True)
    store.cells.insert("n_cells", BATCHES.astype(object), overwrite=True)
    neutral = _group(store, group_by=["sample", "batch"])
    named = _group(store, group_by=["group", "n_cells"])
    assert named.key_columns == ("group", "n_cells")
    assert [list(key.values()) for key in named.keys] == [
        list(key.values()) for key in neutral.keys
    ]
    assert named.codes.tolist() == neutral.codes.tolist()
    # A label artifact's kind and the labels kept apart sit beside them.
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    kinds = np.asarray(["T", "B"] * 4, dtype=object)
    beside = _group(
        store, group_by="group", groups=labels, kept_apart=(("fraction", kinds),)
    )
    assert beside.key_columns == ("group", "cluster_labels", "fraction")
    assert beside.keys[0] == {"group": "s2", "cluster_labels": "x", "fraction": "T"}


def test_groups_that_labels_with_commas_name_alike_are_refused(store) -> None:
    store.cells.insert("first", np.asarray(["x, y"] * 4 + ["x"] * 4), overwrite=True)
    store.cells.insert("second", np.asarray(["z"] * 4 + ["y, z"] * 4), overwrite=True)
    with pytest.raises(ValueError, match="remove the commas"):
        _group(store, group_by=["first", "second"])


def test_key_columns_are_checked(store) -> None:
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    store.cells.insert("cluster_labels", CLUSTERS.astype(object), overwrite=True)
    with pytest.raises(ValueError, match="name 'cluster_labels' twice"):
        _group(store, group_by="cluster_labels", groups=labels)
    # A group_by column whose labels the artifact keeps apart is named as such,
    # and a label artifact of the kept column's name names it twice.
    with pytest.raises(ValueError, match="'batch', whose labels the artifact keeps"):
        _group(store, group_by="batch", kept_apart=(("batch", BATCHES),))
    with pytest.raises(ValueError, match="name 'cluster_labels' twice"):
        _group(store, groups=labels, kept_apart=(("cluster_labels", BATCHES),))
    with pytest.raises(ValueError, match="'absent', which is not a cell metadata"):
        _group(store, group_by=["sample", "absent"])
    with pytest.raises(TypeError, match="groups must be an ArtifactRef or None"):
        _group(store, groups="cluster")
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        _group(store, cell_selection="odd")
    with pytest.raises(ValueError, match="rows must hold one entry per cell"):
        _group(store, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="labels of 'kind' must hold one per"):
        _group(store, kept_apart=(("kind", np.asarray(["T"])),))


def _prepared(store, rows=None, *, selection: str = "every_cell", **options):
    """A call of ``cell_groups`` whose selection and rows are found before it runs."""
    cells = store.snapshot_cell_selection(selection)
    n_cells = int(np.count_nonzero(np.asarray(store.cells.fetch_all(selection))))
    mask = np.ones(n_cells, dtype=bool) if rows is None else np.asarray(rows)
    arguments = {"group_by": None, "groups": None, "cell_selection": None, **options}

    def call():
        return cell_groups(
            store.zw,
            store.cells,
            cells,
            mask,
            ledger=MemoryLedger(None, "Grouping cells"),
            **arguments,
        )

    return call


def _smallest(store, **options) -> tuple[int, int, list[str]]:
    """The smallest budget that a grouping's refusals name, its traced peak there, and the refusals.

    Each refusal must name a budget above the one it refused, and one byte
    below the budget that runs is refused naming it.
    """
    budget, refusals, _grouped, peak = budget_that_runs(
        lambda budget: budget,
        lambda budget: _group(
            store, ledger=MemoryLedger(budget, "Grouping cells"), **options
        ),
    )
    with pytest.raises(MemoryBudgetError) as refused:
        _group(store, ledger=MemoryLedger(budget - 1, "Grouping cells"), **options)
    assert refused.value.needed_bytes == budget
    assert peak is not None and peak <= budget, (peak, budget)
    return budget, peak, refusals


@pytest.mark.parametrize(
    ("n_groups", "n_columns", "label"),
    [
        (20_000, 1, "{:07d}"),
        (20_000, 13, "{:07d}"),
        (2_000, 3, "{:030d}"),
        (2_000, 1, "\u00e9t\u00e9 {:06d}"),
        (2_000, 2, "\U0001f600 {:06d}"),
    ],
    ids=["one column", "13 columns", "long labels", "Latin-1", "four-byte"],
)
def test_fingerprinting_groups_holds_what_its_bytes_state(
    n_groups, n_columns, label
) -> None:
    from cytearc.datastore._operations.cell_groups import CellGroups

    columns = tuple(f"column_{k}" for k in range(n_columns))
    keys = [
        {column: label.format(group) for column in columns} for group in range(n_groups)
    ]
    grouped = CellGroups(columns, keys, np.arange(100_000) % n_groups)
    _result, peak = traced_peak(grouped.fingerprint, 1 << 30)
    stated = grouped.fingerprint_bytes()
    assert peak <= stated <= 1.6 * peak + 4096, (peak, stated)


@pytest.mark.parametrize("n_columns", [1, 3, 13])
@pytest.mark.parametrize("labels", ["own", "few"])
def test_numbering_combinations_holds_what_its_charge_plans(n_columns, labels) -> None:
    from cytearc.datastore._operations.cell_groups import _combining_bytes
    from cytearc.utils.arrays import unique_rows

    # The grouping's step, beside the key columns' codes: 20,000 cells of a
    # label of their own in each column, every cell a group, or of eight.
    rng = np.random.default_rng(n_columns)
    n_cells = 20_000
    coded = [
        rng.permutation(n_cells) if labels == "own" else rng.integers(0, 8, n_cells)
        for _ in range(n_columns)
    ]

    def combine() -> tuple[np.ndarray, np.ndarray]:
        stacked = np.column_stack(coded)
        keep = (stacked >= 0).all(axis=1)
        return unique_rows(stacked[keep])

    _result, peak = traced_peak(combine, 1 << 30)
    charged = _combining_bytes(n_cells, n_columns)
    assert peak <= charged, (peak, charged)
    if labels == "own":
        assert charged <= 1.15 * peak, (peak, charged)


def test_grouping_without_columns_counts_the_chunk_of_the_selection_it_reads(
    store, tmp_path
) -> None:
    from tests.storage_helpers import copy_store, rechunk_column

    # The artifact's selection holds a flag for each cell of the store, in
    # the one chunk of the store's 8 rows. In chunks of 100,000, as a store
    # written before cell columns were chunked by their rows keeps them,
    # reading which 8 cells it holds decodes 100,000 bytes, however few
    # cells the store holds; the grouping charges that read whether or not
    # it groups by anything.
    selection = store.snapshot_cell_selection("every_cell")
    assert artifact_group(store.zw, selection)["values"].chunks == (N_CELLS,)
    older = copy_store(Path(store.zarr_loc), tmp_path)
    rechunk_column(artifact_group(older.zw, selection), "values")
    older = open_store(older.zarr_loc)
    values = artifact_group(older.zw, older.snapshot_cell_selection("every_cell"))[
        "values"
    ]
    assert values.chunks == (100_000,)
    budget, peak, refusals = _smallest(older)
    assert budget > 100_000
    assert refusals[0].startswith(
        "Grouping cells holds about"
    ) and "the cells of the artifact's selection" in " ".join(refusals)
    assert peak > 100_000


@pytest.mark.parametrize(
    "options",
    [
        {"group_by": ["sample", "batch"]},
        {"group_by": "partial"},
        {"group_by": "batch", "cell_selection": "odd"},
        {"groups": "first_four"},
        {"group_by": "batch", "kept_apart": "kinds"},
    ],
    ids=["two columns", "a nullable column", "a selection", "labels", "kept apart"],
)
def test_grouping_runs_within_the_budget_that_its_refusals_name(store, options) -> None:
    options = dict(options)
    if "cell_selection" in options:
        options["cell_selection"] = store.snapshot_cell_selection(
            options["cell_selection"]
        )
    if "groups" in options:
        options["groups"] = store.snapshot_cluster_labels(
            "cluster",
            cell_selection=store.snapshot_cell_selection(options["groups"]),
        )
    if "kept_apart" in options:
        options["kept_apart"] = (("kind", np.asarray(["B", "T"] * 4, dtype=object)),)
    _smallest(store, **options)


def test_what_the_caller_holds_counts_beside_the_grouping(store) -> None:
    alone, _peak, _refusals = _smallest(store, group_by=["sample", "batch"])
    ledger = MemoryLedger(alone + 999, "Grouping cells")
    ledger.hold("the chain slots", 1_000)
    with pytest.raises(MemoryBudgetError) as refused:
        _group(store, group_by=["sample", "batch"], ledger=ledger)
    assert refused.value.needed_bytes == alone + 1_000


def test_each_cells_group_and_the_groups_labels_stay_charged(store) -> None:
    # Four groups of two key columns: each cell's code, 8 bytes, each group's
    # labels and name, and the distinct labels that they keep, each a
    # reference and a string.
    ledger = MemoryLedger(None, "Grouping cells")
    _group(store, group_by=["sample", "batch"], ledger=ledger)
    labels = sum(8 + sys.getsizeof(label) for label in ("s2", "s10", "a", "b"))
    assert ledger.charged == 8 * N_CELLS + labels + 4 * (
        GROUPING_BYTES_PER_GROUP + 2 * GROUPING_BYTES_PER_GROUP_COLUMN
    )
    # Without a column, each cell's group alone.
    ledger = MemoryLedger(None, "Grouping cells")
    _group(store, ledger=ledger)
    assert ledger.charged == 8 * N_CELLS
    # Within groups keep each cell's group in place of its code.
    within = MemoryLedger(None, "Grouping cells")
    codes, keys, _fingerprint = within_groups(
        store.zw,
        store.cells,
        store.snapshot_cell_selection("every_cell"),
        ("batch",),
        np.ones(N_CELLS, dtype=bool),
        ledger=within,
        what="a clone",
    )
    assert codes.tolist() == [0, 0, 1, 1, 0, 0, 1, 1] and len(keys) == 2
    labels = sum(8 + sys.getsizeof(label) for label in ("a", "b"))
    assert within.charged == 8 * N_CELLS + labels + 2 * (
        GROUPING_BYTES_PER_GROUP + GROUPING_BYTES_PER_GROUP_COLUMN
    )


def test_a_column_of_several_chunks_holds_the_chunk_read_before(tmp_path) -> None:
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_CELLS, 3), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.zw["cellData"].create_array("halves", data=SAMPLES, chunks=(4,))
    store = open_store(path)
    assert store.cells._get_array("halves").chunks == (4,)
    # One column read a chunk of four cells at a time, each read holding the
    # work of the read before, as a busy machine can, stays within every
    # interval the ledger plans.
    halves = _prepared(store, group_by="halves")
    # The selection's identity check, outside the budget, runs once in the
    # operation's scope before the grouping.
    with validation_scope():
        halves()
        with hold_finished_reads(*ZARR_READS):
            grouped, overruns = audited(halves)
    assert overruns == []
    assert grouped.keys == [{"halves": "s2"}, {"halves": "s10"}]
    _smallest(store, group_by="halves")


@pytest.mark.parametrize("lead", ["\u00c9", "\U0001f600"], ids=["Latin-1", "four-byte"])
def test_labels_beyond_ascii_read_as_a_string_each_hold_their_utf8_forms(
    tmp_path, lead
) -> None:
    # 20,000 cells of 50 labels beyond ASCII, read in chunks of 2,000 rows as
    # a string for each cell: pandas keeps in each string that it hashes its
    # UTF-8 form, as long as the string lives, which the grouping charges
    # with the labels it reads, with every read's work held until the next
    # read ends and without.
    path = tmp_path / "store.zarr"
    n_cells = 20_000
    write_count_store(str(path), {"RNA": np.ones((n_cells, 1), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    labels = np.asarray([lead * 30 + f"{index % 50:02d}" for index in range(n_cells)])
    store.zw["cellData"].create_array("label", data=labels, chunks=(2_000,))
    store = open_store(path)
    grouped = _prepared(store, group_by="label")
    for held in (False, True):
        with validation_scope():
            grouped()
            if held:
                with hold_finished_reads(*ZARR_READS):
                    result, overruns = audited(grouped)
            else:
                result, overruns = audited(grouped)
        assert overruns == [], (held, overruns)
    assert len(result.keys) == 50


def test_one_group_of_many_cells_holds_no_more_than_each_charge_plans(
    tmp_path,
) -> None:
    # 200,000 cells in one group, without a column: their positions, 1.6 MB,
    # and whether each may join the group are let go only once they are
    # freed, with every read's work held until the next read ends and without.
    path = tmp_path / "store.zarr"
    n_cells = 200_000
    write_count_store(str(path), {"RNA": np.ones((n_cells, 1), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    grouped = _prepared(store)
    for held in (False, True):
        with validation_scope():
            grouped()
            if held:
                with hold_finished_reads(*ZARR_READS):
                    result, overruns = audited(grouped)
            else:
                result, overruns = audited(grouped)
        assert overruns == [], (held, overruns)
    assert result.keys == [{}] and np.all(result.codes == 0)


@pytest.fixture(scope="module")
def many(tmp_path_factory) -> DataStore:
    """20,000 cells with a label of each, in 13 key columns, ASCII and four-byte."""
    path = tmp_path_factory.mktemp("many_groups") / "store.zarr"
    n_cells = 20_000
    write_count_store(str(path), {"RNA": np.ones((n_cells, 2), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    store.cells.insert("own", np.asarray([f"g{index:06d}" for index in range(n_cells)]))
    for copy in range(12):
        store.cells.insert(
            f"own_{copy}", np.asarray([f"g{index:06d}" for index in range(n_cells)])
        )
    store.cells.insert(
        "emoji",
        np.asarray(
            ["\U0001f600" * 30 + f"{index % 500:03d}" for index in range(n_cells)]
        ),
    )
    store.cells.insert("half", np.arange(n_cells) % 2 == 0)
    return store


@pytest.mark.parametrize(
    "options",
    [
        {"group_by": "own"},
        {"group_by": ["own", *(f"own_{copy}" for copy in range(12))]},
        {"group_by": "emoji"},
        {"group_by": "emoji", "cell_selection": "half"},
    ],
    ids=["a label of each cell", "13 key columns", "four-byte labels", "a selection"],
)
def test_grouping_many_cells_holds_no_more_than_each_charge_plans(
    many, options
) -> None:
    options = dict(options)
    if "cell_selection" in options:
        options["cell_selection"] = many.snapshot_cell_selection(
            options["cell_selection"]
        )

    grouped = _prepared(many, **options)

    for held in (False, True):
        # The selections' identity checks, outside the budget, run once in
        # the operation's scope before the grouping.
        with validation_scope():
            grouped()
            if held:
                with hold_finished_reads(*ZARR_READS):
                    _result, overruns = audited(grouped)
            else:
                _result, overruns = audited(grouped)
        assert overruns == [], (held, overruns)
    # The plan counts no more than about twice what the grouping holds.
    _result, taken = charges(grouped)
    planned = max(planned for _item, _nbytes, planned in taken)
    _result, peak = traced_peak(grouped, planned)
    assert peak <= planned <= 2 * peak + 256 * 1024, (peak, planned)


def test_four_byte_labels_of_a_label_artifact_run_within_their_budget(many) -> None:
    # 20,000 cells of 500 labels of 30 characters beyond U+FFFF: each cell's
    # label becomes a Python string of four bytes a character, which the
    # grouping charges before it codes the labels.
    labels = many.snapshot_cluster_labels(
        "emoji", cell_selection=many.snapshot_cell_selection("every_cell")
    )
    _budget, peak, _refusals = _smallest(many, groups=labels)
    assert peak > 20_000 * 4 * 30
    # The strings are charged as the sizes Python gives them, beside their
    # references.
    texts = ["\U0001f600" * 30 + f"{index % 500:03d}" for index in range(20_000)]
    held = sum(map(sys.getsizeof, texts)) + 8 * len(texts)
    _result, taken = charges(lambda: _group(many, groups=labels))
    (charged,) = [
        nbytes
        for item, nbytes, _planned in taken
        if item == "the labels of the groups artifact"
    ]
    assert held <= charged <= held + 1024, (held, charged)


def test_kept_apart_labels_follow_the_rows_that_join_groups(store) -> None:
    rows = np.asarray([False, True, True, False, True, True, False, False])
    kinds = np.asarray(["B", "T", "B", "B", "T", "T", "B", "T"], dtype=object)
    grouped = _group(store, rows, group_by="batch", kept_apart=(("kind", kinds),))
    # By hand: cell1 (a, T), cell2 (b, B), cell4 (a, T), cell5 (a, T).
    assert grouped.keys == [
        {"batch": "a", "kind": "T"},
        {"batch": "b", "kind": "B"},
    ]
    assert grouped.codes.tolist() == [-1, 0, 1, -1, 0, 0, -1, -1]


def _masked_labels(store, values: np.ndarray, missing: np.ndarray):
    """Store cluster labels with a linked mask of the missing ones."""
    planned = plan_cell_data_artifact(
        store.zw,
        scope="datastore",
        kind="cluster_labels",
        operation="test_cell_groups",
        parameters={"case": "masked"},
        inputs={},
        execution_options={},
        cell_selection=store.snapshot_cell_selection("every_cell"),
        arrays={"values": (values.shape, None)},
    )
    mask_name = f"{MISSING_MASK_PREFIX}values"
    with artifact_transaction(store.zw, planned) as group:
        group.create_array("values", data=values)
        group.create_array(mask_name, data=missing)
        group["values"].attrs["missing_mask"] = mask_name
    return planned.ref


def test_an_unsaved_label_artifact_is_refused_before_anything_is_read(store) -> None:
    unsaved = ArtifactRef("datastore", "cluster_labels", "0" * 64)
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        _group(store, groups=unsaved)


def test_a_label_artifact_leaves_its_missing_labels_ungrouped(store) -> None:
    # The masked cells hold a placeholder label, "x", which names no group.
    missing = np.asarray([False, False, True, False, False, True, False, False])
    labels = _masked_labels(store, CLUSTERS.copy(), missing)
    grouped = _group(store, groups=labels)
    assert grouped.keys == [{"cluster_labels": "x"}, {"cluster_labels": "y"}]
    assert grouped.codes.tolist() == [0, 1, -1, 1, 0, -1, 0, 1]


def test_the_fingerprint_follows_each_cells_labels(store) -> None:
    by_sample = _group(store, group_by="sample")
    assert by_sample.fingerprint() == _group(store, group_by="sample").fingerprint()
    # One cell in another sample.
    moved = SAMPLES.copy()
    moved[0] = "s10"
    store.cells.insert("moved", moved.astype(object), overwrite=True)
    assert _group(store, group_by="moved").fingerprint() != by_sample.fingerprint()
    # The same groups under other labels keep their codes.
    renamed = np.where(SAMPLES == "s2", "t2", "t10")
    store.cells.insert("renamed", renamed.astype(object), overwrite=True)
    other = _group(store, group_by="renamed")
    assert other.codes.tolist() == by_sample.codes.tolist()
    assert other.fingerprint() != by_sample.fingerprint()
    # The same labels in a column of another name.
    store.cells.insert("copied", SAMPLES.astype(object), overwrite=True)
    copied = _group(store, group_by="copied")
    assert [list(key.values()) for key in copied.keys] == [
        list(key.values()) for key in by_sample.keys
    ]
    assert copied.fingerprint() != by_sample.fingerprint()
    # The same labels of two columns in the other order, in one group.
    first = np.arange(N_CELLS) == 0
    forward = _group(store, first, group_by=["sample", "batch"])
    backward = _group(store, first, group_by=["batch", "sample"])
    assert forward.codes.tolist() == backward.codes.tolist()
    assert forward.fingerprint() != backward.fingerprint()
