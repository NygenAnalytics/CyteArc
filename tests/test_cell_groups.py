"""Cells grouped by columns, label artifacts and kept-apart labels, checked by hand."""

import numpy as np
import pandas as pd
import pytest

from cytearc.datastore._operations.cell_groups import (
    cell_groups,
    group_columns,
    within_groups,
)
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.artifacts import plan_cell_data_artifact
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.artifact_writer import artifact_transaction
from cytearc.storage.categorical import CategoricalColumn
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
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


@pytest.fixture(scope="module", params=["text", "categorical"])
def store(request, tmp_path_factory) -> DataStore:
    # Every grouping reads the same labels from text columns and from
    # categorical columns, whose codes the reads decode.
    path = tmp_path_factory.mktemp("cell_groups") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_CELLS, 3), dtype=int)}, "uint16")
    store = open_store(path)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[[0, 5]] = True
    if request.param == "text":
        store.cells.insert("sample", SAMPLES.astype(object))
        store.cells.insert("batch", BATCHES.astype(object))
        store.cells.insert("cluster", CLUSTERS.astype(object))
        insert_nullable_cell_column(store, "partial", SAMPLES, missing)
    else:
        store.cells.insert("sample", pd.Categorical(SAMPLES))
        store.cells.insert("batch", pd.Categorical(BATCHES, categories=["b", "a", "c"]))
        store.cells.insert("cluster", pd.Categorical(CLUSTERS))
        store.cells.insert("partial", pd.Categorical(np.where(missing, None, SAMPLES)))
        for column in ("sample", "batch", "cluster", "partial"):
            assert isinstance(store.cells._get_array(column), CategoricalColumn)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.cells.insert("first_four", np.arange(N_CELLS) < 4)
    store.cells.insert("odd", np.arange(N_CELLS) % 2 == 1)
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


def test_what_the_caller_holds_counts_beside_the_grouping(store) -> None:
    # A megabyte groups 8 cells, but not beside a caller that holds all but
    # 10 bytes of it.
    budget = 1_000_000
    grouped = _group(
        store, group_by=["sample", "batch"], ledger=MemoryLedger(budget, "Grouping")
    )
    assert len(grouped.keys) == 4
    ledger = MemoryLedger(budget, "Grouping cells")
    ledger.hold("the chain slots", budget - 10)
    with pytest.raises(MemoryBudgetError, match="^Grouping cells holds about"):
        _group(store, group_by=["sample", "batch"], ledger=ledger)


def test_within_groups_give_each_cell_the_code_of_its_group(store) -> None:
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


def test_a_column_of_several_chunks_groups_its_cells(tmp_path) -> None:
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_CELLS, 3), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.zw["cellData"].create_array("halves", data=SAMPLES, chunks=(4,))
    store = open_store(path)
    assert store.cells._get_array("halves").chunks == (4,)
    # One column read a chunk of four cells at a time.
    grouped = _group(store, group_by="halves")
    assert grouped.keys == [{"halves": "s2"}, {"halves": "s10"}]


@pytest.mark.parametrize("lead", ["\u00c9", "\U0001f600"], ids=["Latin-1", "four-byte"])
def test_labels_beyond_ascii_read_in_chunks_form_their_groups(tmp_path, lead) -> None:
    # 20,000 cells of 50 labels beyond ASCII, read in chunks of 2,000 rows as
    # a string for each cell.
    path = tmp_path / "store.zarr"
    n_cells = 20_000
    write_count_store(str(path), {"RNA": np.ones((n_cells, 1), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    labels = np.asarray([lead * 30 + f"{index % 50:02d}" for index in range(n_cells)])
    store.zw["cellData"].create_array("label", data=labels, chunks=(2_000,))
    store = open_store(path)
    result = _group(store, group_by="label")
    assert len(result.keys) == 50


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
