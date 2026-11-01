"""Cells grouped by columns, label artifacts and kept-apart labels, checked by hand."""

import numpy as np
import pytest

from cytearc.datastore._operations.cell_groups import (
    GROUPING_BYTES_PER_CELL,
    GROUPING_BYTES_PER_COLUMN,
    GROUPING_BYTES_PER_DISTINCT_LABEL,
    GROUPING_BYTES_PER_GROUP,
    GROUPING_BYTES_PER_LABEL_BYTE,
    GROUPING_BYTES_PER_LABELLED_CELL,
    GROUPING_BYTES_PER_SELECTED_CELL,
    cell_groups,
    group_columns,
)
from cytearc.datastore.datastore import DataStore
from cytearc.metadata.artifacts import plan_cell_data_artifact
from cytearc.metadata.rows import READ_OBJECT_BYTES, array_row_selection_parts
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.artifact_writer import artifact_transaction
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from tests.storage_helpers import insert_nullable_cell_column, write_count_store

N_CELLS = 8
# Natural order puts s2 before s10.
SAMPLES = np.asarray(["s2", "s10", "s2", "s10", "s2", "s10", "s2", "s10"])
BATCHES = np.asarray(["a", "a", "b", "b", "a", "a", "b", "b"])
CLUSTERS = np.asarray(["x", "y", "x", "y", "x", "y", "x", "y"])


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> DataStore:
    path = tmp_path_factory.mktemp("cell_groups") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_CELLS, 3), dtype=int)}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
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
        "memory_bytes": 10**9,
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
        ("clones", ValueError, "'clones', which is also a column of the result"),
    ],
)
def test_group_columns_are_checked(value, error, message) -> None:
    with pytest.raises(error, match=message):
        group_columns(value, reserved=frozenset({"clones"}))


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
    with pytest.raises(ValueError, match="name 'batch' twice"):
        _group(store, group_by="batch", kept_apart=(("batch", BATCHES),))
    with pytest.raises(ValueError, match="key column 'cluster_labels' is also a"):
        _group(store, groups=labels, reserved=frozenset({"cluster_labels"}))
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


def _chunk(array) -> int:
    """The bytes of one chunk that reading an array holds, by the store's model."""
    return array_row_selection_parts(array)[0]


def _selection_chunk(store, name: str) -> int:
    """The bytes of one chunk of a cell selection's flags, one per cell of the store."""
    return _chunk(
        artifact_group(store.zw, store.snapshot_cell_selection(name))["values"]
    )


def _assert_needs(store, needed: int, **options) -> None:
    """Grouping runs within ``needed`` bytes and refuses one byte fewer."""
    assert _group(store, memory_bytes=needed, **options).codes.shape == (N_CELLS,)
    with pytest.raises(MemoryError, match=f"holds about {needed} bytes"):
        _group(store, memory_bytes=needed - 1, **options)


def test_grouping_is_checked_against_the_memory_budget(store) -> None:
    # Eight cells by two columns, whose labels are stored 3 and 1 characters
    # wide in 4 bytes each: 8 * (100 + 2 * 70 + 3 * (12 + 4)) bytes, and the
    # larger column's chunk while it is read.
    per_cell = GROUPING_BYTES_PER_CELL + 2 * GROUPING_BYTES_PER_COLUMN
    per_cell += GROUPING_BYTES_PER_LABEL_BYTE * (12 + 4)
    assert N_CELLS * per_cell == 2_304
    chunk = max(_chunk(store.cells._get_array(name)) for name in ("sample", "batch"))
    # Each column holds two distinct labels, and they make four groups.
    named = (2 + 2) * GROUPING_BYTES_PER_DISTINCT_LABEL + 4 * GROUPING_BYTES_PER_GROUP
    _assert_needs(store, 2_304 + chunk + named, group_by=["sample", "batch"])


def test_a_selection_and_a_label_artifact_count_all_their_cells(store) -> None:
    # The four odd cells of a selection are read whatever the artifact holds.
    batch = _chunk(store.cells._get_array("batch"))
    per_cell = (
        GROUPING_BYTES_PER_CELL
        + GROUPING_BYTES_PER_COLUMN
        + GROUPING_BYTES_PER_LABEL_BYTE * 4
    )
    selected = 4 * GROUPING_BYTES_PER_SELECTED_CELL
    # The selection is read right after the artifact's cells, whose stored
    # chunk the read before can still hold, with Zarr's objects.
    before = (
        max(_selection_chunk(store, "every_cell"), _selection_chunk(store, "odd"))
        + READ_OBJECT_BYTES
    )
    named = 2 * GROUPING_BYTES_PER_DISTINCT_LABEL + 2 * GROUPING_BYTES_PER_GROUP
    _assert_needs(
        store,
        N_CELLS * per_cell + batch + selected + before + named,
        group_by="batch",
        cell_selection=store.snapshot_cell_selection("odd"),
    )
    # A label artifact of four cells, labelled one character wide.
    labels = store.snapshot_cluster_labels(
        "cluster", cell_selection=store.snapshot_cell_selection("first_four")
    )
    values = artifact_group(store.zw, labels)["values"]
    assert values.dtype == np.dtype("<U1")
    labelled = 4 * (
        GROUPING_BYTES_PER_LABELLED_CELL + GROUPING_BYTES_PER_LABEL_BYTE * 4
    )
    per_cell = GROUPING_BYTES_PER_CELL + GROUPING_BYTES_PER_COLUMN
    _assert_needs(
        store,
        N_CELLS * per_cell + labelled + _chunk(values) + named,
        groups=labels,
    )


def test_distinct_labels_and_groups_count_beside_the_cells(store) -> None:
    # A label for each cell names eight labels and makes eight groups, which
    # cost more than the cells.
    store.cells.insert(
        "own", np.asarray([f"cell{i}" for i in range(N_CELLS)]), overwrite=True
    )
    own = store.cells._get_array("own")
    per_cell = GROUPING_BYTES_PER_CELL + GROUPING_BYTES_PER_COLUMN
    per_cell += GROUPING_BYTES_PER_LABEL_BYTE * own.dtype.itemsize
    named = 8 * GROUPING_BYTES_PER_DISTINCT_LABEL + 8 * GROUPING_BYTES_PER_GROUP
    assert named == 8_000
    _assert_needs(store, N_CELLS * per_cell + _chunk(own) + named, group_by="own")


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
