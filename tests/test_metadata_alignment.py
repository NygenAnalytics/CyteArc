"""Label-aligned inserts, strict column requests, stored grep names, and lean row indices."""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.metadata import MetaData


def _table(ids: list[str] | None = None, active: list[bool] | None = None) -> MetaData:
    ids = ["c0", "c1", "c2", "c3"] if ids is None else ids
    active = [True] * len(ids) if active is None else active
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.asarray(active, dtype=bool), chunks=(2,))
    group.create_array("ids", data=np.asarray(ids), chunks=(2,))
    group.create_array("names", data=np.asarray(ids), chunks=(2,))
    return MetaData(group)


def _missing_rows(table: MetaData, column: str) -> list[bool]:
    mask = table._get_missing_mask_array(column)
    return [False] * table.N if mask is None else np.asarray(mask[:]).tolist()


def test_series_labelled_by_ids_is_assigned_by_label_not_position():
    table = _table()
    # The annotation table lists the cells in another order than the store.
    labels = pd.Series({"c3": "T", "c0": "B", "c2": "NK", "c1": "Mono"})

    table.insert("cell_type", labels)

    assert table.fetch_all("cell_type").tolist() == ["B", "Mono", "NK", "T"]
    assert _missing_rows(table, "cell_type") == [False] * 4


def test_series_for_the_key_rows_is_aligned_to_the_ids_that_key_selects():
    table = _table(active=[True, False, True, True])
    scores = pd.Series([3.5, 0.5, 2.0], index=["c3", "c0", "c2"])

    table.insert("score", scores)

    values = table.fetch_all("score")
    assert values[[0, 2, 3]].tolist() == [0.5, 2.0, 3.5]
    # The row that I leaves out holds the placeholder and is flagged missing.
    assert np.isnan(values[1])
    assert _missing_rows(table, "score") == [False, True, False, False]


def test_labelled_nullable_values_keep_their_missing_rows_after_alignment():
    table = _table()
    labels = pd.Series(
        pd.Categorical(["T", None, "B", "B"]), index=["c3", "c1", "c0", "c2"]
    )

    table.insert("cell_type", labels)

    assert table.fetch_all("cell_type").tolist() == ["B", "", "B", "T"]
    assert _missing_rows(table, "cell_type") == [False, True, False, False]


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["w", "x", "y", "z"]),
        pd.Series(["w", "x", "y", "z"], index=pd.RangeIndex(5, 9)),
        np.array(["w", "x", "y", "z"]),
        ["w", "x", "y", "z"],
    ],
    ids=["range-index", "offset-range-index", "array", "list"],
)
def test_unlabelled_values_are_still_assigned_by_position(values):
    table = _table()

    table.insert("letter", values)

    assert table.fetch_all("letter").tolist() == ["w", "x", "y", "z"]


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (
            ["c3", "c0", "c2", "c9"],
            r"1 labels are not such ids, such as 'c9', and 1 such ids have no "
            r"label, such as 'c1'",
        ),
        (
            ["c3", "c0", "c2", "c2"],
            "the index repeats a label",
        ),
        (
            [3, 0, 2, 1],
            r"4 labels are not such ids, such as 3, and 4 such ids have no "
            r"label, such as 'c0'",
        ),
    ],
    ids=["unknown-id", "repeated-label", "row-positions"],
)
def test_series_whose_labels_are_not_the_row_ids_is_refused(index, message):
    table = _table()
    values = pd.Series(["a", "b", "c", "d"], index=index)

    with pytest.raises(ValueError, match=message) as error:
        table.insert("cell_type", values)

    assert "the ids of the 4 rows, each once" in str(error.value)
    assert "Pass values.to_numpy() to assign the values by position" in str(error.value)
    assert "cell_type" not in table.columns


def test_series_of_the_wrong_length_names_the_rows_that_key_selects():
    table = _table(active=[True, False, True, True])
    values = pd.Series(["a", "b"], index=["c0", "c2"])

    with pytest.raises(
        ValueError,
        match=r"the ids of the 3 rows that 'I' selects, each once; 0 labels are "
        r"not such ids, and 1 such ids have no label, such as 'c3'",
    ):
        table.insert("cell_type", values)


def test_series_cannot_be_aligned_to_ids_that_repeat():
    table = _table(ids=["c0", "c0", "c2", "c3"])
    values = pd.Series(["a", "b", "c", "d"], index=["c0", "c1", "c2", "c3"])

    with pytest.raises(ValueError, match="the ids of those rows repeat a value"):
        table.insert("cell_type", values)


def test_frame_rows_of_a_key_are_refused_rather_than_misplaced():
    table = _table(active=[True, False, True, True])
    table.insert("score", np.array([0.0, 1.0, 2.0, 3.0]))
    # A frame of the active rows is indexed by row position, not by id.
    frame = table.to_pandas_dataframe(["score"], key="I")
    assert frame.index.tolist() == [0, 2, 3]

    with pytest.raises(ValueError, match="Pass values.to_numpy()"):
        table.insert("doubled", frame["score"] * 2)

    table.insert("doubled", (frame["score"] * 2).to_numpy())
    assert table.fetch("doubled").tolist() == [0.0, 4.0, 6.0]


def test_update_key_aligns_a_labelled_boolean_series():
    table = _table()
    keep = pd.Series([True, False, True, True], index=["c1", "c3", "c0", "c2"])

    table.update_key(keep, key="I")

    assert table.fetch_all("I").tolist() == [True, True, True, False]


def test_to_pandas_dataframe_refuses_columns_the_table_lacks():
    table = _table()
    table.insert("celltype", np.array(["a", "b", "c", "d"]))

    with pytest.raises(
        KeyError, match=r"Columns \['cell_type'\] do not exist in the metadata"
    ):
        table.to_pandas_dataframe(["celltype", "cell_type"])

    frame = table.to_pandas_dataframe(["celltype"])
    assert frame.columns.tolist() == ["celltype"]


def test_grep_returns_stored_names_and_keeps_names_that_differ_in_case():
    table = _table(ids=["Cd4", "CD4", "cd8a", "CD40"])

    # Matching ignores case, and every stored spelling is returned once.
    assert table.grep("cd4$") == ["CD4", "Cd4"]
    assert table.grep("^cd") == ["CD4", "CD40", "Cd4", "cd8a"]
    table.update_key(np.array([False, True, True, True]), key="I")
    assert table.grep("cd4$", only_valid=True) == ["CD4"]


def test_row_index_and_reset_key_do_not_build_python_sequences(monkeypatch):
    n_rows = 1_000_000
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array(
        "I", shape=(n_rows,), dtype=bool, chunks=(n_rows,), fill_value=True
    )

    tracemalloc.start()
    try:
        table = MetaData(group)
        index_peak = tracemalloc.get_traced_memory()[1]
        saved: list[np.ndarray] = []
        monkeypatch.setattr(
            MetaData, "_save", lambda self, column, values: saved.append(values)
        )
        tracemalloc.reset_peak()
        held = tracemalloc.get_traced_memory()[0]
        table.reset_key("I")
        reset_peak = tracemalloc.get_traced_memory()[1] - held
    finally:
        tracemalloc.stop()

    np.testing.assert_array_equal(table.index, np.arange(n_rows))
    assert table.index.dtype == np.int64
    # The index is one int64 array; a list of Python integers costs about 40
    # bytes per row before it becomes that array.
    assert index_peak < 2 * 8 * n_rows
    (values,) = saved
    assert values.dtype == np.bool_ and values.shape == (n_rows,) and values.all()
    # One byte per row; a list of booleans holds eight.
    assert reset_peak < 2 * n_rows
