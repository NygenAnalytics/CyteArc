"""Per-cell values reach a store's cell table only keyed by cell ID.

A store's cells need not be in the order of values computed elsewhere, so the
cell table refuses values that would land on cells by position. Feature
tables, whose order does not change, keep positional values, and CyteArc's own
writers of values computed in store order use a private path.
"""

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc import DataStore
from cytearc.metadata import MetaData
from tests.storage_helpers import by_cell_id, write_count_store
from tests.memory_helpers import held_within_charge

IDS = ["c0", "c1", "c2", "c3", "c4", "c5"]
# A shuffled order of the cells, as the obs of another tool may list them.
SHUFFLED = [4, 0, 5, 2, 1, 3]
REMEDY = 'pandas.Series(values, index=ds.cells.fetch_all("ids"))'


def _table(*, cell_table: bool = True, ids: list[str] | None = None) -> MetaData:
    ids = IDS if ids is None else ids
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(len(ids), dtype=bool), chunks=(4,))
    group.create_array("ids", data=np.asarray(ids), chunks=(4,))
    group.create_array("names", data=np.asarray(ids), chunks=(4,))
    group.create_array(
        "sub", data=np.array([True, False, True, True, False, True]), chunks=(4,)
    )
    return MetaData(group, cell_table=cell_table)


def _missing(table: MetaData, column: str) -> list[bool]:
    mask = table._get_missing_mask_array(column)
    return [False] * table.N if mask is None else np.asarray(mask[:]).tolist()


_POSITIONAL_FORMS = [
    ("array", np.arange(6.0), "a NumPy array"),
    ("list", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "values of type list"),
    ("tuple", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), "values of type tuple"),
    (
        "extension-array",
        pd.array([1, None, 3, 4, 5, 6], dtype="Int64"),
        "values of type pandas IntegerArray",
    ),
    (
        "categorical",
        pd.Categorical(["a", "b", "a", "b", "a", "b"]),
        "values of type pandas Categorical",
    ),
    ("index", pd.Index(np.arange(6.0)), "values of type pandas Index"),
    ("range-index", pd.Series(np.arange(6.0)), "a pandas Series with a RangeIndex"),
    (
        "offset-range-index",
        pd.Series(np.arange(6.0), index=pd.RangeIndex(6, 12)),
        "a pandas Series with a RangeIndex",
    ),
]


@pytest.mark.parametrize(
    ("values", "kind"),
    [(values, kind) for _, values, kind in _POSITIONAL_FORMS],
    ids=[name for name, _, _ in _POSITIONAL_FORMS],
)
@pytest.mark.parametrize("key", ["I", "sub"])
def test_cell_table_insert_refuses_positional_values(values, kind, key):
    table = _table()

    with pytest.raises(TypeError) as raised:
        table.insert("score", values, key=key)

    message = str(raised.value)
    assert message.startswith("cells.insert('score') takes per-cell values only")
    assert "keyed by cell ID" in message
    assert f"got {kind}, which would assign values to cells by position" in message
    assert REMEDY in message
    assert f'index=ds.cells.fetch("ids", key="{key}")' in message
    assert "score" not in table.columns


@pytest.mark.parametrize(
    "values",
    [
        np.array([True, False, True, True, True, True]),
        [True, False, True, True, True, True],
        pd.Series([True, False, True, True, True, True]),
    ],
    ids=["array", "list", "range-index"],
)
def test_cell_table_update_key_refuses_positional_values(values):
    table = _table()

    with pytest.raises(TypeError, match=r"^cells\.update_key\('I'\) takes") as raised:
        table.update_key(values, key="I")

    assert REMEDY in str(raised.value)
    assert table.fetch_all("I").all()


def test_a_series_in_shuffled_id_order_lands_on_the_cells_of_its_ids():
    table = _table()
    # Each cell's value is derived from its ID, independently of any order.
    ids = [IDS[row] for row in SHUFFLED]
    values = pd.Series([10.0 * int(cell[1:]) for cell in ids], index=ids)

    table.insert("score", values)

    assert table.fetch_all("score").tolist() == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert _missing(table, "score") == [False] * 6


def test_a_series_for_the_rows_of_key_lands_on_those_rows_in_any_order():
    table = _table()
    # "sub" selects c0, c2, c3 and c5; the values list them in another order.
    values = pd.Series([5, 0, 3, 2], index=["c5", "c0", "c3", "c2"])

    table.insert("depth", values, key="sub")

    stored = table.fetch_all("depth")
    assert stored[[0, 2, 3, 5]].tolist() == [0, 2, 3, 5]
    assert _missing(table, "depth") == [False, True, False, False, True, False]

    table.insert("filled", values, key="sub", fill_value=-1, overwrite=True)
    assert table.fetch_all("filled").tolist() == [0, -1, 2, 3, -1, 5]
    assert _missing(table, "filled") == [False] * 6


def test_categorical_values_keyed_by_cell_id_keep_their_categories():
    table = _table()
    ids = [IDS[row] for row in SHUFFLED]
    labels = ["T" if int(cell[1:]) % 2 else "B" for cell in ids]
    values = pd.Series(
        pd.Categorical(labels, categories=["T", "B", "NK"], ordered=True), index=ids
    )

    table.insert("cell_type", values)

    stored = table.fetch_categorical("cell_type", key=None)
    assert stored.tolist() == ["B", "T", "B", "T", "B", "T"]
    assert stored.categories.tolist() == ["T", "B", "NK"]
    assert stored.ordered


def test_nullable_values_keyed_by_cell_id_flag_the_missing_cell():
    table = _table()
    ids = [IDS[row] for row in SHUFFLED]
    # Cell c2 has no value.
    numbers = [None if cell == "c2" else int(cell[1:]) for cell in ids]
    values = pd.Series(pd.array(numbers, dtype="Int64"), index=ids)

    table.insert("count", values)

    assert _missing(table, "count") == [False, False, True, False, False, False]
    stored = table.fetch_all("count")
    assert stored[[0, 1, 3, 4, 5]].tolist() == [0, 1, 3, 4, 5]


def test_update_key_takes_a_boolean_series_keyed_by_cell_id():
    table = _table()
    keep = pd.Series(
        [cell not in {"c1", "c4"} for cell in reversed(IDS)], index=list(reversed(IDS))
    )

    table.update_key(keep, key="I")

    assert table.fetch_all("I").tolist() == [True, False, True, True, False, True]
    # Values for the rows that the key selects are matched to those rows.
    table.update_key(
        pd.Series([False, True, True, True], index=["c5", "c0", "c2", "c3"]), key="I"
    )
    assert table.fetch_all("I").tolist() == [True, False, True, True, False, False]


@pytest.mark.parametrize(
    ("index", "problem"),
    [
        (
            ["c0", "c1", "c2", "c3", "c4", "c9"],
            "1 labels are not such ids, such as 'c9', and 1 such ids have no "
            "label, such as 'c5'",
        ),
        (["c0", "c1", "c2", "c3", "c4"], "0 labels are not such ids"),
        (
            ["c0", "c1", "c2", "c3", "c4", "c5", "c6"],
            "1 labels are not such ids, such as 'c6'",
        ),
        (["c0", "c1", "c2", "c3", "c4", "c4"], "the index repeats a label"),
        (
            [0, 1, 2, 3, 4, 5],
            "6 labels are not such ids, such as 0, and 6 such ids have no "
            "label, such as 'c0'",
        ),
    ],
    ids=["unknown-id", "missing-id", "extra-id", "repeated-id", "row-positions"],
)
def test_cell_table_refuses_a_series_not_keyed_by_its_cell_ids(index, problem):
    table = _table()
    values = pd.Series(np.arange(len(index), dtype=np.float64), index=index)

    with pytest.raises(ValueError) as raised:
        table.insert("score", values)

    message = str(raised.value)
    assert message.startswith("cells.insert('score') takes the values of a pandas")
    assert problem in message
    assert REMEDY in message
    # A cell table never suggests assigning the values by position.
    assert "to_numpy" not in message
    assert "score" not in table.columns


def test_cell_table_refuses_a_series_when_cell_ids_repeat():
    table = _table(ids=["c0", "c0", "c2", "c3", "c4", "c5"])

    with pytest.raises(ValueError, match="those ids repeat a value"):
        table.insert("score", pd.Series(np.zeros(6), index=IDS))


def test_update_key_refuses_a_series_not_keyed_by_cell_id():
    table = _table()

    with pytest.raises(ValueError, match=r"^cells\.update_key\('I'\) takes") as raised:
        table.update_key(pd.Series([True] * 6, index=[1, 2, 3, 4, 5, 6]), key="I")

    assert REMEDY in str(raised.value)
    assert table.fetch_all("I").all()


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ],
    ids=["array", "list", "range-index"],
)
def test_feature_tables_keep_positional_values(values):
    table = _table(cell_table=False)

    table.insert("score", values)
    table.update_key(np.array([True, True, False, True, True, True]), "I")

    assert table.fetch_all("score").tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert table.fetch_all("I").tolist() == [True, True, False, True, True, True]
    # A Series labeled by IDs is still aligned by label.
    labeled = pd.Series([5, 0, 3, 2], index=["c5", "c0", "c3", "c2"])
    table.insert("labeled", labeled, key="sub", fill_value=9)
    assert table.fetch_all("labeled").tolist() == [0, 9, 2, 3, 9, 5]


def test_feature_table_messages_still_offer_positional_values():
    table = _table(cell_table=False)

    with pytest.raises(ValueError, match="Pass values.to_numpy"):
        table.insert("score", pd.Series(np.zeros(6), index=[1, 2, 3, 4, 5, 6]))


def test_cell_table_is_named_explicitly_and_validated():
    assert _table().cell_table is True
    assert _table(cell_table=False).cell_table is False
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("ids", data=np.asarray(IDS))
    assert MetaData(group).cell_table is False
    with pytest.raises(TypeError, match="cell_table must be a boolean"):
        MetaData(group, cell_table=1)


def test_private_store_order_writes_refuse_values_that_miss_a_row():
    table = _table()

    with pytest.raises(ValueError, match=r"needs one value per row in store order, 6"):
        table._insert_in_store_order("score", np.zeros(4))
    with pytest.raises(ValueError, match=r"needs one value per row in store order, 6"):
        table._update_key_in_store_order(np.ones(4, dtype=bool), "I")
    assert "score" not in table.columns
    assert table.fetch_all("I").all()

    table._insert_in_store_order("score", np.arange(6.0))
    table._update_key_in_store_order(np.arange(6) != 2, "I")
    assert table.fetch_all("score").tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert table.fetch_all("I").tolist() == [True, True, False, True, True, True]
    with pytest.raises(ValueError, match="already exists"):
        table._insert_in_store_order("score", np.arange(6.0))


def test_by_cell_id_keys_store_order_values_and_refuses_a_series():
    table = _table()

    keyed = by_cell_id(table, np.arange(6))
    assert keyed.index.tolist() == IDS
    selected = by_cell_id(table, [1, 2, 3, 4], key="sub")
    assert selected.index.tolist() == ["c0", "c2", "c3", "c5"]
    with pytest.raises(TypeError, match="not a Series"):
        by_cell_id(table, pd.Series(np.arange(6)))


# Counts of six cells over five genes, two of them mitochondrial. Every
# quality metric below is derived from these numbers by hand.
_COUNTS = np.array(
    [
        [3, 0, 1, 0, 0],
        [0, 0, 0, 0, 2],
        [5, 1, 0, 2, 2],
        [1, 1, 1, 1, 1],
        [0, 4, 0, 0, 0],
        [2, 0, 6, 0, 2],
    ],
    dtype=np.uint16,
)
_GENES = ["MT-CO1", "MT-ND1", "ACTB", "CD3E", "MS4A1"]


@pytest.fixture
def prepared(tmp_path):
    location = str(tmp_path / "store.zarr")
    write_count_store(
        location,
        {"RNA": _COUNTS},
        np.uint16,
        feature_ids={"RNA": _GENES},
        cell_ids=["s3", "s0", "s5", "s1", "s4", "s2"],
    )
    return DataStore(location, default_assay="RNA", min_features_per_cell=1, nthreads=1)


def test_preparation_writes_quality_metrics_in_store_order(prepared):
    cells = prepared.cells

    assert cells.cell_table is True
    assert prepared.RNA.feats.cell_table is False
    assert cells.fetch_all("RNA_nCounts").tolist() == [4, 2, 10, 5, 4, 10]
    assert cells.fetch_all("RNA_nFeatures").tolist() == [2.0, 1.0, 4.0, 5.0, 1.0, 3.0]
    np.testing.assert_allclose(
        cells.fetch_all("RNA_percentMito"),
        [75.0, 0.0, 60.0, 40.0, 100.0, 20.0],
    )
    # The first writable open removes the cells with at most one feature.
    assert cells.fetch_all("I").tolist() == [True, False, True, True, False, True]
    # Feature tables still take values by position.
    prepared.RNA.feats.insert("rank", np.arange(5))
    assert prepared.RNA.feats.fetch_all("rank").tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(TypeError, match=r"cells\.insert\('rank'\)"):
        cells.insert("rank", np.arange(6))


# The shared alignment of per-cell values to the cells they describe.


def test_align_to_cell_ids_reorders_values_and_keeps_their_dtype():
    from cytearc.metadata.alignment import align_to_cell_ids

    cells = np.asarray(IDS)
    labels = pd.Series(
        pd.Categorical([IDS[row] for row in SHUFFLED], categories=IDS[::-1]),
        index=[IDS[row] for row in SHUFFLED],
    )
    frame = pd.DataFrame(
        {"x": [float(row) for row in SHUFFLED]}, index=[IDS[row] for row in SHUFFLED]
    )

    aligned = align_to_cell_ids(labels, cells, name="f(x=)", cells="six", remedy="")
    rows = align_to_cell_ids(
        frame, cells, name="f(x=)", cells="six", remedy="", frame=True
    )

    # Each value is its cell's own ID, so a misplaced value would show.
    assert aligned.tolist() == IDS
    assert aligned.dtype == labels.dtype
    assert isinstance(aligned.index, pd.RangeIndex)
    assert rows["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_align_to_cell_ids_leaves_out_other_cells_only_when_asked():
    from cytearc.metadata.alignment import align_to_cell_ids

    values = pd.Series([5.0, 9.0, 0.0, 2.0], index=["c5", "elsewhere", "c0", "c2"])
    options = {"name": "f(x=)", "cells": "the 3 cells", "remedy": "Key them."}

    kept = align_to_cell_ids(
        values, np.asarray(["c0", "c2", "c5"]), leave_out_others=True, **options
    )
    assert kept.tolist() == [0.0, 2.0, 5.0]
    with pytest.raises(ValueError) as raised:
        align_to_cell_ids(values, np.asarray(["c0", "c2", "c5"]), **options)
    assert str(raised.value) == (
        "f(x=) takes the values of a pandas Series by ID: its index must hold the "
        "ids of the 3 cells, each once; 1 labels are not such ids, such as "
        "'elsewhere', and 0 such ids have no label. Key them."
    )
    # Leaving out other cells still requires every requested cell.
    with pytest.raises(ValueError, match="1 such ids have no label, such as 'c1'"):
        align_to_cell_ids(
            values, np.asarray(["c0", "c1"]), leave_out_others=True, **options
        )
    with pytest.raises(ValueError, match="the index repeats a label"):
        align_to_cell_ids(
            pd.Series([1, 2], index=["c0", "c0"]),
            np.asarray(["c0"]),
            leave_out_others=True,
            **options,
        )


@pytest.mark.parametrize(
    ("values", "frame", "got"),
    [
        (np.ones(3), False, "got a NumPy array, which would assign values"),
        ([1, 2, 3], False, "got values of type list, which would assign values"),
        (pd.Series([1, 2, 3]), False, "got a pandas Series with a RangeIndex, which"),
        (
            pd.DataFrame({"x": [1, 2, 3]}),
            True,
            "got a pandas DataFrame with a RangeIndex",
        ),
        (np.ones((3, 2)), True, "got a NumPy array, which would assign values"),
        # A keyed DataFrame is no Series: it names the type without a reason.
        (pd.DataFrame({"x": [1]}, index=["c0"]), False, "got a pandas DataFrame. "),
        (pd.Series([1], index=["c0"]), True, "got a pandas Series. "),
    ],
    ids=[
        "array",
        "list",
        "range-series",
        "range-frame",
        "array-frame",
        "frame-for-series",
        "series-for-frame",
    ],
)
def test_require_keyed_refuses_positional_and_other_containers(values, frame, got):
    from cytearc.metadata.alignment import keyed_values_remedy, require_keyed

    container = "DataFrame" if frame else "Series"
    with pytest.raises(TypeError) as raised:
        require_keyed(
            values, name="f(x=)", remedy=keyed_values_remedy(container), frame=frame
        )

    message = str(raised.value)
    assert message.startswith(
        f"f(x=) takes per-cell values only as a pandas {container} keyed by cell ID"
    )
    assert got in message
    assert f"pandas.{container}(values, index=adata.obs_names)" in message


@pytest.mark.parametrize("width", [6, 40])
def test_reading_ids_and_aligning_keyed_values_hold_what_their_estimate_states(
    width: int,
) -> None:
    import tracemalloc

    from cytearc.metadata.alignment import align_to_cell_ids, cell_id_alignment_bytes
    from cytearc.metadata.rows import read_array_rows_chunkwise

    n_cells = 20_000
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stored = root.create_array(
        "ids",
        data=np.asarray([f"{index:0{width}d}" for index in range(n_cells)]),
        chunks=(5_000,),
    )
    # Values keyed by every ID in reverse order, beside 100 other labels.
    labels = [f"{index:0{width}d}" for index in range(n_cells + 100)][::-1]
    values = pd.Series(np.arange(n_cells + 100, dtype=np.float64), index=labels)
    rows = np.arange(n_cells)
    # An untraced read and alignment make the first calls of both.
    align_to_cell_ids(
        values,
        read_array_rows_chunkwise(stored, rows),
        name="x",
        cells="c",
        remedy="r",
        leave_out_others=True,
    )
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    ids = read_array_rows_chunkwise(stored, rows)
    aligned = align_to_cell_ids(
        values, ids, name="x", cells="c", remedy="r", leave_out_others=True
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert aligned.to_numpy()[0] == n_cells + 99
    assert held_within_charge(
        peak - before, cell_id_alignment_bytes(stored, n_cells, len(values))
    )
