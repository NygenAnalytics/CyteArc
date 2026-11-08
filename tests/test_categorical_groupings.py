"""Groupings, samples, bulk profiles and plots of a categorical column read
through its codes: the same results as its text twin, without a label per
cell."""

import numpy as np
import pandas as pd
import pytest

from cytearc.datastore._operations.cell_groups import _column_values, cell_groups
from cytearc.datastore._operations.cell_samples import read_cell_samples
from cytearc.metadata.selection import (
    CellField,
    category_codes,
    resolve_cell_column,
    resolve_grouping,
    sample_codes,
    valid_category_mask,
)
from cytearc.plotting._data import _fetch_cell_column, _resolve_grouping
from cytearc.storage.categorical import CategoricalColumn
from cytearc.utils.memory_budget import MemoryLedger
from tests.memory_helpers import climb_refusals, named_budget, recording_ledger
from tests.storage_helpers import open_store, write_count_store

N_CELLS = 60
# Labels of every kind that a grouping meets: digits that sort naturally, text
# beyond ASCII, a blank label that groupings count as missing, and missing.
POOL = np.array(["s2", "s10", "B cell", "γδ T", " ", None], dtype=object)
LABELS = POOL[np.arange(N_CELLS) % len(POOL)]
SUBGROUPS = np.array(["a", "b", "c"], dtype=object)[np.arange(N_CELLS) // 7 % 3]
# An unused category wider than any label, which no grouping names.
CATEGORIES = ["γδ T", "s10", " ", "B cell", "s2", "an unused category"]


def _refuse_decoding(*_args, **_kwargs):
    raise AssertionError("a cell's label was decoded")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A store whose labels are written as text and as a categorical column."""
    path = tmp_path_factory.mktemp("groupings") / "store.zarr"
    rng = np.random.default_rng(4)
    counts = rng.poisson(3.0, size=(N_CELLS, 5)).astype(np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16")
    store = open_store(path)
    store.cells.insert("text", LABELS)
    store.cells.insert("categorical", pd.Categorical(LABELS, categories=CATEGORIES))
    store.cells.insert("sub_text", SUBGROUPS)
    store.cells.insert("sub_categorical", pd.Categorical(SUBGROUPS))
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.cells.insert("some_cells", np.arange(N_CELLS) % 4 != 1)
    assert isinstance(store.cells._get_array("categorical"), CategoricalColumn)
    assert not isinstance(store.cells._get_array("text"), CategoricalColumn)
    return store


def _categorical(codes, categories) -> pd.Categorical:
    return pd.Categorical.from_codes(np.asarray(codes), categories=pd.Index(categories))


def test_a_categorical_is_valid_where_its_code_names_a_valid_category() -> None:
    values = _categorical([0, 1, -1, 2, 3, 0], ["a", " ", "", "b"])
    assert valid_category_mask(values).tolist() == [
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    masked = valid_category_mask(values, missing_mask=[False] * 5 + [True])
    assert masked.tolist() == [True, False, False, False, True, False]
    with pytest.raises(ValueError, match="missing mask must align"):
        valid_category_mask(values, missing_mask=[False])


_CODING_CASES = {
    # Natural order puts B cell first and s2 before s10, and blank and
    # unused categories name no group.
    "text": (
        [4, 1, 3, 0, 2, -1, 4, 1],
        ["γδ T", "s10", " ", "B cell", "s2", "unused"],
        None,
    ),
    "masked": ([0, 1, 1, 0, 2], ["x", "y", "z"], [False, True, True, False, False]),
    "numbers": ([2, 0, 1, -1, 2], [10, 2, -3], None),
    "floats": ([1, 0, 1], [2.5, -0.5], None),
    "booleans": ([1, 0, 0], [False, True], [False, False, True]),
    "empty": ([], ["a", "b"], None),
}


@pytest.mark.parametrize("case", sorted(_CODING_CASES))
def test_a_categorical_is_coded_as_its_labels_are(case) -> None:
    codes, categories, missing = _CODING_CASES[case]
    categorical = _categorical(codes, categories)
    # The labels as a categorical column decodes them, with the placeholder
    # of their dtype where the code is -1.
    lookup = np.asarray(categories)
    placeholder = {"U": "", "i": 0, "f": np.nan, "b": False}[lookup.dtype.kind]
    labels = np.asarray(
        [placeholder if code < 0 else lookup[code] for code in codes],
        dtype=lookup.dtype,
    )
    mask = np.asarray(codes) < 0 if missing is None else np.asarray(missing)
    mask = mask | (np.asarray(codes) < 0)

    got_codes, got_names = category_codes(categorical, mask)
    want_codes, want_names = category_codes(labels, mask)
    assert got_names == want_names
    np.testing.assert_array_equal(got_codes, want_codes)
    assert got_codes.dtype == np.int64


def test_the_text_case_is_coded_by_hand() -> None:
    codes, categories, _missing = _CODING_CASES["text"]
    coded, names = category_codes(_categorical(codes, categories))
    assert names == ["B cell", "s2", "s10", "γδ T"]
    assert coded.tolist() == [1, 2, 0, 3, -1, -1, 1, 2]


def test_random_categoricals_are_coded_as_their_labels_are() -> None:
    rng = np.random.default_rng(9)
    pool = np.array(["s2", "s10", "B cell", " ", "", "b cell", "T/NK", "γδ T", "nan"])
    for _ in range(200):
        categories = pool[rng.permutation(len(pool))[: rng.integers(1, len(pool))]]
        codes = rng.integers(-1, len(categories), rng.integers(0, 40))
        missing = (codes < 0) | (rng.random(len(codes)) < 0.2)
        labels = np.where(codes < 0, "", categories[np.clip(codes, 0, None)])
        got = category_codes(_categorical(codes, categories), missing)
        want = category_codes(labels.astype(categories.dtype), missing)
        assert got[1] == want[1]
        np.testing.assert_array_equal(got[0], want[0])


def test_coding_a_categorical_charges_its_codes_not_a_hash_table_of_labels() -> None:
    n_values = 20_000
    categories = [f"{index}:" + "x" * 60 for index in range(8)]
    codes = np.arange(n_values, dtype=np.int8) % 8
    categorical = _categorical(codes, categories)
    labels = np.asarray(categories, dtype=object)[codes]

    least = {}
    for name, values in (("categorical", categorical), ("labels", labels)):
        refusals, (coded, names) = climb_refusals(
            lambda budget, values=values: category_codes(
                values, ledger=MemoryLedger(budget, "Coding labels")
            )
        )
        assert names == categories
        np.testing.assert_array_equal(coded, codes)
        least[name] = named_budget(refusals[-1])

    # A categorical holds which values hold a category, the codes gathered
    # and each value's code: at most 26 bytes a value beside its categories.
    assert least["categorical"] <= 26 * n_values + 64 * 1024
    assert least["labels"] >= 2 * least["categorical"]


def test_samples_of_a_categorical_are_numbered_as_its_labels_are() -> None:
    first = _categorical([2, 0, 1, 2, 0], ["s2", "s10", "γδ T"])
    second = np.array(["b", "a", "a", "b", "b"], dtype=object)
    labels = np.asarray(["s2", "s10", "γδ T"], dtype=object)[np.asarray(first.codes)]

    got = sample_codes(("sample", "batch"), [first, second], 5, reason="r")
    want = sample_codes(("sample", "batch"), [labels, second], 5, reason="r")
    np.testing.assert_array_equal(got[0], want[0])
    assert got[1] == want[1]
    assert got[2] == want[2]
    # Samples follow the sorted order of their texts, as NumPy sorts them:
    # s10, s2, then γδ T, and a before b.
    assert got[1] == [
        {"sample": "s10", "batch": "a"},
        {"sample": "s2", "batch": "a"},
        {"sample": "s2", "batch": "b"},
        {"sample": "γδ T", "batch": "b"},
    ]
    assert got[0].tolist() == [3, 1, 0, 3, 2]

    blank = _categorical([0, 1, -1, 0], ["s2", " "])
    with pytest.raises(ValueError, match="has no label for 2 selected cells. r$"):
        sample_codes(("sample",), [blank], 4, reason="r")


def test_numbering_a_categorical_charges_its_categories_not_a_text_per_label() -> None:
    n_values = 20_000
    categories = [f"{index}:" + "x" * 60 for index in range(8)]
    codes = np.arange(n_values, dtype=np.int8) % 8
    labels = np.asarray(categories, dtype=object)[codes]

    least = {}
    for name, values in (
        ("categorical", _categorical(codes, categories)),
        ("labels", labels),
    ):
        refusals, result = climb_refusals(
            lambda budget, values=values: sample_codes(
                ("sample",),
                [values],
                n_values,
                reason="r",
                ledger=MemoryLedger(budget, "Numbering samples"),
            )
        )
        least[name] = named_budget(refusals[-1])
        assert result[1] == [{"sample": text} for text in sorted(categories)]

    # The numbering of the labels makes a text of 62 characters per label and
    # sorts it; the categorical numbers eight texts. The fingerprint of the
    # samples, which renders each cell's sample, costs both the same.
    assert least["categorical"] < least["labels"]


def test_a_cell_column_reads_as_its_codes_on_request(store) -> None:
    selection = store.snapshot_cell_selection("some_cells")
    stored = np.asarray(store.cells._get_array("categorical").codes[:])
    rows = np.flatnonzero(np.arange(N_CELLS) % 4 != 1)

    values, missing, cell_idx = resolve_cell_column(
        store.zw, store.cells, "categorical", cell_selection=selection, categorical=True
    )
    assert isinstance(values, pd.Categorical)
    np.testing.assert_array_equal(np.asarray(values.codes), stored[rows])
    assert values.categories.tolist() == CATEGORIES
    np.testing.assert_array_equal(cell_idx, rows)
    np.testing.assert_array_equal(missing, LABELS[rows] == None)  # noqa: E711

    ledgered = resolve_cell_column(
        store.zw,
        store.cells,
        "categorical",
        cell_selection=selection,
        categorical=True,
        ledger=MemoryLedger(None, "Reading a column"),
    )
    np.testing.assert_array_equal(np.asarray(ledgered[0].codes), stored[rows])
    # A text column reads as its values either way.
    for ledger in (None, MemoryLedger(None, "Reading a column")):
        text, _missing, _rows = resolve_cell_column(
            store.zw,
            store.cells,
            "text",
            cell_selection=selection,
            categorical=True,
            ledger=ledger,
        )
        decoded, _missing, _rows = resolve_cell_column(
            store.zw, store.cells, "text", cell_selection=selection, ledger=ledger
        )
        np.testing.assert_array_equal(text, decoded)

    grouping = resolve_grouping(
        store.zw,
        store.cells,
        CellField("categorical"),
        cell_selection=selection,
        categorical=True,
    )
    np.testing.assert_array_equal(np.asarray(grouping.labels.codes), stored[rows])
    assert not isinstance(
        resolve_grouping(
            store.zw, store.cells, CellField("categorical"), cell_selection=selection
        ).labels,
        pd.Categorical,
    )


def test_reading_a_cell_column_as_its_codes_plans_and_charges_no_label() -> None:
    import zarr
    from zarr.storage import MemoryStore

    from cytearc.metadata.table import MetaData

    n_rows = 4_000
    categories = [f"{index}:" + "x" * 118 for index in range(6)]
    rows = np.asarray(categories, dtype=object)[np.arange(n_rows) % 6]
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(n_rows, dtype=bool))
    group.create_array("ids", data=np.array([f"c{index}" for index in range(n_rows)]))
    group.create_array("names", data=np.array([f"n{index}" for index in range(n_rows)]))
    table = MetaData(group)
    table.insert("labels", pd.Categorical(rows, categories=categories))

    least = {}
    for coded in (False, True):

        def read(budget: int, coded: bool = coded):
            return resolve_cell_column(
                None,
                table,
                "labels",
                cell_selection=None,
                ledger=MemoryLedger(budget, "Reading a cell column"),
                categorical=coded,
            )[0]

        refusals, values = climb_refusals(read)
        least[coded] = named_budget(refusals[-1])
        if coded:
            assert (np.asarray(values.codes) == np.arange(n_rows) % 6).all()
        else:
            assert values.tolist() == rows.tolist()

    # The labels take 480 bytes a row; the codes one byte, beside the rows'
    # positions and their ordering, which either read holds, and the six
    # categories.
    assert least[False] >= n_rows * 480
    assert least[True] <= n_rows * 72 + 64 * 1024


def test_groupings_of_a_categorical_column_equal_its_text_twins(store) -> None:
    cells = store.snapshot_cell_selection("every_cell")
    rows = np.ones(N_CELLS, dtype=bool)
    grouped = {
        layout: cell_groups(
            store.zw,
            store.cells,
            cells,
            rows,
            group_by=[layout, f"sub_{layout}"],
            groups=None,
            cell_selection=None,
            ledger=MemoryLedger(None, "Grouping cells"),
        )
        for layout in ("text", "categorical")
    }
    rename = {"text": "categorical", "sub_text": "sub_categorical"}
    assert [
        {rename[key]: label for key, label in keys.items()}
        for keys in grouped["text"].keys
    ] == grouped["categorical"].keys
    np.testing.assert_array_equal(grouped["text"].codes, grouped["categorical"].codes)
    assert grouped["categorical"].keys[0] == {
        "categorical": "B cell",
        "sub_categorical": "a",
    }


def test_coding_a_categorical_column_for_a_grouping_charges_codes_not_labels(
    tmp_path,
) -> None:
    n_cells = 6_000
    categories = [f"{index}:" + "x" * 60 for index in range(8)]
    labels = np.asarray(categories, dtype=object)[np.arange(n_cells) % 8]
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((n_cells, 2), dtype=int)}, "uint16")
    table = open_store(path).cells
    table.insert("text", labels)
    table.insert("categorical", pd.Categorical(labels, categories=categories))

    most = {}
    for layout in ("text", "categorical"):
        ledger = recording_ledger("Grouping cells")
        values, missing, _read = _column_values(
            ledger, table, layout, np.arange(n_cells)
        )
        codes, names = category_codes(values, missing, ledger=ledger)
        assert names == categories
        np.testing.assert_array_equal(codes, np.arange(n_cells) % 8)
        most[layout] = ledger.most
    assert isinstance(values, pd.Categorical)
    # Reading any column's rows orders them, 48 bytes a cell; beside that the
    # categorical column holds its codes and mask, a byte each, while the text
    # holds a string of 62 characters and a hash entry per cell.
    assert most["categorical"] <= 56 * n_cells + 64 * 1024
    assert most["text"] >= 4 * most["categorical"]


def test_groupings_and_samples_of_a_categorical_column_decode_no_label(
    store, monkeypatch
) -> None:
    every = np.arange(N_CELLS)
    expected_groups = cell_groups(
        store.zw,
        store.cells,
        store.snapshot_cell_selection("every_cell"),
        np.ones(N_CELLS, dtype=bool),
        group_by="text",
        groups=None,
        cell_selection=None,
        ledger=MemoryLedger(None, "Grouping cells"),
    )
    with_text = read_cell_samples(
        store, MemoryLedger(None, "Samples"), "sub_text", every, reason="r"
    )
    monkeypatch.setattr(CategoricalColumn, "decode", _refuse_decoding)
    monkeypatch.setattr(CategoricalColumn, "decode_objects", _refuse_decoding)

    grouped = cell_groups(
        store.zw,
        store.cells,
        store.snapshot_cell_selection("every_cell"),
        np.ones(N_CELLS, dtype=bool),
        group_by="categorical",
        groups=None,
        cell_selection=None,
        ledger=MemoryLedger(None, "Grouping cells"),
    )
    np.testing.assert_array_equal(grouped.codes, expected_groups.codes)
    assert [keys["categorical"] for keys in grouped.keys] == [
        keys["text"] for keys in expected_groups.keys
    ]
    samples = read_cell_samples(
        store, MemoryLedger(None, "Samples"), "sub_categorical", every, reason="r"
    )
    np.testing.assert_array_equal(samples.codes, with_text.codes)
    assert samples.names == with_text.names
    # The fingerprint of the samples names their column, so it is that of the
    # same labels under the categorical column's name.
    assert (
        samples.assignments
        == sample_codes(("sub_categorical",), [SUBGROUPS], N_CELLS, reason="r")[2]
    )
    with pytest.raises(ValueError, match="has no label for 20 selected cells"):
        read_cell_samples(
            store, MemoryLedger(None, "Samples"), "categorical", every, reason="r"
        )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"null_vals": ["s10"]},
        {"pseudo_reps": 2},
        {"count_detected": True, "aggr_type": "mean"},
    ],
    ids=["plain", "null_vals", "pseudo_reps", "detected"],
)
@pytest.mark.parametrize("selected", [False, True], ids=["live", "selection"])
def test_bulk_profiles_of_a_categorical_column_equal_its_text_twins(
    store, monkeypatch, options, selected
) -> None:
    selection = store.snapshot_cell_selection("some_cells") if selected else None
    options = {"aggr_type": "sum", "cell_selection": selection, **options}
    text = store.compare.compute_bulk("text", secondary_groups="sub_text", **options)
    monkeypatch.setattr(CategoricalColumn, "decode", _refuse_decoding)
    monkeypatch.setattr(CategoricalColumn, "decode_objects", _refuse_decoding)

    categorical = store.compare.compute_bulk(
        "categorical", secondary_groups="sub_categorical", **options
    )
    pd.testing.assert_frame_equal(categorical.profiles, text.profiles)
    pd.testing.assert_frame_equal(categorical.columns, text.columns)
    if options.get("count_detected"):
        pd.testing.assert_frame_equal(categorical.detected, text.detected)
    groups = categorical.columns.index.get_level_values("group")
    assert set(groups) <= {"s2", "B cell", "γδ T", "s10"}
    assert " " not in set(groups)


def test_bulk_profiles_of_one_categorical_column_equal_its_text_twins(store) -> None:
    text = store.compare.compute_bulk("text", aggr_type="sum")
    categorical = store.compare.compute_bulk("categorical", aggr_type="sum")
    pd.testing.assert_frame_equal(categorical.profiles, text.profiles)
    assert categorical.columns.index.tolist() == ["B cell", "s10", "s2", "γδ T"]


def _decoded_whole(store, column, *, cell_key, cell_idx, labels):
    """A column's plotted values as reading every row decodes them, then picking the plotted rows."""
    from cytearc.metadata.rows import apply_missing_mask
    from cytearc.plotting._data import _cell_column_missing

    values = (
        np.asarray(store.cells.fetch(column, key=cell_key))
        if cell_idx is None
        else np.asarray(store.cells.fetch_all(column))[cell_idx]
    )
    missing = _cell_column_missing(
        store.cells, column, cell_key=cell_key, cell_idx=cell_idx
    )
    return apply_missing_mask(values, missing, labels=labels)


@pytest.mark.parametrize("cell_key", ["every_cell", "some_cells"])
def test_plots_read_a_categorical_column_as_reading_every_row_does(
    store, cell_key
) -> None:
    keys, cell_idx, columns, missing = _resolve_grouping(
        store,
        group_by=("categorical", "sub_categorical"),
        groups=None,
        cell_key=cell_key,
    )
    _keys, text_idx, text_columns, text_missing = _resolve_grouping(
        store, group_by=("text", "sub_text"), groups=None, cell_key=cell_key
    )
    np.testing.assert_array_equal(cell_idx, text_idx)
    np.testing.assert_array_equal(missing, text_missing)
    for key, got, text_labels in zip(keys, columns, text_columns, strict=True):
        want = _decoded_whole(store, key, cell_key=cell_key, cell_idx=None, labels=True)
        assert got.dtype == want.dtype
        assert got.tolist() == want.tolist() == text_labels.tolist()
    assert columns[0].dtype == object and None in columns[0].tolist()
    assert columns[1].dtype.kind == "U"

    every = np.flatnonzero(np.asarray(store.cells.fetch_all(cell_key)))
    # Every other plotted row, and rows that hold no missing label.
    present = every[LABELS[every] != None]  # noqa: E711
    for labels in (False, True):
        for index in (None, every[::2], present):
            got = _fetch_cell_column(
                store, "categorical", cell_key=cell_key, cell_idx=index, labels=labels
            )
            want = _decoded_whole(
                store, "categorical", cell_key=cell_key, cell_idx=index, labels=labels
            )
            text_values = _fetch_cell_column(
                store, "text", cell_key=cell_key, cell_idx=index, labels=labels
            )
            assert got.dtype == want.dtype
            assert got.tolist() == want.tolist() == text_values.tolist()


def test_a_column_of_as_many_categories_as_rows_reads_its_codes_without_the_reserved_entry() -> (
    None
):
    import zarr
    from zarr.storage import MemoryStore

    from cytearc.metadata.table import MetaData
    from cytearc.storage.categorical import PADDED_ATTRIBUTE

    labels = np.array(["s10", "B cell", "s2", "γδ T"], dtype=object)
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(4, dtype=bool))
    group.create_array("ids", data=np.array([f"c{index}" for index in range(4)]))
    group.create_array("names", data=np.array([f"n{index}" for index in range(4)]))
    table = MetaData(group)
    table.insert("labels", pd.Categorical(labels))
    column = table._get_array("labels")
    assert column.categories_array.attrs.get(PADDED_ATTRIBUTE) is True
    assert column.categories_array.shape == (5,)

    values, missing, _rows = resolve_cell_column(
        None, table, "labels", cell_selection=None, categorical=True
    )
    assert values.categories.tolist() == ["B cell", "s10", "s2", "γδ T"]
    assert missing is None
    coded, names = category_codes(values)
    assert names == ["B cell", "s2", "s10", "γδ T"]
    assert coded.tolist() == [2, 0, 1, 3]
