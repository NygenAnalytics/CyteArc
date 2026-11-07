"""Categorical metadata columns: codes beside categories, read as the labels a text column holds.

Each check compares a reading with labels written out by hand, or a text
column of the same labels with a categorical one, so a layout that read or
fingerprinted its labels differently fails.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.datastore._operations.cell_samples import read_cell_samples
from cytearc.metadata import MetaData
from cytearc.metadata.encoding import categorical_codes
from cytearc.metadata.rows import (
    iter_metadata_column_blocks,
    metadata_column_fingerprint,
    metadata_column_stamp,
    metadata_column_stamp_bytes,
    metadata_missing_mask,
    read_metadata_missing_rows,
    read_metadata_rows,
    read_metadata_rows_chunkwise,
    read_rows,
    read_rows_bytes,
    read_text_rows,
    read_whole,
    read_whole_text,
)
from cytearc.metadata.selection import CellField, resolve_cell_column, resolve_grouping
from cytearc.storage.categorical import (
    CategoricalColumn,
    category_codes_dtype,
    linked_categories,
)
from cytearc.storage.identity import replace_metadata_column
from cytearc.storage.metadata_keys import metadata_column_keys
from cytearc.storage.selections import snapshot_run_metadata
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.storage_helpers import open_store, write_count_store

# Labels with a missing row, text beyond ASCII, slashes and underscores, and
# numbers written as text, whose natural order puts "2" before "10".
LABELS = [
    "T/NK",
    "B_cell",
    None,
    "Mono/DC",
    "T/NK",
    "élan",
    "1_10",
    "1_2",
    None,
    "10",
    "2",
    "T/NK",
]
# The categories in an order of their own, with one that no row holds and
# that is wider than every label.
CATEGORIES = [
    "T/NK",
    "B_cell",
    "Mono/DC",
    "élan",
    "1_10",
    "1_2",
    "10",
    "2",
    "unused category",
]
# Each row's position in CATEGORIES, -1 for the missing rows.
CODES = [0, 1, -1, 2, 0, 3, 4, 5, -1, 6, 7, 0]
# What a text column of LABELS holds: the placeholder, empty text, in missing rows.
STORED = [
    "T/NK",
    "B_cell",
    "",
    "Mono/DC",
    "T/NK",
    "élan",
    "1_10",
    "1_2",
    "",
    "10",
    "2",
    "T/NK",
]
MISSING = [label is None for label in LABELS]
N = len(LABELS)
# The rows that I selects in the tables of these tests.
SELECTED = np.array([index % 3 != 1 for index in range(N)])


def _table() -> MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=SELECTED)
    group.create_array("ids", data=np.array([f"cell{index}" for index in range(N)]))
    group.create_array("names", data=np.array([f"name{index}" for index in range(N)]))
    return MetaData(group)


def _insert(table: MetaData, layout: str, name: str = "labels") -> None:
    if layout == "text":
        table.insert(name, np.array(LABELS, dtype=object))
    else:
        table.insert(name, pd.Categorical(LABELS, categories=CATEGORIES))
        assert isinstance(table._get_array(name), CategoricalColumn)


class _RecordingLedger(MemoryLedger):
    """A ledger without a budget that records the most bytes held at once, peaks included."""

    def __init__(self) -> None:
        super().__init__(None, "Reading labels")
        self.most = 0

    def hold(self, item, nbytes, *, remedy=None):
        held = super().hold(item, nbytes, remedy=remedy)
        self.most = max(self.most, self.held_bytes)
        return held

    def peak(self, item, nbytes, *, remedy=None):
        super().peak(item, nbytes, remedy=remedy)
        self.most = max(self.most, self.held_bytes + int(nbytes))


# The layout.


def test_a_categorical_insert_stores_codes_beside_its_categories() -> None:
    table = _table()

    _insert(table, "categorical")

    group = table._group
    codes = group["labels"]
    assert codes.dtype == np.int8
    assert codes[:].tolist() == CODES
    assert codes.chunks == (N,)
    assert codes.attrs.asdict() == {
        "categories": "__cytearc_categories__labels",
        "missing_mask": "__cytearc_missing__labels",
    }
    categories = group["__cytearc_categories__labels"]
    assert categories.dtype == np.dtype("U15")
    assert categories[:].tolist() == CATEGORIES
    assert categories.attrs.asdict() == {"ordered": False}
    assert group["__cytearc_missing__labels"][:].tolist() == MISSING
    assert group["__cytearc_missing__labels"].chunks == (N,)
    # The linked arrays are not columns of the table.
    assert table.columns == ["I", "ids", "names", "labels"]
    assert list(table.head(2).columns) == ["I", "ids", "names", "labels"]


@pytest.mark.parametrize(
    ("n_categories", "dtype"),
    [
        (1, np.int8),
        (128, np.int8),
        (129, np.int16),
        (32_768, np.int16),
        (32_769, np.int32),
    ],
)
def test_codes_take_the_smallest_signed_integer_type_of_their_categories(
    n_categories: int, dtype: type
) -> None:
    # int8 holds the positions 0 to 127 beside -1, int16 0 to 32,767.
    table = _table()
    categories = [f"c{index}" for index in range(n_categories)]
    rows = [categories[-1], categories[0]] * (N // 2)

    table.insert("labels", pd.Categorical(rows, categories=categories))

    stored = table._group["labels"]
    assert stored.dtype == np.dtype(dtype)
    assert stored[:2].tolist() == [n_categories - 1, 0]
    assert table.fetch_all("labels").tolist() == rows
    # No row is missing, so no mask is linked.
    assert "missing_mask" not in stored.attrs
    assert metadata_missing_mask(table, "labels") is None


def test_ordered_categories_keep_their_order() -> None:
    table = _table()
    sizes = ["small", "large", "medium"] * 4

    table.insert(
        "size",
        pd.Categorical(sizes, categories=["small", "medium", "large"], ordered=True),
    )

    assert table._group["__cytearc_categories__size"].attrs["ordered"] is True
    categorical = table.fetch_categorical("size", key=None)
    assert categorical.ordered
    assert categorical.categories.tolist() == ["small", "medium", "large"]
    assert categorical.codes.tolist() == [0, 2, 1] * 4


def test_numeric_categories_are_stored_as_numbers() -> None:
    table = _table()
    integers = [3, 1, None, 3, 1, 1, None, 3, 3, 1, 1, 3]
    fractions = [0.5, None, 2.5, 0.5, 0.5, None, 2.5, 2.5, 0.5, 0.5, 2.5, 0.5]

    table.insert("integers", pd.Categorical(integers, categories=[1, 3, 7]))
    table.insert("fractions", pd.Categorical(fractions))

    assert table._group["__cytearc_categories__integers"][:].tolist() == [1, 3, 7]
    assert table.get_dtype("integers") == np.int64
    assert table.fetch_all("integers").tolist() == [3, 1, 0, 3, 1, 1, 0, 3, 3, 1, 1, 3]
    assert table.get_dtype("fractions") == np.float64
    fetched = table.fetch_all("fractions")
    assert np.isnan(fetched[[1, 5]]).all()
    assert fetched[[0, 2, 3]].tolist() == [0.5, 2.5, 0.5]
    frame = table.to_pandas_dataframe(["integers"])["integers"]
    assert frame.isna().tolist() == [value is None for value in integers]


# Every read returns the same labels from either layout.


@pytest.mark.parametrize("layout", ["text", "categorical"])
def test_reads_return_the_labels_of_either_layout(layout: str) -> None:
    table = _table()
    _insert(table, layout)

    assert table.get_dtype("labels").kind == "U"
    assert table.fetch_all("labels").tolist() == STORED
    assert table.fetch("labels").tolist() == [
        label for label, selected in zip(STORED, SELECTED, strict=True) if selected
    ]
    assert np.asarray(metadata_missing_mask(table, "labels")[:]).tolist() == MISSING
    frame = table.to_pandas_dataframe(["labels"])["labels"]
    assert frame.isna().tolist() == MISSING
    assert frame.dropna().tolist() == [label for label in LABELS if label is not None]
    assert table.head(3)["labels"].isna().tolist() == MISSING[:3]
    # Lookups match text without case and never a missing row.
    assert table.get_index_by(["t/nk", "ÉLAN", ""], "labels").tolist() == [0, 4, 11, 5]
    rows = np.array([11, 2, 6, 0])
    assert read_metadata_rows(table, "labels", rows).tolist() == [
        "T/NK",
        "",
        "1_10",
        "T/NK",
    ]
    assert read_metadata_rows_chunkwise(
        table, "labels", np.array([0, 2, 6, 11])
    ).tolist() == [
        "T/NK",
        "",
        "1_10",
        "T/NK",
    ]
    assert read_metadata_missing_rows(table, "labels", rows).tolist() == [
        False,
        True,
        False,
        False,
    ]
    blocks = list(table.iter_row_blocks(columns=["labels"], block_rows=5))
    assert np.concatenate([block.values["labels"] for block in blocks]).tolist() == [
        label for label, selected in zip(STORED, SELECTED, strict=True) if selected
    ]
    assert (
        np.concatenate(
            list(iter_metadata_column_blocks(table, "labels", block_rows=5))
        ).tolist()
        == STORED
    )


@pytest.mark.parametrize("layout", ["text", "categorical"])
def test_ledger_reads_return_the_labels_of_either_layout(layout: str) -> None:
    table = _table()
    _insert(table, layout)
    array = table._get_array("labels")
    rows = np.array([5, 0, 2, 9])

    ledger = MemoryLedger(None, "Reading labels")
    assert read_rows(ledger, array, rows, item="labels").tolist() == [
        "élan",
        "T/NK",
        "",
        "10",
    ]
    strings = read_text_rows(ledger, array, rows, item="labels")
    assert strings.dtype == object
    assert strings.tolist() == ["élan", "T/NK", "", "10"]
    assert all(type(value) is str for value in strings)
    assert read_whole(ledger, table._group, "labels", item="labels").tolist() == STORED
    assert (
        read_whole_text(ledger, table._group, "labels", item="labels").tolist()
        == STORED
    )


@pytest.mark.parametrize("layout", ["text", "categorical"])
def test_groupings_read_the_labels_of_either_layout(layout: str) -> None:
    table = _table()
    _insert(table, layout, "module")

    grouping = resolve_grouping(None, table, CellField("module"))

    assert grouping.labels.tolist() == STORED
    assert grouping.missing_mask is not None
    assert grouping.missing_mask.tolist() == MISSING


@pytest.mark.parametrize("layout", ["text", "categorical"])
def test_numeric_labels_filter_alike_in_either_layout(layout: str) -> None:
    table = _table()
    values = [3, 1, None, 3, 1, 1, None, 3, 3, 1, 1, 3]
    if layout == "text":
        table.insert("number", np.array(values, dtype=object))
    else:
        table.insert("number", pd.Categorical(values))
        assert isinstance(table._get_array("number"), CategoricalColumn)

    assert table.sift("number", 2, 4).tolist() == [value == 3 for value in values]


def test_a_categorical_column_reads_as_an_array_of_its_labels() -> None:
    table = _table()
    _insert(table, "categorical")

    column = table._get_array("labels")

    assert isinstance(column, CategoricalColumn)
    assert column.shape == (N,) and column.ndim == 1 and column.size == N
    assert column.chunks == (N,)
    assert column.dtype == np.dtype("U15")
    # A whole read returns fifteen characters a label.
    assert column.nbytes == N * 15 * 4
    assert column.attrs["categories"] == "__cytearc_categories__labels"
    assert column.name == "/labels" and column.basename == "labels"
    assert np.asarray(column).tolist() == STORED
    assert np.asarray(column, dtype=object).dtype == object
    assert column[5].tolist() == "élan"
    assert column.get_coordinate_selection(np.array([9, 2])).tolist() == ["10", ""]
    assert column.get_orthogonal_selection((np.array([6, 0]),)).tolist() == [
        "1_10",
        "T/NK",
    ]
    assert repr(column) == (
        "<CategoricalColumn 'labels' of 12 rows, 9 <U15 categories>"
    )
    assert column.used_categories().tolist() == [True] * 8 + [False]
    assert column.stored_labels_dtype() == np.dtype("U7")


def test_a_categorical_column_reads_its_codes_through_another_array() -> None:
    table = _table()
    _insert(table, "categorical")
    column = table._get_array("labels")
    other = zarr.open_group(store=MemoryStore(), mode="w")

    rerouted = column.with_codes(
        other.create_array("labels", data=np.asarray(column.codes[:])[::-1].copy())
    )

    assert rerouted[:].tolist() == STORED[::-1]
    assert rerouted.ordered is column.ordered
    with pytest.raises(ValueError, match="keep their shape and dtype"):
        column.with_codes(other.create_array("short", data=np.zeros(3, dtype=np.int8)))
    with pytest.raises(ValueError, match="keep their shape and dtype"):
        column.with_codes(other.create_array("wide", data=np.zeros(N, dtype=np.int16)))


def test_a_composite_reads_categorical_codes_through_its_source_cache(
    tmp_path: Path,
) -> None:
    from cytearc.composite.create import _CachedMetadataTable, _prefetch_metadata_schema
    from cytearc.composite.source_cache import SourceCache
    from cytearc.storage.budget import ResourceBudget

    group = zarr.open_group(tmp_path / "cells.zarr", mode="w")
    group.create_array("I", data=SELECTED)
    group.create_array("ids", data=np.array([f"cell{index}" for index in range(N)]))
    group.create_array("names", data=np.array([f"name{index}" for index in range(N)]))
    table = MetaData(group)
    _insert(table, "categorical")
    handles = _CachedMetadataTable(table)
    cache = SourceCache()
    try:
        handles.use_source_cache(cache)
        _prefetch_metadata_schema(
            [handles],
            [frozenset()],
            ResourceBudget(1024**2, 2),
            cache=cache,
            resident_bytes=0,
            read_workers=1,
        )
        # The schema scan of a categorical column warms its codes' chunk.
        assert "labels/c/0" in {key for _source, key in cache._entries}
        column = handles._get_array("labels")
        assert isinstance(column, CategoricalColumn)
        assert column[:].tolist() == STORED
        assert handles.get_dtype("labels") == np.dtype("U15")
    finally:
        cache.close()


# Fetching codes and categories.


def test_fetch_categorical_returns_the_stored_codes_and_categories() -> None:
    table = _table()
    _insert(table, "categorical")

    every = table.fetch_categorical("labels", key=None)

    assert every.categories.tolist() == CATEGORIES
    assert every.codes.tolist() == CODES
    assert not every.ordered
    assert [None if pd.isna(label) else label for label in every] == LABELS
    selected = table.fetch_categorical("labels")
    assert selected.codes.tolist() == [
        code for code, keep in zip(CODES, SELECTED, strict=True) if keep
    ]
    assert selected.categories.tolist() == CATEGORIES


def test_fetch_categorical_builds_categories_from_a_column_of_values() -> None:
    table = _table()
    _insert(table, "text")

    every = table.fetch_categorical("labels", key=None)

    # Decimal labels first by value, then the others in natural order.
    natural = ["2", "10", "1_2", "1_10", "B_cell", "Mono/DC", "T/NK", "élan"]
    assert every.categories.tolist() == natural
    assert [None if pd.isna(label) else label for label in every] == LABELS
    assert every.codes.tolist() == [
        -1 if label is None else natural.index(label) for label in LABELS
    ]
    assert not every.ordered
    # The rows that I selects, 0, 2, 3, 5, 6, 8, 9 and 11, hold every label
    # but "B_cell", "1_2" and "2".
    selected = table.fetch_categorical("labels")
    kept = [label for label, keep in zip(LABELS, SELECTED, strict=True) if keep]
    assert [None if pd.isna(label) else label for label in selected] == kept
    assert selected.categories.tolist() == ["10", "1_10", "Mono/DC", "T/NK", "élan"]


def test_fetch_categorical_reads_numbers_ascending_and_nan_as_missing() -> None:
    table = _table()
    values = np.array(
        [3.5, np.nan, 1.0, 3.5, 1.0, 2.0, 1.0, np.nan, 2.0, 3.5, 1.0, 1.0]
    )
    table.insert("score", values)

    categorical = table.fetch_categorical("score", key=None)

    assert categorical.categories.tolist() == [1.0, 2.0, 3.5]
    assert categorical.codes.tolist() == [2, -1, 0, 2, 0, 1, 0, -1, 1, 2, 0, 0]


def test_fetch_categorical_refuses_a_mask_that_disagrees_with_the_codes() -> None:
    table = _table()
    _insert(table, "categorical")
    table._group["__cytearc_missing__labels"][0] = True

    with pytest.raises(ValueError, match="disagrees with its codes"):
        table.fetch_categorical("labels", key=None)


def test_fetch_categorical_refuses_missing_codes_without_a_mask() -> None:
    table = _table()
    table.insert("labels", pd.Categorical(["a", "b"] * (N // 2)))
    assert metadata_missing_mask(table, "labels") is None
    table._group["labels"][0] = -1

    with pytest.raises(ValueError, match="disagrees with its codes"):
        table.fetch_categorical("labels", key=None)


def test_fetch_categorical_reads_times_ascending_and_nat_as_missing() -> None:
    table = _table()
    days = np.array(
        ["2020-01-03", "NaT", "2020-01-01", "2020-01-03"] * 3, dtype="datetime64[D]"
    )
    table.insert("day", days)

    categorical = table.fetch_categorical("day", key=None)

    assert categorical.categories.tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-03"),
    ]
    assert categorical.codes.tolist() == [1, -1, 0, 1] * 3


# Fingerprints and identities follow the labels, not the layout.


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> dict[str, object]:
    """Two stores whose cells hold LABELS and samples, as text in one and as categorical columns in the other."""
    folder = tmp_path_factory.mktemp("categorical_columns")
    counts = (np.arange(N * 4).reshape(N, 4) % 5 + 1).astype(int)
    samples = ["s2", "s10", "s2", "s10", "s2", "s10"] * 2
    opened = {}
    for layout in ("text", "categorical"):
        path = folder / f"{layout}.zarr"
        write_count_store(str(path), {"RNA": counts}, "uint16")
        store = open_store(path)
        if layout == "text":
            store.cells.insert("cell_type", np.array(LABELS, dtype=object))
            store.cells.insert("sample", np.array(samples, dtype=object))
        else:
            store.cells.insert(
                "cell_type", pd.Categorical(LABELS, categories=CATEGORIES)
            )
            store.cells.insert("sample", pd.Categorical(samples))
            assert isinstance(store.cells._get_array("cell_type"), CategoricalColumn)
        store.cells.insert("labelled", ~np.array(MISSING))
        opened[layout] = store
    return opened


def test_column_fingerprints_follow_the_labels(stores) -> None:
    text, categorical = stores["text"], stores["categorical"]

    assert metadata_column_fingerprint(categorical.cells, "cell_type") == (
        metadata_column_fingerprint(text.cells, "cell_type")
    )
    # Other labels give another fingerprint.
    assert metadata_column_fingerprint(categorical.cells, "cell_type") != (
        metadata_column_fingerprint(categorical.cells, "sample")
    )


def test_numeric_column_fingerprints_follow_the_labels() -> None:
    table = _table()
    values = [3, 1, None, 3, 1, 1, None, 3, 3, 1, 1, 3]
    table.insert("text", np.array(values, dtype=object))
    table.insert("categorical", pd.Categorical(values, categories=[1, 3, 7]))

    assert metadata_column_fingerprint(table, "categorical") == (
        metadata_column_fingerprint(table, "text")
    )


def test_run_snapshots_store_and_fingerprint_the_labels(stores) -> None:
    payloads = {}
    fingerprints = {}
    for layout, store in stores.items():
        ref = snapshot_run_metadata(
            store.zw,
            table_path="cellData",
            id_column="ids",
            columns=["cell_type", "sample"],
            axis="cell",
        )
        fingerprints[layout] = store.artifacts.inspect(ref).inputs[
            "column_fingerprints"
        ]
        group = store.artifacts.load(ref)
        payloads[layout] = (
            group["cell_type"][:],
            group["__cytearc_missing__cell_type"][:].tolist(),
        )

    assert fingerprints["categorical"] == fingerprints["text"]
    values, missing = payloads["categorical"]
    # The snapshot stores text at the width of the widest label a row holds.
    assert values.dtype == np.dtype("U7")
    assert values.tolist() == STORED
    assert missing == MISSING
    assert payloads["text"][0].dtype == values.dtype


def test_label_snapshots_samples_and_groups_fingerprint_the_labels(stores) -> None:
    from cytearc.datastore._operations.cell_groups import cell_groups

    found = {}
    for layout, store in stores.items():
        labelled = store.snapshot_cell_selection("labelled")
        snapshot = store.snapshot_cluster_labels("cell_type", cell_selection=labelled)
        samples = read_cell_samples(
            store,
            MemoryLedger(None, "Grouping samples"),
            "sample",
            np.flatnonzero(SELECTED),
            reason="Every cell needs a sample.",
        )
        groups = cell_groups(
            store.zw,
            store.cells,
            store.snapshot_cell_selection("I"),
            np.ones(N, dtype=bool),
            group_by=("cell_type", "sample"),
            groups=None,
            cell_selection=None,
            ledger=MemoryLedger(None, "Grouping cells"),
        )
        found[layout] = (
            store.artifacts.inspect(snapshot).inputs["values_fingerprint"],
            np.asarray(store.artifacts.load(snapshot)["values"][:]).tolist(),
            samples.assignments,
            samples.names,
            samples.codes.tolist(),
            groups.fingerprint(),
            groups.keys,
        )

    assert found["categorical"] == found["text"]
    snapshot_values, names = found["text"][1], found["text"][3]
    assert snapshot_values == [label for label in LABELS if label is not None]
    assert names == ["s10", "s2"]


def test_exports_and_plots_read_the_labels(stores) -> None:
    from cytearc.plotting._data import _category_universe, _fetch_cell_column

    exported = {layout: store.to_anndata() for layout, store in stores.items()}
    loaded = {
        layout: store.artifacts.load_values(
            CellField("cell_type"), cell_selection=store.snapshot_cell_selection("I")
        ).to_pandas()
        for layout, store in stores.items()
    }
    plotted = {
        layout: _fetch_cell_column(
            store, "cell_type", cell_key="I", labels=True
        ).tolist()
        for layout, store in stores.items()
    }
    universe = {
        layout: _category_universe(store, "cell_type").tolist()
        for layout, store in stores.items()
    }

    obs = exported["categorical"].obs["cell_type"]
    assert [None if pd.isna(label) else label for label in obs] == LABELS
    pd.testing.assert_series_equal(obs, exported["text"].obs["cell_type"])
    pd.testing.assert_series_equal(loaded["categorical"], loaded["text"])
    assert [None if pd.isna(label) else label for label in loaded["text"]] == LABELS
    # Every cell of these stores is in I.
    assert plotted["categorical"] == plotted["text"] == LABELS
    assert universe["categorical"] == universe["text"] == LABELS


def test_a_stamp_covers_the_categories_as_well_as_the_codes(stores) -> None:
    store = stores["categorical"]
    before = metadata_column_stamp(store.cells, "cell_type")
    categories = store.zw["cellData/__cytearc_categories__cell_type"]
    renamed = categories[:]
    renamed[0] = "NK/T"

    categories[:] = renamed

    # The codes are unchanged, but the labels are not, so saved clones of
    # the column are not reused.
    assert metadata_column_stamp(store.cells, "cell_type") != before
    assert store.cells.fetch_all("cell_type")[0] == "NK/T"
    categories[:] = np.array(CATEGORIES)
    assert metadata_column_stamp(store.cells, "cell_type") == before
    # A stamp reads stored objects, so it tells the layouts apart.
    assert before != metadata_column_stamp(stores["text"].cells, "cell_type")
    # Stamping holds the larger of the stamps of the codes, the categories and
    # the mask, which the store's objects bound.
    stamping = metadata_column_stamp_bytes(store.cells, "cell_type")
    assert 0 < stamping < 1024**2


def test_clones_from_a_column_read_the_labels_of_either_layout(stores) -> None:
    found = {}
    for layout, store in stores.items():
        labelled = store.snapshot_cell_selection("labelled")
        clones = store.clones.load_clones(
            store.clones.from_column("cell_type", cell_selection=labelled)
        )
        found[layout] = clones.cells["clone"].astype(object).to_dict()

    assert found["categorical"] == found["text"]
    assert sorted(set(found["text"].values())) == sorted(
        {label for label in LABELS if label is not None}
    )


# Memory.


def _long_labels(n_rows: int) -> tuple[np.ndarray, list[str]]:
    categories = [f"{index}:" + "x" * 118 for index in range(6)]
    rows = np.array([categories[index % 6] for index in range(n_rows)], dtype=object)
    return rows, categories


def test_reading_categorical_labels_charges_codes_and_categories_not_a_text_per_row() -> (
    None
):
    n_rows = 4_000
    rows, categories = _long_labels(n_rows)
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(n_rows, dtype=bool))
    group.create_array("ids", data=np.array([f"c{index}" for index in range(n_rows)]))
    group.create_array("names", data=np.array([f"n{index}" for index in range(n_rows)]))
    table = MetaData(group)
    table.insert("text", rows)
    table.insert("categorical", pd.Categorical(rows, categories=categories))
    every = np.arange(n_rows)

    charged = {}
    for name in ("text", "categorical"):
        ledger = _RecordingLedger()
        values = read_text_rows(ledger, table._get_array(name), every, item=name)
        assert values.tolist() == rows.tolist()
        charged[name] = ledger.most

    # A text read holds each stored label, four bytes for each of its 120
    # characters, and a string per row; a categorical read holds a code and
    # a reference per row beside the six strings and their stored values.
    assert charged["text"] >= n_rows * 4 * 120
    assert charged["categorical"] <= n_rows * 64 + 64 * 1024
    # A budget between them reads the categorical labels and refuses the text.
    budget = (charged["text"] + charged["categorical"]) // 2
    read_text_rows(
        MemoryLedger(budget, "Reading labels"),
        table._get_array("categorical"),
        every,
        item="labels",
    )
    with pytest.raises(MemoryBudgetError):
        read_text_rows(
            MemoryLedger(budget, "Reading labels"),
            table._get_array("text"),
            every,
            item="labels",
        )


def test_a_scattered_read_charges_the_codes_chunk_rather_than_a_text_chunk() -> None:
    n_rows = 4_000
    rows, categories = _long_labels(n_rows)
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(n_rows, dtype=bool))
    group.create_array("ids", data=np.array([f"c{index}" for index in range(n_rows)]))
    group.create_array("names", data=np.array([f"n{index}" for index in range(n_rows)]))
    table = MetaData(group)
    table.insert("text", rows)
    table.insert("categorical", pd.Categorical(rows, categories=categories))
    few = np.array([7, 3_001, 1_500])

    charged = {}
    for name in ("text", "categorical"):
        ledger = _RecordingLedger()
        array = table._get_array(name)
        values = read_rows(ledger, array, few, item=name)
        assert values.tolist() == rows[few].tolist()
        charged[name] = ledger.most
        # The checks that plan a read before it runs count at least as much.
        assert charged[name] <= read_rows_bytes(array, len(few))

    # The text column decodes its chunk of 4,000 labels of 480 bytes; the
    # codes decode a chunk of 4,000 bytes beside three labels.
    assert charged["text"] >= n_rows * 4 * 120
    assert charged["categorical"] <= 64 * 1024


def test_a_budgeted_column_read_plans_codes_and_categories() -> None:
    from tests.memory_helpers import climb_refusals, named_budget

    n_rows = 4_000
    rows, categories = _long_labels(n_rows)
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(n_rows, dtype=bool))
    group.create_array("ids", data=np.array([f"c{index}" for index in range(n_rows)]))
    group.create_array("names", data=np.array([f"n{index}" for index in range(n_rows)]))
    table = MetaData(group)
    table.insert("text", rows)
    table.insert("categorical", pd.Categorical(rows, categories=categories))

    least = {}
    for name in ("text", "categorical"):

        def read(budget: int, column: str = name) -> np.ndarray:
            values, _missing, _rows = resolve_cell_column(
                None,
                table,
                column,
                cell_selection=None,
                ledger=MemoryLedger(budget, "Reading a cell column"),
            )
            return values

        refusals, values = climb_refusals(read)
        assert values.tolist() == rows.tolist()
        least[name] = named_budget(refusals[-1])

    # Both return the labels at 480 bytes a row. The text read also decodes
    # the chunk of 4,000 stored labels beside its stored copy; the
    # categorical read decodes 4,000 one-byte codes beside six categories.
    assert least["text"] >= 3 * n_rows * 480
    assert least["categorical"] <= n_rows * (480 + 64) + 64 * 1024


def test_a_categorical_column_stores_fewer_bytes_than_its_text(tmp_path: Path) -> None:
    n_rows = 20_000
    categories = [f"cell type {index:02d} " + "y" * 20 for index in range(50)]
    rows = np.array(
        [categories[(index * 7) % 50] for index in range(n_rows)], dtype=object
    )
    sizes = {}
    for layout in ("text", "categorical"):
        group = zarr.open_group(tmp_path / f"{layout}.zarr", mode="w")
        group.create_array("I", data=np.ones(n_rows, dtype=bool))
        group.create_array(
            "ids", data=np.array([f"c{index}" for index in range(n_rows)])
        )
        group.create_array(
            "names", data=np.array([f"n{index}" for index in range(n_rows)])
        )
        table = MetaData(group)
        table.insert("cell_type", rows if layout == "text" else pd.Categorical(rows))
        assert table.fetch_all("cell_type").tolist() == rows.tolist()
        sizes[layout] = sum(
            path.stat().st_size
            for path in (tmp_path / f"{layout}.zarr").rglob("*")
            if path.is_file() and "cell_type" in path.relative_to(tmp_path).parts[1]
        )

    assert sizes["categorical"] * 5 < sizes["text"]


# Existing columns and stores.


def test_a_column_of_values_keeps_its_layout() -> None:
    table = _table()

    table.insert("labels", np.array(LABELS, dtype=object))
    table.insert(
        "names", pd.Categorical([f"name{index}" for index in range(N)]), overwrite=True
    )

    group = table._group
    for name in ("labels", "names"):
        array = table._get_array(name)
        assert isinstance(array, zarr.Array)
        assert array.dtype.kind == "U"
        assert "categories" not in array.attrs
    assert sorted(group.array_keys()) == [
        "I",
        "__cytearc_missing__labels",
        "ids",
        "labels",
        "names",
    ]


def test_columns_of_an_existing_store_read_as_stored(datastore) -> None:
    for table in (datastore.cells, datastore.RNA.feats):
        for column in table.columns:
            array = table._get_array(column)
            assert isinstance(array, zarr.Array)
            np.testing.assert_array_equal(
                table.fetch_all(column), np.asarray(table._group[column][:])
            )


# Writing, replacing and dropping.


def test_partial_categorical_inserts_fill_or_flag_the_other_rows() -> None:
    table = _table()
    given = ["b", "a", "b", "a", "c", "a", "b", "a"]
    assert int(SELECTED.sum()) == len(given)

    table.insert("flagged", pd.Categorical(given))
    table.insert("filled", pd.Categorical(given), fill_value="c")

    outside = ~SELECTED
    assert table._group["flagged"][:][outside].tolist() == [-1] * int(outside.sum())
    assert (
        np.asarray(metadata_missing_mask(table, "flagged")[:]).tolist()
        == outside.tolist()
    )
    assert table.fetch_all("filled")[outside].tolist() == ["c"] * int(outside.sum())
    assert metadata_missing_mask(table, "filled") is None
    with pytest.raises(ValueError, match="not a category"):
        table.insert("refused", pd.Categorical(given), fill_value="z")
    assert "refused" not in table._group


def test_a_labelled_series_of_categories_is_assigned_by_cell_id() -> None:
    table = _table()
    ids = [f"cell{index}" for index in range(N)]
    series = pd.Series(pd.Categorical(LABELS, categories=CATEGORIES), index=ids)

    table.insert("labels", series.iloc[::-1])

    assert table.fetch_all("labels").tolist() == STORED
    assert table._group["labels"][:].tolist() == CODES


def test_categories_that_read_alike_once_stored_are_refused() -> None:
    table = _table()

    with pytest.raises(ValueError, match="read alike"):
        table.insert("mixed", pd.Categorical([1, "1"] * (N // 2)))

    assert "mixed" not in table._group
    with pytest.raises(TypeError, match="needs pandas categorical values"):
        categorical_codes(np.array(["a", "b"]), name="labels")


def test_categorical_values_of_another_length_are_refused() -> None:
    table = _table()

    with pytest.raises(ValueError, match="incorrect length"):
        table.insert("labels", pd.Categorical(["a", "b", "c"]))

    assert "labels" not in table._group


def test_a_categorical_insert_over_a_nested_group_is_refused() -> None:
    table = _table()
    table._group.create_group("nested")

    with pytest.raises(TypeError, match="nested group"):
        table.insert("nested", pd.Categorical(LABELS), overwrite=True)


@pytest.mark.parametrize("first", ["text", "categorical"])
def test_replacing_and_dropping_a_column_leaves_no_linked_array(first: str) -> None:
    table = _table()
    second = "categorical" if first == "text" else "text"
    _insert(table, first)

    if second == "text":
        table.insert("labels", np.array(["kept"] * N), overwrite=True)
        expected = ["I", "ids", "labels", "names"]
    else:
        table.insert(
            "labels", pd.Categorical(LABELS, categories=CATEGORIES), overwrite=True
        )
        expected = [
            "I",
            "__cytearc_categories__labels",
            "__cytearc_missing__labels",
            "ids",
            "labels",
            "names",
        ]

    assert sorted(table._group.array_keys()) == expected
    table.drop("labels")
    assert sorted(table._group.array_keys()) == ["I", "ids", "names"]


def test_a_failed_replacement_restores_the_categorical_column(monkeypatch) -> None:
    import cytearc.storage.identity as identity

    table = _table()
    table.insert("labels", pd.Categorical(LABELS, categories=CATEGORIES, ordered=True))
    written = identity.create_metadata_column

    def failing(group, key, **options):
        if key == "labels" and options["data"].dtype.kind == "U":
            raise OSError("the store refused the write")
        return written(group, key, **options)

    monkeypatch.setattr(identity, "create_metadata_column", failing)
    with pytest.raises(OSError, match="refused"):
        table.insert("labels", np.array(["other"] * N), overwrite=True)

    group = table._group
    assert group["labels"][:].tolist() == CODES
    assert group["labels"].dtype == np.int8
    assert group["__cytearc_categories__labels"][:].tolist() == CATEGORIES
    assert group["__cytearc_categories__labels"].attrs["ordered"] is True
    assert group["__cytearc_missing__labels"][:].tolist() == MISSING
    assert table.fetch_all("labels").tolist() == STORED


def test_codes_that_name_no_category_are_refused_before_writing() -> None:
    table = _table()

    with pytest.raises(ValueError, match="outside -1 to 2"):
        replace_metadata_column(
            table._group,
            "labels",
            np.array([3] * N),
            categories=np.array(["a", "b", "c"]),
        )
    with pytest.raises(ValueError, match="names a category twice"):
        replace_metadata_column(
            table._group,
            "labels",
            np.zeros(N, dtype=int),
            categories=np.array(["a", "a"]),
        )
    with pytest.raises(TypeError, match="integer code"):
        replace_metadata_column(
            table._group, "labels", np.zeros(N), categories=np.array(["a"])
        )
    with pytest.raises(ValueError, match="NaN among its categories"):
        replace_metadata_column(
            table._group,
            "labels",
            np.zeros(N, dtype=int),
            categories=np.array([0.5, np.nan]),
        )
    with pytest.raises(ValueError, match="one-dimensional categories"):
        replace_metadata_column(
            table._group,
            "labels",
            np.zeros(N, dtype=int),
            categories=np.array([["a", "b"]]),
        )
    with pytest.raises(TypeError, match="ordered must be a bool"):
        replace_metadata_column(
            table._group,
            "labels",
            np.zeros(N, dtype=int),
            categories=np.array(["a"]),
            ordered="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="not with a separate mask"):
        replace_metadata_column(
            table._group,
            "labels",
            np.zeros(N, dtype=int),
            np.zeros(N, dtype=bool),
            categories=np.array(["a"]),
        )
    assert "labels" not in table._group
    with pytest.raises(ValueError, match="cannot have -1 categories"):
        category_codes_dtype(-1)
    # int64 holds the positions 0 to 2**63 - 1, and no signed type one more.
    assert category_codes_dtype(2**63) == np.int64
    with pytest.raises(ValueError, match="No signed integer type"):
        category_codes_dtype(2**63 + 1)


# Malformed layouts fail loudly.


def _break_link(group: zarr.Group) -> None:
    group["labels"].attrs["categories"] = "__cytearc_categories__other"


def _drop_categories(group: zarr.Group) -> None:
    del group["__cytearc_categories__labels"]


def _float_codes(group: zarr.Group) -> None:
    attributes = group["labels"].attrs.asdict()
    group.create_array(
        "labels", data=np.zeros(N), overwrite=True, attributes=attributes
    )


def _unordered(group: zarr.Group) -> None:
    del group["__cytearc_categories__labels"].attrs["ordered"]


def _code_past_the_categories(group: zarr.Group) -> None:
    group["labels"][0] = 12


def _code_below_missing(group: zarr.Group) -> None:
    group["labels"][0] = -2


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_break_link, "malformed categories link"),
        (_drop_categories, "categories array that is missing"),
        (_float_codes, "signed integer code"),
        (_unordered, "malformed categories array"),
        (_code_past_the_categories, "holds the code 12"),
        (_code_below_missing, "holds the code -2"),
    ],
)
def test_a_malformed_categorical_column_fails_when_read(corrupt, message) -> None:
    table = _table()
    _insert(table, "categorical")

    corrupt(table._group)

    with pytest.raises(ValueError, match=message):
        table.fetch_all("labels")


def test_codes_too_narrow_for_their_categories_fail_when_read() -> None:
    table = _table()
    _insert(table, "categorical")
    group = table._group
    group.create_array(
        "__cytearc_categories__labels",
        data=np.array([f"c{index}" for index in range(129)]),
        overwrite=True,
        attributes={"ordered": False},
    )

    with pytest.raises(ValueError, match="129 categories, more than its int8 codes"):
        table.fetch_all("labels")


def test_the_categories_link_is_read_from_the_group_alone() -> None:
    table = _table()
    _insert(table, "categorical")
    group = table._group

    categories = linked_categories(group, "labels")
    column = CategoricalColumn.open(group, "labels")

    assert categories is not None and categories[:].tolist() == CATEGORIES
    assert column is not None and column[:].tolist() == STORED
    assert linked_categories(group, "ids") is None
    assert CategoricalColumn.open(group, "ids") is None


def test_a_categorical_identity_column_is_refused_when_the_table_opens() -> None:
    table = _table()
    group = table._group
    replace_metadata_column(
        group,
        "names",
        np.arange(N),
        categories=np.array([f"name{index}" for index in range(N)]),
    )

    with pytest.raises(ValueError, match="'names' is stored as a categorical column"):
        MetaData(group)


def test_the_categories_prefix_is_reserved() -> None:
    table = _table()

    with pytest.raises(ValueError, match="reserves for the categories"):
        table.insert("__cytearc_categories__labels", np.ones(N))
    # Imports leave out source columns of that name.
    assert metadata_column_keys(["__cytearc_categories__x", "kept"]) == {"kept": "kept"}


# Copies.


def test_subset_repack_and_mount_keep_the_categorical_layout(
    stores, tmp_path: Path
) -> None:
    from cytearc.datastore.datastore import mount_datastore
    from cytearc.tools.repack_zarr import repack_store
    from cytearc.writers import SubsetZarr

    source = stores["categorical"]
    location = source.zarr_loc
    rows = np.array([0, 2, 5, 9, 11])
    SubsetZarr(
        str(tmp_path / "subset.zarr"), [source.RNA], cell_idx=rows, nthreads=1
    ).dump()
    repack_store(str(location), str(tmp_path / "repacked.zarr"))
    mount_datastore(
        str(location),
        at=str(tmp_path / "mounted.zarr"),
        min_features_per_cell=-1,
        nthreads=1,
    )

    for name, expected_codes in (
        ("subset", [CODES[row] for row in rows]),
        ("repacked", CODES),
        ("mounted", CODES),
    ):
        copy = open_store(tmp_path / f"{name}.zarr")
        cells = copy.zw["cellData"]
        assert cells["cell_type"][:].tolist() == expected_codes
        # Every category is copied, those no copied row holds included.
        assert cells["__cytearc_categories__cell_type"][:].tolist() == CATEGORIES
        assert cells["__cytearc_categories__cell_type"].attrs["ordered"] is False
        assert copy.cells.fetch_categorical("cell_type", key=None).codes.tolist() == (
            expected_codes
        )


def test_a_merge_writes_the_labels_of_categorical_columns(
    stores, tmp_path: Path
) -> None:
    from cytearc.merge import DataStoreMerge

    DataStoreMerge(
        [stores["text"], stores["categorical"]],
        str(tmp_path / "merged.zarr"),
        ["text", "categorical"],
        nthreads=1,
    ).dump()

    merged = open_store(tmp_path / "merged.zarr")
    assert isinstance(merged.cells._get_array("orig_cell_type"), zarr.Array)
    # A merged cell is named for its source and its cell.
    rows = {cell: row for row, cell in enumerate(merged.cells.fetch_all("ids"))}
    values = merged.cells.fetch_all("orig_cell_type")
    missing = np.asarray(metadata_missing_mask(merged.cells, "orig_cell_type")[:])
    for source in ("text", "categorical"):
        order = [rows[f"{source}__cell{index}"] for index in range(N)]
        assert values[order].tolist() == STORED
        assert missing[order].tolist() == MISSING
