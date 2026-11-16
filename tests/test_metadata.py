import warnings

import numpy as np
import pytest


@pytest.fixture
def dummy_metadata(tmp_path):
    import zarr

    from cytearc.metadata import MetaData

    fn = str(tmp_path / "dummy_metadata.zarr")
    g = zarr.open_group(fn, mode="w")
    data = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1]).astype(bool)
    g.create_array(
        "I",
        data=data,
        chunks=(100000,),
    )
    yield MetaData(g)


def test_metadata_attrs(dummy_metadata):
    assert dummy_metadata.N == 9
    np.testing.assert_array_equal(dummy_metadata.index, np.arange(9))


def test_metadata_fetch(dummy_metadata):
    dummy_metadata.insert("order", np.arange(10, 19))
    dummy_metadata.insert("keep_tail", np.arange(9) >= 6)

    np.testing.assert_array_equal(dummy_metadata.fetch("I"), [True] * 7)
    np.testing.assert_array_equal(
        dummy_metadata.fetch_all("I"), [1, 1, 1, 1, 0, 0, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        dummy_metadata.fetch("order"), [10, 11, 12, 13, 16, 17, 18]
    )
    np.testing.assert_array_equal(
        dummy_metadata.fetch("order", key="keep_tail"), [16, 17, 18]
    )
    np.testing.assert_array_equal(dummy_metadata.fetch_all("order"), np.arange(10, 19))


def test_metadata_grep_preserves_regex_character_classes(dummy_metadata):
    dummy_metadata.insert(
        "names",
        np.array(["RPS3", "RPSX", "mt-Co1", "MT1A", "g_a", "g-", "x y", "xy", "RPS4"]),
        overwrite=True,
    )

    assert dummy_metadata.grep(r"^RPS\d+$") == ["RPS3", "RPS4"]
    # Matching ignores case; the names come back as stored.
    assert dummy_metadata.grep(r"^g_\w+$") == ["g_a"]
    assert dummy_metadata.grep(r"^x\sy$") == ["x y"]
    assert dummy_metadata.grep("^MT-") == ["mt-Co1"]
    assert dummy_metadata.grep(r"^RPS\d+$", only_valid=True) == ["RPS3", "RPS4"]


def test_metadata_insert_encodes_none_as_missing_text(dummy_metadata):
    from cytearc.metadata.rows import metadata_missing_mask

    values = np.array(["a", None, "b", "", "a", "b", "a", "b", "a"], dtype=object)

    dummy_metadata.insert("group", values)

    stored = dummy_metadata.fetch_all("group")
    assert stored.tolist() == ["a", "", "b", "", "a", "b", "a", "b", "a"]
    # None is missing; the empty text that was supplied is a value.
    missing = metadata_missing_mask(dummy_metadata, "group")[:]
    assert np.flatnonzero(missing).tolist() == [1]
    frame = dummy_metadata.to_pandas_dataframe(["group"])["group"]
    assert frame.isna().tolist() == [i == 1 for i in range(9)]


def test_metadata_active_index(dummy_metadata):
    dummy_metadata.insert("tail", np.arange(9) >= 6)

    active = dummy_metadata.active_index(key="I")
    # Positions are global row indices, as integers that index other columns.
    np.testing.assert_array_equal(active, [0, 1, 2, 3, 6, 7, 8])
    assert active.dtype.kind == "i"
    np.testing.assert_array_equal(dummy_metadata.active_index(key="tail"), [6, 7, 8])


def test_metadata_partial_float_fill_does_not_cast_uninitialized_values(dummy_metadata):
    values = np.arange(7, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        filled, missing = dummy_metadata._expand_to_rows(values, "I", name="x")

    selected = dummy_metadata.fetch_all("I")
    assert filled.dtype == values.dtype
    np.testing.assert_array_equal(filled[selected], values)
    assert np.isnan(filled[~selected]).all()
    np.testing.assert_array_equal(missing, ~selected)


def _cell_table(path, scalar=None, *, beside_columns=True):
    """Return a ``cellData`` group of nine rows, with a 0-d ``scalar`` array if given."""
    import zarr

    group = zarr.open_group(str(path), mode="w").create_group("cellData")
    if beside_columns:
        group.create_array("I", data=np.ones(9, dtype=bool))
        group.create_array("ids", data=np.array([f"c{i}" for i in range(9)]))
        group.create_array("score", data=np.arange(9.0))
    if scalar is not None:
        group.create_array("scalar", data=np.asarray(scalar))
    return group


@pytest.mark.parametrize("beside_columns", [True, False])
@pytest.mark.parametrize("scalar", [5, 1.5, True, "text"])
def test_metadata_table_refuses_a_zero_dimensional_column(
    tmp_path, scalar, beside_columns
):
    import re

    from cytearc.metadata import MetaData

    group = _cell_table(tmp_path / "s.zarr", scalar, beside_columns=beside_columns)
    # A 0-d array used to fail with a bare IndexError, naming no column.
    message = (
        "Column 'scalar' of metadata table '/cellData' has shape (), but a "
        "metadata column holds one value per row"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        MetaData(group)


@pytest.mark.parametrize("table", ["cellData", "RNA/featureData"])
def test_datastore_open_names_a_zero_dimensional_metadata_column(tmp_path, table):
    import re

    import zarr

    from cytearc import DataStore
    from tests.storage_helpers import write_count_store

    path = str(tmp_path / "store.zarr")
    write_count_store(path, {"RNA": np.arange(12).reshape(4, 3)}, "uint16")
    zarr.open_group(path, mode="r+")[table].create_array("scalar", data=np.array(5))
    message = f"Column 'scalar' of metadata table '/{table}' has shape ()"
    with pytest.raises(ValueError, match=re.escape(message)):
        DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)


def test_metadata_table_opens_valid_columns_as_before(tmp_path):
    import pandas as pd

    from cytearc.metadata import MetaData

    group = _cell_table(tmp_path / "valid.zarr")
    table = MetaData(group)
    labels = pd.Categorical(["a", "b", "a", "b", "a", "b", "a", "b", "a"])
    table.insert("group", labels)
    table.update_key(np.arange(9) < 5, "I")
    # The rows that I leaves out are flagged in a missing mask.
    table.insert("value", np.arange(5.0))
    # The categories and the missing mask are arrays of other lengths and
    # layouts beside the columns; reopening counts the rows of the columns.
    reopened = MetaData(group)
    assert reopened.N == 9
    assert reopened.columns == ["I", "ids", "names", "group", "score", "value"]
    assert reopened.fetch_categorical("group", key=None).tolist() == labels.tolist()
    frame = reopened.to_pandas_dataframe(["value"])["value"]
    assert frame.isna().tolist() == [i >= 5 for i in range(9)]

    # Columns of unequal lengths are still refused as before.
    group.create_array("short", data=np.arange(4))
    with pytest.raises(ValueError, match="Not all columns are of same length"):
        MetaData(group)
