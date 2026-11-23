"""Partial and nullable inserts flag missing rows under the linked mask."""

import datetime
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc.metadata import MetaData
from cytearc.metadata.encoding import (
    inexact_time_rows,
    missing_placeholder,
    missing_rows,
    nullable_values,
)
from cytearc.metadata.rows import metadata_column_fingerprint, metadata_missing_mask
from cytearc.storage.selections import _stored_selection_summary
from cytearc.writers._store import write_metadata_column

_SELECTED = np.array([True, False, True])
_OUTSIDE = (~_SELECTED).tolist()
_DATES = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
_DAYS = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
_SECONDS = np.array([1, 2], dtype="timedelta64[s]")
_NANOSECONDS = np.array([1, 2], dtype="timedelta64[ns]")
_NAT = np.datetime64("NaT", "ns")


@pytest.mark.parametrize(("dtype", "expected"), [("U3", ""), ("S3", b"")])
def test_text_missing_placeholders_keep_the_column_dtype(dtype, expected):
    placeholder = missing_placeholder(np.dtype(dtype))
    assert placeholder == expected
    assert np.asarray([placeholder], dtype=dtype).dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["O", "V8"])
def test_missing_placeholders_reject_unencoded_objects_and_records(dtype):
    with pytest.raises(TypeError, match="cannot hold missing values"):
        missing_placeholder(np.dtype(dtype))


def _table() -> MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=_SELECTED)
    group.create_array("ids", data=np.array(["a", "b", "c"]))
    group.create_array("names", data=np.array(["A", "B", "C"]))
    group.create_array("every", data=np.ones(3, dtype=bool))
    return MetaData(group)


def _mask(table: MetaData, column: str) -> list[bool] | None:
    mask = metadata_missing_mask(table, column)
    return None if mask is None else np.asarray(mask[:]).tolist()


def _same(left: list, right: list) -> bool:
    return len(left) == len(right) and all(
        a == b or (pd.isna(a) and pd.isna(b)) for a, b in zip(left, right)
    )


@pytest.mark.parametrize(
    ("values", "dtype", "stored"),
    [
        (np.array(["A", "B"]), "<U1", ["A", "", "B"]),
        (np.array([b"A", b"Bc"]), "<U2", ["A", "", "Bc"]),
        (np.array(["A", "B"], dtype=object), "<U1", ["A", "", "B"]),
        (pd.Categorical(["A", "B"]), "<U1", ["A", "", "B"]),
        (np.array([3, 4]), "int64", [3, 0, 4]),
        (np.array([True, True]), "bool", [True, False, True]),
        (np.array([1.5, 2.5]), "float64", [1.5, np.nan, 2.5]),
        (np.array([1 + 2j, 3j]), "complex128", [1 + 2j, np.nan, 3j]),
        (_DATES, "<M8[ns]", [_DATES[0], _NAT, _DATES[1]]),
        (
            np.array([1, 2], dtype="timedelta64[s]"),
            "<m8[s]",
            [
                np.timedelta64(1, "s"),
                np.timedelta64("NaT", "s"),
                np.timedelta64(2, "s"),
            ],
        ),
        (pd.array([1, 2], dtype="Int64"), "int64", [1, 0, 2]),
        (pd.array([True, False], dtype="boolean"), "bool", [True, False, False]),
    ],
    ids=[
        "U1",
        "bytes",
        "object",
        "categorical",
        "int64",
        "bool",
        "float64",
        "complex",
        "datetime",
        "timedelta",
        "Int64",
        "boolean",
    ],
)
def test_a_partial_insert_flags_the_other_rows_as_missing(values, dtype, stored):
    table = _table()

    table.insert("x", values)

    assert table.get_dtype("x") == np.dtype(dtype)
    raw = table.fetch_all("x").tolist()
    assert _same(raw, list(np.asarray(stored, dtype=dtype).tolist())), raw
    assert _mask(table, "x") == _OUTSIDE
    # fetch stays raw; the rows that I selects hold the supplied values.
    assert len(table.fetch("x")) == 2
    frame = table.to_pandas_dataframe(["x"])["x"]
    assert frame.isna().tolist() == _OUTSIDE
    if np.dtype(dtype).kind in "Mm":
        # Missing datetimes and timedeltas are NaT, so the column keeps its dtype.
        assert frame.dtype == np.dtype(dtype)
    assert table.head(3)["x"].isna().tolist() == _OUTSIDE


def test_missing_supplied_values_are_flagged_in_full_and_partial_inserts():
    table = _table()

    table.insert("full", np.array(["a", None, "b"], dtype=object))
    table.insert("partial", np.array(["a", pd.NA], dtype=object))
    table.insert("nullable", pd.array([1, None], dtype="Int64"))
    table.insert("categories", pd.Categorical(["a", None, "b"]))
    table.insert("nan_text", pd.Series(["a", np.nan, "b"], dtype=object))
    table.insert(
        "strings",
        np.array(["a", None, "b"], dtype=np.dtypes.StringDType(na_object=None)),
    )

    assert table.fetch_all("full").tolist() == ["a", "", "b"]
    assert _mask(table, "full") == [False, True, False]
    assert table.fetch_all("partial").tolist() == ["a", "", ""]
    assert _mask(table, "partial") == [False, True, True]
    assert table.get_dtype("nullable") == np.int64
    assert table.fetch_all("nullable").tolist() == [1, 0, 0]
    assert _mask(table, "nullable") == [False, True, True]
    assert _mask(table, "categories") == [False, True, False]
    assert _mask(table, "nan_text") == [False, True, False]
    assert table.fetch_all("strings").tolist() == ["a", "", "b"]
    assert _mask(table, "strings") == [False, True, False]
    # A float NaN that NumPy already holds stays a value, as imports keep it.
    table.insert("real_nan", np.array([1.0, np.nan, 2.0]))
    assert _mask(table, "real_nan") is None


@pytest.mark.parametrize(
    ("values", "fill_value", "dtype", "stored"),
    [
        (np.array(["A", "B"]), "unknown", "<U7", ["A", "unknown", "B"]),
        (np.array([3, 4]), 7, "int64", [3, 7, 4]),
        (np.array([3, 4], dtype=np.uint8), np.uint8(255), "uint8", [3, 255, 4]),
        (np.array([True, True]), False, "bool", [True, False, True]),
        (np.array([1.5, 2.5]), np.nan, "float64", [1.5, np.nan, 2.5]),
        (np.array([1.5, 2.5]), 10**20, "float64", [1.5, 1e20, 2.5]),
        (np.array([1j, 2j]), 1.5, "complex128", [1j, 1.5, 2j]),
        (
            _DATES,
            np.datetime64("2021-01-01"),
            "<M8[ns]",
            [_DATES[0], np.datetime64("2021-01-01", "ns"), _DATES[1]],
        ),
        (
            _DATES,
            pd.Timestamp("2021-01-01 00:00:00.000000001"),
            "<M8[ns]",
            [_DATES[0], np.datetime64("2021-01-01T00:00:00.000000001"), _DATES[1]],
        ),
        (
            _DAYS,
            datetime.datetime(2021, 1, 1),
            "<M8[D]",
            [_DAYS[0], np.datetime64("2021-01-01", "D"), _DAYS[1]],
        ),
        (_DATES, _NAT, "<M8[ns]", [_DATES[0], _NAT, _DATES[1]]),
        # Within one second of the earliest date that nanoseconds hold,
        # 1677-09-21T00:12:43.145224193 as NumPy documents it.
        (
            _DATES,
            np.datetime64("1677-09-21T00:12:44", "s"),
            "<M8[ns]",
            [_DATES[0], np.datetime64("1677-09-21T00:12:44", "ns"), _DATES[1]],
        ),
        (
            _SECONDS,
            np.timedelta64(2, "m"),
            "<m8[s]",
            [_SECONDS[0], np.timedelta64(120, "s"), _SECONDS[1]],
        ),
        (
            _NANOSECONDS,
            pd.Timedelta(1, "ns"),
            "<m8[ns]",
            [_NANOSECONDS[0], np.timedelta64(1, "ns"), _NANOSECONDS[1]],
        ),
        (
            _SECONDS,
            pd.NaT,
            "<m8[s]",
            [_SECONDS[0], np.timedelta64("NaT", "s"), _SECONDS[1]],
        ),
        (np.array([b"A", b"B"]), "unknown", "<U7", ["A", "unknown", "B"]),
    ],
)
def test_an_explicit_fill_is_stored_as_a_real_value(values, fill_value, dtype, stored):
    table = _table()

    table.insert("x", values, fill_value=fill_value)

    assert table.get_dtype("x") == np.dtype(dtype)
    raw = table.fetch_all("x").tolist()
    assert _same(raw, list(np.asarray(stored, dtype=dtype).tolist())), raw
    assert _mask(table, "x") is None


@pytest.mark.parametrize(
    ("values", "fill_value"),
    [
        (np.array(["A", "B"]), 0),
        (np.array([3, 4]), 2.7),
        (np.array([3, 4]), True),
        (np.array([3, 4], dtype=np.int8), 300),
        (np.array([True, False]), "x"),
        (np.array([1.5, 2.5], dtype=np.float32), 1e300),
        (_DATES, "2021-01-01"),
        # A fill that the unit of the values cannot hold exactly: truncated,
        # out of range, or tied to a time zone.
        (_DAYS, datetime.datetime(2020, 1, 1, 12, 30)),
        (_DATES, datetime.datetime(3000, 1, 1)),
        (_DATES, np.datetime64("1677-09-21T00:12:43", "s")),
        (_DATES, datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)),
        (_SECONDS, np.timedelta64(1500, "ms")),
        (_NANOSECONDS, np.timedelta64(10**6, "D")),
        (_SECONDS, _DATES[0]),
        (np.array([3, 4]), np.timedelta64(5, "s")),
        (np.array([1.5, 2.5]), 10**400),
    ],
)
def test_an_explicit_fill_that_does_not_fit_is_rejected_before_writing(
    values, fill_value
):
    table = _table()
    table.insert("x", np.array([7, 8, 9]))

    with pytest.raises(ValueError, match="fill_value"):
        table.insert("x", values, fill_value=fill_value, overwrite=True)

    assert table.fetch_all("x").tolist() == [7, 8, 9]
    assert _mask(table, "x") is None


@pytest.mark.parametrize(
    ("values", "dtype", "changed"),
    [
        # NumPy documents nanoseconds from 1677-09-21T00:12:43.145224193 to
        # 2262-04-11T23:47:16.854775807.
        (
            np.asarray(
                [
                    "1677-09-21T00:12:43",
                    "1677-09-21T00:12:44",
                    "NaT",
                    "2262-04-11T23:47:16",
                    "2262-04-11T23:47:17",
                ],
                dtype="datetime64[s]",
            ),
            "datetime64[ns]",
            [True, False, False, False, True],
        ),
        # 9223372036854775807 milliseconds is the most that int64 holds.
        (
            np.asarray(
                [
                    -9223372036854776,
                    -9223372036854775,
                    9223372036854775,
                    9223372036854776,
                ],
                dtype="timedelta64[s]",
            ),
            "timedelta64[ms]",
            [True, False, False, True],
        ),
        # Weeks count from Thursday 1 January 1970: June 2000 starts on a
        # Thursday, and July on a Saturday.
        (
            np.asarray(["2000-06", "2000-07", "NaT"], dtype="datetime64[M]"),
            "datetime64[W]",
            [False, True, False],
        ),
        (
            np.asarray(["2000-06", "2000-07"], dtype="datetime64[M]"),
            "datetime64[D]",
            [False, False],
        ),
        (
            np.asarray(["1500", "2000"], dtype="datetime64[Y]"),
            "datetime64[ns]",
            [True, False],
        ),
        # 1500 milliseconds lie between seconds, and 2000 do not.
        (
            np.asarray([1500, 2000, "NaT", -2500], dtype="timedelta64[ms]"),
            "timedelta64[s]",
            [True, False, False, True],
        ),
        # An attosecond lies between weeks.
        (
            np.asarray([0, 1, -1], dtype="timedelta64[as]"),
            "timedelta64[W]",
            [False, True, True],
        ),
        # NumPy refuses to convert any year to attoseconds, so each year is
        # read on its own and is changed, and NaT is not.
        (
            np.asarray(["1970", "NaT", "2000"], dtype="datetime64[Y]"),
            "datetime64[as]",
            [True, False, True],
        ),
        # Units of several steps: 30 and 10 seconds are both whole steps of 5.
        (np.asarray([3, 1], dtype="datetime64[10s]"), "datetime64[5s]", [False, False]),
    ],
    ids=[
        "seconds to nanoseconds",
        "seconds to milliseconds",
        "milliseconds to seconds",
        "months to weeks",
        "months to days",
        "years to nanoseconds",
        "attoseconds to weeks",
        "years to attoseconds",
        "steps of 10 to 5 seconds",
    ],
)
def test_inexact_time_rows_are_those_that_a_unit_cannot_hold(values, dtype, changed):
    import datetime

    assert datetime.date(2000, 6, 1).weekday() == datetime.date(1970, 1, 1).weekday()
    assert datetime.date(2000, 7, 1).weekday() == 5
    assert inexact_time_rows(values, np.dtype(dtype)).tolist() == changed


def test_an_explicit_fill_is_not_needed_when_values_cover_every_row():
    table = _table()

    table.insert("x", np.array([True, False, True]), fill_value=np.nan)

    assert table.fetch_all("x").tolist() == [True, False, True]
    assert _mask(table, "x") is None


def test_reserved_columns_hold_no_missing_values():
    table = _table()
    values = np.array([True, True])

    for supplied in (values, np.array([True, None, True], dtype=object)):
        with pytest.raises(ValueError, match="cannot hold missing values"):
            table.insert("I", supplied, overwrite=True, force=True)
    assert table.fetch_all("I").tolist() == [True, False, True]

    table.insert("I", values, fill_value=False, overwrite=True, force=True)
    assert _mask(table, "I") is None


def test_update_key_keeps_an_explicit_false_without_a_mask():
    table = _table()

    table.update_key(np.array([True, False]), "I")
    assert table.fetch_all("I").tolist() == [True, False, False]
    assert _mask(table, "I") is None

    table.update_key(np.array([True, False, True]), "every")
    assert table.fetch_all("every").tolist() == [True, False, True]
    # A key is only restricted: True never selects a row again.
    table.update_key(np.array([True, True, True]), "every")
    assert table.fetch_all("every").tolist() == [True, False, True]
    with pytest.raises(TypeError, match="booleans"):
        table.update_key(np.array([1, 0, 1]), "every")


def test_a_masked_boolean_key_selects_what_an_explicit_false_selects():
    table = _table()

    table.insert("masked", np.array([True, True]))
    table.insert("filled", np.array([True, True]), fill_value=False)

    assert _mask(table, "masked") == _OUTSIDE
    np.testing.assert_array_equal(table.active_index("masked"), [0, 2])
    np.testing.assert_array_equal(
        table.fetch("ids", key="masked"), table.fetch("ids", key="filled")
    )
    # Stored selections read raw values, so their identity does not change.
    assert _stored_selection_summary(
        table._get_array("masked")
    ) == _stored_selection_summary(table._get_array("filled"))
    # Column fingerprints cover the mask, so they differ.
    assert metadata_column_fingerprint(table, "masked") != metadata_column_fingerprint(
        table, "filled"
    )


def test_nullable_values_follow_the_import_contract():
    values, missing = nullable_values(
        np.array([1, None, 2.5, pd.NA], dtype=object), name="x"
    )
    assert values.dtype == np.float64
    assert _same(values.tolist(), [1.0, np.nan, 2.5, np.nan])
    assert missing.tolist() == [False, True, False, True]
    for shaped in (np.zeros((2, 2)), np.float64(1.0)):
        with pytest.raises(ValueError, match="one value per row"):
            nullable_values(shaped, name="x")
    # A bool is no integer and a timedelta no number, so these share no
    # number dtype and are stored as text, as values that are all missing are.
    for supplied, text in (
        ([True, 2, None], ["True", "2", ""]),
        ([np.timedelta64(1, "s"), None], ["1 seconds", ""]),
        ([None, None], ["", ""]),
    ):
        values, _ = nullable_values(np.array(supplied, dtype=object), name="x")
        assert (values.dtype.kind, values.tolist()) == ("U", text)
    values, missing = nullable_values(np.array([2j, None], dtype=object), name="x")
    assert values.dtype == np.complex128
    assert missing.tolist() == [False, True]
    with pytest.raises(ValueError, match="Column 'x' holds text that is not valid"):
        nullable_values(np.array([b"\xff", None], dtype=object), name="x")
    assert missing_rows(
        np.array([None, np.nan, pd.NA, pd.NaT, "", [1]], dtype=object)
    ).tolist() == [True, True, True, True, False, False]
    assert not missing_rows(np.array([np.nan, 1.0])).any()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([2**63, 1, None], "holds integers outside the int64 range"),
        ([10**400, 1.5, None], "holds numbers too large for float64"),
        pytest.param(
            np.array([np.longdouble("1e400"), None], dtype=object),
            "holds numbers too large for float64",
            marks=pytest.mark.skipif(
                np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
                reason="this platform's long double is a float64",
            ),
        ),
    ],
    ids=["above_int64", "beyond_float64", "long_double_beyond_float64"],
)
def test_numbers_that_their_shared_dtype_cannot_hold_are_rejected(values, message):
    with pytest.raises(ValueError, match=f"Column 'x' {message}"):
        nullable_values(values, name="x")
    table = _table()
    with pytest.raises(ValueError, match=f"Column 'x' {message}"):
        table.insert("x", values)
    assert "x" not in table.columns


def test_lookups_never_match_a_missing_row():
    table = _table()
    table.insert("x", np.array([0, None, 5], dtype=object))
    table.insert("last_two", np.array([False, True, True]))

    # The missing row stores the placeholder 0.
    assert table.fetch_all("x").tolist() == [0, 0, 5]
    assert table.get_index_by([0], "x").tolist() == [0]
    assert table.get_index_by([0, 5], "x", key="last_two").tolist() == [1]


def test_integers_at_the_int64_limits_keep_int64():
    values, missing = nullable_values([2**63 - 1, -(2**63), None], name="x")

    assert values.dtype == np.int64
    assert values.tolist() == [2**63 - 1, -(2**63), 0]
    assert missing.tolist() == [False, False, True]


def test_imports_flag_every_kind_of_missing_object_value():
    group = zarr.open_group(store=MemoryStore(), mode="w")

    write_metadata_column(
        group, "label", np.array(["a", pd.NA, None, pd.NaT], dtype=object)
    )
    write_metadata_column(group, "when", _DATES, missing=np.array([False, True]))

    assert group["label"][:].tolist() == ["a", "", "", ""]
    assert group["__cytearc_missing__label"][:].tolist() == [False, True, True, True]
    assert np.isnat(group["when"][:]).tolist() == [False, True]
    assert group["when"][0] == _DATES[0]


# Two arrays of 1,500 integers that differ at one position, which NumPy prints
# alike, as their first and last three values.
_LONG = np.arange(1_500)
_CHANGED = np.where(_LONG == 700, -1, _LONG)
# How the refusal of an insert of lists, tuples or arrays ends, as written by
# hand.
_STORE_AS_TEXT = (
    "To store such values as text, pass text that keeps every element, such as "
    'values.map(lambda v: ",".join(map(str, v)), na_action="ignore")'
)


def _collections_store(tmp_path):
    from tests.storage_helpers import open_store, write_count_store

    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((4, 4))}, "uint16")
    return open_store(path)


def _refused(
    value: str,
    position: int,
    *,
    item: str = "row",
    name: str = "column",
    suggestion: str | None = None,
) -> str:
    """The refusal of a column whose ``position`` holds ``value``, as written by hand."""
    refusal = (
        f"Column {name!r} needs one value per {item}, not a list, tuple, set, dict, "
        f"array or other collection; {item} {position} holds {value}"
    )
    return refusal if suggestion is None else f"{refusal}. {suggestion}"


def _objects(*values: object) -> np.ndarray:
    """Hold each value as one object, as an object column holds it."""
    held = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        held[position] = value
    return held


def _zero_dimensional(value: object) -> np.ndarray:
    """Hold ``value`` in an array of no dimension."""
    held = np.empty((), dtype=object)
    held[()] = value
    return held


def _within_a_minute(call: Callable[[], Any]) -> Any:
    """Return what ``call`` returns, and fail the test if it runs for a minute."""
    outcome: list = []

    def run() -> None:
        try:
            outcome.append((True, call()))
        except BaseException as error:
            outcome.append((False, error))

    # The [()] of NumPy's masked constant returns itself, so an unwrapping
    # of zero-dimensional arrays that does not stop there never ends.
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "the call never ended"
    returned, result = outcome[0]
    if not returned:
        raise result
    return result


_SERIES = pd.Series([1, 2])
_GENERATOR = (value for value in range(2))
_SQUARE = np.full((2, 2), 0.1 + 0.2)


@pytest.mark.parametrize(
    ("values", "row", "held", "suggestion"),
    [
        # Text can join the elements of lists, tuples and arrays.
        pytest.param(
            _objects([1, 2], [3], [1, 2, 3], [4]),
            1,
            "[1, 2]",
            _STORE_AS_TEXT,
            id="lists",
        ),
        pytest.param(
            _objects((1, 0.5), (2, 1.5), None, (4, 3.5)),
            1,
            "(1, 0.5)",
            _STORE_AS_TEXT,
            id="tuples-and-a-missing-row",
        ),
        pytest.param(
            _objects(_LONG, _CHANGED, _LONG, _CHANGED),
            1,
            repr(_LONG),
            _STORE_AS_TEXT,
            id="long-arrays",
        ),
        # NumPy prints the rows of a deeper array rounded, and cannot iterate
        # a zero-dimensional one.
        pytest.param(
            _objects(_SQUARE, _SQUARE, None, _SQUARE),
            1,
            repr(_SQUARE),
            None,
            id="arrays-of-two-dimensions",
        ),
        pytest.param(
            _objects(*[_zero_dimensional([1, 2])] * 3, None),
            1,
            "[1, 2]",
            None,
            id="lists-in-zero-dimensional-arrays",
        ),
        # Joining would split text, or keep only the keys of a dict.
        pytest.param(
            _objects("a", {"b"}, "c", None), 2, "{'b'}", None, id="a-set-among-text"
        ),
        pytest.param(
            _objects(1, 2.5, None, {"d": 4}),
            4,
            "{'d': 4}",
            None,
            id="a-dict-among-numbers",
        ),
        # The first list is refused, and the dict withholds the suggestion.
        pytest.param(
            _objects([1], [2], {"k": 1}, [3]),
            1,
            "[1]",
            None,
            id="a-dict-among-lists",
        ),
        pytest.param(
            _objects("a", "b", frozenset({3}), "d"),
            3,
            "frozenset({3})",
            None,
            id="a-frozenset",
        ),
        pytest.param(
            _objects("a", range(3), "c", "d"), 2, "range(0, 3)", None, id="a-range"
        ),
        pytest.param(
            _objects("a", "b", deque([1]), "d"), 3, "deque([1])", None, id="a-deque"
        ),
        pytest.param(
            _objects(_SERIES, "b", "c", "d"),
            1,
            repr(_SERIES),
            None,
            id="a-series-per-row",
        ),
        pytest.param(
            _objects("a", {"k": 1}.keys(), "c", "d"),
            2,
            "dict_keys(['k'])",
            None,
            id="a-dict-view",
        ),
        pytest.param(
            _objects("a", "b", "c", _GENERATOR),
            4,
            repr(_GENERATOR),
            None,
            id="a-generator",
        ),
    ],
)
def test_inserts_refuse_collections_and_write_nothing(
    tmp_path, values, row, held, suggestion
):
    store = _collections_store(tmp_path)
    ids = store.cells.fetch_all("ids")
    store.cells.insert("column", pd.Series(["w", "x", "y", "z"], index=ids))
    store.RNA.feats.insert("column", np.array(["w", "x", "y", "z"]))
    cells, features = sorted(store.cells.columns), sorted(store.RNA.feats.columns)
    supplied = pd.Series(values, dtype=object)
    for insert in (
        lambda name, **options: store.cells.insert(
            name, supplied.set_axis(ids), **options
        ),
        lambda name, **options: store.RNA.feats.insert(name, supplied, **options),
    ):
        # A new column, and one that the insert would replace.
        for name, options in (("new", {}), ("column", {"overwrite": True})):
            with pytest.raises(ValueError) as refused:
                insert(name, **options)
            # Each value used to be stored as its printed text, and the two
            # long arrays as one text.
            assert str(refused.value) == _refused(
                held, row, name=name, suggestion=suggestion
            )
    assert sorted(store.cells.columns) == cells
    assert sorted(store.RNA.feats.columns) == features
    assert store.cells.fetch_all("column").tolist() == ["w", "x", "y", "z"]
    assert store.RNA.feats.fetch_all("column").tolist() == ["w", "x", "y", "z"]


def test_inserts_refuse_lists_of_unequal_lengths_and_categories_of_collections(
    tmp_path,
):
    store = _collections_store(tmp_path)
    # NumPy cannot shape lists of unequal lengths, and used to refuse them
    # with a message of its own.
    with pytest.raises(ValueError) as refused:
        store.RNA.feats.insert("lists", [[1, 2], [3], [1, 2, 3], [4]])
    assert str(refused.value) == _refused(
        "[1, 2]", 1, name="lists", suggestion=_STORE_AS_TEXT
    )
    # Categories are typed as values are, and used to be stored as text.
    for insert, values in (
        (store.RNA.feats.insert, pd.Categorical([(1, 2), (3, 4), (1, 2), (3, 4)])),
        (
            store.cells.insert,
            pd.Series(
                pd.Categorical([(1, 2), (3, 4), (1, 2), (3, 4)]),
                index=store.cells.fetch_all("ids"),
            ),
        ),
    ):
        with pytest.raises(ValueError) as refused:
            insert("pairs", values)
        assert str(refused.value) == _refused(
            "(1, 2)", 1, item="category", name="pairs", suggestion=_STORE_AS_TEXT
        )
    assert "lists" not in store.RNA.feats.columns
    assert "pairs" not in store.RNA.feats.columns
    assert "pairs" not in store.cells.columns


def test_update_key_refuses_collections_with_no_suggestion_of_text(tmp_path):
    store = _collections_store(tmp_path)
    ids = store.cells.fetch_all("ids")
    selected = store.cells.fetch_all("I").tolist()
    with pytest.raises(ValueError) as refused:
        store.cells.update_key(
            pd.Series(_objects([True], [False], [True], [True]), index=ids), "I"
        )
    # A key holds booleans, so text would be no remedy.
    assert str(refused.value) == _refused("[True]", 1, name="I")
    assert store.cells.fetch_all("I").tolist() == selected


@pytest.mark.parametrize(
    "categorical", [False, True], ids=["values", "categorical-values"]
)
def test_the_suggestion_of_a_refused_insert_keeps_every_element(tmp_path, categorical):
    store = _collections_store(tmp_path)
    values = pd.Series(
        _objects(_LONG, _CHANGED, [0.1 + 0.2, 2.5], None),
        index=store.cells.fetch_all("ids"),
    )
    if categorical:
        values = pd.Series(
            pd.Categorical([(1, 2), (3, 4), None, (1, 2)]), index=values.index
        )
    with pytest.raises(ValueError) as refused:
        store.cells.insert("joined", values)
    # The code that the refusal suggests, run as it reads.
    suggestion = str(refused.value).rsplit("such as ", 1)[1]
    store.cells.insert("joined", eval(suggestion, {}, {"values": values}))
    stored = store.cells.fetch_all("joined").tolist()
    if categorical:
        assert stored == ["1,2", "3,4", "", "1,2"]
    else:
        # The two long arrays stay apart, and floats keep every digit.
        assert stored[0].split(",") == [str(value) for value in range(1_500)]
        assert stored[1].split(",")[699:702] == ["699", "-1", "701"]
        assert stored[2] == "0.30000000000000004,2.5"
    # A missing row stays missing.
    missing = (
        [False, False, True, False] if categorical else [False, False, False, True]
    )
    assert _mask(store.cells, "joined") == missing


def test_a_zero_dimensional_array_is_stored_as_the_value_it_holds(tmp_path):
    store = _collections_store(tmp_path)
    ids = store.cells.fetch_all("ids")
    nested = _zero_dimensional(np.array(4))
    store.cells.insert(
        "numbers", pd.Series(_objects(np.array(1.5), 2.5, None, nested), index=ids)
    )
    store.cells.insert(
        "labels",
        pd.Series(_objects(np.array("a"), "b", np.array("cc"), "d"), index=ids),
    )
    # Each used to be stored as its printed text, such as "1.5".
    np.testing.assert_array_equal(
        store.cells.fetch_all("numbers"), [1.5, 2.5, np.nan, 4.0]
    )
    assert store.cells.fetch_all("labels").tolist() == ["a", "b", "cc", "d"]
    # A list that a zero-dimensional array holds is refused as a list.
    held = _zero_dimensional([1, 2])
    with pytest.raises(ValueError) as refused:
        store.cells.insert("lists", pd.Series(_objects("a", held, "c", "d"), index=ids))
    assert str(refused.value) == _refused("[1, 2]", 2, name="lists")


@pytest.mark.parametrize(
    ("values", "stored", "missing"),
    [
        # A zero-dimensional NaN used to be stored as the text 'nan', and the
        # numbers beside it as text too.
        (
            _objects(np.array(np.nan), 1.0, 2.0),
            [np.nan, 1.0, 2.0],
            [True, False, False],
        ),
        # A zero-dimensional None used to be stored as the text 'None'.
        (
            _objects(np.array(None, dtype=object), "a", "b"),
            ["", "a", "b"],
            [True, False, False],
        ),
        # NumPy's masked constant used to be stored as the text '--'.
        (_objects(np.ma.masked, 1.0, 2.0), [np.nan, 1.0, 2.0], [True, False, False]),
    ],
    ids=["nan", "none", "masked"],
)
def test_missing_values_in_zero_dimensions_are_missing(values, stored, missing):
    typed, flagged = _within_a_minute(lambda: nullable_values(values, name="x"))
    np.testing.assert_array_equal(typed, stored)
    assert flagged.tolist() == missing
    assert _within_a_minute(lambda: missing_rows(values)).tolist() == missing


def test_an_insert_of_a_masked_arrays_values_ends_with_its_masked_rows_missing(
    tmp_path,
):
    store = _collections_store(tmp_path)
    ids = store.cells.fetch_all("ids")
    masked = np.ma.masked_array([1.5, 2.5, 3.5, 4.5], mask=[False, True, False, False])
    # Its second value is the masked constant, which used to be stored as
    # the text '--', and the column as text.
    _within_a_minute(
        lambda: store.cells.insert("scores", pd.Series(list(masked), index=ids))
    )
    np.testing.assert_array_equal(
        store.cells.fetch_all("scores"), [1.5, np.nan, 3.5, 4.5]
    )
    assert _mask(store.cells, "scores") == [False, True, False, False]


@pytest.mark.parametrize(
    ("values", "stored"),
    [
        # Text that reads like a list stays text, and so do mixed values.
        (["[1, 2]", "(3,)", None, "{}"], ["[1, 2]", "(3,)", "", "{}"]),
        (["a", 1, 2.5, None], ["a", "1", "2.5", ""]),
        ([b"a", "b", None, "c"], ["a", "b", "", "c"]),
        # Python iterates text, which is no collection, bytearray included.
        ([bytearray(b"a"), "b", None, "c"], ["a", "b", "", "c"]),
        ([None, None, None, None], ["", "", "", ""]),
    ],
)
def test_text_and_mixed_scalars_are_still_stored_as_text(values, stored):
    array, missing = nullable_values(np.array(values, dtype=object), name="x")
    assert array.tolist() == stored
    assert missing.tolist() == [value is None for value in values]


def test_imports_refuse_collections_before_they_write_the_column():
    from cytearc.writers._store import write_categorical_column

    group = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ValueError) as refused:
        write_metadata_column(
            group, "column", _objects(np.arange(2), np.arange(3), np.arange(2))
        )
    assert str(refused.value) == _refused("array([0, 1])", 1)
    # Either layout of a categorical names its category, as the default
    # import of categoricals used to store tuples as text.
    categories = _objects((1, 2), (3, 4))
    for write in (
        lambda: write_categorical_column(
            group, "pairs", np.array([0, 1, 0]), categories, ordered=False
        ),
        lambda: write_metadata_column(
            group, "pairs", np.array([0, 1, 0]), categories=categories
        ),
    ):
        with pytest.raises(ValueError) as refused:
            write()
        assert str(refused.value) == _refused(
            "(1, 2)", 1, item="category", name="pairs"
        )
    assert list(group.keys()) == []
