"""Metadata reads open each column once and a DataStore opens its groups once.

On object stores every metadata read is a round trip, so these tests count the
requests that tables and opens send through a store that records them, and pin
the counts exactly. Results must be those of the code that read each column
several times; reference implementations of that code are kept here.
"""

import collections
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import LocalStore, MemoryStore, WrapperStore

from cytearc.metadata import MetaData
from cytearc.metadata.membership import membership_columns
from cytearc.metadata.rows import metadata_missing_mask
from cytearc.storage.metadata_keys import assay_membership_attributes
from tests.storage_helpers import write_count_store


class _CountingStore(WrapperStore[Any]):
    """Record every read, listing, and write that reaches the wrapped store."""

    def __init__(self, store: Any, log: list[tuple[str, str]] | None = None) -> None:
        super().__init__(store)
        self.log: list[tuple[str, str]] = [] if log is None else log

    def _with_store(self, store: Any) -> "_CountingStore":
        return type(self)(store, self.log)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self.log.append(("get", key))
        return await self._store.get(key, prototype, byte_range)

    async def get_partial_values(self, prototype: Any, key_ranges: Any) -> Any:
        key_ranges = list(key_ranges)
        self.log.extend(("get", key) for key, _range in key_ranges)
        return await self._store.get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        self.log.append(("exists", key))
        return await self._store.exists(key)

    async def set(self, key: str, value: Buffer) -> None:
        self.log.append(("set", key))
        await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self.log.append(("set_if_not_exists", key))
        await self._store.set_if_not_exists(key, value)

    async def delete(self, key: str) -> None:
        self.log.append(("delete", key))
        await self._store.delete(key)

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        self.log.append(("list_dir", prefix))
        async for name in self._store.list_dir(prefix):
            yield name

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        self.log.append(("list_prefix", prefix))
        async for name in self._store.list_prefix(prefix):
            yield name

    def counts(self) -> collections.Counter[tuple[str, str]]:
        return collections.Counter(self.log)


def _wide_table() -> tuple[MetaData, _CountingStore]:
    """Return a 63-column table, two columns masked, behind a counting store."""
    store = _CountingStore(MemoryStore())
    group = zarr.open_group(store=store, mode="w")
    rows = 6
    group.create_array("I", data=np.asarray([True, False, True, True, True, False]))
    group.create_array("ids", data=np.asarray([f"c{index}" for index in range(rows)]))
    group.create_array("names", data=np.asarray([f"n{index}" for index in range(rows)]))
    for column in range(60):
        group.create_array(f"col{column:02d}", data=np.arange(rows) + column)
    for name in ("col03", "col41"):
        missing = f"__cytearc_missing__{name}"
        group.create_array(missing, data=np.arange(rows) % 3 == 0)
        group[name].attrs["missing_mask"] = missing
    table = MetaData(group)
    store.log.clear()
    return table, store


def _reference_frame(table: MetaData, columns: list[str], stop: int | None) -> Any:
    """Read the columns as the code that opened each column three times did."""
    from cytearc.metadata.queries import missing_frame_values

    data = {}
    for column in columns:
        values = np.asarray(table._get_array(column)[:stop])
        mask = metadata_missing_mask(table, column)
        data[column] = missing_frame_values(
            values, None if mask is None else mask[:stop]
        )
    return pd.DataFrame(data)


def test_columns_list_the_table_with_one_read_per_member() -> None:
    table, store = _wide_table()

    assert table.columns == [
        "I",
        "ids",
        "names",
        *(f"col{column:02d}" for column in range(60)),
    ]
    counts = store.counts()
    assert counts[("list_dir", "")] == 1
    # Each column and mask is read once, and nothing else is.
    assert sum(counts.values()) == 1 + 65
    assert max(counts.values()) == 1


def test_head_opens_each_column_and_mask_once() -> None:
    table, store = _wide_table()

    head = table.head(4)
    log = list(store.log)

    pd.testing.assert_frame_equal(head, _reference_frame(table, table.columns, 4))
    metadata = [key for (op, key) in log if key.endswith("zarr.json")]
    chunks = [key for (op, key) in log if op == "get" and not key.endswith("zarr.json")]
    # The listing, one read of every column's and mask's metadata, and one
    # chunk of each column and mask.
    assert log[0] == ("list_dir", "")
    assert len(metadata) == len(set(metadata)) == 65
    assert len(chunks) == len(set(chunks)) == 65
    assert len(log) == 1 + 65 + 65


def test_to_pandas_dataframe_opens_only_the_requested_columns() -> None:
    table, store = _wide_table()
    requested = ["col41", "ids", "col07"]

    frame = table.to_pandas_dataframe(requested, key="I")

    expected = _reference_frame(table, requested, None).reindex(table.active_index("I"))
    pd.testing.assert_frame_equal(frame, expected)
    store.log.clear()
    table.to_pandas_dataframe(requested)
    # Three columns and one mask, each read once with one chunk; no listing.
    assert sorted(store.log) == sorted(
        [
            ("get", "col41/zarr.json"),
            ("get", "__cytearc_missing__col41/zarr.json"),
            ("get", "ids/zarr.json"),
            ("get", "col07/zarr.json"),
            ("get", "col41/c/0"),
            ("get", "__cytearc_missing__col41/c/0"),
            ("get", "ids/c/0"),
            ("get", "col07/c/0"),
        ]
    )

    # A column requested twice is opened once and read for each request.
    store.log.clear()
    twice = table.to_pandas_dataframe(["col07", "col07"])
    pd.testing.assert_frame_equal(
        twice, _reference_frame(table, ["col07", "col07"], None)
    )
    assert store.log[:3] == [
        ("get", "col07/zarr.json"),
        ("get", "col07/c/0"),
        ("get", "col07/c/0"),
    ]


def test_to_pandas_dataframe_refuses_unknown_columns_before_reading() -> None:
    table, store = _wide_table()

    requested: list[Any] = ["col01", "missing", "a/b", "__cytearc_missing__col03", 5]
    with pytest.raises(KeyError) as caught:
        table.to_pandas_dataframe(requested)
    assert caught.value.args == (
        "Columns ['missing', 'a/b', '__cytearc_missing__col03', 5] do not exist "
        "in the metadata columns.",
    )
    assert not [
        key for op, key in store.log if op == "get" and not key.endswith("zarr.json")
    ]
    with pytest.raises(TypeError, match="unhashable"):
        table.to_pandas_dataframe([["col01"]])  # type: ignore[list-item]


def test_queries_keep_their_errors_for_reserved_and_nested_names() -> None:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.asarray([True, True]))
    group.create_array("ids", data=np.asarray(["a", "b"]))
    group.create_array("Baseline (ml/min)", data=np.ones(2))
    table = MetaData(group)

    # A reserved column is listed even when the table does not hold it.
    assert "names" in table.columns
    with pytest.raises(KeyError, match="names does not exist"):
        table.to_pandas_dataframe(["names"])
    with pytest.raises(KeyError, match="names does not exist"):
        table.head()
    with pytest.raises(TypeError, match="nested group named 'Baseline \\(ml'"):
        table.to_pandas_dataframe(["ids", "Baseline (ml"])
    # Columns given as a one-shot iterator are read as before: once to check
    # them, which consumes it.
    frame = table.to_pandas_dataframe(iter(["ids"]))  # type: ignore[arg-type]
    assert frame.shape == (0, 0)


def test_membership_columns_read_each_column_once() -> None:
    table, store = _wide_table()
    table._group.create_array(
        "ADT_I",
        data=np.asarray([True, True, False, True, True, True]),
        attributes=assay_membership_attributes("ADT"),
    )
    store.log.clear()

    assert membership_columns(table) == {"ADT": "ADT_I"}
    counts = store.counts()
    # The listing reads every member once, and resolving the one membership
    # column reads it five more times, as before. Each column was read again
    # for its attributes before.
    assert counts[("list_dir", "")] == 1
    assert counts[("get", "ADT_I/zarr.json")] == 1 + 5
    assert sum(counts.values()) == 1 + 66 + 5


def _small_store(path: Path) -> str:
    rng = np.random.default_rng(0)
    write_count_store(
        str(path),
        {
            "RNA": rng.poisson(2.0, size=(40, 12)),
            "ADT": rng.poisson(5.0, size=(40, 3)),
        },
        "uint16",
    )
    from cytearc import DataStore

    DataStore(str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1)
    return str(path)


def test_read_only_open_reads_assay_groups_and_counts_fewer_times(
    tmp_path: Path,
) -> None:
    from cytearc import DataStore

    location = _small_store(tmp_path / "store.zarr")
    store = _CountingStore(LocalStore(location, read_only=True))
    DataStore(store, zarr_mode="r", nthreads=1)
    counts = store.counts()

    # Before: seven reads of each assay group and five of its counts.
    for assay in ("RNA", "ADT"):
        assert counts[("get", f"{assay}/zarr.json")] == 5
        assert counts[("get", f"{assay}/counts/zarr.json")] == 4
    assert len(store.log) == 67


def test_get_cell_vals_does_not_list_the_cell_table(tmp_path: Path) -> None:
    from cytearc import DataStore

    location = _small_store(tmp_path / "store.zarr")
    store = _CountingStore(LocalStore(location, read_only=True))
    datastore = DataStore(store, zarr_mode="r", nthreads=1)
    store.log.clear()

    values = datastore.get_cell_vals("RNA", "I", "RNA_nCounts")
    log = list(store.log)

    np.testing.assert_array_equal(values, datastore.cells.fetch("RNA_nCounts"))
    # One lookup of the column replaces a listing that read every column.
    assert log == [
        ("get", "cellData/I/zarr.json"),
        ("get", "cellData/I/c/0"),
        ("get", "cellData/RNA_nCounts/zarr.json"),
        ("get", "cellData/RNA_nCounts/zarr.json"),
        ("get", "cellData/RNA_nCounts/c/0"),
        ("get", "cellData/I/zarr.json"),
        ("get", "cellData/I/c/0"),
        ("get", "cellData/RNA_nCounts/zarr.json"),
        ("get", "cellData/RNA_nCounts/zarr.json"),
    ]


@pytest.fixture
def workspace_store(tmp_path: Path) -> str:
    from cytearc import DataStore
    from cytearc.readers import CrH5Reader
    from cytearc.writers import CrToZarr
    from tests import full_path

    location = tmp_path / "workspace.zarr"
    CrToZarr(
        CrH5Reader(full_path("1K_pbmc_citeseq.h5")),
        str(location),
        workspace="ws1",
        nthreads=1,
    ).dump()
    DataStore(
        str(location),
        workspace="ws1",
        default_assay="RNA",
        assay_types={"ADT": "ADT"},
        nthreads=1,
    )
    return str(location)


def test_workspace_group_is_read_once_per_open(workspace_store: str) -> None:
    from cytearc import DataStore

    store = _CountingStore(LocalStore(workspace_store))
    datastore = DataStore(store, workspace="ws1", nthreads=1)
    # Before, each of the open's sixteen workspace accesses read the group.
    assert store.counts()[("get", "ws1/zarr.json")] == 3
    assert datastore.zw is datastore.zw

    store.log.clear()
    datastore.qc.filter(["RNA_nCounts"], [500], [None])
    # Group creation still reads it; plain workspace accesses no longer do.
    assert store.counts()[("get", "ws1/zarr.json")] == 4


def test_workspace_group_follows_the_root_and_name(workspace_store: str) -> None:
    from cytearc import DataStore

    datastore = DataStore(workspace_store, workspace="ws1", nthreads=1)
    first = datastore.zw
    assert datastore.zw is first

    datastore.z = zarr.open_group(workspace_store, mode="r")
    assert datastore.zw is not first
    assert datastore.zw.path == "ws1"
    datastore.workspace = None
    assert datastore.zw is datastore.z


def test_assay_type_written_elsewhere_is_seen_after_a_refresh(
    workspace_store: str,
) -> None:
    from cytearc import DataStore

    datastore = DataStore(workspace_store, workspace="ws1", nthreads=1)
    cached = datastore.zw
    other = zarr.open_group(workspace_store, mode="r+")["ws1"]
    other.attrs["assayTypes"] = {**other.attrs["assayTypes"], "extra": "Assay"}

    # The cached group keeps what it read, as the root group of a store does.
    assert "extra" not in datastore.zw.attrs["assayTypes"]
    datastore._refresh_workspace_group()
    assert datastore.zw is not cached
    assert datastore.zw.attrs["assayTypes"]["extra"] == "Assay"


def test_mounted_counts_root_is_the_source_and_looked_up_once(tmp_path: Path) -> None:
    from cytearc.datastore.datastore import mount_datastore

    source = _small_store(tmp_path / "source.zarr")
    mounted = mount_datastore(
        source, str(tmp_path / "target.zarr"), min_features_per_cell=-1, nthreads=1
    )
    roots = mounted.__dict__["_matrixRoots"]

    assert mounted._matrix_root_for_assay("RNA") is mounted._matrix_z
    assert roots["RNA"] == (mounted.z, mounted.workspace, mounted._matrix_z)
    values = mounted.get_cell_vals("RNA", "I", "RNA3")
    assert values.shape == (40,)


def test_counts_root_is_looked_up_once_per_root(tmp_path: Path) -> None:
    from cytearc import DataStore

    location = _small_store(tmp_path / "store.zarr")
    datastore = DataStore(location, nthreads=1)
    assert datastore._matrix_root_for_assay("RNA") is datastore.z

    store = _CountingStore(LocalStore(location))
    datastore.z = zarr.open_group(store=store, mode="r+")
    store.log.clear()
    assert datastore._matrix_root_for_assay("RNA") is datastore.z
    assert datastore._matrix_root_for_assay("RNA") is datastore.z
    assert store.log == [("get", "RNA/counts/zarr.json")]
