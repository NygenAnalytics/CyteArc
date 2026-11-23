"""Merge, repack, assay schema, and count identity paths."""

import re
import runpy
from functools import partial
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc.merge.datasets as merge_datasets
from cytearc import DataStore, mount_datastore
from cytearc.assay import ADTassay, ATACassay, RNAassay
from cytearc.merge import DataStoreMerge
from cytearc.merge.features import align_features
from cytearc.merge.row_plan import iter_row_plan_segments
from cytearc.metadata import MetaData
from cytearc.metadata.membership import membership_attributes
from cytearc.storage.arrays import create_metadata_column
from cytearc.storage.count_matrix import CountMatrixPolicy, create_product_counts_array
from cytearc.storage.identity import CountSummary, finalize_counts
from cytearc.storage.stores import MATRIX_SOURCE_ATTR, load_zarr
from cytearc.tools import repack_zarr
from cytearc.tools.repack_zarr import repack_store
from cytearc.utils import digest
from cytearc.utils.logging import logger
from cytearc.writers import SubsetZarr, create_zarr_count_assay
from cytearc.writers._store import write_membership_column
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import (
    by_cell_id,
    drop_row_nonzero,
    finalize_test_counts,
    write_count_store,
)

_NAMES = ["left", "right"]
_COUNTS = np.random.default_rng(7).poisson(2.0, size=(6, 3)).astype(np.uint16)
_MOUNTED = {MATRIX_SOURCE_ATTR: {"location": "elsewhere"}}


def _write_cells(root: zarr.Group, n_cells: int) -> None:
    """Write cell metadata whose chunks span the cells, as a small import has."""
    cells = root.create_group("cellData")
    ids = np.asarray([f"c{index}" for index in range(n_cells)])
    for column, values in (
        ("ids", ids),
        ("names", ids),
        ("I", np.ones(n_cells, dtype=bool)),
    ):
        create_metadata_column(
            cells, column, data=values, dtype=values.dtype, chunkSize=n_cells
        )


def _write_source(
    location: str | MemoryStore,
    counts: Mapping[str, np.ndarray],
    *,
    ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    names: Sequence[str] | None = None,
    dtype: str = "uint16",
    policy: CountMatrixPolicy | None = None,
) -> str | MemoryStore:
    """Write a fresh import whose assays share one set of cells.

    ``ids`` and ``names`` describe the features of every assay; by default
    they are ``f"{assay}{index}"``. ``ids`` can instead map an assay to its
    own IDs, which other assays then keep by default. ``policy`` sets the
    layout of the counts, the default layout when None.
    """
    root = load_zarr(location, mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    _write_cells(root, n_cells)
    for assay, values in counts.items():
        assay_ids = ids.get(assay) if isinstance(ids, Mapping) else ids
        feature_ids = (
            list(assay_ids)
            if assay_ids is not None
            else [f"{assay}{index}" for index in range(values.shape[1])]
        )
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            n_cells,
            feature_ids,
            feature_ids if names is None else list(names),
            dtype,
            policy=policy,
        )
        array[:] = values
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)
    return location


def _open(location: str | MemoryStore) -> DataStore:
    return DataStore(location, default_assay="RNA", min_features_per_cell=0, nthreads=1)


def _named_sources(values: np.ndarray, dtype: str) -> list[DataStore]:
    """Open sources whose features share names: A, A, B on the left, A, B, B."""
    return [
        _open(_write_source(MemoryStore(), {"RNA": values}, names=names, dtype=dtype))
        for names in (["A", "A", "B"], ["A", "B", "B"])
    ]


def _merge(sources: Sequence[Any], destination: Any, **options: Any) -> DataStoreMerge:
    return DataStoreMerge(
        list(sources), destination, list(_NAMES), seed=0, nthreads=1, **options
    )


def _copy(location: str, tmp_path: Path) -> str:
    destination = str(tmp_path / Path(location).name)
    shutil.copytree(location, destination)
    return destination


def _merged_counts(location: str, path: str = "RNA/counts") -> np.ndarray:
    return np.asarray(zarr.open_group(location, mode="r")[path][:])


def _rows_by_cell(location: str, path: str = "RNA/counts") -> dict[str, list]:
    """Return merged rows keyed by merged cell ID."""
    root = zarr.open_group(location, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    return {
        cell: row.tolist()
        for cell, row in zip(ids, np.asarray(root[path][:]), strict=True)
    }


def _source_rows(values_by_name: Mapping[str, np.ndarray]) -> dict[str, list]:
    """Return the expected merged rows of sources whose cells are c0, c1, ..."""
    return {
        f"{name}__c{index}": row.tolist()
        for name, values in values_by_name.items()
        for index, row in enumerate(values)
    }


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _before_opening(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[zarr.Group], None]
) -> None:
    """Change the destination between inspection and opening, as another writer could."""
    original = DataStoreMerge._open_destination

    def open_destination(self, manifest, inspection):
        change(zarr.open_group(self.zarr_path, mode="a"))
        return original(self, manifest, inspection)

    monkeypatch.setattr(DataStoreMerge, "_open_destination", open_destination)


def _empty(root: zarr.Group) -> None:
    for key in list(root.keys()):
        del root[key]
    root.attrs.put({})


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> list[DataStore]:
    """Two prepared sources of the same cells, which merges only read."""
    root = tmp_path_factory.mktemp("sources")
    return [
        _open(_write_source(str(root / f"{name}.zarr"), {"RNA": _COUNTS + offset}))
        for offset, name in enumerate(_NAMES)
    ]


@pytest.fixture(scope="module")
def merged(tmp_path_factory: pytest.TempPathFactory, sources) -> str:
    """A completed merge of ``sources``; tests change copies of it."""
    location = str(tmp_path_factory.mktemp("merged") / "merged.zarr")
    _merge(sources, location).dump()
    return location


@pytest.fixture(scope="module")
def merged_in_workspace(tmp_path_factory: pytest.TempPathFactory, sources) -> str:
    """A completed merge of ``sources`` into workspace ``ws``."""
    location = str(tmp_path_factory.mktemp("merged_ws") / "merged.zarr")
    _merge(sources, location, out_workspace="ws").dump()
    return location


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A prepared single-assay store; tests change copies of it."""
    location = str(tmp_path_factory.mktemp("prepared") / "prepared.zarr")
    _open(_write_source(location, {"RNA": _COUNTS}))
    return location


# Count identity and assay schema


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_row_summary_kernel_rejects_non_finite_counts(value):
    values = np.array([[value, 1.0], [0.0, 2.0]])
    outputs = (
        np.zeros((2, 2), dtype=np.uint64),
        np.zeros(2),
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
    )
    # The first nonzero value is checked before it is mixed into a digest.
    with pytest.raises(ValueError, match="must hold finite values"):
        digest.summarize_rows.py_func(values, values.view(np.uint64), *outputs)


def test_count_summary_refuses_blocks_and_windows_it_cannot_digest():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts = root.create_array("counts", shape=(4, 2), dtype=np.uint16)
    summary = CountSummary(counts)
    # The compiled kernel indexes rows and columns without bounds checks.
    with pytest.raises(ValueError, match="block has the wrong shape"):
        summary.update(0, np.ones((1, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="rows are outside their window"):
        summary.update(3, np.ones((2, 2), dtype=np.uint16))

    # A fingerprint over rows that were never summarized would be wrong.
    summary.update(0, np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(RuntimeError, match="does not cover its row window"):
        finalize_counts(counts, summary=summary)
    window = CountSummary(counts, rows=(0, 2))
    window.update(0, np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(RuntimeError, match="covers only part of the matrix"):
        finalize_counts(counts, summary=window)
    assert "complete" not in counts.attrs


@pytest.mark.parametrize(
    ("name", "n_cells", "message"),
    [
        ("", 2, "Assay names must be non-empty"),
        ("  ", 2, "Assay names must be non-empty"),
        ("a/b", 2, "must not contain path separators"),
        ("a\\b", 2, "must not contain path separators"),
        ("RNA", -1, "Assay dimensions must be non-negative"),
    ],
)
def test_count_assay_creation_rejects_invalid_names_and_cell_counts(
    name, n_cells, message
):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ValueError, match=message):
        create_zarr_count_assay(root, name, None, n_cells, ["g0"], ["g0"], "uint8")
    assert list(root.group_keys()) == []


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "Count summaries do not match the counts"),
        ("fingerprint", "Count summaries do not match the counts"),
        ("length", "Count summaries are malformed"),
    ],
)
def test_first_preparation_refuses_count_summaries_of_other_counts(
    tmp_path, damage, message
):
    location = str(tmp_path / "import.zarr")
    _write_source(location, {"RNA": _COUNTS})
    matrix = zarr.open_group(location, mode="r+")["RNA"]
    if damage == "missing":
        del matrix["countSummaries"]
    elif damage == "fingerprint":
        matrix["countSummaries"].attrs["source_fingerprint"] = "0" * 64
    else:
        del matrix["countSummaries/rowSums"]
        matrix["countSummaries"].create_array("rowSums", data=np.zeros(2))

    with pytest.raises(ValueError, match=message):
        _open(location)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("unprepared", "is not prepared"),
        ("percentages", "Percentage definitions are malformed"),
        ("pattern", "Required percentage 'RNA_percentX' is missing"),
        ("column", "Required column 'RNA_nCounts' is inconsistent"),
        ("identity", "Copied dataset identity differs from the source"),
    ],
)
def test_repack_refuses_a_source_with_a_damaged_preparation_record(
    tmp_path, prepared, damage, message
):
    if damage == "unprepared":
        source = str(tmp_path / "import.zarr")
        _write_source(source, {"RNA": _COUNTS})
    else:
        source = _copy(prepared, tmp_path)
        root = zarr.open_group(source, mode="r+")
        if damage == "percentages":
            root["RNA"].attrs["percentFeatures"] = ["RNA_percentX"]
        elif damage == "pattern":
            root["RNA"].attrs["percentFeatures"] = {"RNA_percentX": 5}
        elif damage == "column":
            del root["cellData/RNA_nCounts"]
            root["cellData"].create_array(
                "RNA_nCounts", data=np.asarray(["x"] * _COUNTS.shape[0])
            )
        else:
            # The recorded identity is not recomputed until the copy publishes.
            root["RNA"].attrs["dataset_fingerprint"] = "0" * 64
    output = tmp_path / "repacked.zarr"

    with pytest.raises(ValueError, match=message):
        repack_store(source, str(output))
    assert not output.exists()


def test_repack_refuses_counts_stored_in_a_non_numeric_dtype(tmp_path):
    source = str(tmp_path / "foreign.zarr")
    root = load_zarr(source, mode="w")
    _write_cells(root, 2)
    counts = create_zarr_count_assay(root, "RNA", None, 2, ["g0"], ["g0"], "complex64")
    counts[:] = 1
    # A foreign writer marked them finalized; no CyteArc writer can summarize them.
    counts.attrs.update({"complete": True, "content_fingerprint": "0" * 64})
    output = tmp_path / "rebuilt.zarr"

    with pytest.raises(ValueError, match="two-dimensional numeric array"):
        repack_store(source, str(output), data_only=True)
    assert not output.exists()


def test_only_cell_and_feature_tables_protect_prepared_columns(tmp_path, prepared):
    root = zarr.open_group(_copy(prepared, tmp_path), mode="r+")
    n_cells = _COUNTS.shape[0]
    with pytest.raises(ValueError, match="belongs to prepared data"):
        MetaData(root["cellData"]).insert(
            "RNA_nCounts", np.zeros(n_cells), overwrite=True
        )
    # RNA measured every cell, so it has no membership column, but the cell
    # table still reserves the name.
    with pytest.raises(ValueError, match=_RESERVED.format(assay="RNA")):
        MetaData(root["cellData"]).insert("RNA_I", np.ones(n_cells, dtype=bool))
    assert "RNA_I" not in root["cellData"]

    annotations = root.create_group("annotations")
    for column, values in (
        ("ids", np.asarray([f"c{index}" for index in range(n_cells)])),
        ("I", np.ones(n_cells, dtype=bool)),
        ("RNA_nCounts", np.zeros(n_cells)),
        ("RNA_I", np.ones(n_cells, dtype=bool)),
    ):
        create_metadata_column(annotations, column, data=values, dtype=values.dtype)
    table = MetaData(annotations)
    table.insert("RNA_nCounts", np.ones(n_cells), overwrite=True)
    np.testing.assert_array_equal(table.fetch_all("RNA_nCounts"), np.ones(n_cells))
    # Only a cell table reserves the membership names of the store's assays.
    table.insert("RNA_I", np.zeros(n_cells, dtype=bool), overwrite=True)
    table.drop("RNA_I")
    features = MetaData(root["RNA/featureData"])
    features.insert("RNA_I", np.ones(_COUNTS.shape[1], dtype=bool))
    features.drop("RNA_I")


_RESERVED = "Cell column '{assay}_I' is reserved for the membership of assay '{assay}'"
_UNDROPPABLE = "Cell column '{assay}_I' records which cells assay '{assay}' measured"
# Cells c1 and c4 were not measured with ADT, so their ADT counts are zero.
_ADT_MEMBERS = np.asarray([True, False, True, True, False, True])


def _partial_adt_source(location: str | MemoryStore) -> str | MemoryStore:
    """Write a fresh import whose ADT measured only the cells of ``_ADT_MEMBERS``.

    The membership column is written as an import writes it.
    """
    adt = ((_COUNTS[:, :2] + 1) * _ADT_MEMBERS[:, None]).astype(np.uint16)
    _write_source(location, {"RNA": _COUNTS, "ADT": adt})
    write_membership_column(
        load_zarr(location, mode="r+")["cellData"], "ADT", _ADT_MEMBERS
    )
    return location


@pytest.mark.parametrize("workspace", [None, "ws"])
def test_cell_tables_reserve_the_membership_column_of_each_assay(
    tmp_path, workspace, merged_in_workspace
):
    if workspace is None:
        location = str(_partial_adt_source(str(tmp_path / "partial.zarr")))
    else:
        # The merge wrote the membership column of RNA into the workspace.
        location = _copy(merged_in_workspace, tmp_path)
    store = DataStore(
        location,
        workspace=workspace,
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    assay = "ADT" if workspace is None else "RNA"
    column = f"{assay}_I"
    cells = store.cells
    before = cells.fetch_all(column)
    restriction = np.zeros(cells.N, dtype=bool)
    for write in (
        lambda: cells.insert(column, by_cell_id(cells, ~before)),
        lambda: cells.insert(column, by_cell_id(cells, ~before), overwrite=True),
        lambda: cells.update_key(by_cell_id(cells, restriction), column),
        lambda: cells.reset_key(column),
    ):
        with pytest.raises(ValueError, match=_RESERVED.format(assay=assay)):
            write()
    with pytest.raises(ValueError, match=_UNDROPPABLE.format(assay=assay)):
        cells.drop(column)
    np.testing.assert_array_equal(cells.fetch_all(column), before)
    assert cells._get_array(column).attrs.asdict() == membership_attributes(assay)

    if workspace is None:
        # RNA measured every cell; its absent membership column is reserved too.
        for write in (
            lambda: cells.insert(
                "RNA_I", by_cell_id(cells, np.ones(cells.N, dtype=bool))
            ),
            lambda: cells.update_key(by_cell_id(cells, restriction), "RNA_I"),
            lambda: cells.reset_key("RNA_I"),
        ):
            with pytest.raises(ValueError, match=_RESERVED.format(assay="RNA")):
                write()
        assert "RNA_I" not in cells.columns
        # A plain column with this name is not a membership column and can be
        # dropped to repair the store.
        create_metadata_column(
            cells.locations["primary"], "RNA_I", data=np.ones(cells.N, dtype=bool)
        )
        cells.drop("RNA_I")
        assert "RNA_I" not in cells.columns
    # A name that is not the membership column of an assay stays ordinary.
    cells.insert("ATAC_I", by_cell_id(cells, ~before))
    cells.drop("ATAC_I")


def test_subsets_keep_only_the_membership_of_their_assays(tmp_path):
    store = _open(_partial_adt_source(MemoryStore()))
    # The IDs come in another order; subsets keep the cells in store order.
    ids = store.cells.fetch_all("ids")[[4, 0, 1]]
    rows = np.asarray([0, 1, 4])

    only_rna = str(tmp_path / "rna.zarr")
    SubsetZarr(only_rna, [store.RNA], cell_ids=ids, nthreads=1).dump()
    both = str(tmp_path / "both.zarr")
    SubsetZarr(both, [store.RNA, store.ADT], cell_ids=ids, nthreads=1).dump()

    # A subset without ADT holds no ADT membership.
    assert "ADT_I" not in zarr.open_group(only_rna, mode="r")["cellData"]
    # A copy of ADT keeps the membership of the cells it copies.
    column = zarr.open_group(both, mode="r")["cellData/ADT_I"]
    np.testing.assert_array_equal(column[:], _ADT_MEMBERS[rows])
    assert column.attrs.asdict() == membership_attributes("ADT")

    # A store without an assayTypes record still has ADT's columns left out.
    del store.zw.attrs["assayTypes"]
    untyped = DataStore(store.zw.store, zarr_mode="r", default_assay="RNA")
    rna_only = str(tmp_path / "untyped.zarr")
    SubsetZarr(rna_only, [untyped.RNA], cell_ids=ids, nthreads=1).dump()
    cells = zarr.open_group(rna_only, mode="r")["cellData"]
    assert [name for name in cells.array_keys() if name.startswith("ADT_")] == []


@pytest.mark.slow
def test_merge_drops_the_membership_of_assays_it_leaves_out(tmp_path):
    partial = _open(_partial_adt_source(MemoryStore()))
    full = _open(
        _write_source(MemoryStore(), {"RNA": _COUNTS + 1, "ADT": _COUNTS[:, :2] + 2})
    )
    rna_only = str(tmp_path / "rna_only.zarr")

    DataStoreMerge(
        [partial, full],
        rna_only,
        ["partial", "full"],
        assays=["RNA"],
        seed=0,
        nthreads=1,
    ).dump()

    # Before, the partial source's ADT_I was merged as the ordinary column
    # orig_ADT_I, or as ADT_I without a prefix.
    root = zarr.open_group(rna_only, mode="r")
    assert "ADT" not in root
    assert not [name for name in root["cellData"].array_keys() if "ADT_I" in name]


# Merge planning and writing


def test_module_merges_hold_the_rows_of_both_sources(merged, merged_in_workspace):
    # Later tests compare resumed and rebuilt merges with these two.
    expected = _source_rows({"left": _COUNTS, "right": _COUNTS + 1})
    assert _rows_by_cell(merged) == expected
    workspace = zarr.open_group(merged_in_workspace, mode="r")
    ids = np.asarray(workspace["ws/cellData/ids"][:]).astype(str)
    counts = np.asarray(workspace["matrices/RNA/counts"][:])
    assert dict(zip(ids, counts.tolist(), strict=True)) == expected


def test_merge_keeps_the_assay_class_of_each_source_assay(tmp_path):
    counts = {"RNA": _COUNTS, "ADT": _COUNTS[:, :2] * 3, "ATAC": _COUNTS % 2}
    # A merge only reads its sources, so one store serves as both.
    source = _open(_write_source(MemoryStore(), counts))
    sources = [source, source]
    destination = str(tmp_path / "merged.zarr")

    _merge(sources, destination).dump()
    root = zarr.open_group(destination, mode="r")
    assert root.attrs["assayTypes"] == {"RNA": "RNA", "ADT": "ADT", "ATAC": "ATAC"}
    for name, values in counts.items():
        assert _rows_by_cell(destination, f"{name}/counts") == _source_rows(
            {"left": values, "right": values}
        )
    assert {name for name in counts if "countsT" in root[name]} == {"RNA"}
    merged = _open(destination)
    # ADT keeps CLR and ATAC keeps TF-IDF normalization.
    assert type(merged.get_assay("RNA")) is RNAassay
    assert type(merged.get_assay("ADT")) is ADTassay
    assert type(merged.get_assay("ATAC")) is ATACassay


def test_merge_reads_sources_that_live_in_memory(tmp_path):
    sources = [
        _open(_write_source(MemoryStore(), {"RNA": _COUNTS + offset}))
        for offset in range(2)
    ]
    destination = str(tmp_path / "merged.zarr")
    merger = _merge(sources, destination)

    # An in-memory source has no filesystem path that could overlap.
    assert merger.plan().canDump is True
    merger.dump()
    assert _rows_by_cell(destination) == _source_rows(
        {"left": _COUNTS, "right": _COUNTS + 1}
    )


def test_merge_requires_at_least_one_assay(tmp_path, sources):
    with pytest.raises(ValueError, match="No assays available to merge"):
        _merge(sources, str(tmp_path / "merged.zarr"), assays=[])


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("mounted", "Destination is a mounted matrix-source store"),
        ("workspace_array", "Destination workspace 'ws' is not a Zarr group"),
        ("workspace_matrices", "already uses workspace matrix storage"),
        ("foreign_overwrite", "overwrite=True can replace only a merge-owned"),
        ("foreign", "does not contain a matching DataStoreMerge manifest"),
    ],
)
def test_merge_refuses_destinations_it_cannot_own(
    tmp_path, sources, merged_in_workspace, case, reason
):
    destination = str(tmp_path / "merged.zarr")
    options: dict[str, Any] = {}
    if case == "mounted":
        mount_datastore(
            sources[0].zarr_loc,
            at=destination,
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )
    elif case == "workspace_array":
        zarr.open_group(destination, mode="w").create_array(
            "ws", shape=(1,), dtype="uint8"
        )
        options["out_workspace"] = "ws"
    elif case == "workspace_matrices":
        destination = _copy(merged_in_workspace, tmp_path)
    else:
        zarr.open_group(destination, mode="w").create_group("sentinel")
        options["overwrite"] = case == "foreign_overwrite"
    merger = _merge(sources, destination, **options)

    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None and reason in plan.blockedReason
    with pytest.raises(ValueError, match=reason):
        merger.dump()


@pytest.mark.parametrize("workspace", [None, "ws"])
def test_merge_writes_into_an_empty_destination_shell(tmp_path, sources, workspace):
    destination = str(tmp_path / "shell.zarr")
    zarr.open_group(destination, mode="w")
    merger = _merge(sources, destination, out_workspace=workspace)

    plan = merger.plan()
    assert plan.canDump is True
    assert plan.cellDataAction == "write"
    result = merger.dump()
    assert {component.action for component in result.components} == {"write"}
    assert result.resumed is False


def test_merge_validates_the_preparation_of_a_prepared_destination(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    _open(destination)

    plan = _merge(sources, destination).plan()
    assert plan.canDump is True
    assert {plan.cellDataAction} | {
        action
        for assay in plan.assays
        for action in (assay.countsAction, assay.countsTAction)
    } == {"skip"}

    zarr.open_group(destination, mode="r+")["RNA"].attrs["counts_fingerprint"] = (
        "0" * 64
    )
    plan = _merge(sources, destination).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "inconsistent dataset identity" in plan.blockedReason


def test_merge_plan_refuses_to_resume_the_counts_of_a_prepared_assay(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    _open(destination)
    root = zarr.open_group(destination, mode="r+")
    root["RNA"].attrs["complete"] = False
    root.attrs.update({"cytearc:import_complete": False, "complete": False})
    merger = _merge(sources, destination)

    # The preview agrees with dump, which must not delete the prepared assay.
    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "damaged prepared assay requires a fresh destination" in plan.blockedReason
    with pytest.raises(ValueError, match="damaged prepared assay"):
        merger.dump()
    assert zarr.open_group(destination, mode="r")["RNA"].attrs["prepared"] is True


@pytest.mark.parametrize("overwrite", [False, True])
def test_merge_interrupted_before_cell_metadata_is_completed(
    tmp_path, monkeypatch, sources, merged, overwrite
):
    destination = str(tmp_path / "merged.zarr")

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(merge_datasets, "write_cell_metadata", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            _merge(sources, destination).dump()
    interrupted = zarr.open_group(destination, mode="r")
    assert "cellData" not in interrupted
    assert "assayTypes" not in interrupted.attrs

    merger = _merge(sources, destination, overwrite=overwrite)
    assert merger.plan().cellDataAction == "write"
    result = merger.dump()
    actions = {component.name: component.action for component in result.components}
    assert actions["cellData"] == "write"
    assert actions["counts:RNA"] == ("write" if overwrite else "resume")
    np.testing.assert_array_equal(_merged_counts(destination), _merged_counts(merged))


@pytest.mark.parametrize("manifest", ["missing", "assays_not_a_list"])
def test_merge_overwrite_restarts_without_a_usable_stored_manifest(
    tmp_path, sources, merged, manifest
):
    destination = _copy(merged, tmp_path)
    root = zarr.open_group(destination, mode="r+")
    if manifest == "missing":
        del root.attrs["cytearc:merge_manifest"]
    else:
        root.attrs["cytearc:merge_manifest"] = {
            **root.attrs["cytearc:merge_manifest"],
            "assays": "RNA",
        }

    result = _merge(sources, destination, overwrite=True).dump()
    assert {component.action for component in result.components} == {"write"}
    np.testing.assert_array_equal(_merged_counts(destination), _merged_counts(merged))


def test_merge_overwrite_removes_a_pipeline_group_it_leaves_empty(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    zarr.open_group(destination, mode="r+").create_group(
        f"pipeline/runs/{'a' * 64}/stages"
    )

    _merge(sources, destination, overwrite=True).dump()
    assert "pipeline" not in zarr.open_group(destination, mode="r")


def test_merge_refuses_a_destination_mounted_after_planning(
    tmp_path, monkeypatch, sources
):
    _before_opening(monkeypatch, lambda root: root.attrs.update(_MOUNTED))

    with pytest.raises(ValueError, match="mounted matrix-source store"):
        _merge(sources, str(tmp_path / "merged.zarr")).dump()


def test_merge_overwrite_refuses_a_destination_disowned_after_planning(
    tmp_path, monkeypatch, sources, merged
):
    destination = _copy(merged, tmp_path)

    def disown(root: zarr.Group) -> None:
        del root.attrs["cytearc:import_source"]

    _before_opening(monkeypatch, disown)
    with pytest.raises(ValueError, match="changed after planning"):
        _merge(sources, destination, overwrite=True).dump()
    assert "RNA" in zarr.open_group(destination, mode="r")


@pytest.mark.parametrize("workspace", [None, "ws"])
def test_merge_overwrite_rebuilds_a_destination_emptied_after_planning(
    tmp_path, monkeypatch, sources, merged, merged_in_workspace, workspace
):
    template = merged if workspace is None else merged_in_workspace
    destination = _copy(template, tmp_path)
    path = "RNA/counts" if workspace is None else "matrices/RNA/counts"

    _before_opening(monkeypatch, _empty)
    _merge(sources, destination, out_workspace=workspace, overwrite=True).dump()
    np.testing.assert_array_equal(
        _merged_counts(destination, path), _merged_counts(template, path)
    )
    root = zarr.open_group(destination, mode="r")
    attributes = root.attrs if workspace is None else root[workspace].attrs
    assert attributes["complete"] is True


def test_merge_overwrite_rechecks_containment_after_clearing(
    tmp_path, monkeypatch, sources, merged
):
    destination = _copy(merged, tmp_path)
    original = DataStoreMerge._clear_merge_components

    def clear(self, root, stored_manifest):
        original(self, root, stored_manifest)
        root.attrs.update(_MOUNTED)

    monkeypatch.setattr(DataStoreMerge, "_clear_merge_components", clear)
    with pytest.raises(ValueError, match="mounted matrix-source store"):
        _merge(sources, destination, overwrite=True).dump()


def test_merge_finalization_rechecks_containment(
    tmp_path, monkeypatch, sources, merged
):
    destination = _copy(merged, tmp_path)
    # Every component is complete, but the final markers were not written.
    zarr.open_group(destination, mode="r+").attrs.update(
        {"cytearc:import_complete": False, "complete": False}
    )
    original = merge_datasets.load_zarr

    def load(zarr_loc, mode, storage_options=None):
        if mode == "r+":
            zarr.open_group(zarr_loc, mode="r+").attrs.update(_MOUNTED)
        return original(zarr_loc, mode=mode, storage_options=storage_options)

    monkeypatch.setattr(merge_datasets, "load_zarr", load)
    with pytest.raises(ValueError, match="mounted matrix-source store"):
        _merge(sources, destination).dump()
    assert zarr.open_group(destination, mode="r").attrs["complete"] is False


def test_merge_resume_refuses_counts_of_another_feature_union(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    # The right counts under other feature IDs keep the source fingerprints.
    renamed = _open(
        _write_source(MemoryStore(), {"RNA": _COUNTS + 1}, ids=["RNA0", "RNA1", "x"])
    )

    plan = _merge([sources[0], renamed], destination).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "counts shape for 'RNA' is (12, 3), expected (12, 4)" in plan.blockedReason


def test_merge_resume_refuses_counts_narrower_than_the_names_now_summed(
    tmp_path, sources
):
    destination = str(tmp_path / "merged.zarr")
    _merge(sources, destination, feature_key="names").dump()
    # The right counts, now with two features of one name to sum.
    renamed = _open(
        _write_source(
            MemoryStore(), {"RNA": _COUNTS + 1}, names=["RNA0", "RNA0", "RNA1"]
        )
    )

    plan = _merge([sources[0], renamed], destination, feature_key="names").plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "counts dtype for 'RNA' is uint16, expected uint32" in plan.blockedReason


def test_merge_resume_refuses_counts_rewritten_in_another_layout(tmp_path, sources):
    destination = str(tmp_path / "merged.zarr")
    policy = CountMatrixPolicy(unitBytes=64, chunkBytes=16)
    _merge(sources, destination, policy=policy).dump()
    # Hand-written counts in another layout, finalized with a matching record.
    matrix = zarr.open_group(destination, mode="r+")["RNA"]
    values = np.asarray(matrix["counts"][:])
    del matrix["counts"]
    counts = create_product_counts_array(
        matrix,
        *values.shape,
        values.dtype,
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=128, chunkBytes=16),
    )
    counts[:] = values
    finalize_test_counts(counts)

    plan = _merge(sources, destination, policy=policy).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "counts chunks for 'RNA' are (12, 1), expected (10, 1)" in plan.blockedReason


def test_merge_resume_refuses_feature_metadata_without_its_selection(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    del zarr.open_group(destination, mode="r+")["RNA/featureData/I"]

    plan = _merge(sources, destination).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "featureData/I is missing for 'RNA'" in plan.blockedReason


def _counting_refusal(needed: int, budget: int, item: str) -> str:
    """Return how a merge refuses to count the nonzero values of each merged RNA cell."""
    return (
        "Counting the nonzero values of each merged cell of assay 'RNA' holds about "
        f"{needed} bytes for each merged cell's count and {item}, more than the "
        f"memory budget of {budget} bytes. Raise mem_budget to at least {needed} bytes."
    )


def test_merge_refuses_a_feature_profile_beyond_its_budget(tmp_path):
    from cytearc.merge.row_plan import build_row_plan

    counts = np.zeros((4, 25_000), dtype=np.uint16)
    counts[:, ::997] = 1
    sources = [_open(_write_source(MemoryStore(), {"RNA": counts})) for _ in _NAMES]
    destination = tmp_path / "merged.zarr"
    assays = [source.get_assay("RNA") for source in sources]
    alignment = align_features(assays, _NAMES)
    row_plan = build_row_plan(
        [4, 4], [assay.rawData.chunksize[0] for assay in assays], _NAMES, seed=0
    )
    # Every phase holds the row plan, the feature alignment and the
    # destination's count summary, 40 bytes for each of the 8 merged cells
    # and 16 for each of the 25,000 features. Counting each merged cell's
    # values adds 8 bytes for each cell and one more, and a whole read of a
    # source's summary of its 4 cells in one chunk: the counts, a decoded copy
    # and an encoded one at its bound, 1 KiB more, and the 52 KiB of objects
    # of Zarr's read of one chunk.
    needed = (
        row_plan.resident_bytes()
        + alignment.resident_bytes()
        + 40 * 8
        + 16 * 25_000
        + 8 * 9
        + 3 * 8 * 4
        + 1_024
        + 32_768
        + 4_096
        + 16_384
    )

    with pytest.raises(MemoryError) as refused:
        _merge(sources, str(destination), mem_budget=needed - 1).plan()
    assert str(refused.value) == _counting_refusal(
        needed, needed - 1, "the count summaries of source 'left'"
    )
    assert not destination.exists()


# Each pair of features shares a name, so a merge by name sums the pair and
# writes through sparse batches, whose memory grows with their values.
_PAIRED_NAMES = [f"G{index // 2}" for index in range(24)]


def _counts_with_negative_values(
    location: Path, dtype: str, n_cells: int = 400
) -> tuple[str, np.ndarray]:
    """Write a store whose counts are mostly nonzero, nearly half of them negative.

    Float counts hold quarters, which float32 stores exactly, and signed
    counts hold them rounded. Unsigned counts hold their magnitudes, rounded
    up, so they hold no negative value and few zeros.
    """
    values = np.random.default_rng(29).integers(-16, 17, size=(n_cells, 24)) / 4
    if dtype == "uint16":
        values = np.ceil(np.abs(values))
    elif dtype != "float32":
        values = np.round(values)
    path = str(location)
    write_count_store(
        path, {"RNA": values}, dtype, feature_names={"RNA": _PAIRED_NAMES}
    )
    return path, values


def _sparse_unsigned_counts(location: Path) -> tuple[str, np.ndarray]:
    """Write a store of 400 cells whose uint16 counts are about a quarter nonzero."""
    rng = np.random.default_rng(31)
    values = rng.integers(1, 9, size=(400, 24)) * (rng.random((400, 24)) < 0.25)
    path = str(location)
    write_count_store(
        path, {"RNA": values}, "uint16", feature_names={"RNA": _PAIRED_NAMES}
    )
    return path, values


def _most_values_in_any_rows(row_values: np.ndarray, rows: int) -> int:
    """Return the most values that any ``rows`` consecutive merged rows hold."""
    width = min(int(rows), row_values.size)
    if width < 1:
        return 0
    windows = np.lib.stride_tricks.sliding_window_view(row_values, width)
    return int(windows.sum(axis=1).max())


@pytest.mark.parametrize("dtype", ["int16", "float32", "uint16"])
def test_merge_plans_its_sparse_batches_from_every_value_of_each_cell(
    tmp_path, monkeypatch, dtype
):
    import cytearc.merge.writer as merge_writer

    left_path, left = _counts_with_negative_values(tmp_path / "left.zarr", dtype)
    right_path, right = _sparse_unsigned_counts(tmp_path / "right.zarr")
    values = {"left": left, "right": right}
    if dtype != "uint16":
        # Signed and float counts hold negative values, which a count of
        # positive values leaves out.
        assert np.count_nonzero(left < 0) > 0.4 * left.size
    plans = []
    resolve = merge_writer.resolve_sparse_import_batch

    def recorded(destinations, **options):
        plan = resolve(destinations, **options)
        plans.append((destinations, options, plan))
        return plan

    monkeypatch.setattr(merge_writer, "resolve_sparse_import_batch", recorded)
    destination = str(tmp_path / "merged.zarr")
    # At this budget the dense left cells decide how many rows a batch holds.
    _merge(
        [_open(left_path), _open(right_path)],
        destination,
        feature_key="names",
        mem_budget=1_000_000,
    ).dump()

    root = zarr.open_group(destination, mode="r")
    sources = [cell.split("__") for cell in np.asarray(root["cellData/ids"][:])]
    merged_names = np.asarray(root["RNA/featureData/ids"][:]).astype(str)
    stored = np.asarray(root["RNA/counts"][:])
    for (source, cell), merged_row in zip(sources, stored, strict=True):
        row = values[source][int(cell.removeprefix("cell"))]
        assert merged_row.tolist() == [
            row[np.asarray(_PAIRED_NAMES) == name].sum() for name in merged_names
        ]
    # Each merged row holds the nonzero values of its source row, counted
    # here with NumPy, until the batch sums each pair.
    row_values = np.asarray(
        [
            np.count_nonzero(values[source][int(cell.removeprefix("cell"))])
            for source, cell in sources
        ]
    )
    destinations, options, plan = plans[0]
    assert plan.batchRows < len(row_values)
    for rows in (1, 2, 7, plan.batchRows, plan.batchRows + 13, len(row_values)):
        assert options["maxWindowNnz"](rows) == _most_values_in_any_rows(
            row_values, rows
        )
    # The batches that the write planned fit the budget with the values that
    # their rows hold.
    resolve(
        destinations,
        **options
        | {
            "maxWindowNnz": lambda rows: _most_values_in_any_rows(row_values, rows),
            "batchRows": plan.batchRows,
        },
    )


@pytest.mark.parametrize("summaries", ["current", "older"])
@pytest.mark.parametrize("dtype", ["int16", "float32"])
def test_merge_reads_only_the_counts_that_their_summaries_cannot_count(
    tmp_path, monkeypatch, dtype, summaries
):
    from collections import Counter

    from zarr.storage import LocalStore

    left_path, _ = _counts_with_negative_values(tmp_path / "left.zarr", dtype)
    right_path, _ = _sparse_unsigned_counts(tmp_path / "right.zarr")
    if summaries == "older":
        drop_row_nonzero(left_path)
        drop_row_nonzero(right_path)
    sources = [_open(left_path), _open(right_path)]
    fetched: Counter = Counter()
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        if key.startswith("RNA/counts/c/"):
            fetched[store.root.name, key] += 1
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    _merge(sources, str(tmp_path / "merged.zarr"), feature_key="names").plan()

    # Count summaries count each cell's nonzero values, the negative ones
    # among them, so the plan reads no count. Summaries written before they
    # did count only the values of the uint16 counts, so the signed or float
    # counts are read once to count theirs.
    expected = {} if summaries == "current" else {("left.zarr", "RNA/counts/c/0/0"): 1}
    assert fetched == expected


# 24 cells of 600 features, 25 times more features than cells, stored in two
# bands of 12 cells whose chunks hold 100 features each, so that a read can
# take a band one chunk column at a time.
_WIDE_LAYOUT = CountMatrixPolicy(unitBytes=14_400, chunkBytes=2_880)


def _wide_signed_counts(
    location: Path, seed: int, layout: CountMatrixPolicy = _WIDE_LAYOUT
) -> tuple[str, np.ndarray]:
    """Write a store of 24 cells and 600 int16 features, about 43% of them negative.

    Its count summaries lack ``rowNonzero``, as those of stores written
    before it was stored do, so a merge counts its values in a read.
    """
    values = np.random.default_rng(seed).integers(-3, 4, size=(24, 600))
    path = str(location)
    _write_source(path, {"RNA": values}, dtype="int16", policy=layout)
    drop_row_nonzero(path)
    # The first opening prepares the store; the tests open it again.
    _open(path)
    return path, values


def _opened_with_budget(path: str, budget: int) -> DataStore:
    return DataStore(
        path,
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        mem_budget=budget,
    )


@pytest.fixture
def recorded_blocks(monkeypatch) -> list[tuple[int, int, int, int]]:
    """Record the rows and columns of each block that a matrix stream reads."""
    from cytearc.matrix import ChunkedArray

    blocks: list[tuple[int, int, int, int]] = []
    materialize = ChunkedArray._materialize_range

    def recorded(self, start, end):
        columns = np.arange(self.shape[1]) if self._cols is None else self._cols
        blocks.append((start, end, int(columns[0]), int(columns[-1]) + 1))
        return materialize(self, start, end)

    monkeypatch.setattr(ChunkedArray, "_materialize_range", recorded)
    return blocks


def _merged_order_values(
    row_plan: Any, stores: Sequence[tuple[str, np.ndarray]]
) -> np.ndarray:
    """Count each merged cell's nonzero values with NumPy, in the merged order."""
    return np.concatenate(
        [
            np.count_nonzero(stores[segment.sourceIdx][1][segment.localRows], axis=1)
            for segment in iter_row_plan_segments(row_plan)
        ]
    )


def test_merge_counts_signed_values_within_its_budget_or_names_the_budget_it_needs(
    tmp_path, recorded_blocks
):
    import cytearc.merge.writer as merge_writer
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.budget import ResourceBudget
    from cytearc.utils.memory_budget import MemoryBudgetError

    stores = [
        _wide_signed_counts(tmp_path / f"{name}.zarr", seed)
        for seed, name in enumerate(_NAMES)
    ]
    # The sources' own budgets are far below the merge's, which plans the read.
    assays = [_opened_with_budget(path, 1_000).get_assay("RNA") for path, _ in stores]
    assert assays[0].rawData.chunksize == (12, 600)
    assert assays[0].rawData.chunk_columns() == 100
    row_plan = build_row_plan([24, 24], [12, 12], _NAMES, seed=0)
    expected = _merged_order_values(row_plan, stores)
    # Each band, one chunk column at a time, as the smallest read takes them.
    one_chunk_column = sorted(
        [
            (rows, rows + 12, start, start + 100)
            for rows in (0, 12)
            for start in range(0, 600, 100)
        ]
        * 2
    )

    def profile(budget: int, held: int = 1_000) -> np.ndarray:
        recorded_blocks.clear()
        return merge_writer._cumulative_row_nnz(
            assays, row_plan, ResourceBudget(budget, 1), resident_bytes=held
        )

    np.testing.assert_array_equal(np.diff(profile(10**9)), expected)
    # Each source is read in blocks of a stored band and every column.
    assert sorted(recorded_blocks) == sorted([(0, 12, 0, 600), (12, 24, 0, 600)] * 2)
    # Before any read, the merge names the budget of its smallest read.
    with pytest.raises(MemoryBudgetError) as refused:
        profile(2_000)
    needed = refused.value.needed_bytes
    assert str(refused.value) == _counting_refusal(
        needed, 2_000, "a read of the counts of source 'left'"
    )
    # At that budget the read takes each band one chunk column at a time, so
    # it decodes each chunk once, and counts every value.
    np.testing.assert_array_equal(np.diff(profile(needed)), expected)
    assert sorted(recorded_blocks) == one_chunk_column
    with pytest.raises(MemoryBudgetError) as short:
        profile(needed - 1)
    assert short.value.needed_bytes == needed
    # Every larger budget counts every value too, in wider runs of chunk
    # columns up to every column, each of which fits beside what is held and
    # the buffer of the counts' shard file; steps shorter than the 1,584 bytes
    # held reach every run that fits only when what is held is left out.
    from tests.storage_helpers import local_read_buffer

    buffer = local_read_buffer(Path(stores[0][0]) / "zarr.json", sharded=True)
    for budget in range(needed, 136_000 + buffer, 1_499):
        np.testing.assert_array_equal(np.diff(profile(budget)), expected)
    assert sorted(recorded_blocks) == sorted([(0, 12, 0, 600), (12, 24, 0, 600)] * 2)
    # What the merge holds beside the read adds to what the read needs, and
    # leaves the read the same runs.
    with pytest.raises(MemoryBudgetError) as holding_more:
        profile(needed + 99_999, held=101_000)
    assert holding_more.value.needed_bytes == needed + 100_000
    np.testing.assert_array_equal(
        np.diff(profile(needed + 100_000, held=101_000)), expected
    )
    assert sorted(recorded_blocks) == one_chunk_column
    # The read alone, beside the profile of 48 cells (49 int64 values), needs
    # exactly the budget that the merge named before it.
    for budget in (needed, needed - 1):
        resources = ResourceBudget(budget, 1)
        count = partial(
            merge_writer._counted_row_values,
            assays[0],
            "left",
            assays[0].rawData._with_resources(resources),
            resources,
            held_bytes=1_000 + 49 * 8,
        )
        if budget == needed:
            np.testing.assert_array_equal(
                count(), np.count_nonzero(stores[0][1], axis=1)
            )
            continue
        with pytest.raises(MemoryBudgetError) as alone:
            count()
        assert alone.value.needed_bytes == needed


def test_merge_refuses_before_it_reads_any_source_when_one_cannot_be_read(
    tmp_path, recorded_blocks
):
    import cytearc.merge.writer as merge_writer
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.budget import ResourceBudget
    from cytearc.utils.memory_budget import MemoryBudgetError

    # The right source's chunks hold 300 features, so its smallest read is
    # larger than the left source's, which the merge reads first.
    stores = [
        _wide_signed_counts(tmp_path / "left.zarr", 0),
        _wide_signed_counts(
            tmp_path / "right.zarr",
            1,
            CountMatrixPolicy(unitBytes=14_400, chunkBytes=7_200),
        ),
    ]
    assays = [_open(path).get_assay("RNA") for path, _ in stores]
    assert [assay.rawData.chunk_columns() for assay in assays] == [100, 300]
    row_plan = build_row_plan([24, 24], [12, 12], _NAMES, seed=0)

    def profile(budget: int) -> np.ndarray:
        recorded_blocks.clear()
        return merge_writer._cumulative_row_nnz(
            assays, row_plan, ResourceBudget(budget, 1), resident_bytes=1_000
        )

    with pytest.raises(MemoryBudgetError) as refused:
        profile(2_000)
    needed = refused.value.needed_bytes
    assert str(refused.value) == _counting_refusal(
        needed, 2_000, "a read of the counts of source 'right'"
    )
    # One byte short, the left source would fit, and none is read.
    with pytest.raises(MemoryBudgetError) as short:
        profile(needed - 1)
    assert short.value.needed_bytes == needed
    assert recorded_blocks == []
    np.testing.assert_array_equal(
        np.diff(profile(needed)), _merged_order_values(row_plan, stores)
    )


def test_merge_names_the_count_summaries_that_its_budget_cannot_load(
    tmp_path, recorded_blocks
):
    import cytearc.merge.writer as merge_writer
    from cytearc.merge.row_plan import build_row_plan
    from cytearc.storage.budget import ResourceBudget
    from cytearc.utils.memory_budget import MemoryBudgetError

    # Float counts, nearly half of them negative, whose summaries count each
    # cell's nonzero values: 400 cells on the left, 200 on the right.
    stores = [
        _counts_with_negative_values(tmp_path / "left.zarr", "float32"),
        _counts_with_negative_values(tmp_path / "right.zarr", "float32", n_cells=200),
    ]
    assays = [_open(path).get_assay("RNA") for path, _ in stores]
    row_plan = build_row_plan(
        [400, 200], [assay.rawData.chunksize[0] for assay in assays], _NAMES, seed=0
    )

    def profile(budget: int) -> np.ndarray:
        recorded_blocks.clear()
        return merge_writer._cumulative_row_nnz(
            assays, row_plan, ResourceBudget(budget, 1), resident_bytes=1_000
        )

    # Beside the 1,000 bytes held, a count of each of the 600 merged cells
    # and one more, 8 bytes each, and a whole read of the left source's
    # summary of its 400 cells in one chunk: the counts, a decoded copy and
    # an encoded one at its bound, 1/128 more and 1 KiB, the buffer of its
    # file where Python reads it through one, and the 52 KiB of objects of
    # Zarr's read of one chunk.
    from tests.storage_helpers import local_read_buffer

    buffer = local_read_buffer(Path(stores[0][0]) / "zarr.json", sharded=False)
    summary = 3 * 8 * 400 + 8 * 400 // 128 + 1_024 + buffer + 32_768 + 4_096 + 16_384
    needed = 1_000 + 8 * 601 + summary
    with pytest.raises(MemoryBudgetError) as refused:
        profile(needed - 1)
    assert str(refused.value) == _counting_refusal(
        needed, needed - 1, "the count summaries of source 'left'"
    )
    # At that budget each merged cell takes its source cell's nonzero values,
    # counted here with NumPy, and no count is read.
    np.testing.assert_array_equal(
        np.diff(profile(needed)), _merged_order_values(row_plan, stores)
    )
    assert recorded_blocks == []


def test_merge_reads_signed_counts_within_its_own_budget_whatever_the_sources_budgets(
    tmp_path,
):
    stores = [
        _wide_signed_counts(tmp_path / f"{name}.zarr", seed)
        for seed, name in enumerate(_NAMES)
    ]
    sources = [_opened_with_budget(path, 1_000) for path, _ in stores]
    destination = str(tmp_path / "merged.zarr")

    _merge(sources, destination, mem_budget=10**9).dump()
    assert _rows_by_cell(destination) == _source_rows(
        {name: values for name, (_, values) in zip(_NAMES, stores, strict=True)}
    )


_GENES = ["G0", "G1"]


def _zeros_read_as_counts(first: str = "left", *, adt: bool = False) -> str:
    """Return how a merge's warning about unmeasured features ends.

    ``first`` is the first source that holds the assay, and ``adt`` whether
    the assay is of type ADT, which the ADT recipe reads.
    """
    if adt:
        return (
            " The merge writes zero counts for the features a source lacks; "
            f"columns such as 'measured_in_{first}' record what each source "
            "measured, so the ADT recipe leaves those antibodies out of its panel "
            "and markers, while other analyses read the zeros as counts."
        )
    return (
        " The merge writes zero counts for the features a source lacks, which "
        f"analyses read as counts; columns such as 'measured_in_{first}' record "
        "what each source measured."
    )


def test_merge_gives_one_warning_naming_each_assay_whose_sources_share_few_features(
    tmp_path,
):
    # The sources share 1 of 21 genes and 1 of 19 antibodies.
    left = _open(
        _write_source(
            MemoryStore(),
            {"RNA": np.ones((3, 11), np.uint16), "ADT": np.ones((3, 10), np.uint16)},
            ids={
                "RNA": [f"x{index}" for index in range(11)],
                "ADT": [f"a{index}" for index in range(10)],
            },
        )
    )
    right = _open(
        _write_source(
            MemoryStore(),
            {"RNA": np.ones((3, 11), np.uint16), "ADT": np.ones((3, 10), np.uint16)},
            ids={
                "RNA": ["x0", *(f"y{index}" for index in range(1, 11))],
                "ADT": ["a0", *(f"b{index}" for index in range(1, 10))],
            },
        )
    )

    with _captured_warnings() as messages:
        plan = _merge([left, right], str(tmp_path / "merged.zarr")).plan()
    fractions = {item.assayName: item.featureOverlapFraction for item in plan.assays}
    assert fractions == pytest.approx({"RNA": 1 / 21, "ADT": 1 / 19})
    # The merge takes the assays in the order that the sources list them.
    assert messages == [
        "Assay 'ADT': 18 of the 19 merged features were not measured by every "
        "source that holds the assay. Source 'left' lacks 9: 'b1', 'b2', 'b3', "
        "'b4', 'b5' and 4 more. Source 'right' lacks 9: 'a1', 'a2', 'a3', 'a4', "
        "'a5' and 4 more." + _zeros_read_as_counts(adt=True),
        "Assay 'RNA': 20 of the 21 merged features were not measured by every "
        "source that holds the assay. Source 'left' lacks 10: 'y1', 'y2', 'y3', "
        "'y4', 'y5' and 5 more. Source 'right' lacks 10: 'x1', 'x2', 'x3', 'x4', "
        "'x5' and 5 more." + _zeros_read_as_counts(),
    ]


def _panel_source(
    panels: Mapping[str, Sequence[str]], names: Sequence[str] | None = None
) -> DataStore:
    """Open a source of three cells whose assays measure the features named.

    Each assay's counts are 1, 2, 3, ... across the first cell's features,
    then the next cell's.
    """
    counts = {
        assay: np.arange(1, 3 * len(ids) + 1, dtype=np.uint16).reshape(3, len(ids))
        for assay, ids in panels.items()
    }
    return _open(_write_source(MemoryStore(), counts, ids=panels, names=names))


def test_merge_warns_of_the_antibodies_that_each_panel_lacks(tmp_path):
    left = _panel_source(
        {"RNA": _GENES, "ADT": ["CD3", "CD4", "CD8", "CD45RA", "CD27"]}
    )
    right = _panel_source(
        {"RNA": _GENES, "ADT": ["CD3", "CD4", "CD8", "CD14", "CD16", "CD19"]}
    )
    destination = str(tmp_path / "merged.zarr")

    with _captured_warnings() as messages:
        merger = _merge([left, right], destination)
        plan = merger.plan()
        merger.dump()
    # Three of the eight antibodies are shared, and the panel warning is given
    # once, by the plan.
    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert adt.featureOverlapFraction == pytest.approx(3 / 8)
    assert messages == [
        "Assay 'ADT': 5 of the 8 merged features were not measured by every "
        "source that holds the assay. Source 'left' lacks 3: 'CD14', 'CD16', "
        "'CD19'. Source 'right' lacks 2: 'CD45RA', 'CD27'."
        + _zeros_read_as_counts(adt=True)
    ]
    # The merged counts hold the zeros that the warning describes.
    root = zarr.open_group(destination, mode="r")
    assert root["ADT/featureData/ids"][:].tolist() == [
        "CD3",
        "CD4",
        "CD8",
        "CD45RA",
        "CD27",
        "CD14",
        "CD16",
        "CD19",
    ]
    rows = _rows_by_cell(destination, "ADT/counts")
    assert rows["left__c0"] == [1, 2, 3, 4, 5, 0, 0, 0]
    assert rows["right__c0"] == [1, 2, 3, 0, 0, 4, 5, 6]


def test_merge_names_at_most_five_of_the_genes_that_each_source_lacks(tmp_path):
    long_name = "h0" + "x" * 300
    left = _panel_source({"RNA": [f"g{index}" for index in range(1200)]})
    right = _panel_source(
        {
            "RNA": [f"g{index}" for index in range(1000)]
            + [long_name]
            + [f"h{index}" for index in range(1, 3000)]
        }
    )

    with _captured_warnings() as messages:
        plan = _merge([left, right], str(tmp_path / "merged.zarr")).plan()
    assert plan.assays[0].featureOverlapFraction == pytest.approx(1000 / 4200)
    (message,) = messages
    # A name longer than a message quotes keeps only its start and its end.
    head, tail = message.split(" Source 'left' lacks 3,000: 'h0xxx")
    assert head == (
        "Assay 'RNA': 3,200 of the 4,200 merged features were not measured by "
        "every source that holds the assay."
    )
    assert tail.endswith(
        "xxx', 'h1', 'h2', 'h3', 'h4' and 2,995 more. Source 'right' lacks 200: "
        "'g1000', 'g1001', 'g1002', 'g1003', 'g1004' and 195 more."
        + _zeros_read_as_counts()
    )
    assert "x" * 100 not in message
    assert len(message) < 500


@pytest.mark.parametrize(
    ("third", "fraction", "warning"),
    [
        pytest.param(
            ["CD3", "CD4"],
            3 / 4,
            "1 of the 4 merged features was not measured by every source that "
            "holds the assay. Source 'c' lacks 1: 'CD8'.",
            id="the third panel lacks one antibody",
        ),
        pytest.param(
            ["CD4"],
            2 / 4,
            "2 of the 4 merged features were not measured by every source that "
            "holds the assay. Source 'c' lacks 2: 'CD3', 'CD8'.",
            id="the third panel lacks two antibodies",
        ),
    ],
)
def test_merge_plan_counts_the_antibodies_that_every_one_of_three_panels_measured(
    tmp_path, third, fraction, warning
):
    panels = (
        ["CD3", "CD4", "CD8", "CD14"],
        ["CD14", "CD8", "CD3", "CD4"],
        [*third, "CD14"],
    )
    sources = [_panel_source({"RNA": _GENES, "ADT": panel}) for panel in panels]

    with _captured_warnings() as messages:
        plan = DataStoreMerge(
            sources, str(tmp_path / "merged.zarr"), ["a", "b", "c"], nthreads=1
        ).plan()
    # Every antibody is in at least two panels, yet the cells of source c read
    # zero counts of each antibody that its panel lacks.
    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert adt.nFeatures == 4
    assert adt.featureOverlapFraction == fraction
    assert messages == [
        "Assay 'ADT': " + warning + _zeros_read_as_counts("a", adt=True)
    ]


def test_three_panels_of_which_two_overlap_merge_with_no_antibody_in_all_three(
    tmp_path,
):
    panels = (["CD3", "CD4"], ["CD4", "CD3"], ["CD8", "CD14"])
    sources = [_panel_source({"RNA": _GENES, "ADT": panel}) for panel in panels]
    destination = str(tmp_path / "merged.zarr")

    with _captured_warnings() as messages:
        merger = DataStoreMerge(sources, destination, ["a", "b", "c"], nthreads=1)
        plan = merger.plan()
        merger.dump()
    # Panels a and b share their antibodies, so the merge goes ahead, but no
    # antibody is in all three panels.
    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert adt.featureOverlapFraction == 0.0
    assert messages == [
        "Assay 'ADT': 4 of the 4 merged features were not measured by every "
        "source that holds the assay. Source 'a' lacks 2: 'CD8', 'CD14'. Source "
        "'b' lacks 2: 'CD8', 'CD14'. Source 'c' lacks 2: 'CD3', 'CD4'."
        + _zeros_read_as_counts("a", adt=True)
    ]
    rows = _rows_by_cell(destination, "ADT/counts")
    assert [rows["a__c0"], rows["b__c0"], rows["c__c0"]] == [
        [1, 2, 0, 0],
        [2, 1, 0, 0],
        [0, 0, 1, 2],
    ]


@pytest.mark.parametrize("feature_key", ["ids", "names"])
def test_merge_refuses_three_panels_that_share_no_antibody(tmp_path, feature_key):
    panels = (["CD3", "CD4"], ["CD8"], ["CD14", "CD19"])
    sources = [_panel_source({"RNA": _GENES, "ADT": panel}) for panel in panels]
    destination = tmp_path / "merged.zarr"

    with pytest.raises(ValueError, match="No overlapping features found"):
        DataStoreMerge(
            sources,
            str(destination),
            ["a", "b", "c"],
            nthreads=1,
            feature_key=feature_key,
        ).plan()
    assert not destination.exists()


_MISSING_ADT = (
    "Source {!r} is missing assay 'ADT'; writing zeros and marking assay "
    "membership false"
)


@pytest.mark.parametrize(
    ("panels", "options", "expected"),
    [
        pytest.param(
            [
                {"RNA": _GENES, "ADT": ["CD3", "CD4", "CD8"]},
                {"RNA": _GENES, "ADT": ["CD8", "CD3", "CD4"]},
            ],
            {},
            [],
            id="one panel in another order",
        ),
        pytest.param(
            [
                {"RNA": _GENES, "ADT": ["CD3", "CD4"]},
                {"RNA": _GENES},
                {"RNA": _GENES, "ADT": ["CD4", "CD3"]},
            ],
            {},
            [_MISSING_ADT.format("right")],
            id="a source without the assay",
        ),
        pytest.param(
            [{"RNA": _GENES, "ADT": ["CD3", "CD4"]}, {"RNA": _GENES}],
            {},
            [_MISSING_ADT.format("right")],
            id="one source with the assay",
        ),
        pytest.param(
            [{"RNA": ["ENSG01", "ENSG02"]}, {"RNA": ["CD3E", "CD4"]}],
            {"feature_key": "names"},
            [],
            id="one panel by name under other IDs",
        ),
    ],
)
def test_merge_of_sources_that_measured_one_panel_names_no_feature(
    tmp_path, panels, options, expected
):
    names = ["left", "right", "third"][: len(panels)]
    sources = [
        _panel_source(panel, names=["CD3E", "CD4"] if options else None)
        for panel in panels
    ]

    with _captured_warnings() as messages:
        plan = DataStoreMerge(
            sources, str(tmp_path / "merged.zarr"), names, nthreads=1, **options
        ).plan()
    assert messages == expected
    # Every source that holds an assay measured each of its features; a source
    # without the assay is left out of the count.
    assert {item.featureOverlapFraction for item in plan.assays} == {1.0}


_NO_ANTIBODIES = (
    "Assay 'ADT': 2 of the 2 merged features were not measured by every source "
    "that holds the assay. Source 'c' holds no features of the assay, so it "
    "lacks every merged feature." + _zeros_read_as_counts("a", adt=True)
)


@pytest.mark.parametrize(
    ("panels", "expected", "fraction"),
    [
        pytest.param(
            [{"ADT": ["CD3", "CD4"]}, {"ADT": ["CD4", "CD3"]}, {"ADT": []}],
            [_NO_ANTIBODIES],
            0.0,
            id="beside two panels",
        ),
        pytest.param(
            [{"ADT": ["CD3", "CD4"]}, {}, {"ADT": []}],
            [_MISSING_ADT.format("b"), _NO_ANTIBODIES],
            0.0,
            id="beside a panel and a source without the assay",
        ),
        pytest.param(
            [{"ADT": []}, {}, {"ADT": []}],
            [_MISSING_ADT.format("b")],
            1.0,
            id="beside another source of no antibodies",
        ),
        pytest.param(
            [{}, {}, {"ADT": []}],
            [_MISSING_ADT.format("a"), _MISSING_ADT.format("b")],
            1.0,
            id="alone",
        ),
    ],
)
def test_merge_names_a_source_whose_assay_holds_no_features(
    tmp_path, panels, expected, fraction
):
    sources = [_panel_source({"RNA": _GENES, **panel}) for panel in panels]
    destination = str(tmp_path / "merged.zarr")

    with _captured_warnings() as messages:
        merger = DataStoreMerge(sources, destination, ["a", "b", "c"], nthreads=1)
        plan = merger.plan()
        merger.dump()
    assert messages == expected
    # Source c measured no antibody, so its cells read a zero of each antibody
    # that another source measured, and no antibody was measured by every
    # source that holds the assay. Where no source measured one, no cell reads
    # a zero that its source never measured.
    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert adt.featureOverlapFraction == fraction
    rows = _rows_by_cell(destination, "ADT/counts")
    assert [rows[f"c__c{cell}"] for cell in range(3)] == [[0] * adt.nFeatures] * 3


_LEFT_PANEL = ["CD3", "CD4", "CD8", "CD45RA", "CD27"]
# The right panel lists the antibodies that it shares with the left in
# another order, so that a merge places each of them.
_RIGHT_PANEL = ["CD8", "CD14", "CD3", "CD16", "CD4", "CD19"]


def _measured_columns(location: str, assay: str = "ADT") -> dict[str, list[bool]]:
    """Return the columns of a stored feature table that record a source's measured features."""
    table = zarr.open_group(location, mode="r")[f"{assay}/featureData"]
    return {
        name: array[:].tolist()
        for name, array in table.arrays()
        if array.attrs.get("role") == "measured_features"
    }


def _panel_counts(panel: Sequence[str]) -> np.ndarray:
    """Return the counts that ``_panel_source`` writes for a panel."""
    return np.arange(1, 3 * len(panel) + 1).reshape(3, len(panel))


def test_merge_records_which_features_each_source_measured(tmp_path):
    from cytearc.metadata.membership import features_measured_by_every_source

    sources = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
        _panel_source({"RNA": _GENES}),
    ]
    destination = str(tmp_path / "merged.zarr")
    merger = DataStoreMerge(
        sources, destination, ["left", "right", "third"], nthreads=1
    )
    plan = merger.plan()
    merger.dump()

    features = zarr.open_group(destination, mode="r")["ADT/featureData"]
    ids = features["ids"][:].tolist()
    # Each source that holds an assay gets a column, true for the features of
    # its panel, whose attributes keep the source's name; the source without
    # ADT gets none.
    assert _measured_columns(destination) == {
        "measured_in_left": [name in _LEFT_PANEL for name in ids],
        "measured_in_right": [name in _RIGHT_PANEL for name in ids],
    }
    assert {
        name: dict(features[name].attrs)
        for name in ("measured_in_left", "measured_in_right")
    } == {
        "measured_in_left": {"role": "measured_features", "source": "left"},
        "measured_in_right": {"role": "measured_features", "source": "right"},
    }
    # Every source measured both genes, which needs no record.
    assert _measured_columns(destination, "RNA") == {}
    assert (
        "measured_in_left"
        not in zarr.open_group(destination, mode="r")["RNA/featureData"]
    )
    # A merge of the union records no setting in its manifest, as merges
    # written before the setting did.
    assert "features" not in plan.manifest
    # The reopened store reads them as Boolean feature columns.
    merged = _open(destination)
    frame = merged.ADT.feats.to_pandas_dataframe(
        ["names", "measured_in_left", "measured_in_right"]
    )
    shared = frame["measured_in_left"] & frame["measured_in_right"]
    assert frame["names"][shared].tolist() == ["CD3", "CD4", "CD8"]
    assert features_measured_by_every_source(merged.ADT.feats).tolist() == [
        name in {"CD3", "CD4", "CD8"} for name in ids
    ]


def test_intersection_merge_keeps_the_features_that_every_source_measured(tmp_path):
    sources = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
        _panel_source({"RNA": _GENES}),
    ]
    destination = str(tmp_path / "merged.zarr")
    merger = DataStoreMerge(
        sources,
        destination,
        ["left", "right", "third"],
        nthreads=1,
        features="intersection",
    )
    plan = merger.plan()
    merger.dump()

    adt = next(item for item in plan.assays if item.assayName == "ADT")
    assert (adt.nFeatures, adt.featureOverlapFraction) == (3, 1.0)
    assert plan.manifest["features"] == "intersection"
    # The antibodies left out of the merge sum with nothing, so the counts keep
    # the sources' dtype.
    assert adt.dtype == "uint16"
    root = zarr.open_group(destination, mode="r")
    assert root["ADT/counts"].dtype == np.uint16
    assert root["ADT/featureData/ids"][:].tolist() == ["CD3", "CD4", "CD8"]
    assert root["RNA/featureData/ids"][:].tolist() == _GENES
    # Each cell keeps its counts of the three antibodies, wherever its panel
    # holds them; the source without ADT holds zeros, which its membership
    # column marks as unmeasured.
    rows = _rows_by_cell(destination, "ADT/counts")
    left, right = _panel_counts(_LEFT_PANEL), _panel_counts(_RIGHT_PANEL)
    for cell in range(3):
        assert rows[f"left__c{cell}"] == left[cell, [0, 1, 2]].tolist()
        assert rows[f"right__c{cell}"] == right[cell, [2, 4, 0]].tolist()
        assert rows[f"third__c{cell}"] == [0, 0, 0]
    # Every source measured every antibody that the merge keeps, which needs
    # no record.
    assert _measured_columns(destination) == {}


def test_select_features_keeps_what_every_holder_measured_and_renumbers_it():
    from cytearc.merge.features import select_features

    sources = [
        _panel_source({"RNA": _GENES, "ADT": ["CD45RA", "CD3", "CD4"]}),
        _panel_source({"RNA": _GENES, "ADT": ["CD4", "CD19", "CD3"]}),
        _panel_source({"RNA": _GENES}),
    ]
    assays = [source.ADT if "ADT" in source.assay_names else None for source in sources]
    union = align_features(assays, ["left", "right", "third"])
    # The union is CD45RA, CD3, CD4, CD19, of which both holders measured the
    # middle two; the third source holds no ADT and is left out.
    assert union.mergedFeatsMap["ids"].tolist() == ["CD45RA", "CD3", "CD4", "CD19"]
    assert union.overlapFraction == 0.5
    assert select_features("ADT", union, "union") is union

    shared = select_features("ADT", union, "intersection")
    assert shared.mergedFeatsMap.to_dict("list") == {
        "idx": [0, 1],
        "names": ["CD3", "CD4"],
        "ids": ["CD3", "CD4"],
    }
    # Each source feature maps to its kept position, or to -1 when it is
    # left out.
    assert [order_map.tolist() for order_map in shared.featOrderMap] == [
        [-1, 0, 1],
        [1, -1, 0],
        [],
    ]
    assert [vector.tolist() for vector in shared.measuredFeatures] == [
        [True, True],
        [True, True],
        [False, False],
    ]
    assert (shared.nFeats, shared.overlapFraction, shared.sourcePresent) == (
        2,
        1.0,
        (True, True, False),
    )


def test_intersection_merge_refuses_sources_that_share_no_feature(tmp_path):
    # Every antibody is in two panels, so the union merges, but none is in all
    # three, so the intersection would keep none.
    panels = (["CD3", "CD4"], ["CD4", "CD3"], ["CD8", "CD14"])
    sources = [_panel_source({"RNA": _GENES, "ADT": panel}) for panel in panels]
    destination = tmp_path / "merged.zarr"

    with pytest.raises(
        ValueError,
        match=(
            "^No feature of assay 'ADT' was measured by every source that holds it, "
            "so features='intersection' keeps none"
        ),
    ):
        DataStoreMerge(
            sources, str(destination), ["a", "b", "c"], features="intersection"
        ).plan()
    assert not destination.exists()


def test_intersection_merge_keeps_an_assay_that_no_source_measured_a_feature_of(
    tmp_path,
):
    sources = [
        _panel_source({"RNA": _GENES, **panel})
        for panel in ({"ADT": []}, {}, {"ADT": []})
    ]
    destination = str(tmp_path / "merged.zarr")

    merger = DataStoreMerge(
        sources, destination, ["a", "b", "c"], nthreads=1, features="intersection"
    )
    plan = merger.plan()
    merger.dump()
    # The intersection of no features is the union, as for a merge of the
    # union: an assay of no features, which no cell reads a zero of.
    assert {item.assayName: item.nFeatures for item in plan.assays} == {
        "ADT": 0,
        "RNA": 2,
    }
    assert {item.featureOverlapFraction for item in plan.assays} == {1.0}
    assert zarr.open_group(destination, mode="r")["ADT/counts"].shape == (9, 0)

    # A source that holds the assay but none of its features measured none of
    # the features of the others, so no feature is kept.
    sources[1] = _panel_source({"RNA": _GENES, "ADT": ["CD3", "CD4"]})
    with pytest.raises(ValueError, match="^No feature of assay 'ADT' was measured"):
        DataStoreMerge(
            sources,
            str(tmp_path / "shared.zarr"),
            ["a", "b", "c"],
            features="intersection",
        ).plan()


def test_intersection_merge_warns_when_it_keeps_under_a_tenth_of_the_features(
    tmp_path,
):
    left = [f"x{index}" for index in range(11)]
    right = ["x0", *(f"y{index}" for index in range(1, 11))]
    sources = [_panel_source({"RNA": left}), _panel_source({"RNA": right})]

    with _captured_warnings() as messages:
        plan = _merge(
            sources, str(tmp_path / "merged.zarr"), features="intersection"
        ).plan()
    # The sources share x0 alone; without either, the other's 11 genes remain.
    assert messages == [
        "Assay 'RNA': the merge keeps 1 of the 21 features of its sources, since "
        "features='intersection' keeps only the features that every source "
        "measured. It would keep 11 without 'left' and 11 without 'right'."
    ]
    assert (plan.assays[0].nFeatures, plan.assays[0].featureOverlapFraction) == (
        1,
        1.0,
    )


def test_intersection_merge_by_name_sums_the_features_it_keeps(tmp_path):
    # Each source's feature that the other lacks comes before the features
    # that it keeps, so the kept columns are not the first ones.
    left = np.array([[4, 1, 2, 3], [8, 5, 6, 7]])
    right = np.array([[10, 40, 20, 30], [50, 80, 60, 70]])
    sources = [
        _open(_write_source(MemoryStore(), {"RNA": values}, names=names))
        for values, names in (
            (left, ["C", "A", "A", "B"]),
            (right, ["A", "D", "B", "B"]),
        )
    ]
    destination = str(tmp_path / "merged.zarr")

    merger = _merge(sources, destination, feature_key="names", features="intersection")
    # Two features of each source share a name, so the sums need a wider dtype.
    assert merger.plan().assays[0].dtype == "uint32"
    merger.dump()
    root = zarr.open_group(destination, mode="r")
    assert root["RNA/featureData/ids"][:].tolist() == ["A", "B"]
    # The left source leaves C out and sums its middle two features into A;
    # the right leaves D out and sums its last two into B.
    assert _rows_by_cell(destination) == {
        "left__c0": [3, 3],
        "left__c1": [11, 7],
        "right__c0": [10, 50],
        "right__c1": [50, 130],
    }
    assert _measured_columns(destination, "RNA") == {}


def test_merge_refuses_features_that_are_neither_union_nor_intersection(tmp_path):
    sources = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
    ]

    with pytest.raises(
        ValueError,
        match=re.escape("features must be 'union' or 'intersection', got 'shared'"),
    ):
        DataStoreMerge(
            sources, str(tmp_path / "merged.zarr"), _NAMES, features="shared"
        )


def test_merge_records_a_source_whose_name_holds_a_path_separator(tmp_path):
    from cytearc.metadata.membership import measured_feature_columns

    sources = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
    ]
    destination = str(tmp_path / "merged.zarr")
    names = ["GSE1/run1", "GSE2\\run1"]

    DataStoreMerge(sources, destination, names, nthreads=1).dump()
    ids = zarr.open_group(destination, mode="r")["ADT/featureData/ids"][:].tolist()
    # Zarr would nest a column named with a path separator, so each column
    # takes '_' in its place and keeps the source's name in its attributes.
    assert _measured_columns(destination) == {
        "measured_in_GSE1_run1": [name in _LEFT_PANEL for name in ids],
        "measured_in_GSE2_run1": [name in _RIGHT_PANEL for name in ids],
    }
    assert measured_feature_columns(_open(destination).ADT.feats) == {
        "measured_in_GSE1_run1": "GSE1/run1",
        "measured_in_GSE2_run1": "GSE2\\run1",
    }


@pytest.mark.parametrize(
    ("names", "columns"),
    [
        (["Ctrl", "ctrl"], "'measured_in_Ctrl' and 'measured_in_ctrl'"),
        (["a/b", "a_b"], "'measured_in_a_b' and 'measured_in_a_b'"),
    ],
    ids=["case", "separator"],
)
def test_merge_refuses_sources_whose_record_columns_would_be_one(
    tmp_path, names, columns
):
    destination = tmp_path / "merged.zarr"
    panels = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
    ]

    with pytest.raises(ValueError) as refused:
        DataStoreMerge(panels, str(destination), names, nthreads=1).plan()
    assert str(refused.value) == (
        f"Sources {names[0]!r} and {names[1]!r} would record which features they "
        f"measured in the feature columns {columns}, which a file system that "
        "ignores case stores as one column. Name the sources so that the names "
        "differ in more than case, with '/' and '\\' counted as '_'."
    )
    assert not destination.exists()
    # Sources that measured the same features need no record, and merge.
    same = [_panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}) for _ in names]
    DataStoreMerge(same, str(destination), names, nthreads=1).dump()
    assert _measured_columns(str(destination)) == {}


def test_subsets_keep_the_record_of_the_features_each_source_measured(tmp_path):
    from cytearc.metadata.membership import measured_feature_columns

    sources = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
    ]
    merged = str(tmp_path / "merged.zarr")
    _merge(sources, merged).dump()
    store = _open(merged)
    subset = str(tmp_path / "subset.zarr")
    ids = store.cells.fetch_all("ids")[[0, 5]]

    SubsetZarr(subset, [store.ADT], cell_ids=ids, nthreads=1).dump()
    assert _measured_columns(subset) == _measured_columns(merged)
    # The subset keeps each column's source, which a read of the record checks.
    reopened = DataStore(
        subset, default_assay="ADT", min_features_per_cell=0, nthreads=1
    )
    assert measured_feature_columns(reopened.ADT.feats) == {
        "measured_in_left": "left",
        "measured_in_right": "right",
    }


def test_a_merge_of_merged_stores_records_what_every_cell_of_each_source_measured(
    tmp_path,
):
    from cytearc.metadata.membership import features_measured_by_every_source

    inner = str(tmp_path / "inner.zarr")
    _merge(
        [
            _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
            _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
        ],
        inner,
    ).dump()
    third_panel = ["CD3", "CD4", "CD8", "CD45RA"]
    sources = [_open(inner), _panel_source({"RNA": _GENES, "ADT": third_panel})]

    union = str(tmp_path / "union.zarr")
    with _captured_warnings() as messages:
        DataStoreMerge(sources, union, ["inner", "third"], nthreads=1).dump()
    ids = zarr.open_group(union, mode="r")["ADT/featureData/ids"][:].tolist()
    # The warning counts what the inner store's record marks as unmeasured as
    # missing from it, though the store holds those antibodies.
    assert messages == [
        "Assay 'ADT': 5 of the 8 merged features were not measured by every "
        "source that holds the assay. Source 'inner' lacks 5: 'CD45RA', 'CD27', "
        "'CD14', 'CD16', 'CD19'. Source 'third' lacks 4: 'CD27', 'CD14', 'CD16', "
        "'CD19'." + _zeros_read_as_counts("inner", adt=True)
    ]
    # The inner store holds all eight antibodies, but the cells of its right
    # source never stained CD45RA or CD27, and those of its left source never
    # stained CD14, CD16 or CD19. Its own record describes its sources, so the
    # outer merge folds it into one column and keeps no other.
    assert _measured_columns(union) == {
        "measured_in_inner": [name in {"CD3", "CD4", "CD8"} for name in ids],
        "measured_in_third": [name in third_panel for name in ids],
    }
    assert features_measured_by_every_source(_open(union).ADT.feats).tolist() == [
        name in {"CD3", "CD4", "CD8"} for name in ids
    ]

    # Both sources hold CD45RA, but not every cell of the inner one measured
    # it, so the intersection leaves it out.
    shared = str(tmp_path / "shared.zarr")
    DataStoreMerge(
        sources, shared, ["inner", "third"], nthreads=1, features="intersection"
    ).dump()
    assert zarr.open_group(shared, mode="r")["ADT/featureData/ids"][:].tolist() == [
        "CD3",
        "CD4",
        "CD8",
    ]


def test_an_intersection_of_a_merged_store_warns_of_what_its_cells_measured(
    tmp_path,
):
    left = [f"x{index}" for index in range(11)]
    right = ["x0", *(f"y{index}" for index in range(1, 11))]
    inner = str(tmp_path / "inner.zarr")
    _merge([_panel_source({"RNA": left}), _panel_source({"RNA": right})], inner).dump()
    # The inner store holds all 21 genes, but only x0 in every one of its
    # cells, and the third source measured all 21.
    sources = [_open(inner), _panel_source({"RNA": [*left, *right[1:]]})]

    with _captured_warnings() as messages:
        plan = DataStoreMerge(
            sources,
            str(tmp_path / "shared.zarr"),
            ["inner", "third"],
            nthreads=1,
            features="intersection",
        ).plan()
    assert messages == [
        "Assay 'RNA': the merge keeps 1 of the 21 features of its sources, since "
        "features='intersection' keeps only the features that every source "
        "measured. It would keep 21 without 'inner' and 1 without 'third'."
    ]
    assert plan.assays[0].nFeatures == 1


def test_merge_leaves_out_a_source_column_that_its_record_would_replace(tmp_path):
    left = _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL})
    left.ADT.feats.insert("measured_in_right", np.ones(len(_LEFT_PANEL), dtype=bool))
    left.ADT.feats.insert("clone", np.asarray([f"clone{i}" for i in range(5)]))
    right = _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL})
    destination = str(tmp_path / "merged.zarr")

    with _captured_warnings() as messages:
        _merge([left, right], destination).dump()
    assert (
        "Feature columns of the sources were not merged, since the merge records "
        "in each which features one source measured: measured_in_right"
    ) in messages
    ids = zarr.open_group(destination, mode="r")["ADT/featureData/ids"][:].tolist()
    assert _measured_columns(destination)["measured_in_right"] == [
        name in _RIGHT_PANEL for name in ids
    ]
    # Other annotations merge as before.
    clone = zarr.open_group(destination, mode="r")["ADT/featureData/clone"][:]
    assert clone[: len(_LEFT_PANEL)].tolist() == [f"clone{i}" for i in range(5)]


def test_a_feature_table_without_a_record_counts_every_feature_as_measured(tmp_path):
    from cytearc.metadata.membership import (
        features_measured_by_every_source,
        measured_feature_columns,
    )

    destination = str(tmp_path / "merged.zarr")
    sources = [
        _panel_source({"RNA": _GENES, "ADT": _LEFT_PANEL}),
        _panel_source({"RNA": _GENES, "ADT": _RIGHT_PANEL}),
    ]
    _merge(sources, destination).dump()
    # A store merged by an earlier release holds no record.
    root = zarr.open_group(destination, mode="r+")
    for name in ("measured_in_left", "measured_in_right"):
        del root[f"ADT/featureData/{name}"]
    earlier = _open(destination)
    assert measured_feature_columns(earlier.ADT.feats) == {}
    assert features_measured_by_every_source(earlier.ADT.feats).all()


@pytest.mark.parametrize(
    ("values", "attributes", "problem"),
    [
        pytest.param(
            pd.array([True, None, False], dtype="boolean"),
            {"role": "measured_features"},
            "its source attribute, None, does not name it",
            id="copied by an earlier release",
        ),
        pytest.param(
            np.array([True, True, False]),
            {"role": "measured_features", "source": "right"},
            "its source attribute, 'right', does not name it",
            id="of another source",
        ),
        pytest.param(
            np.ones(3, dtype=np.int8),
            {"role": "measured_features", "source": "left"},
            "it has dtype int8, not bool",
            id="not Boolean",
        ),
        pytest.param(
            pd.array([True, None, False], dtype="boolean"),
            {"role": "measured_features", "source": "left"},
            "it has missing values",
            id="with missing values",
        ),
    ],
)
def test_a_malformed_record_of_measured_features_is_refused_with_its_remedy(
    values, attributes, problem
):
    from cytearc.metadata.membership import features_measured_by_every_source

    source = _panel_source({"RNA": _GENES, "ADT": ["CD3", "CD4", "CD8"]})
    source.ADT.feats.insert("measured_in_left", values)
    source.ADT.feats._get_array("measured_in_left").attrs.update(attributes)

    with pytest.raises(ValueError) as refused:
        features_measured_by_every_source(source.ADT.feats)
    assert str(refused.value) == (
        "Feature column 'measured_in_left' has the role of a record of which "
        f"features one source of a merge measured, but {problem}. Earlier "
        "releases of CyteArc copy such a column without its source and with "
        "missing values when they merge stores. Drop it with "
        "ds.ADT.feats.drop('measured_in_left'), or merge the sources again to "
        "record what each one measured."
    )
    # The call that the message names leaves no record.
    source.ADT.feats.drop("measured_in_left")
    assert features_measured_by_every_source(source.ADT.feats).all()


def test_the_remedy_of_a_malformed_record_names_an_assay_that_is_no_attribute():
    from cytearc.metadata.membership import measured_feature_columns

    source = _panel_source({"RNA": _GENES, "ADT-2": ["CD3", "CD4", "CD8"]})
    features = source.get_assay("ADT-2").feats
    features.insert("measured_in_left", np.array([True, False, True]))
    features._get_array("measured_in_left").attrs["role"] = "measured_features"

    # ds.ADT-2 is no Python expression, so the message names get_assay.
    with pytest.raises(
        ValueError,
        match=re.escape("Drop it with ds.get_assay('ADT-2').feats.drop("),
    ):
        measured_feature_columns(features)


def test_merge_by_name_sums_signed_counts_in_a_wider_dtype(tmp_path):
    sources = _named_sources(np.array([[3, -2, 1], [0, 4, 2]]), "int16")
    destination = str(tmp_path / "merged.zarr")
    merger = _merge(sources, destination, feature_key="names")

    assert merger.plan().assays[0].dtype == "int32"
    merger.dump()
    root = zarr.open_group(destination, mode="r")
    assert root["RNA/counts"].dtype == np.int32
    # The left source sums features 0 and 1 into A, the right 1 and 2 into B.
    assert _rows_by_cell(destination) == {
        "left__c0": [1, 1],
        "left__c1": [4, 2],
        "right__c0": [3, -1],
        "right__c1": [0, 6],
    }


def test_merge_by_name_keeps_uint64_counts_it_cannot_widen(tmp_path):
    sources = _named_sources(np.array([[3, 2, 1], [0, 4, 2]]), "uint64")
    destination = str(tmp_path / "merged.zarr")
    merger = _merge(sources, destination, feature_key="names")

    # No wider integer exists, so the sums are stored as uint64.
    assert merger.plan().assays[0].dtype == "uint64"
    merger.dump()
    assert zarr.open_group(destination, mode="r")["RNA/counts"].dtype == np.uint64
    assert _rows_by_cell(destination) == {
        "left__c0": [5, 1],
        "left__c1": [4, 2],
        "right__c0": [3, 3],
        "right__c1": [0, 6],
    }
    # A sum that overflows raises while it is written: the two features that
    # this left source names A sum to 2**64.
    overflowing = _open(
        _write_source(
            MemoryStore(),
            {"RNA": np.array([[2**63, 2**63, 1], [0, 4, 2]], dtype=np.uint64)},
            names=["A", "A", "B"],
            dtype="uint64",
        )
    )
    merger = _merge(
        [overflowing, sources[1]], str(tmp_path / "overflow.zarr"), feature_key="names"
    )
    assert merger.plan().assays[0].dtype == "uint64"
    with pytest.raises(
        OverflowError, match="^Duplicate sparse values exceed supported integer dtypes$"
    ):
        merger.dump()


# Repacking


def test_repack_copies_arrays_of_any_dimensionality(tmp_path, prepared):
    source = _copy(prepared, tmp_path)
    extra = zarr.open_group(source, mode="r+").create_group("extra")
    cube = extra.create_array(
        "cube", data=np.arange(24, dtype=np.float32).reshape(2, 3, 4), chunks=(1, 3, 4)
    )
    cube.attrs["kind"] = "kernel"
    scalar = extra.create_array("scalar", data=np.array(7, dtype=np.int64))
    scalar.attrs["kind"] = "scalar"
    output = str(tmp_path / "repacked.zarr")

    repack_store(source, output)
    copied = zarr.open_group(output, mode="r")["extra"]
    np.testing.assert_array_equal(copied["cube"][:], cube[:])
    assert copied["cube"].chunks == (1, 3, 4)
    assert copied["cube"].attrs["kind"] == "kernel"
    assert copied["scalar"][...] == 7
    assert copied["scalar"].attrs["kind"] == "scalar"


@pytest.mark.parametrize("record", ["elsewhere", {"location": 5}, {}])
def test_repack_refuses_a_malformed_matrix_source_record(tmp_path, prepared, record):
    source = _copy(prepared, tmp_path)
    zarr.open_group(source, mode="r+").attrs[MATRIX_SOURCE_ATTR] = record
    output = tmp_path / "repacked.zarr"

    with pytest.raises(ValueError, match="Malformed matrix source location"):
        repack_store(source, str(output))
    assert not output.exists()


def test_repack_refuses_a_destination_inside_the_mounted_count_owner(
    tmp_path, prepared
):
    source = _copy(prepared, tmp_path)
    target = str(tmp_path / "mount.zarr")
    mount_datastore(
        source, at=target, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    output = Path(source) / "copy.zarr"

    with pytest.raises(ValueError, match="overlaps the mounted count owner"):
        repack_store(target, str(output))
    assert not output.exists()


def test_repack_refuses_a_store_without_assays(tmp_path):
    source = str(tmp_path / "empty.zarr")
    zarr.open_group(source, mode="w")

    with pytest.raises(ValueError, match="No logical assays found"):
        repack_store(source, str(tmp_path / "repacked.zarr"))


@pytest.mark.parametrize(
    "marker", ["cytearc:import_complete", "external:import_complete"]
)
@pytest.mark.parametrize("data_only", [False, True])
def test_repack_refuses_an_interrupted_import(tmp_path, prepared, marker, data_only):
    source = _copy(prepared, tmp_path)
    zarr.open_group(source, mode="r+").attrs[marker] = False
    output = tmp_path / "repacked.zarr"

    with pytest.raises(ValueError, match="An incomplete import cannot be repacked"):
        repack_store(source, str(output), data_only=data_only)
    assert not output.exists()


def test_repack_module_repacks_a_store_from_the_command_line(
    tmp_path, monkeypatch, capsys, prepared
):
    source = _copy(prepared, tmp_path)
    output = str(tmp_path / "repacked.zarr")
    monkeypatch.setattr(sys, "argv", ["repack_zarr", source, output, "--nthreads", "1"])
    # Executed as a script, the module must not shadow its imported copy.
    monkeypatch.delitem(sys.modules, "cytearc.tools.repack_zarr")

    runpy.run_module("cytearc.tools.repack_zarr", run_name="__main__")
    assert f"Repacked {source} -> {output}" in capsys.readouterr().out
    np.testing.assert_array_equal(_merged_counts(output), _COUNTS)


@pytest.mark.parametrize("flag", ["--storage-options", "--source-storage-options"])
@pytest.mark.parametrize(
    ("options", "message"),
    [("{not json", "must be valid JSON"), ("[1]", "must be a JSON object")],
)
def test_repack_command_line_rejects_invalid_storage_options(
    monkeypatch, flag, options, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["repack_zarr", "in.zarr", "out.zarr", flag, options],
    )
    with pytest.raises(SystemExit, match=f"^{flag} {message}"):
        repack_zarr.main()


def test_repack_command_line_passes_source_storage_options(monkeypatch):
    calls = []
    monkeypatch.setattr(
        repack_zarr, "repack_store", lambda *args, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repack_zarr",
            "s3://bucket/mount.zarr",
            "out.zarr",
            "--storage-options",
            '{"region": "eu-west-1"}',
            "--source-storage-options",
            '{"token": false}',
        ],
    )
    repack_zarr.main()
    assert calls[0]["storage_options"] == {"region": "eu-west-1"}
    assert calls[0]["source_storage_options"] == {"token": False}
