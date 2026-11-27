"""Guide calls imported from Cell Ranger's call file for an eight-cell screen.

The module writes the screen and its selection of seven cells once, as a
template. A test that writes copies the template, and the tests of refused
imports share one copy, which they check is left as it was.
"""

import gc
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zarr

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import MULTIPLE, NEGATIVE, GuideCalls
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.identity import replace_metadata_column
from cytearc.storage.selections import read_stored_selection_indices
from cytearc.utils.logging import collect_warnings, logger
from cytearc.utils.memory_budget import MemoryBudgetError, MemoryLedger
from tests.perturbation_helpers import record_cell_reads, write_screen
from tests.storage_helpers import (
    by_cell_id,
    copy_store,
    insert_nullable_cell_column,
    open_store,
    rechunk_feature_tables,
    write_count_store,
    written,
)

# Guides are named CRISPR0 to CRISPR3 and cells cell0 to cell7.
GUIDE_COUNTS = np.asarray(
    [
        [30, 0, 1, 0],  # cell0: CRISPR0 called
        [0, 25, 0, 2],  # cell1: CRISPR1 called
        [20, 20, 0, 0],  # cell2: both called; the tie puts CRISPR0 first
        [0, 0, 0, 0],  # cell3: no guide UMIs
        [1, 0, 2, 0],  # cell4: ambient UMIs only, no call
        [0, 0, 40, 3],  # cell5: CRISPR2 and CRISPR3 called
        [5, 5, 5, 5],  # cell6: called, but left out of the selection
        [0, 9, 0, 0],  # cell7: CRISPR1 called
    ],
    dtype=np.uint16,
)
CALL_FILE = (
    "cell_barcode,num_features,feature_call,num_umis\n"
    "cell0,1,CRISPR0,30\n"
    "cell1,1,CRISPR1,25\n"
    "cell2,2,CRISPR1|CRISPR0,20|20\n"
    "cell5,2,CRISPR3|CRISPR2,3|40\n"
    "cell6,1,CRISPR3,5\n"
    "cell7,1,CRISPR1,9\n"
    "outside-1,1,CRISPR0,4\n"
)
SELECTED = ["cell0", "cell1", "cell2", "cell3", "cell4", "cell5", "cell7"]
# The per-cell arrays of guide calls, which artifacts.load_values reads.
GUIDE_CALL_ARRAYS = (
    "values",
    "n_guides",
    "top_guide",
    "top_umi",
    "top_fraction",
    "total_umi",
    "guide_codes",
    "guide_umis",
)


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _store(path: Path, dtype: str = "uint16") -> DataStore:
    rna = np.random.default_rng(7).poisson(2.0, size=(len(GUIDE_COUNTS), 10))
    return write_screen(path, rna, GUIDE_COUNTS, dtype=dtype)


def _selection(store: DataStore):
    keep = np.ones(len(GUIDE_COUNTS), dtype=bool)
    keep[6] = False
    store.cells.insert("screened", by_cell_id(store.cells, keep))
    return store.snapshot_cell_selection("screened")


def _reader(store: DataStore) -> DataStore:
    return DataStore(store.zarr_loc, default_assay="RNA", zarr_mode="r", nthreads=1)


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Iterator[SimpleNamespace]:
    """The screen and its selection of every cell but cell6."""
    path = tmp_path_factory.mktemp("guide_call_imports") / "store.zarr"
    store = _store(path)
    screen = SimpleNamespace(path=path, selection=_selection(store))
    saved = written(store)
    yield screen
    assert written(open_store(path)) == saved


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    """A copy of the template, for a test that writes."""
    return copy_store(template.path, tmp_path)


@pytest.fixture(scope="module")
def shared_store(template, tmp_path_factory) -> DataStore:
    """One writable copy of the template for the imports that must be refused."""
    return copy_store(template.path, tmp_path_factory.mktemp("imports_refused"))


def _call_file(tmp_path: Path, text: str = CALL_FILE, name: str = "calls.csv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_import_saves_calls_and_count_summaries(store, template, tmp_path) -> None:
    selection = template.selection

    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=selection
    )

    assert ref.kind == "guide_calls"
    assert ref.assay == "CRISPR"
    status = store.artifacts.inspect(ref)
    assert status.operation == "import_guide_calls"
    assert status.parameters == {"barcode_prefix": None}
    loaded = store.perturbation.load_guide_calls(ref)
    assert isinstance(loaded, GuideCalls)
    assert loaded.source == "import_guide_calls"
    assert loaded.guides.tolist() == ["CRISPR0", "CRISPR1", "CRISPR2", "CRISPR3"]
    cells = loaded.cells
    assert cells.index.tolist() == SELECTED
    assert cells["call"].tolist() == [
        "CRISPR0",
        "CRISPR1",
        MULTIPLE,
        NEGATIVE,
        NEGATIVE,
        MULTIPLE,
        "CRISPR1",
    ]
    assert cells["n_guides"].tolist() == [1, 1, 2, 0, 0, 2, 1]
    assert cells["top_guide"].tolist() == [
        "CRISPR0",
        "CRISPR1",
        "CRISPR0",
        "",
        "CRISPR2",
        "CRISPR2",
        "CRISPR1",
    ]
    assert cells["top_umi"].tolist() == [30, 25, 20, 0, 2, 40, 9]
    assert cells["total_umi"].tolist() == [31, 27, 40, 0, 3, 43, 9]
    np.testing.assert_allclose(
        cells["top_fraction"],
        [30 / 31, 25 / 27, 0.5, np.nan, 2 / 3, 40 / 43, 1.0],
        equal_nan=True,
    )
    assert loaded.calls.to_dict("list") == {
        "ids": ["cell0", "cell1", "cell2", "cell2", "cell5", "cell5", "cell7"],
        "guide": [
            "CRISPR0",
            "CRISPR1",
            "CRISPR0",
            "CRISPR1",
            "CRISPR2",
            "CRISPR3",
            "CRISPR1",
        ],
        "umis": [30, 25, 20, 20, 40, 3, 9],
    }
    codes = store.artifacts.load_values(ref, value="guide_codes")
    assert np.asarray(codes.values).tolist() == [
        [0, -1],
        [1, -1],
        [0, 1],
        [-1, -1],
        [-1, -1],
        [2, 3],
        [1, -1],
    ]
    # cell6 is a cell of the store outside the selection, and no cell is
    # named outside-1. Each barcode is its cell's ID, which its example shows
    # once.
    assert status.diagnostics == {
        "fileBarcodes": 7,
        "unmatchedBarcodes": 1,
        "unselectedBarcodes": 1,
        "unmatchedBarcodeExamples": ["outside-1"],
        "unselectedBarcodeExamples": ["cell6"],
    }
    assert "CRISPR_n_guides" not in store.cells.columns


def test_select_cells_reads_the_saved_calls(store, template, tmp_path) -> None:
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=template.selection
    )

    single = store.select_cells(ref, value="n_guides", low=1, high=1, keep_bounds=True)
    carrying = store.select_cells(ref, include=["CRISPR1"])

    names = np.asarray(store.cells.fetch_all("ids"))

    def selected(selection):
        return names[
            read_stored_selection_indices(
                store.zw,
                selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
        ].tolist()

    assert selected(single) == ["cell0", "cell1", "cell7"]
    assert selected(carrying) == ["cell1", "cell7"]


def test_import_is_reused_by_file_contents_not_by_path(
    store, template, tmp_path
) -> None:
    selection = template.selection
    first = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=selection
    )

    moved = _call_file(tmp_path, name="moved.csv")
    assert (
        store.perturbation.import_guide_calls(moved, cell_selection=selection) == first
    )
    edited = _call_file(
        tmp_path,
        CALL_FILE.replace("cell7,1,CRISPR1,9", "cell7,1,CRISPR2,9"),
        "edited.csv",
    )
    changed = store.perturbation.import_guide_calls(edited, cell_selection=selection)
    assert changed != first
    assert (
        store.perturbation.load_guide_calls(changed).cells.loc["cell7", "call"]
        == "CRISPR2"
    )
    again = store.perturbation.import_guide_calls(
        moved, cell_selection=selection, invalidate_cache=True
    )
    assert again != first


def test_a_barcode_prefix_joins_barcodes_to_cell_ids(store, template, tmp_path) -> None:
    text = (
        "cell_barcode,num_features,feature_call,num_umis\n"
        "0,1,CRISPR0,30\n"
        "5,1,CRISPR2,40\n"
        "6,1,CRISPR0,4\n"
    )
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path, text),
        cell_selection=template.selection,
        barcode_prefix="cell",
    )

    cells = store.perturbation.load_guide_calls(ref).cells
    assert cells.loc[["cell0", "cell5"], "call"].tolist() == ["CRISPR0", "CRISPR2"]
    # Barcode 6 names cell6, a cell of the store outside the selection, and
    # its example shows the barcode beside the cell ID it derives.
    diagnostics = store.artifacts.inspect(ref).diagnostics
    assert diagnostics["unmatchedBarcodes"] == 0
    assert diagnostics["unselectedBarcodes"] == 1
    assert diagnostics["unselectedBarcodeExamples"] == ["6 -> cell6"]


def _logged_import(
    store: DataStore, path: Path, **arguments
) -> tuple[ArtifactRef, list[str], list[str]]:
    """Import guide calls; return them, the warnings, and the notes on unselected cells."""
    notes: list[str] = []
    sink = logger.add(
        lambda message: notes.append(message.record["message"]),
        level="INFO",
        filter=lambda record: "outside the selected cells" in record["message"],
    )
    try:
        with collect_warnings() as messages:
            calls = store.perturbation.import_guide_calls(path, **arguments)
    finally:
        logger.remove(sink)
    return calls, list(messages), notes


def test_a_reused_import_logs_the_warning_of_the_first(
    store, template, tmp_path
) -> None:
    selection = template.selection
    path = _call_file(tmp_path)

    logged = _logged_import(store, path, cell_selection=selection)

    # No cell is named outside-1, which a wrong barcode_prefix could cause;
    # cell6 is a cell outside the selection, which needs no warning.
    assert logged[1:] == (
        [
            "1 of 7 barcodes of the call file match no cell of the store, for "
            "example 'outside-1'. A cell matches a barcode when its ID is "
            "barcode_prefix followed by the barcode, so check barcode_prefix when "
            "the cell IDs carry a sample prefix."
        ],
        [
            "1 of 7 barcodes of the call file are cells of the store outside the "
            "selected cells, for example 'cell6'; the import labels the selected "
            "cells only"
        ],
    )
    # The saved calls log both again, alike: for the same file under another
    # name, and in a store opened read-only.
    moved = _call_file(tmp_path, name="moved.csv")
    assert _logged_import(store, moved, cell_selection=selection) == logged
    reader = _reader(store)
    assert _logged_import(reader, path, cell_selection=selection) == logged
    # Calls whose every barcode joins a selected cell log nothing, imported or
    # reused.
    joined = _call_file(
        tmp_path,
        "cell_barcode,num_features,feature_call,num_umis\n"
        "0,1,CRISPR0,30\n"
        "5,1,CRISPR2,40\n",
        "joined.csv",
    )
    arguments = {"cell_selection": selection, "barcode_prefix": "cell"}
    quiet = _logged_import(store, joined, **arguments)
    assert quiet[1:] == ([], [])
    assert _logged_import(store, joined, **arguments) == quiet


def test_barcodes_of_unselected_cells_need_no_warning(
    store, template, tmp_path
) -> None:
    # A selection that leaves out cell6, as QC filtering leaves out cells, and
    # a file whose every barcode is a cell of the store.
    path = _call_file(tmp_path, CALL_FILE.replace("outside-1,1,CRISPR0,4\n", ""))

    calls, messages, notes = _logged_import(
        store, path, cell_selection=template.selection
    )

    assert messages == []
    assert notes == [
        "1 of 6 barcodes of the call file are cells of the store outside the "
        "selected cells, for example 'cell6'; the import labels the selected "
        "cells only"
    ]
    diagnostics = store.artifacts.inspect(calls).diagnostics
    assert diagnostics["unmatchedBarcodes"] == 0
    assert diagnostics["unselectedBarcodes"] == 1


def test_an_empty_barcode_prefix_is_no_prefix(store, template, tmp_path) -> None:
    path = _call_file(tmp_path)

    plain = store.perturbation.import_guide_calls(
        path, cell_selection=template.selection
    )
    empty = store.perturbation.import_guide_calls(
        path, cell_selection=template.selection, barcode_prefix=""
    )

    # The two imports are one, so later steps reuse each other's results.
    assert empty == plain
    assert store.artifacts.inspect(empty).parameters == {"barcode_prefix": None}


def test_an_import_reaches_the_cells_of_one_sample_of_a_merged_store(
    store, tmp_path
) -> None:
    # A store merged from two samples names each cell <sample>__<barcode>.
    ids = np.asarray([f"{'s1' if cell < 4 else 's2'}__cell{cell}" for cell in range(8)])
    zarr.open_group(str(Path(store.zarr_loc) / "cellData"), mode="r+").create_array(
        "ids", data=ids, overwrite=True
    )
    merged = open_store(store.zarr_loc)
    merged.cells.insert("every", by_cell_id(merged.cells, np.ones(8, dtype=bool)))
    merged.cells.insert("first", by_cell_id(merged.cells, np.arange(8) < 4))
    every = merged.snapshot_cell_selection("every")
    first = merged.snapshot_cell_selection("first")
    path = _call_file(tmp_path)
    remedy = "or call the guides from their counts with perturbation.call_guides"
    saved = written(merged)

    # Without a prefix, every cell of the other sample would be labeled
    # Negative; with one, every selected cell outside its sample would be.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The call file holds the barcodes of one sample, but the cells of "
            "cell_selection come from several samples of a merged store, such as "
            "'s1' and 's2'; select only the cells of one sample and pass its "
            f"barcode_prefix, such as 's1__', {remedy}"
        ),
    ):
        merged.perturbation.import_guide_calls(path, cell_selection=every)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The call file's barcodes reach only the cells whose IDs start with "
            "barcode_prefix 's1__', but 4 of the cells of cell_selection, such as "
            f"'s2__cell4', do not; select only the cells of that sample, {remedy}"
        ),
    ):
        merged.perturbation.import_guide_calls(
            path, cell_selection=every, barcode_prefix="s1__"
        )
    assert written(merged) == saved

    calls = merged.perturbation.import_guide_calls(
        path, cell_selection=first, barcode_prefix="s1__"
    )
    cells = merged.perturbation.load_guide_calls(calls).cells
    assert cells["call"].tolist() == ["CRISPR0", "CRISPR1", MULTIPLE, NEGATIVE]


_REMEDY = "or call the guides from their counts with perturbation.call_guides"


def _merged_screen(
    store: DataStore, samples: tuple[str, str], *, sources: bool
) -> DataStore:
    """The screen as a store merged from two samples, cells 0 to 3 of the first and 4 to 7 of the second.

    A merged store names each cell ``<sample>__<barcode>``. With ``sources``,
    it also names each cell's sample in the categorical cell column
    ``dataset``, with the role that a merge gives that column when its
    sources measured different features. The cell columns ``every``,
    ``first`` and ``second`` select the cells of both samples and of each,
    and ``boundary`` the last cell of the first sample and the first of the
    second.
    """
    codes = (np.arange(8) >= 4).astype(np.int64)
    ids = np.asarray(
        [f"{samples[code]}__cell{cell}" for cell, code in enumerate(codes)]
    )
    cells = zarr.open_group(str(Path(store.zarr_loc) / "cellData"), mode="r+")
    cells.create_array("ids", data=ids, overwrite=True)
    if sources:
        replace_metadata_column(cells, "dataset", codes, categories=np.asarray(samples))
        cells["dataset"].attrs.update({"role": "cell_sources"})
    merged = open_store(store.zarr_loc)
    for name, keep in (
        ("every", np.ones(8, dtype=bool)),
        ("first", codes == 0),
        ("second", codes == 1),
        ("boundary", np.isin(np.arange(8), [3, 4])),
    ):
        merged.cells.insert(name, by_cell_id(merged.cells, keep))
    return merged


def _unclear_ids(example: str, name: str) -> str:
    """The refusal of the IDs of the four cells of sample ``name + '_'``, without a column of sources."""
    return re.escape(
        "The call file holds the barcodes of one sample, but the IDs of 4 of the "
        f"cells of cell_selection, such as {example!r}, do not tell their sample: "
        "a merged store names each cell <sample>__<barcode>, so that cell is a "
        f"cell of {name + '_'!r} or a cell of {name!r} whose barcode starts with "
        "'_'; select only the cells of one sample and pass its barcode_prefix, "
        f"such as {name + '___'!r} for {name + '_'!r}, {_REMEDY}"
    )


def _several_samples(first: str, second: str) -> str:
    """The refusal of the cells of two samples that a column of sources names."""
    return re.escape(
        "The call file holds the barcodes of one sample, but the cells of "
        "cell_selection come from several samples of a merged store, such as "
        f"{first!r} and {second!r}; select only the cells of one sample and pass "
        f"its barcode_prefix, such as {first + '__'!r}, {_REMEDY}"
    )


@pytest.mark.parametrize("sources", [False, True], ids=["ids", "column"])
def test_an_import_tells_apart_samples_whose_names_differ_by_a_trailing_underscore(
    store, tmp_path, sources
) -> None:
    merged = _merged_screen(store, ("pool", "pool_"), sources=sources)
    every = merged.snapshot_cell_selection("every")
    boundary = merged.snapshot_cell_selection("boundary")
    path = _call_file(tmp_path)
    saved = written(merged)

    # The cells of pool_ are not read as cells of pool: without a prefix, the
    # cells of the two samples are refused together, by their IDs, which do
    # not tell the sample of a cell of pool_, or by the column of sources.
    with pytest.raises(
        ValueError,
        match=_several_samples("pool", "pool_")
        if sources
        else _unclear_ids("pool___cell4", "pool"),
    ):
        merged.perturbation.import_guide_calls(path, cell_selection=every)
    # The samples are those of the selected cells, here one of each.
    with pytest.raises(ValueError, match="several samples|do not tell their"):
        merged.perturbation.import_guide_calls(path, cell_selection=boundary)
    # With pool's prefix, the cells of pool_ go on from it with "_", and are
    # refused rather than labeled Negative.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The call file's barcodes reach the cells whose IDs are "
            "barcode_prefix 'pool__' and a barcode, but 4 of the cells of "
            "cell_selection, such as 'pool___cell4', go on from it with '_', as "
            "the cells of a sample whose name ends with '_' do; select only the "
            "cells of one sample and pass its barcode_prefix, such as 'pool___' "
            f"for those cells, {_REMEDY}"
        ),
    ):
        merged.perturbation.import_guide_calls(
            path, cell_selection=every, barcode_prefix="pool__"
        )
    assert written(merged) == saved

    # Each sample's prefix reaches its own cells only.
    for selection, prefix, expected in (
        ("first", "pool__", ["CRISPR0", "CRISPR1", MULTIPLE, NEGATIVE]),
        ("second", "pool___", [NEGATIVE, MULTIPLE, "CRISPR3", "CRISPR1"]),
    ):
        calls = merged.perturbation.import_guide_calls(
            path,
            cell_selection=merged.snapshot_cell_selection(selection),
            barcode_prefix=prefix,
        )
        cells = merged.perturbation.load_guide_calls(calls).cells
        assert cells["call"].tolist() == expected


def test_a_merge_of_pool_and_pool_underscore_tells_the_import_each_cells_sample(
    tmp_path,
) -> None:
    from cytearc.merge import DataStoreMerge

    # Two screens whose RNA panels differ, so their merge names each cell's
    # sample in its column of sources.
    rng = np.random.default_rng(3)
    sources = []
    for name, genes in (("pool", 9), ("pool_", 10)):
        location = tmp_path / f"{name}.zarr"
        write_count_store(
            str(location),
            {"RNA": rng.poisson(2.0, size=(8, genes)) + 1, "CRISPR": GUIDE_COUNTS},
            "uint16",
        )
        sources.append(open_store(location))
    DataStoreMerge(
        sources, str(tmp_path / "merged.zarr"), ["pool", "pool_"], seed=0, nthreads=1
    ).dump()
    merged = open_store(tmp_path / "merged.zarr")
    samples = merged.cells.fetch_all("dataset").astype(str)
    ids = merged.cells.fetch_all("ids").astype(str)
    merged.cells.insert("every", by_cell_id(merged.cells, np.ones(16, dtype=bool)))
    merged.cells.insert("pool_", by_cell_id(merged.cells, samples == "pool_"))
    every = merged.snapshot_cell_selection("every")
    path = _call_file(tmp_path)
    first = str(samples[0])

    with pytest.raises(
        ValueError,
        match=_several_samples(first, "pool_" if first == "pool" else "pool"),
    ):
        merged.perturbation.import_guide_calls(path, cell_selection=every)
    # With pool's prefix, the cells of pool_ are refused, the first of them as
    # the example.
    example = str(ids[np.argmax(samples == "pool_")])
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"but 8 of the cells of cell_selection, such as {example!r}, go on "
            "from it with '_'"
        ),
    ):
        merged.perturbation.import_guide_calls(
            path, cell_selection=every, barcode_prefix="pool__"
        )

    calls = merged.perturbation.import_guide_calls(
        path,
        cell_selection=merged.snapshot_cell_selection("pool_"),
        barcode_prefix="pool___",
    )
    called = merged.perturbation.load_guide_calls(calls).cells["call"]
    # Cell Ranger's calls of cell0 to cell7, which pool_'s cells take.
    expected = dict(
        zip(
            [f"pool___cell{cell}" for cell in range(8)],
            [
                "CRISPR0",
                "CRISPR1",
                MULTIPLE,
                NEGATIVE,
                NEGATIVE,
                MULTIPLE,
                "CRISPR3",
                "CRISPR1",
            ],
            strict=True,
        )
    )
    assert called.to_dict() == expected


def test_an_import_reaches_one_sample_of_a_merge_of_a_merged_screen(tmp_path) -> None:
    from cytearc.merge import DataStoreMerge

    # Two screens whose RNA panels differ, merged and then merged with a
    # third, so that the outer store names each cell's sample, such as
    # inner__pool1, in its column of sources.
    rng = np.random.default_rng(3)
    for name, genes in (("pool1", 9), ("pool2", 10), ("third", 11)):
        write_count_store(
            str(tmp_path / f"{name}.zarr"),
            {"RNA": rng.poisson(2.0, size=(8, genes)) + 1, "CRISPR": GUIDE_COUNTS},
            "uint16",
        )
    DataStoreMerge(
        [open_store(tmp_path / "pool1.zarr"), open_store(tmp_path / "pool2.zarr")],
        str(tmp_path / "inner.zarr"),
        ["pool1", "pool2"],
        seed=0,
        nthreads=1,
    ).dump()
    DataStoreMerge(
        [open_store(tmp_path / "inner.zarr"), open_store(tmp_path / "third.zarr")],
        str(tmp_path / "outer.zarr"),
        ["inner", "third"],
        seed=0,
        nthreads=1,
    ).dump()
    merged = open_store(tmp_path / "outer.zarr")
    samples = merged.cells.fetch_all("dataset").astype(str)
    ids = merged.cells.fetch_all("ids").astype(str)
    assert sorted(set(samples)) == ["inner__pool1", "inner__pool2", "third"]
    inner = np.char.startswith(ids, "inner__")
    merged.cells.insert("inner_cells", by_cell_id(merged.cells, inner))
    merged.cells.insert("pool1", by_cell_id(merged.cells, samples == "inner__pool1"))
    path = _call_file(tmp_path)

    # The cells of both samples of the inner screen are refused together, and
    # the refusal suggests the prefix of the first one's sample.
    first = str(samples[inner][0])
    with pytest.raises(
        ValueError,
        match=_several_samples(
            first, "inner__pool2" if first == "inner__pool1" else "inner__pool1"
        ),
    ):
        merged.perturbation.import_guide_calls(
            path, cell_selection=merged.snapshot_cell_selection("inner_cells")
        )
    # The suggested prefix, the sample's name and '__', starts the IDs of its
    # cells, which take Cell Ranger's calls of cell0 to cell7.
    calls = merged.perturbation.import_guide_calls(
        path,
        cell_selection=merged.snapshot_cell_selection("pool1"),
        barcode_prefix="inner__pool1__",
    )
    called = merged.perturbation.load_guide_calls(calls).cells["call"]
    assert called.to_dict() == dict(
        zip(
            [f"inner__pool1__cell{cell}" for cell in range(8)],
            [
                "CRISPR0",
                "CRISPR1",
                MULTIPLE,
                NEGATIVE,
                NEGATIVE,
                MULTIPLE,
                "CRISPR3",
                "CRISPR1",
            ],
            strict=True,
        )
    )


@pytest.mark.parametrize("sources", [False, True], ids=["ids", "column"])
def test_a_refusal_suggests_the_prefix_of_a_sample_whose_name_ends_with_an_underscore(
    store, tmp_path, sources
) -> None:
    merged = _merged_screen(store, ("pool_", "other"), sources=sources)
    path = _call_file(tmp_path)

    with pytest.raises(
        ValueError,
        match=_several_samples("pool_", "other")
        if sources
        else _unclear_ids("pool___cell0", "pool"),
    ):
        merged.perturbation.import_guide_calls(
            path, cell_selection=merged.snapshot_cell_selection("every")
        )

    # The prefix that the refusal suggests reaches the cells of pool_.
    calls = merged.perturbation.import_guide_calls(
        path,
        cell_selection=merged.snapshot_cell_selection("first"),
        barcode_prefix="pool___",
    )
    cells = merged.perturbation.load_guide_calls(calls).cells
    assert cells["call"].tolist() == ["CRISPR0", "CRISPR1", MULTIPLE, NEGATIVE]


def _wide_screen(folder: Path) -> tuple[Path, Path]:
    """A screen with long guide IDs where one cell carries every guide, and its call file.

    4,000 cells and 80 guides whose IDs are 60 characters long; every cell
    carries one guide with 30 UMIs beside ambient UMIs, and cell0 carries
    every guide, so every cell's padded calls hold 80 slots. The call file
    calls each guide with 30 UMIs or more and names one unknown barcode.
    """
    n_cells, n_guides = 4_000, 80
    rng = np.random.default_rng(13)
    guides = rng.poisson(0.2, size=(n_cells, n_guides)).astype(np.uint16)
    guides[np.arange(n_cells), rng.integers(0, n_guides, n_cells)] += 30
    guides[0] = 30
    ids = [
        f"GUIDE{guide:03d}_TARGETS_ENSG{guide:011d}_ACGTACGTACGTACGTACGTACGTAC"
        for guide in range(n_guides)
    ]
    rna = rng.poisson(1.0, size=(n_cells, 3)) + 1
    store_path = folder / "wide.zarr"
    write_count_store(
        str(store_path),
        {"RNA": rna, "CRISPR": guides},
        "uint16",
        feature_ids={"CRISPR": ids},
    )
    lines = [
        "cell_barcode,num_features,feature_call,num_umis",
        "unknown-1,1,{},30".format(ids[0]),
    ]
    for cell in range(n_cells):
        called = np.flatnonzero(guides[cell] >= 30)
        lines.append(
            f"cell{cell},{len(called)},"
            + "|".join(ids[guide] for guide in called)
            + ","
            + "|".join(str(int(guides[cell, guide])) for guide in called)
        )
    call_path = folder / "wide.csv"
    call_path.write_text("\n".join(lines) + "\n")
    return store_path, call_path


@pytest.mark.slow
def test_an_import_refuses_a_small_budget_before_it_writes(tmp_path) -> None:
    store_path, call_path = _wide_screen(tmp_path)
    # Feature tables in chunks of their rows, so that the import's own steps,
    # not a chunk of 100,000 guide IDs, set the budget.
    rechunk_feature_tables(store_path)
    store = open_store(store_path)
    store.cells.insert("every", by_cell_id(store.cells, np.ones(4_000, dtype=bool)))
    every = store.snapshot_cell_selection("every")

    # The refusal comes before anything is written and names the step that it
    # refused.
    small = open_store(store_path, mem_budget=10_000)
    saved = written(small)
    with pytest.raises(
        MemoryError,
        match="^Importing the guide calls of 4000 cells holds about .* Raise mem_budget",
    ):
        small.perturbation.import_guide_calls(call_path, cell_selection=every)
    assert written(open_store(store_path)) == saved


def test_import_refuses_assays_and_guides_it_cannot_use(
    shared_store, template, tmp_path
) -> None:
    store = shared_store
    selection = template.selection
    saved = written(store)
    with pytest.raises(TypeError, match="declared with type 'CRISPR'"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection, from_assay="RNA"
        )
    unknown = _call_file(
        tmp_path, CALL_FILE.replace("cell7,1,CRISPR1", "cell7,1,CRISPR9"), "bad.csv"
    )
    with pytest.raises(ValueError, match="not features of assay 'CRISPR'.*CRISPR9"):
        store.perturbation.import_guide_calls(unknown, cell_selection=selection)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection="screened"
        )
    with pytest.raises(TypeError, match="barcode_prefix"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection, barcode_prefix=1
        )
    assert written(store) == saved


def test_import_refuses_counts_that_are_not_unsigned_integers(tmp_path) -> None:
    store = _store(tmp_path / "store.zarr", dtype="float32")
    with pytest.raises(TypeError, match="non-negative integer counts"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=_selection(store)
        )


_BEYOND_UINT32 = "^Guide UMI counts must be at most 4294967295$"


def _store_beyond_uint32(path: Path) -> DataStore:
    """The screen, with cell0 holding a count beyond the uint32 range."""
    counts = GUIDE_COUNTS.astype(np.uint64)
    counts[0, 0] = 2**33 + 7
    rna = np.random.default_rng(7).poisson(2.0, size=(len(counts), 10))
    return write_screen(path, rna, counts, dtype="uint64")


def test_import_refuses_counts_beyond_uint32_as_guide_calling_does(tmp_path) -> None:
    # The top UMIs are saved as uint32, so a larger count would wrap.
    store = _store_beyond_uint32(tmp_path / "store.zarr")
    selection = _selection(store)
    saved = written(store)
    with pytest.raises(ValueError, match=_BEYOND_UINT32):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
    with pytest.raises(ValueError, match=_BEYOND_UINT32):
        store.perturbation.call_guides(selection, method="fixed", min_umi=5)
    assert written(store) == saved


def test_refusals_mid_stream_leave_zarr_concurrency_as_it_was(tmp_path) -> None:
    # Each refusal stops the stream of counts at its first block. The errors
    # are kept, as a notebook keeps its last error, and so are the frames that
    # hold the stopped streams.
    store = _store_beyond_uint32(tmp_path / "store.zarr")
    selection = _selection(store)
    before = zarr.config.get("async.concurrency")
    with pytest.raises(ValueError, match=_BEYOND_UINT32) as imported:
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
    with pytest.raises(ValueError, match=_BEYOND_UINT32) as called:
        store.perturbation.call_guides(selection, method="fixed", min_umi=5)
    try:
        assert zarr.config.get("async.concurrency") == before
    finally:
        # The collector frees the stopped streams, so a failure here leaves no
        # limit entered for later tests.
        del imported, called
        gc.collect()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"path": 7}, "path must be a path"),
        ({"path": ""}, "^path must be a path to Cell Ranger's call file$"),
        ({"from_assay": 7}, "from_assay must be an assay name"),
        ({"invalidate_cache": "yes"}, "invalidate_cache must be a boolean"),
    ],
)
def test_import_refuses_arguments_of_the_wrong_type(
    shared_store, template, tmp_path, arguments, message
) -> None:
    call = {
        "path": _call_file(tmp_path),
        "cell_selection": template.selection,
        **arguments,
    }
    saved = written(shared_store)
    with pytest.raises(TypeError, match=message):
        shared_store.perturbation.import_guide_calls(call.pop("path"), **call)
    assert written(shared_store) == saved


def test_import_refuses_an_empty_cell_selection(store, tmp_path) -> None:
    store.cells.insert(
        "nobody", by_cell_id(store.cells, np.zeros(len(GUIDE_COUNTS), dtype=bool))
    )
    nobody = store.snapshot_cell_selection("nobody")
    saved = written(store)
    with pytest.raises(ValueError, match="selects no cell"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=nobody
        )
    assert written(store) == saved


def test_import_refuses_calls_larger_than_the_memory_budget(
    shared_store, template, tmp_path, monkeypatch
) -> None:
    saved = written(shared_store)
    # Seven cells and seven calls need about 1.5 kB beside the streamed counts.
    monkeypatch.setattr(shared_store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        shared_store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=template.selection
        )
    assert written(shared_store) == saved


def test_import_checks_measured_cells_before_planning(
    shared_store, template, tmp_path, monkeypatch
) -> None:
    selection = template.selection
    seen: list[tuple] = []

    def refuse(self, assay, cells, *, operation, remedy=None):
        seen.append((assay, cells, operation))
        raise RuntimeError("unmeasured")

    saved = written(shared_store)
    monkeypatch.setattr(DataStore, "_require_measured_cells", refuse)
    with pytest.raises(RuntimeError, match="unmeasured"):
        shared_store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
    assert seen == [("CRISPR", selection, "import_guide_calls")]
    assert written(shared_store) == saved


def test_a_read_only_store_refuses_to_import_new_calls(
    store, template, tmp_path
) -> None:
    selection = template.selection
    saved = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=selection
    )
    reader = _reader(store)

    assert (
        reader.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
        == saved
    )
    edited = _call_file(tmp_path, CALL_FILE.replace(",9\n", ",8\n"), "edited.csv")
    with pytest.raises(PermissionError, match="import_guide_calls"):
        reader.perturbation.import_guide_calls(edited, cell_selection=selection)


def test_a_read_only_store_checks_the_file_and_the_budget_first(
    template, tmp_path, monkeypatch
) -> None:
    selection = template.selection
    reader = DataStore(
        str(template.path), default_assay="RNA", zarr_mode="r", nthreads=1
    )
    unknown = _call_file(
        tmp_path, CALL_FILE.replace("cell7,1,CRISPR1", "cell7,1,CRISPR9"), "bad.csv"
    )

    # The arguments, the call file and the budget are checked before the
    # store must be writable.
    with pytest.raises(ValueError, match="not features of assay 'CRISPR'"):
        reader.perturbation.import_guide_calls(unknown, cell_selection=selection)
    monkeypatch.setattr(reader, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        reader.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )


def test_an_import_of_one_sample_reads_only_its_cell_ids(
    store, tmp_path, monkeypatch
) -> None:
    # A merged store imports each sample's call file for the cells of that
    # sample, here the first four.
    store.cells.insert(
        "sample", by_cell_id(store.cells, np.arange(len(GUIDE_COUNTS)) < 4)
    )
    sample = store.snapshot_cell_selection("sample")
    reads = record_cell_reads(monkeypatch, store, {"ids"})

    with pytest.warns(UserWarning, match="1 of 7 barcodes of the call file match no"):
        ref = store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=sample
        )

    # The barcodes join the selected cells' IDs; those that join none are
    # then looked up among every cell's ID, a block at a time.
    first, *searched = reads["ids"]
    assert set(first.tolist()) == {0, 1, 2, 3}
    assert sorted(np.concatenate(searched).tolist()) == list(range(8))
    assert store.perturbation.load_guide_calls(ref).cells["call"].tolist() == [
        "CRISPR0",
        "CRISPR1",
        MULTIPLE,
        NEGATIVE,
    ]
    # A file whose every barcode joins a selected cell reads their IDs only.
    reads["ids"].clear()
    store.perturbation.import_guide_calls(
        _call_file(tmp_path, CALL_FILE.split("cell5")[0], "sample.csv"),
        cell_selection=sample,
    )
    assert set(np.concatenate(reads["ids"]).tolist()) == {0, 1, 2, 3}


def test_loading_calls_reads_the_cell_ids_once(
    store, template, tmp_path, monkeypatch
) -> None:
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=template.selection
    )
    reads = record_cell_reads(monkeypatch, store, {"ids"})

    cells = store.perturbation.load_guide_calls(ref).cells

    assert cells.index.tolist() == SELECTED
    assert [rows.tolist() for rows in reads["ids"]] == [[0, 1, 2, 3, 4, 5, 7]]


@pytest.mark.slow
def test_loading_calls_checks_the_budget_of_every_array_at_once(
    store, template, tmp_path, monkeypatch
) -> None:
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=template.selection
    )

    def fits(name: str, budget: int) -> bool:
        monkeypatch.setattr(store, "memoryBytes", budget)
        try:
            store.artifacts.load_values(ref, value=name)
        except MemoryError:
            return False
        return True

    def smallest(name: str) -> int:
        low, high = 0, 1 << 30
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (low, middle) if fits(name, middle) else (middle, high)
        return high

    # A budget that reads any one array with the cell IDs, but not all of them.
    budget = max(smallest(name) for name in GUIDE_CALL_ARRAYS)
    monkeypatch.setattr(store, "memoryBytes", budget)

    with pytest.raises(MemoryError, match="memory budget"):
        store.perturbation.load_guide_calls(ref)


def test_loading_calls_refuses_an_array_without_a_row_per_cell(
    store, template, tmp_path
) -> None:
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=template.selection
    )
    group = artifact_group(store.zw, ref)
    del group["top_umi"]
    group.create_array("top_umi", data=np.zeros(3, dtype=np.uint32))

    with pytest.raises(
        ValueError,
        match="^The guide_calls artifact's 'top_umi' array must hold one row per "
        "cell of its cell selection, 7 rows, not 3$",
    ):
        store.perturbation.load_guide_calls(ref)


def test_loading_refuses_other_artifact_kinds(shared_store, template) -> None:
    saved = written(shared_store)
    with pytest.raises(ValueError, match="guide_calls artifact"):
        shared_store.perturbation.load_guide_calls(template.selection)
    with pytest.raises(TypeError, match="ArtifactRef"):
        shared_store.perturbation.load_guide_calls("calls")
    assert written(shared_store) == saved


def test_guide_calls_plot_summarizes_the_calls(store, template, tmp_path) -> None:
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=template.selection
    )

    result = store.plots.guide_calls(ref, max_guides_per_cell=3, show=False)

    try:
        assert set(result.axes) == {
            "guides_per_cell",
            "cells_per_guide",
            "top_fraction",
        }
        per_cell = result.tables["guides_per_cell"]
        assert per_cell.to_dict("list") == {
            "guides": ["0", "1", "2", "3+"],
            "cells": [2, 3, 2, 0],
        }
        per_guide = result.tables["cells_per_guide"].set_index("guide")
        assert per_guide.loc["CRISPR1"].tolist() == [2, 1, 3]
        assert per_guide.loc["CRISPR3"].tolist() == [0, 1, 1]
        assert per_guide["cells"].tolist() == sorted(per_guide["cells"], reverse=True)
        top = result.tables["top_fraction"]
        groups = ["No guide called", "One guide called", "Several guides called"]
        assert top.columns.tolist() == ["bin_start", "bin_stop", *groups]
        np.testing.assert_allclose(top["bin_start"], np.arange(20) / 20)
        np.testing.assert_allclose(top["bin_stop"], np.arange(1, 21) / 20)
        # Each cell's top guide holds this share of its guide UMIs, binned by
        # twentieths: cell4 holds 2/3 in CRISPR2 without a call; cell1, cell0
        # and cell7 hold 25/27, 30/31 and all of theirs in their one call;
        # and cell2 and cell5 hold 1/2 and 40/43 in the top of their two
        # calls. cell3 has no guide UMI, so it has no share.
        binned = {
            group: {
                int(row): int(cells) for row, cells in enumerate(top[group]) if cells
            }
            for group in groups
        }
        assert binned == {
            "No guide called": {13: 1},
            "One guide called": {18: 1, 19: 2},
            "Several guides called": {10: 1, 18: 1},
        }
        assert result.provenance.extras["guide_calls"] == ref.to_dict()
        assert result.provenance.n_cells == 7
    finally:
        result.close()
    with pytest.raises(ValueError, match="max_guides_per_cell"):
        store.plots.guide_calls(ref, max_guides_per_cell=1, show=False)


def test_reference_checks_name_the_kinds_they_take() -> None:
    from cytearc.datastore._operations.perturbation_store import (
        _COORDINATE_KINDS,
        _require_ref,
    )
    from cytearc.graph.feature_projection import NATIVE_COORDINATE_KINDS

    ref = ArtifactRef(scope="datastore", kind="cell_selection", artifact_id="a" * 64)
    _require_ref(ref, "cells")
    _require_ref(ref, "cells", "cell_selection")
    _require_ref(None, "features", "feature_selection", optional=True)
    with pytest.raises(TypeError, match="^cells must be an ArtifactRef$"):
        _require_ref("cells", "cells")
    with pytest.raises(TypeError, match="^cells must be an ArtifactRef$"):
        _require_ref(None, "cells", "cell_selection")
    with pytest.raises(TypeError, match="^features must be an ArtifactRef or None$"):
        _require_ref("hvg", "features", "feature_selection", optional=True)
    for kinds, named in (
        (("guide_calls",), "a guide_calls artifact"),
        (("perturbation_labels", "mixscape_classes"), "a perturbation_labels or "),
        (
            _COORDINATE_KINDS,
            "a reduction, batch_correction, imported_coordinates or normalized "
            "artifact",
        ),
    ):
        with pytest.raises(ValueError, match=f"^labels must be {named}"):
            _require_ref(ref, "labels", *kinds)
    # The refusal names every coordinate kind that the graph reads.
    assert set(_COORDINATE_KINDS) == NATIVE_COORDINATE_KINDS


def test_the_other_screen_steps_refuse_their_budgets_with_memory_budget_error() -> None:
    from cytearc.datastore._operations.perturbation_store import _require_budget

    # The steps that C25 moves onto a ledger keep their words and carry the
    # bytes that pass them, the budget and the step.
    with pytest.raises(MemoryBudgetError) as refused:
        _require_budget(
            10,
            9,
            "The profiles of 3 targets in 2 features hold about",
            beside=" with the read of one feature",
            remedy="pass fewer features",
        )
    assert str(refused.value) == (
        "The profiles of 3 targets in 2 features hold about 10 bytes with the read "
        "of one feature, more than the memory budget of 9 bytes. Raise mem_budget "
        "to at least 1 MiB, or pass fewer features."
    )
    assert (
        refused.value.needed_bytes,
        refused.value.budget_bytes,
        refused.value.item,
    ) == (10, 9, "The profiles of 3 targets in 2 features")
    _require_budget(9, 9, "The profiles of 3 targets hold about")


def test_samples_of_any_column_are_numbered_as_before_within_the_ledger(
    tmp_path,
) -> None:
    from cytearc.datastore._operations.cell_samples import read_cell_samples
    from cytearc.metadata.rows import (
        apply_missing_mask,
        read_metadata_missing_rows_chunkwise,
        read_metadata_rows_chunkwise,
    )
    from cytearc.metadata.selection import sample_codes

    # 20,000 cells, labeled by text, by numbers, and by text with gaps, and a
    # selection of every other cell.
    n_cells = 20_000
    rng = np.random.default_rng(5)
    path = tmp_path / "store.zarr"
    store = write_screen(
        path,
        rng.poisson(2.0, size=(n_cells, 3)),
        rng.poisson(1.0, size=(n_cells, 4)),
        cell_columns={
            "lane": np.asarray(
                [f"L{index % 3}" for index in range(n_cells)], dtype=object
            ),
            "batch": np.arange(n_cells) % 4,
        },
    )
    insert_nullable_cell_column(
        store,
        "donor",
        np.asarray([f"D{index % 5}" for index in range(n_cells)]),
        np.arange(n_cells) % 7 == 1,
    )
    cells = np.arange(0, n_cells, 2, dtype=np.int64)
    for column in ("lane", "batch", "donor", None):

        def number(column=column):
            ledger = MemoryLedger(10**9, "Numbering samples")
            return ledger, read_cell_samples(
                store, ledger, column, cells, reason="every cell needs one"
            )

        if column == "donor":
            # Every seventh cell has no label, which every selected cell needs.
            missing = int(np.count_nonzero(cells % 7 == 1))
            with pytest.raises(
                ValueError, match=f"has no label for {missing} selected"
            ):
                number()
            continue
        # The samples that the labels read unbudgeted give, as before.
        columns = () if column is None else (column,)
        expected = sample_codes(
            columns,
            [
                apply_missing_mask(
                    np.asarray(read_metadata_rows_chunkwise(store.cells, name, cells)),
                    read_metadata_missing_rows_chunkwise(store.cells, name, cells),
                    labels=True,
                )
                for name in columns
            ],
            len(cells),
            reason="every cell needs one",
        )
        _ledger, samples = number()
        assert samples.codes.tolist() == expected[0].tolist()
        assert samples.names == [
            None if column is None else sample[column] for sample in expected[1]
        ]
        assert samples.assignments == expected[2]


def test_bands_without_a_budget_take_the_most_features_they_may(template) -> None:
    from cytearc.datastore._operations.perturbation_store import _count_bands

    store = open_store(template.path)
    bands = _count_bands(
        MemoryLedger(None, "Reading bands"),
        store._get_assay("CRISPR"),
        np.arange(4),
        (np.arange(len(GUIDE_COUNTS)),),
        per_feature=8,
        scratch_per_feature=8,
        most=3,
    )
    assert bands == [(0, 3), (3, 4)]


def test_a_feature_column_of_numbers_is_read_as_python_objects(template) -> None:
    store = open_store(template.path)
    assay = store._get_assay("CRISPR")
    ledger = MemoryLedger(None, "Reading a feature column")
    values = store._guide_feature_column(assay, "nCells", "id_column", ledger=ledger)
    expected = np.asarray(assay.feats.fetch_all("nCells"), dtype=object)
    assert values.dtype == object
    assert values.tolist() == expected.tolist()
    assert ledger.charged >= 8 * len(values)
    assert store._guide_feature_column(assay, None, "id_column", ledger=ledger) is None
