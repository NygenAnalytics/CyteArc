"""Guide calls imported from Cell Ranger's call file for an eight-cell screen.

The module writes the screen and its selection of seven cells once, as a
template. A test that writes copies the template, and the tests of refused
imports share one copy, which they check is left as it was.
"""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import MULTIPLE, NEGATIVE, GuideCalls
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.selections import read_stored_selection_indices
from cytearc.utils.logging import collect_warnings
from tests.perturbation_helpers import (
    copy_store,
    open_store,
    record_cell_reads,
    write_screen,
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
    store.cells.insert("screened", keep)
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
    group = store.artifacts.load(ref)
    assert group.attrs["n_file_barcodes"] == 7
    assert group.attrs["n_unmatched_barcodes"] == 2
    assert group.attrs["unmatched_barcode_examples"] == ["cell6", "outside-1"]
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
    )
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path, text),
        cell_selection=template.selection,
        barcode_prefix="cell",
    )

    cells = store.perturbation.load_guide_calls(ref).cells
    assert cells.loc[["cell0", "cell5"], "call"].tolist() == ["CRISPR0", "CRISPR2"]
    assert store.artifacts.load(ref).attrs["n_unmatched_barcodes"] == 0


def _logged_import(
    store: DataStore, path: Path, **arguments
) -> tuple[ArtifactRef, list[str]]:
    """Import guide calls, and return them with the warnings that CyteArc logs."""
    with collect_warnings() as messages:
        calls = store.perturbation.import_guide_calls(path, **arguments)
    return calls, list(messages)


def test_a_reused_import_logs_the_warning_of_the_first(
    store, template, tmp_path
) -> None:
    selection = template.selection
    path = _call_file(tmp_path)

    calls, messages = _logged_import(store, path, cell_selection=selection)

    # cell6 is outside the selection, and no cell is named outside-1.
    assert messages == [
        "2 of 7 barcodes of the call file match no selected cell, for example "
        "'cell6', 'outside-1'. Check barcode_prefix and cell_selection."
    ]
    # The saved calls log it again, alike: for the same file under another
    # name, and in a store opened read-only.
    moved = _call_file(tmp_path, name="moved.csv")
    assert _logged_import(store, moved, cell_selection=selection) == (calls, messages)
    reader = _reader(store)
    assert _logged_import(reader, path, cell_selection=selection) == (calls, messages)
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
    assert quiet[1] == []
    assert _logged_import(store, joined, **arguments) == quiet


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
    store.cells.insert("nobody", np.zeros(len(GUIDE_COUNTS), dtype=bool))
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
    store.cells.insert("sample", np.arange(len(GUIDE_COUNTS)) < 4)
    sample = store.snapshot_cell_selection("sample")
    reads = record_cell_reads(monkeypatch, store, {"ids"})

    with pytest.warns(UserWarning, match="4 of 7 barcodes of the call file"):
        ref = store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=sample
        )

    assert set(np.concatenate(reads["ids"]).tolist()) == {0, 1, 2, 3}
    assert store.perturbation.load_guide_calls(ref).cells["call"].tolist() == [
        "CRISPR0",
        "CRISPR1",
        MULTIPLE,
        NEGATIVE,
    ]


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


def test_guide_calls_plot_summarises_the_calls(store, template, tmp_path) -> None:
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
        assert (
            int(
                top[["No guide called", "One guide called", "Several guides called"]]
                .sum()
                .sum()
            )
            == 6
        )
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
