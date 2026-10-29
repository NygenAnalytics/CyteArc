from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import MULTIPLE, NEGATIVE, GuideCalls
from cytearc.storage.artifacts import ArtifactRef
from cytearc.storage.selections import read_stored_selection_indices
from cytearc.utils.logging import collect_warnings
from tests.storage_helpers import write_count_store

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


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _store(tmp_path: Path, dtype: str = "uint16") -> DataStore:
    rna = np.random.default_rng(7).poisson(2.0, size=(len(GUIDE_COUNTS), 10))
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": rna, "CRISPR": GUIDE_COUNTS}, dtype)
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


def _selection(store: DataStore):
    keep = np.ones(len(GUIDE_COUNTS), dtype=bool)
    keep[6] = False
    store.cells.insert("screened", keep)
    return store.snapshot_cell_selection("screened")


def _call_file(tmp_path: Path, text: str = CALL_FILE, name: str = "calls.csv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_import_saves_calls_and_count_summaries(tmp_path) -> None:
    store = _store(tmp_path)
    selection = _selection(store)

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


def test_select_cells_reads_the_saved_calls(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=_selection(store)
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


def test_import_is_reused_by_file_contents_not_by_path(tmp_path) -> None:
    store = _store(tmp_path)
    selection = _selection(store)
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


def test_a_barcode_prefix_joins_barcodes_to_cell_ids(tmp_path) -> None:
    store = _store(tmp_path)
    text = (
        "cell_barcode,num_features,feature_call,num_umis\n"
        "0,1,CRISPR0,30\n"
        "5,1,CRISPR2,40\n"
    )
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path, text),
        cell_selection=_selection(store),
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


def test_a_reused_import_logs_the_warning_of_the_first(tmp_path) -> None:
    store = _store(tmp_path)
    selection = _selection(store)
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
    reader = DataStore(
        str(tmp_path / "store.zarr"), default_assay="RNA", zarr_mode="r", nthreads=1
    )
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


def test_import_refuses_assays_and_guides_it_cannot_use(tmp_path) -> None:
    store = _store(tmp_path)
    selection = _selection(store)
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
    assert store.artifacts.list(kind="guide_calls", from_assay="CRISPR") == []


def test_import_refuses_counts_that_are_not_unsigned_integers(tmp_path) -> None:
    store = _store(tmp_path, dtype="float32")
    with pytest.raises(TypeError, match="non-negative integer counts"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=_selection(store)
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"path": 7}, "path must be a path"),
        ({"from_assay": 7}, "from_assay must be an assay name"),
        ({"invalidate_cache": "yes"}, "invalidate_cache must be a boolean"),
    ],
)
def test_import_refuses_arguments_of_the_wrong_type(
    tmp_path, arguments, message
) -> None:
    store = _store(tmp_path)
    call = {
        "path": _call_file(tmp_path),
        "cell_selection": _selection(store),
        **arguments,
    }
    with pytest.raises(TypeError, match=message):
        store.perturbation.import_guide_calls(call.pop("path"), **call)


def test_import_refuses_an_empty_cell_selection(tmp_path) -> None:
    store = _store(tmp_path)
    store.cells.insert("nobody", np.zeros(len(GUIDE_COUNTS), dtype=bool))
    with pytest.raises(ValueError, match="selects no cell"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path),
            cell_selection=store.snapshot_cell_selection("nobody"),
        )


def test_import_refuses_calls_larger_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    selection = _selection(store)
    # Seven cells and seven calls need about 1.5 kB beside the streamed counts.
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
    assert store.artifacts.list(kind="guide_calls", from_assay="CRISPR") == []


def test_import_checks_measured_cells_before_planning(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    selection = _selection(store)
    seen: list[tuple] = []

    def refuse(self, assay, cells, *, operation, remedy=None):
        seen.append((assay, cells, operation))
        raise RuntimeError("unmeasured")

    monkeypatch.setattr(DataStore, "_require_measured_cells", refuse)
    with pytest.raises(RuntimeError, match="unmeasured"):
        store.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
    assert seen == [("CRISPR", selection, "import_guide_calls")]
    assert store.artifacts.list(kind="guide_calls", from_assay="CRISPR") == []


def test_a_read_only_store_refuses_to_import_new_calls(tmp_path) -> None:
    store = _store(tmp_path)
    selection = _selection(store)
    saved = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=selection
    )
    reader = DataStore(
        str(tmp_path / "store.zarr"), default_assay="RNA", zarr_mode="r", nthreads=1
    )

    assert (
        reader.perturbation.import_guide_calls(
            _call_file(tmp_path), cell_selection=selection
        )
        == saved
    )
    edited = _call_file(tmp_path, CALL_FILE.replace(",9\n", ",8\n"), "edited.csv")
    with pytest.raises(PermissionError, match="import_guide_calls"):
        reader.perturbation.import_guide_calls(edited, cell_selection=selection)


def test_loading_refuses_other_artifact_kinds(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="guide_calls artifact"):
        store.perturbation.load_guide_calls(_selection(store))
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.perturbation.load_guide_calls("calls")


def test_guide_calls_plot_summarises_the_calls(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.perturbation.import_guide_calls(
        _call_file(tmp_path), cell_selection=_selection(store)
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
