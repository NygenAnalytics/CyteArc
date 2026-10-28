from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import CONTROL, MULTIPLE, NEGATIVE, PerturbationLabels
from cytearc.perturbation.labels import (
    guide_targets,
    perturbation_label_arrays,
    perturbation_label_frames,
)
from cytearc.storage.artifacts import inspect_artifact
from cytearc.storage.selections import read_stored_selection_indices
from tests.storage_helpers import write_count_store

# Guides CRISPR0 and CRISPR1 target GENE_A, CRISPR2 targets GENE_B, and
# CRISPR3 and CRISPR4 are controls. Every count of 20 is called at min_umi=10.
COUNTS = np.asarray(
    [
        [20, 0, 0, 0, 0],  # cell0: GENE_A
        [20, 20, 0, 0, 0],  # cell1: two guides of GENE_A
        [20, 0, 0, 20, 0],  # cell2: GENE_A and a control guide
        [0, 0, 0, 20, 0],  # cell3: a control guide only
        [0, 0, 0, 0, 3],  # cell4: no call
        [20, 0, 20, 0, 0],  # cell5: GENE_A and GENE_B
        [0, 0, 20, 20, 0],  # cell6: GENE_B and a control guide
        [0, 0, 0, 20, 20],  # cell7: two control targets
        [0, 20, 20, 0, 0],  # cell8: GENE_B and GENE_A
    ],
    dtype=np.uint16,
)
TARGETS = np.asarray(["GENE_A", "GENE_A", "GENE_B", "Non-Targeting", "Scrambled"])
CONTROLS = ["Non-Targeting", "Scrambled"]
SINGLE = [
    "GENE_A",
    "GENE_A",
    "GENE_A",
    CONTROL,
    NEGATIVE,
    MULTIPLE,
    "GENE_B",
    CONTROL,
    MULTIPLE,
]


def _store(tmp_path: Path, targets=TARGETS) -> DataStore:
    rna = np.random.default_rng(7).poisson(2.0, size=(len(COUNTS), 10))
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": rna, "CRISPR": COUNTS}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    if targets is not None:
        store._get_assay("CRISPR").feats.insert(
            "target_gene_name", np.asarray(targets, dtype=object)
        )
    return store


def _calls(store: DataStore):
    cells = store.select_measured_cells("CRISPR")
    return store.perturbation.call_guides(cells, method="fixed", min_umi=10)


def test_labels_name_the_genes_of_each_cells_guides(tmp_path) -> None:
    store = _store(tmp_path)
    calls = _calls(store)

    single = store.perturbation.labels(calls, controls=CONTROLS)
    combined = store.perturbation.labels(
        calls, controls=CONTROLS, policy="combinations"
    )

    assert single.kind == "perturbation_labels"
    loaded = store.perturbation.load_labels(single)
    assert isinstance(loaded, PerturbationLabels)
    assert loaded.cells["label"].tolist() == SINGLE
    assert loaded.cells["n_targets"].tolist() == [1, 1, 1, 0, 0, 2, 1, 0, 2]
    assert loaded.cells["is_control"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert loaded.cells["guide_label"].tolist()[:5] == [
        "CRISPR0",
        MULTIPLE,
        MULTIPLE,
        "CRISPR3",
        NEGATIVE,
    ]
    assert loaded.controls == ("Non-Targeting", "Scrambled")
    assert loaded.policy == "single"
    assert loaded.targets.groupby("ids")["target"].apply(list).to_dict() == {
        "cell0": ["GENE_A"],
        "cell1": ["GENE_A"],
        "cell2": ["GENE_A"],
        "cell5": ["GENE_A", "GENE_B"],
        "cell6": ["GENE_B"],
        "cell8": ["GENE_A", "GENE_B"],
    }
    assert loaded.guides.to_dict("list") == {
        "guide": [f"CRISPR{index}" for index in range(5)],
        "target": TARGETS.tolist(),
        "is_control": [False, False, False, True, True],
    }
    labels = store.perturbation.load_labels(combined).cells["label"].tolist()
    assert labels[5] == labels[8] == "GENE_A+GENE_B"
    assert [label for index, label in enumerate(labels) if index not in (5, 8)] == [
        label for index, label in enumerate(SINGLE) if index not in (5, 8)
    ]
    # The labels group cells as cluster labels do.
    chosen = store.select_cells(single, include=["GENE_A"])
    indices = read_stored_selection_indices(
        store.zw,
        chosen,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    assert indices.tolist() == [0, 1, 2]
    codes = store.artifacts.load_values(single, value="target_codes").values
    assert np.asarray(codes).tolist()[5] == [0, 1]


def test_imported_calls_get_labels_too(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.select_measured_cells("CRISPR")
    path = tmp_path / "calls.csv"
    path.write_text(
        "cell_barcode,num_features,feature_call,num_umis\n"
        "cell0,1,CRISPR0,20\n"
        "cell5,2,CRISPR0|CRISPR2,20|20\n"
    )
    imported = store.perturbation.import_guide_calls(path, cell_selection=cells)
    ref = store.perturbation.labels(imported, controls=["Non-Targeting"])
    labels = store.perturbation.load_labels(ref).cells["label"].tolist()
    assert labels[:6] == ["GENE_A", NEGATIVE, NEGATIVE, NEGATIVE, NEGATIVE, MULTIPLE]


def test_targets_replace_the_column_for_the_guides_they_name(tmp_path) -> None:
    store = _store(tmp_path)
    calls = _calls(store)
    ref = store.perturbation.labels(
        calls, controls=["Non-Targeting"], targets={"CRISPR4": "GENE_C"}
    )
    labels = store.perturbation.load_labels(ref).cells["label"].tolist()
    assert labels[7] == "GENE_C"
    only = store.perturbation.labels(
        calls,
        controls=["Non-Targeting"],
        target_column=None,
        targets=dict(
            zip(
                [f"CRISPR{index}" for index in range(5)],
                ["A", "A", "B", "Non-Targeting", "C"],
                strict=True,
            )
        ),
    )
    labels = store.perturbation.load_labels(only).cells["label"].tolist()
    assert labels[:4] == ["A", "A", "A", CONTROL]
    assert labels[7] == "C"


def test_labels_are_reused_and_follow_their_inputs(tmp_path) -> None:
    store = _store(tmp_path)
    calls = _calls(store)
    ref = store.perturbation.labels(calls, controls=CONTROLS)
    # The order and repeats of the controls do not change the identity.
    same = store.perturbation.labels(
        calls, controls=["Scrambled", "Non-Targeting", "Scrambled"]
    )
    assert same == ref
    status = inspect_artifact(store.zw, ref)
    assert status.parameters == {
        "controls": ["Non-Targeting", "Scrambled"],
        "policy": "single",
    }
    assert set(status.inputs) == {
        "cell_selection",
        "guide_calls",
        "targets_fingerprint",
    }
    other = store.perturbation.labels(
        calls, controls=CONTROLS, targets={"CRISPR1": "GENE_C"}
    )
    assert other != ref
    fresh = store.perturbation.labels(calls, controls=CONTROLS, invalidate_cache=True)
    labels = store.perturbation.load_labels(fresh).cells["label"].tolist()
    assert labels == SINGLE


def test_read_only_store_reuses_labels_but_cannot_label(tmp_path) -> None:
    store = _store(tmp_path)
    calls = _calls(store)
    ref = store.perturbation.labels(calls, controls=CONTROLS)
    reader = DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )
    assert reader.perturbation.labels(calls, controls=CONTROLS) == ref
    with pytest.raises(PermissionError, match="label_perturbations"):
        reader.perturbation.labels(calls, controls=CONTROLS, policy="combinations")


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"controls": "Non-Targeting"}, TypeError, "a sequence of target names"),
        ({"controls": [3]}, TypeError, "a sequence of target names"),
        ({"controls": []}, ValueError, "name at least one target"),
        ({"controls": ["Non-Targeting", " "]}, ValueError, "no name may be blank"),
        ({"target_column": 3}, TypeError, "target_column must be a feature column"),
        ({"target_column": "gene"}, ValueError, "'gene' is not a feature column"),
        ({"targets": {"CRISPR0": 1}}, TypeError, "targets must map guide IDs"),
        ({"target_column": None}, ValueError, "Pass target_column, targets, or both"),
        (
            {"policy": "high_moi"},
            ValueError,
            "policy must be 'single' or 'combinations'",
        ),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache must be a boolean"),
        ({"targets": {"CRISPR9": "GENE_C"}}, ValueError, "not guides of the assay"),
        ({"targets": {"CRISPR0": " "}}, ValueError, "an empty target"),
        ({"targets": {"CRISPR0": "control"}}, ValueError, "reserved"),
        (
            {"targets": {"CRISPR0": "A+B"}, "policy": "combinations"},
            ValueError,
            "must not hold '\\+'",
        ),
        ({"controls": ["Non-targeting"]}, ValueError, "'Non-targeting'"),
    ],
)
def test_labels_refuse_arguments_they_cannot_use(
    tmp_path, arguments, error, message
) -> None:
    store = _store(tmp_path)
    calls = _calls(store)
    with pytest.raises(error, match=message):
        store.perturbation.labels(calls, **({"controls": CONTROLS} | arguments))


def test_labels_refuse_calls_they_cannot_label(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, targets=["GENE_A", "", "GENE_B", "Non-Targeting", None])
    calls = _calls(store)
    with pytest.raises(TypeError, match="calls must be an ArtifactRef"):
        store.perturbation.labels("calls", controls=["Non-Targeting"])
    cells = store.select_measured_cells("CRISPR")
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.labels(cells, controls=["Non-Targeting"])
    # CRISPR1 and CRISPR4 are called but have no target in the column; a
    # read-only store says so too, rather than refusing to write.
    reader = DataStore(
        str(tmp_path / "store.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )
    for source in (store, reader):
        with pytest.raises(
            ValueError,
            match="2 called guides have no target, for example 'CRISPR1', 'CRISPR4'",
        ):
            source.perturbation.labels(calls, controls=["Non-Targeting"])
    targets = {"CRISPR1": "GENE_A", "CRISPR4": "Non-Targeting"}
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Labelling 9 cells"):
        store.perturbation.labels(calls, controls=["Non-Targeting"], targets=targets)
    assert not store.artifacts.list(kind="perturbation_labels")


def test_loading_labels_refuses_other_artifacts(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        store.perturbation.load_labels("labels")
    with pytest.raises(ValueError, match="not 'guide_calls'"):
        store.perturbation.load_labels(_calls(store))


def test_label_arrays_without_targets_keep_one_column() -> None:
    targets = guide_targets(
        np.asarray(["g0", "g1"]),
        np.asarray([np.nan, "Non-Targeting"], dtype=object),
        None,
        ("Non-Targeting",),
        policy="single",
    )
    assert targets.guide_codes.tolist() == [-1, 0]
    assert targets.guide_names().tolist() == ["", "Non-Targeting"]
    arrays = perturbation_label_arrays(
        np.asarray([[1], [-1]]), targets, policy="single"
    )
    assert arrays.values.tolist() == [CONTROL, NEGATIVE]
    assert arrays.target_codes.tolist() == [[-1], [-1]]
    # g0 has no target, so a cell that carries it cannot be labelled.
    with pytest.raises(ValueError, match="1 called guides have no target"):
        perturbation_label_arrays(np.asarray([[0], [1]]), targets, policy="single")
    cells, per_target, guides = perturbation_label_frames(
        np.asarray(["a", "b"]),
        values=arrays.values,
        guide_label=np.asarray(["g1", NEGATIVE]),
        is_control=arrays.is_control,
        n_targets=arrays.n_targets,
        target_codes=arrays.target_codes,
        target_names=targets.names,
        target_is_control=targets.is_control,
        guide_ids=np.asarray(["g0", "g1"]),
        guide_target_codes=targets.guide_codes,
    )
    assert cells["label"].tolist() == [CONTROL, NEGATIVE]
    assert per_target.empty
    assert guides["target"].tolist() == [None, "Non-Targeting"]
    assert guides["is_control"].tolist() == [False, True]
    assert isinstance(cells, pd.DataFrame)
