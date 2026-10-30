"""Knockdown efficiency of the targets of a fourteen-cell screen.

The module writes the screen, its guide calls and its labels once, as a
template. A test that writes copies the template, and the tests of refused
calls share one copy, which they check is left as it was.
"""

import warnings
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import false_discovery_control, mannwhitneyu

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import PerturbationEfficiency
from cytearc.perturbation.efficiency import efficiency_arrays, target_features
from cytearc.storage.artifacts import ArtifactRef, inspect_artifact
from tests.perturbation_helpers import (
    copy_store,
    fixed_labels,
    open_store,
    write_screen,
    written,
)

# Guides, their targets and the feature IDs of the targets. RNA4 has no ID and
# matches the RNA feature named RNA4; GENE_C's ID matches no feature; GENE_D
# is never called.
TARGETS = ["GENE_A", "GENE_A", "GENE_B", "Non-Targeting", "RNA4", "GENE_C", "GENE_D"]
IDS = ["RNA1", "RNA1", "RNA2", "Non-Targeting", "", "ENSG_missing", "RNA5"]
# The one called guide of each cell, or None, or two guides for cell 12.
CARRIED = [0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, None, (0, 2), 5]


def _counts() -> tuple[np.ndarray, np.ndarray]:
    guides = np.zeros((len(CARRIED), len(TARGETS)), dtype=np.uint16)
    for cell, carried in enumerate(CARRIED):
        for guide in (carried,) if isinstance(carried, int) else carried or ():
            guides[cell, guide] = 20
    rna = np.random.default_rng(5).poisson(6.0, size=(len(CARRIED), 6))
    rna[:4, 1] = [0, 1, 0, 1]  # GENE_A knocked down in its cells
    rna[6:10, 2] = 0  # GENE_B never expressed in the control cells
    return guides, rna.astype(np.uint16)


GUIDES, RNA = _counts()


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Iterator[SimpleNamespace]:
    """The screen with its labels, and labels whose control no cell carries."""
    path = tmp_path_factory.mktemp("efficiency") / "store.zarr"
    store = write_screen(path, RNA, GUIDES, targets=TARGETS, target_ids=IDS)
    screen = SimpleNamespace(
        path=path,
        labels=fixed_labels(store, ["Non-Targeting"]),
        # GENE_D's guide is in no cell.
        without_controls=fixed_labels(store, ["GENE_D"]),
    )
    saved = written(store)
    yield screen
    assert written(open_store(path)) == saved


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    """A copy of the template, for a test that writes."""
    return copy_store(template.path, tmp_path)


@pytest.fixture(scope="module")
def shared_store(template, tmp_path_factory) -> DataStore:
    """One writable copy of the template for the calls that must be refused."""
    return copy_store(template.path, tmp_path_factory.mktemp("efficiency_refused"))


def _expected(rows: list[list[int]], genes: list[int]) -> dict[str, np.ndarray]:
    """The efficiency of each row's cells, computed directly."""
    controls = [6, 7, 8, 9]
    compared = sorted({cell for row in rows for cell in row} | set(controls))
    library = RNA.sum(axis=1).astype(float)
    scale = np.median(library[compared])
    expression = RNA * (scale / library)[:, None]
    p_values = []
    out: dict[str, list[float]] = {"mean_umi": [], "remaining_fraction": []}
    for row, gene in zip(rows, genes, strict=True):
        out["mean_umi"].append(RNA[row, gene].mean())
        out["remaining_fraction"].append(
            expression[row, gene].mean() / expression[controls, gene].mean()
        )
        p_values.append(
            mannwhitneyu(
                expression[row, gene],
                expression[controls, gene],
                alternative="two-sided",
                method="asymptotic",
            ).pvalue
        )
    return {
        "mean_umi": np.asarray(out["mean_umi"]),
        "remaining_fraction": np.asarray(out["remaining_fraction"]),
        "p_value": np.asarray(p_values),
        "control_mean_umi": RNA[controls][:, genes].mean(axis=0),
    }


def test_status_constants_name_their_method_and_keep_their_codes() -> None:
    from cytearc.perturbation import edistance, efficiency, mixscape, mixtures

    # The codes that saved artifacts hold, by the method that stored them.
    codes = {
        efficiency: {
            "EFFICIENCY_TESTED": 0,
            "EFFICIENCY_NO_FEATURE": 1,
            "EFFICIENCY_NO_CELLS": 2,
            "EFFICIENCY_NO_CONTROL_EXPRESSION": 3,
        },
        edistance: {"EDISTANCE_TESTED": 0, "EDISTANCE_TOO_FEW_CELLS": 1},
        mixscape: {
            "MIXSCAPE_FITTED": 0,
            "MIXSCAPE_FEW_DE_GENES": 1,
            "MIXSCAPE_FEW_CELLS": 2,
            "MIXSCAPE_NO_FIT": 3,
        },
        mixtures: {
            "GUIDE_FIT_FITTED": 0,
            "GUIDE_FIT_NO_CELLS": 1,
            "GUIDE_FIT_TOO_SPARSE": 2,
            "GUIDE_FIT_NO_FIT": 3,
            "GUIDE_FIT_NO_POSITIVE": 4,
        },
    }
    statuses = {
        efficiency: efficiency.EFFICIENCY_STATUSES,
        edistance: edistance.EDISTANCE_STATUSES,
        mixscape: mixscape.MIXSCAPE_STATUSES,
        mixtures: mixtures.FIT_STATUSES,
    }
    for module, constants in codes.items():
        assert {name: getattr(module, name) for name in constants} == constants
        # Each constant is the position of the status that its name ends with.
        for name, code in constants.items():
            assert name.lower().endswith(statuses[module][code])
    # No module exports a name of another, so a status imported from the
    # wrong module fails to import.
    exported = [name for module in codes for name in module.__all__]
    assert len(exported) == len(set(exported))


def test_efficiency_compares_each_target_with_the_controls(store, template) -> None:
    labels = template.labels

    with pytest.warns(UserWarning, match="1 of 5 rows have no feature of assay 'RNA'"):
        ref = store.perturbation.efficiency(labels)

    assert ref.kind == "perturbation_efficiency"
    assert ref.assay == "RNA"
    loaded = store.perturbation.load_efficiency(ref)
    assert isinstance(loaded, PerturbationEfficiency)
    assert loaded.by == "target"
    assert loaded.from_assay == "RNA"
    table = loaded.table.set_index("target")
    assert table.index.tolist() == ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "RNA4"]
    assert table["feature"].tolist() == ["RNA1", "RNA2", None, "RNA5", "RNA4"]
    assert table["status"].tolist() == [
        "tested",
        "no_control_expression",
        "no_feature",
        "no_cells",
        "tested",
    ]
    assert table["n_cells"].tolist() == [4, 2, 1, 0, 1]
    assert table["n_control_cells"].tolist() == [4] * 5
    # Compared cells: GENE_A 0-3, GENE_B 4-5, GENE_C 13, RNA4 10, controls 6-9.
    expected = _expected([[0, 1, 2, 3], [10]], [1, 4])
    tested = table.loc[["GENE_A", "RNA4"]]
    np.testing.assert_allclose(tested["mean_umi"], expected["mean_umi"])
    np.testing.assert_allclose(tested["control_mean_umi"], expected["control_mean_umi"])
    np.testing.assert_allclose(
        tested["remaining_fraction"], expected["remaining_fraction"]
    )
    np.testing.assert_allclose(
        tested["log2_fold_change"], np.log2(expected["remaining_fraction"])
    )
    np.testing.assert_allclose(tested["p_value"][:1], expected["p_value"][:1])
    assert table.loc["GENE_A", "remaining_fraction"] < 0.2
    assert np.isnan(table.loc["GENE_B", "remaining_fraction"])
    assert np.isfinite(table.loc["GENE_B", "p_value"])
    with_p = table["p_value"].notna()
    np.testing.assert_allclose(
        table.loc[with_p, "adjusted_p_value"],
        false_discovery_control(table.loc[with_p, "p_value"], method="bh"),
    )
    assert table.loc[~with_p, "adjusted_p_value"].isna().all()


def test_efficiency_by_guide_takes_each_cell_with_one_guide(store, template) -> None:
    labels = template.labels
    with pytest.warns(UserWarning, match="no feature"):
        ref = store.perturbation.efficiency(labels, by="guide")
    table = store.perturbation.load_efficiency(ref).table.set_index("guide")
    assert table.index.tolist() == [f"CRISPR{index}" for index in (0, 1, 2, 4, 5, 6)]
    assert table["target"].tolist() == [
        "GENE_A",
        "GENE_A",
        "GENE_B",
        "RNA4",
        "GENE_C",
        "GENE_D",
    ]
    assert table["n_cells"].tolist() == [3, 1, 2, 1, 1, 0]
    expected = _expected([[0, 1, 2], [3]], [1, 1])
    np.testing.assert_allclose(
        table.loc[["CRISPR0", "CRISPR1"], "mean_umi"], expected["mean_umi"]
    )


def test_targets_match_by_name_without_an_id_column(store, template) -> None:
    labels = template.labels
    with pytest.warns(UserWarning, match="4 of 5 rows have no feature"):
        ref = store.perturbation.efficiency(labels, id_column=None)
    table = store.perturbation.load_efficiency(ref).table.set_index("target")
    assert table["feature"].tolist() == [None, None, None, None, "RNA4"]
    # With CRISPR4 renamed, no target matches a feature, so no gene is read.
    calls = inspect_artifact(store.zw, labels).input_ref("guide_calls")
    renamed = store.perturbation.labels(
        calls, controls=["Non-Targeting"], targets={"CRISPR4": "GENE_E"}
    )
    with pytest.warns(UserWarning, match="5 of 5 rows have no feature"):
        none = store.perturbation.efficiency(renamed, id_column=None)
    statuses = store.perturbation.load_efficiency(none).table["status"]
    assert statuses.eq("no_feature").all()


def _warned_efficiency(
    store: DataStore, *args, **kwargs
) -> tuple[ArtifactRef, list[str]]:
    """Measure efficiency, and return the table with the messages it warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = store.perturbation.efficiency(*args, **kwargs)
    return table, [
        str(entry.message) for entry in caught if entry.category is UserWarning
    ]


def test_a_reused_table_repeats_the_warning_of_the_measured_one(
    store, template
) -> None:
    labels = template.labels

    measured, messages = _warned_efficiency(store, labels)

    # GENE_C's guide gives a feature ID that the RNA assay lacks.
    assert messages == [
        "1 of 5 rows have no feature of assay 'RNA' for their target, for example "
        "'GENE_C'. Check id_column, or the target names against the feature names."
    ]
    assert _warned_efficiency(store, labels) == (measured, messages)
    # By name, only RNA4 matches a feature, so the warning names four rows.
    by_name, messages = _warned_efficiency(store, labels, id_column=None)
    assert messages == [
        "4 of 5 rows have no feature of assay 'RNA' for their target, for example "
        "'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'. Check id_column, or the target "
        "names against the feature names."
    ]
    assert _warned_efficiency(store, labels, id_column=None) == (by_name, messages)
    # With GENE_C a control, every row matches a feature, and neither the
    # measured nor the reused table warns.
    matched = fixed_labels(store, ["Non-Targeting", "GENE_C"])
    quiet = _warned_efficiency(store, matched)
    assert quiet[1] == []
    assert _warned_efficiency(store, matched) == quiet


def test_efficiency_is_reused_and_follows_its_inputs(store, template) -> None:
    labels = template.labels
    ref, messages = _warned_efficiency(store, labels)
    assert _warned_efficiency(store, labels) == (ref, messages)
    status = inspect_artifact(store.zw, ref)
    assert status.parameters == {"by": "target"}
    assert set(status.inputs) == {
        "labels",
        "dataset_fingerprint",
        "target_features_fingerprint",
    }
    with pytest.warns(UserWarning):
        by_name = store.perturbation.efficiency(labels, id_column=None)
    assert by_name != ref
    reader = open_store(store.zarr_loc, zarr_mode="r")
    # A store opened read-only reuses the table, and its warning.
    assert _warned_efficiency(reader, labels) == (ref, messages)
    with pytest.raises(PermissionError, match="measure_perturbation_efficiency"):
        reader.perturbation.efficiency(labels, by="guide")


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"from_assay": "ATAC"}, ValueError, "from_assay must name an assay"),
        ({"by": "gene"}, ValueError, "by must be 'target' or 'guide'"),
        ({"id_column": 4}, TypeError, "id_column must be a feature column"),
        ({"id_column": "gene_id"}, ValueError, "'gene_id' is not a feature column"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
    ],
)
def test_efficiency_refuses_arguments_it_cannot_use(
    shared_store, template, arguments, error, message
) -> None:
    saved = written(shared_store)
    with pytest.raises(error, match=message):
        shared_store.perturbation.efficiency(template.labels, **arguments)
    assert written(shared_store) == saved


def test_efficiency_refuses_labels_it_cannot_measure(
    shared_store, template, monkeypatch
) -> None:
    store = shared_store
    saved = written(store)
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        store.perturbation.efficiency("labels")
    cells = store.select_measured_cells("CRISPR")
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.efficiency(cells)
    # Labels whose controls are GENE_D, which no cell carries, have no control cell.
    with pytest.raises(
        ValueError, match="No labelled cell carries control guides only"
    ):
        store.perturbation.efficiency(template.without_controls)
    monkeypatch.setattr(store, "memoryBytes", 100)
    # RNA1, RNA2, RNA4 and RNA5 over the 8 perturbed and 4 control cells.
    with pytest.raises(MemoryError, match="Comparing 4 target genes over 12 cells"):
        store.perturbation.efficiency(template.labels)
    assert written(store) == saved


def test_loading_efficiency_refuses_other_artifacts(shared_store, template) -> None:
    saved = written(shared_store)
    with pytest.raises(TypeError, match="efficiency must be an ArtifactRef"):
        shared_store.perturbation.load_efficiency("efficiency")
    with pytest.raises(ValueError, match="not 'perturbation_labels'"):
        shared_store.perturbation.load_efficiency(template.labels)
    assert written(shared_store) == saved


def test_efficiency_arrays_check_their_inputs() -> None:
    counts = np.ones((3, 1))
    with pytest.raises(ValueError, match="cells x genes"):
        efficiency_arrays(np.ones(3), np.ones(3), np.zeros(3), np.ones(3, bool), [0])
    with pytest.raises(ValueError, match="one entry per cell"):
        efficiency_arrays(counts, np.ones(3), np.zeros(2), np.ones(3, bool), [0])
    with pytest.raises(ValueError, match="no control cell"):
        efficiency_arrays(counts, np.ones(3), np.zeros(3), np.zeros(3, bool), [0])
    # Equal values everywhere give a p-value of 1, and a cell without UMIs
    # keeps an expression of 0.
    arrays = efficiency_arrays(
        counts,
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([0, -1, -1]),
        np.asarray([False, False, True]),
        [0],
    )
    assert arrays.p_value.tolist() == [1.0]
    assert arrays.remaining_fraction.tolist() == [1.0]
    assert arrays.log2_fold_change.tolist() == [0.0]


def test_target_features_prefer_one_id_then_one_name() -> None:
    names = np.asarray(["ENSG1", "ENSG2", "ENSG3", "ENSG4"])
    symbols = np.asarray(["A", "B", "B", "D"])
    targets = np.asarray(["A", "B", "C", "D", "E"])
    # Guide i targets target guide_codes[i] and gives it guide_ids[i].
    guide_codes = np.asarray([0, 1, 1, 2, 3, 4])
    guide_ids = np.asarray(
        ["ENSG1", "ENSG2", "ENSG3", None, float("nan"), " "], dtype=object
    )
    matched = target_features(targets, guide_codes, guide_ids, names, symbols)
    # A matches by ID; B's guides give two IDs and its name is shared; C and E
    # have no ID and no feature of their name; D matches by its unique name.
    assert matched.tolist() == [0, -1, -1, 3, -1]
    assert target_features(targets, guide_codes, None, names, symbols).tolist() == [
        0,
        -1,
        -1,
        3,
        -1,
    ]
    # IDs that repeat among the features match nothing; names still do.
    repeated = np.asarray(["ENSG1", "ENSG1", "ENSG3", "ENSG4"])
    assert target_features(
        targets, guide_codes, guide_ids, repeated, symbols
    ).tolist() == [0, -1, -1, 3, -1]
