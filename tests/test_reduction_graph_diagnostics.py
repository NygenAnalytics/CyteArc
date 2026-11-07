"""The diagnostics that PCA, Harmony and UMAP save, and the warnings they give from them.

Each operation saves the numbers behind its warning with its artifact and
warns from them after it computes or reuses the artifact, so a second
identical call, or a pipeline run that reuses the artifact, warns again. A
neighbor query's self-hit rate is tested with its search settings in
``test_ann_query_settings.py``.
"""

import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cytearc.embeddings.reduction as reduction_module
from cytearc import DataStore, DiagnosticWarning
from cytearc.storage.artifacts import ArtifactRef, artifact_path
from cytearc.utils.logging import collect_warnings
from tests.storage_helpers import open_store, write_count_store

N_CELLS = 120
N_GENES = 40
# Harmony options whose fit stops after two rounds: with epsilon_harmony=0 no
# round converges unless the objective repeats exactly.
_UNCONVERGED = {"n_clusters": 3, "max_iter_harmony": 2, "epsilon_harmony": 0.0}


def _counts() -> np.ndarray:
    rng = np.random.default_rng(31)
    counts = rng.poisson(rng.gamma(1.0, 3.0, size=N_GENES), size=(N_CELLS, N_GENES))
    counts[:, 0] += 3
    return counts


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    """120 cells in two batches, with normalized values, an exact PCA and its graph."""
    zarr_loc = tmp_path_factory.mktemp("reduction_diagnostics") / "store.zarr"
    write_count_store(str(zarr_loc), {"RNA": _counts()}, "uint16")
    store = open_store(zarr_loc)
    store.cells.insert(
        "batch", np.where(np.arange(N_CELLS) % 2, "a", "b").astype(object)
    )
    cells = store.snapshot_cell_selection()
    normalized = store.features.normalize(
        cells, store.features.universe(from_assay="RNA")
    )
    pca = store.reduction.pca(normalized, dims=5)
    graph = store.graph.connectivity(
        store.graph.neighbors(store.graph.ann_index(pca), k=10)
    )
    return zarr_loc, {"normalized": normalized, "pca": pca, "graph": graph}


@pytest.fixture
def store(template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return open_store(target), refs


def _warned(
    call: Callable[[], ArtifactRef], topic: str
) -> tuple[ArtifactRef, list[DiagnosticWarning], list[str]]:
    """Call and return its ref, its DiagnosticWarnings, and every warning on ``topic``.

    The last list holds each warning that starts with ``topic`` and that
    CyteArc issued or logged, so a warning given twice, once from the saved
    diagnostics and once by the computation, appears twice.
    """
    with (
        collect_warnings() as collected,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        ref = call()
    diagnostic = [
        warning.message
        for warning in caught
        if isinstance(warning.message, DiagnosticWarning)
    ]
    return ref, diagnostic, [text for text in collected if text.startswith(topic)]


def _initial(n_cells: int) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n_cells, 2))


# PCA


def test_a_pca_fit_stopped_at_its_pass_limit_warns_when_computed_and_when_reused(
    store, monkeypatch
) -> None:
    datastore, refs = store
    # Above eight features the forty genes take the randomized solver, and two
    # passes leave its components far from settled.
    monkeypatch.setattr(reduction_module, "_GRAM_PCA_MAX_FEATURES", 8)
    monkeypatch.setattr(reduction_module, "_RANDOMIZED_PCA_MAX_PASSES", 2)

    def fit() -> ArtifactRef:
        return datastore.reduction.pca(refs["normalized"], dims=5)

    computed, first, first_texts = _warned(fit, "PCA's randomized solver")
    reused, second, second_texts = _warned(fit, "PCA's randomized solver")

    assert reused == computed
    diagnostics = datastore.artifacts.inspect(computed).diagnostics
    shift = diagnostics["componentShift"]
    assert diagnostics == {"solverPasses": 2, "componentShift": shift}
    # The components moved more than the 0.1 degrees at which the solver stops.
    assert 0.1 < shift <= 90
    expected = (
        "PCA's randomized solver stopped after 2 passes while its leading 5 "
        f"components still moved {shift:.2g} degrees in the last pass. Fit fewer "
        "dims or select fewer features."
    )
    for warned, texts in ((first, first_texts), (second, second_texts)):
        assert texts == [expected]
        (warning,) = warned
        assert str(warning) == expected
        assert (warning.operation, warning.diagnostic, warning.value) == (
            "run_pca",
            "componentShift",
            shift,
        )


def test_a_pca_fit_that_settles_gives_no_warning(store, monkeypatch) -> None:
    datastore, refs = store

    def fit() -> ArtifactRef:
        return datastore.reduction.pca(refs["normalized"], dims=5)

    for _ in range(2):
        exact, warned, _texts = _warned(fit, "PCA's")
        assert warned == []
    assert exact == refs["pca"]
    # The exact solver runs no passes of the randomized one.
    assert datastore.artifacts.inspect(exact).diagnostics == {
        "solverPasses": None,
        "componentShift": None,
    }

    # Blocks of fifteen vectors span all forty features in the third pass,
    # where the randomized fit is exact.
    monkeypatch.setattr(reduction_module, "_GRAM_PCA_MAX_FEATURES", 8)
    for _ in range(2):
        randomized, warned, _texts = _warned(fit, "PCA's")
        assert warned == []
    assert randomized != exact
    assert datastore.artifacts.inspect(randomized).diagnostics == {
        "solverPasses": 3,
        "componentShift": None,
    }


# Harmony


def test_a_harmony_fit_that_stops_before_converging_warns_when_computed_and_when_reused(
    store,
) -> None:
    datastore, refs = store

    def correct() -> ArtifactRef:
        return datastore.reduction.harmony(
            refs["pca"], ["batch"], harmony_params=_UNCONVERGED
        )

    topic = "Harmony stopped before convergence"
    computed, first, first_texts = _warned(correct, topic)
    reused, second, second_texts = _warned(correct, topic)

    assert reused == computed
    fit = datastore.artifacts.load(computed).attrs["fit_parameters"]
    assert (fit["rounds"], fit["converged"]) == (2, False)
    before, last = fit["objective"]
    change = abs(last - before) / abs(before)
    assert datastore.artifacts.inspect(computed).diagnostics == {
        "rounds": 2,
        "convergedRound": None,
        "objectiveChange": pytest.approx(change, rel=1e-12),
        "previousObjectiveChange": None,
        "averageObjectiveChange": None,
    }
    expected = (
        f"Harmony stopped before convergence after 2 rounds: the objective changed "
        f"by {change:.3g} of its value in the last round, above "
        "epsilon_harmony=0.0"
    )
    for warned, texts in ((first, first_texts), (second, second_texts)):
        assert texts == [expected]
        (warning,) = warned
        assert (warning.operation, warning.diagnostic) == (
            "run_harmony",
            "objectiveChange",
        )
        assert warning.value == pytest.approx(change, rel=1e-12)


def test_a_harmony_fit_that_converges_gives_no_warning(store) -> None:
    datastore, refs = store

    def correct() -> ArtifactRef:
        return datastore.reduction.harmony(
            refs["pca"], ["batch"], harmony_params={"n_clusters": 3}
        )

    for _ in range(2):
        corrected, warned, texts = _warned(correct, "Harmony stopped")
        assert (warned, texts) == ([], [])
    fit = datastore.artifacts.load(corrected).attrs["fit_parameters"]
    objective = fit["objective"]
    assert fit["converged"] is True and len(objective) >= 3
    diagnostics = datastore.artifacts.inspect(corrected).diagnostics
    assert diagnostics["rounds"] == diagnostics["convergedRound"] == fit["rounds"]
    assert diagnostics["objectiveChange"] == pytest.approx(
        abs(objective[-1] - objective[-2]) / abs(objective[-2]), rel=1e-12
    )
    assert diagnostics["previousObjectiveChange"] == pytest.approx(
        abs(objective[-2] - objective[-3]) / abs(objective[-3]), rel=1e-12
    )


# UMAP


def _equidistant(path: Path) -> tuple[DataStore, ArtifactRef]:
    """Eight cells on the unit axes, each sqrt(2) from every other, and their graph."""
    write_count_store(
        str(path), {"RNA": np.random.default_rng(2).poisson(2.0, (8, 5))}, "uint16"
    )
    store = open_store(path)
    axes = store.reduction.import_coordinates(
        np.eye(8, dtype=np.float32),
        cell_selection=store.snapshot_cell_selection(),
        name="axes",
        source={"tool": "tests", "version": "1", "parameters": {}},
    )
    neighbors = store.graph.neighbors(store.graph.ann_index(axes), k=7)
    return store, store.graph.connectivity(neighbors)


def test_densmap_over_equal_local_radii_warns_when_computed_and_when_reused(
    tmp_path,
) -> None:
    datastore, graph = _equidistant(tmp_path / "axes.zarr")

    def embed() -> ArtifactRef:
        return datastore.embeddings.umap(
            graph, _initial(8), n_epochs=5, use_density_map=True
        )

    topic = "densMAP local radii"
    computed, first, first_texts = _warned(embed, topic)
    reused, second, second_texts = _warned(embed, topic)

    assert reused == computed
    # Every distance is sqrt(2) in float32, so every local radius is equal.
    assert datastore.artifacts.inspect(computed).diagnostics == {
        "densityRadiusSpan": 0.0
    }
    expected = (
        "densMAP local radii are equal for every cell to float32 resolution, so "
        "densMAP adds no density term and lays the cells out as UMAP does"
    )
    for warned, texts in ((first, first_texts), (second, second_texts)):
        assert texts == [expected]
        (warning,) = warned
        assert (warning.operation, warning.diagnostic, warning.value) == (
            "run_umap",
            "densityRadiusSpan",
            0.0,
        )


@pytest.mark.parametrize("use_density_map", [False, True])
def test_a_layout_over_unequal_local_radii_gives_no_warning(
    store, use_density_map
) -> None:
    datastore, refs = store

    def embed() -> ArtifactRef:
        return datastore.embeddings.umap(
            refs["graph"],
            _initial(N_CELLS),
            n_epochs=5,
            use_density_map=use_density_map,
        )

    for _ in range(2):
        embedding, warned, texts = _warned(embed, "densMAP")
        assert (warned, texts) == ([], [])
    span = datastore.artifacts.inspect(embedding).diagnostics["densityRadiusSpan"]
    if use_density_map:
        # Random cells lie at many densities.
        assert span > 1e-6
    else:
        assert span is None


# Artifacts saved before their operations saved diagnostics


_CALLS: dict[str, tuple[str, str, Callable[[DataStore, dict[str, Any]], Any]]] = {
    "pca": (
        "run_pca",
        "data",
        lambda store, refs: store.reduction.pca(refs["normalized"], dims=5),
    ),
    "harmony": (
        "run_harmony",
        "data",
        lambda store, refs: store.reduction.harmony(
            refs["pca"], ["batch"], harmony_params={"n_clusters": 3}
        ),
    ),
    "umap": (
        "run_umap",
        "values",
        lambda store, refs: store.embeddings.umap(
            refs["graph"], _initial(N_CELLS), n_epochs=5
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(_CALLS))
def test_a_result_saved_without_diagnostics_is_computed_anew(store, name) -> None:
    datastore, refs = store
    operation, array, call = _CALLS[name]
    saved = call(datastore, refs)
    keys = list(datastore.artifacts.inspect(saved).diagnostics)
    # As a result saved before its operation saved diagnostics.
    del datastore.zw[artifact_path(saved)].attrs["diagnostics"]
    with pytest.raises(ValueError, match=f"holds no diagnostics, which {operation}"):
        datastore.artifacts.inspect(saved).diagnostics

    recomputed = call(datastore, refs)

    assert recomputed != saved
    assert list(datastore.artifacts.inspect(recomputed).diagnostics) == keys
    np.testing.assert_array_equal(
        datastore.artifacts.load(recomputed)[array][:],
        datastore.artifacts.load(saved)[array][:],
    )


# Pipeline runs


def test_a_rerun_pipeline_records_the_harmony_warning_of_the_correction_it_reuses(
    store,
) -> None:
    datastore, _refs = store
    settings: dict[str, Any] = {
        "hvg": {"top_n": 30},
        "pca": {"dims": 5},
        "harmony": {"batch_columns": ["batch"], "harmony_params": _UNCONVERGED},
        "filtering": False,
        "cell_cycle": False,
        "umap": False,
        "leiden": False,
        "doublets": False,
        "markers": False,
    }
    stages = []
    for _ in range(2):
        with pytest.warns(DiagnosticWarning, match="Harmony stopped before"):
            run = datastore.pipeline.rna(**settings)
        stages.append({stage["stage"]: stage for stage in run.report()["stages"]})

    first, second = (records["harmony"] for records in stages)
    (warning,) = first["warnings"]
    assert warning.startswith(
        "Harmony stopped before convergence after 2 rounds: the objective changed by "
    )
    assert warning.endswith("above epsilon_harmony=0.0")
    # The second run reuses the correction and still records its warning.
    assert [output["reused"] for output in second["outputs"]] == [True]
    assert second["warnings"] == [warning]
    assert first["summary"] == second["summary"] == {"rounds": 2, "converged": False}
