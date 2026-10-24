"""The ATAC, ADT, and multimodal recipes, the RNA recipe's QC stages, and the
input snapshots and stage summaries that the recipes record.

Module fixtures commit each run once on a private store. A test that runs a
recipe of its own copies that store, so that its run reuses their artifacts.
"""

import inspect
import re
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cytearc import (
    CancellationToken,
    DataStore,
    PipelineCancelled,
    PipelineExecutionError,
    PipelineRun,
)
from cytearc.datastore._pipeline.executor import RecipePlan
from cytearc.datastore._pipeline.recipes import atac as atac_recipe
from cytearc.datastore._pipeline.recipes import multimodal as multimodal_recipe
from cytearc.datastore._pipeline.recipes import rna as rna_recipe
from cytearc.datastore._pipeline.recipes.adt import resolve_adt
from cytearc.datastore._pipeline.recipes.atac import resolve_atac
from cytearc.datastore._pipeline.recipes.common import cell_snapshots
from cytearc.datastore._pipeline.recipes.multimodal import resolve_multimodal
from cytearc.datastore._pipeline.recipes.rna import resolve_rna
from cytearc.datastore.pipeline_accessor import (
    DEFAULT_PANEL_EXCLUDE,
    PipelineAccessor,
    PipelineEvent,
)
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.pipeline_runs import pipeline_stage_path
from tests import full_path
from tests.storage_helpers import write_count_store
from tests.test_cell_calling import _droplets
from tests.test_cell_calling_datastore import _open as _open_droplets
from tests.test_cell_calling_datastore import _write_store as _write_droplets
from tests.test_hto import planted_hto_counts
from tests.test_pipeline_run_foundation import _completed_run, _Owner, _root

# Small settings keep each run to a few seconds: one Leiden partition, and no
# cell cycle, UMAP, doublet, or marker stage unless a test needs one.
RNA_SETTINGS: dict[str, Any] = {
    "cell_cycle": False,
    "hvg": {"top_n": 200},
    "pca": {"dims": 5},
    "neighbors": {"k": 5},
    "leiden": {"partitions": [1.0]},
    "umap": False,
    "doublets": False,
    "markers": False,
}
# Runs on the synthetic stores, of 100 to 200 genes, test the stages before
# the graph, so they skip Leiden.
SMALL_RNA_SETTINGS: dict[str, Any] = {
    **RNA_SETTINGS,
    "hvg": {"top_n": 50},
    "pca": {"dims": 4},
    "leiden": False,
}
ADT_SETTINGS: dict[str, Any] = {
    "neighbors": {"k": 5},
    "leiden": {"partitions": [1.0]},
    "umap": False,
    "markers": False,
}
# Two Leiden candidates, so that cluster selection compares them.
ATAC_SETTINGS: dict[str, Any] = {
    "prevalent_peaks": {"top_n": 2000},
    "lsi": {"dims": 6},
    "neighbors": {"k": 5},
    "leiden": {"partitions": [0.5, 1.0]},
    "umap": False,
    "markers": False,
}
# Few Monte Carlo iterations, with an FDR that their p-values can reach.
CELL_CALLING: dict[str, Any] = {"n_iter": 500, "fdr": 0.01, "random_seed": 3}
# Every RNA stage but the QC stages of raw droplets and hashtags, t-SNE, which
# needs sgtsnepi, and markers, with Harmony over the batch column and 7
# neighbours instead of 5.
FULL_RNA_SETTINGS: dict[str, Any] = {
    **RNA_SETTINGS,
    "cell_cycle": True,
    "harmony": {"batch_columns": ["batch"]},
    "neighbors": {"k": 7},
    "umap": {"n_epochs": 30},
    "paris": True,
    "membership_strength": True,
    "doublets": True,
}
# The QC columns that preparation wrote for the CITE-seq store's RNA assay.
RNA_QC_COLUMNS = ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo"]
_RESOLVERS: dict[str, Callable[..., RecipePlan]] = {
    "rna": resolve_rna,
    "adt": resolve_adt,
    "atac": resolve_atac,
    "multimodal": resolve_multimodal,
}

ATAC_STAGES = [
    "input_snapshot",
    "filtering",
    "prevalent_peaks",
    "normalization",
    "lsi",
    "harmony",
    "ann_index",
    "neighbors",
    "connectivity",
    "embedding_initialization",
    "umap",
    "leiden_0.5",
    "leiden_1.0",
    "paris",
    "cluster_selection",
    "membership_strength",
    "tsne",
    "markers",
]
ADT_STAGES = [
    "input_snapshot",
    "filtering",
    "panel",
    "normalization",
    "pca",
    "harmony",
    "ann_index",
    "neighbors",
    "connectivity",
    "embedding_initialization",
    "umap",
    "leiden_1.0",
    "paris",
    "cluster_selection",
    "membership_strength",
    "tsne",
    "markers",
]
MULTIMODAL_STAGES = [
    "input_snapshot",
    "integration",
    "embedding_initialization",
    "umap",
    "leiden_1.0",
    "paris",
    "cluster_selection",
    "membership_strength",
    "tsne",
    "markers",
]


def _open_cite(location: str | Path) -> DataStore:
    """Open the CITE-seq store with its antibody assay, ``assay2``, typed ADT."""
    return DataStore(str(location), default_assay="RNA", assay_types={"assay2": "ADT"})


def _open_hashed(location: str | Path) -> DataStore:
    """Open the store of hashed cells, which keeps cells of any feature count."""
    return DataStore(str(location), default_assay="RNA", min_features_per_cell=0)


def _copy(location: str | Path, tmp_path: Path) -> Path:
    target = tmp_path / "data.zarr"
    shutil.copytree(location, target)
    return target


def _config(run: PipelineRun) -> dict[str, Any]:
    return run.report()["run"]["config"]


def _stages(run: PipelineRun) -> dict[str, dict[str, Any]]:
    return {stage["stage"]: stage for stage in run.report()["stages"]}


def _statuses(run: PipelineRun) -> dict[str, str]:
    return {stage["stage"]: stage["status"] for stage in run.report()["stages"]}


def _mask(datastore: DataStore, ref: ArtifactRef) -> np.ndarray:
    """Return a stored cell or feature selection as its boolean mask."""
    return np.asarray(datastore.artifacts.load(ref)["values"][:], dtype=bool)


def _run_ids(datastore: DataStore) -> set[str]:
    return {run.run_id for run in datastore.pipeline.list_runs(limit=1000)}


def _counts(n_cells: int, n_features: int, *, seed: int) -> np.ndarray:
    """Return Poisson counts of features whose means are gamma-distributed."""
    rng = np.random.default_rng(seed)
    means = rng.gamma(0.6, 2.0, size=n_features)
    depth = rng.uniform(0.5, 1.5, size=(n_cells, 1))
    return rng.poisson(means[None, :] * depth)


def _labels(n_cells: int, labels: tuple[str, ...]) -> np.ndarray:
    """Return ``labels`` repeated in turn over ``n_cells`` cells."""
    return np.asarray(labels)[np.arange(n_cells) % len(labels)]


def _resolve(recipe: str, datastore: DataStore, **keywords: Any) -> RecipePlan:
    """Resolve a recipe call as ``ds.pipeline.<recipe>`` does, writing nothing.

    Every keyword the call omits takes the public method's default; the
    callback and cancellation token belong to the run, not its resolution.
    The multimodal recipe takes its ``runs`` as a keyword here.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            getattr(PipelineAccessor, recipe)
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in {"callback", "cancel"}
    }
    return _RESOLVERS[recipe](datastore, **{**defaults, **keywords})


def _snapshot_refs(run: PipelineRun) -> dict[str, ArtifactRef]:
    """Return the outputs that a run's input_snapshot stage recorded, by key."""
    return {
        output["outputKey"]: ArtifactRef.from_dict(output["artifact"])
        for output in _stages(run)["input_snapshot"]["outputs"]
    }


def _snapshot_columns(
    datastore: DataStore, run: PipelineRun
) -> dict[str, list[str] | None]:
    """Return each input_snapshot output with the columns that it froze.

    A cell or feature selection among the outputs freezes no columns.
    """
    return {
        key: (
            list(datastore.artifacts.inspect(ref).parameters["ordered_columns"])
            if ref.kind == "metadata_snapshot"
            else None
        )
        for key, ref in _snapshot_refs(run).items()
    }


@pytest.fixture(scope="module")
def cite(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Commit an RNA run, and an ADT run of its cells, on a private CITE-seq store.

    The ADT assay holds ten antibodies, one of them an IgG1 isotype control.
    The ``batch`` and ``donor`` cell columns hold labels in turn, which a run
    reads only when it names them.
    """
    location = tmp_path_factory.mktemp("recipes_cite") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    datastore = _open_cite(location)
    datastore.cells.insert("batch", _labels(datastore.cells.N, ("a", "b")))
    datastore.cells.insert("donor", _labels(datastore.cells.N, ("d1", "d2", "d3")))
    rna = datastore.pipeline.rna(label="rna", **RNA_SETTINGS)
    adt = datastore.pipeline.adt(label="adt", cells=rna, **ADT_SETTINGS)
    return SimpleNamespace(location=location, datastore=datastore, rna=rna, adt=adt)


@pytest.fixture(scope="module")
def joint(cite: SimpleNamespace) -> SimpleNamespace:
    """Integrate the RNA and ADT runs with WNN and with SNN.

    The WNN run embeds the integrated graph with UMAP and ranks markers. The
    SNN run cuts the graph with Paris, scores membership strength, and freezes
    the donor column.
    """
    runs = [cite.rna, cite.adt]
    pipeline = cite.datastore.pipeline
    return SimpleNamespace(
        wnn=pipeline.multimodal(runs, label="wnn", umap={"n_epochs": 30}),
        snn=pipeline.multimodal(
            runs,
            label="snn",
            method="snn",
            umap=False,
            paris=True,
            membership_strength=True,
            markers=False,
            snapshot_columns=["donor"],
        ),
    )


@pytest.fixture(scope="module")
def full(cite: SimpleNamespace) -> PipelineRun:
    """An RNA run of the store's cells through nearly every stage.

    It corrects the batch column with Harmony, which also splits doublet
    scoring, and builds its graph of 7 neighbours instead of 5.
    """
    return cite.datastore.pipeline.rna(label="full", **FULL_RNA_SETTINGS)


@pytest.fixture(scope="module")
def snapshot(cite: SimpleNamespace) -> PipelineRun:
    """The committed RNA run's settings, with the donor column frozen."""
    return cite.datastore.pipeline.rna(snapshot_columns=["donor"], **RNA_SETTINGS)


@pytest.fixture(scope="module")
def wnn_selected(cite: SimpleNamespace) -> PipelineRun:
    """A WNN run of two Leiden partitions that saves the first one."""
    return cite.datastore.pipeline.multimodal(
        [cite.rna, cite.adt],
        integration={"l2_normalize": True},
        leiden={"partitions": [0.5, 1.0], "selected": 0.5},
        umap=False,
        markers=False,
    )


@pytest.fixture(scope="module")
def adt_own(cite: SimpleNamespace) -> PipelineRun:
    """An ADT run of the cells that ``I`` names, with every antibody and 4 PCs."""
    return cite.datastore.pipeline.adt(
        cells="I", panel={"exclude": None}, pca={"dims": 4}, **ADT_SETTINGS
    )


@pytest.fixture(scope="module")
def rna_of_selection(cite: SimpleNamespace) -> PipelineRun:
    """An RNA run given the analysis cells of the committed RNA run."""
    return cite.datastore.pipeline.rna(
        cells=cite.rna["analysis_cell_selection"], **RNA_SETTINGS
    )


@pytest.fixture(scope="module")
def rna_unreduced(cite: SimpleNamespace) -> PipelineRun:
    """An RNA run whose graph is built on its normalized values, without PCA."""
    return cite.datastore.pipeline.rna(
        **{**RNA_SETTINGS, "pca": {"dims": 0}, "leiden": False}
    )


@pytest.fixture(scope="module")
def atac(tmp_path_factory) -> SimpleNamespace:
    """Commit an ATAC run on a private copy of the 500-cell PBMC ATAC store."""
    location = tmp_path_factory.mktemp("recipes_atac")
    with tarfile.open(full_path("500_pbmc_atac.zarr.tar.gz"), "r:gz") as archive:
        archive.extractall(location, filter="data")
    datastore = DataStore(str(location))
    run = datastore.pipeline.atac(label="atac", **ATAC_SETTINGS)
    return SimpleNamespace(datastore=datastore, run=run)


@pytest.fixture(scope="module")
def atac_harmony(atac: SimpleNamespace) -> PipelineRun:
    """The ATAC run's settings with Harmony over a batch column of the store."""
    datastore = atac.datastore
    datastore.cells.insert("batch", _labels(datastore.cells.N, ("a", "b")))
    return datastore.pipeline.atac(
        harmony={"batch_columns": ["batch"]}, **ATAC_SETTINGS
    )


@pytest.fixture(scope="module")
def droplets(tmp_path_factory) -> SimpleNamespace:
    """Call cells in simulated raw droplets as one capture and per capture.

    The store holds 1,000 empty droplets and 90 cells, its ``capture`` column
    assigns the barcodes to two captures in turn, and its ``donor`` column
    holds labels that only the ``filtered`` run freezes. That run calls each
    capture as ``per_capture`` does, and then filters the called cells.
    """
    counts, truth, _ambient = _droplets(seed=7, n_features=100, n_empty=1000)
    location = tmp_path_factory.mktemp("recipes_droplets") / "raw.zarr"
    _write_droplets(location, counts)
    datastore = _open_droplets(location)
    captures = np.where(np.arange(len(counts)) % 2 == 0, "a", "b")
    datastore.cells.insert("capture", captures)
    datastore.cells.insert("donor", _labels(len(counts), ("x", "y", "z")))
    single = datastore.pipeline.rna(
        label="single",
        cell_calling=CELL_CALLING,
        filtering=False,
        **SMALL_RNA_SETTINGS,
    )
    per_capture = datastore.pipeline.rna(
        label="per_capture",
        cell_calling={**CELL_CALLING, "sample_column": "capture"},
        filtering=False,
        **SMALL_RNA_SETTINGS,
    )
    # Filtering is on by default for a run that calls cells.
    filtered = datastore.pipeline.rna(
        label="filtered",
        cell_calling={**CELL_CALLING, "sample_column": "capture"},
        snapshot_columns=["donor"],
        **SMALL_RNA_SETTINGS,
    )
    return SimpleNamespace(
        location=location,
        datastore=datastore,
        truth=truth,
        captures=captures,
        single=single,
        per_capture=per_capture,
        filtered=filtered,
    )


@pytest.fixture(scope="module")
def hashed(tmp_path_factory) -> SimpleNamespace:
    """Demultiplex planted hashtags and analyze the singlets.

    Three hashtags tag 30 cells each, 6 doublets carry two tags, and 6
    negatives carry none.
    """
    hto, truth = planted_hto_counts(n_singlets=30)
    location = tmp_path_factory.mktemp("recipes_hashing") / "data.zarr"
    write_count_store(
        str(location),
        {"RNA": _counts(len(hto), 200, seed=5), "HTO": hto.to_numpy()},
        "uint16",
    )
    datastore = _open_hashed(location)
    run = datastore.pipeline.rna(label="hashed", hashing=True, **SMALL_RNA_SETTINGS)
    return SimpleNamespace(location=location, datastore=datastore, truth=truth, run=run)


# ATAC ------------------------------------------------------------------------


def test_atac_runs_its_stages_in_order_and_records_its_recipe(atac) -> None:
    run = atac.run
    report = run.report()

    assert (run.recipe, run.assay, run.label) == ("atac", "ATAC", "atac")
    assert report["run"]["recipe"] == "atac"
    assert atac.datastore.pipeline.open(label="atac").recipe == "atac"
    assert report["run"]["stageOrder"] == ATAC_STAGES
    skipped = {
        "harmony",
        "embedding_initialization",
        "umap",
        "paris",
        "membership_strength",
        "tsne",
        "markers",
    }
    assert _statuses(run) == {
        stage: "skipped" if stage in skipped else "completed" for stage in ATAC_STAGES
    }
    # Prevalent peaks and LSI take the place of the RNA recipe's HVGs and PCA.
    assert list(run) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "prevalent_peaks",
        "normalized",
        "lsi",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "leiden_0.5",
        "leiden_1.0",
        "cluster_selection",
        "clusters",
    ]
    stages = report["run"]["config"]["stages"]
    assert stages["prevalent_peaks"] == {"top_n": 2000}
    assert stages["lsi"] == {"dims": 6}
    assert stages["filtering"]["attrs"] == ["ATAC_nCounts", "ATAC_nFeatures"]
    lsi = atac.datastore.artifacts.inspect(run["lsi"])
    assert (run["lsi"].kind, lsi.operation) == ("reduction", "run_lsi")
    cells = int(run.cells.fetch_all("I").sum())
    assert atac.datastore.artifacts.load(run["lsi"])["data"].shape == (cells, 6)


def test_atac_records_its_prevalent_peaks_as_a_feature_field(atac) -> None:
    run = atac.run
    datastore = atac.datastore

    assert run.features.columns == ("I", "ids", "names", "prevalent_peaks")
    peaks = run.features.fetch_all("prevalent_peaks")
    assert peaks.dtype == bool
    assert int(peaks.sum()) == 2000
    np.testing.assert_array_equal(peaks, _mask(datastore, run["prevalent_peaks"]))
    normalized = datastore.artifacts.inspect(run["normalized"])
    assert normalized.input_ref("feature_selection") == run["prevalent_peaks"]
    assert normalized.input_ref("cell_selection") == run["analysis_cell_selection"]


def test_atac_cluster_selection_scores_the_lsi_coordinates(atac) -> None:
    run = atac.run
    datastore = atac.datastore

    neighbors = datastore.artifacts.inspect(run["neighbors"])
    assert neighbors.input_ref("coordinates") == run["lsi"]
    selection = datastore.artifacts.inspect(run["cluster_selection"])
    assert selection.operation == "select_clusters_by_silhouette"
    assert selection.input_ref("coordinates") == run["lsi"]
    assert selection.parameters["candidateKeys"] == ["leiden_0.5", "leiden_1.0"]
    summary = _stages(run)["cluster_selection"]["summary"]
    assert set(summary["scores"]) == {"leiden_0.5", "leiden_1.0"}
    assert run["clusters"] == run[summary["selected"]]
    np.testing.assert_array_equal(
        run.cells.fetch("clusters"), run.cells.fetch(summary["selected"])
    )


def _no_assay_message(recipe: str, modality: str) -> str:
    """Return the refusal of a store that has no assay of a recipe's modality."""
    return (
        f"The store has no {modality} assay for the {recipe} recipe; a writable "
        f"open with assay_types={{'<assay>': {modality!r}}} opens an assay as "
        f"{modality}"
    )


def test_recipes_refuse_an_assay_of_another_modality(cite, atac) -> None:
    before = _run_ids(cite.datastore), _run_ids(atac.datastore)

    # An assay of a specific type is never opened as another type, so its
    # refusal names no assay_types remedy.
    for call, message in (
        (
            lambda: cite.datastore.pipeline.atac(assay="RNA"),
            "The atac recipe analyzes ATAC assays; assay 'RNA' is of type 'RNA'",
        ),
        (
            lambda: atac.datastore.pipeline.rna(assay="ATAC"),
            "The rna recipe analyzes RNA assays; assay 'ATAC' is of type 'ATAC'",
        ),
        (
            lambda: cite.datastore.pipeline.rna(assay="assay2"),
            "The rna recipe analyzes RNA assays; assay 'assay2' is of type 'ADT'",
        ),
        (
            lambda: cite.datastore.pipeline.adt(assay="RNA"),
            "The adt recipe analyzes ADT assays; assay 'RNA' is of type 'RNA'",
        ),
        (lambda: cite.datastore.pipeline.atac(), _no_assay_message("atac", "ATAC")),
        (lambda: atac.datastore.pipeline.rna(), _no_assay_message("rna", "RNA")),
        (lambda: atac.datastore.pipeline.adt(), _no_assay_message("adt", "ADT")),
    ):
        with pytest.raises(TypeError) as caught:
            call()
        assert str(caught.value) == message
    assert (_run_ids(cite.datastore), _run_ids(atac.datastore)) == before


def test_atac_settings_are_checked_before_the_run(atac) -> None:
    pipeline = atac.datastore.pipeline
    peaks = int(atac.datastore.ATAC.feats.N)
    before = _run_ids(atac.datastore)

    with pytest.raises(
        TypeError, match="prevalent_peaks must be a mapping because the stage always"
    ):
        pipeline.atac(prevalent_peaks=True)
    with pytest.raises(ValueError, match=r"Unknown lsi settings \['components'\]"):
        pipeline.atac(lsi={"components": 5})
    # The run's prevalent_peaks feature field reserves its name.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "snapshot_columns collide with reserved run fields: ['prevalent_peaks']"
        ),
    ):
        pipeline.atac(snapshot_columns=["prevalent_peaks"])
    for settings, error, message in (
        # Prevalent peaks are fewer than the assay's peaks.
        (
            {"prevalent_peaks": {"top_n": peaks}},
            ValueError,
            f"prevalent_peaks top_n ({peaks}) must be at least 1 and below the "
            f"{peaks} peaks of assay 'ATAC'",
        ),
        (
            {"prevalent_peaks": {"top_n": 0}},
            ValueError,
            "prevalent_peaks['top_n'] must be at least 1",
        ),
        ({"lsi": {"dims": 0}}, ValueError, "lsi['dims'] must be at least 1"),
        # LSI computes the component that skip_first leaves out, so it needs
        # one prevalent peak more than its dims.
        (
            {"prevalent_peaks": {"top_n": 6}, "lsi": {"dims": 6}},
            ValueError,
            "lsi dims, with the first component that skip_first leaves out, need "
            "7 of the prevalent peaks, but prevalent_peaks top_n is 6",
        ),
        ({"lsi": {"skip_first": 1}}, TypeError, "lsi skip_first must be a boolean"),
    ):
        with pytest.raises(error) as caught:
            pipeline.atac(**settings)
        assert str(caught.value) == message
    assert _run_ids(atac.datastore) == before

    # At the limits, the settings resolve.
    _resolve("atac", atac.datastore, prevalent_peaks={"top_n": peaks - 1})
    plan = _resolve(
        "atac",
        atac.datastore,
        prevalent_peaks={"top_n": 6},
        lsi={"dims": 6, "skip_first": False},
    )
    assert plan.config["stages"]["lsi"] == {"dims": 6, "skip_first": False}


def test_atac_prevalent_peaks_default_to_fewer_than_the_assay_holds(
    atac, monkeypatch
) -> None:
    peaks = int(atac.datastore.ATAC.feats.N)
    assert peaks > 10000

    # The run config records the top_n that the recipe resolved, though the
    # call gave none.
    plan = _resolve("atac", atac.datastore)
    assert plan.config["stages"]["prevalent_peaks"] == {"top_n": 10000}
    assert plan.config["stages"]["lsi"] == {}
    # An assay with no more peaks than the default keeps every peak but one.
    monkeypatch.setattr(atac_recipe, "PEAKS_TOP_N_DEFAULT", peaks)
    plan = _resolve("atac", atac.datastore)
    assert plan.config["stages"]["prevalent_peaks"] == {"top_n": peaks - 1}
    # A given top_n is recorded as it is.
    plan = _resolve("atac", atac.datastore, prevalent_peaks={"top_n": 2000})
    assert plan.config["stages"]["prevalent_peaks"] == {"top_n": 2000}


@pytest.mark.slow  # An ATAC run with Harmony, and an export of 65,908 peaks.
def test_atac_harmony_corrects_the_lsi_of_the_run_without_it(
    atac, atac_harmony
) -> None:
    pytest.importorskip("anndata")
    datastore, base, run = atac.datastore, atac.run, atac_harmony

    # Harmony's batches are frozen apart from the cells that filtering reads,
    # so the run reuses the cells, peaks, and LSI of the run without Harmony.
    for key in (
        "analysis_cell_selection",
        "prevalent_peaks",
        "normalized",
        "lsi",
    ):
        assert run[key] == base[key], key
    assert _snapshot_columns(datastore, run)["harmony_snapshot"] == ["batch"]
    harmony = datastore.artifacts.inspect(run["harmony"])
    assert harmony.input_ref("reduction") == base["lsi"]
    assert (
        harmony.input_ref("batch_snapshot") == _snapshot_refs(run)["harmony_snapshot"]
    )
    # The graph and the silhouettes of cluster selection read the corrected LSI.
    for key in ("neighbors", "cluster_selection"):
        status = datastore.artifacts.inspect(run[key])
        assert status.input_ref("coordinates") == run["harmony"], key

    exported = datastore.to_anndata(run=run)
    assert set(exported.obsm) == {"X_lsi", "X_lsi_harmony"}
    for name, key in (("X_lsi", "lsi"), ("X_lsi_harmony", "harmony")):
        np.testing.assert_array_equal(
            exported.obsm[name], datastore.artifacts.load(run[key])["data"][:]
        )


# ADT -------------------------------------------------------------------------


def test_adt_analyzes_the_store_only_adt_assay(cite) -> None:
    adt = cite.adt
    report = adt.report()

    # The default assay is RNA, so the recipe takes the only ADT assay.
    assert (adt.recipe, adt.assay) == ("adt", "assay2")
    assert report["run"]["stageOrder"] == ADT_STAGES
    assert list(adt) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "panel",
        "normalized",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "leiden_1.0",
        "cluster_selection",
        "clusters",
    ]
    assert {adt[key].assay for key in ("panel", "normalized", "pca")} == {"assay2"}


@pytest.mark.parametrize(
    ("assay_type", "remedy"),
    [
        # A hashtag assay holds no antibodies, so no remedy opens it as one.
        ("HTO", ""),
        # An assay of no specific type opens as ADT when an open says so.
        (
            "Assay",
            ", and a writable open with assay_types={'assay2': 'ADT'} opens an "
            "assay as ADT",
        ),
    ],
)
def test_adt_refuses_an_assay_that_is_not_typed_adt(
    datastore_zarr_root: str, tmp_path: Path, assay_type: str, remedy: str
) -> None:
    datastore = DataStore(
        str(_copy(datastore_zarr_root, tmp_path)),
        default_assay="RNA",
        assay_types={"assay2": assay_type},
    )

    with pytest.raises(TypeError) as caught:
        datastore.pipeline.adt(assay="assay2")
    assert str(caught.value) == (
        "The adt recipe analyzes ADT assays; "
        f"assay 'assay2' is of type '{assay_type}'{remedy}"
    )
    with pytest.raises(TypeError) as caught:
        datastore.pipeline.adt()
    assert str(caught.value) == _no_assay_message("adt", "ADT")
    assert datastore.pipeline.list_runs() == ()


def test_adt_panel_leaves_out_isotype_controls_by_default(cite) -> None:
    adt = cite.adt
    stages = _config(adt)["stages"]

    assert DEFAULT_PANEL_EXCLUDE == r"(?i)isotype|control|ctrl"
    assert stages["panel"] == {
        "exclude": DEFAULT_PANEL_EXCLUDE,
        "excluded": ["IgG1_control_TotalSeqC"],
    }
    assert adt.features.columns == ("I", "ids", "names", "panel")
    names = adt.features.fetch_all("names")
    panel = adt.features.fetch_all("panel")
    assert names[~panel].tolist() == ["IgG1_control_TotalSeqC"]
    assert int(panel.sum()) == 9
    normalized = cite.datastore.artifacts.inspect(adt["normalized"])
    assert normalized.input_ref("feature_selection") == adt["panel"]
    # PCA keeps one dimension fewer than the nine antibodies of the panel.
    assert stages["pca"] == {"dims": 8}
    assert cite.datastore.artifacts.load(adt["pca"])["data"].shape[1] == 8


def test_adt_panel_without_exclusions_keeps_every_antibody(cite, adt_own) -> None:
    stages = _config(adt_own)["stages"]

    assert stages["panel"] == {"exclude": None, "excluded": []}
    assert adt_own.features.fetch_all("panel").all()
    # An explicit dims replaces the default of one fewer than the panel.
    assert stages["pca"] == {"dims": 4}
    assert cite.datastore.artifacts.load(adt_own["pca"])["data"].shape[1] == 4


def test_adt_panel_settings_are_checked_before_the_run(cite) -> None:
    datastore = cite.datastore
    before = _run_ids(datastore)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The ADT panel keeps 1 of 10 antibodies after panel['exclude']='CD'; "
            "it needs at least two"
        ),
    ):
        datastore.pipeline.adt(cells=cite.rna, panel={"exclude": "CD"})
    with pytest.raises(ValueError, match=r"panel\['exclude'\] is not a valid pattern"):
        datastore.pipeline.adt(cells=cite.rna, panel={"exclude": "["})
    for exclude in (3, ""):
        with pytest.raises(
            TypeError,
            match=re.escape("panel['exclude'] must be a non-empty pattern or None"),
        ):
            datastore.pipeline.adt(cells=cite.rna, panel={"exclude": exclude})
    with pytest.raises(
        TypeError, match="panel must be a mapping because the stage always runs"
    ):
        datastore.pipeline.adt(cells=cite.rna, panel=True)
    with pytest.raises(ValueError, match=r"Unknown panel settings \['include'\]"):
        datastore.pipeline.adt(cells=cite.rna, panel={"include": "CD"})
    # The run's panel feature field reserves its name.
    with pytest.raises(
        ValueError,
        match=re.escape("snapshot_columns collide with reserved run fields: ['panel']"),
    ):
        datastore.pipeline.adt(cells=cite.rna, snapshot_columns=["panel"])
    # PCA needs one antibody of the panel more than its dimensions.
    with pytest.raises(ValueError) as caught:
        datastore.pipeline.adt(cells=cite.rna, pca={"dims": 9})
    assert str(caught.value) == (
        "pca dims (9) must be below the 9 antibodies of the panel: PCA needs at "
        "least dims + 1 features"
    )
    with pytest.raises(ValueError, match="^pca dims must be at least 1$"):
        datastore.pipeline.adt(cells=cite.rna, pca={"dims": 0})
    assert _run_ids(datastore) == before
    # Without exclusions the panel keeps all ten antibodies, so 9 dims resolve.
    plan = _resolve(
        "adt", datastore, cells=cite.rna, panel={"exclude": None}, pca={"dims": 9}
    )
    assert plan.config["stages"]["pca"] == {"dims": 9}


def test_adt_and_atac_markers_and_membership_need_a_leiden_candidate(
    cite, atac
) -> None:
    for recipe, datastore, keywords in (
        ("adt", cite.datastore, {"cells": cite.rna}),
        ("atac", atac.datastore, {}),
    ):
        run = getattr(datastore.pipeline, recipe)
        before = _run_ids(datastore)
        # Markers are on by default.
        with pytest.raises(
            ValueError, match="^markers require at least one Leiden candidate$"
        ):
            run(leiden=False, **keywords)
        with pytest.raises(
            ValueError,
            match="^membership_strength requires at least one Leiden candidate$",
        ):
            run(leiden=False, markers=False, membership_strength=True, **keywords)
        assert _run_ids(datastore) == before
        # Without them, a run clusters nothing.
        plan = _resolve(recipe, datastore, leiden=False, markers=False, **keywords)
        assert plan.config["stages"]["leiden"] is False
        assert not [stage for stage in plan.stage_order if stage.startswith("leiden")]


def test_adt_with_the_cells_of_an_rna_run_reuses_its_selection(cite) -> None:
    rna, adt = cite.rna, cite.adt
    selection = rna["analysis_cell_selection"]

    assert adt["input_cell_selection"] == selection
    assert adt["analysis_cell_selection"] == selection
    assert _config(adt)["cells"] == {
        "run": {"runId": rna.run_id, "label": "rna"},
        "selection": selection.to_dict(),
    }
    np.testing.assert_array_equal(adt.cells.fetch("ids"), rna.cells.fetch("ids"))


def test_adt_filters_the_cells_of_a_run_when_asked(cite, tmp_path: Path) -> None:
    datastore = _open_cite(_copy(cite.location, tmp_path))
    rna = datastore.pipeline.open(label="rna")

    filtered = datastore.pipeline.adt(
        cells=rna, filtering={"attrs": ["assay2_nCounts"]}, **ADT_SETTINGS
    )

    filtering = _config(filtered)["stages"]["filtering"]
    assert filtering["enabled"] is True
    assert (filtering["method"], filtering["attrs"]) == ("mad", ["assay2_nCounts"])
    stage = _stages(filtered)["filtering"]
    assert stage["status"] == "completed"
    assert filtered["input_cell_selection"] == rna["analysis_cell_selection"]
    # The filter keeps a subset of the RNA run's cells, and its summary
    # accounts for each of them.
    given = _mask(datastore, filtered["input_cell_selection"])
    kept = _mask(datastore, filtered["analysis_cell_selection"])
    assert not (kept & ~given).any()
    summary = stage["summary"]
    assert summary["inputCells"] == int(given.sum())
    assert summary["cells"] == int(kept.sum())
    assert summary["cells"] + summary["removedCells"] == summary["inputCells"]


# Multimodal ------------------------------------------------------------------


def test_wnn_integrates_the_runs_of_one_cell_selection(cite, joint) -> None:
    wnn = joint.wnn
    rna, adt = cite.rna, cite.adt
    report = wnn.report()

    # The joint run's assay and feature view are the first run's.
    assert (wnn.recipe, wnn.assay, wnn.label) == ("multimodal", "RNA", "wnn")
    assert wnn.features.columns == ("I", "ids", "names")
    assert report["run"]["stageOrder"] == MULTIMODAL_STAGES
    completed = [
        stage for stage, status in _statuses(wnn).items() if status == "completed"
    ]
    assert completed == [
        "input_snapshot",
        "integration",
        "embedding_initialization",
        "umap",
        "leiden_1.0",
        "markers",
    ]
    assert list(wnn) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "integrated_graph",
        "embedding_initialization",
        "umap",
        "leiden_1.0",
        "clusters",
        "markers_RNA",
        "markers_assay2",
    ]
    selection = rna["analysis_cell_selection"]
    assert wnn["input_cell_selection"] == selection
    assert wnn["analysis_cell_selection"] == selection
    assert wnn["feature_universe"] == rna["feature_universe"]
    # WNN weighs each run's neighbours in the coordinates that found them.
    graph = cite.datastore.artifacts.inspect(wnn["integrated_graph"])
    assert graph.parameters["method"] == "wnn"
    assert graph.parameters["assays"] == ["RNA", "assay2"]
    for index, run in enumerate((rna, adt)):
        source = graph.inputs[f"source_{index}"]
        assert ArtifactRef.from_dict(dict(source["neighbors"])) == run["neighbors"]
        assert ArtifactRef.from_dict(dict(source["coordinates"])) == run["pca"]
    config = report["run"]["config"]
    assert config["method"] == "wnn"
    assert config["sources"] == [
        {"runId": rna.run_id, "label": "rna", "recipe": "rna", "assay": "RNA"},
        {"runId": adt.run_id, "label": "adt", "recipe": "adt", "assay": "assay2"},
    ]
    assert config["cells"] == {"selection": selection.to_dict()}
    assert config["snapshotColumns"] == []


def test_multimodal_saves_its_only_leiden_partition(joint) -> None:
    wnn = joint.wnn

    assert _config(wnn)["stages"]["leiden"] == {"partitions": [1.0], "selected": 1.0}
    # The integrated graph has no coordinates in which to score silhouettes.
    assert _statuses(wnn)["cluster_selection"] == "skipped"
    assert "cluster_selection" not in wnn
    assert wnn["clusters"] == wnn["leiden_1.0"]
    np.testing.assert_array_equal(
        wnn.cells.fetch("clusters"), wnn.cells.fetch("leiden_1.0")
    )


def test_multimodal_needs_a_selected_partition_among_several(
    cite, wnn_selected
) -> None:
    runs = [cite.rna, cite.adt]
    before = _run_ids(cite.datastore)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "A multimodal run cannot choose among several Leiden partitions by "
            "silhouette; pass leiden={'partitions': [...], 'selected': ...}"
        ),
    ):
        cite.datastore.pipeline.multimodal(runs, leiden={"partitions": [0.5, 1.0]})
    with pytest.raises(ValueError, match="leiden selected must be one of the"):
        cite.datastore.pipeline.multimodal(
            runs, leiden={"partitions": [0.5, 1.0], "selected": 2.0}
        )
    assert _run_ids(cite.datastore) == before

    assert _config(wnn_selected)["stages"]["leiden"] == {
        "partitions": [0.5, 1.0],
        "selected": 0.5,
    }
    assert wnn_selected["clusters"] == wnn_selected["leiden_0.5"]
    assert {"leiden_0.5", "leiden_1.0", "clusters"} <= set(wnn_selected.cells.columns)
    assert _statuses(wnn_selected)["cluster_selection"] == "skipped"
    assert "cluster_selection" not in wnn_selected


def test_wnn_records_the_modality_weights_of_each_cell(cite, joint) -> None:
    wnn = joint.wnn

    assert wnn.cells.columns == (
        "I",
        "ids",
        "names",
        "umap_1",
        "umap_2",
        "leiden_1.0",
        "clusters",
        "weight_RNA",
        "weight_assay2",
    )
    rna_weights = wnn.cells.fetch("weight_RNA")
    adt_weights = wnn.cells.fetch("weight_assay2")
    assert rna_weights.shape == (int(cite.rna.cells.fetch_all("I").sum()),)
    assert ((rna_weights >= 0) & (rna_weights <= 1)).all()
    np.testing.assert_allclose(rna_weights + adt_weights, 1.0, atol=1e-5)
    summary = _stages(wnn)["integration"]["summary"]
    assert summary["assays"] == ["RNA", "assay2"]
    assert summary["meanWeights"]["RNA"] == pytest.approx(
        float(rna_weights.mean()), abs=1e-4
    )
    assert summary["meanWeights"]["assay2"] == pytest.approx(
        float(adt_weights.mean()), abs=1e-4
    )


def test_multimodal_markers_rank_each_assay_for_the_joint_clusters(cite, joint) -> None:
    wnn = joint.wnn
    clusters = len(np.unique(wnn.cells.fetch("clusters")))
    summary = _stages(wnn)["markers"]["summary"]

    assert set(summary) == {"markers_RNA", "markers_assay2"}
    for assay, run in (("RNA", cite.rna), ("assay2", cite.adt)):
        ref = wnn[f"markers_{assay}"]
        assert (ref.kind, ref.assay) == ("marker_table", assay)
        status = cite.datastore.artifacts.inspect(ref)
        assert status.input_ref("clusters") == wnn["clusters"]
        assert status.input_ref("cell_selection") == wnn["analysis_cell_selection"]
        # Every feature of the assay is ranked.
        assert status.input_ref("feature_selection") == run["feature_universe"]
        assert summary[f"markers_{assay}"]["groups"] == clusters


def test_snn_shares_the_connectivity_graphs_of_the_runs(cite, joint) -> None:
    snn = joint.snn
    config = _config(snn)

    assert (config["method"], config["stages"]["integration"]) == ("snn", {})
    assert config["sources"] == _config(joint.wnn)["sources"]
    graph = cite.datastore.artifacts.inspect(snn["integrated_graph"])
    assert graph.parameters == {"method": "snn", "assays": ["RNA", "assay2"]}
    assert graph.input_ref("source_0") == cite.rna["connectivity_map"]
    assert graph.input_ref("source_1") == cite.adt["connectivity_map"]
    # SNN weighs no modality, and markers=False skips the marker stage.
    assert "weight_RNA" not in snn.cells.columns
    assert _statuses(snn)["markers"] == "skipped"
    assert not any(key.startswith("markers") for key in snn)
    assert snn["clusters"] == snn["leiden_1.0"]


def test_a_joint_umap_starts_from_the_first_runs_coordinates(cite, joint) -> None:
    datastore, wnn = cite.datastore, joint.wnn

    # The integrated graph has no coordinates, so the embedding is initialized
    # from the PCA that built the first run's graph.
    initialization = datastore.artifacts.inspect(wnn["embedding_initialization"])
    assert initialization.input_ref("coordinates") == cite.rna["pca"]
    umap = datastore.artifacts.inspect(wnn["umap"])
    assert umap.input_ref("graph") == wnn["integrated_graph"]
    assert umap.input_ref("initialization") == wnn["embedding_initialization"]
    assert umap.input_ref("cell_selection") == wnn["analysis_cell_selection"]
    assert _config(wnn)["stages"]["umap"] == {"n_epochs": 30}
    coordinates = datastore.artifacts.load(wnn["umap"])["values"][:]
    np.testing.assert_array_equal(
        np.column_stack([wnn.cells.fetch("umap_1"), wnn.cells.fetch("umap_2")]),
        coordinates,
    )
    summaries = {name: stage["summary"] for name, stage in _stages(wnn).items()}
    assert summaries["umap"] == {"cells": len(coordinates), "dims": 2}
    assert set(summaries["embedding_initialization"]) == {"centroids", "dims"}


def test_snn_cuts_and_scores_the_integrated_graph(cite, joint) -> None:
    datastore, snn = cite.datastore, joint.snn
    graph = snn["integrated_graph"]

    assert list(snn) == [
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "integrated_graph",
        "leiden_1.0",
        "paris",
        "clusters",
        "membership_strength",
    ]
    assert (
        datastore.artifacts.inspect(snn["paris"]).input_ref("connectivity_map") == graph
    )
    membership = datastore.artifacts.inspect(snn["membership_strength"])
    assert membership.input_ref("clusters") == snn["clusters"]
    assert membership.input_ref("connectivity_map") == graph
    # The frozen donor column comes first among the run's own cell fields.
    assert snn.cells.columns == (
        "I",
        "ids",
        "names",
        "donor",
        "leiden_1.0",
        "paris",
        "clusters",
        "membership_strength",
    )
    selected = snn.cells.fetch_all("I")
    np.testing.assert_array_equal(
        snn.cells.fetch("donor"), cite.datastore.cells.fetch_all("donor")[selected]
    )
    summaries = {name: stage["summary"] for name, stage in _stages(snn).items()}
    assert summaries["integration"] == {"assays": ["RNA", "assay2"]}
    assert summaries["paris"] == {"clusters": len(np.unique(snn.cells.fetch("paris")))}
    strength = snn.cells.fetch("membership_strength")
    assert summaries["membership_strength"] == {
        "mean": pytest.approx(float(strength.mean()), abs=1e-4),
        "belowHalf": int(np.count_nonzero(strength < 0.5)),
    }


def test_a_multimodal_run_exports_the_raw_counts_of_its_first_run(cite, joint) -> None:
    pytest.importorskip("anndata")
    datastore, wnn = cite.datastore, joint.wnn
    selected = wnn.cells.fetch_all("I")

    exported = datastore.to_anndata(run=wnn)

    # The joint run's feature view is the first run's assay, RNA, whose raw
    # counts the export holds for each cell of the run.
    universe = wnn.features.fetch_all("I")
    assert exported.shape == (int(selected.sum()), int(universe.sum()))
    assert exported.obs_names.tolist() == wnn.cells.fetch("ids").astype(str).tolist()
    assert (
        exported.var_names.tolist()
        == datastore.RNA.feats.fetch_all("ids")[universe].astype(str).tolist()
    )
    np.testing.assert_array_equal(
        np.asarray(exported.X.sum(axis=1)).ravel(),
        datastore.cells.fetch_all("RNA_nCounts")[selected],
    )
    # Each cell's modality weights are obs columns, and the joint UMAP its
    # coordinates; the run has no reduction of its own.
    for assay in ("RNA", "assay2"):
        np.testing.assert_array_equal(
            exported.obs[f"weight_{assay}"].to_numpy(),
            wnn.cells.fetch(f"weight_{assay}"),
        )
    assert set(exported.obsm) == {"X_umap"}
    np.testing.assert_array_equal(
        exported.obsm["X_umap"], datastore.artifacts.load(wnn["umap"])["values"][:]
    )
    # A joint run has no normalized values of its own to export.
    with pytest.raises(ValueError) as caught:
        datastore.to_anndata(run=wnn, matrix="normed")
    assert str(caught.value) == (
        f"Pipeline run {wnn.run_id} has no normalized output, so it has no "
        "normalized values to export; export raw counts with matrix='raw'"
    )


def test_multimodal_integrates_modality_runs_not_joint_runs(cite, joint) -> None:
    before = _run_ids(cite.datastore)

    # A joint run has neither the neighbors that WNN weighs nor the
    # connectivity graph that SNN shares.
    for method, output in (("wnn", "neighbors"), ("snn", "connectivity_map")):
        with pytest.raises(ValueError) as caught:
            cite.datastore.pipeline.multimodal([joint.wnn, cite.adt], method=method)
        assert str(caught.value) == (
            f"Run {joint.wnn.run_id} (multimodal) has no [{output!r}] output to "
            "integrate"
        )
    assert _run_ids(cite.datastore) == before


def test_snn_needs_modality_runs_of_one_neighbour_count(cite, full) -> None:
    before = _run_ids(cite.datastore)
    runs = [full, cite.adt]
    # The runs share one cell selection, so only their neighbour counts differ.
    assert full["analysis_cell_selection"] == cite.adt["analysis_cell_selection"]

    with pytest.raises(ValueError) as caught:
        cite.datastore.pipeline.multimodal(runs, method="snn")
    assert str(caught.value) == (
        "SNN integration needs graphs with the same number of neighbors per cell, "
        "but the runs have {'RNA': 7, 'assay2': 5}: run each modality with the "
        "same neighbors={'k': ...}, or integrate with method='wnn', which keeps "
        "the smallest k"
    )
    assert _run_ids(cite.datastore) == before
    # WNN weighs the runs' neighbours as they are.
    assert _resolve("multimodal", cite.datastore, runs=runs).config["method"] == "wnn"


def test_multimodal_needs_runs_of_one_shared_cell_selection(cite, adt_own) -> None:
    rna = cite.rna

    with pytest.raises(
        ValueError,
        match=(
            rf"Run {adt_own.run_id} \(assay2\) analyzes another cell selection "
            rf"than run {rna.run_id} \(RNA\)\. .*even of the same cells, as two "
            r"runs over cells='I' do: .*cells=<first run>, such as "
            r"ds\.pipeline\.adt\(cells=rna_run\)"
        ),
    ):
        cite.datastore.pipeline.multimodal([rna, adt_own])


def test_multimodal_integrates_runs_of_its_own_store(cite, atac) -> None:
    before = _run_ids(cite.datastore)

    # A run of another store fails before this store records anything.
    for method in ("wnn", "snn"):
        with pytest.raises(
            ValueError,
            match=(
                rf"Integrate runs of this datastore; run {atac.run.run_id} is not "
                "recorded in it"
            ),
        ):
            cite.datastore.pipeline.multimodal([cite.rna, atac.run], method=method)
    assert _run_ids(cite.datastore) == before


def test_multimodal_integrates_one_run_per_assay(cite, rna_of_selection) -> None:
    with pytest.raises(
        ValueError, match=re.escape("Integrate one run per assay, not ['RNA', 'RNA']")
    ):
        cite.datastore.pipeline.multimodal([cite.rna, rna_of_selection])


def test_wnn_refuses_a_graph_built_on_normalized_values(cite, rna_unreduced) -> None:
    datastore = cite.datastore

    assert "pca" not in rna_unreduced
    neighbors = datastore.artifacts.inspect(rna_unreduced["neighbors"])
    assert neighbors.input_ref("coordinates") == rna_unreduced["normalized"]
    # Its cells are the ADT run's, so only its coordinates are refused.
    assert (
        rna_unreduced["analysis_cell_selection"] == cite.adt["analysis_cell_selection"]
    )
    with pytest.raises(
        ValueError,
        match=(
            "WNN weighs each modality's reduced coordinates, but run "
            rf"{rna_unreduced.run_id} \(RNA\) built its graph on normalized values"
        ),
    ):
        datastore.pipeline.multimodal([rna_unreduced, cite.adt])


def test_multimodal_checks_its_method_runs_and_settings(cite) -> None:
    datastore = cite.datastore
    runs = [cite.rna, cite.adt]
    before = _run_ids(datastore)

    with pytest.raises(ValueError, match=r"method must be 'wnn' or 'snn'"):
        datastore.pipeline.multimodal(runs, method="cca")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown integration settings ['k']; allowed: "
            "['chunk_size', 'l2_normalize']"
        ),
    ):
        datastore.pipeline.multimodal(runs, integration={"k": 5})
    with pytest.raises(
        TypeError, match="integration must be a mapping because the stage always runs"
    ):
        datastore.pipeline.multimodal(runs, integration=True)
    with pytest.raises(TypeError, match="runs must be a sequence of completed"):
        datastore.pipeline.multimodal(cite.rna)
    with pytest.raises(ValueError, match="integrates at least two modality runs"):
        datastore.pipeline.multimodal([cite.rna])
    with pytest.raises(TypeError, match="runs must contain PipelineRun values"):
        datastore.pipeline.multimodal([cite.rna, "adt"])
    # The joint run's modality weights reserve their names.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "snapshot_columns collide with reserved run fields: ['weight_assay2']"
        ),
    ):
        datastore.pipeline.multimodal(runs, snapshot_columns=["weight_assay2"])
    # Markers, on by default, and membership strength need joint clusters.
    for settings in ({}, {"markers": False, "membership_strength": True}):
        with pytest.raises(
            ValueError,
            match=(
                "^markers and membership_strength require at least one Leiden "
                "candidate$"
            ),
        ):
            datastore.pipeline.multimodal(runs, leiden=False, **settings)
    assert _run_ids(datastore) == before


def test_wnn_needs_neighbors_that_record_their_coordinates(cite, monkeypatch) -> None:
    datastore = cite.datastore
    neighbors = cite.rna["neighbors"]
    before = _run_ids(datastore)
    # A neighbors artifact whose record names no coordinates cannot show WNN
    # the space its graph was built in.
    monkeypatch.setattr(
        multimodal_recipe, "inspect_artifact", lambda *_args: SimpleNamespace(inputs={})
    )

    with pytest.raises(
        ValueError,
        match=f"^Neighbors artifact {neighbors.artifact_id} has no coordinates$",
    ):
        datastore.pipeline.multimodal([cite.rna, cite.adt])
    assert _run_ids(datastore) == before


class _WithoutNeighbors:
    """A completed run seen without its neighbors output."""

    def __init__(self, run: PipelineRun) -> None:
        self._run = run

    def __contains__(self, key: str) -> bool:
        return key != "neighbors" and key in self._run

    def __getitem__(self, key: str) -> Any:
        return self._run[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)


def test_a_joint_embedding_starts_from_the_first_runs_neighbors(
    cite, monkeypatch
) -> None:
    datastore = cite.datastore
    original = multimodal_recipe._resolve_runs

    def first_without_neighbors(
        store: Any, runs: object, method: str
    ) -> tuple[tuple[Any, ...], ArtifactRef]:
        sources, selection = original(store, runs, method)
        return (_WithoutNeighbors(sources[0]), *sources[1:]), selection

    monkeypatch.setattr(multimodal_recipe, "_resolve_runs", first_without_neighbors)
    before = _run_ids(datastore)

    # SNN integrates connectivity graphs, but a joint UMAP still starts from
    # the coordinates of the first run's neighbors.
    with pytest.raises(
        ValueError,
        match=(
            f"^Run {cite.rna.run_id} has no neighbors output whose coordinates "
            "could initialize the joint embedding$"
        ),
    ):
        datastore.pipeline.multimodal([cite.rna, cite.adt], method="snn")
    assert _run_ids(datastore) == before


def test_l2_normalize_applies_only_to_wnn(cite, joint, wnn_selected) -> None:
    with pytest.raises(ValueError, match=r"l2_normalize applies only to method='wnn'"):
        cite.datastore.pipeline.multimodal(
            [cite.rna, cite.adt], method="snn", integration={"l2_normalize": True}
        )

    assert _config(wnn_selected)["stages"]["integration"] == {"l2_normalize": True}
    assert _config(joint.wnn)["stages"]["integration"] == {}
    # True is the default of integration.modalities, so both WNN runs share
    # one integrated graph, and its resolution 1.0 partition.
    graph = cite.datastore.artifacts.inspect(joint.wnn["integrated_graph"])
    assert graph.parameters["l2_normalize"] is True
    assert wnn_selected["integrated_graph"] == joint.wnn["integrated_graph"]
    assert wnn_selected["leiden_1.0"] == joint.wnn["leiden_1.0"]


# RNA QC stages -----------------------------------------------------------------


def test_cell_calling_calls_the_cells_of_raw_droplets(droplets) -> None:
    datastore, run = droplets.datastore, droplets.single

    assert run.report()["run"]["stageOrder"][:5] == [
        "input_snapshot",
        "cell_calling",
        "hashing",
        "filtering",
        "cell_cycle",
    ]
    statuses = _statuses(run)
    assert [statuses[stage] for stage in ("cell_calling", "hashing", "filtering")] == [
        "completed",
        "skipped",
        "skipped",
    ]
    assert list(run)[:4] == [
        "input_cell_selection",
        "cell_calling",
        "called_cell_selection",
        "analysis_cell_selection",
    ]
    statistics = run["cell_calling"]
    assert (statistics.kind, statistics.scope, statistics.assay) == (
        "cell_calling",
        "assay",
        "RNA",
    )
    called = run["called_cell_selection"]
    # The stage calls qc.call_cells on the run's input cells with its settings.
    assert (
        datastore.qc.call_cells(
            cell_selection=run["input_cell_selection"], **CELL_CALLING
        )
        == called
    )
    assert datastore.artifacts.inspect(called).input_ref("cell_calling") == statistics
    mask = _mask(datastore, called)
    truth = droplets.truth
    assert mask[truth].mean() > 0.9
    assert mask[~truth].mean() < 0.01
    # filtering=False analyzes the called cells as they are.
    assert run["analysis_cell_selection"] == called
    np.testing.assert_array_equal(run.cells.fetch_all("I"), mask)
    summary = _stages(run)["cell_calling"]["summary"]
    assert {"retain", "knee", "inflection", "n_ambient"} <= set(summary["cell_calling"])
    assert summary["called_cell_selection"] == {
        "cells": int(mask.sum()),
        "inputCells": len(mask),
        "removedCells": len(mask) - int(mask.sum()),
    }
    assert _config(run)["stages"]["cell_calling"] == CELL_CALLING


def test_cell_calling_calls_each_capture_apart(droplets) -> None:
    datastore, run = droplets.datastore, droplets.per_capture

    assert "cell_calling" not in run
    called = run["called_cell_selection"]
    status = datastore.artifacts.inspect(called)
    assert status.operation == "combine_capture_cell_calls"
    assert status.parameters["sampleColumn"] == "capture"
    assert status.parameters["captures"] == ["a", "b"]
    union = np.zeros(len(droplets.captures), dtype=bool)
    per_capture: dict[str, int] = {}
    capture_calls: dict[str, ArtifactRef] = {}
    for index, capture in enumerate(("a", "b")):
        capture_call = status.input_ref(f"capture_{index}")
        capture_calls[capture] = capture_call
        capture_cells = datastore.artifacts.inspect(capture_call).input_ref(
            "prior_cell_selection"
        )
        np.testing.assert_array_equal(
            _mask(datastore, capture_cells), droplets.captures == capture
        )
        # Each capture is called alone, with its own ambient profile.
        assert (
            datastore.qc.call_cells(cell_selection=capture_cells, **CELL_CALLING)
            == capture_call
        )
        mask = _mask(datastore, capture_call)
        assert not (mask & (droplets.captures != capture)).any()
        per_capture[capture] = int(mask.sum())
        union |= mask
    np.testing.assert_array_equal(_mask(datastore, called), union)
    assert union[droplets.truth].mean() > 0.9
    summary = _stages(run)["cell_calling"]["summary"]
    knees = summary.pop("kneePerCapture")
    assert summary == {
        "cells": int(union.sum()),
        "inputCells": len(union),
        "removedCells": len(union) - int(union.sum()),
        "calledPerCapture": per_capture,
    }
    # Each capture reports the knee of its own barcode-rank curve.
    assert set(knees) == set(per_capture)
    for capture, knee in knees.items():
        fit = datastore.artifacts.inspect(capture_calls[capture]).input_ref(
            "cell_calling"
        )
        assert knee == artifact_group(datastore.zw, fit).attrs["knee"] > 0
    assert _config(run)["stages"]["cell_calling"] == {
        **CELL_CALLING,
        "sample_column": "capture",
    }
    assert run["analysis_cell_selection"] == called


def test_cell_calling_settings_are_checked_before_the_run(droplets) -> None:
    datastore = droplets.datastore
    before = _run_ids(datastore)

    with pytest.raises(
        ValueError, match=r"Unknown cell_calling settings \['iterations'\]"
    ):
        datastore.pipeline.rna(cell_calling={"iterations": 10}, **SMALL_RNA_SETTINGS)
    with pytest.raises(KeyError, match="Capture column 'sample' was not found"):
        datastore.pipeline.rna(
            cell_calling={"sample_column": "sample"}, **SMALL_RNA_SETTINGS
        )
    with pytest.raises(TypeError, match="cell_calling must be a mapping or bool"):
        datastore.pipeline.rna(cell_calling="emptydrops", **SMALL_RNA_SETTINGS)
    # The settings are checked as qc.call_cells checks them, for one capture
    # and per capture alike.
    for settings, message in (
        ({"fdr": 2}, "fdr must be greater than 0 and at most 1"),
        ({"lower": 0}, "lower must be at least 1"),
        ({"n_iter": 0}, "n_iter must be at least 1"),
        ({"alpha": -1}, "alpha must be positive"),
        ({"lower": 50, "retain": 50}, "retain must be at least 51"),
        ({"random_seed": -1}, "random_seed must be at least 0"),
    ):
        for capture in ({}, {"sample_column": "capture"}):
            with pytest.raises(ValueError) as caught:
                datastore.pipeline.rna(
                    cell_calling={**settings, **capture}, **SMALL_RNA_SETTINGS
                )
            assert str(caught.value) == message
    assert _run_ids(datastore) == before


def _unlabeled_cells(column: str, count: str) -> str:
    """Return the pattern of the refusal of ``count`` input cells without a label."""
    verb = "has" if count == "1" else "have"
    return (
        f"Cell calling per capture needs a {re.escape(repr(column))} label for "
        f"every cell, but {count} of the run's cells {verb} none"
    )


def test_cell_calling_per_capture_needs_a_label_for_every_input_cell(
    droplets, tmp_path: Path, monkeypatch
) -> None:
    datastore = _open_droplets(_copy(droplets.location, tmp_path))
    # Cell 5 has no capture label, and cells 7 and 8 a NaN code.
    gaps = droplets.captures.astype(object)
    gaps[5] = None
    datastore.cells.insert("capture_gaps", gaps)
    codes = np.where(droplets.captures == "a", 1.0, 2.0)
    codes[[7, 8]] = np.nan
    datastore.cells.insert("capture_codes", codes)
    before = _run_ids(datastore)

    for column, count in (("capture_gaps", "1"), ("capture_codes", "2")):
        with pytest.raises(ValueError) as caught:
            datastore.pipeline.rna(
                cell_calling={**CELL_CALLING, "sample_column": column},
                **SMALL_RNA_SETTINGS,
            )
        assert re.fullmatch(_unlabeled_cells(column, count), str(caught.value))
    # The cells of a selection are checked as the cells of a column are.
    with pytest.raises(ValueError) as caught:
        datastore.pipeline.rna(
            cells=datastore.snapshot_cell_selection("I"),
            cell_calling={**CELL_CALLING, "sample_column": "capture_gaps"},
            **SMALL_RNA_SETTINGS,
        )
    assert re.fullmatch(_unlabeled_cells("capture_gaps", "1"), str(caught.value))
    assert _run_ids(datastore) == before
    # Only the run's own cells need a label.
    labelled = np.ones(datastore.cells.N, dtype=bool)
    labelled[[5, 7, 8]] = False
    datastore.cells.insert("labelled", labelled)
    for column in ("capture_gaps", "capture_codes"):
        _resolve(
            "rna",
            datastore,
            cells="labelled",
            cell_calling={**CELL_CALLING, "sample_column": column},
            **SMALL_RNA_SETTINGS,
        )

    # The stage checks the labels it froze, which can differ from the live
    # column that the recipe checked.
    monkeypatch.setattr(rna_recipe, "_check_capture_labels", lambda *_args: None)
    with pytest.raises(PipelineExecutionError) as failed:
        datastore.pipeline.rna(
            cell_calling={**CELL_CALLING, "sample_column": "capture_gaps"},
            **SMALL_RNA_SETTINGS,
        )
    assert failed.value.stage == "cell_calling"
    assert re.fullmatch(
        _unlabeled_cells("capture_gaps", "1"), str(failed.value.__cause__)
    )
    assert datastore.pipeline.open(run_id=failed.value.run_id).status == "failed"


def test_a_failed_capture_call_names_its_capture_and_column(
    droplets, tmp_path: Path
) -> None:
    datastore = _open_droplets(_copy(droplets.location, tmp_path))
    # Lane 'a' holds empty droplets of no more than 100 counts, which
    # EmptyDrops cannot test, and lane 'b' every other barcode.
    totals = np.asarray(datastore.cells.fetch_all("RNA_nCounts"))
    lanes = np.where((totals <= 100) & ~droplets.truth, "a", "b")
    lanes[200:] = "b"
    datastore.cells.insert("lane", lanes)

    with pytest.raises(PipelineExecutionError) as failed:
        datastore.pipeline.rna(
            cell_calling={**CELL_CALLING, "lower": 100, "sample_column": "lane"},
            **SMALL_RNA_SETTINGS,
        )

    assert failed.value.stage == "cell_calling"
    cause = failed.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause) == (
        "Cell calling failed for capture 'a' of 'lane': No barcode has more than "
        "lower=100 counts, so no barcode can be tested; decrease lower to test "
        "barcodes with fewer counts."
    )
    assert isinstance(cause.__cause__, ValueError)
    assert datastore.pipeline.open(run_id=failed.value.run_id).status == "failed"


def test_per_capture_calls_ignore_the_filtering_and_snapshot_settings(
    droplets,
) -> None:
    datastore, per_capture, filtered = (
        droplets.datastore,
        droplets.per_capture,
        droplets.filtered,
    )

    # The runs differ in filtering and snapshot columns only. The capture
    # labels are frozen apart from both, so each capture's call is reused.
    plans = _stages(filtered)["cell_calling"]["plans"]
    assert [plan["operation"] for plan in plans] == [
        *("select_capture_cells", "run_cell_calling", "call_cells") * 2,
        "combine_capture_cell_calls",
    ]
    assert {plan["disposition"] for plan in plans} == {"reused"}
    assert filtered["called_cell_selection"] == per_capture["called_cell_selection"]
    # Filtering then keeps a subset of the called cells.
    assert _config(filtered)["stages"]["filtering"]["enabled"] is True
    called = _mask(datastore, filtered["called_cell_selection"])
    kept = _mask(datastore, filtered["analysis_cell_selection"])
    assert not (kept & ~called).any()
    summary = _stages(filtered)["filtering"]["summary"]
    assert (summary["inputCells"], summary["cells"]) == (called.sum(), kept.sum())
    np.testing.assert_array_equal(
        filtered.cells.fetch("donor"), datastore.cells.fetch_all("donor")[kept]
    )


def test_hashing_keeps_the_singlets_of_one_hashtag(hashed) -> None:
    datastore, run = hashed.datastore, hashed.run

    assert list(run)[:4] == [
        "input_cell_selection",
        "hashing",
        "hashed_cell_selection",
        "analysis_cell_selection",
    ]
    assert run["hashing"].kind == "hto_identity"
    # The stage demultiplexes the run's input cells with the HTO assay.
    assert (
        datastore.qc.hto_demultiplexing(run["input_cell_selection"], from_assay="HTO")
        == run["hashing"]
    )
    singlets = ~np.isin(hashed.truth, ["Doublet", "Negative"])
    np.testing.assert_array_equal(
        _mask(datastore, run["hashed_cell_selection"]), singlets
    )
    assert not (run.cells.fetch_all("I") & ~singlets).any()
    assert _stages(run)["hashing"]["summary"] == {
        "hashing": {
            "hashtags": {
                "HTO0": 30,
                "HTO1": 30,
                "HTO2": 30,
                "Doublet": 6,
                "Negative": 6,
            },
            "negative": 6,
            "multiplets": 6,
        },
        "hashed_cell_selection": {"cells": 90, "inputCells": 102, "removedCells": 12},
    }
    assert _config(run)["stages"]["hashing"] == {"assay": "HTO"}


def test_hashing_filters_within_each_hashtag(hashed) -> None:
    datastore, run = hashed.datastore, hashed.run

    filtering = _config(run)["stages"]["filtering"]
    assert filtering["sampleSource"] == "hashing"
    assert (filtering["method"], filtering["sampleColumn"]) == ("mad", None)
    status = datastore.artifacts.inspect(run["analysis_cell_selection"])
    assert status.operation == "filter_pipeline_cells"
    assert status.parameters["sampleSource"] == "hashing"
    assert status.input_ref("hashing") == run["hashing"]
    assert status.input_ref("input_cell_selection") == run["hashed_cell_selection"]
    # MAD bounds come from each of the three hashtags' cells.
    assert _stages(run)["filtering"]["summary"]["samples"] == 3


def test_hashing_needs_an_hto_assay(cite, hashed) -> None:
    before = _run_ids(hashed.datastore), _run_ids(cite.datastore)

    for assay in ("RNA", "HTO2"):
        with pytest.raises(
            ValueError,
            match=re.escape("hashing['assay'] must name an HTO assay, one of ['HTO']"),
        ):
            hashed.datastore.pipeline.rna(
                hashing={"assay": assay}, **SMALL_RNA_SETTINGS
            )
    # An ADT assay holds antibodies, not hashtags.
    with pytest.raises(
        ValueError,
        match=re.escape("hashing['assay'] must name an HTO assay, one of []"),
    ):
        cite.datastore.pipeline.rna(hashing={"assay": "assay2"}, **RNA_SETTINGS)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "hashing needs an assay of type 'HTO'; name it with "
            "hashing={'assay': ...}; HTO assays: []"
        ),
    ):
        cite.datastore.pipeline.rna(hashing=True, **RNA_SETTINGS)
    assert (_run_ids(hashed.datastore), _run_ids(cite.datastore)) == before


def test_hashing_settings_are_checked_before_the_run(hashed) -> None:
    datastore = hashed.datastore
    before = _run_ids(datastore)

    # positive_quantile is checked as qc.hto_demultiplexing checks it.
    for quantile, error, message in (
        (1.5, ValueError, "positive_quantile must be greater than 0 and less than 1"),
        (0, ValueError, "positive_quantile must be greater than 0 and less than 1"),
        (True, TypeError, "positive_quantile must be a real number"),
    ):
        with pytest.raises(error) as caught:
            datastore.pipeline.rna(
                hashing={"positive_quantile": quantile}, **SMALL_RNA_SETTINGS
            )
        assert str(caught.value) == message
    assert _run_ids(datastore) == before


def test_hashing_reads_the_hto_assay_that_it_names(hashed) -> None:
    plan = _resolve(
        "rna", hashed.datastore, hashing={"assay": "HTO"}, **SMALL_RNA_SETTINGS
    )

    # The store's only HTO assay is the default, so naming it plans the
    # committed run again, which records the assay that hashing read.
    recorded = dict(_config(hashed.run))
    del recorded["shutdown"]
    assert plan.config == recorded
    assert plan.config["stages"]["hashing"] == {"assay": "HTO"}
    assert list(plan.stage_order) == hashed.run.report()["run"]["stageOrder"]


def test_hashing_that_keeps_no_singlet_fails_its_stage(
    hashed, tmp_path: Path, monkeypatch
) -> None:
    datastore = _open_hashed(_copy(hashed.location, tmp_path))

    # Every cell reads as a negative, as when no hashtag stands out.
    def negatives(_store, _hashing, _cells, active: np.ndarray) -> np.ndarray:
        return np.full(len(active), "Negative", dtype=object)

    monkeypatch.setattr(rna_recipe, "hashtag_labels", negatives)
    with pytest.raises(PipelineExecutionError) as failed:
        datastore.pipeline.rna(hashing=True, **SMALL_RNA_SETTINGS)

    assert failed.value.stage == "hashing"
    assert str(failed.value.__cause__) == (
        "Hashing called no cell a singlet of one hashtag"
    )
    stopped = datastore.pipeline.open(run_id=failed.value.run_id)
    assert stopped.status == "failed"
    assert _statuses(stopped)["hashing"] == "failed"


# cells and filtering -----------------------------------------------------------


def test_cells_takes_a_column_a_selection_or_a_completed_run(
    cite, rna_of_selection
) -> None:
    rna, adt = cite.rna, cite.adt
    selection = rna["analysis_cell_selection"]

    assert _config(rna)["cells"] == {"column": "I"}
    assert _config(rna_of_selection)["cells"] == {"selection": selection.to_dict()}
    assert rna_of_selection["input_cell_selection"] == selection
    assert rna_of_selection["analysis_cell_selection"] == selection
    assert _config(adt)["cells"] == {
        "run": {"runId": rna.run_id, "label": "rna"},
        "selection": selection.to_dict(),
    }
    # The same cells and settings give the same features and reduction.
    assert (
        rna_of_selection["highly_variable_features"] == rna["highly_variable_features"]
    )
    assert rna_of_selection["pca"] == rna["pca"]


def test_cells_refuses_what_is_not_a_cell_selection(cite) -> None:
    datastore = cite.datastore
    before = _run_ids(datastore)

    with pytest.raises(
        TypeError,
        match=(
            "cells must be a boolean cell column name, a cell-selection "
            "ArtifactRef, or a completed PipelineRun"
        ),
    ):
        datastore.pipeline.rna(cells=3, **RNA_SETTINGS)
    with pytest.raises(ValueError, match="cells must be a datastore cell-selection"):
        datastore.pipeline.rna(cells=cite.rna["pca"], **RNA_SETTINGS)
    with pytest.raises(ValueError, match="cells must be a datastore cell-selection"):
        datastore.pipeline.adt(cells=cite.rna["highly_variable_features"])
    with pytest.raises(TypeError, match="cells must name a boolean metadata column"):
        datastore.pipeline.rna(cells="RNA_nCounts", **RNA_SETTINGS)
    with pytest.raises(KeyError, match="Cell selection column 'kept' was not found"):
        datastore.pipeline.rna(cells="kept", **RNA_SETTINGS)
    assert _run_ids(datastore) == before


def test_cells_refuses_a_run_of_another_store(cite, atac) -> None:
    before = _run_ids(cite.datastore)

    with pytest.raises(
        ValueError,
        match=(
            f"cells must be a pipeline run of this datastore; run "
            f"{atac.run.run_id} is not recorded in it"
        ),
    ):
        cite.datastore.pipeline.adt(cells=atac.run)
    assert _run_ids(cite.datastore) == before


def test_cells_refuses_a_run_that_did_not_complete(cite, tmp_path: Path) -> None:
    datastore = _open_cite(_copy(cite.location, tmp_path))
    token = CancellationToken()

    def cancel_once_started(event: PipelineEvent) -> None:
        if event.kind == "stage_started":
            token.cancel("stopped by the test")

    with pytest.raises(PipelineCancelled) as cancelled:
        datastore.pipeline.rna(
            callback=cancel_once_started, cancel=token, **RNA_SETTINGS
        )
    stopped = datastore.pipeline.open(run_id=cancelled.value.run_id)
    assert stopped.status == "interrupted"

    with pytest.raises(
        ValueError,
        match=(
            f"cells must be a completed pipeline run; run {stopped.run_id} is "
            "'interrupted'"
        ),
    ):
        datastore.pipeline.adt(cells=stopped)
    with pytest.raises(
        ValueError,
        match=f"Run {stopped.run_id} is 'interrupted'; integrate completed runs only",
    ):
        datastore.pipeline.multimodal([datastore.pipeline.open(label="rna"), stopped])


def test_filtering_none_filters_only_cells_that_a_column_names(
    cite, rna_of_selection, adt_own
) -> None:
    rna = cite.rna

    filtering = _config(rna)["stages"]["filtering"]
    assert filtering["enabled"] is True
    assert filtering["attrs"] == [
        "RNA_nCounts",
        "RNA_nFeatures",
        "RNA_percentMito",
        "RNA_percentRibo",
    ]
    summary = _stages(rna)["filtering"]["summary"]
    assert summary["removedCells"] > 0
    assert summary["inputCells"] == summary["cells"] + summary["removedCells"]
    given = _mask(cite.datastore, rna["input_cell_selection"])
    kept = _mask(cite.datastore, rna["analysis_cell_selection"])
    assert not (kept & ~given).any()
    assert (int(given.sum()), int(kept.sum())) == (
        summary["inputCells"],
        summary["cells"],
    )
    # A given selection or run's cells are analyzed as they are, and an ADT
    # run does not filter the cells that a column names either.
    for run in (rna_of_selection, cite.adt, adt_own):
        assert _config(run)["stages"]["filtering"] == {"enabled": False}
        assert _statuses(run)["filtering"] == "skipped"
        assert run["analysis_cell_selection"] == run["input_cell_selection"]
    np.testing.assert_array_equal(
        adt_own.cells.fetch_all("I"), cite.datastore.cells.fetch_all("I")
    )


def test_filtering_none_filters_the_cells_that_qc_stages_keep(
    cite, droplets, hashed
) -> None:
    raw = droplets.single["input_cell_selection"]

    def filtering(plan: RecipePlan) -> dict[str, Any]:
        return plan.config["stages"]["filtering"]

    def filters(plan: RecipePlan) -> bool:
        return {stage.name: stage.action for stage in plan.stages}[
            "filtering"
        ] is not None

    # A run that calls cells filters the cells that it calls, even from a
    # given selection: raw droplets are not cells that passed QC.
    calling = _resolve(
        "rna",
        droplets.datastore,
        cells=raw,
        cell_calling=CELL_CALLING,
        **SMALL_RNA_SETTINGS,
    )
    assert filtering(calling) == _config(droplets.filtered)["stages"]["filtering"]
    assert filtering(calling)["enabled"] is True
    assert filters(calling)
    # So does a run that keeps hashtag singlets, within each hashtag.
    hashing = _resolve(
        "rna",
        hashed.datastore,
        cells=hashed.run["input_cell_selection"],
        hashing=True,
        **SMALL_RNA_SETTINGS,
    )
    assert filtering(hashing) == _config(hashed.run)["stages"]["filtering"]
    assert filtering(hashing)["sampleSource"] == "hashing"
    assert filters(hashing)
    # A run of the same selection without them analyzes its cells as they are.
    plain = _resolve("rna", droplets.datastore, cells=raw, **SMALL_RNA_SETTINGS)
    assert filtering(plain) == {"enabled": False}
    assert not filters(plain)
    # An ADT run filters only when asked, whatever its cells.
    for cells in ("I", cite.rna["analysis_cell_selection"], cite.rna):
        plan = _resolve("adt", cite.datastore, cells=cells)
        assert filtering(plan) == {"enabled": False}
        assert not filters(plan)


# Snapshots and summaries -------------------------------------------------------


def test_cell_snapshots_freeze_each_group_of_columns_apart() -> None:
    mad = {"enabled": True, "method": "mad", "attrs": ["q1", "q2"]}

    # The cell snapshot holds the names and the columns that filtering reads,
    # its MAD sample column among them.
    assert cell_snapshots({"enabled": False}) == (("names",), {})
    assert cell_snapshots({**mad, "sampleColumn": None}) == (("names", "q1", "q2"), {})
    filtering = {**mad, "sampleColumn": "lane"}
    assert cell_snapshots(filtering) == (("names", "q1", "q2", "lane"), {})
    # Each other group is frozen apart, under the key of the stage that
    # reads it, and only when the run reads it.
    assert cell_snapshots(
        filtering,
        harmony_columns=("batch",),
        snapshot_columns=("donor", "visit"),
        capture_column="capture",
        doublet_samples=("batch",),
    ) == (
        ("names", "q1", "q2", "lane"),
        {
            "capture_snapshot": ("capture",),
            "harmony_snapshot": ("batch",),
            "doublet_sample_snapshot": ("batch",),
            "column_snapshot": ("donor", "visit"),
        },
    )
    assert cell_snapshots(filtering, snapshot_columns=("lane",)) == (
        ("names", "q1", "q2", "lane"),
        {"column_snapshot": ("lane",)},
    )


def test_harmony_and_snapshot_columns_leave_the_filtered_cells_alone(
    cite, full, snapshot
) -> None:
    rna = cite.rna
    datastore = cite.datastore
    shared = (
        "input_cell_selection",
        "analysis_cell_selection",
        "feature_universe",
        "highly_variable_features",
        "normalized",
        "pca",
    )

    # Harmony's batches and the snapshot columns are frozen apart from the
    # columns that filtering reads, so both runs filter to the committed
    # run's cells and reuse its features, normalization, and PCA.
    for run in (full, snapshot):
        assert {key: run[key] for key in shared} == {key: rna[key] for key in shared}
    # The snapshot columns change nothing that the run computes.
    assert dict(snapshot) == dict(rna)
    # Harmony corrects that PCA over the batches of its own snapshot.
    harmony = datastore.artifacts.inspect(full["harmony"])
    assert harmony.input_ref("reduction") == rna["pca"]
    assert (
        harmony.input_ref("batch_snapshot") == _snapshot_refs(full)["harmony_snapshot"]
    )
    assert (
        datastore.artifacts.inspect(full["neighbors"]).input_ref("coordinates")
        == full["harmony"]
    )
    # The run's cell view reads a snapshot column from its own snapshot.
    selected = snapshot.cells.fetch_all("I")
    assert snapshot.cells.columns == (
        "I",
        "ids",
        "names",
        "donor",
        "leiden_1.0",
        "clusters",
    )
    np.testing.assert_array_equal(
        snapshot.cells.fetch("donor"), datastore.cells.fetch_all("donor")[selected]
    )
    fields = {
        field["key"]: field
        for field in snapshot.report()["run"]["fields"]
        if field["axis"] == "cells"
    }
    assert (
        ArtifactRef.from_dict(fields["donor"]["artifact"])
        == _snapshot_refs(snapshot)["column_snapshot"]
    )


def test_input_snapshot_freezes_only_the_column_groups_that_a_run_reads(
    cite, full, snapshot, joint, droplets
) -> None:
    datastore = cite.datastore
    base = {
        "input_cell_selection": None,
        "cell_snapshot": ["names", *RNA_QC_COLUMNS],
        "feature_snapshot": ["names"],
        "feature_universe": None,
    }

    assert _snapshot_columns(datastore, cite.rna) == base
    # Harmony's batches also split doublet scoring, which freezes them again.
    assert _snapshot_columns(datastore, full) == {
        **base,
        "harmony_snapshot": ["batch"],
        "doublet_sample_snapshot": ["batch"],
    }
    assert _snapshot_columns(datastore, snapshot) == {
        **base,
        "column_snapshot": ["donor"],
    }
    # Calling each capture apart freezes the capture labels.
    unfiltered = {**base, "cell_snapshot": ["names"]}
    assert _snapshot_columns(droplets.datastore, droplets.single) == unfiltered
    assert _snapshot_columns(droplets.datastore, droplets.per_capture) == {
        **unfiltered,
        "capture_snapshot": ["capture"],
    }
    assert _snapshot_columns(droplets.datastore, droplets.filtered) == {
        **base,
        "cell_snapshot": ["names", "RNA_nCounts", "RNA_nFeatures"],
        "capture_snapshot": ["capture"],
        "column_snapshot": ["donor"],
    }
    # A joint run freezes the names, its snapshot columns, and each
    # modality's features; it filters nothing.
    modalities = {
        "feature_snapshot_RNA": ["names"],
        "feature_universe_RNA": None,
        "feature_snapshot_assay2": ["names"],
        "feature_universe_assay2": None,
    }
    assert _snapshot_columns(datastore, joint.wnn) == {
        "cell_snapshot": ["names"],
        **modalities,
    }
    assert _snapshot_columns(datastore, joint.snn) == {
        "cell_snapshot": ["names"],
        "column_snapshot": ["donor"],
        **modalities,
    }
    # Snapshots are the stage's outputs, not the run's.
    for run in (full, snapshot, joint.snn, droplets.filtered):
        snapshots = {
            key
            for key, ref in _snapshot_refs(run).items()
            if ref.kind == "metadata_snapshot"
        }
        assert len(snapshots) >= 3
        assert not snapshots & set(run)
        with pytest.raises(KeyError):
            run["cell_snapshot"]


def test_each_stage_summary_names_the_values_that_it_records(cite, full) -> None:
    datastore = cite.datastore
    summaries = {name: stage["summary"] for name, stage in _stages(full).items()}
    cells = int(full.cells.fetch_all("I").sum())

    # A summary is computed from attributes and arrays whose names could
    # change; a renamed one drops its values silently, so each key is pinned.
    assert {
        name: None if summary is None else set(summary)
        for name, summary in summaries.items()
    } == {
        "input_snapshot": {"input_cell_selection", "feature_universe"},
        "cell_calling": None,
        "hashing": None,
        "filtering": {"cells", "inputCells", "removedCells", "samples"},
        "cell_cycle": {"phases"},
        "highly_variable_features": {"features"},
        "normalization": {"cells", "features"},
        "pca": {"cells", "dims", "varianceRatio", "varianceExplained"},
        "harmony": {"rounds", "converged"},
        "ann_index": None,
        "neighbors": {"cells", "k", "selfHitRate"},
        "connectivity": {"cells", "k"},
        "embedding_initialization": {"centroids", "dims"},
        "umap": {"cells", "dims"},
        "leiden_1.0": {"clusters", "largestCluster", "smallestCluster"},
        "paris": {"clusters"},
        "cluster_selection": {"selected", "scores"},
        "membership_strength": {"mean", "belowHalf"},
        "tsne": None,
        "doublets": {"predictedDoublets", "rate", "samples"},
        "markers": None,
    }
    assert summaries["input_snapshot"] == {
        "input_cell_selection": {"cells": datastore.cells.N},
        "feature_universe": {"features": datastore.RNA.feats.N},
    }
    assert summaries["highly_variable_features"] == {"features": 200}
    assert summaries["normalization"] == {"cells": cells, "features": 200}
    harmony = summaries["harmony"]
    assert type(harmony["rounds"]) is int and harmony["rounds"] >= 1
    assert type(harmony["converged"]) is bool
    for stage in ("neighbors", "connectivity"):
        assert (summaries[stage]["cells"], summaries[stage]["k"]) == (cells, 7)
    assert 0 <= summaries["neighbors"]["selfHitRate"] <= 100
    # The embedding starts from the 5 corrected dimensions.
    assert summaries["embedding_initialization"]["dims"] == 5
    assert summaries["umap"] == {"cells": cells, "dims": 2}
    assert summaries["paris"] == {"clusters": len(np.unique(full.cells.fetch("paris")))}
    phases, counts = np.unique(
        full.cells.fetch("cell_cycle_phase").astype(str), return_counts=True
    )
    assert summaries["cell_cycle"] == {
        "phases": dict(zip(phases.tolist(), counts.tolist(), strict=True))
    }
    strength = full.cells.fetch("membership_strength")
    assert summaries["membership_strength"] == {
        "mean": pytest.approx(float(strength.mean()), abs=1e-4),
        "belowHalf": int(np.count_nonzero(strength < 0.5)),
    }
    doublets = np.asarray(full.cells.fetch("predicted_doublet"), dtype=bool)
    summary = summaries["doublets"]
    assert summary["predictedDoublets"] == int(doublets.sum())
    assert summary["rate"] == pytest.approx(doublets.mean(), abs=1e-4)
    # Doublets are scored within each Harmony batch.
    assert [sample["labels"] for sample in summary["samples"]] == [
        {"batch": "a"},
        {"batch": "b"},
    ]
    assert sum(sample["cells"] for sample in summary["samples"]) == cells
    assert (
        sum(sample["predictedDoublets"] for sample in summary["samples"])
        == summary["predictedDoublets"]
    )


# Assays, reuse, and earlier records ------------------------------------------


def test_recipes_take_the_default_or_only_assay_of_their_modality(cite, atac) -> None:
    # The default assay is RNA, and assay2 is the store's only ADT assay.
    assert cite.rna.assay == "RNA"
    assert cite.adt.assay == "assay2"
    assert atac.run.assay == "ATAC"


def test_a_recipe_prefers_the_default_assay_and_never_guesses_among_several(
    tmp_path: Path,
) -> None:
    location = tmp_path / "data.zarr"
    write_count_store(
        str(location),
        {
            "ADT": _counts(60, 8, seed=1),
            "RNA": _counts(60, 100, seed=2),
            "URNA": _counts(60, 100, seed=3),
        },
        "uint16",
    )
    datastore = DataStore(str(location), default_assay="ADT", min_features_per_cell=0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The store has several RNA assays, ['RNA', 'URNA']; choose one with "
            "pipeline.rna(assay=...)"
        ),
    ):
        datastore.pipeline.rna()
    assert datastore.pipeline.list_runs() == ()

    datastore = DataStore(str(location), default_assay="URNA", min_features_per_cell=0)
    run = datastore.pipeline.rna(filtering=False, **SMALL_RNA_SETTINGS)
    assert run.assay == "URNA"


def test_an_identical_second_call_creates_no_artifacts(cite, tmp_path: Path) -> None:
    datastore = _open_cite(_copy(cite.location, tmp_path))

    again = datastore.pipeline.rna(**RNA_SETTINGS)

    summary = again.report()["summary"]
    assert summary["createdArtifacts"] == []
    assert summary["reusedArtifacts"]
    assert again.run_id != cite.rna.run_id
    assert dict(again) == dict(cite.rna)


def test_a_basic_rna_analysis_record_still_opens_lists_and_reports() -> None:
    root = _root()
    written = _completed_run(root)
    # Stage records written before stage summaries and warnings lack both.
    stage = root[pipeline_stage_path(written.run_id, 0)]
    attrs = dict(stage.attrs)
    del attrs["summary"], attrs["warnings"]
    stage.attrs.put(attrs)
    pipeline = PipelineAccessor(_Owner(root))

    run = pipeline.open(label="baseline")
    assert (run.run_id, run.recipe, run.status) == (
        written.run_id,
        "basic_rna_analysis",
        "completed",
    )
    assert pipeline.open(run_id=written.run_id).label == "baseline"
    assert [listed.run_id for listed in pipeline.list_runs()] == [written.run_id]
    assert [listed.recipe for listed in pipeline.list_runs(status="completed")] == [
        "basic_rna_analysis"
    ]
    report = run.report()
    assert report["run"]["recipe"] == "basic_rna_analysis"
    assert report["run"]["config"] == {"pcaDims": 21}
    assert [
        (stage["stage"], stage["status"], stage["summary"], stage["warnings"])
        for stage in report["stages"]
    ] == [("input_snapshot", "completed", None, [])]
    assert report["summary"]["warnings"] == []
    assert report["summary"]["signalProtection"] is None
    markdown = run.report(format="markdown")
    assert "- Recipe: `basic_rna_analysis`" in markdown
    assert "| 0 | input_snapshot | completed |" in markdown
    assert "## Diagnostics" not in markdown
    np.testing.assert_array_equal(run.cells.fetch("clusters"), [0, 2])
