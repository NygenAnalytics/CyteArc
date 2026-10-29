import inspect
import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from cytearc import DataStore, PipelineExecutionError
from cytearc.datastore._pipeline.filtering import FILTERING_SETTINGS
from cytearc.datastore._pipeline.recipes.adt import resolve_adt
from cytearc.datastore._pipeline.recipes.rna import resolve_rna
from cytearc.datastore._pipeline.settings import (
    DEFAULT_LEIDEN_PARTITIONS,
    STAGE_SETTINGS,
    parameter_value,
)
from cytearc.datastore.namespaces import (
    EmbeddingsAccessor,
    QcAccessor,
    ReductionAccessor,
)
from cytearc.datastore.pipeline_accessor import PipelineAccessor
from cytearc.features.gene_reference import species_registry
from cytearc.storage.artifacts import ArtifactRef, provenance_hash
from cytearc.storage.feature_selection import read_feature_selection_indices
from cytearc.utils.logging import logger


@contextmanager
def _captured_warnings(level: str = "WARNING") -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level=level,
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _fast(**keywords: Any) -> dict[str, Any]:
    """Return the keywords of a quick RNA run, with ``keywords`` in place."""
    settings: dict[str, Any] = {
        "filtering": False,
        "cell_cycle": False,
        "hvg": {"top_n": 50},
        "pca": {"dims": 3},
        "neighbors": {"k": 3},
        "umap": False,
        "leiden": False,
        "doublets": False,
        "markers": False,
    }
    settings.update(keywords)
    return settings


def _keywords(method: Any) -> dict[str, Any]:
    """Return each keyword of ``method`` with its default."""
    return {
        name: parameter.default
        for name, parameter in inspect.signature(method).parameters.items()
    }


def _recipe_defaults(recipe: str) -> dict[str, Any]:
    """Return the keywords that ``ds.pipeline.<recipe>`` resolves, at its defaults.

    The resolvers take every recipe keyword explicitly. The callback and the
    cancellation token belong to the run, not to its resolution.
    """
    keywords = _keywords(getattr(PipelineAccessor, recipe))
    for name in ("self", "callback", "cancel"):
        keywords.pop(name)
    return keywords


def _resolve(store: DataStore, **keywords: Any) -> Any:
    """Resolve an RNA recipe call as ``ds.pipeline.rna`` does, writing nothing."""
    return resolve_rna(store, **{**_recipe_defaults("rna"), **keywords})


def _written(store: DataStore) -> tuple[list[str], list[Any], list[Any]]:
    """Return what a run writes: run records, and datastore and RNA artifacts."""
    return (
        [run.run_id for run in store.pipeline.list_runs()],
        store.artifacts.list(scope="datastore"),
        store.artifacts.list(from_assay="RNA"),
    )


def _stage_statuses(run: Any) -> dict[str, str]:
    return {stage["stage"]: stage["status"] for stage in run.report()["stages"]}


# The recipe keywords that are not stages, and the RNA stage keywords in order.
_RECIPE_KEYWORDS = frozenset({"assay", "label", "cells", "snapshot_columns", "species"})
_RNA_STAGES = tuple(
    name for name in _recipe_defaults("rna") if name not in _RECIPE_KEYWORDS
)
# Stages that only switch on or off, without settings.
_SWITCHES = ("membership_strength", "markers")
# Stages that always run, so they take settings or None, never a switch.
_ALWAYS_RUN = (
    "hvg",
    "normalization",
    "pca",
    "ann_index",
    "neighbors",
    "connectivity",
    "embedding_initialization",
)
# The QC columns that preparation wrote for the fixture's RNA assay.
_QC_COLUMNS = ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo"]


@pytest.fixture(scope="module")
def settings_base(datastore_zarr_root: str, tmp_path_factory) -> str:
    """A PBMC store holding the artifacts of one quick RNA run."""
    location = tmp_path_factory.mktemp("settings_base") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    DataStore(str(location), default_assay="RNA").pipeline.rna(**_fast())
    return str(location)


@pytest.fixture
def settings_store(settings_base: str, tmp_path) -> DataStore:
    """A writable copy of the base; runs reuse the stages they share with it."""
    location = tmp_path / "data.zarr"
    shutil.copytree(settings_base, location)
    return DataStore(str(location), default_assay="RNA")


@pytest.fixture(scope="module")
def base_view(settings_base: str) -> DataStore:
    """The base store opened read-only, for calls resolved against its run."""
    return DataStore(settings_base, default_assay="RNA", zarr_mode="r")


@pytest.fixture(scope="module")
def readonly_store(datastore_zarr_root: str) -> DataStore:
    """The PBMC store opened read-only; resolving a recipe writes nothing."""
    return DataStore(datastore_zarr_root, default_assay="RNA", zarr_mode="r")


@pytest.fixture(scope="module")
def rejection_store(datastore_zarr_root: str, tmp_path_factory) -> DataStore:
    """One writable store for calls that must be refused before any write."""
    location = tmp_path_factory.mktemp("settings_rejections") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    return DataStore(str(location), default_assay="RNA")


@pytest.fixture(scope="module")
def adt_store(datastore_zarr_root: str, tmp_path_factory) -> DataStore:
    """The CITE-seq store with its antibody assay recorded as an ADT assay."""
    location = tmp_path_factory.mktemp("settings_adt") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    return DataStore(str(location), default_assay="RNA", assay_types={"assay2": "ADT"})


# Settings that configure the recipe, not the stage's method: the Leiden
# candidates, the HTO assay that hashing reads, the capture column whose
# captures cell calling calls apart, the antibodies the panel leaves out, and
# the Cell Ranger call file that guide calling imports in place of calling.
_RECIPE_SETTINGS: dict[str, frozenset[str]] = {
    "leiden": frozenset({"partitions", "selected"}),
    "hashing": frozenset({"assay"}),
    "cell_calling": frozenset({"sample_column"}),
    "panel": frozenset({"exclude"}),
    "guide_calling": frozenset({"path", "barcode_prefix"}),
}
# Method keywords without a default, which the settings must give: Harmony's
# batch columns and the controls of perturbation labels. The doublet
# sample_column is a method keyword with the method's default, except that a
# run with Harmony omitting it scores doublets within each Harmony batch.
_REQUIRED_SETTINGS = {"harmony": {"batch_columns"}, "labels": {"controls"}}
# Settings whose default, an infinity, is no JSON value; a run omits them.
_INFINITE_DEFAULTS = {"hvg": {"min_var", "max_var", "min_mean", "max_mean"}}
# Settings whose default is a tuple, recorded as the equal JSON list, which
# the method takes too.
_TUPLE_DEFAULTS = {"associations": {"n_resamples"}}
# The public method whose keywords each stage's settings are, then the method
# that the stage passes them to where that is another one.
_STAGE_METHODS: dict[str, tuple[str, ...]] = {
    "cell_calling": ("_qc_call_cells",),
    "hashing": ("_qc_hto_demultiplexing",),
    "cell_cycle": ("_scores_cell_cycle", "_run_cell_cycle_scoring_artifact"),
    "hvg": ("_features_hvgs", "_select_hvgs_artifact"),
    "prevalent_peaks": ("_features_prevalent_peaks",),
    "normalization": ("_features_normalize",),
    "pca": ("_reduction_pca",),
    "lsi": ("_reduction_lsi",),
    "harmony": ("_reduction_harmony", "_run_harmony_artifact"),
    "ann_index": ("_graph_ann_index",),
    "neighbors": ("_graph_neighbors",),
    "connectivity": ("_graph_connectivity",),
    "integration": ("_integration_modalities",),
    "embedding_initialization": ("_embeddings_initialization",),
    "umap": ("_embeddings_umap", "_run_umap_artifact"),
    "tsne": ("_embeddings_tsne",),
    "leiden": ("_clusters_leiden", "_run_leiden_artifact"),
    "paris": ("_clusters_paris", "_run_paris_artifact"),
    "doublets": ("_qc_doublets", "_run_doublet_detection_artifact"),
    "guide_calling": ("_perturbation_call_guides",),
    "labels": ("_perturbation_labels",),
    "efficiency": ("_perturbation_efficiency",),
    "edistance": ("_perturbation_edistance",),
    "signature": ("_perturbation_signature",),
    "mixscape": ("_perturbation_mixscape",),
    "associations": ("_perturbation_associations",),
    "profiles": ("_perturbation_profiles",),
    "clusters": ("_perturbation_cluster",),
}
# The filtering settings of each method: automatic filtering takes the
# keywords of qc.auto_filter and manual filtering those of qc.filter.
_AUTOMATIC_FILTERING = frozenset(
    {
        "method",
        "attrs",
        "min_p",
        "max_p",
        "sample_column",
        "n_mads",
        "min_cells_per_sample",
    }
)
_MANUAL_FILTERING = frozenset({"attrs", "lows", "highs", "keep_bounds"})


def _is_json_value(value: Any) -> bool:
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        return False


def test_stage_settings_are_keywords_of_the_methods_they_configure(readonly_store):
    for stage, keys in _RECIPE_SETTINGS.items():
        assert keys <= STAGE_SETTINGS[stage], stage
    settings = {
        stage: keys - _RECIPE_SETTINGS.get(stage, frozenset())
        for stage, keys in STAGE_SETTINGS.items()
    }
    # Membership strength and markers are switches, and the panel's only
    # setting configures the recipe.
    assert {stage for stage, keys in settings.items() if keys} == set(_STAGE_METHODS)
    for stage, names in _STAGE_METHODS.items():
        public, *called = [_keywords(getattr(DataStore, name)) for name in names]
        for key in settings[stage]:
            assert all(key in keywords for keywords in (public, *called)), (stage, key)
            default = public[key]
            if key in _REQUIRED_SETTINGS.get(stage, set()):
                assert default is inspect.Parameter.empty, (stage, key)
                continue
            # Omitting a setting means the public method's default.
            assert all(keywords[key] == default for keywords in called), (stage, key)
            if key in _INFINITE_DEFAULTS.get(stage, set()):
                assert default in (-np.inf, np.inf), (stage, key, default)
            elif key in _TUPLE_DEFAULTS.get(stage, set()):
                assert isinstance(default, tuple), (stage, key, default)
                assert _is_json_value(list(default)), (stage, key, default)
            else:
                assert _is_json_value(default), (stage, key, default)

    assert FILTERING_SETTINGS == _AUTOMATIC_FILTERING | _MANUAL_FILTERING
    automatic = _keywords(QcAccessor.auto_filter)
    assert _AUTOMATIC_FILTERING <= set(automatic)
    assert _MANUAL_FILTERING <= set(_keywords(QcAccessor.filter))
    # Automatic filtering is the MAD method of qc.auto_filter, at its defaults.
    filtering = _resolve(readonly_store, filtering=True).config["stages"]["filtering"]
    assert filtering["method"] == automatic["method"] == "mad"
    assert (
        filtering["minP"],
        filtering["maxP"],
        filtering["nMads"],
        filtering["minCellsPerSample"],
    ) == tuple(
        automatic[name] for name in ("min_p", "max_p", "n_mads", "min_cells_per_sample")
    )


def test_a_default_call_records_each_stage_at_its_method_defaults(readonly_store):
    config = _resolve(readonly_store).config
    automatic = _keywords(QcAccessor.auto_filter)

    assert config["cells"] == {"column": "I"}
    assert config["snapshotColumns"] == []
    assert config["species"] is None
    assert config["doubletSampleColumns"] == []
    # The run records one entry per stage keyword: the settings it was given,
    # none for a stage at its method's defaults, or False for a stage that is
    # off. PCA records its dimensions, which decide whether it runs.
    assert set(config["stages"]) == set(_RNA_STAGES)
    assert config["stages"] == {
        "cell_calling": False,
        "hashing": False,
        "filtering": {
            "enabled": True,
            "method": "mad",
            "attrs": _QC_COLUMNS,
            "minP": automatic["min_p"],
            "maxP": automatic["max_p"],
            "sampleColumn": None,
            "nMads": automatic["n_mads"],
            "minCellsPerSample": automatic["min_cells_per_sample"],
        },
        "cell_cycle": {},
        "hvg": {},
        "normalization": {},
        "pca": {"dims": _keywords(ReductionAccessor.pca)["dims"]},
        "harmony": False,
        "ann_index": {},
        "neighbors": {},
        "connectivity": {},
        "embedding_initialization": {},
        "umap": {},
        "tsne": False,
        "leiden": {"partitions": list(DEFAULT_LEIDEN_PARTITIONS), "selected": None},
        "paris": False,
        "membership_strength": False,
        "doublets": {},
        "markers": True,
    }


def test_stage_settings_are_forwarded_and_recorded(settings_store):
    store = settings_store

    run = store.pipeline.rna(
        **_fast(
            hvg={"top_n": 50, "lowess_frac": 0.2},
            pca={"dims": 3, "feat_scaling": False},
            connectivity={"bandwidth": 1.2},
            leiden={"partitions": [1.0], "random_seed": 7},
        ),
        species="homo_sapiens",
    )

    assert run.status == "completed"
    config = run.report()["run"]["config"]
    assert config["species"] == "homo_sapiens"
    assert config["stages"] == {
        "cell_calling": False,
        "hashing": False,
        "filtering": {"enabled": False},
        "cell_cycle": False,
        "hvg": {"top_n": 50, "lowess_frac": 0.2},
        "normalization": {},
        "pca": {"dims": 3, "feat_scaling": False},
        "harmony": False,
        "ann_index": {},
        "neighbors": {"k": 3},
        "connectivity": {"bandwidth": 1.2},
        "embedding_initialization": {},
        "umap": False,
        "tsne": False,
        "leiden": {"partitions": [1.0], "selected": None, "random_seed": 7},
        "paris": False,
        "membership_strength": False,
        "doublets": False,
        "markers": False,
    }
    hvg_parameters = store.artifacts.inspect(run["highly_variable_features"]).parameters
    assert (hvg_parameters["top_n"], hvg_parameters["lowess_frac"]) == (50, 0.2)
    # The stage methods called with the same settings reuse the run's artifacts.
    assert (
        store.reduction.pca(run["normalized"], dims=3, feat_scaling=False) == run["pca"]
    )
    assert (
        store.graph.connectivity(run["neighbors"], bandwidth=1.2)
        == run["connectivity_map"]
    )
    assert store.graph.connectivity(run["neighbors"]) != run["connectivity_map"]
    assert (
        store.clusters.leiden(run["connectivity_map"], resolution=1.0, random_seed=7)
        == run["leiden_1.0"]
    )


def test_omitted_settings_use_the_stage_method_defaults(settings_store):
    store = settings_store
    (base,) = store.pipeline.list_runs()
    cells = base["analysis_cell_selection"]

    # The quick run sets only top_n, dims, and k, so every other setting of
    # its stages is the method's default, and the methods reuse its artifacts.
    assert (
        store.features.hvgs(cells, top_n=50, show_plot=False)
        == base["highly_variable_features"]
    )
    assert (
        store.features.normalize(cells, base["highly_variable_features"])
        == base["normalized"]
    )
    assert store.reduction.pca(base["normalized"], dims=3) == base["pca"]
    assert store.graph.ann_index(base["pca"]) == base["ann_index"]
    assert (
        store.graph.neighbors(base["ann_index"], coordinates=base["pca"], k=3)
        == base["neighbors"]
    )
    assert store.graph.connectivity(base["neighbors"]) == base["connectivity_map"]

    run = store.pipeline.rna(
        **_fast(umap={"n_epochs": 10}, leiden={"partitions": [1.0]})
    )

    assert (
        store.embeddings.initialization(run["pca"]) == run["embedding_initialization"]
    )
    assert (
        store.embeddings.umap(
            run["connectivity_map"], run["embedding_initialization"], n_epochs=10
        )
        == run["umap"]
    )
    assert (
        store.clusters.leiden(run["connectivity_map"], resolution=1.0)
        == run["leiden_1.0"]
    )
    # The default UMAP dimensions name the run's coordinate fields.
    umap_dims = _keywords(EmbeddingsAccessor.umap)["umap_dims"]
    assert [column for column in run.cells.columns if column.startswith("umap_")] == [
        f"umap_{index}" for index in range(1, umap_dims + 1)
    ]
    # The stages that the runs share reuse the quick run's artifacts.
    assert all(run[key] == base[key] for key in base)


# Pin CyteArc HVG parameter identities for max_cells=None (genes detected in
# more than 99.5% of the selected cells excluded) and max_cells=np.inf (no
# cutoff) on this fixture with top_n=50, the default min_cells (1% of the
# selected cells, from 3 to 20), and the default blacklist of 1.3.0.
_AUTO_DIGEST = "57929bd45c563d7206838229d44deae834440b1f9b89d5244817d735215d6014"
_NO_CUTOFF_DIGEST = "efde872152049c8f582f39ad36802008276e91f0cf9a6489061be1cc13213133"


def _parameter_digest(store: DataStore, ref: ArtifactRef) -> str:
    status = store.artifacts.inspect(ref)
    return provenance_hash(
        {"operation": status.operation, "parameters": status.parameters, "inputs": {}}
    )


def test_hvg_cutoffs_record_cytearc_provenance(settings_store):
    store = settings_store
    (base,) = store.pipeline.list_runs()

    # The base run omits max_cells, so its default None sets the automatic cutoff.
    assert _parameter_digest(store, base["highly_variable_features"]) == _AUTO_DIGEST
    uncut = store.features.hvgs(
        base["analysis_cell_selection"], top_n=50, max_cells=np.inf, show_plot=False
    )
    assert _parameter_digest(store, uncut) == _NO_CUTOFF_DIGEST


def test_setting_values_are_recorded_as_json():
    assert type(parameter_value(np.int64(3), "x")) is int
    assert parameter_value(np.float32(0.5), "x") == 0.5
    assert parameter_value((1, 2), "x") == [1, 2]
    assert parameter_value(np.array([1.0, 2.0]), "x") == [1.0, 2.0]
    assert parameter_value({"a": [np.bool_(True)]}, "x") == {"a": [True]}
    with pytest.raises(TypeError, match="keys must be strings"):
        parameter_value({1: 2}, "x")
    for value in (object(), b"raw"):
        with pytest.raises(TypeError, match="must be a number"):
            parameter_value(value, "x")
    with pytest.raises(ValueError, match="must be finite"):
        parameter_value([1.0, float("nan")], "x")


def test_checked_settings_are_recorded_as_given(readonly_store):
    stages = _resolve(
        readonly_store,
        hvg={
            "min_cells": np.int64(3),
            "n_bins": 50,
            "lowess_frac": 1,
            "keep_bounds": np.bool_(True),
            "bin_strategy": "fixed",
        },
        tsne={
            "box_h": 1,
            "max_iter": np.int64(800),
            "early_iter": 0,
            "symmetric_graph": np.bool_(True),
        },
    ).config["stages"]

    # Checked settings keep their JSON values, so 1 stays an integer.
    assert stages["hvg"] == {
        "min_cells": 3,
        "n_bins": 50,
        "lowess_frac": 1,
        "keep_bounds": True,
        "bin_strategy": "fixed",
    }
    assert type(stages["hvg"]["lowess_frac"]) is int
    assert stages["tsne"] == {
        "box_h": 1,
        "max_iter": 800,
        "early_iter": 0,
        "symmetric_graph": True,
    }
    # An omitted setting is checked as the method's default, and a switch
    # needs no settings.
    assert _resolve(readonly_store, hvg={}).config["stages"]["hvg"] == {}
    assert _resolve(readonly_store, tsne={}).config["stages"]["tsne"] == {}
    assert _resolve(readonly_store, tsne=True).config["stages"]["tsne"] == {}


def test_settings_resolve_into_the_recorded_stages(readonly_store):
    plan = _resolve(
        readonly_store,
        cell_cycle={"n_bins": 30},
        hvg={"top_n": 500, "keep_bounds": True},
        pca={"dims": 10},
        neighbors={"k": 7, "batch_size": 1000},
        umap={"n_epochs": 400, "umap_dims": 3},
        leiden={"partitions": [0.5, 1.0], "selected": 0.5, "random_seed": 1},
        paris={"n_clusters": 5},
        doublets={"save_k": 3},
        tsne=True,
        membership_strength=True,
    )

    stages = plan.config["stages"]
    assert stages["cell_cycle"] == {"n_bins": 30}
    assert stages["hvg"] == {"top_n": 500, "keep_bounds": True}
    assert stages["pca"] == {"dims": 10}
    assert stages["neighbors"] == {"k": 7, "batch_size": 1000}
    assert stages["umap"] == {"n_epochs": 400, "umap_dims": 3}
    assert stages["leiden"] == {
        "partitions": [0.5, 1.0],
        "selected": 0.5,
        "random_seed": 1,
    }
    assert stages["paris"] == {"n_clusters": 5}
    assert stages["doublets"] == {"save_k": 3}
    assert (stages["tsne"], stages["membership_strength"]) == ({}, True)
    assert plan.stage_order == (
        "input_snapshot",
        "cell_calling",
        "hashing",
        "filtering",
        "cell_cycle",
        "highly_variable_features",
        "normalization",
        "pca",
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
        "doublets",
        "markers",
    )
    # A selected resolution is the saved clustering, so cluster selection is
    # skipped along with the stages that are off.
    skipped = {stage.name for stage in plan.stages if stage.action is None}
    assert skipped == {"cell_calling", "hashing", "harmony", "cluster_selection"}


@pytest.mark.parametrize(
    "stage", [stage for stage in _RNA_STAGES if stage not in _SWITCHES]
)
def test_unknown_settings_name_the_allowed_ones(readonly_store, stage):
    allowed = FILTERING_SETTINGS if stage == "filtering" else STAGE_SETTINGS[stage]

    with pytest.raises(ValueError) as caught:
        _resolve(readonly_store, **{stage: {"unknown": 1}})

    assert str(caught.value) == (
        f"Unknown {stage} settings ['unknown']; allowed: {sorted(allowed)!r}"
    )


@pytest.mark.parametrize("stage", _ALWAYS_RUN)
def test_stages_that_always_run_refuse_a_switch(readonly_store, stage):
    for value in (True, False):
        with pytest.raises(
            TypeError,
            match=f"^{stage} must be a mapping because the stage always runs$",
        ):
            _resolve(readonly_store, **{stage: value})
    with pytest.raises(TypeError, match=f"^{stage} must be a mapping or None$"):
        _resolve(readonly_store, **{stage: 1})


@pytest.mark.parametrize(
    ("keywords", "error", "match"),
    [
        ({"umap": {"epochs": 400}}, ValueError, r"Unknown umap settings \['epochs'\]"),
        # t-SNE has one single-threaded backend and no thread setting.
        ({"tsne": {"parallel": True}}, ValueError, r"Unknown tsne settings"),
        ({"cell_cycle": {"genes": []}}, ValueError, "Unknown cell_cycle settings"),
        ({"umap": 400}, TypeError, "umap must be a mapping or bool"),
        ({"pca": True}, TypeError, "always runs"),
        ({"neighbors": False}, TypeError, "always runs"),
        # Membership strength and markers switch without settings.
        ({"markers": {"nthreads": 2}}, TypeError, "markers must be a boolean"),
        (
            {"membership_strength": {}},
            TypeError,
            "membership_strength must be a boolean",
        ),
        # The embeddings run late, so their settings are checked before any
        # stage runs, as are the features.hvgs checks of the hvg settings.
        ({"tsne": {"box_h": -1.0}}, ValueError, "box_h must be positive"),
        ({"umap": {"symmetric_graph": None}}, TypeError, "symmetric_graph must be"),
        (
            {"leiden": {"graph_upper_only": True}},
            ValueError,
            "applies only with symmetric_graph=True",
        ),
        ({"hvg": {"min_mean": float("inf")}}, ValueError, "must be finite"),
        ({"hvg": {"min_cells": 2.5}}, TypeError, "min_cells must be an integer"),
        ({"hvg": {"top_n": 0}}, ValueError, "top_n must be at least 1"),
        (
            {"hvg": {"flavor": "seurat_v3", "n_bins": 20}},
            ValueError,
            "apply only to flavor='cytearc'",
        ),
        ({"filtering": "mad"}, TypeError, "filtering must be a mapping, bool, or"),
        ({"filtering": {"method": "manual"}}, ValueError, "requires lows and highs"),
        ({"species": "human"}, ValueError, "species"),
        ({"harmony": True}, ValueError, "batch_columns"),
        # A list of batch columns was the earlier form of the argument.
        (
            {"harmony": ["ids"]},
            TypeError,
            r"^harmony must be a mapping such as \{'batch_columns': \[\.\.\.\]\}, "
            r"False, or None$",
        ),
        # PCA needs one selected feature more than its dimensions, whether the
        # call sets the HVG count or keeps its default of 1000.
        (
            {"hvg": {"top_n": 5}, "pca": {"dims": 5}},
            ValueError,
            r"^pca dims \(5\) must be below hvg top_n \(5\): PCA needs at least "
            r"dims \+ 1 features$",
        ),
        (
            {"pca": {"dims": 1000}},
            ValueError,
            r"^pca dims \(1000\) must be below hvg top_n \(1000\)",
        ),
        # Harmony options are checked when the recipe is resolved, so a renamed
        # key fails before the stages that precede Harmony run.
        (
            {"harmony": {"batch_columns": ["ids"], "harmony_params": {"nclust": 3}}},
            ValueError,
            "Unsupported Harmony parameter 'nclust'",
        ),
        ({"leiden": {"partitions": [0.5], "selected": 1.0}}, ValueError, "selected"),
        # Doublet scoring and markers are on by default.
        ({"leiden": False}, ValueError, "require at least one Leiden candidate"),
        (
            {"snapshot_columns": ["umap_3"], "umap": {"umap_dims": 3}},
            ValueError,
            "collide with reserved run fields",
        ),
    ],
)
def test_recipe_refuses_invalid_settings_before_writing(
    rejection_store, keywords, error, match
):
    store = rejection_store
    before = _written(store)

    with pytest.raises(error, match=match):
        store.pipeline.rna(**keywords)

    assert _written(store) == before


def test_the_params_mapping_and_its_shortcuts_are_gone(rejection_store):
    store = rejection_store
    before = _written(store)

    # Recipes replace pipeline.run, and stage keywords replace its params
    # mapping and shortcut arguments, without aliases.
    assert not hasattr(store.pipeline, "run")
    for keyword, value in (
        ("params", {"hvg": {"top_n": 50}}),
        ("hvg_count", 50),
        ("pca_dims", 3),
        ("neighbors_k", 3),
        ("harmony_batch_columns", ["ids"]),
        ("cell_key", "I"),
    ):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{keyword}'"):
            store.pipeline.rna(**{keyword: value})

    assert _written(store) == before


def test_harmony_needs_a_mapping_with_batch_columns(readonly_store):
    store = readonly_store
    for harmony, error, message in (
        (True, ValueError, "^harmony needs a mapping with batch_columns$"),
        ({"batch_size": 100}, ValueError, "^harmony needs batch_columns$"),
        (
            {"batch_columns": []},
            ValueError,
            "^harmony batch_columns must not be empty$",
        ),
        (
            {"batch_columns": "ids"},
            TypeError,
            "^harmony batch_columns must be a sequence of column names$",
        ),
        (
            {"batch_columns": ["ids", "ids"]},
            ValueError,
            "^harmony batch_columns must not contain duplicates$",
        ),
        (
            {"batch_columns": ["missing"]},
            KeyError,
            r"Harmony columns were not found: \['missing'\]",
        ),
        (
            {"batch_columns": ["ids"], "theta": 2},
            ValueError,
            "Unknown harmony settings",
        ),
        (
            {"batch_columns": ["ids"], "harmony_params": {"nclust": 3}},
            ValueError,
            "Unsupported Harmony parameter 'nclust'",
        ),
        (
            {"batch_columns": ["ids"], "harmony_params": {"theta": "2"}},
            TypeError,
            "theta must contain real numbers",
        ),
        (
            {"batch_columns": ["ids"], "harmony_params": {"max_iter_harmony": 0}},
            ValueError,
            "max_iter_harmony must be at least 1",
        ),
        (
            {"batch_columns": ["ids"], "harmony_params": ["theta"]},
            TypeError,
            "mapping",
        ),
        # Harmony takes a mapping, never the batch columns themselves.
        *(
            (
                columns,
                TypeError,
                r"^harmony must be a mapping such as \{'batch_columns': \[\.\.\.\]\}, "
                r"False, or None$",
            )
            for columns in (["ids"], ("ids",), "ids")
        ),
    ):
        with pytest.raises(error, match=message):
            _resolve(store, harmony=harmony)

    # Harmony is off unless the call names its batch columns.
    for off in (None, False):
        plan = _resolve(store, harmony=off)
        assert plan.config["stages"]["harmony"] is False
        assert {stage.name: stage.action for stage in plan.stages}["harmony"] is None
        assert plan.config["doubletSampleColumns"] == []
    plan = _resolve(
        store,
        harmony={
            "batch_columns": ["ids"],
            "batch_size": 100,
            "harmony_params": {"n_clusters": np.int64(3), "theta": [1.0, 2.0]},
        },
    )
    assert plan.config["stages"]["harmony"] == {
        "batch_columns": ["ids"],
        "batch_size": 100,
        "harmony_params": {"n_clusters": 3, "theta": [1.0, 2.0]},
    }
    # Without a sample column of its own, a run with Harmony scores doublets
    # within each Harmony batch, so the column cannot be None.
    assert plan.config["doubletSampleColumns"] == ["ids"]
    with pytest.raises(ValueError, match=r"doublets\['sample_column'\] cannot be None"):
        _resolve(
            store,
            harmony={"batch_columns": ["ids"]},
            doublets={"sample_column": None},
        )


def test_leiden_settings_choose_the_candidates(readonly_store):
    store = readonly_store

    def leiden_stages(plan: Any) -> list[str]:
        return [stage for stage in plan.stage_order if stage.startswith("leiden_")]

    default = _resolve(store)
    assert default.config["stages"]["leiden"] == {
        "partitions": list(DEFAULT_LEIDEN_PARTITIONS),
        "selected": None,
    }
    assert leiden_stages(default) == [
        f"leiden_{resolution}" for resolution in DEFAULT_LEIDEN_PARTITIONS
    ]
    # Settings without partitions keep the default candidates.
    assert _resolve(store, leiden={"random_seed": 1}).config["stages"]["leiden"] == {
        "partitions": list(DEFAULT_LEIDEN_PARTITIONS),
        "selected": None,
        "random_seed": 1,
    }
    # Integer and float spellings of one resolution are one candidate.
    plan = _resolve(store, leiden={"partitions": [2, 0.5], "selected": 2})
    assert plan.config["stages"]["leiden"] == {
        "partitions": [2.0, 0.5],
        "selected": 2.0,
    }
    assert leiden_stages(plan) == ["leiden_2.0", "leiden_0.5"]
    for leiden, error, message in (
        (
            {"partitions": [0.5], "selected": 1.0},
            ValueError,
            "^leiden selected must be one of the partitions$",
        ),
        ({"partitions": []}, ValueError, "^leiden partitions must not be empty$"),
        ({"partitions": [1, 1.0]}, ValueError, "duplicate resolutions"),
        ({"partitions": 1.0}, TypeError, "must be a non-empty sequence"),
        ({"partitions": [0]}, ValueError, "finite and positive"),
        ({"partitions": [1.0], "selected": "1.0"}, TypeError, "must be a number"),
    ):
        with pytest.raises(error, match=message):
            _resolve(store, leiden=leiden)

    # Without candidates there are no saved clusters to score doublets,
    # markers, or membership strength on.
    off = {"leiden": False, "doublets": False, "markers": False}
    plan = _resolve(store, **off)
    assert plan.config["stages"]["leiden"] is False
    assert leiden_stages(plan) == []
    with pytest.raises(
        ValueError, match="^doublets and markers require at least one Leiden candidate$"
    ):
        _resolve(store, leiden=False, doublets=False)
    with pytest.raises(
        ValueError, match="^membership_strength requires at least one Leiden candidate$"
    ):
        _resolve(store, **off, membership_strength=True)


def test_a_selected_resolution_is_the_saved_clustering(settings_store):
    store = settings_store

    run = store.pipeline.rna(
        **_fast(
            leiden={"partitions": [0.5, 1.0], "selected": 1.0},
            membership_strength=True,
        )
    )

    outputs = list(run)
    assert run["clusters"] == run["leiden_1.0"]
    assert "cluster_selection" not in outputs
    statuses = _stage_statuses(run)
    assert statuses["cluster_selection"] == "skipped"
    assert statuses["leiden_0.5"] == statuses["leiden_1.0"] == "completed"
    assert {"leiden_0.5", "leiden_1.0", "clusters", "membership_strength"} <= set(
        run.cells.columns
    )
    np.testing.assert_array_equal(
        run.cells.fetch("clusters"), run.cells.fetch("leiden_1.0")
    )
    # Membership strength is computed on the saved clustering and run graph.
    inputs = store.artifacts.inspect(run["membership_strength"]).inputs
    assert ArtifactRef.from_dict(inputs["clusters"]) == run["leiden_1.0"]
    assert ArtifactRef.from_dict(inputs["connectivity_map"]) == run["connectivity_map"]
    strength = run.cells.fetch("membership_strength")
    assert strength.shape == run.cells.fetch("clusters").shape
    assert np.all((strength >= 0) & (strength <= 1))
    assert run.report()["run"]["config"]["stages"]["leiden"] == {
        "partitions": [0.5, 1.0],
        "selected": 1.0,
    }


def test_doublets_on_one_selected_cluster_name_the_recipe_settings(settings_store):
    store = settings_store
    # At this resolution the k=15 graph of the fixture is one Leiden cluster,
    # which forced heterotypic doublets cannot pair.
    keywords = _fast(
        neighbors={"k": 15},
        leiden={"partitions": [0.0001], "selected": 0.0001},
        doublets={"heterotypic_fraction": 0.8},
    )

    with pytest.raises(PipelineExecutionError) as caught:
        store.pipeline.rna(**keywords)

    assert caught.value.stage == "doublets"
    assert isinstance(caught.value.__cause__, ValueError)
    message = str(caught.value)
    assert "every selected cell has the same cluster label" in message
    for setting in (
        "leiden={'selected': ...}",
        "doublets={'heterotypic_fraction': 0}",
        "doublets=False",
    ):
        assert setting in message
    assert store.pipeline.open(run_id=caught.value.run_id).status == "failed"


def test_tsne_settings_reach_the_embedding(settings_store, monkeypatch):
    store = settings_store
    calls: list[dict[str, Any]] = []

    def run_sgtsne(_graph, initial, **kwargs):
        calls.append(kwargs)
        return np.asarray(initial, dtype=np.float64).T

    monkeypatch.setattr("cytearc.embeddings.sgtsne.run_sgtsne", run_sgtsne)
    monkeypatch.setattr(
        "cytearc.embeddings.sgtsne.require_sgtsnepi", lambda: run_sgtsne
    )

    run = store.pipeline.rna(**_fast(tsne={"max_iter": 60, "early_iter": 20}))

    (call,) = calls
    assert (call["max_iter"], call["early_iter"]) == (60, 20)
    # The omitted settings are the method's defaults.
    defaults = _keywords(EmbeddingsAccessor.tsne)
    for name in ("tsne_dims", "alpha", "lambda_scale", "box_h"):
        assert call[name] == defaults[name], name
    # The pipeline runs t-SNE quietly and never passes thread or file options.
    assert call["verbose"] is False
    assert not {"parallel", "nthreads", "temp_file_loc"} & set(call)
    outputs = list(run)
    assert "embedding_initialization" in outputs
    assert "umap" not in outputs
    assert [column for column in run.cells.columns if column.startswith("tsne_")] == [
        f"tsne_{index}" for index in range(1, defaults["tsne_dims"] + 1)
    ]
    assert run.report()["run"]["config"]["stages"]["tsne"] == {
        "max_iter": 60,
        "early_iter": 20,
    }


def test_a_tsne_recipe_warns_at_resolution_when_sgtsnepi_is_missing(
    readonly_store, monkeypatch
):
    # An entry of None makes every import of sgtsnepi raise ImportError.
    monkeypatch.setitem(sys.modules, "sgtsnepi", None)

    with _captured_warnings() as messages:
        plan = _resolve(readonly_store, tsne={"max_iter": 800})
    assert plan.config["stages"]["tsne"] == {"max_iter": 800}
    (message,) = [message for message in messages if "sgtsnepi" in message]
    assert "tsne extra" in message and "cytearc[tsne]" in message
    # The run is not refused: an embedding the store holds is reused.
    assert "reuse" in message

    with _captured_warnings() as messages:
        _resolve(readonly_store, tsne=False)
        _resolve(readonly_store)
    assert not [message for message in messages if "sgtsnepi" in message]


def test_pca_dims_zero_skips_pca_without_settings(readonly_store):
    store = readonly_store

    plan = _resolve(store, pca={"dims": 0}, doublets=False)

    assert plan.config["stages"]["pca"] == {"dims": 0}
    assert {stage.name: stage.action for stage in plan.stages}["pca"] is None
    for pca, error, message in (
        (
            {"dims": 0, "feat_scaling": False},
            ValueError,
            "^PCA settings need dims above 0; dims=0 skips PCA$",
        ),
        ({"dims": -1}, ValueError, "^pca dims must be at least 0$"),
        ({"dims": 2.0}, TypeError, "^pca dims must be an integer$"),
        ({"dims": True}, TypeError, "^pca dims must be an integer$"),
    ):
        with pytest.raises(error, match=message):
            _resolve(store, pca=pca, doublets=False)


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        (
            _fast(pca={"dims": 0}, harmony={"batch_columns": ["ids"]}),
            "pca={'dims': 0} builds the graph on normalized values, which Harmony "
            "cannot correct; set pca dims above 0 or omit harmony",
        ),
        (
            # Doublet scoring is on by default.
            {"pca": {"dims": 0}},
            "pca={'dims': 0} builds the graph on normalized values, but doublet "
            "scoring needs a PCA graph; pass doublets=False or set pca dims above 0",
        ),
    ],
    ids=["harmony", "doublets-default"],
)
def test_recipe_rejects_pca_dims_zero_with_harmony_or_doublets(
    rejection_store, keywords, message
):
    store = rejection_store
    before = _written(store)

    with pytest.raises(ValueError) as caught:
        store.pipeline.rna(**keywords)

    assert str(caught.value) == message
    # The recipe raised before the run record or any artifact was written.
    assert _written(store) == before


def test_pca_dims_zero_builds_the_graph_on_normalized_values(
    settings_store, monkeypatch
):
    store = settings_store
    (base,) = store.pipeline.list_runs()
    n_features = len(
        read_feature_selection_indices(
            store.zw, "RNA", base["highly_variable_features"]
        )
    )
    assert n_features == 50
    reductions = store.artifacts.list(kind="reduction", from_assay="RNA")

    def no_custom_reduction(*_args, **_kwargs):
        raise AssertionError("pca={'dims': 0} registered a custom reduction")

    square_matrices: list[int] = []
    eye = np.eye

    def recorded_eye(n, *args, **kwargs):
        if n >= n_features:
            square_matrices.append(n)
        return eye(n, *args, **kwargs)

    monkeypatch.setattr(type(store), "_reduction_custom", no_custom_reduction)
    monkeypatch.setattr(np, "eye", recorded_eye)

    run = store.pipeline.rna(**_fast(pca={"dims": 0}, leiden={"partitions": [1.0]}))

    outputs = list(run)
    assert "pca" not in outputs and "reduction" not in outputs
    assert {"normalized", "ann_index", "neighbors", "connectivity_map"} <= set(outputs)
    assert _stage_statuses(run)["pca"] == "skipped"
    assert run.report()["run"]["config"]["stages"]["pca"] == {"dims": 0}
    # Each selected feature is one graph coordinate: the ANN index and the
    # neighbors read the normalized artifact itself.
    for key in ("ann_index", "neighbors"):
        status = store.artifacts.inspect(run[key])
        assert status.input_ref("coordinates") == run["normalized"]
    assert store.artifacts.load(run["normalized"])["data"].shape[1] == n_features
    # No identity loadings: nothing registered a reduction or built an
    # F x F matrix.
    assert store.artifacts.list(kind="reduction", from_assay="RNA") == reductions
    assert square_matrices == []
    # Silhouette selection scores the candidates on the normalized values,
    # the coordinates of the graph they partition.
    decision = store.artifacts.inspect(run["cluster_selection"])
    assert decision.input_ref("coordinates") == run["normalized"]
    assert decision.input_ref("connectivityMap") == run["connectivity_map"]
    assert run["clusters"] == run["leiden_1.0"]
    assert "clusters" in run.cells.columns


def test_species_is_validated_and_recorded(readonly_store):
    assert _resolve(readonly_store).config["species"] is None
    plan = _resolve(readonly_store, species="mus_musculus")
    assert plan.config["species"] == "mus_musculus"
    message = f"species must be one of {sorted(species_registry())!r}"
    for species in ("human", "Homo_sapiens", 1):
        with pytest.raises(ValueError) as caught:
            _resolve(readonly_store, species=species)
        assert str(caught.value) == message


def test_filtering_settings_resolve_to_the_filter_parameters(readonly_store):
    automatic = _keywords(QcAccessor.auto_filter)

    def filtering(value: object) -> dict[str, Any]:
        return _resolve(readonly_store, filtering=value).config["stages"]["filtering"]

    mad = {
        "enabled": True,
        "method": "mad",
        "attrs": _QC_COLUMNS,
        "minP": automatic["min_p"],
        "maxP": automatic["max_p"],
        "sampleColumn": None,
        "nMads": automatic["n_mads"],
        "minCellsPerSample": automatic["min_cells_per_sample"],
    }
    # Automatic filtering is MAD filtering over the QC columns that
    # preparation wrote for the assay.
    assert filtering(True) == mad
    assert filtering({"method": "auto"}) == mad
    assert filtering({"method": "mad"}) == mad
    assert filtering(
        {
            "attrs": ["RNA_nCounts"],
            "n_mads": 4,
            "sample_column": "ids",
            "min_cells_per_sample": 5,
        }
    ) == {
        **mad,
        "attrs": ["RNA_nCounts"],
        "nMads": 4.0,
        "sampleColumn": "ids",
        "minCellsPerSample": 5,
    }
    # Gaussian filtering takes quantile bounds instead of MADs.
    assert filtering({"method": "gaussian", "min_p": 0.05, "max_p": 0.95}) == {
        **mad,
        "method": "gaussian",
        "minP": 0.05,
        "maxP": 0.95,
    }
    # Manual filtering takes one low and one high bound for each column.
    assert filtering(
        {
            "method": "manual",
            "attrs": ["RNA_nCounts", "RNA_percentMito"],
            "lows": [500, None],
            "highs": [None, 10],
            "keep_bounds": True,
        }
    ) == {
        "enabled": True,
        "method": "manual",
        "attrs": ["RNA_nCounts", "RNA_percentMito"],
        "lows": [500, None],
        "highs": [None, 10],
        "keepBounds": True,
    }
    manual = {
        "method": "manual",
        "attrs": ["RNA_nCounts"],
        "lows": [1],
        "highs": [None],
    }
    assert filtering(manual)["keepBounds"] is False
    assert filtering(False) == {"enabled": False}


@pytest.mark.parametrize(
    ("filtering", "error", "match"),
    [
        ("mad", TypeError, "^filtering must be a mapping, bool, or None$"),
        (
            {"method": "median"},
            ValueError,
            "^filtering method must be 'auto', 'mad', 'gaussian', or 'manual'$",
        ),
        ({"attrs": ["missing"]}, KeyError, r"Filtering columns were not found"),
        ({"attrs": []}, ValueError, "no QC columns were found"),
        # MAD bounds come from each metric's median, so its quantiles are fixed.
        ({"min_p": 0.05}, ValueError, "cannot be changed with method='mad'"),
        (
            {"method": "auto", "lows": [0]},
            ValueError,
            r"Unknown automatic filtering options: \['lows'\]",
        ),
        ({"n_mads": 0}, ValueError, "n_mads must be finite and positive"),
        (
            {"min_cells_per_sample": 1},
            ValueError,
            "min_cells_per_sample must be at least 2",
        ),
        ({"sample_column": "missing"}, KeyError, "Sample column 'missing'"),
        (
            {"method": "gaussian", "min_p": 0.6, "max_p": 0.4},
            ValueError,
            "requires 0 < min_p < max_p < 1",
        ),
        (
            {"method": "gaussian", "sample_column": "ids"},
            ValueError,
            "Gaussian filtering does not support a sample source",
        ),
        ({"method": "gaussian", "n_mads": 4.0}, ValueError, "apply only to"),
        (
            {"method": "manual", "attrs": ["RNA_nCounts"]},
            ValueError,
            "Manual filtering requires lows and highs",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0, 1],
                "highs": [2, 3],
            },
            ValueError,
            "must align with attrs",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0],
                "highs": [2],
                "n_mads": 3.0,
            },
            ValueError,
            r"Unknown manual filtering options: \['n_mads'\]",
        ),
        (
            {"method": "manual", "attrs": ["RNA_nCounts"], "lows": [3], "highs": [2]},
            ValueError,
            "cannot exceed",
        ),
    ],
)
def test_filtering_settings_are_checked_for_their_method(
    readonly_store, filtering, error, match
):
    with pytest.raises(error, match=match):
        _resolve(readonly_store, filtering=filtering)


def test_qc_stage_settings_name_existing_columns_and_assays(readonly_store):
    store = readonly_store
    for keywords, error, match in (
        # The fixture has no HTO assay for hashing to read.
        (
            {"hashing": True},
            ValueError,
            r"hashing needs an assay of type 'HTO'.*HTO assays: \[\]",
        ),
        (
            {"hashing": {"assay": "RNA"}},
            ValueError,
            r"hashing\['assay'\] must name an HTO assay",
        ),
        (
            {"cell_calling": {"sample_column": "missing"}},
            KeyError,
            "Capture column 'missing' was not found",
        ),
        (
            {"cell_calling": {"sample_column": ""}},
            TypeError,
            r"cell_calling\['sample_column'\] must be a cell column",
        ),
        (
            {"doublets": {"sample_column": "missing"}},
            KeyError,
            "Doublet sample column 'missing' was not found",
        ),
        (
            {"doublets": {"sample_column": ""}},
            TypeError,
            r"doublets\['sample_column'\] must be a non-empty string or None",
        ),
        (
            {"hvg": {"batch": "missing"}},
            KeyError,
            "HVG batch column 'missing' was not found",
        ),
        (
            {"hvg": {"batch": 1}},
            TypeError,
            r"hvg\['batch'\] must be a cell metadata column",
        ),
    ):
        with pytest.raises(error, match=match):
            _resolve(store, **keywords)

    # Cell calling records its capture column with its settings, and a
    # doublet sample column splits doublet scoring.
    plan = _resolve(
        store,
        cell_calling={"lower": 50, "sample_column": "ids"},
        doublets={"sample_column": "ids"},
    )
    assert plan.config["stages"]["cell_calling"] == {
        "lower": 50,
        "sample_column": "ids",
    }
    assert plan.config["stages"]["doublets"] == {"sample_column": "ids"}
    assert plan.config["doubletSampleColumns"] == ["ids"]


def test_filtering_none_filters_only_cells_that_a_column_names(base_view):
    store = base_view
    (base,) = store.pipeline.list_runs()
    selection = base["analysis_cell_selection"]

    def filtering(**keywords: Any) -> dict[str, Any]:
        return _resolve(store, **keywords).config["stages"]["filtering"]

    # Cells that a column names are filtered on their QC metrics.
    assert filtering()["enabled"] is True
    assert filtering(cells="I") == filtering(filtering=True)
    # A given selection's or run's cells are taken as already filtered.
    assert filtering(cells=selection) == {"enabled": False}
    assert filtering(cells=base) == {"enabled": False}
    # Unless the run calls cells: it filters the cells that it calls.
    assert filtering(cells=selection, cell_calling=True) == filtering(filtering=True)
    assert filtering(cells=base, cell_calling=True) == filtering(filtering=True)
    # Filtering that the call asks for applies to any cells.
    assert filtering(cells=selection, filtering=True) == filtering(filtering=True)
    gaussian = filtering(cells=base, filtering={"method": "gaussian"})
    assert gaussian["method"] == "gaussian"
    assert filtering(filtering=False) == {"enabled": False}
    # The run records where its cells came from.
    assert _resolve(store, cells=selection).config["cells"] == {
        "selection": selection.to_dict()
    }
    assert _resolve(store, cells=base).config["cells"] == {
        "run": {"runId": base.run_id, "label": base.label},
        "selection": selection.to_dict(),
    }


def test_an_adt_run_filters_only_when_asked(adt_store):
    def filtering(**keywords: Any) -> dict[str, Any]:
        plan = resolve_adt(adt_store, **{**_recipe_defaults("adt"), **keywords})
        return plan.config["stages"]["filtering"]

    # An ADT run usually takes the QC-passing cells of an RNA run, so None
    # leaves even the cells that a column names unfiltered.
    assert filtering() == {"enabled": False}
    # Nearly every cell detects nearly every antibody of a panel, so a
    # requested filter reads only the total antibody counts by default.
    assert filtering(filtering=True)["attrs"] == ["assay2_nCounts"]


def test_a_run_given_a_run_analyzes_its_cells_as_they_are(settings_store):
    store = settings_store
    (base,) = store.pipeline.list_runs()

    run = store.pipeline.rna(cells=base, **_fast(filtering=None))

    # The run's input is the base run's analysis selection itself, and the
    # default filtering keeps it as it is.
    assert run["input_cell_selection"] == base["analysis_cell_selection"]
    assert run["analysis_cell_selection"] == base["analysis_cell_selection"]
    assert _stage_statuses(run)["filtering"] == "skipped"
    config = run.report()["run"]["config"]
    assert config["stages"]["filtering"] == {"enabled": False}
    assert config["cells"]["run"] == {"runId": base.run_id, "label": base.label}
    # With the base run's settings, the run reuses each of its artifacts.
    assert all(run[key] == base[key] for key in base)
