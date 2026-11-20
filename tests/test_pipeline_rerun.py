"""Rerun a pipeline run with the settings it recorded, changing some of them.

A module fixture commits one RNA run on a private copy of the PBMC store.
Tests that run a call copy that store, so their runs reuse the committed run's
artifacts. The modules that commit runs of every recipe check that a plan of
each run with its own settings reuses all of it: ``test_pipeline_recipes.py``
and ``test_pipeline_perturbation.py``.
"""

import dataclasses
import inspect
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from cytearc import DataStore, PipelineExecutionError, PipelineRun
from cytearc.datastore._pipeline.filtering import (
    filtering_argument,
    resolve_filtering,
)
from cytearc.datastore.pipeline_accessor import (
    _RECORDED_CALLS,
    _RESOLVERS,
    PipelineAccessor,
    PipelineEvent,
)
from cytearc.storage.pipeline_runs import (
    load_pipeline_run_record,
    pipeline_run_path,
)
from tests.test_pipeline_plan import (
    _digest,
    _dispositions,
    assert_plan_reuses_run,
    read_only_plan,
)
from tests.test_pipeline_run_foundation import _completed_run, _root
from tests.storage_helpers import by_cell_id

# Two Leiden candidates, UMAP and markers, kept small: 200 genes and 5 PCs.
SETTINGS: dict[str, Any] = {
    "cell_cycle": False,
    "hvg": {"top_n": 200},
    "pca": {"dims": 5},
    "neighbors": {"k": 5},
    "umap": {"n_epochs": 30},
    "leiden": {"partitions": [0.5, 1.0]},
    "doublets": False,
    "markers": True,
}
# The QC columns that preparation wrote for the store's RNA assay.
QC_COLUMNS = ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo"]
# The stages that a change to the Leiden partitions computes again.
LEIDEN_STAGES = {"leiden", "cluster_selection", "markers"}


@pytest.fixture(scope="module")
def completed(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Commit the run of ``SETTINGS`` once on a private copy of the PBMC store.

    The store's ``batch`` column labels the cells in turn, for filtering
    settings that name a sample column; the run does not read it.
    """
    location = tmp_path_factory.mktemp("rerun_completed") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    datastore = DataStore(str(location), default_assay="RNA")
    batches = np.where(np.arange(datastore.cells.N) % 2 == 0, "a", "b")
    datastore.cells.insert("batch", by_cell_id(datastore.cells, batches))
    run = datastore.pipeline.rna(label="baseline", **SETTINGS)
    return SimpleNamespace(location=location, datastore=datastore, run=run)


def _copy(location: Path, tmp_path: Path) -> DataStore:
    target = tmp_path / "data.zarr"
    shutil.copytree(location, target)
    return DataStore(str(target), default_assay="RNA")


def _stage_config(run: PipelineRun) -> dict[str, Any]:
    return run.report()["run"]["config"]["stages"]


def assert_settings_name_every_keyword(run: PipelineRun) -> None:
    """Check that a run of this release records every keyword of its recipe.

    A run recorded by an earlier release may lack a keyword that the recipe
    gained since, but a run of this release lacks one only when its recipe's
    ``recorded_call`` leaves it out.
    """
    recipe = getattr(PipelineAccessor, run.recipe)
    keywords = set(inspect.signature(recipe).parameters)
    assert set(run.settings()) == keywords - {"self", "label", "callback", "cancel"}


# Settings --------------------------------------------------------------------


def test_settings_are_the_keywords_that_the_run_was_called_with(completed) -> None:
    settings = completed.run.settings()

    assert settings == {
        "assay": "RNA",
        "cells": "I",
        "cell_calling": False,
        "hashing": False,
        "filtering": {"method": "mad", "attrs": QC_COLUMNS},
        "cell_cycle": False,
        "hvg": {"top_n": 200},
        # The call named no normalizer, so the run records the assay's.
        "normalization": {"method": "lib_size"},
        "pca": {"dims": 5},
        "harmony": False,
        "ann_index": {},
        "neighbors": {"k": 5},
        "connectivity": {},
        "embedding_initialization": {},
        "umap": {"n_epochs": 30},
        "localmap": False,
        "tsne": False,
        "leiden": {"partitions": [0.5, 1.0], "selected": None},
        "paris": False,
        "membership_strength": False,
        "doublets": False,
        "markers": True,
        "snapshot_columns": [],
        "species": None,
    }
    # They are every keyword of the recipe but those of a new run.
    assert_settings_name_every_keyword(completed.run)


def test_every_recipe_has_a_resolver_and_a_translation_of_its_record() -> None:
    # The recipes of this release, each a method of the pipeline accessor. The
    # modules that commit runs of each recipe check that the translation of
    # its record names every keyword of the method.
    recipes = {"adt", "atac", "multimodal", "perturbation", "rna"}
    assert set(_RESOLVERS) == set(_RECORDED_CALLS) == recipes
    for recipe in recipes:
        assert (
            "label" in inspect.signature(getattr(PipelineAccessor, recipe)).parameters
        )


def test_settings_are_copies_that_leave_the_record_unchanged(completed) -> None:
    run = completed.run
    recorded = _stage_config(run)

    settings = run.settings()
    settings["leiden"]["partitions"].append(2.0)
    settings["filtering"]["attrs"].clear()
    settings["hvg"]["top_n"] = 1

    assert run.settings()["leiden"] == {"partitions": [0.5, 1.0], "selected": None}
    assert run.settings()["filtering"]["attrs"] == QC_COLUMNS
    assert run.settings()["hvg"] == {"top_n": 200}
    assert _stage_config(run) == recorded


@pytest.mark.parametrize(
    "keyword",
    [
        True,
        False,
        {"method": "auto"},
        {"method": "mad", "n_mads": 4, "min_cells_per_sample": 5},
        {"method": "mad", "sample_column": "batch", "attrs": QC_COLUMNS[:2]},
        {"method": "gaussian", "min_p": 0.05, "max_p": 0.9},
        {
            "method": "manual",
            "attrs": QC_COLUMNS[:2],
            "lows": [100, None],
            "highs": [None, 5000],
            "keep_bounds": True,
        },
        {
            "method": "manual",
            "attrs": QC_COLUMNS[:1],
            "lows": [100.5],
            "highs": [20000],
        },
    ],
)
def test_the_filtering_keyword_resolves_to_the_settings_it_was_read_from(
    completed, keyword
) -> None:
    store = completed.datastore
    recorded = resolve_filtering(store, keyword, default_attrs=QC_COLUMNS)

    argument = filtering_argument(recorded)

    assert resolve_filtering(store, argument, default_attrs=QC_COLUMNS) == recorded
    # The source of a hashed run's MAD bounds follows from its hashing.
    hashed = filtering_argument({**recorded, "sampleSource": "hashing"})
    assert hashed == argument


def test_the_filtering_keyword_names_only_the_options_that_differ_from_defaults(
    completed,
) -> None:
    store = completed.datastore

    def argument(keyword: object) -> object:
        recorded = resolve_filtering(store, keyword, default_attrs=QC_COLUMNS)
        return filtering_argument(recorded)

    assert argument(False) is False
    assert argument(True) == {"method": "mad", "attrs": QC_COLUMNS}
    assert argument({"method": "mad", "n_mads": 4, "sample_column": "batch"}) == {
        "method": "mad",
        "attrs": QC_COLUMNS,
        "sample_column": "batch",
        "n_mads": 4,
    }
    assert argument({"method": "gaussian", "max_p": 0.9}) == {
        "method": "gaussian",
        "attrs": QC_COLUMNS,
        "max_p": 0.9,
    }
    manual = {"method": "manual", "attrs": ["RNA_nCounts"], "lows": [1], "highs": [9]}
    assert argument(manual) == manual
    assert argument({**manual, "keep_bounds": False}) == manual


def test_the_filtering_keyword_leaves_out_options_that_the_run_did_not_record(
    completed,
) -> None:
    store = completed.datastore
    mad = resolve_filtering(store, {"method": "mad"}, default_attrs=QC_COLUMNS)
    manual = {"method": "manual", "attrs": ["RNA_nCounts"], "lows": [1], "highs": [9]}
    bounds = resolve_filtering(store, manual, default_attrs=QC_COLUMNS)
    options = {"minP", "maxP", "nMads", "minCellsPerSample", "sampleColumn"}

    # A run recorded before an option existed takes the option's default.
    earlier_mad = {key: value for key, value in mad.items() if key not in options}
    earlier_manual = {
        key: value for key, value in bounds.items() if key != "keepBounds"
    }

    assert filtering_argument(earlier_mad) == {"method": "mad", "attrs": QC_COLUMNS}
    assert filtering_argument(earlier_manual) == manual


# Plans of a run -----------------------------------------------------------------


def test_a_plan_of_the_run_reuses_every_stage(completed) -> None:
    run = completed.run

    assert_plan_reuses_run(completed.datastore.pipeline.plan(run), run)
    # A plan of a run, like any plan, works on a read-only store.
    assert_plan_reuses_run(read_only_plan(completed.location, run), run)


def test_a_changed_setting_computes_only_the_stages_that_read_it(completed) -> None:
    run = completed.run

    plan = completed.datastore.pipeline.plan(run, leiden={"partitions": [2.0]})

    computed = {
        stage.stage: stage.reason for stage in plan.stages if stage.status == "computed"
    }
    assert computed == {
        "leiden": "changed",
        "cluster_selection": "upstream",
        "markers": "upstream",
    }
    reused = [stage.stage for stage in plan.stages if stage.status == "reused"]
    assert reused[0] == "input_snapshot" and reused[-1] == "umap"
    stages = plan.config["stages"]
    assert stages["leiden"] == {"partitions": [2.0], "selected": None}
    # Every other setting is the run's.
    recorded = _stage_config(run)
    assert {key: value for key, value in stages.items() if key != "leiden"} == {
        key: value for key, value in recorded.items() if key != "leiden"
    }


def test_a_mapping_changes_a_stage_key_by_key(completed) -> None:
    pipeline = completed.datastore.pipeline
    run = completed.run

    plan = pipeline.plan(run, umap={"min_dist": 0.3})

    assert plan.config["stages"]["umap"] == {"n_epochs": 30, "min_dist": 0.3}
    # Leiden, cluster selection and markers read the graph, not UMAP.
    assert plan.computed == ("umap",)
    # A key that the run set takes the new value.
    plan = pipeline.plan(run, leiden={"selected": 1.0})
    assert plan.config["stages"]["leiden"] == {
        "partitions": [0.5, 1.0],
        "selected": 1.0,
    }
    # A requested resolution skips cluster selection, and markers rank the
    # clusters at 1.0 in place of the silhouette choice.
    assert plan["cluster_selection"].status == "skipped"
    assert plan.computed == ("markers",)
    # The kept selected resolution must be one of new partitions.
    with pytest.raises(ValueError, match="leiden selected must be one of the"):
        pipeline.plan(run, leiden={"selected": 1.0, "partitions": [2.0]})


def test_any_other_value_replaces_the_run_setting(completed) -> None:
    pipeline = completed.datastore.pipeline
    run = completed.run

    plan = pipeline.plan(run, umap=False, leiden=True, filtering=False)

    stages = plan.config["stages"]
    assert stages["umap"] is False
    assert plan["umap"].status == "skipped"
    # True runs Leiden at the recipe's default partitions.
    assert stages["leiden"] == {"partitions": [0.5, 0.75, 1.0, 1.25], "selected": None}
    assert stages["filtering"] == {"enabled": False}


def test_new_cells_keep_the_run_filtering_unless_the_call_resets_it(completed) -> None:
    pipeline = completed.datastore.pipeline
    run = completed.run
    recorded = _stage_config(run)["filtering"]

    plan = pipeline.plan(run, cells=run)

    cells = run["analysis_cell_selection"]
    assert plan.config["cells"] == {
        "run": {"runId": run.run_id, "label": "baseline"},
        "selection": cells.to_dict(),
    }
    assert plan.config["stages"]["filtering"] == recorded
    # None takes the default of a call that omits filtering: a run's cells are
    # taken as filtered.
    plan = pipeline.plan(run, cells=run, filtering=None)
    assert plan.config["stages"]["filtering"] == {"enabled": False}


def test_filtering_changes_its_method_without_the_earlier_options(completed) -> None:
    pipeline = completed.datastore.pipeline
    run = completed.run

    plan = pipeline.plan(run, filtering={"method": "gaussian"})

    assert plan.config["stages"]["filtering"] == {
        "enabled": True,
        "method": "gaussian",
        "attrs": QC_COLUMNS,
        "minP": 0.01,
        "maxP": 0.99,
        "sampleColumn": None,
        "nMads": 3.0,
        "minCellsPerSample": 20,
    }
    manual = {"method": "manual", "lows": [1, 1, 0, 0], "highs": [1e9, 1e9, 50, 50]}
    plan = pipeline.plan(run, filtering=manual)
    assert plan.config["stages"]["filtering"] == {
        "enabled": True,
        "method": "manual",
        "attrs": QC_COLUMNS,
        "lows": [1, 1, 0, 0],
        "highs": [1e9, 1e9, 50, 50],
        "keepBounds": False,
    }


# Reruns --------------------------------------------------------------------------


def test_a_rerun_runs_the_recipe_with_the_changed_settings(completed, tmp_path) -> None:
    datastore = _copy(completed.location, tmp_path)
    base = datastore.pipeline.open(label="baseline")
    settings = base.settings()
    events: list[PipelineEvent] = []

    fine = datastore.pipeline.rerun(
        base, label="fine", callback=events.append, leiden={"partitions": [2.0]}
    )

    assert (fine.label, fine.recipe, fine.status) == ("fine", "rna", "completed")
    leiden = {"partitions": [2.0], "selected": None}
    assert fine.settings() == {**settings, "leiden": leiden}
    dispositions = _dispositions(fine)
    assert {
        stage
        for stage, disposition in dispositions.items()
        if disposition == "computed"
    } == LEIDEN_STAGES
    assert dispositions["umap"] == "reused"
    assert events[0].kind == "run_started" and events[-1].kind == "run_completed"
    assert {event.run_id for event in events} == {fine.run_id}
    # The earlier run is unchanged.
    assert datastore.pipeline.open(label="baseline").settings() == settings


def test_a_failed_run_runs_again_with_the_failing_setting_changed(
    completed, tmp_path
) -> None:
    datastore = _copy(completed.location, tmp_path)
    base = datastore.pipeline.open(label="baseline")
    # Paris refuses more clusters than the graph's cells only when it reads
    # the graph.
    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.rerun(base, label="retry", paris={"n_clusters": 100_000})
    failed = datastore.pipeline.open(run_id=caught.value.run_id)
    assert (failed.status, caught.value.stage) == ("failed", "paris")

    retried = datastore.pipeline.rerun(failed, label="retry", paris={"n_clusters": 5})

    assert (retried.label, retried.status) == ("retry", "completed")
    assert retried.settings()["paris"] == {"n_clusters": 5}
    dispositions = _dispositions(retried)
    stages = list(dispositions)
    before_paris = stages[: stages.index("paris")]
    assert {dispositions[stage] for stage in before_paris} <= {"reused", "skipped"}


def test_a_rerun_is_refused_before_anything_is_written(
    completed, datastore_zarr_root, tmp_path
) -> None:
    datastore = _copy(completed.location, tmp_path)
    pipeline = datastore.pipeline
    run = pipeline.open(label="baseline")
    before = _digest(tmp_path / "data.zarr")

    with pytest.raises(TypeError, match="run must be a PipelineRun, not str"):
        pipeline.rerun("baseline")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=re.escape("do not include ['nope']")):
        pipeline.rerun(run, nope=1)
    with pytest.raises(TypeError, match=re.escape("do not include ['nope']")):
        pipeline.plan(run, nope=1)
    with pytest.raises(TypeError, match="not positional arguments"):
        pipeline.plan(run, [run])
    with pytest.raises(TypeError, match="takes no callback"):
        pipeline.plan(run, callback=print)
    # A check of the recipe refuses the merged call as it refuses the call.
    with pytest.raises(ValueError, match=re.escape("pca dims (200) must be below")):
        pipeline.rerun(run, pca={"dims": 200})
    # A run of another store is not one of this store's.
    other = tmp_path / "other.zarr"
    shutil.copytree(datastore_zarr_root, other)
    with pytest.raises(ValueError, match="is not recorded in it"):
        DataStore(str(other), default_assay="RNA").pipeline.plan(run)
    readonly = DataStore(str(tmp_path / "data.zarr"), zarr_mode="r")
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        readonly.pipeline.rerun(run, leiden={"partitions": [2.0]})

    assert _digest(tmp_path / "data.zarr") == before
    assert len(pipeline.list_runs(limit=100)) == 1


def test_a_run_of_a_recipe_this_release_lacks_has_no_settings(completed) -> None:
    saved_by_run = _completed_run(_root())

    with pytest.raises(ValueError, match=re.escape("pipeline.run() of an earlier")):
        saved_by_run.settings()
    # A recipe of another release names the recipes of this one.
    record = load_pipeline_run_record(completed.datastore.zw, completed.run.run_id)
    other = dataclasses.replace(record, recipe="spatial")
    with pytest.raises(ValueError) as caught:
        PipelineRun(completed.datastore, other).settings()
    assert str(caught.value) == (
        f"Pipeline run {record.run_id} records the recipe 'spatial', which is not "
        "one of ['adt', 'atac', 'multimodal', 'perturbation', 'rna']"
    )


# Runs of earlier releases -----------------------------------------------------


def _record_as_an_earlier_release(location: Path, run_id: str, edit: Any) -> None:
    """Rewrite a stored run record as an earlier release would have written it."""
    group = zarr.open_group(str(location), mode="r+")[pipeline_run_path(run_id)]
    attrs = dict(group.attrs)
    edit(attrs["config"])
    attrs["cytearcVersion"] = "1.3.5"
    group.attrs.put(attrs)


def test_a_run_recorded_before_a_setting_existed_takes_its_default(
    completed, tmp_path
) -> None:
    location = tmp_path / "data.zarr"
    shutil.copytree(completed.location, location)
    run_id = completed.run.run_id

    def before_localmap(config: dict[str, Any]) -> None:
        # A run recorded before the recipe gained LocalMAP, and before
        # filtering took min_cells_per_sample.
        del config["stages"]["localmap"]
        del config["stages"]["filtering"]["minCellsPerSample"]

    _record_as_an_earlier_release(location, run_id, before_localmap)
    datastore = DataStore(str(location), default_assay="RNA")
    earlier = datastore.pipeline.open(run_id=run_id)

    expected = completed.run.settings()
    del expected["localmap"]
    assert earlier.settings() == expected
    # The defaults reproduce the run, so its plan reuses every stage with the
    # run's artifacts and resolves the settings that a run records today.
    plan = datastore.pipeline.plan(earlier)
    current = completed.datastore.pipeline.plan(completed.run)
    assert [(stage.stage, stage.status, stage.outputs) for stage in plan.stages] == [
        (stage.stage, stage.status, stage.outputs) for stage in current.stages
    ]
    assert plan.computed == ()
    assert plan.config == current.config
    # A call turns the new stage on, and only that stage computes.
    assert datastore.pipeline.plan(earlier, localmap=True).computed == ("localmap",)


def test_a_run_that_records_a_keyword_the_recipe_no_longer_takes_is_refused(
    completed,
) -> None:
    datastore = completed.datastore
    record = load_pipeline_run_record(datastore.zw, completed.run.run_id)
    stages = {**record.config["stages"], "retired": False}
    config = {**record.config, "stages": stages}
    retired = dataclasses.replace(record, config=config, cytearc_version="1.3.5")

    with pytest.raises(ValueError) as caught:
        PipelineRun(datastore, retired).settings()
    assert str(caught.value) == (
        f"Pipeline run {record.run_id}, recorded by CyteArc 1.3.5, records "
        "['retired'], which this release's rna recipe does not take; "
        "report()['run']['config'] holds the settings that it recorded"
    )
