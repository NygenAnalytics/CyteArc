"""The ``leiden`` stage and output of the recipes, beside runs of earlier records.

A recipe clusters its graph at each candidate resolution in one ``leiden``
stage, and ``run["leiden"]`` holds the candidates ordered by resolution.
Earlier releases wrote one ``leiden_<resolution>`` stage, output, and cell
field per candidate; such a run is written here as those releases wrote it,
from the Leiden artifacts that their stages computed, and must still open,
report, plot, export, and give its artifacts to a new run.
"""

import copy
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc import ArtifactRef, DataStore
from cytearc.datastore._pipeline.cluster_selection import run_cluster_selection
from cytearc.datastore._pipeline.fields import _categorical
from cytearc.datastore.pipeline_accessor import PipelineEvent
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.pipeline_runs import (
    PipelineOutputRecord,
    PipelinePlanRecord,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_stage_records,
    pipeline_run_path,
    pipeline_stage_path,
    start_pipeline_stage_record,
)

SETTINGS: dict[str, Any] = {
    "filtering": False,
    "cell_cycle": False,
    "hvg": {"top_n": 50},
    "pca": {"dims": 3},
    "neighbors": {"k": 5},
    "umap": {"n_epochs": 10},
    "doublets": False,
    "markers": False,
}
RESOLUTIONS = (0.5, 1.0)
EARLIER_KEYS = tuple(f"leiden_{resolution}" for resolution in RESOLUTIONS)


def _labels(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(artifact_group(store.zw, ref)["values"][:])


def _cluster_summary(store: DataStore, ref: ArtifactRef) -> dict[str, int]:
    sizes = np.unique(_labels(store, ref), return_counts=True)[1]
    return {
        "clusters": len(sizes),
        "largestCluster": int(sizes.max()),
        "smallestCluster": int(sizes.min()),
    }


def _write_earlier_run(
    store: DataStore,
    base: Any,
    candidates: tuple[ArtifactRef, ...],
) -> str:
    """Record ``base`` with Leiden candidates as releases before one stage did.

    Each candidate has its own ``leiden_<resolution>`` stage, run output, and
    cell field, the silhouette decision names candidates by those keys, and
    no run or stage record holds ``outputLists``.
    """
    root = store.zw
    decision, selected_key, selected_ref = run_cluster_selection(
        store,
        coordinates=base["pca"],
        connectivity_map=base["connectivity_map"],
        cell_selection=base["analysis_cell_selection"],
        candidates=list(zip(EARLIER_KEYS, candidates, strict=True)),
    )
    report = base.report()
    stages = {
        stage.stage: stage for stage in load_pipeline_stage_records(root, base.run_id)
    }
    order: list[str] = []
    for name in report["run"]["stageOrder"]:
        order.extend(EARLIER_KEYS if name == "leiden" else (name,))
    config = copy.deepcopy(report["run"]["config"])
    config["stages"]["leiden"] = {"partitions": list(RESOLUTIONS), "selected": None}
    record = create_pipeline_run_record(
        root,
        recipe="rna",
        requested_label="earlier",
        assay="RNA",
        config=config,
        stage_order=order,
        cytearc_version="1.3.0",
    )
    metrics = stages["input_snapshot"].metrics
    assert metrics is not None
    scores = artifact_group(root, decision)["scores"][:]
    for ordinal, name in enumerate(order):
        start_pipeline_stage_record(
            root, run_id=record.run_id, ordinal=ordinal, stage=name
        )
        if name in EARLIER_KEYS:
            ref = candidates[EARLIER_KEYS.index(name)]
            details: dict[str, Any] = {
                "status": "completed",
                "outputs": (PipelineStageOutputRecord(name, ref, True),),
                "plans": (PipelinePlanRecord("run_leiden_clustering", ref, "reused"),),
                "metrics": metrics,
                "summary": _cluster_summary(store, ref),
            }
        elif name == "cluster_selection":
            details = {
                "status": "completed",
                "outputs": (PipelineStageOutputRecord(name, decision, False),),
                "plans": (
                    PipelinePlanRecord(
                        "select_clusters_by_silhouette", decision, "created"
                    ),
                ),
                "metrics": metrics,
                "summary": {
                    "selected": selected_key,
                    "scores": {
                        key: round(float(score), 4)
                        for key, score in zip(EARLIER_KEYS, scores, strict=True)
                    },
                },
            }
        else:
            stage = stages[name]
            details = {
                "status": stage.status,
                "outputs": stage.outputs,
                "plans": stage.plans,
                "metrics": stage.metrics,
                "summary": stage.summary,
            }
        finish_pipeline_stage_record(
            root, run_id=record.run_id, ordinal=ordinal, **details
        )
    outputs = (
        *(PipelineOutputRecord(key, base[key]) for key in base),
        *(
            PipelineOutputRecord(key, ref)
            for key, ref in zip(EARLIER_KEYS, candidates, strict=True)
        ),
        PipelineOutputRecord("cluster_selection", decision),
        PipelineOutputRecord("clusters", selected_ref),
    )
    fields = base._record.fields
    cells = [field for field in fields if field.axis == "cells"]
    features = [field for field in fields if field.axis == "features"]
    complete_pipeline_run_record(
        root,
        run_id=record.run_id,
        outputs=outputs,
        fields=(
            *cells,
            *(
                _categorical(root, key=key, ref=ref, source="values")
                for key, ref in (
                    *zip(EARLIER_KEYS, candidates, strict=True),
                    ("clusters", selected_ref),
                )
            ),
            *features,
        ),
    )
    # Records of earlier releases hold no output lists.
    for path in (
        pipeline_run_path(record.run_id),
        *(pipeline_stage_path(record.run_id, ordinal) for ordinal in range(len(order))),
    ):
        group = root[path]
        attrs = dict(group.attrs)
        del attrs["outputLists"]
        group.attrs.put(attrs)
    return record.run_id


@pytest.fixture(scope="module")
def runs(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """An earlier run's record and Leiden artifacts, then a new run on the store."""
    location = tmp_path_factory.mktemp("leiden_candidates") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    store = DataStore(str(location), default_assay="RNA")
    base = store.pipeline.rna(leiden=False, **SETTINGS)
    # The artifacts that an earlier release's leiden_<resolution> stages
    # computed: Leiden on the run's graph with the recipe's default settings.
    candidates = tuple(
        store.clusters.leiden(base["connectivity_map"], resolution=resolution)
        for resolution in RESOLUTIONS
    )
    earlier_id = _write_earlier_run(store, base, candidates)
    plan = store.pipeline.plan(
        "rna", leiden={"partitions": list(RESOLUTIONS)}, **SETTINGS
    )
    events: list[PipelineEvent] = []
    new = store.pipeline.rna(
        label="new",
        leiden={"partitions": list(RESOLUTIONS)},
        callback=events.append,
        **SETTINGS,
    )
    return SimpleNamespace(
        store=store,
        candidates=candidates,
        earlier=store.pipeline.open(run_id=earlier_id),
        plan=plan,
        new=new,
        events=events,
    )


# A new run ------------------------------------------------------------------


def test_a_new_run_clusters_every_candidate_in_one_stage_ordered_by_resolution(
    runs: SimpleNamespace,
) -> None:
    store, new = runs.store, runs.new
    report = new.report()
    order = report["run"]["stageOrder"]
    assert order.count("leiden") == 1
    assert not [
        name for name in (*order, *new, *new.cells.columns) if "leiden_" in name
    ]
    candidates = new["leiden"]
    assert type(candidates) is tuple
    assert [
        store.artifacts.inspect(ref).parameters["resolution"] for ref in candidates
    ] == list(RESOLUTIONS)
    # The candidates are found by their parameters.
    for resolution, ref in zip(RESOLUTIONS, candidates, strict=True):
        assert (
            store.artifacts.find(
                "clusters",
                graph=new["connectivity_map"],
                method="leiden",
                resolution=resolution,
            )
            == ref
        )
    # Outputs of one artifact come first, then the candidates, counted once.
    assert list(new)[-1] == "leiden"
    assert len(new) == len(list(new)) == len(set(new))
    assert dict(new.items())["leiden"] == candidates
    assert new.cells.columns == ("I", "ids", "names", "umap_1", "umap_2", "clusters")
    # The run and the stage record hold the candidates as one output list.
    assert report["run"]["outputLists"] == [
        {"key": "leiden", "artifacts": [ref.to_dict() for ref in candidates]}
    ]
    stage = next(item for item in report["stages"] if item["stage"] == "leiden")
    assert stage["outputs"] == []
    assert [
        ArtifactRef.from_dict(item["artifact"])
        for item in stage["outputLists"][0]["artifacts"]
    ] == list(candidates)
    assert stage["summary"] == {
        "candidates": [
            {
                "resolution": resolution,
                **_cluster_summary(store, ref),
                "artifact": ref.artifact_id,
            }
            for resolution, ref in zip(RESOLUTIONS, candidates, strict=True)
        ]
    }
    markdown = new.report(format="markdown")
    assert (
        "- `leiden`: " + ", ".join(f"`{ref.artifact_id}`" for ref in candidates)
        in markdown
    )


def test_cluster_selection_names_each_candidate_by_its_resolution(
    runs: SimpleNamespace,
) -> None:
    store, new = runs.store, runs.new
    candidates = new["leiden"]
    decision = artifact_group(store.zw, new["cluster_selection"])
    assert decision.attrs["candidateKeys"] == ["0.5", "1.0"]
    assert [ArtifactRef.from_dict(ref) for ref in decision.attrs["candidateRefs"]] == (
        list(candidates)
    )
    selected = ["0.5", "1.0"].index(decision.attrs["selectedKey"])
    assert new["clusters"] == candidates[selected]
    np.testing.assert_array_equal(
        new.cells.fetch("clusters"), _labels(store, candidates[selected])
    )
    scores = decision["scores"][:]
    summary = next(
        stage["summary"]
        for stage in new.report()["stages"]
        if stage["stage"] == "cluster_selection"
    )
    assert summary == {
        "selected": RESOLUTIONS[selected],
        "candidates": [
            {
                "resolution": resolution,
                "score": round(float(score), 4),
                "artifact": ref.artifact_id,
            }
            for resolution, score, ref in zip(
                RESOLUTIONS, scores, candidates, strict=True
            )
        ],
    }


def test_the_leiden_stage_event_gives_one_pair_per_candidate(
    runs: SimpleNamespace,
) -> None:
    (event,) = (
        event
        for event in runs.events
        if event.kind == "stage_completed" and event.stage == "leiden"
    )
    candidates = runs.new["leiden"]
    assert event.outputs == tuple(("leiden", ref) for ref in candidates)
    # The earlier run's artifacts were reused, every one of them.
    assert event.reused == ("leiden",)
    assert [(item["key"], item["reused"]) for item in event.to_dict()["outputs"]] == [
        ("leiden", True),
        ("leiden", True),
    ]


def test_a_new_candidate_beside_reused_ones_is_recorded_as_created(
    runs: SimpleNamespace,
) -> None:
    store, new = runs.store, runs.new
    events: list[PipelineEvent] = []
    # The call lists its resolutions out of order; the run orders them.
    run = store.pipeline.rna(
        leiden={"partitions": [1.0, 0.75, 0.5]}, callback=events.append, **SETTINGS
    )

    candidates = run["leiden"]
    assert run.report()["run"]["config"]["stages"]["leiden"]["partitions"] == [
        0.5,
        0.75,
        1.0,
    ]
    assert (candidates[0], candidates[2]) == new["leiden"]
    record = next(
        stage
        for stage in load_pipeline_stage_records(store.zw, run.run_id)
        if stage.stage == "leiden"
    )
    (listed,) = record.output_lists
    assert listed.reused == (True, False, True)
    assert [plan.disposition for plan in record.plans] == [
        "reused",
        "created",
        "reused",
    ]
    (event,) = (
        event
        for event in events
        if event.kind == "stage_completed" and event.stage == "leiden"
    )
    # An output counts as reused only when every artifact was reused.
    assert event.reused == ()
    assert [ref for _key, ref in event.outputs] == list(candidates)


def test_run_plots_color_by_any_candidate_and_refuse_the_list_as_a_layout(
    runs: SimpleNamespace,
) -> None:
    store, new = runs.store, runs.new
    other = next(ref for ref in new["leiden"] if ref != new["clusters"])
    result = store.plots.embedding(run=new, color_by=other, show=False)
    try:
        assert result.provenance.extras["color_artifacts"] == [other.to_dict()]
    finally:
        result.close()
    foreign = store.clusters.leiden(new["connectivity_map"], resolution=2.0)
    with pytest.raises(ValueError, match="only by outputs of this run"):
        store.plots.embedding(run=new, color_by=foreign, show=False)
    with pytest.raises(ValueError, match="'leiden' holds several artifacts"):
        store.plots.embedding(run=new, layout="leiden", show=False)


def test_the_lineage_of_a_run_names_each_candidate_by_its_position(
    runs: SimpleNamespace,
) -> None:
    store, new = runs.store, runs.new
    outputs = store.artifacts.lineage(new).outputs
    assert outputs["leiden[0]"] == new["leiden"][0]
    assert outputs["leiden[1]"] == new["leiden"][1]
    assert "leiden" not in outputs
    with pytest.raises(ValueError, match="also the name of a listed ref"):
        store.artifacts.lineage({"leiden": new["leiden"], "leiden[0]": new["clusters"]})
    with pytest.raises(TypeError, match="non-empty tuple"):
        store.artifacts.lineage({"leiden": ()})


def test_a_new_run_exports_only_its_saved_clustering(
    runs: SimpleNamespace,
) -> None:
    pytest.importorskip("anndata")
    exported = runs.store.to_anndata(run=runs.new)
    assert "clusters" in exported.obs
    assert not [column for column in exported.obs if column.startswith("leiden")]


# An earlier run ----------------------------------------------------------


def test_an_earlier_run_opens_with_its_own_output_names(runs: SimpleNamespace) -> None:
    earlier = runs.earlier
    assert earlier.label == "earlier"
    assert "outputLists" not in runs.store.zw[pipeline_run_path(earlier.run_id)].attrs
    keys = list(earlier)
    assert keys[-4:] == [*EARLIER_KEYS, "cluster_selection", "clusters"]
    assert len(earlier) == len(keys)
    assert (earlier["leiden_0.5"], earlier["leiden_1.0"]) == runs.candidates
    # No list is made up for a record that holds none.
    assert "leiden" not in earlier
    with pytest.raises(KeyError):
        earlier["leiden"]
    assert earlier["clusters"] in runs.candidates
    assert earlier.cells.columns[-3:] == (*EARLIER_KEYS, "clusters")
    for key, ref in zip(EARLIER_KEYS, runs.candidates, strict=True):
        np.testing.assert_array_equal(
            earlier.cells.fetch(key), _labels(runs.store, ref)
        )


def test_an_earlier_run_reports_its_stages_and_outputs(runs: SimpleNamespace) -> None:
    earlier = runs.earlier
    report = earlier.report()
    order = report["run"]["stageOrder"]
    assert [name for name in order if name.startswith("leiden")] == list(EARLIER_KEYS)
    assert report["run"]["outputLists"] == []
    stages = {stage["stage"]: stage for stage in report["stages"]}
    for key, ref in zip(EARLIER_KEYS, runs.candidates, strict=True):
        assert stages[key]["outputLists"] == []
        assert stages[key]["summary"] == _cluster_summary(runs.store, ref)
    assert set(stages["cluster_selection"]["summary"]["scores"]) == set(EARLIER_KEYS)
    markdown = earlier.report(format="markdown")
    for key, ref in zip(EARLIER_KEYS, runs.candidates, strict=True):
        assert f"| {key} | completed |" in markdown
        assert f"- `{key}`: `{ref.artifact_id}`" in markdown


def test_an_earlier_run_plots_exports_and_traces_its_candidates(
    runs: SimpleNamespace,
) -> None:
    store, earlier = runs.store, runs.earlier
    result = store.plots.embedding(
        run=earlier, color_by=["leiden_0.5", earlier["leiden_1.0"]], show=False
    )
    try:
        assert result.provenance.extras["color_artifacts"] == [
            earlier["leiden_1.0"].to_dict()
        ]
    finally:
        result.close()
    lineage = store.artifacts.lineage(earlier)
    assert lineage.outputs["leiden_0.5"] == earlier["leiden_0.5"]
    pytest.importorskip("anndata")
    exported = store.to_anndata(run=earlier)
    for key in EARLIER_KEYS:
        column = exported.obs[key]
        assert isinstance(column.dtype, pd.CategoricalDtype)
        assert column.astype(str).tolist() == [
            str(value) for value in earlier.cells.fetch(key)
        ]


def test_a_new_run_reuses_the_leiden_artifacts_of_an_earlier_run(
    runs: SimpleNamespace,
) -> None:
    store, new, plan = runs.store, runs.new, runs.plan
    # The plan shows one Leiden stage, which reuses the earlier artifacts.
    assert [stage.stage for stage in plan.stages].count("leiden") == 1
    assert not [stage.stage for stage in plan.stages if "leiden_" in stage.stage]
    assert plan["leiden"].status == "reused"
    assert plan["leiden"].outputs == tuple(("leiden", ref) for ref in runs.candidates)
    # The decision names candidates by resolution now, so it is computed once.
    assert plan["cluster_selection"].status == "computed"
    assert new["leiden"] == runs.candidates
    record = next(
        stage
        for stage in load_pipeline_stage_records(store.zw, new.run_id)
        if stage.stage == "leiden"
    )
    assert [plan.disposition for plan in record.plans] == ["reused", "reused"]
    assert new["clusters"] in runs.candidates
    # The identity that both releases give a candidate: its graph and these
    # parameters, which a change to the recipe's Leiden settings would break.
    for resolution, ref in zip(RESOLUTIONS, runs.candidates, strict=True):
        status = store.artifacts.inspect(ref)
        assert status.operation == "run_leiden_clustering"
        assert status.parameters == {
            "resolution": resolution,
            "backend": "cytearc",
            "edge_weighting": "graph",
            "symmetric_graph": False,
            "graph_upper_only": False,
            "random_seed": 4444,
        }
        assert status.input_ref("graph") == new["connectivity_map"]


def test_a_decision_summary_without_candidates_or_resolutions_says_so() -> None:
    import zarr
    from zarr.storage import MemoryStore

    from cytearc.datastore._pipeline.summaries import _cluster_selection

    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("decision")
    # A decision that records no candidates has nothing to summarize.
    assert _cluster_selection(root, group) == {}
    absent = ArtifactRef(
        scope="assay", assay="RNA", kind="cluster_labels", artifact_id="a" * 64
    )
    group.attrs.update(
        {
            "candidateKeys": ["0.5"],
            "candidateRefs": [absent.to_dict()],
            "selectedKey": "1.0",
        }
    )
    # A candidate without a recorded resolution, a selected key that names no
    # candidate, and a decision without scores leave those values None.
    assert _cluster_selection(root, group) == {
        "selected": None,
        "candidates": [{"resolution": None, "score": None, "artifact": "a" * 64}],
    }
