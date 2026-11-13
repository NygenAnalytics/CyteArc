"""The plan of a recipe call: what each stage would reuse or compute, and why.

A module fixture commits one RNA run on a private copy of the PBMC store.
Tests that change the store, or run a call to compare with its plan, copy
that store, so their runs reuse the committed run's artifacts.
"""

import hashlib
import re
import shutil
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

from cytearc import CancellationToken, DataStore, PipelineRun
from cytearc.assay import norm_lib_size, norm_pearson_residuals
from cytearc.datastore._pipeline.executor import (
    COORDINATES,
    PlannedStage,
    RecipePlan,
    RunState,
)
from cytearc.datastore._pipeline.planning import plan_recipe
from cytearc.datastore.pipeline_accessor import PipelinePlan
from cytearc.storage import operation_revisions
from cytearc.storage.artifact_writer import (
    ArtifactComputation,
    ArtifactComputationRequired,
)
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.diagnostics import DIAGNOSTICS_ATTRIBUTE
from cytearc.storage.pipeline_runs import (
    load_pipeline_stage_records,
    pipeline_stage_path,
)
from tests.test_metadata_store_reads import _CountingStore

# A run of every stage after the graph but Paris, membership strength, and
# t-SNE, kept small: 200 genes, 5 PCs, and two Leiden candidates.
SETTINGS: dict[str, Any] = {
    "cell_cycle": False,
    "hvg": {"top_n": 200},
    "pca": {"dims": 5},
    "neighbors": {"k": 5},
    "umap": {"n_epochs": 30},
    "leiden": {"partitions": [0.5, 1.0]},
    "doublets": True,
    "markers": True,
}


@pytest.fixture(scope="module")
def completed(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Commit the run of ``SETTINGS`` once on a private copy of the PBMC store."""
    location = tmp_path_factory.mktemp("plan_completed") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    datastore = DataStore(str(location), default_assay="RNA")
    run = datastore.pipeline.rna(label="baseline", **SETTINGS)
    return SimpleNamespace(location=location, datastore=datastore, run=run)


def _copy(location: Path, tmp_path: Path) -> DataStore:
    target = tmp_path / "data.zarr"
    shutil.copytree(location, target)
    return DataStore(str(target), default_assay="RNA")


# A chunk of an assay's count matrix, or of its transpose.
_COUNT_CHUNK = re.compile(r"^[^/]+/countsT?/c/")
_WRITES = {"set", "set_if_not_exists", "delete"}


def read_only_plan(
    location: Path | str,
    recipe: str,
    *args: Any,
    default_assay: str | None = None,
    **settings: Any,
) -> PipelinePlan:
    """Plan a call on a read-only store that records each request it receives.

    The plan must neither write nor read a chunk of a count matrix.
    """
    store = _CountingStore(LocalStore(str(location), read_only=True))
    datastore = DataStore(store, default_assay=default_assay, zarr_mode="r")
    store.log.clear()
    plan = datastore.pipeline.plan(recipe, *args, **settings)
    assert not [entry for entry in store.log if entry[0] in _WRITES]
    chunks = [
        key for operation, key in store.log if operation == "get" and "/c/" in key
    ]
    assert not [key for key in chunks if _COUNT_CHUNK.match(key)]
    return plan


def _digest(location: Path) -> str:
    """Return a digest of every file's path and bytes below ``location``."""
    digest = hashlib.sha256()
    for path in sorted(location.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(location)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _dispositions(run: PipelineRun) -> dict[str, str]:
    """Return what each stage of a run did, as its stage record shows it.

    A stage that created an artifact computed, and one whose every plan was
    reused reused.
    """
    dispositions = {}
    for record in run.report()["stages"]:
        if record["status"] == "skipped":
            dispositions[record["stage"]] = "skipped"
        elif any(plan["disposition"] == "created" for plan in record["plans"]):
            dispositions[record["stage"]] = "computed"
        else:
            dispositions[record["stage"]] = "reused"
    return dispositions


def assert_plan_reuses_run(plan: PipelinePlan, run: PipelineRun) -> None:
    """Check that a plan reuses every stage of ``run`` with the refs it recorded."""
    report = run.report()
    records = {record["stage"]: record for record in report["stages"]}
    assert [stage.stage for stage in plan.stages] == list(report["run"]["stageOrder"])
    for stage in plan.stages:
        record = records[stage.stage]
        if record["status"] == "skipped":
            assert (stage.status, stage.outputs) == ("skipped", ()), stage.stage
            continue
        assert (stage.status, stage.reason, stage.estimate) == ("reused", None, None)
        assert stage.outputs == tuple(
            (output["outputKey"], ArtifactRef.from_dict(output["artifact"]))
            for output in record["outputs"]
        ), stage.stage
    assert plan.computed == ()
    config = report["run"]["config"]
    assert plan.config == {
        key: value for key, value in config.items() if key != "shutdown"
    }


def _later(plan: PipelinePlan, stage: str) -> list[Any]:
    """Return the plans of the stages after ``stage`` that the call runs."""
    names = [planned.stage for planned in plan.stages]
    return [
        planned
        for planned in plan.stages[names.index(stage) + 1 :]
        if planned.status != "skipped"
    ]


def _earlier(plan: PipelinePlan, stage: str) -> list[Any]:
    names = [planned.stage for planned in plan.stages]
    return list(plan.stages[: names.index(stage)])


def test_a_completed_runs_plan_reuses_every_stage_with_its_refs(completed) -> None:
    plan = completed.datastore.pipeline.plan("rna", **SETTINGS)

    assert_plan_reuses_run(plan, completed.run)
    assert (plan.recipe, plan.assay, plan.label) == ("rna", "RNA", None)
    assert plan.reused == tuple(
        stage.stage for stage in plan.stages if stage.status != "skipped"
    )
    assert plan["umap"].outputs == (("umap", completed.run["umap"]),)
    with pytest.raises(KeyError, match="The rna plan has no stage 'tsne_2'"):
        plan["tsne_2"]
    assert plan.to_dict()["stages"][0] == {
        "stage": "input_snapshot",
        "ordinal": 0,
        "status": "reused",
        "reason": None,
        "detail": None,
        "operation": None,
        "outputs": [
            {"key": key, "artifact": ref.to_dict()}
            for key, ref in plan["input_snapshot"].outputs
        ],
        "existing": None,
        "changedParameters": [],
        "changedInputs": [],
        "upstream": [],
        "estimate": None,
    }


def test_a_plan_writes_nothing_reads_no_counts_and_opens_read_only(
    completed, tmp_path
) -> None:
    datastore = _copy(completed.location, tmp_path)
    location = tmp_path / "data.zarr"
    before = _digest(location)
    changed = {**SETTINGS, "hvg": {"top_n": 150}}

    statuses = [
        [stage.status for stage in datastore.pipeline.plan("rna", **call).stages]
        for call in (SETTINGS, changed)
    ]

    assert _digest(location) == before
    for call, expected in zip((SETTINGS, changed), statuses, strict=True):
        plan = read_only_plan(location, "rna", default_assay="RNA", **call)
        assert [stage.status for stage in plan.stages] == expected
    # The recording store sees each request, so a plan that wrote or read a
    # count chunk would fail the checks above. Reuse checks read small arrays
    # of the stored artifacts.
    store = _CountingStore(LocalStore(str(location), read_only=True))
    readonly = DataStore(store, default_assay="RNA", zarr_mode="r")
    store.log.clear()
    readonly.pipeline.plan("rna", **SETTINGS)
    assert any("/artifacts/" in key and "/c/" in key for _operation, key in store.log)
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        readonly.pipeline.rna(**SETTINGS)


def test_a_changed_setting_computes_its_stage_and_every_stage_that_reads_it(
    completed, tmp_path
) -> None:
    changed = {**SETTINGS, "hvg": {"top_n": 150}}

    plan = completed.datastore.pipeline.plan("rna", **changed)

    hvg = plan["highly_variable_features"]
    assert (hvg.status, hvg.reason, hvg.operation) == (
        "computed",
        "changed",
        "select_hvgs",
    )
    assert hvg.changed_parameters == ("top_n",)
    assert hvg.changed_inputs == ()
    assert hvg.existing == completed.run["highly_variable_features"]
    assert hvg.detail == "The closest select_hvgs artifact differs in parameters top_n"
    assert {stage.status for stage in _earlier(plan, "highly_variable_features")} == {
        "reused",
        "skipped",
    }
    later = _later(plan, "highly_variable_features")
    assert {(stage.status, stage.reason) for stage in later} == {
        ("computed", "upstream")
    }
    # Each stage waits on the stages whose outputs it reads.
    assert {stage.stage: stage.upstream for stage in later} == {
        "normalization": ("highly_variable_features",),
        "pca": ("normalization",),
        "ann_index": ("pca",),
        "neighbors": ("pca", "ann_index"),
        "connectivity": ("neighbors",),
        "embedding_initialization": ("pca",),
        "umap": ("connectivity", "embedding_initialization"),
        "leiden_0.5": ("connectivity",),
        "leiden_1.0": ("connectivity",),
        "cluster_selection": ("pca", "connectivity", "leiden_0.5", "leiden_1.0"),
        "doublets": ("connectivity", "cluster_selection"),
        "markers": ("cluster_selection",),
    }
    assert plan["pca"].detail == (
        "It reads what normalization would compute, so its identity is known "
        "only after they run"
    )
    # The run that follows computes what the plan computes and reuses the rest.
    run = _copy(completed.location, tmp_path).pipeline.rna(**changed)
    assert _dispositions(run) == {stage.stage: stage.status for stage in plan.stages}


def test_a_stage_that_reads_no_changed_value_stays_reused(completed, tmp_path) -> None:
    changed = {**SETTINGS, "umap": {"n_epochs": 40}}

    plan = completed.datastore.pipeline.plan("rna", **changed)

    assert plan.computed == ("umap",)
    umap = plan["umap"]
    assert (umap.reason, umap.changed_parameters, umap.existing) == (
        "changed",
        ("n_epochs",),
        completed.run["umap"],
    )
    # The Leiden candidates and the stages after them read the graph, not UMAP.
    assert plan["leiden_0.5"].outputs == (("leiden_0.5", completed.run["leiden_0.5"]),)
    assert plan["markers"].status == "reused"
    run = _copy(completed.location, tmp_path).pipeline.rna(**changed)
    assert _dispositions(run) == {stage.stage: stage.status for stage in plan.stages}


def test_a_revision_that_supersedes_a_stored_artifact_is_the_reason(
    completed, monkeypatch
) -> None:
    revision = operation_revisions.OperationRevision(
        revision=len(operation_revisions.OPERATION_REVISIONS["run_pca"]) + 2,
        release="9.9.9",
        change="A test revision of every PCA",
        applies=None,
    )
    monkeypatch.setattr(
        operation_revisions,
        "OPERATION_REVISIONS",
        MappingProxyType(
            {
                **operation_revisions.OPERATION_REVISIONS,
                "run_pca": (
                    *operation_revisions.OPERATION_REVISIONS["run_pca"],
                    revision,
                ),
            }
        ),
    )

    plan = completed.datastore.pipeline.plan("rna", **SETTINGS)

    pca = plan["pca"]
    assert (pca.status, pca.reason, pca.operation, pca.existing) == (
        "computed",
        "superseded",
        "run_pca",
        completed.run["pca"],
    )
    assert pca.detail == (
        "A revision supersedes the stored run_pca artifact: A test revision of "
        "every PCA"
    )
    assert {stage.status for stage in _earlier(plan, "pca")} == {"reused", "skipped"}
    assert {stage.reason for stage in _later(plan, "pca")} == {"upstream"}


def test_an_artifact_without_its_diagnostics_is_the_reason(completed, tmp_path) -> None:
    datastore = _copy(completed.location, tmp_path)
    # A PCA saved before CyteArc recorded its diagnostics holds none.
    del artifact_group(datastore.zw, completed.run["pca"]).attrs[DIAGNOSTICS_ATTRIBUTE]

    plan = datastore.pipeline.plan("rna", **SETTINGS)

    pca = plan["pca"]
    assert (pca.status, pca.reason, pca.existing) == (
        "computed",
        "missing_diagnostics",
        completed.run["pca"],
    )
    assert pca.detail == (
        "The stored run_pca artifact lacks the diagnostics that run_pca saves"
    )
    assert {stage.reason for stage in _later(plan, "pca")} == {"upstream"}


def test_a_fresh_store_computes_every_stage(datastore_zarr_root: str) -> None:
    datastore = DataStore(datastore_zarr_root, default_assay="RNA", zarr_mode="r")

    plan = datastore.pipeline.plan("rna", label="fresh", **SETTINGS)

    first = plan["input_snapshot"]
    assert (first.status, first.reason, first.operation, first.existing) == (
        "computed",
        "absent",
        "snapshot_pipeline_input_selection",
        None,
    )
    assert first.detail == (
        "The store holds no snapshot_pipeline_input_selection artifact"
    )
    assert plan.label == "fresh"
    assert plan.reused == ()
    assert {
        (stage.reason, stage.estimate) for stage in _later(plan, "input_snapshot")
    } == {("upstream", None)}
    assert plan["filtering"].upstream == ("input_snapshot",)


def test_estimates_come_from_the_latest_run_that_computed_each_stage(
    completed, tmp_path
) -> None:
    datastore = _copy(completed.location, tmp_path)
    # A second run of the same call reuses every artifact, so its records do
    # not measure a computation.
    again = datastore.pipeline.rna(**SETTINGS)
    assert set(_dispositions(again).values()) == {"reused", "skipped"}
    records = {
        record.stage: record.metrics
        for record in load_pipeline_stage_records(datastore.zw, completed.run.run_id)
    }

    plan = datastore.pipeline.plan(
        "rna", **{**SETTINGS, "hvg": {"top_n": 150}, "paris": True}
    )

    for stage in ("highly_variable_features", "normalization", "markers"):
        estimate = plan[stage].estimate
        metrics = records[stage]
        assert estimate is not None and metrics is not None
        assert metrics.resources is not None
        assert (
            estimate.run_id,
            estimate.wall_seconds,
            estimate.rss_peak_bytes,
            estimate.rss_incremental_peak_bytes,
            estimate.cpu_seconds,
            estimate.threads_peak,
            estimate.network_rx_bytes,
            estimate.network_tx_bytes,
            estimate.resources,
        ) == (
            completed.run.run_id,
            metrics.wall_seconds,
            metrics.rss_peak_bytes,
            metrics.rss_incremental_peak_bytes,
            metrics.cpu_seconds,
            metrics.threads_peak,
            metrics.network_rx_bytes,
            metrics.network_tx_bytes,
            metrics.resources.to_dict(),
        )
        assert estimate.resources["workers"] == completed.datastore.nthreads
        start = metrics.rss_high_water_start_bytes
        end = metrics.rss_high_water_end_bytes
        assert start is not None and end is not None
        # The process's peak counts only from a stage that raised it.
        assert estimate.rss_high_water_bytes == (end if end > start else None)
    hvg = records["highly_variable_features"]
    assert hvg.resources is not None
    assert plan["highly_variable_features"].to_dict()["estimate"] == {
        "runId": completed.run.run_id,
        "wallSeconds": hvg.wall_seconds,
        "rssPeakBytes": hvg.rss_peak_bytes,
        "rssIncrementalPeakBytes": hvg.rss_incremental_peak_bytes,
        "cpuSeconds": hvg.cpu_seconds,
        "threadsPeak": hvg.threads_peak,
        "rssHighWaterBytes": plan[
            "highly_variable_features"
        ].estimate.rss_high_water_bytes,
        "networkRxBytes": hvg.network_rx_bytes,
        "networkTxBytes": hvg.network_tx_bytes,
        "resources": hvg.resources.to_dict(),
    }
    # No run computed Paris, and reused stages carry no estimate.
    assert plan["paris"].status == "computed"
    assert plan["paris"].estimate is None
    assert {stage.estimate for stage in plan.stages if stage.status == "reused"} == {
        None
    }


def test_estimates_from_records_without_resource_use_hold_none(
    completed, tmp_path
) -> None:
    datastore = _copy(completed.location, tmp_path)
    # Strip what an earlier release did not record from every stage record.
    resource_fields = (
        "cpuSeconds",
        "threadsPeak",
        "rssHighWaterStartBytes",
        "rssHighWaterEndBytes",
        "networkRxBytes",
        "networkTxBytes",
        "resources",
        "trace",
        "unavailable",
    )
    stage_order = completed.run.report()["run"]["stageOrder"]
    for ordinal in range(len(stage_order)):
        group = datastore.zw[pipeline_stage_path(completed.run.run_id, ordinal)]
        attrs = dict(group.attrs)
        attrs["metrics"] = {
            key: value
            for key, value in attrs["metrics"].items()
            if key not in resource_fields
        }
        group.attrs.put(attrs)

    plan = datastore.pipeline.plan("rna", **{**SETTINGS, "hvg": {"top_n": 150}})

    estimate = plan["highly_variable_features"].estimate
    assert estimate is not None and estimate.wall_seconds > 0
    assert (
        estimate.cpu_seconds,
        estimate.threads_peak,
        estimate.rss_high_water_bytes,
        estimate.network_rx_bytes,
        estimate.network_tx_bytes,
        estimate.resources,
    ) == (None,) * 6


def test_a_plan_checks_the_call_as_the_recipe_does(completed) -> None:
    pipeline = completed.datastore.pipeline

    with pytest.raises(ValueError, match=r"recipe must be one of \['adt', 'atac'"):
        pipeline.plan("scrna")
    with pytest.raises(ValueError, match="recipe must be one of"):
        pipeline.plan(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="takes no callback or cancel"):
        pipeline.plan("rna", callback=print, cancel=CancellationToken())
    with pytest.raises(TypeError, match="unexpected keyword argument 'hvgs'"):
        pipeline.plan("rna", hvgs={"top_n": 10})
    with pytest.raises(TypeError, match="too many positional arguments"):
        pipeline.plan("rna", [completed.run])
    # The label and the settings are checked as the recipe checks them.
    with pytest.raises(ValueError, match="'baseline' is already committed by run"):
        pipeline.plan("rna", label="baseline", **SETTINGS)
    message = "doublet scoring needs a PCA graph"
    with pytest.raises(ValueError, match=message):
        pipeline.rna(**{**SETTINGS, "pca": {"dims": 0}})
    with pytest.raises(ValueError, match=message):
        pipeline.plan("rna", **{**SETTINGS, "pca": {"dims": 0}})


def test_a_run_and_its_plan_refuse_doublets_without_library_size_before_writing(
    completed, tmp_path
) -> None:
    datastore = _copy(completed.location, tmp_path)
    location = tmp_path / "data.zarr"
    datastore.RNA.normMethod = norm_pearson_residuals
    message = (
        "Assay 'RNA' normalizes with PearsonResidualNormalizer, but doublet "
        "scoring normalizes its simulated doublets by library size and needs "
        "norm_lib_size; pass doublets=False or set the assay's normMethod to "
        "cytearc.assay.norm_lib_size"
    )
    before = _digest(location)

    with pytest.raises(ValueError) as planned:
        datastore.pipeline.plan("rna", **SETTINGS)
    with pytest.raises(ValueError) as ran:
        datastore.pipeline.rna(**SETTINGS)

    assert str(planned.value) == message
    assert str(ran.value) == message
    # Neither call wrote a run record, an artifact, or any other file.
    assert _digest(location) == before
    assert [run.run_id for run in datastore.pipeline.list_runs()] == [
        completed.run.run_id
    ]
    # Without doublet scoring the plan goes ahead with the residuals: gene
    # selection, whose statistics record the normalizer, is the first stage
    # that changes.
    plan = datastore.pipeline.plan("rna", **{**SETTINGS, "doublets": False})
    hvg = plan["highly_variable_features"]
    assert (hvg.status, hvg.reason) == ("computed", "changed")
    assert hvg.changed_parameters == ("normalization_method",)
    assert plan["normalization"].upstream == ("highly_variable_features",)
    assert plan["doublets"].status == "skipped"
    # With the library-size default back, the same call runs and reuses the
    # committed doublet scores.
    datastore.RNA.normMethod = norm_lib_size
    run = datastore.pipeline.rna(**SETTINGS)
    assert run.status == "completed"
    assert run["doublets"] == completed.run["doublets"]


# The planner over synthetic stages -------------------------------------------


def _ref(index: int) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore", kind="cell_selection", artifact_id=f"{index:064x}"
    )


def _computes(state: RunState) -> Any:
    raise ArtifactComputationRequired(
        ArtifactComputation(
            operation="test_operation", kind="normalized", reason="absent"
        )
    )


def _plan(*stages: PlannedStage) -> PipelinePlan:
    """Plan synthetic stages on an empty store."""
    store = SimpleNamespace(zw=zarr.open_group(store=MemoryStore(), mode="w"))
    recipe = RecipePlan(
        recipe="rna",
        assay="RNA",
        label=None,
        config={"stages": {}},
        stages=stages,
        output_keys=(),
        fields=lambda state: (),
        start=lambda store: RunState(store=store, assay=None, coordinates=_ref(0)),
    )
    return plan_recipe(store, recipe)


def test_a_skipped_stage_passes_on_the_stage_that_its_alias_waits_on() -> None:
    def alias(state: RunState) -> None:
        state.artifacts["alias"] = state.artifacts["first"]

    plan = _plan(
        PlannedStage("first", _computes, outputs=("first", COORDINATES)),
        PlannedStage(
            "skipped", None, on_skip=alias, inputs=("first",), outputs=("alias",)
        ),
        PlannedStage(
            "reader",
            lambda state: (("read", _ref(1)),),
            inputs=("alias",),
            outputs=("read",),
        ),
        PlannedStage(
            "coordinates",
            lambda state: (("other", _ref(2)),),
            inputs=(COORDINATES,),
            outputs=("other",),
        ),
        PlannedStage(
            "independent", lambda state: (("own", _ref(3)),), outputs=("own",)
        ),
    )

    assert [(stage.stage, stage.status, stage.upstream) for stage in plan.stages] == [
        ("first", "computed", ()),
        ("skipped", "skipped", ()),
        ("reader", "computed", ("first",)),
        ("coordinates", "computed", ("first",)),
        ("independent", "reused", ()),
    ]
    assert plan["first"].detail == "The store holds no test_operation artifact"
    assert plan["independent"].outputs == (("own", _ref(3)),)


def test_a_computed_stage_leaves_no_earlier_value_in_its_place() -> None:
    seen = []

    def undeclared(state: RunState) -> Any:
        # Reading the coordinates without declaring them gets no stale value.
        seen.append(state.coordinates)
        return ()

    _plan(
        PlannedStage("first", _computes, outputs=(COORDINATES,)),
        PlannedStage("reader", undeclared),
    )

    assert seen == [None]


def test_a_stage_that_uses_an_undeclared_artifact_key_is_refused() -> None:
    def reads(state: RunState) -> Any:
        return (("own", state.artifacts["first"]),)

    def writes(state: RunState) -> Any:
        state.artifacts["stray"] = _ref(2)
        return ()

    first = PlannedStage(
        "first", lambda state: (("first", _ref(1)),), outputs=("first",)
    )

    with pytest.raises(
        RuntimeError,
        match="'reader' uses 'first', which it does not declare among its inputs",
    ):
        _plan(first, PlannedStage("reader", reads, outputs=("own",)))
    with pytest.raises(
        RuntimeError,
        match="'reader' uses 'first', which it does not declare among its inputs",
    ):
        _plan(first, PlannedStage("reader", lambda state: state.artifacts.get("first")))
    with pytest.raises(
        RuntimeError,
        match="'writer' uses 'stray', which it does not declare among its outputs",
    ):
        _plan(first, PlannedStage("writer", writes))
    with pytest.raises(
        RuntimeError, match=r"'first' returned \['first'\], which it does not declare"
    ):
        _plan(PlannedStage("first", lambda state: (("first", _ref(1)),)))
