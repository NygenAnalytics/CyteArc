"""Pause a pipeline run after a stage and resume it under the same run ID.

A module fixture pauses one labeled RNA run after ``pca`` on a private copy of
the PBMC store; tests that resume it copy that store. The checks compare a
paused and resumed run with a single call: on the same store, the single call
must reuse every artifact of the resumed run, and on another copy of the store
it must compute what the resumed run computed.
"""

import concurrent.futures
import inspect
import json
import shutil
import subprocess
import sys
import textwrap
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

import cytearc.datastore._pipeline.executor as executor_module
import cytearc.datastore.pipeline_accessor as accessor_module
import cytearc.storage.pipeline_runs as pipeline_run_storage
from cytearc import (
    CancellationToken,
    DataStore,
    PipelineCancelled,
    PipelineExecutionError,
)
from cytearc.assay import norm_lib_size
from cytearc.datastore._pipeline.recipes.rna import resolve_rna
from cytearc.datastore.pipeline_accessor import PipelineAccessor, recorded_settings
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.pipeline_runs import (
    create_pipeline_run_record,
    fail_pipeline_run_record,
    interrupt_pipeline_run_record,
    load_pipeline_run_record,
    load_pipeline_stage_records,
    pipeline_run_path,
    pipeline_stage_path,
)
from cytearc.tools.repack_zarr import repack_store
from tests.test_pipeline_plan import _dispositions
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
_SHUTDOWN = {"signalGuardAvailable": True, "unavailableReason": None}


def _copy(source: Path | str, folder: Path) -> DataStore:
    target = folder / "data.zarr"
    shutil.copytree(source, target)
    return DataStore(str(target), default_assay="RNA")


@pytest.fixture(scope="module")
def paused(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Pause a labeled RNA run after ``pca`` once, on a private copy of the PBMC store."""
    location = tmp_path_factory.mktemp("resume_paused") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    store = DataStore(str(location), default_assay="RNA")
    events: list[tuple[str, str | None]] = []
    run = store.pipeline.rna(
        label="split",
        run_until="pca",
        callback=lambda event: events.append((event.kind, event.stage)),
        **SETTINGS,
    )
    return SimpleNamespace(location=location, run_id=run.run_id, run=run, events=events)


def _paused_copy(paused: SimpleNamespace, tmp_path: Path) -> DataStore:
    return _copy(paused.location, tmp_path)


def _runs(store: DataStore) -> list[str]:
    return sorted(run.run_id for run in store.pipeline.list_runs(limit=100))


def _stage_names(store: DataStore, run_id: str) -> list[str]:
    return [record.stage for record in load_pipeline_stage_records(store.zw, run_id)]


def _running_stages(store: DataStore) -> list[str]:
    """The stages that a call of ``SETTINGS`` runs, as its plan reports them."""
    plan = store.pipeline.plan("rna", **SETTINGS)
    return [stage.stage for stage in plan.stages if stage.status != "skipped"]


def _without_ids(value: Any) -> Any:
    """Return ``value`` without the artifact IDs that a stage summary names."""
    if isinstance(value, dict):
        return {
            key: _without_ids(item) for key, item in value.items() if key != "artifact"
        }
    if isinstance(value, list):
        return [_without_ids(item) for item in value]
    return value


def _files(store: DataStore) -> dict[str, tuple[int, int]]:
    """Return the size and modification time of each file of the store, by path.

    A write changes a file's modification time even when it writes the same
    bytes, so equal snapshots show that nothing was written between them.
    """
    root = Path(store.zarr_loc)
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _attrs(store: DataStore, run_id: str) -> dict[str, Any]:
    return dict(
        zarr.open_group(store.zarr_loc, mode="r")[pipeline_run_path(run_id)].attrs
    )


# Pausing ---------------------------------------------------------------------


def test_a_paused_run_records_its_stages_and_stops_after_run_until(paused) -> None:
    run = paused.run
    report = run.report()

    assert (run.status, run.label, run.finished_at_ns) == ("paused", None, None)
    assert report["run"]["requestedLabel"] == "split"
    assert report["run"]["complete"] is True
    stages = [record["stage"] for record in report["stages"]]
    order = report["run"]["stageOrder"]
    # The stages up to and including pca, and no record of a later one.
    assert stages == order[: order.index("pca") + 1]
    assert {record["status"] for record in report["stages"]} == {"completed", "skipped"}
    (segment,) = report["run"]["segments"]
    assert segment["stages"] == stages
    assert segment["reason"] == "paused"
    assert segment["startedAtNs"] == report["run"]["startedAtNs"]
    assert segment["startedAtNs"] <= segment["stoppedAtNs"]
    # The process's signal protection, as the configuration records it.
    assert segment["shutdown"] == report["run"]["config"]["shutdown"]
    # A paused run has no outputs, and its run events end with run_paused.
    assert report["run"]["outputs"] == [] and report["run"]["fields"] == []
    assert paused.events[0] == ("run_started", None)
    assert paused.events[-1] == ("run_paused", None)
    with pytest.raises(RuntimeError, match="'paused'; .* requires a completed run"):
        run["pca"]


def test_paused_runs_are_listed_by_their_status(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)

    assert [run.run_id for run in store.pipeline.list_runs(status="paused")] == [
        paused.run_id
    ]
    assert store.pipeline.list_runs(status="running") == ()
    markdown = store.pipeline.open(run_id=paused.run_id).report(format="markdown")
    assert "- Status: `paused`" in markdown
    assert "## Segments" in markdown and "| paused |" in markdown


def test_run_until_refuses_a_stage_outside_the_recipe_or_skipped_before_writing(
    datastore_zarr_root, tmp_path
) -> None:
    store = _copy(datastore_zarr_root, tmp_path)
    runs = _running_stages(store)

    with pytest.raises(ValueError) as outside:
        store.pipeline.rna(run_until="clustering", **SETTINGS)
    assert str(outside.value) == (
        "run_until='clustering' is not a stage of the rna recipe; the stages that "
        f"this run runs are {runs!r}"
    )
    # Harmony is off and doublets are off in these settings.
    for skipped in ("harmony", "doublets"):
        with pytest.raises(
            ValueError, match="names a stage that this run's settings skip"
        ):
            store.pipeline.rna(run_until=skipped, **SETTINGS)
    with pytest.raises(TypeError, match="run_until must name a stage of the run"):
        store.pipeline.rna(run_until=["pca"], **SETTINGS)
    assert _runs(store) == []


def test_run_until_at_the_last_stage_that_runs_completes_the_run(
    datastore_zarr_root, tmp_path
) -> None:
    store = _copy(datastore_zarr_root, tmp_path)
    settings = {**SETTINGS, "umap": False, "markers": False}
    plan = store.pipeline.plan("rna", **settings)
    running = [stage.stage for stage in plan.stages if stage.status != "skipped"]
    # Only skipped stages follow the last stage that runs.
    assert running[-1] != plan.stages[-1].stage

    run = store.pipeline.rna(run_until=running[-1], **settings)

    assert run.status == "completed"
    assert "segments" not in run.report()["run"]


def test_a_store_without_atomic_writes_neither_pauses_nor_resumes_a_run(
    paused, tmp_path, monkeypatch
) -> None:
    store = _paused_copy(paused, tmp_path)
    before = _runs(store)
    # As on an fsspec store, which looks a key up before it writes it.
    monkeypatch.setattr(
        executor_module, "has_atomic_conditional_writes", lambda _: False
    )

    with pytest.raises(RuntimeError, match="^Pausing a pipeline run requires a Zarr"):
        store.pipeline.rna(run_until="pca", **SETTINGS)
    with pytest.raises(RuntimeError, match="^Resuming a pipeline run requires a Zarr"):
        store.pipeline.resume(paused.run_id)

    assert _runs(store) == before
    assert load_pipeline_run_record(store.zw, paused.run_id).status == "paused"


def test_a_plan_takes_no_run_until(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)

    with pytest.raises(TypeError, match="takes no run_until"):
        store.pipeline.plan("rna", run_until="pca", **SETTINGS)


def test_a_rerun_pauses_with_run_until(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    run = store.pipeline.open(run_id=paused.run_id)

    rerun = store.pipeline.rerun(run, run_until="normalization")

    assert rerun.status == "paused" and rerun.run_id != run.run_id
    assert _stage_names(store, rerun.run_id)[-1] == "normalization"


# Resuming --------------------------------------------------------------------


def test_a_resumed_run_completes_as_a_single_call_would(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    events: list[tuple[str, str | None]] = []
    labels: set[str | None] = set()

    def listen(event: Any) -> None:
        events.append((event.kind, event.stage))
        labels.add(event.label)

    run = store.pipeline.resume(paused.run_id, callback=listen)

    assert (run.run_id, run.status, run.label) == (paused.run_id, "completed", "split")
    # The resumed process reports the label that the run requested.
    assert labels == {"split"}
    report = run.report()
    order = report["run"]["stageOrder"]
    assert [record["stage"] for record in report["stages"]] == order
    first, second = report["run"]["segments"]
    assert (first["reason"], second["reason"]) == ("paused", "completed")
    assert first["stages"] + second["stages"] == order
    assert second["stages"][0] == order[order.index("pca") + 1]
    # The resumed process emits run_resumed, then only the stages it runs.
    assert events[0] == ("run_resumed", None) and events[-1] == ("run_completed", None)
    assert ("run_started", None) not in events
    assert not [stage for kind, stage in events if stage in first["stages"]]
    # A single call on the same store reuses every artifact of the run.
    single = store.pipeline.rna(**SETTINGS)
    assert set(_dispositions(single).values()) <= {"reused", "skipped"}
    assert dict(single.items()) == dict(run.items())


def test_resume_pauses_again_after_a_later_run_until(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    running = _running_stages(store)
    left = running[running.index("pca") + 1 :]

    # The refusals of a resumed run list the stages that it has left to run.
    for run_until in ("clustering", "doublets", "pca"):
        with pytest.raises(ValueError) as refused:
            store.pipeline.resume(paused.run_id, run_until=run_until)
        assert str(refused.value).endswith(
            f"the stages that run {paused.run_id} has left to run are {left!r}"
        )
    again = store.pipeline.resume(paused.run_id, run_until="neighbors")

    assert again.status == "paused"
    segments = again.report()["run"]["segments"]
    assert [segment["reason"] for segment in segments] == ["paused", "paused"]
    assert segments[1]["stages"][-1] == "neighbors"
    with pytest.raises(ValueError, match="names a stage that run .* has already run"):
        store.pipeline.resume(paused.run_id, run_until="pca")
    assert store.pipeline.resume(paused.run_id).status == "completed"


def test_a_replay_stopped_before_the_claim_writes_nothing(
    paused, tmp_path, monkeypatch
) -> None:
    store = _paused_copy(paused, tmp_path)

    class Stop(Exception):
        pass

    def stop(*args: Any, **kwargs: Any) -> Any:
        raise Stop("stopped after the replay, before the claim")

    monkeypatch.setattr(executor_module, "resume_pipeline_run_record", stop)
    before = _files(store)

    with pytest.raises(Stop):
        store.pipeline.resume(paused.run_id)

    # No file of the store was written, not even with the bytes it held.
    assert _files(store) == before


def test_a_plan_of_a_paused_run_previews_what_its_resume_computes(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    finished = _stage_names(store, paused.run_id)

    plan = store.pipeline.plan(store.pipeline.open(run_id=paused.run_id))
    run = store.pipeline.resume(paused.run_id)

    # The finished stages reuse their artifacts; the stages that the resume
    # ran compute exactly where the plan reported it.
    assert {plan[stage].status for stage in finished} <= {"reused", "skipped"}
    resumed = {
        stage: did for stage, did in _dispositions(run).items() if stage not in finished
    }
    assert plan.computed == tuple(
        stage for stage, did in resumed.items() if did == "computed"
    )
    assert plan.computed


@pytest.mark.slow
def test_a_run_paused_after_each_stage_in_turn_matches_a_single_call(
    datastore_zarr_root, tmp_path
) -> None:
    split = _copy(datastore_zarr_root, tmp_path / "split")
    single_store = _copy(datastore_zarr_root, tmp_path / "single")
    running = _running_stages(split)

    run = split.pipeline.rna(label="chained", run_until=running[0], **SETTINGS)
    for stage in running[1:-1]:
        assert run.status == "paused"
        run = split.pipeline.resume(run.run_id, run_until=stage)
    run = split.pipeline.resume(run.run_id)
    single = single_store.pipeline.rna(label="chained", **SETTINGS)

    report = run.report()
    assert (run.status, run.label) == ("completed", "chained")
    segments = report["run"]["segments"]
    assert [segment["reason"] for segment in segments] == (
        ["paused"] * (len(running) - 1) + ["completed"]
    )
    # Each process stopped after the stage that run_until named.
    assert [segment["stages"][-1] for segment in segments[:-1]] == running[:-1]
    # Each stage computed once, in its own segment, as in the single call: the
    # replays of the finished stages wrote no records and no artifacts.
    assert _dispositions(run) == _dispositions(single)
    single_report = single.report()
    for mine, theirs in zip(report["stages"], single_report["stages"], strict=True):
        assert (mine["stage"], mine["status"]) == (theirs["stage"], theirs["status"])
        assert [o["outputKey"] for o in mine["outputs"]] == [
            o["outputKey"] for o in theirs["outputs"]
        ]
        # Artifact IDs differ between two stores; every other value agrees.
        assert _without_ids(mine["summary"]) == _without_ids(theirs["summary"]), mine[
            "stage"
        ]
    # The same artifacts, with the same provenance, as the single call's.
    assert run.keys() == single.keys()
    for key in run:
        refs, others = run[key], single[key]
        pairs = zip(refs, others) if isinstance(refs, tuple) else [(refs, others)]
        for ref, other in pairs:
            mine_status = split.artifacts.inspect(ref)
            their_status = single_store.artifacts.inspect(other)
            assert mine_status.operation == their_status.operation, key
            assert mine_status.parameters == their_status.parameters, key
    # The run's frozen fields hold the single call's values.
    for axis in ("cells", "features"):
        mine_view, their_view = getattr(run, axis), getattr(single, axis)
        assert mine_view.columns == their_view.columns
        for column in mine_view.columns:
            np.testing.assert_array_equal(
                mine_view.fetch_all(column),
                their_view.fetch_all(column),
                err_msg=column,
            )
    # On the store it ran on, a single call reuses every artifact of the run.
    again = split.pipeline.rna(**SETTINGS)
    assert set(_dispositions(again).values()) <= {"reused", "skipped"}
    assert dict(again.items()) == dict(run.items())


_RESUME_IN_A_NEW_PROCESS = textwrap.dedent(
    """
    import sys

    from cytearc import DataStore

    store = DataStore(sys.argv[1], default_assay="RNA")
    run = store.pipeline.resume(sys.argv[2])
    print(run.status)
    """
)


@pytest.mark.slow
def test_a_new_process_resumes_the_run_from_the_store_alone(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)

    # Nothing of the pausing process reaches the one that resumes but the store.
    completed = subprocess.run(
        [sys.executable, "-c", _RESUME_IN_A_NEW_PROCESS, store.zarr_loc, paused.run_id],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().splitlines()[-1] == "completed"

    run = store.pipeline.open(run_id=paused.run_id)
    assert (run.status, run.label) == ("completed", "split")
    first, second = run.report()["run"]["segments"]
    assert (
        second["reason"] == "completed"
        and second["startedAtNs"] >= first["stoppedAtNs"]
    )
    single = store.pipeline.rna(**SETTINGS)
    assert set(_dispositions(single).values()) <= {"reused", "skipped"}
    assert dict(single.items()) == dict(run.items())


# Other sessions and machines -------------------------------------------------


def log_library_size(assay: Any, counts: Any) -> Any:
    """A custom normalizer: the log of library-size normalized counts."""
    return np.log1p(norm_lib_size(assay, counts))


@pytest.mark.slow
def test_a_run_of_a_custom_normalizer_resumes_where_the_session_applies_it(
    datastore_zarr_root, tmp_path
) -> None:
    first = _copy(datastore_zarr_root, tmp_path)
    first.RNA.normMethod = log_library_size
    paused = first.pipeline.rna(run_until="pca", **SETTINGS)
    # Another machine opens the store; its session normalizes as by default.
    second = DataStore(first.zarr_loc, default_assay="RNA")
    before = _files(second)

    with pytest.raises(ValueError) as refused:
        second.pipeline.resume(paused.run_id)

    message = str(refused.value)
    assert message.startswith(
        f"Pipeline run {paused.run_id} normalizes assay 'RNA' with the custom "
        "normalizer"
    )
    assert "set that assay's normMethod to the same function in this session" in message
    # A resume takes no normalization setting, so the error suggests none.
    assert "normalization=" not in message
    assert _files(second) == before
    second.RNA.normMethod = log_library_size
    assert second.pipeline.resume(paused.run_id).status == "completed"


@pytest.mark.slow
def test_a_run_paused_on_a_small_machine_resumes_with_a_larger_ones_resources(
    datastore_zarr_root, tmp_path
) -> None:
    shutil.copytree(datastore_zarr_root, tmp_path / "small.zarr")
    small = DataStore(
        str(tmp_path / "small.zarr"), default_assay="RNA", nthreads=1, mem_budget="1GB"
    )
    paused = small.pipeline.rna(label="split", run_until="pca", **SETTINGS)
    # The store moves to a larger machine while the run is paused.
    shutil.copytree(tmp_path / "small.zarr", tmp_path / "large.zarr")
    large = DataStore(
        str(tmp_path / "large.zarr"), default_assay="RNA", nthreads=4, mem_budget="2GB"
    )

    run = large.pipeline.resume(paused.run_id)

    assert (run.status, run.label) == ("completed", "split")
    report = run.report()
    first, second = (set(segment["stages"]) for segment in report["run"]["segments"])
    # Each stage record holds the workers and budget of the process that ran it.
    for record in report["stages"]:
        resources = record["metrics"]["resources"]
        machine = small if record["stage"] in first else large
        assert record["stage"] in first | second
        assert (resources["workers"], resources["memoryBudgetBytes"]) == (
            machine.nthreads,
            machine.memoryBytes,
        ), record["stage"]
    assert (small.nthreads, small.memoryBytes) != (large.nthreads, large.memoryBytes)
    single = large.pipeline.rna(**SETTINGS)
    assert set(_dispositions(single).values()) <= {"reused", "skipped"}
    assert dict(single.items()) == dict(run.items())


@pytest.mark.slow
def test_a_paused_run_resumes_in_a_repacked_copy_of_its_store(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    repack_store(store.zarr_loc, str(tmp_path / "repacked.zarr"))
    repacked = DataStore(str(tmp_path / "repacked.zarr"), default_assay="RNA")
    assert repacked.pipeline.open(run_id=paused.run_id).status == "paused"

    run = repacked.pipeline.resume(paused.run_id)

    assert (run.status, run.label) == ("completed", "split")
    single = repacked.pipeline.rna(**SETTINGS)
    assert set(_dispositions(single).values()) <= {"reused", "skipped"}
    assert dict(single.items()) == dict(run.items())


# Refusals --------------------------------------------------------------------


def _ended(store: DataStore, status: str) -> str:
    """Record a run of status ``status``, which has run no stage."""
    record = create_pipeline_run_record(
        store.zw,
        recipe="rna",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        cytearc_version="1.0.0",
    )
    if status == "failed":
        fail_pipeline_run_record(
            store.zw, run_id=record.run_id, error=RuntimeError("x")
        )
    elif status == "interrupted":
        from cytearc.storage.pipeline_runs import PipelineInterruptionRecord

        interrupt_pipeline_run_record(
            store.zw,
            run_id=record.run_id,
            interruption=PipelineInterruptionRecord(
                "signal", "stopped", 1, 15, "SIGTERM"
            ),
        )
    return record.run_id


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_resume_refuses_a_run_that_is_not_paused(status, paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    run_id = _ended(store, status)
    before = _files(store)

    with pytest.raises(
        ValueError, match=rf"is '{status}', and only a paused run resumes"
    ) as refused:
        store.pipeline.resume(run_id)
    # Another process may run a running run, so a rerun would repeat its work.
    advice = "ds.pipeline.rerun(run) starts a new run that reuses its finished stages"
    assert (advice in str(refused.value)) == (status != "running")
    assert _files(store) == before


def test_resume_takes_a_run_and_refuses_a_completed_or_unknown_one(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    resumed = store.pipeline.resume(store.pipeline.open(run_id=paused.run_id))
    assert (resumed.run_id, resumed.status) == (paused.run_id, "completed")
    before = _files(store)

    with pytest.raises(
        ValueError,
        match=r"is 'completed', and only a paused run resumes; ds\.pipeline\.open",
    ):
        store.pipeline.resume(paused.run_id)
    with pytest.raises(KeyError, match="Pipeline run does not exist"):
        store.pipeline.resume("0" * 64)
    with pytest.raises(TypeError, match="run must be a PipelineRun or a run's ID"):
        store.pipeline.resume(42)  # type: ignore[arg-type]
    assert _files(store) == before


def _edit_run(store: DataStore, run_id: str, **changes: Any) -> None:
    group = zarr.open_group(store.zarr_loc, mode="r+")[pipeline_run_path(run_id)]
    group.attrs.update(changes)


def test_resume_refuses_another_release_and_a_torn_pause(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    recorded = _attrs(store, paused.run_id)

    _edit_run(store, paused.run_id, cytearcVersion="1.4.0")
    before = _files(store)
    with pytest.raises(
        ValueError,
        match="recorded by CyteArc 1.4.0, so it resumes only with that release, "
        "not .*; resume it with CyteArc 1.4.0",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before
    _edit_run(
        store, paused.run_id, cytearcVersion=recorded["cytearcVersion"], complete=False
    )
    before = _files(store)
    # Nothing is replayed: the error says how to reuse the finished stages.
    with pytest.raises(
        ValueError,
        match=r"did not finish writing its pause, so it cannot resume; ds\.pipeline\.rerun",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


def test_resume_refuses_a_finished_stage_whose_input_changed_and_writes_nothing(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    # The run's cells change, so its input snapshot would select other cells.
    keep = np.ones(store.cells.N, dtype=bool)
    keep[:5] = False
    store.cells.insert("I", by_cell_id(store.cells, keep), overwrite=True, force=True)
    before = _files(store)

    with pytest.raises(
        ValueError,
        match="its finished stage 'input_snapshot' would compute again instead of "
        "reusing .* ds\\.pipeline\\.rerun\\(run\\) starts a new run that computes it",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


def test_resume_refuses_a_finished_stage_whose_artifact_is_incomplete(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    (pca,) = (
        record
        for record in load_pipeline_stage_records(store.zw, paused.run_id)
        if record.stage == "pca"
    )
    # As when the store was copied to this machine while a write was unfinished.
    artifact_group(store.zw, pca.outputs[0].artifact).attrs["complete"] = False
    before = _files(store)

    with pytest.raises(ValueError) as refused:
        store.pipeline.resume(paused.run_id)
    # The error names the stage, its artifact, why it would be computed, and
    # what to do instead.
    message = str(refused.value)
    reference = pca.outputs[0].artifact
    assert "its finished stage 'pca' would compute again instead of reusing" in message
    assert f"({reference.kind} {reference.artifact_id[:12]})" in message
    assert "artifact is incomplete. ds.pipeline.rerun(run) starts a new run" in message
    assert _files(store) == before


def test_resume_refuses_a_finished_stage_that_reuses_another_artifact(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    records = load_pipeline_stage_records(store.zw, paused.run_id)
    pca = next(record for record in records if record.stage == "pca")
    group = zarr.open_group(store.zarr_loc, mode="r+")[
        pipeline_stage_path(paused.run_id, pca.ordinal)
    ]
    # The record names another PCA than the one that the store holds now, as
    # when the store was replaced by another that computed the same stage.
    attrs = dict(group.attrs)
    recorded = attrs["outputs"][0]["artifact"]["artifact_id"]
    attrs["outputs"][0]["artifact"]["artifact_id"] = "f" * len(recorded)
    group.attrs.put(attrs)
    before = _files(store)

    with pytest.raises(
        ValueError,
        match=r"its finished stage 'pca' recorded 'pca' as .* but the store now "
        r"gives .*; ds\.pipeline\.rerun\(run\) starts a new run",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


def test_resume_refuses_a_run_whose_stage_records_differ_from_its_pause(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    attrs = _attrs(store, paused.run_id)
    # The record names one stage fewer than the stage records hold.
    segment = attrs["segments"][0]
    _edit_run(
        store,
        paused.run_id,
        segments=[{**segment, "stages": segment["stages"][:-1]}],
    )
    before = _files(store)

    with pytest.raises(
        ValueError,
        match="its stage records no longer match the stages that it ran before it "
        "paused; ds\\.pipeline\\.rerun",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


@pytest.mark.parametrize("changed", ["settings", "stage_order"])
def test_resume_refuses_a_run_whose_settings_resolve_to_another_plan(
    paused, tmp_path, changed
) -> None:
    store = _paused_copy(paused, tmp_path)
    attrs = _attrs(store, paused.run_id)
    if changed == "settings":
        config = attrs["config"]
        normalization = dict(config["stages"]["normalization"])
        assert normalization.pop("method") == "lib_size"
        # The record no longer names its normalizer, so its settings take the
        # one that its saved normalized values record, and resolve to others.
        stages = {**config["stages"], "normalization": normalization}
        _edit_run(store, paused.run_id, config={**config, "stages": stages})
    else:
        # The record holds a stage that the recipe does not run.
        _edit_run(store, paused.run_id, stageOrder=[*attrs["stageOrder"], "extra"])
    before = _files(store)

    with pytest.raises(
        ValueError,
        match="no longer resolve to the stages and settings that it recorded; "
        "ds\\.pipeline\\.rerun",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


@pytest.mark.parametrize(
    ("stage", "action", "message"),
    [
        (
            "cell_cycle",
            lambda state: (),
            "it skipped stage 'cell_cycle', which its settings now run",
        ),
        (
            "normalization",
            None,
            "it ran stage 'normalization', which its settings now skip",
        ),
    ],
    ids=["skipped_now_runs", "ran_now_skipped"],
)
def test_the_replay_refuses_a_plan_that_runs_or_skips_a_finished_stage_otherwise(
    paused, tmp_path, stage, action, message
) -> None:
    store = _paused_copy(paused, tmp_path)
    record = load_pipeline_run_record(store.zw, paused.run_id)
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            PipelineAccessor.rna
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in {"run_until", "callback", "cancel"}
    }
    plan = resolve_rna(store, **{**defaults, **recorded_settings(store, record)})
    # Settings that resolve to the recorded ones run the same stages, so only
    # a changed plan reaches these refusals.
    stages = tuple(
        replace(planned, action=action) if planned.name == stage else planned
        for planned in plan.stages
    )
    finished = load_pipeline_stage_records(store.zw, paused.run_id)

    with pytest.raises(ValueError, match=message):
        executor_module._replay_finished_stages(
            replace(plan, stages=stages), plan.start(store), finished, paused.run_id
        )


def test_a_resume_of_a_pause_that_another_resume_claimed_is_refused(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    # Another process claimed this pause and has not recorded its resume: it
    # is about to, or it stopped in between.
    claim = (
        Path(store.zarr_loc) / "pipeline/runs/.resume-claims" / paused.run_id / "1.json"
    )
    claim.parent.mkdir(parents=True)
    claim.write_text(
        json.dumps({"runId": paused.run_id, "segment": 1, "token": "0" * 32})
    )
    zarr.open_group(store.zarr_loc, mode="r+")["pipeline/runs"].create_array(
        ".resume-claims", shape=(0,), dtype="uint8"
    )
    before = _files(store)

    with pytest.raises(
        RuntimeError,
        match="is claimed by another resume that has not yet recorded itself: .* "
        "ds\\.pipeline\\.end_paused\\(run, release_claim=True\\) ends it, while "
        "ds\\.pipeline\\.rerun\\(run\\) starts a new run",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


def test_of_two_processes_that_resume_one_run_at_once_one_wins(
    paused, tmp_path, monkeypatch
) -> None:
    store = _paused_copy(paused, tmp_path)
    paused_record = _attrs(store, paused.run_id)
    claim = pipeline_run_storage._claim_pipeline_run_resume
    at_claim = threading.Barrier(2, timeout=60)
    claimed, tried = threading.Event(), threading.Event()
    seen: dict[str, Any] = {}

    def claim_in_order(root: Any, run_id: str, segment: int) -> bool:
        # Both processes have replayed the finished stages and read the
        # paused run before either claims it.
        at_claim.wait()
        if threading.current_thread().name == "winner":
            won = claim(root, run_id, segment)
            claimed.set()
            # The winner records its resume once the loser has tried.
            tried.wait(timeout=60)
            return won
        claimed.wait(timeout=60)
        return claim(root, run_id, segment)

    monkeypatch.setattr(
        pipeline_run_storage, "_claim_pipeline_run_resume", claim_in_order
    )
    outcomes: dict[str, object] = {}

    def resume() -> None:
        name = threading.current_thread().name
        resumer = DataStore(store.zarr_loc, default_assay="RNA")
        try:
            outcomes[name] = resumer.pipeline.resume(paused.run_id).status
        except RuntimeError as error:
            outcomes[name] = error
        finally:
            if name == "loser":
                # Neither process has written to the run record yet.
                seen["after the loser"] = _attrs(store, paused.run_id)
                tried.set()

    threads = [
        threading.Thread(target=resume, name=name) for name in ("winner", "loser")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes["winner"] == "completed"
    assert isinstance(outcomes["loser"], RuntimeError)
    assert "is claimed by another resume that has not yet recorded itself" in str(
        outcomes["loser"]
    )
    # The loser left the run record as the pause wrote it, for the winner.
    assert seen["after the loser"] == paused_record
    record = load_pipeline_run_record(store.zw, paused.run_id)
    assert (record.status, len(record.segments)) == ("completed", 2)


@pytest.mark.parametrize("rival_runs", [False, True], ids=["paused_again", "running"])
def test_a_resume_refuses_a_run_that_another_process_resumed_before_its_replay(
    paused, tmp_path, monkeypatch, rival_runs
) -> None:
    store = _paused_copy(paused, tmp_path)
    finished_stages = executor_module._finished_stages
    seen: dict[str, Any] = {}

    def after_a_rival(*args: Any, **kwargs: Any) -> Any:
        # Another process resumed the run after this one read its record.
        monkeypatch.setattr(executor_module, "_finished_stages", finished_stages)
        rival = DataStore(store.zarr_loc, default_assay="RNA")
        if rival_runs:
            # It has claimed the pause and runs its first stage.
            pipeline_run_storage.resume_pipeline_run_record(
                rival.zw, run_id=paused.run_id, segment=1, shutdown=_SHUTDOWN
            )
        else:
            rival.pipeline.resume(paused.run_id, run_until="ann_index")
        seen["before"] = _files(store)
        return finished_stages(*args, **kwargs)

    monkeypatch.setattr(executor_module, "_finished_stages", after_a_rival)

    with pytest.raises(RuntimeError) as refused:
        store.pipeline.resume(paused.run_id)

    # The error says what happened, without the advice to rerun the run.
    assert str(refused.value) == (
        f"Pipeline run {paused.run_id} was resumed or ended by another process after this "
        "process read it; open the run again to see its status"
    )
    assert _files(store) == seen["before"]


def test_a_resume_refuses_a_pause_that_another_process_resumed_after_it_read_it(
    paused, tmp_path, monkeypatch
) -> None:
    store = _paused_copy(paused, tmp_path)
    claim = executor_module.resume_pipeline_run_record

    def after_another_resume(*args: Any, **kwargs: Any) -> Any:
        # Another process resumes the pause and pauses again after this one
        # replayed its finished stages, before this one claims the pause.
        monkeypatch.setattr(executor_module, "resume_pipeline_run_record", claim)

        def other() -> None:
            rival = DataStore(store.zarr_loc, default_assay="RNA")
            rival.pipeline.resume(paused.run_id, run_until="ann_index")

        worker = threading.Thread(target=other)
        worker.start()
        worker.join()
        seen["before"] = _files(store)
        return claim(*args, **kwargs)

    monkeypatch.setattr(
        executor_module, "resume_pipeline_run_record", after_another_resume
    )
    seen: dict[str, Any] = {}

    with pytest.raises(
        RuntimeError, match="was resumed or ended by another process after"
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == seen["before"]

    # The other process's pause stands, and this process recorded no stage.
    record = load_pipeline_run_record(store.zw, paused.run_id)
    assert record.status == "paused"
    assert [segment.stages[-1] for segment in record.segments] == ["pca", "ann_index"]
    assert _stage_names(store, paused.run_id)[-1] == "ann_index"


def test_a_paused_run_does_not_resume_after_another_run_commits_its_label(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    # The paused run reserves nothing, so another run takes its label.
    other = store.pipeline.rna(label="split", **SETTINGS)
    before = _files(store)

    with pytest.raises(
        ValueError,
        match=f"already committed by run {other.run_id}; "
        "ds\\.pipeline\\.rerun\\(run, label=\\.\\.\\.\\) starts a new run under another label",
    ):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == before


@pytest.mark.slow
def test_a_run_whose_label_another_run_committed_fails_when_it_pauses(
    datastore_zarr_root, tmp_path
) -> None:
    store = _copy(datastore_zarr_root, tmp_path)
    events: list[tuple[str, str | None]] = []

    def commit_the_label(event: Any) -> None:
        events.append((event.kind, event.stage))
        if (event.kind, event.stage) == ("stage_completed", "normalization"):
            # Another process completes a run with the same label meanwhile.
            def other() -> None:
                rival = DataStore(store.zarr_loc, default_assay="RNA")
                rival.pipeline.rna(label="taken", **SETTINGS)

            worker = threading.Thread(target=other)
            worker.start()
            worker.join()

    with pytest.raises(PipelineExecutionError, match="during stage 'pause'") as failed:
        store.pipeline.rna(
            label="taken", run_until="pca", callback=commit_the_label, **SETTINGS
        )

    record = load_pipeline_run_record(store.zw, failed.value.run_id)
    assert record.status == "failed" and record.error is not None
    assert record.error.type == "PipelineLabelConflict"
    assert events[-1] == ("run_failed", "pause")


# Ending a paused run ---------------------------------------------------------


def test_a_paused_run_ends_and_a_rerun_reuses_its_finished_stages(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    finished = _stage_names(store, paused.run_id)

    ended = store.pipeline.end_paused(
        store.pipeline.open(run_id=paused.run_id), reason="analysis dropped"
    )

    assert (ended.run_id, ended.status, ended.label) == (
        paused.run_id,
        "interrupted",
        None,
    )
    report = ended.report()
    assert (
        report["run"]["interruption"]["kind"],
        report["run"]["interruption"]["message"],
    ) == (
        "abandoned_pause",
        "analysis dropped",
    )
    assert report["run"]["segments"][-1]["reason"] == "interrupted"
    assert store.pipeline.list_runs(status="paused") == ()
    with pytest.raises(
        ValueError, match="is 'interrupted', and only a paused run resumes"
    ):
        store.pipeline.resume(paused.run_id)
    # The run reserved no label and kept its stages, so a rerun can take the
    # label and reuses every finished stage, as its plan reports.
    plan = store.pipeline.plan(ended, label="split")
    assert {plan[stage].status for stage in finished} <= {"reused", "skipped"}
    assert plan.computed and not set(plan.computed) & set(finished)


def test_a_pause_whose_write_was_cut_off_ends(paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    _edit_run(store, paused.run_id, complete=False)

    ended = store.pipeline.end_paused(paused.run_id)

    assert ended.status == "interrupted"
    assert ended.report()["run"]["interruption"]["message"] == "ended while paused"


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_end_paused_refuses_a_run_that_is_not_paused(status, paused, tmp_path) -> None:
    store = _paused_copy(paused, tmp_path)
    run_id = _ended(store, status)
    before = _files(store)

    with pytest.raises(
        ValueError, match=rf"is '{status}', and only a paused run ends this way"
    ):
        store.pipeline.end_paused(run_id)
    assert _files(store) == before


def test_end_paused_refuses_other_values_and_a_read_only_store(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    before = _files(store)

    with pytest.raises(KeyError, match="Pipeline run does not exist"):
        store.pipeline.end_paused("0" * 64)
    with pytest.raises(TypeError, match="run must be a PipelineRun or a run's ID"):
        store.pipeline.end_paused(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reason must be a non-empty string"):
        store.pipeline.end_paused(paused.run_id, reason="")
    with pytest.raises(TypeError, match="release_claim must be a boolean"):
        store.pipeline.end_paused(paused.run_id, release_claim=1)  # type: ignore[arg-type]
    reader = DataStore(store.zarr_loc, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError, match="requires zarr_mode='r\\+'"):
        reader.pipeline.end_paused(paused.run_id)
    assert _files(store) == before


def test_a_paused_run_that_a_stopped_resume_claimed_ends_once_its_claim_is_released(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    # A resume claimed the pause and stopped before it recorded itself.
    assert pipeline_run_storage._claim_pipeline_run_resume(store.zw, paused.run_id, 1)
    before = _files(store)

    for attempt in (store.pipeline.resume, store.pipeline.end_paused):
        with pytest.raises(
            RuntimeError,
            match="is claimed by another resume that has not yet recorded itself",
        ) as refused:
            attempt(paused.run_id)
        assert "ds.pipeline.end_paused(run, release_claim=True)" in str(refused.value)
    assert _files(store) == before

    ended = store.pipeline.end_paused(paused.run_id, release_claim=True)

    assert ended.status == "interrupted"
    assert ended.report()["run"]["interruption"]["kind"] == "abandoned_pause"


@pytest.mark.parametrize("first", ["end", "resume"])
def test_of_a_resume_and_an_end_of_one_pause_only_one_goes_ahead(
    paused, tmp_path, monkeypatch, first
) -> None:
    store = _paused_copy(paused, tmp_path)
    claim = pipeline_run_storage._claim_pipeline_run_resume
    at_claim = threading.Barrier(2, timeout=60)
    recorded, tried = threading.Event(), threading.Event()
    seen: dict[str, Any] = {}

    def claim_in_order(root: Any, run_id: str, segment: int) -> bool:
        # Both have read the paused run, and the resume has replayed its
        # finished stages, before either claims the pause.
        at_claim.wait()
        if threading.current_thread().name == first:
            return claim(root, run_id, segment)
        recorded.wait(timeout=60)
        seen["before"] = _files(store)
        return claim(root, run_id, segment)

    def recording(inner: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = inner(*args, **kwargs)
            if threading.current_thread().name == first:
                # The first records its resume or end, and goes on once the
                # second has tried.
                recorded.set()
                tried.wait(timeout=60)
            return result

        return wrapped

    monkeypatch.setattr(
        pipeline_run_storage, "_claim_pipeline_run_resume", claim_in_order
    )
    monkeypatch.setattr(
        executor_module,
        "resume_pipeline_run_record",
        recording(executor_module.resume_pipeline_run_record),
    )
    monkeypatch.setattr(
        accessor_module,
        "end_paused_pipeline_run_record",
        recording(accessor_module.end_paused_pipeline_run_record),
    )
    outcomes: dict[str, object] = {}

    def call() -> None:
        name = threading.current_thread().name
        session = DataStore(store.zarr_loc, default_assay="RNA")
        try:
            if name == "end":
                outcomes[name] = session.pipeline.end_paused(paused.run_id).status
            else:
                outcomes[name] = session.pipeline.resume(paused.run_id).status
        except RuntimeError as error:
            outcomes[name] = error
        finally:
            if name != first:
                seen["after"] = _files(store)
                tried.set()

    threads = [threading.Thread(target=call, name=name) for name in ("end", "resume")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    second = "resume" if first == "end" else "end"
    assert outcomes[first] == ("interrupted" if first == "end" else "completed")
    assert isinstance(outcomes[second], RuntimeError)
    assert "was resumed or ended by another process" in str(outcomes[second])
    # The second wrote nothing.
    assert seen["after"] == seen["before"]


def test_a_resume_that_read_the_run_before_it_ended_writes_nothing(
    paused, tmp_path, monkeypatch
) -> None:
    store = _paused_copy(paused, tmp_path)
    finished_stages = executor_module._finished_stages
    seen: dict[str, Any] = {}

    def after_an_end(*args: Any, **kwargs: Any) -> Any:
        # Another process ended the run after this one read its record.
        pipeline_run_storage.end_paused_pipeline_run_record(
            store.zw, run_id=paused.run_id, segment=1, reason="dropped"
        )
        seen["before"] = _files(store)
        return finished_stages(*args, **kwargs)

    monkeypatch.setattr(executor_module, "_finished_stages", after_an_end)

    with pytest.raises(RuntimeError, match="was resumed or ended by another process"):
        store.pipeline.resume(paused.run_id)
    assert _files(store) == seen["before"]
    assert load_pipeline_run_record(store.zw, paused.run_id).status == "interrupted"


# Ends of a resumed run -------------------------------------------------------


def test_a_resumed_run_that_is_cancelled_ends_its_segment_interrupted(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    cancel = CancellationToken()

    def stop_at_neighbors(event: Any) -> None:
        if (event.kind, event.stage) == ("stage_started", "neighbors"):
            cancel.cancel("stopped while resumed")

    with pytest.raises(PipelineCancelled):
        store.pipeline.resume(paused.run_id, callback=stop_at_neighbors, cancel=cancel)

    record = load_pipeline_run_record(store.zw, paused.run_id)
    assert record.status == "interrupted" and record.complete
    first, second = record.segments
    assert second.reason == "interrupted" and second.stopped_at_ns is not None
    # The second segment holds the stages that this process recorded, the
    # interrupted one last.
    stages = _stage_names(store, paused.run_id)
    assert first.stages + second.stages == tuple(stages)
    assert stages[-1] == "neighbors"


def test_a_cancellation_after_the_pause_returns_the_paused_run(
    paused, tmp_path
) -> None:
    store = _paused_copy(paused, tmp_path)
    cancel = CancellationToken()

    def cancel_once_paused(event: Any) -> None:
        if event.kind == "run_paused":
            cancel.cancel("too late")

    run = store.pipeline.rna(
        run_until="pca", callback=cancel_once_paused, cancel=cancel, **SETTINGS
    )

    # The pause is durable before the run stops, so the call returns it.
    assert run.status == "paused"
    assert load_pipeline_run_record(store.zw, run.run_id).status == "paused"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (OSError("the store refused the write"), "failed"),
        (concurrent.futures.CancelledError(), "interrupted"),
    ],
    ids=["storage_error", "cancelled_zarr_call"],
)
def test_a_pause_that_cannot_be_written_ends_the_run(
    paused, tmp_path, monkeypatch, error, status
) -> None:
    store = _paused_copy(paused, tmp_path)
    events: list[tuple[str, str | None]] = []

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(executor_module, "pause_pipeline_run_record", refused)

    with pytest.raises((PipelineExecutionError, concurrent.futures.CancelledError)):
        store.pipeline.rna(
            run_until="pca",
            callback=lambda event: events.append((event.kind, event.stage)),
            **SETTINGS,
        )

    (run,) = [
        run
        for run in store.pipeline.list_runs(limit=100)
        if run.run_id != paused.run_id
    ]
    record = load_pipeline_run_record(store.zw, run.run_id)
    assert (record.status, record.complete, record.segments) == (status, True, ())
    assert _stage_names(store, run.run_id)[-1] == "pca"
    expected = (
        ("run_failed", "pause") if status == "failed" else ("run_interrupted", None)
    )
    assert events[-1] == expected


def test_a_resumed_run_whose_stage_fails_ends_its_segment_failed(
    paused, tmp_path, monkeypatch
) -> None:
    store = _paused_copy(paused, tmp_path)

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Leiden failed on purpose")

    monkeypatch.setattr(type(store), "_run_leiden_artifact", failing)

    with pytest.raises(PipelineExecutionError, match="during stage 'leiden'"):
        store.pipeline.resume(paused.run_id)

    record = load_pipeline_run_record(store.zw, paused.run_id)
    assert record.status == "failed"
    assert [segment.reason for segment in record.segments] == ["paused", "failed"]
    assert record.segments[1].stages[-1] == "leiden"
