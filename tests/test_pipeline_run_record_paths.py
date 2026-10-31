"""Durable pipeline-run records, label claims, and pipeline execution outcomes."""

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import FsspecStore, MemoryStore

import cytearc.datastore._pipeline.executor as pipeline_executor
import cytearc.datastore._pipeline.recipes.rna as rna_recipe
import cytearc.storage.pipeline_runs as pipeline_run_storage
from cytearc import DataStore, PipelineExecutionError, PipelineRun
from cytearc.datastore.pipeline_accessor import PipelineEvent
from cytearc.storage.artifacts import ArtifactRef, new_artifact_id
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.pipeline_runs import (
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    PipelineOutputRecord,
    PipelineRunRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    fail_pipeline_run_record,
    finish_pipeline_stage_record,
    interrupt_pipeline_run_record,
    list_pipeline_run_records,
    load_pipeline_run_record,
    load_pipeline_stage_record,
    load_pipeline_stage_records,
    open_pipeline_run_record,
    start_pipeline_stage_record,
)
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.storage.stores import load_zarr
from cytearc.tools.repack_zarr import repack_store
from cytearc.utils.logging import logger
from cytearc.utils.shutdown import (
    ShutdownRequested,
    ShutdownToken,
    shutdown_checkpoint,
)
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts, write_count_store

_CLAIMS = "pipeline/runs/.label-claims"
_FAST_RUN: dict[str, Any] = {
    "filtering": False,
    "cell_cycle": False,
    "umap": False,
    "leiden": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    # PCA needs dims + 1 of the small store's 8 genes.
    "pca": {"dims": 3},
}
# Every graph stage runs on the small graph store, and nothing after them.
_GRAPH_STAGES: dict[str, Any] = {
    **_FAST_RUN,
    "hvg": {"top_n": 20},
    "pca": {"dims": 3},
    "neighbors": {"k": 5},
}
# The stage records of a completed run with _GRAPH_STAGES, in order.
_GRAPH_STAGE_STATUSES = (
    ("input_snapshot", "completed"),
    ("cell_calling", "skipped"),
    ("hashing", "skipped"),
    ("filtering", "skipped"),
    ("cell_cycle", "skipped"),
    ("highly_variable_features", "completed"),
    ("normalization", "completed"),
    ("pca", "completed"),
    ("harmony", "skipped"),
    ("ann_index", "completed"),
    ("neighbors", "completed"),
    ("connectivity", "completed"),
    ("embedding_initialization", "skipped"),
    ("umap", "skipped"),
    ("paris", "skipped"),
    ("cluster_selection", "skipped"),
    ("membership_strength", "skipped"),
    ("tsne", "skipped"),
    ("doublets", "skipped"),
    ("markers", "skipped"),
)
_PCA = [stage for stage, _status in _GRAPH_STAGE_STATUSES].index("pca")
# Paris supplies a cluster cut for tree plots.
_GRAPH_RUN: dict[str, Any] = {**_GRAPH_STAGES, "paris": True}


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.01,
        rss_baseline_bytes=None,
        rss_peak_bytes=None,
        rss_incremental_peak_bytes=None,
        sample_interval_seconds=0.1,
        sample_count=0,
        sampling_error_count=0,
        rss_unavailable_reason="not sampled in tests",
    )


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _new_run(
    root: zarr.Group,
    *,
    label: str | None = None,
    stages: tuple[str, ...] = ("one",),
) -> PipelineRunRecord:
    return create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=label,
        assay="RNA",
        config={},
        stage_order=stages,
        cytearc_version="1.0.0",
    )


def _run_stage(root: zarr.Group, run: PipelineRunRecord, ordinal: int) -> None:
    start_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=ordinal,
        stage=run.stage_order[ordinal],
    )
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=ordinal,
        status="completed",
        metrics=_metrics(),
    )


def _complete(root: zarr.Group, run: PipelineRunRecord) -> PipelineRunRecord:
    return complete_pipeline_run_record(root, run_id=run.run_id, outputs=(), fields=())


def _committed_run(root: zarr.Group, label: str | None) -> PipelineRunRecord:
    run = _new_run(root, label=label)
    _run_stage(root, run, 0)
    return _complete(root, run)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_raw(root: zarr.Group, key: str, payload: bytes) -> None:
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(root.store.set(key, buffer))


def _claim_payload(label: str, run_id: str) -> bytes:
    return json.dumps({"label": label, "runId": run_id}).encode("utf-8")


def _write_claim(
    root: zarr.Group,
    *,
    label: str,
    predecessor: str,
    run_id: str,
) -> None:
    """Write one durable label claim as a finalizer stores it."""
    _write_raw(
        root,
        f"{_CLAIMS}/{_digest(label)}/{predecessor}.json",
        _claim_payload(label, run_id),
    )


def _copy_run(source: zarr.Group, destination: zarr.Group, run_id: str) -> None:
    runs = destination.require_group("pipeline/runs")
    source_run = source[f"pipeline/runs/{run_id}"]
    copied = runs.create_group(run_id, attributes=dict(source_run.attrs))
    stages = copied.create_group("stages")
    for name, stage in source_run["stages"].groups():
        stages.create_group(name, attributes=dict(stage.attrs))


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _small_store(path: Path) -> str:
    counts = np.random.default_rng(3).poisson(3.0, size=(16, 8)).astype(np.uint32)
    write_count_store(str(path), {"RNA": counts}, np.uint32)
    return str(path)


def _prepared_store(path: Path) -> zarr.Group:
    location = _small_store(path)
    # The first writable open prepares a fresh import.
    DataStore(location, default_assay="RNA", min_features_per_cell=1)
    return zarr.open_group(location, mode="r+")


def test_run_record_stored_under_another_run_id_is_rejected_and_skipped() -> None:
    root = _root()
    run = _committed_run(root, "baseline")
    copy_id = "f" * 64
    runs = root["pipeline/runs"]
    runs.create_group(copy_id, attributes=dict(runs[run.run_id].attrs)).create_group(
        "stages"
    )

    with pytest.raises(ValueError, match=f"{copy_id!r} contains {run.run_id!r}"):
        load_pipeline_run_record(root, copy_id)
    assert list_pipeline_run_records(root) == (run,)
    assert open_pipeline_run_record(root, label="baseline") == run


def test_torn_pipeline_group_without_runs_reads_as_an_empty_catalog() -> None:
    root = _root()
    # The catalog is created one group at a time, so a stop can leave only
    # the pipeline group.
    root.create_group("pipeline")

    assert list_pipeline_run_records(root) == ()
    with pytest.raises(KeyError, match="No completed pipeline run"):
        open_pipeline_run_record(root, label="baseline")

    run = _committed_run(root, "baseline")
    assert open_pipeline_run_record(root, label="baseline") == run


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"label": "baseline", "runId"', "invalid durable claim"),
        (_claim_payload("other", "a" * 64), "collides with another durable claim"),
    ],
    ids=["unreadable", "foreign_label"],
)
def test_labeled_run_refuses_to_start_over_a_corrupt_durable_claim(
    payload: bytes,
    message: str,
) -> None:
    root = _root()
    _committed_run(root, "previous")
    _write_raw(root, f"{_CLAIMS}/{_digest('baseline')}/head.json", payload)
    runs_before = set(root["pipeline/runs"].group_keys())

    with pytest.raises(ValueError, match=message):
        _new_run(root, label="baseline")

    assert set(root["pipeline/runs"].group_keys()) == runs_before
    assert _new_run(root, label="unaffected").status == "running"


@pytest.mark.parametrize("corruption", ["cycle", "foreign_owner"])
def test_commit_fails_durably_when_the_claim_chain_changed_during_the_run(
    corruption: str,
) -> None:
    root = _root()
    # An earlier labeled commit created the claim namespace.
    _committed_run(root, "previous")
    if corruption == "cycle":
        stale = _new_run(root, label="baseline")
        fail_pipeline_run_record(root, run_id=stale.run_id, error=RuntimeError("x"))
        run = _new_run(root, label="baseline")
        _write_claim(root, label="baseline", predecessor="head", run_id=stale.run_id)
        _write_claim(
            root, label="baseline", predecessor=stale.run_id, run_id=stale.run_id
        )
        message = "cyclic durable claim"
    else:
        foreign = _new_run(root, label="other")
        run = _new_run(root, label="baseline")
        _write_claim(root, label="baseline", predecessor="head", run_id=foreign.run_id)
        message = "claim from an incompatible run"
    _run_stage(root, run, 0)

    with pytest.raises(ValueError, match=message):
        _complete(root, run)

    failed = load_pipeline_run_record(root, run.run_id)
    assert failed.status == "failed"
    assert failed.complete
    assert failed.label is None
    assert failed.error is not None
    assert failed.error.type == "PipelineLabelConflict"


def test_commit_through_a_backend_without_atomic_claims_fails_durably(
    tmp_path: Path,
) -> None:
    location = str(tmp_path / "runs.zarr")
    root = zarr.open_group(location, mode="w")
    run = _new_run(root, label="baseline")
    _run_stage(root, run, 0)
    # The same files, opened through a store without conditional writes.
    fsspec_root = zarr.open_group(
        store=FsspecStore.from_url(f"file://{location}"),
        mode="r+",
    )

    with pytest.raises(RuntimeError, match="atomic set_if_not_exists"):
        _complete(fsspec_root, run)

    failed = load_pipeline_run_record(root, run.run_id)
    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.type == "PipelineLabelClaimUnavailable"
    with pytest.raises(KeyError, match="No completed pipeline run"):
        open_pipeline_run_record(root, label="baseline")


def test_commit_shares_a_claim_container_created_by_a_concurrent_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = _new_run(root, label="baseline")
    _run_stage(root, run, 0)
    create_array = zarr.Group.create_array

    def create_after_competitor(group: zarr.Group, name: str, **options: Any) -> Any:
        if name == ".label-claims":
            # Another first labeled commit creates the container after this
            # one found it missing.
            create_array(group, name, **options)
        return create_array(group, name, **options)

    monkeypatch.setattr(zarr.Group, "create_array", create_after_competitor)
    completed = _complete(root, run)

    assert completed.label == "baseline"
    assert open_pipeline_run_record(root, label="baseline") == completed


def test_stage_records_must_match_their_path_and_the_run_stage_order() -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two"))
    _run_stage(root, run, 0)
    stages = root[f"pipeline/runs/{run.run_id}/stages"]
    stage = stages["0"]
    stored = dict(stage.attrs)

    stage.attrs.put({**stored, "ordinal": 1})
    with pytest.raises(ValueError, match="Pipeline stage path 0 contains 1"):
        load_pipeline_stage_record(root, run.run_id, 0)
    stage.attrs.put({**stored, "stage": "two"})
    with pytest.raises(ValueError, match="does not match its run stage order"):
        load_pipeline_stage_records(root, run.run_id)
    stage.attrs.put(stored)
    stages.create_group("2")
    with pytest.raises(ValueError, match="ordinal is out of range: 2"):
        load_pipeline_stage_records(root, run.run_id)


def test_stage_lifecycle_rejects_unplanned_repeated_and_late_writes() -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two", "three"))
    for ordinal, stage in ((0, "two"), (3, "one")):
        with pytest.raises(ValueError, match="do not match the persisted stage order"):
            start_pipeline_stage_record(
                root, run_id=run.run_id, ordinal=ordinal, stage=stage
            )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=0, stage="one")
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(TypeError, match="PipelineStageOutputRecord"):
        finish_pipeline_stage_record(
            root,
            run_id=run.run_id,
            ordinal=0,
            status="completed",
            outputs=(PipelineOutputRecord("selection", selection),),  # type: ignore[arg-type]
            metrics=_metrics(),
        )
    finish_pipeline_stage_record(
        root, run_id=run.run_id, ordinal=0, status="completed", metrics=_metrics()
    )
    with pytest.raises(ValueError, match="Pipeline stage is already terminal"):
        finish_pipeline_stage_record(
            root, run_id=run.run_id, ordinal=0, status="failed", metrics=_metrics()
        )
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="two")

    # A worker that outlives its run may not record more stage results.
    fail_pipeline_run_record(root, run_id=run.run_id, error=RuntimeError("stopped"))
    with pytest.raises(ValueError, match="Cannot finish a stage on a terminal"):
        finish_pipeline_stage_record(
            root, run_id=run.run_id, ordinal=1, status="completed", metrics=_metrics()
        )
    with pytest.raises(ValueError, match="Cannot start a stage on a terminal"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=2, stage="three")
    assert load_pipeline_stage_record(root, run.run_id, 1).status == "running"


def test_stage_started_by_another_writer_after_the_check_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two"))
    _run_stage(root, run, 0)
    load_stages = pipeline_run_storage._load_pipeline_stage_records_for_run

    def start_competing_stage(
        store_root: zarr.Group, record: PipelineRunRecord
    ) -> tuple[Any, ...]:
        prior = load_stages(store_root, record)
        # Another writer starts the stage after this one listed the stages.
        store_root[f"pipeline/runs/{record.run_id}/stages"].create_group("1")
        return prior

    monkeypatch.setattr(
        pipeline_run_storage,
        "_load_pipeline_stage_records_for_run",
        start_competing_stage,
    )

    with pytest.raises(FileExistsError, match="Pipeline stage 1 already exists"):
        start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="two")


def test_terminal_runs_cannot_end_again() -> None:
    root = _root()
    completed = _committed_run(root, None)
    interruption = PipelineInterruptionRecord(
        kind="keyboard_interrupt",
        message="stop",
        requested_at_ns=1,
    )

    with pytest.raises(ValueError, match="already terminal"):
        _complete(root, completed)
    with pytest.raises(ValueError, match="already terminal"):
        fail_pipeline_run_record(root, run_id=completed.run_id, error=ValueError("x"))
    with pytest.raises(ValueError, match="already terminal"):
        interrupt_pipeline_run_record(
            root, run_id=completed.run_id, interruption=interruption
        )
    assert load_pipeline_run_record(root, completed.run_id) == completed


def test_completion_requires_every_stage_to_succeed_and_typed_results() -> None:
    root = _root()
    run = _new_run(root, stages=("one", "two"))
    _run_stage(root, run, 0)
    with pytest.raises(ValueError, match="requires every stage record"):
        _complete(root, run)
    start_pipeline_stage_record(root, run_id=run.run_id, ordinal=1, stage="two")
    with pytest.raises(ValueError, match="requires terminal successful stages"):
        _complete(root, run)
    finish_pipeline_stage_record(
        root,
        run_id=run.run_id,
        ordinal=1,
        status="failed",
        metrics=_metrics(),
        error=RuntimeError("stage failed"),
    )
    with pytest.raises(ValueError, match="requires terminal successful stages"):
        _complete(root, run)

    finished = _new_run(root)
    _run_stage(root, finished, 0)
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(TypeError, match="outputs must contain PipelineOutputRecord"):
        complete_pipeline_run_record(
            root,
            run_id=finished.run_id,
            outputs=(PipelineStageOutputRecord("selection", selection, False),),  # type: ignore[arg-type]
            fields=(),
        )
    with pytest.raises(TypeError, match="fields must contain PipelineFieldDescriptor"):
        complete_pipeline_run_record(
            root,
            run_id=finished.run_id,
            outputs=(),
            fields=(PipelineOutputRecord("selection", selection),),  # type: ignore[arg-type]
        )
    assert load_pipeline_run_record(root, finished.run_id).status == "running"


def test_label_lookup_fails_closed_when_copied_runs_share_a_label() -> None:
    root, other = _root(), _root()
    kept = _committed_run(root, "baseline")
    copied = _committed_run(other, "baseline")
    _copy_run(other, root, copied.run_id)

    with pytest.raises(ValueError, match="'baseline' is duplicated by runs"):
        open_pipeline_run_record(root, label="baseline")
    assert {record.run_id for record in list_pipeline_run_records(root)} == {
        kept.run_id,
        copied.run_id,
    }


def test_repack_copies_runs_from_namespaces_without_label_claims(
    tmp_path: Path,
) -> None:
    root = _prepared_store(tmp_path / "source.zarr")
    unlabeled = _committed_run(root, None)
    # A workspace whose run catalog was only partly created.
    root.create_group("analysis/pipeline")
    output = str(tmp_path / "repacked.zarr")

    repack_store(str(tmp_path / "source.zarr"), output)

    result = zarr.open_group(output, mode="r")
    assert load_pipeline_run_record(result, unlabeled.run_id) == unlabeled
    assert ".label-claims" not in result["pipeline/runs"]
    assert list_pipeline_run_records(result["analysis"]) == ()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not a claim", "Pipeline label claim is invalid"),
        (
            _claim_payload("baseline", "a" * 64),
            "Pipeline label claim digest is invalid",
        ),
    ],
    ids=["unreadable", "misfiled"],
)
def test_repack_refuses_to_copy_corrupt_label_claims(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    source = tmp_path / "source.zarr"
    root = _prepared_store(source)
    _committed_run(root, "baseline")
    _write_raw(root, f"{_CLAIMS}/{_digest('other')}/head.json", payload)

    with pytest.raises(ValueError, match=message):
        repack_store(str(source), str(tmp_path / "repacked.zarr"))


def test_repack_refuses_a_label_claim_deleted_after_it_was_listed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _committed_run(_prepared_store(source), "baseline")
    claim = source / _CLAIMS / _digest("baseline") / "head.json"
    list_claims = pipeline_run_storage.collect_aiterator

    def list_then_delete(keys: Any) -> tuple[Any, ...]:
        listed = list_claims(keys)
        # Another process deletes the claim after the copy listed it.
        claim.unlink()
        return listed

    monkeypatch.setattr(pipeline_run_storage, "collect_aiterator", list_then_delete)

    with pytest.raises(
        ValueError, match="Pipeline label claim disappeared during copy"
    ):
        repack_store(str(source), str(tmp_path / "repacked.zarr"))


def _lifecycle(events: list[PipelineEvent]) -> list[tuple[str, str | None]]:
    """Return the run's events in order, without the timing-dependent progress."""
    return [
        (event.kind, event.stage) for event in events if event.kind != "stage_progress"
    ]


@pytest.fixture
def run_tokens(monkeypatch: pytest.MonkeyPatch) -> list[ShutdownToken]:
    """Record each run's shutdown token.

    Callbacks run outside the run's shutdown scope, so a test reaches the
    token that a request from a callback must reach through this record.
    """
    tokens: list[ShutdownToken] = []

    class RecordedToken(ShutdownToken):
        def __init__(self) -> None:
            super().__init__()
            tokens.append(self)

    monkeypatch.setattr(pipeline_executor, "ShutdownToken", RecordedToken)
    return tokens


def _request_shutdown(tokens: list[ShutdownToken], reason: str) -> None:
    (token,) = tokens
    token.request(reason=reason)


def test_pipeline_run_rejects_a_bad_callback_and_read_only_stores(
    tmp_path: Path,
) -> None:
    location = _small_store(tmp_path / "data.zarr")
    datastore = DataStore(location, default_assay="RNA", min_features_per_cell=1)
    with pytest.raises(TypeError, match="callback must be callable"):
        datastore.pipeline.rna(callback="print", **_FAST_RUN)  # type: ignore[arg-type]

    read_only = DataStore(location, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError, match=r"zarr_mode='r\+'"):
        read_only.pipeline.rna(**_FAST_RUN)
    assert datastore.pipeline.list_runs() == ()


def test_shutdown_requested_between_stages_ends_the_run_interrupted_once(
    tmp_path: Path,
    run_tokens: list[ShutdownToken],
) -> None:
    location = _small_store(tmp_path / "data.zarr")
    datastore = DataStore(location, default_assay="RNA", min_features_per_cell=1)
    events: list[PipelineEvent] = []

    def stop_after_first_stage(event: PipelineEvent) -> None:
        events.append(event)
        if event.kind == "stage_completed":
            _request_shutdown(run_tokens, "stop between stages")

    with pytest.raises(ShutdownRequested, match="stop between stages"):
        datastore.pipeline.rna(callback=stop_after_first_stage, **_FAST_RUN)

    # No stage was running when the run stopped, so its one interruption
    # event names none.
    assert _lifecycle(events) == [
        ("run_started", None),
        ("stage_started", "input_snapshot"),
        ("stage_completed", "input_snapshot"),
        ("run_interrupted", None),
    ]
    interrupted_event = events[-1]
    assert interrupted_event.ordinal is None
    assert isinstance(interrupted_event.error, ShutdownRequested)
    assert datastore.pipeline.list_runs(status="running") == ()
    (interrupted,) = datastore.pipeline.list_runs(status="interrupted")
    assert interrupted_event.run_id == interrupted.run_id
    report = interrupted.report()
    assert report["run"]["complete"] is True
    assert report["run"]["interruption"]["kind"] == "shutdown_request"
    assert report["run"]["interruption"]["message"] == "stop between stages"
    assert [(stage["stage"], stage["status"]) for stage in report["stages"]] == [
        ("input_snapshot", "completed")
    ]


_N_CELLS = 120
_N_FEATURES = 40


def _graph_counts() -> np.ndarray:
    rng = np.random.default_rng(11)
    rates = rng.uniform(0.5, 4.0, size=_N_FEATURES)
    counts = rng.poisson(rates, size=(_N_CELLS, _N_FEATURES)).astype(np.uint32)
    # Both features named DUP repeat feature A.
    counts[:, 1] = counts[:, 0]
    counts[:, 2] = counts[:, 0]
    return counts


_GRAPH_COUNTS = _graph_counts()
_GRAPH_NAMES = np.asarray(
    ["A", "DUP", "DUP", *(f"G{index}" for index in range(3, _N_FEATURES))]
)


def _write_graph_store(path: Path) -> str:
    root = load_zarr(zarr_loc=str(path), mode="w")
    cell_ids = np.asarray([f"cell{index}" for index in range(_N_CELLS)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        _N_CELLS,
        feat_ids=np.asarray([f"gene{index}" for index in range(_N_FEATURES)]),
        feat_names=_GRAPH_NAMES,
        dtype="uint32",
    )
    counts[:] = _GRAPH_COUNTS
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    return str(path)


def _graph_datastore(path: Path) -> DataStore:
    location = _write_graph_store(path)
    return DataStore(location, default_assay="RNA", min_features_per_cell=1)


def _stage_statuses(run: PipelineRun) -> list[tuple[str, str]]:
    stages = run.report()["stages"]
    # A stopped run never leaves a stage running.
    assert all(stage["complete"] for stage in stages)
    return [(stage["stage"], stage["status"]) for stage in stages]


@pytest.mark.parametrize(
    ("boundary", "stages", "tail"),
    [
        pytest.param(
            ("run_started", None),
            (),
            [("run_started", None), ("run_interrupted", None)],
            id="before_the_first_stage",
        ),
        pytest.param(
            ("stage_skipped", "cell_calling"),
            _GRAPH_STAGE_STATUSES[:2],
            [("stage_skipped", "cell_calling"), ("run_interrupted", None)],
            id="after_a_skipped_stage",
        ),
        pytest.param(
            "inside_pca",
            (*_GRAPH_STAGE_STATUSES[:_PCA], ("pca", "interrupted")),
            [
                ("stage_started", "pca"),
                ("stage_interrupted", "pca"),
                ("run_interrupted", "pca"),
            ],
            id="inside_a_stage",
        ),
        pytest.param(
            "after_pca",
            _GRAPH_STAGE_STATUSES[: _PCA + 1],
            # The stage's work finished, so the run stopped between stages.
            [
                ("stage_started", "pca"),
                ("stage_completed", "pca"),
                ("run_interrupted", None),
            ],
            id="after_a_stage_action",
        ),
        pytest.param(
            ("stage_skipped", "markers"),
            _GRAPH_STAGE_STATUSES,
            [("stage_skipped", "markers"), ("run_interrupted", None)],
            id="after_the_last_stage",
        ),
    ],
)
def test_shutdown_at_each_run_boundary_ends_the_run_interrupted_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_tokens: list[ShutdownToken],
    boundary: tuple[str, str | None] | str,
    stages: tuple[tuple[str, str], ...],
    tail: list[tuple[str, str | None]],
) -> None:
    datastore = _graph_datastore(tmp_path / "data.zarr")
    reason = "stop at a run boundary"
    events: list[PipelineEvent] = []

    def observe(event: PipelineEvent) -> None:
        events.append(event)
        if (event.kind, event.stage) == boundary:
            _request_shutdown(run_tokens, reason)

    if isinstance(boundary, str):
        original_pca = type(datastore)._reduction_pca

        def interrupted_pca(store: DataStore, *args: Any, **kwargs: Any) -> Any:
            if boundary == "inside_pca":
                _request_shutdown(run_tokens, reason)
                shutdown_checkpoint()
            result = original_pca(store, *args, **kwargs)
            # The request arrives after the stage's work has returned.
            _request_shutdown(run_tokens, reason)
            return result

        monkeypatch.setattr(type(datastore), "_reduction_pca", interrupted_pca)

    with pytest.raises(ShutdownRequested, match=reason):
        datastore.pipeline.rna(callback=observe, **_GRAPH_STAGES)

    lifecycle = _lifecycle(events)
    assert lifecycle[-len(tail) :] == tail
    assert [kind for kind, _stage in lifecycle].count("run_interrupted") == 1
    assert datastore.pipeline.list_runs(status="running") == ()
    (interrupted,) = datastore.pipeline.list_runs(status="interrupted")
    assert {event.run_id for event in events} == {interrupted.run_id}
    run = interrupted.report()["run"]
    assert run["complete"] is True
    assert run["outputs"] == []
    assert (run["interruption"]["kind"], run["interruption"]["message"]) == (
        "shutdown_request",
        reason,
    )
    assert _stage_statuses(interrupted) == list(stages)


def test_shutdown_after_the_run_completed_keeps_it_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_tokens: list[ShutdownToken],
) -> None:
    datastore = _graph_datastore(tmp_path / "data.zarr")
    complete = pipeline_executor.complete_pipeline_run_record

    def complete_then_request(*args: Any, **kwargs: Any) -> PipelineRunRecord:
        record = complete(*args, **kwargs)
        _request_shutdown(run_tokens, "stop after completion")
        return record

    monkeypatch.setattr(
        pipeline_executor, "complete_pipeline_run_record", complete_then_request
    )
    events: list[PipelineEvent] = []

    with pytest.raises(ShutdownRequested, match="stop after completion"):
        datastore.pipeline.rna(label="handoff", callback=events.append, **_GRAPH_STAGES)

    # The shutdown still reaches the caller, but the durable run is complete.
    run = datastore.pipeline.open(label="handoff")
    assert run.status == "completed"
    assert run.report()["run"]["interruption"] is None
    assert _stage_statuses(run) == list(_GRAPH_STAGE_STATUSES)
    assert {"analysis_cell_selection", "pca", "connectivity_map"} <= set(run)
    assert all(datastore.artifacts.inspect(ref).complete for ref in run.values())
    # No event reports an outcome that the durable record contradicts.
    kinds = {kind for kind, _stage in _lifecycle(events)}
    assert not kinds & {"run_failed", "run_interrupted", "stage_failed"}


def test_failed_stage_emits_stage_failed_then_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = _graph_datastore(tmp_path / "data.zarr")
    expected = RuntimeError("deliberate PCA failure")

    def failing_pca(_store: DataStore, *_args: Any, **_kwargs: Any) -> Any:
        raise expected

    monkeypatch.setattr(type(datastore), "_reduction_pca", failing_pca)
    events: list[PipelineEvent] = []

    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.rna(label="failed", callback=events.append, **_GRAPH_STAGES)

    error = caught.value
    assert error.stage == "pca"
    assert error.__cause__ is expected
    lifecycle = _lifecycle(events)
    assert lifecycle[-3:] == [
        ("stage_started", "pca"),
        ("stage_failed", "pca"),
        ("run_failed", "pca"),
    ]
    assert [kind for kind, _stage in lifecycle].count("run_failed") == 1
    stage_failed, run_failed = events[-2:]
    # The stage event carries the stage's error; the run event, the run's.
    assert stage_failed.error is expected
    assert run_failed.error is error
    assert stage_failed.ordinal == run_failed.ordinal == _PCA
    assert {event.run_id for event in events} == {error.run_id}
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    assert failed.label is None
    assert failed.report()["run"]["error"] == {
        "type": "RuntimeError",
        "message": str(expected),
    }
    assert _stage_statuses(failed) == [
        *_GRAPH_STAGE_STATUSES[:_PCA],
        ("pca", "failed"),
    ]
    with pytest.raises(KeyError, match="No completed pipeline run"):
        datastore.pipeline.open(label="failed")


def test_finalize_failure_fails_the_run_after_every_stage_ended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = _graph_datastore(tmp_path / "data.zarr")
    expected = RuntimeError("deliberate finalize failure")

    def failing_completion(*_args: Any, **_kwargs: Any) -> PipelineRunRecord:
        raise expected

    monkeypatch.setattr(
        pipeline_executor, "complete_pipeline_run_record", failing_completion
    )
    events: list[PipelineEvent] = []

    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.rna(
            label="finalized", callback=events.append, **_GRAPH_STAGES
        )

    error = caught.value
    assert error.stage == "finalize"
    assert error.__cause__ is expected
    # Finalizing is not a stage: the run fails without a stage event, and its
    # event names the step outside the stage order.
    lifecycle = _lifecycle(events)
    assert lifecycle[-2:] == [("stage_skipped", "markers"), ("run_failed", "finalize")]
    assert "stage_failed" not in {kind for kind, _stage in lifecycle}
    assert (events[-1].ordinal, events[-1].error) == (None, error)
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    assert failed.label is None
    report = failed.report()
    assert report["run"]["outputs"] == []
    assert report["run"]["error"] == {"type": "RuntimeError", "message": str(expected)}
    assert _stage_statuses(failed) == list(_GRAPH_STAGE_STATUSES)
    with pytest.raises(KeyError, match="No completed pipeline run"):
        datastore.pipeline.open(label="finalized")


@pytest.mark.parametrize(
    ("axis", "code"),
    [
        # The cell view reads only fields computed on the run's cells.
        ("cells", "pipeline_field_selection_mismatch"),
        # The feature view reads only fields of the run's assay.
        ("features", "pipeline_field_axis_mismatch"),
    ],
)
def test_a_field_that_its_view_refuses_fails_the_run_before_it_takes_its_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    code: str,
) -> None:
    datastore = _graph_datastore(tmp_path / "data.zarr")
    name = "cell_fields" if axis == "cells" else "feature_fields"
    declared = getattr(rna_recipe, name)

    def with_unreadable_field(*args: Any, **kwargs: Any) -> list[Any]:
        # The artifacts follow the store, and for feature fields the assay.
        artifacts = args[-1]
        # The run's input cells are a complete artifact, which completing the
        # record accepts, but neither view can read them as a field.
        unreadable = PipelineFieldDescriptor(
            key="unreadable",
            axis="cells" if axis == "cells" else "features",
            artifact=artifacts["input_cell_selection"],
            source_value="values",
            value_index=None,
            dtype=np.dtype(bool).str,
            fill=False,
            missing_mask=None,
            display=None,
        )
        return [*declared(*args, **kwargs), unreadable]

    monkeypatch.setattr(rna_recipe, name, with_unreadable_field)
    events: list[PipelineEvent] = []

    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.rna(
            label="finalized", callback=events.append, **_GRAPH_STAGES
        )

    error = caught.value
    assert error.stage == "finalize"
    assert isinstance(error.__cause__, ArtifactResolutionError)
    assert error.__cause__.code == code
    assert _lifecycle(events)[-2:] == [
        ("stage_skipped", "markers"),
        ("run_failed", "finalize"),
    ]
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert (failed.status, failed.label) == ("failed", None)
    report = failed.report()
    assert (report["run"]["outputs"], report["run"]["fields"]) == ([], [])
    assert report["run"]["error"]["type"] == "ArtifactResolutionError"
    assert _stage_statuses(failed) == list(_GRAPH_STAGE_STATUSES)
    # The run claimed no label, so the next run with that label takes it.
    claim_owner = pipeline_run_storage._pipeline_label_claim_owner
    assert claim_owner(datastore.zw, "finalized") is None
    with pytest.raises(KeyError, match="No completed pipeline run"):
        datastore.pipeline.open(label="finalized")
    monkeypatch.setattr(rna_recipe, name, declared)
    run = datastore.pipeline.rna(label="finalized", **_GRAPH_STAGES)
    assert (run.status, run.label) == ("completed", "finalized")
    assert claim_owner(datastore.zw, "finalized") == run.run_id


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: Any) -> tuple[DataStore, PipelineRun, str]:
    location = _write_graph_store(tmp_path_factory.mktemp("graph_run") / "data.zarr")
    datastore = DataStore(location, default_assay="RNA", min_features_per_cell=1)
    run = datastore.pipeline.rna(label="baseline", **_GRAPH_RUN)
    assert run.status == "completed"
    return datastore, run, location


@pytest.mark.slow
def test_run_aware_export_aligns_layers_to_the_frozen_features(
    completed_run: tuple[DataStore, PipelineRun, str],
) -> None:
    datastore, run, _location = completed_run

    exported = datastore.to_anndata(run=run, matrix="normed", layers={"raw": "RNA"})

    # Normalized values are the run's own: the stored float32 values of its
    # normalized artifact, which cover its highly variable features.
    cells = np.flatnonzero(run.cells.fetch_all("I"))
    features = np.flatnonzero(run.features.fetch_all("highly_variable_features"))
    stored = np.asarray(datastore.artifacts.load(run["normalized"])["data"][:])
    feature_ids = datastore.RNA.feats.fetch_all("ids").astype(str)
    expected = _GRAPH_COUNTS[np.ix_(cells, features)]
    assert exported.X.dtype == np.float32
    np.testing.assert_array_equal(exported.X.toarray(), stored)
    assert exported.var_names.tolist() == feature_ids[features].tolist()
    np.testing.assert_array_equal(exported.layers["raw"].toarray(), expected)
    assert not np.array_equal(exported.X.toarray(), expected)


@pytest.mark.slow
def test_cluster_tree_fill_values_resolve_feature_names(
    completed_run: tuple[DataStore, PipelineRun, str],
) -> None:
    datastore, run, _location = completed_run
    graph, cut = run["connectivity_map"], run["paris"]

    with pytest.raises(ValueError, match="MISSING not found in RNA assay"):
        datastore.plots.cluster_tree(
            graph=graph, clusters=cut, fill_by_value="MISSING", show=False
        )
    single = datastore.plots.cluster_tree(
        graph=graph, clusters=cut, fill_by_value="A", show=False
    )
    with _captured_warnings() as messages:
        duplicated = datastore.plots.cluster_tree(
            graph=graph, clusters=cut, fill_by_value="DUP", show=False
        )
    try:
        assert "Plotting mean of 2 features because DUP is not unique." in messages
        # Both DUP features repeat A, so their mean colors the tree like A.
        assert duplicated.legends[0].label == "DUP"
        assert duplicated.legends[0].extras == single.legends[0].extras
    finally:
        single.close()
        duplicated.close()

    missing_cut = ArtifactRef(
        scope=graph.scope,
        assay=graph.assay,
        kind="cluster_cut",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(ValueError, match="complete Paris cluster-cut artifact"):
        datastore.plots.cluster_tree(graph=graph, clusters=missing_cut, show=False)


@pytest.mark.slow
def test_label_committed_by_another_run_fails_this_run_at_finalize(
    completed_run: tuple[DataStore, PipelineRun, str],
    tmp_path: Path,
) -> None:
    _datastore, _run, location = completed_run
    copy = tmp_path / "copy.zarr"
    shutil.copytree(location, copy)
    datastore = DataStore(str(copy), default_assay="RNA", min_features_per_cell=1)
    competitor: list[str] = []
    events: list[PipelineEvent] = []

    def commit_competing_run(event: PipelineEvent) -> None:
        events.append(event)
        # Another process commits the same label while this run executes.
        if event.kind == "stage_completed" and not competitor:
            competitor.append(_committed_run(datastore.zw, "contested").run_id)

    with pytest.raises(PipelineExecutionError) as caught:
        datastore.pipeline.rna(
            label="contested", callback=commit_competing_run, **_GRAPH_RUN
        )

    error = caught.value
    assert error.stage == "finalize"
    assert isinstance(error.__cause__, ValueError)
    assert "already committed" in str(error.__cause__)
    assert _lifecycle(events)[-1] == ("run_failed", "finalize")
    assert events[-1].error is error
    failed = datastore.pipeline.open(run_id=error.run_id)
    assert failed.status == "failed"
    assert failed.label is None
    assert failed.report()["run"]["error"]["type"] == "PipelineLabelConflict"
    assert all(stage["status"] != "running" for stage in failed.report()["stages"])
    assert datastore.pipeline.open(label="contested").run_id == competitor[0]
