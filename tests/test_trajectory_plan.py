"""Plan the calls of a trajectory with ds.trajectory.plan() before running them.

A plan is checked against what the chain then does: the artifacts that each
real call creates, which ``artifact_plan_scope`` records as ``plan_artifact``
decides them, and the refs that the calls return. Planning is checked to
write nothing, by a digest of every stored file, and to read no count matrix,
through a store that logs each request.
"""

import hashlib
import json
import multiprocessing
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from zarr.storage import LocalStore

from cytearc import ArtifactRef, DataStore
from cytearc.datastore.trajectory_plan import TrajectoryCallPlan, TrajectoryPlan
from cytearc.storage.artifact_writer import artifact_plan_scope
from cytearc.trajectory import SourceSinkLabelsError
from tests.test_metadata_store_reads import _CountingStore

# A chunk of an assay's count matrix, or of its transpose.
_COUNT_CHUNK = re.compile(r"^[^/]+/countsT?/c/")
_WRITES = {"set", "set_if_not_exists", "delete"}
# The calls of the chain in order, each with the earlier calls whose results it takes.
_CHAIN_INPUTS = {
    "lineage_tree": (),
    "pseudotime": (),
    "fate": ("pseudotime",),
    "markers": ("pseudotime",),
    "aggregation": ("pseudotime",),
    "curves": ("fate",),
    "lineage_membership": ("curves",),
    "branch_cells": ("lineage_membership",),
    "lineage_markers": ("fate",),
    "drivers": ("fate",),
    "lineage_aggregation": ("fate",),
    "states": (),
}


def _open(location: Path | str, **options: Any) -> DataStore:
    return DataStore(str(location), default_assay="RNA", nthreads=1, **options)


def _score(ds: DataStore) -> pd.Series:
    """A per-cell score keyed by cell ID, which states saves as a snapshot."""
    counts = ds.cells.fetch("RNA_nCounts", key="I").astype(float)
    return pd.Series(counts, index=ds.cells.fetch("ids", key="I"))


def _chain(
    ds: DataStore, refs: dict[str, ArtifactRef], score: pd.Series, *, sinks=(2, 3, 4)
) -> dict[str, Any]:
    """Make the calls of a trajectory, each from the results of those before it."""
    trajectory = ds.trajectory
    graph, clusters = refs["graph"], refs["leiden"]
    made: dict[str, Any] = {}
    made["lineage_tree"] = trajectory.lineage_tree(graph, clusters, root=1)
    made["pseudotime"] = trajectory.pseudotime(
        graph, source_sink=clusters, sources=[1], sinks=[2]
    )
    made["fate"] = trajectory.fate(made["pseudotime"], clusters, sinks=list(sinks))
    made["markers"] = trajectory.markers(made["pseudotime"], features=refs["detected"])
    made["aggregation"] = trajectory.aggregation(
        made["pseudotime"],
        features=refs["detected"],
        n_clusters=15,
        window_size=50,
        chunk_size=10,
    )
    made["curves"] = trajectory.curves(made["fate"], n_points=40)
    made["lineage_membership"] = trajectory.lineage_membership(made["curves"])
    made["branch_cells"] = trajectory.branch_cells(made["lineage_membership"], [2, 3])
    made["lineage_markers"] = trajectory.lineage_markers(
        made["fate"], features=refs["hvgs"]
    )
    made["drivers"] = trajectory.drivers(made["fate"], features=refs["hvgs"])
    made["lineage_aggregation"] = trajectory.lineage_aggregation(
        made["fate"], features=refs["hvgs"], n_bins=12, n_clusters=5
    )
    made["states"] = trajectory.states(graph, score, n_states=3, labels=clusters)
    return made


def _plan(ds: DataStore, refs, score, **options: Any) -> TrajectoryPlan:
    with ds.trajectory.plan() as plan:
        _chain(ds, refs, score, **options)
    return plan


class _Receipts:
    """Make each call of the chain for real, recording what ``plan_artifact`` decided."""

    def __init__(self, ds: DataStore) -> None:
        self._trajectory = ds.trajectory
        self.made: dict[str, list[tuple[str, str]]] = {}

    def __getattr__(self, name: str) -> Any:
        call = getattr(self._trajectory, name)

        def recorded(*args: Any, **kwargs: Any) -> ArtifactRef:
            with artifact_plan_scope() as receipts:
                ref = call(*args, **kwargs)
            self.made[name] = [
                (receipt.operation, receipt.disposition) for receipt in receipts
            ]
            return ref

        return recorded


def _run_recorded(ds: DataStore, refs, score, **options: Any):
    receipts = _Receipts(ds)
    made = _chain(SimpleNamespace(trajectory=receipts), refs, score, **options)
    return made, receipts.made


def _digest(location: Path) -> str:
    """Return a digest of every file's path and bytes below ``location``."""
    digest = hashlib.sha256()
    for path in sorted(location.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(location)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_plan_matches_run(plan: TrajectoryPlan, made, receipts) -> None:
    """Check a plan against the artifacts that the chain then created and reused."""
    names = list(_CHAIN_INPUTS)
    assert [call.call for call in plan.calls] == names
    for ordinal, name in enumerate(names):
        planned = plan[ordinal]
        created = [operation for operation, done in receipts[name] if done == "created"]
        assert planned.ordinal == ordinal
        assert planned.status == ("computed" if created else "reused"), name
        if planned.status == "reused":
            assert planned.ref == made[name]
            assert planned.reason is None
        elif planned.reason == "upstream":
            assert planned.operation is None
        else:
            # The first artifact that the call would compute is the one it
            # created first.
            assert planned.operation == created[0], name


def _expected_upstream(plan: TrajectoryPlan, name: str) -> tuple[int, ...]:
    """The positions of the computed calls whose results ``name`` takes."""
    positions = {call.call: call.ordinal for call in plan.calls}
    return tuple(
        sorted(
            positions[source]
            for source in _CHAIN_INPUTS[name]
            if plan[source].status == "computed"
        )
    )


@pytest.fixture(scope="module")
def chained(clustered_store, tmp_path_factory):
    """Plan the chain on a store without trajectories, then run it for real."""
    source, refs = clustered_store
    location = tmp_path_factory.mktemp("trajectory_plan") / "data.zarr"
    shutil.copytree(source, location)
    ds = _open(location)
    score = _score(ds)
    first = _plan(ds, refs, score)
    made, receipts = _run_recorded(ds, refs, score)
    return SimpleNamespace(
        location=location,
        refs=refs,
        score=score,
        first=first,
        made=made,
        receipts=receipts,
    )


def _copy(location: Path, tmp_path: Path) -> Path:
    target = tmp_path / "data.zarr"
    shutil.copytree(location, target)
    return target


@pytest.mark.slow
def test_a_plan_says_what_the_chain_then_computes_and_reuses(chained) -> None:
    first = chained.first
    _assert_plan_matches_run(first, chained.made, chained.receipts)
    # The store held no trajectory: the calls on the graph alone compute
    # because nothing of their operations exists, and every other call waits
    # on the call whose result it takes.
    assert first.reused == ()
    for name in ("lineage_tree", "pseudotime", "states"):
        assert first[name].reason == "absent"
    assert first["states"].operation == "snapshot_fate_direction"
    for name, sources in _CHAIN_INPUTS.items():
        if sources:
            assert first[name].reason == "upstream"
            assert first[name].upstream == _expected_upstream(first, name)
            assert all(first[i].call in sources for i in first[name].upstream)

    # On the chained store, a plan reuses every call with the chain's refs and
    # writes nothing.
    before = _digest(chained.location)
    again = _plan(_open(chained.location), chained.refs, chained.score)
    assert _digest(chained.location) == before
    assert again.computed == ()
    assert [call.ref for call in again.calls] == list(chained.made.values())


@pytest.mark.slow
def test_a_plan_of_a_chained_store_reads_no_counts_on_a_read_only_open(chained) -> None:
    store = _CountingStore(LocalStore(str(chained.location), read_only=True))
    ds = DataStore(store, default_assay="RNA", zarr_mode="r", nthreads=1)
    store.log.clear()
    plan = _plan(ds, chained.refs, chained.score)
    assert plan.computed == ()
    assert not [entry for entry in store.log if entry[0] in _WRITES]
    gets = [key for operation, key in store.log if operation == "get"]
    assert not [key for key in gets if _COUNT_CHUNK.match(key)]
    # The calls load the graph that they walk, as they do when they run.
    assert any("/connectivity_map/" in key and "/c/" in key for key in gets)


def test_a_plan_of_a_store_without_trajectories_writes_nothing_and_reads_no_counts(
    clustered_store, tmp_path
) -> None:
    source, refs = clustered_store
    location = _copy(source, tmp_path)
    before = _digest(location)
    plan = _plan(_open(location), refs, _score(_open(location)))
    assert _digest(location) == before
    assert plan.computed == tuple(_CHAIN_INPUTS)
    for name, sources in _CHAIN_INPUTS.items():
        assert plan[name].reason == ("upstream" if sources else "absent")
        assert plan[name].upstream == _expected_upstream(plan, name)

    store = _CountingStore(LocalStore(str(location), read_only=True))
    ds = DataStore(store, default_assay="RNA", zarr_mode="r", nthreads=1)
    score = _score(ds)
    store.log.clear()
    read_only = _plan(ds, refs, score)
    assert read_only.to_dict() == plan.to_dict()
    assert not [entry for entry in store.log if entry[0] in _WRITES]
    gets = [key for operation, key in store.log if operation == "get"]
    assert not [key for key in gets if _COUNT_CHUNK.match(key)]


@pytest.mark.slow
def test_a_changed_argument_computes_its_call_and_every_call_that_takes_its_result(
    chained, tmp_path
) -> None:
    location = _copy(chained.location, tmp_path)
    ds = _open(location)
    plan = _plan(ds, chained.refs, chained.score, sinks=(2, 3))
    assert plan["fate"].status == "computed"
    assert plan["fate"].reason == "changed"
    assert plan["fate"].changed_parameters == ("sinks",)
    assert plan["fate"].existing == chained.made["fate"]
    waiting = ("curves", "lineage_membership", "branch_cells", "lineage_markers")
    for name in (*waiting, "drivers", "lineage_aggregation"):
        assert plan[name].reason == "upstream"
        assert plan[name].upstream == _expected_upstream(plan, name)
    assert plan.reused == (
        "lineage_tree",
        "pseudotime",
        "markers",
        "aggregation",
        "states",
    )

    made, receipts = _run_recorded(ds, chained.refs, chained.score, sinks=(2, 3))
    _assert_plan_matches_run(plan, made, receipts)


def _plan_in_process(location: str, refs: dict, score: pd.Series, results) -> None:
    try:
        given = {name: ArtifactRef.from_dict(value) for name, value in refs.items()}
        plan = _plan(_open(location), given, score)
        results.put(("planned", plan.to_dict()))
    except BaseException as error:
        results.put(("failed", f"{type(error).__name__}: {error}"))


@pytest.mark.slow
def test_a_plan_made_in_another_process_matches_the_plan_made_here(chained) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    given = {name: ref.to_dict() for name, ref in chained.refs.items()}
    process = context.Process(
        target=_plan_in_process,
        args=(str(chained.location), given, chained.score, results),
    )
    process.start()
    try:
        outcome, planned = results.get(timeout=600)
        process.join(timeout=60)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert outcome == "planned", planned
    here = _plan(_open(chained.location), chained.refs, chained.score)
    assert planned == here.to_dict()
    assert [call["status"] for call in planned["calls"]] == ["reused"] * 12


def test_the_limits_of_a_plans_block_name_what_to_do_instead(
    clustered_store, tmp_path
) -> None:
    source, refs = clustered_store
    location = _copy(source, tmp_path)
    ds = _open(location)
    graph, clusters = refs["graph"], refs["leiden"]
    before = _digest(location)

    # A call that the plan does not cover may reuse, and may not compute.
    with pytest.raises(
        RuntimeError,
        match=r"^A call inside ds\.trajectory\.plan\(\) would compute a new "
        r"run_leiden_clustering result, and a plan writes nothing\. .* make any "
        r"other call before the plan or after its with block\.$",
    ):
        with ds.trajectory.plan():
            assert ds.clusters.leiden(graph) == clusters
            ds.clusters.leiden(graph, resolution=0.33)
    with pytest.raises(RuntimeError, match="^A trajectory plan is already open"):
        with ds.trajectory.plan():
            with ds.trajectory.plan():
                pass
    other = _open(location)
    with pytest.raises(
        RuntimeError,
        match="^A trajectory plan of another DataStore is open, so lineage_tree",
    ):
        with ds.trajectory.plan():
            other.trajectory.lineage_tree(graph, clusters, root=1)
    used = ds.trajectory.plan()
    with used:
        planned = ds.trajectory.pseudotime(
            graph, source_sink=clusters, sources=[1], sinks=[2]
        )
    with pytest.raises(
        RuntimeError, match="^This trajectory plan has already been used"
    ):
        with used:
            pass
    assert _digest(location) == before

    # The plan of a call stands for no saved result: neither another call, nor
    # a planned call in a later plan or after the block, takes it.
    assert isinstance(planned, TrajectoryCallPlan)
    refused = (
        r"was given the plan of pseudotime, call 0 of a trajectory plan, which "
        r"would compute a result that does not exist yet\. Run pseudotime outside "
        r"ds\.trajectory\.plan\(\) to save its result, and pass the ref that it "
        r"returns\.$"
    )
    with pytest.raises(TypeError, match="^inspect " + refused):
        ds.artifacts.inspect(planned)
    with pytest.raises(TypeError, match="^load_pseudotime " + refused):
        ds.trajectory.load_pseudotime(planned)
    with pytest.raises(TypeError, match="^fate " + refused):
        with ds.trajectory.plan():
            ds.trajectory.fate(planned, clusters, sinks=[2, 3])
    with pytest.raises(TypeError, match="^fate " + refused):
        ds.trajectory.fate(planned, clusters, sinks=[2, 3])
    assert _digest(location) == before

    # After the block, the calls run and save their results unplanned.
    tree = ds.trajectory.lineage_tree(graph, clusters, root=1)
    assert len(used.calls) == 1
    assert ds.artifacts.inspect(tree).complete


def test_a_plans_calls_are_found_by_position_or_name(clustered_store, tmp_path) -> None:
    source, refs = clustered_store
    ds = _open(_copy(source, tmp_path))
    graph, clusters = refs["graph"], refs["leiden"]
    tree = ds.trajectory.lineage_tree(graph, clusters, root=1)
    with ds.trajectory.plan() as plan:
        assert plan.calls == ()
        # A call whose result the store holds returns the result's ref.
        assert ds.trajectory.lineage_tree(graph, clusters, root=1) == tree
        first = ds.trajectory.pseudotime(
            graph, source_sink=clusters, sources=[1], sinks=[2]
        )
        second = ds.trajectory.pseudotime(
            graph, source_sink=clusters, sources=[1], sinks=[3]
        )
        fate = ds.trajectory.fate(second, clusters, sinks=[3])
        # A call that refuses its arguments raises from the block, as it does
        # when it runs.
        with pytest.raises(SourceSinkLabelsError):
            ds.trajectory.pseudotime(
                graph, source_sink=clusters, sources=[1], sinks=[999]
            )
    reused = plan[0]
    assert plan.calls == (reused, first, second, fate)
    assert (reused.call, reused.status, reused.ref, reused.reason) == (
        "lineage_tree",
        "reused",
        tree,
        None,
    )
    assert plan[3] is fate
    assert plan["fate"] is fate
    assert fate.upstream == (2,)
    assert fate.detail == (
        "It takes what pseudotime (call 2) would compute, so its identity is "
        "known only after that runs"
    )
    with pytest.raises(KeyError, match=r"2 pseudotime calls, at positions \[1, 2\]"):
        plan["pseudotime"]
    with pytest.raises(KeyError, match="no curves call"):
        plan["curves"]
    with pytest.raises(IndexError):
        plan[4]
    with pytest.raises(TypeError, match="by position or name"):
        plan[True]
    assert plan.computed == ("pseudotime", "pseudotime", "fate")
    assert plan.reused == ("lineage_tree",)
    stored = json.loads(json.dumps(plan.to_dict()))
    assert stored["calls"][0]["ref"] == tree.to_dict()
    assert stored["calls"][3] == {
        "call": "fate",
        "ordinal": 3,
        "status": "computed",
        "reason": "upstream",
        "detail": fate.detail,
        "operation": None,
        "ref": None,
        "existing": None,
        "changedParameters": [],
        "changedInputs": [],
        "upstream": [2],
    }
    assert stored["calls"][1]["operation"] == "run_pseudotime_scoring"
    assert stored["calls"][1]["reason"] == "absent"
