"""Several processes analyze one store at once.

Each test starts real processes with multiprocessing's spawn context and
releases their calls together with a barrier once every process has opened
its store. What they save is compared with the same calls made one after
another, in this process, on another copy of the store: the canonical
provenance of each result, in which every input ref is replaced by the
provenance of what it names so that random artifact IDs drop out, its
attributes, and the values of every array it holds.
"""

import hashlib
import json
import multiprocessing
import os
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from cytearc import ArtifactRef, DataStore
from cytearc.storage.copy import stage_normed_array
from cytearc.storage.errors import ArtifactResolutionError

pytestmark = pytest.mark.slow

_ARTIFACT_FILE = re.compile(r"^(?:[^/]+/)?artifacts/[^/]+/[0-9a-f]{64}/")
_PER_PRODUCER = {"artifact_id", "created_at_ns", "cytearc_version", "complete"}
# The calls of a wave need only the results of earlier waves.
_TRAJECTORY_WAVES = (
    ("pseudotime", "lineage_tree"),
    ("fate", "markers", "aggregation"),
    ("curves", "lineage_markers", "lineage_aggregation", "drivers"),
    ("lineage_membership",),
)


def _open(location: Path | str) -> DataStore:
    return DataStore(str(location), default_assay="RNA", nthreads=1)


def _input(ds: DataStore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return ArtifactRef.from_dict(ds.artifacts.inspect(ref).inputs[name])


def _calls(ds: DataStore) -> dict[str, Callable[[dict[str, ArtifactRef]], ArtifactRef]]:
    """The named calls, each from the refs of what it reads."""

    def umap(refs: dict[str, ArtifactRef]) -> ArtifactRef:
        coordinates = _input(ds, _input(ds, refs["graph"], "neighbors"), "coordinates")
        initialization = ds.embeddings.initialization(coordinates)
        return ds.embeddings.umap(refs["graph"], initialization, n_epochs=50)

    trajectory = ds.trajectory
    return {
        "doublets": lambda r: ds.qc.doublets(r["leiden"], r["graph"]),
        "leiden_0.5": lambda r: ds.clusters.leiden(r["graph"], resolution=0.5),
        "leiden_0.7": lambda r: ds.clusters.leiden(r["graph"], resolution=0.7),
        "umap": umap,
        "pseudotime": lambda r: trajectory.pseudotime(
            r["graph"], source_sink=r["leiden"], sources=[1], sinks=[2]
        ),
        "pseudotime_to_3": lambda r: trajectory.pseudotime(
            r["graph"], source_sink=r["leiden"], sources=[1], sinks=[3]
        ),
        "lineage_tree": lambda r: trajectory.lineage_tree(
            r["graph"], r["leiden"], root=1
        ),
        "fate": lambda r: trajectory.fate(
            r["pseudotime"], r["leiden"], sinks=[2, 3, 4]
        ),
        "markers": lambda r: trajectory.markers(
            r["pseudotime"], features=r["detected"]
        ),
        "aggregation": lambda r: trajectory.aggregation(
            r["pseudotime"],
            features=r["detected"],
            n_clusters=15,
            window_size=50,
            chunk_size=10,
        ),
        "curves": lambda r: trajectory.curves(r["fate"], n_points=40),
        "lineage_markers": lambda r: trajectory.lineage_markers(
            r["fate"], features=r["hvgs"]
        ),
        "lineage_aggregation": lambda r: trajectory.lineage_aggregation(
            r["fate"], features=r["hvgs"], n_bins=12, n_clusters=5
        ),
        "drivers": lambda r: trajectory.drivers(r["fate"], features=r["hvgs"]),
        "lineage_membership": lambda r: trajectory.lineage_membership(r["curves"]),
    }


def _make_calls(
    ds: DataStore, names: Iterable[str], refs: dict[str, ArtifactRef]
) -> dict[str, ArtifactRef]:
    calls = _calls(ds)
    made: dict[str, ArtifactRef] = {}
    for name in names:
        made[name] = calls[name]({**refs, **made})
    return made


def _call_in_process(
    location: str,
    names: Sequence[str],
    refs: dict[str, dict[str, str]],
    size_factor: float | None,
    barrier: Any,
    result_queue: Any,
) -> None:
    try:
        ds = _open(location)
        if size_factor is not None:
            ds.RNA.sf = size_factor
        given = {name: ArtifactRef.from_dict(value) for name, value in refs.items()}
        barrier.wait(timeout=120)
        made = _make_calls(ds, names, given)
        result_queue.put(("made", {name: ref.to_dict() for name, ref in made.items()}))
    except BaseException as error:
        # The other processes stop waiting for this one.
        barrier.abort()
        result_queue.put(("failed", f"{type(error).__name__}: {error}"))


def _at_once(
    location: Path,
    groups: Sequence[Sequence[str]],
    refs: dict[str, ArtifactRef],
    *,
    size_factor: float | None = None,
) -> list[dict[str, ArtifactRef]]:
    """Make each group of calls in a process of its own, all released together."""
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(groups))
    result_queue = context.Queue()
    given = {name: ref.to_dict() for name, ref in refs.items()}
    processes = [
        context.Process(
            target=_call_in_process,
            args=(
                str(location),
                tuple(names),
                given,
                size_factor,
                barrier,
                result_queue,
            ),
        )
        for names in groups
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [result_queue.get(timeout=600) for _ in processes]
        for process in processes:
            process.join(timeout=60)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    failures = [detail for outcome, detail in outcomes if outcome == "failed"]
    assert failures == []
    return [
        {name: ArtifactRef.from_dict(value) for name, value in detail.items()}
        for _, detail in outcomes
    ]


def _merged(results: Iterable[dict[str, ArtifactRef]]) -> dict[str, ArtifactRef]:
    merged: dict[str, ArtifactRef] = {}
    for result in results:
        merged.update(result)
    return merged


def _canonical(ds: DataStore, value: Any) -> Any:
    """``value`` with each artifact ref replaced by what the artifact records."""
    if isinstance(value, dict) and value.get("type") == "artifact":
        status = ds.artifacts.inspect(ArtifactRef.from_dict(value))
        assert status.complete
        return {
            "scope": value["scope"],
            "kind": value["kind"],
            "assay": value.get("assay"),
            "provenance": _canonical(ds, status.provenance),
        }
    if isinstance(value, dict):
        return {key: _canonical(ds, item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(ds, item) for item in value]
    return value


def _arrays(group: zarr.Group, prefix: str = "") -> Iterable[tuple[str, zarr.Array]]:
    for name, member in sorted(group.members(), key=lambda item: item[0]):
        if isinstance(member, zarr.Array):
            yield prefix + name, member
        else:
            yield from _arrays(member, f"{prefix}{name}/")


def _saved(ds: DataStore, ref: ArtifactRef) -> dict[str, Any]:
    """What an artifact holds, apart from its own ID and creation."""
    group = ds.artifacts.load(ref)
    arrays = {}
    for path, array in _arrays(group):
        values = np.asarray(array[...])
        digest = hashlib.sha256(f"{values.dtype.str}{values.shape}".encode())
        if values.dtype.kind in "OUT":
            digest.update(json.dumps([str(item) for item in values.ravel()]).encode())
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
        arrays[path] = (digest.hexdigest(), _canonical(ds, dict(array.attrs)))
    attributes = {
        key: value for key, value in group.attrs.items() if key not in _PER_PRODUCER
    }
    return {
        "artifact": _canonical(ds, ref.to_dict()),
        "attributes": _canonical(ds, attributes),
        "arrays": arrays,
    }


def _files(location: Path) -> dict[str, str]:
    return {
        str(path.relative_to(location)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(location.rglob("*"))
        if path.is_file()
    }


def _outside_artifacts(files: dict[str, str]) -> dict[str, str]:
    return {
        path: digest for path, digest in files.items() if not _ARTIFACT_FILE.match(path)
    }


def _copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


@pytest.mark.parametrize(
    "calls",
    [
        ("doublets", "leiden_0.5", "umap"),
        ("pseudotime", "pseudotime_to_3", "lineage_tree"),
    ],
    ids=["doublets_beside_clustering", "trajectory_calls_on_one_graph"],
)
def test_calls_at_once_in_separate_processes_save_what_one_process_saves(
    clustered_store: tuple[Path, dict[str, ArtifactRef]],
    tmp_path: Path,
    calls: tuple[str, ...],
) -> None:
    location, refs = clustered_store
    together = _copy(location, tmp_path / "together")
    before = _files(together)
    made = _merged(_at_once(together, [(name,) for name in calls], refs))

    # The same calls one after another, in this process, on another copy.
    apart = _copy(location, tmp_path / "apart")
    alone = _make_calls(_open(apart), calls, refs)

    ds, apart_ds = _open(together), _open(apart)
    for name in calls:
        assert _saved(ds, made[name]) == _saved(apart_ds, alone[name]), name
    after = _files(together)
    # The calls added files and changed none of those already there.
    assert {
        path for path, digest in before.items() if after.get(path) != digest
    } == set()
    # Outside the artifacts that they saved, both copies hold the same files,
    # among them the group of a kind that the calls saved first.
    assert _outside_artifacts(after) == _outside_artifacts(_files(apart))
    assert ds.artifacts.prune_incomplete() == []
    assert [path for path in after if path.endswith(".partial")] == []


def test_one_call_at_once_in_separate_processes_saves_equal_results(
    clustered_store: tuple[Path, dict[str, ArtifactRef]], tmp_path: Path
) -> None:
    location, refs = clustered_store
    together = _copy(location, tmp_path / "together")
    made = [
        result["leiden_0.7"]
        for result in _at_once(together, [("leiden_0.7",)] * 3, refs)
    ]
    apart = _copy(location, tmp_path / "apart")
    expected = _saved(
        _open(apart), _make_calls(_open(apart), ["leiden_0.7"], refs)["leiden_0.7"]
    )

    ds = _open(together)

    def saved_of_the_call() -> list[ArtifactRef]:
        return ds.artifacts.list(
            kind=made[0].kind,
            complete_only=True,
            parameters={"resolution": 0.7},
            inputs={"graph": refs["graph"]},
        )

    saved = saved_of_the_call()
    # Each process returned a result that it or another saved, and the store
    # holds no other result of the call: each one equal to the call made alone.
    assert set(saved) == set(made)
    for ref in saved:
        assert _saved(ds, ref) == expected
    # A later call, in another process, takes the newest and saves nothing.
    newest = max(
        saved,
        key=lambda ref: (ds.artifacts.inspect(ref).created_at_ns, ref.artifact_id),
    )
    [later] = _at_once(together, [("leiden_0.7",)], refs)
    assert later["leiden_0.7"] == newest
    assert set(saved_of_the_call()) == set(saved)
    if len(saved) == 1:
        found = ds.artifacts.find(
            "clusters", graph=refs["graph"], method="leiden", resolution=0.7
        )
        assert found == newest
    else:
        with pytest.raises(
            ArtifactResolutionError,
            match=f"^{len(saved)} complete clusters results match",
        ):
            ds.artifacts.find(
                "clusters", graph=refs["graph"], method="leiden", resolution=0.7
            )


def test_trajectory_calls_in_separate_processes_record_the_chained_provenance(
    clustered_store: tuple[Path, dict[str, ArtifactRef]], tmp_path: Path
) -> None:
    location, refs = clustered_store
    names = [name for wave in _TRAJECTORY_WAVES for name in wave]
    chained_store = _copy(location, tmp_path / "chained")
    chained = _make_calls(_open(chained_store), names, refs)

    # Each call in a process of its own, from the refs of the calls before it;
    # the calls of a wave at once.
    separate_store = _copy(location, tmp_path / "separate")
    separate: dict[str, ArtifactRef] = {}
    for wave in _TRAJECTORY_WAVES:
        separate.update(
            _merged(
                _at_once(
                    separate_store, [(name,) for name in wave], {**refs, **separate}
                )
            )
        )
    chained_ds, separate_ds = _open(chained_store), _open(separate_store)
    for name in names:
        assert _saved(separate_ds, separate[name]) == _saved(
            chained_ds, chained[name]
        ), name

    # Another process makes each call again on the chained store: it returns
    # the chained results and writes nothing.
    before = _files(chained_store)
    [again] = _at_once(chained_store, [names], refs)
    assert again == chained
    assert _files(chained_store) == before

    # A setting of the session that a result records changes its provenance,
    # which the comparison above would report.
    [rescaled] = _at_once(
        chained_store, [("markers",)], {**refs, **chained}, size_factor=5000.0
    )
    assert _canonical(chained_ds, rescaled["markers"].to_dict()) != _canonical(
        chained_ds, chained["markers"].to_dict()
    )


def _stage_in_process(
    source_location: str,
    cache_paths: Sequence[str],
    barrier: Any,
    result_queue: Any,
) -> None:
    try:
        source = zarr.open_array(source_location, mode="r")
        expected = np.asarray(source[...])
        whole = []
        for cache_path in cache_paths:
            barrier.wait(timeout=120)
            staged = stage_normed_array(cache_path, source)
            whole.append(bool(np.array_equal(np.asarray(staged[...]), expected)))
        result_queue.put(("staged", whole))
    except BaseException as error:
        # The other processes stop waiting for this one.
        barrier.abort()
        result_queue.put(("failed", f"{type(error).__name__}: {error}"))


def test_processes_that_stage_one_array_into_one_cache_folder_read_whole_copies(
    tmp_path: Path,
) -> None:
    source_location = str(tmp_path / "source.zarr")
    source = zarr.create_array(
        source_location, shape=(20_000, 64), chunks=(2_000, 64), dtype="float32"
    )
    source[:] = np.random.default_rng(0).random((20_000, 64), dtype=np.float32)
    # Each round, three processes stage the array into one folder at once.
    cache_paths = [
        str(tmp_path / f"cache-{index}" / "artifact" / "normed.zarr")
        for index in range(8)
    ]
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_stage_in_process,
            args=(source_location, cache_paths, barrier, result_queue),
        )
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [result_queue.get(timeout=600) for _ in processes]
        for process in processes:
            process.join(timeout=60)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert outcomes == [("staged", [True] * len(cache_paths))] * 3
    for cache_path in cache_paths:
        assert os.listdir(os.path.dirname(cache_path)) == ["normed.zarr"]
