"""Composite creation reads and summarizes the segments of its count pass at once.

The count pass of ``create_composite`` reads each segment of the row plan, the
rows of one source row chunk, with ``copy_source_selection`` and adds them to
the count summary and the feature sums. It hands the segments to
``stream_shards``, so as many run at once as ``admitted_worker_split`` admits
within the budget, and each read goes through the store check of
``run_array_read``.

The tests order events instead of timing them: a read that must see another
read beside it, or a stop, waits for that event, and a 5 s deadline ends the
wait, so a wrong implementation fails instead of hanging. Every composite is
checked against summaries of its expected counts, built from the source values
and summarized in one band.
"""

import asyncio
import threading
from typing import Any

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

from cytearc import DataStore
from cytearc.composite import create_composite
from cytearc.composite import create as creation
from cytearc.composite import reader as composite_reader
from cytearc.storage import virtual_arrays
from cytearc.storage.count_matrix import CountMatrixPolicy
from cytearc.storage.identity import CountSummary
from tests.test_composite import _write_source

FAMILIES = {("MT-CO1", "MT-ND1"), ("RPL3", "RPS6")}
FEATURES = ["MT-CO1", "MT-ND1", "RPL3", "RPS6", *(f"g{index}" for index in range(8))]
# Sources a and b get chunks of 21 rows, three segments each, and c, with one
# more gene, chunks of 10 rows, four segments: ten in all.
POLICY = CountMatrixPolicy(512, 128)
DEADLINE = 5


@pytest.fixture(scope="module")
def count_pass_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("count_pass_sources")
    sources = {}
    for owner, (name, n_cells, roll, extra) in enumerate(
        (("a", 48, 0, []), ("b", 48, 5, []), ("c", 40, 9, ["x0"]))
    ):
        features = [*np.roll(FEATURES, roll).tolist(), *extra]
        values = (
            np.arange(n_cells * len(features)).reshape(n_cells, len(features))
            * (owner + 3)
            % 7
        ).astype(np.uint16)
        path = root / f"{name}.zarr"
        source = _write_source(path, values, features, policy=POLICY)
        source.z.store.close()
        sources[name] = path, values, features
    return sources


def _create(sources: dict[str, Any], at: Any, *, nthreads: int = 4) -> DataStore:
    opened = {
        name: DataStore(
            str(path),
            zarr_mode="r",
            nthreads=nthreads,
            mem_budget="64M",
            min_features_per_cell=-1,
        )
        for name, (path, _, _) in sources.items()
    }
    return create_composite(
        opened, at=at, features="intersection", mem_budget="64M", nthreads=nthreads
    )


def _expected(
    joint: DataStore, sources: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """The composite's counts in its cell and feature order, from the source values."""
    features = [str(feature) for feature in joint.RNA.feats.fetch_all("ids")]
    ids = [str(identity) for identity in joint.cells.fetch_all("ids")]
    expected = np.zeros((len(ids), len(features)), dtype=np.uint16)
    for row, identity in enumerate(ids):
        name, cell = identity.split("__")
        _, values, source_features = sources[name]
        source_row = int(cell.removeprefix("c"))
        for column, feature in enumerate(features):
            expected[row, column] = values[source_row, source_features.index(feature)]
    return expected, features


def _summaries(joint: DataStore) -> dict[str, Any]:
    matrix = joint.RNA.matrixGroup
    totals = matrix["countSummaries"]
    out: dict[str, Any] = {
        "fingerprint": matrix["counts"].attrs["content_fingerprint"],
        **{
            name: totals[name][:]
            for name in ("rowSums", "rowPositive", "columnPositive")
        },
    }
    for key, array in totals["featureSums"].arrays():
        out[f"featureSums/{key}"] = (array.attrs["features"], array[:])
    return out


def _assert_summaries_of(joint: DataStore, sources: dict[str, Any]) -> None:
    expected, features = _expected(joint, sources)
    reference = CountSummary(joint.RNA.matrixGroup["counts"])
    reference.update(0, expected)
    observed = _summaries(joint)
    assert observed["fingerprint"] == reference.hexdigest()
    np.testing.assert_array_equal(
        observed["rowSums"], expected.sum(axis=1, dtype=np.float64)
    )
    np.testing.assert_array_equal(observed["rowPositive"], (expected > 0).sum(axis=1))
    np.testing.assert_array_equal(
        observed["columnPositive"], (expected > 0).sum(axis=0)
    )
    families = set()
    for key, value in observed.items():
        if key.startswith("featureSums/"):
            indices, sums = value
            families.add(tuple(sorted(features[index] for index in indices)))
            np.testing.assert_array_equal(
                sums, expected[:, indices].sum(axis=1, dtype=np.float64)
            )
    assert families == FAMILIES


def _source_name(array: Any, sources: dict[str, Any]) -> str:
    store = array.store
    while isinstance(store, WrapperStore):
        store = store._store
    assert isinstance(store, LocalStore)
    roots = {str(path.resolve()): name for name, (path, _, _) in sources.items()}
    return roots[str(store.root.resolve())]


class SegmentReads:
    """Record the count pass's segment reads, each a ``copy_source_selection`` call."""

    def __init__(self, sources: dict[str, Any]) -> None:
        self.sources = sources
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls: list[dict[str, Any]] = []
        self.beside = threading.Event()

    def install(self, monkeypatch, *, hold_first: bool = False) -> None:
        original = composite_reader.copy_source_selection

        async def recorded(source, *args, **kwargs):  # type: ignore[no-untyped-def]
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active > 1:
                    self.beside.set()
                first = not self.calls
                self.calls.append(
                    {
                        "source": _source_name(source, self.sources),
                        "thread": threading.current_thread().name,
                        "readWorkers": kwargs.get("read_workers"),
                    }
                )
            try:
                if hold_first and first:
                    # The first read waits until a second read runs beside it.
                    async with asyncio.timeout(DEADLINE):
                        while not self.beside.is_set():
                            await asyncio.sleep(0.001)
                return await original(source, *args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

        monkeypatch.setattr(composite_reader, "copy_source_selection", recorded)


def test_segments_are_read_at_once(count_pass_sources, tmp_path, monkeypatch):
    reads = SegmentReads(count_pass_sources)
    reads.install(monkeypatch, hold_first=True)

    joint = _create(count_pass_sources, tmp_path / "joint.zarr")
    try:
        assert reads.beside.is_set(), "a second segment never ran beside the first"
        assert reads.peak > 1
        assert len(reads.calls) == 10
        # Every read ran in a worker thread of the stream, on a loop of its own.
        assert all(call["thread"] != "zarr_io" for call in reads.calls)
        _assert_summaries_of(joint, count_pass_sources)
    finally:
        joint.z.store.close()


def test_segments_at_once_give_the_summaries_of_one_segment_at_a_time(
    count_pass_sources, tmp_path, monkeypatch
):
    reads = SegmentReads(count_pass_sources)
    reads.install(monkeypatch)
    together = _create(count_pass_sources, tmp_path / "together.zarr", nthreads=4)
    peak_together, reads.peak = reads.peak, 0
    alone = _create(count_pass_sources, tmp_path / "alone.zarr", nthreads=1)
    try:
        assert peak_together > 1
        # One thread admits one segment at a time: the serial pass.
        assert reads.peak == 1
        _assert_summaries_of(together, count_pass_sources)
        _assert_summaries_of(alone, count_pass_sources)
        observed, serial = _summaries(together), _summaries(alone)
        assert observed.keys() == serial.keys()
        for key, value in observed.items():
            if key.startswith("featureSums/"):
                assert value[0] == serial[key][0]
                np.testing.assert_array_equal(value[1], serial[key][1])
            else:
                np.testing.assert_array_equal(value, serial[key])
    finally:
        together.z.store.close()
        alone.z.store.close()


def test_a_failing_segment_raises_and_stops_the_others(
    count_pass_sources, tmp_path, monkeypatch
):
    held = threading.Event()
    stopped = threading.Event()
    original_copy = composite_reader.copy_source_selection
    original_segment = creation._summarize_count_segment

    async def hold(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Reads of the later segments wait until the stream stops them.
        held.set()
        try:
            async with asyncio.timeout(DEADLINE):
                while True:
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            stopped.set()
            raise
        except TimeoutError:
            return await original_copy(*args, **kwargs)

    def segment(segment, **kwargs):  # type: ignore[no-untyped-def]
        if segment.destStart == 0:
            # The first segment fails once a later one is reading beside it.
            held.wait(timeout=DEADLINE)
            raise OSError("the first segment's read failed")
        return original_segment(segment, **kwargs)

    monkeypatch.setattr(composite_reader, "copy_source_selection", hold)
    monkeypatch.setattr(creation, "_summarize_count_segment", segment)

    with pytest.raises(OSError, match="the first segment's read failed"):
        _create(count_pass_sources, tmp_path / "joint.zarr")
    assert held.is_set()
    assert stopped.is_set(), "the stream did not stop the segments beside the failure"


def test_a_source_outside_the_store_check_reads_on_zarrs_loop(
    count_pass_sources, tmp_path, monkeypatch
):
    reads = SegmentReads(count_pass_sources)
    reads.install(monkeypatch)
    check = virtual_arrays._reads_on_any_loop

    def outside_for_b(store, path):  # type: ignore[no-untyped-def]
        # CyteArc opens composite sources only on stores that pass the check,
        # so source b is marked as one that does not, as a natively
        # asynchronous fsspec store would be.
        inner = store
        while isinstance(inner, WrapperStore):
            inner = inner._store
        if isinstance(inner, LocalStore) and inner.root.name == "b.zarr":
            return False
        return check(store, path)

    monkeypatch.setattr(virtual_arrays, "_reads_on_any_loop", outside_for_b)

    joint = _create(count_pass_sources, tmp_path / "joint.zarr")
    try:
        threads = {name: set() for name in count_pass_sources}
        for call in reads.calls:
            threads[call["source"]].add(call["thread"])
        assert threads["b"] == {"zarr_io"}
        assert threads["a"] and "zarr_io" not in threads["a"] | threads["c"]
        _assert_summaries_of(joint, count_pass_sources)
    finally:
        joint.z.store.close()


@pytest.mark.parametrize("nthreads", (1, 2, 4))
def test_the_segments_admitted_fit_the_budget(
    count_pass_sources, tmp_path, monkeypatch, nthreads
):
    reads = SegmentReads(count_pass_sources)
    reads.install(monkeypatch)
    admissions: list[dict[str, Any]] = []
    split = creation.admitted_worker_split

    def recorded_split(resources, **kwargs):  # type: ignore[no-untyped-def]
        result = split(resources, **kwargs)
        if kwargs.get("reservedFor") == "the count summary and feature sums":
            admissions.append({"resources": resources, **kwargs, "result": result})
        return result

    streams: list[tuple[int, int]] = []
    stream = creation.stream_shards

    def recorded_stream(items, fn, **kwargs):  # type: ignore[no-untyped-def]
        items = list(items)
        streams.append((len(items), kwargs["workers"]))
        return stream(items, fn, **kwargs)

    monkeypatch.setattr(creation, "admitted_worker_split", recorded_split)
    monkeypatch.setattr(creation, "stream_shards", recorded_stream)

    joint = _create(count_pass_sources, tmp_path / "joint.zarr", nthreads=nthreads)
    try:
        [admission] = admissions
        segments, reads_each = admission["result"]
        resources = admission["resources"]
        task_bytes = admission["taskBytes"]
        free = (
            resources.memoryBytes
            - admission["residentBytes"]
            - admission["reservedBytes"]
        )
        # The admitted segments and their reads fit the memory and the threads.
        assert segments * task_bytes(reads_each) <= free
        assert segments * reads_each <= nthreads
        assert segments == min(nthreads, admission["nTasks"])
        # The stream ran exactly the admitted segments, each with its reads.
        assert streams == [(admission["nTasks"], segments)]
        assert reads.peak <= segments
        assert {call["readWorkers"] for call in reads.calls} == {reads_each}
        _assert_summaries_of(joint, count_pass_sources)
    finally:
        joint.z.store.close()
