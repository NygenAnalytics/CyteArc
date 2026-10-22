"""The first writable open prepares a store once: one process, one cell filter."""

import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cytearc import DataStore, mount_datastore
from cytearc.datastore.base_datastore import (
    CELL_FILTER_ATTR,
    validate_min_features_per_cell,
)
from cytearc.storage.preparation import (
    PREPARATION_CLAIM,
    _claim_document,
    preparation_claim,
)
from cytearc.utils.logging import logger
from tests.storage_helpers import write_count_store

# The cells hold 1, 2, 3, 4, and 5 features, and the sixth holds all 12.
_FEATURES = np.array([1, 2, 3, 4, 5, 12])
_COUNTS = (np.arange(12) < _FEATURES[:, None]).astype(np.uint32)


def _fresh(path: Path) -> str:
    write_count_store(str(path), {"RNA": _COUNTS}, "uint32")
    return str(path)


def _open(location: str, **options) -> DataStore:
    return DataStore(location, default_assay="RNA", nthreads=1, **options)


@contextmanager
def _messages(level: str) -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level=level
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _filter_record(location: str) -> object:
    return zarr.open_group(location, mode="r")["cellData"].attrs.get(CELL_FILTER_ATTR)


def test_only_the_first_writable_open_filters_and_records_its_threshold(tmp_path):
    location = _fresh(tmp_path / "store.zarr")

    ds = _open(location, min_features_per_cell=2)
    assert ds.cells.fetch_all("I").tolist() == [False, False, True, True, True, True]
    assert _filter_record(location) == {"assay": "RNA", "threshold": 2}

    # The analyst brings the low-feature cells back; no later open removes them.
    ds.cells.reset_key("I")
    with _messages("INFO") as messages:
        reopened = _open(location)
        assert reopened.cells.fetch_all("I").all()
        same = _open(location, min_features_per_cell=2)
        assert same.cells.fetch_all("I").all()
        different = _open(location, min_features_per_cell=4)
    assert different.cells.fetch_all("I").all()
    assert _filter_record(location) == {"assay": "RNA", "threshold": 2}
    assert [message for message in messages if "is not applied" in message] == [
        "min_features_per_cell=4 is not applied: only the first writable open of "
        "a store filters I, and it applied min_features_per_cell=2 to assay 'RNA'. "
        "Filter cells explicitly with qc.filter or cells.update_key instead."
    ]


def test_the_first_open_applies_the_default_threshold_when_none_is_given(tmp_path):
    location = _fresh(tmp_path / "store.zarr")
    # Five of six cells hold at most ten features: half or more stay.
    with pytest.warns(UserWarning) as caught:
        ds = _open(location)
    assert ds.cells.fetch_all("I").all()
    warnings = [str(warning.message) for warning in caught]
    assert any("Will not remove low quality cells" in message for message in warnings)
    assert _filter_record(location) == {"assay": "RNA", "threshold": 10}

    keep_all = _fresh(tmp_path / "all.zarr")
    _open(keep_all, min_features_per_cell=-1)
    assert _filter_record(keep_all) == {"assay": "RNA", "threshold": -1}
    assert validate_min_features_per_cell(None) is None


def test_a_store_without_active_cells_records_the_filter_without_changing_i(tmp_path):
    location = _fresh(tmp_path / "store.zarr")
    _open(location, min_features_per_cell=-1).cells.update_key(
        np.zeros(6, dtype=bool), key="I"
    )
    root = zarr.open_group(location, mode="r+")
    del root["cellData"].attrs[CELL_FILTER_ATTR]

    ds = _open(location, min_features_per_cell=3)

    assert not ds.cells.fetch_all("I").any()
    assert _filter_record(location) == {"assay": "RNA", "threshold": 3}


def test_a_store_prepared_by_an_earlier_release_is_filtered_once_more(tmp_path):
    location = _fresh(tmp_path / "store.zarr")
    _open(location, min_features_per_cell=-1)
    # Releases before 1.3.0 recorded no filter and filtered on every open.
    root = zarr.open_group(location, mode="r+")
    del root["cellData"].attrs[CELL_FILTER_ATTR]

    filtered = _open(location, min_features_per_cell=1)
    assert filtered.cells.fetch_all("I").tolist() == [False] + [True] * 5
    filtered.cells.reset_key("I")
    assert _open(location, min_features_per_cell=1).cells.fetch_all("I").all()


def test_read_only_opens_never_filter_or_record(tmp_path):
    location = _fresh(tmp_path / "store.zarr")
    _open(location, min_features_per_cell=-1)
    root = zarr.open_group(location, mode="r+")
    del root["cellData"].attrs[CELL_FILTER_ATTR]

    reader = _open(location, zarr_mode="r", min_features_per_cell=4)

    assert reader.cells.fetch_all("I").all()
    assert _filter_record(location) is None


@pytest.mark.parametrize(
    "record",
    [
        10,
        {"assay": "RNA"},
        {"assay": "", "threshold": 1},
        {"assay": "RNA", "threshold": True},
        {"assay": "RNA", "threshold": -2},
        {"assay": "RNA", "threshold": 1, "extra": 0},
    ],
)
def test_a_malformed_filter_record_fails_closed(tmp_path, record):
    location = _fresh(tmp_path / "store.zarr")
    _open(location, min_features_per_cell=-1)
    zarr.open_group(location, mode="r+")["cellData"].attrs[CELL_FILTER_ATTR] = record

    with pytest.raises(ValueError, match="minFeaturesPerCell attribute is malformed"):
        _open(location)


def test_a_mount_filters_its_cells_on_creation_only(tmp_path):
    source = _fresh(tmp_path / "source.zarr")
    _open(source, min_features_per_cell=-1)
    target = str(tmp_path / "target.zarr")

    mounted = mount_datastore(
        source, at=target, default_assay="RNA", min_features_per_cell=2
    )
    assert mounted.cells.fetch_all("I").tolist() == [False] * 2 + [True] * 4
    assert _filter_record(target) == {"assay": "RNA", "threshold": 2}
    mounted.cells.reset_key("I")
    assert _open(target).cells.fetch_all("I").all()
    # The source keeps its own record.
    assert _filter_record(source) == {"assay": "RNA", "threshold": -1}


def test_a_held_claim_refuses_a_second_preparation_and_is_removed_after_use():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    claim_key = f"{PREPARATION_CLAIM}/zarr.json"

    with preparation_claim(root):
        assert root.store._store_dict.get(claim_key) is not None
        # The claim is a group record, so hierarchy walks stay warning-free.
        assert PREPARATION_CLAIM in dict(root.groups())
        with pytest.raises(
            RuntimeError,
            match=r"This store is being prepared by another process \(process "
            rf"{os.getpid()} on host .*, since 20\d\d-.*\)\. Open it again once that "
            r"open has finished\. If no process is preparing it, a preparation was "
            r"killed and left its claim: remove .*\.cytearc-preparation and open the "
            r"store again\.",
        ):
            with preparation_claim(root):
                raise AssertionError("a second holder must not start")
    assert claim_key not in root.store._store_dict

    with pytest.raises(OSError, match="lost"):
        with preparation_claim(root):
            raise OSError("lost")
    assert claim_key not in root.store._store_dict


def test_an_unreadable_or_vanishing_claim_is_reported_as_held(monkeypatch):
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    root = zarr.open_group(store=MemoryStore(), mode="w")
    document = root.store_path / PREPARATION_CLAIM / "zarr.json"
    sync(document.set(default_buffer_prototype().buffer.from_bytes(b"{torn")))
    with pytest.raises(
        RuntimeError, match=r"process unknown on host unknown, since an unknown time"
    ):
        with preparation_claim(root):
            raise AssertionError("unreachable")

    sync(document.delete())
    from zarr.storage._common import StorePath

    # A claim that its owner removes between the write and the read of every
    # attempt leaves this process without one.
    monkeypatch.setattr(StorePath, "set_if_not_exists", _never_written)
    with pytest.raises(RuntimeError, match="process unknown"):
        with preparation_claim(root):
            raise AssertionError("unreachable")


async def _never_written(self, value) -> None:
    return None


def test_a_store_without_conditional_writes_is_prepared_without_a_claim(
    monkeypatch,
):
    from zarr.storage._common import StorePath

    root = zarr.open_group(store=MemoryStore(), mode="w")

    async def unsupported(self, value) -> None:
        raise NotImplementedError("conditional puts are not supported")

    monkeypatch.setattr(StorePath, "set_if_not_exists", unsupported)
    ran = []
    with _messages("WARNING") as warnings:
        with preparation_claim(root):
            ran.append(True)
    assert ran == [True]
    assert warnings == [
        "Could not claim the preparation of this store (conditional puts are not "
        "supported); it is prepared without a guard against another process that "
        "opens it at the same time."
    ]
    assert f"{PREPARATION_CLAIM}/zarr.json" not in root.store._store_dict
    # A failed preparation reports its own error, not the refused write.
    with pytest.raises(RuntimeError, match="preparation failed") as failed:
        with preparation_claim(root):
            raise RuntimeError("preparation failed")
    assert failed.value.__context__ is None


def test_a_claim_that_cannot_be_removed_is_reported(monkeypatch):
    from zarr.storage._common import StorePath

    root = zarr.open_group(store=MemoryStore(), mode="w")

    async def refuse(self) -> None:
        raise OSError("store is unavailable")

    monkeypatch.setattr(StorePath, "delete_dir", refuse)
    with _messages("WARNING") as warnings:
        with preparation_claim(root):
            pass
    assert len(warnings) == 1
    assert warnings[0].startswith("Could not remove the preparation claim ")
    assert warnings[0].endswith(
        ": store is unavailable. If the store is not prepared, remove the claim "
        "before opening it again."
    )
    assert _claim_document("token").to_bytes().startswith(b"{")


def test_a_datastore_refuses_to_prepare_a_store_another_process_prepares(tmp_path):
    from zarr.core.sync import sync

    location = _fresh(tmp_path / "store.zarr")
    root = zarr.open_group(location, mode="r+")
    claim = root.store_path / PREPARATION_CLAIM / "zarr.json"
    sync(claim.set(_claim_document("another process")))

    with pytest.raises(RuntimeError, match="is being prepared by another process"):
        _open(location)
    # Nothing was prepared beside the other process.
    assert zarr.open_group(location, mode="r")["RNA"].attrs["prepared"] is False
    assert "RNA_nCounts" not in zarr.open_group(location, mode="r")["cellData"]

    # Once the other process has finished, the claim is gone and the store
    # opens; a prepared store never consults a claim.
    sync((root.store_path / PREPARATION_CLAIM).delete_dir())
    assert _open(location).cells.N == 6
    assert not (Path(location) / PREPARATION_CLAIM).exists()
    sync(claim.set(_claim_document("stale")))
    assert _open(location).cells.N == 6


_CHILD = textwrap.dedent(
    """
    import os
    import signal
    import sys
    import time
    from pathlib import Path

    import cytearc.assay.base as base
    from cytearc import DataStore

    location, mode = sys.argv[1], sys.argv[2]
    original = base.Assay.prepare

    def paused(self, patterns):
        if mode == "hold":
            print("PREPARING", flush=True)
            release = Path(location + ".release")
            while not release.exists():
                time.sleep(0.02)
        elif mode == "kill":
            print("PREPARING", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        return original(self, patterns)

    base.Assay.prepare = paused
    try:
        ds = DataStore(location, default_assay="RNA", nthreads=1)
    except RuntimeError as error:
        print("REFUSED", error, flush=True)
        sys.exit(3)
    print("OPENED", int(ds.cells.fetch_all("I").sum()), flush=True)
    """
)


def _child(location: str, mode: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD, location, mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for(process: subprocess.Popen, line: str, timeout: float = 60.0) -> None:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.5)
        if ready:
            output = process.stdout.readline()
            if output.strip() == line:
                return
            if not output:
                break
    process.kill()
    _, errors = process.communicate()
    raise AssertionError(f"child did not print {line!r}: {errors}")


def _residue(location: str) -> list[str]:
    return sorted(
        str(path.relative_to(location))
        for path in Path(location).rglob("*")
        if path.name.endswith(".partial") or PREPARATION_CLAIM in path.parts
    )


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
def test_concurrent_first_opens_in_separate_processes_prepare_once(tmp_path):
    location = _fresh(tmp_path / "store.zarr")
    holder = _child(location, "hold")
    try:
        _wait_for(holder, "PREPARING")
        # A second process that opens the fresh import meanwhile is refused
        # with the claim's owner, instead of preparing the same columns.
        second = _child(location, "open")
        stdout, _ = second.communicate(timeout=120)
        assert second.returncode == 3
        assert stdout.startswith(
            f"REFUSED This store is being prepared by another process (process "
            f"{holder.pid} on host "
        )
        assert zarr.open_group(location, mode="r")["RNA"].attrs["prepared"] is False
        Path(location + ".release").touch()
        stdout, stderr = holder.communicate(timeout=120)
    finally:
        holder.kill()
    assert holder.returncode == 0, stderr
    assert stdout.splitlines()[-1] == "OPENED 6"
    assert zarr.open_group(location, mode="r")["RNA"].attrs["prepared"] is True
    assert _residue(location) == []


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
def test_a_killed_preparation_names_its_claim_until_it_is_removed(tmp_path):
    import shutil

    location = _fresh(tmp_path / "store.zarr")
    killed = _child(location, "kill")
    killed.communicate(timeout=120)
    assert killed.returncode == -signal.SIGKILL

    refused = _child(location, "open")
    stdout, _ = refused.communicate(timeout=120)
    assert refused.returncode == 3
    assert f"process {killed.pid} on host " in stdout
    assert stdout.rstrip().endswith(
        f"remove file://{location}/{PREPARATION_CLAIM} and open the store again."
    )

    # The operator confirms that the process stopped and removes the claim.
    shutil.rmtree(Path(location) / PREPARATION_CLAIM)
    opened = _child(location, "open")
    stdout, stderr = opened.communicate(timeout=120)
    assert opened.returncode == 0, stderr
    assert stdout.splitlines()[-1] == "OPENED 6"
    assert _residue(location) == []
