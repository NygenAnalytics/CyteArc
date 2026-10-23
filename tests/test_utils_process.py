"""Behavioral tests for process helpers."""

import os
import io
import sys
import threading
from pathlib import Path

import pytest

import cytearc.utils.process as process_module
from cytearc.utils.process import (
    process_rss_mb,
    read_process_tree_rss_bytes,
    rss_text,
    sample_process_tree_rss,
    suppress_native_output,
)


def _fake_proc_root(tmp_path: Path, status: bytes) -> Path:
    proc_root = tmp_path / "proc"
    (proc_root / "self").mkdir(parents=True)
    (proc_root / "self" / "status").write_bytes(status)
    return proc_root


def test_process_rss_mb_reads_proc_status(tmp_path, monkeypatch):
    # A process name truncated inside a multibyte character is not UTF-8.
    proc_root = _fake_proc_root(tmp_path, b"Name:\tpy\xe6\x97\nVmRSS:\t2048 kB\n")
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)
    assert process_rss_mb() == pytest.approx(2.0)
    assert rss_text() == "2 MiB"


@pytest.mark.parametrize("status", [None, b"Name:\tpython\nVmSize:\t2048 kB\n"])
def test_process_rss_mb_is_none_without_proc_vmrss(tmp_path, monkeypatch, status):
    # macOS and Windows have no /proc. CyteArc reports nothing rather than the
    # peak resident size from getrusage, which macOS reports in bytes.
    proc_root = (
        tmp_path / "missing" if status is None else _fake_proc_root(tmp_path, status)
    )
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)
    assert process_rss_mb() is None
    assert rss_text() == "n/a"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux reports VmRSS in /proc")
def test_process_rss_mb_reports_current_resident_memory_on_linux():
    rss = process_rss_mb()
    assert rss is not None and rss > 0
    amount, unit = rss_text().split(" ")
    assert unit == "MiB" and int(amount) > 0


def test_rna_feature_stats_log_rss_without_proc(tmp_path, monkeypatch):
    import numpy as np

    from cytearc.assay import RNAassay
    from cytearc.metadata import MetaData
    from cytearc.utils.logging import logger
    from tests.test_counts_t import _memory_root, _write_small_assay

    root = _memory_root()
    values = np.array([[4, 0, 1], [3, 2, 1], [0, 5, 1]], dtype=np.uint32)
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(root, "RNA", cells, workspace=None, nthreads=1)
    assay.sf = 1000
    monkeypatch.setattr(process_module, "_PROC_ROOT", tmp_path / "missing")
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message), level="DEBUG")
    try:
        assay._streaming_feature_stats(np.arange(3), np.arange(3))
    finally:
        logger.remove(sink)

    bands = [message for message in messages if "feature stats band" in message]
    assert bands
    assert all(message.rstrip().endswith("rss n/a") for message in bands), bands


def test_process_tree_sampler_starts_no_thread_without_proc(tmp_path, monkeypatch):
    monkeypatch.setattr(process_module, "_PROC_ROOT", tmp_path / "missing")
    started: list[str] = []
    real_start = threading.Thread.start

    def recording_start(thread: threading.Thread) -> None:
        started.append(thread.name)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", recording_start)
    with sample_process_tree_rss(interval_seconds=0.01) as measurement:
        during = measurement()
    final = measurement()

    assert "cytearc-pipeline-rss" not in started
    for observed in (during, final):
        assert (observed.baseline_bytes, observed.peak_bytes) == (None, None)
        assert observed.incremental_peak_bytes is None
        assert (observed.sample_count, observed.sampling_error_count) == (0, 0)
        assert observed.sample_interval_seconds == 0.01
        assert observed.unavailable_reason == (
            "process-tree RSS requires the Linux /proc filesystem"
        )


def test_process_tree_sampler_reads_the_patched_proc_root(tmp_path, monkeypatch):
    proc_root = _fake_proc_root(tmp_path, b"PPid:\t1\nVmRSS:\t4 kB\n")
    pid_status = proc_root / str(os.getpid()) / "status"
    pid_status.parent.mkdir()
    pid_status.write_bytes(b"PPid:\t1\nVmRSS:\t8 kB\n")
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)

    with sample_process_tree_rss(interval_seconds=3600.0) as measurement:
        pass

    final = measurement()
    assert (final.baseline_bytes, final.peak_bytes) == (8 * 1024, 8 * 1024)
    assert (final.sample_count, final.sampling_error_count) == (2, 0)
    assert final.unavailable_reason is None


def test_process_tree_rss_sums_only_root_and_descendants(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    statuses = {
        100: "PPid:\t1\nVmRSS:\t10 kB\n",
        101: "PPid:\t100\nVmRSS:\t20 kB\n",
        102: "PPid:\t101\nVmRSS:\t30 kB\n",
        200: "PPid:\t1\nVmRSS:\t999 kB\n",
    }
    for pid, contents in statuses.items():
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text(contents)

    assert read_process_tree_rss_bytes(100, proc_root=proc_root) == 60 * 1024


def _write_proc_tree(
    proc_root: Path,
    processes: dict[int, tuple[int, str | None, dict[int, list[int]]]],
) -> None:
    """Write status and per-thread children files of fake processes.

    Each process maps to its parent, its VmRSS line (None for a process, such
    as a zombie, that reports none), and the children of each of its threads.
    """
    for pid, (parent, rss, threads) in processes.items():
        directory = proc_root / str(pid)
        (directory / "task").mkdir(parents=True)
        status = f"Name:\tp{pid}\nPPid:\t{parent}\n" + ("" if rss is None else rss)
        (directory / "status").write_text(status)
        for tid, children in threads.items():
            (directory / "task" / str(tid)).mkdir()
            listed = "".join(f"{child} " for child in children)
            (directory / "task" / str(tid) / "children").write_text(listed)


_PROC_TREES = {
    "chain": {
        100: (1, "VmRSS:\t10 kB\n", {100: [101]}),
        101: (100, "VmRSS:\t20 kB\n", {101: [102]}),
        102: (101, "VmRSS:\t30 kB\n", {102: []}),
        200: (1, "VmRSS:\t999 kB\n", {200: [201]}),
        201: (200, "VmRSS:\t5 kB\n", {201: []}),
    },
    # Children forked by different threads are listed by those threads.
    "threads": {
        100: (1, "VmRSS:\t1 mB\n", {100: [101], 105: [102, 103], 106: []}),
        101: (100, "VmRSS:\t2 kB\n", {101: []}),
        102: (100, "VmRSS:\t3 kB\n", {102: [104]}),
        103: (100, None, {103: []}),
        104: (102, "VmRSS:\t4 kB\n", {104: []}),
        300: (1, "VmRSS:\t7 kB\n", {300: []}),
    },
    "leaf_root": {
        100: (1, "VmRSS:\t6 kB\n", {100: []}),
        101: (1, "VmRSS:\t9 kB\n", {101: []}),
    },
    "no_measurement": {
        100: (1, None, {100: [101]}),
        101: (100, None, {101: []}),
    },
}


@pytest.mark.parametrize("name", sorted(_PROC_TREES))
def test_process_tree_walk_finds_what_the_parent_scan_finds(
    tmp_path: Path, name: str
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_tree(proc_root, _PROC_TREES[name])

    walked = read_process_tree_rss_bytes(100, proc_root=proc_root)
    scanned = read_process_tree_rss_bytes(
        100,
        proc_root=proc_root,
        list_pids=process_module._default_list_pids,
    )
    assert walked == scanned
    assert process_module._tree_pids_by_children(proc_root, 100) == (
        process_module._tree_pids_by_scan(
            proc_root,
            100,
            {},
            process_module._default_list_pids,
            process_module._default_read_text,
        )
    )


def test_process_tree_walk_reads_only_the_tree(tmp_path: Path, monkeypatch) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_tree(proc_root, _PROC_TREES["threads"])
    read: list[str] = []

    def read_text(path: Path) -> str:
        read.append(path.parent.name)
        return path.read_text()

    def no_listing(_root: Path) -> list[int]:
        raise AssertionError("the walk must not list every process")

    monkeypatch.setattr(process_module, "_default_list_pids", no_listing)
    total = read_process_tree_rss_bytes(100, proc_root=proc_root, read_text=read_text)

    assert total == 1024**2 + (2 + 3 + 4) * 1024
    assert sorted(read) == ["100", "101", "102", "103", "104"]


def test_process_tree_walk_skips_processes_and_threads_that_exited(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_tree(
        proc_root,
        {
            100: (1, "VmRSS:\t10 kB\n", {100: [101, 102, 100]}),
            101: (100, "VmRSS:\t20 kB\n", {101: []}),
        },
    )
    # A thread that exited after the listing leaves no directory to read.
    (proc_root / "100" / "task" / "107").write_text("")
    # A children file that cannot be read as text is passed over.
    (proc_root / "101" / "task" / "108" / "children").mkdir(parents=True)

    # Process 102 exited: it has no status and no threads, and the root's own
    # PID among its children does not loop.
    assert read_process_tree_rss_bytes(100, proc_root=proc_root) == 30 * 1024
    assert process_module._tree_pids_by_children(proc_root, 100) == {100, 101, 102}


def test_process_tree_rss_is_unavailable_when_processes_cannot_be_listed() -> None:
    def listing(_root: Path) -> list[int]:
        raise PermissionError("no /proc listing")

    assert read_process_tree_rss_bytes(100, list_pids=listing) is None


def test_process_tree_falls_back_to_the_parent_scan_without_children_files(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_tree(proc_root, _PROC_TREES["chain"])
    # A kernel without CONFIG_PROC_CHILDREN lists threads without the file.
    (proc_root / "100" / "task" / "100" / "children").unlink()

    assert process_module._tree_pids_by_children(proc_root, 100) is None
    assert read_process_tree_rss_bytes(100, proc_root=proc_root) == 60 * 1024


@pytest.mark.parametrize("listed", [b"101 x ", b"\xff101 "], ids=["text", "binary"])
def test_process_tree_falls_back_to_the_parent_scan_for_unreadable_lists(
    tmp_path: Path, listed: bytes
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_tree(proc_root, _PROC_TREES["chain"])
    (proc_root / "100" / "task" / "100" / "children").write_bytes(listed)

    assert process_module._tree_pids_by_children(proc_root, 100) is None
    assert read_process_tree_rss_bytes(100, proc_root=proc_root) == 60 * 1024


@pytest.mark.skipif(sys.platform != "linux", reason="Linux lists children in /proc")
def test_process_tree_walk_matches_the_parent_scan_on_this_host() -> None:
    import subprocess

    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE),
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]),
    ]
    try:
        assert processes[0].stdout is not None
        grandchild = int(processes[0].stdout.readline())
        proc_root = Path("/proc")
        walked = process_module._tree_pids_by_children(proc_root, os.getpid())
        scanned = process_module._tree_pids_by_scan(
            proc_root,
            os.getpid(),
            {},
            process_module._default_list_pids,
            process_module._default_read_text,
        )
        assert walked is not None
        assert {os.getpid(), grandchild, *(p.pid for p in processes)} <= walked
        assert walked == scanned
    finally:
        for process in processes:
            process.kill()
            process.wait()


def test_process_tree_rss_skips_malformed_status_lines() -> None:
    statuses = {
        100: "Name:\tpython\nno separator here\nPPid:\t1\nVmRSS:\t10 kB\n",
        # A parent that is not a number and an unreadable amount are ignored.
        101: "PPid:\tnot-a-pid\nVmRSS:\tlots kB\n",
        102: "PPid:\t100\nVmRSS:\t2 mB\nVmRSS:\n",
        103: "PPid:\t100\nVmRSS:\t5 pages\n",
    }

    def read_text(path: Path) -> str:
        return statuses[int(path.parent.name)]

    total = read_process_tree_rss_bytes(
        100, read_text=read_text, list_pids=lambda _root: list(statuses)
    )
    # 10 kB of the root and 2 MiB of its child; 103 reports no known unit.
    assert total == 10 * 1024 + 2 * 1024**2
    assert (
        read_process_tree_rss_bytes(
            101, read_text=read_text, list_pids=lambda _root: list(statuses)
        )
        is None
    )


def test_process_tree_rss_survives_cycles_and_unreadable_roots() -> None:
    # Each process names the other as its parent, as a recycled PID can.
    statuses = {
        100: "PPid:\t101\nVmRSS:\t1 kB\n",
        101: "PPid:\t100\nVmRSS:\t2 kB\n",
    }

    def read_text(path: Path) -> str:
        pid = int(path.parent.name)
        if pid not in statuses:
            raise OSError("process exited")
        return statuses[pid]

    def listing(_root: Path) -> list[int]:
        return [100, 101]

    assert (
        read_process_tree_rss_bytes(100, read_text=read_text, list_pids=listing)
        == 3 * 1024
    )
    # A root that exited has no record, so nothing is measured.
    assert (
        read_process_tree_rss_bytes(555, read_text=read_text, list_pids=listing) is None
    )
    for invalid in (0, -4, True, "100"):
        assert (
            read_process_tree_rss_bytes(invalid, read_text=read_text, list_pids=listing)  # type: ignore[arg-type]
            is None
        )


def test_process_tree_sampler_reports_unavailable_measurements() -> None:
    # The sampling interval outlasts the test, so exactly the first and the
    # closing samples are taken.
    with sample_process_tree_rss(
        interval_seconds=3600.0,
        root_pid=123,
        reader=lambda _pid: None,
    ) as measurement:
        observed = measurement()

    assert observed.baseline_bytes is None
    assert observed.peak_bytes is None
    assert observed.incremental_peak_bytes is None
    assert observed.sample_count == observed.sampling_error_count == 1
    assert observed.sample_interval_seconds == 3600.0
    assert observed.unavailable_reason == "process-tree RSS is unavailable"
    final = measurement()
    assert final.sample_count == final.sampling_error_count == 2


def test_process_tree_sampler_preserves_baseline_and_sampled_peak() -> None:
    readings = iter((100, 140))

    with sample_process_tree_rss(
        interval_seconds=3600.0,
        root_pid=123,
        reader=lambda _pid: next(readings),
    ) as measurement:
        first = measurement()
        assert first.baseline_bytes == 100
        assert first.peak_bytes == 100

    final = measurement()
    assert final.baseline_bytes == 100
    assert final.peak_bytes == 140
    assert final.incremental_peak_bytes == 40
    assert (final.sample_count, final.sampling_error_count) == (2, 0)
    assert final.unavailable_reason is None


@pytest.mark.parametrize("bad_reading", [RuntimeError("proc unavailable"), -1, True])
def test_process_tree_sampler_counts_failed_and_invalid_readings(bad_reading) -> None:
    readings = iter((bad_reading, 300))

    def reader(_pid: int) -> int:
        reading = next(readings)
        if isinstance(reading, Exception):
            raise reading
        return reading

    with sample_process_tree_rss(
        interval_seconds=3600.0, root_pid=123, reader=reader
    ) as measurement:
        pass

    final = measurement()
    # The failed first reading leaves the baseline to the next valid one.
    assert (final.sample_count, final.sampling_error_count) == (2, 1)
    assert (final.baseline_bytes, final.peak_bytes) == (300, 300)
    assert final.incremental_peak_bytes == 0


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_process_tree_sampler_requires_a_positive_interval(interval) -> None:
    with pytest.raises(ValueError, match="interval_seconds must be positive"):
        with sample_process_tree_rss(interval_seconds=interval):
            pass


def _standard_descriptors() -> dict[int, tuple[int, int]]:
    """Return the device and inode behind standard output and error."""
    return {fd: (os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)}


def _null_descriptors() -> dict[int, tuple[int, int]]:
    """Return what ``_standard_descriptors`` reports while both are discarded."""
    status = os.stat(os.devnull)
    return {fd: (status.st_dev, status.st_ino) for fd in (1, 2)}


def test_suppress_native_output_discards_descriptor_writes_and_restores(capfd):
    import ctypes

    libc = ctypes.CDLL(None)
    before = _standard_descriptors()
    # capfd replaces sys.stdout, so the original streams exercise the flushes
    # that keep buffered text on the correct side of the redirection.
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    assert stdout is not None and stderr is not None
    stdout.write("python-before\n")

    with suppress_native_output():
        assert _standard_descriptors() == _null_descriptors()
        stdout.write("python-inside\n")
        stderr.write("python-error-inside\n")
        libc.printf(b"native-inside\n")
        os.write(1, b"descriptor-inside\n")
        os.write(2, b"descriptor-error-inside\n")

    assert _standard_descriptors() == before
    with open(os.devnull, "w") as later:
        assert later.fileno() > 2
    os.write(1, b"descriptor-after\n")
    captured = capfd.readouterr()
    assert captured.out == "python-before\ndescriptor-after\n"
    assert captured.err == ""


def test_suppress_native_output_restores_descriptors_after_errors(capfd):
    before = _standard_descriptors()

    with pytest.raises(RuntimeError, match="native failure"):
        with suppress_native_output():
            os.write(2, b"hidden\n")
            raise RuntimeError("native failure")

    assert _standard_descriptors() == before
    os.write(2, b"visible\n")
    assert capfd.readouterr().err == "visible\n"


@pytest.mark.parametrize("missing_libc", [False, True])
def test_suppress_native_output_handles_closed_streams(
    capfd, monkeypatch, missing_libc
):
    import ctypes

    stream = io.TextIOWrapper(io.BytesIO())
    stream.close()
    before = _standard_descriptors()

    def unavailable_libc(*_args):
        raise OSError("C standard library unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", stream)
        patch.setattr(sys, "stderr", stream)
        patch.setattr(sys, "__stdout__", None)
        patch.setattr(sys, "__stderr__", None)
        if missing_libc:
            patch.setattr(ctypes, "CDLL", unavailable_libc)
        with suppress_native_output():
            assert _standard_descriptors() == _null_descriptors()
            os.write(1, b"hidden\n")
            os.write(2, b"hidden error\n")

    assert _standard_descriptors() == before
    os.write(1, b"visible\n")
    os.write(2, b"visible error\n")
    captured = capfd.readouterr()
    assert captured.out == "visible\n"
    assert captured.err == "visible error\n"
