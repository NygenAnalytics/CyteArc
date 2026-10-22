import sys
import threading
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path


type _ReadText = Callable[[Path], str]
type _ListPids = Callable[[Path], Iterable[int]]

# Linux reports resident memory in /proc. macOS and Windows have no /proc, so
# their RSS readings are unavailable rather than estimated another way.
_PROC_ROOT = Path("/proc")
_PROC_UNAVAILABLE_REASON = "process-tree RSS requires the Linux /proc filesystem"


def _default_read_text(path: Path) -> str:
    # A process name cut inside a multibyte character is not valid UTF-8; the
    # parsed keys and amounts are ASCII.
    return path.read_text(encoding="utf-8", errors="replace")


def _default_list_pids(proc_root: Path) -> list[int]:
    return [
        int(entry.name)
        for entry in proc_root.iterdir()
        if entry.name.isdigit() and entry.is_dir()
    ]


def _nonnegative_int(value: str) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_read(path: Path, reader: _ReadText) -> str | None:
    try:
        return reader(path)
    except (OSError, RuntimeError):
        return None


def _parse_proc_status(value: str) -> tuple[int | None, int | None]:
    parent_pid: int | None = None
    rss_bytes: int | None = None
    for line in value.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.split()
        if not parts:
            continue
        if key == "PPid":
            parent_pid = _nonnegative_int(parts[0])
        elif key == "VmRSS":
            amount = _nonnegative_int(parts[0])
            if amount is None:
                continue
            unit = parts[1].lower() if len(parts) > 1 else "b"
            scale = {"b": 1, "kb": 1024, "mb": 1024**2}.get(unit)
            if scale is not None:
                rss_bytes = amount * scale
    return parent_pid, rss_bytes


def _status_rss_bytes(status_path: Path) -> int | None:
    status = _optional_read(status_path, _default_read_text)
    return None if status is None else _parse_proc_status(status)[1]


def _task_children(proc_root: Path, pid: int) -> list[int] | None:
    """Return the child processes of every thread of ``pid``.

    Returns None when ``/proc`` cannot list them: the process has exited, or the
    kernel has no ``task/<tid>/children`` files (``CONFIG_PROC_CHILDREN``).
    """
    tasks = proc_root / str(pid) / "task"
    try:
        threads = sorted(tasks.iterdir())
    except OSError:
        return None
    children: list[int] = []
    for thread in threads:
        try:
            listed = (thread / "children").read_text(encoding="ascii")
            pids = [int(value) for value in listed.split()]
        except OSError:
            if thread.is_dir() and not (thread / "children").exists():
                return None
            # The thread exited after the listing.
            continue
        except ValueError:
            # A list that holds anything but process IDs cannot be walked.
            return None
        children.extend(pids)
    return children


def _tree_pids_by_children(proc_root: Path, root_pid: int) -> set[int] | None:
    """Return ``root_pid`` and its descendants from the children files.

    Only processes in the tree are read, so the cost follows the tree rather
    than every process on the host. Returns None when the root's children
    cannot be listed.
    """
    root_children = _task_children(proc_root, root_pid)
    if root_children is None:
        return None
    tree = {root_pid}
    pending = root_children
    while pending:
        pid = pending.pop()
        if pid in tree:
            continue
        tree.add(pid)
        # A descendant that exited meanwhile has no children left to list.
        pending.extend(_task_children(proc_root, pid) or ())
    return tree


def _tree_pids_by_scan(
    proc_root: Path,
    root_pid: int,
    statuses: dict[int, tuple[int | None, int | None]],
    pid_reader: _ListPids,
    text_reader: _ReadText,
) -> set[int] | None:
    """Return ``root_pid`` and its descendants from every process's parent.

    Each status that is read is kept in ``statuses``. Returns None when the
    processes cannot be listed.
    """
    try:
        pids = {int(pid) for pid in pid_reader(proc_root)}
    except Exception:
        return None
    pids.add(root_pid)
    children: dict[int, set[int]] = {}
    for pid in pids:
        status = _optional_read(proc_root / str(pid) / "status", text_reader)
        if status is None:
            continue
        parent_pid, rss_bytes = _parse_proc_status(status)
        statuses[pid] = (parent_pid, rss_bytes)
        if parent_pid is not None:
            children.setdefault(parent_pid, set()).add(pid)

    tree_pids: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in tree_pids:
            continue
        tree_pids.add(pid)
        pending.extend(children.get(pid, ()))
    return tree_pids


def read_process_tree_rss_bytes(
    root_pid: int,
    *,
    proc_root: str | Path = "/proc",
    read_text: _ReadText | None = None,
    list_pids: _ListPids | None = None,
) -> int | None:
    """Return sampled RSS for one process and all discoverable descendants.

    The tree is walked through each process's ``task/<tid>/children`` files.
    Where those are unavailable, or when ``list_pids`` is given, every listed
    process's parent is read instead; both find the same processes.
    """

    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        return None
    resolved_root = Path(proc_root)
    text_reader = _default_read_text if read_text is None else read_text
    statuses: dict[int, tuple[int | None, int | None]] = {}
    tree_pids = (
        None
        if list_pids is not None
        else _tree_pids_by_children(resolved_root, root_pid)
    )
    if tree_pids is None:
        tree_pids = _tree_pids_by_scan(
            resolved_root,
            root_pid,
            statuses,
            _default_list_pids if list_pids is None else list_pids,
            text_reader,
        )
        if tree_pids is None:
            return None
    else:
        for pid in tree_pids:
            status = _optional_read(resolved_root / str(pid) / "status", text_reader)
            if status is not None:
                statuses[pid] = _parse_proc_status(status)

    values: list[int] = []
    for pid in tree_pids:
        if pid not in statuses:
            continue
        value = statuses[pid][1]
        if value is not None:
            values.append(value)
    return None if not values else sum(values)


@dataclass(frozen=True, slots=True)
class ProcessTreeRssMeasurement:
    baseline_bytes: int | None
    peak_bytes: int | None
    incremental_peak_bytes: int | None
    sample_interval_seconds: float
    sample_count: int
    sampling_error_count: int
    unavailable_reason: str | None


@contextmanager
def sample_process_tree_rss(
    *,
    interval_seconds: float = 0.1,
    root_pid: int | None = None,
    reader: Callable[[int], int | None] | None = None,
) -> Iterator[Callable[[], ProcessTreeRssMeasurement]]:
    """Sample process-tree RSS; reported peaks are lower-bound observations."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    sample_reader = reader
    if sample_reader is None:
        proc_root = _PROC_ROOT
        if _status_rss_bytes(proc_root / "self" / "status") is None:
            unavailable = ProcessTreeRssMeasurement(
                baseline_bytes=None,
                peak_bytes=None,
                incremental_peak_bytes=None,
                sample_interval_seconds=float(interval_seconds),
                sample_count=0,
                sampling_error_count=0,
                unavailable_reason=_PROC_UNAVAILABLE_REASON,
            )
            yield lambda: unavailable
            return
        sample_reader = partial(read_process_tree_rss_bytes, proc_root=proc_root)
    pid = os.getpid() if root_pid is None else root_pid
    values: list[int] = []
    sample_count = 0
    error_count = 0
    lock = threading.Lock()
    stop = threading.Event()

    def sample() -> None:
        nonlocal sample_count, error_count
        try:
            value = sample_reader(pid)
        except Exception:
            value = None
        with lock:
            sample_count += 1
            if value is None:
                error_count += 1
            elif isinstance(value, bool) or value < 0:
                error_count += 1
            else:
                values.append(int(value))

    sample()

    def sample_loop() -> None:
        while not stop.wait(interval_seconds):
            sample()

    thread = threading.Thread(
        target=sample_loop,
        name="cytearc-pipeline-rss",
        daemon=True,
    )
    thread.start()

    def measurement() -> ProcessTreeRssMeasurement:
        with lock:
            baseline = values[0] if values else None
            peak = max(values) if values else None
            count = sample_count
            errors = error_count
        return ProcessTreeRssMeasurement(
            baseline_bytes=baseline,
            peak_bytes=peak,
            incremental_peak_bytes=(
                None if baseline is None or peak is None else max(0, peak - baseline)
            ),
            sample_interval_seconds=float(interval_seconds),
            sample_count=count,
            sampling_error_count=errors,
            unavailable_reason=(
                None if peak is not None else "process-tree RSS is unavailable"
            ),
        )

    try:
        yield measurement
    finally:
        stop.set()
        thread.join(timeout=max(2.0, interval_seconds * 4))
        sample()


def _flush_output_streams() -> None:
    """Flush Python and C stdio buffers before file descriptors change."""
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        flush = getattr(stream, "flush", None)
        if flush is None:
            continue
        try:
            flush()
        except (OSError, ValueError):
            pass
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError, TypeError):
        pass


@contextmanager
def suppress_native_output() -> Iterator[None]:
    """Send process-level stdout and stderr to the null device.

    Native extensions write to file descriptors 1 and 2 directly, so
    redirecting ``sys.stdout`` cannot silence them. Both descriptors are
    duplicated on entry and restored on exit; every other open file keeps its
    descriptor. Output written by other threads while the context is active is
    also discarded.
    """
    _flush_output_streams()
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved: list[tuple[int, int]] = []
    try:
        for fd in (1, 2):
            saved.append((fd, os.dup(fd)))
        for fd, _copy in saved:
            os.dup2(null_fd, fd)
        yield
    finally:
        _flush_output_streams()
        for fd, copy in saved:
            os.dup2(copy, fd)
            os.close(copy)
        os.close(null_fd)


def process_rss_mb() -> float | None:
    """Return this process's resident memory in MiB, or None where unavailable."""
    rss_bytes = _status_rss_bytes(_PROC_ROOT / "self" / "status")
    return None if rss_bytes is None else rss_bytes / 1024**2


def rss_text() -> str:
    """Return this process's resident memory for a log line."""
    rss_mb = process_rss_mb()
    return "n/a" if rss_mb is None else f"{rss_mb:.0f} MiB"
