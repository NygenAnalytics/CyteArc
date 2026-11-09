import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cache, partial
from pathlib import Path
from typing import NamedTuple


type _ReadText = Callable[[Path], str]
type _ListPids = Callable[[Path], Iterable[int]]

# Linux reports resident memory, threads and network counters in /proc. macOS
# and Windows have no /proc, so those readings are unavailable rather than
# estimated another way.
_PROC_ROOT = Path("/proc")
_PROC_UNAVAILABLE_REASON = "process-tree RSS requires the Linux /proc filesystem"
_THREADS_UNAVAILABLE_REASON = "OS thread counts require the Linux /proc filesystem"
_NETWORK_UNAVAILABLE_REASON = "network counters require the Linux /proc filesystem"
_HIGH_WATER_UNAVAILABLE_REASON = (
    "the peak RSS requires getrusage, which this platform does not provide"
)

# The measurements that ProcessResourceMeasurement.unavailable names.
THREADS = "threads"
NETWORK = "network"
HIGH_WATER = "high_water"

# A trace keeps at most this many samples; a full trace merges pairwise.
TRACE_POINT_LIMIT = 64


def _read_bytes(path: Path) -> bytes:
    """Return a file's bytes through one descriptor, without pathlib's layers."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65536):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _default_read_text(path: Path) -> str:
    # A process name cut inside a multibyte character is not valid UTF-8; the
    # parsed keys and amounts are ASCII.
    return _read_bytes(path).decode("utf-8", errors="replace")


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


class _ProcStatus(NamedTuple):
    parent_pid: int | None
    rss_bytes: int | None
    threads: int | None


def _parse_proc_status(value: str) -> _ProcStatus:
    parent_pid: int | None = None
    rss_bytes: int | None = None
    threads: int | None = None
    for line in value.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.split()
        if not parts:
            continue
        if key == "PPid":
            parent_pid = _nonnegative_int(parts[0])
        elif key == "Threads":
            threads = _nonnegative_int(parts[0])
        elif key == "VmRSS":
            amount = _nonnegative_int(parts[0])
            if amount is None:
                continue
            unit = parts[1].lower() if len(parts) > 1 else "b"
            scale = {"b": 1, "kb": 1024, "mb": 1024**2}.get(unit)
            if scale is not None:
                rss_bytes = amount * scale
    return _ProcStatus(parent_pid, rss_bytes, threads)


def _status_rss_bytes(status_path: Path) -> int | None:
    status = _optional_read(status_path, _default_read_text)
    return None if status is None else _parse_proc_status(status).rss_bytes


def _parse_proc_stat_cpu_ticks(value: str) -> int | None:
    """Return the CPU ticks of a process and of the children it waited for.

    These are fields 14 to 17 of ``proc_pid_stat(5)``: ``utime``, ``stime``,
    ``cutime`` and ``cstime``. The command name before them, in parentheses,
    may itself hold spaces and parentheses.
    """
    _name, separator, rest = value.rpartition(")")
    fields = rest.split()
    # fields[0] is field 3, the process state.
    if not separator or len(fields) < 15:
        return None
    ticks = [_nonnegative_int(fields[index]) for index in (11, 12, 13, 14)]
    if any(tick is None for tick in ticks):
        return None
    return sum(tick for tick in ticks if tick is not None)


@cache
def _clock_ticks_per_second() -> int:
    return int(os.sysconf("SC_CLK_TCK"))


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
            listed = _read_bytes(thread / "children").decode("ascii")
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
    statuses: dict[int, _ProcStatus],
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
        parsed = _parse_proc_status(status)
        statuses[pid] = parsed
        if parsed.parent_pid is not None:
            children.setdefault(parsed.parent_pid, set()).add(pid)

    tree_pids: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in tree_pids:
            continue
        tree_pids.add(pid)
        pending.extend(children.get(pid, ()))
    return tree_pids


class _TreeReading(NamedTuple):
    rss_bytes: int | None
    threads: int | None
    # The CPU seconds of every descendant, each with the children it waited for.
    descendant_cpu_seconds: float


def _read_process_tree(
    root_pid: int,
    *,
    proc_root: Path,
    read_text: _ReadText | None = None,
    list_pids: _ListPids | None = None,
    descendant_cpu: bool = False,
) -> _TreeReading | None:
    """Read RSS and threads of one process and its descendants, None if none.

    The tree is walked through each process's ``task/<tid>/children`` files.
    Where those are unavailable, or when ``list_pids`` is given, every listed
    process's parent is read instead; both find the same processes. With
    ``descendant_cpu``, each descendant's ``stat`` file adds its CPU time.
    """
    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        return None
    text_reader = _default_read_text if read_text is None else read_text
    statuses: dict[int, _ProcStatus] = {}
    tree_pids = (
        None if list_pids is not None else _tree_pids_by_children(proc_root, root_pid)
    )
    if tree_pids is None:
        tree_pids = _tree_pids_by_scan(
            proc_root,
            root_pid,
            statuses,
            _default_list_pids if list_pids is None else list_pids,
            text_reader,
        )
        if tree_pids is None:
            return None
    else:
        for pid in tree_pids:
            status = _optional_read(proc_root / str(pid) / "status", text_reader)
            if status is not None:
                statuses[pid] = _parse_proc_status(status)
    found = [statuses[pid] for pid in tree_pids if pid in statuses]
    rss = [status.rss_bytes for status in found if status.rss_bytes is not None]
    threads = [status.threads for status in found if status.threads is not None]
    ticks = 0
    if descendant_cpu:
        for pid in tree_pids - {root_pid}:
            stat = _optional_read(proc_root / str(pid) / "stat", text_reader)
            # A descendant that ended meanwhile counts once its parent waits.
            ticks += (None if stat is None else _parse_proc_stat_cpu_ticks(stat)) or 0
    return _TreeReading(
        rss_bytes=sum(rss) if rss else None,
        threads=sum(threads) if threads else None,
        descendant_cpu_seconds=ticks / _clock_ticks_per_second() if ticks else 0.0,
    )


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
    reading = _read_process_tree(
        root_pid, proc_root=Path(proc_root), read_text=read_text, list_pids=list_pids
    )
    return None if reading is None else reading.rss_bytes


def _parse_net_dev(value: str) -> tuple[int, int] | None:
    """Return the bytes received and sent on every interface but loopback.

    ``/proc/<pid>/net/dev`` counts each interface of the process's network
    namespace: two header lines, then one line per interface whose first and
    ninth numbers are the bytes received and sent.
    """
    lines = value.splitlines()
    if len(lines) < 2 or "Receive" not in lines[0] or "Transmit" not in lines[0]:
        return None
    received = sent = 0
    for line in lines[2:]:
        name, separator, data = line.partition(":")
        if not separator:
            continue
        if name.strip() == "lo":
            continue
        values = data.split()
        if len(values) < 9:
            return None
        interface_received = _nonnegative_int(values[0])
        interface_sent = _nonnegative_int(values[8])
        if interface_received is None or interface_sent is None:
            return None
        received += interface_received
        sent += interface_sent
    return received, sent


def _ended_children_cpu_seconds() -> float:
    """Return the CPU seconds of child processes that ended and were waited for."""
    try:
        import resource
    except ImportError:
        # Windows keeps no such count.
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _process_cpu_seconds() -> float:
    """Return the CPU seconds of this process and of its children that ended."""
    return time.process_time() + _ended_children_cpu_seconds()


def _has_child_processes() -> bool:
    """Return whether this process has a child, waited for or not.

    ``waitid`` sees the children that any thread of the process started, and
    ``WNOWAIT`` leaves an ended child for its own waiter.
    """
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return False
    except (AttributeError, OSError):
        # Without a definite answer the tree is walked.
        return True
    return True


def _high_water_reader() -> Callable[[], int] | None:
    """Return a reader of the kernel's mark of this process's peak RSS, or None.

    ``getrusage`` reports ``ru_maxrss`` in KiB on Linux and in bytes on macOS;
    Windows has no ``getrusage``. On Linux, ``exec`` folds the previous
    image's mark into the new program's, so a process started with ``vfork``,
    as Python's ``subprocess`` starts one, begins with its parent's peak.
    """
    try:
        import resource
    except ImportError:
        return None
    scale = 1 if sys.platform == "darwin" else 1024

    def read() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale

    return read


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """What one sample of a process reads.

    Attributes:
        cpu_seconds: CPU time, user and system, of the process tree since the
            process started: the process's own, that of its children that
            ended, and that of its running descendants.
        rss_bytes: The process tree's resident memory, or None where it was
            not read.
        threads: The process tree's OS threads, or None.
        network_rx_bytes: Bytes received so far on every interface but
            loopback by every process of the network namespace, or None.
        network_tx_bytes: Bytes sent so far, likewise, or None.
    """

    cpu_seconds: float
    rss_bytes: int | None = None
    threads: int | None = None
    network_rx_bytes: int | None = None
    network_tx_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ProcessReaders:
    """How a sampler reads its process.

    Attributes:
        sample: Takes one sample.
        high_water_bytes: Returns the process's peak RSS since it started, or
            None where the platform keeps no such mark.
        rss_unavailable_reason: Why the platform samples no RSS, or None.
        unavailable: Why the platform cannot take a measurement, by
            ``THREADS``, ``NETWORK`` and ``HIGH_WATER``.
    """

    sample: Callable[[], ResourceSample]
    high_water_bytes: Callable[[], int] | None = None
    rss_unavailable_reason: str | None = None
    unavailable: Mapping[str, str] = field(default_factory=dict)


def _proc_sample(proc_root: Path, pid: int) -> ResourceSample:
    """Read one sample of this process from ``/proc``.

    The process tree is walked only when the process has children; otherwise
    its own ``status`` file holds the tree's RSS and threads.
    """
    cpu = _process_cpu_seconds()
    rss: int | None
    threads: int | None
    if _has_child_processes():
        tree = _read_process_tree(pid, proc_root=proc_root, descendant_cpu=True)
        rss = None if tree is None else tree.rss_bytes
        threads = None if tree is None else tree.threads
        cpu += 0.0 if tree is None else tree.descendant_cpu_seconds
    else:
        status = _optional_read(proc_root / str(pid) / "status", _default_read_text)
        parsed = None if status is None else _parse_proc_status(status)
        rss = None if parsed is None else parsed.rss_bytes
        threads = None if parsed is None else parsed.threads
    network = _optional_read(proc_root / str(pid) / "net" / "dev", _default_read_text)
    counters = None if network is None else _parse_net_dev(network)
    received, sent = (None, None) if counters is None else counters
    return ResourceSample(cpu, rss, threads, received, sent)


def _cpu_sample() -> ResourceSample:
    return ResourceSample(_process_cpu_seconds())


def default_process_readers() -> ProcessReaders:
    """Return the readers that this platform offers for the calling process.

    Linux's ``/proc`` gives RSS, threads and network counters; CPU time comes
    from the process clock on every platform, and the peak RSS from
    ``getrusage`` where the platform has it.
    """
    proc_root = _PROC_ROOT
    pid = os.getpid()
    high_water = _high_water_reader()
    unavailable: dict[str, str] = {}
    if high_water is None:
        unavailable[HIGH_WATER] = _HIGH_WATER_UNAVAILABLE_REASON
    status = _optional_read(proc_root / str(pid) / "status", _default_read_text)
    parsed = None if status is None else _parse_proc_status(status)
    if parsed is None or parsed.rss_bytes is None:
        unavailable[THREADS] = _THREADS_UNAVAILABLE_REASON
        unavailable[NETWORK] = _NETWORK_UNAVAILABLE_REASON
        return ProcessReaders(
            sample=_cpu_sample,
            high_water_bytes=high_water,
            rss_unavailable_reason=_PROC_UNAVAILABLE_REASON,
            unavailable=unavailable,
        )
    if parsed.threads is None:
        unavailable[THREADS] = "the /proc status file reports no thread count"
    network_path = proc_root / str(pid) / "net" / "dev"
    network = _optional_read(network_path, _default_read_text)
    if network is None or _parse_net_dev(network) is None:
        unavailable[NETWORK] = f"{network_path} cannot be read as network counters"
    return ProcessReaders(
        sample=partial(_proc_sample, proc_root, pid),
        high_water_bytes=high_water,
        unavailable=unavailable,
    )


@dataclass(frozen=True, slots=True)
class ResourceTrace:
    """The samples of one measured window, at most ``TRACE_POINT_LIMIT``.

    Each point is a sample: its ``seconds`` since the window opened, the CPU
    seconds and network bytes counted since then, and the largest RSS and
    thread count sampled since the previous point. A series that the platform
    cannot sample is None, and a point whose reading failed holds None.
    """

    seconds: tuple[float, ...]
    cpu_seconds: tuple[float, ...]
    rss_bytes: tuple[int | None, ...] | None
    threads: tuple[int | None, ...] | None
    network_rx_bytes: tuple[int, ...] | None
    network_tx_bytes: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ProcessResourceMeasurement:
    """What a process used during one measured window.

    The RSS fields describe sampled process-tree RSS, as before: reported
    peaks are lower bounds. CPU time and network bytes come from the readings
    that open and close the window, so a gap in sampling leaves them exact.

    Attributes:
        baseline_bytes: The first RSS that a sample read, or None.
        peak_bytes: The largest sampled RSS, or None.
        incremental_peak_bytes: That peak above the baseline, or None.
        sample_interval_seconds: The interval between samples.
        sample_count: The samples that read RSS, 0 where it is not sampled.
        sampling_error_count: Those whose RSS reading failed.
        unavailable_reason: Why RSS is unavailable, or None.
        cpu_seconds: CPU time of the process tree during the window.
        threads_peak: The most OS threads that a sample counted, or None.
        high_water_start_bytes: The kernel's mark of the process's peak RSS
            as the window opened, or None. On Linux a process starts with the
            mark of the process that started it.
        high_water_end_bytes: The same mark as the window closed, or None;
            when it exceeds the start, the window set a new peak.
        network_rx_bytes: Bytes that the network namespace received during
            the window, or None.
        network_tx_bytes: Bytes that it sent, or None.
        trace: The window's samples.
        unavailable: Why a measurement is None, by ``THREADS``, ``NETWORK``
            and ``HIGH_WATER``.
    """

    baseline_bytes: int | None
    peak_bytes: int | None
    incremental_peak_bytes: int | None
    sample_interval_seconds: float
    sample_count: int
    sampling_error_count: int
    unavailable_reason: str | None
    cpu_seconds: float
    threads_peak: int | None
    high_water_start_bytes: int | None
    high_water_end_bytes: int | None
    network_rx_bytes: int | None
    network_tx_bytes: int | None
    trace: ResourceTrace
    unavailable: Mapping[str, str]


@dataclass(slots=True)
class _Point:
    seconds: float
    cpu_seconds: float
    rss_bytes: int | None
    threads: int | None
    network_rx_bytes: int | None
    network_tx_bytes: int | None


def _larger(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    return first if second is None else max(first, second)


def _merge_trace(points: list[_Point]) -> list[_Point]:
    """Halve a trace: the first point stays, and each later pair merges.

    The later point of a pair keeps its time and counted values and takes the
    larger RSS and thread counts, so every point stays a real sample.
    """
    merged = [points[0]]
    for index in range(1, len(points) - 1, 2):
        earlier, later = points[index], points[index + 1]
        later.rss_bytes = _larger(earlier.rss_bytes, later.rss_bytes)
        later.threads = _larger(earlier.threads, later.threads)
        merged.append(later)
    if len(points) % 2 == 0:
        merged.append(points[-1])
    return merged


def _valid_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class _ResourceWindow:
    """The samples of one measured window, and what they add up to.

    ``record`` takes each sample with its seconds since the window opened;
    the first sample opens the window. Methods hold a lock, so a sampling
    thread and a reader of the measurement can share a window.
    """

    def __init__(self, readers: ProcessReaders, interval_seconds: float) -> None:
        self._readers = readers
        self._interval_seconds = float(interval_seconds)
        self._rss_sampled = readers.rss_unavailable_reason is None
        self._threads_sampled = THREADS not in readers.unavailable
        self._network_sampled = NETWORK not in readers.unavailable
        self._lock = threading.Lock()
        self._points: list[_Point] = []
        self._samples = 0
        self._errors = 0
        self._rss_baseline: int | None = None
        self._rss_peak: int | None = None
        self._threads_peak: int | None = None
        self._network_problem: str | None = None
        self._high_water_start: int | None = None
        self._high_water_end: int | None = None
        self._closed = False

    def read_high_water(self) -> int | None:
        read = self._readers.high_water_bytes
        if read is None:
            return None
        try:
            return _valid_count(read())
        except Exception:
            return None

    def open(self, high_water: int | None) -> None:
        self._high_water_start = high_water

    def close(self, high_water: int | None) -> None:
        with self._lock:
            self._high_water_end = high_water
            self._closed = True

    def record(self, reading: ResourceSample | None, seconds: float) -> None:
        """Add a sample; None stands for a reading that raised.

        A sample that arrives once the window closed, from a sampling thread
        that outlived its join, is left out.
        """
        with self._lock:
            if not self._closed:
                self._record(reading, seconds)

    def _record(self, reading: ResourceSample | None, seconds: float) -> None:
        rss = None if reading is None else _valid_count(reading.rss_bytes)
        if self._rss_sampled:
            self._samples += 1
            if rss is None:
                self._errors += 1
            else:
                if self._rss_baseline is None:
                    self._rss_baseline = rss
                self._rss_peak = _larger(self._rss_peak, rss)
        if reading is None:
            return
        threads = _valid_count(reading.threads) if self._threads_sampled else None
        self._threads_peak = _larger(self._threads_peak, threads)
        received = _valid_count(reading.network_rx_bytes)
        sent = _valid_count(reading.network_tx_bytes)
        previous = self._points[-1] if self._points else None
        if self._network_sampled and self._network_problem is None:
            if received is None or sent is None:
                self._network_problem = "a sample could not read the network counters"
            elif previous is not None and (
                received < (previous.network_rx_bytes or 0)
                or sent < (previous.network_tx_bytes or 0)
            ):
                self._network_problem = (
                    "the network counters decreased during the window, as when an "
                    "interface goes away"
                )
        # CPU time only grows; a child that ends between two reads of one
        # sample counts at the next. Two threads can each time a sample
        # before either records it, so a point never precedes the one before.
        cpu = float(reading.cpu_seconds)
        moment = float(seconds)
        if previous is not None:
            cpu = max(cpu, previous.cpu_seconds)
            moment = max(moment, previous.seconds)
        self._points.append(
            _Point(
                seconds=moment,
                cpu_seconds=cpu,
                rss_bytes=rss if self._rss_sampled else None,
                threads=threads,
                network_rx_bytes=received,
                network_tx_bytes=sent,
            )
        )
        if len(self._points) > TRACE_POINT_LIMIT:
            self._points = _merge_trace(self._points)

    def measurement(self) -> ProcessResourceMeasurement:
        """Return what the window measured so far, or in all once it closed."""
        end = None if self._closed else self.read_high_water()
        with self._lock:
            if self._closed:
                end = self._high_water_end
            if not self._points:
                raise RuntimeError("A resource window holds no sample yet")
            window = [replace(point) for point in self._points]
            samples, errors = self._samples, self._errors
            baseline, peak = self._rss_baseline, self._rss_peak
            threads_peak = self._threads_peak
            network_problem = self._network_problem
            start = self._high_water_start
        unavailable = {
            name: reason
            for name, reason in self._readers.unavailable.items()
            if name in (THREADS, NETWORK, HIGH_WATER)
        }
        if self._threads_sampled and threads_peak is None:
            unavailable[THREADS] = "no sample could read the OS thread count"
        if network_problem is not None:
            unavailable[NETWORK] = network_problem
        if HIGH_WATER not in unavailable and (start is None or end is None):
            unavailable[HIGH_WATER] = "getrusage could not read the peak RSS"
        if HIGH_WATER in unavailable:
            start = end = None
        first, last = window[0], window[-1]

        def counted(values: list[int | None]) -> tuple[int, ...] | None:
            if NETWORK in unavailable:
                return None
            return tuple((value or 0) - (values[0] or 0) for value in values)

        received = counted([point.network_rx_bytes for point in window])
        sent = counted([point.network_tx_bytes for point in window])
        rss_reason = self._readers.rss_unavailable_reason
        if rss_reason is None and peak is None:
            rss_reason = "process-tree RSS is unavailable"
        return ProcessResourceMeasurement(
            baseline_bytes=baseline,
            peak_bytes=peak,
            incremental_peak_bytes=(
                None if baseline is None or peak is None else max(0, peak - baseline)
            ),
            sample_interval_seconds=self._interval_seconds,
            sample_count=samples,
            sampling_error_count=errors,
            unavailable_reason=rss_reason,
            cpu_seconds=last.cpu_seconds - first.cpu_seconds,
            threads_peak=None if THREADS in unavailable else threads_peak,
            high_water_start_bytes=start,
            high_water_end_bytes=end,
            network_rx_bytes=None if received is None else received[-1],
            network_tx_bytes=None if sent is None else sent[-1],
            trace=ResourceTrace(
                seconds=tuple(
                    round(point.seconds - first.seconds, 3) for point in window
                ),
                cpu_seconds=tuple(
                    round(point.cpu_seconds - first.cpu_seconds, 3) for point in window
                ),
                rss_bytes=(
                    tuple(point.rss_bytes for point in window)
                    if self._rss_sampled
                    else None
                ),
                threads=(
                    None
                    if THREADS in unavailable
                    else tuple(point.threads for point in window)
                ),
                network_rx_bytes=received,
                network_tx_bytes=sent,
            ),
            unavailable=unavailable,
        )


@contextmanager
def sample_process_resources(
    *,
    interval_seconds: float = 0.1,
    readers: ProcessReaders | None = None,
) -> Iterator[Callable[[], ProcessResourceMeasurement]]:
    """Measure the calling process while the block runs.

    A thread samples every ``interval_seconds`` on every platform. The first
    sample opens the window and a last one closes it as the block ends. A
    sample that the GIL delays leaves a longer interval in the trace; the
    CPU time and network bytes counted across it stay exact. The yielded
    callable returns the measurement, during the block or after it.

    Raises:
        ValueError: If ``interval_seconds`` is not positive.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    resolved = default_process_readers() if readers is None else readers
    window = _ResourceWindow(resolved, interval_seconds)
    stop = threading.Event()
    window.open(window.read_high_water())
    opened = time.perf_counter()
    # A reader that fails before the window opens has nothing to measure.
    window.record(resolved.sample(), time.perf_counter() - opened)

    def sample() -> None:
        try:
            reading: ResourceSample | None = resolved.sample()
        except Exception:
            reading = None
        window.record(reading, time.perf_counter() - opened)

    def sample_loop() -> None:
        while not stop.wait(interval_seconds):
            sample()

    thread = threading.Thread(
        target=sample_loop,
        name="cytearc-resource-sampler",
        daemon=True,
    )
    thread.start()
    try:
        yield window.measurement
    finally:
        stop.set()
        thread.join(timeout=max(2.0, interval_seconds * 4))
        sample()
        window.close(window.read_high_water())


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
