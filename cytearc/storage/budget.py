"""Resource limits used by one storage or analysis operation."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

_FALLBACK_MEMORY_BYTES = 8 * 1024 * 1024 * 1024
_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_MIN_RAW_BYTES = 1024 * 1024
_CGROUP_UNLIMITED_THRESHOLD = 1 << 60


@dataclass(frozen=True)
class ResourceBudget:
    """Memory and CPU limits for one operation.

    Attributes:
        memoryBytes: Planning budget in bytes. Not a hard process RSS cap.
        workers: Maximum compute-worker budget.
    """

    memoryBytes: int
    workers: int


def _read_int(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed <= 0 or parsed >= _CGROUP_UNLIMITED_THRESHOLD:
        return None
    return parsed


def _physical_memory_bytes() -> int:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and total_pages > 0:
            return int(page_size) * int(total_pages)
    except (ValueError, OSError, AttributeError):
        pass
    return _FALLBACK_MEMORY_BYTES


def _process_cgroup_entry(controller: str) -> tuple[list[str], str] | None:
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers = fields[1].split(",") if fields[1] else []
        if (not controllers and controller == "") or controller in controllers:
            return controllers, fields[2].lstrip("/")
    return None


def _process_cgroup_path(controller: str) -> str | None:
    entry = _process_cgroup_entry(controller)
    return None if entry is None else entry[1]


def _unified_cgroup_files(name: str) -> list[str]:
    path = _process_cgroup_path("")
    parts = Path(path).parts if path else ()
    candidates = [
        str(Path("/sys/fs/cgroup", *parts[:depth], name))
        for depth in range(len(parts), -1, -1)
    ]
    return list(dict.fromkeys(candidates))


def _legacy_cgroup_files(controller: str, name: str) -> list[str]:
    entry = _process_cgroup_entry(controller)
    controllers, path = ([], "") if entry is None else entry
    parts = Path(path).parts if path else ()
    mount_names = list(dict.fromkeys([",".join(controllers), controller]))
    candidates = [
        str(Path("/sys/fs/cgroup", mount_name, *parts[:depth], name))
        for mount_name in mount_names
        if mount_name
        for depth in range(len(parts), -1, -1)
    ]
    return list(dict.fromkeys(candidates))


def _cgroup_memory_bytes() -> int | None:
    paths = _unified_cgroup_files("memory.max")
    paths.extend(_legacy_cgroup_files("memory", "memory.limit_in_bytes"))
    limits = [value for path in paths if (value := _read_int(path)) is not None]
    return min(limits) if limits else None


def detect_total_memory_bytes() -> int:
    """Return the effective physical or cgroup memory limit."""
    physical = _physical_memory_bytes()
    cgroup = _cgroup_memory_bytes()
    return physical if cgroup is None else min(physical, cgroup)


def _cpu_quota_workers() -> int | None:
    limits: list[int] = []
    for path in _unified_cgroup_files("cpu.max"):
        try:
            quota_text, period_text = Path(path).read_text().strip().split()
        except (OSError, ValueError):
            continue
        if quota_text != "max":
            try:
                quota = int(quota_text)
                period = int(period_text)
            except ValueError:
                continue
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))

    for legacy_quota_path in _legacy_cgroup_files("cpu", "cpu.cfs_quota_us"):
        legacy_period_path = str(Path(legacy_quota_path).with_name("cpu.cfs_period_us"))
        legacy_quota = _read_int(legacy_quota_path)
        legacy_period = _read_int(legacy_period_path)
        if legacy_quota is not None and legacy_period is not None:
            limits.append(max(1, legacy_quota // legacy_period))
    return min(limits) if limits else None


def detect_workers() -> int:
    limits = [max(1, os.cpu_count() or 1)]
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            limits.append(max(1, len(sched_getaffinity(0))))
        except OSError:
            pass
    quota = _cpu_quota_workers()
    if quota is not None:
        limits.append(quota)
    return min(limits)


def _finite_number(text: str, spec: str | float, guidance: str) -> float:
    """Return ``text`` as a positive finite number, naming ``spec`` otherwise."""
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid memory spec: {spec!r}. {guidance}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Memory budget must be finite, got {spec!r}.")
    if number <= 0:
        raise ValueError(f"Memory budget must be positive, got {spec!r}.")
    return number


def _parse_memory_spec(spec: str | int | float) -> int:
    guidance = (
        "Use raw bytes, a suffixed size like '8G', '8GB', '8GiB', or '512M', "
        "or a fraction strictly between 0 and 1 like '0.6'."
    )
    if isinstance(spec, bool):
        raise ValueError(f"Invalid memory spec: {spec!r}. {guidance}")
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError(f"Memory budget must be positive, got {spec!r}.")
        return spec

    text = str(spec).strip()
    if not text:
        raise ValueError("Empty memory spec. " + guidance)
    # K, M, G, and T are powers of 1024, alone or followed by B or iB.
    unit = text.upper().removesuffix("B").removesuffix("I")
    if unit[-1:] in _SUFFIXES:
        size = _finite_number(unit[:-1], spec, guidance) * _SUFFIXES[unit[-1]]
        if not math.isfinite(size):
            raise ValueError(f"Memory budget must be finite, got {spec!r}.")
        return max(1, int(size))

    number = _finite_number(text, spec, guidance)
    if number < 1:
        return max(1, int(number * detect_total_memory_bytes()))
    if number < _MIN_RAW_BYTES:
        raise ValueError(
            f"Ambiguous memory spec {spec!r}: a bare number >= 1 is read as bytes, "
            f"which is implausibly small. {guidance}"
        )
    return int(number)


def resolve_budget(
    memory: int | str | None = None,
    workers: int | None = None,
) -> ResourceBudget:
    """Resolve limits, defaulting to 75% of detected memory and detected workers.

    The default memory budget is 75% of the detected total memory, the
    physical memory or a lower container limit, not of the memory that is
    free when the budget is resolved. ``memory`` takes bytes, a size whose
    suffix K, M, G, or T is a power of 1024, alone or followed by ``B`` or
    ``iB`` (``'8G'``, ``'8GB'``, and ``'8GiB'`` are all ``8 * 1024**3``), or a
    fraction of the total strictly between 0 and 1.

    Raises:
        ValueError: If ``memory`` or ``CYTEARC_MEM_BUDGET`` is not a positive
            finite size, or ``workers`` or ``CYTEARC_WORKERS`` is below 1.
    """
    if memory is None:
        env_mem = os.environ.get("CYTEARC_MEM_BUDGET")
        if env_mem:
            memory_bytes = _parse_memory_spec(env_mem)
        else:
            memory_bytes = detect_total_memory_bytes() * 3 // 4
    else:
        memory_bytes = _parse_memory_spec(memory)

    if workers is None:
        env_workers = os.environ.get("CYTEARC_WORKERS")
        if env_workers:
            try:
                worker_count = int(env_workers)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid CYTEARC_WORKERS={env_workers!r}; expected an integer."
                ) from exc
            if worker_count < 1:
                raise ValueError(
                    f"Invalid CYTEARC_WORKERS={env_workers!r}; expected at least 1."
                )
        else:
            worker_count = detect_workers()
    else:
        worker_count = int(workers)
        if worker_count < 1:
            raise ValueError(f"The worker count must be at least 1, got {workers!r}.")

    return ResourceBudget(memoryBytes=max(1, memory_bytes), workers=worker_count)


DEFAULT_READ_AHEAD_BLOCKS = 2
"""Read depth a public streaming API uses when the caller has not planned one."""
