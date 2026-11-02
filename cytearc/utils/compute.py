import itertools
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np

from .progress import progress_requested

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _ThreadLimit:
    """One active limit: its thread count, whether it reaches every runtime or
    BLAS alone, and the thread it began on."""

    threads: int | None
    every_runtime: bool
    thread: int


_thread_limit_lock = threading.Lock()
_thread_limit_tokens = itertools.count()
_active_thread_limits: dict[int, _ThreadLimit] = {}
# The thread counts of the runtimes that count for the whole process, BLAS
# among them, before the first active limit, by library file.
_original_threads: dict[str, int] = {}
# Each thread's OpenMP thread counts before its first active OpenMP limit, by
# thread and library file.
_original_openmp_threads: dict[int, dict[str, int]] = {}
_thread_pool_controller: Any | None = None
_thread_pool_modules = -1


def _shared_thread_pool_controller() -> Any:
    """Return the process's threadpoolctl controller, built again after any import.

    Building a controller scans the loaded shared libraries and wraps each
    native thread pool in ctypes objects that form reference cycles, about
    14 KB for each controller, which only the cycle collector frees. The
    libraries that a limit reaches, BLAS and OpenMP runtimes, are loaded by
    importing extension modules, and threadpoolctl gives no signal of a new
    library, so a controller built since ``sys.modules`` last changed size
    still knows every one of them. Callers hold ``_thread_limit_lock``.
    """
    from threadpoolctl import ThreadpoolController

    global _thread_pool_controller, _thread_pool_modules
    if _thread_pool_controller is None or len(sys.modules) != _thread_pool_modules:
        _thread_pool_controller = ThreadpoolController()
        # Building the first controller imports modules of its own.
        _thread_pool_modules = len(sys.modules)
    return _thread_pool_controller


def _record_original_threads(libraries: list[Any], thread: int, openmp: bool) -> None:
    """Note the thread counts of libraries that no active limit has changed yet.

    A library that an import loads while limits are active is noted when a
    later limit first sees it, before any limit reaches it. With ``openmp``,
    the calling thread's OpenMP counts are noted too.
    """
    for library in libraries:
        if library.user_api == "openmp":
            if openmp:
                recorded = _original_openmp_threads.setdefault(thread, {})
                recorded.setdefault(library.filepath, library.get_num_threads())
        else:
            _original_threads.setdefault(library.filepath, library.get_num_threads())


def _apply_thread_limits(libraries: list[Any], thread: int, openmp: bool) -> None:
    """Set the libraries to the most recent remaining limit, or to their originals.

    A BLAS-only limit reaches the BLAS libraries, and other limits every
    runtime that threadpoolctl knows. The counts of BLAS and of runtimes
    other than OpenMP, such as SciPy's Matrix Market reader, apply to the
    whole process. OpenMP runtimes keep a count for each thread, which only
    that thread can read or set, so with ``openmp`` this sets the calling
    thread's count from the limits that began on it.
    """
    limits = list(reversed(_active_thread_limits.values()))

    def latest(every_runtime: bool, own_thread: bool = False) -> int | None:
        return next(
            (
                limit.threads
                for limit in limits
                if limit.threads is not None
                and (limit.every_runtime or not every_runtime)
                and (limit.thread == thread or not own_thread)
            ),
            None,
        )

    blas = latest(every_runtime=False)
    others = latest(every_runtime=True)
    own = latest(every_runtime=True, own_thread=True)
    recorded = _original_openmp_threads.get(thread, {})
    for library in libraries:
        if library.user_api == "openmp":
            if not openmp:
                continue
            target = own if own is not None else recorded.get(library.filepath)
        else:
            count = blas if library.user_api == "blas" else others
            target = (
                count if count is not None else _original_threads.get(library.filepath)
            )
        if target is not None and library.get_num_threads() != target:
            library.set_num_threads(target)


def enter_thread_limit(
    limits: int | None, *, user_api: Literal["blas"] | None = None
) -> int:
    """Start a native thread-pool limit and return its token.

    The limit reaches the BLAS libraries and, unless ``user_api`` is
    ``"blas"``, every other runtime that threadpoolctl knows, OpenMP among
    them. Most counts apply to the whole process, so limits held by
    generators or begun on other threads can end in any order: the first
    active limit records the counts in force, the most recent remaining limit
    applies while limits overlap, and the last to end restores the recorded
    counts. OpenMP runtimes keep a count for each thread, so a limit sets the
    OpenMP count of the thread it begins on, by the same rule among the
    limits begun on that thread. ``None`` changes nothing but still takes part.
    """
    if user_api not in (None, "blas"):
        raise ValueError(f"user_api must be None or 'blas', not {user_api!r}")
    every_runtime = user_api is None
    thread = threading.get_ident()
    with _thread_limit_lock:
        libraries = _shared_thread_pool_controller().lib_controllers
        _record_original_threads(libraries, thread, every_runtime)
        token = next(_thread_limit_tokens)
        _active_thread_limits[token] = _ThreadLimit(limits, every_runtime, thread)
        try:
            _apply_thread_limits(libraries, thread, every_runtime)
        except BaseException:
            # The caller gets no token to end, so the limit ends here.
            _end_thread_limit(token, thread)
            raise
    return token


def _end_thread_limit(token: int, thread: int) -> None:
    """End a limit on ``thread``; callers hold ``_thread_limit_lock``.

    The originals are forgotten only once they are restored, so a restore
    that fails is tried again when the next limits end.
    """
    ended = _active_thread_limits.pop(token)
    openmp = ended.every_runtime and ended.thread == thread
    libraries = _shared_thread_pool_controller().lib_controllers
    _record_original_threads(libraries, thread, openmp)
    _apply_thread_limits(libraries, thread, openmp)
    if not _active_thread_limits:
        _original_threads.clear()
    if openmp and not any(
        limit.every_runtime and limit.thread == thread
        for limit in _active_thread_limits.values()
    ):
        _original_openmp_threads.pop(thread, None)


def exit_thread_limit(token: int) -> None:
    """End a limit started by ``enter_thread_limit``.

    A limit that ends on another thread than it began on leaves the OpenMP
    count of its first thread until a limit next begins or ends there.
    """
    with _thread_limit_lock:
        _end_thread_limit(token, threading.get_ident())


@contextmanager
def process_thread_limit(
    limits: int | None, *, user_api: Literal["blas"] | None = None
) -> Iterator[None]:
    """Hold a limit from ``enter_thread_limit`` for the duration of a block."""
    token = enter_thread_limit(limits, user_api=user_api)
    try:
        yield
    finally:
        exit_thread_limit(token)


def thread_pool_info() -> list[dict[str, Any]]:
    """Describe the native thread pools that a limit reaches, as threadpoolctl does.

    The description comes from the shared controller, so it builds none.
    """
    with _thread_limit_lock:
        return list(_shared_thread_pool_controller().info())


def controlled_compute(arr: Any, nthreads: int) -> np.ndarray:
    """Materialize a deferred array with a bounded thread count."""
    if hasattr(arr, "compute"):
        return np.asarray(arr.compute(nthreads))
    return np.asarray(arr)


def compute_with_progress(
    arr: Any,
    msg: str | None = None,
    nthreads: int = 1,
) -> np.ndarray:
    """Materialize a deferred array while reporting progress."""
    if hasattr(arr, "compute"):
        progress_msg = msg if progress_requested() else None
        return np.asarray(arr.compute(nthreads, progress_msg))
    return np.asarray(arr)


def pairwise_merge_tree(values: Sequence[T], merge: Callable[[T, T], T]) -> T:
    """Reduce values with a fixed pairwise tree independent of completion order.

    Position in ``values`` is the unit index. Adjacent pairs merge first, then
    the next level, so the association does not depend on which unit finished
    first. An odd leftover at a level is promoted unchanged.
    """
    if not values:
        raise ValueError("merge tree needs at least one value")
    items = list(values)
    while len(items) > 1:
        nxt: list[T] = []
        for index in range(0, len(items), 2):
            if index + 1 < len(items):
                nxt.append(merge(items[index], items[index + 1]))
            else:
                nxt.append(items[index])
        items = nxt
    return items[0]
