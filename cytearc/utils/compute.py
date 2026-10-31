import itertools
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import numpy as np

from .progress import progress_requested

T = TypeVar("T")

_thread_limit_lock = threading.Lock()
_thread_limit_tokens = itertools.count()
_active_thread_limits: dict[int, int | None] = {}
_original_thread_limits: Any | None = None
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


def enter_thread_limit(limits: int | None) -> int:
    """Start a process-wide native thread-pool limit and return its token.

    Native thread-pool limits apply to the whole process, so scopes held by
    generators or entered on other threads can end in any order. The first
    active scope records the limits in force, the most recent remaining request
    applies while scopes overlap, and the last scope to end restores the
    recorded limits. ``None`` changes nothing but still takes part.
    """
    global _original_thread_limits
    with _thread_limit_lock:
        controller = _shared_thread_pool_controller()
        if not _active_thread_limits:
            _original_thread_limits = controller.limit(limits=None)
        token = next(_thread_limit_tokens)
        _active_thread_limits[token] = limits
        if limits is not None:
            controller.limit(limits=limits)
    return token


def exit_thread_limit(token: int) -> None:
    """End a limit started by ``enter_thread_limit``."""
    global _original_thread_limits
    with _thread_limit_lock:
        del _active_thread_limits[token]
        original = cast(Any, _original_thread_limits)
        requested = [
            limits for limits in _active_thread_limits.values() if limits is not None
        ]
        if not _active_thread_limits:
            _original_thread_limits = None
        if requested:
            _shared_thread_pool_controller().limit(limits=requested[-1])
        else:
            original.restore_original_limits()


@contextmanager
def process_thread_limit(limits: int | None) -> Iterator[None]:
    """Hold a limit from ``enter_thread_limit`` for the duration of a block."""
    token = enter_thread_limit(limits)
    try:
        yield
    finally:
        exit_thread_limit(token)


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
