import hashlib
import threading
import weakref
from collections.abc import Callable
from functools import cache, wraps
from typing import Any

from .logging import logger


def restore_numba_threads[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Restore Numba's process thread count after a call."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        import numba

        previous = numba.get_num_threads()
        try:
            return fn(*args, **kwargs)
        finally:
            numba.set_num_threads(previous)

    return wrapped


_CACHE_LOCK = threading.Lock()
# The dispatchers already handled in this process, cached or not.
_HANDLED_DISPATCHERS: weakref.WeakSet[Any] = weakref.WeakSet()


def enable_dispatcher_cache(*dispatchers: object) -> None:
    """Give Numba dispatchers compiled without a cache an on-disk cache.

    Third-party kernels such as umap-learn's are compiled without
    ``cache=True``, so each new process compiles them again. Called before a
    dispatcher's first use, this lets a later process load its machine code
    from Numba's cache directory, ``NUMBA_CACHE_DIR`` when it is set. Only
    Numba dispatchers whose cache is Numba's null cache are changed, each once
    per process; other objects and dispatchers that have a cache are left
    alone.

    Numba keys a cache entry by the function, its bytecode, the argument
    types, and the CPU, but not by target options such as ``parallel`` or
    ``fastmath``, and it keeps every entry of a function in one index file,
    which it rewrites without a lock. These caches keep the entries of each
    set of target options in their own files and also key them by those
    options, so dispatchers that compile one function with different options,
    such as umap-learn's serial and parallel layout epochs, never load each
    other's machine code, even when processes compile them at the same time.

    Caching never fails a computation. A dispatcher for which Numba finds no
    writable cache directory, as when ``NUMBA_CACHE_DIR`` is unset and neither
    the package's directory nor the user's cache directory is writable, stays
    uncached, and a cache entry that cannot be read or written is compiled
    instead. Both are logged at debug level.

    Args:
        dispatchers: The objects to cache, typically Numba dispatchers.
    """
    try:
        from numba.core.caching import NullCache
        from numba.core.dispatcher import Dispatcher

        cache_class = _option_keyed_cache_class()
    except Exception as error:
        logger.debug(f"Numba dispatchers stay uncached: {error}")
        return
    with _CACHE_LOCK:
        for dispatcher in dispatchers:
            if not isinstance(dispatcher, Dispatcher):
                continue
            if dispatcher in _HANDLED_DISPATCHERS:
                continue
            _HANDLED_DISPATCHERS.add(dispatcher)
            if not isinstance(getattr(dispatcher, "_cache", None), NullCache):
                continue
            try:
                dispatcher._cache = cache_class(dispatcher)
            except Exception as error:
                logger.debug(
                    f"Numba cannot cache {_function_name(dispatcher.py_func)}, "
                    f"so each process compiles it: {error}"
                )


def _function_name(function: Any) -> str:
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", repr(function))
    return name if module is None else f"{module}.{name}"


@cache
def _option_keyed_cache_class() -> type[Any]:
    """Return a Numba function cache with its own files for each set of
    target options."""
    from numba.core.caching import FunctionCache, IndexDataCacheFile

    class OptionKeyedFunctionCache(FunctionCache):
        def __init__(self, dispatcher: Any) -> None:
            self._label = _function_name(dispatcher.py_func)
            self._target_options = repr(sorted(dispatcher.targetoptions.items()))
            super().__init__(dispatcher.py_func)
            # Numba rewrites an index from the copy that it read, without a
            # lock, and names a new data file after the first number free in
            # that copy. Processes that save different options to one index at
            # once can leave one key pointing at the other options' machine
            # code, so each set of options keeps its own index and data files.
            digest = hashlib.sha256(self._target_options.encode()).hexdigest()[:16]
            self._cache_file = IndexDataCacheFile(
                cache_path=self._cache_path,
                filename_base=f"{self._impl.filename_base}-{digest}",
                source_stamp=self._impl.locator.get_source_stamp(),
            )

        def _index_key(self, sig: Any, codegen: Any) -> tuple[Any, ...]:
            return (*super()._index_key(sig, codegen), self._target_options)

        def load_overload(self, sig: Any, target_context: Any) -> Any:
            try:
                return super().load_overload(sig, target_context)
            except Exception as error:
                logger.debug(
                    f"Compiling {self._label} because its Numba cache "
                    f"entry cannot be read: {error}"
                )
                return None

        def save_overload(self, sig: Any, data: Any) -> None:
            try:
                super().save_overload(sig, data)
            except Exception as error:
                logger.debug(
                    f"Numba cannot cache {self._label}, so the next "
                    f"process compiles it again: {error}"
                )

    return OptionKeyedFunctionCache
