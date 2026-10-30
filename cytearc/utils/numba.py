import hashlib
import importlib
import inspect
import os
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import cache, wraps
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, overload

from .logging import logger

if TYPE_CHECKING:
    from numba.core.dispatcher import Dispatcher

# The directory of the cytearc package.
_CYTEARC_DIR = Path(__file__).resolve().parent.parent


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


# Typed as numba.njit is, so each kernel keeps its function's signature.
@overload
def njit[F: Callable[..., Any]](function: F, /, **options: Any) -> "Dispatcher[F]": ...
@overload
def njit[F: Callable[..., Any]](**options: Any) -> "Callable[[F], Dispatcher[F]]": ...
def njit(*args: Any, **options: Any) -> Any:
    """Compile a CyteArc kernel with ``numba.njit``.

    Use it as ``@njit`` or ``@njit(**options)``, with Numba's options. Numba
    checks a cached kernel only against the file that defines it, although
    the machine code also holds the kernels it calls and the constants it
    reads from other files. With ``cache=True``, the kernel gets the cache of
    :func:`enable_dispatcher_cache`, whose entries also hold CyteArc's version
    and a digest of every CyteArc module that uses ``njit``, so a change to
    any of those modules compiles the kernel again.

    Kernels compile at their first call, so ``njit`` takes no signature. With
    ``NUMBA_DISABLE_JIT`` set, it returns the Python function.
    """
    if len(args) > 1 or (args and not callable(args[0])):
        raise TypeError(
            "njit compiles a kernel at its first call and takes no signature: "
            f"pass the kernel function or Numba's keyword options, not {args!r}"
        )
    cached = options.pop("cache", False)
    if not isinstance(cached, bool):
        raise TypeError(f"njit's cache option must be True or False; got {cached!r}")

    def compile_kernel(function: Callable[..., Any]) -> Any:
        import numba
        from numba.core.dispatcher import Dispatcher

        kernel = numba.njit(**options)(function)
        # With NUMBA_DISABLE_JIT set, numba.njit returns the Python function.
        if cached and isinstance(kernel, Dispatcher):
            kernel._cache = _kernel_cache_class()(kernel)
        return kernel

    return compile_kernel(args[0]) if args else compile_kernel


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

    Numba checks a cache entry only against the file that defines the
    function, although the machine code also holds the kernels it calls and
    the constants it reads from other files. These caches also stamp each
    entry with a digest of every Python file of the package that defines the
    function, so a change to any of those files compiles it again.

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

        cache_class = _kernel_cache_class()
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
def _kernel_cache_class() -> type[Any]:
    """Return the Numba function cache of every kernel that CyteArc caches.

    Each set of target options keeps its own files, and each index is stamped
    with the build token of the package that defines the kernel.
    """
    from numba.core.caching import FunctionCache, IndexDataCacheFile

    class KernelCache(FunctionCache):
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
                # Numba loads an entry only while the index holds this stamp,
                # and its own stamp covers only the file that defines the
                # kernel, so the stamp adds the build token.
                source_stamp=(
                    self._impl.locator.get_source_stamp(),
                    *_build_token(dispatcher.py_func),
                ),
            )

        def _index_key(self, sig: Any, codegen: Any) -> tuple[Any, ...]:
            return (*super()._index_key(sig, codegen), self._target_options)

        # Processes on machines with different CPUs that share a cache write
        # more than two entries to an index, which the unlocked rewrite can
        # cross, so loads share the cache directory's lock and saves hold it.
        def load_overload(self, sig: Any, target_context: Any) -> Any:
            try:
                with cache_directory_lock(self._cache_path, shared=True):
                    return super().load_overload(sig, target_context)
            except Exception as error:
                logger.debug(
                    f"Compiling {self._label} because its Numba cache "
                    f"entry cannot be read: {error}"
                )
                return None

        def save_overload(self, sig: Any, data: Any) -> None:
            try:
                with cache_directory_lock(self._cache_path):
                    super().save_overload(sig, data)
            except Exception as error:
                logger.debug(
                    f"Numba cannot cache {self._label}, so the next "
                    f"process compiles it again: {error}"
                )

    return KernelCache


def _build_token(function: Callable[..., Any]) -> tuple[str, ...]:
    """Return the build token of a kernel compiled from ``function``: values
    that change whenever the source that its machine code holds may change.

    For CyteArc's kernels, these are CyteArc's version and a digest of every
    CyteArc module that uses ``njit``. For another package's kernels, which
    nobody edits, the token is a digest of every Python file of the package.
    """
    path = Path(inspect.getfile(function)).resolve()
    if path.is_relative_to(_CYTEARC_DIR):
        return _cytearc_build_token()
    return (_package_digest(_outermost_package(path)),)


@cache
def _cytearc_build_token() -> tuple[str, str]:
    """Return CyteArc's version and a digest of its modules that use ``njit``.

    Every file that a kernel's machine code depends on uses ``njit``, as
    ``tests/test_numba_dispatcher_cache.py`` checks, so editing any other
    module compiles no kernel again. The version also covers a release whose
    kernel modules are unchanged; an editable install keeps the version that
    it was installed with while its sources change.
    """
    from .. import __version__

    return __version__, _source_digest(_kernel_module_sources(_CYTEARC_DIR))


def _kernel_module_sources(directory: Path) -> dict[str, bytes]:
    """Return the source of each module under ``directory`` that uses
    ``njit``, by its path relative to ``directory``."""
    sources = {}
    for path in directory.rglob("*.py"):
        source = path.read_bytes()
        if b"njit" in source:
            sources[path.relative_to(directory).as_posix()] = source
    return sources


@cache
def _package_digest(location: Path) -> str:
    """Return a digest of every Python file of the package directory
    ``location``, or of ``location`` itself, a module outside any package."""
    if location.is_dir():
        directory, files = location, list(location.rglob("*.py"))
    else:
        directory, files = location.parent, [location]
    return _source_digest(
        {path.relative_to(directory).as_posix(): path.read_bytes() for path in files}
    )


def _outermost_package(path: Path) -> Path:
    """Return the outermost package directory that holds the module at
    ``path``, or ``path`` for a module outside any package."""
    package = path
    while (package.parent / "__init__.py").is_file():
        package = package.parent
    return package


def _source_digest(sources: Mapping[str, bytes]) -> str:
    """Return the SHA-256 digest of source files given by their relative paths."""
    digest = hashlib.sha256()
    for name in sorted(sources):
        for part in (name.encode(), sources[name]):
            # Each part's length keeps different files from giving one stream.
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


# Each Numba cache directory holds this lock file beside its index and data
# files.
_LOCK_FILE_NAME = ".cytearc-numba.lock"
# The cache directories whose lock each thread holds.
_HELD_LOCKS = threading.local()


def _held_locks() -> set[str]:
    held: set[str] | None = getattr(_HELD_LOCKS, "directories", None)
    if held is None:
        held = _HELD_LOCKS.directories = set()
    return held


def _locked_file(path: str, *, shared: bool) -> IO[str] | None:
    """Return ``path`` opened and locked, or None where it cannot be locked."""
    try:
        import fcntl
    except ImportError:
        # Windows has no flock, so its processes share Numba caches unlocked.
        return None
    try:
        handle = open(path, "a")
    except OSError as error:
        logger.debug(f"Numba cache lock {path} cannot be opened: {error}")
        return None
    try:
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
    except OSError as error:
        handle.close()
        logger.debug(f"Numba cache lock {path} cannot be taken: {error}")
        return None
    return handle


@contextmanager
def cache_directory_lock(directory: str, *, shared: bool = False) -> Iterator[None]:
    """Hold a Numba cache directory's lock, which every process shares, in the block.

    Numba saves an entry by reading the function's index, naming the data
    file after the first number free in that copy, and writing the index and
    then the data, without a lock. Processes that save entries of one function
    at once can leave the index pointing one entry at another's machine code,
    so that every later load of it fails. A block that writes holds the lock
    alone; ``shared`` blocks, which only read, hold it together.

    A thread that already holds the lock enters again without waiting, so it
    must not ask to write while it reads. Where the lock file cannot be
    created or locked, as on Windows, which has no ``flock``, the block runs
    unlocked.
    """
    held = _held_locks()
    handle = (
        None
        if directory in held
        else _locked_file(os.path.join(directory, _LOCK_FILE_NAME), shared=shared)
    )
    if handle is None:
        yield
        return
    held.add(directory)
    try:
        yield
    finally:
        held.discard(directory)
        # Closing the file releases its lock.
        handle.close()


def _has_cache_entry(dispatcher: Any, sig: Any) -> bool:
    """Return whether the cache index of ``dispatcher`` lists ``sig``."""
    try:
        cache_ = dispatcher._cache
        key = cache_._index_key(sig, dispatcher.targetctx.codegen())
        return key in cache_._cache_file._load_index()
    except Exception:
        return False


_COMPILE_HOOK_LOCK = threading.Lock()


@contextmanager
def _caching_import_time_kernels(packages: frozenset[str]) -> Iterator[None]:
    """Cache the kernels of ``packages`` that this thread compiles in the block.

    A kernel whose decorator fixes its signatures compiles while its module
    is imported, before a caller can give it a cache. In the block, each such
    kernel of ``packages`` that this thread compiles first gets the cache of
    :func:`enable_dispatcher_cache`. A kernel missing from its cache compiles
    and is saved holding its cache directory's lock, and Numba looks in the
    cache again once the lock is held, so processes that import at once
    compile each kernel once and load it from the process that compiled it;
    cached kernels load together. A kernel that already has a cache keeps it.
    Compilations in other threads, which may not wait for this one, are left
    alone.
    """
    try:
        from numba.core.dispatcher import Dispatcher
    except Exception as error:
        logger.debug(f"Numba kernels compile uncached while importing: {error}")
        yield
        return
    with _COMPILE_HOOK_LOCK:
        compile_kernel = Dispatcher.compile
        owner = threading.get_ident()

        def compile_cached(dispatcher: Any, sig: Any) -> Any:
            module = getattr(dispatcher.py_func, "__module__", None) or ""
            if threading.get_ident() != owner or module.partition(".")[0] not in (
                packages
            ):
                return compile_kernel(dispatcher, sig)
            enable_dispatcher_cache(dispatcher)
            directory = dispatcher._cache.cache_path
            if directory is None:
                return compile_kernel(dispatcher, sig)
            with cache_directory_lock(directory, shared=True):
                cached = _has_cache_entry(dispatcher, sig)
            if cached:
                return compile_kernel(dispatcher, sig)
            with cache_directory_lock(directory):
                return compile_kernel(dispatcher, sig)

        setattr(Dispatcher, "compile", compile_cached)
        try:
            yield
        finally:
            setattr(Dispatcher, "compile", compile_kernel)


_UMAP_IMPORT_LOCK = threading.Lock()
_umap_imported = False


def import_umap() -> None:
    """Import umap-learn once, caching the kernels that importing it compiles.

    umap-learn imports pynndescent, whose modules compile about 30 kernels
    with fixed signatures while they load, without an on-disk cache, which
    took 5 to 30 seconds in every new process. Imported here, the kernels of
    both packages compile into the caches of :func:`enable_dispatcher_cache`,
    so a later process loads them, and processes that import at once compile
    each kernel once. Where no cache directory is writable, the import runs
    as it would without this function, and umap-learn imported elsewhere
    first compiles its kernels as it always did.
    """
    global _umap_imported
    if _umap_imported:
        return
    with _UMAP_IMPORT_LOCK:
        if not _umap_imported:
            with _caching_import_time_kernels(frozenset({"pynndescent", "umap"})):
                importlib.import_module("umap.umap_")
            _umap_imported = True
