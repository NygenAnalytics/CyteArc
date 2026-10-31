import hashlib
import importlib
import inspect
import os
import re
import threading
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import cache, wraps
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, NamedTuple, overload

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


@contextmanager
def numba_threads(nthreads: int) -> Iterator[None]:
    """Run Numba's parallel kernels on ``nthreads`` threads, then restore the count.

    The count is at least 1 and at most the threads that Numba started with.
    """
    import numba

    previous = numba.get_num_threads()
    numba.set_num_threads(min(max(1, int(nthreads)), numba.config.NUMBA_NUM_THREADS))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


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
    :func:`enable_dispatcher_cache`, whose entries also hold the release part
    of CyteArc's version and a digest of every CyteArc module that uses
    ``njit``, so a change to any of those modules compiles the kernel again.

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
    function, so a change to any of those files compiles it again. A process
    that sees one of those files change after it digested them neither loads
    nor saves the package's entries from then on, because the modules that it
    imported before and after the change may hold different sources.

    Caching never fails a computation. A dispatcher for which Numba finds no
    writable cache directory, as when ``NUMBA_CACHE_DIR`` is unset and neither
    the package's directory nor the user's cache directory is writable, stays
    uncached, and a cache entry that cannot be read or written is compiled
    instead. An index file that cannot be read counts as empty, so the next
    save rewrites it. Each of these is logged at debug level.

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

    class KernelCacheFile(IndexDataCacheFile):
        # Numba reads the index before it saves an entry and raises when the
        # index cannot be read, so such an index would never be rewritten and
        # every later process would compile the kernel.
        def _load_index(self) -> dict[Any, Any]:
            try:
                overloads: dict[Any, Any] = super()._load_index()
            except Exception as error:
                logger.debug(
                    f"Numba cache index {self._index_path} cannot be read, so "
                    f"it counts as empty and the next save rewrites it: {error}"
                )
                return {}
            return overloads

    class KernelCache(FunctionCache):
        def __init__(self, dispatcher: Any) -> None:
            self._label = _function_name(dispatcher.py_func)
            self._target_options = repr(sorted(dispatcher.targetoptions.items()))
            super().__init__(dispatcher.py_func)
            self._token = _build_token(dispatcher.py_func)
            # Numba rewrites an index from the copy that it read, without a
            # lock, and names a new data file after the first number free in
            # that copy. Processes that save different options to one index at
            # once can leave one key pointing at the other options' machine
            # code, so each set of options keeps its own index and data files.
            digest = hashlib.sha256(self._target_options.encode()).hexdigest()[:16]
            self._cache_file = KernelCacheFile(
                cache_path=self._cache_path,
                filename_base=f"{self._impl.filename_base}-{digest}",
                # Numba loads an entry only while the index holds this stamp,
                # and its own stamp covers only the file that defines the
                # kernel, so the stamp adds the build token.
                source_stamp=(
                    self._impl.locator.get_source_stamp(),
                    *self._token.values,
                ),
            )

        def _index_key(self, sig: Any, codegen: Any) -> tuple[Any, ...]:
            return (*super()._index_key(sig, codegen), self._target_options)

        # Processes on machines with different CPUs that share a cache write
        # more than two entries to an index, which the unlocked rewrite can
        # cross, so loads share the cache directory's lock and saves hold it.
        def load_overload(self, sig: Any, target_context: Any) -> Any:
            try:
                if not self._token.stats_match():
                    logger.debug(
                        f"Compiling {self._label} without its Numba cache, "
                        "because its package's source files changed after "
                        "this process read them"
                    )
                    return None
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
                if not self._token.sources_match():
                    logger.debug(
                        f"Numba does not cache {self._label}, because its "
                        "package's source files changed after this process "
                        "read them"
                    )
                    return
                with cache_directory_lock(self._cache_path):
                    super().save_overload(sig, data)
            except Exception as error:
                logger.debug(
                    f"Numba cannot cache {self._label}, so the next "
                    f"process compiles it again: {error}"
                )

    return KernelCache


class _SourceFile(NamedTuple):
    """A source file as it was read."""

    path: str
    # The file's modification time in nanoseconds and its size, taken before
    # its source was read.
    stat: tuple[int, int]
    source: bytes


class _BuildToken:
    """The values that stamp the cache entries of a package's kernels: values
    that change whenever the source that their machine code holds may change.

    The values hold a digest of the package's source files as this process
    first read them. Once one of those files changes, the modules that the
    process imported before and after the change may hold different sources,
    so the machine code that it compiles may match neither that digest nor
    the files on disk. The token then stays stale for the rest of the process,
    and the kernels that it stamps neither load nor save cache entries.
    """

    def __init__(self, version: tuple[str, ...], files: dict[str, _SourceFile]) -> None:
        self._digest = _source_digest(
            {name: file.source for name, file in files.items()}
        )
        # The path and stat of each file, by its name in the digest.
        self._files = {name: (file.path, file.stat) for name, file in files.items()}
        self._stale = False
        self.values = (*version, self._digest)

    def stats_match(self) -> bool:
        """Return whether the token still matches its source files, judged by
        the modification time and size of each file that it digests.

        From the first check that finds a change on, every check returns False.
        """
        if not self._stale:
            self._stale = any(
                _file_stat(path) != stat for path, stat in self._files.values()
            )
        return not self._stale

    def sources_match(self) -> bool:
        """Return whether the token still matches its source files, judged by
        their stats and by digesting them again, which also finds an edit
        that keeps a file's modification time and size."""
        if self.stats_match():
            sources = {
                name: Path(path).read_bytes() for name, (path, _) in self._files.items()
            }
            self._stale = _source_digest(sources) != self._digest
        return not self._stale


def _build_token(function: Callable[..., Any]) -> _BuildToken:
    """Return the build token of a kernel compiled from ``function``.

    For CyteArc's kernels, its values are the release part of CyteArc's
    version and a digest of every CyteArc module that uses ``njit``. For
    another package's kernels, which nobody edits, its value is a digest of
    every Python file of the package.
    """
    path = Path(inspect.getfile(function)).resolve()
    if path.is_relative_to(_CYTEARC_DIR):
        return _cytearc_build_token()
    return _package_build_token(_outermost_package(path))


@cache
def _cytearc_build_token() -> _BuildToken:
    """Return the build token of CyteArc's kernels: the release part of
    CyteArc's version and a digest of its modules that use ``njit``.

    Every file that a kernel's machine code depends on uses ``njit``, as
    ``tests/test_numba_dispatcher_cache.py`` checks, so the digest changes
    whenever that machine code may change. The token keeps only the release
    part of the version, such as ``1.3.1`` of ``1.3.1.dev35+g1eadbb736``,
    because the development version changes with every commit. A new release
    compiles every kernel again, also when its kernel modules are unchanged,
    and a commit or an edit that changes no module that uses ``njit``
    compiles none again. A version without a release part, such as
    ``unavailable``, stays whole.
    """
    from .. import __version__

    release = re.match(r"\d+(?:\.\d+)*", __version__)
    version = __version__ if release is None else release.group()
    return _BuildToken((version,), _kernel_module_sources(_CYTEARC_DIR))


@cache
def _package_build_token(location: Path) -> _BuildToken:
    """Return the build token of the kernels of the package directory
    ``location``, or of ``location`` itself, a module outside any package:
    a digest of every Python file of the package."""
    return _BuildToken((), _package_sources(location))


def _kernel_module_sources(directory: Path) -> dict[str, _SourceFile]:
    """Read each module under ``directory`` that uses ``njit``."""
    return _read_sources(directory, directory.rglob("*.py"), holding=b"njit")


def _package_sources(location: Path) -> dict[str, _SourceFile]:
    """Read every Python file of the package directory ``location``, or
    ``location`` itself, a module outside any package."""
    if location.is_dir():
        return _read_sources(location, location.rglob("*.py"))
    return _read_sources(location.parent, [location])


def _read_sources(
    directory: Path, paths: Iterable[Path], *, holding: bytes = b""
) -> dict[str, _SourceFile]:
    """Read each file of ``paths`` whose source holds ``holding``, by its path
    relative to ``directory``."""
    files = {}
    for path in paths:
        # The stat comes first, so a write between it and the read leaves a
        # stat that no longer matches the file.
        status = os.stat(path)
        source = path.read_bytes()
        if holding in source:
            files[path.relative_to(directory).as_posix()] = _SourceFile(
                os.fspath(path), (status.st_mtime_ns, status.st_size), source
            )
    return files


def _file_stat(path: str) -> tuple[int, int] | None:
    """Return the modification time in nanoseconds and the size of the file
    at ``path``, or None where it cannot be found."""
    try:
        status = os.stat(path)
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


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
