"""On-disk caching of Numba kernels: CyteArc's own, and those that third-party
packages compile without a cache."""

import ast
import builtins
import dis
import importlib
import inspect
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numba.core.caching import FunctionCache, IndexDataCacheFile, NullCache
from numba.core.dispatcher import Dispatcher
from scipy import sparse

import cytearc
from cytearc.embeddings.umap import fit_transform
from cytearc.neighbors.graph import build_connectivity_arrays
from cytearc.utils.arrays import rolling_window
from cytearc.utils.logging import logger
from cytearc.utils.numba import (
    _build_token,
    _BuildToken,
    _caching_import_time_kernels,
    _cytearc_build_token,
    _has_cache_entry,
    _kernel_cache_class,
    _kernel_module_sources,
    _package_build_token,
    _package_sources,
    _source_digest,
    cache_directory_lock,
    enable_dispatcher_cache,
    njit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CYTEARC_DIR = REPOSITORY_ROOT / "cytearc"
UMAP_KERNELS = {
    "smooth_knn_dist": ("umap.umap_", "smooth_knn_dist"),
    "compute_membership_strengths": ("umap.umap_", "compute_membership_strengths"),
    "serial_epoch": ("umap.layouts", "_nb_optimize_layout_euclidean_single_epoch"),
    "parallel_epoch": (
        "umap.layouts",
        "_nb_optimize_layout_euclidean_single_epoch_parallel",
    ),
}


def is_nan(value):
    """Return whether ``value`` is NaN, which fastmath code assumes it never is."""
    return value != value


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send the Numba caches that a test creates to its own directory."""
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_cache_locator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave Numba no writable cache directory to find, as read-only systems do."""
    monkeypatch.setattr(FunctionCache._impl_class, "_locator_classes", [])


@contextmanager
def debug_messages() -> Iterator[list[str]]:
    """Collect CyteArc's debug log messages while the block runs."""
    messages: list[str] = []
    handler = logger.add(
        lambda message: messages.append(str(message)), level="DEBUG", format="{message}"
    )
    try:
        yield messages
    finally:
        logger.remove(handler)


def compiled(dispatcher: Any) -> int:
    return sum(dispatcher.stats.cache_misses.values())


def loaded(dispatcher: Any) -> int:
    return sum(dispatcher.stats.cache_hits.values())


def test_a_dispatcher_without_a_cache_gets_one_on_disk(cache_dir: Path) -> None:
    dispatcher = numba.njit(is_nan)

    enable_dispatcher_cache(dispatcher)

    assert Path(dispatcher.stats.cache_path).is_relative_to(cache_dir)
    assert dispatcher(math.nan)
    assert (compiled(dispatcher), loaded(dispatcher)) == (1, 0)
    # A new dispatcher of the function, as in a new process, loads its code.
    later = numba.njit(is_nan)
    enable_dispatcher_cache(later)
    assert later(math.nan)
    assert (compiled(later), loaded(later)) == (0, 1)


def test_only_dispatchers_with_the_null_cache_are_changed(cache_dir: Path) -> None:
    cached = numba.njit(cache=True)(is_nan)
    own_cache = cached._cache

    enable_dispatcher_cache(cached, is_nan, None, "is_nan")

    assert cached._cache is own_cache


def test_each_dispatcher_is_handled_once_per_process(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = numba.njit(is_nan)
    enable_dispatcher_cache(dispatcher)
    cache = dispatcher._cache

    enable_dispatcher_cache(dispatcher)

    assert dispatcher._cache is cache
    # A dispatcher that Numba could not cache is not tried again.
    monkeypatch.setattr(FunctionCache._impl_class, "_locator_classes", [])
    uncached = numba.njit(is_nan)
    with debug_messages() as messages:
        enable_dispatcher_cache(uncached)
        enable_dispatcher_cache(uncached)
    assert len(messages) == 1


@pytest.mark.parametrize("fast_first", [False, True], ids=["exact_first", "fast_first"])
def test_dispatchers_of_one_function_keep_the_code_of_their_options(
    cache_dir: Path, fast_first: bool
) -> None:
    # Fastmath code assumes that no value is NaN, so it answers False. Numba's
    # own cache key leaves out target options, so without them in the key the
    # second dispatcher would load the first one's code and its answer.
    order = [{"fastmath": True}, {}] if fast_first else [{}, {"fastmath": True}]
    dispatchers = [numba.njit(is_nan, **options) for options in order]
    enable_dispatcher_cache(*dispatchers)

    for options, dispatcher in zip(order, dispatchers, strict=True):
        assert bool(dispatcher(math.nan)) is not bool(options)
        assert (compiled(dispatcher), loaded(dispatcher)) == (1, 0)

    # New dispatchers, as in a new process, load the code of their options.
    for options in order:
        later = numba.njit(is_nan, **options)
        enable_dispatcher_cache(later)
        assert bool(later(math.nan)) is not bool(options)
        assert (compiled(later), loaded(later)) == (0, 1)


def test_options_saved_by_racing_processes_keep_their_own_entries(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Numba rewrites a cache index from the copy that it read, without a lock.
    # Processes that compile different options at once each read the index
    # before the other saves, so in one shared index the later save drops the
    # earlier entry or points its key at the other options' machine code.
    order = [{}, {"fastmath": True}]
    with monkeypatch.context() as patch:
        # Each save reads the index as it was before the other one saved.
        patch.setattr(IndexDataCacheFile, "_load_index", lambda self: {})
        dispatchers = [numba.njit(is_nan, **options) for options in order]
        enable_dispatcher_cache(*dispatchers)
        for dispatcher in dispatchers:
            dispatcher(math.nan)

    index_paths = {
        dispatcher._cache._cache_file._index_path for dispatcher in dispatchers
    }
    assert len(index_paths) == 2
    # New dispatchers, as in a later process, load the code of their options.
    for options in order:
        later = numba.njit(is_nan, **options)
        enable_dispatcher_cache(later)
        assert bool(later(math.nan)) is not bool(options)
        assert (compiled(later), loaded(later)) == (0, 1)


def test_a_dispatcher_numba_cannot_cache_still_computes(no_cache_locator) -> None:
    dispatcher = numba.njit(is_nan)

    with debug_messages() as messages:
        enable_dispatcher_cache(dispatcher)

    assert isinstance(dispatcher._cache, NullCache)
    assert dispatcher(math.nan)
    assert not dispatcher(1.0)
    (message,) = messages
    assert "is_nan, so each process compiles it" in message
    assert "no locator available" in message


def test_dispatchers_stay_uncached_without_numbas_cache_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Numba release that moves its cache classes leaves every kernel uncached.
    monkeypatch.delattr("numba.core.caching.NullCache")
    dispatcher = numba.njit(is_nan)

    with debug_messages() as messages:
        enable_dispatcher_cache(dispatcher)

    assert isinstance(dispatcher._cache, NullCache)
    assert dispatcher(math.nan)
    (message,) = messages
    assert "Numba dispatchers stay uncached" in message


def test_a_cache_entry_that_cannot_be_saved_does_not_fail_the_call(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(self, key, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(IndexDataCacheFile, "save", disk_full)
    dispatcher = numba.njit(is_nan)
    enable_dispatcher_cache(dispatcher)

    with debug_messages() as messages:
        assert dispatcher(math.nan)

    (message,) = messages
    assert "the next process compiles it again" in message
    assert "No space left on device" in message


def test_a_corrupt_cache_entry_is_compiled_again(cache_dir: Path) -> None:
    dispatcher = numba.njit(is_nan)
    enable_dispatcher_cache(dispatcher)
    assert dispatcher(math.nan)
    entries = list(cache_dir.rglob("*.nbc"))
    assert entries
    for entry in entries:
        entry.write_bytes(b"truncated")
    later = numba.njit(is_nan)
    enable_dispatcher_cache(later)

    with debug_messages() as messages:
        assert later(math.nan)

    assert (compiled(later), loaded(later)) == (1, 0)
    assert any("cache entry cannot be read" in message for message in messages)


def _make_unreadable(index: Path) -> None:
    index.chmod(0)
    if os.access(index, os.R_OK):
        pytest.skip("the file stays readable, as it does for root")


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda index: index.write_bytes(b""),
        lambda index: index.write_bytes(b"\x80\x05truncated"),
        # Numba's version, which Numba reads first, then a payload that is
        # not a stamp and its entries.
        lambda index: index.write_bytes(
            pickle.dumps(numba.__version__) + pickle.dumps(5)
        ),
        _make_unreadable,
    ],
    ids=["empty", "truncated", "foreign_payload", "no_read_permission"],
)
def test_an_unreadable_cache_index_is_rewritten_by_the_next_save(
    cache_dir: Path, corrupt: Callable[[Path], None]
) -> None:
    # Numba reads the index before it saves an entry, so an index that raises
    # there would make every later process compile the kernel again.
    kernel = njit(cache=True)(is_nan)
    assert kernel(math.nan)
    (index,) = cache_dir.rglob("*.nbi")
    corrupt(index)
    after = njit(cache=True)(is_nan)

    with debug_messages() as messages:
        assert after(math.nan)
    later = njit(cache=True)(is_nan)
    assert later(math.nan)

    # The first dispatcher after the damage, as in the next process, compiles
    # and rewrites the index, and the one after it loads what that one saved.
    assert (compiled(after), loaded(after)) == (1, 0)
    assert (compiled(later), loaded(later)) == (0, 1)
    assert any(
        message.startswith(f"Numba cache index {index} cannot be read, so it counts ")
        for message in messages
    )


# A package whose decorators fix the signatures of two kernels, which then
# compile while the package imports, as pynndescent's do.
_PROBE_PACKAGE = "cytearc_import_time_kernels"
_PROBE_SOURCE = """
import numba


@numba.njit("float64(float64)")
def double(value):
    return 2.0 * value


@numba.njit(["int64(int64)", "float64(float64)"])
def triple(value):
    return 3 * value


@numba.njit
def increment(value):
    return value + 1
"""


@pytest.fixture
def probe_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Return the directory that holds an importable copy of the probe package."""
    root = tmp_path / "packages"
    (root / _PROBE_PACKAGE).mkdir(parents=True)
    (root / _PROBE_PACKAGE / "__init__.py").write_text(_PROBE_SOURCE)
    monkeypatch.syspath_prepend(str(root))
    yield root
    sys.modules.pop(_PROBE_PACKAGE, None)


def _import_probe() -> Any:
    sys.modules.pop(_PROBE_PACKAGE, None)
    with _caching_import_time_kernels(frozenset({_PROBE_PACKAGE})):
        return importlib.import_module(_PROBE_PACKAGE)


def test_kernels_that_compile_while_their_package_imports_get_a_cache(
    cache_dir: Path, probe_package: Path
) -> None:
    compile_kernel = Dispatcher.compile

    module = _import_probe()

    assert Dispatcher.compile is compile_kernel
    for kernel, signatures in ((module.double, 1), (module.triple, 2)):
        assert Path(kernel.stats.cache_path).is_relative_to(cache_dir)
        assert (compiled(kernel), loaded(kernel)) == (signatures, 0)
    # A kernel that compiles at its first call compiles after the import.
    assert isinstance(module.increment._cache, NullCache)
    # Imported again, as by a later process, the kernels load their code.
    later = _import_probe()
    for kernel, signatures in ((later.double, 1), (later.triple, 2)):
        assert (compiled(kernel), loaded(kernel)) == (0, signatures)
    assert later.double(2.5) == 5.0
    assert (later.triple(3), later.triple(1.5)) == (9, 4.5)


def test_only_this_threads_compilations_of_the_packages_get_a_cache(
    cache_dir: Path, probe_package: Path
) -> None:
    def identity(value):
        return value

    # A function of the package, compiled by this thread and by another.
    identity.__module__ = _PROBE_PACKAGE
    kernels: dict[str, Any] = {}
    with _caching_import_time_kernels(frozenset({_PROBE_PACKAGE})):
        thread = threading.Thread(
            target=lambda: kernels.update(
                thread=numba.njit("float64(float64)")(identity)
            )
        )
        thread.start()
        thread.join()
        kernels["this thread"] = numba.njit("float64(float64)")(identity)
        kernels["other package"] = numba.njit("boolean(float64)")(is_nan)

    assert isinstance(kernels["thread"]._cache, NullCache)
    assert isinstance(kernels["other package"]._cache, NullCache)
    assert Path(kernels["this thread"].stats.cache_path).is_relative_to(cache_dir)


def test_compilation_is_restored_when_the_import_fails() -> None:
    compile_kernel = Dispatcher.compile

    with pytest.raises(RuntimeError, match="^import failed$"):
        with _caching_import_time_kernels(frozenset({_PROBE_PACKAGE})):
            raise RuntimeError("import failed")

    assert Dispatcher.compile is compile_kernel


def test_package_kernels_compile_uncached_without_a_cache_directory(
    no_cache_locator, probe_package: Path
) -> None:
    with debug_messages() as messages:
        module = _import_probe()

    assert isinstance(module.double._cache, NullCache)
    assert isinstance(module.triple._cache, NullCache)
    assert (module.double(1.5), module.triple(2)) == (3.0, 6)
    assert sum("so each process compiles it" in message for message in messages) == 2


def test_imports_compile_as_before_without_numbas_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Numba release that moves its dispatcher leaves imports as they were.
    monkeypatch.setitem(sys.modules, "numba.core.dispatcher", None)
    ran = False

    with debug_messages() as messages:
        with _caching_import_time_kernels(frozenset({_PROBE_PACKAGE})):
            ran = True

    assert ran
    (message,) = messages
    assert message.startswith("Numba kernels compile uncached while importing: ")


def test_an_object_without_a_numba_cache_has_no_cache_entry() -> None:
    assert not _has_cache_entry(object(), "float64(float64)")


_IMPORTING_PROCESS = """
import importlib
import json
import sys
import time

import numba

from cytearc.utils.numba import _caching_import_time_kernels

# Start importing with the other processes.
time.sleep(max(0.0, float(sys.argv[2]) - time.time()))
with _caching_import_time_kernels(frozenset({sys.argv[1]})):
    module = importlib.import_module(sys.argv[1])
kernels = (module.double, module.triple)
print(json.dumps({
    "compiled": sum(sum(kernel.stats.cache_misses.values()) for kernel in kernels),
    "loaded": sum(sum(kernel.stats.cache_hits.values()) for kernel in kernels),
    "values": [module.double(2.5), module.triple(3), module.triple(1.5)],
}))
"""


def test_processes_that_import_at_once_compile_each_kernel_once(
    tmp_path: Path, probe_package: Path
) -> None:
    env = {
        **os.environ,
        "NUMBA_CACHE_DIR": str(tmp_path / "cache"),
        "PYTHONPATH": os.pathsep.join(
            [str(probe_package), os.environ.get("PYTHONPATH", "")]
        ),
    }
    start = time.time() + 3.0
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _IMPORTING_PROCESS, _PROBE_PACKAGE, str(start)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=300)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout.splitlines()[-1]))

    # Each of the three signatures compiles in one process and loads in two.
    assert sum(result["compiled"] for result in results) == 3
    assert sum(result["loaded"] for result in results) == 6
    assert all(result["values"] == [5.0, 9, 4.5] for result in results)


def _lockable_by_another_process(directory: Path, *, shared: bool) -> bool:
    """Whether another process could take the directory's lock without waiting."""
    code = (
        "import fcntl, sys\n"
        "handle = open(sys.argv[1], 'a')\n"
        "mode = fcntl.LOCK_SH if sys.argv[2] == 'shared' else fcntl.LOCK_EX\n"
        "fcntl.flock(handle, mode | fcntl.LOCK_NB)\n"
    )
    lock_file = directory / ".cytearc-numba.lock"
    mode = "shared" if shared else "exclusive"
    process = subprocess.run(
        [sys.executable, "-c", code, str(lock_file), mode], capture_output=True
    )
    return process.returncode == 0


def test_cache_directory_lock_shares_reads_and_excludes_writes(tmp_path: Path) -> None:
    pytest.importorskip("fcntl")

    with cache_directory_lock(str(tmp_path)):
        assert not _lockable_by_another_process(tmp_path, shared=True)
        # The thread that holds the lock enters again without waiting.
        with cache_directory_lock(str(tmp_path), shared=True):
            with cache_directory_lock(str(tmp_path)):
                pass
        assert not _lockable_by_another_process(tmp_path, shared=True)
    with cache_directory_lock(str(tmp_path), shared=True):
        assert _lockable_by_another_process(tmp_path, shared=True)
        assert not _lockable_by_another_process(tmp_path, shared=False)
    assert _lockable_by_another_process(tmp_path, shared=False)


def test_a_cache_directory_that_cannot_be_locked_runs_unlocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fcntl = pytest.importorskip("fcntl")
    ran = []

    # A directory that holds no lock file.
    missing = tmp_path / "missing"
    with debug_messages() as opened:
        with cache_directory_lock(str(missing)):
            ran.append("missing")
    # A filesystem that refuses locks.
    with monkeypatch.context() as patch:

        def refuse(handle: Any, operation: int) -> None:
            raise OSError(37, "No locks available")

        patch.setattr(fcntl, "flock", refuse)
        with debug_messages() as refused:
            with cache_directory_lock(str(tmp_path)):
                ran.append("refused")
    # A system without flock, such as Windows.
    monkeypatch.setitem(sys.modules, "fcntl", None)
    with debug_messages() as unsupported:
        with cache_directory_lock(str(tmp_path)):
            ran.append("unsupported")

    assert ran == ["missing", "refused", "unsupported"]
    (message,) = opened
    assert message.startswith(
        f"Numba cache lock {missing / '.cytearc-numba.lock'} cannot be opened: "
    )
    assert [message.rstrip() for message in refused] == [
        f"Numba cache lock {tmp_path / '.cytearc-numba.lock'} cannot be taken: "
        "[Errno 37] No locks available"
    ]
    assert unsupported == []


def _knn(n_cells: int = 60, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    indices = np.stack(
        [
            rng.choice(np.delete(np.arange(n_cells), cell), k, replace=False)
            for cell in range(n_cells)
        ]
    )
    distances = np.sort(rng.random((n_cells, k)).astype(np.float32), axis=1)
    return indices, distances


def _layout(graph: sparse.coo_matrix, *, parallel: bool) -> np.ndarray:
    initialization = np.random.default_rng(1).random((graph.shape[0], 2))
    embedding, _a, _b = fit_transform(
        graph,
        initialization.astype(np.float32),
        spread=2.0,
        min_dist=1.0,
        n_epochs=10,
        random_seed=1,
        repulsion_strength=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5,
        densmap_kwds={},
        parallel=parallel,
        nthreads=2,
        verbose=False,
    )
    return embedding


def _umap_kernel(name: str) -> Any:
    module, attribute = UMAP_KERNELS[name]
    return getattr(importlib.import_module(module), attribute)


def run_umap_kernels() -> dict[str, Any]:
    """Build a connectivity map and lay it out serially and in parallel.

    Returns the results and, for each umap-learn kernel, its cache directory
    and how many signatures it compiled and loaded from that cache.
    """
    indices, distances = _knn()
    edges, weights = build_connectivity_arrays(
        indices, distances, local_connectivity=1.0, bandwidth=1.0
    )
    graph = sparse.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(len(indices), len(indices))
    )
    layout = _layout(graph, parallel=False)
    parallel_layout = _layout(graph, parallel=True)
    kernels = {name: _umap_kernel(name) for name in UMAP_KERNELS}
    return {
        "cachePaths": {
            name: kernel.stats.cache_path for name, kernel in kernels.items()
        },
        "compiled": {name: compiled(kernel) for name, kernel in kernels.items()},
        "loaded": {name: loaded(kernel) for name, kernel in kernels.items()},
        "edges": edges.tolist(),
        "weights": weights.tolist(),
        "layout": layout.tolist(),
        "parallelLayoutIsFinite": bool(np.isfinite(parallel_layout).all()),
    }


def _fresh_umap_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace umap-learn's kernels with new dispatchers of the same options.

    Like the kernels of a new process, they have compiled nothing and have
    no cache yet.
    """
    for name, (module, attribute) in UMAP_KERNELS.items():
        kernel = _umap_kernel(name)
        fresh = type(kernel)(
            kernel.py_func, locals=kernel.locals, targetoptions=kernel.targetoptions
        )
        monkeypatch.setattr(importlib.import_module(module), attribute, fresh)


def test_connectivity_and_umap_layouts_cache_their_umap_learn_kernels() -> None:
    result = run_umap_kernels()

    assert result["parallelLayoutIsFinite"]
    for name in UMAP_KERNELS:
        kernel = _umap_kernel(name)
        assert isinstance(kernel._cache, FunctionCache), name
        assert result["cachePaths"][name] is not None, name


@pytest.mark.slow
def test_connectivity_and_umap_layouts_compute_when_numba_cannot_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = run_umap_kernels()
    _fresh_umap_kernels(monkeypatch)
    monkeypatch.setattr(FunctionCache._impl_class, "_locator_classes", [])

    with debug_messages() as messages:
        result = run_umap_kernels()

    assert result["cachePaths"] == dict.fromkeys(UMAP_KERNELS)
    assert result["compiled"] == dict.fromkeys(UMAP_KERNELS, 1)
    assert sum("no locator available" in message for message in messages) == 4
    for name in ("edges", "weights", "layout"):
        assert result[name] == expected[name], name
    assert result["parallelLayoutIsFinite"]


_LATER_PROCESS = """
import json
import sys

import numba

from cytearc.utils.numba import import_umap
from tests.test_numba_dispatcher_cache import run_umap_kernels

# This process imports umap-learn while the first process compiles the
# kernels that umap-learn calls, and waits until it has saved them.
import_umap()
sys.stdin.readline()
# The kernels that the import compiled have their caches by now, so only the
# caches enabled from here on, those of the kernels umap-learn calls, use this
# one.
numba.config.CACHE_DIR = sys.argv[1]
print(json.dumps(run_umap_kernels()))
"""


@pytest.mark.slow
def test_a_later_process_loads_the_umap_learn_kernels_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    later_process = subprocess.Popen(
        [sys.executable, "-c", _LATER_PROCESS, str(tmp_path)],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # This process compiles the kernels as a first process does, into a
        # cache directory of their own.
        _fresh_umap_kernels(monkeypatch)
        monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path))
        first = run_umap_kernels()
    except BaseException:
        later_process.kill()
        later_process.communicate()
        raise
    # The later process reads the cache only once the kernels are saved.
    stdout, stderr = later_process.communicate("\n")
    assert later_process.returncode == 0, stderr
    later = json.loads(stdout.splitlines()[-1])

    for result in (first, later):
        assert set(result["cachePaths"]) == set(UMAP_KERNELS)
        for path in result["cachePaths"].values():
            assert Path(path).is_relative_to(tmp_path)
        assert result["parallelLayoutIsFinite"]
    assert first["compiled"] == dict.fromkeys(UMAP_KERNELS, 1)
    assert first["loaded"] == dict.fromkeys(UMAP_KERNELS, 0)
    # The later process compiles none of the kernels, the parallel layout
    # epoch included, and computes the same graph and serial layout.
    assert later["compiled"] == dict.fromkeys(UMAP_KERNELS, 0)
    assert later["loaded"] == dict.fromkeys(UMAP_KERNELS, 1)
    for name in ("edges", "weights", "layout"):
        assert later[name] == first[name], name


_UMAP_IMPORT_PROCESS = """
import json
import sys

from numba.core.dispatcher import Dispatcher

from cytearc.utils.numba import import_umap

import_umap()
# The kernels whose signatures umap-learn and pynndescent fix, which compiled
# while they imported.
kernels = {
    id(value): value
    for name, module in list(sys.modules.items())
    if name.partition(".")[0] in ("pynndescent", "umap")
    for value in vars(module).values()
    if isinstance(value, Dispatcher) and not value._can_compile
}.values()

from tests.test_numba_dispatcher_cache import run_umap_kernels

result = run_umap_kernels()
print(json.dumps({
    **{name: result[name] for name in ("edges", "weights", "layout")},
    "kernels": len(kernels),
    "uncached": sum(kernel._cache.cache_path is None for kernel in kernels),
    "compiled": sum(sum(kernel.stats.cache_misses.values()) for kernel in kernels),
    "loaded": sum(sum(kernel.stats.cache_hits.values()) for kernel in kernels),
}))
"""


@pytest.mark.slow
def test_a_later_process_loads_the_kernels_that_importing_umap_compiles(
    tmp_path: Path,
) -> None:
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}
    results = []
    for _ in range(2):
        process = subprocess.run(
            [sys.executable, "-c", _UMAP_IMPORT_PROCESS],
            cwd=REPOSITORY_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr
        results.append(json.loads(process.stdout.splitlines()[-1]))
    first, later = results

    assert first["kernels"] == later["kernels"] > 0
    assert first["uncached"] == later["uncached"] == 0
    assert first["compiled"] > 0
    assert first["loaded"] == 0
    # The later process compiles none of them and computes the same graph and
    # layout with the machine code it loaded.
    assert (later["compiled"], later["loaded"]) == (0, first["compiled"])
    for name in ("edges", "weights", "layout"):
        assert later[name] == first[name], name


# CyteArc's kernels compile with cytearc.utils.numba.njit, whose caches also
# hold CyteArc's version and a digest of its modules that use njit.


# A package in which a caller kernel calls a callee kernel and reads a constant
# of the callee's file. Version 1 computes run(1) == 20; version 2, which
# changes only the callee's file, computes run(1) == 300.
_TWO_FILE_PACKAGE = "kernel_cache_probe"
_CALLEE_SOURCE = """{imports}

SCALE = {scale}


{decorator}
def step(value):
    return value + {increment}
"""
_CALLER_SOURCE = """{imports}

from .callee import SCALE, step


{decorator}
def run(value):
    return step(value) * SCALE
"""
# How each kind of package imports and compiles its kernels. The third-party
# package compiles them without a cache, as umap-learn does, and its files
# never use njit.
_KERNEL_STYLES = {
    "cytearc": ("from ..utils.numba import njit", "@njit(cache=True)"),
    "numba": ("from numba import njit", "@njit(cache=True)"),
    "third_party": ("import numba", "@numba.jit(nopython=True)"),
}
_TWO_FILE_PROCESS = """
import importlib
import json
import sys

caller = importlib.import_module(sys.argv[1])
if sys.argv[2] == "enable_dispatcher_cache":
    from cytearc.utils.numba import enable_dispatcher_cache

    enable_dispatcher_cache(caller.run)
value = caller.run(1)
cytearc = sys.modules.get("cytearc")
print(json.dumps({
    "value": int(value),
    "compiled": sum(caller.run.stats.cache_misses.values()),
    "loaded": sum(caller.run.stats.cache_hits.values()),
    "cachePath": caller.run.stats.cache_path,
    "cytearc": None if cytearc is None else cytearc.__file__,
}))
"""


def _callee_source(style: str, *, increment: int, scale: int) -> str:
    imports, decorator = _KERNEL_STYLES[style]
    return _CALLEE_SOURCE.format(
        imports=imports, decorator=decorator, increment=increment, scale=scale
    )


def _write_callee(package: Path, style: str, *, increment: int, scale: int) -> None:
    (package / "callee.py").write_text(
        _callee_source(style, increment=increment, scale=scale)
    )


def _write_two_file_package(package: Path, style: str) -> None:
    imports, decorator = _KERNEL_STYLES[style]
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "caller.py").write_text(
        _CALLER_SOURCE.format(imports=imports, decorator=decorator)
    )
    _write_callee(package, style, increment=1, scale=10)


def _cytearc_copy(root: Path) -> Path:
    """Copy CyteArc's package into ``root`` and return the copy."""
    copy = root / "cytearc"
    shutil.copytree(CYTEARC_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def _probe_environment(**variables: str) -> dict[str, str]:
    """Return this process's environment for a probe process, without the
    variables that choose where or whether Numba caches, plus ``variables``."""
    unset = {"NUMBA_CACHE_DIR", "NUMBA_CACHE_LOCATOR_CLASSES", "NUMBA_DISABLE_JIT"}
    environment = {
        name: value for name, value in os.environ.items() if name not in unset
    }
    return {**environment, "PYTHONDONTWRITEBYTECODE": "1", **variables}


def _user_directories(root: Path) -> dict[str, str]:
    """Return the variables that put the user's cache directory, where Numba
    caches the kernels of read-only packages, under ``root / "user"``:
    ``XDG_CACHE_HOME`` on Linux and the home directory on macOS."""
    return {
        "HOME": str(root / "user"),
        "XDG_CACHE_HOME": str(root / "user" / "cache"),
    }


@contextmanager
def _read_only(directory: Path) -> Iterator[None]:
    """Make ``directory`` read-only in the block, skipping the test where it
    stays writable, as it does for root."""
    mode = directory.stat().st_mode
    directory.chmod(0o555)
    try:
        if os.access(directory, os.W_OK):
            pytest.skip("the directory stays writable, as it does for root")
        yield
    finally:
        directory.chmod(mode)


def _runs_across_a_callee_change(
    package: Path,
    style: str,
    caller: str,
    *,
    cwd: Path,
    env: dict[str, str],
    enable_cache: bool = False,
) -> list[tuple[int, int, int, dict[str, Any]]]:
    """Run the caller in two processes, change only the callee's file, and run
    it in two more.

    Returns, for each process, the value of ``run(1)``, how many signatures of
    ``run`` it compiled and loaded, and its whole report.
    """

    def run() -> tuple[int, int, int, dict[str, Any]]:
        mode = "enable_dispatcher_cache" if enable_cache else "own_cache"
        process = subprocess.run(
            [sys.executable, "-c", _TWO_FILE_PROCESS, caller, mode],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert process.returncode == 0, process.stderr
        report = json.loads(process.stdout.splitlines()[-1])
        return report["value"], report["compiled"], report["loaded"], report

    results = [run(), run()]
    _write_callee(package, style, increment=2, scale=100)
    return [*results, run(), run()]


@pytest.mark.slow
@pytest.mark.parametrize("location", ["in_tree", "cache_dir", "read_only"])
def test_a_caller_compiles_again_when_its_callees_file_changes(
    tmp_path: Path, location: str
) -> None:
    # The package sits in a copy of CyteArc, so its kernels are CyteArc's
    # kernels and their token digests the copy.
    root = tmp_path / "copy"
    package = _cytearc_copy(root) / _TWO_FILE_PACKAGE
    _write_two_file_package(package, "cytearc")
    env = _probe_environment(PYTHONPATH=str(root), **_user_directories(tmp_path))
    # Coverage cannot map the copy's sources onto this checkout's.
    for name in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START"):
        env.pop(name, None)
    if location == "cache_dir":
        env["NUMBA_CACHE_DIR"] = str(tmp_path / "cache")
    cache_root = {
        "in_tree": package / "__pycache__",
        "cache_dir": tmp_path / "cache",
        # Numba falls back to the user's cache directory.
        "read_only": tmp_path / "user",
    }[location]

    with _read_only(package) if location == "read_only" else nullcontext():
        results = _runs_across_a_callee_change(
            package, "cytearc", f"cytearc.{_TWO_FILE_PACKAGE}.caller", cwd=root, env=env
        )

    assert [result[:3] for result in results] == [
        (20, 1, 0),
        (20, 0, 1),
        (300, 1, 0),
        (300, 0, 1),
    ]
    for *_, report in results:
        assert Path(report["cytearc"]).is_relative_to(root)
        assert Path(report["cachePath"]).is_relative_to(cache_root)


# Runs the steps on its command line in a process that imports the two-file
# package from a copy of CyteArc, and reports what run(1) returned at each
# "run" step and how many signatures the caller's last dispatcher compiled and
# loaded.
_EDITING_PROCESS = """
import importlib
import json
import sys
from pathlib import Path

package, callee_path, original, edited, *steps = sys.argv[1:]
values = []
for step in steps:
    if step == "import_kernels":
        # A CyteArc kernel module computes the build token as it imports.
        importlib.import_module("cytearc.utils.strided")
    elif step == "import":
        caller = importlib.import_module(package + ".caller")
    elif step in ("edit", "revert"):
        Path(callee_path).write_text(edited if step == "edit" else original)
    elif step == "reload":
        importlib.reload(sys.modules[package + ".callee"])
        caller = importlib.reload(caller)
    elif step == "run":
        values.append(int(caller.run(1)))
    else:
        raise ValueError(step)
print(json.dumps({
    "values": values,
    "compiled": sum(caller.run.stats.cache_misses.values()),
    "loaded": sum(caller.run.stats.cache_hits.values()),
}))
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    ("session", "session_result", "later_result"),
    [
        # A kernel module computes the token before the callee's file
        # changes, and the caller first imports after the change.
        (
            ["import_kernels", "edit", "import", "run", "revert"],
            ([300], 1, 0),
            ([20], 1, 0),
        ),
        # The callee's file changes and both modules reload before the
        # caller's first call.
        (["import", "edit", "reload", "run", "revert"], ([300], 1, 0), ([20], 1, 0)),
        # The caller compiles and saves before the change and the reload.
        (["import", "run", "edit", "reload", "run"], ([20, 300], 1, 0), ([300], 1, 0)),
    ],
    ids=["import_after_a_change", "reload_then_revert", "reload"],
)
def test_a_process_that_sees_a_kernel_module_change_neither_loads_nor_saves(
    tmp_path: Path,
    session: list[str],
    session_result: tuple[list[int], int, int],
    later_result: tuple[list[int], int, int],
) -> None:
    # The token holds a digest of the sources as the process first read them.
    # Saved under it, machine code compiled from a changed callee would load
    # in a later process once the change is reverted; loaded under it after a
    # reload, the old machine code would run in place of the changed callee.
    root = tmp_path / "copy"
    package = _cytearc_copy(root) / _TWO_FILE_PACKAGE
    _write_two_file_package(package, "cytearc")
    callee = package / "callee.py"
    original = callee.read_text()
    edited = _callee_source("cytearc", increment=2, scale=100)
    env = _probe_environment(
        PYTHONPATH=str(root), NUMBA_CACHE_DIR=str(tmp_path / "cache")
    )
    # Coverage cannot map the copy's sources onto this checkout's.
    for name in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START"):
        env.pop(name, None)

    def run(steps: list[str]) -> tuple[list[int], int, int]:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _EDITING_PROCESS,
                f"cytearc.{_TWO_FILE_PACKAGE}",
                str(callee),
                original,
                edited,
                *steps,
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert process.returncode == 0, process.stderr
        report = json.loads(process.stdout.splitlines()[-1])
        return report["values"], report["compiled"], report["loaded"]

    assert run(session) == session_result
    # A new process computes the callee as the session left its file.
    assert run(["import", "run"]) == later_result


@pytest.mark.slow
def test_numbas_own_cache_keeps_the_old_callee(tmp_path: Path) -> None:
    # Numba checks a cached kernel only against the file that defines it. If
    # this fails, Numba tracks the files of the kernels a kernel calls, and
    # CyteArc's build token is no longer needed.
    package = tmp_path / _TWO_FILE_PACKAGE
    _write_two_file_package(package, "numba")

    results = _runs_across_a_callee_change(
        package,
        "numba",
        f"{_TWO_FILE_PACKAGE}.caller",
        cwd=tmp_path,
        env=_probe_environment(PYTHONPATH=str(tmp_path)),
    )

    assert [result[:3] for result in results] == [
        (20, 1, 0),
        (20, 0, 1),
        (20, 0, 1),
        (20, 0, 1),
    ]


@pytest.mark.slow
def test_a_third_party_caller_compiles_again_when_its_callees_file_changes(
    tmp_path: Path,
) -> None:
    package = tmp_path / _TWO_FILE_PACKAGE
    _write_two_file_package(package, "third_party")

    results = _runs_across_a_callee_change(
        package,
        "third_party",
        f"{_TWO_FILE_PACKAGE}.caller",
        cwd=tmp_path,
        env=_probe_environment(PYTHONPATH=str(tmp_path)),
        enable_cache=True,
    )

    assert [result[:3] for result in results] == [
        (20, 1, 0),
        (20, 0, 1),
        (300, 1, 0),
        (300, 0, 1),
    ]
    for *_, report in results:
        assert Path(report["cachePath"]) == package / "__pycache__"


def _kernel_digest(directory: Path) -> str:
    files = _kernel_module_sources(directory)
    return _source_digest({name: file.source for name, file in files.items()})


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text() + text)


def test_editing_a_kernel_module_changes_the_build_token(tmp_path: Path) -> None:
    copy = _cytearc_copy(tmp_path)
    before = _kernel_digest(copy)

    _append(copy / "perturbation" / "glm.py", "\n# A comment.\n")

    assert _kernel_digest(copy) != before


def test_editing_a_module_without_njit_keeps_the_build_token(tmp_path: Path) -> None:
    copy = _cytearc_copy(tmp_path)
    before = _kernel_digest(copy)
    module = copy / "datastore" / "datastore.py"
    assert b"njit" not in module.read_bytes()

    _append(module, "\n# A comment.\n")

    assert _kernel_digest(copy) == before


def test_renaming_a_kernel_module_changes_the_build_token(tmp_path: Path) -> None:
    copy = _cytearc_copy(tmp_path)
    before = _kernel_digest(copy)

    (copy / "utils" / "strided.py").rename(copy / "utils" / "strided_views.py")

    assert _kernel_digest(copy) != before


def test_one_tree_at_two_paths_has_one_build_token(tmp_path: Path) -> None:
    first = _cytearc_copy(tmp_path / "first")
    second = _cytearc_copy(tmp_path / "second" / "nested")

    assert _kernel_digest(first) == _kernel_digest(second)
    assert _kernel_digest(first) == _kernel_digest(CYTEARC_DIR)


def _cytearc_token_values(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> tuple[str, ...]:
    """Return the values of the build token that CyteArc's kernels get in a
    process whose CyteArc version is ``version``."""
    monkeypatch.setattr(cytearc, "__version__", version)
    _cytearc_build_token.cache_clear()
    try:
        return _cytearc_build_token().values
    finally:
        # The next token, computed after monkeypatch restores the version, is
        # the process's own again.
        _cytearc_build_token.cache_clear()


@pytest.mark.parametrize(
    ("version", "release"),
    [
        ("1.3.1.dev35+g1eadbb736", "1.3.1"),
        # setuptools-scm marks a tree with uncommitted changes by a date.
        ("1.3.1.dev35+g1eadbb736.d20261031", "1.3.1"),
        ("1.3.1", "1.3.1"),
        ("1.10.0rc1", "1.10.0"),
        # CyteArc's version where neither its metadata nor _version.py exists.
        ("unavailable", "unavailable"),
    ],
)
def test_the_build_token_holds_the_release_part_of_cytearcs_version(
    monkeypatch: pytest.MonkeyPatch, version: str, release: str
) -> None:
    assert _cytearc_token_values(monkeypatch, version) == (
        release,
        _kernel_digest(CYTEARC_DIR),
    )


def test_commits_share_the_build_token_and_releases_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each commit gets its own development version, which compiled every
    # kernel again in the first process of each commit.
    assert _cytearc_token_values(
        monkeypatch, "1.3.1.dev32+g7835e8fb2"
    ) == _cytearc_token_values(monkeypatch, "1.3.1.dev33+gc80bf25d7")
    assert _cytearc_token_values(monkeypatch, "1.3.1") != _cytearc_token_values(
        monkeypatch, "1.3.2"
    )


def test_the_source_digest_keeps_file_boundaries_and_ignores_order() -> None:
    # Moving bytes across a boundary between files, or between a file's path
    # and its source, gives the same byte stream without each part's length.
    assert _source_digest({"a.py": b"xy", "b.py": b"z"}) != _source_digest(
        {"a.py": b"x", "b.py": b"yz"}
    )
    assert _source_digest({"a.py": b"bx"}) != _source_digest({"a.pyb": b"x"})
    assert _source_digest({"a.py": b"x", "b.py": b"y"}) == _source_digest(
        {"b.py": b"y", "a.py": b"x"}
    )


def _function_in_file(path: Path, name: str = "increment") -> Callable[..., Any]:
    """Write a module that defines the function ``name`` and return the
    function, defined from that file as an import would define it."""
    source = f"def {name}(value):\n    return value + 1\n"
    path.write_text(source)
    namespace: dict[str, Any] = {"__name__": path.stem}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[name]


def test_a_third_party_kernels_token_covers_every_file_of_its_package(
    tmp_path: Path,
) -> None:
    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    (outer / "__init__.py").write_text("")
    (outer / "inner" / "__init__.py").write_text("")
    (outer / "settings.py").write_text("SCALE = 10\n")
    (tmp_path / "unrelated.py").write_text("SCALE = 10\n")
    in_package = _function_in_file(outer / "inner" / "kernels.py")
    alone = _function_in_file(tmp_path / "alone.py")

    def tokens() -> tuple[tuple[str, ...], tuple[str, ...]]:
        # Each process digests a package once.
        _package_build_token.cache_clear()
        return _build_token(in_package).values, _build_token(alone).values

    first = tokens()
    (tmp_path / "unrelated.py").write_text("SCALE = 100\n")
    assert tokens() == first
    # A file without kernels in the outermost package, beside the kernel's own
    # subpackage, changes the package kernel's token only.
    (outer / "settings.py").write_text("SCALE = 100\n")
    second = tokens()
    assert second[0] != first[0]
    assert second[1] == first[1]
    _append(tmp_path / "alone.py", "\n# A comment.\n")
    assert tokens()[1] != second[1]


def _write_package(root: Path) -> Path:
    """Write a package that holds a module of constants, and return its
    directory."""
    package = root / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "settings.py").write_text("SCALE = 10\n")
    return package


def _keep_stat(path: Path, write: Callable[[], None]) -> None:
    """Run ``write`` on ``path`` and give the file back its modification time,
    as copies that keep timestamps do."""
    before = path.stat()
    write()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_a_build_token_stays_stale_once_a_file_it_digests_changes(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path)
    settings = package / "settings.py"
    token = _BuildToken((), _package_sources(package))
    assert token.stats_match() and token.sources_match()

    _keep_stat(settings, lambda: settings.write_text("SCALE = 100\n"))

    assert not token.stats_match()
    # With the source and the stat back as they were, modules that the
    # process imported meanwhile may still hold the change.
    _keep_stat(settings, lambda: settings.write_text("SCALE = 10\n"))
    assert os.stat(settings).st_size == len("SCALE = 10\n")
    assert not token.stats_match()
    assert not token.sources_match()
    # A new process gets a token that matches, with the same values.
    fresh = _BuildToken((), _package_sources(package))
    assert fresh.stats_match() and fresh.sources_match()
    assert fresh.values == token.values


def test_a_build_token_digests_its_files_again_before_a_save(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    settings = package / "settings.py"
    token = _BuildToken((), _package_sources(package))

    # An edit that keeps the file's size and modification time.
    _keep_stat(settings, lambda: settings.write_text("SCALE = 20\n"))

    assert token.stats_match()
    assert not token.sources_match()
    assert not token.stats_match()


def test_a_build_token_is_stale_once_a_file_it_digests_moves(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    token = _BuildToken((), _package_sources(package))

    (package / "settings.py").rename(package / "constants.py")

    assert not token.stats_match()


def test_a_token_stays_current_while_a_module_without_njit_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / "kernels.py").write_text(
        "from cytearc.utils.numba import njit\n\n\n"
        "@njit(cache=True)\ndef increment(value):\n    return value + 1\n"
    )
    (tmp_path / "settings.py").write_text("SCALE = 10\n")
    token = _BuildToken(("1.3.1",), _kernel_module_sources(tmp_path))

    (tmp_path / "settings.py").write_text("SCALE = 100\n")

    assert token.stats_match() and token.sources_match()
    _append(tmp_path / "kernels.py", "\n# A comment.\n")
    assert not token.stats_match()


def test_a_kernel_neither_loads_nor_saves_once_its_packages_files_change(
    cache_dir: Path, tmp_path: Path
) -> None:
    package = _write_package(tmp_path)
    function = _function_in_file(package / "kernels.py")
    first = njit(cache=True)(function)
    assert first(1) == 2
    assert (compiled(first), loaded(first)) == (1, 0)
    (index,) = cache_dir.rglob("*.nbi")
    saved = index.read_bytes()

    (package / "settings.py").write_text("SCALE = 100\n")
    later = njit(cache=True)(function)
    with debug_messages() as messages:
        assert later(1) == 2
        assert later(1.5) == 2.5

    # The entry saved for integers stays unread, and neither signature is
    # saved.
    assert (compiled(later), loaded(later)) == (2, 0)
    assert index.read_bytes() == saved
    assert len(list(cache_dir.rglob("*.nbc"))) == 1
    reason = "because its package's source files changed after this process read them"
    assert [message.rstrip() for message in messages] == 2 * [
        f"Compiling kernels.increment without its Numba cache, {reason}",
        f"Numba does not cache kernels.increment, {reason}",
    ]


@pytest.mark.parametrize("location", ["in_tree", "cache_dir", "user_cache"])
def test_scoped_caches_stay_where_numba_puts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    function = _function_in_file(package / "kernels.py")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        numba.config, "CACHE_DIR", str(cache_dir) if location == "cache_dir" else ""
    )
    for name, value in _user_directories(tmp_path).items():
        monkeypatch.setenv(name, value)
    cache_root = {
        "in_tree": package / "__pycache__",
        "cache_dir": cache_dir,
        "user_cache": tmp_path / "user",
    }[location]

    with _read_only(package) if location == "user_cache" else nullcontext():
        cache_path = njit(cache=True)(function)._cache.cache_path
        numba_cache_path = FunctionCache(function).cache_path

    assert cache_path == numba_cache_path
    assert Path(cache_path).is_relative_to(cache_root)


@pytest.mark.parametrize(
    "arguments",
    [
        ("float64(float64)",),
        (["int64(int64)", "float64(float64)"],),
        (numba.float64(numba.float64),),
        (is_nan, is_nan),
    ],
    ids=["text", "list", "signature", "two_functions"],
)
def test_njit_takes_no_signature(arguments: tuple[Any, ...]) -> None:
    with pytest.raises(TypeError, match="takes no signature"):
        njit(*arguments)


def test_njit_refuses_a_cache_option_other_than_true_or_false() -> None:
    with pytest.raises(
        TypeError, match="^njit's cache option must be True or False; got 'yes'$"
    ):
        njit(cache="yes")


def test_njit_returns_the_python_function_without_the_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(numba.config, "DISABLE_JIT", True)

    assert njit(is_nan) is is_nan
    assert njit(cache=True)(is_nan) is is_nan


def test_njit_without_a_cache_leaves_numbas_null_cache() -> None:
    for kernel in (njit(is_nan), njit(cache=False)(is_nan)):
        assert isinstance(kernel, Dispatcher)
        assert isinstance(kernel._cache, NullCache)
    # Numba's other options reach the dispatcher.
    assert njit(is_nan, nogil=True).targetoptions["nogil"] is True


def test_a_kernel_that_njit_caches_loads_in_a_new_dispatcher(cache_dir: Path) -> None:
    kernel = njit(cache=True, nogil=True)(is_nan)

    assert type(kernel._cache) is _kernel_cache_class()
    assert kernel(math.nan)
    assert (compiled(kernel), loaded(kernel)) == (1, 0)
    # A new dispatcher of the function, as in a new process, loads its code.
    later = njit(cache=True, nogil=True)(is_nan)
    assert later(math.nan)
    assert (compiled(later), loaded(later)) == (0, 1)
    assert Path(later.stats.cache_path).is_relative_to(cache_dir)


def _module_name(package: str, relative: str) -> str:
    parts = [package, *Path(relative).with_suffix("").parts]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_njit(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return isinstance(target, ast.Name) and target.id == "njit"


def _kernels(directory: Path, package: str) -> dict[str, tuple[Any, bool]]:
    """Import each module of the package that uses njit and return each kernel
    it compiles, by qualified name, with whether a call wraps its function."""
    kernels: dict[str, tuple[Any, bool]] = {}
    for relative in sorted(_kernel_module_sources(directory)):
        name = _module_name(package, relative)
        module = importlib.import_module(name)
        for node in ast.walk(ast.parse((directory / relative).read_text())):
            if isinstance(node, ast.FunctionDef):
                if not any(_is_njit(decorator) for decorator in node.decorator_list):
                    continue
                kernel, wrapped = node.name, False
            elif (
                isinstance(node, ast.Call)
                and _is_njit(node.func)
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                kernel, wrapped = node.args[0].id, True
            else:
                continue
            # A kernel defined inside a function is out of this walk's reach.
            assert hasattr(module, kernel), f"{name}.{kernel} is not a module global"
            kernels[f"{name}.{kernel}"] = getattr(module, kernel), wrapped
    return kernels


def test_every_cytearc_kernel_has_the_build_scoped_cache() -> None:
    kernels = _kernels(CYTEARC_DIR, "cytearc")
    wrapped = sorted(name for name, (_, is_wrapped) in kernels.items() if is_wrapped)
    # rolling_window compiles its kernel at its first call; a new kernel that
    # compiles the same way needs its own line here.
    assert wrapped == ["cytearc.utils.arrays._rolling_window_kernel"]
    rolling_window(np.ones((3, 1)), 2)
    dispatchers = {
        name: kernel for name, (kernel, is_wrapped) in kernels.items() if not is_wrapped
    }
    dispatchers[wrapped[0]] = importlib.import_module(
        "cytearc.utils.arrays"
    )._rollingWindowImpl

    assert len(dispatchers) > 200
    token = _cytearc_build_token()
    for name, kernel in dispatchers.items():
        assert isinstance(kernel, Dispatcher), name
        assert type(kernel._cache) is _kernel_cache_class(), name
        stamp = kernel._cache._cache_file._source_stamp
        assert len(stamp) == 3 and stamp[1:] == token.values, name


def _code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _code_objects(constant)


def _global_reads(function: Any) -> set[tuple[str, ...]]:
    """Return each global name that ``function``'s code reads, with the
    attributes that it reads from the global in a row."""
    reads: set[tuple[str, ...]] = set()
    for code in _code_objects(function.__code__):
        chain: tuple[str, ...] = ()
        for instruction in dis.get_instructions(code):
            if instruction.opname == "LOAD_ATTR" and chain:
                chain = (*chain, instruction.argval)
                continue
            if instruction.opname == "EXTENDED_ARG":
                continue
            if chain:
                reads.add(chain)
            chain = (instruction.argval,) if instruction.opname == "LOAD_GLOBAL" else ()
        if chain:
            reads.add(chain)
    return reads


def _module_level_statements(statements: list[ast.stmt]) -> Iterator[ast.stmt]:
    for statement in statements:
        yield statement
        if not isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            for field in ("body", "orelse", "finalbody", "handlers"):
                yield from _module_level_statements(getattr(statement, field, []))


def _bound_names(target: ast.expr) -> list[str]:
    return [node.id for node in ast.walk(target) if isinstance(node, ast.Name)]


def _free_names(expression: ast.expr) -> set[str]:
    """Return the names that ``expression`` reads from its module."""
    names = {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
    local = {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    local |= {node.arg for node in ast.walk(expression) if isinstance(node, ast.arg)}
    return names - local


class _ModuleBindings:
    """Where each module-level name of a module comes from, read from its source."""

    def __init__(self, module: types.ModuleType) -> None:
        self.path = Path(inspect.getfile(module)).resolve()
        self.imports: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        self.values: dict[str, list[ast.expr]] = defaultdict(list)
        self.definitions: set[str] = set()
        for statement in _module_level_statements(
            ast.parse(self.path.read_text()).body
        ):
            match statement:
                case ast.Import(names=aliases):
                    for alias in aliases:
                        if alias.asname:
                            self.imports[alias.asname].append((alias.name, None))
                        else:
                            root = alias.name.partition(".")[0]
                            self.imports[root].append((root, None))
                case ast.ImportFrom(module=source, names=aliases, level=level):
                    origin = importlib.util.resolve_name(
                        "." * level + (source or ""), module.__package__
                    )
                    for alias in aliases:
                        self.imports[alias.asname or alias.name].append(
                            (origin, alias.name)
                        )
                case ast.Assign(targets=targets, value=value):
                    for target in targets:
                        for name in _bound_names(target):
                            self.values[name].append(value)
                case (
                    ast.AnnAssign(target=target, value=value)
                    | ast.AugAssign(target=target, value=value)
                ) if value is not None:
                    for name in _bound_names(target):
                        self.values[name].append(value)
                case ast.For(target=target, iter=value):
                    for name in _bound_names(target):
                        self.values[name].append(value)
                case ast.FunctionDef(name=name) | ast.ClassDef(name=name):
                    self.definitions.add(name)

    def binds(self, name: str) -> bool:
        return name in self.imports or name in self.values or name in self.definitions


class _KernelWalk:
    """Find the files whose source a package's kernels compile into their
    machine code: their own, those of the kernels they call, and those of the
    constants they read, following imports and the names that module
    constants are computed from."""

    def __init__(self, directory: Path, package: str) -> None:
        self.directory = directory.resolve()
        self.package = package
        self.bindings: dict[str, _ModuleBindings] = {}

    def in_package(self, module_name: str) -> bool:
        return module_name == self.package or module_name.startswith(f"{self.package}.")

    def module_bindings(self, module_name: str) -> _ModuleBindings:
        if module_name not in self.bindings:
            self.bindings[module_name] = _ModuleBindings(sys.modules[module_name])
        return self.bindings[module_name]

    def origin(
        self, module_name: str, name: str, seen: set[tuple[str, str]]
    ) -> set[Path]:
        """Return the files whose source sets the global ``name`` of a module
        of the package."""
        if (module_name, name) in seen:
            return set()
        seen.add((module_name, name))
        bindings = self.module_bindings(module_name)
        assert bindings.binds(name), f"cannot tell where {module_name}.{name} is set"
        files = {bindings.path}
        for source, attribute in bindings.imports.get(name, []):
            submodule = sys.modules.get(f"{source}.{attribute}")
            if attribute is None or submodule is not None:
                # A module bound to the name: any global it holds may count.
                imported = sys.modules[source] if submodule is None else submodule
                if self.in_package(imported.__name__):
                    files.add(Path(inspect.getfile(imported)).resolve())
            elif self.in_package(source):
                files |= self.origin(source, attribute, seen)
        for value in bindings.values.get(name, []):
            for free in _free_names(value):
                if bindings.binds(free) or free not in vars(builtins):
                    files |= self.origin(module_name, free, seen)
        return files

    def files_read(self, function: Any, pending: list[Any]) -> set[Path]:
        """Return the files that ``function``'s machine code depends on, and
        add the kernels of the package that it calls to ``pending``."""
        assert function.__closure__ is None, f"{function.__qualname__} has a closure"
        files = {Path(inspect.getfile(function)).resolve()}
        module_globals = function.__globals__
        for root, *attributes in _global_reads(function):
            if root not in module_globals:
                # A builtin, which Numba compiles itself.
                continue
            owner, name, value = module_globals["__name__"], root, module_globals[root]
            for attribute in attributes:
                if not isinstance(value, types.ModuleType) or not self.in_package(
                    value.__name__
                ):
                    # Numba compiles its own versions of other packages'
                    # functions, such as NumPy's, and reads their constants.
                    break
                owner, name = value.__name__, attribute
                value = getattr(value, attribute)
            if isinstance(value, types.ModuleType):
                continue
            python_function = getattr(value, "py_func", value)
            if isinstance(value, Dispatcher) or (
                isinstance(value, types.FunctionType | type)
                and self.in_package(value.__module__ or "")
            ):
                files.add(Path(inspect.getfile(python_function)).resolve())
                if isinstance(python_function, types.FunctionType):
                    pending.append(python_function)
            elif self.in_package(owner) and not isinstance(
                value, types.FunctionType | types.BuiltinFunctionType | type
            ):
                files |= self.origin(owner, name, set())
        return files

    def dependencies(self) -> dict[str, set[str]]:
        """Return, for each kernel of the package and each kernel it calls,
        the files whose source its machine code holds, relative to the package
        where they lie in it."""
        pending = [
            getattr(kernel, "py_func", kernel)
            for kernel, _ in _kernels(self.directory, self.package).values()
        ]
        dependencies: dict[str, set[str]] = {}
        while pending:
            function = pending.pop()
            name = f"{function.__module__}.{function.__qualname__}"
            if name in dependencies:
                continue
            files = self.files_read(function, pending)
            dependencies[name] = {
                path.relative_to(self.directory).as_posix()
                if path.is_relative_to(self.directory)
                else str(path)
                for path in files
            }
        return dependencies


def test_kernels_read_globals_only_from_kernel_modules() -> None:
    # The build token digests only the modules that use njit, so every file
    # that a kernel's machine code depends on has to be one of them.
    dependencies = _KernelWalk(CYTEARC_DIR, "cytearc").dependencies()
    digested = set(_kernel_module_sources(CYTEARC_DIR))

    assert {
        kernel: sorted(files - digested)
        for kernel, files in dependencies.items()
        if files - digested
    } == {}
    # The walk sees the kernels that call kernels or read constants of other
    # files, such as Leiden's and those of the perturbation tests.
    assert dependencies["cytearc.perturbation.calling._call_block"] == {
        "perturbation/calling.py",
        "perturbation/histograms.py",
    }
    assert sum(len(files) > 1 for files in dependencies.values()) > 30


_WALK_PACKAGE = "kernel_walk_probe"
_WALK_SOURCES = {
    "__init__.py": "",
    # No kernel, and no use of njit.
    "settings.py": "SCALE = 10\n",
    # A kernel compiled by Numba's own decorator, without njit.
    "legacy.py": """import numba


@numba.jit(nopython=True)
def triple(value):
    return 3 * value
""",
    "helpers.py": """from cytearc.utils.numba import njit

STEP = 1


@njit
def step(value):
    return value + STEP
""",
    "kernels.py": """from cytearc.utils.numba import njit

from . import settings
from .helpers import STEP, step
from .legacy import triple
from .settings import SCALE

LIMIT = tuple(SCALE * factor for factor in range(2))[-1]


@njit
def scaled(value):
    return value * SCALE


@njit
def limited(value):
    return min(value, LIMIT)


@njit
def shifted(value):
    return value + settings.SCALE


@njit
def tripled(value):
    return triple(value)


@njit
def stepped(value):
    return step(value) + STEP
""",
}


def test_the_kernel_walk_reports_files_without_njit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / _WALK_PACKAGE
    package.mkdir()
    for name, source in _WALK_SOURCES.items():
        (package / name).write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        dependencies = _KernelWalk(package, _WALK_PACKAGE).dependencies()
    finally:
        for name in [name for name in sys.modules if name.startswith(_WALK_PACKAGE)]:
            del sys.modules[name]
    digested = set(_kernel_module_sources(package))

    assert digested == {"helpers.py", "kernels.py"}
    # An imported constant, a module constant computed from it, a constant read
    # through its module, and a kernel of a file without njit.
    assert {
        kernel.removeprefix(f"{_WALK_PACKAGE}."): sorted(files - digested)
        for kernel, files in dependencies.items()
        if files - digested
    } == {
        "kernels.scaled": ["settings.py"],
        "kernels.limited": ["settings.py"],
        "kernels.shifted": ["settings.py"],
        "kernels.tripled": ["legacy.py"],
        "legacy.triple": ["legacy.py"],
    }
    assert dependencies[f"{_WALK_PACKAGE}.kernels.stepped"] == {
        "helpers.py",
        "kernels.py",
    }
