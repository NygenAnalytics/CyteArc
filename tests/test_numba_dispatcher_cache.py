"""On-disk caching of Numba kernels that third-party packages compile without it."""

import importlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numba.core.caching import FunctionCache, IndexDataCacheFile, NullCache
from numba.core.dispatcher import Dispatcher
from scipy import sparse

from cytearc.embeddings.umap import fit_transform
from cytearc.neighbors.graph import build_connectivity_arrays
from cytearc.utils.logging import logger
from cytearc.utils.numba import (
    _caching_import_time_kernels,
    _has_cache_entry,
    cache_directory_lock,
    enable_dispatcher_cache,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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
