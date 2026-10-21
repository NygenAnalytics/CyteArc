"""On-disk caching of Numba kernels that third-party packages compile without it."""

import json
import math
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numba.core.caching import FunctionCache, IndexDataCacheFile, NullCache
from scipy import sparse

from cytearc.embeddings.umap import fit_transform
from cytearc.neighbors.graph import build_connectivity_arrays
from cytearc.utils.logging import logger
from cytearc.utils.numba import enable_dispatcher_cache

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
    return getattr(sys.modules[module], attribute)


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
        monkeypatch.setattr(sys.modules[module], attribute, fresh)


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
import umap.layouts
import umap.umap_

from tests.test_numba_dispatcher_cache import run_umap_kernels

# The kernels compiled with cache=True have their caches by now, so only the
# caches enabled from here on, those of the umap-learn kernels, use this one.
numba.config.CACHE_DIR = sys.argv[1]
print(json.dumps(run_umap_kernels()))
"""


@pytest.mark.slow
def test_a_later_process_loads_the_umap_learn_kernels_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This process compiles the kernels as a first process does, into a cache
    # directory of their own.
    _fresh_umap_kernels(monkeypatch)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path))
    first = run_umap_kernels()

    process = subprocess.run(
        [sys.executable, "-c", _LATER_PROCESS, str(tmp_path)],
        check=True,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    later = json.loads(process.stdout.splitlines()[-1])

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
