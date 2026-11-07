import ast
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from threadpoolctl import ThreadpoolController, threadpool_limits

from cytearc.utils import compute
from cytearc.utils.compute import (
    enter_thread_limit,
    exit_thread_limit,
    process_thread_limit,
    thread_pool_info,
)
from tests.thread_helpers import (
    blas_threads,
    load_openmp_runtime,
    openmp_threads,
    raised_threads,
)


@pytest.fixture
def two_blas_threads():
    with raised_threads(blas=2):
        yield


def test_the_latest_remaining_limit_applies_after_a_middle_exit(two_blas_threads):
    first = enter_thread_limit(1)
    second = enter_thread_limit(2)
    third = enter_thread_limit(1)
    exit_thread_limit(third)
    assert blas_threads() == {2}
    exit_thread_limit(second)
    assert blas_threads() == {1}
    exit_thread_limit(first)
    assert blas_threads() == {2}


def test_a_scope_without_a_limit_still_restores_the_original(two_blas_threads):
    unchanged = enter_thread_limit(None)
    assert blas_threads() == {2}
    limited = enter_thread_limit(1)
    exit_thread_limit(limited)
    # Only the unchanged scope remains, so the original limits apply.
    assert blas_threads() == {2}
    exit_thread_limit(unchanged)
    assert blas_threads() == {2}


def test_the_context_manager_restores_after_an_exception(two_blas_threads):
    with pytest.raises(RuntimeError, match="stage failed"):
        with process_thread_limit(1):
            assert blas_threads() == {1}
            raise RuntimeError("stage failed")
    assert blas_threads() == {2}


def _built_controllers(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Note each ThreadpoolController built from now on."""
    built: list[None] = []
    build = ThreadpoolController.__init__

    def noted(self: ThreadpoolController) -> None:
        built.append(None)
        build(self)

    monkeypatch.setattr(ThreadpoolController, "__init__", noted)
    return built


def test_scopes_reuse_one_controller_while_nothing_is_imported(monkeypatch):
    built = _built_controllers(monkeypatch)
    exit_thread_limit(enter_thread_limit(None))
    before = len(built)

    for _ in range(50):
        exit_thread_limit(enter_thread_limit(1))
        with process_thread_limit(None):
            pass

    # Each scope built two controllers, one at each end, which only the cycle
    # collector frees.
    assert len(built) == before


def test_a_scope_after_an_import_builds_the_controller_again(monkeypatch):
    import sys
    import types

    built = _built_controllers(monkeypatch)
    exit_thread_limit(enter_thread_limit(None))
    before = len(built)
    # An extension module that loads a BLAS or OpenMP runtime is imported.
    stand_in = types.ModuleType("cytearc_native_library_stand_in")
    monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)

    exit_thread_limit(enter_thread_limit(None))
    exit_thread_limit(enter_thread_limit(None))

    assert len(built) == before + 1


def test_many_scopes_leave_no_controller_to_the_cycle_collector():
    import gc

    exit_thread_limit(enter_thread_limit(None))
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        # As many scopes as the blocks of a stream that limits each block's
        # BLAS threads.
        for _ in range(40):
            with process_thread_limit(1):
                pass
        gc.collect()
        left = [obj for obj in gc.garbage if isinstance(obj, ThreadpoolController)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        if enabled:
            gc.enable()

    assert left == []


def test_a_blas_only_limit_leaves_the_openmp_runtimes_as_they_were():
    load_openmp_runtime()
    with raised_threads(blas=3, openmp=3):
        with process_thread_limit(1, user_api="blas"):
            assert blas_threads() == {1}
            assert openmp_threads() == {3}
        with process_thread_limit(1):
            assert blas_threads() == {1}
            assert openmp_threads() == {1}
        assert blas_threads() == {3}
        assert openmp_threads() == {3}


def _on_another_thread(read):
    seen = []
    thread = threading.Thread(target=lambda: seen.append(read()))
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    return seen[0]


def test_an_openmp_limit_sets_the_count_of_its_own_thread_only():
    load_openmp_runtime()
    with raised_threads(blas=3, openmp=3):
        unlimited = _on_another_thread(openmp_threads)
        with process_thread_limit(1):
            assert openmp_threads() == {1}
            # BLAS counts apply to the whole process, OpenMP counts per thread.
            assert _on_another_thread(blas_threads) == {1}
            assert _on_another_thread(openmp_threads) == unlimited
        assert openmp_threads() == {3}


@pytest.mark.parametrize("first_ends_first", [True, False])
def test_overlapping_limits_on_two_threads_restore_what_the_first_recorded(
    first_ends_first,
):
    load_openmp_runtime()
    with raised_threads(blas=3, openmp=3):
        entered = threading.Event()
        release = threading.Event()
        worker: dict[str, object] = {}

        def limit_on_a_worker() -> None:
            worker["before"] = openmp_threads()
            with process_thread_limit(1):
                worker["inside"] = (blas_threads(), openmp_threads())
                entered.set()
                assert release.wait(10)
            worker["after"] = openmp_threads()

        first = enter_thread_limit(2)
        thread = threading.Thread(target=limit_on_a_worker)
        thread.start()
        try:
            assert entered.wait(10)
            # The worker's limit is the most recent, so BLAS follows it; this
            # thread's OpenMP count follows the limit that began here.
            assert blas_threads() == {1}
            assert openmp_threads() == {2}
            if first_ends_first:
                exit_thread_limit(first)
                assert blas_threads() == {1}
                assert openmp_threads() == {3}
                release.set()
                thread.join(10)
            else:
                release.set()
                thread.join(10)
                assert blas_threads() == {2}
                assert openmp_threads() == {2}
                exit_thread_limit(first)
        finally:
            release.set()
            thread.join(10)
        assert not thread.is_alive()
        assert blas_threads() == {3}
        assert openmp_threads() == {3}
        assert worker["inside"] == ({1}, {1})
        assert worker["after"] == worker["before"]
        assert compute._active_thread_limits == {}


def test_a_limit_without_a_count_leaves_the_limits_before_it(two_blas_threads):
    limited = enter_thread_limit(1)
    unchanged = enter_thread_limit(None)
    # The most recent limit sets no count, so the one before it applies.
    assert blas_threads() == {1}
    exit_thread_limit(unchanged)
    assert blas_threads() == {1}
    exit_thread_limit(limited)
    assert blas_threads() == {2}


def test_a_thread_forgets_its_openmp_counts_once_its_limits_end():
    load_openmp_runtime()
    with raised_threads(blas=2, openmp=3):
        with process_thread_limit(1):
            assert openmp_threads() == {1}
        assert openmp_threads() == {3}
        # The count changes between limits, as a caller may set it, and the
        # next limit restores the new count, not the one before.
        with threadpool_limits(limits={"openmp": 5}):
            with process_thread_limit(1):
                assert openmp_threads() == {1}
            assert openmp_threads() == {5}
        assert compute._original_openmp_threads == {}


class _FakeLibrary:
    """A library that threadpoolctl would find, with a refused count."""

    def __init__(
        self,
        filepath: str,
        threads: int,
        refused: int | None = None,
        user_api: str = "blas",
    ):
        self.filepath = filepath
        self.threads = threads
        self.refused = refused
        self.user_api = user_api
        self.history = [threads]

    def get_num_threads(self) -> int:
        return self.threads

    def set_num_threads(self, threads: int) -> None:
        if threads == self.refused:
            raise RuntimeError(f"{self.filepath} refused {threads} threads")
        self.threads = threads
        self.history.append(threads)


def _with_libraries(monkeypatch, libraries: list) -> None:
    controller = types.SimpleNamespace(lib_controllers=libraries)
    monkeypatch.setattr(compute, "_shared_thread_pool_controller", lambda: controller)


def test_a_limit_that_cannot_be_set_ends_and_leaves_the_counts(monkeypatch):
    fine = _FakeLibrary("fine.so", 4)
    refusing = _FakeLibrary("refusing.so", 4, refused=1)
    _with_libraries(monkeypatch, [fine, refusing])

    with pytest.raises(RuntimeError, match="^refusing.so refused 1 threads$"):
        enter_thread_limit(1)
    # The caller holds no token, so the limit was ended and the counts restored.
    assert compute._active_thread_limits == {}
    assert compute._original_threads == {}
    assert fine.history == [4, 1, 4]
    assert refusing.threads == 4


def test_a_library_loaded_during_a_limit_gets_its_own_count_back(monkeypatch):
    numpy_blas = _FakeLibrary("numpy.so", 8)
    libraries = [numpy_blas]
    _with_libraries(monkeypatch, libraries)

    first = enter_thread_limit(2)
    # An import loads another BLAS while the first limit holds.
    scipy_blas = _FakeLibrary("scipy.so", 6)
    libraries.append(scipy_blas)
    second = enter_thread_limit(1)
    assert (numpy_blas.threads, scipy_blas.threads) == (1, 1)
    exit_thread_limit(second)
    assert (numpy_blas.threads, scipy_blas.threads) == (2, 2)
    exit_thread_limit(first)
    assert (numpy_blas.threads, scipy_blas.threads) == (8, 6)
    assert compute._original_threads == {}


def test_a_blas_only_limit_leaves_the_other_runtimes_of_the_process(monkeypatch):
    # SciPy's Matrix Market reader registers its thread pool with threadpoolctl
    # under its own name, and its count applies to the whole process.
    blas = _FakeLibrary("openblas.so", 8)
    reader = _FakeLibrary("_fmm_core.so", 0, user_api="scipy")
    _with_libraries(monkeypatch, [blas, reader])

    every = enter_thread_limit(3)
    assert (blas.threads, reader.threads) == (3, 3)
    blas_only = enter_thread_limit(1, user_api="blas")
    # The BLAS-only limit is the most recent, and reaches BLAS alone.
    assert (blas.threads, reader.threads) == (1, 3)
    exit_thread_limit(every)
    assert (blas.threads, reader.threads) == (1, 0)
    exit_thread_limit(blas_only)
    assert (blas.threads, reader.threads) == (8, 0)
    assert reader.history == [0, 3, 0]


def test_a_limit_names_only_blas_or_every_runtime():
    with pytest.raises(
        ValueError, match="^user_api must be None or 'blas', not 'openmp'$"
    ):
        enter_thread_limit(1, user_api="openmp")  # type: ignore[arg-type]
    assert compute._active_thread_limits == {}


def test_thread_pool_info_describes_the_pools_without_building_a_controller(
    monkeypatch,
):
    from threadpoolctl import threadpool_info

    expected = sorted(pool["filepath"] for pool in threadpool_info())
    built = _built_controllers(monkeypatch)
    exit_thread_limit(enter_thread_limit(None))
    before = len(built)
    pools = thread_pool_info()
    assert len(built) == before
    assert sorted(pool["filepath"] for pool in pools) == expected


# Native thread-limit calls that the package makes only inside its two helpers:
# threadpoolctl's limits and controllers, and Numba's thread count. hnswlib's
# index threads are an index setting, not a native pool, and stay with the index.
_THREADPOOLCTL_CALLS = {"threadpool_limits", "ThreadpoolController", "threadpool_info"}
_ALLOWED_THREAD_CALLS = {
    ("utils/compute.py", "_shared_thread_pool_controller", "ThreadpoolController"),
    ("utils/numba.py", "numba_threads", "set_num_threads"),
}


def _thread_limit_calls(source: str) -> list[tuple[int, str, str | None]]:
    """Each use of a native thread-limit call, with its line and function.

    Follows the names that a module binds to threadpoolctl and Numba, imports
    from them, attribute chains from them and ``getattr`` with the name.
    """
    tree = ast.parse(source)
    modules: dict[str, str] = {}
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in ("threadpoolctl", "numba"):
                    modules[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            for alias in node.names:
                if root == "threadpoolctl" and alias.name in _THREADPOOLCTL_CALLS:
                    imported[alias.asname or alias.name] = alias.name
                elif root == "numba" and alias.name == "set_num_threads":
                    imported[alias.asname or alias.name] = alias.name
                elif root == "numba":
                    modules[alias.asname or alias.name] = "numba"

    def root_module(node: ast.expr) -> str | None:
        while isinstance(node, ast.Attribute):
            node = node.value
        return modules.get(node.id) if isinstance(node, ast.Name) else None

    found: list[tuple[int, str, str | None]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def note(self, node: ast.AST, name: str) -> None:
            function = self.functions[-1] if self.functions else None
            found.append((node.lineno, name, function))  # type: ignore[attr-defined]

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if (
                    alias.name in imported.values()
                    and (alias.asname or alias.name) in imported
                ):
                    self.note(node, alias.name)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            module = root_module(node.value)
            if module == "threadpoolctl" and node.attr in _THREADPOOLCTL_CALLS:
                self.note(node, node.attr)
            elif module == "numba" and node.attr == "set_num_threads":
                self.note(node, node.attr)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and root_module(node.args[0]) is not None
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _THREADPOOLCTL_CALLS | {"set_num_threads"}
            ):
                self.note(node, str(node.args[1].value))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def test_only_the_two_helpers_limit_native_threads() -> None:
    import cytearc

    package = Path(cytearc.__file__).parent
    allowed_seen = set()
    refused = []
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package).as_posix()
        for line, name, function in _thread_limit_calls(path.read_text()):
            if (module, function, name) in _ALLOWED_THREAD_CALLS:
                allowed_seen.add((module, function, name))
            else:
                refused.append(f"cytearc/{module}:{line} {name} in {function}")
    assert refused == [], (
        "Limit native threads through cytearc.utils.compute.process_thread_limit "
        "and Numba's through cytearc.utils.numba.numba_threads:\n" + "\n".join(refused)
    )
    # The helpers themselves are still found, so the guard still sees the calls.
    assert allowed_seen == _ALLOWED_THREAD_CALLS


@pytest.mark.parametrize(
    "source",
    [
        "from threadpoolctl import threadpool_limits",
        "from threadpoolctl import ThreadpoolController as Controller",
        "from threadpoolctl import threadpool_info",
        "import threadpoolctl\nthreadpoolctl.threadpool_limits(limits=1)",
        "import threadpoolctl as pools\npools.ThreadpoolController()",
        "import threadpoolctl\ngetattr(threadpoolctl, 'threadpool_limits')(1)",
        "import numba\nnumba.set_num_threads(2)",
        "import numba as nb\nnb.set_num_threads(2)",
        "from numba import set_num_threads",
        "from numba import set_num_threads as threads",
        "import numba\nsetter = getattr(numba, 'set_num_threads')",
        "import numba.np.ufunc.parallel\nnumba.np.ufunc.parallel.set_num_threads(2)",
        "from numba.np.ufunc.parallel import set_num_threads",
        "from numba.np.ufunc import parallel\nparallel.set_num_threads(2)",
    ],
)
def test_the_guard_finds_each_way_to_limit_threads(source: str) -> None:
    calls = _thread_limit_calls(
        "def change():\n" + "".join(f"    {line}\n" for line in source.splitlines())
    )
    assert [function for _line, _name, function in calls] == ["change"]


@pytest.mark.parametrize(
    "source",
    [
        "index.set_num_threads(4)",
        "import numba\nnumba.get_num_threads()",
        "import numba\nnumba.config.NUMBA_NUM_THREADS",
        "from numba import prange, get_num_threads",
        "import threadpoolctl\nthreadpoolctl.__version__",
        "set_num_threads = 4",
    ],
)
def test_the_guard_passes_other_thread_settings(source: str) -> None:
    assert _thread_limit_calls(source) == []


# One call of each family of thread limits, built once and run twice: the
# first call imports and compiles, the second is the one measured.
def _variability_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.features.variability import fit_lowess

    rng = np.random.default_rng(0)
    means = rng.lognormal(0.0, 1.0, 400)
    variances = means * rng.lognormal(0.2, 0.3, 400)
    return lambda: fit_lowess(means, variances, 20, 0.2, bin_strategy="adaptive")


def _harmony_call(_tmp_path: Path) -> Callable[[], object]:
    import pandas as pd

    from cytearc.embeddings.harmony import fit_harmony

    rng = np.random.default_rng(1)
    values = rng.normal(size=(5, 300))
    batches = pd.DataFrame({"batch": np.repeat(["a", "b"], 150)})
    return lambda: fit_harmony(
        values, batches, n_clusters=3, max_iter_harmony=1, nthreads=2
    )


def _pca_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.embeddings.reduction import fit_pca
    from cytearc.matrix import ChunkedArray

    values = np.random.default_rng(2).normal(size=(200, 12))
    data = ChunkedArray.from_numpy(values, block_size=50, nthreads=2)
    rows = np.ones(len(values), dtype=bool)
    return lambda: fit_pca(
        data, dims=3, use_for_pca=rows, scale=None, random_state=0, nthreads=2
    )


def _stage_values() -> Any:
    from cytearc.matrix import ChunkedArray

    values = np.abs(np.random.default_rng(3).normal(size=(120, 6)))
    return values, ChunkedArray.from_numpy(values, block_size=40, nthreads=2)


def _reduction_stage(method: str) -> Callable[[Path], Callable[[], object]]:
    def build(_tmp_path: Path) -> Callable[[], object]:
        from cytearc.neighbors.stages import ReductionTransform

        values, data = _stage_values()

        def run() -> object:
            return ReductionTransform(
                data=data,
                method=method,
                dims=2,
                loadings=None,
                use_for_pca=np.ones(len(values), dtype=bool),
                mu=values.mean(axis=0),
                sigma=values.std(axis=0),
                batch_size=40,
                nthreads=2,
                rand_state=4466,
                disable_scaling=False,
                lsi_skip_first=False,
                lsi_params={},
            )

        return run

    return build


def _kmeans_stage_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.neighbors.stages import (
        ChunkedCoordinateStream,
        KMeansInitializationStage,
    )

    values, data = _stage_values()
    # One batch of every row fits the model in memory, with its ``fit``.
    return lambda: KMeansInitializationStage.fit(
        stream=ChunkedCoordinateStream(data, 2),
        n_rows=len(values),
        batch_size=len(values),
        n_clusters=3,
        rand_state=4466,
        nthreads=2,
    )


def _harmony_stage_call(_tmp_path: Path) -> Callable[[], object]:
    import pandas as pd

    from cytearc.matrix import ChunkedArray
    from cytearc.neighbors.stages import BatchCorrectionStage, ChunkedCoordinateStream

    values = np.random.default_rng(9).normal(size=(200, 4))
    data = ChunkedArray.from_numpy(values, block_size=50, nthreads=2)
    batches = pd.DataFrame({"batch": np.repeat(["a", "b"], 100)})
    return lambda: BatchCorrectionStage(
        stream=ChunkedCoordinateStream(data, 2),
        n_cells=len(values),
        dims=values.shape[1],
        batch_size=50,
        batches=batches,
        parameters={"n_clusters": 3, "max_iter_harmony": 1},
        corrected_data=None,
        nthreads=2,
    ).ensure_corrected()


def _integration_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.neighbors.integration import _wnn_integration_many

    rng = np.random.default_rng(4)
    n_cells, k = 60, 5
    # Each cell's neighbors are the next cells, so no row holds its own cell.
    neighbors = (np.arange(n_cells)[:, None] + 1 + np.arange(k)) % n_cells
    modalities = [
        (name, neighbors.copy(), rng.normal(size=(n_cells, 4)).astype(np.float32))
        for name in "ab"
    ]
    return lambda: _wnn_integration_many(modalities, 2)


def _doublet_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.mapping.symphony import ScaledPCAProjectionModel
    from cytearc.matrix import ChunkedArray
    from cytearc.quality_control import doublets
    from cytearc.storage.budget import ResourceBudget

    rng = np.random.default_rng(5)
    counts = rng.poisson(3.0, size=(40, 12)).astype(np.int16)
    features = np.arange(12)
    model = ScaledPCAProjectionModel(
        feature_means=rng.normal(size=12),
        feature_scales=rng.uniform(0.5, 2.0, size=12),
        center=rng.normal(size=12),
        loadings=rng.normal(size=(12, 3)),
    )
    reference = types.SimpleNamespace(
        model=model,
        normalization_parameters={
            "size_factor": 100.0,
            "log_transform": True,
            "renormalize_subset": True,
        },
    )
    assay = types.SimpleNamespace(rawData=ChunkedArray.from_numpy(counts), name="RNA")
    members = np.arange(len(counts))
    (plan,) = doublets.plan_doublet_samples(
        [len(counts)], simulation_ratio=2.0, n_neighbors=1
    )
    return lambda: doublets._simulate_sample(
        assay,
        reference,
        np.random.default_rng(6),
        members,
        plan,
        active_indices=members,
        labels=np.zeros(len(counts), dtype=np.int64),
        feature_indices=features,
        cell_totals=None,
        cluster_sample_fraction=1.0,
        max_cells_per_cluster=None,
        heterotypic_fraction=0.0,
        resources=ResourceBudget(1_000_000, 2),
        resident_bytes=0,
    )


def _runner_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.storage.async_execution import AsyncStorageRunner
    from cytearc.storage.budget import ResourceBudget
    from cytearc.storage.execution import WorkShape, plan_operation

    plan = plan_operation(
        ResourceBudget(1 << 20, 2), WorkShape(nUnits=4, unitBytes=1024, ordered=False)
    )

    async def operation(runner: AsyncStorageRunner) -> list[float]:
        import asyncio

        return list(
            await asyncio.gather(
                *(runner.compute(lambda: float(np.ones(4) @ np.ones(4))) for _ in "ab")
            )
        )

    return lambda: AsyncStorageRunner(operation=plan).run(operation)


def _stream_items_call(_tmp_path: Path) -> Callable[[], object]:
    from cytearc.matrix import ChunkedArray
    from cytearc.storage.budget import ResourceBudget

    values = np.arange(64.0).reshape(16, 4)
    matrix = ChunkedArray.from_numpy(
        values, block_size=4, nthreads=2, resources=ResourceBudget(1 << 20, 2)
    )
    return lambda: matrix.sum(axis=0).compute(nthreads=2)


_SCREEN_CELLS = 4 * 12 + 24


def _screen_store(tmp_path: Path) -> object:
    from tests.perturbation_helpers import write_screen

    rng = np.random.default_rng(7)
    n_targets, per_target, n_controls, n_genes = 4, 12, 24, 30
    guide_of = np.r_[
        np.repeat(np.arange(n_targets), per_target), np.full(n_controls, n_targets)
    ]
    rna = rng.poisson(5.0, size=(len(guide_of), n_genes)).astype(np.uint16)
    guides = np.zeros((len(guide_of), n_targets + 1), dtype=np.uint16)
    guides[np.arange(len(guide_of)), guide_of] = 20
    targets = [*(f"T{target}" for target in range(n_targets)), "NT"]
    return write_screen(
        tmp_path / "screen.zarr", rna, guides, targets=targets, nthreads=2
    )


def _signature_call(tmp_path: Path) -> Callable[[], object]:
    from tests.perturbation_helpers import fixed_labels

    store: Any = _screen_store(tmp_path)
    labels = fixed_labels(store, ["NT"])
    cells = store.select_measured_cells("RNA")
    coordinates = store.reduction.import_coordinates(
        np.random.default_rng(8).normal(size=(_SCREEN_CELLS, 4)),
        cell_selection=cells,
        name="X",
    )
    features = store.features.snapshot(feature_indexes=np.arange(30))
    return lambda: store.perturbation.signature(
        labels,
        coordinates,
        features=features,
        dims=4,
        n_neighbors=5,
        invalidate_cache=True,
    )


def _profile_calls(tmp_path: Path) -> tuple[Any, Any]:
    from tests.perturbation_helpers import fixed_labels

    store: Any = _screen_store(tmp_path)
    labels = fixed_labels(store, ["NT"])
    features = store.features.snapshot(feature_indexes=np.arange(30))
    profiles = store.perturbation.profiles(
        labels, features, min_cells=5, id_column=None
    )
    return store, profiles


def _cluster_call(tmp_path: Path) -> Callable[[], object]:
    store, profiles = _profile_calls(tmp_path)
    return lambda: store.perturbation.cluster(
        profiles, min_cluster_size=2, invalidate_cache=True
    )


def _load_clusters_call(tmp_path: Path) -> Callable[[], object]:
    store, profiles = _profile_calls(tmp_path)
    clusters = store.perturbation.cluster(profiles, min_cluster_size=2)
    return lambda: store.perturbation.load_clusters(clusters)


class _Family(NamedTuple):
    """One call of a family, the work inside its limit, and the counts it runs with.

    ``blas`` is the BLAS count the work runs with; ``openmp`` says whether the
    limit reaches OpenMP too, at the same count, or leaves it as it was.
    """

    build: Callable[[Path], Callable[[], object]]
    work: tuple[str, str] | None = None
    blas: int | None = None
    openmp: bool = True


_FAMILIES = {
    "feature variability": _Family(
        _variability_call,
        ("cytearc.features.variability", "_fit_lowess_adaptive"),
        blas=1,
        openmp=False,
    ),
    "Harmony": _Family(_harmony_call, ("cytearc.embeddings.harmony.fit", "_fit"), 2),
    "PCA": _Family(_pca_call, ("cytearc.embeddings.reduction", "_fit_gram_pca"), 2),
    "PCA stage": _Family(
        _reduction_stage("pca"),
        ("cytearc.neighbors.stages:ReductionTransform", "_fit_pca"),
        2,
    ),
    "LSI stage": _Family(
        _reduction_stage("lsi"),
        ("cytearc.neighbors.stages:ReductionTransform", "_fit_lsi"),
        2,
    ),
    "k-means stage": _Family(
        _kmeans_stage_call, ("sklearn.cluster:MiniBatchKMeans", "fit"), 2
    ),
    "Harmony stage": _Family(
        _harmony_stage_call, ("cytearc.neighbors.stages", "_fitted_harmony"), 2
    ),
    "WNN integration": _Family(
        _integration_call, ("cytearc.neighbors.integration", "_integrate_blocks"), 2
    ),
    "doublets": _Family(
        _doublet_call, ("cytearc.quality_control.doublets", "sum_doublet_pairs"), 2
    ),
    "storage runner": _Family(_runner_call),
    "block stream items": _Family(_stream_items_call),
    "Mixscape signature": _Family(
        _signature_call,
        ("cytearc.perturbation.signature", "nearest_controls"),
        blas=2,
        openmp=False,
    ),
    "profile clusters": _Family(
        _cluster_call,
        ("cytearc.perturbation.clusters", "cluster_profiles"),
        blas=2,
        openmp=False,
    ),
    "loaded profile clusters": _Family(
        _load_clusters_call,
        ("cytearc.perturbation.clusters", "profile_distances"),
        blas=2,
        openmp=False,
    ),
}
_SCREEN_FAMILIES = {"Mixscape signature", "profile clusters", "loaded profile clusters"}


def _family_params(names) -> list[Any]:
    return [
        pytest.param(name, marks=pytest.mark.slow) if name in _SCREEN_FAMILIES else name
        for name in names
    ]


@pytest.mark.parametrize("family", _family_params(_FAMILIES))
def test_each_family_limits_threads_without_a_controller_of_its_own(
    family, tmp_path, monkeypatch
):
    import gc

    run = _FAMILIES[family].build(tmp_path)
    run()
    # The first call may import modules after its last limit; a limit now
    # builds the shared controller that sees them, as the next call would.
    exit_thread_limit(enter_thread_limit(None))
    built = _built_controllers(monkeypatch)
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        run()
        gc.collect()
        left = [obj for obj in gc.garbage if isinstance(obj, ThreadpoolController)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        if enabled:
            gc.enable()
    assert built == []
    assert left == []
    assert compute._active_thread_limits == {}


@pytest.mark.parametrize(
    "family",
    _family_params(name for name, family in _FAMILIES.items() if family.work),
)
def test_each_family_limits_the_runtimes_it_limits(family, tmp_path, monkeypatch):
    import importlib

    load_openmp_runtime()
    chosen = _FAMILIES[family]
    assert chosen.work is not None
    run = chosen.build(tmp_path)
    # The first call loads every runtime the family uses, such as Numba's
    # OpenMP layer, so that the counts below reach them all.
    run()
    module_name, attribute = chosen.work
    module_name, _, class_name = module_name.partition(":")
    owner = importlib.import_module(module_name)
    if class_name:
        owner = getattr(owner, class_name)
    original = getattr(owner, attribute)
    seen: list[tuple[set[int], set[int]]] = []

    def watched(*args: Any, **kwargs: Any) -> Any:
        seen.append((blas_threads(), openmp_threads()))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, watched)
    # Three threads outside, apart from each family's count and from one.
    with raised_threads(blas=3, openmp=3):
        run()
        assert (blas_threads(), openmp_threads()) == ({3}, {3})
    expected = ({chosen.blas}, {chosen.blas} if chosen.openmp else {3})
    assert seen and all(counts == expected for counts in seen)
