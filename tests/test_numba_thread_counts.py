"""Each parallel kernel runs on the Numba threads its caller asks for.

Every caller sets Numba's count through ``cytearc.utils.numba.numba_threads``,
so the count of the calling thread comes back once the call returns or fails.
Each case wraps the kernel that runs inside the setting and reads the count
the kernel starts with.
"""

from collections.abc import Callable
from typing import Any

import numba
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

_REQUESTED = 2


def _harmony() -> tuple[str, str, Callable[[], object]]:
    from cytearc.embeddings.harmony import fit_harmony

    values = np.random.default_rng(0).normal(size=(4, 120))
    batches = pd.DataFrame({"batch": np.repeat(["a", "b"], 60)})
    return (
        "cytearc.embeddings.harmony.fit",
        "_fit",
        lambda: fit_harmony(
            values, batches, n_clusters=3, max_iter_harmony=1, nthreads=_REQUESTED
        ),
    )


def _paris() -> tuple[str, str, Callable[[], object]]:
    from cytearc.clustering._paris_core import fit_paris_hierarchy

    graph = sp.random(40, 40, density=0.2, random_state=1, format="csr")
    graph = graph + graph.T
    return (
        "cytearc.clustering._paris_core",
        "_fit_component",
        lambda: fit_paris_hierarchy(graph, nthreads=_REQUESTED),
    )


def _leiden() -> tuple[str, str, Callable[[], object]]:
    from cytearc.clustering._leiden_native import native_leiden_membership

    ring = sp.diags([np.ones(29)], [1], shape=(30, 30), format="csr")
    return (
        "cytearc.clustering._leiden_native",
        "build_leiden_graph",
        lambda: native_leiden_membership(ring, 1.0, 1, _REQUESTED),
    )


def _curve() -> tuple[str, str, Callable[[], object]]:
    from cytearc.trajectory.curves import project_onto_curve

    rows = np.random.default_rng(2).normal(size=(50, 2))
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    return (
        "cytearc.trajectory.curves",
        "_project_rows",
        lambda: project_onto_curve(
            rows, points, np.array([0.0, 1.0, 2.0]), nthreads=_REQUESTED
        ),
    )


def _correlations() -> tuple[str, str, Callable[[], object]]:
    from cytearc.trajectory.lineages import weighted_correlations

    rng = np.random.default_rng(3)
    values = rng.normal(size=(30, 3))
    weights = rng.uniform(size=(30, 2))
    regressors = rng.normal(size=(30, 2))
    return (
        "cytearc.trajectory.lineages",
        "_weighted_correlation_kernel",
        lambda: weighted_correlations(
            values,
            weights,
            regressors,
            min_mass=0.0,
            nthreads=_REQUESTED,
            feature_labels=["a", "b", "c"],
        ),
    )


def _profiles() -> tuple[str, str, Callable[[], object]]:
    from cytearc.trajectory.lineages import lineage_profiles

    rng = np.random.default_rng(4)
    values = rng.normal(size=(20, 3))
    # Two lineages of two bins: each cell sits in one bin of each lineage.
    operator = sp.csr_matrix(rng.uniform(size=(4, 20)))
    return (
        "cytearc.trajectory.lineages",
        "_bin_sums",
        lambda: lineage_profiles(values, operator, None, 2, nthreads=_REQUESTED),
    )


def _coupling() -> tuple[str, str, Callable[[], object]]:
    from cytearc.clones.fate import clone_coupling

    groups, clones, cells = (
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 2]),
        np.array([2, 3, 1, 4]),
    )
    return (
        "cytearc.clones.fate",
        "_permuted_couplings",
        lambda: clone_coupling(
            groups, clones, cells, 2, permutations=20, workers=_REQUESTED
        ),
    )


def _cell_calling() -> tuple[str, str, Callable[[], object]]:
    from cytearc.quality_control import cell_calling

    proportions = np.array([0.4, 0.3, 0.2, 0.1])
    counts = np.array([[1, 1, 0, 0], [0, 0, 2, 1], [3, 0, 0, 1], [1, 1, 1, 1]])
    totals = counts.sum(axis=1)
    sampler = cell_calling._sampler(proportions, 1.0)
    parts, _ = cell_calling._feature_parts([counts], len(counts), sampler)
    log_probabilities = cell_calling._total_parts(totals, 1.0) + parts
    return (
        "cytearc.quality_control.cell_calling",
        "_simulate",
        lambda: cell_calling._monte_carlo_p_values(
            totals,
            parts,
            log_probabilities,
            sampler,
            n_iter=50,
            random_seed=1,
            nthreads=_REQUESTED,
        ),
    )


def _integration() -> tuple[str, str, Callable[[], object]]:
    from cytearc.neighbors.integration import _wnn_integration_many

    rng = np.random.default_rng(5)
    n_cells, k = 40, 4
    neighbors = (np.arange(n_cells)[:, None] + 1 + np.arange(k)) % n_cells
    modalities = [
        (name, neighbors.copy(), rng.normal(size=(n_cells, 3)).astype(np.float32))
        for name in "ab"
    ]
    return (
        "cytearc.neighbors.integration",
        "_integrate_blocks",
        lambda: _wnn_integration_many(modalities, _REQUESTED),
    )


class _FeatureBatches:
    """A stand-in assay that yields one batch of normalized features."""

    nthreads = _REQUESTED

    def __init__(self, cells_by_features: np.ndarray) -> None:
        self.values = cells_by_features

    def iter_normed_feature_wise(self, **_options: Any) -> Any:
        yield self.values.T.copy(), np.arange(self.values.shape[1])


def _marker_ranks() -> tuple[str, str, Callable[[], object]]:
    from cytearc.features.markers.search import _rank_normed_batches

    rng = np.random.default_rng(6)
    values = rng.poisson(2.0, size=(40, 5)).astype(np.float64)
    codes = np.repeat(np.arange(2), 20).astype(np.int64)
    group_sizes = np.bincount(codes)
    statistics = np.zeros((5, 2, 8))
    return (
        "cytearc.features.markers.search",
        "_batch_stats",
        lambda: _rank_normed_batches(
            _FeatureBatches(values),
            codes,
            group_sizes,
            np.arange(40),
            np.arange(5),
            statistics,
            nthreads=_REQUESTED,
            held_bytes=0,
            norm_params={},
            exact=None,
        ),
    )


def _marker_regression() -> tuple[str, str, Callable[[], object]]:
    from cytearc.features.markers.search import find_markers_by_regression

    rng = np.random.default_rng(7)
    values = rng.poisson(2.0, size=(30, 4)).astype(np.float64)
    return (
        "cytearc.features.markers.search",
        "_regression_batch_results",
        lambda: find_markers_by_regression(
            _FeatureBatches(values),
            np.arange(30),
            np.arange(4),
            np.arange(30.0),
            2,
            log_transform=None,
        ),
    )


_CASES = {
    "Harmony": _harmony,
    "Paris": _paris,
    "Leiden": _leiden,
    "trajectory curves": _curve,
    "lineage correlations": _correlations,
    "lineage profiles": _profiles,
    "clone coupling": _coupling,
    "cell calling": _cell_calling,
    "WNN integration": _integration,
    "marker ranks": _marker_ranks,
    "marker regression": _marker_regression,
}


@pytest.mark.parametrize("case", list(_CASES))
def test_each_kernel_runs_on_the_requested_threads_and_the_count_comes_back(
    case, monkeypatch
):
    import importlib

    if numba.config.NUMBA_NUM_THREADS < _REQUESTED:
        pytest.skip(f"needs Numba to start {_REQUESTED} threads")
    module_name, kernel, call = _CASES[case]()
    module = importlib.import_module(module_name)
    original = getattr(module, kernel)
    seen: list[int] = []

    def watched(*args: Any, **kwargs: Any) -> Any:
        seen.append(numba.get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, kernel, watched)
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        call()
        after = numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)
    assert seen and set(seen) == {_REQUESTED}
    assert after == 1


def test_a_setting_that_ends_on_another_thread_changes_nothing_there():
    import threading

    from cytearc.utils.numba import numba_threads

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("needs Numba to start 2 threads")
    entered: list[tuple[Any, int]] = []

    def enter_on_a_worker() -> None:
        context = numba_threads(2)
        context.__enter__()
        entered.append((context, numba.get_num_threads()))

    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        worker = threading.Thread(target=enter_on_a_worker)
        worker.start()
        worker.join(10)
        ((context, inside),) = entered
        # As a pool's worker holds its count for its life and the runner ends
        # the setting once the worker has finished.
        context.__exit__(None, None, None)
        after = numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)
    assert inside == 2
    assert after == 1
