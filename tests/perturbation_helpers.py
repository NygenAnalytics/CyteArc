"""Shared helpers for the tests of ``cytearc.perturbation``."""

from types import ModuleType

import numpy as np
import pytest


def use_python_kernels(monkeypatch: pytest.MonkeyPatch, *modules: ModuleType) -> None:
    """Replace every numba kernel of ``modules`` with its Python function.

    Kernels look up the kernels they call at call time when they run as Python,
    so the whole call tree runs uncompiled and coverage sees every line.
    """
    for module in modules:
        for name, value in list(vars(module).items()):
            if hasattr(value, "py_func"):
                monkeypatch.setattr(module, name, value.py_func)


def screen_counts(
    n_cells: int,
    n_guides: int,
    *,
    seed: int,
    ambient: float = 0.3,
    carried: float = 0.8,
    log2_mean: float = 6.0,
    log2_std: float = 1.0,
    dtype: type = np.uint32,
) -> np.ndarray:
    """Simulate the guide counts of a low-MOI screen.

    Every cell holds Poisson ambient counts of every guide, and a share of the
    cells also carries one guide with a count drawn on the log2 scale.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(ambient, size=(n_cells, n_guides)).astype(np.int64)
    guide = rng.integers(0, n_guides, size=n_cells)
    carriers = np.flatnonzero(rng.random(n_cells) < carried)
    signal = np.round(2.0 ** rng.normal(log2_mean, log2_std, size=len(carriers)))
    counts[carriers, guide[carriers]] += signal.astype(np.int64) + 1
    return counts.astype(dtype)
