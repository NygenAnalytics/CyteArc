"""Shared helpers for the tests of ``cytearc.perturbation``."""

import warnings
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import optimize, special, stats


class _RecordedRows:
    """A cell column that notes the rows that each read of it takes."""

    def __init__(self, array: Any, rows: list[np.ndarray]) -> None:
        self._array = array
        self._rows = rows

    def __getattr__(self, name: str) -> Any:
        return getattr(self._array, name)

    def __getitem__(self, selection: Any) -> Any:
        self._rows.append(np.arange(self._array.shape[0])[selection])
        return self._array[selection]

    def get_orthogonal_selection(self, selection: Any) -> Any:
        self._rows.append(np.asarray(selection[0]).reshape(-1))
        return self._array.get_orthogonal_selection(selection)


def record_cell_reads(
    monkeypatch: pytest.MonkeyPatch, store: Any, columns: set[str]
) -> dict[str, list[np.ndarray]]:
    """Note the rows that every read of these cell columns of ``store`` takes.

    Each column maps to a list that gains the rows of each read, whether a
    whole column, a slice or chosen rows.
    """
    reads: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    get_array = store.cells._get_array

    def recording(column: str) -> Any:
        array = get_array(column)
        return _RecordedRows(array, reads[column]) if column in reads else array

    monkeypatch.setattr(store.cells, "_get_array", recording)
    return reads


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


def size_root(y: np.ndarray, mu: np.ndarray, degrees: float) -> float:
    """The dispersion by a root of the likelihood's score, or the Pearson fallback."""
    if ((y - mu) ** 2 - y).sum() <= 0:
        pearson = (y - mu) ** 2 / mu
        if pearson.sum() <= degrees:
            return 0.0
        return optimize.brentq(
            lambda value: (pearson / (1 + value * mu)).sum() - degrees,
            0.0,
            1e6,
            xtol=1e-15,
        )

    def score(log_size: float) -> float:
        size = np.exp(log_size)
        return float(
            (
                special.digamma(y + size)
                - special.digamma(size)
                + np.log(size / (size + mu))
                + (mu - y) / (size + mu)
            ).sum()
        )

    # The best size on a grid, then the root of the score beside it. Beyond
    # sizes of e^18 SciPy's likelihood and the score are lost in rounding,
    # so the Poisson likelihood stands for them.
    grid = np.linspace(-18.0, 18.0, 361)
    likelihood = [
        stats.nbinom.logpmf(y, np.exp(g), np.exp(g) / (np.exp(g) + mu)).sum()
        for g in grid
    ]
    best = int(np.argmax(likelihood))
    if best == len(grid) - 1 or likelihood[best] <= stats.poisson.logpmf(y, mu).sum():
        return 0.0
    root = optimize.brentq(score, grid[max(best - 1, 0)], grid[best + 1], xtol=1e-14)
    return float(np.exp(-root))


def dense_pair(
    y: np.ndarray, design: np.ndarray, treated: np.ndarray
) -> tuple[float, float, float]:
    """A pair's statistic, fold change and standard error, computed densely."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = sm.GLM(y, design, family=sm.families.Poisson()).fit(tol=1e-14).mu
    dispersion = size_root(y, mu, len(y) - design.shape[1])
    weights = mu / (1 + dispersion * mu)
    scores = (y - mu) / (1 + dispersion * mu)
    weighted = (weights[treated, None] * design[treated]).sum(axis=0)
    information = design.T @ (weights[:, None] * design)
    variance = weights[treated].sum() - weighted @ np.linalg.solve(
        information, weighted
    )
    observed = y[treated].sum()
    ratio = observed / mu[treated].sum()
    tangent = (mu[treated, None] * design[treated]).sum(axis=0) / mu[treated].sum()
    spread = tangent @ np.linalg.solve(information, tangent)
    if observed > 0:
        spread += (observed + dispersion * ratio**2 * (mu[treated] ** 2).sum()) / (
            observed**2
        )
    return (
        scores[treated].sum() / np.sqrt(variance),
        np.log2(ratio) if ratio > 0 else -np.inf,
        np.sqrt(spread) / np.log(2),
    )
