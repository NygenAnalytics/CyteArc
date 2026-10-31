"""Shared helpers for the tests of ``cytearc.perturbation``."""

import re
import shutil
import tracemalloc
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import optimize, special, stats

from tests.storage_helpers import write_count_store

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore
    from cytearc.storage.refs import ArtifactRef


def open_store(path: str | Path, **options: Any) -> "DataStore":
    """Open a screen store with RNA as its default assay, on one thread by default."""
    # Imported here, so that the domain tests do not import the datastore.
    from cytearc.datastore.datastore import DataStore

    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        **{"nthreads": 1, **options},
    )


def write_screen(
    path: Path,
    rna: np.ndarray,
    guides: np.ndarray,
    *,
    targets: Sequence[str | None] | None = None,
    target_ids: Sequence[str] | None = None,
    cell_columns: Mapping[str, np.ndarray] | None = None,
    dtype: str = "uint16",
    **options: Any,
) -> "DataStore":
    """Write RNA and guide counts, cells x features, to a store and open it.

    The guide assay is named CRISPR. ``targets`` and ``target_ids`` fill its
    ``target_gene_name`` and ``target_gene_id`` columns, and ``cell_columns``
    adds cell columns.
    """
    write_count_store(str(path), {"RNA": rna, "CRISPR": guides}, dtype)
    store = open_store(path, **options)
    features = store._get_assay("CRISPR").feats
    if targets is not None:
        features.insert("target_gene_name", np.asarray(targets, dtype=object))
    if target_ids is not None:
        features.insert("target_gene_id", np.asarray(target_ids, dtype=object))
    for name, values in (cell_columns or {}).items():
        store.cells.insert(name, values)
    return store


def named_budget(refusal: BaseException) -> int:
    """Return the memory budget that a refusal names in its closing ``raise mem_budget to at least N bytes``.

    A refusal that says the call holds about M bytes must name M. A refusal
    that closes with ``Raise mem_budget.`` names no budget, as Mixscape's
    refusal of a classification batch does, and asks for the M bytes the call
    holds.
    """
    message = str(refusal)
    held = re.search(r"holds? about (\d+) bytes", message)
    if message.endswith(" Raise mem_budget."):
        assert held is not None, f"The refusal names no budget: {message}"
        return int(held.group(1))
    named = re.search(
        r"[Rr]aise mem_budget to at least (\d+) bytes(?:, or [^.]*)?\.$", message
    )
    assert named is not None, f"The refusal does not end naming a budget: {message}"
    assert held is None or held.group(1) == named.group(1), (
        f"The refusal names a budget other than what it holds: {message}"
    )
    return int(named.group(1))


def traced_peak(call: Callable[[], Any], budget: int) -> tuple[Any, int]:
    """Return the result of ``call()`` and the most bytes it held beside what was held before it.

    tracemalloc counts every allocation in the process while the call runs,
    other threads' included, so a peak above ``budget`` is traced again on a
    second run, and the smaller peak is returned: a call that holds more than
    its budget does so in both runs.
    """

    def trace() -> tuple[Any, int]:
        tracemalloc.start()
        try:
            start = tracemalloc.get_traced_memory()[0]
            result = call()
            return result, tracemalloc.get_traced_memory()[1] - start
        finally:
            tracemalloc.stop()

    result, peak = trace()
    if peak > budget:
        result, again = trace()
        peak = min(peak, again)
    return result, peak


def budget_that_runs(
    path: str | Path, call: Callable[["DataStore"], Any], **options: Any
) -> tuple[int, list[str], Any, int]:
    """Follow the refusals of ``call`` from a budget of one byte to the budget that runs it.

    ``call`` runs an operation on the store at ``path``, opened with each
    budget and ``options``. Each refusal must name a budget above the one it
    refused, which the next attempt opens the store with, and a check that a
    later step makes may refuse that budget and name more. Each attempt is
    traced as :func:`traced_peak` traces it, and ten refusals fail the search.

    Returns:
        The budget that runs the call, the messages of the refusals on the
        way, and the result and traced peak of the call at that budget.
    """
    budget, refusals = 1, []
    while len(refusals) < 10:
        store = open_store(path, mem_budget=budget, **options)
        try:
            result, peak = traced_peak(partial(call, store), budget)
        except MemoryError as refusal:
            named = named_budget(refusal)
            assert named > budget, f"{refusal} names no budget above {budget}"
            refusals.append(str(refusal))
            budget = named
        else:
            return budget, refusals, result, peak
    raise AssertionError(f"Ten refusals named no budget that runs: {refusals}")


def check_budget(
    path: str | Path,
    call: Callable[["DataStore"], Any],
    *,
    check_writes: bool = True,
    **options: Any,
) -> tuple[int, list[str], Any]:
    """Check that ``call`` runs within the budget that its refusals name.

    :func:`budget_that_runs` finds that budget and traces the call at it. The
    traced peak must be within the budget, and one byte less must be refused
    naming the same budget. With ``check_writes``, the refused call must leave
    what the store holds unchanged. ``call`` should compute again rather than
    reuse.

    Returns:
        The budget, the messages of the refusals on the way, and the result
        of the call at that budget.
    """
    budget, refusals, result, peak = budget_that_runs(path, call, **options)
    assert peak <= budget, f"The call held {peak} bytes within a budget of {budget}"
    saved = written(open_store(path)) if check_writes else None
    with pytest.raises(MemoryError) as refused:
        call(open_store(path, mem_budget=budget - 1, **options))
    assert named_budget(refused.value) == budget
    if check_writes:
        assert written(open_store(path)) == saved
    return budget, refusals, result


def fixed_labels(
    store: "DataStore", controls: Sequence[str], *, min_umi: int = 10
) -> "ArtifactRef":
    """Call every guide with at least ``min_umi`` UMIs, and label the cells."""
    calls = store.perturbation.call_guides(
        store.select_measured_cells("CRISPR"), method="fixed", min_umi=min_umi
    )
    return store.perturbation.labels(calls, controls=list(controls))


def copy_store(template: Path, folder: Path, **options: Any) -> "DataStore":
    """Copy a template store into ``folder`` and open the copy."""
    target = folder / template.name
    shutil.copytree(template, target)
    return open_store(target, **options)


def written(store: "DataStore") -> tuple[Any, ...]:
    """Return what a call can write: run records, and every artifact of the store."""
    return (
        [run.run_id for run in store.pipeline.list_runs(limit=1000)],
        store.artifacts.list(scope="datastore"),
        *(store.artifacts.list(from_assay=assay) for assay in store.assay_names),
    )


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


def histogram_of(
    histograms: Any, sample: int, guide: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return one guide's distinct nonzero counts in one sample, and their cells.

    The counts are in ascending order, as histogram entry
    ``sample * n_guides + guide`` lists them.
    """
    entry = sample * histograms.n_guides + guide
    start, stop = histograms.offsets[entry], histograms.offsets[entry + 1]
    return histograms.counts[start:stop], histograms.cells[start:stop]


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


def memory_screen(path: Path) -> SimpleNamespace:
    """Write a screen whose counts and signatures outweigh what every call reads besides.

    4,000 cells in 400 genes: 16 targets of 175 cells each, a target lowering
    its own gene and 10 others to a tenth in 80% of its cells, and 1,200
    control cells. The coordinates hold 20 dimensions, and the features are
    the first 200 genes. A store keeps its cell and feature columns in
    chunks of 100,000 rows, which Zarr decodes whole for any row, so a
    smaller screen's budgets would not cover these reads.

    Returns:
        The store's path, labels, coordinates and features.
    """
    rng = np.random.default_rng(5)
    n_cells, n_genes, n_targets, n_controls = 4000, 400, 16, 1200
    carried = np.r_[
        np.arange(n_cells - n_controls) % n_targets, np.full(n_controls, n_targets)
    ]
    base = rng.gamma(2.0, 3.0, size=n_genes)
    rates = np.outer(rng.lognormal(0, 0.2, size=n_cells), base)
    for target in range(n_targets):
        cells = np.flatnonzero(carried == target)
        genes = np.r_[target, rng.choice(np.arange(n_targets, n_genes), 10, False)]
        rates[np.ix_(cells[: int(0.8 * len(cells))], genes)] *= 0.1
    guides = np.zeros((n_cells, n_targets + 1), dtype=np.uint16)
    guides[np.arange(n_cells), carried] = 20
    store = write_screen(
        path,
        rng.poisson(rates).astype(np.uint16),
        guides,
        targets=[f"RNA{target}" for target in range(n_targets)] + ["NT"],
        cell_columns={
            "run": np.where(np.arange(n_cells) % 3 == 0, "run1", "run2").astype(object)
        },
        nthreads=2,
    )
    labels = fixed_labels(store, ["NT"])
    coordinates = store.reduction.import_coordinates(
        rng.normal(size=(n_cells, 20)),
        cell_selection=store.select_measured_cells("RNA"),
        name="X",
    )
    features = store.features.snapshot(feature_indexes=np.arange(200))
    return SimpleNamespace(
        path=path, labels=labels, coordinates=coordinates, features=features
    )
