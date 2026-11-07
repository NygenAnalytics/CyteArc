"""Transcription-factor to target importance by stochastic gradient boosting."""

import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.arguments import integer_argument
from ..utils.shutdown import shutdown_checkpoint

__all__ = [
    "GRNBOOST2_PARAMETERS",
    "GrnParameters",
    "grn_fit_bytes",
    "grn_process_bytes",
    "infer_grn",
]

# Targets are fitted in the calling process for at least this long before the
# rest go to worker processes, about as long as starting the workers takes, so
# a call whose fits end sooner starts none.
_IN_PROCESS_SECONDS = 2.0
# Each worker process holds at most this many targets' expression at once:
# the one it fits and the next.
_TARGETS_PER_PROCESS = 2


@dataclass(frozen=True, slots=True)
class GrnParameters:
    """Settings of the per-target gradient boosting regressor.

    The defaults are those of arboreto's GRNBoost2 (``SGBM_KWARGS`` with an
    out-of-bag early-stopping window of 25 rounds), the regressor that
    pySCENIC runs.

    Attributes:
        learningRate: Shrinkage of each boosting stage.
        maxStages: Upper bound on boosting stages; early stopping usually ends
            a fit much sooner.
        maxFeatures: Fraction of the transcription factors drawn at each split.
        subsample: Fraction of cells drawn for each stage. Must be below 1,
            because early stopping reads the out-of-bag improvement.
        earlyStopWindow: Stop once the mean out-of-bag improvement over this
            many most recent stages is negative.
        earlyStopping: How a stage's out-of-bag improvement is measured.
            ``"stage"`` compares the loss of the stage's out-of-bag cells
            before and after the stage, as GRNBoost2 was published and as
            arboreto computes it with scikit-learn before 1.3, which the
            pySCENIC 0.12.1 container uses. ``"consecutive"`` subtracts the
            out-of-bag losses of consecutive stages, which are measured on
            different cells, as arboreto computes it with scikit-learn 1.3 and
            later.
    """

    learningRate: float = 0.01
    maxStages: int = 5000
    maxFeatures: float = 0.1
    subsample: float = 0.9
    earlyStopWindow: int = 25
    earlyStopping: Literal["stage", "consecutive"] = "stage"

    def __post_init__(self) -> None:
        if not 0 < self.learningRate <= 1:
            raise ValueError("learningRate must be in (0, 1]")
        integer_argument(self.maxStages, "maxStages", minimum=1)
        if not 0 < self.maxFeatures <= 1:
            raise ValueError("maxFeatures must be in (0, 1]")
        if not 0 < self.subsample < 1:
            raise ValueError(
                "subsample must be in (0, 1) so out-of-bag early stopping applies"
            )
        integer_argument(self.earlyStopWindow, "earlyStopWindow", minimum=1)
        if self.earlyStopping not in ("stage", "consecutive"):
            raise ValueError("earlyStopping must be 'stage' or 'consecutive'")

    def regressor_kwargs(self) -> dict[str, float | int]:
        """Keyword arguments for ``sklearn.ensemble.GradientBoostingRegressor``."""
        return {
            "learning_rate": float(self.learningRate),
            "n_estimators": int(self.maxStages),
            "max_features": float(self.maxFeatures),
            "subsample": float(self.subsample),
        }


#: GRNBoost2 defaults (arboreto 0.1.6 ``SGBM_KWARGS`` and its early-stopping window).
GRNBOOST2_PARAMETERS = GrnParameters()


class _EarlyStopMonitor:
    """Stop when the mean out-of-bag improvement of the last window is negative.

    This is arboreto's ``EarlyStopMonitor``; scikit-learn calls it after each
    stage with the stage index, and a true result ends the fit. It reads
    scikit-learn's ``oob_improvement_``, whose meaning changed in
    scikit-learn 1.3 (the ``"consecutive"`` rule there).
    """

    def __init__(self, window_length: int) -> None:
        self.window_length = window_length

    def __call__(self, current_round: int, regressor: object, _: object) -> bool:
        if current_round < self.window_length - 1:
            return False
        low = max(0, current_round - self.window_length + 1)
        improvements = regressor.oob_improvement_[low : current_round + 1]  # type: ignore[attr-defined]
        return bool(np.mean(improvements) < 0)


class _StageEarlyStopMonitor:
    """GRNBoost2's early stopping with each stage's own out-of-bag improvement.

    A stage's improvement is the mean squared error of the cells left out of
    that stage before the stage minus after it, which is ``oob_improvement_``
    in scikit-learn before 1.3. scikit-learn passes the monitor its fitting
    loop's variables, from which the stage's mask and the predictions before
    and after the stage are read, so this rule holds on every version.
    """

    def __init__(self, window_length: int) -> None:
        self.window_length = window_length
        self.improvements: list[float] = []
        self.previous: np.ndarray | None = None

    def __call__(
        self, current_round: int, regressor: Any, fit_locals: Mapping[str, Any]
    ) -> bool:
        try:
            in_bag = np.asarray(fit_locals["sample_mask"], dtype=bool)
            current = np.asarray(fit_locals["raw_predictions"], dtype=np.float64)
            target = np.asarray(fit_locals["y"], dtype=np.float64)
        except KeyError as exc:
            raise RuntimeError(
                "scikit-learn no longer passes the boosting state that stage "
                "early stopping reads; use earlyStopping='consecutive'"
            ) from exc
        current = current.reshape(len(target), -1)[:, 0]
        if self.previous is None:
            constant = np.asarray(regressor.init_.constant_, dtype=np.float64)
            self.previous = np.full(len(target), float(constant.reshape(-1)[0]))
        out_of_bag = ~in_bag
        before = target[out_of_bag] - self.previous[out_of_bag]
        after = target[out_of_bag] - current[out_of_bag]
        self.improvements.append(
            float(np.mean(before * before) - np.mean(after * after))
        )
        self.previous = current.copy()
        if current_round < self.window_length - 1:
            return False
        window = self.improvements[current_round - self.window_length + 1 :]
        return bool(np.mean(window) < 0)


def _fit_target(
    predictors: np.ndarray,
    target: np.ndarray,
    *,
    parameters: GrnParameters,
    seed: int,
) -> np.ndarray:
    """Return arboreto's importances of each predictor column for one target.

    Importances are scikit-learn's normalized impurity importances multiplied
    by the number of fitted stages, as arboreto reports them.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    regressor = GradientBoostingRegressor(
        random_state=seed, **parameters.regressor_kwargs()
    )
    monitor: _EarlyStopMonitor | _StageEarlyStopMonitor = (
        _StageEarlyStopMonitor(parameters.earlyStopWindow)
        if parameters.earlyStopping == "stage"
        else _EarlyStopMonitor(parameters.earlyStopWindow)
    )
    regressor.fit(predictors, target, monitor=monitor)
    importances = np.asarray(regressor.feature_importances_, dtype=np.float64)
    return importances * len(regressor.estimators_)


def infer_grn(
    expression: np.ndarray,
    gene_names: Sequence[str],
    tf_names: Sequence[str],
    *,
    targets: Sequence[str] | None = None,
    seed: int,
    parameters: GrnParameters = GRNBOOST2_PARAMETERS,
    nthreads: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Score how well each transcription factor predicts each target gene.

    For every target gene, a stochastic gradient boosting regressor predicts
    its expression from the expression of the transcription factors, leaving
    the target out when it is a factor itself, and each factor's importance
    is recorded. This reproduces arboreto's GRNBoost2 as pySCENIC runs it:
    the same regressor settings, the same seed for every target, factors in
    expression column order, and importances scaled by the stages fitted. A
    target with constant expression has no links.

    Args:
        expression: Cells-by-genes matrix. SCENIC uses raw counts.
        gene_names: Names of the expression columns; must be unique.
        tf_names: Candidate regulators. Names absent from ``gene_names`` are
            ignored.
        targets: Target genes to model, by default every gene.
        seed: Random state given to every target's regressor.
        parameters: Regressor settings; GRNBoost2 defaults when omitted.
        nthreads: The most targets fitted at once. Targets are fitted in the
            calling process, and with ``nthreads`` above 1, those left after
            about two seconds of fitting go to ``nthreads`` worker processes,
            each with its own copy of the factors' expression. Worker processes
            start as new interpreters that import the main module, so a script
            that calls this keeps its own work under ``if __name__ ==
            "__main__":``. The links do not depend on ``nthreads``.
        progress: Optional ``callback(done, total)`` after each target.

    Returns:
        A DataFrame with ``TF``, ``target``, ``importance``, and
        ``correlation`` (Pearson, across cells) columns, one row per positive
        importance, sorted by decreasing
        importance, then by target and factor name for a deterministic order.
        pySCENIC splits modules into activating and repressing links by the
        correlation.
    """
    matrix = np.asarray(expression)
    if matrix.ndim != 2:
        raise ValueError("expression must be a cells-by-genes matrix")
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        raise ValueError("expression must be real-valued")
    # A sum is non-finite exactly when a value is, without a mask of the matrix.
    if not np.isfinite(matrix.sum(dtype=np.float64)):
        raise ValueError("expression must be finite")
    names = [str(name) for name in gene_names]
    if len(names) != matrix.shape[1]:
        raise ValueError("gene_names must align with the expression columns")
    if len(set(names)) != len(names):
        raise ValueError("gene_names must be unique")
    # scikit-learn accepts 32-bit unsigned random states only.
    seed = integer_argument(seed, "seed", minimum=0, maximum=2**32 - 1)
    nthreads = integer_argument(nthreads, "nthreads", minimum=1)
    if matrix.shape[0] < 2:
        raise ValueError("At least two cells are required")
    tf_set = {str(name) for name in tf_names}
    tf_columns = [index for index, name in enumerate(names) if name in tf_set]
    if not tf_columns:
        raise ValueError("None of tf_names is an expression column")
    position = {name: index for index, name in enumerate(names)}
    if targets is None:
        target_columns = list(range(len(names)))
    else:
        missing = [str(name) for name in targets if str(name) not in position]
        if missing:
            raise ValueError(f"Targets are not expression columns: {missing[:5]}")
        target_columns = list(dict.fromkeys(position[str(name)] for name in targets))
    tf_codes, target_codes, importance = _grn_links(
        matrix,
        names,
        np.asarray(tf_columns, dtype=np.int64),
        np.asarray(target_columns, dtype=np.int64),
        seed=seed,
        parameters=parameters,
        nthreads=nthreads,
        progress=progress,
    )
    labels = np.asarray(names, dtype=object)
    return pd.DataFrame(
        {
            "TF": labels[tf_codes],
            "target": labels[target_codes],
            "importance": importance,
            "correlation": _pair_correlations(matrix, tf_codes, target_codes),
        }
    ).astype({"TF": "str", "target": "str"})


def grn_fit_bytes(n_cells: int, n_regulators: int, max_stages: int) -> int:
    """Estimate the bytes that one target's fit holds beside its process's factors.

    A factor that is also a target is fitted on a copy of the factors without
    its own column. The fit holds a few float64 vectors of the cells, and each
    of up to ``max_stages`` fitted stages keeps its tree, about 2 KiB and 15
    bytes per factor with scikit-learn 1.9, which this rounds up.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    regulators = integer_argument(n_regulators, "n_regulators", minimum=1)
    stages = integer_argument(max_stages, "max_stages", minimum=1)
    return 4 * cells * regulators + 64 * cells + stages * (2048 + 16 * regulators)


def grn_process_bytes(n_cells: int, n_regulators: int, max_stages: int) -> int:
    """Estimate the bytes that one worker process of GRN inference holds.

    A worker holds its own copy of the float32 factors, the expression of the
    targets that it holds at once, and one fit (:func:`grn_fit_bytes`).

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    fit = grn_fit_bytes(n_cells, n_regulators, max_stages)
    return 4 * n_cells * n_regulators + 8 * n_cells * _TARGETS_PER_PROCESS + fit


def _fit_links(
    tf_matrix: np.ndarray,
    tf_columns: np.ndarray,
    own: int | None,
    target: np.ndarray,
    parameters: GrnParameters,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the factor columns and importances of one target's links.

    ``own`` is the target's position among the factors, or None when it is
    not one, and the factor never predicts itself.
    """
    predictors, codes = tf_matrix, tf_columns
    if own is not None:
        predictors = np.asfortranarray(np.delete(tf_matrix, own, axis=1))
        codes = np.delete(tf_columns, own)
    importances = _fit_target(predictors, target, parameters=parameters, seed=seed)
    keep = importances > 0
    return codes[keep], importances[keep]


# The factors, their columns, the settings and the seed of a worker process,
# which _start_worker sets once.
_worker_factors: tuple[np.ndarray, np.ndarray, GrnParameters, int] | None = None


def _start_worker(
    tf_matrix: np.ndarray,
    tf_columns: np.ndarray,
    parameters: GrnParameters,
    seed: int,
) -> None:
    """Keep a worker process's factors and hold its BLAS and OpenMP to one thread."""
    from ..utils.compute import enter_thread_limit

    global _worker_factors
    enter_thread_limit(1)
    _worker_factors = (tf_matrix, tf_columns, parameters, seed)


def _fit_in_worker(
    task: tuple[int, int | None, np.ndarray],
) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
    """Fit one target in a worker process and return its position and links."""
    if _worker_factors is None:
        raise RuntimeError("A GRN worker process fitted a target before it started")
    position, own, target = task
    tf_matrix, tf_columns, parameters, seed = _worker_factors
    return position, _fit_links(tf_matrix, tf_columns, own, target, parameters, seed)


def _fit_in_processes(
    tasks: Iterator[tuple[int, int | None, np.ndarray]],
    n_tasks: int,
    tf_matrix: np.ndarray,
    tf_columns: np.ndarray,
    *,
    parameters: GrnParameters,
    seed: int,
    processes: int,
    record: Callable[[int, tuple[np.ndarray, np.ndarray]], None],
) -> None:
    """Fit targets on a pool of ``processes`` new worker processes.

    Each worker receives the factors once and then the expression of each
    target it fits, at most ``_TARGETS_PER_PROCESS`` targets per worker at a
    time. ``record`` takes each target's position and links as they arrive.
    A worker's exception reaches the caller, and so does a shutdown request,
    after the targets being fitted finish and before any other starts.
    """
    from multiprocessing import get_context

    # A forked child of a process that runs threads can deadlock, so workers
    # start as new interpreters.
    executor = ProcessPoolExecutor(
        max_workers=min(processes, n_tasks),
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(tf_matrix, tf_columns, parameters, seed),
    )
    outstanding: set[Future[tuple[int, tuple[np.ndarray, np.ndarray]]]] = set()
    try:
        for task in tasks:
            if len(outstanding) >= processes * _TARGETS_PER_PROCESS:
                finished, outstanding = wait(outstanding, return_when=FIRST_COMPLETED)
                for future in finished:
                    record(*future.result())
                shutdown_checkpoint()
            outstanding.add(executor.submit(_fit_in_worker, task))
        while outstanding:
            finished, outstanding = wait(outstanding, return_when=FIRST_COMPLETED)
            for future in finished:
                record(*future.result())
            shutdown_checkpoint()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _grn_links(
    matrix: np.ndarray,
    names: Sequence[str],
    tf_columns: np.ndarray,
    target_columns: np.ndarray,
    *,
    seed: int,
    parameters: GrnParameters,
    nthreads: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the factor and target columns and importances of every link.

    Targets are fitted in the calling process, and with ``nthreads`` above 1,
    those left once fitting has taken ``_IN_PROCESS_SECONDS`` are fitted on
    up to ``nthreads`` worker processes (:func:`_fit_in_processes`). Every
    target's regressor has the same seed and sees the same values in any
    process, so the links do not depend on ``nthreads``. Links are sorted by
    decreasing importance, then by target and factor name.
    """
    # scikit-learn gathers predictors by column, faster in Fortran order.
    tf_matrix = np.asfortranarray(matrix[:, tf_columns], dtype=np.float32)
    own_position = {column: index for index, column in enumerate(tf_columns.tolist())}
    columns = target_columns.tolist()
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(columns)
    done = 0

    def record(position: int, links: tuple[np.ndarray, np.ndarray] | None) -> None:
        nonlocal done
        results[position] = links
        done += 1
        if progress is not None:
            progress(done, len(columns))

    def target_of(position: int) -> np.ndarray:
        return np.asarray(matrix[:, columns[position]], dtype=np.float64)

    pending: deque[int] = deque()
    for position, column in enumerate(columns):
        target = target_of(position)
        # A constant target and a lone factor, which no other factor can
        # predict, have no links.
        if np.all(target == target[0]) or (
            column in own_position and len(tf_columns) == 1
        ):
            record(position, None)
        else:
            pending.append(position)
    started = time.monotonic()
    while pending and (
        nthreads == 1
        or len(pending) == 1
        or time.monotonic() - started < _IN_PROCESS_SECONDS
    ):
        shutdown_checkpoint()
        position = pending.popleft()
        own = own_position.get(columns[position])
        record(
            position,
            _fit_links(
                tf_matrix, tf_columns, own, target_of(position), parameters, seed
            ),
        )
    if pending:
        _fit_in_processes(
            (
                (position, own_position.get(columns[position]), target_of(position))
                for position in pending
            ),
            len(pending),
            tf_matrix,
            tf_columns,
            parameters=parameters,
            seed=seed,
            processes=nthreads,
            record=record,
        )
    fitted = [(column, result) for column, result in zip(columns, results) if result]
    if not fitted:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))
    tf_codes = np.concatenate([codes for _, (codes, _) in fitted])
    importance = np.concatenate([values for _, (_, values) in fitted])
    target_codes = np.repeat(
        [column for column, _ in fitted], [len(values) for _, (_, values) in fitted]
    )
    name_rank = np.argsort(np.argsort(np.asarray(names, dtype=object), kind="stable"))
    order = np.lexsort((name_rank[tf_codes], name_rank[target_codes], -importance))
    return tf_codes[order], target_codes[order], importance[order]


def _pair_correlations(
    matrix: np.ndarray, tf_index: np.ndarray, target_index: np.ndarray
) -> np.ndarray:
    """Pearson correlation of each pair of expression columns."""
    result = np.full(len(tf_index), np.nan, dtype=np.float64)
    if len(tf_index) == 0:
        return result

    def standardized(columns: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix[:, columns], dtype=np.float64)
        values = values - values.mean(axis=0)
        norms = np.sqrt((values * values).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            # A constant column has no correlation; NaN propagates to its pairs.
            return values / np.where(norms > 0, norms, np.nan)

    tfs, tf_local = np.unique(tf_index, return_inverse=True)
    targets, target_local = np.unique(target_index, return_inverse=True)
    tf_values = standardized(tfs)
    # Bound the standardized target block to the size of the factor block.
    chunk = max(1, min(len(targets), max(len(tfs), 256)))
    for start in range(0, len(targets), chunk):
        stop = min(start + chunk, len(targets))
        block = tf_values.T @ standardized(targets[start:stop])
        rows = np.flatnonzero((target_local >= start) & (target_local < stop))
        values = block[tf_local[rows], target_local[rows] - start]
        result[rows] = np.clip(values, -1.0, 1.0)
    return result
