"""Transcription-factor to target importance by stochastic gradient boosting."""

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.arguments import integer_argument

__all__ = [
    "GRNBOOST2_PARAMETERS",
    "GrnParameters",
    "infer_grn",
]


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
        nthreads: Targets fitted concurrently. scikit-learn builds trees
            without holding the GIL, so threads scale.
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

    Links are sorted by decreasing importance, then by target and factor name.
    """
    # scikit-learn gathers predictors by column, faster in Fortran order.
    tf_matrix = np.asfortranarray(matrix[:, tf_columns], dtype=np.float32)
    own_position = {column: index for index, column in enumerate(tf_columns.tolist())}

    def fit(column: int) -> tuple[np.ndarray, np.ndarray] | None:
        target = np.asarray(matrix[:, column], dtype=np.float64)
        if np.all(target == target[0]):
            return None
        own = own_position.get(column)
        predictors, codes = tf_matrix, tf_columns
        if own is not None:
            if len(tf_columns) == 1:
                return None
            predictors = np.asfortranarray(np.delete(tf_matrix, own, axis=1))
            codes = np.delete(tf_columns, own)
        importances = _fit_target(predictors, target, parameters=parameters, seed=seed)
        keep = importances > 0
        return codes[keep], importances[keep]

    lock = threading.Lock()
    done = 0

    def run(column: int) -> tuple[np.ndarray, np.ndarray] | None:
        nonlocal done
        result = fit(column)
        if progress is not None:
            with lock:
                done += 1
                progress(done, len(target_columns))
        return result

    columns = target_columns.tolist()
    if nthreads == 1 or len(columns) == 1:
        results = [run(column) for column in columns]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            results = list(executor.map(run, columns))
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
