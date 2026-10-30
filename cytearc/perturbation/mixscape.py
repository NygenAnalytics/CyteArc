"""Mixscape: knocked-out cells and escapers among the cells of each target.

In a knockout screen, some cells that carry a targeting guide keep their
target's function, because the edit left the gene working or was never made.
Mixscape (Papalexi et al. 2021) separates the cells that respond (KO) from the
escapers (NP, not perturbed), as Seurat's ``RunMixscape`` does, on the
perturbation signatures of :mod:`cytearc.perturbation.signature`.

For each target, within each split, the signatures of the genes that differ
between the target's cells and the control cells are scaled over those cells,
and each cell is projected onto the difference between the mean of the cells
currently called KO and the mean of the control cells. A mixture of two normal
distributions fits the projections: the control component keeps the mean and
standard deviation of the control cells' projections, and the other component
and the mixing weights are fitted by expectation maximisation. A cell of the
target is called KO when its likelihood under the perturbed component exceeds
that under the control component, Seurat's rule, which leaves out the mixing
weights. The calls are repeated with the new KO cells until they stop changing.

These functions read values that a caller passes and never touch storage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import prange
from scipy.special import ndtr

from cytearc.utils.arguments import float_argument, integer_argument
from cytearc.utils.numba import njit

__all__ = [
    "MIXSCAPE_FEW_CELLS",
    "MIXSCAPE_FEW_DE_GENES",
    "MIXSCAPE_FITTED",
    "MIXSCAPE_NO_FIT",
    "MIXSCAPE_STATUSES",
    "MixscapeFit",
    "MixscapeResult",
    "classify_target",
    "constrained_mixture",
    "group_totals",
    "mixscape_settings",
    "rank_sum_p_values",
    "scale_like_seurat",
    "sorted_log_values",
]

#: Target statuses, stored as positions in this tuple.
MIXSCAPE_STATUSES = ("fitted", "few_de_genes", "few_cells", "no_fit")
#: The mixture classified the target's cells.
MIXSCAPE_FITTED = 0
#: Fewer DE genes than ``min_de_genes``: every cell is an escaper.
MIXSCAPE_FEW_DE_GENES = 1
#: Fewer KO cells than ``min_cells``: every cell is an escaper.
MIXSCAPE_FEW_CELLS = 2
#: The perturbed component collapsed: every cell is an escaper.
MIXSCAPE_NO_FIT = 3

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
# Smallest standard deviation of the fitted component before it counts as
# collapsed onto a few cells.
_MIN_SD = 1e-12
# The log of the largest float64, about 709.78, above which exp overflows.
_LOG_LARGEST = float(np.log(np.finfo(np.float64).max))


def mixscape_settings(
    min_de_genes: object,
    min_cells: object,
    max_iterations: object,
    logfc_threshold: object,
    min_fraction: object,
    p_value_cutoff: object,
    scale: object,
    perturbation_type: object,
    split_column: object,
) -> tuple[int, int, int, float, float, float]:
    """Check the settings of Mixscape, as ``mixscape`` takes them.

    The pipeline recipe checks the same settings before its run starts.
    Whether the split column exists is checked against the store.

    Returns:
        ``min_de_genes``, ``min_cells``, ``max_iterations``,
        ``logfc_threshold``, ``min_fraction`` and ``p_value_cutoff``, as
        Python numbers.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If a number is out of range, or ``perturbation_type`` is
            empty or holds a space.
    """
    de_floor = integer_argument(min_de_genes, "min_de_genes", minimum=1)
    cell_floor = integer_argument(min_cells, "min_cells", minimum=1)
    iterations = integer_argument(max_iterations, "max_iterations", minimum=1)
    fold = float_argument(logfc_threshold, "logfc_threshold")
    fraction = float_argument(min_fraction, "min_fraction")
    cutoff = float_argument(p_value_cutoff, "p_value_cutoff")
    if fold < 0:
        raise ValueError("logfc_threshold must be at least 0")
    if not 0 <= fraction <= 1:
        raise ValueError("min_fraction must be from 0 to 1")
    if not 0 < cutoff <= 1:
        raise ValueError("p_value_cutoff must be above 0 and at most 1")
    if not isinstance(scale, bool):
        raise TypeError("scale must be a boolean")
    if not isinstance(perturbation_type, str):
        raise TypeError("perturbation_type must be a string")
    if not perturbation_type or " " in perturbation_type:
        raise ValueError(
            "perturbation_type must be one word, such as 'KO', not "
            f"{perturbation_type!r}"
        )
    if split_column is not None and (
        not isinstance(split_column, str) or not split_column
    ):
        raise TypeError("split_column must be a non-empty string or None")
    return de_floor, cell_floor, iterations, fold, fraction, cutoff


@njit(cache=True, nogil=True, parallel=True)
def _group_totals(
    values: np.ndarray, groups: np.ndarray, sums: np.ndarray, detected: np.ndarray
) -> None:
    """Add each group's sum and count of positive values of each feature."""
    n_features, n_cells = values.shape
    for feature in prange(n_features):
        source = values[feature]
        for cell in range(n_cells):
            group = groups[cell]
            if group >= 0:
                value = source[cell]
                sums[group, feature] += value
                if value > 0:
                    detected[group, feature] += 1.0


def group_totals(
    values: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return each group's sum and number of positive values of each feature.

    Args:
        values: Values, features x cells.
        groups: Each cell's group, from 0 to ``n_groups`` - 1, or -1 for none.
        n_groups: The number of groups.

    Returns:
        The sums and the numbers of positive values, groups x features, as
        float64.

    Raises:
        ValueError: If ``groups`` does not give one group or -1 per cell.
    """
    held = np.ascontiguousarray(values, dtype=np.float64)
    codes = np.asarray(groups, dtype=np.int64)
    if (
        held.ndim != 2
        or codes.shape != held.shape[1:]
        or not np.all((codes >= -1) & (codes < n_groups))
    ):
        raise ValueError(
            "values must be features x cells and groups one group or -1 per cell"
        )
    sums = np.zeros((n_groups, len(held)))
    detected = np.zeros((n_groups, len(held)))
    _group_totals(held, codes, sums, detected)
    return sums, detected


@njit(cache=True, nogil=True, parallel=True)
def _sorted_log_values(
    values: np.ndarray,
    features: np.ndarray,
    cells: np.ndarray,
    out: np.ndarray,
    ties: np.ndarray,
) -> None:
    """Sort log1p of each feature's values over ``cells`` and sum its ties."""
    n_cells = len(cells)
    for column in prange(len(features)):
        source = values[features[column]]
        ordered = out[column]
        for index in range(n_cells):
            ordered[index] = np.log1p(source[cells[index]])
        ordered.sort()
        total = 0.0
        start = 0
        while start < n_cells:
            stop = start + 1
            while stop < n_cells and ordered[stop] == ordered[start]:
                stop += 1
            tied = float(stop - start)
            total += tied * tied * tied - tied
            start = stop
        ties[column] = total


def sorted_log_values(
    values: np.ndarray, features: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each feature's log1p values over some cells, sorted, and its ties.

    Sorted once, the control cells' values serve the rank test of every
    target that :func:`rank_sum_p_values` compares with them.

    Args:
        values: Normalized values, features x cells.
        features: The rows of ``values`` to sort.
        cells: The columns of ``values`` of the cells.

    Returns:
        The sorted values, one row per feature, and each feature's rank tie
        term, the sum of t^3 - t over its groups of t equal values.

    Raises:
        ValueError: If ``features`` or ``cells`` fall outside ``values``.
    """
    held = np.ascontiguousarray(values, dtype=np.float64)
    rows = np.asarray(features, dtype=np.int64)
    columns = np.asarray(cells, dtype=np.int64)
    if (
        held.ndim != 2
        or not np.all((rows >= 0) & (rows < len(held)))
        or not np.all((columns >= 0) & (columns < held.shape[1]))
    ):
        raise ValueError(
            "values must be features x cells, with features and cells inside it"
        )
    out = np.empty((len(rows), len(columns)))
    ties = np.empty(len(rows))
    _sorted_log_values(held, rows, columns, out, ties)
    return out, ties


@njit(cache=True, nogil=True, parallel=True)
def _rank_sum_z(
    values: np.ndarray,
    features: np.ndarray,
    cells: np.ndarray,
    slots: np.ndarray,
    controls: np.ndarray,
    control_ties: np.ndarray,
    out: np.ndarray,
) -> None:
    """Write the z statistic of each feature's two-sided rank-sum test.

    The ranks of the cells' values among the sorted control values and the
    tie term follow from the counts of values below and equal to each value,
    and the statistic follows scipy's ``mannwhitneyu``, operation by
    operation.
    """
    n_target = len(cells)
    n_control = controls.shape[1]
    n = n_target + n_control
    product = n_target * n_control
    for column in prange(len(features)):
        source = values[features[column]]
        target = np.empty(n_target)
        for index in range(n_target):
            target[index] = np.log1p(source[cells[index]])
        target.sort()
        reference = controls[slots[column]]
        rank_sum = 0.0
        ties = control_ties[slots[column]]
        start = 0
        while start < n_target:
            value = target[start]
            stop = start + 1
            while stop < n_target and target[stop] == value:
                stop += 1
            below = np.searchsorted(reference, value)
            equal = np.searchsorted(reference, value, side="right") - below
            in_target = stop - start
            # Every cell of a run of equal values takes their mean rank.
            rank_sum += in_target * (below + start + (equal + in_target + 1) / 2.0)
            tied = float(equal + in_target)
            shared = float(equal)
            ties += (tied * tied * tied - tied) - (shared * shared * shared - shared)
            start = stop
        u_target = rank_sum - n_target * (n_target + 1) / 2.0
        statistic = max(u_target, product - u_target)
        variance = product / 12.0 * ((n + 1) - ties / (n * (n - 1.0)))
        if variance > 0:
            out[column] = (statistic - product / 2.0 - 0.5) / np.sqrt(variance)
        else:
            # Every value is equal, which gives a p-value of 1.
            out[column] = -np.inf


def rank_sum_p_values(
    values: np.ndarray,
    features: np.ndarray,
    cells: np.ndarray,
    slots: np.ndarray,
    sorted_controls: np.ndarray,
    control_ties: np.ndarray,
) -> np.ndarray:
    """Return two-sided Wilcoxon rank-sum p-values of cells against controls.

    Each feature's log1p values over ``cells`` are ranked among the control
    cells' values, and the p-value comes from the normal approximation with
    the tie and continuity corrections. These are the p-values of scipy's
    ``mannwhitneyu(method="asymptotic", use_continuity=True)`` and of R's
    ``wilcox.test`` with ``correct=TRUE`` when it does not compute exact
    p-values, which Seurat's ``FindMarkers`` runs. A feature with one value
    in every cell gets 1, where R and recent scipy releases give NaN.

    Args:
        values: Normalized values, features x cells.
        features: The rows of ``values`` to test.
        cells: The columns of ``values`` of the cells to compare.
        slots: Each feature's row of ``sorted_controls``.
        sorted_controls: The control cells' sorted values, one row per slot,
            from :func:`sorted_log_values`.
        control_ties: Each slot's rank tie term, from
            :func:`sorted_log_values`.

    Returns:
        Each feature's p-value.

    Raises:
        ValueError: If the arrays do not align.
    """
    held = np.ascontiguousarray(values, dtype=np.float64)
    rows = np.asarray(features, dtype=np.int64)
    columns = np.asarray(cells, dtype=np.int64)
    places = np.asarray(slots, dtype=np.int64)
    controls = np.ascontiguousarray(sorted_controls, dtype=np.float64)
    ties = np.asarray(control_ties, dtype=np.float64)
    if (
        held.ndim != 2
        or controls.ndim != 2
        or places.shape != rows.shape
        or ties.shape != (len(controls),)
        or not np.all((rows >= 0) & (rows < len(held)))
        or not np.all((columns >= 0) & (columns < held.shape[1]))
        or not np.all((places >= 0) & (places < len(controls)))
    ):
        raise ValueError(
            "features, cells and slots must fall inside values and the sorted "
            "controls, with one tie term per slot"
        )
    z = np.empty(len(rows))
    _rank_sum_z(held, rows, columns, places, controls, ties, z)
    p_values: np.ndarray = np.clip(2.0 * ndtr(-z), 0.0, 1.0)
    return p_values


def scale_like_seurat(values: np.ndarray, scale_max: float = 10.0) -> np.ndarray:
    """Centre and scale each column over the rows, as Seurat's ``ScaleData``.

    Each column, a gene over cells, is centred on its mean and divided by its
    standard deviation with n - 1 degrees of freedom; a column without spread
    becomes 0. Seurat clips its scaled values above ``scale_max``. Its
    signatures are the negative of CyteArc's, so this clips below
    ``-scale_max``, which keeps the classes equal.

    Args:
        values: Signatures, cells x genes.
        scale_max: The bound of the clipped scaled values.

    Returns:
        The scaled values, cells x genes, as float64.
    """
    data = np.asarray(values, dtype=np.float64)
    mean = data.mean(axis=0)
    sd = data.std(axis=0, ddof=1) if len(data) > 1 else np.zeros(data.shape[1])
    spread = sd > 0
    scaled = np.zeros_like(data)
    scaled[:, spread] = (data[:, spread] - mean[spread]) / sd[spread]
    return np.maximum(scaled, -float(scale_max))


@dataclass(frozen=True, slots=True)
class MixtureFit:
    """A two-component normal mixture with a fixed control component."""

    mean: float
    sd: float
    weight: float
    log_likelihood: float
    steps: int
    collapsed: bool


def _log_normal(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    gap = (values - mean) / sd
    out: np.ndarray = -0.5 * gap * gap - np.log(sd) - _LOG_SQRT_2PI
    return out


def _knocked_out_posterior(log_ratio: np.ndarray) -> np.ndarray:
    """Return ``1 / (1 + exp(log_ratio))`` without overflowing exp.

    ``log_ratio`` is each cell's log likelihood under the control component
    minus that under the fitted one, so this is the probability of the fitted
    component under equal priors. Up to log(DBL_MAX), about 709.78, it is
    computed as written. Above it exp overflows, and the posterior equals
    ``exp(-log_ratio) / (1 + exp(-log_ratio))``, where ``exp(-log_ratio)`` is
    below 2**-1022 and leaves the 1 unchanged: the posterior is
    ``exp(-log_ratio)``, a subnormal number, or 0 above about 745.
    """
    posterior = np.empty_like(log_ratio)
    beyond = log_ratio > _LOG_LARGEST
    posterior[~beyond] = 1.0 / (1.0 + np.exp(log_ratio[~beyond]))
    posterior[beyond] = np.exp(-log_ratio[beyond])
    return posterior


def constrained_mixture(
    scores: np.ndarray,
    control_mean: float,
    control_sd: float,
    start_mean: float,
    start_sd: float,
    *,
    max_steps: int = 5000,
    tolerance: float = 1e-8,
) -> MixtureFit:
    """Fit a normal mixture whose first component is fixed, by EM.

    The first component keeps ``control_mean`` and ``control_sd``. The second
    component starts at ``start_mean`` and ``start_sd``, the weights at one
    half each, and expectation maximisation updates the weights and the second
    component until the log-likelihood rises by less than ``tolerance`` or
    after ``max_steps`` steps. This is the model that ``RunMixscape`` fits with
    mixtools' ``normalmixEM`` and its ``mean.constr`` and ``sd.constr``, and
    the two reach the same fit. Their steps differ, so under the same
    tolerance they stop at slightly different points, about 1e-5 apart in the
    fitted mean.

    Args:
        scores: The values to fit.
        control_mean: The fixed mean of the first component.
        control_sd: The fixed standard deviation of the first component.
        start_mean: The starting mean of the second component.
        start_sd: The starting standard deviation of the second component.
        max_steps: The most EM steps.
        tolerance: The rise in log-likelihood below which the fit stops.

    Returns:
        The second component, its weight, the log-likelihood and the steps
        taken, and whether the second component collapsed to a standard
        deviation of about 0, which ends the fit.
    """
    values = np.asarray(scores, dtype=np.float64)
    fixed = _log_normal(values, control_mean, control_sd)
    mean, sd, weight = float(start_mean), float(start_sd), 0.5
    previous = -np.inf
    log_likelihood = -np.inf
    steps = 0
    collapsed = not sd > _MIN_SD
    while not collapsed and steps < max_steps:
        first = np.log1p(-weight) + fixed
        second = np.log(weight) + _log_normal(values, mean, sd)
        total = np.logaddexp(first, second)
        log_likelihood = float(total.sum())
        if log_likelihood - previous <= tolerance:
            break
        previous = log_likelihood
        responsibility = np.exp(second - total)
        mass = float(responsibility.sum())
        steps += 1
        weight = mass / len(values)
        if not 0.0 < weight < 1.0:
            collapsed = True
            break
        mean = float(responsibility @ values) / mass
        gap = values - mean
        sd = float(np.sqrt((responsibility @ (gap * gap)) / mass))
        collapsed = not sd > _MIN_SD
    return MixtureFit(
        mean=mean,
        sd=sd,
        weight=weight,
        log_likelihood=log_likelihood,
        steps=steps,
        collapsed=collapsed,
    )


@dataclass(frozen=True, slots=True)
class MixscapeFit:
    """The classes of one target's cells within one split.

    Attributes:
        ko: Whether each of the target's cells is knocked out.
        posterior: Each cell's probability of being knocked out under equal
            priors, NaN without a fit.
        score: Each cell's projection onto the perturbation direction of the
            last iteration, NaN without a fit.
        status: The position of the target's status in
            :data:`MIXSCAPE_STATUSES`.
        iterations: The iterations run.
        converged: Whether the calls stopped changing.
        control_mean: The control component's mean.
        control_sd: The control component's standard deviation.
        perturbed_mean: The fitted component's mean.
        perturbed_sd: The fitted component's standard deviation.
        perturbed_weight: The fitted component's weight.
    """

    ko: np.ndarray
    posterior: np.ndarray
    score: np.ndarray
    status: int
    iterations: int
    converged: bool
    control_mean: float = np.nan
    control_sd: float = np.nan
    perturbed_mean: float = np.nan
    perturbed_sd: float = np.nan
    perturbed_weight: float = np.nan


def _escapers(n_cells: int, status: int, iterations: int = 0) -> MixscapeFit:
    """Every cell an escaper, with no fit."""
    return MixscapeFit(
        ko=np.zeros(n_cells, dtype=bool),
        posterior=np.full(n_cells, np.nan),
        score=np.full(n_cells, np.nan),
        status=status,
        iterations=iterations,
        converged=False,
    )


def classify_target(
    target_values: np.ndarray,
    control_values: np.ndarray,
    *,
    n_de_genes: int,
    min_de_genes: int,
    min_cells: int,
    max_iterations: int,
    scale: bool = True,
) -> MixscapeFit:
    """Call each of a target's cells knocked out or an escaper.

    Args:
        target_values: The signatures of the target's cells over its DE genes,
            cells x genes.
        control_values: The signatures of the control cells over the same
            genes.
        n_de_genes: The target's DE genes.
        min_de_genes: Fewer DE genes make every cell an escaper.
        min_cells: Fewer KO cells after an iteration make every cell an
            escaper.
        max_iterations: The most iterations.
        scale: Scale each gene over the target's and control cells first.

    Returns:
        The calls, posteriors and scores of the target's cells, and the fit.
    """
    n_target = len(target_values)
    if n_de_genes < min_de_genes:
        return _escapers(n_target, MIXSCAPE_FEW_DE_GENES)
    data = np.concatenate(
        [
            np.asarray(target_values, dtype=np.float64),
            np.asarray(control_values, dtype=np.float64),
        ]
    )
    if scale:
        data = scale_like_seurat(data)
    target, control = data[:n_target], data[n_target:]
    ko = np.ones(n_target, dtype=bool)
    control_center = control.mean(axis=0)
    posterior = np.full(n_target, np.nan)
    projections = np.empty(len(data))
    fit: MixtureFit | None = None
    control_mean = control_sd = np.nan
    converged = False
    iterations = 0
    while not converged and iterations < max_iterations:
        iterations += 1
        direction = target[ko].mean(axis=0) - control_center
        length = float(direction @ direction)
        if not length > 0:
            return _escapers(n_target, MIXSCAPE_NO_FIT, iterations)
        projections = data @ direction / length
        control_mean = float(projections[n_target:].mean())
        control_sd = float(projections[n_target:].std(ddof=1))
        started = projections[:n_target][ko]
        fit = constrained_mixture(
            projections,
            control_mean,
            control_sd,
            float(started.mean()),
            float(started.std(ddof=1)) if len(started) > 1 else 0.0,
        )
        if fit.collapsed or not control_sd > 0:
            return _escapers(n_target, MIXSCAPE_NO_FIT, iterations)
        scores = projections[:n_target]
        log_ratio = _log_normal(scores, control_mean, control_sd) - _log_normal(
            scores, fit.mean, fit.sd
        )
        posterior = _knocked_out_posterior(log_ratio)
        called = np.where(posterior > 0.5, True, np.where(posterior < 0.5, False, ko))
        if np.count_nonzero(called) < min_cells:
            return _escapers(n_target, MIXSCAPE_FEW_CELLS, iterations)
        converged = bool(np.array_equal(called, ko))
        ko = called
    assert fit is not None
    return MixscapeFit(
        ko=ko,
        posterior=posterior,
        score=projections[:n_target].copy(),
        status=MIXSCAPE_FITTED,
        iterations=iterations,
        converged=converged,
        control_mean=control_mean,
        control_sd=control_sd,
        perturbed_mean=fit.mean,
        perturbed_sd=fit.sd,
        perturbed_weight=fit.weight,
    )


@dataclass(frozen=True, slots=True)
class MixscapeResult:
    """Mixscape's classes of the cells of each target.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``, with ``mixscape_class``
            (such as ``"GENE1 KO"`` or ``"GENE1 NP"``, ``"control"``, or the
            cell's label when it carries no single target), ``global_class``
            (``"KO"``, ``"NP"``, ``"control"`` or that label), ``posterior``
            (the probability of being knocked out, NaN off target or without
            a fit), ``score`` (the projection onto the target's perturbation
            direction) and ``split``.
        targets: One row per target and split, with ``target``, ``split``,
            ``status``, ``n_cells``, ``n_ko``, ``n_de_genes``, ``iterations``,
            ``converged`` and the fitted mixture: ``control_mean``,
            ``control_sd``, ``perturbed_mean``, ``perturbed_sd`` and
            ``perturbed_weight``.
        de_genes: One row per target, split and DE gene, with ``target``,
            ``split`` and ``feature`` (the feature ID).
        perturbation_type: The word that names knocked-out cells, ``"KO"`` by
            default.
        split_column: The cell metadata column of the splits, or None.
    """

    cells: pd.DataFrame = field(repr=False)
    targets: pd.DataFrame = field(repr=False)
    de_genes: pd.DataFrame = field(repr=False)
    perturbation_type: str
    split_column: str | None
