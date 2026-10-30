"""Test perturbations of high-MOI screens against all other cells, with calibrated p-values.

In a high-MOI screen each cell carries many guides, so few cells carry a target
alone and almost none carry control guides only. SCEPTRE's high-MOI test
(Barry et al. 2021, 2024) compares the cells that carry any guide of a group,
its treatment cells, with every other cell, the group's complement. Every
group's treatment and comparison cells are then all cells, so each gene's
negative binomial regression on the cell covariates is fitted once, on all
cells, and every group paired with the gene shares the score and the weight
that the fit gives each cell. A pair's statistic is the treatment cells'
summed score, standardized: the score statistic of the low-MOI test.

The p-value comes from random sets of cells, in the rounds of the low-MOI
test. Conditional resampling, SCEPTRE's conditional randomization test, fits a
logistic regression of each group's indicator on the covariates, and each
random set takes every cell independently with its fitted probability, so the
sets follow the covariates, such as sequencing depth, that drive both the
detection of guides and the expression of genes. A conditional set that holds
no cell or every cell has no statistic, so it is drawn again. A group's
treatment cells are never none or all of the cells, so sets drawn under that
condition keep the p-values valid. Permutation resampling draws as many cells
as the group holds, uniformly from all cells.

Genes are tested a band at a time. A band's scores and weights are held cell
by cell, so each cell of a set adds one row of the band. A group's random sets
are keyed by the seed, the group, the round and the set; a gene's fit is the
same in any band; each set's cells are summed in the order the set lists
them; and a group's probabilities come from the same arithmetic in every
band. The results therefore depend neither on threads nor on bands.

These functions hold counts in memory and never touch storage.
"""

import math
from dataclasses import dataclass

import numba
import numpy as np
from numba import prange

from cytearc.perturbation.glm import (
    count_array,
    fit_logistic,
    fit_negative_binomial,
    logistic_probabilities,
    score_statistic,
    set_test,
)
from cytearc.perturbation.p_values import (
    RESAMPLING,
    STATISTICS_BYTES,
    PairResults,
    staged_p_values,
)
from cytearc.perturbation.random_sets import (
    N_BUCKETS,
    draw_bernoulli_set,
    draw_uniform_set,
    probability_buckets,
    resampling_keys,
    set_key,
)
from cytearc.utils.numba import njit

__all__ = [
    "GATHER_BYTES",
    "ComplementGroups",
    "GeneFits",
    "cells_by_code",
    "complement_tests",
    "complement_thread_bytes",
    "fit_detection_models",
    "nonzero_group_counts",
]

#: The most that each thread's copy of a group's genes takes, by default.
GATHER_BYTES = 1 << 26


@dataclass(frozen=True, slots=True)
class ComplementGroups:
    """The treatment cells of each group, each compared with every other cell.

    Groups may share cells, as the targets of a cell with several guides do.

    Attributes:
        members: The positions of each group's treatment cells, ascending and
            distinct, group after group.
        offsets: Where each group's positions start in ``members``, and the
            end of the last group.
        codes: A distinct, non-negative code for each group, which keys its
            random sets through
            :func:`~cytearc.perturbation.random_sets.resampling_keys`.
    """

    members: np.ndarray
    offsets: np.ndarray
    codes: np.ndarray


@dataclass(frozen=True, slots=True)
class GeneFits:
    """Each gene's null fit on all cells, and the score and weight it gives each cell.

    With mean mu and dispersion phi = 1 / size, a cell's score is
    (y - mu) / (1 + phi mu) and its weight mu / (1 + phi mu).

    Attributes:
        coefficients: The Poisson coefficients, genes x covariates, NaN for a
            gene without counts.
        dispersion: Each gene's dispersion, NaN without a fit.
        scores: Each cell's score, cells x genes, 0 without a fit.
        weights: Each cell's weight, cells x genes, 0 without a fit.
        factors: The lower Cholesky factor of each fit's information, genes x
            covariates x covariates, NaN without a fit.
    """

    coefficients: np.ndarray
    dispersion: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    factors: np.ndarray


@dataclass(frozen=True, slots=True)
class _Band:
    """What the random sets of a band's pairs read; see :func:`complement_tests`."""

    covariates: np.ndarray
    fits: GeneFits
    sizes: np.ndarray
    codes: np.ndarray
    models: np.ndarray
    conditional: bool
    pair_groups: np.ndarray
    pair_columns: np.ndarray
    z: np.ndarray
    seed: int
    gather_columns: int


@dataclass(frozen=True, slots=True)
class _Units:
    """The pieces of work of a round: each group's genes, read in place or copied.

    A unit reads ``width`` columns from ``first`` of the band when it is not
    ``compact``, and otherwise copies its ``columns`` together. Its pairs are
    those from ``pair_offsets[u]``, and ``local`` gives each pair's column
    within its unit.
    """

    group_offsets: np.ndarray
    first: np.ndarray
    width: np.ndarray
    compact: np.ndarray
    columns: np.ndarray
    column_offsets: np.ndarray
    pair_offsets: np.ndarray
    local: np.ndarray


def _design(covariates: np.ndarray) -> np.ndarray:
    """Check the covariates of the cells and return them as C-contiguous float64."""
    design = np.ascontiguousarray(covariates, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError("covariates must be cells x covariates")
    n_cells, n_covariates = design.shape
    if n_cells <= n_covariates:
        raise ValueError(
            f"The {n_cells} cells must outnumber the {n_covariates} covariates"
        )
    if not (design[:, 0] == 1.0).all():
        raise ValueError("The first covariate must be the intercept, 1 for every cell")
    finite = np.isfinite(design).all(axis=1)
    if not finite.all():
        raise ValueError(
            "The covariates must be finite, but those of "
            f"{np.count_nonzero(~finite)} of the {n_cells} cells are not"
        )
    return design


def _group_arrays(
    groups: ComplementGroups, n_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check groups of treatment cells among ``n_cells`` cells and return their arrays."""
    members = np.ascontiguousarray(groups.members, dtype=np.int64)
    offsets = np.ascontiguousarray(groups.offsets, dtype=np.int64)
    codes = np.ascontiguousarray(groups.codes, dtype=np.int64)
    if (
        members.ndim != 1
        or offsets.ndim != 1
        or len(offsets) < 1
        or offsets[0] != 0
        or offsets[-1] != len(members)
    ):
        raise ValueError(
            "groups.offsets must start at 0 and end at the number of members"
        )
    sizes = np.diff(offsets)
    if (sizes < 1).any() or (sizes >= n_cells).any():
        raise ValueError(
            f"Every group needs at least one of the {n_cells} cells and must leave "
            "at least one out"
        )
    n_groups = len(sizes)
    if codes.shape != (n_groups,) or (codes < 0).any():
        raise ValueError("groups.codes needs a non-negative code for each group")
    if len(np.unique(codes)) != n_groups:
        raise ValueError("groups.codes must be distinct, since they key the sets")
    if len(members) and (members.min() < 0 or members.max() >= n_cells):
        raise ValueError(f"groups.members must be positions among the {n_cells} cells")
    # Within a group, each member follows the one before it.
    steps = np.diff(members) > 0
    steps[offsets[1:-1] - 1] = True
    if not steps.all():
        raise ValueError("Each group's members must be distinct and ascending")
    return members, offsets, codes


@njit(cache=True, nogil=True, parallel=True)
def _fit_detection(
    covariates: np.ndarray, members: np.ndarray, offsets: np.ndarray, models: np.ndarray
) -> None:
    """Fit each group's logistic regression; see :func:`fit_detection_models`."""
    n, p = covariates.shape
    for group in prange(len(offsets) - 1):
        size = offsets[group + 1] - offsets[group]
        y = np.zeros(n)
        for k in range(offsets[group], offsets[group + 1]):
            y[members[k]] = 1.0
        coefficients = np.zeros(p)
        coefficients[0] = math.log(size / (n - size))
        fit_logistic(y, covariates, coefficients, np.empty(n), np.empty(n))
        models[group, :] = coefficients


def fit_detection_models(
    covariates: np.ndarray, groups: ComplementGroups
) -> np.ndarray:
    """Fit each group's logistic regression of its indicator on the covariates.

    Each cell's probability of being one of the group's treatment cells,
    1 / (1 + exp(-z' delta)) for its covariates z, is what the conditional
    randomization test draws the cell into a random set with. The fit takes
    Newton's steps from the log odds of the group's share of the cells,
    halving those that lower the likelihood, until a step would gain less
    than 1e-12 of the log-likelihood, which it then takes whole, or after 50
    steps; a direction that the cells do not determine keeps its coefficient
    of 0. Under separation, such as a sample without treatment cells, those
    cells' probabilities fall towards 0 and the cells all but never enter a
    set, the limit of the fit. The groups are fitted in parallel, and each
    thread holds 24 bytes per cell while it fits a group.

    Args:
        covariates: The covariates of the cells, cells x covariates, the
            intercept first.
        groups: The groups of treatment cells.

    Returns:
        Each group's coefficients, groups x covariates.

    Raises:
        ValueError: If the covariates or the groups are malformed, or a
            covariate is not finite.
    """
    design = _design(covariates)
    members, offsets, _ = _group_arrays(groups, design.shape[0])
    models = np.empty((len(offsets) - 1, design.shape[1]))
    _fit_detection(design, members, offsets, models)
    return models


@njit(cache=True, nogil=True, parallel=True)
def _nonzero_group_counts(
    values: np.ndarray,
    cell_groups: np.ndarray,
    cell_offsets: np.ndarray,
    in_groups: np.ndarray,
    totals: np.ndarray,
) -> None:
    """Count each group's cells with counts of each feature; see :func:`nonzero_group_counts`."""
    for feature in prange(values.shape[0]):
        total = 0
        for cell in range(values.shape[1]):
            if values[feature, cell] > 0:
                total += 1
                for k in range(cell_offsets[cell], cell_offsets[cell + 1]):
                    in_groups[cell_groups[k], feature] += 1
        totals[feature] = total


def nonzero_group_counts(
    values: np.ndarray, groups: ComplementGroups
) -> tuple[np.ndarray, np.ndarray]:
    """Return how many cells of each group, and of all cells, hold counts of each feature.

    A group's complement holds the cells with counts that the group does not:
    the total less the group's. A cell counts once in every group it belongs
    to, and the work follows the cells with counts and their groups. The
    counts are 64-bit integers, 8 bytes per group and feature, and finding
    each cell's groups takes 16 bytes per cell and 24 per member.

    Args:
        values: Counts, features x cells, held as float32.
        groups: The groups of treatment cells.

    Returns:
        The cells with counts of each group, groups x features, and of all
        cells, one per feature.

    Raises:
        ValueError: If the counts or the groups are malformed.
    """
    counts = count_array(values)
    if counts.ndim != 2:
        raise ValueError("values must be features x cells")
    n_features, n_cells = counts.shape
    members, offsets, _ = _group_arrays(groups, n_cells)
    # Each cell's groups: the groups of the members sorted by cell, and where
    # each cell's groups start among them.
    group_of = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    cell_groups = np.ascontiguousarray(group_of[np.argsort(members, kind="stable")])
    cell_offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(members, minlength=n_cells), out=cell_offsets[1:])
    in_groups = np.zeros((len(offsets) - 1, n_features), dtype=np.int64)
    totals = np.zeros(n_features, dtype=np.int64)
    _nonzero_group_counts(counts, cell_groups, cell_offsets, in_groups, totals)
    return in_groups, totals


@njit(cache=True, nogil=True, parallel=True)
def _fit_band(
    values: np.ndarray,
    rows: np.ndarray,
    covariates: np.ndarray,
    members: np.ndarray,
    offsets: np.ndarray,
    pair_groups: np.ndarray,
    pair_offsets: np.ndarray,
    pair_order: np.ndarray,
    coefficients: np.ndarray,
    dispersion: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    factors: np.ndarray,
    z: np.ndarray,
    fold: np.ndarray,
    se: np.ndarray,
) -> None:
    """Fit the genes of ``rows`` on all cells and test their pairs' treatment cells.

    Column k of the fits is gene ``rows[k]``, and ``pair_order`` lists the
    pairs of column k from ``pair_offsets[k]``. A pair's treatment cells add
    up in ascending order, as a random set's cells do in its order.
    """
    n, p = covariates.shape
    for k in prange(len(rows)):
        row = rows[k]
        y = np.empty(n)
        total = 0.0
        for i in range(n):
            y[i] = values[row, i]
            total += y[i]
        if total <= 0:
            coefficients[k, :] = np.nan
            dispersion[k] = np.nan
            factors[k, :, :] = np.nan
            for t in range(pair_offsets[k], pair_offsets[k + 1]):
                pair = pair_order[t]
                z[pair] = np.nan
                fold[pair] = np.nan
                se[pair] = np.nan
            continue
        fitted = np.zeros(p)
        fitted[0] = math.log(total / n)
        eta = np.empty(n)
        mu = np.empty(n)
        phi = fit_negative_binomial(y, covariates, fitted, eta, mu, factors[k])
        coefficients[k, :] = fitted
        dispersion[k] = phi
        for i in range(n):
            scale = 1.0 + phi * mu[i]
            scores[i, k] = (y[i] - mu[i]) / scale
            weights[i, k] = mu[i] / scale
        solved = np.empty(p)
        for t in range(pair_offsets[k], pair_offsets[k + 1]):
            pair = pair_order[t]
            group = pair_groups[pair]
            statistic, change, error = set_test(
                y,
                mu,
                covariates,
                phi,
                members[offsets[group] : offsets[group + 1]],
                factors[k],
                solved,
            )
            z[pair] = statistic
            fold[pair] = change
            se[pair] = error


def _band_fits(
    counts: np.ndarray,
    design: np.ndarray,
    rows: np.ndarray,
    members: np.ndarray,
    offsets: np.ndarray,
    pair_groups: np.ndarray,
    pair_columns: np.ndarray,
) -> tuple[GeneFits, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the genes of ``rows`` and return their fits and each pair's observed test."""
    n_cells, p = design.shape
    n_genes = len(rows)
    pair_order = np.argsort(pair_columns, kind="stable")
    pair_offsets = np.zeros(n_genes + 1, dtype=np.int64)
    np.cumsum(np.bincount(pair_columns, minlength=n_genes), out=pair_offsets[1:])
    fits = GeneFits(
        coefficients=np.empty((n_genes, p)),
        dispersion=np.empty(n_genes),
        scores=np.zeros((n_cells, n_genes)),
        weights=np.zeros((n_cells, n_genes)),
        factors=np.empty((n_genes, p, p)),
    )
    n_pairs = len(pair_groups)
    z = np.empty(n_pairs)
    fold = np.empty(n_pairs)
    se = np.empty(n_pairs)
    _fit_band(
        counts,
        np.ascontiguousarray(rows, dtype=np.int64),
        design,
        members,
        offsets,
        pair_groups,
        pair_offsets,
        pair_order,
        fits.coefficients,
        fits.dispersion,
        fits.scores,
        fits.weights,
        fits.factors,
        z,
        fold,
        se,
    )
    return fits, z, fold, se


@njit(cache=True, nogil=True)
def _sampler(
    covariates: np.ndarray, model: np.ndarray, conditional: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a group's probabilities and their buckets, all empty for permutation sets."""
    n = covariates.shape[0] if conditional else 0
    probabilities = np.empty(n)
    order = np.empty(n, dtype=np.int64)
    offsets = np.empty(N_BUCKETS + 1 if conditional else 0, dtype=np.int64)
    bounds = np.empty(N_BUCKETS if conditional else 0)
    if conditional:
        logistic_probabilities(covariates, model, np.empty(n), probabilities)
        probability_buckets(probabilities, order, offsets, bounds)
    return probabilities, order, offsets, bounds


@njit(cache=True, nogil=True)
def _gather(
    scores: np.ndarray, weights: np.ndarray, factors: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy the scores, weights and factors of some genes of a band together."""
    n = scores.shape[0]
    width = len(columns)
    p = factors.shape[1]
    gathered_scores = np.empty((n, width))
    gathered_weights = np.empty((n, width))
    for i in range(n):
        for k in range(width):
            gathered_scores[i, k] = scores[i, columns[k]]
            gathered_weights[i, k] = weights[i, columns[k]]
    gathered_factors = np.empty((width, p, p))
    for k in range(width):
        gathered_factors[k] = factors[columns[k]]
    return gathered_scores, gathered_weights, gathered_factors


@njit(cache=True, nogil=True)
def _accumulate(
    cells: np.ndarray,
    count: int,
    scores: np.ndarray,
    weights: np.ndarray,
    first: int,
    width: int,
    covariates: np.ndarray,
    sums: np.ndarray,
) -> None:
    """Sum a set's scores, and its weights times each covariate, over some genes.

    The genes are the columns ``first`` to ``first + width`` of ``scores``
    and ``weights``; ``sums`` receives the score sums in its first row and
    one row per covariate after it. The cells add up in the order listed.
    """
    p = covariates.shape[1]
    sums[:, :] = 0.0
    for i in range(count):
        j = cells[i]
        # Rows of the genes, indexed from 0, compile to tighter loops than
        # offsets into the band.
        cell_scores = scores[j, first : first + width]
        cell_weights = weights[j, first : first + width]
        score_sums = sums[0]
        for k in range(width):
            score_sums[k] += cell_scores[k]
        for c in range(p):
            value = covariates[j, c]
            weighted_sums = sums[1 + c]
            for k in range(width):
                weighted_sums[k] += cell_weights[k] * value


@njit(cache=True, nogil=True)
def _score_sets(
    first_set: int | np.int64,
    step: int,
    n_draws: int,
    key: np.uint64,
    conditional: bool,
    probabilities: np.ndarray,
    order: np.ndarray,
    bucket_offsets: np.ndarray,
    bounds: np.ndarray,
    size: int,
    scores: np.ndarray,
    weights: np.ndarray,
    factors: np.ndarray,
    column: int,
    width: int,
    covariates: np.ndarray,
    pair_local: np.ndarray,
    pair_slots: np.ndarray,
    observed: np.ndarray,
    counts: np.ndarray,
    statistics: np.ndarray,
) -> None:
    """Score a group's random sets ``first_set``, ``first_set + step``, ... on some genes.

    The genes are the columns ``column`` to ``column + width`` of the scores,
    weights and factors. Each pair reads its gene's statistic at
    ``pair_local``; with rows in ``statistics`` it is stored in the pair's
    slot, and otherwise ``counts`` counts the sets at most and at least the
    pair's observed statistic.
    """
    n, p = covariates.shape
    drawn = np.empty(n if conditional else size, dtype=np.int64)
    marked = np.zeros(0 if conditional else n, dtype=np.bool_)
    sums = np.empty((p + 1, width))
    vector = np.empty(p)
    solved = np.empty(p)
    values = np.empty(width)
    store = statistics.shape[0] > 0
    for d in range(first_set, n_draws, step):
        key_d = set_key(key, d)
        count = size
        if conditional:
            count = draw_bernoulli_set(
                key_d, probabilities, order, bucket_offsets, bounds, drawn
            )
        else:
            draw_uniform_set(key_d, size, n, marked, drawn)
        _accumulate(drawn, count, scores, weights, column, width, covariates, sums)
        for k in range(width):
            for c in range(p):
                vector[c] = sums[1 + c, k]
            values[k] = score_statistic(
                sums[0, k], sums[1, k], vector, factors[column + k], solved
            )
        for i in range(len(pair_local)):
            value = values[pair_local[i]]
            slot = pair_slots[i]
            if store:
                statistics[slot, d] = value
            else:
                if value <= observed[i]:
                    counts[slot, 0] += 1
                if value >= observed[i]:
                    counts[slot, 1] += 1


@njit(cache=True, nogil=True, parallel=True)
def _score_groups(
    scores: np.ndarray,
    weights: np.ndarray,
    factors: np.ndarray,
    covariates: np.ndarray,
    sizes: np.ndarray,
    keys: np.ndarray,
    models: np.ndarray,
    conditional: bool,
    active: np.ndarray,
    group_offsets: np.ndarray,
    unit_first: np.ndarray,
    unit_width: np.ndarray,
    unit_compact: np.ndarray,
    unit_columns: np.ndarray,
    column_offsets: np.ndarray,
    pair_offsets: np.ndarray,
    pair_local: np.ndarray,
    pair_slots: np.ndarray,
    observed: np.ndarray,
    n_draws: int,
    counts: np.ndarray,
    statistics: np.ndarray,
) -> None:
    """Score the random sets of the active groups, a group per thread."""
    for t in prange(len(active)):
        group = active[t]
        probabilities, order, bucket_offsets, bounds = _sampler(
            covariates, models[group], conditional
        )
        for u in range(group_offsets[t], group_offsets[t + 1]):
            if unit_compact[u]:
                unit_scores, unit_weights, unit_factors = _gather(
                    scores,
                    weights,
                    factors,
                    unit_columns[column_offsets[u] : column_offsets[u + 1]],
                )
                column = 0
            else:
                unit_scores, unit_weights, unit_factors = scores, weights, factors
                column = unit_first[u]
            start = pair_offsets[u]
            end = pair_offsets[u + 1]
            # int() gives int64 values rather than literals, so every call uses
            # one compiled version of the kernel.
            _score_sets(
                int(0),
                int(1),
                n_draws,
                keys[group],
                conditional,
                probabilities,
                order,
                bucket_offsets,
                bounds,
                sizes[group],
                unit_scores,
                unit_weights,
                unit_factors,
                column,
                unit_width[u],
                covariates,
                pair_local[start:end],
                pair_slots[start:end],
                observed[start:end],
                counts,
                statistics,
            )


@njit(cache=True, nogil=True, parallel=True)
def _score_blocks(
    n_blocks: int,
    n_draws: int,
    key: np.uint64,
    conditional: bool,
    probabilities: np.ndarray,
    order: np.ndarray,
    bucket_offsets: np.ndarray,
    bounds: np.ndarray,
    size: int,
    scores: np.ndarray,
    weights: np.ndarray,
    factors: np.ndarray,
    column: int,
    width: int,
    covariates: np.ndarray,
    pair_local: np.ndarray,
    pair_slots: np.ndarray,
    observed: np.ndarray,
    counts: np.ndarray,
    statistics: np.ndarray,
) -> None:
    """Score one group's random sets on some genes, a block of sets per thread.

    Each block counts the pairs in its own rows, one per pair given, which
    add up in block order after; stored statistics go straight to the pairs'
    rows.
    """
    store = statistics.shape[0] > 0
    n_pairs = len(pair_slots)
    block_counts = np.zeros((n_blocks, 0 if store else n_pairs, 2), dtype=np.int64)
    slots = pair_slots if store else np.arange(n_pairs)
    for block in prange(n_blocks):
        # prange's index is unsigned; np.int64 keeps every call on one compiled
        # version of the kernel.
        _score_sets(
            np.int64(block),
            n_blocks,
            n_draws,
            key,
            conditional,
            probabilities,
            order,
            bucket_offsets,
            bounds,
            size,
            scores,
            weights,
            factors,
            column,
            width,
            covariates,
            pair_local,
            slots,
            observed,
            block_counts[block],
            statistics,
        )
    for block in range(n_blocks):
        for i in range(block_counts.shape[1]):
            counts[pair_slots[i], 0] += block_counts[block, i, 0]
            counts[pair_slots[i], 1] += block_counts[block, i, 1]


def _plan_units(columns: np.ndarray, bounds: np.ndarray, gather_columns: int) -> _Units:
    """Split each group's genes into the units of a round.

    ``columns`` holds the pairs' genes, sorted by group and then gene, and a
    group's pairs run from ``bounds[t]`` to ``bounds[t + 1]``. A group whose
    genes fit ``gather_columns`` is copied together; otherwise genes that fill
    at least half of their span of the band are read in place, and scattered
    ones are copied ``gather_columns`` at a time.
    """
    group_offsets = [0]
    first: list[int] = []
    width: list[int] = []
    compact: list[bool] = []
    copied: list[np.ndarray] = []
    column_offsets = [0]
    pair_offsets = [0]
    local = np.empty(len(columns), dtype=np.int64)
    for t in range(len(bounds) - 1):
        start, end = int(bounds[t]), int(bounds[t + 1])
        group_columns = columns[start:end]
        genes = np.unique(group_columns)
        span = int(genes[-1] - genes[0]) + 1
        if len(genes) > gather_columns and 2 * len(genes) >= span:
            first.append(int(genes[0]))
            width.append(span)
            compact.append(False)
            column_offsets.append(column_offsets[-1])
            local[start:end] = group_columns - genes[0]
            pair_offsets.append(end)
        else:
            for begin in range(0, len(genes), gather_columns):
                chunk = genes[begin : begin + gather_columns]
                lower = start + int(np.searchsorted(group_columns, chunk[0], "left"))
                upper = start + int(np.searchsorted(group_columns, chunk[-1], "right"))
                local[lower:upper] = np.searchsorted(chunk, columns[lower:upper])
                first.append(0)
                width.append(len(chunk))
                compact.append(True)
                copied.append(chunk)
                column_offsets.append(column_offsets[-1] + len(chunk))
                pair_offsets.append(upper)
        group_offsets.append(len(first))
    return _Units(
        group_offsets=np.asarray(group_offsets, dtype=np.int64),
        first=np.asarray(first, dtype=np.int64),
        width=np.asarray(width, dtype=np.int64),
        compact=np.asarray(compact, dtype=np.bool_),
        columns=np.concatenate([np.empty(0, dtype=np.int64), *copied]).astype(np.int64),
        column_offsets=np.asarray(column_offsets, dtype=np.int64),
        pair_offsets=np.asarray(pair_offsets, dtype=np.int64),
        local=local,
    )


def _resample(
    band: _Band, chosen: np.ndarray, stage: int, n_draws: int, *, store: bool
) -> np.ndarray:
    """Return the random sets' statistics or counts of some pairs in one round.

    With ``store``, each pair's statistics, pairs x sets; otherwise, per pair,
    the sets at most and at least its observed statistic, pairs x 2. With
    many groups, each thread takes whole groups; otherwise the groups run one
    at a time, their sets split among the threads. Both give the same
    results.
    """
    n_chosen = len(chosen)
    counts = np.zeros((0 if store else n_chosen, 2), dtype=np.int64)
    statistics = np.empty((n_chosen if store else 0, n_draws))
    if not n_chosen:
        return statistics if store else counts
    groups_of = band.pair_groups[chosen]
    columns_of = band.pair_columns[chosen]
    order = np.lexsort((columns_of, groups_of))
    active, starts = np.unique(groups_of[order], return_index=True)
    units = _plan_units(
        columns_of[order], np.append(starts, n_chosen), band.gather_columns
    )
    slots = order.astype(np.int64)
    observed = np.ascontiguousarray(band.z[chosen][order])
    keys = resampling_keys(band.seed, band.codes, stage)
    fits = band.fits
    threads = numba.get_num_threads()
    if len(active) >= 2 * threads:
        # Groups differ in work by their genes, so threads take them one at a
        # time rather than in fixed blocks.
        with numba.parallel_chunksize(1):
            _score_groups(
                fits.scores,
                fits.weights,
                fits.factors,
                band.covariates,
                band.sizes,
                keys,
                band.models,
                band.conditional,
                active.astype(np.int64),
                units.group_offsets,
                units.first,
                units.width,
                units.compact,
                units.columns,
                units.column_offsets,
                units.pair_offsets,
                units.local,
                slots,
                observed,
                n_draws,
                counts,
                statistics,
            )
        return statistics if store else counts
    for t, group in enumerate(active):
        probabilities, bucket_order, bucket_offsets, bounds = _sampler(
            band.covariates, band.models[group], band.conditional
        )
        for u in range(units.group_offsets[t], units.group_offsets[t + 1]):
            if units.compact[u]:
                scores, weights, factors = _gather(
                    fits.scores,
                    fits.weights,
                    fits.factors,
                    units.columns[
                        units.column_offsets[u] : units.column_offsets[u + 1]
                    ],
                )
                column = 0
            else:
                scores, weights, factors = fits.scores, fits.weights, fits.factors
                column = int(units.first[u])
            pairs = slice(units.pair_offsets[u], units.pair_offsets[u + 1])
            _score_blocks(
                min(threads, n_draws),
                n_draws,
                keys[group],
                band.conditional,
                probabilities,
                bucket_order,
                bucket_offsets,
                bounds,
                int(band.sizes[group]),
                scores,
                weights,
                factors,
                column,
                int(units.width[u]),
                band.covariates,
                units.local[pairs],
                slots[pairs],
                observed[pairs],
                counts,
                statistics,
            )
            # Free this unit's copy before the next unit makes its own.
            del scores, weights, factors
    return statistics if store else counts


@njit(cache=True, nogil=True, parallel=True)
def _expected_set_sizes(
    covariates: np.ndarray, models: np.ndarray, groups: np.ndarray, sizes: np.ndarray
) -> None:
    """Write how many cells the detection model of each of ``groups`` expects in a set."""
    n = covariates.shape[0]
    for t in prange(len(groups)):
        probabilities = np.empty(n)
        logistic_probabilities(
            covariates, models[groups[t]], np.empty(n), probabilities
        )
        sizes[t] = probabilities.sum()


def _resampling_models(
    resampling: object,
    detection_models: np.ndarray | None,
    design: np.ndarray,
    n_groups: int,
    used: np.ndarray,
) -> np.ndarray:
    """Check the resampling and return each group's detection model, 0 for permutations.

    A conditional set is drawn again until it holds some cells but not all,
    so the model of each group in ``used`` must expect at least half a cell in
    a set and leave at least half a cell out; then at least one draw in six
    succeeds. A fitted model expects the group's own cells.
    """
    n_cells, p = design.shape
    if resampling not in RESAMPLING:
        raise ValueError(
            f"resampling must be 'conditional' or 'permutation', not {resampling!r}"
        )
    if resampling == "permutation":
        if detection_models is not None:
            raise ValueError("detection_models go with resampling='conditional'")
        return np.zeros((n_groups, p))
    if detection_models is None:
        raise ValueError(
            "Conditional resampling needs each group's detection model; pass "
            "detection_models from fit_detection_models"
        )
    models = np.ascontiguousarray(detection_models, dtype=np.float64)
    if models.shape != (n_groups, p) or not np.isfinite(models).all():
        raise ValueError(
            f"detection_models must hold finite coefficients, {n_groups} groups x "
            f"{p} covariates"
        )
    expected = np.empty(len(used))
    _expected_set_sizes(design, models, used, expected)
    poor = np.flatnonzero((expected < 0.5) | (expected > n_cells - 0.5))
    if len(poor):
        raise ValueError(
            f"The detection model of group {used[poor[0]]} expects "
            f"{expected[poor[0]]:.3g} of the {n_cells} cells in a set; a model must "
            "expect at least half a cell and leave at least half a cell out, as "
            "the models of fit_detection_models do"
        )
    return models


def cells_by_code(codes: np.ndarray, n_codes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cells that hold each code, ascending, and where each code's cells start.

    ``codes`` holds each cell's codes from 0, padded with -1, cells x slots,
    and a cell holds each of its codes once.
    """
    cells, slots = np.nonzero(codes >= 0)
    held = codes[cells, slots].astype(np.int64)
    # A stable sort keeps each code's cells in the ascending order of nonzero.
    members = cells[np.argsort(held, kind="stable")].astype(np.int64)
    offsets = np.zeros(n_codes + 1, dtype=np.int64)
    np.cumsum(np.bincount(held, minlength=n_codes), out=offsets[1:])
    return members, offsets


def complement_thread_bytes(n_cells: int, gather_bytes: int) -> int:
    """Return the most bytes that each thread of :func:`complement_tests` holds.

    Beyond the band, the pairs' arrays and the second round's statistics, a
    thread holds a group's conditional sets, 24 bytes per cell, and its copy
    of some of the group's genes, at least one gene of 16 bytes per cell and
    at most ``gather_bytes`` otherwise; 24 bytes per cell also bound the
    permutation sets and 32 bytes per cell the fit of a gene.
    """
    return 24 * n_cells + max(gather_bytes, 16 * n_cells)


def complement_tests(
    values: np.ndarray,
    covariates: np.ndarray,
    groups: ComplementGroups,
    pair_groups: np.ndarray,
    pair_rows: np.ndarray,
    *,
    resampling: str = "conditional",
    detection_models: np.ndarray | None = None,
    seed: int = 0,
    n_resamples: tuple[int, int, int] = (499, 4999, 24999),
    stage_threshold: float = 0.02,
    approximation: str = "skew_normal",
    side: str = "both",
    statistics_bytes: int = STATISTICS_BYTES,
    gather_bytes: int = GATHER_BYTES,
) -> PairResults:
    """Test pairs of a group and a gene of a band against all other cells.

    Each gene paired in the band is fitted once on all cells: Poisson
    coefficients by Newton's method from the log of the gene's mean, then the
    size that maximizes the likelihood with the Poisson means held fixed, as
    the low-MOI test fits a pair on its cells; a gene without counts gets no
    fit. With mean mu and dispersion phi = 1 / size, a cell's
    weight is w = mu / (1 + phi mu) and its score a = (y - mu) / (1 + phi
    mu), and the statistic of a set X of cells is sum(a over X) / sqrt(sum(w
    over X) - v' J^-1 v), where v sums w times the covariates over X and J
    is the information of the fit. The pair's statistic is that of the
    group's treatment cells; its fold change is log2 of their counts over
    their means under the fit, with a delta-method standard error.

    The p-values come from random sets in up to three rounds, as
    :func:`~cytearc.perturbation.p_values.staged_p_values` describes.
    With ``resampling="conditional"`` each set takes every cell independently
    with its probability under the group's detection model, and a set that
    holds no cell or every cell, which has no statistic, is drawn again, as
    :func:`~cytearc.perturbation.random_sets.draw_bernoulli_set` describes;
    with ``"permutation"`` it holds as many cells as the group, drawn
    uniformly. A group's sets are drawn once for all its genes in the band,
    and a pair of a gene without counts has no statistic and no p-value.

    For n cells, the band's scores and weights take 16 n bytes per gene. Each
    thread holds 32 n bytes while it fits a gene, and later, while it draws a
    group's sets, 24 n bytes for conditional sets (each cell's probability,
    the cells by bucket and a set) or n bytes and 8 per treatment cell for
    permutation sets, plus its copy of some of the group's genes, at most
    max(gather_bytes, 16 n) bytes. With T threads, the work therefore holds at
    most T (24 n + max(gather_bytes, 16 n)) bytes beyond the band, the pairs'
    arrays and the second round's statistics.

    Args:
        values: Counts of the band's genes, genes x cells, float32 or
            float64.
        covariates: The covariates of the cells, cells x covariates, the
            intercept first.
        groups: The groups of treatment cells.
        pair_groups: Each pair's group.
        pair_rows: Each pair's gene, a row of ``values``.
        resampling: ``"conditional"`` or ``"permutation"``.
        detection_models: Each group's logistic coefficients from
            :func:`fit_detection_models`, with conditional resampling only.
            Each paired group's model must expect at least half a cell in a
            set and leave at least half a cell out, as a fitted model does,
            so that the draws often hold some cells but not all.
        seed: The seed of the random sets.
        n_resamples: The random sets of the three rounds.
        stage_threshold: The p-value at or below which a pair takes the
            second round.
        approximation: ``"skew_normal"`` or ``"none"``.
        side: ``"both"``, ``"left"`` or ``"right"``.
        statistics_bytes: The memory that the second round's statistics may
            hold at once, at least one pair's, ``8 * n_resamples[1]`` bytes;
            their checks take a working copy no larger.
        gather_bytes: The memory that each thread's copy of a group's genes
            may take, at 16 bytes per cell and gene; a copy holds at least one
            gene, so it takes at most max(gather_bytes, 16 n) bytes for n
            cells. A group's genes are copied together so that its sets read
            them from one place; when they exceed it, genes that fill at least
            half of their span of the band are read in place, and scattered
            genes are copied as many at a time as it holds, or one at a time.

    Returns:
        Each pair's test.

    Raises:
        ValueError: If the counts, covariates, groups or pairs are malformed,
            a covariate is not finite, ``resampling`` and ``detection_models``
            do not go together, or a detection model expects less than half
            a cell in a set or leaves less than half a cell out.
    """
    counts = count_array(values)
    design = _design(covariates)
    if counts.ndim != 2 or counts.shape[1] != design.shape[0]:
        raise ValueError("values must be genes x cells with one covariate row per cell")
    n_cells = design.shape[0]
    members, offsets, codes = _group_arrays(groups, n_cells)
    n_groups = len(offsets) - 1
    pair_group = np.ascontiguousarray(pair_groups, dtype=np.int64)
    rows = np.ascontiguousarray(pair_rows, dtype=np.int64)
    if pair_group.ndim != 1 or pair_group.shape != rows.shape:
        raise ValueError("pair_groups and pair_rows need one entry per pair")
    if len(rows) and (pair_group.min() < 0 or pair_group.max() >= n_groups):
        raise ValueError(f"pair_groups must be groups from 0 to {n_groups - 1}")
    if len(rows) and (rows.min() < 0 or rows.max() >= counts.shape[0]):
        raise ValueError(
            f"pair_rows must be rows of values, from 0 to {len(counts) - 1}"
        )
    models = _resampling_models(
        resampling, detection_models, design, n_groups, np.unique(pair_group)
    )
    genes, pair_columns = np.unique(rows, return_inverse=True)
    pair_columns = pair_columns.astype(np.int64)
    fits, z, fold, se = _band_fits(
        counts, design, genes, members, offsets, pair_group, pair_columns
    )
    band = _Band(
        covariates=design,
        fits=fits,
        sizes=np.diff(offsets),
        codes=codes,
        models=models,
        conditional=resampling == "conditional",
        pair_groups=pair_group,
        pair_columns=pair_columns,
        z=z,
        seed=seed,
        gather_columns=max(1, gather_bytes // (16 * n_cells)),
    )
    tested = np.flatnonzero(~np.isnan(z))
    first_counts = np.zeros((len(z), 2), dtype=np.int64)
    first_counts[tested] = _resample(band, tested, 1, n_resamples[0], store=False)
    p_value, stage, location, scale, shape = staged_p_values(
        z,
        first_counts,
        lambda chosen, n_draws: _resample(band, chosen, 2, n_draws, store=True),
        lambda chosen, n_draws: _resample(band, chosen, 3, n_draws, store=False),
        n_resamples=n_resamples,
        stage_threshold=stage_threshold,
        approximation=approximation,
        side=side,
        statistics_bytes=statistics_bytes,
    )
    return PairResults(
        z=z,
        p_value=p_value,
        stage=stage,
        skew_location=location,
        skew_scale=scale,
        skew_shape=shape,
        log2_fold_change=fold,
        se_log2_fold_change=se,
    )
