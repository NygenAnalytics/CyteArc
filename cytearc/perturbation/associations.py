"""Test whether each perturbation changes each gene, with calibrated p-values.

SCEPTRE's low-MOI test (Barry et al. 2024) compares, for each pair of a
perturbation and a gene, the cells that carry the perturbation, its treatment
cells, with the control cells. A negative binomial regression of the gene's
counts on cell covariates, fitted under the null hypothesis of no effect, gives
each cell a score, and the pair's statistic is the treatment cells' summed
score, standardized. Its p-value comes from the same statistic over random sets
of as many cells, drawn from the treatment and control cells together: a first
round of sets, and when that round leaves the p-value at or below a threshold,
a second, larger round, whose statistics a skew-normal distribution fits, so
that p-values far below the reciprocal of the number of sets come from the
fitted tail.

Each pair's regression is fitted on its treatment and control cells together,
as the SCEPTRE book describes: Poisson coefficients, and the negative binomial
size that maximizes the likelihood with the Poisson means held fixed, or the
root of the Pearson equation when the likelihood shows no overdispersion. That
fit does not depend on which of those
cells are the treatment cells, so the treatment cells and every random set are
scored by one function of the pair's cells, and the permutation test is exact.
Each gene's fit on the control cells starts the fits of its pairs, and each
pair's fit gives its fold change: the treatment cells' counts over their means
under that fit.

The random sets of a group of treatment cells come from a counter-based
generator keyed by the seed, the group and the round, so every gene paired with
the group sees the same sets, and the results do not depend on threads or on
how genes are split into blocks.

These functions hold counts in memory and never touch storage.
"""

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats

from cytearc.perturbation.efficiency import id_column_setting
from cytearc.perturbation.glm import (
    cholesky,
    count_array,
    fit_poisson,
    log2_fold_change,
    negative_binomial_information,
    score_statistic,
    size_dispersion,
)
from cytearc.perturbation.random_sets import (
    draw_uniform_range,
    draw_uniform_sets,
    resampling_keys,
)
from cytearc.utils.arguments import float_argument, integer_argument

__all__ = [
    "ASSOCIATION_SETS",
    "RECORD_BYTES",
    "CALIBRATION_GRADES",
    "MOIS",
    "SIDES",
    "PairResults",
    "PerturbationAssociations",
    "TreatmentGroups",
    "adjusted_p_values",
    "association_covariates",
    "association_settings",
    "covariate_columns",
    "calibration_grade",
    "calibration_pairs",
    "fit_null_models",
    "fit_skew_normals",
    "negative_control_groups",
    "nonzero_counts",
    "pair_tests",
    "record_batch",
    "resampled_statistics",
    "resampling_counts",
    "resampling_p_values",
    "skew_normal_p_values",
    "skew_normal_tail_ratios",
    "staged_p_values",
]

#: The sets of pairs that an association test holds.
ASSOCIATION_SETS = ("discovery", "calibration", "positive")
#: The grade of a calibration check by its false discoveries, as the SCEPTRE
#: book grades them: the largest count of each grade, and above the last,
#: ``"severe"``.
CALIBRATION_GRADES = ((1, "excellent"), (6, "mild"), (14, "moderate"))
#: The alternatives of a test: either direction, a decrease or an increase.
SIDES = ("both", "left", "right")
#: The multiplicities of infection that the tests take: ``"low"`` compares the
#: cells that carry a target alone with the control cells, and ``"high"`` the
#: cells that carry any of its guides with all other cells.
MOIS = ("low", "high")

# A start of a pair's fit with a coefficient beyond this size, on covariates
# that are standardized or indicators, sits where the fit cannot move.
_START_LIMIT = 10.0
# Above this skewness the moment equations of the skew-normal have no
# solution; its largest skewness is about 0.9953.
_SKEWNESS_LIMIT = 0.995
# sceptre's fits take this skewness, with its sign, for a sample beyond the
# skew-normal's reach.
_SKEWNESS_FALLBACK = 0.9 * _SKEWNESS_LIMIT
# A fit whose tail at the 99th percentile holds less than this share of the
# statistics' own gives way to the third round.
_TAIL_RATIO_FLOOR = 0.5
#: The memory that one thread's records of a batch of pairs may take: about a
#: thread's share of a core's L2 cache, so the rows that each random set reads
#: stay in cache when every core is busy.
RECORD_BYTES = 1 << 20
# The fewest and the most pairs whose records one thread scores together. Two
# pairs use more of each cache line they read than one, even past RECORD_BYTES.
_BATCH_FLOOR = 2
_BATCH_LIMIT = 32
# The second round's statistics that its counts, fits and checks work on at
# once, in whole rows: a working copy that stays in a core's L2 cache through
# its passes.
_SPAN_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class TreatmentGroups:
    """The treatment cells of each group, as positions among the tested cells.

    The tested cells hold the control cells first, at positions 0 to
    ``n_controls - 1``. A group whose treatment cells lie outside the control
    cells, the cells of a target, is compared with all control cells. A group
    whose treatment cells are control cells, a negative-control group of the
    calibration check, is compared with the other control cells; such groups
    may share cells.

    Attributes:
        members: The positions of the treatment cells, group after group.
        offsets: Where each group's positions start in ``members``, and the
            end of the last group.
        among_controls: Whether each group's treatment cells are control
            cells.
        codes: A distinct, non-negative code for each group, which keys its
            random sets through :func:`resampling_keys`.
        n_controls: The number of control cells.
    """

    members: np.ndarray
    offsets: np.ndarray
    among_controls: np.ndarray
    codes: np.ndarray
    n_controls: int


@dataclass(frozen=True, slots=True)
class PairResults:
    """The test of each pair of a group and a gene.

    Attributes:
        z: The standardized score statistic; positive when the treatment
            cells express the gene more than the fit predicts.
        p_value: The p-value of the side tested.
        stage: The round whose random sets gave the p-value: 1, 2 with a
            skew-normal fit or without the approximation, or 3 when the fit
            failed its check.
        skew_location: The location of the skew-normal fit that gave the
            p-value, NaN when the random sets gave it.
        skew_scale: The scale of that fit.
        skew_shape: The shape of that fit.
        log2_fold_change: log2 of the treatment cells' counts over their
            means under the pair's fit.
        se_log2_fold_change: The standard error of ``log2_fold_change``.
    """

    z: np.ndarray
    p_value: np.ndarray
    stage: np.ndarray
    skew_location: np.ndarray
    skew_scale: np.ndarray
    skew_shape: np.ndarray
    log2_fold_change: np.ndarray
    se_log2_fold_change: np.ndarray


@dataclass(frozen=True, slots=True)
class PerturbationAssociations:
    """The pairs of a ``perturbation_associations`` artifact.

    Each table has one row per pair with ``target``, ``feature``, the cell
    counts ``n_cells``, ``n_control_cells``, ``n_nonzero_cells`` and
    ``n_nonzero_control_cells``, ``pass_qc``, and for pairs that pass QC
    ``z``, ``p_value``, ``adjusted_p_value``, ``significant``,
    ``log2_fold_change``, ``se_log2_fold_change``, ``stage`` and the
    skew-normal fit, ``skew_location``, ``skew_scale`` and ``skew_shape``.

    Attributes:
        discovery: The pairs asked about.
        calibration: The negative-control pairs, whose treatment cells carry
            groups of control guides; their ``target`` names the group and
            ``control_guides`` its guides.
        positive: The positive-control pairs, such as each target with its own
            gene.
        alpha: The false discovery rate of ``significant``.
        side: The alternative tested: ``"both"``, ``"left"`` or ``"right"``.
        integration: ``"union"`` when a target's guides are tested together,
            ``"singleton"`` when each guide is tested alone.
        moi: ``"low"`` when a target's cells carry it alone and are compared
            with the control cells, ``"high"`` when they carry any of its
            guides and are compared with all other cells.
        resampling: How the random sets were drawn: ``"permutation"``, as
            many cells as the treatment cells, or ``"conditional"``, each
            cell with its fitted probability of carrying the perturbation.
        from_assay: The expression assay.
        n_false_discoveries: The negative-control pairs found significant.
        calibration_grade: That count's grade, ``"excellent"``, ``"mild"``,
            ``"moderate"`` or ``"severe"``, or ``"none"`` when no calibration
            check ran.
    """

    discovery: pd.DataFrame
    calibration: pd.DataFrame
    positive: pd.DataFrame
    alpha: float
    side: str
    integration: str
    moi: str
    resampling: str
    from_assay: str
    n_false_discoveries: int
    calibration_grade: str


def association_covariates(
    n_umis: np.ndarray,
    n_detected: np.ndarray,
    samples: np.ndarray | None = None,
    extra: np.ndarray | None = None,
    *,
    names: Sequence[str] = (),
    guide_umis: np.ndarray | None = None,
    guide_detected: np.ndarray | None = None,
) -> np.ndarray:
    """Return the covariates of the regressions, cells x covariates.

    The columns are an intercept, log detected features and log UMIs of each
    cell, with guide counts its log detected guides and log guide UMIs, then
    indicators of every sample but the first, and any ``extra`` columns, the
    terms of sceptre's default formula. The continuous columns are centered
    and scaled, which leaves the fits unchanged but keeps their linear
    systems well conditioned.

    Args:
        n_umis: Each cell's total counts in the expression assay.
        n_detected: Each cell's number of features with counts.
        samples: Each cell's sample as a code from 0, or None for one sample.
        extra: More numeric covariates, cells x columns, or None.
        names: The names of the ``extra`` columns, for error messages.
        guide_umis: Each cell's total counts in the guide assay, which the
            high-MOI test adjusts for, or None.
        guide_detected: Each cell's number of guides with counts, given with
            ``guide_umis``.

    Returns:
        The covariates as float64, the intercept first.

    Raises:
        ValueError: If a cell has no counts or, with guide counts, no guide
            UMIs, a sample has no cell, or a covariate is constant or not
            finite.
    """
    umis = np.asarray(n_umis, dtype=np.float64)
    detected = np.asarray(n_detected, dtype=np.float64)
    if umis.shape != detected.shape or umis.ndim != 1:
        raise ValueError("n_umis and n_detected need one value per cell")
    if len(umis) and (umis.min() <= 0 or detected.min() <= 0):
        raise ValueError(
            "Every tested cell needs counts; filter cells without counts first"
        )
    # Each column with its name, and whether it is centered and scaled.
    columns = [
        (np.ones(len(umis)), "intercept", False),
        (np.log(detected), "log detected features", True),
        (np.log(umis), "log UMIs", True),
    ]
    if (guide_umis is None) != (guide_detected is None):
        raise ValueError("guide_umis and guide_detected go together")
    if guide_umis is not None:
        guide_total = np.asarray(guide_umis, dtype=np.float64)
        guides = np.asarray(guide_detected, dtype=np.float64)
        if guide_total.shape != umis.shape or guides.shape != umis.shape:
            raise ValueError("guide_umis and guide_detected need one value per cell")
        without = int(np.count_nonzero((guide_total <= 0) | (guides <= 0)))
        if without:
            verb = "has" if without == 1 else "have"
            raise ValueError(
                f"{without} of the {len(umis)} tested cells {verb} no guide UMIs, "
                "and the test adjusts for the log of each cell's guide UMIs; "
                "select cells with guide UMIs"
            )
        columns += [
            (np.log(guides), "log detected guides", True),
            (np.log(guide_total), "log guide UMIs", True),
        ]
    if samples is not None:
        codes = np.asarray(samples, dtype=np.int64)
        if codes.shape != umis.shape or (len(codes) and codes.min() < 0):
            raise ValueError("samples needs one code from 0 per cell")
        present = np.bincount(codes, minlength=1)
        if (present == 0).any():
            raise ValueError("Every sample needs a cell; number the samples from 0")
        for code in range(1, len(present)):
            columns.append(
                ((codes == code).astype(np.float64), f"sample {code}", False)
            )
    if extra is not None:
        values = np.asarray(extra, dtype=np.float64).reshape(len(umis), -1)
        given = list(names) or [f"column {k}" for k in range(values.shape[1])]
        for k in range(values.shape[1]):
            if not np.isfinite(values[:, k]).all():
                raise ValueError(
                    f"The covariate {given[k]!r} has values that are not finite"
                )
            columns.append((values[:, k].copy(), given[k], True))
    for column, name, scaled in columns:
        if scaled:
            spread = column.std()
            if not spread > 0:
                raise ValueError(
                    f"The covariate {name!r} is constant over the tested cells"
                )
            column -= column.mean()
            column /= spread
    return np.ascontiguousarray(np.column_stack([column for column, _, _ in columns]))


@njit(cache=True, nogil=True, parallel=True)
def _fit_null(
    values: np.ndarray,
    rows: np.ndarray,
    covariates: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    """Fit the genes of ``rows`` on the first cells; see :func:`fit_null_models`."""
    n, p = covariates.shape
    for k in prange(len(rows)):
        feature = rows[k]
        y = np.empty(n)
        total = 0.0
        for i in range(n):
            y[i] = values[feature, i]
            total += y[i]
        if total <= 0:
            coefficients[feature, :] = np.nan
            continue
        fitted = np.zeros(p)
        fitted[0] = math.log(total / n)
        eta = np.empty(n)
        mu = np.empty(n)
        fit_poisson(y, covariates, fitted, eta, mu)
        coefficients[feature, :] = fitted


def fit_null_models(
    values: np.ndarray,
    covariates: np.ndarray,
    n_controls: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Fit each gene's Poisson regression on the control cells, the first cells.

    These fits start the fit of each pair of the gene. A gene without counts
    in the control cells gets no fit.

    Args:
        values: Counts, genes x cells, the control cells first.
        covariates: The covariates of the cells, from
            :func:`association_covariates`.
        n_controls: The number of control cells.
        rows: The rows of ``values`` to fit, or None for every row; the
            others get NaN.

    Returns:
        Each gene's coefficients, genes x covariates, NaN without a fit.

    Raises:
        ValueError: If the shapes disagree or the control cells do not
            outnumber the covariates.
    """
    counts = count_array(values)
    design = np.ascontiguousarray(covariates, dtype=np.float64)
    if counts.ndim != 2 or design.ndim != 2 or counts.shape[1] != design.shape[0]:
        raise ValueError("values must be genes x cells with one covariate row per cell")
    n_features, p = counts.shape[0], design.shape[1]
    if n_controls <= p:
        raise ValueError(
            f"The {n_controls} control cells must outnumber the {p} covariates"
        )
    coefficients = np.full((n_features, p), np.nan)
    chosen = np.arange(n_features) if rows is None else np.unique(rows)
    _fit_null(
        counts,
        chosen.astype(np.int64),
        np.ascontiguousarray(design[:n_controls]),
        coefficients,
    )
    return coefficients


@njit(cache=True, nogil=True)
def _union_design(
    covariates: np.ndarray, n_controls: int, members: np.ndarray, among: bool
) -> np.ndarray:
    """Return the covariates of a group's cells: the control cells, then its members."""
    n_union = n_controls if among else n_controls + len(members)
    design = np.empty((n_union, covariates.shape[1]))
    design[:n_controls] = covariates[:n_controls]
    if not among:
        for k in range(len(members)):
            design[n_controls + k] = covariates[members[k]]
    return design


@njit(cache=True, nogil=True)
def _treated_positions(n_controls: int, members: np.ndarray, among: bool) -> np.ndarray:
    """Return where a group's treatment cells sit among its cells."""
    if among:
        return members.copy()
    return np.arange(n_controls, n_controls + len(members))


@njit(cache=True, nogil=True)
def _fit_pair(
    values: np.ndarray,
    row: int,
    coefficients: np.ndarray,
    design: np.ndarray,
    n_controls: int,
    members: np.ndarray,
    among: bool,
    treated: np.ndarray,
    records: np.ndarray,
    column: int,
    factor: np.ndarray,
) -> tuple[float, float, float]:
    """Fit one pair on its group's cells and write each cell's record.

    A cell's record, in ``records[:, column : column + p + 1]``, is its score
    and its weight times each covariate, the first being its weight, so a
    set's sums add whole records; ``factor`` becomes the Cholesky factor of
    the fit's information. Returns the treatment cells' statistic, the log2
    fold change and its standard error, all NaN for a gene without a fit.
    """
    if np.isnan(coefficients[row, 0]):
        return np.nan, np.nan, np.nan
    n_union, p = design.shape
    y = np.empty(n_union)
    y[:n_controls] = values[row, :n_controls]
    if not among:
        for k in range(len(members)):
            y[n_controls + k] = values[row, members[k]]
    eta = np.empty(n_union)
    mu = np.empty(n_union)
    # The fit starts from the gene's fit on the control cells, unless a
    # sample without counts among the control cells drove that fit's
    # coefficients far out, where the pair's other cells carry almost no
    # information; the pair's mean starts it then.
    fitted = coefficients[row].copy()
    if np.abs(fitted).max() > _START_LIMIT:
        fitted[:] = 0.0
        fitted[0] = math.log(y.sum() / n_union)
    fit_poisson(y, design, fitted, eta, mu)
    dispersion = size_dispersion(y, mu, float(n_union - p))
    for j in range(n_union):
        scale = 1.0 + dispersion * mu[j]
        weight = mu[j] / scale
        records[j, column] = (y[j] - mu[j]) / scale
        for c in range(p):
            records[j, column + 1 + c] = weight * design[j, c]
    information = np.empty((p, p))
    negative_binomial_information(design, mu, dispersion, information)
    cholesky(information, factor)
    sums = np.zeros(p + 1)
    for j in treated:
        for c in range(p + 1):
            sums[c] += records[j, column + c]
    solved = np.empty(p)
    observed = score_statistic(sums[0], sums[1], sums[1:], factor, solved)
    # sum(y) / sum(mu) over the treatment cells, with mu from the pair's fit.
    count_sum = 0.0
    mean_sum = 0.0
    square_sum = 0.0
    tangent = np.zeros(p)
    for j in treated:
        count_sum += y[j]
        mean_sum += mu[j]
        square_sum += mu[j] * mu[j]
        for c in range(p):
            tangent[c] += mu[j] * design[j, c]
    fold, error = log2_fold_change(
        count_sum, mean_sum, square_sum, tangent, factor, dispersion, solved
    )
    return observed, fold, error


@njit(cache=True, nogil=True)
def _score_batch(
    values: np.ndarray,
    coefficients: np.ndarray,
    design: np.ndarray,
    n_controls: int,
    members: np.ndarray,
    among: bool,
    treated: np.ndarray,
    draws: np.ndarray,
    pairs: np.ndarray,
    pair_rows: np.ndarray,
    slots: np.ndarray,
    z: np.ndarray,
    fold: np.ndarray,
    se: np.ndarray,
    counts: np.ndarray,
    statistics: np.ndarray,
    records: np.ndarray,
    factors: np.ndarray,
) -> None:
    """Fit a batch of one group's pairs and score their random sets together.

    The records of the batch's pairs sit side by side, so each cell of a set
    adds one contiguous row. With rows in ``fold`` the fold change and its
    standard error are stored; with rows in ``statistics`` every statistic is
    stored, in the row that ``slots`` gives each pair, and otherwise
    ``counts`` counts those at most and at least the observed one.
    """
    p = design.shape[1]
    width = p + 1
    n_batch = len(pairs)
    span = n_batch * width
    with_fold = fold.shape[0] > 0
    store = statistics.shape[0] > 0
    observed = np.empty(n_batch)
    for k in range(n_batch):
        pair = pairs[k]
        statistic, change, error = _fit_pair(
            values,
            pair_rows[pair],
            coefficients,
            design,
            n_controls,
            members,
            among,
            treated,
            records,
            k * width,
            factors[k],
        )
        observed[k] = statistic
        z[pair] = statistic
        if with_fold:
            fold[pair] = change
            se[pair] = error
    sums = np.empty(span)
    solved = np.empty(p)
    for d in range(draws.shape[0]):
        sums[:] = 0.0
        for i in range(draws.shape[1]):
            j = draws[d, i]
            for m in range(span):
                sums[m] += records[j, m]
        for k in range(n_batch):
            pair = pairs[k]
            base = k * width
            value = np.nan
            if not np.isnan(observed[k]):
                value = score_statistic(
                    sums[base],
                    sums[base + 1],
                    sums[base + 1 : base + width],
                    factors[k],
                    solved,
                )
            if store:
                statistics[slots[pair], d] = value
            else:
                if value <= observed[k]:
                    counts[pair, 0] += 1
                if value >= observed[k]:
                    counts[pair, 1] += 1


@njit(cache=True, nogil=True, parallel=True)
def _score_group(
    values: np.ndarray,
    coefficients: np.ndarray,
    design: np.ndarray,
    n_controls: int,
    members: np.ndarray,
    among: bool,
    treated: np.ndarray,
    draws: np.ndarray,
    pairs: np.ndarray,
    pair_rows: np.ndarray,
    slots: np.ndarray,
    batch: int,
    z: np.ndarray,
    fold: np.ndarray,
    se: np.ndarray,
    counts: np.ndarray,
    statistics: np.ndarray,
) -> None:
    """Score the pairs of one group, a batch of pairs per thread."""
    n_union, p = design.shape
    n_batches = (len(pairs) + batch - 1) // batch
    for b in prange(n_batches):
        chosen = pairs[b * batch : min((b + 1) * batch, len(pairs))]
        _score_batch(
            values,
            coefficients,
            design,
            n_controls,
            members,
            among,
            treated,
            draws,
            chosen,
            pair_rows,
            slots,
            z,
            fold,
            se,
            counts,
            statistics,
            np.empty((n_union, len(chosen) * (p + 1))),
            np.empty((len(chosen), p, p)),
        )


@njit(cache=True, nogil=True, parallel=True)
def _score_groups(
    values: np.ndarray,
    covariates: np.ndarray,
    n_controls: int,
    members: np.ndarray,
    offsets: np.ndarray,
    among_controls: np.ndarray,
    keys: np.ndarray,
    chosen_groups: np.ndarray,
    pair_offsets: np.ndarray,
    pair_rows: np.ndarray,
    slots: np.ndarray,
    coefficients: np.ndarray,
    n_draws: int,
    batch: int,
    z: np.ndarray,
    fold: np.ndarray,
    se: np.ndarray,
    counts: np.ndarray,
    statistics: np.ndarray,
) -> None:
    """Score the pairs of the chosen groups, a group per thread, in batches.

    Pairs are sorted by group, ``pair_offsets`` holds each group's first pair,
    and every chosen group has at least one pair; ``slots`` gives each pair's
    row of ``statistics``.
    """
    p = covariates.shape[1]
    for k in prange(len(chosen_groups)):
        group = chosen_groups[k]
        first = pair_offsets[group]
        last = pair_offsets[group + 1]
        cells = members[offsets[group] : offsets[group + 1]]
        among = among_controls[group]
        design = _union_design(covariates, n_controls, cells, among)
        treated = _treated_positions(n_controls, cells, among)
        draws = np.empty((n_draws, len(cells)), dtype=np.int32)
        draw_uniform_range(keys[group], 0, 1, len(cells), design.shape[0], draws)
        records = np.empty((design.shape[0], batch * (p + 1)))
        factors = np.empty((batch, p, p))
        for start in range(first, last, batch):
            chosen = np.arange(start, min(start + batch, last))
            _score_batch(
                values,
                coefficients,
                design,
                n_controls,
                cells,
                among,
                treated,
                draws,
                chosen,
                pair_rows,
                slots,
                z,
                fold,
                se,
                counts,
                statistics,
                records,
                factors,
            )


def record_batch(n_union: int, n_covariates: int) -> int:
    """Return how many pairs of a group are scored together.

    Their records, of ``n_covariates + 1`` values per cell of the group's
    ``n_union`` cells, take at most :data:`RECORD_BYTES` per thread, with at
    least 2 and at most 32 pairs together.
    """
    per_pair = 8 * max(n_union, 1) * (n_covariates + 1)
    return max(_BATCH_FLOOR, min(_BATCH_LIMIT, RECORD_BYTES // per_pair))


def _sorted_pairs(
    n_groups: int, pair_groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the order that sorts pairs by group and each group's first pair."""
    groups = np.asarray(pair_groups, dtype=np.int64)
    order = np.argsort(groups, kind="stable")
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(groups, minlength=n_groups), out=offsets[1:])
    return order, offsets


def _run_pairs(
    values: np.ndarray,
    covariates: np.ndarray,
    groups: TreatmentGroups,
    keys: np.ndarray,
    pair_groups: np.ndarray,
    pair_rows: np.ndarray,
    coefficients: np.ndarray,
    n_draws: int,
    *,
    store: bool,
    with_fold: bool,
    draw_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score pairs in any order and return their results in that order.

    With many groups whose random sets fit ``draw_bytes`` side by side, each
    thread takes whole groups; otherwise the groups run one at a time, their
    sets drawn once and their pairs split among the threads. Both give the
    same results. The pairs are scored by group, and each pair's statistics
    go straight to its own row, so they are held once.
    """
    order, pair_offsets = _sorted_pairs(len(groups.among_controls), pair_groups)
    n_pairs = len(order)
    z = np.empty(n_pairs)
    fold = np.empty(n_pairs if with_fold else 0)
    se = np.empty(n_pairs if with_fold else 0)
    counts = np.zeros((0 if store else n_pairs, 2), dtype=np.int64)
    statistics = np.empty((n_pairs if store else 0, n_draws))
    counts_values = count_array(values)
    design_all = np.ascontiguousarray(covariates, dtype=np.float64)
    members = np.ascontiguousarray(groups.members, dtype=np.int64)
    offsets = np.ascontiguousarray(groups.offsets, dtype=np.int64)
    among = np.ascontiguousarray(groups.among_controls, dtype=np.bool_)
    group_keys = np.ascontiguousarray(keys, dtype=np.uint64)
    rows = np.ascontiguousarray(np.asarray(pair_rows, dtype=np.int64)[order])
    n_group_pairs = np.diff(pair_offsets)
    active = np.flatnonzero(n_group_pairs > 0)
    sizes = np.diff(offsets)
    threads = numba.get_num_threads()
    largest = int(sizes[active].max()) if len(active) else 0
    batch = record_batch(groups.n_controls + largest, design_all.shape[1])
    # A group with a batch for every thread keeps them all busy alone; the
    # smaller groups each go to one thread when there are enough of them and
    # their random sets fit side by side.
    small = active[n_group_pairs[active] < batch * threads]
    widest = int(sizes[small].max()) if len(small) else 0
    shared = len(small) >= 2 * threads and threads * 4 * n_draws * widest <= draw_bytes
    if shared:
        # Groups differ in work by their pairs, so threads take them one at a
        # time rather than in fixed blocks.
        with numba.parallel_chunksize(1):
            _score_groups(
                counts_values,
                design_all,
                groups.n_controls,
                members,
                offsets,
                among,
                group_keys,
                small,
                pair_offsets,
                rows,
                order,
                coefficients,
                n_draws,
                batch,
                z,
                fold,
                se,
                counts,
                statistics,
            )
    for group in active:
        if shared and n_group_pairs[group] < batch * threads:
            continue
        cells = members[offsets[group] : offsets[group + 1]]
        design = _union_design(design_all, groups.n_controls, cells, bool(among[group]))
        _score_group(
            counts_values,
            coefficients,
            design,
            groups.n_controls,
            cells,
            bool(among[group]),
            _treated_positions(groups.n_controls, cells, bool(among[group])),
            draw_uniform_sets(group_keys[group], n_draws, len(cells), design.shape[0]),
            np.arange(pair_offsets[group], pair_offsets[group + 1]),
            rows,
            order,
            batch,
            z,
            fold,
            se,
            counts,
            statistics,
        )
    results = []
    for array, used in (
        (z, True),
        (fold, with_fold),
        (se, with_fold),
        (counts, not store),
    ):
        if used:
            restored = np.empty_like(array)
            restored[order] = array
            results.append(restored)
        else:
            results.append(array)
    return results[0], results[1], results[2], results[3], statistics


def resampling_counts(
    values: np.ndarray,
    covariates: np.ndarray,
    groups: TreatmentGroups,
    keys: np.ndarray,
    pair_groups: np.ndarray,
    pair_rows: np.ndarray,
    coefficients: np.ndarray,
    n_draws: int,
    *,
    draw_bytes: int = 1 << 28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Test each pair against random sets, and return its statistic and fold change.

    Each pair's regression is fitted on its group's treatment and control
    cells, starting from the gene's fit on the control cells: Poisson
    coefficients, then the negative binomial size that maximizes the
    likelihood with the Poisson means held fixed. With mean mu and
    dispersion phi = 1 / size, a cell's weight is w = mu / (1 + phi mu) and
    its score a = (y - mu) / (1 + phi mu), and the statistic of a set X of
    cells is sum(a over X) / sqrt(sum(w over X) - v' J^-1 v), where v sums w
    times the covariates over X and J is the information of the fit, the
    score statistic of Barry et al. (2024). Each random set holds as many
    cells as the treatment cells, drawn without replacement from the
    treatment and control cells. The fold change is log2 of the treatment
    cells' counts over their means under the fit, the estimator of the
    SCEPTRE book, with a delta-method standard error from the negative
    binomial variance of the counts and the information of the fit.

    Args:
        values: Counts, genes x cells, the control cells first.
        covariates: The covariates of the cells, the intercept first.
        groups: The groups of treatment cells.
        keys: Each group's key of this round, from :func:`resampling_keys`;
            a group's sets depend only on its key.
        pair_groups: Each pair's group.
        pair_rows: Each pair's gene, a row of ``values``.
        coefficients: The genes' fits on the control cells, from
            :func:`fit_null_models`.
        n_draws: The random sets of each group.
        draw_bytes: The memory that the groups' random sets may take at once.

    Returns:
        Each pair's statistic, NaN for a gene without a fit; per pair, the
        sets with a statistic at most and at least the observed one, pairs x
        2; and each pair's log2 fold change and its standard error.
    """
    z, fold, se, counts, _ = _run_pairs(
        values,
        covariates,
        groups,
        keys,
        pair_groups,
        pair_rows,
        coefficients,
        n_draws,
        store=False,
        with_fold=True,
        draw_bytes=draw_bytes,
    )
    return z, counts, fold, se


def resampled_statistics(
    values: np.ndarray,
    covariates: np.ndarray,
    groups: TreatmentGroups,
    keys: np.ndarray,
    pair_groups: np.ndarray,
    pair_rows: np.ndarray,
    coefficients: np.ndarray,
    n_draws: int,
    *,
    draw_bytes: int = 1 << 28,
) -> np.ndarray:
    """Return the statistic of every random set of each pair, pairs x sets.

    The arguments are those of :func:`resampling_counts`. Each statistic is
    written straight to its pair's row, so the statistics, 8 bytes per pair
    and set, are held once.
    """
    *_, statistics = _run_pairs(
        values,
        covariates,
        groups,
        keys,
        pair_groups,
        pair_rows,
        coefficients,
        n_draws,
        store=True,
        with_fold=False,
        draw_bytes=draw_bytes,
    )
    return statistics


def resampling_p_values(counts: np.ndarray, n_draws: int, side: str) -> np.ndarray:
    """Return the p-values that the random sets give.

    With ``n_draws`` sets, the left p-value is (1 + sets at most the observed
    statistic) / (n_draws + 1) and the right one counts sets at least the
    observed statistic; ``"both"`` doubles the smaller, up to 1.

    Args:
        counts: Per pair, the sets at most and at least the observed
            statistic, from :func:`resampling_counts`.
        n_draws: The number of sets.
        side: ``"both"``, ``"left"`` or ``"right"``.

    Returns:
        Each pair's p-value.
    """
    left = (1.0 + counts[:, 0]) / (n_draws + 1.0)
    right = (1.0 + counts[:, 1]) / (n_draws + 1.0)
    return _sided(left, right, side)


def _sided(left: np.ndarray, right: np.ndarray, side: str) -> np.ndarray:
    """Return the p-value of ``side`` from the two one-sided p-values."""
    if side == "left":
        return left
    if side == "right":
        return right
    both: np.ndarray = np.minimum(1.0, 2.0 * np.minimum(left, right))
    return both


def _span_rows(n_columns: int) -> int:
    """Return how many rows of ``n_columns`` float64 values a span holds, at least 1."""
    return max(1, _SPAN_BYTES // (8 * max(n_columns, 1)))


def _row_spans(values: np.ndarray) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield spans of a matrix's rows and their values as C-contiguous float64.

    The rows of a C-contiguous float64 matrix are views, not copies.
    """
    step = _span_rows(values.shape[1])
    for begin in range(0, values.shape[0], step):
        rows = slice(begin, begin + step)
        yield rows, np.ascontiguousarray(values[rows], dtype=np.float64)


def _working_copy(values: np.ndarray) -> np.ndarray:
    """Return room for one span of a matrix's rows, uninitialized."""
    n_rows, n_columns = values.shape
    return np.empty((min(_span_rows(n_columns), n_rows), n_columns))


def _row_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each row's mean and its second and third central moments, divisor n.

    Each span of rows is centered in one working copy and squared, then
    centered again and cubed, so the moments take one span beyond the values.
    """
    n_rows = values.shape[0]
    mean = np.empty(n_rows)
    second = np.empty(n_rows)
    third = np.empty(n_rows)
    work = _working_copy(values)
    for rows, block in _row_spans(values):
        centered = work[: len(block)]
        middle = block.mean(axis=1, keepdims=True)
        mean[rows] = middle[:, 0]
        np.subtract(block, middle, out=centered)
        np.square(centered, out=centered)
        second[rows] = centered.mean(axis=1)
        np.subtract(block, middle, out=centered)
        np.power(centered, 3, out=centered)
        third[rows] = centered.mean(axis=1)
    return mean, second, third


def _select_ranks(ordered: np.ndarray, ranks: list[int]) -> None:
    """Partially sort each row in place so that it holds its sorted values at ``ranks``.

    ``ranks`` are sorted and distinct. NumPy selects one rank much faster than
    several at once, so the rows are split at the middle rank, and each side
    then selects the ranks it holds.
    """
    if ranks:
        middle = len(ranks) // 2
        rank = ranks[middle]
        ordered.partition(rank, axis=1)
        _select_ranks(ordered[:, :rank], ranks[:middle])
        _select_ranks(
            ordered[:, rank + 1 :], [other - rank - 1 for other in ranks[middle + 1 :]]
        )


def _order_statistics(values: np.ndarray, ranks: list[int]) -> np.ndarray:
    """Return the values at ``ranks`` of each row sorted, rows x ranks.

    A partial sort of one working copy of a span of rows at a time finds the
    same values as a full sort, NaN last, in one span beyond the values.
    """
    found = np.empty((values.shape[0], len(ranks)))
    work = _working_copy(values)
    for rows, block in _row_spans(values):
        ordered = work[: len(block)]
        ordered[...] = block
        _select_ranks(ordered, sorted(set(ranks)))
        found[rows] = ordered[:, ranks]
    return found


def _counts_beyond(statistics: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return each row's statistics at most and at least its observed one, rows x 2.

    The rows are compared a span at a time.
    """
    counts = np.empty((len(observed), 2), dtype=np.int64)
    for rows, block in _row_spans(statistics):
        point = observed[rows, None]
        counts[rows, 0] = (block <= point).sum(axis=1)
        counts[rows, 1] = (block >= point).sum(axis=1)
    return counts


def fit_skew_normals(
    statistics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a skew-normal distribution to each row of statistics by its moments.

    From the mean m, the standard deviation s with divisor n and the skewness
    g, the fit takes |delta| = sqrt(pi/2 |g|^(2/3) / (|g|^(2/3) + ((4 -
    pi)/2)^(2/3))) with the sign of g, shape alpha = delta / sqrt(1 -
    delta^2), scale omega = s / sqrt(1 - 2 delta^2 / pi) and location
    xi = m - omega delta sqrt(2 / pi). A skewness beyond 0.995 in size, more
    than a skew-normal reaches, becomes 0.9 times 0.995 with its sign, as in
    sceptre's fits.

    The moments are taken a span of rows at a time, so besides a few values
    per row the fit holds one working copy of about 1 MiB of the statistics,
    or of one row when a row is larger, never more than the statistics
    themselves; statistics that are not C-contiguous float64 take a float64
    copy of each span too.

    Args:
        statistics: Statistics, pairs x sets.

    Returns:
        Each row's location, scale and shape; a row without spread or with a
        missing statistic gets NaN.
    """
    values = np.asarray(statistics)
    n_rows = values.shape[0]
    location = np.full(n_rows, np.nan)
    scale = np.full(n_rows, np.nan)
    shape = np.full(n_rows, np.nan)
    mean, second, third = _row_moments(values)
    usable = np.isfinite(second) & (second > 0)
    if not usable.any():
        return location, scale, shape
    deviation = np.sqrt(second[usable])
    skewness = third[usable] / deviation**3
    skewness = np.where(
        np.abs(skewness) > _SKEWNESS_LIMIT,
        np.sign(skewness) * _SKEWNESS_FALLBACK,
        skewness,
    )
    power = np.abs(skewness) ** (2.0 / 3.0)
    delta = np.sign(skewness) * np.sqrt(
        np.pi / 2.0 * power / (power + ((4.0 - np.pi) / 2.0) ** (2.0 / 3.0))
    )
    fitted_scale = deviation / np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    location[usable] = mean[usable] - fitted_scale * delta * np.sqrt(2.0 / np.pi)
    scale[usable] = fitted_scale
    shape[usable] = delta / np.sqrt(1.0 - delta**2)
    return location, scale, shape


def skew_normal_tail_ratios(
    statistics: np.ndarray,
    z: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    shape: np.ndarray,
    side: str,
) -> np.ndarray:
    """Return how each fitted tail compares with its statistics' own.

    On the side tested, or with ``side="both"`` the side of the statistics'
    median that ``z`` lies on, the fitted tail probability beyond the
    statistics' 99th percentile, or 1st for the left side, is divided by the
    share of the statistics beyond it. Below 1, the fitted tail is thinner
    there than the statistics', and the p-values it gives farther out are too
    small.

    The percentiles and the median are the statistics at those positions of
    each row sorted, found by a partial sort of a span of rows at a time, so
    besides a few values per row the check holds one working copy of about
    1 MiB of the statistics, or of one row when a row is larger, never more
    than the statistics themselves; statistics that are not C-contiguous
    float64 take a float64 copy of each span too.

    Args:
        statistics: Statistics, pairs x sets.
        z: Each pair's observed statistic.
        location: Each pair's fitted location.
        scale: Each pair's fitted scale.
        shape: Each pair's fitted shape.
        side: ``"both"``, ``"left"`` or ``"right"``.

    Returns:
        Each pair's ratio, NaN without a fit.
    """
    values = np.asarray(statistics)
    n_draws = values.shape[1]
    observed = np.asarray(z, dtype=np.float64)
    median, upper, lower = _order_statistics(
        values,
        [n_draws // 2, math.ceil(0.99 * n_draws) - 1, math.floor(0.01 * n_draws)],
    ).T
    above = np.empty(len(upper))
    below = np.empty(len(lower))
    for rows, block in _row_spans(values):
        above[rows] = (block >= upper[rows, None]).mean(axis=1)
        below[rows] = (block <= lower[rows, None]).mean(axis=1)
    if side == "both":
        right = observed >= median
    else:
        right = np.full(len(observed), side == "right")
    point = np.where(right, upper, lower)
    beyond = np.where(right, above, below)
    standardized = (point - location) / scale
    fitted = np.where(
        right,
        stats.skewnorm.sf(standardized, shape),
        stats.skewnorm.cdf(standardized, shape),
    )
    ratios: np.ndarray = fitted / beyond
    return ratios


def skew_normal_p_values(
    z: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    shape: np.ndarray,
    side: str,
) -> np.ndarray:
    """Return the p-values of statistics under fitted skew-normal distributions.

    The tails are computed in log space, so they stay accurate far beyond the
    random sets.

    Args:
        z: Each pair's statistic.
        location: Each pair's fitted location.
        scale: Each pair's fitted scale.
        shape: Each pair's fitted shape.
        side: ``"both"``, ``"left"`` or ``"right"``.

    Returns:
        Each pair's p-value.
    """
    standardized = (np.asarray(z, dtype=np.float64) - location) / scale
    left = np.exp(stats.skewnorm.logcdf(standardized, shape))
    right = np.exp(stats.skewnorm.logsf(standardized, shape))
    return _sided(left, right, side)


def association_settings(
    positive_controls: object,
    id_column: object,
    integration: object,
    side: object,
    approximation: object,
    n_resamples: object,
    stage_threshold: object,
    min_nonzero: object,
    n_calibration_pairs: object,
    alpha: object,
    seed: object,
    moi: object,
    resampling: object,
) -> tuple[tuple[int, int, int], float, int, int | None, float, int, str]:
    """Check the settings of association tests, as ``associations`` takes them.

    The pipeline recipe checks the same settings before its run starts. The
    pairs, features, samples and covariates are checked against the store.

    Args:
        positive_controls: ``"targets"``, a table of pairs, or None; any
            other string is refused.
        id_column: The guide assay's column of target feature IDs, or None.
        integration: ``"union"`` or ``"singleton"``.
        side: One of :data:`SIDES`.
        approximation: ``"skew_normal"`` or ``"none"``.
        n_resamples: The random sets of the three rounds, a sequence of three
            integers.
        stage_threshold: The p-value at or below which a pair takes the
            second round.
        min_nonzero: The fewest perturbed and control cells that express a
            tested gene.
        n_calibration_pairs: The calibration pairs, or None.
        alpha: The false discovery rate.
        seed: The seed of the random sets and the calibration pairs.
        moi: One of :data:`MOIS`.
        resampling: ``"conditional"``, ``"permutation"``, or None for
            ``"conditional"`` at high MOI and ``"permutation"`` at low MOI.

    Returns:
        The random sets of each round, the stage threshold, ``min_nonzero``,
        the calibration pairs, the false discovery rate and the seed, as
        Python numbers, and the resampling.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If a setting is unknown or out of range, or
            ``resampling="conditional"`` goes with ``moi="low"``.
    """
    # The test against all other cells draws its sets either way; it imports
    # this module, so its resampling names are read when the check runs.
    from cytearc.perturbation.high_moi import RESAMPLING

    if isinstance(positive_controls, str) and positive_controls != "targets":
        raise ValueError(
            "positive_controls must be 'targets', a DataFrame or None, not "
            f"{positive_controls!r}"
        )
    id_column_setting(id_column)
    if integration not in ("union", "singleton"):
        raise ValueError(
            f"integration must be 'union' or 'singleton', not {integration!r}"
        )
    if side not in SIDES:
        raise ValueError(f"side must be 'both', 'left' or 'right', not {side!r}")
    if approximation not in ("skew_normal", "none"):
        raise ValueError(
            f"approximation must be 'skew_normal' or 'none', not {approximation!r}"
        )
    if (
        isinstance(n_resamples, str | bytes)
        or not isinstance(n_resamples, Sequence)
        or len(n_resamples) != 3
    ):
        raise TypeError("n_resamples must be a sequence of three integers")
    # A list, such as a JSON configuration gives, and the equal tuple record
    # the same tests.
    resamples = tuple(n_resamples)
    draws = (
        integer_argument(resamples[0], "n_resamples[0]", minimum=1),
        integer_argument(resamples[1], "n_resamples[1]", minimum=2),
        integer_argument(resamples[2], "n_resamples[2]", minimum=1),
    )
    threshold = float_argument(stage_threshold, "stage_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("stage_threshold must be from 0 to 1")
    floor = integer_argument(min_nonzero, "min_nonzero", minimum=1)
    requested = (
        None
        if n_calibration_pairs is None
        else integer_argument(n_calibration_pairs, "n_calibration_pairs", minimum=0)
    )
    rate = float_argument(alpha, "alpha")
    if not 0 < rate < 1:
        raise ValueError("alpha must be above 0 and below 1")
    resolved_seed = integer_argument(seed, "seed", minimum=0)
    if not isinstance(moi, str) or moi not in MOIS:
        raise ValueError(f"moi must be 'low' or 'high', not {moi!r}")
    if resampling is None:
        resampling = "conditional" if moi == "high" else "permutation"
    elif not isinstance(resampling, str) or resampling not in RESAMPLING:
        raise ValueError(
            "resampling must be 'conditional', 'permutation' or None, not "
            f"{resampling!r}"
        )
    elif resampling == "conditional" and moi == "low":
        raise ValueError(
            "resampling='conditional' goes with moi='high'; the low-MOI test "
            "compares each target with the control cells by permutation"
        )
    return draws, threshold, floor, requested, rate, resolved_seed, resampling


def covariate_columns(covariates: Sequence[str] | None) -> tuple[str, ...]:
    """Check ``covariates``, the cell columns to adjust for, and return them.

    Whether the columns exist and hold numbers is checked against the store.

    Raises:
        TypeError: If ``covariates`` is not a sequence of column names or None.
    """
    if covariates is not None and (
        isinstance(covariates, str)
        or not all(isinstance(column, str) for column in covariates)
    ):
        raise TypeError("covariates must be a sequence of column names or None")
    return tuple(covariates or ())


def calibration_grade(n_false_discoveries: int) -> str:
    """Return the grade of a calibration check with this many false discoveries."""
    for largest, grade in CALIBRATION_GRADES:
        if n_false_discoveries <= largest:
            return grade
    return "severe"


def staged_p_values(
    z: np.ndarray,
    counts: np.ndarray,
    second_statistics: Callable[[np.ndarray, int], np.ndarray],
    third_counts: Callable[[np.ndarray, int], np.ndarray],
    *,
    n_resamples: tuple[int, int, int],
    stage_threshold: float,
    approximation: str,
    side: str,
    statistics_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return each pair's p-value from up to three rounds of random sets.

    The first round's counts give each pair's p-value. A pair whose p-value
    is then at most ``stage_threshold`` is compared with ``n_resamples[1]``
    fresh sets, and with ``approximation="skew_normal"`` a skew-normal fit to
    their statistics gives the p-value, unless its tail beyond their 99th
    percentile, on the observed statistic's side, holds less than half of the
    statistics' own share there; then ``n_resamples[2]`` fresh sets give the
    p-value. With ``approximation="none"``, the second round's sets give it.
    The second round takes its pairs in batches whose statistics fit
    ``statistics_bytes``. Both tests of this package, against control cells
    and against all other cells, take their p-values this way.

    Args:
        z: Each pair's statistic, NaN for a pair without one.
        counts: Per pair, the first round's sets with a statistic at most and
            at least ``z``, pairs x 2.
        second_statistics: Returns the statistics of the second round's sets
            of the pairs it is given, pairs x sets, from their positions and
            the number of sets.
        third_counts: Returns the third round's counts of the pairs it is
            given, pairs x 2, likewise.
        n_resamples: The random sets of the three rounds.
        stage_threshold: The p-value at or below which a pair takes the
            second round.
        approximation: ``"skew_normal"`` or ``"none"``.
        side: ``"both"``, ``"left"`` or ``"right"``.
        statistics_bytes: The memory that the second round's statistics may
            hold at once, at least one pair's.

    Returns:
        Each pair's p-value, NaN without a statistic; the round that gave it;
        and the location, scale and shape of the skew-normal fit that gave
        it, NaN where random sets gave it.
    """
    first_draws, second_draws, third_draws = n_resamples
    p_value = resampling_p_values(counts, first_draws, side)
    p_value[np.isnan(z)] = np.nan
    n_pairs = len(z)
    stage = np.ones(n_pairs, dtype=np.int8)
    location = np.full(n_pairs, np.nan)
    scale = np.full(n_pairs, np.nan)
    shape = np.full(n_pairs, np.nan)
    second = np.flatnonzero(p_value <= stage_threshold)
    batch = max(1, statistics_bytes // (8 * second_draws))
    third: list[np.ndarray] = []
    for begin in range(0, len(second), batch):
        chosen = second[begin : begin + batch]
        statistics = second_statistics(chosen, second_draws)
        empirical = _counts_beyond(statistics, z[chosen])
        p_value[chosen] = resampling_p_values(empirical, second_draws, side)
        stage[chosen] = 2
        if approximation == "skew_normal":
            fit_location, fit_scale, fit_shape = fit_skew_normals(statistics)
            ratios = skew_normal_tail_ratios(
                statistics, z[chosen], fit_location, fit_scale, fit_shape, side
            )
            good = ratios >= _TAIL_RATIO_FLOOR
            p_value[chosen[good]] = skew_normal_p_values(
                z[chosen][good],
                fit_location[good],
                fit_scale[good],
                fit_shape[good],
                side,
            )
            location[chosen[good]] = fit_location[good]
            scale[chosen[good]] = fit_scale[good]
            shape[chosen[good]] = fit_shape[good]
            third.append(chosen[~good])
    if third:
        poor = np.concatenate(third)
        if len(poor):
            counted = third_counts(poor, third_draws)
            p_value[poor] = resampling_p_values(counted, third_draws, side)
            stage[poor] = 3
    return p_value, stage, location, scale, shape


def pair_tests(
    values: np.ndarray,
    covariates: np.ndarray,
    groups: TreatmentGroups,
    pair_groups: np.ndarray,
    pair_rows: np.ndarray,
    *,
    seed: int = 0,
    n_resamples: tuple[int, int, int] = (499, 4999, 24999),
    stage_threshold: float = 0.02,
    approximation: str = "skew_normal",
    side: str = "both",
    statistics_bytes: int = 1 << 28,
    draw_bytes: int = 1 << 28,
) -> PairResults:
    """Test pairs of groups of treatment cells and genes in up to three rounds.

    Each pair's statistic is compared with ``n_resamples[0]`` random sets. A
    pair whose p-value is then at most ``stage_threshold`` is compared with
    ``n_resamples[1]`` fresh sets, and with ``approximation="skew_normal"``
    a skew-normal fit to their statistics gives the p-value, unless its tail
    beyond their 99th percentile, on the observed statistic's side, holds
    less than half of the statistics' own share there; then
    ``n_resamples[2]`` fresh sets give the p-value. With
    ``approximation="none"``, the second round's sets give it.

    Args:
        values: Counts, genes x cells, the control cells first.
        covariates: The covariates of the cells, the intercept first.
        groups: The groups of treatment cells.
        pair_groups: Each pair's group.
        pair_rows: Each pair's gene, a row of ``values``.
        seed: The seed of the random sets.
        n_resamples: The random sets of the three rounds.
        stage_threshold: The p-value at or below which a pair takes the
            second round.
        approximation: ``"skew_normal"`` or ``"none"``.
        side: ``"both"``, ``"left"`` or ``"right"``.
        statistics_bytes: The memory that the second round's statistics may
            hold at once. A pair's statistics, ``8 * n_resamples[1]`` bytes,
            are held whole, so they take at least that. Their counts, fits
            and checks work on about 1 MiB of them at a time, or on one
            pair's when those are larger, in one working copy never larger
            than the statistics, so the second round holds at most twice the
            statistics' memory besides the random sets, the records that
            score them and a few values per pair.
        draw_bytes: The memory that the groups' random sets may take at once.

    Returns:
        Each pair's test.
    """
    rows = np.asarray(pair_rows, dtype=np.int64)
    pair_group = np.asarray(pair_groups, dtype=np.int64)
    coefficients = fit_null_models(values, covariates, groups.n_controls, rows)
    shared = (values, covariates, groups)
    z, counts, fold, se = resampling_counts(
        *shared,
        resampling_keys(seed, groups.codes, 1),
        pair_group,
        rows,
        coefficients,
        n_resamples[0],
        draw_bytes=draw_bytes,
    )
    second_keys = resampling_keys(seed, groups.codes, 2)

    def second_statistics(chosen: np.ndarray, n_draws: int) -> np.ndarray:
        return resampled_statistics(
            *shared,
            second_keys,
            pair_group[chosen],
            rows[chosen],
            coefficients,
            n_draws,
            draw_bytes=draw_bytes,
        )

    def third_counts(chosen: np.ndarray, n_draws: int) -> np.ndarray:
        _, counted, _, _ = resampling_counts(
            *shared,
            resampling_keys(seed, groups.codes, 3),
            pair_group[chosen],
            rows[chosen],
            coefficients,
            n_draws,
            draw_bytes=draw_bytes,
        )
        return counted

    p_value, stage, location, scale, shape = staged_p_values(
        z,
        counts,
        second_statistics,
        third_counts,
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


@njit(cache=True, nogil=True, parallel=True)
def _nonzero_counts(values: np.ndarray, units: np.ndarray, out: np.ndarray) -> None:
    """Count each unit's cells with counts of each feature; see :func:`nonzero_counts`."""
    for feature in prange(values.shape[0]):
        for cell in range(values.shape[1]):
            unit = units[cell]
            if unit >= 0 and values[feature, cell] > 0:
                out[unit, feature] += 1


def nonzero_counts(values: np.ndarray, units: np.ndarray, n_units: int) -> np.ndarray:
    """Return how many cells of each unit hold counts of each feature.

    Args:
        values: Counts, features x cells, float32 or float64.
        units: Each cell's unit from 0, or -1 for a cell in none.
        n_units: The number of units.

    Returns:
        The cells with counts, units x features.
    """
    out = np.zeros((n_units, values.shape[0]), dtype=np.int64)
    _nonzero_counts(
        count_array(values), np.ascontiguousarray(units, dtype=np.int64), out
    )
    return out


def negative_control_groups(
    n_guides: int, group_size: int, n_groups: int, seed: int
) -> list[np.ndarray]:
    """Draw distinct random groups of control guides for the calibration check.

    Groups may share guides, as the SCEPTRE book describes: each is a set of
    ``group_size`` guides drawn at random, and no two groups are the same set.
    When fewer than ``n_groups`` sets exist, every set is a group.

    Args:
        n_guides: The number of control guides.
        group_size: The guides of each group, at most ``n_guides``.
        n_groups: The groups to draw.
        seed: The seed of the draw.

    Returns:
        Each group's guides, as sorted positions among the control guides,
        in the order drawn.
    """
    rng = np.random.default_rng([seed, 1])
    possible = math.comb(n_guides, group_size)
    if possible <= 4 * n_groups:
        every = np.asarray(
            list(itertools.combinations(range(n_guides), group_size)), dtype=np.int64
        ).reshape(possible, group_size)
        chosen = rng.permutation(possible)[: min(n_groups, possible)]
        return [every[k] for k in chosen]
    groups: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(groups) < n_groups:
        members = np.sort(rng.choice(n_guides, size=group_size, replace=False))
        key = tuple(members.tolist())
        if key not in seen:
            seen.add(key)
            groups.append(members)
    return groups


def calibration_pairs(
    passing: np.ndarray, n_pairs: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw negative-control pairs at random among those that pass QC.

    Args:
        passing: Whether each pair of a negative-control group and a feature
            passes QC, groups x features.
        n_pairs: The pairs to draw; all of them when fewer pass.
        seed: The seed of the draw.

    Returns:
        The group and the feature position of each drawn pair, sorted by
        group, then feature.
    """
    candidates = np.flatnonzero(np.asarray(passing, dtype=bool).ravel())
    chosen = np.random.default_rng([seed, 2]).choice(
        candidates, size=min(n_pairs, len(candidates)), replace=False
    )
    chosen.sort()
    n_features = passing.shape[1]
    return chosen // n_features, chosen % n_features


def adjusted_p_values(
    p_values: np.ndarray, tested: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return Benjamini-Hochberg adjusted p-values and the pairs significant at ``alpha``.

    Args:
        p_values: Each pair's p-value.
        tested: Whether each pair was tested; the others get NaN and are not
            significant.
        alpha: The false discovery rate.

    Returns:
        The adjusted p-values and whether each is at most ``alpha``.
    """
    adjusted = np.full(len(p_values), np.nan)
    chosen = np.flatnonzero(np.asarray(tested, dtype=bool) & ~np.isnan(p_values))
    if len(chosen):
        adjusted[chosen] = stats.false_discovery_control(
            np.asarray(p_values, dtype=np.float64)[chosen], method="bh"
        )
    significant = np.zeros(len(p_values), dtype=bool)
    significant[chosen] = adjusted[chosen] <= alpha
    return adjusted, significant
