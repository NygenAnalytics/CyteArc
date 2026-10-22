"""Call cells in unfiltered droplet count matrices with EmptyDrops.

Droplet-based protocols capture far more barcodes than cells. Most barcodes
mark empty droplets that hold only ambient RNA from the cell suspension.
EmptyDrops estimates the ambient profile from barcodes with few counts and
tests every other barcode against it: a barcode whose counts the ambient
profile explains poorly holds a cell. This module implements the method from
its paper and supplementary materials:

- Barcodes with at most ``lower`` counts form the ambient pool. Their summed
  counts per feature are smoothed with Simple Good-Turing estimation, which
  gives every feature counted in some barcode a positive proportion.
- The counts of a barcode follow a Dirichlet-multinomial distribution whose
  expected proportions are the ambient profile and whose precision ``alpha``
  is the maximum likelihood estimate over the ambient pool, the profile taken
  as known.
- A barcode's p-value is ``(R + 1) / (n_iter + 1)``, where ``R`` counts the
  simulated ambient count vectors of its total whose probability is at most
  its own. A simulation adds one molecule at a time and records the
  probability at every tested total, so one simulation serves all totals, as
  Section 3 of the paper's supplementary materials describes.
- Barcodes with at least ``retain`` counts, by default the total at the knee
  of the barcode-rank curve, are cells with p-value 0, and p-values are
  adjusted with the Benjamini-Hochberg procedure over all barcodes above
  ``lower``.

References:
    Lun ATL, Riesenfeld S, Andrews T, Dao TP, Gomes T, participants in the 1st
    Human Cell Atlas Jamboree, Marioni JC. EmptyDrops: distinguishing cells
    from empty droplets in droplet-based single-cell RNA sequencing data.
    Genome Biology 20, 63 (2019). doi:10.1186/s13059-019-1662-y

    Gale WA, Sampson G. Good-Turing frequency estimation without tears.
    Journal of Quantitative Linguistics 2, 217-237 (1995).
    doi:10.1080/09296179508590051
"""

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import sparse
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import brentq
from scipy.special import gammaln

from ..utils.arguments import float_argument, integer_argument
from ..utils.logging import logger

__all__ = [
    "BarcodeRanks",
    "EmptyDropsResult",
    "barcode_ranks",
    "empty_drops",
    "good_turing_proportions",
]

#: Barcodes with at most this many counts form the ambient pool by default.
DEFAULT_LOWER = 100
#: Monte Carlo iterations of the paper's default.
DEFAULT_N_ITER = 10_000

# The barcode-rank curve is a cubic smoothing spline of log10 total against
# log10 rank, fitted to the mean squared error of the distinct totals above
# ``lower`` with this penalty on the integrated squared second derivative.
# Penalties from 1e-7 to 1e-5 place the knee within the cells of real and
# simulated rank curves; weaker ones bend at the few largest barcodes.
_KNEE_SMOOTHING = 1e-6
# A smoothing spline needs this many distinct totals.
_MIN_CURVE_POINTS = 5
# A simulated count vector counts against a barcode when its log-probability
# is at most the barcode's plus this fraction of its magnitude (at least 1),
# so draws as probable as the barcode count however rounding orders the sums.
_TIE_TOLERANCE = 1e-9
# Candidate precisions for the alpha search: 10 per decade from 1e-3 to 1e9.
# A likelihood still rising at the top is the multinomial limit.
_ALPHA_GRID = np.geomspace(1e-3, 1e9, 121)
_ROOT_TOLERANCE = 1e-12
# Rows of an in-memory matrix read at a time, up to 64 MiB of dense counts.
_BLOCK_ROWS = 4096
_BLOCK_BYTES = 64 * 1024**2

# Counter-based random numbers: every draw is a pure function of the seed,
# the iteration, and the draw's position in it, so no state is shared between
# threads. These are the constants of the SplitMix64 generator (Steele, Lea
# and Flood 2014, doi:10.1145/2714064.2660195). Changing them changes every
# p-value, so they are fixed.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_SEED_SALT = np.uint64(0x456D707479447270)
_SHIFT_11 = np.uint64(11)
_SHIFT_27 = np.uint64(27)
_SHIFT_30 = np.uint64(30)
_SHIFT_31 = np.uint64(31)
_UNIT_INTERVAL = 1.0 / 9007199254740992.0  # 2**-53


@dataclass(frozen=True, slots=True, eq=False)
class BarcodeRanks:
    """The barcode-rank curve with its knee and inflection points.

    Barcodes are ranked by decreasing total count; tied totals share their
    average rank. The curve is a cubic smoothing spline of ``log10(total)``
    against ``log10(rank)`` over the distinct totals above ``lower``. The
    inflection point is the total where the fitted curve falls most steeply,
    and the knee is the total, at or above the inflection, where its signed
    curvature ``f'' / (1 + f'^2)^1.5`` is lowest: the bend from the cell
    plateau into the drop towards empty droplets (Lun et al. 2019).

    Attributes:
        rank: Average rank of each distinct total, in increasing order.
        total: The distinct totals, in decreasing order.
        fitted: Fitted total at each rank above ``lower``, and NaN at the rest.
        knee: Total of the barcodes at the knee, or NaN when fewer than five
            distinct totals exceed ``lower``.
        inflection: Total of the barcodes at the inflection point, or NaN.
        lower: Totals at or below this bound were left out of the fit.
    """

    rank: np.ndarray = field(repr=False)
    total: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)
    knee: float
    inflection: float
    lower: int

    def to_pandas(self) -> pd.DataFrame:
        """Return the curve as a table with ``rank``, ``total``, and ``fitted``."""
        return pd.DataFrame(
            {"rank": self.rank, "total": self.total, "fitted": self.fitted}
        )


@dataclass(frozen=True, slots=True, eq=False)
class EmptyDropsResult:
    """EmptyDrops statistics of every barcode.

    Arrays hold one value per barcode, in input order. Barcodes at or below
    ``lower`` are not tested: their statistics are NaN and ``limited`` is
    False. Barcodes at or above ``retain`` are cells with p-value 0.

    Attributes:
        barcodes: Barcode identifiers, or None when the input had none.
        total: Total count of each barcode.
        log_probability: Log-probability of the barcode's counts under the
            ambient Dirichlet-multinomial model.
        p_value: Monte Carlo p-value against the ambient model.
        fdr: Benjamini-Hochberg adjusted p-value over the tested barcodes.
        limited: True where no simulated count vector was as improbable as the
            barcode, so the p-value is the smallest that ``n_iter`` iterations
            give and more iterations could lower it.
        ambient: Good-Turing proportion of each feature in the ambient pool;
            features without counts in any barcode have 0.
        lower: Bound of the ambient pool.
        retain: Barcodes with at least this total were retained.
        knee: Total at the knee of the barcode-rank curve, or NaN.
        inflection: Total at the inflection of the barcode-rank curve, or NaN.
        alpha: Estimated or given Dirichlet-multinomial precision, or None
            when the ambient pool fits a multinomial best.
        n_iter: Monte Carlo iterations.
        random_seed: Seed of the simulations.
        n_ambient: Barcodes with a total from 1 to ``lower``.
    """

    barcodes: np.ndarray | None = field(repr=False)
    total: np.ndarray = field(repr=False)
    log_probability: np.ndarray = field(repr=False)
    p_value: np.ndarray = field(repr=False)
    fdr: np.ndarray = field(repr=False)
    limited: np.ndarray = field(repr=False)
    ambient: np.ndarray = field(repr=False)
    lower: int
    retain: int
    knee: float
    inflection: float
    alpha: float | None
    n_iter: int
    random_seed: int
    n_ambient: int

    def is_cell(self, fdr: float = 0.001) -> np.ndarray:
        """Return True for each barcode whose adjusted p-value is at most ``fdr``.

        Raises:
            TypeError: If ``fdr`` is not a real number.
            ValueError: If ``fdr`` is not greater than 0 and at most 1.
        """
        threshold = validated_fdr(fdr)
        return np.asarray(self.fdr <= threshold, dtype=bool)

    def to_pandas(self) -> pd.DataFrame:
        """Return the tested barcodes with their statistics, one row each."""
        tested = np.flatnonzero(self.total > self.lower)
        index = (
            pd.Index(tested, name="position")
            if self.barcodes is None
            else pd.Index(np.asarray(self.barcodes)[tested], name="barcode")
        )
        return pd.DataFrame(
            {
                "total": self.total[tested],
                "log_probability": self.log_probability[tested],
                "p_value": self.p_value[tested],
                "fdr": self.fdr[tested],
                "limited": self.limited[tested],
            },
            index=index,
        )


def validated_fdr(value: object) -> float:
    """Return an FDR threshold greater than 0 and at most 1 as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not greater than 0 and at most 1.
    """
    resolved = float_argument(value, "fdr")
    if not 0.0 < resolved <= 1.0:
        raise ValueError("fdr must be greater than 0 and at most 1")
    return resolved


def validated_alpha(value: object) -> float | None:
    """Return a given Dirichlet-multinomial precision, or None to estimate it.

    Raises:
        TypeError: If ``value`` is not None or a real number.
        ValueError: If ``value`` is not finite and positive.
    """
    if value is None:
        return None
    resolved = float_argument(value, "alpha")
    if resolved <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def validated_retain(value: object, lower: int) -> int | None:
    """Return a given retention total above ``lower``, or None for the knee.

    Raises:
        TypeError: If ``value`` is not None or an integer.
        ValueError: If ``value`` is not greater than ``lower``.
    """
    if value is None:
        return None
    return integer_argument(value, "retain", minimum=lower + 1)


def _validated_totals(totals: Any) -> np.ndarray:
    values = np.asarray(totals)
    if values.ndim != 1:
        raise ValueError("totals must be one-dimensional")
    if values.dtype.kind not in "iuf":
        raise TypeError("totals must be real numbers")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("totals must be finite and non-negative")
    return values


def barcode_ranks(totals: Any, *, lower: int = DEFAULT_LOWER) -> BarcodeRanks:
    """Locate the knee and inflection of the barcode-rank curve.

    Barcodes are ranked by decreasing total. The function fits a cubic
    smoothing spline to ``log10(total)`` against ``log10(rank)`` over the
    distinct totals above ``lower`` and evaluates its derivatives at those
    totals. The inflection point is where the curve falls most steeply; the
    knee is where the signed curvature is lowest among the totals at or above
    the inflection, following the knee point definition of Lun et al. (2019).

    Args:
        totals: Total count of each barcode, such as a raw matrix's
            ``RNA_nCounts``.
        lower: Totals at or below this value, mostly empty droplets, are left
            out of the fit.

    Returns:
        The curve, with NaN for the knee and inflection when fewer than five
        distinct totals exceed ``lower``.

    Raises:
        TypeError: If ``totals`` is not numeric or ``lower`` not an integer.
        ValueError: If ``totals`` is not one-dimensional, finite, and
            non-negative, or ``lower`` is negative.
    """
    values = _validated_totals(totals)
    lower = integer_argument(lower, "lower", minimum=0)
    distinct, counts = np.unique(values, return_counts=True)
    distinct = distinct[::-1]
    counts = counts[::-1]
    # Barcodes with a larger total precede each distinct total.
    above = np.cumsum(counts) - counts
    rank = above + (counts + 1) / 2.0
    fitted = np.full(len(distinct), np.nan)
    knee = inflection = math.nan
    fit = distinct > lower
    if np.count_nonzero(fit) >= _MIN_CURVE_POINTS:
        x = np.log10(rank[fit])
        y = np.log10(distinct[fit])
        spline = make_smoothing_spline(
            x, y, w=np.full(len(x), 1.0 / len(x)), lam=_KNEE_SMOOTHING
        )
        fitted[fit] = 10.0 ** spline(x)
        slope = spline.derivative(1)(x)
        curvature = spline.derivative(2)(x) / (1.0 + slope**2) ** 1.5
        steepest = int(np.argmin(slope))
        bend = int(np.argmin(curvature[: steepest + 1]))
        knee = float(distinct[fit][bend])
        inflection = float(distinct[fit][steepest])
    return BarcodeRanks(
        rank=rank,
        total=distinct,
        fitted=fitted,
        knee=knee,
        inflection=inflection,
        lower=lower,
    )


def good_turing_proportions(counts: Any) -> np.ndarray:
    """Estimate species proportions from counts with Simple Good-Turing.

    Implements the Simple Good-Turing method of Gale and Sampson (1995). For
    each count ``r`` observed for ``N_r`` species, the averaging transform
    ``Z_r = N_r / (0.5 * (t - q))``, with ``q`` and ``t`` the neighbouring
    observed counts, is fitted by least squares as ``log Z = a + b log r``.
    The adjusted count ``r*`` is the Turing estimate ``(r + 1) N_{r+1} / N_r``
    while it differs from the smoothed estimate ``(r + 1) (1 + 1/r)^b`` by
    more than 1.96 standard deviations, and the smoothed estimate from the
    first count where it does not, or where ``N_{r+1}`` is 0. Observed species
    share ``1 - N_1 / N`` in proportion to ``r*``, and species with a count
    of 0 share ``N_1 / N`` equally. When every species was observed, the
    proportions are scaled to sum to 1.

    Args:
        counts: Non-negative integer count of each species.

    Returns:
        A float64 proportion for each species, summing to 1.

    Raises:
        TypeError: If ``counts`` is not an array of integers.
        ValueError: If ``counts`` is not one-dimensional and non-negative, or
            holds fewer than two distinct positive counts.
    """
    values = np.asarray(counts)
    if values.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if values.dtype.kind not in "iu":
        raise TypeError("counts must be integers")
    values = values.astype(np.int64)
    if np.any(values < 0):
        raise ValueError("counts must be non-negative")
    observed = values > 0
    seen, frequency = np.unique(values[observed], return_counts=True)
    if len(seen) < 2:
        raise ValueError(
            "Good-Turing smoothing needs at least two distinct positive counts"
        )
    total = float(values.sum())
    unseen_share = float(frequency[0]) / total if seen[0] == 1 else 0.0
    previous = np.concatenate(([0], seen[:-1]))
    following = np.concatenate((seen[1:], [2 * seen[-1] - previous[-1]]))
    averaged = frequency / (0.5 * (following - previous))
    slope = float(np.polyfit(np.log(seen), np.log(averaged), 1)[0])
    if slope >= -1.0:
        logger.warning(
            f"The Good-Turing frequency fit has slope {slope:.3f}, which is not "
            "below -1, so its smoothed proportions may be unreliable."
        )
    frequency_of = dict(zip(seen.tolist(), frequency.tolist(), strict=True))
    adjusted = np.empty(len(seen))
    use_turing = True
    for index, (count, n_count) in enumerate(
        zip(seen.tolist(), frequency.tolist(), strict=True)
    ):
        smoothed = (count + 1) * (1.0 + 1.0 / count) ** slope
        n_next = frequency_of.get(count + 1, 0)
        if use_turing and n_next > 0:
            turing = (count + 1) * n_next / n_count
            spread = 1.96 * math.sqrt(
                (count + 1) ** 2 * (n_next / n_count**2) * (1.0 + n_next / n_count)
            )
            if abs(smoothed - turing) > spread:
                adjusted[index] = turing
                continue
        use_turing = False
        adjusted[index] = smoothed
    per_count = (1.0 - unseen_share) * adjusted / float(np.sum(frequency * adjusted))
    proportions = np.zeros(len(values))
    proportions[observed] = per_count[np.searchsorted(seen, values[observed])]
    n_unseen = len(values) - int(np.count_nonzero(observed))
    if n_unseen:
        proportions[~observed] = unseen_share / n_unseen
    else:
        proportions /= proportions.sum()
    return proportions


# Ambient pool -------------------------------------------------------------


@njit(cache=True, nogil=True)
def _add_ambient_count(
    feature: int,
    count: np.int64,
    is_ambient: bool,
    lower: int,
    ambient: np.ndarray,
    expressed: np.ndarray,
    histogram: np.ndarray,
) -> None:
    """Record one positive count of a barcode.

    The feature is expressed. A count of an ambient barcode also adds to the
    feature's ambient sum and to ``histogram[feature, count]``; a count above
    ``lower`` cannot belong to one, whose row total then disagrees.
    """
    expressed[feature] = True
    if is_ambient and count <= lower:
        ambient[feature] += count
        histogram[feature, count] += 1


@njit(cache=True, nogil=True)
def _accumulate_ambient(
    block: np.ndarray,
    totals: np.ndarray,
    lower: int,
    ambient: np.ndarray,
    expressed: np.ndarray,
    histogram: np.ndarray,
) -> int:
    """Add a dense block of barcodes to the ambient pool.

    Every barcode marks the features it counts as expressed, and a barcode
    with at most ``lower`` counts adds them to ``ambient`` and ``histogram``.
    Returns the row of the first barcode whose counts do not add up to its
    total, or -1.
    """
    n_rows, n_features = block.shape
    for row in range(n_rows):
        is_ambient = totals[row] <= lower
        row_total = np.int64(0)
        for feature in range(n_features):
            count = np.int64(block[row, feature])
            if count == 0:
                continue
            row_total += count
            _add_ambient_count(
                feature, count, is_ambient, lower, ambient, expressed, histogram
            )
        if row_total != totals[row]:
            return row
    return -1


@njit(cache=True, nogil=True)
def _accumulate_ambient_sparse(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    totals: np.ndarray,
    lower: int,
    ambient: np.ndarray,
    expressed: np.ndarray,
    histogram: np.ndarray,
) -> int:
    """Add a block of barcodes in canonical CSR form to the ambient pool.

    The same as ``_accumulate_ambient`` for rows without repeated features.
    """
    for row in range(indptr.shape[0] - 1):
        is_ambient = totals[row] <= lower
        row_total = np.int64(0)
        for entry in range(indptr[row], indptr[row + 1]):
            count = np.int64(data[entry])
            if count == 0:
                continue
            row_total += count
            _add_ambient_count(
                indices[entry], count, is_ambient, lower, ambient, expressed, histogram
            )
        if row_total != totals[row]:
            return row
    return -1


@dataclass(frozen=True, slots=True, eq=False)
class _AmbientPool:
    """What one pass over the barcodes learns about the ambient pool.

    ``tested`` holds the count rows of the barcodes above ``lower`` as CSR
    blocks when the pass was asked to keep them, and is None otherwise.
    """

    counts: np.ndarray
    expressed: np.ndarray
    histogram: np.ndarray
    totals: np.ndarray
    tested: list[sparse.csr_matrix] | None

    @property
    def n_barcodes(self) -> int:
        return len(self.totals)


def _collect_ambient(
    blocks: Iterable[Any],
    totals: np.ndarray,
    n_features: int,
    lower: int,
    *,
    keep_tested: bool,
) -> _AmbientPool:
    """Accumulate the ambient pool over blocks of the barcodes with counts.

    ``blocks`` yields, in order, the count rows of the barcodes whose totals
    are ``totals``, as dense arrays or canonical CSR matrices. With
    ``keep_tested``, the rows of the barcodes above ``lower`` are kept in CSR
    form, so they need not be read again.

    Raises:
        RuntimeError: If a barcode's counts do not add up to its total.
    """
    ambient = np.zeros(n_features, dtype=np.int64)
    expressed = np.zeros(n_features, dtype=np.bool_)
    histogram = np.zeros((n_features, lower + 1), dtype=np.int64)
    tested: list[sparse.csr_matrix] | None = [] if keep_tested else None
    offset = 0
    for block in blocks:
        stop = offset + block.shape[0]
        block_totals = totals[offset:stop]
        sums = (ambient, expressed, histogram)
        if sparse.issparse(block):
            row = _accumulate_ambient_sparse(
                block.indptr, block.indices, block.data, block_totals, lower, *sums
            )
        else:
            row = _accumulate_ambient(block, block_totals, lower, *sums)
        if row >= 0:
            raise RuntimeError(
                f"The counts of barcode {offset + row} add up to "
                f"{int(block[row].sum())}, not its recorded total "
                f"{int(totals[offset + row])}"
            )
        if tested is not None:
            tested.append(sparse.csr_matrix(block[block_totals > lower]))
        offset = stop
    if offset != len(totals):
        raise RuntimeError(
            f"Read {offset} barcodes for the ambient pool, expected {len(totals)}"
        )
    ambient_totals = totals[totals <= lower]
    return _AmbientPool(ambient, expressed, histogram, ambient_totals, tested)


def _ambient_proportions(pool: _AmbientPool) -> np.ndarray:
    """Return the Good-Turing ambient profile over the expressed features."""
    if not pool.counts.any():
        raise ValueError(
            "No barcode has a total from 1 to lower, so there is no ambient pool "
            "to test against. Import the barcodes with low totals, for example "
            "with CrH5Reader(..., is_filtered=False, filtering_cutoff=0), or "
            "raise lower."
        )
    proportions = np.zeros(len(pool.counts))
    proportions[pool.expressed] = good_turing_proportions(pool.counts[pool.expressed])
    return proportions


@dataclass(frozen=True, slots=True, eq=False)
class _AlphaTerms:
    """Sufficient statistics of the ambient likelihood in ``alpha``.

    Up to terms constant in ``alpha``, the Dirichlet-multinomial
    log-likelihood of the ambient barcodes, with proportions ``p``, is
    ``sum(weight * log(p[feature] + step / alpha)) -
    sum(barcodes * log1p(total_step / alpha))``: ``weight`` counts the
    barcodes whose count of ``feature`` exceeds ``step``, and ``barcodes`` the
    barcodes whose total exceeds ``total_step``.
    """

    proportion: np.ndarray
    step: np.ndarray
    weight: np.ndarray
    total_step: np.ndarray
    barcodes: np.ndarray

    def score(self, log_alpha: float) -> float:
        """Return the derivative of the log-likelihood in ``log(alpha)``."""
        alpha = math.exp(log_alpha)
        return float(
            np.dot(self.barcodes, self.total_step / (alpha + self.total_step))
            - np.dot(self.weight, self.step / (alpha * self.proportion + self.step))
        )

    def log_likelihood(self, alpha: float) -> float:
        """Return the log-likelihood up to a constant; ``inf`` is multinomial."""
        return float(
            np.dot(self.weight, np.log(self.proportion + self.step / alpha))
            - np.dot(self.barcodes, np.log1p(self.total_step / alpha))
        )


def _alpha_terms(pool: _AmbientPool, proportions: np.ndarray) -> _AlphaTerms:
    lower = pool.histogram.shape[1] - 1
    # at_least[f, c] counts the barcodes with at least c counts of feature f,
    # so column c + 1 counts those that exceed c.
    at_least = np.cumsum(pool.histogram[:, ::-1], axis=1)[:, ::-1]
    features, column = np.nonzero(at_least[:, 2:])
    weight = at_least[:, 2:][features, column]
    by_total = np.bincount(pool.totals.astype(np.int64), minlength=lower + 1)
    total_at_least = np.cumsum(by_total[::-1])[::-1]
    barcodes = total_at_least[2:]
    return _AlphaTerms(
        proportion=proportions[features],
        step=(column + 1).astype(np.float64),
        weight=weight.astype(np.float64),
        total_step=np.arange(1, len(barcodes) + 1, dtype=np.float64),
        barcodes=barcodes.astype(np.float64),
    )


def _estimate_alpha(terms: _AlphaTerms) -> float:
    """Return the maximum likelihood precision, ``inf`` for the multinomial.

    The score is evaluated on a grid of precisions. Each fall of the score
    from positive to at most zero brackets a local maximum, refined with
    Brent's method; a score still non-negative at the top of the grid adds
    the multinomial limit, and one at most zero at the bottom adds the
    smallest precision. The candidate of largest likelihood wins, and the
    multinomial wins ties.
    """
    log_grid = np.log(_ALPHA_GRID)
    scores = np.array([terms.score(value) for value in log_grid])
    candidates: list[float] = []
    if scores[-1] >= 0.0:
        candidates.append(math.inf)
    for index in np.flatnonzero((scores[:-1] > 0.0) & (scores[1:] <= 0.0)):
        root = brentq(
            terms.score,
            log_grid[index],
            log_grid[index + 1],
            xtol=_ROOT_TOLERANCE,
        )
        candidates.append(math.exp(root))
    if scores[0] <= 0.0:
        candidates.append(float(_ALPHA_GRID[0]))
    return max(candidates, key=terms.log_likelihood)


# Candidate likelihoods ----------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Sampler:
    """The ambient proportions in the forms that the simulation reads."""

    log_proportion: np.ndarray
    inverse_scale: np.ndarray
    alpha: float
    multinomial: bool
    cumulative: np.ndarray
    guide: np.ndarray


def _sampler(proportions: np.ndarray, alpha: float) -> _Sampler:
    multinomial = not math.isfinite(alpha)
    positive = proportions > 0
    log_proportion = np.full(len(proportions), -np.inf)
    log_proportion[positive] = np.log(proportions[positive])
    inverse_scale = np.zeros(len(proportions))
    if not multinomial:
        inverse_scale[positive] = 1.0 / (alpha * proportions[positive])
    # The last feature of positive proportion ends the cumulative proportions
    # at exactly 1, so every draw below 1 selects a feature of positive
    # proportion; the change is within the rounding of the sum.
    cumulative = np.cumsum(proportions)
    cumulative[np.flatnonzero(positive)[-1] :] = 1.0
    steps = np.arange(len(proportions)) / len(proportions)
    guide = np.searchsorted(cumulative, steps, side="right").astype(np.int64)
    return _Sampler(
        log_proportion=log_proportion,
        inverse_scale=inverse_scale,
        alpha=alpha,
        multinomial=multinomial,
        cumulative=cumulative,
        guide=guide,
    )


@njit(cache=True, nogil=True)
def _count_term(
    feature: int,
    count: np.int64,
    log_proportion: np.ndarray,
    inverse_scale: np.ndarray,
) -> float:
    """Return the term of one positive count in a barcode's log-probability.

    For counts ``y`` with total ``t``, the Dirichlet-multinomial
    log-probability is ``lgamma(t + 1) - sum(log1p(j / alpha), 0 < j < t)``
    plus, for each counted feature, ``y log p - lgamma(y + 1) +
    sum(log1p(j / (alpha p)), 0 < j < y)``. ``inverse_scale`` holds ``1 /
    (alpha p)``, 0 for the multinomial and for features of proportion 0,
    whose term is then minus infinity.
    """
    term = count * log_proportion[feature] - math.lgamma(count + 1.0)
    scale = inverse_scale[feature]
    if scale != 0.0:
        for step in range(1, count):
            term += math.log1p(step * scale)
    return float(term)


@njit(cache=True, nogil=True)
def _barcode_terms(
    block: np.ndarray,
    log_proportion: np.ndarray,
    inverse_scale: np.ndarray,
    feature_part: np.ndarray,
    totals: np.ndarray,
) -> None:
    """Write the feature part of each dense barcode row and its total."""
    n_rows, n_features = block.shape
    for row in range(n_rows):
        part = 0.0
        total = np.int64(0)
        for feature in range(n_features):
            count = np.int64(block[row, feature])
            if count == 0:
                continue
            total += count
            part += _count_term(feature, count, log_proportion, inverse_scale)
        feature_part[row] = part
        totals[row] = total


@njit(cache=True, nogil=True)
def _barcode_terms_sparse(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    log_proportion: np.ndarray,
    inverse_scale: np.ndarray,
    feature_part: np.ndarray,
    totals: np.ndarray,
) -> None:
    """Write the feature part of each canonical CSR barcode row and its total."""
    for row in range(indptr.shape[0] - 1):
        part = 0.0
        total = np.int64(0)
        for entry in range(indptr[row], indptr[row + 1]):
            count = np.int64(data[entry])
            if count == 0:
                continue
            total += count
            part += _count_term(indices[entry], count, log_proportion, inverse_scale)
        feature_part[row] = part
        totals[row] = total


def _feature_parts(
    blocks: Iterable[Any],
    n_barcodes: int,
    sampler: _Sampler,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the feature parts of the tested barcodes and their totals."""
    parts = np.empty(n_barcodes)
    totals = np.empty(n_barcodes, dtype=np.int64)
    offset = 0
    for block in blocks:
        stop = offset + block.shape[0]
        outputs = (
            sampler.log_proportion,
            sampler.inverse_scale,
            parts[offset:stop],
            totals[offset:stop],
        )
        if sparse.issparse(block):
            _barcode_terms_sparse(block.indptr, block.indices, block.data, *outputs)
        else:
            _barcode_terms(block, *outputs)
        offset = stop
    if offset != n_barcodes:
        raise RuntimeError(f"Read {offset} barcodes to test, expected {n_barcodes}")
    return parts, totals


def _total_parts(totals: np.ndarray, alpha: float) -> np.ndarray:
    """Return ``lgamma(t + 1) - sum(log1p(j / alpha), j < t)`` for each total."""
    largest = int(totals.max())
    steps = np.zeros(largest + 1)
    if math.isfinite(alpha):
        steps[2:] = np.cumsum(np.log1p(np.arange(1, largest) / alpha))
    parts: np.ndarray = gammaln(totals + 1.0) - steps[totals]
    return parts


# Monte Carlo p-values -----------------------------------------------------


@njit(cache=True, nogil=True)
def _mix64(value: np.uint64) -> np.uint64:
    """Return the SplitMix64 finaliser of a 64-bit unsigned integer."""
    mixed = (value ^ (value >> _SHIFT_30)) * _MIX_1
    mixed = (mixed ^ (mixed >> _SHIFT_27)) * _MIX_2
    return mixed ^ (mixed >> _SHIFT_31)


@njit(cache=True, nogil=True)
def _iteration_key(seed: np.uint64, iteration: int) -> np.uint64:
    """Return the key of the random stream of one iteration."""
    return _mix64(_mix64(seed ^ _SEED_SALT) ^ _mix64(np.uint64(iteration) + _GOLDEN))


@njit(cache=True, nogil=True)
def _uniform(key: np.uint64, counter: int) -> float:
    """Return draw ``counter`` of stream ``key``, uniform on [0, 1)."""
    mixed = _mix64(key + np.uint64(counter) * _GOLDEN)
    return np.float64(mixed >> _SHIFT_11) * _UNIT_INTERVAL


@njit(cache=True, nogil=True)
def _draw_feature(cumulative: np.ndarray, guide: np.ndarray, draw: float) -> int:
    """Return the first feature whose cumulative proportion exceeds the draw.

    ``draw`` is uniform on [0, 1), and ``cumulative`` ends at exactly 1 from
    the last feature of positive proportion on, so the answer is that
    feature or an earlier one. ``guide`` holds, for each of its equal steps
    of [0, 1), the first feature whose cumulative proportion exceeds the
    step; the search starts there and corrects any rounding of the step.
    """
    feature = guide[min(int(draw * guide.shape[0]), guide.shape[0] - 1)]
    while feature > 0 and cumulative[feature - 1] > draw:
        feature -= 1
    while cumulative[feature] <= draw:
        feature += 1
    return int(feature)


@njit(cache=True, nogil=True)
def _next_feature(
    key: np.uint64,
    counter: int,
    drawn_count: int,
    alpha: float,
    multinomial: bool,
    cumulative: np.ndarray,
    guide: np.ndarray,
    drawn: np.ndarray,
) -> int:
    """Draw the feature of the next ambient molecule with the Polya urn.

    After ``drawn_count`` molecules, the next one copies a uniformly chosen
    earlier molecule with probability ``drawn_count / (drawn_count + alpha)``
    and is otherwise drawn from the ambient proportions, which samples the
    Dirichlet-multinomial one molecule at a time (Blackwell and MacQueen
    1973): the paper's draw of proportions from the Dirichlet followed by
    multinomial sampling gives the same distribution. Uses draws ``counter``
    and ``counter + 1``.
    """
    choice = _uniform(key, counter)
    draw = _uniform(key, counter + 1)
    if not multinomial and choice * (drawn_count + alpha) < drawn_count:
        return int(drawn[min(int(draw * drawn_count), drawn_count - 1)])
    return _draw_feature(cumulative, guide, draw)


@njit(cache=True, nogil=True)
def _first_at_least(values: np.ndarray, start: int, stop: int, target: float) -> int:
    """Return the first position in sorted ``values[start:stop]`` at least ``target``."""
    low = start
    high = stop
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True, nogil=True)
def _simulate_iteration(
    iteration: int,
    seed: np.uint64,
    group_totals: np.ndarray,
    group_starts: np.ndarray,
    thresholds: np.ndarray,
    log_proportion: np.ndarray,
    inverse_scale: np.ndarray,
    alpha: float,
    multinomial: bool,
    cumulative: np.ndarray,
    guide: np.ndarray,
    log_counts: np.ndarray,
    hits: np.ndarray,
    drawn: np.ndarray,
    feature_counts: np.ndarray,
) -> None:
    """Simulate one ambient count vector and record it against each total.

    Molecules are added one at a time up to the largest tested total. The
    feature part of the log-probability gains ``log p - log(z + 1) + log1p(z
    / (alpha p))`` for a molecule of a feature with ``z`` earlier molecules.
    At each tested total, ``hits`` gains one at the first barcode of that
    total whose threshold is at least the simulated feature part; the
    barcodes from there to the group's end count the draw.
    """
    # A compiled call returns a Python integer to Python callers.
    key = np.uint64(_iteration_key(seed, iteration))
    group = 0
    part = 0.0
    largest = group_totals[group_totals.shape[0] - 1]
    for drawn_count in range(largest):
        feature = _next_feature(
            key,
            2 * drawn_count,
            drawn_count,
            alpha,
            multinomial,
            cumulative,
            guide,
            drawn,
        )
        drawn[drawn_count] = feature
        earlier = feature_counts[feature]
        part += (
            log_proportion[feature]
            - log_counts[earlier + 1]
            + math.log1p(earlier * inverse_scale[feature])
        )
        feature_counts[feature] = earlier + 1
        if group_totals[group] == drawn_count + 1:
            position = _first_at_least(
                thresholds, group_starts[group], group_starts[group + 1], part
            )
            hits[position + group] += 1
            group += 1
    for index in range(largest):
        feature_counts[drawn[index]] = 0


@njit(cache=True, nogil=True, parallel=True)
def _simulate(
    n_iter: int,
    seed: np.uint64,
    group_totals: np.ndarray,
    group_starts: np.ndarray,
    thresholds: np.ndarray,
    log_proportion: np.ndarray,
    inverse_scale: np.ndarray,
    alpha: float,
    multinomial: bool,
    cumulative: np.ndarray,
    guide: np.ndarray,
    log_counts: np.ndarray,
    hits: np.ndarray,
    drawn: np.ndarray,
    feature_counts: np.ndarray,
) -> None:
    """Run the iterations, worker ``w`` taking ``w, w + W, w + 2W, ...``.

    Each worker writes its own rows of ``hits``, ``drawn``, and
    ``feature_counts``, and the draws of an iteration depend only on the seed
    and the iteration, so the summed hits do not depend on the worker count.
    """
    n_workers = hits.shape[0]
    for worker in prange(n_workers):
        for iteration in range(worker, n_iter, n_workers):
            _simulate_iteration(
                iteration,
                seed,
                group_totals,
                group_starts,
                thresholds,
                log_proportion,
                inverse_scale,
                alpha,
                multinomial,
                cumulative,
                guide,
                log_counts,
                hits[worker],
                drawn[worker],
                feature_counts[worker],
            )


@njit(cache=True, nogil=True)
def _simulated_features(
    iteration: int,
    seed: np.uint64,
    n_molecules: int,
    alpha: float,
    multinomial: bool,
    cumulative: np.ndarray,
    guide: np.ndarray,
) -> np.ndarray:
    """Return the features of the molecules that one iteration draws."""
    # A compiled call returns a Python integer to Python callers.
    key = np.uint64(_iteration_key(seed, iteration))
    drawn = np.empty(n_molecules, dtype=np.int64)
    for drawn_count in range(n_molecules):
        drawn[drawn_count] = _next_feature(
            key,
            2 * drawn_count,
            drawn_count,
            alpha,
            multinomial,
            cumulative,
            guide,
            drawn,
        )
    return drawn


def _monte_carlo_p_values(
    totals: np.ndarray,
    feature_parts: np.ndarray,
    log_probabilities: np.ndarray,
    sampler: _Sampler,
    *,
    n_iter: int,
    random_seed: int,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Monte Carlo p-values and limited flags of the tested barcodes.

    A simulated count vector counts against a barcode of the same total when
    its log-probability is at most the barcode's, within ``_TIE_TOLERANCE``.
    Both share the terms of the total, so the feature parts are compared.
    """
    import numba

    tolerance = np.where(
        np.isfinite(log_probabilities),
        _TIE_TOLERANCE * np.maximum(1.0, np.abs(log_probabilities)),
        0.0,
    )
    thresholds = feature_parts + tolerance
    order = np.lexsort((thresholds, totals))
    sorted_totals = totals[order]
    group_totals, group_sizes = np.unique(sorted_totals, return_counts=True)
    group_starts = np.concatenate(([0], np.cumsum(group_sizes))).astype(np.int64)
    n_groups = len(group_totals)
    largest = int(group_totals[-1])
    available = int(getattr(numba.config, "NUMBA_NUM_THREADS"))
    workers = max(1, min(int(nthreads), available, n_iter))
    hits = np.zeros((workers, len(totals) + n_groups), dtype=np.int64)
    drawn = np.zeros((workers, largest), dtype=np.int64)
    feature_counts = np.zeros((workers, len(sampler.log_proportion)), dtype=np.int64)
    # log_counts[k] = log(k) for the molecules of one feature; 0 is unused.
    log_counts = np.zeros(largest + 1)
    log_counts[1:] = np.log(np.arange(1, largest + 1))
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(workers)
    try:
        _simulate(
            n_iter,
            np.uint64(random_seed % 2**64),
            group_totals.astype(np.int64),
            group_starts,
            thresholds[order],
            sampler.log_proportion,
            sampler.inverse_scale,
            sampler.alpha,
            sampler.multinomial,
            sampler.cumulative,
            sampler.guide,
            log_counts,
            hits,
            drawn,
            feature_counts,
        )
    finally:
        numba.set_num_threads(previous_threads)
    summed = hits.sum(axis=0)
    group_of = np.repeat(np.arange(n_groups), group_sizes)
    # The slot of sorted barcode q is q plus its group; a draw that reaches
    # slot s counts for every barcode from s to its group's end.
    at_position = summed[np.arange(len(totals)) + group_of]
    running = np.cumsum(at_position)
    before_group = np.concatenate(([0], running))[group_starts[:-1]]
    exceeded = np.empty(len(totals), dtype=np.int64)
    exceeded[order] = running - np.repeat(before_group, group_sizes)
    p_values = (exceeded + 1.0) / (n_iter + 1.0)
    return p_values, exceeded == 0


# Orchestration -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmptyDropsSettings:
    """Validated settings of one EmptyDrops run."""

    lower: int
    retain: int | None
    n_iter: int
    alpha: float | None
    random_seed: int


def validated_settings(
    *,
    lower: object,
    retain: object,
    n_iter: object,
    alpha: object,
    random_seed: object,
) -> EmptyDropsSettings:
    """Validate EmptyDrops settings before any data is read.

    Raises:
        TypeError: If a setting has the wrong type.
        ValueError: If a setting is out of range.
    """
    resolved_lower = integer_argument(lower, "lower", minimum=1)
    return EmptyDropsSettings(
        lower=resolved_lower,
        retain=validated_retain(retain, resolved_lower),
        n_iter=integer_argument(n_iter, "n_iter", minimum=1),
        alpha=validated_alpha(alpha),
        random_seed=integer_argument(random_seed, "random_seed", minimum=0),
    )


def run_empty_drops(
    totals: np.ndarray,
    read_rows: Callable[[np.ndarray], Iterable[Any]],
    n_features: int,
    settings: EmptyDropsSettings,
    *,
    nthreads: int,
    barcodes: np.ndarray | None = None,
    keep_tested: bool = False,
) -> EmptyDropsResult:
    """Run EmptyDrops over barcodes whose counts ``read_rows`` streams.

    ``read_rows(positions)`` yields the count rows of the barcodes at the
    given sorted positions, in order and in blocks of bounded size, as dense
    arrays or canonical CSR matrices. It reads every barcode with counts to
    collect the ambient pool, and then the barcodes above ``lower`` to
    compute their probabilities, unless ``keep_tested`` keeps their rows from
    the first read in CSR form.

    Raises:
        ValueError: If no barcode exceeds ``lower``, the ambient pool cannot
            be smoothed, or the knee cannot be located for a default
            ``retain``.
        RuntimeError: If a barcode's counts do not add up to its total.
    """
    from ..features.statistical import adjust_pvalues

    totals = np.asarray(totals, dtype=np.int64)
    lower = settings.lower
    tested = np.flatnonzero(totals > lower)
    if not len(tested):
        raise ValueError(
            f"No barcode has more than lower={lower} counts, so no barcode can "
            "be tested; lower lower to test barcodes with fewer counts."
        )
    ranks = barcode_ranks(totals, lower=lower)
    if settings.retain is not None:
        retain = settings.retain
    elif math.isnan(ranks.knee):
        raise ValueError(
            "The barcode-rank curve has fewer than five distinct totals above "
            f"lower={lower}, so its knee cannot be located; pass retain."
        )
    else:
        retain = int(ranks.knee)
    with_counts = np.flatnonzero(totals > 0)
    pool = _collect_ambient(
        read_rows(with_counts),
        totals[with_counts],
        n_features,
        lower,
        keep_tested=keep_tested,
    )
    proportions = _ambient_proportions(pool)
    alpha = (
        _estimate_alpha(_alpha_terms(pool, proportions))
        if settings.alpha is None
        else settings.alpha
    )
    sampler = _sampler(proportions, alpha)
    feature_parts, tested_totals = _feature_parts(
        read_rows(tested) if pool.tested is None else pool.tested,
        len(tested),
        sampler,
    )
    if not np.array_equal(tested_totals, totals[tested]):
        raise RuntimeError("The counts of the tested barcodes changed while read")
    log_probabilities = _total_parts(tested_totals, alpha) + feature_parts
    p_values = np.zeros(len(tested))
    limited = np.zeros(len(tested), dtype=bool)
    simulated = tested_totals < retain
    if simulated.any():
        p_values[simulated], limited[simulated] = _monte_carlo_p_values(
            tested_totals[simulated],
            feature_parts[simulated],
            log_probabilities[simulated],
            sampler,
            n_iter=settings.n_iter,
            random_seed=settings.random_seed,
            nthreads=nthreads,
        )
    n_barcodes = len(totals)
    full_log_probability = np.full(n_barcodes, np.nan)
    full_log_probability[tested] = log_probabilities
    full_p_value = np.full(n_barcodes, np.nan)
    full_p_value[tested] = p_values
    full_limited = np.zeros(n_barcodes, dtype=bool)
    full_limited[tested] = limited
    return EmptyDropsResult(
        barcodes=barcodes,
        total=totals,
        log_probability=full_log_probability,
        p_value=full_p_value,
        fdr=adjust_pvalues(full_p_value, "fdr_bh"),
        limited=full_limited,
        ambient=proportions,
        lower=lower,
        retain=retain,
        knee=ranks.knee,
        inflection=ranks.inflection,
        alpha=None if math.isinf(alpha) else alpha,
        n_iter=settings.n_iter,
        random_seed=settings.random_seed,
        n_ambient=pool.n_barcodes,
    )


def _validated_matrix(counts: Any) -> np.ndarray | sparse.csr_matrix:
    """Return the counts as a dense array or a canonical CSR matrix.

    A sparse matrix is converted to CSR, and copied only to sum repeated
    entries, so the caller's matrix is never changed.
    """
    matrix: np.ndarray | sparse.csr_matrix
    if sparse.issparse(counts):
        matrix = sparse.csr_matrix(counts)
        values = matrix.data
    else:
        matrix = np.asarray(counts)
        values = matrix
        if matrix.ndim != 2:
            raise ValueError(
                "counts must be a two-dimensional barcodes by features matrix"
            )
    if values.dtype.kind not in "iuf":
        raise TypeError("counts must be numeric")
    if values.size and (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError("counts must be non-negative integers")
    if isinstance(matrix, sparse.csr_matrix) and not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
    return matrix


def _matrix_rows(
    matrix: np.ndarray | sparse.csr_matrix,
    positions: np.ndarray,
) -> Iterator[Any]:
    """Yield the rows at ``positions`` in blocks of at most 64 MiB of counts."""
    row_bytes = matrix.shape[1] * matrix.dtype.itemsize
    block_rows = (
        _BLOCK_ROWS
        if isinstance(matrix, sparse.csr_matrix)
        else max(1, min(_BLOCK_ROWS, _BLOCK_BYTES // max(1, row_bytes)))
    )
    for start in range(0, len(positions), block_rows):
        yield matrix[positions[start : start + block_rows]]


def empty_drops(
    counts: Any,
    *,
    lower: int = DEFAULT_LOWER,
    retain: int | None = None,
    n_iter: int = DEFAULT_N_ITER,
    alpha: float | None = None,
    random_seed: int = 0,
    nthreads: int | None = None,
    barcodes: Any = None,
) -> EmptyDropsResult:
    """Test every barcode of an unfiltered count matrix against ambient RNA.

    Implements EmptyDrops (Lun et al. 2019) on an in-memory matrix: see the
    module documentation for the method. ``DataStore.qc.call_cells`` runs the
    same computation over a stored matrix in bounded memory.

    Args:
        counts: Barcodes by features matrix of non-negative integer counts, a
            NumPy array or SciPy sparse matrix, with every barcode of the
            droplet run, empty droplets included.
        lower: Barcodes with at most this many counts form the ambient pool
            and are not tested.
        retain: Barcodes with at least this many counts are cells with
            p-value 0. None uses the knee of the barcode-rank curve.
        n_iter: Monte Carlo iterations; the smallest p-value is
            ``1 / (n_iter + 1)``.
        alpha: Dirichlet-multinomial precision of the ambient model. None
            estimates it by maximum likelihood over the ambient pool.
        random_seed: Seed of the simulations; a seed gives the same p-values
            on any number of threads.
        nthreads: Threads for the simulations. None uses Numba's thread count.
        barcodes: Optional identifiers of the barcodes, one per row.

    Returns:
        The statistics of every barcode.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, no barcode exceeds
            ``lower``, or the ambient pool is too small to smooth.
    """
    import numba

    settings = validated_settings(
        lower=lower,
        retain=retain,
        n_iter=n_iter,
        alpha=alpha,
        random_seed=random_seed,
    )
    threads = (
        numba.get_num_threads()
        if nthreads is None
        else integer_argument(nthreads, "nthreads", minimum=1)
    )
    matrix = _validated_matrix(counts)
    identifiers = None if barcodes is None else np.asarray(barcodes)
    if identifiers is not None and identifiers.shape != (matrix.shape[0],):
        raise ValueError("barcodes must hold one identifier per row of counts")
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    return run_empty_drops(
        totals,
        lambda positions: _matrix_rows(matrix, positions),
        int(matrix.shape[1]),
        settings,
        nthreads=threads,
        barcodes=identifiers,
    )
