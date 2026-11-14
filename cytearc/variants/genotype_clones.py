"""Genotype clones called by likelihood, with a search for the number of clones that scales to 200,000 cells.

The model is SCClone's mixture of clone genotypes with a dropout and an error
rate per site (:mod:`cytearc.variants.genotype_mixture`). The number of clones
is chosen by Schwarz's criterion ("Estimating the Dimension of a Model",
Annals of Statistics 6:461-464, 1978, doi:10.1214/aos/1176344136) as a score
to maximize, ``loglik - log(cells) / 2 * (clones + edits)``, where the edits
are the sites that each genotype differs from its neighbor on a minimum
spanning tree rooted at the wild type, so a subclone costs the mutations it
adds rather than every site. SCClone chooses the number of clones by an
inter-cluster score over every pair of cells, which does not scale to
200,000 cells, so this search replaces it:

1. EM from one clone of the sites most cells read mutant, beside the wild
   type.
2. Rounds of splits. Each component's split is seeded at sites whose
   other-state calls (wild type where the clone carries the mutation, mutant
   where it does not) exceed the pooled rates and co-occur with other sites'
   in the same cells. Each seed is fitted on the component's cells at the
   sites where its children can differ, with those sites' rates in closed
   form, and scored by the evidence lower bound it guarantees for the whole
   fit (:func:`~cytearc.variants.genotype_mixture.split_fit`), so a split
   needs no EM over every cell; the three best are refitted with their sites
   widened to every site where a child refitted at all sites differs from
   its parent. The wild type may split off a clone. Each round accepts every
   component's best split while their joint bound raises the score by a nat
   or more, then runs five iterations of EM.
3. EM to convergence, then merges of whole clones into others or into the
   wild type while the score of the hard assignment, with the sites' rates
   in closed form, rises.
4. Clones of fewer than ``min_size`` cells whose most probable component
   they are, and clones whose removal raises the score, are removed one at a
   time, each followed by EM.

Each cell then takes its most probable component. A cell whose posterior of
it is below ``min_posterior`` is ambiguous and takes no clone, as does a
cell of the wild type or a cell without a mutant call, and a clone that
ends with fewer than ``min_size`` cells leaves its cells without one.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..utils.arguments import quoted_value
from ..utils.memory_budget import MemoryLedger
from ..utils.logging import logger
from ..utils.numba import numba_threads
from .genotype_mixture import (
    DROPOUT_PRIOR_CALLS,
    ERROR_PRIOR_CALLS,
    MAX_POOLED_ERROR,
    MAX_SITE_ERROR_RATIO,
    DENSE_PATTERN_SITES,
    N_CHUNKS,
    SUPPORT,
    GenotypeMixture,
    MixtureFit,
    SiteRates,
    any_bits,
    call_patterns,
    column_counts,
    distances,
    fit_arrays,
    hard_call_sums,
    local_bits,
    n_words,
    other_state_bits,
    pack_mask,
    rate_update,
    removal_logliks,
    site_counts,
    site_term,
    spanning_tree,
    split_fit,
    weighted_site_counts,
)

#: The posterior that a cell's most probable component must reach, else the
#: cell is ambiguous, and the fewest cells of a clone, where the call leaves
#: them unset.
MIN_POSTERIOR = 0.6
MIN_SIZE = 2
#: The rise in Schwarz's score, in nats, that a split's bound must clear.
MIN_GAIN = 1.0
#: The seeds of a component's split that are fitted and ranked by their
#: bounds, the best of them refitted at widened sites, and how often.
N_SEEDS = 30
N_WIDENED = 3
MAX_WIDENINGS = 2
#: The sites with the most other-state calls among which seeds are chosen.
N_SEED_CANDIDATES = 30
#: The most sites at which a split's children may differ from their parent.
MAX_SPLIT_SITES = 256
#: Iterations of a split's local fit, and its tolerance on the bound.
SPLIT_ITERATIONS = 30
SPLIT_TOLERANCE = 1e-6
#: Iterations of EM: after each round of splits, and to convergence.
ROUND_ITERATIONS = 5
MAX_ITERATIONS = 100
#: Rounds of splits in each pass, passes, and rounds of merges.
MAX_ROUNDS = 50
MAX_PASSES = 3
MAX_MERGE_ROUNDS = 20
#: The most clones of a fit, beside the wild type.
MAX_CLONES = 500


#: The rules that call genotype clones: ``"likelihood"`` fits SCClone's
#: mixture, ``"exact"`` links cells that agree at every site both were
#: called at.
GENOTYPE_CLONE_METHODS: tuple[str, ...] = ("likelihood", "exact")


@dataclass(frozen=True, slots=True)
class GenotypeCloneRules:
    """A genotype clone method with its rules resolved.

    Attributes:
        method: ``"likelihood"`` or ``"exact"``.
        dropout_probability: The dropout rate held at every site, or None to
            estimate each site's; None for ``"exact"``.
        error_probability: The error rate held at every site, or None to
            estimate each site's; None for ``"exact"``.
        min_size: The fewest cells of a clone; None for ``"exact"``.
        min_posterior: The posterior that a cell's most probable component
            must reach; None for ``"exact"``.
    """

    method: str
    dropout_probability: float | None
    error_probability: float | None
    min_size: int | None
    min_posterior: float | None


def _probability(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a number or None")
    number = float(value)
    if not 0.0 <= number < 1.0:
        raise ValueError(
            f"{name} must lie from 0 up to but not including 1, not {number}"
        )
    return number


def genotype_clone_rules(
    method: object,
    *,
    dropout_probability: object = None,
    error_probability: object = None,
    min_size: object = None,
    min_posterior: object = None,
) -> GenotypeCloneRules:
    """Check a genotype clone method's arguments and resolve its rules.

    For ``"likelihood"``, ``min_size`` None resolves to ``MIN_SIZE`` and
    ``min_posterior`` None to ``MIN_POSTERIOR``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the method is unknown, an argument is given to
            ``"exact"``, which uses none of them, or a rule is out of range.
    """
    if not isinstance(method, str) or method not in GENOTYPE_CLONE_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(GENOTYPE_CLONE_METHODS)}, not "
            f"{quoted_value(method)}"
        )
    given = {
        "dropout_probability": dropout_probability,
        "error_probability": error_probability,
        "min_size": min_size,
        "min_posterior": min_posterior,
    }
    if method == "exact":
        unused = [name for name, value in given.items() if value is not None]
        if unused:
            raise ValueError(
                f"method='exact' uses no {', '.join(unused)}; it links cells that "
                "agree at every site both were called at"
            )
        return GenotypeCloneRules("exact", None, None, None, None)
    dropout = _probability(dropout_probability, "dropout_probability")
    error = _probability(error_probability, "error_probability")
    if dropout is not None and error is not None and dropout + error >= 1.0:
        raise ValueError(
            "dropout_probability and error_probability must sum to less than 1, so "
            "that a cell reads mutant more often where its clone carries the "
            f"mutation than where it does not; got {dropout} and {error}"
        )
    if min_size is None:
        size = MIN_SIZE
    elif isinstance(min_size, bool | np.bool_) or not isinstance(
        min_size, int | np.integer
    ):
        raise TypeError("min_size must be an integer or None")
    else:
        size = int(min_size)
        if size < 1:
            raise ValueError(f"min_size must be at least 1, not {size}")
    if min_posterior is None:
        posterior = MIN_POSTERIOR
    elif isinstance(min_posterior, bool | np.bool_) or not isinstance(
        min_posterior, int | float | np.integer | np.floating
    ):
        raise TypeError("min_posterior must be a number or None")
    else:
        posterior = float(min_posterior)
        if not 0.0 <= posterior <= 1.0:
            raise ValueError(f"min_posterior must lie from 0 to 1, not {posterior}")
    return GenotypeCloneRules("likelihood", dropout, error, size, posterior)


@dataclass(frozen=True, slots=True)
class LikelihoodClones:
    """The clones that the likelihood rule calls in one group of cells.

    Attributes:
        codes: Each cell's clone, ranked from 0 by size, -1 for none.
        probability: Each cell's posterior of its clone, NaN without one.
        ambiguous: Whether each cell with a mutant call has no component with
            a posterior of at least ``min_posterior``.
        genotypes: Each clone's genotype by rank, (clones, sites).
        sizes: The cells of each clone by rank.
        dropout: Each site's dropout rate.
        error: Each site's error rate.
        pooled_dropout: The pooled dropout rate that the fit used, which the
            sites' rates are shrunk toward, or the rate held.
        pooled_error: The pooled error rate that the fit used, or the rate
            held.
        carried_calls: The expected calls at sites that each cell's
            component carries, over the cells, which weigh the pooled dropout
            rate when groups are pooled.
        absent_calls: The expected calls at sites that the cells' components
            lack, which weigh the pooled error rate.
        loglik: The log-likelihood of the calls under the fit.
        score: Schwarz's score of the fit.
        cells_in_small_clones: Cells left without a clone because their clone
            held fewer than ``min_size`` cells.
        bound_checks: For each round of splits, the rise in log-likelihood
            that EM's first E-step found and the joint bound on it.
        capped: Whether the search stopped splitting at ``MAX_CLONES``
            clones, the most that a fit holds, so that clones its score would
            split further may hold several of the cells' clones.
        held_bytes: The bytes that the call left charged to its ledger for
            these clones, which the caller releases once it lets them go.
    """

    codes: np.ndarray
    probability: np.ndarray
    ambiguous: np.ndarray
    genotypes: np.ndarray
    sizes: np.ndarray
    dropout: np.ndarray
    error: np.ndarray
    pooled_dropout: float
    pooled_error: float
    carried_calls: float
    absent_calls: float
    loglik: float
    score: float
    cells_in_small_clones: int
    bound_checks: tuple[tuple[float, float], ...]
    capped: bool
    held_bytes: int


def capped_search_message(searches: int, cells: int) -> str:
    """The warning of ``searches`` searches, over ``cells`` cells in all, that stopped splitting at ``MAX_CLONES`` clones."""
    if searches == 1:
        return (
            f"The search for the clones of {cells} cells stopped splitting at "
            f"{MAX_CLONES} clones, the most that a fit holds, so clones that its "
            "score would split further may hold several of the cells' clones"
        )
    return (
        f"The searches for the clones of {searches} within groups, {cells} cells "
        f"in all, stopped splitting at {MAX_CLONES} clones, the most that a fit "
        "holds, so clones that their scores would split further may hold several "
        "of the cells' clones"
    )


def tree_length(genotypes: np.ndarray) -> int:
    """The length of a minimum spanning tree over the wild type and ``genotypes``, in sites."""
    gbits = pack_mask(
        np.vstack([np.zeros((1, genotypes.shape[1]), dtype=bool), genotypes])
    )
    return int(spanning_tree(gbits)[0])


def schwarz_score(loglik: float, genotypes: np.ndarray, n_cells: int) -> float:
    """Schwarz's criterion as a score to maximize: each clone costs its weight and its tree edits."""
    n_params = len(genotypes) + tree_length(genotypes)
    return loglik - 0.5 * math.log(max(n_cells, 2)) * n_params


def _genotype_key(genotype: np.ndarray) -> bytes:
    return np.packbits(genotype).tobytes()


def _site_column(bits: np.ndarray, site: int) -> np.ndarray:
    """Whether each row's bit for ``site`` is set."""
    return ((bits[:, site // 64] >> np.uint64(site % 64)) & np.uint64(1)).astype(bool)


def seed_bytes(n_cells: int, n_rows: int, n_sites: int) -> int:
    """What choosing a component's seeds holds for its ``n_rows`` cells, beside the seeds it keeps.

    The cells' positions, a mask over every cell, their other-state calls
    packed, a seed site's column as it is unpacked with the rows that read
    the other state there, and the sites' counts and statistics.
    """
    return n_cells + (8 * n_words(n_sites) + 33) * n_rows + 160 * n_sites


def kept_seed_bytes(n_rows: int, n_sites: int) -> int:
    """What a component's seeds keep while they are fitted: each seed's cells and starting genotypes."""
    return N_SEED_CANDIDATES * (n_rows + 2 * n_sites + 8 * MAX_SPLIT_SITES + 512)


def pattern_bytes(n_rows: int, n_local: int) -> int:
    """What :func:`~cytearc.variants.genotype_mixture.call_patterns` builds for ``n_rows`` cells at ``n_local`` sites, beside their packed calls.

    Up to ten sites, each cell's code and pattern beside the dense table of
    codes and the patterns. Above, NumPy's unique of the rows: its sorted
    copy, order, mask and inverse beside the keys and the patterns, of one
    word a row up to 32 sites, where a mutant and a wild-type word share
    one key, and of both words otherwise.
    """
    if n_local <= DENSE_PATTERN_SITES:
        codes = int(3**n_local)
        return 16 * n_rows + 16 * codes + 24 * min(n_rows, codes)
    if n_local <= 32:
        return 72 * n_rows
    return (72 * n_words(n_local) + 64) * n_rows


def split_fit_bytes(n_rows: int, n_local: int, n_sites: int) -> int:
    """What fitting one split at ``n_local`` sites over ``n_rows`` cells builds and lets go.

    The cells' calls at the sites, packed; their distinct patterns
    (:func:`pattern_bytes`); the posteriors of the children per cell and
    their weights; and a refit of both children at every site.
    """
    return (
        16 * n_words(n_local) * n_rows
        + pattern_bytes(n_rows, n_local)
        + 24 * n_rows
        + 16 * N_CHUNKS * n_sites
        + 256 * n_sites
        + 160 * n_local
    )


def round_bytes(n_cells: int, n_sites: int, n_components: int) -> int:
    """What a round of splits holds beside its fit.

    Each cell's most probable component and a mask of a component's cells,
    the sites' statistics as they are summed over the components, the
    genotypes' keys and every component's best split.
    """
    return (
        9 * n_cells
        + 64 * n_sites
        + 26 * n_components * n_sites
        + n_components * (n_sites // 8 + 128)
        + n_components * (3 * n_sites + 24 * MAX_SPLIT_SITES + 512)
    )


def joint_bytes(n_sites: int, n_components: int) -> int:
    """What the joint bound of a round's splits builds for a fit of ``n_components`` components: the sites' statistics and rates, and the genotypes with their spanning tree, twice."""
    return (
        192 * n_sites
        + 4 * n_components * n_sites
        + 32 * n_components * n_words(n_sites)
    )


def prune_bytes(n_cells: int, n_sites: int, n_components: int) -> int:
    """What scoring the removal of each clone builds: each cell's most probable component, the chunks' sums of each removal and the genotypes without one clone."""
    return 8 * n_cells + 24 * N_CHUNKS * n_components + 3 * n_components * n_sites


def merge_bytes(n_cells: int, n_sites: int, n_components: int) -> int:
    """What a pass of merges holds beside its fit.

    Each cell's most probable component, each component's hard calls per
    site with their chunks, the genotypes' distances, and the score of every
    merge of a source, arrays of components by sites.
    """
    return (
        8 * n_cells
        + 16 * N_CHUNKS * n_components * n_sites
        + 16 * n_components * n_sites
        + 8 * n_components * n_components
        + 200 * n_components * n_sites
    )


def assignment_bytes(n_cells: int, n_sites: int, n_components: int) -> int:
    """What assigning each cell its clone builds: its most probable component and posterior, its flags, code and probability, and the calls' totals over the components."""
    return 64 * n_cells + 64 * n_components + 48 * n_components * n_sites


def result_bytes(n_cells: int, n_sites: int, n_clones: int) -> int:
    """What the clones that :func:`likelihood_clones` returns hold: each cell's clone, posterior and flag, the genotypes and sizes, and each site's rates."""
    return 17 * n_cells + n_clones * (n_sites + 8) + 16 * n_sites


@dataclass(slots=True)
class _Split:
    """A component's best split: its bound on the score change, the bound's parts and the children."""

    delta: float
    gain: float
    mixing: float
    component: int
    site: int
    sites: np.ndarray
    children: np.ndarray
    share: float
    mutant: np.ndarray
    wild: np.ndarray


def _site_stats(
    fit: MixtureFit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expected wild-type calls at carried sites, calls at carried sites, mutant calls at absent sites and calls at absent sites."""
    arrays = fit_arrays(fit)
    n_sites = fit.genotypes.shape[1]
    carried = np.vstack([np.zeros((1, n_sites), dtype=bool), fit.genotypes])
    known = arrays.mutant + arrays.wild
    carried_wild = (arrays.wild * carried).sum(axis=0)
    carried_all = (known * carried).sum(axis=0)
    absent_mutant = (arrays.mutant * ~carried).sum(axis=0)
    absent_all = (known * ~carried).sum(axis=0)
    return carried_wild, carried_all, absent_mutant, absent_all


def _majority(
    n_mutant: np.ndarray, n_wild: np.ndarray, fallback: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The majority call at each site from mutant and wild-type counts, ``fallback`` where none was read, and the mutant share."""
    genotype = n_mutant > n_wild
    unread = (n_mutant + n_wild) == 0
    genotype[unread] = fallback[unread]
    share = np.where(unread, 0.5, n_mutant / np.maximum(n_mutant + n_wild, 1))
    return genotype, share


class _Search:
    """The search over splits of one group's fit."""

    def __init__(self, mixture: GenotypeMixture, *, min_size: int) -> None:
        self.mixture = mixture
        self.ledger = mixture.ledger
        self.min_size = min_size
        self.half_log_n = 0.5 * math.log(max(mixture.n_cells, 2))
        # Whether a round stopped proposing splits at MAX_CLONES.
        self.capped = False

    def parent_of(self, fit: MixtureFit, component: int) -> np.ndarray:
        if component == 0:
            return np.zeros(self.mixture.n_sites, dtype=bool)
        genotype: np.ndarray = fit.genotypes[component - 1]
        return genotype

    def seeds(
        self, rows: np.ndarray, parent: np.ndarray, rates: SiteRates
    ) -> list[tuple[int, np.ndarray]]:
        """Seed sites whose other-state calls co-occur with other sites' in the same cells.

        A subclone that differs from its parent at several sites makes cells
        read the other state at those sites together, while a site's own
        dropout or error does not; each candidate, among the sites with the
        most excess over the pooled rates, is scored by the G statistic of
        its two strongest partners. Returns each seed site with whether each
        of ``rows`` reads the other state there.
        """
        mixture = self.mixture
        n_sites = mixture.n_sites
        other = other_state_bits(
            mixture.mbits, mixture.wbits, rows, pack_mask(parent[None, :])[0]
        )
        n_other = column_counts(other, n_sites).astype(np.float64)
        known = (
            site_counts(mixture.mbits, rows, n_sites)
            + site_counts(mixture.wbits, rows, n_sites)
        ).astype(np.float64)
        expected = np.where(parent, rates.pooled_dropout, rates.pooled_error) * known
        excess = n_other - expected
        candidates = [
            int(site)
            for site in np.argsort(-excess, kind="stable")[:N_SEED_CANDIDATES]
            if excess[site] > 1
        ]
        n_rows = len(rows)
        scored = []
        for site in candidates:
            flipped = _site_column(other, site)
            together = site_counts(other, np.flatnonzero(flipped), n_sites).astype(
                np.float64
            )
            together[site] = 0.0
            expected_together = n_other * (n_other[site] / max(n_rows, 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                statistic = np.where(
                    together > expected_together,
                    together * np.log(together / np.maximum(expected_together, 1e-12)),
                    0.0,
                )
            statistic[site] = 0.0
            strongest = np.sort(statistic)[-2:].sum()
            scored.append((float(strongest), site, flipped))
        scored.sort(key=lambda item: -item[0])
        return [
            (site, flipped)
            for strength, site, flipped in scored[:N_SEEDS]
            if strength > 0
        ]

    def propose(
        self,
        fit: MixtureFit,
        component: int,
        hard: np.ndarray,
        stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        edits: int,
        others_keys: set[bytes],
    ) -> _Split | None:
        """The best split of ``component`` by the bound on the score change, or None.

        Every seed gets a local fit at the sites where its majority calls
        differ; the ``N_WIDENED`` seeds with the highest bounds are refitted
        with their candidate sites widened to every site where a child's
        refit genotype differs from its parent. Component 0 is the wild
        type: it keeps the all-zero genotype and splits off one new clone.
        """
        mixture = self.mixture
        ledger = self.ledger
        n_sites = mixture.n_sites
        arrays = fit_arrays(fit)
        wild_type = component == 0
        n_hard = int(np.count_nonzero(hard == component))
        if n_hard < 2 * self.min_size:
            return None
        ledger.peak(
            f"seeding a split of {n_hard} cells",
            seed_bytes(mixture.n_cells, n_hard, n_sites)
            + kept_seed_bytes(n_hard, n_sites),
        )
        hard_rows = np.flatnonzero(hard == component)
        parent = self.parent_of(fit, component)
        seeds = self.seeds(hard_rows, parent, fit.rates)
        if not seeds:
            return None
        responsibility = arrays.posteriors[:, component]
        n_support = int(np.count_nonzero(responsibility >= SUPPORT))
        # The seeds and the cells of the split, each with its posterior.
        kept = ledger.hold(
            f"the seeds of a split of {n_hard} cells",
            kept_seed_bytes(n_hard, n_sites)
            + 8 * n_hard
            + 16 * n_support
            + mixture.n_cells,
        )
        try:
            rows = np.flatnonzero(responsibility >= SUPPORT)
            weights = np.ascontiguousarray(responsibility[rows])
            others = (
                fit.genotypes
                if wild_type
                else np.delete(fit.genotypes, component - 1, axis=0)
            )
            clone_mutant = site_counts(mixture.mbits, hard_rows, n_sites)
            clone_wild = site_counts(mixture.wbits, hard_rows, n_sites)
            starts = []
            for site, flipped in seeds:
                child_rows = hard_rows[flipped]
                child_mutant = site_counts(mixture.mbits, child_rows, n_sites)
                child_wild = site_counts(mixture.wbits, child_rows, n_sites)
                child, child_share = _majority(child_mutant, child_wild, parent)
                child[site] = ~parent[site]
                if wild_type:
                    remaining = parent.copy()
                    remaining_share = np.zeros(n_sites)
                else:
                    remaining, remaining_share = _majority(
                        clone_mutant - child_mutant, clone_wild - child_wild, parent
                    )
                    remaining[site] = parent[site]
                # The seed site is among them, since the child flips it.
                candidates = np.flatnonzero(
                    (child != remaining) | (child != parent) | (remaining != parent)
                )
                if len(candidates) > MAX_SPLIT_SITES:
                    strength = np.abs(child_share - remaining_share)[candidates]
                    strength[candidates == site] = np.inf
                    candidates = np.sort(
                        candidates[
                            np.argsort(-strength, kind="stable")[:MAX_SPLIT_SITES]
                        ]
                    )
                first_share = float(np.clip(flipped.mean(), 0.05, 0.95))
                starts.append((int(site), child, remaining, candidates, first_share))
            del seeds
            context = _SplitContext(
                component=component,
                wild_type=wild_type,
                parent=parent,
                rows=rows,
                weights=weights,
                stats=stats,
                edits=edits,
                others=others,
                others_keys=others_keys,
                settings=mixture.rate_settings(fit.rates),
            )
            fitted = [
                (self.fit_seed(fit, context, start, 0), start) for start in starts
            ]
            ranked = sorted(
                [(split, start) for split, start in fitted if split is not None],
                key=lambda item: -item[0].delta,
            )
            best = ranked[0][0] if ranked else None
            for _split, start in ranked[:N_WIDENED]:
                widened = self.fit_seed(fit, context, start, MAX_WIDENINGS)
                if (
                    widened is not None
                    and best is not None
                    and widened.delta > best.delta
                ):
                    best = widened
            return best
        finally:
            ledger.release(kept)

    def fit_seed(
        self,
        fit: MixtureFit,
        context: "_SplitContext",
        start: tuple[int, np.ndarray, np.ndarray, np.ndarray, float],
        widenings: int,
    ) -> _Split | None:
        """One seed's local split fit and its bound, with up to ``widenings`` widenings of its candidate sites."""
        mixture = self.mixture
        ledger = self.ledger
        n_sites = mixture.n_sites
        arrays = fit_arrays(fit)
        site, child, remaining, candidates, first_share = start
        component = context.component
        parent = context.parent
        rows = context.rows
        weights = context.weights
        carried_wild, carried_all, absent_mutant, absent_all = context.stats
        first_full = child.copy()
        second_full = remaining.copy()
        site_gain = mixing = share = 0.0
        first_mutant = first_wild = np.empty(0)
        for widening in range(widenings + 1):
            ledger.peak(
                f"fitting a split of {len(rows)} cells at {len(candidates)} sites",
                split_fit_bytes(len(rows), len(candidates), n_sites),
            )
            local_mutant, local_wild = local_bits(
                mixture.mbits, mixture.wbits, rows, candidates
            )
            pattern_mutant, pattern_wild, pattern_weights, inverse = call_patterns(
                local_mutant, local_wild, weights, len(candidates)
            )
            del local_mutant, local_wild
            shares = np.empty(len(pattern_weights))
            first = first_full[candidates].copy()
            second = second_full[candidates].copy()
            (
                site_gain,
                mixing,
                share,
                _dropout,
                _error,
                first_mutant,
                first_wild,
                _iterations,
            ) = split_fit(
                pattern_mutant,
                pattern_wild,
                pattern_weights,
                shares,
                parent[candidates].copy(),
                first,
                second,
                np.ascontiguousarray(arrays.mutant[component][candidates]),
                np.ascontiguousarray(arrays.wild[component][candidates]),
                np.ascontiguousarray(carried_wild[candidates]),
                np.ascontiguousarray(carried_all[candidates]),
                np.ascontiguousarray(absent_mutant[candidates]),
                np.ascontiguousarray(absent_all[candidates]),
                np.ascontiguousarray(fit.rates.dropout[candidates]),
                np.ascontiguousarray(fit.rates.error[candidates]),
                *context.settings,
                SPLIT_ITERATIONS,
                SPLIT_TOLERANCE,
                context.wild_type,
                first_share,
            )
            del pattern_mutant, pattern_wild, pattern_weights
            first_full = parent.copy()
            second_full = parent.copy()
            first_full[candidates] = first
            second_full[candidates] = second
            if widening == widenings:
                break
            # Refit both children at every site from the split's posteriors;
            # sites where either now differs from its parent join the fit.
            mutant_all, wild_all = weighted_site_counts(
                mixture.mbits,
                mixture.wbits,
                rows,
                weights * shares[inverse],
                n_sites,
                N_CHUNKS,
            )
            del inverse, shares
            carried_mutant, carried_wild_weight = fit.rates.weights()
            outside = np.ones(n_sites, dtype=bool)
            outside[candidates] = False
            child_all = (
                mutant_all * carried_mutant + wild_all * carried_wild_weight
            ) > 0
            unread = (mutant_all + wild_all) <= SUPPORT
            child_all[unread] = parent[unread]
            if context.wild_type:
                remaining_all = parent.copy()
            else:
                remaining_mutant = arrays.mutant[component] - mutant_all
                remaining_wild = arrays.wild[component] - wild_all
                remaining_all = (
                    remaining_mutant * carried_mutant
                    + remaining_wild * carried_wild_weight
                ) > 0
                unread = (remaining_mutant + remaining_wild) <= SUPPORT
                remaining_all[unread] = parent[unread]
            extra = np.flatnonzero(
                outside & ((child_all != parent) | (remaining_all != parent))
            )
            if len(extra) == 0 or len(candidates) + len(extra) > MAX_SPLIT_SITES:
                break
            first_full[extra] = child_all[extra]
            second_full[extra] = remaining_all[extra]
            candidates = np.sort(np.concatenate([candidates, extra]))
            first_share = float(np.clip(share, 0.05, 0.95))
        gain = site_gain + mixing
        total = float(weights.sum())
        if total * share < self.min_size or total * (1 - share) < self.min_size:
            return None
        children = np.vstack([first_full, second_full])
        if (
            (children[0] == children[1]).all()
            or not children[0].any()
            or (not context.wild_type and not children[1].any())
        ):
            return None
        new_children = children[:1] if context.wild_type else children
        if any(
            _genotype_key(genotype) in context.others_keys for genotype in new_children
        ):
            return None
        new_edits = tree_length(np.vstack([context.others, new_children]))
        delta = gain - self.half_log_n * (1 + new_edits - context.edits)
        return _Split(
            delta,
            gain,
            mixing,
            component,
            site,
            candidates,
            children,
            float(share),
            first_mutant,
            first_wild,
        )

    def joint(
        self,
        fit: MixtureFit,
        stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        splits: list[_Split],
        edits: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The joint bound of applying ``splits`` together: the score change, the log-likelihood change, genotypes, log weights and rates."""
        mixture = self.mixture
        arrays = fit_arrays(fit)
        n_sites = mixture.n_sites
        carried_wild, carried_all, absent_mutant, absent_all = (
            values.copy() for values in stats
        )
        touched = np.zeros(n_sites, dtype=bool)
        extra = 0.0
        settings = mixture.rate_settings(fit.rates)
        dropout = fit.rates.dropout.copy()
        error = fit.rates.error.copy()
        for split in splits:
            component = split.component
            sites = split.sites
            parent = self.parent_of(fit, component)[sites]
            clone_mutant = arrays.mutant[component][sites]
            clone_wild = arrays.wild[component][sites]
            # Remove the clone's calls and add its children's.
            carried_wild[sites] -= np.where(parent, clone_wild, 0.0)
            carried_all[sites] -= np.where(parent, clone_mutant + clone_wild, 0.0)
            absent_mutant[sites] -= np.where(parent, 0.0, clone_mutant)
            absent_all[sites] -= np.where(parent, 0.0, clone_mutant + clone_wild)
            for genotype, mutant, wild in (
                (split.children[0][sites], split.mutant, split.wild),
                (
                    split.children[1][sites],
                    clone_mutant - split.mutant,
                    clone_wild - split.wild,
                ),
            ):
                carried_wild[sites] += np.where(genotype, wild, 0.0)
                carried_all[sites] += np.where(genotype, mutant + wild, 0.0)
                absent_mutant[sites] += np.where(genotype, 0.0, mutant)
                absent_all[sites] += np.where(genotype, 0.0, mutant + wild)
            touched[sites] = True
            extra += split.mixing
        changed = np.flatnonzero(touched)
        old = sum(
            site_term(
                stats[0][site],
                stats[1][site],
                stats[2][site],
                stats[3][site],
                dropout[site],
                error[site],
            )
            for site in changed
        )
        new = 0.0
        new_dropout = dropout.copy()
        new_error = error.copy()
        for site in changed:
            new_dropout[site], new_error[site] = rate_update(
                carried_wild[site],
                carried_all[site],
                absent_mutant[site],
                absent_all[site],
                *settings,
            )
            new += site_term(
                carried_wild[site],
                carried_all[site],
                absent_mutant[site],
                absent_all[site],
                new_dropout[site],
                new_error[site],
            )
        genotypes, log_weights = self.apply(fit, splits)
        new_edits = tree_length(genotypes)
        gain = new - old + extra
        delta = gain - self.half_log_n * (len(splits) + new_edits - edits)
        return delta, gain, genotypes, log_weights, new_dropout, new_error

    def apply(
        self, fit: MixtureFit, splits: list[_Split]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genotypes and log weights with each split applied: a clone's children replace it; the wild type's new clone goes last."""
        replaced = {split.component: split for split in splits}
        genotypes = []
        log_weights = [fit.log_weights[0]]
        for index, genotype in enumerate(fit.genotypes):
            split = replaced.get(index + 1)
            if split is None:
                genotypes.append(genotype)
                log_weights.append(fit.log_weights[index + 1])
                continue
            genotypes.extend([split.children[0], split.children[1]])
            log_weights.extend(
                [
                    fit.log_weights[index + 1] + math.log(split.share),
                    fit.log_weights[index + 1] + math.log1p(-split.share),
                ]
            )
        split = replaced.get(0)
        if split is not None:
            log_weights[0] = fit.log_weights[0] + math.log1p(-split.share)
            genotypes.append(split.children[0])
            log_weights.append(fit.log_weights[0] + math.log(split.share))
        return (
            np.array(genotypes, dtype=bool).reshape(-1, self.mixture.n_sites),
            np.array(log_weights),
        )

    def round(
        self, fit: MixtureFit, failed: set[bytes]
    ) -> tuple[MixtureFit, float] | None:
        """One round: propose a split of every component not yet failed, and accept the best jointly.

        Returns the fit that the accepted splits give, without arrays, and
        their joint bound on the log-likelihood's rise, or None.
        """
        mixture = self.mixture
        ledger = self.ledger
        arrays = fit_arrays(fit)
        held = ledger.hold(
            f"a round of splits of {fit.n_components} components",
            round_bytes(mixture.n_cells, mixture.n_sites, fit.n_components),
        )
        try:
            hard = np.argmax(arrays.posteriors, axis=1)
            stats = _site_stats(fit)
            edits = tree_length(fit.genotypes)
            keys = {_genotype_key(genotype) for genotype in fit.genotypes}
            proposals: list[_Split] = []
            order = [0] + [
                int(index) + 1
                for index in np.argsort(-fit.log_weights[1:], kind="stable")
            ]
            for component in order:
                if len(fit.genotypes) + len(proposals) >= MAX_CLONES:
                    self.capped = True
                    break
                key = (
                    b"wild type"
                    if component == 0
                    else _genotype_key(fit.genotypes[component - 1])
                )
                if key in failed:
                    continue
                split = self.propose(fit, component, hard, stats, edits, keys - {key})
                if split is None or split.delta <= MIN_GAIN:
                    failed.add(key)
                    continue
                proposals.append(split)
            del hard
            if not proposals:
                return None
            proposals.sort(key=lambda split: -split.delta)
            accepted: list[_Split] = []
            best = None
            child_keys: set[bytes] = set()
            for split in proposals:
                new_keys = {
                    _genotype_key(child)
                    for child in (
                        split.children[:1] if split.component == 0 else split.children
                    )
                }
                if new_keys & child_keys:
                    continue
                ledger.peak(
                    f"the joint bound of {len(accepted) + 1} splits",
                    joint_bytes(mixture.n_sites, fit.n_components + len(accepted) + 2),
                )
                trial = self.joint(fit, stats, [*accepted, split], edits)
                if trial[0] > MIN_GAIN and (best is None or trial[0] > best[0]):
                    accepted.append(split)
                    best = trial
                    child_keys |= new_keys
            if best is None:
                for split in proposals:
                    failed.add(
                        b"wild type"
                        if split.component == 0
                        else _genotype_key(fit.genotypes[split.component - 1])
                    )
                return None
            _delta, gain, genotypes, log_weights, new_dropout, new_error = best
            rates = SiteRates(
                new_dropout,
                new_error,
                fit.rates.pooled_dropout,
                fit.rates.pooled_error,
            )
            return MixtureFit(genotypes, log_weights, rates), gain
        finally:
            ledger.release(held)


@dataclass(frozen=True, slots=True)
class _SplitContext:
    """What every seed of one component's split shares."""

    component: int
    wild_type: bool
    parent: np.ndarray
    rows: np.ndarray
    weights: np.ndarray
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    edits: int
    others: np.ndarray
    others_keys: set[bytes]
    settings: tuple[float, float, float, float, float, float, float]


def _drop_empty(fit: MixtureFit, n_cells: int, min_cells: float) -> MixtureFit | None:
    """The fit without clones whose expected cells fall below ``min_cells`` or that repeat another genotype, or None where it keeps every clone."""
    totals = np.exp(fit.log_weights[1:]) * n_cells
    keep = (totals >= min_cells) & fit.genotypes.any(axis=1)
    seen: set[bytes] = set()
    for index in range(len(fit.genotypes)):
        key = _genotype_key(fit.genotypes[index])
        if key in seen:
            keep[index] = False
        seen.add(key)
    if keep.all():
        return None
    log_weights = np.concatenate([[fit.log_weights[0]], fit.log_weights[1:][keep]])
    log_weights = log_weights - np.logaddexp.reduce(log_weights)
    return MixtureFit(fit.genotypes[keep], log_weights, fit.rates, fit.loglik)


def _profiled_loglik(
    mixture: GenotypeMixture,
    carried_wild: np.ndarray,
    carried_all: np.ndarray,
    absent_mutant: np.ndarray,
    absent_all: np.ndarray,
) -> np.ndarray:
    """Rows of candidates by sites: the calls' log-likelihood with the rates profiled out in closed form."""
    pooled_dropout = (carried_wild.sum(axis=-1, keepdims=True) + 1.0) / (
        carried_all.sum(axis=-1, keepdims=True) + 2.0
    )
    pooled_error = np.minimum(
        (absent_mutant.sum(axis=-1, keepdims=True) + 1.0)
        / (absent_all.sum(axis=-1, keepdims=True) + 2.0),
        MAX_POOLED_ERROR,
    )
    if mixture.fixed_dropout is not None:
        dropout = np.full_like(carried_all, mixture.fixed_dropout)
    else:
        dropout = (carried_wild + DROPOUT_PRIOR_CALLS * pooled_dropout) / (
            carried_all + DROPOUT_PRIOR_CALLS
        )
    if mixture.fixed_error is not None:
        error = np.full_like(absent_all, mixture.fixed_error)
    else:
        error = (absent_mutant + ERROR_PRIOR_CALLS * pooled_error) / (
            absent_all + ERROR_PRIOR_CALLS
        )
        error = np.minimum(error, MAX_SITE_ERROR_RATIO * pooled_error)
    dropout = np.clip(dropout, 1e-12, 1 - 1e-12)
    error = np.clip(error, 1e-12, 1 - 1e-12)
    profiled: np.ndarray = (
        carried_wild * np.log(dropout)
        + (carried_all - carried_wild) * np.log1p(-dropout)
        + absent_mutant * np.log(error)
        + (absent_all - absent_mutant) * np.log1p(-error)
    ).sum(axis=-1)
    return profiled


def _merge_pass(mixture: GenotypeMixture, fit: MixtureFit) -> MixtureFit | None:
    """Merge clones into one another, or into the wild type, while Schwarz's score of the hard assignment rises.

    Each clone's cells' mutant and wild-type calls per site are summed once.
    A merge moves one clone's cells onto another's genotype, the site rates
    are profiled out in closed form, and a removed clone saves its weight
    and the edits to its nearest other genotype, so every candidate of a
    source is scored at once. Returns the merged fit without arrays, or None
    where no merge raises the score.
    """
    ledger = mixture.ledger
    n_cells = mixture.n_cells
    n_sites = mixture.n_sites
    n_components = fit.n_components
    ledger.peak(
        f"merging {n_components - 1} clones",
        merge_bytes(n_cells, n_sites, n_components),
    )
    arrays = fit_arrays(fit)
    hard = np.argmax(arrays.posteriors, axis=1)
    mutant = np.empty((n_components, n_sites))
    wild = np.empty((n_components, n_sites))
    hard_call_sums(
        mixture.mbits,
        mixture.wbits,
        hard,
        n_sites,
        np.empty((2, N_CHUNKS, n_components, n_sites)),
        mutant,
        wild,
    )
    sizes = np.bincount(hard, minlength=n_components).astype(np.float64)
    del hard
    genotypes = np.vstack([np.zeros((1, n_sites), dtype=bool), fit.genotypes])
    alive = np.ones(n_components, dtype=bool)
    half_log_n = 0.5 * math.log(max(n_cells, 2))
    distance = distances(pack_mask(genotypes))
    merges = 0

    def weights_loglik(counts: np.ndarray) -> float:
        counts = counts[counts > 0]
        return float((counts * np.log(counts / n_cells)).sum())

    def term(counts: np.ndarray | float) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.float64)
        return np.where(
            counts > 0, counts * np.log(np.maximum(counts, 1.0) / n_cells), 0.0
        )

    while alive[1:].sum() > 0:
        carried = genotypes[alive]
        alive_mutant, alive_wild = mutant[alive], wild[alive]
        carried_wild = (alive_wild * carried).sum(axis=0)
        carried_all = ((alive_mutant + alive_wild) * carried).sum(axis=0)
        absent_mutant = (alive_mutant * ~carried).sum(axis=0)
        absent_all = ((alive_mutant + alive_wild) * ~carried).sum(axis=0)
        del carried, alive_mutant, alive_wild
        base_weights = weights_loglik(sizes[alive])
        base_sites = float(
            _profiled_loglik(
                mixture,
                carried_wild[None],
                carried_all[None],
                absent_mutant[None],
                absent_all[None],
            )[0]
        )
        indices = np.flatnonzero(alive)
        best = None
        for position, source in enumerate(indices):
            if source == 0:
                continue
            targets = np.delete(indices, position)
            gained = genotypes[targets] & ~genotypes[source]
            lost = ~genotypes[targets] & genotypes[source]
            source_mutant = mutant[source][None, :]
            source_wild = wild[source][None, :]
            source_calls = source_mutant + source_wild
            sites = _profiled_loglik(
                mixture,
                carried_wild
                + np.where(gained, source_wild, 0.0)
                - np.where(lost, source_wild, 0.0),
                carried_all
                + np.where(gained, source_calls, 0.0)
                - np.where(lost, source_calls, 0.0),
                absent_mutant
                - np.where(gained, source_mutant, 0.0)
                + np.where(lost, source_mutant, 0.0),
                absent_all
                - np.where(gained, source_calls, 0.0)
                + np.where(lost, source_calls, 0.0),
            )
            del gained, lost
            # The weights term changes only for the source and its target.
            n_source = sizes[source]
            n_target = sizes[targets]
            merged_weights = (
                base_weights
                - term(n_source)
                - term(n_target)
                + term(n_target + n_source)
            )
            saved = 1 + int(distance[source, targets].min())
            delta = (
                sites - base_sites + merged_weights - base_weights + half_log_n * saved
            )
            row = int(np.argmax(delta))
            if delta[row] > 0 and (best is None or delta[row] > best[0]):
                best = (float(delta[row]), int(source), int(targets[row]))
        if best is None:
            break
        _delta, merged, target = best
        mutant[target] += mutant[merged]
        wild[target] += wild[merged]
        sizes[target] += sizes[merged]
        alive[merged] = False
        merges += 1
    if merges == 0:
        return None
    keep = alive[1:]
    log_weights = np.concatenate([[fit.log_weights[0]], fit.log_weights[1:][keep]])
    log_weights = log_weights - np.logaddexp.reduce(log_weights)
    return MixtureFit(fit.genotypes[keep], log_weights, fit.rates, fit.loglik)


def _removal_deltas(fit: MixtureFit, n_cells: int) -> np.ndarray:
    """The change in log-likelihood from removing each component, its weight shared by the rest, before refitting."""
    arrays = fit_arrays(fit)
    log_weights = fit.log_weights - np.logaddexp.reduce(fit.log_weights)
    lost = removal_logliks(arrays.relative, log_weights, arrays.posteriors, N_CHUNKS)
    # The weight that the rest keep, the log of the sum of their weights,
    # which stays exact where one component holds nearly all the weight.
    renormalized = np.array(
        [
            np.logaddexp.reduce(np.delete(log_weights, component))
            for component in range(len(log_weights))
        ]
    )
    deltas: np.ndarray = lost - n_cells * renormalized
    return deltas


def _assigned(
    posteriors: np.ndarray, has_mutant: np.ndarray, min_posterior: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's most probable component, its posterior, whether it is ambiguous and whether it takes a clone."""
    component = np.argmax(posteriors, axis=1)
    probability = np.take_along_axis(posteriors, component[:, None], axis=1)[:, 0]
    ambiguous = has_mutant & (probability < min_posterior)
    clone = has_mutant & ~ambiguous & (component > 0)
    return component, probability, ambiguous, clone


def _pruned(mixture: GenotypeMixture, fit: MixtureFit, min_size: int) -> MixtureFit:
    """Remove one clone at a time, each followed by EM, while one holds fewer than ``min_size`` of the cells whose most probable component it is, or its removal raises Schwarz's score.

    A clone below ``min_size`` goes first, then the removal of the largest
    rise, the lowest index among equal ones. ``fit`` holds the arrays of EM,
    which the fit returned holds in its place.
    """
    ledger = mixture.ledger
    n_cells = mixture.n_cells
    n_sites = mixture.n_sites
    while len(fit.genotypes) > 0:
        score = schwarz_score(fit.loglik, fit.genotypes, n_cells)
        ledger.peak(
            f"scoring the removal of {len(fit.genotypes)} clones",
            prune_bytes(n_cells, n_sites, fit.n_components),
        )
        hard = np.argmax(fit_arrays(fit).posteriors, axis=1)
        sizes = np.bincount(hard, minlength=fit.n_components)[1:]
        del hard
        candidates = [
            (np.inf, index)
            for index in range(len(fit.genotypes))
            if sizes[index] < min_size
        ]
        if not candidates:
            deltas = _removal_deltas(fit, n_cells)
            for index in range(len(fit.genotypes)):
                removed = np.delete(fit.genotypes, index, axis=0)
                delta = (
                    schwarz_score(
                        fit.loglik + float(deltas[index + 1]), removed, n_cells
                    )
                    - score
                )
                if delta > 0:
                    candidates.append((delta, index))
        if not candidates:
            break
        # The largest rise first, the lowest index among equal rises.
        _delta, index = max(candidates, key=lambda item: (item[0], -item[1]))
        log_weights = np.delete(fit.log_weights, index + 1)
        pruned = MixtureFit(
            np.delete(fit.genotypes, index, axis=0),
            log_weights - np.logaddexp.reduce(log_weights),
            fit.rates,
            fit.loglik,
        )
        mixture.release(fit)
        fit = mixture.em(pruned, max_iter=MAX_ITERATIONS)
    return fit


def likelihood_clones(
    mutant_bits: np.ndarray,
    wild_bits: np.ndarray,
    n_sites: int,
    *,
    dropout_probability: float | None,
    error_probability: float | None,
    min_size: int,
    min_posterior: float,
    ledger: MemoryLedger,
    nthreads: int,
) -> LikelihoodClones:
    """Call the clones of one group of cells by the likelihood rule; see the module docstring.

    A search that stops splitting at ``MAX_CLONES`` clones logs a warning, and
    the result's ``capped`` records it.

    Args:
        mutant_bits: Each cell's mutant calls, packed as
            :func:`~cytearc.variants.genotype_mixture.pack_mask` packs them,
            (cells, words) uint64; the caller holds them.
        wild_bits: Each cell's wild-type calls, packed the same way.
        n_sites: The sites.
        dropout_probability: The chance that a cell carrying a site's
            mutation reads wild type there, held at every site, or None to
            estimate each site's.
        error_probability: The chance that a cell without a site's mutation
            reads mutant there, held at every site, or None to estimate each
            site's.
        min_size: The fewest cells of a clone.
        min_posterior: The posterior that a cell's most probable component
            must reach, else the cell is ambiguous.
        ledger: The caller's memory ledger, which every step is charged to.
        nthreads: The threads of the parallel kernels.

    Raises:
        MemoryBudgetError: If a step does not fit the ledger's budget.
    """
    with numba_threads(nthreads):
        clones = _likelihood_clones(
            mutant_bits,
            wild_bits,
            int(n_sites),
            dropout_probability=dropout_probability,
            error_probability=error_probability,
            min_size=int(min_size),
            min_posterior=float(min_posterior),
            ledger=ledger,
        )
    if clones.capped:
        logger.warning(capped_search_message(1, len(clones.codes)))
    return clones


def _likelihood_clones(
    mutant_bits: np.ndarray,
    wild_bits: np.ndarray,
    n_sites: int,
    *,
    dropout_probability: float | None,
    error_probability: float | None,
    min_size: int,
    min_posterior: float,
    ledger: MemoryLedger,
) -> LikelihoodClones:
    """Call the clones as :func:`likelihood_clones` does, on the caller's threads, without its warning.

    The genotype clone operation saves how many searches were capped and
    warns from its saved diagnostics, whether it computes or reuses its
    clones.
    """
    ledger.peak("each site's calls over the cells", 32 * n_sites)
    mixture = GenotypeMixture(
        mutant_bits,
        wild_bits,
        n_sites,
        dropout_probability=dropout_probability,
        error_probability=error_probability,
        ledger=ledger,
    )
    held = ledger.hold("each site's calls over the cells", 16 * n_sites)
    try:
        return _search(mixture, min_size=min_size, min_posterior=min_posterior)
    finally:
        ledger.release(held)


def _search(
    mixture: GenotypeMixture, *, min_size: int, min_posterior: float
) -> LikelihoodClones:
    ledger = mixture.ledger
    n_cells = mixture.n_cells
    n_sites = mixture.n_sites
    # One clone of the sites most cells read mutant, beside the wild type.
    first = (mixture.column_mutant > mixture.column_wild)[None, :]
    if not first.any():
        first = np.zeros((1, n_sites), dtype=bool)
        first[0, int(np.argmax(mixture.column_mutant))] = True
    fit = mixture.em(
        MixtureFit(first, np.log(np.array([0.5, 0.5])), mixture.initial_rates()),
        max_iter=MAX_ITERATIONS,
    )
    search = _Search(mixture, min_size=min_size)
    checks: list[tuple[float, float]] = []
    for _pass in range(MAX_PASSES):
        failed: set[bytes] = set()
        accepted_any = False
        for _round in range(MAX_ROUNDS):
            proposed = search.round(fit, failed)
            if proposed is None:
                break
            proposal, gain = proposed
            before = fit.loglik
            mixture.release(fit)
            fit = mixture.em(proposal, max_iter=ROUND_ITERATIONS)
            # The first E-step's log-likelihood must reach the bound's gain.
            checks.append((fit.first_loglik - before, gain))
            dropped = _drop_empty(fit, n_cells, min_size)
            if dropped is not None:
                mixture.release(fit)
                fit = mixture.em(dropped, max_iter=ROUND_ITERATIONS)
            accepted_any = True
        if not accepted_any:
            break
    converged = mixture.release(fit)
    fit = mixture.em(converged, max_iter=MAX_ITERATIONS)
    for _round in range(MAX_MERGE_ROUNDS):
        merged = _merge_pass(mixture, fit)
        if merged is None:
            break
        mixture.release(fit)
        fit = mixture.em(merged, max_iter=MAX_ITERATIONS)
    flags = ledger.hold("whether each cell has a mutant call", n_cells)
    has_mutant = np.empty(n_cells, dtype=bool)
    any_bits(mixture.mbits, has_mutant)
    fit = _pruned(mixture, fit, min_size)
    if not (
        np.isfinite(fit.loglik)
        and np.isfinite(fit.rates.dropout).all()
        and np.isfinite(fit.rates.error).all()
    ):
        raise RuntimeError(
            f"The fit of the clones of {n_cells} cells at {n_sites} sites ended "
            f"with a log-likelihood of {fit.loglik} or rates that are not "
            "finite; this is a bug of the model's fit"
        )
    n_clones = len(fit.genotypes)
    ledger.peak(
        f"assigning {n_cells} cells to {n_clones} clones",
        assignment_bytes(n_cells, n_sites, fit.n_components),
    )
    arrays = fit_arrays(fit)
    component, probability, ambiguous, clone = _assigned(
        arrays.posteriors, has_mutant, min_posterior
    )
    del has_mutant
    sizes = np.bincount(component[clone] - 1, minlength=n_clones)
    # A clone of fewer than min_size cells leaves its cells without one.
    small = (sizes > 0) & (sizes < min_size)
    # Whether each component is a small clone, the wild type first.
    small_component = np.concatenate([[False], small])
    in_small = clone & small_component[component]
    cells_in_small_clones = int(np.count_nonzero(in_small))
    clone &= ~in_small
    del in_small
    sizes[small] = 0
    kept = np.flatnonzero(sizes > 0)
    order = kept[np.argsort(-sizes[kept], kind="stable")]
    rank = np.full(n_clones, -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[clone] = rank[component[clone] - 1]
    del component
    carried = np.vstack([np.zeros((1, n_sites), dtype=bool), fit.genotypes])
    calls = arrays.mutant + arrays.wild
    carried_calls = float((calls * carried).sum())
    absent_calls = float((calls * ~carried).sum())
    del carried, calls, arrays
    # The clones returned stay charged, for the caller to release.
    held = ledger.hold(
        f"the clones of {n_cells} cells",
        result_bytes(n_cells, n_sites, len(order)),
    )
    result = LikelihoodClones(
        codes=codes,
        probability=np.where(clone, probability, np.nan),
        ambiguous=ambiguous,
        genotypes=fit.genotypes[order],
        sizes=sizes[order],
        dropout=np.array(fit.rates.dropout, dtype=np.float64),
        error=np.array(fit.rates.error, dtype=np.float64),
        pooled_dropout=float(fit.rates.pooled_dropout),
        pooled_error=float(fit.rates.pooled_error),
        carried_calls=carried_calls,
        absent_calls=absent_calls,
        loglik=fit.loglik,
        score=schwarz_score(fit.loglik, fit.genotypes, n_cells),
        cells_in_small_clones=cells_in_small_clones,
        bound_checks=tuple(checks),
        capped=search.capped,
        held_bytes=held,
    )
    del probability, clone
    mixture.release(fit)
    ledger.release(flags)
    return result
