"""Genotype calls of targeted variant sites, in one vocabulary.

A genotype call says what a cell carries at a site: the wild type, the
alternative allele on one copy (``heterozygous``) or on every copy
(``homozygous_alternative``), the alternative without a claim of how many
copies carry it (``alternative_present``), no evidence at all (``missing``),
or evidence that a rule did not call (``filtered``). Every source's calls map
to these six, and each call keeps the source's own label, such as GoT's
``MUT`` or Tapestri's ``NGT`` value.

Three methods call genotypes:

- ``source`` adopts the calls that a tool wrote into its files, through
  :data:`SOURCE_CALLS`, which maps each call column's labels: GoT tables on
  GEO (``Genotype``), GoT-Multi (``genotype_merged``, ``genotype_final`` and
  its channels), Gotcha (``genotype_pred``) and Tapestri (``NGT``). A label
  that means no call becomes ``filtered`` where the cell holds evidence at
  the site, such as Tapestri's ``NGT`` 3 at a covered site, and ``missing``
  where it holds none.
- ``threshold`` calls each cell from its UMIs of the wild type and the
  alternative by rules given as parameters, since GoT studies differ: at
  least one mutant UMI calls a patient's cell mutant, and in cell lines a
  share above 90% calls the cell mutant or wild type.
- ``gift`` applies the probabilistic model of GIFT (Blattman et al., bioRxiv
  2026): each UMI of a probe may come from template switching during PCR, with
  a probability that falls with its reads; each observed gapfill is assigned
  to the wild type or the mutant by its edit distance to them; and the UMIs
  of a cell give the posterior of each genotype under a uniform prior, called
  above ``min_posterior``. The model adapts the genotype functions of GIFT's
  MIT-licensed reproducibility code, as ``THIRD_PARTY_NOTICES.md`` records.

These functions never touch storage.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef

__all__ = [
    "ALTERNATIVE_PRESENT",
    "CALLS",
    "FILTERED",
    "GAPFILL_ERROR",
    "GENOTYPE_METHODS",
    "HETEROZYGOUS",
    "HIGH_CONFIDENCE_SWAP",
    "HOMOZYGOUS_ALTERNATIVE",
    "MISSING",
    "MUTANT_CALLS",
    "SOURCE_CALLS",
    "SOURCE_PRECEDENCE",
    "UNKNOWN_CALLS",
    "WILD_TYPE",
    "VariantGenotypes",
    "GenotypeRules",
    "edit_distance",
    "gapfill_wild_type_probabilities",
    "genotype_frames",
    "genotype_rules",
    "gift_calls",
    "gift_posteriors",
    "source_call_codes",
    "swap_probability_table",
    "threshold_calls",
    "unit_swap_probabilities",
]

#: The genotype calls, in the order of their stored codes.
CALLS: tuple[str, ...] = (
    "wild_type",
    "heterozygous",
    "homozygous_alternative",
    "alternative_present",
    "missing",
    "filtered",
)
WILD_TYPE = 0
HETEROZYGOUS = 1
HOMOZYGOUS_ALTERNATIVE = 2
ALTERNATIVE_PRESENT = 3
MISSING = 4
FILTERED = 5
#: The calls of a cell that carries the alternative allele.
MUTANT_CALLS: frozenset[int] = frozenset(
    {HETEROZYGOUS, HOMOZYGOUS_ALTERNATIVE, ALTERNATIVE_PRESENT}
)
#: The calls that say neither that a cell carries the alternative nor that it
#: lacks it.
UNKNOWN_CALLS: frozenset[int] = frozenset({MISSING, FILTERED})
#: The ways of calling genotypes.
GENOTYPE_METHODS: tuple[str, ...] = ("source", "threshold", "gift")

_GOT = {"WT": WILD_TYPE, "MUT": ALTERNATIVE_PRESENT}
_GOT_MULTI = {**_GOT, "Unprofiled": MISSING}
#: How the labels of each tool's call column map to calls. GoT counts a cell's
#: mutant and wild-type UMIs, which do not tell how many copies carry the
#: mutation, so its ``MUT`` is ``alternative_present``; GoT-Multi's
#: ``Unprofiled`` means no evidence and its refined ``FalsePositive`` a mutant
#: call that its model rejected; Gotcha's ``HET`` is ``heterozygous``; and
#: Tapestri's ``NGT`` codes 0 to 3 are wild type, heterozygous, homozygous
#: alternative and no call.
SOURCE_CALLS: Mapping[str, Mapping[str, int]] = {
    "Genotype": _GOT,
    "genotype": _GOT,
    "genotype_geneseq": _GOT_MULTI,
    "genotype_probebc": _GOT_MULTI,
    "genotype_merged": _GOT_MULTI,
    "genotype_final": {**_GOT_MULTI, "FalsePositive": FILTERED},
    "genotype_pred": {**_GOT, "HET": HETEROZYGOUS},
    "NGT": {
        "0": WILD_TYPE,
        "1": HETEROZYGOUS,
        "2": HOMOZYGOUS_ALTERNATIVE,
        "3": MISSING,
    },
}
#: The order of a site's call columns, from which each cell takes the first
#: call it holds: GoT-Multi's refined call, then its merged call of both
#: channels, then the call of a single column, and its channels' own calls
#: last.
SOURCE_PRECEDENCE: tuple[str, ...] = (
    "genotype_final",
    "genotype_merged",
    "NGT",
    "Genotype",
    "genotype",
    "genotype_pred",
    "genotype_geneseq",
    "genotype_probebc",
)
#: The likelihood of a gapfill one edit away from an allele, relative to the
#: allele itself, which GIFT uses for cell lines; each further edit multiplies
#: it again.
GAPFILL_ERROR = 0.01
#: The swap probability below which a UMI counts as confident, as GIFT counts
#: the UMIs of a genotyped cell.
HIGH_CONFIDENCE_SWAP = 0.1


@dataclass(frozen=True, slots=True)
class GenotypeRules:
    """The rules of a genotype method, as resolved from its arguments.

    Attributes:
        method: ``"source"``, ``"threshold"`` or ``"gift"``.
        min_umis: The fewest UMIs of a call, or None for ``"source"``.
        min_mutant_umis: The fewest mutant UMIs of a ``"threshold"`` mutant
            call, or None.
        min_fraction: The share that a ``"threshold"`` call's allele must
            exceed, or None.
        min_posterior: The posterior that a ``"gift"`` call must exceed, or
            None for the other methods.
        swap: GIFT's swap probabilities as ascending reads and their
            probabilities, or None.
        alleles: Each named site's wild type and alternative, by label.
    """

    method: str
    min_umis: int | None
    min_mutant_umis: int | None
    min_fraction: float | None
    min_posterior: float | None
    swap: tuple[np.ndarray, np.ndarray] | None
    alleles: Mapping[str, tuple[str, str]]

    @property
    def swap_record(self) -> tuple[tuple[int, float], ...]:
        """The swap probabilities as (reads, probability) pairs, or empty."""
        if self.swap is None:
            return ()
        reads, values = self.swap
        return tuple(
            (int(read), float(value))
            for read, value in zip(reads.tolist(), values.tolist(), strict=True)
        )

    def record(self) -> dict[str, Any]:
        """The rules as JSON-ready values."""
        return {
            "min_umis": self.min_umis,
            "min_mutant_umis": self.min_mutant_umis,
            "min_fraction": self.min_fraction,
            "min_posterior": self.min_posterior,
            "swap_probabilities": [list(pair) for pair in self.swap_record],
        }


def _count_rule(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be a whole number or None")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


def _share(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not low <= number < high:
        raise ValueError(f"{name} must lie from {low} to below {high}, not {number}")
    return number


def _site_alleles(alleles: Any) -> dict[str, tuple[str, str]]:
    if alleles is None:
        return {}
    if not isinstance(alleles, Mapping):
        raise TypeError(
            "alleles must map site labels to a wild-type and an alternative allele, "
            "such as {'JAK2 c.1849G>T': ('CAC', 'AAC')}"
        )
    resolved: dict[str, tuple[str, str]] = {}
    for label, pair in alleles.items():
        if not isinstance(label, str):
            raise TypeError("alleles must be keyed by site labels")
        if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise TypeError(
                f"alleles must give the site {label!r} a wild-type and an "
                "alternative allele"
            )
        wild_type, alternative = pair
        if not isinstance(wild_type, str) or not isinstance(alternative, str):
            raise TypeError(f"The alleles of the site {label!r} must be text")
        if wild_type == alternative:
            raise ValueError(
                f"alleles gives the site {label!r} one allele, {wild_type!r}, as both "
                "the wild type and the alternative"
            )
        resolved[label] = (wild_type, alternative)
    return resolved


def genotype_rules(
    method: Any,
    *,
    min_umis: Any = None,
    min_mutant_umis: Any = None,
    min_fraction: Any = None,
    min_posterior: Any = 0.6,
    swap_probabilities: Any = None,
    alleles: Any = None,
) -> GenotypeRules:
    """Check a genotype method's arguments and resolve its rules.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the method is unknown, an argument is given to a method
            that does not use it, or a rule is out of range.
    """
    if not isinstance(method, str) or method not in GENOTYPE_METHODS:
        raise ValueError(f"method must be one of {', '.join(GENOTYPE_METHODS)}")
    umis = _count_rule(min_umis, "min_umis")
    mutant_umis = _count_rule(min_mutant_umis, "min_mutant_umis")
    fraction = (
        None if min_fraction is None else _share(min_fraction, "min_fraction", 0.5, 1.0)
    )
    posterior = _share(min_posterior, "min_posterior", 0.0, 1.0)
    pairs = _site_alleles(alleles)
    given = {
        "min_umis": umis is not None,
        "min_mutant_umis": mutant_umis is not None,
        "min_fraction": fraction is not None,
        "min_posterior": posterior != 0.6,
        "swap_probabilities": swap_probabilities is not None,
        "alleles": bool(pairs),
    }
    uses = {
        "source": set(),
        "threshold": {"min_umis", "min_mutant_umis", "min_fraction", "alleles"},
        "gift": {"min_umis", "min_posterior", "swap_probabilities", "alleles"},
    }[method]
    unused = [name for name, present in given.items() if present and name not in uses]
    if unused:
        raise ValueError(
            f"method={method!r} uses no {', '.join(unused)}; "
            + {
                "source": "it adopts the calls that the tool wrote",
                "threshold": "it calls cells from their UMIs by min_umis, "
                "min_mutant_umis and min_fraction",
                "gift": "it calls cells by GIFT's posterior, min_posterior, "
                "min_umis and swap_probabilities",
            }[method]
        )
    if method == "threshold" and mutant_umis is None and fraction is None:
        raise ValueError(
            "method='threshold' needs min_mutant_umis or min_fraction, since GoT "
            "studies apply different rules: min_mutant_umis=1 for patients' cells, "
            "min_fraction=0.9 for cell lines"
        )
    return GenotypeRules(
        method=method,
        min_umis=None if method == "source" else (1 if umis is None else umis),
        min_mutant_umis=mutant_umis,
        min_fraction=fraction,
        min_posterior=posterior if method == "gift" else None,
        swap=(
            None
            if swap_probabilities is None
            else swap_probability_table(swap_probabilities)
        ),
        alleles=pairs,
    )


def source_call_codes(
    column: str, labels: np.ndarray, covered: np.ndarray
) -> np.ndarray:
    """The calls that a source column's labels give.

    Args:
        column: The column of the source's files that holds the calls, such
            as ``"genotype_merged"``.
        labels: Each call's label as the source wrote it.
        covered: Whether the cell holds evidence at the site, which turns a
            label of no call into ``filtered``.

    Returns:
        Each label's call code, as int8.

    Raises:
        ValueError: If the column has no vocabulary, or a label is not one
            of its labels.
    """
    mapping = SOURCE_CALLS.get(column)
    if mapping is None:
        raise ValueError(
            f"The calls of the column {column!r} have no vocabulary; the known call "
            f"columns are {', '.join(SOURCE_CALLS)}. Call genotypes from the "
            "counts with method='threshold' instead"
        )
    texts = np.asarray(labels, dtype=object)
    distinct, inverse = np.unique(texts.astype(str), return_inverse=True)
    unknown = [label for label in distinct.tolist() if label not in mapping]
    if unknown:
        raise ValueError(
            f"The column {column!r} holds the call {unknown[0]!r}, which is none of "
            f"its labels: {', '.join(mapping)}"
        )
    codes = np.asarray([mapping[label] for label in distinct.tolist()], dtype=np.int8)
    calls = codes[inverse].reshape(-1)
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    calls[(calls == MISSING) & covered] = FILTERED
    return calls


def threshold_calls(
    reference: np.ndarray,
    alternative: np.ndarray,
    *,
    min_umis: int,
    min_mutant_umis: int | None,
    min_fraction: float | None,
) -> np.ndarray:
    """Call each cell at a site from its UMIs of the wild type and the alternative.

    A cell with fewer than ``min_umis`` UMIs of the two is ``filtered``.
    Without ``min_fraction``, a cell is ``alternative_present`` with at least
    ``min_mutant_umis`` mutant UMIs and ``wild_type`` with none, as GoT calls
    a patient's cells. With it, a cell is ``alternative_present`` when its
    mutant share exceeds ``min_fraction``, and its mutant UMIs reach
    ``min_mutant_umis`` when that is given, and ``wild_type`` when its
    wild-type share exceeds it, as GoT calls cell lines. Every other cell is
    ``filtered``.

    Args:
        reference: Each cell's UMIs of the wild type.
        alternative: Each cell's UMIs of the alternative.
        min_umis: The fewest UMIs of the two that a call needs, at least 1.
        min_mutant_umis: The fewest mutant UMIs of a mutant call, or None.
        min_fraction: The share, from 0.5 up to 1, that a call's allele must
            exceed, or None.

    Returns:
        Each cell's call code, as int8.
    """
    reference = np.asarray(reference, dtype=np.int64)
    alternative = np.asarray(alternative, dtype=np.int64)
    total = reference + alternative
    enough = total >= min_umis
    calls = np.full(len(total), FILTERED, dtype=np.int8)
    if min_fraction is None:
        assert min_mutant_umis is not None
        mutant = enough & (alternative >= min_mutant_umis)
        wild_type = enough & (alternative == 0)
    else:
        safe = np.maximum(total, 1)
        mutant = enough & (alternative / safe > min_fraction)
        if min_mutant_umis is not None:
            mutant &= alternative >= min_mutant_umis
        wild_type = enough & (reference / safe > min_fraction)
    calls[wild_type] = WILD_TYPE
    calls[mutant] = ALTERNATIVE_PRESENT
    return calls


def edit_distance(first: str, second: str) -> int:
    """The Levenshtein distance: the fewest substitutions, insertions and deletions."""
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def gapfill_wild_type_probabilities(
    gapfills: Sequence[str],
    wild_type: Sequence[str],
    mutant: Sequence[str],
    *,
    error: float = GAPFILL_ERROR,
) -> np.ndarray:
    """The probability that each observed gapfill came from its probe's wild type.

    Each gapfill's likelihood under an allele is ``error`` to the power of
    its edit distance to the allele, so a gapfill that matches the wild type
    and lies one edit from the mutant came from the wild type with
    probability ``1 / (1 + error)``, and a stray gapfill goes to the allele
    it lies closer to. The two likelihoods are normalized over the probe's
    two alleles, as GIFT assigns gapfills that match no allele of its
    feature set.

    Args:
        gapfills: Each observed gapfill.
        wild_type: The wild type of each gapfill's probe.
        mutant: The mutant of each gapfill's probe.
        error: The likelihood of one edit.

    Returns:
        Each gapfill's probability of the wild type; its probability of the
        mutant is one less it.
    """
    distances = np.asarray(
        [
            (edit_distance(gapfill, wild), edit_distance(gapfill, mut))
            for gapfill, wild, mut in zip(gapfills, wild_type, mutant, strict=True)
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    # Relative to the closer allele, so that far gapfills do not underflow.
    closer = distances.min(axis=1, keepdims=True)
    likelihoods = np.power(error, distances - closer)
    return np.asarray(likelihoods[:, 0] / likelihoods.sum(axis=1), dtype=np.float64)


def swap_probability_table(swap_probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    """Check a table of swap probabilities by reads per UMI.

    Args:
        swap_probabilities: A mapping from reads per UMI, a whole number of
            at least 1, to the probability, from 0 to 1, that a UMI of that
            many reads came from template switching.

    Returns:
        The reads in ascending order and their probabilities.

    Raises:
        TypeError: If it is not a mapping of whole numbers to numbers.
        ValueError: If it is empty, a key or probability lies out of range,
            or it gives no probability for UMIs of 1 read.
    """
    if not isinstance(swap_probabilities, Mapping):
        raise TypeError(
            "swap_probabilities must map reads per UMI to the probability that a "
            "UMI of that many reads came from template switching"
        )
    if not swap_probabilities:
        raise ValueError("swap_probabilities must give at least one probability")
    reads: list[int] = []
    values: list[float] = []
    for key, value in swap_probabilities.items():
        if isinstance(key, bool | np.bool_) or not isinstance(key, int | np.integer):
            raise TypeError("swap_probabilities' reads per UMI must be whole numbers")
        if isinstance(value, bool | np.bool_) or not isinstance(
            value, int | float | np.integer | np.floating
        ):
            raise TypeError("swap_probabilities' probabilities must be numbers")
        if int(key) < 1:
            raise ValueError("swap_probabilities' reads per UMI must be at least 1")
        probability = float(value)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"swap_probabilities gives {probability} for UMIs of {int(key)} "
                "reads; probabilities lie from 0 to 1"
            )
        reads.append(int(key))
        values.append(probability)
    if min(reads) != 1:
        raise ValueError(
            f"swap_probabilities starts at UMIs of {min(reads)} reads; give the "
            "probability of UMIs of 1 read too, so that every UMI has one"
        )
    order = np.argsort(reads, kind="stable")
    return (
        np.asarray(reads, dtype=np.int64)[order],
        np.asarray(values, dtype=np.float64)[order],
    )


def unit_swap_probabilities(
    umis: np.ndarray, reads: np.ndarray, table: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """The swap probability of each UMI of an entry, from its reads per UMI.

    An entry counts a cell's UMIs of one gapfill and their reads, so each of
    its UMIs takes the entry's whole reads per UMI, rounded down; that is
    each UMI's own reads where the entry holds one UMI. The table is a step
    function: a UMI takes the probability of the most reads it gives that do
    not exceed the UMI's.

    Raises:
        ValueError: If a UMI has fewer reads than the table starts at.
    """
    keys, values = table
    per_umi = np.asarray(reads, dtype=np.int64) // np.maximum(
        np.asarray(umis, dtype=np.int64), 1
    )
    position = np.searchsorted(keys, per_umi, side="right") - 1
    if np.any(position < 0):
        fewest = int(per_umi[position < 0].min())
        raise ValueError(
            f"swap_probabilities starts at UMIs of {int(keys[0])} reads, so it gives "
            f"no probability for UMIs of {fewest} reads; give one for every count "
            "of reads from 1"
        )
    return values[position]


def gift_posteriors(
    rows: np.ndarray,
    n_rows: int,
    umis: np.ndarray,
    wild_type_probabilities: np.ndarray,
    swap_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The posterior of each genotype of a cell at a probe, from its UMIs.

    Each UMI scores each genotype as GIFT does: the larger of its swap
    probability and its gapfill's probability under the genotype, which for
    the wild type is its probability of the wild type, for the homozygous
    mutant that of the mutant, and for the heterozygote the larger of half of
    each, as GIFT's reproducibility code takes the largest term. A cell's
    likelihood of a genotype is the product of its UMIs' scores, and the
    posterior divides it by the sum over the three genotypes, a uniform
    prior. The products are summed as logarithms, so many UMIs do not
    underflow.

    Args:
        rows: Each entry's row, a cell and probe, below ``n_rows``.
        n_rows: The rows.
        umis: Each entry's UMIs, which share its gapfill and swap probability.
        wild_type_probabilities: Each entry's gapfill's probability of the
            wild type.
        swap_probabilities: Each entry's UMIs' swap probability.

    Returns:
        Each row's posteriors of the wild type, the heterozygote and the
        homozygous mutant, shaped ``(n_rows, 3)``, and its UMIs whose swap
        probability lies below :data:`HIGH_CONFIDENCE_SWAP`.
    """
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(umis, dtype=np.float64)
    wild = np.asarray(wild_type_probabilities, dtype=np.float64)
    swap = np.asarray(swap_probabilities, dtype=np.float64)
    mutant = 1.0 - wild
    scores = np.stack(
        [
            np.maximum(swap, wild),
            np.maximum(swap, np.maximum(0.5 * wild, 0.5 * mutant)),
            np.maximum(swap, mutant),
        ],
        axis=1,
    )
    with np.errstate(divide="ignore"):
        logs = np.log(scores) * weights[:, None]
    totals = np.zeros((n_rows, 3), dtype=np.float64)
    for genotype in range(3):
        totals[:, genotype] = np.bincount(
            rows, weights=logs[:, genotype], minlength=n_rows
        )
    highest = totals.max(axis=1, keepdims=True)
    # A row whose every genotype scores 0 has no posterior.
    defined = np.isfinite(highest[:, 0])
    posteriors = np.full((n_rows, 3), np.nan, dtype=np.float64)
    if defined.any():
        shifted = np.exp(totals[defined] - highest[defined])
        posteriors[defined] = shifted / shifted.sum(axis=1, keepdims=True)
    confident = np.bincount(
        rows,
        weights=np.where(swap < HIGH_CONFIDENCE_SWAP, weights, 0.0),
        minlength=n_rows,
    )
    return posteriors, np.rint(confident).astype(np.int64)


def gift_calls(
    posteriors: np.ndarray,
    confident_umis: np.ndarray,
    *,
    min_posterior: float,
    min_umis: int,
) -> np.ndarray:
    """Call each row at its most probable genotype when the posterior is high enough.

    A row whose largest posterior exceeds ``min_posterior`` and that holds at
    least ``min_umis`` confident UMIs is called ``wild_type``,
    ``heterozygous`` or ``homozygous_alternative``. Ties, which only a
    ``min_posterior`` below 0.5 can call, go to that order. Every other row
    is ``filtered``.
    """
    posteriors = np.asarray(posteriors, dtype=np.float64)
    calls = np.full(len(posteriors), FILTERED, dtype=np.int8)
    defined = ~np.isnan(posteriors).any(axis=1)
    best = np.argmax(np.where(defined[:, None], posteriors, -1.0), axis=1)
    largest = np.where(defined, posteriors.max(axis=1, initial=-1.0), -1.0)
    called = (
        defined
        & (largest > min_posterior)
        & (np.asarray(confident_umis, dtype=np.int64) >= min_umis)
    )
    codes = np.asarray([WILD_TYPE, HETEROZYGOUS, HOMOZYGOUS_ALTERNATIVE], dtype=np.int8)
    calls[called] = codes[best[called]]
    return calls


def genotype_frames(
    *,
    cell_ids: np.ndarray,
    site_labels: np.ndarray,
    site_reference: np.ndarray,
    site_alternative: np.ndarray,
    site_sources: np.ndarray,
    chosen: np.ndarray,
    cells: np.ndarray,
    sites: np.ndarray,
    calls: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    alternative: np.ndarray,
    posteriors: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Return the tables of a :class:`VariantGenotypes` record.

    Args:
        cell_ids: The IDs of the genotypes' cells.
        site_labels: Each genotyped site's label.
        site_reference: Each genotyped site's wild type, or empty.
        site_alternative: Each genotyped site's alternative, or empty.
        site_sources: The call columns each site's calls came from, or empty.
        chosen: The sites to show, as ascending positions among the
            genotyped sites.
        cells: Each call row's cell, as a position in ``cell_ids``.
        sites: Each call row's site, as a position among the genotyped sites.
        calls: Each call row's call code.
        labels: Each call row's label.
        reference: Each call row's UMIs of the wild type, -1 where unknown.
        alternative: Each call row's UMIs of the alternative, -1 where unknown.
        posteriors: Each call row's three posteriors, NaN where none.
    """
    shown = np.asarray(site_labels, dtype=object)[chosen]
    site_categories = pd.CategoricalDtype(pd.Index(shown, dtype=object), ordered=True)
    cell_categories = pd.CategoricalDtype(
        pd.Index(np.asarray(cell_ids, dtype=object), dtype=object), ordered=True
    )
    site_rank = np.full(len(site_labels), -1, dtype=np.int64)
    site_rank[chosen] = np.arange(len(chosen))
    call_codes = np.asarray(calls, dtype=np.int64)
    call_table = pd.DataFrame(
        {
            "cell": pd.Categorical.from_codes(
                np.asarray(cells, dtype=np.int64), dtype=cell_categories
            ),
            "site": pd.Categorical.from_codes(
                site_rank[np.asarray(sites, dtype=np.int64)], dtype=site_categories
            ),
            "call": pd.Categorical.from_codes(
                call_codes,
                dtype=pd.CategoricalDtype(pd.Index(list(CALLS)), ordered=True),
            ),
            "label": pd.array(np.asarray(labels, dtype=object), dtype="string"),
            "reference_umis": np.asarray(reference, dtype=np.int64),
            "alternative_umis": np.asarray(alternative, dtype=np.int64),
            "posterior_wild_type": np.asarray(posteriors[:, 0], dtype=np.float64),
            "posterior_heterozygous": np.asarray(posteriors[:, 1], dtype=np.float64),
            "posterior_homozygous_alternative": np.asarray(
                posteriors[:, 2], dtype=np.float64
            ),
        }
    )
    counts = np.zeros((len(chosen), len(CALLS)), dtype=np.int64)
    row_sites = site_rank[np.asarray(sites, dtype=np.int64)]
    np.add.at(counts, (row_sites, call_codes), 1)
    # The cells that the calls do not list at a site are missing there.
    n_cells = len(cell_ids)
    counts[:, MISSING] += n_cells - counts.sum(axis=1)
    row_cells = np.asarray(cells, dtype=np.int64)
    mutant = np.isin(call_codes, list(MUTANT_CALLS))
    called = mutant | (call_codes == WILD_TYPE)
    site_table = pd.DataFrame(
        {
            "reference": pd.array(
                np.asarray(site_reference, dtype=object)[chosen], dtype="string"
            ),
            "alternative": pd.array(
                np.asarray(site_alternative, dtype=object)[chosen], dtype="string"
            ),
            "source": pd.array(
                np.asarray(site_sources, dtype=object)[chosen], dtype="string"
            ),
            **{name: counts[:, code] for code, name in enumerate(CALLS)},
        },
        index=pd.Index(shown, name="site"),
    )
    cell_table = pd.DataFrame(
        {
            "n_mutant_sites": np.bincount(row_cells[mutant], minlength=n_cells),
            "n_called_sites": np.bincount(row_cells[called], minlength=n_cells),
        },
        index=pd.Index(np.asarray(cell_ids, dtype=object), name="ids"),
    )
    return {"sites": site_table, "cells": cell_table, "calls": call_table}


@dataclass(frozen=True, slots=True)
class VariantGenotypes:
    """Genotype calls of the cells at targeted sites, read from a ``variant_genotypes`` artifact.

    A cell that the ``calls`` table does not list at a site holds no evidence
    there, which is the call ``missing``.

    Attributes:
        sites: One row per site shown, indexed by its label:
            ``reference`` and ``alternative``, the alleles the calls compare
            (empty for adopted source calls), ``source`` (the call columns
            that ``method="source"`` adopted, in the order a cell's call is
            taken from them, else empty) and the cells of each call, those
            that the calls do not list counted ``missing``.
        cells: One row per cell, indexed by cell ID: ``n_mutant_sites``
            (sites shown called heterozygous, homozygous alternative or
            alternative present) and ``n_called_sites`` (those and the sites
            called wild type).
        calls: One row per cell and site with evidence: ``cell``, ``site``,
            ``call`` (one of :data:`CALLS`, ordered), ``label`` (the source's
            own label, empty for the other methods), ``reference_umis`` and
            ``alternative_umis`` (-1 where the counts hold none), and the
            posteriors ``posterior_wild_type``, ``posterior_heterozygous``
            and ``posterior_homozygous_alternative`` (NaN unless
            ``method="gift"``).
        counts: The ``variant_counts`` artifact the calls came from.
        method: ``"source"``, ``"threshold"`` or ``"gift"``.
        rules: The rules as the method applied them.
    """

    sites: pd.DataFrame = field(repr=False)
    cells: pd.DataFrame = field(repr=False)
    calls: pd.DataFrame = field(repr=False)
    counts: ArtifactRef
    method: str
    rules: Mapping[str, Any]
