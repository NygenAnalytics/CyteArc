"""Call clones from rare lineage recorder alleles.

A lineage recorder's edits are inherited, so cells that share an allele
descend from the cell in which it arose, unless it arose more than once:
CRISPR target arrays repeat their most likely edits in unrelated cells,
which is called homoplasy. Rare alleles mark clones, frequent ones do not.
An allele here is one character's edit: the text of a mark or of an unknown
mark that names its allele, at one site of one integration, such as a whole
DARLIN array's ``23_268del,55_248del``. The unedited state, placeholders of
unknown marks such as CARLIN's ``X``, and missing or inapplicable sites
name no allele, so they never link cells.

Two methods decide which alleles are rare, within each group of cells such
as a mouse:

- ``"graph"``, DARLIN's rule (Li et al., Cell 2023): an allele is rare when
  its frequency is at most ``max_frequency``, and when it was found in at
  most ``max_samples`` of its bank's samples, such as the mice in which a
  bank was collected.
- ``"p_clonal"``, CARLIN's test (Bowling et al., Cell 2020): an allele of
  bank rate λ among the M cells of its group with an edit at its character
  arose more than once with probability p_clonal = (1 - e^(-λM) - λM
  e^(-λM)) / (1 - e^(-λM)), and is rare when that p-value survives the
  Benjamini-Hochberg procedure at ``fdr`` over every allele of its group,
  and when it passes ``max_samples``.

An allele's frequency is its ``normalized_count`` in the bank of its
character, the share of the bank's edited observations that hold it, as
DARLIN's and CARLIN's banks give it; an allele missing from the bank counts
as rare, with a frequency of 0, as DARLIN's paper counts it. Without a bank,
an allele's frequency is the share of the cells of its group with an edit at
its character that hold it, which clonal expansion inflates.

Cells of a group that share a rare allele, at any character, are linked, and
the connected components are the clones, through
:func:`~cytearc.clones.calling.set_clones` with ``method="shared"``. A clone
whose cells hold more than ``max_alleles`` distinct rare alleles leaves its
cells without one, as DARLIN keeps clones of at most three alleles: a cell
holds at most one allele per array, so a clone of more is likely two clones
that a homoplasic allele joined.

These functions never touch storage.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import pdtrc

from ..utils.arguments import float_argument, integer_argument, quoted_value
from ..utils.arrays import sorted_unique, unique_inverse
from .calling import cell_sets, set_clones

__all__ = [
    "ALLELE_METHODS",
    "BANK_COLUMNS",
    "AlleleBank",
    "AlleleCallSettings",
    "AlleleClones",
    "AlleleObservations",
    "allele_banks",
    "allele_call_settings",
    "allele_clone_calls",
    "benjamini_hochberg",
    "clonal_p_values",
]

#: The methods that decide which alleles are rare.
ALLELE_METHODS: tuple[str, ...] = ("graph", "p_clonal")
#: The columns of an allele bank that calling reads: each allele's text, its
#: frequency among the bank's edited observations and, where the bank counts
#: them, the samples that hold it, as DARLIN's banks name them.
BANK_COLUMNS: tuple[str, ...] = ("allele", "normalized_count", "sample_count")
_ALLELE, _FREQUENCY, _SAMPLES = BANK_COLUMNS
# Bytes of each observation, a cell's edit at a character, while calling sorts
# them: its cell, character and allele in input order, their order and sorted
# copies, the masks of new runs and the distinct observations.
_OBSERVATION_BYTES = 88
# Bytes of each observation that names an allele while its test is found:
# its cell, character, allele and group, their order and sorted copies, and
# its test and item.
_NAMED_BYTES = 96
# Bytes of each test, an allele of a group and character: its group,
# character, allele, cells, observations, rate, samples, p-value, order and
# flags.
_TEST_BYTES = 160
# Bytes of each distinct allele of a character with a bank as its text is
# looked up: its Python string and the lookup's table.
_LOOKUP_BYTES = 120
# Bytes of each observation that names an allele while the rare ones become
# pairs of a cell and an item: whether it is rare, and its cell, item and their
# copies.
_ITEM_BYTES = 40
# Bytes of each rare pair of a cell and an allele while the clones' alleles
# are counted: the clone of its cell, its item, their order and copies.
_CLONE_ALLELE_BYTES = 40
# Bytes of each cell while its group is copied as int64, and once copied; of
# each cell while the clones are numbered, its flag and code; and of each
# clone, its alleles, flag and rank.
_GROUP_COPY_BYTES = 16
_GROUP_BYTES = 8
_NUMBERING_CELL_BYTES = 9
_NUMBERING_CLONE_BYTES = 24
# Bytes that each step holds whatever its size.
_FIXED_BYTES = 1 << 16
# Below this mean, p_clonal is its series x / 2 - x² / 12, exact to the
# float's precision there, since the Poisson tails underflow near the
# smallest floats.
_SERIES_BELOW = 1e-8


def _check(needed: int, max_bytes: int | None, what: str) -> None:
    if max_bytes is not None and needed > max_bytes:
        raise MemoryError(
            f"Calling clones from recorder alleles holds about {needed} bytes for "
            f"{what}, more than the {max_bytes} bytes that the memory budget leaves "
            "it. Raise mem_budget."
        )


@dataclass(slots=True)
class _Room:
    """The bytes that calling holds beside its inputs, and the most it may hold."""

    limit: int | None
    held: int = 0

    def check(self, needed: int, what: str) -> None:
        _check(self.held + needed, self.limit, what)

    def left(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.held)


@dataclass(frozen=True, slots=True)
class AlleleBank:
    """One character's bank: the frequency and samples of each allele it lists.

    Attributes:
        alleles: The alleles' text, distinct and sorted by code point.
        frequencies: Each allele's frequency among the bank's edited
            observations, from 0 to 1, as float64.
        samples: The samples that hold each allele as int64, or None when
            the bank counts none.
    """

    alleles: np.ndarray
    frequencies: np.ndarray
    samples: np.ndarray | None

    def __len__(self) -> int:
        return len(self.alleles)

    def positions(self, texts: np.ndarray) -> np.ndarray:
        """Each text's position among the bank's alleles, -1 for one it lacks."""
        return np.asarray(
            pd.Index(self.alleles, dtype=object).get_indexer(
                np.asarray(texts, dtype=object)
            ),
            dtype=np.int64,
        )


def _bank_texts(values: pd.Series, character: str) -> np.ndarray:
    texts: np.ndarray = values.to_numpy(dtype=object)
    bad = [
        position
        for position, value in enumerate(texts.tolist())
        if not isinstance(value, str)
    ]
    if bad:
        examples = ", ".join(quoted_value(texts[position]) for position in bad[:3])
        raise ValueError(
            f"The bank of {quoted_value(character)} holds {len(bad)} alleles that are not text, "
            f"such as {examples}; every allele must be text, as the import writes it"
        )
    # The import strips each token, so the bank's alleles compare stripped.
    stripped: np.ndarray = np.asarray(
        [value.strip() for value in texts.tolist()], dtype=object
    )
    return stripped


def _bank_numbers(values: pd.Series, character: str, column: str) -> np.ndarray:
    try:
        numbers: np.ndarray = pd.to_numeric(values, errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"The bank of {quoted_value(character)} holds {column} values that are not numbers"
        ) from None
    bad = int(np.count_nonzero(~np.isfinite(numbers)))
    if bad:
        raise ValueError(
            f"The bank of {quoted_value(character)} holds {bad} {column} values that are missing "
            "or not finite"
        )
    return numbers


def allele_banks(frequencies: Any) -> dict[str, AlleleBank]:
    """Check the banks a user gives, one table per character, and sort them.

    Args:
        frequencies: A mapping from each character's name, its integration
            or ``{integration}-{site}`` as ``export_characters`` names it, to
            a pandas DataFrame with the columns ``allele``, the allele's
            text, compared without surrounding spaces as the import reads
            alleles, and ``normalized_count``, its frequency among the
            bank's edited observations from 0 to 1, and optionally
            ``sample_count``, the samples that hold it, as DARLIN's banks
            write them. Other columns are not read.

    Raises:
        TypeError: If ``frequencies`` is not a mapping of names to
            DataFrames.
        ValueError: If a bank lacks a column, holds an allele twice, an
            allele that is not text, a frequency outside 0 to 1 or a sample
            count that is not a whole number of at least 0, or holds no row.
    """
    if not isinstance(frequencies, Mapping):
        raise TypeError(
            "frequencies must map each character's name to its bank, a DataFrame "
            "with the columns allele and normalized_count, or be None"
        )
    if not frequencies:
        raise ValueError(
            "frequencies names no bank; pass None to take each allele's frequency "
            "from the cells"
        )
    banks: dict[str, AlleleBank] = {}
    for character, table in frequencies.items():
        if not isinstance(character, str) or not character:
            raise TypeError("frequencies must be keyed by character names, as text")
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"The bank of {quoted_value(character)} must be a pandas DataFrame, not "
                f"{type(table).__name__}"
            )
        missing = [name for name in (_ALLELE, _FREQUENCY) if name not in table.columns]
        if missing:
            raise ValueError(
                f"The bank of {quoted_value(character)} lacks the column {quoted_value(missing[0])}; a bank "
                f"needs {quoted_value(_ALLELE)} and {quoted_value(_FREQUENCY)}, and {quoted_value(_SAMPLES)} for "
                "max_samples, as DARLIN's banks name them"
            )
        if not len(table):
            raise ValueError(f"The bank of {quoted_value(character)} holds no allele")
        texts = _bank_texts(table[_ALLELE], character)
        rates = _bank_numbers(table[_FREQUENCY], character, _FREQUENCY)
        outside = (rates < 0.0) | (rates > 1.0)
        if np.any(outside):
            raise ValueError(
                f"The bank of {quoted_value(character)} holds {int(np.count_nonzero(outside))} "
                f"{_FREQUENCY} values outside 0 to 1, such as "
                f"{float(rates[outside][0])}; a frequency is a share of the bank's "
                "edited observations"
            )
        samples = None
        if _SAMPLES in table.columns:
            counts = _bank_numbers(table[_SAMPLES], character, _SAMPLES)
            wrong = (counts < 0) | (counts != np.floor(counts))
            if np.any(wrong):
                raise ValueError(
                    f"The bank of {quoted_value(character)} holds {int(np.count_nonzero(wrong))} "
                    f"{_SAMPLES} values that are not whole numbers of at least 0, "
                    f"such as {float(counts[wrong][0])}"
                )
            samples = counts.astype(np.int64)
        order = np.argsort(texts, kind="stable")
        texts = texts[order]
        repeated = np.flatnonzero(texts[1:] == texts[:-1])
        if len(repeated):
            examples = ", ".join(
                quoted_value(texts[position]) for position in repeated[:3]
            )
            raise ValueError(
                f"The bank of {quoted_value(character)} lists {len(repeated)} alleles more than "
                f"once, such as {examples}; each allele needs one row"
            )
        banks[character] = AlleleBank(
            alleles=texts,
            frequencies=rates[order],
            samples=None if samples is None else samples[order],
        )
    return banks


@dataclass(frozen=True, slots=True)
class AlleleCallSettings:
    """The resolved rules of one call of ``allele_clones``.

    Attributes:
        method: ``"graph"`` or ``"p_clonal"``.
        max_frequency: The highest frequency of a rare allele, for
            ``"graph"``; None for no such filter.
        max_samples: The most samples of a bank in which a rare allele is
            found, or None for no such filter.
        max_alleles: The most distinct rare alleles of a clone, or None for
            no such filter.
        fdr: The false discovery rate of the Benjamini-Hochberg procedure,
            for ``"p_clonal"``; None for ``"graph"``.
    """

    method: str
    max_frequency: float | None
    max_samples: int | None
    max_alleles: int | None
    fdr: float | None

    def record(self) -> dict[str, Any]:
        """The resolved rules as the artifact records them."""
        return {
            "method": self.method,
            "max_frequency": self.max_frequency,
            "max_samples": self.max_samples,
            "max_alleles": self.max_alleles,
            "fdr": self.fdr,
        }


def allele_call_settings(
    method: Any,
    *,
    max_frequency: Any = None,
    max_samples: Any = None,
    max_alleles: Any = None,
    fdr: Any = 0.05,
    banks: Mapping[str, AlleleBank] | None = None,
) -> AlleleCallSettings:
    """Resolve and check the rules of a call.

    ``"graph"`` needs ``max_frequency``, ``max_samples`` or both, since a
    graph through every allele joins clones through their most frequent
    edits. ``"p_clonal"`` needs a bank, since rates measured in the cells
    themselves give an allele seen in n cells a mean λM of n, so every
    p_clonal is at least (e - 2) / (e - 1), about 0.42, and it tests each
    allele in place of ``max_frequency``. ``max_samples`` needs banks that count
    each allele's samples.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the method is unknown, a value is out of range, or a
            rule is missing or does not apply to the method or the banks.
    """
    if not isinstance(method, str) or method not in ALLELE_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(map(repr, ALLELE_METHODS))}, not "
            f"{quoted_value(method)}: 'graph' links cells through alleles at most max_frequency, "
            "as DARLIN does, and 'p_clonal' through alleles whose CARLIN p-value "
            "survives the false discovery rate"
        )
    frequency = None
    if max_frequency is not None:
        frequency = float_argument(max_frequency, "max_frequency")
        if not 0.0 < frequency <= 1.0:
            raise ValueError(
                f"max_frequency must be greater than 0 and at most 1, not {frequency}"
            )
    samples = (
        None
        if max_samples is None
        else integer_argument(max_samples, "max_samples", minimum=0)
    )
    alleles = (
        None
        if max_alleles is None
        else integer_argument(max_alleles, "max_alleles", minimum=1)
    )
    rate = float_argument(fdr, "fdr")
    if not 0.0 < rate < 1.0:
        raise ValueError(f"fdr must be greater than 0 and less than 1, not {rate}")
    if samples is not None:
        if banks is None:
            raise ValueError(
                "max_samples counts the samples of a bank that hold an allele, its "
                "sample_count; pass frequencies, the banks of the characters"
            )
        uncounted = sorted(name for name, bank in banks.items() if bank.samples is None)
        if uncounted:
            raise ValueError(
                f"max_samples reads each bank's sample_count, which the bank of "
                f"{quoted_value(uncounted[0])} lacks"
            )
    if method == "graph":
        if frequency is None and samples is None:
            raise ValueError(
                "method='graph' links cells through rare alleles, so it needs "
                "max_frequency, max_samples or both: DARLIN linked cells through "
                "alleles of frequency below 2e-5 found in at most one mouse of its "
                "bank, which max_frequency=float(np.nextafter(2e-5, 0)) and "
                "max_samples=1 reproduce"
            )
        return AlleleCallSettings(
            method=method,
            max_frequency=frequency,
            max_samples=samples,
            max_alleles=alleles,
            fdr=None,
        )
    if frequency is not None:
        raise ValueError(
            "max_frequency is the highest frequency at which method='graph' links "
            "cells; "
            "method='p_clonal' tests each allele's p-value at fdr instead"
        )
    if banks is None:
        raise ValueError(
            "method='p_clonal' tests each allele against the rate of a bank of "
            "independently edited cells, so it needs frequencies: rates taken from "
            "the cells themselves give every allele a p_clonal of about 0.42 or "
            "more, so no allele would survive"
        )
    return AlleleCallSettings(
        method=method,
        max_frequency=None,
        max_samples=samples,
        max_alleles=alleles,
        fdr=rate,
    )


def clonal_p_values(rates: Any, observations: Any) -> np.ndarray:
    """CARLIN's p_clonal: the chance that an allele arose more than once.

    An allele that arises at rate λ per observation arises X times among M
    observations, X Poisson with mean x = λM, and given that it was seen,
    p_clonal = P(X > 1 | X > 0) = (1 - e^(-x) - x e^(-x)) / (1 - e^(-x)).
    It is computed as the Poisson tails P(X ≥ 2) / P(X ≥ 1), which stay
    accurate as x approaches 0, and below 1e-8 as x / 2 - x² / 12, the
    leading terms of their ratio, where the tails would underflow; a rate of
    0, an allele missing from the bank, gives 0.

    Args:
        rates: Each allele's rate, at least 0.
        observations: The observations, M, of each allele's sample.

    Returns:
        Each allele's p_clonal, as float64.
    """
    means = np.asarray(rates, dtype=np.float64) * np.asarray(
        observations, dtype=np.float64
    )
    if np.any(~np.isfinite(means)) or np.any(means < 0):
        raise ValueError("rates and observations must be finite and at least 0")
    p_values = np.zeros(means.shape, dtype=np.float64)
    tails = means >= _SERIES_BELOW
    p_values[tails] = pdtrc(1, means[tails]) / pdtrc(0, means[tails])
    small = ~tails
    p_values[small] = means[small] / 2 - means[small] ** 2 / 12
    return p_values


def benjamini_hochberg(p_values: Any, families: Any, fdr: float) -> np.ndarray:
    """Which tests the Benjamini-Hochberg procedure keeps, within each family.

    Within a family of m tests ranked by p-value, the procedure keeps the
    tests of rank 1 to k, the largest rank whose p-value p satisfies
    p·m ≤ k·fdr; tests of equal p-value take consecutive ranks, so all of
    them pass when the last does.

    Args:
        p_values: Each test's p-value.
        families: Each test's family as an integer code; the procedure runs
            on each family apart.
        fdr: The false discovery rate.

    Returns:
        Whether each test survives.
    """
    values = np.asarray(p_values, dtype=np.float64)
    codes = np.asarray(families, dtype=np.int64)
    if values.shape != codes.shape or values.ndim != 1:
        raise ValueError("p_values and families need one value per test")
    n_tests = len(values)
    kept = np.zeros(n_tests, dtype=bool)
    if not n_tests:
        return kept
    order = np.lexsort((values, codes))
    sorted_codes = codes[order]
    starts = np.ones(n_tests, dtype=bool)
    starts[1:] = sorted_codes[1:] != sorted_codes[:-1]
    first = np.flatnonzero(starts)
    run = np.cumsum(starts) - 1
    ranks = np.arange(n_tests, dtype=np.int64) - first[run] + 1
    sizes = np.diff(np.append(first, n_tests))[run]
    passes = values[order] * sizes <= ranks * fdr
    largest = np.zeros(len(first), dtype=np.int64)
    np.maximum.at(largest, run[passes], ranks[passes])
    kept[order] = ranks <= largest[run]
    return kept


@dataclass(frozen=True, slots=True)
class AlleleObservations:
    """Each cell's edits that calling reads, one per record of an edited site.

    Attributes:
        cells: Each edit's cell, as int64.
        characters: Each edit's character, a position in
            ``character_names``, as int64.
        alleles: Each edit's allele, a position in ``allele_texts``, or -1
            for a placeholder of an unknown mark, which names no allele, as
            int64.
        character_names: The name of each character, ``{integration}-{site}``
            or its integration alone.
        allele_texts: The text of each allele, as an object array.
        unedited_characters: The names of the alleles' characters that hold
            no edit, such as an array that every cell reads unedited, which a
            bank may name.
    """

    cells: np.ndarray
    characters: np.ndarray
    alleles: np.ndarray
    character_names: tuple[str, ...]
    allele_texts: np.ndarray = field(repr=False)
    unedited_characters: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.cells)


@dataclass(frozen=True, slots=True)
class AlleleClones:
    """The clones called from rare alleles.

    Attributes:
        codes: Each cell's clone, numbered by size, largest first, then by
            first cell; -1 for none.
        sizes: The cells of each clone.
        n_alleles: Each cell's distinct rare alleles.
        clone_alleles: The distinct rare alleles of each clone.
        counts: What each rule found, by name.
        coverage: For each character with a bank, in the order of the
            characters: ``character``, its ``alleles`` and the
            ``alleles_in_bank``, and its ``observations``, the cells and
            alleles, and the ``observations_in_bank``.
    """

    codes: np.ndarray
    sizes: np.ndarray
    n_alleles: np.ndarray
    clone_alleles: np.ndarray
    counts: dict[str, int]
    coverage: list[dict[str, Any]]

    @property
    def n_clones(self) -> int:
        return len(self.sizes)


def _runs(*keys: np.ndarray) -> np.ndarray:
    """Whether each row of sorted keys starts a run of equal keys."""
    n_rows = len(keys[0])
    new = np.ones(n_rows, dtype=bool)
    if n_rows > 1:
        changed = np.zeros(n_rows - 1, dtype=bool)
        for key in keys:
            changed |= key[1:] != key[:-1]
        new[1:] = changed
    return new


def _group_codes(groups: np.ndarray | None, n_cells: int) -> np.ndarray:
    if groups is None:
        return np.zeros(n_cells, dtype=np.int64)
    codes = np.asarray(groups)
    if codes.shape != (n_cells,) or (n_cells and codes.dtype.kind not in "iu"):
        raise ValueError("groups must hold one integer code per cell")
    if n_cells and codes.min() < 0:
        raise ValueError("group codes must be at least 0")
    return codes.astype(np.int64)


def _check_observations(observations: AlleleObservations, n_cells: int) -> None:
    n_records = len(observations.cells)
    for name in ("characters", "alleles"):
        if len(getattr(observations, name)) != n_records:
            raise ValueError("Each observation needs one cell, character and allele")
    if not n_records:
        return
    if observations.cells.min() < 0 or observations.cells.max() >= n_cells:
        raise ValueError(f"cells must be positions below {n_cells}")
    n_characters = len(observations.character_names)
    if (
        observations.characters.min() < 0
        or observations.characters.max() >= n_characters
    ):
        raise ValueError("characters must be positions among the character names")
    if observations.alleles.min() < -1 or observations.alleles.max() >= len(
        observations.allele_texts
    ):
        raise ValueError("alleles must be positions among the allele texts, or -1")


def allele_clone_calls(
    observations: AlleleObservations,
    n_cells: int,
    *,
    groups: np.ndarray | None,
    settings: AlleleCallSettings,
    banks: Mapping[str, AlleleBank] | None = None,
    max_bytes: int | None = None,
) -> AlleleClones:
    """Call clones from each cell's rare alleles, within each group.

    Args:
        observations: The edits of the cells.
        n_cells: The cells, including those without an edit.
        groups: Each cell's group as an integer code, within which alleles
            are tested and clones called, or None for one group.
        settings: The rules, from :func:`allele_call_settings`.
        banks: The bank of each character by name, from
            :func:`allele_banks`, or None to take frequencies from the cells.
        max_bytes: The most bytes calling may hold beside its inputs, or
            None.

    Raises:
        ValueError: If the inputs are inconsistent, a bank names no
            character of the alleles, or two characters share a name that a
            bank names.
        MemoryError: If a step does not fit ``max_bytes``.
    """
    room = _Room(None if max_bytes is None else int(max_bytes))
    room.check(_GROUP_COPY_BYTES * n_cells + _FIXED_BYTES, "the cells")
    group_codes = _group_codes(groups, n_cells)
    room.held += _GROUP_BYTES * n_cells
    _check_observations(observations, n_cells)
    names = observations.character_names
    if banks is not None:
        known = set(names) | set(observations.unedited_characters)
        unknown = sorted(name for name in banks if name not in known)
        if unknown:
            listed = ", ".join(map(repr, sorted(known))) if known else "none"
            raise ValueError(
                f"frequencies names {quoted_value(unknown[0])}, which is no character of these "
                f"alleles; their characters are {listed}"
            )
        shared = sorted(name for name in set(names) if names.count(name) > 1)
        if shared:
            raise ValueError(
                f"Two characters of these alleles share the name {quoted_value(shared[0])}, so a "
                "bank cannot name one of them"
            )
    n_characters = max(1, len(names))
    n_allele_codes = max(1, len(observations.allele_texts))

    # The distinct edits: each cell's allele at a character once; then the
    # cells of each group with an edit at each character, the M of CARLIN's
    # test, a placeholder's cells among them; then the edits that name an
    # allele.
    room.check(
        _OBSERVATION_BYTES * len(observations) + _FIXED_BYTES, "the cells' edits"
    )
    order = np.lexsort(
        (observations.alleles, observations.characters, observations.cells)
    )
    cells = observations.cells[order]
    characters = observations.characters[order]
    alleles = observations.alleles[order]
    del order
    distinct = _runs(cells, characters, alleles)
    cells, characters, alleles = (
        cells[distinct],
        characters[distinct],
        alleles[distinct],
    )
    del distinct
    edited = _runs(cells, characters)
    edited_keys = group_codes[cells[edited]] * n_characters + characters[edited]
    del edited
    group_characters, edited_inverse = unique_inverse(edited_keys)
    edited_cells = np.bincount(edited_inverse, minlength=len(group_characters))
    del edited_keys, edited_inverse
    named = alleles >= 0
    cells, characters, alleles = cells[named], characters[named], alleles[named]
    del named
    n_named = len(cells)
    held_edits = 24 * n_named + 16 * len(group_characters)
    room.held += held_edits

    # Each test is an allele of a group at a character.
    room.check(_NAMED_BYTES * n_named + _FIXED_BYTES, "the alleles' tests")
    cell_groups = group_codes[cells]
    order = np.lexsort((alleles, characters, cell_groups))
    starts = _runs(cell_groups[order], characters[order], alleles[order])
    tests = np.empty(n_named, dtype=np.int64)
    tests[order] = np.cumsum(starts) - 1
    heads = order[starts]
    del order, starts
    test_groups = cell_groups[heads]
    test_characters = characters[heads]
    test_alleles = alleles[heads]
    del heads, cell_groups
    n_tests = len(test_groups)
    held_tests = 8 * n_named + 24 * n_tests
    room.held += held_tests
    room.check(_TEST_BYTES * n_tests + _FIXED_BYTES, "the alleles' tests")
    test_cells = np.bincount(tests, minlength=n_tests).astype(np.int64)
    test_observations = edited_cells[
        np.searchsorted(group_characters, test_groups * n_characters + test_characters)
    ].astype(np.int64)
    del group_characters, edited_cells
    rates, samples, coverage = _bank_rates(
        observations, banks, test_characters, test_alleles, tests, room
    )
    if banks is None:
        rates = test_cells / test_observations
    counts = {
        "alleles": n_tests,
        "alleles_too_frequent": 0,
        "alleles_not_significant": 0,
        "alleles_in_too_many_samples": 0,
    }
    if settings.method == "graph":
        rare = np.ones(n_tests, dtype=bool)
        if settings.max_frequency is not None:
            rare = rates <= settings.max_frequency
            counts["alleles_too_frequent"] = int(np.count_nonzero(~rare))
    else:
        assert settings.fdr is not None
        p_values = clonal_p_values(rates, test_observations)
        rare = benjamini_hochberg(p_values, test_groups, settings.fdr)
        del p_values
        counts["alleles_not_significant"] = int(np.count_nonzero(~rare))
    if settings.max_samples is not None:
        assert samples is not None
        common = rare & (samples > settings.max_samples)
        counts["alleles_in_too_many_samples"] = int(np.count_nonzero(common))
        rare &= ~common
        del common
    counts["rare_alleles"] = int(np.count_nonzero(rare))
    del rates, samples, test_cells, test_observations
    del test_groups, test_characters, test_alleles

    # Cells that share a rare allele link; the components are the clones.
    room.check(_ITEM_BYTES * n_named + _FIXED_BYTES, "the cells' rare alleles")
    linked = rare[tests]
    del tests, rare
    linked_cells = cells[linked]
    items = characters[linked]
    items *= n_allele_codes
    items += alleles[linked]
    del linked, cells, characters, alleles
    room.held -= held_edits + held_tests
    n_linked = len(items)
    room.held += 16 * n_linked
    sets = cell_sets(linked_cells, items, n_cells, max_bytes=room.left())
    del linked_cells, items
    room.held += 8 * (n_cells + 1 + len(sets.items)) - 16 * n_linked
    calls = set_clones(sets, groups=group_codes, method="shared", max_bytes=room.left())
    n_alleles = sets.sizes
    # The calls hold each cell's clone and its doublet and ambiguous flags and
    # each clone's size, beside each cell's rare alleles.
    room.held += 18 * n_cells + 8 * calls.n_clones + 8 * n_cells
    counts["cells_with_rare_alleles"] = int(np.count_nonzero(n_alleles > 0))

    # A clone's alleles are the distinct rare alleles of its cells.
    room.check(
        _CLONE_ALLELE_BYTES * len(sets.items)
        + _NUMBERING_CELL_BYTES * n_cells
        + _NUMBERING_CLONE_BYTES * calls.n_clones
        + _FIXED_BYTES,
        "the clones",
    )
    in_clone = calls.codes >= 0
    pair_clones = np.repeat(calls.codes, n_alleles)
    kept_pairs = pair_clones >= 0
    pair_clones = pair_clones[kept_pairs]
    pair_items = np.asarray(sets.items, dtype=np.int64)[kept_pairs]
    del kept_pairs, sets
    order = np.lexsort((pair_items, pair_clones))
    pair_clones = pair_clones[order]
    first = _runs(pair_clones, pair_items[order])
    del order, pair_items
    clone_alleles = np.bincount(pair_clones[first], minlength=calls.n_clones).astype(
        np.int64
    )
    del pair_clones, first
    keep = np.ones(calls.n_clones, dtype=bool)
    if settings.max_alleles is not None:
        keep = clone_alleles <= settings.max_alleles
    counts["clones_with_too_many_alleles"] = int(np.count_nonzero(~keep))
    counts["cells_in_clones_with_too_many_alleles"] = int(calls.sizes[~keep].sum())
    # Clones keep their order, by size and then first cell, as they are
    # renumbered without those left out.
    rank = np.full(calls.n_clones, -1, dtype=np.int64)
    rank[keep] = np.arange(int(np.count_nonzero(keep)))
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[in_clone] = rank[calls.codes[in_clone]]
    return AlleleClones(
        codes=codes,
        sizes=calls.sizes[keep].astype(np.int64),
        n_alleles=n_alleles.astype(np.int64),
        clone_alleles=clone_alleles[keep],
        counts=counts,
        coverage=coverage,
    )


def _bank_rates(
    observations: AlleleObservations,
    banks: Mapping[str, AlleleBank] | None,
    test_characters: np.ndarray,
    test_alleles: np.ndarray,
    tests: np.ndarray,
    room: _Room,
) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, Any]]]:
    """Each test's bank rate and samples, and each bank's coverage.

    An allele missing from its character's bank has rate 0 and no samples.

    Returns:
        Each test's rate (zeros without banks), its samples or None when a
        bank counts none, and the coverage of each character with a bank.
    """
    n_tests = len(test_characters)
    rates = np.zeros(n_tests, dtype=np.float64)
    if banks is None:
        return rates, None, []
    names = observations.character_names
    present = sorted_unique(test_characters).tolist()
    lacking = sorted({names[character] for character in present} - set(banks))
    if lacking:
        raise ValueError(
            f"frequencies holds no bank of {quoted_value(lacking[0])}, whose cells hold alleles; "
            "pass a bank for every character with alleles, or import only the "
            "characters that have one"
        )
    counted = all(bank.samples is not None for bank in banks.values())
    samples = np.zeros(n_tests, dtype=np.int64) if counted else None
    observed = np.bincount(tests, minlength=n_tests)
    coverage: list[dict[str, Any]] = []
    for character in present:
        name = names[character]
        bank = banks[name]
        chosen = np.flatnonzero(test_characters == character)
        codes, inverse = unique_inverse(test_alleles[chosen])
        texts = observations.allele_texts[codes]
        room.check(
            _LOOKUP_BYTES * len(codes)
            + sum(len(text) for text in texts.tolist())
            + _LOOKUP_BYTES * len(bank)
            + _FIXED_BYTES,
            f"the alleles of {quoted_value(name)} in its bank",
        )
        found = bank.positions(texts)
        positions = found[inverse]
        listed = positions >= 0
        rates[chosen[listed]] = bank.frequencies[positions[listed]]
        if samples is not None:
            assert bank.samples is not None
            samples[chosen[listed]] = bank.samples[positions[listed]]
        allele_observations = np.bincount(inverse, weights=observed[chosen])
        coverage.append(
            {
                "character": name,
                "alleles": len(codes),
                "alleles_in_bank": int(np.count_nonzero(found >= 0)),
                "observations": int(allele_observations.sum()),
                "observations_in_bank": int(allele_observations[found >= 0].sum()),
            }
        )
    coverage.sort(key=lambda entry: str(entry["character"]))
    return rates, samples, coverage
