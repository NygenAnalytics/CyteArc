"""Call clones from lineage barcodes, by the rules of the method that made them.

Cells that carry the same lineage barcodes descend from one labelled cell.
The barcodes that extraction tools report hold noise that each method's
published rules remove before grouping cells: molecules of few reads, which
are mostly sequencing errors and chimeras; barcodes a few substitutions or
edits from an abundant one, which are its sequencing errors; barcodes
outside the library's allowlist; barcodes of few UMIs in a cell, which are
ambient or contaminating molecules; and cells with too few or too many
barcodes. :func:`barcode_clone_calls` applies the rules in this order:

1. molecule filters: ``min_reads``, the fewest reads of a molecule;
2. correction: within each group and library, barcodes within ``distance``
   of each other by ``metric`` are linked, and each connected component
   merges into its most abundant barcode, ties going to the barcode that
   sorts last, as TREX picks its representatives; then ``allowlist``;
3. pair filters: ``min_umis``, the fewest UMIs of a cell's barcode in a
   library, and ``min_fraction``, the share of the cell's top barcode of the
   library that it must reach, as BARtab filters;
4. cell filters: ``barcodes_per_cell``, the fewest and most barcodes of a
   cell that is called;
5. grouping: :func:`~cytearc.clones.calling.set_clones` on each cell's set of
   barcodes, by identical sets or a Jaccard graph, with TREX's pruning and
   doublets, and clones of at least ``min_size`` cells.

``PRESETS`` holds the published rules of LARRY, TREX, CellTagR and
CellTag-multi; every value a preset sets can be given instead.

Counts are read in each library's unit. A molecule counts one UMI; a pair
counts its UMIs where its library counts UMIs for every pair, or its reads
in a library that counts no UMIs for any pair but reads for every one, such
as CellTag-multi's scATAC-seq tags. A library that counts UMIs for some pairs
and not for others, or neither, has no unit. A rule that needs a count it
cannot read refuses rather than guess: ``min_reads`` needs every molecule's
reads, correction and the pair filters need a unit for every library they
test, and TREX's rules need molecules with their reads.

TREX's preset adds TREX's own rules for its partial cloneIDs, in which ``-``
marks a base that no read covered and ``0`` a deletion: molecules whose
barcode has a Shannon entropy below 1 bit are dropped before correction, two
cloneIDs are compared at the positions where both hold a base and need 20 of
them, a cloneID needs 20 bases after correction, a one-UMI barcode of a cell
is dropped when another cell of its group holds the barcode, and so is a
barcode whose only molecule has one read. These rules follow TREX's
``run10x``, adapted from its MIT-licensed source.

These functions never touch storage.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log2
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..metadata.records import take_text
from ..utils.arguments import integer_argument
from ..utils.arrays import sorted_unique
from .calling import SetClones, cell_sets, set_clones
from .sequences import neighbor_pairs

__all__ = [
    "BARCODE_METRICS",
    "PRESETS",
    "BarcodeCallSettings",
    "BarcodeClones",
    "BarcodeRecords",
    "barcode_clone_calls",
    "barcode_call_settings",
    "clone_labels",
]

type Unit = Literal["umis", "reads"]

#: The metrics of barcode correction.
BARCODE_METRICS: tuple[str, ...] = ("hamming", "levenshtein")
# A value of PRESETS that the user must give, such as a library's allowlist.
_REQUIRED = "required"
#: The published rules of each preset, by argument of ``barcode_clones``.
#: ``similarity`` is one value, or a pair: within one library and across
#: libraries. ``"required"`` marks a value that only the user can give.
PRESETS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        # Weinreb et al. 2018 (bioRxiv v2): triples of at least 10 reads,
        # barcodes within Hamming distance 3 collapsed by connected
        # components, pairs of at least 3 UMIs, and clones of cells with
        # identical barcode sets.
        "larry": MappingProxyType(
            {
                "min_reads": 10,
                "distance": 3,
                "metric": "hamming",
                "min_umis": 3,
                "method": "sets",
                "min_size": 1,
            }
        ),
        # TREX's run10x: cloneIDs merged within 5 mismatches over 20 bases
        # that both hold, a Jaccard index above 0.7, bridges pruned, local
        # cut vertices removed as doublets, and singletons kept.
        "trex": MappingProxyType(
            {
                "distance": 5,
                "metric": "hamming",
                "method": "jaccard",
                "similarity": 0.7,
                "inclusive": False,
                "prune": True,
                "doublets": True,
                "min_size": 1,
            }
        ),
        # CellTagR and Biddy et al. 2018: at least 2 UMIs, the library's
        # allowlist, 2 to 20 CellTags per cell, a Jaccard index above 0.7, and
        # clones of at least 3 cells.
        "celltag": MappingProxyType(
            {
                "allowlist": _REQUIRED,
                "min_umis": 2,
                "barcodes_per_cell": (2, 20),
                "method": "jaccard",
                "similarity": 0.7,
                "inclusive": False,
                "min_size": 3,
            }
        ),
        # CellTag-multi (Jindal et al. 2024): one-read scRNA-seq molecules
        # dropped, Starcode within 2 edits, the allowlist, more than 1 UMI or,
        # in scATAC-seq, read, 2 to 25 CellTags per cell, and a Jaccard index
        # above 0.6 within a modality and 0.5 across; its clone tables hold
        # clones of 2 or more cells.
        "celltag_multi": MappingProxyType(
            {
                "min_reads": 2,
                "distance": 2,
                "metric": "levenshtein",
                "allowlist": _REQUIRED,
                "min_umis": 2,
                "barcodes_per_cell": (2, 25),
                "method": "jaccard",
                "similarity": (0.6, 0.5),
                "inclusive": False,
                "min_size": 2,
            }
        ),
    }
)
# TREX's rules for its partial cloneIDs: the characters that hold no base,
# the bases two cloneIDs must both hold to be compared and a cloneID must
# hold to be kept, and the entropy below which a cloneID is low complexity.
_TREX_UNKNOWN = "-0"
_TREX_MIN_BASES = 20
_TREX_MIN_ENTROPY = 1.0
# The ways of grouping barcode sets, and the arguments that only a
# method="jaccard" call uses.
_METHODS = ("sets", "jaccard")
_JACCARD_ONLY = ("similarity", "inclusive", "prune", "doublets")
# Calling holds, beside its inputs, for each cell its group, barcodes, flag
# and modality, and its set's pointer; and for each record, until the records
# collapse into pairs, its kept and molecule flags, block and corrected
# barcode.
_CELL_BYTES = 40
_HELD_RECORD_BYTES = 18
# Each step that reads the records holds more for each record while it runs,
# beyond what is held: looking up barcodes, the kept codes, their sorted copy
# and the flags looked up; the correction, each kept record's count in its
# library's unit, its block and code, their order and sorted copies, and its
# entry; and the collapse into pairs, the copies of the kept records' fields,
# their order and their sorted copies, with each run's sums. The whole call
# held at most 113 bytes per record of 441,000 molecules and 138 per record of
# 144,000 pairs, as traced.
_LOOKUP_RECORD_BYTES = 32
_CORRECTION_RECORD_BYTES = 96
_PAIRING_RECORD_BYTES = 136
# Each distinct barcode of a block while it is corrected: its block, code,
# abundance, component and representative, beside the text that the search
# holds; each decoded barcode, its Python string and its place in the list of
# texts, beside four bytes per character; each stored barcode, its flag in a
# lookup; and each allowlisted barcode, its entry in the index looked up.
_ENTRY_BYTES = 64
_TEXT_BYTES = 120
_FLAG_BYTES = 1
_ALLOWED_BYTES = 200
# Each pair of a cell and a barcode is held as its cell, library, barcode,
# UMIs, reads and kept flag; the pair filters hold its count, its block's
# total or its cell's top count with their sorts; and building the sets, the
# copies of its cell and barcode passed.
_PAIR_BYTES = 48
_FILTER_PAIR_BYTES = 80
_SET_ARGUMENT_BYTES = 16


@dataclass(slots=True)
class _Room:
    """The bytes that calling holds beside its inputs, and the most it may hold."""

    limit: int | None
    held: int = 0

    def check(self, needed: int, what: str) -> None:
        """Raise MemoryError if ``needed`` more bytes would pass the limit."""
        _check(self.held + needed, self.limit, what)

    def left(self) -> int | None:
        """The bytes a step that checks its own peak may hold."""
        return None if self.limit is None else max(0, self.limit - self.held)


def _check(needed: int, max_bytes: int | None, what: str) -> None:
    if max_bytes is not None and needed > max_bytes:
        raise MemoryError(
            f"Calling clones from lineage barcodes holds about {needed} bytes for "
            f"{what}, more than the memory budget of {max_bytes} bytes. Raise "
            "mem_budget."
        )


@dataclass(frozen=True, slots=True)
class BarcodeCallSettings:
    """The resolved rules of one call of ``barcode_clones``.

    Attributes:
        preset: The preset that set the defaults, or None.
        min_reads: The fewest reads of a molecule, 0 for no filter.
        distance: The largest distance of corrected barcodes, 0 for none.
        metric: ``"hamming"`` or ``"levenshtein"``, None without correction.
        allowlist: Whether an allowlist applies.
        min_umis: The fewest UMIs, or reads, of a pair, 0 for no filter.
        min_fraction: The share of the cell's top barcode that a pair must
            reach, 0 for no filter.
        barcodes_per_cell: The fewest and the most barcodes of a called
            cell, the most None for no limit, or None for no filter.
        method: ``"sets"`` or ``"jaccard"``.
        similarity: The Jaccard index that links two cells, or the one
            within a library when ``across`` is set; None for sets.
        across: The Jaccard index that links cells of different libraries,
            or None.
        inclusive: Whether an index equal to the similarity links; None for
            sets.
        prune: Whether TREX's bridges are removed.
        doublets: Whether TREX's doublets are removed.
        min_size: The fewest cells of a clone.
        trex: Whether TREX's rules for partial cloneIDs and its pair filters
            apply.
    """

    preset: str | None
    min_reads: int
    distance: int
    metric: str | None
    allowlist: bool
    min_umis: int
    min_fraction: float
    barcodes_per_cell: tuple[int, int | None] | None
    method: str
    similarity: float | None
    across: float | None
    inclusive: bool | None
    prune: bool
    doublets: bool
    min_size: int
    trex: bool

    def record(self) -> dict[str, Any]:
        """The resolved rules as the artifact records them."""
        return {
            "preset": self.preset,
            "min_reads": self.min_reads,
            "distance": self.distance,
            "metric": self.metric,
            "min_umis": self.min_umis,
            "min_fraction": self.min_fraction,
            "barcodes_per_cell": None
            if self.barcodes_per_cell is None
            else list(self.barcodes_per_cell),
            "method": self.method,
            "similarity": None
            if self.similarity is None
            else (
                self.similarity
                if self.across is None
                else [self.similarity, self.across]
            ),
            "inclusive": self.inclusive,
            "prune": self.prune,
            "doublets": self.doublets,
            "min_size": self.min_size,
            "trex_rules": self.trex,
        }


def _count_argument(value: Any, name: str) -> int:
    return integer_argument(value, name, minimum=0)


def _similarities(value: Any) -> tuple[float, float | None]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError(
                "similarity is one number, or two: within one library and across "
                "libraries"
            )
        within, across = (_unit_number(part, "similarity") for part in value)
        return within, across
    return _unit_number(value, "similarity"), None


def _unit_number(value: Any, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"{name} must be a number from 0 to 1")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1, not {number}")
    return number


def _bounds(value: Any) -> tuple[int, int | None] | None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise TypeError(
            "barcodes_per_cell must be two numbers, the fewest and the most "
            "barcodes of a called cell, the most None for no limit"
        )
    low = integer_argument(value[0], "the fewest barcodes_per_cell", minimum=0)
    high = (
        None
        if value[1] is None
        else integer_argument(value[1], "the most barcodes_per_cell", minimum=1)
    )
    if high is not None and high < low:
        raise ValueError("barcodes_per_cell must not allow more than its most")
    if low <= 1 and high is None:
        # Every cell with a barcode passes, so the filter is none.
        return None
    return low, high


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def barcode_call_settings(
    preset: str | None = None,
    *,
    min_reads: Any = None,
    distance: Any = None,
    metric: Any = None,
    allowlist: Any = None,
    min_umis: Any = None,
    min_fraction: Any = None,
    barcodes_per_cell: Any = None,
    method: Any = None,
    similarity: Any = None,
    inclusive: Any = None,
    prune: Any = None,
    doublets: Any = None,
    min_size: Any = None,
) -> BarcodeCallSettings:
    """Resolve the rules of a call: a preset's values, replaced by those given.

    An argument left None takes the preset's value. Without a preset, every
    filter is off and correction does not run, and ``method`` is required;
    ``method="jaccard"`` needs ``similarity`` and ``inclusive``, and
    ``distance`` a ``metric``. A count of 0 turns its filter off, so
    ``min_reads=0`` lifts LARRY's read filter. ``allowlist=False`` calls a
    CellTag preset without the allowlist it requires.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the preset is unknown, a required value is missing, a
            value is out of range, or an argument is given to a method that
            does not use it.
    """
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)} or None")
    rules = dict(PRESETS[preset]) if preset is not None else {}
    given = {
        name: value
        for name, value in (
            ("min_reads", min_reads),
            ("distance", distance),
            ("metric", metric),
            ("allowlist", allowlist),
            ("min_umis", min_umis),
            ("min_fraction", min_fraction),
            ("barcodes_per_cell", barcodes_per_cell),
            ("method", method),
            ("similarity", similarity),
            ("inclusive", inclusive),
            ("prune", prune),
            ("doublets", doublets),
            ("min_size", min_size),
        )
        if value is not None
    }
    values = {**rules, **given}
    chosen = values.get("method")
    if chosen is None:
        raise ValueError(
            "barcode_clones needs a preset or a method: pass preset='larry', "
            "'trex', 'celltag' or 'celltag_multi', or method='sets' or 'jaccard' "
            "with the thresholds you choose"
        )
    if chosen not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(_METHODS)}")
    if chosen == "sets":
        # A preset's graph settings fall away under sets; given ones refuse.
        stray = sorted(name for name in _JACCARD_ONLY if name in given)
        if stray:
            raise ValueError(
                f"method='sets' groups identical barcode sets alone, so "
                f"{', '.join(stray)} would change nothing; use method='jaccard'"
            )
        for name in _JACCARD_ONLY:
            values.pop(name, None)
    else:
        for name, example in (
            ("similarity", "such as 0.7"),
            (
                "inclusive",
                "True links at a Jaccard index equal to the similarity, as "
                "bartools and STICR do, False only above it, as TREX and "
                "CellTagR do",
            ),
        ):
            if name not in values:
                raise ValueError(f"method='jaccard' needs {name}: {example}")
    distance_value = _count_argument(values.get("distance", 0), "distance")
    metric_value = values.get("metric")
    if distance_value == 0:
        if "metric" in given:
            raise ValueError(
                "metric applies to barcode correction, which a distance of 0 turns "
                "off; give a distance"
            )
        metric_value = None
    elif metric_value is None:
        raise ValueError(
            f"distance={distance_value} needs a metric: 'hamming' or 'levenshtein'"
        )
    elif metric_value not in BARCODE_METRICS:
        raise ValueError(f"metric must be one of {', '.join(BARCODE_METRICS)}")
    trex = preset == "trex"
    if trex and metric_value not in (None, "hamming"):
        raise ValueError(
            "preset='trex' compares cloneIDs at the positions where both hold a "
            "base, which needs metric='hamming'"
        )
    listed = values.get("allowlist")
    if isinstance(listed, np.bool_):
        listed = bool(listed)
    if isinstance(listed, str) and listed == _REQUIRED and "allowlist" not in given:
        raise ValueError(
            f"preset={preset!r} keeps the CellTags of the library's allowlist, as "
            "its published rules do; pass allowlist=, the CellTags of the "
            "library, or allowlist=False to call clones without one"
        )
    if listed is True or isinstance(listed, str | bytes):
        raise TypeError("allowlist must be a sequence of barcodes, False or None")
    has_allowlist = listed is not None and listed is not False
    fraction = values.get("min_fraction", 0.0)
    fraction_value = _unit_number(fraction, "min_fraction")
    if chosen == "jaccard":
        within, across = _similarities(values["similarity"])
        inclusive_value: bool | None = _flag(values["inclusive"], "inclusive")
    else:
        within, across, inclusive_value = None, None, None
    min_size_value = integer_argument(values.get("min_size", 1), "min_size", minimum=1)
    bounds = values.get("barcodes_per_cell")
    return BarcodeCallSettings(
        preset=preset,
        min_reads=_count_argument(values.get("min_reads", 0), "min_reads"),
        distance=distance_value,
        metric=metric_value,
        allowlist=has_allowlist,
        min_umis=_count_argument(values.get("min_umis", 0), "min_umis"),
        min_fraction=fraction_value,
        barcodes_per_cell=None if bounds is None else _bounds(bounds),
        method=chosen,
        similarity=within,
        across=across,
        inclusive=inclusive_value,
        prune=_flag(values.get("prune", False), "prune"),
        doublets=_flag(values.get("doublets", False), "doublets"),
        min_size=min_size_value,
        trex=trex,
    )


@dataclass(frozen=True, slots=True)
class BarcodeRecords:
    """The records of a ``lineage_barcodes`` artifact that calling reads.

    Attributes:
        cells: Each record's cell, its position among the selected cells.
        sources: Each record's library, by position.
        barcodes: Each record's barcode, a code into the distinct barcodes,
            which are sorted, so a larger code sorts later.
        umi_codes: Each record's UMI as a code, -1 for a pair.
        umis: Each record's UMIs, 1 for a molecule, -1 where unknown.
        reads: Each record's reads, -1 where unknown.
        barcode_data: The UTF-8 bytes of the distinct barcodes.
        barcode_offsets: Where each distinct barcode starts and ends.
        libraries: Each library's name and level, ``"molecules"`` or
            ``"pairs"``.
    """

    cells: np.ndarray
    sources: np.ndarray
    barcodes: np.ndarray
    umi_codes: np.ndarray
    umis: np.ndarray
    reads: np.ndarray
    barcode_data: np.ndarray = field(repr=False)
    barcode_offsets: np.ndarray = field(repr=False)
    libraries: tuple[tuple[str, str], ...]

    @property
    def n_records(self) -> int:
        return len(self.cells)

    def texts(self, codes: np.ndarray) -> np.ndarray:
        """Decode the barcodes of some codes."""
        return take_text(self.barcode_data, self.barcode_offsets, codes)


@dataclass(frozen=True, slots=True)
class BarcodeClones:
    """The clones that the barcodes give, with what each rule removed.

    Attributes:
        calls: The clones of :func:`~cytearc.clones.calling.set_clones`.
        n_barcodes: Each cell's barcodes after the pair filters.
        units: Each library's unit, ``"umis"``, ``"reads"`` or None.
        counts: What each rule removed or changed, by name.
    """

    calls: SetClones
    n_barcodes: np.ndarray
    units: tuple[str | None, ...]
    counts: dict[str, int]


def _library_units(
    records: BarcodeRecords,
) -> tuple[tuple[Unit | None, ...], list[dict[str, Any]]]:
    """Each library's unit, with the records that count no UMIs or reads."""
    n_libraries = len(records.libraries)
    sources = np.asarray(records.sources, dtype=np.int64)
    n_records = np.bincount(sources, minlength=n_libraries)
    no_umis = np.bincount(sources[records.umis < 0], minlength=n_libraries)
    no_reads = np.bincount(sources[records.reads < 0], minlength=n_libraries)
    units: list[Unit | None] = []
    details = []
    for position, (name, level) in enumerate(records.libraries):
        held = int(n_records[position])
        if level == "molecules" or not no_umis[position]:
            unit: Unit | None = "umis"
        elif no_umis[position] == held and not no_reads[position]:
            unit = "reads"
        else:
            unit = None
        units.append(unit)
        details.append(
            {
                "library": name,
                "level": level,
                "records": held,
                "without_umis": int(no_umis[position]),
                "without_reads": int(no_reads[position]),
            }
        )
    return tuple(units), details


def _unknown_unit(detail: Mapping[str, Any], rule: str) -> ValueError:
    return ValueError(
        f"{rule} counts each barcode of a cell in its library's UMIs, or reads in a "
        f"library that counts no UMIs, and library {detail['library']!r} counts "
        f"UMIs for {detail['records'] - detail['without_umis']} of its "
        f"{detail['records']} pairs and reads for "
        f"{detail['records'] - detail['without_reads']}; import its counts again, or "
        "turn the rule off"
    )


def _texts_bytes(records: BarcodeRecords, n_stored: int) -> int:
    """The bytes of decoding every stored barcode, the most a lookup decodes."""
    characters = int(records.barcode_offsets[-1]) if n_stored else 0
    return _TEXT_BYTES * n_stored + 4 * characters


def _entropy(text: str) -> float:
    """The Shannon entropy of a text's characters, in bits."""
    counts = Counter(text).values()
    length = len(text)
    return -sum(count / length * log2(count / length) for count in counts)


def _known_bases(text: str) -> int:
    return len(text) - sum(text.count(character) for character in _TREX_UNKNOWN)


def allowlist_texts(allowlist: Any) -> np.ndarray:
    """The distinct barcodes of an allowlist, sorted, as an object array.

    Raises:
        TypeError: If it is one text, or holds anything but text.
        ValueError: If it holds no barcode.
    """
    if isinstance(allowlist, str | bytes) or not hasattr(allowlist, "__iter__"):
        raise TypeError("allowlist must be a sequence of barcodes, False or None")
    values = np.asarray(list(allowlist), dtype=object).reshape(-1)
    if not all(isinstance(value, str) for value in values.tolist()):
        raise TypeError("allowlist must hold barcodes as text")
    if not len(values):
        raise ValueError(
            "allowlist holds no barcode, so every barcode would be dropped; pass "
            "the library's barcodes, or allowlist=False"
        )
    return np.asarray(sorted(set(values.tolist())), dtype=object)


def clone_labels(n_clones: int) -> np.ndarray:
    """Name clones by rank: ``clone_1`` is the largest."""
    return np.asarray(
        [f"clone_{rank}" for rank in range(1, n_clones + 1)], dtype=object
    )


def _first_of_runs(*keys: np.ndarray) -> np.ndarray:
    """Whether each row of sorted keys starts a run of equal keys."""
    n_rows = len(keys[0])
    new = np.ones(n_rows, dtype=bool)
    if n_rows > 1:
        changed = np.zeros(n_rows - 1, dtype=bool)
        for key in keys:
            changed |= key[1:] != key[:-1]
        new[1:] = changed
    return new


def _record_counts(
    sources: np.ndarray,
    is_molecule: np.ndarray,
    umis: np.ndarray,
    reads: np.ndarray,
    units: Sequence[Unit | None],
    details: Sequence[Mapping[str, Any]],
    rule: str,
) -> np.ndarray:
    """Each record's count in its library's unit; refused where a library has none."""
    for position in np.unique(sources[~is_molecule]).tolist():
        if units[position] is None:
            raise _unknown_unit(details[position], rule)
    unit_reads = np.asarray([unit == "reads" for unit in units], dtype=bool)
    counts = np.where(unit_reads[sources], reads, umis)
    return np.where(is_molecule, 1, counts).astype(np.int64)


def _corrected(
    records: BarcodeRecords,
    kept: np.ndarray,
    blocks: np.ndarray,
    counts: np.ndarray,
    settings: BarcodeCallSettings,
    room: _Room,
) -> tuple[np.ndarray, int]:
    """Merge each kept record's barcode into its component's most abundant one.

    Barcodes are linked within each block, a group and a library, when
    their distance by the metric is within the distance; a connected
    component of linked barcodes merges into the barcode with the most
    UMIs, or reads, in the block, and of those the one that sorts last.

    Returns:
        Each kept record's barcode code after correction, and the distinct
        barcodes of the blocks that merged into another.
    """
    from .calling import Edges, connected_components

    codes = np.asarray(records.barcodes, dtype=np.int64)[kept]
    block_of = blocks[kept]
    order = np.lexsort((codes, block_of))
    new = _first_of_runs(block_of[order], codes[order])
    starts = np.flatnonzero(new)
    entry_of = np.empty(len(order), dtype=np.int64)
    entry_of[order] = np.cumsum(new) - 1
    entry_codes = codes[order][starts]
    entry_blocks = block_of[order][starts]
    abundance = np.add.reduceat(counts[order], starts) if len(starts) else starts
    del order, new
    n_entries = len(starts)
    distinct, positions = np.unique(entry_codes, return_inverse=True)
    lengths = records.barcode_offsets[distinct + 1] - records.barcode_offsets[distinct]
    # The entries and their texts are held while the search runs, which gets
    # what is left.
    entries = (
        _ENTRY_BYTES * n_entries + _TEXT_BYTES * len(distinct) + 4 * int(lengths.sum())
    )
    room.check(entries, f"correcting {n_entries} barcodes of their blocks")
    room.held += entries
    texts = records.texts(distinct)[positions.reshape(-1)]
    first, second = neighbor_pairs(
        texts,
        str(settings.metric),
        settings.distance,
        entry_blocks,
        max_bytes=room.left(),
        unknown=_TREX_UNKNOWN if settings.trex else "",
        min_compared=_TREX_MIN_BASES if settings.trex else 0,
    )
    del texts
    # The texts are freed; the entries and the pairs found are held while the
    # components are found, which get what is left.
    text_bytes = _TEXT_BYTES * len(distinct) + 4 * int(lengths.sum())
    pair_bytes = 16 * len(first)
    room.held += pair_bytes - text_bytes
    components = connected_components(
        n_entries, Edges(first=first, second=second), max_bytes=room.left()
    )
    del first, second
    room.held -= entries - text_bytes + pair_bytes
    # The last entry of each component, ordered by abundance and then code,
    # is its representative: the most abundant barcode, sorting last on ties.
    by_rank = np.lexsort((entry_codes, abundance, components))
    last = np.ones(n_entries, dtype=bool)
    if n_entries > 1:
        last[:-1] = components[by_rank][1:] != components[by_rank][:-1]
    representative = np.empty(int(components.max()) + 1 if n_entries else 0, np.int64)
    representative[components[by_rank][last]] = entry_codes[by_rank][last]
    merged = representative[components]
    return merged[entry_of], int(np.count_nonzero(merged != entry_codes))


def barcode_clone_calls(
    records: BarcodeRecords,
    n_cells: int,
    *,
    groups: np.ndarray,
    settings: BarcodeCallSettings,
    allowlist: np.ndarray | None = None,
    max_bytes: int | None = None,
) -> BarcodeClones:
    """Call clones from the records of lineage barcodes by ``settings``.

    Args:
        records: The molecules and pairs of the cells.
        n_cells: The selected cells, whose positions the records name.
        groups: Each cell's group, within which clones are called and
            barcodes corrected, as integer codes.
        settings: The rules, from :func:`barcode_call_settings`.
        allowlist: The sorted distinct barcodes of :func:`allowlist_texts`
            when ``settings.allowlist`` is set, else None.
        max_bytes: The most bytes the calling may hold beside its inputs.

    Raises:
        ValueError: If a rule needs a count that a library does not hold, or
            TREX's rules meet pairs.
        MemoryError: If a step does not fit ``max_bytes``.
    """
    if settings.allowlist != (allowlist is not None):
        raise ValueError("Pass the allowlist exactly when the settings use one")
    n_records = records.n_records
    n_libraries = len(records.libraries)
    group_codes = np.asarray(groups, dtype=np.int64)
    if group_codes.shape != (n_cells,):
        raise ValueError("groups must hold one code per cell")
    units, details = _library_units(records)
    counted: dict[str, int] = {}
    room = _Room(max_bytes)
    room.check(
        _CELL_BYTES * n_cells + _HELD_RECORD_BYTES * n_records,
        f"the {n_records} molecules and pairs",
    )
    room.held = _CELL_BYTES * n_cells + _HELD_RECORD_BYTES * n_records
    cells = np.asarray(records.cells, dtype=np.int64)
    sources = np.asarray(records.sources, dtype=np.int64)
    barcodes = np.asarray(records.barcodes, dtype=np.int64)
    reads = np.asarray(records.reads, dtype=np.int64)
    umis = np.asarray(records.umis, dtype=np.int64)
    levels = np.asarray(
        [level == "molecules" for _name, level in records.libraries], dtype=bool
    )
    is_molecule = levels[sources] if n_libraries else np.zeros(0, dtype=bool)
    keep = np.ones(n_records, dtype=bool)
    n_stored = max(0, len(records.barcode_offsets) - 1)
    if settings.trex:
        for detail in details:
            if detail["level"] != "molecules" and detail["records"]:
                raise ValueError(
                    "preset='trex' applies TREX's rules to molecules, and library "
                    f"{detail['library']!r} holds pairs of a cell and a barcode"
                )
        missing = reads < 0
        if np.any(missing):
            source = int(sources[np.argmax(missing)])
            of_library = sources == source
            raise ValueError(
                "preset='trex' drops a barcode whose only molecule has one read, "
                f"and {int(np.count_nonzero(missing & of_library))} of the "
                f"{int(np.count_nonzero(of_library))} molecules of library "
                f"{details[source]['library']!r} count no reads. TREX writes "
                "read_count to molecules.txt since September 2024; the reads.txt "
                "of an earlier run gives each molecule's reads, imported with "
                "format='reads' and columns={'cell': 'cell_id', 'umi': 'umi', "
                "'barcode': 'clone_id'}"
            )
        del missing
        room.check(
            _LOOKUP_RECORD_BYTES * n_records
            + _texts_bytes(records, n_stored)
            + _FLAG_BYTES * n_stored,
            "the cloneIDs' complexity",
        )
        distinct = sorted_unique(barcodes)
        low = np.zeros(n_stored, dtype=bool)
        low[distinct] = np.fromiter(
            (_entropy(text) < _TREX_MIN_ENTROPY for text in records.texts(distinct)),
            dtype=bool,
            count=len(distinct),
        )
        del distinct
        dropped = low[barcodes]
        keep &= ~dropped
        counted["molecules_of_low_complexity"] = int(np.count_nonzero(dropped))
        del low, dropped
    if settings.min_reads:
        if not np.any(levels):
            raise ValueError(
                "min_reads filters molecules by their reads, and every library of "
                "the barcodes holds pairs of a cell and a barcode; pass min_reads=0"
            )
        tested = keep & is_molecule
        unknown = tested & (reads < 0)
        if np.any(unknown):
            detail = details[int(sources[np.argmax(unknown)])]
            raise ValueError(
                "min_reads needs the reads of every molecule, and library "
                f"{detail['library']!r} counts none for {int(unknown.sum())} of "
                "its molecules; import a table with reads, or pass min_reads=0"
            )
        dropped = tested & (reads < settings.min_reads)
        keep &= ~dropped
        counted["molecules_below_min_reads"] = int(np.count_nonzero(dropped))
        del tested, unknown, dropped
    blocks = group_codes[cells] * max(n_libraries, 1) + sources
    corrected = barcodes.copy()
    n_kept = int(np.count_nonzero(keep))
    if settings.distance:
        room.check(
            _CORRECTION_RECORD_BYTES * n_kept,
            f"correcting {n_kept} molecules and pairs",
        )
        room.held += _CORRECTION_RECORD_BYTES * n_kept
        kept_index = np.flatnonzero(keep)
        # Each kept record's count in its library's unit ranks the barcodes.
        counts = _record_counts(
            sources[keep],
            is_molecule[keep],
            umis[keep],
            reads[keep],
            units,
            details,
            "Barcode correction",
        )
        corrected[kept_index], counted["barcodes_corrected"] = _corrected(
            records, keep, blocks, counts, settings, room
        )
        counted["records_corrected"] = int(
            np.count_nonzero(corrected[keep] != barcodes[keep])
        )
        del counts, kept_index
        room.held -= _CORRECTION_RECORD_BYTES * n_kept
    if allowlist is not None or settings.trex:
        room.check(
            _LOOKUP_RECORD_BYTES * n_records
            + _texts_bytes(records, n_stored)
            + _FLAG_BYTES * n_stored
            + (0 if allowlist is None else _ALLOWED_BYTES * len(allowlist)),
            "the barcodes looked up",
        )
        distinct = sorted_unique(corrected[keep])
        texts = records.texts(distinct)
        if allowlist is not None:
            allowed = np.zeros(n_stored, dtype=bool)
            allowed[distinct] = pd.Index(allowlist).get_indexer(texts) >= 0
            dropped = keep & ~allowed[corrected]
            keep &= ~dropped
            counted["records_outside_allowlist"] = int(np.count_nonzero(dropped))
            del allowed, dropped
        if settings.trex:
            short = np.zeros(n_stored, dtype=bool)
            short[distinct] = np.fromiter(
                (_known_bases(text) < _TREX_MIN_BASES for text in texts),
                dtype=bool,
                count=len(texts),
            )
            dropped = keep & short[corrected]
            keep &= ~dropped
            counted["molecules_of_few_bases"] = int(np.count_nonzero(dropped))
            del short, dropped
        del texts, distinct
    n_kept = int(np.count_nonzero(keep))
    room.check(
        _PAIRING_RECORD_BYTES * n_kept, f"collapsing {n_kept} molecules and pairs"
    )
    pairs = _cell_pairs(
        cells[keep],
        sources[keep],
        corrected[keep],
        np.asarray(records.umi_codes, dtype=np.int64)[keep],
        umis[keep],
        reads[keep],
        is_molecule[keep],
    )
    del keep, corrected, blocks, is_molecule
    n_pairs = len(pairs["cells"])
    counted["pairs"] = n_pairs
    # The records' flags are released, and the pairs held instead.
    room.held = _CELL_BYTES * n_cells + _PAIR_BYTES * n_pairs
    room.check(0, f"the {n_pairs} barcodes of the cells")
    keep_pair = np.ones(n_pairs, dtype=bool)
    if settings.trex:
        room.check(_FILTER_PAIR_BYTES * n_pairs, "TREX's filters of the barcodes")
        pair_blocks = (
            group_codes[pairs["cells"]] * max(n_libraries, 1) + pairs["sources"]
        )
        totals = _block_totals(pair_blocks, pairs["barcodes"], pairs["umis"])
        del pair_blocks
        single = pairs["umis"] == 1
        shared = single & (totals > 1)
        lone_read = single & (totals == 1) & (pairs["reads"] == 1)
        keep_pair &= ~(shared | lone_read)
        counted["pairs_of_one_umi_shared"] = int(np.count_nonzero(shared))
        counted["pairs_of_one_read"] = int(np.count_nonzero(lone_read))
        del totals, single, shared, lone_read
    if settings.min_umis or settings.min_fraction:
        rule = "min_umis" if settings.min_umis else "min_fraction"
        room.check(_FILTER_PAIR_BYTES * n_pairs, f"the barcodes that {rule} tests")
        pair_counts = _record_counts(
            pairs["sources"],
            np.zeros(n_pairs, dtype=bool),
            pairs["umis"],
            pairs["reads"],
            units,
            details,
            rule,
        )
        if settings.min_fraction:
            top = _top_counts(pairs["cells"], pairs["sources"], pair_counts, keep_pair)
        if settings.min_umis:
            dropped = keep_pair & (pair_counts < settings.min_umis)
            keep_pair &= ~dropped
            counted["pairs_below_min_umis"] = int(np.count_nonzero(dropped))
        if settings.min_fraction:
            dropped = keep_pair & (pair_counts / settings.min_fraction < top)
            keep_pair &= ~dropped
            counted["pairs_below_min_fraction"] = int(np.count_nonzero(dropped))
            del top
        del pair_counts
    room.check(_SET_ARGUMENT_BYTES * n_pairs, "the cells' sets of barcodes")
    room.held += _SET_ARGUMENT_BYTES * n_pairs
    sets = cell_sets(
        pairs["cells"][keep_pair],
        pairs["barcodes"][keep_pair],
        n_cells,
        max_bytes=room.left(),
    )
    n_barcodes = sets.sizes
    counted["cells_with_barcodes"] = int(np.count_nonzero(n_barcodes))
    if settings.barcodes_per_cell is not None:
        low_bound, high_bound = settings.barcodes_per_cell
        called = (n_barcodes >= low_bound) & (
            True if high_bound is None else n_barcodes <= high_bound
        )
        counted["cells_outside_barcodes_per_cell"] = int(
            np.count_nonzero((n_barcodes > 0) & ~called)
        )
        chosen = keep_pair & called[pairs["cells"]]
        # The sets of the called cells replace those of every cell.
        del sets
        sets = cell_sets(
            pairs["cells"][chosen],
            pairs["barcodes"][chosen],
            n_cells,
            max_bytes=room.left(),
        )
        del chosen, called
    modalities = None
    if settings.across is not None:
        modalities = _cell_modalities(
            pairs["cells"][keep_pair], pairs["sources"][keep_pair], n_cells, n_libraries
        )
    del pairs, keep_pair
    # The sets are held while the clones are called, which get what is left.
    room.held = _CELL_BYTES * n_cells + 8 * (len(sets.items) + n_cells + 1)
    calls = set_clones(
        sets,
        groups=group_codes,
        method=settings.method,
        similarity=settings.similarity,
        inclusive=settings.inclusive,
        across=settings.across,
        modalities=modalities,
        prune=settings.prune,
        doublets=settings.doublets,
        min_size=settings.min_size,
        max_bytes=room.left(),
    )
    return BarcodeClones(
        calls=calls,
        n_barcodes=n_barcodes.astype(np.int32),
        units=units,
        counts=counted,
    )


def _cell_pairs(
    cells: np.ndarray,
    sources: np.ndarray,
    barcodes: np.ndarray,
    umi_codes: np.ndarray,
    umis: np.ndarray,
    reads: np.ndarray,
    is_molecule: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collapse records into pairs of a cell and a barcode within each library.

    Molecules that correction gave one barcode and that share a UMI are one
    molecule, whose reads add up; a pair's UMIs are its distinct molecules,
    or, for counted pairs, their UMIs added up. A sum that holds an unknown
    count is unknown.
    """
    order = np.lexsort((umi_codes, barcodes, sources, cells))
    cells, sources, barcodes = cells[order], sources[order], barcodes[order]
    umi_codes, umis, reads = umi_codes[order], umis[order], reads[order]
    is_molecule = is_molecule[order]
    del order
    # A molecule is one UMI however many records name it; counted pairs that
    # correction merged add their UMIs.
    starts = np.flatnonzero(_first_of_runs(cells, sources, barcodes, umi_codes))
    from .barcodes import known_sums

    unit_umis = np.where(is_molecule[starts], 1, known_sums(umis, starts))
    unit_reads = known_sums(reads, starts)
    cells, sources, barcodes = cells[starts], sources[starts], barcodes[starts]
    pair_starts = np.flatnonzero(_first_of_runs(cells, sources, barcodes))
    return {
        "cells": cells[pair_starts],
        "sources": sources[pair_starts],
        "barcodes": barcodes[pair_starts],
        "umis": known_sums(unit_umis, pair_starts),
        "reads": known_sums(unit_reads, pair_starts),
    }


def _block_totals(
    blocks: np.ndarray, barcodes: np.ndarray, umis: np.ndarray
) -> np.ndarray:
    """Each pair's barcode's UMIs over every cell of its block."""
    order = np.lexsort((barcodes, blocks))
    starts = np.flatnonzero(_first_of_runs(blocks[order], barcodes[order]))
    run = np.cumsum(_first_of_runs(blocks[order], barcodes[order])) - 1
    sums = np.add.reduceat(umis[order], starts) if len(starts) else starts
    totals = np.empty(len(blocks), dtype=np.int64)
    totals[order] = sums[run]
    return totals


def _top_counts(
    cells: np.ndarray, sources: np.ndarray, counts: np.ndarray, kept: np.ndarray
) -> np.ndarray:
    """Each pair's cell's largest count in its library, among the kept pairs."""
    keys = cells * (int(sources.max()) + 1 if len(sources) else 1) + sources
    distinct, inverse = np.unique(keys, return_inverse=True)
    top = np.zeros(len(distinct), dtype=np.int64)
    np.maximum.at(top, inverse[kept], counts[kept])
    return top[inverse]


def _cell_modalities(
    cells: np.ndarray, sources: np.ndarray, n_cells: int, n_libraries: int
) -> np.ndarray:
    """Each cell's modality: the set of libraries its barcodes come from, coded."""
    if n_libraries > 62:
        raise ValueError(
            "A similarity across libraries tells apart at most 62 libraries; "
            "pass one similarity"
        )
    masks = np.zeros(n_cells, dtype=np.int64)
    np.bitwise_or.at(masks, cells, np.left_shift(np.int64(1), sources))
    return masks
