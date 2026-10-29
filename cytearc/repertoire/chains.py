"""The receptor chains of each cell and the quality of their pairing.

A T cell receptor pairs an alpha chain (TRA) with a beta chain (TRB), or a
gamma chain (TRG) with a delta chain (TRD), and a B cell receptor pairs a heavy
chain (IGH) with a kappa (IGK) or lambda (IGL) light chain. The chains whose
genes recombine V and J segments, TRA, TRG, IGK and IGL, form the VJ arm of a
receptor, and those that also join a D segment, TRB, TRD and IGH, its VDJ arm.

CyteArc follows the receptor model of scirpy (Sturm et al. 2020): a cell has
up to two chains of each arm, so allelically included cells with two alpha or
two light chains keep both. The chains that count are the contigs with a locus
of one of the seven loci, a junction amino-acid sequence and, unless
non-productive contigs are kept, a productive rearrangement. They are ranked
within each arm by UMI count, then duplicate count, then read count (AIRR's
``consensus_count``), each most first, then by the junction nucleotide and
amino-acid sequences, last in sort order first as scirpy ranks them, and last
by their order in the files, which an import puts in a fixed order. The first two chains of an arm are its primary and secondary chains
(``VJ_1``, ``VJ_2``, ``VDJ_1``, ``VDJ_2``); a cell with more than two chains
of an arm is a multichain cell, often a doublet.

Each cell's chains give its receptor type, subtype and chain pairing, with
scirpy's categories and rules:

- Receptor type: ``"TCR"`` (only T cell receptor chains), ``"BCR"`` (only B
  cell receptor chains), ``"ambiguous"`` (both), ``"multichain"`` and
  ``"no IR"`` (no chain, IR standing for immune receptor).
- Receptor subtype: ``"TRA+TRB"`` and ``"TRG+TRD"`` (chains of that T cell
  receptor only), ``"IGH"`` (heavy chains only), ``"IGH+IGK"`` and
  ``"IGH+IGL"`` (light chains of one kind, with or without heavy chains),
  ``"IGH+IGK/L"`` (heavy chains with kappa and lambda chains),
  ``"ambiguous"`` (any other combination, such as TRA with TRD, or kappa and
  lambda chains without a heavy chain), ``"multichain"`` and ``"no IR"``.
- Chain pairing: ``"single pair"`` (one VJ and one VDJ chain), ``"orphan VJ"``
  and ``"orphan VDJ"`` (chains of one arm only), ``"extra VJ"`` and
  ``"extra VDJ"`` (a pair and a second chain of that arm), ``"two full
  chains"`` (two chains of each arm), ``"ambiguous"`` (an ambiguous subtype),
  ``"multichain"`` and ``"no IR"``.

These functions work on encoded contigs and never touch storage.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contigs import ContigTable, EncodedText

__all__ = [
    "CHAIN_PAIRINGS",
    "LOCI",
    "NO_RECEPTOR",
    "RECEPTOR_SUBTYPES",
    "RECEPTOR_TYPES",
    "SLOTS",
    "VDJ_LOCI",
    "VJ_LOCI",
    "ChainIndex",
    "call_locus",
    "ReceptorQuality",
    "chain_arms",
    "index_chains",
    "infer_loci",
    "receptor_quality",
]

#: Loci whose chains form the VJ arm of a receptor.
VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")
#: Loci whose chains form the VDJ arm of a receptor.
VDJ_LOCI = ("TRB", "TRD", "IGH")
#: Every locus of a receptor chain.
LOCI = VJ_LOCI + VDJ_LOCI
#: The chain slots of a cell, in the column order of ``ChainIndex.slots``.
SLOTS = ("VJ_1", "VJ_2", "VDJ_1", "VDJ_2")
#: The label of a cell without a receptor chain.
NO_RECEPTOR = "no IR"
#: Receptor types, in display order.
RECEPTOR_TYPES = ("TCR", "BCR", "ambiguous", "multichain", NO_RECEPTOR)
#: Receptor subtypes, in display order.
RECEPTOR_SUBTYPES = (
    "TRA+TRB",
    "TRG+TRD",
    "IGH",
    "IGH+IGK",
    "IGH+IGL",
    "IGH+IGK/L",
    "ambiguous",
    "multichain",
    NO_RECEPTOR,
)
#: Chain pairings, in display order.
CHAIN_PAIRINGS = (
    "single pair",
    "orphan VJ",
    "orphan VDJ",
    "extra VJ",
    "extra VDJ",
    "two full chains",
    "ambiguous",
    "multichain",
    NO_RECEPTOR,
)

_GENE_FIELDS = ("v_call", "d_call", "j_call", "c_call")


def call_locus(call: str) -> str | None:
    """Return the locus that a gene call names, "" for none, None for no locus.

    A call names the locus of its first three letters, uppercased, such as
    ``TRB`` for ``TRBV20-1*01``. A V call such as ``TRAV29/DV5`` names an alpha
    V gene that delta chains also use, so it names no locus and returns "".
    A call whose first three letters name no locus returns None.
    """
    text = str(call).strip()
    prefix = text[:3].upper()
    if prefix not in LOCI:
        return None
    if prefix == "TRA" and text[3:4].upper() == "V" and "/DV" in text.upper():
        return ""
    return prefix


def infer_loci(calls: Mapping[str, np.ndarray]) -> np.ndarray:
    """Infer each contig's locus from its gene calls.

    A contig's locus is the one locus that all of its V, D, J and constant gene
    calls name, so a contig with calls of two loci, or with a call that names
    no locus, has none. A V call such as ``TRAV29/DV5``, of a gene that alpha
    and delta chains share, takes no part, as in scirpy.

    Args:
        calls: The parsed ``v_call``, ``d_call``, ``j_call`` and ``c_call`` of
            each contig, None where missing; absent fields take no part.

    Returns:
        Each contig's locus, None where it cannot be inferred.
    """
    lengths = {len(np.asarray(values)) for values in calls.values()}
    if len(lengths) > 1:
        raise ValueError("Every gene call field needs one value per contig")
    n_contigs = lengths.pop() if lengths else 0
    inferred: list[str | None] = [""] * n_contigs
    conflict = np.zeros(n_contigs, dtype=bool)
    cache: dict[str, str | None] = {}
    for name in _GENE_FIELDS:
        values = calls.get(name)
        if values is None:
            continue
        for row, value in enumerate(np.asarray(values, dtype=object)):
            if value is None or conflict[row]:
                continue
            text = str(value)
            if text not in cache:
                cache[text] = call_locus(text)
            locus = cache[text]
            if locus is None:
                conflict[row] = True
            elif locus:
                if not inferred[row]:
                    inferred[row] = locus
                elif inferred[row] != locus:
                    conflict[row] = True
    result = np.asarray(inferred, dtype=object)
    result[conflict | (result == "")] = None
    return result


def chain_arms(locus: EncodedText) -> np.ndarray:
    """Return each contig's arm: 0 for VJ, 1 for VDJ, -1 for no locus of a chain."""
    by_category = np.full(len(locus.categories), -1, dtype=np.int8)
    for position, name in enumerate(np.asarray(locus.categories, dtype=object)):
        if name in VJ_LOCI:
            by_category[position] = 0
        elif name in VDJ_LOCI:
            by_category[position] = 1
    codes = np.asarray(locus.codes)
    arms = np.full(len(codes), -1, dtype=np.int8)
    present = codes >= 0
    arms[present] = by_category[codes[present]]
    return arms


@dataclass(frozen=True, slots=True)
class ChainIndex:
    """The ranked receptor chains of each cell.

    Attributes:
        slots: The position of each cell's ``VJ_1``, ``VJ_2``, ``VDJ_1`` and
            ``VDJ_2`` chain among the contigs, -1 for an empty slot, as an
            int64 array of one row per cell and four columns.
        n_vj: The number of chains of each cell's VJ arm, as int32.
        n_vdj: The number of chains of each cell's VDJ arm, as int32.
        eligible: Whether each contig counts as a chain.
    """

    slots: np.ndarray
    n_vj: np.ndarray
    n_vdj: np.ndarray
    eligible: np.ndarray

    @property
    def multichain(self) -> np.ndarray:
        """Whether each cell has more than two chains of an arm."""
        return np.asarray((self.n_vj > 2) | (self.n_vdj > 2), dtype=bool)

    @property
    def has_receptor(self) -> np.ndarray:
        """Whether each cell has at least one chain."""
        return np.asarray((self.n_vj + self.n_vdj) > 0, dtype=bool)


def _descending(codes: np.ndarray) -> np.ndarray:
    """Return a key that sorts ascending as ``codes`` sort descending."""
    return -np.asarray(codes, dtype=np.int64)


def _count_key(table: ContigTable, name: str) -> np.ndarray:
    """Return a count field for ranking, missing counted as 0, as scirpy does."""
    values = table.values.get(name)
    if values is None:
        return np.zeros(table.n_contigs, dtype=np.int64)
    return np.maximum(np.asarray(values, dtype=np.int64), 0)


def _text_codes(table: ContigTable, name: str) -> np.ndarray:
    encoded = table.text.get(name)
    if encoded is None:
        return np.full(table.n_contigs, -1, dtype=np.int32)
    return np.asarray(encoded.codes)


def index_chains(
    table: ContigTable,
    n_cells: int,
    *,
    locus: EncodedText,
    productive_only: bool = True,
) -> ChainIndex:
    """Rank each cell's chains and fill its primary and secondary slots.

    A contig counts as a chain when it has a locus of a receptor chain and a
    junction amino-acid sequence, is productive unless ``productive_only`` is
    False, and is not flagged False in ``is_cell`` or ``high_confidence``,
    Cell Ranger's flags of contigs outside a called cell or of low confidence.

    Args:
        table: The encoded contigs, ordered by cell.
        n_cells: The number of cells.
        locus: Each contig's locus, read or inferred.
        productive_only: Count only productive contigs as chains; otherwise a
            contig counts whatever its ``productive`` flag.

    Returns:
        The slots of each cell and the number of chains of each arm.
    """
    arms = chain_arms(locus)
    junction_aa = _text_codes(table, "junction_aa")
    eligible = (arms >= 0) & (junction_aa >= 0)
    if productive_only:
        productive = table.values.get("productive")
        if productive is None:
            eligible &= False
        else:
            eligible &= np.asarray(productive) == 1
    # Cell Ranger's all_contig_annotations.csv also holds contigs outside the
    # cells it calls and contigs of low confidence, which its filtered file
    # leaves out; a missing flag excludes nothing.
    for flag in ("is_cell", "high_confidence"):
        flags = table.values.get(flag)
        if flags is not None:
            eligible &= np.asarray(flags) != 0
    chosen = np.flatnonzero(eligible)
    cells = np.asarray(table.cells, dtype=np.int64)[chosen]
    chosen_arms = arms[chosen].astype(np.int64)
    # np.lexsort sorts by its last key first: cell, then arm, then the ranks.
    # The contigs' own order breaks full ties: by source, then by file row.
    order = np.lexsort(
        (
            chosen,
            _descending(junction_aa[chosen]),
            _descending(_text_codes(table, "junction")[chosen]),
            -_count_key(table, "consensus_count")[chosen],
            -_count_key(table, "duplicate_count")[chosen],
            -_count_key(table, "umi_count")[chosen],
            chosen_arms,
            cells,
        )
    )
    ranked = chosen[order]
    groups = cells[order] * 2 + chosen_arms[order]
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    rank = np.arange(len(ranked)) - np.repeat(
        starts, np.diff(np.r_[starts, len(ranked)])
    )
    counts = np.bincount(groups, minlength=2 * n_cells).reshape(n_cells, 2)
    slots = np.full((n_cells, 4), -1, dtype=np.int64)
    keep = rank < 2
    slot_columns = (groups[keep] % 2) * 2 + rank[keep]
    slots[groups[keep] // 2, slot_columns] = ranked[keep]
    return ChainIndex(
        slots=slots,
        n_vj=counts[:, 0].astype(np.int32),
        n_vdj=counts[:, 1].astype(np.int32),
        eligible=eligible,
    )


@dataclass(frozen=True, slots=True)
class ReceptorQuality:
    """The receptor type, subtype and chain pairing of each cell.

    Attributes:
        receptor_type: One of ``RECEPTOR_TYPES`` per cell.
        receptor_subtype: One of ``RECEPTOR_SUBTYPES`` per cell.
        chain_pairing: One of ``CHAIN_PAIRINGS`` per cell.
    """

    receptor_type: np.ndarray
    receptor_subtype: np.ndarray
    chain_pairing: np.ndarray


def receptor_quality(index: ChainIndex, locus: EncodedText) -> ReceptorQuality:
    """Classify each cell by the loci and number of its ranked chains.

    The rules are scirpy's ``chain_qc``, applied to the four slots of each
    cell: chains beyond a cell's second of an arm count only towards
    ``"multichain"``.

    Args:
        index: The ranked chains of each cell.
        locus: Each contig's locus, as ranked.

    Returns:
        The receptor type, subtype and chain pairing of each cell.
    """
    slots = np.asarray(index.slots)
    n_cells = slots.shape[0]
    codes = np.asarray(locus.codes)
    names = np.asarray(locus.categories, dtype=object)
    slot_loci = np.full(slots.shape, "", dtype=object)
    filled = slots >= 0
    slot_loci[filled] = names[codes[slots[filled]]]

    def has(name: str, columns: slice) -> np.ndarray:
        return np.asarray((slot_loci[:, columns] == name).any(axis=1), dtype=bool)

    vj, vdj = slice(0, 2), slice(2, 4)
    has_tra, has_trg = has("TRA", vj), has("TRG", vj)
    has_igk, has_igl = has("IGK", vj), has("IGL", vj)
    has_trb, has_trd, has_igh = has("TRB", vdj), has("TRD", vdj), has("IGH", vdj)
    has_tr = has_tra | has_trg | has_trb | has_trd
    has_ig = has_igk | has_igl | has_igh
    has_receptor = index.has_receptor
    multichain = has_receptor & index.multichain

    receptor_type = np.full(n_cells, "ambiguous", dtype=object)
    receptor_type[~has_receptor] = NO_RECEPTOR
    receptor_type[has_tr & ~has_ig] = "TCR"
    receptor_type[~has_tr & has_ig] = "BCR"
    receptor_type[multichain] = "multichain"

    receptor_subtype = np.full(n_cells, "ambiguous", dtype=object)
    receptor_subtype[~has_receptor] = NO_RECEPTOR
    receptor_subtype[(has_tra | has_trb) & ~(has_trg | has_trd | has_ig)] = "TRA+TRB"
    receptor_subtype[(has_trg | has_trd) & ~(has_tra | has_trb | has_ig)] = "TRG+TRD"
    receptor_subtype[has_igh & ~(has_igk | has_igl | has_tr)] = "IGH"
    receptor_subtype[has_igl & ~(has_tr | has_igk)] = "IGH+IGL"
    receptor_subtype[has_igk & ~(has_tr | has_igl)] = "IGH+IGK"
    receptor_subtype[has_igk & has_igl & has_igh & ~has_tr] = "IGH+IGK/L"
    receptor_subtype[multichain] = "multichain"

    has_vj1, has_vj2 = filled[:, 0], filled[:, 1]
    has_vdj1, has_vdj2 = filled[:, 2], filled[:, 3]
    pairs = has_vj1 & has_vdj1
    chain_pairing = np.full(n_cells, NO_RECEPTOR, dtype=object)
    chain_pairing[has_vj1] = "orphan VJ"
    chain_pairing[has_vdj1] = "orphan VDJ"
    chain_pairing[pairs] = "single pair"
    chain_pairing[pairs & has_vj2] = "extra VJ"
    chain_pairing[pairs & has_vdj2] = "extra VDJ"
    chain_pairing[pairs & has_vj2 & has_vdj2] = "two full chains"
    chain_pairing[receptor_subtype == "ambiguous"] = "ambiguous"
    chain_pairing[multichain] = "multichain"
    return ReceptorQuality(
        receptor_type=receptor_type,
        receptor_subtype=receptor_subtype,
        chain_pairing=chain_pairing,
    )
