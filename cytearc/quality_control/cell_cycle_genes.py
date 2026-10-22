"""Cell-cycle marker genes with current HGNC symbols (Seurat 2019 update)."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "cell_cycle_previous_symbols",
    "g2m_phase_genes",
    "g2m_phase_genes_mouse",
    "match_gene_names",
    "s_phase_genes",
    "s_phase_genes_mouse",
]

s_phase_genes: list[str] = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "CENPU",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]

g2m_phase_genes: list[str] = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "PIMREG",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "JPT1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]


def _mouse_symbol(human: str) -> str:
    """Title-case mouse ortholog naming used by common Seurat-style lists."""
    if not human:
        return human
    return human[0] + human[1:].lower()


s_phase_genes_mouse: list[str] = [_mouse_symbol(name) for name in s_phase_genes]
g2m_phase_genes_mouse: list[str] = [_mouse_symbol(name) for name in g2m_phase_genes]

#: Previous HGNC symbols of the default cell-cycle genes, from the HGNC REST
#: service (genenames.org, CC0, retrieved October 2026). Annotations made
#: before a rename name the gene by an earlier symbol, such as MLF1IP for
#: CENPU, FAM64A for PIMREG, and HN1 for JPT1. Only symbols that no other gene
#: holds as its symbol, previous symbol, or alias are listed, so each names
#: one gene; CDC2 (CDK1), TS (TYMS), CCNL1 (MCM2), CDC25 (CDC25C), and SD
#: (RANGAP1) are left out. Names match without case, so a mouse feature with
#: the same earlier symbol, such as ``Fam64a`` or ``Hn1``, matches too; MGI
#: symbols that differ from the human ones are not listed.
cell_cycle_previous_symbols: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "MCM5": ("CDC46",),
        "FEN1": ("RAD2",),
        "MCM2": ("CDCL1",),
        "MCM4": ("CDC21",),
        "UNG": ("DGU",),
        "CENPU": ("MLF1IP",),
        "UBR7": ("C14orf130",),
        "MSH2": ("COCA1",),
        "RAD51": ("RAD51A", "RECA"),
        "RRM2": ("C2orf48",),
        "CDC45": ("CDC45L2", "CDC45L"),
        "CDC6": ("CDC18L",),
        "POLA1": ("POLA", "NSX"),
        "HMGB2": ("HMG2",),
        "BIRC5": ("API4",),
        "TPX2": ("C20orf2", "C20orf1"),
        "TOP2A": ("TOP2",),
        "NDC80": ("KNTC2",),
        "NUF2": ("CDCA1",),
        "PIMREG": ("FAM64A",),
        "SMC4": ("SMC4L1",),
        "AURKB": ("STK12",),
        "BUB1": ("BUB1L",),
        "KIF11": ("KNSL1",),
        "TUBB4B": ("TUBB2C",),
        "KIF20B": ("MPHOSPH1",),
        "JPT1": ("HN1",),
        "KIF2C": ("KNSL6",),
        "DLGAP5": ("DLG7",),
        "KIF23": ("KNSL5",),
        "AURKA": ("STK15", "STK6"),
        "G2E3": ("KIAA1333",),
    }
)


def _symbol_alternatives() -> dict[str, tuple[str, ...]]:
    """Map each symbol of a renamed gene to its other symbols, in upper case.

    A current symbol maps to its previous symbols, and a previous symbol to
    the current symbol and then to the other previous symbols.
    """
    alternatives: dict[str, tuple[str, ...]] = {}
    for current, previous in cell_cycle_previous_symbols.items():
        symbols = (current, *previous)
        for symbol in symbols:
            alternatives[symbol.upper()] = tuple(
                other.upper() for other in symbols if other != symbol
            )
    return alternatives


_SYMBOL_ALTERNATIVES = _symbol_alternatives()


@dataclass(frozen=True, slots=True)
class GeneMatch:
    """The features that a list of gene names matched.

    Attributes:
        indices: Feature indices in the order of the genes. A gene matches
            every feature with its name, so a gene listed twice, or names that
            differ only in case, repeat indices.
        aliases: Genes matched by another symbol of the same gene, mapped to
            the feature name they matched.
        unmatched: Genes that matched no feature.
    """

    indices: list[int]
    aliases: dict[str, str]
    unmatched: list[str]


def match_gene_names(
    feature_names: Sequence[object] | np.ndarray,
    genes: Sequence[str],
) -> GeneMatch:
    """Match gene names to features without case, falling back to renames.

    A gene matches every feature whose name equals it without case. A gene
    that matches none matches the features named by another symbol of the
    same gene in ``cell_cycle_previous_symbols``: a previous symbol for a
    current one, or the current symbol for a previous one, in that order.

    Args:
        feature_names: The name of each feature.
        genes: Gene names to match.

    Returns:
        The matched feature indices, the aliases used, and the genes that
        matched nothing.
    """
    by_name: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        by_name.setdefault(str(name).upper(), []).append(index)
    indices: list[int] = []
    aliases: dict[str, str] = {}
    unmatched: list[str] = []
    for gene in genes:
        found = by_name.get(gene.upper())
        if found is None:
            for alternative in _SYMBOL_ALTERNATIVES.get(gene.upper(), ()):
                found = by_name.get(alternative)
                if found is not None:
                    aliases[gene] = str(feature_names[found[0]])
                    break
        if found is None:
            unmatched.append(gene)
            continue
        indices.extend(found)
    return GeneMatch(indices, aliases, unmatched)
