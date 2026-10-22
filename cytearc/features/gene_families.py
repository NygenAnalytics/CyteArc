"""Gene families that CyteArc recognizes by feature name.

Each family is a regular expression whose alternatives are anchored at the
start of a feature name. Families match case-insensitively, as the default HVG
blacklist and the default RNA percentage columns apply them, so mouse ``mt-``
genes are mitochondrial.

``ribosomal`` covers cytosolic and mitochondrial ribosomal protein genes (RPS,
RPL, MRPS, and MRPL), as the default ``percentRibo`` column does.
``mitochondrial`` covers ``MT-`` genes, as the default ``percentMito`` column
does. ``cellCycleCcn`` covers the cyclins, whose symbols continue ``CCN`` with
a letter (``CCNA2``, ``CCND1``, ``CCNT1``), and leaves out the CCN family of
matricellular proteins, ``CCN1`` to ``CCN6`` (formerly ``CYR61``, ``CTGF``,
``NOV``, and ``WISP1`` to ``WISP3``). ``histone`` covers the
replication-dependent histone genes under their pre-2020 cluster symbols
(``HIST1H2AC``) and under their current HGNC and MGI symbols: the linker
histones ``H1-1`` to ``H1-6`` (mouse ``H1f1`` to ``H1f6``), the cluster genes
``H2AC*``, ``H2BC*``, ``H3C*``, and ``H4C*``, and ``H2AW``, ``H3-4``, and
``H4-16``. Replacement histones such as ``H3-3A``, ``H2AZ1``, and ``H1-0`` are
not in it. ``sexLinked`` lists ten sex-linked genes by exact name. The default
HVG blacklist excludes every registered family, in registry order.
"""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from ..utils.arrays import regex_match_mask

__all__ = [
    "DEFAULT_PERCENT_PATTERNS",
    "GENE_FAMILY_PATTERNS",
    "PERCENT_FAMILIES",
    "gene_family_mask",
]

# Case-insensitive name pattern of each registered gene family.
GENE_FAMILY_PATTERNS = MappingProxyType(
    {
        "mitochondrial": "^MT-",
        "ribosomal": "^RPS|^RPL|^MRPS|^MRPL",
        "cellCycleCcn": "^CCN[A-Z]",
        "hla": "^HLA-",
        "h2": "^H2-",
        "histone": (
            "^HIST|^H1-[1-6]$|^H1F[1-6]$|^H2AC\\d+$|^H2BC\\d+$|^H3C\\d+$|^H4C\\d+$|"
            "^H2AW$|^H3-4$|^H4-16$"
        ),
        "sexLinked": (
            "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|"
            "^TMSB4Y$|^NLGN4Y$"
        ),
    }
)
# The gene family that each default RNA percentage column measures.
PERCENT_FAMILIES = MappingProxyType(
    {"percentMito": "mitochondrial", "percentRibo": "ribosomal"}
)
# The feature-name pattern of each default RNA percentage column.
DEFAULT_PERCENT_PATTERNS = MappingProxyType(
    {
        suffix: GENE_FAMILY_PATTERNS[family]
        for suffix, family in PERCENT_FAMILIES.items()
    }
)


def gene_family_mask(names: Sequence[Any] | np.ndarray, family: str) -> np.ndarray:
    """Return where feature names belong to one registered gene family.

    Args:
        names: Feature names. Each value is matched as text.
        family: A key of ``GENE_FAMILY_PATTERNS``.

    Returns:
        A boolean array aligned with ``names``.

    Raises:
        ValueError: If ``family`` is not registered.
    """
    pattern = GENE_FAMILY_PATTERNS.get(family)
    if pattern is None:
        raise ValueError(f"Unknown gene family {family!r}")
    return regex_match_mask(names, pattern)
