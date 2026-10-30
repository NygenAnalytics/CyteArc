"""Mitochondrial variants that artifacts mimic, as the papers that found them list them.

Reads of a nuclear copy of mitochondrial DNA (a NUMT) that align to the
mitochondrial genome, differences between a transcript and the DNA it was
copied from, and homopolymers that sequencers misread all show as variants
in some cells. Each mask lists such variants in the coordinates of the
revised Cambridge Reference Sequence (rCRS, NC_012920), which hg38's
``chrM`` and GRCh38's ``MT`` share, with the paper that reports them:

- ``"numt"``: the 25 false positives that two polymorphic NUMTs, each
  carried by about 40% of people, cause when their reads align to the
  mitochondrial genome (Laricchia et al. 2022, Supplemental Table S1).
- ``"rna_editing"``: the RNA-DNA differences found in human mitochondrial
  transcripts at positions 295 (C to U), 2617 and 13710 (A to U and A to G)
  (Bar-Yaacov et al. 2013), which RNA libraries such as MAESTER's read as
  variants.
- ``"artifact_prone"``: every variant at positions 301, 302, 310, 316, 3107
  and 16182, the homopolymers and the reference's ``N`` that gnomAD filters
  (Laricchia et al. 2022).

These functions never touch storage.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["RCRS_LENGTH", "VARIANT_MASKS", "VariantMask", "masked_variants"]

#: The length of the revised Cambridge Reference Sequence, in bases.
RCRS_LENGTH = 16_569


@dataclass(frozen=True, slots=True)
class VariantMask:
    """A list of variants that artifacts mimic.

    Attributes:
        name: The name that ``mask`` takes.
        variants: Each variant as (position, reference, alternative) in rCRS
            coordinates; an empty alternative stands for every alternative at
            the position.
        source: The paper that lists the variants, as a DOI.
    """

    name: str
    variants: tuple[tuple[int, str, str], ...]
    source: str


_NUMT = (
    (12684, "G", "A"),
    (12705, "C", "T"),
    (13062, "A", "G"),
    (13095, "T", "C"),
    (13105, "A", "G"),
    (16189, "T", "A"),
    (16218, "C", "T"),
    (16230, "A", "G"),
    (16249, "T", "C"),
    (16274, "G", "A"),
    (16278, "C", "T"),
    (16284, "A", "G"),
    (16288, "T", "C"),
    (16290, "C", "T"),
    (16293, "A", "C"),
    (16301, "C", "T"),
    (16311, "T", "C"),
    (16355, "C", "T"),
    (16356, "T", "C"),
    (16368, "T", "C"),
    (16390, "G", "A"),
    (16399, "A", "G"),
    (16496, "G", "A"),
    (16519, "T", "C"),
    (16527, "C", "T"),
)
_RNA_EDITING = (
    (295, "C", "T"),
    (2617, "A", "G"),
    (2617, "A", "T"),
    (13710, "A", "G"),
    (13710, "A", "T"),
)
# The reference's N at 3107 is not a site of mgatk's counts, but other
# sources may hold one there.
_ARTIFACT_PRONE = (
    (301, "A", ""),
    (302, "A", ""),
    (310, "T", ""),
    (316, "G", ""),
    (3107, "N", ""),
    (16182, "A", ""),
)

#: The masks that ``mask`` names, by name.
VARIANT_MASKS: Mapping[str, VariantMask] = {
    mask.name: mask
    for mask in (
        VariantMask("numt", _NUMT, "10.1101/gr.276013.121"),
        VariantMask("rna_editing", _RNA_EDITING, "10.1101/gr.161265.113"),
        VariantMask("artifact_prone", _ARTIFACT_PRONE, "10.1101/gr.276013.121"),
    )
}


def masked_variants(
    names: Sequence[str],
    *,
    positions: np.ndarray,
    alternatives: np.ndarray,
    contig_lengths: np.ndarray,
) -> np.ndarray:
    """Mark the variants that the named masks list.

    A listed variant matches by position and alternative allele, whatever
    the counts' reference base there: a NUMT's reads show its base whichever
    base a cell's own mitochondrial DNA carries, and a source such as
    cellsnp-lite may call the major allele the reference.

    Args:
        names: Names of ``VARIANT_MASKS``.
        positions: Each variant's 1-based position, -1 for a named site.
        alternatives: Each variant's alternative allele.
        contig_lengths: The length of each variant's contig, -1 where the
            files do not say or the site is named.

    Returns:
        Whether each variant is masked.

    Raises:
        ValueError: If a name is unknown, or a variant lies on a contig whose
            length is not rCRS's, so its positions are not rCRS coordinates.
    """
    unknown = [name for name in names if name not in VARIANT_MASKS]
    if unknown:
        raise ValueError(
            f"Unknown masks {unknown}; masks are {', '.join(VARIANT_MASKS)}"
        )
    positions = np.asarray(positions, dtype=np.int64)
    masked = np.zeros(len(positions), dtype=bool)
    if not names or not len(positions):
        return masked
    lengths = np.asarray(contig_lengths, dtype=np.int64)
    if np.any(lengths != RCRS_LENGTH):
        row = int(np.flatnonzero(lengths != RCRS_LENGTH)[0])
        described = (
            "a named site"
            if positions[row] < 0
            else "a contig of unknown length"
            if lengths[row] < 0
            else f"a contig of {int(lengths[row])} bases"
        )
        raise ValueError(
            f"Masks list variants in rCRS coordinates on a {RCRS_LENGTH}-base "
            f"mitochondrial genome, but the counts hold a variant on {described}, "
            "such as hg19's 16,571-base chrM; pass mask=None"
        )
    alternatives = np.asarray(alternatives, dtype=object)
    by_position: dict[int, list[int]] = {}
    for row, position in enumerate(positions.tolist()):
        by_position.setdefault(position, []).append(row)
    for name in names:
        for position, _reference, alternative in VARIANT_MASKS[name].variants:
            for row in by_position.get(position, ()):
                if not alternative or alternatives[row] == alternative:
                    masked[row] = True
    return masked
