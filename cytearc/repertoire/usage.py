"""Gene usage and junction lengths of groups of cells.

A repertoire's V, D, J and constant genes, and the lengths of its junctions,
summarize how its receptors were assembled: a sample dominated by a few
expanded clones shows their genes, and a public or invariant receptor shows
its gene pair. Each cell counts by its chains in the chosen slots, as scirpy
counts them, and each slot's locus keeps the chains of each arm apart.
Gene calls are named at one of three levels: as called, with their allele
(``"TRBV12-3*01"``), as genes (``"TRBV12-3"``), or as families
(``"TRBV12"``). These functions work on labels and never touch storage.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef

__all__ = [
    "GENE_FIELDS",
    "GENE_LEVELS",
    "GeneUsage",
    "Spectratype",
    "count_units",
    "gene_names",
]

#: The gene calls that usage counts.
GENE_FIELDS = ("v_call", "d_call", "j_call", "c_call")
#: The levels at which gene calls are named.
GENE_LEVELS = ("allele", "gene", "family")
# A V, D or J gene's family: its locus, segment and first number.
_FAMILY = re.compile(r"^([A-Z]+[VDJ]\d+)")


def gene_names(calls: Any, level: str) -> np.ndarray:
    """Name gene calls at a level, None where a call is missing.

    A call naming several genes, such as ``"TRBV6-2,TRBV6-3"``, keeps its
    first. ``"allele"`` keeps the call; ``"gene"`` drops the allele after
    ``*``; ``"family"`` keeps the locus, segment and first number of a V, D
    or J gene, such as ``"TRBV12"`` for ``"TRBV12-3"``, ``"TRAV29"`` for
    ``"TRAV29/DV5"`` and ``"IGKV1"`` for ``"IGKV1D-39"``, and the gene of a
    constant region.

    Raises:
        ValueError: If ``level`` is not one of ``GENE_LEVELS``.
    """
    if level not in GENE_LEVELS:
        raise ValueError(f"level must be one of {', '.join(GENE_LEVELS)}")
    values = np.asarray(calls, dtype=object).reshape(-1)
    # Each distinct call is named once.
    codes, distinct = pd.factorize(values, use_na_sentinel=True)
    names = np.full(len(distinct), None, dtype=object)
    for position, call in enumerate(np.asarray(distinct, dtype=object).tolist()):
        if not isinstance(call, str) or not call.strip():
            continue
        name = call.split(",")[0].strip()
        if level != "allele":
            name = name.split("*")[0]
        if level == "family":
            match = _FAMILY.match(name)
            if match is not None:
                name = match.group(1)
        names[position] = name
    named = np.full(len(values), None, dtype=object)
    named[codes >= 0] = names[codes[codes >= 0]]
    return named


def count_units(
    units: pd.DataFrame, keys: list[str], values: list[str], count: str
) -> pd.DataFrame:
    """Count units by keys and values, with each count's share of its keys.

    Args:
        units: One row per unit, such as a cell's chain, with the ``keys``
            and ``values`` columns; rows with a missing value are left out.
        keys: The columns whose combinations share one total, such as the
            group and locus.
        values: The counted columns, such as the V gene.
        count: The name of the count column.

    Returns:
        One row per combination of keys and values with a unit, in the
        natural order of the keys and by count, most first: the keys, the
        values, ``count`` and ``fraction``.
    """
    kept = units.dropna(subset=values)
    if kept.empty:
        return pd.DataFrame(columns=[*keys, *values, count, "fraction"])
    table = (
        kept.groupby([*keys, *values], sort=False, dropna=False)
        .size()
        .rename(count)
        .reset_index()
    )
    totals = table.groupby(keys, sort=False, dropna=False)[count].transform("sum")
    table["fraction"] = table[count] / totals
    table[count] = table[count].astype(np.int64)
    return table.sort_values(
        [*keys, count, *values],
        ascending=[*([True] * len(keys)), False, *([True] * len(values))],
        kind="stable",
    ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class GeneUsage:
    """The gene usage of groups of cells.

    Attributes:
        table: One row per group, locus and gene, or gene combination: the
            group's key columns, ``locus``, one column per gene field, the
            ``cells`` or ``clonotypes`` with the genes, and their
            ``fraction`` of the group's chains of that locus.
        source: The ``immune_receptors`` or ``clonotypes`` artifact.
        genes: The gene fields, such as ``("v_call", "j_call")``.
        level: ``"allele"``, ``"gene"`` or ``"family"``.
        slots: The chain slots counted.
        count: ``"cells"`` or ``"clonotypes"``.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
        n_cells: The cells with a counted chain.
    """

    table: pd.DataFrame = field(repr=False)
    source: ArtifactRef
    genes: tuple[str, ...]
    level: str
    slots: tuple[str, ...]
    count: str
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
    n_cells: int


@dataclass(frozen=True, slots=True)
class Spectratype:
    """The junction lengths of groups of cells.

    Attributes:
        table: One row per group, locus and length: the group's key columns,
            ``locus``, ``length`` (residues or nucleotides), the ``cells`` or
            ``clonotypes`` with junctions of that length, and their
            ``fraction`` of the group's chains of that locus.
        source: The ``immune_receptors`` or ``clonotypes`` artifact.
        sequence: ``"aa"`` for residues or ``"nt"`` for nucleotides.
        slots: The chain slots counted.
        count: ``"cells"`` or ``"clonotypes"``.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
        n_cells: The cells with a counted chain.
    """

    table: pd.DataFrame = field(repr=False)
    source: ArtifactRef
    sequence: str
    slots: tuple[str, ...]
    count: str
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
    n_cells: int
