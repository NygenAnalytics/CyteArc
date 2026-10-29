"""Gene usage and junction lengths of groups of cells.

A repertoire's V, D, J and constant genes, and the lengths of its junctions,
summarize how its receptors were assembled: a sample dominated by a few
expanded clones shows their genes, and a public or invariant receptor shows
its gene pair. Each cell counts by its chains in the chosen slots, as scirpy
counts them, and each slot's locus keeps the chains of each arm apart.
Gene calls are named at one of three levels: as called, with their allele
(``"TRBV12-3*01"``), as genes (``"TRBV12-3"``), or as families
(``"TRBV12"``). Chains are counted by integer codes, and each distinct gene
call or junction is named or measured once. These functions work on labels
and codes and never touch storage.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef
from ..utils.arrays import sorted_unique, unique_rows
from .contigs import take_text

__all__ = [
    "GENE_FIELDS",
    "GENE_LEVELS",
    "GeneUsage",
    "Spectratype",
    "count_chains",
    "count_units",
    "gene_codes",
    "gene_names",
    "text_lengths",
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


def gene_codes(
    codes: np.ndarray, calls: np.ndarray, level: str
) -> tuple[np.ndarray, np.ndarray]:
    """Name the gene calls of chains at a level, naming each distinct call once.

    Args:
        codes: Each chain's call, a position in ``calls``, -1 for none.
        calls: The distinct calls.
        level: ``"allele"``, ``"gene"`` or ``"family"``, as
            :func:`gene_names` names calls.

    Returns:
        Each chain's name, a position in the names, -1 where the call is
        missing or names nothing, and the names, sorted.
    """
    named = gene_names(calls, level)
    present = np.asarray([name is not None for name in named.tolist()], dtype=bool)
    names = np.asarray(sorted(set(named[present].tolist())), dtype=object)
    ranks = np.full(len(named), -1, dtype=np.int64)
    ranks[present] = np.searchsorted(names, named[present])
    codes = np.asarray(codes, dtype=np.int64)
    named_codes = np.full(len(codes), -1, dtype=np.int64)
    known = codes >= 0
    named_codes[known] = ranks[codes[known]]
    return named_codes, names


def text_lengths(
    data: np.ndarray, offsets: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """The characters of some values of packed UTF-8 text.

    ASCII text, such as junction residues and nucleotides, has one byte per
    character, so its lengths follow from the offsets; other text is decoded.

    Args:
        data: The packed bytes.
        offsets: The offsets of every value in ``data``.
        positions: The values to measure, as positions.
    """
    positions = np.asarray(positions, dtype=np.int64)
    data = np.asarray(data)
    if not len(data) or int(data.max()) < 0x80:
        return np.asarray(offsets[positions + 1] - offsets[positions], dtype=np.int64)
    texts = take_text(data, offsets, positions).tolist()
    return np.asarray([len(text) for text in texts], dtype=np.int64)


def _distinct(
    columns: list[np.ndarray], widths: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of integer columns, in lexicographic order, and their counts.

    Each column holds values from 0 to below its width. Rows whose widths
    multiply below ``2**62`` are counted as one integer each, by sorting;
    others by :func:`~cytearc.utils.arrays.unique_rows`.
    """
    if math.prod(widths) >= 2**62:
        distinct, inverse = unique_rows(np.column_stack(columns))
        return distinct, np.bincount(inverse, minlength=len(distinct))
    combined = np.zeros(len(columns[0]), dtype=np.int64)
    for column, width in zip(columns, widths, strict=True):
        combined *= width
        combined += column
    combined.sort()
    ordered = combined
    starts = np.flatnonzero(np.diff(ordered, prepend=-1) != 0)
    counts = np.diff(starts, append=len(ordered))
    remaining = ordered[starts]
    digits = []
    for width in reversed(widths):
        digits.append(remaining % width)
        remaining = remaining // width
    return np.column_stack(digits[::-1]), counts


def count_chains(
    keys: pd.DataFrame,
    groups: np.ndarray,
    cells: np.ndarray,
    loci: tuple[np.ndarray, np.ndarray],
    values: Mapping[str, tuple[np.ndarray, np.ndarray]],
    count: str,
    *,
    clonotypes: np.ndarray | None = None,
    by_count: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Count chains by group, locus and values, with each count's share of its keys.

    Each chain's locus and values are codes, positions among names in the
    order of the table, and a chain with a missing value, a code of -1, is
    left out. ``count="clonotypes"`` counts each clonotype once per group,
    locus and values.

    Args:
        keys: One row per group: ``group``, its name, and its label in each
            key column.
        groups: Each chain's group, a position in ``keys``.
        cells: Each chain's cell.
        loci: Each chain's locus, as a code, and the loci.
        values: For each counted column, each chain's code and the names.
        count: ``"cells"`` or ``"clonotypes"``, the name of the count column.
        clonotypes: Each chain's clonotype code, to count clonotypes.
        by_count: Order the rows of each group and locus by count, most
            first, and then by value; otherwise by value.

    Returns:
        One row per group, locus and values with a chain, by group and
        locus: the columns of ``keys``, ``locus``, the values, ``count`` and
        ``fraction``, the count's share of the group's chains of that locus;
        and the cells with a counted chain.
    """
    columns = [*keys.columns, "locus", *values, count, "fraction"]
    codes = [
        np.asarray(chain_codes, dtype=np.int64) for chain_codes, _ in values.values()
    ]
    known = np.ones(len(groups), dtype=bool)
    for chain_codes in codes:
        known &= chain_codes >= 0
    n_cells = len(sorted_unique(np.asarray(cells, dtype=np.int64)[known]))
    if not known.any():
        return pd.DataFrame(columns=columns), n_cells
    rows = [
        np.asarray(groups, dtype=np.int64)[known],
        np.asarray(loci[0], dtype=np.int64)[known],
        *(chain_codes[known] for chain_codes in codes),
    ]
    widths = [len(keys), len(loci[1]), *(len(names) for _, names in values.values())]
    if count == "clonotypes":
        # Each clonotype once per group, locus and values.
        identities = np.asarray(clonotypes, dtype=np.int64)[known]
        with_clonotypes = _distinct(
            [*rows, identities], [*widths, int(identities.max()) + 1]
        )[0]
        rows = [with_clonotypes[:, at] for at in range(len(widths))]
    distinct, counts = _distinct(rows, widths)
    # The rows of each group and locus are adjacent among the distinct rows.
    changes = np.diff(distinct[:, :2], axis=0, prepend=-1) != 0
    starts = np.flatnonzero(changes.any(axis=1))
    totals = np.add.reduceat(counts, starts)
    fraction = counts / np.repeat(totals, np.diff(starts, append=len(distinct)))
    if by_count:
        order = np.lexsort(
            (*(distinct[:, at] for at in range(len(codes) + 1, 1, -1)), -counts)
            + (distinct[:, 1], distinct[:, 0])
        )
    else:
        order = np.arange(len(distinct))
    table = keys.take(distinct[order, 0]).reset_index(drop=True)
    table["locus"] = np.asarray(loci[1], dtype=object)[distinct[order, 1]]
    for at, (name, (_, names)) in enumerate(values.items()):
        table[name] = np.asarray(names)[distinct[order, 2 + at]]
    table[count] = counts[order]
    table["fraction"] = fraction[order]
    return table, n_cells


@dataclass(frozen=True, slots=True)
class GeneUsage:
    """The gene usage of groups of cells.

    Attributes:
        table: One row per group, locus and gene, or gene combination:
            ``group`` (its labels joined by commas, or ``"all cells"``), the
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
        table: One row per group, locus and length, by group, locus and
            length: ``group`` (its labels joined by commas, or ``"all
            cells"``), the group's key columns, ``locus``, ``length``
            (residues or nucleotides), the ``cells`` or ``clonotypes`` with
            junctions of that length, and their ``fraction`` of the group's
            chains of that locus.
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
