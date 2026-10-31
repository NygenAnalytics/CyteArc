"""Clonotypes: cells that share a receptor.

A clonotype groups the cells of one T or B cell clone, which share their
receptor because they descend from one cell. CyteArc defines clonotypes as
scirpy's ``define_clonotypes`` does (Sturm et al. 2020): two cells share a
clonotype when the junctions of their chains are identical, nucleotide by
nucleotide or residue by residue, under rules for which arms and which chains
must agree, and clonotypes are the connected groups that these links form.

- ``receptor_arms`` chooses the arms that must agree: ``"all"`` (both arms
  when both cells have both, else the one arm that each has), ``"any"`` (one
  arm is enough), ``"VJ"`` or ``"VDJ"`` (that arm alone).
- ``dual_ir`` chooses the chains of an arm that must agree: ``"any"`` (one
  of a cell's up to two chains matching one of the other's), ``"all"`` (both
  cells with the same number of chains, each chain matched), or
  ``"primary_only"`` (the first chains alone).
- ``same_v_gene`` and ``same_j_gene`` also require matched chains to share
  their V or J gene call.
- Cells of different groups, such as receptor types or donors, never share a
  clonotype.

Under ``"all"``, an arm counts for a cell when its primary chain has the
compared sequence: a cell with both arms is linked only to cells with both,
through both arms, and a cell with one arm only to cells with one, through
either arm. A cell without a compared sequence among the slots that take part
gets no clonotype.

Links are found through keys rather than through pairs of cells. Each chain
that takes part becomes an element, its sequence with the genes that must
match. Each distinct receptor configuration, the elements of a cell's slots
with its group, gets the keys that the rules derive from its elements: under
``"any"`` one key per element; under ``"all"`` the arm's sorted elements as one
key; with both arms required, every combination of a VJ and a VDJ key. Two
configurations are linked exactly when they share a key, and clonotypes are
the connected components of the graph of configurations and keys, so their
cost grows with the number of chains, not with the number of pairs of cells.

These functions work on encoded chains and never touch storage.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..storage.refs import ArtifactRef
from ..utils.arrays import unique_rows
from .chains import SLOTS

__all__ = [
    "DUAL_IR",
    "RECEPTOR_ARMS",
    "REPRESENTATIVE_FIELDS",
    "SEQUENCES",
    "ClonotypeAssignment",
    "Clonotypes",
    "assign_clonotypes",
    "clonotype_frames",
    "clonotype_labels",
    "representative_cells",
    "shared_text",
]

#: The values of ``receptor_arms``.
RECEPTOR_ARMS: tuple[str, ...] = ("all", "any", "VJ", "VDJ")
#: The values of ``dual_ir``.
DUAL_IR: tuple[str, ...] = ("any", "all", "primary_only")
#: The values of ``sequence``: the junction's nucleotides or its residues.
SEQUENCES: tuple[str, ...] = ("nt", "aa")

_VJ, _VDJ = 0, 1
# Key kinds keep the keys of different rules apart.
_ARM_KEY, _PAIR_KEY, _SINGLE_ARM_KEY = 0, 1, 2


def clonotype_labels(n_clonotypes: int) -> np.ndarray:
    """Return the labels of clonotypes by rank: ``clonotype_1`` is the largest."""
    return np.asarray(
        [f"clonotype_{rank}" for rank in range(1, int(n_clonotypes) + 1)],
        dtype=object,
    )


@dataclass(frozen=True, slots=True)
class ClonotypeAssignment:
    """The clonotype of each cell.

    Attributes:
        codes: Each cell's clonotype as its rank minus one, 0 for the largest
            clonotype, and -1 for a cell without one.
        sizes: The cells of each clonotype, largest first.
        configurations: Each cell's receptor configuration, -1 without one.
            Cells with equal configurations have equal compared chains and
            groups; configurations are numbered in the sorted order of their
            codes.
        configuration_clonotypes: The clonotype of each configuration.
    """

    codes: np.ndarray
    sizes: np.ndarray
    configurations: np.ndarray
    configuration_clonotypes: np.ndarray

    @property
    def n_clonotypes(self) -> int:
        return len(self.sizes)


def _element_ids(sequences: np.ndarray, genes: list[np.ndarray]) -> np.ndarray:
    """Number each slot's element, its sequence with its genes, -1 without one."""
    present = sequences >= 0
    ids = np.full(sequences.shape, -1, dtype=np.int64)
    if not present.any():
        return ids
    stacked = np.column_stack(
        [sequences[present]] + [gene[present] for gene in genes]
    ).astype(np.int64)
    ids[present] = unique_rows(stacked)[1]
    return ids


@dataclass(frozen=True, slots=True)
class _Configurations:
    """The distinct configurations of the cells that take part in clonotypes.

    Attributes:
        n_cells: The cells, those that take part and those that do not.
        cells: The cells that take part: with a group and an element.
        inverse: The configuration of each cell that takes part.
        configurations: Each cell's configuration, -1 for no part.
        rows: Each configuration's group and elements, one row each.
        groups: Each configuration's group.
        arms: Each configuration's elements of each arm that takes part.
        taking: The arms that take part.
    """

    n_cells: int
    cells: np.ndarray
    inverse: np.ndarray
    configurations: np.ndarray
    rows: np.ndarray
    groups: np.ndarray
    arms: dict[int, np.ndarray]
    taking: list[int]

    def unassigned(self) -> "ClonotypeAssignment":
        """The assignment of cells none of which takes part."""
        return ClonotypeAssignment(
            codes=np.full(self.n_cells, -1, dtype=np.int64),
            sizes=np.empty(0, dtype=np.int64),
            configurations=self.configurations,
            configuration_clonotypes=np.empty(0, dtype=np.int64),
        )


def _configurations(
    sequences: np.ndarray,
    groups: np.ndarray,
    v_genes: np.ndarray | None,
    j_genes: np.ndarray | None,
    receptor_arms: str,
    dual_ir: str,
) -> _Configurations:
    """Check the rules and number the cells' distinct configurations.

    A cell's configuration is its group and the elements, sequences with
    their gene calls, in the slots that the rules compare; cells of one
    configuration share a clonotype under any rule.

    Raises:
        ValueError: If an option is unknown or the arrays do not align.
    """
    if receptor_arms not in RECEPTOR_ARMS:
        raise ValueError(f"receptor_arms must be one of {', '.join(RECEPTOR_ARMS)}")
    if dual_ir not in DUAL_IR:
        raise ValueError(f"dual_ir must be one of {', '.join(DUAL_IR)}")
    sequences = np.asarray(sequences, dtype=np.int64)
    if sequences.ndim != 2 or sequences.shape[1] != 4:
        raise ValueError("sequences must hold four slots per cell")
    n_cells = sequences.shape[0]
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if len(groups) != n_cells:
        raise ValueError("groups must hold one group per cell")
    genes = [
        np.asarray(gene, dtype=np.int64)
        for gene in (v_genes, j_genes)
        if gene is not None
    ]
    if any(gene.shape != sequences.shape for gene in genes):
        raise ValueError("Gene codes must align with the sequences")
    chains = [0] if dual_ir == "primary_only" else [0, 1]
    taking = (
        [_VJ, _VDJ]
        if receptor_arms in ("all", "any")
        else [_VJ if receptor_arms == "VJ" else _VDJ]
    )
    columns = [2 * arm + chain for arm in taking for chain in chains]
    elements = _element_ids(sequences[:, columns], [gene[:, columns] for gene in genes])
    cells = np.flatnonzero((groups >= 0) & (elements >= 0).any(axis=1))
    configurations = np.full(n_cells, -1, dtype=np.int64)
    # Cells with the same group and elements in every slot that takes part
    # share a configuration, and so a clonotype.
    rows, inverse = unique_rows(np.column_stack([groups[cells], elements[cells]]))
    configurations[cells] = inverse
    width = len(chains)
    return _Configurations(
        n_cells=n_cells,
        cells=cells,
        inverse=inverse,
        configurations=configurations,
        rows=rows,
        groups=rows[:, 0],
        arms={
            arm: rows[:, 1 + position * width : 1 + (position + 1) * width]
            for position, arm in enumerate(taking)
        },
        taking=taking,
    )


def assign_clonotypes(
    sequences: np.ndarray,
    *,
    groups: np.ndarray,
    v_genes: np.ndarray | None = None,
    j_genes: np.ndarray | None = None,
    receptor_arms: str = "all",
    dual_ir: str = "any",
) -> ClonotypeAssignment:
    """Group cells into clonotypes by identical chains.

    Args:
        sequences: Each cell's compared sequence in its ``VJ_1``, ``VJ_2``,
            ``VDJ_1`` and ``VDJ_2`` slots, as codes that are equal exactly when
            the sequences are, -1 for an empty slot or a missing sequence; one
            row per cell.
        groups: Each cell's group, such as its receptor type and donor, as a
            code; -1 keeps a cell out, such as a cell without a receptor.
        v_genes: Each slot's V gene code, when matched chains must share it;
            -1 for a missing call, which matches another missing call.
        j_genes: Each slot's J gene code, alike.
        receptor_arms: The arms that must agree.
        dual_ir: The chains of an arm that must agree.

    Returns:
        The clonotype of each cell, ranked by size and then by configuration.

    Raises:
        ValueError: If an option is unknown or the arrays do not align.
    """
    found = _configurations(sequences, groups, v_genes, j_genes, receptor_arms, dual_ir)
    if not len(found.cells):
        return found.unassigned()
    config_groups, arm_elements, taking = found.groups, found.arms, found.taking

    owners: list[np.ndarray] = []
    keys: list[np.ndarray] = []

    def arm_keys(arm: int, subset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each key of an arm: its configuration and its key columns."""
        values = arm_elements[arm][subset]
        if dual_ir == "all":
            # The sorted elements of the arm form one key, so two
            # configurations share it exactly when they have the same number
            # of chains and every chain is matched. An arm without elements
            # has no key.
            keep = (values >= 0).any(axis=1)
            missing = np.iinfo(np.int64).max
            ordered = np.sort(np.where(values < 0, missing, values), axis=1)
            ordered[ordered == missing] = -1
            return subset[keep], ordered[keep]
        # One key per element that is present.
        owner = np.repeat(subset, values.shape[1])
        flat = values.reshape(-1)
        keep = flat >= 0
        return owner[keep], flat[keep, None]

    ids = np.arange(len(found.rows), dtype=np.int64)
    if receptor_arms == "all":
        primary = {arm: arm_elements[arm][:, 0] >= 0 for arm in taking}
        both = ids[primary[_VJ] & primary[_VDJ]]
        vj_owner, vj_key = arm_keys(_VJ, both)
        vdj_owner, vdj_key = arm_keys(_VDJ, both)
        # Every combination of a VJ and a VDJ key of one configuration.
        pairs = pd.merge(
            pd.DataFrame({"owner": vj_owner, "vj": np.arange(len(vj_owner))}),
            pd.DataFrame({"owner": vdj_owner, "vdj": np.arange(len(vdj_owner))}),
            on="owner",
        )
        owner = pairs["owner"].to_numpy(dtype=np.int64)
        _append(
            owners,
            keys,
            owner,
            [
                np.full((len(owner), 1), _PAIR_KEY),
                config_groups[owner, None],
                vj_key[pairs["vj"].to_numpy()],
                vdj_key[pairs["vdj"].to_numpy()],
            ],
        )
        # Configurations with one arm, or none whose primary chain has the
        # sequence: the keys of either arm, kept apart from configurations
        # with both arms and by their number of such arms.
        arm_count = primary[_VJ].astype(np.int64) + primary[_VDJ].astype(np.int64)
        single = ids[arm_count < 2]
        for arm in taking:
            owner, key = arm_keys(arm, single)
            _append(
                owners,
                keys,
                owner,
                [
                    np.full((len(owner), 1), _SINGLE_ARM_KEY),
                    arm_count[owner, None],
                    np.full((len(owner), 1), arm),
                    config_groups[owner, None],
                    key,
                ],
            )
    else:
        for arm in taking:
            owner, key = arm_keys(arm, ids)
            _append(
                owners,
                keys,
                owner,
                [
                    np.full((len(owner), 1), _ARM_KEY),
                    np.full((len(owner), 1), arm),
                    config_groups[owner, None],
                    key,
                ],
            )
    clusters = _components(len(found.rows), owners, keys)
    return _ranked(
        clusters, found.inverse, found.cells, found.n_cells, found.configurations
    )


def _append(
    owners: list[np.ndarray],
    keys: list[np.ndarray],
    owner: np.ndarray,
    columns: list[np.ndarray],
) -> None:
    """Add keys and their configurations, when there are any."""
    if len(owner):
        owners.append(np.asarray(owner, dtype=np.int64))
        keys.append(np.column_stack(columns).astype(np.int64))


def _components(
    n_configurations: int, owners: list[np.ndarray], keys: list[np.ndarray]
) -> np.ndarray:
    """Return the connected component of each configuration through shared keys.

    Every configuration has at least one key, since it has an element.
    """
    width = max(key.shape[1] for key in keys)
    # Keys of different shapes never collide: kinds differ in their first
    # column, and the padding value -2 occurs in no key.
    padded = np.concatenate(
        [
            np.pad(key, ((0, 0), (0, width - key.shape[1])), constant_values=-2)
            for key in keys
        ]
    )
    all_owners = np.concatenate(owners)
    _unique, key_ids = unique_rows(padded)
    n_keys = int(key_ids.max()) + 1
    size = n_configurations + n_keys
    graph = coo_matrix(
        (
            np.ones(len(all_owners), dtype=np.int8),
            (all_owners, n_configurations + key_ids),
        ),
        shape=(size, size),
    )
    _count, labels = connected_components(graph, directed=False)
    return np.asarray(labels[:n_configurations], dtype=np.int64)


def _ranked(
    clusters: np.ndarray,
    inverse: np.ndarray,
    cells: np.ndarray,
    n_cells: int,
    configurations: np.ndarray,
) -> ClonotypeAssignment:
    """Number clonotypes by size, largest first, ties by their first configuration."""
    n_clusters = int(clusters.max()) + 1
    sizes = np.bincount(clusters[inverse], minlength=n_clusters)
    # Configurations are numbered in the sorted order of their codes, so a
    # cluster's first configuration breaks ties independently of cell order.
    first = np.full(n_clusters, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, clusters, np.arange(len(clusters), dtype=np.int64))
    present = np.flatnonzero(sizes > 0)
    order = np.lexsort((first[present], -sizes[present]))
    rank = np.full(n_clusters, -1, dtype=np.int64)
    rank[present[order]] = np.arange(len(present), dtype=np.int64)
    codes = np.full(n_cells, -1, dtype=np.int64)
    codes[cells] = rank[clusters[inverse]]
    return ClonotypeAssignment(
        codes=codes,
        sizes=sizes[present[order]].astype(np.int64),
        configurations=configurations,
        configuration_clonotypes=rank[clusters],
    )


def representative_cells(
    assignment: ClonotypeAssignment,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a representative cell of each clonotype and its configurations.

    A clonotype's representative is the first cell, in cell order, of its most
    frequent configuration, ties going to the configuration first in sorted
    order, so the representative's chains are those of most of its cells.

    Returns:
        The representative cell of each clonotype, by rank, and the number of
        distinct configurations of each.
    """
    n_clonotypes = assignment.n_clonotypes
    cells = np.flatnonzero(assignment.codes >= 0)
    if not len(cells):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    configurations = assignment.configurations[cells]
    present, first, counts = np.unique(
        configurations, return_index=True, return_counts=True
    )
    owners = assignment.configuration_clonotypes[present]
    order = np.lexsort((present, -counts, owners))
    leading = np.r_[True, owners[order][1:] != owners[order][:-1]]
    representatives = np.empty(n_clonotypes, dtype=np.int64)
    representatives[owners[order][leading]] = cells[first[order][leading]]
    return representatives, np.bincount(owners, minlength=n_clonotypes).astype(np.int64)


#: The fields of each slot of a clonotype's representative chains, in column
#: order, of those that the receptors store.
REPRESENTATIVE_FIELDS = ("junction_aa", "junction", "v_call", "j_call")


@dataclass(frozen=True, slots=True)
class Clonotypes:
    """The clonotypes of a set of cells, read from a ``clonotypes`` artifact.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``: ``clonotype`` (its
            label, as a categorical of the labels in rank order, NA without
            one) and ``clone_size`` (the cells of its clonotype, 0 without
            one).
        clonotypes: One row per clonotype, largest first: ``clonotype``,
            ``n_cells``, ``fraction`` (its share of the cells with a
            clonotype), ``n_configurations`` (its distinct combinations of
            compared chains), ``receptor_type`` and the label of each
            ``within`` column of its cells, and the chains of its
            representative cell, the first cell of its most frequent
            configuration: for each slot, ``VJ_1`` to ``VDJ_2``, the slot's
            ``junction_aa``, ``junction``, ``v_call`` and ``j_call`` that the
            receptors store, such as ``VDJ_1_junction_aa``, as pandas text,
            missing where the slot is empty.
        receptors: The ``immune_receptors`` artifact that the clonotypes group.
        sequence: ``"nt"`` or ``"aa"``.
        receptor_arms: The arms that had to agree.
        dual_ir: The chains of an arm that had to agree.
        same_v_gene: Whether matched chains had to share their V gene call.
        same_j_gene: Whether matched chains had to share their J gene call.
        within: The cell metadata columns that kept clonotypes apart.
        within_receptor_type: Whether receptor types kept clonotypes apart.
    """

    cells: pd.DataFrame = field(repr=False)
    clonotypes: pd.DataFrame = field(repr=False)
    receptors: ArtifactRef
    sequence: str
    receptor_arms: str
    dual_ir: str
    same_v_gene: bool
    same_j_gene: bool
    within: tuple[str, ...]
    within_receptor_type: bool


def clonotype_frames(
    cell_ids: np.ndarray,
    *,
    codes: np.ndarray,
    clone_size: np.ndarray,
    clonotype_ids: np.ndarray,
    clonotype_sizes: np.ndarray,
    n_configurations: np.ndarray,
    receptor_type: np.ndarray,
    within: Mapping[str, np.ndarray],
    slots: np.ndarray,
    chains: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the tables of cells and clonotypes of stored clonotypes.

    Args:
        cell_ids: The ``ids`` of the cells.
        codes: Each cell's clonotype, a position in ``clonotype_ids``, -1
            without one.
        clone_size: Each cell's clone size.
        clonotype_ids: The labels of the clonotypes, by rank.
        clonotype_sizes: The cells of each clonotype.
        n_configurations: The configurations of each clonotype.
        receptor_type: The receptor type of each clonotype's representative.
        within: The label of each ``within`` column of each clonotype.
        slots: The contigs in the four slots of each clonotype's
            representative, -1 for an empty slot.
        chains: Each representative field's decoded values per contig.
    """
    # The labels are categories coded by the stored codes, ordered as the
    # clonotypes rank, so the table holds no text per cell; both tables share
    # one string per label.
    labels = np.asarray(clonotype_ids, dtype=object)
    cells = pd.DataFrame(
        {
            "clonotype": pd.Categorical.from_codes(
                np.asarray(codes, dtype=np.int64), categories=pd.Index(labels)
            ),
            "clone_size": np.asarray(clone_size, dtype=np.int64),
        },
        index=pd.Index(np.asarray(cell_ids), name="ids"),
        copy=False,
    )
    sizes = np.asarray(clonotype_sizes, dtype=np.int64)
    total = int(sizes.sum())
    table: dict[str, Any] = {
        "clonotype": labels,
        "n_cells": sizes,
        "fraction": sizes / total if total else np.zeros(len(sizes)),
        "n_configurations": np.asarray(n_configurations, dtype=np.int64),
        "receptor_type": shared_text(receptor_type),
    }
    for column, column_labels in within.items():
        table[column] = shared_text(column_labels)
    slots = np.asarray(slots, dtype=np.int64).reshape(len(sizes), 4)
    for position, slot in enumerate(SLOTS):
        contigs = slots[:, position]
        filled = contigs >= 0
        for name in REPRESENTATIVE_FIELDS:
            if name not in chains:
                continue
            taken = np.full(len(sizes), None, dtype=object)
            taken[filled] = np.asarray(chains[name], dtype=object)[contigs[filled]]
            table[f"{slot}_{name}"] = pd.array(taken, dtype="string")
    # The columns are not copied again.
    return cells, pd.DataFrame(table, copy=False)


def shared_text(values: Any) -> np.ndarray:
    """Return text as objects, with one string for each distinct value.

    Fixed-width text becomes a string per value when it is converted; coding
    it first makes the few distinct values, such as receptor types, the only
    strings. Missing values stay None.
    """
    array = np.asarray(values)
    if array.dtype.kind != "U":
        return np.asarray(array, dtype=object)
    distinct, inverse = np.unique(array, return_inverse=True)
    return np.asarray(distinct.tolist(), dtype=object)[inverse.reshape(array.shape)]
