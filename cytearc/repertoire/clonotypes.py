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
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
]

type ReceptorArms = Literal["all", "any", "VJ", "VDJ"]
type DualIr = Literal["any", "all", "primary_only"]
type SequenceKind = Literal["nt", "aa"]

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
    return _elements(sequences, genes)[0]


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
    eligible = (groups >= 0) & (elements >= 0).any(axis=1)
    cells = np.flatnonzero(eligible)
    configurations = np.full(n_cells, -1, dtype=np.int64)
    if not len(cells):
        return ClonotypeAssignment(
            codes=np.full(n_cells, -1, dtype=np.int64),
            sizes=np.empty(0, dtype=np.int64),
            configurations=configurations,
            configuration_clonotypes=np.empty(0, dtype=np.int64),
        )

    # Cells with the same group and elements in every slot that takes part
    # share a configuration, and so a clonotype.
    unique_rows, inverse = np.unique(
        np.column_stack([groups[cells], elements[cells]]), axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    configurations[cells] = inverse
    config_groups = unique_rows[:, 0]
    width = len(chains)
    arm_elements = {
        arm: unique_rows[:, 1 + position * width : 1 + (position + 1) * width]
        for position, arm in enumerate(taking)
    }

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

    ids = np.arange(len(unique_rows), dtype=np.int64)
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
    clusters = _components(len(unique_rows), owners, keys)
    return _ranked(clusters, inverse, cells, n_cells, configurations)


def assign_clonotype_clusters(
    sequences: np.ndarray,
    texts: Any,
    *,
    groups: np.ndarray,
    metric: str,
    cutoff: int,
    v_genes: np.ndarray | None = None,
    j_genes: np.ndarray | None = None,
    receptor_arms: str = "all",
    dual_ir: str = "any",
) -> ClonotypeAssignment:
    """Group cells into clonotype clusters by similar chains.

    The rules are those of :func:`assign_clonotypes`, with two chains matching
    when their sequences are within ``cutoff`` of each other by ``metric``,
    and share their gene calls when those are given, rather than identical:
    as in scirpy's ``define_clonotype_clusters``, two cells are linked when
    the arms that ``receptor_arms`` requires match, an arm matching through
    any one pair of chains (``"any"``), its primary chains
    (``"primary_only"``), or every chain in some pairing of two arms of the
    same number of chains (``"all"``). Clusters are the connected groups of
    linked cells.

    Args:
        sequences: Each cell's sequence code in its four slots, -1 for none.
        texts: The sequence of each code.
        groups: Each cell's group; -1 keeps a cell out.
        metric: ``"hamming"``, ``"normalized_hamming"`` or ``"levenshtein"``.
        cutoff: The largest distance of matching chains.
        v_genes: Each slot's V gene code, when matched chains must share it.
        j_genes: Each slot's J gene code, alike.
        receptor_arms: The arms that must agree.
        dual_ir: The chains of an arm that must agree.

    Returns:
        The clonotype cluster of each cell, ranked as clonotypes are.
    """
    from .distances import DISTANCE_METRICS, neighbor_pairs

    if receptor_arms not in RECEPTOR_ARMS:
        raise ValueError(f"receptor_arms must be one of {', '.join(RECEPTOR_ARMS)}")
    if dual_ir not in DUAL_IR:
        raise ValueError(f"dual_ir must be one of {', '.join(DUAL_IR)}")
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"metric must be one of {', '.join(DISTANCE_METRICS)}")
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
    names = np.asarray(texts, dtype=object)

    chains = [0] if dual_ir == "primary_only" else [0, 1]
    taking = (
        [_VJ, _VDJ]
        if receptor_arms in ("all", "any")
        else [_VJ if receptor_arms == "VJ" else _VDJ]
    )
    columns = [2 * arm + chain for arm in taking for chain in chains]
    elements, element_rows = _elements(
        sequences[:, columns], [gene[:, columns] for gene in genes]
    )
    eligible = (groups >= 0) & (elements >= 0).any(axis=1)
    cells = np.flatnonzero(eligible)
    configurations = np.full(n_cells, -1, dtype=np.int64)
    if not len(cells):
        return ClonotypeAssignment(
            codes=np.full(n_cells, -1, dtype=np.int64),
            sizes=np.empty(0, dtype=np.int64),
            configurations=configurations,
            configuration_clonotypes=np.empty(0, dtype=np.int64),
        )
    unique_rows, inverse = np.unique(
        np.column_stack([groups[cells], elements[cells]]), axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    configurations[cells] = inverse
    n_configurations = len(unique_rows)
    config_groups = unique_rows[:, 0]
    width = len(chains)
    arm_elements = {
        arm: unique_rows[:, 1 + position * width : 1 + (position + 1) * width]
        for position, arm in enumerate(taking)
    }

    # Elements within the cutoff, with matching genes when asked: their
    # sequences are compared within blocks of equal gene calls.
    n_elements = len(element_rows)
    blocks = (
        np.unique(element_rows[:, 1:], axis=0, return_inverse=True)[1].reshape(-1)
        if element_rows.shape[1] > 1
        else None
    )
    low, high = neighbor_pairs(
        names[element_rows[:, 0]], metric, int(cutoff), blocks=blocks
    )
    near = coo_matrix(
        (
            np.ones(2 * len(low) + n_elements, dtype=np.int8),
            (
                np.concatenate([low, high, np.arange(n_elements)]),
                np.concatenate([high, low, np.arange(n_elements)]),
            ),
        ),
        shape=(n_elements, n_elements),
    ).tocsr()
    near_codes = np.unique(
        np.concatenate(
            [
                low * n_elements + high,
                high * n_elements + low,
                np.arange(n_elements) * (n_elements + 1),
            ]
        )
    )

    def is_near(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Whether element pairs are within the cutoff; -1 is never near."""
        valid = (first >= 0) & (second >= 0)
        encoded = np.where(valid, first * n_elements + second, -1)
        found = np.searchsorted(near_codes, encoded)
        found = np.minimum(found, len(near_codes) - 1)
        return np.asarray(valid & (near_codes[found] == encoded), dtype=bool)

    def arm_links(arm: int) -> tuple[np.ndarray, np.ndarray]:
        """The pairs of configurations, first below second, whose arm matches."""
        values = arm_elements[arm]
        owner = np.repeat(np.arange(n_configurations), values.shape[1])
        flat = values.reshape(-1)
        keep = flat >= 0
        incidence = coo_matrix(
            (np.ones(int(keep.sum()), dtype=np.int8), (owner[keep], flat[keep])),
            shape=(n_configurations, n_elements),
        ).tocsr()
        linked = sparse.triu(
            (incidence @ near @ incidence.T).tocoo(), k=1, format="coo"
        )
        first, second = linked.row.astype(np.int64), linked.col.astype(np.int64)
        same = config_groups[first] == config_groups[second]
        first, second = first[same], second[same]
        if dual_ir != "all":
            return first, second
        # Every chain matched: the same number of chains, paired in order or
        # crossed.
        ordered = np.sort(np.where(values < 0, np.iinfo(np.int64).max, values), axis=1)
        ordered[ordered == np.iinfo(np.int64).max] = -1
        counts = (values >= 0).sum(axis=1)
        a, b = ordered[first], ordered[second]
        straight = is_near(a[:, 0], b[:, 0]) & (
            (counts[first] == 1) | is_near(a[:, 1], b[:, 1])
        )
        crossed = (
            (counts[first] == 2) & is_near(a[:, 0], b[:, 1]) & is_near(a[:, 1], b[:, 0])
        )
        keep_pair = (counts[first] == counts[second]) & (straight | crossed)
        return first[keep_pair], second[keep_pair]

    links = {arm: arm_links(arm) for arm in taking}

    def as_set(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return np.asarray(pair[0] * n_configurations + pair[1], dtype=np.int64)

    if receptor_arms == "all":
        primary = {arm: arm_elements[arm][:, 0] >= 0 for arm in taking}
        arm_count = primary[_VJ].astype(np.int64) + primary[_VDJ].astype(np.int64)
        both = np.intersect1d(as_set(links[_VJ]), as_set(links[_VDJ]))
        either = np.union1d(as_set(links[_VJ]), as_set(links[_VDJ]))
        first, second = both // n_configurations, both % n_configurations
        paired = (arm_count[first] == 2) & (arm_count[second] == 2)
        one_first, one_second = either // n_configurations, either % n_configurations
        single = (arm_count[one_first] < 2) & (
            arm_count[one_first] == arm_count[one_second]
        )
        first = np.concatenate([first[paired], one_first[single]])
        second = np.concatenate([second[paired], one_second[single]])
    else:
        merged = np.unique(np.concatenate([as_set(links[arm]) for arm in taking]))
        first, second = merged // n_configurations, merged % n_configurations
    graph = coo_matrix(
        (np.ones(len(first), dtype=np.int8), (first, second)),
        shape=(n_configurations, n_configurations),
    )
    _n, clusters = connected_components(graph, directed=False)
    return _ranked(clusters.astype(np.int64), inverse, cells, n_cells, configurations)


def _elements(
    sequences: np.ndarray, genes: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Number each slot's element, -1 without one, and return each element's codes."""
    present = sequences >= 0
    ids = np.full(sequences.shape, -1, dtype=np.int64)
    if not present.any():
        return ids, np.empty((0, 1 + len(genes)), dtype=np.int64)
    stacked = np.column_stack(
        [sequences[present]] + [gene[present] for gene in genes]
    ).astype(np.int64)
    rows, inverse = np.unique(stacked, axis=0, return_inverse=True)
    ids[present] = inverse.reshape(-1)
    return ids, rows


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
    _unique, key_ids = np.unique(padded, axis=0, return_inverse=True)
    key_ids = key_ids.reshape(-1)
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
            label, NA without one) and ``clone_size`` (the cells of its
            clonotype, 0 without one).
        clonotypes: One row per clonotype, largest first: ``clonotype``,
            ``n_cells``, ``fraction`` (its share of the cells with a
            clonotype), ``n_configurations`` (its distinct combinations of
            compared chains), ``receptor_type`` and the label of each
            ``within`` column of its cells, and the chains of its
            representative cell, the first cell of its most frequent
            configuration: for each slot, ``VJ_1`` to ``VDJ_2``, the slot's
            ``junction_aa``, ``junction``, ``v_call`` and ``j_call`` that the
            receptors store, such as ``VDJ_1_junction_aa``.
        receptors: The ``immune_receptors`` artifact that the clonotypes group.
        sequence: ``"nt"`` or ``"aa"``.
        receptor_arms: The arms that had to agree.
        dual_ir: The chains of an arm that had to agree.
        same_v_gene: Whether matched chains had to share their V gene call.
        same_j_gene: Whether matched chains had to share their J gene call.
        within: The cell metadata columns that kept clonotypes apart.
        within_receptor_type: Whether receptor types kept clonotypes apart.
        metric: ``"identity"`` or the distance that links similar chains.
        cutoff: The largest distance of linked chains, 0 for identity.
    """

    cells: pd.DataFrame = field(repr=False)
    clonotypes: pd.DataFrame = field(repr=False)
    receptors: Any
    sequence: str
    receptor_arms: str
    dual_ir: str
    same_v_gene: bool
    same_j_gene: bool
    within: tuple[str, ...]
    within_receptor_type: bool
    metric: str
    cutoff: int


def clonotype_frames(
    cell_ids: np.ndarray,
    *,
    values: np.ndarray,
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
        values: Each cell's clonotype label, empty without one.
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
    labels = np.asarray(values, dtype=object).copy()
    labels[labels == ""] = None
    cells = pd.DataFrame(
        {
            "clonotype": pd.array(labels),
            "clone_size": np.asarray(clone_size, dtype=np.int64),
        },
        index=pd.Index(np.asarray(cell_ids), name="ids"),
    )
    sizes = np.asarray(clonotype_sizes, dtype=np.int64)
    total = int(sizes.sum())
    table: dict[str, Any] = {
        "clonotype": np.asarray(clonotype_ids, dtype=object),
        "n_cells": sizes,
        "fraction": sizes / total if total else np.zeros(len(sizes)),
        "n_configurations": np.asarray(n_configurations, dtype=np.int64),
        "receptor_type": np.asarray(receptor_type, dtype=object),
    }
    for column, column_labels in within.items():
        table[column] = np.asarray(column_labels, dtype=object)
    slots = np.asarray(slots, dtype=np.int64).reshape(len(sizes), 4)
    for position, slot in enumerate(SLOTS):
        contigs = slots[:, position]
        filled = contigs >= 0
        for name in REPRESENTATIVE_FIELDS:
            if name not in chains:
                continue
            taken = np.full(len(sizes), None, dtype=object)
            taken[filled] = np.asarray(chains[name], dtype=object)[contigs[filled]]
            table[f"{slot}_{name}"] = pd.array(taken)
    return cells, pd.DataFrame(table)
