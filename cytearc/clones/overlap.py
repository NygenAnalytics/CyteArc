"""The clones that groups of cells share, and the cells of chosen clones.

Two samples of one donor share the clones that both sampled, and two cell
types share the clones whose cells took both fates. Presence indices count
shared clones: their number, the Jaccard and Sørensen indices, the overlap
coefficient, and Ochiai's cosine of presence. Abundance indices weigh each
clone by its cells: the Morisita-Horn index (Horn 1966) and the cosine of the
clone sizes, both near 1 when the same expanded clones dominate both groups.
Every comparison multiplies the cells of each clone in each group, counted by
:func:`~cytearc.clones.groups.pair_counts`, as a sparse matrix of groups by
clones. These functions work on codes and never touch storage.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..storage.refs import ArtifactRef
from ..utils.arrays import sorted_unique

__all__ = [
    "OVERLAP_INDICES",
    "CloneOverlap",
    "CloneTracking",
    "group_clone_matrix",
    "overlap_extent",
    "overlap_pairs",
    "top_clones",
    "tracking_cells",
    "tracking_reach",
]

#: The indices of each pair of groups, in the order of the table's columns.
OVERLAP_INDICES = (
    "shared",
    "jaccard",
    "sorensen",
    "overlap",
    "cosine",
    "morisita_horn",
    "abundance_cosine",
)


def group_clone_matrix(
    pair_groups: np.ndarray,
    pair_clones: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
) -> sparse.csr_matrix:
    """The cells of each clone in each group, one row per group and one column per clone.

    Args:
        pair_groups: The group of each distinct pair of group and clone, as
            :func:`~cytearc.clones.groups.pair_counts` counts them.
        pair_clones: The clone code of each pair.
        pair_cells: The cells of the clone in the group.
        n_groups: The number of groups, the matrix's rows.

    Returns:
        A float64 matrix of ``n_groups`` rows and one column per clone code up
        to the largest, whose entries are whole numbers of cells.
    """
    n_clones = int(np.max(pair_clones)) + 1 if len(pair_clones) else 0
    return sparse.csr_matrix(
        (
            np.asarray(pair_cells, dtype=np.float64),
            (np.asarray(pair_groups), np.asarray(pair_clones)),
        ),
        shape=(n_groups, n_clones),
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide, NaN where the denominator is 0."""
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _strata_members(n_groups: int, strata: np.ndarray | None) -> list[np.ndarray]:
    """The groups of each stratum, ascending, or every group as one stratum."""
    if strata is None:
        return [np.arange(n_groups)]
    names, codes = np.unique(np.asarray(strata), return_inverse=True)
    codes = codes.reshape(-1)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(names) + 1))
    return [order[bounds[at] : bounds[at + 1]] for at in range(len(names))]


def overlap_extent(n_groups: int, strata: np.ndarray | None = None) -> tuple[int, int]:
    """The pairs that :func:`overlap_pairs` compares, and the groups of its largest stratum."""
    sizes = np.asarray([len(members) for members in _strata_members(n_groups, strata)])
    return int((sizes * (sizes - 1) // 2).sum()), int(sizes.max(initial=0))


def overlap_pairs(
    pair_groups: np.ndarray,
    pair_clones: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
    strata: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare every pair of groups by the clones they share.

    Args:
        pair_groups: The group of each distinct pair of group and clone.
        pair_clones: The clone code of each pair.
        pair_cells: The cells of the clone in the group, each above 0.
        n_groups: The number of groups.
        strata: Each group's stratum, such as its labels in the columns that
            the clones keep apart; only groups of one stratum are compared.
            None compares every pair.

    Returns:
        One row per pair ``first < second`` of group positions, by ``first``
        and then ``second``: ``first``, ``second``, the indices of
        ``OVERLAP_INDICES``, and ``shared_fraction_1`` and
        ``shared_fraction_2``, the share of the first and the second group's
        cells whose clone the other group holds.

        ``shared`` counts the clones of both groups; ``jaccard`` is
        ``shared / (S1 + S2 - shared)``, ``sorensen`` ``2 shared / (S1 +
        S2)``, ``overlap`` ``shared / min(S1, S2)`` and ``cosine`` ``shared /
        sqrt(S1 S2)``, with ``S1`` and ``S2`` the groups' clones.
        ``morisita_horn`` is ``2 sum(x y) / ((sum(x^2) / X^2 + sum(y^2) / Y^2)
        X Y)`` and ``abundance_cosine`` ``sum(x y) / sqrt(sum(x^2) sum(y^2))``,
        with ``x`` and ``y`` the clone sizes and ``X`` and ``Y`` the cells.
    """
    pair_groups = np.asarray(pair_groups, dtype=np.int64)
    pair_cells = np.asarray(pair_cells, dtype=np.float64)
    cells = group_clone_matrix(pair_groups, pair_clones, pair_cells, n_groups)
    presence = cells.copy()
    presence.data[:] = 1.0
    # Every product is a sum of whole numbers, exact in floating point.
    richness = np.bincount(pair_groups, minlength=n_groups).astype(np.float64)
    totals = np.bincount(pair_groups, weights=pair_cells, minlength=n_groups)
    squares = np.bincount(pair_groups, weights=pair_cells**2, minlength=n_groups)
    # Only groups of one stratum are compared, so the products are computed
    # for each stratum's groups, a block of those of all groups.
    blocks: list[dict[str, np.ndarray]] = []
    for members in _strata_members(n_groups, strata):
        upper, lower = np.triu_indices(len(members), k=1)
        block, block_presence = cells[members], presence[members]
        parts = {"first": members[upper], "second": members[lower]}
        for name, left, right in (
            ("shared", block_presence, block_presence),
            ("dot", block, block),
            ("reached", block, block_presence),
        ):
            dense = np.asarray((left @ right.T).todense())
            parts[name] = dense[upper, lower]
            if name == "reached":
                parts["reached_back"] = dense[lower, upper]
            del dense
        blocks.append(parts)

    def joined(name: str, dtype: Any = np.float64) -> np.ndarray:
        """One value of every pair, each block's taken out as it is joined."""
        return np.concatenate(
            [np.empty(0, dtype=dtype), *(b.pop(name) for b in blocks)]
        )

    first, second = joined("first", np.int64), joined("second", np.int64)
    # The pairs of all strata, by first and then second group.
    order = np.argsort(first * n_groups + second, kind="stable")
    first, second = first[order], second[order]

    def ordered(name: str) -> np.ndarray:
        return joined(name)[order]

    both = ordered("shared")
    s1, s2 = richness[first], richness[second]
    columns = {
        "first": first,
        "second": second,
        "shared": both.astype(np.int64),
        "jaccard": _ratio(both, s1 + s2 - both),
        "sorensen": _ratio(2 * both, s1 + s2),
        "overlap": _ratio(both, np.minimum(s1, s2)),
        "cosine": _ratio(both, np.sqrt(s1 * s2)),
    }
    del both, s1, s2
    dot = ordered("dot")
    n1, n2 = totals[first], totals[second]
    q1, q2 = squares[first], squares[second]
    columns["morisita_horn"] = _ratio(
        2 * dot, (_ratio(q1, n1**2) + _ratio(q2, n2**2)) * n1 * n2
    )
    columns["abundance_cosine"] = _ratio(dot, np.sqrt(q1 * q2))
    del dot, q1, q2
    columns["shared_fraction_1"] = _ratio(ordered("reached"), n1)
    columns["shared_fraction_2"] = _ratio(ordered("reached_back"), n2)
    return pd.DataFrame(columns)


def top_clones(
    pair_groups: np.ndarray,
    pair_clones: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
    top: int,
) -> np.ndarray:
    """The union of each group's ``top`` clones with the most cells.

    Ties go to the lower clone code, the larger clone overall, so the choice
    does not depend on the order of the cells.

    Returns:
        The chosen clone codes, ascending.
    """
    order = np.lexsort((pair_clones, -np.asarray(pair_cells), pair_groups))
    ordered_groups = np.asarray(pair_groups)[order]
    first = np.searchsorted(ordered_groups, np.arange(n_groups), side="left")
    rank = np.arange(len(order)) - first[ordered_groups]
    return sorted_unique(np.asarray(pair_clones, dtype=np.int64)[order][rank < top])


def _chosen_pairs(
    pair_clones: np.ndarray, chosen: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of chosen clones, and the position of each one's clone."""
    pair_clones = np.asarray(pair_clones, dtype=np.int64)
    at = np.searchsorted(chosen, pair_clones)
    found = at < len(chosen)
    found[found] = chosen[at[found]] == pair_clones[found]
    return np.flatnonzero(found), at[found]


def tracking_reach(
    pair_groups: np.ndarray,
    pair_clones: np.ndarray,
    chosen: np.ndarray,
    strata: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """The strata of groups through which each chosen clone is followed.

    A clone is followed through the groups of every stratum in which it has
    cells, and a clone without cells through the groups of its ``fallback``
    stratum.

    Args:
        pair_groups: The group of each distinct pair of group and clone.
        pair_clones: The clone code of each pair.
        chosen: The clone codes to follow, ascending.
        strata: Each group's stratum, from 0, such as a code of its labels in
            the columns that the clones keep apart.
        fallback: The stratum of each chosen clone without cells, -1 for
            none.

    Returns:
        Whether each chosen clone, a row, is followed through the groups of
        each stratum, a column.
    """
    strata = np.asarray(strata, dtype=np.int64)
    chosen = np.asarray(chosen, dtype=np.int64)
    fallback = np.asarray(fallback, dtype=np.int64)
    reach = np.zeros((len(chosen), int(strata.max(initial=-1)) + 1), dtype=bool)
    pairs, positions = _chosen_pairs(pair_clones, chosen)
    reach[positions, strata[np.asarray(pair_groups)[pairs]]] = True
    lone = np.flatnonzero(~reach.any(axis=1) & (fallback >= 0))
    reach[lone, fallback[lone]] = True
    return reach


def tracking_cells(
    pair_groups: np.ndarray,
    pair_clones: np.ndarray,
    pair_cells: np.ndarray,
    chosen: np.ndarray,
    strata: np.ndarray,
    reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells of each chosen clone in each group that follows it.

    Args:
        pair_groups: The group of each distinct pair of group and clone.
        pair_clones: The clone code of each pair.
        pair_cells: The cells of the clone in the group.
        chosen: The clone codes to follow, ascending.
        strata: Each group's stratum, from 0.
        reach: The strata of groups through which each chosen clone is
            followed, as :func:`tracking_reach` gives them.

    Returns:
        For each row, by chosen clone and then group, both ascending: the
        position of its clone in ``chosen``, its group, and the clone's cells
        there, 0 included.
    """
    chosen = np.asarray(chosen, dtype=np.int64)
    strata = np.asarray(strata, dtype=np.int64)
    n_groups = len(strata)
    positions, groups = np.nonzero(reach[:, strata])
    pairs, pair_positions = _chosen_pairs(pair_clones, chosen)
    keys = pair_positions * n_groups + np.asarray(pair_groups, dtype=np.int64)[pairs]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    wanted = positions * n_groups + groups
    at = np.minimum(np.searchsorted(keys, wanted), max(len(keys) - 1, 0))
    found = (keys[at] == wanted) if len(keys) else np.zeros(len(wanted), dtype=bool)
    cells = np.zeros(len(wanted), dtype=np.int64)
    cells[found] = np.asarray(pair_cells, dtype=np.int64)[pairs[order[at[found]]]]
    return positions.astype(np.int64), groups.astype(np.int64), cells


@dataclass(frozen=True, slots=True)
class CloneOverlap:
    """The clones that groups of cells share.

    The groups are the combinations of the ``group_by`` columns' labels and
    the ``groups`` artifact's labels; each table names a group by its labels
    joined by commas.

    Attributes:
        group_sizes: One row per group: ``group``, the key columns, ``cells``
            and ``clones``.
        pairs: One row per pair of groups: ``group_1`` and ``group_2``, the
            indices ``shared``, ``jaccard``, ``sorensen``, ``overlap``,
            ``cosine``, ``morisita_horn`` and ``abundance_cosine``, and
            ``shared_fraction_1`` and ``shared_fraction_2``, the share of each
            group's cells whose clone the other holds.
        clones: The clone artifact, such as a ``clones`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
    """

    group_sizes: pd.DataFrame = field(repr=False)
    pairs: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None


@dataclass(frozen=True, slots=True)
class CloneTracking:
    """The cells of chosen clones in each group of cells.

    Attributes:
        cells: One row per chosen clone and group, zeros included: ``clone``,
            ``group`` (its labels joined by commas), the key columns,
            ``cells`` and ``fraction``, the share of the group's cells with a
            clone.
        clones: The clone artifact, such as a ``clones`` artifact.
        tracked: The chosen clone labels, largest first.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        cell_selection: The cells kept, or None for every cell.
    """

    cells: pd.DataFrame = field(repr=False)
    clones: ArtifactRef
    tracked: tuple[str, ...]
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    cell_selection: ArtifactRef | None
