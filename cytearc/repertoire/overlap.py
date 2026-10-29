"""The clonotypes that groups of cells share, and the cells of chosen clonotypes.

Two samples of one donor share the clonotypes of clones that both sampled,
and two cell types share the clonotypes of clones whose cells took both fates.
Presence indices count shared clonotypes: their number, the Jaccard and
Sørensen indices, the overlap coefficient, and Ochiai's cosine of presence.
Abundance indices weigh each clonotype by its cells: the Morisita-Horn index
(Horn 1966) and the cosine of the clone sizes, both near 1 when the same
expanded clones dominate both groups. These functions work on codes and never
touch storage.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from ..storage.refs import ArtifactRef

__all__ = [
    "OVERLAP_INDICES",
    "ClonotypeOverlap",
    "ClonotypeTracking",
    "group_names",
    "overlap_pairs",
    "top_clonotypes",
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


def _matrix(
    pair_groups: np.ndarray,
    pair_clonotypes: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
) -> sparse.csr_matrix:
    """The cells of each clonotype, one row per group and one column per clonotype."""
    n_clonotypes = int(pair_clonotypes.max()) + 1 if len(pair_clonotypes) else 0
    return sparse.csr_matrix(
        (
            np.asarray(pair_cells, dtype=np.float64),
            (np.asarray(pair_groups), np.asarray(pair_clonotypes)),
        ),
        shape=(n_groups, n_clonotypes),
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide, NaN where the denominator is 0."""
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def overlap_pairs(
    pair_groups: np.ndarray,
    pair_clonotypes: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
    strata: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare every pair of groups by the clonotypes they share.

    Args:
        pair_groups: The group of each distinct pair of group and clonotype.
        pair_clonotypes: The clonotype code of each pair.
        pair_cells: The cells of the clonotype in the group, each above 0.
        n_groups: The number of groups.
        strata: Each group's stratum, such as its receptor type; only groups
            of one stratum are compared. None compares every pair.

    Returns:
        One row per pair ``first < second`` of group positions: ``first``,
        ``second``, the indices of ``OVERLAP_INDICES``, and
        ``shared_fraction_1`` and ``shared_fraction_2``, the share of the
        first and the second group's cells whose clonotype the other group
        holds.

        ``shared`` counts the clonotypes of both groups; ``jaccard`` is
        ``shared / (S1 + S2 - shared)``, ``sorensen`` ``2 shared / (S1 +
        S2)``, ``overlap`` ``shared / min(S1, S2)`` and ``cosine`` ``shared /
        sqrt(S1 S2)``, with ``S1`` and ``S2`` the groups' clonotypes.
        ``morisita_horn`` is ``2 sum(x y) / ((sum(x^2) / X^2 + sum(y^2) / Y^2)
        X Y)`` and ``abundance_cosine`` ``sum(x y) / sqrt(sum(x^2) sum(y^2))``,
        with ``x`` and ``y`` the clone sizes and ``X`` and ``Y`` the cells.
    """
    cells = _matrix(pair_groups, pair_clonotypes, pair_cells, n_groups)
    presence = cells.copy()
    presence.data[:] = 1.0
    shared = np.asarray((presence @ presence.T).todense())
    products = np.asarray((cells @ cells.T).todense())
    reached = np.asarray((cells @ presence.T).todense())
    richness = np.diag(shared)
    totals = np.asarray(cells.sum(axis=1)).reshape(-1)
    squares = np.diag(products)
    first, second = np.triu_indices(n_groups, k=1)
    if strata is not None:
        same = np.asarray(strata)[first] == np.asarray(strata)[second]
        first, second = first[same], second[same]
    both = shared[first, second]
    s1, s2 = richness[first], richness[second]
    n1, n2 = totals[first], totals[second]
    q1, q2 = squares[first], squares[second]
    dot = products[first, second]
    table = pd.DataFrame(
        {
            "first": first.astype(np.int64),
            "second": second.astype(np.int64),
            "shared": both.astype(np.int64),
            "jaccard": _ratio(both, s1 + s2 - both),
            "sorensen": _ratio(2 * both, s1 + s2),
            "overlap": _ratio(both, np.minimum(s1, s2)),
            "cosine": _ratio(both, np.sqrt(s1 * s2)),
            "morisita_horn": _ratio(
                2 * dot, (_ratio(q1, n1**2) + _ratio(q2, n2**2)) * n1 * n2
            ),
            "abundance_cosine": _ratio(dot, np.sqrt(q1 * q2)),
            "shared_fraction_1": _ratio(reached[first, second], n1),
            "shared_fraction_2": _ratio(reached[second, first], n2),
        }
    )
    return table


def top_clonotypes(
    pair_groups: np.ndarray,
    pair_clonotypes: np.ndarray,
    pair_cells: np.ndarray,
    n_groups: int,
    top: int,
) -> np.ndarray:
    """The union of each group's ``top`` clonotypes with the most cells.

    Ties go to the lower clonotype code, the larger clonotype overall, so the
    choice does not depend on the order of the cells.

    Returns:
        The chosen clonotype codes, ascending.
    """
    order = np.lexsort((pair_clonotypes, -np.asarray(pair_cells), pair_groups))
    ordered_groups = np.asarray(pair_groups)[order]
    first = np.searchsorted(ordered_groups, np.arange(n_groups), side="left")
    rank = np.arange(len(order)) - first[ordered_groups]
    return np.unique(np.asarray(pair_clonotypes)[order][rank < top])


@dataclass(frozen=True, slots=True)
class ClonotypeOverlap:
    """The clonotypes that groups of cells share.

    The groups are the combinations of the ``group_by`` columns' labels and
    the ``groups`` artifact's labels; with ``within_receptor_type``, the
    receptor type is part of each group and only groups of one receptor type
    are compared.

    Attributes:
        group_sizes: One row per group: ``group`` (its labels joined by
            commas), the key columns, ``cells`` and ``clonotypes``.
        pairs: One row per pair of groups: ``group_1`` and ``group_2``, the
            indices ``shared``, ``jaccard``, ``sorensen``, ``overlap``,
            ``cosine``, ``morisita_horn`` and ``abundance_cosine``, and
            ``shared_fraction_1`` and ``shared_fraction_2``, the share of each
            group's cells whose clonotype the other holds.
        clonotypes: The ``clonotypes`` artifact.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        within_receptor_type: Whether the receptor type is part of each group.
        cell_selection: The cells kept, or None for every cell.
    """

    group_sizes: pd.DataFrame = field(repr=False)
    pairs: pd.DataFrame = field(repr=False)
    clonotypes: ArtifactRef
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    within_receptor_type: bool
    cell_selection: ArtifactRef | None


@dataclass(frozen=True, slots=True)
class ClonotypeTracking:
    """The cells of chosen clonotypes in each group of cells.

    Attributes:
        cells: One row per chosen clonotype and group, zeros included:
            ``clonotype``, ``group`` (its labels joined by commas), the key
            columns, ``cells`` and ``fraction``, the share of the group's
            cells with a clonotype.
        clonotypes: The ``clonotypes`` artifact.
        tracked: The chosen clonotype labels, largest first.
        group_by: The cell metadata columns of the groups.
        groups: The label artifact of the groups, or None.
        within_receptor_type: Whether the receptor type is part of each group.
        cell_selection: The cells kept, or None for every cell.
    """

    cells: pd.DataFrame = field(repr=False)
    clonotypes: ArtifactRef
    tracked: tuple[str, ...]
    group_by: tuple[str, ...]
    groups: ArtifactRef | None
    within_receptor_type: bool
    cell_selection: ArtifactRef | None


def group_names(group_keys: Sequence[dict[str, str]]) -> list[str]:
    """Name each group by its labels joined by commas, or ``"all cells"``."""
    return [", ".join(key.values()) if key else "all cells" for key in group_keys]
