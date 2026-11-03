"""Group the cells of a cell-aligned artifact for the figures that split by group."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..clones.groups import group_names
from ..storage.refs import ArtifactRef

__all__ = ["cell_groups", "labels_by_group"]


def labels_by_group(
    cell_groups: Any, group_keys: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[str, np.ndarray]]:
    """Each cell's label in each column, from its group's row of ``group_keys``.

    ``cell_groups`` holds each cell's group name, missing for none, as a
    categorical or as text, and ``group_keys`` one row per group, indexed by
    its name, as a summary's record holds it. A cell without a group, or of a
    group that ``group_keys`` does not hold, has the label None. A
    categorical is matched through its codes, so the names are looked up
    once per category.
    """
    names = pd.Index(group_keys.index)
    values = pd.Series(cell_groups)
    if isinstance(values.dtype, pd.CategoricalDtype):
        # Each category's row, and -1 at the end for the code of no category.
        at = np.append(names.get_indexer(values.cat.categories), -1)
        rows = at[np.asarray(values.cat.codes)]
    else:
        rows = names.get_indexer(values.to_numpy(dtype=object))
    # The row -1 takes the None after the groups' labels.
    found = []
    for column in columns:
        labels = np.empty(len(names) + 1, dtype=object)
        labels[:-1] = group_keys[column].to_numpy(dtype=object)
        found.append((column, labels[rows]))
    return found


def cell_groups(
    store: Any,
    artifact: ArtifactRef,
    *,
    group_by: str | Sequence[str] | None,
    groups: ArtifactRef | None,
    kept_apart: Sequence[tuple[str, np.ndarray]] = (),
) -> tuple[np.ndarray | None, list[str], str | None]:
    """Return the group of each cell of a cell-aligned artifact, as the summaries group cells.

    The groups are the combinations of the labels of the ``group_by`` cell
    metadata columns, one or several, of the ``groups`` cell-label artifact,
    such as a clustering, and of ``kept_apart``, each with every artifact
    cell's label, such as the receptor types that clonotypes keep apart. The
    store groups the cells as ``ds.clones.diversity`` groups them, within the
    memory budget, and names each group by its labels joined by commas. A
    label artifact may hold fewer cells than ``artifact``: the cells it lacks
    have no group, nor do cells with a missing or blank label. Cells are
    matched by their positions in the store.

    Returns:
        Each of the artifact's cells' group, as a position in the names, -1
        without one, or None without a grouping; the names of the groups that
        some cell holds, in the natural order of their labels; and the
        grouping's name, its key columns joined by commas.

    Raises:
        TypeError: If the store does not group cells for figures, or an
            argument has the wrong type.
        ValueError: If a ``group_by`` column is missing, named twice or kept
            apart already, or two groups get one name.
        MemoryError: If grouping the cells does not fit the memory budget.
    """
    if group_by is None and groups is None and not kept_apart:
        return None, [], None
    method = getattr(store, "_figure_groups", None)
    if not callable(method):
        raise TypeError("store does not provide the groups of figures")
    grouped = method(
        artifact, group_by=group_by, groups=groups, kept_apart=tuple(kept_apart)
    )
    if not grouped.key_columns:
        return None, [], None
    return grouped.codes, group_names(grouped.keys), ", ".join(grouped.key_columns)
