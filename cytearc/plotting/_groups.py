"""Group the cells of a cell-aligned artifact for the figures that split by group."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..clones.groups import group_names
from ..storage.refs import ArtifactRef
from ..utils.arguments import quoted_value

__all__ = ["cell_groups", "labels_by_group", "record_groups", "refuse_named_columns"]


def record_groups(table: pd.DataFrame, values_from: str) -> pd.DataFrame:
    """Return one row per group of a summary's table: ``group`` and its key columns.

    A summary names each group in its table's ``group`` column and holds the
    group's label in each key column, the columns between ``group`` and
    ``values_from``, the first column that holds values. Figures read the
    groups of a summary through this function alone.

    Raises:
        ValueError: If the table has no ``group`` column, or no
            ``values_from`` column after it.
    """
    columns = [str(column) for column in table.columns]
    if "group" not in columns or values_from not in columns:
        raise ValueError(
            f"A summary's table must hold 'group' and {quoted_value(values_from)}; it holds "
            f"{columns}"
        )
    start, stop = columns.index("group") + 1, columns.index(values_from)
    if stop < start:
        raise ValueError(
            f"A summary's table must hold 'group' before {quoted_value(values_from)}"
        )
    keys = columns[start:stop]
    return table[["group", *keys]].drop_duplicates("group", ignore_index=True)


def labels_by_group(
    cell_groups: Any, groups: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[str, np.ndarray]]:
    """Each cell's label in each column, from its group's row of ``groups``.

    ``cell_groups`` holds each cell's group name, missing for none, and
    ``groups`` one row per group, as :func:`record_groups` returns them; a cell
    without a group has no label.
    """
    names = np.asarray(cell_groups, dtype=object)
    by_group = groups.set_index("group")
    return [
        (column, by_group[column].reindex(names).to_numpy(dtype=object))
        for column in columns
    ]


def refuse_named_columns(
    labels: Sequence[Any], columns: Sequence[str], *, what: str, table: str
) -> None:
    """Refuse labels named like a table's own columns, which the labels sit beside.

    Raises:
        ValueError: If a label's text names one of ``columns``.
    """
    clashing = sorted({str(label) for label in labels} & set(columns))
    if clashing:
        raise ValueError(
            f"A {what} is named {quoted_value(clashing[0])}, which the table of {table} uses "
            f"for a column of its own; its own columns are "
            f"{', '.join(map(quoted_value, columns))}, so rename that label"
        )


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
