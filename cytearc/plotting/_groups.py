"""Group the cells of a cell-aligned artifact for the figures that split by group."""

from typing import Any

import numpy as np
import pandas as pd

from ..metadata.selection import CellField, category_codes, resolve_grouping
from ..storage.artifacts import inspect_artifact
from ..storage.refs import ArtifactRef
from ..storage.selections import read_stored_selection_indices
from ..utils.arrays import sorted_unique

__all__ = ["cell_groups"]


def cell_groups(
    store: Any,
    artifact: ArtifactRef,
    *,
    group_by: str | None,
    groups: ArtifactRef | None,
) -> tuple[np.ndarray | None, list[str], str | None]:
    """Return the group of each cell of a cell-aligned artifact.

    ``group_by`` names a cell metadata column and ``groups`` a cell-label
    artifact, such as a clustering; at most one may be given. A label artifact
    may hold fewer cells than ``artifact``: the cells it lacks have no group,
    nor do cells with a missing or blank label. Cells are matched by their
    positions in the store.

    Returns:
        Each of the artifact's cells' group, as a position in the names, -1
        without one, or None without a grouping; the labels of the groups
        that some cell holds, in their natural order; and the grouping's name.

    Raises:
        TypeError: If ``group_by`` is not text or ``groups`` not an
            ArtifactRef.
        ValueError: If both are given.
    """
    if group_by is not None and groups is not None:
        raise ValueError("Pass group_by or groups, not both")
    if group_by is None and groups is None:
        return None, [], None
    selection = inspect_artifact(store.zw, artifact).input_ref("cell_selection")
    if group_by is not None:
        if not isinstance(group_by, str):
            raise TypeError("group_by must be the name of a cell metadata column")
        resolved = resolve_grouping(
            store.zw,
            store.cells,
            CellField(group_by, kind="categorical"),
            cell_selection=selection,
        )
        name = group_by
    else:
        if not isinstance(groups, ArtifactRef):
            raise TypeError("groups must be an ArtifactRef")
        resolved = resolve_grouping(store.zw, store.cells, groups)
        name = groups.kind
    cell_index = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    codes, names = category_codes(resolved.labels, resolved.missing_mask)
    at = pd.Index(np.asarray(resolved.cell_idx, dtype=np.int64)).get_indexer(
        np.asarray(cell_index, dtype=np.int64)
    )
    aligned = np.full(len(at), -1, dtype=np.int64)
    aligned[at >= 0] = codes[at[at >= 0]]
    # Only the labels of the artifact's cells name groups.
    held = sorted_unique(aligned[aligned >= 0])
    aligned[aligned >= 0] = np.searchsorted(held, aligned[aligned >= 0])
    return aligned, [names[code] for code in held.tolist()], name
