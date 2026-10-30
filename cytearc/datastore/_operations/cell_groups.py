"""Group the cells of a cell-aligned artifact by their labels.

Summaries such as clonal expansion and diversity are measured within groups
of cells: the combinations of the labels of cell metadata columns, such as
the sample, of a cell-label artifact, such as a clustering, and of labels that
the measured artifact keeps apart. ``cell_groups`` codes each column's labels
once, in the natural order of their texts, and numbers the combinations that
the cells hold, so every summary groups and names its groups alike.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ...clones.groups import group_names
from ...metadata.rows import read_metadata_missing_rows, read_metadata_rows
from ...metadata.selection import category_codes, resolve_grouping
from ...metadata.table import MetaData
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...utils.arrays import unique_rows

__all__ = [
    "GROUPING_BYTES_PER_CELL",
    "GROUPING_BYTES_PER_COLUMN",
    "CellGroups",
    "cell_groups",
    "group_columns",
]

#: Grouping cells holds, at its peak, about these bytes per cell and per
#: column of labels: the cells' positions, codes and groups, and each column's
#: labels while they are coded; 84 bytes per cell without labels and 206 with
#: four columns, as measured on 1,000,000 cells.
GROUPING_BYTES_PER_CELL = 100
GROUPING_BYTES_PER_COLUMN = 70


def group_columns(
    value: Any, name: str = "group_by", reserved: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Return the cell metadata columns of an argument as a tuple of names.

    A column may not share the name of a column of the result, in
    ``reserved``, which it would overwrite.

    Raises:
        TypeError: If ``value`` is not a column name, a sequence of names, or
            None.
        ValueError: If it names a column twice, or a column of the result.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        names: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        names = list(value)
    else:
        raise TypeError(f"{name} must be a column name, a sequence of names, or None")
    if not all(isinstance(column, str) and column for column in names):
        raise TypeError(f"{name} must be a column name, a sequence of names, or None")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} names a column twice")
    clashes = [column for column in names if column in reserved]
    if clashes:
        raise ValueError(
            f"{name} names {clashes[0]!r}, which is also a column of the result; "
            "copy the cell metadata column under another name"
        )
    return tuple(names)


@dataclass(frozen=True, slots=True)
class CellGroups:
    """Each cell's group and the labels that name the groups.

    Attributes:
        key_columns: The columns of the groups' labels: the ``group_by``
            columns, the kind of the ``groups`` artifact, then the columns
            that the artifact keeps apart.
        keys: Each group's label in each key column, in the natural order of
            the labels. Only the combinations that some cell holds are
            groups.
        codes: Each cell's group, a position in ``keys``, -1 for none.
    """

    key_columns: tuple[str, ...]
    keys: list[dict[str, str]]
    codes: np.ndarray


def _selection_indices(zw: zarr.Group, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)


def cell_groups(
    zw: zarr.Group,
    cells: MetaData,
    artifact_cells: ArtifactRef,
    rows: np.ndarray,
    *,
    group_by: str | Sequence[str] | None,
    groups: ArtifactRef | None,
    cell_selection: ArtifactRef | None,
    memory_bytes: int,
    kept_apart: Sequence[tuple[str, np.ndarray]] = (),
    reserved: frozenset[str] = frozenset(),
) -> CellGroups:
    """Group the cells of a cell-aligned artifact by their labels.

    The groups are the combinations of the labels of the ``group_by``
    columns, of the ``groups`` artifact and of ``kept_apart``. A cell outside
    ``rows`` or ``cell_selection``, or without a label, is in no group. A
    label artifact may hold fewer cells than the artifact: the cells it lacks
    have no group.

    Args:
        zw: The store's root group.
        cells: The store's cell metadata.
        artifact_cells: The cell selection of the artifact's cells.
        rows: The artifact's cells that may join a group, as a mask aligned
            to its cells.
        group_by: Cell metadata columns.
        groups: A cell-label artifact.
        cell_selection: Only these cells, or None.
        memory_bytes: The memory budget.
        kept_apart: Key columns whose labels the artifact keeps apart, each
            with every artifact cell's label, such as a receptor type.
        reserved: The columns of the result, which no key column may name.

    Returns:
        The groups' key columns and labels, and each cell's group.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a ``group_by`` column is missing or named twice, two
            key columns share a name, a key column names a column of the
            result, or two groups get one name.
        MemoryError: If grouping the cells does not fit the memory budget.
    """
    columns = group_columns(group_by, "group_by", reserved)
    if groups is not None and not isinstance(groups, ArtifactRef):
        raise TypeError("groups must be an ArtifactRef or None")
    if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef or None")
    for column in columns:
        if column not in cells.columns:
            raise ValueError(
                f"group_by names {column!r}, which is not a cell metadata column"
            )
    key_columns = (
        *columns,
        *((groups.kind,) if groups is not None else ()),
        *(name for name, _labels in kept_apart),
    )
    repeated = [column for column in key_columns if key_columns.count(column) > 1]
    if repeated:
        raise ValueError(
            f"The groups' key columns {list(key_columns)} name {repeated[0]!r} "
            "twice; copy the cell metadata column under another name"
        )
    clashes = [column for column in key_columns[len(columns) :] if column in reserved]
    if clashes:
        raise ValueError(
            f"The groups' key column {clashes[0]!r} is also a column of the "
            "result; group the cells by labels of another kind"
        )
    has = np.asarray(rows, dtype=bool).copy()
    n_cells = len(has)
    n_columns = len(key_columns)
    resident = n_cells * (
        GROUPING_BYTES_PER_CELL + GROUPING_BYTES_PER_COLUMN * n_columns
    )
    if resident > memory_bytes:
        raise MemoryError(
            f"Grouping {n_cells} cells by {n_columns} columns holds about "
            f"{resident} bytes, more than the memory budget of {memory_bytes} "
            "bytes. Raise mem_budget."
        )
    cell_index = _selection_indices(zw, artifact_cells)
    if len(cell_index) != n_cells:
        raise ValueError("rows must hold one entry per cell of the artifact")
    if cell_selection is not None:
        has &= np.isin(cell_index, _selection_indices(zw, cell_selection))
    cell_index = cell_index[has]
    # Each column's labels are coded once, in the natural order of their
    # texts, and only the distinct labels are checked and named.
    coded: list[tuple[np.ndarray, list[str]]] = [
        category_codes(
            read_metadata_rows(cells, column, cell_index),
            read_metadata_missing_rows(cells, column, cell_index),
        )
        for column in columns
    ]
    if groups is not None:
        resolved = resolve_grouping(zw, cells, groups)
        codes, names = category_codes(resolved.labels, resolved.missing_mask)
        at = pd.Index(np.asarray(resolved.cell_idx, dtype=np.int64)).get_indexer(
            cell_index
        )
        aligned = np.full(len(cell_index), -1, dtype=np.int64)
        aligned[at >= 0] = codes[at[at >= 0]]
        coded.append((aligned, names))
    for name, labels in kept_apart:
        values = np.asarray(labels)
        if values.shape != (n_cells,):
            raise ValueError(f"The labels of {name!r} must hold one per artifact cell")
        coded.append(category_codes(values[has]))
    cell_codes = np.full(n_cells, -1, dtype=np.int64)
    if not coded:
        cell_codes[has] = 0
        return CellGroups(key_columns, [{}] if has.any() else [], cell_codes)
    stacked = np.column_stack([codes for codes, _names in coded])
    keep = (stacked >= 0).all(axis=1)
    combinations, group_codes = unique_rows(stacked[keep])
    keys = [
        {
            column: names[code]
            for column, (_codes, names), code in zip(
                key_columns, coded, row, strict=True
            )
        }
        for row in combinations.tolist()
    ]
    selected = np.flatnonzero(has)
    cell_codes[selected[keep]] = group_codes
    # Groups are named by their labels joined by commas, and need one name
    # each.
    group_names(keys)
    return CellGroups(key_columns, keys, cell_codes)
