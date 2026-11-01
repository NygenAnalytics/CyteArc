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
from ...metadata.rows import (
    array_row_selection_parts,
    metadata_missing_mask,
    previous_read_bytes,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import (
    category_codes,
    grouping_value_name,
    resolve_grouping,
)
from ...metadata.table import MetaData
from ...storage.artifacts import (
    ValueFingerprintBuilder,
    artifact_group,
    canonical_bytes,
    open_artifact,
)
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices, validate_cell_selection
from ...storage.types import as_zarr_array
from ...utils.arrays import unique_rows

__all__ = [
    "GROUPING_BYTES_PER_CELL",
    "GROUPING_BYTES_PER_COLUMN",
    "GROUPING_BYTES_PER_DISTINCT_LABEL",
    "GROUPING_BYTES_PER_GROUP",
    "GROUPING_BYTES_PER_LABEL_BYTE",
    "GROUPING_BYTES_PER_LABELLED_CELL",
    "GROUPING_BYTES_PER_SELECTED_CELL",
    "CellGroups",
    "cell_groups",
    "group_columns",
]

#: Grouping cells holds, at its peak, about ``GROUPING_BYTES_PER_CELL`` for
#: each cell of the artifact: its position, code and group. Each key column
#: adds ``GROUPING_BYTES_PER_COLUMN`` per cell for the cell's code, and a cell
#: metadata column also ``GROUPING_BYTES_PER_LABEL_BYTE`` times the stored
#: width of its labels, which are read and coded as text; one column of
#: 20-character labels took 181 bytes per cell in ASCII and 288 in emoji, as
#: traced on 200,000 cells.
GROUPING_BYTES_PER_CELL = 100
GROUPING_BYTES_PER_COLUMN = 70
GROUPING_BYTES_PER_LABEL_BYTE = 3
#: Each cell of a ``cell_selection`` costs this, its row and its place in the
#: match with the artifact's cells; 25 bytes as traced.
GROUPING_BYTES_PER_SELECTED_CELL = 32
#: Each cell of a label artifact costs this, its row, position and code, and
#: ``GROUPING_BYTES_PER_LABEL_BYTE`` times the stored width of its labels,
#: whatever cells the artifact being grouped holds.
GROUPING_BYTES_PER_LABELLED_CELL = 130
#: Each distinct label of a key column costs this while it is named and
#: ordered, 420 bytes as traced on 200,000 distinct labels, and each group this
#: for its labels and name, 400 bytes as traced on 180,057 groups; a grouping
#: with a label per cell holds far more than its cells.
GROUPING_BYTES_PER_DISTINCT_LABEL = 500
GROUPING_BYTES_PER_GROUP = 500


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

    def fingerprint(self) -> str:
        """Fingerprint each cell's group by its labels.

        The groups' labels and each cell's code identify every cell's labels,
        so other labels give another fingerprint, at 8 bytes per cell rather
        than a text per cell.
        """
        builder = ValueFingerprintBuilder()
        builder.update_bytes("key_columns", canonical_bytes(list(self.key_columns)))
        builder.update_bytes("keys", canonical_bytes(self.keys))
        builder.update_array("codes", np.asarray(self.codes, dtype=np.int64))
        return builder.hexdigest()


def _selection_indices(zw: zarr.Group, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)


def selection_chunk_bytes(zw: zarr.Group, selection: ArtifactRef) -> int:
    """The bytes of one stored chunk of a cell selection, which reading it holds.

    A selection stores a flag for each cell of the store, so the chunk can
    outweigh the cells it selects.
    """
    values = as_zarr_array(artifact_group(zw, selection)["values"], name="values")
    return array_row_selection_parts(values)[0]


def _column_chunk_bytes(cells: MetaData, column: str) -> int:
    """The bytes of one chunk of a column and its missing mask, which a read holds."""
    chunk, _per_row = array_row_selection_parts(cells._get_array(column))
    mask = metadata_missing_mask(cells, column)
    if mask is not None:
        chunk += array_row_selection_parts(mask)[0]
    return chunk


def _stored_in_chunks(cells: MetaData, column: str) -> bool:
    """Whether a column holds more than one stored chunk, which its rows can span."""
    array = cells._get_array(column)
    return int(array.shape[0]) > int(array.chunks[0])


def _label_extent(zw: zarr.Group, groups: ArtifactRef) -> tuple[int, int, int]:
    """The cells of a label artifact, the stored width of its labels, and a chunk's bytes.

    Only the array's metadata is read. An artifact that is missing or
    incomplete counts nothing here, since ``resolve_grouping`` refuses it
    before reading anything.
    """
    value_name = grouping_value_name(groups.kind)
    status, group = open_artifact(zw, groups)
    if group is None or not status.complete or value_name not in group:
        return 0, 0, 0
    values = as_zarr_array(group[value_name], name=value_name)
    chunk, _per_row = array_row_selection_parts(values)
    n_labelled = int(values.shape[0]) if values.shape else 0
    return n_labelled, int(values.dtype.itemsize), chunk


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
    label_bytes = sum(cells.get_dtype(column).itemsize for column in columns)
    resident = n_cells * (
        GROUPING_BYTES_PER_CELL
        + GROUPING_BYTES_PER_COLUMN * n_columns
        + GROUPING_BYTES_PER_LABEL_BYTE * label_bytes
    )
    # Reading a column, or the label artifact, also holds the rest of its
    # last chunk; they are read one at a time, a column a stored chunk at a
    # time.
    chunks = [_column_chunk_bytes(cells, column) for column in columns]
    # A cell selection and a label artifact are read whole, whatever cells
    # the artifact holds.
    if cell_selection is not None:
        n_selected = validate_cell_selection(zw, cell_selection).selected_count
        resident += GROUPING_BYTES_PER_SELECTED_CELL * int(n_selected)
        # The selection is read right after the artifact's cells, whose read
        # can still hold its stored chunk, a flag for each cell of the store.
        resident += previous_read_bytes(
            selection_chunk_bytes(zw, artifact_cells),
            selection_chunk_bytes(zw, cell_selection),
        )
    if groups is not None:
        n_labelled, width, label_chunk = _label_extent(zw, groups)
        resident += n_labelled * (
            GROUPING_BYTES_PER_LABELLED_CELL + GROUPING_BYTES_PER_LABEL_BYTE * width
        )
        chunks.append(label_chunk)
    resident += max(chunks, default=0)
    # The read of a chunk can still hold the chunk read before it, of the same
    # column or of the column or labels before, so a column stored in several
    # chunks, or several columns and labels, count the largest chunk again.
    if len(chunks) > 1 or any(_stored_in_chunks(cells, column) for column in columns):
        resident += previous_read_bytes(*chunks)

    def held(total: int) -> int:
        """Return the bytes that grouping holds, refused beyond the budget."""
        if total > memory_bytes:
            raise MemoryError(
                f"Grouping {n_cells} cells by {n_columns} columns holds about "
                f"{total} bytes, more than the memory budget of {memory_bytes} "
                "bytes. Raise mem_budget."
            )
        return total

    def coded_labels(
        values: np.ndarray, missing: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Code labels as ``category_codes`` does, once their distinct labels fit.

        Naming and ordering a distinct label costs far more than coding a
        cell, so the cells are factorized first and only the distinct labels
        are named, after the budget is checked with their number.
        """
        nonlocal resident
        factorized, distinct = pd.factorize(values, use_na_sentinel=True)
        resident = held(resident + GROUPING_BYTES_PER_DISTINCT_LABEL * len(distinct))
        distinct_codes, names = category_codes(np.asarray(distinct, dtype=object))
        present = factorized >= 0
        if missing is not None:
            present &= ~np.asarray(missing, dtype=bool)
        codes = np.full(len(factorized), -1, dtype=np.int64)
        codes[present] = distinct_codes[factorized[present]]
        return codes, names

    resident = held(resident)
    cell_index = _selection_indices(zw, artifact_cells)
    if len(cell_index) != n_cells:
        raise ValueError("rows must hold one entry per cell of the artifact")
    if cell_selection is not None:
        has &= np.isin(cell_index, _selection_indices(zw, cell_selection))
    cell_index = cell_index[has]
    # Each column's labels are coded once, in the natural order of their
    # texts, and only the distinct labels are checked and named.
    coded: list[tuple[np.ndarray, list[str]]] = [
        coded_labels(
            np.asarray(read_metadata_rows_chunkwise(cells, column, cell_index)),
            read_metadata_missing_rows_chunkwise(cells, column, cell_index),
        )
        for column in columns
    ]
    if groups is not None:
        resolved = resolve_grouping(zw, cells, groups)
        codes, names = coded_labels(np.asarray(resolved.labels), resolved.missing_mask)
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
        coded.append(coded_labels(values[has]))
    cell_codes = np.full(n_cells, -1, dtype=np.int64)
    if not coded:
        cell_codes[has] = 0
        return CellGroups(key_columns, [{}] if has.any() else [], cell_codes)
    stacked = np.column_stack([codes for codes, _names in coded])
    keep = (stacked >= 0).all(axis=1)
    combinations, group_codes = unique_rows(stacked[keep])
    # Each group's labels and name are Python objects, built once the groups
    # are counted.
    held(resident + GROUPING_BYTES_PER_GROUP * len(combinations))
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
