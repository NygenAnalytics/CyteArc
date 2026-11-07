"""Group the cells of a cell-aligned artifact by their labels.

Summaries such as clonal expansion and diversity are measured within groups
of cells: the combinations of the labels of cell metadata columns, such as
the sample, of a cell-label artifact, such as a clustering, and of labels that
the measured artifact keeps apart. ``cell_groups`` codes each column's labels
once, in the natural order of their texts, and numbers the combinations that
the cells hold, so every summary groups and names its groups alike. It charges
what it reads and builds to the caller's memory ledger.
"""

import sys
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
    read_rows,
    read_text_rows,
)
from ...metadata.selection import (
    category_codes,
    hashed_text_bytes,
    plan_cell_selection_read,
    read_cell_selection,
    resolve_grouping,
)
from ...metadata.table import MetaData
from ...storage.artifacts import (
    ValueFingerprintBuilder,
    artifact_group,
    canonical_bytes,
)
from ...storage.categorical import CategoricalColumn
from ...storage.refs import ArtifactRef
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scope
from ...utils.arrays import unique_rows
from ...utils.arguments import quoted_value
from ...utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
    hash_table_bytes,
    memory_refusal,
    object_array_bytes,
    python_text_bytes,
    sort_bytes,
)

__all__ = [
    "GROUPING_BYTES_PER_GROUP",
    "GROUPING_BYTES_PER_GROUP_COLUMN",
    "CellGroups",
    "cell_groups",
    "group_columns",
    "selection_chunk_bytes",
    "within_groups",
]

#: Each group's labels, a dictionary of its labels by key column, cost this
#: and this for each key column beside the labels' text, which the codes of
#: each column name, while they are built from the groups' codes as lists of
#: numbers and while the groups are named by their labels joined.
GROUPING_BYTES_PER_GROUP = 250
GROUPING_BYTES_PER_GROUP_COLUMN = 70


def _combining_bytes(n_cells: int, n_columns: int) -> int:
    """What numbering the combinations of ``n_columns`` key columns holds for ``n_cells`` grouped cells, beside the columns' codes.

    While a column is combined: the stacked codes and their copy of the cells
    with every label, 16 bytes a cell for each column, the flags of the
    labels, a byte for each, and 70 a cell for the column's offsets, its
    combined codes, their order, sorted copy and distinct values, and each
    cell's position among them. Once the last is combined: the stacked codes,
    their copy and the distinct rows, 24 bytes a cell for each column when
    each cell is a group of its own, and 56 a cell for the codes and the
    rows' representatives.
    """
    return int(n_cells) * max(17 * n_columns + 70, 24 * n_columns + 56)


def group_columns(value: Any, name: str = "group_by") -> tuple[str, ...]:
    """Return the cell metadata columns of an argument as a tuple of names.

    A column may take any name, since results hold each group's labels in a
    table of their own, apart from their own columns.

    Raises:
        TypeError: If ``value`` is not a column name, a sequence of names, or
            None.
        ValueError: If it names a column twice.
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
        n_outside_rows: The cells of ``cell_selection``, or of the artifact
            without one, that are not among the rows that may join a group,
            such as the cells without a clone.
        n_without_group: The cells among those rows that have no group,
            since a column or the label artifact gives them no label.
    """

    key_columns: tuple[str, ...]
    keys: list[dict[str, str]]
    codes: np.ndarray
    n_outside_rows: int = 0
    n_without_group: int = 0

    @property
    def diagnostics(self) -> dict[str, int]:
        """The grouping's diagnostics, as a summary of clones saves them.

        A clone summary groups the cells with a clone, so
        ``cellsWithoutClone`` counts the measured cells outside the rows,
        ``cellsWithoutGroup`` the cells with a clone but no group, and
        ``groups`` the groups.
        """
        return {
            "cellsWithoutClone": int(self.n_outside_rows),
            "cellsWithoutGroup": int(self.n_without_group),
            "groups": len(self.keys),
        }

    def fingerprint_bytes(self) -> int:
        """Return the most bytes that :meth:`fingerprint` holds, beside the codes, which it reads in place.

        Each group's keys become a tree of lists, about 180 bytes and 176 for
        each key column, and then JSON text and its UTF-8 bytes: 14
        characters a group and, for each key column, its name, its label and
        17 more. Python keeps the whole text at the width of its widest
        character, so each character takes 2 bytes for ASCII text, 3 to
        U+00FF, 5 to U+FFFF and 8 beyond.
        """
        texts = [text for key in self.keys for item in key.items() for text in item]
        characters = len(self.keys) * (14 + 17 * len(self.key_columns)) + sum(
            map(len, texts)
        )
        widest = max((ord(max(text)) for text in texts if text), default=0)
        width = next(
            width
            for limit, width in ((0x7F, 2), (0xFF, 3), (0xFFFF, 5), (0x10FFFF, 8))
            if widest <= limit
        )
        return (
            len(self.keys) * (180 + 176 * len(self.key_columns))
            + width * (characters)
            + 4096
        )

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


def selection_chunk_bytes(zw: zarr.Group, selection: ArtifactRef) -> int:
    """The bytes of one stored chunk of a cell selection, which reading it holds.

    A selection stores a flag for each cell of the store, so the chunk can
    outweigh the cells it selects.
    """
    values = as_zarr_array(artifact_group(zw, selection)["values"], name="values")
    return array_row_selection_parts(values)[0]


def _column_values(
    ledger: MemoryLedger, cells: MetaData, column: str, cell_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Read a column's rows of the cells, text as Python strings, and its missing mask.

    Each string beyond ASCII is charged beside it the UTF-8 form that coding
    the labels leaves in it, as
    :func:`~cytearc.metadata.selection.hashed_text_bytes` counts it; the
    rows of a categorical column share one string per category.

    Returns:
        The values, the mask or None, and the bytes that both keep charged.
    """
    array = cells._get_array(column)
    item = f"the labels of {quoted_value(column)}"
    charged = ledger.charged
    if np.dtype(array.dtype).kind == "U":
        values = read_text_rows(ledger, array, cell_index, item=item)
        strings = (
            array.categories.tolist()
            if isinstance(array, CategoricalColumn)
            else values
        )
        ledger.hold(item, hashed_text_bytes(strings))
    else:
        values = read_rows(ledger, array, cell_index, item=item)
    mask = metadata_missing_mask(cells, column)
    missing = (
        None
        if mask is None
        else np.asarray(read_rows(ledger, mask, cell_index, item=item), dtype=bool)
    )
    return values, missing, ledger.charged - charged


def _hold_names(ledger: MemoryLedger, names: list[str]) -> None:
    """Charge the distinct labels of a key column, which the groups' labels keep as they are."""
    ledger.hold(
        "the distinct labels of a key column",
        object_array_bytes(len(names)) + sum(map(sys.getsizeof, names)),
    )


def _keys_bytes(n_groups: int, n_columns: int) -> int:
    """The bytes of the labels and names of ``n_groups`` groups of ``n_columns`` key columns."""
    return n_groups * (
        GROUPING_BYTES_PER_GROUP + GROUPING_BYTES_PER_GROUP_COLUMN * n_columns
    )


def cell_groups(
    zw: zarr.Group,
    cells: MetaData,
    artifact_cells: ArtifactRef,
    rows: np.ndarray,
    *,
    group_by: str | Sequence[str] | None,
    groups: ArtifactRef | None,
    cell_selection: ArtifactRef | None,
    ledger: MemoryLedger,
    kept_apart: Sequence[tuple[str, np.ndarray]] = (),
) -> CellGroups:
    """Group the cells of a cell-aligned artifact by their labels.

    The groups are the combinations of the labels of the ``group_by``
    columns, of the ``groups`` artifact and of ``kept_apart``. A cell outside
    ``rows`` or ``cell_selection``, or without a label, is in no group. A
    label artifact may hold fewer cells than the artifact: the cells it lacks
    have no group.

    What grouping reads and builds is charged to ``ledger`` before each step,
    beside what the caller holds there: the cells of the artifact's
    selection, read whatever the grouping; the cells of ``cell_selection``
    and their match; each column's labels, read as
    :func:`~cytearc.metadata.rows.read_text_rows` and
    :func:`~cytearc.metadata.rows.read_rows` charge them and coded as
    :func:`~cytearc.metadata.selection.category_codes` charges it; the label
    artifact's labels and their alignment; the combinations of the cells'
    codes; and the groups' labels and names. Each cell's group and the
    groups' labels stay charged when the call returns, as the caller's.

    Args:
        zw: The store's root group.
        cells: The store's cell metadata.
        artifact_cells: The cell selection of the artifact's cells.
        rows: The artifact's cells that may join a group, as a mask aligned
            to its cells.
        group_by: Cell metadata columns.
        groups: A cell-label artifact.
        cell_selection: Only these cells, or None.
        ledger: The memory ledger of the caller.
        kept_apart: Key columns whose labels the artifact keeps apart, each
            with every artifact cell's label, such as a receptor type.

    Returns:
        The groups' key columns and labels, and each cell's group. A key
        column may take any name, even one of the result's own columns,
        since results hold the groups' labels in a table of their own.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a ``group_by`` column is missing, named twice or kept
            apart already, two key columns share a name, or two groups get
            one name.
        MemoryBudgetError: If grouping the cells does not fit the ledger's
            budget.
    """
    columns = group_columns(group_by, "group_by")
    if groups is not None and not isinstance(groups, ArtifactRef):
        raise TypeError("groups must be an ArtifactRef or None")
    if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef or None")
    kept = {name for name, _labels in kept_apart}
    for column in columns:
        if column in kept:
            raise ValueError(
                f"group_by names {quoted_value(column)}, whose labels the artifact keeps apart "
                "already; leave it out, or copy the cell metadata column under "
                "another name"
            )
        if not cells._has_column(column):
            raise ValueError(
                f"group_by names {quoted_value(column)}, which is not a cell metadata column"
            )
    key_columns = (
        *columns,
        *((groups.kind,) if groups is not None else ()),
        *(name for name, _labels in kept_apart),
    )
    repeated = [column for column in key_columns if key_columns.count(column) > 1]
    if repeated:
        raise ValueError(
            f"The groups' key columns {list(key_columns)} name {quoted_value(repeated[0])} "
            "twice; copy the cell metadata column under another name"
        )
    for name, labels in kept_apart:
        if np.shape(labels) != np.shape(rows):
            raise ValueError(
                f"The labels of {quoted_value(name)} must hold one per artifact cell"
            )
    # The selections' identity checks, which read their row identifiers
    # outside the budget, run once, before anything is charged.
    with validation_scope():
        artifact_read = plan_cell_selection_read(zw, artifact_cells)
        selection_read = (
            None
            if cell_selection is None
            else plan_cell_selection_read(zw, cell_selection)
        )
        n_cells = len(rows)
        n_columns = len(key_columns)
        # Whether each cell may join a group, and each cell's group, are held
        # until the call returns.
        ledger.hold("the cells that may join a group", n_cells)
        has = np.asarray(rows, dtype=bool).copy()
        cell_index = read_cell_selection(
            ledger, zw, artifact_read, item="the cells of the artifact's selection"
        )
        if len(cell_index) != n_cells:
            raise ValueError("rows must hold one entry per cell of the artifact")
        if selection_read is not None:
            selected = read_cell_selection(
                ledger, zw, selection_read, item="the cells of cell_selection"
            )
            # NumPy matches the two sorted sets of positions by a stable argsort
            # of both together: their copy, its order, the sorted positions and
            # the flags that mark the matches.
            both = n_cells + len(selected)
            ledger.peak(
                "the cells of cell_selection",
                16 * both + sort_bytes(both, 8, stable=True, order=True) + 4 * both,
            )
            in_selection = np.isin(cell_index, selected)
            n_measured = int(np.count_nonzero(in_selection))
            has &= in_selection
            ledger.release(selected.nbytes)
            del selected, in_selection
        else:
            n_measured = n_cells
        ledger.peak("the grouped cells", 8 * int(np.count_nonzero(has)))
        grouped_index = cell_index[has]
        ledger.hold("the grouped cells", grouped_index.nbytes)
        index_bytes = cell_index.nbytes
        del cell_index
        ledger.release(index_bytes)
        n_grouped = len(grouped_index)
        n_outside_rows = n_measured - n_grouped
        # Each column's labels are coded once, in the natural order of their
        # texts, and only the codes are kept.
        coded: list[tuple[np.ndarray, list[str]]] = []
        for column in columns:
            values, missing, read = _column_values(ledger, cells, column, grouped_index)
            codes, names = category_codes(values, missing, ledger=ledger)
            ledger.hold(f"each cell's code of {quoted_value(column)}", codes.nbytes)
            # The distinct labels share their strings with the cells' labels,
            # so they are held before the cells' labels go.
            _hold_names(ledger, names)
            del values, missing
            ledger.release(read)
            coded.append((codes, names))
        if groups is not None:
            # The label artifact is read whole, whatever cells the artifact
            # being grouped holds, within the ledger, which keeps its labels,
            # missing mask and cells' positions charged, and aligned to the
            # grouped cells through an index of its cells.
            entry = ledger.charged
            try:
                resolved = resolve_grouping(zw, cells, groups, ledger=ledger)
            except MemoryBudgetError as refused:
                # The read of a label artifact's values words its refusal as
                # every read of cell values does; the grouping names the same
                # bytes and the budget that passes them in the ledger's words.
                raise memory_refusal(
                    ledger.doing,
                    refused.item,
                    refused.needed_bytes,
                    refused.budget_bytes,
                    remedy=ledger.remedy,
                ) from refused
            n_labelled = len(resolved.cell_idx)
            labels = np.asarray(resolved.labels)
            if labels.dtype.kind == "U":
                # Stored text becomes the Python strings that coding hashes,
                # each measured by its characters and the widest of them.
                characters = int(np.char.str_len(labels).sum()) if labels.size else 0
                widest = int(labels.view(np.uint32).max()) if labels.size else 0
                ledger.hold(
                    "the labels of the groups artifact",
                    object_array_bytes(len(labels))
                    + python_text_bytes(len(labels), characters, widest=widest),
                )
                labels = labels.astype(object)
            labelled = ledger.charged - entry
            codes, names = category_codes(labels, resolved.missing_mask, ledger=ledger)
            ledger.hold("each labelled cell's code", codes.nbytes)
            ledger.peak(
                "the groups artifact's cells",
                hash_table_bytes(n_labelled) + 8 * n_labelled + 16 * n_grouped,
            )
            at = pd.Index(np.asarray(resolved.cell_idx, dtype=np.int64)).get_indexer(
                grouped_index
            )
            aligned = np.full(n_grouped, -1, dtype=np.int64)
            aligned[at >= 0] = codes[at[at >= 0]]
            ledger.hold("each cell's code of the groups artifact", aligned.nbytes)
            labelled_codes = codes.nbytes
            _hold_names(ledger, names)
            del resolved, labels, at, codes
            ledger.release(labelled + labelled_codes)
            coded.append((aligned, names))
        for name, labels in kept_apart:
            # The labels kept apart are the caller's; their grouped cells' are
            # picked out to be coded.
            picking = ledger.hold(
                f"the labels of {quoted_value(name)}", object_array_bytes(n_grouped)
            )
            picked = np.asarray(labels)[has]
            codes, names = category_codes(picked, ledger=ledger)
            ledger.hold(f"each cell's code of {quoted_value(name)}", codes.nbytes)
            _hold_names(ledger, names)
            del picked
            ledger.release(picking)
            coded.append((codes, names))
        ledger.hold("each cell's group", 8 * n_cells)
        cell_codes = np.full(n_cells, -1, dtype=np.int64)
        kept_codes = sum(codes.nbytes for codes, _names in coded)
        if not coded:
            cell_codes[has] = 0
            keys: list[dict[str, str]] = [{}] if has.any() else []
            index_bytes = grouped_index.nbytes
            del has, grouped_index
            ledger.release(n_cells + index_bytes)
            return CellGroups(key_columns, keys, cell_codes, n_outside_rows, 0)
        # The codes are stacked, the cells with a label in every column kept,
        # and their combinations numbered a column at a time.
        ledger.peak(
            "the combinations of the cells' labels",
            _combining_bytes(n_grouped, n_columns),
        )
        stacked = np.column_stack([codes for codes, _names in coded])
        keep = (stacked >= 0).all(axis=1)
        # The grouped cells without a label in some column have no group.
        n_without_group = n_grouped - int(np.count_nonzero(keep))
        combinations, group_codes = unique_rows(stacked[keep])
        del stacked
        combined = keep.nbytes + combinations.nbytes + group_codes.nbytes
        ledger.hold("the combinations of the cells' labels", combined)
        # Each group's labels and name are Python objects, built once the groups
        # are counted from the combinations as lists, and named once.
        keys_bytes = _keys_bytes(len(combinations), n_columns)
        ledger.hold("each group's labels and name", keys_bytes)
        keys = [
            {
                column: names[code]
                for column, (_codes, names), code in zip(
                    key_columns, coded, row, strict=True
                )
            }
            for row in combinations.tolist()
        ]
        # The grouped cells' positions, and those with a label in every column.
        ledger.peak("each cell's group", 16 * n_grouped)
        selected_cells = np.flatnonzero(has)
        cell_codes[selected_cells[keep]] = group_codes
        # Groups are named by their labels joined by commas, and need one name
        # each, which is checked beside what the grouping still holds.
        group_names(keys)
        index_bytes = grouped_index.nbytes
        del coded, keep, combinations, group_codes, selected_cells, has, grouped_index
        ledger.release(n_cells + index_bytes + kept_codes + combined)
        return CellGroups(
            key_columns, keys, cell_codes, n_outside_rows, n_without_group
        )


def within_groups(
    zw: zarr.Group,
    cells: MetaData,
    cell_selection: ArtifactRef,
    columns: tuple[str, ...],
    members: np.ndarray,
    *,
    ledger: MemoryLedger,
    what: str,
) -> tuple[np.ndarray, list[dict[str, str]], str | None]:
    """Group an artifact's member cells by their labels in the ``within`` columns.

    Clone-defining methods define clones within groups, such as each mouse:
    the groups are those of :func:`cell_groups`, in the natural order of
    their labels, identified by its fingerprint of each cell's group, and
    are charged to ``ledger`` as :func:`cell_groups` charges them. Each
    cell's group and the groups' labels stay charged, as the caller's.

    Args:
        zw: The store's root group.
        cells: The store's cell metadata.
        cell_selection: The selection of the artifact's cells.
        columns: The within columns.
        members: The cells that need a group, as a mask aligned to the
            artifact's cells, such as the cells with a receptor.
        ledger: The memory ledger of the caller.
        what: What the member cells hold, such as ``"a receptor"``, which
            the error names.

    Returns:
        Each cell's group, 0 for a cell that is not a member; each group's
        labels by column; and the fingerprint of the cells' groups. Without
        columns, every cell is in one group without labels, and the
        fingerprint is None.

    Raises:
        ValueError: If a column is not a cell metadata column, or a member
            cell has no label in one.
        MemoryBudgetError: If grouping the cells does not fit the ledger's
            budget.
    """
    n_cells = len(members)
    if not columns:
        ledger.hold("each cell's group", 8 * n_cells)
        return np.zeros(n_cells, dtype=np.int64), [{}], None
    for column in columns:
        if not cells._has_column(column):
            raise ValueError(
                f"within names {quoted_value(column)}, which is not a cell metadata column"
            )
    grouped = cell_groups(
        zw,
        cells,
        cell_selection,
        members,
        group_by=columns,
        groups=None,
        cell_selection=None,
        ledger=ledger,
    )
    unlabelled = int(np.count_nonzero(members & (grouped.codes < 0)))
    if unlabelled:
        raise ValueError(
            f"The within columns have no label for {unlabelled} cells with {what}. "
            f"Every cell with {what} needs a label in the within columns."
        )
    # Each cell's group in place of its code, which is let go, and the
    # fingerprint of the groups' labels beside it.
    ledger.peak("each cell's group", 8 * n_cells)
    codes = np.maximum(grouped.codes, 0)
    ledger.peak(
        "fingerprinting the groups' labels",
        codes.nbytes + grouped.fingerprint_bytes(),
    )
    return codes, grouped.keys, grouped.fingerprint()
