"""Group selected cells into samples by their label in a cell metadata column.

Operations that work within each sample, such as guide calling, Mixscape and
hashtag demultiplexing, read one column's labels for their selected cells and
number the samples alike, charging the read to the caller's memory ledger.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...metadata.rows import (
    apply_missing_mask,
    metadata_missing_mask,
    read_categorical_rows,
    read_rows,
    read_text_rows,
)
from ...metadata.selection import sample_codes
from ...storage.categorical import CategoricalColumn
from ...utils.memory_budget import MemoryLedger, object_array_bytes

# The bytes of a label that is not text as a Python object, such as a
# number, beside its reference.
_NUMBER_BYTES = 32


@dataclass(frozen=True, slots=True)
class CellSamples:
    """The samples of some cells, by their label in one cell metadata column.

    Attributes:
        columns: The column, or no column for one sample of every cell.
        codes: Each cell's sample, numbered in the sorted order of the labels.
        names: Each sample's label, None for the one sample without a column.
        assignments: The fingerprint of the cells' samples, None without a
            column.
    """

    columns: tuple[str, ...]
    codes: np.ndarray
    names: list[str | None]
    assignments: str | None


def require_cell_column(store: Any, column: str | None, argument: str) -> None:
    """Refuse an ``argument`` that names a cell metadata column the store lacks.

    Raises:
        ValueError: If ``column`` is not None and not a cell column.
    """
    if column is not None and column not in store.cells.columns:
        raise ValueError(f"{argument} '{column}' not found in cell metadata")


def read_cell_samples(
    store: Any,
    ledger: MemoryLedger,
    column: str | None,
    cell_index: np.ndarray,
    *,
    reason: str,
) -> CellSamples:
    """Group cells into samples by their label in a cell metadata column.

    Without a column every cell is in one sample. ``reason`` says why every
    cell needs a label; it ends the error for a cell without one. The labels
    and their missing mask are read through ``ledger``, and numbering them
    charges it as :func:`~cytearc.metadata.selection.sample_codes` states;
    the cells' sample codes stay charged and the labels are let go. A
    categorical column is read as a pandas Categorical of the cells' codes,
    whose code -1 marks each row that its missing mask flags, and numbered
    through its categories, so no cell's label is made.
    """
    columns = () if column is None else (column,)
    n_cells = len(cell_index)
    labels: list[np.ndarray | pd.Categorical] = []
    before = ledger.charged
    for name in columns:
        array = store.cells._get_array(name)
        item = f"the cells' labels in {name!r}"
        if isinstance(array, CategoricalColumn):
            labels.append(read_categorical_rows(ledger, array, cell_index, item=item))
            continue
        if np.dtype(array.dtype).kind == "U":
            values = read_text_rows(ledger, array, cell_index, item=item)
        else:
            values = read_rows(ledger, array, cell_index, item=item)
            # The labels as Python objects, as numbering them makes them.
            ledger.hold(item, object_array_bytes(n_cells) + _NUMBER_BYTES * n_cells)
            values = values.astype(object)
        mask = metadata_missing_mask(store.cells, name)
        missing = (
            None
            if mask is None
            else read_rows(
                ledger, mask, cell_index, item=f"the cells' missing labels in {name!r}"
            ).astype(bool, copy=False)
        )
        if missing is not None and missing.any():
            # The labels with the missing ones as None, an object copy.
            ledger.hold(item, object_array_bytes(n_cells))
        labels.append(apply_missing_mask(values, missing, labels=True))
        del values, missing
    held = ledger.charged - before
    codes, sample_labels, assignments = sample_codes(
        columns, labels, n_cells, reason=reason, ledger=ledger
    )
    del labels
    ledger.release(held)
    return CellSamples(
        columns=columns,
        codes=codes,
        names=[None if column is None else by[column] for by in sample_labels],
        assignments=assignments,
    )
