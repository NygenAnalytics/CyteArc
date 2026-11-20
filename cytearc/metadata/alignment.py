"""Align per-cell values to the cells they describe by cell ID.

A store's cells need not be in the order of values computed elsewhere: a
merge reorders them, and so does any tool that sorts or subsets its own
cells. Public calls that take one value per cell therefore take the values
keyed by cell ID, as a pandas Series or DataFrame whose index holds the IDs,
and refuse values that would land on cells by position.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import logger
from ..utils.memory_budget import (
    fixed_text_bytes,
    hash_table_bytes,
    object_array_bytes,
)
from .rows import STRING_BYTES, text_read_bytes, text_width


def positional_kind(values: Any) -> str:
    """Describe values that a call refuses, for messages."""
    if isinstance(values, pd.Series | pd.DataFrame):
        container = f"a pandas {type(values).__name__}"
        return (
            f"{container} with a RangeIndex"
            if isinstance(values.index, pd.RangeIndex)
            else container
        )
    if isinstance(values, np.ndarray):
        return "a NumPy array"
    kind = type(values)
    package = kind.__module__.partition(".")[0]
    return f"values of type {'pandas ' if package == 'pandas' else ''}{kind.__name__}"


def keyed_values_remedy(container: str = "Series") -> str:
    """Return how to key the values of a call that leaves out other cells.

    ``container`` names the pandas type that the call takes, ``"Series"`` or
    ``"DataFrame"``.
    """
    return (
        "Key them by the IDs of the cells they describe, such as "
        f"pandas.{container}(values, index=adata.obs_names), or "
        f'pandas.{container}(values, index=ds.cells.fetch_all("ids")) for '
        "values of every cell in store order; values of other cells are left out."
    )


def require_keyed(values: Any, *, name: str, remedy: str, frame: bool = False) -> None:
    """Refuse values that would land on cells by position, before reading any ID.

    Takes the arguments of :func:`align_to_cell_ids` that name the call.

    Raises:
        TypeError: If the values are not a Series (or with ``frame`` a
            DataFrame), or their index is a ``RangeIndex``.
    """
    container = pd.DataFrame if frame else pd.Series
    if not isinstance(values, container) or isinstance(values.index, pd.RangeIndex):
        keyed = isinstance(values, pd.Series | pd.DataFrame) and not isinstance(
            values.index, pd.RangeIndex
        )
        reason = "" if keyed else ", which would assign values to cells by position"
        raise TypeError(
            f"{name} takes per-cell values only as a pandas {container.__name__} "
            "keyed by cell ID, whose index holds the cell IDs; got "
            f"{positional_kind(values)}{reason}. {remedy}"
        )


def align_to_cell_ids(
    values: Any,
    cell_ids: np.ndarray,
    *,
    name: str,
    cells: str,
    remedy: str,
    frame: bool = False,
    leave_out_others: bool = False,
) -> Any:
    """Return values keyed by cell ID in the order of ``cell_ids``.

    ``values`` must be a pandas Series, or with ``frame`` a DataFrame, whose
    index holds every ID of ``cell_ids`` once. The values come back in the
    order of ``cell_ids`` with a ``RangeIndex``, their dtype kept. Labels of
    other cells are refused, or with ``leave_out_others`` left out.

    Args:
        values: The values that a call received.
        cell_ids: The IDs of the cells that the values must describe, in the
            order the call needs them, usually store order.
        name: The call and parameter that took the values, as messages name
            them, such as ``"trajectory.pseudotime(ss_vec=)"``.
        cells: The cells of ``cell_ids``, as messages name them, such as
            ``"the 40 cells of the graph"``.
        remedy: The sentence that tells how to key the values.
        frame: Take a DataFrame, aligned by its rows, instead of a Series.
        leave_out_others: Leave out labels that are not in ``cell_ids``
            instead of refusing them.

    Raises:
        TypeError: If the values are not a Series (or DataFrame), or their
            index is a ``RangeIndex``, so they would land on cells by position.
        ValueError: If the index repeats a label, lacks an ID of
            ``cell_ids``, or, without ``leave_out_others``, holds another
            label; or if ``cell_ids`` repeat an ID.
    """
    require_keyed(values, name=name, remedy=remedy, frame=frame)
    container = pd.DataFrame if frame else pd.Series
    labels = values.index
    target = pd.Index(np.asarray(cell_ids))
    unique = labels.is_unique and target.is_unique
    positions = labels.get_indexer(target) if unique else np.empty(0, dtype=np.int64)
    covered = len(positions) == len(target) and bool((positions >= 0).all())
    if covered and (leave_out_others or len(labels) == len(target)):
        if len(labels) > len(target):
            logger.info(
                f"{name}: left out the values of {len(labels) - len(target)} "
                f"cells outside {cells}"
            )
        return values.iloc[positions].reset_index(drop=True)
    if not target.is_unique:
        problem = "those ids repeat a value"
    elif not labels.is_unique:
        problem = "the index repeats a label"
    else:
        given = set(labels)
        absent = [cell for cell in target if cell not in given]
        problem = f"{len(absent)} such ids have no label" + (
            f", such as {absent[0]!r}" if absent else ""
        )
        if not leave_out_others:
            known = set(target)
            extra = [label for label in labels if label not in known]
            problem = (
                f"{len(extra)} labels are not such ids"
                + (f", such as {extra[0]!r}" if extra else "")
                + f", and {problem}"
            )
    raise ValueError(
        f"{name} takes the values of a pandas {container.__name__} by ID: its "
        f"index must hold the ids of {cells}, each once; {problem}. {remedy}"
    )


def cell_id_alignment_bytes(ids: Any, n_cells: int, n_labels: int) -> int:
    """Return the most bytes that reading cell IDs and aligning keyed values to them holds.

    ``ids`` is the stored ID column, of which ``n_cells`` rows are read, as
    :func:`~cytearc.metadata.rows.text_read_bytes` states, at the stored
    width, four bytes a character. :func:`align_to_cell_ids` then makes them
    a pandas index, which holds a Python string for each ID and, on pandas 3,
    another copy at the stored width; it hashes the index and the values'
    ``n_labels`` labels, whose table stays with the values; and it holds
    each value's position and the values in order.
    """
    n_cells, n_labels = int(n_cells), int(n_labels)
    width = text_width(ids)
    aligning = (
        2 * fixed_text_bytes(n_cells, width)
        + n_cells * (STRING_BYTES + 4 * width)
        + object_array_bytes(n_cells)
        + hash_table_bytes(n_cells)
        + hash_table_bytes(n_labels)
        + 16 * n_cells
    )
    return max(text_read_bytes(ids, n_cells), aligning)
