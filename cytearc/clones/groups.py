"""The groups of cells that clone summaries count within, and their clones.

A group is a combination of labels, one in each key column, such as a sample
and a cell type. Tables and figures name it by its labels joined by commas.
The summaries count each clone's cells within each group, from each cell's
group and clone codes. These functions work on labels and codes and never
touch storage.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["group_frame", "group_names", "pair_counts"]


def group_names(group_keys: Sequence[dict[str, str]]) -> list[str]:
    """Name each group by its labels joined by commas, or ``"all cells"``.

    Raises:
        ValueError: If two groups get one name, as labels that hold a comma
            can give them.
    """
    names = [", ".join(key.values()) if key else "all cells" for key in group_keys]
    first: dict[str, dict[str, str]] = {}
    for key, name in zip(group_keys, names, strict=True):
        if name in first and first[name] != key:
            raise ValueError(
                f"The groups {first[name]} and {key} are both named {name!r}; "
                "remove the commas from their labels"
            )
        first.setdefault(name, key)
    return names


def group_frame(
    key_columns: Sequence[str], group_keys: Sequence[dict[str, str]]
) -> pd.DataFrame:
    """One row per group: ``group``, its name, and its label in each key column.

    The tables of grouped results take their first columns from these rows.

    Raises:
        ValueError: If two groups get one name, as :func:`group_names` names
            them.
    """
    return pd.DataFrame(
        {
            "group": group_names(group_keys),
            **{column: [key[column] for key in group_keys] for column in key_columns},
        }
    )


def pair_counts(
    groups: np.ndarray, clones: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count the cells of each distinct pair of group and clone.

    Args:
        groups: Each cell's group code, from 0.
        clones: Each cell's clone code, from 0.

    Returns:
        The group, the clone code and the cells of each distinct pair, sorted
        by group and clone code.
    """
    groups = np.asarray(groups, dtype=np.int64)
    clones = np.asarray(clones, dtype=np.int64)
    n_codes = int(clones.max()) + 1 if len(clones) else 1
    # Sorting and comparing neighbours counts each pair, as np.unique does,
    # without hashing many distinct pairs.
    ordered = np.sort(groups * n_codes + clones)
    starts = np.flatnonzero(np.diff(ordered, prepend=-1) != 0)
    pairs = ordered[starts]
    cells = np.diff(starts, append=len(ordered)).astype(np.int64)
    return pairs // n_codes, pairs % n_codes, cells
