"""The groups of cells that clone summaries count within, and their clones.

A group is a combination of labels, one in each key column, such as a sample
and a cell type. Tables and figures name it by its labels joined by commas,
and a summary's record holds each group's labels apart, in a table of one row
per group indexed by that name, so a key column may take any name, even one
of a table's own columns. The summaries count each clone's cells within each
group, from each cell's group and clone codes. These functions work on labels
and codes and never touch storage.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from ..utils.arguments import quoted_value

__all__ = ["checked_group_names", "group_key_table", "group_names", "pair_counts"]


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
                f"The groups {first[name]} and {key} are both named {quoted_value(name)}; "
                "remove the commas from their labels"
            )
        first.setdefault(name, key)
    return names


def group_key_table(
    key_columns: Sequence[str],
    group_keys: Sequence[dict[str, str]],
    *,
    index: str = "group",
) -> pd.DataFrame:
    """One row per group, indexed by its name, with its label in each key column.

    The index, named ``index``, holds each group's name as
    :func:`group_names` gives it, the name that the tables of grouped
    results hold in their own column of that name, and the columns are the
    key columns under their own names, whatever those are, so
    ``table.join(keys, on="group")`` gives each row of such a table its
    group's labels; pandas asks for an ``rsuffix`` when a key column shares
    a name with a column of the table.

    Raises:
        ValueError: If two groups get one name, as :func:`group_names` names
            them.
    """
    return pd.DataFrame(
        {column: [key[column] for key in group_keys] for column in key_columns},
        index=pd.Index(group_names(group_keys), name=index),
    )


def checked_group_names(names: Sequence[str], n_groups: int | None = None) -> pd.Index:
    """The groups' names as an index, one text name for each group.

    The table builders of the summaries take the groups' names, so a frame
    of names and labels, whose rows pandas would turn into tuples, is
    refused rather than named row by row.

    Args:
        names: Each group's name, in the order of the groups.
        n_groups: The number of groups, or None when ``names`` sets it.

    Raises:
        ValueError: If ``names`` is not one text name for each group, names
            two groups alike, or names another number of groups than
            ``n_groups``.
    """
    if isinstance(names, pd.DataFrame | np.ndarray) and np.ndim(names) != 1:
        raise ValueError("names must hold one text name for each group")
    named = pd.Index(names)
    if (
        isinstance(named, pd.MultiIndex)
        or named.hasnans
        or (len(named) and pd.api.types.infer_dtype(named, skipna=False) != "string")
    ):
        raise ValueError("names must hold one text name for each group")
    if n_groups is not None and len(named) != n_groups:
        raise ValueError(
            f"names must name each of the {n_groups} groups, not {len(named)}"
        )
    if named.has_duplicates:
        repeated = named[named.duplicated()][0]
        raise ValueError(
            f"names must name each group once; {quoted_value(repeated)} names two"
        )
    return named


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
    # Sorting and comparing neighbors counts each pair, as np.unique does,
    # without hashing many distinct pairs.
    ordered = np.sort(groups * n_codes + clones)
    starts = np.flatnonzero(np.diff(ordered, prepend=-1) != 0)
    pairs = ordered[starts]
    cells = np.diff(starts, append=len(ordered)).astype(np.int64)
    return pairs // n_codes, pairs % n_codes, cells


def pair_count_bytes(n_cells: int, n_pairs: int) -> int:
    """Return the most bytes that :func:`pair_counts` holds beside its inputs, for ``n_cells`` cells and at most ``n_pairs`` pairs.

    Sorting the cells' pairs holds three int64 values a cell at once: the
    sorted values, those values beside a leading -1, and their differences.
    Making each pair's group, clone and cells holds the sorted values beside
    40 bytes a pair, of which the 24 bytes of the three results stay.
    """
    return int(max(24 * n_cells, 8 * n_cells + 40 * n_pairs))
