"""The batch design of a Harmony2 fit, held as integer level codes.

Harmony models each cell's batch membership with a one-hot design matrix that has
one row per batch level and an extra intercept row. CyteArc never builds that
matrix: every cell stores the global index of its level in each batch column, and
the cells are grouped by their combination of levels, which is all the clustering
and the regression need.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BatchDesign:
    """Batch levels of every cell in design order, grouped by level combination.

    The design order lists the levels of the first batch column, then those of
    the second, and so on. Per-level parameters, the level axis of every level
    table and the result's ``phiColumns`` share this order.

    Attributes:
        columns: Batch column names, as text.
        levels: The levels of each column, as text, in design order.
        level_offsets: ``(columns + 1,)`` int64 global index of each column's first level.
        level_codes: ``(columns, cells)`` int32 global level of each cell in each column.
        level_counts: ``(levels,)`` float64 number of cells in each level.
        level_proportions: ``(levels,)`` float64 fraction of the cells in each level.
        combination_levels: ``(combinations, columns)`` int32 global levels of each
            distinct combination of levels that the cells hold.
        cells_by_combination: ``(cells,)`` int64 cell indices sorted by combination.
        combination_starts: ``(combinations + 1,)`` int64 offsets into
            ``cells_by_combination``; the cells of combination ``q`` lie between
            entries ``q`` and ``q + 1``.
    """

    columns: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    level_offsets: np.ndarray
    level_codes: np.ndarray
    level_counts: np.ndarray
    level_proportions: np.ndarray
    combination_levels: np.ndarray
    cells_by_combination: np.ndarray
    combination_starts: np.ndarray

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return int(self.level_codes.shape[1])

    @property
    def n_columns(self) -> int:
        """Number of batch columns."""
        return int(self.level_codes.shape[0])

    @property
    def n_levels(self) -> int:
        """Number of batch levels over all columns."""
        return int(self.level_offsets[-1])

    @property
    def n_combinations(self) -> int:
        """Number of distinct level combinations among the cells."""
        return int(self.combination_levels.shape[0])

    @property
    def column_sizes(self) -> tuple[int, ...]:
        """Number of levels of each batch column."""
        return tuple(int(size) for size in np.diff(self.level_offsets))

    @property
    def largest_column(self) -> int:
        """Index of the column with the most levels, the first one on ties."""
        return int(np.argmax(np.diff(self.level_offsets)))

    @property
    def phi_columns(self) -> tuple[str, ...]:
        """One ``column_level`` name per batch level, in design order."""
        return tuple(
            f"{column}_{level}"
            for column, column_levels in zip(self.columns, self.levels, strict=True)
            for level in column_levels
        )


def check_batch_metadata(metadata: object, n_cells: int) -> pd.DataFrame:
    """Refuse metadata that is not one row per cell with unique, complete batch columns.

    Args:
        metadata: Candidate batch metadata.
        n_cells: The number of cells that the metadata must describe.

    Returns:
        ``metadata`` itself.

    Raises:
        TypeError: If ``metadata`` is not a pandas DataFrame.
        ValueError: If it has the wrong number of rows, no column, repeated
            column names, or a missing value.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "metadata must be a pandas DataFrame with one column per batch variable"
        )
    if len(metadata) != n_cells:
        raise ValueError(
            f"metadata rows ({len(metadata)}) must equal the number of cells ({n_cells})"
        )
    if metadata.shape[1] == 0:
        raise ValueError("Harmony needs at least one batch metadata column")
    names = list(metadata.columns)
    repeated = [name for name, count in Counter(names).items() if count > 1]
    if repeated:
        raise ValueError(
            "Batch metadata column names must be unique; repeated: "
            f"{sorted(repeated, key=str)}"
        )
    for position, name in enumerate(names):
        if metadata.iloc[:, position].isna().any():
            raise ValueError(f"Batch column {name!r} cannot contain missing values")
    return metadata


def _ordered_levels(column: pd.Series, name: object, out: np.ndarray) -> list[object]:
    """Write a column's local level codes into ``out`` and return its levels in order.

    A categorical column keeps the declared order of the categories that occur. Any
    other column orders its distinct values with Python's ``sorted``, so numbers
    sort numerically and text by code point.
    """
    if isinstance(column.dtype, pd.CategoricalDtype):
        used = column.cat.remove_unused_categories()
        out[...] = used.cat.codes.to_numpy()
        return list(used.cat.categories.tolist())
    codes, uniques = pd.factorize(column, sort=False)
    values = list(uniques.tolist())
    try:
        order = sorted(range(len(values)), key=values.__getitem__)
    except TypeError as error:
        raise ValueError(
            f"Batch column {name!r} mixes values that cannot be ordered; convert it "
            "to text or to an ordered categorical"
        ) from error
    rank = np.empty(len(order), dtype=np.int32)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int32)
    np.take(rank, codes, out=out, mode="clip")
    return [values[index] for index in order]


def _level_texts(levels: list[object], name: object) -> tuple[str, ...]:
    texts = tuple(str(level) for level in levels)
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            raise ValueError(
                f"Batch column {name!r} has distinct values with the same text "
                f"{text!r}; convert the column to text first"
            )
        seen.add(text)
    return texts


def encode_batch_design(metadata: pd.DataFrame) -> BatchDesign:
    """Encode batch columns as global level codes in design order and group cells by level combination.

    Unused categories of a categorical column are not levels. Two levels of one
    column may not share their text, because results name levels by text.

    Args:
        metadata: One row per cell and one column per batch variable, already
            checked by :func:`check_batch_metadata`.

    Returns:
        The encoded design.

    Raises:
        ValueError: If a column's values cannot be ordered or two of its levels
            have the same text.
    """
    n_cells = len(metadata)
    n_columns = int(metadata.shape[1])
    names = list(metadata.columns)
    level_codes = np.empty((n_columns, n_cells), dtype=np.int32)
    levels: list[tuple[str, ...]] = []
    for position, name in enumerate(names):
        ordered = _ordered_levels(
            metadata.iloc[:, position], name, level_codes[position]
        )
        levels.append(_level_texts(ordered, name))
    sizes = [len(column_levels) for column_levels in levels]
    level_offsets = np.zeros(n_columns + 1, dtype=np.int64)
    np.cumsum(sizes, out=level_offsets[1:])
    n_levels = int(level_offsets[-1])
    level_counts = np.empty(n_levels, dtype=np.float64)
    for position, size in enumerate(sizes):
        start = int(level_offsets[position])
        level_counts[start : start + size] = np.bincount(
            level_codes[position], minlength=size
        )
    level_proportions = level_counts / n_cells

    # Combinations are numbered from the local codes, before the column offsets
    # make them global. Renumbering after each column keeps every key below the
    # cell count, so the next product cannot overflow.
    key: np.ndarray
    if n_columns == 1:
        key = level_codes[0]
        n_combinations = sizes[0]
    else:
        key = level_codes[0].astype(np.int64)
        for position in range(1, n_columns):
            key *= sizes[position]
            key += level_codes[position]
            renumbered, _uniques = pd.factorize(key, sort=False)
            del _uniques
            key = renumbered
        n_combinations = int(key.max()) + 1
    for position in range(1, n_columns):
        level_codes[position] += int(level_offsets[position])
    if n_columns == 1:
        combination_levels = np.arange(n_combinations, dtype=np.int32).reshape(-1, 1)
    else:
        combination_levels = np.empty((n_combinations, n_columns), dtype=np.int32)
        combination_levels[key] = level_codes.T
    cells_by_combination = np.argsort(key, kind="stable").astype(np.int64, copy=False)
    combination_starts = np.zeros(n_combinations + 1, dtype=np.int64)
    np.cumsum(np.bincount(key, minlength=n_combinations), out=combination_starts[1:])
    del key
    return BatchDesign(
        columns=tuple(str(name) for name in names),
        levels=tuple(levels),
        level_offsets=level_offsets,
        level_codes=level_codes,
        level_counts=level_counts,
        level_proportions=level_proportions,
        combination_levels=combination_levels,
        cells_by_combination=cells_by_combination,
        combination_starts=combination_starts,
    )
