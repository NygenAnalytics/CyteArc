"""Clonal expansion: how many cells each clonotype holds within a group.

A clonotype that holds more than one cell has expanded: its founding cell
divided. Expansion is measured within groups of cells, such as samples, since
a clonotype large in one sample may be small in another. A cell's clone size
in its group places it in an expansion category, either by bounds on the
number of cells, as scirpy's ``clonal_expansion`` does with ``breakpoints``,
or by bounds on the clonotype's frequency, its share of the group's cells with
a clonotype, as scRepertoire's ``clonalHomeostasis`` does. The clonal space of
a group is the share of its cells in each category, and the share of its
cells held by its clonotypes ranked into windows by size, such as the ten
largest, as scRepertoire's ``clonalProportion`` reports.

These functions work on clonotype and group codes and never touch storage.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef

__all__ = [
    "DEFAULT_BREAKPOINTS",
    "DEFAULT_FREQUENCY_BREAKPOINTS",
    "DEFAULT_RANK_SPLITS",
    "ClonalExpansion",
    "breakpoint_labels",
    "expansion_categories",
    "expansion_tables",
    "frequency_labels",
    "group_clone_sizes",
    "rank_window_labels",
    "rank_windows",
]

#: The default bounds of the expansion categories in cells: clonotypes of 1,
#: 2, 3 to 5, 6 to 20, 21 to 100, and more than 100 cells.
DEFAULT_BREAKPOINTS = (1, 2, 5, 20, 100)
#: The bounds of scRepertoire's frequency categories, which it names Rare,
#: Small, Medium, Large and Hyperexpanded.
DEFAULT_FREQUENCY_BREAKPOINTS = (0.0001, 0.001, 0.01, 0.1, 1.0)
#: The rank windows of scRepertoire's ``clonalProportion``: the 10 largest
#: clonotypes, 11 to 100, 101 to 1000, 1001 to 10000, 10001 to 30000, and
#: 30001 to 100000, and here also the rest.
DEFAULT_RANK_SPLITS = (10, 100, 1_000, 10_000, 30_000, 100_000)


def checked_bounds(values: Any, name: str, *, whole: bool) -> tuple[Any, ...]:
    """Return positive, strictly increasing bounds as ints or floats.

    Raises:
        TypeError: If ``values`` is not a sequence of numbers, or of whole
            numbers when ``whole``.
        ValueError: If it is empty, or not positive and strictly increasing.
    """
    kind = Integral if whole else Real
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    if any(isinstance(value, bool) or not isinstance(value, kind) for value in values):
        raise TypeError(f"{name} must hold {'whole numbers' if whole else 'numbers'}")
    bounds = tuple(int(value) if whole else float(value) for value in values)
    if not bounds:
        raise ValueError(f"{name} must hold at least one bound")
    if not all(np.isfinite(bounds)) or bounds[0] <= 0:
        raise ValueError(f"{name} must be positive and finite")
    if any(later <= earlier for earlier, later in pairwise(bounds)):
        raise ValueError(f"{name} must be strictly increasing")
    return bounds


def _whole_ranges(bounds: tuple[int, ...]) -> list[str]:
    """Name the ranges of whole numbers that end at each bound."""
    labels = []
    lower = 1
    for upper in bounds:
        labels.append(str(upper) if upper == lower else f"{lower} to {upper}")
        lower = upper + 1
    return labels


def breakpoint_labels(breakpoints: Sequence[int]) -> tuple[str, ...]:
    """Name the categories that bounds in cells make, such as ``"3 to 5 cells"``.

    A category holds the clonotypes with at most its bound of cells and more
    than the bound before it; the last, ``"more than"`` the last bound, holds
    the larger ones. Every name starts with a number but the last and none is
    a bare number, so the names sort in this order naturally, as plots and
    groupings order labels.
    """
    bounds = checked_bounds(breakpoints, "breakpoints", whole=True)
    return (
        *(
            f"{name} {'cell' if name == '1' else 'cells'}"
            for name in _whole_ranges(bounds)
        ),
        f"more than {bounds[-1]} {'cell' if bounds[-1] == 1 else 'cells'}",
    )


def frequency_labels(frequency_breakpoints: Sequence[float]) -> tuple[str, ...]:
    """Name the categories that frequency bounds make, such as ``"0.001 to 0.01"``.

    A category holds the clonotypes whose frequency is at most its bound and
    more than the bound before it, as scRepertoire's bins ``(lower, upper]``
    do; the first starts at 0. The last bound must be 1, so every clonotype
    has a category.
    """
    bounds = checked_bounds(frequency_breakpoints, "frequency_breakpoints", whole=False)
    if bounds[-1] != 1.0:
        raise ValueError("The last of frequency_breakpoints must be 1")
    lowers = (0.0, *bounds[:-1])
    return tuple(
        f"{lower:g} to {upper:g}" for lower, upper in zip(lowers, bounds, strict=True)
    )


def rank_window_labels(splits: Sequence[int]) -> tuple[str, ...]:
    """Name the rank windows that splits make, such as ``"11 to 100"``."""
    bounds = checked_bounds(splits, "splits", whole=True)
    return (*_whole_ranges(bounds), f"after {bounds[-1]}")


def _pair_codes(clonotypes: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Number each counted cell's pair of group and clonotype, densely."""
    pairs = groups * (int(clonotypes.max()) + 1) + clonotypes
    codes, _uniques = pd.factorize(pairs)
    return np.asarray(codes, dtype=np.int64)


def group_clone_sizes(
    clonotypes: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Count the cells of each cell's clonotype within the cell's group.

    Args:
        clonotypes: Each cell's clonotype code, -1 without one.
        groups: Each cell's group code, -1 without one.

    Returns:
        Each cell's clone size in its group, 0 for a cell without a
        clonotype or a group, and its clone frequency, the size as a share of
        the group's cells with a clonotype, NaN for those cells.
    """
    clonotypes = np.asarray(clonotypes, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if clonotypes.shape != groups.shape or clonotypes.ndim != 1:
        raise ValueError("clonotypes and groups must be aligned one-dimensional codes")
    counted = (clonotypes >= 0) & (groups >= 0)
    sizes = np.zeros(len(clonotypes), dtype=np.int64)
    frequencies = np.full(len(clonotypes), np.nan, dtype=np.float64)
    if counted.any():
        codes = _pair_codes(clonotypes[counted], groups[counted])
        sizes[counted] = np.bincount(codes)[codes]
        totals = np.bincount(groups[counted])
        frequencies[counted] = sizes[counted] / totals[groups[counted]]
    return sizes, frequencies


def expansion_categories(
    sizes: np.ndarray,
    frequencies: np.ndarray,
    *,
    breakpoints: Sequence[int] | None = None,
    frequency_breakpoints: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Place each cell in an expansion category.

    Exactly one of ``breakpoints``, bounds in cells, and
    ``frequency_breakpoints``, bounds on the clone frequency, sets the
    categories. A cell's category is the first whose bound its clonotype does
    not exceed.

    Args:
        sizes: Each cell's clone size in its group, 0 without a clonotype.
        frequencies: Each cell's clone frequency in its group.
        breakpoints: Bounds in cells.
        frequency_breakpoints: Bounds on the frequency, the last 1.

    Returns:
        Each cell's category, a position in the names, -1 for a cell without
        a clonotype, and the names of the categories, smallest first.

    Raises:
        ValueError: If both or neither bounds are given, or they are invalid.
    """
    if (breakpoints is None) == (frequency_breakpoints is None):
        raise ValueError("Pass exactly one of breakpoints and frequency_breakpoints")
    sizes = np.asarray(sizes, dtype=np.int64)
    counted = sizes > 0
    categories = np.full(len(sizes), -1, dtype=np.int64)
    if breakpoints is not None:
        labels = breakpoint_labels(breakpoints)
        bounds: np.ndarray = np.asarray(breakpoints, dtype=np.int64)
        measured = sizes[counted]
    else:
        assert frequency_breakpoints is not None
        labels = frequency_labels(frequency_breakpoints)
        bounds = np.asarray(frequency_breakpoints, dtype=np.float64)
        measured = np.asarray(frequencies, dtype=np.float64)[counted]
    # side="left" finds the first bound that the value does not exceed, so a
    # category holds the values in (previous bound, its bound].
    categories[counted] = np.searchsorted(bounds, measured, side="left")
    return categories, labels


def rank_windows(
    clonotypes: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    splits: Sequence[int],
) -> np.ndarray:
    """Count the cells that each group's clonotypes hold, by rank window.

    Within a group, its clonotypes are ranked by their cells in the group,
    largest first. The first window holds the cells of the clonotypes ranked
    1 to ``splits[0]``, and the last those ranked after ``splits[-1]``. Ties
    do not change the counts, since tied clonotypes hold the same cells.

    Args:
        clonotypes: Each cell's clonotype code, -1 without one.
        groups: Each cell's group code, -1 without one.
        n_groups: The number of groups.
        splits: The last rank of each window but the last.

    Returns:
        The cells of each group, one row per group, in each window, one
        column per window.
    """
    bounds = np.asarray(checked_bounds(splits, "splits", whole=True), dtype=np.int64)
    clonotypes = np.asarray(clonotypes, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    windows = np.zeros((n_groups, len(bounds) + 1), dtype=np.int64)
    counted = (clonotypes >= 0) & (groups >= 0)
    if not counted.any():
        return windows
    codes = _pair_codes(clonotypes[counted], groups[counted])
    cells = np.bincount(codes)
    pair_group = np.empty(len(cells), dtype=np.int64)
    pair_group[codes] = groups[counted]
    # Within each group, the largest clonotypes first.
    order = np.lexsort((-cells, pair_group))
    ordered_groups = pair_group[order]
    first = np.searchsorted(ordered_groups, np.arange(n_groups), side="left")
    rank = np.arange(len(order)) - first[ordered_groups] + 1
    window = np.searchsorted(bounds, rank, side="left")
    np.add.at(windows, (ordered_groups, window), cells[order])
    return windows


@dataclass(frozen=True, slots=True)
class ClonalExpansion:
    """The clonal expansion of a set of cells, from a ``clonal_expansion`` artifact.

    The groups are the combinations of the ``within`` columns' labels and,
    with ``within_receptor_type``, of the receptor type; these columns are
    each group's key in the tables.

    Attributes:
        cells: One row per cell, indexed by cell ``ids``: ``clonotype``,
            ``category`` (its expansion category, NA without a clonotype),
            the group's key,
            ``group_clone_size`` (the cells of its clonotype in its group, 0
            without one) and ``group_clone_frequency`` (their share of the
            group's cells with a clonotype, NaN without one).
        categories: One row per group and category, every category of every
            group: the group's key, ``category``, ``cells``, ``clonotypes``,
            and ``fraction``, the share of the group's cells with a clonotype
            that the category holds.
        rank_windows: One row per group and rank window: the group's key,
            ``window``, the ``cells`` that the group's clonotypes ranked in the
            window hold, and their ``fraction`` of the group's cells.
        clonotypes: The ``clonotypes`` artifact.
        labels: The categories, smallest clonotypes first.
        breakpoints: The bounds in cells, or None with frequency bounds.
        frequency_breakpoints: The frequency bounds, or None with bounds in
            cells.
        splits: The last rank of each window but the last.
        within: The cell metadata columns of the groups.
        within_receptor_type: Whether each receptor type is a group apart.
    """

    cells: pd.DataFrame = field(repr=False)
    categories: pd.DataFrame = field(repr=False)
    rank_windows: pd.DataFrame = field(repr=False)
    clonotypes: ArtifactRef
    labels: tuple[str, ...]
    breakpoints: tuple[int, ...] | None
    frequency_breakpoints: tuple[float, ...] | None
    splits: tuple[int, ...]
    within: tuple[str, ...]
    within_receptor_type: bool


def _shares(counts: np.ndarray) -> np.ndarray:
    """Divide each row by its total, NaN for a row without counts."""
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(counts, totals, out=shares, where=totals > 0)
    return shares


def expansion_tables(
    cell_ids: np.ndarray,
    *,
    categories: np.ndarray,
    labels: Sequence[str],
    clonotypes: np.ndarray,
    clonotype_ids: np.ndarray,
    groups: np.ndarray,
    group_keys: Sequence[dict[str, str]],
    key_columns: Sequence[str],
    sizes: np.ndarray,
    frequencies: np.ndarray,
    splits: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the tables of cells, categories and rank windows.

    Args:
        cell_ids: The ``ids`` of the cells.
        categories: Each cell's category, a position in ``labels``, -1 without
            one.
        labels: The names of the categories.
        clonotypes: Each cell's clonotype code, -1 without one.
        clonotype_ids: The label of each clonotype code.
        groups: Each cell's group, a position in ``group_keys``, -1 without
            one.
        group_keys: Each group's label in each key column.
        key_columns: The columns of the groups' keys.
        sizes: Each cell's clone size in its group.
        frequencies: Each cell's clone frequency in its group.
        splits: The last rank of each window but the last.
    """
    categories = np.asarray(categories, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    clonotypes = np.asarray(clonotypes, dtype=np.int64)
    names = np.asarray(list(labels), dtype=object)
    n_groups, n_categories = len(group_keys), len(names)
    counted = categories >= 0
    keys = {
        column: np.asarray([key[column] for key in group_keys], dtype=object)
        for column in key_columns
    }
    cell_columns: dict[str, Any] = {}
    cell_clonotype = np.full(len(clonotypes), None, dtype=object)
    has = clonotypes >= 0
    cell_clonotype[has] = np.asarray(clonotype_ids, dtype=object)[clonotypes[has]]
    cell_columns["clonotype"] = pd.array(cell_clonotype, dtype="string")
    cell_category = np.full(len(categories), None, dtype=object)
    cell_category[counted] = names[categories[counted]]
    cell_columns["category"] = pd.array(cell_category, dtype="string")
    grouped = groups >= 0
    for column, column_keys in keys.items():
        values = np.full(len(groups), None, dtype=object)
        values[grouped] = column_keys[groups[grouped]]
        cell_columns[column] = pd.array(values, dtype="string")
    cell_columns["group_clone_size"] = np.asarray(sizes, dtype=np.int64)
    cell_columns["group_clone_frequency"] = np.asarray(frequencies, dtype=np.float64)
    cells = pd.DataFrame(cell_columns, index=pd.Index(np.asarray(cell_ids), name="ids"))

    cell_counts = np.zeros((n_groups, n_categories), dtype=np.int64)
    np.add.at(cell_counts, (groups[counted], categories[counted]), 1)
    # A clonotype has one size in a group, so one category there.
    pairs = np.unique(
        np.column_stack([groups[counted], clonotypes[counted], categories[counted]]),
        axis=0,
    )
    clonotype_counts = np.zeros((n_groups, n_categories), dtype=np.int64)
    np.add.at(clonotype_counts, (pairs[:, 0], pairs[:, 2]), 1)
    table = pd.DataFrame(
        {
            **{
                column: pd.array(np.repeat(column_keys, n_categories), dtype="string")
                for column, column_keys in keys.items()
            },
            "category": pd.array(np.tile(names, n_groups), dtype="string"),
            "cells": cell_counts.reshape(-1),
            "clonotypes": clonotype_counts.reshape(-1),
            "fraction": _shares(cell_counts).reshape(-1),
        }
    )
    windows = rank_windows(clonotypes, np.where(counted, groups, -1), n_groups, splits)
    window_names = np.asarray(rank_window_labels(splits), dtype=object)
    ranks = pd.DataFrame(
        {
            **{
                column: pd.array(
                    np.repeat(column_keys, len(window_names)), dtype="string"
                )
                for column, column_keys in keys.items()
            },
            "window": pd.array(np.tile(window_names, n_groups), dtype="string"),
            "cells": windows.reshape(-1),
            "fraction": _shares(windows).reshape(-1),
        }
    )
    return cells, table, ranks
