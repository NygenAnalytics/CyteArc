from collections import Counter

import numpy as np
import pandas as pd
import pytest

from cytearc.repertoire.expansion import (
    DEFAULT_BREAKPOINTS,
    DEFAULT_FREQUENCY_BREAKPOINTS,
    DEFAULT_RANK_SPLITS,
    breakpoint_labels,
    checked_bounds,
    expansion_categories,
    expansion_tables,
    frequency_labels,
    group_clone_sizes,
    rank_window_labels,
    rank_windows,
)
from cytearc.utils.arrays import sort_categories


def test_category_names_describe_their_bounds_and_sort_naturally() -> None:
    assert breakpoint_labels(DEFAULT_BREAKPOINTS) == (
        "1 cell",
        "2 cells",
        "3 to 5 cells",
        "6 to 20 cells",
        "21 to 100 cells",
        "more than 100 cells",
    )
    assert breakpoint_labels((2, 5)) == (
        "1 to 2 cells",
        "3 to 5 cells",
        "more than 5 cells",
    )
    assert breakpoint_labels((1,)) == ("1 cell", "more than 1 cell")
    assert frequency_labels(DEFAULT_FREQUENCY_BREAKPOINTS) == (
        "0 to 0.0001",
        "0.0001 to 0.001",
        "0.001 to 0.01",
        "0.01 to 0.1",
        "0.1 to 1",
    )
    assert frequency_labels((1,)) == ("0 to 1",)
    assert rank_window_labels(DEFAULT_RANK_SPLITS) == (
        "1 to 10",
        "11 to 100",
        "101 to 1000",
        "1001 to 10000",
        "10001 to 30000",
        "30001 to 100000",
        "after 100000",
    )
    assert rank_window_labels((1, 2)) == ("1", "2", "after 2")
    # Plots and groupings order categories naturally, which keeps the names
    # of bounds in cells, and the default frequency names, from the smallest
    # clonotypes to the largest.
    for names in (
        breakpoint_labels(DEFAULT_BREAKPOINTS),
        breakpoint_labels((1, 9, 10, 99, 1000)),
        breakpoint_labels((5, 6, 7, 100)),
        frequency_labels(DEFAULT_FREQUENCY_BREAKPOINTS),
    ):
        assert tuple(sort_categories(names)) == names


@pytest.mark.parametrize(
    ("values", "whole", "error", "message"),
    [
        ("125", True, TypeError, "sequence of numbers"),
        (5, True, TypeError, "sequence of numbers"),
        ((1, True), True, TypeError, "whole numbers"),
        ((1, 2.0), True, TypeError, "whole numbers"),
        ((0.5, "1"), False, TypeError, "must hold numbers"),
        ((), True, ValueError, "at least one bound"),
        ((0, 2), True, ValueError, "positive and finite"),
        ((0.5, float("inf")), False, ValueError, "positive and finite"),
        ((0.5, float("nan")), False, ValueError, "positive and finite"),
        ((2, 2), True, ValueError, "strictly increasing"),
        ((0.5, 0.1, 1), False, ValueError, "strictly increasing"),
    ],
)
def test_bounds_are_checked(values, whole, error, message) -> None:
    with pytest.raises(error, match=message):
        checked_bounds(values, "bounds", whole=whole)


def test_bounds_become_ints_or_floats() -> None:
    assert checked_bounds([np.int64(1), 3], "bounds", whole=True) == (1, 3)
    bounds = checked_bounds((np.float32(0.5), 1), "bounds", whole=False)
    assert bounds == (0.5, 1.0)
    assert all(type(value) is float for value in bounds)
    with pytest.raises(ValueError, match="last of frequency_breakpoints must be 1"):
        frequency_labels((0.1, 0.5))


def test_clone_sizes_count_cells_within_each_group() -> None:
    clonotypes = np.array([0, 0, 0, 1, 1, 2, -1, 0, 3, 3])
    groups = np.array([0, 0, 1, 0, 0, 0, 0, 1, -1, 1])
    sizes, frequencies = group_clone_sizes(clonotypes, groups)
    # Clonotype 0 holds two cells in group 0 and two in group 1; a cell
    # without a clonotype or a group has size 0.
    assert sizes.tolist() == [2, 2, 2, 2, 2, 1, 0, 2, 0, 1]
    np.testing.assert_allclose(
        frequencies,
        [0.4, 0.4, 2 / 3, 0.4, 0.4, 0.2, np.nan, 2 / 3, np.nan, 1 / 3],
    )
    empty_sizes, empty_frequencies = group_clone_sizes(
        np.array([-1, 2]), np.array([0, -1])
    )
    assert empty_sizes.tolist() == [0, 0]
    assert np.isnan(empty_frequencies).all()
    with pytest.raises(ValueError, match="aligned one-dimensional codes"):
        group_clone_sizes(np.array([0, 1]), np.array([0]))


def test_categories_hold_the_sizes_up_to_their_bound() -> None:
    sizes = np.array([0, 1, 2, 3, 5, 6, 20, 21, 100, 101])
    frequencies = sizes / 101
    categories, labels = expansion_categories(
        sizes, frequencies, breakpoints=DEFAULT_BREAKPOINTS
    )
    assert labels == breakpoint_labels(DEFAULT_BREAKPOINTS)
    assert categories.tolist() == [-1, 0, 1, 2, 2, 3, 3, 4, 4, 5]
    # A frequency equal to a bound falls in the bin that the bound closes.
    sizes = np.array([1, 2, 3, 4, 0])
    frequencies = np.array([0.25, 0.5, 0.75, 1.0, np.nan])
    categories, labels = expansion_categories(
        sizes, frequencies, frequency_breakpoints=(0.25, 0.5, 1)
    )
    assert labels == ("0 to 0.25", "0.25 to 0.5", "0.5 to 1")
    assert categories.tolist() == [0, 1, 2, 2, -1]
    with pytest.raises(ValueError, match="exactly one of breakpoints"):
        expansion_categories(sizes, frequencies)
    with pytest.raises(ValueError, match="exactly one of breakpoints"):
        expansion_categories(
            sizes, frequencies, breakpoints=(1,), frequency_breakpoints=(1,)
        )


def test_frequency_bins_match_exact_quotients_at_their_bounds() -> None:
    # 1 of 1000 cells is exactly the 0.001 bound, as R compares it too.
    clonotypes = np.arange(1000)
    sizes, frequencies = group_clone_sizes(clonotypes, np.zeros(1000, dtype=int))
    categories, _labels = expansion_categories(
        sizes, frequencies, frequency_breakpoints=DEFAULT_FREQUENCY_BREAKPOINTS
    )
    assert set(categories.tolist()) == {1}


def test_rank_windows_count_the_cells_of_ranked_clonotypes() -> None:
    clonotypes = np.array([0, 0, 0, 1, 1, 2, -1, 0, 3, 3, 4])
    groups = np.array([0, 0, 1, 0, 0, 0, 0, 1, -1, 1, 1])
    windows = rank_windows(clonotypes, groups, 3, (1, 2))
    # Group 0 ranks clonotypes 0 and 1 (2 cells each, tied) before 2; group 1
    # ranks clonotype 0 (2 cells) before 3 and 4; group 2 is empty.
    assert windows.tolist() == [[2, 2, 1], [2, 1, 1], [0, 0, 0]]
    assert rank_windows(np.array([-1]), np.array([0]), 1, (1,)).tolist() == [[0, 0]]
    with pytest.raises(ValueError, match="strictly increasing"):
        rank_windows(clonotypes, groups, 3, (2, 1))


def _reference(clonotypes, groups, n_groups, breakpoints, splits):
    """Count with dictionaries, cell by cell."""
    counted = [
        (group, clonotype)
        for clonotype, group in zip(clonotypes.tolist(), groups.tolist(), strict=True)
        if clonotype >= 0 and group >= 0
    ]
    pair_cells = Counter(counted)
    group_cells = Counter(group for group, _clonotype in counted)
    sizes = []
    for clonotype, group in zip(clonotypes.tolist(), groups.tolist(), strict=True):
        sizes.append(
            pair_cells[(group, clonotype)] if clonotype >= 0 and group >= 0 else 0
        )
    categories = []
    for size in sizes:
        if size == 0:
            categories.append(-1)
            continue
        categories.append(
            next(
                (index for index, bound in enumerate(breakpoints) if size <= bound),
                len(breakpoints),
            )
        )
    windows = np.zeros((n_groups, len(splits) + 1), dtype=int)
    for group in range(n_groups):
        ranked = sorted(
            (cells for (g, _c), cells in pair_cells.items() if g == group),
            reverse=True,
        )
        for rank, cells in enumerate(ranked, start=1):
            window = next(
                (index for index, bound in enumerate(splits) if rank <= bound),
                len(splits),
            )
            windows[group, window] += cells
    frequencies = [
        size / group_cells[group] if size else np.nan
        for size, group in zip(sizes, groups.tolist(), strict=True)
    ]
    return sizes, frequencies, categories, windows


@pytest.mark.parametrize("seed", range(20))
def test_counts_match_a_cell_by_cell_reference(seed) -> None:
    rng = np.random.default_rng(seed)
    n_cells = int(rng.integers(1, 400))
    n_groups = int(rng.integers(1, 5))
    # Heavy-tailed clone sizes with cells missing a clonotype or a group.
    clonotypes = np.minimum(rng.zipf(1.6, size=n_cells), 60) - 1
    clonotypes[rng.random(n_cells) < 0.1] = -1
    groups = rng.integers(0, n_groups, size=n_cells)
    groups[rng.random(n_cells) < 0.05] = -1
    breakpoints = (1, 2, 5, 20)
    splits = (1, 3, 10)
    sizes, frequencies, categories, windows = _reference(
        clonotypes, groups, n_groups, breakpoints, splits
    )
    got_sizes, got_frequencies = group_clone_sizes(clonotypes, groups)
    assert got_sizes.tolist() == sizes
    np.testing.assert_allclose(got_frequencies, frequencies)
    got_categories, _labels = expansion_categories(
        got_sizes, got_frequencies, breakpoints=breakpoints
    )
    assert got_categories.tolist() == categories
    np.testing.assert_array_equal(
        rank_windows(clonotypes, groups, n_groups, splits), windows
    )
    # The windows hold every counted cell, and the categories partition them.
    assert windows.sum() == int(np.count_nonzero(got_sizes))


def test_tables_count_cells_and_clonotypes_per_group() -> None:
    clonotypes = np.array([0, 0, 0, 1, 1, 2, -1, 0, 3, 3])
    groups = np.array([0, 0, 1, 0, 0, 0, -1, 1, -1, 1])
    sizes, frequencies = group_clone_sizes(clonotypes, groups)
    categories, labels = expansion_categories(sizes, frequencies, breakpoints=(1, 2))
    cells, table, ranks = expansion_tables(
        np.asarray([f"cell{index}" for index in range(10)]),
        categories=categories,
        labels=labels,
        clonotypes=clonotypes,
        clonotype_ids=np.asarray(["c1", "c2", "c3", "c4"]),
        groups=groups,
        group_keys=[{"sample": "A"}, {"sample": "B"}],
        key_columns=["sample"],
        sizes=sizes,
        frequencies=frequencies,
        splits=(1, 2),
    )
    assert cells.index.name == "ids"
    assert cells.columns.tolist() == [
        "clonotype",
        "category",
        "group",
        "sample",
        "group_clone_size",
        "group_clone_frequency",
    ]
    assert cells.loc["cell0"].tolist()[:5] == ["c1", "2 cells", "A", "A", 2]
    # Per-cell labels are categories coded by the stored codes; the expansion
    # categories keep their order, smallest clonotypes first.
    assert cells["clonotype"].cat.categories.tolist() == ["c1", "c2", "c3", "c4"]
    assert cells["category"].cat.ordered
    assert cells["category"].cat.categories.tolist() == list(labels)
    assert cells["group"].cat.codes.tolist() == groups.tolist()
    assert pd.isna(cells.loc["cell6", "clonotype"])
    assert pd.isna(cells.loc["cell6", "category"])
    assert pd.isna(cells.loc["cell8", "sample"])
    # cell8 has a clonotype without a group, so it has no category.
    assert cells.loc["cell8", "clonotype"] == "c4"
    assert pd.isna(cells.loc["cell8", "category"])
    assert table.to_dict("list") == {
        "group": ["A", "A", "A", "B", "B", "B"],
        "sample": ["A", "A", "A", "B", "B", "B"],
        "category": [
            "1 cell",
            "2 cells",
            "more than 2 cells",
            "1 cell",
            "2 cells",
            "more than 2 cells",
        ],
        "cells": [1, 4, 0, 1, 2, 0],
        "clonotypes": [1, 2, 0, 1, 1, 0],
        "fraction": [0.2, 0.8, 0.0, 1 / 3, 2 / 3, 0.0],
    }
    assert ranks.to_dict("list") == {
        "group": ["A", "A", "A", "B", "B", "B"],
        "sample": ["A", "A", "A", "B", "B", "B"],
        "window": ["1", "2", "after 2", "1", "2", "after 2"],
        "cells": [2, 2, 1, 2, 1, 0],
        "fraction": [0.4, 0.4, 0.2, 2 / 3, 1 / 3, 0.0],
    }


def test_tables_without_clonotypes_or_key_columns() -> None:
    cells, table, ranks = expansion_tables(
        np.asarray(["cell0", "cell1"]),
        categories=np.array([-1, -1]),
        labels=("1 cell", "more than 1 cell"),
        clonotypes=np.array([-1, -1]),
        clonotype_ids=np.asarray([], dtype=str),
        groups=np.array([-1, -1]),
        group_keys=[],
        key_columns=[],
        sizes=np.zeros(2, dtype=int),
        frequencies=np.full(2, np.nan),
        splits=(10,),
    )
    assert cells.columns.tolist() == [
        "clonotype",
        "category",
        "group",
        "group_clone_size",
        "group_clone_frequency",
    ]
    assert cells["category"].isna().all()
    assert cells["group"].isna().all()
    assert table.empty and table.columns.tolist() == [
        "group",
        "category",
        "cells",
        "clonotypes",
        "fraction",
    ]
    assert ranks.empty and ranks.columns.tolist() == [
        "group",
        "window",
        "cells",
        "fraction",
    ]
    # One group without key columns, whose cells all lack a clonotype.
    _cells, table, ranks = expansion_tables(
        np.asarray(["cell0"]),
        categories=np.array([-1]),
        labels=("1 cell", "more than 1 cell"),
        clonotypes=np.array([-1]),
        clonotype_ids=np.asarray([], dtype=str),
        groups=np.array([0]),
        group_keys=[{}],
        key_columns=[],
        sizes=np.zeros(1, dtype=int),
        frequencies=np.full(1, np.nan),
        splits=(10,),
    )
    assert table["cells"].tolist() == [0, 0]
    assert table["group"].tolist() == ["all cells", "all cells"]
    assert table["fraction"].isna().all()
    assert ranks["cells"].tolist() == [0, 0]
