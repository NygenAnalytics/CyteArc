"""Benchmarks of the clone summaries: expansion, diversity and rarefaction.

Clonal expansion counts each clone's cells within samples, and diversity and
rarefaction summarize each group's clone sizes. Each benchmark runs one step
on simulated clones whose sizes follow a heavy-tailed draw, so a few clones
hold many cells and most cells hold a clone of their own, with one cell in ten
without a clone, and checks the result, so it doubles as a test at the smoke
size. The summaries of groups, such as samples by clusters, are measured over
the number of groups as well, on simulated groups of about 150 cells.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

LADDER = Ladder(sizes=(20_000, 40_000, 80_000, 160_000), smoke=500)
# Groups such as samples by clusters: a few thousand in a large study.
GROUPS = Ladder(
    sizes=(125, 250, 500, 1_000), smoke=8, unit="groups", targets=(2_000, 10_000)
)


@lru_cache(maxsize=1)
def _clone_codes(n_cells: int) -> np.ndarray:
    """Simulate each cell's clone, -1 for one cell in ten.

    Clone sizes are drawn from a Zipf law with exponent 2.2, capped at 500
    cells, as the clonotypes of a T cell repertoire spread.
    """
    rng = np.random.default_rng([inputs.SEED, n_cells])
    sizes = np.minimum(rng.zipf(2.2, size=n_cells), 500)
    codes = np.repeat(np.arange(n_cells), sizes)[:n_cells]
    codes[rng.random(n_cells) < 0.1] = -1
    return codes


def _samples(n_cells: int) -> np.ndarray:
    """Spread the cells over eight samples."""
    return np.random.default_rng([inputs.SEED, n_cells, 8]).integers(0, 8, n_cells)


def test_clonal_expansion(bench) -> None:
    from cytearc.clones.expansion import (
        DEFAULT_BREAKPOINTS,
        DEFAULT_RANK_SPLITS,
        expansion_categories,
        group_clone_sizes,
        rank_windows,
    )

    def make(n_cells: int):
        codes = _clone_codes(n_cells)
        groups = np.where(codes >= 0, _samples(n_cells), -1)

        def run():
            sizes, frequencies = group_clone_sizes(codes, groups)
            categories, _labels = expansion_categories(
                sizes, frequencies, breakpoints=DEFAULT_BREAKPOINTS
            )
            windows = rank_windows(codes, groups, 8, DEFAULT_RANK_SPLITS)
            return sizes, categories, windows

        return run

    def check(n_cells: int, result) -> None:
        sizes, categories, windows = result
        codes = _clone_codes(n_cells)
        has = codes >= 0
        frame = pd.DataFrame({"sample": _samples(n_cells)[has], "code": codes[has]})
        expected = frame.groupby(["sample", "code"])["code"].transform("size")
        np.testing.assert_array_equal(sizes[has], expected.to_numpy())
        assert (sizes[~has] == 0).all() and (categories[~has] == -1).all()
        assert (categories[has] >= 0).all()
        np.testing.assert_array_equal(
            windows.sum(axis=1), np.bincount(frame["sample"], minlength=8)
        )

    bench("clones.clonal_expansion", make, LADDER, check=check)


def test_rarefaction_curve(bench) -> None:
    from cytearc.clones.diversity import (
        CloneSizes,
        default_sizes,
        hill_curve,
        indices,
    )

    def make(n_cells: int):
        codes = _clone_codes(n_cells)
        clones = CloneSizes.from_sizes(np.bincount(codes[codes >= 0]))
        sizes = default_sizes(clones.n).tolist()

        def run():
            curves = [hill_curve(clones, sizes, order) for order in (0, 1, 2)]
            return clones, indices(clones), curves

        return run

    def check(n_cells: int, result) -> None:
        clones, table, curves = result
        codes = _clone_codes(n_cells)
        assert clones.n == int((codes >= 0).sum())
        assert table["clones"] == len(np.unique(codes[codes >= 0]))
        for values, methods in curves:
            observed = methods.index("observed")
            # Rarefied curves rise to the observed value.
            assert np.all(np.diff(values[: observed + 1]) >= -1e-9)
        richness, methods = curves[0]
        assert richness[methods.index("observed")] == clones.richness

    bench("clones.rarefaction_curve", make, LADDER, check=check)


@lru_cache(maxsize=1)
def _group_pairs(n_groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the clones of ``n_groups`` groups of about 150 cells each.

    A clone's cells fall in its home group with probability 0.7 and
    otherwise in any group, so groups share clones.

    Returns:
        Each distinct pair of group and clone's group, clone code and cells,
        sorted by group and clone.
    """
    from cytearc.clones.groups import pair_counts

    rng = np.random.default_rng([inputs.SEED, n_groups, 2])
    n_cells = 150 * n_groups
    sizes = np.minimum(rng.zipf(2.0, size=n_cells), 300)
    clone_of_cell = np.repeat(np.arange(n_cells), sizes)[:n_cells]
    n_clones = int(clone_of_cell.max()) + 1
    home = rng.integers(0, n_groups, size=n_clones)
    stray = rng.integers(0, n_groups, size=n_cells)
    groups = np.where(rng.random(n_cells) < 0.7, home[clone_of_cell], stray)
    return pair_counts(groups, clone_of_cell)


@lru_cache(maxsize=1)
def _group_clone_sizes(n_groups: int):
    from cytearc.clones.diversity import CloneSizes

    pair_groups, _codes, pair_cells = _group_pairs(n_groups)
    bounds = np.searchsorted(pair_groups, np.arange(n_groups + 1))
    groups = [
        CloneSizes.from_sizes(pair_cells[bounds[at] : bounds[at + 1]])
        for at in range(n_groups)
    ]
    keys = pd.DataFrame(
        {
            "group": [f"g{at:05d}" for at in range(n_groups)],
            "sample": [f"g{at:05d}" for at in range(n_groups)],
        }
    )
    return groups, keys


def test_diversity_tables(bench) -> None:
    from cytearc.clones.diversity import (
        ORDERS,
        coverage_at,
        diversity_at,
        diversity_tables,
        indices,
    )

    def make(n_groups: int):
        groups, keys = _group_clone_sizes(n_groups)
        return lambda: diversity_tables(groups, keys)

    def check(n_groups: int, result) -> None:
        groups, keys = _group_clone_sizes(n_groups)
        table, hill, size, level = result
        assert table["group"].tolist() == keys["group"].tolist()
        assert table["cells"].tolist() == [clones.n for clones in groups]
        assert size == min(2 * clones.n for clones in groups)
        assert 0 < level < 1
        assert len(hill) == len(ORDERS) * n_groups
        # Spot-check groups against the scalar functions.
        for at in (0, n_groups - 1):
            clones = groups[at]
            expected = indices(clones)
            for column in ("clones", "singletons", "d50"):
                assert table[column].iat[at] == expected[column]
            rows = hill.iloc[len(ORDERS) * at : len(ORDERS) * (at + 1)]
            for order, value in zip(ORDERS, rows["at_size"], strict=True):
                assert value == pytest.approx(
                    diversity_at(clones, size, order)[0], rel=1e-12
                )
            assert rows["coverage_at_size"].iat[0] == pytest.approx(
                coverage_at(clones, size), rel=1e-12
            )
            reached = coverage_at(clones, rows["size_at_coverage"].iat[0])
            assert reached == pytest.approx(level, rel=1e-9)

    bench("clones.diversity_tables", make, GROUPS, check=check)


def test_rarefaction_curves(bench) -> None:
    from cytearc.clones.diversity import (
        default_sizes,
        observed_hill,
        rarefaction_curves,
    )

    def make(n_groups: int):
        groups, keys = _group_clone_sizes(n_groups)
        return lambda: rarefaction_curves(groups, keys)

    def check(n_groups: int, curves) -> None:
        groups, keys = _group_clone_sizes(n_groups)
        lengths = [3 * len(default_sizes(clones.n)) for clones in groups]
        assert len(curves) == sum(lengths)
        observed = curves[curves["method"] == "observed"]
        assert len(observed) == 3 * n_groups
        for at in (0, n_groups - 1):
            rows = observed[observed["group"] == keys["group"].iat[at]]
            for order, value in zip(rows["order"], rows["diversity"], strict=True):
                assert value == pytest.approx(
                    observed_hill(groups[at], int(order)), rel=1e-12
                )
        # Rarefied curves rise to the observed value, and coverage with them.
        rarefied = curves[curves["method"] != "extrapolation"]
        for _key, rows in rarefied.groupby(["group", "order"], sort=False):
            assert np.all(np.diff(rows["diversity"].to_numpy()) >= -1e-9)
            assert np.all(np.diff(rows["coverage"].to_numpy()) >= -1e-12)

    bench("clones.rarefaction_curves", make, GROUPS, check=check)
