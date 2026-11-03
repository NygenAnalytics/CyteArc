"""Allele frequencies, masked cosine neighbourhoods, the shared-neighbour graph and
the clones they give, against values derived by hand."""

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from cytearc.neighbors.graph import shared_neighbor_graph
from cytearc.variants.counts import Contig, SiteTable
from cytearc.variants.frequencies import (
    CLONE_STATUSES,
    band_frequencies,
    clone_status_counts,
    cosine_neighborhoods,
    frequency_clones,
)
from tests.memory_helpers import smallest_budget, traced_peak


def _sites() -> SiteTable:
    """Two sites: position 5 (A, with C and G) and position 9 (T, with C)."""
    return SiteTable(
        contigs=(Contig("chrM", 20, None),),
        site_contig=np.asarray([0, 0]),
        site_position=np.asarray([5, 9]),
        site_name=np.asarray(["", ""], dtype=object),
        site_channel=np.asarray(["", ""], dtype=object),
        allele_site=np.asarray([0, 0, 0, 1, 1]),
        allele=np.asarray(["A", "C", "G", "T", "C"], dtype=object),
        allele_role=np.asarray(
            ["reference", "alternative", "alternative", "reference", "alternative"],
            dtype=object,
        ),
    )


def test_band_frequencies_leave_uncovered_cells_undefined() -> None:
    # Cell 0: depth 10 at 5 (4 C), depth 4 at 9 (1 C). Cell 1: depth 8 at 5
    # with no C, not covered at 9. Cell 2: not covered anywhere.
    frequencies = band_frequencies(
        n_rows=3,
        depth_rows=np.asarray([0, 0, 1]),
        depth_sites=np.asarray([0, 1, 0]),
        depth=np.asarray([10, 4, 8]),
        count_rows=np.asarray([0, 0, 0, 1]),
        count_alleles=np.asarray([0, 1, 4, 0]),
        counts=np.asarray([6, 4, 1, 8]),
        alleles=np.asarray([1, 2, 4]),
        sites=_sites(),
    )
    expected = np.asarray(
        [[0.4, 0.0, 0.25], [0.0, 0.0, np.nan], [np.nan, np.nan, np.nan]]
    )
    np.testing.assert_array_equal(np.isnan(frequencies), np.isnan(expected))
    np.testing.assert_allclose(frequencies, expected)


@pytest.mark.parametrize(
    ("depth", "counts", "message"),
    [
        ([10], [4, 4], "no depth at its site"),
        ([10, 4], [4, 11], "exceeds the depth"),
    ],
)
def test_band_frequencies_refuse_counts_beyond_depth(depth, counts, message) -> None:
    with pytest.raises(ValueError, match=message):
        band_frequencies(
            n_rows=1,
            depth_rows=np.zeros(len(depth), dtype=int),
            depth_sites=np.arange(len(depth)),
            depth=np.asarray(depth),
            count_rows=np.zeros(2, dtype=int),
            count_alleles=np.asarray([1, 4]),
            counts=np.asarray(counts),
            alleles=np.asarray([1, 4]),
            sites=_sites(),
        )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """The cosine over the variants both cover, 0 without shared signal."""
    shared = ~np.isnan(left) & ~np.isnan(right)
    a, b = left[shared], right[shared]
    norm = np.sqrt((a * a).sum() * (b * b).sum())
    return float(a @ b / norm) if norm > 0 else 0.0


def test_neighbourhoods_rank_cells_by_cosine_over_shared_coverage() -> None:
    values = np.sqrt(
        np.asarray(
            [
                [0.8, 0.0, np.nan],
                [0.6, 0.0, 0.0],  # like cell 0 on the variants both cover
                [np.nan, 0.5, 0.5],  # shares no signal with cell 0
                [0.0, 0.0, 0.9],
                [0.4, 0.4, np.nan],
            ]
        )
    )
    neighborhoods = cosine_neighborhoods(values, n_neighbors=3, block_rows=2)
    assert neighborhoods[:, 0].tolist() == [0, 1, 2, 3, 4]
    for cell in range(5):
        similarities = {
            other: _cosine(values[cell], values[other])
            for other in range(5)
            if other != cell
        }
        ranked = sorted(
            (other for other, value in similarities.items() if value > 0),
            key=lambda other: (-similarities[other], other),
        )[:2]
        expected = ranked + [-1] * (2 - len(ranked))
        assert neighborhoods[cell, 1:].tolist() == expected, cell
    # Cell 0's best match is cell 1 (cosine 1 over the variants both cover).
    assert neighborhoods[0, 1] == 1


def test_neighbourhoods_break_ties_by_position_whatever_the_blocks() -> None:
    # Six identical cells: each takes the lowest positions as neighbours.
    values = np.full((6, 2), 0.5)
    for block_rows in (1, 4, 6):
        neighborhoods = cosine_neighborhoods(
            values, n_neighbors=3, block_rows=block_rows
        )
        assert neighborhoods.tolist() == [
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1],
            [3, 0, 1],
            [4, 0, 1],
            [5, 0, 1],
        ]


def test_neighbourhoods_of_small_inputs_and_bad_arguments() -> None:
    assert cosine_neighborhoods(
        np.ones((1, 2)), n_neighbors=10, block_rows=5
    ).tolist() == [[0] + [-1] * 9]
    with pytest.raises(ValueError, match="matrix of cells by variants"):
        cosine_neighborhoods(np.ones(3), n_neighbors=2, block_rows=1)
    with pytest.raises(ValueError, match="non-negative"):
        cosine_neighborhoods(-np.ones((2, 2)), n_neighbors=2, block_rows=1)
    with pytest.raises(ValueError, match="at least 1"):
        cosine_neighborhoods(np.ones((2, 2)), n_neighbors=0, block_rows=1)


def _jaccard_by_sets(neighborhoods: np.ndarray) -> dict[tuple[int, int], float]:
    """The Jaccard index of every pair of cells that share a neighbour."""
    sets = [set(row[row >= 0].tolist()) for row in neighborhoods]
    pairs = {}
    for first in range(len(sets)):
        for second in range(first + 1, len(sets)):
            shared = len(sets[first] & sets[second])
            if shared:
                pairs[(first, second)] = shared / len(sets[first] | sets[second])
    return pairs


def test_shared_neighbour_graph_weighs_the_jaccard_index_of_neighbourhoods() -> None:
    rng = np.random.default_rng(3)
    neighborhoods = np.stack(
        [
            np.concatenate(
                [
                    [cell],
                    rng.choice([c for c in range(30) if c != cell], 4, replace=False),
                ]
            )
            for cell in range(30)
        ]
    )
    neighborhoods[7, 3:] = -1  # a cell with two neighbours
    graph = shared_neighbor_graph(neighborhoods, prune=0.0)
    expected = _jaccard_by_sets(neighborhoods)
    found = {
        (int(row), int(column)): float(value)
        for row, column, value in zip(graph.row, graph.col, graph.data, strict=True)
    }
    assert found.keys() == expected.keys()
    for pair, value in expected.items():
        assert found[pair] == pytest.approx(value)
    # Seurat drops joins below 1/15: shared / (2k - shared) of full rows.
    pruned = shared_neighbor_graph(neighborhoods)
    assert np.all(pruned.data >= 1 / 15)
    assert len(pruned.data) == sum(value >= 1 / 15 for value in expected.values())
    # The joins come sorted by row and then column.
    keys = graph.row.astype(np.int64) * 30 + graph.col
    assert np.all(np.diff(keys) > 0)
    # Blocks of cells change nothing.
    blocked = shared_neighbor_graph(neighborhoods, prune=0.0, max_entries=40)
    np.testing.assert_array_equal(blocked.row, graph.row)
    np.testing.assert_array_equal(blocked.col, graph.col)
    np.testing.assert_array_equal(blocked.data, graph.data)
    # A graph of more joins than max_joins is refused; one of as many is not.
    exact = shared_neighbor_graph(
        neighborhoods, prune=0.0, max_joins=len(expected), max_entries=40
    )
    assert len(exact.data) == len(expected)
    with pytest.raises(MemoryError, match="holds more than the 99 joins"):
        shared_neighbor_graph(neighborhoods, prune=0.0, max_joins=99, max_entries=40)
    with pytest.raises(MemoryError, match="may share neighbours with"):
        shared_neighbor_graph(neighborhoods, max_entries=3)


def test_a_jaccard_index_of_one_fifteenth_is_kept_and_less_is_dropped() -> None:
    # Cell 0's ten cells share one with cell 10's six: 1 / (10 + 6 - 1) is
    # 1/15, which Seurat keeps. Cell 20's seven share one with cell 0: 1/16.
    neighborhoods = np.full((26, 10), -1)
    for cell in range(26):
        neighborhoods[cell, 0] = cell
    neighborhoods[0] = np.arange(10)
    neighborhoods[10, :6] = [10, 1, 11, 12, 13, 14]
    neighborhoods[20, :7] = [20, 2, 21, 22, 23, 24, 25]
    graph = shared_neighbor_graph(neighborhoods)
    weights = {
        (int(row), int(column)): float(value)
        for row, column, value in zip(graph.row, graph.col, graph.data, strict=True)
    }
    assert weights[(0, 10)] == 1 / 15
    assert (0, 20) not in weights
    assert (0, 20) in {
        (int(row), int(column))
        for row, column in zip(
            *shared_neighbor_graph(neighborhoods, prune=1 / 16).nonzero(), strict=True
        )
    }


def test_cells_that_tie_join_every_other_such_cell_as_seurat_joins_them() -> None:
    # 200 identical cells take the lowest positions as neighbours, so every
    # pair shares them and Seurat's graph joins every pair: 9/11 between two
    # cells beyond the first ten, which share nine of eleven cells.
    values = np.full((200, 3), 0.5)
    neighborhoods = cosine_neighborhoods(values, n_neighbors=10, block_rows=50)
    graph = shared_neighbor_graph(neighborhoods)
    assert len(graph.data) == 200 * 199 // 2
    beyond = (graph.row >= 10) & (graph.col >= 10)
    np.testing.assert_allclose(graph.data[beyond], 9 / 11)
    called = frequency_clones(
        values**2, n_neighbors=10, resolution=1.0, random_seed=1, memory_bytes=1 << 30
    )
    assert called.n_edges == 200 * 199 // 2
    assert np.all(called.clusters > 0)
    assert clone_status_counts(called.status)["clustered"] == 200
    # A budget that holds the similarities but not the 19,900 joins fails
    # before any clone is called, and says why.
    with pytest.raises(MemoryError, match="Cells that tie") as raised:
        frequency_clones(
            values**2,
            n_neighbors=10,
            resolution=1.0,
            random_seed=1,
            memory_bytes=1_000_000,
        )
    assert "holds more than" in str(raised.value)


def test_shared_neighbour_graph_refuses_malformed_neighbourhoods() -> None:
    with pytest.raises(ValueError, match="integer matrix"):
        shared_neighbor_graph(np.ones((2, 2), dtype=float))
    with pytest.raises(ValueError, match="outside 0"):
        shared_neighbor_graph(np.asarray([[0, 5], [1, 0]]))
    with pytest.raises(ValueError, match="lists a cell twice"):
        shared_neighbor_graph(np.asarray([[0, 1, 1], [1, 0, -1]]))
    with pytest.raises(ValueError, match="prune must be"):
        shared_neighbor_graph(np.asarray([[0, 1], [1, 0]]), prune=2.0)


def _two_clones(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Forty cells of two clones, each marked by a variant of its own at a
    frequency from 0.3 to 0.9; the other two variants are read without an
    alternative allele."""
    truth = np.repeat([0, 1], 20)
    frequencies = np.zeros((40, 4))
    frequencies[truth == 0, 0] = rng.uniform(0.3, 0.9, size=20)
    frequencies[truth == 1, 2] = rng.uniform(0.3, 0.9, size=20)
    covered = rng.random(frequencies.shape) >= 0.1
    # Each cell stays covered at its clone's first variant.
    covered[np.arange(40), 2 * truth] = True
    frequencies[~covered] = np.nan
    return frequencies, truth


def test_frequency_clones_recover_two_clones_and_explain_the_rest() -> None:
    rng = np.random.default_rng(0)
    frequencies, truth = _two_clones(rng)
    extra = np.asarray(
        [
            [np.nan] * 4,  # not covered at any variant
            [0.0, 0.0, 0.0, np.nan],  # covered, showing none
        ]
    )
    called = frequency_clones(
        np.vstack([frequencies, extra]),
        n_neighbors=10,
        resolution=1.0,
        random_seed=4444,
        memory_bytes=1 << 30,
    )
    clusters = called.clusters[:40]
    assert set(clusters[truth == 0].tolist()) | set(clusters[truth == 1].tolist()) == {
        1,
        2,
    }
    assert len(set(clusters[truth == 0].tolist())) == 1
    assert len(set(clusters[truth == 1].tolist())) == 1
    assert called.clusters[40:].tolist() == [0, 0]
    assert [CLONE_STATUSES[code] for code in called.status[40:]] == [
        "no_coverage",
        "no_alternative",
    ]
    assert clone_status_counts(called.status) == {
        "clustered": 40,
        "no_coverage": 1,
        "no_alternative": 1,
        "no_neighbor": 0,
        "no_group": 0,
    }


def test_frequency_clones_keep_groups_apart_and_skip_cells_without_one() -> None:
    rng = np.random.default_rng(1)
    frequencies, truth = _two_clones(rng)
    # The same clones in two groups; one cell has no group.
    groups = np.tile([0, 1], 20)
    groups[5] = -1
    called = frequency_clones(
        frequencies,
        groups=groups,
        n_neighbors=5,
        resolution=1.0,
        random_seed=4444,
        memory_bytes=1 << 30,
    )
    assert called.clusters[5] == 0
    assert CLONE_STATUSES[called.status[5]] == "no_group"
    for group in (0, 1):
        members = (groups == group) & (called.clusters > 0)
        labels = called.clusters[members]
        # Each group's clones number from 1 and follow the truth.
        assert set(labels.tolist()) == {1, 2}
        for clone in (0, 1):
            assert len(set(labels[truth[members] == clone].tolist())) == 1


def test_frequency_clones_leave_lone_cells_without_a_clone() -> None:
    # Two cells share variant 0; a third shows only variant 1.
    frequencies = np.asarray([[0.5, 0.0], [0.6, 0.0], [0.0, 0.7]])
    called = frequency_clones(
        frequencies, n_neighbors=2, resolution=1.0, random_seed=1, memory_bytes=1 << 30
    )
    assert CLONE_STATUSES[called.status[2]] == "no_neighbor"
    assert called.clusters.tolist()[:2] == [1, 1]
    assert called.clusters[2] == 0
    assert called.n_edges == 1


def test_frequency_clones_skip_a_group_whose_cells_join_no_other() -> None:
    # Group 0 holds two cells that share variant 0; group 1 one cell that
    # shows it, alone in its group.
    frequencies = np.asarray([[0.5, 0.0], [0.6, 0.0], [0.7, 0.0]])
    called = frequency_clones(
        frequencies,
        groups=np.asarray([0, 0, 1]),
        n_neighbors=2,
        resolution=1.0,
        random_seed=1,
        memory_bytes=1 << 30,
    )
    assert called.clusters.tolist() == [1, 1, 0]
    assert [CLONE_STATUSES[code] for code in called.status] == [
        "clustered",
        "clustered",
        "no_neighbor",
    ]
    assert called.n_edges == 1


def test_frequency_clones_check_inputs_and_memory() -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        frequency_clones(
            np.asarray([[1.5]]),
            n_neighbors=2,
            resolution=1.0,
            random_seed=1,
            memory_bytes=1 << 20,
        )
    with pytest.raises(ValueError, match="one group per cell"):
        frequency_clones(
            np.asarray([[0.5]]),
            groups=np.asarray([0, 1]),
            n_neighbors=2,
            resolution=1.0,
            random_seed=1,
            memory_bytes=1 << 20,
        )
    with pytest.raises(ValueError, match="matrix of cells by variants"):
        frequency_clones(
            np.asarray([0.5]),
            n_neighbors=2,
            resolution=1.0,
            random_seed=1,
            memory_bytes=1,
        )
    frequencies, _truth = _two_clones(np.random.default_rng(2))
    # Enough for the coverage masks but not for the similarities.
    with pytest.raises(MemoryError, match="Comparing the allele frequencies"):
        frequency_clones(
            frequencies, n_neighbors=5, resolution=1.0, random_seed=1, memory_bytes=5000
        )
    # Enough for the similarities but not for the graph's fixed arrays, then
    # not for its joins.
    for budget, message in (
        (12_000, "before it holds a join"),
        (40_000, "graph of 40 cells does not fit"),
    ):
        with pytest.raises(MemoryError, match=message):
            frequency_clones(
                frequencies,
                n_neighbors=5,
                resolution=1.0,
                random_seed=1,
                memory_bytes=budget,
            )
    # Enough for the similarities and the graph, but not for partitioning the
    # two cells that join, which needs the Leiden backend's fixed 256 KiB.
    with pytest.raises(
        MemoryError, match="Partitioning the shared-neighbour graph of 2 cells"
    ):
        frequency_clones(
            np.asarray([[0.5, 0.0], [0.6, 0.0], [0.0, 0.7]]),
            n_neighbors=2,
            resolution=1.0,
            random_seed=1,
            memory_bytes=10_000,
        )


@pytest.mark.parametrize(
    ("n_cells", "n_variants", "n_groups"),
    [
        pytest.param(600, 100, 1, marks=pytest.mark.slow),
        pytest.param(600, 100, 3, marks=pytest.mark.slow),
        (1500, 8, 1),
    ],
)
def test_frequency_clones_hold_no_more_than_the_smallest_budget_they_accept(
    n_cells, n_variants, n_groups
) -> None:
    # Clones marked by variants of their own, sparse noise and uncovered
    # entries; every frequency is held apart from the budget, as the input.
    rng = np.random.default_rng(n_cells + n_variants)
    clone = rng.integers(0, 8, size=n_cells)
    owner = rng.integers(0, 8, size=n_variants)
    frequencies = np.where(
        clone[:, None] == owner[None, :],
        rng.uniform(0.2, 0.9, size=(n_cells, n_variants)),
        0.0,
    )
    noise = rng.random(frequencies.shape) < 0.02
    frequencies[noise] = rng.uniform(0.01, 0.1, size=int(noise.sum()))
    frequencies[rng.random(frequencies.shape) < 0.15] = np.nan
    groups = np.arange(n_cells) % n_groups

    def call(budget: int) -> None:
        frequency_clones(
            frequencies,
            groups=groups,
            n_neighbors=10,
            resolution=1.0,
            random_seed=1,
            memory_bytes=budget,
            nthreads=1,
        )

    budget = smallest_budget(call, 1 << 30)
    assert traced_peak(lambda: call(budget), budget)[1] <= budget


def test_frequency_clones_refuse_a_budget_below_the_coverage_masks() -> None:
    # The similarities of each group of two cells fit, but not the mask of
    # all 500 cells' coverage, a byte per frequency.
    frequencies = np.full((500, 400), 0.5)
    with pytest.raises(MemoryError, match="Checking the coverage of 500 cells"):
        frequency_clones(
            frequencies,
            groups=np.arange(500) // 2,
            n_neighbors=2,
            resolution=1.0,
            random_seed=1,
            memory_bytes=190_000,
        )


def test_shared_neighbour_graph_is_symmetric_once_mirrored() -> None:
    neighborhoods = np.asarray([[0, 1, 2], [1, 0, 2], [2, 1, 0], [3, 2, -1]])
    graph = shared_neighbor_graph(neighborhoods, prune=0.0)
    assert np.all(graph.row < graph.col)
    mirrored = (graph + graph.T).toarray()
    np.testing.assert_allclose(mirrored, mirrored.T)
    assert isinstance(graph, coo_matrix)
