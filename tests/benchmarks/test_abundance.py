"""Benchmarks of the Milo kernels that stream a kNN graph.

``make_neighborhoods`` builds the cell-major membership in row blocks with
``block_membership``, and ``count_neighborhoods`` streams it into the
neighbourhood x sample counts with ``count_block``. Each benchmark checks its
result against a direct construction of miloR's undirected neighbourhoods.
"""

import numpy as np
import pytest

from .harness import Ladder

pytestmark = pytest.mark.benchmark

K = 30
PROP = 0.05
N_SAMPLES = 16
BLOCK_ROWS = 65_536
GRAPH_CELLS = Ladder(sizes=(25_000, 50_000, 100_000, 200_000), smoke=600)


def _graph(n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(812)
    knn = rng.integers(0, n_cells - 1, size=(n_cells, K), dtype=np.int64)
    knn += knn >= np.arange(n_cells)[:, None]
    index_cells = np.sort(
        rng.choice(n_cells, size=max(1, int(PROP * n_cells)), replace=False)
    )
    samples = rng.integers(0, N_SAMPLES, size=n_cells)
    return knn, index_cells, samples


def _membership(knn: np.ndarray, index_cells: np.ndarray):
    from cytearc.abundance.milo import block_membership, reverse_neighbors

    n_cells = knn.shape[0]
    reverse = reverse_neighbors(index_cells, knn[index_cells])
    position = np.full(n_cells, -1, dtype=np.int32)
    position[index_cells] = np.arange(index_cells.size, dtype=np.int32)
    blocks = []
    for start in range(0, n_cells, BLOCK_ROWS):
        block = knn[start : start + BLOCK_ROWS]
        blocks.append(
            (start,)
            + block_membership(start, block, position, reverse, index_cells.size)
        )
    return blocks


def _expected_members(knn: np.ndarray, index_cells: np.ndarray) -> list[set[int]]:
    members = [{int(cell), *knn[cell].tolist()} for cell in index_cells]
    position = {int(cell): nhood for nhood, cell in enumerate(index_cells)}
    for cell, row in enumerate(knn):
        for neighbor in row.tolist():
            if neighbor in position:
                members[position[neighbor]].add(cell)
    return members


def test_milo_neighborhood_membership(bench) -> None:
    def make(n_cells: int):
        knn, index_cells, _ = _graph(n_cells)
        return lambda: _membership(knn, index_cells)

    def check(n_cells: int, blocks) -> None:
        knn, index_cells, _ = _graph(n_cells)
        actual: list[set[int]] = [set() for _ in index_cells]
        for start, row_counts, ids in blocks:
            cells = start + np.repeat(np.arange(row_counts.size), row_counts)
            for cell, nhood in zip(cells.tolist(), ids.tolist(), strict=True):
                actual[nhood].add(cell)
        assert actual == _expected_members(knn, index_cells)

    bench("milo.neighborhood_membership", make, GRAPH_CELLS, check=check)


def test_milo_neighborhood_counts(bench) -> None:
    from cytearc.abundance.milo import count_block

    def make(n_cells: int):
        knn, index_cells, samples = _graph(n_cells)
        blocks = _membership(knn, index_cells)

        def call() -> np.ndarray:
            counts = np.zeros((index_cells.size, N_SAMPLES), dtype=np.int64)
            for start, row_counts, ids in blocks:
                count_block(
                    counts, row_counts, ids, samples[start : start + row_counts.size]
                )
            return counts

        return call

    def check(n_cells: int, counts) -> None:
        knn, index_cells, samples = _graph(n_cells)
        expected = np.zeros((index_cells.size, N_SAMPLES), dtype=np.int64)
        for nhood, members in enumerate(_expected_members(knn, index_cells)):
            np.add.at(expected[nhood], samples[sorted(members)], 1)
        np.testing.assert_array_equal(counts, expected)

    bench("milo.neighborhood_counts", make, GRAPH_CELLS, check=check)
