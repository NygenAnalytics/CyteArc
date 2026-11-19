"""LocalMAP's kernels, against pacmap 0.9.1's outputs and the method's invariants."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cytearc.embeddings import localmap
from cytearc.utils.numba import numba_threads
from tests.perturbation_helpers import use_python_kernels
from tests.random_stream_helpers import child_key, draw

REFERENCE = np.load(
    Path(__file__).parent / "data" / "localmap" / "pacmap_0_9_1_reference.npz"
)
ITERATIONS = (100, 100, 250)


def _python_kernels(monkeypatch: pytest.MonkeyPatch, python_kernels: bool) -> None:
    """Run the kernels uncompiled, whose 64-bit products wrap as compiled ones do."""
    if python_kernels:
        use_python_kernels(monkeypatch, localmap)
        np.seterr(over="ignore")


@pytest.fixture(autouse=True)
def _restore_floating_point_errors():
    previous = np.geterr()
    yield
    np.seterr(**previous)


def _rows_graph(rows: np.ndarray, weights: np.ndarray | None = None) -> csr_matrix:
    """A graph whose row i holds an edge to each cell of ``rows[i]``."""
    n_cells, width = rows.shape
    values = np.ones(rows.size) if weights is None else np.asarray(weights).ravel()
    return csr_matrix(
        (values, rows.ravel(), np.arange(n_cells + 1) * width), shape=(n_cells, n_cells)
    )


def _clusters(
    n_per_cluster: int = 60, n_clusters: int = 4, dims: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10.0, size=(n_clusters, dims))
    cells = np.concatenate(
        [center + rng.normal(size=(n_per_cluster, dims)) for center in centers]
    ).astype(np.float32)
    return cells, np.repeat(np.arange(n_clusters), n_per_cluster)


def _nearest(cells: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(cells[:, None] - cells[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :k]


def _relative_error(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def _reference_inputs() -> tuple[tuple[np.ndarray, ...], ...]:
    graph = _rows_graph(REFERENCE["neighbor_pairs"])
    mid_near = np.ascontiguousarray(REFERENCE["mid_near_pairs"], dtype=np.int32)
    further = np.ascontiguousarray(REFERENCE["further_pairs"], dtype=np.int32)
    return (
        localmap.neighbor_partners(graph),
        localmap.pair_partners(mid_near),
        localmap.pair_partners(further),
    )


@pytest.mark.parametrize("python_kernels", [False, True])
@pytest.mark.parametrize("iteration", [0, 50, 150, 200])
def test_gradient_matches_pacmap(monkeypatch, iteration, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    neighbors, mid_near, further = _reference_inputs()
    w_mid_near, w_neighbors, w_further = localmap.phase_weights(iteration, ITERATIONS)
    state = REFERENCE["state"]
    gradient = np.empty_like(state)

    localmap._gradient(
        state,
        *neighbors,
        *mid_near,
        *further,
        w_neighbors,
        w_mid_near,
        w_further,
        0.0,
        gradient,
        3,
    )

    assert _relative_error(gradient, REFERENCE[f"gradient_{iteration}"]) < 1e-5


@pytest.mark.parametrize("python_kernels", [False, True])
def test_local_gradient_matches_pacmap(monkeypatch, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    neighbors, mid_near, further = _reference_inputs()
    w_mid_near, w_neighbors, w_further = localmap.phase_weights(300, ITERATIONS)
    state = REFERENCE["state"]
    gradient = np.empty_like(state)

    localmap._gradient(
        state,
        *neighbors,
        *mid_near,
        *further,
        w_neighbors,
        w_mid_near,
        w_further,
        5.0,
        gradient,
        3,
    )

    assert _relative_error(gradient, REFERENCE["gradient_local"]) < 1e-5


def test_first_two_phases_match_pacmap():
    layout = localmap.localmap_layout(
        REFERENCE["start"],
        _rows_graph(REFERENCE["neighbor_pairs"]),
        REFERENCE["mid_near_pairs"],
        REFERENCE["further_pairs"],
        iterations=(100, 100, 0),
    )

    assert _relative_error(layout, REFERENCE["layout_200"]) < 1e-3


def _replayed_schedule(
    start, neighbors, mid_near, further, iterations, learning_rate, local_scale
):
    """Replay PaCMAP's loop over fixed pairs: its phase weights and Adam's steps.

    The third phase, after the first two, scales the neighbors' attraction
    by ``local_scale``, half of LocalMAP's ``max_distance``.
    """
    layout = start.copy()
    first, second = np.zeros_like(layout), np.zeros_like(layout)
    gradient = np.zeros_like(layout)
    for iteration in range(sum(iterations)):
        w_mid_near, w_neighbors, w_further = localmap.phase_weights(
            iteration, iterations
        )
        local = iteration > iterations[0] + iterations[1]
        localmap._gradient(
            layout,
            *neighbors,
            *mid_near,
            *further,
            w_neighbors,
            w_mid_near,
            w_further,
            local_scale if local else 0.0,
            gradient,
            3,
        )
        step = (
            learning_rate
            * math.sqrt(1 - 0.999 ** (iteration + 1))
            / (1 - 0.9 ** (iteration + 1))
        )
        localmap._adam_step(layout, gradient, first, second, step)
    return layout


def test_three_phases_from_fixed_pairs_match_pacmap():
    # localmap_layout draws further pairs again in the third phase, which
    # pacmap does from shared random state, so this replays its loop with
    # the reference's fixed pairs.
    layout = _replayed_schedule(
        REFERENCE["start"], *_reference_inputs(), ITERATIONS, 1.0, 5.0
    )

    assert _relative_error(layout, REFERENCE["layout_450"]) < 2e-2


def test_a_layout_whose_further_pairs_cannot_change_replays_pacmaps_schedule():
    # Every cell names every other cell as a neighbor, so the third phase has
    # no cell to draw as a further pair, and the layout keeps the pairs it is
    # given. Its loop is then PaCMAP's, with a third phase from iteration 21
    # whose neighbors attract at half of max_distance.
    n_cells = 30
    rng = np.random.default_rng(6)
    rows = np.stack([np.delete(np.arange(n_cells), cell) for cell in range(n_cells)])
    graph = _rows_graph(rows, rng.uniform(0.2, 1.0, size=rows.shape))
    mid_near = rng.integers(0, n_cells, size=(n_cells, 4)).astype(np.int32)
    further = rng.integers(0, n_cells, size=(n_cells, 6)).astype(np.int32)
    start = localmap.initial_layout(rng.normal(size=(n_cells, 2)))
    iterations = (10, 10, 25)

    layout = localmap.localmap_layout(
        start,
        graph,
        mid_near,
        further,
        iterations=iterations,
        learning_rate=0.7,
        max_distance=6.0,
        random_seed=2,
    )

    expected = _replayed_schedule(
        start,
        localmap.neighbor_partners(graph),
        localmap.pair_partners(mid_near),
        localmap.pair_partners(further),
        iterations,
        0.7,
        3.0,
    )
    np.testing.assert_array_equal(layout, expected)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_adam_steps_follow_the_bias_corrected_moments(monkeypatch, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(8)
    signs = rng.choice([-1.0, 1.0], size=(4, 6, 2))
    gradients = (rng.uniform(1.0, 5.0, size=(4, 6, 2)) * signs).astype(np.float32)
    layout = np.zeros((6, 2), dtype=np.float32)
    first, second = np.zeros_like(layout), np.zeros_like(layout)
    expected = np.zeros((6, 2))
    mean, square = np.zeros((6, 2)), np.zeros((6, 2))

    for t, gradient in enumerate(gradients, start=1):
        # The step that localmap_layout passes, at a learning rate of 0.5.
        step = 0.5 * math.sqrt(1 - 0.999**t) / (1 - 0.9**t)
        localmap._adam_step(layout, gradient, first, second, step)
        mean = 0.9 * mean + 0.1 * gradient
        square = 0.999 * square + 0.001 * gradient.astype(float) ** 2
        # Adam's epsilon of 1e-7, which this leaves out, moves each of these
        # steps, about 0.5 long, by under 2e-6.
        expected -= 0.5 * (mean / (1 - 0.9**t)) / np.sqrt(square / (1 - 0.999**t))
        if t == 1:
            # The first step moves each coordinate by the learning rate,
            # against its gradient, whatever the gradient's size.
            np.testing.assert_allclose(layout, -0.5 * signs[0], rtol=1e-5)

    np.testing.assert_allclose(first, mean, rtol=1e-5)
    np.testing.assert_allclose(second, square, rtol=1e-5)
    np.testing.assert_allclose(layout, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_weighted_gradient_sums_each_edge_in_both_directions(
    monkeypatch, python_kernels
):
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(5)
    n_cells = 30
    rows = np.stack(
        [
            rng.choice(np.delete(np.arange(n_cells), i), 4, replace=False)
            for i in range(n_cells)
        ]
    )
    weights = rng.uniform(0.1, 1.0, size=rows.shape)
    graph = _rows_graph(rows, weights)
    state = rng.normal(scale=2.0, size=(n_cells, 2)).astype(np.float32)
    empty = (np.zeros(n_cells + 1, dtype=np.int64), np.zeros(0, dtype=np.int32))
    gradient = np.empty_like(state)

    localmap._gradient(
        state,
        *localmap.neighbor_partners(graph),
        *empty,
        *empty,
        2.0,
        0.0,
        1.0,
        0.0,
        gradient,
        4,
    )

    # Each edge (i, j) pulls i towards j and j towards i with the same force.
    scale = localmap.NEIGHBOR_PAIRS * n_cells / weights.sum()
    expected = np.zeros((n_cells, 2))
    for i in range(n_cells):
        for j, weight in zip(rows[i], weights[i], strict=True):
            offset = state[i].astype(float) - state[j]
            distance = 1.0 + offset @ offset
            force = 2.0 * weight * scale * 20.0 / (10.0 + distance) ** 2 * offset
            expected[i] += force
            expected[j] -= force
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-7)


def test_phase_weights_follow_the_schedule():
    assert localmap.phase_weights(0, ITERATIONS) == (1000.0, 2.0, 1.0)
    assert localmap.phase_weights(50, ITERATIONS) == pytest.approx((501.5, 2.0, 1.0))
    assert localmap.phase_weights(100, ITERATIONS) == (3.0, 3.0, 1.0)
    assert localmap.phase_weights(199, ITERATIONS) == (3.0, 3.0, 1.0)
    assert localmap.phase_weights(200, ITERATIONS) == (0.0, 1.0, 1.0)


def _partners_by_definition(indptr, indices, weights):
    n_cells = len(indptr) - 1
    partners = [
        list(
            zip(
                indices[indptr[i] : indptr[i + 1]],
                weights[indptr[i] : indptr[i + 1]],
                strict=True,
            )
        )
        for i in range(n_cells)
    ]
    for source in range(n_cells):
        for p in range(indptr[source], indptr[source + 1]):
            partners[indices[p]].append((source, weights[p]))
    return partners


@pytest.mark.parametrize("python_kernels", [False, True])
@pytest.mark.parametrize("n_chunks", [1, 2, 7, 50])
def test_partners_list_both_directions_in_a_fixed_order(
    monkeypatch, n_chunks, python_kernels
):
    _python_kernels(monkeypatch, python_kernels)
    rng = np.random.default_rng(3)
    degrees = rng.integers(0, 6, size=50)
    indptr = np.concatenate([[0], np.cumsum(degrees)]).astype(np.int64)
    indices = rng.integers(0, 50, size=indptr[-1]).astype(np.int32)
    weights = rng.uniform(size=indptr[-1]).astype(np.float32)

    out_indptr, partners, out_weights = localmap._partners(
        indptr, indices, weights, n_chunks
    )

    expected = _partners_by_definition(indptr, indices, weights)
    assert (
        out_indptr.tolist() == np.cumsum([0] + [len(row) for row in expected]).tolist()
    )
    assert partners.tolist() == [cell for row in expected for cell, _ in row]
    assert out_weights.tolist() == [weight for row in expected for _, weight in row]


@pytest.mark.parametrize("python_kernels", [False, True])
def test_pair_partners_carry_no_weights(monkeypatch, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    pairs = np.array([[1, 2], [2, 0], [0, 1]], dtype=np.int32)

    indptr, partners = localmap.pair_partners(pairs)

    assert indptr.tolist() == [0, 4, 8, 12]
    assert partners.tolist() == [1, 2, 1, 2, 2, 0, 0, 2, 0, 1, 0, 1]


def test_neighbor_partners_scale_weights_to_ten_pairs_per_cell():
    rng = np.random.default_rng(1)
    rows = np.stack(
        [rng.choice(np.delete(np.arange(40), i), 7, replace=False) for i in range(40)]
    )
    weights = rng.uniform(0.2, 1.0, size=rows.shape)

    indptr, partners, scaled = localmap.neighbor_partners(_rows_graph(rows, weights))

    assert scaled.dtype == np.float32
    assert scaled.sum() == pytest.approx(2 * localmap.NEIGHBOR_PAIRS * 40, rel=1e-5)
    # Scaling keeps the weights' proportions.
    own = scaled[indptr[0] : indptr[0] + 7]
    np.testing.assert_allclose(own / own[0], weights[0] / weights[0, 0], rtol=1e-5)
    assert partners[:7].tolist() == rows[0].tolist()


def test_neighbor_partners_ignore_zero_weights():
    rows = np.array([[(i + 1) % 6, (i + 2) % 6] for i in range(6)])
    weights = np.ones(rows.shape)
    weights[0, 1] = 0.0

    indptr, partners, scaled = localmap.neighbor_partners(_rows_graph(rows, weights))

    # Cell 0 keeps its edge to cell 1 and is named by cells 4 and 5; cell 2
    # keeps its own two edges and is named by cell 1 alone.
    assert partners[indptr[0] : indptr[1]].tolist() == [1, 4, 5]
    assert partners[indptr[2] : indptr[3]].tolist() == [3, 4, 1]
    # The eleven edges left share the attraction of ten pairs per cell.
    np.testing.assert_allclose(scaled, 10 * 6 / 11, rtol=1e-6)


@pytest.mark.parametrize(
    ("graph", "error", "message"),
    [
        (np.ones((3, 3)), TypeError, "sparse"),
        (csr_matrix(np.ones((3, 4))), ValueError, "square"),
        (csr_matrix(np.array([[0, -1.0], [1.0, 0]])), ValueError, "non-negative"),
        (csr_matrix(np.array([[0, np.nan], [1.0, 0]])), ValueError, "finite"),
        (csr_matrix(np.array([[1.0, 1.0], [1.0, 0]])), ValueError, "itself"),
        (
            csr_matrix(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])),
            ValueError,
            "cell 2",
        ),
    ],
)
def test_neighbor_partners_reject_unusable_graphs(graph, error, message):
    with pytest.raises(error, match=message):
        localmap.neighbor_partners(graph)


def test_mid_near_pairs_are_second_closest_of_six():
    cells, _ = _clusters(n_per_cluster=100)
    pairs = localmap.mid_near_pairs(cells, 5, 11)

    # The second smallest of six uniform ranks sits at 2/7 of the order.
    distances = np.linalg.norm(cells[:, None] - cells[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    ranks = np.argsort(np.argsort(distances, axis=1), axis=1)
    fractions = np.take_along_axis(ranks, pairs, axis=1) / (cells.shape[0] - 1)
    assert fractions.mean() == pytest.approx(2 / 7, abs=0.02)
    assert not np.any(pairs == np.arange(cells.shape[0])[:, None])
    assert all(len(set(row)) == row.size for row in pairs)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_each_mid_near_pair_ranks_second_of_six_among_the_cells_left(
    monkeypatch, python_kernels
):
    _python_kernels(monkeypatch, python_kernels)
    # Cells at powers of two, so no two distances from a cell tie, and five
    # pairs, which leave seven cells for the six candidates of the last pair.
    positions = 2.0 ** np.arange(12)

    pairs = localmap.mid_near_pairs(positions[:, None], 5, 3)

    for cell, row in enumerate(pairs.tolist()):
        for pair, partner in enumerate(row):
            left = sorted(
                (
                    other
                    for other in range(12)
                    if other != cell and other not in row[:pair]
                ),
                key=lambda other: abs(positions[other] - positions[cell]),
            )
            # The second closest of six distinct candidates has one of them
            # closer and four farther.
            assert 1 <= left.index(partner) <= len(left) - 5


@pytest.mark.parametrize("python_kernels", [False, True])
def test_cell_keys_and_draws_are_splitmix64_streams(monkeypatch, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    key = localmap.draw_key(7, 2)

    for cell in (0, 1, 2**31 - 2):
        assert int(localmap._cell_key(key, cell)) == child_key(int(key), cell)
    for counter, bound in ((0, 1), (1, 6), (5, 1000), (2**40, 2**31 - 1)):
        drawn = localmap._random_below(key, counter, bound)
        # The top 32 bits of the draw, scaled to the bound.
        assert drawn == (draw(int(key), counter) >> 32) * bound >> 32
        assert 0 <= drawn < bound


def test_mid_near_pairs_depend_on_the_seed_only():
    cells, _ = _clusters()

    with numba_threads(1):
        one = localmap.mid_near_pairs(cells, 5, 4)
    with numba_threads(3):
        three = localmap.mid_near_pairs(cells, 5, 4)

    np.testing.assert_array_equal(one, three)
    assert not np.array_equal(one, localmap.mid_near_pairs(cells, 5, 5))


def test_mid_near_pairs_reject_too_few_cells_and_invalid_coordinates():
    with pytest.raises(ValueError, match="more than 11 cells"):
        localmap.mid_near_pairs(np.zeros((11, 3), dtype=np.float32), 5, 0)
    with pytest.raises(ValueError, match="finite"):
        localmap.mid_near_pairs(np.full((20, 3), np.inf), 5, 0)
    with pytest.raises(ValueError, match="matrix"):
        localmap.mid_near_pairs(np.zeros(20), 5, 0)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_further_pairs_avoid_the_cell_and_its_neighbors(monkeypatch, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    cells, _ = _clusters()
    neighbors = _nearest(cells, 10)

    pairs = localmap.further_pairs(_rows_graph(neighbors), 20, 2)

    n_cells = cells.shape[0]
    for cell in range(n_cells):
        row = set(pairs[cell].tolist())
        assert len(row) == 20
        assert cell not in row
        assert not row & set(neighbors[cell].tolist())
    # Uniform draws name each cell about 20 times.
    counts = np.bincount(pairs.ravel(), minlength=n_cells)
    assert counts.mean() == 20
    assert counts.max() < 45


@pytest.mark.parametrize("python_kernels", [False, True])
def test_further_pairs_take_every_other_cell_when_just_enough_remain(
    monkeypatch, python_kernels
):
    _python_kernels(monkeypatch, python_kernels)
    # Six cells in a ring, each naming the next, leave four cells for four pairs.
    graph = _rows_graph(np.array([[(row + 1) % 6] for row in range(6)]))

    pairs = localmap.further_pairs(graph, 4, 5)

    for cell, row in enumerate(pairs.tolist()):
        assert sorted(row) == sorted(set(range(6)) - {cell, (cell + 1) % 6})


def test_further_pairs_reject_too_few_cells():
    graph = _rows_graph(np.array([[(row + 1) % 5] for row in range(5)]))

    with pytest.raises(ValueError, match="at least 6 cells"):
        localmap.further_pairs(graph, 4, 0)


@pytest.mark.parametrize("python_kernels", [False, True])
def test_redraw_keeps_pairs_without_a_nearby_candidate(monkeypatch, python_kernels):
    _python_kernels(monkeypatch, python_kernels)
    cells, _ = _clusters()
    n_cells = cells.shape[0]
    neighbors = _nearest(cells, 10)
    graph = localmap._checked_graph(_rows_graph(neighbors))
    pairs = localmap.further_pairs(graph, 20, 0)
    layout = np.random.default_rng(1).normal(size=(n_cells, 2)).astype(np.float32)
    layout[0] = [1e6, 1e6]
    before = pairs.copy()

    localmap._redraw_nearby_further_pairs(
        layout, graph.indptr, graph.indices, pairs, np.uint64(9), 100.0, 4
    )

    np.testing.assert_array_equal(pairs[0], before[0])
    changed = pairs != before
    assert changed[1:].mean() > 0.9
    rows, columns = np.nonzero(changed)
    drawn = pairs[rows, columns]
    assert np.all(np.linalg.norm(layout[rows] - layout[drawn], axis=1) <= 10.0)
    assert not np.any(drawn == rows)
    assert not np.any(neighbors[rows] == drawn[:, None])
    for cell in range(1, n_cells):
        new = pairs[cell][changed[cell]]
        assert len(set(new.tolist())) == new.size


def _redrawn_by_definition(layout, graph, pairs, key, max_distance):
    """Each cell's further pairs after a redraw, drawn by the rules of the third phase.

    A cell draws each pair in turn from its own SplitMix64 stream, at most
    100 times, and takes the first cell that is not itself, lies within
    ``max_distance`` in the layout, is not one of its neighbors and is not
    one of its new pairs; a pair that no draw fills keeps its cell.
    """
    n_cells = layout.shape[0]
    redrawn = pairs.copy()
    for cell in range(n_cells):
        stream = child_key(int(key), cell)
        neighbors = set(graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]])
        counter = 0
        new: list[int] = []
        for pair in range(pairs.shape[1]):
            for _trial in range(100):
                candidate = (draw(stream, counter) >> 32) * n_cells >> 32
                counter += 1
                offset = layout[cell].astype(np.float64) - layout[candidate]
                if (
                    candidate != cell
                    and offset @ offset <= max_distance**2
                    and candidate not in neighbors
                    and candidate not in new
                ):
                    new.append(candidate)
                    redrawn[cell, pair] = candidate
                    break
    return redrawn


@pytest.mark.parametrize("python_kernels", [False, True])
def test_redraw_takes_the_first_draws_of_each_cells_stream_that_fit(
    monkeypatch, python_kernels
):
    _python_kernels(monkeypatch, python_kernels)
    # Twelve cells on a line, one apart, each naming the next two as its
    # neighbors. Within a distance of 4, cell 0 has two cells to draw and
    # cell 1 three, cell 0 among them.
    n_cells = 12
    layout = np.zeros((n_cells, 2), dtype=np.float32)
    layout[:, 0] = np.arange(n_cells)
    rows = np.array([[(i + 1) % n_cells, (i + 2) % n_cells] for i in range(n_cells)])
    graph = localmap._checked_graph(_rows_graph(rows))
    pairs = np.array(
        [
            [(i + 5) % n_cells, (i + 6) % n_cells, (i + 7) % n_cells]
            for i in range(n_cells)
        ],
        dtype=np.int32,
    )
    key = localmap.draw_key(5, 3, 210)
    expected = _redrawn_by_definition(layout, graph, pairs, key, 4.0)
    # The draws give cell 1 the cell 0, and leave cell 0's third pair as it was.
    assert 0 in expected[1].tolist()
    assert expected[0, 2] == pairs[0, 2]

    localmap._redraw_nearby_further_pairs(
        layout, graph.indptr, graph.indices, pairs, key, 16.0, 4
    )

    np.testing.assert_array_equal(pairs, expected)


def test_initial_layout_keeps_proportions_at_a_small_scale():
    values = np.column_stack([np.linspace(-30, 50, 101), np.linspace(1, 3, 101)])

    layout = localmap.initial_layout(values)

    assert layout.dtype == np.float32
    np.testing.assert_allclose(layout.mean(axis=0), 0, atol=1e-10)
    assert np.sqrt(np.mean(layout.astype(float) ** 2)) == pytest.approx(1e-4, rel=1e-5)
    ratio = values.std(axis=0)[0] / values.std(axis=0)[1]
    assert layout.std(axis=0)[0] / layout.std(axis=0)[1] == pytest.approx(
        ratio, rel=1e-4
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones((5, 2)), "same coordinates"),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), "finite"),
        (np.ones(5), "matrix"),
    ],
)
def test_initial_layout_rejects_unusable_values(values, message):
    with pytest.raises(ValueError, match=message):
        localmap.initial_layout(values)


def _layout_inputs(seed: int = 0):
    cells, labels = _clusters()
    neighbors = _nearest(cells, 10)
    weights = np.random.default_rng(seed).uniform(0.3, 1.0, size=neighbors.shape)
    graph = _rows_graph(neighbors, weights)
    mid_near = localmap.mid_near_pairs(cells, 5, seed)
    further = localmap.further_pairs(graph, 20, seed)
    start = localmap.initial_layout(
        np.random.default_rng(seed).normal(size=(cells.shape[0], 2))
    )
    return start, graph, mid_near, further, labels


def test_layout_separates_clusters():
    start, graph, mid_near, further, labels = _layout_inputs()

    layout = localmap.localmap_layout(start, graph, mid_near, further, random_seed=1)

    assert np.mean(labels[_nearest(layout, 10)] == labels[:, None]) > 0.99
    centers = np.stack([layout[labels == c].mean(axis=0) for c in range(4)])
    spread = max(
        np.linalg.norm(layout[labels == c] - centers[c], axis=1).max() for c in range(4)
    )
    gaps = np.linalg.norm(centers[:, None] - centers[None], axis=2)[
        np.triu_indices(4, 1)
    ]
    assert gaps.min() > spread


def test_layout_does_not_depend_on_threads_and_leaves_inputs_unchanged():
    start, graph, mid_near, further, _ = _layout_inputs()
    copies = [array.copy() for array in (start, graph.toarray(), mid_near, further)]

    one = localmap.localmap_layout(
        start, graph, mid_near, further, random_seed=3, nthreads=1
    )
    three = localmap.localmap_layout(
        start, graph, mid_near, further, random_seed=3, nthreads=3
    )

    np.testing.assert_array_equal(one, three)
    for array, copy in zip(
        (start, graph.toarray(), mid_near, further), copies, strict=True
    ):
        np.testing.assert_array_equal(array, copy)
    other = localmap.localmap_layout(start, graph, mid_near, further, random_seed=4)
    assert not np.array_equal(one, other)


def test_layout_reports_each_iteration():
    start, graph, mid_near, further, _ = _layout_inputs()
    seen: list[int] = []

    localmap.localmap_layout(
        start, graph, mid_near, further, iterations=(2, 2, 3), on_iteration=seen.append
    )

    assert seen == list(range(7))


def test_layout_rejects_inputs_of_other_cells():
    start, graph, mid_near, further, _ = _layout_inputs()

    with pytest.raises(ValueError, match="mid_near must hold one row"):
        localmap.localmap_layout(start, graph, mid_near[:-1], further)
    with pytest.raises(ValueError, match="same cells"):
        localmap.localmap_layout(start[:-1], graph, mid_near[:-1], further[:-1])
    with pytest.raises(ValueError, match="matrix"):
        localmap.localmap_layout(start[:, 0], graph, mid_near, further)


def test_peak_bytes_grow_with_each_input():
    base = localmap.localmap_peak_bytes(1000, 11000, dims=2, n_mid_near=5, n_further=20)

    assert (
        localmap.localmap_peak_bytes(2000, 11000, dims=2, n_mid_near=5, n_further=20)
        > base
    )
    assert (
        localmap.localmap_peak_bytes(1000, 22000, dims=2, n_mid_near=5, n_further=20)
        > base
    )
    assert (
        localmap.localmap_peak_bytes(1000, 11000, dims=3, n_mid_near=5, n_further=20)
        > base
    )
    assert (
        localmap.localmap_peak_bytes(1000, 11000, dims=2, n_mid_near=5, n_further=30)
        > base
    )
    with pytest.raises(ValueError, match="n_cells"):
        localmap.localmap_peak_bytes(0, 0, dims=2, n_mid_near=5, n_further=20)


def test_peak_bytes_count_the_lists_and_arrays_that_a_layout_keeps():
    n_cells = 40
    rng = np.random.default_rng(9)
    rows = np.stack(
        [
            rng.choice(np.delete(np.arange(n_cells), i), 8, replace=False)
            for i in range(n_cells)
        ]
    )
    narrow, wide = _rows_graph(rows[:, :7]), _rows_graph(rows)
    cells, _ = _clusters(n_per_cluster=10)
    mid_near = localmap.mid_near_pairs(cells, 5, 0)
    further = localmap.further_pairs(wide, 20, 0)

    def estimate(graph, dims=2):
        return localmap.localmap_peak_bytes(
            n_cells, graph.nnz, dims=dims, n_mid_near=5, n_further=20
        )

    def held(arrays):
        return sum(array.nbytes for array in arrays)

    # An edge adds what its partner entries hold in the neighbors' list: a
    # cell and a weight for each of its two cells.
    assert estimate(wide) - estimate(narrow) == held(
        localmap.neighbor_partners(wide)
    ) - held(localmap.neighbor_partners(narrow))
    # A dimension adds a float32 value of each cell to the layout, its
    # gradient and Adam's two moments.
    assert estimate(wide, dims=3) - estimate(wide) == 4 * 4 * n_cells
    # The estimate covers the lists and rows of every pair and the four
    # arrays of the optimizer, which the layout keeps from its start.
    kept = (
        held(localmap.neighbor_partners(wide))
        + held(localmap.pair_partners(mid_near))
        + held(localmap.pair_partners(further))
        + mid_near.nbytes
        + further.nbytes
        + 4 * 4 * n_cells * 2
    )
    assert estimate(wide) >= kept


def test_graph_check_copies_only_graphs_it_must_change():
    rows = np.array([[(i + 1) % 6, (i + 2) % 6] for i in range(6)])
    weights = np.ones(rows.shape)
    weights[0, 1] = 0.0
    graph = _rows_graph(rows, weights)

    checked = localmap._checked_graph(graph)

    # The caller's float64 graph with a stored zero is copied, not changed.
    assert checked is not graph
    assert graph.nnz == 12 and graph.data.dtype == np.float64
    assert checked.nnz == 11 and checked.data.dtype == np.float32
    assert localmap._checked_graph(checked) is checked
    # A float32 graph is copied as well when it stores a zero.
    single = _rows_graph(rows, weights.astype(np.float32))
    assert single.data.dtype == np.float32
    checked_single = localmap._checked_graph(single)
    assert checked_single is not single
    assert single.nnz == 12 and checked_single.nnz == 11


def test_graphs_of_more_cells_than_int32_partners_hold_are_refused():
    # SciPy keeps int64 offsets for 2**31 rows, so one stored zero repeated by
    # a zero stride stands for the offsets of a graph without edges, and the
    # graph holds a few bytes, not 16 GiB. Below 2**31 rows SciPy would copy
    # the offsets into int32.
    n_cells = 2**31
    offsets = np.broadcast_to(np.zeros(1, dtype=np.int64), (n_cells + 1,))
    graph = csr_matrix(
        (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), offsets),
        shape=(n_cells, n_cells),
    )
    assert graph.indptr.strides == (0,)

    with pytest.raises(ValueError, match=r"fewer than 2\*\*31 - 1 cells"):
        localmap.neighbor_partners(graph)
    with pytest.raises(ValueError, match=r"fewer than 2\*\*31 - 1 cells"):
        localmap.further_pairs(graph, 4, 0)


_SETTINGS = {
    "localmap_dims": 2,
    "n_mid_near": 5,
    "n_further": 20,
    "iterations": ITERATIONS,
    "learning_rate": 1.0,
    "max_distance": 10.0,
    "random_seed": 4444,
}


def test_settings_are_returned_as_python_numbers():
    settings = localmap.localmap_settings(
        **{
            **_SETTINGS,
            "localmap_dims": np.int64(3),
            "iterations": [np.int64(10), 20, 30],
            "learning_rate": 1,
            "max_distance": np.float32(8.0),
        }
    )

    assert settings == localmap.LocalmapSettings(3, 5, 20, (10, 20, 30), 1.0, 8.0, 4444)
    assert type(settings.localmap_dims) is int
    assert all(type(phase) is int for phase in settings.iterations)
    assert type(settings.learning_rate) is type(settings.max_distance) is float


@pytest.mark.parametrize(
    ("setting", "value", "error", "message"),
    [
        ("localmap_dims", True, TypeError, "^localmap_dims must be an integer$"),
        ("n_further", 0, ValueError, "^n_further must be at least 1$"),
        ("iterations", "abc", TypeError, "^iterations must be three integers"),
        ("iterations", (1, 2, 3, 4), TypeError, "^iterations must be three integers"),
        ("iterations", (1, 2.5, 3), TypeError, r"^iterations\[1\] must be an integer$"),
        ("learning_rate", math.inf, ValueError, "^learning_rate must be finite$"),
        ("max_distance", 0, ValueError, "^max_distance must be finite and positive$"),
        ("random_seed", -1, ValueError, "^random_seed must be at least 0$"),
    ],
)
def test_settings_refuse_invalid_values(setting, value, error, message):
    with pytest.raises(error, match=message):
        localmap.localmap_settings(**{**_SETTINGS, setting: value})
