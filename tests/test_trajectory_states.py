"""The initial and terminal states of a walk on a cell graph.

``trajectory.states(graph, direction, n_states=...)`` splits the walk that a
fate map takes into metastable states by G-PCCA. The tests check the search
against values worked out by hand on walks between blocks of cells, against
SciPy's dense real Schur form and dense stationary distributions, against the
outputs that CellRank 2.4.0 and pyGPCCA 1.1.0 gave on a saved walk, and on a
synthetic process whose start, ends and side pocket are known. They then
check the operation's record, reuse, refusals and warnings, and that its
labels give a fate map its sinks.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.linalg import schur
from scipy.optimize import linear_sum_assignment

import cytearc.trajectory.states as states_module
from cytearc.storage.artifacts import ArtifactRef, artifact_group, list_artifacts
from cytearc.storage.diagnostics import DiagnosticWarning
from cytearc.trajectory import TrajectoryStatesResult
from cytearc.trajectory.fate import _make_transition, _normalize_pseudotime
from cytearc.trajectory.states import (
    coarse_grain,
    feasible_rotation,
    find_states,
    initial_rotation,
    invariant_basis,
    leading_eigenvectors,
    optimize_rotation,
    search_bytes,
    splits_conjugate_pair,
    state_cells,
    state_names,
)
from cytearc.utils.memory_budget import MemoryBudgetError

from .test_trajectory_record_paths import (
    _count_calls,
    _set_budget,
    _trajectory_store,
)

_REFERENCE = Path(__file__).parent / "data" / "trajectory_states" / "cellrank_reference"


def _blocks(sizes: list[int], within: float, cross: list[list[float]]) -> np.ndarray:
    """Weights of complete blocks of cells, ``cross[a][b]`` from each cell of block a to each of b."""
    starts = np.cumsum([0, *sizes])
    weights = np.zeros((starts[-1], starts[-1]))
    for first in range(len(sizes)):
        for second in range(len(sizes)):
            weights[
                starts[first] : starts[first + 1], starts[second] : starts[second + 1]
            ] = within if first == second else cross[first][second]
    np.fill_diagonal(weights, 0.0)
    return weights


def _row_normalized(weights: np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix(weights / weights.sum(axis=1, keepdims=True))


def _fate_walk(
    weights: np.ndarray, direction: np.ndarray, beta: float = 10.0
) -> np.ndarray:
    """The walk that a fate map takes, written from its definition.

    An edge to a cell whose score, rescaled to [0, 1], is lower by ``d``
    keeps ``2 / (1 + exp(beta * d))`` of its weight; rows sum to one.
    """
    scores = (direction - direction.min()) / (direction.max() - direction.min())
    drop = scores[:, None] - scores[None, :]
    kept = np.where(
        drop > 0, 2.0 / (1.0 + np.exp(beta * np.clip(drop, 0.0, None))), 1.0
    )
    biased = np.asarray(weights, dtype=np.float64) * kept
    np.fill_diagonal(biased, 0.0)
    return biased / biased.sum(axis=1, keepdims=True)


def _stationary(walk: np.ndarray) -> np.ndarray:
    """The walk's stationary distribution, from a dense left eigenvector."""
    values, vectors = np.linalg.eig(walk.T)
    pi = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return pi / pi.sum()


def _reference() -> dict[str, np.ndarray]:
    with np.load(f"{_REFERENCE}.npz") as loaded:
        return {name: loaded[name] for name in loaded.files}


def _reference_graph(reference: dict[str, np.ndarray]) -> sparse.csr_matrix:
    n_cells = len(reference["direction"])
    return sparse.csr_matrix(
        (
            reference["graph_data"],
            reference["graph_indices"],
            reference["graph_indptr"],
        ),
        shape=(n_cells, n_cells),
    )


def _matched(found: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair the states of two membership matrices by the correlation of their columns."""
    n_states = found.shape[1]
    correlation = np.corrcoef(found.T, expected.T)[:n_states, n_states:]
    rows, columns = linear_sum_assignment(-correlation)
    return rows, columns


def _crispness(memberships: np.ndarray) -> float:
    """``trace(diag(chi'1)^-1 chi'chi) / n_states``, from the memberships themselves."""
    return float(
        np.sum(np.diag(memberships.T @ memberships) / memberships.sum(axis=0))
        / memberships.shape[1]
    )


# Search steps, against values worked out by hand ---------------------------------


def test_two_joined_blocks_give_their_indicators_and_the_coarse_walk_worked_by_hand() -> (
    None
):
    # Two blocks of 30 cells: each cell links every other cell of its block
    # with weight 1, and each cell of the other block with weight 0.002 from
    # the first block and 0.001 from the second. One step leaves the first
    # block with probability e1 = 30 * 0.002 / (29 + 30 * 0.002), the second
    # with e2, so block-constant vectors are invariant under the walk, with
    # eigenvalues 1 and 1 - e1 - e2, and each block is a state.
    size = 30
    weights = _blocks([size, size], 1.0, [[0.0, 0.002], [0.001, 0.0]])
    e1 = size * 0.002 / (size - 1 + size * 0.002)
    e2 = size * 0.001 / (size - 1 + size * 0.001)
    walk = _row_normalized(weights)

    values, vectors = leading_eigenvectors(walk, 3)
    basis = invariant_basis(walk, values, vectors, 2)
    memberships = basis @ optimize_rotation(basis)
    coarse, walk_share = coarse_grain(walk, memberships)

    assert values[1].real == pytest.approx(1.0 - e1 - e2, abs=1e-12)
    first = np.r_[np.ones(size), np.zeros(size)]
    order = np.argsort(-memberships[0])
    np.testing.assert_allclose(
        memberships[:, order], np.c_[first, 1.0 - first], atol=1e-10
    )
    np.testing.assert_allclose(
        coarse[np.ix_(order, order)], [[1.0 - e1, e1], [e2, 1.0 - e2]], atol=1e-12
    )
    # The coarse walk's stationary shares, (e2, e1) / (e1 + e2).
    np.testing.assert_allclose(
        walk_share[order], np.array([e2, e1]) / (e1 + e2), atol=1e-12
    )


def test_three_blocks_in_a_row_put_the_start_first_and_the_end_last() -> None:
    # Blocks A, B and C in a row, scored 0, 0.5 and 1: the walk drains A and
    # B into C. The roles are checked against the walk's dense stationary
    # distribution and the memberships' own mean scores.
    size = 40
    weights = _blocks([size] * 3, 1.0, [[0, 0.02, 0], [0.02, 0, 0.02], [0, 0.02, 0]])
    rng = np.random.default_rng(0)
    direction = np.repeat([0.0, 0.5, 1.0], size) + rng.normal(scale=0.01, size=3 * size)

    found = find_states(sparse.csr_matrix(weights), direction, 3, cells_per_state=10)

    walk = _fate_walk(weights, direction)
    shares = found.memberships.T @ _stationary(walk)
    enrichment = shares / found.memberships.mean(axis=0)
    scores = (direction - direction.min()) / np.ptp(direction)
    levels = scores @ found.memberships / found.memberships.sum(axis=0)
    expected = ["initial"] + [
        "terminal" if value > 1.0 else "intermediate" for value in enrichment[1:]
    ]
    assert np.all(np.diff(levels) >= 0.0)
    assert list(found.roles) == expected == ["initial", "intermediate", "terminal"]
    # The most likely cells of each state lie in its block.
    for state in range(3):
        cells = np.flatnonzero(found.cells == state)
        assert len(cells) == 10
        assert set(cells // size) == {state}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_basis_spans_the_leading_subspace_of_the_dense_real_schur_form(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(60, 2))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    nearest = np.argsort(distances, axis=1)[:, 1:7]
    weights = np.zeros((60, 60))
    for cell, neighbors in enumerate(nearest):
        weights[cell, neighbors] = np.exp(-distances[cell, neighbors])
    weights = np.maximum(weights, weights.T)
    walk = _fate_walk(weights, points[:, 0])
    values, vectors = leading_eigenvectors(sparse.csr_matrix(walk), 5)
    n_states = 3 if not splits_conjugate_pair(values, 3) else 4
    threshold = 0.5 * (values[n_states - 1].real + values[n_states].real)

    basis = invariant_basis(sparse.csr_matrix(walk), values, vectors, n_states)

    form, vectors_dense, kept = schur(
        walk, output="real", sort=lambda re, _im: re > threshold
    )
    assert kept == n_states
    leading = vectors_dense[:, :n_states]
    orthonormal = basis / np.sqrt(60)
    np.testing.assert_allclose(
        orthonormal.T @ orthonormal, np.eye(n_states), atol=1e-12
    )
    assert np.all(basis[:, 0] == 1.0)
    residual = leading - orthonormal @ (orthonormal.T @ leading)
    assert np.linalg.norm(residual, 2) < 1e-9


def test_corner_rows_of_points_in_a_simplex_give_the_simplex_back() -> None:
    # Rows that are convex combinations of four corner rows, which are rows
    # too: the corner search finds the corners, and the initial guess maps
    # every row to its weights, the unique solution of Weber (2006),
    # Theorem 3.12.
    rng = np.random.default_rng(3)
    corners = np.c_[np.ones(4), rng.normal(size=(4, 3))]
    weights = np.vstack([np.eye(4), rng.dirichlet(np.ones(4), size=50)])
    rows = weights @ corners

    memberships = rows @ initial_rotation(rows)

    order = [int(np.argmax(memberships[corner])) for corner in range(4)]
    assert sorted(order) == [0, 1, 2, 3]
    np.testing.assert_allclose(memberships[:, order], weights, atol=1e-10)
    assert memberships.min() >= -1e-12


def test_the_feasible_rotation_makes_memberships_and_is_a_projection() -> None:
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(np.c_[np.ones(80), rng.normal(size=(80, 4))])
    basis = q * np.sqrt(80)
    basis[:, 0] = 1.0

    feasible = feasible_rotation(rng.normal(size=(5, 5)), basis)
    memberships = basis @ feasible

    np.testing.assert_allclose(memberships.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(memberships.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(feasible_rotation(feasible, basis), feasible, atol=1e-12)
    # The basis's columns after the first give the same rotation.
    np.testing.assert_allclose(
        feasible_rotation(feasible, np.ascontiguousarray(basis[:, 1:])), feasible
    )


def test_each_state_takes_its_most_likely_cells_and_a_shared_cell_goes_where_it_is_likelier() -> (
    None
):
    memberships = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.5, 0.5],
            [0.1, 0.9],
        ]
    )
    # With two cells each, state 0 takes cell 0 and the earlier of the cells
    # tied at 0.7, and state 1 takes cells 7 and 4.
    np.testing.assert_array_equal(
        state_cells(memberships, 2), [0, 0, -1, -1, 1, -1, -1, 1]
    )
    np.testing.assert_array_equal(
        state_cells(memberships, 3), [0, 0, 0, -1, 1, 1, -1, 1]
    )
    # With four each, state 1's fourth is cell 6, which no other state takes.
    np.testing.assert_array_equal(state_cells(memberships, 4), [0, 0, 0, 0, 1, 1, 1, 1])
    # With five, both take cell 3, likelier in state 0 (0.6 against 0.4), and
    # cell 6, a tie that goes to the earlier state.
    np.testing.assert_array_equal(state_cells(memberships, 5), [0, 0, 0, 0, 1, 1, 0, 1])
    # With three cells each, both states take cells 1 and 2, which go to the
    # later state, where each is likelier.
    shared = np.array([[0.9, 0.1], [0.45, 0.55], [0.1, 0.9], [0.0, 1.0]])
    np.testing.assert_array_equal(state_cells(shared, 3), [0, 1, 1, 1])


def test_states_take_the_label_most_of_their_cells_hold() -> None:
    cells = np.array([0, 0, 0, 1, 1, 2, 2, -1, 3])
    codes = np.array([0, 0, 1, 1, 1, 0, 1, 2, -1])
    # State 0 holds two "a" and one "b"; state 1 two "b"; state 2 ties "a"
    # and "b", the earlier label winning; state 3's cell has no label.
    assert state_names(cells, 4, codes, ["a", "b", "c"]) == ("a_1", "b", "a_2", "4")
    assert state_names(cells, 4) == ("1", "2", "3", "4")
    # A suffix that another state's name already takes is skipped.
    assert state_names(np.array([0, 1, 2]), 3, np.array([0, 0, 1]), ["x", "x_1"]) == (
        "x_2",
        "x_3",
        "x_1",
    )
    # A label of empty text names no state.
    assert state_names(np.array([0, 1]), 2, np.array([0, 1]), ["", "b"]) == ("1", "b")


def test_a_number_of_states_that_splits_a_complex_pair_is_refused_naming_its_neighbours() -> (
    None
):
    # Six blocks in a ring, each linking the next with weight 0.02 and the
    # previous with 0.005: the walk circulates, and the eigenvalues after one
    # are complex conjugate pairs.
    ring = [[0.0] * 6 for _ in range(6)]
    for block in range(6):
        ring[block][(block + 1) % 6] = 0.02
        ring[block][(block - 1) % 6] = 0.005
    walk = _row_normalized(_blocks([10] * 6, 1.0, ring))
    values, vectors = leading_eigenvectors(walk, 5)
    assert values[1] == pytest.approx(np.conj(values[2]))
    assert values[1].imag > 0.0

    with pytest.raises(ValueError, match=r"pass n_states=3$"):
        invariant_basis(walk, values, vectors, 2)
    with pytest.raises(ValueError, match=r"pass n_states=3 or n_states=5$"):
        invariant_basis(walk, values, vectors, 4)
    basis = invariant_basis(walk, values, vectors, 3)
    moved = walk @ basis
    np.testing.assert_allclose(moved, basis @ (basis.T @ moved / 60), atol=1e-10)


def test_eigenvectors_that_span_no_invariant_subspace_are_refused() -> None:
    weights = _blocks([20, 20, 20], 1.0, [[0, 0.01, 0], [0.01, 0, 0.01], [0, 0.01, 0]])
    walk = _row_normalized(weights)
    values, vectors = leading_eigenvectors(walk, 4)
    noise = np.random.default_rng(5).normal(size=vectors.shape)

    with pytest.raises(RuntimeError, match="span no invariant subspace"):
        invariant_basis(walk, values, vectors + 0.1 * noise, 3)


def test_a_walk_of_two_parts_that_never_meet_is_refused() -> None:
    weights = _blocks([40, 40], 1.0, [[0, 0], [0, 0]])
    direction = np.r_[np.zeros(40), np.ones(40)]

    with pytest.raises(RuntimeError, match="never meet"):
        find_states(sparse.csr_matrix(weights), direction, 2, cells_per_state=10)


@pytest.mark.parametrize("max_iterations", [1, 3])
def test_the_eigen_solve_stops_at_max_iterations_within_its_bound_of_products(
    monkeypatch: pytest.MonkeyPatch, max_iterations: int
) -> None:
    reference = _reference()
    graph = _reference_graph(reference)
    walk = _make_transition(
        graph.copy(),
        _normalize_pseudotime(reference["direction"]),
        np.zeros(graph.shape[0], dtype=bool),
        10.0,
    )
    solves: list[Any] = []

    class Recorded(states_module._SolveProgress):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            solves.append(self)

    monkeypatch.setattr(states_module, "_SolveProgress", Recorded)

    with pytest.raises(
        RuntimeError, match=f"did not converge within {max_iterations} restarts"
    ) as refused:
        leading_eigenvectors(walk, 7, max_iterations=max_iterations)

    (solve,) = solves
    assert f"after {solve.products} products" in str(refused.value)
    # SciPy's Arnoldi basis holds max(2 * 7 + 1, 20) vectors, which ARPACK
    # builds before its first restart; the bound that the progress reports
    # must hold every product that follows.
    assert 20 <= solve.products <= solve._limit[1]
    assert solve._limit == (max_iterations, 20 + 1 + max_iterations * (20 - 7))


def test_a_search_needs_enough_cells_and_a_direction_that_varies() -> None:
    weights = _blocks([20, 20], 1.0, [[0, 0.01], [0.01, 0]])
    with pytest.raises(ValueError, match="needs at least 60 connected cells"):
        find_states(sparse.csr_matrix(weights), np.arange(40.0), 2)
    with pytest.raises(ValueError, match="gives the walk no direction"):
        find_states(sparse.csr_matrix(weights), np.ones(40), 2, cells_per_state=5)
    with pytest.raises(ValueError, match="one score per cell"):
        find_states(sparse.csr_matrix(weights), np.arange(39.0), 2, cells_per_state=5)
    with pytest.raises(ValueError, match="finite score"):
        find_states(
            sparse.csr_matrix(weights),
            np.r_[np.nan, np.arange(39.0)],
            2,
            cells_per_state=5,
        )


# Against CellRank 2.4.0 and pyGPCCA 1.1.0 ---------------------------------------


def test_the_saved_walk_is_the_one_that_a_fate_map_takes() -> None:
    reference = _reference()
    graph = _reference_graph(reference)
    n_cells = graph.shape[0]
    walk = _make_transition(
        graph.copy(),
        _normalize_pseudotime(reference["direction"]),
        np.zeros(n_cells, dtype=bool),
        10.0,
    )
    expected = sparse.csr_matrix(
        (reference["walk_data"], reference["walk_indices"], reference["walk_indptr"]),
        shape=(n_cells, n_cells),
    )
    assert abs(walk - expected).max() == 0.0
    np.testing.assert_allclose(
        walk.toarray(), _fate_walk(graph.toarray(), reference["direction"]), atol=1e-15
    )


def test_eigenvalues_and_schur_vectors_match_pygpcca() -> None:
    reference = _reference()
    n_cells = len(reference["direction"])
    walk = sparse.csr_matrix(
        (reference["walk_data"], reference["walk_indices"], reference["walk_indptr"]),
        shape=(n_cells, n_cells),
    )

    values, vectors = leading_eigenvectors(walk, 8)
    basis = invariant_basis(walk, values, vectors, 6)

    np.testing.assert_allclose(values, reference["schur_eigenvalues"], atol=1e-10)
    expected = reference["schur_vectors"][:, :6]
    for column in range(6):
        assert (
            min(
                np.abs(basis[:, column] - expected[:, column]).max(),
                np.abs(basis[:, column] + expected[:, column]).max(),
            )
            < 1e-8
        )


@pytest.mark.parametrize("n_states", [2, 3, 4, 5, 6])
def test_memberships_and_coarse_walk_match_cellrank(n_states: int) -> None:
    reference = _reference()
    graph = _reference_graph(reference)
    record = json.loads(Path(f"{_REFERENCE}.json").read_text())
    assert record["environment"]["cellrank"] == "2.4.0"

    found = find_states(graph, reference["direction"], n_states)

    expected = reference[f"memberships_{n_states}"]
    rows, columns = _matched(found.memberships, expected)
    # Below six states both tools take one path; at six, Nelder-Mead parts.
    tolerance = 1e-10 if n_states < 6 else 1e-3
    np.testing.assert_allclose(
        found.memberships[:, rows], expected[:, columns], atol=tolerance
    )
    np.testing.assert_allclose(
        found.coarse_transitions[np.ix_(rows, rows)],
        reference[f"coarse_transitions_{n_states}"][np.ix_(columns, columns)],
        atol=tolerance * 1e-2,
    )
    np.testing.assert_allclose(
        found.walk_share[rows],
        reference[f"coarse_stationary_{n_states}"][columns],
        atol=tolerance,
    )
    assert found.crispness == pytest.approx(_crispness(expected), abs=1e-3)
    if n_states < 6:
        # CellRank's most likely cells of each state are the same cells.
        cellrank_cells = reference[f"macrostates_{n_states}"]
        for row, column in zip(rows, columns, strict=True):
            np.testing.assert_array_equal(
                np.flatnonzero(found.cells == row),
                np.flatnonzero(cellrank_cells == column),
            )


def test_five_states_of_the_saved_process_find_its_start_and_its_three_ends() -> None:
    reference = _reference()
    labels = reference["labels"].astype(str)

    found = find_states(_reference_graph(reference), reference["direction"], 5)

    codes, categories = pd.factorize(labels)
    names = state_names(found.cells, 5, codes, list(categories))
    roles = dict(zip(names, found.roles, strict=True))
    assert roles == {
        "start": "initial",
        "pocket": "intermediate",
        "end0": "terminal",
        "end1": "terminal",
        "end2": "terminal",
    }


def test_the_search_is_the_same_twice_and_for_cells_in_another_order() -> None:
    reference = _reference()
    graph = _reference_graph(reference)
    direction = reference["direction"]
    found = find_states(graph, direction, 4)
    again = find_states(graph, direction, 4)
    for name in ("memberships", "cells", "coarse_transitions", "walk_share"):
        np.testing.assert_array_equal(getattr(found, name), getattr(again, name))

    order = np.random.default_rng(8).permutation(graph.shape[0])
    shuffled = find_states(graph[order][:, order], direction[order], 4)

    np.testing.assert_allclose(
        shuffled.memberships, found.memberships[order], atol=1e-8
    )
    np.testing.assert_array_equal(shuffled.cells, found.cells[order])
    assert shuffled.roles == found.roles


# The operation -----------------------------------------------------------------

#: The saved process with an island of 12 cells, a component of its own.
_ISLAND = 12


def _process_store() -> tuple[Any, pd.Series, np.ndarray]:
    reference = _reference()
    graph = _reference_graph(reference).toarray()
    n_cells = graph.shape[0]
    island = np.ones((_ISLAND, _ISLAND)) - np.eye(_ISLAND)
    adjacency = np.zeros((n_cells + _ISLAND, n_cells + _ISLAND))
    adjacency[:n_cells, :n_cells] = graph
    adjacency[n_cells:, n_cells:] = island
    trajectory = _trajectory_store(adjacency)
    island_scores = np.linspace(0.0, 1.0, _ISLAND)
    direction = pd.Series(
        np.r_[reference["direction"], island_scores], index=trajectory.cell_ids
    )
    labels = np.r_[reference["labels"].astype(str), np.full(_ISLAND, "island")]
    return trajectory, direction, labels


def _states(trajectory: Any, direction: Any, **options: Any) -> ArtifactRef:
    options.setdefault("n_states", 5)
    return trajectory.store.trajectory.states(trajectory.graph, direction, **options)


def _state_artifacts(trajectory: Any) -> list[ArtifactRef]:
    return list_artifacts(
        trajectory.store.zw, scope="assay", assay="RNA", kind="trajectory_states"
    )


def test_the_search_names_the_known_start_and_ends_and_leaves_out_the_island() -> None:
    trajectory, direction, labels = _process_store()
    named = trajectory.labels(labels)

    with pytest.warns(DiagnosticWarning, match=f"left out {_ISLAND} cells"):
        ref = _states(trajectory, direction, labels=named)
    found = trajectory.store.trajectory.load_states(ref)

    assert isinstance(found, TrajectoryStatesResult)
    assert found.initial == ("start",)
    assert sorted(found.terminal) == ["end0", "end1", "end2"]
    assert found.table.loc["pocket", "role"] == "intermediate"
    assert list(found.table.index) == list(found.names)
    island = np.arange(len(labels)) >= len(labels) - _ISLAND
    assert not found.valid[island].any() and found.valid[~island].all()
    assert np.isnan(found.memberships[island]).all()
    assert found.labels[island].isna().all()
    np.testing.assert_allclose(found.memberships[~island].sum(axis=1), 1.0, atol=1e-6)
    # Each state's label marks its cells, which hold mostly its namesake's cells.
    for name in found.names:
        cells = np.asarray(found.labels == name)
        assert cells.sum() == found.table.loc[name, "cells"] == 30
        assert np.mean(labels[cells] == name) > 0.5
    table = found.table
    np.testing.assert_allclose(
        table["enrichment"], table["walk_share"] / table["cell_share"]
    )
    np.testing.assert_allclose(table["stability"], np.diag(found.coarse_transitions))
    assert found.labels_artifact == named
    assert found.graph == trajectory.graph
    assert 0.0 < found.crispness <= 1.0


def test_the_states_give_fate_its_sinks_as_the_same_labels_would() -> None:
    trajectory, direction, labels = _process_store()
    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction, labels=trajectory.labels(labels))
    found = trajectory.store.trajectory.load_states(ref)
    sinks = list(found.terminal)
    same_labels = trajectory.labels(
        np.where(found.labels.isna(), "", found.labels.astype(object)).astype(str)
    )

    from_states = trajectory.store.trajectory.load_fate(
        trajectory.store.trajectory.fate(
            sink_labels=ref, graph=trajectory.graph, direction=direction, sinks=sinks
        )
    )
    from_labels = trajectory.store.trajectory.load_fate(
        trajectory.store.trajectory.fate(
            sink_labels=same_labels,
            graph=trajectory.graph,
            direction=direction,
            sinks=sinks,
        )
    )

    np.testing.assert_array_equal(from_states.valid, from_labels.valid)
    np.testing.assert_array_equal(from_states.values, from_labels.values)
    # The initial state's cells are no sink, and walk mostly to the ends.
    start = np.asarray(found.labels == "start")
    assert from_states.valid[start].all()


def test_a_saved_search_is_reused_and_a_read_only_store_refuses_a_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, direction, _labels = _process_store()
    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction)
    searches: list[int] = []
    original = states_module.find_states
    import cytearc.datastore._operations.trajectory as operations

    def counted(*args: Any, **kwargs: Any) -> Any:
        searches.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "_find_states_impl", counted)

    # The reused search warns again, from its saved diagnostics.
    with pytest.warns(DiagnosticWarning, match=f"left out {_ISLAND} cells"):
        assert _states(trajectory, direction) == ref
    trajectory.store.zarr_mode = "r"
    with pytest.warns(DiagnosticWarning):
        assert _states(trajectory, direction) == ref
    with pytest.raises(PermissionError, match="find_trajectory_states"):
        _states(trajectory, direction, n_states=4)
    assert searches == []
    assert len(_state_artifacts(trajectory)) == 1


def test_a_series_and_its_snapshot_direct_one_search() -> None:
    trajectory, direction, _labels = _process_store()
    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction)
    record = trajectory.store.artifacts.inspect(ref)
    snapshot = ArtifactRef.from_dict(record.inputs["direction"])
    assert (
        trajectory.store.artifacts.inspect(snapshot).operation
        == "snapshot_fate_direction"
    )

    with pytest.warns(DiagnosticWarning):
        assert _states(trajectory, snapshot) == ref
    assert set(record.inputs) == {"connectivity_map", "direction", "cell_selection"}
    assert set(record.parameters) == {"n_states", "cells_per_state", "beta"}

    with pytest.warns(DiagnosticWarning):
        bounded = _states(trajectory, direction, max_iterations=500)
    assert bounded != ref
    assert (
        trajectory.store.artifacts.inspect(bounded).parameters["max_iterations"] == 500
    )


def test_a_walk_that_collects_in_its_initial_state_warns_that_it_found_no_terminal_state() -> (
    None
):
    # A dense clique of 40 cells scored low and a path of 40 cells scored
    # high, joined by one edge. Without a bias, beta=0, a long walk spends
    # its time in proportion to each cell's degree, so it collects in the
    # clique, which the score places first.
    clique = np.ones((40, 40)) - np.eye(40)
    path = np.diag(np.ones(39), 1)
    adjacency = np.zeros((80, 80))
    adjacency[:40, :40] = clique
    adjacency[40:, 40:] = path + path.T
    adjacency[39, 40] = adjacency[40, 39] = 1.0
    trajectory = _trajectory_store(adjacency)
    direction = pd.Series(
        np.r_[np.zeros(40), np.linspace(0.5, 1.0, 40)], index=trajectory.cell_ids
    )
    degree = adjacency.sum(axis=1)
    assert degree[40:].sum() / degree.sum() < 0.5

    with pytest.warns(
        DiagnosticWarning, match="no terminal state apart from the initial one"
    ):
        ref = _states(trajectory, direction, n_states=2, beta=0.0)
    found = trajectory.store.trajectory.load_states(ref)

    assert list(found.table["role"]) == ["initial", "intermediate"]
    assert found.terminal == ()
    assert found.table["enrichment"].iloc[0] > 1.0 > found.table["enrichment"].iloc[1]
    with pytest.warns(DiagnosticWarning, match="no terminal state"):
        assert _states(trajectory, direction, n_states=2, beta=0.0) == ref


#: A path of 1,500 cells has few edges, so 16 states need more than its graph:
#: ARPACK's 2 * 17 + 1 Arnoldi vectors and the memberships hold at least 8
#: bytes a cell for each.
_PATH_CELLS, _PATH_STATES = 1500, 16
_PATH_LEAST = 8 * _PATH_CELLS * (2 * (_PATH_STATES + 1) + 1 + _PATH_STATES)


def _path_store() -> tuple[Any, pd.Series]:
    path = np.diag(np.ones(_PATH_CELLS - 1), 1)
    trajectory = _trajectory_store(path + path.T)
    return trajectory, pd.Series(
        np.linspace(0.0, 1.0, _PATH_CELLS), index=trajectory.cell_ids
    )


def test_a_search_over_the_memory_budget_is_refused_before_the_graph_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, direction = _path_store()
    loads = _count_calls(monkeypatch, trajectory.store, "_load_graph_artifact")
    _set_budget(trajectory.store, _PATH_LEAST - 1)

    with pytest.raises(MemoryBudgetError, match="Finding states") as refused:
        _states(trajectory, direction, n_states=_PATH_STATES)

    assert refused.value.item == "the state search"
    assert refused.value.needed_bytes >= _PATH_LEAST
    assert loads["_load_graph_artifact"] == 0
    assert _state_artifacts(trajectory) == []


@pytest.mark.slow
def test_the_budget_that_a_refusal_names_runs_the_search() -> None:
    trajectory, direction = _path_store()
    _set_budget(trajectory.store, _PATH_LEAST - 1)
    with pytest.raises(MemoryBudgetError) as refused:
        _states(trajectory, direction, n_states=_PATH_STATES)

    _set_budget(trajectory.store, refused.value.needed_bytes)
    _states(trajectory, direction, n_states=_PATH_STATES)

    assert len(_state_artifacts(trajectory)) == 1


def test_a_subspace_that_the_eigen_solve_misses_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, direction, _labels = _process_store()
    original = states_module.leading_eigenvectors

    def noisy(walk: Any, n_vectors: int, **options: Any) -> Any:
        values, vectors = original(walk, n_vectors, **options)
        return values, vectors + 0.1 * np.random.default_rng(9).normal(
            size=vectors.shape
        )

    monkeypatch.setattr(states_module, "leading_eigenvectors", noisy)

    with pytest.raises(RuntimeError, match="span no invariant subspace"):
        _states(trajectory, direction)
    assert _state_artifacts(trajectory) == []


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"n_states": 1}, ValueError, "n_states must be at least 2"),
        ({"n_states": 2.0}, TypeError, "n_states must be an integer"),
        ({"cells_per_state": 0}, ValueError, "cells_per_state must be at least 1"),
        ({"beta": -1.0}, ValueError, "beta must be at least 0"),
        ({"max_iterations": 0}, ValueError, "max_iterations must be at least 1"),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache"),
        ({"n_states": 17}, ValueError, "needs at least 510 connected cells"),
    ],
)
def test_arguments_out_of_range_are_refused(
    options: dict[str, Any], error: type[Exception], message: str
) -> None:
    trajectory, direction, _labels = _process_store()
    with pytest.raises(error, match=message):
        _states(trajectory, direction, **options)
    assert _state_artifacts(trajectory) == []


def test_inputs_of_the_wrong_kind_are_refused() -> None:
    trajectory, direction, labels = _process_store()
    store = trajectory.store
    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        store.trajectory.states("graph", direction, n_states=5)
    with pytest.raises(TypeError, match="keyed by cell ID"):
        store.trajectory.states(trajectory.graph, direction.to_numpy(), n_states=5)
    pseudotime = ArtifactRef(
        scope="assay", assay="RNA", kind="pseudotime", artifact_id="0" * 64
    )
    with pytest.raises(ValueError, match="A pseudotime cannot direct the state search"):
        store.trajectory.states(trajectory.graph, pseudotime, n_states=5)
    with pytest.raises(ValueError, match="holds real values"):
        _states(
            trajectory,
            direction,
            labels=trajectory.labels(np.arange(len(labels)) / 7.0),
        )
    other_cells = trajectory.labels(labels, selection=trajectory.other_selection)
    with pytest.raises(ValueError, match="labels must hold the cells of the graph"):
        _states(trajectory, direction, labels=other_cells)
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        _states(trajectory, direction, labels="clusters")


def test_the_loader_refuses_other_artifacts_and_changed_arrays() -> None:
    trajectory, direction, _labels = _process_store()
    store = trajectory.store
    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction)
    with pytest.raises(TypeError, match="states must be an ArtifactRef"):
        store.trajectory.load_states("states")
    with pytest.raises(ValueError, match="trajectory_states artifact"):
        store.trajectory.load_states(trajectory.graph)

    group = artifact_group(store.zw, ref)
    memberships = group["memberships"]
    changed = np.asarray(memberships[:])
    changed[0, 0] += 0.25
    memberships[:] = changed

    with pytest.raises(ValueError, match="do not match their contract"):
        store.trajectory.load_states(ref)


def test_labels_of_several_values_a_cell_and_cells_without_labels() -> None:
    trajectory, direction, _labels = _process_store()
    from .test_trajectory_record_paths import _embedding

    with pytest.raises(ValueError, match="labels must hold one label per cell"):
        _states(trajectory, direction, labels=_embedding(trajectory))
    unlabeled = trajectory.labels(np.full(len(trajectory.cell_ids), None, dtype=object))
    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction, labels=unlabeled)
    # No cell holds a label, so the states take their numbers.
    assert trajectory.store.trajectory.load_states(ref).names == (
        "1",
        "2",
        "3",
        "4",
        "5",
    )


def test_a_graph_without_cells_and_a_search_that_does_not_exist_are_refused() -> None:
    from .test_trajectory_record_paths import _empty_graph_payload

    trajectory, direction, _labels = _process_store()
    missing = ArtifactRef(
        scope="assay", assay="RNA", kind="trajectory_states", artifact_id="1" * 64
    )
    with pytest.raises(ValueError, match="unavailable or invalid"):
        trajectory.store.trajectory.load_states(missing)
    _empty_graph_payload(trajectory)
    with pytest.raises(ValueError, match="holds no cells"):
        _states(trajectory, direction)


def _tampered(name: str) -> Any:
    """A change to a stored search that breaks one rule of its contract."""

    def change(group: Any) -> None:
        if name == "attribute":
            group.attrs["note"] = "added"
        elif name == "array":
            group.create_array("extra", data=np.zeros(3))
        elif name == "dtype":
            values = np.asarray(group["cells"][:])
            del group["cells"]
            group.create_array("cells", data=values.astype(np.float64))
        elif name == "link":
            group["values"].attrs.clear()
        elif name == "names":
            names = np.asarray(group["state_names"][:])
            names[1] = names[0]
            group["state_names"][:] = names
        elif name == "roles":
            roles = np.asarray(group["roles"][:]).astype("U12")
            roles[0] = "terminal"
            del group["roles"]
            group.create_array("roles", data=roles)
        elif name == "none valid":
            group["valid"][:] = False
        elif name == "not finite":
            values = np.asarray(group["memberships"][:])
            values[np.flatnonzero(group["valid"][:])[0], 0] = np.nan
            group["memberships"][:] = values
        elif name == "row sum":
            values = np.asarray(group["memberships"][:])
            values[np.flatnonzero(group["valid"][:])[0]] *= 2.0
            group["memberships"][:] = values
        elif name == "outside":
            values = np.asarray(group["memberships"][:])
            values[np.flatnonzero(~np.asarray(group["valid"][:]))[0]] = 0.5
            group["memberships"][:] = values
        elif name == "label outside":
            # A cell outside the search that names a real state.
            outside = np.flatnonzero(~np.asarray(group["valid"][:]))[0]
            missing = np.asarray(group[_LABELS_MISSING][:])
            missing[outside] = False
            group[_LABELS_MISSING][:] = missing
            values = np.asarray(group["values"][:])
            values[outside] = np.asarray(group["state_names"][:])[0]
            group["values"][:] = values
        elif name == "unknown label":
            values = np.asarray(group["values"][:])
            values[np.flatnonzero(~np.asarray(group[_LABELS_MISSING][:]))[0]] = "other"
            group["values"][:] = values
        else:  # pragma: no cover
            raise AssertionError(name)

    return change


_LABELS_MISSING = "__cytearc_missing__values"


@pytest.mark.parametrize(
    "change",
    [
        "attribute",
        "array",
        "dtype",
        "link",
        "names",
        "roles",
        "none valid",
        "not finite",
        "row sum",
        "outside",
        "label outside",
        "unknown label",
    ],
)
def test_a_stored_search_that_breaks_its_contract_is_refused(change: str) -> None:
    from cytearc.storage.artifacts import fingerprint_stored_arrays
    from cytearc.trajectory.artifacts import STATES_PAYLOAD, states_payload_is_valid

    trajectory, direction, _labels = _process_store()
    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction)
    group = artifact_group(trajectory.store.zw, ref)
    n_cells = len(trajectory.cell_ids)
    valid = np.asarray(group["valid"][:])
    assert states_payload_is_valid(
        group, n_cells=n_cells, n_states=5, expected_valid=valid
    )
    assert not states_payload_is_valid(
        group, n_cells=n_cells, n_states=5, expected_valid=~valid
    )

    _tampered(change)(group)
    if change != "array":
        # The rule, not the fingerprint, must refuse the change.
        group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
            group, STATES_PAYLOAD
        )

    assert not states_payload_is_valid(group, n_cells=n_cells, n_states=5)
    with pytest.raises(ValueError, match="do not match their contract"):
        trajectory.store.trajectory.load_states(ref)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("names", ("only",), "at least two distinct states"),
        ("valid", np.zeros((2, 2), dtype=bool), "validity must be one-dimensional"),
        ("memberships", np.zeros((3, 2)), "cells by states"),
        ("coarse_transitions", np.zeros((2, 3)), "states by states"),
        ("eigenvalues", np.zeros(2), "n_states \\+ 1 eigenvalues"),
    ],
)
def test_a_states_record_refuses_fields_that_do_not_fit(
    field: str, value: Any, message: str
) -> None:
    import dataclasses

    trajectory, direction, _labels = _process_store()
    with pytest.warns(DiagnosticWarning):
        found = trajectory.store.trajectory.load_states(
            _states(trajectory, direction, n_states=2)
        )
    assert found.memberships.flags.writeable is False

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(found, **{field: value})
    with pytest.raises(ValueError, match="name a state of each labeled cell"):
        dataclasses.replace(found, labels=found.labels.rename_categories(["x", "y"]))
    with pytest.raises(ValueError, match="one row per state"):
        dataclasses.replace(found, table=found.table.iloc[::-1])


def test_the_search_reports_its_products_and_evaluations_while_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from loguru import logger

    import cytearc.trajectory.pseudotime as pseudotime_module

    monkeypatch.setattr(pseudotime_module, "_PROGRESS_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(pseudotime_module, "_BAR_INTERVAL_SECONDS", 0.0)
    reference = _reference()
    graph = _reference_graph(reference)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
        filter=lambda record: record["message"].startswith("Finding states: "),
    )
    try:
        find_states(graph, reference["direction"], 3)
        find_states(graph, reference["direction"], 3, max_iterations=50)
    finally:
        logger.remove(sink)

    solves = [message for message in messages if "operator products" in message]
    evaluations = [
        message for message in messages if "objective evaluations" in message
    ]
    assert any(
        message.endswith("set max_iterations to bound the solve") for message in solves
    )
    assert any(
        " of at most " in message and message.endswith("(max_iterations=50)")
        for message in solves
    )
    assert evaluations and all(
        " of at most 800 objective evaluations" in message
        and message.endswith("solving the memberships of 3 states of 480 cells")
        for message in evaluations
    )


def test_a_matrix_whose_rows_do_not_sum_to_one_is_refused() -> None:
    weights = _blocks([20, 20], 1.0, [[0, 0.01], [0.01, 0]])
    walk = _row_normalized(weights) * 0.9
    values, vectors = leading_eigenvectors(walk, 3)

    with pytest.raises(RuntimeError, match="rows do not sum to one"):
        invariant_basis(walk, values, vectors, 2)


def test_a_basis_whose_rows_span_fewer_states_is_refused() -> None:
    rng = np.random.default_rng(6)
    column = rng.normal(size=50)
    with pytest.raises(RuntimeError, match="do not span 3 states"):
        initial_rotation(np.c_[np.ones(50), column, column])


def test_a_trial_that_leaves_a_state_without_membership_is_the_worst() -> None:
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(np.c_[np.ones(60), rng.normal(size=(60, 2))])
    tail = np.ascontiguousarray(q[:, 1:] * np.sqrt(60))
    start = np.eye(3)
    # No free entry at all leaves every membership where the first row puts it.
    assert states_module._trial_gap(np.zeros(4), start, tail) == np.inf
    # A state whose free entries are zero has no membership anywhere.
    free = rng.normal(size=(2, 2))
    free[:, 1] = 0.0
    assert states_module._trial_gap(free.ravel(), start, tail) == np.inf
    finite = states_module._trial_gap(rng.normal(size=4), start, tail)
    assert 0.0 <= finite < 3.0


def test_states_take_the_text_of_number_labels() -> None:
    trajectory, direction, labels = _process_store()
    numbers = np.unique(labels, return_inverse=True)[1]

    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction, labels=trajectory.labels(numbers))

    found = trajectory.store.trajectory.load_states(ref)
    expected = {str(code) for code in np.unique(numbers)}
    assert all(name.split("_")[0] in expected for name in found.names)


@pytest.mark.parametrize(
    ("n_cells", "n_states"), [(1_000, 2), (1_000, 12), (1_000_000, 12)]
)
def test_the_search_charges_at_least_its_arnoldi_vectors_and_its_names(
    n_cells: int, n_states: int
) -> None:
    graph_bytes, n_entries, name_width = 30 * n_cells, 20 * n_cells, 14
    ncv = max(2 * (n_states + 1) + 1, 20)
    charged = search_bytes(n_cells, graph_bytes, n_entries, n_states, name_width)
    # The graph, ARPACK's basis and the n_states + 1 Ritz vectors that it
    # returns, then the names of up to name_width characters of every cell.
    assert charged >= graph_bytes + 8 * n_cells * (ncv + n_states + 1)
    assert charged >= graph_bytes + 4 * n_cells * name_width
    # Wider names never charge less, and very wide ones take the lead.
    wider = search_bytes(n_cells, graph_bytes, n_entries, n_states, 2 * name_width)
    assert wider >= charged
    widest = search_bytes(n_cells, graph_bytes, n_entries, n_states, 1000)
    assert widest >= graph_bytes + 4 * n_cells * 1000


def test_cells_without_labels_do_not_name_a_state() -> None:
    trajectory, direction, labels = _process_store()
    # A fifth of the cells keep their population, the others hold empty text,
    # as a snapshot stores missing labels: each state takes the population
    # most of its labeled cells hold.
    kept = np.random.default_rng(10).random(len(labels)) < 0.2
    sparse_labels = np.where(kept, labels, "")

    with pytest.warns(DiagnosticWarning):
        ref = _states(trajectory, direction, labels=trajectory.labels(sparse_labels))

    found = trajectory.store.trajectory.load_states(ref)
    assert sorted(found.names) == ["end0", "end1", "end2", "pocket", "start"]


def test_the_eigen_solve_dominates_the_charge_of_a_sparse_graph_with_few_states() -> (
    None
):
    n_cells, n_states, n_entries = 1_000, 2, 2_000
    graph_bytes = 12 * n_entries
    ncv, k = 20, n_states + 1
    # The walk, ARPACK's basis, its real Ritz vectors and the complex
    # eigenvectors that eigs returns are held at once.
    eigen_solve = (
        graph_bytes + 12 * n_entries + 8 * n_cells * (ncv + k + 1) + 16 * n_cells * k
    )
    assert search_bytes(n_cells, graph_bytes, n_entries, n_states, 1) >= eigen_solve
