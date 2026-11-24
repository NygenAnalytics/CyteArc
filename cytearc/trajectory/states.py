"""Find the initial and terminal states of a walk on a cell graph.

The walk is the one that a fate map takes: the graph's edges, an edge to a
cell of lower direction score keeping ``2 / (1 + exp(beta * d))`` of its
weight, rows normalized. The search splits it into metastable states by
generalized robust Perron cluster analysis (G-PCCA), which uses real Schur
vectors instead of eigenvectors so that it handles walks that are not
reversible (Reuter et al. (2018), Journal of Chemical Theory and Computation
14:3579-3594, doi:10.1021/acs.jctc.8b00079; Reuter, Fackeldey and Weber
(2019), Journal of Chemical Physics 150:174103, doi:10.1063/1.5064530), as
CellRank applies it to cells (Lange et al. (2022), Nature Methods 19:159-170,
doi:10.1038/s41592-021-01346-6):

1. ARPACK's restarted Arnoldi iteration finds the leading eigenvalues of the
   walk ``P``, those of largest real part, which recover metastable states.
2. The constant vector and the real and imaginary parts of the eigenvectors,
   orthonormalized in order, are the real Schur vectors of the leading part
   of ``P``. Scaled to ``X``, they are orthonormal for the uniform weights
   ``D = I / n``, ``X'DX = I``, and their first column is constant.
3. The memberships are ``chi = X A``. Weber's robust Perron cluster analysis
   (Weber (2006), Meshless Methods in Conformation Dynamics, doctoral
   thesis, Freie Universität Berlin, Algorithms 3.8 to 3.11) guesses ``A``
   from the rows of ``X`` nearest the corners of the simplex that holds
   them, makes any ``A`` feasible (memberships nonnegative and summing to
   one in each cell), and optimizes the free entries of ``A`` by
   Nelder-Mead. The objective is crispness, ``trace(S) / n_states`` with
   ``S = diag(chi'D1)^-1 chi'D chi``, which for this basis is
   ``sum_ij A_ij^2 / A_0j / n_states`` (Lange et al., Methods).
4. The coarse-grained walk between the states is
   ``P_c = (chi'D chi)^-1 (chi'D P chi)``. Its left eigenvector for the
   eigenvalue one is ``chi' pi``, each state's share of a long walk's time,
   since ``chi`` spans an invariant subspace of ``P``.

Each state's ``cells_per_state`` most likely cells make up its cells, a cell
taken by several states going to the one in which its membership is largest,
as CellRank 2.4.0 selects them (BSD-3-Clause). States are ordered by their
mean direction score, earliest first. The first is the initial state, and
each other state in which a long walk spends a larger share of its time than
the state's share of the cells is a terminal state.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import fmin
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from ..neighbors.graph import _sparse_index_bytes
from ..utils.compute import process_thread_limit
from .fate import _make_transition, _normalize_pseudotime
from .pseudotime import _counted_operator, _SolveProgress

__all__ = [
    "TIE_TOLERANCE",
    "FoundStates",
    "boundary_gap",
    "coarse_grain",
    "feasible_rotation",
    "find_states",
    "initial_rotation",
    "invariant_basis",
    "leading_eigenvectors",
    "optimize_rotation",
    "search_bytes",
    "splits_conjugate_pair",
    "state_cells",
    "state_labels",
    "state_names",
    "trial_limit",
    "vertex_rows",
]

#: The eigen-solve's tolerance, SciPy's default for ARPACK: machine precision.
_EIGEN_TOLERANCE = 0.0
#: The seed of the eigen-solve's start vector, fixed so that a search does
#: not depend on the start that ARPACK's own generator would draw.
_START_SEED = 211
#: The largest residual ``max |P X - X R|``, relative to ``max |X|``, of a
#: basis that the search accepts as an invariant subspace of the walk.
_INVARIANCE_TOLERANCE = 1e-6
#: How far below one the walk's second eigenvalue must lie: a walk of one
#: connected component has a single eigenvalue one.
_SEPARATION_TOLERANCE = 1e-12
#: The smallest squared distance from the span of the corners found so far,
#: in units of the last corner's length, at which a row is a new corner.
_SPAN_TOLERANCE = 1e-24
#: What a search holds whatever its cells and states: NumPy's buffers for
#: reductions and the small arrays of each step, measured at under 0.2 MiB.
_SMALL_ARRAY_BYTES = 2**20
#: How close the last eigenvalue kept and the first one left out may come,
#: relative to the latter's distance from one, before they count as tied.
#: Identical parts of a walk tie them to 1e-9; on the pancreas walks and the
#: synthetic processes of the tests the closest pair stayed 7e-3 apart.
TIE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class FoundStates:
    """The states of a walk, ordered by their mean direction score.

    ``memberships`` holds each cell's membership in each state, cells by
    states, each row summing to one. ``cells`` gives the state whose most
    likely cells each cell is among, or -1. ``coarse_transitions`` is the
    walk between the states, ``cell_share`` each state's mean membership,
    ``walk_share`` its share of a long walk's time, ``direction`` its mean
    direction score rescaled to [0, 1], and ``roles`` its role.
    ``eigenvalues`` holds the walk's ``n_states + 1`` leading eigenvalues,
    largest real part first. ``optimization_converged`` is False when
    Nelder-Mead stopped at its limit of :func:`trial_limit` trials before
    the crispness converged.
    """

    memberships: np.ndarray = field(repr=False)
    cells: np.ndarray = field(repr=False)
    coarse_transitions: np.ndarray
    cell_share: np.ndarray
    walk_share: np.ndarray
    direction: np.ndarray
    roles: tuple[str, ...]
    eigenvalues: np.ndarray
    crispness: float
    optimization_converged: bool


def splits_conjugate_pair(eigenvalues: np.ndarray, n_states: int) -> bool:
    """Whether keeping the first ``n_states`` eigenvalues splits a complex conjugate pair.

    The eigenvalues are sorted by decreasing real part, the member of a pair
    with positive imaginary part first, so a pair splits exactly when the
    last eigenvalue kept is such a first member.
    """
    return bool(eigenvalues[n_states - 1].imag > 0.0)


def boundary_gap(eigenvalues: np.ndarray, n_states: int) -> float:
    """How far apart the last eigenvalue kept and the first one left out lie, relative to the latter's distance from one.

    When they tie, as the eigenvalues of identical parts of a walk do, any
    mix of their two eigenvectors spans an invariant subspace, so the last
    state is one choice among many; a gap below :data:`TIE_TOLERANCE`
    counts as a tie.
    """
    kept, left_out = eigenvalues[n_states - 1], eigenvalues[n_states]
    return float(abs(kept - left_out) / abs(1.0 - left_out))


def leading_eigenvectors(
    walk: csr_matrix, n_vectors: int, *, max_iterations: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the walk's ``n_vectors`` eigenvalues of largest real part and their eigenvectors.

    SciPy's ``eigs`` runs ARPACK's implicitly restarted Arnoldi iteration
    from a fixed start vector, to machine precision. The eigenvalues are
    sorted by decreasing real part, and a complex conjugate pair by
    decreasing imaginary part. The solve advances a progress bar with its
    products of the walk and logs their count at INFO level every 30
    seconds.

    Raises:
        RuntimeError: If the solve reaches ``max_iterations`` restarts
            before every eigenvalue converges.
    """
    n_cells = int(walk.shape[0])
    ncv = min(n_cells, max(2 * n_vectors + 1, 20))
    # ARPACK builds its first basis of ncv vectors with one product more,
    # and each restart adds at most ncv - n_vectors products.
    limit = (
        None
        if max_iterations is None
        else (max_iterations, ncv + 1 + max_iterations * (ncv - n_vectors))
    )
    progress = _SolveProgress(
        n_cells,
        f"for the {n_vectors} leading eigenvalues of the walk",
        limit,
        title="State search",
        call="Finding states",
    )
    start = np.random.default_rng(_START_SEED).standard_normal(n_cells)
    converged = False
    try:
        values, vectors = eigs(
            _counted_operator(walk, progress),
            k=n_vectors,
            which="LR",
            v0=start,
            ncv=ncv,
            maxiter=max_iterations,
            tol=_EIGEN_TOLERANCE,
        )
        converged = True
    except ArpackNoConvergence as exc:
        bound = 10 * n_cells if max_iterations is None else max_iterations
        raise RuntimeError(
            f"The state search's eigen-solve did not converge within {bound} "
            f"restarts: {len(exc.eigenvalues)} of the {n_vectors} leading "
            f"eigenvalues of the walk of {n_cells} cells converged after "
            f"{progress.products} products. Raise max_iterations, or leave it "
            "unset"
        ) from exc
    finally:
        progress.close(converged)
    order = np.lexsort((-values.imag, -values.real))
    return values[order], vectors[:, order]


def invariant_basis(
    walk: csr_matrix,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Return an orthonormal basis of the walk's leading invariant subspace, constant first.

    The basis holds the constant vector, then each real eigenvector and the
    real and imaginary parts of each complex one, orthonormalized in order:
    the real Schur vectors of the walk's ``n_states`` leading eigenvalues.
    Scaled by ``sqrt(n)``, the columns are orthonormal for the uniform
    weights ``D = I / n``, and the first is one in every cell.

    Raises:
        ValueError: If ``n_states`` would split a complex conjugate pair of
            eigenvalues, which no real subspace can do.
        RuntimeError: If the walk's second eigenvalue rounds to one, as for
            parts joined too weakly for the walk to cross, or the basis is
            not invariant under the walk.
    """
    n_cells = int(walk.shape[0])
    if eigenvalues[1].real >= 1.0 - _SEPARATION_TOLERANCE:
        raise RuntimeError(
            f"The walk's second eigenvalue, {eigenvalues[1].real:.15g}, rounds "
            "to one: the graph joins parts of its cells too weakly for the walk "
            "to cross between them, so no split of the walk weighs one part "
            "against another. Build the graph so that its parts join more "
            "strongly, such as with more neighbors, or search each part on a "
            "graph of its own cells"
        )
    if splits_conjugate_pair(eigenvalues, n_states):
        choices = (
            f"n_states={n_states - 1} or n_states={n_states + 1}"
            if n_states > 2
            else f"n_states={n_states + 1}"
        )
        raise ValueError(
            f"n_states={n_states} would split the complex conjugate pair of "
            f"eigenvalues {eigenvalues[n_states - 1]:.6g} and "
            f"{np.conj(eigenvalues[n_states - 1]):.6g}, whose two directions "
            f"only together span a real invariant subspace of the walk; pass "
            f"{choices}"
        )
    columns = [np.ones(n_cells)]
    index = 1
    while index < n_states:
        vector = eigenvectors[:, index]
        if eigenvalues[index].imag != 0.0:
            columns.append(vector.real)
            columns.append(vector.imag)
            index += 2
        else:
            columns.append(vector.real)
            index += 1
    q, _ = np.linalg.qr(np.column_stack(columns))
    del columns
    basis = q * math.sqrt(n_cells)
    del q
    # The first column is the constant vector, whatever sign QR gave it.
    basis[:, 0] = 1.0
    moved = walk @ basis
    schur = (basis.T @ moved) / n_cells
    residual = float(np.max(np.abs(moved - basis @ schur)))
    scale = max(1.0, float(np.max(np.abs(basis))))
    if not residual <= _INVARIANCE_TOLERANCE * scale:
        raise RuntimeError(
            "The eigenvectors that the state search found span no invariant "
            f"subspace of the walk (largest residual {residual:.2e}, at most "
            f"{_INVARIANCE_TOLERANCE * scale:.2e} allowed), so its eigenvectors "
            "could not be computed accurately enough; ask for another n_states, "
            "or pass a lower beta, which directs the walk less strongly"
        )
    return basis


def vertex_rows(basis: np.ndarray) -> np.ndarray:
    """Return rows of the basis that approximate the corners of the simplex holding its rows.

    Weber (2006), Algorithm 3.8: take the row of largest norm, move every
    row so that it becomes the origin, then repeatedly take the row farthest
    from the span of the rows taken so far, rescaled to unit length, and
    remove its direction from every row. A tie goes to the earlier row.
    """
    points = np.array(basis, dtype=np.float64, copy=True)
    n_states = points.shape[1]
    rows = np.empty(n_states, dtype=np.int64)
    norms = np.einsum("ij,ij->i", points, points)
    rows[0] = int(np.argmax(norms))
    points -= points[rows[0]]
    for state in range(1, n_states):
        norms = np.einsum("ij,ij->i", points, points)
        rows[state] = int(np.argmax(norms))
        largest = float(norms[rows[state]])
        # The rows are scaled so that the last corner has unit length; a
        # distance below 1e-12 of that from the corners' span is rounding.
        if not largest > _SPAN_TOLERANCE:
            raise RuntimeError(
                "The cells' coordinates in the walk's invariant subspace do not "
                f"span {n_states} states; ask for fewer states"
            )
        points /= math.sqrt(largest)
        unit = points[rows[state]].copy()
        points -= np.outer(points @ unit, unit)
    return rows


def initial_rotation(basis: np.ndarray) -> np.ndarray:
    """Return Weber's (2006) Algorithm 3.9 guess of the rotation, the inverse of the corner rows.

    The corner search finds rows that no earlier corner's span holds, so
    they are independent.
    """
    return np.linalg.inv(basis[vertex_rows(basis)])


def feasible_rotation(rotation: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return the feasible rotation that Weber's (2006) Algorithm 3.10 makes of ``rotation``.

    The first column makes every cell's memberships sum to one, the first
    row makes every state's smallest membership zero, and dividing by the
    first row's sum makes the memberships sum to one exactly. Only the
    entries outside the first row and column of ``rotation`` matter.
    ``basis`` may be the whole basis or its columns after the first.
    """
    tail = basis if basis.shape[1] == rotation.shape[0] - 1 else basis[:, 1:]
    feasible = np.array(rotation, dtype=np.float64, copy=True)
    feasible[1:, 0] = -feasible[1:, 1:].sum(axis=1)
    feasible[0, :] = -(tail @ feasible[1:, :]).min(axis=0)
    feasible /= feasible[0, :].sum()
    return feasible


def _trial_gap(free: np.ndarray, start: np.ndarray, tail: np.ndarray) -> float:
    """Return ``n_states - trace(S)`` of the feasible rotation of a trial's free entries.

    ``free`` holds the entries outside the first row and column, which
    Weber's (2006) Algorithm 3.10 completes, and the objective of a trial
    that leaves a state without any membership is infinite.
    """
    n_states = start.shape[0]
    trial = start.copy()
    trial[1:, 1:] = free.reshape(n_states - 1, n_states - 1)
    trial[1:, 0] = -trial[1:, 1:].sum(axis=1)
    trial[0, :] = -(tail @ trial[1:, :]).min(axis=0)
    total = float(trial[0, :].sum())
    if not total > 0.0:
        return math.inf
    trial /= total
    first = trial[0, :]
    if not bool(np.all(first > 0.0)):
        return math.inf
    return float(n_states - np.sum(trial**2 / first[np.newaxis, :]))


def trial_limit(n_states: int) -> int:
    """Return the most trials that Nelder-Mead evaluates for ``n_states`` states.

    SciPy's ``fmin`` allows by default 200 evaluations of the objective, and
    as many iterations, for each of the ``(n_states - 1)**2`` free entries
    of the rotation.
    """
    return 200 * (n_states - 1) ** 2


def optimize_rotation(basis: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the feasible rotation of the basis that Nelder-Mead finds most crisp, and whether it converged.

    Weber (2006), Algorithm 3.11, as Lange et al. (2022) run it: SciPy's
    ``fmin`` with its defaults optimizes the entries of the rotation outside
    its first row and column, making each trial feasible before evaluating
    it, and stops after at most :func:`trial_limit` evaluations, the default
    limit, which the call passes explicitly. It starts from the entries of
    Algorithm 3.9's guess. Algorithm 3.11 starts from the
    feasible guess, whose entries Algorithm 3.10 divides by the sum of its
    first row; the objective does not depend on that scale, but ``fmin``'s
    absolute tolerances do, and the guess before division gives CellRank
    2.4.0's memberships. Two states have one free entry, which only scales
    the rotation and so leaves the objective unchanged, so their feasible
    guess is returned, converged. The optimization advances a progress bar
    with its evaluations.

    The second value is False when ``fmin`` stopped at its limit of
    evaluations or of iterations before its simplex converged. An iteration
    evaluates at least one trial beside the first simplex's, so the
    evaluations reach their limit first.
    """
    n_states = basis.shape[1]
    # Each trial reads the basis's columns after the first, held contiguous
    # once rather than copied by every product.
    tail = np.ascontiguousarray(basis[:, 1:])
    guess = initial_rotation(basis)
    start = feasible_rotation(guess, tail)
    if n_states == 2:
        return start, True
    most = trial_limit(n_states)
    progress = _SolveProgress(
        basis.shape[0],
        f"the memberships of {n_states} states",
        (most, most),
        title="State memberships",
        call="Finding states",
        unit="objective evaluations",
        bound=None,
    )

    def objective(free: np.ndarray) -> float:
        progress.add_product()
        return _trial_gap(free, start, tail)

    finished = False
    try:
        # The warning flag is 0 when the simplex converged, 1 or 2 when the
        # evaluations or the iterations reached their limit.
        free, _value, _iterations, _evaluations, warnflag = fmin(
            objective,
            guess[1:, 1:].ravel(),
            maxiter=most,
            maxfun=most,
            disp=False,
            full_output=True,
        )
        finished = True
    finally:
        progress.close(finished)
    best = start.copy()
    best[1:, 1:] = np.asarray(free).reshape(n_states - 1, n_states - 1)
    return feasible_rotation(best, tail), int(warnflag) == 0


def coarse_grain(
    walk: csr_matrix, memberships: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the walk between the states and each state's share of a long walk's time.

    The coarse-grained walk is ``(chi'D chi)^-1 (chi'D P chi)`` with the
    uniform weights ``D``, which cancel. Because ``chi`` spans an invariant
    subspace, the coarse walk fixes ``chi' pi``, ``pi`` the walk's
    stationary distribution, so its left eigenvector for the eigenvalue one,
    scaled to sum to one, is each state's share of the time.
    """
    gram = memberships.T @ memberships
    coarse = np.linalg.solve(gram, memberships.T @ (walk @ memberships))
    values, vectors = np.linalg.eig(coarse.T)
    stationary = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    return coarse, stationary / stationary.sum()


def state_cells(memberships: np.ndarray, cells_per_state: int) -> np.ndarray:
    """Return the state whose most likely cells each cell is among, or -1.

    Each state takes its ``cells_per_state`` cells of largest membership, a
    tie going to the earlier cell, and a cell that several states take goes
    to the one in which its membership is largest, a tie going to the
    earlier state, as CellRank 2.4.0's ``_fuzzy_to_discrete`` assigns them
    with ``remove_overlap=False``.
    """
    n_cells, n_states = memberships.shape
    chosen = np.zeros((n_cells, n_states), dtype=bool)
    count = min(cells_per_state, n_cells)
    for state in range(n_states):
        column = memberships[:, state]
        threshold = np.partition(column, n_cells - count)[n_cells - count]
        above = np.flatnonzero(column > threshold)
        tied = np.flatnonzero(column == threshold)[: count - len(above)]
        chosen[above, state] = True
        chosen[tied, state] = True
    cells = np.full(n_cells, -1, dtype=np.int64)
    taken = np.flatnonzero(chosen.any(axis=1))
    likely = np.where(chosen[taken], memberships[taken], -np.inf)
    cells[taken] = np.argmax(likely, axis=1)
    return cells


def state_labels(
    cells: np.ndarray,
    n_states: int,
    label_codes: np.ndarray | None = None,
    label_categories: Sequence[Any] | None = None,
) -> list[str]:
    """Return the text of the label that most of each state's cells hold, or empty text.

    A tie goes to the earlier label in ``label_categories``. A state none of
    whose cells holds a label takes empty text, and without labels so does
    every state. ``label_codes`` holds each cell's position in
    ``label_categories``, or -1 for no label.
    """
    if label_codes is None:
        return [""] * n_states
    assert label_categories is not None
    labels: list[str] = []
    for state in range(n_states):
        codes = label_codes[(cells == state) & (label_codes >= 0)]
        labels.append(
            str(
                label_categories[
                    int(np.argmax(np.bincount(codes, minlength=len(label_categories))))
                ]
            )
            if codes.size
            else ""
        )
    return labels


def state_names(labels: Sequence[str]) -> tuple[str, ...]:
    """Name each state by its label, or by its number where the label is empty text.

    Names that several states take gain the suffixes ``_1``, ``_2`` in the
    states' order, as CellRank 2.4.0 names macrostates after reference
    clusters, and a suffix is skipped where it would repeat another state's
    name.
    """
    base = [str(label) or str(state + 1) for state, label in enumerate(labels)]
    repeats = Counter(base)
    taken = {name for name, count in repeats.items() if count == 1}
    suffixes: Counter[str] = Counter()
    names: list[str] = []
    for name in base:
        if repeats[name] == 1:
            names.append(name)
            continue
        while True:
            suffixes[name] += 1
            candidate = f"{name}_{suffixes[name]}"
            if candidate not in taken:
                break
        taken.add(candidate)
        names.append(candidate)
    return tuple(names)


def find_states(
    graph: csr_matrix,
    direction: np.ndarray,
    n_states: int,
    *,
    beta: float = 10.0,
    cells_per_state: int = 30,
    max_iterations: int | None = None,
    _copy_graph: bool = True,
) -> FoundStates:
    """Split the walk on a connected graph that ``direction`` directs into ``n_states`` states.

    ``graph`` is the symmetric graph of one connected component, with
    positive weights, and ``direction`` one finite score per cell, higher
    meaning later. The walk is the one that a fate map with ``beta`` takes.
    The dense steps run on one BLAS thread, so a search gives the same
    result whatever the machine's thread settings.

    Raises:
        ValueError: If the graph and the direction do not match, the graph
            holds fewer cells than the states need, the direction does not
            vary, or ``n_states`` would split a complex conjugate pair of
            eigenvalues.
        RuntimeError: If the walk falls into parts that never meet, or the
            eigen-solve does not converge within ``max_iterations`` or finds
            no invariant subspace.
    """
    n_cells = int(graph.shape[0])
    scores = np.asarray(direction, dtype=np.float64)
    if graph.shape != (n_cells, n_cells) or scores.shape != (n_cells,):
        raise ValueError(
            f"The direction holds {scores.shape[0] if scores.ndim == 1 else scores.shape} "
            f"scores for a graph of shape {graph.shape}; give one score per cell "
            "of a square graph"
        )
    if not bool(np.isfinite(scores).all()):
        raise ValueError("The direction must hold a finite score for every cell")
    needed = max(n_states * cells_per_state, n_states + 3)
    if n_cells < needed:
        raise ValueError(
            f"Finding {n_states} states of {cells_per_state} cells each needs at "
            f"least {needed} connected cells, and the cells searched hold "
            f"{n_cells}; ask for fewer states or fewer cells_per_state"
        )
    if float(scores.max()) - float(scores.min()) <= np.finfo(np.float64).eps * max(
        1.0, float(np.abs(scores).max())
    ):
        raise ValueError(
            "The direction holds one value for every searched cell, so it gives "
            "the walk no direction; pass a score that rises along the process"
        )
    scores = _normalize_pseudotime(scores)
    walk = _make_transition(
        graph.copy() if _copy_graph else graph,
        scores,
        np.zeros(n_cells, dtype=bool),
        beta,
    )
    with process_thread_limit(1, user_api="blas"):
        eigenvalues, eigenvectors = leading_eigenvectors(
            walk, n_states + 1, max_iterations=max_iterations
        )
        basis = invariant_basis(walk, eigenvalues, eigenvectors, n_states)
        del eigenvectors
        rotation, optimization_converged = optimize_rotation(basis)
        memberships = basis @ rotation
        del basis
        coarse, walk_share = coarse_grain(walk, memberships)
    del walk
    totals = memberships.sum(axis=0)
    levels = (scores @ memberships) / totals
    order = np.argsort(levels, kind="stable")
    memberships = np.ascontiguousarray(memberships[:, order])
    coarse = coarse[np.ix_(order, order)]
    walk_share = walk_share[order]
    cell_share = totals[order] / n_cells
    levels = levels[order]
    cells = state_cells(memberships, cells_per_state)
    roles = ["initial"] + [
        "terminal" if walk_share[state] > cell_share[state] else "intermediate"
        for state in range(1, n_states)
    ]
    crispness = float(np.sum(rotation**2 / rotation[0, :][np.newaxis, :])) / n_states
    return FoundStates(
        memberships=memberships,
        cells=cells,
        coarse_transitions=coarse,
        cell_share=cell_share,
        walk_share=walk_share,
        direction=levels,
        roles=tuple(roles),
        eigenvalues=eigenvalues,
        crispness=crispness,
        optimization_converged=optimization_converged,
    )


def search_bytes(
    n_cells: int, graph_bytes: int, n_entries: int, n_states: int, name_width: int
) -> int:
    """Return the most bytes that finding ``n_states`` states of ``n_cells`` cells holds.

    The call holds the graph, of ``graph_bytes`` and at most ``n_entries``
    entries, and 64 bytes a cell: the scores, the components, the label
    codes, the searched cells and each cell's state. Taking the searched
    cells' graph holds at most two copies of the graph beside it and the
    cells' places in the graph's index type. The search then holds the walk,
    a float64 copy of that graph, beside the largest of its steps:

    - the eigen-solve, for ``k = n_states + 1`` eigenvalues: ARPACK's start
      vector, residual, ``ncv`` Arnoldi vectors, three work vectors and
      work array, and when it extracts the eigenvectors, its ``k + 1`` real
      Ritz vectors, their complex copy and the copy of ``k`` of them that
      it reorders when it finds one eigenvalue more;
    - the optimization: the basis, its contiguous tail, and the copy that
      the corner search projects with the outer product that it subtracts,
      beside Nelder-Mead's simplex of ``(n_states - 1)**2 + 1`` trials, the
      copy that sorts it and the differences that its stopping test takes;
    - the results: the memberships, their product with the walk and their
      reordered copy, beside the float32 memberships and the names of up to
      ``name_width`` characters written for every cell.

    Forming the basis holds at least 104 bytes a cell less than the
    eigen-solve, so it never decides. A mebibyte more covers NumPy's buffers
    for reductions and the small arrays of each step.
    """
    n, m = int(n_cells), int(n_states)
    k = m + 1
    ncv = min(n, max(2 * k + 1, 20))
    free = (m - 1) ** 2
    index_bytes = _sparse_index_bytes(n, int(n_entries))
    taking = 2 * int(graph_bytes) + (8 + 4 * index_bytes) * n
    walk = (8 + index_bytes) * int(n_entries) + 8 * (n + 1)
    solving = 8 * n * (ncv + 5) + 24 * ncv * (ncv + 2) + 24 * n * (k + 1) + 16 * n * k
    optimizing = 32 * n * m + 8 * n + 24 * free * (free + 1)
    writing = 28 * n * m + 4 * n * int(name_width) + 2 * n
    steps = max(solving, optimizing, writing)
    return int(graph_bytes) + 64 * n + max(taking, walk + steps) + _SMALL_ARRAY_BYTES
