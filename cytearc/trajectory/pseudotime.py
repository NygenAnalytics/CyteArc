import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, sparray, spmatrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    aslinearoperator,
    minres,
    svds,
)

from ..neighbors.diffusion import inverse_degree_diagonal
from ..utils.arguments import float_argument, integer_argument
from ..utils.logging import logger
from ..utils.progress import tqdmbar
from .errors import (
    PseudotimeConvergenceError,
    SourceSinkLabelsError,
    _check_missing_from,
)

_COMPONENT_POLICIES = ("largest", "sources", "error")
# The seconds between the INFO records that report a running solve.
_PROGRESS_INTERVAL_SECONDS = 30.0
# The seconds between updates of the solve's progress bar. A progress
# observer receives every update, so the bar is not updated per product.
_BAR_INTERVAL_SECONDS = 0.5
# The products of SciPy's check, before MINRES iterates, that its operator is
# symmetric.
_SYMMETRY_CHECK_PRODUCTS = 2


def validate_source_sink_labels(
    labels: pd.Series,
    sources: list[Any],
    sinks: list[Any],
    context: str,
    *,
    missing_from: Literal["labels", "scored component"] = "labels",
    component_sizes: Sequence[int] | None = None,
) -> None:
    """Check that ``labels`` hold every source and sink label.

    ``context`` names the labeled cells in the message. A label that no cell
    holds raises :class:`SourceSinkLabelsError`, which carries
    ``missing_from`` and, for ``"scored component"``, ``component_sizes``,
    the cell counts of the graph's components, largest first.

    Raises:
        ValueError: If a label is both a source and a sink.
        SourceSinkLabelsError: If no cell holds a source or sink label.
    """
    _check_missing_from(missing_from, component_sizes)
    overlap = sorted(set(sources) & set(sinks))
    if overlap:
        raise ValueError(f"Source and sink labels overlap in {context}: {overlap}")

    present = set(pd.unique(labels))
    missing_sources = [source for source in sources if source not in present]
    missing_sinks = [sink for sink in sinks if sink not in present]
    if missing_sources or missing_sinks:
        message = (
            f"Source/sink labels were not found in {context}. "
            f"Missing sources: {missing_sources}; missing sinks: {missing_sinks}"
        )
        if component_sizes is not None:
            message += (
                f". The graph's connected components hold {list(component_sizes)} cells"
            )
        raise SourceSinkLabelsError(
            message,
            missing_sources=missing_sources,
            missing_sinks=missing_sinks,
            missing_from=missing_from,
            component_sizes=component_sizes,
        )


def make_source_sink_vector(
    labels: pd.Series,
    sources: list[Any],
    sinks: list[Any],
) -> np.ndarray:
    sink_mask = labels.isin(sinks).to_numpy(dtype=bool)
    source_mask = labels.isin(sources).to_numpy(dtype=bool)
    labeled = sink_mask | source_mask
    if labeled.all():
        raise ValueError(
            "All selected cells are labeled as sources or sinks, so the "
            "source/sink vector cannot be balanced over unlabeled cells"
        )

    vector = np.zeros(labels.shape[0], dtype=float)
    vector[sink_mask] = 1.0
    vector[source_mask] = -1.0
    vector[~labeled] = -vector.sum() / int((~labeled).sum())
    return vector


def validate_source_sink_vector(
    values: np.ndarray,
    n_cells: int,
    context: str,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must contain numeric values") from exc

    if vector.ndim == 2 and vector.shape == (n_cells, 1):
        vector = vector[:, 0]
    elif vector.ndim != 1:
        raise ValueError(
            f"{context} must be one-dimensional or have shape ({n_cells}, 1)"
        )

    if vector.shape[0] != n_cells:
        raise ValueError(
            f"Size mismatch between {context} ({vector.shape[0]}) and graph ({n_cells})"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{context} must contain only finite values")

    tolerance = 1e-10 * max(1.0, float(np.abs(vector).sum()))
    if not np.isclose(vector.sum(), 0.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"The values in {context} must sum to zero")
    return vector


def pseudotime_components(
    graph: csr_matrix,
    selected_cell_indices: np.ndarray,
    component_policy: Literal["largest", "sources", "error"],
    *,
    source_cells: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Number a graph's connected components and choose the one to score.

    Components are numbered by decreasing size. Components of equal size are
    numbered in the order of the smallest selected cell index that each
    holds, then of SciPy's component labels, so component 0 is the one that
    ``"largest"`` scores. ``"sources"`` scores the component that holds the
    most of ``source_cells``, a boolean mask that it requires, and a tie goes
    to the lower number. ``"error"`` refuses a graph of more than one
    component.

    Returns:
        Each cell's component number, the components' cell counts in number
        order, and the number of the scored component.
    """
    if component_policy not in _COMPONENT_POLICIES:
        raise ValueError("component_policy must be 'largest', 'sources' or 'error'")
    if graph.shape[0] != graph.shape[1] or graph.shape[0] != len(selected_cell_indices):
        raise ValueError(
            "The graph must be square with one row per selected cell index"
        )
    if component_policy == "sources":
        if source_cells is None:
            raise ValueError("component_policy='sources' needs the source cells")
        source_cells = np.asarray(source_cells)
        if source_cells.dtype != np.bool_ or source_cells.shape != (graph.shape[0],):
            raise ValueError(
                "source_cells must be a boolean mask with one entry per graph row"
            )
        if not source_cells.any():
            raise ValueError(
                "component_policy='sources' needs at least one source cell: a "
                "source label or a negative ss_vec value"
            )
    elif source_cells is not None:
        raise ValueError("source_cells applies only to component_policy='sources'")

    n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    sizes = np.bincount(labels, minlength=n_components)
    smallest_index = np.full(n_components, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(
        smallest_index,
        labels,
        np.asarray(selected_cell_indices, dtype=np.int64),
    )
    # Sort by decreasing size, then smallest selected index, then label.
    order = np.lexsort((np.arange(n_components), smallest_index, -sizes))
    numbers = np.empty(n_components, dtype=np.int64)
    numbers[order] = np.arange(n_components, dtype=np.int64)
    component = numbers[labels]
    component_sizes = tuple(int(size) for size in sizes[order])
    if n_components > 1 and component_policy == "error":
        raise ValueError(
            f"The selected graph has {n_components} connected components with "
            f"sizes {list(component_sizes)}"
        )
    scored = 0
    if component_policy == "sources":
        source_counts = np.bincount(component[source_cells], minlength=n_components)
        scored = int(np.argmax(source_counts))
    return component, component_sizes, scored


def select_pseudotime_component(
    graph: csr_matrix,
    selected_cell_indices: np.ndarray,
    component_policy: Literal["largest", "sources", "error"],
    *,
    source_cells: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Return the cells of the component to score and every component's size.

    :func:`pseudotime_components` chooses the component. The sizes are listed
    largest first.
    """
    component, component_sizes, scored = pseudotime_components(
        graph,
        selected_cell_indices,
        component_policy,
        source_cells=source_cells,
    )
    return np.asarray(component == scored, dtype=bool), list(component_sizes)


def random_walk_laplacian_transpose(graph: csr_matrix) -> csr_matrix:
    degree = np.asarray(graph.sum(axis=1), dtype=float).ravel()
    if np.any(degree <= 0):
        raise ValueError("The retained graph contains isolated cells")
    n_cells = graph.shape[0]
    inverse_degree = inverse_degree_diagonal(degree)
    identity = csr_matrix(
        (np.ones(n_cells), (range(n_cells), range(n_cells))),
        shape=(n_cells, n_cells),
    )
    return identity - graph.dot(inverse_degree)


def _solver_bounds(
    solver_tol: float | None,
    max_iterations: int | None,
) -> tuple[float | None, int | None]:
    """Check a potential solve's tolerance and iteration bound, either None."""
    if solver_tol is not None:
        solver_tol = float_argument(solver_tol, "solver_tol")
        if not 0.0 < solver_tol < 1.0:
            raise ValueError("solver_tol must be greater than 0 and less than 1")
    if max_iterations is not None:
        max_iterations = integer_argument(max_iterations, "max_iterations", minimum=1)
    return solver_tol, max_iterations


def _product_bound(n_cells: int, n_modes: int, max_iterations: int) -> int:
    """Return the most Laplacian products of a solve within ``max_iterations``.

    ARPACK builds its first Lanczos basis of SciPy's default size,
    ``min(n_cells, max(2 * n_modes + 1, 20))``, with one product of ``L Lᵀ``
    more than the basis holds, and each restart adds at most
    ``basis - n_modes`` products; fewer once some modes converge. Each
    product of ``L Lᵀ`` takes two of the Laplacian.
    """
    basis = min(n_cells, max(2 * n_modes + 1, 20))
    return 2 * (1 + basis + max_iterations * (basis - n_modes))


class _SolveProgress:
    """Count the Laplacian products of a pseudotime solve and report them.

    A progress bar counts the products against ``bound``, the most that
    ``max_iterations`` allows, or without a total when ``max_iterations`` is
    None, and an INFO record reports them every
    ``_PROGRESS_INTERVAL_SECONDS``. ``solving`` names what the solve finds,
    such as ``"for 30 singular modes"``.
    """

    def __init__(
        self,
        n_cells: int,
        solving: str,
        max_iterations: int | None,
        bound: int | None,
    ) -> None:
        if (max_iterations is None) != (bound is None):
            raise ValueError(
                "bound must be given with max_iterations, and only with it"
            )
        self.products = 0
        self._n_cells = n_cells
        self._solving = solving
        self._max_iterations = max_iterations
        self._bound = bound
        if self._bound is None:
            # Without a total, the bar shows the count and its rate.
            self._bar = tqdmbar(
                desc="Pseudotime solve",
                bar_format="{desc}: {n_fmt} operator products [{elapsed}, {rate_fmt}]",
            )
        else:
            self._bar = tqdmbar(total=self._bound, desc="Pseudotime solve")
        self._started = self._recorded = self._shown = time.monotonic()
        self._shown_products = 0

    def add_product(self) -> None:
        self.products += 1
        now = time.monotonic()
        if now - self._shown >= _BAR_INTERVAL_SECONDS:
            self._show(now)
        if now - self._recorded >= _PROGRESS_INTERVAL_SECONDS:
            self._recorded = now
            self._record(now)

    def _show(self, now: float) -> None:
        self._bar.update(self.products - self._shown_products)
        self._shown_products = self.products
        self._shown = now

    def _record(self, now: float) -> None:
        if self._bound is None:
            count = f"{self.products} operator products"
            bound = "; set max_iterations to bound the solve"
        else:
            count = f"{self.products} of at most {self._bound} operator products"
            bound = f" (max_iterations={self._max_iterations})"
        logger.info(
            f"Pseudotime scoring: {count} after {now - self._started:.0f} s, "
            f"solving {self._solving} of {self._n_cells} cells{bound}"
        )

    def close(self, converged: bool) -> None:
        """Show the last count, or a full bar when a bounded solve converged."""
        try:
            if converged and self._bound is not None:
                self._bar.update(max(0, self._bound - self._shown_products))
            else:
                self._show(time.monotonic())
        finally:
            self._bar.close()


def _counted_operator(
    laplacian_transpose: csr_matrix,
    progress: _SolveProgress,
) -> LinearOperator:
    """Wrap the Laplacian so that each product with a vector is counted.

    The products are those of SciPy's own operator for the matrix, so the
    solve computes exactly what it computes on the matrix.
    """
    operator = aslinearoperator(laplacian_transpose)

    def matvec(values: np.ndarray) -> np.ndarray:
        progress.add_product()
        return np.asarray(operator.matvec(values))

    def rmatvec(values: np.ndarray) -> np.ndarray:
        progress.add_product()
        return np.asarray(operator.rmatvec(values))

    return LinearOperator(
        shape=operator.shape,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=operator.matmat,
        rmatmat=operator.rmatmat,
        dtype=operator.dtype,
    )


def truncated_pba_potential(
    laplacian_transpose: csr_matrix,
    n_singular_vals: int,
    random_seed: int,
    source_sink: np.ndarray,
    *,
    solver_tol: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Return the population balance potential of the smallest singular modes.

    The potential keeps the ``n_singular_vals - 1`` smallest nonzero singular
    modes of the pseudoinverse that ``pba_potential`` applies in full; it
    is the potential of ``method="truncated"``. SciPy's ``svds`` finds the
    ``n_singular_vals`` smallest singular values of the transposed
    random-walk Laplacian, the null mode included, with ARPACK's Lanczos
    solve. ``solver_tol`` is their relative tolerance,
    machine precision when None, and ``max_iterations`` bounds the solve's
    implicit restarts, ten per cell when None; both defaults are SciPy's. A
    looser tolerance can end the search before it finds the smallest singular
    values, so the potential can lose modes; a result without the null mode
    is refused. The solve advances a progress bar with its products of the
    Laplacian and its transpose, and logs their count at INFO level every 30
    seconds.

    Raises:
        PseudotimeConvergenceError: If the solve reaches ``max_iterations``
            before every mode converges.
    """
    solver_tol, max_iterations = _solver_bounds(solver_tol, max_iterations)
    random_state = np.random.RandomState(random_seed)
    initial_vector = random_state.rand(laplacian_transpose.shape[0])
    logger.debug(
        f"Pseudotime scoring: calculating SVD "
        f"(shape={laplacian_transpose.shape}, nnz={laplacian_transpose.nnz}, "
        f"k={n_singular_vals})"
    )
    n_cells = int(laplacian_transpose.shape[0])
    progress = _SolveProgress(
        n_cells,
        f"for {n_singular_vals} singular modes",
        max_iterations,
        None
        if max_iterations is None
        else _product_bound(n_cells, n_singular_vals, max_iterations),
    )
    converged = False
    try:
        left_vectors, singular_values, right_vectors_t = svds(
            _counted_operator(laplacian_transpose, progress),
            k=n_singular_vals,
            which="SM",
            v0=initial_vector,
            # SciPy's defaults are 0, machine precision, and ten restarts per cell.
            tol=0.0 if solver_tol is None else solver_tol,
            maxiter=max_iterations,
        )
        converged = True
    except ArpackNoConvergence as exc:
        # ARPACK stops only once its restarts reach the bound.
        bound = 10 * n_cells if max_iterations is None else max_iterations
        raise PseudotimeConvergenceError(
            method="truncated",
            n_cells=n_cells,
            n_modes=n_singular_vals,
            converged_modes=len(exc.eigenvalues),
            restarts=bound,
            operator_products=progress.products,
            max_iterations=bound,
            solver_tol=solver_tol,
        ) from exc
    finally:
        progress.close(converged)

    order = np.argsort(singular_values)
    singular_values = singular_values[order]
    left_vectors = left_vectors[:, order]
    right_vectors = right_vectors_t[order, :].T

    rank_tolerance = (
        max(laplacian_transpose.shape)
        * np.finfo(singular_values.dtype).eps
        * max(1.0, float(singular_values.max()))
    )
    if singular_values[0] > max(1e-8, rank_tolerance):
        message = "The graph Laplacian does not contain the expected null mode"
        if solver_tol is not None:
            # A loose tolerance lets the unshifted search stop before it
            # finds the smallest singular values.
            message += (
                f", which the solve can miss at solver_tol={solver_tol}; lower "
                "solver_tol or leave it unset"
            )
        raise ValueError(message)
    if np.any(singular_values[1:] <= rank_tolerance):
        raise ValueError(
            "The graph Laplacian contains additional near-zero singular modes"
        )

    inverse_singular_values = 1.0 / singular_values[1:]
    left_vectors = left_vectors[:, 1:]
    right_vectors = right_vectors[:, 1:]
    return np.asarray(
        left_vectors
        @ (inverse_singular_values * (right_vectors.T @ source_sink).ravel())
    )


def pba_potential(
    graph: spmatrix | sparray,
    source_sink: np.ndarray,
    *,
    solver_tol: float | None = None,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Return the population balance potential of a graph over all of its modes.

    The potential is ``pinv(L) R``, the minimum-norm least-squares solution
    of ``L V = R`` for the random-walk Laplacian ``L = I - D^-1 W`` of the
    symmetric graph ``W`` with degrees ``D``, and the source-sink vector
    ``R``. Population balance analysis defines the potential as
    ``1/2 pinv(L) R`` (Weinreb et al. 2018). The range of ``L`` is orthogonal
    to the degree vector, so ``R`` first loses its component along it, as the
    pseudoinverse drops it. ``L V = R`` is then ``L_sym (D^1/2 V) = D^1/2 R``
    for the symmetric Laplacian ``L_sym = I - D^-1/2 W D^-1/2``, which SciPy's
    ``minres`` solves after it checks that the operator is symmetric. The
    potential is centered at zero, as the pseudoinverse centers it.

    ``solver_tol`` is MINRES's ``rtol``: the solve stops once its residual
    falls below ``solver_tol`` times the estimated norms of ``L_sym`` and of
    its solution. None, the default, solves to machine precision.
    ``max_iterations`` bounds the MINRES iterations, one product with
    ``L_sym`` each, SciPy's five per cell when None. The solve advances a
    progress bar with its products and logs their count at INFO level every
    30 seconds.

    Raises:
        TypeError: If the graph is not a SciPy sparse matrix.
        ValueError: If the graph is not square, symmetric or connected, or
            holds isolated cells, or ``source_sink`` does not hold one finite
            value per cell.
        PseudotimeConvergenceError: If the solve reaches ``max_iterations``
            before it meets ``solver_tol``.
    """
    solver_tol, max_iterations = _solver_bounds(solver_tol, max_iterations)
    if not issparse(graph):
        raise TypeError("graph must be a SciPy sparse matrix")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    n_cells = int(graph.shape[0])
    values = np.asarray(source_sink, dtype=np.float64)
    if values.shape != (n_cells,) or not np.isfinite(values).all():
        raise ValueError(
            f"source_sink must hold one finite value for each of the {n_cells} "
            "cells of the graph"
        )
    weights = graph.tocsr()
    if weights.dtype != np.float64:
        weights = weights.astype(np.float64)
    degree = np.asarray(weights.sum(axis=1), dtype=np.float64).ravel()
    if np.any(degree <= 0):
        raise ValueError("The retained graph contains isolated cells")
    n_components = connected_components(weights, directed=False, return_labels=False)
    if n_components > 1:
        # Each component adds a null mode to L, so the pseudoinverse would
        # leave the potentials of the components unrelated.
        raise ValueError(
            f"graph must be connected, but it has {n_components} connected "
            "components; score one component at a time"
        )
    magnitude = float(np.abs(values).max(initial=0.0))
    if magnitude == 0.0:
        return np.zeros(n_cells)
    # The potential is linear in R, so the solve takes R scaled to a largest
    # magnitude of 1, which none of its products can overflow.
    unit = values / magnitude
    projected = unit - degree * float(degree @ unit) / float(degree @ degree)
    sqrt_degree = np.sqrt(degree)
    inverse_sqrt_degree = 1.0 / sqrt_degree
    right_hand_side = sqrt_degree * projected

    def symmetric_laplacian(vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            vector - inverse_sqrt_degree * weights.dot(inverse_sqrt_degree * vector)
        )

    progress = _SolveProgress(
        n_cells,
        "for the untruncated potential",
        max_iterations,
        None if max_iterations is None else max_iterations + _SYMMETRY_CHECK_PRODUCTS,
    )

    def counted(vector: np.ndarray) -> np.ndarray:
        progress.add_product()
        return symmetric_laplacian(np.asarray(vector, dtype=np.float64).ravel())

    logger.debug(
        f"Pseudotime scoring: solving for the untruncated potential "
        f"(cells={n_cells}, nnz={weights.nnz})"
    )
    converged = False
    try:
        try:
            solution, info = minres(
                LinearOperator(
                    shape=(n_cells, n_cells),
                    matvec=counted,
                    dtype=np.dtype(np.float64),
                ),
                right_hand_side,
                # 0 solves to machine precision.
                rtol=0.0 if solver_tol is None else solver_tol,
                maxiter=max_iterations,
                check=True,
            )
        except ValueError as exc:
            if "non-symmetric" not in str(exc):
                raise
            raise ValueError(
                "graph must be symmetric, with the same weight on each edge in "
                "both directions"
            ) from exc
        converged = info == 0
    finally:
        progress.close(converged)
    if not converged:
        residual = np.linalg.norm(
            symmetric_laplacian(solution) - right_hand_side
        ) / np.linalg.norm(right_hand_side)
        raise PseudotimeConvergenceError(
            method="untruncated",
            n_cells=n_cells,
            operator_products=progress.products,
            # SciPy's minres stops after five iterations per cell by default.
            max_iterations=5 * n_cells if max_iterations is None else max_iterations,
            solver_tol=solver_tol,
            relative_residual=float(residual),
        )
    potential = solution * inverse_sqrt_degree
    potential -= potential.mean()
    return np.asarray(potential * magnitude)
