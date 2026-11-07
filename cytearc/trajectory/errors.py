"""Typed refusals of pseudotime scoring and fate mapping.

Each error carries the numbers or labels that a caller acts on, so a caller
reads its fields instead of parsing its message.
"""

from collections.abc import Sequence
from typing import Any, Literal

_MISSING_FROM = ("labels", "scored component")


def _check_missing_from(missing_from: object, component_sizes: object) -> None:
    """Refuse a location of missing labels, or sizes that do not go with it."""
    if missing_from not in _MISSING_FROM:
        raise ValueError(
            f"missing_from must be one of {_MISSING_FROM}, not {missing_from!r}"
        )
    if (component_sizes is None) != (missing_from == "labels"):
        raise ValueError(
            "component_sizes must be given with missing_from='scored component', "
            "and only with it"
        )


class PseudotimeConvergenceError(RuntimeError):
    """The singular value solve of a pseudotime stopped at its bound.

    Attributes:
        n_cells: The cells of the scored component, one row of the graph
            Laplacian each.
        n_modes: The singular modes that the solve sought, the null mode
            included.
        converged_modes: The modes that had converged when the solve stopped.
        restarts: The implicit restarts that the Lanczos solve ran, which
            reached ``max_iterations``.
        operator_products: The products of the Laplacian or its transpose
            with a vector that the solve computed.
        max_iterations: The bound on restarts that stopped the solve: the
            caller's, or SciPy's default of ten per cell.
        solver_tol: The relative tolerance of the singular values, or None
            for machine precision.
    """

    def __init__(
        self,
        n_cells: int,
        n_modes: int,
        converged_modes: int,
        restarts: int,
        operator_products: int,
        max_iterations: int,
        solver_tol: float | None,
    ) -> None:
        self.n_cells = int(n_cells)
        self.n_modes = int(n_modes)
        self.converged_modes = int(converged_modes)
        self.restarts = int(restarts)
        self.operator_products = int(operator_products)
        self.max_iterations = int(max_iterations)
        self.solver_tol = None if solver_tol is None else float(solver_tol)
        super().__init__(
            "Pseudotime scoring did not converge on the graph Laplacian of "
            f"{self.n_cells} cells: {self.converged_modes} of {self.n_modes} "
            "singular modes converged before the solve reached "
            f"max_iterations={self.max_iterations} after {self.restarts} restarts "
            f"and {self.operator_products} operator products "
            f"(solver_tol={self.solver_tol}). The smallest singular values lie too "
            "close together to separate, as on large graphs and on graphs whose "
            "parts are joined by few edges, and fewer modes do not help. Score "
            "fewer cells, or score each weakly joined part on a graph of its own "
            "cells. A larger max_iterations helps only when most modes converged."
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            type(self),
            (
                self.n_cells,
                self.n_modes,
                self.converged_modes,
                self.restarts,
                self.operator_products,
                self.max_iterations,
                self.solver_tol,
            ),
            self.__dict__,
        )


class SourceSinkLabelsError(ValueError):
    """Source or sink labels that a pseudotime or a fate map needs are missing.

    Attributes:
        missing_sources: The source labels that no checked cell holds, in the
            order given.
        missing_sinks: The sink labels that no checked cell holds, in the
            order given.
        missing_from: ``"labels"`` when no selected cell holds them, or
            ``"scored component"`` when no cell of the graph component that
            the pseudotime scores holds them.
        component_sizes: The cell counts of the graph's connected components,
            largest first, when ``missing_from`` is ``"scored component"``,
            and None otherwise.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_sources: Sequence[Any],
        missing_sinks: Sequence[Any],
        missing_from: Literal["labels", "scored component"],
        component_sizes: Sequence[int] | None = None,
    ) -> None:
        _check_missing_from(missing_from, component_sizes)
        super().__init__(message)
        self.missing_sources = tuple(missing_sources)
        self.missing_sinks = tuple(missing_sinks)
        self.missing_from: Literal["labels", "scored component"] = missing_from
        self.component_sizes = (
            None
            if component_sizes is None
            else tuple(int(size) for size in component_sizes)
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            _restore_source_sink_labels_error,
            (
                type(self),
                str(self.args[0]),
                self.missing_sources,
                self.missing_sinks,
                self.missing_from,
                self.component_sizes,
            ),
            self.__dict__,
        )


def _restore_source_sink_labels_error(
    error_type: type[SourceSinkLabelsError],
    message: str,
    missing_sources: tuple[Any, ...],
    missing_sinks: tuple[Any, ...],
    missing_from: Literal["labels", "scored component"],
    component_sizes: tuple[int, ...] | None,
) -> SourceSinkLabelsError:
    """Rebuild a pickled SourceSinkLabelsError, whose fields are keywords."""
    error = SourceSinkLabelsError.__new__(error_type)
    SourceSinkLabelsError.__init__(
        error,
        message,
        missing_sources=missing_sources,
        missing_sinks=missing_sinks,
        missing_from=missing_from,
        component_sizes=component_sizes,
    )
    return error
