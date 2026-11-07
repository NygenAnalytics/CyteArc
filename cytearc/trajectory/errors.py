"""Typed refusals of pseudotime scoring and fate mapping.

Each error carries the numbers or labels that a caller acts on, so a caller
reads its fields instead of parsing its message.
"""

from collections.abc import Sequence
from typing import Any, Literal

_MISSING_FROM = ("labels", "scored component")
_POTENTIAL_METHODS = ("untruncated", "truncated")


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
    """The solve of a pseudotime's potential stopped at its bound.

    ``method`` names the solve: ``"untruncated"`` for MINRES on the symmetric
    graph Laplacian, ``"truncated"`` for the Lanczos solve of the smallest
    singular values. The fields of the other solve are None.

    Attributes:
        method: The potential that the solve computed, ``"untruncated"`` or
            ``"truncated"``.
        n_cells: The cells of the scored component, one row of the graph
            Laplacian each.
        operator_products: The products of the graph Laplacian, or of its
            transpose, with a vector that the solve computed.
        max_iterations: The bound that stopped the solve, on MINRES iterations
            or on the implicit restarts of the singular value solve: the
            caller's, or SciPy's default of five iterations or ten restarts
            per cell.
        solver_tol: The tolerance of the solve, or None for machine precision.
        relative_residual: For ``"untruncated"``, the residual of the
            symmetric system relative to its right-hand side when the solve
            stopped.
        n_modes: For ``"truncated"``, the singular modes that the solve
            sought, the null mode included.
        converged_modes: For ``"truncated"``, the modes that had converged
            when the solve stopped.
        restarts: For ``"truncated"``, the implicit restarts that the Lanczos
            solve ran, which reached ``max_iterations``.
    """

    def __init__(
        self,
        *,
        method: Literal["untruncated", "truncated"],
        n_cells: int,
        operator_products: int,
        max_iterations: int,
        solver_tol: float | None,
        relative_residual: float | None = None,
        n_modes: int | None = None,
        converged_modes: int | None = None,
        restarts: int | None = None,
    ) -> None:
        if method not in _POTENTIAL_METHODS:
            raise ValueError(
                f"method must be one of {_POTENTIAL_METHODS}, not {method!r}"
            )
        truncated = method == "truncated"
        modes = (n_modes, converged_modes, restarts)
        if any((value is None) == truncated for value in modes) or (
            (relative_residual is None) != truncated
        ):
            raise ValueError(
                "n_modes, converged_modes and restarts must be given with "
                "method='truncated', and relative_residual with "
                "method='untruncated', each only with it"
            )
        self.method: Literal["untruncated", "truncated"] = method
        self.n_cells = int(n_cells)
        self.operator_products = int(operator_products)
        self.max_iterations = int(max_iterations)
        self.solver_tol = None if solver_tol is None else float(solver_tol)
        self.relative_residual = (
            None if relative_residual is None else float(relative_residual)
        )
        self.n_modes = None if n_modes is None else int(n_modes)
        self.converged_modes = None if converged_modes is None else int(converged_modes)
        self.restarts = None if restarts is None else int(restarts)
        if truncated:
            message = (
                f"{self.converged_modes} of {self.n_modes} singular modes converged "
                f"before the solve reached max_iterations={self.max_iterations} "
                f"after {self.restarts} restarts and {self.operator_products} "
                f"operator products (solver_tol={self.solver_tol}). The smallest "
                "singular values lie too close together to separate, as on large "
                "graphs and on graphs whose parts are joined by few edges, and "
                "fewer modes do not help. Score fewer cells, or score each weakly "
                "joined part on a graph of its own cells. A larger max_iterations "
                "helps only when most modes converged."
            )
        else:
            message = (
                f"MINRES reached max_iterations={self.max_iterations} after "
                f"{self.operator_products} operator products with a relative "
                f"residual of {self.relative_residual:.1e} "
                f"(solver_tol={self.solver_tol}). MINRES needs more iterations "
                "the more weakly the graph's parts are joined. Raise "
                "max_iterations, or score each weakly joined part on a graph of "
                "its own cells."
            )
        super().__init__(
            "Pseudotime scoring did not converge on the graph Laplacian of "
            f"{self.n_cells} cells: {message}"
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return (
            _restore_pseudotime_convergence_error,
            (
                type(self),
                {
                    "method": self.method,
                    "n_cells": self.n_cells,
                    "operator_products": self.operator_products,
                    "max_iterations": self.max_iterations,
                    "solver_tol": self.solver_tol,
                    "relative_residual": self.relative_residual,
                    "n_modes": self.n_modes,
                    "converged_modes": self.converged_modes,
                    "restarts": self.restarts,
                },
            ),
            self.__dict__,
        )


def _restore_pseudotime_convergence_error(
    error_type: type[PseudotimeConvergenceError],
    fields: dict[str, Any],
) -> PseudotimeConvergenceError:
    """Rebuild a pickled PseudotimeConvergenceError, whose fields are keywords."""
    error = PseudotimeConvergenceError.__new__(error_type)
    PseudotimeConvergenceError.__init__(error, **fields)
    return error


class SourceSinkLabelsError(ValueError):
    """Source or sink labels that a pseudotime or a fate map needs are missing.

    Attributes:
        missing_sources: The source labels that no checked cell holds, in the
            order given.
        missing_sinks: The sink labels that no checked cell holds, in the
            order given.
        missing_from: ``"labels"`` when no selected cell holds them, or
            ``"scored component"`` when no scored cell holds them: no cell of
            the graph component that the pseudotime scores, or of the
            components that the direction score of a fate map scores.
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
