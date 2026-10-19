"""Options of a Harmony2 fit: defaults, checks, and expansion to batch levels and clusters.

The option names follow the papers and CyteArc's conventions. Keys of the
earlier CyteArc implementation that were renamed or removed raise an error that
names their replacement; there are no aliases.

References:
    Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019). doi:10.1038/s41592-019-0619-0
    Patikas N, Yao H, et al. Integration of large, complex single-cell datasets with Harmony2. bioRxiv (2026). doi:10.64898/2026.03.16.711825
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ...utils.arguments import float_argument, integer_argument
from .design import BatchDesign

WORK_CHUNK_CELLS = 8192
"""Cells per compute chunk; every per-chunk buffer has at most this many rows."""

DEFAULT_RIDGE_ALPHA = 0.2
"""Scale of the dynamic ridge penalty when ``ridge_alpha`` is not given (Harmony2)."""

HARMONY_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "n_clusters": None,
        "theta": 2.0,
        "sigma": 0.1,
        "tau": 0.0,
        "ridge_penalty": None,
        "ridge_alpha": None,
        "prune_threshold": 1e-5,
        "block_size": 0.05,
        "max_iter_cluster": 200,
        "max_iter_harmony": 40,
        "epsilon_cluster": 1e-5,
        "epsilon_harmony": 1e-4,
        "random_state": 4466,
    }
)
"""The defaults of ``fit_harmony``'s options, in its keyword order."""

HARMONY_PARAMETER_NAMES: frozenset[str] = frozenset(HARMONY_DEFAULTS)
"""The keys that ``harmony_params`` accepts: the options of ``fit_harmony``."""

RENAMED_HARMONY_PARAMETERS: Mapping[str, str] = MappingProxyType(
    {
        "nclust": "n_clusters",
        "max_iter_kmeans": "max_iter_cluster",
        "lamb": "ridge_penalty",
    }
)
"""Keys of the earlier implementation and the keys that replace them."""

REMOVED_HARMONY_PARAMETERS: Mapping[str, str] = MappingProxyType(
    {
        "cluster_fn": (
            "Harmony2 seeds its clusters with k-means++ and has no initialisation hook"
        ),
    }
)
"""Keys of the earlier implementation that have no replacement, with the reason."""

FIT_ARGUMENTS_NOT_PARAMETERS: Mapping[str, str] = MappingProxyType(
    {
        "values": "the DataStore supplies the coordinates",
        "metadata": "the DataStore supplies the batch labels",
        "nthreads": "the DataStore's nthreads sets the threads",
    }
)
"""Arguments of ``fit_harmony`` that ``harmony_params`` cannot set, with the reason."""

_LAMB_NOTE = (
    "; ridge_penalty is a fixed penalty, so leave it out to use Harmony2's "
    "dynamic penalty"
)
# Level and column lists in shape errors stop after this many entries.
_LISTED_NAMES = 20


def validate_harmony_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check Harmony parameters before any data is read; return a new dict equal to the input.

    The checks need no data: the keys, the types, and the ranges of the values.
    Rules that depend on the data, such as the cluster count against the cell
    count or the length of per-level values, are checked by the fit.

    Args:
        parameters: A mapping of option names of
            :func:`cytearc.embeddings.fit_harmony` to values, or None.

    Returns:
        A new dict equal to ``parameters``, with the values untouched; ``{}`` for None.

    Raises:
        TypeError: If ``parameters`` is not a mapping with text keys, or a value
            has the wrong type.
        ValueError: If a key is not an option, was renamed or removed, or a value
            is out of range.
    """
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError(
            "harmony_params must be a mapping of Harmony parameter names to values"
        )
    if not all(isinstance(key, str) for key in parameters):
        raise TypeError("harmony_params keys must be strings")
    for key in sorted(parameters):
        if key in RENAMED_HARMONY_PARAMETERS:
            note = _LAMB_NOTE if key == "lamb" else ""
            raise ValueError(
                f"Unsupported Harmony parameter {key!r}: it was renamed to "
                f"{RENAMED_HARMONY_PARAMETERS[key]!r}{note}"
            )
        reason = REMOVED_HARMONY_PARAMETERS.get(
            key, FIT_ARGUMENTS_NOT_PARAMETERS.get(key)
        )
        if reason is not None:
            raise ValueError(f"Unsupported Harmony parameter {key!r}: {reason}")
    _require_known(parameters)
    check_harmony_options(**parameters)
    return dict(parameters)


def _require_known(options: Mapping[str, Any]) -> None:
    unknown = set(options) - HARMONY_PARAMETER_NAMES
    if unknown:
        raise ValueError(
            f"Unsupported Harmony parameters: {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(HARMONY_PARAMETER_NAMES))}"
        )


def harmony_cluster_count(n_cells: int, n_clusters: int | None = None) -> int:
    """Return the cluster count a fit of n_cells cells uses: n_cells / 30 rounded half to even, kept in [1, 100], or the explicit count.

    Args:
        n_cells: The number of cells.
        n_clusters: An explicit cluster count, or None for the default.

    Returns:
        The cluster count.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If ``n_cells`` is below 1 or ``n_clusters`` lies outside
            ``[1, n_cells]``.
    """
    cells = integer_argument(n_cells, "n_cells", minimum=1)
    if n_clusters is None:
        return min(100, max(1, round(cells / 30)))
    count = integer_argument(n_clusters, "n_clusters")
    if not 1 <= count <= cells:
        raise ValueError(
            f"n_clusters must be between one and the cell count ({cells}); got {count}"
        )
    return count


def _real_values(value: object, name: str) -> np.ndarray:
    """Return a real scalar or one-dimensional sequence as a new float64 array.

    A scalar gives a zero-dimensional array, so that callers can tell it from a
    sequence of one value.
    """
    if isinstance(value, str | bytes | Mapping | bool | np.bool_):
        raise TypeError(f"{name} must contain real numbers")
    if isinstance(value, list | tuple) and any(
        isinstance(item, str | bytes | Mapping | bool | np.bool_) for item in value
    ):
        raise TypeError(f"{name} must contain real numbers")
    try:
        array = np.asarray(value)
    except ValueError:
        # NumPy refuses sequences whose items have different lengths.
        raise ValueError(
            f"{name} must be a scalar or a one-dimensional sequence"
        ) from None
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or a one-dimensional sequence")
    if array.ndim == 1 and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return np.array(array, dtype=np.float64)


def _theta_option(value: object) -> np.ndarray:
    values = _real_values(value, "theta")
    if not (np.all(np.isfinite(values)) and np.all(values >= 0)):
        raise ValueError("theta values must be finite and non-negative")
    return values


def _sigma_option(value: object) -> np.ndarray:
    values = _real_values(value, "sigma")
    if not (np.all(np.isfinite(values)) and np.all(values > 0)):
        raise ValueError("sigma values must be finite and positive")
    return values


def _ridge_penalty_option(value: object) -> np.ndarray | None:
    if value is None:
        return None
    values = _real_values(value, "ridge_penalty")
    if not (np.all(np.isfinite(values)) and np.all(values > 0)):
        raise ValueError("ridge_penalty values must be finite and positive")
    return values


def _cluster_count_option(value: object) -> int | None:
    if value is None:
        return None
    return integer_argument(value, "n_clusters", minimum=1)


def _real_option(
    name: str,
    accepts: Callable[[float], bool],
    rule: str,
    *,
    optional: bool = False,
) -> Callable[[object], float | None]:
    def check(value: object) -> float | None:
        if optional and value is None:
            return None
        resolved = float_argument(value, name)
        if not accepts(resolved):
            raise ValueError(f"{name} {rule}")
        return resolved

    return check


def _integer_option(name: str, minimum: int) -> Callable[[object], int]:
    def check(value: object) -> int:
        return integer_argument(value, name, minimum=minimum)

    return check


# One check per option, in fit_harmony's keyword order, which also orders errors.
_OPTION_CHECKS: Mapping[str, Callable[[object], Any]] = MappingProxyType(
    {
        "n_clusters": _cluster_count_option,
        "theta": _theta_option,
        "sigma": _sigma_option,
        "tau": _real_option("tau", lambda value: value >= 0, "must be at least 0"),
        "ridge_penalty": _ridge_penalty_option,
        "ridge_alpha": _real_option(
            "ridge_alpha",
            lambda value: value > 0,
            "must be greater than 0",
            optional=True,
        ),
        "prune_threshold": _real_option(
            "prune_threshold",
            lambda value: 0 <= value < 1,
            "must be at least 0 and less than 1",
        ),
        "block_size": _real_option(
            "block_size",
            lambda value: 0 < value <= 1,
            "must be greater than 0 and at most 1",
        ),
        "max_iter_cluster": _integer_option("max_iter_cluster", 1),
        "max_iter_harmony": _integer_option("max_iter_harmony", 1),
        "epsilon_cluster": _real_option(
            "epsilon_cluster", lambda value: value >= 0, "must be at least 0"
        ),
        "epsilon_harmony": _real_option(
            "epsilon_harmony", lambda value: value >= 0, "must be at least 0"
        ),
        "random_state": _integer_option("random_state", 0),
    }
)


def check_harmony_options(**options: Any) -> dict[str, Any]:
    """Validate option values that do not depend on the data; return them normalised (Python ints and floats, float64 arrays for array-capable options).

    Only the options given are checked. ``theta``, ``sigma`` and
    ``ridge_penalty`` come back as new float64 arrays: zero-dimensional for a
    scalar, one-dimensional for a sequence.

    Raises:
        TypeError: If a value has the wrong type.
        ValueError: If an option is unknown, a value is out of range, or both
            ``ridge_penalty`` and ``ridge_alpha`` are given.
    """
    _require_known(options)
    checked = {
        name: check(options[name])
        for name, check in _OPTION_CHECKS.items()
        if name in options
    }
    if (
        checked.get("ridge_penalty") is not None
        and checked.get("ridge_alpha") is not None
    ):
        raise ValueError(
            "Pass ridge_penalty for a fixed ridge penalty or ridge_alpha to scale "
            "the dynamic one, not both"
        )
    return checked


@dataclass(frozen=True, slots=True)
class ResolvedHarmonyParameters:
    """Options of one fit, expanded to its batch levels and clusters.

    Attributes:
        n_clusters: The cluster count ``K``.
        theta: ``(levels,)`` diversity penalty of each level, after tau discounting.
        sigma: ``(clusters,)`` entropy weight of each cluster.
        tau: The discounting scale; 0 leaves theta unchanged.
        ridge_penalty: ``(levels,)`` fixed ridge penalty of each level, or None
            for the dynamic penalty.
        ridge_alpha: Scale of the dynamic penalty, or None for a fixed penalty.
        prune_threshold: Smallest ``O_kb / N_b`` for level ``b`` to enter the
            regression of cluster ``k``.
        block_size: Fraction of the cells in each online update block.
        max_iter_cluster: Clustering passes per round, at most.
        max_iter_harmony: Rounds of clustering and correction, at most.
        epsilon_cluster: Relative objective change that ends a round's clustering.
        epsilon_harmony: Relative objective change per round that ends the fit
            (see ``rounds_converged`` in ``fit.py``).
        random_state: Seed of the fit's random generator.
    """

    n_clusters: int
    theta: np.ndarray
    sigma: np.ndarray
    tau: float
    ridge_penalty: np.ndarray | None
    ridge_alpha: float | None
    prune_threshold: float
    block_size: float
    max_iter_cluster: int
    max_iter_harmony: int
    epsilon_cluster: float
    epsilon_harmony: float
    random_state: int


def _listed(names: Sequence[str]) -> str:
    text = ", ".join(names[:_LISTED_NAMES])
    return text + ", ..." if len(names) > _LISTED_NAMES else text


def _per_level(values: np.ndarray, name: str, design: BatchDesign) -> np.ndarray:
    """Expand a scalar, one value per column, or one value per level to the levels."""
    if values.ndim == 0:
        return np.full(design.n_levels, float(values), dtype=np.float64)
    if values.shape[0] == design.n_levels:
        return np.array(values, dtype=np.float64)
    if values.shape[0] == design.n_columns:
        return np.repeat(values, design.column_sizes).astype(np.float64)
    levels = [
        f"{column}={level}"
        for column, column_levels in zip(design.columns, design.levels, strict=True)
        for level in column_levels
    ]
    raise ValueError(
        f"Each Harmony batch level must have a {name}: pass a scalar, one value per "
        f"batch column ({_listed(design.columns)}), or one per batch level "
        f"({_listed(levels)}); got shape {values.shape}"
    )


def resolve_harmony_parameters(
    options: Mapping[str, Any], design: BatchDesign
) -> ResolvedHarmonyParameters:
    """Expand theta, sigma and fixed penalties to levels and clusters and apply tau discounting.

    Args:
        options: Options of :func:`cytearc.embeddings.fit_harmony`; an
            option that is left out takes its default.
        design: The encoded batch design of the cells.

    Returns:
        The resolved options.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If an option is out of range, the cluster count lies outside
            ``[1, cells]``, or per-level or per-cluster values have the wrong length.
    """
    merged = {**HARMONY_DEFAULTS, **options}
    n_clusters = harmony_cluster_count(design.n_cells, merged.pop("n_clusters"))
    checked = check_harmony_options(**merged)
    theta = _per_level(checked["theta"], "theta", design)
    tau = checked["tau"]
    if tau > 0:
        # Theta discounting of the 2019 paper: small batches get a smaller penalty.
        scaled = design.level_counts / (n_clusters * tau)
        theta *= 1.0 - np.exp(-(scaled * scaled))
    sigma = checked["sigma"]
    if sigma.ndim == 0:
        sigma = np.full(n_clusters, float(sigma), dtype=np.float64)
    elif sigma.shape[0] != n_clusters:
        raise ValueError(
            f"sigma must be scalar or have one value per cluster ({n_clusters}); "
            f"got shape {sigma.shape}"
        )
    fixed = checked["ridge_penalty"]
    ridge_penalty = (
        None if fixed is None else _per_level(fixed, "ridge_penalty", design)
    )
    ridge_alpha: float | None = None
    if ridge_penalty is None:
        alpha = checked["ridge_alpha"]
        ridge_alpha = DEFAULT_RIDGE_ALPHA if alpha is None else alpha
    return ResolvedHarmonyParameters(
        n_clusters=n_clusters,
        theta=theta,
        sigma=sigma,
        tau=tau,
        ridge_penalty=ridge_penalty,
        ridge_alpha=ridge_alpha,
        prune_threshold=checked["prune_threshold"],
        block_size=checked["block_size"],
        max_iter_cluster=checked["max_iter_cluster"],
        max_iter_harmony=checked["max_iter_harmony"],
        epsilon_cluster=checked["epsilon_cluster"],
        epsilon_harmony=checked["epsilon_harmony"],
        random_state=checked["random_state"],
    )


def harmony_work_sizes(n_cells: int, block_size: float) -> tuple[int, int]:
    """Return the cells per online update block and per compute chunk."""
    block_cells = max(1, math.ceil(block_size * n_cells))
    return block_cells, min(WORK_CHUNK_CELLS, block_cells)
