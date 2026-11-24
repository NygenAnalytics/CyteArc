"""Shared parameter resolution for trajectory algorithms and artifacts."""

from collections.abc import Mapping
from typing import Any

from ..utils.warnings import warn


# The distance between two feature profiles that the neighbor graph of the
# feature modules uses, and its default.
AGGREGATION_ANN_DEFAULTS: tuple[tuple[str, str | int], ...] = (("space", "l2"),)
AGGREGATION_ANN_STATIC_PARAMETER_NAMES = frozenset(
    name for name, _value in AGGREGATION_ANN_DEFAULTS
)
AGGREGATION_ANN_DERIVED_PARAMETER_NAMES = frozenset({"dim"})
# The settings of the HNSW search that found approximate feature neighbors
# until exact neighbors replaced it. The modules saved before then record them,
# and a call that passes one is warned that it has no effect.
AGGREGATION_HNSW_PARAMETER_NAMES = frozenset(
    {"ef_construction", "M", "random_seed", "ef", "num_threads", "max_elements"}
)
AGGREGATION_ANN_PARAMETER_NAMES = (
    AGGREGATION_ANN_STATIC_PARAMETER_NAMES
    | AGGREGATION_ANN_DERIVED_PARAMETER_NAMES
    | AGGREGATION_HNSW_PARAMETER_NAMES
)


def requested_ann_params(ann_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the ``ann_params`` of a call without the settings of an HNSW search.

    The feature neighbors are exact, so a setting of the HNSW search, such as
    ``ef``, has no effect: it is left out, with one ``FutureWarning`` that
    names it.

    Raises:
        TypeError: If ``ann_params`` is neither a mapping nor None.
        ValueError: If it names a parameter that is neither the distance nor a
            setting of the HNSW search.
    """
    if ann_params is None:
        return {}
    if not isinstance(ann_params, Mapping):
        raise TypeError("ann_params must be a mapping or None")
    unknown = sorted(
        str(name) for name in ann_params if name not in AGGREGATION_ANN_PARAMETER_NAMES
    )
    if unknown:
        raise ValueError(
            "ann_params contains unsupported parameters: "
            + ", ".join(repr(name) for name in unknown)
            + "; it takes 'space', the distance between profiles: 'l2', 'ip' or "
            "'cosine'"
        )
    ignored = sorted(
        str(name) for name in ann_params if name in AGGREGATION_HNSW_PARAMETER_NAMES
    )
    if ignored:
        warn(
            "Feature modules find each feature's nearest features exactly, so "
            "these settings of an approximate search in ann_params have no "
            "effect and can be dropped: " + ", ".join(repr(name) for name in ignored),
            FutureWarning,
        )
    return {
        name: value
        for name, value in ann_params.items()
        if name not in AGGREGATION_HNSW_PARAMETER_NAMES
    }


def resolve_aggregation_ann_params(
    ann_params: Mapping[str, Any] | None,
    *,
    dim: int,
) -> dict[str, Any]:
    """Resolve the distance between feature profiles before artifact planning."""
    resolved: dict[str, Any] = dict(AGGREGATION_ANN_DEFAULTS)
    resolved.update(requested_ann_params(ann_params))
    if "dim" in resolved and resolved["dim"] != dim:
        raise ValueError("ann_params.dim must match the effective bin count")
    resolved["dim"] = dim
    return resolved
