"""Shared parameter resolution for trajectory algorithms and artifacts."""

from collections.abc import Mapping
from typing import Any


# The distance between two feature profiles that the neighbor graph of the
# feature modules uses, and its default.
AGGREGATION_ANN_DEFAULTS: tuple[tuple[str, str | int], ...] = (("space", "l2"),)
AGGREGATION_ANN_STATIC_PARAMETER_NAMES = frozenset(
    name for name, _value in AGGREGATION_ANN_DEFAULTS
)
AGGREGATION_ANN_DERIVED_PARAMETER_NAMES = frozenset({"dim"})
# The settings of the HNSW search that found approximate feature neighbors
# until exact neighbors replaced it. The modules saved before then record them,
# and a call that passes one is refused.
AGGREGATION_HNSW_PARAMETER_NAMES = frozenset(
    {"ef_construction", "M", "random_seed", "ef", "num_threads", "max_elements"}
)
AGGREGATION_ANN_PARAMETER_NAMES = (
    AGGREGATION_ANN_STATIC_PARAMETER_NAMES
    | AGGREGATION_ANN_DERIVED_PARAMETER_NAMES
    | AGGREGATION_HNSW_PARAMETER_NAMES
)


def requested_ann_params(ann_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the ``ann_params`` of a call, refusing the settings of an HNSW search.

    Raises:
        TypeError: If ``ann_params`` is neither a mapping nor None.
        ValueError: If it names a setting of the HNSW search, such as ``ef``.
    """
    if ann_params is None:
        return {}
    if not isinstance(ann_params, Mapping):
        raise TypeError("ann_params must be a mapping or None")
    search = sorted(
        str(name) for name in ann_params if name in AGGREGATION_HNSW_PARAMETER_NAMES
    )
    if search:
        raise ValueError(
            "Feature modules find each feature's nearest features exactly, so "
            "ann_params takes no settings of an approximate search; drop "
            + ", ".join(repr(name) for name in search)
            + ". It takes only 'space', the distance between profiles: 'l2', "
            "'ip' or 'cosine'."
        )
    return dict(ann_params)


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
