from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .feature_dynamics import (
        aggregate_feature_profiles as aggregate_feature_profiles,
        scatter_feature_clusters as scatter_feature_clusters,
        validate_pseudotime_regressor as validate_pseudotime_regressor,
    )
    from .pseudotime import (
        make_source_sink_vector as make_source_sink_vector,
        random_walk_laplacian_transpose as random_walk_laplacian_transpose,
        select_pseudotime_component as select_pseudotime_component,
        truncated_pba_potential as truncated_pba_potential,
        validate_source_sink_labels as validate_source_sink_labels,
        validate_source_sink_vector as validate_source_sink_vector,
    )
    from .results import (
        CurveProjection as CurveProjection,
        FateMappingResult as FateMappingResult,
        LineageAggregationResult as LineageAggregationResult,
        LineageCurvesResult as LineageCurvesResult,
        LineageDriverResult as LineageDriverResult,
        LineageMarkerResult as LineageMarkerResult,
        LineageTreeResult as LineageTreeResult,
        PseudotimeAggregationResult as PseudotimeAggregationResult,
        PseudotimeMarkerResult as PseudotimeMarkerResult,
        PseudotimeScoreResult as PseudotimeScoreResult,
    )

__all__ = [
    "CurveProjection",
    "FateMappingResult",
    "LineageAggregationResult",
    "LineageCurvesResult",
    "LineageDriverResult",
    "LineageMarkerResult",
    "LineageTreeResult",
    "PseudotimeAggregationResult",
    "PseudotimeMarkerResult",
    "PseudotimeScoreResult",
    "aggregate_feature_profiles",
    "make_source_sink_vector",
    "random_walk_laplacian_transpose",
    "select_pseudotime_component",
    "scatter_feature_clusters",
    "truncated_pba_potential",
    "validate_source_sink_labels",
    "validate_source_sink_vector",
    "validate_pseudotime_regressor",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "aggregate_feature_profiles": ".feature_dynamics",
        "make_source_sink_vector": ".pseudotime",
        "random_walk_laplacian_transpose": ".pseudotime",
        "select_pseudotime_component": ".pseudotime",
        "scatter_feature_clusters": ".feature_dynamics",
        "truncated_pba_potential": ".pseudotime",
        "validate_source_sink_labels": ".pseudotime",
        "validate_source_sink_vector": ".pseudotime",
        "validate_pseudotime_regressor": ".feature_dynamics",
        "CurveProjection": ".results",
        "FateMappingResult": ".results",
        "LineageAggregationResult": ".results",
        "LineageCurvesResult": ".results",
        "LineageDriverResult": ".results",
        "LineageMarkerResult": ".results",
        "LineageTreeResult": ".results",
        "PseudotimeAggregationResult": ".results",
        "PseudotimeMarkerResult": ".results",
        "PseudotimeScoreResult": ".results",
    },
)
