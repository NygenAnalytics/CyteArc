from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np
import zarr

from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    plan_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    serialize_artifact_value,
)

type ArgumentRole = Literal["input", "parameter", "execution"]


def graph_flag(value: object, name: str) -> bool:
    """Return a graph-loading flag as a Python bool; NumPy bools are accepted."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise TypeError(f"{name} must be a boolean")


def graph_load_flags(
    symmetric_graph: object,
    graph_upper_only: object,
) -> tuple[bool, bool]:
    """Return the graph-loading flags of a graph consumer as Python bools.

    ``graph_upper_only`` keeps the upper triangle of the symmetrized graph, so
    it applies only with ``symmetric_graph``. Alone it would change nothing
    and record an identical result under a second identity, so it raises.
    """
    symmetric = graph_flag(symmetric_graph, "symmetric_graph")
    upper_only = graph_flag(graph_upper_only, "graph_upper_only")
    if upper_only and not symmetric:
        raise ValueError(
            "graph_upper_only=True applies only with symmetric_graph=True; pass "
            "symmetric_graph=True as well, or leave graph_upper_only False"
        )
    return symmetric, upper_only


def umap_graph_flags(
    symmetric_graph: object,
    graph_upper_only: object,
) -> tuple[bool, bool]:
    """Check the graph-loading flags of ``embeddings.umap``; both must be False.

    UMAP optimizes the fuzzy union ``w_ij + w_ji - w_ij * w_ji`` of the
    graph's edge weights, which it forms itself. A graph that
    ``symmetric_graph`` already symmetrized would be united twice, and the
    union of its upper triangle is the union of the graph, so neither flag
    has a correct effect of its own.
    """
    symmetric = graph_flag(symmetric_graph, "symmetric_graph")
    upper_only = graph_flag(graph_upper_only, "graph_upper_only")
    if symmetric or upper_only:
        raise ValueError(
            "embeddings.umap symmetrizes the graph itself with the fuzzy union "
            "w_ij + w_ji - w_ij * w_ji, so symmetric_graph=True would apply that "
            "union twice and graph_upper_only=True would change nothing; leave "
            "both False"
        )
    return symmetric, upper_only


def parameter(
    default: Any = MISSING,
    *,
    omit_if_none: bool = False,
) -> Any:
    """Declare an identity parameter.

    ``omit_if_none`` leaves an unset parameter out of the argument record, so
    adding it keeps the identities of artifacts that do not use it.
    """
    metadata = {"argument_role": "parameter", "omit_if_none": omit_if_none}
    if default is not MISSING:
        return field(default=default, metadata=metadata)
    return field(metadata=metadata)


def execution(default: Any = MISSING) -> Any:
    if default is not MISSING:
        return field(default=default, metadata={"argument_role": "execution"})
    return field(metadata={"argument_role": "execution"})


def artifact_input() -> Any:
    return field(metadata={"argument_role": "input"})


@dataclass(frozen=True, slots=True)
class ArgumentRecord:
    parameters: dict[str, Any]
    execution_options: dict[str, Any]
    inputs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OperationArguments:
    operation: ClassVar[str]
    artifact_kind: ClassVar[str]

    def to_record(self) -> ArgumentRecord:
        partitions: dict[ArgumentRole, dict[str, Any]] = {
            "input": {},
            "parameter": {},
            "execution": {},
        }
        for model_field in fields(self):
            role = model_field.metadata.get("argument_role")
            if role not in partitions:
                raise TypeError(
                    f"{type(self).__name__}.{model_field.name} has no argument role"
                )
            value = getattr(self, model_field.name)
            if value is None and model_field.metadata.get("omit_if_none"):
                continue
            partitions[role][model_field.name] = serialize_artifact_value(value)
        return ArgumentRecord(
            parameters=partitions["parameter"],
            execution_options=partitions["execution"],
            inputs=partitions["input"],
        )

    def plan(
        self,
        root: zarr.Group,
        *,
        scope: ArtifactScope,
        assay: str | None = None,
        invalidate_cache: bool = False,
        required_arrays: tuple[str | ArrayRequirement, ...] = (),
        required_attributes: tuple[str | AttributeRequirement, ...] = (),
        reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
    ) -> PlannedArtifact:
        record = self.to_record()
        return plan_artifact(
            root,
            scope=scope,
            assay=assay,
            kind=self.artifact_kind,
            operation=self.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            required_attributes=required_attributes,
            reuse_validator=reuse_validator,
        )


@dataclass(frozen=True, slots=True)
class NormalizationArguments(OperationArguments):
    operation: ClassVar[str] = "run_normalization"
    artifact_kind: ClassVar[str] = "normalized"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    dataset_fingerprint: str = artifact_input()
    normalization_method: Callable[..., Any] | str = parameter()
    size_factor: float | None = parameter()
    log_transform: bool = parameter()
    renormalize_subset: bool = parameter()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class FeatureScalingArguments(OperationArguments):
    operation: ClassVar[str] = "calculate_feature_scaling"
    artifact_kind: ClassVar[str] = "feature_scaling"

    normalized: ArtifactRef = artifact_input()
    enabled: bool = parameter()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class PcaArguments(OperationArguments):
    operation: ClassVar[str] = "run_pca"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = artifact_input()
    feature_scaling: ArtifactRef = artifact_input()
    pca_cell_selection: ArtifactRef = artifact_input()
    dims: int = parameter()
    feat_scaling: bool = parameter()
    batch_size: int = execution()
    show_elbow_plot: bool = execution()
    invalidate_cache: bool = execution(False)
    # "randomized" when more than 4096 features are fitted with the randomized
    # solver, whose result approximates the exact one. Exact fits leave it unset.
    solver: Literal["randomized"] | None = parameter(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class LsiArguments(OperationArguments):
    operation: ClassVar[str] = "run_lsi"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = artifact_input()
    feature_scaling: ArtifactRef = artifact_input()
    dims: int = parameter()
    skip_first: bool = parameter()
    rand_state: int = parameter()
    solver: Literal["streaming", "materialized"] = parameter()
    n_iter: int = parameter()
    n_oversamples: int = parameter()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class CustomReductionArguments(OperationArguments):
    operation: ClassVar[str] = "run_custom_reduction"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = artifact_input()
    feature_scaling: ArtifactRef = artifact_input()
    loadings: np.ndarray = artifact_input()
    dims: int = parameter()
    feat_scaling: bool = parameter()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class HarmonyArguments(OperationArguments):
    operation: ClassVar[str] = "run_harmony"
    artifact_kind: ClassVar[str] = "batch_correction"

    reduction: ArtifactRef = artifact_input()
    batch_snapshot: ArtifactRef = artifact_input()
    batch_columns: tuple[str, ...] = parameter()
    harmony_parameters: Mapping[str, Any] = parameter()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "harmony_parameters",
            MappingProxyType(dict(self.harmony_parameters)),
        )


@dataclass(frozen=True, slots=True)
class AnnIndexArguments(OperationArguments):
    operation: ClassVar[str] = "build_ann_index"
    artifact_kind: ClassVar[str] = "ann_index"

    coordinates: ArtifactRef = artifact_input()
    ann_metric: str = parameter()
    ann_efc: int = parameter()
    # The smallest default search depth of the index's queries, which
    # graph.neighbors and mapping projections read from this record.
    ann_ef: int = parameter()
    ann_m: int = parameter()
    rand_state: int = parameter()
    ann_parallel: bool = parameter()
    # Parallel insertion on several threads can build a different index on
    # every run, whatever the thread count, so only ann_parallel identifies an
    # index and the thread count is the execution option nthreads.
    nthreads: int = execution()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class NeighborQueryArguments(OperationArguments):
    operation: ClassVar[str] = "query_neighbors"
    artifact_kind: ClassVar[str] = "neighbors"

    ann_index: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    k: int = parameter()
    distance_metric: str = parameter()
    # Queries of a fixed index return the same neighbors on any thread count.
    nthreads: int = execution()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)
    # The search depth (hnswlib ``ef``) the query used, as the resolved
    # integer, when it differs from the depth the index records, which
    # hnswlib raises to k + 1. Absent, the query searched the index's depth,
    # as every query did before queries chose their own.
    ann_ef: int | None = parameter(None, omit_if_none=True)


@dataclass(frozen=True, slots=True)
class ConnectivityMapArguments(OperationArguments):
    operation: ClassVar[str] = "build_connectivity_map"
    artifact_kind: ClassVar[str] = "connectivity_map"

    neighbors: ArtifactRef = artifact_input()
    local_connectivity: float = parameter()
    bandwidth: float = parameter()
    invalidate_cache: bool = execution(False)

    def __post_init__(self) -> None:
        from ..neighbors.graph import validate_connectivity_parameters

        local_connectivity, bandwidth = validate_connectivity_parameters(
            self.local_connectivity,
            self.bandwidth,
        )
        object.__setattr__(self, "local_connectivity", local_connectivity)
        object.__setattr__(self, "bandwidth", bandwidth)


@dataclass(frozen=True, slots=True)
class EmbeddingInitializationArguments(OperationArguments):
    operation: ClassVar[str] = "build_embedding_initialization"
    artifact_kind: ClassVar[str] = "embedding_initialization"

    coordinates: ArtifactRef = artifact_input()
    n_centroids: int = parameter()
    rand_state: int = parameter()
    batch_size: int = parameter()
    kmeans_sampling: float = parameter(0.1)
    kmeans_batch_size: int = parameter(10_000)
    invalidate_cache: bool = execution(False)
