from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ...graph.arguments import graph_flag
from ...graph.feature_projection import graph_cell_selection
from ...graph.kinds import require_graph_kind
from ...metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.arguments import LeidenArguments
from ...storage.validation_scope import validation_scoped
from ...storage.artifacts import (
    ArtifactRef,
    inspect_artifact,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
    reused_artifact_group,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.errors import ArtifactResolutionError
from ...utils.arguments import integer_argument
from ...utils.logging import logger
from ...utils.shutdown import shutdown_checkpoint

if TYPE_CHECKING:
    from ...clustering.paris_multiscale import ParisClusteringResult
    from .graph import _GraphOperationsMixin as _ClusteringOperationsBase
else:
    _ClusteringOperationsBase = object


@dataclass(frozen=True, slots=True)
class _PreparedLeidenClustering:
    planned: PlannedArtifact
    graph: ArtifactRef
    resolution: float
    backend: Literal["igraph", "leidenalg"]
    symmetric_graph: bool
    graph_upper_only: bool
    random_seed: int
    n_cells: int


class _ClusteringOperationsMixin(_ClusteringOperationsBase):
    def _clustering_graph(self, graph: ArtifactRef) -> tuple[str, int, int]:
        """Return a complete clustering graph's location, cell count, and k."""
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        require_graph_kind(graph)
        status = inspect_artifact(self.zw, graph)
        if not status.complete:
            raise ValueError("Graph artifact is unavailable or incomplete")
        n_cells, k = self._get_graph_ncells_k(status.path)
        return status.path, n_cells, k

    def _run_paris_from_artifacts(
        self,
        *,
        graph_ref: ArtifactRef,
        graph_loc: str,
        fixed_cluster_count: int | None,
        effective_min_cluster_size: int | None,
        invalidate_cache: bool,
    ) -> "ParisClusteringResult":
        from ...clustering._paris_core import ParisHierarchy
        from ...clustering._paris_modularity import modularity_split_gains
        from ...clustering.paris import (
            fit_paris_hierarchy,
            fixed_cut,
            hierarchy_to_dendrogram,
        )
        from ...clustering.paris_multiscale import (
            ParisClusteringResult,
            PlateauForest,
            adaptive_cut,
            collapse_equal_height_plateaus,
        )
        from .paris_persistence import (
            hierarchy_array_requirements,
            hierarchy_attribute_requirements,
            load_hierarchy_group,
            plan_paris_dendrogram,
            preflight_hierarchy_artifact_cut,
            preflight_paris_adaptive_cut,
            preflight_paris_fit,
            read_paris_cut_diagnostics,
            write_hierarchy_group,
            write_paris_dendrogram,
        )

        artifact_scope = graph_ref.scope
        artifact_assay = graph_ref.assay
        cell_selection = graph_cell_selection(self.zw, graph_ref)
        n_cells, _effective_k = self._get_graph_ncells_k(graph_loc)
        cut_mode: Literal["adaptive", "fixed"] = (
            "fixed" if fixed_cluster_count is not None else "adaptive"
        )
        mode: Literal["auto", "fixed"] = (
            "fixed" if fixed_cluster_count is not None else "auto"
        )
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        budget = self.resources
        # Structurally incomplete hierarchies are skipped here and refitted.
        hierarchy_plan = plan_artifact(
            self.zw,
            scope=artifact_scope,
            assay=artifact_assay,
            kind="cluster_hierarchy",
            operation="fit_paris_hierarchy",
            parameters={},
            inputs={"connectivity_map": graph_ref},
            execution_options={"invalidate_cache": invalidate_cache},
            invalidate_cache=invalidate_cache,
            required_arrays=hierarchy_array_requirements(),
            required_attributes=hierarchy_attribute_requirements(n_cells),
        )
        loaded: tuple[ParisHierarchy, PlateauForest] | None = None
        fitted_graph = None

        def hierarchy_payload() -> tuple[ParisHierarchy, PlateauForest]:
            # Load or fit the hierarchy only when a cut or dendrogram needs it.
            nonlocal loaded, fitted_graph
            if loaded is not None:
                return loaded
            if hierarchy_plan.reused:
                hierarchy_group = reused_artifact_group(self.zw, hierarchy_plan)
                preflight_hierarchy_artifact_cut(
                    hierarchy_group,
                    cut_mode,
                    budget,
                )
                try:
                    loaded = load_hierarchy_group(
                        hierarchy_group,
                        hierarchy_plan.ref.artifact_id,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ArtifactResolutionError(
                        f"Paris hierarchy artifact {hierarchy_plan.ref.artifact_id} "
                        "is unreadable. Rerun with invalidate_cache=True to "
                        "recompute it.",
                        code="corrupt_payload",
                        context={"artifact_id": hierarchy_plan.ref.artifact_id},
                    ) from error
            else:
                estimated_peak_bytes = preflight_paris_fit(
                    graph_group,
                    n_cells,
                    budget,
                )
                fitted_graph = self._load_graph_artifact(
                    graph_ref,
                    symmetric=False,
                    upper_only=False,
                    use_k=None,
                )
                shutdown_checkpoint()
                hierarchy = fit_paris_hierarchy(
                    fitted_graph,
                    nthreads=budget.workers,
                )
                shutdown_checkpoint()
                plateau_forest = collapse_equal_height_plateaus(hierarchy)
                with artifact_transaction(self.zw, hierarchy_plan) as hierarchy_group:
                    write_hierarchy_group(hierarchy_group, hierarchy, plateau_forest)
                    hierarchy_group.attrs["estimated_peak_bytes"] = estimated_peak_bytes
                loaded = hierarchy, plateau_forest
            if loaded[0].n_leaves != n_cells:
                raise ValueError("Paris hierarchy size does not match graph")
            return loaded

        cut_parameters = (
            {"mode": mode, "n_clusters": fixed_cluster_count}
            if fixed_cluster_count is not None
            else {
                "mode": mode,
                "min_cluster_size": effective_min_cluster_size,
            }
        )
        cut_inputs = {
            "cluster_hierarchy": hierarchy_plan.ref,
            "connectivity_map": graph_ref,
            "cell_selection": cell_selection,
        }

        def valid_cluster_count(value: object) -> bool:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                return False
            return fixed_cluster_count is None or value == fixed_cluster_count

        def readable_cut(_ref: ArtifactRef, group: Any) -> bool:
            try:
                read_paris_cut_diagnostics(group, mode)
            except (TypeError, ValueError):
                return False
            return True

        # A new hierarchy has a fresh identity, so its cut is never reused.
        cut_plan = plan_artifact(
            self.zw,
            scope=artifact_scope,
            assay=artifact_assay,
            kind="cluster_cut",
            operation="cut_paris_hierarchy",
            parameters=cut_parameters,
            inputs=cut_inputs,
            execution_options={"invalidate_cache": invalidate_cache},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("labels", shape=(n_cells,), dtype_kind="i"),
            ),
            required_attributes=(
                AttributeRequirement("n_clusters", predicate=valid_cluster_count),
            ),
            reuse_validator=readable_cut,
        )
        if cut_plan.reused:
            cut_group = reused_artifact_group(self.zw, cut_plan)
            result = ParisClusteringResult(
                labels=np.asarray(
                    as_zarr_array(cut_group["labels"], name="labels")[:],
                    dtype=np.int32,
                ),
                mode=mode,
                n_clusters=int(cast(int, cut_group.attrs["n_clusters"])),
                diagnostics=read_paris_cut_diagnostics(cut_group, mode),
                min_cluster_size=effective_min_cluster_size,
            )
        else:
            hierarchy, plateau_forest = hierarchy_payload()
            if fixed_cluster_count is None:
                assert effective_min_cluster_size is not None
                if fitted_graph is None:
                    preflight_paris_adaptive_cut(
                        graph_group,
                        n_cells,
                        budget,
                    )
                    fitted_graph = self._load_graph_artifact(
                        graph_ref,
                        symmetric=None,
                        upper_only=None,
                        use_k=None,
                    )
                split_gate = modularity_split_gains(
                    hierarchy,
                    plateau_forest,
                    fitted_graph,
                )
                shutdown_checkpoint()
                result = adaptive_cut(
                    hierarchy,
                    effective_min_cluster_size,
                    plateau_forest=plateau_forest,
                    split_gate=split_gate,
                )
                shutdown_checkpoint()
            else:
                n_components = len(hierarchy.component_roots)
                if 1 < fixed_cluster_count < n_components:
                    raise ValueError(
                        f"The graph has {n_components} connected components, so a "
                        f"fixed Paris cut cannot produce {fixed_cluster_count} "
                        f"clusters. Request n_clusters=1, at least "
                        f"{n_components} clusters, or n_clusters='auto'."
                    )
                labels = fixed_cut(hierarchy, fixed_cluster_count).astype(
                    np.int32,
                    copy=False,
                )
                shutdown_checkpoint()
                result = ParisClusteringResult(
                    labels=labels,
                    mode="fixed",
                    n_clusters=fixed_cluster_count,
                )
            with artifact_transaction(self.zw, cut_plan) as cut_group:
                labels_array = create_zarr_dataset(
                    cut_group,
                    "labels",
                    (min(max(n_cells, 1), 100_000),),
                    "i4",
                    result.labels.shape,
                )
                labels_array[:] = result.labels
                cut_group.attrs["n_clusters"] = int(result.n_clusters)
                cut_group.attrs["diagnostics"] = [
                    asdict(diagnostic) for diagnostic in result.diagnostics
                ]

        if fixed_cluster_count is not None:
            dendrogram_plan = plan_paris_dendrogram(self.zw, hierarchy_plan.ref)
            if not dendrogram_plan.reused:
                write_paris_dendrogram(
                    self.zw,
                    dendrogram_plan,
                    hierarchy_to_dendrogram(
                        hierarchy_payload()[0],
                        compatibility=True,
                    ),
                )

        action = "Reused" if cut_plan.reused else "Stored"
        logger.info(f"{action} Paris clustering with {result.n_clusters} clusters")
        return replace(
            result,
            hierarchy_artifact_id=hierarchy_plan.ref.artifact_id,
            ref=cut_plan.ref,
        )

    def _prepare_leiden_clustering(
        self,
        graph: ArtifactRef,
        *,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> _PreparedLeidenClustering:
        from ...clustering.leiden import canonical_random_seed, canonical_resolution

        if backend not in {"igraph", "leidenalg"}:
            raise ValueError("backend must be 'igraph' or 'leidenalg'")
        resolution = canonical_resolution(resolution)
        random_seed = canonical_random_seed(random_seed)
        symmetric_graph = graph_flag(symmetric_graph, "symmetric_graph")
        graph_upper_only = graph_flag(graph_upper_only, "graph_upper_only")
        _graph_loc, n_cells, _k = self._clustering_graph(graph)
        graph_input = graph
        artifact_scope = graph_input.scope
        selection = graph_cell_selection(self.zw, graph_input)
        arguments = LeidenArguments(
            graph=graph_input,
            resolution=resolution,
            backend=backend,
            edge_weighting="graph",
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=artifact_scope,
            assay=(graph_input.assay if graph_input.scope == "assay" else None),
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=selection,
            arrays={"values": ((n_cells,), "i")},
            invalidate_cache=invalidate_cache,
        )
        return _PreparedLeidenClustering(
            planned=planned,
            graph=graph_input,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            n_cells=n_cells,
        )

    def _run_leiden_artifact(
        self,
        graph: ArtifactRef,
        *,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Execute Leiden clustering and return its immutable artifact.

        Args:
            graph: Explicit connectivity map or integrated graph to partition.
            resolution: Finite positive Leiden resolution, recorded as a float.
            backend: Leiden implementation. Native igraph is the default.
            symmetric_graph: Forwarded to `graph.load`.
            graph_upper_only: Forwarded to `graph.load`.
            random_seed: Non-negative integer seed for the Leiden optimizer.
            invalidate_cache: Force a new cluster-labels artifact.

        Returns:
            Reference to the cluster-labels artifact.
        """
        from ...clustering.leiden import leiden_membership

        prepared = self._prepare_leiden_clustering(
            graph,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        if prepared.planned.reused:
            logger.info("Reused Leiden clustering artifact")
            return prepared.planned.ref
        graph_matrix = self._load_graph_artifact(
            prepared.graph,
            symmetric=prepared.symmetric_graph,
            upper_only=prepared.graph_upper_only,
            use_k=None,
        )
        shutdown_checkpoint()
        membership = np.asarray(
            leiden_membership(
                graph_matrix,
                prepared.resolution,
                prepared.random_seed,
                backend=prepared.backend,
            )
        )
        shutdown_checkpoint()
        if membership.shape != (prepared.n_cells,):
            raise ValueError("Leiden membership must contain one label per graph cell")
        if membership.dtype.kind not in {"i", "u"}:
            raise TypeError("Leiden membership must contain integer labels")
        write_cell_data_artifact(self.zw, prepared.planned, {"values": membership})
        logger.info(
            f"Stored Leiden clustering with {np.unique(membership).size} clusters"
        )
        return prepared.planned.ref

    @validation_scoped
    def _clusters_leiden(
        self,
        graph: ArtifactRef,
        *,
        resolution: float = 1.0,
        backend: Literal["igraph", "leidenalg"] = "igraph",
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build and return immutable Leiden cluster labels.

        ``resolution`` must be a finite positive number and is recorded as a
        float, so ``1`` and ``1.0`` identify the same artifact. ``random_seed``
        must be a non-negative integer because stored labels are reused.
        """
        return self._run_leiden_artifact(
            graph,
            resolution=resolution,
            backend=backend,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )

    def _run_paris_artifact(
        self,
        graph: ArtifactRef,
        *,
        n_clusters: int | Literal["auto"] = "auto",
        min_cluster_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit the canonical Paris hierarchy and write a fixed or adaptive cut.

        Pass ``graph`` to partition an explicit connectivity map or integrated
        graph. The returned reference identifies the immutable cut artifact;
        ``clusters.load_paris`` reconstructs labels and diagnostics.
        """
        if isinstance(n_clusters, (bool, np.bool_)):
            raise TypeError("n_clusters must be an integer or 'auto'")
        if isinstance(n_clusters, str):
            if n_clusters != "auto":
                raise ValueError("n_clusters must be an integer or 'auto'")
            fixed_cluster_count = None
        elif isinstance(n_clusters, (int, np.integer)):
            if n_clusters < 1:
                raise ValueError("n_clusters must be positive")
            fixed_cluster_count = int(n_clusters)
        else:
            raise TypeError("n_clusters must be an integer or 'auto'")
        if fixed_cluster_count is not None and min_cluster_size is not None:
            raise ValueError("min_cluster_size is only valid when n_clusters='auto'")
        graph_loc, n_cells, effective_k = self._clustering_graph(graph)
        if fixed_cluster_count is not None and fixed_cluster_count > n_cells:
            raise ValueError(f"n_clusters must not exceed the graph size ({n_cells})")

        if fixed_cluster_count is None:
            if min_cluster_size is None:
                effective_min_cluster_size = effective_k + 1
            else:
                effective_min_cluster_size = integer_argument(
                    min_cluster_size, "min_cluster_size", minimum=2
                )
        else:
            effective_min_cluster_size = None

        result = self._run_paris_from_artifacts(
            graph_ref=graph,
            graph_loc=graph_loc,
            fixed_cluster_count=fixed_cluster_count,
            effective_min_cluster_size=effective_min_cluster_size,
            invalidate_cache=invalidate_cache,
        )
        if result.ref is None:
            raise RuntimeError("Paris clustering did not produce an artifact")
        return result.ref

    def _load_paris_artifact_result(
        self,
        ref: ArtifactRef,
    ) -> "ParisClusteringResult":
        from ...clustering.paris_multiscale import ParisClusteringResult
        from .paris_persistence import read_paris_cut_diagnostics

        status = inspect_artifact(self.zw, ref)
        if not status.complete or status.operation != "cut_paris_hierarchy":
            raise ValueError("Paris cut artifact is unavailable or invalid")
        group = as_zarr_group(self.zw[status.path], name=status.path)
        labels = np.asarray(
            as_zarr_array(group["labels"], name="labels")[:],
            dtype=np.int32,
        )
        parameters = status.parameters or {}
        mode = parameters.get("mode")
        if mode not in {"auto", "fixed"}:
            raise ValueError("Paris cut mode is invalid")
        try:
            diagnostics = read_paris_cut_diagnostics(
                group,
                cast(Literal["auto", "fixed"], mode),
            )
        except (TypeError, ValueError) as error:
            raise ArtifactResolutionError(
                f"Paris cut artifact {ref.artifact_id} does not match the current "
                "diagnostics schema. Recompute it with clusters.paris("
                "invalidate_cache=True).",
                code="corrupt_payload",
                context={"artifact_id": ref.artifact_id},
            ) from error
        raw_hierarchy = (status.inputs or {}).get("cluster_hierarchy")
        if not isinstance(raw_hierarchy, dict):
            raise ArtifactResolutionError(
                f"Paris cut artifact {ref.artifact_id} does not name its hierarchy",
                code="corrupt_payload",
                context={"artifact_id": ref.artifact_id},
            )
        return ParisClusteringResult(
            labels=labels,
            mode=cast(Literal["auto", "fixed"], mode),
            n_clusters=int(cast(int | float | str, group.attrs["n_clusters"])),
            diagnostics=diagnostics,
            min_cluster_size=(
                int(parameters["min_cluster_size"]) if mode == "auto" else None
            ),
            hierarchy_artifact_id=ArtifactRef.from_dict(raw_hierarchy).artifact_id,
            ref=ref,
        )

    @validation_scoped
    def _clusters_load_paris(
        self,
        ref: ArtifactRef,
    ) -> "ParisClusteringResult":
        """Load a Paris result from an explicit completed cut artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "cluster_cut":
            raise ValueError("ref must be a cluster_cut artifact")
        return self._load_paris_artifact_result(ref)

    @validation_scoped
    def _clusters_paris(
        self,
        graph: ArtifactRef,
        *,
        n_clusters: int | Literal["auto"] = "auto",
        min_cluster_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build and return an immutable Paris cut artifact.

        A fixed integer ``n_clusters`` must be 1 or at least the number of
        connected components in ``graph``, and the hierarchy must split into
        exactly that many clusters at one height. Otherwise a ``ValueError``
        names the alternatives, such as ``n_clusters='auto'``.
        """
        return self._run_paris_artifact(
            graph,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            invalidate_cache=invalidate_cache,
        )
