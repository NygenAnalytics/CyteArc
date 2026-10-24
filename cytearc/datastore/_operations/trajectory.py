from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ...assay import Assay
from ...assay.normalization import (
    reject_unknown_normalization_params,
    resolve_normalization_params,
)
from ...features.values import ResolvedFeature, iter_normalized_feature_blocks
from ...graph.feature_projection import (
    NATIVE_COORDINATE_KINDS,
    graph_cell_selection,
    resolve_coordinate_inputs,
    resolve_graph_source_assay,
    resolve_native_graph_inputs,
)
from ...graph.kinds import require_graph_kind
from ...matrix import ChunkedArray
from ...metadata.rows import (
    read_array_rows_chunkwise,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.arguments import (
    FateMappingArguments,
    LineageAggregationArguments,
    LineageCurveArguments,
    LineageDriverArguments,
    LineageMarkerArguments,
    LineageTreeArguments,
    PseudotimeAggregationArguments,
    PseudotimeMarkerArguments,
    PseudotimeScoringArguments,
)
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.table import CaseInsensitiveIndex
from ...storage.validation_scope import validation_scoped
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    artifact_transaction,
    plan_artifact,
    reused_artifact_group,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    callable_identity,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    fingerprint_strings,
    inspect_artifact,
    parse_artifact_ref,
)
from ...storage.copy import copy_metadata_array
from ...storage.feature_selection import read_feature_selection_indices
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.selections import (
    read_stored_selection_indices,
    resolve_metadata_snapshot,
    validate_stored_selection_integrity,
)
from ...storage.arrays import create_zarr_dataset
from ...trajectory.feature_dynamics import (
    scatter_feature_clusters as _scatter_feature_clusters_impl,
    validate_pseudotime_regressor,
)
from ...trajectory.parameters import resolve_aggregation_ann_params
from ...trajectory.artifacts import (
    AGGREGATION_INPUTS as _AGGREGATION_INPUTS,
    AGGREGATION_PAYLOAD as _AGGREGATION_PAYLOAD,
    FATE_INPUTS as _FATE_INPUTS,
    FATE_PARAMETERS as _FATE_PARAMETERS,
    LINEAGE_AGGREGATION_INPUTS as _LINEAGE_AGGREGATION_INPUTS,
    LINEAGE_AGGREGATION_PAYLOAD as _LINEAGE_AGGREGATION_PAYLOAD,
    LINEAGE_CURVE_INPUTS as _LINEAGE_CURVE_INPUTS,
    LINEAGE_CURVE_PAYLOAD as _LINEAGE_CURVE_PAYLOAD,
    LINEAGE_DRIVER_INPUTS as _LINEAGE_DRIVER_INPUTS,
    LINEAGE_DRIVER_METHODS as _LINEAGE_DRIVER_METHODS,
    LINEAGE_DRIVER_PAYLOAD as _LINEAGE_DRIVER_PAYLOAD,
    LINEAGE_MARKER_INPUTS as _LINEAGE_MARKER_INPUTS,
    LINEAGE_MARKER_METHODS as _LINEAGE_MARKER_METHODS,
    LINEAGE_MARKER_PAYLOAD as _LINEAGE_MARKER_PAYLOAD,
    LINEAGE_TREE_INPUTS as _LINEAGE_TREE_INPUTS,
    LINEAGE_TREE_PAYLOAD as _LINEAGE_TREE_PAYLOAD,
    MARKER_INPUTS as _MARKER_INPUTS,
    MARKER_PAYLOAD as _MARKER_PAYLOAD,
    PSEUDOTIME_INPUTS as _PSEUDOTIME_INPUTS,
    PSEUDOTIME_PARAMETERS as _PSEUDOTIME_PARAMETERS,
    aggregation_payload_is_valid as _aggregation_payload_is_valid,
    diffusion_load_bytes,
    diffusion_payload_is_valid as _diffusion_payload_is_valid,
    load_diffusion_payload,
    write_diffusion_payload,
    fate_payload_is_valid as _fate_payload_is_valid,
    labels_with_missing_mask as _labels_with_missing_mask,
    lineage_aggregation_payload_is_valid as _lineage_aggregation_payload_is_valid,
    lineage_curve_payload_is_valid as _lineage_curve_payload_is_valid,
    lineage_driver_payload_is_valid as _lineage_driver_payload_is_valid,
    lineage_marker_payload_is_valid as _lineage_marker_payload_is_valid,
    lineage_tree_payload_is_valid as _lineage_tree_payload_is_valid,
    load_cell_artifact_values as _load_cell_artifact_values,
    marker_payload_is_valid as _marker_payload_is_valid,
    pseudotime_payload_is_valid as _pseudotime_payload_is_valid,
    require_exact_record_keys as _require_exact_record_keys,
    selection_size as _selection_size,
    true_array_indices as _true_array_indices,
    validate_aggregation_parameters as _validate_aggregation_parameters,
    validate_fate_parameters as _validate_fate_parameters,
    validate_lineage_aggregation_parameters as _validate_lineage_aggregation_parameters,
    validate_lineage_curve_parameters as _validate_lineage_curve_parameters,
    validate_lineage_driver_parameters as _validate_lineage_driver_parameters,
    validate_lineage_marker_parameters as _validate_lineage_marker_parameters,
    validate_lineage_tree_parameters as _validate_lineage_tree_parameters,
    validate_marker_parameters as _validate_marker_parameters,
    validate_pseudotime_parameters as _validate_pseudotime_parameters,
    validate_resolved_ann_parameters as _validate_resolved_ann_parameters,
)
from ...trajectory.curves import (
    CURVE_FIT_BINS,
    CURVE_SPLINE_BASIS,
    fit_lineage_curves as _fit_lineage_curves_impl,
    lineage_ranks as _lineage_ranks_impl,
    project_onto_curve as _project_onto_curve_impl,
    projection_scratch_bytes,
)
from ...trajectory.fate import (
    compute_fate_probabilities as _compute_fate_probabilities_impl,
    fate_solver_bytes,
)
from ...trajectory.lineage_tree import (
    cluster_distances as _cluster_distances_impl,
    cluster_moments as _cluster_moments_impl,
    spanning_tree as _spanning_tree_impl,
    tree_lineages as _tree_lineages_impl,
)
from ...trajectory.lineages import (
    LINEAGE_OK,
    LINEAGE_SCRATCH_ITEMSIZE,
    adjust_lineage_p_values as _adjust_lineage_p_values_impl,
    bin_operator_bytes as _bin_operator_bytes_impl,
    bin_smoothing_matrix as _bin_smoothing_matrix_impl,
    equal_mass_bin_operator as _equal_mass_bin_operator_impl,
    finalize_lineage_profiles as _finalize_lineage_profiles_impl,
    fisher_z_statistics as _fisher_z_statistics_impl,
    kish_effective_sizes as _kish_effective_sizes_impl,
    kish_p_values as _kish_p_values_impl,
    select_lineages as _select_lineages_impl,
    stream_lineage_profiles as _stream_lineage_profiles_impl,
    stream_weighted_correlations as _stream_weighted_correlations_impl,
)
from ...trajectory.pseudotime import (
    make_source_sink_vector as _make_source_sink_vector_impl,
    random_walk_laplacian_transpose as _random_walk_laplacian_transpose_impl,
    select_pseudotime_component as _select_pseudotime_component_impl,
    truncated_pba_potential as _truncated_pba_potential_impl,
    validate_source_sink_labels as _validate_source_sink_labels_impl,
    validate_source_sink_vector as _validate_source_sink_vector_impl,
)
from ...trajectory.results import (
    FateMappingResult,
    LineageAggregationResult,
    LineageCurvesResult,
    LineageDriverResult,
    LineageMarkerResult,
    LineageTreeResult,
    PseudotimeAggregationResult,
    PseudotimeMarkerResult,
    PseudotimeScoreResult,
)
from ...utils.arguments import integer_argument
from ...utils.arrays import array_digest, sort_categories
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _TrajectoryFeatureOperationsBase,
    )
    from .graph import _GraphOperationsMixin as _TrajectoryOperationsBase
else:
    _TrajectoryFeatureOperationsBase = object
    _TrajectoryOperationsBase = object


def _validate_normalization_identity(
    assay: Assay,
    *,
    normalization_method: dict[str, str],
    size_factor: float | None,
    context: str,
) -> None:
    # The dataset fingerprint is resolved once per public call, so only the
    # live normalization settings can be rechecked while an operation runs.
    try:
        current_method = callable_identity(assay.normMethod)
    except ValueError as exc:
        raise ValueError(
            f"{context} normalization settings changed during computation"
        ) from exc
    raw_size_factor = getattr(assay, "sf", None)
    if raw_size_factor is None:
        current_size_factor = None
    elif (
        isinstance(raw_size_factor, bool | np.bool_)
        or not isinstance(
            raw_size_factor,
            int | float | np.integer | np.floating,
        )
        or not np.isfinite(raw_size_factor)
        or float(raw_size_factor) <= 0.0
    ):
        raise ValueError(f"{context} normalization settings changed during computation")
    else:
        current_size_factor = float(raw_size_factor)
    if current_method != normalization_method or current_size_factor != size_factor:
        raise ValueError(f"{context} normalization settings changed during computation")


def _feature_identity_arrays(assay: Assay) -> tuple[Any, Any]:
    feature_data = as_zarr_group(
        assay.z["featureData"],
        name=f"{assay.name}/featureData",
    )
    names = as_zarr_array(
        feature_data["names"],
        name=f"{assay.name}/featureData/names",
    )
    ids = as_zarr_array(
        feature_data["ids"],
        name=f"{assay.name}/featureData/ids",
    )
    if names.ndim != 1 or ids.ndim != 1 or names.shape != ids.shape:
        raise ValueError("Feature identities do not align with the assay")
    return names, ids


def _feature_names_at(assay: Assay, feature_indices: np.ndarray) -> np.ndarray:
    """Return the names of the selected features, which errors cite."""
    names, _ = _feature_identity_arrays(assay)
    return np.asarray(read_array_rows_chunkwise(names, feature_indices)).astype(str)


def _validate_invalidate_cache(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("invalidate_cache must be a boolean")
    return value


def _resolve_feature_indices(
    store: Any,
    assay: Assay,
    features: ArtifactRef,
) -> tuple[ArtifactRef, np.ndarray]:
    if not isinstance(features, ArtifactRef):
        raise TypeError("features must be an ArtifactRef")
    feature_selection = store._features_resolve(assay.name, features)
    feature_indices = read_feature_selection_indices(
        store.zw,
        assay.name,
        feature_selection,
    )
    return feature_selection, feature_indices.astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class _FrozenFeatureIdentity:
    """Feature names and IDs captured when a pseudotime feature analysis starts."""

    names: Any
    ids: Any
    names_fingerprint: str
    ids_fingerprint: str

    @classmethod
    def capture(cls, assay: Assay) -> Self:
        names, ids = _feature_identity_arrays(assay)
        return cls(
            names=names,
            ids=ids,
            names_fingerprint=fingerprint_stored_strings(names),
            ids_fingerprint=fingerprint_stored_strings(ids),
        )

    def store(self, group: Any, context: str) -> None:
        """Copy the identities into ``group`` and check that they did not change."""
        stored_names = copy_metadata_array(self.names, group, "feature_names")
        stored_ids = copy_metadata_array(self.ids, group, "feature_ids")
        if (
            fingerprint_stored_strings(stored_names) != self.names_fingerprint
            or fingerprint_stored_strings(stored_ids) != self.ids_fingerprint
        ):
            raise ValueError(f"Feature identities changed during {context}")


@dataclass(frozen=True, slots=True)
class _PseudotimeFeatureCells:
    """Feature and cell axes of one pseudotime feature analysis."""

    feature_selection: ArtifactRef
    feature_indices: np.ndarray
    dataset_fingerprint: str
    pseudotime: PseudotimeScoreResult
    cell_indices: np.ndarray
    ordering: np.ndarray


@dataclass(frozen=True, slots=True)
class _PseudotimeFeatureRecord:
    """Validated lineage of a stored pseudotime feature artifact."""

    group: Any
    parameters: dict[str, Any]
    assay: Assay
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    pseudotime: ArtifactRef
    feature_indices: np.ndarray
    ids_fingerprint: str
    names_fingerprint: str
    pseudotime_result: PseudotimeScoreResult


def _default_normalization(
    assay: Assay, norm_params: dict[str, Any], caller: str
) -> dict[str, Any]:
    """Apply the unlogged, whole-library defaults of pseudotime feature analyses.

    A flag that the assay's normalizer cannot apply raises ``ValueError``
    when it is True.
    """
    return resolve_normalization_params(assay, norm_params, caller=caller)


def _normalization_guard(
    assay: Assay,
    validated_parameters: Mapping[str, Any],
    context: str,
) -> Callable[[], None]:
    """Return a check that the normalization settings did not change."""
    return partial(
        _validate_normalization_identity,
        assay,
        normalization_method=validated_parameters["normalization_method"],
        size_factor=validated_parameters["size_factor"],
        context=context,
    )


def _write_feature_vector(group: Any, name: str, values: np.ndarray) -> None:
    output = create_zarr_dataset(
        group,
        name,
        (min(max(len(values), 1), 100_000),),
        values.dtype,
        values.shape,
    )
    output[:] = values


def _write_feature_matrix(
    group: Any,
    name: str,
    feature_indices: np.ndarray,
    values: np.ndarray,
    n_features: int,
) -> None:
    """Write features-by-lineages statistics, missing for unselected features."""
    full = np.full((n_features, values.shape[1]), np.nan, dtype=np.float64)
    full[feature_indices] = values
    output = create_zarr_dataset(
        group,
        name,
        (min(max(n_features, 1), 100_000), values.shape[1]),
        np.float64,
        full.shape,
    )
    output[...] = full


def _stored_values(group: Any, name: str) -> np.ndarray:
    """Read one stored payload array in full."""
    return np.asarray(as_zarr_array(group[name], name=name)[...])


@dataclass(frozen=True, slots=True)
class _FeatureRecordHead:
    """The validated record of an assay feature artifact and its feature identity."""

    path: str
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    assay: Assay
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    upstream: ArtifactRef
    feature_indices: np.ndarray
    ids_fingerprint: str
    names_fingerprint: str


@dataclass(frozen=True, slots=True)
class _FateInputs:
    """A fate map with the pseudotime and sink labels that validated it."""

    fate: FateMappingResult
    pseudotime: PseudotimeScoreResult
    sink_values: np.ndarray


@dataclass(frozen=True, slots=True)
class _LineageCells:
    """The scored cells of a fate map, their order, and their lineage weights."""

    fate: FateMappingResult
    lineages: tuple[Any, ...]
    selected_cell_indices: np.ndarray
    cell_indices: np.ndarray
    ordering: np.ndarray
    weights: np.ndarray
    absorbing: np.ndarray

    @property
    def nbytes(self) -> int:
        """Bytes these cells hold, including the fate probabilities they keep."""
        return int(
            sum(
                values.nbytes
                for values in (
                    self.selected_cell_indices,
                    self.cell_indices,
                    self.ordering,
                    self.weights,
                    self.absorbing,
                    self.fate.values,
                    self.fate.valid,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class _LineageFeatureRecord:
    """Validated lineage of a stored lineage feature artifact."""

    group: Any
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    assay: Assay
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    fate: ArtifactRef
    feature_indices: np.ndarray
    ids_fingerprint: str
    names_fingerprint: str
    axes: _LineageCells


class _TrajectoryOperationsMixin(_TrajectoryOperationsBase):
    @validation_scoped
    def _imputation_diffusion(
        self,
        graph: ArtifactRef,
        *,
        t: int = 2,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Compute and persist a sparse MAGIC diffusion operator.

        The returned artifact is aligned to the graph's exact stored cell
        selection. Repeating the same graph and diffusion power reuses the
        complete artifact unless ``invalidate_cache`` is true.

        Raises:
            MemoryError: A diffusion step, or loading the finished operator,
                would exceed the datastore memory budget. The check runs
                before anything is persisted.
        """
        from ...neighbors.diffusion import bounded_diffusion_operator

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        require_graph_kind(graph)
        power = integer_argument(t, "t", minimum=1)
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        graph_ref = graph
        graph_status = inspect_artifact(self.zw, graph_ref)
        if not graph_status.complete:
            raise ValueError("Graph artifact is unavailable or incomplete")
        selection = graph_cell_selection(self.zw, graph_ref)
        validated_selection = validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        n_cells, _ = self._get_graph_ncells_k(graph_status.path)
        if n_cells != validated_selection.selected_count:
            raise ValueError(
                "Graph cell count does not match its stored cell selection"
            )
        planned = plan_artifact(
            self.zw,
            scope=graph_ref.scope,
            assay=graph_ref.assay,
            kind="diffusion_operator",
            operation="run_diffusion_operator",
            parameters={"t": power},
            inputs={
                "connectivity_map": graph_ref,
                "cell_selection": selection,
            },
            execution_options={"invalidate_cache": invalidate_cache},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("row", shape=(None,), dtype=np.uint64),
                ArrayRequirement("col", shape=(None,), dtype=np.uint64),
                ArrayRequirement("data", shape=(None,), dtype=np.float64),
            ),
            required_attributes=(
                AttributeRequirement(
                    "n_cells",
                    expected_types=(int, np.integer),
                    predicate=lambda value: (
                        not isinstance(value, bool) and int(value) == n_cells
                    ),
                ),
                AttributeRequirement(
                    "payload_fingerprint",
                    expected_types=(str,),
                ),
            ),
            reuse_validator=lambda _ref, group: _diffusion_payload_is_valid(
                group,
                n_cells=n_cells,
            ),
        )
        if planned.reused:
            reused_artifact_group(self.zw, planned)
            return planned.ref
        self._require_writable("run_diffusion_operator")

        graph_matrix = self._load_graph_artifact(
            graph_ref,
            symmetric=True,
            upper_only=False,
            use_k=None,
        )
        diff_op = bounded_diffusion_operator(
            graph_matrix,
            power,
            memory_bytes=self.memoryBytes,
        )
        del graph_matrix
        # Persist only an operator that loaders can read under this budget.
        nnz = int(diff_op.indptr[-1])
        if diffusion_load_bytes(nnz, n_cells) >= self.memoryBytes:
            raise MemoryError(
                f"The diffusion operator has {nnz} entries and could not be "
                f"loaded within the memory budget of {self.memoryBytes} bytes; "
                "increase the memory budget or use a smaller diffusion power t."
            )
        with artifact_transaction(self.zw, planned) as store:
            write_diffusion_payload(store, diff_op)
        return planned.ref

    def _load_diffusion_operator_with_lineage(
        self,
        diffusion: ArtifactRef,
        *,
        imputed_features: int = 0,
    ) -> tuple[coo_matrix, ArtifactRef, ArtifactRef]:
        if not isinstance(diffusion, ArtifactRef):
            raise TypeError("diffusion must be an ArtifactRef")
        if diffusion.kind != "diffusion_operator":
            raise ValueError("diffusion must be a diffusion_operator artifact")
        status = inspect_artifact(self.zw, diffusion)
        if not status.complete:
            raise ValueError("Diffusion-operator artifact is unavailable or incomplete")
        if status.operation != "run_diffusion_operator":
            raise ValueError(
                "Diffusion-operator artifact was not produced by run_diffusion_operator"
            )
        parameters = status.parameters or {}
        if set(parameters) != {"t"}:
            raise ValueError("Diffusion-operator parameters are malformed")
        try:
            integer_argument(parameters["t"], "t", minimum=1)
        except (TypeError, ValueError) as error:
            raise ValueError("Diffusion-operator power is malformed") from error
        inputs = status.inputs or {}
        if set(inputs) != {"connectivity_map", "cell_selection"}:
            raise ValueError("Diffusion-operator lineage inputs are malformed")
        graph = parse_artifact_ref(inputs["connectivity_map"], "Graph input")
        selection = parse_artifact_ref(inputs["cell_selection"], "Cell-selection input")
        if diffusion.scope != graph.scope or diffusion.assay != graph.assay:
            raise ValueError("Diffusion-operator scope does not match its graph input")
        graph_status = inspect_artifact(self.zw, graph)
        if not graph_status.complete:
            raise ValueError("Diffusion-operator graph is unavailable or incomplete")
        graph_selection = graph_cell_selection(self.zw, graph)
        if selection != graph_selection:
            raise ValueError(
                "Diffusion-operator cell selection does not match its graph lineage"
            )
        validated_selection = validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        graph_n_cells, _ = self._get_graph_ncells_k(graph_status.path)
        if graph_n_cells != validated_selection.selected_count:
            raise ValueError(
                "Diffusion-operator graph count does not match its cell selection"
            )

        group = as_zarr_group(self.zw[status.path], name=status.path)
        operator = load_diffusion_payload(
            group,
            n_cells=graph_n_cells,
            memory_bytes=self.memoryBytes,
            imputed_features=imputed_features,
        )
        return operator, graph, selection

    @validation_scoped
    def _imputation_load_diffusion(self, diffusion: ArtifactRef) -> coo_matrix:
        """Load a validated diffusion-operator artifact as a COO matrix."""
        operator, _graph, _selection = self._load_diffusion_operator_with_lineage(
            diffusion
        )
        return operator

    @validation_scoped
    def _imputation_compute_imputed(
        self,
        feature_name: str | Sequence[str] | np.ndarray | pd.Series,
        diffusion: ArtifactRef,
        *,
        from_assay: str | None = None,
    ) -> np.ndarray:
        """Impute feature values by diffusing along the KNN graph (MAGIC-style).

        Each name is resolved in two steps. A name that exactly matches a live
        cell metadata column (case-sensitive) diffuses that column's values.
        Any other name matches assay feature names case-insensitively, and
        the mean of all matching features is diffused when a name is not
        unique. A metadata column whose linked missing mask flags a cell in
        the operator's cell selection raises ``ValueError``, because diffusion
        would spread its stored placeholder to neighbouring cells.

        Args:
            feature_name: One name, or a sequence, one-dimensional string
                array, or Series of names in the desired output order.
            diffusion: Explicit diffusion-operator artifact returned by
                ``imputation.diffusion``.
            from_assay: Assay whose features are imputed. A native graph's
                operator uses the graph's own assay, which a given value must
                match. An integrated graph's operator requires it unless every
                name is a cell metadata column.

        Returns:
            A vector for one name, or a cells-by-features array for several
            names.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a cell of the operator.
        """
        single_feature = isinstance(feature_name, str)
        if isinstance(feature_name, str):
            names = [feature_name]
        elif isinstance(feature_name, np.ndarray | pd.Series):
            if feature_name.ndim != 1:
                raise ValueError("feature_name arrays must be one-dimensional")
            names = feature_name.tolist()
        elif isinstance(feature_name, Sequence):
            names = list(feature_name)
        else:
            raise TypeError("feature_name must be a string or a sequence of strings")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_name must contain non-empty strings")
        diff_op, graph, selection = self._load_diffusion_operator_with_lineage(
            diffusion, imputed_features=len(names)
        )
        cell_indices = read_stored_selection_indices(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        metadata_columns = set(self.cells.columns)
        metadata_slots = [i for i, name in enumerate(names) if name in metadata_columns]
        for slot in metadata_slots:
            missing = read_metadata_missing_rows_chunkwise(
                self.cells,
                names[slot],
                cell_indices,
            )
            if missing is not None and missing.any():
                raise ValueError(
                    f"Cell metadata column {names[slot]!r} contains missing values "
                    "in the diffusion operator's cell selection. Build the "
                    "diffusion operator over a selection without them, such as "
                    "one from qc.filter, which excludes missing values"
                )
        feature_slots = [
            i for i, name in enumerate(names) if name not in metadata_columns
        ]
        # Metadata columns need no assay; a given from_assay is still checked.
        assay_name = (
            resolve_graph_source_assay(
                self.zw,
                graph,
                from_assay,
                parameter_name="from_assay",
            )
            if feature_slots or from_assay is not None
            else None
        )
        resolved: list[ResolvedFeature] = []
        if feature_slots and assay_name is not None:
            # Feature names read the assay; metadata columns read no assay.
            self._require_measured_cells(
                assay_name, cell_indices, operation="get_imputed", remedy="graph"
            )
            assay = self._get_assay(assay_name)
            feature_names = assay.feats.fetch_all("names")
            feature_ids = assay.feats.fetch_all("ids")
            name_index = CaseInsensitiveIndex(feature_names)
            for slot in feature_slots:
                name = names[slot]
                indices = name_index.positions(name)
                if not indices:
                    raise ValueError(f"ERROR: {name} not found in {assay_name} assay.")
                if len(indices) > 1:
                    logger.warning(
                        f"Imputing the mean of {len(indices)} features because "
                        f"{name} is not unique."
                    )
                resolved.append(
                    ResolvedFeature(
                        assay=assay_name,
                        by="name",
                        indices=tuple(indices),
                        ids=tuple(str(feature_ids[i]) for i in indices),
                        names=tuple(str(feature_names[i]) for i in indices),
                        label=name,
                        reduction="mean" if len(indices) > 1 else None,
                        raw=name,
                    )
                )

        sparse = diff_op.tocsc()
        del diff_op
        output_bytes = len(cell_indices) * len(names) * np.dtype(np.float64).itemsize
        # Reserve the sparse column slice and dense streaming temporaries.
        resident_bytes = 3 * output_bytes + 2 * sum(
            array.nbytes for array in (sparse.data, sparse.indices, sparse.indptr)
        )
        if resident_bytes >= self.memoryBytes:
            raise MemoryError(
                "Imputed output and diffusion operator exceed the memory budget; "
                "increase the memory budget or request fewer features per call."
            )
        result = np.zeros((len(cell_indices), len(names)), dtype=np.float64)
        for slot in metadata_slots:
            values = np.asarray(
                read_metadata_rows_chunkwise(self.cells, names[slot], cell_indices),
                dtype=np.float64,
            )
            result[:, slot] = sparse.dot(values)
        for slots, start, values in iter_normalized_feature_blocks(
            self, resolved, cell_indices, resident_bytes=resident_bytes
        ):
            imputed = sparse[:, start : start + len(values)].dot(values)
            for column, slot in enumerate(slots):
                result[:, feature_slots[slot]] += imputed[:, column]
        return result[:, 0] if single_feature else result

    @validation_scoped
    def _trajectory_pseudotime(
        self,
        graph: ArtifactRef,
        *,
        n_singular_vals: int = 30,
        source_sink: ArtifactRef | None = None,
        sources: list[Any] | None = None,
        sinks: list[Any] | None = None,
        ss_vec: np.ndarray | None = None,
        min_max_norm_ptime: bool = True,
        random_seed: int = 4444,
        component_policy: Literal["largest", "error"] = "largest",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Calculate pseudotime and return an immutable cell artifact.

        This is a reimplementation of the population balance analysis approach
        from Weinreb et al. 2018. The graph's stored cell selection defines the
        output axis. Disconnected cells excluded by ``component_policy`` remain
        represented with NaN values and a false ``valid`` entry.

        Args:
            graph: Explicit connectivity-map or integrated-graph artifact.
            n_singular_vals: Number of smallest singular modes requested, including the null mode.
            source_sink: Explicit axis-aligned label artifact. Required with
                ``sources`` or ``sinks`` and omitted when ``ss_vec`` is supplied.
            sources: A list of group/cluster ids from ``source_sink`` to be treated as sources. Sources are
                     usually progenitor/precursor or other actively dividing cell states.
            sinks: A list of group/cluster ids from ``source_sink`` to be treated as sinks. Sinks are usually
                   more differentiated (or terminally differentiated) cell states.
            ss_vec: Custom source-sink value for each selected cell. It must sum
                to zero, with negative values for sources and positive values for
                sinks. This is mutually exclusive with label-based arguments.
            min_max_norm_ptime: Whether to perform min-max normalization on the final pseudotime values so that values
                                are in 0 to 1 range. (Default: True)
            random_seed: A random seed for svds (Default: 4444)
            component_policy: How to handle a disconnected selected graph. ``'largest'`` scores the largest connected
                              component and marks other selected cells as unscored. ``'error'`` raises instead.

        Returns:
            Reference to an artifact containing ``pseudotime`` and ``valid``.
        """

        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        require_graph_kind(graph)
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        if sources is not None and not isinstance(sources, list):
            raise TypeError("sources must be a list")
        if sinks is not None and not isinstance(sinks, list):
            raise TypeError("sinks must be a list")
        validated_parameters = _validate_pseudotime_parameters(
            {
                "n_singular_vals": n_singular_vals,
                "sources": [] if sources is None else sources,
                "sinks": [] if sinks is None else sinks,
                "min_max_norm_ptime": min_max_norm_ptime,
                "random_seed": random_seed,
                "component_policy": component_policy,
            }
        )
        n_singular_vals = validated_parameters["n_singular_vals"]
        source_labels = list(validated_parameters["sources"])
        sink_labels = list(validated_parameters["sinks"])
        min_max_norm_ptime = validated_parameters["min_max_norm_ptime"]
        random_seed = validated_parameters["random_seed"]
        component_policy = cast(
            Literal["largest", "error"],
            validated_parameters["component_policy"],
        )
        graph_ref = graph
        stored_selection = graph_cell_selection(self.zw, graph_ref)

        logger.info(f"Pseudotime scoring: loading graph {graph_ref.artifact_id}")
        graph_matrix = self._load_graph_artifact(
            graph_ref,
            symmetric=True,
            upper_only=False,
            use_k=None,
        )

        if graph_matrix.shape[0] == 0:
            raise ValueError("No cells were selected for pseudotime scoring")
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            stored_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if graph_matrix.shape[0] != len(selected_cell_indices):
            raise ValueError(
                "Graph cell count does not match its stored cell selection"
            )
        retained_mask, component_sizes = _select_pseudotime_component_impl(
            graph_matrix,
            selected_cell_indices,
            component_policy,
        )
        retained_graph = (
            graph_matrix
            if retained_mask.all()
            else graph_matrix[retained_mask][:, retained_mask]
        ).tocsr()
        if len(component_sizes) > 1:
            logger.warning(
                f"Selected graph components have sizes {component_sizes}. "
                f"Scoring the largest component with {retained_graph.shape[0]} cells"
            )

        retained_n_cells = retained_graph.shape[0]
        if retained_n_cells < 4:
            raise ValueError(
                "The retained graph must contain at least 4 cells for pseudotime scoring"
            )
        effective_k = min(n_singular_vals, retained_n_cells - 2)
        if effective_k != n_singular_vals:
            logger.warning(
                f"Reducing n_singular_vals from {n_singular_vals} to {effective_k} "
                "for the retained graph size"
            )

        label_arguments_supplied = (
            source_sink is not None or sources is not None or sinks is not None
        )
        if ss_vec is not None and label_arguments_supplied:
            raise ValueError(
                "Provide either ss_vec or source_sink with source/sink labels, not both"
            )
        if ss_vec is None and source_sink is None:
            if sources is not None or sinks is not None:
                raise ValueError(
                    "source_sink is required when sources or sinks are provided"
                )
            raise ValueError("Provide source/sink labels or a custom zero-sum ss_vec")

        if ss_vec is not None:
            owned_source_sink = np.array(ss_vec, copy=True)
            full_source_sink = np.array(
                _validate_source_sink_vector_impl(
                    owned_source_sink,
                    graph_matrix.shape[0],
                    "ss_vec",
                ),
                dtype=np.float64,
                copy=True,
            )
            full_source_sink.setflags(write=False)
            selected_row_ids = read_metadata_rows_chunkwise(
                self.cells,
                "ids",
                selected_cell_indices,
            )
            source_sink_input: ArtifactRef | np.ndarray = resolve_metadata_snapshot(
                self.zw,
                values=full_source_sink,
                row_ids=selected_row_ids,
                operation="snapshot_pseudotime_source_sink",
                parameters={},
                inputs={"cell_selection": stored_selection},
                source_columns=["ss_vec"],
                invalidate_cache=invalidate_cache,
            )
            retained_source_sink = _validate_source_sink_vector_impl(
                full_source_sink[retained_mask],
                retained_n_cells,
                "ss_vec restricted to the retained component",
            )
        else:
            if not source_labels and not sink_labels:
                raise ValueError("At least one source or sink label must be provided")

            assert source_sink is not None
            raw_labels, label_selection, label_missing = _load_cell_artifact_values(
                self.zw,
                source_sink,
            )
            if label_selection != stored_selection:
                raise ValueError("Source/sink labels do not match the graph selection")
            if raw_labels.ndim != 1 or raw_labels.shape[0] != graph_matrix.shape[0]:
                raise ValueError("Source/sink labels do not align with graph rows")
            selected_labels = pd.Series(
                _labels_with_missing_mask(
                    raw_labels,
                    label_missing,
                    "Source/sink labels",
                )
            )
            _validate_source_sink_labels_impl(
                selected_labels,
                source_labels,
                sink_labels,
                "the selected cells",
            )
            retained_labels = selected_labels.iloc[
                np.flatnonzero(retained_mask)
            ].reset_index(drop=True)
            _validate_source_sink_labels_impl(
                retained_labels,
                source_labels,
                sink_labels,
                "the retained connected component",
            )
            retained_source_sink = _validate_source_sink_vector_impl(
                _make_source_sink_vector_impl(
                    retained_labels,
                    source_labels,
                    sink_labels,
                ),
                retained_n_cells,
                "generated source/sink vector",
            )
            source_sink_input = source_sink

        arguments = PseudotimeScoringArguments(
            connectivity_map=graph_ref,
            source_sink=source_sink_input,
            cell_selection=stored_selection,
            n_singular_vals=n_singular_vals,
            sources=tuple(source_labels),
            sinks=tuple(sink_labels),
            min_max_norm_ptime=min_max_norm_ptime,
            random_seed=random_seed,
            component_policy=component_policy,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        artifact_scope = graph_ref.scope
        planned = plan_cell_data_artifact(
            self.zw,
            scope=artifact_scope,
            assay=graph_ref.assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=stored_selection,
            arrays={
                "pseudotime": ((graph_matrix.shape[0],), "f"),
                "valid": ((graph_matrix.shape[0],), "b"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=(
                AttributeRequirement(
                    "payload_fingerprint",
                    expected_types=(str,),
                ),
            ),
            reuse_validator=lambda _ref, group: _pseudotime_payload_is_valid(
                group,
                n_cells=graph_matrix.shape[0],
                min_max_normalized=min_max_norm_ptime,
                expected_valid=retained_mask,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_pseudotime_scoring")

        logger.debug("Pseudotime scoring: constructing Laplacian")
        laplacian_transpose = _random_walk_laplacian_transpose_impl(retained_graph)
        retained_ptime = _truncated_pba_potential_impl(
            laplacian_transpose,
            effective_k,
            random_seed,
            retained_source_sink,
        )
        if not np.isfinite(retained_ptime).all():
            raise ValueError("Pseudotime calculation produced non-finite values")
        value_range = float(np.ptp(retained_ptime))
        potential_scale = max(1.0, float(np.abs(retained_ptime).max()))
        if value_range <= np.finfo(float).eps * potential_scale:
            raise ValueError("Pseudotime calculation produced a constant potential")
        if min_max_norm_ptime:
            retained_ptime = (retained_ptime - retained_ptime.min()) / value_range
            retained_ptime = np.clip(retained_ptime, 0.0, 1.0)
            if not np.isfinite(retained_ptime).all():
                raise ValueError("Pseudotime normalization produced non-finite values")

        ptime = np.full(graph_matrix.shape[0], np.nan, dtype=float)
        ptime[retained_mask] = retained_ptime
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "pseudotime": ptime,
                "valid": retained_mask,
            },
            fingerprint_payload=True,
        )
        if not retained_mask.all():
            logger.warning("Unscored cells contain NaN pseudotime in the artifact")
        logger.info(
            f"Stored pseudotime scores for "
            f"{int(retained_mask.sum())}/{len(retained_mask)} cells"
        )
        return planned.ref

    @validation_scoped
    def _trajectory_load_pseudotime(
        self,
        ref: ArtifactRef,
    ) -> PseudotimeScoreResult:
        """Load pseudotime values from an explicit completed artifact."""
        result, _graph_matrix = self._load_pseudotime_scoring_with_graph(ref)
        return result

    def _load_pseudotime_scoring_with_graph(
        self,
        ref: ArtifactRef,
    ) -> tuple[PseudotimeScoreResult, csr_matrix]:
        """Load pseudotime values and the symmetric graph used to validate them."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "pseudotime":
            raise ValueError("ref must be a pseudotime artifact")
        status = inspect_artifact(self.zw, ref)
        if (
            not status.exists
            or not status.complete
            or status.operation != "run_pseudotime_scoring"
        ):
            raise ValueError("Pseudotime artifact is unavailable or invalid")
        inputs = status.inputs or {}
        parameters = status.parameters or {}
        _require_exact_record_keys(inputs, _PSEUDOTIME_INPUTS, "Pseudotime inputs")
        _require_exact_record_keys(
            parameters,
            _PSEUDOTIME_PARAMETERS,
            "Pseudotime parameters",
        )
        try:
            validated_parameters = _validate_pseudotime_parameters(parameters)
        except (TypeError, ValueError) as exc:
            raise ValueError("Pseudotime parameters are malformed") from exc
        graph = parse_artifact_ref(
            inputs.get("connectivity_map"), "Pseudotime graph input"
        )
        selection = parse_artifact_ref(
            inputs.get("cell_selection"),
            "Pseudotime cell selection input",
        )
        source_sink = parse_artifact_ref(
            inputs.get("source_sink"),
            "Pseudotime source/sink input",
        )
        graph_selection = graph_cell_selection(self.zw, graph)
        if graph_selection != selection:
            raise ValueError("Pseudotime graph and cell selection do not match")
        if ref.scope != graph.scope or ref.assay != graph.assay:
            raise ValueError("Pseudotime artifact does not share its graph scope")
        n_cells = _selection_size(self.zw, selection)
        min_max_normalized = validated_parameters["min_max_norm_ptime"]
        component_policy = validated_parameters["component_policy"]
        selected_indices = read_stored_selection_indices(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        graph_matrix = self._load_graph_artifact(
            graph,
            symmetric=True,
            upper_only=False,
            use_k=None,
        )
        if graph_matrix.shape != (n_cells, n_cells):
            raise ValueError("Pseudotime graph does not match its stored selection")
        expected_valid, _ = _select_pseudotime_component_impl(
            graph_matrix,
            selected_indices,
            cast(Literal["largest", "error"], component_policy),
        )
        group = as_zarr_group(self.zw[status.path], name=status.path)
        if not _pseudotime_payload_is_valid(
            group,
            n_cells=n_cells,
            min_max_normalized=min_max_normalized,
            expected_valid=expected_valid,
        ):
            raise ValueError("Pseudotime artifact payload is invalid")
        # The validated payload has no missing masks and uses the recorded selection.
        values, _, _ = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="pseudotime",
        )
        valid, _, _ = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="valid",
        )
        source_values, source_selection, source_missing = _load_cell_artifact_values(
            self.zw,
            source_sink,
        )
        if source_selection != selection:
            raise ValueError("Pseudotime source/sink input uses a different selection")
        raw_sources = list(validated_parameters["sources"])
        raw_sinks = list(validated_parameters["sinks"])
        if not raw_sources and not raw_sinks:
            source_status = inspect_artifact(self.zw, source_sink)
            source_inputs = source_status.inputs or {}
            source_parameters = source_status.parameters or {}
            if (
                source_sink.kind != "metadata_snapshot"
                or source_sink.scope != "datastore"
                or source_sink.assay is not None
                or source_status.operation != "snapshot_pseudotime_source_sink"
                or set(source_inputs)
                != {
                    "cell_selection",
                    "ordered_row_ids_fingerprint",
                    "values_fingerprint",
                }
                or set(source_parameters) != {"shape"}
                or source_parameters.get("shape") != [n_cells]
                or source_inputs.get("cell_selection") != selection.to_dict()
            ):
                raise ValueError("Pseudotime source/sink snapshot is invalid")
            selected_ids = read_metadata_rows_chunkwise(
                self.cells,
                "ids",
                selected_indices,
            )
            source_vector = _validate_source_sink_vector_impl(
                source_values,
                n_cells,
                "stored source/sink vector",
            )
            if (
                source_missing is not None
                or source_inputs.get("ordered_row_ids_fingerprint")
                != fingerprint_strings(selected_ids)
                or source_inputs.get("values_fingerprint")
                != fingerprint_array(source_vector)
            ):
                raise ValueError("Pseudotime source/sink snapshot is invalid")
            _validate_source_sink_vector_impl(
                source_vector[expected_valid],
                int(expected_valid.sum()),
                "stored source/sink vector restricted to the retained component",
            )
        else:
            source_labels = pd.Series(
                _labels_with_missing_mask(
                    source_values,
                    source_missing,
                    "Pseudotime source/sink labels",
                )
            )
            _validate_source_sink_labels_impl(
                source_labels,
                raw_sources,
                raw_sinks,
                "the stored pseudotime cells",
            )
            _validate_source_sink_labels_impl(
                source_labels.iloc[np.flatnonzero(expected_valid)].reset_index(
                    drop=True
                ),
                raw_sources,
                raw_sinks,
                "the stored retained pseudotime component",
            )
        result = PseudotimeScoreResult(
            ref=ref,
            graph=graph,
            cell_selection=selection,
            values=np.asarray(values, dtype=np.float64),
            valid=np.asarray(valid, dtype=bool),
        )
        return result, graph_matrix

    @validation_scoped
    def _trajectory_fate(
        self,
        pseudotime: ArtifactRef,
        sink_labels: ArtifactRef,
        *,
        sinks: list[Any] | None = None,
        beta: float = 10.0,
        solver_tol: float = 1e-6,
        max_iterations: int = 1000,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Compute fate probabilities from explicit pseudotime and labels.

        Args:
            pseudotime: Explicit pseudotime artifact. Its graph lineage is used.
            sink_labels: Explicit axis-aligned cell-label artifact.
            sinks: Ordered sink labels. Every matching selected cell becomes a
                   fate boundary.
            beta: Strength of the penalty applied to backward graph edges.
            solver_tol: Absolute tolerance on the Bellman residual of each
                solved sink column. A solve stops once its largest residual
                over all cells is at most ``solver_tol``, or half of its share
                of the validation limit when that is smaller, independent of
                sink size. Results are rejected when any residual exceeds
                ``10 * solver_tol * (len(sinks) - 1)``, bounded below by
                float32 precision and above by ``1e-3``.
            max_iterations: Maximum number of preconditioned GMRES inner
                iterations per sink. GMRES restart is fixed at 20.

        Returns:
            Reference to an artifact containing ``probabilities`` and ``valid``.
        """
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        if sinks is None:
            raise ValueError("sinks must be provided")
        if not isinstance(sinks, list):
            raise TypeError("sinks must be a list")
        if not sinks:
            raise ValueError("At least one sink label must be provided")
        validated_parameters = _validate_fate_parameters(
            {
                "sinks": sinks,
                "beta": beta,
                "solver_tol": solver_tol,
                "max_iterations": max_iterations,
            }
        )
        requested_sink_labels = tuple(validated_parameters["sinks"])
        beta = validated_parameters["beta"]
        solver_tol = validated_parameters["solver_tol"]
        max_iterations = validated_parameters["max_iterations"]
        if not isinstance(pseudotime, ArtifactRef):
            raise TypeError("pseudotime must be an ArtifactRef")
        if not isinstance(sink_labels, ArtifactRef):
            raise TypeError("sink_labels must be an ArtifactRef")
        # Validating the pseudotime loads its symmetric graph, which is the
        # same graph the fate solve needs, so it is loaded only once.
        pseudotime_result, graph_matrix = self._load_pseudotime_scoring_with_graph(
            pseudotime
        )
        graph_ref = pseudotime_result.graph
        stored_selection = pseudotime_result.cell_selection
        raw_sink_values, sink_selection, sink_missing = _load_cell_artifact_values(
            self.zw,
            sink_labels,
        )
        sink_values = _labels_with_missing_mask(
            raw_sink_values,
            sink_missing,
            "Sink labels",
        )
        if sink_selection != stored_selection:
            raise ValueError("Sink labels do not match the pseudotime cell selection")
        ptime = pseudotime_result.values
        ptime_valid = pseudotime_result.valid

        arguments = FateMappingArguments(
            connectivity_map=graph_ref,
            pseudotime=pseudotime,
            sink_labels=sink_labels,
            cell_selection=stored_selection,
            sinks=requested_sink_labels,
            beta=beta,
            solver_tol=solver_tol,
            max_iterations=max_iterations,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=graph_ref.scope,
            assay=graph_ref.assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=stored_selection,
            arrays={
                "probabilities": (
                    (len(ptime), len(requested_sink_labels)),
                    "f",
                ),
                "valid": ((len(ptime),), "b"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=(
                AttributeRequirement(
                    "payload_fingerprint",
                    expected_types=(str,),
                ),
            ),
            reuse_validator=lambda _ref, group: _fate_payload_is_valid(
                group,
                n_cells=len(ptime),
                n_sinks=len(requested_sink_labels),
                pseudotime_valid=ptime_valid,
                sink_values=sink_values,
                sink_labels=requested_sink_labels,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_fate_mapping")
        logger.info(f"Fate mapping: solving on graph {graph_ref.artifact_id}")
        retained_graph = (
            graph_matrix
            if ptime_valid.all()
            else graph_matrix[ptime_valid][:, ptime_valid]
        ).tocsr()
        del graph_matrix
        solver_bytes = fate_solver_bytes(
            retained_graph.shape[0], retained_graph.nnz, len(requested_sink_labels)
        )
        if solver_bytes >= self.memoryBytes:
            raise MemoryError(
                f"Fate mapping needs about {solver_bytes} bytes of working memory, "
                f"which exceeds the memory budget of {self.memoryBytes} bytes. "
                "Increase the memory budget or compute the pseudotime on fewer cells."
            )
        retained_ptime = ptime[ptime_valid]
        retained_sink_values = sink_values[ptime_valid]
        retained_probabilities, retained_valid, _ = _compute_fate_probabilities_impl(
            retained_graph,
            retained_ptime,
            retained_sink_values,
            list(requested_sink_labels),
            beta=beta,
            solver_tol=solver_tol,
            max_iterations=max_iterations,
            # The graph was loaded for this call, so the solve may modify it.
            _copy_graph=False,
        )
        probabilities = np.full(
            (len(ptime), len(requested_sink_labels)),
            np.nan,
            dtype=retained_probabilities.dtype,
        )
        valid = np.zeros(len(ptime), dtype=bool)
        probabilities[ptime_valid] = retained_probabilities
        valid[ptime_valid] = retained_valid
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "probabilities": probabilities,
                "valid": valid,
            },
            fingerprint_payload=True,
        )
        logger.info(
            f"Stored fate probabilities for {len(requested_sink_labels)} sinks "
            f"across {int(valid.sum())}/{len(valid)} cells"
        )
        return planned.ref

    @validation_scoped
    def _trajectory_load_fate(
        self,
        ref: ArtifactRef,
    ) -> FateMappingResult:
        """Load fate probabilities from an explicit completed artifact."""
        return self._load_fate_inputs(ref).fate

    @validation_scoped
    def _load_fate_inputs(self, ref: ArtifactRef) -> _FateInputs:
        """Load a fate map with the pseudotime and sink labels that validate it.

        Lineage operations need both again, so they keep what validation read.
        """
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "fate_map":
            raise ValueError("ref must be a fate_map artifact")
        status = inspect_artifact(self.zw, ref)
        if (
            not status.exists
            or not status.complete
            or status.operation != "run_fate_mapping"
        ):
            raise ValueError("Fate-map artifact is unavailable or invalid")
        inputs = status.inputs or {}
        parameters = status.parameters or {}
        _require_exact_record_keys(inputs, _FATE_INPUTS, "Fate-map inputs")
        _require_exact_record_keys(parameters, _FATE_PARAMETERS, "Fate-map parameters")
        try:
            validated_parameters = _validate_fate_parameters(parameters)
        except (TypeError, ValueError) as exc:
            raise ValueError("Fate-map parameters are malformed") from exc
        graph = parse_artifact_ref(
            inputs.get("connectivity_map"), "Fate-map graph input"
        )
        pseudotime = parse_artifact_ref(
            inputs.get("pseudotime"),
            "Fate-map pseudotime input",
        )
        sink_labels = parse_artifact_ref(
            inputs.get("sink_labels"),
            "Fate-map sink labels input",
        )
        selection = parse_artifact_ref(
            inputs.get("cell_selection"),
            "Fate-map cell selection input",
        )
        raw_sinks = list(validated_parameters["sinks"])
        pseudotime_result = self._trajectory_load_pseudotime(pseudotime)
        if (
            pseudotime_result.graph != graph
            or pseudotime_result.cell_selection != selection
        ):
            raise ValueError("Fate-map lineage does not match its pseudotime")
        if ref.scope != graph.scope or ref.assay != graph.assay:
            raise ValueError("Fate-map artifact does not share its graph scope")
        sink_values, sink_selection, sink_missing = _load_cell_artifact_values(
            self.zw,
            sink_labels,
        )
        if sink_selection != selection:
            raise ValueError("Fate-map sink labels use a different cell selection")
        masked_sink_values = _labels_with_missing_mask(
            sink_values,
            sink_missing,
            "Fate-map sink labels",
        )
        retained_labels = masked_sink_values[pseudotime_result.valid]
        for sink in raw_sinks:
            matches = np.asarray(retained_labels == sink)
            if matches.ndim != 1 or matches.dtype.kind != "b" or not matches.any():
                raise ValueError("Fate-map sink labels are malformed")
        n_cells = _selection_size(self.zw, selection)
        group = as_zarr_group(self.zw[status.path], name=status.path)
        if not _fate_payload_is_valid(
            group,
            n_cells=n_cells,
            n_sinks=len(raw_sinks),
            pseudotime_valid=pseudotime_result.valid,
            sink_values=masked_sink_values,
            sink_labels=raw_sinks,
        ):
            raise ValueError("Fate-map artifact payload is invalid")
        # The validated payload has no missing masks and uses the recorded selection.
        probabilities, _, _ = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="probabilities",
        )
        valid, _, _ = _load_cell_artifact_values(
            self.zw,
            ref,
            value_name="valid",
        )
        return _FateInputs(
            fate=FateMappingResult(
                ref=ref,
                graph=graph,
                pseudotime=pseudotime,
                sink_labels_artifact=sink_labels,
                cell_selection=selection,
                sink_labels=tuple(raw_sinks),
                values=np.asarray(probabilities),
                valid=np.asarray(valid, dtype=bool),
            ),
            pseudotime=pseudotime_result,
            sink_values=masked_sink_values,
        )

    def _lineage_axes(
        self,
        inputs: _FateInputs,
        lineages: list[Any] | tuple[Any, ...] | None,
    ) -> _LineageCells:
        """Return the scored cells, ordering, weights, and sinks of a fate map.

        The pseudotime and sink labels are those that validated the fate map,
        so nothing is read twice.
        """
        fate_result = inputs.fate
        lineage_labels, columns = _select_lineages_impl(
            fate_result.sink_labels, lineages
        )
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            fate_result.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        valid = fate_result.valid
        cell_indices = np.asarray(selected_cell_indices[valid], dtype=np.int64)
        ordering = validate_pseudotime_regressor(
            inputs.pseudotime.values[valid],
            len(cell_indices),
            "pseudotime artifact",
            "pseudotime validity mask",
            has_validity_column=True,
        )
        scored_labels = inputs.sink_values[valid]
        absorbing = np.zeros(len(cell_indices), dtype=bool)
        for sink in fate_result.sink_labels:
            absorbing |= np.asarray(scored_labels == sink, dtype=bool)
        return _LineageCells(
            fate=fate_result,
            lineages=lineage_labels,
            selected_cell_indices=np.asarray(selected_cell_indices, dtype=np.int64),
            cell_indices=cell_indices,
            ordering=np.asarray(ordering, dtype=np.float64),
            weights=np.asarray(fate_result.values[valid][:, columns], dtype=np.float64),
            absorbing=absorbing,
        )

    def _lineage_coordinates(
        self,
        graph: ArtifactRef,
        coordinates: ArtifactRef | None,
        cell_selection: ArtifactRef,
    ) -> ArtifactRef:
        """Return the coordinates of a lineage analysis, by default the graph's."""
        if coordinates is None:
            if graph.kind == "integrated_graph":
                raise ValueError(
                    "An integrated graph has no coordinates of its own; pass "
                    "coordinates of the same cells, such as one assay's reduction"
                )
            return resolve_native_graph_inputs(self.zw, graph).coordinates
        if not isinstance(coordinates, ArtifactRef):
            raise TypeError("coordinates must be an ArtifactRef or None")
        if coordinates.kind not in NATIVE_COORDINATE_KINDS:
            raise ValueError(
                "coordinates must be a reduction, batch_correction, "
                "imported_coordinates, or normalized artifact"
            )
        if resolve_coordinate_inputs(self.zw, coordinates).cell_selection != (
            cell_selection
        ):
            raise ValueError("coordinates must hold the cells of the trajectory")
        return coordinates

    def _lineage_coordinate_blocks(
        self,
        data: ChunkedArray,
        *,
        message: str,
        resident_bytes: int,
        scratch_bytes_per_row: int,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(start, block)`` row blocks of coordinates under the budget."""
        start = 0
        for block in data._stream_blocks(
            nthreads=self.nthreads,
            msg=message,
            prefetch=None,
            row_mask=None,
            resident_bytes=resident_bytes,
            memory_bytes=self.memoryBytes,
            scratch_bytes_per_row=scratch_bytes_per_row,
        ):
            values = np.asarray(block, dtype=np.float64)
            yield start, values
            start += values.shape[0]

    @validation_scoped
    def _trajectory_curves(
        self,
        fate: ArtifactRef,
        *,
        lineages: list[Any] | None = None,
        coordinates: ArtifactRef | None = None,
        n_points: int = 100,
        df: float = 10.0,
        batch_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Fit one fate-weighted curve per lineage and place every cell on them.

        A lineage's curve is the fate-weighted mean of the cells' coordinates
        as a smooth function of their weighted pseudotime rank along the
        lineage: the share of its fate weight carried by cells earlier in
        pseudotime. Cells near a lineage's sink weigh most in its curve, and
        cells of a shared trunk shape every curve, so the curves share the
        trunk and part where the fate probabilities part. Each coordinate is a
        penalized cubic spline of the rank with ``df`` effective degrees of
        freedom (Eilers and Marx 1996), fitted to fine rank bins in one pass
        over the coordinates. These are principal curves (Hastie and Stuetzle
        1989) whose projection index stays at the pseudotime rank; their
        reprojection iterations are not run, because they fold curves into
        loops inside compact start and end clusters.

        Every scored cell is then projected orthogonally onto every curve.
        Its ``progress`` is the rank of the nearest curve point, from zero at
        the start to one at the sink, and its ``distance`` the Euclidean
        distance to it. Draw the curves with
        :meth:`~cytearc.datastore.plot_accessor.DataStorePlotAccessor.lineage_curves`
        and place other cells with
        :meth:`~cytearc.trajectory.LineageCurvesResult.project`.

        Args:
            fate: Explicit fate-map artifact. Its pseudotime orders the cells
                and its probabilities weight them.
            lineages: Sink labels to fit, in order. ``None`` fits every sink.
            coordinates: Coordinates of the fate map's cells to fit the curves
                in. ``None`` uses the coordinates its graph was built on; an
                integrated graph has none, so it needs this argument.
            n_points: Points on each curve.
            df: Effective degrees of freedom of each coordinate's spline, from
                just above 2, a straight line, to below 32. Larger values
                follow sharper bends and noise.
            batch_size: Rows per coordinate block. ``None`` sizes blocks to
                the memory budget.
            invalidate_cache: Fit new curves even when matching ones exist.

        Returns:
            Reference to an artifact with ``points``, ``curve_pseudotime``,
            and each cell's ``progress``, ``distance``, and ``valid`` entry.
        """
        if not isinstance(fate, ArtifactRef):
            raise TypeError("fate must be an ArtifactRef")
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        if lineages is not None and not isinstance(lineages, list):
            raise TypeError("lineages must be a list of sink labels or None")
        fate_inputs = self._load_fate_inputs(fate)
        fate_result = fate_inputs.fate
        axes = self._lineage_axes(fate_inputs, lineages)
        lineage_labels = axes.lineages
        validated = _validate_lineage_curve_parameters(
            {
                "lineages": list(lineage_labels),
                "n_points": n_points,
                "df": df,
                "fit_bins": CURVE_FIT_BINS,
                "spline_basis": CURVE_SPLINE_BASIS,
            }
        )
        n_points = validated["n_points"]
        degrees_of_freedom = validated["df"]
        selection = fate_result.cell_selection
        coordinates_ref = self._lineage_coordinates(
            fate_result.graph, coordinates, selection
        )
        valid = fate_result.valid
        n_cells = int(valid.shape[0])
        n_lineages = len(lineage_labels)
        data = self._coordinate_data(coordinates_ref, batch_size=batch_size)
        if int(data.shape[0]) != n_cells:
            raise ValueError("Coordinates do not align with the fate map's cells")
        n_dims = int(data.shape[1])
        arguments = LineageCurveArguments(
            fate=fate,
            coordinates=coordinates_ref,
            cell_selection=selection,
            lineages=lineage_labels,
            n_points=n_points,
            df=degrees_of_freedom,
            fit_bins=CURVE_FIT_BINS,
            spline_basis=CURVE_SPLINE_BASIS,
            batch_size=batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope=fate.scope,
            assay=fate.assay,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "points", shape=(n_lineages, n_points, n_dims), dtype_kind="f"
                ),
                ArrayRequirement(
                    "curve_pseudotime", shape=(n_lineages, n_points), dtype_kind="f"
                ),
                ArrayRequirement(
                    "progress", shape=(n_cells, n_lineages), dtype_kind="f"
                ),
                ArrayRequirement(
                    "distance", shape=(n_cells, n_lineages), dtype_kind="f"
                ),
                ArrayRequirement("valid", shape=(n_cells,), dtype_kind="b"),
            ),
            required_attributes=(
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
            ),
            reuse_validator=lambda _ref, group: _lineage_curve_payload_is_valid(
                group,
                n_cells=n_cells,
                n_lineages=n_lineages,
                n_points=n_points,
                n_dims=n_dims,
                expected_valid=valid,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("fit_lineage_curves")
        # Weights, ranks, and positions per cell and lineage; the pseudotime;
        # and the rank bins of every lineage.
        resident_bytes = (
            4 * n_cells * n_lineages * 8
            + n_cells * 8
            + n_lineages * CURVE_FIT_BINS * (n_dims + 3) * 8
        )
        if resident_bytes >= self.memoryBytes:
            raise MemoryError(
                f"Lineage curves need about {resident_bytes} bytes of working "
                f"memory, which exceeds the memory budget of {self.memoryBytes} "
                "bytes. Increase the memory budget or fit fewer lineages."
            )
        # The coordinates hold every cell of the selection; unscored cells
        # carry no weight and no rank.
        ordering = np.zeros(n_cells, dtype=np.float64)
        ordering[valid] = axes.ordering
        weights = np.zeros((n_cells, n_lineages), dtype=np.float64)
        weights[valid] = axes.weights
        ranks = np.full((n_cells, n_lineages), np.nan, dtype=np.float64)
        ranks[valid] = _lineage_ranks_impl(axes.ordering, axes.weights)
        del axes, fate_result, fate_inputs
        logger.info(f"Lineage curves: fitting {n_lineages} curves in {n_dims} dims")
        fit = _fit_lineage_curves_impl(
            self._lineage_coordinate_blocks(
                data,
                message="Fitting lineage curves",
                resident_bytes=resident_bytes,
                scratch_bytes_per_row=n_dims * 8,
            ),
            ranks,
            weights,
            ordering,
            n_dims=n_dims,
            n_points=n_points,
            degrees_of_freedom=degrees_of_freedom,
        )
        chunk_rows = min(max(n_cells, 1), 100_000)
        with artifact_transaction(self.zw, planned) as group:
            for name, values in (
                ("points", fit.points),
                ("curve_pseudotime", fit.ordering),
            ):
                output = create_zarr_dataset(
                    group, name, values.shape, np.float64, values.shape
                )
                output[...] = values
            positions = create_zarr_dataset(
                group,
                "progress",
                (chunk_rows, n_lineages),
                np.float64,
                (n_cells, n_lineages),
            )
            distances = create_zarr_dataset(
                group,
                "distance",
                (chunk_rows, n_lineages),
                np.float64,
                (n_cells, n_lineages),
            )
            for start, block in self._lineage_coordinate_blocks(
                data,
                message="Placing cells on lineage curves",
                resident_bytes=resident_bytes,
                scratch_bytes_per_row=(
                    n_dims * 8 + 2 * n_lineages * 8 + projection_scratch_bytes(n_points)
                ),
            ):
                stop = start + block.shape[0]
                scored = valid[start:stop]
                block_progress = np.full(
                    (block.shape[0], n_lineages), np.nan, dtype=np.float64
                )
                block_distance = np.full_like(block_progress, np.nan)
                if scored.any():
                    rows = block[scored]
                    for lineage in range(n_lineages):
                        position, distance = _project_onto_curve_impl(
                            rows,
                            fit.points[lineage],
                            fit.progress,
                            nthreads=self.nthreads,
                        )
                        block_progress[scored, lineage] = position
                        block_distance[scored, lineage] = distance
                positions[start:stop] = block_progress
                distances[start:stop] = block_distance
            valid_array = create_zarr_dataset(
                group, "valid", (chunk_rows,), np.bool_, (n_cells,)
            )
            valid_array[:] = valid
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                _LINEAGE_CURVE_PAYLOAD,
            )
        logger.info(f"Stored {n_lineages} lineage curves for {int(valid.sum())} cells")
        return planned.ref

    @validation_scoped
    def _trajectory_load_curves(
        self,
        ref: ArtifactRef,
    ) -> LineageCurvesResult:
        """Load lineage curves and cell positions from an explicit artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "lineage_curves":
            raise ValueError("ref must be a lineage_curves artifact")
        status = inspect_artifact(self.zw, ref)
        if (
            not status.exists
            or not status.complete
            or status.operation != "fit_lineage_curves"
        ):
            raise ValueError("Lineage-curve artifact is unavailable or invalid")
        inputs = status.inputs or {}
        _require_exact_record_keys(
            inputs, _LINEAGE_CURVE_INPUTS, "Lineage-curve inputs"
        )
        try:
            parameters = _validate_lineage_curve_parameters(status.parameters or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("Lineage-curve parameters are malformed") from exc
        fate = parse_artifact_ref(inputs.get("fate"), "Lineage-curve fate input")
        coordinates = parse_artifact_ref(
            inputs.get("coordinates"), "Lineage-curve coordinates input"
        )
        selection = parse_artifact_ref(
            inputs.get("cell_selection"), "Lineage-curve cell selection input"
        )
        fate_result = self._trajectory_load_fate(fate)
        if fate_result.cell_selection != selection:
            raise ValueError("Lineage curves do not use their fate map's cells")
        if ref.scope != fate.scope or ref.assay != fate.assay:
            raise ValueError("Lineage-curve artifact does not share its fate scope")
        lineage_labels = tuple(parameters["lineages"])
        try:
            _, columns = _select_lineages_impl(
                fate_result.sink_labels, list(lineage_labels)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Lineage-curve lineages are not sinks of the fate map"
            ) from exc
        self._lineage_coordinates(fate_result.graph, coordinates, selection)
        n_cells = int(fate_result.valid.shape[0])
        n_dims = int(
            as_zarr_array(
                artifact_group(self.zw, coordinates)["data"], name="data"
            ).shape[1]
        )
        group = as_zarr_group(self.zw[status.path], name=status.path)
        if not _lineage_curve_payload_is_valid(
            group,
            n_cells=n_cells,
            n_lineages=len(lineage_labels),
            n_points=parameters["n_points"],
            n_dims=n_dims,
            expected_valid=fate_result.valid,
        ):
            raise ValueError("Lineage-curve artifact payload is invalid")
        return LineageCurvesResult(
            ref=ref,
            fate=fate,
            coordinates=coordinates,
            cell_selection=selection,
            lineages=lineage_labels,
            points=np.asarray(_stored_values(group, "points"), dtype=np.float64),
            curve_pseudotime=np.asarray(
                _stored_values(group, "curve_pseudotime"), dtype=np.float64
            ),
            progress=np.asarray(_stored_values(group, "progress"), dtype=np.float64),
            distance=np.asarray(_stored_values(group, "distance"), dtype=np.float64),
            weights=np.asarray(fate_result.values[:, columns], dtype=np.float64),
            valid=np.asarray(_stored_values(group, "valid"), dtype=bool),
        )

    def _lineage_tree_clusters(
        self,
        labels: ArtifactRef,
        selection: ArtifactRef,
    ) -> tuple[list[Any], np.ndarray]:
        """Return the sorted cluster labels of a label artifact and each cell's code."""
        values, label_selection, missing = _load_cell_artifact_values(self.zw, labels)
        if label_selection != selection:
            raise ValueError("labels must hold the cells of the graph")
        if missing is not None and np.any(missing):
            raise ValueError(
                "Every cell needs a label; select the labelled cells before "
                "building the graph"
            )
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError("labels must hold one label per cell")
        # Factorize the cells in compiled code; only the distinct labels are
        # sorted and remapped in Python.
        first_codes, uniques = pd.factorize(raw, use_na_sentinel=False)
        native = [
            value.item() if isinstance(value, np.generic) else value
            for value in np.asarray(uniques, dtype=object)
        ]
        clusters = sort_categories(native)
        index = {cluster: code for code, cluster in enumerate(clusters)}
        remap = np.asarray([index[value] for value in native], dtype=np.int64)
        return clusters, remap[np.asarray(first_codes, dtype=np.int64)]

    @validation_scoped
    def _trajectory_lineage_tree(
        self,
        graph: ArtifactRef,
        labels: ArtifactRef,
        *,
        coordinates: ArtifactRef | None = None,
        root: Any = None,
        ends: list[Any] | None = None,
        batch_size: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Join clusters by a minimum spanning tree and propose lineages.

        This is the lineage step of Slingshot (Street et al. 2018). The
        distance between two clusters is
        ``(m_i - m_j)^T (S_i + S_j)^{-1} (m_i - m_j)`` for their mean
        coordinates ``m`` and covariances ``S``, using the diagonal where
        ``S_i + S_j`` is singular. A minimum spanning tree joins the clusters,
        and the paths from ``root`` to the other leaves propose lineages.
        ``ends`` makes clusters leaves: the tree first spans the others, then
        each end joins its nearest cluster.

        A tree must join every cluster, so similar terminal clusters can be
        chained through one another and the result depends on clustering
        resolution. Treat leaves as candidate sinks to check with markers
        before passing them to :meth:`fate`.

        Args:
            graph: Explicit graph artifact; its cells are the tree's cells.
            labels: Explicit cell-label artifact of the graph's cells, such
                as ``cluster_labels``. Every cell needs a label.
            coordinates: Coordinates of the graph's cells. ``None`` uses the
                coordinates the graph was built on; an integrated graph has
                none, so it needs this argument.
            root: Cluster label that starts the lineages, or ``None`` for an
                unrooted tree without lineages.
            ends: Cluster labels to hold as leaves.
            batch_size: Rows per coordinate block. ``None`` sizes blocks to
                the memory budget.
            invalidate_cache: Build a new tree even when a matching one exists.

        Returns:
            Reference to a ``lineage_tree`` artifact.
        """
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        if not isinstance(labels, ArtifactRef):
            raise TypeError("labels must be an ArtifactRef")
        require_graph_kind(graph)
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        if ends is not None and not isinstance(ends, list):
            raise TypeError("ends must be a list of cluster labels or None")
        validated = _validate_lineage_tree_parameters(
            {"root": root, "ends": [] if ends is None else ends}
        )
        selection = graph_cell_selection(self.zw, graph)
        coordinates_ref = self._lineage_coordinates(graph, coordinates, selection)
        clusters, codes = self._lineage_tree_clusters(labels, selection)
        if len(clusters) < 2:
            raise ValueError("A lineage tree needs at least two clusters")
        positions = {cluster: index for index, cluster in enumerate(clusters)}

        def cluster_index(label: Any, name: str) -> int:
            matches = [
                index for cluster, index in positions.items() if bool(cluster == label)
            ]
            if not matches:
                raise ValueError(f"{name} {label!r} is not a cluster label")
            return matches[0]

        root_index = (
            None
            if validated["root"] is None
            else cluster_index(validated["root"], "root")
        )
        # The tree does not depend on the order of the ends, so they are
        # recorded in cluster order and any order reuses one tree.
        end_pairs = sorted(
            ((cluster_index(end, "End"), end) for end in validated["ends"]),
            key=lambda pair: pair[0],
        )
        end_indices = [index for index, _ in end_pairs]
        ordered_ends = tuple(end for _, end in end_pairs)
        data = self._coordinate_data(coordinates_ref, batch_size=batch_size)
        if int(data.shape[0]) != codes.shape[0]:
            raise ValueError("Coordinates do not align with the graph's cells")
        n_dims = int(data.shape[1])
        n_clusters = len(clusters)
        arguments = LineageTreeArguments(
            labels=labels,
            coordinates=coordinates_ref,
            cell_selection=selection,
            root=validated["root"],
            ends=ordered_ends,
            batch_size=batch_size,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope=graph.scope,
            assay=graph.assay,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("sizes", shape=(n_clusters,), dtype_kind="i"),
                ArrayRequirement("means", shape=(n_clusters, n_dims), dtype_kind="f"),
                ArrayRequirement("edges", shape=(n_clusters - 1, 2), dtype_kind="i"),
                ArrayRequirement("lengths", shape=(n_clusters - 1,), dtype_kind="f"),
            ),
            required_attributes=(
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
                AttributeRequirement("cluster_labels", expected_types=(list,)),
            ),
            reuse_validator=lambda _ref, group: (
                group.attrs.get("cluster_labels") == list(clusters)
                and _lineage_tree_payload_is_valid(
                    group, n_clusters=n_clusters, n_dims=n_dims
                )
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("build_lineage_tree")
        # The codes, each cluster's mean and covariance, and for one cluster
        # pair the pooled covariance, the copy its eigenvalues are computed
        # from, and its Cholesky factor.
        resident_bytes = (
            codes.nbytes
            + n_clusters * (n_dims + n_dims * n_dims) * 8
            + (3 * n_dims * n_dims + n_dims) * 8
        )
        if resident_bytes >= self.memoryBytes:
            raise MemoryError(
                f"The lineage tree needs about {resident_bytes} bytes of working "
                f"memory, which exceeds the memory budget of {self.memoryBytes} "
                "bytes. Use coordinates with fewer dimensions."
            )
        counts, means, covariances = _cluster_moments_impl(
            self._lineage_coordinate_blocks(
                data,
                message="Summarizing clusters",
                resident_bytes=resident_bytes,
                scratch_bytes_per_row=2 * n_dims * 8,
            ),
            codes,
            n_clusters=n_clusters,
            n_dims=n_dims,
        )
        tree = _spanning_tree_impl(
            _cluster_distances_impl(means, covariances),
            root=root_index,
            ends=end_indices,
            labels=clusters,
        )
        with artifact_transaction(self.zw, planned) as group:
            for name, values in (
                ("sizes", counts.astype(np.int64)),
                ("means", means),
                ("edges", tree.edges),
                ("lengths", tree.lengths),
            ):
                output = create_zarr_dataset(
                    group, name, values.shape, values.dtype, values.shape
                )
                output[...] = values
            group.attrs["cluster_labels"] = list(clusters)
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                _LINEAGE_TREE_PAYLOAD,
            )
        logger.info(
            f"Stored a lineage tree over {n_clusters} clusters with "
            f"{len(tree.leaves)} leaves"
        )
        return planned.ref

    @validation_scoped
    def _trajectory_load_lineage_tree(
        self,
        ref: ArtifactRef,
    ) -> LineageTreeResult:
        """Load a cluster tree and its proposed lineages from an explicit artifact."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != "lineage_tree":
            raise ValueError("ref must be a lineage_tree artifact")
        status = inspect_artifact(self.zw, ref)
        if (
            not status.exists
            or not status.complete
            or status.operation != "build_lineage_tree"
        ):
            raise ValueError("Lineage-tree artifact is unavailable or invalid")
        inputs = status.inputs or {}
        _require_exact_record_keys(inputs, _LINEAGE_TREE_INPUTS, "Lineage-tree inputs")
        try:
            parameters = _validate_lineage_tree_parameters(status.parameters or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("Lineage-tree parameters are malformed") from exc
        labels = parse_artifact_ref(inputs.get("labels"), "Lineage-tree labels input")
        coordinates = parse_artifact_ref(
            inputs.get("coordinates"), "Lineage-tree coordinates input"
        )
        selection = parse_artifact_ref(
            inputs.get("cell_selection"), "Lineage-tree cell selection input"
        )
        if resolve_coordinate_inputs(self.zw, coordinates).cell_selection != (
            selection
        ):
            raise ValueError("Lineage-tree coordinates do not hold its cells")
        clusters, _codes = self._lineage_tree_clusters(labels, selection)
        n_dims = int(
            as_zarr_array(
                artifact_group(self.zw, coordinates)["data"], name="data"
            ).shape[1]
        )
        group = as_zarr_group(self.zw[status.path], name=status.path)
        if group.attrs.get("cluster_labels") != list(
            clusters
        ) or not _lineage_tree_payload_is_valid(
            group, n_clusters=len(clusters), n_dims=n_dims
        ):
            raise ValueError("Lineage-tree artifact payload is invalid")
        root = parameters["root"]
        ends = tuple(parameters["ends"])
        index = {cluster: position for position, cluster in enumerate(clusters)}
        for label in (*([] if root is None else [root]), *ends):
            if not any(bool(cluster == label) for cluster in clusters):
                raise ValueError("Lineage-tree root or ends are not cluster labels")
        edges = np.asarray(_stored_values(group, "edges"), dtype=np.int64)
        root_index = None if root is None else index[root]
        leaves, paths = _tree_lineages_impl(len(clusters), edges, root_index)
        return LineageTreeResult(
            ref=ref,
            labels=labels,
            coordinates=coordinates,
            cell_selection=selection,
            clusters=tuple(clusters),
            sizes=np.asarray(_stored_values(group, "sizes"), dtype=np.int64),
            means=np.asarray(_stored_values(group, "means"), dtype=np.float64),
            edges=pd.DataFrame(
                {
                    "source": [clusters[first] for first in edges[:, 0]],
                    "target": [clusters[second] for second in edges[:, 1]],
                    "length": np.asarray(
                        _stored_values(group, "lengths"), dtype=np.float64
                    ),
                }
            ),
            root=root,
            ends=ends,
            leaves=tuple(clusters[leaf] for leaf in leaves),
            lineages=tuple(
                tuple(clusters[cluster] for cluster in path) for path in paths
            ),
        )


class _TrajectoryFeatureOperationsMixin(_TrajectoryFeatureOperationsBase):
    def _pseudotime_feature_assay(
        self,
        pseudotime: ArtifactRef,
        features: ArtifactRef,
    ) -> Assay:
        if not isinstance(pseudotime, ArtifactRef):
            raise TypeError("pseudotime must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.scope != "assay" or features.assay is None:
            raise ValueError("features must be an assay-scoped feature selection")
        return self._get_assay(features.assay)

    def _pseudotime_feature_cells(
        self,
        assay: Assay,
        pseudotime: ArtifactRef,
        features: ArtifactRef,
    ) -> _PseudotimeFeatureCells:
        feature_selection, feature_indices = _resolve_feature_indices(
            self,
            assay,
            features,
        )
        dataset_fingerprint = self._ensure_dataset_fingerprint(assay.name)
        ptime_result = self._trajectory_load_pseudotime(pseudotime)
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            ptime_result.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cell_indices = np.asarray(
            selected_cell_indices[ptime_result.valid],
            dtype=np.int64,
        )
        ordering = validate_pseudotime_regressor(
            ptime_result.values[ptime_result.valid],
            len(cell_indices),
            "pseudotime artifact",
            "pseudotime validity mask",
            has_validity_column=True,
        )
        return _PseudotimeFeatureCells(
            feature_selection=feature_selection,
            feature_indices=feature_indices,
            dataset_fingerprint=dataset_fingerprint,
            pseudotime=ptime_result,
            cell_indices=cell_indices,
            ordering=ordering,
        )

    def _feature_record_head(
        self,
        ref: ArtifactRef,
        *,
        kind: str,
        operation: str,
        label: str,
        input_names: frozenset[str],
        validate_parameters: Callable[[Mapping[str, Any]], dict[str, Any]],
        upstream: str,
    ) -> _FeatureRecordHead:
        """Validate the record and feature identity of an assay feature artifact.

        ``upstream`` names the input the features were computed along, such as
        ``"pseudotime"`` or ``"fate"``; the caller loads and checks it.
        """
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.kind != kind or ref.scope != "assay":
            raise ValueError(f"ref must be an assay-scoped {kind} artifact")
        status = inspect_artifact(self.zw, ref)
        if not status.exists or not status.complete or status.operation != operation:
            raise ValueError(f"{label} artifact is unavailable or invalid")
        inputs = status.inputs or {}
        _require_exact_record_keys(inputs, input_names, f"{label} inputs")
        try:
            parameters = validate_parameters(status.parameters or {})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} parameters do not match the required contract; "
                f"rerun {operation} to recompute it"
            ) from exc
        cell_selection = parse_artifact_ref(
            inputs.get("cell_selection"),
            f"{label} cell selection input",
        )
        feature_selection = parse_artifact_ref(
            inputs.get("feature_selection"),
            f"{label} feature selection input",
        )
        upstream_ref = parse_artifact_ref(
            inputs.get(upstream),
            f"{label} {upstream} input",
        )
        assay = self._get_assay(ref.assay)
        _live_feature_names, live_feature_ids = _feature_identity_arrays(assay)
        ids_fingerprint = inputs.get("ordered_feature_ids_fingerprint")
        names_fingerprint = inputs.get("ordered_feature_names_fingerprint")
        if not isinstance(ids_fingerprint, str) or not isinstance(
            names_fingerprint,
            str,
        ):
            raise ValueError(f"{label} feature identities are malformed")
        if ids_fingerprint != fingerprint_stored_strings(live_feature_ids):
            raise ValueError(f"{label} feature ID identity has changed")
        if inputs.get("dataset_fingerprint") != self._ensure_dataset_fingerprint(
            assay.name
        ):
            raise ValueError(f"{label} dataset identity has changed")
        _, feature_indices = _resolve_feature_indices(
            self,
            assay,
            feature_selection,
        )
        return _FeatureRecordHead(
            path=status.path,
            inputs=inputs,
            parameters=parameters,
            assay=assay,
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            upstream=upstream_ref,
            feature_indices=feature_indices,
            ids_fingerprint=ids_fingerprint,
            names_fingerprint=names_fingerprint,
        )

    def _load_pseudotime_feature_record(
        self,
        ref: ArtifactRef,
        *,
        kind: str,
        operation: str,
        label: str,
        input_names: frozenset[str],
        validate_parameters: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> _PseudotimeFeatureRecord:
        head = self._feature_record_head(
            ref,
            kind=kind,
            operation=operation,
            label=label,
            input_names=input_names,
            validate_parameters=validate_parameters,
            upstream="pseudotime",
        )
        ptime_result = self._trajectory_load_pseudotime(head.upstream)
        if ptime_result.cell_selection != head.cell_selection:
            raise ValueError(f"{label} cell selection does not match pseudotime")
        return _PseudotimeFeatureRecord(
            group=as_zarr_group(self.zw[head.path], name=head.path),
            parameters=head.parameters,
            assay=head.assay,
            cell_selection=head.cell_selection,
            feature_selection=head.feature_selection,
            pseudotime=head.upstream,
            feature_indices=head.feature_indices,
            ids_fingerprint=head.ids_fingerprint,
            names_fingerprint=head.names_fingerprint,
            pseudotime_result=ptime_result,
        )

    @validation_scoped
    def _trajectory_markers(
        self,
        pseudotime: ArtifactRef,
        *,
        features: ArtifactRef,
        min_cells: int = 10,
        gene_batch_size: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Store feature correlations with an explicit pseudotime artifact.

        Args:
            pseudotime: Explicit pseudotime artifact defining cells and ordering.
            features: Explicit assay-scoped feature-selection artifact.
            min_cells: Minimum number of expressing cells required for a feature.
            gene_batch_size: Number of features loaded per batch. When None,
                selected features are grouped into chunk-aligned blocks that
                fit the operation memory budget.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Reference to the immutable pseudotime-marker artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a valid pseudotime cell.
        """
        from ...features.markers import find_markers_by_regression

        reject_unknown_normalization_params(
            norm_params,
            caller="run_pseudotime_marker_search",
        )
        gene_batch_size = (
            None
            if gene_batch_size is None
            else integer_argument(gene_batch_size, "gene_batch_size", minimum=1)
        )
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        assay = self._pseudotime_feature_assay(pseudotime, features)
        identity = _FrozenFeatureIdentity.capture(assay)
        validated_parameters = _validate_marker_parameters(
            {
                "normalization": _default_normalization(
                    assay, norm_params, "run_pseudotime_marker_search"
                ),
                "normalization_method": callable_identity(assay.normMethod),
                "size_factor": getattr(assay, "sf", None),
                "association_method": "pearson",
                "p_value_method": "student_t",
                "adjustment_method": "fdr_bh",
                "adjustment_scope": "tested_features",
                "min_cells": min_cells,
            }
        )
        resolved_norm_params = validated_parameters["normalization"]
        min_cells = validated_parameters["min_cells"]
        check_normalization = _normalization_guard(
            assay,
            validated_parameters,
            "Pseudotime marker search",
        )
        axes = self._pseudotime_feature_cells(assay, pseudotime, features)
        self._require_measured_cells(
            assay.name,
            axes.cell_indices,
            operation="run_pseudotime_marker_search",
            remedy="graph",
        )
        feature_index = axes.feature_indices
        logger.info(
            f"Pseudotime markers: correlating features "
            f"(cells={len(axes.cell_indices)}, features={len(feature_index)}, "
            f"batch_size={gene_batch_size if gene_batch_size is not None else 'auto'})"
        )
        arguments = PseudotimeMarkerArguments(
            cell_selection=axes.pseudotime.cell_selection,
            feature_selection=axes.feature_selection,
            pseudotime=pseudotime,
            dataset_fingerprint=axes.dataset_fingerprint,
            ordered_feature_ids_fingerprint=identity.ids_fingerprint,
            ordered_feature_names_fingerprint=identity.names_fingerprint,
            normalization=resolved_norm_params,
            normalization_method=validated_parameters["normalization_method"],
            size_factor=validated_parameters["size_factor"],
            association_method=validated_parameters["association_method"],
            p_value_method=validated_parameters["p_value_method"],
            adjustment_method=validated_parameters["adjustment_method"],
            adjustment_scope=validated_parameters["adjustment_scope"],
            min_cells=min_cells,
            gene_batch_size=gene_batch_size,
            nthreads=int(
                getattr(
                    assay,
                    "nthreads",
                    getattr(self, "nthreads", 1),
                )
            ),
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                *(
                    ArrayRequirement(
                        name,
                        shape=(assay.feats.N,),
                        dtype_kind="f",
                    )
                    for name in ("r_value", "p_value", "p_value_adjusted")
                ),
                ArrayRequirement("feature_names", shape=(assay.feats.N,)),
                ArrayRequirement("feature_ids", shape=(assay.feats.N,)),
            ),
            required_attributes=(
                AttributeRequirement(
                    "payload_fingerprint",
                    expected_types=(str,),
                ),
            ),
            reuse_validator=lambda _ref, group: _marker_payload_is_valid(
                group,
                n_features=assay.feats.N,
                selected_features=feature_index,
                expected_feature_ids_fingerprint=identity.ids_fingerprint,
                expected_feature_names_fingerprint=identity.names_fingerprint,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_pseudotime_marker_search")
        check_normalization()
        markers = find_markers_by_regression(
            assay=assay,
            cell_idx=axes.cell_indices,
            feat_idx=feature_index,
            regressor=axes.ordering,
            min_cells=min_cells,
            batch_size=gene_batch_size,
            **resolved_norm_params,
        ).reindex(feature_index)
        # A feature the regression did not return is missing after the reindex.
        if markers["r_value"].isna().any():
            raise ValueError(
                "Pseudotime marker correlations are missing or not finite for "
                "some selected features"
            )
        with artifact_transaction(self.zw, planned) as marker_group:
            check_normalization()
            for name in ("r_value", "p_value", "p_value_adjusted"):
                values = np.full(assay.feats.N, np.nan, dtype=np.float64)
                values[feature_index] = np.asarray(markers[name].values)
                _write_feature_vector(marker_group, name, values)
            identity.store(marker_group, "pseudotime marker computation")
            marker_group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                marker_group,
                _MARKER_PAYLOAD,
            )
            check_normalization()
        logger.info(f"Stored pseudotime marker scores for {len(markers)} features")
        return planned.ref

    @validation_scoped
    def _trajectory_load_markers(
        self,
        ref: ArtifactRef,
    ) -> PseudotimeMarkerResult:
        """Load a pseudotime-marker table from an explicit artifact."""
        record = self._load_pseudotime_feature_record(
            ref,
            kind="pseudotime_markers",
            operation="run_pseudotime_marker_search",
            label="Pseudotime-marker",
            input_names=_MARKER_INPUTS,
            validate_parameters=_validate_marker_parameters,
        )
        group = record.group
        feature_indices = record.feature_indices
        if not _marker_payload_is_valid(
            group,
            n_features=record.assay.feats.N,
            selected_features=feature_indices,
            expected_feature_ids_fingerprint=record.ids_fingerprint,
            expected_feature_names_fingerprint=record.names_fingerprint,
        ):
            raise ValueError("Pseudotime-marker artifact payload is invalid")
        columns = {
            name: read_array_rows_chunkwise(
                as_zarr_array(group[name], name=name),
                feature_indices,
            )
            for name in (
                "feature_names",
                "feature_ids",
                "r_value",
                "p_value",
                "p_value_adjusted",
            )
        }
        table = pd.DataFrame(
            {
                "feature_index": feature_indices,
                "feature_name": columns["feature_names"].astype(str),
                "feature_id": columns["feature_ids"].astype(str),
                "r_value": columns["r_value"],
                "p_value": columns["p_value"],
                "p_value_adjusted": columns["p_value_adjusted"],
            }
        )
        return PseudotimeMarkerResult(
            ref=ref,
            table=table,
            assay=record.assay.name,
            cell_selection=record.cell_selection,
            feature_selection=record.feature_selection,
            pseudotime=record.pseudotime,
        )

    @validation_scoped
    def _trajectory_aggregation(
        self,
        pseudotime: ArtifactRef,
        *,
        features: ArtifactRef,
        min_exp: float = 1e-3,
        window_size: int = 200,
        chunk_size: int = 50,
        smoothen: bool = True,
        z_scale: bool = True,
        n_neighbours: int = 11,
        n_clusters: int = 10,
        batch_size: int | None = None,
        ann_params: dict | None = None,
        nan_cluster_value: int = -1,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Cluster features by pseudotime and store one immutable artifact.

        Args:
            pseudotime: Explicit pseudotime artifact defining cells and ordering.
            features: Explicit assay-scoped feature-selection artifact.
            min_exp: Minimum mean normalized expression required for clustering.
            window_size: Rolling window size used to smooth feature values.
            chunk_size: Number of pseudotime bins to create.
            smoothen: Whether to smooth expression along pseudotime.
            z_scale: Whether to standardize each retained feature.
            n_neighbours: Number of neighbors in the feature graph.
            n_clusters: Number of feature modules to create.
            batch_size: Number of features processed per batch. When None,
                selected features are grouped into chunk-aligned blocks that
                fit the operation memory budget.
            ann_params: Parameters forwarded to the HNSW index.
            nan_cluster_value: Value assigned to features excluded from clustering.
            **norm_params: Extra keyword arguments forwarded to normalized expression.

        Returns:
            Reference to the immutable pseudotime-aggregation artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a valid pseudotime cell.
        """
        from ...trajectory.feature_dynamics import knn_clustering

        reject_unknown_normalization_params(
            norm_params,
            caller="run_pseudotime_aggregation",
        )
        batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        assay = self._pseudotime_feature_assay(pseudotime, features)
        identity = _FrozenFeatureIdentity.capture(assay)
        validated_parameters = _validate_aggregation_parameters(
            {
                "normalization": _default_normalization(
                    assay, norm_params, "run_pseudotime_aggregation"
                ),
                "normalization_method": callable_identity(assay.normMethod),
                "size_factor": getattr(assay, "sf", None),
                "min_exp": min_exp,
                "window_size": window_size,
                "chunk_size": chunk_size,
                "smoothen": smoothen,
                "z_scale": z_scale,
                "n_neighbours": n_neighbours,
                "n_clusters": n_clusters,
                "ann_params": {} if ann_params is None else ann_params,
                "nan_cluster_value": nan_cluster_value,
            }
        )
        resolved_norm_params = validated_parameters["normalization"]
        min_exp = validated_parameters["min_exp"]
        smoothen = validated_parameters["smoothen"]
        z_scale = validated_parameters["z_scale"]
        n_neighbours = validated_parameters["n_neighbours"]
        n_clusters = validated_parameters["n_clusters"]
        nan_cluster_value = validated_parameters["nan_cluster_value"]
        check_normalization = _normalization_guard(
            assay,
            validated_parameters,
            "Pseudotime aggregation",
        )
        axes = self._pseudotime_feature_cells(assay, pseudotime, features)
        if len(axes.feature_indices) < 2:
            raise ValueError("At least two selected features are required")
        if n_neighbours >= len(axes.feature_indices):
            raise ValueError(
                "n_neighbours must be smaller than the selected feature count"
            )
        if n_clusters > len(axes.feature_indices):
            raise ValueError("n_clusters cannot exceed the selected feature count")
        self._require_measured_cells(
            assay.name,
            axes.cell_indices,
            operation="run_pseudotime_aggregation",
            remedy="graph",
        )
        (
            cell_ordering,
            cell_indices,
            feature_indices,
            effective_window,
            effective_bins,
            input_fingerprints,
        ) = assay._prepare_aggregated_ordering(
            axes.cell_indices,
            axes.feature_indices,
            axes.ordering,
            window_size=validated_parameters["window_size"],
            chunk_size=validated_parameters["chunk_size"],
        )
        resolved_ann_params = resolve_aggregation_ann_params(
            validated_parameters["ann_params"],
            dim=effective_bins,
        )
        arguments = PseudotimeAggregationArguments(
            cell_selection=axes.pseudotime.cell_selection,
            feature_selection=axes.feature_selection,
            pseudotime=pseudotime,
            dataset_fingerprint=axes.dataset_fingerprint,
            ordered_feature_ids_fingerprint=identity.ids_fingerprint,
            ordered_feature_names_fingerprint=identity.names_fingerprint,
            normalization=resolved_norm_params,
            normalization_method=validated_parameters["normalization_method"],
            size_factor=validated_parameters["size_factor"],
            min_exp=min_exp,
            window_size=validated_parameters["window_size"],
            chunk_size=validated_parameters["chunk_size"],
            smoothen=smoothen,
            z_scale=z_scale,
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            ann_params=resolved_ann_params,
            nan_cluster_value=nan_cluster_value,
            batch_size=batch_size,
            nthreads=self.nthreads,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(len(feature_indices), effective_bins),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "feature_indices",
                    shape=(len(feature_indices),),
                    dtype_kind="u",
                ),
                ArrayRequirement(
                    "valid_features",
                    shape=(len(feature_indices),),
                    dtype_kind="b",
                ),
                ArrayRequirement(
                    "feature_clusters",
                    shape=(len(feature_indices),),
                    dtype_kind="i",
                ),
                ArrayRequirement(
                    "cluster_values",
                    shape=(assay.feats.N,),
                    dtype_kind="i",
                ),
                ArrayRequirement("feature_names", shape=(assay.feats.N,)),
                ArrayRequirement("feature_ids", shape=(assay.feats.N,)),
            ),
            required_attributes=(
                AttributeRequirement(
                    "input_fingerprints",
                    expected_types=(list, tuple),
                ),
                AttributeRequirement(
                    "nan_cluster_value",
                    expected_types=(int,),
                ),
                AttributeRequirement("effective_window", expected_types=(int,)),
                AttributeRequirement("effective_bins", expected_types=(int,)),
                AttributeRequirement(
                    "payload_fingerprint",
                    expected_types=(str,),
                ),
            ),
            reuse_validator=lambda _ref, group: _aggregation_payload_is_valid(
                group,
                n_features=assay.feats.N,
                selected_features=feature_indices,
                n_bins=effective_bins,
                n_clusters=n_clusters,
                n_neighbours=n_neighbours,
                nan_cluster_value=nan_cluster_value,
                ann_params=resolved_ann_params,
                expected_input_fingerprints=input_fingerprints,
                expected_feature_ids_fingerprint=identity.ids_fingerprint,
                expected_feature_names_fingerprint=identity.names_fingerprint,
                effective_window=effective_window,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_pseudotime_aggregation")
        logger.info("Pseudotime modules: aggregating feature profiles")
        check_normalization()
        full_data, stored_feature_indices, valid_features = (
            assay._aggregate_ordering_profiles(
                cell_idx=cell_indices,
                cell_ordering=cell_ordering,
                feat_idx=feature_indices,
                min_exp=min_exp,
                effective_window=effective_window,
                effective_bins=effective_bins,
                smoothen=smoothen,
                z_scale=z_scale,
                batch_size=batch_size,
                norm_params=resolved_norm_params,
            )
        )
        valid_count = int(np.count_nonzero(valid_features))
        if valid_count < 2 or n_neighbours >= valid_count or n_clusters > valid_count:
            raise ValueError(
                f"Only {valid_count} of {len(stored_feature_indices)} selected "
                f"features have mean expression above min_exp={min_exp} and a "
                "non-constant profile. Clustering needs at least two such "
                f"features, more than n_neighbours={n_neighbours}, and at least "
                f"n_clusters={n_clusters}. Lower min_exp or reduce n_neighbours "
                "or n_clusters."
            )
        valid_feature_indices = stored_feature_indices[valid_features]
        clusts = knn_clustering(
            d_array=ChunkedArray.from_numpy(
                full_data[valid_features],
                nthreads=self.nthreads,
                resources=self.resources,
            ),
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            nthreads=self.nthreads,
            ann_params=resolved_ann_params,
        )
        check_normalization()
        with artifact_transaction(self.zw, planned) as aggregation_group:
            assay._write_aggregated_ordering_group(
                aggregation_group,
                data=full_data,
                feature_indices=stored_feature_indices,
                valid=valid_features,
            )
            stored_feature_clusters = np.full(
                len(stored_feature_indices),
                nan_cluster_value,
                dtype=np.int64,
            )
            stored_feature_clusters[valid_features] = clusts
            _write_feature_vector(
                aggregation_group,
                "feature_clusters",
                stored_feature_clusters,
            )
            _write_feature_vector(
                aggregation_group,
                "cluster_values",
                _scatter_feature_clusters_impl(
                    assay.feats.N,
                    valid_feature_indices,
                    clusts,
                    nan_cluster_value,
                ),
            )
            identity.store(aggregation_group, "pseudotime aggregation")
            aggregation_group.attrs["input_fingerprints"] = input_fingerprints
            aggregation_group.attrs["nan_cluster_value"] = nan_cluster_value
            aggregation_group.attrs["effective_window"] = effective_window
            aggregation_group.attrs["effective_bins"] = effective_bins
            aggregation_group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                aggregation_group,
                _AGGREGATION_PAYLOAD,
            )
            check_normalization()
        logger.info(f"Stored {np.unique(clusts).size} pseudotime modules")
        return planned.ref

    @validation_scoped
    def _trajectory_load_aggregation(
        self,
        ref: ArtifactRef,
    ) -> PseudotimeAggregationResult:
        """Load a lazy pseudotime aggregation from an explicit artifact."""
        record = self._load_pseudotime_feature_record(
            ref,
            kind="pseudotime_aggregation",
            operation="run_pseudotime_aggregation",
            label="Pseudotime-aggregation",
            input_names=_AGGREGATION_INPUTS,
            validate_parameters=_validate_aggregation_parameters,
        )
        parameters = record.parameters
        ptime_result = record.pseudotime_result
        selected_cell_indices = read_stored_selection_indices(
            self.zw,
            record.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cell_indices = np.asarray(
            selected_cell_indices[ptime_result.valid],
            dtype=np.int64,
        )
        cell_ordering = np.asarray(
            ptime_result.values[ptime_result.valid],
            dtype=np.float64,
        )
        effective_window = min(parameters["window_size"], len(cell_indices))
        effective_bins = min(parameters["chunk_size"], len(cell_indices))
        try:
            ann_params = _validate_resolved_ann_parameters(
                parameters["ann_params"],
                dim=effective_bins,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Pseudotime-aggregation ANN parameters are malformed"
            ) from exc
        expected_input_fingerprints = [
            array_digest(values)
            for values in (cell_indices, record.feature_indices, cell_ordering)
        ]
        group = record.group
        if not _aggregation_payload_is_valid(
            group,
            n_features=record.assay.feats.N,
            selected_features=record.feature_indices,
            n_bins=effective_bins,
            n_clusters=parameters["n_clusters"],
            n_neighbours=parameters["n_neighbours"],
            nan_cluster_value=parameters["nan_cluster_value"],
            ann_params=ann_params,
            expected_input_fingerprints=expected_input_fingerprints,
            expected_feature_ids_fingerprint=record.ids_fingerprint,
            expected_feature_names_fingerprint=record.names_fingerprint,
            effective_window=effective_window,
        ):
            raise ValueError("Pseudotime-aggregation artifact payload is invalid")
        full_data = ChunkedArray(
            as_zarr_array(group["data"], name="data"),
            nthreads=self.nthreads,
            resources=self.resources,
        )
        valid_positions = _true_array_indices(group["valid_features"])
        feature_indices = read_array_rows_chunkwise(
            as_zarr_array(group["feature_indices"], name="feature_indices"),
            valid_positions,
        )
        feature_clusters = read_array_rows_chunkwise(
            as_zarr_array(group["feature_clusters"], name="feature_clusters"),
            valid_positions,
        ).astype(np.int64)
        frozen_feature_names = read_array_rows_chunkwise(
            as_zarr_array(group["feature_names"], name="feature_names"),
            feature_indices,
        ).astype(str)
        frozen_feature_ids = read_array_rows_chunkwise(
            as_zarr_array(group["feature_ids"], name="feature_ids"),
            feature_indices,
        ).astype(str)
        result = PseudotimeAggregationResult(
            ref=ref,
            data=full_data[valid_positions],
            feature_indices=feature_indices,
            feature_clusters=feature_clusters,
            assay=record.assay.name,
            cell_selection=record.cell_selection,
            feature_selection=record.feature_selection,
            pseudotime=record.pseudotime,
        )
        result._attach_feature_identity(
            frozen_feature_names,
            frozen_feature_ids,
        )
        return result

    def _lineage_feature_assay(
        self,
        fate: ArtifactRef,
        features: ArtifactRef,
    ) -> Assay:
        if not isinstance(fate, ArtifactRef):
            raise TypeError("fate must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.scope != "assay" or features.assay is None:
            raise ValueError("features must be an assay-scoped feature selection")
        return self._get_assay(features.assay)

    def _lineage_driver_mask(
        self,
        axes: _LineageCells,
        driver_cells: ArtifactRef,
        include_sinks: bool,
    ) -> np.ndarray:
        """Return which scored cells a driver search correlates."""
        if driver_cells == axes.fate.cell_selection:
            in_subset = np.ones(len(axes.cell_indices), dtype=bool)
        else:
            subset = read_stored_selection_indices(
                self.zw,
                driver_cells,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            if not np.isin(subset, axes.selected_cell_indices).all():
                raise ValueError(
                    "cells must be a subset of the fate map's cell selection"
                )
            in_subset = np.isin(axes.cell_indices, subset)
        return in_subset & (include_sinks | ~axes.absorbing)

    def _lineage_feature_statistics(
        self,
        assay: Assay,
        cell_indices: np.ndarray,
        feature_indices: np.ndarray,
        weights: np.ndarray,
        regressors: np.ndarray,
        *,
        min_mass: float,
        batch_size: int | None,
        norm_params: dict[str, Any],
        message: str,
        held_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stream normalized features and correlate them per lineage.

        ``held_bytes`` counts what the caller keeps alive besides ``weights``
        and ``regressors``, such as the fate probabilities.
        """
        # What the caller holds, the weights, the regressors, their centred
        # copy, which the stream makes once before the first block, the
        # feature names, and four result arrays per feature.
        feature_names = _feature_names_at(assay, feature_indices)
        resident_bytes = (
            held_bytes
            + weights.nbytes
            + regressors.nbytes
            + regressors.shape[0] * regressors.shape[1] * 8
            + feature_names.nbytes
            + 4 * len(feature_indices) * weights.shape[1] * 8
        )

        def blocks() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for item in assay.iter_normed_feature_wise(
                cell_indices,
                feature_indices,
                batch_size,
                message,
                False,
                scratch_itemsize=LINEAGE_SCRATCH_ITEMSIZE,
                resident_bytes=resident_bytes,
                **norm_params,
            ):
                feature_major, labels = cast(tuple[np.ndarray, np.ndarray], item)
                yield np.asarray(feature_major, dtype=np.float64).T, labels

        return _stream_weighted_correlations_impl(
            blocks(),
            feature_indices,
            weights,
            regressors,
            min_mass=min_mass,
            nthreads=int(getattr(assay, "nthreads", self.nthreads)),
            feature_names=feature_names,
        )

    @validation_scoped
    def _trajectory_lineage_markers(
        self,
        fate: ArtifactRef,
        *,
        features: ArtifactRef,
        lineages: list[Any] | None = None,
        min_cells: float = 10.0,
        gene_batch_size: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Store each lineage's fate-weighted feature associations with pseudotime.

        :meth:`markers` correlates features with one pseudotime over every
        scored cell, so a gene that rises on one branch is diluted by the
        others. Here each lineage weighs every scored cell by its fate
        probability, and a feature's weighted Pearson correlation with
        pseudotime is computed per lineage. P-values come from a Student t
        test on Kish's effective number of cells, ``(sum w)^2 / sum w^2``, and
        are adjusted by Benjamini-Hochberg within each lineage over its
        tested features. They treat cells as independent and are not
        inference across biological replicates.

        Args:
            fate: Explicit fate-map artifact. Its pseudotime and probabilities
                define the cells, their order, and their weights.
            features: Explicit assay-scoped feature-selection artifact.
            lineages: Sink labels to analyse, in order. ``None`` analyses
                every sink.
            min_cells: Smallest fate weight, in cells, that a feature's
                expressing cells must carry in a lineage for it to be tested.
            gene_batch_size: Number of features loaded per batch. ``None``
                sizes batches to the memory budget.
            invalidate_cache: Compute a new result even when one matches.
            **norm_params: Extra keyword arguments forwarded to normalized
                expression.

        Returns:
            Reference to the immutable ``lineage_markers`` artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a scored cell.
        """
        reject_unknown_normalization_params(
            norm_params,
            caller="run_lineage_marker_search",
        )
        gene_batch_size = (
            None
            if gene_batch_size is None
            else integer_argument(gene_batch_size, "gene_batch_size", minimum=1)
        )
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        if lineages is not None and not isinstance(lineages, list):
            raise TypeError("lineages must be a list of sink labels or None")
        assay = self._lineage_feature_assay(fate, features)
        identity = _FrozenFeatureIdentity.capture(assay)
        feature_selection, feature_indices = _resolve_feature_indices(
            self, assay, features
        )
        dataset_fingerprint = self._ensure_dataset_fingerprint(assay.name)
        axes = self._lineage_axes(self._load_fate_inputs(fate), lineages)
        validated = _validate_lineage_marker_parameters(
            {
                "normalization": _default_normalization(
                    assay, norm_params, "run_lineage_marker_search"
                ),
                "normalization_method": callable_identity(assay.normMethod),
                "size_factor": getattr(assay, "sf", None),
                "lineages": list(axes.lineages),
                "min_cells": min_cells,
                **_LINEAGE_MARKER_METHODS,
            }
        )
        check_normalization = _normalization_guard(
            assay, validated, "Lineage marker search"
        )
        self._require_measured_cells(
            assay.name,
            axes.cell_indices,
            operation="run_lineage_marker_search",
            remedy="graph",
        )
        n_lineages = len(axes.lineages)
        n_features = assay.feats.N
        arguments = LineageMarkerArguments(
            cell_selection=axes.fate.cell_selection,
            feature_selection=feature_selection,
            fate=fate,
            dataset_fingerprint=dataset_fingerprint,
            ordered_feature_ids_fingerprint=identity.ids_fingerprint,
            ordered_feature_names_fingerprint=identity.names_fingerprint,
            normalization=validated["normalization"],
            normalization_method=validated["normalization_method"],
            size_factor=validated["size_factor"],
            lineages=axes.lineages,
            association_method=validated["association_method"],
            p_value_method=validated["p_value_method"],
            adjustment_method=validated["adjustment_method"],
            adjustment_scope=validated["adjustment_scope"],
            min_cells=validated["min_cells"],
            gene_batch_size=gene_batch_size,
            nthreads=int(getattr(assay, "nthreads", self.nthreads)),
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                *(
                    ArrayRequirement(
                        name, shape=(n_features, n_lineages), dtype_kind="f"
                    )
                    for name in ("r_value", "p_value", "p_value_adjusted")
                ),
                ArrayRequirement(
                    "effective_cells", shape=(n_lineages,), dtype_kind="f"
                ),
                ArrayRequirement("feature_names", shape=(n_features,)),
                ArrayRequirement("feature_ids", shape=(n_features,)),
            ),
            required_attributes=(
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
            ),
            reuse_validator=lambda _ref, group: _lineage_marker_payload_is_valid(
                group,
                n_features=n_features,
                n_lineages=n_lineages,
                selected_features=feature_indices,
                expected_feature_ids_fingerprint=identity.ids_fingerprint,
                expected_feature_names_fingerprint=identity.names_fingerprint,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_lineage_marker_search")
        check_normalization()
        logger.info(
            f"Lineage markers: correlating {len(feature_indices)} features over "
            f"{len(axes.cell_indices)} cells and {n_lineages} lineages"
        )
        r_values, status = self._lineage_feature_statistics(
            assay,
            axes.cell_indices,
            feature_indices,
            axes.weights,
            np.repeat(axes.ordering[:, np.newaxis], n_lineages, axis=1),
            min_mass=validated["min_cells"],
            batch_size=gene_batch_size,
            norm_params=validated["normalization"],
            message="Finding lineage markers",
            # The weights are the axes' own; the repeated ordering is new.
            held_bytes=axes.nbytes - axes.weights.nbytes,
        )
        effective = _kish_effective_sizes_impl(axes.weights)
        p_values, status = _kish_p_values_impl(r_values, status, effective)
        adjusted = _adjust_lineage_p_values_impl(p_values, status)
        r_values = np.where(status == LINEAGE_OK, r_values, np.nan)
        with artifact_transaction(self.zw, planned) as group:
            check_normalization()
            for name, values in (
                ("r_value", r_values),
                ("p_value", p_values),
                ("p_value_adjusted", adjusted),
            ):
                _write_feature_matrix(group, name, feature_indices, values, n_features)
            _write_feature_vector(group, "effective_cells", effective)
            identity.store(group, "lineage marker search")
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                _LINEAGE_MARKER_PAYLOAD,
            )
            check_normalization()
        logger.info(f"Stored lineage markers for {n_lineages} lineages")
        return planned.ref

    def _load_lineage_feature_record(
        self,
        ref: ArtifactRef,
        *,
        kind: str,
        operation: str,
        label: str,
        input_names: frozenset[str],
        validate_parameters: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> _LineageFeatureRecord:
        head = self._feature_record_head(
            ref,
            kind=kind,
            operation=operation,
            label=label,
            input_names=input_names,
            validate_parameters=validate_parameters,
            upstream="fate",
        )
        fate_inputs = self._load_fate_inputs(head.upstream)
        if fate_inputs.fate.cell_selection != head.cell_selection:
            raise ValueError(f"{label} cell selection does not match its fate map")
        lineages = list(head.parameters["lineages"])
        try:
            _select_lineages_impl(fate_inputs.fate.sink_labels, lineages)
        except ValueError as exc:
            raise ValueError(f"{label} lineages are not sinks of its fate map") from exc
        return _LineageFeatureRecord(
            group=as_zarr_group(self.zw[head.path], name=head.path),
            inputs=head.inputs,
            parameters=head.parameters,
            assay=head.assay,
            cell_selection=head.cell_selection,
            feature_selection=head.feature_selection,
            fate=head.upstream,
            feature_indices=head.feature_indices,
            ids_fingerprint=head.ids_fingerprint,
            names_fingerprint=head.names_fingerprint,
            axes=self._lineage_axes(fate_inputs, lineages),
        )

    def _lineage_feature_table(
        self,
        record: _LineageFeatureRecord,
        names: tuple[str, ...],
    ) -> pd.DataFrame:
        """Return the long table of per-feature, per-lineage statistics."""
        group = record.group
        indices = record.feature_indices
        identity = {
            name: read_array_rows_chunkwise(
                as_zarr_array(group[name], name=name), indices
            ).astype(str)
            for name in ("feature_names", "feature_ids")
        }
        statistics = {
            name: read_array_rows_chunkwise(
                as_zarr_array(group[name], name=name), indices
            )
            for name in names
        }
        frames = []
        for column, lineage in enumerate(record.axes.lineages):
            frame = pd.DataFrame(
                {
                    "feature_index": indices,
                    "feature_name": identity["feature_names"],
                    "feature_id": identity["feature_ids"],
                    "lineage": [lineage] * len(indices),
                }
            )
            for name in names:
                frame[name] = np.asarray(statistics[name][:, column], dtype=np.float64)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @validation_scoped
    def _trajectory_load_lineage_markers(
        self,
        ref: ArtifactRef,
    ) -> LineageMarkerResult:
        """Load a lineage-marker table from an explicit artifact."""
        record = self._load_lineage_feature_record(
            ref,
            kind="lineage_markers",
            operation="run_lineage_marker_search",
            label="Lineage-marker",
            input_names=_LINEAGE_MARKER_INPUTS,
            validate_parameters=_validate_lineage_marker_parameters,
        )
        if not _lineage_marker_payload_is_valid(
            record.group,
            n_features=record.assay.feats.N,
            n_lineages=len(record.axes.lineages),
            selected_features=record.feature_indices,
            expected_feature_ids_fingerprint=record.ids_fingerprint,
            expected_feature_names_fingerprint=record.names_fingerprint,
        ):
            raise ValueError("Lineage-marker artifact payload is invalid")
        return LineageMarkerResult(
            ref=ref,
            table=self._lineage_feature_table(
                record, ("r_value", "p_value", "p_value_adjusted")
            ),
            assay=record.assay.name,
            lineages=record.axes.lineages,
            effective_cells=np.asarray(
                _stored_values(record.group, "effective_cells"), dtype=np.float64
            ),
            cell_selection=record.cell_selection,
            feature_selection=record.feature_selection,
            fate=record.fate,
        )

    @validation_scoped
    def _trajectory_drivers(
        self,
        fate: ArtifactRef,
        *,
        features: ArtifactRef,
        lineages: list[Any] | None = None,
        cells: ArtifactRef | None = None,
        include_sinks: bool = False,
        min_cells: int = 10,
        confidence_level: float = 0.95,
        gene_batch_size: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Store each feature's correlation with every lineage's fate probability.

        As CellRank's lineage drivers (Lange et al. 2022), a feature's Pearson
        correlation with a lineage's fate probability ranks genes whose
        expression rises with the commitment to that lineage. P-values and
        confidence intervals use Fisher's z-transform, and p-values are
        adjusted by Benjamini-Hochberg within each lineage over its tested
        features.

        Cells of the sinks are left out by default: their probabilities are
        fixed at zero or one by the model, so including them mostly recovers
        the markers that defined the sinks. Restrict ``cells`` to the cells
        around a branch to ask which genes separate its fates there.

        Args:
            fate: Explicit fate-map artifact.
            features: Explicit assay-scoped feature-selection artifact.
            lineages: Sink labels to analyse, in order. ``None`` analyses
                every sink.
            cells: Cell-selection artifact within the fate map's cells to
                correlate. ``None`` uses every scored cell.
            include_sinks: Also correlate the cells of the sinks.
            min_cells: Fewest expressing cells for a feature to be tested.
            confidence_level: Coverage of the confidence intervals.
            gene_batch_size: Number of features loaded per batch. ``None``
                sizes batches to the memory budget.
            invalidate_cache: Compute a new result even when one matches.
            **norm_params: Extra keyword arguments forwarded to normalized
                expression.

        Returns:
            Reference to the immutable ``lineage_drivers`` artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a correlated cell.
        """
        reject_unknown_normalization_params(
            norm_params,
            caller="run_lineage_driver_search",
        )
        gene_batch_size = (
            None
            if gene_batch_size is None
            else integer_argument(gene_batch_size, "gene_batch_size", minimum=1)
        )
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        if lineages is not None and not isinstance(lineages, list):
            raise TypeError("lineages must be a list of sink labels or None")
        if cells is not None and not isinstance(cells, ArtifactRef):
            raise TypeError("cells must be a cell-selection ArtifactRef or None")
        assay = self._lineage_feature_assay(fate, features)
        identity = _FrozenFeatureIdentity.capture(assay)
        feature_selection, feature_indices = _resolve_feature_indices(
            self, assay, features
        )
        dataset_fingerprint = self._ensure_dataset_fingerprint(assay.name)
        axes = self._lineage_axes(self._load_fate_inputs(fate), lineages)
        validated = _validate_lineage_driver_parameters(
            {
                "normalization": _default_normalization(
                    assay, norm_params, "run_lineage_driver_search"
                ),
                "normalization_method": callable_identity(assay.normMethod),
                "size_factor": getattr(assay, "sf", None),
                "lineages": list(axes.lineages),
                "include_sinks": include_sinks,
                "min_cells": min_cells,
                "confidence_level": confidence_level,
                **_LINEAGE_DRIVER_METHODS,
            }
        )
        driver_cells = axes.fate.cell_selection if cells is None else cells
        mask = self._lineage_driver_mask(axes, driver_cells, validated["include_sinks"])
        n_cells = int(mask.sum())
        if n_cells < 4:
            raise ValueError(
                f"Lineage drivers need at least four correlated cells, got {n_cells}"
            )
        check_normalization = _normalization_guard(
            assay, validated, "Lineage driver search"
        )
        driver_indices = axes.cell_indices[mask]
        self._require_measured_cells(
            assay.name,
            driver_indices,
            operation="run_lineage_driver_search",
            remedy="graph",
        )
        n_lineages = len(axes.lineages)
        n_features = assay.feats.N
        arguments = LineageDriverArguments(
            cell_selection=axes.fate.cell_selection,
            driver_cells=driver_cells,
            feature_selection=feature_selection,
            fate=fate,
            dataset_fingerprint=dataset_fingerprint,
            ordered_feature_ids_fingerprint=identity.ids_fingerprint,
            ordered_feature_names_fingerprint=identity.names_fingerprint,
            normalization=validated["normalization"],
            normalization_method=validated["normalization_method"],
            size_factor=validated["size_factor"],
            lineages=axes.lineages,
            include_sinks=validated["include_sinks"],
            min_cells=validated["min_cells"],
            confidence_level=validated["confidence_level"],
            association_method=validated["association_method"],
            p_value_method=validated["p_value_method"],
            adjustment_method=validated["adjustment_method"],
            adjustment_scope=validated["adjustment_scope"],
            gene_batch_size=gene_batch_size,
            nthreads=int(getattr(assay, "nthreads", self.nthreads)),
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                *(
                    ArrayRequirement(
                        name, shape=(n_features, n_lineages), dtype_kind="f"
                    )
                    for name in _LINEAGE_DRIVER_PAYLOAD[:5]
                ),
                ArrayRequirement("feature_names", shape=(n_features,)),
                ArrayRequirement("feature_ids", shape=(n_features,)),
            ),
            required_attributes=(
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
                AttributeRequirement("n_cells", expected_types=(int,)),
            ),
            reuse_validator=lambda _ref, group: _lineage_driver_payload_is_valid(
                group,
                n_features=n_features,
                n_lineages=n_lineages,
                n_cells=n_cells,
                selected_features=feature_indices,
                expected_feature_ids_fingerprint=identity.ids_fingerprint,
                expected_feature_names_fingerprint=identity.names_fingerprint,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_lineage_driver_search")
        check_normalization()
        logger.info(
            f"Lineage drivers: correlating {len(feature_indices)} features over "
            f"{n_cells} cells and {n_lineages} lineages"
        )
        probabilities = axes.weights[mask]
        r_values, status = self._lineage_feature_statistics(
            assay,
            driver_indices,
            feature_indices,
            np.ones_like(probabilities),
            probabilities,
            min_mass=float(validated["min_cells"]),
            batch_size=gene_batch_size,
            norm_params=validated["normalization"],
            message="Finding lineage drivers",
            held_bytes=axes.nbytes,
        )
        p_values, low, high, status = _fisher_z_statistics_impl(
            r_values, status, n_cells, validated["confidence_level"]
        )
        adjusted = _adjust_lineage_p_values_impl(p_values, status)
        r_values = np.where(status == LINEAGE_OK, r_values, np.nan)
        with artifact_transaction(self.zw, planned) as group:
            check_normalization()
            for name, values in (
                ("r_value", r_values),
                ("p_value", p_values),
                ("p_value_adjusted", adjusted),
                ("ci_low", low),
                ("ci_high", high),
            ):
                _write_feature_matrix(group, name, feature_indices, values, n_features)
            identity.store(group, "lineage driver search")
            group.attrs["n_cells"] = n_cells
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                _LINEAGE_DRIVER_PAYLOAD,
            )
            check_normalization()
        logger.info(f"Stored lineage drivers for {n_lineages} lineages")
        return planned.ref

    @validation_scoped
    def _trajectory_load_drivers(
        self,
        ref: ArtifactRef,
    ) -> LineageDriverResult:
        """Load a lineage-driver table from an explicit artifact."""
        record = self._load_lineage_feature_record(
            ref,
            kind="lineage_drivers",
            operation="run_lineage_driver_search",
            label="Lineage-driver",
            input_names=_LINEAGE_DRIVER_INPUTS,
            validate_parameters=_validate_lineage_driver_parameters,
        )
        driver_cells = parse_artifact_ref(
            record.inputs.get("driver_cells"), "Lineage-driver cells input"
        )
        n_cells = int(
            self._lineage_driver_mask(
                record.axes, driver_cells, record.parameters["include_sinks"]
            ).sum()
        )
        if not _lineage_driver_payload_is_valid(
            record.group,
            n_features=record.assay.feats.N,
            n_lineages=len(record.axes.lineages),
            n_cells=n_cells,
            selected_features=record.feature_indices,
            expected_feature_ids_fingerprint=record.ids_fingerprint,
            expected_feature_names_fingerprint=record.names_fingerprint,
        ):
            raise ValueError("Lineage-driver artifact payload is invalid")
        return LineageDriverResult(
            ref=ref,
            table=self._lineage_feature_table(record, _LINEAGE_DRIVER_PAYLOAD[:5]),
            assay=record.assay.name,
            lineages=record.axes.lineages,
            n_cells=n_cells,
            cell_selection=record.cell_selection,
            driver_cells=driver_cells,
            feature_selection=record.feature_selection,
            fate=record.fate,
        )

    @validation_scoped
    def _trajectory_lineage_aggregation(
        self,
        fate: ArtifactRef,
        *,
        features: ArtifactRef,
        lineages: list[Any] | None = None,
        n_bins: int = 50,
        smoothing: float = 2.0,
        min_exp: float = 1e-3,
        z_scale: bool = True,
        n_neighbours: int = 11,
        n_clusters: int = 10,
        batch_size: int | None = None,
        ann_params: dict | None = None,
        nan_cluster_value: int = -1,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Store fate-weighted feature profiles along each lineage, with modules.

        Along pseudotime, each lineage's fate weight is cut into ``n_bins``
        bins of equal weight, a cell whose weight straddles a cut being
        shared in proportion. A feature's profile is its weighted mean in
        every bin, smoothed across neighbouring bins by a Gaussian of
        ``smoothing`` bins, so every lineage runs from the shared start to its
        sink. With ``z_scale``, each feature is scaled by one mean and
        standard deviation over all lineages, which keeps its lineages
        comparable. Features with similar profiles across all lineages are
        grouped into ``n_clusters`` modules, as :meth:`aggregation` groups
        them along one pseudotime.

        Args:
            fate: Explicit fate-map artifact.
            features: Explicit assay-scoped feature-selection artifact.
            lineages: Sink labels to profile, in order. ``None`` profiles
                every sink.
            n_bins: Bins per lineage. Each lineage needs a total fate weight
                of at least one cell per bin.
            smoothing: Standard deviation, in bins, of the Gaussian that
                smooths each profile; 0 leaves the bin means as they are.
                A lineage with little fate weight holds few cells per bin, so
                its profiles need this smoothing most.
            min_exp: Minimum mean normalized expression required for clustering.
            z_scale: Whether to standardize each retained feature.
            n_neighbours: Number of neighbors in the feature graph.
            n_clusters: Number of feature modules to create.
            batch_size: Number of features processed per batch. ``None``
                sizes batches to the memory budget.
            ann_params: Parameters forwarded to the HNSW index.
            nan_cluster_value: Value assigned to features excluded from clustering.
            invalidate_cache: Compute a new result even when one matches.
            **norm_params: Extra keyword arguments forwarded to normalized
                expression.

        Returns:
            Reference to the immutable ``lineage_aggregation`` artifact.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a scored cell.
        """
        from ...trajectory.feature_dynamics import knn_clustering

        reject_unknown_normalization_params(
            norm_params,
            caller="run_lineage_aggregation",
        )
        batch_size = (
            None
            if batch_size is None
            else integer_argument(batch_size, "batch_size", minimum=1)
        )
        invalidate_cache = _validate_invalidate_cache(invalidate_cache)
        if lineages is not None and not isinstance(lineages, list):
            raise TypeError("lineages must be a list of sink labels or None")
        assay = self._lineage_feature_assay(fate, features)
        identity = _FrozenFeatureIdentity.capture(assay)
        feature_selection, feature_indices = _resolve_feature_indices(
            self, assay, features
        )
        dataset_fingerprint = self._ensure_dataset_fingerprint(assay.name)
        axes = self._lineage_axes(self._load_fate_inputs(fate), lineages)
        validated = _validate_lineage_aggregation_parameters(
            {
                "normalization": _default_normalization(
                    assay, norm_params, "run_lineage_aggregation"
                ),
                "normalization_method": callable_identity(assay.normMethod),
                "size_factor": getattr(assay, "sf", None),
                "lineages": list(axes.lineages),
                "n_bins": n_bins,
                "smoothing": smoothing,
                "min_exp": min_exp,
                "z_scale": z_scale,
                "n_neighbours": n_neighbours,
                "n_clusters": n_clusters,
                "ann_params": {} if ann_params is None else ann_params,
                "nan_cluster_value": nan_cluster_value,
            }
        )
        n_bins = validated["n_bins"]
        n_neighbours = validated["n_neighbours"]
        n_clusters = validated["n_clusters"]
        nan_cluster_value = validated["nan_cluster_value"]
        if len(feature_indices) < 2:
            raise ValueError("At least two selected features are required")
        if n_neighbours >= len(feature_indices):
            raise ValueError(
                "n_neighbours must be smaller than the selected feature count"
            )
        if n_clusters > len(feature_indices):
            raise ValueError("n_clusters cannot exceed the selected feature count")
        check_normalization = _normalization_guard(
            assay, validated, "Lineage aggregation"
        )
        self._require_measured_cells(
            assay.name,
            axes.cell_indices,
            operation="run_lineage_aggregation",
            remedy="graph",
        )
        n_lineages = len(axes.lineages)
        lineage_totals = axes.weights.sum(axis=0)
        light = np.flatnonzero(lineage_totals < n_bins)
        if light.size:
            raise ValueError(
                f"Lineage {axes.lineages[int(light[0])]!r} carries a total fate "
                f"weight of {float(lineage_totals[light[0]]):.3g}, below "
                f"n_bins={n_bins}; request fewer bins"
            )
        smoothing_matrix = _bin_smoothing_matrix_impl(n_bins, validated["smoothing"])
        dims = n_lineages * n_bins
        resolved_ann_params = resolve_aggregation_ann_params(
            validated["ann_params"], dim=dims
        )
        input_fingerprints = [
            array_digest(values)
            for values in (
                axes.cell_indices,
                feature_indices,
                axes.ordering,
                axes.weights,
            )
        ]
        n_features = assay.feats.N
        n_selected = len(feature_indices)
        arguments = LineageAggregationArguments(
            cell_selection=axes.fate.cell_selection,
            feature_selection=feature_selection,
            fate=fate,
            dataset_fingerprint=dataset_fingerprint,
            ordered_feature_ids_fingerprint=identity.ids_fingerprint,
            ordered_feature_names_fingerprint=identity.names_fingerprint,
            normalization=validated["normalization"],
            normalization_method=validated["normalization_method"],
            size_factor=validated["size_factor"],
            lineages=axes.lineages,
            n_bins=n_bins,
            smoothing=validated["smoothing"],
            min_exp=validated["min_exp"],
            z_scale=validated["z_scale"],
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            ann_params=resolved_ann_params,
            nan_cluster_value=nan_cluster_value,
            batch_size=batch_size,
            nthreads=self.nthreads,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "data", shape=(n_selected, n_lineages, n_bins), dtype_kind="f"
                ),
                ArrayRequirement(
                    "feature_indices", shape=(n_selected,), dtype_kind="u"
                ),
                ArrayRequirement("valid_features", shape=(n_selected,), dtype_kind="b"),
                ArrayRequirement(
                    "feature_clusters", shape=(n_selected,), dtype_kind="i"
                ),
                ArrayRequirement("cluster_values", shape=(n_features,), dtype_kind="i"),
                ArrayRequirement(
                    "bin_pseudotime", shape=(n_lineages, n_bins), dtype_kind="f"
                ),
                ArrayRequirement("lineage_weight", shape=(n_lineages,), dtype_kind="f"),
                ArrayRequirement("feature_names", shape=(n_features,)),
                ArrayRequirement("feature_ids", shape=(n_features,)),
            ),
            required_attributes=(
                AttributeRequirement(
                    "input_fingerprints", expected_types=(list, tuple)
                ),
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
            ),
            reuse_validator=lambda _ref, group: _lineage_aggregation_payload_is_valid(
                group,
                n_features=n_features,
                selected_features=feature_indices,
                n_lineages=n_lineages,
                n_bins=n_bins,
                n_clusters=n_clusters,
                n_neighbours=n_neighbours,
                nan_cluster_value=nan_cluster_value,
                expected_input_fingerprints=input_fingerprints,
                expected_feature_ids_fingerprint=identity.ids_fingerprint,
                expected_feature_names_fingerprint=identity.names_fingerprint,
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_lineage_aggregation")
        check_normalization()
        logger.info(
            f"Lineage profiles: binning {n_selected} features into {n_bins} bins "
            f"over {n_lineages} lineages"
        )
        # Building the operator holds it alongside the cells' fate weights.
        operator_bytes = axes.nbytes + _bin_operator_bytes_impl(
            len(axes.cell_indices), n_lineages, n_bins
        )
        if operator_bytes >= self.memoryBytes:
            raise MemoryError(
                f"Binning {len(axes.cell_indices)} cells along {n_lineages} "
                f"lineages needs about {operator_bytes} bytes, which exceeds the "
                f"memory budget of {self.memoryBytes} bytes. Increase the memory "
                "budget or profile fewer lineages."
            )
        operator, bin_ordering, lineage_weight = _equal_mass_bin_operator_impl(
            axes.ordering, axes.weights, n_bins
        )
        feature_names = _feature_names_at(assay, feature_indices)
        # The cells' fate weights, the operator, the feature names, and three
        # copies of the profiles: streamed, finalized, and clustered.
        profile_bytes = n_selected * dims * 8
        resident_bytes = (
            axes.nbytes
            + int(
                operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes
            )
            + feature_names.nbytes
            + 3 * profile_bytes
        )

        def blocks() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for item in assay.iter_normed_feature_wise(
                axes.cell_indices,
                feature_indices,
                batch_size,
                "Binning features along lineages",
                False,
                scratch_itemsize=LINEAGE_SCRATCH_ITEMSIZE,
                resident_bytes=resident_bytes,
                **validated["normalization"],
            ):
                feature_major, labels = cast(tuple[np.ndarray, np.ndarray], item)
                yield np.asarray(feature_major, dtype=np.float64).T, labels

        profiles, means = _stream_lineage_profiles_impl(
            blocks(),
            feature_indices,
            operator,
            smoothing_matrix,
            n_lineages,
            nthreads=self.nthreads,
            feature_names=feature_names,
        )
        data, valid = _finalize_lineage_profiles_impl(
            profiles,
            means,
            min_expression=validated["min_exp"],
            z_scale=validated["z_scale"],
        )
        valid_count = int(np.count_nonzero(valid))
        if valid_count < 2 or n_neighbours >= valid_count or n_clusters > valid_count:
            raise ValueError(
                f"Only {valid_count} of {n_selected} selected features have mean "
                f"expression above min_exp={validated['min_exp']} and a "
                "non-constant profile. Clustering needs at least two such "
                f"features, more than n_neighbours={n_neighbours}, and at least "
                f"n_clusters={n_clusters}. Lower min_exp or reduce n_neighbours "
                "or n_clusters."
            )
        clusters = knn_clustering(
            d_array=ChunkedArray.from_numpy(
                data[valid].reshape(valid_count, dims),
                nthreads=self.nthreads,
                resources=self.resources,
            ),
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            nthreads=self.nthreads,
            ann_params=resolved_ann_params,
        )
        check_normalization()
        feature_clusters = np.full(n_selected, nan_cluster_value, dtype=np.int64)
        feature_clusters[valid] = clusters
        with artifact_transaction(self.zw, planned) as group:
            check_normalization()
            data_array = create_zarr_dataset(
                group,
                "data",
                (
                    max(1, min(n_selected, max(1, 1_000_000 // max(dims, 1)))),
                    n_lineages,
                    n_bins,
                ),
                np.float64,
                (n_selected, n_lineages, n_bins),
            )
            data_array[...] = data
            _write_feature_vector(
                group, "feature_indices", feature_indices.astype(np.uint64)
            )
            _write_feature_vector(group, "valid_features", valid)
            _write_feature_vector(group, "feature_clusters", feature_clusters)
            _write_feature_vector(
                group,
                "cluster_values",
                _scatter_feature_clusters_impl(
                    n_features,
                    feature_indices[valid],
                    clusters,
                    nan_cluster_value,
                ).astype(np.int64),
            )
            for name, values in (
                ("bin_pseudotime", bin_ordering),
                ("lineage_weight", lineage_weight),
            ):
                output = create_zarr_dataset(
                    group, name, values.shape, np.float64, values.shape
                )
                output[...] = values
            identity.store(group, "lineage aggregation")
            group.attrs["input_fingerprints"] = input_fingerprints
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                _LINEAGE_AGGREGATION_PAYLOAD,
            )
            check_normalization()
        logger.info(f"Stored {np.unique(clusters).size} lineage modules")
        return planned.ref

    @validation_scoped
    def _trajectory_load_lineage_aggregation(
        self,
        ref: ArtifactRef,
    ) -> LineageAggregationResult:
        """Load fate-weighted lineage profiles from an explicit artifact."""
        record = self._load_lineage_feature_record(
            ref,
            kind="lineage_aggregation",
            operation="run_lineage_aggregation",
            label="Lineage-aggregation",
            input_names=_LINEAGE_AGGREGATION_INPUTS,
            validate_parameters=_validate_lineage_aggregation_parameters,
        )
        parameters = record.parameters
        axes = record.axes
        expected_input_fingerprints = [
            array_digest(values)
            for values in (
                axes.cell_indices,
                record.feature_indices,
                axes.ordering,
                axes.weights,
            )
        ]
        group = record.group
        n_lineages = len(axes.lineages)
        if not _lineage_aggregation_payload_is_valid(
            group,
            n_features=record.assay.feats.N,
            selected_features=record.feature_indices,
            n_lineages=n_lineages,
            n_bins=parameters["n_bins"],
            n_clusters=parameters["n_clusters"],
            n_neighbours=parameters["n_neighbours"],
            nan_cluster_value=parameters["nan_cluster_value"],
            expected_input_fingerprints=expected_input_fingerprints,
            expected_feature_ids_fingerprint=record.ids_fingerprint,
            expected_feature_names_fingerprint=record.names_fingerprint,
        ):
            raise ValueError("Lineage-aggregation artifact payload is invalid")
        valid_positions = _true_array_indices(group["valid_features"])
        feature_indices = read_array_rows_chunkwise(
            as_zarr_array(group["feature_indices"], name="feature_indices"),
            valid_positions,
        ).astype(np.int64)
        return LineageAggregationResult(
            ref=ref,
            data=read_array_rows_chunkwise(
                as_zarr_array(group["data"], name="data"), valid_positions
            ).astype(np.float64),
            feature_indices=feature_indices,
            feature_clusters=read_array_rows_chunkwise(
                as_zarr_array(group["feature_clusters"], name="feature_clusters"),
                valid_positions,
            ).astype(np.int64),
            feature_names=read_array_rows_chunkwise(
                as_zarr_array(group["feature_names"], name="feature_names"),
                feature_indices,
            ).astype(str),
            feature_ids=read_array_rows_chunkwise(
                as_zarr_array(group["feature_ids"], name="feature_ids"),
                feature_indices,
            ).astype(str),
            lineages=axes.lineages,
            bin_pseudotime=np.asarray(
                _stored_values(group, "bin_pseudotime"), dtype=np.float64
            ),
            lineage_weight=np.asarray(
                _stored_values(group, "lineage_weight"), dtype=np.float64
            ),
            assay=record.assay.name,
            cell_selection=record.cell_selection,
            feature_selection=record.feature_selection,
            fate=record.fate,
        )
