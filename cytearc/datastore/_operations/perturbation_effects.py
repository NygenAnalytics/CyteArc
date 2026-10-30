from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    PerturbationEDistanceArguments,
    PerturbationEfficiencyArguments,
    PerturbationLabelArguments,
)
from ...metadata.artifacts import write_cell_data_artifact
from ...metadata.rows import array_row_selection_parts
from ...metadata.selection import cell_selection_indices
from ...perturbation.edistance import (
    EDISTANCE_STATUSES,
    EDISTANCE_TESTED,
    PerturbationEDistance,
    edistance_arrays,
    edistance_memory_bytes,
    edistance_settings,
)
from ...perturbation.efficiency import (
    EFFICIENCY_NO_FEATURE,
    EFFICIENCY_STATUSES,
    EFFICIENCY_TESTED,
    PerturbationEfficiency,
    efficiency_arrays,
    efficiency_settings,
)
from ...storage.artifact_writer import ArrayRequirement, AttributeRequirement
from ...storage.artifacts import ArtifactRef, artifact_group, require_complete_artifact
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.warnings import warn
from .perturbation_store import (
    _COORDINATE_KINDS,
    _count_blocks,
    _label_groups,
    _label_rows,
    _matched_ids,
    _pairs_fingerprint,
    _positions_in,
    _require_budget,
    _require_ref,
    _row_features,
    _status_labels,
)

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _PerturbationEffectOperationsBase,
    )
else:
    _PerturbationEffectOperationsBase = object

# The floating-point columns of a perturbation efficiency table.
_EFFICIENCY_VALUES = (
    "mean_umi",
    "control_mean_umi",
    "mean_expression",
    "control_mean_expression",
    "remaining_fraction",
    "log2_fold_change",
    "p_value",
    "adjusted_p_value",
)


def _float64_rows(data: zarr.Array, rows: np.ndarray) -> np.ndarray:
    """Read ascending rows of a two-dimensional array as float64.

    The rows are read one chunk of rows at a time, so that a read holds the
    values of one chunk in their stored type, not those of every row.
    """
    values = np.empty((len(rows), int(data.shape[1])))
    step = int(data.chunks[0])
    # Where the rows of each chunk start among the rows, once per chunk that
    # holds any.
    edges = np.unique(
        np.searchsorted(rows, np.arange(0, int(data.shape[0]) + step, step))
    )
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        values[start:stop] = data.oindex[rows[start:stop], :]
    return values


def _unmatched_targets_warning(
    names: np.ndarray, status: np.ndarray, assay: str
) -> str | None:
    """Name the efficiency rows whose target matches no feature, or return None.

    A measured table warns with this message, and so does a saved table when
    ``efficiency`` reuses it.

    Args:
        names: Each row's name: its target, or its guide.
        status: Each row's status, as a position in ``EFFICIENCY_STATUSES``.
        assay: The expression assay.
    """
    unmatched = names[np.asarray(status) == EFFICIENCY_NO_FEATURE]
    if not len(unmatched):
        return None
    return (
        f"{len(unmatched)} of {len(names)} rows have no feature of assay "
        f"{assay!r} for their target, for example "
        + ", ".join(repr(str(name)) for name in unmatched[:5])
        + ". Check id_column, or the target names against the feature names."
    )


class _PerturbationEffectOperationsMixin(_PerturbationEffectOperationsBase):
    @validation_scoped
    def _perturbation_efficiency(
        self,
        labels: ArtifactRef,
        *,
        from_assay: str = "RNA",
        by: str = "target",
        id_column: str | None = "target_gene_id",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Measure how strongly each perturbation changes its target gene.

        For each target gene, or each guide with ``by="guide"``, the target
        gene's expression in the cells that carry the perturbation alone is
        compared with its expression in the control cells of the labels. A
        cell carries a target alone when its called guides target that gene
        and no other, with or without control guides, and a guide alone when
        it is the cell's one called guide. The comparison reports the mean
        UMIs of the gene in each group, as Cell Ranger's perturbation
        efficiency does, the mean expression after scaling each cell to the
        median library size of the compared cells, the fraction of the control
        expression that remains, its base-2 logarithm, and a two-sided
        Mann-Whitney test on the scaled expression with Benjamini-Hochberg
        adjustment. A knockdown leaves a fraction below 1 and an activation a
        fraction above 1. A row whose control cells hold no UMI of the gene
        has no fraction and the status ``"no_control_expression"``, but keeps
        its p-value, which counts in the adjustment.

        Each target is matched to a feature of ``from_assay`` by the feature
        ID that ``id_column`` of the guide assay gives its guides, which 10x
        imports fill from Cell Ranger's feature reference, when its guides
        give one ID and the feature IDs are unique; otherwise by the one
        feature whose name equals the target, when no other feature shares
        that name. Only the target genes are read, feature-major when that
        reads less. A target that matches no feature is kept with the status
        ``"no_feature"`` and named in a warning, which a reused table repeats.

        Args:
            labels: A ``perturbation_labels`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.labels`.
            from_assay: The expression assay.
            by: ``"target"`` for one row per target gene, or ``"guide"`` for
                one row per guide.
            id_column: The feature column of the guide assay that holds each
                guide's target feature ID, or None to match by name only.
            invalidate_cache: Measure again instead of reusing a saved table.

        Returns:
            A complete ``perturbation_efficiency`` artifact scoped to
            ``from_assay``. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_efficiency`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` is not a ``perturbation_labels``
                artifact, ``from_assay`` is not an assay, ``by`` is unknown,
                ``id_column`` is not a feature column of the guide assay, or
                no labelled cell carries control guides only.
            PermissionError: If the store is read-only and the table is not
                saved yet.
            MemoryError: If the target genes of the compared cells do not fit
                the memory budget.
        """

        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        if not isinstance(from_assay, str) or from_assay not in self.assay_names:
            raise ValueError(f"from_assay must name an assay, not {from_assay!r}")
        efficiency_settings(by, id_column)
        bool_argument(invalidate_cache, "invalidate_cache")
        rows = _label_rows(self, labels, by)
        status = rows.status
        row_names = rows.names
        expression = self._get_assay(from_assay)
        feature_ids = np.asarray(expression.feats.fetch_all("ids")).astype(str)
        feature_names = np.asarray(expression.feats.fetch_all("names")).astype(str)
        row_features = _row_features(
            self, labels, rows, id_column, feature_ids, feature_names
        )
        self._require_measured_cells(
            expression.name,
            status.input_ref("cell_selection"),
            operation=PerturbationEfficiencyArguments.operation,
        )
        row_feature_ids = _matched_ids(row_features, feature_ids)
        arguments = PerturbationEfficiencyArguments(
            labels=labels,
            dataset_fingerprint=self._ensure_dataset_fingerprint(expression.name),
            target_features_fingerprint=_pairs_fingerprint(row_names, row_feature_ids),
            by=by,
            invalidate_cache=invalidate_cache,
        )
        n_rows = len(row_names)
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=expression.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("names", shape=(n_rows,)),
                ArrayRequirement("status", shape=(n_rows,), dtype_kind="i"),
                *(
                    ArrayRequirement(name, shape=(n_rows,), dtype_kind="i")
                    for name in ("n_cells", "n_control_cells")
                ),
                *(
                    ArrayRequirement(name, shape=(n_rows,), dtype_kind="f")
                    for name in _EFFICIENCY_VALUES
                ),
            ),
            required_attributes=(
                AttributeRequirement(
                    "status_names",
                    predicate=lambda value: list(value) == list(EFFICIENCY_STATUSES),
                ),
                AttributeRequirement("by", predicate=lambda value: value == by),
            ),
        )
        if planned.reused:
            # A saved table warns as it did when it was measured.
            stored = artifact_group(self.zw, planned.ref)
            message = _unmatched_targets_warning(
                read_array(stored, "names").astype(str),
                read_array(stored, "status"),
                expression.name,
            )
            if message is not None:
                warn(message)
            return planned.ref
        groups, control = _label_groups(self, labels, rows, by)
        compared = np.flatnonzero((groups >= 0) | control)
        cell_index = cell_selection_indices(
            self.zw, status.input_ref("cell_selection")
        )[compared]
        needed_features = np.unique(row_features[row_features >= 0])
        resident = len(compared) * (16 * len(needed_features) + 40)
        _require_budget(
            resident,
            self.memoryBytes,
            f"Comparing {len(needed_features)} target genes over {len(compared)} "
            "cells holds about",
        )
        self._require_writable(arguments.operation)
        if len(needed_features):
            _, blocks = _count_blocks(
                self,
                expression,
                needed_features,
                cell_index,
                resident_bytes=resident,
                scratch_bytes_per_row=8 * len(needed_features),
            )
            counts = np.concatenate(
                [np.asarray(block, dtype=np.float64) for block in blocks]
            )
        else:
            counts = np.zeros((len(compared), 0))
        row_columns = np.where(
            row_features >= 0, np.searchsorted(needed_features, row_features), -1
        )
        arrays = efficiency_arrays(
            counts,
            expression._cell_count_totals(cell_index),
            groups[compared],
            control[compared],
            row_columns,
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "names": row_names,
                "targets": rows.target_names[rows.codes],
                "features": row_feature_ids,
                **{
                    name: getattr(arrays, name)
                    for name in (
                        "status",
                        "n_cells",
                        "n_control_cells",
                        *_EFFICIENCY_VALUES,
                    )
                },
            },
            attributes={"status_names": list(EFFICIENCY_STATUSES), "by": by},
        )
        message = _unmatched_targets_warning(row_names, arrays.status, expression.name)
        if message is not None:
            warn(message)
        logger.info(
            "Measured the efficiency of "
            f"{int(np.count_nonzero(arrays.status == EFFICIENCY_TESTED))} "
            f"of {n_rows} {by}s against {int(control.sum())} control cells"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_efficiency(
        self, efficiency: ArtifactRef
    ) -> PerturbationEfficiency:
        """Read a saved perturbation efficiency table.

        Args:
            efficiency: A ``perturbation_efficiency`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.efficiency`.

        Returns:
            One row per target or guide.

        Raises:
            TypeError: If ``efficiency`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_efficiency`` artifact.
        """
        _require_ref(
            efficiency, "efficiency", PerturbationEfficiencyArguments.artifact_kind
        )
        status = require_complete_artifact(self.zw, efficiency)
        group = artifact_group(self.zw, efficiency)

        by = str((status.parameters or {})["by"])
        features = read_array(group, "features").astype(str)
        columns: dict[str, Any] = {}
        if by == "guide":
            columns["guide"] = read_array(group, "names").astype(str)
        columns["target"] = read_array(group, "targets").astype(str)
        columns["feature"] = pd.Series(
            [None if feature == "" else feature for feature in features], dtype=object
        )
        columns["status"] = _status_labels(group, EFFICIENCY_STATUSES)
        for name in ("n_cells", "n_control_cells", *_EFFICIENCY_VALUES):
            columns[name] = read_array(group, name)
        return PerturbationEfficiency(
            table=pd.DataFrame(columns),
            by=by,
            from_assay=str(efficiency.assay),
        )

    @validation_scoped
    def _perturbation_edistance(
        self,
        labels: ArtifactRef,
        coordinates: ArtifactRef,
        *,
        by: str = "target",
        n_permutations: int = 1000,
        random_seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Measure how far each perturbation moves cells from the control cells.

        The E-distance of Peidli et al. (2024, scPerturb) compares the cells
        that carry a target alone, or a guide alone with ``by="guide"``, with
        the control cells of the labels in a coordinate space such as a PCA:
        twice the mean squared distance between the groups, minus the mean
        squared distance within each group over distinct pairs. Its expected
        value is twice the squared distance between the group means, so it
        grows as a perturbation moves the centre of its cells away from the
        controls, and a change in their spread alone leaves it near 0. A
        permutation test
        draws ``n_permutations`` groups of the perturbed group's size from the
        pooled cells, and the p-value is the share of draws, counting the
        observed groups, whose E-distance reaches the observed one, adjusted
        across rows by Benjamini-Hochberg. Each row draws from its own seed,
        derived from ``random_seed`` and the row.

        The coordinates can be any reduction, batch correction, imported
        coordinates or normalized values. Cells of the labels without
        coordinates are left out and counted in a log message. A row with
        fewer than two perturbed cells has no E-distance and the status
        ``"too_few_cells"``.

        Args:
            labels: A ``perturbation_labels`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.labels`.
            coordinates: The coordinate artifact.
            by: ``"target"`` for one row per target gene, or ``"guide"`` for
                one row per guide.
            n_permutations: The permutations of each test, from 0, which skips
                the tests, to 1,000,000.
            random_seed: The seed of the permutations, at least 0.
            invalidate_cache: Measure again instead of reusing a saved table.

        Returns:
            A complete ``perturbation_edistance`` artifact scoped to the assay
            of the coordinates. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_edistance`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` is not a ``perturbation_labels``
                artifact, ``coordinates`` is not a coordinate artifact, ``by``
                is unknown, a number is out of range, or fewer than two control
                cells have coordinates.
            PermissionError: If the store is read-only and the table is not
                saved yet.
            MemoryError: If the coordinates of the compared cells do not fit
                the memory budget.
        """
        from ...graph.feature_projection import resolve_coordinate_inputs

        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        _require_ref(coordinates, "coordinates", *_COORDINATE_KINDS)
        permutations, seed = edistance_settings(by, n_permutations, random_seed)
        bool_argument(invalidate_cache, "invalidate_cache")
        rows = _label_rows(self, labels, by)
        inputs = resolve_coordinate_inputs(self.zw, coordinates)
        arguments = PerturbationEDistanceArguments(
            labels=labels,
            coordinates=coordinates,
            by=by,
            n_permutations=permutations,
            random_seed=seed,
            invalidate_cache=invalidate_cache,
        )
        n_rows = len(rows.names)
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=str(coordinates.assay),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("status", shape=(n_rows,), dtype_kind="i"),
                *(
                    ArrayRequirement(name, shape=(n_rows,), dtype_kind="i")
                    for name in ("n_cells", "n_control_cells")
                ),
                *(
                    ArrayRequirement(name, shape=(n_rows,), dtype_kind="f")
                    for name in ("edistance", "p_value", "adjusted_p_value")
                ),
            ),
            required_attributes=(
                AttributeRequirement(
                    "status_names",
                    predicate=lambda value: list(value) == list(EDISTANCE_STATUSES),
                ),
                AttributeRequirement("by", predicate=lambda value: value == by),
            ),
        )
        if planned.reused:
            return planned.ref
        groups, control = _label_groups(self, labels, rows, by)

        labelled = cell_selection_indices(
            self.zw, rows.status.input_ref("cell_selection")
        )
        located = cell_selection_indices(self.zw, inputs.cell_selection)
        compared = np.flatnonzero((groups >= 0) | control)
        positions, present = _positions_in(located, labelled[compared])
        if not present.all():
            logger.info(
                f"{int(np.count_nonzero(~present))} of {len(compared)} compared cells "
                "have no coordinates and are left out"
            )
        kept = compared[present]
        # The measurement needs only the kept cells' coordinate rows, groups
        # and flags, so the arrays of every labelled cell are released here.
        coordinate_rows = positions[present]
        kept_groups = groups[kept]
        kept_control = control[kept]
        del groups, control, labelled, located, compared, positions, present, kept
        n_kept = len(kept_groups)
        n_control = int(np.count_nonzero(kept_control))
        if n_control < 2:
            raise ValueError(
                f"E-distances need at least two control cells with coordinates, "
                f"and {n_control} of the labels' control cells have them"
            )
        data = as_zarr_array(artifact_group(self.zw, coordinates)["data"], name="data")
        n_dims = int(data.shape[1])
        # The coordinates are read as float64 one chunk of rows at a time,
        # beside each kept cell's group, flag and row, and the measurement
        # then holds what edistance_memory_bytes counts.
        read_fixed, read_per_row = array_row_selection_parts(data)
        reading = (
            n_kept * (8 * n_dims + 17)
            + read_fixed
            + read_per_row * min(n_kept, int(data.chunks[0]))
        )
        resident = max(
            edistance_memory_bytes(
                n_kept, n_dims, n_control, n_rows=n_rows, nthreads=self.nthreads
            ),
            reading,
        )
        _require_budget(
            resident,
            self.memoryBytes,
            f"The E-distances of {n_kept} cells in {n_dims} dimensions hold about",
        )
        self._require_writable(arguments.operation)
        values = _float64_rows(data, coordinate_rows)
        del coordinate_rows
        arrays = edistance_arrays(
            values,
            kept_groups,
            kept_control,
            n_rows,
            n_permutations=permutations,
            random_seed=seed,
            nthreads=self.nthreads,
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "names": rows.names,
                "targets": rows.target_names[rows.codes],
                **{
                    name: getattr(arrays, name)
                    for name in (
                        "status",
                        "n_cells",
                        "n_control_cells",
                        "edistance",
                        "p_value",
                        "adjusted_p_value",
                    )
                },
            },
            attributes={"status_names": list(EDISTANCE_STATUSES), "by": by},
        )
        logger.info(
            "Measured the E-distances of "
            f"{int(np.count_nonzero(arrays.status == EDISTANCE_TESTED))} "
            f"of {n_rows} {by}s with {permutations} permutations each"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_edistance(
        self, edistance: ArtifactRef
    ) -> PerturbationEDistance:
        """Read a saved E-distance table.

        Args:
            edistance: A ``perturbation_edistance`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.edistance`.

        Returns:
            One row per target or guide, with the permutations and seed.

        Raises:
            TypeError: If ``edistance`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_edistance`` artifact.
        """
        _require_ref(
            edistance, "edistance", PerturbationEDistanceArguments.artifact_kind
        )
        status = require_complete_artifact(self.zw, edistance)
        group = artifact_group(self.zw, edistance)

        parameters = status.parameters or {}
        by = str(parameters["by"])
        columns: dict[str, Any] = {}
        if by == "guide":
            columns["guide"] = read_array(group, "names").astype(str)
        columns["target"] = read_array(group, "targets").astype(str)
        columns["status"] = _status_labels(group, EDISTANCE_STATUSES)
        for name in (
            "n_cells",
            "n_control_cells",
            "edistance",
            "p_value",
            "adjusted_p_value",
        ):
            columns[name] = read_array(group, name)
        return PerturbationEDistance(
            table=pd.DataFrame(columns),
            by=by,
            n_permutations=int(parameters["n_permutations"]),
            random_seed=int(parameters["random_seed"]),
        )
